// bam_lz4_fused_q6.cu — Fused BaM I/O + nvCOMPdx LZ4 decompress + Q6 scan
// 4 warps/block, each warp independently handles IO+decomp for 1 field.
// __launch_bounds__(128, 8) → 8 blocks/SM for maximum decomp parallelism.
// CUB BlockReduce removed → per-thread atomicAdd to minimize register pressure.
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q6.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "tpch/page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>

// From common/pruning.cuh — duplicated here to avoid pulling in __global__
// kernels from that header (which causes multiple-definition link errors).
static constexpr uint32_t kZonemapMaxPagesPerBlock = 512;

__device__ inline void zonemap_compact_block_pages(
    const uint8_t* __restrict__ d_mask,
    uint32_t total_pages,
    uint32_t* s_active,
    uint32_t* s_count)
{
    const uint32_t tid    = threadIdx.x;
    const uint32_t stride = gridDim.x;
    const uint32_t first  = blockIdx.x;
    const uint32_t max_pg = (total_pages > first)
        ? (total_pages - first + stride - 1) / stride : 0;

    if (!d_mask) {
        for (uint32_t i = tid; i < max_pg; i += blockDim.x)
            s_active[i] = first + i * stride;
        if (tid == 0) *s_count = max_pg;
        __syncthreads();
        return;
    }

    uint32_t pg = first + tid * stride;
    bool is_active = (tid < max_pg && pg < total_pages) && d_mask[pg];

    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;
    uint32_t ballot      = __ballot_sync(0xffffffff, is_active);
    uint32_t lane_prefix = __popc(ballot & ((1u << lane) - 1));
    uint32_t warp_cnt    = __popc(ballot);

    __shared__ uint32_t s_wpfx[32];
    if (lane == 0) s_wpfx[warp_id] = warp_cnt;
    __syncthreads();

    if (tid == 0) {
        uint32_t sum = 0;
        for (uint32_t w = 0; w < 32; w++) {
            uint32_t c = s_wpfx[w];
            s_wpfx[w] = sum;
            sum += c;
        }
        *s_count = sum;
    }
    __syncthreads();

    if (is_active)
        s_active[s_wpfx[warp_id] + lane_prefix] = pg;
    __syncthreads();
}

#define FUSED_Q6_CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// ── BaM nblk alignment fix ──
// BaM controller has issues with 2-page (9-16 block) reads; skip to 3 pages.
__device__ static uint32_t fused_q6_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// ── I/O parameter computation helper (warp-spec kernel) ──
__device__ static void fused_q6_io_params(
    const Q6WarpSpecParams& p,
    uint32_t field, uint32_t orig_pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.field_start_page_ids[field] + orig_pg;
    dev = global_pg % ndev;
    if (p.is_compressed[field]) {
        lba = p.partition_start_lbas[dev] +
              p.d_comp_offsets[field][orig_pg] / 512;
        comp_sz = p.d_comp_sizes[field][orig_pg];
        nblk = fused_q6_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] +
              local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

// ════════════════════════════════════════════════════════════════
// Fused Q6 kernel: 4 warps/block, block-stride persistent loop
// __launch_bounds__(128, 8) → 8 blocks/SM, 32 warps/SM
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void bam_lz4_fused_q6_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ6Params p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS = 4;
    constexpr uint32_t THREADS = WARPS * 32;  // 128

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;

    // Each warp owns a dedicated page_cache slot
    const uint32_t slot = blockIdx.x * WARPS + warp_id;

    // Shared memory: nvCOMPdx per-warp region
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Page data layout: pag_head (12 bytes = 3 int32) + data
    constexpr uint32_t HDR_INT32 = 3;  // sizeof(pag_head) / sizeof(int32_t)

    // Block-stride persistent loop over pages
    for (uint64_t pg = blockIdx.x; pg < p.npages; pg += gridDim.x) {

        // Zone map skip
        if (p.d_page_active && !p.d_page_active[pg]) {
            __syncthreads();
            continue;
        }

        // ── Phase 1+2: Each warp reads + decompresses its field ──
        // warp 0→L_SHIPDATE, 1→L_QUANTITY, 2→L_EXTENDEDPRICE, 3→L_DISCOUNT
        char* my_decomp = d_decomp_buf +
            ((uint64_t)blockIdx.x * WARPS + warp_id) * p.page_size;

        {
            uint64_t global_pg = p.field_start_page_ids[warp_id] + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            uint32_t comp_sz;

            if (p.is_compressed[warp_id]) {
                lba = p.partition_start_lbas[dev] +
                      p.d_comp_offsets[warp_id][pg] / 512;
                comp_sz = p.d_comp_sizes[warp_id][pg];
                // roundup4096(comp_sz) / 512
                nblk = fused_q6_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] +
                      local_pg * (p.page_size / 512);
                nblk = p.page_size / 512;
                comp_sz = p.page_size;  // will trigger direct copy in helper
            }

            bam_lz4_io_decomp_warp<PAGE_SIZE_CONST>(
                ctrls, pc, (void*)pc_base_addr,
                slot, my_decomp,
                lba, nblk, dev, comp_sz, p.page_size, my_smem);
        }
        __syncthreads();  // Wait for all 4 fields to complete

        // ── Phase 3: Q6 predicate + revenue (all 128 threads) ──
        uint64_t base = (uint64_t)blockIdx.x * WARPS * p.page_size;
        const int32_t* sd = (const int32_t*)(d_decomp_buf + base + 0 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* qt = (const int32_t*)(d_decomp_buf + base + 1 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* ep = (const int32_t*)(d_decomp_buf + base + 2 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* dc = (const int32_t*)(d_decomp_buf + base + 3 * (uint64_t)p.page_size) + HDR_INT32;

        // nalloc from L_SHIPDATE page header (canonical)
        uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base + 0 * (uint64_t)p.page_size);

        int64_t my_value = 0;
        for (uint32_t r = tid; r < nalloc; r += THREADS) {
            int32_t s = sd[r], q = qt[r], d = dc[r], e = ep[r];
            // Q6 predicates: shipdate in [sd_low, sd_high), discount in [5,7], quantity < 2400
            if (s >= p.sd_low && s < p.sd_high &&
                d >= 5 && d <= 7 && q < 2400) {
                my_value += (int64_t)e * d;
            }
        }

        // Per-thread atomicAdd (no CUB → fewer registers → higher occupancy)
        if (my_value != 0) {
            atomicAdd(reinterpret_cast<unsigned long long int*>(p.d_revenue),
                      static_cast<unsigned long long int>(my_value));
        }

        __syncthreads();  // Before next page iteration
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

struct BAMFusedQ6Context {
    bam_io_page_cache_t io_pc;      // opaque page_cache handle
    void*       d_ctrls;            // Controller** (device)
    void*       d_pc_ptr;           // page_cache_d_t* (device)
    const char* pc_base_addr;       // page_cache base address (device)
    char*       d_decomp_buf;       // [num_slots * page_size]
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_q6_ctx_t bam_fused_q6_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ6Context();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 4 slots per block (1 per warp)
    const uint32_t num_slots = num_blocks * 4;

    // Create page cache via C++11 wrapper
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    // Extract opaque pointers for kernel
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: 1 page per warp slot (4 fields per block)
    size_t decomp_size = (size_t)num_slots * page_size;
    FUSED_Q6_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q6_ctx_t>(ctx);
}

static void bam_fused_q6_launch(
    BAMFusedQ6Context* ctx,
    const BAMFusedQ6Params& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * 4;
        auto kernel_fn = bam_lz4_fused_q6_kernel<PS>;
        FUSED_Q6_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_Q6_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q6_run(
    bam_fused_q6_ctx_t ctx_handle,
    const BAMFusedQ6Params& params)
{
    auto* ctx = static_cast<BAMFusedQ6Context*>(ctx_handle);
    bam_fused_q6_launch(ctx, params, 0);
    FUSED_Q6_CUDA_CHECK(cudaDeviceSynchronize());
}

void bam_fused_q6_run_async(
    bam_fused_q6_ctx_t ctx_handle,
    const BAMFusedQ6Params& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ6Context*>(ctx_handle);
    bam_fused_q6_launch(ctx, params, stream);
}

void bam_fused_q6_destroy(bam_fused_q6_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ6Context*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ════════════════════════════════════════════════════════════════
// Decomp+Scan only kernel: no BaM IO code
//
// Reads compressed pages from GPU staging buffer, decompresses with
// nvCOMPdx warp-level LZ4, then runs Q6 predicate scan.
// No BaM IO functions → fewer registers → better occupancy.
// 4 warps/block (one per field), block-stride loop within batch.
// ════════════════════════════════════════════════════════════════
static constexpr int DS_Q6_WARPS = 4;

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void q6_decomp_scan_kernel(
    char*               d_decomp_buf,
    Q6DecompScanParams  p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS   = DS_Q6_WARPS;
    constexpr uint32_t THREADS = WARPS * 32;
    constexpr uint32_t HDR_INT32 = 3;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    // Decomp buffer: per-block, 4 pages (one per warp/field)
    const uint32_t slot = blockIdx.x * WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    // Block-stride loop over batch pages
    for (uint32_t local_pg = blockIdx.x; local_pg < p.batch_count; local_pg += gridDim.x) {

        // Original page index (for comp_sizes lookup)
        uint32_t orig_pg = p.d_batch_page_ids[local_pg];

        // ── nvCOMPdx LZ4 decompress from staging_io ──
        // Staging layout: field f pages at staging_io + f * batch_count * page_size
        const char* src = p.staging_io +
            ((uint64_t)warp_id * p.batch_count + local_pg) * p.page_size;

        if (p.is_compressed[warp_id]) {
            uint32_t comp_sz = p.d_comp_sizes[warp_id][orig_pg];
            if (comp_sz < p.page_size) {
                auto decompressor = lz4_decomp_t();
                size_t dsz = 0;
                decompressor.execute(src, my_decomp, (size_t)comp_sz, &dsz, my_smem, nullptr);
            } else {
                const uint32_t n4 = p.page_size / 4;
                for (uint32_t i = lane; i < n4; i += 32)
                    reinterpret_cast<uint32_t*>(my_decomp)[i] =
                        reinterpret_cast<const uint32_t*>(src)[i];
            }
        } else {
            const uint32_t n4 = p.page_size / 4;
            for (uint32_t i = lane; i < n4; i += 32)
                reinterpret_cast<uint32_t*>(my_decomp)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
        }

        __syncthreads();  // Wait for all 4 fields

        // ── Q6 predicate + revenue (all 128 threads) ──
        uint64_t base = (uint64_t)blockIdx.x * WARPS * p.page_size;
        const int32_t* sd = (const int32_t*)(d_decomp_buf + base + 0 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* qt = (const int32_t*)(d_decomp_buf + base + 1 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* ep = (const int32_t*)(d_decomp_buf + base + 2 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* dc = (const int32_t*)(d_decomp_buf + base + 3 * (uint64_t)p.page_size) + HDR_INT32;

        uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base + 0 * (uint64_t)p.page_size);

        int64_t my_value = 0;
        for (uint32_t r = tid; r < nalloc; r += THREADS) {
            int32_t s = sd[r], q = qt[r], d = dc[r], e = ep[r];
            if (s >= p.sd_low && s < p.sd_high &&
                d >= 5 && d <= 7 && q < 2400) {
                my_value += (int64_t)e * d;
            }
        }

        if (my_value != 0) {
            atomicAdd(reinterpret_cast<unsigned long long int*>(p.d_revenue),
                      static_cast<unsigned long long int>(my_value));
        }

        __syncthreads();  // Before next page
    }
}

// ════════════════════════════════════════════════════════════════
// Decomp+Scan context and host API
// ════════════════════════════════════════════════════════════════
struct Q6DecompScanContext {
    char*    d_decomp_buf;
    uint32_t page_size;
    uint32_t num_blocks;
};

q6_decomp_scan_ctx_t q6_decomp_scan_create(
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new Q6DecompScanContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * DS_Q6_WARPS;
    size_t decomp_size = (size_t)num_slots * page_size;
    FUSED_Q6_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<q6_decomp_scan_ctx_t>(ctx);
}

void q6_decomp_scan_async(
    q6_decomp_scan_ctx_t ctx_handle,
    const Q6DecompScanParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<Q6DecompScanContext*>(ctx_handle);
    constexpr uint32_t THREADS = DS_Q6_WARPS * 32;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>()
                         * DS_Q6_WARPS;
        auto kernel_fn = q6_decomp_scan_kernel<PS>;
        FUSED_Q6_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_decomp_buf, p);
    });

    FUSED_Q6_CUDA_CHECK(cudaGetLastError());
}

void q6_decomp_scan_destroy(q6_decomp_scan_ctx_t ctx_handle)
{
    auto* ctx = static_cast<Q6DecompScanContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    delete ctx;
}

// ════════════════════════════════════════════════════════════════
// Producer-Consumer Q6 kernel (single kernel, role-based blocks)
//
// blocks [0, io_blocks) → IO Producer: BaM page_cache reads
// blocks [io_blocks, io_blocks+consumer_blocks) → Consumer: nvCOMPdx decomp + Q6 scan
//
// Single kernel guarantees all blocks are co-scheduled, avoiding
// the deadlock that occurs with two separate persistent kernels.
//
// Ring protocol (page_cache slots as ring buffer):
//   ring_page[entry] == -1  → slot empty, IO producer can fill
//   ring_page[entry] == idx → slot filled with page idx, consumer can read
// ════════════════════════════════════════════════════════════════

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void q6_prodcons_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    Q6ProdConsParams p,
    uint32_t    io_blocks)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS   = 4;
    constexpr uint32_t THREADS = WARPS * 32;
    constexpr uint32_t HDR_INT32 = 3;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    volatile int32_t* rp = p.ring_page;
    __shared__ uint32_t s_page_idx;

    const bool is_io = (blockIdx.x < io_blocks);

    if (tid == 0 && blockIdx.x == 0)
        printf("[PRODCONS] kernel started: is_io=%d total_pages=%u io_blocks=%u\n",
               (int)is_io, p.total_pages, io_blocks);

    if (is_io) {
        // ════════════════════════════════════════
        // IO Producer path
        // ════════════════════════════════════════
        const uint32_t ndev = p.n_devices > 1 ? p.n_devices : 1;

        while (true) {
            if (tid == 0)
                s_page_idx = atomicAdd(p.d_io_counter, 1);
            __syncthreads();
            uint32_t page_idx = s_page_idx;
            if (page_idx >= p.total_pages) break;

            uint32_t ring_entry = page_idx % p.n_ring;
            uint32_t orig_pg    = p.d_active_page_ids[page_idx];

            // Wait for ring slot empty
            if (tid == 0) {
                unsigned ns = 8;
                uint32_t spin_cnt = 0;
                while (rp[ring_entry] != -1) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                    __nanosleep(ns);
                    if (ns < 256) ns *= 2;
#endif
                    if (++spin_cnt % 5000000 == 0)
                        printf("[IO-STUCK] blk=%u pg=%u re=%u cur=%d\n",
                               blockIdx.x, page_idx, ring_entry, (int)rp[ring_entry]);
                }
            }
            __syncthreads();

            uint32_t slot = ring_entry * WARPS + warp_id;

            uint64_t global_pg = p.field_start_page_ids[warp_id] + orig_pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;

            if (p.is_compressed[warp_id]) {
                lba  = p.partition_start_lbas[dev] +
                       p.d_comp_offsets[warp_id][orig_pg] / 512;
                uint32_t comp_sz = p.d_comp_sizes[warp_id][orig_pg];
                nblk = fused_q6_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba  = p.partition_start_lbas[dev] +
                       local_pg * (PAGE_SIZE_CONST / 512);
                nblk = PAGE_SIZE_CONST / 512;
            }

            if (lane == 0)
                bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
            __syncwarp();
            __syncthreads();

            __threadfence();
            if (tid == 0)
                rp[ring_entry] = (int32_t)page_idx;
        }
    } else {
        // ════════════════════════════════════════
        // Consumer path (decomp + scan)
        // ════════════════════════════════════════
        extern __shared__ __align__(8) uint8_t smem[];
        constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
        uint8_t* my_smem = smem + warp_id * warp_smem;

        // Consumer block index (0-based within consumer blocks)
        const uint32_t cons_bid = blockIdx.x - io_blocks;
        char* my_decomp = d_decomp_buf +
            ((uint64_t)cons_bid * WARPS + warp_id) * p.page_size;

        while (true) {
            if (tid == 0)
                s_page_idx = atomicAdd(p.d_scan_counter, 1);
            __syncthreads();
            uint32_t page_idx = s_page_idx;
            if (page_idx >= p.total_pages) break;

            uint32_t ring_entry = page_idx % p.n_ring;
            uint32_t orig_pg    = p.d_active_page_ids[page_idx];

            // Wait for ring slot to contain our page
            if (tid == 0) {
                unsigned ns = 8;
                uint32_t spin_cnt = 0;
                while (rp[ring_entry] != (int32_t)page_idx) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                    __nanosleep(ns);
                    if (ns < 256) ns *= 2;
#endif
                    if (++spin_cnt % 5000000 == 0)
                        printf("[CONS-STUCK] bid=%u pg=%u re=%u cur=%d want=%d\n",
                               cons_bid, page_idx, ring_entry, (int)rp[ring_entry], (int)page_idx);
                }
            }
            __syncthreads();

            // Decompress from page_cache to decomp_buf
            uint32_t pc_slot = ring_entry * WARPS + warp_id;
            const char* src = pc_base_addr + (uint64_t)pc_slot * p.page_size;

            // DEBUG: skip decomp+scan, just release ring slot immediately
            __syncthreads();
            __threadfence();
            if (tid == 0)
                rp[ring_entry] = -1;
            __syncthreads();
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Query max co-resident blocks for producer-consumer kernel
// ════════════════════════════════════════════════════════════════

uint32_t q6_prodcons_max_blocks(uint32_t page_size)
{
    int max_blocks_per_sm = 0;
    constexpr uint32_t THREADS = 128;

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * 4;
        auto kfn = q6_prodcons_kernel<PS>;
        cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kfn, THREADS, smem_size);
    });

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    uint32_t total = (uint32_t)max_blocks_per_sm * (uint32_t)sm_count;
    fprintf(stderr, "[q6_prodcons] max_blocks_per_sm=%d sm_count=%d max_total=%u\n",
            max_blocks_per_sm, sm_count, total);
    return total;
}

// ════════════════════════════════════════════════════════════════
// Producer-Consumer launch function (single kernel)
// ════════════════════════════════════════════════════════════════

void q6_prodcons_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const Q6ProdConsParams& p,
    uint32_t io_blocks,
    uint32_t consumer_blocks,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 128;
    uint32_t total_blocks = io_blocks + consumer_blocks;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * 4;
        auto kfn = q6_prodcons_kernel<PS>;
        FUSED_Q6_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kfn<<<total_blocks, THREADS, smem_size, stream>>>(
            d_ctrls, d_pc_ptr, pc_base_addr, d_decomp_buf, p, io_blocks);
    });
    FUSED_Q6_CUDA_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized Q6 kernel: 32 warps (1024 threads) / block
//   Warps 0-3:   IO (BaM page reads, 1 field/warp)
//   Warps 4-31:  Decomp (7 groups × 4 warps, nvCOMPdx LZ4)
//   All 1024 threads: Q6 scan
//
// Batch processing: BATCH=7 pages per batch.
// IO warps read BATCH pages of their field sequentially.
// 7 decomp groups decompress BATCH pages in parallel (~7× throughput).
//
// Double-buffer pipeline: IO[batch+1] overlaps with Decomp[batch].
//   Prolog  → IO batch 0 into buf[0]
//   Loop    → IO[b] buf[cur] | Decomp[b-1] buf[prev]  →  Scan[b-1]
//   Epilog  → Decomp last batch  →  Scan last batch
//
// Page cache: blockIdx.x * 56 + buf * 28 + j * 4 + field  (56 slots/block)
// Decomp buf: same slot indexing × page_size
// ════════════════════════════════════════════════════════════════

static constexpr uint32_t Q6WS_BATCH      = 7;
static constexpr uint32_t Q6WS_N_BUF      = 2;
static constexpr uint32_t Q6WS_NUM_FIELDS = 4;
static constexpr uint32_t Q6WS_IO_WARPS   = 4;
static constexpr uint32_t Q6WS_DECOMP_GROUPS = 7;
static constexpr uint32_t Q6WS_SLOTS_PER_BUF   = Q6WS_BATCH * Q6WS_NUM_FIELDS;   // 28
static constexpr uint32_t Q6WS_SLOTS_PER_BLOCK = Q6WS_N_BUF * Q6WS_SLOTS_PER_BUF; // 56

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(1024, 1)
void q6_warp_spec_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    Q6WarpSpecParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t BATCH      = Q6WS_BATCH;
    constexpr uint32_t N_BUF      = Q6WS_N_BUF;
    constexpr uint32_t NUM_FIELDS = Q6WS_NUM_FIELDS;
    constexpr uint32_t IO_WARPS   = Q6WS_IO_WARPS;
    constexpr uint32_t SLOTS_PER_BUF   = Q6WS_SLOTS_PER_BUF;
    constexpr uint32_t SLOTS_PER_BLOCK = Q6WS_SLOTS_PER_BLOCK;
    constexpr uint32_t THREADS    = 1024;
    constexpr uint32_t HDR_INT32  = 3;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;   // 0..31
    const uint32_t lane    = tid % 32;

    // ── Shared memory ──
    // Static: IO→decomp metadata communication
    __shared__ uint32_t s_comp_sz[N_BUF][BATCH][NUM_FIELDS];
    __shared__ uint32_t s_batch_count[N_BUF];

    // Active page list (compacted from d_page_mask before IO loop)
    __shared__ uint32_t s_active_pgs[kZonemapMaxPagesPerBlock];
    __shared__ uint32_t s_num_active;

    // Dynamic: nvCOMPdx per-decomp-warp region (28 warps)
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = (warp_id >= IO_WARPS)
        ? smem + (warp_id - IO_WARPS) * warp_smem : nullptr;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;
    if (blockIdx.x >= p.total_pages) return;

    // Active page list: global compact path or per-block compact fallback
    if (p.d_active_page_ids) {
        const uint32_t stride = gridDim.x;
        const uint32_t first  = blockIdx.x;
        const uint32_t max_pg = (p.total_pages > first)
            ? (p.total_pages - first + stride - 1) / stride : 0;
        for (uint32_t i = tid; i < max_pg; i += blockDim.x)
            s_active_pgs[i] = p.d_active_page_ids[first + i * stride];
        if (tid == 0) s_num_active = max_pg;
        __syncthreads();
    } else {
        zonemap_compact_block_pages(p.d_page_mask, p.total_pages,
                                    s_active_pgs, &s_num_active);
    }
    const uint32_t my_pages = s_num_active;
    if (my_pages == 0) return;
    const uint32_t block_slot_base = blockIdx.x * SLOTS_PER_BLOCK;

    // ── Prolog: IO warps read first batch into buf[0] ──
    {
        const uint32_t b_count = (BATCH < my_pages) ? BATCH : my_pages;
        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < b_count; j++) {
                uint32_t orig_pg = s_active_pgs[j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                fused_q6_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = block_slot_base + j * NUM_FIELDS + field;
                if (lane == 0) {
                    bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
                    s_comp_sz[0][j][field] = comp_sz;
                }
                __syncwarp();
            }
        }
        if (tid == 0)
            s_batch_count[0] = (BATCH < my_pages) ? BATCH : my_pages;
    }
    __syncthreads();

    // ── Main loop: IO[batch b] || Decomp[batch b-1] → Scan[batch b-1] ──
    uint32_t prev_buf = 0;

    for (uint32_t bstart = BATCH; bstart < my_pages; bstart += BATCH) {
        const uint32_t cur_buf    = 1 - prev_buf;
        const uint32_t rem        = my_pages - bstart;
        const uint32_t cur_count  = (BATCH < rem) ? BATCH : rem;
        const uint32_t prev_count = s_batch_count[prev_buf];

        // Phase A: IO reads next batch || Decomp groups process previous batch
        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < cur_count; j++) {
                uint32_t orig_pg = s_active_pgs[bstart + j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                fused_q6_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = block_slot_base + cur_buf * SLOTS_PER_BUF
                              + j * NUM_FIELDS + field;
                if (lane == 0) {
                    bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
                    s_comp_sz[cur_buf][j][field] = comp_sz;
                }
                __syncwarp();
            }
            if (warp_id == 0 && lane == 0)
                s_batch_count[cur_buf] = cur_count;
        } else {
            const uint32_t dw    = warp_id - IO_WARPS;     // 0..27
            const uint32_t group = dw / NUM_FIELDS;         // 0..6
            const uint32_t field = dw % NUM_FIELDS;         // 0..3

            if (group < prev_count) {
                uint32_t slot = block_slot_base + prev_buf * SLOTS_PER_BUF
                              + group * NUM_FIELDS + field;
                uint32_t comp_sz = s_comp_sz[prev_buf][group][field];
                char* dst = d_decomp_buf + (uint64_t)slot * p.page_size;

                bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                    pc_base_addr, slot, dst, comp_sz, p.page_size, my_smem);
            }
        }
        __syncthreads();

        // Phase B: Scan previous batch (all 1024 threads)
        {
            int64_t batch_val = 0;
            for (uint32_t j = 0; j < prev_count; j++) {
                uint64_t base = (uint64_t)(block_slot_base
                    + prev_buf * SLOTS_PER_BUF + j * NUM_FIELDS) * p.page_size;
                const int32_t* sd = (const int32_t*)(d_decomp_buf + base
                    + 0 * (uint64_t)p.page_size) + HDR_INT32;
                const int32_t* qt = (const int32_t*)(d_decomp_buf + base
                    + 1 * (uint64_t)p.page_size) + HDR_INT32;
                const int32_t* ep = (const int32_t*)(d_decomp_buf + base
                    + 2 * (uint64_t)p.page_size) + HDR_INT32;
                const int32_t* dc = (const int32_t*)(d_decomp_buf + base
                    + 3 * (uint64_t)p.page_size) + HDR_INT32;
                uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base);

                for (uint32_t r = tid; r < nalloc; r += THREADS) {
                    int32_t s = sd[r], q = qt[r], d = dc[r], e = ep[r];
                    if (s >= p.sd_low && s < p.sd_high &&
                        d >= 5 && d <= 7 && q < 2400) {
                        batch_val += (int64_t)e * d;
                    }
                }
            }
            if (batch_val != 0)
                atomicAdd(reinterpret_cast<unsigned long long int*>(p.d_revenue),
                          static_cast<unsigned long long int>(batch_val));
        }
        __syncthreads();

        prev_buf = cur_buf;
    }

    // ── Epilog: Decomp + Scan last batch ──
    {
        const uint32_t last_count = s_batch_count[prev_buf];

        // Decomp
        if (warp_id >= IO_WARPS) {
            const uint32_t dw    = warp_id - IO_WARPS;
            const uint32_t group = dw / NUM_FIELDS;
            const uint32_t field = dw % NUM_FIELDS;

            if (group < last_count) {
                uint32_t slot = block_slot_base + prev_buf * SLOTS_PER_BUF
                              + group * NUM_FIELDS + field;
                uint32_t comp_sz = s_comp_sz[prev_buf][group][field];
                char* dst = d_decomp_buf + (uint64_t)slot * p.page_size;

                bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                    pc_base_addr, slot, dst, comp_sz, p.page_size, my_smem);
            }
        }
        __syncthreads();

        // Scan
        {
            int64_t batch_val = 0;
            for (uint32_t j = 0; j < last_count; j++) {
                uint64_t base = (uint64_t)(block_slot_base
                    + prev_buf * SLOTS_PER_BUF + j * NUM_FIELDS) * p.page_size;
                const int32_t* sd = (const int32_t*)(d_decomp_buf + base
                    + 0 * (uint64_t)p.page_size) + HDR_INT32;
                const int32_t* qt = (const int32_t*)(d_decomp_buf + base
                    + 1 * (uint64_t)p.page_size) + HDR_INT32;
                const int32_t* ep = (const int32_t*)(d_decomp_buf + base
                    + 2 * (uint64_t)p.page_size) + HDR_INT32;
                const int32_t* dc = (const int32_t*)(d_decomp_buf + base
                    + 3 * (uint64_t)p.page_size) + HDR_INT32;
                uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base);

                for (uint32_t r = tid; r < nalloc; r += THREADS) {
                    int32_t s = sd[r], q = qt[r], d = dc[r], e = ep[r];
                    if (s >= p.sd_low && s < p.sd_high &&
                        d >= 5 && d <= 7 && q < 2400) {
                        batch_val += (int64_t)e * d;
                    }
                }
            }
            if (batch_val != 0)
                atomicAdd(reinterpret_cast<unsigned long long int*>(p.d_revenue),
                          static_cast<unsigned long long int>(batch_val));
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized: max co-resident blocks query
// ════════════════════════════════════════════════════════════════

uint32_t q6_warp_spec_max_blocks(uint32_t page_size)
{
    int max_blocks_per_sm = 0;
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = Q6WS_DECOMP_GROUPS * Q6WS_NUM_FIELDS;  // 28

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = q6_warp_spec_kernel<PS>;
        cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kfn, THREADS, smem_size);
    });

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    uint32_t total = (uint32_t)max_blocks_per_sm * (uint32_t)sm_count;
    fprintf(stderr, "[q6_warp_spec] max_blocks_per_sm=%d sm_count=%d max_total=%u\n",
            max_blocks_per_sm, sm_count, total);
    return total;
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized: launch function
// ════════════════════════════════════════════════════════════════

void q6_warp_spec_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const Q6WarpSpecParams& params,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = Q6WS_DECOMP_GROUPS * Q6WS_NUM_FIELDS;  // 28

    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = q6_warp_spec_kernel<PS>;
        FUSED_Q6_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kfn<<<num_blocks, THREADS, smem_size, stream>>>(
            d_ctrls, d_pc_ptr, pc_base_addr, d_decomp_buf, params);
    });
    FUSED_Q6_CUDA_CHECK(cudaGetLastError());
}
