// bam_lz4_fused_revenue.cu — Fused BaM I/O + nvCOMPdx LZ4 decompress + Revenue scan
// Compiled as CUDA C++17 with separable compilation + device linking.
//
// Based on bam_lz4_fused_q6.cu with different predicates:
//   Q6:      sd ∈ [low,high) && d ∈ [5,7] && q < 2400
//   Revenue: sd ∈ [low,high) && (qt_max==0 || q < qt_max)

#include "bam_lz4_fused_revenue.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "tpch/page_size_dispatch.h"

#include <cub/cub.cuh>
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

#define FUSED_REV_CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// ── BaM nblk alignment fix ──
// BaM controller has issues with 2-page (9-16 block) reads; skip to 3 pages.
__device__ static uint32_t fused_rev_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// ── I/O parameter computation helper (warp-spec kernel) ──
__device__ static void fused_rev_io_params(
    const RevenueWarpSpecParams& p,
    uint32_t field, uint32_t orig_pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.field_start_page_ids[field] + orig_pg;
    dev = global_pg % ndev;
    if (p.is_compressed[field]) {
        lba = p.partition_start_lbas[dev] +
              p.d_comp_offsets[field][orig_pg] / 512;
        comp_sz = p.d_comp_sizes[field][orig_pg];
        nblk = fused_rev_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] +
              local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

// ════════════════════════════════════════════════════════════════
// Fused Revenue kernel: 4 warps/block, block-stride persistent loop
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ void bam_lz4_fused_revenue_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedRevenueParams p)
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

    // cub::BlockReduce for revenue aggregation
    using BlockReduceInt = cub::BlockReduce<int64_t, THREADS>;
    __shared__ typename BlockReduceInt::TempStorage reduce_temp;

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
                nblk = fused_rev_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
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

        // ── Phase 3: Revenue predicate + aggregation (all 128 threads) ──
        uint64_t base = (uint64_t)blockIdx.x * WARPS * p.page_size;
        const int32_t* sd = (const int32_t*)(d_decomp_buf + base + 0 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* qt = (const int32_t*)(d_decomp_buf + base + 1 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* ep = (const int32_t*)(d_decomp_buf + base + 2 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* dc = (const int32_t*)(d_decomp_buf + base + 3 * (uint64_t)p.page_size) + HDR_INT32;

        // nalloc from L_SHIPDATE page header (canonical)
        uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base + 0 * (uint64_t)p.page_size);

        int64_t my_value = 0;
        for (uint32_t r = tid; r < nalloc; r += THREADS) {
            int32_t s = sd[r], e = ep[r], d = dc[r];
            bool pass = (s >= p.sd_low && s < p.sd_high);
            if (p.disc_lo != 0 || p.disc_hi != INT32_MAX)
                pass = pass && (d >= p.disc_lo && d <= p.disc_hi);
            if (p.qt_max > 0) {
                int32_t q = qt[r];
                pass = pass && (q < p.qt_max);
            }
            if (pass) {
                my_value += (int64_t)e * d;
            }
        }

        // Block-level reduction
        int64_t aggregate = BlockReduceInt(reduce_temp).Sum(my_value);
        if (tid == 0 && aggregate != 0) {
            atomicAdd(reinterpret_cast<unsigned long long int*>(p.d_revenue),
                      static_cast<unsigned long long int>(aggregate));
        }

        __syncthreads();  // Before next page iteration (protect reduce_temp)
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

struct BAMFusedRevenueContext {
    bam_io_page_cache_t io_pc;      // opaque page_cache handle
    void*       d_ctrls;            // Controller** (device)
    void*       d_pc_ptr;           // page_cache_d_t* (device)
    const char* pc_base_addr;       // page_cache base address (device)
    char*       d_decomp_buf;       // [num_slots * page_size]
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_revenue_ctx_t bam_fused_revenue_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedRevenueContext();
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
    FUSED_REV_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_revenue_ctx_t>(ctx);
}

static void bam_fused_revenue_launch(
    BAMFusedRevenueContext* ctx,
    const BAMFusedRevenueParams& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * 4;
        auto kernel_fn = bam_lz4_fused_revenue_kernel<PS>;
        FUSED_REV_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_REV_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_revenue_run_async(
    bam_fused_revenue_ctx_t ctx_handle,
    const BAMFusedRevenueParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedRevenueContext*>(ctx_handle);
    bam_fused_revenue_launch(ctx, params, stream);
}

void bam_fused_revenue_destroy(bam_fused_revenue_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedRevenueContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ════════════════════════════════════════════════════════════════
// Decomp+Scan only kernel: no BaM IO code
// Same as Q6 variant but with revenue predicates + CUB BlockReduce.
// ════════════════════════════════════════════════════════════════
static constexpr int DS_REV_WARPS = 4;

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void revenue_decomp_scan_kernel(
    char*                    d_decomp_buf,
    RevenueDecompScanParams  p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS   = DS_REV_WARPS;
    constexpr uint32_t THREADS = WARPS * 32;
    constexpr uint32_t HDR_INT32 = 3;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    const uint32_t slot = blockIdx.x * WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    for (uint32_t local_pg = blockIdx.x; local_pg < p.batch_count; local_pg += gridDim.x) {

        uint32_t orig_pg = p.d_batch_page_ids[local_pg];

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

        __syncthreads();

        uint64_t base = (uint64_t)blockIdx.x * WARPS * p.page_size;
        const int32_t* sd = (const int32_t*)(d_decomp_buf + base + 0 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* qt = (const int32_t*)(d_decomp_buf + base + 1 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* ep = (const int32_t*)(d_decomp_buf + base + 2 * (uint64_t)p.page_size) + HDR_INT32;
        const int32_t* dc = (const int32_t*)(d_decomp_buf + base + 3 * (uint64_t)p.page_size) + HDR_INT32;

        uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base + 0 * (uint64_t)p.page_size);

        int64_t my_value = 0;
        for (uint32_t r = tid; r < nalloc; r += THREADS) {
            int32_t s = sd[r], e = ep[r], d = dc[r];
            bool pass = (s >= p.sd_low && s < p.sd_high);
            if (p.disc_lo != 0 || p.disc_hi != INT32_MAX)
                pass = pass && (d >= p.disc_lo && d <= p.disc_hi);
            if (p.qt_max > 0) {
                int32_t q = qt[r];
                pass = pass && (q < p.qt_max);
            }
            if (pass) {
                my_value += (int64_t)e * d;
            }
        }

        if (my_value != 0) {
            atomicAdd(reinterpret_cast<unsigned long long int*>(p.d_revenue),
                      static_cast<unsigned long long int>(my_value));
        }

        __syncthreads();
    }
}

// ════════════════════════════════════════════════════════════════
// Decomp+Scan context and host API
// ════════════════════════════════════════════════════════════════
struct RevenueDecompScanContext {
    char*    d_decomp_buf;
    uint32_t page_size;
    uint32_t num_blocks;
};

revenue_decomp_scan_ctx_t revenue_decomp_scan_create(
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new RevenueDecompScanContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * DS_REV_WARPS;
    size_t decomp_size = (size_t)num_slots * page_size;
    FUSED_REV_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<revenue_decomp_scan_ctx_t>(ctx);
}

void revenue_decomp_scan_async(
    revenue_decomp_scan_ctx_t ctx_handle,
    const RevenueDecompScanParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<RevenueDecompScanContext*>(ctx_handle);
    constexpr uint32_t THREADS = DS_REV_WARPS * 32;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>()
                         * DS_REV_WARPS;
        auto kernel_fn = revenue_decomp_scan_kernel<PS>;
        FUSED_REV_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_decomp_buf, p);
    });

    FUSED_REV_CUDA_CHECK(cudaGetLastError());
}

void revenue_decomp_scan_destroy(revenue_decomp_scan_ctx_t ctx_handle)
{
    auto* ctx = static_cast<RevenueDecompScanContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    delete ctx;
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized Revenue kernel: 32 warps (1024 threads) / block
//   Warps 0-3:   IO (BaM page reads, 1 field/warp)
//   Warps 4-31:  Decomp (7 groups × 4 warps, nvCOMPdx LZ4)
//   All 1024 threads: Revenue scan
//
// Batch=7 pages per batch, double-buffered IO/decomp pipeline.
// Same structure as Q6 warp-spec kernel, different predicate:
//   Revenue: sd ∈ [low,high) && (qt_max==0 || q < qt_max)
// ════════════════════════════════════════════════════════════════

static constexpr uint32_t REVWS_BATCH      = 7;
static constexpr uint32_t REVWS_N_BUF      = 2;
static constexpr uint32_t REVWS_NUM_FIELDS = 4;
static constexpr uint32_t REVWS_IO_WARPS   = 4;
static constexpr uint32_t REVWS_DECOMP_GROUPS = 7;
static constexpr uint32_t REVWS_SLOTS_PER_BUF   = REVWS_BATCH * REVWS_NUM_FIELDS;   // 28
static constexpr uint32_t REVWS_SLOTS_PER_BLOCK = REVWS_N_BUF * REVWS_SLOTS_PER_BUF; // 56

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(1024, 1)
void revenue_warp_spec_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    RevenueWarpSpecParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t BATCH      = REVWS_BATCH;
    constexpr uint32_t N_BUF      = REVWS_N_BUF;
    constexpr uint32_t NUM_FIELDS = REVWS_NUM_FIELDS;
    constexpr uint32_t IO_WARPS   = REVWS_IO_WARPS;
    constexpr uint32_t SLOTS_PER_BUF   = REVWS_SLOTS_PER_BUF;
    constexpr uint32_t SLOTS_PER_BLOCK = REVWS_SLOTS_PER_BLOCK;
    constexpr uint32_t THREADS    = 1024;
    constexpr uint32_t HDR_INT32  = 3;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    __shared__ uint32_t s_comp_sz[N_BUF][BATCH][NUM_FIELDS];
    __shared__ uint32_t s_batch_count[N_BUF];

    // Active page list (compacted from d_page_mask before IO loop)
    __shared__ uint32_t s_active_pgs[kZonemapMaxPagesPerBlock];
    __shared__ uint32_t s_num_active;

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

    // Cache predicate constants in registers
    const int32_t qt_max = p.qt_max;
    const int32_t disc_lo = p.disc_lo;
    const int32_t disc_hi = p.disc_hi;

    // ── Prolog: IO warps read first batch into buf[0] ──
    {
        const uint32_t b_count = (BATCH < my_pages) ? BATCH : my_pages;
        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < b_count; j++) {
                uint32_t orig_pg = s_active_pgs[j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                fused_rev_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
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

    // ── Main loop ──
    uint32_t prev_buf = 0;

    for (uint32_t bstart = BATCH; bstart < my_pages; bstart += BATCH) {
        const uint32_t cur_buf    = 1 - prev_buf;
        const uint32_t rem        = my_pages - bstart;
        const uint32_t cur_count  = (BATCH < rem) ? BATCH : rem;
        const uint32_t prev_count = s_batch_count[prev_buf];

        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < cur_count; j++) {
                uint32_t orig_pg = s_active_pgs[bstart + j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                fused_rev_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
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
            const uint32_t dw    = warp_id - IO_WARPS;
            const uint32_t group = dw / NUM_FIELDS;
            const uint32_t field = dw % NUM_FIELDS;

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

        // Scan previous batch — revenue predicate
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
                    int32_t s = sd[r], e = ep[r], d = dc[r];
                    bool pass = (s >= p.sd_low && s < p.sd_high);
                    if (disc_lo != 0 || disc_hi != INT32_MAX)
                        pass = pass && (d >= disc_lo && d <= disc_hi);
                    if (qt_max > 0) {
                        int32_t q = qt[r];
                        pass = pass && (q < qt_max);
                    }
                    if (pass)
                        batch_val += (int64_t)e * d;
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
                    int32_t s = sd[r], e = ep[r], d = dc[r];
                    bool pass = (s >= p.sd_low && s < p.sd_high);
                    if (disc_lo != 0 || disc_hi != INT32_MAX)
                        pass = pass && (d >= disc_lo && d <= disc_hi);
                    if (qt_max > 0) {
                        int32_t q = qt[r];
                        pass = pass && (q < qt_max);
                    }
                    if (pass)
                        batch_val += (int64_t)e * d;
                }
            }
            if (batch_val != 0)
                atomicAdd(reinterpret_cast<unsigned long long int*>(p.d_revenue),
                          static_cast<unsigned long long int>(batch_val));
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Revenue Warp-Specialized: max co-resident blocks query
// ════════════════════════════════════════════════════════════════

uint32_t revenue_warp_spec_max_blocks(uint32_t page_size)
{
    int max_blocks_per_sm = 0;
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = REVWS_DECOMP_GROUPS * REVWS_NUM_FIELDS;  // 28

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = revenue_warp_spec_kernel<PS>;
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
    fprintf(stderr, "[revenue_warp_spec] max_blocks_per_sm=%d sm_count=%d max_total=%u\n",
            max_blocks_per_sm, sm_count, total);
    return total;
}

// ════════════════════════════════════════════════════════════════
// Revenue Warp-Specialized: launch function
// ════════════════════════════════════════════════════════════════

void revenue_warp_spec_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const RevenueWarpSpecParams& params,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = REVWS_DECOMP_GROUPS * REVWS_NUM_FIELDS;  // 28

    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = revenue_warp_spec_kernel<PS>;
        FUSED_REV_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kfn<<<num_blocks, THREADS, smem_size, stream>>>(
            d_ctrls, d_pc_ptr, pc_base_addr, d_decomp_buf, params);
    });
    FUSED_REV_CUDA_CHECK(cudaGetLastError());
}
