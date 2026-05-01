// bam_lz4_split_q16_partsupp.cu — Split IO/Compute for Q16 PARTSUPP
//
// Kernel 1 (IO Submit):  1 thread per IO, submits NVMe reads, writes (qp,cid) queue.
// Kernel 2 (Compute):    4 warps/block, polls queue, nvCOMPdx LZ4 decomp, HT probe.
//
// Compiled as C++17 with separable compilation + device linking.

#include "bam_lz4_split_q16_partsupp.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "tpch/page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define SPLIT_Q16PS_CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// ════════════════════════════════════════════════════════════════
// IO queue entry: written by submit kernel, read by compute kernel
// ════════════════════════════════════════════════════════════════
struct SplitQ16PSIoEntry {
    void*    qp;        // 8B  QueuePair* for polling
    uint16_t cid;       // 2B  NVMe command ID
    uint16_t pad;       // 2B
    uint32_t comp_sz;   // 4B  compressed size (for decomp)
};  // 16B

static constexpr uint64_t SPLIT_Q16PS_HT_EMPTY = UINT64_MAX;

// ════════════════════════════════════════════════════════════════
// Kernel 1: IO Submit — lightweight, 1 thread per IO
// ════════════════════════════════════════════════════════════════

__device__ static uint32_t split_q16ps_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

__global__ void split_q16ps_submit_kernel(
    void*       ctrls,
    void*       pc,
    SplitQ16PSIoEntry* queue,
    uint32_t    batch_npages,
    uint32_t    pg_base,
    // Field metadata (passed by value for simplicity)
    uint64_t    field_start_0, uint64_t field_start_1,
    const uint64_t* d_comp_offsets_0, const uint64_t* d_comp_offsets_1,
    const uint32_t* d_comp_sizes_0,   const uint32_t* d_comp_sizes_1,
    bool        is_compressed_0, bool is_compressed_1,
    uint64_t    part_lba_0, uint64_t part_lba_1,
    uint64_t    part_lba_2, uint64_t part_lba_3,
    uint32_t    n_devices,
    uint32_t    page_size)
{
    const uint32_t total_ios = batch_npages * 2;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_ios) return;

    // Field-major layout: [PK_page_0..PK_page_B-1, SK_page_0..SK_page_B-1]
    const uint32_t field = idx / batch_npages;   // 0=PK, 1=SK
    const uint32_t local_pg = idx % batch_npages;
    const uint32_t pg = pg_base + local_pg;

    const uint64_t field_start = (field == 0) ? field_start_0 : field_start_1;
    const uint64_t* comp_offsets = (field == 0) ? d_comp_offsets_0 : d_comp_offsets_1;
    const uint32_t* comp_sizes = (field == 0) ? d_comp_sizes_0 : d_comp_sizes_1;
    const bool is_compressed = (field == 0) ? is_compressed_0 : is_compressed_1;

    const uint64_t part_lbas[4] = {part_lba_0, part_lba_1, part_lba_2, part_lba_3};
    const uint32_t ndev = (n_devices > 1) ? n_devices : 1;

    uint64_t global_pg = field_start + pg;
    uint32_t dev = global_pg % ndev;
    uint64_t lba;
    uint32_t nblk;
    uint32_t comp_sz;

    if (is_compressed) {
        lba = part_lbas[dev] + comp_offsets[pg] / 512;
        comp_sz = comp_sizes[pg];
        nblk = split_q16ps_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg_dev = global_pg / ndev;
        lba = part_lbas[dev] + local_pg_dev * (page_size / 512);
        nblk = page_size / 512;
        comp_sz = page_size;
    }

    // Submit NVMe read: slot = idx (unique per IO in this batch)
    void* qp_out = nullptr;
    uint16_t cid_out = 0;
    bam_io_submit_page_device(ctrls, pc, lba, nblk, idx, dev, &qp_out, &cid_out);

    queue[idx].qp = qp_out;
    queue[idx].cid = cid_out;
    queue[idx].pad = 0;
    queue[idx].comp_sz = comp_sz;
}

// ════════════════════════════════════════════════════════════════
// Kernel 2: Compute — Poll + nvCOMPdx LZ4 decomp + HT probe
// ════════════════════════════════════════════════════════════════

__device__ static uint32_t split_q16ps_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

__device__ static uint32_t split_q16ps_ht_probe(
    const uint64_t *keys, const uint32_t *group_ids,
    uint32_t mask, uint64_t key)
{
    uint32_t slot = split_q16ps_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return group_ids[slot];
        if (k == SPLIT_Q16PS_HT_EMPTY) return UINT32_MAX;
        slot = (slot + 1) & mask;
    }
}

__device__ static bool split_q16ps_excl_probe(
    const uint64_t *keys, uint32_t mask, uint64_t key)
{
    uint32_t slot = split_q16ps_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return true;
        if (k == SPLIT_Q16PS_HT_EMPTY) return false;
        slot = (slot + 1) & mask;
    }
}

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void split_q16ps_compute_kernel(
    const char*             pc_base_addr,
    char*                   d_decomp_buf,
    const SplitQ16PSIoEntry* __restrict__ queue,
    BAMSplitQ16PSParams     p,
    uint32_t                batch_npages,
    uint32_t                pg_base)
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

    const uint32_t lane    = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t global_warp = blockIdx.x * WARPS + warp_id;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t total_warps = gridDim.x * WARPS;

    // Per-warp decomp buffers: 2 pages (PK + SK)
    char* my_pk_buf = d_decomp_buf + (uint64_t)global_warp * 2 * PAGE_SIZE_CONST;
    char* my_sk_buf = my_pk_buf + PAGE_SIZE_CONST;

    // Warp-stride loop over pages in this batch
    for (uint32_t local_pg = global_warp; local_pg < batch_npages; local_pg += total_warps) {
        // Queue indices: field-major layout
        const uint32_t pk_idx = local_pg;                 // PK: [0, batch_npages)
        const uint32_t sk_idx = batch_npages + local_pg;  // SK: [batch_npages, 2*batch_npages)

        // ── Poll + Decomp PK ──
        // IO was pre-submitted by Kernel 1; poll should return near-instantly.
        if (lane == 0)
            bam_io_poll_page_device(queue[pk_idx].qp, queue[pk_idx].cid);
        __syncwarp();

        {
            uint32_t comp_sz = queue[pk_idx].comp_sz;
            const char* src = pc_base_addr + (uint64_t)pk_idx * PAGE_SIZE_CONST;
            if (comp_sz < PAGE_SIZE_CONST) {
                auto decompressor = lz4_decomp_t();
                size_t dsz = 0;
                decompressor.execute(src, my_pk_buf, (size_t)comp_sz, &dsz, my_smem, nullptr);
            } else {
                const uint32_t n4 = PAGE_SIZE_CONST / 4;
                for (uint32_t i = lane; i < n4; i += 32)
                    reinterpret_cast<uint32_t*>(my_pk_buf)[i] =
                        reinterpret_cast<const uint32_t*>(src)[i];
            }
        }

        // ── Poll + Decomp SK ──
        if (lane == 0)
            bam_io_poll_page_device(queue[sk_idx].qp, queue[sk_idx].cid);
        __syncwarp();

        {
            uint32_t comp_sz = queue[sk_idx].comp_sz;
            const char* src = pc_base_addr + (uint64_t)sk_idx * PAGE_SIZE_CONST;
            if (comp_sz < PAGE_SIZE_CONST) {
                auto decompressor = lz4_decomp_t();
                size_t dsz = 0;
                decompressor.execute(src, my_sk_buf, (size_t)comp_sz, &dsz, my_smem, nullptr);
            } else {
                const uint32_t n4 = PAGE_SIZE_CONST / 4;
                for (uint32_t i = lane; i < n4; i += 32)
                    reinterpret_cast<uint32_t*>(my_sk_buf)[i] =
                        reinterpret_cast<const uint32_t*>(src)[i];
            }
        }

        // ── Probe ──
        const uint32_t pg = pg_base + local_pg;
        uint32_t nalloc = *(const uint32_t*)my_pk_buf;
        const uint64_t* pk = (const uint64_t*)(my_pk_buf + 16);
        const uint64_t* sk = (const uint64_t*)(my_sk_buf + 16);
        uint64_t row_base = p.d_ps[pg];

        for (uint32_t r = lane; r < nalloc; r += 32) {
            uint64_t partkey = pk[r];
            uint64_t suppkey = sk[r];

            uint32_t group_id = split_q16ps_ht_probe(
                p.d_ht_keys, p.d_ht_group_ids, p.ht_mask, partkey);
            if (group_id == UINT32_MAX) {
                p.d_emit_pairs[row_base + r] = SPLIT_Q16PS_HT_EMPTY;
                continue;
            }

            if (split_q16ps_excl_probe(p.d_excl_keys, p.excl_mask, suppkey)) {
                p.d_emit_pairs[row_base + r] = SPLIT_Q16PS_HT_EMPTY;
                continue;
            }

            p.d_emit_pairs[row_base + r] = ((uint64_t)group_id << 32) | (uint64_t)(uint32_t)suppkey;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

// Decomp buffer: 2 pages per warp (PK + SK), 4 warps per block
static constexpr uint32_t SPLIT_Q16PS_DECOMP_PAGES_PER_BLOCK = 8;

struct BAMSplitQ16PSContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    SplitQ16PSIoEntry* d_queue;
    cudaEvent_t submit_done;
    uint32_t    page_size;
    uint32_t    max_batch_pages;
    uint32_t    compute_blocks;
};

bam_split_q16ps_ctx_t bam_split_q16ps_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t max_batch_pages,
    uint32_t compute_blocks)
{
    fprintf(stderr, "[split_q16ps_create] page_size=%u batch=%u compute_blocks=%u\n",
            page_size, max_batch_pages, compute_blocks);

    auto* ctx = new BAMSplitQ16PSContext();
    ctx->page_size       = page_size;
    ctx->max_batch_pages = max_batch_pages;
    ctx->compute_blocks  = compute_blocks;

    // Page cache: 2 slots per page (PK + SK), one slot per IO
    const uint32_t total_slots = max_batch_pages * 2;
    fprintf(stderr, "[split_q16ps_create] creating page_cache with %u slots...\n", total_slots);
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, total_slots);
    fprintf(stderr, "[split_q16ps_create] page_cache created OK\n");

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: compute_blocks * 8 pages (4 warps × 2 fields)
    size_t decomp_size = (size_t)compute_blocks * SPLIT_Q16PS_DECOMP_PAGES_PER_BLOCK * page_size;
    fprintf(stderr, "[split_q16ps_create] cudaMalloc decomp_buf %zu bytes...\n", decomp_size);
    SPLIT_Q16PS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    // IO queue: one entry per IO
    SPLIT_Q16PS_CUDA_CHECK(cudaMalloc(&ctx->d_queue, total_slots * sizeof(SplitQ16PSIoEntry)));

    SPLIT_Q16PS_CUDA_CHECK(cudaEventCreate(&ctx->submit_done));
    fprintf(stderr, "[split_q16ps_create] all done\n");

    return static_cast<bam_split_q16ps_ctx_t>(ctx);
}

void bam_split_q16ps_run_async(
    bam_split_q16ps_ctx_t ctx_handle,
    const BAMSplitQ16PSParams& params,
    cudaStream_t stream_io,
    cudaStream_t stream_comp)
{
    auto* ctx = static_cast<BAMSplitQ16PSContext*>(ctx_handle);
    const uint32_t B = ctx->max_batch_pages;
    const uint32_t npages = params.npages;

    for (uint32_t pg_base = 0; pg_base < npages; pg_base += B) {
        uint32_t batch_np = std::min(B, npages - pg_base);
        uint32_t total_ios = batch_np * 2;

        // ── Kernel 1: IO Submit (lightweight, on stream_io) ──
        {
            constexpr uint32_t TPB = 32;  // 1 warp per block
            uint32_t grid = (total_ios + TPB - 1) / TPB;
            split_q16ps_submit_kernel<<<grid, TPB, 0, stream_io>>>(
                ctx->d_ctrls, ctx->d_pc_ptr, ctx->d_queue,
                batch_np, pg_base,
                params.field_start_page_ids[0], params.field_start_page_ids[1],
                params.d_comp_offsets[0], params.d_comp_offsets[1],
                params.d_comp_sizes[0], params.d_comp_sizes[1],
                params.is_compressed[0], params.is_compressed[1],
                params.partition_start_lbas[0], params.partition_start_lbas[1],
                params.partition_start_lbas[2], params.partition_start_lbas[3],
                params.n_devices, params.page_size);
            SPLIT_Q16PS_CUDA_CHECK(cudaGetLastError());
        }

        // Ensure all submissions done before compute kernel polls
        SPLIT_Q16PS_CUDA_CHECK(cudaEventRecord(ctx->submit_done, stream_io));
        SPLIT_Q16PS_CUDA_CHECK(cudaStreamWaitEvent(stream_comp, ctx->submit_done));

        // ── Kernel 2: Compute (on stream_comp) ──
        {
            constexpr uint32_t THREADS = 128;
            constexpr uint32_t WARPS   = 4;
            uint32_t eff_blocks = std::min(ctx->compute_blocks,
                                           (batch_np + WARPS - 1) / WARPS);

            dispatch_page_size(params.page_size, [&](auto ps_tag) {
                constexpr unsigned PS = decltype(ps_tag)::value;
                size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * WARPS;
                auto kernel_fn = split_q16ps_compute_kernel<PS>;
                SPLIT_Q16PS_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
                kernel_fn<<<eff_blocks, THREADS, smem_size, stream_comp>>>(
                    ctx->pc_base_addr, ctx->d_decomp_buf, ctx->d_queue,
                    params, batch_np, pg_base);
            });
            SPLIT_Q16PS_CUDA_CHECK(cudaGetLastError());
        }

        // Sync before next batch (reuse page_cache slots + queue)
        if (pg_base + B < npages) {
            SPLIT_Q16PS_CUDA_CHECK(cudaStreamSynchronize(stream_comp));
        }
    }
}

void bam_split_q16ps_destroy(bam_split_q16ps_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMSplitQ16PSContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    if (ctx->d_queue) cudaFree(ctx->d_queue);
    cudaEventDestroy(ctx->submit_done);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}
