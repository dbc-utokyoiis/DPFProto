// bam_lz4_fused_q3_lineitem.cu — Fused BaM I/O + nvCOMPdx LZ4 + Q3 LINEITEM probe
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q3_lineitem.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>

// From common/pruning.cuh — duplicated to avoid link errors
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

#define FUSED_Q3LI_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

__device__ static uint32_t fused_q3li_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// Binary search on prefix sum: find page containing global row gid
__device__ static uint32_t fused_q3li_ps_find_page(
    const uint64_t *ps, uint32_t n_entries, uint64_t gid)
{
    uint32_t lo = 0, hi = n_entries;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (ps[mid] <= gid) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

// Hash function — must match q3_hash64 used in q3_scan.cu for HT build
__device__ static uint32_t fused_q3li_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

static constexpr uint64_t FUSED_Q3LI_HT_EMPTY = UINT64_MAX;

// ORDERS HT probe (keys + uint64_t payloads)
__device__ static uint64_t fused_q3li_ht_probe_kv(
    const uint64_t *keys, const uint64_t *payloads,
    uint32_t mask, uint64_t key)
{
    uint32_t slot = fused_q3li_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return payloads[slot];
        if (k == FUSED_Q3LI_HT_EMPTY) return FUSED_Q3LI_HT_EMPTY;
        slot = (slot + 1) & mask;
    }
}

// Max INT64 pages that can span one INT32 page
constexpr uint32_t Q3LI_MAX_I64_PER_I32 = 3;

// Decomp buffer layout per block:
// [0] L_SHIPDATE (INT32)
// [1] L_EXTPRICE (INT32)
// [2] L_DISCOUNT (INT32)
// [3] L_ORDERKEY INT64 page 0
// [4] L_ORDERKEY INT64 page 1
// [5] L_ORDERKEY INT64 page 2 (rare)
constexpr uint32_t Q3LI_DECOMP_PAGES_PER_BLOCK = 3 + Q3LI_MAX_I64_PER_I32;  // 6

// ════════════════════════════════════════════════════════════════
// Fused Q3 LINEITEM kernel
// __launch_bounds__(128, 8) → 8 blocks/SM, 32 warps/SM
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void bam_lz4_fused_q3li_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ3LIParams p)
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
    constexpr uint32_t THREADS = WARPS * 32;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t slot    = blockIdx.x * WARPS + warp_id;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    // Shared memory for INT64 page mapping (computed by thread 0)
    __shared__ uint32_t s_i64_start;
    __shared__ uint32_t s_i64_count;
    __shared__ uint64_t s_i64_row_offset;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Helper: compute LBA/nblk for an INT32 field
    auto compute_i32 = [&](uint32_t fi, uint32_t pg,
                           uint64_t &lba, uint32_t &nblk, uint32_t &dev, uint32_t &comp_sz) {
        uint64_t global_pg = p.i32_field_start_page_ids[fi] + pg;
        dev = global_pg % ndev;
        if (p.is_compressed_i32[fi]) {
            lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i32[fi][pg] / 512;
            comp_sz = p.d_comp_sizes_i32[fi][pg];
            nblk = fused_q3li_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
        } else {
            uint64_t local_pg = global_pg / ndev;
            lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
            nblk = p.page_size / 512;
            comp_sz = p.page_size;
        }
    };

    // Helper: compute LBA/nblk for an INT64 field
    auto compute_i64 = [&](uint32_t pg,
                           uint64_t &lba, uint32_t &nblk, uint32_t &dev, uint32_t &comp_sz) {
        uint64_t global_pg = p.i64_field_start_page_ids[0] + pg;
        dev = global_pg % ndev;
        if (p.is_compressed_i64[0]) {
            lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i64[0][pg] / 512;
            comp_sz = p.d_comp_sizes_i64[0][pg];
            nblk = fused_q3li_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
        } else {
            uint64_t local_pg = global_pg / ndev;
            lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
            nblk = p.page_size / 512;
            comp_sz = p.page_size;
        }
    };

    // Helper: I/O + decompress one page
    auto io_decomp = [&](uint64_t lba, uint32_t nblk, uint32_t dev,
                         uint32_t comp_sz, char* dst) {
        bam_lz4_io_decomp_warp<PAGE_SIZE_CONST>(
            ctrls, pc, (void*)pc_base_addr,
            slot, dst, lba, nblk, dev, comp_sz, p.page_size, my_smem);
    };

    const uint64_t block_base = (uint64_t)blockIdx.x * Q3LI_DECOMP_PAGES_PER_BLOCK * p.page_size;

    // Active page count and stride
    const uint32_t n_active = p.n_active_pages ? p.n_active_pages : p.npages_i32;

    // Block-stride loop over active INT32 pages only
    for (uint32_t idx = blockIdx.x; idx < n_active; idx += gridDim.x) {
        uint32_t pg = p.d_active_pages ? p.d_active_pages[idx] : idx;

        // Thread 0: compute INT64 page range for this INT32 page
        if (tid == 0) {
            uint64_t first_row = p.d_ps_i32[pg];
            uint64_t last_row  = p.d_ps_i32[pg + 1];
            if (last_row > first_row) last_row--;
            s_i64_start = fused_q3li_ps_find_page(p.d_ps_i64, p.npages_i64 + 1, first_row);
            uint32_t i64_end = fused_q3li_ps_find_page(p.d_ps_i64, p.npages_i64 + 1, last_row);
            s_i64_count = i64_end - s_i64_start + 1;
            if (s_i64_count > Q3LI_MAX_I64_PER_I32) s_i64_count = Q3LI_MAX_I64_PER_I32;
            s_i64_row_offset = first_row - p.d_ps_i64[s_i64_start];
        }
        __syncthreads();

        uint32_t i64_start = s_i64_start;
        uint32_t i64_count = s_i64_count;
        uint64_t i64_row_offset = s_i64_row_offset;

        // ── Round 1: 4 warps read 4 pages in parallel ──
        // warp 0 → L_SHIPDATE[pg]          → decomp[0]
        // warp 1 → L_EXTPRICE[pg]          → decomp[1]
        // warp 2 → L_DISCOUNT[pg]          → decomp[2]
        // warp 3 → L_ORDERKEY[i64_start]   → decomp[3]
        {
            uint64_t lba; uint32_t nblk, dev, comp_sz;
            char* dst;

            if (warp_id < 3) {
                // INT32 fields: 0=L_SHIPDATE, 1=L_EXTPRICE, 2=L_DISCOUNT
                compute_i32(warp_id, pg, lba, nblk, dev, comp_sz);
                dst = d_decomp_buf + block_base + (uint64_t)warp_id * p.page_size;
            } else {
                // warp 3 → L_ORDERKEY[i64_start]
                compute_i64(i64_start, lba, nblk, dev, comp_sz);
                dst = d_decomp_buf + block_base + 3ULL * p.page_size;
            }
            io_decomp(lba, nblk, dev, comp_sz, dst);
        }
        __syncthreads();

        // ── Round 2+: additional INT64 pages (typically 1 more) ──
        for (uint32_t k = 1; k < i64_count; k++) {
            if (warp_id == 0) {
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                compute_i64(i64_start + k, lba, nblk, dev, comp_sz);
                char* dst = d_decomp_buf + block_base +
                    (uint64_t)(3 + k) * p.page_size;
                io_decomp(lba, nblk, dev, comp_sz, dst);
            }
            __syncthreads();
        }

        // ── Phase: Q3 LINEITEM probe + aggregate (all 128 threads) ──
        const char* shipdate_page = d_decomp_buf + block_base + 0 * (uint64_t)p.page_size;
        const char* extprice_page = d_decomp_buf + block_base + 1 * (uint64_t)p.page_size;
        const char* discount_page = d_decomp_buf + block_base + 2 * (uint64_t)p.page_size;

        uint32_t nalloc = *(const uint32_t*)shipdate_page;
        const int32_t* sd = (const int32_t*)(shipdate_page + 12);  // offset 12 for INT32
        const int32_t* ep = (const int32_t*)(extprice_page + 12);
        const int32_t* dc = (const int32_t*)(discount_page + 12);

        // Read nalloc of each INT64 page
        uint32_t i64_nalloc[Q3LI_MAX_I64_PER_I32];
        for (uint32_t k = 0; k < i64_count; k++) {
            const char* ok_page = d_decomp_buf + block_base +
                (uint64_t)(3 + k) * p.page_size;
            i64_nalloc[k] = *(const uint32_t*)ok_page;
        }

        for (uint32_t r = tid; r < nalloc; r += THREADS) {
            // Filter: l_shipdate > 19950315 (skipped for Q3SEL)
            int32_t shipdate = sd[r];
            if (!p.skip_shipdate_filter && shipdate <= 19950315) continue;

            // Map record to INT64 page
            uint64_t i64_local_row = i64_row_offset + r;
            uint32_t i64_pg_local = 0;
            uint64_t cumul = 0;
            for (uint32_t k = 0; k < i64_count; k++) {
                if (i64_local_row < cumul + i64_nalloc[k]) {
                    i64_pg_local = k;
                    break;
                }
                cumul += i64_nalloc[k];
                i64_pg_local = k + 1;
            }
            uint32_t i64_rec = (uint32_t)(i64_local_row - cumul);

            // Read L_ORDERKEY from INT64 page
            const char* ok_page = d_decomp_buf + block_base +
                (uint64_t)(3 + i64_pg_local) * p.page_size;
            uint64_t orderkey = *(const uint64_t*)(ok_page + 16 + (uint64_t)i64_rec * 8);

            // Probe ORDERS HT
            uint64_t payload = fused_q3li_ht_probe_kv(
                p.d_orders_ht_keys, p.d_orders_ht_payloads, p.orders_ht_mask, orderkey);
            if (payload == FUSED_Q3LI_HT_EMPTY) continue;

            // Revenue: l_extendedprice * (100 - l_discount)
            int32_t extprice = ep[r];
            int32_t discount = dc[r];
            int64_t revenue = (int64_t)extprice * (int64_t)(100 - discount);

            // Aggregate into GROUP BY hash table (key: l_orderkey)
            uint32_t aggr_slot = fused_q3li_hash64(orderkey) & p.aggr_mask;
            while (true) {
                uint64_t prev = atomicCAS(
                    reinterpret_cast<unsigned long long *>(&p.d_aggr_keys[aggr_slot]),
                    (unsigned long long)FUSED_Q3LI_HT_EMPTY,
                    (unsigned long long)orderkey);
                if (prev == FUSED_Q3LI_HT_EMPTY || prev == orderkey) {
                    atomicAdd(reinterpret_cast<unsigned long long *>(&p.d_aggr_revenues[aggr_slot]),
                              (unsigned long long)revenue);
                    break;
                }
                aggr_slot = (aggr_slot + 1) & p.aggr_mask;
            }
        }

        __syncthreads();
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

struct BAMFusedQ3LIContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_q3li_ctx_t bam_fused_q3li_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ3LIContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * 4;
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: Q3LI_DECOMP_PAGES_PER_BLOCK pages per block
    size_t decomp_size = (size_t)num_blocks * Q3LI_DECOMP_PAGES_PER_BLOCK * page_size;
    FUSED_Q3LI_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q3li_ctx_t>(ctx);
}

static void bam_fused_q3li_launch(
    BAMFusedQ3LIContext* ctx,
    const BAMFusedQ3LIParams& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * 4;
        auto kernel_fn = bam_lz4_fused_q3li_kernel<PS>;
        FUSED_Q3LI_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_Q3LI_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q3li_run_async(
    bam_fused_q3li_ctx_t ctx_handle,
    const BAMFusedQ3LIParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ3LIContext*>(ctx_handle);
    bam_fused_q3li_launch(ctx, params, stream);
}

void bam_fused_q3li_destroy(bam_fused_q3li_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ3LIContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized Q3 LINEITEM kernel
//
// 32 warps = 1024 threads per block:
//   Warps 0-3:   IO (4 warps)
//     Warps 0-2: each reads 1 INT32 field for all BATCH pages
//     Warp 3:    computes INT64 metadata + reads INT64 pages
//   Warps 4-31:  Decomp (7 groups × 4 warps)
//     Group g processes page g of batch:
//       warp_in_group 0: L_SHIPDATE decomp
//       warp_in_group 1: L_EXTPRICE decomp
//       warp_in_group 2: L_DISCOUNT decomp
//       warp_in_group 3: L_ORDERKEY INT64[0] decomp
//       Round 2 (if i64_count>1): warp 0→INT64[1], warp 1→INT64[2]
//   All 1024 threads: scan (HT probe + aggregation)
//
// Double-buffered: IO[batch N+1] || Decomp[batch N] → Scan[batch N]
//
// Slot layout per page in batch:
//   [0] L_SHIPDATE, [1] L_EXTPRICE, [2] L_DISCOUNT,
//   [3] L_ORDERKEY INT64[0], [4] INT64[1], [5] INT64[2]
// ════════════════════════════════════════════════════════════════

static constexpr uint32_t Q3LIWS_BATCH      = 7;
static constexpr uint32_t Q3LIWS_N_BUF      = 2;
static constexpr uint32_t Q3LIWS_I32_FIELDS = 3;
static constexpr uint32_t Q3LIWS_MAX_I64    = 3;
static constexpr uint32_t Q3LIWS_SLOTS_PER_PAGE  = Q3LIWS_I32_FIELDS + Q3LIWS_MAX_I64;  // 6
static constexpr uint32_t Q3LIWS_IO_WARPS   = 4;
static constexpr uint32_t Q3LIWS_DECOMP_GROUPS = 7;
static constexpr uint32_t Q3LIWS_WARPS_PER_GROUP = 4;
static constexpr uint32_t Q3LIWS_SLOTS_PER_BUF   = Q3LIWS_BATCH * Q3LIWS_SLOTS_PER_PAGE;  // 42
static constexpr uint32_t Q3LIWS_SLOTS_PER_BLOCK = Q3LIWS_N_BUF * Q3LIWS_SLOTS_PER_BUF;   // 84

// I/O parameter helpers
__device__ static void q3liws_io_params_i32(
    const Q3LIWarpSpecParams& p,
    uint32_t fi, uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.i32_field_start_page_ids[fi] + pg;
    dev = global_pg % ndev;
    if (p.is_compressed_i32[fi]) {
        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i32[fi][pg] / 512;
        comp_sz = p.d_comp_sizes_i32[fi][pg];
        nblk = fused_q3li_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

__device__ static void q3liws_io_params_i64(
    const Q3LIWarpSpecParams& p,
    uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.i64_field_start_page_ids[0] + pg;
    dev = global_pg % ndev;
    if (p.is_compressed_i64[0]) {
        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i64[0][pg] / 512;
        comp_sz = p.d_comp_sizes_i64[0][pg];
        nblk = fused_q3li_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

// Prefix sum binary search (same as fused_q3li_ps_find_page)
__device__ static uint32_t q3liws_ps_find_page(
    const uint64_t *ps, uint32_t n_entries, uint64_t gid)
{
    uint32_t lo = 0, hi = n_entries;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (ps[mid] <= gid) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(1024, 1)
void q3li_warp_spec_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    Q3LIWarpSpecParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t BATCH      = Q3LIWS_BATCH;
    constexpr uint32_t N_BUF      = Q3LIWS_N_BUF;
    constexpr uint32_t I32_FIELDS = Q3LIWS_I32_FIELDS;
    constexpr uint32_t MAX_I64    = Q3LIWS_MAX_I64;
    constexpr uint32_t SPP        = Q3LIWS_SLOTS_PER_PAGE;
    constexpr uint32_t IO_WARPS   = Q3LIWS_IO_WARPS;
    constexpr uint32_t SLOTS_PER_BUF   = Q3LIWS_SLOTS_PER_BUF;
    constexpr uint32_t SLOTS_PER_BLOCK = Q3LIWS_SLOTS_PER_BLOCK;
    constexpr uint32_t THREADS    = 1024;
    constexpr uint32_t HDR_I32    = 3;   // INT32 page header: nalloc(4)+flags(4)+stride(4) = 12B = 3 int32s
    constexpr uint64_t HT_EMPTY   = UINT64_MAX;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    // ── Shared memory ──
    __shared__ uint32_t s_comp_sz[N_BUF][BATCH][SPP];
    __shared__ uint32_t s_batch_count[N_BUF];
    __shared__ uint32_t s_i64_count[N_BUF][BATCH];
    __shared__ uint32_t s_i64_start[N_BUF][BATCH];
    __shared__ uint64_t s_i64_row_offset[N_BUF][BATCH];

    // Dynamic: nvCOMPdx per-decomp-warp region (28 warps)
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = (warp_id >= IO_WARPS)
        ? smem + (warp_id - IO_WARPS) * warp_smem : nullptr;

    // Active page list (compacted from d_page_mask before IO loop)
    __shared__ uint32_t s_active_pgs[kZonemapMaxPagesPerBlock];
    __shared__ uint32_t s_num_active;

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

    // ── IO helper lambda: read one batch of pages ──
    // Warps 0-2: INT32 fields, Warp 3: metadata + INT64 pages
    auto io_read_batch = [&](uint32_t bstart, uint32_t bcount, uint32_t buf) {
        if (warp_id < I32_FIELDS) {
            // Warps 0-2: each reads its INT32 field for all pages in batch
            const uint32_t fi = warp_id;
            for (uint32_t j = 0; j < bcount; j++) {
                uint32_t orig_pg = s_active_pgs[bstart + j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                q3liws_io_params_i32(p, fi, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = block_slot_base + buf * SLOTS_PER_BUF
                              + j * SPP + fi;
                if (lane == 0) {
                    bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
                    s_comp_sz[buf][j][fi] = comp_sz;
                }
                __syncwarp();
            }
        } else if (warp_id == 3) {
            // Warp 3: compute INT64 metadata + read INT64 pages
            for (uint32_t j = 0; j < bcount; j++) {
                uint32_t orig_pg = s_active_pgs[bstart + j];

                // Compute INT64 page range
                uint32_t i64_s, i64_c;
                uint64_t i64_ro;
                if (lane == 0) {
                    uint64_t first_row = p.d_ps_i32[orig_pg];
                    uint64_t last_row  = p.d_ps_i32[orig_pg + 1];
                    if (last_row > first_row) last_row--;
                    i64_s = q3liws_ps_find_page(p.d_ps_i64, p.npages_i64 + 1, first_row);
                    uint32_t i64_e = q3liws_ps_find_page(p.d_ps_i64, p.npages_i64 + 1, last_row);
                    i64_c = i64_e - i64_s + 1;
                    if (i64_c > MAX_I64) i64_c = MAX_I64;
                    i64_ro = first_row - p.d_ps_i64[i64_s];
                    s_i64_count[buf][j] = i64_c;
                    s_i64_start[buf][j] = i64_s;
                    s_i64_row_offset[buf][j] = i64_ro;
                }
                i64_c = __shfl_sync(0xFFFFFFFF, i64_c, 0);
                i64_s = __shfl_sync(0xFFFFFFFF, i64_s, 0);

                // Read INT64 pages
                for (uint32_t k = 0; k < i64_c; k++) {
                    uint64_t lba; uint32_t nblk, dev, comp_sz;
                    q3liws_io_params_i64(p, i64_s + k, ndev, lba, nblk, dev, comp_sz);
                    uint32_t slot = block_slot_base + buf * SLOTS_PER_BUF
                                  + j * SPP + I32_FIELDS + k;
                    if (lane == 0) {
                        bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
                        s_comp_sz[buf][j][I32_FIELDS + k] = comp_sz;
                    }
                    __syncwarp();
                }
            }
        }
        if (warp_id == 0 && lane == 0)
            s_batch_count[buf] = bcount;
    };

    // ── Decomp helper: decomp warps process one batch ──
    auto decomp_batch = [&](uint32_t buf) {
        if (warp_id < IO_WARPS) return;

        const uint32_t dw    = warp_id - IO_WARPS;     // 0..27
        const uint32_t group = dw / Q3LIWS_WARPS_PER_GROUP;   // 0..6
        const uint32_t wig   = dw % Q3LIWS_WARPS_PER_GROUP;   // 0..3

        const uint32_t bcount = s_batch_count[buf];
        if (group >= bcount) return;

        const uint32_t i64_c = s_i64_count[buf][group];

        // Round 1: 4 parallel decomps
        // wig 0→L_SHIPDATE, 1→L_EXTPRICE, 2→L_DISCOUNT, 3→INT64[0]
        {
            uint32_t field_idx = (wig < I32_FIELDS) ? wig : I32_FIELDS;
            uint32_t slot = block_slot_base + buf * SLOTS_PER_BUF
                          + group * SPP + field_idx;
            uint32_t comp_sz = s_comp_sz[buf][group][field_idx];
            char* dst = d_decomp_buf + (uint64_t)slot * p.page_size;
            bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                pc_base_addr, slot, dst, comp_sz, p.page_size, my_smem);
        }

        // Round 2: additional INT64 pages (i64_count > 1)
        if (i64_c > 1 && wig < (i64_c - 1)) {
            uint32_t field_idx = I32_FIELDS + 1 + wig;  // INT64[1] or INT64[2]
            uint32_t slot = block_slot_base + buf * SLOTS_PER_BUF
                          + group * SPP + field_idx;
            uint32_t comp_sz = s_comp_sz[buf][group][field_idx];
            char* dst = d_decomp_buf + (uint64_t)slot * p.page_size;
            bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                pc_base_addr, slot, dst, comp_sz, p.page_size, my_smem);
        }
    };

    // ── Scan helper: all 1024 threads scan one batch ──
    auto scan_batch = [&](uint32_t buf) {
        const uint32_t bcount = s_batch_count[buf];

        for (uint32_t j = 0; j < bcount; j++) {
            uint64_t base = (uint64_t)(block_slot_base
                + buf * SLOTS_PER_BUF + j * SPP) * p.page_size;

            const int32_t* sd = (const int32_t*)(d_decomp_buf + base
                + 0 * (uint64_t)p.page_size) + HDR_I32;
            const int32_t* ep = (const int32_t*)(d_decomp_buf + base
                + 1 * (uint64_t)p.page_size) + HDR_I32;
            const int32_t* dc = (const int32_t*)(d_decomp_buf + base
                + 2 * (uint64_t)p.page_size) + HDR_I32;
            uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base);

            uint32_t i64_c  = s_i64_count[buf][j];
            uint64_t i64_ro = s_i64_row_offset[buf][j];

            // Read nalloc of each INT64 page
            uint32_t i64_nalloc[MAX_I64];
            for (uint32_t k = 0; k < i64_c; k++) {
                const char* ok_page = d_decomp_buf + base
                    + (uint64_t)(I32_FIELDS + k) * p.page_size;
                i64_nalloc[k] = *(const uint32_t*)ok_page;
            }

            for (uint32_t r = tid; r < nalloc; r += THREADS) {
                int32_t shipdate = sd[r];
                if (!p.skip_shipdate_filter && shipdate <= 19950315) continue;

                // Map record to INT64 page
                uint64_t i64_local_row = i64_ro + r;
                uint32_t i64_pg_local = 0;
                uint64_t cumul = 0;
                for (uint32_t k = 0; k < i64_c; k++) {
                    if (i64_local_row < cumul + i64_nalloc[k]) {
                        i64_pg_local = k;
                        break;
                    }
                    cumul += i64_nalloc[k];
                    i64_pg_local = k + 1;
                }
                uint32_t i64_rec = (uint32_t)(i64_local_row - cumul);

                // Read L_ORDERKEY from INT64 page
                const char* ok_page = d_decomp_buf + base
                    + (uint64_t)(I32_FIELDS + i64_pg_local) * p.page_size;
                uint64_t orderkey = *(const uint64_t*)(ok_page + 16 + (uint64_t)i64_rec * 8);

                // Probe ORDERS HT
                uint64_t payload = fused_q3li_ht_probe_kv(
                    p.d_orders_ht_keys, p.d_orders_ht_payloads,
                    p.orders_ht_mask, orderkey);
                if (payload == HT_EMPTY) continue;

                // Revenue
                int64_t revenue = (int64_t)ep[r] * (int64_t)(100 - dc[r]);

                // Aggregate into GROUP BY hash table
                uint32_t aggr_slot = fused_q3li_hash64(orderkey) & p.aggr_mask;
                while (true) {
                    uint64_t prev = atomicCAS(
                        reinterpret_cast<unsigned long long*>(&p.d_aggr_keys[aggr_slot]),
                        (unsigned long long)HT_EMPTY,
                        (unsigned long long)orderkey);
                    if (prev == HT_EMPTY || prev == orderkey) {
                        atomicAdd(
                            reinterpret_cast<unsigned long long*>(&p.d_aggr_revenues[aggr_slot]),
                            (unsigned long long)revenue);
                        break;
                    }
                    aggr_slot = (aggr_slot + 1) & p.aggr_mask;
                }
            }
        }
    };

    // ══════════════════════════════════════════
    // Prolog: IO warps read first batch into buf[0]
    // ══════════════════════════════════════════
    {
        uint32_t b_count = (BATCH < my_pages) ? BATCH : my_pages;
        io_read_batch(0, b_count, 0);
    }
    __syncthreads();

    // ══════════════════════════════════════════
    // Main loop: IO[batch N+1] || Decomp[batch N] → Scan[batch N]
    // ══════════════════════════════════════════
    uint32_t prev_buf = 0;

    for (uint32_t bstart = BATCH; bstart < my_pages; bstart += BATCH) {
        const uint32_t cur_buf   = 1 - prev_buf;
        const uint32_t rem       = my_pages - bstart;
        const uint32_t cur_count = (BATCH < rem) ? BATCH : rem;

        // Phase A: IO reads next batch || Decomp processes previous batch
        if (warp_id < IO_WARPS) {
            io_read_batch(bstart, cur_count, cur_buf);
        } else {
            decomp_batch(prev_buf);
        }
        __syncthreads();

        // Phase B: Scan previous batch (all 1024 threads)
        scan_batch(prev_buf);
        __syncthreads();

        prev_buf = cur_buf;
    }

    // ══════════════════════════════════════════
    // Epilog: Decomp + Scan last batch
    // ══════════════════════════════════════════
    {
        decomp_batch(prev_buf);
        __syncthreads();
        scan_batch(prev_buf);
    }
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized: max co-resident blocks query
// ════════════════════════════════════════════════════════════════

uint32_t q3li_warp_spec_max_blocks(uint32_t page_size)
{
    int max_blocks_per_sm = 0;
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = Q3LIWS_DECOMP_GROUPS * Q3LIWS_WARPS_PER_GROUP;  // 28

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = q3li_warp_spec_kernel<PS>;
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
    fprintf(stderr, "[q3li_warp_spec] max_blocks_per_sm=%d sm_count=%d max_total=%u\n",
            max_blocks_per_sm, sm_count, total);
    return total;
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized: launch function
// ════════════════════════════════════════════════════════════════

void q3li_warp_spec_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const Q3LIWarpSpecParams& params,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = Q3LIWS_DECOMP_GROUPS * Q3LIWS_WARPS_PER_GROUP;  // 28

    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = q3li_warp_spec_kernel<PS>;
        FUSED_Q3LI_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kfn<<<num_blocks, THREADS, smem_size, stream>>>(
            d_ctrls, d_pc_ptr, pc_base_addr, d_decomp_buf, params);
    });
    FUSED_Q3LI_CUDA_CHECK(cudaGetLastError());
}
