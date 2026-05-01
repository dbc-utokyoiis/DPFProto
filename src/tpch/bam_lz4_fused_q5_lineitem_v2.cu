// bam_lz4_fused_q5_lineitem_v2.cu — Q1-style balanced pipeline for Q5 LINEITEM
//
// 1 IO warp + 8 decomp warps (288 threads = 9 warps) per block, grid-stride
// loop over active INT32 pages. Double-buffered IO ring + decomp face.

#include "bam_lz4_fused_q5_lineitem_v2.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>

#define FUSED_Q5LI_V2_CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

__device__ static uint32_t q5li_v2_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

__device__ static uint32_t q5li_v2_ps_find_page(
    const uint64_t* ps, uint32_t n_entries, uint64_t gid)
{
    uint32_t lo = 0, hi = n_entries;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (ps[mid] <= gid) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

__device__ static uint32_t q5li_v2_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

__device__ static int32_t q5li_v2_ht_probe(
    const uint64_t* keys, const int32_t* values,
    uint32_t mask, uint64_t key)
{
    uint32_t slot = q5li_v2_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return values[slot];
        if (k == 0xFFFFFFFFFFFFFFFFULL) return -1;
        slot = (slot + 1) & mask;
    }
}

static constexpr uint32_t Q5LI_V2_I32_FIELDS = 2;
static constexpr uint32_t Q5LI_V2_I64_FIELDS = 2;
static constexpr uint32_t Q5LI_V2_MAX_I64    = 3;
static constexpr uint32_t Q5LI_V2_SLOTS_PER_PAGE =
    Q5LI_V2_I32_FIELDS + Q5LI_V2_MAX_I64 * Q5LI_V2_I64_FIELDS;  // 8
static constexpr uint32_t Q5LI_V2_NBUF  = 2;
static constexpr uint32_t Q5LI_V2_NFACE = 2;
static constexpr uint32_t Q5LI_V2_WARPS = 9;  // 1 IO + 8 decomp
static constexpr uint32_t Q5LI_V2_THREADS = Q5LI_V2_WARPS * 32;  // 288
static constexpr uint32_t Q5LI_V2_SLOTS_PER_BLOCK =
    Q5LI_V2_NBUF * Q5LI_V2_SLOTS_PER_PAGE;  // 16

// IO warp lane 0: issue all slots for one INT32 page and record metadata in shared mem.
__device__ __forceinline__ static void q5li_v2_issue_page_io(
    void* ctrls, void* pc,
    const BAMFusedQ5LIV2Params& p,
    uint32_t orig_pg,
    uint32_t ring,
    uint32_t ndev,
    uint32_t (&s_comp_sz)[Q5LI_V2_NBUF][Q5LI_V2_SLOTS_PER_PAGE],
    uint32_t (&s_i64_c)[Q5LI_V2_NBUF],
    uint64_t (&s_i64_row_offset)[Q5LI_V2_NBUF])
{
    // INT64 mapping (needed before INT64 IOs)
    uint64_t first_row = p.d_ps_i32[orig_pg];
    uint64_t last_row  = p.d_ps_i32[orig_pg + 1];
    if (last_row > first_row) last_row--;
    uint32_t i64_s = q5li_v2_ps_find_page(p.d_ps_i64, p.npages_i64 + 1, first_row);
    uint32_t i64_e = q5li_v2_ps_find_page(p.d_ps_i64, p.npages_i64 + 1, last_row);
    uint32_t i64_c = i64_e - i64_s + 1;
    if (i64_c > Q5LI_V2_MAX_I64) i64_c = Q5LI_V2_MAX_I64;
    s_i64_c[ring] = i64_c;
    s_i64_row_offset[ring] = first_row - p.d_ps_i64[i64_s];

    // INT32 slots 0..1
    for (uint32_t fi = 0; fi < Q5LI_V2_I32_FIELDS; fi++) {
        uint64_t global_pg = p.i32_field_start_page_ids[fi] + orig_pg;
        uint32_t dev = global_pg % ndev;
        uint64_t lba; uint32_t nblk, comp_sz;
        if (p.is_compressed_i32[fi]) {
            lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i32[fi][orig_pg] / 512;
            comp_sz = p.d_comp_sizes_i32[fi][orig_pg];
            nblk = q5li_v2_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
        } else {
            uint64_t local_pg = global_pg / ndev;
            lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
            nblk = p.page_size / 512;
            comp_sz = p.page_size;
        }
        uint32_t slot = blockIdx.x * Q5LI_V2_SLOTS_PER_BLOCK
                      + ring * Q5LI_V2_SLOTS_PER_PAGE + fi;
        bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
        s_comp_sz[ring][fi] = comp_sz;
    }

    // INT64 slots 2..2+2*i64_c
    for (uint32_t k = 0; k < i64_c; k++) {
        for (uint32_t fi64 = 0; fi64 < Q5LI_V2_I64_FIELDS; fi64++) {
            uint32_t slot_idx = Q5LI_V2_I32_FIELDS + k * Q5LI_V2_I64_FIELDS + fi64;
            uint64_t global_pg = p.i64_field_start_page_ids[fi64] + (i64_s + k);
            uint32_t dev = global_pg % ndev;
            uint64_t lba; uint32_t nblk, comp_sz;
            if (p.is_compressed_i64[fi64]) {
                lba = p.partition_start_lbas[dev] +
                      p.d_comp_offsets_i64[fi64][i64_s + k] / 512;
                comp_sz = p.d_comp_sizes_i64[fi64][i64_s + k];
                nblk = q5li_v2_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
                nblk = p.page_size / 512;
                comp_sz = p.page_size;
            }
            uint32_t slot = blockIdx.x * Q5LI_V2_SLOTS_PER_BLOCK
                          + ring * Q5LI_V2_SLOTS_PER_PAGE + slot_idx;
            bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
            s_comp_sz[ring][slot_idx] = comp_sz;
        }
    }
}

// Scan one decompressed INT32 page: probe ORDERS & SUPPLIER HT, accumulate revenue.
// Uses i64_c INT64 pages laid out at slots [2..2+2*i64_c).
__device__ __forceinline__ static void q5li_v2_scan_page(
    const char* base,
    uint32_t page_size,
    uint32_t tid,
    uint32_t i64_c,
    uint64_t i64_row_offset,
    const BAMFusedQ5LIV2Params& p)
{
    const char* extprice_page = base + 0 * (uint64_t)page_size;
    const char* discount_page = base + 1 * (uint64_t)page_size;

    uint32_t nalloc = *(const uint32_t*)extprice_page;
    const int32_t* ep = (const int32_t*)(extprice_page + 12);
    const int32_t* dc = (const int32_t*)(discount_page + 12);

    uint32_t i64_nalloc[Q5LI_V2_MAX_I64];
    #pragma unroll
    for (uint32_t k = 0; k < Q5LI_V2_MAX_I64; k++) {
        if (k < i64_c) {
            const char* ok_page = base +
                (uint64_t)(Q5LI_V2_I32_FIELDS + k * Q5LI_V2_I64_FIELDS) * page_size;
            i64_nalloc[k] = *(const uint32_t*)ok_page;
        } else {
            i64_nalloc[k] = 0;
        }
    }

    for (uint32_t r = tid; r < nalloc; r += Q5LI_V2_THREADS) {
        uint64_t i64_local_row = i64_row_offset + r;
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

        const char* ok_page = base +
            (uint64_t)(Q5LI_V2_I32_FIELDS + i64_pg_local * Q5LI_V2_I64_FIELDS) * page_size;
        uint64_t orderkey = *(const uint64_t*)(ok_page + 16 + (uint64_t)i64_rec * 8);

        int32_t cust_nation_idx = q5li_v2_ht_probe(
            p.d_ht_ord_keys, p.d_ht_ord_values, p.ht_ord_mask, orderkey);
        if (cust_nation_idx < 0) continue;

        const char* sk_page = base +
            (uint64_t)(Q5LI_V2_I32_FIELDS + i64_pg_local * Q5LI_V2_I64_FIELDS + 1) * page_size;
        uint64_t suppkey = *(const uint64_t*)(sk_page + 16 + (uint64_t)i64_rec * 8);

        int32_t supp_nation_idx = q5li_v2_ht_probe(
            p.d_ht_supp_keys, p.d_ht_supp_values, p.ht_supp_mask, suppkey);
        if (supp_nation_idx < 0) continue;

        if (cust_nation_idx != supp_nation_idx) continue;

        int64_t revenue = (int64_t)ep[r] * (int64_t)(100 - dc[r]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&p.d_revenue[cust_nation_idx]),
                  (unsigned long long)revenue);
    }
}

// Resolve the N-th active page index in the grid-stride sequence for this block.
// Returns npages_i32 (invalid sentinel) if out-of-range; otherwise the original INT32 page ID.
__device__ __forceinline__ static uint32_t q5li_v2_nth_active(
    const BAMFusedQ5LIV2Params& p, uint32_t n)
{
    if (n >= p.total_pages) return p.npages_i32;
    if (p.d_active_page_ids) return p.d_active_page_ids[n];
    return n;  // no pruning: sequential full range
}

__device__ __forceinline__ static bool q5li_v2_page_active(
    const BAMFusedQ5LIV2Params& p, uint32_t orig_pg)
{
    if (p.d_active_page_ids) return true;  // list contains only active
    if (!p.d_page_mask) return true;
    return p.d_page_mask[orig_pg] != 0;
}

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(288, 1)
void bam_lz4_fused_q5li_v2_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ5LIV2Params p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t NBUF  = Q5LI_V2_NBUF;
    constexpr uint32_t NFACE = Q5LI_V2_NFACE;
    constexpr uint32_t SPP   = Q5LI_V2_SLOTS_PER_PAGE;
    constexpr uint32_t WARPS = Q5LI_V2_WARPS;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    // Shared: comp_sz communication IO → decomp, per ring buffer
    __shared__ uint32_t s_comp_sz[NBUF][SPP];
    __shared__ uint32_t s_i64_c[NBUF];
    __shared__ uint64_t s_i64_row_offset[NBUF];

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Face stride: one face = SPP × page_size (only first 2+2*i64_c slots populated)
    const uint64_t face_stride  = (uint64_t)SPP * p.page_size;
    const uint64_t block_stride = (uint64_t)NFACE * face_stride;
    const uint64_t block_base   = (uint64_t)blockIdx.x * block_stride;

    const uint32_t pg_stride = gridDim.x;
    const uint32_t pg_start  = blockIdx.x;

    int  io_ring     = 0;
    int  decomp_face = 0;
    bool prev_valid  = false;

    // IO warp helper: issue all slots for one INT32 page
    auto do_io = [&](uint32_t orig_pg, uint32_t ring) {
        if (lane == 0) {
            q5li_v2_issue_page_io(ctrls, pc, p, orig_pg, ring, ndev,
                                  s_comp_sz, s_i64_c, s_i64_row_offset);
        }
        __syncwarp();
    };

    // Priming: IO warp reads first active page → ring[0]
    const uint32_t first_pg = q5li_v2_nth_active(p, pg_start);
    if (first_pg < p.npages_i32) {
        if (warp_id == 0) do_io(first_pg, 0);
    }
    __syncthreads();

    for (uint32_t nth = pg_start; nth < p.total_pages; nth += pg_stride) {
        const uint32_t cur_pg  = q5li_v2_nth_active(p, nth);
        const uint32_t nxt_nth = nth + pg_stride;
        const bool     has_next = (nxt_nth < p.total_pages);
        const uint32_t nxt_pg  = has_next ? q5li_v2_nth_active(p, nxt_nth) : p.npages_i32;

        // Phase A: IO(next) ∥ decomp(current)
        if (warp_id == 0 && has_next && nxt_pg < p.npages_i32) {
            const int next_ring = 1 - io_ring;
            do_io(nxt_pg, next_ring);
        }

        // Decomp warps 1..8: each handles slot (warp_id - 1)
        if (warp_id >= 1 && warp_id <= SPP) {
            const uint32_t sl = warp_id - 1;
            const uint32_t cur_i64_c = s_i64_c[io_ring];
            const uint32_t nslots = Q5LI_V2_I32_FIELDS + cur_i64_c * Q5LI_V2_I64_FIELDS;
            if (sl < nslots) {
                const uint32_t comp_sz = s_comp_sz[io_ring][sl];
                const uint32_t slot = blockIdx.x * Q5LI_V2_SLOTS_PER_BLOCK
                                    + io_ring * SPP + sl;
                char* dst = d_decomp_buf + block_base
                          + (uint64_t)decomp_face * face_stride
                          + (uint64_t)sl * p.page_size;
                bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                    pc_base_addr, slot, dst, comp_sz, p.page_size, my_smem);
            }
        }

        __syncthreads();

        // Phase B: scan previous decompressed page
        if (prev_valid) {
            const uint32_t prev_ring = 1 - io_ring;  // same ring as previous iter's io_ring
            // face stride toggled opposite to decomp_face
            const uint64_t read_base = block_base
                                     + (uint64_t)(1 - decomp_face) * face_stride;
            q5li_v2_scan_page(d_decomp_buf + read_base, p.page_size, tid,
                              s_i64_c[prev_ring], s_i64_row_offset[prev_ring], p);
        }

        __syncthreads();

        prev_valid  = true;
        io_ring     = 1 - io_ring;
        decomp_face = 1 - decomp_face;
    }

    // Epilogue: scan last decompressed page
    if (prev_valid) {
        const uint32_t prev_ring = 1 - io_ring;
        const uint64_t read_base = block_base
                                 + (uint64_t)(1 - decomp_face) * face_stride;
        q5li_v2_scan_page(d_decomp_buf + read_base, p.page_size, tid,
                          s_i64_c[prev_ring], s_i64_row_offset[prev_ring], p);
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

struct BAMFusedQ5LIV2Context {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_q5li_v2_ctx_t bam_fused_q5li_v2_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ5LIV2Context();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * Q5LI_V2_SLOTS_PER_BLOCK;
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    size_t decomp_size = (size_t)num_blocks * Q5LI_V2_NFACE
                       * Q5LI_V2_SLOTS_PER_PAGE * page_size;
    FUSED_Q5LI_V2_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q5li_v2_ctx_t>(ctx);
}

void bam_fused_q5li_v2_run_async(
    bam_fused_q5li_v2_ctx_t ctx_handle,
    const BAMFusedQ5LIV2Params& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ5LIV2Context*>(ctx_handle);

    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * Q5LI_V2_WARPS;
        auto kernel_fn = bam_lz4_fused_q5li_v2_kernel<PS>;
        FUSED_Q5LI_V2_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<ctx->num_blocks, Q5LI_V2_THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, params);
    });

    FUSED_Q5LI_V2_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q5li_v2_destroy(bam_fused_q5li_v2_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ5LIV2Context*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

uint32_t q5li_v2_max_blocks(uint32_t page_size)
{
    int max_blocks_per_sm = 0;
    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * Q5LI_V2_WARPS;
        auto kfn = bam_lz4_fused_q5li_v2_kernel<PS>;
        cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kfn, Q5LI_V2_THREADS, smem_size);
    });
    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    uint32_t total = (uint32_t)max_blocks_per_sm * (uint32_t)sm_count;
    fprintf(stderr, "[q5li_v2] max_blocks_per_sm=%d sm_count=%d max_total=%u\n",
            max_blocks_per_sm, sm_count, total);
    return total;
}
