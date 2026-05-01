// bam_lz4_fused_q5_orders.cu — Fused BaM I/O + nvCOMPdx LZ4 + Q5 ORDERS probe+build
// Balanced pipeline: 1 IO warp + 6 decomp warps (224 threads/block).
// 1-page batch: IO warp reads 1 INT32 page's fields; 6 decomp warps each handle 1 field.
// __launch_bounds__(224, 4) -> 4 blocks/SM.
// NBUF=2 I/O ring; NFACE=2 decomp output; NMETA=3 metadata triple-buffer.
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q5_orders.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>

#define FUSED_Q5ORD_CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// Zone map compaction (parallel chunked version for < 1024 thread kernels)
#include "common/zonemap_compact.cuh"
static constexpr uint32_t kQ5OrdZonemapMaxPagesPerBlock = 2048;

// Configuration constants
static constexpr int FUSED_Q5ORD_NBUF        = 2;   // double-buffered I/O ring
static constexpr int FUSED_Q5ORD_NFACE       = 2;   // double-buffered decomp output
static constexpr int FUSED_Q5ORD_NMETA       = 3;   // triple-buffered metadata
static constexpr int FUSED_Q5ORD_MAX_FPP     = 7;   // max fields per page (1 INT32 + 3x2 INT64)
static constexpr int FUSED_Q5ORD_DECOMP_WARPS = 6;  // 1 warp per field (up to 6 fields/page)
static constexpr int FUSED_Q5ORD_WARPS       = 1 + FUSED_Q5ORD_DECOMP_WARPS;  // 7
static constexpr int FUSED_Q5ORD_MAX_FIELDS  = FUSED_Q5ORD_MAX_FPP;  // 7
static constexpr int FUSED_Q5ORD_MAX_I64     = 3;

// -- BaM nblk alignment fix --
__device__ static uint32_t fused_q5ord_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// Binary search on prefix sum: find page containing global row gid
__device__ static uint32_t fused_q5ord_ps_find_page(
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

// Hash function -- must match q5_hash64 used in q5_scan.cu for HT build
__device__ static uint32_t fused_q5ord_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

static constexpr uint64_t FUSED_Q5ORD_HT_EMPTY = UINT64_MAX;

// CUSTOMER HT probe (open addressing, key->value): custkey -> nation_idx
__device__ static int32_t fused_q5ord_ht_probe(
    const uint64_t *keys, const int32_t *values,
    uint32_t mask, uint64_t key)
{
    uint32_t slot = fused_q5ord_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return values[slot];
        if (k == FUSED_Q5ORD_HT_EMPTY) return -1;
        slot = (slot + 1) & mask;
    }
}

// ORDERS HT insert (key=orderkey, value=nation_idx)
__device__ static void fused_q5ord_ht_insert(
    uint64_t *keys, int32_t *values, uint32_t mask,
    uint64_t key, int32_t value)
{
    uint32_t slot = fused_q5ord_hash64(key) & mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)FUSED_Q5ORD_HT_EMPTY,
            (unsigned long long)key);
        if (prev == FUSED_Q5ORD_HT_EMPTY || prev == key) {
            values[slot] = value;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

// -- I/O parameter computation helpers --
__device__ static void fused_q5ord_io_params_i32(
    const BAMFusedQ5OrdParams& p,
    uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.i32_field_start_page_id + pg;
    dev = global_pg % ndev;
    if (p.is_compressed_i32) {
        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i32[pg] / 512;
        comp_sz = p.d_comp_sizes_i32[pg];
        nblk = fused_q5ord_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

__device__ static void fused_q5ord_io_params_i64(
    const BAMFusedQ5OrdParams& p,
    uint32_t fi, uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.i64_field_start_page_ids[fi] + pg;
    dev = global_pg % ndev;
    if (p.is_compressed_i64[fi]) {
        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i64[fi][pg] / 512;
        comp_sz = p.d_comp_sizes_i64[fi][pg];
        nblk = fused_q5ord_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

// -- Q5 ORDERS scan helper: filter date, probe CUSTOMER HT, build ORDERS HT --
__device__ __forceinline__ static void fused_q5ord_scan_page(
    const char* face_base,
    uint32_t page_size,
    uint32_t i64_count,
    uint64_t i64_row_offset,
    uint32_t tid,
    uint32_t num_threads,
    const BAMFusedQ5OrdParams& p)
{
    // Field layout: [0]=O_ORDERDATE(INT32), [1+k*2]=O_ORDERKEY(INT64 page k), [1+k*2+1]=O_CUSTKEY(INT64 page k)
    const char* odate_page = face_base;

    uint32_t nalloc = *(const uint32_t*)odate_page;
    const int32_t* od = (const int32_t*)(odate_page + 12);

    uint32_t i64_nalloc[FUSED_Q5ORD_MAX_I64];
    for (uint32_t k = 0; k < i64_count; k++) {
        const char* ok_page = face_base + (uint64_t)(1 + k * 2) * page_size;
        i64_nalloc[k] = *(const uint32_t*)ok_page;
    }

    for (uint32_t r = tid; r < nalloc; r += num_threads) {
        int32_t odate = od[r];
        if (odate < p.date_low || odate >= p.date_high) continue;

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

        // Read O_CUSTKEY from INT64 page
        const char* ck_page = face_base + (uint64_t)(1 + i64_pg_local * 2 + 1) * page_size;
        uint64_t custkey = *(const uint64_t*)(ck_page + 16 + (uint64_t)i64_rec * 8);

        // Probe CUSTOMER HT: custkey -> nation_idx
        int32_t nation_idx = fused_q5ord_ht_probe(
            p.d_ht_cust_keys, p.d_ht_cust_values, p.ht_cust_mask, custkey);
        if (nation_idx < 0) continue;

        // Read O_ORDERKEY from INT64 page
        const char* ok_page = face_base + (uint64_t)(1 + i64_pg_local * 2) * page_size;
        uint64_t orderkey = *(const uint64_t*)(ok_page + 16 + (uint64_t)i64_rec * 8);

        // Build ORDERS HT: orderkey -> nation_idx
        fused_q5ord_ht_insert(
            p.d_ht_ord_keys, p.d_ht_ord_values, p.ht_ord_mask,
            orderkey, nation_idx);
    }
}

// -- IO warp helper: read all fields for one INT32 page into ring --
__device__ static void fused_q5ord_io_read_one_page(
    void* ctrls, void* pc,
    const BAMFusedQ5OrdParams& p,
    uint32_t pg, uint32_t ndev,
    uint32_t i64_start_val, uint32_t i64_count_val,
    uint32_t ring, uint32_t ring_field_base,
    uint32_t slots_per_block,
    uint32_t* shared_comp_sz_ring)
{
    const uint32_t lane = threadIdx.x % 32;
    constexpr uint32_t MAX_FIELDS = FUSED_Q5ORD_MAX_FIELDS;

    // INT32 field: O_ORDERDATE (field 0)
    {
        uint64_t lba; uint32_t nblk, dev, comp_sz;
        fused_q5ord_io_params_i32(p, pg, ndev, lba, nblk, dev, comp_sz);
        if (lane == 0) {
            bam_io_read_page_device(ctrls, pc, lba, nblk,
                blockIdx.x * slots_per_block + ring * MAX_FIELDS
                    + ring_field_base + 0,
                dev);
            shared_comp_sz_ring[ring_field_base + 0] = comp_sz;
        }
        __syncwarp();
    }
    // INT64 fields: O_ORDERKEY + O_CUSTKEY for each INT64 page
    for (uint32_t k = 0; k < i64_count_val; k++) {
        for (uint32_t fi64 = 0; fi64 < 2; fi64++) {
            uint32_t field_idx = 1 + k * 2 + fi64;
            uint64_t lba; uint32_t nblk, dev, comp_sz;
            fused_q5ord_io_params_i64(p, fi64, i64_start_val + k,
                                      ndev, lba, nblk, dev, comp_sz);
            if (lane == 0) {
                bam_io_read_page_device(ctrls, pc, lba, nblk,
                    blockIdx.x * slots_per_block + ring * MAX_FIELDS
                        + ring_field_base + field_idx,
                    dev);
                shared_comp_sz_ring[ring_field_base + field_idx] = comp_sz;
            }
            __syncwarp();
        }
    }
}

// -- IO warp helper: compute INT64 metadata for one page --
__device__ static void fused_q5ord_compute_meta(
    const BAMFusedQ5OrdParams& p, uint32_t pg,
    bool& active, uint32_t& nfields,
    uint32_t& i64_start_val, uint32_t& i64_count_val,
    uint64_t& i64_row_offset_val)
{
    active = true;
    i64_start_val = 0;
    i64_count_val = 0;
    i64_row_offset_val = 0;
    nfields = 0;

    {
        uint64_t first_row = p.d_ps_i32[pg];
        uint64_t last_row  = p.d_ps_i32[pg + 1];
        if (last_row > first_row) last_row--;
        i64_start_val = fused_q5ord_ps_find_page(
            p.d_ps_i64, p.npages_i64 + 1, first_row);
        uint32_t i64_end = fused_q5ord_ps_find_page(
            p.d_ps_i64, p.npages_i64 + 1, last_row);
        i64_count_val = i64_end - i64_start_val + 1;
        if (i64_count_val > FUSED_Q5ORD_MAX_I64)
            i64_count_val = FUSED_Q5ORD_MAX_I64;
        i64_row_offset_val = first_row - p.d_ps_i64[i64_start_val];
        nfields = 1 + i64_count_val * 2;  // 1 INT32 + i64_count * 2 INT64
    }
}

// ================================================================
// Fused Q5 ORDERS kernel: 1-page balanced pipeline with double-buffered scan
//
// Pipeline:
//   Priming: IO reads first page -> ring[0], meta[0]
//   Main loop:
//     IO(N+1) || Decomp(N->face[f]) || Scan(N-1<-face[1-f])
//     __syncthreads__
//   Epilogue: scan last page
// ================================================================
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(224, 4)
void bam_lz4_fused_q5ord_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ5OrdParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr int      NBUF       = FUSED_Q5ORD_NBUF;
    constexpr int      NFACE      = FUSED_Q5ORD_NFACE;
    constexpr int      NMETA      = FUSED_Q5ORD_NMETA;
    constexpr uint32_t MAX_FIELDS = FUSED_Q5ORD_MAX_FIELDS;
    constexpr uint32_t WARPS      = FUSED_Q5ORD_WARPS;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;   // 0..6
    const uint32_t lane    = tid % 32;

    // -- Shared memory --
    __shared__ uint32_t shared_comp_sz[NBUF][MAX_FIELDS];
    __shared__ uint32_t s_nfields[NMETA];
    __shared__ uint32_t s_i64_count[NMETA];
    __shared__ uint64_t s_i64_row_offset[NMETA];
    __shared__ uint32_t s_i64_start[NMETA];
    __shared__ bool     s_active[NMETA];

    // Dynamic: nvCOMPdx per-warp region (IO warp doesn't need it)
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = (warp_id > 0)
        ? (smem + (warp_id - 1) * warp_smem)
        : nullptr;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // -- Slot assignment --
    constexpr uint32_t SLOTS_PER_BLOCK = NBUF * MAX_FIELDS;

    // -- Decomp buffer layout: [block][face][field] x page_size --
    const uint64_t face_stride  = (uint64_t)MAX_FIELDS * p.page_size;
    const uint64_t block_stride = (uint64_t)NFACE * face_stride;
    const uint64_t block_base   = (uint64_t)blockIdx.x * block_stride;

    // -- Zone map compaction --
    __shared__ uint32_t s_active_pgs[kQ5OrdZonemapMaxPagesPerBlock];
    __shared__ uint32_t s_num_active;
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
        zonemap_compact_block_pages_chunked(p.d_page_mask, p.total_pages,
                                            s_active_pgs, &s_num_active);
    }
    if (s_num_active == 0) return;

    // -- Pipeline state --
    int  io_ring     = 0;
    int  decomp_face = 0;
    int  meta_slot   = 0;
    bool prev_active = false;

    // ================================================================
    // Priming: IO warp reads first active page -> ring[0], meta[0]
    // ================================================================
    {
        if (warp_id == 0) {
            uint32_t pg = s_active_pgs[0];
            bool active; uint32_t nf, i64s, i64c; uint64_t i64ro;
            fused_q5ord_compute_meta(p, pg, active, nf, i64s, i64c, i64ro);
            if (lane == 0) {
                s_active[0] = active;
                s_nfields[0] = nf;
                s_i64_count[0] = i64c;
                s_i64_row_offset[0] = i64ro;
                s_i64_start[0] = i64s;
            }
            fused_q5ord_io_read_one_page(
                ctrls, pc, p, pg, ndev, i64s, i64c,
                0, 0, SLOTS_PER_BLOCK, shared_comp_sz[0]);
        }
    }
    __syncthreads();

    // ================================================================
    // Main loop: IO(N+1) || Decomp(N) || Scan(N-1)
    // ================================================================
    for (uint32_t j = 0; j < s_num_active; j++)
    {
        const int cur_meta  = meta_slot;
        const int next_meta = (meta_slot + 1) % NMETA;
        const int prev_meta = (meta_slot + 2) % NMETA;

        const bool active = s_active[cur_meta];

        const bool has_next = (j + 1 < s_num_active);

        // -- IO warp: read next active page --
        if (warp_id == 0 && has_next) {
            const int next_ring = 1 - io_ring;
            uint32_t next_pg = s_active_pgs[j + 1];
            bool nact; uint32_t nf, i64s, i64c; uint64_t i64ro;
            fused_q5ord_compute_meta(p, next_pg, nact, nf, i64s, i64c, i64ro);
            if (lane == 0) {
                s_active[next_meta] = nact;
                s_nfields[next_meta] = nf;
                s_i64_count[next_meta] = i64c;
                s_i64_row_offset[next_meta] = i64ro;
                s_i64_start[next_meta] = i64s;
            }
            fused_q5ord_io_read_one_page(
                ctrls, pc, p, next_pg, ndev, i64s, i64c,
                next_ring, 0, SLOTS_PER_BLOCK,
                shared_comp_sz[next_ring]);
        }

        // -- Decomp warps 1-6: decompress current page --
        if (warp_id >= 1 && active) {
            const uint32_t fi = warp_id - 1;  // 0..5
            const uint32_t nf = s_nfields[cur_meta];
            if (fi < nf) {
                const uint32_t comp_sz = shared_comp_sz[io_ring][fi];
                const uint32_t slot = blockIdx.x * SLOTS_PER_BLOCK
                                    + io_ring * MAX_FIELDS + fi;
                char* dst = d_decomp_buf + block_base
                          + (uint64_t)decomp_face * face_stride
                          + (uint64_t)fi * p.page_size;
                bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                    pc_base_addr, slot, dst, comp_sz, p.page_size, my_smem);
            }
            // Extra round for nf > 6 (i64_count=3, rare: nf=7)
            if (nf > 6 && fi < (nf - 6)) {
                const uint32_t fi2 = fi + 6;
                const uint32_t comp_sz2 = shared_comp_sz[io_ring][fi2];
                const uint32_t slot2 = blockIdx.x * SLOTS_PER_BLOCK
                                     + io_ring * MAX_FIELDS + fi2;
                char* dst2 = d_decomp_buf + block_base
                           + (uint64_t)decomp_face * face_stride
                           + (uint64_t)fi2 * p.page_size;
                bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                    pc_base_addr, slot2, dst2, comp_sz2, p.page_size, my_smem);
            }
        }

        // -- All warps: scan PREVIOUS page (reads face[1-decomp_face]) --
        if (prev_active) {
            const int scan_face = 1 - decomp_face;
            const char* scan_base = d_decomp_buf + block_base
                                  + (uint64_t)scan_face * face_stride;
            constexpr uint32_t SCAN_THREADS = WARPS * 32;
            const uint32_t i64c = s_i64_count[prev_meta];
            const uint64_t i64ro = s_i64_row_offset[prev_meta];
            fused_q5ord_scan_page(
                scan_base, p.page_size, i64c, i64ro,
                tid, SCAN_THREADS, p);
        }

        __syncthreads();

        prev_active = active;
        io_ring     = 1 - io_ring;
        decomp_face = 1 - decomp_face;
        meta_slot   = next_meta;
    }

    // ================================================================
    // Epilogue: scan the last page
    // ================================================================
    if (prev_active) {
        const int prev_meta = (meta_slot + 2) % NMETA;
        const int scan_face = 1 - decomp_face;
        const char* scan_base = d_decomp_buf + block_base
                              + (uint64_t)scan_face * face_stride;
        constexpr uint32_t SCAN_THREADS = WARPS * 32;
        const uint32_t i64c = s_i64_count[prev_meta];
        const uint64_t i64ro = s_i64_row_offset[prev_meta];
        fused_q5ord_scan_page(
            scan_base, p.page_size, i64c, i64ro,
            tid, SCAN_THREADS, p);
    }
}

// ================================================================
// Host API
// ================================================================

struct BAMFusedQ5OrdContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_q5ord_ctx_t bam_fused_q5ord_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ5OrdContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * FUSED_Q5ORD_NBUF * FUSED_Q5ORD_MAX_FIELDS;
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: NFACE x MAX_FIELDS pages per block
    size_t decomp_size = (size_t)num_blocks * FUSED_Q5ORD_NFACE
                       * FUSED_Q5ORD_MAX_FIELDS * page_size;
    FUSED_Q5ORD_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q5ord_ctx_t>(ctx);
}

static void bam_fused_q5ord_launch(
    BAMFusedQ5OrdContext* ctx,
    const BAMFusedQ5OrdParams& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = FUSED_Q5ORD_WARPS * 32;  // 224
    constexpr uint32_t DECOMP_WARPS = FUSED_Q5ORD_DECOMP_WARPS;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>()
                         * DECOMP_WARPS;
        auto kernel_fn = bam_lz4_fused_q5ord_kernel<PS>;
        FUSED_Q5ORD_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_Q5ORD_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q5ord_run_async(
    bam_fused_q5ord_ctx_t ctx_handle,
    const BAMFusedQ5OrdParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ5OrdContext*>(ctx_handle);
    bam_fused_q5ord_launch(ctx, params, stream);
}

void bam_fused_q5ord_destroy(bam_fused_q5ord_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ5OrdContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized Q5 ORDERS kernel
//
// 32 warps = 1024 threads per block:
//   Warps 0-3:   IO (4 warps)
//     Warp 0:    O_ORDERDATE INT32 reads for all BATCH pages
//     Warp 1:    INT64 metadata + O_ORDERKEY/O_CUSTKEY reads
//     Warps 2-3: idle
//   Warps 4-31:  Decomp (7 groups × 4 warps)
//     Group g processes page g of batch:
//       wig 0: O_ORDERDATE decomp
//       wig 1: O_ORDERKEY INT64[0] decomp
//       wig 2: O_CUSTKEY INT64[0] decomp
//       wig 3: O_ORDERKEY INT64[1] decomp (if i64_count>=2)
//       Round 2: wig 0→CUSTKEY[1], wig 1→ORDERKEY[2], wig 2→CUSTKEY[2]
//   All 1024 threads: scan (date filter + CUSTOMER HT probe + ORDERS HT build)
//
// Double-buffered: IO[batch N+1] || Decomp[batch N] → Scan[batch N]
//
// Slot layout per page in batch:
//   [0] O_ORDERDATE, [1] O_ORDERKEY INT64[0], [2] O_CUSTKEY INT64[0],
//   [3] O_ORDERKEY INT64[1], [4] O_CUSTKEY INT64[1],
//   [5] O_ORDERKEY INT64[2], [6] O_CUSTKEY INT64[2]
// ════════════════════════════════════════════════════════════════

static constexpr uint32_t Q5ORDWS_BATCH      = 7;
static constexpr uint32_t Q5ORDWS_N_BUF      = 2;
static constexpr uint32_t Q5ORDWS_I32_FIELDS = 1;
static constexpr uint32_t Q5ORDWS_I64_FIELDS = 2;
static constexpr uint32_t Q5ORDWS_MAX_I64    = 3;
static constexpr uint32_t Q5ORDWS_SLOTS_PER_PAGE  = Q5ORDWS_I32_FIELDS + Q5ORDWS_MAX_I64 * Q5ORDWS_I64_FIELDS;  // 7
static constexpr uint32_t Q5ORDWS_IO_WARPS   = 4;
static constexpr uint32_t Q5ORDWS_DECOMP_GROUPS = 7;
static constexpr uint32_t Q5ORDWS_WARPS_PER_GROUP = 4;
static constexpr uint32_t Q5ORDWS_SLOTS_PER_BUF   = Q5ORDWS_BATCH * Q5ORDWS_SLOTS_PER_PAGE;  // 49
static constexpr uint32_t Q5ORDWS_SLOTS_PER_BLOCK = Q5ORDWS_N_BUF * Q5ORDWS_SLOTS_PER_BUF;   // 98

// I/O parameter helpers (warp-spec versions)
__device__ static void q5ordws_io_params_i32(
    const Q5OrdWarpSpecParams& p,
    uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.i32_field_start_page_id + pg;
    dev = global_pg % ndev;
    if (p.is_compressed_i32) {
        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i32[pg] / 512;
        comp_sz = p.d_comp_sizes_i32[pg];
        nblk = fused_q5ord_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

__device__ static void q5ordws_io_params_i64(
    const Q5OrdWarpSpecParams& p,
    uint32_t fi, uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.i64_field_start_page_ids[fi] + pg;
    dev = global_pg % ndev;
    if (p.is_compressed_i64[fi]) {
        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i64[fi][pg] / 512;
        comp_sz = p.d_comp_sizes_i64[fi][pg];
        nblk = fused_q5ord_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(1024, 1)
void q5ord_warp_spec_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    Q5OrdWarpSpecParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t BATCH      = Q5ORDWS_BATCH;
    constexpr uint32_t N_BUF      = Q5ORDWS_N_BUF;
    constexpr uint32_t I32_FIELDS = Q5ORDWS_I32_FIELDS;
    constexpr uint32_t I64_FIELDS = Q5ORDWS_I64_FIELDS;
    constexpr uint32_t MAX_I64    = Q5ORDWS_MAX_I64;
    constexpr uint32_t SPP        = Q5ORDWS_SLOTS_PER_PAGE;
    constexpr uint32_t IO_WARPS   = Q5ORDWS_IO_WARPS;
    constexpr uint32_t SLOTS_PER_BUF   = Q5ORDWS_SLOTS_PER_BUF;
    constexpr uint32_t SLOTS_PER_BLOCK = Q5ORDWS_SLOTS_PER_BLOCK;
    constexpr uint32_t THREADS    = 1024;
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

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = (warp_id >= IO_WARPS)
        ? smem + (warp_id - IO_WARPS) * warp_smem : nullptr;

    __shared__ uint32_t s_active_pgs[kQ5OrdZonemapMaxPagesPerBlock];
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
        zonemap_compact_block_pages_chunked(p.d_page_mask, p.total_pages,
                                            s_active_pgs, &s_num_active);
    }
    const uint32_t my_pages = s_num_active;
    if (my_pages == 0) return;
    const uint32_t block_slot_base = blockIdx.x * SLOTS_PER_BLOCK;

    // ── IO helper: read one batch of pages ──
    auto io_read_batch = [&](uint32_t bstart, uint32_t bcount, uint32_t buf) {
        if (warp_id == 0) {
            // Warp 0: O_ORDERDATE INT32 reads for all pages in batch
            for (uint32_t j = 0; j < bcount; j++) {
                uint32_t orig_pg = s_active_pgs[bstart + j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                q5ordws_io_params_i32(p, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = block_slot_base + buf * SLOTS_PER_BUF
                              + j * SPP + 0;
                if (lane == 0) {
                    bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
                    s_comp_sz[buf][j][0] = comp_sz;
                }
                __syncwarp();
            }
        } else if (warp_id == 1) {
            // Warp 1: metadata computation + INT64 reads (ORDERKEY + CUSTKEY)
            for (uint32_t j = 0; j < bcount; j++) {
                uint32_t orig_pg = s_active_pgs[bstart + j];

                // Compute INT64 page range
                uint32_t i64_s, i64_c;
                uint64_t i64_ro;
                if (lane == 0) {
                    uint64_t first_row = p.d_ps_i32[orig_pg];
                    uint64_t last_row  = p.d_ps_i32[orig_pg + 1];
                    if (last_row > first_row) last_row--;
                    i64_s = fused_q5ord_ps_find_page(p.d_ps_i64, p.npages_i64 + 1, first_row);
                    uint32_t i64_e = fused_q5ord_ps_find_page(p.d_ps_i64, p.npages_i64 + 1, last_row);
                    i64_c = i64_e - i64_s + 1;
                    if (i64_c > MAX_I64) i64_c = MAX_I64;
                    i64_ro = first_row - p.d_ps_i64[i64_s];
                    s_i64_count[buf][j] = i64_c;
                    s_i64_start[buf][j] = i64_s;
                    s_i64_row_offset[buf][j] = i64_ro;
                }
                i64_c = __shfl_sync(0xFFFFFFFF, i64_c, 0);
                i64_s = __shfl_sync(0xFFFFFFFF, i64_s, 0);

                // Read INT64 pages: ORDERKEY + CUSTKEY for each INT64 page
                for (uint32_t k = 0; k < i64_c; k++) {
                    for (uint32_t fi64 = 0; fi64 < I64_FIELDS; fi64++) {
                        uint64_t lba; uint32_t nblk, dev, comp_sz;
                        q5ordws_io_params_i64(p, fi64, i64_s + k, ndev, lba, nblk, dev, comp_sz);
                        uint32_t field_idx = I32_FIELDS + k * I64_FIELDS + fi64;
                        uint32_t slot = block_slot_base + buf * SLOTS_PER_BUF
                                      + j * SPP + field_idx;
                        if (lane == 0) {
                            bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
                            s_comp_sz[buf][j][field_idx] = comp_sz;
                        }
                        __syncwarp();
                    }
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
        const uint32_t group = dw / Q5ORDWS_WARPS_PER_GROUP;   // 0..6
        const uint32_t wig   = dw % Q5ORDWS_WARPS_PER_GROUP;   // 0..3

        const uint32_t bcount = s_batch_count[buf];
        if (group >= bcount) return;

        const uint32_t i64_c = s_i64_count[buf][group];
        // nfields = 1 (INT32) + i64_c * 2 (INT64 pairs)
        const uint32_t nfields = I32_FIELDS + i64_c * I64_FIELDS;

        // Round 1: up to 4 parallel decomps (wig 0..3)
        if (wig < nfields) {
            uint32_t field_idx = wig;
            uint32_t slot = block_slot_base + buf * SLOTS_PER_BUF
                          + group * SPP + field_idx;
            uint32_t comp_sz = s_comp_sz[buf][group][field_idx];
            char* dst = d_decomp_buf + (uint64_t)slot * p.page_size;
            bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                pc_base_addr, slot, dst, comp_sz, p.page_size, my_smem);
        }

        // Round 2: overflow fields (nfields > 4)
        if (nfields > 4 && wig < (nfields - 4)) {
            uint32_t field_idx = 4 + wig;
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

            // O_ORDERDATE at slot offset 0
            const char* odate_page = d_decomp_buf + base;
            uint32_t nalloc = *(const uint32_t*)odate_page;
            const int32_t* od = (const int32_t*)(odate_page + 12);  // 12B header

            uint32_t i64_c  = s_i64_count[buf][j];
            uint64_t i64_ro = s_i64_row_offset[buf][j];

            // Read nalloc of each INT64 page (ORDERKEY pages)
            uint32_t i64_nalloc[MAX_I64];
            for (uint32_t k = 0; k < i64_c; k++) {
                const char* ok_page = d_decomp_buf + base
                    + (uint64_t)(I32_FIELDS + k * I64_FIELDS) * p.page_size;
                i64_nalloc[k] = *(const uint32_t*)ok_page;
            }

            for (uint32_t r = tid; r < nalloc; r += THREADS) {
                int32_t odate = od[r];
                if (odate < p.date_low || odate >= p.date_high) continue;

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

                // Read O_CUSTKEY
                const char* ck_page = d_decomp_buf + base
                    + (uint64_t)(I32_FIELDS + i64_pg_local * I64_FIELDS + 1) * p.page_size;
                uint64_t custkey = *(const uint64_t*)(ck_page + 16 + (uint64_t)i64_rec * 8);

                // Probe CUSTOMER HT
                int32_t nation_idx = fused_q5ord_ht_probe(
                    p.d_ht_cust_keys, p.d_ht_cust_values, p.ht_cust_mask, custkey);
                if (nation_idx < 0) continue;

                // Read O_ORDERKEY
                const char* ok_page = d_decomp_buf + base
                    + (uint64_t)(I32_FIELDS + i64_pg_local * I64_FIELDS) * p.page_size;
                uint64_t orderkey = *(const uint64_t*)(ok_page + 16 + (uint64_t)i64_rec * 8);

                // Build ORDERS HT
                fused_q5ord_ht_insert(
                    p.d_ht_ord_keys, p.d_ht_ord_values, p.ht_ord_mask,
                    orderkey, nation_idx);
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

        if (warp_id < IO_WARPS) {
            io_read_batch(bstart, cur_count, cur_buf);
        } else {
            decomp_batch(prev_buf);
        }
        __syncthreads();

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

uint32_t q5ord_warp_spec_max_blocks(uint32_t page_size)
{
    int max_blocks_per_sm = 0;
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = Q5ORDWS_DECOMP_GROUPS * Q5ORDWS_WARPS_PER_GROUP;  // 28

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = q5ord_warp_spec_kernel<PS>;
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
    fprintf(stderr, "[q5ord_warp_spec] max_blocks_per_sm=%d sm_count=%d max_total=%u\n",
            max_blocks_per_sm, sm_count, total);
    return total;
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized: launch function
// ════════════════════════════════════════════════════════════════

void q5ord_warp_spec_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const Q5OrdWarpSpecParams& params,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = Q5ORDWS_DECOMP_GROUPS * Q5ORDWS_WARPS_PER_GROUP;  // 28

    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = q5ord_warp_spec_kernel<PS>;
        FUSED_Q5ORD_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kfn<<<num_blocks, THREADS, smem_size, stream>>>(
            d_ctrls, d_pc_ptr, pc_base_addr, d_decomp_buf, params);
    });
    FUSED_Q5ORD_CUDA_CHECK(cudaGetLastError());
}
