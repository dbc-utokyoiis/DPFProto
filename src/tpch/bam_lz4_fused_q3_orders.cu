// bam_lz4_fused_q3_orders.cu — Fused BaM I/O + nvCOMPdx LZ4 + Q3 ORDERS probe+build
// Balanced pipeline: 1 IO warp + 6 decomp warps (224 threads/block).
// 1-page batch: IO warp reads 1 INT32 page's fields; 6 decomp warps each handle 1 field.
// __launch_bounds__(224, 4) → 4 blocks/SM.
// NBUF=2 I/O ring; NFACE=2 decomp output; NMETA=3 metadata triple-buffer.
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q3_orders.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>

#define FUSED_Q3ORD_CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// Zone map compaction (parallel chunked version for < 1024 thread kernels)
#include "common/zonemap_compact.cuh"
static constexpr uint32_t kQ3OrdZonemapMaxPagesPerBlock = 2048;

// Configuration constants
static constexpr int FUSED_Q3ORD_NBUF        = 2;   // double-buffered I/O ring
static constexpr int FUSED_Q3ORD_NFACE       = 2;   // double-buffered decomp output
static constexpr int FUSED_Q3ORD_NMETA       = 3;   // triple-buffered metadata
static constexpr int FUSED_Q3ORD_MAX_FPP     = 8;   // max fields per page (2 INT32 + 3×2 INT64)
static constexpr int FUSED_Q3ORD_DECOMP_WARPS = 6;   // 1 warp per field (up to 6 fields/page)
static constexpr int FUSED_Q3ORD_WARPS       = 1 + FUSED_Q3ORD_DECOMP_WARPS;  // 7
static constexpr int FUSED_Q3ORD_MAX_FIELDS  = FUSED_Q3ORD_MAX_FPP;  // 8
static constexpr int FUSED_Q3ORD_MAX_I64     = 3;

// ── BaM nblk alignment fix ──
__device__ static uint32_t fused_q3ord_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// Binary search on prefix sum: find page containing global row gid
__device__ static uint32_t fused_q3ord_ps_find_page(
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

// Hash function — must match q3_hash64 used in q3_scan.cu
__device__ static uint32_t fused_q3ord_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

static constexpr uint64_t FUSED_Q3ORD_HT_EMPTY = UINT64_MAX;

// CUSTOMER hash set probe
__device__ static bool fused_q3ord_hashset_probe(
    const uint64_t *keys, uint32_t mask, uint64_t key)
{
    uint32_t slot = fused_q3ord_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return true;
        if (k == FUSED_Q3ORD_HT_EMPTY) return false;
        slot = (slot + 1) & mask;
    }
}

// ORDERS HT insert (key + payload)
__device__ static void fused_q3ord_ht_insert_kv(
    uint64_t *keys, uint64_t *payloads, uint32_t mask,
    uint64_t key, uint64_t payload)
{
    uint32_t slot = fused_q3ord_hash64(key) & mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)FUSED_Q3ORD_HT_EMPTY,
            (unsigned long long)key);
        if (prev == FUSED_Q3ORD_HT_EMPTY || prev == key) {
            payloads[slot] = payload;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

// ── I/O parameter computation helpers ──
__device__ static void fused_q3ord_io_params_i32(
    const BAMFusedQ3OrdParams& p,
    uint32_t fi, uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.i32_field_start_page_ids[fi] + pg;
    dev = global_pg % ndev;
    if (p.is_compressed_i32[fi]) {
        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i32[fi][pg] / 512;
        comp_sz = p.d_comp_sizes_i32[fi][pg];
        nblk = fused_q3ord_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

__device__ static void fused_q3ord_io_params_i64(
    const BAMFusedQ3OrdParams& p,
    uint32_t fi, uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.i64_field_start_page_ids[fi] + pg;
    dev = global_pg % ndev;
    if (p.is_compressed_i64[fi]) {
        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i64[fi][pg] / 512;
        comp_sz = p.d_comp_sizes_i64[fi][pg];
        nblk = fused_q3ord_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

// ── Q3 ORDERS scan helper: probe CUSTOMER hash set + build ORDERS HT ──
__device__ __forceinline__ static void fused_q3ord_scan_page(
    const char* face_base,
    uint32_t page_size,
    uint32_t i64_count,
    uint64_t i64_row_offset,
    uint32_t tid,
    uint32_t num_threads,
    const BAMFusedQ3OrdParams& p)
{
    const char* odate_page    = face_base;
    const char* shipprio_page = face_base + (uint64_t)page_size;

    uint32_t nalloc = *(const uint32_t*)odate_page;
    const int32_t* od = (const int32_t*)(odate_page + 12);
    const int32_t* sp = (const int32_t*)(shipprio_page + 12);

    uint32_t i64_nalloc[FUSED_Q3ORD_MAX_I64];
    for (uint32_t k = 0; k < i64_count; k++) {
        const char* ok_page = face_base + (uint64_t)(2 + k * 2) * page_size;
        i64_nalloc[k] = *(const uint32_t*)ok_page;
    }

    for (uint32_t r = tid; r < nalloc; r += num_threads) {
        int32_t odate = od[r];
        if (!p.skip_date_filter && odate >= 19950315) continue;

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

        const char* ck_page = face_base + (uint64_t)(3 + i64_pg_local * 2) * page_size;
        uint64_t custkey = *(const uint64_t*)(ck_page + 16 + (uint64_t)i64_rec * 8);

        if (!fused_q3ord_hashset_probe(p.d_custkey_set, p.custkey_set_mask, custkey))
            continue;

        const char* ok_page = face_base + (uint64_t)(2 + i64_pg_local * 2) * page_size;
        uint64_t orderkey = *(const uint64_t*)(ok_page + 16 + (uint64_t)i64_rec * 8);

        int32_t shippriority = sp[r];

        uint64_t payload = ((uint64_t)(uint32_t)odate << 32) | (uint64_t)(uint32_t)shippriority;
        fused_q3ord_ht_insert_kv(
            p.d_orders_ht_keys, p.d_orders_ht_payloads, p.orders_ht_mask,
            orderkey, payload);
    }
}

// ── IO warp helper: read all fields for one INT32 page into ring ──
// ring_field_base = batch_idx * MAX_FPP (0 for page A, 8 for page B)
__device__ static void fused_q3ord_io_read_one_page(
    void* ctrls, void* pc,
    const BAMFusedQ3OrdParams& p,
    uint32_t pg, uint32_t ndev,
    uint32_t i64_start_val, uint32_t i64_count_val,
    uint32_t ring, uint32_t ring_field_base,
    uint32_t slots_per_block,
    uint32_t* shared_comp_sz_ring)  // shared_comp_sz[ring]
{
    const uint32_t lane = threadIdx.x % 32;
    constexpr uint32_t MAX_FPP = FUSED_Q3ORD_MAX_FPP;

    // INT32 fields
    for (uint32_t fi = 0; fi < 2; fi++) {
        uint64_t lba; uint32_t nblk, dev, comp_sz;
        fused_q3ord_io_params_i32(p, fi, pg, ndev, lba, nblk, dev, comp_sz);
        if (lane == 0) {
            bam_io_read_page_device(ctrls, pc, lba, nblk,
                blockIdx.x * slots_per_block + ring * FUSED_Q3ORD_MAX_FIELDS
                    + ring_field_base + fi,
                dev);
            shared_comp_sz_ring[ring_field_base + fi] = comp_sz;
        }
        __syncwarp();
    }
    // INT64 fields
    for (uint32_t k = 0; k < i64_count_val; k++) {
        for (uint32_t fi64 = 0; fi64 < 2; fi64++) {
            uint32_t field_idx = 2 + k * 2 + fi64;
            uint64_t lba; uint32_t nblk, dev, comp_sz;
            fused_q3ord_io_params_i64(p, fi64, i64_start_val + k,
                                      ndev, lba, nblk, dev, comp_sz);
            if (lane == 0) {
                bam_io_read_page_device(ctrls, pc, lba, nblk,
                    blockIdx.x * slots_per_block + ring * FUSED_Q3ORD_MAX_FIELDS
                        + ring_field_base + field_idx,
                    dev);
                shared_comp_sz_ring[ring_field_base + field_idx] = comp_sz;
            }
            __syncwarp();
        }
    }
}

// ── IO warp helper: compute INT64 metadata for one page ──
__device__ static void fused_q3ord_compute_meta(
    const BAMFusedQ3OrdParams& p, uint32_t pg,
    bool& active, uint32_t& nfields,
    uint32_t& i64_start_val, uint32_t& i64_count_val,
    uint64_t& i64_row_offset_val)
{
    active = true;  // caller iterates only over active pages
    i64_start_val = 0;
    i64_count_val = 0;
    i64_row_offset_val = 0;
    nfields = 0;

    {
        uint64_t first_row = p.d_ps_i32[pg];
        uint64_t last_row  = p.d_ps_i32[pg + 1];
        if (last_row > first_row) last_row--;
        i64_start_val = fused_q3ord_ps_find_page(
            p.d_ps_i64, p.npages_i64 + 1, first_row);
        uint32_t i64_end = fused_q3ord_ps_find_page(
            p.d_ps_i64, p.npages_i64 + 1, last_row);
        i64_count_val = i64_end - i64_start_val + 1;
        if (i64_count_val > FUSED_Q3ORD_MAX_I64)
            i64_count_val = FUSED_Q3ORD_MAX_I64;
        i64_row_offset_val = first_row - p.d_ps_i64[i64_start_val];
        nfields = 2 + i64_count_val * 2;
    }
}

// ════════════════════════════════════════════════════════════════
// Fused Q3 ORDERS kernel: 1-page balanced pipeline with double-buffered scan
//
// 1 IO warp reads 1 INT32 page's fields (~6 reads).
// 6 decomp warps each decompress 1 field in parallel.
// All 7 warps scan the PREVIOUS page after their primary task.
//
// Double buffering: decomp writes face[f], scan reads face[1-f].
// Only 1 __syncthreads__ per iteration.
//
// Pipeline:
//   Priming: IO reads first page → ring[0], meta[0]
//   Main loop:
//     IO(N+1) ∥ Decomp(N→face[f]) ∥ Scan(N-1←face[1-f])
//     __syncthreads__
//   Epilogue: scan last page
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(224, 4)
void bam_lz4_fused_q3ord_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ3OrdParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr int      NBUF       = FUSED_Q3ORD_NBUF;
    constexpr int      NFACE      = FUSED_Q3ORD_NFACE;
    constexpr int      NMETA      = FUSED_Q3ORD_NMETA;
    constexpr uint32_t MAX_FPP    = FUSED_Q3ORD_MAX_FPP;
    constexpr uint32_t MAX_FIELDS = FUSED_Q3ORD_MAX_FIELDS;
    constexpr uint32_t WARPS      = FUSED_Q3ORD_WARPS;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;   // 0..6
    const uint32_t lane    = tid % 32;

    // ── Shared memory ──
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

    // ── Slot assignment ──
    constexpr uint32_t SLOTS_PER_BLOCK = NBUF * MAX_FIELDS;

    // ── Decomp buffer layout: [block][face][field] × page_size ──
    const uint64_t face_stride  = (uint64_t)MAX_FIELDS * p.page_size;
    const uint64_t block_stride = (uint64_t)NFACE * face_stride;
    const uint64_t block_base   = (uint64_t)blockIdx.x * block_stride;

    // ── Zone map compaction ──
    __shared__ uint32_t s_active_pgs[kQ3OrdZonemapMaxPagesPerBlock];
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

    // ── Pipeline state ──
    int  io_ring     = 0;
    int  decomp_face = 0;
    int  meta_slot   = 0;
    bool prev_active = false;

    // ════════════════════════════════════════════════════════════
    // Priming: IO warp reads first active page → ring[0], meta[0]
    // ════════════════════════════════════════════════════════════
    {
        if (warp_id == 0) {
            uint32_t pg = s_active_pgs[0];
            bool active; uint32_t nf, i64s, i64c; uint64_t i64ro;
            fused_q3ord_compute_meta(p, pg, active, nf, i64s, i64c, i64ro);
            if (lane == 0) {
                s_active[0] = active;
                s_nfields[0] = nf;
                s_i64_count[0] = i64c;
                s_i64_row_offset[0] = i64ro;
                s_i64_start[0] = i64s;
            }
            fused_q3ord_io_read_one_page(
                ctrls, pc, p, pg, ndev, i64s, i64c,
                0, 0, SLOTS_PER_BLOCK, shared_comp_sz[0]);
        }
    }
    __syncthreads();

    // ════════════════════════════════════════════════════════════
    // Main loop: iterate only over active pages
    // IO(N+1) ∥ Decomp(N) ∥ Scan(N-1) — single sync per iteration
    // Decomp writes face[decomp_face], Scan reads face[1-decomp_face]
    // ════════════════════════════════════════════════════════════
    for (uint32_t j = 0; j < s_num_active; j++)
    {
        const int cur_meta  = meta_slot;
        const int next_meta = (meta_slot + 1) % NMETA;
        const int prev_meta = (meta_slot + 2) % NMETA;

        const bool active = s_active[cur_meta];

        const bool has_next = (j + 1 < s_num_active);

        // ── IO warp: read next active page ──
        if (warp_id == 0 && has_next) {
            const int next_ring = 1 - io_ring;
            uint32_t next_pg = s_active_pgs[j + 1];
            bool nact; uint32_t nf, i64s, i64c; uint64_t i64ro;
            fused_q3ord_compute_meta(p, next_pg, nact, nf, i64s, i64c, i64ro);
            if (lane == 0) {
                s_active[next_meta] = nact;
                s_nfields[next_meta] = nf;
                s_i64_count[next_meta] = i64c;
                s_i64_row_offset[next_meta] = i64ro;
                s_i64_start[next_meta] = i64s;
            }
            fused_q3ord_io_read_one_page(
                ctrls, pc, p, next_pg, ndev, i64s, i64c,
                next_ring, 0, SLOTS_PER_BLOCK,
                shared_comp_sz[next_ring]);
        }

        // ── Decomp warps 1-6: decompress current page ──
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
            // Extra round for nf > 6 (i64_count=3, rare)
            if (fi < (nf - 6) && nf > 6) {
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

        // ── All warps: scan PREVIOUS page (reads face[1-decomp_face]) ──
        if (prev_active) {
            const int scan_face = 1 - decomp_face;
            const char* scan_base = d_decomp_buf + block_base
                                  + (uint64_t)scan_face * face_stride;
            constexpr uint32_t SCAN_THREADS = WARPS * 32;
            const uint32_t i64c = s_i64_count[prev_meta];
            const uint64_t i64ro = s_i64_row_offset[prev_meta];
            fused_q3ord_scan_page(
                scan_base, p.page_size, i64c, i64ro,
                tid, SCAN_THREADS, p);
        }

        __syncthreads();  // single sync: IO + decomp + scan all done

        prev_active = active;
        io_ring     = 1 - io_ring;
        decomp_face = 1 - decomp_face;
        meta_slot   = next_meta;
    }

    // ════════════════════════════════════════════════════════════
    // Epilogue: scan the last page
    // ════════════════════════════════════════════════════════════
    if (prev_active) {
        const int prev_meta = (meta_slot + 2) % NMETA;
        const int scan_face = 1 - decomp_face;
        const char* scan_base = d_decomp_buf + block_base
                              + (uint64_t)scan_face * face_stride;
        constexpr uint32_t SCAN_THREADS = WARPS * 32;
        const uint32_t i64c = s_i64_count[prev_meta];
        const uint64_t i64ro = s_i64_row_offset[prev_meta];
        fused_q3ord_scan_page(
            scan_base, p.page_size, i64c, i64ro,
            tid, SCAN_THREADS, p);
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

struct BAMFusedQ3OrdContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
    bool        owns_resources;  // true = we allocated io_pc + d_decomp_buf
};

bam_fused_q3ord_ctx_t bam_fused_q3ord_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ3OrdContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;
    ctx->owns_resources = true;

    // 2-page batch: NBUF × BATCH × MAX_FPP = 2 × 2 × 8 = 32 slots per block
    const uint32_t num_slots = num_blocks * FUSED_Q3ORD_NBUF * FUSED_Q3ORD_MAX_FIELDS;
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: NFACE × MAX_FIELDS pages per block
    size_t decomp_size = (size_t)num_blocks * FUSED_Q3ORD_NFACE
                       * FUSED_Q3ORD_MAX_FIELDS * page_size;
    FUSED_Q3ORD_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q3ord_ctx_t>(ctx);
}

static void bam_fused_q3ord_launch(
    BAMFusedQ3OrdContext* ctx,
    const BAMFusedQ3OrdParams& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = FUSED_Q3ORD_WARPS * 32;  // 224
    // Only 6 decomp warps need nvCOMPdx smem (IO warp doesn't)
    constexpr uint32_t DECOMP_WARPS = FUSED_Q3ORD_DECOMP_WARPS;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>()
                         * DECOMP_WARPS;
        auto kernel_fn = bam_lz4_fused_q3ord_kernel<PS>;
        FUSED_Q3ORD_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_Q3ORD_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q3ord_run_async(
    bam_fused_q3ord_ctx_t ctx_handle,
    const BAMFusedQ3OrdParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ3OrdContext*>(ctx_handle);
    bam_fused_q3ord_launch(ctx, params, stream);
}

void bam_fused_q3ord_destroy(bam_fused_q3ord_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ3OrdContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->owns_resources) {
        if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
        bam_io_page_cache_destroy(ctx->io_pc);
    }
    delete ctx;
}

bam_fused_q3ord_ctx_t bam_fused_q3ord_create_shared(
    bam_io_page_cache_t shared_pc,
    char* shared_decomp_buf,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ3OrdContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;
    ctx->owns_resources = false;
    ctx->io_pc = shared_pc;
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(shared_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(shared_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(shared_pc);
    ctx->d_decomp_buf = shared_decomp_buf;
    return static_cast<bam_fused_q3ord_ctx_t>(ctx);
}
