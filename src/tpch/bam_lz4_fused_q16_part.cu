// bam_lz4_fused_q16_part.cu — Fused BaM I/O + nvCOMPdx LZ4 + Q16 PART HT build
// Balanced pipeline: 1 IO warp + 6 decomp warps (224 threads/block).
// Per P_SIZE page: reads 1 INT32 page + up to 4 P_BRAND (CHAR) + up to 3 P_PARTKEY (INT64).
// __launch_bounds__(224, 4) → 4 blocks/SM.
// NBUF=2 I/O ring; NFACE=2 decomp output; NMETA=3 metadata triple-buffer.
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q16_part.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>

#define FUSED_Q16PART_CUDA_CHECK(call) do {                                    \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                           \
                cudaGetErrorString(err), __FILE__, __LINE__);                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// Configuration constants
static constexpr int FUSED_Q16PART_NBUF          = 2;   // double-buffered I/O ring
static constexpr int FUSED_Q16PART_NFACE         = 2;   // double-buffered decomp output
static constexpr int FUSED_Q16PART_NMETA         = 3;   // triple-buffered metadata
static constexpr int FUSED_Q16PART_MAX_BRAND     = 4;   // max P_BRAND pages per P_SIZE page
static constexpr int FUSED_Q16PART_MAX_PK        = 3;   // max P_PARTKEY pages per P_SIZE page
static constexpr int FUSED_Q16PART_MAX_FPP       = 1 + FUSED_Q16PART_MAX_BRAND + FUSED_Q16PART_MAX_PK;  // 8
static constexpr int FUSED_Q16PART_DECOMP_WARPS  = 6;
static constexpr int FUSED_Q16PART_WARPS         = 1 + FUSED_Q16PART_DECOMP_WARPS;  // 7
static constexpr int FUSED_Q16PART_MAX_FIELDS    = FUSED_Q16PART_MAX_FPP;

// ── BaM nblk alignment fix ──
__device__ static uint32_t fused_q16part_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// ── Binary search on prefix sum ──
__device__ static uint32_t fused_q16part_ps_find(
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

// ── Hash function (matches q16_scan.cu) ──
__device__ static uint32_t fused_q16part_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

static constexpr uint64_t FUSED_Q16PART_HT_EMPTY = UINT64_MAX;

// ── PART HT insert (key + partial_gid + row_idx) ──
__device__ static void fused_q16part_ht_insert(
    uint64_t *keys, uint32_t *group_ids, uint32_t *row_idxs,
    uint32_t mask, uint64_t key, uint32_t partial_gid, uint32_t row_idx)
{
    uint32_t slot = fused_q16part_hash64(key) & mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)FUSED_Q16PART_HT_EMPTY,
            (unsigned long long)key);
        if (prev == FUSED_Q16PART_HT_EMPTY || prev == key) {
            group_ids[slot] = partial_gid;
            row_idxs[slot] = row_idx;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

// ── IO parameter computation ──
__device__ static void fused_q16part_io_psize(
    const BAMFusedQ16PartParams& p, uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.psize_start_page_id + pg;
    dev = global_pg % ndev;
    if (p.psize_is_compressed) {
        lba = p.partition_start_lbas[dev] + p.d_psize_comp_offsets[pg] / 512;
        comp_sz = p.d_psize_comp_sizes[pg];
        nblk = fused_q16part_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

__device__ static void fused_q16part_io_brand(
    const BAMFusedQ16PartParams& p, uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.brand_start_page_id + pg;
    dev = global_pg % ndev;
    if (p.brand_is_compressed) {
        lba = p.partition_start_lbas[dev] + p.d_brand_comp_offsets[pg] / 512;
        comp_sz = p.d_brand_comp_sizes[pg];
        nblk = fused_q16part_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

__device__ static void fused_q16part_io_pk(
    const BAMFusedQ16PartParams& p, uint32_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.pk_start_page_id + pg;
    dev = global_pg % ndev;
    if (p.pk_is_compressed) {
        lba = p.partition_start_lbas[dev] + p.d_pk_comp_offsets[pg] / 512;
        comp_sz = p.d_pk_comp_sizes[pg];
        nblk = fused_q16part_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

// ── Scan helper: process one P_SIZE page and build HT ──
__device__ __forceinline__ static void fused_q16part_scan_page(
    const char* face_base,
    uint32_t page_size,
    uint32_t brand_count,
    uint64_t brand_row_offset,
    uint32_t brand_start_arg,
    uint32_t pk_count,
    uint64_t pk_row_offset,
    uint32_t pk_start_arg,
    uint64_t psize_row_base,
    uint32_t psize_nalloc,
    uint32_t tid,
    uint32_t num_threads,
    const BAMFusedQ16PartParams& p)
{
    constexpr int MAX_BRAND = FUSED_Q16PART_MAX_BRAND;
    constexpr int MAX_PK    = FUSED_Q16PART_MAX_PK;

    // P_SIZE page at face[0] — use prefix-sum nalloc for consistency
    const char* psize_page = face_base;
    uint32_t nalloc = psize_nalloc;
    const int32_t* size_vals = (const int32_t*)(psize_page + 12);

    // Compute P_BRAND page nallocs from prefix sum (avoids decompressed header mismatch)
    uint32_t brand_nalloc[MAX_BRAND];
    for (uint32_t k = 0; k < brand_count && k < MAX_BRAND; k++) {
        uint32_t abs_pg = brand_start_arg + k;
        brand_nalloc[k] = (uint32_t)(p.d_ps_brand[abs_pg + 1] - p.d_ps_brand[abs_pg]);
    }

    // Compute P_PARTKEY page nallocs from prefix sum
    uint32_t pk_nalloc[MAX_PK];
    for (uint32_t k = 0; k < pk_count && k < MAX_PK; k++) {
        uint32_t abs_pg = pk_start_arg + k;
        pk_nalloc[k] = (uint32_t)(p.d_ps_pk[abs_pg + 1] - p.d_ps_pk[abs_pg]);
    }

    for (uint32_t r = tid; r < nalloc; r += num_threads) {
        // 1. Read and check P_SIZE
        uint32_t size_val = (uint32_t)size_vals[r];
        if (size_val >= 64 || !((p.p_size_bitmask >> size_val) & 1)) continue;

        if (p.d_dbg_total_scanned) atomicAdd(p.d_dbg_total_scanned, 1);

        // 2. Map to P_BRAND page via cumulative nalloc
        uint64_t brand_local_row = brand_row_offset + r;
        uint32_t brand_pg_local = 0;
        uint64_t brand_cumul = 0;
        for (uint32_t k = 0; k < brand_count; k++) {
            if (brand_local_row < brand_cumul + brand_nalloc[k]) {
                brand_pg_local = k;
                break;
            }
            brand_cumul += brand_nalloc[k];
            brand_pg_local = k + 1;
        }
        if (brand_pg_local >= brand_count) {
            if (p.d_dbg_brand_overflow) atomicAdd(p.d_dbg_brand_overflow, 1);
            continue;
        }
        uint32_t brand_rec = (uint32_t)(brand_local_row - brand_cumul);

        // 3. Read brand and check
        const char* brand_page = face_base + (uint64_t)(1 + brand_pg_local) * page_size;
        const char* brand_str = brand_page + 12 + (uint64_t)p.brand_padded_len * brand_rec;
        uint32_t d1 = brand_str[6] - '1';
        uint32_t d2 = brand_str[7] - '1';
        uint32_t brand_id = d1 * 5 + d2;
        if (brand_id == p.brand_exclude_id) continue;

        // 4. Map to P_PARTKEY page via cumulative nalloc
        uint64_t pk_local_row = pk_row_offset + r;
        uint32_t pk_pg_local = 0;
        uint64_t pk_cumul = 0;
        for (uint32_t k = 0; k < pk_count; k++) {
            if (pk_local_row < pk_cumul + pk_nalloc[k]) {
                pk_pg_local = k;
                break;
            }
            pk_cumul += pk_nalloc[k];
            pk_pg_local = k + 1;
        }
        if (pk_pg_local >= pk_count) {
            if (p.d_dbg_pk_overflow) atomicAdd(p.d_dbg_pk_overflow, 1);
            continue;
        }
        uint32_t pk_rec = (uint32_t)(pk_local_row - pk_cumul);

        // 5. Read P_PARTKEY
        const char* pk_page = face_base + (uint64_t)(1 + brand_count + pk_pg_local) * page_size;
        uint64_t partkey = *(const uint64_t*)(pk_page + 16 + (uint64_t)pk_rec * 8);

        // 6. Insert into HT with partial_gid (type_id applied later in Stage 2)
        uint32_t partial_gid = (brand_id << 8) | (size_val - 1);
        uint32_t row_idx = (uint32_t)(psize_row_base + r);
        fused_q16part_ht_insert(
            p.d_ht_keys, p.d_ht_group_ids, p.d_ht_row_idx,
            p.ht_mask, partkey, partial_gid, row_idx);

        if (p.d_dbg_ht_inserted) atomicAdd(p.d_dbg_ht_inserted, 1);
    }
}

// ── IO warp: read all fields for one P_SIZE page into ring ──
__device__ static void fused_q16part_io_read_one_page(
    void* ctrls, void* pc,
    const BAMFusedQ16PartParams& p,
    uint32_t pg, uint32_t ndev,
    uint32_t brand_start_val, uint32_t brand_count_val,
    uint32_t pk_start_val, uint32_t pk_count_val,
    uint32_t ring,
    uint32_t slots_per_block,
    uint32_t* shared_comp_sz_ring)
{
    const uint32_t lane = threadIdx.x % 32;

    // P_SIZE page at fi=0
    {
        uint64_t lba; uint32_t nblk, dev, comp_sz;
        fused_q16part_io_psize(p, pg, ndev, lba, nblk, dev, comp_sz);
        if (lane == 0) {
            bam_io_read_page_device(ctrls, pc, lba, nblk,
                blockIdx.x * slots_per_block + ring * FUSED_Q16PART_MAX_FIELDS + 0,
                dev);
            shared_comp_sz_ring[0] = comp_sz;
        }
        __syncwarp();
    }

    // P_BRAND pages at fi=1..brand_count
    for (uint32_t k = 0; k < brand_count_val; k++) {
        uint32_t fi = 1 + k;
        uint64_t lba; uint32_t nblk, dev, comp_sz;
        fused_q16part_io_brand(p, brand_start_val + k, ndev, lba, nblk, dev, comp_sz);
        if (lane == 0) {
            bam_io_read_page_device(ctrls, pc, lba, nblk,
                blockIdx.x * slots_per_block + ring * FUSED_Q16PART_MAX_FIELDS + fi,
                dev);
            shared_comp_sz_ring[fi] = comp_sz;
        }
        __syncwarp();
    }

    // P_PARTKEY pages at fi=1+brand_count..
    for (uint32_t k = 0; k < pk_count_val; k++) {
        uint32_t fi = 1 + brand_count_val + k;
        uint64_t lba; uint32_t nblk, dev, comp_sz;
        fused_q16part_io_pk(p, pk_start_val + k, ndev, lba, nblk, dev, comp_sz);
        if (lane == 0) {
            bam_io_read_page_device(ctrls, pc, lba, nblk,
                blockIdx.x * slots_per_block + ring * FUSED_Q16PART_MAX_FIELDS + fi,
                dev);
            shared_comp_sz_ring[fi] = comp_sz;
        }
        __syncwarp();
    }
}

// ── IO warp: compute metadata for one P_SIZE page ──
__device__ static void fused_q16part_compute_meta(
    const BAMFusedQ16PartParams& p, uint32_t pg,
    uint64_t& psize_row_base, uint32_t& psize_nalloc,
    uint32_t& brand_start, uint32_t& brand_count, uint64_t& brand_row_offset,
    uint32_t& pk_start, uint32_t& pk_count, uint64_t& pk_row_offset,
    uint32_t& nfields)
{
    constexpr int MAX_BRAND = FUSED_Q16PART_MAX_BRAND;
    constexpr int MAX_PK    = FUSED_Q16PART_MAX_PK;

    uint64_t first_row = p.d_ps_psize[pg];
    uint64_t last_row  = p.d_ps_psize[pg + 1];
    psize_row_base = first_row;
    psize_nalloc = (uint32_t)(last_row - first_row);
    if (last_row > first_row) last_row--;

    // P_BRAND page range
    brand_start = fused_q16part_ps_find(p.d_ps_brand, p.npages_brand + 1, first_row);
    uint32_t brand_end = fused_q16part_ps_find(p.d_ps_brand, p.npages_brand + 1, last_row);
    brand_count = brand_end - brand_start + 1;
    if (brand_count > (uint32_t)MAX_BRAND) brand_count = MAX_BRAND;
    brand_row_offset = first_row - p.d_ps_brand[brand_start];

    // P_PARTKEY page range
    pk_start = fused_q16part_ps_find(p.d_ps_pk, p.npages_pk + 1, first_row);
    uint32_t pk_end = fused_q16part_ps_find(p.d_ps_pk, p.npages_pk + 1, last_row);
    pk_count = pk_end - pk_start + 1;
    if (pk_count > (uint32_t)MAX_PK) pk_count = MAX_PK;
    pk_row_offset = first_row - p.d_ps_pk[pk_start];

    nfields = 1 + brand_count + pk_count;
}

// ════════════════════════════════════════════════════════════════
// Fused Q16 PART kernel: 1-page balanced pipeline with double-buffered scan
//
// Pipeline (same structure as Q3 ORDERS):
//   Priming: IO reads first page → ring[0], meta[0]
//   Main loop:
//     IO(N+1) ∥ Decomp(N→face[f]) ∥ Scan(N-1←face[1-f])
//     __syncthreads__
//   Epilogue: scan last page
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(224, 4)
void bam_lz4_fused_q16part_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ16PartParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr int      NBUF       = FUSED_Q16PART_NBUF;
    constexpr int      NFACE      = FUSED_Q16PART_NFACE;
    constexpr int      NMETA      = FUSED_Q16PART_NMETA;
    constexpr uint32_t MAX_FIELDS = FUSED_Q16PART_MAX_FIELDS;
    constexpr uint32_t WARPS      = FUSED_Q16PART_WARPS;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;   // 0..6
    const uint32_t lane    = tid % 32;

    // ── Shared memory ──
    __shared__ uint32_t shared_comp_sz[NBUF][MAX_FIELDS];
    __shared__ uint32_t s_nfields[NMETA];
    __shared__ uint32_t s_brand_count[NMETA];
    __shared__ uint64_t s_brand_row_offset[NMETA];
    __shared__ uint32_t s_brand_start[NMETA];
    __shared__ uint32_t s_pk_count[NMETA];
    __shared__ uint64_t s_pk_row_offset[NMETA];
    __shared__ uint32_t s_pk_start[NMETA];
    __shared__ uint64_t s_psize_row_base[NMETA];
    __shared__ uint32_t s_psize_nalloc[NMETA];

    // Dynamic shared memory for nvCOMPdx (IO warp doesn't need it)
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

    const uint32_t pg_stride = gridDim.x;
    const uint32_t n_pages   = p.npages_psize;

    // ── Pipeline state ──
    int  io_ring     = 0;
    int  decomp_face = 0;
    int  meta_slot   = 0;
    bool prev_active = false;

    // ════════════════════════════════════════════
    // Priming: IO warp reads first P_SIZE page
    // ════════════════════════════════════════════
    {
        uint32_t idx = blockIdx.x;
        if (warp_id == 0 && idx < n_pages) {
            uint64_t prb; uint32_t pna, bs, bc, ps2, pc2, nf;
            uint64_t bro, pro;
            fused_q16part_compute_meta(p, idx, prb, pna, bs, bc, bro, ps2, pc2, pro, nf);
            if (lane == 0) {
                s_nfields[0] = nf;
                s_brand_count[0] = bc;
                s_brand_row_offset[0] = bro;
                s_brand_start[0] = bs;
                s_pk_count[0] = pc2;
                s_pk_row_offset[0] = pro;
                s_pk_start[0] = ps2;
                s_psize_row_base[0] = prb;
                s_psize_nalloc[0] = pna;
            }
            fused_q16part_io_read_one_page(
                ctrls, pc, p, idx, ndev, bs, bc, ps2, pc2,
                0, SLOTS_PER_BLOCK, shared_comp_sz[0]);
        } else if (warp_id == 0 && lane == 0) {
            s_nfields[0] = 0;
        }
    }
    __syncthreads();

    // ════════════════════════════════════════════
    // Main loop: IO(N+1) ∥ Decomp(N) ∥ Scan(N-1)
    // ════════════════════════════════════════════
    for (uint32_t idx = blockIdx.x; idx < n_pages; idx += pg_stride)
    {
        const int cur_meta  = meta_slot;
        const int next_meta = (meta_slot + 1) % NMETA;
        const int prev_meta = (meta_slot + 2) % NMETA;

        const uint32_t nf = s_nfields[cur_meta];
        const bool active = (nf > 0);

        const uint32_t next_idx = idx + pg_stride;
        const bool has_next = (next_idx < n_pages);

        // ── IO warp: read next P_SIZE page ──
        if (warp_id == 0 && has_next) {
            const int next_ring = 1 - io_ring;
            uint64_t prb; uint32_t pna, bs, bc, ps2, pc2, nf2;
            uint64_t bro, pro;
            fused_q16part_compute_meta(p, next_idx, prb, pna, bs, bc, bro, ps2, pc2, pro, nf2);
            if (lane == 0) {
                s_nfields[next_meta] = nf2;
                s_brand_count[next_meta] = bc;
                s_brand_row_offset[next_meta] = bro;
                s_brand_start[next_meta] = bs;
                s_pk_count[next_meta] = pc2;
                s_pk_row_offset[next_meta] = pro;
                s_pk_start[next_meta] = ps2;
                s_psize_row_base[next_meta] = prb;
                s_psize_nalloc[next_meta] = pna;
            }
            fused_q16part_io_read_one_page(
                ctrls, pc, p, next_idx, ndev, bs, bc, ps2, pc2,
                next_ring, SLOTS_PER_BLOCK,
                shared_comp_sz[next_ring]);
        }

        // ── Decomp warps 1-6: decompress current page fields ──
        if (warp_id >= 1 && active) {
            const uint32_t fi = warp_id - 1;  // 0..5
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
            // Extra round for nf > 6 (rare: 1 + 4 brand + 3 pk = 8 fields)
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

        // ── All warps: scan PREVIOUS page ──
        if (prev_active) {
            const int scan_face = 1 - decomp_face;
            const char* scan_base = d_decomp_buf + block_base
                                  + (uint64_t)scan_face * face_stride;
            constexpr uint32_t SCAN_THREADS = WARPS * 32;
            fused_q16part_scan_page(
                scan_base, p.page_size,
                s_brand_count[prev_meta],
                s_brand_row_offset[prev_meta],
                s_brand_start[prev_meta],
                s_pk_count[prev_meta],
                s_pk_row_offset[prev_meta],
                s_pk_start[prev_meta],
                s_psize_row_base[prev_meta],
                s_psize_nalloc[prev_meta],
                tid, SCAN_THREADS, p);
        }

        __syncthreads();

        prev_active = active;
        io_ring     = 1 - io_ring;
        decomp_face = 1 - decomp_face;
        meta_slot   = next_meta;
    }

    // ════════════════════════════════════════════
    // Epilogue: scan the last page
    // ════════════════════════════════════════════
    if (prev_active) {
        const int prev_meta = (meta_slot + 2) % NMETA;
        const int scan_face = 1 - decomp_face;
        const char* scan_base = d_decomp_buf + block_base
                              + (uint64_t)scan_face * face_stride;
        constexpr uint32_t SCAN_THREADS = WARPS * 32;
        fused_q16part_scan_page(
            scan_base, p.page_size,
            s_brand_count[prev_meta],
            s_brand_row_offset[prev_meta],
            s_brand_start[prev_meta],
            s_pk_count[prev_meta],
            s_pk_row_offset[prev_meta],
            s_pk_start[prev_meta],
            s_psize_row_base[prev_meta],
            s_psize_nalloc[prev_meta],
            tid, SCAN_THREADS, p);
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

struct BAMFusedQ16PartContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_q16part_ctx_t bam_fused_q16part_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ16PartContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * FUSED_Q16PART_NBUF * FUSED_Q16PART_MAX_FIELDS;
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: NFACE × MAX_FIELDS pages per block
    size_t decomp_size = (size_t)num_blocks * FUSED_Q16PART_NFACE
                       * FUSED_Q16PART_MAX_FIELDS * page_size;
    FUSED_Q16PART_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q16part_ctx_t>(ctx);
}

static void bam_fused_q16part_launch(
    BAMFusedQ16PartContext* ctx,
    const BAMFusedQ16PartParams& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = FUSED_Q16PART_WARPS * 32;  // 224
    constexpr uint32_t DECOMP_WARPS = FUSED_Q16PART_DECOMP_WARPS;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>()
                         * DECOMP_WARPS;
        auto kernel_fn = bam_lz4_fused_q16part_kernel<PS>;
        FUSED_Q16PART_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_Q16PART_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q16part_run_async(
    bam_fused_q16part_ctx_t ctx_handle,
    const BAMFusedQ16PartParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ16PartContext*>(ctx_handle);
    bam_fused_q16part_launch(ctx, params, stream);
}

void bam_fused_q16part_destroy(bam_fused_q16part_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ16PartContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}
