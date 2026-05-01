// bam_lz4_q3sel_scan.cu — Q3SEL GPU-side IO pruning + IO+decomp + scan
//
// All IO pruning (INT64 page derivation, tile planning, LBA computation)
// is performed GPU-side.  IO+decomp and scan are separated so the scan
// kernel gets full GPU parallelism (all SMs).
//
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_q3sel_scan.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "bam_bulk_read.cuh"
#include "page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>

#define Q3SEL_CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// ── Common device helpers ──

__device__ static uint32_t q3sel_ps_find_page(
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

__device__ static uint32_t q3sel_lower_bound_u32(
    const uint32_t* arr, uint32_t n, uint32_t val)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ static uint32_t q3sel_upper_bound_u32(
    const uint32_t* arr, uint32_t n, uint32_t val)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (arr[mid] <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ static uint32_t q3sel_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

static constexpr uint64_t Q3SEL_HT_EMPTY = UINT64_MAX;

__device__ static bool q3sel_hashset_probe(
    const uint64_t *keys, uint32_t mask, uint64_t key)
{
    uint32_t slot = q3sel_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return true;
        if (k == Q3SEL_HT_EMPTY) return false;
        slot = (slot + 1) & mask;
    }
}

__device__ static void q3sel_ht_insert_kv(
    uint64_t *keys, uint64_t *payloads, uint32_t mask,
    uint64_t key, uint64_t payload)
{
    uint32_t slot = q3sel_hash64(key) & mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)Q3SEL_HT_EMPTY,
            (unsigned long long)key);
        if (prev == Q3SEL_HT_EMPTY || prev == key) {
            payloads[slot] = payload;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

__device__ static uint64_t q3sel_ht_probe_kv(
    const uint64_t *keys, const uint64_t *payloads,
    uint32_t mask, uint64_t key)
{
    uint32_t slot = q3sel_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return payloads[slot];
        if (k == Q3SEL_HT_EMPTY) return Q3SEL_HT_EMPTY;
        slot = (slot + 1) & mask;
    }
}

// ════════════════════════════════════════════════════════════════
// GPU-side IO Pruning Kernel 1: Derive needed INT64 pages
//
// Single-block, 256 threads.  For each active INT32 page, binary
// search d_ps_i32/d_ps_i64 to find referenced INT64 pages, mark
// d_i64_mask, then warp-ballot compact to d_needed_i64.
// ════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(256, 1)
q3sel_derive_i64_kernel(
    const uint32_t* __restrict__ d_active_ids, uint32_t n_active,
    const uint64_t* __restrict__ d_ps_i32,
    const uint64_t* __restrict__ d_ps_i64,
    uint32_t npages_i64,
    uint8_t* __restrict__ d_i64_mask,
    uint32_t* __restrict__ d_needed_i64,
    uint32_t* __restrict__ d_n_needed_i64)
{
    const uint32_t tid = threadIdx.x;

    // Phase 1: Clear mask
    for (uint32_t i = tid; i < npages_i64; i += blockDim.x)
        d_i64_mask[i] = 0;
    __syncthreads();

    // Phase 2: Mark needed INT64 pages
    for (uint32_t i = tid; i < n_active; i += blockDim.x) {
        uint32_t pg = d_active_ids[i];
        uint64_t first_row = d_ps_i32[pg];
        uint64_t last_row  = d_ps_i32[pg + 1];
        if (first_row >= last_row) continue;
        last_row--;

        uint32_t first_i64 = q3sel_ps_find_page(d_ps_i64, npages_i64 + 1, first_row);
        uint32_t last_i64  = q3sel_ps_find_page(d_ps_i64, npages_i64 + 1, last_row);

        for (uint32_t j = first_i64; j <= last_i64 && j < npages_i64; j++)
            d_i64_mask[j] = 1;
    }
    __syncthreads();

    // Phase 3: Warp-ballot compact mask → d_needed_i64
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;
    constexpr uint32_t NWARPS = 8;  // 256 / 32

    __shared__ uint32_t s_wpfx[NWARPS];
    __shared__ uint32_t s_base;

    if (tid == 0) s_base = 0;
    __syncthreads();

    for (uint32_t chunk = 0; chunk < npages_i64; chunk += blockDim.x) {
        uint32_t pg = chunk + tid;
        bool is_active = (pg < npages_i64) && d_i64_mask[pg];

        uint32_t ballot      = __ballot_sync(0xffffffff, is_active);
        uint32_t lane_prefix  = __popc(ballot & ((1u << lane) - 1));
        uint32_t warp_cnt    = __popc(ballot);

        if (lane == 0) s_wpfx[warp_id] = warp_cnt;
        __syncthreads();

        if (tid == 0) {
            uint32_t b = s_base, sum = 0;
            for (uint32_t w = 0; w < NWARPS; w++) {
                uint32_t c = s_wpfx[w];
                s_wpfx[w] = b + sum;
                sum += c;
            }
            s_base = b + sum;
        }
        __syncthreads();

        if (is_active)
            d_needed_i64[s_wpfx[warp_id] + lane_prefix] =
                static_cast<uint32_t>(pg);
        __syncthreads();
    }

    if (tid == 0) *d_n_needed_i64 = s_base;
}

// ════════════════════════════════════════════════════════════════
// GPU-side IO Pruning Kernel 2: Tile plan for LINEITEM
//
// Single-thread sequential binary search.  Partitions active INT32
// pages into tiles that fit the staging buffer capacity.
// ════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(32, 1)
q3sel_tile_plan_kernel(
    const uint32_t* __restrict__ d_active_ids, uint32_t n_active,
    const uint32_t* __restrict__ d_needed_i64, uint32_t n_needed_i64,
    const uint64_t* __restrict__ d_ps_i32,
    const uint64_t* __restrict__ d_ps_i64,
    uint32_t npages_i64,
    uint32_t staging_capacity,
    uint32_t n_i32_fields, uint32_t n_i64_fields,
    Q3SelTileInfo* __restrict__ d_tiles,
    uint32_t* __restrict__ d_n_tiles)
{
    if (threadIdx.x != 0) return;

    uint32_t pos = 0;
    uint32_t n_tiles = 0;

    while (pos < n_active) {
        uint32_t best = 0;
        uint32_t lo_s = 1, hi_s = n_active - pos;

        while (lo_s <= hi_s) {
            uint32_t mid = lo_s + (hi_s - lo_s) / 2;

            uint32_t first_pg = d_active_ids[pos];
            uint32_t last_pg  = d_active_ids[pos + mid - 1];
            uint64_t fr = d_ps_i32[first_pg];
            uint64_t lr = d_ps_i32[last_pg + 1];
            uint32_t n_i64 = 0;

            if (lr > fr) {
                lr--;
                uint32_t fi64 = q3sel_ps_find_page(d_ps_i64, npages_i64 + 1, fr);
                uint32_t li64 = q3sel_ps_find_page(d_ps_i64, npages_i64 + 1, lr);
                uint32_t lb = q3sel_lower_bound_u32(d_needed_i64, n_needed_i64, fi64);
                uint32_t ub = q3sel_upper_bound_u32(d_needed_i64, n_needed_i64, li64);
                n_i64 = ub - lb;
            }

            if (n_i32_fields * mid + n_i64_fields * n_i64 <= staging_capacity) {
                best = mid;
                lo_s = mid + 1;
            } else {
                hi_s = mid - 1;
            }
        }
        if (best == 0) best = 1;

        Q3SelTileInfo tile;
        tile.i32_start = pos;
        tile.i32_count = best;

        uint32_t first_pg = d_active_ids[pos];
        uint32_t last_pg  = d_active_ids[pos + best - 1];
        uint64_t fr = d_ps_i32[first_pg];
        uint64_t lr = d_ps_i32[last_pg + 1];
        if (lr > fr) {
            lr--;
            uint32_t fi64 = q3sel_ps_find_page(d_ps_i64, npages_i64 + 1, fr);
            uint32_t li64 = q3sel_ps_find_page(d_ps_i64, npages_i64 + 1, lr);
            tile.i64_start = q3sel_lower_bound_u32(d_needed_i64, n_needed_i64, fi64);
            uint32_t ub = q3sel_upper_bound_u32(d_needed_i64, n_needed_i64, li64);
            tile.i64_count = ub - tile.i64_start;
        } else {
            tile.i64_start = 0;
            tile.i64_count = 0;
        }
        tile.total_descs = n_i32_fields * best + n_i64_fields * tile.i64_count;

        d_tiles[n_tiles++] = tile;
        pos += best;
    }

    *d_n_tiles = n_tiles;
}

// ════════════════════════════════════════════════════════════════
// GPU-side IO Pruning Kernel 3: Per-tile setup
//
// Builds d_i64_remap (global INT64 page → staging slot for this tile),
// d_active_ps (prefix sum of active page row counts), *d_nrecs.
// ════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(256, 1)
q3sel_tile_setup_kernel(
    const uint32_t* __restrict__ d_active_ids,
    uint32_t tile_i32_start, uint32_t tile_i32_count,
    const uint32_t* __restrict__ d_needed_i64,
    uint32_t tile_i64_start, uint32_t tile_i64_count,
    const uint64_t* __restrict__ d_ps_i32_full,
    uint32_t npages_i64,
    uint64_t* __restrict__ d_active_ps,
    uint32_t* __restrict__ d_i64_remap,
    uint64_t* __restrict__ d_nrecs)
{
    const uint32_t tid = threadIdx.x;

    // Phase 1: Clear i64_remap
    for (uint32_t i = tid; i < npages_i64; i += blockDim.x)
        d_i64_remap[i] = UINT32_MAX;
    __syncthreads();

    // Phase 2: Build i64_remap for this tile
    for (uint32_t j = tid; j < tile_i64_count; j += blockDim.x)
        d_i64_remap[d_needed_i64[tile_i64_start + j]] = j;

    // Phase 3: Build active prefix sum (sequential — tile_i32_count is small)
    if (tid == 0) {
        uint64_t acc = 0;
        d_active_ps[0] = 0;
        for (uint32_t i = 0; i < tile_i32_count; i++) {
            uint32_t pg = d_active_ids[tile_i32_start + i];
            acc += d_ps_i32_full[pg + 1] - d_ps_i32_full[pg];
            d_active_ps[i + 1] = acc;
        }
        *d_nrecs = acc;
    }
}

// ════════════════════════════════════════════════════════════════
// IO+Decomp kernel: GPU-side LBA computation
//
// 128 threads = 4 warps per block.  Each warp handles one staging
// slot at a time.  LBA, nblocks, device are computed from field
// metadata (d_active_ids, d_needed_i64, d_comp_offsets, etc.).
//
// Staging layout:
//   [i32_field_0 × n_active] [i32_field_1 × n_active] ...
//   [i64_field_0 × n_needed_i64] [i64_field_1 × n_needed_i64] ...
// ════════════════════════════════════════════════════════════════

template <unsigned int PAGE_SIZE_CONST>
__global__ void __launch_bounds__(128, 2)
q3sel_io_decomp_kernel(
    void* d_ctrls, void* d_pc, const char* pc_base_addr,
    char* d_staging,
    Q3SelIODecompParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS_PER_BLOCK = 4;

    const uint32_t warp_id     = threadIdx.x / 32;
    const uint32_t global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const uint32_t total_warps = gridDim.x * WARPS_PER_BLOCK;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t slot = global_warp;

    const uint32_t total_i32   = p.n_active * p.n_i32_fields;
    const uint32_t total_descs = total_i32 + p.n_needed_i64 * p.n_i64_fields;

    for (uint32_t j = global_warp; j < total_descs; j += total_warps) {
        uint32_t pg;
        uint64_t global_pg;
        uint64_t lba;
        uint32_t nblocks, comp_sz, device;

        if (j < total_i32) {
            // INT32 field descriptor
            uint32_t fi  = j / p.n_active;
            uint32_t idx = j % p.n_active;
            pg = p.d_active_ids[idx];
            global_pg = p.field_start_page_ids_i32[fi] + pg;
            device = global_pg % p.n_devices;
            if (p.is_compressed_i32[fi]) {
                lba = p.partition_start_lbas[device] +
                      p.d_comp_offsets_i32[fi][pg] / 512;
                comp_sz = p.d_comp_sizes_i32[fi][pg];
                nblocks = ((comp_sz + 4095u) & ~4095u) / 512;
                if (nblocks > 8 && nblocks <= 16) nblocks = 24;
            } else {
                uint64_t local_pg = global_pg / p.n_devices;
                lba = p.partition_start_lbas[device] +
                      local_pg * (p.page_size / 512);
                nblocks = p.page_size / 512;
                comp_sz = p.page_size;
            }
        } else {
            // INT64 field descriptor
            uint32_t k   = j - total_i32;
            uint32_t fi  = k / p.n_needed_i64;
            uint32_t idx = k % p.n_needed_i64;
            pg = p.d_needed_i64[idx];
            global_pg = p.field_start_page_ids_i64[fi] + pg;
            device = global_pg % p.n_devices;
            if (p.is_compressed_i64[fi]) {
                lba = p.partition_start_lbas[device] +
                      p.d_comp_offsets_i64[fi][pg] / 512;
                comp_sz = p.d_comp_sizes_i64[fi][pg];
                nblocks = ((comp_sz + 4095u) & ~4095u) / 512;
                if (nblocks > 8 && nblocks <= 16) nblocks = 24;
            } else {
                uint64_t local_pg = global_pg / p.n_devices;
                lba = p.partition_start_lbas[device] +
                      local_pg * (p.page_size / 512);
                nblocks = p.page_size / 512;
                comp_sz = p.page_size;
            }
        }

        char* dst = d_staging + (uint64_t)j * p.page_size;
        bam_lz4_io_decomp_warp<PAGE_SIZE_CONST>(
            d_ctrls, d_pc, (void*)pc_base_addr,
            slot, dst,
            lba, nblocks, device,
            comp_sz, p.page_size, my_smem);
    }
}

// ════════════════════════════════════════════════════════════════
// ORDERS scan kernel (unchanged)
// ════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(256, 4)
q3sel_orders_scan_kernel(Q3SelOrdersScanParams p)
{
    const uint64_t gtid   = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t r = gtid; r < p.nrecs_active; r += stride) {
        uint32_t pg_idx = q3sel_ps_find_page(p.d_active_ps_i32, p.n_active_i32 + 1, r);
        uint32_t local_rec = (uint32_t)(r - p.d_active_ps_i32[pg_idx]);

        const char* odate_page = p.d_staging +
            (uint64_t)(p.odate_pg_off + pg_idx) * p.page_size;
        int32_t odate = *(const int32_t*)(odate_page + 12 + (uint64_t)local_rec * 4);
        if (p.o_orderdate_limit != 0 && odate >= p.o_orderdate_limit) continue;

        uint32_t global_i32_page = p.d_active_pages_i32[pg_idx];
        uint64_t global_row = p.d_ps_i32_full[global_i32_page] + local_rec;

        uint32_t i64_global_page = q3sel_ps_find_page(
            p.d_ps_i64_full, p.npages_i64 + 1, global_row);
        uint32_t staging_i64 = p.d_i64_remap[i64_global_page];
        uint32_t i64_local_rec = (uint32_t)(global_row - p.d_ps_i64_full[i64_global_page]);

        const char* ckey_page = p.d_staging +
            (uint64_t)(p.ckey_pg_off + staging_i64) * p.page_size;
        uint64_t custkey = *(const uint64_t*)(ckey_page + 16 + (uint64_t)i64_local_rec * 8);

        if (!q3sel_hashset_probe(p.d_custkey_set, p.custkey_set_mask, custkey))
            continue;

        const char* okey_page = p.d_staging +
            (uint64_t)(p.okey_pg_off + staging_i64) * p.page_size;
        uint64_t orderkey = *(const uint64_t*)(okey_page + 16 + (uint64_t)i64_local_rec * 8);

        const char* sp_page = p.d_staging +
            (uint64_t)(p.sp_pg_off + pg_idx) * p.page_size;
        int32_t shippriority = *(const int32_t*)(sp_page + 12 + (uint64_t)local_rec * 4);

        uint64_t payload = ((uint64_t)(uint32_t)odate << 32) | (uint64_t)(uint32_t)shippriority;
        q3sel_ht_insert_kv(
            p.d_orders_ht_keys, p.d_orders_ht_payloads, p.orders_ht_mask,
            orderkey, payload);
    }
}

// ════════════════════════════════════════════════════════════════
// LINEITEM scan kernel (unchanged)
// ════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(256, 4)
q3sel_lineitem_scan_kernel(Q3SelLineitemScanParams p)
{
    const uint64_t gtid   = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t r = gtid; r < p.nrecs_active; r += stride) {
        uint32_t pg_idx = q3sel_ps_find_page(p.d_active_ps_i32, p.n_active_i32 + 1, r);
        uint32_t local_rec = (uint32_t)(r - p.d_active_ps_i32[pg_idx]);

        uint32_t global_i32_page = p.d_active_pages_i32[pg_idx];
        uint64_t global_row = p.d_ps_i32_full[global_i32_page] + local_rec;

        uint32_t i64_global_page = q3sel_ps_find_page(
            p.d_ps_i64_full, p.npages_i64 + 1, global_row);
        uint32_t staging_i64 = p.d_i64_remap[i64_global_page];
        uint32_t i64_local_rec = (uint32_t)(global_row - p.d_ps_i64_full[i64_global_page]);

        const char* okey_page = p.d_staging +
            (uint64_t)(p.okey_pg_off + staging_i64) * p.page_size;
        uint64_t orderkey = *(const uint64_t*)(okey_page + 16 + (uint64_t)i64_local_rec * 8);

        uint64_t payload = q3sel_ht_probe_kv(
            p.d_orders_ht_keys, p.d_orders_ht_payloads, p.orders_ht_mask, orderkey);
        if (payload == Q3SEL_HT_EMPTY) continue;

        const char* ep_page = p.d_staging +
            (uint64_t)(p.extprice_pg_off + pg_idx) * p.page_size;
        const char* dc_page = p.d_staging +
            (uint64_t)(p.discount_pg_off + pg_idx) * p.page_size;
        int32_t extprice = *(const int32_t*)(ep_page + 12 + (uint64_t)local_rec * 4);
        int32_t discount = *(const int32_t*)(dc_page + 12 + (uint64_t)local_rec * 4);

        int64_t revenue = (int64_t)extprice * (int64_t)(100 - discount);

        uint32_t aggr_slot = q3sel_hash64(orderkey) & p.aggr_mask;
        while (true) {
            uint64_t prev = atomicCAS(
                reinterpret_cast<unsigned long long *>(&p.d_aggr_keys[aggr_slot]),
                (unsigned long long)Q3SEL_HT_EMPTY,
                (unsigned long long)orderkey);
            if (prev == Q3SEL_HT_EMPTY || prev == orderkey) {
                atomicAdd(reinterpret_cast<unsigned long long *>(&p.d_aggr_revenues[aggr_slot]),
                          (unsigned long long)revenue);
                break;
            }
            aggr_slot = (aggr_slot + 1) & p.aggr_mask;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

void q3sel_derive_i64_launch(
    const uint32_t* d_active_ids, uint32_t n_active,
    const uint64_t* d_ps_i32, const uint64_t* d_ps_i64,
    uint32_t npages_i64,
    uint8_t* d_i64_mask,
    uint32_t* d_needed_i64,
    uint32_t* d_n_needed_i64,
    cudaStream_t stream)
{
    if (n_active == 0) {
        uint32_t zero = 0;
        Q3SEL_CUDA_CHECK(cudaMemcpyAsync(d_n_needed_i64, &zero, sizeof(uint32_t),
                                          cudaMemcpyHostToDevice, stream));
        return;
    }
    q3sel_derive_i64_kernel<<<1, 256, 0, stream>>>(
        d_active_ids, n_active, d_ps_i32, d_ps_i64, npages_i64,
        d_i64_mask, d_needed_i64, d_n_needed_i64);
}

void q3sel_tile_plan_launch(
    const uint32_t* d_active_ids, uint32_t n_active,
    const uint32_t* d_needed_i64, uint32_t n_needed_i64,
    const uint64_t* d_ps_i32, const uint64_t* d_ps_i64,
    uint32_t npages_i64,
    uint32_t staging_capacity,
    uint32_t n_i32_fields, uint32_t n_i64_fields,
    Q3SelTileInfo* d_tiles, uint32_t* d_n_tiles,
    cudaStream_t stream)
{
    if (n_active == 0) {
        uint32_t zero = 0;
        Q3SEL_CUDA_CHECK(cudaMemcpyAsync(d_n_tiles, &zero, sizeof(uint32_t),
                                          cudaMemcpyHostToDevice, stream));
        return;
    }
    q3sel_tile_plan_kernel<<<1, 32, 0, stream>>>(
        d_active_ids, n_active, d_needed_i64, n_needed_i64,
        d_ps_i32, d_ps_i64, npages_i64,
        staging_capacity, n_i32_fields, n_i64_fields,
        d_tiles, d_n_tiles);
}

void q3sel_tile_setup_launch(
    const uint32_t* d_active_ids,
    uint32_t tile_i32_start, uint32_t tile_i32_count,
    const uint32_t* d_needed_i64,
    uint32_t tile_i64_start, uint32_t tile_i64_count,
    const uint64_t* d_ps_i32_full,
    uint32_t npages_i64,
    uint64_t* d_active_ps,
    uint32_t* d_i64_remap,
    uint64_t* d_nrecs,
    cudaStream_t stream)
{
    q3sel_tile_setup_kernel<<<1, 256, 0, stream>>>(
        d_active_ids, tile_i32_start, tile_i32_count,
        d_needed_i64, tile_i64_start, tile_i64_count,
        d_ps_i32_full, npages_i64,
        d_active_ps, d_i64_remap, d_nrecs);
}

uint32_t q3sel_io_decomp_max_blocks(uint32_t page_size)
{
    int sm_count = 0;
    Q3SEL_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    // 2 blocks/SM with __launch_bounds__(128, 2)
    return (uint32_t)sm_count * 2;
}

void q3sel_io_decomp_launch(
    void* d_ctrls, void* d_pc, const char* pc_base_addr,
    char* d_staging,
    const Q3SelIODecompParams& params,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    uint32_t total_descs = params.n_active * params.n_i32_fields +
                           params.n_needed_i64 * params.n_i64_fields;
    if (total_descs == 0) return;

    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PSC = decltype(ps_tag)::value;
        using lz4_decomp_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<PSC>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
        size_t smem_bytes = 4 * warp_smem;  // 4 warps per block

        auto kernel_fn = q3sel_io_decomp_kernel<PSC>;
        Q3SEL_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

        kernel_fn<<<num_blocks, 128, smem_bytes, stream>>>(
            d_ctrls, d_pc, pc_base_addr, d_staging, params);
    });
}

void q3sel_orders_scan_launch(
    const Q3SelOrdersScanParams& params,
    cudaStream_t stream)
{
    if (params.nrecs_active == 0) return;

    int sm_count = 0;
    Q3SEL_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

    uint32_t num_blocks = (uint32_t)sm_count * 4;
    uint64_t needed = (params.nrecs_active + 255) / 256;
    if (num_blocks > needed) num_blocks = (uint32_t)needed;

    q3sel_orders_scan_kernel<<<num_blocks, 256, 0, stream>>>(params);
}

void q3sel_lineitem_scan_launch(
    const Q3SelLineitemScanParams& params,
    cudaStream_t stream)
{
    if (params.nrecs_active == 0) return;

    int sm_count = 0;
    Q3SEL_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

    uint32_t num_blocks = (uint32_t)sm_count * 4;
    uint64_t needed = (params.nrecs_active + 255) / 256;
    if (num_blocks > needed) num_blocks = (uint32_t)needed;

    q3sel_lineitem_scan_kernel<<<num_blocks, 256, 0, stream>>>(params);
}
