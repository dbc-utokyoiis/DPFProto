// bam_lz4_fused_ssb.cu — SSB Fused BaM I/O + nvCOMPdx LZ4 + scan kernels
//
// Warp-specialized persistent kernels (1024 threads, __launch_bounds__(1024, 1)):
//   Q1x/Q2x/Q3x: 4 IO warps + 7 decomp groups × 4 fields = 32 warps
//     BATCH=7, N_BUF=2, SLOTS_PER_BLOCK=56
//   Q4x: 6 IO warps + 4 decomp groups × 6 fields = 30 warps (+2 idle)
//     BATCH=4, N_BUF=2, SLOTS_PER_BLOCK=48
//
// Pipeline: Prolog(IO batch 0) → Main(IO[b]‖Decomp[b-1]→Scan[b-1]) → Epilog
//
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_ssb.cuh"
#include "tpch/bam_lz4_io_decomp.cuh"
#include "tpch/page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>

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

    // Fast path: no mask → fill identity in parallel
    if (!d_mask) {
        for (uint32_t i = tid; i < max_pg; i += blockDim.x)
            s_active[i] = first + i * stride;
        if (tid == 0) *s_count = max_pg;
        __syncthreads();
        return;
    }

    // Each thread checks one page (max_pg ≤ kZonemapMaxPagesPerBlock < blockDim.x)
    uint32_t pg = first + tid * stride;
    bool is_active = (tid < max_pg && pg < total_pages) && d_mask[pg];

    // Warp-level vote + intra-warp prefix count
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;
    uint32_t ballot      = __ballot_sync(0xffffffff, is_active);
    uint32_t lane_prefix = __popc(ballot & ((1u << lane) - 1));
    uint32_t warp_cnt    = __popc(ballot);

    // Collect per-warp active counts
    __shared__ uint32_t s_wpfx[32];
    if (lane == 0) s_wpfx[warp_id] = warp_cnt;
    __syncthreads();

    // Exclusive prefix sum across 32 warps (thread 0, 32 iters on smem)
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

    // Scatter: order-preserving write
    if (is_active)
        s_active[s_wpfx[warp_id] + lane_prefix] = pg;
    __syncthreads();
}

#define SSB_FUSED_CHECK(call) do {                                            \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// ════════════════════════════════════════════════════════════════
// Device helpers
// ════════════════════════════════════════════════════════════════

// PRP2 danger zone fix: BaM 9-16 block reads are unsafe → skip to 24
__device__ static uint32_t ssb_fused_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// IO parameter computation (generic, works for any number of fields)
template <typename P>
__device__ __forceinline__
static void ssb_ws_io_params(
    const P& p, uint32_t field, uint32_t orig_pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.field_start_page_ids[field] + orig_pg;
    dev = global_pg % ndev;
    if (p.is_compressed[field]) {
        lba = p.partition_start_lbas[dev] +
              p.d_comp_offsets[field][orig_pg] / 512;
        comp_sz = p.d_comp_sizes[field][orig_pg];
        nblk = ssb_fused_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

// Hash table probe
__device__ static uint32_t ssb_fused_hash32(uint32_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

__device__ static int32_t ssb_fused_ht_probe(
    const int32_t* keys, const int32_t* values, uint32_t mask, int32_t key)
{
    uint32_t slot = ssb_fused_hash32((uint32_t)key) & mask;
    while (true) {
        int32_t k = keys[slot];
        if (k == key) return values[slot];
        if (k == -1) return -1;
        slot = (slot + 1) & mask;
    }
}

// Date/dimension constants
static constexpr int32_t  FUSED_YEAR_MIN      = 1992;
static constexpr int32_t  FUSED_MAX_BRANDS    = 40;
static constexpr int32_t  FUSED_MAX_YEARS     = 7;

// ════════════════════════════════════════════════════════════════
// Scan helpers — overloaded by params type for template dispatch
// ════════════════════════════════════════════════════════════════

// Q1x scan: datekey HT + disc/qty filter → SUM(extprice * discount)
__device__ __forceinline__
static void ssb_ws_scan(
    const char* d_decomp_buf, const SSBFusedQ1xParams& p,
    uint32_t base_slot, uint32_t count, uint32_t tid)
{
    constexpr uint32_t T = 1024, NF = 4, HDR = 3;
    int64_t batch_val = 0;
    for (uint32_t j = 0; j < count; j++) {
        uint64_t base = (uint64_t)(base_slot + j * NF) * p.page_size;
        const int32_t* od = (const int32_t*)(d_decomp_buf + base) + HDR;
        const int32_t* qt = (const int32_t*)(d_decomp_buf + base + (uint64_t)p.page_size) + HDR;
        const int32_t* dc = (const int32_t*)(d_decomp_buf + base + 2*(uint64_t)p.page_size) + HDR;
        const int32_t* ep = (const int32_t*)(d_decomp_buf + base + 3*(uint64_t)p.page_size) + HDR;
        uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base);
        for (uint32_t r = tid; r < nalloc; r += T) {
            int32_t orderdate = od[r], quantity = qt[r], discount = dc[r], extprice = ep[r];
            bool dm = (ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values,
                                          p.date_ht_mask, orderdate) >= 0);
            if (dm && discount >= p.disc_lo && discount <= p.disc_hi &&
                quantity >= p.qty_lo && quantity < p.qty_hi) {
                batch_val += (int64_t)extprice * discount;
            }
        }
    }
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        batch_val += __shfl_down_sync(0xffffffff, batch_val, offset);

    // Inter-warp reduction via shared memory
    __shared__ int64_t s_warp_sums[32];
    const uint32_t warp_id = tid / 32;
    const uint32_t lane = tid % 32;
    if (lane == 0) s_warp_sums[warp_id] = batch_val;
    __syncthreads();

    if (warp_id == 0) {
        int64_t val = s_warp_sums[lane];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0 && val != 0)
            atomicAdd(reinterpret_cast<unsigned long long int*>(p.d_revenue),
                      static_cast<unsigned long long int>(val));
    }
    __syncthreads();
}

// Q2x scan: date year + supp HT + part HT → revenue[year × brand]
// Shared memory histogram privatization (280 entries = 2240 bytes)
__device__ __forceinline__
static void ssb_ws_scan(
    const char* d_decomp_buf, const SSBFusedQ2xParams& p,
    uint32_t base_slot, uint32_t count, uint32_t tid)
{
    constexpr uint32_t T = 1024, NF = 4, HDR = 3;
    constexpr uint32_t HIST_SIZE = FUSED_MAX_YEARS * FUSED_MAX_BRANDS;  // 7 * 40 = 280
    __shared__ int64_t s_hist[HIST_SIZE];

    // Initialize shared histogram
    for (uint32_t i = tid; i < HIST_SIZE; i += T) s_hist[i] = 0;
    __syncthreads();

    for (uint32_t j = 0; j < count; j++) {
        uint64_t base = (uint64_t)(base_slot + j * NF) * p.page_size;
        const int32_t* od = (const int32_t*)(d_decomp_buf + base) + HDR;
        const int32_t* pk = (const int32_t*)(d_decomp_buf + base + (uint64_t)p.page_size) + HDR;
        const int32_t* sk = (const int32_t*)(d_decomp_buf + base + 2*(uint64_t)p.page_size) + HDR;
        const int32_t* rv = (const int32_t*)(d_decomp_buf + base + 3*(uint64_t)p.page_size) + HDR;
        uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base);
        for (uint32_t r = tid; r < nalloc; r += T) {
            int32_t orderdate = od[r], partkey = pk[r], suppkey = sk[r], revenue = rv[r];
            int32_t yi = ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, orderdate);
            if (yi < 0) continue;
            int32_t sv = ssb_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, suppkey);
            if (sv < 0) continue;
            int32_t bi = ssb_fused_ht_probe(p.d_part_ht_keys, p.d_part_ht_values, p.part_ht_mask, partkey);
            if (bi < 0) continue;
            int32_t gi = yi * FUSED_MAX_BRANDS + bi;
            atomicAdd(reinterpret_cast<unsigned long long int*>(&s_hist[gi]),
                      static_cast<unsigned long long int>((int64_t)revenue));
        }
    }
    __syncthreads();

    // Flush shared histogram to global
    for (uint32_t i = tid; i < HIST_SIZE; i += T) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int*>(&p.d_revenue[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
    __syncthreads();
}

// Q3x scan: date year + cust HT + supp HT → revenue[cust × supp × year]
// Reuses extern shared memory (nvCOMPdx region, safe during scan phase)
__device__ __forceinline__
static void ssb_ws_scan(
    const char* d_decomp_buf, const SSBFusedQ3xParams& p,
    uint32_t base_slot, uint32_t count, uint32_t tid)
{
    extern __shared__ __align__(8) uint8_t smem[];
    int64_t* s_hist = reinterpret_cast<int64_t*>(smem);
    constexpr uint32_t T = 1024, NF = 4, HDR = 3;
    const uint32_t hs = p.hist_size;

    for (uint32_t i = tid; i < hs; i += T) s_hist[i] = 0;
    __syncthreads();

    for (uint32_t j = 0; j < count; j++) {
        uint64_t base = (uint64_t)(base_slot + j * NF) * p.page_size;
        const int32_t* od = (const int32_t*)(d_decomp_buf + base) + HDR;
        const int32_t* ck = (const int32_t*)(d_decomp_buf + base + (uint64_t)p.page_size) + HDR;
        const int32_t* sk = (const int32_t*)(d_decomp_buf + base + 2*(uint64_t)p.page_size) + HDR;
        const int32_t* rv = (const int32_t*)(d_decomp_buf + base + 3*(uint64_t)p.page_size) + HDR;
        uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base);
        for (uint32_t r = tid; r < nalloc; r += T) {
            int32_t orderdate = od[r], custkey = ck[r], suppkey = sk[r], revenue = rv[r];
            int32_t yi = ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, orderdate);
            if (yi < 0) continue;
            int32_t cd = ssb_fused_ht_probe(p.d_cust_ht_keys, p.d_cust_ht_values, p.cust_ht_mask, custkey);
            if (cd < 0) continue;
            int32_t sd = ssb_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, suppkey);
            if (sd < 0) continue;
            int32_t gi = cd * p.num_supp_dims * FUSED_MAX_YEARS + sd * FUSED_MAX_YEARS + yi;
            atomicAdd(reinterpret_cast<unsigned long long int*>(&s_hist[gi]),
                      static_cast<unsigned long long int>((int64_t)revenue));
        }
    }
    __syncthreads();

    // Flush shared histogram to global
    for (uint32_t i = tid; i < hs; i += T) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int*>(&p.d_revenue[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
}

// Q4x scan: 6 fields, date year + 3 HTs → profit[year × cust × supp × part]
// Reuses extern shared memory (nvCOMPdx region, safe during scan phase)
__device__ __forceinline__
static void ssb_q4x_scan(
    const char* d_decomp_buf, const SSBFusedQ4xParams& p,
    uint32_t base_slot, uint32_t count, uint32_t tid)
{
    extern __shared__ __align__(8) uint8_t smem[];
    int64_t* s_hist = reinterpret_cast<int64_t*>(smem);
    constexpr uint32_t T = 1024, NF = 6, HDR = 3;
    const uint32_t hs = p.hist_size;

    for (uint32_t i = tid; i < hs; i += T) s_hist[i] = 0;
    __syncthreads();

    for (uint32_t j = 0; j < count; j++) {
        uint64_t base = (uint64_t)(base_slot + j * NF) * p.page_size;
        const int32_t* od = (const int32_t*)(d_decomp_buf + base) + HDR;
        const int32_t* ck = (const int32_t*)(d_decomp_buf + base + (uint64_t)p.page_size) + HDR;
        const int32_t* pk = (const int32_t*)(d_decomp_buf + base + 2*(uint64_t)p.page_size) + HDR;
        const int32_t* sk = (const int32_t*)(d_decomp_buf + base + 3*(uint64_t)p.page_size) + HDR;
        const int32_t* rv = (const int32_t*)(d_decomp_buf + base + 4*(uint64_t)p.page_size) + HDR;
        const int32_t* sc = (const int32_t*)(d_decomp_buf + base + 5*(uint64_t)p.page_size) + HDR;
        uint32_t nalloc = *(const uint32_t*)(d_decomp_buf + base);
        for (uint32_t r = tid; r < nalloc; r += T) {
            int32_t orderdate = od[r], custkey = ck[r], partkey = pk[r];
            int32_t suppkey = sk[r], revenue = rv[r], supplycost = sc[r];
            int32_t yi = ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, orderdate);
            if (yi < 0) continue;
            int32_t cv = ssb_fused_ht_probe(p.d_cust_ht_keys, p.d_cust_ht_values, p.cust_ht_mask, custkey);
            if (cv < 0) continue;
            int32_t sv = ssb_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, suppkey);
            if (sv < 0) continue;
            int32_t pv = ssb_fused_ht_probe(p.d_part_ht_keys, p.d_part_ht_values, p.part_ht_mask, partkey);
            if (pv < 0) continue;
            int32_t gi = yi * p.stride_year
                       + cv * (p.supp_dims * p.part_dims)
                       + sv * p.part_dims + pv;
            int64_t profit = (int64_t)revenue - (int64_t)supplycost;
            atomicAdd(reinterpret_cast<unsigned long long int*>(&s_hist[gi]),
                      static_cast<unsigned long long int>(profit));
        }
    }
    __syncthreads();

    for (uint32_t i = tid; i < hs; i += T) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int*>(&p.d_profit[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized 4-field kernel (Q1x/Q2x/Q3x)
//   32 warps (1024 threads): warps 0-3 IO, warps 4-31 decomp (7 groups × 4)
//   BATCH=7, N_BUF=2, SLOTS_PER_BLOCK=56
//   Scan helper selected by ParamType overload resolution
// ════════════════════════════════════════════════════════════════

template <unsigned int PAGE_SIZE_CONST, typename ParamType>
__global__ __launch_bounds__(1024, 1)
void ssb_fused_4f_ws_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    ParamType   p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t BATCH    = 7;
    constexpr uint32_t NF       = 4;
    constexpr uint32_t IO_WARPS = 4;
    constexpr uint32_t SPB      = 28;  // SLOTS_PER_BUF
    constexpr uint32_t SPBLK    = 56;  // SLOTS_PER_BLOCK
    constexpr uint32_t THREADS  = 1024;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    // Shared memory: IO→decomp metadata
    __shared__ uint32_t s_comp_sz[2][BATCH][NF];
    __shared__ uint32_t s_batch_count[2];

    // Active page list (compacted from d_page_mask before IO loop)
    __shared__ uint32_t s_active_pgs[kZonemapMaxPagesPerBlock];
    __shared__ uint32_t s_num_active;

    // Dynamic shared: nvCOMPdx per-decomp-warp (28 warps)
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
    const uint32_t bsb = blockIdx.x * SPBLK;  // block slot base

    // ── Prolog: IO warps read first batch into buf[0] ──
    {
        const uint32_t bc = (BATCH < my_pages) ? BATCH : my_pages;
        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < bc; j++) {
                uint32_t orig_pg = s_active_pgs[j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                ssb_ws_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = bsb + j * NF + field;
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

    // ── Main loop: IO[b] ‖ Decomp[b-1] → Scan[b-1] ──
    uint32_t prev_buf = 0;
    for (uint32_t bstart = BATCH; bstart < my_pages; bstart += BATCH) {
        const uint32_t cur_buf   = 1 - prev_buf;
        const uint32_t rem       = my_pages - bstart;
        const uint32_t cur_count = (BATCH < rem) ? BATCH : rem;
        const uint32_t pc_count  = s_batch_count[prev_buf];

        // Phase A: IO next batch || Decomp previous batch
        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < cur_count; j++) {
                uint32_t orig_pg = s_active_pgs[bstart + j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                ssb_ws_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = bsb + cur_buf * SPB + j * NF + field;
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
            const uint32_t group = dw / NF;
            const uint32_t field = dw % NF;
            if (group < pc_count) {
                uint32_t slot = bsb + prev_buf * SPB + group * NF + field;
                uint32_t csz  = s_comp_sz[prev_buf][group][field];
                char* dst     = d_decomp_buf + (uint64_t)slot * p.page_size;
                bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                    pc_base_addr, slot, dst, csz, p.page_size, my_smem);
            }
        }
        __syncthreads();

        // Phase B: Scan previous batch (all 1024 threads)
        ssb_ws_scan(d_decomp_buf, p, bsb + prev_buf * SPB, pc_count, tid);
        __syncthreads();

        prev_buf = cur_buf;
    }

    // ── Epilog: Decomp + Scan last batch ──
    {
        const uint32_t last_count = s_batch_count[prev_buf];

        // Decomp
        if (warp_id >= IO_WARPS) {
            const uint32_t dw    = warp_id - IO_WARPS;
            const uint32_t group = dw / NF;
            const uint32_t field = dw % NF;
            if (group < last_count) {
                uint32_t slot = bsb + prev_buf * SPB + group * NF + field;
                uint32_t csz  = s_comp_sz[prev_buf][group][field];
                char* dst     = d_decomp_buf + (uint64_t)slot * p.page_size;
                bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                    pc_base_addr, slot, dst, csz, p.page_size, my_smem);
            }
        }
        __syncthreads();

        // Scan
        ssb_ws_scan(d_decomp_buf, p, bsb + prev_buf * SPB, last_count, tid);
    }
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized 6-field kernel (Q4x)
//   32 warps (1024 threads): warps 0-5 IO, warps 6-29 decomp (4 groups × 6)
//   Warps 30-31 idle during IO/decomp, active during scan
//   BATCH=4, N_BUF=2, SLOTS_PER_BLOCK=48
// ════════════════════════════════════════════════════════════════

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(1024, 1)
void ssb_fused_q4x_ws_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    SSBFusedQ4xParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t BATCH      = 4;
    constexpr uint32_t NF         = 6;
    constexpr uint32_t IO_WARPS   = 6;
    constexpr uint32_t DG         = 4;   // decomp groups
    constexpr uint32_t DW_COUNT   = DG * NF;  // 24 decomp warps
    constexpr uint32_t SPB        = 24;  // SLOTS_PER_BUF
    constexpr uint32_t SPBLK      = 48;  // SLOTS_PER_BLOCK
    constexpr uint32_t THREADS    = 1024;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    // Shared memory: IO→decomp metadata
    __shared__ uint32_t s_comp_sz[2][BATCH][NF];
    __shared__ uint32_t s_batch_count[2];

    // Active page list (compacted from d_page_mask before IO loop)
    __shared__ uint32_t s_active_pgs[kZonemapMaxPagesPerBlock];
    __shared__ uint32_t s_num_active;

    // Dynamic shared: nvCOMPdx per-decomp-warp (24 warps)
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = nullptr;
    if (warp_id >= IO_WARPS && warp_id < IO_WARPS + DW_COUNT)
        my_smem = smem + (warp_id - IO_WARPS) * warp_smem;

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
    const uint32_t bsb = blockIdx.x * SPBLK;

    // ── Prolog: IO warps read first batch into buf[0] ──
    {
        const uint32_t bc = (BATCH < my_pages) ? BATCH : my_pages;
        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < bc; j++) {
                uint32_t orig_pg = s_active_pgs[j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                ssb_ws_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = bsb + j * NF + field;
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
        const uint32_t cur_buf   = 1 - prev_buf;
        const uint32_t rem       = my_pages - bstart;
        const uint32_t cur_count = (BATCH < rem) ? BATCH : rem;
        const uint32_t pc_count  = s_batch_count[prev_buf];

        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < cur_count; j++) {
                uint32_t orig_pg = s_active_pgs[bstart + j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                ssb_ws_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = bsb + cur_buf * SPB + j * NF + field;
                if (lane == 0) {
                    bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
                    s_comp_sz[cur_buf][j][field] = comp_sz;
                }
                __syncwarp();
            }
            if (warp_id == 0 && lane == 0)
                s_batch_count[cur_buf] = cur_count;
        } else if (warp_id < IO_WARPS + DW_COUNT) {
            const uint32_t dw    = warp_id - IO_WARPS;
            const uint32_t group = dw / NF;
            const uint32_t field = dw % NF;
            if (group < pc_count) {
                uint32_t slot = bsb + prev_buf * SPB + group * NF + field;
                uint32_t csz  = s_comp_sz[prev_buf][group][field];
                char* dst     = d_decomp_buf + (uint64_t)slot * p.page_size;
                bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                    pc_base_addr, slot, dst, csz, p.page_size, my_smem);
            }
        }
        // warps 30-31: idle during IO/decomp
        __syncthreads();

        // Scan previous batch (all 1024 threads)
        ssb_q4x_scan(d_decomp_buf, p, bsb + prev_buf * SPB, pc_count, tid);
        __syncthreads();

        prev_buf = cur_buf;
    }

    // ── Epilog ──
    {
        const uint32_t last_count = s_batch_count[prev_buf];

        if (warp_id >= IO_WARPS && warp_id < IO_WARPS + DW_COUNT) {
            const uint32_t dw    = warp_id - IO_WARPS;
            const uint32_t group = dw / NF;
            const uint32_t field = dw % NF;
            if (group < last_count) {
                uint32_t slot = bsb + prev_buf * SPB + group * NF + field;
                uint32_t csz  = s_comp_sz[prev_buf][group][field];
                char* dst     = d_decomp_buf + (uint64_t)slot * p.page_size;
                bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                    pc_base_addr, slot, dst, csz, p.page_size, my_smem);
            }
        }
        __syncthreads();

        ssb_q4x_scan(d_decomp_buf, p, bsb + prev_buf * SPB, last_count, tid);
    }
}

// ════════════════════════════════════════════════════════════════
// Host API: Context management
// ════════════════════════════════════════════════════════════════

struct SSBFusedContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

ssb_fused_ctx_t ssb_fused_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks,
    uint32_t slots_per_block)
{
    auto* ctx = new SSBFusedContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * slots_per_block;

    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    size_t decomp_size = (size_t)num_slots * page_size;
    SSB_FUSED_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<ssb_fused_ctx_t>(ctx);
}

void* ssb_fused_get_d_ctrls(ssb_fused_ctx_t h) {
    return static_cast<SSBFusedContext*>(h)->d_ctrls;
}
void* ssb_fused_get_d_pc_ptr(ssb_fused_ctx_t h) {
    return static_cast<SSBFusedContext*>(h)->d_pc_ptr;
}
const char* ssb_fused_get_pc_base(ssb_fused_ctx_t h) {
    return static_cast<SSBFusedContext*>(h)->pc_base_addr;
}
char* ssb_fused_get_decomp_buf(ssb_fused_ctx_t h) {
    return static_cast<SSBFusedContext*>(h)->d_decomp_buf;
}

void ssb_fused_destroy(ssb_fused_ctx_t h) {
    auto* ctx = static_cast<SSBFusedContext*>(h);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ════════════════════════════════════════════════════════════════
// Host API: Max co-resident blocks
// ════════════════════════════════════════════════════════════════

uint32_t ssb_fused_q1x_max_blocks(uint32_t page_size)
{
    int max_bpsm = 0;
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = SSB_WS4_DECOMP_GROUPS * SSB_WS4_NUM_FIELDS;  // 28

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = ssb_fused_4f_ws_kernel<PS, SSBFusedQ1xParams>;
        SSB_FUSED_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bpsm, kfn, THREADS, smem_size);
    });

    int device; cudaGetDevice(&device);
    int sm_count; cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    uint32_t total = (uint32_t)max_bpsm * (uint32_t)sm_count;
    fprintf(stderr, "[ssb_fused_q1x] max_bpsm=%d sm=%d total=%u\n", max_bpsm, sm_count, total);
    return total;
}

uint32_t ssb_fused_q4x_max_blocks(uint32_t page_size)
{
    int max_bpsm = 0;
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DECOMP_WARPS = SSB_WS6_DECOMP_GROUPS * SSB_WS6_NUM_FIELDS;  // 24

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * DECOMP_WARPS;
        auto kfn = ssb_fused_q4x_ws_kernel<PS>;
        fprintf(stderr, "[ssb_fused_q4x_max_blocks] PS=%u decomp_smem=%zu (%zu per warp * %u warps)\n",
                PS, smem_size, bam_lz4_io_decomp_smem_per_warp<PS>(), DECOMP_WARPS);
        SSB_FUSED_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bpsm, kfn, THREADS, smem_size);
    });

    int device; cudaGetDevice(&device);
    int sm_count; cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    uint32_t total = (uint32_t)max_bpsm * (uint32_t)sm_count;
    fprintf(stderr, "[ssb_fused_q4x] max_bpsm=%d sm=%d total=%u\n", max_bpsm, sm_count, total);
    return total;
}

// ════════════════════════════════════════════════════════════════
// Host API: Kernel launch functions
// ════════════════════════════════════════════════════════════════

void ssb_fused_q1x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ1xParams& p,
    uint32_t num_blocks, cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DW = SSB_WS4_DECOMP_GROUPS * SSB_WS4_NUM_FIELDS;
    bool all_uncomp = true;
    for (int i = 0; i < 4; i++) if (p.is_compressed[i]) all_uncomp = false;
    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem = all_uncomp ? 0 : bam_lz4_io_decomp_smem_per_warp<PS>() * DW;
        auto kfn = ssb_fused_4f_ws_kernel<PS, SSBFusedQ1xParams>;
        if (!all_uncomp)
            SSB_FUSED_CHECK(cudaFuncSetAttribute(
                kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<num_blocks, THREADS, smem, stream>>>(
            d_ctrls, d_pc_ptr, pc_base, d_decomp_buf, p);
    });
    SSB_FUSED_CHECK(cudaGetLastError());
}

void ssb_fused_q2x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ2xParams& p,
    uint32_t num_blocks, cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DW = SSB_WS4_DECOMP_GROUPS * SSB_WS4_NUM_FIELDS;
    bool all_uncomp = true;
    for (int i = 0; i < 4; i++) if (p.is_compressed[i]) all_uncomp = false;
    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem = all_uncomp ? 0 : bam_lz4_io_decomp_smem_per_warp<PS>() * DW;
        auto kfn = ssb_fused_4f_ws_kernel<PS, SSBFusedQ2xParams>;
        if (!all_uncomp)
            SSB_FUSED_CHECK(cudaFuncSetAttribute(
                kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<num_blocks, THREADS, smem, stream>>>(
            d_ctrls, d_pc_ptr, pc_base, d_decomp_buf, p);
    });
    SSB_FUSED_CHECK(cudaGetLastError());
}

void ssb_fused_q3x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ3xParams& p,
    uint32_t num_blocks, cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DW = SSB_WS4_DECOMP_GROUPS * SSB_WS4_NUM_FIELDS;
    bool all_uncomp = true;
    for (int i = 0; i < 4; i++) if (p.is_compressed[i]) all_uncomp = false;
    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t decomp_smem = all_uncomp ? 0 : bam_lz4_io_decomp_smem_per_warp<PS>() * DW;
        size_t hist_smem = (size_t)p.hist_size * sizeof(int64_t);
        size_t smem = (decomp_smem > hist_smem) ? decomp_smem : hist_smem;
        auto kfn = ssb_fused_4f_ws_kernel<PS, SSBFusedQ3xParams>;
        SSB_FUSED_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<num_blocks, THREADS, smem, stream>>>(
            d_ctrls, d_pc_ptr, pc_base, d_decomp_buf, p);
    });
    SSB_FUSED_CHECK(cudaGetLastError());
}

void ssb_fused_q4x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ4xParams& p,
    uint32_t num_blocks, cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    constexpr uint32_t DW = SSB_WS6_DECOMP_GROUPS * SSB_WS6_NUM_FIELDS;
    bool all_uncomp = true;
    for (int i = 0; i < 6; i++) if (p.is_compressed[i]) all_uncomp = false;
    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t decomp_smem = all_uncomp ? 0 : bam_lz4_io_decomp_smem_per_warp<PS>() * DW;
        size_t hist_smem = (size_t)p.hist_size * sizeof(int64_t);
        size_t smem = (decomp_smem > hist_smem) ? decomp_smem : hist_smem;
        fprintf(stderr, "[ssb_fused_q4x_launch] PS=%u decomp_smem=%zu hist_smem=%zu (hist_size=%u) smem=%zu all_uncomp=%d num_blocks=%u\n",
                PS, decomp_smem, hist_smem, p.hist_size, smem, all_uncomp, num_blocks);
        auto kfn = ssb_fused_q4x_ws_kernel<PS>;
        SSB_FUSED_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<num_blocks, THREADS, smem, stream>>>(
            d_ctrls, d_pc_ptr, pc_base, d_decomp_buf, p);
    });
    SSB_FUSED_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Dim table fused IO + nvCOMPdx LZ4 decomp kernel
//
// 1 warp (32 threads) per block, persistent loop over pages.
// Each iteration: BaM read (lane 0) + nvCOMPdx decomp (all lanes).
// Replaces the 2-kernel path (batch_read + nvCOMP host decomp).
// ════════════════════════════════════════════════════════════════

namespace {
// Layout-compatible with BAMBatchReadEntry {uint64_t lba; uint32_t dev; uint32_t nblk}
struct DimIOEntry {
    uint64_t lba;
    uint32_t dev;
    uint32_t nblk;
};
}

template <unsigned int PS>
__global__ void bam_dim_io_decomp_kernel(
    void* ctrls, void* pc, void* pc_base,
    const DimIOEntry* entries,
    const uint32_t* comp_sizes,
    char* d_output,
    uint32_t total_pages,
    uint32_t page_size,
    uint32_t slot_base)
{
    extern __shared__ uint8_t smem[];
    for (uint32_t pg = blockIdx.x; pg < total_pages; pg += gridDim.x) {
        bam_lz4_io_decomp_warp<PS>(
            ctrls, pc, pc_base,
            slot_base + blockIdx.x,
            d_output + (uint64_t)pg * page_size,
            entries[pg].lba,
            entries[pg].nblk,
            entries[pg].dev,
            comp_sizes[pg],
            page_size,
            smem);
    }
}

void bam_dim_io_decomp_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    uint32_t slot_base, uint32_t max_slots, uint32_t page_size,
    char* d_output,
    const void* d_entries,
    const uint32_t* d_comp_sizes,
    uint32_t total_pages,
    cudaStream_t stream)
{
    if (total_pages == 0) return;
    uint32_t num_blocks = (total_pages < max_slots) ? total_pages : max_slots;
    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem = bam_lz4_io_decomp_smem_per_warp<PS>();
        auto kfn = bam_dim_io_decomp_kernel<PS>;
        SSB_FUSED_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<num_blocks, 32, smem, stream>>>(
            d_ctrls, d_pc_ptr, (void*)pc_base,
            static_cast<const DimIOEntry*>(d_entries),
            d_comp_sizes,
            d_output,
            total_pages, page_size, slot_base);
    });
    SSB_FUSED_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Dim table fused IO + block-wide copy kernel (for NONE fields)
//
// 256 threads per block, persistent loop over pages.
// Thread 0: BaM IO read, all 256 threads: copy from slot → d_output.
// ════════════════════════════════════════════════════════════════

__global__ void bam_dim_io_copy_kernel(
    void* ctrls, void* pc, void* pc_base,
    const DimIOEntry* entries,
    char* d_output,
    uint32_t total_pages,
    uint32_t page_size,
    uint32_t slot_base)
{
    const uint32_t tid = threadIdx.x;
    for (uint32_t pg = blockIdx.x; pg < total_pages; pg += gridDim.x) {
        const uint32_t slot = slot_base + blockIdx.x;
        if (tid == 0)
            bam_io_read_page_device(ctrls, pc,
                entries[pg].lba, entries[pg].nblk, slot, entries[pg].dev);
        __syncthreads();
        const uint32_t* src = reinterpret_cast<const uint32_t*>(
            (const char*)pc_base + (uint64_t)slot * page_size);
        uint32_t* dst = reinterpret_cast<uint32_t*>(
            d_output + (uint64_t)pg * page_size);
        const uint32_t n4 = page_size / 4;
        for (uint32_t i = tid; i < n4; i += blockDim.x)
            dst[i] = src[i];
        __syncthreads();
    }
}

void bam_dim_io_copy_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    uint32_t slot_base, uint32_t max_slots, uint32_t page_size,
    char* d_output,
    const void* d_entries,
    uint32_t total_pages,
    cudaStream_t stream)
{
    if (total_pages == 0) return;
    uint32_t num_blocks = (total_pages < max_slots) ? total_pages : max_slots;
    bam_dim_io_copy_kernel<<<num_blocks, 256, 0, stream>>>(
        d_ctrls, d_pc_ptr, (void*)pc_base,
        static_cast<const DimIOEntry*>(d_entries),
        d_output,
        total_pages, page_size, slot_base);
    SSB_FUSED_CHECK(cudaGetLastError());
}
