// bam_pfor_fused_ssb.cu — SSB PFOR Warp-Spec Fused BaM I/O + PFOR decomp + scan
//
// Mirror of bam_lz4_fused_ssb.cu but replacing nvCOMPdx LZ4 with PFOR.
// Much less shared memory → __launch_bounds__(1024, 1) → 2x occupancy.
//
// Warp layout (same as LZ4):
//   Q1x/Q2x/Q3x: 4 IO warps + 7 decomp groups × 4 fields = 32 warps
//   Q4x:          6 IO warps + 4 decomp groups × 6 fields = 30 warps (+2 idle)
//
// Pipeline: Prolog(IO batch 0) → Main(IO[b] ‖ Decomp[b-1] → Scan[b-1]) → Epilog

#include "bam_pfor_fused_ssb.cuh"
#include "bam_pfor_decomp_warp.cuh"
#include "tpch/bam_io_device.cuh"

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

#define PFOR_FUSED_CHECK(call) do {                                           \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// ════════════════════════════════════════════════════════════════
// Device helpers (same as bam_lz4_fused_ssb.cu)
// ════════════════════════════════════════════════════════════════

// PRP2 danger zone fix: BaM 9-16 block reads are unsafe → skip to 24
__device__ static uint32_t pfor_fused_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// IO parameter computation
template <typename P>
__device__ __forceinline__
static void pfor_ws_io_params(
    const P& p, uint32_t field, uint32_t orig_pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.field_start_page_ids[field] + orig_pg;
    dev = global_pg % ndev;
    if (p.is_compressed[field]) {
        lba = p.partition_start_lbas[dev] +
              p.d_comp_offsets[field][orig_pg] / 512;
        comp_sz = p.d_comp_sizes[field][orig_pg];
        nblk = pfor_fused_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

// Hash table probe
__device__ static uint32_t pfor_fused_hash32(uint32_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

__device__ static int32_t pfor_fused_ht_probe(
    const int32_t* keys, const int32_t* values, uint32_t mask, int32_t key)
{
    uint32_t slot = pfor_fused_hash32((uint32_t)key) & mask;
    while (true) {
        int32_t k = keys[slot];
        if (k == key) return values[slot];
        if (k == -1) return -1;
        slot = (slot + 1) & mask;
    }
}

// Dimension constants
static constexpr int32_t  PFOR_YEAR_MIN      = 1992;
static constexpr int32_t  PFOR_MAX_BRANDS    = 40;
static constexpr int32_t  PFOR_MAX_YEARS     = 7;

// ════════════════════════════════════════════════════════════════
// Scan helpers — overloaded by params type for template dispatch
// ════════════════════════════════════════════════════════════════

// Q1x scan: date HT probe + disc/qty filter → SUM(extprice * discount)
// Hierarchical reduction: shfl_down + shared → 1 atomicAdd/block
__device__ __forceinline__
static void pfor_ws_scan(
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
            bool dm = (pfor_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values,
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
static void pfor_ws_scan(
    const char* d_decomp_buf, const SSBFusedQ2xParams& p,
    uint32_t base_slot, uint32_t count, uint32_t tid)
{
    constexpr uint32_t T = 1024, NF = 4, HDR = 3;
    constexpr uint32_t HIST_SIZE = PFOR_MAX_YEARS * PFOR_MAX_BRANDS;  // 7 * 40 = 280
    __shared__ int64_t s_hist[HIST_SIZE];

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
            int32_t yi = pfor_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, orderdate);
            if (yi < 0) continue;
            int32_t sv = pfor_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, suppkey);
            if (sv < 0) continue;
            int32_t bi = pfor_fused_ht_probe(p.d_part_ht_keys, p.d_part_ht_values, p.part_ht_mask, partkey);
            if (bi < 0) continue;
            int32_t gi = yi * PFOR_MAX_BRANDS + bi;
            atomicAdd(reinterpret_cast<unsigned long long int*>(&s_hist[gi]),
                      static_cast<unsigned long long int>((int64_t)revenue));
        }
    }
    __syncthreads();

    for (uint32_t i = tid; i < HIST_SIZE; i += T) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int*>(&p.d_revenue[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
    __syncthreads();
}

// Q3x scan: date year + cust HT + supp HT → revenue[cust × supp × year]
// Uses dynamic shared memory for variable-size histogram
__device__ __forceinline__
static void pfor_ws_scan(
    const char* d_decomp_buf, const SSBFusedQ3xParams& p,
    uint32_t base_slot, uint32_t count, uint32_t tid)
{
    extern __shared__ __align__(8) uint8_t pfor_smem[];
    int64_t* s_hist = reinterpret_cast<int64_t*>(pfor_smem);
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
            int32_t yi = pfor_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, orderdate);
            if (yi < 0) continue;
            int32_t cd = pfor_fused_ht_probe(p.d_cust_ht_keys, p.d_cust_ht_values, p.cust_ht_mask, custkey);
            if (cd < 0) continue;
            int32_t sd = pfor_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, suppkey);
            if (sd < 0) continue;
            int32_t gi = cd * p.num_supp_dims * PFOR_MAX_YEARS + sd * PFOR_MAX_YEARS + yi;
            atomicAdd(reinterpret_cast<unsigned long long int*>(&s_hist[gi]),
                      static_cast<unsigned long long int>((int64_t)revenue));
        }
    }
    __syncthreads();

    for (uint32_t i = tid; i < hs; i += T) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int*>(&p.d_revenue[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
}

// Q4x scan: 6 fields, date year + 3 HTs → profit[year × cust × supp × part]
__device__ __forceinline__
static void pfor_q4x_scan(
    const char* d_decomp_buf, const SSBFusedQ4xParams& p,
    uint32_t base_slot, uint32_t count, uint32_t tid)
{
    extern __shared__ __align__(8) uint8_t pfor_smem[];
    int64_t* s_hist = reinterpret_cast<int64_t*>(pfor_smem);
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
            int32_t yi = pfor_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, orderdate);
            if (yi < 0) continue;
            int32_t cv = pfor_fused_ht_probe(p.d_cust_ht_keys, p.d_cust_ht_values, p.cust_ht_mask, custkey);
            if (cv < 0) continue;
            int32_t sv = pfor_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, suppkey);
            if (sv < 0) continue;
            int32_t pv = pfor_fused_ht_probe(p.d_part_ht_keys, p.d_part_ht_values, p.part_ht_mask, partkey);
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
// Warp-Specialized 4-field PFOR kernel (Q1x/Q2x/Q3x)
//   32 warps (1024 threads): warps 0-3 IO, warps 4-31 decomp (7 groups × 4)
//   BATCH=7, N_BUF=2, SLOTS_PER_BLOCK=56
//   __launch_bounds__(1024, 1): PFOR needs ~1.4 KB smem → 2 blocks/SM
// ════════════════════════════════════════════════════════════════

template <typename ParamType>
__global__ __launch_bounds__(1024, 1)
void ssb_pfor_4f_ws_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    ParamType   p)
{
    constexpr uint32_t BATCH    = 7;
    constexpr uint32_t NF       = 4;
    constexpr uint32_t IO_WARPS = 4;
    constexpr uint32_t SPB      = 28;  // SLOTS_PER_BUF
    constexpr uint32_t SPBLK    = 56;  // SLOTS_PER_BLOCK

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    // Shared memory: PFOR bitwidths/offsets per decomp warp + batch metadata
    __shared__ uint32_t s_batch_count[2];
    __shared__ uint s_pfor_bws[28][4];   // 28 decomp warps × 4 miniblocks
    __shared__ uint s_pfor_offs[28][4];
    __shared__ uint32_t s_active_pgs[kZonemapMaxPagesPerBlock];
    __shared__ uint32_t s_num_active;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

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
                pfor_ws_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = bsb + j * NF + field;
                if (lane == 0)
                    bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
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

        // Phase A: IO next batch || PFOR decomp previous batch
        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < cur_count; j++) {
                uint32_t orig_pg = s_active_pgs[bstart + j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                pfor_ws_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = bsb + cur_buf * SPB + j * NF + field;
                if (lane == 0)
                    bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
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
                char* dst     = d_decomp_buf + (uint64_t)slot * p.page_size;
                bam_pfor_decomp_warp(
                    pc_base_addr, slot, dst, p.page_size,
                    p.is_compressed[field],
                    s_pfor_bws[dw], s_pfor_offs[dw]);
            }
        }
        __syncthreads();

        // Phase B: Scan previous batch (all 1024 threads)
        pfor_ws_scan(d_decomp_buf, p, bsb + prev_buf * SPB, pc_count, tid);
        __syncthreads();

        prev_buf = cur_buf;
    }

    // ── Epilog: Decomp + Scan last batch ──
    {
        const uint32_t last_count = s_batch_count[prev_buf];

        if (warp_id >= IO_WARPS) {
            const uint32_t dw    = warp_id - IO_WARPS;
            const uint32_t group = dw / NF;
            const uint32_t field = dw % NF;
            if (group < last_count) {
                uint32_t slot = bsb + prev_buf * SPB + group * NF + field;
                char* dst     = d_decomp_buf + (uint64_t)slot * p.page_size;
                bam_pfor_decomp_warp(
                    pc_base_addr, slot, dst, p.page_size,
                    p.is_compressed[field],
                    s_pfor_bws[dw], s_pfor_offs[dw]);
            }
        }
        __syncthreads();

        pfor_ws_scan(d_decomp_buf, p, bsb + prev_buf * SPB, last_count, tid);
    }
}

// ════════════════════════════════════════════════════════════════
// Warp-Specialized 6-field PFOR kernel (Q4x)
//   32 warps (1024 threads): warps 0-5 IO, warps 6-29 decomp (4 groups × 6)
//   Warps 30-31 idle during IO/decomp, active during scan
//   BATCH=4, N_BUF=2, SLOTS_PER_BLOCK=48
// ════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(1024, 1)
void ssb_pfor_q4x_ws_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    SSBFusedQ4xParams p)
{
    constexpr uint32_t BATCH      = 4;
    constexpr uint32_t NF         = 6;
    constexpr uint32_t IO_WARPS   = 6;
    constexpr uint32_t DG         = 4;   // decomp groups
    constexpr uint32_t DW_COUNT   = DG * NF;  // 24 decomp warps
    constexpr uint32_t SPB        = 24;  // SLOTS_PER_BUF
    constexpr uint32_t SPBLK      = 48;  // SLOTS_PER_BLOCK

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    __shared__ uint32_t s_batch_count[2];
    __shared__ uint s_pfor_bws[24][4];   // 24 decomp warps × 4 miniblocks
    __shared__ uint s_pfor_offs[24][4];
    __shared__ uint32_t s_active_pgs[kZonemapMaxPagesPerBlock];
    __shared__ uint32_t s_num_active;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

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

    // ── Prolog ──
    {
        const uint32_t bc = (BATCH < my_pages) ? BATCH : my_pages;
        if (warp_id < IO_WARPS) {
            const uint32_t field = warp_id;
            for (uint32_t j = 0; j < bc; j++) {
                uint32_t orig_pg = s_active_pgs[j];
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                pfor_ws_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = bsb + j * NF + field;
                if (lane == 0)
                    bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
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
                pfor_ws_io_params(p, field, orig_pg, ndev, lba, nblk, dev, comp_sz);
                uint32_t slot = bsb + cur_buf * SPB + j * NF + field;
                if (lane == 0)
                    bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
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
                char* dst     = d_decomp_buf + (uint64_t)slot * p.page_size;
                bam_pfor_decomp_warp(
                    pc_base_addr, slot, dst, p.page_size,
                    p.is_compressed[field],
                    s_pfor_bws[dw], s_pfor_offs[dw]);
            }
        }
        __syncthreads();

        pfor_q4x_scan(d_decomp_buf, p, bsb + prev_buf * SPB, pc_count, tid);
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
                char* dst     = d_decomp_buf + (uint64_t)slot * p.page_size;
                bam_pfor_decomp_warp(
                    pc_base_addr, slot, dst, p.page_size,
                    p.is_compressed[field],
                    s_pfor_bws[dw], s_pfor_offs[dw]);
            }
        }
        __syncthreads();

        pfor_q4x_scan(d_decomp_buf, p, bsb + prev_buf * SPB, last_count, tid);
    }
}

// ════════════════════════════════════════════════════════════════
// Host API: Max co-resident blocks
// ════════════════════════════════════════════════════════════════

uint32_t ssb_pfor_fused_q1x_max_blocks(uint32_t page_size)
{
    int max_bpsm = 0;
    constexpr uint32_t THREADS = 1024;

    // PFOR uses only static shared memory — no dynamic smem
    auto kfn = ssb_pfor_4f_ws_kernel<SSBFusedQ1xParams>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bpsm, kfn, THREADS, 0);

    int device; cudaGetDevice(&device);
    int sm_count; cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    uint32_t total = (uint32_t)max_bpsm * (uint32_t)sm_count;
    fprintf(stderr, "[ssb_pfor_fused_q1x] max_bpsm=%d sm=%d total=%u\n",
            max_bpsm, sm_count, total);
    return total;
}

uint32_t ssb_pfor_fused_q4x_max_blocks(uint32_t page_size)
{
    int max_bpsm = 0;
    constexpr uint32_t THREADS = 1024;

    auto kfn = ssb_pfor_q4x_ws_kernel;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bpsm, kfn, THREADS, 0);

    int device; cudaGetDevice(&device);
    int sm_count; cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    uint32_t total = (uint32_t)max_bpsm * (uint32_t)sm_count;
    fprintf(stderr, "[ssb_pfor_fused_q4x] max_bpsm=%d sm=%d total=%u\n",
            max_bpsm, sm_count, total);
    return total;
}

// ════════════════════════════════════════════════════════════════
// Host API: Kernel launch functions
// ════════════════════════════════════════════════════════════════

void ssb_pfor_fused_q1x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ1xParams& p,
    uint32_t num_blocks, cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    // Q1x: no dynamic shared memory (histogram fits in static s_warp_sums)
    ssb_pfor_4f_ws_kernel<SSBFusedQ1xParams>
        <<<num_blocks, THREADS, 0, stream>>>(
            d_ctrls, d_pc_ptr, pc_base, d_decomp_buf, p);
    PFOR_FUSED_CHECK(cudaGetLastError());
}

void ssb_pfor_fused_q2x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ2xParams& p,
    uint32_t num_blocks, cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    // Q2x: histogram is static __shared__ s_hist[280], no dynamic smem
    ssb_pfor_4f_ws_kernel<SSBFusedQ2xParams>
        <<<num_blocks, THREADS, 0, stream>>>(
            d_ctrls, d_pc_ptr, pc_base, d_decomp_buf, p);
    PFOR_FUSED_CHECK(cudaGetLastError());
}

void ssb_pfor_fused_q3x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ3xParams& p,
    uint32_t num_blocks, cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    // Q3x: variable-size histogram → dynamic shared memory
    size_t smem = (size_t)p.hist_size * sizeof(int64_t);
    auto kfn = ssb_pfor_4f_ws_kernel<SSBFusedQ3xParams>;
    PFOR_FUSED_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kfn<<<num_blocks, THREADS, smem, stream>>>(
        d_ctrls, d_pc_ptr, pc_base, d_decomp_buf, p);
    PFOR_FUSED_CHECK(cudaGetLastError());
}

void ssb_pfor_fused_q4x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ4xParams& p,
    uint32_t num_blocks, cudaStream_t stream)
{
    constexpr uint32_t THREADS = 1024;
    // Q4x: variable-size histogram → dynamic shared memory
    size_t smem = (size_t)p.hist_size * sizeof(int64_t);
    auto kfn = ssb_pfor_q4x_ws_kernel;
    PFOR_FUSED_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kfn<<<num_blocks, THREADS, smem, stream>>>(
        d_ctrls, d_pc_ptr, pc_base, d_decomp_buf, p);
    PFOR_FUSED_CHECK(cudaGetLastError());
}
