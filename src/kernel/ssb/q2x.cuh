#pragma once

#include <cub/cub.cuh>
#include "common/pag_core.h"

// SSB Flight 2: Q2.1, Q2.2, Q2.3
// SELECT SUM(lo_revenue), d_year, p_brand1
// FROM lineorder, ddate, part, supplier
// WHERE lo_orderdate = d_datekey
//   AND lo_partkey = p_partkey
//   AND lo_suppkey = s_suppkey
//   AND <part_filter>
//   AND <supplier_filter>
// GROUP BY d_year, p_brand1
// ORDER BY d_year, p_brand1
//
// Q2.1: p_category='MFGR#12', s_region='AMERICA'
// Q2.2: p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228', s_region='ASIA'
// Q2.3: p_brand1 = 'MFGR#2221', s_region='EUROPE'

// ── Hash table primitives ──────────────────────────────────────
static constexpr int32_t SSB_HT_EMPTY = -1;

__device__ __forceinline__ uint32_t ssb_hash32(uint32_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

__device__ __forceinline__ void ssb_ht_insert(
    int32_t *keys, int32_t *values, uint32_t mask,
    int32_t key, int32_t value)
{
    uint32_t slot = ssb_hash32((uint32_t)key) & mask;
    while (true) {
        int32_t old = atomicCAS(&keys[slot], SSB_HT_EMPTY, key);
        if (old == SSB_HT_EMPTY || old == key) {
            values[slot] = value;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

__device__ __forceinline__ int32_t ssb_ht_probe(
    const int32_t *keys, const int32_t *values, uint32_t mask,
    int32_t key)
{
    uint32_t slot = ssb_hash32((uint32_t)key) & mask;
    while (true) {
        int32_t k = keys[slot];
        if (k == key) return values[slot];
        if (k == SSB_HT_EMPTY) return -1;
        slot = (slot + 1) & mask;
    }
}

// Year range: 1992..1998 → indices 0..6
static constexpr int32_t SSB_YEAR_MIN = 1992;
static constexpr int32_t SSB_YEAR_MAX = 1998;
static constexpr int32_t SSB_NUM_YEARS = SSB_YEAR_MAX - SSB_YEAR_MIN + 1;

// Max brand1 values per category: MFGR#X21..MFGR#X240 = 40 brands
static constexpr int32_t SSB_MAX_BRANDS = 40;

// Group-by buffer size for Q2x: NUM_YEARS * MAX_BRANDS
static constexpr int32_t SSB_Q2X_GROUPS = SSB_NUM_YEARS * SSB_MAX_BRANDS;

// ── Q2x LINEORDER probe kernel (flat array version) ──────────
static constexpr int SSB_Q2X_BLOCK_SIZE = 128;

__global__ void ssb_q2x_probe_flat_kernel(
    const uint64_t *lo_orderdate,
    const uint64_t *lo_partkey,
    const uint64_t *lo_suppkey,
    const uint64_t *lo_revenue,
    uint64_t nrecs,
    // Date hash table: datekey → year_idx (0-based), -1 if not found
    const int32_t *d_date_ht_keys,
    const int32_t *d_date_ht_values,
    uint32_t date_ht_mask,
    // Supplier hash set: suppkey → 1 if valid, -1 if not
    const int32_t *d_supp_ht_keys,
    const int32_t *d_supp_ht_values,
    uint32_t supp_ht_mask,
    // Part hash table: partkey → brand1_idx (0..39), -1 if filtered
    const int32_t *d_part_ht_keys,
    const int32_t *d_part_ht_values,
    uint32_t part_ht_mask,
    // Output: revenue[year_idx * MAX_BRANDS + brand1_idx]
    int64_t *d_revenue)
{
    constexpr uint32_t HIST_SIZE = SSB_NUM_YEARS * SSB_MAX_BRANDS;  // 7 * 40 = 280
    __shared__ int64_t s_hist[HIST_SIZE];

    for (uint32_t i = threadIdx.x; i < HIST_SIZE; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrecs) {
        int32_t orderdate = (int32_t)lo_orderdate[idx];
        int32_t partkey   = (int32_t)lo_partkey[idx];
        int32_t suppkey   = (int32_t)lo_suppkey[idx];
        int32_t revenue   = (int32_t)lo_revenue[idx];

        int32_t year_idx = ssb_ht_probe(d_date_ht_keys, d_date_ht_values, date_ht_mask, orderdate);
        if (year_idx >= 0) {
            int32_t supp_val = ssb_ht_probe(d_supp_ht_keys, d_supp_ht_values, supp_ht_mask, suppkey);
            if (supp_val >= 0) {
                int32_t brand1_idx = ssb_ht_probe(d_part_ht_keys, d_part_ht_values, part_ht_mask, partkey);
                if (brand1_idx >= 0) {
                    int32_t group_idx = year_idx * SSB_MAX_BRANDS + brand1_idx;
                    atomicAdd(reinterpret_cast<unsigned long long int *>(&s_hist[group_idx]),
                              static_cast<unsigned long long int>((int64_t)revenue));
                }
            }
        }
    }
    __syncthreads();

    // Flush shared histogram to global
    for (uint32_t i = threadIdx.x; i < HIST_SIZE; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int *>(&d_revenue[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
}

static cudaError_t ssb_q2x_probe_flat(
    const uint64_t *lo_orderdate,
    const uint64_t *lo_partkey,
    const uint64_t *lo_suppkey,
    const uint64_t *lo_revenue,
    uint64_t nrecs,
    const int32_t *d_date_ht_keys,
    const int32_t *d_date_ht_values,
    uint32_t date_ht_mask,
    const int32_t *d_supp_ht_keys,
    const int32_t *d_supp_ht_values,
    uint32_t supp_ht_mask,
    const int32_t *d_part_ht_keys,
    const int32_t *d_part_ht_values,
    uint32_t part_ht_mask,
    int64_t *d_revenue,
    cudaStream_t stream)
{
    constexpr int BLK = SSB_Q2X_BLOCK_SIZE;
    int grid_dim = (nrecs + BLK - 1) / BLK;

    ssb_q2x_probe_flat_kernel<<<grid_dim, BLK, 0, stream>>>(
        lo_orderdate, lo_partkey, lo_suppkey, lo_revenue,
        nrecs,
        d_date_ht_keys, d_date_ht_values, date_ht_mask,
        d_supp_ht_keys, d_supp_ht_values, supp_ht_mask,
        d_part_ht_keys, d_part_ht_values, part_ht_mask,
        d_revenue);

    return cudaSuccess;
}
