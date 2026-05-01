#pragma once

#include "common/pag_core.h"

// SSB Flight 4: Q4.1, Q4.2, Q4.3 (flat array version)
// SUM(lo_revenue - lo_supplycost) as profit

static constexpr int32_t SSB_Q4X_MAX_YEARS = 7;
static constexpr int32_t SSB_Q4X_MAX_CUST_DIMS = 5;
static constexpr int32_t SSB_Q4X_MAX_SUPP_DIMS = 10;
static constexpr int32_t SSB_Q4X_MAX_PART_DIMS = 40;
static constexpr int32_t SSB_Q4X_MAX_GROUPS =
    SSB_Q4X_MAX_YEARS * SSB_Q4X_MAX_CUST_DIMS * SSB_Q4X_MAX_SUPP_DIMS * SSB_Q4X_MAX_PART_DIMS;

static constexpr int SSB_Q4X_BLOCK_SIZE = 128;

__global__ void ssb_q4x_probe_flat_kernel(
    const uint64_t *lo_orderdate,
    const uint64_t *lo_custkey,
    const uint64_t *lo_partkey,
    const uint64_t *lo_suppkey,
    const uint64_t *lo_revenue,
    const uint64_t *lo_supplycost,
    uint64_t nrecs,
    // Date HT: datekey → year_idx (0-based), -1 if not found
    const int32_t *d_date_ht_keys,
    const int32_t *d_date_ht_values,
    uint32_t date_ht_mask,
    // Customer HT
    const int32_t *d_cust_ht_keys,
    const int32_t *d_cust_ht_values,
    uint32_t cust_ht_mask,
    // Supplier HT
    const int32_t *d_supp_ht_keys,
    const int32_t *d_supp_ht_values,
    uint32_t supp_ht_mask,
    // Part HT
    const int32_t *d_part_ht_keys,
    const int32_t *d_part_ht_values,
    uint32_t part_ht_mask,
    // Group-by dimension strides
    int32_t supp_dims,
    int32_t part_dims,
    int32_t stride_year,  // = cust_dims * supp_dims * part_dims
    uint32_t hist_size,   // = MAX_YEARS * stride_year
    // Output
    int64_t *d_profit)
{
    extern __shared__ int64_t s_hist[];
    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrecs) {
        int32_t orderdate  = (int32_t)lo_orderdate[idx];
        int32_t custkey    = (int32_t)lo_custkey[idx];
        int32_t partkey    = (int32_t)lo_partkey[idx];
        int32_t suppkey    = (int32_t)lo_suppkey[idx];
        int32_t revenue    = (int32_t)lo_revenue[idx];
        int32_t supplycost = (int32_t)lo_supplycost[idx];

        int32_t year_idx = ssb_ht_probe(d_date_ht_keys, d_date_ht_values, date_ht_mask, orderdate);
        if (year_idx >= 0) {
            int32_t cust_val = ssb_ht_probe(d_cust_ht_keys, d_cust_ht_values, cust_ht_mask, custkey);
            if (cust_val >= 0) {
                int32_t supp_val = ssb_ht_probe(d_supp_ht_keys, d_supp_ht_values, supp_ht_mask, suppkey);
                if (supp_val >= 0) {
                    int32_t part_val = ssb_ht_probe(d_part_ht_keys, d_part_ht_values, part_ht_mask, partkey);
                    if (part_val >= 0) {
                        int32_t gi = year_idx * stride_year
                                   + cust_val * (supp_dims * part_dims)
                                   + supp_val * part_dims + part_val;
                        int64_t profit = (int64_t)revenue - (int64_t)supplycost;
                        atomicAdd(reinterpret_cast<unsigned long long int *>(&s_hist[gi]),
                                  static_cast<unsigned long long int>(profit));
                    }
                }
            }
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int *>(&d_profit[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
}

static cudaError_t ssb_q4x_probe_flat(
    const uint64_t *lo_orderdate,
    const uint64_t *lo_custkey,
    const uint64_t *lo_partkey,
    const uint64_t *lo_suppkey,
    const uint64_t *lo_revenue,
    const uint64_t *lo_supplycost,
    uint64_t nrecs,
    const int32_t *d_date_ht_keys,
    const int32_t *d_date_ht_values,
    uint32_t date_ht_mask,
    const int32_t *d_cust_ht_keys,
    const int32_t *d_cust_ht_values,
    uint32_t cust_ht_mask,
    const int32_t *d_supp_ht_keys,
    const int32_t *d_supp_ht_values,
    uint32_t supp_ht_mask,
    const int32_t *d_part_ht_keys,
    const int32_t *d_part_ht_values,
    uint32_t part_ht_mask,
    int32_t supp_dims,
    int32_t part_dims,
    int32_t stride_year,
    int32_t total_groups,
    int64_t *d_profit,
    cudaStream_t stream)
{
    constexpr int BLK = SSB_Q4X_BLOCK_SIZE;
    int grid_dim = (nrecs + BLK - 1) / BLK;
    uint32_t hist_size = (uint32_t)total_groups;
    size_t smem = (size_t)hist_size * sizeof(int64_t);

    ssb_q4x_probe_flat_kernel<<<grid_dim, BLK, smem, stream>>>(
        lo_orderdate, lo_custkey, lo_partkey, lo_suppkey,
        lo_revenue, lo_supplycost,
        nrecs,
        d_date_ht_keys, d_date_ht_values, date_ht_mask,
        d_cust_ht_keys, d_cust_ht_values, cust_ht_mask,
        d_supp_ht_keys, d_supp_ht_values, supp_ht_mask,
        d_part_ht_keys, d_part_ht_values, part_ht_mask,
        supp_dims, part_dims, stride_year, hist_size,
        d_profit);

    return cudaSuccess;
}
