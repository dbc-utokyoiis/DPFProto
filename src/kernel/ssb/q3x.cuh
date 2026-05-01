#pragma once

#include "common/pag_core.h"

// SSB Flight 3: Q3.1, Q3.2, Q3.3, Q3.4 (flat array version)

static constexpr int32_t SSB_Q3X_MAX_CUST_DIMS = 25;
static constexpr int32_t SSB_Q3X_MAX_SUPP_DIMS = 25;
static constexpr int32_t SSB_Q3X_MAX_YEARS = 7;
static constexpr int32_t SSB_Q3X_MAX_GROUPS =
    SSB_Q3X_MAX_CUST_DIMS * SSB_Q3X_MAX_SUPP_DIMS * SSB_Q3X_MAX_YEARS;

static constexpr int SSB_Q3X_BLOCK_SIZE = 128;

__global__ void ssb_q3x_probe_flat_kernel(
    const uint64_t *lo_orderdate,
    const uint64_t *lo_custkey,
    const uint64_t *lo_suppkey,
    const uint64_t *lo_revenue,
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
    int32_t num_supp_dims,
    uint32_t hist_size,
    int64_t *d_revenue)
{
    extern __shared__ int64_t s_hist[];
    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrecs) {
        int32_t orderdate = (int32_t)lo_orderdate[idx];
        int32_t custkey   = (int32_t)lo_custkey[idx];
        int32_t suppkey   = (int32_t)lo_suppkey[idx];
        int32_t revenue   = (int32_t)lo_revenue[idx];

        int32_t year_idx = ssb_ht_probe(d_date_ht_keys, d_date_ht_values, date_ht_mask, orderdate);
        if (year_idx >= 0) {
            int32_t cust_dim = ssb_ht_probe(d_cust_ht_keys, d_cust_ht_values, cust_ht_mask, custkey);
            if (cust_dim >= 0) {
                int32_t supp_dim = ssb_ht_probe(d_supp_ht_keys, d_supp_ht_values, supp_ht_mask, suppkey);
                if (supp_dim >= 0) {
                    int32_t gi = cust_dim * num_supp_dims * SSB_Q3X_MAX_YEARS
                               + supp_dim * SSB_Q3X_MAX_YEARS + year_idx;
                    atomicAdd(reinterpret_cast<unsigned long long int *>(&s_hist[gi]),
                              static_cast<unsigned long long int>((int64_t)revenue));
                }
            }
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int *>(&d_revenue[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
}

static cudaError_t ssb_q3x_probe_flat(
    const uint64_t *lo_orderdate,
    const uint64_t *lo_custkey,
    const uint64_t *lo_suppkey,
    const uint64_t *lo_revenue,
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
    int32_t num_supp_dims,
    int32_t num_cust_dims,
    int64_t *d_revenue,
    cudaStream_t stream)
{
    constexpr int BLK = SSB_Q3X_BLOCK_SIZE;
    int grid_dim = (nrecs + BLK - 1) / BLK;
    uint32_t hist_size = (uint32_t)(num_cust_dims * num_supp_dims * SSB_Q3X_MAX_YEARS);
    size_t smem = hist_size * sizeof(int64_t);

    ssb_q3x_probe_flat_kernel<<<grid_dim, BLK, smem, stream>>>(
        lo_orderdate, lo_custkey, lo_suppkey, lo_revenue,
        nrecs,
        d_date_ht_keys, d_date_ht_values, date_ht_mask,
        d_cust_ht_keys, d_cust_ht_values, cust_ht_mask,
        d_supp_ht_keys, d_supp_ht_values, supp_ht_mask,
        num_supp_dims, hist_size,
        d_revenue);

    return cudaSuccess;
}
