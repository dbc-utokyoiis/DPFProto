#pragma once

#include <cub/cub.cuh>
#include "common/pag_core.h"
#include "kernel/ssb/q2x.cuh"   // ssb_hash32, ssb_ht_lookup, SSB_HT_EMPTY

// SSB Flight 1: Q1.1, Q1.2, Q1.3
// LINEORDER-only scan with DATE lookup via hash table.
//
// SELECT SUM(lo_extendedprice * lo_discount) AS revenue
// FROM lineorder, date
// WHERE lo_orderdate = d_datekey
//   AND <date_filter>
//   AND lo_discount BETWEEN <disc_lo> AND <disc_hi>
//   AND lo_quantity < <qty_max>
//
// Q1.1: d_year=1993, discount [1,3], quantity < 25
// Q1.2: d_yearmonthnum=199401, discount [4,6], quantity BETWEEN 26 AND 35
// Q1.3: d_weeknuminyear=6 AND d_year=1994, discount [5,7], quantity BETWEEN 26 AND 35

static constexpr int SSB_Q1X_BLOCK_SIZE = 128;

__global__ void ssb_q1x_kernel(
    void *lo_orderdate_addr,
    void *lo_quantity_addr,
    void *lo_discount_addr,
    void *lo_extendedprice_addr,
    uint64_t nrecs_total,
    uint32_t capacity,
    uint32_t page_size,
    const int32_t *d_date_ht_keys,
    const int32_t *d_date_ht_values,
    uint32_t date_ht_mask,
    int32_t disc_lo,
    int32_t disc_hi,
    int32_t qty_lo,
    int32_t qty_hi,
    int64_t *revenue)
{
    size_t total_idx = blockDim.x * blockIdx.x + threadIdx.x;

    using BlockReduceInt = cub::BlockReduce<int64_t, SSB_Q1X_BLOCK_SIZE>;
    __shared__ typename BlockReduceInt::TempStorage temp_storage;

    int64_t my_value = 0;

    if (total_idx < nrecs_total) {
        size_t page_idx = total_idx / capacity;
        size_t rec_idx = total_idx % capacity;

        size_t nalloc = reinterpret_cast<pag_head *>(
            reinterpret_cast<char *>(lo_orderdate_addr) + page_idx * page_size
        )->nalloc;

        if (rec_idx < nalloc) {
            int32_t *orderdate_page = reinterpret_cast<int32_t *>(
                reinterpret_cast<uint8_t *>(lo_orderdate_addr) + page_idx * page_size) + 3;
            int32_t *quantity_page = reinterpret_cast<int32_t *>(
                reinterpret_cast<uint8_t *>(lo_quantity_addr) + page_idx * page_size) + 3;
            int32_t *discount_page = reinterpret_cast<int32_t *>(
                reinterpret_cast<uint8_t *>(lo_discount_addr) + page_idx * page_size) + 3;
            int32_t *extprice_page = reinterpret_cast<int32_t *>(
                reinterpret_cast<uint8_t *>(lo_extendedprice_addr) + page_idx * page_size) + 3;

            int32_t orderdate = orderdate_page[rec_idx];
            int32_t quantity = quantity_page[rec_idx];
            int32_t discount = discount_page[rec_idx];
            int32_t extprice = extprice_page[rec_idx];

            // DATE filter via hash table
            bool date_match = (ssb_ht_probe(d_date_ht_keys, d_date_ht_values,
                                             date_ht_mask, orderdate) >= 0);

            if (date_match &&
                discount >= disc_lo && discount <= disc_hi &&
                quantity >= qty_lo && quantity < qty_hi) {
                my_value = (int64_t)extprice * discount;
            }
        }
    }

    int64_t aggregate = BlockReduceInt(temp_storage).Sum(my_value);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long int *>(revenue),
                  static_cast<unsigned long long int>(aggregate));
    }
}

cudaError_t ssb_q1x(
    void *lo_orderdate,
    void *lo_quantity,
    void *lo_discount,
    void *lo_extendedprice,
    uint64_t npages,
    uint32_t page_size,
    uint64_t nrecs,
    const int32_t *d_date_ht_keys,
    const int32_t *d_date_ht_values,
    uint32_t date_ht_mask,
    int32_t disc_lo,
    int32_t disc_hi,
    int32_t qty_lo,
    int32_t qty_hi,
    int64_t *d_revenue,
    cudaStream_t stream)
{
    constexpr int BLK = SSB_Q1X_BLOCK_SIZE;
    int grid_dim = (nrecs + BLK - 1) / BLK;
    uint32_t capacity = (page_size - sizeof(pag_head)) / sizeof(int32_t);

    ssb_q1x_kernel<<<grid_dim, BLK, 0, stream>>>(
        lo_orderdate, lo_quantity, lo_discount, lo_extendedprice,
        nrecs, capacity, page_size,
        d_date_ht_keys, d_date_ht_values, date_ht_mask,
        disc_lo, disc_hi, qty_lo, qty_hi,
        d_revenue);

    return cudaSuccess;
}
