// Revenue kernel: same scan plan as Q6 (4 fields) but shipdate + optional quantity predicate.
// revenue = sum(extprice * discount) WHERE shipdate >= sd_low AND shipdate < sd_high
//   AND (qt_max == 0 || quantity < qt_max)
//
// Takes the same 4 field parameters as q6_col_vardate.

#include <cub/cub.cuh>

#include "revenue.cuh"

struct pag_head_rev {
    uint32_t nalloc;
    uint32_t watermark;
    uint32_t lfreespace;
};

static constexpr int REV_BLOCK_SIZE = 128;

__global__ void revenue_col_kernel(
    void *l_shipdate_addr,
    void *l_quantity_addr,
    void *l_discount_addr,
    void *l_extendedprice_addr,
    uint64_t nrecs_total,
    uint32_t capacity_nrecs_in_page_int32,
    int64_t *revenue,
    uint32_t page_size,
    int32_t sd_low,
    int32_t sd_high,
    int32_t qt_max)
{
    size_t total_idx = blockDim.x * blockIdx.x + threadIdx.x;

    using BlockReduceInt = cub::BlockReduce<int64_t, REV_BLOCK_SIZE>;
    __shared__ typename BlockReduceInt::TempStorage temp_storage;

    int64_t my_value = 0;

    if (total_idx < nrecs_total) {
        size_t page_idx = total_idx / capacity_nrecs_in_page_int32;
        size_t rec_idx = total_idx % capacity_nrecs_in_page_int32;

        size_t nalloc = reinterpret_cast<pag_head_rev*>(
                reinterpret_cast<char*>(l_shipdate_addr) + page_idx * page_size
            )->nalloc;

        if (rec_idx < nalloc) {
            int32_t *l_shipdate_page = reinterpret_cast<int32_t*>(
                reinterpret_cast<uint8_t*>(l_shipdate_addr) + page_idx * page_size);
            int32_t *l_discount_page = reinterpret_cast<int32_t*>(
                reinterpret_cast<uint8_t*>(l_discount_addr) + page_idx * page_size);
            int32_t *l_extendedprice_page = reinterpret_cast<int32_t*>(
                reinterpret_cast<uint8_t*>(l_extendedprice_addr) + page_idx * page_size);

            /* skip header */
            l_shipdate_page += 3;
            l_discount_page += 3;
            l_extendedprice_page += 3;

            uint32_t l_shipdate = l_shipdate_page[rec_idx];
            uint32_t l_discount = l_discount_page[rec_idx];
            uint32_t l_extendedprice = l_extendedprice_page[rec_idx];

            bool pass = (l_shipdate >= (uint32_t)sd_low && l_shipdate < (uint32_t)sd_high);
            if (qt_max > 0) {
                int32_t *l_quantity_page = reinterpret_cast<int32_t*>(
                    reinterpret_cast<uint8_t*>(l_quantity_addr) + page_idx * page_size);
                l_quantity_page += 3;  // skip header
                uint32_t l_quantity = l_quantity_page[rec_idx];
                pass = pass && (l_quantity < (uint32_t)qt_max);
            }
            if (pass) {
                my_value = (int64_t)l_extendedprice * l_discount;
            }
        }
    }

    int64_t aggregate = BlockReduceInt(temp_storage).Sum(my_value);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long int*>(revenue),
                  static_cast<unsigned long long int>(aggregate));
    }
}

cudaError_t revenue_col(
    void *l_shipdate,
    void *l_quantity,
    void *l_discount,
    void *l_extendedprice,
    uint64_t npages,
    uint32_t page_size,
    uint64_t nrecs_lineitem,
    int64_t *d_revenue,
    cudaStream_t stream,
    int32_t sd_low,
    int32_t sd_high,
    int32_t qt_max)
{
    constexpr int BLOCK_SIZE = REV_BLOCK_SIZE;
    int grid_dim = (nrecs_lineitem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t capacity_nrecs_in_page_int32 =
        (page_size - sizeof(pag_head_rev)) / sizeof(int32_t);

    revenue_col_kernel<<<grid_dim, BLOCK_SIZE, 0, stream>>>(
        l_shipdate, l_quantity, l_discount, l_extendedprice,
        nrecs_lineitem, capacity_nrecs_in_page_int32,
        d_revenue, page_size,
        sd_low, sd_high, qt_max);

    return cudaSuccess;
}
