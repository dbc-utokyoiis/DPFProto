#include <cub/cub.cuh>
//#include "q3_grace.cu"
//#include "lineitem.cu"

#include "q6.cuh"

struct pag_head {
    // char eyecatch; // magic number for page identification
    uint32_t nalloc; // number of allocated records
    uint32_t watermark; // free space watermark
    uint32_t lfreespace;
};


#if 0
__global__ void q6_col_kernel(
    void *l_shipdate_addr,
    void *l_quantity_addr,
    void *l_discount_addr,
    void *l_extendedprice_addr,
    uint64_t nrecs_total,
    uint32_t capacity_nrecs_in_page_int32,
    int64_t *revenue,
    uint32_t page_size
)
{
    /* Assuming that blockIdx.x is constant 128 */
    size_t total_idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t page_idx = total_idx / capacity_nrecs_in_page_int32;
    size_t rec_idx = total_idx % capacity_nrecs_in_page_int32;

    //size_t nalloc = reinterpret_cast<int32_t*>(l_shipdate_addr) + 1;
    size_t nalloc = reinterpret_cast<pag_head*>(
            reinterpret_cast<char*>(l_shipdate_addr) + page_idx * page_size
        )->nalloc;

    int32_t *l_shipdate_page  = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_shipdate_addr) + page_idx * page_size);
    int32_t *l_discount_page = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_discount_addr) + page_idx * page_size);
    int32_t *l_extendedprice_page = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_extendedprice_addr) + page_idx * page_size);
    int32_t *l_quantity_page = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_quantity_addr) + page_idx * page_size);

    /* skip header */
    l_shipdate_page += 3;
    l_discount_page += 3;
    l_extendedprice_page += 3;
    l_quantity_page += 3;

    uint32_t l_shipdate = l_shipdate_page[rec_idx];
    uint32_t l_discount = l_discount_page[rec_idx];
    uint32_t l_extendedprice = l_extendedprice_page[rec_idx];
    uint32_t l_quantity = l_quantity_page[rec_idx];

    using BlockReduceInt = cub::BlockReduce<int64_t, 1>;
    __shared__ typename BlockReduceInt::TempStorage temp_storage[1];

    size_t t = threadIdx.x;
    // printf("gridDim.x: %d, blockDim.x: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, idx_total: %lu, nrecs_total: %lu, nrecs_per_subpage: %lu, siz_subpage: %lu\n",
    //     gridDim.x, blockDim.x, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, idx_total, nrecs_total, nrecs_per_subpage, siz_subpage);

    extern __shared__ int64_t smem[];

    if (total_idx < nrecs_total) {
        if (l_shipdate >= 19940101 && l_shipdate < 19940101 + 10000 &&
            6 - 1 <= l_discount && l_discount <= 6 + 1 &&
            l_quantity < 2400) {
            smem[t] = l_extendedprice * l_discount;
        }
        else {
            smem[t] = 0;
        }
    } else {
        // printf("PASSED: gridDim.x: %d, blockDim.x: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, idx_total: %lu, nrecs_total: %lu, nrecs_per_subpage: %lu, siz_subpage: %lu\n",
        //     gridDim.x, blockDim.x, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, idx_total, nrecs_total, nrecs_per_subpage, siz_subpage);
        smem[t] = 0;
    }

    long long int aggregate1 = BlockReduceInt(temp_storage[0]).Sum(smem[t]);
    __syncthreads();

    if (t == 0 && total_idx < nrecs_total) {
        atomicAdd(reinterpret_cast<unsigned long long int*>(revenue), static_cast<unsigned long long int>(aggregate1));
    }
}
#else
static constexpr int Q6_BLOCK_SIZE = 128;
constexpr int Q6_INDIRECT_BLOCK_SIZE = 128;

__global__ void q6_col_kernel(
    void *l_shipdate_addr,
    void *l_quantity_addr,
    void *l_discount_addr,
    void *l_extendedprice_addr,
    uint64_t nrecs_total,
    uint32_t capacity_nrecs_in_page_int32,
    int64_t *revenue,
    uint32_t page_size
)
{
    size_t total_idx = blockDim.x * blockIdx.x + threadIdx.x;

    /* FIX (Bug E): BlockReduce template parameter must match actual block size.
     * Was: cub::BlockReduce<int64_t, 1> — only 1 thread's value was reduced.
     * Fix: cub::BlockReduce<int64_t, Q6_BLOCK_SIZE> — all 128 threads participate. */
    using BlockReduceInt = cub::BlockReduce<int64_t, Q6_BLOCK_SIZE>;
    __shared__ typename BlockReduceInt::TempStorage temp_storage;

    /* FIX (Bug D): Use a register variable instead of shared memory smem[].
     * The original code wrote to smem[threadIdx.x] then read it back for BlockReduce —
     * a register is sufficient and avoids the dynamic shared memory allocation. */
    int64_t my_value = 0;

    /* FIX: Guard ALL memory reads with bounds check FIRST.
     * Original code performed reads at lines 59-77 unconditionally, then checked
     * total_idx < nrecs_total at line 88. Excess threads (total_idx >= nrecs_total)
     * computed page_idx beyond loaded pages, causing out-of-bounds global reads. */
    if (total_idx < nrecs_total) {
        size_t page_idx = total_idx / capacity_nrecs_in_page_int32;
        size_t rec_idx = total_idx % capacity_nrecs_in_page_int32;

        size_t nalloc = reinterpret_cast<pag_head*>(
                reinterpret_cast<char*>(l_shipdate_addr) + page_idx * page_size
            )->nalloc;

        /* FIX (Bug D): Check rec_idx < nalloc for per-page bounds.
         * Original code read nalloc but never used it, allowing reads of
         * garbage data beyond the actual record count in each page. */
        if (rec_idx < nalloc) {
            int32_t *l_shipdate_page  = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_shipdate_addr) + page_idx * page_size);
            int32_t *l_discount_page = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_discount_addr) + page_idx * page_size);
            int32_t *l_extendedprice_page = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_extendedprice_addr) + page_idx * page_size);
            int32_t *l_quantity_page = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_quantity_addr) + page_idx * page_size);

            /* skip header */
            l_shipdate_page += 3;
            l_discount_page += 3;
            l_extendedprice_page += 3;
            l_quantity_page += 3;

            uint32_t l_shipdate = l_shipdate_page[rec_idx];
            uint32_t l_discount = l_discount_page[rec_idx];
            uint32_t l_extendedprice = l_extendedprice_page[rec_idx];
            uint32_t l_quantity = l_quantity_page[rec_idx];

            if (l_shipdate >= 19940101 && l_shipdate < 19940101 + 10000 &&
                6 - 1 <= l_discount && l_discount <= 6 + 1 &&
                l_quantity < 2400) {
                my_value = (int64_t)l_extendedprice * l_discount;
            }
        }
    }

    /* FIX (Bug E): All threads participate in the reduction (including those with my_value=0).
     * BlockReduce requires all threads in the block to call Sum(). */
    int64_t aggregate = BlockReduceInt(temp_storage).Sum(my_value);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long int*>(revenue), static_cast<unsigned long long int>(aggregate));
    }
}
#endif


__global__ void q6_col_kernel_indirect(
    void **l_shipdate_pages,
    void **l_quantity_pages,
    void **l_discount_pages,
    void **l_extendedprice_pages,
    uint64_t nrecs_total,
    uint32_t capacity_nrecs_in_page,
    int64_t *revenue,
    uint32_t page_size)
{
    size_t total_idx = blockDim.x * blockIdx.x + threadIdx.x;

    using BlockReduceInt = cub::BlockReduce<int64_t, Q6_INDIRECT_BLOCK_SIZE>;
    __shared__ typename BlockReduceInt::TempStorage temp_storage;

    int64_t my_value = 0;

    if (total_idx < nrecs_total) {
        size_t page_idx = total_idx / capacity_nrecs_in_page;
        size_t rec_idx  = total_idx % capacity_nrecs_in_page;

        uint8_t *shipdate_page = (uint8_t *)l_shipdate_pages[page_idx];
        size_t nalloc = ((pag_head *)shipdate_page)->nalloc;

        if (rec_idx < nalloc) {
            int32_t *sd = (int32_t *)(shipdate_page) + 3;
            int32_t *qt = (int32_t *)((uint8_t *)l_quantity_pages[page_idx]) + 3;
            int32_t *ep = (int32_t *)((uint8_t *)l_extendedprice_pages[page_idx]) + 3;
            int32_t *dc = (int32_t *)((uint8_t *)l_discount_pages[page_idx]) + 3;

            uint32_t l_shipdate      = sd[rec_idx];
            uint32_t l_quantity      = qt[rec_idx];
            uint32_t l_extendedprice = ep[rec_idx];
            uint32_t l_discount      = dc[rec_idx];

            if (l_shipdate >= 19940101 && l_shipdate < 19940101 + 10000 &&
                6 - 1 <= l_discount && l_discount <= 6 + 1 &&
                l_quantity < 2400) {
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


cudaError_t q6_col(
    /* Continuous PAG ARRAY */
    void *l_shipdate,
    void *l_quantity,
    void *l_discount,
    void *l_extendedprice,
    uint64_t npages,
    uint32_t page_size,
    uint64_t nrecs_lineitem,
    int64_t *d_revenue,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = Q6_BLOCK_SIZE;
    constexpr int block_dim = BLOCK_SIZE;
    int grid_dim = (nrecs_lineitem + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t nrecs_total = nrecs_lineitem;
    size_t shared_mem_size = sizeof(int64_t) * block_dim;


#if 0
    int32_t capacity_nrecs_in_page_int32 = (page_size - 12);

    q6_col_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        l_shipdate, l_quantity, l_discount, l_extendedprice,
        nrecs_total, capacity_nrecs_in_page_int32,
        d_revenue, page_size);
#else
    int32_t capacity_nrecs_in_page_int32 = (page_size - sizeof(pag_head)) / sizeof(int32_t);

    q6_col_kernel<<<grid_dim, block_dim, 0, stream>>>(
        l_shipdate, l_quantity, l_discount, l_extendedprice,
        nrecs_total, capacity_nrecs_in_page_int32,
        d_revenue, page_size);
#endif

    return cudaSuccess;
}

cudaError_t q6_col_indirect(
    void **d_l_shipdate_pages,
    void **d_l_quantity_pages,
    void **d_l_discount_pages,
    void **d_l_extendedprice_pages,
    uint32_t page_size,
    uint64_t nrecs_lineitem,
    int64_t *d_revenue,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = Q6_INDIRECT_BLOCK_SIZE;
    int grid_dim = (nrecs_lineitem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t capacity = (page_size - 12) / sizeof(int32_t);

    q6_col_kernel_indirect<<<grid_dim, BLOCK_SIZE, 0, stream>>>(
        d_l_shipdate_pages, d_l_quantity_pages,
        d_l_discount_pages, d_l_extendedprice_pages,
        nrecs_lineitem, capacity,
        d_revenue, page_size);

    return cudaSuccess;
}

// ── Variable-date Q6 kernel for selectivity experiments ──
// Same as q6_col_kernel but with parameterized L_SHIPDATE bounds.
static constexpr int Q6_VARDATE_BLOCK_SIZE = 128;

__global__ void q6_col_kernel_vardate(
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
    int32_t disc_low,
    int32_t disc_high,
    int32_t qt_max)
{
    size_t total_idx = blockDim.x * blockIdx.x + threadIdx.x;

    using BlockReduceInt = cub::BlockReduce<int64_t, Q6_VARDATE_BLOCK_SIZE>;
    __shared__ typename BlockReduceInt::TempStorage temp_storage;

    int64_t my_value = 0;

    if (total_idx < nrecs_total) {
        size_t page_idx = total_idx / capacity_nrecs_in_page_int32;
        size_t rec_idx = total_idx % capacity_nrecs_in_page_int32;

        size_t nalloc = reinterpret_cast<pag_head*>(
                reinterpret_cast<char*>(l_shipdate_addr) + page_idx * page_size
            )->nalloc;

        if (rec_idx < nalloc) {
            int32_t *l_shipdate_page  = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_shipdate_addr) + page_idx * page_size);
            int32_t *l_discount_page = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_discount_addr) + page_idx * page_size);
            int32_t *l_extendedprice_page = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_extendedprice_addr) + page_idx * page_size);
            int32_t *l_quantity_page = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(l_quantity_addr) + page_idx * page_size);

            l_shipdate_page += 3;
            l_discount_page += 3;
            l_extendedprice_page += 3;
            l_quantity_page += 3;

            uint32_t l_shipdate = l_shipdate_page[rec_idx];
            uint32_t l_discount = l_discount_page[rec_idx];
            uint32_t l_extendedprice = l_extendedprice_page[rec_idx];
            uint32_t l_quantity = l_quantity_page[rec_idx];

            if (l_shipdate >= (uint32_t)sd_low && l_shipdate < (uint32_t)sd_high &&
                (uint32_t)disc_low <= l_discount && l_discount <= (uint32_t)disc_high &&
                l_quantity < (uint32_t)qt_max) {
                my_value = (int64_t)l_extendedprice * l_discount;
            }
        }
    }

    int64_t aggregate = BlockReduceInt(temp_storage).Sum(my_value);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long int*>(revenue), static_cast<unsigned long long int>(aggregate));
    }
}

cudaError_t q6_col_vardate(
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
    int32_t disc_low,
    int32_t disc_high,
    int32_t qt_max)
{
    constexpr int BLOCK_SIZE = Q6_VARDATE_BLOCK_SIZE;
    int grid_dim = (nrecs_lineitem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t capacity_nrecs_in_page_int32 = (page_size - sizeof(pag_head)) / sizeof(int32_t);

    q6_col_kernel_vardate<<<grid_dim, BLOCK_SIZE, 0, stream>>>(
        l_shipdate, l_quantity, l_discount, l_extendedprice,
        nrecs_lineitem, capacity_nrecs_in_page_int32,
        d_revenue, page_size,
        sd_low, sd_high, disc_low, disc_high, qt_max);

    return cudaSuccess;
}
