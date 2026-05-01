#include "q3_grace.cu"
#include "lineitem.cu"

__global__ void q6_shared_kernel(
    const Lineitem *lineitems,
    size_t len,
    int64_t *revenue)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t t = threadIdx.x;

    extern __shared__ int64_t smem[];

    if (idx < len) {
        Lineitem l = lineitems[idx];

        if (l.shipdate >= 19940101 && l.shipdate < 19940101 + 10000 &&
            6 - 1 <= l.discount && l.discount <= 6 + 1 &&
            l.quantity < 2400) {
            smem[t] = l.extendedprice * l.discount;
        }
        else {
            smem[t] = 0;
        }
    }

    __syncthreads();

    size_t m = blockDim.x;

    while (m > 1) {
        size_t h = (m + 1) / 2;
        if (idx + h < len && t + h < m) {
            smem[t] += smem[t + h];
        }
        __syncthreads();
        m = h;
    }

    if (t == 0) {
        atomicAdd(reinterpret_cast<unsigned long long int *>(revenue), static_cast<unsigned long long int>(smem[t]));
    }
}

extern "C" cudaError_t q6_shared(
    const Lineitem *lineitems,
    size_t len,
    int64_t *revenue,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    size_t shared_mem_size = sizeof(int64_t) * block_dim;

    q6_shared_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        lineitems, len, revenue);

    return cudaSuccess;
}

__global__ void q6_shared_kernel_debug(
    const Lineitem *lineitems,
    size_t len,
    int64_t *revenue)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t t = threadIdx.x;

    extern __shared__ int64_t smem[];

    if (idx < len) {
        Lineitem l = lineitems[idx];
        printf("(l.orderkey, l.linenumber): %ld %ld. \n", l.orderkey, l.linenumber);
        //int64_t orderkey = l.orderkey;
        //printf("(l.orderkey, l.linenumber): %ld,%ld. => ", orderkey, l.linenumber);

        if (l.shipdate >= 19940101 && l.shipdate < 19940101 + 10000 &&
            6 - 1 <= l.discount && l.discount <= 6 + 1 &&
            l.quantity < 2400) {
            smem[t] = l.extendedprice * l.discount;
        }
        else {
            smem[t] = 0;
        }
    }

    __syncthreads();

    size_t m = blockDim.x;

    while (m > 1) {
        size_t h = (m + 1) / 2;
        if (idx + h < len && t + h < m) {
            smem[t] += smem[t + h];
        }
        __syncthreads();
        m = h;
    }

    if (t == 0) {
        atomicAdd(reinterpret_cast<unsigned long long int *>(revenue), static_cast<unsigned long long int>(smem[t]));
        // printf("revenue: %ld.\n", *reinterpret_cast<unsigned long long int *>(revenue));
    }
}

extern "C" cudaError_t q6_shared_debug(
    const Lineitem *lineitems,
    size_t len,
    int64_t *revenue,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    size_t shared_mem_size = sizeof(int64_t) * block_dim;

    q6_shared_kernel_debug<<<grid_dim, block_dim, shared_mem_size, stream>>>(
    //q6_shared_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        lineitems, len, revenue);

    //printf("revenue: %ld");

    return cudaSuccess;
}

__global__ void q6_shared_kernel_subpage(
    Lineitem *lineitems,
    uint8_t *buf,
    int64_t *revenue,
    size_t nrecs_total,
    size_t nrecs_per_subpage,
    size_t siz_subpage
    )
{
    size_t idx_total = blockDim.x * blockIdx.x + threadIdx.x;
    size_t idx = threadIdx.x;
    size_t t = threadIdx.x;
    Lineitem *lineitems_base = reinterpret_cast<Lineitem*>(&buf[blockIdx.x * siz_subpage]);
    // printf("gridDim.x: %d, blockDim.x: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, idx_total: %lu, nrecs_total: %lu, nrecs_per_subpage: %lu, siz_subpage: %lu\n",
    //     gridDim.x, blockDim.x, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, idx_total, nrecs_total, nrecs_per_subpage, siz_subpage);

    extern __shared__ int64_t smem[];

    if (idx_total < nrecs_total) {
        Lineitem l = lineitems_base[idx];

        if (l.shipdate >= 19940101 && l.shipdate < 19940101 + 10000 &&
            6 - 1 <= l.discount && l.discount <= 6 + 1 &&
            l.quantity < 2400) {
            smem[t] = l.extendedprice * l.discount;
        }
        else {
            smem[t] = 0;
        }
    } else {
        // printf("PASSED: gridDim.x: %d, blockDim.x: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, idx_total: %lu, nrecs_total: %lu, nrecs_per_subpage: %lu, siz_subpage: %lu\n",
        //     gridDim.x, blockDim.x, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, idx_total, nrecs_total, nrecs_per_subpage, siz_subpage);
        smem[t] = 0;
    }

    __syncthreads();

    size_t m = blockDim.x;

    while (m > 1) {
        size_t h = (m + 1) / 2;
        if (idx + h < nrecs_total && t + h < m) {
            smem[t] += smem[t + h];
        }
        __syncthreads();
        m = h;
    }

    if (t == 0 && idx_total < nrecs_total) {
        atomicAdd(reinterpret_cast<unsigned long long int *>(revenue), static_cast<unsigned long long int>(smem[t]));
    }
}

extern "C" cudaError_t q6_shared_subpage(
    Lineitem *lineitems_buf,
    size_t siz_page,
    size_t siz_subpage,
    size_t nlineitem,
    size_t nlineitem_per_subpage,
    size_t nlineitem_per_subpage_final,
    size_t nsubpage,
    int64_t *revenue,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims_subpage(nlineitem, nlineitem_per_subpage);
    // std::cout << "grid_dim: " << grid_dim << ", block_dim: " << block_dim
    //   << ", nlineitem: " << nlineitem << std::endl;
    // nlineitem_per_subpage;
    // unsigned int grid_dim = nsubpage;
    // unsigned int block_dim = nsubpage;

    size_t nrecs_total = nlineitem;
    size_t nrecs_per_subpage = nlineitem_per_subpage;
    size_t shared_mem_size = sizeof(int64_t) * block_dim;

    //q6_shared_kernel_subpage<<<1, 1, shared_mem_size, stream>>>(
    q6_shared_kernel_subpage<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        lineitems_buf, reinterpret_cast<uint8_t*>(lineitems_buf), revenue,
        nrecs_total, nrecs_per_subpage, siz_subpage);

    return cudaSuccess;
}

#if 0
extern "C" cudaError_t q6_shared_subpage_multipages(
    Lineitem *lineitems_buf,
    size_t siz_page,
    size_t siz_subpage,
    size_t nlineitem,
    size_t nlineitem_per_subpage,
    size_t nlineitem_per_subpage_final,
    size_t nsubpage,
    int64_t *revenue,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims_subpage(nlineitem, nlineitem_per_subpage);
    // std::cout<< "grid_dim: " << grid_dim << ", block_dim: " << block_dim << std::endl;
    // nlineitem_per_subpage;
    // unsigned int grid_dim = nsubpage;
    // unsigned int block_dim = nsubpage;

    size_t nrecs_total = nlineitem;
    size_t nrecs_per_subpage = nlineitem_per_subpage;
    size_t shared_mem_size = sizeof(int64_t) * block_dim;

    //q6_shared_kernel_subpage<<<1, 1, shared_mem_size, stream>>>(
    q6_shared_kernel_subpage<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        lineitems_buf, reinterpret_cast<uint8_t*>(lineitems_buf), revenue,
        nrecs_total, nrecs_per_subpage, siz_subpage);

    return cudaSuccess;
}
#endif
