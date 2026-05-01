#include "flatten.cuh"

static constexpr int SSB_FLATTEN_BLOCK = 256;

template <typename T>
__device__ __forceinline__ int ssb_upper_bound(
    const T *__restrict__ data, int n, const T &val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (data[mid] <= val)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__global__ void ssb_flatten_int32_pages_ps_kernel(
    const char *__restrict__ pages,
    uint32_t page_size,
    const uint64_t *__restrict__ prefix_sum,
    uint32_t npages,
    uint64_t nrecs_total,
    uint64_t *__restrict__ out)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_total) return;

    uint32_t page_idx = ssb_upper_bound(prefix_sum, (int)npages, idx);
    uint32_t local_idx = (page_idx == 0) ? (uint32_t)idx
                                         : (uint32_t)(idx - prefix_sum[page_idx - 1]);

    const char *page = pages + (uint64_t)page_idx * page_size;
    const int32_t *values = reinterpret_cast<const int32_t *>(page + 12);
    out[idx] = (uint64_t)(uint32_t)values[local_idx];
}

cudaError_t ssb_flatten_int32_pages_ps(
    const char *pages, uint32_t page_size,
    const uint64_t *prefix_sum, uint32_t npages,
    uint64_t nrecs_total, uint64_t *out, cudaStream_t stream)
{
    int grid = (int)((nrecs_total + SSB_FLATTEN_BLOCK - 1) / SSB_FLATTEN_BLOCK);
    ssb_flatten_int32_pages_ps_kernel<<<grid, SSB_FLATTEN_BLOCK, 0, stream>>>(
        pages, page_size, prefix_sum, npages, nrecs_total, out);
    return cudaGetLastError();
}
