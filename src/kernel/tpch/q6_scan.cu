// Q6 flat evaluation kernel: scan flat INT32 arrays, accumulate revenue
// Used by PiG tile execution (flatten + eval pattern)

#include <cuda_runtime.h>
#include <cstdint>

#include "q6_scan.cuh"

static constexpr int Q6_BLOCK_SIZE = 256;

__global__ void q6_scan_flat_kernel(
    const int32_t *__restrict__ d_shipdate,
    const int32_t *__restrict__ d_quantity,
    const int32_t *__restrict__ d_discount,
    const int32_t *__restrict__ d_extprice,
    uint64_t nrows,
    int64_t *__restrict__ d_revenue,
    int32_t sd_low,
    int32_t sd_high)
{
    int64_t local_rev = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nrows;
         i += (uint64_t)blockDim.x * gridDim.x) {
        int32_t sd = d_shipdate[i];
        int32_t qt = d_quantity[i];
        int32_t dc = d_discount[i];
        int32_t ep = d_extprice[i];
        if (sd >= sd_low && sd < sd_high &&
            dc >= 5 && dc <= 7 &&
            qt < 2400) {
            local_rev += (int64_t)ep * dc;
        }
    }
    if (local_rev != 0)
        atomicAdd((unsigned long long *)d_revenue, (unsigned long long)local_rev);
}

cudaError_t q6_scan_flat(
    const int32_t* d_shipdate,
    const int32_t* d_quantity,
    const int32_t* d_discount,
    const int32_t* d_extprice,
    uint64_t nrows,
    int64_t* d_revenue,
    int32_t sd_low,
    int32_t sd_high,
    cudaStream_t stream)
{
    int grid = (int)((nrows + Q6_BLOCK_SIZE - 1) / Q6_BLOCK_SIZE);
    q6_scan_flat_kernel<<<grid, Q6_BLOCK_SIZE, 0, stream>>>(
        d_shipdate, d_quantity, d_discount, d_extprice,
        nrows, d_revenue, sd_low, sd_high);
    return cudaGetLastError();
}
