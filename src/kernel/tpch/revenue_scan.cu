// Revenue flat evaluation kernel: scan flat INT32 arrays, accumulate revenue
// Used by PiG tile execution (flatten + eval pattern)
// Same scan plan as Q6 (4 fields) but shipdate + optional quantity predicate.

#include <cuda_runtime.h>
#include <cstdint>

#include "revenue_scan.cuh"

static constexpr int REV_FLAT_BLOCK_SIZE = 256;

__global__ void revenue_scan_flat_kernel(
    const int32_t *__restrict__ d_shipdate,
    const int32_t *__restrict__ d_quantity,
    const int32_t *__restrict__ d_discount,
    const int32_t *__restrict__ d_extprice,
    uint64_t nrows,
    int64_t *__restrict__ d_revenue,
    int32_t sd_low,
    int32_t sd_high,
    int32_t qt_max,
    int32_t dc_low,
    int32_t dc_high)
{
    int64_t local_rev = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nrows;
         i += (uint64_t)blockDim.x * gridDim.x) {
        int32_t sd = d_shipdate[i];
        int32_t ep = d_extprice[i];
        int32_t dc = d_discount[i];
        bool pass = (sd >= sd_low && sd < sd_high);
        if (qt_max > 0) {
            int32_t qt = d_quantity[i];
            pass = pass && (qt < qt_max);
        }
        if (dc_high > 0) {
            pass = pass && (dc >= dc_low && dc <= dc_high);
        }
        if (pass) {
            local_rev += (int64_t)ep * dc;
        }
    }
    if (local_rev != 0)
        atomicAdd((unsigned long long *)d_revenue, (unsigned long long)local_rev);
}

cudaError_t revenue_scan_flat(
    const int32_t* d_shipdate,
    const int32_t* d_quantity,
    const int32_t* d_discount,
    const int32_t* d_extprice,
    uint64_t nrows,
    int64_t* d_revenue,
    int32_t sd_low,
    int32_t sd_high,
    int32_t qt_max,
    cudaStream_t stream,
    int32_t dc_low,
    int32_t dc_high)
{
    int grid = (int)((nrows + REV_FLAT_BLOCK_SIZE - 1) / REV_FLAT_BLOCK_SIZE);
    revenue_scan_flat_kernel<<<grid, REV_FLAT_BLOCK_SIZE, 0, stream>>>(
        d_shipdate, d_quantity, d_discount, d_extprice,
        nrows, d_revenue, sd_low, sd_high, qt_max, dc_low, dc_high);
    return cudaGetLastError();
}
