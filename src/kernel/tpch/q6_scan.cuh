#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Q6 flat evaluation kernel: operates on flat INT32 arrays (no widening needed)
// Filter: l_shipdate >= sd_low && l_shipdate < sd_high
//         && l_discount >= 5 && l_discount <= 7
//         && l_quantity < 2400
// Revenue: sum(l_extendedprice * l_discount)
cudaError_t q6_scan_flat(
    const int32_t* d_shipdate,
    const int32_t* d_quantity,
    const int32_t* d_discount,
    const int32_t* d_extprice,
    uint64_t nrows,
    int64_t* d_revenue,
    int32_t sd_low,
    int32_t sd_high,
    cudaStream_t stream);
