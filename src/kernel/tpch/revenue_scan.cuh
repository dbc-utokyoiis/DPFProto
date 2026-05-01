#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Revenue flat evaluation kernel: operates on flat INT32 arrays
// Filter: l_shipdate >= sd_low && l_shipdate < sd_high
//         && (qt_max == 0 || l_quantity < qt_max)
//         && (dc_high == 0 || (l_discount >= dc_low && l_discount <= dc_high))
// Revenue: sum(l_extendedprice * l_discount)
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
    int32_t dc_low = 0,
    int32_t dc_high = 0);
