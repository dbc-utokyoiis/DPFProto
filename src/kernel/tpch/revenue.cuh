#pragma once

// Revenue kernel: same scan plan as Q6 (4 fields) but only shipdate predicate.
// revenue = sum(extprice * discount) WHERE shipdate >= sd_low AND shipdate < sd_high
//   AND (qt_max == 0 || quantity < qt_max)
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
    int32_t qt_max = 0);
