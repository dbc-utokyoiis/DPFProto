#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Flatten INT32 column pages using prefix_sum (contiguous output).
// Each page: [pag_head(12B)][int32_t data...]
// Output: uint64_t flat array (zero-extended from int32_t).
cudaError_t ssb_flatten_int32_pages_ps(
    const char *pages,
    uint32_t page_size,
    const uint64_t *prefix_sum,
    uint32_t npages,
    uint64_t nrecs_total,
    uint64_t *out,
    cudaStream_t stream);
