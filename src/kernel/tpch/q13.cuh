#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <utility>

// Flatten INT32 column pages into flat uint64_t array
cudaError_t q13_flatten_int32_pages(
    const char *pages,
    uint32_t page_size,
    uint32_t capacity,       // (page_size - 12) / sizeof(int32_t)
    uint64_t nrecs_total,
    uint64_t *out,
    cudaStream_t stream);

// Flatten INT64 column pages into flat uint64_t array
cudaError_t q13_flatten_int64_pages(
    const char *pages,
    uint32_t page_size,
    uint32_t capacity,       // (page_size - 16) / sizeof(int64_t)  [+4B padding for alignment]
    uint64_t nrecs_total,
    uint64_t *out,
    cudaStream_t stream);

// Flatten INT32 column pages using prefix_sum (contiguous output)
cudaError_t q13_flatten_int32_pages_ps(
    const char *pages,
    uint32_t page_size,
    const uint64_t *prefix_sum,
    uint32_t npages,
    uint64_t nrecs_total,
    uint64_t *out,
    cudaStream_t stream);

// Flatten INT64 column pages using prefix_sum (contiguous output)
cudaError_t q13_flatten_int64_pages_ps(
    const char *pages,
    uint32_t page_size,
    const uint64_t *prefix_sum,
    uint32_t npages,
    uint64_t nrecs_total,
    uint64_t *out,
    cudaStream_t stream);

// Flatten INT32 column pages using prefix_sum with page-active mask + fill_value.
// Inactive pages (page_active[pg] == 0) produce fill_value instead of reading data.
// page_active == nullptr means all pages are active (equivalent to q13_flatten_int32_pages_ps).
cudaError_t q13_flatten_int32_pages_ps_masked(
    const char *pages,
    uint32_t page_size,
    const uint64_t *prefix_sum,
    uint32_t npages,
    uint64_t nrecs_total,
    const uint8_t *page_active,
    uint64_t fill_value,
    uint64_t *out,
    cudaStream_t stream);

// Q13 GOLAP pipeline: scan → sort → RLE → probe → sort → RLE → sort
// Returns result as (c_count, custdist) pairs in descending order.
cudaError_t q13_golap(
    const char *o_comment_pages,
    uint32_t o_comment_npages,
    const char *o_custkey_pages,        // for non-prefix_sum mode
    uint32_t o_custkey_capacity,        // for non-prefix_sum mode
    const uint64_t *d_prefix_sum,       // for prefix_sum mode (nullptr if unused)
    const uint64_t *d_o_custkey_flat,   // for prefix_sum mode (nullptr if unused)
    uint32_t page_size,
    uint32_t max_capacity_vchar,        // for non-prefix_sum mode
    uint64_t nrecs_orders,
    uint64_t nrecs_customer,
    const uint64_t *d_c_custkey,
    bool use_prefix_sum,
    // KMP tables (device)
    const char *d_patterns,
    const int *d_next,
    const int *d_pattern_offsets,
    const int *d_pattern_lengths,
    int num_patterns,
    int total_pattern_chars,
    // Output (host, filled on return)
    std::vector<std::pair<uint32_t, uint32_t>> &result,
    cudaStream_t stream);

// Q13 scan batch: scan a batch of decompressed O_COMMENT pages (prefix_sum mode).
// o_comment_pages, d_prefix_sum, d_o_custkey_flat, d_o_aggr_custkey are batch-offset
// (i.e., caller passes pointers offset to batch start for custkey/aggr arrays).
// d_count is a single atomic accumulator shared across batches.
cudaError_t q13_scan_batch(
    const char *o_comment_pages,
    const uint64_t *d_prefix_sum,
    uint32_t npages,
    uint32_t page_size,
    uint64_t nrecs_batch,
    const char *d_patterns,
    const int *d_next,
    const int *d_pattern_offsets,
    const int *d_pattern_lengths,
    int num_patterns,
    int total_pattern_chars,
    const uint64_t *d_o_custkey_flat,
    uint64_t *d_o_aggr_custkey,
    uint64_t *d_count,
    cudaStream_t stream);

// ============================================================
// Q13 pipeline: pre-allocated buffers (no malloc/free).
// ============================================================

struct Q13PipelineBuffers {
    uint64_t *d_sort_alt;            // [nrecs_orders]
    uint64_t *d_rle_keys;           // [nrecs_orders]
    uint32_t *d_rle_counts;         // [nrecs_orders]
    uint64_t *d_num_rle;            // [1]
    uint32_t *d_c_count;            // [nrecs_customer]
    uint32_t *d_c_count_alt;        // [nrecs_customer]
    uint32_t *d_aggr2_keys;         // [nrecs_customer]
    uint32_t *d_aggr2_counts;       // [nrecs_customer]
    uint64_t *d_composite_keys;     // [nrecs_customer]
    uint64_t *d_composite_keys_alt; // [nrecs_customer]
    uint64_t *d_composite_vals;     // [nrecs_customer]
    uint64_t *d_composite_vals_alt; // [nrecs_customer]
    void     *d_cub_temp;           // CUB scratch
    size_t    cub_temp_bytes;
    // Host result buffer (pre-allocated, avoids heap alloc in timed section)
    uint64_t *h_composite;          // [h_composite_capacity]
    size_t    h_composite_capacity;
};

// Compute required CUB scratch size for Q13 pipeline.
size_t q13_pipeline_cub_temp_size(uint64_t nrecs_orders, uint64_t nrecs_customer);

// Q13 aggregation: phases 2-7 (Sort → RLE → Probe → Sort → RLE → Pack).
// Takes a pre-populated d_o_aggr_custkey array (UINT64_MAX for LIKE matches,
// real custkey for non-matches) and produces (c_count, custdist) result pairs.
// Caller must pre-allocate all fields in bufs before the measurement point.
// Does NOT free d_o_aggr_custkey — caller owns it.
cudaError_t q13_pig_aggregate(
    const Q13PipelineBuffers &bufs,
    uint64_t *d_o_aggr_custkey,
    uint64_t nrecs_orders,
    const uint64_t *d_c_custkey,
    uint64_t nrecs_customer,
    std::vector<std::pair<uint32_t, uint32_t>> &result,
    cudaStream_t stream);
