#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q13_comment.cuh — Fused BaM I/O + nvCOMPdx LZ4 decompress + KMP scan
//
// Single persistent kernel for O_COMMENT VCHAR column:
//   Warp 0: BaM I/O → LZ4 decompress (via bam_lz4_io_decomp_warp)
//   All 128 threads: KMP multi-pattern scan + custkey extraction
// Block-stride loop over all pages.
// ============================================================

typedef void* bam_fused_q13c_ctx_t;
typedef void* bam_ctrl_handle_t;  // defined in bam_kernel.cuh, redeclared here

struct BAMFusedQ13CParams {
    // O_COMMENT field metadata
    uint64_t  field_start_page_id;
    uint64_t* d_comp_offsets;      // [npages] byte offsets on disk (GPU)
    uint32_t* d_comp_sizes;        // [npages] compressed sizes (GPU)
    bool      is_compressed;

    // Multi-device
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;

    // Page geometry
    uint32_t  page_size;
    uint64_t  npages;
    uint32_t  num_blocks;

    // Prefix sum for row ID mapping
    const uint64_t* d_prefix_sum;      // [npages] cumulative row counts

    // O_CUSTKEY flat array (pre-loaded)
    const uint64_t* d_o_custkey_flat;  // [nrecs_orders]

    // Output
    uint64_t* d_o_aggr_custkey;        // [nrecs_orders] (pre-filled 0xFF)
    uint64_t* d_count;                 // atomic counter for qualifying records

    // KMP pattern tables (device)
    const char* d_patterns;
    const int*  d_next;
    const int*  d_pattern_offsets;
    const int*  d_pattern_lengths;
    int         num_patterns;
};

bam_fused_q13c_ctx_t bam_fused_q13c_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q13c_run_async(
    bam_fused_q13c_ctx_t ctx,
    const BAMFusedQ13CParams& params,
    cudaStream_t stream);

void bam_fused_q13c_destroy(bam_fused_q13c_ctx_t ctx);
