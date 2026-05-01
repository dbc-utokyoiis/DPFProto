#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q13_comment_v2.cuh — Q6-pattern independent warps
//
// Fused BaM I/O + nvCOMPdx LZ4 decompress + KMP scan for Q13 O_COMMENT.
// 4 warps/block, __launch_bounds__(128, 8) -> 8 blocks/SM.
// ============================================================

typedef void* bam_fused_q13c_v2_ctx_t;
typedef void* bam_ctrl_handle_t;

struct BAMFusedQ13Cv2Params {
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

// Fused BaM IO + LZ4 decomp + INT64 flatten (for Phase 1/2)
struct BAMFusedFlattenI64Params {
    uint64_t        field_start_page_id;
    const uint64_t* d_comp_offsets;    // [npages] byte offsets on disk (GPU)
    const uint32_t* d_comp_sizes;      // [npages] compressed sizes (GPU)
    bool            is_compressed;
    uint64_t        partition_start_lbas[4];
    uint32_t        n_devices;
    uint32_t        page_size;
    uint64_t        npages;
    const uint64_t* d_prefix_sum;      // [npages] cumulative row counts
    uint64_t*       d_output;          // flat output array [nrecs]
};

// ============================================================
// Decomp+Scan only kernel (no BaM IO — reads from GPU staging buffer)
// Used with separate BaM bulk read + pipeline overlap.
// ============================================================

struct Q13DecompScanParams {
    const char* staging_io;          // compressed pages [batch_count * page_size]
    const uint32_t* d_comp_sizes;    // global comp sizes array [total_npages]
    uint64_t  pg_start;              // starting page index in global array
    uint32_t  batch_count;           // pages in this batch
    uint32_t  page_size;
    uint32_t  num_blocks;
    bool      is_compressed;

    // Global prefix sum for row ID mapping
    const uint64_t* d_prefix_sum;    // [total_npages] cumulative row counts

    // O_CUSTKEY flat array (pre-loaded)
    const uint64_t* d_o_custkey_flat;

    // Output
    uint64_t* d_o_aggr_custkey;
    uint64_t* d_count;

    // KMP pattern tables (device)
    const char* d_patterns;
    const int*  d_next;
    const int*  d_pattern_offsets;
    const int*  d_pattern_lengths;
    int         num_patterns;
};

typedef void* q13_decomp_scan_ctx_t;

q13_decomp_scan_ctx_t q13_decomp_scan_create(
    uint32_t page_size,
    uint32_t num_blocks);

void q13_decomp_scan_async(
    q13_decomp_scan_ctx_t ctx,
    const Q13DecompScanParams& params,
    cudaStream_t stream);

void q13_decomp_scan_destroy(q13_decomp_scan_ctx_t ctx);

// ============================================================
// Original fused IO+decomp+scan kernel API (BaM inside kernel)
// ============================================================

bam_fused_q13c_v2_ctx_t bam_fused_q13c_v2_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q13c_v2_run_async(
    bam_fused_q13c_v2_ctx_t ctx,
    const BAMFusedQ13Cv2Params& params,
    cudaStream_t stream);

void bam_fused_q13c_v2_flatten_i64_async(
    bam_fused_q13c_v2_ctx_t ctx,
    const BAMFusedFlattenI64Params& params,
    cudaStream_t stream);

void bam_fused_q13c_v2_destroy(bam_fused_q13c_v2_ctx_t ctx);
