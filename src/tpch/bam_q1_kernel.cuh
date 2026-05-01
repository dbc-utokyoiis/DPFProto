#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ── Forward declarations for BaM types (opaque in this header) ──
typedef void* bam_q1_fused_io_ctx_t;
typedef void* bam_ctrl_handle_t;  // defined in bam_kernel.cuh, redeclared here

// ============================================================
// Q1 Unified Kernel: 7-field fused IO + Decomp + Eval
//
// All 7 LINEITEM columns are INT32+PFOR (CHAR(1) stored as ASCII int):
//   [0]=L_SHIPDATE, [1]=L_QUANTITY, [2]=L_EXTENDEDPRICE,
//   [3]=L_DISCOUNT, [4]=L_TAX, [5]=L_RETURNFLAG, [6]=L_LINESTATUS
//
// Single kernel: IO → PFOR decompress → evaluate → aggregate.
// ============================================================

// Aggregation constants (must match bam_kernel.cuh / q1.cuh)
#ifndef Q1_NUM_GROUPS
#define Q1_NUM_GROUPS 6
#endif
#ifndef Q1_NUM_AGGS
#define Q1_NUM_AGGS 7
#endif

struct BAMq1UnifiedParams {
    // Device & partitioning
    uint64_t partition_start_lbas[4];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;
    uint32_t num_blocks;

    // 7 fields: [0]=L_SHIPDATE, [1]=L_QUANTITY, [2]=L_EXTENDEDPRICE,
    //           [3]=L_DISCOUNT, [4]=L_TAX, [5]=L_RETURNFLAG, [6]=L_LINESTATUS
    uint64_t field_start_page_ids[7];
    uint16_t comp_methods[7];       // all PFOR
    uint32_t* d_comp_sizes[7];
    uint64_t* d_comp_offsets[7];

    // Prefix sum for driving field (field[0] = L_SHIPDATE)
    uint64_t* d_prefix_sum;

    // Pre-flattened arrays for fields with different page layout (CHAR fields)
    // nullptr if field matches driving prefix sum (can use paged decompress)
    const int32_t* d_flat_rf;    // L_RETURNFLAG [nrows], nullptr if paged
    const int32_t* d_flat_ls;    // L_LINESTATUS [nrows], nullptr if paged

    // Zone map
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;

    // Aggregation output
    int64_t* d_agg;              // [Q1_NUM_GROUPS * Q1_NUM_AGGS]

    // Per-block scratch for all 7 fields (num_blocks * 7 * scratch_stride int32_t)
    int32_t* d_scratch;
    uint32_t scratch_stride;
};

// ── Context management ──

bam_q1_fused_io_ctx_t bam_q1_unified_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_q1_unified_run(
    bam_q1_fused_io_ctx_t ctx,
    const BAMq1UnifiedParams& params,
    cudaStream_t stream);

void bam_q1_unified_destroy(bam_q1_fused_io_ctx_t ctx);
