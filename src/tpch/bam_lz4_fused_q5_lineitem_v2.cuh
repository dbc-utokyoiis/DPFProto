#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// bam_lz4_fused_q5_lineitem_v2.cuh — Q1-style balanced pipeline for Q5 LINEITEM
//
// Per-block layout (288 threads = 9 warps):
//   Warp 0:     IO warp — issues all slots (I32×2 + I64×2×i64_c) sequentially
//   Warps 1-8:  Decomp warps (8), one per slot:
//     warp 1 → L_EXTPRICE (INT32)
//     warp 2 → L_DISCOUNT (INT32)
//     warp 3 → L_ORDERKEY[i64 page 0]
//     warp 4 → L_SUPPKEY [i64 page 0]
//     warp 5 → L_ORDERKEY[i64 page 1]
//     warp 6 → L_SUPPKEY [i64 page 1]
//     warp 7 → L_ORDERKEY[i64 page 2]   (only when i64_c==3)
//     warp 8 → L_SUPPKEY [i64 page 2]   (only when i64_c==3)
//
// Double-buffered IO ring (NBUF=2) + double-buffered decomp output (NFACE=2).
// Pipeline:
//   Priming: IO warp reads first INT32 page's slots → ring[0]
//   Main loop (grid-stride over active pages):
//     Phase A: IO(next) ∥ decomp(current)
//     __syncthreads
//     Phase B: all 288 threads scan previous page (HT probes + revenue)
//     __syncthreads
//   Epilogue: scan last page
//
// Block count fixed at 108 (= SM count on A100); grid-stride loop handles
// arbitrary active page counts.

struct BAMFusedQ5LIV2Params {
    // INT32 fields: 0=L_EXTPRICE, 1=L_DISCOUNT
    uint64_t  i32_field_start_page_ids[2];
    uint64_t* d_comp_offsets_i32[2];
    uint32_t* d_comp_sizes_i32[2];
    bool      is_compressed_i32[2];

    // INT64 fields: 0=L_ORDERKEY, 1=L_SUPPKEY
    uint64_t  i64_field_start_page_ids[2];
    uint64_t* d_comp_offsets_i64[2];
    uint32_t* d_comp_sizes_i64[2];
    bool      is_compressed_i64[2];

    // Prefix sums (INT32→INT64 page mapping)
    const uint64_t* d_ps_i32;   // [npages_i32 + 1]
    const uint64_t* d_ps_i64;   // [npages_i64 + 1]
    uint32_t  npages_i32;
    uint32_t  npages_i64;

    // Multi-device
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;

    // Page geometry
    uint32_t  page_size;
    uint32_t  num_blocks;       // launch grid dim (typically SM count)

    // Hash tables
    const uint64_t* d_ht_ord_keys;
    const int32_t*  d_ht_ord_values;
    uint32_t        ht_ord_mask;
    const uint64_t* d_ht_supp_keys;
    const int32_t*  d_ht_supp_values;
    uint32_t        ht_supp_mask;

    // Output: revenue per nation [25]
    int64_t*  d_revenue;

    // Zone map: active-ID list or mask
    const uint32_t* d_active_page_ids; // dense compact IDs (nullptr → use d_page_mask)
    const uint8_t*  d_page_mask;       // per-page mask (nullptr → all active)
    uint32_t        total_pages;       // if d_active_page_ids: num_active; else npages_i32
};

typedef void* bam_fused_q5li_v2_ctx_t;
typedef void* bam_ctrl_handle_t;

bam_fused_q5li_v2_ctx_t bam_fused_q5li_v2_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q5li_v2_run_async(
    bam_fused_q5li_v2_ctx_t ctx,
    const BAMFusedQ5LIV2Params& params,
    cudaStream_t stream);

void bam_fused_q5li_v2_destroy(bam_fused_q5li_v2_ctx_t ctx);

uint32_t q5li_v2_max_blocks(uint32_t page_size);
