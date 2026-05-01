#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q5_lineitem.cuh — Fused BaM I/O + nvCOMPdx LZ4 + Q5 LINEITEM probe
//
// Single persistent kernel for Q5 Phase 4 (LINEITEM tile loop).
// Reads 2 INT32 fields (L_EXTPRICE, L_DISCOUNT) and 2 INT64 fields
// (L_ORDERKEY, L_SUPPKEY) per INT32 page, decompresses, and probes
// ORDERS/SUPPLIER hash tables to accumulate revenue.
//
// INT32→INT64 page mapping via prefix sum binary search on GPU.
// ============================================================

typedef void* bam_fused_q5li_ctx_t;
typedef void* bam_ctrl_handle_t;

struct BAMFusedQ5LIParams {
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

    // Prefix sums on GPU (for INT32→INT64 page mapping)
    const uint64_t* d_ps_i32;   // [npages_i32 + 1], ps[0]=0, ps[i]=cumulative nalloc
    const uint64_t* d_ps_i64;   // [npages_i64 + 1]
    uint32_t  npages_i32;
    uint32_t  npages_i64;

    // Multi-device
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;

    // Page geometry
    uint32_t  page_size;
    uint32_t  num_blocks;

    // Hash tables (already on GPU, built in prior phases)
    const uint64_t* d_ht_ord_keys;
    const int32_t*  d_ht_ord_values;
    uint32_t        ht_ord_mask;
    const uint64_t* d_ht_supp_keys;
    const int32_t*  d_ht_supp_values;
    uint32_t        ht_supp_mask;

    // Output: revenue per nation [25]
    int64_t*  d_revenue;

    // Zone map
    const uint32_t* d_active_page_ids; // global compact: dense active page IDs (nullptr = use d_page_mask)
    const uint8_t* d_page_mask;    // GPU mask (nullptr = all active)
    uint32_t       total_pages;    // total INT32 pages, or num_active when d_active_page_ids set
};

bam_fused_q5li_ctx_t bam_fused_q5li_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q5li_run_async(
    bam_fused_q5li_ctx_t ctx,
    const BAMFusedQ5LIParams& params,
    cudaStream_t stream);

void bam_fused_q5li_destroy(bam_fused_q5li_ctx_t ctx);

// ============================================================
// Warp-Specialized Q5 LINEITEM kernel (1024 threads, 7 decomp groups):
//
// 32 warps per block:
//   Warps 0-3:   IO (BaM page reads)
//     Warp 0:    L_EXTPRICE INT32 reads
//     Warp 1:    L_DISCOUNT INT32 reads
//     Warp 2:    INT64 metadata + L_ORDERKEY/L_SUPPKEY reads
//     Warp 3:    idle (spare)
//   Warps 4-31:  Decomp (7 groups × 4 warps, nvCOMPdx LZ4)
//   All 1024 threads: Q5 LINEITEM scan (HT probe + revenue aggregation)
//
// Batch=7 pages per batch, double-buffered IO/decomp pipeline.
// Handles INT32→INT64 page mapping via prefix sum binary search.
// ============================================================

struct Q5LIWarpSpecParams {
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

    // Prefix sums on GPU (for INT32→INT64 page mapping)
    const uint64_t* d_ps_i32;   // [npages_i32 + 1]
    const uint64_t* d_ps_i64;   // [npages_i64 + 1]
    uint32_t  npages_i32;
    uint32_t  npages_i64;

    // Multi-device
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;

    // Page geometry
    uint32_t  page_size;
    uint32_t  total_pages;             // total INT32 pages, or num_active when d_active_page_ids set
    const uint32_t* d_active_page_ids; // global compact: dense active page IDs (nullptr = use d_page_mask)
    const uint8_t* d_page_mask;        // GPU zonemap mask (nullptr = all active)

    // Hash tables (already on GPU, built in prior phases)
    const uint64_t* d_ht_ord_keys;
    const int32_t*  d_ht_ord_values;
    uint32_t        ht_ord_mask;
    const uint64_t* d_ht_supp_keys;
    const int32_t*  d_ht_supp_values;
    uint32_t        ht_supp_mask;

    // Output: revenue per nation [25]
    int64_t*  d_revenue;
};

uint32_t q5li_warp_spec_max_blocks(uint32_t page_size);

void q5li_warp_spec_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const Q5LIWarpSpecParams& params,
    uint32_t num_blocks,
    cudaStream_t stream);
