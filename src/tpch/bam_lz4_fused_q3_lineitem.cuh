#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q3_lineitem.cuh — Fused BaM I/O + nvCOMPdx LZ4 + Q3 LINEITEM probe
//
// Single persistent kernel for Q3 Phase 3 (LINEITEM tile loop).
// Reads 3 INT32 fields (L_SHIPDATE, L_EXTPRICE, L_DISCOUNT) and 1 INT64 field
// (L_ORDERKEY) per INT32 page, decompresses, filters on l_shipdate > 19950315,
// probes ORDERS HT, and aggregates revenue per l_orderkey.
//
// INT32→INT64 page mapping via prefix sum binary search on GPU.
// ============================================================

typedef void* bam_fused_q3li_ctx_t;
typedef void* bam_ctrl_handle_t;

struct BAMFusedQ3LIParams {
    // INT32 fields: 0=L_SHIPDATE, 1=L_EXTPRICE, 2=L_DISCOUNT
    uint64_t  i32_field_start_page_ids[3];
    uint64_t* d_comp_offsets_i32[3];
    uint32_t* d_comp_sizes_i32[3];
    bool      is_compressed_i32[3];

    // INT64 fields: 0=L_ORDERKEY
    uint64_t  i64_field_start_page_ids[1];
    uint64_t* d_comp_offsets_i64[1];
    uint32_t* d_comp_sizes_i64[1];
    bool      is_compressed_i64[1];

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

    // ORDERS hash table (keys + payloads, built in prior phase)
    const uint64_t* d_orders_ht_keys;
    const uint64_t* d_orders_ht_payloads;
    uint32_t        orders_ht_mask;

    // Aggregation hash table (GROUP BY l_orderkey → revenue)
    uint64_t* d_aggr_keys;
    int64_t*  d_aggr_revenues;
    uint32_t  aggr_mask;

    // Zone map: active page list (replaces d_page_active iteration)
    const uint32_t* d_active_pages;  // [n_active_pages] — list of active INT32 page indices
    uint32_t        n_active_pages;  // number of active pages (0 = process all)

    // Q3SEL: skip l_shipdate > 19950315 filter when true
    bool            skip_shipdate_filter;
};

bam_fused_q3li_ctx_t bam_fused_q3li_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q3li_run_async(
    bam_fused_q3li_ctx_t ctx,
    const BAMFusedQ3LIParams& params,
    cudaStream_t stream);

void bam_fused_q3li_destroy(bam_fused_q3li_ctx_t ctx);

// ============================================================
// Warp-Specialized Q3 LINEITEM kernel (1024 threads, 7 decomp groups):
//
// 32 warps per block:
//   Warps 0-3:   IO (BaM page reads)
//     Warps 0-2: INT32 fields (L_SHIPDATE, L_EXTPRICE, L_DISCOUNT)
//     Warp 3:    INT64 field (L_ORDERKEY) + metadata computation
//   Warps 4-31:  Decomp (7 groups × 4 warps, nvCOMPdx LZ4)
//   All 1024 threads: Q3 LINEITEM scan (HT probe + aggregation)
//
// Batch=7 pages per batch, double-buffered IO/decomp pipeline.
// Handles INT32→INT64 page mapping via prefix sum binary search.
// ============================================================

struct Q3LIWarpSpecParams {
    // INT32 fields: 0=L_SHIPDATE, 1=L_EXTPRICE, 2=L_DISCOUNT
    uint64_t  i32_field_start_page_ids[3];
    uint64_t* d_comp_offsets_i32[3];
    uint32_t* d_comp_sizes_i32[3];
    bool      is_compressed_i32[3];

    // INT64 field: 0=L_ORDERKEY
    uint64_t  i64_field_start_page_ids[1];
    uint64_t* d_comp_offsets_i64[1];
    uint32_t* d_comp_sizes_i64[1];
    bool      is_compressed_i64[1];

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

    // ORDERS hash table
    const uint64_t* d_orders_ht_keys;
    const uint64_t* d_orders_ht_payloads;
    uint32_t        orders_ht_mask;

    // Aggregation hash table
    uint64_t* d_aggr_keys;
    int64_t*  d_aggr_revenues;
    uint32_t  aggr_mask;

    // Q3SEL: skip l_shipdate > 19950315 filter when true
    bool      skip_shipdate_filter;
};

uint32_t q3li_warp_spec_max_blocks(uint32_t page_size);

void q3li_warp_spec_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const Q3LIWarpSpecParams& params,
    uint32_t num_blocks,
    cudaStream_t stream);
