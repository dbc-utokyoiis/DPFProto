#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q5_orders.cuh — Fused BaM I/O + nvCOMPdx LZ4 + Q5 ORDERS probe+build
//
// Single persistent kernel for Q5 Phase 3 (ORDERS).
// Reads 1 INT32 field (O_ORDERDATE) and 2 INT64 fields
// (O_ORDERKEY, O_CUSTKEY) per INT32 page, decompresses, filters on
// date range, probes CUSTOMER HT, and builds ORDERS HT.
//
// INT32→INT64 page mapping via prefix sum binary search on GPU.
// ============================================================

typedef void* bam_fused_q5ord_ctx_t;
typedef void* bam_ctrl_handle_t;

struct BAMFusedQ5OrdParams {
    // INT32 field: O_ORDERDATE
    uint64_t  i32_field_start_page_id;
    uint64_t* d_comp_offsets_i32;
    uint32_t* d_comp_sizes_i32;
    bool      is_compressed_i32;

    // INT64 fields: 0=O_ORDERKEY, 1=O_CUSTKEY
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
    uint32_t  num_blocks;

    // Date filter
    int32_t   date_low;    // inclusive
    int32_t   date_high;   // exclusive

    // CUSTOMER hash table (for probe: custkey → nation_idx)
    const uint64_t* d_ht_cust_keys;
    const int32_t*  d_ht_cust_values;
    uint32_t        ht_cust_mask;

    // ORDERS hash table (to build: orderkey → nation_idx)
    uint64_t* d_ht_ord_keys;
    int32_t*  d_ht_ord_values;
    uint32_t  ht_ord_mask;

    // Zone map
    const uint32_t* d_active_page_ids; // global compact: dense active page IDs (nullptr = use d_page_mask)
    const uint8_t* d_page_mask;    // GPU mask (nullptr = all active)
    uint32_t       total_pages;    // total INT32 pages, or num_active when d_active_page_ids set
};

bam_fused_q5ord_ctx_t bam_fused_q5ord_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q5ord_run_async(
    bam_fused_q5ord_ctx_t ctx,
    const BAMFusedQ5OrdParams& params,
    cudaStream_t stream);

void bam_fused_q5ord_destroy(bam_fused_q5ord_ctx_t ctx);

// ============================================================
// Warp-Specialized Q5 ORDERS kernel (1024 threads, 7 decomp groups):
//
// 32 warps per block:
//   Warps 0-3:   IO (BaM page reads)
//     Warp 0:    O_ORDERDATE INT32 reads
//     Warp 1:    INT64 metadata + O_ORDERKEY/O_CUSTKEY reads
//     Warps 2-3: idle (spare)
//   Warps 4-31:  Decomp (7 groups × 4 warps, nvCOMPdx LZ4)
//   All 1024 threads: Q5 ORDERS scan (date filter + HT probe/build)
//
// Batch=7 pages per batch, double-buffered IO/decomp pipeline.
// Handles INT32→INT64 page mapping via prefix sum binary search.
// ============================================================

struct Q5OrdWarpSpecParams {
    // INT32 field: O_ORDERDATE
    uint64_t  i32_field_start_page_id;
    uint64_t* d_comp_offsets_i32;
    uint32_t* d_comp_sizes_i32;
    bool      is_compressed_i32;

    // INT64 fields: 0=O_ORDERKEY, 1=O_CUSTKEY
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

    // Date filter
    int32_t   date_low;    // inclusive
    int32_t   date_high;   // exclusive

    // CUSTOMER hash table (for probe: custkey → nation_idx)
    const uint64_t* d_ht_cust_keys;
    const int32_t*  d_ht_cust_values;
    uint32_t        ht_cust_mask;

    // ORDERS hash table (to build: orderkey → nation_idx)
    uint64_t* d_ht_ord_keys;
    int32_t*  d_ht_ord_values;
    uint32_t  ht_ord_mask;
};

uint32_t q5ord_warp_spec_max_blocks(uint32_t page_size);

void q5ord_warp_spec_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const Q5OrdWarpSpecParams& params,
    uint32_t num_blocks,
    cudaStream_t stream);
