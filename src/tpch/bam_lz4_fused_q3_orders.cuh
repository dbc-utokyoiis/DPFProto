#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q3_orders.cuh — Fused BaM I/O + nvCOMPdx LZ4 + Q3 ORDERS probe+build
//
// Single persistent kernel for Q3 Phase 2 (ORDERS).
// Reads 2 INT32 fields (O_ORDERDATE, O_SHIPPRIORITY) and 2 INT64 fields
// (O_ORDERKEY, O_CUSTKEY) per INT32 page, decompresses, filters on
// o_orderdate < 19950315, probes CUSTOMER hash set, and builds ORDERS HT.
//
// INT32→INT64 page mapping via prefix sum binary search on GPU.
// ============================================================

typedef void* bam_fused_q3ord_ctx_t;
typedef void* bam_ctrl_handle_t;
typedef void* bam_io_page_cache_t;

struct BAMFusedQ3OrdParams {
    // INT32 fields: 0=O_ORDERDATE, 1=O_SHIPPRIORITY
    uint64_t  i32_field_start_page_ids[2];
    uint64_t* d_comp_offsets_i32[2];
    uint32_t* d_comp_sizes_i32[2];
    bool      is_compressed_i32[2];

    // INT64 fields: 0=O_ORDERKEY, 1=O_CUSTKEY
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

    // CUSTOMER hash set (for probe)
    const uint64_t* d_custkey_set;
    uint32_t        custkey_set_mask;

    // ORDERS hash table (to build)
    uint64_t* d_orders_ht_keys;
    uint64_t* d_orders_ht_payloads;
    uint32_t  orders_ht_mask;

    // Zone map
    const uint32_t* d_active_page_ids; // global compact: dense active page IDs (nullptr = use d_page_mask)
    const uint8_t* d_page_mask;    // GPU mask (nullptr = all active)
    uint32_t       total_pages;    // total INT32 pages, or num_active when d_active_page_ids set

    // Q3SEL: skip o_orderdate < 19950315 filter when true
    bool            skip_date_filter;
};

bam_fused_q3ord_ctx_t bam_fused_q3ord_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q3ord_run_async(
    bam_fused_q3ord_ctx_t ctx,
    const BAMFusedQ3OrdParams& params,
    cudaStream_t stream);

void bam_fused_q3ord_destroy(bam_fused_q3ord_ctx_t ctx);

// Create using externally-owned (shared) page_cache and decomp_buf.
// The returned context does NOT own the resources — caller must free them separately.
bam_fused_q3ord_ctx_t bam_fused_q3ord_create_shared(
    bam_io_page_cache_t shared_pc,
    char* shared_decomp_buf,
    uint32_t page_size,
    uint32_t num_blocks);
