#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_q3sel_scan.cuh — Q3SEL GPU-side IO pruning + IO+decomp + scan
//
// For compute-bound Q3SEL where scan is the bottleneck,
// IO+decomp and scan are separated so the scan kernel
// gets full GPU parallelism (all SMs).
//
// All IO pruning (INT64 page derivation, descriptor construction,
// tile planning) is performed GPU-side.
// ============================================================

// IO+decomp parameters: field metadata for GPU-side LBA computation
struct Q3SelIODecompParams {
    // Active INT32 pages (from zone map d_active_ids)
    const uint32_t* d_active_ids;
    uint32_t n_active;

    // Needed INT64 pages (from q3sel_derive_i64_launch)
    const uint32_t* d_needed_i64;
    uint32_t n_needed_i64;

    // INT32 fields (up to 4)
    uint32_t n_i32_fields;
    uint64_t field_start_page_ids_i32[4];
    const uint64_t* d_comp_offsets_i32[4];
    const uint32_t* d_comp_sizes_i32[4];
    bool is_compressed_i32[4];

    // INT64 fields (up to 4)
    uint32_t n_i64_fields;
    uint64_t field_start_page_ids_i64[4];
    const uint64_t* d_comp_offsets_i64[4];
    const uint32_t* d_comp_sizes_i64[4];
    bool is_compressed_i64[4];

    // Storage layout
    uint64_t partition_start_lbas[4];
    uint32_t n_devices;
    uint32_t page_size;
};

// ORDERS scan parameters
struct Q3SelOrdersScanParams {
    const char* d_staging;
    uint32_t page_size;

    // Staging field offsets (in pages from d_staging)
    uint32_t odate_pg_off;       // O_ORDERDATE pages start
    uint32_t sp_pg_off;          // O_SHIPPRIORITY pages start
    uint32_t okey_pg_off;        // O_ORDERKEY pages start
    uint32_t ckey_pg_off;        // O_CUSTKEY pages start

    // Active INT32 pages
    uint32_t n_active_i32;
    const uint64_t* d_active_ps_i32;    // [n_active_i32 + 1], ps[0]=0
    const uint32_t* d_active_pages_i32; // [n_active_i32] global page indices

    // Full prefix sums (for global row → INT64 page mapping)
    const uint64_t* d_ps_i32_full;      // [npages_i32 + 1]
    const uint64_t* d_ps_i64_full;      // [npages_i64 + 1]
    uint32_t npages_i64;

    // INT64 global page → staging index remap
    const uint32_t* d_i64_remap;        // [npages_i64]

    // CUSTOMER hash set (probe)
    const uint64_t* d_custkey_set;
    uint32_t custkey_set_mask;

    // ORDERS HT (build)
    uint64_t* d_orders_ht_keys;
    uint64_t* d_orders_ht_payloads;
    uint32_t orders_ht_mask;

    // Date filter: 0 = disabled, >0 = filter o_orderdate < limit
    int32_t o_orderdate_limit;

    uint64_t nrecs_active;
};

// LINEITEM scan parameters
struct Q3SelLineitemScanParams {
    const char* d_staging;
    uint32_t page_size;

    // Staging field offsets (in pages)
    uint32_t extprice_pg_off;
    uint32_t discount_pg_off;
    uint32_t okey_pg_off;

    // Active INT32 pages for this tile
    uint32_t n_active_i32;
    const uint64_t* d_active_ps_i32;
    const uint32_t* d_active_pages_i32;

    // Full prefix sums (global)
    const uint64_t* d_ps_i32_full;
    const uint64_t* d_ps_i64_full;
    uint32_t npages_i64;

    // INT64 remap for this tile
    const uint32_t* d_i64_remap;

    // ORDERS HT (probe)
    const uint64_t* d_orders_ht_keys;
    const uint64_t* d_orders_ht_payloads;
    uint32_t orders_ht_mask;

    // Aggregation
    uint64_t* d_aggr_keys;
    int64_t* d_aggr_revenues;
    uint32_t aggr_mask;

    // Shipdate filter: 0 = disabled, >0 = filter l_shipdate > limit
    int32_t l_shipdate_limit;

    uint64_t nrecs_active;
};

// Tile descriptor for LINEITEM tiling (GPU-computed)
struct Q3SelTileInfo {
    uint32_t i32_start, i32_count;
    uint32_t i64_start, i64_count;
    uint32_t total_descs;
};

// ── GPU-side IO pruning ──

// Derive needed INT64 pages from active INT32 pages via prefix sum binary search.
// Single-block kernel: mark → warp-ballot compact.
// Output: d_needed_i64[<=npages_i64] (dense sorted), *d_n_needed_i64 (count).
void q3sel_derive_i64_launch(
    const uint32_t* d_active_ids, uint32_t n_active,
    const uint64_t* d_ps_i32, const uint64_t* d_ps_i64,
    uint32_t npages_i64,
    uint8_t* d_i64_mask,           // scratch [npages_i64]
    uint32_t* d_needed_i64,        // output [npages_i64]
    uint32_t* d_n_needed_i64,      // output [1]
    cudaStream_t stream);

// Compute tile plan for LINEITEM tiling (GPU kernel, single-thread sequential).
// Outputs tile descriptors fitting staging_capacity.
void q3sel_tile_plan_launch(
    const uint32_t* d_active_ids, uint32_t n_active,
    const uint32_t* d_needed_i64, uint32_t n_needed_i64,
    const uint64_t* d_ps_i32, const uint64_t* d_ps_i64,
    uint32_t npages_i64,
    uint32_t staging_capacity,
    uint32_t n_i32_fields, uint32_t n_i64_fields,
    Q3SelTileInfo* d_tiles, uint32_t* d_n_tiles,
    cudaStream_t stream);

// Per-tile setup: build i64_remap + active prefix sum + nrecs (GPU kernel).
void q3sel_tile_setup_launch(
    const uint32_t* d_active_ids,
    uint32_t tile_i32_start, uint32_t tile_i32_count,
    const uint32_t* d_needed_i64,
    uint32_t tile_i64_start, uint32_t tile_i64_count,
    const uint64_t* d_ps_i32_full,
    uint32_t npages_i64,
    uint64_t* d_active_ps,
    uint32_t* d_i64_remap,
    uint64_t* d_nrecs,
    cudaStream_t stream);

// ── IO+decomp (GPU-side LBA computation) ──

uint32_t q3sel_io_decomp_max_blocks(uint32_t page_size);

void q3sel_io_decomp_launch(
    void* d_ctrls, void* d_pc, const char* pc_base_addr,
    char* d_staging,
    const Q3SelIODecompParams& params,
    uint32_t num_blocks,
    cudaStream_t stream);

// ── Scan (full GPU parallelism) ──

void q3sel_orders_scan_launch(
    const Q3SelOrdersScanParams& params,
    cudaStream_t stream);

void q3sel_lineitem_scan_launch(
    const Q3SelLineitemScanParams& params,
    cudaStream_t stream);
