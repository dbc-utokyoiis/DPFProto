#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q5_dim.cuh — Q5 Dimension Table Process Kernel
//
// Kernel 2 for Q5 SUPPLIER/CUSTOMER phases:
//   Reads decompressed KEY (INT64) + NATIONKEY (INT32) from
//   page-indexed staging buffer, filters by nationkey, and
//   builds hash table in a single non-cooperative grid-stride pass.
//
//   Kernel 1 (IO+decomp) reuses q3_cust_io_decomp_launch.
// ============================================================

struct Q5DimProcessParams {
    const char* d_staging;       // decompressed pages [total_pages * page_size]
    uint32_t page_size;

    // KEY: INT64 field (first key_npages pages in staging)
    const uint64_t* key_prefix_sum;  // [key_npages] inclusive
    uint32_t key_npages;

    // NK: INT32 field (at staging offset nk_page_offset * page_size)
    uint32_t nk_page_offset;         // NK pages start at d_staging + nk_page_offset * page_size
    const uint64_t* nk_prefix_sum;   // [nk_npages] inclusive
    uint32_t nk_npages;

    // Total records
    uint64_t nrecs;

    // Nation filter
    const int8_t* d_nationkey_to_idx;  // [25], -1 = not in target region

    // HT (build)
    uint64_t* d_ht_keys;
    int32_t*  d_ht_values;
    uint32_t  ht_mask;
};

uint32_t q5_dim_process_max_blocks();

void q5_dim_process_launch(
    const Q5DimProcessParams& params,
    uint32_t num_blocks,
    cudaStream_t stream);

// ============================================================
// Phase 0: Combined REGION + NATION processing on GPU
//
// Single-block kernel that:
//   1. Scans R_NAME for "ASIA" → extracts asia_regionkey from R_REGIONKEY
//   2. Filters N_REGIONKEY by asia_regionkey → builds d_nationkey_to_idx[25]
//
// Reads nalloc from page headers (offset 0). No host-side
// parameters needed for record counts.
// ============================================================

void q5_phase0_region_nation_launch(
    const char* d_r_rkey_page,      // REGION regionkey page (INT32)
    const char* d_r_name_page,      // REGION name page (CHAR padded 28)
    const char* d_n_nkey_page,      // NATION nationkey page (INT32)
    const char* d_n_rkey_page,      // NATION regionkey page (INT32)
    int8_t* d_nationkey_to_idx,     // output [25], pre-filled with -1
    int32_t* d_asia_regionkey,      // output: asia_regionkey
    cudaStream_t stream);
