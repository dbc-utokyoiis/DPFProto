#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q16_part.cuh — Fused BaM I/O + nvCOMPdx LZ4 + Q16 PART HT build
//
// Single persistent kernel for Q16 Phase 1 (PART).
// Reads P_SIZE (INT32), P_BRAND (CHAR 12B), P_PARTKEY (INT64)
// per P_SIZE page, decompresses, filters on p_size bitmask and
// p_brand != Brand#45, then inserts (partkey, partial_gid, row_idx) into HT.
//
// Type_id filtering is deferred to a Stage 2 fixup kernel
// so this kernel can run concurrently with P_TYPE.
//
// P_SIZE→P_BRAND and P_SIZE→P_PARTKEY page mapping via prefix sum
// binary search on GPU.
// ============================================================

typedef void* bam_fused_q16part_ctx_t;
typedef void* bam_ctrl_handle_t;

struct BAMFusedQ16PartParams {
    // P_SIZE (INT32, reference field)
    uint64_t  psize_start_page_id;
    const uint64_t* d_psize_comp_offsets;
    const uint32_t* d_psize_comp_sizes;
    bool      psize_is_compressed;

    // P_BRAND (CHAR padded)
    uint64_t  brand_start_page_id;
    const uint64_t* d_brand_comp_offsets;
    const uint32_t* d_brand_comp_sizes;
    bool      brand_is_compressed;
    uint32_t  brand_padded_len;  // 12

    // P_PARTKEY (INT64)
    uint64_t  pk_start_page_id;
    const uint64_t* d_pk_comp_offsets;
    const uint32_t* d_pk_comp_sizes;
    bool      pk_is_compressed;

    // Exclusive prefix sums on GPU: ps[0]=0, ps[i]=cumulative nalloc through page i-1
    const uint64_t* d_ps_psize;  // [npages_psize + 1]
    const uint64_t* d_ps_brand;  // [npages_brand + 1]
    const uint64_t* d_ps_pk;     // [npages_pk + 1]
    uint32_t npages_psize;
    uint32_t npages_brand;
    uint32_t npages_pk;

    // Multi-device
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;

    // Page geometry
    uint32_t  page_size;
    uint32_t  num_blocks;

    // Filter parameters
    uint64_t  p_size_bitmask;
    uint32_t  brand_exclude_id;  // 19 = Brand#45

    // PART HT output (partial — no type_id filtering)
    // partial_gid = (brand_id << 8) | (size_val - 1)
    uint64_t* d_ht_keys;      // [ht_capacity], init to 0xFF
    uint32_t* d_ht_group_ids; // [ht_capacity], stores partial_gid during Stage 1
    uint32_t* d_ht_row_idx;   // [ht_capacity], stores global row index for Stage 2
    uint32_t  ht_mask;

    // Debug counters (optional, may be null)
    uint32_t* d_dbg_total_scanned;    // total records scanned
    uint32_t* d_dbg_brand_overflow;   // records skipped due to brand page overflow
    uint32_t* d_dbg_pk_overflow;      // records skipped due to PK page overflow
    uint32_t* d_dbg_ht_inserted;      // records inserted into HT
};

bam_fused_q16part_ctx_t bam_fused_q16part_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q16part_run_async(
    bam_fused_q16part_ctx_t ctx,
    const BAMFusedQ16PartParams& params,
    cudaStream_t stream);

void bam_fused_q16part_destroy(bam_fused_q16part_ctx_t ctx);
