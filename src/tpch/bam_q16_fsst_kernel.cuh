#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// Q16 FSST Fused IO + Decomp + Filter Kernels
//
// Three persistent kernels for Q16 FSST-compressed VCHAR/CHAR columns:
//   1. S_COMMENT: IO + FSST decomp + inline KMP scan → excluded suppkeys
//   2. P_BRAND:   IO + FSST decomp + CHAR(10) brand_id extraction
//   3. P_TYPE:    IO + FSST decomp + VCHAR type_id dictionary extraction
//
// No decomp_buf needed (FSST decode in smem + registers).
// Double-buffered IO via BaM page_cache (2 slots per block).
// ============================================================

typedef void* bam_q16_fsst_io_ctx_t;
typedef void* bam_ctrl_handle_t;

// ── S_COMMENT: IO + FSST decomp + KMP scan → excluded suppkeys ──

struct BAMq16FsstSCommentParams {
    const uint32_t* d_comp_sizes;
    const uint64_t* d_comp_offsets;
    const uint64_t* d_prefix_sum;
    const uint64_t* d_s_suppkey_flat;
    uint64_t*       d_excl_suppkeys;
    uint32_t*       d_excl_count;
    const char*     d_patterns;
    const int*      d_next;
    const int*      d_pattern_offsets;
    const int*      d_pattern_lengths;
    int             num_patterns;
    uint64_t        partition_start_lba;
    uint64_t        partition_start_lbas[4];
    uint32_t        n_devices;
    uint64_t        field_start_page_id;
    uint32_t        page_size;
    uint64_t        npages;
    uint32_t        num_blocks;
    uint64_t*       d_phase_cycles;
};

// ── P_BRAND: IO + FSST decomp + brand_id extraction ──

struct BAMq16FsstBrandParams {
    const uint32_t* d_comp_sizes;
    const uint64_t* d_comp_offsets;
    const uint64_t* d_prefix_sum;
    uint32_t*       d_brand_ids;
    uint64_t        partition_start_lba;
    uint64_t        partition_start_lbas[4];
    uint32_t        n_devices;
    uint64_t        field_start_page_id;
    uint32_t        page_size;
    uint64_t        npages;
    uint32_t        num_blocks;
    uint64_t*       d_phase_cycles;
};

// ── P_TYPE: IO + FSST decomp + type_id dictionary extraction ──

struct BAMq16FsstTypeParams {
    const uint32_t* d_comp_sizes;
    const uint64_t* d_comp_offsets;
    const uint64_t* d_prefix_sum;
    uint64_t*       d_dict_keys;
    uint32_t*       d_dict_type_ids;
    char*           d_dict_strs;
    uint16_t*       d_dict_lens;
    uint32_t*       d_type_id_counter;
    uint32_t*       d_type_ids;
    uint64_t        partition_start_lba;
    uint64_t        partition_start_lbas[4];
    uint32_t        n_devices;
    uint64_t        field_start_page_id;
    uint32_t        page_size;
    uint64_t        npages;
    uint32_t        num_blocks;
    uint64_t*       d_phase_cycles;
};

// ── Context management ──

bam_q16_fsst_io_ctx_t bam_q16_fsst_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_q16_fsst_s_comment_async(
    bam_q16_fsst_io_ctx_t ctx,
    const BAMq16FsstSCommentParams& params,
    cudaStream_t stream);

void bam_q16_fsst_p_brand_async(
    bam_q16_fsst_io_ctx_t ctx,
    const BAMq16FsstBrandParams& params,
    cudaStream_t stream);

void bam_q16_fsst_p_type_async(
    bam_q16_fsst_io_ctx_t ctx,
    const BAMq16FsstTypeParams& params,
    cudaStream_t stream);

void bam_q16_fsst_io_destroy(bam_q16_fsst_io_ctx_t ctx);
