#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ── Forward declarations for BaM types (opaque in this header) ──
typedef void* bam_q16_fused_io_ctx_t;
typedef void* bam_ctrl_handle_t;  // defined in bam_kernel.cuh, redeclared here

// ============================================================
// Q16 Fused IO + Decomp + Filter Kernels
//
// Three persistent kernels for Q16 VCHAR columns, each following
// the Q13 warp-per-page pattern:
//   Phase 1: GPU-initiated NVMe IO (BaM page_cache)
//   Phase 2: PAR-32K LZ4 decompress (nvCOMPdx)
//   Phase 3: Column-specific filter/extraction
//
// All three kernels share a single fused IO context.
// ============================================================

// ── S_COMMENT: IO + decomp + KMP scan → excluded suppkeys ──

struct BAMq16FusedSCommentParams {
    const uint32_t* d_comp_sizes;      // [npages] compressed page sizes (nullptr if uncomp)
    const uint64_t* d_comp_offsets;    // [npages] disk byte offsets (nullptr if uncomp)
    const uint64_t* d_prefix_sum;      // [npages] cumulative row counts
    const uint64_t* d_s_suppkey_flat;  // [nrecs_supplier] preloaded S_SUPPKEY
    uint64_t*       d_excl_suppkeys;   // output: excluded suppkeys
    uint32_t*       d_excl_count;      // atomic counter for excluded count
    const char*     d_patterns;
    const int*      d_next;
    const int*      d_pattern_offsets;
    const int*      d_pattern_lengths;
    int             num_patterns;
    uint64_t        partition_start_lba;
    uint64_t        partition_start_lbas[4]; // per-device partition LBA (RAID0)
    uint32_t        n_devices;         // 1 = single device (default), >1 = RAID0 striping
    uint64_t        field_start_page_id; // for global page ID computation
    uint32_t        page_size;
    uint16_t        comp_method;       // 0=uncompressed, nonzero=compressed
    uint64_t        npages;
    uint32_t        num_blocks;        // grid size
    uint64_t*       d_phase_cycles;    // [3]: io, decomp, scan (clock64 cycles)
};

// ── P_BRAND: IO + decomp + CHAR(10) brand_id extraction ──

struct BAMq16FusedBrandParams {
    const uint32_t* d_comp_sizes;
    const uint64_t* d_comp_offsets;
    const uint64_t* d_prefix_sum;
    uint32_t        padded_len;        // 12 for CHAR(10)
    uint32_t*       d_brand_ids;       // output [nrecs_part]
    uint64_t        partition_start_lba;
    uint64_t        partition_start_lbas[4]; // per-device partition LBA (RAID0)
    uint32_t        n_devices;         // 1 = single device (default), >1 = RAID0 striping
    uint64_t        field_start_page_id; // for global page ID computation
    uint32_t        page_size;
    uint16_t        comp_method;       // 0=uncompressed, nonzero=compressed
    uint64_t        npages;
    uint32_t        num_blocks;
    uint64_t*       d_phase_cycles;
};

// ── P_TYPE: IO + decomp + VCHAR type_id dictionary extraction ──

struct BAMq16FusedTypeParams {
    const uint32_t* d_comp_sizes;
    const uint64_t* d_comp_offsets;
    const uint64_t* d_prefix_sum;
    uint64_t*       d_dict_keys;       // [512] FNV-1a hash keys
    uint32_t*       d_dict_type_ids;   // [512] type_id per dict slot
    char*           d_dict_strs;       // [512 * 32] string data
    uint16_t*       d_dict_lens;       // [512] string lengths
    uint32_t*       d_type_id_counter; // atomic counter for new type_ids
    uint32_t*       d_type_ids;        // output [nrecs_part]
    uint64_t        partition_start_lba;
    uint64_t        partition_start_lbas[4]; // per-device partition LBA (RAID0)
    uint32_t        n_devices;         // 1 = single device (default), >1 = RAID0 striping
    uint64_t        field_start_page_id; // for global page ID computation
    uint32_t        page_size;
    uint16_t        comp_method;       // 0=uncompressed, nonzero=compressed
    uint64_t        npages;
    uint32_t        num_blocks;
    uint64_t*       d_phase_cycles;
};

// ── Context management ──

bam_q16_fused_io_ctx_t bam_q16_fused_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_q16_fused_s_comment_async(
    bam_q16_fused_io_ctx_t ctx,
    const BAMq16FusedSCommentParams& params,
    cudaStream_t stream);

void bam_q16_fused_p_brand_async(
    bam_q16_fused_io_ctx_t ctx,
    const BAMq16FusedBrandParams& params,
    cudaStream_t stream);

void bam_q16_fused_p_type_async(
    bam_q16_fused_io_ctx_t ctx,
    const BAMq16FusedTypeParams& params,
    cudaStream_t stream);

void bam_q16_fused_io_destroy(bam_q16_fused_io_ctx_t ctx);
