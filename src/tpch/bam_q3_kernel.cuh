#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ── Forward declarations for BaM types (opaque in this header) ──
typedef void* bam_q3_fused_io_ctx_t;
typedef void* bam_ctrl_handle_t;  // defined in bam_kernel.cuh, redeclared here

// ============================================================
// Q3 Fused IO + Decomp + Filter Kernel
//
// Persistent kernel for C_MKTSEGMENT (CHAR(10) lz4par):
//   Phase 1: GPU-initiated NVMe IO (BaM page_cache)
//   Phase 2: PAR-32K LZ4 decompress (nvCOMPdx)
//   Phase 3: BUILDING filter + custkey hash set insert
//
// Block-per-page with cooperative decomp + double-buffered IO.
// 128 threads/block (4 warps), all warps cooperate on decompress.
// ============================================================

struct BAMq3FusedMktsegParams {
    const uint32_t* d_comp_sizes;      // [npages] compressed page sizes (nullptr if uncomp)
    const uint64_t* d_comp_offsets;    // [npages] disk byte offsets (nullptr if uncomp)
    const uint64_t* d_prefix_sum;      // [npages] cumulative row counts
    uint32_t        padded_len;        // 12 for CHAR(10)
    const uint64_t* d_c_custkey_flat;  // preloaded C_CUSTKEY flat array
    uint64_t*       d_custkey_set;     // output: BUILDING custkey hash set
    uint32_t        custkey_set_mask;
    uint64_t        partition_start_lba;
    uint64_t        partition_start_lbas[4]; // per-device partition LBA (RAID0)
    uint32_t        n_devices;         // 1 = single device (default), >1 = RAID0 striping
    uint64_t        field_start_page_id; // for global page ID computation
    uint32_t        page_size;
    uint64_t        npages;
    uint32_t        num_blocks;
    uint16_t        comp_method;       // 0=NONE, 9=LZ4PAR
    // Q3SEL: multi-segment support (0 = BUILDING only, >0 = check segment_values)
    uint32_t        num_segments;
    uint64_t        segment_values[5];
};

// ── Context management ──

bam_q3_fused_io_ctx_t bam_q3_fused_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_q3_fused_mktseg_async(
    bam_q3_fused_io_ctx_t ctx,
    const BAMq3FusedMktsegParams& params,
    cudaStream_t stream);

void bam_q3_fused_io_destroy(bam_q3_fused_io_ctx_t ctx);

// ============================================================
// Q3 FSST Fused IO + Decomp + BUILDING Filter Kernel
//
// For C_MKTSEGMENT stored with FSST compression:
//   Phase 1: GPU-initiated NVMe IO (BaM page_cache, double-buffered)
//   Phase 2: FSST decompress → compare first 8 bytes with "BUILDING"
//   Phase 3: If match, insert custkey into hash set
//
// Same IO context as LZ4PAR variant (no decomp_buf needed for FSST).
// ============================================================

typedef void* bam_q3_fsst_io_ctx_t;

struct BAMq3FsstMktsegParams {
    const uint32_t* d_comp_sizes;      // [npages] compressed page sizes
    const uint64_t* d_comp_offsets;    // [npages] disk byte offsets
    const uint64_t* d_prefix_sum;      // [npages] cumulative row counts
    const uint64_t* d_c_custkey_flat;  // preloaded C_CUSTKEY flat array
    uint64_t*       d_custkey_set;     // output: BUILDING custkey hash set
    uint32_t        custkey_set_mask;
    uint64_t        partition_start_lbas[4];
    uint32_t        n_devices;
    uint64_t        field_start_page_id;
    uint32_t        page_size;
    uint64_t        npages;
    uint32_t        num_blocks;
    // Q3SEL: multi-segment support (0 = BUILDING only, >0 = check segment_values)
    uint32_t        num_segments;
    uint64_t        segment_values[5];
};

bam_q3_fsst_io_ctx_t bam_q3_fsst_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_q3_fused_mktseg_fsst_async(
    bam_q3_fsst_io_ctx_t ctx,
    const BAMq3FsstMktsegParams& params,
    cudaStream_t stream);

void bam_q3_fsst_io_destroy(bam_q3_fsst_io_ctx_t ctx);
