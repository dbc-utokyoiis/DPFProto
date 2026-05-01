#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_revenue.cuh — Fused BaM I/O + nvCOMPdx LZ4 decompress + Revenue scan
//
// Single persistent kernel: GPU-initiated NVMe IO → LZ4 decompress → Revenue predicate.
// 4 warps per block, each warp handles one field (L_SHIPDATE, L_QUANTITY,
// L_EXTENDEDPRICE, L_DISCOUNT). After decompression, all 128 threads
// evaluate revenue predicates and accumulate revenue via atomicAdd.
//
// Based on bam_lz4_fused_q6.cuh with different predicates:
//   Q6:      sd ∈ [low,high) && d ∈ [5,7] && q < 2400
//   Revenue: sd ∈ [low,high) && (qt_max==0 || q < qt_max)
// ============================================================

typedef void* bam_fused_revenue_ctx_t;
typedef void* bam_ctrl_handle_t;  // defined in bam_kernel.cuh, redeclared here

struct BAMFusedRevenueParams {
    // Per-field (0=L_SHIPDATE, 1=L_QUANTITY, 2=L_EXTENDEDPRICE, 3=L_DISCOUNT)
    uint64_t  field_start_page_ids[4];
    uint64_t* d_comp_offsets[4];   // [npages] byte offsets on disk (GPU)
    uint32_t* d_comp_sizes[4];     // [npages] compressed sizes (GPU)
    bool      is_compressed[4];

    // Multi-device
    uint64_t  partition_start_lbas[4]; // per-device partition LBA (RAID0)
    uint32_t  n_devices;

    // Page geometry
    uint32_t  page_size;
    uint64_t  npages;              // total pages (same for all 4 INT32 fields)
    uint32_t  num_blocks;          // grid size

    // Revenue predicate parameters
    int32_t   sd_low;
    int32_t   sd_high;
    int32_t   disc_lo;             // discount lower bound (inclusive), 0 = disabled
    int32_t   disc_hi;             // discount upper bound (inclusive), INT32_MAX = disabled
    int32_t   qt_max;              // quantity threshold (0 = disabled)

    // Output
    int64_t*  d_revenue;           // atomicAdd target

    // Zone map (optional)
    const uint8_t* d_page_active;  // [npages] 1=active, 0=skip (or nullptr = all active)
};

// Create fused revenue context (page_cache + decomp_buf allocation).
// num_blocks: CUDA grid size (typically min(npages, sm_count * 2)).
bam_fused_revenue_ctx_t bam_fused_revenue_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

// Launch fused revenue kernel (asynchronous — returns immediately).
void bam_fused_revenue_run_async(
    bam_fused_revenue_ctx_t ctx,
    const BAMFusedRevenueParams& params,
    cudaStream_t stream);

// Destroy context (free page_cache + decomp_buf).
void bam_fused_revenue_destroy(bam_fused_revenue_ctx_t ctx);

// ============================================================
// Decomp+Scan only kernel (no BaM IO — reads from GPU staging buffer)
// ============================================================

struct RevenueDecompScanParams {
    const char* staging_io;           // [4][batch_count][page_size] compressed pages
    uint32_t    batch_count;          // active pages in this batch
    uint32_t    page_size;
    uint32_t    num_blocks;

    // Comp metadata (global arrays already on GPU)
    const uint32_t* d_comp_sizes[4];  // [npages] per field
    bool      is_compressed[4];

    // Mapping: local slot → original page index
    const uint32_t* d_batch_page_ids; // [batch_count]

    // Revenue predicates
    int32_t   sd_low;
    int32_t   sd_high;
    int32_t   disc_lo;
    int32_t   disc_hi;
    int32_t   qt_max;

    // Output
    int64_t*  d_revenue;
};

typedef void* revenue_decomp_scan_ctx_t;

revenue_decomp_scan_ctx_t revenue_decomp_scan_create(
    uint32_t page_size,
    uint32_t num_blocks);

void revenue_decomp_scan_async(
    revenue_decomp_scan_ctx_t ctx,
    const RevenueDecompScanParams& params,
    cudaStream_t stream);

void revenue_decomp_scan_destroy(revenue_decomp_scan_ctx_t ctx);

// ============================================================
// Warp-Specialized Revenue kernel (1024 threads, 7 decomp groups):
//
// 32 warps per block:
//   Warps 0-3:   IO (BaM page reads, 1 field/warp)
//   Warps 4-31:  Decomp (7 groups × 4 warps, nvCOMPdx LZ4)
//   All 1024 threads: Revenue scan
//
// Batch=7 pages per batch, double-buffered IO/decomp pipeline.
// ============================================================

struct RevenueWarpSpecParams {
    uint64_t  field_start_page_ids[4];
    uint64_t* d_comp_offsets[4];
    uint32_t* d_comp_sizes[4];
    bool      is_compressed[4];
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;
    uint32_t  page_size;
    uint32_t  total_pages;             // total pages, or num_active when d_active_page_ids set
    const uint32_t* d_active_page_ids; // global compact: dense active page IDs (nullptr = use d_page_mask)
    const uint8_t* d_page_mask;        // GPU zonemap mask (nullptr = all active)
    int32_t   sd_low;
    int32_t   sd_high;
    int32_t   disc_lo;
    int32_t   disc_hi;
    int32_t   qt_max;              // quantity threshold (0 = disabled)
    int64_t*  d_revenue;
};

uint32_t revenue_warp_spec_max_blocks(uint32_t page_size);

void revenue_warp_spec_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const RevenueWarpSpecParams& params,
    uint32_t num_blocks,
    cudaStream_t stream);
