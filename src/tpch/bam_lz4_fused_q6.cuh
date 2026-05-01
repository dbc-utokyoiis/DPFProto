#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q6.cuh — Fused BaM I/O + nvCOMPdx LZ4 decompress + Q6 scan
//
// Single persistent kernel: GPU-initiated NVMe IO → LZ4 decompress → Q6 predicate.
// 4 warps per block, each warp handles one field (L_SHIPDATE, L_QUANTITY,
// L_EXTENDEDPRICE, L_DISCOUNT). After decompression, all 128 threads
// evaluate Q6 predicates and accumulate revenue via atomicAdd.
// ============================================================

typedef void* bam_fused_q6_ctx_t;
typedef void* bam_ctrl_handle_t;  // defined in bam_kernel.cuh, redeclared here

struct BAMFusedQ6Params {
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

    // Q6 predicate parameters
    int32_t   sd_low;
    int32_t   sd_high;

    // Output
    int64_t*  d_revenue;           // atomicAdd target

    // Zone map (optional)
    const uint8_t* d_page_active;  // [npages] 1=active, 0=skip (or nullptr = all active)
};

// Create fused Q6 context (page_cache + decomp_buf allocation).
// num_blocks: CUDA grid size (typically min(npages, sm_count * 2)).
bam_fused_q6_ctx_t bam_fused_q6_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

// Launch fused Q6 kernel (synchronous — waits for completion).
void bam_fused_q6_run(
    bam_fused_q6_ctx_t ctx,
    const BAMFusedQ6Params& params);

// Launch fused Q6 kernel (asynchronous — returns immediately).
void bam_fused_q6_run_async(
    bam_fused_q6_ctx_t ctx,
    const BAMFusedQ6Params& params,
    cudaStream_t stream);

// Destroy context (free page_cache + decomp_buf).
void bam_fused_q6_destroy(bam_fused_q6_ctx_t ctx);

// ============================================================
// Decomp+Scan only kernel (no BaM IO — reads from GPU staging buffer)
// Used with separate BaM bulk read + pipeline overlap.
// No BaM IO functions → no register constraint → better occupancy.
// ============================================================

struct Q6DecompScanParams {
    const char* staging_io;           // [4][batch_count][page_size] compressed pages
    uint32_t    batch_count;          // active pages in this batch
    uint32_t    page_size;
    uint32_t    num_blocks;

    // Comp metadata (global arrays already on GPU)
    const uint32_t* d_comp_sizes[4];  // [npages] per field
    bool      is_compressed[4];

    // Mapping: local slot → original page index
    const uint32_t* d_batch_page_ids; // [batch_count]

    // Q6 predicates
    int32_t   sd_low;
    int32_t   sd_high;

    // Output
    int64_t*  d_revenue;
};

typedef void* q6_decomp_scan_ctx_t;

q6_decomp_scan_ctx_t q6_decomp_scan_create(
    uint32_t page_size,
    uint32_t num_blocks);

void q6_decomp_scan_async(
    q6_decomp_scan_ctx_t ctx,
    const Q6DecompScanParams& params,
    cudaStream_t stream);

void q6_decomp_scan_destroy(q6_decomp_scan_ctx_t ctx);

// ============================================================
// Producer-Consumer pattern for Q6 fusion:
//
// IO Producer kernel: BaM page_cache reads (no nvCOMPdx, low registers)
// Decomp+Scan Consumer kernel: nvCOMPdx LZ4 decomp + Q6 scan (high registers)
//
// Communication via ring buffer in page_cache memory:
//   ring_page[slot] = -1 → empty (IO producer can write)
//   ring_page[slot] = page_idx → filled with that page (consumer can read)
// ============================================================

struct Q6ProdConsParams {
    // Ring buffer protocol
    int32_t*  ring_page;             // [n_ring] -1=empty, page_idx=filled
    uint32_t  n_ring;
    uint32_t* d_io_counter;          // atomic page counter for IO producer
    uint32_t* d_scan_counter;        // atomic page counter for consumer
    uint32_t  total_pages;           // number of active pages
    const uint32_t* d_active_page_ids; // [total_pages] active→original page idx

    // Page geometry
    uint32_t  page_size;
    uint32_t  n_devices;

    // Per-field IO metadata (indexed by field 0..3)
    uint64_t  field_start_page_ids[4];
    uint64_t* d_comp_offsets[4];     // [npages] byte offset on disk (GPU)
    uint32_t* d_comp_sizes[4];       // [npages] compressed sizes (GPU)
    bool      is_compressed[4];
    uint64_t  partition_start_lbas[4]; // per-device partition LBA

    // Q6 predicates
    int32_t   sd_low;
    int32_t   sd_high;
    int64_t*  d_revenue;
};

// Query the maximum total blocks that can be co-resident for the
// producer-consumer kernel (depends on smem, registers, etc.)
uint32_t q6_prodcons_max_blocks(uint32_t page_size);

// Launch combined producer-consumer kernel (single kernel, async).
// blocks [0, io_blocks) → IO producer (BaM reads)
// blocks [io_blocks, io_blocks+consumer_blocks) → consumer (decomp+scan)
// d_decomp_buf: [consumer_blocks * 4 * page_size] decompression buffer.
// Page cache must have n_ring * 4 slots (4 fields per ring entry).
// IMPORTANT: io_blocks + consumer_blocks MUST NOT exceed q6_prodcons_max_blocks().
void q6_prodcons_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const Q6ProdConsParams& params,
    uint32_t io_blocks,
    uint32_t consumer_blocks,
    cudaStream_t stream);

// ============================================================
// Warp-Specialized Q6 kernel:
//
// 8 warps (256 threads) per block:
//   Warps 0-3: IO (BaM page reads, 1 field/warp)
//   Warps 4-7: Decomp (nvCOMPdx LZ4, 1 field/warp)
//   All 256 threads: Q6 scan
//
// Intra-block sync via __syncthreads() — no cross-block issues.
// ============================================================

struct Q6WarpSpecParams {
    uint64_t  field_start_page_ids[4];
    uint64_t* d_comp_offsets[4];
    uint32_t* d_comp_sizes[4];
    bool      is_compressed[4];
    uint64_t  partition_start_lbas[4]; // per-device partition LBA
    uint32_t  n_devices;
    uint32_t  page_size;
    uint32_t  total_pages;             // total number of pages (not just active), or num_active when d_active_page_ids set
    const uint32_t* d_active_page_ids; // global compact: dense active page IDs (nullptr = use d_page_mask)
    const uint8_t* d_page_mask;        // [total_pages] 1=active, 0=skip (nullptr = all active)
    int32_t   sd_low;
    int32_t   sd_high;
    int64_t*  d_revenue;
};

uint32_t q6_warp_spec_max_blocks(uint32_t page_size);

void q6_warp_spec_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base_addr,
    char* d_decomp_buf,
    const Q6WarpSpecParams& params,
    uint32_t num_blocks,
    cudaStream_t stream);
