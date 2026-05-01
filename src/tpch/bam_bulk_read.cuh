#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "bam_kernel.cuh"  // for bam_ctrl_handle_t

// ============================================================
// BaM Bulk Read — GPU-initiated NVMe page reads into GPU buffers
//
// Each BamBulkReadDesc describes a single page read:
//   lba, nblocks → NVMe read address and size
//   dest         → GPU destination pointer
//   copy_bytes   → bytes to copy from page cache slot to dest
//   device       → target device index (RAID0)
// ============================================================

struct BamBulkReadDesc {
    uint64_t lba;           // starting LBA (512-byte sectors)
    uint32_t nblocks;       // number of 512-byte blocks to read
    uint32_t device;        // target device index (RAID0)
    char*    dest;          // GPU destination buffer
    uint32_t copy_bytes;    // bytes to copy from page cache to dest
};

struct BamBulkReadResult {
    uint64_t io_count;      // number of I/O operations
    uint64_t io_bytes;      // total bytes read from NVMe
};

// Bulk-read pages from NVMe to GPU memory via BaM GPU-initiated I/O.
// descs:       host-side array of read descriptors (copied to GPU internally)
// ndescs:      number of descriptors
// page_size:   page size in bytes (for page cache slot sizing)
// num_blocks:  number of CUDA blocks for the kernel
// stream:      CUDA stream for async operations
BamBulkReadResult bam_bulk_read(
    bam_ctrl_handle_t ctrl,
    const BamBulkReadDesc* descs,
    uint32_t ndescs,
    uint32_t page_size,
    uint32_t num_blocks,
    cudaStream_t stream);

// ============================================================
// Persistent context for pipelined bulk reads.
// Create once, reuse across batches — avoids per-batch
// page_cache create/destroy and descriptor alloc/free.
// ============================================================
struct BamBulkReadCtx {
    void*    pc;                  // bam_io_page_cache_t (created once)
    void*    d_ctrls;             // cached from pc
    void*    d_pc;                // cached from pc
    void*    pc_base;             // cached from pc
    BamBulkReadDesc* d_descs[2];  // double-buffered descriptor buffer on GPU
    BamBulkReadDesc* h_descs[2];  // double-buffered pinned host staging
    cudaEvent_t      h2d_done[2]; // signaled after H2D memcpy of h_descs[i]
    uint32_t max_descs;           // capacity per buffer
    uint32_t page_size;
    uint32_t num_blocks;
    uint32_t slots_per_block;     // page_cache slots per block (1=original, 2-4=pipelined IO)
};

// Create a persistent bulk-read context.
// ctrl:       BaM controller handle
// page_size:  page size in bytes
// num_blocks: number of CUDA blocks for the kernel (typically sm_count)
// max_descs:  maximum number of descriptors per batch
BamBulkReadCtx bam_bulk_read_ctx_create(
    bam_ctrl_handle_t ctrl,
    uint32_t page_size,
    uint32_t num_blocks,
    uint32_t max_descs,
    uint32_t slots_per_block = 1);

void bam_bulk_read_ctx_destroy(BamBulkReadCtx& ctx);

// Async bulk read: launches kernel without synchronization.
// Caller must use cudaEvent / cudaStreamSynchronize for ordering.
// Caller writes descriptors directly into ctx.h_descs[pipe_idx] before calling.
// pipe_idx selects which double-buffer slot to use (0 or 1).
void bam_bulk_read_async(
    BamBulkReadCtx& ctx,
    uint32_t ndescs,
    int pipe_idx,
    cudaStream_t stream);

// ============================================================
// Zero-copy API: NVMe reads into page_cache, no GPU copy.
// One page_cache slot per descriptor.
// After completion, data for desc j is at:
//   (char*)ctx.pc_base + j * slot_size
// ============================================================

// Create context: page_cache has max_descs slots of slot_size bytes.
// slot_size: page_cache slot size (= page_size, or smaller for compressed-only data)
// num_blocks: CUDA blocks for kernel (e.g., sm_count * 16 for high NVMe concurrency)
BamBulkReadCtx bam_bulk_read_nocopy_ctx_create(
    bam_ctrl_handle_t ctrl,
    uint32_t slot_size,
    uint32_t num_blocks,
    uint32_t max_descs);

// Async nocopy read: NVMe-only, no copy. 32 threads/block.
// Uploads descriptors (H2D) and launches kernel.
void bam_bulk_read_nocopy_async(
    BamBulkReadCtx& ctx,
    uint32_t ndescs,
    int pipe_idx,
    cudaStream_t stream);

// Per-field nocopy launch: reads a subset of already-uploaded descriptors.
// desc_offset: starting index into on-GPU descriptor array.
// ndescs: count for this field.  slot_base: page_cache slot offset.
// Caller must upload descriptors first (via nocopy_async or manual H2D).
void bam_bulk_read_nocopy_field_async(
    BamBulkReadCtx& ctx,
    uint32_t desc_offset,
    uint32_t ndescs,
    uint32_t slot_base,
    int pipe_idx,
    cudaStream_t stream);

// ============================================================
// BaM Zonemap: fused NVMe stats read + GPU predicate evaluation
// in a single kernel launch.
//
// Flow:
//   1. Create ctx outside timing (Rule 4)
//   2. Fill h_reads[] and h_preds[] from metadata (outside timing)
//   3. Inside timing: bam_zonemap_eval_async() → cudaStreamSync
//      → read h_mask[0..npages-1]
//   4. Destroy ctx after use
// ============================================================

// One NVMe page read for stats data.
// page_cache slot = sequential index in the reads array.
struct BamZonemapStatsRead {
    uint64_t lba;
    uint32_t nblocks;
    uint32_t device;
};

// One predicate evaluated against stats in page_cache.
struct BamZonemapPred {
    uint32_t stats_page_offset;  // first page_cache slot for this source
    uint64_t nstats;
    int32_t  pred_lo;
    int32_t  pred_hi;
};

static constexpr uint32_t kBamZonemapMaxReads = 32;
static constexpr uint32_t kBamZonemapMaxPreds = 4;

struct BamZonemapCtx {
    void* d_ctrls;
    void* d_pc;
    void* pc_base;
    BamZonemapStatsRead* d_reads;
    BamZonemapStatsRead* h_reads;   // pinned
    BamZonemapPred* d_preds;
    BamZonemapPred* h_preds;        // pinned
    uint8_t* d_mask;
    uint8_t* h_mask;                // pinned
    uint32_t* d_active_ids;         // compact output: dense active page IDs
    uint32_t* d_num_active;         // compact output: count (1 element)
    uint32_t* h_num_active;         // pinned host mirror of d_num_active
    uint32_t page_size;
    uint64_t max_npages;

    // Optional Phase 4: derive INT64 page mask from INT32 mask.
    // Set d_ps_i32 != nullptr to enable.  Prefix sums are truncated
    // format (npages entries, ps[i] = cumulative nalloc after page i).
    const uint64_t* d_ps_i32  = nullptr;
    const uint64_t* d_ps_i64  = nullptr;
    uint8_t*        d_mask_i64 = nullptr;   // output: caller-allocated [npages_i64]
    uint32_t        npages_i64 = 0;
};

// Create a zonemap context that borrows an existing page_cache.
// The page_cache must have at least kBamZonemapMaxReads slots of page_size.
BamZonemapCtx bam_zonemap_ctx_create(
    void* d_ctrls, void* d_pc, void* pc_base,
    uint32_t page_size,
    uint64_t max_npages);

void bam_zonemap_ctx_destroy(BamZonemapCtx& ctx);

// Fused eval: H2D reads+preds → kernel (IO + eval) → D2H mask.
// After cudaStreamSynchronize, h_mask[pg] is 1 for active pages.
void bam_zonemap_eval_async(
    BamZonemapCtx& ctx,
    uint64_t npages,
    uint32_t nreads,
    uint32_t npreds,
    cudaStream_t stream);

// Pre-issue a single dummy IO to initialize BaM page_cache DMA registration.
// Call before total_start so that the lazy init cost is excluded from timing.
void bam_pre_io(void* d_ctrls, void* d_pc, cudaStream_t stream);
