#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// BaM IO device wrapper — compiled as C++11, device-linked.
//
// Provides opaque-pointer API so C++17 kernels (nvCOMPdx) can
// issue GPU-initiated NVMe reads without including BaM headers.
// ============================================================

// Opaque handle for page_cache resources.
typedef void* bam_io_page_cache_t;
typedef void* bam_ctrl_handle_t;  // same as bam_kernel.cuh

// ── Device function: NVMe page read ──
// Reads `nblk` NVMe blocks starting at `lba` into page_cache slot `bid`.
// Must be called by a single thread (e.g., tid==0).
__device__ void bam_io_read_page_device(
    void* ctrls,    // Controller** (opaque)
    void* pc,       // page_cache_d_t* (opaque)
    uint64_t lba,
    uint32_t nblk,
    uint32_t bid,
    uint32_t dev = 0);  // target device index (RAID0)

// ── Device function: NVMe page read (async submit only) ──
// Submits a read but does NOT poll for completion.
// Caller must later call bam_io_poll_page_device() with the returned qp/cid.
__device__ void bam_io_submit_page_device(
    void* ctrls,    // Controller** (opaque)
    void* pc,       // page_cache_d_t* (opaque)
    uint64_t lba,
    uint32_t nblk,
    uint32_t slot,
    uint32_t dev,
    void** out_qp,       // returned QueuePair* (opaque)
    uint16_t* out_cid);  // returned command ID

// ── Device function: NVMe page read (async submit, explicit QP selection) ──
// Like bam_io_submit_page_device, but uses qp_hint (not pc_slot) for QP selection.
// This allows pipeline batches to use non-overlapping QP ranges.
__device__ void bam_io_submit_page_device_qp(
    void* ctrls,
    void* pc,
    uint64_t lba,
    uint32_t nblk,
    uint32_t pc_slot,    // page_cache slot for DMA address
    uint32_t qp_hint,    // QP = qp_hint % n_qps (independent of pc_slot)
    uint32_t dev,
    void** out_qp,
    uint16_t* out_cid);

// ── Device function: poll completion of a submitted read ──
__device__ void bam_io_poll_page_device(
    void* qp,       // QueuePair* from bam_io_submit_page_device
    uint16_t cid);  // command ID from bam_io_submit_page_device

// ── Device function: non-blocking poll (try once) ──
// Single-pass CQ scan: returns true if completion found (dequeue + CID release done),
// false if not yet ready (caller should yield and retry).
__device__ bool bam_io_try_poll_page_device(
    void* qp,       // QueuePair* from bam_io_submit_page_device
    uint16_t cid);  // command ID from bam_io_submit_page_device

// ── Device function: CQ-order poll (consumes CQ head, returns CID) ──
// Polls the CQ head entry (whatever completed next), not a specific CID.
// Must be called by a single consumer per QP to avoid contention.
// Returns the CID of the completed command. Dequeue + CID release done internally.
__device__ uint16_t bam_io_poll_next_page_device(
    void* qp);      // QueuePair* — must have exclusive consumer access

// ── Host functions: page_cache lifecycle ──

// Create a page_cache with `num_blocks` entries (1 per CUDA block).
// Also sets cudaLimitStackSize = 8192 for BaM I/O.
bam_io_page_cache_t bam_io_page_cache_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

// Extract opaque pointers for kernel parameters.
void* bam_io_page_cache_get_d_ctrls(bam_io_page_cache_t pc);
void* bam_io_page_cache_get_d_pc_ptr(bam_io_page_cache_t pc);
void* bam_io_page_cache_get_base_addr(bam_io_page_cache_t pc);

// Destroy page_cache resources.
void bam_io_page_cache_destroy(bam_io_page_cache_t pc);

// Get the number of QueuePairs for a given device.
uint32_t bam_io_page_cache_get_n_qps(bam_io_page_cache_t pc, uint32_t dev);

// Get the device pointer to a specific QueuePair.
void* bam_io_page_cache_get_qp(bam_io_page_cache_t pc, uint32_t dev, uint32_t qp_idx);
