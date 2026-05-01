#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q16_phase01.cuh — Fused BaM I/O + nvCOMPdx LZ4 kernels
// for Q16 Phase 0 (SUPPLIER) and Phase 1 (PART) smaller fields.
//
// Shared context (page cache) for all kernels.
// 4 warps/block, __launch_bounds__(128, 8).
// ============================================================

typedef void* bam_fused_q16p01_ctx_t;
typedef void* bam_ctrl_handle_t;

// ── Common IO metadata (shared by all params) ──
struct BAMFusedQ16IOBase {
    uint64_t  field_start_page_id;
    const uint64_t* d_comp_offsets;
    const uint32_t* d_comp_sizes;
    bool      is_compressed;
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;
    uint32_t  page_size;
    uint64_t  npages;
    uint32_t  num_blocks;
    const uint64_t* d_prefix_sum;  // inclusive cumulative [npages]
};

// ── INT64 flatten ──
struct BAMFusedQ16FlattenI64Params : BAMFusedQ16IOBase {
    uint64_t* d_output;  // [nrecs]
};

// ── INT32 flatten → uint32_t (skip u64 intermediate) ──
struct BAMFusedQ16FlattenI32Params : BAMFusedQ16IOBase {
    uint32_t* d_output;  // [nrecs]
};

// ── INT32 flatten → uint64_t (zero-extend, for HT key/value usage) ──
struct BAMFusedQ16FlattenI32WidenParams : BAMFusedQ16IOBase {
    uint64_t* d_output;  // [nrecs]
};

// ── CHAR brand_id extraction ──
struct BAMFusedQ16BrandParams : BAMFusedQ16IOBase {
    uint32_t  padded_len;
    uint32_t* d_brand_ids;  // [nrecs]
};

// ── VCHAR S_COMMENT KMP scan → excluded suppkeys ──
struct BAMFusedQ16SupplierScanParams : BAMFusedQ16IOBase {
    const uint64_t* d_s_suppkey_flat;  // pre-flattened S_SUPPKEY
    const char* d_patterns;
    const int*  d_next;
    const int*  d_pattern_offsets;
    const int*  d_pattern_lengths;
    int         num_patterns;
    uint64_t*   d_excl_suppkeys;  // output
    uint32_t*   d_excl_count;     // atomic counter
};

bam_fused_q16p01_ctx_t bam_fused_q16p01_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_slots);

void bam_fused_q16p01_flatten_i64_async(
    bam_fused_q16p01_ctx_t ctx,
    const BAMFusedQ16FlattenI64Params& params,
    cudaStream_t stream);

void bam_fused_q16p01_flatten_i32_async(
    bam_fused_q16p01_ctx_t ctx,
    const BAMFusedQ16FlattenI32Params& params,
    cudaStream_t stream);

void bam_fused_q16p01_flatten_i32_widen_async(
    bam_fused_q16p01_ctx_t ctx,
    const BAMFusedQ16FlattenI32WidenParams& params,
    cudaStream_t stream);

void bam_fused_q16p01_brand_async(
    bam_fused_q16p01_ctx_t ctx,
    const BAMFusedQ16BrandParams& params,
    cudaStream_t stream);

void bam_fused_q16p01_supplier_scan_async(
    bam_fused_q16p01_ctx_t ctx,
    const BAMFusedQ16SupplierScanParams& params,
    cudaStream_t stream);

void bam_fused_q16p01_destroy(bam_fused_q16p01_ctx_t ctx);

// ── Split IO/Decomp API ──
// Submit IOs for pages [pg_base..pg_base+batch_np) of a field, then
// poll+decompress them. After return, decompressed data is contiguous at:
//   bam_fused_q16p01_get_decomp_buf(ctx) + [0 .. batch_np * page_size)
// stream_io: used for IO submit kernel
// stream_comp: used for poll+decomp kernel (waits for submit via event)
// slot_base: starting page_cache slot (for parallel streams with slot partitioning)
void bam_fused_q16p01_split_batch(
    bam_fused_q16p01_ctx_t ctx,
    const BAMFusedQ16IOBase& io,
    uint32_t pg_base,
    uint32_t batch_np,
    uint32_t slot_base,
    cudaStream_t stream_io,
    cudaStream_t stream_comp);

// Submit IOs only (no decomp). slot_base partitions the page_cache slots.
void bam_fused_q16p01_split_submit_only(
    bam_fused_q16p01_ctx_t ctx,
    const BAMFusedQ16IOBase& io,
    uint32_t pg_base,
    uint32_t batch_np,
    uint32_t slot_base,
    cudaStream_t stream_io);

// Decomp only (poll + LZ4). Must be called after submit completes.
void bam_fused_q16p01_split_decomp_only(
    bam_fused_q16p01_ctx_t ctx,
    uint32_t page_size,
    uint32_t batch_np,
    uint32_t slot_base,
    cudaStream_t stream_comp);

// Get pointer to decompressed data buffer (for scan kernels after split_batch).
char* bam_fused_q16p01_get_decomp_buf(bam_fused_q16p01_ctx_t ctx);

// Get max batch size (page_cache slots / 2 for safety, or total slots).
uint32_t bam_fused_q16p01_get_num_slots(bam_fused_q16p01_ctx_t ctx);

// Get the submit-done event (for manual record+wait between submit_only and decomp_only).
cudaEvent_t bam_fused_q16p01_get_submit_event(bam_fused_q16p01_ctx_t ctx);
