#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q16_partsupp.cuh — Fused BaM I/O + nvCOMPdx LZ4 + Q16 PARTSUPP probe
//
// Single persistent kernel for Q16 PARTSUPP phase.
// Reads 2 INT64 fields (PS_PARTKEY, PS_SUPPKEY) per page, decompresses,
// probes PART HT + excluded supplier anti-join, and emits composite
// (group_id << 32 | ps_suppkey) pairs to d_emit_pairs[].
//
// Both fields are INT64 with the same page count (same table).
// ============================================================

typedef void* bam_fused_q16ps_ctx_t;
typedef void* bam_ctrl_handle_t;

struct BAMFusedQ16PSParams {
    // INT64 fields: 0=PS_PARTKEY, 1=PS_SUPPKEY
    uint64_t  field_start_page_ids[2];
    uint64_t* d_comp_offsets[2];
    uint32_t* d_comp_sizes[2];
    bool      is_compressed[2];

    // Prefix sum on GPU (exclusive, npages+1 entries)
    // ps[0]=0, ps[i]=cumulative nalloc through page i-1
    const uint64_t* d_ps;
    uint32_t  npages;

    // Multi-device
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;

    // Page geometry
    uint32_t  page_size;
    uint32_t  num_blocks;

    // Split IO/Compute tuning (0 = auto for both)
    uint32_t  split_compute_blocks;  // compute kernel blocks (0 = pages/4)
    uint32_t  split_batch_pages;     // pages per pipeline batch (0 = max/no pipeline)

    // PART hash table (keys + group_ids)
    const uint64_t* d_ht_keys;
    const uint32_t* d_ht_group_ids;
    uint32_t        ht_mask;

    // Excluded supplier hash table (anti-join)
    const uint64_t* d_excl_keys;
    uint32_t        excl_mask;

    // Output: d_emit_pairs[nrecs_partsupp]
    // Written as (group_id << 32) | (uint32_t)ps_suppkey, or UINT64_MAX
    uint64_t* d_emit_pairs;
};

bam_fused_q16ps_ctx_t bam_fused_q16ps_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q16ps_run_async(
    bam_fused_q16ps_ctx_t ctx,
    const BAMFusedQ16PSParams& params,
    cudaStream_t stream);

// Split IO/Compute variant: IO submission on stream_io, poll+decomp+probe on stream_comp.
// Uses the same context (same page_cache, same decomp buffer).
void bam_fused_q16ps_run_split_async(
    bam_fused_q16ps_ctx_t ctx,
    const BAMFusedQ16PSParams& params,
    cudaStream_t stream_io,
    cudaStream_t stream_comp);

// Split IO/Decomp/Probe variant: IO+decomp in one kernel, probe in separate kernel.
// Probe uses 128 threads/block (4× parallelism vs warp-level probe).
void bam_fused_q16ps_run_split_probe_async(
    bam_fused_q16ps_ctx_t ctx,
    const BAMFusedQ16PSParams& params,
    cudaStream_t stream_io,
    cudaStream_t stream_comp);

void bam_fused_q16ps_destroy(bam_fused_q16ps_ctx_t ctx);
