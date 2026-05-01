#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_split_q16_partsupp.cuh — Split IO/Compute for Q16 PARTSUPP
//
// Kernel 1 (IO Submit):  Lightweight, submits all NVMe reads at once.
//                         Stores (qp, cid) completion info to GPU queue.
// Kernel 2 (Compute):    Polls completions + nvCOMPdx decomp + HT probe.
//
// Key advantage: IO submission parallelism and compute parallelism
// are independently tunable.  Compute kernel polls are near-free
// because all IOs were pre-submitted by Kernel 1.
// ============================================================

typedef void* bam_split_q16ps_ctx_t;
typedef void* bam_ctrl_handle_t;

// Reuse the same parameter layout as the fused kernel.
struct BAMSplitQ16PSParams {
    // INT64 fields: 0=PS_PARTKEY, 1=PS_SUPPKEY
    uint64_t  field_start_page_ids[2];
    uint64_t* d_comp_offsets[2];
    uint32_t* d_comp_sizes[2];
    bool      is_compressed[2];

    // Prefix sum on GPU (exclusive, npages+1 entries)
    const uint64_t* d_ps;
    uint32_t  npages;

    // Multi-device
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;

    // Page geometry
    uint32_t  page_size;

    // PART hash table (keys + group_ids)
    const uint64_t* d_ht_keys;
    const uint32_t* d_ht_group_ids;
    uint32_t        ht_mask;

    // Excluded supplier hash table (anti-join)
    const uint64_t* d_excl_keys;
    uint32_t        excl_mask;

    // Output: d_emit_pairs[nrecs_partsupp]
    uint64_t* d_emit_pairs;
};

// Create a split IO/compute context.
// page_size:       page size in bytes
// max_batch_pages: max pages per batch (page_cache = max_batch_pages * 2 slots)
// compute_blocks:  CUDA blocks for the compute kernel (tunable independently)
bam_split_q16ps_ctx_t bam_split_q16ps_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t max_batch_pages,
    uint32_t compute_blocks);

// Run the split IO/compute pipeline.
// stream_io:   stream for IO submission kernel
// stream_comp: stream for compute kernel (poll + decomp + probe)
void bam_split_q16ps_run_async(
    bam_split_q16ps_ctx_t ctx,
    const BAMSplitQ16PSParams& params,
    cudaStream_t stream_io,
    cudaStream_t stream_comp);

void bam_split_q16ps_destroy(bam_split_q16ps_ctx_t ctx);
