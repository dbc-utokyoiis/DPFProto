#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ── Forward declarations for BaM types (opaque in this header) ──
typedef void* bam_q13_fused_io_ctx_t;
typedef void* bam_ctrl_handle_t;  // defined in bam_kernel.cuh, redeclared here

// ── Fused IO + Decomp + KMP Scan for Q13 O_COMMENT ──
// Single persistent kernel: GPU-initiated NVMe IO → PAR-32K LZ4 decompress → KMP scan.
// Block-stride loop over all pages (no host-side batching).

struct BAMq13FusedParams {
    const uint32_t* d_comp_sizes;      // [npages] compressed page sizes
    const uint64_t* d_comp_offsets;    // [npages] disk byte offsets
    const uint64_t* d_prefix_sum;      // [npages] cumulative row counts
    const uint64_t* d_o_custkey_flat;  // [nrecs_orders] preloaded O_CUSTKEY
    uint64_t*       d_o_aggr_custkey;  // [nrecs_orders] output (pre-filled 0xFF)
    uint64_t*       d_count;           // atomic counter for qualifying records
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
    uint64_t*       d_phase_cycles;    // [3]: io, decomp, scan (clock64 cycles, atomicAdd)
};

bam_q13_fused_io_ctx_t bam_q13_fused_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_q13_fused_io_decomp_scan(
    bam_q13_fused_io_ctx_t ctx,
    const BAMq13FusedParams& params);

void bam_q13_fused_io_decomp_scan_async(
    bam_q13_fused_io_ctx_t ctx,
    const BAMq13FusedParams& params,
    cudaStream_t stream);

void bam_q13_fused_io_destroy(bam_q13_fused_io_ctx_t ctx);

// ── Legacy: batch-based decomp+scan (separate IO) ──
// Decompress PAR-32K LZ4 + KMP scan for Q13 O_COMMENT.
// Decompresses a batch of compressed VCHAR pages, runs KMP multi-pattern
// matching ("special", "requests"), and writes qualifying O_CUSTKEY values
// to d_o_aggr_custkey (UINT64_MAX for LIKE matches, custkey for non-matches).
//
// d_staging_buf:    compressed pages [batch_pages * page_size] from IO v2.
// d_comp_sizes:     per-page compressed sizes [npages_total].
// d_decomp_buf:     scratch for decompressed pages [batch_pages * page_size].
// d_prefix_sum:     O_COMMENT prefix_sum [npages_total] (cumulative row counts).
// d_o_custkey_flat: preloaded O_CUSTKEY flat array [nrecs_orders].
// d_o_aggr_custkey: output array [nrecs_orders], pre-filled with 0xFF.
// d_count:          atomic counter for qualifying (NOT LIKE) records.
// batch_start:      index of first page in this batch.
// batch_pages:      number of pages to process (grid size).
// npages_total:     total O_COMMENT page count.
void bam_q13_decomp_scan_par32k_async(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    const uint64_t* d_prefix_sum,
    const uint64_t* d_o_custkey_flat,
    uint64_t*       d_o_aggr_custkey,
    uint64_t*       d_count,
    const char*     d_patterns,
    const int*      d_next,
    const int*      d_pattern_offsets,
    const int*      d_pattern_lengths,
    int             num_patterns,
    int             total_pattern_chars,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_pages,
    uint64_t        npages_total,
    cudaStream_t    stream);
