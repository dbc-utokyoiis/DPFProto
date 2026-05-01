#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// Q13 FSST Fused IO + Decomp + KMP Scan Kernel
//
// Persistent kernel for Q13 O_COMMENT (FSST compressed):
//   IO + FSST decomp + inline KMP → qualifying custkeys
//
// No decomp_buf: FSST decode in smem + registers.
// Double-buffered IO: 2 page_cache slots per block.
// Block-stride loop over all pages, __syncthreads().
//
// BaM IO is called via bam_io_device.cuh (compiled C++11, device-linked).
// Compiled as CUDA C++17 (no nvCOMPdx dependency).
// ============================================================

typedef void* bam_q13_fsst_io_ctx_t;
typedef void* bam_ctrl_handle_t;

struct BAMq13FsstParams {
    const uint32_t* d_comp_sizes;
    const uint64_t* d_comp_offsets;
    const uint64_t* d_prefix_sum;       // O_COMMENT prefix_sum
    const uint64_t* d_o_custkey_flat;   // flattened O_CUSTKEY
    uint64_t*       d_o_aggr_custkey;   // output: qualifying custkeys (or UINT64_MAX)
    uint64_t*       d_count;            // output: total qualifying count
    const char*     d_patterns;
    const int*      d_next;
    const int*      d_pattern_offsets;
    const int*      d_pattern_lengths;
    int             num_patterns;
    uint64_t        partition_start_lbas[4];
    uint32_t        n_devices;
    uint64_t        field_start_page_id;
    uint32_t        page_size;
    uint64_t        npages;
    uint32_t        num_blocks;
    uint64_t        nrecs_total;        // total O_COMMENT records (for bounds check)
    uint64_t*       d_phase_cycles;     // [3]: io, symtab_load, scan (optional)
};

// ── Context management ──

bam_q13_fsst_io_ctx_t bam_q13_fsst_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_q13_fsst_o_comment_async(
    bam_q13_fsst_io_ctx_t ctx,
    const BAMq13FsstParams& params,
    cudaStream_t stream);

void bam_q13_fsst_io_destroy(bam_q13_fsst_io_ctx_t ctx);
