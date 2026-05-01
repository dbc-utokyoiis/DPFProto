#pragma once

#include <cstdint>
#include "bam_io_device.cuh"

struct LZ4DecompBenchResult {
    uint32_t num_warps;
    uint32_t total_pages;
    uint32_t page_size;
    double   elapsed_ms;
    double   decomp_throughput_gbs;  // decompressed bytes / time
    double   us_per_page_per_warp;   // average latency per page per warp
};

// Run LZ4 decomp-only benchmark for a single field.
//
// Phase 1: pre-load compressed pages into page_cache via BaM sync I/O.
// Phase 2: run decomp-only kernel (no I/O) and measure with cudaEvents.
//
// h_comp_sizes:  [total_pages] compressed size per page (nullptr if uncompressed)
// h_comp_offsets: [total_pages] compressed byte offset per page (nullptr if uncompressed)
LZ4DecompBenchResult bam_lz4_decomp_bench_run(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t total_pages,
    uint64_t field_start_page_id,
    uint32_t n_devices,
    const uint64_t* partition_start_lbas,
    const uint32_t* h_comp_sizes,
    const uint64_t* h_comp_offsets,
    uint32_t num_warps);
