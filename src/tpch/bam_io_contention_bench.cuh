#pragma once

#include "bam_io_device.cuh"
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// BaM I/O contention microbenchmark
//
// Measures I/O throughput as a function of (num_warps, ios_per_warp)
// to quantify QP contention effects.
//
// Each warp manages N outstanding async I/Os in a completion-driven
// loop (libaio-style): submit N → poll any → process → submit replacement.
// ============================================================

struct BamIoBenchConfig {
    uint32_t num_warps;       // CUDA blocks (1 warp per block)
    uint32_t ios_per_warp;    // outstanding I/Os per warp
};

struct BamIoBenchResult {
    uint32_t num_warps;
    uint32_t ios_per_warp;
    uint32_t total_outstanding;  // num_warps * ios_per_warp
    uint32_t total_pages;
    double   elapsed_ms;
    double   io_throughput_gbs;
    double   warps_per_qp;       // num_warps / (n_qps * n_devices)
};

// Run a single benchmark configuration.
// Returns result with timing and throughput.
BamIoBenchResult bam_io_contention_bench_run(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t total_pages,        // pages to read
    uint64_t field_start_page_id,// global page ID of first page
    uint32_t n_devices,
    const uint64_t* partition_start_lbas,  // host array [n_devices]
    const BamIoBenchConfig& config);
