#pragma once
#include <cstdint>
#include <cstddef>

struct NvcompdxLz4BenchResult {
    uint32_t num_warps;
    uint32_t total_pages;
    uint32_t page_size;
    double   elapsed_ms;
    double   decomp_throughput_gbs;  // decompressed bytes / elapsed
    double   us_per_page_per_warp;
};

// Pure nvCOMPdx LZ4 decomp benchmark.
// Compressed pages are already in GPU memory (d_comp_buf, packed contiguously
// at offsets d_comp_offsets[0..total_pages-1]).
// Decompressed output goes to d_decomp_buf.
NvcompdxLz4BenchResult nvcompdx_lz4_bench_run(
    const char* d_comp_buf,         // GPU: packed compressed pages
    const uint64_t* d_comp_offsets, // GPU: [total_pages] byte offset of each compressed page
    const uint32_t* d_comp_sizes,   // GPU: [total_pages] compressed size of each page
    char* d_decomp_buf,             // GPU: [num_warps * page_size] output buffer
    uint32_t total_pages,
    uint32_t page_size,
    uint32_t num_warps);
