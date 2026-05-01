#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// Device-side LZ4 batch decompression using nvCOMPdx
//
// Decompresses multiple LZ4-compressed pages in a single kernel.
// Pages that are incompressible (comp_size >= page_size) are
// copied directly without decompression.
// ============================================================

// Batch decompress LZ4-compressed pages.
// d_comp_pages:   compressed input buffer (page_size stride per slot)
// d_decomp_pages: decompressed output buffer (page_size stride per page)
// d_comp_sizes:   per-page compressed sizes [npages]
// d_page_indices: per-slot → output page index mapping [npages]
// npages:         number of pages to decompress
// page_size:      page size in bytes (must be 65536 or 1048576)
// stream:         CUDA stream
void bam_lz4_batch_decompress(
    const char* d_comp_pages,
    char* d_decomp_pages,
    const uint32_t* d_comp_sizes,
    const uint32_t* d_page_indices,
    uint32_t npages,
    uint32_t page_size,
    cudaStream_t stream);
