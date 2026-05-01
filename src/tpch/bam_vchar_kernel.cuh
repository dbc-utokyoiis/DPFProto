#pragma once

#include <cstdint>

// Result returned from scan_o_comment.
struct BAMVcharResult {
    uint64_t total_records;    // total VCHAR record count
    uint64_t total_strlen;     // sum of all VCHAR string lengths
    uint64_t total_byte_sum;   // sum of all bytes in all VCHAR strings (checksum)
};

// Decompress + scan a batch of LZ4-compressed VCHAR pages.
// d_staging_buf: device buffer with compressed page data [batch_size * page_size].
// d_comp_sizes:  device array [npages] of per-page compressed sizes.
// d_decomp_buf:  device scratch for decompressed pages [batch_size * page_size].
// d_total_*:     device accumulators (incremented, not reset).
// batch_start:   index of first page in this batch.
// batch_size:    number of pages to process (<= num_blocks).
void bam_vchar_decomp_scan_batch(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_size);

// Decompress + scan v2: each warp decompresses its page, all 128 threads scan.
// batch_blocks: number of thread blocks (each handles 4 pages).
// npages_total: total page count (for bounds checking at tail).
void bam_vchar_decomp_scan_batch_v2(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_blocks,
    uint64_t        npages_total);

// Decompress + scan v3: same as v2 but with cp.async for byte-sum scan.
// Async: launches kernel on caller-provided stream without synchronizing.
void bam_vchar_decomp_scan_batch_v3_async(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_blocks,
    uint64_t        npages_total,
    cudaStream_t    stream);

// Decompress + scan v6: PAR-32K nvCOMPdx warp-cooperative decompression.
// 128 threads (4 warps), each warp decompresses 1 page (32 × 32KiB chunks).
// All 128 threads cooperatively scan decompressed pages.
// Async: launches kernel on caller-provided stream without synchronizing.
void bam_vchar_decomp_scan_par32k_async(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_blocks,
    uint64_t        npages_total,
    cudaStream_t    stream);

// Decompress + scan v7: PAR-8K nvCOMPdx warp-cooperative decompression.
// 128 threads (4 warps), 1 page per block.
// Each warp decompresses 32 × 8KiB chunks (= 256KiB, 1/4 of page).
// All 128 threads cooperatively scan 1 decompressed page.
// batch_pages: grid size = number of pages in this batch.
void bam_vchar_decomp_scan_par8k_async(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_pages,
    uint64_t        npages_total,
    cudaStream_t    stream);

// Decompress + scan v8: PAR-32K nvCOMPdx, 1-page-per-block.
// 128 threads (4 warps), 1 page per block.
// Each warp decompresses 8 × 32KiB chunks (= 256KiB, 1/4 of page).
// All 128 threads cooperatively scan 1 decompressed page.
// batch_pages: grid size = number of pages in this batch.
void bam_vchar_decomp_scan_par32k_v8_async(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_pages,
    uint64_t        npages_total,
    cudaStream_t    stream);

// Decompress only (no scan): PAR-32K nvCOMPdx, 1-page-per-block.
// 128 threads (4 warps), 1 page per block.
// Decompresses compressed VCHAR/CHAR pages from staging buffer
// directly into d_output_pages at the correct page offset:
//   d_output_pages[(batch_start + bid) * page_size]
// No scratch/decomp buffer needed.
// batch_pages: grid size = number of pages in this batch.
void bam_vchar_decomp_par32k_async(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_output_pages,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_pages,
    uint64_t        npages_total,
    cudaStream_t    stream);
