#pragma once
#include "fsst.h"
#include <cstdint>
#include <cstddef>

// AVX512-optimized FSST batch decompression (to output buffer)
// Uses 8-code AVX512 gather for no-escape fast path
void fsst_decompress_avx512_output(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    unsigned char* output,
    uint32_t* decomp_lens,
    uint32_t slot_size);

// AVX512-optimized FSST batch decompression (checksum-only)
// Computes FNV-1a inline, no output buffer
void fsst_decompress_avx512_cksum(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    uint32_t* checksums,
    uint32_t* decomp_lens);

// 4-way interleaved decompression compiled with g++ -O3 (not nvcc)
// Output-to-buffer variant
void fsst_decompress_interleaved4_output_native(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    unsigned char* output,
    uint32_t* decomp_lens,
    uint32_t slot_size);

// 4-way interleaved decompression compiled with g++ -O3 (not nvcc)
// Checksum-only variant
void fsst_decompress_interleaved4_cksum_native(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    uint32_t* checksums,
    uint32_t* decomp_lens);

// Cache-resident output: decompress to small per-thread buffer (stays in L2/L3)
// Each thread reuses a 256B buffer for each string, so no DRAM write traffic.
void fsst_decompress_cache_output_native(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    uint32_t* decomp_lens);

// AVX512 gather cache-resident output
void fsst_decompress_avx512_cache_output(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    uint32_t* decomp_lens);
