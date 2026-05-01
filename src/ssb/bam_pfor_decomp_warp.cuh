#pragma once

// bam_pfor_decomp_warp.cuh — PFOR warp-level decompression for BaM page cache slots
//
// Decompresses one PFOR-compressed page (from page cache) to a destination buffer.
// Output layout: [nalloc(4B)][watermark(4B)][lfreespace(4B)][int32_t data[...]]
// This matches LZ4 decomp output, so the same scan functions work unmodified.
//
// Shared memory: 32 bytes/warp (shared_bws[4] + shared_offs[4]).

#include "kernel/binpack_kernel.cuh"

struct bam_pfor_pag_head {
    uint32_t nalloc;
    uint32_t watermark;
    uint32_t lfreespace;
};
#define BAM_PFOR_HDR_BYTES 12

// Decompress one PFOR page from page cache slot to dst buffer.
// 32 threads (one warp) cooperate.
//   pc_base_addr: page cache base address
//   slot:         page cache slot index
//   dst:          output buffer (page_size bytes, laid out as header + int32 data)
//   page_size:    page size in bytes
//   is_compressed: true if PFOR-compressed, false if uncompressed
//   shared_bws:   __shared__ uint[4] per warp — miniblock bitwidths
//   shared_offs:  __shared__ uint[4] per warp — miniblock offsets
__device__ __forceinline__
void bam_pfor_decomp_warp(
    const char* pc_base_addr, uint32_t slot,
    char* dst, uint32_t page_size, bool is_compressed,
    uint* shared_bws, uint* shared_offs)
{
    const uint32_t lane = threadIdx.x & 31;
    const char* src = pc_base_addr + (uint64_t)slot * page_size;

    // Copy 12-byte page header to dst
    volatile bam_pfor_pag_head* hdr = (volatile bam_pfor_pag_head*)src;
    uint32_t nalloc    = hdr->nalloc;
    uint32_t watermark = hdr->watermark;

    if (lane == 0) {
        ((uint32_t*)dst)[0] = nalloc;
        ((uint32_t*)dst)[1] = watermark;
        ((uint32_t*)dst)[2] = hdr->lfreespace;
    }

    int32_t* decomp_out = (int32_t*)(dst + BAM_PFOR_HDR_BYTES);

    if (is_compressed) {
        // PFOR decode: watermark = padded count (multiple of 128)
        uint32_t nblocks = watermark / 128;
        uint32_t* block_start = (uint32_t*)(src + BAM_PFOR_HDR_BYTES);
        uint32_t* data_ptr = block_start + (nblocks + 1);

        for (uint32_t b = 0; b < nblocks; b++) {
            uint32_t* blk_data = data_ptr + block_start[b];

            // Threads 0-3: extract miniblock bitwidths and offsets
            if (lane < 4) {
                uint32_t mb_bw_packed = blk_data[1];
                uint32_t packed_off = (mb_bw_packed << 8)
                                    + (mb_bw_packed << 16)
                                    + (mb_bw_packed << 24);
                shared_bws[lane]  = (mb_bw_packed >> (lane << 3)) & 255;
                shared_offs[lane] = (packed_off >> (lane << 3)) & 255;
            }
            __syncwarp();

            // 32 threads × 4 iterations = 128 elements per block
            for (uint32_t k = 0; k < 4; k++) {
                uint32_t i      = k * 32 + lane;
                uint32_t mb_idx = i >> 5;
                uint32_t mb_pos = i & 31;
                int val = decodeElement(i, mb_idx, mb_pos,
                                        blk_data, shared_bws, shared_offs);
                uint32_t idx = b * 128 + i;
                if (idx < watermark) decomp_out[idx] = val;
            }
            __syncwarp();
        }
    } else {
        // Uncompressed: stride-32 copy
        const int32_t* raw = (const int32_t*)(src + BAM_PFOR_HDR_BYTES);
        for (uint32_t i = lane; i < nalloc; i += 32) {
            decomp_out[i] = raw[i];
        }
        __syncwarp();
    }
}
