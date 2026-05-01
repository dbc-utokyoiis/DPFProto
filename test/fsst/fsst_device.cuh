#pragma once
#include "fsst_page.h"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

// ============================================================
// GPU Decompression Kernels for FSST String Pages
//
// Kernel launch:
//   grid  = total_comp_blocks (1 block per comp block)
//   block = FSST_BLOCK_SIZE (128 threads)
//   smem  = fsst_coalesced_smem_bytes(cb_metas, total_comp_blocks, max_decomp_len)
// ============================================================

static constexpr int FSST_BLOCK_SIZE = 128;

// ── Compute shared memory size for coalesced kernel ──

inline uint32_t fsst_coalesced_smem_bytes(
    const FsstCompBlockKernelMeta* h_cb_metas,
    uint32_t                       total_comp_blocks,
    uint32_t                       max_decomp_len)
{
    uint32_t max_cb_data = 0, max_nrecs = 0;
    for (uint32_t i = 0; i < total_comp_blocks; i++) {
        if (h_cb_metas[i].comp_block_data_size > max_cb_data)
            max_cb_data = h_cb_metas[i].comp_block_data_size;
        if (h_cb_metas[i].nrecs > max_nrecs)
            max_nrecs = h_cb_metas[i].nrecs;
    }
    uint32_t offsets_start = FSST_SMEM_SYMTAB_OFFSET + ((max_cb_data + 3) & ~3u);
    uint32_t staging_start = offsets_start + (max_nrecs + 1) * sizeof(uint32_t);
    staging_start = (staging_start + 7) & ~7u;
    uint32_t smem = staging_start + max_nrecs * max_decomp_len + 8;
    return (smem + 7) & ~7u;
}

// ============================================================
// decompress_string_with_fsst: 2-pass coalesced writeback
// ============================================================

__global__ void decompress_string_with_fsst(
    const char*                    d_pages,
    const FsstCompBlockKernelMeta* d_cb_meta,
    uint32_t                       total_comp_blocks,
    char*                          d_output,
    uint32_t*                      d_decomp_lens,
    uint32_t                       max_decomp_len)
{
    if (blockIdx.x >= total_comp_blocks) return;

    const FsstCompBlockKernelMeta meta = d_cb_meta[blockIdx.x];
    const char* page_base = d_pages + meta.page_byte_offset;
    const char* symtab_base = page_base + meta.symtab_byte_offset;
    const char* cb_base = page_base + meta.comp_block_byte_offset;

    const uint32_t tid = threadIdx.x;
    const uint32_t nrecs = meta.nrecs;
    const uint32_t cb_data_size = meta.comp_block_data_size;
    const uint32_t rec_base = meta.rec_base;

    extern __shared__ char smem[];

    uint8_t*  s_sym_len  = (uint8_t*)  smem;
    uint64_t* s_sym_val  = (uint64_t*)(smem + 256);
    uint8_t*  s_cb_data  = (uint8_t*)(smem + FSST_SMEM_SYMTAB_OFFSET);

    uint32_t offsets_start = FSST_SMEM_SYMTAB_OFFSET + ((cb_data_size + 3) & ~3u);
    uint32_t* s_offsets = (uint32_t*)(smem + offsets_start);

    uint32_t staging_start = offsets_start + (nrecs + 1) * sizeof(uint32_t);
    staging_start = (staging_start + 7) & ~7u;
    char* s_staging = smem + staging_start;

    // Phase 0: Cooperative coalesced load
    for (uint32_t i = tid; i < 256; i += blockDim.x)
        s_sym_len[i] = ((const uint8_t*)symtab_base)[i];
    for (uint32_t i = tid; i < 255; i += blockDim.x)
        memcpy(&s_sym_val[i], symtab_base + FSST_SYMTAB_LEN_BYTES + i * 8, 8);
    {
        uint32_t num_words = (cb_data_size + 3) / 4;
        const uint32_t* src = (const uint32_t*)cb_base;
        uint32_t* dst = (uint32_t*)s_cb_data;
        for (uint32_t i = tid; i < num_words; i += blockDim.x)
            dst[i] = src[i];
    }
    __syncthreads();

    const uint16_t* offset_table = (const uint16_t*)s_cb_data;
    const uint8_t* comp_data = s_cb_data + (nrecs + 1) * sizeof(uint16_t);

    // Phase 1: Compute decompressed length per record
    for (uint32_t r = tid; r < nrecs; r += blockDim.x) {
        uint16_t comp_start = offset_table[r];
        uint16_t comp_len = offset_table[r + 1] - comp_start;
        const uint8_t* comp_ptr = comp_data + comp_start;

        uint32_t posIn = 0, posOut = 0;
        while (posIn < comp_len) {
            uint8_t code = comp_ptr[posIn++];
            if (code < 255) {
                posOut += s_sym_len[code];
            } else {
                posIn++;
                posOut++;
            }
        }
        s_offsets[r] = posOut;
        d_decomp_lens[rec_base + r] = posOut;
    }
    __syncthreads();

    // Phase 1.5: Prefix sum (thread 0)
    if (tid == 0) {
        uint32_t sum = 0;
        for (uint32_t r = 0; r < nrecs; r++) {
            uint32_t len = s_offsets[r];
            s_offsets[r] = sum;
            sum += len;
        }
        s_offsets[nrecs] = sum;
    }
    __syncthreads();

    // Phase 2: Decompress to staging buffer
    for (uint32_t r = tid; r < nrecs; r += blockDim.x) {
        uint16_t comp_start = offset_table[r];
        uint16_t comp_len = offset_table[r + 1] - comp_start;
        const uint8_t* comp_ptr = comp_data + comp_start;

        uint32_t stg_off = s_offsets[r];
        uint32_t posIn = 0, posOut = 0;

        while (posIn < comp_len) {
            uint8_t code = comp_ptr[posIn++];
            uint64_t sym_val;
            uint8_t sym_len;

            if (code < 255) {
                sym_len = s_sym_len[code];
                sym_val = s_sym_val[code];
            } else {
                sym_val = (uint64_t)comp_ptr[posIn++];
                sym_len = 1;
            }

            memcpy(s_staging + stg_off + posOut, &sym_val, 8);
            posOut += sym_len;
        }
    }
    __syncthreads();

    // Phase 3: Coalesced writeback
    {
        uint32_t total_decoded = s_offsets[nrecs];
        char* out_base = d_output + (uint64_t)rec_base * max_decomp_len;
        uint32_t num_words = (total_decoded + 3) / 4;
        const uint32_t* src = (const uint32_t*)s_staging;
        uint32_t* dst = (uint32_t*)out_base;
        for (uint32_t i = tid; i < num_words; i += blockDim.x)
            dst[i] = src[i];
    }
}

// ============================================================
// Fused FSST decomp + KMP scan (V2)
// ============================================================

__global__ void decompress_scan_string_with_fsst(
    const char*                    d_pages,
    const FsstCompBlockKernelMeta* d_cb_meta,
    uint32_t                       total_comp_blocks,
    const char*                    d_patterns,
    const int*                     d_next,
    const int*                     d_pattern_offsets,
    const int*                     d_pattern_lengths,
    int                            num_patterns,
    uint64_t*                      d_count)
{
    if (blockIdx.x >= total_comp_blocks) return;

    const FsstCompBlockKernelMeta meta = d_cb_meta[blockIdx.x];
    const char* page_base = d_pages + meta.page_byte_offset;
    const char* symtab_base = page_base + meta.symtab_byte_offset;
    const char* cb_base = page_base + meta.comp_block_byte_offset;

    const uint32_t tid = threadIdx.x;
    const uint32_t nrecs = meta.nrecs;
    const uint32_t cb_data_size = meta.comp_block_data_size;

    extern __shared__ char smem[];

    uint8_t*  s_sym_len    = (uint8_t*)  smem;
    uint64_t* s_sym_val    = (uint64_t*)(smem + 256);
    char*     s_kmp_pat    = smem + 2296;
    int*      s_kmp_next   = (int*)(smem + 2328);
    int*      s_kmp_offsets= (int*)(smem + 2456);
    int*      s_kmp_lengths= (int*)(smem + 2472);
    uint8_t*  s_cb_data    = (uint8_t*)(smem + FSST_SMEM_SYMTAB_OFFSET);

    // Load symbol table
    for (uint32_t i = tid; i < 256; i += blockDim.x)
        s_sym_len[i] = ((const uint8_t*)symtab_base)[i];
    for (uint32_t i = tid; i < 255; i += blockDim.x)
        memcpy(&s_sym_val[i], symtab_base + FSST_SYMTAB_LEN_BYTES + i * 8, 8);

    // KMP data
    if (tid < 32) {
        s_kmp_pat[tid] = (tid < num_patterns * 16) ? d_patterns[tid] : 0;
        s_kmp_next[tid] = (tid < num_patterns * 16) ? d_next[tid] : 0;
    }
    if (tid < 4) {
        s_kmp_offsets[tid] = (tid < num_patterns) ? d_pattern_offsets[tid] : 0;
        s_kmp_lengths[tid] = (tid < num_patterns) ? d_pattern_lengths[tid] : 0;
    }

    // Load comp block data
    {
        uint32_t num_words = (cb_data_size + 3) / 4;
        const uint32_t* src = (const uint32_t*)cb_base;
        uint32_t* dst = (uint32_t*)s_cb_data;
        for (uint32_t i = tid; i < num_words; i += blockDim.x)
            dst[i] = src[i];
    }

    __syncthreads();

    const uint16_t* offset_table = (const uint16_t*)s_cb_data;
    const uint8_t* comp_data = s_cb_data + (nrecs + 1) * sizeof(uint16_t);

    uint64_t my_qualifying = 0;

    for (uint32_t r = tid; r < nrecs; r += blockDim.x) {
        uint16_t comp_start = offset_table[r];
        uint16_t comp_len = offset_table[r + 1] - comp_start;
        const uint8_t* comp_ptr = comp_data + comp_start;

        int current_pat = 0;
        int l = 0;
        int p_offset = s_kmp_offsets[0];
        int p_len = s_kmp_lengths[0];
        bool done = false;

        uint32_t posIn = 0;
        while (posIn < comp_len && !done) {
            uint8_t code = comp_ptr[posIn++];
            uint64_t sym_val;
            uint8_t sym_len;

            if (code < 255) {
                sym_len = s_sym_len[code];
                sym_val = s_sym_val[code];
            } else {
                sym_val = (uint64_t)comp_ptr[posIn++];
                sym_len = 1;
            }

            for (uint8_t j = 0; j < sym_len; j++) {
                char c = (char)(sym_val & 0xFF);
                sym_val >>= 8;

                while (l > 0 && s_kmp_pat[p_offset + l] != c)
                    l = s_kmp_next[p_offset + l - 1];
                if (s_kmp_pat[p_offset + l] == c) l++;
                if (l == p_len) {
                    current_pat++;
                    l = 0;
                    if (current_pat >= num_patterns) { done = true; break; }
                    p_offset = s_kmp_offsets[current_pat];
                    p_len = s_kmp_lengths[current_pat];
                }
            }
        }

        if (!done) my_qualifying++;
    }

    // Block-level reduction (1 atomicAdd per block instead of per thread)
    typedef cub::BlockReduce<uint64_t, FSST_BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduce_temp;
    uint64_t block_total = BlockReduce(reduce_temp).Sum(my_qualifying);
    if (tid == 0 && block_total > 0) {
        atomicAdd((unsigned long long*)d_count,
                  (unsigned long long)block_total);
    }
}
