// ============================================================
// Q16 Fused IO + Decomp + Filter kernels v2 (nvCOMPdx LZ4 PAR-32K).
//
// Three persistent kernels for Q16 VCHAR columns:
//   1. S_COMMENT: IO + decomp + KMP scan → excluded suppkeys
//   2. P_BRAND:   IO + decomp + CHAR(10) brand_id extraction
//   3. P_TYPE:    IO + decomp + VCHAR type_id dictionary extraction
//
// v2: Block-per-page with cooperative decomp + double-buffered IO.
//   128 threads/block (4 warps), all warps cooperate on decompress.
//   2 page_cache slots per block for IO/filter overlap.
//   Block-stride loop over all pages, __syncthreads().
//
// BaM IO is called via bam_io_device.cuh (compiled C++11, device-linked).
// Compiled as CUDA C++17 for nvCOMPdx.
// ============================================================

#include "bam_q16_kernel.cuh"
#include "bam_io_device.cuh"

#include <nvcompdx.hpp>

#include "tpch/page_size_dispatch.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BAM_Q16_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ── VCHAR page access helpers ──

__device__ __forceinline__ uint32_t q16f_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}

__device__ __forceinline__ uint32_t q16f_pag_get_oslt(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}

__device__ __forceinline__ uint16_t q16f_pagcol_vchar_len(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = q16f_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}

__device__ __forceinline__ const char *q16f_pagcol_vchar_data(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = q16f_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);  // skip len_u16 + pad_u16
}

// Read inline rowid from a non-pivoted VCHAR record.
// Record layout: [uint16_t vlen][uint16_t pad][char[vlen_aligned]][uint64_t rowid]
// The rowid is only 4-byte aligned, so read as two uint32_t.
__device__ __forceinline__ uint64_t q16f_pagcol_vchar_rowid(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = q16f_pag_get_oslt(page, slotid, page_size);
    uint16_t vlen = *reinterpret_cast<const uint16_t *>(page + oslt);
    uint32_t vlen_aligned = (vlen + 3u) & ~3u;
    const char* rowid_ptr = page + oslt + 2 * sizeof(uint16_t) + vlen_aligned;
    uint32_t lo = *reinterpret_cast<const uint32_t *>(rowid_ptr);
    uint32_t hi = *reinterpret_cast<const uint32_t *>(rowid_ptr + 4);
    return ((uint64_t)hi << 32) | lo;
}

// Read footer-based rowid for CHAR columns (pivoted layout).
// Rowid at: page + page_size - 8*(slotid+1), 8-byte aligned.
__device__ __forceinline__ uint64_t q16f_pagcol_char_rowid(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint64_t *>(
        page + page_size - sizeof(uint64_t) * (slotid + 1));
}

// ── KMP multi-pattern matching ──

__device__ bool q16f_kmp_match(
    const char* __restrict__ str,
    int str_len,
    const char* __restrict__ patterns,
    const int*  __restrict__ next,
    const int*  __restrict__ pattern_offsets,
    const int*  __restrict__ pattern_lengths,
    int num_patterns)
{
    int current_pat = 0;
    int l = 0;

    int p_offset = pattern_offsets[current_pat];
    int p_len    = pattern_lengths[current_pat];

    for (int i = 0; i < str_len; i++) {
        char c = str[i];
        while (l > 0 && patterns[p_offset + l] != c) {
            l = next[p_offset + l - 1];
        }
        if (patterns[p_offset + l] == c) l++;
        if (l == p_len) {
            current_pat++;
            l = 0;
            if (current_pat >= num_patterns) return true;
            p_offset = pattern_offsets[current_pat];
            p_len    = pattern_lengths[current_pat];
        }
    }
    return false;
}

// ── FNV-1a 64-bit hash ──

__device__ __forceinline__ uint64_t q16f_fnv1a64(const char *s, uint16_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (uint16_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ============================================================
// Constants
// ============================================================

static constexpr uint32_t Q16F_CHUNK_SZ = 32768u;

// NVMe PRP boundary workaround (same as bam_kernel.cu safe_io_nblocks_vchar_io)
#define Q16F_NVM_CTRL_PAGE_BLOCKS 8

__device__ __forceinline__ uint32_t q16f_safe_io_nblocks(uint32_t comp_bytes) {
    uint32_t nblk = (comp_bytes + 511) / 512;
    if (nblk > Q16F_NVM_CTRL_PAGE_BLOCKS && nblk <= Q16F_NVM_CTRL_PAGE_BLOCKS * 2)
        nblk = Q16F_NVM_CTRL_PAGE_BLOCKS * 2 + 1;
    return nblk;
}
static constexpr uint32_t Q16F_TYPE_DICT_CAP  = 512;
static constexpr uint32_t Q16F_TYPE_DICT_MASK = Q16F_TYPE_DICT_CAP - 1;
static constexpr uint32_t Q16F_TYPE_MAX_LEN   = 32;

// ============================================================
// Kernel 1: S_COMMENT — Fused IO + Decomp + KMP Scan
//
// Block-per-page: 128 threads cooperatively decompress.
// Double-buffered IO: 2 page_cache slots per block.
// ============================================================

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_q16_fused_s_comment_par32k_kernel(
    void*           ctrls_opaque,
    void*           pc_opaque,
    const char*     pc_base_addr,
    char*           d_decomp_buf,
    BAMq16FusedSCommentParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<Q16F_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t bid     = blockIdx.x;

    extern __shared__ __align__(8) uint8_t smem_q16sc[];
    constexpr size_t warp_smem_size = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem_q16sc + warp_id * warp_smem_size;

    constexpr uint32_t n_chunks       = PAGE_SIZE_CONST / Q16F_CHUNK_SZ;
    constexpr uint32_t hdr_bytes      = n_chunks * sizeof(uint32_t);
    constexpr uint32_t chunks_per_warp = n_chunks / 4;

    // Double-buffer: 2 page_cache slots per block
    const uint32_t slot_a = bid * 2;
    const uint32_t slot_b = bid * 2 + 1;
    uint32_t cur_slot = slot_a;
    uint32_t nxt_slot = slot_b;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;
    const bool is_compressed = (p.comp_method != 0);
    constexpr uint32_t blocks_per_page = PAGE_SIZE_CONST / 512;

    // Helper: compute multi-device LBA + nblocks for page pg
    auto compute_io = [&](uint64_t pg_idx, uint32_t& out_dev, uint32_t& out_nblk) -> uint64_t {
        uint64_t global_pg = p.field_start_page_id + pg_idx;
        out_dev = global_pg % ndev;
        if (is_compressed) {
            out_nblk = q16f_safe_io_nblocks(p.d_comp_sizes[pg_idx]);
            return p.partition_start_lbas[out_dev] + p.d_comp_offsets[pg_idx] / 512;
        } else {
            uint64_t local_pg = global_pg / ndev;
            out_nblk = blocks_per_page;
            return p.partition_start_lbas[out_dev] + local_pg * blocks_per_page;
        }
    };

    // Prefetch first page
    uint64_t pg = (uint64_t)bid;
    if (pg < p.npages) {
        if (tid == 0) {
            uint32_t dev, nblk;
            uint64_t lba = compute_io(pg, dev, nblk);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, cur_slot, dev);
        }
        __syncthreads();
    }

    for (pg = (uint64_t)bid; pg < p.npages; pg += gridDim.x) {
        uint64_t next_pg = pg + gridDim.x;
        bool has_next = (next_pg < p.npages);

        long long t0 = clock64();

        // ── Phase 2: Decompress (or direct read for uncompressed) ──
        const char* page_ptr;
        if (is_compressed) {
            const uint8_t* comp_page = (const uint8_t*)(
                pc_base_addr + (unsigned long long)cur_slot * p.page_size);
            uint8_t* decomp_page = (uint8_t*)(
                d_decomp_buf + (unsigned long long)cur_slot * p.page_size);

            const uint32_t* chunk_hdr = (const uint32_t*)comp_page;
            auto decompressor = lz4_decomp_t();

            uint32_t first_chunk = warp_id * chunks_per_warp;
            uint32_t data_offset = hdr_bytes;
            for (uint32_t i = 0; i < first_chunk; i++)
                data_offset += chunk_hdr[i];

            for (uint32_t i = 0; i < chunks_per_warp; i++) {
                uint32_t chunk_idx = first_chunk + i;
                uint32_t comp_len = chunk_hdr[chunk_idx];
                size_t decomp_size = 0;
                decompressor.execute(
                    comp_page + data_offset,
                    decomp_page + (uint64_t)chunk_idx * Q16F_CHUNK_SZ,
                    comp_len, &decomp_size, my_smem, nullptr);
                data_offset += comp_len;
            }
            page_ptr = (const char*)decomp_page;
        } else {
            page_ptr = pc_base_addr + (unsigned long long)cur_slot * p.page_size;
        }
        __syncthreads();

        long long t1 = clock64();

        // ── Phase 1: Prefetch next page (tid==0, overlaps with filter) ──
        if (has_next && tid == 0) {
            uint32_t dev, nblk;
            uint64_t lba = compute_io(next_pg, dev, nblk);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, nxt_slot, dev);
        }

        // ── Phase 3: KMP scan + excluded suppkey extraction (128 threads) ──
        {
            const char* page = page_ptr;
            uint32_t nalloc = q16f_pag_get_nalloc(page);

            for (uint32_t s = tid; s < nalloc; s += 128) {
                uint64_t row_id = q16f_pagcol_vchar_rowid(page, s, p.page_size);

                uint16_t vlen = q16f_pagcol_vchar_len(page, s, p.page_size);
                const char* vdata = q16f_pagcol_vchar_data(page, s, p.page_size);

                if (q16f_kmp_match(vdata, (int)vlen,
                        p.d_patterns, p.d_next,
                        p.d_pattern_offsets, p.d_pattern_lengths,
                        p.num_patterns)) {
                    uint32_t pos = atomicAdd(p.d_excl_count, 1);
                    p.d_excl_suppkeys[pos] = p.d_s_suppkey_flat[row_id];
                }
            }
        }
        __syncthreads();

        long long t2 = clock64();

        if (tid == 0 && p.d_phase_cycles) {
            atomicAdd((unsigned long long*)&p.d_phase_cycles[0], 0ULL);
            atomicAdd((unsigned long long*)&p.d_phase_cycles[1],
                      (unsigned long long)(t1 - t0));
            atomicAdd((unsigned long long*)&p.d_phase_cycles[2],
                      (unsigned long long)(t2 - t1));
        }

        // Swap double-buffer slots
        uint32_t tmp = cur_slot; cur_slot = nxt_slot; nxt_slot = tmp;
    }
}

// ============================================================
// Kernel 2: P_BRAND — Fused IO + Decomp + Brand ID Extraction
// ============================================================

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_q16_fused_p_brand_par32k_kernel(
    void*           ctrls_opaque,
    void*           pc_opaque,
    const char*     pc_base_addr,
    char*           d_decomp_buf,
    BAMq16FusedBrandParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<Q16F_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t bid     = blockIdx.x;

    extern __shared__ __align__(8) uint8_t smem_q16br[];
    constexpr size_t warp_smem_size = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem_q16br + warp_id * warp_smem_size;

    constexpr uint32_t n_chunks       = PAGE_SIZE_CONST / Q16F_CHUNK_SZ;
    constexpr uint32_t hdr_bytes      = n_chunks * sizeof(uint32_t);
    constexpr uint32_t chunks_per_warp = n_chunks / 4;

    const uint32_t slot_a = bid * 2;
    const uint32_t slot_b = bid * 2 + 1;
    uint32_t cur_slot = slot_a;
    uint32_t nxt_slot = slot_b;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;
    const bool is_compressed = (p.comp_method != 0);
    constexpr uint32_t blocks_per_page_brand = PAGE_SIZE_CONST / 512;

    auto compute_io = [&](uint64_t pg_idx, uint32_t& out_dev, uint32_t& out_nblk) -> uint64_t {
        uint64_t global_pg = p.field_start_page_id + pg_idx;
        out_dev = global_pg % ndev;
        if (is_compressed) {
            out_nblk = q16f_safe_io_nblocks(p.d_comp_sizes[pg_idx]);
            return p.partition_start_lbas[out_dev] + p.d_comp_offsets[pg_idx] / 512;
        } else {
            uint64_t local_pg = global_pg / ndev;
            out_nblk = blocks_per_page_brand;
            return p.partition_start_lbas[out_dev] + local_pg * blocks_per_page_brand;
        }
    };

    // Prefetch first page
    uint64_t pg = (uint64_t)bid;
    if (pg < p.npages) {
        if (tid == 0) {
            uint32_t dev, nblk;
            uint64_t lba = compute_io(pg, dev, nblk);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, cur_slot, dev);
        }
        __syncthreads();
    }

    for (pg = (uint64_t)bid; pg < p.npages; pg += gridDim.x) {
        uint64_t next_pg = pg + gridDim.x;
        bool has_next = (next_pg < p.npages);

        long long t0 = clock64();

        // ── Phase 2: Decompress (or direct read for uncompressed) ──
        const char* page_ptr;
        if (is_compressed) {
            const uint8_t* comp_page = (const uint8_t*)(
                pc_base_addr + (unsigned long long)cur_slot * p.page_size);
            uint8_t* decomp_page = (uint8_t*)(
                d_decomp_buf + (unsigned long long)cur_slot * p.page_size);

            const uint32_t* chunk_hdr = (const uint32_t*)comp_page;
            auto decompressor = lz4_decomp_t();

            uint32_t first_chunk = warp_id * chunks_per_warp;
            uint32_t data_offset = hdr_bytes;
            for (uint32_t i = 0; i < first_chunk; i++)
                data_offset += chunk_hdr[i];

            for (uint32_t i = 0; i < chunks_per_warp; i++) {
                uint32_t chunk_idx = first_chunk + i;
                uint32_t comp_len = chunk_hdr[chunk_idx];
                size_t decomp_size = 0;
                decompressor.execute(
                    comp_page + data_offset,
                    decomp_page + (uint64_t)chunk_idx * Q16F_CHUNK_SZ,
                    comp_len, &decomp_size, my_smem, nullptr);
                data_offset += comp_len;
            }
            page_ptr = (const char*)decomp_page;
        } else {
            page_ptr = pc_base_addr + (unsigned long long)cur_slot * p.page_size;
        }
        __syncthreads();

        long long t1 = clock64();

        // ── Phase 1: Prefetch next page ──
        if (has_next && tid == 0) {
            uint32_t dev, nblk;
            uint64_t lba = compute_io(next_pg, dev, nblk);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, nxt_slot, dev);
        }

        // ── Phase 3: CHAR(10) brand_id extraction (128 threads) ──
        // CHAR columns are ordered by rowid, so row_base + s is correct.
        {
            const char* page = page_ptr;
            uint32_t nalloc = q16f_pag_get_nalloc(page);
            uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

            for (uint32_t s = tid; s < nalloc; s += 128) {
                uint64_t row_id = row_base + s;
                const char* brand = page + 12 + p.padded_len * s;
                uint32_t d1 = brand[6] - '1';
                uint32_t d2 = brand[7] - '1';
                p.d_brand_ids[row_id] = d1 * 5 + d2;
            }
        }
        __syncthreads();

        long long t2 = clock64();

        if (tid == 0 && p.d_phase_cycles) {
            atomicAdd((unsigned long long*)&p.d_phase_cycles[0], 0ULL);
            atomicAdd((unsigned long long*)&p.d_phase_cycles[1],
                      (unsigned long long)(t1 - t0));
            atomicAdd((unsigned long long*)&p.d_phase_cycles[2],
                      (unsigned long long)(t2 - t1));
        }

        uint32_t tmp = cur_slot; cur_slot = nxt_slot; nxt_slot = tmp;
    }
}

// ============================================================
// Kernel 3: P_TYPE — Fused IO + Decomp + Type ID Dictionary Extraction
// ============================================================

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_q16_fused_p_type_par32k_kernel(
    void*           ctrls_opaque,
    void*           pc_opaque,
    const char*     pc_base_addr,
    char*           d_decomp_buf,
    BAMq16FusedTypeParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<Q16F_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t bid     = blockIdx.x;

    extern __shared__ __align__(8) uint8_t smem_q16ty[];
    constexpr size_t warp_smem_size = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem_q16ty + warp_id * warp_smem_size;

    constexpr uint32_t n_chunks       = PAGE_SIZE_CONST / Q16F_CHUNK_SZ;
    constexpr uint32_t hdr_bytes      = n_chunks * sizeof(uint32_t);
    constexpr uint32_t chunks_per_warp = n_chunks / 4;

    const uint32_t slot_a = bid * 2;
    const uint32_t slot_b = bid * 2 + 1;
    uint32_t cur_slot = slot_a;
    uint32_t nxt_slot = slot_b;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;
    const bool is_compressed = (p.comp_method != 0);
    constexpr uint32_t blocks_per_page_type = PAGE_SIZE_CONST / 512;

    auto compute_io = [&](uint64_t pg_idx, uint32_t& out_dev, uint32_t& out_nblk) -> uint64_t {
        uint64_t global_pg = p.field_start_page_id + pg_idx;
        out_dev = global_pg % ndev;
        if (is_compressed) {
            out_nblk = q16f_safe_io_nblocks(p.d_comp_sizes[pg_idx]);
            return p.partition_start_lbas[out_dev] + p.d_comp_offsets[pg_idx] / 512;
        } else {
            uint64_t local_pg = global_pg / ndev;
            out_nblk = blocks_per_page_type;
            return p.partition_start_lbas[out_dev] + local_pg * blocks_per_page_type;
        }
    };

    // Prefetch first page
    uint64_t pg = (uint64_t)bid;
    if (pg < p.npages) {
        if (tid == 0) {
            uint32_t dev, nblk;
            uint64_t lba = compute_io(pg, dev, nblk);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, cur_slot, dev);
        }
        __syncthreads();
    }

    for (pg = (uint64_t)bid; pg < p.npages; pg += gridDim.x) {
        uint64_t next_pg = pg + gridDim.x;
        bool has_next = (next_pg < p.npages);

        long long t0 = clock64();

        // ── Phase 2: Decompress (or direct read for uncompressed) ──
        const char* page_ptr;
        if (is_compressed) {
            const uint8_t* comp_page = (const uint8_t*)(
                pc_base_addr + (unsigned long long)cur_slot * p.page_size);
            uint8_t* decomp_page = (uint8_t*)(
                d_decomp_buf + (unsigned long long)cur_slot * p.page_size);

            const uint32_t* chunk_hdr = (const uint32_t*)comp_page;
            auto decompressor = lz4_decomp_t();

            uint32_t first_chunk = warp_id * chunks_per_warp;
            uint32_t data_offset = hdr_bytes;
            for (uint32_t i = 0; i < first_chunk; i++)
                data_offset += chunk_hdr[i];

            for (uint32_t i = 0; i < chunks_per_warp; i++) {
                uint32_t chunk_idx = first_chunk + i;
                uint32_t comp_len = chunk_hdr[chunk_idx];
                size_t decomp_size = 0;
                decompressor.execute(
                    comp_page + data_offset,
                    decomp_page + (uint64_t)chunk_idx * Q16F_CHUNK_SZ,
                    comp_len, &decomp_size, my_smem, nullptr);
                data_offset += comp_len;
            }
            page_ptr = (const char*)decomp_page;
        } else {
            page_ptr = pc_base_addr + (unsigned long long)cur_slot * p.page_size;
        }
        __syncthreads();

        long long t1 = clock64();

        // ── Phase 1: Prefetch next page ──
        if (has_next && tid == 0) {
            uint32_t dev, nblk;
            uint64_t lba = compute_io(next_pg, dev, nblk);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, nxt_slot, dev);
        }

        // ── Phase 3: VCHAR type_id extraction with dictionary (128 threads) ──
        {
            const char* page = page_ptr;
            uint32_t nalloc = q16f_pag_get_nalloc(page);

            for (uint32_t s = tid; s < nalloc; s += 128) {
                uint64_t row_id = q16f_pagcol_vchar_rowid(page, s, p.page_size);

                uint16_t vlen = q16f_pagcol_vchar_len(page, s, p.page_size);
                const char* vdata = q16f_pagcol_vchar_data(page, s, p.page_size);

                // Check NOT LIKE 'MEDIUM POLISHED%'
                bool is_medium_polished = false;
                if (vlen >= 15) {
                    const char mp[] = "MEDIUM POLISHED";
                    is_medium_polished = true;
                    for (int k = 0; k < 15; k++) {
                        if (vdata[k] != mp[k]) {
                            is_medium_polished = false;
                            break;
                        }
                    }
                }
                if (is_medium_polished) {
                    p.d_type_ids[row_id] = UINT32_MAX;
                    continue;
                }

                // FNV-1a hash → dictionary probe/insert
                uint64_t h = q16f_fnv1a64(vdata, vlen);
                uint32_t dict_slot = (uint32_t)h & Q16F_TYPE_DICT_MASK;

                while (true) {
                    uint64_t prev = atomicCAS(
                        reinterpret_cast<unsigned long long*>(&p.d_dict_keys[dict_slot]),
                        (unsigned long long)UINT64_MAX,
                        (unsigned long long)h);

                    if (prev == UINT64_MAX) {
                        uint32_t new_tid = atomicAdd(p.d_type_id_counter, 1);
                        char* dst = p.d_dict_strs + (uint64_t)dict_slot * Q16F_TYPE_MAX_LEN;
                        for (uint16_t k = 0; k < vlen; k++) dst[k] = vdata[k];
                        p.d_dict_lens[dict_slot] = vlen;
                        __threadfence();
                        p.d_dict_type_ids[dict_slot] = new_tid;
                        p.d_type_ids[row_id] = new_tid;
                        break;
                    }
                    if (prev == h) {
                        uint32_t existing_tid;
                        do {
                            __threadfence();
                            existing_tid = p.d_dict_type_ids[dict_slot];
                        } while (existing_tid == UINT32_MAX);
                        p.d_type_ids[row_id] = existing_tid;
                        break;
                    }
                    dict_slot = (dict_slot + 1) & Q16F_TYPE_DICT_MASK;
                }
            }
        }
        __syncthreads();

        long long t2 = clock64();

        if (tid == 0 && p.d_phase_cycles) {
            atomicAdd((unsigned long long*)&p.d_phase_cycles[0], 0ULL);
            atomicAdd((unsigned long long*)&p.d_phase_cycles[1],
                      (unsigned long long)(t1 - t0));
            atomicAdd((unsigned long long*)&p.d_phase_cycles[2],
                      (unsigned long long)(t2 - t1));
        }

        uint32_t tmp = cur_slot; cur_slot = nxt_slot; nxt_slot = tmp;
    }
}

// ============================================================
// Fused IO Context
// ============================================================

struct BAMq16FusedIOContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_q16_fused_io_ctx_t bam_q16_fused_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMq16FusedIOContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 2 slots per block (double-buffered IO)
    const uint32_t num_slots = num_blocks * 2;

    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    size_t decomp_size = (size_t)num_slots * page_size;
    BAM_Q16_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_q16_fused_io_ctx_t>(ctx);
}

void bam_q16_fused_io_destroy(bam_q16_fused_io_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMq16FusedIOContext*>(ctx_handle);
    if (!ctx) return;

    cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ============================================================
// Launch helpers
// ============================================================

using decomp_type_t = decltype(
    nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
    nvcompdx::DataType<nvcompdx::datatype::uint8>() +
    nvcompdx::Direction<nvcompdx::direction::decompress>() +
    nvcompdx::MaxUncompChunkSize<Q16F_CHUNK_SZ>() +
    nvcompdx::Warp() +
    nvcompdx::SM<800>());

static size_t q16f_smem_size() {
    return decomp_type_t().shmem_size_group() * 4;  // 4 warps
}

// ── S_COMMENT launch ──

void bam_q16_fused_s_comment_async(
    bam_q16_fused_io_ctx_t ctx_handle,
    const BAMq16FusedSCommentParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq16FusedIOContext*>(ctx_handle);
    size_t smem = q16f_smem_size();
    constexpr uint32_t TPB = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kfn = bam_q16_fused_s_comment_par32k_kernel<PS>;
        BAM_Q16_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<p.num_blocks, TPB, smem, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });
    BAM_Q16_CUDA_CHECK(cudaGetLastError());
}

// ── P_BRAND launch ──

void bam_q16_fused_p_brand_async(
    bam_q16_fused_io_ctx_t ctx_handle,
    const BAMq16FusedBrandParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq16FusedIOContext*>(ctx_handle);
    size_t smem = q16f_smem_size();
    constexpr uint32_t TPB = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kfn = bam_q16_fused_p_brand_par32k_kernel<PS>;
        BAM_Q16_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<p.num_blocks, TPB, smem, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });
    BAM_Q16_CUDA_CHECK(cudaGetLastError());
}

// ── P_TYPE launch ──

void bam_q16_fused_p_type_async(
    bam_q16_fused_io_ctx_t ctx_handle,
    const BAMq16FusedTypeParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq16FusedIOContext*>(ctx_handle);
    size_t smem = q16f_smem_size();
    constexpr uint32_t TPB = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kfn = bam_q16_fused_p_type_par32k_kernel<PS>;
        BAM_Q16_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<p.num_blocks, TPB, smem, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });
    BAM_Q16_CUDA_CHECK(cudaGetLastError());
}
