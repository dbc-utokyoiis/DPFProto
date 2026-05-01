// ============================================================
// Q13 decompress + KMP scan kernel (nvCOMPdx LZ4 PAR-32K).
//
// Based on bam_vchar_kernel.cu v8 (PAR-32K, 1 page/block, 4 warps).
// Phase 1: PAR-32K LZ4 decompress (identical to v8).
// Phase 2: KMP multi-pattern matching on O_COMMENT VCHAR records,
//          writing qualifying O_CUSTKEY to d_o_aggr_custkey.
//
// Also contains: fused IO+decomp+scan kernel (GPU-initiated NVMe I/O
// via BaM page_cache + PAR-32K nvCOMPdx + KMP scan in one kernel).
// BaM IO is called via bam_io_device.cuh (compiled C++11, device-linked).
//
// Compiled as CUDA C++17 for nvCOMPdx.
// ============================================================

#include "bam_q13_kernel.cuh"
#include "bam_io_device.cuh"

#include <nvcompdx.hpp>

#include "tpch/page_size_dispatch.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BAM_Q13_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// PRP2 workaround (same as bam_kernel.cu / bam_q3_kernel.cu)
#define Q13F_NVM_CTRL_PAGE_BLOCKS 8
__device__ __forceinline__ uint32_t q13f_fix_nblk(uint32_t nblk) {
    if (nblk > Q13F_NVM_CTRL_PAGE_BLOCKS && nblk <= Q13F_NVM_CTRL_PAGE_BLOCKS * 2)
        nblk = Q13F_NVM_CTRL_PAGE_BLOCKS * 2 + 1;
    return nblk;
}

// ── VCHAR page access helpers ──

__device__ __forceinline__ uint32_t q13_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}

__device__ __forceinline__ uint32_t q13_pag_get_oslt(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}

__device__ __forceinline__ uint16_t q13_pagcol_vchar_len(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = q13_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}

__device__ __forceinline__ const char *q13_pagcol_vchar_data(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = q13_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);  // skip len_u16 + pad_u16
}

// ── KMP multi-pattern matching ──
// Matches "special" then "requests" sequentially in the string.
// Returns true if BOTH patterns are found (i.e., LIKE '%special%requests%' matches).
// Q13 filter: NOT LIKE → qualifying records are those where this returns FALSE.

__device__ bool q13_kmp_match(
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

// ============================================================
// Q13 PAR-32K decompress + KMP scan kernel
// ============================================================

static constexpr uint32_t Q13_CHUNK_SZ = 32768u;

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_q13_decomp_scan_par32k_kernel(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    const uint64_t* d_prefix_sum,
    const uint64_t* d_o_custkey_flat,
    uint64_t*       d_o_aggr_custkey,
    uint64_t*       d_count,
    const char*     d_patterns,
    const int*      d_next,
    const int*      d_pattern_offsets,
    const int*      d_pattern_lengths,
    int             num_patterns,
    int             total_pattern_chars,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint64_t        npages_total)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<Q13_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t tid     = threadIdx.x;   // 0..127
    const uint32_t warp_id = tid / 32;      // 0..3
    const uint32_t bid     = blockIdx.x;

    const uint64_t pg = batch_start + (uint64_t)bid;
    if (pg >= npages_total) return;

    // Shared memory: 4 regions for 4 warp decompressors
    extern __shared__ __align__(8) uint8_t smem_q13[];
    constexpr size_t warp_smem_size = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem_q13 + warp_id * warp_smem_size;

    const uint32_t slot = bid;  // 1 page per block

    constexpr uint32_t n_chunks       = PAGE_SIZE_CONST / Q13_CHUNK_SZ;  // 32
    constexpr uint32_t hdr_bytes      = n_chunks * sizeof(uint32_t);     // 128
    constexpr uint32_t chunks_per_warp = n_chunks / 4;                   // 8

    // ── Phase 1: PAR-32K LZ4 decompress (identical to v8) ──
    {
        const uint8_t* comp_page = (const uint8_t*)(
            d_staging_buf + (uint64_t)slot * page_size);
        uint8_t* decomp_page = (uint8_t*)(
            d_decomp_buf + (uint64_t)slot * page_size);

        const uint32_t* chunk_hdr = (const uint32_t*)comp_page;

        auto decompressor = lz4_decomp_t();

        uint32_t first_chunk = warp_id * chunks_per_warp;

        // Compute starting byte offset for this warp's first chunk
        uint32_t data_offset = hdr_bytes;
        for (uint32_t i = 0; i < first_chunk; i++) {
            data_offset += chunk_hdr[i];
        }

        for (uint32_t i = 0; i < chunks_per_warp; i++) {
            uint32_t chunk_idx = first_chunk + i;
            uint32_t comp_len = chunk_hdr[chunk_idx];
            size_t decomp_size = 0;

            decompressor.execute(
                comp_page + data_offset,
                decomp_page + (uint64_t)chunk_idx * Q13_CHUNK_SZ,
                comp_len,
                &decomp_size,
                my_smem,
                nullptr);

            data_offset += comp_len;
        }
    }
    __syncthreads();

    // ── Phase 2: KMP scan + custkey extraction (all 128 threads) ──
    {
        const char* page = d_decomp_buf + (uint64_t)slot * page_size;
        uint32_t nalloc = q13_pag_get_nalloc(page);

        // Row ID base for this page
        uint64_t row_base = (pg == 0) ? 0 : d_prefix_sum[pg - 1];

        uint64_t my_qualifying = 0;

        for (uint32_t s = tid; s < nalloc; s += blockDim.x) {
            uint64_t row_id = row_base + s;

            uint16_t vlen = q13_pagcol_vchar_len(page, s, page_size);
            const char* vdata = q13_pagcol_vchar_data(page, s, page_size);

            bool matched = q13_kmp_match(
                vdata, (int)vlen,
                d_patterns, d_next,
                d_pattern_offsets, d_pattern_lengths,
                num_patterns);

            // NOT LIKE: qualifying = !matched
            if (matched) {
                d_o_aggr_custkey[row_id] = UINT64_MAX;
            } else {
                d_o_aggr_custkey[row_id] = d_o_custkey_flat[row_id];
                my_qualifying++;
            }
        }

        // Count qualifying records via warp reduction + atomicAdd
        // Simple atomicAdd per-thread is fine for this workload
        if (my_qualifying > 0) {
            atomicAdd((unsigned long long*)d_count,
                      (unsigned long long)my_qualifying);
        }
    }
}

// ============================================================
// Host wrapper
// ============================================================

void bam_q13_decomp_scan_par32k_async(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    const uint64_t* d_prefix_sum,
    const uint64_t* d_o_custkey_flat,
    uint64_t*       d_o_aggr_custkey,
    uint64_t*       d_count,
    const char*     d_patterns,
    const int*      d_next,
    const int*      d_pattern_offsets,
    const int*      d_pattern_lengths,
    int             num_patterns,
    int             total_pattern_chars,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_pages,
    uint64_t        npages_total,
    cudaStream_t    stream)
{
    using decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<Q13_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    size_t smem_size = decomp_t().shmem_size_group() * 4;  // 4 warps
    const uint32_t threads_per_block = 128;

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kernel_fn = bam_q13_decomp_scan_par32k_kernel<PS>;
        BAM_Q13_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<batch_pages, threads_per_block, smem_size, stream>>>(
            d_staging_buf, d_comp_sizes, d_decomp_buf,
            d_prefix_sum, d_o_custkey_flat, d_o_aggr_custkey, d_count,
            d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
            num_patterns, total_pattern_chars,
            page_size, batch_start, npages_total);
    });
}

// ============================================================
// Fused IO + Decomp + KMP Scan kernel (GPU-initiated NVMe I/O)
//
// Single persistent kernel: warp-stride loop over all pages.
// Each warp independently handles its own page stream:
//   warp_global_id = blockIdx.x * 4 + warp_id
//   pages: warp_global_id, warp_global_id + total_warps, ...
//
// Phase 1: lane 0 reads compressed page from NVMe via BaM.
// Phase 2: 1 warp decompresses all 32 PAR-32K LZ4 chunks.
// Phase 3: 32 lanes run KMP scan + custkey extraction.
//
// No __syncthreads() — warps are fully independent, enabling
// natural pipelining within each block (warp 0 may be doing IO
// while warp 1 is decompressing and warp 2 is scanning).
// ============================================================

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_q13_fused_io_decomp_scan_par32k_kernel(
    void*           ctrls_opaque,       // Controller** (opaque, for BaM IO)
    void*           pc_opaque,          // page_cache_d_t* (opaque, for BaM IO)
    const char*     pc_base_addr,       // page_cache base address (data access)
    char*           d_decomp_buf,
    BAMq13FusedParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<Q13_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t tid     = threadIdx.x;   // 0..127
    const uint32_t warp_id = tid / 32;      // 0..3
    const uint32_t lane_id = tid % 32;      // 0..31
    const uint32_t bid     = blockIdx.x;

    // Each warp gets its own shared memory region for nvCOMPdx
    extern __shared__ __align__(8) uint8_t smem_fused[];
    constexpr size_t warp_smem_size = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem_fused + warp_id * warp_smem_size;

    constexpr uint32_t n_chunks  = PAGE_SIZE_CONST / Q13_CHUNK_SZ;  // 32
    constexpr uint32_t hdr_bytes = n_chunks * sizeof(uint32_t);     // 128

    // Each warp owns a dedicated page_cache slot and decomp buffer region
    const uint32_t slot = bid * 4 + warp_id;
    const uint32_t total_warps = gridDim.x * 4;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;
    const bool is_compressed = (p.comp_method != 0);
    constexpr uint32_t blocks_per_page = PAGE_SIZE_CONST / 512;

    // Helper: compute multi-device LBA for page pg
    auto compute_lba_dev = [&](uint64_t pg_idx, uint32_t& out_dev, uint32_t& out_nblk) -> uint64_t {
        uint64_t global_pg = p.field_start_page_id + pg_idx;
        out_dev = global_pg % ndev;
        if (is_compressed) {
            out_nblk = q13f_fix_nblk((p.d_comp_sizes[pg_idx] + 511) / 512);
            return p.partition_start_lbas[out_dev] + p.d_comp_offsets[pg_idx] / 512;
        } else {
            uint64_t local_pg = global_pg / ndev;
            out_nblk = blocks_per_page;
            return p.partition_start_lbas[out_dev] + local_pg * blocks_per_page;
        }
    };

    // Warp-stride loop: each warp processes its own page stream
    for (uint64_t pg = slot; pg < p.npages; pg += total_warps) {

        long long t0 = clock64();

        // ── Phase 1: NVMe I/O (lane 0 of each warp) ──
        if (lane_id == 0) {
            uint32_t dev, nblk;
            uint64_t lba = compute_lba_dev(pg, dev, nblk);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, slot, dev);
        }
        __syncwarp();

        long long t1 = clock64();

        // ── Phase 2: PAR-32K LZ4 decompress (1 warp, all 32 chunks) ──
        const char* page;
        if (is_compressed) {
            const uint8_t* comp_page = (const uint8_t*)(
                pc_base_addr + (unsigned long long)slot * p.page_size);
            uint8_t* decomp_page = (uint8_t*)(
                d_decomp_buf + (unsigned long long)slot * p.page_size);

            const uint32_t* chunk_hdr = (const uint32_t*)comp_page;

            auto decompressor = lz4_decomp_t();

            uint32_t data_offset = hdr_bytes;

            for (uint32_t i = 0; i < n_chunks; i++) {
                uint32_t comp_len = chunk_hdr[i];
                size_t decomp_size = 0;

                decompressor.execute(
                    comp_page + data_offset,
                    decomp_page + (uint64_t)i * Q13_CHUNK_SZ,
                    comp_len,
                    &decomp_size,
                    my_smem,
                    nullptr);

                data_offset += comp_len;
            }
            page = (const char*)decomp_page;
        } else {
            page = pc_base_addr + (unsigned long long)slot * p.page_size;
        }
        __syncwarp();

        long long t2 = clock64();

        // ── Phase 3: KMP scan + custkey extraction (32 lanes) ──
        {
            uint32_t nalloc = q13_pag_get_nalloc(page);

            uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

            uint64_t my_qualifying = 0;

            for (uint32_t s = lane_id; s < nalloc; s += 32) {
                uint64_t row_id = row_base + s;

                uint16_t vlen = q13_pagcol_vchar_len(page, s, p.page_size);
                const char* vdata = q13_pagcol_vchar_data(page, s, p.page_size);

                bool matched = q13_kmp_match(
                    vdata, (int)vlen,
                    p.d_patterns, p.d_next,
                    p.d_pattern_offsets, p.d_pattern_lengths,
                    p.num_patterns);

                if (matched) {
                    p.d_o_aggr_custkey[row_id] = UINT64_MAX;
                } else {
                    p.d_o_aggr_custkey[row_id] = p.d_o_custkey_flat[row_id];
                    my_qualifying++;
                }
            }

            if (my_qualifying > 0) {
                atomicAdd((unsigned long long*)p.d_count,
                          (unsigned long long)my_qualifying);
            }
        }
        __syncwarp();

        long long t3 = clock64();

        // Accumulate phase cycles (lane 0 of each warp)
        if (lane_id == 0 && p.d_phase_cycles) {
            atomicAdd((unsigned long long*)&p.d_phase_cycles[0],
                      (unsigned long long)(t1 - t0));
            atomicAdd((unsigned long long*)&p.d_phase_cycles[1],
                      (unsigned long long)(t2 - t1));
            atomicAdd((unsigned long long*)&p.d_phase_cycles[2],
                      (unsigned long long)(t3 - t2));
        }
    }
}

// ── Fused IO Context ──
// page_cache creation/destruction is handled by bam_io_device.cu (C++11).
// This file only stores opaque handles and the decomp buffer.

struct BAMq13FusedIOContext {
    bam_io_page_cache_t io_pc;  // opaque page_cache handle
    void*       d_ctrls;        // Controller** (device)
    void*       d_pc_ptr;       // page_cache_d_t* (device)
    const char* pc_base_addr;   // page_cache base address (device)
    char*       d_decomp_buf;   // [num_slots * page_size] (num_slots = num_blocks * 4)
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_q13_fused_io_ctx_t bam_q13_fused_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMq13FusedIOContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 4 slots per block (1 per warp): page_cache + decomp buffer
    const uint32_t num_slots = num_blocks * 4;

    // Create page cache via C++11 wrapper (4 entries per block)
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    // Extract opaque pointers for kernel launch
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: 1 page per warp slot
    size_t decomp_size = (size_t)num_slots * page_size;
    BAM_Q13_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_q13_fused_io_ctx_t>(ctx);
}

static void bam_q13_fused_launch(
    BAMq13FusedIOContext* ctx,
    const BAMq13FusedParams& p,
    cudaStream_t stream)
{
    using decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<Q13_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    size_t smem_size = decomp_t().shmem_size_group() * 4;  // 4 warps
    const uint32_t threads_per_block = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kernel_fn = bam_q13_fused_io_decomp_scan_par32k_kernel<PS>;
        BAM_Q13_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<p.num_blocks, threads_per_block, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    BAM_Q13_CUDA_CHECK(cudaGetLastError());
}

void bam_q13_fused_io_decomp_scan(
    bam_q13_fused_io_ctx_t ctx_handle,
    const BAMq13FusedParams& params)
{
    auto* ctx = static_cast<BAMq13FusedIOContext*>(ctx_handle);
    bam_q13_fused_launch(ctx, params, 0);
    BAM_Q13_CUDA_CHECK(cudaDeviceSynchronize());
}

void bam_q13_fused_io_decomp_scan_async(
    bam_q13_fused_io_ctx_t ctx_handle,
    const BAMq13FusedParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq13FusedIOContext*>(ctx_handle);
    bam_q13_fused_launch(ctx, params, stream);
}

void bam_q13_fused_io_destroy(bam_q13_fused_io_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMq13FusedIOContext*>(ctx_handle);
    if (!ctx) return;

    cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}
