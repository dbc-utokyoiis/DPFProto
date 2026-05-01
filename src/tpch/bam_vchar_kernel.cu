// ============================================================
// VCHAR decompress + scan kernel (nvCOMPdx LZ4, device-side).
//
// Compiled as CUDA C++17 for nvCOMPdx.
// No BAM headers — receives pre-staged compressed data from
// bam_vchar_io_kernel (bam_kernel.cu, C++11).
// ============================================================

#include "bam_vchar_kernel.cuh"
#include "page_size_dispatch.h"

#include <nvcompdx.hpp>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BAM_VCHAR_CUDA_CHECK(call) do {                                    \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ── VCHAR page access helpers ──

__device__ __forceinline__ uint32_t dev_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}

__device__ __forceinline__ uint32_t dev_pag_get_oslt(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}

__device__ __forceinline__ uint16_t dev_pagcol_vchar_len(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = dev_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}

__device__ __forceinline__ const char *dev_pagcol_vchar_data(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = dev_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);  // skip len_u16 + pad_u16
}

// ============================================================
// PTX helpers for cp.async prefetch (same pattern as scan.cu)
// ============================================================
template <int Bytes>
__device__ __forceinline__ void vchar_cp_async_ca(void* dst_smem, const void* src_gmem) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp.async supports 4/8/16 B only");
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
        "r"(smem_addr), "l"(src_gmem), "n"(Bytes));
}

__device__ __forceinline__ void vchar_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int NGroup>
__device__ __forceinline__ void vchar_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NGroup) : "memory");
}

__device__ __forceinline__ void vchar_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// ============================================================
// Kernel: decompress_scan_vchar
//
// Thread block: 128 threads (4 warps).
//   Phase 1: Warp 0 (32 threads) decompresses via nvCOMPdx LZ4 Warp API.
//   Phase 2: All 128 threads scan VCHAR records in parallel.
//
// One block processes one page (no grid-stride loop — batch
// processing is done by the host, one batch per kernel launch).
// ============================================================
template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_decompress_scan_vchar_kernel(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start)
{
    // nvCOMPdx decompressor type (warp-level, LZ4)
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t tid     = threadIdx.x;       // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t bid     = blockIdx.x;
    const uint64_t pg      = batch_start + bid;

    // Shared memory: nvCOMPdx workspace for warp 0
    extern __shared__ __align__(8) uint8_t smem[];

    // Per-thread accumulators
    uint64_t my_records  = 0;
    uint64_t my_strlen   = 0;
    uint64_t my_byte_sum = 0;

    // ── Phase 1: LZ4 Decompress (warp 0, 32 threads) ──
    if (warp_id == 0) {
        const void* comp_src = d_staging_buf + (uint64_t)bid * page_size;
        void* decomp_dst = d_decomp_buf + (uint64_t)bid * page_size;

        uint32_t comp_size = d_comp_sizes[pg];
        size_t decomp_size = 0;

        auto decompressor = lz4_decomp_t();
        decompressor.execute(
            comp_src,
            decomp_dst,
            (size_t)comp_size,
            &decomp_size,
            smem,
            nullptr);
    }
    __syncthreads();

    // ── Phase 2: VCHAR record scan with cp.async prefetch (all 128 threads) ──
    constexpr int PREFETCH_BYTES = 4;
    char* smem_buf1 = reinterpret_cast<char*>(smem) + tid * PREFETCH_BYTES * 2;
    char* smem_buf2 = smem_buf1 + PREFETCH_BYTES;

    const char* page = d_decomp_buf + (uint64_t)bid * page_size;
    uint32_t nalloc = dev_pag_get_nalloc(page);

    for (uint32_t slot = tid; slot < nalloc; slot += blockDim.x) {
        uint16_t len = dev_pagcol_vchar_len(page, slot, page_size);
        const char* data = dev_pagcol_vchar_data(page, slot, page_size);

        my_records++;
        my_strlen += len;

        uint64_t bsum = 0;
        int ntiles = len / PREFETCH_BYTES;
        if (ntiles > 0) {
            vchar_cp_async_ca<PREFETCH_BYTES>(smem_buf1, &data[0]);
            vchar_cp_async_commit();
            char* rbuf = smem_buf1;
            int bufidx = 1;

            for (int j = 1; j < ntiles; j++) {
                if (bufidx == 0) {
                    vchar_cp_async_ca<PREFETCH_BYTES>(smem_buf1, &data[j * PREFETCH_BYTES]);
                    rbuf = smem_buf2;
                } else {
                    vchar_cp_async_ca<PREFETCH_BYTES>(smem_buf2, &data[j * PREFETCH_BYTES]);
                    rbuf = smem_buf1;
                }
                bufidx = (bufidx + 1) & 1;
                vchar_cp_async_commit();
                vchar_cp_async_wait_group<1>();

                for (int k = 0; k < PREFETCH_BYTES; k++) {
                    bsum += (uint8_t)rbuf[k];
                }
            }
            rbuf = (bufidx == 0) ? smem_buf2 : smem_buf1;
            vchar_cp_async_wait_all();
            for (int k = 0; k < PREFETCH_BYTES; k++) {
                bsum += (uint8_t)rbuf[k];
            }
        }
        for (uint16_t b = ntiles * PREFETCH_BYTES; b < len; b++) {
            bsum += (uint8_t)data[b];
        }
        my_byte_sum += bsum;
    }

    // ── Reduce per-thread accumulators to global ──
    if (my_records > 0)
        atomicAdd((unsigned long long*)d_total_records, (unsigned long long)my_records);
    if (my_strlen > 0)
        atomicAdd((unsigned long long*)d_total_strlen, (unsigned long long)my_strlen);
    if (my_byte_sum > 0)
        atomicAdd((unsigned long long*)d_total_byte_sum, (unsigned long long)my_byte_sum);
}

// ============================================================
// Host wrapper: bam_vchar_decomp_scan_batch
// ============================================================

void bam_vchar_decomp_scan_batch(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_size)
{
    // ── Shared memory size: max(nvCOMPdx workspace, cp.async buffers) ──
    // cp.async needs 128 threads × 8 bytes = 1024 bytes (reuses smem after decompress)
    constexpr size_t cp_async_smem = 128 * 4 * 2;
    size_t smem_size;
    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        using decomp_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<PS>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        smem_size = decomp_t().shmem_size_group();
    });
    if (smem_size < cp_async_smem) smem_size = cp_async_smem;

    const uint32_t threads_per_block = 128;

    cudaStream_t stream;
    BAM_VCHAR_CUDA_CHECK(cudaStreamCreate(&stream));

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kernel_fn = bam_decompress_scan_vchar_kernel<PS>;
        BAM_VCHAR_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<batch_size, threads_per_block, smem_size, stream>>>(
            d_staging_buf, d_comp_sizes, d_decomp_buf,
            d_total_records, d_total_strlen, d_total_byte_sum,
            page_size, batch_start);
    });

    BAM_VCHAR_CUDA_CHECK(cudaStreamSynchronize(stream));
    BAM_VCHAR_CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Kernel v2: decompress_scan_vchar_v2
//
// Thread block: 128 threads (4 warps).
//   Phase 1: Each warp decompresses its own page via nvCOMPdx LZ4 Warp API.
//            4 pages per block, 4 warps decompress concurrently.
//   Phase 2: All 128 threads scan all 4 decompressed pages together.
//
// Staging buffer layout: slot = bid * 4 + warp_id
// npages_total used for bounds checking at tail batch.
// ============================================================
template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_decompress_scan_vchar_kernel_v2(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint64_t        npages_total)
{
    // nvCOMPdx decompressor type (warp-level, LZ4)
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t tid     = threadIdx.x;       // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t bid     = blockIdx.x;

    // Shared memory: 4 separate regions for 4 warp decompressors
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem_size = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem_size;

    // How many valid pages this block handles (1..4)
    const uint64_t block_page_start = batch_start + (uint64_t)bid * 4;
    const uint32_t valid_pages = (uint32_t)min((uint64_t)4, npages_total - block_page_start);

    // Per-thread accumulators
    uint64_t my_records  = 0;
    uint64_t my_strlen   = 0;
    uint64_t my_byte_sum = 0;

    // ── Phase 1: LZ4 Decompress (each warp decompresses its own page) ──
    if (warp_id < valid_pages) {
        const uint64_t pg = block_page_start + warp_id;
        const uint32_t slot = bid * 4 + warp_id;

        const void* comp_src = d_staging_buf + (uint64_t)slot * page_size;
        void* decomp_dst = d_decomp_buf + (uint64_t)slot * page_size;

        uint32_t comp_size = d_comp_sizes[pg];
        size_t decomp_size = 0;

        auto decompressor = lz4_decomp_t();
        decompressor.execute(
            comp_src,
            decomp_dst,
            (size_t)comp_size,
            &decomp_size,
            my_smem,
            nullptr);
    }
    __syncthreads();

    // ── Phase 2: VCHAR record scan with cp.async prefetch ──
    constexpr int PREFETCH_BYTES = 4;
    char* smem_buf1 = reinterpret_cast<char*>(smem) + tid * PREFETCH_BYTES * 2;
    char* smem_buf2 = smem_buf1 + PREFETCH_BYTES;

    for (uint32_t p = 0; p < valid_pages; p++) {
        const uint32_t slot = bid * 4 + p;
        const char* page = d_decomp_buf + (uint64_t)slot * page_size;
        uint32_t nalloc = dev_pag_get_nalloc(page);

        for (uint32_t s = tid; s < nalloc; s += blockDim.x) {
            uint16_t len = dev_pagcol_vchar_len(page, s, page_size);
            const char* data = dev_pagcol_vchar_data(page, s, page_size);

            my_records++;
            my_strlen += len;

            uint64_t bsum = 0;
            int ntiles = len / PREFETCH_BYTES;
            if (ntiles > 0) {
                vchar_cp_async_ca<PREFETCH_BYTES>(smem_buf1, &data[0]);
                vchar_cp_async_commit();
                char* rbuf = smem_buf1;
                int bufidx = 1;

                for (int j = 1; j < ntiles; j++) {
                    if (bufidx == 0) {
                        vchar_cp_async_ca<PREFETCH_BYTES>(smem_buf1, &data[j * PREFETCH_BYTES]);
                        rbuf = smem_buf2;
                    } else {
                        vchar_cp_async_ca<PREFETCH_BYTES>(smem_buf2, &data[j * PREFETCH_BYTES]);
                        rbuf = smem_buf1;
                    }
                    bufidx = (bufidx + 1) & 1;
                    vchar_cp_async_commit();
                    vchar_cp_async_wait_group<1>();

                    for (int k = 0; k < PREFETCH_BYTES; k++) {
                        bsum += (uint8_t)rbuf[k];
                    }
                }
                rbuf = (bufidx == 0) ? smem_buf2 : smem_buf1;
                vchar_cp_async_wait_all();
                for (int k = 0; k < PREFETCH_BYTES; k++) {
                    bsum += (uint8_t)rbuf[k];
                }
            }
            for (uint16_t b = ntiles * PREFETCH_BYTES; b < len; b++) {
                bsum += (uint8_t)data[b];
            }
            my_byte_sum += bsum;
        }
    }

    // ── Reduce per-thread accumulators to global ──
    if (my_records > 0)
        atomicAdd((unsigned long long*)d_total_records, (unsigned long long)my_records);
    if (my_strlen > 0)
        atomicAdd((unsigned long long*)d_total_strlen, (unsigned long long)my_strlen);
    if (my_byte_sum > 0)
        atomicAdd((unsigned long long*)d_total_byte_sum, (unsigned long long)my_byte_sum);
}

// ============================================================
// Host wrapper: bam_vchar_decomp_scan_batch_v2
// ============================================================

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
    uint64_t        npages_total)
{
    // Shared memory: 4 warp regions for concurrent decompression
    size_t smem_size;
    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        using decomp_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<PS>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        smem_size = decomp_t().shmem_size_group() * 4;
    });

    const uint32_t threads_per_block = 128;

    cudaStream_t stream;
    BAM_VCHAR_CUDA_CHECK(cudaStreamCreate(&stream));

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kernel_fn = bam_decompress_scan_vchar_kernel_v2<PS>;
        BAM_VCHAR_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<batch_blocks, threads_per_block, smem_size, stream>>>(
            d_staging_buf, d_comp_sizes, d_decomp_buf,
            d_total_records, d_total_strlen, d_total_byte_sum,
            page_size, batch_start, npages_total);
    });

    BAM_VCHAR_CUDA_CHECK(cudaStreamSynchronize(stream));
    BAM_VCHAR_CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Kernel v3: decompress_scan_vchar_v3
//
// Same as v2 (4 warps decompress concurrently, 128 threads scan)
// but uses cp.async for VCHAR byte-sum: global -> shared memory
// 4B double-buffered prefetch for stable memory access.
//
// Shared memory layout:
//   Phase 1 (decompress): 4 x warp_smem_size for nvCOMPdx
//   Phase 2 (scan): reuses smem[tid*8 .. tid*8+7] for cp.async
//                    (128 x 8 = 1024 bytes << nvCOMPdx workspace)
// ============================================================
template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_decompress_scan_vchar_kernel_v3(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint64_t        npages_total)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t bid     = blockIdx.x;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem_size = lz4_decomp_t().shmem_size_group();

    const uint64_t block_page_start = batch_start + (uint64_t)bid * 4;
    const uint32_t valid_pages = (uint32_t)min((uint64_t)4, npages_total - block_page_start);

    uint64_t my_records  = 0;
    uint64_t my_strlen   = 0;
    uint64_t my_byte_sum = 0;

    // ── Phase 1: LZ4 Decompress (each warp decompresses its own page) ──
    if (warp_id < valid_pages) {
        uint8_t* my_smem = smem + warp_id * warp_smem_size;
        const uint64_t pg = block_page_start + warp_id;
        const uint32_t slot = bid * 4 + warp_id;

        const void* comp_src = d_staging_buf + (uint64_t)slot * page_size;
        void* decomp_dst = d_decomp_buf + (uint64_t)slot * page_size;

        uint32_t comp_size = d_comp_sizes[pg];
        size_t decomp_size = 0;

        auto decompressor = lz4_decomp_t();
        decompressor.execute(
            comp_src, decomp_dst,
            (size_t)comp_size, &decomp_size,
            my_smem, nullptr);
    }
    __syncthreads();

    // ── Phase 2: VCHAR record scan with cp.async prefetch ──
    // Reuse shared memory for per-thread double buffer (4B x 2 = 8B per thread)
    constexpr int PREFETCH_BYTES = 4;
    char* smem_buf1 = reinterpret_cast<char*>(smem) + tid * PREFETCH_BYTES * 2;
    char* smem_buf2 = smem_buf1 + PREFETCH_BYTES;

    for (uint32_t p = 0; p < valid_pages; p++) {
        const uint32_t slot = bid * 4 + p;
        const char* page = d_decomp_buf + (uint64_t)slot * page_size;
        uint32_t nalloc = dev_pag_get_nalloc(page);

        for (uint32_t s = tid; s < nalloc; s += blockDim.x) {
            uint16_t len = dev_pagcol_vchar_len(page, s, page_size);
            const char* data = dev_pagcol_vchar_data(page, s, page_size);

            my_records++;
            my_strlen += len;

            // Byte-sum with cp.async double-buffered prefetch (4B tiles)
            uint64_t bsum = 0;
            int ntiles = len / PREFETCH_BYTES;
            if (ntiles > 0) {
                // Prefetch first tile
                vchar_cp_async_ca<PREFETCH_BYTES>(smem_buf1, &data[0]);
                vchar_cp_async_commit();
                char* rbuf = smem_buf1;
                int bufidx = 1;

                for (int j = 1; j < ntiles; j++) {
                    if (bufidx == 0) {
                        vchar_cp_async_ca<PREFETCH_BYTES>(
                            smem_buf1, &data[j * PREFETCH_BYTES]);
                        rbuf = smem_buf2;
                    } else {
                        vchar_cp_async_ca<PREFETCH_BYTES>(
                            smem_buf2, &data[j * PREFETCH_BYTES]);
                        rbuf = smem_buf1;
                    }
                    bufidx = (bufidx + 1) & 1;
                    vchar_cp_async_commit();
                    vchar_cp_async_wait_group<1>();

                    for (int k = 0; k < PREFETCH_BYTES; k++) {
                        bsum += (uint8_t)rbuf[k];
                    }
                }

                // Process final tile
                rbuf = (bufidx == 0) ? smem_buf2 : smem_buf1;
                vchar_cp_async_wait_all();
                for (int k = 0; k < PREFETCH_BYTES; k++) {
                    bsum += (uint8_t)rbuf[k];
                }
            }
            // Handle remainder (len % 4)
            for (uint16_t b = ntiles * PREFETCH_BYTES; b < len; b++) {
                bsum += (uint8_t)data[b];
            }
            my_byte_sum += bsum;
        }
    }

    if (my_records > 0)
        atomicAdd((unsigned long long*)d_total_records, (unsigned long long)my_records);
    if (my_strlen > 0)
        atomicAdd((unsigned long long*)d_total_strlen, (unsigned long long)my_strlen);
    if (my_byte_sum > 0)
        atomicAdd((unsigned long long*)d_total_byte_sum, (unsigned long long)my_byte_sum);
}

// ============================================================
// Host wrapper: bam_vchar_decomp_scan_batch_v3_async
// Launches kernel on caller-provided stream without synchronizing.
// ============================================================

static size_t bam_vchar_v3_smem_size(uint32_t page_size) {
    size_t decomp_smem;
    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        using decomp_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<PS>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        decomp_smem = decomp_t().shmem_size_group() * 4;
    });
    return decomp_smem;
}

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
    cudaStream_t    stream)
{
    size_t smem_size = bam_vchar_v3_smem_size(page_size);
    const uint32_t threads_per_block = 128;

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kernel_fn = bam_decompress_scan_vchar_kernel_v3<PS>;
        BAM_VCHAR_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<batch_blocks, threads_per_block, smem_size, stream>>>(
            d_staging_buf, d_comp_sizes, d_decomp_buf,
            d_total_records, d_total_strlen, d_total_byte_sum,
            page_size, batch_start, npages_total);
    });
}

// ============================================================
// Kernel v6: PAR-32K nvCOMPdx decompress + scan
//
// 128 threads (4 warps) per block, each warp handles 1 page.
// Compressed page layout (PAR-32K):
//   [32 × uint32_t comp_sizes][chunk_0][chunk_1]...[chunk_31]
//
// Phase 1: Each warp sequentially decompresses 32 × 32KiB chunks
//          using nvCOMPdx Warp() API (32 threads cooperative).
// Phase 2: All 128 threads cooperatively scan 4 decompressed pages.
// ============================================================

static constexpr uint32_t V6_CHUNK_SZ = 32768u;

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_decompress_scan_vchar_par32k_kernel(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint64_t        npages_total)
{
    using lz4_decomp_32k_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<V6_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_32k_t);

    const uint32_t tid     = threadIdx.x;       // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t bid     = blockIdx.x;

    // Shared memory: 4 separate regions for 4 warp decompressors
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem_size = lz4_decomp_32k_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem_size;

    const uint64_t block_page_start = batch_start + (uint64_t)bid * 4;
    const uint32_t valid_pages = (uint32_t)min((uint64_t)4,
                                                npages_total - block_page_start);

    uint64_t my_records  = 0;
    uint64_t my_strlen   = 0;
    uint64_t my_byte_sum = 0;

    constexpr uint32_t n_chunks = PAGE_SIZE_CONST / V6_CHUNK_SZ;
    constexpr uint32_t hdr_bytes = n_chunks * sizeof(uint32_t);

    // ── Phase 1: PAR-32K decompress (each warp handles 1 page) ──
    if (warp_id < valid_pages) {
        const uint32_t slot = bid * 4 + warp_id;
        const uint8_t* comp_page = (const uint8_t*)(
            d_staging_buf + (uint64_t)slot * page_size);
        uint8_t* decomp_page = (uint8_t*)(
            d_decomp_buf + (uint64_t)slot * page_size);

        // PAR-32K header: [n_chunks × uint32_t comp_sizes]
        const uint32_t* chunk_hdr = (const uint32_t*)comp_page;

        auto decompressor = lz4_decomp_32k_t();
        uint32_t data_offset = hdr_bytes;

        for (uint32_t i = 0; i < n_chunks; i++) {
            uint32_t comp_len = chunk_hdr[i];
            size_t decomp_size = 0;

            decompressor.execute(
                comp_page + data_offset,
                decomp_page + (uint64_t)i * V6_CHUNK_SZ,
                comp_len,
                &decomp_size,
                my_smem,
                nullptr);

            data_offset += comp_len;
        }
    }
    __syncthreads();

    // ── Phase 2: VCHAR scan (all 128 threads, 4 pages) ──
    for (uint32_t p = 0; p < valid_pages; p++) {
        const uint32_t slot = bid * 4 + p;
        const char* page = d_decomp_buf + (uint64_t)slot * page_size;
        uint32_t nalloc = dev_pag_get_nalloc(page);

        for (uint32_t s = tid; s < nalloc; s += blockDim.x) {
            uint16_t len = dev_pagcol_vchar_len(page, s, page_size);
            const char* data = dev_pagcol_vchar_data(page, s, page_size);

            my_records++;
            my_strlen += len;

            uint64_t bsum = 0;
            for (uint16_t b = 0; b < len; b++) {
                bsum += (uint8_t)data[b];
            }
            my_byte_sum += bsum;
        }
    }

    if (my_records > 0)
        atomicAdd((unsigned long long*)d_total_records, (unsigned long long)my_records);
    if (my_strlen > 0)
        atomicAdd((unsigned long long*)d_total_strlen, (unsigned long long)my_strlen);
    if (my_byte_sum > 0)
        atomicAdd((unsigned long long*)d_total_byte_sum, (unsigned long long)my_byte_sum);
}

// ============================================================
// Host wrapper: bam_vchar_decomp_scan_par32k_async
// ============================================================

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
    cudaStream_t    stream)
{
    using decomp_32k_1M_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<V6_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    size_t smem_size = decomp_32k_1M_t().shmem_size_group() * 4;  // 4 warps
    const uint32_t threads_per_block = 128;

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kernel_fn = bam_decompress_scan_vchar_par32k_kernel<PS>;
        BAM_VCHAR_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<batch_blocks, threads_per_block, smem_size, stream>>>(
            d_staging_buf, d_comp_sizes, d_decomp_buf,
            d_total_records, d_total_strlen, d_total_byte_sum,
            page_size, batch_start, npages_total);
    });
}

// ============================================================
// Kernel v7: PAR-8K nvCOMPdx decompress + scan
//
// 128 threads (4 warps) per block, 1 page per block.
// Compressed page layout (PAR-8K):
//   [128 × uint32_t comp_sizes][chunk_0][chunk_1]...[chunk_127]
//
// Phase 1: 4 warps cooperatively decompress 1 page.
//          Each warp handles 32 × 8KiB chunks (= 256 KiB).
// Phase 2: All 128 threads scan the decompressed page.
// ============================================================

static constexpr uint32_t V7_CHUNK_SZ = 8192u;

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_decompress_scan_vchar_par8k_kernel(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint64_t        npages_total)
{
    using lz4_decomp_8k_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<V7_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_8k_t);

    const uint32_t tid     = threadIdx.x;       // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t bid     = blockIdx.x;

    const uint64_t pg = batch_start + (uint64_t)bid;
    if (pg >= npages_total) return;

    // Shared memory: 4 separate regions for 4 warp decompressors
    extern __shared__ __align__(8) uint8_t smem_8k[];
    constexpr size_t warp_smem_size = lz4_decomp_8k_t().shmem_size_group();
    uint8_t* my_smem = smem_8k + warp_id * warp_smem_size;

    // staging/decomp slot = bid (1 page per block)
    const uint32_t slot = bid;

    constexpr uint32_t n_chunks = PAGE_SIZE_CONST / V7_CHUNK_SZ;
    constexpr uint32_t hdr_bytes = n_chunks * sizeof(uint32_t);
    constexpr uint32_t chunks_per_warp = n_chunks / 4;

    // ── Phase 1: PAR-8K decompress (4 warps cooperate on 1 page) ──
    {
        const uint8_t* comp_page = (const uint8_t*)(
            d_staging_buf + (uint64_t)slot * page_size);
        uint8_t* decomp_page = (uint8_t*)(
            d_decomp_buf + (uint64_t)slot * page_size);

        const uint32_t* chunk_hdr = (const uint32_t*)comp_page;

        auto decompressor = lz4_decomp_8k_t();

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
                decomp_page + (uint64_t)chunk_idx * V7_CHUNK_SZ,
                comp_len,
                &decomp_size,
                my_smem,
                nullptr);

            data_offset += comp_len;
        }
    }
    __syncthreads();

    // ── Phase 2: VCHAR scan (all 128 threads, 1 page) ──
    uint64_t my_records  = 0;
    uint64_t my_strlen   = 0;
    uint64_t my_byte_sum = 0;

    {
        const char* page = d_decomp_buf + (uint64_t)slot * page_size;
        uint32_t nalloc = dev_pag_get_nalloc(page);

        for (uint32_t s = tid; s < nalloc; s += blockDim.x) {
            uint16_t len = dev_pagcol_vchar_len(page, s, page_size);
            const char* data = dev_pagcol_vchar_data(page, s, page_size);

            my_records++;
            my_strlen += len;

            uint64_t bsum = 0;
            for (uint16_t b = 0; b < len; b++) {
                bsum += (uint8_t)data[b];
            }
            my_byte_sum += bsum;
        }
    }

    if (my_records > 0)
        atomicAdd((unsigned long long*)d_total_records, (unsigned long long)my_records);
    if (my_strlen > 0)
        atomicAdd((unsigned long long*)d_total_strlen, (unsigned long long)my_strlen);
    if (my_byte_sum > 0)
        atomicAdd((unsigned long long*)d_total_byte_sum, (unsigned long long)my_byte_sum);
}

// ============================================================
// Host wrapper: bam_vchar_decomp_scan_par8k_async
// ============================================================

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
    cudaStream_t    stream)
{
    using decomp_8k_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<V7_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    size_t smem_size = decomp_8k_t().shmem_size_group() * 4;  // 4 warps
    const uint32_t threads_per_block = 128;

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kernel_fn = bam_decompress_scan_vchar_par8k_kernel<PS>;
        BAM_VCHAR_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<batch_pages, threads_per_block, smem_size, stream>>>(
            d_staging_buf, d_comp_sizes, d_decomp_buf,
            d_total_records, d_total_strlen, d_total_byte_sum,
            page_size, batch_start, npages_total);
    });
}

// ============================================================
// Kernel v8: PAR-32K nvCOMPdx decompress + scan, 1-page-per-block
//
// 128 threads (4 warps) per block, 1 page per block.
// Compressed page layout (PAR-32K):
//   [32 × uint32_t comp_sizes][chunk_0][chunk_1]...[chunk_31]
//
// Phase 1: 4 warps cooperatively decompress 1 page.
//          Each warp handles 8 × 32KiB chunks (= 256 KiB).
// Phase 2: All 128 threads scan the decompressed page.
// ============================================================

static constexpr uint32_t V8_CHUNK_SZ = 32768u;

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_decompress_scan_vchar_par32k_v8_kernel(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint64_t        npages_total)
{
    using lz4_decomp_32k_v8_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<V8_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_32k_v8_t);

    const uint32_t tid     = threadIdx.x;       // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t bid     = blockIdx.x;

    const uint64_t pg = batch_start + (uint64_t)bid;
    if (pg >= npages_total) return;

    // Shared memory: 4 separate regions for 4 warp decompressors
    extern __shared__ __align__(8) uint8_t smem_32k_v8[];
    constexpr size_t warp_smem_size = lz4_decomp_32k_v8_t().shmem_size_group();
    uint8_t* my_smem = smem_32k_v8 + warp_id * warp_smem_size;

    // staging/decomp slot = bid (1 page per block)
    const uint32_t slot = bid;

    constexpr uint32_t n_chunks = PAGE_SIZE_CONST / V8_CHUNK_SZ;       // 32
    constexpr uint32_t hdr_bytes = n_chunks * sizeof(uint32_t);        // 128
    constexpr uint32_t chunks_per_warp = n_chunks / 4;                 // 8

    // ── Phase 1: PAR-32K decompress (4 warps cooperate on 1 page) ──
    {
        const uint8_t* comp_page = (const uint8_t*)(
            d_staging_buf + (uint64_t)slot * page_size);
        uint8_t* decomp_page = (uint8_t*)(
            d_decomp_buf + (uint64_t)slot * page_size);

        const uint32_t* chunk_hdr = (const uint32_t*)comp_page;

        auto decompressor = lz4_decomp_32k_v8_t();

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
                decomp_page + (uint64_t)chunk_idx * V8_CHUNK_SZ,
                comp_len,
                &decomp_size,
                my_smem,
                nullptr);

            data_offset += comp_len;
        }
    }
    __syncthreads();

    // ── Phase 2: VCHAR scan (all 128 threads, 1 page) ──
    uint64_t my_records  = 0;
    uint64_t my_strlen   = 0;
    uint64_t my_byte_sum = 0;

    {
        const char* page = d_decomp_buf + (uint64_t)slot * page_size;
        uint32_t nalloc = dev_pag_get_nalloc(page);

        for (uint32_t s = tid; s < nalloc; s += blockDim.x) {
            uint16_t len = dev_pagcol_vchar_len(page, s, page_size);
            const char* data = dev_pagcol_vchar_data(page, s, page_size);

            my_records++;
            my_strlen += len;

            uint64_t bsum = 0;
            for (uint16_t b = 0; b < len; b++) {
                bsum += (uint8_t)data[b];
            }
            my_byte_sum += bsum;
        }
    }

    if (my_records > 0)
        atomicAdd((unsigned long long*)d_total_records, (unsigned long long)my_records);
    if (my_strlen > 0)
        atomicAdd((unsigned long long*)d_total_strlen, (unsigned long long)my_strlen);
    if (my_byte_sum > 0)
        atomicAdd((unsigned long long*)d_total_byte_sum, (unsigned long long)my_byte_sum);
}

// ============================================================
// Host wrapper: bam_vchar_decomp_scan_par32k_v8_async
// ============================================================

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
    cudaStream_t    stream)
{
    using decomp_32k_v8_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<V8_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    size_t smem_size = decomp_32k_v8_t().shmem_size_group() * 4;  // 4 warps
    const uint32_t threads_per_block = 128;

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kernel_fn = bam_decompress_scan_vchar_par32k_v8_kernel<PS>;
        BAM_VCHAR_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<batch_pages, threads_per_block, smem_size, stream>>>(
            d_staging_buf, d_comp_sizes, d_decomp_buf,
            d_total_records, d_total_strlen, d_total_byte_sum,
            page_size, batch_start, npages_total);
    });
}

// ============================================================
// PAR-32K decompress only (no scan): bam_vchar_decomp_par32k_async
//
// Decompresses LZ4-compressed VCHAR/CHAR pages from a staging
// buffer into a contiguous output page buffer at the correct
// page offset.  No VCHAR scanning is performed.
//
// 128 threads (4 warps), 1 page per block.
// Each warp decompresses 8 × 32KiB chunks (= 256KiB, 1/4 of page).
// Output: d_output_pages[(batch_start + bid) * page_size] per block.
// ============================================================

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_vchar_decomp_par32k_kernel(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_output_pages,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint64_t        npages_total)
{
    using lz4_decomp_32k_v8_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<V8_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_32k_v8_t);

    const uint32_t tid     = threadIdx.x;       // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t bid     = blockIdx.x;

    const uint64_t pg = batch_start + (uint64_t)bid;
    if (pg >= npages_total) return;

    // Shared memory: 4 separate regions for 4 warp decompressors
    extern __shared__ __align__(8) uint8_t smem_decomp_only[];
    constexpr size_t warp_smem_size = lz4_decomp_32k_v8_t().shmem_size_group();
    uint8_t* my_smem = smem_decomp_only + warp_id * warp_smem_size;

    // staging slot = bid (1 page per block)
    const uint32_t slot = bid;

    constexpr uint32_t n_chunks = PAGE_SIZE_CONST / V8_CHUNK_SZ;       // 32
    constexpr uint32_t hdr_bytes = n_chunks * sizeof(uint32_t);        // 128
    constexpr uint32_t chunks_per_warp = n_chunks / 4;                 // 8

    // Decompress from staging into output_pages at correct offset
    const uint8_t* comp_page = (const uint8_t*)(
        d_staging_buf + (uint64_t)slot * page_size);
    uint8_t* decomp_page = (uint8_t*)(
        d_output_pages + pg * page_size);

    const uint32_t* chunk_hdr = (const uint32_t*)comp_page;

    auto decompressor = lz4_decomp_32k_v8_t();

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
            decomp_page + (uint64_t)chunk_idx * V8_CHUNK_SZ,
            comp_len,
            &decomp_size,
            my_smem,
            nullptr);

        data_offset += comp_len;
    }
}

// ============================================================
// Host wrapper: bam_vchar_decomp_par32k_async
// ============================================================

void bam_vchar_decomp_par32k_async(
    const char*     d_staging_buf,
    const uint32_t* d_comp_sizes,
    char*           d_output_pages,
    uint32_t        page_size,
    uint64_t        batch_start,
    uint32_t        batch_pages,
    uint64_t        npages_total,
    cudaStream_t    stream)
{
    using decomp_32k_v8_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<V8_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    size_t smem_size = decomp_32k_v8_t().shmem_size_group() * 4;  // 4 warps
    const uint32_t threads_per_block = 128;

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kernel_fn = bam_vchar_decomp_par32k_kernel<PS>;
        BAM_VCHAR_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<batch_pages, threads_per_block, smem_size, stream>>>(
            d_staging_buf, d_comp_sizes, d_output_pages,
            page_size, batch_start, npages_total);
    });
}
