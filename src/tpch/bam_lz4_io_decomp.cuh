#pragma once

// ============================================================
// bam_lz4_io_decomp.cuh — Warp-level device helper: BaM I/O + nvCOMPdx LZ4 decompress
//
// Called by per-query fused kernels. Each warp reads one compressed page
// from NVMe via BaM (lane 0), then decompresses it with nvCOMPdx (all 32 lanes).
//
// Compiled as part of C++17 CUDA TUs with separable compilation + device linking.
// ============================================================

#include "bam_io_device.cuh"
#include <nvcompdx.hpp>

// ── Warp-level BaM I/O + nvCOMPdx LZ4 decompress ──
//
// Must be called by all 32 lanes of the warp.
//
// Parameters:
//   ctrls, pc:       BaM opaque controller/page_cache pointers
//   pc_base_addr:    page_cache base address (for reading I/O result)
//   slot:            page_cache slot (unique per warp, e.g. blockIdx.x * 4 + warp_id)
//   dst:             output buffer for decompressed page (page_size bytes)
//   lba:             NVMe LBA for this page
//   nblk:            number of 512-byte NVMe blocks to read
//   dev:             target device index (RAID0)
//   comp_sz:         compressed size (if >= page_size, direct copy)
//   page_size:       page size in bytes
//   my_smem:         per-warp shared memory for nvCOMPdx (shmem_size_group() bytes)
template <unsigned int PAGE_SIZE_CONST>
__device__ void bam_lz4_io_decomp_warp(
    void* ctrls, void* pc, void* pc_base_addr,
    uint32_t slot, char* dst,
    uint64_t lba, uint32_t nblk, uint32_t dev,
    uint32_t comp_sz, uint32_t page_size,
    uint8_t* my_smem)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    const uint32_t lane = threadIdx.x % 32;

    // Phase 1: BaM I/O (lane 0 only)
    if (lane == 0)
        bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
    __syncwarp();

    // Phase 2: nvCOMPdx LZ4 decompress (all 32 lanes)
    const char* src = (const char*)pc_base_addr + (uint64_t)slot * page_size;
    if (comp_sz < page_size) {
        auto decompressor = lz4_decomp_t();
        size_t dsz = 0;
        decompressor.execute(src, dst, (size_t)comp_sz, &dsz, my_smem, nullptr);
    } else {
        // Incompressible or uncompressed: direct copy (all 32 lanes cooperate)
        const uint32_t n4 = page_size / 4;
        for (uint32_t i = lane; i < n4; i += 32)
            reinterpret_cast<uint32_t*>(dst)[i] =
                reinterpret_cast<const uint32_t*>(src)[i];
    }
}

// ── Warp-level nvCOMPdx LZ4 decompress only (no I/O) ──
//
// Decompresses data already present in page_cache slot `slot`.
// Must be called by all 32 lanes of the warp AFTER the I/O for `slot` has been polled.
//
// Parameters:
//   pc_base_addr:    page_cache base address
//   slot:            page_cache slot containing compressed data
//   dst:             output buffer for decompressed page (page_size bytes)
//   comp_sz:         compressed size (if >= page_size, direct copy)
//   page_size:       page size in bytes
//   my_smem:         per-warp shared memory for nvCOMPdx (shmem_size_group() bytes)
template <unsigned int PAGE_SIZE_CONST>
__device__ void bam_lz4_decomp_only_warp(
    const char* pc_base_addr,
    uint32_t slot, char* dst,
    uint32_t comp_sz, uint32_t page_size,
    uint8_t* my_smem)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    const uint32_t lane = threadIdx.x % 32;
    const char* src = pc_base_addr + (uint64_t)slot * page_size;
    if (comp_sz < page_size) {
        auto decompressor = lz4_decomp_t();
        size_t dsz = 0;
        decompressor.execute(src, dst, (size_t)comp_sz, &dsz, my_smem, nullptr);
    } else {
        // Incompressible or uncompressed: direct copy (all 32 lanes cooperate)
        const uint32_t n4 = page_size / 4;
        for (uint32_t i = lane; i < n4; i += 32)
            reinterpret_cast<uint32_t*>(dst)[i] =
                reinterpret_cast<const uint32_t*>(src)[i];
    }
}

// ── nvCOMPdx shared memory size query (for kernel launch config) ──
template <unsigned int PAGE_SIZE_CONST>
inline size_t bam_lz4_io_decomp_smem_per_warp() {
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());
    return lz4_decomp_t().shmem_size_group();
}
