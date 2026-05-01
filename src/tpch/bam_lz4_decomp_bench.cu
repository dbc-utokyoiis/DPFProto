// ============================================================
// BaM LZ4 decomp-only microbenchmark.
//
// Phase 1 (preload): BaM sync I/O fills page_cache slots with compressed data.
// Phase 2 (decomp):  Warp-level nvCOMPdx LZ4 decomp, measured with cudaEvents.
//
// Compiled as C++17 with separable compilation + device linking
// (calls bam_io_read_page_device from bam_io_device,
//  uses nvCOMPdx for LZ4 decompression).
// ============================================================

#include "bam_lz4_decomp_bench.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#define DBENCH_CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// ── BaM nblk alignment fix (same as fused kernel) ──
__device__ static uint32_t decomp_bench_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// ── Phase 1 kernel: pre-load compressed pages into page_cache via BaM sync I/O ──
// 1 warp per block. Each warp loads multiple pages round-robin.
__global__ void bam_lz4_decomp_bench_preload(
    void*       ctrls,
    void*       pc,
    uint32_t    total_pages,
    uint64_t    field_start_page_id,
    uint64_t*   d_partition_start_lbas,
    uint64_t*   d_comp_offsets,     // nullptr if uncompressed
    uint32_t*   d_comp_sizes,       // nullptr if uncompressed
    uint32_t    page_nblk,          // page_size / 512
    uint32_t    n_devices,
    uint32_t    page_size)
{
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp_id = blockIdx.x;

    for (uint32_t pg = warp_id; pg < total_pages; pg += gridDim.x) {
        uint64_t global_pg = field_start_page_id + pg;
        uint32_t dev = global_pg % n_devices;
        uint64_t lba;
        uint32_t nblk;

        if (d_comp_offsets) {
            lba = d_partition_start_lbas[dev] + d_comp_offsets[pg] / 512;
            uint32_t comp_sz = d_comp_sizes[pg];
            nblk = decomp_bench_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
        } else {
            uint64_t local_pg = global_pg / n_devices;
            lba = d_partition_start_lbas[dev] + local_pg * page_nblk;
            nblk = page_nblk;
        }

        uint32_t slot = pg;  // 1:1 mapping: slot = page index
        if (lane == 0)
            bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
        __syncwarp();
    }
}

// ── Phase 2 kernel: decomp-only, measured with cudaEvents ──
// 1 warp per block (32 threads). Each warp decompresses assigned pages.
template <unsigned int PAGE_SIZE_CONST>
__global__ void bam_lz4_decomp_bench_kernel(
    const char* pc_base_addr,
    char*       d_decomp_buf,       // [num_warps * page_size]
    uint32_t    total_pages,
    uint32_t*   d_comp_sizes,       // [total_pages], nullptr if uncompressed
    uint32_t    page_size)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    extern __shared__ __align__(8) uint8_t smem[];

    const uint32_t warp_id = blockIdx.x;
    const uint32_t lane = threadIdx.x % 32;
    uint8_t* my_smem = smem;  // 1 warp per block → offset 0

    // Each warp writes to its own decomp buffer region (reused across pages)
    char* my_decomp = d_decomp_buf + (uint64_t)warp_id * page_size;

    for (uint32_t pg = warp_id; pg < total_pages; pg += gridDim.x) {
        uint32_t slot = pg;
        uint32_t comp_sz = d_comp_sizes ? d_comp_sizes[pg] : page_size;

        bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
            pc_base_addr, slot, my_decomp,
            comp_sz, page_size, my_smem);
    }
}

// ── Host API ──

LZ4DecompBenchResult bam_lz4_decomp_bench_run(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t total_pages,
    uint64_t field_start_page_id,
    uint32_t n_devices,
    const uint64_t* partition_start_lbas,
    const uint32_t* h_comp_sizes,
    const uint64_t* h_comp_offsets,
    uint32_t num_warps)
{
    const uint32_t page_nblk = page_size / 512;

    // ── Page cache: one slot per page (pre-loaded, then reused by decomp) ──
    bam_io_page_cache_t io_pc = bam_io_page_cache_create(
        ctrl_handle, page_size, total_pages);

    void* d_ctrls        = bam_io_page_cache_get_d_ctrls(io_pc);
    void* d_pc           = bam_io_page_cache_get_d_pc_ptr(io_pc);
    const char* pc_base  = (const char*)bam_io_page_cache_get_base_addr(io_pc);

    // ── Upload partition_start_lbas ──
    uint64_t* d_partition_start_lbas;
    DBENCH_CUDA_CHECK(cudaMalloc(&d_partition_start_lbas, n_devices * sizeof(uint64_t)));
    DBENCH_CUDA_CHECK(cudaMemcpy(d_partition_start_lbas, partition_start_lbas,
                                  n_devices * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // ── Upload comp metadata (if compressed) ──
    uint64_t* d_comp_offsets = nullptr;
    uint32_t* d_comp_sizes_gpu = nullptr;

    if (h_comp_sizes && h_comp_offsets) {
        DBENCH_CUDA_CHECK(cudaMalloc(&d_comp_offsets, total_pages * sizeof(uint64_t)));
        DBENCH_CUDA_CHECK(cudaMemcpy(d_comp_offsets, h_comp_offsets,
                                      total_pages * sizeof(uint64_t), cudaMemcpyHostToDevice));
        DBENCH_CUDA_CHECK(cudaMalloc(&d_comp_sizes_gpu, total_pages * sizeof(uint32_t)));
        DBENCH_CUDA_CHECK(cudaMemcpy(d_comp_sizes_gpu, h_comp_sizes,
                                      total_pages * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // ── Phase 1: Pre-load all pages into page_cache ──
    {
        // Use up to 512 warps for fast pre-load
        uint32_t preload_warps = std::min(total_pages, 512u);
        bam_lz4_decomp_bench_preload<<<preload_warps, 32>>>(
            d_ctrls, d_pc, total_pages, field_start_page_id,
            d_partition_start_lbas, d_comp_offsets, d_comp_sizes_gpu,
            page_nblk, n_devices, page_size);
        DBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        fprintf(stderr, "  Pre-loaded %u pages into page_cache\n", total_pages);
    }

    // ── Decomp output buffer: one page per warp ──
    char* d_decomp_buf;
    DBENCH_CUDA_CHECK(cudaMalloc(&d_decomp_buf, (size_t)num_warps * page_size));

    // ── Phase 2: Decomp benchmark ──
    cudaEvent_t start, stop;
    DBENCH_CUDA_CHECK(cudaEventCreate(&start));
    DBENCH_CUDA_CHECK(cudaEventCreate(&stop));

    // Determine smem size
    size_t smem_size;
    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        smem_size = bam_lz4_io_decomp_smem_per_warp<PS>();
    });

    // Warmup
    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto fn = bam_lz4_decomp_bench_kernel<PS>;
        DBENCH_CUDA_CHECK(cudaFuncSetAttribute(
            fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        fn<<<num_warps, 32, smem_size>>>(
            pc_base, d_decomp_buf, total_pages, d_comp_sizes_gpu, page_size);
    });
    DBENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    DBENCH_CUDA_CHECK(cudaEventRecord(start));
    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto fn = bam_lz4_decomp_bench_kernel<PS>;
        fn<<<num_warps, 32, smem_size>>>(
            pc_base, d_decomp_buf, total_pages, d_comp_sizes_gpu, page_size);
    });
    DBENCH_CUDA_CHECK(cudaEventRecord(stop));
    DBENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    DBENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // ── Compute results ──
    double total_decompressed_bytes = (double)total_pages * page_size;
    double throughput_gbs = total_decompressed_bytes / (elapsed_ms * 1e6);

    // pages_per_warp = ceil(total_pages / num_warps)
    uint32_t pages_per_warp = (total_pages + num_warps - 1) / num_warps;
    double us_per_page = (elapsed_ms * 1000.0) / pages_per_warp;

    LZ4DecompBenchResult result;
    result.num_warps = num_warps;
    result.total_pages = total_pages;
    result.page_size = page_size;
    result.elapsed_ms = elapsed_ms;
    result.decomp_throughput_gbs = throughput_gbs;
    result.us_per_page_per_warp = us_per_page;

    // ── Cleanup ──
    DBENCH_CUDA_CHECK(cudaEventDestroy(start));
    DBENCH_CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_decomp_buf);
    if (d_comp_offsets) cudaFree(d_comp_offsets);
    if (d_comp_sizes_gpu) cudaFree(d_comp_sizes_gpu);
    cudaFree(d_partition_start_lbas);
    bam_io_page_cache_destroy(io_pc);

    return result;
}
