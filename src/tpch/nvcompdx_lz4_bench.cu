// Pure nvCOMPdx warp-level LZ4 decompression benchmark.
// No BaM dependency. Compressed data in plain GPU memory.

#include <cuda_runtime.h>
#include <nvcompdx.hpp>
#include <cstdio>
#include "nvcompdx_lz4_bench.cuh"
#include "page_size_dispatch.h"

#define NVLZ4_CUDA_CHECK(call)                                          \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                            \
        }                                                               \
    } while (0)

// ── Decomp kernel: 1 warp per block ──
template <unsigned int PAGE_SIZE_CONST>
__global__ void nvcompdx_lz4_bench_kernel(
    const char* __restrict__ comp_buf,
    const uint64_t* __restrict__ comp_offsets,
    const uint32_t* __restrict__ comp_sizes,
    char*       decomp_buf,
    uint32_t    total_pages,
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
    uint8_t* my_smem = smem;

    // Each warp reuses the same decomp region (output is discarded)
    char* my_decomp = decomp_buf + (uint64_t)warp_id * page_size;

    for (uint32_t pg = warp_id; pg < total_pages; pg += gridDim.x) {
        const char* src = comp_buf + comp_offsets[pg];
        uint32_t comp_sz = comp_sizes[pg];

        if (comp_sz < page_size) {
            auto decompressor = lz4_decomp_t();
            size_t dsz = 0;
            decompressor.execute(src, my_decomp, (size_t)comp_sz, &dsz, my_smem, nullptr);
        } else {
            // Incompressible: warp-cooperative copy
            const uint32_t lane = threadIdx.x % 32;
            const uint32_t n4 = page_size / 4;
            for (uint32_t i = lane; i < n4; i += 32)
                reinterpret_cast<uint32_t*>(my_decomp)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
        }
    }
}

// ── Shared memory size query ──
template <unsigned int PAGE_SIZE_CONST>
static size_t nvcompdx_lz4_smem_per_warp() {
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());
    return lz4_decomp_t().shmem_size_group();
}

// ── Host API ──
NvcompdxLz4BenchResult nvcompdx_lz4_bench_run(
    const char* d_comp_buf,
    const uint64_t* d_comp_offsets,
    const uint32_t* d_comp_sizes,
    char* d_decomp_buf,
    uint32_t total_pages,
    uint32_t page_size,
    uint32_t num_warps)
{
    size_t smem_size;
    void (*kernel_fn)(const char*, const uint64_t*, const uint32_t*,
                      char*, uint32_t, uint32_t);

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        smem_size = nvcompdx_lz4_smem_per_warp<PS>();
        kernel_fn = nvcompdx_lz4_bench_kernel<PS>;
    });

    cudaFuncSetAttribute(kernel_fn,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    // Warmup
    kernel_fn<<<num_warps, 32, smem_size>>>(
        d_comp_buf, d_comp_offsets, d_comp_sizes,
        d_decomp_buf, total_pages, page_size);
    NVLZ4_CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    NVLZ4_CUDA_CHECK(cudaEventCreate(&start));
    NVLZ4_CUDA_CHECK(cudaEventCreate(&stop));

    NVLZ4_CUDA_CHECK(cudaEventRecord(start));
    kernel_fn<<<num_warps, 32, smem_size>>>(
        d_comp_buf, d_comp_offsets, d_comp_sizes,
        d_decomp_buf, total_pages, page_size);
    NVLZ4_CUDA_CHECK(cudaEventRecord(stop));
    NVLZ4_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    NVLZ4_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    double total_decompressed_bytes = (double)total_pages * page_size;
    double throughput_gbs = total_decompressed_bytes / (elapsed_ms * 1e6);

    uint32_t pages_per_warp = (total_pages + num_warps - 1) / num_warps;
    double us_per_page = (elapsed_ms * 1000.0) / pages_per_warp;

    NvcompdxLz4BenchResult result;
    result.num_warps = num_warps;
    result.total_pages = total_pages;
    result.page_size = page_size;
    result.elapsed_ms = elapsed_ms;
    result.decomp_throughput_gbs = throughput_gbs;
    result.us_per_page_per_warp = us_per_page;

    NVLZ4_CUDA_CHECK(cudaEventDestroy(start));
    NVLZ4_CUDA_CHECK(cudaEventDestroy(stop));

    return result;
}
