// bam_lz4_decomp.cu — Device-side LZ4 batch decompression via nvCOMPdx
// Compiled as CUDA C++17 with separable compilation.

#include "bam_lz4_decomp.cuh"
#include "tpch/page_size_dispatch.h"
#include <nvcompdx.hpp>

// ────────────────────────────────────────────────────────
// Kernel: 4 warps per block, each warp decompresses one page.
// Pages larger than comp_size >= page_size are copied directly.
// ────────────────────────────────────────────────────────
template <unsigned int PAGE_SIZE_CONST>
__global__ void bam_lz4_decomp_kernel(
    const char* __restrict__ d_comp_pages,
    char*       __restrict__ d_decomp_pages,
    const uint32_t* __restrict__ d_comp_sizes,
    const uint32_t* __restrict__ d_page_indices,
    uint32_t npages,
    uint32_t page_size)
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
    const uint32_t lane    = tid % 32;
    const uint32_t warps_per_block = blockDim.x / 32;

    // Each warp processes pages in round-robin across blocks
    uint32_t slot = blockIdx.x * warps_per_block + warp_id;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem_size = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem_size;

    for (; slot < npages; slot += gridDim.x * warps_per_block) {
        uint32_t comp_sz = d_comp_sizes[slot];
        uint32_t out_pg  = d_page_indices[slot];

        const char* src = d_comp_pages  + (uint64_t)slot  * page_size;
        char*       dst = d_decomp_pages + (uint64_t)out_pg * page_size;

        if (comp_sz < page_size) {
            // LZ4 decompress (warp-cooperative)
            size_t decomp_size = 0;
            auto decompressor = lz4_decomp_t();
            decompressor.execute(
                src, dst,
                static_cast<size_t>(comp_sz),
                &decomp_size,
                my_smem,
                nullptr);
        } else {
            // Incompressible page: direct copy (all 32 lanes cooperate)
            const uint32_t n4 = page_size / 4;
            for (uint32_t i = lane; i < n4; i += 32) {
                reinterpret_cast<uint32_t*>(dst)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
            }
        }
    }
}

// ────────────────────────────────────────────────────────
// Host API
// ────────────────────────────────────────────────────────
void bam_lz4_batch_decompress(
    const char* d_comp_pages,
    char* d_decomp_pages,
    const uint32_t* d_comp_sizes,
    const uint32_t* d_page_indices,
    uint32_t npages,
    uint32_t page_size,
    cudaStream_t stream)
{
    if (npages == 0) return;

    constexpr uint32_t WARPS_PER_BLOCK = 4;
    constexpr uint32_t THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128

    // One warp per page, 4 warps per block
    uint32_t num_blocks = (npages + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    // Cap at SM count for efficiency
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    num_blocks = std::min(num_blocks, static_cast<uint32_t>(sm_count * 2));

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        using decomp_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<PS>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        size_t smem_size = decomp_t().shmem_size_group() * WARPS_PER_BLOCK;

        auto kernel = bam_lz4_decomp_kernel<PS>;
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));

        kernel<<<num_blocks, THREADS_PER_BLOCK, smem_size, stream>>>(
            d_comp_pages, d_decomp_pages, d_comp_sizes, d_page_indices,
            npages, page_size);
    });
}
