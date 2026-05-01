// bam_lz4_fused_q13_comment_v2.cu — Q6-pattern: independent warps, max occupancy
// 4 warps/block, each warp independently handles IO+decomp+scan for different pages.
// __launch_bounds__(128, 8) → 8 blocks/SM → 32 independent decomp warps/SM.
// No cross-warp __syncthreads — fully independent per-warp page processing.
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q13_comment_v2.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "tpch/page_size_dispatch.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define FUSED_Q13CV2_CUDA_CHECK(call) do {                                    \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

static constexpr int FUSED_Q13CV2_WARPS = 4;

// BaM nblk alignment fix
__device__ static uint32_t fused_q13cv2_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// -- VCHAR page access helpers --

__device__ __forceinline__ static uint32_t fq13cv2_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}

__device__ __forceinline__ static uint32_t fq13cv2_pag_get_oslt(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}

__device__ __forceinline__ static uint16_t fq13cv2_pagcol_vchar_len(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = fq13cv2_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}

__device__ __forceinline__ static const char *fq13cv2_pagcol_vchar_data(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = fq13cv2_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);  // skip len_u16 + pad_u16
}

// -- KMP multi-pattern matching --
__device__ static bool fq13cv2_kmp_match(
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
        while (l > 0 && patterns[p_offset + l] != c)
            l = next[p_offset + l - 1];
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

// ================================================================
// Decomp+Scan only kernel: no BaM IO code
//
// Reads compressed pages from GPU staging buffer, decompresses with
// nvCOMPdx warp-level LZ4, then runs KMP scan.
// No BaM IO functions → no 62-reg constraint → potentially better occupancy.
// 4 warps/block, persistent warp-stride loop within batch.
// ================================================================
static constexpr int DS_Q13_WARPS = 4;

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void q13_decomp_scan_kernel(
    char*               d_decomp_buf,
    Q13DecompScanParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS = DS_Q13_WARPS;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    const uint32_t slot = blockIdx.x * WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t global_warp = blockIdx.x * WARPS + warp_id;
    const uint32_t warp_stride = gridDim.x * WARPS;

    for (uint32_t local_pg = global_warp; local_pg < p.batch_count; local_pg += warp_stride) {
        uint64_t abs_pg = p.pg_start + local_pg;

        // ── nvCOMPdx LZ4 decompress from staging_io ──
        const char* src = p.staging_io + (uint64_t)local_pg * p.page_size;
        if (p.is_compressed) {
            uint32_t comp_sz = p.d_comp_sizes[abs_pg];
            if (comp_sz < p.page_size) {
                auto decompressor = lz4_decomp_t();
                size_t dsz = 0;
                decompressor.execute(src, my_decomp, (size_t)comp_sz, &dsz, my_smem, nullptr);
            } else {
                const uint32_t n4 = p.page_size / 4;
                for (uint32_t i = lane; i < n4; i += 32)
                    reinterpret_cast<uint32_t*>(my_decomp)[i] =
                        reinterpret_cast<const uint32_t*>(src)[i];
            }
        } else {
            const uint32_t n4 = p.page_size / 4;
            for (uint32_t i = lane; i < n4; i += 32)
                reinterpret_cast<uint32_t*>(my_decomp)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
        }

        // ── KMP scan ──
        {
            const char* page = my_decomp;
            uint32_t nalloc = fq13cv2_pag_get_nalloc(page);
            uint64_t row_base = (abs_pg == 0) ? 0 : p.d_prefix_sum[abs_pg - 1];
            uint64_t my_qualifying = 0;

            for (uint32_t s = lane; s < nalloc; s += 32) {
                uint64_t row_id = row_base + s;

                uint16_t vlen = fq13cv2_pagcol_vchar_len(page, s, p.page_size);
                const char* vdata = fq13cv2_pagcol_vchar_data(page, s, p.page_size);

                bool matched = fq13cv2_kmp_match(
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
    }
}

// ================================================================
// Decomp+Scan context and host API
// ================================================================
struct Q13DecompScanContext {
    char*    d_decomp_buf;
    uint32_t page_size;
    uint32_t num_blocks;
};

q13_decomp_scan_ctx_t q13_decomp_scan_create(
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new Q13DecompScanContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * DS_Q13_WARPS;
    size_t decomp_size = (size_t)num_slots * page_size;
    FUSED_Q13CV2_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<q13_decomp_scan_ctx_t>(ctx);
}

void q13_decomp_scan_async(
    q13_decomp_scan_ctx_t ctx_handle,
    const Q13DecompScanParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<Q13DecompScanContext*>(ctx_handle);
    constexpr uint32_t THREADS = DS_Q13_WARPS * 32;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>()
                         * DS_Q13_WARPS;
        auto kernel_fn = q13_decomp_scan_kernel<PS>;
        FUSED_Q13CV2_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_decomp_buf, p);
    });

    FUSED_Q13CV2_CUDA_CHECK(cudaGetLastError());
}

void q13_decomp_scan_destroy(q13_decomp_scan_ctx_t ctx_handle)
{
    auto* ctx = static_cast<Q13DecompScanContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    delete ctx;
}

// ================================================================
// Fused Q13 O_COMMENT kernel: Q6-pattern independent warps
//
// Each warp independently: IO → decomp → scan → next page
// 4 warps/block × 8 blocks/SM = 32 decomp warps/SM
// ================================================================
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void bam_lz4_fused_q13cv2_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ13Cv2Params p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS = FUSED_Q13CV2_WARPS;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    // Each warp owns a dedicated page_cache slot and decomp buffer
    const uint32_t slot = blockIdx.x * WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    // Shared memory: nvCOMPdx per-warp region
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Global warp index and stride for page assignment
    const uint32_t global_warp = blockIdx.x * WARPS + warp_id;
    const uint32_t warp_stride = gridDim.x * WARPS;

    // Warp-stride persistent loop: each warp processes independent pages
    for (uint64_t pg = global_warp; pg < p.npages; pg += warp_stride) {

        // ── IO + LZ4 decompress (warp-cooperative) ──
        {
            uint64_t global_pg = p.field_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            uint32_t comp_sz;

            if (p.is_compressed) {
                lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                comp_sz = p.d_comp_sizes[pg];
                nblk = fused_q13cv2_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
                nblk = p.page_size / 512;
                comp_sz = p.page_size;
            }

            bam_lz4_io_decomp_warp<PAGE_SIZE_CONST>(
                ctrls, pc, (void*)pc_base_addr,
                slot, my_decomp,
                lba, nblk, dev, comp_sz, p.page_size, my_smem);
        }
        // No __syncthreads — warp is self-contained

        // ── KMP scan (32 threads within this warp) ──
        {
            const char* page = my_decomp;
            uint32_t nalloc = fq13cv2_pag_get_nalloc(page);
            uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
            uint64_t my_qualifying = 0;

            for (uint32_t s = lane; s < nalloc; s += 32) {
                uint64_t row_id = row_base + s;

                uint16_t vlen = fq13cv2_pagcol_vchar_len(page, s, p.page_size);
                const char* vdata = fq13cv2_pagcol_vchar_data(page, s, p.page_size);

                bool matched = fq13cv2_kmp_match(
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
    }
}

// ================================================================
// Fused BaM IO + LZ4 decomp + INT64 flatten kernel
// Same warp pattern: 4 warps/block, independent, persistent loop
// ================================================================
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void bam_lz4_fused_flatten_i64_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedFlattenI64Params p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS = FUSED_Q13CV2_WARPS;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    const uint32_t slot = blockIdx.x * WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;
    const uint32_t global_warp = blockIdx.x * WARPS + warp_id;
    const uint32_t warp_stride = gridDim.x * WARPS;

    for (uint64_t pg = global_warp; pg < p.npages; pg += warp_stride) {
        // ── IO + LZ4 decompress ──
        {
            uint64_t global_pg = p.field_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            uint32_t comp_sz;

            if (p.is_compressed) {
                lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                comp_sz = p.d_comp_sizes[pg];
                nblk = fused_q13cv2_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
                nblk = p.page_size / 512;
                comp_sz = p.page_size;
            }

            bam_lz4_io_decomp_warp<PAGE_SIZE_CONST>(
                ctrls, pc, (void*)pc_base_addr,
                slot, my_decomp,
                lba, nblk, dev, comp_sz, p.page_size, my_smem);
        }

        // ── Flatten INT64: page[16..] → d_output[row_base..] ──
        {
            const char* page = my_decomp;
            uint32_t nalloc = *reinterpret_cast<const uint32_t*>(page);
            uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
            const int64_t* values = reinterpret_cast<const int64_t*>(page + 16);

            for (uint32_t s = lane; s < nalloc; s += 32) {
                p.d_output[row_base + s] = (uint64_t)values[s];
            }
        }
    }
}

// ================================================================
// Host API
// ================================================================

struct BAMFusedQ13Cv2Context {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_q13c_v2_ctx_t bam_fused_q13c_v2_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ13Cv2Context();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 4 slots per block (1 per warp)
    const uint32_t num_slots = num_blocks * FUSED_Q13CV2_WARPS;
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: 1 page per warp slot
    size_t decomp_size = (size_t)num_slots * page_size;
    FUSED_Q13CV2_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q13c_v2_ctx_t>(ctx);
}

static void bam_fused_q13cv2_launch(
    BAMFusedQ13Cv2Context* ctx,
    const BAMFusedQ13Cv2Params& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = FUSED_Q13CV2_WARPS * 32;  // 128

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>()
                         * FUSED_Q13CV2_WARPS;
        auto kernel_fn = bam_lz4_fused_q13cv2_kernel<PS>;
        FUSED_Q13CV2_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_Q13CV2_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q13c_v2_run_async(
    bam_fused_q13c_v2_ctx_t ctx_handle,
    const BAMFusedQ13Cv2Params& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ13Cv2Context*>(ctx_handle);
    bam_fused_q13cv2_launch(ctx, params, stream);
}

void bam_fused_q13c_v2_flatten_i64_async(
    bam_fused_q13c_v2_ctx_t ctx_handle,
    const BAMFusedFlattenI64Params& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ13Cv2Context*>(ctx_handle);
    constexpr uint32_t THREADS = FUSED_Q13CV2_WARPS * 32;
    uint32_t num_blocks = ctx->num_blocks;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>()
                         * FUSED_Q13CV2_WARPS;
        auto kernel_fn = bam_lz4_fused_flatten_i64_kernel<PS>;
        FUSED_Q13CV2_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_size));
        kernel_fn<<<num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_Q13CV2_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q13c_v2_destroy(bam_fused_q13c_v2_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ13Cv2Context*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}
