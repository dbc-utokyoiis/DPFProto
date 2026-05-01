// bam_lz4_fused_q13_comment.cu — Fused BaM I/O + custom LZ4 decomp + KMP scan
// for Q13 O_COMMENT (standard LZ4 compression).
// Uses lz4_decompress_warp (no nvCOMPdx, no shared memory for decomp).

#include "bam_lz4_fused_q13_comment.cuh"
#include "bam_io_device.cuh"
#include "lz4_decomp.cuh"

#include <cstdio>
#include <cstdlib>

#define FUSED_Q13C_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// BaM nblk alignment fix
__device__ static uint32_t fused_q13c_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// ── VCHAR page access helpers ──

__device__ __forceinline__ static uint32_t fq13c_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}

__device__ __forceinline__ static uint32_t fq13c_pag_get_oslt(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}

__device__ __forceinline__ static uint16_t fq13c_pagcol_vchar_len(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = fq13c_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}

__device__ __forceinline__ static const char *fq13c_pagcol_vchar_data(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = fq13c_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);  // skip len_u16 + pad_u16
}

// ── KMP multi-pattern matching ──
// Matches "special" then "requests" sequentially.
// Returns true if BOTH patterns are found (LIKE '%special%requests%').
// Q13 filter: NOT LIKE → qualifying = !matched.

__device__ static bool fq13c_kmp_match(
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

// ════════════════════════════════════════════════════════════════
// Fused Q13 O_COMMENT kernel: block-stride persistent loop
// Warp 0: I/O + LZ4 decompress (1 page per iteration)
// All 128 threads: KMP scan + custkey extraction
// Uses lz4_decompress_warp (no nvCOMPdx, no shared memory for decomp).
// ════════════════════════════════════════════════════════════════
__global__ __launch_bounds__(128, 8)
void bam_lz4_fused_q13c_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ13CParams p)
{
    constexpr uint32_t THREADS = 128;

    const uint32_t tid     = threadIdx.x;
    const uint32_t lane    = tid % 32;
    const uint32_t warp_id = tid / 32;

    // Warp 0 owns the page_cache slot for this block
    const uint32_t slot = blockIdx.x;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Block-stride persistent loop
    for (uint64_t pg = blockIdx.x; pg < p.npages; pg += gridDim.x) {

        // ── Phase 1+2: Warp 0 does I/O + LZ4 decompress ──
        char* my_decomp = d_decomp_buf + (uint64_t)blockIdx.x * p.page_size;

        if (warp_id == 0) {
            uint64_t global_pg = p.field_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            uint32_t comp_sz;

            if (p.is_compressed) {
                lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                comp_sz = p.d_comp_sizes[pg];
                nblk = fused_q13c_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
                nblk = p.page_size / 512;
                comp_sz = p.page_size;
            }

            // BaM I/O (lane 0 only)
            if (lane == 0)
                bam_io_read_page_device(ctrls, pc, lba, nblk, slot, dev);
            __syncwarp();

            // LZ4 decompress
            const char* src = pc_base_addr + (uint64_t)slot * p.page_size;
            if (comp_sz < p.page_size) {
                lz4_decompress_warp(
                    (const uint8_t*)src, comp_sz,
                    (uint8_t*)my_decomp, p.page_size);
            } else {
                // Incompressible: cooperative copy
                const uint32_t n4 = p.page_size / 4;
                for (uint32_t i = lane; i < n4; i += 32)
                    reinterpret_cast<uint32_t*>(my_decomp)[i] =
                        reinterpret_cast<const uint32_t*>(src)[i];
            }
        }
        __syncthreads();  // Wait for decompress to complete

        // ── Phase 3: KMP scan + custkey extraction (all 128 threads) ──
        {
            const char* page = my_decomp;
            uint32_t nalloc = fq13c_pag_get_nalloc(page);

            uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

            uint64_t my_qualifying = 0;

            for (uint32_t s = tid; s < nalloc; s += THREADS) {
                uint64_t row_id = row_base + s;

                uint16_t vlen = fq13c_pagcol_vchar_len(page, s, p.page_size);
                const char* vdata = fq13c_pagcol_vchar_data(page, s, p.page_size);

                bool matched = fq13c_kmp_match(
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

        __syncthreads();  // Before next page iteration
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

struct BAMFusedQ13CContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;   // [num_blocks * page_size] (1 page per block)
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_q13c_ctx_t bam_fused_q13c_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ13CContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 1 slot per block (only warp 0 does I/O)
    const uint32_t num_slots = num_blocks;

    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: 1 page per block
    size_t decomp_size = (size_t)num_blocks * page_size;
    FUSED_Q13C_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q13c_ctx_t>(ctx);
}

static void bam_fused_q13c_launch(
    BAMFusedQ13CContext* ctx,
    const BAMFusedQ13CParams& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 128;
    bam_lz4_fused_q13c_kernel<<<p.num_blocks, THREADS, 0, stream>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
        ctx->d_decomp_buf, p);
    FUSED_Q13C_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q13c_run_async(
    bam_fused_q13c_ctx_t ctx_handle,
    const BAMFusedQ13CParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ13CContext*>(ctx_handle);
    bam_fused_q13c_launch(ctx, params, stream);
}

void bam_fused_q13c_destroy(bam_fused_q13c_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ13CContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}
