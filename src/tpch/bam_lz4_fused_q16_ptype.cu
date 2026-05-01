// bam_lz4_fused_q16_ptype.cu — Fused BaM I/O + nvCOMPdx LZ4 + Q16 P_TYPE scan
// 4 warps/block, each warp independently handles IO+decomp+scan for one page.
// __launch_bounds__(128, 8) for max occupancy.
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q16_ptype.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "tpch/page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>

#define FUSED_Q16PT_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

static constexpr int FUSED_Q16PT_WARPS = 4;

__device__ static uint32_t fused_q16pt_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// -- VCHAR page access helpers --

__device__ __forceinline__ static uint32_t fq16pt_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}

__device__ __forceinline__ static uint32_t fq16pt_pag_get_oslt(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}

__device__ __forceinline__ static uint16_t fq16pt_pagcol_vchar_len(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = fq16pt_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}

__device__ __forceinline__ static const char *fq16pt_pagcol_vchar_data(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = fq16pt_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);  // skip len_u16 + pad_u16
}

// -- FNV-1a 64-bit hash --
__device__ __forceinline__ static uint64_t fq16pt_fnv1a64(const char *s, uint16_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (uint16_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Dictionary constants (must match q16.cuh / q16_scan.cu)
static constexpr uint32_t FQ16PT_DICT_CAP  = 512;
static constexpr uint32_t FQ16PT_DICT_MASK = FQ16PT_DICT_CAP - 1;
static constexpr uint32_t FQ16PT_STR_MAX   = 32;

// ════════════════════════════════════════════════════════════════
// Fused Q16 P_TYPE kernel
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void bam_lz4_fused_q16pt_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ16PTypeParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS = FUSED_Q16PT_WARPS;

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    // Each warp owns a dedicated page_cache slot and decomp buffer
    const uint32_t slot = blockIdx.x * WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;
    const uint32_t global_warp = blockIdx.x * WARPS + warp_id;
    const uint32_t warp_stride = gridDim.x * WARPS;

    // Warp-stride persistent loop
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
                nblk = fused_q16pt_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
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

        // ── VCHAR scan: NOT LIKE 'MEDIUM POLISHED%' + dictionary type_id ──
        {
            const char* page = my_decomp;
            uint32_t nalloc = fq16pt_pag_get_nalloc(page);
            uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

            for (uint32_t s = lane; s < nalloc; s += 32) {
                uint64_t row_id = row_base + s;

                uint16_t vlen = fq16pt_pagcol_vchar_len(page, s, p.page_size);
                const char* vdata = fq16pt_pagcol_vchar_data(page, s, p.page_size);

                // Check NOT LIKE 'MEDIUM POLISHED%'
                if (vlen >= 15) {
                    const char mp[] = "MEDIUM POLISHED";
                    bool match = true;
                    for (int i = 0; i < 15; i++) {
                        if (vdata[i] != mp[i]) { match = false; break; }
                    }
                    if (match) {
                        p.d_type_ids[row_id] = UINT32_MAX;
                        continue;
                    }
                }

                // Hash type string
                uint64_t h = fq16pt_fnv1a64(vdata, vlen);

                // Probe/insert into dictionary
                uint32_t dslot = (uint32_t)h & FQ16PT_DICT_MASK;
                while (true) {
                    uint64_t prev = atomicCAS(
                        reinterpret_cast<unsigned long long *>(&p.d_dict_keys[dslot]),
                        (unsigned long long)UINT64_MAX,
                        (unsigned long long)h);

                    if (prev == UINT64_MAX) {
                        // We inserted — assign new type_id
                        uint32_t tid_val = atomicAdd(p.d_type_id_counter, 1);
                        char *dst = p.d_dict_strs + (uint64_t)dslot * FQ16PT_STR_MAX;
                        for (uint16_t i = 0; i < vlen; i++) dst[i] = vdata[i];
                        p.d_dict_lens[dslot] = vlen;
                        __threadfence();
                        p.d_dict_type_ids[dslot] = tid_val;
                        p.d_type_ids[row_id] = tid_val;
                        break;
                    }
                    if (prev == h) {
                        // Key exists — wait for type_id to be written
                        uint32_t tid_val;
                        do {
                            __threadfence();
                            tid_val = p.d_dict_type_ids[dslot];
                        } while (tid_val == UINT32_MAX);
                        p.d_type_ids[row_id] = tid_val;
                        break;
                    }
                    dslot = (dslot + 1) & FQ16PT_DICT_MASK;
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

struct BAMFusedQ16PTypeContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_q16pt_ctx_t bam_fused_q16pt_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ16PTypeContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 4 slots per block (1 per warp)
    const uint32_t num_slots = num_blocks * FUSED_Q16PT_WARPS;
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: 1 page per warp slot
    size_t decomp_size = (size_t)num_slots * page_size;
    FUSED_Q16PT_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q16pt_ctx_t>(ctx);
}

static void bam_fused_q16pt_launch(
    BAMFusedQ16PTypeContext* ctx,
    const BAMFusedQ16PTypeParams& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * FUSED_Q16PT_WARPS;
        auto kernel_fn = bam_lz4_fused_q16pt_kernel<PS>;
        FUSED_Q16PT_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_Q16PT_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q16pt_run_async(
    bam_fused_q16pt_ctx_t ctx_handle,
    const BAMFusedQ16PTypeParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ16PTypeContext*>(ctx_handle);
    bam_fused_q16pt_launch(ctx, params, stream);
}

void bam_fused_q16pt_destroy(bam_fused_q16pt_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ16PTypeContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}
