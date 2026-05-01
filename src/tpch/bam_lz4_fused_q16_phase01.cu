// bam_lz4_fused_q16_phase01.cu — Fused BaM I/O + nvCOMPdx LZ4 kernels
// for Q16 Phase 0 (SUPPLIER) and Phase 1 (PART) smaller fields.
// 4 warps/block, __launch_bounds__(128, 8), persistent warp-stride loop.
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q16_phase01.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "tpch/page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>

#define FUSED_Q16P01_CUDA_CHECK(call) do {                                    \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

static constexpr int Q16P01_WARPS = 4;

__device__ static uint32_t q16p01_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// ── Common IO+decomp helper macro ──
// Computes LBA, nblk, dev, comp_sz from IOBase fields, then calls bam_lz4_io_decomp_warp
#define Q16P01_IO_DECOMP(PAGE_SIZE_CONST, p, pg, slot, my_decomp, my_smem, ctrls, pc, pc_base_addr) \
    do {                                                                      \
        uint64_t global_pg = (p).field_start_page_id + (pg);                  \
        uint32_t ndev = ((p).n_devices > 1) ? (p).n_devices : 1;             \
        uint32_t dev = global_pg % ndev;                                      \
        uint64_t lba; uint32_t nblk; uint32_t comp_sz;                        \
        if ((p).is_compressed) {                                              \
            lba = (p).partition_start_lbas[dev] + (p).d_comp_offsets[pg] / 512; \
            comp_sz = (p).d_comp_sizes[pg];                                   \
            nblk = q16p01_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);      \
        } else {                                                              \
            uint64_t local_pg = global_pg / ndev;                             \
            lba = (p).partition_start_lbas[dev] + local_pg * ((p).page_size / 512); \
            nblk = (p).page_size / 512;                                       \
            comp_sz = (p).page_size;                                          \
        }                                                                     \
        bam_lz4_io_decomp_warp<PAGE_SIZE_CONST>(                              \
            ctrls, pc, (void*)pc_base_addr,                                   \
            slot, my_decomp, lba, nblk, dev, comp_sz, (p).page_size, my_smem); \
    } while (0)

// ════════════════════════════════════════════════════════════════
// Kernel 1: INT64 flatten
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void q16p01_flatten_i64_kernel(
    void* ctrls, void* pc, const char* pc_base_addr,
    char* d_decomp_buf, BAMFusedQ16FlattenI64Params p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() + nvcompdx::SM<800>());
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t slot = blockIdx.x * Q16P01_WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t warp_stride = gridDim.x * Q16P01_WARPS;

    for (uint64_t pg = blockIdx.x * Q16P01_WARPS + warp_id; pg < p.npages; pg += warp_stride) {
        Q16P01_IO_DECOMP(PAGE_SIZE_CONST, p, pg, slot, my_decomp, my_smem, ctrls, pc, pc_base_addr);

        const char* page = my_decomp;
        uint32_t nalloc = *reinterpret_cast<const uint32_t*>(page);
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        const int64_t* values = reinterpret_cast<const int64_t*>(page + 16);

        for (uint32_t s = lane; s < nalloc; s += 32)
            p.d_output[row_base + s] = (uint64_t)values[s];
    }
}

// ════════════════════════════════════════════════════════════════
// Kernel 2: INT32 flatten → uint32_t
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void q16p01_flatten_i32_kernel(
    void* ctrls, void* pc, const char* pc_base_addr,
    char* d_decomp_buf, BAMFusedQ16FlattenI32Params p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() + nvcompdx::SM<800>());
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t slot = blockIdx.x * Q16P01_WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t warp_stride = gridDim.x * Q16P01_WARPS;

    for (uint64_t pg = blockIdx.x * Q16P01_WARPS + warp_id; pg < p.npages; pg += warp_stride) {
        Q16P01_IO_DECOMP(PAGE_SIZE_CONST, p, pg, slot, my_decomp, my_smem, ctrls, pc, pc_base_addr);

        const char* page = my_decomp;
        uint32_t nalloc = *reinterpret_cast<const uint32_t*>(page);
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        const int32_t* values = reinterpret_cast<const int32_t*>(page + 12);

        for (uint32_t s = lane; s < nalloc; s += 32)
            p.d_output[row_base + s] = (uint32_t)values[s];
    }
}

// ════════════════════════════════════════════════════════════════
// Kernel 2b: INT32 flatten → uint64_t (zero-extend)
// Same as Kernel 2 but outputs uint64_t for direct use as HT keys/values.
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void q16p01_flatten_i32_widen_kernel(
    void* ctrls, void* pc, const char* pc_base_addr,
    char* d_decomp_buf, BAMFusedQ16FlattenI32WidenParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() + nvcompdx::SM<800>());
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t slot = blockIdx.x * Q16P01_WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t warp_stride = gridDim.x * Q16P01_WARPS;

    for (uint64_t pg = blockIdx.x * Q16P01_WARPS + warp_id; pg < p.npages; pg += warp_stride) {
        Q16P01_IO_DECOMP(PAGE_SIZE_CONST, p, pg, slot, my_decomp, my_smem, ctrls, pc, pc_base_addr);

        const char* page = my_decomp;
        uint32_t nalloc = *reinterpret_cast<const uint32_t*>(page);
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        const int32_t* values = reinterpret_cast<const int32_t*>(page + 12);

        for (uint32_t s = lane; s < nalloc; s += 32)
            p.d_output[row_base + s] = static_cast<uint64_t>(static_cast<uint32_t>(values[s]));
    }
}

// ════════════════════════════════════════════════════════════════
// Kernel 3: CHAR brand_id extraction
// CHAR page format: pag_head(12B) + padded_len * slot_id
// Brand string: "Brand#XY" → brand_id = (X-'1')*5 + (Y-'1')
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void q16p01_brand_kernel(
    void* ctrls, void* pc, const char* pc_base_addr,
    char* d_decomp_buf, BAMFusedQ16BrandParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() + nvcompdx::SM<800>());
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t slot = blockIdx.x * Q16P01_WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t warp_stride = gridDim.x * Q16P01_WARPS;

    for (uint64_t pg = blockIdx.x * Q16P01_WARPS + warp_id; pg < p.npages; pg += warp_stride) {
        Q16P01_IO_DECOMP(PAGE_SIZE_CONST, p, pg, slot, my_decomp, my_smem, ctrls, pc, pc_base_addr);

        const char* page = my_decomp;
        uint32_t nalloc = *reinterpret_cast<const uint32_t*>(page);
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        const char* data = page + 12;  // pag_head = 12B

        for (uint32_t s = lane; s < nalloc; s += 32) {
            const char* brand = data + (uint64_t)p.padded_len * s;
            uint32_t d1 = brand[6] - '1';
            uint32_t d2 = brand[7] - '1';
            p.d_brand_ids[row_base + s] = d1 * 5 + d2;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Kernel 4: VCHAR S_COMMENT KMP scan → excluded suppkeys
// ════════════════════════════════════════════════════════════════

// VCHAR page access helpers
__device__ __forceinline__ static uint32_t q16p01_pag_get_oslt(
    const char* page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t*>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}

__device__ __forceinline__ static uint16_t q16p01_vchar_len(
    const char* page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = q16p01_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t*>(page + oslt);
}

__device__ __forceinline__ static const char* q16p01_vchar_data(
    const char* page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = q16p01_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);
}

__device__ static bool q16p01_kmp_match(
    const char* __restrict__ str, int str_len,
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

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void q16p01_supplier_scan_kernel(
    void* ctrls, void* pc, const char* pc_base_addr,
    char* d_decomp_buf, BAMFusedQ16SupplierScanParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() + nvcompdx::SM<800>());
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t slot = blockIdx.x * Q16P01_WARPS + warp_id;
    char* my_decomp = d_decomp_buf + (uint64_t)slot * p.page_size;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t warp_stride = gridDim.x * Q16P01_WARPS;

    for (uint64_t pg = blockIdx.x * Q16P01_WARPS + warp_id; pg < p.npages; pg += warp_stride) {
        Q16P01_IO_DECOMP(PAGE_SIZE_CONST, p, pg, slot, my_decomp, my_smem, ctrls, pc, pc_base_addr);

        const char* page = my_decomp;
        uint32_t nalloc = *reinterpret_cast<const uint32_t*>(page);
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

        for (uint32_t s = lane; s < nalloc; s += 32) {
            uint16_t vlen = q16p01_vchar_len(page, s, p.page_size);
            const char* vdata = q16p01_vchar_data(page, s, p.page_size);

            bool matched = q16p01_kmp_match(
                vdata, (int)vlen,
                p.d_patterns, p.d_next,
                p.d_pattern_offsets, p.d_pattern_lengths,
                p.num_patterns);

            if (matched) {
                uint64_t suppkey = p.d_s_suppkey_flat[row_base + s];
                uint32_t idx = atomicAdd(p.d_excl_count, 1);
                p.d_excl_suppkeys[idx] = suppkey;
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

// ── Split IO entry (same layout as PARTSUPP Q16PSSplitIoEntry) ──
struct Q16P01SplitIoEntry {
    void*    qp;
    uint16_t cid;
    uint16_t pad;
    uint32_t comp_sz;
};

struct BAMFusedQ16P01Context {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
    uint32_t    num_slots;
    // Split mode resources
    Q16P01SplitIoEntry* d_queue;
    cudaEvent_t         ev_submit_done;
};

bam_fused_q16p01_ctx_t bam_fused_q16p01_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_slots)
{
    auto* ctx = new BAMFusedQ16P01Context();
    ctx->page_size = page_size;

    // BaM page_cache has a HW/driver limit on slot count.
    // Cap at sm_count * 8 * WARPS (proven-working limit).
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const uint32_t max_slots = (uint32_t)(sm_count * 8) * Q16P01_WARPS;
    if (num_slots > max_slots)
        num_slots = max_slots;
    ctx->num_slots = num_slots;

    ctx->num_blocks = std::min((num_slots + Q16P01_WARPS - 1) / Q16P01_WARPS,
                               (uint32_t)(sm_count * 8));

    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    size_t decomp_size = (size_t)num_slots * page_size;
    FUSED_Q16P01_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    // Split mode: IO queue + event
    FUSED_Q16P01_CUDA_CHECK(cudaMalloc(&ctx->d_queue, num_slots * sizeof(Q16P01SplitIoEntry)));
    FUSED_Q16P01_CUDA_CHECK(cudaEventCreate(&ctx->ev_submit_done));

    return static_cast<bam_fused_q16p01_ctx_t>(ctx);
}

// ── Launch helpers ──

template <typename KernelFunc, typename Params>
static void q16p01_launch(BAMFusedQ16P01Context* ctx, KernelFunc kernel_fn,
                           const Params& p, size_t smem_size, cudaStream_t stream)
{
    FUSED_Q16P01_CUDA_CHECK(cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
    kernel_fn<<<p.num_blocks, 128, smem_size, stream>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
        ctx->d_decomp_buf, p);
    FUSED_Q16P01_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q16p01_flatten_i64_async(
    bam_fused_q16p01_ctx_t ctx_handle,
    const BAMFusedQ16FlattenI64Params& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem = bam_lz4_io_decomp_smem_per_warp<PS>() * Q16P01_WARPS;
        q16p01_launch(ctx, q16p01_flatten_i64_kernel<PS>, params, smem, stream);
    });
}

void bam_fused_q16p01_flatten_i32_async(
    bam_fused_q16p01_ctx_t ctx_handle,
    const BAMFusedQ16FlattenI32Params& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem = bam_lz4_io_decomp_smem_per_warp<PS>() * Q16P01_WARPS;
        q16p01_launch(ctx, q16p01_flatten_i32_kernel<PS>, params, smem, stream);
    });
}

void bam_fused_q16p01_flatten_i32_widen_async(
    bam_fused_q16p01_ctx_t ctx_handle,
    const BAMFusedQ16FlattenI32WidenParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem = bam_lz4_io_decomp_smem_per_warp<PS>() * Q16P01_WARPS;
        q16p01_launch(ctx, q16p01_flatten_i32_widen_kernel<PS>, params, smem, stream);
    });
}

void bam_fused_q16p01_brand_async(
    bam_fused_q16p01_ctx_t ctx_handle,
    const BAMFusedQ16BrandParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem = bam_lz4_io_decomp_smem_per_warp<PS>() * Q16P01_WARPS;
        q16p01_launch(ctx, q16p01_brand_kernel<PS>, params, smem, stream);
    });
}

void bam_fused_q16p01_supplier_scan_async(
    bam_fused_q16p01_ctx_t ctx_handle,
    const BAMFusedQ16SupplierScanParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem = bam_lz4_io_decomp_smem_per_warp<PS>() * Q16P01_WARPS;
        q16p01_launch(ctx, q16p01_supplier_scan_kernel<PS>, params, smem, stream);
    });
}

// ════════════════════════════════════════════════════════════════
// Split IO/Decomp kernels
// ════════════════════════════════════════════════════════════════

// Submit kernel: 1 thread per page, submits BaM IO and records queue entry.
__global__ void q16p01_split_submit_kernel(
    void* ctrls, void* pc,
    Q16P01SplitIoEntry* queue,
    uint32_t npages,
    uint32_t slot_base,
    uint32_t qp_base,
    uint32_t qp_range,
    uint64_t field_start_page_id,
    const uint64_t* d_comp_offsets,
    const uint32_t* d_comp_sizes,
    bool is_compressed,
    uint64_t part_lba0, uint64_t part_lba1, uint64_t part_lba2, uint64_t part_lba3,
    uint32_t n_devices,
    uint32_t page_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npages) return;

    uint64_t global_pg = field_start_page_id + idx;
    uint32_t ndev = (n_devices > 1) ? n_devices : 1;
    uint32_t dev = global_pg % ndev;

    uint64_t part_lbas[4] = {part_lba0, part_lba1, part_lba2, part_lba3};
    uint64_t lba;
    uint32_t nblk;
    uint32_t comp_sz;

    if (is_compressed) {
        lba = part_lbas[dev] + d_comp_offsets[idx] / 512;
        comp_sz = d_comp_sizes[idx];
        nblk = q16p01_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = part_lbas[dev] + local_pg * (page_size / 512);
        nblk = page_size / 512;
        comp_sz = page_size;
    }

    void* qp_out = nullptr;
    uint16_t cid_out = 0;
    bam_io_submit_page_device_qp(ctrls, pc, lba, nblk,
                                  slot_base + idx, qp_base + (idx % qp_range), dev,
                                  &qp_out, &cid_out);

    queue[idx].qp = qp_out;
    queue[idx].cid = cid_out;
    queue[idx].pad = 0;
    queue[idx].comp_sz = comp_sz;
}

// Decomp kernel: each warp polls 1 page at a time, decompresses to output buffer.
// Output: d_output_buf[pg * PAGE_SIZE_CONST] for page pg.
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128)
void q16p01_split_decomp_kernel(
    const char*              pc_base_addr,
    char*                    d_output_buf,
    const Q16P01SplitIoEntry* __restrict__ queue,
    uint32_t                 total_pages,
    uint32_t                 slot_base)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() + nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS = Q16P01_WARPS;
    const uint32_t lane    = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t global_warp = blockIdx.x * WARPS + warp_id;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t total_warps = gridDim.x * WARPS;

    for (uint32_t pg = global_warp; pg < total_pages; pg += total_warps) {
        // Poll for IO completion
        if (lane == 0)
            bam_io_poll_page_device(queue[pg].qp, queue[pg].cid);
        __syncwarp();

        uint32_t comp_sz = queue[pg].comp_sz;
        const char* src = pc_base_addr + (uint64_t)(slot_base + pg) * PAGE_SIZE_CONST;
        char*       dst = d_output_buf + (uint64_t)pg * PAGE_SIZE_CONST;

        if (comp_sz < PAGE_SIZE_CONST) {
            auto decompressor = lz4_decomp_t();
            size_t dsz = 0;
            decompressor.execute(src, dst, (size_t)comp_sz, &dsz, my_smem, nullptr);
        } else {
            const uint32_t n4 = PAGE_SIZE_CONST / 4;
            for (uint32_t i = lane; i < n4; i += 32)
                reinterpret_cast<uint32_t*>(dst)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Split Host API
// ════════════════════════════════════════════════════════════════

void bam_fused_q16p01_split_batch(
    bam_fused_q16p01_ctx_t ctx_handle,
    const BAMFusedQ16IOBase& io,
    uint32_t pg_base,
    uint32_t batch_np,
    uint32_t slot_base,
    cudaStream_t stream_io,
    cudaStream_t stream_comp)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    const uint32_t n_qps = bam_io_page_cache_get_n_qps(ctx->io_pc, 0);

    // Phase 1: Submit all IOs for this batch
    constexpr uint32_t TPB = 32;
    uint32_t grid = (batch_np + TPB - 1) / TPB;
    q16p01_split_submit_kernel<<<grid, TPB, 0, stream_io>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->d_queue,
        batch_np, slot_base,
        0 /*qp_base*/, n_qps /*qp_range*/,
        io.field_start_page_id + pg_base,
        io.d_comp_offsets ? io.d_comp_offsets + pg_base : nullptr,
        io.d_comp_sizes   ? io.d_comp_sizes   + pg_base : nullptr,
        io.is_compressed,
        io.partition_start_lbas[0], io.partition_start_lbas[1],
        io.partition_start_lbas[2], io.partition_start_lbas[3],
        io.n_devices, io.page_size);
    FUSED_Q16P01_CUDA_CHECK(cudaGetLastError());
    FUSED_Q16P01_CUDA_CHECK(cudaEventRecord(ctx->ev_submit_done, stream_io));

    // Phase 2: Poll + decomp (on stream_comp, waits for submit)
    FUSED_Q16P01_CUDA_CHECK(cudaStreamWaitEvent(stream_comp, ctx->ev_submit_done));

    constexpr uint32_t WARPS = Q16P01_WARPS;
    uint32_t decomp_blocks = std::min((batch_np + WARPS - 1) / WARPS, ctx->num_blocks);

    dispatch_page_size(io.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * WARPS;
        auto kernel_fn = q16p01_split_decomp_kernel<PS>;
        FUSED_Q16P01_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<decomp_blocks, WARPS * 32, smem_size, stream_comp>>>(
            ctx->pc_base_addr, ctx->d_decomp_buf, ctx->d_queue,
            batch_np, slot_base);
    });
    FUSED_Q16P01_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q16p01_split_submit_only(
    bam_fused_q16p01_ctx_t ctx_handle,
    const BAMFusedQ16IOBase& io,
    uint32_t pg_base,
    uint32_t batch_np,
    uint32_t slot_base,
    cudaStream_t stream_io)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    const uint32_t n_qps = bam_io_page_cache_get_n_qps(ctx->io_pc, 0);

    constexpr uint32_t TPB = 32;
    uint32_t grid = (batch_np + TPB - 1) / TPB;
    q16p01_split_submit_kernel<<<grid, TPB, 0, stream_io>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->d_queue + slot_base,
        batch_np, slot_base,
        0 /*qp_base*/, n_qps /*qp_range*/,
        io.field_start_page_id + pg_base,
        io.d_comp_offsets ? io.d_comp_offsets + pg_base : nullptr,
        io.d_comp_sizes   ? io.d_comp_sizes   + pg_base : nullptr,
        io.is_compressed,
        io.partition_start_lbas[0], io.partition_start_lbas[1],
        io.partition_start_lbas[2], io.partition_start_lbas[3],
        io.n_devices, io.page_size);
    FUSED_Q16P01_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q16p01_split_decomp_only(
    bam_fused_q16p01_ctx_t ctx_handle,
    uint32_t page_size,
    uint32_t batch_np,
    uint32_t slot_base,
    cudaStream_t stream_comp)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);

    constexpr uint32_t WARPS = Q16P01_WARPS;
    uint32_t decomp_blocks = std::min((batch_np + WARPS - 1) / WARPS, ctx->num_blocks);

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * WARPS;
        auto kernel_fn = q16p01_split_decomp_kernel<PS>;
        FUSED_Q16P01_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<decomp_blocks, WARPS * 32, smem_size, stream_comp>>>(
            ctx->pc_base_addr, ctx->d_decomp_buf, ctx->d_queue + slot_base,
            batch_np, slot_base);
    });
    FUSED_Q16P01_CUDA_CHECK(cudaGetLastError());
}

char* bam_fused_q16p01_get_decomp_buf(bam_fused_q16p01_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    return ctx->d_decomp_buf;
}

uint32_t bam_fused_q16p01_get_num_slots(bam_fused_q16p01_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    return ctx->num_slots;
}

cudaEvent_t bam_fused_q16p01_get_submit_event(bam_fused_q16p01_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    return ctx->ev_submit_done;
}

void bam_fused_q16p01_destroy(bam_fused_q16p01_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ16P01Context*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    if (ctx->d_queue) cudaFree(ctx->d_queue);
    cudaEventDestroy(ctx->ev_submit_done);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}
