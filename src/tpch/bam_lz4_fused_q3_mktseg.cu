// bam_lz4_fused_q3_mktseg.cu — CUSTOMER Phase 1 kernels
// for Q3 gidp+bam+fusion.
//
// Kernel 1: Warp-stride IO for all CUSTOMER pages (CK+MK),
//           then nvCOMPdx LZ4 decomp ALL pages to page-indexed staging.
//           128 threads/block (4 warps), nvCOMPdx shared memory.
// Kernel 2: Cooperative kernel — record-level parallel
//           CK flatten → grid.sync() → MK BUILDING scan + hash insert.
//           256 threads/block, NO shared memory, maximum parallelism.

#include "bam_lz4_fused_q3_mktseg.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "bam_bulk_read.cuh"
#include "page_size_dispatch.h"

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

#define Q3P1_CUDA_CHECK(call) do {                                             \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                           \
                cudaGetErrorString(err), __FILE__, __LINE__);                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

static constexpr uint32_t Q3P1_WARPS_PER_BLOCK = 4;
static constexpr uint32_t Q3P1_THREADS_K1 = 128;
static constexpr uint32_t Q3P1_THREADS_K2 = 256;

// "BUILDING" as uint64_t in little-endian
static constexpr uint64_t Q3P1_BUILDING_U64 = 0x474E49444C495542ULL;

// Hash function (same as q3_scan.cu / bam_q3_kernel.cu)
__device__ __forceinline__ uint32_t q3p1_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

// Binary search upper_bound (same as q13_scan.cu)
__device__ __forceinline__ uint32_t q3p1_upper_bound(
    const uint64_t* __restrict__ data, uint32_t n, uint64_t val)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (data[mid] <= val)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ════════════════════════════════════════════════════════════════
// Kernel 1: IO all pages + decomp ALL pages to page-indexed staging
//
// Warp-stride: each warp round-robins pages.
//   - Lane 0 does bam_io_read_page_device (blocking)
//   - ALL pages (CK and MK): decompress to d_staging[j * page_size]
// ════════════════════════════════════════════════════════════════

template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128)
void q3_cust_io_decomp_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    const BamBulkReadDesc* __restrict__ d_descs,
    char*       d_staging,
    Q3CustIODecompParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t warp_id     = threadIdx.x / 32;
    const uint32_t lane        = threadIdx.x % 32;
    const uint32_t global_warp = blockIdx.x * Q3P1_WARPS_PER_BLOCK + warp_id;
    const uint32_t total_warps = gridDim.x * Q3P1_WARPS_PER_BLOCK;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    for (uint32_t j = global_warp; j < p.total_descs; j += total_warps) {
        // IO: read page j into page_cache slot j
        const BamBulkReadDesc& desc = d_descs[j];
        if (lane == 0)
            bam_io_read_page_device(ctrls, pc, desc.lba, desc.nblocks, j, desc.device);
        __syncwarp();

        // Decomp: all pages to staging[j * page_size]
        uint32_t comp_sz;
        if (j < p.ck_npages)
            comp_sz = p.ck_comp_sizes ? p.ck_comp_sizes[j] : p.page_size;
        else
            comp_sz = p.mk_comp_sizes ? p.mk_comp_sizes[j - p.ck_npages] : p.page_size;

        char* dst = d_staging + (uint64_t)j * p.page_size;
        bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
            pc_base_addr, j, dst, comp_sz, p.page_size, my_smem);
    }
}

// ════════════════════════════════════════════════════════════════
// Kernel 2: CK flatten + grid.sync() + MK BUILDING scan + hash insert
//
// Cooperative kernel: record-level parallelism, grid-stride loops.
// Phase A: flatten CK INT64 pages (binary search upper_bound)
// Phase B: MK BUILDING string comparison + hash set insert
// ════════════════════════════════════════════════════════════════

__global__ void q3_cust_process_kernel(Q3CustProcessParams p)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const uint64_t tid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    // Phase A: CK flatten — INT64 pages, data at offset 16
    for (uint64_t idx = tid; idx < p.nrecs_customer; idx += stride) {
        uint32_t page_idx = q3p1_upper_bound(p.ck_prefix_sum, p.ck_npages, idx);
        uint32_t local_idx = (page_idx == 0)
            ? (uint32_t)idx
            : (uint32_t)(idx - p.ck_prefix_sum[page_idx - 1]);

        const char* page = p.d_staging + (uint64_t)page_idx * p.page_size;
        const int64_t* values = reinterpret_cast<const int64_t*>(page + 16);
        p.d_c_custkey_flat[idx] = (uint64_t)values[local_idx];
    }

    grid.sync();

    // Phase B: MK segment scan + hash set insert
    // num_segments == 0 → all customers pass (Q3SEL 100%)
    // num_segments > 0 → check against segment_values[]
    for (uint64_t idx = tid; idx < p.nrecs_customer; idx += stride) {
        uint32_t page_idx = q3p1_upper_bound(p.mk_prefix_sum, p.mk_npages, idx);
        uint32_t local_slot = (page_idx == 0)
            ? (uint32_t)idx
            : (uint32_t)(idx - p.mk_prefix_sum[page_idx - 1]);

        if (p.num_segments > 0) {
            const char* page = p.d_staging
                + (uint64_t)(p.mk_page_offset + page_idx) * p.page_size;
            const char* rec = page + 12 + p.padded_len * local_slot;

            uint32_t lo = *reinterpret_cast<const uint32_t*>(rec);
            uint32_t hi = *reinterpret_cast<const uint32_t*>(rec + 4);
            uint64_t val8 = ((uint64_t)hi << 32) | lo;
            bool match = false;
            for (uint32_t s = 0; s < p.num_segments; s++) {
                if (val8 == p.segment_values[s]) { match = true; break; }
            }
            if (!match) continue;
        }

        // Filter passed → insert custkey into hash set
        uint64_t custkey = p.d_c_custkey_flat[idx];
        uint32_t hslot = q3p1_hash64(custkey) & p.custkey_set_mask;
        while (true) {
            uint64_t prev = atomicCAS(
                (unsigned long long*)&p.d_custkey_set[hslot],
                0xFFFFFFFFFFFFFFFFULL, (unsigned long long)custkey);
            if (prev == 0xFFFFFFFFFFFFFFFFULL || prev == custkey) break;
            hslot = (hslot + 1) & p.custkey_set_mask;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// max_blocks queries
// ════════════════════════════════════════════════════════════════

uint32_t q3_cust_io_decomp_max_blocks(uint32_t page_size)
{
    int max_blocks_per_sm = 0;

    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * Q3P1_WARPS_PER_BLOCK;
        auto kfn = q3_cust_io_decomp_kernel<PS>;
        cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kfn, Q3P1_THREADS_K1, smem_size);
    });

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    uint32_t total = (uint32_t)max_blocks_per_sm * (uint32_t)sm_count;
    fprintf(stderr, "[q3_cust_io_decomp] max_blocks_per_sm=%d sm_count=%d max_total=%u\n",
            max_blocks_per_sm, sm_count, total);
    return total;
}

uint32_t q3_cust_process_max_blocks()
{
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, q3_cust_process_kernel, Q3P1_THREADS_K2, 0);

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    uint32_t total = (uint32_t)max_blocks_per_sm * (uint32_t)sm_count;
    fprintf(stderr, "[q3_cust_proc] max_blocks_per_sm=%d sm_count=%d max_total=%u\n",
            max_blocks_per_sm, sm_count, total);
    return total;
}

// ════════════════════════════════════════════════════════════════
// Launch functions
// ════════════════════════════════════════════════════════════════

void q3_cust_io_decomp_launch(
    void* d_ctrls, void* d_pc, const char* pc_base_addr,
    const BamBulkReadDesc* d_descs,
    char* d_staging,
    const Q3CustIODecompParams& params,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * Q3P1_WARPS_PER_BLOCK;
        auto kfn = q3_cust_io_decomp_kernel<PS>;
        Q3P1_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kfn<<<num_blocks, Q3P1_THREADS_K1, smem_size, stream>>>(
            d_ctrls, d_pc, pc_base_addr, d_descs, d_staging, params);
    });
    Q3P1_CUDA_CHECK(cudaGetLastError());
}

void q3_cust_process_launch(
    const Q3CustProcessParams& params,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    Q3CustProcessParams p = params;
    void* kernel_args[] = {(void*)&p};
    Q3P1_CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)q3_cust_process_kernel,
        dim3(num_blocks), dim3(Q3P1_THREADS_K2),
        kernel_args, 0, stream));
}
