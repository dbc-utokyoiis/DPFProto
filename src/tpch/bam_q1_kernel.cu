// ============================================================
// Q1 Unified Kernel: 7-field fused IO + PFOR decompress + eval
//
// All 7 LINEITEM columns are INT32+PFOR (CHAR(1) stored as ASCII int):
//   [0]=L_SHIPDATE, [1]=L_QUANTITY, [2]=L_EXTENDEDPRICE,
//   [3]=L_DISCOUNT, [4]=L_TAX, [5]=L_RETURNFLAG, [6]=L_LINESTATUS
//
// Single kernel: BaM IO → PFOR decompress → evaluate → aggregate.
// 1 page_cache slot per block (sequential field-by-field IO with slot reuse).
// 128 threads/block.
//
// BaM IO is called via bam_io_device.cuh (compiled C++11, device-linked).
// Compiled as CUDA C++17 with separable compilation + device linking.
// ============================================================

#include "bam_q1_kernel.cuh"
#include "bam_io_device.cuh"

// PFOR decompression support (LoadBinPackTile, decodeElement)
#include "kernel/binpack_kernel.cuh"

// Q1 aggregation index constants (must match kernel/tpch/q1.cuh)
#ifndef Q1_SUM_QTY
enum Q1U_AggIdx {
    Q1_SUM_QTY = 0,
    Q1_SUM_BASE_PRICE = 1,
    Q1_SUM_DISC_PRICE = 2,
    Q1_SUM_CHARGE = 3,
    Q1_SUM_DISCOUNT = 4,
    Q1_COUNT = 5,
    Q1_SUM_CHARGE_HI = 6,
};
#endif

#include "tpch/page_size_dispatch.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BAM_Q1_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// NVMe PRP boundary workaround: transfers of 9-16 sectors (crossing one
// 4KB controller page boundary) may not DMA correctly with BaM page_cache.
// Bump to 17 sectors to use a different PRP path.
#define Q1F_NVM_CTRL_PAGE_BLOCKS 8

__device__ __forceinline__ uint32_t q1f_safe_io_nblocks(uint32_t comp_bytes) {
    uint32_t nblk = (comp_bytes + 511) / 512;
    if (nblk > Q1F_NVM_CTRL_PAGE_BLOCKS && nblk <= Q1F_NVM_CTRL_PAGE_BLOCKS * 2)
        nblk = Q1F_NVM_CTRL_PAGE_BLOCKS * 2 + 1;
    return nblk;
}

// ============================================================
// PFOR decompression for INT32 fields (ported from bam_kernel.cu)
// Minimal page header definition (mirrors bam_kernel.cu:91-96)
// ============================================================

struct q1u_pag_head {
    uint32_t nalloc;
    uint32_t watermark;
    uint32_t lfreespace;
};
static constexpr uint32_t Q1U_PAG_HDR_BYTES = 12;

// Decompress a single PFOR-encoded INT32 page using 128 cooperative threads.
// shared_buf must have at least 549 uint elements.
// Equivalent to decompress_page_128 in bam_kernel.cu:404-472.
static __device__ void q1u_decompress_pfor_128(
    char* page_ptr,
    int32_t* decomp_out,
    uint16_t comp_method,
    uint* shared_buf,
    uint32_t tid,
    uint32_t* out_nalloc)
{
    q1u_pag_head* hdr = (q1u_pag_head*)page_ptr;
    uint32_t nalloc = hdr->nalloc;
    uint32_t watermark = hdr->watermark;

    if (out_nalloc && tid == 0) *out_nalloc = nalloc;

    if (comp_method != 0) {
        uint32_t nblocks = watermark / 128;
        uint32_t ntiles = nblocks / 4;
        uint32_t remaining_blocks = nblocks - ntiles * 4;
        uint32_t* block_start = (uint32_t*)(page_ptr + Q1U_PAG_HDR_BYTES);
        uint32_t* data_ptr = block_start + (nblocks + 1);

        for (uint32_t t = 0; t < ntiles; t++) {
            int items[4];
            LoadBinPackTile<128, 4>(t, block_start, data_ptr,
                                    shared_buf, items, false, 512);
            for (int it = 0; it < 4; it++) {
                uint32_t idx = t * 512 + it * 128 + tid;
                if (idx < nalloc) {
                    decomp_out[idx] = items[it];
                }
            }
            __syncthreads();
        }

        for (uint32_t b = 0; b < remaining_blocks; b++) {
            uint32_t block_idx = ntiles * 4 + b;
            uint32_t* blk_data = data_ptr + block_start[block_idx];

            uint* rem_bws  = &shared_buf[0];
            uint* rem_offs = &shared_buf[4];
            if (tid < 4) {
                uint32_t mb_bw_packed = blk_data[1];
                uint32_t packed_off = (mb_bw_packed << 8)
                                    + (mb_bw_packed << 16)
                                    + (mb_bw_packed << 24);
                rem_bws[tid]  = (mb_bw_packed >> (tid << 3)) & 255;
                rem_offs[tid] = (packed_off >> (tid << 3)) & 255;
            }
            __syncthreads();

            uint32_t mb_idx = tid >> 5;
            uint32_t mb_pos = tid & 31;
            int val = decodeElement(tid, mb_idx, mb_pos,
                                    blk_data, rem_bws, rem_offs);

            uint32_t idx = block_idx * 128 + tid;
            if (idx < nalloc) {
                decomp_out[idx] = val;
            }
            __syncthreads();
        }
    } else {
        int32_t* src = (int32_t*)(page_ptr + Q1U_PAG_HDR_BYTES);
        for (uint32_t i = tid; i < nalloc; i += 128) {
            decomp_out[i] = src[i];
        }
        __syncthreads();
    }
}

// ============================================================
// Q1 Unified Kernel: 7-field IO + PFOR decompress + eval
//   All 7 fields are INT32+PFOR (CHAR(1) stored as ASCII int).
//   1 page_cache slot per block (slot reuse).
//   Sequential field-by-field IO + decompress, then evaluate.
// ============================================================

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_q1_unified_kernel(
    void*             ctrls_opaque,
    void*             pc_opaque,
    const char*       pc_base_addr,
    BAMq1UnifiedParams p)
{
    const uint32_t tid     = threadIdx.x;
    const uint32_t bid     = blockIdx.x;

    // Dynamic shared memory: PFOR only (549 uint)
    extern __shared__ __align__(8) uint8_t smem_q1u[];

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Per-thread local accumulators (register-allocated).
    // 6 groups × 6 aggs = 36 int64_t.
    constexpr int LA = 6;  // QTY, BASE_PRICE, DISC_PRICE, CHARGE, DISCOUNT, COUNT
    int64_t local_agg[Q1_NUM_GROUPS * LA];
    for (int k = 0; k < Q1_NUM_GROUPS * LA; k++) local_agg[k] = 0;

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;

        // ── Sequential IO + decompress: paged fields only ──
        // Fields 5 (L_RETURNFLAG) and 6 (L_LINESTATUS) may be pre-flattened
        // if their page layout differs from the driving field (L_SHIPDATE).
        constexpr int N_PAGED = 5;  // fields 0..4 always paged
        int32_t* field_data[N_PAGED];
        {
            uint* pfor_smem = (uint*)smem_q1u;
            for (int fi = 0; fi < N_PAGED; fi++) {
                if (tid == 0) {
                    uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                    uint32_t dev = global_pg % ndev;
                    uint64_t lba;
                    uint32_t nblk;
                    if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                        lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                        nblk = q1f_safe_io_nblocks(p.d_comp_sizes[fi][pg]);
                    } else {
                        uint64_t local_pg = global_pg / ndev;
                        lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                        nblk = p.blocks_per_page;
                    }
                    bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, bid, dev);
                }
                __syncthreads();

                char* page_ptr = (char*)pc_base_addr + (uint64_t)bid * p.page_size;
                field_data[fi] = p.d_scratch
                    + ((uint64_t)bid * 7 + fi) * p.scratch_stride;
                uint32_t nalloc_dummy;
                q1u_decompress_pfor_128(page_ptr, field_data[fi], p.comp_methods[fi],
                                        pfor_smem, tid, &nalloc_dummy);
                __syncthreads();
            }
        }

        // Also decompress CHAR fields from pages if not pre-flattened
        int32_t* rf_data = nullptr;
        int32_t* ls_data = nullptr;
        if (!p.d_flat_rf) {
            uint* pfor_smem = (uint*)smem_q1u;
            for (int fi = 5; fi <= 6; fi++) {
                if (tid == 0) {
                    uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                    uint32_t dev = global_pg % ndev;
                    uint64_t lba;
                    uint32_t nblk;
                    if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                        lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                        nblk = q1f_safe_io_nblocks(p.d_comp_sizes[fi][pg]);
                    } else {
                        uint64_t local_pg = global_pg / ndev;
                        lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                        nblk = p.blocks_per_page;
                    }
                    bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, bid, dev);
                }
                __syncthreads();

                char* page_ptr = (char*)pc_base_addr + (uint64_t)bid * p.page_size;
                int32_t* out = p.d_scratch
                    + ((uint64_t)bid * 7 + fi) * p.scratch_stride;
                uint32_t nalloc_dummy;
                q1u_decompress_pfor_128(page_ptr, out, p.comp_methods[fi],
                                        pfor_smem, tid, &nalloc_dummy);
                __syncthreads();
                if (fi == 5) rf_data = out;
                else         ls_data = out;
            }
        }

        // nrows from driving field (L_SHIPDATE) prefix sum
        uint64_t nrows = (pg == 0) ? p.d_prefix_sum[0]
                                   : p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1];
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

        // ── Evaluate ──
        for (uint64_t i = tid; i < nrows; i += blockDim.x) {
            int32_t shipdate = field_data[0][i];
            if (shipdate > 19980902) continue;

            // L_RETURNFLAG and L_LINESTATUS: from flat arrays or paged decompress
            int32_t returnflag, linestatus;
            if (p.d_flat_rf) {
                uint64_t global_row = row_base + i;
                returnflag = p.d_flat_rf[global_row];
                linestatus = p.d_flat_ls[global_row];
            } else {
                returnflag = rf_data[i];
                linestatus = ls_data[i];
            }

            int row;
            switch (returnflag) {
                case 'A': row = 0; break;
                case 'N': row = 1; break;
                case 'R': row = 2; break;
                default: continue;
            }
            int col = (linestatus == 'F') ? 0 : 1;
            int gid = row * 2 + col;

            int32_t quantity      = field_data[1][i];
            int32_t extendedprice = field_data[2][i];
            int32_t discount      = field_data[3][i];
            int32_t tax           = field_data[4][i];

            int64_t disc_price = (int64_t)extendedprice * (int64_t)(100 - discount);
            int64_t charge = disc_price * (int64_t)(100 + tax);

            int64_t* la = local_agg + gid * LA;
            la[0] += quantity;
            la[1] += extendedprice;
            la[2] += disc_price;
            la[3] += charge;
            la[4] += discount;
            la[5] += 1;
        }
    }

    // Flush per-thread local accumulators → global aggregates
    for (int g = 0; g < Q1_NUM_GROUPS; g++) {
        int64_t* la = local_agg + g * LA;
        if (la[5] == 0) continue;
        int64_t* ga = p.d_agg + g * Q1_NUM_AGGS;
        atomicAdd((unsigned long long*)&ga[Q1_SUM_QTY],
                  (unsigned long long)la[0]);
        atomicAdd((unsigned long long*)&ga[Q1_SUM_BASE_PRICE],
                  (unsigned long long)la[1]);
        atomicAdd((unsigned long long*)&ga[Q1_SUM_DISC_PRICE],
                  (unsigned long long)la[2]);
        {
            unsigned long long old_lo = atomicAdd(
                (unsigned long long*)&ga[Q1_SUM_CHARGE],
                (unsigned long long)la[3]);
            if (old_lo + (unsigned long long)la[3] < old_lo) {
                atomicAdd((unsigned long long*)&ga[Q1_SUM_CHARGE_HI], 1ULL);
            }
        }
        atomicAdd((unsigned long long*)&ga[Q1_SUM_DISCOUNT],
                  (unsigned long long)la[4]);
        atomicAdd((unsigned long long*)&ga[Q1_COUNT],
                  (unsigned long long)la[5]);
    }
}

// ============================================================
// Q1 Unified IO Context
// ============================================================

struct BAMq1UnifiedContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_q1_fused_io_ctx_t bam_q1_unified_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMq1UnifiedContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 1 slot per block: sequential IO with slot reuse
    const uint32_t num_slots = num_blocks;

    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    return static_cast<bam_q1_fused_io_ctx_t>(ctx);
}

void bam_q1_unified_destroy(bam_q1_fused_io_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMq1UnifiedContext*>(ctx_handle);
    if (!ctx) return;

    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ============================================================
// Q1 Unified Launch Helper
// ============================================================

void bam_q1_unified_run(
    bam_q1_fused_io_ctx_t ctx_handle,
    const BAMq1UnifiedParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq1UnifiedContext*>(ctx_handle);
    // Shared memory: PFOR only (549 uint ≈ 2.2 KB)
    size_t smem = 549 * sizeof(uint);

    constexpr uint32_t TPB = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kfn = bam_q1_unified_kernel<PS>;
        BAM_Q1_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<p.num_blocks, TPB, smem, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr, p);
    });
    BAM_Q1_CUDA_CHECK(cudaGetLastError());
}
