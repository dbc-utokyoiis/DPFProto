// ============================================================
// Q3 Fused IO + Decomp + Filter kernel (nvCOMPdx LZ4 PAR-32K).
//
// C_MKTSEGMENT: IO + decomp + BUILDING filter + custkey hash set insert
//
// Block-per-page with cooperative decomp + double-buffered IO.
//   128 threads/block (4 warps), all warps cooperate on decompress.
//   2 page_cache slots per block for IO/filter overlap.
//   Block-stride loop over all pages, __syncthreads().
//
// BaM IO is called via bam_io_device.cuh (compiled C++11, device-linked).
// Compiled as CUDA C++17 for nvCOMPdx.
// ============================================================

#include "bam_q3_kernel.cuh"
#include "bam_io_device.cuh"
#include "../common/fsst_page.h"

#include <nvcompdx.hpp>

#include "tpch/page_size_dispatch.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BAM_Q3_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ── Constants ──
static constexpr uint32_t Q3F_CHUNK_SZ = 32768u;

// PRP2 workaround: NVMe controller interprets PRP2 as direct address when
// transfer spans exactly 2 controller pages (9-16 blocks), but BaM sets it
// as a PRP list pointer.  Bump to 17 blocks (≥3 pages) so PRP2 is treated
// as a PRP list.
#define Q3F_NVM_CTRL_PAGE_BLOCKS 8
__device__ __forceinline__ uint32_t q3f_fix_nblk(uint32_t nblk) {
    if (nblk > Q3F_NVM_CTRL_PAGE_BLOCKS && nblk <= Q3F_NVM_CTRL_PAGE_BLOCKS * 2)
        nblk = Q3F_NVM_CTRL_PAGE_BLOCKS * 2 + 1;
    return nblk;
}

// "BUILDING" as uint64_t in little-endian: B=0x42 U=0x55 I=0x49 L=0x4C D=0x44 I=0x49 N=0x4E G=0x47
static constexpr uint64_t Q3F_BUILDING_U64 = 0x474E49444C495542ULL;

// Hash function (same as q3_scan.cu)
__device__ __forceinline__ uint32_t q3f_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

// ============================================================
// Kernel: C_MKTSEGMENT — Fused IO + Decomp + BUILDING Filter
// Handles both LZ4PAR (comp_method=9) and uncompressed (comp_method=0).
// ============================================================

template<unsigned int PAGE_SIZE_CONST>
__global__ void bam_q3_fused_mktseg_kernel(
    void*           ctrls_opaque,
    void*           pc_opaque,
    const char*     pc_base_addr,
    char*           d_decomp_buf,
    BAMq3FusedMktsegParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<Q3F_CHUNK_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t bid     = blockIdx.x;

    extern __shared__ __align__(8) uint8_t smem_q3mk[];
    constexpr size_t warp_smem_size = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem_q3mk + warp_id * warp_smem_size;

    constexpr uint32_t n_chunks       = PAGE_SIZE_CONST / Q3F_CHUNK_SZ;
    constexpr uint32_t hdr_bytes      = n_chunks * sizeof(uint32_t);
    constexpr uint32_t chunks_per_warp = n_chunks / 4;
    constexpr uint32_t blocks_per_page = PAGE_SIZE_CONST / 512;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;
    const bool is_compressed = (p.comp_method != 0);

    // Helper: compute multi-device LBA + nblocks for page pg
    auto compute_io = [&](uint64_t pg_idx, uint32_t& out_dev, uint32_t& out_nblk) -> uint64_t {
        uint64_t global_pg = p.field_start_page_id + pg_idx;
        out_dev = global_pg % ndev;
        if (is_compressed) {
            out_nblk = q3f_fix_nblk((p.d_comp_sizes[pg_idx] + 511) / 512);
            return p.partition_start_lbas[out_dev] + p.d_comp_offsets[pg_idx] / 512;
        } else {
            uint64_t local_pg = global_pg / ndev;
            out_nblk = blocks_per_page;
            return p.partition_start_lbas[out_dev] + local_pg * blocks_per_page;
        }
    };

    // Double-buffer: 2 page_cache slots per block
    const uint32_t slot_a = bid * 2;
    const uint32_t slot_b = bid * 2 + 1;
    uint32_t cur_slot = slot_a;
    uint32_t nxt_slot = slot_b;

    // Prefetch first page
    uint64_t pg = (uint64_t)bid;
    if (pg < p.npages) {
        if (tid == 0) {
            uint32_t dev, nblk;
            uint64_t lba = compute_io(pg, dev, nblk);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, cur_slot, dev);
        }
        __syncthreads();
    }

    for (pg = (uint64_t)bid; pg < p.npages; pg += gridDim.x) {
        uint64_t next_pg = pg + gridDim.x;
        bool has_next = (next_pg < p.npages);

        // ── Phase 2: Decompress (or direct copy for uncompressed) ──
        const char* page_ptr;
        if (is_compressed) {
            // 4-warp cooperative PAR-32K decompress
            const uint8_t* comp_page = (const uint8_t*)(
                pc_base_addr + (unsigned long long)cur_slot * p.page_size);
            uint8_t* decomp_page = (uint8_t*)(
                d_decomp_buf + (unsigned long long)cur_slot * p.page_size);

            const uint32_t* chunk_hdr = (const uint32_t*)comp_page;
            auto decompressor = lz4_decomp_t();

            uint32_t first_chunk = warp_id * chunks_per_warp;
            uint32_t data_offset = hdr_bytes;
            for (uint32_t i = 0; i < first_chunk; i++)
                data_offset += chunk_hdr[i];

            for (uint32_t i = 0; i < chunks_per_warp; i++) {
                uint32_t chunk_idx = first_chunk + i;
                uint32_t comp_len = chunk_hdr[chunk_idx];
                size_t decomp_size = 0;
                decompressor.execute(
                    comp_page + data_offset,
                    decomp_page + (uint64_t)chunk_idx * Q3F_CHUNK_SZ,
                    comp_len, &decomp_size, my_smem, nullptr);
                data_offset += comp_len;
            }
            page_ptr = (const char*)decomp_page;
        } else {
            // Uncompressed: page_cache slot already contains the page
            page_ptr = pc_base_addr + (unsigned long long)cur_slot * p.page_size;
        }
        __syncthreads();

        // ── Phase 1: Prefetch next page (tid==0, overlaps with filter) ──
        if (has_next && tid == 0) {
            uint32_t dev, nblk;
            uint64_t lba = compute_io(next_pg, dev, nblk);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, nxt_slot, dev);
        }

        // ── Phase 3: CHAR(10) BUILDING filter + hash set insert (128 threads) ──
        {
            uint32_t nalloc = *(const uint32_t*)page_ptr;
            uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

            for (uint32_t s = tid; s < nalloc; s += 128) {
                const char* rec = page_ptr + 12 + p.padded_len * s;
                // Compare "BUILDING" as 8 bytes LE (4B aligned reads)
                uint32_t lo = *reinterpret_cast<const uint32_t*>(rec);
                uint32_t hi = *reinterpret_cast<const uint32_t*>(rec + 4);
                uint64_t val8 = ((uint64_t)hi << 32) | lo;
                if (p.num_segments == 0) {
                    if (val8 != Q3F_BUILDING_U64) continue;
                } else {
                    bool match = false;
                    for (uint32_t seg = 0; seg < p.num_segments; seg++) {
                        if (val8 == p.segment_values[seg]) { match = true; break; }
                    }
                    if (!match) continue;
                }

                uint64_t row_id = row_base + s;
                uint64_t custkey = p.d_c_custkey_flat[row_id];
                // Hash set insert (open addressing, 0xFF sentinel)
                uint32_t slot = q3f_hash64(custkey) & p.custkey_set_mask;
                while (true) {
                    uint64_t prev = atomicCAS(
                        (unsigned long long*)&p.d_custkey_set[slot],
                        0xFFFFFFFFFFFFFFFFULL, (unsigned long long)custkey);
                    if (prev == 0xFFFFFFFFFFFFFFFFULL || prev == custkey) break;
                    slot = (slot + 1) & p.custkey_set_mask;
                }
            }
        }
        __syncthreads();

        // Swap double-buffer slots
        uint32_t tmp = cur_slot; cur_slot = nxt_slot; nxt_slot = tmp;
    }
}

// ============================================================
// Fused IO Context
// ============================================================

struct BAMq3FusedIOContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_q3_fused_io_ctx_t bam_q3_fused_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMq3FusedIOContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 2 slots per block (double-buffered IO)
    const uint32_t num_slots = num_blocks * 2;

    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    size_t decomp_size = (size_t)num_slots * page_size;
    BAM_Q3_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_q3_fused_io_ctx_t>(ctx);
}

void bam_q3_fused_io_destroy(bam_q3_fused_io_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMq3FusedIOContext*>(ctx_handle);
    if (!ctx) return;

    cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ============================================================
// Launch helper
// ============================================================

using q3_decomp_type_t = decltype(
    nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
    nvcompdx::DataType<nvcompdx::datatype::uint8>() +
    nvcompdx::Direction<nvcompdx::direction::decompress>() +
    nvcompdx::MaxUncompChunkSize<Q3F_CHUNK_SZ>() +
    nvcompdx::Warp() +
    nvcompdx::SM<800>());

static size_t q3f_smem_size() {
    return q3_decomp_type_t().shmem_size_group() * 4;  // 4 warps
}

void bam_q3_fused_mktseg_async(
    bam_q3_fused_io_ctx_t ctx_handle,
    const BAMq3FusedMktsegParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq3FusedIOContext*>(ctx_handle);
    size_t smem = q3f_smem_size();
    constexpr uint32_t TPB = 128;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        auto kfn = bam_q3_fused_mktseg_kernel<PS>;
        BAM_Q3_CUDA_CHECK(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        kfn<<<p.num_blocks, TPB, smem, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });
    BAM_Q3_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Q3 FSST Fused IO + Decomp + BUILDING Filter Kernel
//
// Same double-buffered IO pattern as Q16 FSST kernels.
// FSST decode → register → compare first 8 bytes with "BUILDING"
// → if match, insert custkey into hash set.
// ============================================================

// "BUILDING" as uint64_t LE
static constexpr uint64_t Q3F_BUILDING_U64_FSST = 0x474E49444C495542ULL;

// ── Helper: cooperative load of FSST symbol table into smem ──
__device__ __forceinline__ void q3f_load_symtab(
    uint8_t*  s_sym_len,
    uint64_t* s_sym_val,
    const char* symtab_base,
    uint32_t tid)
{
    for (uint32_t i = tid; i < 256; i += blockDim.x)
        s_sym_len[i] = ((const uint8_t*)symtab_base)[i];
    for (uint32_t i = tid; i < 255; i += blockDim.x)
        memcpy(&s_sym_val[i], symtab_base + FSST_SYMTAB_LEN_BYTES + i * 8, 8);
}

// ── Helper: cooperative load of comp block data into smem ──
__device__ __forceinline__ void q3f_load_cb_data(
    uint8_t* s_cb_data,
    const char* cb_base,
    uint32_t cb_data_size,
    uint32_t tid)
{
    uint32_t num_words = (cb_data_size + 3) / 4;
    const uint32_t* src = (const uint32_t*)cb_base;
    uint32_t* dst = (uint32_t*)s_cb_data;
    for (uint32_t i = tid; i < num_words; i += blockDim.x)
        dst[i] = src[i];
}

__global__ void bam_q3_fused_mktseg_fsst_kernel(
    void*           ctrls_opaque,
    void*           pc_opaque,
    const char*     pc_base_addr,
    BAMq3FsstMktsegParams p)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    extern __shared__ __align__(8) char smem_q3fsst[];

    uint8_t*  s_sym_len = (uint8_t*)  smem_q3fsst;
    uint64_t* s_sym_val = (uint64_t*)(smem_q3fsst + 256);
    uint8_t*  s_cb_data = (uint8_t*)(smem_q3fsst + FSST_SMEM_SYMTAB_OFFSET);

    const uint32_t slot_a = bid * 2;
    const uint32_t slot_b = bid * 2 + 1;
    uint32_t cur_slot = slot_a;
    uint32_t nxt_slot = slot_b;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    auto compute_lba_dev = [&](uint64_t pg_idx, uint32_t& out_dev) -> uint64_t {
        uint64_t global_pg = p.field_start_page_id + pg_idx;
        out_dev = global_pg % ndev;
        return p.partition_start_lbas[out_dev] + p.d_comp_offsets[pg_idx] / 512;
    };

    // Prefetch first page
    uint64_t pg = (uint64_t)bid;
    if (pg < p.npages) {
        if (tid == 0) {
            uint32_t dev;
            uint64_t lba = compute_lba_dev(pg, dev);
            uint32_t nblk = q3f_fix_nblk((p.d_comp_sizes[pg] + 511) / 512);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, cur_slot, dev);
        }
        __syncthreads();
    }

    for (pg = (uint64_t)bid; pg < p.npages; pg += gridDim.x) {
        uint64_t next_pg = pg + gridDim.x;
        bool has_next = (next_pg < p.npages);

        const char* comp_page = pc_base_addr
            + (unsigned long long)cur_slot * p.page_size;

        // Parse FSST page header
        uint32_t watermark = *reinterpret_cast<const uint32_t*>(comp_page + 4);
        uint32_t n_comp_blocks = *reinterpret_cast<const uint32_t*>(
            comp_page + FSST_PAGE_NCB_OFFSET);
        const FsstCompBlockDirEntry* dir =
            reinterpret_cast<const FsstCompBlockDirEntry*>(
                comp_page + FSST_PAGE_NCB_OFFSET + 4);
        const char* symtab_base = comp_page + FSST_PAGE_NCB_OFFSET + 4
            + n_comp_blocks * sizeof(FsstCompBlockDirEntry);
        // For FSST_ROWID pages, string data ends before the PFOR64 section
        uint32_t fsst_data_end = fsst_page_string_data_end(
            comp_page, watermark, n_comp_blocks);

        // Load symbol table to smem
        q3f_load_symtab(s_sym_len, s_sym_val, symtab_base, tid);
        __syncthreads();

        // Process each comp block
        uint64_t cb_rec_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

        for (uint32_t cb = 0; cb < n_comp_blocks; cb++) {
            uint32_t cb_offset = dir[cb].offset;
            uint32_t cb_nrecs  = dir[cb].nrecs;
            uint32_t cb_end = (cb + 1 < n_comp_blocks)
                ? dir[cb + 1].offset : fsst_data_end;
            uint32_t cb_data_size = cb_end - cb_offset;

            const char* cb_base = comp_page + cb_offset;

            q3f_load_cb_data(s_cb_data, cb_base, cb_data_size, tid);
            __syncthreads();

            // Prefetch next page during last comp block
            if (cb == n_comp_blocks - 1 && has_next && tid == 0) {
                uint32_t dev;
                uint64_t lba = compute_lba_dev(next_pg, dev);
                uint32_t nblk = q3f_fix_nblk((p.d_comp_sizes[next_pg] + 511) / 512);
                bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk,
                                        nxt_slot, dev);
            }

            // FSST decode → "BUILDING" check → hash set insert
            const uint16_t* offset_table = (const uint16_t*)s_cb_data;
            const uint8_t* comp_data = s_cb_data
                + (cb_nrecs + 1) * sizeof(uint16_t);

            for (uint32_t r = tid; r < cb_nrecs; r += blockDim.x) {
                uint16_t comp_start = offset_table[r];
                uint16_t comp_len = offset_table[r + 1] - comp_start;
                const uint8_t* comp_ptr = comp_data + comp_start;

                // Decode first 8 bytes to register
                uint64_t decoded8 = 0;
                uint32_t posIn = 0, posOut = 0;

                while (posIn < comp_len && posOut < 8) {
                    uint8_t code = comp_ptr[posIn++];
                    uint64_t sym_val;
                    uint8_t sym_len;

                    if (code < 255) {
                        sym_len = s_sym_len[code];
                        sym_val = s_sym_val[code];
                    } else {
                        sym_val = (uint64_t)comp_ptr[posIn++];
                        sym_len = 1;
                    }

                    for (uint8_t j = 0; j < sym_len && posOut < 8; j++) {
                        decoded8 |= (sym_val & 0xFF) << (posOut * 8);
                        sym_val >>= 8;
                        posOut++;
                    }
                }

                if (p.num_segments == 0) {
                    if (decoded8 != Q3F_BUILDING_U64_FSST) continue;
                } else {
                    bool match = false;
                    for (uint32_t seg = 0; seg < p.num_segments; seg++) {
                        if (decoded8 == p.segment_values[seg]) { match = true; break; }
                    }
                    if (!match) continue;
                }

                // Match — insert custkey into hash set
                uint64_t row_id = cb_rec_base + r;
                uint64_t custkey = p.d_c_custkey_flat[row_id];
                uint32_t slot = q3f_hash64(custkey) & p.custkey_set_mask;
                while (true) {
                    uint64_t prev = atomicCAS(
                        (unsigned long long*)&p.d_custkey_set[slot],
                        0xFFFFFFFFFFFFFFFFULL, (unsigned long long)custkey);
                    if (prev == 0xFFFFFFFFFFFFFFFFULL || prev == custkey) break;
                    slot = (slot + 1) & p.custkey_set_mask;
                }
            }
            __syncthreads();

            cb_rec_base += cb_nrecs;
        }

        // Swap double-buffer slots
        uint32_t tmp = cur_slot; cur_slot = nxt_slot; nxt_slot = tmp;
    }
}

// ============================================================
// FSST Fused IO Context (no decomp_buf — decode to registers)
// ============================================================

struct BAMq3FsstIOContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_q3_fsst_io_ctx_t bam_q3_fsst_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMq3FsstIOContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * 2;
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    return static_cast<bam_q3_fsst_io_ctx_t>(ctx);
}

void bam_q3_fsst_io_destroy(bam_q3_fsst_io_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMq3FsstIOContext*>(ctx_handle);
    if (!ctx) return;
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

static size_t q3f_fsst_smem_size(uint32_t page_size) {
    size_t cb_max = (page_size < 98304) ? page_size : 98304;
    return FSST_SMEM_SYMTAB_OFFSET + cb_max;
}

void bam_q3_fused_mktseg_fsst_async(
    bam_q3_fsst_io_ctx_t ctx_handle,
    const BAMq3FsstMktsegParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq3FsstIOContext*>(ctx_handle);
    size_t smem = q3f_fsst_smem_size(p.page_size);
    constexpr uint32_t TPB = 128;

    auto kfn = bam_q3_fused_mktseg_fsst_kernel;
    BAM_Q3_CUDA_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kfn<<<p.num_blocks, TPB, smem, stream>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr, p);
    BAM_Q3_CUDA_CHECK(cudaGetLastError());
}
