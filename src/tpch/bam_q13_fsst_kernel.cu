// ============================================================
// Q13 FSST Fused IO + Decomp + KMP Scan Kernel
//
// Persistent kernel for Q13 O_COMMENT (FSST compressed):
//   IO + FSST decomp + inline KMP → qualifying custkeys
//
// Per page: BaM read → parse FSST header → load symtab →
//   iterate comp blocks → FSST decode + inline KMP
// If NOT LIKE (qualifying): write custkey to d_o_aggr_custkey[row_id]
// If LIKE (non-qualifying): write UINT64_MAX
//
// Double-buffered IO: 2 page_cache slots per block.
// Block-stride loop over all pages.
//
// Based on bam_q16_fsst_kernel.cu (S_COMMENT kernel).
// ============================================================

#include "bam_q13_fsst_kernel.cuh"
#include "bam_io_device.cuh"
#include "../common/fsst_page.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BAM_Q13F_CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                      \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// NVMe PRP boundary workaround (same as Q16)
#define Q13FF_NVM_CTRL_PAGE_BLOCKS 8

__device__ __forceinline__ uint32_t q13ff_safe_io_nblocks(uint32_t comp_bytes) {
    uint32_t nblk = (comp_bytes + 511) / 512;
    if (nblk > Q13FF_NVM_CTRL_PAGE_BLOCKS && nblk <= Q13FF_NVM_CTRL_PAGE_BLOCKS * 2)
        nblk = Q13FF_NVM_CTRL_PAGE_BLOCKS * 2 + 1;
    return nblk;
}

// ── Helper: cooperative load of FSST symbol table into smem ──
__device__ __forceinline__ void q13ff_load_symtab(
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

// (cb_data is now read directly from page cache — no smem copy needed)

// ============================================================
// Kernel: O_COMMENT — FSST Fused IO + Decomp + KMP Scan
//
// Per page: parse FSST header → load symtab → iterate comp blocks
// Per comp block: load cb_data → FSST decode + inline KMP
// IO overlap: prefetch next page during last comp block
// ============================================================

__global__ void bam_q13_fsst_o_comment_kernel(
    void*           ctrls_opaque,
    void*           pc_opaque,
    const char*     pc_base_addr,
    BAMq13FsstParams p)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    extern __shared__ __align__(8) char smem[];

    uint8_t*  s_sym_len    = (uint8_t*)  smem;
    uint64_t* s_sym_val    = (uint64_t*)(smem + 256);
    // KMP tables (within symtab padding region: 2296..2495)
    char*     s_kmp_pat     = smem + 2296;
    int*      s_kmp_next    = (int*)(smem + 2328);
    int*      s_kmp_offsets = (int*)(smem + 2456);
    int*      s_kmp_lengths = (int*)(smem + 2472);

    // Double-buffer slots
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

    // Load KMP tables to smem (once, small)
    if (tid < (uint32_t)p.num_patterns) {
        s_kmp_offsets[tid] = p.d_pattern_offsets[tid];
        s_kmp_lengths[tid] = p.d_pattern_lengths[tid];
    } else if (tid < 4) {
        s_kmp_offsets[tid] = 0;
        s_kmp_lengths[tid] = 0;
    }
    {
        int total_pat_chars = 0;
        for (int pi = 0; pi < p.num_patterns; pi++)
            total_pat_chars += p.d_pattern_lengths[pi];
        if (tid < (uint32_t)total_pat_chars) {
            s_kmp_pat[tid] = p.d_patterns[tid];
            s_kmp_next[tid] = p.d_next[tid];
        } else if (tid < 32) {
            s_kmp_pat[tid] = 0;
            s_kmp_next[tid] = 0;
        }
    }

    // Prefetch first page
    uint64_t pg = (uint64_t)bid;
    if (pg < p.npages) {
        if (tid == 0) {
            uint32_t dev;
            uint64_t lba = compute_lba_dev(pg, dev);
            uint32_t nblk = q13ff_safe_io_nblocks(p.d_comp_sizes[pg]);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk, cur_slot, dev);
        }
        __syncthreads();
    }

    for (pg = (uint64_t)bid; pg < p.npages; pg += gridDim.x) {
        uint64_t next_pg = pg + gridDim.x;
        bool has_next = (next_pg < p.npages);

        long long t0 = clock64();

        const char* comp_page = pc_base_addr
            + (unsigned long long)cur_slot * p.page_size;

        // Parse FSST page header
        uint32_t watermark = *reinterpret_cast<const uint32_t*>(comp_page + 4);
        uint32_t n_comp_blocks = *reinterpret_cast<const uint32_t*>(
            comp_page + FSST_PAGE_NCB_OFFSET);

        // Page-level validity check: watermark and n_comp_blocks must be sane
        uint32_t comp_sz = p.d_comp_sizes[pg];
        bool page_valid = (n_comp_blocks <= 1024 && watermark <= comp_sz);
        if (!page_valid)
            n_comp_blocks = 0;

        const FsstCompBlockDirEntry* dir =
            reinterpret_cast<const FsstCompBlockDirEntry*>(
                comp_page + FSST_PAGE_NCB_OFFSET + 4);
        const char* symtab_base = comp_page + FSST_PAGE_NCB_OFFSET + 4
            + n_comp_blocks * sizeof(FsstCompBlockDirEntry);
        // For FSST_ROWID pages, string data ends before the PFOR64 section
        uint32_t fsst_data_end = fsst_page_string_data_end(
            comp_page, watermark, n_comp_blocks);

        // Load symbol table to smem
        q13ff_load_symtab(s_sym_len, s_sym_val, symtab_base, tid);
        __syncthreads();

        long long t1 = clock64();

        // Process each comp block
        uint64_t cb_rec_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        uint64_t page_qualifying = 0;

        for (uint32_t cb = 0; cb < n_comp_blocks; cb++) {
            uint32_t cb_offset = dir[cb].offset;
            uint32_t cb_nrecs  = dir[cb].nrecs;
            uint32_t cb_end = (cb + 1 < n_comp_blocks)
                ? dir[cb + 1].offset : fsst_data_end;
            uint32_t cb_data_size = cb_end - cb_offset;

            // Bounds check: skip invalid comp blocks
            const uint32_t cb_max = (p.page_size < 98304) ? p.page_size : 98304;
            if (cb_data_size > cb_max || cb_nrecs == 0 || cb_nrecs > 65536
                || cb_offset >= comp_sz || cb_end > comp_sz) {
                cb_rec_base += cb_nrecs;
                continue;
            }

            // Prefetch next page during last comp block (IO/compute overlap)
            if (cb == n_comp_blocks - 1 && has_next && tid == 0) {
                uint32_t dev;
                uint64_t lba = compute_lba_dev(next_pg, dev);
                uint32_t nblk = q13ff_safe_io_nblocks(p.d_comp_sizes[next_pg]);
                bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk,
                                        nxt_slot, dev);
            }

            // FSST decode + inline KMP scan (read directly from page cache)
            const uint8_t* cb_data = (const uint8_t*)(comp_page + cb_offset);
            const uint16_t* offset_table = (const uint16_t*)cb_data;
            const uint8_t* comp_data = cb_data
                + (cb_nrecs + 1) * sizeof(uint16_t);

            for (uint32_t r = tid; r < cb_nrecs; r += blockDim.x) {
                uint16_t comp_start = offset_table[r];
                uint16_t comp_len = offset_table[r + 1] - comp_start;
                const uint8_t* comp_ptr = comp_data + comp_start;

                // Inline KMP state
                int current_pat = 0;
                int l = 0;
                int kp_offset = s_kmp_offsets[0];
                int kp_len = s_kmp_lengths[0];
                bool done = false;

                uint32_t posIn = 0;
                while (posIn < comp_len && !done) {
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

                    for (uint8_t j = 0; j < sym_len; j++) {
                        char c = (char)(sym_val & 0xFF);
                        sym_val >>= 8;

                        while (l > 0 && s_kmp_pat[kp_offset + l] != c)
                            l = s_kmp_next[kp_offset + l - 1];
                        if (s_kmp_pat[kp_offset + l] == c) l++;
                        if (l == kp_len) {
                            current_pat++;
                            l = 0;
                            if (current_pat >= p.num_patterns) {
                                done = true; break;
                            }
                            kp_offset = s_kmp_offsets[current_pat];
                            kp_len = s_kmp_lengths[current_pat];
                        }
                    }
                }

                uint64_t row_id = cb_rec_base + r;

                // Bounds check: row_id must be within total records
                if (row_id >= p.nrecs_total) continue;

                if (!done) {
                    // Qualifying (NOT LIKE): write custkey
                    p.d_o_aggr_custkey[row_id] = p.d_o_custkey_flat[row_id];
                    page_qualifying++;
                } else {
                    // Non-qualifying (LIKE): mark as excluded
                    p.d_o_aggr_custkey[row_id] = UINT64_MAX;
                }
            }

            cb_rec_base += cb_nrecs;
        }

        // Fallback prefetch for n_comp_blocks==0 (no in-loop prefetch occurred)
        if (n_comp_blocks == 0 && has_next && tid == 0) {
            uint32_t dev;
            uint64_t lba = compute_lba_dev(next_pg, dev);
            uint32_t nblk = q13ff_safe_io_nblocks(p.d_comp_sizes[next_pg]);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk,
                                    nxt_slot, dev);
        }
        __syncthreads();

        long long t2 = clock64();

        // Accumulate qualifying count (per-thread atomic)
        if (page_qualifying > 0) {
            atomicAdd((unsigned long long*)p.d_count,
                      (unsigned long long)page_qualifying);
        }

        // Phase timing
        if (tid == 0 && p.d_phase_cycles) {
            atomicAdd((unsigned long long*)&p.d_phase_cycles[0], 0ULL);
            atomicAdd((unsigned long long*)&p.d_phase_cycles[1],
                      (unsigned long long)(t1 - t0));
            atomicAdd((unsigned long long*)&p.d_phase_cycles[2],
                      (unsigned long long)(t2 - t1));
        }

        // Swap double-buffer slots
        uint32_t tmp = cur_slot; cur_slot = nxt_slot; nxt_slot = tmp;
    }
}

// ============================================================
// FSST Fused IO Context — no decomp_buf (FSST in smem+registers)
// ============================================================

struct BAMq13FsstIOContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_q13_fsst_io_ctx_t bam_q13_fsst_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMq13FsstIOContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 2 slots per block (double-buffered IO) — NO decomp_buf
    const uint32_t num_slots = num_blocks * 2;

    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    return static_cast<bam_q13_fsst_io_ctx_t>(ctx);
}

void bam_q13_fsst_io_destroy(bam_q13_fsst_io_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMq13FsstIOContext*>(ctx_handle);
    if (!ctx) return;

    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ── Shared memory size ──
// symtab only (2496B) — comp block data read directly from page cache
static size_t q13ff_smem_size(uint32_t /*page_size*/) {
    return FSST_SMEM_SYMTAB_OFFSET;  // 2496 bytes
}

// ── Launch helper ──

void bam_q13_fsst_o_comment_async(
    bam_q13_fsst_io_ctx_t ctx_handle,
    const BAMq13FsstParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq13FsstIOContext*>(ctx_handle);
    size_t smem = q13ff_smem_size(p.page_size);
    constexpr uint32_t TPB = 128;

    auto kfn = bam_q13_fsst_o_comment_kernel;
    BAM_Q13F_CUDA_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    kfn<<<p.num_blocks, TPB, smem, stream>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr, p);

    BAM_Q13F_CUDA_CHECK(cudaGetLastError());
}
