// ============================================================
// Q16 FSST Fused IO + Decomp + Filter kernels.
//
// Three persistent kernels for Q16 FSST-compressed columns:
//   1. S_COMMENT: IO + FSST decomp + inline KMP → excluded suppkeys
//   2. P_BRAND:   IO + FSST decomp + brand_id extraction
//   3. P_TYPE:    IO + FSST decomp + FNV-1a dict → type_ids
//
// No decomp_buf: FSST decode in smem + registers.
// Double-buffered IO: 2 page_cache slots per block.
// Block-stride loop over all pages, __syncthreads().
//
// BaM IO is called via bam_io_device.cuh (compiled C++11, device-linked).
// Compiled as CUDA C++17 (no nvCOMPdx dependency).
// ============================================================

#include "bam_q16_fsst_kernel.cuh"
#include "bam_io_device.cuh"
#include "../common/fsst_page.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BAM_Q16F_CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                      \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ============================================================
// Constants
// ============================================================

// NVMe PRP boundary workaround
#define Q16FF_NVM_CTRL_PAGE_BLOCKS 8

__device__ __forceinline__ uint32_t q16ff_safe_io_nblocks(uint32_t comp_bytes) {
    uint32_t nblk = (comp_bytes + 511) / 512;
    if (nblk > Q16FF_NVM_CTRL_PAGE_BLOCKS && nblk <= Q16FF_NVM_CTRL_PAGE_BLOCKS * 2)
        nblk = Q16FF_NVM_CTRL_PAGE_BLOCKS * 2 + 1;
    return nblk;
}

static constexpr uint32_t Q16FF_TYPE_DICT_CAP  = 512;
static constexpr uint32_t Q16FF_TYPE_DICT_MASK = Q16FF_TYPE_DICT_CAP - 1;
static constexpr uint32_t Q16FF_TYPE_MAX_LEN   = 32;

// ── FNV-1a 64-bit hash ──
__device__ __forceinline__ uint64_t q16ff_fnv1a64(const char *s, uint16_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (uint16_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ============================================================
// Shared memory layout for FSST fused kernels:
//   [0..255]      : s_sym_len[256]       (uint8_t)
//   [256..2295]   : s_sym_val[255]       (uint64_t, 8B each)
//   [2296..]      : kernel-specific data (KMP tables / temp)
//   [FSST_SMEM_SYMTAB_OFFSET..] : comp block data  (cb_data)
//
// FSST_SMEM_SYMTAB_OFFSET = 2496 (padded for alignment)
// ============================================================

// ── Helper: cooperative load of FSST symbol table into smem ──
__device__ __forceinline__ void q16ff_load_symtab(
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
// Kernel 1: S_COMMENT — FSST Fused IO + Decomp + KMP Scan
//
// Per page: parse FSST header → load symtab → iterate comp blocks
// Per comp block: load cb_data → FSST decode + inline KMP
// IO overlap: prefetch next page during last comp block
// ============================================================

__global__ void bam_q16_fsst_s_comment_kernel(
    void*           ctrls_opaque,
    void*           pc_opaque,
    const char*     pc_base_addr,
    BAMq16FsstSCommentParams p)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    extern __shared__ __align__(8) char smem_fsc[];

    uint8_t*  s_sym_len  = (uint8_t*)  smem_fsc;
    uint64_t* s_sym_val  = (uint64_t*)(smem_fsc + 256);
    // KMP tables after symtab (at offset 2296)
    char*     s_kmp_pat     = smem_fsc + 2296;
    int*      s_kmp_next    = (int*)(smem_fsc + 2328);
    int*      s_kmp_offsets = (int*)(smem_fsc + 2456);
    int*      s_kmp_lengths = (int*)(smem_fsc + 2472);

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

    // Prefetch first page
    uint64_t pg = (uint64_t)bid;
    if (pg < p.npages) {
        if (tid == 0) {
            uint32_t dev;
            uint64_t lba = compute_lba_dev(pg, dev);
            uint32_t nblk = q16ff_safe_io_nblocks(p.d_comp_sizes[pg]);
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

        // Page-level validity check
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
        q16ff_load_symtab(s_sym_len, s_sym_val, symtab_base, tid);

        // Load KMP tables to smem (once, small)
        if (pg == (uint64_t)bid) {  // first iteration only
            if (tid < 32) {
                s_kmp_pat[tid] = (tid < (uint32_t)(p.num_patterns * 16))
                    ? p.d_patterns[tid] : 0;
                s_kmp_next[tid] = (tid < (uint32_t)(p.num_patterns * 16))
                    ? p.d_next[tid] : 0;
            }
            if (tid < 4) {
                s_kmp_offsets[tid] = (tid < (uint32_t)p.num_patterns)
                    ? p.d_pattern_offsets[tid] : 0;
                s_kmp_lengths[tid] = (tid < (uint32_t)p.num_patterns)
                    ? p.d_pattern_lengths[tid] : 0;
            }
        }
        __syncthreads();

        long long t1 = clock64();

        // Process each comp block
        uint64_t cb_rec_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

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
                uint32_t nblk = q16ff_safe_io_nblocks(p.d_comp_sizes[next_pg]);
                bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk,
                                        nxt_slot, dev);
            }

            // FSST decode + inline KMP scan (read directly from page cache)
            const uint8_t* cb_data_ptr = (const uint8_t*)(comp_page + cb_offset);
            const uint16_t* offset_table = (const uint16_t*)cb_data_ptr;
            const uint8_t* comp_data = cb_data_ptr
                + (cb_nrecs + 1) * sizeof(uint16_t);

            for (uint32_t r = tid; r < cb_nrecs; r += blockDim.x) {
                uint16_t comp_start = offset_table[r];
                uint16_t comp_len = offset_table[r + 1] - comp_start;
                const uint8_t* comp_ptr = comp_data + comp_start;

                // Inline KMP state
                int current_pat = 0;
                int l = 0;
                int p_offset = s_kmp_offsets[0];
                int p_len = s_kmp_lengths[0];
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

                        while (l > 0 && s_kmp_pat[p_offset + l] != c)
                            l = s_kmp_next[p_offset + l - 1];
                        if (s_kmp_pat[p_offset + l] == c) l++;
                        if (l == p_len) {
                            current_pat++;
                            l = 0;
                            if (current_pat >= p.num_patterns) {
                                done = true; break;
                            }
                            p_offset = s_kmp_offsets[current_pat];
                            p_len = s_kmp_lengths[current_pat];
                        }
                    }
                }

                // KMP matched → this supplier should be excluded
                if (done) {
                    uint64_t row_id = cb_rec_base + r;
                    uint32_t pos = atomicAdd(p.d_excl_count, 1);
                    p.d_excl_suppkeys[pos] = p.d_s_suppkey_flat[row_id];
                }
            }

            cb_rec_base += cb_nrecs;
        }

        // Fallback prefetch for n_comp_blocks==0
        if (n_comp_blocks == 0 && has_next && tid == 0) {
            uint32_t dev;
            uint64_t lba = compute_lba_dev(next_pg, dev);
            uint32_t nblk = q16ff_safe_io_nblocks(p.d_comp_sizes[next_pg]);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk,
                                    nxt_slot, dev);
        }
        __syncthreads();

        long long t2 = clock64();

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
// Kernel 2: P_BRAND — FSST Fused IO + Decomp + Brand ID Extraction
//
// FSST decode → register buffer → extract brand[6,7] → brand_id
// ============================================================

__global__ void bam_q16_fsst_p_brand_kernel(
    void*           ctrls_opaque,
    void*           pc_opaque,
    const char*     pc_base_addr,
    BAMq16FsstBrandParams p)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    extern __shared__ __align__(8) char smem_fbr[];

    uint8_t*  s_sym_len = (uint8_t*)  smem_fbr;
    uint64_t* s_sym_val = (uint64_t*)(smem_fbr + 256);

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
            uint32_t nblk = q16ff_safe_io_nblocks(p.d_comp_sizes[pg]);
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

        // Page-level validity check
        uint32_t comp_sz = p.d_comp_sizes[pg];
        bool page_valid = (n_comp_blocks <= 1024 && watermark <= comp_sz);
        if (!page_valid)
            n_comp_blocks = 0;

        const FsstCompBlockDirEntry* dir =
            reinterpret_cast<const FsstCompBlockDirEntry*>(
                comp_page + FSST_PAGE_NCB_OFFSET + 4);
        const char* symtab_base = comp_page + FSST_PAGE_NCB_OFFSET + 4
            + n_comp_blocks * sizeof(FsstCompBlockDirEntry);
        uint32_t fsst_data_end = fsst_page_string_data_end(
            comp_page, watermark, n_comp_blocks);

        // Load symbol table to smem
        q16ff_load_symtab(s_sym_len, s_sym_val, symtab_base, tid);
        __syncthreads();

        long long t1 = clock64();

        // Process each comp block
        uint64_t cb_rec_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

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
                uint32_t nblk = q16ff_safe_io_nblocks(p.d_comp_sizes[next_pg]);
                bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk,
                                        nxt_slot, dev);
            }

            // FSST decode → brand_id extraction (read directly from page cache)
            const uint8_t* cb_data_ptr = (const uint8_t*)(comp_page + cb_offset);
            const uint16_t* offset_table = (const uint16_t*)cb_data_ptr;
            const uint8_t* comp_data = cb_data_ptr
                + (cb_nrecs + 1) * sizeof(uint16_t);

            for (uint32_t r = tid; r < cb_nrecs; r += blockDim.x) {
                uint16_t comp_start = offset_table[r];
                uint16_t comp_len = offset_table[r + 1] - comp_start;
                const uint8_t* comp_ptr = comp_data + comp_start;

                // Decode to register, only need positions 6 and 7
                char decoded_6 = 0, decoded_7 = 0;
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
                        char c = (char)(sym_val & 0xFF);
                        sym_val >>= 8;
                        if (posOut == 6) decoded_6 = c;
                        if (posOut == 7) decoded_7 = c;
                        posOut++;
                    }
                }

                uint64_t row_id = cb_rec_base + r;
                uint32_t d1 = decoded_6 - '1';
                uint32_t d2 = decoded_7 - '1';
                p.d_brand_ids[row_id] = d1 * 5 + d2;
            }

            cb_rec_base += cb_nrecs;
        }

        // Fallback prefetch for n_comp_blocks==0
        if (n_comp_blocks == 0 && has_next && tid == 0) {
            uint32_t dev;
            uint64_t lba = compute_lba_dev(next_pg, dev);
            uint32_t nblk = q16ff_safe_io_nblocks(p.d_comp_sizes[next_pg]);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk,
                                    nxt_slot, dev);
        }
        __syncthreads();

        long long t2 = clock64();

        if (tid == 0 && p.d_phase_cycles) {
            atomicAdd((unsigned long long*)&p.d_phase_cycles[0], 0ULL);
            atomicAdd((unsigned long long*)&p.d_phase_cycles[1],
                      (unsigned long long)(t1 - t0));
            atomicAdd((unsigned long long*)&p.d_phase_cycles[2],
                      (unsigned long long)(t2 - t1));
        }

        uint32_t tmp = cur_slot; cur_slot = nxt_slot; nxt_slot = tmp;
    }
}

// ============================================================
// Kernel 3: P_TYPE — FSST Fused IO + Decomp + Type ID Dictionary
//
// FSST decode → register buffer → MEDIUM POLISHED check →
// FNV-1a hash → dictionary probe/insert
// ============================================================

__global__ void bam_q16_fsst_p_type_kernel(
    void*           ctrls_opaque,
    void*           pc_opaque,
    const char*     pc_base_addr,
    BAMq16FsstTypeParams p)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    extern __shared__ __align__(8) char smem_fty[];

    uint8_t*  s_sym_len = (uint8_t*)  smem_fty;
    uint64_t* s_sym_val = (uint64_t*)(smem_fty + 256);

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
            uint32_t nblk = q16ff_safe_io_nblocks(p.d_comp_sizes[pg]);
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

        // Page-level validity check
        uint32_t comp_sz = p.d_comp_sizes[pg];
        bool page_valid = (n_comp_blocks <= 1024 && watermark <= comp_sz);
        if (!page_valid)
            n_comp_blocks = 0;

        const FsstCompBlockDirEntry* dir =
            reinterpret_cast<const FsstCompBlockDirEntry*>(
                comp_page + FSST_PAGE_NCB_OFFSET + 4);
        const char* symtab_base = comp_page + FSST_PAGE_NCB_OFFSET + 4
            + n_comp_blocks * sizeof(FsstCompBlockDirEntry);
        uint32_t fsst_data_end = fsst_page_string_data_end(
            comp_page, watermark, n_comp_blocks);

        // Load symbol table to smem
        q16ff_load_symtab(s_sym_len, s_sym_val, symtab_base, tid);
        __syncthreads();

        long long t1 = clock64();

        // Process each comp block
        uint64_t cb_rec_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

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
                uint32_t nblk = q16ff_safe_io_nblocks(p.d_comp_sizes[next_pg]);
                bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk,
                                        nxt_slot, dev);
            }

            // FSST decode → register buffer → type_id extraction (read directly from page cache)
            const uint8_t* cb_data_ptr = (const uint8_t*)(comp_page + cb_offset);
            const uint16_t* offset_table = (const uint16_t*)cb_data_ptr;
            const uint8_t* comp_data = cb_data_ptr
                + (cb_nrecs + 1) * sizeof(uint16_t);

            for (uint32_t r = tid; r < cb_nrecs; r += blockDim.x) {
                uint16_t comp_start = offset_table[r];
                uint16_t comp_len = offset_table[r + 1] - comp_start;
                const uint8_t* comp_ptr = comp_data + comp_start;

                // Full FSST decode to register buffer (type strings ≤ 25 chars)
                char decoded[Q16FF_TYPE_MAX_LEN];
                uint32_t posIn = 0, posOut = 0;

                while (posIn < comp_len && posOut < Q16FF_TYPE_MAX_LEN) {
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

                    for (uint8_t j = 0; j < sym_len && posOut < Q16FF_TYPE_MAX_LEN; j++) {
                        decoded[posOut++] = (char)(sym_val & 0xFF);
                        sym_val >>= 8;
                    }
                }

                uint16_t vlen = (uint16_t)posOut;
                uint64_t row_id = cb_rec_base + r;

                // Check NOT LIKE 'MEDIUM POLISHED%'
                bool is_medium_polished = false;
                if (vlen >= 15) {
                    const char mp[] = "MEDIUM POLISHED";
                    is_medium_polished = true;
                    for (int k = 0; k < 15; k++) {
                        if (decoded[k] != mp[k]) {
                            is_medium_polished = false;
                            break;
                        }
                    }
                }
                if (is_medium_polished) {
                    p.d_type_ids[row_id] = UINT32_MAX;
                    continue;
                }

                // FNV-1a hash → dictionary probe/insert
                uint64_t h = q16ff_fnv1a64(decoded, vlen);
                uint32_t dict_slot = (uint32_t)h & Q16FF_TYPE_DICT_MASK;

                while (true) {
                    uint64_t prev = atomicCAS(
                        reinterpret_cast<unsigned long long*>(
                            &p.d_dict_keys[dict_slot]),
                        (unsigned long long)UINT64_MAX,
                        (unsigned long long)h);

                    if (prev == UINT64_MAX) {
                        // New entry
                        uint32_t new_tid = atomicAdd(p.d_type_id_counter, 1);
                        char* dst = p.d_dict_strs
                            + (uint64_t)dict_slot * Q16FF_TYPE_MAX_LEN;
                        for (uint16_t k = 0; k < vlen; k++)
                            dst[k] = decoded[k];
                        p.d_dict_lens[dict_slot] = vlen;
                        __threadfence();
                        p.d_dict_type_ids[dict_slot] = new_tid;
                        p.d_type_ids[row_id] = new_tid;
                        break;
                    }
                    if (prev == h) {
                        // Existing entry — wait for type_id to be written
                        uint32_t existing_tid;
                        do {
                            __threadfence();
                            existing_tid = p.d_dict_type_ids[dict_slot];
                        } while (existing_tid == UINT32_MAX);
                        p.d_type_ids[row_id] = existing_tid;
                        break;
                    }
                    dict_slot = (dict_slot + 1) & Q16FF_TYPE_DICT_MASK;
                }
            }

            cb_rec_base += cb_nrecs;
        }

        // Fallback prefetch for n_comp_blocks==0
        if (n_comp_blocks == 0 && has_next && tid == 0) {
            uint32_t dev;
            uint64_t lba = compute_lba_dev(next_pg, dev);
            uint32_t nblk = q16ff_safe_io_nblocks(p.d_comp_sizes[next_pg]);
            bam_io_read_page_device(ctrls_opaque, pc_opaque, lba, nblk,
                                    nxt_slot, dev);
        }
        __syncthreads();

        long long t2 = clock64();

        if (tid == 0 && p.d_phase_cycles) {
            atomicAdd((unsigned long long*)&p.d_phase_cycles[0], 0ULL);
            atomicAdd((unsigned long long*)&p.d_phase_cycles[1],
                      (unsigned long long)(t1 - t0));
            atomicAdd((unsigned long long*)&p.d_phase_cycles[2],
                      (unsigned long long)(t2 - t1));
        }

        uint32_t tmp = cur_slot; cur_slot = nxt_slot; nxt_slot = tmp;
    }
}

// ============================================================
// FSST Fused IO Context — no decomp_buf
// ============================================================

struct BAMq16FsstIOContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_q16_fsst_io_ctx_t bam_q16_fsst_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMq16FsstIOContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // 2 slots per block (double-buffered IO) — NO decomp_buf
    const uint32_t num_slots = num_blocks * 2;

    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);
    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    return static_cast<bam_q16_fsst_io_ctx_t>(ctx);
}

void bam_q16_fsst_io_destroy(bam_q16_fsst_io_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMq16FsstIOContext*>(ctx_handle);
    if (!ctx) return;

    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}

// ============================================================
// Launch helpers
// ============================================================

// smem: symtab only (2496B) — comp block data read directly from page cache
static size_t q16ff_smem_size(uint32_t /*page_size*/) {
    return FSST_SMEM_SYMTAB_OFFSET;  // 2496 bytes
}

// ── S_COMMENT launch ──

void bam_q16_fsst_s_comment_async(
    bam_q16_fsst_io_ctx_t ctx_handle,
    const BAMq16FsstSCommentParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq16FsstIOContext*>(ctx_handle);
    size_t smem = q16ff_smem_size(p.page_size);
    constexpr uint32_t TPB = 128;

    auto kfn = bam_q16_fsst_s_comment_kernel;
    BAM_Q16F_CUDA_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kfn<<<p.num_blocks, TPB, smem, stream>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr, p);
    BAM_Q16F_CUDA_CHECK(cudaGetLastError());
}

// ── P_BRAND launch ──

void bam_q16_fsst_p_brand_async(
    bam_q16_fsst_io_ctx_t ctx_handle,
    const BAMq16FsstBrandParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq16FsstIOContext*>(ctx_handle);
    size_t smem = q16ff_smem_size(p.page_size);
    constexpr uint32_t TPB = 128;

    auto kfn = bam_q16_fsst_p_brand_kernel;
    BAM_Q16F_CUDA_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kfn<<<p.num_blocks, TPB, smem, stream>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr, p);
    BAM_Q16F_CUDA_CHECK(cudaGetLastError());
}

// ── P_TYPE launch ──

void bam_q16_fsst_p_type_async(
    bam_q16_fsst_io_ctx_t ctx_handle,
    const BAMq16FsstTypeParams& p,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMq16FsstIOContext*>(ctx_handle);
    size_t smem = q16ff_smem_size(p.page_size);
    constexpr uint32_t TPB = 128;

    auto kfn = bam_q16_fsst_p_type_kernel;
    BAM_Q16F_CUDA_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kfn<<<p.num_blocks, TPB, smem, stream>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr, p);
    BAM_Q16F_CUDA_CHECK(cudaGetLastError());
}
