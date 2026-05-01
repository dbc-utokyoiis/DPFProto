#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q1.cuh — Fused BaM I/O + nvCOMPdx LZ4 decompress + Q1 scan
//
// Balanced pipeline: 1 IO warp + 7 decomp warps = 256 threads/block.
// IO warp reads all 7 INT32 LINEITEM fields sequentially; each decomp warp
// decompresses one field with nvCOMPdx LZ4.
// __launch_bounds__(256, 4) → 4 blocks/SM for high SM utilization.
// After decompression, all 256 threads evaluate Q1 predicates and accumulate
// aggregates via per-thread local accumulators + atomicAdd flush.
// ============================================================

typedef void* bam_fused_q1_ctx_t;
typedef void* bam_ctrl_handle_t;  // defined in bam_kernel.cuh, redeclared here

// Q1 field indices (matches TPCH::Query::Q1::SCAN_TARGET_COLS order)
// 0=L_QUANTITY, 1=L_EXTENDEDPRICE, 2=L_DISCOUNT, 3=L_TAX,
// 4=L_RETURNFLAG, 5=L_LINESTATUS, 6=L_SHIPDATE
constexpr uint32_t FUSED_Q1_NUM_FIELDS = 7;

// Optional per-phase cycle accumulator (nullptr to disable).
// Layout: [0]=IO cycles sum, [1]=decomp cycles sum, [2]=scan cycles sum,
//         [3]=iterations counted, [4]=total block cycles sum
struct BAMFusedQ1Profile {
    unsigned long long* d_cycles;  // uint64_t[5]
};

struct BAMFusedQ1Params {
    // Per-field (7 fields)
    uint64_t  field_start_page_ids[FUSED_Q1_NUM_FIELDS];
    uint64_t* d_comp_offsets[FUSED_Q1_NUM_FIELDS];   // [npages] byte offsets (GPU)
    uint32_t* d_comp_sizes[FUSED_Q1_NUM_FIELDS];     // [npages] compressed sizes (GPU)
    bool      is_compressed[FUSED_Q1_NUM_FIELDS];

    // Multi-device
    uint64_t  partition_start_lbas[4]; // per-device partition LBA (RAID0)
    uint32_t  n_devices;

    // Page geometry
    uint32_t  page_size;
    uint64_t  npages;              // total pages (same for all 7 INT32 fields)
    uint32_t  num_blocks;          // grid size

    // Output: Q1 aggregate array
    // [Q1_NUM_GROUPS * Q1_NUM_AGGS] = [6 * 7] = 42 int64_t, pre-zeroed
    int64_t*  d_agg;

    // Zone map (optional)
    const uint8_t* d_page_active;  // [npages] 1=active, 0=skip (or nullptr)
    const uint32_t* d_active_page_ids;  // dense active page IDs (or nullptr)
    uint32_t num_active_pages;

    // Optional profile accumulator (nullptr disables)
    unsigned long long* d_cycles;  // uint64_t[5]: IO, decomp, scan, iters, total
};

// Create fused Q1 context (page_cache + decomp_buf allocation).
// num_blocks: CUDA grid size (typically min(npages, sm_count * 2)).
bam_fused_q1_ctx_t bam_fused_q1_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

// Launch fused Q1 kernel (asynchronous — returns immediately).
void bam_fused_q1_run_async(
    bam_fused_q1_ctx_t ctx,
    const BAMFusedQ1Params& params,
    cudaStream_t stream);

// Destroy context (free page_cache + decomp_buf).
void bam_fused_q1_destroy(bam_fused_q1_ctx_t ctx);
