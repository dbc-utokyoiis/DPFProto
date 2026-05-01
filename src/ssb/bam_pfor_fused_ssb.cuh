#pragma once

// bam_pfor_fused_ssb.cuh — SSB PFOR Warp-Spec Fused BaM I/O + PFOR decomp + scan
//
// Mirror of bam_lz4_fused_ssb.cuh but for PFOR decompression.
// Reuses SSBFusedQ{1,2,3,4}xParams from bam_lz4_fused_ssb.cuh.
// No nvCOMPdx dependency — much less shared memory, enabling 2 blocks/SM.
//
// Warp layout (same as LZ4):
//   Q1x/Q2x/Q3x: 4 IO warps + 7 decomp groups × 4 fields = 32 warps
//   Q4x:          6 IO warps + 4 decomp groups × 6 fields = 30 warps (+2 idle)

#include "bam_lz4_fused_ssb.cuh"  // SSBFusedQ{1,2,3,4}xParams, ssb_fused_ctx_t, etc.

// Max co-resident blocks (PFOR version — typically 2x LZ4 due to lower smem).
uint32_t ssb_pfor_fused_q1x_max_blocks(uint32_t page_size);  // also for Q2x/Q3x
uint32_t ssb_pfor_fused_q4x_max_blocks(uint32_t page_size);

// Kernel launch functions (PFOR version).
void ssb_pfor_fused_q1x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ1xParams& p,
    uint32_t num_blocks, cudaStream_t stream);

void ssb_pfor_fused_q2x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ2xParams& p,
    uint32_t num_blocks, cudaStream_t stream);

void ssb_pfor_fused_q3x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ3xParams& p,
    uint32_t num_blocks, cudaStream_t stream);

void ssb_pfor_fused_q4x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ4xParams& p,
    uint32_t num_blocks, cudaStream_t stream);
