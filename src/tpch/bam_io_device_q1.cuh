#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// BaM IO device wrapper — Q1-dedicated v2 variant.
//
// Separate TU compiled with CUDA_STANDARD 17 + LTO so that
// bam_lz4_fused_q1.cu can request __launch_bounds__(*, 5)
// without the external-TU regcount ceiling imposed by the
// shared bam_io_device library.
// ============================================================

__device__ void bam_io_read_page_device_q1(
    void* ctrls,
    void* pc,
    uint64_t lba,
    uint32_t nblk,
    uint32_t bid,
    uint32_t dev = 0);

__device__ void bam_io_submit_page_device_q1(
    void* ctrls,
    void* pc,
    uint64_t lba,
    uint32_t nblk,
    uint32_t slot,
    uint32_t dev,
    void** out_qp,
    uint16_t* out_cid);

__device__ void bam_io_poll_page_device_q1(
    void* qp,
    uint16_t cid);
