// ============================================================
// BaM IO device wrapper — Q1 v2 variant.
//
// Identical logic to bam_io_device.cu but compiled as a
// separate LTO-enabled TU (CUDA 17 + -dlto) dedicated to
// bam_lz4_fused_q1. This allows nvlink to re-optimize register
// usage for higher-occupancy launch_bounds without perturbing
// the shared bam_io_device library used by other fused kernels.
// ============================================================

#include "bam_io_device_q1.cuh"

#include <ctrl.h>
// Rename __flush symbols to avoid multiple-definition conflict with
// bam_io_device.cu and bam_kernel.cu (all include page_cache.h which
// defines a __global__ __flush kernel).
#define __flush __bam_io_device_q1_flush
#include <page_cache.h>
#undef __flush
#include <nvm_parallel_queue.h>
#include <nvm_cmd.h>

#include <cuda_runtime.h>

__device__ void bam_io_read_page_device_q1(
    void* ctrls_v,
    void* pc_v,
    uint64_t lba,
    uint32_t nblk,
    uint32_t bid,
    uint32_t dev)
{
    Controller** ctrls = (Controller**)ctrls_v;
    page_cache_d_t* pc = (page_cache_d_t*)pc_v;

    QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);
    uint16_t cid = 0;
    uint16_t sq_pos = 0;
    access_data_async(pc, qp, lba, nblk, (unsigned long long)bid,
                      NVM_IO_READ, &cid, &sq_pos);
    uint32_t poll_loc, poll_head;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    put_cid(&qp->sq, cid);
}

__device__ void bam_io_submit_page_device_q1(
    void* ctrls_v,
    void* pc_v,
    uint64_t lba,
    uint32_t nblk,
    uint32_t slot,
    uint32_t dev,
    void** out_qp,
    uint16_t* out_cid)
{
    Controller** ctrls = (Controller**)ctrls_v;
    page_cache_d_t* pc = (page_cache_d_t*)pc_v;
    QueuePair* qp = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
    uint16_t sq_pos = 0;
    access_data_async(pc, qp, lba, nblk, (unsigned long long)slot,
                      NVM_IO_READ, out_cid, &sq_pos);
    *out_qp = (void*)qp;
}

__device__ void bam_io_poll_page_device_q1(void* qp_v, uint16_t cid)
{
    QueuePair* qp = (QueuePair*)qp_v;
    uint32_t pl, ph;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &pl, &ph);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    put_cid(&qp->sq, cid);
}
