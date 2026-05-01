// ============================================================
// BaM IO device wrapper — CUDA C++11 with separable compilation.
//
// Provides a __device__ function for GPU-initiated NVMe reads
// and host-side page_cache management, callable from C++17 code
// (e.g., nvCOMPdx kernels) via device linking.
//
// Compiled separately to avoid BaM header vs C++17/CCCL conflicts.
// ============================================================

#include "bam_io_device.cuh"

#include <ctrl.h>
// Rename __flush to avoid multiple-definition conflict with bam_kernel.cu
// (both TUs include page_cache.h which defines a __global__ __flush kernel).
#define __flush __bam_io_device_flush
#include <page_cache.h>
#undef __flush
#include <nvm_parallel_queue.h>
#include <nvm_cmd.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BAM_IO_CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ── Device function ──

__device__ void bam_io_read_page_device(
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

__device__ void bam_io_submit_page_device(
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

__device__ void bam_io_submit_page_device_qp(
    void* ctrls_v,
    void* pc_v,
    uint64_t lba,
    uint32_t nblk,
    uint32_t pc_slot,
    uint32_t qp_hint,
    uint32_t dev,
    void** out_qp,
    uint16_t* out_cid)
{
    Controller** ctrls = (Controller**)ctrls_v;
    page_cache_d_t* pc = (page_cache_d_t*)pc_v;
    QueuePair* qp = ctrls[dev]->d_qps + (qp_hint % ctrls[dev]->n_qps);
    uint16_t sq_pos = 0;
    access_data_async(pc, qp, lba, nblk, (unsigned long long)pc_slot,
                      NVM_IO_READ, out_cid, &sq_pos);
    *out_qp = (void*)qp;
}

__device__ void bam_io_poll_page_device(void* qp_v, uint16_t cid)
{
    QueuePair* qp = (QueuePair*)qp_v;
    uint32_t pl, ph;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &pl, &ph);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    put_cid(&qp->sq, cid);
}

__device__ bool bam_io_try_poll_page_device(void* qp_v, uint16_t cid)
{
    QueuePair* qp = (QueuePair*)qp_v;
    nvm_queue_t* cq = &qp->cq;

    // Single-pass CQ scan (non-blocking)
    uint32_t head = cq->head.load(simt::memory_order_relaxed);
    for (size_t i = 0; i < cq->qs_minus_1; i++) {
        uint32_t cur_head = head + i;
        bool search_phase = ((~(cur_head >> cq->qs_log2)) & 0x01);
        uint32_t loc = cur_head & (cq->qs_minus_1);
        uint32_t cpl_entry = ((nvm_cpl_t*)cq->vaddr)[loc].dword[3];
        uint32_t entry_cid = (cpl_entry & 0x0000ffff);
        bool phase = (cpl_entry & 0x00010000) >> 16;

        if ((entry_cid == cid) && (phase == search_phase)) {
            // Found — complete dequeue + CID release
            cq_dequeue(cq, loc, &qp->sq);
            put_cid(&qp->sq, cid);
            return true;
        }
        if (phase != search_phase)
            break;
    }
    return false;  // not ready yet
}

// ── Device function: CQ-order poll (consumes CQ head, returns CID) ──
// Polls the CQ head entry (whatever completed next), not a specific CID.
// Must be called by a single consumer per QP to avoid contention.
// Returns the CID of the completed command. Dequeue + CID release done internally.
__device__ uint16_t bam_io_poll_next_page_device(void* qp_v)
{
    QueuePair* qp = (QueuePair*)qp_v;
    nvm_queue_t* cq = &qp->cq;

    unsigned int ns = 8;
    while (true) {
        uint32_t head = cq->head.load(simt::memory_order_relaxed);
        bool expected_phase = ((~(head >> cq->qs_log2)) & 0x01);
        uint32_t pos = head & cq->qs_minus_1;
        uint32_t cpl_entry = ((nvm_cpl_t*)cq->vaddr)[pos].dword[3];
        uint16_t cid = (cpl_entry & 0x0000ffff);
        bool phase = (cpl_entry & 0x00010000) >> 16;

        if (phase == expected_phase) {
            if ((cpl_entry >> 17) != 0)
                printf("NVM Error: %llx\tcid: %u\n",
                       (unsigned long long)(cpl_entry >> 17), (unsigned)cid);
            // Dequeue at head position — trivially succeeds since we're
            // the sole consumer and always at head. Default loc_=0/cur_head_=0
            // makes the wait loop a no-op.
            cq_dequeue(cq, pos, &qp->sq);
            put_cid(&qp->sq, cid);
            return cid;
        }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) ns *= 2;
#endif
    }
}

// ── Host functions ──

// BAMCtrlHandle definition (must match bam_kernel.cu)
struct BAMCtrlHandle {
    Controller* ctrl;
    std::vector<Controller*> ctrls;
    int cuda_device;
};

struct BAMIOPageCacheContext {
    page_cache_t* h_pc;
    BAMCtrlHandle* ctrl_handle;
    uint32_t      page_size;
    uint32_t      num_blocks;
};

bam_io_page_cache_t bam_io_page_cache_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    auto* ctx = new BAMIOPageCacheContext();
    ctx->ctrl_handle = h;
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint64_t n_pc_pages = (uint64_t)num_blocks;
    const uint64_t max_range = 64;
    ctx->h_pc = new page_cache_t(page_size, n_pc_pages, h->cuda_device,
                                  *h->ctrl, max_range, h->ctrls);

    BAM_IO_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    return static_cast<bam_io_page_cache_t>(ctx);
}

void* bam_io_page_cache_get_d_ctrls(bam_io_page_cache_t handle) {
    auto* ctx = static_cast<BAMIOPageCacheContext*>(handle);
    return (void*)ctx->h_pc->pdt.d_ctrls;
}

void* bam_io_page_cache_get_d_pc_ptr(bam_io_page_cache_t handle) {
    auto* ctx = static_cast<BAMIOPageCacheContext*>(handle);
    return (void*)ctx->h_pc->d_pc_ptr;
}

void* bam_io_page_cache_get_base_addr(bam_io_page_cache_t handle) {
    auto* ctx = static_cast<BAMIOPageCacheContext*>(handle);
    return (void*)ctx->h_pc->pdt.base_addr;
}

void bam_io_page_cache_destroy(bam_io_page_cache_t handle) {
    auto* ctx = static_cast<BAMIOPageCacheContext*>(handle);
    if (!ctx) return;
    delete ctx->h_pc;
    delete ctx;
}

uint32_t bam_io_page_cache_get_n_qps(bam_io_page_cache_t handle, uint32_t dev) {
    auto* ctx = static_cast<BAMIOPageCacheContext*>(handle);
    return ctx->ctrl_handle->ctrls[dev]->n_qps;
}

void* bam_io_page_cache_get_qp(bam_io_page_cache_t handle, uint32_t dev, uint32_t qp_idx) {
    auto* ctx = static_cast<BAMIOPageCacheContext*>(handle);
    // Return device pointer to QueuePair (d_qps + qp_idx)
    return (void*)(ctx->ctrl_handle->ctrls[dev]->d_qps + qp_idx);
}
