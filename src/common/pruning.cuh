#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <cstdint>
#include "filter.cuh"

// ============================================================
// GPU Zonemap Pruning Utilities
//
// Convention: active page list is always built on GPU.
//   gidp:  D2H the list for CPU-side GDS IO threads.
//   BaM:   use d_active_ids directly in GPU kernels.
//
// Typical flow (inside timed section):
//   1. Read zonemap stats to GPU (GDS or BaM — caller's responsibility)
//   2. zonemap_init_mask()
//   3. zonemap_eval_range / zonemap_eval_point  (repeatable, AND-composed)
//   4. zonemap_compact_active()  → d_active_ids + count
// ============================================================

// ── Evaluation kernels ──────────────────────────────────────

// Range predicate: prune page if stats don't overlap [pred_lo, pred_hi].
// ANDs with existing d_mask (call zonemap_init_mask first).
__global__ void zonemap_eval_range_kernel(
    const Stats<int32_t>* __restrict__ d_stats,
    uint64_t nstats, uint64_t npages,
    int32_t pred_lo, int32_t pred_hi,
    uint8_t* __restrict__ d_mask)
{
    uint64_t pg = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (pg >= npages || d_mask[pg] == 0) return;
    if (pg >= nstats) return;  // no stats for this page → keep active
    int32_t mn = d_stats[pg].min_val;
    int32_t mx = d_stats[pg].max_val;
    if (mn > mx || max(mn, pred_lo) > min(mx, pred_hi))
        d_mask[pg] = 0;
}

// Point predicate: prune page if stats don't contain pred_val.
__global__ void zonemap_eval_point_kernel(
    const Stats<int32_t>* __restrict__ d_stats,
    uint64_t nstats, uint64_t npages,
    int32_t pred_val,
    uint8_t* __restrict__ d_mask)
{
    uint64_t pg = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (pg >= npages || d_mask[pg] == 0) return;
    if (pg >= nstats) return;
    int32_t mn = d_stats[pg].min_val;
    int32_t mx = d_stats[pg].max_val;
    if (mn > mx || pred_val < mn || pred_val > mx)
        d_mask[pg] = 0;
}

// ── Host-side wrappers ──────────────────────────────────────

// Initialize mask to all-active.
static inline void zonemap_init_mask(uint8_t* d_mask, uint64_t npages,
                                     cudaStream_t stream)
{
    cudaMemsetAsync(d_mask, 1, npages, stream);
}

// Evaluate range predicate on GPU (launch kernel).
static inline void zonemap_eval_range(
    const Stats<int32_t>* d_stats, uint64_t nstats, uint64_t npages,
    int32_t pred_lo, int32_t pred_hi,
    uint8_t* d_mask, cudaStream_t stream)
{
    if (nstats == 0) return;
    uint32_t nblk = (uint32_t)((npages + 255) / 256);
    zonemap_eval_range_kernel<<<nblk, 256, 0, stream>>>(
        d_stats, nstats, npages, pred_lo, pred_hi, d_mask);
}

// Evaluate point predicate on GPU (launch kernel).
static inline void zonemap_eval_point(
    const Stats<int32_t>* d_stats, uint64_t nstats, uint64_t npages,
    int32_t pred_val,
    uint8_t* d_mask, cudaStream_t stream)
{
    if (nstats == 0) return;
    uint32_t nblk = (uint32_t)((npages + 255) / 256);
    zonemap_eval_point_kernel<<<nblk, 256, 0, stream>>>(
        d_stats, nstats, npages, pred_val, d_mask);
}

// ── Fused multi-predicate evaluation (single kernel) ────────

struct ZonemapPred {
    const Stats<int32_t>* d_stats;  // GPU-resident Stats array
    uint64_t nstats;
    int32_t lo, hi;                 // range [lo,hi]; point: lo==hi
};

static constexpr uint32_t kZonemapMaxPreds = 4;

__global__ void zonemap_eval_preds_kernel(
    uint64_t npages,
    const ZonemapPred* __restrict__ d_preds, uint32_t npreds,
    uint8_t* __restrict__ d_mask)
{
    uint64_t pg = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (pg >= npages) return;
    uint8_t active = 1;
    for (uint32_t i = 0; i < npreds && active; i++) {
        if (pg >= d_preds[i].nstats) continue;
        int32_t mn = d_preds[i].d_stats[pg].min_val;
        int32_t mx = d_preds[i].d_stats[pg].max_val;
        if (mn > mx || max(mn, d_preds[i].lo) > min(mx, d_preds[i].hi))
            active = 0;
    }
    d_mask[pg] = active;
}

// Evaluate all predicates in a single kernel (AND-composed).
// d_preds: pre-allocated device buffer (Rule 4).
// h_preds: host array copied to d_preds via cudaMemcpyAsync.
static inline void zonemap_eval_preds(
    uint64_t npages,
    const ZonemapPred* h_preds, uint32_t npreds,
    ZonemapPred* d_preds,
    uint8_t* d_mask, cudaStream_t stream)
{
    cudaMemcpyAsync(d_preds, h_preds, npreds * sizeof(ZonemapPred),
                    cudaMemcpyHostToDevice, stream);
    uint32_t nblk = (uint32_t)((npages + 255) / 256);
    zonemap_eval_preds_kernel<<<nblk, 256, 0, stream>>>(
        npages, d_preds, npreds, d_mask);
}

// ── GPU compaction: mask → active page ID list ──────────────

// Query CUB temp storage size for compaction.
// Call once outside timed section to pre-allocate.
static inline size_t zonemap_compact_query_temp(uint64_t max_pages)
{
    size_t temp_bytes = 0;
    thrust::counting_iterator<uint32_t> iota(0);
    cub::DeviceSelect::Flagged(
        nullptr, temp_bytes,
        iota, static_cast<const uint8_t*>(nullptr),
        static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr),
        static_cast<int>(max_pages));
    return temp_bytes;
}

// Compact d_mask → d_active_ids on GPU.
// d_num_selected (device memory, uint32_t) receives the count of active pages.
// Caller must cudaStreamSynchronize before reading d_num_selected.
static inline void zonemap_compact_active(
    const uint8_t* d_mask, uint64_t npages,
    uint32_t* d_active_ids,
    uint32_t* d_num_selected,
    void* d_cub_temp, size_t cub_temp_bytes,
    cudaStream_t stream)
{
    thrust::counting_iterator<uint32_t> iota(0);
    cub::DeviceSelect::Flagged(
        d_cub_temp, cub_temp_bytes,
        iota, d_mask, d_active_ids, d_num_selected,
        static_cast<int>(npages), stream);
}

// ── GDS zonemap reader ──────────────────────────────────────
// Reads Stats<int32_t> pages from NVMe via cuFileRead to GPU buffer.
// d_buf must be cuFile-registered. Pages are device-striped.
// Usable for both column stats and sideways stats.

// ── In-kernel compaction for persistent kernels ────────────
#include "zonemap_compact.cuh"

// ── GDS zonemap reader ──────────────────────────────────────

#ifdef __CUFILE_H_
static inline void gds_read_zonemap(
    const CUfileHandle_t* cufile_handles,
    uint64_t num_devices,
    uint64_t stats_start_page_id, uint64_t stats_npg,
    size_t page_size,
    void* d_buf)
{
    for (uint64_t j = 0; j < stats_npg; j++) {
        uint64_t pg_id = stats_start_page_id + j;
        uint32_t dev = pg_id % num_devices;
        uint64_t local = pg_id / num_devices;
        off_t file_offset = static_cast<off_t>(local * page_size);
        off_t buf_offset = static_cast<off_t>(j * page_size);
        cuFileRead(cufile_handles[dev], d_buf, page_size,
                   file_offset, buf_offset);
    }
}
#endif

// ============================================================
// GDS Zonemap eval — stats already on GPU (no BaM)
//
// For gidp mode: stats pages are uploaded to GPU via cudaMemcpy,
// then a GPU kernel evaluates predicates + compact + Phase 4.
// ============================================================

struct GdsZonemapPred {
    const int32_t* d_stats;     // GPU ptr: Stats<int32_t>[nstats] (min/max pairs)
    uint64_t nstats;
    int32_t  pred_lo;
    int32_t  pred_hi;
};

static constexpr uint32_t kGdsZonemapMaxPreds = 4;

struct GdsZonemapCtx {
    // GPU-side
    GdsZonemapPred* d_preds      = nullptr;
    uint8_t*        d_mask       = nullptr;
    uint32_t*       d_active_ids = nullptr;
    uint32_t*       d_num_active = nullptr;

    // Pinned host mirrors
    GdsZonemapPred* h_preds      = nullptr;
    uint8_t*        h_mask       = nullptr;
    uint32_t*       h_num_active = nullptr;

    uint64_t max_npages = 0;

    // Phase 4 (optional): set d_ps_i32 != nullptr to enable
    const uint64_t* d_ps_i32  = nullptr;
    const uint64_t* d_ps_i64  = nullptr;
    uint8_t*        d_mask_i64 = nullptr;
    uint32_t        npages_i64 = 0;
};

__global__ void gds_zonemap_eval_kernel(
    const GdsZonemapPred* __restrict__ d_preds, uint32_t npreds,
    uint64_t npages,
    uint8_t* __restrict__ d_mask,
    uint32_t* __restrict__ d_active_ids,
    uint32_t* __restrict__ d_num_active,
    const uint64_t* __restrict__ d_ps_i32,
    const uint64_t* __restrict__ d_ps_i64,
    uint8_t* __restrict__ d_mask_i64,
    uint32_t npages_i64)
{
    const uint32_t tid = threadIdx.x;

    // Phase 2: evaluate predicates (stats data accessed via d_preds[i].d_stats)
    for (uint64_t pg = tid; pg < npages; pg += blockDim.x) {
        uint8_t active = 1;
        for (uint32_t i = 0; i < npreds && active; i++) {
            if (pg >= d_preds[i].nstats) continue;
            const int32_t* s = d_preds[i].d_stats + pg * 2;
            int32_t mn = s[0], mx = s[1];
            if (mn > mx || max(mn, d_preds[i].pred_lo) > min(mx, d_preds[i].pred_hi))
                active = 0;
        }
        d_mask[pg] = active;
    }

    // Phase 3: compact d_mask → d_active_ids
    __syncthreads();
    {
        const uint32_t warp_id = tid / 32;
        const uint32_t lane    = tid % 32;
        constexpr uint32_t NWARPS = 8;  // 256 / 32

        __shared__ uint32_t s_wpfx[NWARPS];
        __shared__ uint32_t s_base;

        if (tid == 0) s_base = 0;
        __syncthreads();

        for (uint64_t chunk = 0; chunk < npages; chunk += blockDim.x) {
            uint64_t pg = chunk + tid;
            bool is_active = (pg < npages) && d_mask[pg];

            uint32_t ballot      = __ballot_sync(0xffffffff, is_active);
            uint32_t lane_prefix = __popc(ballot & ((1u << lane) - 1));
            uint32_t warp_cnt    = __popc(ballot);

            if (lane == 0) s_wpfx[warp_id] = warp_cnt;
            __syncthreads();

            if (tid == 0) {
                uint32_t b = s_base, sum = 0;
                for (uint32_t w = 0; w < NWARPS; w++) {
                    uint32_t c = s_wpfx[w];
                    s_wpfx[w] = b + sum;
                    sum += c;
                }
                s_base = b + sum;
            }
            __syncthreads();

            if (is_active)
                d_active_ids[s_wpfx[warp_id] + lane_prefix] =
                    static_cast<uint32_t>(pg);
            __syncthreads();
        }

        if (tid == 0) *d_num_active = s_base;
    }

    // Phase 4 (optional): derive INT64 page mask from INT32 mask
    if (d_ps_i32) {
        __syncthreads();
        for (uint32_t j = tid; j < npages_i64; j += blockDim.x)
            d_mask_i64[j] = 0;
        __syncthreads();
        for (uint64_t pg = tid; pg < npages; pg += blockDim.x) {
            if (!d_mask[pg]) continue;
            uint64_t base_row = (pg == 0) ? 0 : d_ps_i32[pg - 1];
            uint64_t end_row  = d_ps_i32[pg];
            if (end_row <= base_row) continue;
            uint64_t last_row = end_row - 1;
            uint32_t lo = 0, hi = npages_i64;
            while (lo < hi) {
                uint32_t mid = lo + (hi - lo) / 2;
                if (d_ps_i64[mid] <= base_row) lo = mid + 1;
                else hi = mid;
            }
            uint32_t i64_lo = lo;
            lo = i64_lo; hi = npages_i64;
            while (lo < hi) {
                uint32_t mid = lo + (hi - lo) / 2;
                if (d_ps_i64[mid] <= last_row) lo = mid + 1;
                else hi = mid;
            }
            uint32_t i64_hi = lo;
            for (uint32_t j = i64_lo; j <= i64_hi && j < npages_i64; j++)
                d_mask_i64[j] = 1;
        }
    }
}

static inline GdsZonemapCtx gds_zonemap_ctx_create(uint64_t max_npages)
{
    GdsZonemapCtx ctx{};
    ctx.max_npages = max_npages;

    cudaMalloc(&ctx.d_preds, kGdsZonemapMaxPreds * sizeof(GdsZonemapPred));
    cudaMallocHost(&ctx.h_preds, kGdsZonemapMaxPreds * sizeof(GdsZonemapPred));
    cudaMalloc(&ctx.d_mask, max_npages);
    cudaMallocHost(&ctx.h_mask, max_npages);
    cudaMalloc(&ctx.d_active_ids, max_npages * sizeof(uint32_t));
    cudaMalloc(&ctx.d_num_active, sizeof(uint32_t));
    cudaMallocHost(&ctx.h_num_active, sizeof(uint32_t));

    return ctx;
}

static inline void gds_zonemap_ctx_destroy(GdsZonemapCtx& ctx)
{
    if (ctx.d_preds)      { cudaFree(ctx.d_preds);        ctx.d_preds      = nullptr; }
    if (ctx.h_preds)      { cudaFreeHost(ctx.h_preds);    ctx.h_preds      = nullptr; }
    if (ctx.d_mask)       { cudaFree(ctx.d_mask);          ctx.d_mask       = nullptr; }
    if (ctx.h_mask)       { cudaFreeHost(ctx.h_mask);      ctx.h_mask       = nullptr; }
    if (ctx.d_active_ids) { cudaFree(ctx.d_active_ids);    ctx.d_active_ids = nullptr; }
    if (ctx.d_num_active) { cudaFree(ctx.d_num_active);    ctx.d_num_active = nullptr; }
    if (ctx.h_num_active) { cudaFreeHost(ctx.h_num_active); ctx.h_num_active = nullptr; }
}

static inline void gds_zonemap_eval_async(
    GdsZonemapCtx& ctx,
    uint64_t npages,
    uint32_t npreds,
    cudaStream_t stream)
{
    cudaMemcpyAsync(ctx.d_preds, ctx.h_preds,
                    npreds * sizeof(GdsZonemapPred),
                    cudaMemcpyHostToDevice, stream);

    gds_zonemap_eval_kernel<<<1, 256, 0, stream>>>(
        ctx.d_preds, npreds,
        npages, ctx.d_mask,
        ctx.d_active_ids, ctx.d_num_active,
        ctx.d_ps_i32, ctx.d_ps_i64, ctx.d_mask_i64, ctx.npages_i64);

    cudaMemcpyAsync(ctx.h_mask, ctx.d_mask, npages,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ctx.h_num_active, ctx.d_num_active, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
}
