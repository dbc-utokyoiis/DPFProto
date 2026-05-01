#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ── In-kernel compaction for persistent kernels ────────────
// Compacts mask-filtered page IDs into shared memory for one
// persistent-kernel block.  Pages are round-robin:
//   block b owns pages b, b+gridDim.x, b+2*gridDim.x, ...
//
// When d_mask is nullptr every page is active (no pruning).
// s_active must be large enough for all pages of this block.
// *s_count receives the number of active pages.
// Must be called by the entire block (__syncthreads inside).

static constexpr uint32_t kZonemapMaxPagesPerBlock = 512;

// Single-pass version: requires blockDim.x >= max_pg (one thread per page).
// Use for warp-spec kernels with 1024 threads.
__device__ inline void zonemap_compact_block_pages(
    const uint8_t* __restrict__ d_mask,
    uint32_t total_pages,
    uint32_t* s_active,
    uint32_t* s_count)
{
    const uint32_t tid    = threadIdx.x;
    const uint32_t stride = gridDim.x;
    const uint32_t first  = blockIdx.x;
    const uint32_t max_pg = (total_pages > first)
        ? (total_pages - first + stride - 1) / stride : 0;

    // Fast path: no mask -> fill identity in parallel
    if (!d_mask) {
        for (uint32_t i = tid; i < max_pg; i += blockDim.x)
            s_active[i] = first + i * stride;
        if (tid == 0) *s_count = max_pg;
        __syncthreads();
        return;
    }

    // Each thread checks one page (max_pg <= blockDim.x)
    uint32_t pg = first + tid * stride;
    bool is_active = (tid < max_pg && pg < total_pages) && d_mask[pg];

    // Warp-level vote + intra-warp prefix count
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;
    uint32_t ballot      = __ballot_sync(0xffffffff, is_active);
    uint32_t lane_prefix = __popc(ballot & ((1u << lane) - 1));
    uint32_t warp_cnt    = __popc(ballot);

    // Collect per-warp active counts
    __shared__ uint32_t s_wpfx[32];
    if (lane == 0) s_wpfx[warp_id] = warp_cnt;
    __syncthreads();

    // Exclusive prefix sum across 32 warps (thread 0, 32 iters on smem)
    if (tid == 0) {
        uint32_t sum = 0;
        for (uint32_t w = 0; w < 32; w++) {
            uint32_t c = s_wpfx[w];
            s_wpfx[w] = sum;
            sum += c;
        }
        *s_count = sum;
    }
    __syncthreads();

    // Scatter: order-preserving write
    if (is_active)
        s_active[s_wpfx[warp_id] + lane_prefix] = pg;
    __syncthreads();
}

// Chunked version: works for any blockDim.x (must be a multiple of 32).
// Processes pages in chunks of blockDim.x with warp-ballot compaction.
// Same semantics and signature as zonemap_compact_block_pages().
__device__ inline void zonemap_compact_block_pages_chunked(
    const uint8_t* __restrict__ d_mask,
    uint32_t total_pages,
    uint32_t* s_active,
    uint32_t* s_count)
{
    const uint32_t tid    = threadIdx.x;
    const uint32_t stride = gridDim.x;
    const uint32_t first  = blockIdx.x;
    const uint32_t max_pg = (total_pages > first)
        ? (total_pages - first + stride - 1) / stride : 0;

    // Fast path: no mask -> fill identity in parallel
    if (!d_mask) {
        for (uint32_t i = tid; i < max_pg; i += blockDim.x)
            s_active[i] = first + i * stride;
        if (tid == 0) *s_count = max_pg;
        __syncthreads();
        return;
    }

    const uint32_t nwarps  = blockDim.x / 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane    = tid % 32;

    __shared__ uint32_t s_wpfx_c[32];
    __shared__ uint32_t s_chunk_base;

    if (tid == 0) s_chunk_base = 0;
    __syncthreads();

    for (uint32_t chunk_start = 0; chunk_start < max_pg;
         chunk_start += blockDim.x)
    {
        const uint32_t i  = chunk_start + tid;
        const uint32_t pg = first + i * stride;
        const bool is_active =
            (i < max_pg && pg < total_pages) && d_mask[pg];

        uint32_t ballot      = __ballot_sync(0xffffffff, is_active);
        uint32_t lane_prefix = __popc(ballot & ((1u << lane) - 1));
        uint32_t warp_cnt    = __popc(ballot);

        if (lane == 0) s_wpfx_c[warp_id] = warp_cnt;
        __syncthreads();

        if (tid == 0) {
            uint32_t base = s_chunk_base;
            uint32_t sum = 0;
            for (uint32_t w = 0; w < nwarps; w++) {
                uint32_t c = s_wpfx_c[w];
                s_wpfx_c[w] = base + sum;
                sum += c;
            }
            s_chunk_base = base + sum;
        }
        __syncthreads();

        if (is_active)
            s_active[s_wpfx_c[warp_id] + lane_prefix] = pg;
        __syncthreads();
    }

    if (tid == 0) *s_count = s_chunk_base;
    __syncthreads();
}
