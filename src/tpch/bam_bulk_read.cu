// bam_bulk_read.cu — BaM GPU-initiated bulk page reads
// Compiled as C++17 with separable compilation, linked to bam_io_device.

#include "bam_bulk_read.cuh"
#include "bam_io_device.cuh"
#include <cstdio>

// ────────────────────────────────────────────────────────
// Kernel: each CUDA block reads pages in round-robin from
// the descriptor array.  Pages are read into page_cache
// slots (one slot per block) and then copied to the final
// destination buffer.
// ────────────────────────────────────────────────────────
__global__ void bam_bulk_read_kernel(
    void*  d_ctrls,
    void*  d_pc,
    void*  pc_base_addr,
    const BamBulkReadDesc* __restrict__ d_descs,
    uint32_t ndescs,
    uint32_t page_size)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;

    for (uint32_t j = bid; j < ndescs; j += gridDim.x) {
        const BamBulkReadDesc& d = d_descs[j];

        // Phase 1: GPU-initiated NVMe read into page_cache slot (tid==0 only)
        if (tid == 0) {
            bam_io_read_page_device(d_ctrls, d_pc, d.lba, d.nblocks, bid, d.device);
        }
        __syncthreads();

        // Phase 2: Copy from page cache slot to destination (all threads, coalesced)
        const char* src = static_cast<const char*>(pc_base_addr)
                        + static_cast<uint64_t>(bid) * page_size;
        char* dst = d.dest;
        const uint32_t n4 = d.copy_bytes / 4;
        for (uint32_t i = tid; i < n4; i += nthreads) {
            reinterpret_cast<uint32_t*>(dst)[i] =
                reinterpret_cast<const uint32_t*>(src)[i];
        }
        // Handle trailing bytes (unlikely for page-aligned data)
        if (tid == 0) {
            for (uint32_t i = n4 * 4; i < d.copy_bytes; i++)
                dst[i] = src[i];
        }
        __syncthreads();
    }
}

// ────────────────────────────────────────────────────────
// Pipelined kernel: each block has K page_cache slots and
// keeps K outstanding NVMe reads in flight.  This increases
// NVMe queue depth from num_blocks to num_blocks×K, letting
// the NVMe controller overlap more reads internally.
//
// Template parameter K = slots_per_block (1,2,4).
// K=1 degenerates to the original serial read→copy loop.
// ────────────────────────────────────────────────────────
template<uint32_t K>
__global__ void bam_bulk_read_pipelined_kernel(
    void*  d_ctrls,
    void*  d_pc,
    void*  pc_base_addr,
    const BamBulkReadDesc* __restrict__ d_descs,
    uint32_t ndescs,
    uint32_t page_size)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;

    // Count how many descriptors this block handles
    uint32_t my_count = 0;
    for (uint32_t j = bid; j < ndescs; j += gridDim.x) my_count++;
    if (my_count == 0) return;

    // Per-slot IO state (only tid==0 uses these; registers persist across __syncthreads)
    void*    slot_qp[K];
    uint16_t slot_cid[K];

    const uint32_t eff_k = (K < my_count) ? K : my_count;

    // ── Prime: submit up to K reads ──
    if (tid == 0) {
        for (uint32_t s = 0; s < eff_k; s++) {
            uint32_t desc_idx = bid + s * gridDim.x;
            const BamBulkReadDesc& d = d_descs[desc_idx];
            uint32_t slot = bid * K + s;
            bam_io_submit_page_device(d_ctrls, d_pc, d.lba, d.nblocks,
                                      slot, d.device,
                                      &slot_qp[s], &slot_cid[s]);
        }
    }
    __syncthreads();

    // ── Steady-state loop: poll → copy → submit next ──
    for (uint32_t i = 0; i < my_count; i++) {
        const uint32_t s = i % K;
        const uint32_t desc_idx = bid + i * gridDim.x;
        const BamBulkReadDesc& d = d_descs[desc_idx];

        // 1. Wait for NVMe read completion on slot s
        if (tid == 0) {
            bam_io_poll_page_device(slot_qp[s], slot_cid[s]);
        }
        __syncthreads();

        // 2. Coalesced copy from page_cache slot to destination
        const char* src = static_cast<const char*>(pc_base_addr)
                        + static_cast<uint64_t>(bid * K + s) * page_size;
        char* dst = d.dest;
        const uint32_t n4 = d.copy_bytes / 4;
        for (uint32_t j = tid; j < n4; j += nthreads) {
            reinterpret_cast<uint32_t*>(dst)[j] =
                reinterpret_cast<const uint32_t*>(src)[j];
        }
        if (tid == 0) {
            for (uint32_t j = n4 * 4; j < d.copy_bytes; j++)
                dst[j] = src[j];
        }
        __syncthreads();  // copy must finish before slot reuse

        // 3. Submit next read into this slot (non-blocking)
        const uint32_t next_i = i + eff_k;
        if (tid == 0 && next_i < my_count) {
            uint32_t next_desc_idx = bid + next_i * gridDim.x;
            const BamBulkReadDesc& nd = d_descs[next_desc_idx];
            bam_io_submit_page_device(d_ctrls, d_pc, nd.lba, nd.nblocks,
                                      bid * K + s, nd.device,
                                      &slot_qp[s], &slot_cid[s]);
        }
        // No __syncthreads needed: next iteration's poll (tid==0 only)
        // will block until submit's read completes. Other threads
        // advance to the poll's __syncthreads and wait there.
    }
}

// ────────────────────────────────────────────────────────
// Zero-copy kernel: NVMe read only, no GPU copy.
// Each descriptor gets its own page_cache slot (slot = j).
// After completion, data at pc_base + j * page_size.
// 32 threads/block: only tid==0 issues IO, but 1 warp is
// the minimum for efficient scheduling.
// ────────────────────────────────────────────────────────
__global__ void bam_bulk_read_nocopy_kernel(
    void*  d_ctrls,
    void*  d_pc,
    const BamBulkReadDesc* __restrict__ d_descs,
    uint32_t ndescs,
    uint32_t slot_base)
{
    const uint32_t tid = threadIdx.x;
    for (uint32_t j = blockIdx.x; j < ndescs; j += gridDim.x) {
        if (tid == 0) {
            const BamBulkReadDesc& d = d_descs[j];
            bam_io_read_page_device(d_ctrls, d_pc,
                                    d.lba, d.nblocks, slot_base + j, d.device);
        }
        __syncthreads();
    }
}

// ────────────────────────────────────────────────────────
// Host API (legacy: per-call page_cache + sync)
// ────────────────────────────────────────────────────────
BamBulkReadResult bam_bulk_read(
    bam_ctrl_handle_t ctrl,
    const BamBulkReadDesc* descs,
    uint32_t ndescs,
    uint32_t page_size,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    if (ndescs == 0) return {0, 0};

    // Create page cache: one slot per CUDA block
    bam_io_page_cache_t pc = bam_io_page_cache_create(ctrl, page_size, num_blocks);
    void* d_ctrls   = bam_io_page_cache_get_d_ctrls(pc);
    void* d_pc      = bam_io_page_cache_get_d_pc_ptr(pc);
    void* pc_base   = bam_io_page_cache_get_base_addr(pc);

    // Copy descriptors to GPU
    BamBulkReadDesc* d_descs = nullptr;
    cudaMalloc(&d_descs, ndescs * sizeof(BamBulkReadDesc));
    cudaMemcpyAsync(d_descs, descs, ndescs * sizeof(BamBulkReadDesc),
                    cudaMemcpyHostToDevice, stream);

    // Launch kernel: 128 threads per block (4 warps, good for coalesced copy)
    bam_bulk_read_kernel<<<num_blocks, 128, 0, stream>>>(
        d_ctrls, d_pc, pc_base, d_descs, ndescs, page_size);

    cudaStreamSynchronize(stream);

    // Aggregate I/O stats
    uint64_t io_bytes = 0;
    for (uint32_t i = 0; i < ndescs; i++)
        io_bytes += static_cast<uint64_t>(descs[i].nblocks) * 512;

    cudaFree(d_descs);
    bam_io_page_cache_destroy(pc);

    return {ndescs, io_bytes};
}

// ────────────────────────────────────────────────────────
// Persistent context API (for pipelined execution)
// ────────────────────────────────────────────────────────
BamBulkReadCtx bam_bulk_read_ctx_create(
    bam_ctrl_handle_t ctrl,
    uint32_t page_size,
    uint32_t num_blocks,
    uint32_t max_descs,
    uint32_t slots_per_block)
{
    BamBulkReadCtx ctx{};
    ctx.page_size       = page_size;
    ctx.num_blocks      = num_blocks;
    ctx.max_descs       = max_descs;
    ctx.slots_per_block = slots_per_block;

    // Allocate num_blocks × slots_per_block page_cache pages so each block
    // can keep slots_per_block NVMe reads in flight simultaneously.
    const uint32_t total_slots = num_blocks * slots_per_block;
    ctx.pc      = bam_io_page_cache_create(ctrl, page_size, total_slots);
    ctx.d_ctrls = bam_io_page_cache_get_d_ctrls(ctx.pc);
    ctx.d_pc    = bam_io_page_cache_get_d_pc_ptr(ctx.pc);
    ctx.pc_base = bam_io_page_cache_get_base_addr(ctx.pc);

    for (int i = 0; i < 2; i++) {
        cudaMalloc(&ctx.d_descs[i], max_descs * sizeof(BamBulkReadDesc));
        cudaMallocHost(&ctx.h_descs[i], max_descs * sizeof(BamBulkReadDesc));
        cudaEventCreate(&ctx.h2d_done[i]);
    }

    return ctx;
}

void bam_bulk_read_ctx_destroy(BamBulkReadCtx& ctx)
{
    for (int i = 0; i < 2; i++) {
        if (ctx.d_descs[i]) { cudaFree(ctx.d_descs[i]);      ctx.d_descs[i] = nullptr; }
        if (ctx.h_descs[i]) { cudaFreeHost(ctx.h_descs[i]);   ctx.h_descs[i] = nullptr; }
        cudaEventDestroy(ctx.h2d_done[i]);
    }
    if (ctx.pc)      { bam_io_page_cache_destroy(ctx.pc); ctx.pc = nullptr; }
}

void bam_bulk_read_async(
    BamBulkReadCtx& ctx,
    uint32_t ndescs,
    int pipe_idx,
    cudaStream_t stream)
{
    if (ndescs == 0) return;

    // Async H2D from double-buffered pinned staging
    cudaMemcpyAsync(ctx.d_descs[pipe_idx], ctx.h_descs[pipe_idx],
                    ndescs * sizeof(BamBulkReadDesc),
                    cudaMemcpyHostToDevice, stream);
    cudaEventRecord(ctx.h2d_done[pipe_idx], stream);

    uint32_t num_blocks = (ndescs < ctx.num_blocks) ? ndescs : ctx.num_blocks;

    if (ctx.slots_per_block >= 4) {
        bam_bulk_read_pipelined_kernel<4><<<num_blocks, 128, 0, stream>>>(
            ctx.d_ctrls, ctx.d_pc, ctx.pc_base,
            ctx.d_descs[pipe_idx], ndescs, ctx.page_size);
    } else if (ctx.slots_per_block >= 2) {
        bam_bulk_read_pipelined_kernel<2><<<num_blocks, 128, 0, stream>>>(
            ctx.d_ctrls, ctx.d_pc, ctx.pc_base,
            ctx.d_descs[pipe_idx], ndescs, ctx.page_size);
    } else {
        bam_bulk_read_kernel<<<num_blocks, 128, 0, stream>>>(
            ctx.d_ctrls, ctx.d_pc, ctx.pc_base,
            ctx.d_descs[pipe_idx], ndescs, ctx.page_size);
    }
}

// ────────────────────────────────────────────────────────
// Zero-copy API: page_cache has one slot per descriptor.
// After completion, data for desc j at pc_base + j * slot_size.
// ────────────────────────────────────────────────────────
BamBulkReadCtx bam_bulk_read_nocopy_ctx_create(
    bam_ctrl_handle_t ctrl,
    uint32_t slot_size,
    uint32_t num_blocks,
    uint32_t max_descs)
{
    BamBulkReadCtx ctx{};
    ctx.page_size       = slot_size;
    ctx.num_blocks      = num_blocks;
    ctx.max_descs       = max_descs;
    ctx.slots_per_block = 0;  // sentinel: nocopy mode

    // One page_cache slot per descriptor
    ctx.pc      = bam_io_page_cache_create(ctrl, slot_size, max_descs);
    ctx.d_ctrls = bam_io_page_cache_get_d_ctrls(ctx.pc);
    ctx.d_pc    = bam_io_page_cache_get_d_pc_ptr(ctx.pc);
    ctx.pc_base = bam_io_page_cache_get_base_addr(ctx.pc);

    for (int i = 0; i < 2; i++) {
        cudaMalloc(&ctx.d_descs[i], max_descs * sizeof(BamBulkReadDesc));
        cudaMallocHost(&ctx.h_descs[i], max_descs * sizeof(BamBulkReadDesc));
        cudaEventCreate(&ctx.h2d_done[i]);
    }

    return ctx;
}

void bam_bulk_read_nocopy_async(
    BamBulkReadCtx& ctx,
    uint32_t ndescs,
    int pipe_idx,
    cudaStream_t stream)
{
    if (ndescs == 0) return;

    cudaMemcpyAsync(ctx.d_descs[pipe_idx], ctx.h_descs[pipe_idx],
                    ndescs * sizeof(BamBulkReadDesc),
                    cudaMemcpyHostToDevice, stream);
    cudaEventRecord(ctx.h2d_done[pipe_idx], stream);

    uint32_t num_blocks = (ndescs < ctx.num_blocks) ? ndescs : ctx.num_blocks;
    bam_bulk_read_nocopy_kernel<<<num_blocks, 32, 0, stream>>>(
        ctx.d_ctrls, ctx.d_pc, ctx.d_descs[pipe_idx], ndescs, /*slot_base=*/0);
}

// Per-field nocopy launch: reads a subset of already-uploaded descriptors.
// desc_offset: starting index into the on-GPU descriptor array.
// ndescs: number of descriptors for this field.
// slot_base: page_cache slot offset (= desc_offset for standard layout).
void bam_bulk_read_nocopy_field_async(
    BamBulkReadCtx& ctx,
    uint32_t desc_offset,
    uint32_t ndescs,
    uint32_t slot_base,
    int pipe_idx,
    cudaStream_t stream)
{
    if (ndescs == 0) return;
    uint32_t num_blocks = (ndescs < ctx.num_blocks) ? ndescs : ctx.num_blocks;
    bam_bulk_read_nocopy_kernel<<<num_blocks, 32, 0, stream>>>(
        ctx.d_ctrls, ctx.d_pc,
        ctx.d_descs[pipe_idx] + desc_offset,
        ndescs, slot_base);
}

// ────────────────────────────────────────────────────────
// Fused BaM Zonemap kernel: reads stats pages from NVMe
// and evaluates predicates in a single kernel launch.
// Pre-issue a single dummy IO to initialize BaM page_cache DMA registration.
__global__ void bam_pre_io_kernel(void* d_ctrls, void* d_pc)
{
    if (threadIdx.x == 0)
        bam_io_read_page_device(d_ctrls, d_pc, 0, 1, 0, 0);
}

void bam_pre_io(void* d_ctrls, void* d_pc, cudaStream_t stream)
{
    bam_pre_io_kernel<<<1, 1, 0, stream>>>(d_ctrls, d_pc);
    cudaStreamSynchronize(stream);
}

//
// Phase 1 (tid==0): read stats pages to page_cache slots.
// Phase 2 (all threads): AND-evaluate predicates, write mask.
// ────────────────────────────────────────────────────────
__global__ void bam_zonemap_fused_kernel(
    void* d_ctrls,
    void* d_pc,
    void* pc_base,
    const BamZonemapStatsRead* __restrict__ d_reads, uint32_t nreads,
    const BamZonemapPred* __restrict__ d_preds, uint32_t npreds,
    uint64_t npages,
    uint8_t* __restrict__ d_mask,
    uint32_t* __restrict__ d_active_ids,
    uint32_t* __restrict__ d_num_active,
    uint32_t page_size,
    // Phase 4 (optional): INT32 mask → INT64 mask derivation
    const uint64_t* __restrict__ d_ps_i32,   // nullptr = skip Phase 4
    const uint64_t* __restrict__ d_ps_i64,
    uint8_t* __restrict__ d_mask_i64,
    uint32_t npages_i64)
{
    const uint32_t tid = threadIdx.x;

    // Phase 1: GPU-initiated NVMe reads (parallel across threads)
    for (uint32_t r = tid; r < nreads; r += blockDim.x) {
        bam_io_read_page_device(d_ctrls, d_pc,
            d_reads[r].lba, d_reads[r].nblocks, r, d_reads[r].device);
    }
    __syncthreads();

    // Phase 2: evaluate all predicates (stride loop, AND-composed)
    for (uint64_t pg = tid; pg < npages; pg += blockDim.x) {
        uint8_t active = 1;
        for (uint32_t i = 0; i < npreds && active; i++) {
            if (pg >= d_preds[i].nstats) continue;
            const char* stats_base = static_cast<const char*>(pc_base)
                + static_cast<uint64_t>(d_preds[i].stats_page_offset) * page_size;
            const int32_t* s = reinterpret_cast<const int32_t*>(stats_base) + pg * 2;
            int32_t mn = s[0], mx = s[1];
            if (mn > mx || max(mn, d_preds[i].pred_lo) > min(mx, d_preds[i].pred_hi))
                active = 0;
        }
        d_mask[pg] = active;
    }

    // Phase 3: compact d_mask → d_active_ids (order-preserving warp-ballot)
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
        // Zero d_mask_i64
        for (uint32_t j = tid; j < npages_i64; j += blockDim.x)
            d_mask_i64[j] = 0;
        __syncthreads();
        // For each active INT32 page, mark overlapping INT64 pages
        for (uint64_t pg = tid; pg < npages; pg += blockDim.x) {
            if (!d_mask[pg]) continue;
            uint64_t base_row = (pg == 0) ? 0 : d_ps_i32[pg - 1];
            uint64_t end_row  = d_ps_i32[pg];
            if (end_row <= base_row) continue;
            uint64_t last_row = end_row - 1;
            // Binary search: find INT64 page containing base_row
            uint32_t lo = 0, hi = npages_i64;
            while (lo < hi) {
                uint32_t mid = lo + (hi - lo) / 2;
                if (d_ps_i64[mid] <= base_row) lo = mid + 1;
                else hi = mid;
            }
            uint32_t i64_lo = lo;
            // Binary search: find INT64 page containing last_row
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

// ────────────────────────────────────────────────────────
// Zonemap context lifecycle
// ────────────────────────────────────────────────────────
BamZonemapCtx bam_zonemap_ctx_create(
    void* d_ctrls, void* d_pc, void* pc_base,
    uint32_t page_size,
    uint64_t max_npages)
{
    BamZonemapCtx ctx{};
    ctx.page_size  = page_size;
    ctx.max_npages = max_npages;
    ctx.d_ctrls    = d_ctrls;
    ctx.d_pc       = d_pc;
    ctx.pc_base    = pc_base;

    cudaMalloc(&ctx.d_reads, kBamZonemapMaxReads * sizeof(BamZonemapStatsRead));
    cudaMallocHost(&ctx.h_reads, kBamZonemapMaxReads * sizeof(BamZonemapStatsRead));
    cudaMalloc(&ctx.d_preds, kBamZonemapMaxPreds * sizeof(BamZonemapPred));
    cudaMallocHost(&ctx.h_preds, kBamZonemapMaxPreds * sizeof(BamZonemapPred));
    cudaMalloc(&ctx.d_mask, max_npages);
    cudaMallocHost(&ctx.h_mask, max_npages);
    cudaMalloc(&ctx.d_active_ids, max_npages * sizeof(uint32_t));
    cudaMalloc(&ctx.d_num_active, sizeof(uint32_t));
    cudaMallocHost(&ctx.h_num_active, sizeof(uint32_t));

    return ctx;
}

void bam_zonemap_ctx_destroy(BamZonemapCtx& ctx)
{
    if (ctx.d_reads)  { cudaFree(ctx.d_reads);      ctx.d_reads  = nullptr; }
    if (ctx.h_reads)  { cudaFreeHost(ctx.h_reads);   ctx.h_reads  = nullptr; }
    if (ctx.d_preds)  { cudaFree(ctx.d_preds);      ctx.d_preds  = nullptr; }
    if (ctx.h_preds)  { cudaFreeHost(ctx.h_preds);   ctx.h_preds  = nullptr; }
    if (ctx.d_mask)   { cudaFree(ctx.d_mask);        ctx.d_mask   = nullptr; }
    if (ctx.h_mask)   { cudaFreeHost(ctx.h_mask);     ctx.h_mask   = nullptr; }
    if (ctx.d_active_ids)  { cudaFree(ctx.d_active_ids);       ctx.d_active_ids  = nullptr; }
    if (ctx.d_num_active)  { cudaFree(ctx.d_num_active);       ctx.d_num_active  = nullptr; }
    if (ctx.h_num_active)  { cudaFreeHost(ctx.h_num_active);   ctx.h_num_active  = nullptr; }
}

void bam_zonemap_eval_async(
    BamZonemapCtx& ctx,
    uint64_t npages,
    uint32_t nreads,
    uint32_t npreds,
    cudaStream_t stream)
{
    // H2D: upload reads and preds descriptors
    cudaMemcpyAsync(ctx.d_reads, ctx.h_reads,
                    nreads * sizeof(BamZonemapStatsRead),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ctx.d_preds, ctx.h_preds,
                    npreds * sizeof(BamZonemapPred),
                    cudaMemcpyHostToDevice, stream);

    // Single-block kernel: Phase 1 IO + Phase 2 eval + Phase 3 compact
    //                     + optional Phase 4 INT32→INT64 mask derivation
    bam_zonemap_fused_kernel<<<1, 256, 0, stream>>>(
        ctx.d_ctrls, ctx.d_pc, ctx.pc_base,
        ctx.d_reads, nreads,
        ctx.d_preds, npreds,
        npages, ctx.d_mask,
        ctx.d_active_ids, ctx.d_num_active,
        ctx.page_size,
        ctx.d_ps_i32, ctx.d_ps_i64, ctx.d_mask_i64, ctx.npages_i64);

    // D2H: download mask + active count
    cudaMemcpyAsync(ctx.h_mask, ctx.d_mask, npages,
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ctx.h_num_active, ctx.d_num_active, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
}


