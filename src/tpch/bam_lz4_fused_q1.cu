// bam_lz4_fused_q1.cu — Fused BaM I/O + nvCOMPdx LZ4 decompress + Q1 scan
// Balanced pipeline: 1 IO warp + 7 decomp warps (256 threads/block).
// IO warp reads all 7 fields sequentially; each decomp warp decompresses 1 field.
// __launch_bounds__(256, 4) → 4 blocks/SM for better SM utilization during IO polling.
// NBUF=2 ring buffer (double-buffered I/O); NFACE=2 double-buffered decomp output.
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q1.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "bam_io_device_q1.cuh"
#include "tpch/page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define FUSED_Q1_CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// Configuration constants (shared between kernel and host API)
static constexpr int FUSED_Q1_NBUF    = 2;    // double-buffered I/O ring
static constexpr int FUSED_Q1_NFACE   = 2;    // double-buffered decomp output

// ── BaM nblk alignment fix ──
__device__ static uint32_t fused_q1_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// ── I/O parameter computation helper ──
__device__ static void fused_q1_io_params(
    const BAMFusedQ1Params& p,
    uint32_t fi, uint64_t pg, uint32_t ndev,
    uint64_t& lba, uint32_t& nblk, uint32_t& dev, uint32_t& comp_sz)
{
    uint64_t global_pg = p.field_start_page_ids[fi] + pg;
    dev = global_pg % ndev;
    if (p.is_compressed[fi]) {
        lba = p.partition_start_lbas[dev] +
              p.d_comp_offsets[fi][pg] / 512;
        comp_sz = p.d_comp_sizes[fi][pg];
        nblk = fused_q1_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg = global_pg / ndev;
        lba = p.partition_start_lbas[dev] +
              local_pg * (p.page_size / 512);
        nblk = p.page_size / 512;
        comp_sz = p.page_size;
    }
}

// Q1 aggregation constants (must match q1.cuh)
constexpr int FQ1_NUM_GROUPS   = 6;   // 3 returnflag (A/N/R) x 2 linestatus (F/O)
constexpr int FQ1_NUM_AGGS     = 7;
constexpr int FQ1_LOCAL_AGGS   = 6;   // per-thread local aggs per group

// Aggregate indices in global d_agg
constexpr int FQ1_SUM_QTY        = 0;
constexpr int FQ1_SUM_BASE_PRICE = 1;
constexpr int FQ1_SUM_DISC_PRICE = 2;
constexpr int FQ1_SUM_CHARGE     = 3;
constexpr int FQ1_SUM_DISCOUNT   = 4;
constexpr int FQ1_COUNT          = 5;
constexpr int FQ1_SUM_CHARGE_HI  = 6;

// ── Q1 scan helper: 3-stage aggregation (thread register → block smem → global atomic) ──
// Reduces per-page atomicAdd count from ~6144 to 36 (6 groups × 6 aggs).
__device__ __forceinline__ static void fused_q1_scan_page(
    const char* base,       // decomp base for this face (7 fields × page_size)
    uint32_t page_size,
    uint32_t tid_in_group,
    int64_t* __restrict__ d_agg)
{
    constexpr uint32_t THREADS_PER_GROUP = 256;
    constexpr uint32_t WARPS_PER_GROUP   = THREADS_PER_GROUP / 32;  // 8
    constexpr uint32_t HDR_INT32 = 3;

    const int32_t* qty     = (const int32_t*)(base + 0 * (uint64_t)page_size) + HDR_INT32;
    const int32_t* eprice  = (const int32_t*)(base + 1 * (uint64_t)page_size) + HDR_INT32;
    const int32_t* disc    = (const int32_t*)(base + 2 * (uint64_t)page_size) + HDR_INT32;
    const int32_t* tax     = (const int32_t*)(base + 3 * (uint64_t)page_size) + HDR_INT32;
    const int32_t* rflag   = (const int32_t*)(base + 4 * (uint64_t)page_size) + HDR_INT32;
    const int32_t* lstatus = (const int32_t*)(base + 5 * (uint64_t)page_size) + HDR_INT32;
    const int32_t* sd      = (const int32_t*)(base + 6 * (uint64_t)page_size) + HDR_INT32;

    uint32_t nalloc = *(const uint32_t*)(base);

    int64_t local_agg[FQ1_NUM_GROUPS * FQ1_LOCAL_AGGS];
    for (int i = 0; i < FQ1_NUM_GROUPS * FQ1_LOCAL_AGGS; i++)
        local_agg[i] = 0;

    for (uint32_t r = tid_in_group; r < nalloc; r += THREADS_PER_GROUP) {
        int32_t shipdate = sd[r];
        if (shipdate > 19980902) continue;

        int32_t returnflag_val = rflag[r];
        int32_t linestatus_val = lstatus[r];
        char returnflag = (char)(uint8_t)returnflag_val;
        char linestatus = (char)(uint8_t)linestatus_val;

        int row;
        switch (returnflag) {
            case 'A': row = 0; break;
            case 'N': row = 1; break;
            case 'R': row = 2; break;
            default: continue;
        }
        int col = (linestatus == 'F') ? 0 : 1;
        int gid = row * 2 + col;

        int32_t quantity      = qty[r];
        int32_t extendedprice = eprice[r];
        int32_t discount      = disc[r];
        int32_t tax_val       = tax[r];

        int64_t disc_price = (int64_t)extendedprice * (int64_t)(100 - discount);
        int64_t charge     = disc_price * (int64_t)(100 + tax_val);

        int64_t *la = local_agg + gid * FQ1_LOCAL_AGGS;
        la[0] += quantity;
        la[1] += extendedprice;
        la[2] += disc_price;
        la[3] += charge;
        la[4] += discount;
        la[5] += 1;
    }

    // Stage 2: warp reduce (shfl_down) → lane 0 holds warp sum
    const uint32_t lane    = tid_in_group & 31;
    const uint32_t warp_id = tid_in_group >> 5;

    for (int i = 0; i < FQ1_NUM_GROUPS * FQ1_LOCAL_AGGS; i++) {
        int64_t v = local_agg[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        local_agg[i] = v;
    }

    // Stage 3: cross-warp reduce via smem (warp 0 sums all WARPS_PER_GROUP partial sums)
    __shared__ int64_t s_partials[WARPS_PER_GROUP][FQ1_NUM_GROUPS * FQ1_LOCAL_AGGS];

    if (lane == 0) {
        for (int i = 0; i < FQ1_NUM_GROUPS * FQ1_LOCAL_AGGS; i++) {
            s_partials[warp_id][i] = local_agg[i];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int i = lane; i < FQ1_NUM_GROUPS * FQ1_LOCAL_AGGS; i += 32) {
            int64_t sum = 0;
            #pragma unroll
            for (int w = 0; w < (int)WARPS_PER_GROUP; w++) {
                sum += s_partials[w][i];
            }
            // Stage 4: 1 atomicAdd per agg per block
            int g = i / FQ1_LOCAL_AGGS;
            int a = i % FQ1_LOCAL_AGGS;
            int64_t* ga = d_agg + g * FQ1_NUM_AGGS;
            if (sum == 0 && a != FQ1_COUNT) continue;
            switch (a) {
                case 0:
                    atomicAdd((unsigned long long*)&ga[FQ1_SUM_QTY],        (unsigned long long)sum);
                    break;
                case 1:
                    atomicAdd((unsigned long long*)&ga[FQ1_SUM_BASE_PRICE], (unsigned long long)sum);
                    break;
                case 2:
                    atomicAdd((unsigned long long*)&ga[FQ1_SUM_DISC_PRICE], (unsigned long long)sum);
                    break;
                case 3: {
                    unsigned long long old_lo = atomicAdd(
                        (unsigned long long*)&ga[FQ1_SUM_CHARGE], (unsigned long long)sum);
                    if (old_lo + (unsigned long long)sum < old_lo) {
                        atomicAdd((unsigned long long*)&ga[FQ1_SUM_CHARGE_HI], 1ULL);
                    }
                    break;
                }
                case 4:
                    atomicAdd((unsigned long long*)&ga[FQ1_SUM_DISCOUNT],   (unsigned long long)sum);
                    break;
                case 5:
                    if (sum != 0) {
                        atomicAdd((unsigned long long*)&ga[FQ1_COUNT],      (unsigned long long)sum);
                    }
                    break;
            }
        }
    }
    // caller __syncthreads() after scan_page() serves as barrier for next call
}

// ════════════════════════════════════════════════════════════════
// Fused Q1 kernel: Balanced pipeline (1 IO warp + 7 decomp warps)
//
// Motivation: t_decomp / t_io ≈ 11x (io_decomp_latency benchmark).
// Allocating 1 IO warp + 7 decomp warps per block (256 threads) and
// __launch_bounds__(256, 4) allows 4 blocks/SM. Different blocks at
// different pipeline stages let SM overlap IO polling with compute.
//
// Double-buffered I/O ring (NBUF=2): IO writes ring[1-io_ring] while
// decomp reads ring[io_ring]. Double-buffered decomp output (NFACE=2).
//
// Pipeline:
//   Priming: IO warp sync-reads first page → ring[0]
//   Main loop:
//     Phase A: IO warp sync-reads next page → ring[1-io_ring]  (overlaps)
//              Decomp warps decompress from ring[io_ring]       (parallel)
//     __syncthreads()
//     Phase B: All 8 warps scan face[1-decomp_face] (PREVIOUS page)
//     __syncthreads()
//   Epilogue: scan last page
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(256, 5)
void bam_lz4_fused_q1_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ1Params p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr int      NBUF       = FUSED_Q1_NBUF;         // 2
    constexpr int      NFACE      = FUSED_Q1_NFACE;         // 2
    constexpr uint32_t NUM_FIELDS = FUSED_Q1_NUM_FIELDS;    // 7
    constexpr uint32_t WARPS      = 8;                      // 1 IO + 7 decomp

    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid / 32;   // 0..7
    const uint32_t lane    = tid % 32;

    // ── Shared memory ──
    // Static: comp_sz communication from IO warp to decomp warps
    __shared__ uint32_t shared_comp_sz[NBUF][NUM_FIELDS];  // [ring][field]

    // Dynamic: nvCOMPdx per-warp region
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

#ifdef FQ1_PROFILE
    // Per-phase cycle accumulators (block-local)
    __shared__ unsigned long long s_cyc_io;
    __shared__ unsigned long long s_cyc_decomp;
    __shared__ unsigned long long s_cyc_scan;
    __shared__ unsigned long long s_cyc_iters;
    __shared__ unsigned long long s_cyc_total_start;
    if (tid == 0) {
        s_cyc_io = 0ULL;
        s_cyc_decomp = 0ULL;
        s_cyc_scan = 0ULL;
        s_cyc_iters = 0ULL;
        s_cyc_total_start = (unsigned long long)clock64();
    }
    __syncthreads();
#endif

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // ── Slot assignment: NBUF rings × NUM_FIELDS fields per block ──
    // slot(ring, field) = blockIdx.x * (NBUF * NUM_FIELDS) + ring * NUM_FIELDS + field
    constexpr uint32_t SLOTS_PER_BLOCK = NBUF * NUM_FIELDS;  // 14

    // ── Decomp buffer layout: [block][face][field] × page_size ──
    const uint64_t face_stride  = (uint64_t)NUM_FIELDS * p.page_size;  // 7 × page_size
    const uint64_t block_stride = (uint64_t)NFACE * face_stride;       // 14 × page_size
    const uint64_t block_base   = (uint64_t)blockIdx.x * block_stride;

    // ── Page assignment: stride over grid ──
    const uint64_t pg_stride = gridDim.x;
    const uint64_t pg_start  = blockIdx.x;

    // ── State ──
    int  io_ring     = 0;
    int  decomp_face = 0;
    bool prev_active = false;

    // ════════════════════════════════════════════════════════════
    // Priming: IO warp sync-reads all 7 fields of first page → ring[0]
    // ════════════════════════════════════════════════════════════
    if (pg_start < p.npages) {
        if (warp_id == 0) {
            for (uint32_t fi = 0; fi < NUM_FIELDS; fi++) {
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                fused_q1_io_params(p, fi, pg_start, ndev, lba, nblk, dev, comp_sz);
                if (lane == 0) {
                    bam_io_read_page_device_q1(ctrls, pc, lba, nblk,
                        blockIdx.x * SLOTS_PER_BLOCK + 0 * NUM_FIELDS + fi, dev);
                    shared_comp_sz[0][fi] = comp_sz;
                }
                __syncwarp();
            }
        }
    }
    __syncthreads();  // All warps wait for priming

    // ════════════════════════════════════════════════════════════
    // Main loop: double-buffered IO ring + decomp faces
    //
    // Phase A: IO reads next page into ring[1-io_ring],
    //          decomp warps decompress current page from ring[io_ring].
    // Phase B: All warps scan PREVIOUS page from face[1-decomp_face].
    // Epilogue: scan last page after loop ends.
    // ════════════════════════════════════════════════════════════
    for (uint64_t pg = pg_start; pg < p.npages; pg += pg_stride) {
        const bool active = !p.d_page_active || p.d_page_active[pg];
        const uint64_t next_pg = pg + pg_stride;
        const bool has_next = (next_pg < p.npages);

        // ── Phase A: IO(next) ∥ decomp(current) ──

#ifdef FQ1_PROFILE
        long long t_a_start = 0;
        if (tid == 0) t_a_start = clock64();

        // IO warp (warp 0): sync-read next page into ring[1-io_ring]
        long long t_io_start = 0, t_io_end = 0;
        if (warp_id == 0 && lane == 0) t_io_start = clock64();
#endif
        if (warp_id == 0 && has_next) {
            const int next_ring = 1 - io_ring;
            for (uint32_t fi = 0; fi < NUM_FIELDS; fi++) {
                uint64_t lba; uint32_t nblk, dev, comp_sz;
                fused_q1_io_params(p, fi, next_pg, ndev, lba, nblk, dev, comp_sz);
                if (lane == 0) {
                    bam_io_read_page_device_q1(ctrls, pc, lba, nblk,
                        blockIdx.x * SLOTS_PER_BLOCK + next_ring * NUM_FIELDS + fi, dev);
                    shared_comp_sz[next_ring][fi] = comp_sz;
                }
                __syncwarp();
            }
        }
#ifdef FQ1_PROFILE
        if (warp_id == 0 && lane == 0) {
            t_io_end = clock64();
            atomicAdd(&s_cyc_io, (unsigned long long)(t_io_end - t_io_start));
        }

        // Decomp warps (warps 1-7): decompress from ring[io_ring] → face[decomp_face]
        long long t_dec_start = 0, t_dec_end = 0;
        if (warp_id == 1 && lane == 0) t_dec_start = clock64();
#endif
        if (warp_id >= 1 && warp_id <= NUM_FIELDS && active) {
            const uint32_t fi = warp_id - 1;
            const uint32_t comp_sz = shared_comp_sz[io_ring][fi];
            const uint32_t slot = blockIdx.x * SLOTS_PER_BLOCK
                                + io_ring * NUM_FIELDS + fi;
            char* dst = d_decomp_buf + block_base
                      + (uint64_t)decomp_face * face_stride
                      + (uint64_t)fi * p.page_size;
            bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
                pc_base_addr, slot, dst, comp_sz, p.page_size, my_smem);
        }
#ifdef FQ1_PROFILE
        if (warp_id == 1 && lane == 0) {
            t_dec_end = clock64();
            atomicAdd(&s_cyc_decomp, (unsigned long long)(t_dec_end - t_dec_start));
        }
#endif

        __syncthreads();  // IO + decomp done

        // ── Phase B: scan PREVIOUS page's decompressed data ──
#ifdef FQ1_PROFILE
        long long t_scan_start = 0, t_scan_end = 0;
        if (tid == 0) t_scan_start = clock64();
#endif
        if (prev_active) {
            const uint64_t read_base = block_base
                                     + (uint64_t)(1 - decomp_face) * face_stride;
            fused_q1_scan_page(d_decomp_buf + read_base, p.page_size,
                               tid, p.d_agg);
        }

        __syncthreads();  // Scan done before next iteration
#ifdef FQ1_PROFILE
        if (tid == 0) {
            t_scan_end = clock64();
            atomicAdd(&s_cyc_scan, (unsigned long long)(t_scan_end - t_scan_start));
            s_cyc_iters++;
        }
#endif

        prev_active = active;
        io_ring     = 1 - io_ring;
        decomp_face = 1 - decomp_face;
    }

    // ════════════════════════════════════════════════════════════
    // Epilogue: scan the last decompressed page
    // ════════════════════════════════════════════════════════════
#ifdef FQ1_PROFILE
    long long t_epi_start = 0;
    if (tid == 0) t_epi_start = clock64();
#endif
    if (prev_active) {
        const uint64_t read_base = block_base
                                 + (uint64_t)(1 - decomp_face) * face_stride;
        fused_q1_scan_page(d_decomp_buf + read_base, p.page_size,
                           tid, p.d_agg);
    }
    __syncthreads();
#ifdef FQ1_PROFILE
    if (tid == 0) {
        unsigned long long t_epi_end = (unsigned long long)clock64();
        atomicAdd(&s_cyc_scan, t_epi_end - (unsigned long long)t_epi_start);
    }

    // Flush block totals to global cycle accumulators
    if (p.d_cycles && tid == 0) {
        unsigned long long total = (unsigned long long)clock64() - s_cyc_total_start;
        atomicAdd(&p.d_cycles[0], s_cyc_io);
        atomicAdd(&p.d_cycles[1], s_cyc_decomp);
        atomicAdd(&p.d_cycles[2], s_cyc_scan);
        atomicAdd(&p.d_cycles[3], s_cyc_iters);
        atomicAdd(&p.d_cycles[4], total);
    }
#endif
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

struct BAMFusedQ1Context {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;       // [num_blocks * NFACE * 7 * page_size]
    uint32_t    page_size;
    uint32_t    num_blocks;
};

bam_fused_q1_ctx_t bam_fused_q1_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* ctx = new BAMFusedQ1Context();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // Balanced pipeline: NBUF rings × 7 fields = 14 page_cache slots per block
    const uint32_t num_slots = num_blocks * FUSED_Q1_NBUF * FUSED_Q1_NUM_FIELDS;

    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: NFACE × 7 fields per block (no NGROUPS)
    size_t decomp_size = (size_t)num_blocks * FUSED_Q1_NFACE
                       * FUSED_Q1_NUM_FIELDS * page_size;
    FUSED_Q1_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    return static_cast<bam_fused_q1_ctx_t>(ctx);
}

static void bam_fused_q1_launch(
    BAMFusedQ1Context* ctx,
    const BAMFusedQ1Params& p,
    cudaStream_t stream)
{
    // Balanced pipeline: 8 warps (1 IO + 7 decomp) = 256 threads
    constexpr uint32_t WARPS   = 8;
    constexpr uint32_t THREADS = WARPS * 32;  // 256

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * WARPS;
        auto kernel_fn = bam_lz4_fused_q1_kernel<PS>;
        FUSED_Q1_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });

    FUSED_Q1_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q1_run_async(
    bam_fused_q1_ctx_t ctx_handle,
    const BAMFusedQ1Params& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ1Context*>(ctx_handle);
    bam_fused_q1_launch(ctx, params, stream);
}

void bam_fused_q1_destroy(bam_fused_q1_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ1Context*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}
