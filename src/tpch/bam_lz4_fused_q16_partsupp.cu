// bam_lz4_fused_q16_partsupp.cu — Fused BaM I/O + nvCOMPdx LZ4 decomp + Q16 PARTSUPP probe
// 4 warps/block, each warp independently handles IO+decomp+probe for page pairs.
// Compiled as CUDA C++17 with separable compilation + device linking.

#include "bam_lz4_fused_q16_partsupp.cuh"
#include "bam_lz4_io_decomp.cuh"
#include "tpch/page_size_dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define FUSED_Q16PS_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                cudaGetErrorString(err), __FILE__, __LINE__);                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

__device__ static uint32_t fused_q16ps_fix_nblk(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 24;
    return nblk;
}

// Hash function — must match hash64 used in q16_scan.cu
__device__ static uint32_t fused_q16ps_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

static constexpr uint64_t FUSED_Q16PS_HT_EMPTY = UINT64_MAX;

// Probe PART HT: returns group_id or UINT32_MAX
__device__ static uint32_t fused_q16ps_ht_probe(
    const uint64_t *keys, const uint32_t *group_ids,
    uint32_t mask, uint64_t key)
{
    uint32_t slot = fused_q16ps_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return group_ids[slot];
        if (k == FUSED_Q16PS_HT_EMPTY) return UINT32_MAX;
        slot = (slot + 1) & mask;
    }
}

// Probe excluded suppkey set: returns true if excluded
__device__ static bool fused_q16ps_excl_probe(
    const uint64_t *keys, uint32_t mask, uint64_t key)
{
    uint32_t slot = fused_q16ps_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return true;
        if (k == FUSED_Q16PS_HT_EMPTY) return false;
        slot = (slot + 1) & mask;
    }
}

// Decomp buffer: 8 pages per block (4 warps × 2 pages each: PS_PARTKEY + PS_SUPPKEY)
static constexpr uint32_t Q16PS_DECOMP_PAGES_PER_BLOCK = 8;

// Split IO/Compute kernel configuration
static constexpr uint32_t Q16PS_SPLIT_WARPS   = 4;
static constexpr uint32_t Q16PS_SPLIT_THREADS = Q16PS_SPLIT_WARPS * 32;  // 128

// ════════════════════════════════════════════════════════════════
// Fused Q16 PARTSUPP kernel — warp-stride, fully independent warps
// Uses nvCOMPdx warp-level LZ4 decompress.
// Each warp independently: IO+decomp PS_PARTKEY → IO+decomp PS_SUPPKEY → probe
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(128, 8)
void bam_lz4_fused_q16ps_kernel(
    void*       ctrls,
    void*       pc,
    const char* pc_base_addr,
    char*       d_decomp_buf,
    BAMFusedQ16PSParams p)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS = 4;

    const uint32_t lane    = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t global_warp = blockIdx.x * WARPS + warp_id;
    const uint32_t slot    = global_warp;

    // Per-warp shared memory for nvCOMPdx
    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Each warp has 2 private decomp pages: [partkey_page, suppkey_page]
    const uint64_t warp_base = (uint64_t)global_warp * 2 * p.page_size;
    char* my_pk_buf = d_decomp_buf + warp_base;
    char* my_sk_buf = d_decomp_buf + warp_base + p.page_size;

    // Page cache slot base address (reused for both fields)
    const char* slot_src = pc_base_addr + (uint64_t)slot * p.page_size;

    // Helper: compute LBA/nblk for an INT64 field
    auto compute_field = [&](uint32_t fi, uint32_t pg,
                             uint64_t &lba, uint32_t &nblk, uint32_t &dev, uint32_t &comp_sz) {
        uint64_t global_pg = p.field_start_page_ids[fi] + pg;
        dev = global_pg % ndev;
        if (p.is_compressed[fi]) {
            lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
            comp_sz = p.d_comp_sizes[fi][pg];
            nblk = fused_q16ps_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
        } else {
            uint64_t local_pg = global_pg / ndev;
            lba = p.partition_start_lbas[dev] + local_pg * (p.page_size / 512);
            nblk = p.page_size / 512;
            comp_sz = p.page_size;
        }
    };

    // Helper: warp-cooperative copy (4-byte granularity)
    auto warp_copy = [&](const char* src, char* dst, uint32_t nbytes) {
        const uint32_t n4 = (nbytes + 3) / 4;
        for (uint32_t i = lane; i < n4; i += 32)
            reinterpret_cast<uint32_t*>(dst)[i] =
                reinterpret_cast<const uint32_t*>(src)[i];
    };

    const uint32_t total_warps = gridDim.x * WARPS;

    // Warp-stride loop: each warp processes independent page-pairs
    for (uint32_t pg = global_warp; pg < p.npages; pg += total_warps) {
        uint64_t lba_pk, lba_sk;
        uint32_t nblk_pk, nblk_sk, dev_pk, dev_sk, comp_sz_pk, comp_sz_sk;

        // Compute IO parameters for both fields
        compute_field(0, pg, lba_pk, nblk_pk, dev_pk, comp_sz_pk);
        compute_field(1, pg, lba_sk, nblk_sk, dev_sk, comp_sz_sk);

        // ── Step 1: IO PS_PARTKEY (sync, into slot) ──
        if (lane == 0)
            bam_io_read_page_device(ctrls, pc, lba_pk, nblk_pk, slot, dev_pk);
        __syncwarp();

        // ── Step 2: Copy PK compressed data from slot → sk_buf (staging) ──
        uint32_t pk_copy_sz = (comp_sz_pk < p.page_size) ? comp_sz_pk : p.page_size;
        warp_copy(slot_src, my_sk_buf, pk_copy_sz);

        // ── Step 3: Submit IO PS_SUPPKEY (async, reuses same slot) ──
        void* qp_sk = nullptr;
        uint16_t cid_sk = 0;
        if (lane == 0)
            bam_io_submit_page_device(ctrls, pc, lba_sk, nblk_sk, slot, dev_sk, &qp_sk, &cid_sk);

        // ── Step 4: Decomp PK from sk_buf (staging) → pk_buf ──
        //            (overlaps with IO SK in NVMe pipeline)
        if (comp_sz_pk < p.page_size) {
            auto decompressor = lz4_decomp_t();
            size_t dsz = 0;
            decompressor.execute(my_sk_buf, my_pk_buf,
                                 (size_t)comp_sz_pk, &dsz, my_smem, nullptr);
        } else {
            warp_copy(my_sk_buf, my_pk_buf, p.page_size);
        }

        // ── Step 5: Poll IO SK completion ──
        if (lane == 0)
            bam_io_poll_page_device(qp_sk, cid_sk);
        __syncwarp();

        // ── Step 6: Decomp SK from slot → sk_buf ──
        bam_lz4_decomp_only_warp<PAGE_SIZE_CONST>(
            pc_base_addr, slot, my_sk_buf,
            comp_sz_sk, p.page_size, my_smem);

        // Probe (32 threads per warp)
        uint32_t nalloc = *(const uint32_t*)my_pk_buf;
        const uint64_t* pk = (const uint64_t*)(my_pk_buf + 16);
        const uint64_t* sk = (const uint64_t*)(my_sk_buf + 16);
        uint64_t row_base = p.d_ps[pg];

        for (uint32_t r = lane; r < nalloc; r += 32) {
            uint64_t partkey = pk[r];
            uint64_t suppkey = sk[r];

            uint32_t group_id = fused_q16ps_ht_probe(
                p.d_ht_keys, p.d_ht_group_ids, p.ht_mask, partkey);
            if (group_id == UINT32_MAX) {
                p.d_emit_pairs[row_base + r] = FUSED_Q16PS_HT_EMPTY;
                continue;
            }

            if (fused_q16ps_excl_probe(p.d_excl_keys, p.excl_mask, suppkey)) {
                p.d_emit_pairs[row_base + r] = FUSED_Q16PS_HT_EMPTY;
                continue;
            }

            p.d_emit_pairs[row_base + r] = ((uint64_t)group_id << 32) | (uint64_t)(uint32_t)suppkey;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Host API
// ════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════
// Split IO/Compute — IO queue entry
// ════════════════════════════════════════════════════════════════
struct Q16PSSplitIoEntry {
    void*    qp;        // 8B  QueuePair* for polling
    uint16_t cid;       // 2B  NVMe command ID
    uint16_t pad;       // 2B
    uint32_t comp_sz;   // 4B  compressed size (for decomp)
};  // 16B

// ════════════════════════════════════════════════════════════════
// Split IO Submit kernel — 1 thread per IO, lightweight
// ════════════════════════════════════════════════════════════════
__global__ void q16ps_split_submit_kernel(
    void*       ctrls,
    void*       pc,
    Q16PSSplitIoEntry* queue,
    uint32_t    batch_npages,
    uint32_t    pg_base,
    uint32_t    slot_base,   // page_cache slot offset (for double buffering)
    uint32_t    qp_base,     // QP range offset (buf * n_qps/NPIPE, avoids cross-batch CQ mixing)
    uint32_t    qp_range,    // QPs per buffer (n_qps/NPIPE), for modular wrapping
    uint64_t    field_start_0, uint64_t field_start_1,
    const uint64_t* d_comp_offsets_0, const uint64_t* d_comp_offsets_1,
    const uint32_t* d_comp_sizes_0,   const uint32_t* d_comp_sizes_1,
    bool        is_compressed_0, bool is_compressed_1,
    uint64_t    part_lba_0, uint64_t part_lba_1,
    uint64_t    part_lba_2, uint64_t part_lba_3,
    uint32_t    n_devices,
    uint32_t    page_size)
{
    const uint32_t total_ios = batch_npages * 2;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_ios) return;

    // Interleaved layout: [PK0,SK0,PK1,SK1,...] so PK→even QPs, SK→odd QPs.
    // This prevents PK/SK from sharing a CQ, avoiding CQ head-of-line deadlock.
    const uint32_t local_pg = idx / 2;
    const uint32_t field = idx & 1;   // 0=PK, 1=SK
    const uint32_t pg = pg_base + local_pg;

    const uint64_t field_start = (field == 0) ? field_start_0 : field_start_1;
    const uint64_t* comp_offsets = (field == 0) ? d_comp_offsets_0 : d_comp_offsets_1;
    const uint32_t* comp_sizes = (field == 0) ? d_comp_sizes_0 : d_comp_sizes_1;
    const bool is_compressed = (field == 0) ? is_compressed_0 : is_compressed_1;

    const uint64_t part_lbas[4] = {part_lba_0, part_lba_1, part_lba_2, part_lba_3};
    const uint32_t ndev = (n_devices > 1) ? n_devices : 1;

    uint64_t global_pg = field_start + pg;
    uint32_t dev = global_pg % ndev;
    uint64_t lba;
    uint32_t nblk;
    uint32_t comp_sz;

    if (is_compressed) {
        lba = part_lbas[dev] + comp_offsets[pg] / 512;
        comp_sz = comp_sizes[pg];
        nblk = fused_q16ps_fix_nblk(((comp_sz + 4095u) & ~4095u) / 512);
    } else {
        uint64_t local_pg_dev = global_pg / ndev;
        lba = part_lbas[dev] + local_pg_dev * (page_size / 512);
        nblk = page_size / 512;
        comp_sz = page_size;
    }

    void* qp_out = nullptr;
    uint16_t cid_out = 0;
    // pc_slot: unique page_cache slot for DMA address
    // qp_hint: qp_base + idx ensures each pipeline buffer uses distinct QP range
    bam_io_submit_page_device_qp(ctrls, pc, lba, nblk,
                                  slot_base + idx, qp_base + (idx % qp_range), dev,
                                  &qp_out, &cid_out);

    queue[idx].qp = qp_out;
    queue[idx].cid = cid_out;
    queue[idx].pad = 0;
    queue[idx].comp_sz = comp_sz;
}

// ════════════════════════════════════════════════════════════════
// Split Compute kernel — poll + nvCOMPdx LZ4 decomp + optional HT probe
// DO_PROBE=false: decomp only (for separate probe kernel)
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST, bool DO_PROBE = true>
__global__ __launch_bounds__(Q16PS_SPLIT_THREADS)
void q16ps_split_compute_kernel(
    const char*              pc_base_addr,
    char*                    d_decomp_buf,
    const Q16PSSplitIoEntry* __restrict__ queue,
    BAMFusedQ16PSParams      p,
    uint32_t                 batch_npages,
    uint32_t                 pg_base,
    uint32_t                 slot_base)   // page_cache slot offset (for double buffering)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS = Q16PS_SPLIT_WARPS;

    const uint32_t lane    = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t global_warp = blockIdx.x * WARPS + warp_id;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t total_warps = gridDim.x * WARPS;

    char* my_pk_buf = d_decomp_buf + (uint64_t)global_warp * 2 * PAGE_SIZE_CONST;
    char* my_sk_buf = my_pk_buf + PAGE_SIZE_CONST;

    for (uint32_t local_pg = global_warp; local_pg < batch_npages; local_pg += total_warps) {
        // Interleaved layout: PK at even slots, SK at odd slots
        const uint32_t pk_idx = local_pg * 2;
        const uint32_t sk_idx = local_pg * 2 + 1;

        // Poll + Decomp PK
        if (lane == 0)
            bam_io_poll_page_device(queue[pk_idx].qp, queue[pk_idx].cid);
        __syncwarp();

        {
            uint32_t comp_sz = queue[pk_idx].comp_sz;
            // slot_base offsets into page_cache for double buffering
            const char* src = pc_base_addr + (uint64_t)(slot_base + pk_idx) * PAGE_SIZE_CONST;
            if (comp_sz < PAGE_SIZE_CONST) {
                auto decompressor = lz4_decomp_t();
                size_t dsz = 0;
                decompressor.execute(src, my_pk_buf, (size_t)comp_sz, &dsz, my_smem, nullptr);
            } else {
                const uint32_t n4 = PAGE_SIZE_CONST / 4;
                for (uint32_t i = lane; i < n4; i += 32)
                    reinterpret_cast<uint32_t*>(my_pk_buf)[i] =
                        reinterpret_cast<const uint32_t*>(src)[i];
            }
        }

        // Poll + Decomp SK
        if (lane == 0)
            bam_io_poll_page_device(queue[sk_idx].qp, queue[sk_idx].cid);
        __syncwarp();

        {
            uint32_t comp_sz = queue[sk_idx].comp_sz;
            const char* src = pc_base_addr + (uint64_t)(slot_base + sk_idx) * PAGE_SIZE_CONST;
            if (comp_sz < PAGE_SIZE_CONST) {
                auto decompressor = lz4_decomp_t();
                size_t dsz = 0;
                decompressor.execute(src, my_sk_buf, (size_t)comp_sz, &dsz, my_smem, nullptr);
            } else {
                const uint32_t n4 = PAGE_SIZE_CONST / 4;
                for (uint32_t i = lane; i < n4; i += 32)
                    reinterpret_cast<uint32_t*>(my_sk_buf)[i] =
                        reinterpret_cast<const uint32_t*>(src)[i];
            }
        }

        // Probe (skipped when DO_PROBE=false for separate probe kernel)
        if constexpr (DO_PROBE) {
            const uint32_t pg = pg_base + local_pg;
            uint32_t nalloc = *(const uint32_t*)my_pk_buf;
            const uint64_t* pk = (const uint64_t*)(my_pk_buf + 16);
            const uint64_t* sk = (const uint64_t*)(my_sk_buf + 16);
            uint64_t row_base = p.d_ps[pg];

            for (uint32_t r = lane; r < nalloc; r += 32) {
                uint64_t partkey = pk[r];
                uint64_t suppkey = sk[r];

                uint32_t group_id = fused_q16ps_ht_probe(
                    p.d_ht_keys, p.d_ht_group_ids, p.ht_mask, partkey);
                if (group_id == UINT32_MAX) {
                    p.d_emit_pairs[row_base + r] = FUSED_Q16PS_HT_EMPTY;
                    continue;
                }

                if (fused_q16ps_excl_probe(p.d_excl_keys, p.excl_mask, suppkey)) {
                    p.d_emit_pairs[row_base + r] = FUSED_Q16PS_HT_EMPTY;
                    continue;
                }

                p.d_emit_pairs[row_base + r] = ((uint64_t)group_id << 32) | (uint64_t)(uint32_t)suppkey;
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Independent decomp kernel — each warp handles one field-page
// Total jobs = npages * 2 (PK and SK treated independently).
// Queue layout: [PK0, SK0, PK1, SK1, ...] → job j maps to queue[j].
// Output: d_decomp_buf[j * PAGE_SIZE] for job j.
// ════════════════════════════════════════════════════════════════
template <unsigned int PAGE_SIZE_CONST>
__global__ __launch_bounds__(Q16PS_SPLIT_THREADS)
void q16ps_indep_decomp_kernel(
    const char*              pc_base_addr,
    char*                    d_decomp_buf,
    const Q16PSSplitIoEntry* __restrict__ queue,
    uint32_t                 total_jobs,    // npages * 2
    uint32_t                 slot_base)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    constexpr uint32_t WARPS = Q16PS_SPLIT_WARPS;

    const uint32_t lane    = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t global_warp = blockIdx.x * WARPS + warp_id;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_decomp_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    const uint32_t total_warps = gridDim.x * WARPS;

    for (uint32_t job = global_warp; job < total_jobs; job += total_warps) {
        // Poll for IO completion
        if (lane == 0)
            bam_io_poll_page_device(queue[job].qp, queue[job].cid);
        __syncwarp();

        uint32_t comp_sz = queue[job].comp_sz;
        const char* src = pc_base_addr + (uint64_t)(slot_base + job) * PAGE_SIZE_CONST;
        char*       dst = d_decomp_buf + (uint64_t)job * PAGE_SIZE_CONST;

        if (comp_sz < PAGE_SIZE_CONST) {
            auto decompressor = lz4_decomp_t();
            size_t dsz = 0;
            decompressor.execute(src, dst, (size_t)comp_sz, &dsz, my_smem, nullptr);
        } else {
            const uint32_t n4 = PAGE_SIZE_CONST / 4;
            for (uint32_t i = lane; i < n4; i += 32)
                reinterpret_cast<uint32_t*>(dst)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Probe-only kernel — 128 threads/block, reads from decomp buffer
// Separate from IO+decomp for full-block parallelism on HT probe.
// ════════════════════════════════════════════════════════════════
__global__ __launch_bounds__(128)
void q16ps_probe_kernel(
    const char*         d_decomp_buf,
    BAMFusedQ16PSParams p,
    uint32_t            page_size_val)
{
    const uint32_t tid = threadIdx.x;

    for (uint32_t pg = blockIdx.x; pg < p.npages; pg += gridDim.x) {
        const char* pk_buf = d_decomp_buf + (uint64_t)pg * 2 * page_size_val;
        const char* sk_buf = pk_buf + page_size_val;

        uint32_t nalloc = *(const uint32_t*)pk_buf;
        const uint64_t* pk = (const uint64_t*)(pk_buf + 16);
        const uint64_t* sk = (const uint64_t*)(sk_buf + 16);
        uint64_t row_base = p.d_ps[pg];

        for (uint32_t r = tid; r < nalloc; r += 128) {
            uint64_t partkey = pk[r];
            uint64_t suppkey = sk[r];

            uint32_t group_id = fused_q16ps_ht_probe(
                p.d_ht_keys, p.d_ht_group_ids, p.ht_mask, partkey);
            if (group_id == UINT32_MAX) {
                p.d_emit_pairs[row_base + r] = FUSED_Q16PS_HT_EMPTY;
                continue;
            }

            if (fused_q16ps_excl_probe(p.d_excl_keys, p.excl_mask, suppkey)) {
                p.d_emit_pairs[row_base + r] = FUSED_Q16PS_HT_EMPTY;
                continue;
            }

            p.d_emit_pairs[row_base + r] = ((uint64_t)group_id << 32) | (uint64_t)(uint32_t)suppkey;
        }
    }
}

static constexpr uint32_t Q16PS_NPIPE = 2;  // double buffering

struct BAMFusedQ16PSContext {
    bam_io_page_cache_t io_pc;
    void*       d_ctrls;
    void*       d_pc_ptr;
    const char* pc_base_addr;
    char*       d_decomp_buf;
    uint32_t    page_size;
    uint32_t    num_blocks;
    // Split mode: double-buffered resources
    Q16PSSplitIoEntry* d_queue[Q16PS_NPIPE];
    cudaEvent_t        submit_done[Q16PS_NPIPE];
    cudaEvent_t        compute_done[Q16PS_NPIPE];
};

bam_fused_q16ps_ctx_t bam_fused_q16ps_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    // Print smem per warp for tuning
    dispatch_page_size(page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem = bam_lz4_io_decomp_smem_per_warp<PS>();
        fprintf(stderr, "[Q16PS] nvCOMPdx smem/warp=%zu B (page_size=%u), "
                "A100 164KB → max %zu warps/block\n", smem, page_size, 164*1024 / smem);
    });

    auto* ctx = new BAMFusedQ16PSContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    const uint32_t num_slots = num_blocks * 4;
    ctx->io_pc = bam_io_page_cache_create(ctrl_handle, page_size, num_slots);

    ctx->d_ctrls      = bam_io_page_cache_get_d_ctrls(ctx->io_pc);
    ctx->d_pc_ptr     = bam_io_page_cache_get_d_pc_ptr(ctx->io_pc);
    ctx->pc_base_addr = (const char*)bam_io_page_cache_get_base_addr(ctx->io_pc);

    // Decomp buffer: Q16PS_DECOMP_PAGES_PER_BLOCK pages per block
    size_t decomp_size = (size_t)num_blocks * Q16PS_DECOMP_PAGES_PER_BLOCK * page_size;
    FUSED_Q16PS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf, decomp_size));

    // Split mode: double-buffered IO queues + events
    const uint32_t half_slots = num_slots / Q16PS_NPIPE;
    for (uint32_t i = 0; i < Q16PS_NPIPE; i++) {
        FUSED_Q16PS_CUDA_CHECK(cudaMalloc(&ctx->d_queue[i], half_slots * sizeof(Q16PSSplitIoEntry)));
        FUSED_Q16PS_CUDA_CHECK(cudaEventCreate(&ctx->submit_done[i]));
        FUSED_Q16PS_CUDA_CHECK(cudaEventCreate(&ctx->compute_done[i]));
    }

    return static_cast<bam_fused_q16ps_ctx_t>(ctx);
}

static void bam_fused_q16ps_launch(
    BAMFusedQ16PSContext* ctx,
    const BAMFusedQ16PSParams& p,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = 128;
    constexpr uint32_t WARPS   = 4;

    dispatch_page_size(p.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * WARPS;
        auto kernel_fn = bam_lz4_fused_q16ps_kernel<PS>;
        FUSED_Q16PS_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<p.num_blocks, THREADS, smem_size, stream>>>(
            ctx->d_ctrls, ctx->d_pc_ptr, ctx->pc_base_addr,
            ctx->d_decomp_buf, p);
    });
    FUSED_Q16PS_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q16ps_run_async(
    bam_fused_q16ps_ctx_t ctx_handle,
    const BAMFusedQ16PSParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMFusedQ16PSContext*>(ctx_handle);
    bam_fused_q16ps_launch(ctx, params, stream);
}

// Helper: launch submit kernel for one batch
static void q16ps_launch_submit(
    BAMFusedQ16PSContext* ctx,
    const BAMFusedQ16PSParams& params,
    uint32_t buf, uint32_t batch_np, uint32_t pg_base,
    uint32_t slot_base, uint32_t qp_base, uint32_t qp_range,
    cudaStream_t stream_io)
{
    constexpr uint32_t TPB = 32;
    uint32_t total_ios = batch_np * 2;
    uint32_t grid = (total_ios + TPB - 1) / TPB;
    q16ps_split_submit_kernel<<<grid, TPB, 0, stream_io>>>(
        ctx->d_ctrls, ctx->d_pc_ptr, ctx->d_queue[buf],
        batch_np, pg_base, slot_base, qp_base, qp_range,
        params.field_start_page_ids[0], params.field_start_page_ids[1],
        params.d_comp_offsets[0], params.d_comp_offsets[1],
        params.d_comp_sizes[0], params.d_comp_sizes[1],
        params.is_compressed[0], params.is_compressed[1],
        params.partition_start_lbas[0], params.partition_start_lbas[1],
        params.partition_start_lbas[2], params.partition_start_lbas[3],
        params.n_devices, params.page_size);
    FUSED_Q16PS_CUDA_CHECK(cudaGetLastError());
}

// Helper: launch compute kernel for one batch
// DO_PROBE=true: decomp + probe (existing behavior)
// DO_PROBE=false: decomp only, requires 1 page per warp (for separate probe kernel)
template <bool DO_PROBE = true>
static void q16ps_launch_compute(
    BAMFusedQ16PSContext* ctx,
    const BAMFusedQ16PSParams& params,
    uint32_t buf, uint32_t batch_np, uint32_t pg_base,
    uint32_t slot_base,
    cudaStream_t stream_comp)
{
    constexpr uint32_t THREADS = Q16PS_SPLIT_THREADS;
    constexpr uint32_t WARPS   = Q16PS_SPLIT_WARPS;
    uint32_t eff_blocks;
    if constexpr (!DO_PROBE) {
        eff_blocks = (batch_np + WARPS - 1) / WARPS;
    } else {
        if (params.split_compute_blocks > 0)
            eff_blocks = params.split_compute_blocks;
        else
            eff_blocks = std::min(ctx->num_blocks, (batch_np + WARPS - 1) / WARPS);
    }
    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * WARPS;
        auto kernel_fn = q16ps_split_compute_kernel<PS, DO_PROBE>;
        FUSED_Q16PS_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<eff_blocks, THREADS, smem_size, stream_comp>>>(
            ctx->pc_base_addr, ctx->d_decomp_buf, ctx->d_queue[buf],
            params, batch_np, pg_base, slot_base);
    });
    FUSED_Q16PS_CUDA_CHECK(cudaGetLastError());
}

// Helper: launch independent decomp kernel (1 job per field-page, 2× parallelism)
static void q16ps_launch_indep_decomp(
    BAMFusedQ16PSContext* ctx,
    const BAMFusedQ16PSParams& params,
    uint32_t total_jobs,   // npages * 2
    uint32_t slot_base,
    cudaStream_t stream)
{
    constexpr uint32_t THREADS = Q16PS_SPLIT_THREADS;
    constexpr uint32_t WARPS   = Q16PS_SPLIT_WARPS;
    uint32_t eff_blocks = (total_jobs + WARPS - 1) / WARPS;
    dispatch_page_size(params.page_size, [&](auto ps_tag) {
        constexpr unsigned PS = decltype(ps_tag)::value;
        size_t smem_size = bam_lz4_io_decomp_smem_per_warp<PS>() * WARPS;
        auto kernel_fn = q16ps_indep_decomp_kernel<PS>;
        FUSED_Q16PS_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
        kernel_fn<<<eff_blocks, THREADS, smem_size, stream>>>(
            ctx->pc_base_addr, ctx->d_decomp_buf, ctx->d_queue[0],
            total_jobs, slot_base);
    });
    FUSED_Q16PS_CUDA_CHECK(cudaGetLastError());
}

void bam_fused_q16ps_run_split_async(
    bam_fused_q16ps_ctx_t ctx_handle,
    const BAMFusedQ16PSParams& params,
    cudaStream_t stream_io,
    cudaStream_t stream_comp)
{
    auto* ctx = static_cast<BAMFusedQ16PSContext*>(ctx_handle);
    const uint32_t npages = params.npages;

    // Double-buffered pipeline: each buffer gets half the page_cache slots.
    // page_cache has num_blocks*4 slots total, each buffer gets num_blocks*2.
    // Each page needs 2 interleaved slots (PK+SK), so max pages per buffer = num_blocks.
    const uint32_t half_slots = ctx->num_blocks * 2;   // slots per buffer
    const uint32_t hw_max_batch = half_slots / 2;       // hardware limit per buffer
    // Allow user override via split_batch_pages (0 = use hw max)
    const uint32_t max_batch = (params.split_batch_pages > 0)
        ? std::min(params.split_batch_pages, hw_max_batch)
        : hw_max_batch;
    const uint32_t slot_bases[Q16PS_NPIPE] = {0, half_slots};

    // Per-buffer QP range isolation: each buffer uses disjoint QP set
    // to prevent cross-batch CQ head-of-line blocking deadlock.
    const uint32_t n_qps = bam_io_page_cache_get_n_qps(ctx->io_pc, 0);
    const uint32_t qp_range = (n_qps >= Q16PS_NPIPE) ? (n_qps / Q16PS_NPIPE) : 1;
    const uint32_t qp_bases[Q16PS_NPIPE] = {0, qp_range};

    // Collect batch descriptors
    uint32_t n_batches = 0;
    struct { uint32_t pg_base, np; } batches[256];
    for (uint32_t pg = 0; pg < npages; pg += max_batch)
        batches[n_batches++] = {pg, std::min(max_batch, npages - pg)};

    if (n_batches == 0) return;

    // Priming: submit batch 0 on stream_io
    {
        uint32_t buf = 0;
        q16ps_launch_submit(ctx, params, buf, batches[0].np, batches[0].pg_base,
                            slot_bases[buf], qp_bases[buf], qp_range, stream_io);
        FUSED_Q16PS_CUDA_CHECK(cudaEventRecord(ctx->submit_done[buf], stream_io));
    }

    for (uint32_t i = 1; i <= n_batches; i++) {
        uint32_t prev_buf = (i - 1) % Q16PS_NPIPE;
        uint32_t cur_buf  = i % Q16PS_NPIPE;

        // Submit batch i (on stream_io, overlaps with compute of batch i-1)
        if (i < n_batches) {
            // Wait for compute of batch i-2 to finish before reusing this buffer
            if (i >= Q16PS_NPIPE) {
                FUSED_Q16PS_CUDA_CHECK(
                    cudaStreamWaitEvent(stream_io, ctx->compute_done[cur_buf]));
            }
            q16ps_launch_submit(ctx, params, cur_buf, batches[i].np, batches[i].pg_base,
                                slot_bases[cur_buf], qp_bases[cur_buf], qp_range, stream_io);
            FUSED_Q16PS_CUDA_CHECK(cudaEventRecord(ctx->submit_done[cur_buf], stream_io));
        }

        // Compute batch i-1 (on stream_comp, after its submit is done)
        FUSED_Q16PS_CUDA_CHECK(
            cudaStreamWaitEvent(stream_comp, ctx->submit_done[prev_buf]));
        q16ps_launch_compute(ctx, params, prev_buf, batches[i-1].np, batches[i-1].pg_base,
                             slot_bases[prev_buf], stream_comp);
        FUSED_Q16PS_CUDA_CHECK(cudaEventRecord(ctx->compute_done[prev_buf], stream_comp));
    }
}

void bam_fused_q16ps_run_split_probe_async(
    bam_fused_q16ps_ctx_t ctx_handle,
    const BAMFusedQ16PSParams& params,
    cudaStream_t stream_io,
    cudaStream_t stream_comp)
{
    auto* ctx = static_cast<BAMFusedQ16PSContext*>(ctx_handle);
    const uint32_t npages = params.npages;
    if (npages == 0) return;

    const uint32_t num_slots = ctx->num_blocks * 4;
    // Each page needs 2 interleaved slots (PK + SK)
    const uint32_t max_batch = num_slots / 2;
    const uint32_t n_qps = bam_io_page_cache_get_n_qps(ctx->io_pc, 0);

    const bool need_batching = (npages > max_batch);
    if (need_batching) {
        printf("[Q16-SPLIT-PROBE] Batched: %u pages, max_batch=%u (%u batches)\n",
               npages, max_batch, (npages + max_batch - 1) / max_batch);
    }

    cudaEvent_t ev_start, ev_end;
    FUSED_Q16PS_CUDA_CHECK(cudaEventCreate(&ev_start));
    FUSED_Q16PS_CUDA_CHECK(cudaEventCreate(&ev_end));
    FUSED_Q16PS_CUDA_CHECK(cudaEventRecord(ev_start, stream_comp));

    for (uint32_t pg_base = 0; pg_base < npages; pg_base += max_batch) {
        const uint32_t batch_np = std::min(max_batch, npages - pg_base);
        const uint32_t slot_base = 0;

        // Phase 1: Submit batch IOs
        q16ps_launch_submit(ctx, params, /*buf=*/0, batch_np, pg_base,
                            slot_base, /*qp_base=*/0, /*qp_range=*/n_qps, stream_io);
        FUSED_Q16PS_CUDA_CHECK(cudaEventRecord(ctx->submit_done[0], stream_io));

        // Phase 2: Decomp batch (writes to d_decomp_buf[0..batch_np*2) batch-local)
        FUSED_Q16PS_CUDA_CHECK(cudaStreamWaitEvent(stream_comp, ctx->submit_done[0]));
        q16ps_launch_indep_decomp(ctx, params, batch_np * 2, slot_base, stream_comp);

        // Phase 3: Probe batch — adjust d_ps pointer so kernel sees correct global row offsets
        BAMFusedQ16PSParams batch_params = params;
        batch_params.npages = batch_np;
        batch_params.d_ps = params.d_ps + pg_base;
        q16ps_probe_kernel<<<batch_np, 128, 0, stream_comp>>>(
            ctx->d_decomp_buf, batch_params, params.page_size);
        FUSED_Q16PS_CUDA_CHECK(cudaGetLastError());

        // Must sync before reusing page_cache slots for next batch
        FUSED_Q16PS_CUDA_CHECK(cudaStreamSynchronize(stream_comp));
        FUSED_Q16PS_CUDA_CHECK(cudaStreamSynchronize(stream_io));
    }

    FUSED_Q16PS_CUDA_CHECK(cudaEventRecord(ev_end, stream_comp));
    FUSED_Q16PS_CUDA_CHECK(cudaStreamSynchronize(stream_comp));
    float ms_total = 0;
    FUSED_Q16PS_CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev_start, ev_end));
    printf("[Q16-SPLIT-PROBE] total: %.3f ms (%u pages, %u batches)\n",
           ms_total, npages, (npages + max_batch - 1) / max_batch);

    FUSED_Q16PS_CUDA_CHECK(cudaEventDestroy(ev_start));
    FUSED_Q16PS_CUDA_CHECK(cudaEventDestroy(ev_end));
}

void bam_fused_q16ps_destroy(bam_fused_q16ps_ctx_t ctx_handle)
{
    auto* ctx = static_cast<BAMFusedQ16PSContext*>(ctx_handle);
    if (!ctx) return;
    if (ctx->d_decomp_buf) cudaFree(ctx->d_decomp_buf);
    for (uint32_t i = 0; i < Q16PS_NPIPE; i++) {
        if (ctx->d_queue[i]) cudaFree(ctx->d_queue[i]);
        cudaEventDestroy(ctx->submit_done[i]);
        cudaEventDestroy(ctx->compute_done[i]);
    }
    bam_io_page_cache_destroy(ctx->io_pc);
    delete ctx;
}
