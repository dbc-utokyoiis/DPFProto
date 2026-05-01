// ============================================================
// lz4_decomp_bench — Multi-page LZ4 decompression benchmark
//
// Each warp decompresses a DIFFERENT page (realistic L2 cache behavior).
// Generates N_PAGES (1024) pages per pattern, each with unique data but
// similar compression characteristics.
//
// Compares: nvCOMPdx, v3 (GPU-parsed), v5 (pre-computed cooperative),
//           v6 (pre-computed adaptive + smem staging),
//           v7 (GPU-parsed + v6 execution, no metadata required)
//
// Usage:
//   ./build/lz4_decomp_bench [max_warps]
// ============================================================

#include <nvcompdx.hpp>
#include <lz4.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "lz4_decomp.cuh"
#include "../src/common/lz4_helper.cuh"

#define CHECK(call) do {                                                     \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                         \
                cudaGetErrorString(err), __FILE__, __LINE__);                \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

static constexpr uint32_t PAGE_SIZE = 1048576;  // 1 MB
static constexpr int WARMUP_ITERS = 3;
static constexpr int BENCH_ITERS  = 10;
static constexpr int N_PAGES      = 1024;

// ============================================================
// Per-page descriptor (lives in GPU global memory)
// ============================================================
struct PageDesc {
    const uint8_t* comp;
    uint32_t       comp_sz;
    const Lz4Seq*  seqs;
    uint32_t       n_seqs;
};

// ============================================================
// LZ4PARSEQ: PFOR-compressed metadata (3 fields: lit_len, offset, match_len)
// GPU reconstructs di[] and lit_src[] from these at decode time.
//
// Per-page metadata format (uint32_t array):
//   [0] n_seqs
//   [1] field1_word_offset  (offset[] field, from parseq start)
//   [2] field2_word_offset  (match_len[] field, from parseq start)
//   [3..] field 0 (lit_len), field 1 (offset), field 2 (match_len)
//
// Per-field format:
//   [n_blocks][block_off[0]..block_off[n_blocks]][block0 data][block1 data]...
//   block_off[b] = word offset from start of packed block data to block b
//   Block data: [min_val][bw_packed (4×same bw)][bit-packed words...]
// ============================================================

struct ParseqPageDesc {
    const uint8_t*  comp;
    uint32_t        comp_sz;
    const uint32_t* parseq;     // PFOR-compressed metadata on GPU
    uint32_t        n_seqs;
};

// pfor_encode_field, parseq_encode, parse_lz4_sequences are in lz4_helper.cuh

// ── GPU PFOR decoder (warp-level) ──

__device__ __forceinline__ uint32_t parseq_pfor_decode_one(
    const uint32_t* blk_data, uint32_t idx_in_block, uint32_t bw)
{
    uint32_t min_val = blk_data[0];
    if (bw == 0) return min_val;

    uint32_t mb_idx = idx_in_block >> 5;
    uint32_t mb_pos = idx_in_block & 31;
    uint32_t mb_offset = mb_idx * bw;

    uint32_t bit_pos = bw * mb_pos;
    uint32_t word_idx = 2 + mb_offset + (bit_pos >> 5);
    uint32_t bit_shift = bit_pos & 31;

    uint64_t two_words = ((uint64_t)blk_data[word_idx + 1] << 32) | blk_data[word_idx];
    uint32_t element = (uint32_t)((two_words >> bit_shift) & ((1ULL << bw) - 1));
    return min_val + element;
}

// Decode one PFOR field: n_blocks blocks of 128 elements each.
// field_ptr: [n_blocks][block_off[0..n_blocks]][packed data...]
__device__ inline void parseq_pfor_decode_field(
    const uint32_t* field_ptr, uint32_t n_seqs,
    uint32_t* out, uint32_t lane)
{
    uint32_t n_blocks = field_ptr[0];
    const uint32_t* block_offs = field_ptr + 1;
    const uint32_t* packed_base = field_ptr + 1 + (n_blocks + 1);

    for (uint32_t b = 0; b < n_blocks; b++) {
        const uint32_t* blk_data = packed_base + block_offs[b];
        uint32_t bw = blk_data[1] & 0xFF;

        for (uint32_t k = 0; k < 4; k++) {
            uint32_t idx = b * 128 + k * 32 + lane;
            if (idx < n_seqs) {
                out[idx] = parseq_pfor_decode_one(blk_data, k * 32 + lane, bw);
            }
        }
        __syncwarp();
    }
}

// Full LZ4PARSEQ decompression: PFOR decode → reconstruct Lz4Seq → v6
__device__ inline uint32_t lz4_decompress_warp_parseq(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    const uint32_t* parseq,
    Lz4Seq*         seq_buf,    // per-warp scratch in global memory [max_seqs]
    uint8_t*        warp_smem)  // v6 staging buffer [LZ4_V6_SMEM_PER_WARP]
{
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t n_seqs = parseq[0];
    const uint32_t field1_off = parseq[1];
    const uint32_t field2_off = parseq[2];

    // Decode 3 fields into temporary SoA arrays AFTER the Lz4Seq region.
    // Layout in seq_buf: [Lz4Seq × n_seqs][lit_len[n]][offset[n]][match_len[n]]
    uint32_t* d_lit_len   = reinterpret_cast<uint32_t*>(seq_buf + n_seqs);
    uint32_t* d_offset    = d_lit_len + n_seqs;
    uint32_t* d_match_len = d_offset + n_seqs;

    const uint32_t* field0_ptr = parseq + 3;                  // lit_len
    const uint32_t* field1_ptr = parseq + field1_off;         // offset
    const uint32_t* field2_ptr = parseq + field2_off;         // match_len

    parseq_pfor_decode_field(field0_ptr, n_seqs, d_lit_len, lane);
    parseq_pfor_decode_field(field1_ptr, n_seqs, d_offset, lane);
    parseq_pfor_decode_field(field2_ptr, n_seqs, d_match_len, lane);
    __syncwarp();

    // ── Reconstruct di[] and lit_src[] — warp parallel prefix sum ──
    {
        const uint32_t chunk = (n_seqs + 31) / 32;
        const uint32_t my_start = lane * chunk;
        const uint32_t my_end = (my_start + chunk < n_seqs) ? my_start + chunk : n_seqs;

        // Phase 1: local scan
        uint32_t local_di = 0;
        uint32_t local_ci = 0;
        for (uint32_t i = my_start; i < my_end; i++) {
            uint32_t ll  = d_lit_len[i];
            uint32_t off = d_offset[i];
            uint32_t ml  = d_match_len[i];

            uint32_t ext_lit = (ll >= 15) ? (1 + (ll - 15) / 255) : 0;
            uint32_t lit_src = local_ci + 1 + ext_lit;

            seq_buf[i].lit_src   = lit_src;
            seq_buf[i].lit_len   = ll;
            seq_buf[i].offset    = off;
            seq_buf[i].match_len = ml;
            seq_buf[i].di        = local_di;

            local_ci = lit_src + ll;
            if (off > 0) {
                local_ci += 2;
                if (ml >= 19)
                    local_ci += 1 + (ml - 19) / 255;
            }
            local_di += ll + ml;
        }

        // Phase 2: warp-level inclusive prefix sum of lane totals
        uint32_t di_sum = local_di;
        uint32_t ci_sum = local_ci;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            uint32_t di_up = __shfl_up_sync(0xFFFFFFFF, di_sum, d);
            uint32_t ci_up = __shfl_up_sync(0xFFFFFFFF, ci_sum, d);
            if (lane >= (uint32_t)d) {
                di_sum += di_up;
                ci_sum += ci_up;
            }
        }

        uint32_t di_base = di_sum - local_di;
        uint32_t ci_base = ci_sum - local_ci;

        // Phase 3: add base offsets
        for (uint32_t i = my_start; i < my_end; i++) {
            seq_buf[i].di      += di_base;
            seq_buf[i].lit_src += ci_base;
        }
    }
    __syncwarp();

    return lz4_decompress_warp_v6(input, output, seq_buf, n_seqs, warp_smem);
}

// ============================================================
// Kernels — each warp looks up its own page via PageDesc array
// ============================================================

template <unsigned int PAGE_SZ, int WARPS_PER_BLOCK>
__global__ void kernel_nvcompdx_mp(
    const PageDesc* pages, uint32_t n_pages,
    uint8_t* d_decomp, uint32_t n_warps)
{
    using lz4_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SZ>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_t);

    extern __shared__ __align__(8) uint8_t smem[];

    const uint32_t warp_id_local = threadIdx.x / 32;
    const uint32_t global_warp   = blockIdx.x * WARPS_PER_BLOCK + warp_id_local;
    if (global_warp >= n_warps) return;

    const PageDesc p = pages[global_warp % n_pages];
    uint8_t* my_smem = smem + warp_id_local * lz4_t().shmem_size_group();
    uint8_t* my_out  = d_decomp + (uint64_t)global_warp * PAGE_SZ;

    auto decompressor = lz4_t();
    size_t dsz = 0;
    decompressor.execute(p.comp, my_out, (size_t)p.comp_sz, &dsz, my_smem, nullptr);
}

template <int WARPS_PER_BLOCK>
__global__ void kernel_warp_v3_mp(
    const PageDesc* pages, uint32_t n_pages,
    uint8_t* d_decomp, uint32_t n_warps)
{
    extern __shared__ __align__(8) uint8_t dyn_smem[];

    const uint32_t warp_id_local = threadIdx.x / 32;
    const uint32_t global_warp   = blockIdx.x * WARPS_PER_BLOCK + warp_id_local;
    if (global_warp >= n_warps) return;

    const PageDesc p = pages[global_warp % n_pages];
    uint8_t* my_smem = dyn_smem + warp_id_local * LZ4_WARP_V3_SMEM;
    uint8_t* my_out  = d_decomp + (uint64_t)global_warp * PAGE_SIZE;

    lz4_decompress_warp_v3(p.comp, p.comp_sz, my_out, PAGE_SIZE, my_smem);
}

template <int WARPS_PER_BLOCK>
__global__ void kernel_warp_v5_mp(
    const PageDesc* pages, uint32_t n_pages,
    uint8_t* d_decomp, uint32_t n_warps)
{
    const uint32_t warp_id_local = threadIdx.x / 32;
    const uint32_t global_warp   = blockIdx.x * WARPS_PER_BLOCK + warp_id_local;
    if (global_warp >= n_warps) return;

    const PageDesc p = pages[global_warp % n_pages];
    uint8_t* my_out = d_decomp + (uint64_t)global_warp * PAGE_SIZE;

    lz4_decompress_warp_v5(p.comp, my_out, p.seqs, p.n_seqs);
}

template <int WARPS_PER_BLOCK>
__global__ void kernel_warp_v6_mp(
    const PageDesc* pages, uint32_t n_pages,
    uint8_t* d_decomp, uint32_t n_warps)
{
    extern __shared__ __align__(8) uint8_t dyn_smem[];

    const uint32_t warp_id_local = threadIdx.x / 32;
    const uint32_t global_warp   = blockIdx.x * WARPS_PER_BLOCK + warp_id_local;
    if (global_warp >= n_warps) return;

    const PageDesc p = pages[global_warp % n_pages];
    uint8_t* my_smem = dyn_smem + warp_id_local * LZ4_V6_SMEM_PER_WARP;
    uint8_t* my_out  = d_decomp + (uint64_t)global_warp * PAGE_SIZE;

    lz4_decompress_warp_v6(p.comp, my_out, p.seqs, p.n_seqs, my_smem);
}

template <int WARPS_PER_BLOCK>
__global__ void kernel_warp_v7_mp(
    const PageDesc* pages, uint32_t n_pages,
    uint8_t* d_decomp, uint32_t n_warps)
{
    extern __shared__ __align__(8) uint8_t dyn_smem[];

    const uint32_t warp_id_local = threadIdx.x / 32;
    const uint32_t global_warp   = blockIdx.x * WARPS_PER_BLOCK + warp_id_local;
    if (global_warp >= n_warps) return;

    const PageDesc p = pages[global_warp % n_pages];
    uint8_t* my_smem = dyn_smem + warp_id_local * LZ4_V7_SMEM;
    uint8_t* my_out  = d_decomp + (uint64_t)global_warp * PAGE_SIZE;

    lz4_decompress_warp_v7(p.comp, p.comp_sz, my_out, PAGE_SIZE, my_smem);
}

template <int WARPS_PER_BLOCK>
__global__ void kernel_lz4parseq_mp(
    const ParseqPageDesc* pages, uint32_t n_pages,
    uint8_t* d_decomp, Lz4Seq* d_seq_scratch,
    uint32_t max_seqs_per_page, uint32_t n_warps)
{
    extern __shared__ __align__(8) uint8_t dyn_smem[];

    const uint32_t warp_id_local = threadIdx.x / 32;
    const uint32_t global_warp   = blockIdx.x * WARPS_PER_BLOCK + warp_id_local;
    if (global_warp >= n_warps) return;

    const ParseqPageDesc p = pages[global_warp % n_pages];
    uint8_t* my_smem = dyn_smem + warp_id_local * LZ4_V6_SMEM_PER_WARP;
    uint8_t* my_out  = d_decomp + (uint64_t)global_warp * PAGE_SIZE;
    Lz4Seq*  my_seqs = d_seq_scratch + (uint64_t)global_warp * max_seqs_per_page;

    lz4_decompress_warp_parseq(p.comp, my_out, p.parseq, my_seqs, my_smem);
}

// ============================================================
// Verification
// ============================================================
static bool verify_output(
    const uint8_t* h_reference,
    const uint8_t* d_decomp,
    const char* method_name)
{
    uint8_t* h_out = new uint8_t[PAGE_SIZE];
    CHECK(cudaMemcpy(h_out, d_decomp, PAGE_SIZE, cudaMemcpyDeviceToHost));
    bool ok = (memcmp(h_reference, h_out, PAGE_SIZE) == 0);
    if (!ok) {
        fprintf(stderr, "  VERIFY FAIL: %s\n", method_name);
        for (uint32_t i = 0; i < PAGE_SIZE; i++) {
            if (h_reference[i] != h_out[i]) {
                fprintf(stderr, "    first mismatch at byte %u: expected 0x%02x got 0x%02x\n",
                        i, h_reference[i], h_out[i]);
                break;
            }
        }
    }
    delete[] h_out;
    return ok;
}

// ============================================================
// Fill functions (with seed for per-page variation)
// ============================================================
static void fill_low_card(uint8_t* page, uint32_t seed) {
    int32_t* p = (int32_t*)page;
    for (uint32_t i = 0; i < PAGE_SIZE / sizeof(int32_t); i++)
        p[i] = (i + seed) % 3;
}

static void fill_sequential(uint8_t* page, uint32_t seed) {
    int32_t* p = (int32_t*)page;
    for (uint32_t i = 0; i < PAGE_SIZE / sizeof(int32_t); i++)
        p[i] = (i + seed) % 100;
}

static void fill_random(uint8_t* page, uint32_t seed) {
    uint32_t* p = (uint32_t*)page;
    uint32_t s = 12345 + seed * 7919;
    for (uint32_t i = 0; i < PAGE_SIZE / sizeof(uint32_t); i++) {
        s = s * 1103515245 + 12345;
        p[i] = s;
    }
}

static void fill_mixed_30pct(uint8_t* page, uint32_t seed) {
    uint32_t* p = (uint32_t*)page;
    uint32_t n = PAGE_SIZE / sizeof(uint32_t);
    uint32_t s = 42 + seed * 7919;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t block = i / 16;
        if (block % 4 == 0) {
            s = s * 1103515245 + 12345;
            p[i] = s;
        } else {
            p[i] = (uint32_t)((i + seed) % 16);
        }
    }
}

// ============================================================
// Multi-page test data
// ============================================================
struct MultiPageData {
    const char* name;

    // Host: page 0 uncompressed (for verification)
    uint8_t* h_page0;

    // GPU: packed compressed pages + metadata
    uint8_t*  d_comp_packed;
    Lz4Seq*   d_seqs_packed;
    PageDesc* d_pages;

    // LZ4PARSEQ: PFOR-compressed metadata
    uint32_t*       d_parseq_packed;
    ParseqPageDesc* d_parseq_pages;
    Lz4Seq*         d_parseq_scratch;   // per-warp decode buffer
    uint32_t        max_seqs_per_page;
    size_t          total_parseq_words;

    // Stats
    double   avg_ratio;
    uint32_t avg_n_seqs;
    int      avg_comp_sz;
    size_t   total_comp_bytes;
    size_t   total_seq_count;
    int      avg_parseq_bytes;          // PFOR metadata per page
};

using FillFn = void(*)(uint8_t*, uint32_t);

static MultiPageData make_multi_page(const char* name, FillFn fill) {
    MultiPageData mp = {};
    mp.name = name;

    uint8_t* page = new uint8_t[PAGE_SIZE];
    int max_comp = LZ4_compressBound(PAGE_SIZE);
    uint8_t* comp_buf = new uint8_t[max_comp];

    // First pass: compress all pages, collect sizes
    std::vector<std::vector<uint8_t>> all_comp(N_PAGES);
    std::vector<std::vector<Lz4Seq>>  all_seqs(N_PAGES);
    size_t total_comp = 0, total_seqs = 0;

    for (int i = 0; i < N_PAGES; i++) {
        fill(page, (uint32_t)i);
        if (i == 0) {
            mp.h_page0 = new uint8_t[PAGE_SIZE];
            memcpy(mp.h_page0, page, PAGE_SIZE);
        }

        int csz = LZ4_compress_default(
            (const char*)page, (char*)comp_buf, PAGE_SIZE, max_comp);
        all_comp[i].assign(comp_buf, comp_buf + csz);
        total_comp += csz;

        parse_lz4_sequences(comp_buf, csz, all_seqs[i]);
        total_seqs += all_seqs[i].size();
    }

    mp.avg_comp_sz = (int)(total_comp / N_PAGES);
    mp.avg_ratio   = (double)PAGE_SIZE / mp.avg_comp_sz;
    mp.avg_n_seqs  = (uint32_t)(total_seqs / N_PAGES);
    mp.total_comp_bytes = total_comp;
    mp.total_seq_count  = total_seqs;

    // Pack compressed pages into one contiguous GPU buffer
    std::vector<uint8_t> comp_packed(total_comp);
    std::vector<size_t>  comp_offsets(N_PAGES);
    size_t off = 0;
    for (int i = 0; i < N_PAGES; i++) {
        comp_offsets[i] = off;
        memcpy(comp_packed.data() + off, all_comp[i].data(), all_comp[i].size());
        off += all_comp[i].size();
    }

    CHECK(cudaMalloc(&mp.d_comp_packed, total_comp));
    CHECK(cudaMemcpy(mp.d_comp_packed, comp_packed.data(), total_comp,
                     cudaMemcpyHostToDevice));

    // Pack metadata into one contiguous GPU buffer
    std::vector<Lz4Seq> seqs_packed(total_seqs);
    std::vector<size_t>  seq_offsets(N_PAGES);
    size_t soff = 0;
    for (int i = 0; i < N_PAGES; i++) {
        seq_offsets[i] = soff;
        memcpy(seqs_packed.data() + soff, all_seqs[i].data(),
               all_seqs[i].size() * sizeof(Lz4Seq));
        soff += all_seqs[i].size();
    }

    CHECK(cudaMalloc(&mp.d_seqs_packed, total_seqs * sizeof(Lz4Seq)));
    CHECK(cudaMemcpy(mp.d_seqs_packed, seqs_packed.data(),
                     total_seqs * sizeof(Lz4Seq), cudaMemcpyHostToDevice));

    // Build PageDesc array
    std::vector<PageDesc> h_pages(N_PAGES);
    for (int i = 0; i < N_PAGES; i++) {
        h_pages[i].comp    = mp.d_comp_packed + comp_offsets[i];
        h_pages[i].comp_sz = (uint32_t)all_comp[i].size();
        h_pages[i].seqs    = mp.d_seqs_packed + seq_offsets[i];
        h_pages[i].n_seqs  = (uint32_t)all_seqs[i].size();
    }

    CHECK(cudaMalloc(&mp.d_pages, N_PAGES * sizeof(PageDesc)));
    CHECK(cudaMemcpy(mp.d_pages, h_pages.data(), N_PAGES * sizeof(PageDesc),
                     cudaMemcpyHostToDevice));

    // ── LZ4PARSEQ: PFOR-encode metadata for each page ──
    std::vector<std::vector<uint32_t>> all_parseq(N_PAGES);
    size_t total_parseq = 0;
    uint32_t max_seqs = 0;
    for (int i = 0; i < N_PAGES; i++) {
        all_parseq[i] = parseq_encode(all_seqs[i]);
        total_parseq += all_parseq[i].size();
        max_seqs = std::max(max_seqs, (uint32_t)all_seqs[i].size());
    }
    mp.total_parseq_words = total_parseq;
    mp.avg_parseq_bytes = (int)(total_parseq * sizeof(uint32_t) / N_PAGES);
    mp.max_seqs_per_page = max_seqs;

    // Pack PARSEQ into contiguous GPU buffer
    std::vector<uint32_t> parseq_packed(total_parseq);
    std::vector<size_t>   parseq_offsets(N_PAGES);
    size_t poff = 0;
    for (int i = 0; i < N_PAGES; i++) {
        parseq_offsets[i] = poff;
        memcpy(parseq_packed.data() + poff, all_parseq[i].data(),
               all_parseq[i].size() * sizeof(uint32_t));
        poff += all_parseq[i].size();
    }

    CHECK(cudaMalloc(&mp.d_parseq_packed, total_parseq * sizeof(uint32_t)));
    CHECK(cudaMemcpy(mp.d_parseq_packed, parseq_packed.data(),
                     total_parseq * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Build ParseqPageDesc array
    std::vector<ParseqPageDesc> h_parseq_pages(N_PAGES);
    for (int i = 0; i < N_PAGES; i++) {
        h_parseq_pages[i].comp    = mp.d_comp_packed + comp_offsets[i];
        h_parseq_pages[i].comp_sz = (uint32_t)all_comp[i].size();
        h_parseq_pages[i].parseq  = mp.d_parseq_packed + parseq_offsets[i];
        h_parseq_pages[i].n_seqs  = (uint32_t)all_seqs[i].size();
    }

    CHECK(cudaMalloc(&mp.d_parseq_pages, N_PAGES * sizeof(ParseqPageDesc)));
    CHECK(cudaMemcpy(mp.d_parseq_pages, h_parseq_pages.data(),
                     N_PAGES * sizeof(ParseqPageDesc), cudaMemcpyHostToDevice));

    // Allocate per-warp scratch buffer for PARSEQ decode (will be sized in main)
    mp.d_parseq_scratch = nullptr;

    delete[] page;
    delete[] comp_buf;
    return mp;
}

static void free_multi_page(MultiPageData& mp) {
    delete[] mp.h_page0;
    CHECK(cudaFree(mp.d_comp_packed));
    CHECK(cudaFree(mp.d_seqs_packed));
    CHECK(cudaFree(mp.d_pages));
    CHECK(cudaFree(mp.d_parseq_packed));
    CHECK(cudaFree(mp.d_parseq_pages));
    if (mp.d_parseq_scratch) CHECK(cudaFree(mp.d_parseq_scratch));
}

// ============================================================
// Benchmark runner
// ============================================================
struct BenchResult {
    double avg_ms;
    double throughput_gbs;
};

template <typename LaunchFn>
static BenchResult run_bench(LaunchFn launch, uint32_t n_warps) {
    for (int i = 0; i < WARMUP_ITERS; i++)
        launch();
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < BENCH_ITERS; i++)
        launch();
    CHECK(cudaEventRecord(ev1));
    CHECK(cudaEventSynchronize(ev1));

    float total_ms = 0;
    CHECK(cudaEventElapsedTime(&total_ms, ev0, ev1));
    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));

    double avg_ms = total_ms / BENCH_ITERS;
    double bytes  = (double)n_warps * PAGE_SIZE;
    double gbs    = bytes / (avg_ms * 1e6);

    return {avg_ms, gbs};
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    uint32_t max_warps = (argc >= 2) ? (uint32_t)atoi(argv[1]) : 1024;

    CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    fprintf(stderr, "=== LZ4 Multi-Page Decomp Benchmark ===\n");
    fprintf(stderr, "GPU: %s (SM %d.%d, %d SMs, %.0f MHz)\n",
            prop.name, prop.major, prop.minor,
            prop.multiProcessorCount, prop.clockRate / 1000.0);
    fprintf(stderr, "Page size: %u B (%u KB)\n", PAGE_SIZE, PAGE_SIZE / 1024);
    fprintf(stderr, "Pages: %d (each warp decompresses a different page)\n", N_PAGES);
    fprintf(stderr, "Max warps: %u\n\n", max_warps);

    // ── Shared memory setup ──
    using lz4_query_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());
    size_t smem_per_warp_nv = lz4_query_t().shmem_size_group();

    int max_smem = 0;
    CHECK(cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    constexpr int WPB_NV = 4;
    constexpr int WPB    = 4;

    size_t nv_smem = WPB_NV * smem_per_warp_nv;
    auto nv_fn = kernel_nvcompdx_mp<PAGE_SIZE, WPB_NV>;
    CHECK(cudaFuncSetAttribute(nv_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)nv_smem));

    size_t v3_smem = WPB * LZ4_WARP_V3_SMEM;
    auto v3_fn = kernel_warp_v3_mp<WPB>;
    CHECK(cudaFuncSetAttribute(v3_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)v3_smem));

    size_t v6_smem = WPB * LZ4_V6_SMEM_PER_WARP;
    auto v6_fn = kernel_warp_v6_mp<WPB>;
    CHECK(cudaFuncSetAttribute(v6_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)v6_smem));

    size_t v7_smem = WPB * LZ4_V7_SMEM;
    auto v7_fn = kernel_warp_v7_mp<WPB>;
    CHECK(cudaFuncSetAttribute(v7_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)v7_smem));

    // LZ4PARSEQ uses same smem as v6 (for v6 staging)
    size_t parseq_smem = WPB * LZ4_V6_SMEM_PER_WARP;
    auto parseq_fn = kernel_lz4parseq_mp<WPB>;
    CHECK(cudaFuncSetAttribute(parseq_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)parseq_smem));

    fprintf(stderr, "nvCOMPdx smem/warp: %zu B\n", smem_per_warp_nv);
    fprintf(stderr, "v3 smem/warp: %zu B\n", LZ4_WARP_V3_SMEM);
    fprintf(stderr, "v6 smem/warp: %u B\n", LZ4_V6_SMEM_PER_WARP);
    fprintf(stderr, "v7 smem/warp: %zu B\n\n", LZ4_V7_SMEM);

    // ── Output buffer ──
    uint8_t* d_decomp;
    CHECK(cudaMalloc(&d_decomp, (size_t)max_warps * PAGE_SIZE));

    // ── Prepare multi-page test data ──
    fprintf(stderr, "Generating %d pages per pattern...\n", N_PAGES);

    struct PatternDef { const char* name; FillFn fill; };
    PatternDef patterns[] = {
        {"low_card (i%3)",     fill_low_card},
        {"sequential (i%100)", fill_sequential},
        {"mixed (~30%)",       fill_mixed_30pct},
        {"random (LCG)",       fill_random},
    };
    constexpr int N_PATTERNS = sizeof(patterns) / sizeof(patterns[0]);

    // Find max_seqs across all patterns for scratch buffer sizing
    uint32_t global_max_seqs = 0;

    MultiPageData tests[N_PATTERNS];
    for (int t = 0; t < N_PATTERNS; t++) {
        tests[t] = make_multi_page(patterns[t].name, patterns[t].fill);
        global_max_seqs = std::max(global_max_seqs, tests[t].max_seqs_per_page);
        int raw_meta = (int)(tests[t].total_seq_count * sizeof(Lz4Seq) / N_PAGES);
        fprintf(stderr, "  %-22s  avg_comp=%d  ratio=%.1fx  avg_n_seqs=%u  "
                "meta: raw=%dB  pfor=%dB  (%.1fx)\n",
                tests[t].name, tests[t].avg_comp_sz, tests[t].avg_ratio,
                tests[t].avg_n_seqs,
                raw_meta, tests[t].avg_parseq_bytes,
                raw_meta > 0 ? (double)raw_meta / tests[t].avg_parseq_bytes : 0.0);
    }
    fprintf(stderr, "\n");

    // Allocate PARSEQ scratch buffer per warp:
    //   [Lz4Seq × max_seqs] + [3 × uint32 × max_seqs] for SoA decode
    // Total = max_seqs * (sizeof(Lz4Seq) + 3*4) = max_seqs * 32 bytes
    size_t scratch_per_warp = (size_t)global_max_seqs * (sizeof(Lz4Seq) + 3 * sizeof(uint32_t));
    // Redefine max_seqs_per_page for kernel: in units of Lz4Seq stride
    // The kernel uses: seq_buf + global_warp * max_seqs_per_page
    // But we need to pass stride in Lz4Seq units that covers the full scratch.
    // scratch_per_warp / sizeof(Lz4Seq) = max_seqs * 32/20 → not integer.
    // Instead, pass byte stride and compute in kernel. But for simplicity:
    // Allocate as char* and compute pointer in kernel.
    // Actually, keep it simple: allocate as Lz4Seq* with enough padding.
    uint32_t parseq_stride = (uint32_t)(scratch_per_warp / sizeof(Lz4Seq) + 1);
    for (int t = 0; t < N_PATTERNS; t++) {
        size_t scratch_sz = (size_t)max_warps * parseq_stride * sizeof(Lz4Seq);
        CHECK(cudaMalloc(&tests[t].d_parseq_scratch, scratch_sz));
        tests[t].max_seqs_per_page = parseq_stride;
    }

    // ── Warp sweep ──
    int sweep[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_sweep = sizeof(sweep) / sizeof(sweep[0]);

    for (int t = 0; t < N_PATTERNS; t++) {
        MultiPageData& mp = tests[t];
        fprintf(stderr, "============================================================\n");
        fprintf(stderr, "Pattern: %-22s  avg_ratio=%.1fx  %d pages\n",
                mp.name, mp.avg_ratio, N_PAGES);
        fprintf(stderr, "============================================================\n");

        // ── Verification (page 0, 1 warp) ──
        // Use a single-page PageDesc for verification
        PageDesc h_page0_desc;
        {
            // Upload page 0 data separately for single-warp verification
            PageDesc h_tmp;
            CHECK(cudaMemcpy(&h_tmp, mp.d_pages, sizeof(PageDesc),
                             cudaMemcpyDeviceToHost));
            h_page0_desc = h_tmp;
        }

        CHECK(cudaMemset(d_decomp, 0, PAGE_SIZE));
        nv_fn<<<1, WPB_NV * 32, nv_smem>>>(mp.d_pages, N_PAGES, d_decomp, 1);
        CHECK(cudaDeviceSynchronize());
        bool ok_nv = verify_output(mp.h_page0, d_decomp, "nvCOMPdx");

        CHECK(cudaMemset(d_decomp, 0, PAGE_SIZE));
        v3_fn<<<1, WPB * 32, v3_smem>>>(mp.d_pages, N_PAGES, d_decomp, 1);
        CHECK(cudaDeviceSynchronize());
        bool ok_v3 = verify_output(mp.h_page0, d_decomp, "v3");

        CHECK(cudaMemset(d_decomp, 0, PAGE_SIZE));
        kernel_warp_v5_mp<WPB><<<1, WPB * 32>>>(mp.d_pages, N_PAGES, d_decomp, 1);
        CHECK(cudaDeviceSynchronize());
        bool ok_v5 = verify_output(mp.h_page0, d_decomp, "v5");

        CHECK(cudaMemset(d_decomp, 0, PAGE_SIZE));
        v6_fn<<<1, WPB * 32, v6_smem>>>(mp.d_pages, N_PAGES, d_decomp, 1);
        CHECK(cudaDeviceSynchronize());
        bool ok_v6 = verify_output(mp.h_page0, d_decomp, "v6");

        CHECK(cudaMemset(d_decomp, 0, PAGE_SIZE));
        v7_fn<<<1, WPB * 32, v7_smem>>>(mp.d_pages, N_PAGES, d_decomp, 1);
        CHECK(cudaDeviceSynchronize());
        bool ok_v7 = verify_output(mp.h_page0, d_decomp, "v7");

        CHECK(cudaMemset(d_decomp, 0, PAGE_SIZE));
        parseq_fn<<<1, WPB * 32, parseq_smem>>>(
            mp.d_parseq_pages, N_PAGES, d_decomp, mp.d_parseq_scratch,
            mp.max_seqs_per_page, 1);
        CHECK(cudaDeviceSynchronize());
        bool ok_pq = verify_output(mp.h_page0, d_decomp, "parseq");

        fprintf(stderr, "  Verification: nvCOMPdx=%s  v3=%s  v5=%s  v6=%s  v7=%s  parseq=%s\n\n",
                ok_nv ? "OK" : "FAIL", ok_v3 ? "OK" : "FAIL",
                ok_v5 ? "OK" : "FAIL", ok_v6 ? "OK" : "FAIL",
                ok_v7 ? "OK" : "FAIL", ok_pq ? "OK" : "FAIL");

        if (!ok_nv || !ok_v3 || !ok_v5 || !ok_v6 || !ok_v7 || !ok_pq) {
            fprintf(stderr, "  SKIPPING benchmarks due to verification failure.\n\n");
            continue;
        }

        // ── Header ──
        fprintf(stderr, "  %-6s  %-9s %-9s  %-9s %-9s  %-9s %-9s  %-9s %-9s  %-9s %-9s  %-9s %-9s  %s\n",
                "warps",
                "nv_ms",  "nv_GBs",
                "v3_ms",  "v3_GBs",
                "v5_ms",  "v5_GBs",
                "v6_ms",  "v6_GBs",
                "v7_ms",  "v7_GBs",
                "pq_ms",  "pq_GBs",
                "v6/nv  pq/nv  v6/pq");
        fprintf(stderr, "  %-6s  %-9s %-9s  %-9s %-9s  %-9s %-9s  %-9s %-9s  %-9s %-9s  %-9s %-9s  %s\n",
                "------",
                "------", "------",
                "------", "------",
                "------", "------",
                "------", "------",
                "------", "------",
                "------", "------",
                "-----  -----  -----");

        for (int s = 0; s < n_sweep; s++) {
            uint32_t nw = (uint32_t)sweep[s];
            if (nw > max_warps) break;

            uint32_t nb_nv = (nw + WPB_NV - 1) / WPB_NV;
            uint32_t nb    = (nw + WPB - 1) / WPB;

            auto res_nv = run_bench([&]() {
                nv_fn<<<nb_nv, WPB_NV * 32, nv_smem>>>(
                    mp.d_pages, N_PAGES, d_decomp, nw);
            }, nw);

            auto res_v3 = run_bench([&]() {
                v3_fn<<<nb, WPB * 32, v3_smem>>>(
                    mp.d_pages, N_PAGES, d_decomp, nw);
            }, nw);

            auto res_v5 = run_bench([&]() {
                kernel_warp_v5_mp<WPB><<<nb, WPB * 32>>>(
                    mp.d_pages, N_PAGES, d_decomp, nw);
            }, nw);

            auto res_v6 = run_bench([&]() {
                v6_fn<<<nb, WPB * 32, v6_smem>>>(
                    mp.d_pages, N_PAGES, d_decomp, nw);
            }, nw);

            auto res_v7 = run_bench([&]() {
                v7_fn<<<nb, WPB * 32, v7_smem>>>(
                    mp.d_pages, N_PAGES, d_decomp, nw);
            }, nw);

            auto res_pq = run_bench([&]() {
                parseq_fn<<<nb, WPB * 32, parseq_smem>>>(
                    mp.d_parseq_pages, N_PAGES, d_decomp, mp.d_parseq_scratch,
                    mp.max_seqs_per_page, nw);
            }, nw);

            double v6_vs_nv = res_v6.throughput_gbs / res_nv.throughput_gbs;
            double pq_vs_nv = res_pq.throughput_gbs / res_nv.throughput_gbs;
            double v6_vs_pq = res_v6.throughput_gbs / res_pq.throughput_gbs;

            fprintf(stderr,
                "  %-6u  %-9.3f %-9.2f  %-9.3f %-9.2f  %-9.3f %-9.2f  %-9.3f %-9.2f  %-9.3f %-9.2f  %-9.3f %-9.2f  %.2fx  %.2fx  %.2fx\n",
                nw,
                res_nv.avg_ms, res_nv.throughput_gbs,
                res_v3.avg_ms, res_v3.throughput_gbs,
                res_v5.avg_ms, res_v5.throughput_gbs,
                res_v6.avg_ms, res_v6.throughput_gbs,
                res_v7.avg_ms, res_v7.throughput_gbs,
                res_pq.avg_ms, res_pq.throughput_gbs,
                v6_vs_nv, pq_vs_nv, v6_vs_pq);
        }

        fprintf(stderr, "\n");
    }

    // ── Cleanup ──
    CHECK(cudaFree(d_decomp));
    for (int t = 0; t < N_PATTERNS; t++)
        free_multi_page(tests[t]);

    return 0;
}
