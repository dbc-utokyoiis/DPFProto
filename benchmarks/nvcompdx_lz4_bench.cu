// nvcompdx_lz4_bench.cu — nvCOMPdx vs nvCOMP LZ4 decompression throughput benchmark
//
// Measures GPU LZ4 decompression throughput with various nvCOMPdx configurations
// and compares against the nvCOMP host-launched batch API baseline.
//
// Build:  cmake --build build --target nvcompdx_lz4_bench
// Run:    ./nvcompdx_lz4_bench [--npages=N] [--page-size=S]

#include <nvcompdx.hpp>
#include <nvcomp/lz4.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>

// ── Error checking ──────────────────────────────────────────

#define CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",             \
                cudaGetErrorString(err), __FILE__, __LINE__);     \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

#define NVCOMP_CHECK(call) do {                                   \
    nvcompStatus_t s = (call);                                    \
    if (s != nvcompSuccess) {                                     \
        fprintf(stderr, "nvCOMP error: %d at %s:%d\n",           \
                (int)s, __FILE__, __LINE__);                      \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

// ── Constants ───────────────────────────────────────────────

static constexpr int WARMUP_ITERS = 5;
static constexpr int BENCH_ITERS  = 20;

// ── Benchmark data ──────────────────────────────────────────

struct BenchData {
    // Device
    char*     d_uncomp;          // original uncompressed [npages * page_size]
    char*     d_comp;            // compressed [npages * max_comp_size]
    char*     d_decomp;          // decomp output [npages * page_size]
    char*     d_decomp_ref;      // nvCOMP reference output [npages * page_size]
    size_t*   d_comp_sizes_sz;   // size_t per-chunk compressed sizes
    size_t*   d_decomp_sizes_sz; // size_t per-chunk decomp sizes (= page_size)
    uint32_t* d_comp_sizes_u32;  // uint32_t version for nvCOMPdx kernels

    // nvCOMP pointer arrays (device)
    void**  d_comp_ptrs;
    void**  d_decomp_ptrs;
    size_t* d_actual_sizes;
    nvcompStatus_t* d_statuses;
    void*   d_temp;
    size_t  temp_bytes;

    // Host
    std::vector<uint32_t> h_comp_sizes;

    uint32_t npages;
    uint32_t page_size;
    size_t   max_comp_size;
    double   compression_ratio;
};

// ── Synthetic data generation ───────────────────────────────

static void generate_synthetic_data(char* h_buf, size_t total_bytes) {
    // Generate data resembling TPC-H INT32 columns under LZ4 compression.
    // TPC-H columns like L_QUANTITY (1-50), L_DISCOUNT (0-10), L_TAX (0-8),
    // L_RETURNFLAG ('A'/'N'/'R'), L_SHIPDATE (19950101-19981231) are stored as
    // INT32 with many zero upper bytes. This pattern gives ~2-3x LZ4 compression.
    //
    // We generate page-by-page with varying field characteristics to simulate
    // realistic mixed-field compression.
    std::mt19937 rng(42);
    uint8_t* p = reinterpret_cast<uint8_t*>(h_buf);
    size_t n = total_bytes;

    // Mix of patterns: runs of zeros, small repeated values, occasional random
    size_t i = 0;
    while (i < n) {
        int pattern = rng() % 100;
        if (pattern < 40) {
            // Zero run (upper bytes of small INT32 values): 8-64 bytes
            size_t len = std::min<size_t>(8 + (rng() % 57), n - i);
            memset(p + i, 0, len);
            i += len;
        } else if (pattern < 75) {
            // Small value repeated: e.g., low byte of L_QUANTITY
            uint8_t val = rng() % 50;
            size_t len = std::min<size_t>(4 + (rng() % 28), n - i);
            memset(p + i, val, len);
            i += len;
        } else if (pattern < 90) {
            // Short sequence of ascending bytes (dates, keys)
            size_t len = std::min<size_t>(4 + (rng() % 16), n - i);
            uint8_t base = rng() % 200;
            for (size_t j = 0; j < len; j++)
                p[i + j] = base + (j & 7);
            i += len;
        } else {
            // Random bytes (incompressible fraction)
            size_t len = std::min<size_t>(4 + (rng() % 8), n - i);
            for (size_t j = 0; j < len; j++)
                p[i + j] = rng() % 256;
            i += len;
        }
    }
}

// ── Data preparation: compress with nvCOMP ──────────────────

static BenchData prepare_data(uint32_t npages, uint32_t page_size) {
    BenchData bd{};
    bd.npages = npages;
    bd.page_size = page_size;

    size_t total = (size_t)npages * page_size;

    // Generate synthetic data on host
    std::vector<char> h_data(total);
    generate_synthetic_data(h_data.data(), total);

    // Copy to GPU
    CUDA_CHECK(cudaMalloc(&bd.d_uncomp, total));
    CUDA_CHECK(cudaMemcpy(bd.d_uncomp, h_data.data(), total, cudaMemcpyHostToDevice));

    // Get max compressed chunk size
    NVCOMP_CHECK(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
        page_size, nvcompBatchedLZ4CompressDefaultOpts, &bd.max_comp_size));

    // Allocate compressed buffer
    CUDA_CHECK(cudaMalloc(&bd.d_comp, (size_t)npages * bd.max_comp_size));

    // Allocate decomp output + reference
    CUDA_CHECK(cudaMalloc(&bd.d_decomp, total));
    CUDA_CHECK(cudaMalloc(&bd.d_decomp_ref, total));

    // Allocate pointer arrays for nvCOMP batch API
    std::vector<void*> h_comp_ptrs(npages), h_decomp_ptrs(npages), h_uncomp_ptrs(npages);
    std::vector<size_t> h_chunk_sizes(npages, page_size);

    for (uint32_t i = 0; i < npages; i++) {
        h_uncomp_ptrs[i] = bd.d_uncomp + (size_t)i * page_size;
        h_comp_ptrs[i]   = bd.d_comp   + (size_t)i * bd.max_comp_size;
        h_decomp_ptrs[i] = bd.d_decomp_ref + (size_t)i * page_size;
    }

    // Device arrays
    CUDA_CHECK(cudaMalloc(&bd.d_comp_ptrs,      npages * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&bd.d_decomp_ptrs,    npages * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&bd.d_comp_sizes_sz,  npages * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&bd.d_decomp_sizes_sz,npages * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&bd.d_actual_sizes,   npages * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&bd.d_statuses,       npages * sizeof(nvcompStatus_t)));
    CUDA_CHECK(cudaMalloc(&bd.d_comp_sizes_u32, npages * sizeof(uint32_t)));

    // Upload chunk size arrays
    void** d_uncomp_ptrs;
    size_t* d_chunk_sizes;
    CUDA_CHECK(cudaMalloc(&d_uncomp_ptrs, npages * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_chunk_sizes, npages * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_uncomp_ptrs, h_uncomp_ptrs.data(), npages * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chunk_sizes, h_chunk_sizes.data(), npages * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd.d_comp_ptrs, h_comp_ptrs.data(), npages * sizeof(void*), cudaMemcpyHostToDevice));

    // Get temp size for compression
    size_t comp_temp_bytes = 0;
    NVCOMP_CHECK(nvcompBatchedLZ4CompressGetTempSizeAsync(
        npages, page_size, nvcompBatchedLZ4CompressDefaultOpts,
        &comp_temp_bytes, total));

    void* d_comp_temp = nullptr;
    if (comp_temp_bytes > 0)
        CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

    // Compress
    NVCOMP_CHECK(nvcompBatchedLZ4CompressAsync(
        (const void* const*)d_uncomp_ptrs,
        d_chunk_sizes,
        page_size,
        npages,
        d_comp_temp,
        comp_temp_bytes,
        (void* const*)bd.d_comp_ptrs,
        bd.d_comp_sizes_sz,
        nvcompBatchedLZ4CompressDefaultOpts,
        bd.d_statuses,
        0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back compressed sizes
    std::vector<size_t> h_comp_sizes_sz(npages);
    CUDA_CHECK(cudaMemcpy(h_comp_sizes_sz.data(), bd.d_comp_sizes_sz,
                           npages * sizeof(size_t), cudaMemcpyDeviceToHost));

    bd.h_comp_sizes.resize(npages);
    size_t total_comp = 0;
    for (uint32_t i = 0; i < npages; i++) {
        bd.h_comp_sizes[i] = (uint32_t)h_comp_sizes_sz[i];
        total_comp += h_comp_sizes_sz[i];
    }
    bd.compression_ratio = (double)total / (double)total_comp;

    // Upload uint32_t comp sizes for nvCOMPdx kernels
    CUDA_CHECK(cudaMemcpy(bd.d_comp_sizes_u32, bd.h_comp_sizes.data(),
                           npages * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Setup decomp sizes (all = page_size)
    CUDA_CHECK(cudaMemcpy(bd.d_decomp_sizes_sz, d_chunk_sizes,
                           npages * sizeof(size_t), cudaMemcpyHostToDevice));

    // Get temp size for decompression
    bd.temp_bytes = 0;
    NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetTempSizeAsync(
        npages, page_size, nvcompBatchedLZ4DecompressDefaultOpts,
        &bd.temp_bytes, total));
    if (bd.temp_bytes > 0)
        CUDA_CHECK(cudaMalloc(&bd.d_temp, bd.temp_bytes));

    // Cleanup temp
    if (d_comp_temp) cudaFree(d_comp_temp);
    cudaFree(d_uncomp_ptrs);
    cudaFree(d_chunk_sizes);

    printf("Data: %u pages x %u B = %.1f MiB\n", npages, page_size,
           (double)total / (1024.0 * 1024.0));
    printf("Compression ratio: %.2fx (%.1f MiB -> %.1f MiB)\n",
           bd.compression_ratio, (double)total / (1024.0*1024.0),
           (double)total_comp / (1024.0*1024.0));

    return bd;
}

static void cleanup_data(BenchData& bd) {
    cudaFree(bd.d_uncomp);
    cudaFree(bd.d_comp);
    cudaFree(bd.d_decomp);
    cudaFree(bd.d_decomp_ref);
    cudaFree(bd.d_comp_sizes_sz);
    cudaFree(bd.d_decomp_sizes_sz);
    cudaFree(bd.d_comp_sizes_u32);
    cudaFree(bd.d_comp_ptrs);
    cudaFree(bd.d_decomp_ptrs);
    cudaFree(bd.d_actual_sizes);
    cudaFree(bd.d_statuses);
    if (bd.d_temp) cudaFree(bd.d_temp);
}

// ── Verification ────────────────────────────────────────────

static bool verify_output(const BenchData& bd, const char* config_name) {
    size_t total = (size_t)bd.npages * bd.page_size;
    std::vector<char> h_out(total), h_ref(total);
    CUDA_CHECK(cudaMemcpy(h_out.data(), bd.d_decomp, total, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), bd.d_uncomp, total, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < total; i++) {
        if (h_out[i] != h_ref[i]) {
            size_t page = i / bd.page_size;
            size_t off = i % bd.page_size;
            fprintf(stderr, "VERIFY FAIL [%s]: page %zu offset %zu: got 0x%02x expected 0x%02x\n",
                    config_name, page, off, (uint8_t)h_out[i], (uint8_t)h_ref[i]);
            return false;
        }
    }
    return true;
}

// ════════════════════════════════════════════════════════════
// nvCOMPdx Warp-level kernel
// ════════════════════════════════════════════════════════════

template <unsigned int PAGE_SIZE_CONST, unsigned int WARPS_PER_BLOCK>
__global__ void bench_warp_kernel(
    const char* __restrict__ d_comp,
    char*       __restrict__ d_decomp,
    const uint32_t* __restrict__ d_comp_sizes,
    uint32_t npages,
    uint32_t page_size,
    size_t comp_stride)
{
    using lz4_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_t);

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane    = threadIdx.x % 32;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    uint32_t slot = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    for (; slot < npages; slot += gridDim.x * WARPS_PER_BLOCK) {
        uint32_t csz = d_comp_sizes[slot];
        const char* src = d_comp   + (uint64_t)slot * comp_stride;
        char*       dst = d_decomp + (uint64_t)slot * page_size;

        if (csz < page_size) {
            size_t dsz = 0;
            lz4_t().execute(src, dst, (size_t)csz, &dsz, my_smem, nullptr);
        } else {
            for (uint32_t i = lane; i < page_size / 4; i += 32)
                reinterpret_cast<uint32_t*>(dst)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
        }
    }
}

// ════════════════════════════════════════════════════════════
// nvCOMPdx Warp-level kernel with __launch_bounds__
// ════════════════════════════════════════════════════════════

template <unsigned int PAGE_SIZE_CONST, unsigned int WARPS_PER_BLOCK, unsigned int MIN_BLOCKS>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, MIN_BLOCKS)
void bench_warp_lb_kernel(
    const char* __restrict__ d_comp,
    char*       __restrict__ d_decomp,
    const uint32_t* __restrict__ d_comp_sizes,
    uint32_t npages,
    uint32_t page_size,
    size_t comp_stride)
{
    using lz4_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_t);

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane    = threadIdx.x % 32;

    extern __shared__ __align__(8) uint8_t smem[];
    constexpr size_t warp_smem = lz4_t().shmem_size_group();
    uint8_t* my_smem = smem + warp_id * warp_smem;

    uint32_t slot = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    for (; slot < npages; slot += gridDim.x * WARPS_PER_BLOCK) {
        uint32_t csz = d_comp_sizes[slot];
        const char* src = d_comp   + (uint64_t)slot * comp_stride;
        char*       dst = d_decomp + (uint64_t)slot * page_size;

        if (csz < page_size) {
            size_t dsz = 0;
            lz4_t().execute(src, dst, (size_t)csz, &dsz, my_smem, nullptr);
        } else {
            for (uint32_t i = lane; i < page_size / 4; i += 32)
                reinterpret_cast<uint32_t*>(dst)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
        }
    }
}

// ════════════════════════════════════════════════════════════
// nvCOMPdx Block-level kernel
// ════════════════════════════════════════════════════════════

template <unsigned int PAGE_SIZE_CONST, unsigned int BLOCK_THREADS>
__global__ void bench_block_kernel(
    const char* __restrict__ d_comp,
    char*       __restrict__ d_decomp,
    const uint32_t* __restrict__ d_comp_sizes,
    uint32_t npages,
    uint32_t page_size,
    size_t comp_stride)
{
    using lz4_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Block() +
        nvcompdx::BlockDim<BLOCK_THREADS, 1, 1>() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_t);

    extern __shared__ __align__(8) uint8_t smem[];
    const uint32_t tid = threadIdx.x;

    for (uint32_t pg = blockIdx.x; pg < npages; pg += gridDim.x) {
        uint32_t csz = d_comp_sizes[pg];
        const char* src = d_comp   + (uint64_t)pg * comp_stride;
        char*       dst = d_decomp + (uint64_t)pg * page_size;

        if (csz < page_size) {
            size_t dsz = 0;
            lz4_t().execute(src, dst, (size_t)csz, &dsz, smem, nullptr);
        } else {
            for (uint32_t i = tid; i < page_size / 4; i += BLOCK_THREADS)
                reinterpret_cast<uint32_t*>(dst)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
        }
        __syncthreads();
    }
}

// ════════════════════════════════════════════════════════════
// Benchmark runner
// ════════════════════════════════════════════════════════════

struct BenchResult {
    const char* name;
    float       throughput_gbs;
    size_t      smem_per_block;
    int         blocks_per_sm;
    float       occupancy_pct;
    bool        verified;
};

template <typename LaunchFn>
static BenchResult run_bench(
    const char* name,
    BenchData& bd,
    LaunchFn launch_fn,
    size_t smem_per_block,
    int threads_per_block)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t total_out = (size_t)bd.npages * bd.page_size;

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        launch_fn(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(BENCH_ITERS);
    for (int i = 0; i < BENCH_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch_fn(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    float avg_ms = std::accumulate(times.begin(), times.end(), 0.0f) / BENCH_ITERS;
    float min_ms = *std::min_element(times.begin(), times.end());
    float max_ms = *std::max_element(times.begin(), times.end());

    // Verify
    bool ok = verify_output(bd, name);

    // Throughput (output bytes basis)
    float throughput = (float)total_out / (avg_ms * 1e6f);  // GB/s

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("  %-42s %8.1f GB/s  (min=%.2f avg=%.2f max=%.2f ms)  %s\n",
           name, throughput, min_ms, avg_ms, max_ms, ok ? "OK" : "FAIL");

    BenchResult r;
    r.name = name;
    r.throughput_gbs = throughput;
    r.smem_per_block = smem_per_block;
    r.blocks_per_sm = 0;
    r.occupancy_pct = 0;
    r.verified = ok;
    return r;
}

// ════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    uint32_t npages = 1024;
    uint32_t page_size = 1048576;  // 1 MiB

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg.find("--npages=") == 0)
            npages = (uint32_t)std::stoul(arg.substr(9));
        else if (arg.find("--page-size=") == 0)
            page_size = (uint32_t)std::stoul(arg.substr(12));
        else {
            fprintf(stderr, "Usage: %s [--npages=N] [--page-size=S]\n", argv[0]);
            return 1;
        }
    }

    // GPU info
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs, CC %d.%d)\n", prop.name, sm_count, prop.major, prop.minor);

    // Prepare data
    BenchData bd = prepare_data(npages, page_size);

    printf("\n=== nvCOMPdx LZ4 Decompression Benchmark ===\n");
    printf("Pages: %u x %u B, MaxCompSize: %zu B\n\n", npages, page_size, bd.max_comp_size);

    std::vector<BenchResult> results;

    // ── 1. nvCOMP batch API baseline ────────────────────────

    {
        // Setup decomp pointer array pointing to d_decomp (not d_decomp_ref)
        std::vector<void*> h_decomp_ptrs(npages);
        for (uint32_t i = 0; i < npages; i++)
            h_decomp_ptrs[i] = bd.d_decomp + (size_t)i * page_size;
        void** d_bench_decomp_ptrs;
        CUDA_CHECK(cudaMalloc(&d_bench_decomp_ptrs, npages * sizeof(void*)));
        CUDA_CHECK(cudaMemcpy(d_bench_decomp_ptrs, h_decomp_ptrs.data(),
                               npages * sizeof(void*), cudaMemcpyHostToDevice));

        auto launch = [&](cudaStream_t stream) {
            NVCOMP_CHECK(nvcompBatchedLZ4DecompressAsync(
                (const void* const*)bd.d_comp_ptrs,
                bd.d_comp_sizes_sz,
                bd.d_decomp_sizes_sz,
                bd.d_actual_sizes,
                npages,
                bd.d_temp,
                bd.temp_bytes,
                (void* const*)d_bench_decomp_ptrs,
                nvcompBatchedLZ4DecompressDefaultOpts,
                bd.d_statuses,
                stream));
        };

        auto r = run_bench("nvCOMP batch API", bd, launch, 0, 0);
        r.blocks_per_sm = -1;
        r.occupancy_pct = -1;
        results.push_back(r);

        cudaFree(d_bench_decomp_ptrs);
    }

    // ── Helper: query occupancy and launch nvCOMPdx kernel ──

    auto query_occupancy = [&](auto kernel_fn, int threads, size_t smem) -> std::pair<int,float> {
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, kernel_fn, threads, smem));
        float occ = 100.0f * blocks_per_sm * threads / 2048.0f;
        return {blocks_per_sm, occ};
    };

    // ── 2. nvCOMPdx Warp, 4 warps/block (128 threads) ──────

    if (page_size == 1048576) {
        using lz4_w4_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<1048576>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        constexpr size_t smem = lz4_w4_t().shmem_size_group() * 4;
        constexpr int threads = 128;

        auto kernel = bench_warp_kernel<1048576, 4>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

        int num_blocks = std::min((int)((npages + 3) / 4), sm_count * 2);
        auto [bps, occ] = query_occupancy(kernel, threads, smem);

        auto launch = [&](cudaStream_t stream) {
            kernel<<<num_blocks, threads, smem, stream>>>(
                bd.d_comp, bd.d_decomp, bd.d_comp_sizes_u32,
                npages, page_size, bd.max_comp_size);
        };

        auto r = run_bench("nvCOMPdx Warp 4w/blk (128 thr)", bd, launch, smem, threads);
        r.blocks_per_sm = bps;
        r.occupancy_pct = occ;
        results.push_back(r);
    }

    // ── 2b. nvCOMPdx Warp, 4 warps/block, UNCAPPED blocks ─────

    if (page_size == 1048576) {
        using lz4_w4_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<1048576>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        constexpr size_t smem = lz4_w4_t().shmem_size_group() * 4;
        constexpr int threads = 128;

        auto kernel = bench_warp_kernel<1048576, 4>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

        // No sm_count*2 cap — use as many blocks as needed for 1 iteration
        int num_blocks = (npages + 3) / 4;
        auto [bps, occ] = query_occupancy(kernel, threads, smem);

        printf("  [4w uncapped: %d blocks, stride=%d]\n", num_blocks, num_blocks * 4);
        auto launch = [&](cudaStream_t stream) {
            kernel<<<num_blocks, threads, smem, stream>>>(
                bd.d_comp, bd.d_decomp, bd.d_comp_sizes_u32,
                npages, page_size, bd.max_comp_size);
        };

        auto r = run_bench("nvCOMPdx Warp 4w UNCAPPED", bd, launch, smem, threads);
        r.blocks_per_sm = bps;
        r.occupancy_pct = occ;
        results.push_back(r);
    }

    // ── 3. nvCOMPdx Warp, 8 warps/block (256 threads) ──────

    if (page_size == 1048576) {
        using lz4_w8_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<1048576>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        constexpr size_t smem = lz4_w8_t().shmem_size_group() * 8;
        constexpr int threads = 256;

        auto kernel = bench_warp_kernel<1048576, 8>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

        int num_blocks = std::min((int)((npages + 7) / 8), sm_count * 2);
        auto [bps, occ] = query_occupancy(kernel, threads, smem);

        auto launch = [&](cudaStream_t stream) {
            kernel<<<num_blocks, threads, smem, stream>>>(
                bd.d_comp, bd.d_decomp, bd.d_comp_sizes_u32,
                npages, page_size, bd.max_comp_size);
        };

        auto r = run_bench("nvCOMPdx Warp 8w/blk (256 thr)", bd, launch, smem, threads);
        r.blocks_per_sm = bps;
        r.occupancy_pct = occ;
        results.push_back(r);
    }

    // ── 4. nvCOMPdx Block, BlockDim<128> ────────────────────

    if (page_size == 1048576) {
        using lz4_b128_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<1048576>() +
            nvcompdx::Block() +
            nvcompdx::BlockDim<128, 1, 1>() +
            nvcompdx::SM<800>());
        constexpr size_t smem = lz4_b128_t().shmem_size_group();
        constexpr int threads = 128;

        auto kernel = bench_block_kernel<1048576, 128>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

        int num_blocks = std::min((int)npages, sm_count * 2);
        auto [bps, occ] = query_occupancy(kernel, threads, smem);

        auto launch = [&](cudaStream_t stream) {
            kernel<<<num_blocks, threads, smem, stream>>>(
                bd.d_comp, bd.d_decomp, bd.d_comp_sizes_u32,
                npages, page_size, bd.max_comp_size);
        };

        auto r = run_bench("nvCOMPdx Block<128> (128 thr)", bd, launch, smem, threads);
        r.blocks_per_sm = bps;
        r.occupancy_pct = occ;
        results.push_back(r);
    }

    // ── 5. nvCOMPdx Block, BlockDim<256> ────────────────────

    if (page_size == 1048576) {
        using lz4_b256_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<1048576>() +
            nvcompdx::Block() +
            nvcompdx::BlockDim<256, 1, 1>() +
            nvcompdx::SM<800>());
        constexpr size_t smem = lz4_b256_t().shmem_size_group();
        constexpr int threads = 256;

        auto kernel = bench_block_kernel<1048576, 256>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

        int num_blocks = std::min((int)npages, sm_count * 2);
        auto [bps, occ] = query_occupancy(kernel, threads, smem);

        auto launch = [&](cudaStream_t stream) {
            kernel<<<num_blocks, threads, smem, stream>>>(
                bd.d_comp, bd.d_decomp, bd.d_comp_sizes_u32,
                npages, page_size, bd.max_comp_size);
        };

        auto r = run_bench("nvCOMPdx Block<256> (256 thr)", bd, launch, smem, threads);
        r.blocks_per_sm = bps;
        r.occupancy_pct = occ;
        results.push_back(r);
    }

    // ── 6. nvCOMPdx Warp 4w + __launch_bounds__(128, 2) ─────

    if (page_size == 1048576) {
        using lz4_w4_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<1048576>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        constexpr size_t smem = lz4_w4_t().shmem_size_group() * 4;
        constexpr int threads = 128;

        auto kernel = bench_warp_lb_kernel<1048576, 4, 2>;
        CUDA_CHECK(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

        int num_blocks = std::min((int)((npages + 3) / 4), sm_count * 2);
        auto [bps, occ] = query_occupancy(kernel, threads, smem);

        auto launch = [&](cudaStream_t stream) {
            kernel<<<num_blocks, threads, smem, stream>>>(
                bd.d_comp, bd.d_decomp, bd.d_comp_sizes_u32,
                npages, page_size, bd.max_comp_size);
        };

        auto r = run_bench("nvCOMPdx Warp 4w + LB(128,2)", bd, launch, smem, threads);
        r.blocks_per_sm = bps;
        r.occupancy_pct = occ;
        results.push_back(r);
    }

    // ── Summary table ───────────────────────────────────────

    printf("\n%-44s %10s %10s %9s %10s %6s\n",
           "Config", "GB/s(out)", "smem/blk", "blks/SM", "occupancy", "OK");
    printf("%-44s %10s %10s %9s %10s %6s\n",
           std::string(44, '-').c_str(), "--------", "--------", "-------", "---------", "----");

    for (auto& r : results) {
        char smem_str[16], bps_str[16], occ_str[16];
        if (r.blocks_per_sm < 0) {
            snprintf(smem_str, sizeof(smem_str), "N/A");
            snprintf(bps_str, sizeof(bps_str), "N/A");
            snprintf(occ_str, sizeof(occ_str), "N/A");
        } else {
            snprintf(smem_str, sizeof(smem_str), "%zu", r.smem_per_block);
            snprintf(bps_str, sizeof(bps_str), "%d", r.blocks_per_sm);
            snprintf(occ_str, sizeof(occ_str), "%.1f%%", r.occupancy_pct);
        }
        printf("%-44s %8.1f   %10s %9s %10s %6s\n",
               r.name, r.throughput_gbs, smem_str, bps_str, occ_str,
               r.verified ? "OK" : "FAIL");
    }

    cleanup_data(bd);
    printf("\nDone.\n");
    return 0;
}
