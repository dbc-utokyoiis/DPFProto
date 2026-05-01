// ============================================================
// IO / Decomp Single-Page Latency Benchmark
//
// Measures per-page latency for:
//   1. BaM NVMe sync read (1 warp, 1 page, 10 iterations)
//   2. nvCOMPdx warp-level LZ4 decompression (1 warp, 1 page, 10 iterations)
//
// Results give the exact t_io and t_decomp values needed for the
// cost model: use_lz4 iff max(t_io_comp, t_decomp) < t_io_none
//
// Usage:
//   sudo ./samples/io_decomp_latency <devices> <start_page>
//
// Example:
//   sudo ./samples/io_decomp_latency /dev/libnvm0,...,/dev/libnvm3 48241
// ============================================================

#include "bam_io_device.cuh"
#include "bam_kernel.cuh"

#include <nvcompdx.hpp>
#include <lz4.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define BENCH_CHECK(call) do {                                               \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                         \
                cudaGetErrorString(err), __FILE__, __LINE__);                \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

static constexpr uint32_t PAGE_SIZE = 1048576;
static constexpr uint32_t PAGE_NBLK = PAGE_SIZE / 512;
static constexpr int N_ITERS = 10;

// ============================================================
// Kernel 1: IO-only latency (1 warp = 32 threads, 1 block)
//
// Lane 0 does BaM sync read, all lanes time with clock64().
// ============================================================
__global__ void kernel_io_latency(
    void*     ctrls,
    void*     pc,
    uint64_t  lba,
    uint32_t  dev,
    uint64_t* d_latencies)  // [N_ITERS]
{
    const uint32_t lane = threadIdx.x;

    for (int iter = 0; iter < N_ITERS; iter++) {
        uint64_t t0 = clock64();

        if (lane == 0)
            bam_io_read_page_device(ctrls, pc, lba, PAGE_NBLK, 0, dev);
        __syncwarp();

        uint64_t t1 = clock64();
        if (lane == 0)
            d_latencies[iter] = t1 - t0;
    }
}

// ============================================================
// Kernel 2: Decomp-only latency (1 warp = 32 threads, 1 block)
//
// Data already in d_comp_buf (GPU memory, uploaded from host).
// Decompresses to d_decomp_buf with nvCOMPdx.
// ============================================================
template <unsigned int PAGE_SIZE_CONST>
__global__ void kernel_decomp_latency(
    const char* d_comp_buf,
    uint32_t    comp_sz,
    char*       d_decomp_buf,
    uint64_t*   d_latencies)  // [N_ITERS]
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    extern __shared__ __align__(8) uint8_t smem[];
    const uint32_t lane = threadIdx.x;

    for (int iter = 0; iter < N_ITERS; iter++) {
        uint64_t t0 = clock64();

        auto decompressor = lz4_decomp_t();
        size_t dsz = 0;
        decompressor.execute(d_comp_buf, d_decomp_buf,
                             (size_t)comp_sz, &dsz, smem, nullptr);
        __syncwarp();

        uint64_t t1 = clock64();
        if (lane == 0)
            d_latencies[iter] = t1 - t0;
    }
}

// ============================================================
// Kernel 3: IO + Decomp fused latency (1 warp, 1 block)
//
// BaM sync read → nvCOMPdx decomp.  Reports IO and decomp
// latencies separately.
// ============================================================
template <unsigned int PAGE_SIZE_CONST>
__global__ void kernel_io_decomp_latency(
    void*       ctrls,
    void*       pc,
    void*       pc_base_addr,
    uint64_t    lba,
    uint32_t    nblk,
    uint32_t    dev,
    uint32_t    comp_sz,
    char*       d_decomp_buf,
    uint64_t*   d_io_lat,      // [N_ITERS]
    uint64_t*   d_decomp_lat)  // [N_ITERS]
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());

    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    extern __shared__ __align__(8) uint8_t smem[];
    const uint32_t lane = threadIdx.x;

    for (int iter = 0; iter < N_ITERS; iter++) {
        // Phase 1: IO
        uint64_t t0 = clock64();
        if (lane == 0)
            bam_io_read_page_device(ctrls, pc, lba, nblk, 0, dev);
        __syncwarp();
        uint64_t t1 = clock64();

        // Phase 2: Decomp from page_cache → decomp_buf
        const char* src = (const char*)pc_base_addr;  // slot 0
        if (comp_sz < PAGE_SIZE_CONST) {
            auto decompressor = lz4_decomp_t();
            size_t dsz = 0;
            decompressor.execute(src, d_decomp_buf,
                                 (size_t)comp_sz, &dsz, smem, nullptr);
        } else {
            const uint32_t n4 = PAGE_SIZE_CONST / 4;
            for (uint32_t i = lane; i < n4; i += 32)
                reinterpret_cast<uint32_t*>(d_decomp_buf)[i] =
                    reinterpret_cast<const uint32_t*>(src)[i];
        }
        __syncwarp();
        uint64_t t2 = clock64();

        if (lane == 0) {
            d_io_lat[iter] = t1 - t0;
            d_decomp_lat[iter] = t2 - t1;
        }
    }
}

// ============================================================
// Kernel 4: Balanced pipeline (1 IO warp + N decomp warps)
//
// Warp 0: IO producer — BaM sync reads, signals via smem counter.
// Warps 1..N_DECOMP: nvCOMPdx LZ4 decomp, wait for IO signal.
//
// Decomp uses a pre-compressed GPU buffer (NVMe data is not
// compressed), correctly measuring pipeline overlap throughput.
// ============================================================
static constexpr int N_DECOMP_WARPS = 11;
static constexpr int BALANCED_BLOCK_SIZE = (N_DECOMP_WARPS + 1) * 32;  // 384

template <unsigned int PAGE_SIZE_CONST>
__global__ void kernel_balanced_pipeline(
    void*       ctrls,
    void*       pc,
    uint64_t    pages_start,   // global page ID of first page
    uint32_t    n_devices,
    uint32_t    total_pages,
    const char* d_comp_buf,    // pre-compressed page (GPU memory)
    uint32_t    comp_sz,
    uint32_t    smem_per_warp,
    char*       d_decomp_buf,  // [total_pages * PAGE_SIZE]
    uint64_t*   d_block_elapsed)
{
    using lz4_decomp_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE_CONST>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());
    NVCOMPDX_SKIP_IF_NOT_APPLICABLE(lz4_decomp_t);

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane    = threadIdx.x % 32;
    const uint32_t bid     = blockIdx.x;
    const uint32_t nblocks = gridDim.x;

    __shared__ volatile uint32_t io_produced;
    if (threadIdx.x == 0)
        io_produced = 0;
    __syncthreads();

    // Per-block page range
    uint32_t pages_per_block = (total_pages + nblocks - 1) / nblocks;
    uint32_t my_start = bid * pages_per_block;
    uint32_t my_end   = my_start + pages_per_block;
    if (my_end > total_pages) my_end = total_pages;
    uint32_t my_count = (my_end > my_start) ? (my_end - my_start) : 0;

    extern __shared__ __align__(8) uint8_t dyn_smem[];

    uint64_t t_start = clock64();

    if (warp_id == 0) {
        // ── IO warp (producer) ──
        constexpr uint32_t nblk = PAGE_SIZE_CONST / 512;
        for (uint32_t i = 0; i < my_count; i++) {
            uint64_t gpage = pages_start + my_start + i;
            uint32_t dev   = gpage % n_devices;
            uint64_t lpg   = gpage / n_devices;
            uint64_t lba   = 2048 + lpg * nblk;
            if (lane == 0) {
                bam_io_read_page_device(ctrls, pc, lba, nblk, bid, dev);
                __threadfence_block();
                io_produced = i + 1;
            }
            __syncwarp();
        }
    } else {
        // ── Decomp warps (consumers) ──
        uint32_t decomp_id = warp_id - 1;
        uint8_t* my_smem = dyn_smem + decomp_id * smem_per_warp;

        for (uint32_t i = decomp_id; i < my_count; i += N_DECOMP_WARPS) {
            // Wait for IO to produce page i
            if (lane == 0)
                while (io_produced <= i) {}
            __syncwarp();

            // Decompress pre-compressed buffer → output
            uint32_t page_id = my_start + i;
            char* dst = d_decomp_buf + (uint64_t)page_id * PAGE_SIZE_CONST;
            auto decompressor = lz4_decomp_t();
            size_t dsz = 0;
            decompressor.execute(d_comp_buf, dst,
                                 (size_t)comp_sz, &dsz, my_smem, nullptr);
            __syncwarp();
        }
    }

    __syncthreads();
    uint64_t t_end = clock64();

    if (threadIdx.x == 0)
        d_block_elapsed[bid] = t_end - t_start;
}

// ============================================================
// Host helpers
// ============================================================
static int parse_devices(const char* arg, char paths[][256], int max_devices) {
    int n = 0;
    const char* p = arg;
    while (*p && n < max_devices) {
        const char* comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);
        if (len >= 256) len = 255;
        memcpy(paths[n], p, len);
        paths[n][len] = '\0';
        n++;
        if (!comma) break;
        p = comma + 1;
    }
    return n;
}

static void print_latencies(const char* label, uint64_t* lat, int n, int gpu_clock_khz) {
    double sum_us = 0, min_us = 1e18, max_us = 0;
    fprintf(stderr, "  %-20s", label);
    for (int i = 0; i < n; i++) {
        double us = (double)lat[i] / (gpu_clock_khz / 1000.0);
        sum_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }
    double avg_us = sum_us / n;
    double avg_ms = avg_us / 1000.0;
    double bw_gbs = (double)PAGE_SIZE / (avg_us * 1000.0);  // GB/s
    fprintf(stderr, "avg=%.1f us (%.3f ms)  min=%.1f  max=%.1f  → %.2f GB/s\n",
            avg_us, avg_ms, min_us, max_us, bw_gbs);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <devices> <start_page> [npages]\n"
            "\n"
            "Measures single-page latencies for IO and LZ4 decomp,\n"
            "then runs balanced pipeline (1 IO + 11 decomp warps/block).\n"
            "\n"
            "  devices:    comma-separated BaM device paths\n"
            "  start_page: page ID of a field (e.g., 48241 for L_QUANTITY SF300)\n"
            "  npages:     total pages for Part 4 pipeline bench (default: 256)\n"
            "\n"
            "Example:\n"
            "  sudo %s /dev/libnvm0,...,/dev/libnvm3 48241\n"
            "  sudo %s /dev/libnvm0,...,/dev/libnvm3 48241 1024\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    char dev_paths[8][256];
    int n_devices = parse_devices(argv[1], dev_paths, 8);
    uint64_t start_page = strtoull(argv[2], nullptr, 10);

    cudaSetDevice(0);

    int gpu_clock_khz = 0;
    BENCH_CHECK(cudaDeviceGetAttribute(&gpu_clock_khz,
                                        cudaDevAttrClockRate, 0));

    // Query nvCOMPdx shared memory requirement per warp
    using lz4_decomp_query_t = decltype(
        nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
        nvcompdx::DataType<nvcompdx::datatype::uint8>() +
        nvcompdx::Direction<nvcompdx::direction::decompress>() +
        nvcompdx::MaxUncompChunkSize<PAGE_SIZE>() +
        nvcompdx::Warp() +
        nvcompdx::SM<800>());
    size_t smem_per_warp = lz4_decomp_query_t().shmem_size_group();

    int max_smem_per_block = 0;
    BENCH_CHECK(cudaDeviceGetAttribute(&max_smem_per_block,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    int max_decomp_warps = max_smem_per_block / (int)smem_per_warp;

    fprintf(stderr, "=== IO / Decomp Single-Page Latency ===\n");
    fprintf(stderr, "Devices: %d, Page: %lu, GPU clock: %.0f MHz\n",
            n_devices, start_page, gpu_clock_khz / 1000.0);
    fprintf(stderr, "Iterations: %d\n", N_ITERS);
    fprintf(stderr, "nvCOMPdx smem/warp: %zu B, max smem/block: %d B\n",
            smem_per_warp, max_smem_per_block);
    fprintf(stderr, "Max decomp warps/block: %d  (+ 1 IO warp = %d threads)\n\n",
            max_decomp_warps, (max_decomp_warps + 1) * 32);

    const char* paths[8];
    for (int i = 0; i < n_devices; i++) paths[i] = dev_paths[i];
    bam_ctrl_handle_t ctrl = bam_ctrl_open_multi(
        paths, n_devices, 1, 0, 1024, 128);

    uint64_t partition_start_lbas[8];
    for (int i = 0; i < n_devices; i++)
        partition_start_lbas[i] = 2048;

    // ================================================================
    // Part 1: IO-only latency (1 full page = 1 MB)
    // ================================================================
    {
        fprintf(stderr, "--- Part 1: IO Latency (1 MB page, BaM sync read) ---\n");

        bam_io_page_cache_t io_pc = bam_io_page_cache_create(
            ctrl, PAGE_SIZE, 1);  // 1 slot
        void* d_ctrls = bam_io_page_cache_get_d_ctrls(io_pc);
        void* d_pc = bam_io_page_cache_get_d_pc_ptr(io_pc);

        uint32_t dev = start_page % n_devices;
        uint64_t local_pg = start_page / n_devices;
        uint64_t lba = partition_start_lbas[dev] + local_pg * PAGE_NBLK;

        uint64_t* d_lat;
        BENCH_CHECK(cudaMalloc(&d_lat, N_ITERS * sizeof(uint64_t)));

        // Warmup
        kernel_io_latency<<<1, 32>>>(d_ctrls, d_pc, lba, dev, d_lat);
        BENCH_CHECK(cudaDeviceSynchronize());

        // Timed
        kernel_io_latency<<<1, 32>>>(d_ctrls, d_pc, lba, dev, d_lat);
        BENCH_CHECK(cudaDeviceSynchronize());

        uint64_t h_lat[N_ITERS];
        BENCH_CHECK(cudaMemcpy(h_lat, d_lat, N_ITERS * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost));

        print_latencies("IO (1 MB):", h_lat, N_ITERS, gpu_clock_khz);

        cudaFree(d_lat);
        bam_io_page_cache_destroy(io_pc);
    }

    // ================================================================
    // Part 2: Decomp-only latency (synthetic compressed pages)
    //
    // Generate test pages with different data patterns → different
    // compression ratios.  Compress on host with CPU LZ4, upload to
    // GPU, decompress with nvCOMPdx, measure.
    // ================================================================
    {
        fprintf(stderr, "\n--- Part 2: Decomp Latency (nvCOMPdx warp LZ4) ---\n");

        // smem size query
        using lz4_decomp_t = decltype(
            nvcompdx::Algorithm<nvcompdx::algorithm::lz4>() +
            nvcompdx::DataType<nvcompdx::datatype::uint8>() +
            nvcompdx::Direction<nvcompdx::direction::decompress>() +
            nvcompdx::MaxUncompChunkSize<PAGE_SIZE>() +
            nvcompdx::Warp() +
            nvcompdx::SM<800>());
        size_t smem_size = lz4_decomp_t().shmem_size_group();

        auto kernel_fn = kernel_decomp_latency<PAGE_SIZE>;
        BENCH_CHECK(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_size));

        // Allocate GPU buffers
        char* d_comp_buf;
        char* d_decomp_buf;
        uint64_t* d_lat;
        int max_comp_size = LZ4_compressBound(PAGE_SIZE);
        BENCH_CHECK(cudaMalloc(&d_comp_buf, max_comp_size));
        BENCH_CHECK(cudaMalloc(&d_decomp_buf, PAGE_SIZE));
        BENCH_CHECK(cudaMalloc(&d_lat, N_ITERS * sizeof(uint64_t)));

        // Host buffers
        char* h_page = new char[PAGE_SIZE];
        char* h_comp = new char[max_comp_size];

        // Pattern: sequential int32 (typical column data, ~2.5-3x ratio)
        struct TestPattern {
            const char* name;
            int ratio_hint;  // approximate expected ratio
        };
        // We generate different patterns to see how decomp latency varies
        // with compression ratio.
        auto fill_sequential = [&]() {
            int32_t* p = (int32_t*)h_page;
            int n = PAGE_SIZE / sizeof(int32_t);
            for (int i = 0; i < n; i++)
                p[i] = i % 100;  // ~3x ratio (few distinct values)
        };

        auto fill_low_card = [&]() {
            int32_t* p = (int32_t*)h_page;
            int n = PAGE_SIZE / sizeof(int32_t);
            for (int i = 0; i < n; i++)
                p[i] = i % 3;  // very low cardinality → high ratio
        };

        auto fill_random = [&]() {
            uint32_t* p = (uint32_t*)h_page;
            int n = PAGE_SIZE / sizeof(uint32_t);
            uint32_t s = 12345;
            for (int i = 0; i < n; i++) {
                s = s * 1103515245 + 12345;
                p[i] = s;
            }
        };

        fprintf(stderr, "  %-22s %-10s %-10s %s\n",
                "pattern", "comp_sz", "ratio", "latency");
        fprintf(stderr, "  %-22s %-10s %-10s %s\n",
                "-------", "-------", "-----", "-------");

        // Test pattern 1: low cardinality
        fill_low_card();
        {
            int comp_sz = LZ4_compress_default(h_page, h_comp, PAGE_SIZE, max_comp_size);
            double ratio = (double)PAGE_SIZE / comp_sz;
            BENCH_CHECK(cudaMemcpy(d_comp_buf, h_comp, comp_sz, cudaMemcpyHostToDevice));

            kernel_fn<<<1, 32, smem_size>>>(d_comp_buf, comp_sz, d_decomp_buf, d_lat);
            BENCH_CHECK(cudaDeviceSynchronize());
            // Warmup done, now measure
            kernel_fn<<<1, 32, smem_size>>>(d_comp_buf, comp_sz, d_decomp_buf, d_lat);
            BENCH_CHECK(cudaDeviceSynchronize());

            uint64_t h_lat[N_ITERS];
            BENCH_CHECK(cudaMemcpy(h_lat, d_lat, N_ITERS * sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost));
            double sum_us = 0;
            for (int i = 0; i < N_ITERS; i++)
                sum_us += (double)h_lat[i] / (gpu_clock_khz / 1000.0);
            double avg_us = sum_us / N_ITERS;
            double bw_gbs = (double)PAGE_SIZE / (avg_us * 1000.0);
            fprintf(stderr, "  %-22s %-10d %-10.1fx avg=%.1f us → %.2f GB/s\n",
                    "low_card (i%3)", comp_sz, ratio, avg_us, bw_gbs);
        }

        // Test pattern 2: sequential
        fill_sequential();
        {
            int comp_sz = LZ4_compress_default(h_page, h_comp, PAGE_SIZE, max_comp_size);
            double ratio = (double)PAGE_SIZE / comp_sz;
            BENCH_CHECK(cudaMemcpy(d_comp_buf, h_comp, comp_sz, cudaMemcpyHostToDevice));

            kernel_fn<<<1, 32, smem_size>>>(d_comp_buf, comp_sz, d_decomp_buf, d_lat);
            BENCH_CHECK(cudaDeviceSynchronize());
            kernel_fn<<<1, 32, smem_size>>>(d_comp_buf, comp_sz, d_decomp_buf, d_lat);
            BENCH_CHECK(cudaDeviceSynchronize());

            uint64_t h_lat[N_ITERS];
            BENCH_CHECK(cudaMemcpy(h_lat, d_lat, N_ITERS * sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost));
            double sum_us = 0;
            for (int i = 0; i < N_ITERS; i++)
                sum_us += (double)h_lat[i] / (gpu_clock_khz / 1000.0);
            double avg_us = sum_us / N_ITERS;
            double bw_gbs = (double)PAGE_SIZE / (avg_us * 1000.0);
            fprintf(stderr, "  %-22s %-10d %-10.1fx avg=%.1f us → %.2f GB/s\n",
                    "sequential (i%100)", comp_sz, ratio, avg_us, bw_gbs);
        }

        // Test pattern 3: random
        fill_random();
        {
            int comp_sz = LZ4_compress_default(h_page, h_comp, PAGE_SIZE, max_comp_size);
            double ratio = (double)PAGE_SIZE / comp_sz;
            BENCH_CHECK(cudaMemcpy(d_comp_buf, h_comp, comp_sz, cudaMemcpyHostToDevice));

            kernel_fn<<<1, 32, smem_size>>>(d_comp_buf, comp_sz, d_decomp_buf, d_lat);
            BENCH_CHECK(cudaDeviceSynchronize());
            kernel_fn<<<1, 32, smem_size>>>(d_comp_buf, comp_sz, d_decomp_buf, d_lat);
            BENCH_CHECK(cudaDeviceSynchronize());

            uint64_t h_lat[N_ITERS];
            BENCH_CHECK(cudaMemcpy(h_lat, d_lat, N_ITERS * sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost));
            double sum_us = 0;
            for (int i = 0; i < N_ITERS; i++)
                sum_us += (double)h_lat[i] / (gpu_clock_khz / 1000.0);
            double avg_us = sum_us / N_ITERS;
            double bw_gbs = (double)PAGE_SIZE / (avg_us * 1000.0);
            fprintf(stderr, "  %-22s %-10d %-10.1fx avg=%.1f us → %.2f GB/s\n",
                    "random (LCG)", comp_sz, ratio, avg_us, bw_gbs);
        }

        delete[] h_page;
        delete[] h_comp;
        cudaFree(d_comp_buf);
        cudaFree(d_decomp_buf);
        cudaFree(d_lat);
    }

    // ================================================================
    // Part 3: IO + Decomp fused latency (reads compressed page from
    // NVMe, then decompresses — like the real fusion kernel)
    //
    // To run this, we pre-compress a page on host, write it to page_cache
    // via BaM IO (read from any page on disk), then re-read + decomp.
    //
    // Since we can't easily write compressed data to NVMe, we instead:
    //   - Read 1 page from NVMe (uncompressed, represents raw IO latency)
    //   - The IO latency is independent of data content
    //   - Decomp latency was measured in Part 2
    //   - We report the sum as the fused cost estimate
    // ================================================================
    {
        fprintf(stderr, "\n--- Part 3: Cost Model Summary ---\n");

        // Re-measure IO to get a clean number
        bam_io_page_cache_t io_pc = bam_io_page_cache_create(ctrl, PAGE_SIZE, 1);
        void* d_ctrls = bam_io_page_cache_get_d_ctrls(io_pc);
        void* d_pc = bam_io_page_cache_get_d_pc_ptr(io_pc);

        uint32_t dev = start_page % n_devices;
        uint64_t local_pg = start_page / n_devices;
        uint64_t lba = partition_start_lbas[dev] + local_pg * PAGE_NBLK;

        uint64_t* d_lat;
        BENCH_CHECK(cudaMalloc(&d_lat, N_ITERS * sizeof(uint64_t)));

        kernel_io_latency<<<1, 32>>>(d_ctrls, d_pc, lba, dev, d_lat);
        BENCH_CHECK(cudaDeviceSynchronize());
        kernel_io_latency<<<1, 32>>>(d_ctrls, d_pc, lba, dev, d_lat);
        BENCH_CHECK(cudaDeviceSynchronize());

        uint64_t h_io_lat[N_ITERS];
        BENCH_CHECK(cudaMemcpy(h_io_lat, d_lat, N_ITERS * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost));

        double io_sum = 0;
        for (int i = 0; i < N_ITERS; i++)
            io_sum += (double)h_io_lat[i] / (gpu_clock_khz / 1000.0);
        double t_io_us = io_sum / N_ITERS;
        double io_bw_gbs = (double)PAGE_SIZE / (t_io_us * 1000.0);

        fprintf(stderr, "\n");
        fprintf(stderr, "  t_io (1 MB page)       = %.1f us  →  IO_BW = %.2f GB/s\n",
                t_io_us, io_bw_gbs);
        fprintf(stderr, "\n");
        fprintf(stderr, "  Cost model: LZ4 is beneficial when\n");
        fprintf(stderr, "    max(t_io_comp, t_decomp) < t_io_none\n");
        fprintf(stderr, "  where:\n");
        fprintf(stderr, "    t_io_none  = %.1f us  (= 1 MB / %.2f GB/s)\n",
                t_io_us, io_bw_gbs);
        fprintf(stderr, "    t_io_comp  = t_io_none / R  (R = compression ratio)\n");
        fprintf(stderr, "    t_decomp   = from Part 2 measurements\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "  If t_decomp > t_io_none: LZ4 is NEVER beneficial\n");
        fprintf(stderr, "  If t_decomp < t_io_none: LZ4 is ALWAYS beneficial\n");
        fprintf(stderr, "  Threshold decomp BW = IO_BW = %.2f GB/s\n", io_bw_gbs);

        cudaFree(d_lat);
        bam_io_page_cache_destroy(io_pc);
    }

    // ================================================================
    // Part 4: Balanced pipeline (1 IO warp + 11 decomp warps/block)
    //
    // Sweeps num_blocks to find aggregate throughput saturation.
    // IO warp does real BaM NVMe reads; decomp warps decompress
    // a pre-compressed buffer.  Producer-consumer sync via smem.
    // ================================================================
    {
        fprintf(stderr, "\n--- Part 4: Balanced Pipeline "
                "(1 IO + %d decomp warps/block, %d threads) ---\n",
                N_DECOMP_WARPS, BALANCED_BLOCK_SIZE);

        // Pre-compress a synthetic page (sequential i%100 pattern)
        char* h_page = new char[PAGE_SIZE];
        {
            int32_t* p = (int32_t*)h_page;
            for (uint32_t i = 0; i < PAGE_SIZE / sizeof(int32_t); i++)
                p[i] = i % 100;
        }
        int max_comp_size = LZ4_compressBound(PAGE_SIZE);
        char* h_comp = new char[max_comp_size];
        int comp_sz = LZ4_compress_default(h_page, h_comp, PAGE_SIZE, max_comp_size);
        fprintf(stderr, "  Compressed test page: %u → %d B (%.1fx)\n",
                PAGE_SIZE, comp_sz, (double)PAGE_SIZE / comp_sz);

        char* d_comp_buf;
        BENCH_CHECK(cudaMalloc(&d_comp_buf, comp_sz));
        BENCH_CHECK(cudaMemcpy(d_comp_buf, h_comp, comp_sz, cudaMemcpyHostToDevice));

        uint32_t total_pages = (argc >= 4) ? (uint32_t)atoi(argv[3]) : 256;
        fprintf(stderr, "  Total pages: %u (%u MB)\n",
                total_pages, total_pages * (PAGE_SIZE / (1024 * 1024)));

        char* d_decomp_buf;
        BENCH_CHECK(cudaMalloc(&d_decomp_buf, (size_t)total_pages * PAGE_SIZE));

        constexpr int MAX_BLOCKS = 36;
        uint64_t* d_block_elapsed;
        BENCH_CHECK(cudaMalloc(&d_block_elapsed, MAX_BLOCKS * sizeof(uint64_t)));

        auto balanced_fn = kernel_balanced_pipeline<PAGE_SIZE>;
        size_t dyn_smem = N_DECOMP_WARPS * smem_per_warp;
        BENCH_CHECK(cudaFuncSetAttribute(
            balanced_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem));

        bam_io_page_cache_t io_pc = bam_io_page_cache_create(
            ctrl, PAGE_SIZE, MAX_BLOCKS);
        void* d_ctrls = bam_io_page_cache_get_d_ctrls(io_pc);
        void* d_pc    = bam_io_page_cache_get_d_pc_ptr(io_pc);

        fprintf(stderr, "  Dynamic smem: %zu B (%d warps × %zu B)\n\n",
                dyn_smem, N_DECOMP_WARPS, smem_per_warp);

        fprintf(stderr, "  %-8s %-12s %-12s %-10s %s\n",
                "blocks", "elapsed_ms", "eff_GB/s", "pages/ms", "");
        fprintf(stderr, "  %-8s %-12s %-12s %-10s %s\n",
                "------", "----------", "--------", "--------", "");

        cudaEvent_t ev0, ev1;
        BENCH_CHECK(cudaEventCreate(&ev0));
        BENCH_CHECK(cudaEventCreate(&ev1));

        int trials[] = {1, 2, 3, 6, 9, 12, 18, 27};
        for (int t = 0; t < (int)(sizeof(trials)/sizeof(trials[0])); t++) {
            int nb = trials[t];

            // Warmup
            balanced_fn<<<nb, BALANCED_BLOCK_SIZE, dyn_smem>>>(
                d_ctrls, d_pc, start_page, (uint32_t)n_devices, total_pages,
                d_comp_buf, (uint32_t)comp_sz, (uint32_t)smem_per_warp,
                d_decomp_buf, d_block_elapsed);
            BENCH_CHECK(cudaDeviceSynchronize());

            // Timed
            BENCH_CHECK(cudaEventRecord(ev0));
            balanced_fn<<<nb, BALANCED_BLOCK_SIZE, dyn_smem>>>(
                d_ctrls, d_pc, start_page, (uint32_t)n_devices, total_pages,
                d_comp_buf, (uint32_t)comp_sz, (uint32_t)smem_per_warp,
                d_decomp_buf, d_block_elapsed);
            BENCH_CHECK(cudaEventRecord(ev1));
            BENCH_CHECK(cudaEventSynchronize(ev1));

            float ms = 0;
            BENCH_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

            double bytes = (double)total_pages * PAGE_SIZE;
            double gbs   = bytes / (ms * 1e6);
            double ppm   = total_pages / (double)ms;

            fprintf(stderr, "  %-8d %-12.2f %-12.2f %-10.1f%s\n",
                    nb, ms, gbs, ppm,
                    gbs > 20.0 ? "  ← NVMe saturated" :
                    gbs > 15.0 ? "  ← near saturation" : "");
        }

        BENCH_CHECK(cudaEventDestroy(ev0));
        BENCH_CHECK(cudaEventDestroy(ev1));
        cudaFree(d_comp_buf);
        cudaFree(d_decomp_buf);
        cudaFree(d_block_elapsed);
        bam_io_page_cache_destroy(io_pc);
        delete[] h_page;
        delete[] h_comp;
    }

    bam_ctrl_close(ctrl);
    return 0;
}
