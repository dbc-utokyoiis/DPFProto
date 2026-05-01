// ============================================================
// IO Pipeline Microbenchmark v3 — Low-occupancy focus
//
// Part A: IO saturation sweep
//   How many async IO warps (1 warp/block) are needed to saturate NVMe?
//   Sweep num_blocks × ios_per_warp.
//
// Part B: Producer-consumer handoff
//   2 warps per block (IO + decomp), real IO + simulated decomp.
//   Measures IO throughput with handoff overhead.
//   Reports handoff latency via clock64() delta.
//
// Usage:
//   sudo ./samples/io_pipeline_bench <devices> <start_page> <npages> [decomp_us]
// ============================================================

#include "bam_io_device.cuh"
#include "bam_kernel.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define BENCH_CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                      \
                cudaGetErrorString(err), __FILE__, __LINE__);             \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

static constexpr uint32_t PAGE_SIZE = 1048576;
static constexpr uint32_t PAGE_NBLK = PAGE_SIZE / 512;
static constexpr int MAX_IOS = 16;
static constexpr int RING_SIZE = 8;

__device__ __forceinline__ void busy_wait_cycles(uint64_t cycles) {
    uint64_t start = clock64();
    while (clock64() - start < cycles) {}
}

// ============================================================
// Part A: IO-only async benchmark (1 warp per block)
// Completion-driven: submit N → try_poll round-robin → replace
// ============================================================
__global__ void kernel_io_async(
    void*       ctrls,
    void*       pc,
    uint32_t    total_pages,
    uint64_t    field_start_page_id,
    uint64_t*   d_partition_start_lbas,
    uint32_t    n_devices,
    uint32_t    n_ios)  // outstanding IOs per warp (<=MAX_IOS)
{
    const uint32_t lane = threadIdx.x;
    const uint32_t warp_id = blockIdx.x;
    const uint32_t total_warps = gridDim.x;

    if (lane != 0) return;

    void*    slot_qp[MAX_IOS];
    uint16_t slot_cid[MAX_IOS];
    bool     slot_active[MAX_IOS];
    for (int i = 0; i < MAX_IOS; i++) slot_active[i] = false;

    uint32_t next_page_idx = warp_id;
    uint32_t active_count = 0;
    const uint32_t slot_base = warp_id * n_ios;

    // Priming
    for (uint32_t i = 0; i < n_ios && next_page_idx < total_pages; i++) {
        uint64_t global_pg = field_start_page_id + next_page_idx;
        uint32_t dev = global_pg % n_devices;
        uint64_t local_pg = global_pg / n_devices;
        uint64_t lba = d_partition_start_lbas[dev] + local_pg * PAGE_NBLK;
        bam_io_submit_page_device(ctrls, pc, lba, PAGE_NBLK, slot_base + i, dev,
                                  &slot_qp[i], &slot_cid[i]);
        slot_active[i] = true;
        active_count++;
        next_page_idx += total_warps;
    }

    while (active_count > 0) {
        for (uint32_t i = 0; i < n_ios; i++) {
            if (!slot_active[i]) continue;
            if (bam_io_try_poll_page_device(slot_qp[i], slot_cid[i])) {
                slot_active[i] = false;
                active_count--;
                if (next_page_idx < total_pages) {
                    uint64_t global_pg = field_start_page_id + next_page_idx;
                    uint32_t dev = global_pg % n_devices;
                    uint64_t local_pg = global_pg / n_devices;
                    uint64_t lba = d_partition_start_lbas[dev] + local_pg * PAGE_NBLK;
                    bam_io_submit_page_device(ctrls, pc, lba, PAGE_NBLK,
                                              slot_base + i, dev,
                                              &slot_qp[i], &slot_cid[i]);
                    slot_active[i] = true;
                    active_count++;
                    next_page_idx += total_warps;
                }
            }
        }
    }
}

// ============================================================
// Part B: Separated IO + decomp (2 warps per block)
//
// Warp 0: IO producer — async submit/poll, signals via ring
// Warp 1: Decomp consumer — waits for signal, busy-waits, signals done
//
// Handoff latency measured via clock64() timestamps in shared memory.
// ============================================================

struct HandoffShmem {
    uint32_t io_done;         // pages completed by IO warp
    uint32_t decomp_done;     // pages consumed by decomp warp
    uint64_t io_done_ts;      // clock64() when IO warp signals
    uint64_t handoff_sum;     // sum of handoff latencies (cycles)
    uint32_t handoff_count;   // number of handoffs measured
    uint32_t total_pages_for_block;
};

// Per-block output
struct HandoffResult {
    uint64_t handoff_sum;
    uint32_t handoff_count;
};

__global__ void kernel_separated_handoff(
    void*           ctrls,
    void*           pc,
    uint32_t        total_pages,
    uint64_t        field_start_page_id,
    uint64_t*       d_partition_start_lbas,
    uint32_t        n_devices,
    uint32_t        n_ios,          // outstanding IOs for IO warp
    uint64_t        decomp_cycles,
    HandoffResult*  d_results)      // [num_blocks] output
{
    __shared__ HandoffShmem shm;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_in_block = tid / 32;
    const uint32_t lane = tid % 32;
    const uint32_t block_id = blockIdx.x;
    const uint32_t total_blocks = gridDim.x;

    // Init shared memory
    if (tid == 0) {
        shm.io_done = 0;
        shm.decomp_done = 0;
        shm.io_done_ts = 0;
        shm.handoff_sum = 0;
        shm.handoff_count = 0;
        uint32_t count = 0;
        for (uint32_t pg = block_id; pg < total_pages; pg += total_blocks)
            count++;
        shm.total_pages_for_block = count;
    }
    __syncthreads();

    if (lane != 0) return;

    const uint32_t my_total = shm.total_pages_for_block;

    if (warp_in_block == 0) {
        // ---- IO PRODUCER ----
        void*    slot_qp[MAX_IOS];
        uint16_t slot_cid[MAX_IOS];
        bool     slot_active[MAX_IOS];
        for (int i = 0; i < MAX_IOS; i++) slot_active[i] = false;

        uint32_t next_page_idx = block_id;
        uint32_t active_count = 0;
        uint32_t pages_done = 0;
        const uint32_t slot_base = block_id * n_ios;

        // Priming
        for (uint32_t i = 0; i < n_ios && next_page_idx < total_pages; i++) {
            uint64_t global_pg = field_start_page_id + next_page_idx;
            uint32_t dev = global_pg % n_devices;
            uint64_t local_pg = global_pg / n_devices;
            uint64_t lba = d_partition_start_lbas[dev] + local_pg * PAGE_NBLK;
            bam_io_submit_page_device(ctrls, pc, lba, PAGE_NBLK,
                                      slot_base + i, dev,
                                      &slot_qp[i], &slot_cid[i]);
            slot_active[i] = true;
            active_count++;
            next_page_idx += total_blocks;
        }

        while (active_count > 0) {
            for (uint32_t i = 0; i < n_ios; i++) {
                if (!slot_active[i]) continue;
                if (bam_io_try_poll_page_device(slot_qp[i], slot_cid[i])) {
                    slot_active[i] = false;
                    active_count--;
                    pages_done++;

                    // Back-pressure: wait if ring full
                    while (pages_done -
                           *(volatile uint32_t*)&shm.decomp_done >= RING_SIZE) {
                        __nanosleep(64);
                    }

                    // Signal with timestamp
                    uint64_t ts = clock64();
                    *(volatile uint64_t*)&shm.io_done_ts = ts;
                    __threadfence_block();
                    *(volatile uint32_t*)&shm.io_done = pages_done;
                    __threadfence_block();

                    // Submit replacement
                    if (next_page_idx < total_pages) {
                        uint64_t global_pg = field_start_page_id + next_page_idx;
                        uint32_t dev = global_pg % n_devices;
                        uint64_t local_pg = global_pg / n_devices;
                        uint64_t lba = d_partition_start_lbas[dev] + local_pg * PAGE_NBLK;
                        bam_io_submit_page_device(ctrls, pc, lba, PAGE_NBLK,
                                                  slot_base + i, dev,
                                                  &slot_qp[i], &slot_cid[i]);
                        slot_active[i] = true;
                        active_count++;
                        next_page_idx += total_blocks;
                    }
                }
            }
        }
    }
    else if (warp_in_block == 1) {
        // ---- DECOMP CONSUMER ----
        uint32_t consumed = 0;
        while (consumed < my_total) {
            // Wait for IO warp
            while (*(volatile uint32_t*)&shm.io_done <= consumed) {
                __nanosleep(64);
            }

            // Measure handoff latency
            uint64_t ts_io = *(volatile uint64_t*)&shm.io_done_ts;
            uint64_t ts_now = clock64();
            uint64_t delta = ts_now - ts_io;
            shm.handoff_sum += delta;
            shm.handoff_count++;

            // Simulated decompression
            if (decomp_cycles > 0) {
                busy_wait_cycles(decomp_cycles);
            }

            consumed++;
            *(volatile uint32_t*)&shm.decomp_done = consumed;
            __threadfence_block();
        }

        // Write results
        d_results[block_id].handoff_sum = shm.handoff_sum;
        d_results[block_id].handoff_count = shm.handoff_count;
    }
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

static uint64_t us_to_cycles(uint32_t us, int gpu_clock_khz) {
    return (uint64_t)us * gpu_clock_khz / 1000;
}

// ============================================================
// Run Part A: IO-only async
// ============================================================
struct BenchResult {
    double elapsed_ms;
    double io_throughput_gbs;
};

static BenchResult run_io_only(
    bam_ctrl_handle_t ctrl,
    uint32_t total_pages,
    uint64_t field_start_page_id,
    uint32_t n_devices,
    const uint64_t* partition_start_lbas,
    uint32_t num_blocks,
    uint32_t n_ios)
{
    uint32_t num_slots = num_blocks * n_ios;
    bam_io_page_cache_t io_pc = bam_io_page_cache_create(
        ctrl, PAGE_SIZE, num_slots);
    void* d_ctrls = bam_io_page_cache_get_d_ctrls(io_pc);
    void* d_pc = bam_io_page_cache_get_d_pc_ptr(io_pc);

    uint64_t* d_lbas;
    BENCH_CUDA_CHECK(cudaMalloc(&d_lbas, n_devices * sizeof(uint64_t)));
    BENCH_CUDA_CHECK(cudaMemcpy(d_lbas, partition_start_lbas,
                                n_devices * sizeof(uint64_t),
                                cudaMemcpyHostToDevice));

    // Warmup
    kernel_io_async<<<num_blocks, 32>>>(
        d_ctrls, d_pc, total_pages, field_start_page_id,
        d_lbas, n_devices, n_ios);
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    cudaEvent_t start, stop;
    BENCH_CUDA_CHECK(cudaEventCreate(&start));
    BENCH_CUDA_CHECK(cudaEventCreate(&stop));
    BENCH_CUDA_CHECK(cudaEventRecord(start));
    kernel_io_async<<<num_blocks, 32>>>(
        d_ctrls, d_pc, total_pages, field_start_page_id,
        d_lbas, n_devices, n_ios);
    BENCH_CUDA_CHECK(cudaEventRecord(stop));
    BENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    BENCH_CUDA_CHECK(cudaEventDestroy(start));
    BENCH_CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_lbas);
    bam_io_page_cache_destroy(io_pc);

    double bytes = (double)total_pages * PAGE_SIZE;
    return {(double)ms, bytes / (ms * 1e6)};
}

// ============================================================
// Run Part B: Separated IO+decomp with handoff measurement
// ============================================================
struct HandoffBenchResult {
    double elapsed_ms;
    double io_throughput_gbs;
    double avg_handoff_cycles;
    double avg_handoff_us;
};

static HandoffBenchResult run_separated(
    bam_ctrl_handle_t ctrl,
    uint32_t total_pages,
    uint64_t field_start_page_id,
    uint32_t n_devices,
    const uint64_t* partition_start_lbas,
    uint32_t num_blocks,
    uint32_t n_ios,
    uint64_t decomp_cycles,
    int gpu_clock_khz)
{
    uint32_t num_slots = num_blocks * n_ios;
    bam_io_page_cache_t io_pc = bam_io_page_cache_create(
        ctrl, PAGE_SIZE, num_slots);
    void* d_ctrls = bam_io_page_cache_get_d_ctrls(io_pc);
    void* d_pc = bam_io_page_cache_get_d_pc_ptr(io_pc);

    uint64_t* d_lbas;
    BENCH_CUDA_CHECK(cudaMalloc(&d_lbas, n_devices * sizeof(uint64_t)));
    BENCH_CUDA_CHECK(cudaMemcpy(d_lbas, partition_start_lbas,
                                n_devices * sizeof(uint64_t),
                                cudaMemcpyHostToDevice));

    HandoffResult* d_results;
    BENCH_CUDA_CHECK(cudaMalloc(&d_results, num_blocks * sizeof(HandoffResult)));
    BENCH_CUDA_CHECK(cudaMemset(d_results, 0, num_blocks * sizeof(HandoffResult)));

    // Warmup
    kernel_separated_handoff<<<num_blocks, 64>>>(
        d_ctrls, d_pc, total_pages, field_start_page_id,
        d_lbas, n_devices, n_ios, decomp_cycles, d_results);
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Clear results for timed run
    BENCH_CUDA_CHECK(cudaMemset(d_results, 0, num_blocks * sizeof(HandoffResult)));

    // Timed
    cudaEvent_t start, stop;
    BENCH_CUDA_CHECK(cudaEventCreate(&start));
    BENCH_CUDA_CHECK(cudaEventCreate(&stop));
    BENCH_CUDA_CHECK(cudaEventRecord(start));
    kernel_separated_handoff<<<num_blocks, 64>>>(
        d_ctrls, d_pc, total_pages, field_start_page_id,
        d_lbas, n_devices, n_ios, decomp_cycles, d_results);
    BENCH_CUDA_CHECK(cudaEventRecord(stop));
    BENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Read handoff results
    HandoffResult* h_results = new HandoffResult[num_blocks];
    BENCH_CUDA_CHECK(cudaMemcpy(h_results, d_results,
                                num_blocks * sizeof(HandoffResult),
                                cudaMemcpyDeviceToHost));

    uint64_t total_sum = 0;
    uint32_t total_count = 0;
    for (uint32_t i = 0; i < num_blocks; i++) {
        total_sum += h_results[i].handoff_sum;
        total_count += h_results[i].handoff_count;
    }
    double avg_cycles = (total_count > 0) ? (double)total_sum / total_count : 0;
    double avg_us = avg_cycles / (gpu_clock_khz / 1000.0);

    delete[] h_results;
    BENCH_CUDA_CHECK(cudaEventDestroy(start));
    BENCH_CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_lbas);
    cudaFree(d_results);
    bam_io_page_cache_destroy(io_pc);

    double bytes = (double)total_pages * PAGE_SIZE;
    return {(double)ms, bytes / (ms * 1e6), avg_cycles, avg_us};
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <devices> <start_page> <npages> [decomp_us]\n"
            "\n"
            "Part A: IO saturation sweep — find minimum warps to saturate NVMe\n"
            "Part B: IO+decomp handoff — measure signaling overhead\n"
            "\n"
            "Example:\n"
            "  sudo %s /dev/libnvm0,...,/dev/libnvm3 48241 6912 50\n",
            argv[0], argv[0]);
        return 1;
    }

    char dev_paths[8][256];
    int n_devices = parse_devices(argv[1], dev_paths, 8);
    uint64_t start_page = strtoull(argv[2], nullptr, 10);
    uint32_t npages = (uint32_t)strtoul(argv[3], nullptr, 10);
    uint32_t decomp_us = (argc > 4) ? (uint32_t)strtoul(argv[4], nullptr, 10) : 50;

    cudaSetDevice(0);

    int gpu_clock_khz = 0;
    BENCH_CUDA_CHECK(cudaDeviceGetAttribute(&gpu_clock_khz,
                                             cudaDevAttrClockRate, 0));
    uint64_t decomp_cycles = us_to_cycles(decomp_us, gpu_clock_khz);

    fprintf(stderr, "=== IO Pipeline Benchmark v3 ===\n");
    fprintf(stderr, "Devices: %d, Pages: %u (%u MiB)\n", n_devices, npages, npages);
    fprintf(stderr, "GPU clock: %.0f MHz, Decomp: %u us = %lu cycles\n",
            gpu_clock_khz / 1000.0, decomp_us, decomp_cycles);
    fprintf(stderr, "\n");

    const char* paths[8];
    for (int i = 0; i < n_devices; i++) paths[i] = dev_paths[i];
    bam_ctrl_handle_t ctrl = bam_ctrl_open_multi(
        paths, n_devices, 1, 0, 1024, 128);

    uint64_t partition_start_lbas[8];
    for (int i = 0; i < n_devices; i++)
        partition_start_lbas[i] = 2048;

    // ================================================================
    // Part A: IO Saturation Sweep
    // ================================================================
    fprintf(stderr, "=== Part A: IO Saturation (1 warp/block, async) ===\n");
    fprintf(stderr, "%-8s %-8s %-12s %-10s %-12s\n",
            "blocks", "ios/w", "outstanding", "time(ms)", "IO GB/s");
    fprintf(stderr, "%-8s %-8s %-12s %-10s %-12s\n",
            "------", "------", "----------", "--------", "--------");

    uint32_t block_sweep[] = {27, 54, 108, 216, 432, 864, 1728, 3456};
    uint32_t ios_sweep[] = {1, 2, 4, 8};

    for (uint32_t ios : ios_sweep) {
        for (uint32_t nb : block_sweep) {
            auto r = run_io_only(ctrl, npages, start_page, n_devices,
                                 partition_start_lbas, nb, ios);
            fprintf(stderr, "%-8u %-8u %-12u %-10.1f %-12.2f\n",
                    nb, ios, nb * ios, r.elapsed_ms, r.io_throughput_gbs);
        }
        fprintf(stderr, "\n");
    }

    // ================================================================
    // Part B: Handoff Throughput (2 warps/block: IO + decomp)
    // ================================================================
    fprintf(stderr, "=== Part B: IO + Decomp Handoff ===\n");

    // B1: Handoff overhead (decomp=0, compare with io_only)
    fprintf(stderr, "\n--- B1: Handoff Overhead (decomp=0, ios/w=4) ---\n");
    fprintf(stderr, "%-8s %-12s %-12s %-12s %-12s\n",
            "blocks", "io_only", "separated", "overhead%", "handoff_us");
    fprintf(stderr, "%-8s %-12s %-12s %-12s %-12s\n",
            "------", "--------", "--------", "--------", "--------");

    uint32_t b1_blocks[] = {54, 108, 216, 432};
    for (uint32_t nb : b1_blocks) {
        auto rio = run_io_only(ctrl, npages, start_page, n_devices,
                               partition_start_lbas, nb, 4);
        auto rse = run_separated(ctrl, npages, start_page, n_devices,
                                 partition_start_lbas, nb, 4, 0, gpu_clock_khz);
        double overhead = (rse.elapsed_ms / rio.elapsed_ms - 1.0) * 100;
        fprintf(stderr, "%-8u %-12.2f %-12.2f %-11.1f%% %-12.2f\n",
                nb, rio.io_throughput_gbs, rse.io_throughput_gbs,
                overhead, rse.avg_handoff_us);
    }

    // B2: Decomp latency sweep (108 blocks, ios/w=4)
    fprintf(stderr, "\n--- B2: Decomp Sweep (108 blocks, ios/w=4) ---\n");
    fprintf(stderr, "%-10s %-12s %-12s %-12s %-12s\n",
            "decomp_us", "io_only", "separated", "overhead%", "handoff_us");
    fprintf(stderr, "%-10s %-12s %-12s %-12s %-12s\n",
            "--------", "--------", "--------", "--------", "--------");

    uint32_t d_sweep[] = {0, 10, 25, 50, 100, 200, 500};
    auto rio_base = run_io_only(ctrl, npages, start_page, n_devices,
                                partition_start_lbas, 108, 4);
    for (uint32_t d_us : d_sweep) {
        uint64_t d_cyc = us_to_cycles(d_us, gpu_clock_khz);
        auto rse = run_separated(ctrl, npages, start_page, n_devices,
                                 partition_start_lbas, 108, 4,
                                 d_cyc, gpu_clock_khz);
        double overhead = (rse.elapsed_ms / rio_base.elapsed_ms - 1.0) * 100;
        fprintf(stderr, "%-10u %-12.2f %-12.2f %-11.1f%% %-12.2f\n",
                d_us, rio_base.io_throughput_gbs, rse.io_throughput_gbs,
                overhead, rse.avg_handoff_us);
    }

    // B3: ios_per_warp sweep for separated (108 blocks, decomp=50us)
    fprintf(stderr, "\n--- B3: IOs/warp Sweep (108 blocks, decomp=%u us) ---\n",
            decomp_us);
    fprintf(stderr, "%-8s %-12s %-12s %-12s\n",
            "ios/w", "io_only", "separated", "handoff_us");
    fprintf(stderr, "%-8s %-12s %-12s %-12s\n",
            "------", "--------", "--------", "--------");

    uint32_t io_sweep2[] = {1, 2, 4, 8};
    for (uint32_t ios : io_sweep2) {
        auto rio2 = run_io_only(ctrl, npages, start_page, n_devices,
                                partition_start_lbas, 108, ios);
        auto rse2 = run_separated(ctrl, npages, start_page, n_devices,
                                  partition_start_lbas, 108, ios,
                                  decomp_cycles, gpu_clock_khz);
        fprintf(stderr, "%-8u %-12.2f %-12.2f %-12.2f\n",
                ios, rio2.io_throughput_gbs, rse2.io_throughput_gbs,
                rse2.avg_handoff_us);
    }

    bam_ctrl_close(ctrl);
    return 0;
}
