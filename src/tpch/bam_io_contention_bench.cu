// ============================================================
// BaM I/O contention microbenchmark — completion-driven I/O loop.
//
// Each warp manages N outstanding async I/Os:
//   submit N → round-robin try_poll → on completion, submit replacement
//
// Compiled as C++17 with separable compilation + device linking
// (calls bam_io_submit_page_device / bam_io_try_poll_page_device
//  from bam_io_device via device linking).
// ============================================================

#include "bam_io_contention_bench.cuh"
#include "bam_io_device.cuh"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#define BENCH_CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                      \
                cudaGetErrorString(err), __FILE__, __LINE__);             \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

// Maximum outstanding IOs per warp (compile-time bound for register arrays).
static constexpr int BENCH_MAX_IOS = 64;

// ── Kernel ──
// 1 warp per block, 32 threads. Only lane 0 does I/O management.
__global__ void bam_io_bench_kernel(
    void*       ctrls,
    void*       pc,
    uint32_t    total_pages,
    uint32_t    n_ios_per_warp,     // actual N, <= BENCH_MAX_IOS
    uint64_t    field_start_page_id,
    uint64_t*   d_partition_start_lbas,
    uint32_t    page_nblk,          // page_size / 512
    uint32_t    n_devices,
    uint32_t    page_size)
{
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp_id = blockIdx.x;
    const uint32_t total_warps = gridDim.x;

    if (warp_id >= total_warps || lane != 0) return;

    const uint32_t N = n_ios_per_warp;
    const uint32_t slot_base = warp_id * N;

    // Per-slot state (registers)
    void*    slot_qp[BENCH_MAX_IOS];
    uint16_t slot_cid[BENCH_MAX_IOS];
    bool     slot_active[BENCH_MAX_IOS];
    for (int i = 0; i < BENCH_MAX_IOS; i++) slot_active[i] = false;

    uint32_t next_page_idx = warp_id;  // interleaved assignment
    uint32_t active_count = 0;

    // ── Priming: submit initial N I/Os ──
    for (uint32_t i = 0; i < N && next_page_idx < total_pages; i++) {
        uint64_t global_pg = field_start_page_id + next_page_idx;
        uint32_t dev = global_pg % n_devices;
        uint64_t local_pg = global_pg / n_devices;
        uint64_t lba = d_partition_start_lbas[dev] + local_pg * page_nblk;
        uint32_t slot = slot_base + i;

        bam_io_submit_page_device(ctrls, pc, lba, page_nblk, slot, dev,
                                  &slot_qp[i], &slot_cid[i]);
        slot_active[i] = true;
        active_count++;
        next_page_idx += total_warps;
    }

    // ── Completion-driven main loop ──
    while (active_count > 0) {
        bool any_completed = false;
        for (uint32_t i = 0; i < N; i++) {
            if (!slot_active[i]) continue;

            if (bam_io_try_poll_page_device(slot_qp[i], slot_cid[i])) {
                slot_active[i] = false;
                active_count--;
                any_completed = true;

                // Submit replacement if pages remain
                if (next_page_idx < total_pages) {
                    uint64_t global_pg = field_start_page_id + next_page_idx;
                    uint32_t dev = global_pg % n_devices;
                    uint64_t local_pg = global_pg / n_devices;
                    uint64_t lba = d_partition_start_lbas[dev] + local_pg * page_nblk;
                    uint32_t slot = slot_base + i;

                    bam_io_submit_page_device(ctrls, pc, lba, page_nblk, slot, dev,
                                              &slot_qp[i], &slot_cid[i]);
                    slot_active[i] = true;
                    active_count++;
                    next_page_idx += total_warps;
                }
            }
        }
        if (!any_completed) __nanosleep(256);
    }
}

// ── Host API ──

BamIoBenchResult bam_io_contention_bench_run(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t total_pages,
    uint64_t field_start_page_id,
    uint32_t n_devices,
    const uint64_t* partition_start_lbas,
    const BamIoBenchConfig& config)
{
    const uint32_t num_warps = config.num_warps;
    const uint32_t ios_per_warp = config.ios_per_warp;
    const uint32_t num_slots = num_warps * ios_per_warp;
    const uint32_t page_nblk = page_size / 512;

    // Create page_cache
    bam_io_page_cache_t io_pc = bam_io_page_cache_create(
        ctrl_handle, page_size, num_slots);

    void* d_ctrls = bam_io_page_cache_get_d_ctrls(io_pc);
    void* d_pc    = bam_io_page_cache_get_d_pc_ptr(io_pc);

    // Copy partition_start_lbas to device
    uint64_t* d_partition_start_lbas;
    BENCH_CUDA_CHECK(cudaMalloc(&d_partition_start_lbas,
                                n_devices * sizeof(uint64_t)));
    BENCH_CUDA_CHECK(cudaMemcpy(d_partition_start_lbas, partition_start_lbas,
                                n_devices * sizeof(uint64_t),
                                cudaMemcpyHostToDevice));

    // Warmup run
    bam_io_bench_kernel<<<num_warps, 32>>>(
        d_ctrls, d_pc, total_pages, ios_per_warp,
        field_start_page_id, d_partition_start_lbas,
        page_nblk, n_devices, page_size);
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    BENCH_CUDA_CHECK(cudaEventCreate(&start));
    BENCH_CUDA_CHECK(cudaEventCreate(&stop));

    BENCH_CUDA_CHECK(cudaEventRecord(start));
    bam_io_bench_kernel<<<num_warps, 32>>>(
        d_ctrls, d_pc, total_pages, ios_per_warp,
        field_start_page_id, d_partition_start_lbas,
        page_nblk, n_devices, page_size);
    BENCH_CUDA_CHECK(cudaEventRecord(stop));
    BENCH_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    BENCH_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Compute throughput
    double total_bytes = (double)total_pages * page_size;
    double throughput_gbs = total_bytes / (elapsed_ms * 1e6);

    // n_qps per device = 128 (standard BaM config)
    double total_qps = 128.0 * n_devices;
    double warps_per_qp = (double)num_warps / total_qps;

    BamIoBenchResult result;
    result.num_warps = num_warps;
    result.ios_per_warp = ios_per_warp;
    result.total_outstanding = num_warps * ios_per_warp;
    result.total_pages = total_pages;
    result.elapsed_ms = elapsed_ms;
    result.io_throughput_gbs = throughput_gbs;
    result.warps_per_qp = warps_per_qp;

    // Cleanup
    BENCH_CUDA_CHECK(cudaEventDestroy(start));
    BENCH_CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_partition_start_lbas);
    bam_io_page_cache_destroy(io_pc);

    return result;
}
