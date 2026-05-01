// ============================================================
// BaM Submit/Poll Separation Test
//
// Tests whether splitting GPU-initiated NVMe IO into separate
// submit and poll kernels works reliably.
//
// Test 1 (Fused baseline): single kernel does submit+poll+copy
// Test 2 (SP split): submit kernel → sync → poll+copy kernel
// Test 3 (SP multi-batch): repeat SP across multiple batches
//                          to detect intermittent deadlock
//
// Usage:
//   sudo ./samples/bam_sp_test <devices> <start_page> <npages> [batch_size]
//
// Example:
//   sudo ./samples/bam_sp_test /dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3 608 48
//   sudo ./samples/bam_sp_test /dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3 608 48 16
// ============================================================

#include "bam_io_device.cuh"
#include "bam_kernel.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>

#define SP_CUDA_CHECK(call) do {                                             \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                         \
                cudaGetErrorString(err), __FILE__, __LINE__);                \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

static constexpr uint32_t PAGE_SIZE = 1048576;
static constexpr uint32_t PAGE_NBLK = PAGE_SIZE / 512;

// ============================================================
// Kernel 1: Fused read (baseline) — submit+poll+copy in one kernel
// Each block reads one page via bam_io_read_page_device, then
// copies from page cache to destination buffer.
// ============================================================
__global__ void fused_read_kernel(
    void*    d_ctrls,
    void*    d_pc,
    void*    pc_base,
    uint32_t total_pages,
    uint64_t field_start_page,
    uint64_t* d_partition_lbas,
    uint32_t n_devices,
    char*    d_output,       // output buffer (total_pages * PAGE_SIZE)
    uint32_t page_size)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;

    for (uint32_t pg = bid; pg < total_pages; pg += gridDim.x) {
        uint64_t global_pg = field_start_page + pg;
        uint32_t dev = global_pg % n_devices;
        uint64_t local_pg = global_pg / n_devices;
        uint64_t lba = d_partition_lbas[dev] + local_pg * PAGE_NBLK;

        if (tid == 0) {
            bam_io_read_page_device(d_ctrls, d_pc, lba, PAGE_NBLK, bid, dev);
        }
        __syncthreads();

        // Copy from page cache slot to output
        const char* src = (const char*)pc_base + (uint64_t)bid * page_size;
        char* dst = d_output + (uint64_t)pg * page_size;
        const uint32_t n4 = page_size / 4;
        for (uint32_t i = tid; i < n4; i += nthreads) {
            ((uint32_t*)dst)[i] = ((const uint32_t*)src)[i];
        }
        __syncthreads();
    }
}

// ============================================================
// Kernel 2a: Submit-only kernel
// Each block round-robins through descriptors, submitting NVMe reads.
// Saves QP pointer and CID for later poll.
// ============================================================
struct SPDesc {
    uint64_t lba;
    uint32_t nblocks;
    uint32_t dev;
    uint32_t slot;    // page cache slot index
    uint32_t pg_idx;  // original page index (for output copy)
};

__global__ void submit_kernel(
    void*         d_ctrls,
    void*         d_pc,
    const SPDesc* __restrict__ d_descs,
    uint32_t      ndescs,
    void**  __restrict__ d_qp_ptrs,
    uint16_t* __restrict__ d_cids)
{
    // Single thread per block (avoids intra-warp deadlock in BaM sq_enqueue)
    for (uint32_t j = blockIdx.x; j < ndescs; j += gridDim.x) {
        const SPDesc& d = d_descs[j];
        bam_io_submit_page_device(d_ctrls, d_pc, d.lba, d.nblocks,
                                   d.slot, d.dev,
                                   &d_qp_ptrs[j], &d_cids[j]);
    }
}

// ============================================================
// Kernel 2b: Poll+Copy kernel
// Each block round-robins through descriptors, polls for NVMe
// completion, then copies data from page cache to output.
// ============================================================
__global__ void poll_copy_kernel(
    void*          pc_base,
    const SPDesc*  __restrict__ d_descs,
    uint32_t       ndescs,
    uint32_t       page_size,
    void* const*   __restrict__ d_qp_ptrs,
    const uint16_t* __restrict__ d_cids,
    char*          d_output)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;

    for (uint32_t j = bid; j < ndescs; j += gridDim.x) {
        if (tid == 0) {
            // Poll with timeout diagnostic
            bool ok = false;
            for (uint64_t attempt = 0; attempt < 500000000ULL; attempt++) {
                ok = bam_io_try_poll_page_device(d_qp_ptrs[j], d_cids[j]);
                if (ok) break;
            }
            if (!ok) {
                printf("[POLL TIMEOUT] j=%u ndescs=%u slot=%u cid=%u qp=%p\n",
                       j, ndescs, d_descs[j].slot, (unsigned)d_cids[j], d_qp_ptrs[j]);
                // Blocking poll as last resort
                bam_io_poll_page_device(d_qp_ptrs[j], d_cids[j]);
            }
        }
        __syncthreads();

        // Copy from page cache slot to output
        const SPDesc& d = d_descs[j];
        const char* src = (const char*)pc_base + (uint64_t)d.slot * page_size;
        char* dst = d_output + (uint64_t)d.pg_idx * page_size;
        const uint32_t n4 = page_size / 4;
        for (uint32_t i = tid; i < n4; i += nthreads) {
            ((uint32_t*)dst)[i] = ((const uint32_t*)src)[i];
        }
        __syncthreads();
    }
}

// ============================================================
// FIFO entry for async submit/poll (Test 7)
// ============================================================
struct FifoEntry {
    void*    qp_ptr;
    uint16_t cid;
    uint16_t _pad;
    uint32_t slot;
    uint32_t copy_bytes;
    char*    dest;
};

// ============================================================
// Kernel 3a: FIFO Submit kernel (producer)
// Submits NVMe commands and pushes (qp, cid, dest) to FIFO.
// ============================================================
__global__ void fifo_submit_kernel(
    void*         d_ctrls,
    void*         d_pc,
    const SPDesc* __restrict__ d_descs,
    uint32_t      ndescs,
    FifoEntry*    __restrict__ d_fifo,
    uint32_t*     __restrict__ d_write_pos,
    char*         d_output,
    void*         pc_base,
    uint32_t      page_size)
{
    for (uint32_t j = blockIdx.x; j < ndescs; j += gridDim.x) {
        const SPDesc& d = d_descs[j];
        void* qp_ptr;
        uint16_t cid;
        bam_io_submit_page_device(d_ctrls, d_pc, d.lba, d.nblocks,
                                   d.slot, d.dev, &qp_ptr, &cid);

        // Allocate FIFO slot (atomic increment)
        uint32_t pos = atomicAdd(d_write_pos, 1);

        // Write entry (dest computed from pg_idx)
        FifoEntry entry;
        entry.qp_ptr    = qp_ptr;
        entry.cid       = cid;
        entry._pad      = 0;
        entry.slot       = d.slot;
        entry.copy_bytes = page_size;
        entry.dest       = d_output + (uint64_t)d.pg_idx * page_size;
        d_fifo[pos] = entry;

        // Fence: ensure entry is visible before write_pos is observed by consumer
        __threadfence();
    }
}

// ============================================================
// Kernel 3b: FIFO Poll+Copy kernel (consumer)
// Pops entries from FIFO in order, polls for completion, copies.
// ============================================================
__global__ void fifo_poll_copy_kernel(
    void*            pc_base,
    const FifoEntry* __restrict__ d_fifo,
    uint32_t*        __restrict__ d_write_pos,
    uint32_t*        __restrict__ d_read_pos,
    uint32_t         total_descs,
    uint32_t         page_size,
    char*            d_output)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;

    while (true) {
        // Grab next FIFO slot
        uint32_t my_pos;
        if (tid == 0) {
            my_pos = atomicAdd(d_read_pos, 1);
        }
        // Broadcast to all threads in block
        my_pos = __shfl_sync(0xFFFFFFFF, my_pos, 0);
        // For blocks with >32 threads, use shared memory
        __shared__ uint32_t s_pos;
        if (tid == 0) s_pos = my_pos;
        __syncthreads();
        my_pos = s_pos;

        if (my_pos >= total_descs) break;

        // Spin-wait for producer to fill this FIFO slot
        if (tid == 0) {
            while (atomicAdd(d_write_pos, 0) <= my_pos) {
                __nanosleep(100);
            }
        }
        __syncthreads();

        // Read FIFO entry
        const FifoEntry& e = d_fifo[my_pos];

        // Poll for NVMe completion
        if (tid == 0) {
            bam_io_poll_page_device(e.qp_ptr, e.cid);
        }
        __syncthreads();

        // Coalesced copy from page_cache slot to destination
        const char* src = (const char*)pc_base + (uint64_t)e.slot * page_size;
        char* dst = e.dest;
        const uint32_t n4 = e.copy_bytes / 4;
        for (uint32_t i = tid; i < n4; i += nthreads) {
            ((uint32_t*)dst)[i] = ((const uint32_t*)src)[i];
        }
        __syncthreads();
    }
}

// ============================================================
// Test 8: Per-QP CQ-order consumer
// CID reverse map: given (qp_flat_idx, cid), find page cache slot and dest.
// ============================================================
struct CidPageInfo {
    uint32_t slot;     // page cache slot
    uint32_t pg_idx;   // output page index
};

// Kernel 4a: Submit + build CID reverse map
// For each desc, submits NVMe read and records (qp_flat_idx, cid) → (slot, pg_idx)
__global__ void cqorder_submit_kernel(
    void*         d_ctrls,
    void*         d_pc,
    const SPDesc* __restrict__ d_descs,
    uint32_t      ndescs,
    uint32_t      n_qps,       // QPs per device
    uint32_t      n_devices,
    CidPageInfo*  __restrict__ d_cid_map,     // [total_qps * MAX_CID]
    uint32_t      max_cid,                    // MAX_CID dimension
    uint32_t*     __restrict__ d_qp_nsubmit)  // [total_qps] — atomic count per QP
{
    for (uint32_t j = blockIdx.x; j < ndescs; j += gridDim.x) {
        const SPDesc& d = d_descs[j];
        void* qp_ptr;
        uint16_t cid;
        bam_io_submit_page_device(d_ctrls, d_pc, d.lba, d.nblocks,
                                   d.slot, d.dev, &qp_ptr, &cid);

        // Compute flat QP index: dev * n_qps + (slot % n_qps)
        uint32_t qp_idx = d.slot % n_qps;
        uint32_t flat_qp = d.dev * n_qps + qp_idx;

        // Record reverse map
        d_cid_map[flat_qp * max_cid + cid].slot   = d.slot;
        d_cid_map[flat_qp * max_cid + cid].pg_idx = d.pg_idx;

        // Fence: ensure CID map entry is globally visible before
        // NVMe completion can be observed by the poll kernel.
        __threadfence();

        // Increment per-QP submit counter
        atomicAdd(&d_qp_nsubmit[flat_qp], 1);
    }
}

// Kernel 4b: Per-QP CQ-order consumer
// One block per QP. Consumes completions in CQ order, looks up CID→page info.
__global__ void cqorder_poll_copy_kernel(
    void**        __restrict__ d_qp_ptrs_arr,  // [total_qps] — QP device pointers
    uint32_t      total_qps,
    const CidPageInfo* __restrict__ d_cid_map,
    uint32_t      max_cid,
    const uint32_t* __restrict__ d_qp_nsubmit,
    void*         pc_base,
    uint32_t      page_size,
    char*         d_output)
{
    const uint32_t qp_flat = blockIdx.x;
    if (qp_flat >= total_qps) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;

    uint32_t n_to_consume = d_qp_nsubmit[qp_flat];
    if (n_to_consume == 0) return;

    void* qp = d_qp_ptrs_arr[qp_flat];

    for (uint32_t i = 0; i < n_to_consume; i++) {
        // Thread 0 polls CQ head — gets whatever CID completed next
        __shared__ uint16_t s_cid;
        if (tid == 0) {
            s_cid = bam_io_poll_next_page_device(qp);
        }
        __syncthreads();

        // Look up CID → page info
        uint16_t cid = s_cid;
        const CidPageInfo& info = d_cid_map[qp_flat * max_cid + cid];

        // Coalesced copy from page cache slot to output
        const char* src = (const char*)pc_base + (uint64_t)info.slot * page_size;
        char* dst = d_output + (uint64_t)info.pg_idx * page_size;
        const uint32_t n4 = page_size / 4;
        for (uint32_t w = tid; w < n4; w += nthreads) {
            ((uint32_t*)dst)[w] = ((const uint32_t*)src)[w];
        }
        __syncthreads();
    }
}

// ============================================================
// Host: parse device list, open controllers
// ============================================================
static void parse_devices(const char* arg,
                          char devices[][256], uint32_t& n_devices)
{
    n_devices = 0;
    const char* p = arg;
    while (*p && n_devices < MAX_BAM_DEVICES) {
        const char* comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);
        memcpy(devices[n_devices], p, len);
        devices[n_devices][len] = '\0';
        n_devices++;
        p = comma ? comma + 1 : p + len;
    }
}

static uint32_t checksum_pages(const char* buf, uint32_t npages, uint32_t page_size)
{
    uint32_t sum = 0;
    const uint32_t* p = (const uint32_t*)buf;
    uint64_t n4 = (uint64_t)npages * page_size / 4;
    for (uint64_t i = 0; i < n4; i++) sum += p[i];
    return sum;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <devices> <start_page> <npages> [batch_size] [num_batches] [--only TEST]\n", argv[0]);
        fprintf(stderr, "  devices: comma-separated /dev/libnvmN\n");
        fprintf(stderr, "  start_page: starting page ID on disk\n");
        fprintf(stderr, "  npages: total pages to read\n");
        fprintf(stderr, "  batch_size: pages per SP batch (default=npages)\n");
        fprintf(stderr, "  num_batches: repeat count for multi-batch test (default=10)\n");
        fprintf(stderr, "  --only TEST: run only specified test (1,2,3,4,5a,5b,5c,6,7,8,9)\n");
        fprintf(stderr, "  --nfields N: number of fields for Test 6 (default=7)\n");
        fprintf(stderr, "  --same-lba:  all fields read same LBA range (Test 6)\n");
        fprintf(stderr, "  --field-spacing N: pages between field starts (default=npages, 0=same-lba)\n");
        return 1;
    }

    char dev_paths[MAX_BAM_DEVICES][256];
    uint32_t n_devices = 0;
    parse_devices(argv[1], dev_paths, n_devices);

    uint64_t start_page = strtoull(argv[2], nullptr, 10);
    uint32_t npages     = (uint32_t)strtoul(argv[3], nullptr, 10);
    uint32_t batch_size = (argc > 4) ? (uint32_t)strtoul(argv[4], nullptr, 10) : npages;
    uint32_t num_batches = (argc > 5) ? (uint32_t)strtoul(argv[5], nullptr, 10) : 10;

    // Parse --only and --nfields flags
    const char* only_test = nullptr;
    uint32_t test6_nfields = 7;
    bool test6_same_lba = false;
    int64_t test6_field_spacing = -1;  // -1 = use npages (default)
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--only") == 0) {
            only_test = argv[i + 1];
        } else if (strcmp(argv[i], "--nfields") == 0) {
            test6_nfields = (uint32_t)strtoul(argv[i + 1], nullptr, 10);
        } else if (strcmp(argv[i], "--field-spacing") == 0) {
            test6_field_spacing = strtol(argv[i + 1], nullptr, 10);
        }
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--same-lba") == 0) test6_same_lba = true;
    }

    if (batch_size > npages) batch_size = npages;

    auto should_run = [&](const char* name) -> bool {
        if (!only_test) return true;
        return strstr(name, only_test) != nullptr || strcmp(only_test, name) == 0;
    };

    printf("=== BaM Submit/Poll Separation Test ===\n");
    if (only_test) printf("Running only: %s\n", only_test);
    printf("Devices: %u\n", n_devices);
    for (uint32_t i = 0; i < n_devices; i++)
        printf("  [%u] %s\n", i, dev_paths[i]);
    printf("Start page: %lu, npages: %u, batch_size: %u, num_batches: %u, nfields: %u\n",
           start_page, npages, batch_size, num_batches, test6_nfields);

    // Open BaM controllers
    const char* paths[MAX_BAM_DEVICES];
    for (uint32_t i = 0; i < n_devices; i++) paths[i] = dev_paths[i];
    bam_ctrl_handle_t ctrl = bam_ctrl_open_multi(paths, n_devices, 1, 0, 1024, 128);
    if (!ctrl) { fprintf(stderr, "Failed to open BaM controllers\n"); return 1; }

    // Get partition LBAs (assume 2048 for all devices)
    uint64_t h_partition_lbas[MAX_BAM_DEVICES];
    for (uint32_t i = 0; i < n_devices; i++) h_partition_lbas[i] = 2048;

    uint64_t* d_partition_lbas = nullptr;
    SP_CUDA_CHECK(cudaMalloc(&d_partition_lbas, MAX_BAM_DEVICES * sizeof(uint64_t)));
    SP_CUDA_CHECK(cudaMemcpy(d_partition_lbas, h_partition_lbas,
                             MAX_BAM_DEVICES * sizeof(uint64_t), cudaMemcpyHostToDevice));

    uint64_t total_bytes = (uint64_t)npages * PAGE_SIZE;
    printf("Total data: %lu MB\n", total_bytes / (1024 * 1024));

    // Allocate output buffers
    char* d_output_fused = nullptr;
    char* d_output_sp    = nullptr;
    char* h_output_fused = nullptr;
    char* h_output_sp    = nullptr;
    SP_CUDA_CHECK(cudaMalloc(&d_output_fused, total_bytes));
    SP_CUDA_CHECK(cudaMalloc(&d_output_sp,    total_bytes));
    SP_CUDA_CHECK(cudaMallocHost(&h_output_fused, total_bytes));
    SP_CUDA_CHECK(cudaMallocHost(&h_output_sp,    total_bytes));

    int sm_count = 0;
    SP_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    printf("SM count: %d\n", sm_count);

    cudaStream_t stream;
    SP_CUDA_CHECK(cudaStreamCreate(&stream));

    // ────────────────────────────────────────────────────────
    // Test 1: Fused baseline read
    // ────────────────────────────────────────────────────────
    if (should_run("1")) {
    printf("\n--- Test 1: Fused baseline read ---\n");
    {
        uint32_t num_blocks = (npages < (uint32_t)sm_count) ? npages : (uint32_t)sm_count;
        bam_io_page_cache_t pc = bam_io_page_cache_create(ctrl, PAGE_SIZE, num_blocks);
        void* d_ctrls  = bam_io_page_cache_get_d_ctrls(pc);
        void* d_pc     = bam_io_page_cache_get_d_pc_ptr(pc);
        void* pc_base  = bam_io_page_cache_get_base_addr(pc);

        SP_CUDA_CHECK(cudaMemset(d_output_fused, 0, total_bytes));

        auto t0 = std::chrono::steady_clock::now();
        fused_read_kernel<<<num_blocks, 128, 0, stream>>>(
            d_ctrls, d_pc, pc_base,
            npages, start_page, d_partition_lbas, n_devices,
            d_output_fused, PAGE_SIZE);
        SP_CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbs = (double)total_bytes / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        printf("Fused: %.2f ms, %.2f GB/s\n", ms, gbs);

        SP_CUDA_CHECK(cudaMemcpy(h_output_fused, d_output_fused, total_bytes, cudaMemcpyDeviceToHost));
        uint32_t cksum_fused = checksum_pages(h_output_fused, npages, PAGE_SIZE);
        printf("Fused checksum: 0x%08x\n", cksum_fused);

        bam_io_page_cache_destroy(pc);
    }
    } // Test 1

    // ────────────────────────────────────────────────────────
    // Test 2: Submit/Poll split (single batch = all pages)
    // ────────────────────────────────────────────────────────
    if (should_run("2")) {
    printf("\n--- Test 2: Submit/Poll split (single batch, %u pages) ---\n", npages);
    {
        // Page cache needs one slot per descriptor
        bam_io_page_cache_t pc = bam_io_page_cache_create(ctrl, PAGE_SIZE, npages);
        void* d_ctrls  = bam_io_page_cache_get_d_ctrls(pc);
        void* d_pc     = bam_io_page_cache_get_d_pc_ptr(pc);
        void* pc_base  = bam_io_page_cache_get_base_addr(pc);

        // Build descriptors
        SPDesc* h_descs = (SPDesc*)malloc(npages * sizeof(SPDesc));
        for (uint32_t i = 0; i < npages; i++) {
            uint64_t global_pg = start_page + i;
            uint32_t dev = global_pg % n_devices;
            uint64_t local_pg = global_pg / n_devices;
            h_descs[i].lba     = h_partition_lbas[dev] + local_pg * PAGE_NBLK;
            h_descs[i].nblocks = PAGE_NBLK;
            h_descs[i].dev     = dev;
            h_descs[i].slot    = i;
            h_descs[i].pg_idx  = i;
        }

        SPDesc* d_descs = nullptr;
        void** d_qp_ptrs = nullptr;
        uint16_t* d_cids = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d_descs, npages * sizeof(SPDesc)));
        SP_CUDA_CHECK(cudaMalloc(&d_qp_ptrs, npages * sizeof(void*)));
        SP_CUDA_CHECK(cudaMalloc(&d_cids, npages * sizeof(uint16_t)));
        SP_CUDA_CHECK(cudaMemcpy(d_descs, h_descs, npages * sizeof(SPDesc), cudaMemcpyHostToDevice));

        SP_CUDA_CHECK(cudaMemset(d_output_sp, 0, total_bytes));

        uint32_t submit_blocks = (npages < (uint32_t)sm_count) ? npages : (uint32_t)sm_count;
        uint32_t poll_blocks   = (npages < (uint32_t)sm_count) ? npages : (uint32_t)sm_count;

        auto t0 = std::chrono::steady_clock::now();

        // Submit
        fprintf(stderr, "  submit(%u)...", npages);
        submit_kernel<<<submit_blocks, 1, 0, stream>>>(
            d_ctrls, d_pc, d_descs, npages, d_qp_ptrs, d_cids);
        SP_CUDA_CHECK(cudaStreamSynchronize(stream));
        fprintf(stderr, " ok\n");

        // Poll+Copy
        fprintf(stderr, "  poll(%u)...", npages);
        poll_copy_kernel<<<poll_blocks, 128, 0, stream>>>(
            pc_base, d_descs, npages, PAGE_SIZE, d_qp_ptrs, d_cids, d_output_sp);
        SP_CUDA_CHECK(cudaStreamSynchronize(stream));
        fprintf(stderr, " ok\n");

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbs = (double)total_bytes / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        printf("SP: %.2f ms, %.2f GB/s\n", ms, gbs);

        SP_CUDA_CHECK(cudaMemcpy(h_output_sp, d_output_sp, total_bytes, cudaMemcpyDeviceToHost));
        uint32_t cksum_sp = checksum_pages(h_output_sp, npages, PAGE_SIZE);
        uint32_t cksum_fused = checksum_pages(h_output_fused, npages, PAGE_SIZE);
        printf("SP checksum: 0x%08x (fused: 0x%08x) %s\n",
               cksum_sp, cksum_fused,
               (cksum_sp == cksum_fused) ? "MATCH" : "MISMATCH!");

        if (cksum_sp != cksum_fused) {
            // Find first mismatched page
            for (uint32_t pg = 0; pg < npages; pg++) {
                if (memcmp(h_output_fused + (uint64_t)pg * PAGE_SIZE,
                           h_output_sp + (uint64_t)pg * PAGE_SIZE, PAGE_SIZE) != 0) {
                    printf("  First mismatch at page %u\n", pg);
                    break;
                }
            }
        }

        SP_CUDA_CHECK(cudaFree(d_descs));
        SP_CUDA_CHECK(cudaFree(d_qp_ptrs));
        SP_CUDA_CHECK(cudaFree(d_cids));
        free(h_descs);
        bam_io_page_cache_destroy(pc);
    }
    } // Test 2

    // ────────────────────────────────────────────────────────
    // Test 3: Submit/Poll multi-batch (reuse QPs across batches)
    // This is the scenario that triggers intermittent deadlock.
    // ────────────────────────────────────────────────────────
    if (should_run("3")) {
    printf("\n--- Test 3: Submit/Poll multi-batch (%u batches of %u pages) ---\n",
           num_batches, batch_size);
    {
        // Page cache: one slot per batch descriptor
        bam_io_page_cache_t pc = bam_io_page_cache_create(ctrl, PAGE_SIZE, batch_size);
        void* d_ctrls  = bam_io_page_cache_get_d_ctrls(pc);
        void* d_pc     = bam_io_page_cache_get_d_pc_ptr(pc);
        void* pc_base  = bam_io_page_cache_get_base_addr(pc);

        SPDesc* d_descs = nullptr;
        void** d_qp_ptrs = nullptr;
        uint16_t* d_cids = nullptr;
        SPDesc* h_descs = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d_descs, batch_size * sizeof(SPDesc)));
        SP_CUDA_CHECK(cudaMalloc(&d_qp_ptrs, batch_size * sizeof(void*)));
        SP_CUDA_CHECK(cudaMalloc(&d_cids, batch_size * sizeof(uint16_t)));
        SP_CUDA_CHECK(cudaMallocHost(&h_descs, batch_size * sizeof(SPDesc)));

        uint32_t submit_blocks = (batch_size < (uint32_t)sm_count) ? batch_size : (uint32_t)sm_count;
        uint32_t poll_blocks   = (batch_size < (uint32_t)sm_count) ? batch_size : (uint32_t)sm_count;

        // Allocate per-batch output (batch_size pages)
        uint64_t batch_bytes = (uint64_t)batch_size * PAGE_SIZE;
        char* d_batch_output = nullptr;
        char* h_batch_output = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d_batch_output, batch_bytes));
        SP_CUDA_CHECK(cudaMallocHost(&h_batch_output, batch_bytes));

        auto t0 = std::chrono::steady_clock::now();
        bool all_ok = true;

        for (uint32_t b = 0; b < num_batches; b++) {
            // Build descriptors: read batch_size pages starting from different offsets (cyclic)
            uint32_t pages_this_batch = batch_size;
            for (uint32_t i = 0; i < pages_this_batch; i++) {
                uint32_t pg_in_file = (b * batch_size + i) % npages;
                uint64_t global_pg = start_page + pg_in_file;
                uint32_t dev = global_pg % n_devices;
                uint64_t local_pg = global_pg / n_devices;
                h_descs[i].lba     = h_partition_lbas[dev] + local_pg * PAGE_NBLK;
                h_descs[i].nblocks = PAGE_NBLK;
                h_descs[i].dev     = dev;
                h_descs[i].slot    = i;
                h_descs[i].pg_idx  = i;
            }

            SP_CUDA_CHECK(cudaMemcpyAsync(d_descs, h_descs,
                          pages_this_batch * sizeof(SPDesc),
                          cudaMemcpyHostToDevice, stream));

            SP_CUDA_CHECK(cudaMemsetAsync(d_batch_output, 0, batch_bytes, stream));

            // Submit
            fprintf(stderr, "  batch %u/%u: submit(%u)...", b, num_batches, pages_this_batch);
            submit_kernel<<<submit_blocks, 1, 0, stream>>>(
                d_ctrls, d_pc, d_descs, pages_this_batch, d_qp_ptrs, d_cids);
            SP_CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, " ok");

            // Poll+Copy
            fprintf(stderr, " poll...");
            poll_copy_kernel<<<poll_blocks, 128, 0, stream>>>(
                pc_base, d_descs, pages_this_batch, PAGE_SIZE,
                d_qp_ptrs, d_cids, d_batch_output);
            SP_CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, " ok");

            // Verify against fused baseline
            SP_CUDA_CHECK(cudaMemcpy(h_batch_output, d_batch_output, batch_bytes, cudaMemcpyDeviceToHost));
            bool batch_ok = true;
            for (uint32_t i = 0; i < pages_this_batch; i++) {
                uint32_t pg_in_file = (b * batch_size + i) % npages;
                if (memcmp(h_batch_output + (uint64_t)i * PAGE_SIZE,
                           h_output_fused + (uint64_t)pg_in_file * PAGE_SIZE,
                           PAGE_SIZE) != 0) {
                    fprintf(stderr, " MISMATCH page[%u] (file page %u)", i, pg_in_file);
                    batch_ok = false;
                    all_ok = false;
                    break;
                }
            }
            if (batch_ok) fprintf(stderr, " verified");
            fprintf(stderr, "\n");
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("Multi-batch: %u batches in %.2f ms (%.2f ms/batch)\n",
               num_batches, ms, ms / num_batches);
        printf("Result: %s\n", all_ok ? "ALL PASSED" : "FAILURES DETECTED");

        SP_CUDA_CHECK(cudaFree(d_descs));
        SP_CUDA_CHECK(cudaFree(d_qp_ptrs));
        SP_CUDA_CHECK(cudaFree(d_cids));
        SP_CUDA_CHECK(cudaFree(d_batch_output));
        SP_CUDA_CHECK(cudaFreeHost(h_descs));
        SP_CUDA_CHECK(cudaFreeHost(h_batch_output));
        bam_io_page_cache_destroy(pc);
    }
    } // Test 3

    // ────────────────────────────────────────────────────────
    // Test 4: Multi-field SP (Q1 pattern)
    // Simulates Q1: 7 fields × batch_pages descs per batch.
    // Each field starts at a different page offset (stride = npages).
    // Total descs = 7 × batch_size per batch.
    // ────────────────────────────────────────────────────────
    if (should_run("4")) {
    {
        const uint32_t NFIELDS = 7;
        // Field start pages: field[i] starts at start_page + i*npages
        // This mimics Q1 where L_QUANTITY, L_EXTENDEDPRICE, etc. are at
        // different positions on disk (each with npages pages).
        uint64_t field_start_pages[NFIELDS];
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            field_start_pages[fi] = start_page + (uint64_t)fi * npages;

        uint32_t pages_per_field = batch_size;
        if (pages_per_field > npages) pages_per_field = npages;
        uint32_t total_descs = NFIELDS * pages_per_field;

        printf("\n--- Test 4: Multi-field SP (Q1 pattern) ---\n");
        printf("  %u fields x %u pages = %u descs/batch, %u batches\n",
               NFIELDS, pages_per_field, total_descs, num_batches);
        printf("  descs/QP = ~%u\n", total_descs / 128);

        // Page cache: one slot per descriptor
        bam_io_page_cache_t pc = bam_io_page_cache_create(ctrl, PAGE_SIZE, total_descs);
        void* d_ctrls  = bam_io_page_cache_get_d_ctrls(pc);
        void* d_pc     = bam_io_page_cache_get_d_pc_ptr(pc);
        void* pc_base  = bam_io_page_cache_get_base_addr(pc);

        SPDesc* d_descs = nullptr;
        void** d_qp_ptrs = nullptr;
        uint16_t* d_cids = nullptr;
        SPDesc* h_descs = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d_descs, total_descs * sizeof(SPDesc)));
        SP_CUDA_CHECK(cudaMalloc(&d_qp_ptrs, total_descs * sizeof(void*)));
        SP_CUDA_CHECK(cudaMalloc(&d_cids, total_descs * sizeof(uint16_t)));
        SP_CUDA_CHECK(cudaMallocHost(&h_descs, total_descs * sizeof(SPDesc)));

        uint32_t submit_blocks = (total_descs < (uint32_t)sm_count) ? total_descs : (uint32_t)sm_count;
        uint32_t poll_blocks   = (total_descs < (uint32_t)sm_count) ? total_descs : (uint32_t)sm_count;

        // Output: total_descs pages (we just verify data, not output layout)
        uint64_t out_bytes = (uint64_t)total_descs * PAGE_SIZE;
        char* d_mf_output = nullptr;
        char* h_mf_output = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d_mf_output, out_bytes));
        SP_CUDA_CHECK(cudaMallocHost(&h_mf_output, out_bytes));

        // Per-field fused baseline for verification
        printf("  Reading per-field fused baselines...\n");
        char* h_field_baselines[NFIELDS];
        for (uint32_t fi = 0; fi < NFIELDS; fi++) {
            uint64_t field_bytes = (uint64_t)npages * PAGE_SIZE;
            h_field_baselines[fi] = (char*)malloc(field_bytes);

            char* d_tmp = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d_tmp, field_bytes));
            SP_CUDA_CHECK(cudaMemset(d_tmp, 0, field_bytes));

            uint32_t nblk = (npages < (uint32_t)sm_count) ? npages : (uint32_t)sm_count;
            bam_io_page_cache_t pc_tmp = bam_io_page_cache_create(ctrl, PAGE_SIZE, nblk);
            fused_read_kernel<<<nblk, 128, 0, stream>>>(
                bam_io_page_cache_get_d_ctrls(pc_tmp),
                bam_io_page_cache_get_d_pc_ptr(pc_tmp),
                bam_io_page_cache_get_base_addr(pc_tmp),
                npages, field_start_pages[fi], d_partition_lbas, n_devices,
                d_tmp, PAGE_SIZE);
            SP_CUDA_CHECK(cudaStreamSynchronize(stream));
            SP_CUDA_CHECK(cudaMemcpy(h_field_baselines[fi], d_tmp, field_bytes, cudaMemcpyDeviceToHost));
            bam_io_page_cache_destroy(pc_tmp);
            SP_CUDA_CHECK(cudaFree(d_tmp));

            uint32_t cksum = checksum_pages(h_field_baselines[fi], npages, PAGE_SIZE);
            printf("    field[%u] start_page=%lu checksum=0x%08x\n",
                   fi, field_start_pages[fi], cksum);
        }

        auto t0 = std::chrono::steady_clock::now();
        bool all_ok = true;

        for (uint32_t b = 0; b < num_batches; b++) {
            // Build descriptors: for each field, pages_per_field pages
            // Layout: [field0_page0, field0_page1, ..., field1_page0, ...]
            // (matches Q1 build_descs pattern)
            uint32_t nd = 0;
            for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                for (uint32_t pg = 0; pg < pages_per_field; pg++) {
                    // Cycle through pages for multi-batch
                    uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                    uint64_t global_pg = field_start_pages[fi] + pg_in_file;
                    uint32_t dev = global_pg % n_devices;
                    uint64_t local_pg = global_pg / n_devices;
                    h_descs[nd].lba     = h_partition_lbas[dev] + local_pg * PAGE_NBLK;
                    h_descs[nd].nblocks = PAGE_NBLK;
                    h_descs[nd].dev     = dev;
                    h_descs[nd].slot    = nd;  // sequential slot assignment
                    h_descs[nd].pg_idx  = nd;  // output page index
                    nd++;
                }
            }

            SP_CUDA_CHECK(cudaMemcpyAsync(d_descs, h_descs,
                          nd * sizeof(SPDesc), cudaMemcpyHostToDevice, stream));
            SP_CUDA_CHECK(cudaMemsetAsync(d_mf_output, 0, out_bytes, stream));

            // Submit
            fprintf(stderr, "  batch %u/%u: submit(%u descs, %u/QP)...",
                    b, num_batches, nd, nd / 128);
            submit_kernel<<<submit_blocks, 1, 0, stream>>>(
                d_ctrls, d_pc, d_descs, nd, d_qp_ptrs, d_cids);
            SP_CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, " ok");

            // Poll+Copy
            fprintf(stderr, " poll...");
            poll_copy_kernel<<<poll_blocks, 128, 0, stream>>>(
                pc_base, d_descs, nd, PAGE_SIZE, d_qp_ptrs, d_cids, d_mf_output);
            SP_CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, " ok");

            // Verify each field's pages against fused baseline
            SP_CUDA_CHECK(cudaMemcpy(h_mf_output, d_mf_output, out_bytes, cudaMemcpyDeviceToHost));
            bool batch_ok = true;
            uint32_t desc_idx = 0;
            for (uint32_t fi = 0; fi < NFIELDS && batch_ok; fi++) {
                for (uint32_t pg = 0; pg < pages_per_field && batch_ok; pg++) {
                    uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                    if (memcmp(h_mf_output + (uint64_t)desc_idx * PAGE_SIZE,
                               h_field_baselines[fi] + (uint64_t)pg_in_file * PAGE_SIZE,
                               PAGE_SIZE) != 0) {
                        fprintf(stderr, " MISMATCH field[%u] page[%u] (file page %u, desc %u)",
                                fi, pg, pg_in_file, desc_idx);
                        batch_ok = false;
                        all_ok = false;
                    }
                    desc_idx++;
                }
            }
            if (batch_ok) fprintf(stderr, " verified");
            fprintf(stderr, "\n");
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbs = (double)out_bytes * num_batches / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        printf("Multi-field: %u batches of %u descs in %.2f ms (%.2f ms/batch, %.2f GB/s)\n",
               num_batches, total_descs, ms, ms / num_batches, gbs);
        printf("Result: %s\n", all_ok ? "ALL PASSED" : "FAILURES DETECTED");

        for (uint32_t fi = 0; fi < NFIELDS; fi++) free(h_field_baselines[fi]);
        SP_CUDA_CHECK(cudaFree(d_descs));
        SP_CUDA_CHECK(cudaFree(d_qp_ptrs));
        SP_CUDA_CHECK(cudaFree(d_cids));
        SP_CUDA_CHECK(cudaFree(d_mf_output));
        SP_CUDA_CHECK(cudaFreeHost(h_descs));
        SP_CUDA_CHECK(cudaFreeHost(h_mf_output));
        bam_io_page_cache_destroy(pc);
    }
    } // Test 4

    // ────────────────────────────────────────────────────────
    // Test 5: Per-field SP (Q1 optimized pattern)
    //
    // 5a: Sequential — 1 stream, field-by-field submit→poll
    // 5b: Parallel — 7 streams, all fields submit concurrently,
    //     then all fields poll concurrently
    // 5c: Fused per-field baseline — 7 fused reads sequentially
    //     (for fair throughput comparison)
    // ────────────────────────────────────────────────────────
    if (should_run("5")) {
    {
        const uint32_t NFIELDS = 7;
        uint64_t field_start_pages[NFIELDS];
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            field_start_pages[fi] = start_page + (uint64_t)fi * npages;

        uint32_t pages_per_field = batch_size;
        if (pages_per_field > npages) pages_per_field = npages;
        uint64_t per_field_bytes = (uint64_t)pages_per_field * PAGE_SIZE;
        uint64_t total_7f_bytes = (uint64_t)NFIELDS * per_field_bytes;

        printf("\n--- Test 5: Per-field SP ---\n");
        printf("  %u fields x %u pages, %u batches\n",
               NFIELDS, pages_per_field, num_batches);

        // Reuse field baselines from Test 4 scope if they exist.
        // Re-read them here (Test 4 freed them).
        char* h_field_baselines[NFIELDS];
        printf("  Reading per-field fused baselines...\n");
        for (uint32_t fi = 0; fi < NFIELDS; fi++) {
            uint64_t field_bytes = (uint64_t)npages * PAGE_SIZE;
            h_field_baselines[fi] = (char*)malloc(field_bytes);

            char* d_tmp = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d_tmp, field_bytes));
            SP_CUDA_CHECK(cudaMemset(d_tmp, 0, field_bytes));

            uint32_t nblk = (npages < (uint32_t)sm_count) ? npages : (uint32_t)sm_count;
            bam_io_page_cache_t pc_tmp = bam_io_page_cache_create(ctrl, PAGE_SIZE, nblk);
            fused_read_kernel<<<nblk, 128, 0, stream>>>(
                bam_io_page_cache_get_d_ctrls(pc_tmp),
                bam_io_page_cache_get_d_pc_ptr(pc_tmp),
                bam_io_page_cache_get_base_addr(pc_tmp),
                npages, field_start_pages[fi], d_partition_lbas, n_devices,
                d_tmp, PAGE_SIZE);
            SP_CUDA_CHECK(cudaStreamSynchronize(stream));
            SP_CUDA_CHECK(cudaMemcpy(h_field_baselines[fi], d_tmp, field_bytes, cudaMemcpyDeviceToHost));
            bam_io_page_cache_destroy(pc_tmp);
            SP_CUDA_CHECK(cudaFree(d_tmp));
        }

        // Shared per-field SP buffers (reused across 5a/5b)
        // One page cache with pages_per_field slots (sufficient for per-field SP)
        uint32_t sub_blocks = (pages_per_field < (uint32_t)sm_count) ? pages_per_field : (uint32_t)sm_count;

        // Per-field output (on GPU, 7 fields)
        char* d_field_out[NFIELDS];
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            SP_CUDA_CHECK(cudaMalloc(&d_field_out[fi], per_field_bytes));

        // ── 5c: Fused per-field (baseline throughput) ──
        printf("\n  [5c] Fused per-field (sequential, 1 stream):\n");
        {
            bam_io_page_cache_t pc5 = bam_io_page_cache_create(ctrl, PAGE_SIZE, sub_blocks);
            void* d5_ctrls = bam_io_page_cache_get_d_ctrls(pc5);
            void* d5_pc    = bam_io_page_cache_get_d_pc_ptr(pc5);
            void* d5_base  = bam_io_page_cache_get_base_addr(pc5);

            auto t0 = std::chrono::steady_clock::now();
            for (uint32_t b = 0; b < num_batches; b++) {
                for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                    uint32_t pg_off = (b * pages_per_field) % npages;
                    uint64_t fsp = field_start_pages[fi] + pg_off;
                    uint32_t ppf = pages_per_field;
                    if (pg_off + ppf > npages) ppf = npages - pg_off;

                    fused_read_kernel<<<sub_blocks, 128, 0, stream>>>(
                        d5_ctrls, d5_pc, d5_base,
                        ppf, fsp, d_partition_lbas, n_devices,
                        d_field_out[fi], PAGE_SIZE);
                }
                SP_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gbs = (double)total_7f_bytes * num_batches / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
            printf("    %u batches in %.2f ms (%.2f ms/batch, %.2f GB/s)\n",
                   num_batches, ms, ms / num_batches, gbs);
            bam_io_page_cache_destroy(pc5);
        }

        // ── 5a: Sequential per-field SP (with per-field progress) ──
        // Two variants:
        //   5a-1: Persistent page_cache (reused across all fields and batches)
        //   5a-2: Fresh page_cache per batch (recreate to reset QP state)
        // Run variant 2 (fresh page_cache) first, then variant 1 (persistent)
        int variant_order[] = {2, 1};
        for (int vi = 0; vi < 2; vi++) {
            int variant = variant_order[vi];
            printf("\n  [5a-%d] Sequential per-field SP (%s page_cache):\n",
                   variant, (variant == 1) ? "persistent" : "fresh-per-batch");

            SPDesc* d5_descs = nullptr;
            void** d5_qp = nullptr;
            uint16_t* d5_cids = nullptr;
            SPDesc* h5_descs = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d5_descs, pages_per_field * sizeof(SPDesc)));
            SP_CUDA_CHECK(cudaMalloc(&d5_qp, pages_per_field * sizeof(void*)));
            SP_CUDA_CHECK(cudaMalloc(&d5_cids, pages_per_field * sizeof(uint16_t)));
            SP_CUDA_CHECK(cudaMallocHost(&h5_descs, pages_per_field * sizeof(SPDesc)));

            // Persistent page_cache (variant 1 only, variant 2 creates per-batch)
            bam_io_page_cache_t pc5 = nullptr;
            void *d5_ctrls = nullptr, *d5_pc = nullptr, *d5_base = nullptr;
            if (variant == 1) {
                pc5 = bam_io_page_cache_create(ctrl, PAGE_SIZE, pages_per_field);
                d5_ctrls = bam_io_page_cache_get_d_ctrls(pc5);
                d5_pc    = bam_io_page_cache_get_d_pc_ptr(pc5);
                d5_base  = bam_io_page_cache_get_base_addr(pc5);
            }

            auto t0 = std::chrono::steady_clock::now();
            bool all_ok = true;

            for (uint32_t b = 0; b < num_batches; b++) {
                // Variant 2: fresh page_cache per batch
                if (variant == 2) {
                    if (pc5) bam_io_page_cache_destroy(pc5);
                    pc5 = bam_io_page_cache_create(ctrl, PAGE_SIZE, pages_per_field);
                    d5_ctrls = bam_io_page_cache_get_d_ctrls(pc5);
                    d5_pc    = bam_io_page_cache_get_d_pc_ptr(pc5);
                    d5_base  = bam_io_page_cache_get_base_addr(pc5);
                }

                fprintf(stderr, "    batch %u/%u:", b, num_batches);
                for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                    uint32_t nd = 0;
                    for (uint32_t pg = 0; pg < pages_per_field; pg++) {
                        uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                        uint64_t global_pg = field_start_pages[fi] + pg_in_file;
                        uint32_t dev = global_pg % n_devices;
                        uint64_t local_pg = global_pg / n_devices;
                        h5_descs[nd].lba     = h_partition_lbas[dev] + local_pg * PAGE_NBLK;
                        h5_descs[nd].nblocks = PAGE_NBLK;
                        h5_descs[nd].dev     = dev;
                        h5_descs[nd].slot    = nd;
                        h5_descs[nd].pg_idx  = nd;
                        nd++;
                    }

                    SP_CUDA_CHECK(cudaMemcpyAsync(d5_descs, h5_descs,
                                  nd * sizeof(SPDesc), cudaMemcpyHostToDevice, stream));

                    fprintf(stderr, " f%u:S", fi);
                    submit_kernel<<<sub_blocks, 1, 0, stream>>>(
                        d5_ctrls, d5_pc, d5_descs, nd, d5_qp, d5_cids);
                    SP_CUDA_CHECK(cudaStreamSynchronize(stream));

                    fprintf(stderr, "P");
                    poll_copy_kernel<<<sub_blocks, 128, 0, stream>>>(
                        d5_base, d5_descs, nd, PAGE_SIZE, d5_qp, d5_cids, d_field_out[fi]);
                    SP_CUDA_CHECK(cudaStreamSynchronize(stream));
                    fprintf(stderr, "ok");
                }

                // Verify (spot-check first+last page of each field)
                char* h_verify_buf = (char*)malloc(PAGE_SIZE);
                bool batch_ok = true;
                for (uint32_t fi = 0; fi < NFIELDS && batch_ok; fi++) {
                    for (uint32_t which = 0; which < 2 && batch_ok; which++) {
                        uint32_t pg = (which == 0) ? 0 : pages_per_field - 1;
                        uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                        SP_CUDA_CHECK(cudaMemcpy(h_verify_buf,
                                      d_field_out[fi] + (uint64_t)pg * PAGE_SIZE,
                                      PAGE_SIZE, cudaMemcpyDeviceToHost));
                        if (memcmp(h_verify_buf,
                                   h_field_baselines[fi] + (uint64_t)pg_in_file * PAGE_SIZE,
                                   PAGE_SIZE) != 0) {
                            fprintf(stderr, " MISMATCH f%u p%u", fi, pg);
                            batch_ok = false;
                            all_ok = false;
                        }
                    }
                }
                free(h_verify_buf);
                if (batch_ok) fprintf(stderr, " verified");
                fprintf(stderr, "\n");
            }

            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gbs = (double)total_7f_bytes * num_batches / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
            printf("    %u batches in %.2f ms (%.2f ms/batch, %.2f GB/s)\n",
                   num_batches, ms, ms / num_batches, gbs);
            printf("    Result: %s\n", all_ok ? "ALL PASSED" : "FAILURES DETECTED");

            SP_CUDA_CHECK(cudaFree(d5_descs));
            SP_CUDA_CHECK(cudaFree(d5_qp));
            SP_CUDA_CHECK(cudaFree(d5_cids));
            SP_CUDA_CHECK(cudaFreeHost(h5_descs));
            if (pc5) bam_io_page_cache_destroy(pc5);
        }

        // ── 5b: Parallel per-field SP (7 streams) ──
        printf("\n  [5b] Parallel per-field SP (%u streams):\n", NFIELDS);
        {
            // Each stream gets its own page_cache region.
            // Create one large page_cache: NFIELDS * pages_per_field slots.
            // Field fi uses slots [fi*pages_per_field, (fi+1)*pages_per_field).
            uint32_t total_slots = NFIELDS * pages_per_field;
            bam_io_page_cache_t pc5 = bam_io_page_cache_create(ctrl, PAGE_SIZE, total_slots);
            void* d5_ctrls = bam_io_page_cache_get_d_ctrls(pc5);
            void* d5_pc    = bam_io_page_cache_get_d_pc_ptr(pc5);
            void* d5_base  = bam_io_page_cache_get_base_addr(pc5);

            // Per-field resources
            cudaStream_t field_streams[NFIELDS];
            SPDesc* d5_descs[NFIELDS];
            void**  d5_qp[NFIELDS];
            uint16_t* d5_cids[NFIELDS];
            SPDesc* h5_descs[NFIELDS];

            for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                SP_CUDA_CHECK(cudaStreamCreate(&field_streams[fi]));
                SP_CUDA_CHECK(cudaMalloc(&d5_descs[fi], pages_per_field * sizeof(SPDesc)));
                SP_CUDA_CHECK(cudaMalloc(&d5_qp[fi], pages_per_field * sizeof(void*)));
                SP_CUDA_CHECK(cudaMalloc(&d5_cids[fi], pages_per_field * sizeof(uint16_t)));
                SP_CUDA_CHECK(cudaMallocHost(&h5_descs[fi], pages_per_field * sizeof(SPDesc)));
            }

            auto t0 = std::chrono::steady_clock::now();
            bool all_ok = true;

            for (uint32_t b = 0; b < num_batches; b++) {
                fprintf(stderr, "    batch %u/%u:", b, num_batches);

                // Build and submit all 7 fields in parallel
                for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                    uint32_t slot_base = fi * pages_per_field;
                    uint32_t nd = 0;
                    for (uint32_t pg = 0; pg < pages_per_field; pg++) {
                        uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                        uint64_t global_pg = field_start_pages[fi] + pg_in_file;
                        uint32_t dev = global_pg % n_devices;
                        uint64_t local_pg = global_pg / n_devices;
                        h5_descs[fi][nd].lba     = h_partition_lbas[dev] + local_pg * PAGE_NBLK;
                        h5_descs[fi][nd].nblocks = PAGE_NBLK;
                        h5_descs[fi][nd].dev     = dev;
                        h5_descs[fi][nd].slot    = slot_base + nd;
                        h5_descs[fi][nd].pg_idx  = nd;  // local output index
                        nd++;
                    }

                    SP_CUDA_CHECK(cudaMemcpyAsync(d5_descs[fi], h5_descs[fi],
                                  nd * sizeof(SPDesc), cudaMemcpyHostToDevice, field_streams[fi]));

                    submit_kernel<<<sub_blocks, 1, 0, field_streams[fi]>>>(
                        d5_ctrls, d5_pc, d5_descs[fi], nd, d5_qp[fi], d5_cids[fi]);
                }

                // Wait for all submits to complete
                for (uint32_t fi = 0; fi < NFIELDS; fi++)
                    SP_CUDA_CHECK(cudaStreamSynchronize(field_streams[fi]));
                fprintf(stderr, " submit");

                // Poll+copy all 7 fields in parallel
                for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                    poll_copy_kernel<<<sub_blocks, 128, 0, field_streams[fi]>>>(
                        d5_base, d5_descs[fi], pages_per_field, PAGE_SIZE,
                        d5_qp[fi], d5_cids[fi], d_field_out[fi]);
                }
                for (uint32_t fi = 0; fi < NFIELDS; fi++)
                    SP_CUDA_CHECK(cudaStreamSynchronize(field_streams[fi]));
                fprintf(stderr, " poll");

                // Spot-check verify
                char* h_vb5b = (char*)malloc(PAGE_SIZE);
                bool batch_ok = true;
                for (uint32_t fi = 0; fi < NFIELDS && batch_ok; fi++) {
                    for (uint32_t which = 0; which < 2 && batch_ok; which++) {
                        uint32_t pg = (which == 0) ? 0 : pages_per_field - 1;
                        uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                        SP_CUDA_CHECK(cudaMemcpy(h_vb5b,
                                      d_field_out[fi] + (uint64_t)pg * PAGE_SIZE,
                                      PAGE_SIZE, cudaMemcpyDeviceToHost));
                        if (memcmp(h_vb5b,
                                   h_field_baselines[fi] + (uint64_t)pg_in_file * PAGE_SIZE,
                                   PAGE_SIZE) != 0) {
                            fprintf(stderr, " MISMATCH f%u p%u", fi, pg);
                            batch_ok = false;
                            all_ok = false;
                        }
                    }
                }
                free(h_vb5b);
                if (batch_ok) fprintf(stderr, " verified");
                fprintf(stderr, "\n");
            }

            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gbs = (double)total_7f_bytes * num_batches / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
            printf("    %u batches in %.2f ms (%.2f ms/batch, %.2f GB/s)\n",
                   num_batches, ms, ms / num_batches, gbs);
            printf("    Result: %s\n", all_ok ? "ALL PASSED" : "FAILURES DETECTED");

            for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                SP_CUDA_CHECK(cudaStreamDestroy(field_streams[fi]));
                SP_CUDA_CHECK(cudaFree(d5_descs[fi]));
                SP_CUDA_CHECK(cudaFree(d5_qp[fi]));
                SP_CUDA_CHECK(cudaFree(d5_cids[fi]));
                SP_CUDA_CHECK(cudaFreeHost(h5_descs[fi]));
            }
            bam_io_page_cache_destroy(pc5);
        }

        for (uint32_t fi = 0; fi < NFIELDS; fi++) free(h_field_baselines[fi]);
        for (uint32_t fi = 0; fi < NFIELDS; fi++) SP_CUDA_CHECK(cudaFree(d_field_out[fi]));
    }
    } // Test 5

    // ────────────────────────────────────────────────────────
    // Test 6: Pure multi-field SP (no fused commands at all)
    // Opens fresh controller, no baselines, no verification.
    // Only checks for deadlock / completion.
    // ────────────────────────────────────────────────────────
    if (should_run("6")) {
        const uint32_t NFIELDS = test6_nfields;
        if (NFIELDS < 1 || NFIELDS > 16) {
            fprintf(stderr, "ERROR: --nfields must be 1..16\n");
            return 1;
        }
        uint64_t field_start_pages[16];
        uint64_t spacing = test6_same_lba ? 0
                         : (test6_field_spacing >= 0 ? (uint64_t)test6_field_spacing : npages);
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            field_start_pages[fi] = start_page + (uint64_t)fi * spacing;

        uint32_t pages_per_field = batch_size;
        if (pages_per_field > npages) pages_per_field = npages;
        uint64_t per_field_bytes = (uint64_t)pages_per_field * PAGE_SIZE;

        printf("\n--- Test 6: Pure multi-field SP (no fused, no verification) ---\n");
        printf("  %u fields x %u pages, %u batches, spacing=%lu pages\n",
               NFIELDS, pages_per_field, num_batches, spacing);
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            printf("    f%u: start_page=%lu\n", fi, field_start_pages[fi]);

        uint32_t sub_blocks = (pages_per_field < (uint32_t)sm_count) ? pages_per_field : (uint32_t)sm_count;

        // Per-field output buffers
        char* d_field_out6[16] = {};
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            SP_CUDA_CHECK(cudaMalloc(&d_field_out6[fi], per_field_bytes));

        // SP buffers
        SPDesc* d6_descs = nullptr;
        void** d6_qp = nullptr;
        uint16_t* d6_cids = nullptr;
        SPDesc* h6_descs = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d6_descs, pages_per_field * sizeof(SPDesc)));
        SP_CUDA_CHECK(cudaMalloc(&d6_qp, pages_per_field * sizeof(void*)));
        SP_CUDA_CHECK(cudaMalloc(&d6_cids, pages_per_field * sizeof(uint16_t)));
        SP_CUDA_CHECK(cudaMallocHost(&h6_descs, pages_per_field * sizeof(SPDesc)));

        // Fresh page_cache per batch
        auto t0 = std::chrono::steady_clock::now();

        for (uint32_t b = 0; b < num_batches; b++) {
            bam_io_page_cache_t pc6 = bam_io_page_cache_create(ctrl, PAGE_SIZE, pages_per_field);
            void* d6c = bam_io_page_cache_get_d_ctrls(pc6);
            void* d6p = bam_io_page_cache_get_d_pc_ptr(pc6);
            void* d6b = bam_io_page_cache_get_base_addr(pc6);

            fprintf(stderr, "  batch %u/%u:", b, num_batches);
            for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                uint32_t nd = 0;
                for (uint32_t pg = 0; pg < pages_per_field; pg++) {
                    uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                    uint64_t global_pg = field_start_pages[fi] + pg_in_file;
                    uint32_t dev = global_pg % n_devices;
                    uint64_t local_pg = global_pg / n_devices;
                    h6_descs[nd].lba     = h_partition_lbas[dev] + local_pg * PAGE_NBLK;
                    h6_descs[nd].nblocks = PAGE_NBLK;
                    h6_descs[nd].dev     = dev;
                    h6_descs[nd].slot    = nd;
                    h6_descs[nd].pg_idx  = nd;
                    nd++;
                }

                SP_CUDA_CHECK(cudaMemcpyAsync(d6_descs, h6_descs,
                              nd * sizeof(SPDesc), cudaMemcpyHostToDevice, stream));

                fprintf(stderr, " f%u:S", fi);
                submit_kernel<<<sub_blocks, 1, 0, stream>>>(
                    d6c, d6p, d6_descs, nd, d6_qp, d6_cids);
                SP_CUDA_CHECK(cudaStreamSynchronize(stream));

                fprintf(stderr, "P");
                poll_copy_kernel<<<sub_blocks, 128, 0, stream>>>(
                    d6b, d6_descs, nd, PAGE_SIZE, d6_qp, d6_cids, d_field_out6[fi]);
                SP_CUDA_CHECK(cudaStreamSynchronize(stream));
                fprintf(stderr, "ok");
            }
            fprintf(stderr, " done\n");
            bam_io_page_cache_destroy(pc6);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        uint64_t total_6_bytes = (uint64_t)NFIELDS * per_field_bytes * num_batches;
        double gbs = (double)total_6_bytes / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        printf("  %u batches in %.2f ms (%.2f ms/batch, %.2f GB/s)\n",
               num_batches, ms, ms / num_batches, gbs);
        printf("  ALL COMPLETED (no deadlock)\n");

        SP_CUDA_CHECK(cudaFree(d6_descs));
        SP_CUDA_CHECK(cudaFree(d6_qp));
        SP_CUDA_CHECK(cudaFree(d6_cids));
        SP_CUDA_CHECK(cudaFreeHost(h6_descs));
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            SP_CUDA_CHECK(cudaFree(d_field_out6[fi]));
    } // Test 6

    // ────────────────────────────────────────────────────────
    // Test 7: FIFO-based async Submit/Poll
    // Submit kernel pushes (qp, cid) into global FIFO.
    // Poll kernel consumes in FIFO order (≈ submission order).
    // Both run concurrently on separate streams.
    // ────────────────────────────────────────────────────────
    if (should_run("7")) {
        const uint32_t NFIELDS = test6_nfields;
        if (NFIELDS < 1 || NFIELDS > 16) {
            fprintf(stderr, "ERROR: --nfields must be 1..16\n");
            return 1;
        }
        uint64_t field_start_pages[16];
        uint64_t spacing = test6_same_lba ? 0
                         : (test6_field_spacing >= 0 ? (uint64_t)test6_field_spacing : npages);
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            field_start_pages[fi] = start_page + (uint64_t)fi * spacing;

        // Total descs = NFIELDS * batch_size
        uint32_t pages_per_field = batch_size;
        if (pages_per_field > npages) pages_per_field = npages;
        uint32_t total_descs = NFIELDS * pages_per_field;
        uint64_t total_data = (uint64_t)total_descs * PAGE_SIZE;

        printf("\n--- Test 7: FIFO-based async Submit/Poll ---\n");
        printf("  %u fields x %u pages = %u descs, spacing=%lu, %u batches\n",
               NFIELDS, pages_per_field, total_descs, spacing, num_batches);

        // FIFO capacity = total_descs (no wrap needed for single batch)
        FifoEntry* d_fifo = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d_fifo, total_descs * sizeof(FifoEntry)));

        // Atomic write_pos and read_pos
        uint32_t* d_write_pos = nullptr;  // producer increments
        uint32_t* d_read_pos  = nullptr;  // consumer increments
        SP_CUDA_CHECK(cudaMalloc(&d_write_pos, sizeof(uint32_t)));
        SP_CUDA_CHECK(cudaMalloc(&d_read_pos,  sizeof(uint32_t)));

        // Output buffer (all fields concatenated)
        char* d_out7 = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d_out7, total_data));

        // Build host descs (all fields interleaved: f0p0, f0p1, ..., f1p0, f1p1, ...)
        SPDesc* h7_descs = nullptr;
        SPDesc* d7_descs = nullptr;
        SP_CUDA_CHECK(cudaMallocHost(&h7_descs, total_descs * sizeof(SPDesc)));
        SP_CUDA_CHECK(cudaMalloc(&d7_descs, total_descs * sizeof(SPDesc)));

        uint32_t sub_blocks = (total_descs < (uint32_t)sm_count) ? total_descs : (uint32_t)sm_count;
        uint32_t poll_blocks = sub_blocks;

        // Streams for concurrent submit + poll
        cudaStream_t stream_submit, stream_poll;
        SP_CUDA_CHECK(cudaStreamCreate(&stream_submit));
        SP_CUDA_CHECK(cudaStreamCreate(&stream_poll));

        auto t0 = std::chrono::steady_clock::now();

        for (uint32_t b = 0; b < num_batches; b++) {
            // Build descs for this batch
            uint32_t nd = 0;
            for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                for (uint32_t pg = 0; pg < pages_per_field; pg++) {
                    uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                    uint64_t global_pg = field_start_pages[fi] + pg_in_file;
                    uint32_t dev = global_pg % n_devices;
                    uint64_t local_pg = global_pg / n_devices;
                    h7_descs[nd].lba     = h_partition_lbas[dev] + local_pg * PAGE_NBLK;
                    h7_descs[nd].nblocks = PAGE_NBLK;
                    h7_descs[nd].dev     = dev;
                    h7_descs[nd].slot    = nd;  // unique slot per desc
                    h7_descs[nd].pg_idx  = nd;
                    nd++;
                }
            }

            // Create fresh page_cache for this batch
            bam_io_page_cache_t pc7 = bam_io_page_cache_create(ctrl, PAGE_SIZE, total_descs);
            void* d7c = bam_io_page_cache_get_d_ctrls(pc7);
            void* d7p = bam_io_page_cache_get_d_pc_ptr(pc7);
            void* d7b = bam_io_page_cache_get_base_addr(pc7);

            // Reset FIFO positions
            SP_CUDA_CHECK(cudaMemset(d_write_pos, 0, sizeof(uint32_t)));
            SP_CUDA_CHECK(cudaMemset(d_read_pos,  0, sizeof(uint32_t)));

            // Upload descs
            SP_CUDA_CHECK(cudaMemcpy(d7_descs, h7_descs,
                                     nd * sizeof(SPDesc), cudaMemcpyHostToDevice));

            fprintf(stderr, "  batch %u/%u: submit+poll(%u descs)...", b, num_batches, nd);

            // Launch submit kernel (producer) — pushes to FIFO
            // Uses a lambda-style kernel defined below via the fifo_submit_kernel
            fifo_submit_kernel<<<sub_blocks, 1, 0, stream_submit>>>(
                d7c, d7p, d7_descs, nd, d_fifo, d_write_pos, d_out7, d7b, PAGE_SIZE);

            // Launch poll+copy kernel (consumer) — pops from FIFO
            fifo_poll_copy_kernel<<<poll_blocks, 128, 0, stream_poll>>>(
                d7b, d_fifo, d_write_pos, d_read_pos, nd, PAGE_SIZE, d_out7);

            SP_CUDA_CHECK(cudaStreamSynchronize(stream_submit));
            SP_CUDA_CHECK(cudaStreamSynchronize(stream_poll));
            fprintf(stderr, " done\n");

            bam_io_page_cache_destroy(pc7);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbs = (double)total_data * num_batches / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        printf("  %u batches in %.2f ms (%.2f ms/batch, %.2f GB/s)\n",
               num_batches, ms, ms / num_batches, gbs);
        printf("  ALL COMPLETED (no deadlock)\n");

        SP_CUDA_CHECK(cudaStreamDestroy(stream_submit));
        SP_CUDA_CHECK(cudaStreamDestroy(stream_poll));
        SP_CUDA_CHECK(cudaFree(d_fifo));
        SP_CUDA_CHECK(cudaFree(d_write_pos));
        SP_CUDA_CHECK(cudaFree(d_read_pos));
        SP_CUDA_CHECK(cudaFree(d_out7));
        SP_CUDA_CHECK(cudaFree(d7_descs));
        SP_CUDA_CHECK(cudaFreeHost(h7_descs));
    } // Test 7

    // ────────────────────────────────────────────────────────
    // Test 8: Per-QP CQ-order consumer (deadlock-free SP)
    // Submit kernel builds CID→page reverse map.
    // Poll kernel has one block per QP, consumes CQ entries
    // in CQ head order (not CID-specific) and uses reverse
    // map to find which page was completed.
    // ────────────────────────────────────────────────────────
    if (should_run("8")) {
        const uint32_t NFIELDS = test6_nfields;
        if (NFIELDS < 1 || NFIELDS > 16) {
            fprintf(stderr, "ERROR: --nfields must be 1..16\n");
            return 1;
        }
        uint64_t field_start_pages[16];
        uint64_t spacing = test6_same_lba ? 0
                         : (test6_field_spacing >= 0 ? (uint64_t)test6_field_spacing : npages);
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            field_start_pages[fi] = start_page + (uint64_t)fi * spacing;

        uint32_t pages_per_field = batch_size;
        if (pages_per_field > npages) pages_per_field = npages;
        uint32_t total_descs = NFIELDS * pages_per_field;
        uint64_t total_data = (uint64_t)total_descs * PAGE_SIZE;

        printf("\n--- Test 8: Per-QP CQ-order consumer (deadlock-free SP) ---\n");
        printf("  %u fields x %u pages = %u descs, spacing=%lu, %u batches\n",
               NFIELDS, pages_per_field, total_descs, spacing, num_batches);

        // Build host descs
        SPDesc* h8_descs = nullptr;
        SPDesc* d8_descs = nullptr;
        SP_CUDA_CHECK(cudaMallocHost(&h8_descs, total_descs * sizeof(SPDesc)));
        SP_CUDA_CHECK(cudaMalloc(&d8_descs, total_descs * sizeof(SPDesc)));

        // Output buffer
        char* d_out8 = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d_out8, total_data));

        uint32_t sub_blocks = (total_descs < (uint32_t)sm_count) ? total_descs : (uint32_t)sm_count;

        auto t0 = std::chrono::steady_clock::now();

        for (uint32_t b = 0; b < num_batches; b++) {
            // Build descs
            uint32_t nd = 0;
            for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                for (uint32_t pg = 0; pg < pages_per_field; pg++) {
                    uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                    uint64_t global_pg = field_start_pages[fi] + pg_in_file;
                    uint32_t dev = global_pg % n_devices;
                    uint64_t local_pg = global_pg / n_devices;
                    h8_descs[nd].lba     = h_partition_lbas[dev] + local_pg * PAGE_NBLK;
                    h8_descs[nd].nblocks = PAGE_NBLK;
                    h8_descs[nd].dev     = dev;
                    h8_descs[nd].slot    = nd;
                    h8_descs[nd].pg_idx  = nd;
                    nd++;
                }
            }

            // Create fresh page_cache
            bam_io_page_cache_t pc8 = bam_io_page_cache_create(ctrl, PAGE_SIZE, total_descs);
            void* d8c = bam_io_page_cache_get_d_ctrls(pc8);
            void* d8p = bam_io_page_cache_get_d_pc_ptr(pc8);
            void* d8b = bam_io_page_cache_get_base_addr(pc8);

            // Get QP topology
            uint32_t n_qps = bam_io_page_cache_get_n_qps(pc8, 0);
            uint32_t total_qps = n_devices * n_qps;
            if (b == 0) {
                printf("  n_qps=%u per device, total_qps=%u, descs/QP≈%u\n",
                       n_qps, total_qps, nd / total_qps);
            }

            // CID reverse map: [total_qps][MAX_CID]
            // CID space is 16-bit but bounded by CQ depth; use 1024 as safe max
            const uint32_t MAX_CID = 1024;
            CidPageInfo* d_cid_map = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d_cid_map,
                          (uint64_t)total_qps * MAX_CID * sizeof(CidPageInfo)));

            // Per-QP submit counters
            uint32_t* d_qp_nsubmit = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d_qp_nsubmit, total_qps * sizeof(uint32_t)));
            SP_CUDA_CHECK(cudaMemset(d_qp_nsubmit, 0, total_qps * sizeof(uint32_t)));

            // QP pointer array (for poll kernel)
            void** h_qp_ptrs = (void**)malloc(total_qps * sizeof(void*));
            for (uint32_t d = 0; d < n_devices; d++)
                for (uint32_t qi = 0; qi < n_qps; qi++)
                    h_qp_ptrs[d * n_qps + qi] = bam_io_page_cache_get_qp(pc8, d, qi);
            void** d_qp_ptrs = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d_qp_ptrs, total_qps * sizeof(void*)));
            SP_CUDA_CHECK(cudaMemcpy(d_qp_ptrs, h_qp_ptrs,
                          total_qps * sizeof(void*), cudaMemcpyHostToDevice));

            // Upload descs
            SP_CUDA_CHECK(cudaMemcpy(d8_descs, h8_descs,
                                     nd * sizeof(SPDesc), cudaMemcpyHostToDevice));

            fprintf(stderr, "  batch %u/%u: submit(%u descs)...", b, num_batches, nd);

            // Submit + build CID reverse map
            cqorder_submit_kernel<<<sub_blocks, 1, 0, stream>>>(
                d8c, d8p, d8_descs, nd,
                n_qps, n_devices,
                d_cid_map, MAX_CID, d_qp_nsubmit);
            SP_CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, " ok");

            // Per-QP CQ-order poll+copy
            fprintf(stderr, " poll(%u QPs)...", total_qps);
            cqorder_poll_copy_kernel<<<total_qps, 128, 0, stream>>>(
                d_qp_ptrs, total_qps,
                d_cid_map, MAX_CID, d_qp_nsubmit,
                d8b, PAGE_SIZE, d_out8);
            SP_CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, " ok\n");

            free(h_qp_ptrs);
            SP_CUDA_CHECK(cudaFree(d_cid_map));
            SP_CUDA_CHECK(cudaFree(d_qp_nsubmit));
            SP_CUDA_CHECK(cudaFree(d_qp_ptrs));
            bam_io_page_cache_destroy(pc8);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbs = (double)total_data * num_batches / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        printf("  %u batches in %.2f ms (%.2f ms/batch, %.2f GB/s)\n",
               num_batches, ms, ms / num_batches, gbs);
        printf("  ALL COMPLETED (no deadlock)\n");

        SP_CUDA_CHECK(cudaFree(d_out8));
        SP_CUDA_CHECK(cudaFree(d8_descs));
        SP_CUDA_CHECK(cudaFreeHost(h8_descs));
    } // Test 8

    // ────────────────────────────────────────────────────────
    // Test 9: Concurrent submit+poll (2 streams)
    // Same CQ-order consumer as Test 8, but submit and poll
    // kernels run concurrently on separate streams.
    // Per-QP submit counts are precomputed on host.
    // ────────────────────────────────────────────────────────
    if (should_run("9")) {
        const uint32_t NFIELDS = test6_nfields;
        if (NFIELDS < 1 || NFIELDS > 16) {
            fprintf(stderr, "ERROR: --nfields must be 1..16\n");
            return 1;
        }
        uint64_t field_start_pages[16];
        uint64_t spacing = test6_same_lba ? 0
                         : (test6_field_spacing >= 0 ? (uint64_t)test6_field_spacing : npages);
        for (uint32_t fi = 0; fi < NFIELDS; fi++)
            field_start_pages[fi] = start_page + (uint64_t)fi * spacing;

        uint32_t pages_per_field = batch_size;
        if (pages_per_field > npages) pages_per_field = npages;
        uint32_t total_descs = NFIELDS * pages_per_field;
        uint64_t total_data = (uint64_t)total_descs * PAGE_SIZE;

        printf("\n--- Test 9: Concurrent submit+poll (2 streams, CQ-order) ---\n");
        printf("  %u fields x %u pages = %u descs, spacing=%lu, %u batches\n",
               NFIELDS, pages_per_field, total_descs, spacing, num_batches);

        // Build host descs
        SPDesc* h9_descs = nullptr;
        SPDesc* d9_descs = nullptr;
        SP_CUDA_CHECK(cudaMallocHost(&h9_descs, total_descs * sizeof(SPDesc)));
        SP_CUDA_CHECK(cudaMalloc(&d9_descs, total_descs * sizeof(SPDesc)));

        // Output buffer
        char* d_out9 = nullptr;
        SP_CUDA_CHECK(cudaMalloc(&d_out9, total_data));

        uint32_t sub_blocks = (total_descs < (uint32_t)sm_count) ? total_descs : (uint32_t)sm_count;

        // Streams for concurrent submit + poll
        cudaStream_t stream_submit9, stream_poll9;
        SP_CUDA_CHECK(cudaStreamCreate(&stream_submit9));
        SP_CUDA_CHECK(cudaStreamCreate(&stream_poll9));

        auto t0 = std::chrono::steady_clock::now();

        for (uint32_t b = 0; b < num_batches; b++) {
            // Build descs
            uint32_t nd = 0;
            for (uint32_t fi = 0; fi < NFIELDS; fi++) {
                for (uint32_t pg = 0; pg < pages_per_field; pg++) {
                    uint32_t pg_in_file = (b * pages_per_field + pg) % npages;
                    uint64_t global_pg = field_start_pages[fi] + pg_in_file;
                    uint32_t dev = global_pg % n_devices;
                    uint64_t local_pg = global_pg / n_devices;
                    h9_descs[nd].lba     = h_partition_lbas[dev] + local_pg * PAGE_NBLK;
                    h9_descs[nd].nblocks = PAGE_NBLK;
                    h9_descs[nd].dev     = dev;
                    h9_descs[nd].slot    = nd;
                    h9_descs[nd].pg_idx  = nd;
                    nd++;
                }
            }

            // Create fresh page_cache
            bam_io_page_cache_t pc9 = bam_io_page_cache_create(ctrl, PAGE_SIZE, total_descs);
            void* d9c = bam_io_page_cache_get_d_ctrls(pc9);
            void* d9p = bam_io_page_cache_get_d_pc_ptr(pc9);
            void* d9b = bam_io_page_cache_get_base_addr(pc9);

            // Get QP topology
            uint32_t n_qps = bam_io_page_cache_get_n_qps(pc9, 0);
            uint32_t total_qps = n_devices * n_qps;
            if (b == 0) {
                printf("  n_qps=%u per device, total_qps=%u, descs/QP≈%u\n",
                       n_qps, total_qps, nd / total_qps);
            }

            const uint32_t MAX_CID = 1024;
            CidPageInfo* d_cid_map = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d_cid_map,
                          (uint64_t)total_qps * MAX_CID * sizeof(CidPageInfo)));

            // Precompute per-QP submit counts on host
            uint32_t* h_qp_nsubmit = (uint32_t*)calloc(total_qps, sizeof(uint32_t));
            for (uint32_t j = 0; j < nd; j++) {
                uint32_t dev = h9_descs[j].dev;
                uint32_t qp_idx = h9_descs[j].slot % n_qps;
                uint32_t flat_qp = dev * n_qps + qp_idx;
                h_qp_nsubmit[flat_qp]++;
            }

            uint32_t* d_qp_nsubmit = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d_qp_nsubmit, total_qps * sizeof(uint32_t)));
            SP_CUDA_CHECK(cudaMemcpy(d_qp_nsubmit, h_qp_nsubmit,
                          total_qps * sizeof(uint32_t), cudaMemcpyHostToDevice));

            // QP pointer array
            void** h_qp_ptrs = (void**)malloc(total_qps * sizeof(void*));
            for (uint32_t d = 0; d < n_devices; d++)
                for (uint32_t qi = 0; qi < n_qps; qi++)
                    h_qp_ptrs[d * n_qps + qi] = bam_io_page_cache_get_qp(pc9, d, qi);
            void** d_qp_ptrs = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d_qp_ptrs, total_qps * sizeof(void*)));
            SP_CUDA_CHECK(cudaMemcpy(d_qp_ptrs, h_qp_ptrs,
                          total_qps * sizeof(void*), cudaMemcpyHostToDevice));

            // Upload descs
            SP_CUDA_CHECK(cudaMemcpy(d9_descs, h9_descs,
                                     nd * sizeof(SPDesc), cudaMemcpyHostToDevice));

            fprintf(stderr, "  batch %u/%u: submit+poll(%u descs)...", b, num_batches, nd);

            // Launch submit (stream_submit9) — the submit kernel uses d_qp_nsubmit
            // but it's already set from host, so we pass a dummy counter (unused by poll).
            // For concurrent mode, submit kernel doesn't need to update d_qp_nsubmit;
            // the poll kernel uses the host-precomputed values.
            // We reuse cqorder_submit_kernel but d_qp_nsubmit is pre-filled (atomic adds
            // will overshoot but poll only reads the precomputed values).
            // -> Need a separate counter for submit's atomicAdd to avoid corruption.
            uint32_t* d_qp_nsubmit_dummy = nullptr;
            SP_CUDA_CHECK(cudaMalloc(&d_qp_nsubmit_dummy, total_qps * sizeof(uint32_t)));
            SP_CUDA_CHECK(cudaMemset(d_qp_nsubmit_dummy, 0, total_qps * sizeof(uint32_t)));

            cqorder_submit_kernel<<<sub_blocks, 1, 0, stream_submit9>>>(
                d9c, d9p, d9_descs, nd,
                n_qps, n_devices,
                d_cid_map, MAX_CID, d_qp_nsubmit_dummy);

            // Launch poll (stream_poll9) — concurrent with submit
            cqorder_poll_copy_kernel<<<total_qps, 128, 0, stream_poll9>>>(
                d_qp_ptrs, total_qps,
                d_cid_map, MAX_CID, d_qp_nsubmit,
                d9b, PAGE_SIZE, d_out9);

            SP_CUDA_CHECK(cudaStreamSynchronize(stream_submit9));
            SP_CUDA_CHECK(cudaStreamSynchronize(stream_poll9));
            fprintf(stderr, " ok\n");

            free(h_qp_ptrs);
            free(h_qp_nsubmit);
            SP_CUDA_CHECK(cudaFree(d_cid_map));
            SP_CUDA_CHECK(cudaFree(d_qp_nsubmit));
            SP_CUDA_CHECK(cudaFree(d_qp_nsubmit_dummy));
            SP_CUDA_CHECK(cudaFree(d_qp_ptrs));
            bam_io_page_cache_destroy(pc9);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbs = (double)total_data * num_batches / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        printf("  %u batches in %.2f ms (%.2f ms/batch, %.2f GB/s)\n",
               num_batches, ms, ms / num_batches, gbs);
        printf("  ALL COMPLETED (no deadlock)\n");

        SP_CUDA_CHECK(cudaStreamDestroy(stream_submit9));
        SP_CUDA_CHECK(cudaStreamDestroy(stream_poll9));
        SP_CUDA_CHECK(cudaFree(d_out9));
        SP_CUDA_CHECK(cudaFree(d9_descs));
        SP_CUDA_CHECK(cudaFreeHost(h9_descs));
    } // Test 9

    // Cleanup
    SP_CUDA_CHECK(cudaStreamDestroy(stream));
    SP_CUDA_CHECK(cudaFree(d_output_fused));
    SP_CUDA_CHECK(cudaFree(d_output_sp));
    SP_CUDA_CHECK(cudaFree(d_partition_lbas));
    SP_CUDA_CHECK(cudaFreeHost(h_output_fused));
    SP_CUDA_CHECK(cudaFreeHost(h_output_sp));
    bam_ctrl_close(ctrl);

    printf("\nDone.\n");
    return 0;
}
