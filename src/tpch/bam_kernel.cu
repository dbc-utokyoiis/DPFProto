// ============================================================
// BAM Q6 kernel + host wrapper
//
// Compiled as a separate CUDA C++11 target (libbam_q6) to avoid
// freestanding-libcxx vs C++20 header conflicts.
//
// Uses per-field prefix sum arrays to handle greedy-packed pages
// where different columns may have different nalloc per page.
//
// Warp-level kernel (32 threads per block = 1 warp) with
// per-lane I/O via access_data_async/poll_async.
// ============================================================

#include "bam_kernel.cuh"
#include "../kernel/tpch/q1.cuh"

#include <ctrl.h>
#include <page_cache.h>
#include <nvm_parallel_queue.h>
#include <nvm_cmd.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

// PFOR decompression kernel (tile_idx parameterized variant)
#include "binpack_kernel.cuh"

#define BAM_CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                   \
                cudaGetErrorString(err), __FILE__, __LINE__);          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

// Page cache entries per block:
//   0: L_SHIPDATE page
//   1: L_QUANTITY page A (containing row_begin)
//   2: L_QUANTITY page B (if boundary crossing)
//   3: L_EXTENDEDPRICE page A
//   4: L_EXTENDEDPRICE page B
//   5: L_DISCOUNT page A
//   6: L_DISCOUNT page B
#define ENTRIES_PER_BLOCK 7

// Max decompressed int32 elements per page slot.
// page_size = 65536, header = 12 bytes → max (65536-12)/4 = 16381 values.
// Rounded up to safe bound.
#define DECOMP_ELEMS_PER_SLOT 16384

// Maximum I/O multiplicity (ring buffer depth) for pipelined kernel.
#define MAX_IO_MULTI 8

// NVMe controller page size in 512-byte blocks (4 KiB = 8 blocks).
// BAM's page cache (for page_size > 8 KiB) stores a PRP *list* address
// in PRP2.  The NVMe controller interprets PRP2 differently based on
// the transfer size:
//   - ≤ 1 NVMe page  (≤ 8 blk):  PRP2 ignored           → safe
//   - 2 NVMe pages   (9-16 blk):  PRP2 = direct address  → UNSAFE
//   - ≥ 3 NVMe pages (≥ 17 blk):  PRP2 = PRP list        → safe
// For exactly-2-page transfers the controller treats the PRP list
// address as a data destination, corrupting both the cache entry
// and the PRP list.  We avoid this zone by bumping to 17 blocks.
#define NVM_CTRL_PAGE_BLOCKS 8

__device__ __forceinline__ uint32_t safe_io_nblocks(uint32_t comp_bytes) {
    uint32_t nblk = (comp_bytes + 511) / 512;
    if (nblk > NVM_CTRL_PAGE_BLOCKS && nblk <= NVM_CTRL_PAGE_BLOCKS * 2)
        nblk = NVM_CTRL_PAGE_BLOCKS * 2 + 1;   // 17 blocks = 8.5 KiB
    return nblk;
}

// Per-slot metadata saved in shared memory for ring-buffered I/O pipeline.
struct SlotMeta {
    uint64_t sd_pg;            // driving column page index
    uint64_t row_begin;
    uint64_t row_end;
    uint64_t fi_pg[3];         // non-driving field page indices (QT, EP, DC)
    uint64_t fi_bound[3];      // boundary row for each non-driving field
    int      needs_boundary[3]; // whether boundary page was loaded
    int      n_ios;            // 0 = empty/skipped, 4-7 = active
    uint32_t io_bytes;         // total bytes submitted for this slot
};

// Minimal pag_head for C++11 BAM kernel (mirrors src/common/pag.h)
struct bam_pag_head {
    uint32_t nalloc;     // actual valid element count
    uint32_t watermark;  // padded element count (multiple of 128) for compressed pages
    uint32_t lfreespace;
};
#define BAM_PAG_HDR_BYTES 12

// ============================================================
// Opaque Controller handle (shared across page cache instances)
// ============================================================
struct BAMCtrlHandle {
    Controller* ctrl;
    std::vector<Controller*> ctrls;
    int cuda_device;
};

bam_ctrl_handle_t bam_ctrl_open(const char* path, uint32_t ns_id,
                                 int cuda_device, uint32_t queue_depth,
                                 uint32_t num_queues) {
    BAM_CUDA_CHECK(cudaSetDevice(cuda_device));
    auto* h = new BAMCtrlHandle;
    h->ctrls.resize(1);
    h->ctrls[0] = new Controller(path, ns_id, cuda_device,
                                  queue_depth, num_queues);
    h->ctrl = h->ctrls[0];
    h->cuda_device = cuda_device;
    return h;
}

bam_ctrl_handle_t bam_ctrl_open_multi(const char** paths, uint32_t n_devices,
                                       uint32_t ns_id, int cuda_device,
                                       uint32_t queue_depth, uint32_t num_queues) {
    BAM_CUDA_CHECK(cudaSetDevice(cuda_device));
    auto* h = new BAMCtrlHandle;
    h->ctrls.resize(n_devices);
    for (uint32_t i = 0; i < n_devices; i++) {
        h->ctrls[i] = new Controller(paths[i], ns_id, cuda_device,
                                      queue_depth, num_queues);
    }
    h->ctrl = h->ctrls[0];
    h->cuda_device = cuda_device;
    return h;
}

void bam_ctrl_close(bam_ctrl_handle_t handle) {
    auto* h = static_cast<BAMCtrlHandle*>(handle);
    for (auto* c : h->ctrls) delete c;
    delete h;
}

// ============================================================
// Simple single-page read kernel (1 warp reads 1 page)
// ============================================================
__global__ void bam_read_page_kernel(
    Controller** ctrls,
    page_cache_d_t* pc,
    uint64_t lba,
    uint32_t n_blocks,
    uint32_t dev_idx)
{
    unsigned long long pc_entry = 0;
    QueuePair* qp = ctrls[dev_idx]->d_qps;
    read_data(pc, qp, lba, n_blocks, pc_entry);
}

// ============================================================
// bam_read_page — read one page via BAM, copy to host buffer.
// Uses an existing controller; creates a temporary page cache.
// dev_idx selects which controller to use (0 = default).
// ============================================================
int bam_read_page(bam_ctrl_handle_t handle, uint64_t page_size,
                  uint64_t lba, void* out_buf, uint32_t dev_idx) {
    auto* h = static_cast<BAMCtrlHandle*>(handle);

    const uint64_t n_pc_pages = 1;
    const uint64_t max_range = 1;
    page_cache_t pc(page_size, n_pc_pages, h->cuda_device,
                    *h->ctrls[dev_idx], max_range, h->ctrls);

    uint32_t n_blocks = (uint32_t)(page_size / 512);

    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    // 1 warp = 32 threads
    bam_read_page_kernel<<<1, 32, 0, stream>>>(pc.pdt.d_ctrls, pc.d_pc_ptr,
                                                lba, n_blocks, dev_idx);
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy page data from GPU page cache to host
    BAM_CUDA_CHECK(cudaMemcpyAsync(out_buf, pc.pdt.base_addr, page_size,
                                    cudaMemcpyDeviceToHost, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    BAM_CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}

// ============================================================
// GPU metadata (device-side)
// ============================================================
struct BAMKernelMeta {
    uint64_t field_start_page_ids[4];
    uint64_t field_npages;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t partition_start_lba;           // device 0 (backward compat)
    uint64_t partition_start_lbas[MAX_BAM_DEVICES]; // per-device partition LBA
    uint32_t n_devices;                     // 1 = single device, >1 = RAID0
    uint32_t num_blocks;
    uint64_t nrows;
    // GPU pointers to prefix sum arrays (field_npages+1 elements each)
    const uint64_t* d_prefix_sums[4];
    // Zone map: per-page (min, max) pairs for L_SHIPDATE. nullptr = no pruning.
    const int32_t* d_shipdate_stats;
    uint64_t nstats;
    // Compression support
    uint16_t compression_method[4];
    const uint32_t* d_comp_sizes[4];    // per-page compressed byte sizes
    const uint64_t* d_comp_offsets[4];  // per-page disk offsets (bytes)
    int32_t* d_decomp_buf;              // decompression output buffer
    uint32_t decomp_elems_per_slot;     // max decompressed elements per slot
    uint32_t io_multi;                  // ring buffer depth (1 = sync, >1 = pipeline)
    int32_t sd_low;                     // Q6 L_SHIPDATE lower bound (inclusive)
    int32_t sd_high;                    // Q6 L_SHIPDATE upper bound (exclusive)
    // Coalesced I/O support (used only by bam_q6_kernel_coalesced_sync)
    uint32_t coalesce_k;               // pages per coalesced read (1 = no coalescing)
    uint32_t original_page_size;       // uncoalesced page_size (page_size = original * k)
    // Revenue query: L_QUANTITY < revenue_qt_max (0 = no filter)
    int32_t revenue_qt_max;
};

// ============================================================
// Diagnostic counters (checksum verification)
// ============================================================
struct BAMDiagCounters {
    uint64_t sum_shipdate;
    uint64_t sum_quantity;
    uint64_t sum_extprice;
    uint64_t sum_discount;
};

// ============================================================
// Profiling counters (clock64 cycles, accumulated via atomicAdd)
// ============================================================
struct BAMPerfCounters {
    uint64_t io_cycles;       // total clock64 cycles in read_data
    uint64_t decomp_cycles;   // total clock64 cycles in decompress_page
    uint64_t eval_cycles;     // total clock64 cycles in Q6 predicate eval
    uint64_t page_count;      // total pages processed
    uint64_t io_count;        // total read_data calls
    uint64_t io_bytes;        // total bytes read from NVMe
    uint64_t boundary_count;  // total boundary page reads (non-driving field spans 2 pages)
    uint64_t pf_submit_count; // prefetch IOs submitted
    uint64_t pf_hit_count;    // prefetch data successfully reused
};

// ============================================================
// Per-slot metadata for work queue kernel.
// Written by IO blocks, read by compute blocks.
// ============================================================
struct WQSlotMeta {
    uint64_t row_begin;
    uint64_t row_end;
    int all_comp;
    int has_boundary;
    uint64_t fi_pg[3];
    uint64_t fi_bound[3];
    int needs_boundary[3];
};

#define WQ_RING_SIZE 128

// ============================================================
// Pre-computed I/O descriptor for prescan kernel.
// Stores all metadata needed to issue NVMe reads for one
// qualifying SD page, eliminating runtime zone-map checks,
// binary searches, and LBA computations from the execution kernel.
// ============================================================
struct QualPageIO {
    uint64_t lba[7];          // primary 4 + boundary up to 3
    uint32_t nblocks[7];      // NVMe block count per I/O
    uint32_t n_ios;           // valid I/O count (4-7)
    uint32_t nalloc;          // SD page nalloc (rows in this page)
    uint8_t  needs_boundary;  // bit flags: bit0=QT, bit1=EP, bit2=DC
    uint8_t  all_comp;        // 1 if all 4 fields are compressed
    uint8_t  pad[2];
};

// ============================================================
// Binary search: find page p such that ps[p] <= row < ps[p+1]
// ============================================================
__device__ uint64_t find_page_for_row(const uint64_t* ps, uint64_t npages, uint64_t row) {
    uint64_t lo = 0, hi = npages;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (ps[mid + 1] <= row) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ============================================================
// Decompress one page: 32-thread warp version using decodeElement.
// Returns nalloc (actual valid element count) via pointer.
// ============================================================
__device__ void decompress_page_warp(
    char* page_ptr,
    int32_t* decomp_out,
    uint16_t comp_method,
    uint* shared_bws,    // __shared__ uint[4]
    uint* shared_offs,   // __shared__ uint[4]
    uint32_t tid,
    uint32_t* out_nalloc)
{
    volatile bam_pag_head* hdr = (volatile bam_pag_head*)page_ptr;
    uint32_t nalloc = hdr->nalloc;
    uint32_t watermark = hdr->watermark;

    if (out_nalloc && tid == 0) *out_nalloc = nalloc;

    if (comp_method != 0) {
        // PFOR compressed: watermark = padded element count (multiple of 128)
        uint32_t nblocks = watermark / 128;
        uint32_t* block_start = (uint32_t*)(page_ptr + BAM_PAG_HDR_BYTES);
        uint32_t* data_ptr = block_start + (nblocks + 1);

        // Process all PFOR blocks with 32 threads × 4 iterations = 128 elements
        for (uint32_t b = 0; b < nblocks; b++) {
            uint32_t* blk_data = data_ptr + block_start[b];

            // threads 0-3: extract bitwidths/offsets
            if (tid < 4) {
                uint32_t mb_bw_packed = blk_data[1];
                uint32_t packed_off = (mb_bw_packed << 8)
                                    + (mb_bw_packed << 16)
                                    + (mb_bw_packed << 24);
                shared_bws[tid]  = (mb_bw_packed >> (tid << 3)) & 255;
                shared_offs[tid] = (packed_off >> (tid << 3)) & 255;
            }
            __syncwarp();

            // 32 threads × 4 iterations = 128 elements
            for (uint32_t k = 0; k < 4; k++) {
                uint32_t i = k * 32 + tid;
                uint32_t mb_idx = i >> 5;
                uint32_t mb_pos = i & 31;
                int val = decodeElement(i, mb_idx, mb_pos,
                                        blk_data, shared_bws, shared_offs);
                uint32_t idx = b * 128 + i;
                if (idx < watermark) decomp_out[idx] = val;
            }
            __syncwarp();
        }
    } else {
        // Uncompressed: stride 32 copy
        int32_t* src = (int32_t*)(page_ptr + BAM_PAG_HDR_BYTES);
        for (uint32_t i = tid; i < nalloc; i += 32) {
            decomp_out[i] = src[i];
        }
        __syncwarp();
    }
}

// ============================================================
// Decode one PFOR block via shared memory (32-thread warp).
//
// Loads compressed block data from global memory (page cache)
// into s_comp_blk (shared memory), extracts bitwidths/offsets,
// and decodes 128 elements into register array vals[4].
//
// This avoids per-element global memory reads during decodeElement
// by operating on shared memory instead.
// ============================================================
__device__ __forceinline__ void decode_pfor_block_smem(
    uint32_t* blk_data_global,  // global memory: data_ptr + block_start[b]
    uint32_t  blk_size,         // block_start[b+1] - block_start[b] (in uint32)
    uint32_t* s_comp_blk,       // __shared__ uint32_t[136]
    uint*     shared_bws,       // __shared__ uint[4]
    uint*     shared_offs,      // __shared__ uint[4]
    uint32_t  tid,
    int32_t   (&vals)[4])       // output: 4 decoded values per thread (128 total)
{
    // 1. Cooperative load: global → shared memory (~4-5 iterations for 32 threads)
    for (uint32_t i = tid; i < blk_size; i += 32)
        s_comp_blk[i] = blk_data_global[i];
    __syncwarp();

    // 2. Extract miniblock bitwidths and offsets (threads 0-3)
    if (tid < 4) {
        uint32_t mb_bw_packed = s_comp_blk[1];
        uint32_t packed_off = (mb_bw_packed << 8)
                            + (mb_bw_packed << 16)
                            + (mb_bw_packed << 24);
        shared_bws[tid]  = (mb_bw_packed >> (tid << 3)) & 255;
        shared_offs[tid] = (packed_off >> (tid << 3)) & 255;
    }
    __syncwarp();

    // 3. Decode 128 elements (32 threads × 4 iterations) from shared memory
    for (uint32_t k = 0; k < 4; k++) {
        uint32_t i = k * 32 + tid;
        uint32_t mb_idx = i >> 5;
        uint32_t mb_pos = i & 31;
        vals[k] = decodeElement(i, mb_idx, mb_pos,
                                s_comp_blk, shared_bws, shared_offs);
    }
    __syncwarp();
}

// ============================================================
// 128-thread decompress (kept for test kernels only)
// ============================================================
__device__ void decompress_page_128(
    char* page_ptr,
    int32_t* decomp_out,
    uint16_t comp_method,
    uint* shared_buf,
    uint32_t tid,
    uint32_t* out_nalloc)
{
    volatile bam_pag_head* hdr = (volatile bam_pag_head*)page_ptr;
    uint32_t nalloc = hdr->nalloc;
    uint32_t watermark = hdr->watermark;

    if (out_nalloc && tid == 0) *out_nalloc = nalloc;

    if (comp_method != 0) {
        uint32_t nblocks = watermark / 128;
        uint32_t ntiles = nblocks / 4;
        uint32_t remaining_blocks = nblocks - ntiles * 4;
        uint32_t* block_start = (uint32_t*)(page_ptr + BAM_PAG_HDR_BYTES);
        uint32_t* data_ptr = block_start + (nblocks + 1);

        for (uint32_t t = 0; t < ntiles; t++) {
            int items[4];
            LoadBinPackTile<128, 4>(t, block_start, data_ptr,
                                    shared_buf, items, false, 512);
            for (int it = 0; it < 4; it++) {
                uint32_t idx = t * 512 + it * 128 + tid;
                if (idx < nalloc) {
                    decomp_out[idx] = items[it];
                }
            }
            __syncthreads();
        }

        for (uint32_t b = 0; b < remaining_blocks; b++) {
            uint32_t block_idx = ntiles * 4 + b;
            uint32_t* blk_data = data_ptr + block_start[block_idx];

            uint* rem_bws  = &shared_buf[0];
            uint* rem_offs = &shared_buf[4];
            if (tid < 4) {
                uint32_t mb_bw_packed = blk_data[1];
                uint32_t packed_off = (mb_bw_packed << 8)
                                    + (mb_bw_packed << 16)
                                    + (mb_bw_packed << 24);
                rem_bws[tid]  = (mb_bw_packed >> (tid << 3)) & 255;
                rem_offs[tid] = (packed_off >> (tid << 3)) & 255;
            }
            __syncthreads();

            uint32_t mb_idx = tid >> 5;
            uint32_t mb_pos = tid & 31;
            int val = decodeElement(tid, mb_idx, mb_pos,
                                    blk_data, rem_bws, rem_offs);

            uint32_t idx = block_idx * 128 + tid;
            if (idx < nalloc) {
                decomp_out[idx] = val;
            }
            __syncthreads();
        }
    } else {
        volatile int32_t* src = (volatile int32_t*)(page_ptr + BAM_PAG_HDR_BYTES);
        for (uint32_t i = tid; i < nalloc; i += 128) {
            decomp_out[i] = src[i];
        }
        __syncthreads();
    }
}

// ============================================================
// Prescan kernel: pre-compute qualifying page list + I/O descriptors.
// 1 thread = 1 SD page. Eliminates zone-map checks, binary searches,
// and LBA computations from the execution kernel.
// ============================================================
__global__ void bam_q6_prescan_kernel(
    BAMKernelMeta* meta,
    QualPageIO*    d_qual_pages,
    uint32_t*      d_num_qual)
{
    uint32_t sd_pg = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t npages = meta->field_npages;
    if (sd_pg >= npages) return;

    // 1. Zone map pruning
    const int32_t* stats = meta->d_shipdate_stats;
    if (stats && sd_pg < meta->nstats) {
        int32_t page_min = stats[sd_pg * 2];
        int32_t page_max = stats[sd_pg * 2 + 1];
        if (page_max < meta->sd_low || page_min >= meta->sd_high) {
            return;
        }
    }

    // 2. Row range from prefix sums
    const uint64_t* ps0 = meta->d_prefix_sums[0];
    uint64_t row_begin = ps0[sd_pg];
    uint64_t row_end   = ps0[sd_pg + 1];
    if (row_end <= row_begin) return;
    uint32_t nalloc = (uint32_t)(row_end - row_begin);

    // 3. Non-driving field page IDs via binary search
    const uint64_t* ps_arr[3] = {
        meta->d_prefix_sums[1], meta->d_prefix_sums[2], meta->d_prefix_sums[3]
    };
    uint64_t fi_pg[3];
    uint64_t fi_bound[3];
    bool needs_boundary[3];
    for (int f = 0; f < 3; f++) {
        fi_pg[f] = find_page_for_row(ps_arr[f], npages, row_begin);
        fi_bound[f] = ps_arr[f][fi_pg[f] + 1];
        needs_boundary[f] = (fi_bound[f] < row_end && fi_pg[f] + 1 < npages);
    }

    // 4. Build I/O descriptors
    const uint16_t* comp = meta->compression_method;
    const uint32_t blocks_per_page = meta->blocks_per_page;

    QualPageIO qp;
    int n_ios = 0;

    // SD primary page
    if (comp[0] != 0) {
        qp.lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[0][sd_pg] / 512;
        qp.nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[0][sd_pg]);
    } else {
        uint64_t pg_id = meta->field_start_page_ids[0] + sd_pg;
        qp.lba[n_ios] = meta->partition_start_lba + pg_id * blocks_per_page;
        qp.nblocks[n_ios] = blocks_per_page;
    }
    n_ios++;

    // QT, EP, DC primary pages
    for (int f = 0; f < 3; f++) {
        if (comp[f + 1] != 0) {
            qp.lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[f + 1][fi_pg[f]] / 512;
            qp.nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f]]);
        } else {
            uint64_t pg_id = meta->field_start_page_ids[f + 1] + fi_pg[f];
            qp.lba[n_ios] = meta->partition_start_lba + pg_id * blocks_per_page;
            qp.nblocks[n_ios] = blocks_per_page;
        }
        n_ios++;
    }

    // Boundary pages
    uint8_t boundary_bits = 0;
    for (int f = 0; f < 3; f++) {
        if (needs_boundary[f]) {
            boundary_bits |= (1 << f);
            if (comp[f + 1] != 0) {
                qp.lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[f + 1][fi_pg[f] + 1] / 512;
                qp.nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f] + 1]);
            } else {
                uint64_t pg_id = meta->field_start_page_ids[f + 1] + fi_pg[f] + 1;
                qp.lba[n_ios] = meta->partition_start_lba + pg_id * blocks_per_page;
                qp.nblocks[n_ios] = blocks_per_page;
            }
            n_ios++;
        }
    }

    qp.n_ios = n_ios;
    qp.nalloc = nalloc;
    qp.needs_boundary = boundary_bits;
    qp.all_comp = (comp[0] != 0 && comp[1] != 0 && comp[2] != 0 && comp[3] != 0) ? 1 : 0;

    // 5. Atomic append
    uint32_t idx = atomicAdd(d_num_qual, 1);
    d_qual_pages[idx] = qp;
}

// ============================================================
// Prescan execution kernel: reads pre-computed QualPageIO descriptors.
// No zone-map checks, no binary search, no LBA computation.
// Same fused decomp+eval as bam_q6_kernel_comp_sync.
// ============================================================
__global__ void bam_q6_kernel_prescan_sync(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    const QualPageIO* d_qual_pages,
    uint32_t num_qual,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    QueuePair* qp = ctrls[0]->d_qps + (block_id % ctrls[0]->n_qps);

    const uint64_t page_size = meta->page_size;
    const uint16_t* comp = meta->compression_method;
    char* base = (char*)pc->base_addr;

    __shared__ uint shared_bws[4];
    __shared__ uint shared_offs[4];
    __shared__ uint32_t s_comp_blk[136];

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;
    int32_t* decomp_base = meta->d_decomp_buf
        + (uint64_t)block_id * ENTRIES_PER_BLOCK * elems_per_slot;

    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    for (uint32_t qi = block_id; qi < num_qual; qi += gridDim.x) {
        const QualPageIO& qio = d_qual_pages[qi];

        blk_page_count++;

        // === I/O submit: use pre-computed LBAs ===
        // Entry layout: same as sync kernel (block_id * 7 + offset)
        unsigned long long io_entry[7];
        io_entry[0] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 0;  // SD
        io_entry[1] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 1;  // QT_A
        io_entry[2] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 3;  // EP_A
        io_entry[3] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 5;  // DC_A
        // Boundary entries (slots 2, 4, 6) — packed after primary
        int be = 4;
        if (qio.needs_boundary & 1) io_entry[be++] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 2;
        if (qio.needs_boundary & 2) io_entry[be++] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 4;
        if (qio.needs_boundary & 4) io_entry[be++] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 6;

        long long t0 = clock64();

        uint16_t my_cid = 0;
        uint16_t my_sq_pos = 0;
        if ((int)tid < (int)qio.n_ios) {
            access_data_async(pc, qp, qio.lba[tid], qio.nblocks[tid],
                              io_entry[tid], NVM_IO_READ, &my_cid, &my_sq_pos);
        }
        __syncwarp();

        if ((int)tid < (int)qio.n_ios) {
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, my_cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, my_cid);
        }
        __syncwarp();

        long long t1 = clock64();
        blk_io_cycles += (uint64_t)(t1 - t0);
        blk_io_count += qio.n_ios;
        for (uint32_t ii = 0; ii < qio.n_ios; ii++) blk_io_bytes += (uint64_t)qio.nblocks[ii] * 512;
        blk_boundary_count += __popc(qio.needs_boundary);

        // === Fused decompress + Q6 eval ===
        long long td0 = clock64();

        bool has_boundary = (qio.needs_boundary != 0);

        int64_t local_rev = 0;
        uint64_t local_sum_sd = 0, local_sum_qt = 0, local_sum_ep = 0, local_sum_dc = 0;

        if (qio.all_comp && !has_boundary) {
            // Case A: all compressed, no boundary — fused shared-memory path
            char* sub_page_ptr[4] = {
                base + io_entry[0] * page_size,
                base + io_entry[1] * page_size,
                base + io_entry[2] * page_size,
                base + io_entry[3] * page_size,
            };

            bam_pag_head* hdr0 = (bam_pag_head*)sub_page_ptr[0];
            uint32_t nalloc = hdr0->nalloc;
            uint32_t nblocks_pfor = hdr0->watermark / 128;

            uint32_t* blk_start[4];
            uint32_t* data_ptr_f[4];
            for (int f = 0; f < 4; f++) {
                blk_start[f] = (uint32_t*)(sub_page_ptr[f] + BAM_PAG_HDR_BYTES);
                data_ptr_f[f] = blk_start[f] + (nblocks_pfor + 1);
            }

            for (uint32_t b = 0; b < nblocks_pfor; b++) {
                int32_t sd_v[4], qt_v[4], ep_v[4], dc_v[4];

                decode_pfor_block_smem(
                    data_ptr_f[0] + blk_start[0][b],
                    blk_start[0][b + 1] - blk_start[0][b],
                    s_comp_blk, shared_bws, shared_offs, tid, sd_v);
                decode_pfor_block_smem(
                    data_ptr_f[1] + blk_start[1][b],
                    blk_start[1][b + 1] - blk_start[1][b],
                    s_comp_blk, shared_bws, shared_offs, tid, qt_v);
                decode_pfor_block_smem(
                    data_ptr_f[2] + blk_start[2][b],
                    blk_start[2][b + 1] - blk_start[2][b],
                    s_comp_blk, shared_bws, shared_offs, tid, ep_v);
                decode_pfor_block_smem(
                    data_ptr_f[3] + blk_start[3][b],
                    blk_start[3][b + 1] - blk_start[3][b],
                    s_comp_blk, shared_bws, shared_offs, tid, dc_v);

                for (uint32_t ki = 0; ki < 4; ki++) {
                    uint32_t idx = b * 128 + ki * 32 + tid;
                    if (idx < nalloc) {
                        local_sum_sd += (uint32_t)sd_v[ki];
                        local_sum_qt += (uint32_t)qt_v[ki];
                        local_sum_ep += (uint32_t)ep_v[ki];
                        local_sum_dc += (uint32_t)dc_v[ki];

                        if (sd_v[ki] >= meta->sd_low && sd_v[ki] < meta->sd_high &&
                            dc_v[ki] >= 5 && dc_v[ki] <= 7 &&
                            qt_v[ki] < 2400) {
                            local_rev += (int64_t)ep_v[ki] * dc_v[ki];
                        }
                    }
                }
            }
        } else {
            // Case B: fallback — decompress to global memory + eval
            int32_t* sd_decomp = decomp_base + 0 * elems_per_slot;
            decompress_page_warp(base + io_entry[0] * page_size, sd_decomp,
                                 comp[0], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + io_entry[1] * page_size,
                                 decomp_base + 1 * elems_per_slot,
                                 comp[1], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + io_entry[2] * page_size,
                                 decomp_base + 3 * elems_per_slot,
                                 comp[2], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + io_entry[3] * page_size,
                                 decomp_base + 5 * elems_per_slot,
                                 comp[3], shared_bws, shared_offs, tid, nullptr);

            // Boundary pages (if any)
            int bi = 4;
            if (qio.needs_boundary & 1) {
                decompress_page_warp(base + io_entry[bi++] * page_size,
                                     decomp_base + 2 * elems_per_slot,
                                     comp[1], shared_bws, shared_offs, tid, nullptr);
            }
            if (qio.needs_boundary & 2) {
                decompress_page_warp(base + io_entry[bi++] * page_size,
                                     decomp_base + 4 * elems_per_slot,
                                     comp[2], shared_bws, shared_offs, tid, nullptr);
            }
            if (qio.needs_boundary & 4) {
                decompress_page_warp(base + io_entry[bi++] * page_size,
                                     decomp_base + 6 * elems_per_slot,
                                     comp[3], shared_bws, shared_offs, tid, nullptr);
            }

            // Eval from global memory
            for (uint32_t i = tid; i < qio.nalloc; i += 32) {
                int32_t l_shipdate = sd_decomp[i];
                int32_t l_quantity = (decomp_base + 1 * elems_per_slot)[i];
                int32_t l_extendedprice = (decomp_base + 3 * elems_per_slot)[i];
                int32_t l_discount = (decomp_base + 5 * elems_per_slot)[i];

                local_sum_sd += (uint32_t)l_shipdate;
                local_sum_qt += (uint32_t)l_quantity;
                local_sum_ep += (uint32_t)l_extendedprice;
                local_sum_dc += (uint32_t)l_discount;

                if (l_shipdate >= meta->sd_low && l_shipdate < meta->sd_high &&
                    l_discount >= 5 && l_discount <= 7 &&
                    l_quantity < 2400) {
                    local_rev += (int64_t)l_extendedprice * l_discount;
                }
            }
        }

        long long td1 = clock64();
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        // Warp reduction + atomicAdd
        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
            local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
            local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
            local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
            local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
        }

        if (tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(local_rev));
            if (d_diag) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                          (unsigned long long)local_sum_sd);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                          (unsigned long long)local_sum_qt);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                          (unsigned long long)local_sum_ep);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                          (unsigned long long)local_sum_dc);
            }
        }
    }

    // Write perf counters
    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
    }
}

// ============================================================
// BAM Q6 kernel — warp-level with per-lane I/O
//
// 1 CUDA block = 1 warp (32 threads).
// Each lane independently issues I/O via access_data_async/poll_async.
// All 32 threads participate in decompression and Q6 eval.
// ============================================================
__global__ void bam_q6_kernel_comp_sync(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;            // 0..31

    const uint32_t ndev = meta->n_devices;

    const uint64_t npages = meta->field_npages;
    const uint32_t blocks_per_page = meta->blocks_per_page;
    const uint64_t page_size = meta->page_size;

    // Prefix sum GPU pointers
    const uint64_t* ps0 = meta->d_prefix_sums[0]; // L_SHIPDATE
    const uint64_t* ps1 = meta->d_prefix_sums[1]; // L_QUANTITY
    const uint64_t* ps2 = meta->d_prefix_sums[2]; // L_EXTENDEDPRICE
    const uint64_t* ps3 = meta->d_prefix_sums[3]; // L_DISCOUNT

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;

    const uint16_t* comp = meta->compression_method;

    // Shared memory for decompression (32-thread warp version)
    __shared__ uint shared_bws[4];
    __shared__ uint shared_offs[4];
    __shared__ int s_skip;
    __shared__ uint32_t s_comp_blk[136];  // for fused decode_pfor_block_smem

    // Decompression output buffer for this block: 7 slots
    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;
    int32_t* decomp_base = meta->d_decomp_buf
        + (uint64_t)block_id * ENTRIES_PER_BLOCK * elems_per_slot;

    // Per-block profiling accumulators (thread 0 only)
    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    for (uint64_t sd_pg = block_id; sd_pg < npages; sd_pg += gridDim.x) {
        // === 0. Zone map pruning ===
        if (tid == 0) {
            s_skip = 0;
            if (stats && sd_pg < nstats) {
                int32_t page_min = stats[sd_pg * 2];
                int32_t page_max = stats[sd_pg * 2 + 1];
                if (page_max < 19940101 || page_min > 19950100) {
                    s_skip = 1;
                }
            }
        }
        __syncwarp();
        if (s_skip) continue;

        blk_page_count++;

        // === 1. Pre-compute row range from prefix sums ===
        uint64_t row_begin = ps0[sd_pg];
        uint64_t row_end   = ps0[sd_pg + 1];
        uint64_t nrows_this = row_end - row_begin;
        if (nrows_this == 0) continue;

        // === 2. Pre-compute non-driving field page IDs and boundary flags ===
        const uint64_t* ps_arr[3] = { ps1, ps2, ps3 };
        uint64_t fi_pg[3];
        uint64_t fi_bound[3];
        bool needs_boundary[3];

        for (int f = 0; f < 3; f++) {
            fi_pg[f] = find_page_for_row(ps_arr[f], npages, row_begin);
            fi_bound[f] = ps_arr[f][fi_pg[f] + 1];
            needs_boundary[f] = (fi_bound[f] < row_end && fi_pg[f] + 1 < npages);
            if (!needs_boundary[f]) fi_bound[f] = row_end;
        }

        // === 3. Pack I/O descriptors into arrays ===
        // Slots: 0=SD, 1=QT_A, 2=EP_A, 3=DC_A, 4+=boundary pages
        uint64_t io_lba[7];
        unsigned long long io_entry[7];
        uint32_t io_nblocks[7];  // per-IO block count (compressed = actual size)
        uint32_t io_dev[7];      // target device index (RAID0)
        int n_ios = 0;

        // L_SHIPDATE
        {
            uint64_t sd_global_pg = meta->field_start_page_ids[0] + sd_pg;
            uint32_t dev = sd_global_pg % ndev;
            if (comp[0] != 0) {
                io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[0][sd_pg] / 512;
                io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[0][sd_pg]);
            } else {
                uint64_t local_pg = sd_global_pg / ndev;
                io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                io_nblocks[n_ios] = blocks_per_page;
            }
            io_dev[n_ios] = dev;
        }
        io_entry[n_ios] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 0;
        n_ios++;

        // Non-driving fields: primary pages
        unsigned long long fi_entry_a[3], fi_entry_b[3];
        for (int f = 0; f < 3; f++) {
            fi_entry_a[f] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 1 + f * 2;
            fi_entry_b[f] = fi_entry_a[f] + 1;

            uint64_t global_pg = meta->field_start_page_ids[f + 1] + fi_pg[f];
            uint32_t dev = global_pg % ndev;
            if (comp[f + 1] != 0) {
                io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[f + 1][fi_pg[f]] / 512;
                io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f]]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                io_nblocks[n_ios] = blocks_per_page;
            }
            io_dev[n_ios] = dev;
            io_entry[n_ios] = fi_entry_a[f];
            n_ios++;
        }

        // Boundary pages (packed after primary)
        int boundary_this_page = 0;
        for (int f = 0; f < 3; f++) {
            if (needs_boundary[f]) {
                uint64_t global_pg = meta->field_start_page_ids[f + 1] + fi_pg[f] + 1;
                uint32_t dev = global_pg % ndev;
                if (comp[f + 1] != 0) {
                    io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[f + 1][fi_pg[f] + 1] / 512;
                    io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f] + 1]);
                } else {
                    uint64_t local_pg = global_pg / ndev;
                    io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                    io_nblocks[n_ios] = blocks_per_page;
                }
                io_dev[n_ios] = dev;
                io_entry[n_ios] = fi_entry_b[f];
                n_ios++;
                boundary_this_page++;
            }
        }
        blk_boundary_count += boundary_this_page;

        // === 4. I/O: per-lane async submit → syncwarp → poll → syncwarp ===
        long long t0 = clock64();

        uint16_t my_cid = 0;
        uint16_t my_sq_pos = 0;

        // Submit: each lane independently issues 1 NVMe command to its target device
        if ((int)tid < n_ios) {
            QueuePair* my_qp = ctrls[io_dev[tid]]->d_qps
                             + (block_id % ctrls[io_dev[tid]]->n_qps);
            access_data_async(pc, my_qp, io_lba[tid], io_nblocks[tid],
                              io_entry[tid], NVM_IO_READ, &my_cid, &my_sq_pos);
        }
        __syncwarp();

        // Poll: each lane polls its own CID on its target device's QP
        if ((int)tid < n_ios) {
            QueuePair* my_qp = ctrls[io_dev[tid]]->d_qps
                             + (block_id % ctrls[io_dev[tid]]->n_qps);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&my_qp->cq, my_cid, &poll_loc, &poll_head);
            cq_dequeue(&my_qp->cq, cq_pos, &my_qp->sq);
            put_cid(&my_qp->sq, my_cid);
        }
        __syncwarp();

        long long t1 = clock64();
        blk_io_cycles += (uint64_t)(t1 - t0);
        blk_io_count += n_ios;
        for (int ii = 0; ii < n_ios; ii++) blk_io_bytes += (uint64_t)io_nblocks[ii] * 512;

        // === 5. Fused decompress + Q6 eval ===
        long long td0 = clock64();

        bool all_comp = (comp[0] != 0 && comp[1] != 0 &&
                         comp[2] != 0 && comp[3] != 0);
        bool has_boundary = (needs_boundary[0] || needs_boundary[1] || needs_boundary[2]);

        int64_t local_rev = 0;
        uint64_t local_sum_sd = 0, local_sum_qt = 0, local_sum_ep = 0, local_sum_dc = 0;

        if (all_comp && !has_boundary) {
            // ── Case A: all compressed, no boundary ──
            // Decode PFOR blocks via shared memory, eval Q6 on registers.
            char* sub_page_ptr[4] = {
                base + io_entry[0]   * page_size,  // SD
                base + fi_entry_a[0] * page_size,  // QT
                base + fi_entry_a[1] * page_size,  // EP
                base + fi_entry_a[2] * page_size,  // DC
            };

            bam_pag_head* hdr0 = (bam_pag_head*)sub_page_ptr[0];
            uint32_t nalloc = hdr0->nalloc;
            uint32_t nblocks_pfor = hdr0->watermark / 128;

            uint32_t* blk_start[4];
            uint32_t* data_ptr_f[4];
            for (int f = 0; f < 4; f++) {
                blk_start[f] = (uint32_t*)(sub_page_ptr[f] + BAM_PAG_HDR_BYTES);
                data_ptr_f[f] = blk_start[f] + (nblocks_pfor + 1);
            }

            for (uint32_t b = 0; b < nblocks_pfor; b++) {
                int32_t sd_v[4], qt_v[4], ep_v[4], dc_v[4];

                decode_pfor_block_smem(
                    data_ptr_f[0] + blk_start[0][b],
                    blk_start[0][b + 1] - blk_start[0][b],
                    s_comp_blk, shared_bws, shared_offs, tid, sd_v);
                decode_pfor_block_smem(
                    data_ptr_f[1] + blk_start[1][b],
                    blk_start[1][b + 1] - blk_start[1][b],
                    s_comp_blk, shared_bws, shared_offs, tid, qt_v);
                decode_pfor_block_smem(
                    data_ptr_f[2] + blk_start[2][b],
                    blk_start[2][b + 1] - blk_start[2][b],
                    s_comp_blk, shared_bws, shared_offs, tid, ep_v);
                decode_pfor_block_smem(
                    data_ptr_f[3] + blk_start[3][b],
                    blk_start[3][b + 1] - blk_start[3][b],
                    s_comp_blk, shared_bws, shared_offs, tid, dc_v);

                for (uint32_t ki = 0; ki < 4; ki++) {
                    uint32_t idx = b * 128 + ki * 32 + tid;
                    if (idx < nalloc) {
                        local_sum_sd += (uint32_t)sd_v[ki];
                        local_sum_qt += (uint32_t)qt_v[ki];
                        local_sum_ep += (uint32_t)ep_v[ki];
                        local_sum_dc += (uint32_t)dc_v[ki];

                        if (sd_v[ki] >= 19940101 && sd_v[ki] < 19950101 &&
                            dc_v[ki] >= 5 && dc_v[ki] <= 7 &&
                            qt_v[ki] < 2400) {
                            local_rev += (int64_t)ep_v[ki] * dc_v[ki];
                        }
                    }
                }
            }
        } else {
            // ── Case B: fallback (boundary or mixed compression) ──
            int32_t* sd_decomp = decomp_base + 0 * elems_per_slot;
            decompress_page_warp(base + io_entry[0] * page_size, sd_decomp,
                                 comp[0], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + fi_entry_a[0] * page_size,
                                 decomp_base + 1 * elems_per_slot,
                                 comp[1], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + fi_entry_a[1] * page_size,
                                 decomp_base + 3 * elems_per_slot,
                                 comp[2], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + fi_entry_a[2] * page_size,
                                 decomp_base + 5 * elems_per_slot,
                                 comp[3], shared_bws, shared_offs, tid, nullptr);

            if (needs_boundary[0]) {
                decompress_page_warp(base + fi_entry_b[0] * page_size,
                                     decomp_base + 2 * elems_per_slot,
                                     comp[1], shared_bws, shared_offs, tid, nullptr);
            }
            if (needs_boundary[1]) {
                decompress_page_warp(base + fi_entry_b[1] * page_size,
                                     decomp_base + 4 * elems_per_slot,
                                     comp[2], shared_bws, shared_offs, tid, nullptr);
            }
            if (needs_boundary[2]) {
                decompress_page_warp(base + fi_entry_b[2] * page_size,
                                     decomp_base + 6 * elems_per_slot,
                                     comp[3], shared_bws, shared_offs, tid, nullptr);
            }

            int32_t* qt_a = decomp_base + 1 * elems_per_slot;
            int32_t* qt_b = decomp_base + 2 * elems_per_slot;
            int32_t* ep_a = decomp_base + 3 * elems_per_slot;
            int32_t* ep_b = decomp_base + 4 * elems_per_slot;
            int32_t* dc_a = decomp_base + 5 * elems_per_slot;
            int32_t* dc_b = decomp_base + 6 * elems_per_slot;

            uint64_t qt_base = ps1[fi_pg[0]];
            uint64_t ep_base = ps2[fi_pg[1]];
            uint64_t dc_base = ps3[fi_pg[2]];

            for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 32) {
                uint64_t gr = row_begin + i;
                int32_t l_shipdate = sd_decomp[i];
                int32_t l_quantity = (gr < fi_bound[0])
                    ? qt_a[gr - qt_base] : qt_b[gr - fi_bound[0]];
                int32_t l_extendedprice = (gr < fi_bound[1])
                    ? ep_a[gr - ep_base] : ep_b[gr - fi_bound[1]];
                int32_t l_discount = (gr < fi_bound[2])
                    ? dc_a[gr - dc_base] : dc_b[gr - fi_bound[2]];

                local_sum_sd += (uint32_t)l_shipdate;
                local_sum_qt += (uint32_t)l_quantity;
                local_sum_ep += (uint32_t)l_extendedprice;
                local_sum_dc += (uint32_t)l_discount;

                if (l_shipdate >= 19940101 && l_shipdate < 19950101 &&
                    l_discount >= 5 && l_discount <= 7 &&
                    l_quantity < 2400) {
                    local_rev += (int64_t)l_extendedprice * l_discount;
                }
            }
        }

        long long td1 = clock64();
        // Fused decomp+eval: report all cycles under decomp (eval = 0).
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        // === 7. Warp-level reduction via shuffle, then atomicAdd ===
        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
            local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
            local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
            local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
            local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
        }

        if (tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(local_rev));
            if (d_diag) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                          (unsigned long long)local_sum_sd);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                          (unsigned long long)local_sum_qt);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                          (unsigned long long)local_sum_ep);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                          (unsigned long long)local_sum_dc);
            }
        }
    }

    // === Flush per-block profiling counters to global memory ===
    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
    }
}

// ============================================================
// BAM Q6 kernel — 128 threads/block (4 warps)
//
// Warp 0 handles NVMe I/O (submit + poll) using 32 lanes in
// parallel, identical to the sync kernel's I/O path.
// All 4 warps decode 4 Q6 fields in parallel: each warp handles
// one field's PFOR blocks using the existing decode_pfor_block_smem
// (32-thread) function with per-warp shared memory buffers.
// After decode, all 128 threads evaluate Q6 predicates via a
// shared-memory cross-warp transpose.
//
// This reduces decompress time by ~4x (parallel field decode),
// shortening the NVMe non-contributing gap and improving throughput.
// ============================================================

__global__ void bam_q6_kernel_128t_sync(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;           // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t lane_id = tid % 32;          // 0..31

    // RAID0: per-IO device routing (ndev=1 degenerates to single device)
    const uint32_t ndev = meta->n_devices;

    const uint64_t npages = meta->field_npages;
    const uint32_t blocks_per_page = meta->blocks_per_page;
    const uint64_t page_size = meta->page_size;

    // Prefix sum GPU pointers
    const uint64_t* ps0 = meta->d_prefix_sums[0]; // L_SHIPDATE
    const uint64_t* ps1 = meta->d_prefix_sums[1]; // L_QUANTITY
    const uint64_t* ps2 = meta->d_prefix_sums[2]; // L_EXTENDEDPRICE
    const uint64_t* ps3 = meta->d_prefix_sums[3]; // L_DISCOUNT

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;

    const uint16_t* comp = meta->compression_method;

    // Page cache entry offsets for [SD, QT, EP, DC] within ENTRIES_PER_BLOCK
    const int field_entry_off[4] = {0, 1, 3, 5};

    // ── Shared memory ──
    __shared__ int s_skip;

    // Per-warp decode buffers (4 warps × 136 uint32)
    __shared__ uint32_t s_comp_blk[4 * 136];
    __shared__ uint shared_bws[4 * 4];
    __shared__ uint shared_offs[4 * 4];

    // Cross-warp decode output: s_vals[field][element]
    __shared__ int32_t s_vals[4 * 128];

    // Page metadata (warp 0 → all warps)
    __shared__ uint32_t s_nblocks_pfor;
    __shared__ uint32_t s_nalloc;
    __shared__ int s_all_comp;
    __shared__ int s_has_boundary;
    __shared__ uint64_t s_row_begin;
    __shared__ uint64_t s_row_end;
    __shared__ uint64_t s_fi_pg[3];
    __shared__ uint64_t s_fi_bound[3];
    __shared__ int s_needs_boundary[3];

    // Reduction
    __shared__ int64_t s_warp_rev[4];
    __shared__ uint64_t s_warp_sum[4 * 4]; // [warp * 4 + {sd,qt,ep,dc}]

    // Decompression output buffer for Case B fallback
    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;
    int32_t* decomp_base = meta->d_decomp_buf
        + (uint64_t)block_id * ENTRIES_PER_BLOCK * elems_per_slot;

    // Per-block profiling accumulators (thread 0 only)
    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    for (uint64_t sd_pg = block_id; sd_pg < npages; sd_pg += gridDim.x) {

        // === Phase 0: Zone map pruning (thread 0) ===
        if (tid == 0) {
            s_skip = 0;
            if (stats && sd_pg < nstats) {
                int32_t page_min = stats[sd_pg * 2];
                int32_t page_max = stats[sd_pg * 2 + 1];
                if (page_max < meta->sd_low || page_min > meta->sd_high - 1) {
                    s_skip = 1;
                }
            }
        }
        __syncthreads();
        if (s_skip) continue;

        // === Phase 1: I/O — warp 0 only ===
        if (warp_id == 0) {
            uint64_t row_begin = ps0[sd_pg];
            uint64_t row_end   = ps0[sd_pg + 1];
            uint64_t nrows_this = row_end - row_begin;

            if (lane_id == 0) {
                if (nrows_this == 0) {
                    s_skip = 1;
                } else {
                    s_skip = 0;
                    s_row_begin = row_begin;
                    s_row_end = row_end;
                }
            }
            __syncwarp();

            if (!s_skip) {
                // Binary search for non-driving field pages
                const uint64_t* ps_arr[3] = { ps1, ps2, ps3 };
                uint64_t fi_pg_local[3];
                uint64_t fi_bound_local[3];
                int needs_boundary_local[3];

                for (int f = 0; f < 3; f++) {
                    fi_pg_local[f] = find_page_for_row(ps_arr[f], npages, row_begin);
                    fi_bound_local[f] = ps_arr[f][fi_pg_local[f] + 1];
                    needs_boundary_local[f] = (fi_bound_local[f] < row_end && fi_pg_local[f] + 1 < npages) ? 1 : 0;
                    if (!needs_boundary_local[f]) fi_bound_local[f] = row_end;
                }

                // Write metadata to shared memory
                if (lane_id == 0) {
                    for (int f = 0; f < 3; f++) {
                        s_fi_pg[f] = fi_pg_local[f];
                        s_fi_bound[f] = fi_bound_local[f];
                        s_needs_boundary[f] = needs_boundary_local[f];
                    }
                    s_all_comp = (comp[0] != 0 && comp[1] != 0 &&
                                  comp[2] != 0 && comp[3] != 0);
                    s_has_boundary = (needs_boundary_local[0] ||
                                     needs_boundary_local[1] ||
                                     needs_boundary_local[2]);
                }

                // Pack I/O descriptors (same as sync kernel)
                unsigned long long entry_base =
                    (unsigned long long)block_id * ENTRIES_PER_BLOCK;
                uint64_t io_lba[7];
                unsigned long long io_entry[7];
                uint32_t io_nblocks[7];
                uint32_t io_dev[7];      // RAID0: target device index
                int n_ios = 0;

                // L_SHIPDATE
                {
                    uint64_t sd_global_pg = meta->field_start_page_ids[0] + sd_pg;
                    uint32_t dev = sd_global_pg % ndev;
                    if (comp[0] != 0) {
                        io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[0][sd_pg] / 512;
                        io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[0][sd_pg]);
                    } else {
                        uint64_t local_pg = sd_global_pg / ndev;
                        io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                        io_nblocks[n_ios] = blocks_per_page;
                    }
                    io_dev[n_ios] = dev;
                }
                io_entry[n_ios] = entry_base + 0;
                n_ios++;

                // Non-driving fields: primary pages
                unsigned long long fi_entry_a[3], fi_entry_b[3];
                for (int f = 0; f < 3; f++) {
                    fi_entry_a[f] = entry_base + 1 + f * 2;
                    fi_entry_b[f] = fi_entry_a[f] + 1;

                    uint64_t fi_global_pg = meta->field_start_page_ids[f + 1] + fi_pg_local[f];
                    uint32_t dev = fi_global_pg % ndev;
                    if (comp[f + 1] != 0) {
                        io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[f + 1][fi_pg_local[f]] / 512;
                        io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg_local[f]]);
                    } else {
                        uint64_t local_pg = fi_global_pg / ndev;
                        io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                        io_nblocks[n_ios] = blocks_per_page;
                    }
                    io_dev[n_ios] = dev;
                    io_entry[n_ios] = fi_entry_a[f];
                    n_ios++;
                }

                // Boundary pages
                int boundary_this = 0;
                for (int f = 0; f < 3; f++) {
                    if (needs_boundary_local[f]) {
                        uint64_t fi_global_pg = meta->field_start_page_ids[f + 1] + fi_pg_local[f] + 1;
                        uint32_t dev = fi_global_pg % ndev;
                        if (comp[f + 1] != 0) {
                            io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[f + 1][fi_pg_local[f] + 1] / 512;
                            io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg_local[f] + 1]);
                        } else {
                            uint64_t local_pg = fi_global_pg / ndev;
                            io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                            io_nblocks[n_ios] = blocks_per_page;
                        }
                        io_dev[n_ios] = dev;
                        io_entry[n_ios] = fi_entry_b[f];
                        n_ios++;
                        boundary_this++;
                    }
                }

                // Submit: lanes 0..n_ios-1 each issue 1 NVMe command
                long long t0 = clock64();
                uint16_t my_cid = 0;
                uint16_t my_sq_pos = 0;
                QueuePair* my_qp = nullptr;
                if ((int)lane_id < n_ios) {
                    my_qp = ctrls[io_dev[lane_id]]->d_qps
                          + (block_id % ctrls[io_dev[lane_id]]->n_qps);
                    access_data_async(pc, my_qp, io_lba[lane_id], io_nblocks[lane_id],
                                      io_entry[lane_id], NVM_IO_READ, &my_cid, &my_sq_pos);
                }
                __syncwarp();

                // Poll
                if ((int)lane_id < n_ios) {
                    uint32_t poll_loc, poll_head;
                    uint32_t cq_pos = cq_poll(&my_qp->cq, my_cid, &poll_loc, &poll_head);
                    cq_dequeue(&my_qp->cq, cq_pos, &my_qp->sq);
                    put_cid(&my_qp->sq, my_cid);
                }
                __syncwarp();

                long long t1 = clock64();
                blk_io_cycles += (uint64_t)(t1 - t0);
                blk_io_count += n_ios;
                for (int ii = 0; ii < n_ios; ii++) blk_io_bytes += (uint64_t)io_nblocks[ii] * 512;
                blk_boundary_count += boundary_this;
                blk_page_count++;
            }
        }
        __syncthreads();
        if (s_skip) continue;   // empty page: all threads skip

        // === Phase 2: Fused decompress + Q6 eval ===
        long long td0 = clock64();

        unsigned long long entry_base =
            (unsigned long long)block_id * ENTRIES_PER_BLOCK;

        int64_t local_rev = 0;
        uint64_t local_sum_sd = 0, local_sum_qt = 0, local_sum_ep = 0, local_sum_dc = 0;

        if (s_all_comp && !s_has_boundary) {
            // ── Case A: 4 warps decode 4 fields in parallel ──
            char* my_page_ptr = base + (entry_base + field_entry_off[warp_id]) * page_size;

            // Warp 0 reads SD header and shares nblocks_pfor, nalloc
            if (warp_id == 0 && lane_id == 0) {
                bam_pag_head* hdr0 = (bam_pag_head*)(base + entry_base * page_size);
                s_nblocks_pfor = hdr0->watermark / 128;
                s_nalloc = hdr0->nalloc;
            }
            __syncthreads();

            uint32_t nblocks_pfor = s_nblocks_pfor;
            uint32_t nalloc = s_nalloc;

            // Each warp's field pointers
            uint32_t* my_blk_start = (uint32_t*)(my_page_ptr + BAM_PAG_HDR_BYTES);
            uint32_t* my_data_ptr = my_blk_start + (nblocks_pfor + 1);

            // Per-warp shared memory pointers
            uint32_t* my_comp_blk = s_comp_blk + warp_id * 136;
            uint* my_bws = shared_bws + warp_id * 4;
            uint* my_offs = shared_offs + warp_id * 4;

            for (uint32_t b = 0; b < nblocks_pfor; b++) {
                // Each warp decodes its field's PFOR block
                int32_t field_vals[4];
                decode_pfor_block_smem(
                    my_data_ptr + my_blk_start[b],
                    my_blk_start[b + 1] - my_blk_start[b],
                    my_comp_blk, my_bws, my_offs,
                    lane_id, field_vals);

                // Write decoded values to cross-warp shared buffer
                for (int k = 0; k < 4; k++) {
                    s_vals[warp_id * 128 + k * 32 + lane_id] = field_vals[k];
                }
                __syncthreads();

                // All 128 threads evaluate Q6 predicates
                uint32_t idx = b * 128 + tid;
                if (idx < nalloc) {
                    int32_t sd_v = s_vals[0 * 128 + tid]; // SD
                    int32_t qt_v = s_vals[1 * 128 + tid]; // QT
                    int32_t ep_v = s_vals[2 * 128 + tid]; // EP
                    int32_t dc_v = s_vals[3 * 128 + tid]; // DC

                    local_sum_sd += (uint32_t)sd_v;
                    local_sum_qt += (uint32_t)qt_v;
                    local_sum_ep += (uint32_t)ep_v;
                    local_sum_dc += (uint32_t)dc_v;

                    if (sd_v >= meta->sd_low && sd_v < meta->sd_high &&
                        dc_v >= 5 && dc_v <= 7 &&
                        qt_v < 2400) {
                        local_rev += (int64_t)ep_v * dc_v;
                    }
                }
                __syncthreads(); // barrier before next block's shared write
            }
        } else {
            // ── Case B: fallback (boundary or mixed compression) ──
            // Warp 0 decompresses all fields; warps 1-3 wait.
            // Case B is rare (~2% of pages), so single-warp decomp is acceptable.
            if (warp_id == 0) {
                decompress_page_warp(base + (entry_base + 0) * page_size,
                    decomp_base + 0 * elems_per_slot,
                    comp[0], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 1) * page_size,
                    decomp_base + 1 * elems_per_slot,
                    comp[1], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 3) * page_size,
                    decomp_base + 3 * elems_per_slot,
                    comp[2], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 5) * page_size,
                    decomp_base + 5 * elems_per_slot,
                    comp[3], shared_bws, shared_offs, lane_id, nullptr);

                if (s_needs_boundary[0]) {
                    decompress_page_warp(base + (entry_base + 2) * page_size,
                        decomp_base + 2 * elems_per_slot,
                        comp[1], shared_bws, shared_offs, lane_id, nullptr);
                }
                if (s_needs_boundary[1]) {
                    decompress_page_warp(base + (entry_base + 4) * page_size,
                        decomp_base + 4 * elems_per_slot,
                        comp[2], shared_bws, shared_offs, lane_id, nullptr);
                }
                if (s_needs_boundary[2]) {
                    decompress_page_warp(base + (entry_base + 6) * page_size,
                        decomp_base + 6 * elems_per_slot,
                        comp[3], shared_bws, shared_offs, lane_id, nullptr);
                }
            }
            __syncthreads();

            // Eval: all 128 threads, stride 128
            int32_t* sd_decomp = decomp_base + 0 * elems_per_slot;
            int32_t* qt_a = decomp_base + 1 * elems_per_slot;
            int32_t* qt_b = decomp_base + 2 * elems_per_slot;
            int32_t* ep_a = decomp_base + 3 * elems_per_slot;
            int32_t* ep_b = decomp_base + 4 * elems_per_slot;
            int32_t* dc_a = decomp_base + 5 * elems_per_slot;
            int32_t* dc_b = decomp_base + 6 * elems_per_slot;

            uint64_t qt_base_off = ps1[s_fi_pg[0]];
            uint64_t ep_base_off = ps2[s_fi_pg[1]];
            uint64_t dc_base_off = ps3[s_fi_pg[2]];

            uint64_t nrows_this = s_row_end - s_row_begin;
            for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 128) {
                uint64_t gr = s_row_begin + i;
                int32_t l_shipdate = sd_decomp[i];
                int32_t l_quantity = (gr < s_fi_bound[0])
                    ? qt_a[gr - qt_base_off] : qt_b[gr - s_fi_bound[0]];
                int32_t l_extendedprice = (gr < s_fi_bound[1])
                    ? ep_a[gr - ep_base_off] : ep_b[gr - s_fi_bound[1]];
                int32_t l_discount = (gr < s_fi_bound[2])
                    ? dc_a[gr - dc_base_off] : dc_b[gr - s_fi_bound[2]];

                local_sum_sd += (uint32_t)l_shipdate;
                local_sum_qt += (uint32_t)l_quantity;
                local_sum_ep += (uint32_t)l_extendedprice;
                local_sum_dc += (uint32_t)l_discount;

                if (l_shipdate >= meta->sd_low && l_shipdate < meta->sd_high &&
                    l_discount >= 5 && l_discount <= 7 &&
                    l_quantity < 2400) {
                    local_rev += (int64_t)l_extendedprice * l_discount;
                }
            }
        }

        long long td1 = clock64();
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        // === Phase 3: Block-level reduction ===
        // Step 1: Warp-level shuffle reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
            local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
            local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
            local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
            local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
        }

        // Step 2: Lane 0 of each warp writes to shared memory
        if (lane_id == 0) {
            s_warp_rev[warp_id] = local_rev;
            s_warp_sum[warp_id * 4 + 0] = local_sum_sd;
            s_warp_sum[warp_id * 4 + 1] = local_sum_qt;
            s_warp_sum[warp_id * 4 + 2] = local_sum_ep;
            s_warp_sum[warp_id * 4 + 3] = local_sum_dc;
        }
        __syncthreads();

        // Step 3: Thread 0 combines and atomicAdds
        if (tid == 0) {
            int64_t block_rev = s_warp_rev[0] + s_warp_rev[1] + s_warp_rev[2] + s_warp_rev[3];
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(block_rev));
            if (d_diag) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                          (unsigned long long)(s_warp_sum[0] + s_warp_sum[4] + s_warp_sum[8] + s_warp_sum[12]));
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                          (unsigned long long)(s_warp_sum[1] + s_warp_sum[5] + s_warp_sum[9] + s_warp_sum[13]));
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                          (unsigned long long)(s_warp_sum[2] + s_warp_sum[6] + s_warp_sum[10] + s_warp_sum[14]));
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                          (unsigned long long)(s_warp_sum[3] + s_warp_sum[7] + s_warp_sum[11] + s_warp_sum[15]));
            }
        }
        // Reset per-page accumulators for next iteration
        local_rev = 0;
        local_sum_sd = local_sum_qt = local_sum_ep = local_sum_dc = 0;
        __syncthreads();
    }

    // Flush per-block profiling counters to global memory
    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
    }
}

// ============================================================
// Pattern 2: 128 threads (4 warps), warp 0 prefetch.
//
// Same structure as 128t_sync but with double-buffered page cache
// entries (14 per block). After Phase 1 IO, warp 0 submits prefetch
// IO for the next page. On the next iteration, warp 0 polls the
// prefetch instead of doing fresh IO, hiding NVMe latency.
//
// CLI: -B 128 -i 2  (io_multi=2 selects prefetch variant)
// ============================================================

// Helper: pack IO descriptors for a given sd_pg into io_lba/io_entry/io_nblocks.
// Returns n_ios (4-7). Binary search for non-driving fields is done here.
// Writes page metadata (fi_pg, fi_bound, needs_boundary, all_comp, has_boundary,
// row_begin, row_end) into the output parameters.
__device__ __forceinline__ int pack_io_descriptors(
    uint64_t sd_pg,
    BAMKernelMeta* meta,
    const uint64_t* ps0, const uint64_t* ps1,
    const uint64_t* ps2, const uint64_t* ps3,
    unsigned long long entry_base,
    // outputs:
    uint64_t* io_lba,             // [7]
    unsigned long long* io_entry, // [7]
    uint32_t* io_nblocks,         // [7]
    uint64_t* out_fi_pg,          // [3]
    uint64_t* out_fi_bound,       // [3]
    int* out_needs_boundary,      // [3]
    int* out_all_comp,
    int* out_has_boundary,
    int* out_boundary_count,
    uint64_t* out_row_begin,
    uint64_t* out_row_end)
{
    const uint64_t npages = meta->field_npages;
    const uint32_t blocks_per_page = meta->blocks_per_page;
    const uint16_t* comp = meta->compression_method;

    uint64_t row_begin = ps0[sd_pg];
    uint64_t row_end   = ps0[sd_pg + 1];
    *out_row_begin = row_begin;
    *out_row_end = row_end;

    if (row_end == row_begin) return 0; // empty page

    // Binary search for non-driving field pages
    const uint64_t* ps_arr[3] = { ps1, ps2, ps3 };
    for (int f = 0; f < 3; f++) {
        out_fi_pg[f] = find_page_for_row(ps_arr[f], npages, row_begin);
        out_fi_bound[f] = ps_arr[f][out_fi_pg[f] + 1];
        out_needs_boundary[f] = (out_fi_bound[f] < row_end && out_fi_pg[f] + 1 < npages) ? 1 : 0;
        if (!out_needs_boundary[f]) out_fi_bound[f] = row_end;
    }

    *out_all_comp = (comp[0] != 0 && comp[1] != 0 && comp[2] != 0 && comp[3] != 0);
    *out_has_boundary = (out_needs_boundary[0] || out_needs_boundary[1] || out_needs_boundary[2]);

    int n_ios = 0;
    int boundary_cnt = 0;

    // L_SHIPDATE
    if (comp[0] != 0) {
        io_lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[0][sd_pg] / 512;
        io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[0][sd_pg]);
    } else {
        uint64_t sd_page_id = meta->field_start_page_ids[0] + sd_pg;
        io_lba[n_ios] = meta->partition_start_lba + sd_page_id * blocks_per_page;
        io_nblocks[n_ios] = blocks_per_page;
    }
    io_entry[n_ios] = entry_base + 0;
    n_ios++;

    // Non-driving fields: primary pages
    unsigned long long fi_entry_a[3], fi_entry_b[3];
    for (int f = 0; f < 3; f++) {
        fi_entry_a[f] = entry_base + 1 + f * 2;
        fi_entry_b[f] = fi_entry_a[f] + 1;

        if (comp[f + 1] != 0) {
            io_lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[f + 1][out_fi_pg[f]] / 512;
            io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][out_fi_pg[f]]);
        } else {
            uint64_t pg_id = meta->field_start_page_ids[f + 1] + out_fi_pg[f];
            io_lba[n_ios] = meta->partition_start_lba + pg_id * blocks_per_page;
            io_nblocks[n_ios] = blocks_per_page;
        }
        io_entry[n_ios] = fi_entry_a[f];
        n_ios++;
    }

    // Boundary pages
    for (int f = 0; f < 3; f++) {
        if (out_needs_boundary[f]) {
            if (comp[f + 1] != 0) {
                io_lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[f + 1][out_fi_pg[f] + 1] / 512;
                io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][out_fi_pg[f] + 1]);
            } else {
                uint64_t pg_id = meta->field_start_page_ids[f + 1] + out_fi_pg[f] + 1;
                io_lba[n_ios] = meta->partition_start_lba + pg_id * blocks_per_page;
                io_nblocks[n_ios] = blocks_per_page;
            }
            io_entry[n_ios] = fi_entry_b[f];
            n_ios++;
            boundary_cnt++;
        }
    }

    *out_boundary_count = boundary_cnt;
    return n_ios;
}

// ── Entries per block for double-buffered kernels: 2 × 7 = 14 ──
#define ENTRIES_PER_BLOCK_DBL 14

__global__ void bam_q6_kernel_128t_pf_sync(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;           // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t lane_id = tid % 32;          // 0..31

    QueuePair* qp = ctrls[0]->d_qps + (block_id % ctrls[0]->n_qps);

    const uint64_t npages = meta->field_npages;
    const uint64_t page_size = meta->page_size;

    const uint64_t* ps0 = meta->d_prefix_sums[0];
    const uint64_t* ps1 = meta->d_prefix_sums[1];
    const uint64_t* ps2 = meta->d_prefix_sums[2];
    const uint64_t* ps3 = meta->d_prefix_sums[3];

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;
    const uint16_t* comp = meta->compression_method;

    const int field_entry_off[4] = {0, 1, 3, 5};

    // ── Shared memory ──
    __shared__ int s_skip;

    __shared__ uint32_t s_comp_blk[4 * 136];
    __shared__ uint shared_bws[4 * 4];
    __shared__ uint shared_offs[4 * 4];
    __shared__ int32_t s_vals[4 * 128];

    // Current page metadata
    __shared__ uint32_t s_nblocks_pfor;
    __shared__ uint32_t s_nalloc;
    __shared__ int s_all_comp;
    __shared__ int s_has_boundary;
    __shared__ uint64_t s_row_begin;
    __shared__ uint64_t s_row_end;
    __shared__ uint64_t s_fi_pg[3];
    __shared__ uint64_t s_fi_bound[3];
    __shared__ int s_needs_boundary[3];

    // Prefetch metadata (separate storage so it doesn't collide with current page)
    __shared__ uint64_t s_pf_row_begin;
    __shared__ uint64_t s_pf_row_end;
    __shared__ uint64_t s_pf_fi_pg[3];
    __shared__ uint64_t s_pf_fi_bound[3];
    __shared__ int s_pf_needs_boundary[3];
    __shared__ int s_pf_all_comp;
    __shared__ int s_pf_has_boundary;

    // Prefetch IO state
    __shared__ int s_pf_active;
    __shared__ uint64_t s_pf_pg;      // which sd_pg was prefetched
    __shared__ int s_pf_slot;          // which slot (0 or 1)
    __shared__ uint16_t s_pf_cid[7];  // CIDs from prefetch submit
    __shared__ int s_pf_n_ios;         // number of prefetch IOs
    __shared__ int s_curr_slot;        // double-buffer slot (visible to all warps)

    __shared__ int64_t s_warp_rev[4];
    __shared__ uint64_t s_warp_sum[4 * 4];

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;
    int32_t* decomp_base_block = meta->d_decomp_buf
        + (uint64_t)block_id * ENTRIES_PER_BLOCK_DBL * elems_per_slot;

    // Profiling
    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;
    uint64_t blk_pf_submit = 0;
    uint64_t blk_pf_hit = 0;

    if (tid == 0) { s_pf_active = 0; s_curr_slot = 0; }
    __syncthreads();

    for (uint64_t sd_pg = block_id; sd_pg < npages; sd_pg += gridDim.x) {

        // === Phase 0: Zone map + poll prefetch ===
        if (tid == 0) {
            s_skip = 0;
            if (stats && sd_pg < nstats) {
                int32_t page_min = stats[sd_pg * 2];
                int32_t page_max = stats[sd_pg * 2 + 1];
                if (page_max < meta->sd_low || page_min > meta->sd_high - 1) {
                    s_skip = 1;
                }
            }
        }
        __syncthreads();

        // If prefetch matches current page, poll and use it.
        // Otherwise keep prefetch in flight (scan-ahead: target may be pages away).
        if (warp_id == 0 && s_pf_active && s_pf_pg == sd_pg) {
            if ((int)lane_id < s_pf_n_ios) {
                uint32_t poll_loc, poll_head;
                uint32_t cq_pos = cq_poll(&qp->cq, s_pf_cid[lane_id], &poll_loc, &poll_head);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq);
                put_cid(&qp->sq, s_pf_cid[lane_id]);
            }
            __syncwarp();

            if (lane_id == 0) {
                if (s_skip == 0) {
                    s_curr_slot = s_pf_slot;
                    s_row_begin = s_pf_row_begin;
                    s_row_end = s_pf_row_end;
                    s_all_comp = s_pf_all_comp;
                    s_has_boundary = s_pf_has_boundary;
                    for (int f = 0; f < 3; f++) {
                        s_fi_pg[f] = s_pf_fi_pg[f];
                        s_fi_bound[f] = s_pf_fi_bound[f];
                        s_needs_boundary[f] = s_pf_needs_boundary[f];
                    }
                    s_skip = -1;  // signal: use prefetched data
                    blk_pf_hit++;
                }
                s_pf_active = 0;
            }
            __syncwarp();
        }
        __syncthreads();

        if (s_skip == 1) continue;  // zone map skip

        int used_prefetch = (s_skip == -1);
        if (tid == 0) s_skip = 0;  // reset flag

        // === Phase 1: IO (warp 0) — fresh or already prefetched ===
        if (warp_id == 0 && !used_prefetch) {
            // Need fresh IO — use a fresh slot
            s_curr_slot = 1 - s_curr_slot;

            uint64_t io_lba[7];
            unsigned long long io_entry[7];
            uint32_t io_nblocks_arr[7];
            uint64_t fi_pg_l[3], fi_bound_l[3];
            int needs_boundary_l[3];
            int all_comp_l, has_boundary_l, boundary_cnt_l;
            uint64_t row_begin_l, row_end_l;

            unsigned long long entry_base =
                (unsigned long long)block_id * ENTRIES_PER_BLOCK_DBL + s_curr_slot * 7;

            int n_ios = pack_io_descriptors(
                sd_pg, meta, ps0, ps1, ps2, ps3, entry_base,
                io_lba, io_entry, io_nblocks_arr,
                fi_pg_l, fi_bound_l, needs_boundary_l,
                &all_comp_l, &has_boundary_l, &boundary_cnt_l,
                &row_begin_l, &row_end_l);

            if (n_ios == 0) {
                if (lane_id == 0) s_skip = 1;
            } else {
                if (lane_id == 0) {
                    s_row_begin = row_begin_l;
                    s_row_end = row_end_l;
                    s_all_comp = all_comp_l;
                    s_has_boundary = has_boundary_l;
                    for (int f = 0; f < 3; f++) {
                        s_fi_pg[f] = fi_pg_l[f];
                        s_fi_bound[f] = fi_bound_l[f];
                        s_needs_boundary[f] = needs_boundary_l[f];
                    }
                }

                // Submit + poll
                long long t0 = clock64();
                uint16_t my_cid = 0;
                uint16_t my_sq_pos = 0;
                if ((int)lane_id < n_ios) {
                    access_data_async(pc, qp, io_lba[lane_id], io_nblocks_arr[lane_id],
                                      io_entry[lane_id], NVM_IO_READ, &my_cid, &my_sq_pos);
                }
                __syncwarp();
                if ((int)lane_id < n_ios) {
                    uint32_t poll_loc, poll_head;
                    uint32_t cq_pos = cq_poll(&qp->cq, my_cid, &poll_loc, &poll_head);
                    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
                    put_cid(&qp->sq, my_cid);
                }
                __syncwarp();
                long long t1 = clock64();
                blk_io_cycles += (uint64_t)(t1 - t0);
                blk_io_count += n_ios;
                for (int ii = 0; ii < n_ios; ii++) blk_io_bytes += (uint64_t)io_nblocks_arr[ii] * 512;
                blk_boundary_count += boundary_cnt_l;
                blk_page_count++;
            }
        }
        if (warp_id == 0 && used_prefetch) {
            blk_page_count++;
        }
        __syncthreads();
        if (s_skip) continue;

        // === Prefetch submit: scan-ahead for next qualifying page (warp 0) ===
        if (warp_id == 0 && !s_pf_active) {
            for (uint64_t pf_pg = sd_pg + gridDim.x; pf_pg < npages; pf_pg += gridDim.x) {
                // Zone map check
                int pf_skip = 0;
                if (stats && pf_pg < nstats) {
                    int32_t page_min = stats[pf_pg * 2];
                    int32_t page_max = stats[pf_pg * 2 + 1];
                    if (page_max < meta->sd_low || page_min > meta->sd_high - 1)
                        pf_skip = 1;
                }
                if (pf_skip) continue;  // skip non-qualifying, keep scanning

                int pf_slot = 1 - s_curr_slot;
                unsigned long long pf_entry_base =
                    (unsigned long long)block_id * ENTRIES_PER_BLOCK_DBL + pf_slot * 7;

                uint64_t pf_io_lba[7];
                unsigned long long pf_io_entry[7];
                uint32_t pf_io_nblocks[7];
                uint64_t pf_fi_pg[3], pf_fi_bound[3];
                int pf_needs_boundary[3];
                int pf_all_comp, pf_has_boundary, pf_boundary_cnt;
                uint64_t pf_row_begin, pf_row_end;

                int pf_n = pack_io_descriptors(
                    pf_pg, meta, ps0, ps1, ps2, ps3, pf_entry_base,
                    pf_io_lba, pf_io_entry, pf_io_nblocks,
                    pf_fi_pg, pf_fi_bound, pf_needs_boundary,
                    &pf_all_comp, &pf_has_boundary, &pf_boundary_cnt,
                    &pf_row_begin, &pf_row_end);

                if (pf_n > 0) {
                    if (lane_id == 0) {
                        s_pf_row_begin = pf_row_begin;
                        s_pf_row_end = pf_row_end;
                        s_pf_all_comp = pf_all_comp;
                        s_pf_has_boundary = pf_has_boundary;
                        for (int f = 0; f < 3; f++) {
                            s_pf_fi_pg[f] = pf_fi_pg[f];
                            s_pf_fi_bound[f] = pf_fi_bound[f];
                            s_pf_needs_boundary[f] = pf_needs_boundary[f];
                        }
                        s_pf_slot = pf_slot;
                        s_pf_pg = pf_pg;
                        s_pf_n_ios = pf_n;
                    }
                    __syncwarp();

                    if ((int)lane_id < pf_n) {
                        uint16_t cid = 0;
                        uint16_t sq_pos = 0;
                        access_data_async(pc, qp, pf_io_lba[lane_id],
                                          pf_io_nblocks[lane_id],
                                          pf_io_entry[lane_id],
                                          NVM_IO_READ, &cid, &sq_pos);
                        s_pf_cid[lane_id] = cid;
                    }
                    __syncwarp();

                    if (lane_id == 0) {
                        s_pf_active = 1;
                        blk_pf_submit++;
                    }
                    blk_io_count += pf_n;
                    for (int ii = 0; ii < pf_n; ii++)
                        blk_io_bytes += (uint64_t)pf_io_nblocks[ii] * 512;
                    blk_boundary_count += pf_boundary_cnt;
                }
                break;  // found qualifying page (or pack failed), stop scanning
            }
        }
        // No __syncthreads() needed: prefetch state is only read by warp 0

        // === Phase 2: Fused decompress + Q6 eval ===
        long long td0 = clock64();

        unsigned long long entry_base =
            (unsigned long long)block_id * ENTRIES_PER_BLOCK_DBL + s_curr_slot * 7;

        int32_t* decomp_base = decomp_base_block + s_curr_slot * 7 * elems_per_slot;

        int64_t local_rev = 0;
        uint64_t local_sum_sd = 0, local_sum_qt = 0, local_sum_ep = 0, local_sum_dc = 0;

        if (s_all_comp && !s_has_boundary) {
            // ── Case A: 4 warps decode 4 fields in parallel ──
            char* my_page_ptr = base + (entry_base + field_entry_off[warp_id]) * page_size;

            if (warp_id == 0 && lane_id == 0) {
                bam_pag_head* hdr0 = (bam_pag_head*)(base + entry_base * page_size);
                s_nblocks_pfor = hdr0->watermark / 128;
                s_nalloc = hdr0->nalloc;
            }
            __syncthreads();

            uint32_t nblocks_pfor = s_nblocks_pfor;
            uint32_t nalloc = s_nalloc;

            uint32_t* my_blk_start = (uint32_t*)(my_page_ptr + BAM_PAG_HDR_BYTES);
            uint32_t* my_data_ptr = my_blk_start + (nblocks_pfor + 1);

            uint32_t* my_comp_blk = s_comp_blk + warp_id * 136;
            uint* my_bws = shared_bws + warp_id * 4;
            uint* my_offs = shared_offs + warp_id * 4;

            for (uint32_t b = 0; b < nblocks_pfor; b++) {
                int32_t field_vals[4];
                decode_pfor_block_smem(
                    my_data_ptr + my_blk_start[b],
                    my_blk_start[b + 1] - my_blk_start[b],
                    my_comp_blk, my_bws, my_offs,
                    lane_id, field_vals);

                for (int k = 0; k < 4; k++) {
                    s_vals[warp_id * 128 + k * 32 + lane_id] = field_vals[k];
                }
                __syncthreads();

                uint32_t idx = b * 128 + tid;
                if (idx < nalloc) {
                    int32_t sd_v = s_vals[0 * 128 + tid];
                    int32_t qt_v = s_vals[1 * 128 + tid];
                    int32_t ep_v = s_vals[2 * 128 + tid];
                    int32_t dc_v = s_vals[3 * 128 + tid];

                    local_sum_sd += (uint32_t)sd_v;
                    local_sum_qt += (uint32_t)qt_v;
                    local_sum_ep += (uint32_t)ep_v;
                    local_sum_dc += (uint32_t)dc_v;

                    if (sd_v >= meta->sd_low && sd_v < meta->sd_high &&
                        dc_v >= 5 && dc_v <= 7 &&
                        qt_v < 2400) {
                        local_rev += (int64_t)ep_v * dc_v;
                    }
                }
                __syncthreads();
            }
        } else {
            // ── Case B: fallback ──
            if (warp_id == 0) {
                decompress_page_warp(base + (entry_base + 0) * page_size,
                    decomp_base + 0 * elems_per_slot,
                    comp[0], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 1) * page_size,
                    decomp_base + 1 * elems_per_slot,
                    comp[1], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 3) * page_size,
                    decomp_base + 3 * elems_per_slot,
                    comp[2], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 5) * page_size,
                    decomp_base + 5 * elems_per_slot,
                    comp[3], shared_bws, shared_offs, lane_id, nullptr);

                if (s_needs_boundary[0]) {
                    decompress_page_warp(base + (entry_base + 2) * page_size,
                        decomp_base + 2 * elems_per_slot,
                        comp[1], shared_bws, shared_offs, lane_id, nullptr);
                }
                if (s_needs_boundary[1]) {
                    decompress_page_warp(base + (entry_base + 4) * page_size,
                        decomp_base + 4 * elems_per_slot,
                        comp[2], shared_bws, shared_offs, lane_id, nullptr);
                }
                if (s_needs_boundary[2]) {
                    decompress_page_warp(base + (entry_base + 6) * page_size,
                        decomp_base + 6 * elems_per_slot,
                        comp[3], shared_bws, shared_offs, lane_id, nullptr);
                }
            }
            __syncthreads();

            int32_t* sd_decomp = decomp_base + 0 * elems_per_slot;
            int32_t* qt_a = decomp_base + 1 * elems_per_slot;
            int32_t* qt_b = decomp_base + 2 * elems_per_slot;
            int32_t* ep_a = decomp_base + 3 * elems_per_slot;
            int32_t* ep_b = decomp_base + 4 * elems_per_slot;
            int32_t* dc_a = decomp_base + 5 * elems_per_slot;
            int32_t* dc_b = decomp_base + 6 * elems_per_slot;

            uint64_t qt_base_off = ps1[s_fi_pg[0]];
            uint64_t ep_base_off = ps2[s_fi_pg[1]];
            uint64_t dc_base_off = ps3[s_fi_pg[2]];

            uint64_t nrows_this = s_row_end - s_row_begin;
            for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 128) {
                uint64_t gr = s_row_begin + i;
                int32_t l_shipdate = sd_decomp[i];
                int32_t l_quantity = (gr < s_fi_bound[0])
                    ? qt_a[gr - qt_base_off] : qt_b[gr - s_fi_bound[0]];
                int32_t l_extendedprice = (gr < s_fi_bound[1])
                    ? ep_a[gr - ep_base_off] : ep_b[gr - s_fi_bound[1]];
                int32_t l_discount = (gr < s_fi_bound[2])
                    ? dc_a[gr - dc_base_off] : dc_b[gr - s_fi_bound[2]];

                local_sum_sd += (uint32_t)l_shipdate;
                local_sum_qt += (uint32_t)l_quantity;
                local_sum_ep += (uint32_t)l_extendedprice;
                local_sum_dc += (uint32_t)l_discount;

                if (l_shipdate >= meta->sd_low && l_shipdate < meta->sd_high &&
                    l_discount >= 5 && l_discount <= 7 &&
                    l_quantity < 2400) {
                    local_rev += (int64_t)l_extendedprice * l_discount;
                }
            }
        }

        long long td1 = clock64();
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        // === Phase 3: Reduction ===
        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
            local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
            local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
            local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
            local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
        }

        if (lane_id == 0) {
            s_warp_rev[warp_id] = local_rev;
            s_warp_sum[warp_id * 4 + 0] = local_sum_sd;
            s_warp_sum[warp_id * 4 + 1] = local_sum_qt;
            s_warp_sum[warp_id * 4 + 2] = local_sum_ep;
            s_warp_sum[warp_id * 4 + 3] = local_sum_dc;
        }
        __syncthreads();

        if (tid == 0) {
            int64_t block_rev = s_warp_rev[0] + s_warp_rev[1] + s_warp_rev[2] + s_warp_rev[3];
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(block_rev));
            if (d_diag) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                          (unsigned long long)(s_warp_sum[0] + s_warp_sum[4] + s_warp_sum[8] + s_warp_sum[12]));
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                          (unsigned long long)(s_warp_sum[1] + s_warp_sum[5] + s_warp_sum[9] + s_warp_sum[13]));
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                          (unsigned long long)(s_warp_sum[2] + s_warp_sum[6] + s_warp_sum[10] + s_warp_sum[14]));
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                          (unsigned long long)(s_warp_sum[3] + s_warp_sum[7] + s_warp_sum[11] + s_warp_sum[15]));
            }
        }
        local_rev = 0;
        local_sum_sd = local_sum_qt = local_sum_ep = local_sum_dc = 0;
        __syncthreads();
    }

    // Flush profiling counters
    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->pf_submit_count),
                  (unsigned long long)blk_pf_submit);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->pf_hit_count),
                  (unsigned long long)blk_pf_hit);
    }
}

// ============================================================
// Pattern 1: 160 threads (5 warps), warp 0 = IO exclusive.
//
// Warp 0 never participates in decode/eval. During the PFOR decode
// loop, warp 0 computes the IO plan for the next page and submits
// prefetch IO in parallel with decode. True concurrent IO+decode.
//
// CLI: -B 160
// ============================================================

__global__ void bam_q6_kernel_160t_sync(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;           // 0..159
    const uint32_t warp_id = tid / 32;          // 0..4
    const uint32_t lane_id = tid % 32;          // 0..31

    // Decode warps: warp 1-4 → decode_warp 0-3
    const int decode_warp = (int)warp_id - 1;
    // Eval thread index within 128-thread decode group
    const uint32_t eval_tid = tid - 32;  // valid only for warp_id > 0

    QueuePair* qp = ctrls[0]->d_qps + (block_id % ctrls[0]->n_qps);

    const uint64_t npages = meta->field_npages;
    const uint64_t page_size = meta->page_size;

    const uint64_t* ps0 = meta->d_prefix_sums[0];
    const uint64_t* ps1 = meta->d_prefix_sums[1];
    const uint64_t* ps2 = meta->d_prefix_sums[2];
    const uint64_t* ps3 = meta->d_prefix_sums[3];

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;
    const uint16_t* comp = meta->compression_method;

    const int field_entry_off[4] = {0, 1, 3, 5};

    // ── Shared memory ──
    __shared__ int s_skip;

    // Decode buffers for warps 1-4 (4 decode warps)
    __shared__ uint32_t s_comp_blk[4 * 136];
    __shared__ uint shared_bws[4 * 4];
    __shared__ uint shared_offs[4 * 4];
    __shared__ int32_t s_vals[4 * 128];

    // Current page metadata
    __shared__ uint32_t s_nblocks_pfor;
    __shared__ uint32_t s_nalloc;
    __shared__ int s_all_comp;
    __shared__ int s_has_boundary;
    __shared__ uint64_t s_row_begin;
    __shared__ uint64_t s_row_end;
    __shared__ uint64_t s_fi_pg[3];
    __shared__ uint64_t s_fi_bound[3];
    __shared__ int s_needs_boundary[3];

    // Prefetch metadata
    __shared__ uint64_t s_pf_row_begin;
    __shared__ uint64_t s_pf_row_end;
    __shared__ uint64_t s_pf_fi_pg[3];
    __shared__ uint64_t s_pf_fi_bound[3];
    __shared__ int s_pf_needs_boundary[3];
    __shared__ int s_pf_all_comp;
    __shared__ int s_pf_has_boundary;

    // Prefetch IO state
    __shared__ int s_pf_active;
    __shared__ uint64_t s_pf_pg;
    __shared__ int s_pf_slot;
    __shared__ uint16_t s_pf_cid[7];
    __shared__ int s_pf_n_ios;
    __shared__ int s_curr_slot;        // double-buffer slot (visible to all warps)

    // Reduction (5 warps, but only warps 1-4 contribute)
    __shared__ int64_t s_warp_rev[5];
    __shared__ uint64_t s_warp_sum[5 * 4];

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;
    int32_t* decomp_base_block = meta->d_decomp_buf
        + (uint64_t)block_id * ENTRIES_PER_BLOCK_DBL * elems_per_slot;

    // Profiling (thread 0 = warp 0 lane 0)
    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;
    uint64_t blk_pf_submit = 0;
    uint64_t blk_pf_hit = 0;

    if (tid == 0) {
        s_pf_active = 0;
        s_curr_slot = 0;
        s_warp_rev[0] = 0;
        s_warp_sum[0] = s_warp_sum[1] = s_warp_sum[2] = s_warp_sum[3] = 0;
    }
    __syncthreads();

    for (uint64_t sd_pg = block_id; sd_pg < npages; sd_pg += gridDim.x) {

        // === Phase 0: Zone map + poll prefetch ===
        if (tid == 0) {
            s_skip = 0;
            if (stats && sd_pg < nstats) {
                int32_t page_min = stats[sd_pg * 2];
                int32_t page_max = stats[sd_pg * 2 + 1];
                if (page_max < meta->sd_low || page_min > meta->sd_high - 1) {
                    s_skip = 1;
                }
            }
        }
        __syncthreads();

        // Poll prefetch if active and matches current page (scan-ahead)
        if (warp_id == 0 && s_pf_active && s_pf_pg == sd_pg) {
            if ((int)lane_id < s_pf_n_ios) {
                uint32_t poll_loc, poll_head;
                uint32_t cq_pos = cq_poll(&qp->cq, s_pf_cid[lane_id], &poll_loc, &poll_head);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq);
                put_cid(&qp->sq, s_pf_cid[lane_id]);
            }
            __syncwarp();

            if (lane_id == 0) {
                if (s_skip == 0) {
                    s_curr_slot = s_pf_slot;
                    s_row_begin = s_pf_row_begin;
                    s_row_end = s_pf_row_end;
                    s_all_comp = s_pf_all_comp;
                    s_has_boundary = s_pf_has_boundary;
                    for (int f = 0; f < 3; f++) {
                        s_fi_pg[f] = s_pf_fi_pg[f];
                        s_fi_bound[f] = s_pf_fi_bound[f];
                        s_needs_boundary[f] = s_pf_needs_boundary[f];
                    }
                    s_skip = -1;
                    blk_pf_hit++;
                }
                s_pf_active = 0;
            }
            __syncwarp();
        }
        __syncthreads();

        if (s_skip == 1) continue;

        int used_prefetch = (s_skip == -1);
        if (tid == 0) s_skip = 0;

        // === Phase 1: IO (warp 0 only) ===
        if (warp_id == 0 && !used_prefetch) {
            s_curr_slot = 1 - s_curr_slot;

            uint64_t io_lba[7];
            unsigned long long io_entry[7];
            uint32_t io_nblocks_arr[7];
            uint64_t fi_pg_l[3], fi_bound_l[3];
            int needs_boundary_l[3];
            int all_comp_l, has_boundary_l, boundary_cnt_l;
            uint64_t row_begin_l, row_end_l;

            unsigned long long entry_base =
                (unsigned long long)block_id * ENTRIES_PER_BLOCK_DBL + s_curr_slot * 7;

            int n_ios = pack_io_descriptors(
                sd_pg, meta, ps0, ps1, ps2, ps3, entry_base,
                io_lba, io_entry, io_nblocks_arr,
                fi_pg_l, fi_bound_l, needs_boundary_l,
                &all_comp_l, &has_boundary_l, &boundary_cnt_l,
                &row_begin_l, &row_end_l);

            if (n_ios == 0) {
                if (lane_id == 0) s_skip = 1;
            } else {
                if (lane_id == 0) {
                    s_row_begin = row_begin_l;
                    s_row_end = row_end_l;
                    s_all_comp = all_comp_l;
                    s_has_boundary = has_boundary_l;
                    for (int f = 0; f < 3; f++) {
                        s_fi_pg[f] = fi_pg_l[f];
                        s_fi_bound[f] = fi_bound_l[f];
                        s_needs_boundary[f] = needs_boundary_l[f];
                    }
                }

                long long t0 = clock64();
                uint16_t my_cid = 0;
                uint16_t my_sq_pos = 0;
                if ((int)lane_id < n_ios) {
                    access_data_async(pc, qp, io_lba[lane_id], io_nblocks_arr[lane_id],
                                      io_entry[lane_id], NVM_IO_READ, &my_cid, &my_sq_pos);
                }
                __syncwarp();
                if ((int)lane_id < n_ios) {
                    uint32_t poll_loc, poll_head;
                    uint32_t cq_pos = cq_poll(&qp->cq, my_cid, &poll_loc, &poll_head);
                    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
                    put_cid(&qp->sq, my_cid);
                }
                __syncwarp();
                long long t1 = clock64();
                blk_io_cycles += (uint64_t)(t1 - t0);
                blk_io_count += n_ios;
                for (int ii = 0; ii < n_ios; ii++) blk_io_bytes += (uint64_t)io_nblocks_arr[ii] * 512;
                blk_boundary_count += boundary_cnt_l;
                blk_page_count++;
            }
        }
        if (warp_id == 0 && used_prefetch) {
            blk_page_count++;
        }
        __syncthreads();
        if (s_skip) continue;

        // === Phase 2: Decode (warps 1-4) + Prefetch (warp 0) ===
        long long td0 = clock64();

        unsigned long long entry_base =
            (unsigned long long)block_id * ENTRIES_PER_BLOCK_DBL + s_curr_slot * 7;

        int32_t* decomp_base = decomp_base_block + s_curr_slot * 7 * elems_per_slot;

        int64_t local_rev = 0;
        uint64_t local_sum_sd = 0, local_sum_qt = 0, local_sum_ep = 0, local_sum_dc = 0;

        if (s_all_comp && !s_has_boundary) {
            // ── Case A: Warps 1-4 decode, warp 0 prefetches ──
            // Read header (warp 1 = decode_warp 0)
            if (warp_id == 1 && lane_id == 0) {
                bam_pag_head* hdr0 = (bam_pag_head*)(base + entry_base * page_size);
                s_nblocks_pfor = hdr0->watermark / 128;
                s_nalloc = hdr0->nalloc;
            }
            __syncthreads();

            uint32_t nblocks_pfor = s_nblocks_pfor;
            uint32_t nalloc = s_nalloc;

            // Warp 0: compute and submit prefetch in iteration b == 0
            int pf_submitted = 0;

            for (uint32_t b = 0; b < nblocks_pfor; b++) {

                if (warp_id == 0) {
                    // Prefetch: scan-ahead for next qualifying page during first PFOR block
                    if (b == 0 && !pf_submitted && !s_pf_active) {
                        for (uint64_t pf_pg = sd_pg + gridDim.x; pf_pg < npages; pf_pg += gridDim.x) {
                            int pf_skip = 0;
                            if (stats && pf_pg < nstats) {
                                int32_t page_min = stats[pf_pg * 2];
                                int32_t page_max = stats[pf_pg * 2 + 1];
                                if (page_max < meta->sd_low || page_min > meta->sd_high - 1)
                                    pf_skip = 1;
                            }
                            if (pf_skip) continue;

                            int pf_slot = 1 - s_curr_slot;
                            unsigned long long pf_entry_base =
                                (unsigned long long)block_id * ENTRIES_PER_BLOCK_DBL + pf_slot * 7;

                            uint64_t pf_io_lba[7];
                            unsigned long long pf_io_entry[7];
                            uint32_t pf_io_nblocks[7];
                            uint64_t pf_fi_pg[3], pf_fi_bound[3];
                            int pf_needs_boundary[3];
                            int pf_all_comp, pf_has_boundary, pf_boundary_cnt;
                            uint64_t pf_row_begin, pf_row_end;

                            int pf_n = pack_io_descriptors(
                                pf_pg, meta, ps0, ps1, ps2, ps3, pf_entry_base,
                                pf_io_lba, pf_io_entry, pf_io_nblocks,
                                pf_fi_pg, pf_fi_bound, pf_needs_boundary,
                                &pf_all_comp, &pf_has_boundary, &pf_boundary_cnt,
                                &pf_row_begin, &pf_row_end);

                            if (pf_n > 0) {
                                if (lane_id == 0) {
                                    s_pf_row_begin = pf_row_begin;
                                    s_pf_row_end = pf_row_end;
                                    s_pf_all_comp = pf_all_comp;
                                    s_pf_has_boundary = pf_has_boundary;
                                    for (int f = 0; f < 3; f++) {
                                        s_pf_fi_pg[f] = pf_fi_pg[f];
                                        s_pf_fi_bound[f] = pf_fi_bound[f];
                                        s_pf_needs_boundary[f] = pf_needs_boundary[f];
                                    }
                                    s_pf_slot = pf_slot;
                                    s_pf_pg = pf_pg;
                                    s_pf_n_ios = pf_n;
                                }
                                __syncwarp();

                                if ((int)lane_id < pf_n) {
                                    uint16_t cid = 0, sq_pos = 0;
                                    access_data_async(pc, qp, pf_io_lba[lane_id],
                                                      pf_io_nblocks[lane_id],
                                                      pf_io_entry[lane_id],
                                                      NVM_IO_READ, &cid, &sq_pos);
                                    s_pf_cid[lane_id] = cid;
                                }
                                __syncwarp();

                                if (lane_id == 0) { s_pf_active = 1; blk_pf_submit++; }
                                blk_io_count += pf_n;
                                for (int ii = 0; ii < pf_n; ii++)
                                    blk_io_bytes += (uint64_t)pf_io_nblocks[ii] * 512;
                                blk_boundary_count += pf_boundary_cnt;
                            }
                            break;  // found qualifying page, stop scanning
                        }
                        pf_submitted = 1;
                    }
                    // Warp 0 reaches __syncthreads() below (participates in barrier)
                } else {
                    // Warps 1-4: decode their field's PFOR block
                    char* my_page_ptr = base + (entry_base + field_entry_off[decode_warp]) * page_size;
                    uint32_t* my_blk_start = (uint32_t*)(my_page_ptr + BAM_PAG_HDR_BYTES);
                    uint32_t* my_data_ptr = my_blk_start + (nblocks_pfor + 1);

                    uint32_t* my_comp_blk = s_comp_blk + decode_warp * 136;
                    uint* my_bws = shared_bws + decode_warp * 4;
                    uint* my_offs = shared_offs + decode_warp * 4;

                    int32_t field_vals[4];
                    decode_pfor_block_smem(
                        my_data_ptr + my_blk_start[b],
                        my_blk_start[b + 1] - my_blk_start[b],
                        my_comp_blk, my_bws, my_offs,
                        lane_id, field_vals);

                    for (int k = 0; k < 4; k++) {
                        s_vals[decode_warp * 128 + k * 32 + lane_id] = field_vals[k];
                    }
                }
                __syncthreads();

                // Eval: warps 1-4 only (128 threads, eval_tid 0..127)
                if (warp_id > 0) {
                    uint32_t idx = b * 128 + eval_tid;
                    if (idx < nalloc) {
                        int32_t sd_v = s_vals[0 * 128 + eval_tid];
                        int32_t qt_v = s_vals[1 * 128 + eval_tid];
                        int32_t ep_v = s_vals[2 * 128 + eval_tid];
                        int32_t dc_v = s_vals[3 * 128 + eval_tid];

                        local_sum_sd += (uint32_t)sd_v;
                        local_sum_qt += (uint32_t)qt_v;
                        local_sum_ep += (uint32_t)ep_v;
                        local_sum_dc += (uint32_t)dc_v;

                        if (sd_v >= meta->sd_low && sd_v < meta->sd_high &&
                            dc_v >= 5 && dc_v <= 7 &&
                            qt_v < 2400) {
                            local_rev += (int64_t)ep_v * dc_v;
                        }
                    }
                }
                __syncthreads();
            }
        } else {
            // ── Case B: fallback ──
            // Warp 1 decompresses all fields (warp 0 is IO-exclusive)
            if (warp_id == 1) {
                decompress_page_warp(base + (entry_base + 0) * page_size,
                    decomp_base + 0 * elems_per_slot,
                    comp[0], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 1) * page_size,
                    decomp_base + 1 * elems_per_slot,
                    comp[1], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 3) * page_size,
                    decomp_base + 3 * elems_per_slot,
                    comp[2], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 5) * page_size,
                    decomp_base + 5 * elems_per_slot,
                    comp[3], shared_bws, shared_offs, lane_id, nullptr);

                if (s_needs_boundary[0]) {
                    decompress_page_warp(base + (entry_base + 2) * page_size,
                        decomp_base + 2 * elems_per_slot,
                        comp[1], shared_bws, shared_offs, lane_id, nullptr);
                }
                if (s_needs_boundary[1]) {
                    decompress_page_warp(base + (entry_base + 4) * page_size,
                        decomp_base + 4 * elems_per_slot,
                        comp[2], shared_bws, shared_offs, lane_id, nullptr);
                }
                if (s_needs_boundary[2]) {
                    decompress_page_warp(base + (entry_base + 6) * page_size,
                        decomp_base + 6 * elems_per_slot,
                        comp[3], shared_bws, shared_offs, lane_id, nullptr);
                }
            }
            // Warp 0: scan-ahead prefetch during Case B decomp
            if (warp_id == 0 && !s_pf_active) {
                for (uint64_t pf_pg = sd_pg + gridDim.x; pf_pg < npages; pf_pg += gridDim.x) {
                    int pf_skip = 0;
                    if (stats && pf_pg < nstats) {
                        int32_t page_min = stats[pf_pg * 2];
                        int32_t page_max = stats[pf_pg * 2 + 1];
                        if (page_max < meta->sd_low || page_min > meta->sd_high - 1)
                            pf_skip = 1;
                    }
                    if (pf_skip) continue;

                    int pf_slot = 1 - s_curr_slot;
                    unsigned long long pf_entry_base =
                        (unsigned long long)block_id * ENTRIES_PER_BLOCK_DBL + pf_slot * 7;

                    uint64_t pf_io_lba[7];
                    unsigned long long pf_io_entry[7];
                    uint32_t pf_io_nblocks[7];
                    uint64_t pf_fi_pg[3], pf_fi_bound[3];
                    int pf_needs_boundary[3];
                    int pf_all_comp, pf_has_boundary, pf_boundary_cnt;
                    uint64_t pf_row_begin, pf_row_end;

                    int pf_n = pack_io_descriptors(
                        pf_pg, meta, ps0, ps1, ps2, ps3, pf_entry_base,
                        pf_io_lba, pf_io_entry, pf_io_nblocks,
                        pf_fi_pg, pf_fi_bound, pf_needs_boundary,
                        &pf_all_comp, &pf_has_boundary, &pf_boundary_cnt,
                        &pf_row_begin, &pf_row_end);

                    if (pf_n > 0) {
                        if (lane_id == 0) {
                            s_pf_row_begin = pf_row_begin;
                            s_pf_row_end = pf_row_end;
                            s_pf_all_comp = pf_all_comp;
                            s_pf_has_boundary = pf_has_boundary;
                            for (int f = 0; f < 3; f++) {
                                s_pf_fi_pg[f] = pf_fi_pg[f];
                                s_pf_fi_bound[f] = pf_fi_bound[f];
                                s_pf_needs_boundary[f] = pf_needs_boundary[f];
                            }
                            s_pf_slot = pf_slot;
                            s_pf_pg = pf_pg;
                            s_pf_n_ios = pf_n;
                        }
                        __syncwarp();
                        if ((int)lane_id < pf_n) {
                            uint16_t cid = 0, sq_pos = 0;
                            access_data_async(pc, qp, pf_io_lba[lane_id],
                                              pf_io_nblocks[lane_id],
                                              pf_io_entry[lane_id],
                                              NVM_IO_READ, &cid, &sq_pos);
                            s_pf_cid[lane_id] = cid;
                        }
                        __syncwarp();
                        if (lane_id == 0) { s_pf_active = 1; blk_pf_submit++; }
                        blk_io_count += pf_n;
                        for (int ii = 0; ii < pf_n; ii++)
                            blk_io_bytes += (uint64_t)pf_io_nblocks[ii] * 512;
                        blk_boundary_count += pf_boundary_cnt;
                    }
                    break;
                }
            }
            __syncthreads();

            // Eval: warps 1-4 only
            if (warp_id > 0) {
                int32_t* sd_decomp = decomp_base + 0 * elems_per_slot;
                int32_t* qt_a = decomp_base + 1 * elems_per_slot;
                int32_t* qt_b = decomp_base + 2 * elems_per_slot;
                int32_t* ep_a = decomp_base + 3 * elems_per_slot;
                int32_t* ep_b = decomp_base + 4 * elems_per_slot;
                int32_t* dc_a = decomp_base + 5 * elems_per_slot;
                int32_t* dc_b = decomp_base + 6 * elems_per_slot;

                uint64_t qt_base_off = ps1[s_fi_pg[0]];
                uint64_t ep_base_off = ps2[s_fi_pg[1]];
                uint64_t dc_base_off = ps3[s_fi_pg[2]];

                uint64_t nrows_this = s_row_end - s_row_begin;
                for (uint32_t i = eval_tid; i < (uint32_t)nrows_this; i += 128) {
                    uint64_t gr = s_row_begin + i;
                    int32_t l_shipdate = sd_decomp[i];
                    int32_t l_quantity = (gr < s_fi_bound[0])
                        ? qt_a[gr - qt_base_off] : qt_b[gr - s_fi_bound[0]];
                    int32_t l_extendedprice = (gr < s_fi_bound[1])
                        ? ep_a[gr - ep_base_off] : ep_b[gr - s_fi_bound[1]];
                    int32_t l_discount = (gr < s_fi_bound[2])
                        ? dc_a[gr - dc_base_off] : dc_b[gr - s_fi_bound[2]];

                    local_sum_sd += (uint32_t)l_shipdate;
                    local_sum_qt += (uint32_t)l_quantity;
                    local_sum_ep += (uint32_t)l_extendedprice;
                    local_sum_dc += (uint32_t)l_discount;

                    if (l_shipdate >= meta->sd_low && l_shipdate < meta->sd_high &&
                        l_discount >= 5 && l_discount <= 7 &&
                        l_quantity < 2400) {
                        local_rev += (int64_t)l_extendedprice * l_discount;
                    }
                }
            }
        }

        long long td1 = clock64();
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        // === Phase 3: Reduction (5 warps, warp 0 contributes 0) ===
        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
            local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
            local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
            local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
            local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
        }

        if (lane_id == 0) {
            s_warp_rev[warp_id] = local_rev;
            s_warp_sum[warp_id * 4 + 0] = local_sum_sd;
            s_warp_sum[warp_id * 4 + 1] = local_sum_qt;
            s_warp_sum[warp_id * 4 + 2] = local_sum_ep;
            s_warp_sum[warp_id * 4 + 3] = local_sum_dc;
        }
        __syncthreads();

        if (tid == 0) {
            // Warps 1-4 contribute (indices 1-4); warp 0 is always 0
            int64_t block_rev = s_warp_rev[1] + s_warp_rev[2] + s_warp_rev[3] + s_warp_rev[4];
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(block_rev));
            if (d_diag) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                          (unsigned long long)(s_warp_sum[4] + s_warp_sum[8] + s_warp_sum[12] + s_warp_sum[16]));
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                          (unsigned long long)(s_warp_sum[5] + s_warp_sum[9] + s_warp_sum[13] + s_warp_sum[17]));
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                          (unsigned long long)(s_warp_sum[6] + s_warp_sum[10] + s_warp_sum[14] + s_warp_sum[18]));
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                          (unsigned long long)(s_warp_sum[7] + s_warp_sum[11] + s_warp_sum[15] + s_warp_sum[19]));
            }
        }
        local_rev = 0;
        local_sum_sd = local_sum_qt = local_sum_ep = local_sum_dc = 0;
        __syncthreads();
    }

    // Flush profiling counters
    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->pf_submit_count),
                  (unsigned long long)blk_pf_submit);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->pf_hit_count),
                  (unsigned long long)blk_pf_hit);
    }
}

// ============================================================
// BAM Q6 kernel — pipelined I/O with N-way ring buffer
//
// Same warp-level structure as sync kernel, but uses io_multi
// slots to overlap NVMe I/O with decompress+eval compute.
//
// Design: "submit one ahead" — at most one new slot is submitted
// per outer loop iteration before processing the oldest slot.
// Entry indices are derived deterministically from
// (block_id, io_multi, slot_index), avoiding large per-thread arrays.
// Only ring_cid[MAX_IO_MULTI] is stored per thread.
// ============================================================

// Helper: submit I/O for one page, filling slot `s` in the ring buffer.
// Uses temporary local arrays (io_lba[7], io_entry[7]) that are
// discarded after the submit; not stored per-slot.
__forceinline__ __device__ void bam_q6_submit_slot(
    uint32_t s, uint64_t sd_pg,
    page_cache_d_t* pc, QueuePair* qp, BAMKernelMeta* meta,
    SlotMeta* slot_meta,
    uint16_t* out_cid,     // &ring_cid[s]
    uint32_t tid, uint32_t block_id, uint32_t io_multi,
    int* s_skip)           // shared flag
{
    const uint64_t npages = meta->field_npages;
    const uint32_t blocks_per_page = meta->blocks_per_page;
    const uint16_t* comp = meta->compression_method;

    const uint64_t* ps0 = meta->d_prefix_sums[0];
    const uint64_t* ps1 = meta->d_prefix_sums[1];
    const uint64_t* ps2 = meta->d_prefix_sums[2];
    const uint64_t* ps3 = meta->d_prefix_sums[3];

    uint64_t row_begin = ps0[sd_pg];
    uint64_t row_end   = ps0[sd_pg + 1];

    // Compute non-driving field page IDs and boundary flags
    const uint64_t* ps_arr[3] = { ps1, ps2, ps3 };
    uint64_t fi_pg[3];
    uint64_t fi_bound[3];
    int needs_boundary[3];

    for (int f = 0; f < 3; f++) {
        fi_pg[f] = find_page_for_row(ps_arr[f], npages, row_begin);
        fi_bound[f] = ps_arr[f][fi_pg[f] + 1];
        needs_boundary[f] = (fi_bound[f] < row_end && fi_pg[f] + 1 < npages) ? 1 : 0;
        if (!needs_boundary[f]) fi_bound[f] = row_end;
    }

    // Write slot metadata to shared memory
    if (tid == 0) {
        SlotMeta& sm = slot_meta[s];
        sm.sd_pg = sd_pg;
        sm.row_begin = row_begin;
        sm.row_end = row_end;
        for (int f = 0; f < 3; f++) {
            sm.fi_pg[f] = fi_pg[f];
            sm.fi_bound[f] = fi_bound[f];
            sm.needs_boundary[f] = needs_boundary[f];
        }
    }
    __syncwarp();

    // Pack I/O descriptors into temporary local arrays
    unsigned long long entry_base =
        (unsigned long long)(block_id * io_multi + s) * ENTRIES_PER_BLOCK;

    uint64_t io_lba[7];
    unsigned long long io_entry[7];
    uint32_t io_nblocks[7];  // per-IO block count (compressed = actual size)
    int n_ios = 0;

    // L_SHIPDATE
    if (comp[0] != 0) {
        io_lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[0][sd_pg] / 512;
        io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[0][sd_pg]);
    } else {
        uint64_t sd_page_id = meta->field_start_page_ids[0] + sd_pg;
        io_lba[n_ios] = meta->partition_start_lba + sd_page_id * blocks_per_page;
        io_nblocks[n_ios] = blocks_per_page;
    }
    io_entry[n_ios] = entry_base + 0;
    n_ios++;

    // Non-driving fields: primary pages
    for (int f = 0; f < 3; f++) {
        unsigned long long ea = entry_base + 1 + f * 2;
        if (comp[f + 1] != 0) {
            io_lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[f + 1][fi_pg[f]] / 512;
            io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f]]);
        } else {
            uint64_t pg_id = meta->field_start_page_ids[f + 1] + fi_pg[f];
            io_lba[n_ios] = meta->partition_start_lba + pg_id * blocks_per_page;
            io_nblocks[n_ios] = blocks_per_page;
        }
        io_entry[n_ios] = ea;
        n_ios++;
    }

    // Boundary pages
    for (int f = 0; f < 3; f++) {
        if (needs_boundary[f]) {
            unsigned long long eb = entry_base + 2 + f * 2;
            if (comp[f + 1] != 0) {
                io_lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[f + 1][fi_pg[f] + 1] / 512;
                io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f] + 1]);
            } else {
                uint64_t pg_id = meta->field_start_page_ids[f + 1] + fi_pg[f] + 1;
                io_lba[n_ios] = meta->partition_start_lba + pg_id * blocks_per_page;
                io_nblocks[n_ios] = blocks_per_page;
            }
            io_entry[n_ios] = eb;
            n_ios++;
        }
    }

    if (tid == 0) {
        slot_meta[s].n_ios = n_ios;
        uint32_t total_bytes = 0;
        for (int ii = 0; ii < n_ios; ii++) total_bytes += io_nblocks[ii] * 512;
        slot_meta[s].io_bytes = total_bytes;
    }
    __syncwarp();

    // Submit I/O: per-lane async (read only compressed size, not full page)
    *out_cid = 0;
    uint16_t sq_pos = 0;
    if ((int)tid < n_ios) {
        access_data_async(pc, qp, io_lba[tid], io_nblocks[tid],
                          io_entry[tid], NVM_IO_READ,
                          out_cid, &sq_pos);
    }
    __syncwarp();
}

__global__ void bam_q6_kernel_comp_io_multi(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;            // 0..31
    const uint32_t io_multi = meta->io_multi;

    // Use per-slot QueuePairs to avoid CQ interleaving deadlock.
    // Each slot gets its own QP so completions never intermix.
    const uint32_t n_qps = ctrls[0]->n_qps;
    QueuePair* d_qps = ctrls[0]->d_qps;

    const uint64_t npages = meta->field_npages;
    const uint64_t page_size = meta->page_size;

    const uint64_t* ps1 = meta->d_prefix_sums[1]; // L_QUANTITY
    const uint64_t* ps2 = meta->d_prefix_sums[2]; // L_EXTENDEDPRICE
    const uint64_t* ps3 = meta->d_prefix_sums[3]; // L_DISCOUNT

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;

    const uint16_t* comp = meta->compression_method;

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;

    __shared__ uint shared_bws[4];
    __shared__ uint shared_offs[4];
    __shared__ int s_skip;
    __shared__ uint32_t s_comp_blk[136];  // for fused decode_pfor_block_smem

    // Ring buffer slot metadata in shared memory
    __shared__ SlotMeta slot_meta[MAX_IO_MULTI];

    // Per-thread per-slot CID (only state that must persist across submit→poll)
    uint16_t ring_cid[MAX_IO_MULTI];

    // Per-block profiling accumulators
    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    // Ring buffer state
    uint64_t cursor = block_id;       // next page to submit (grid-stride)
    uint32_t ring_head = 0;           // oldest outstanding slot
    uint32_t ring_count = 0;          // number of outstanding slots

    // ────── Helper: try to fill one ring slot from cursor ──────
    // Returns true if a slot was filled, false if no more pages.
    // Advances cursor past zone-map-pruned and empty pages.
    // Each slot uses its own QueuePair to avoid CQ interleaving.
    #define TRY_FILL_ONE_SLOT() do {                                        \
        bool _filled = false;                                               \
        while (!_filled && cursor < npages) {                               \
            uint64_t _sd_pg = cursor;                                       \
            cursor += gridDim.x;                                            \
            /* Zone map pruning */                                          \
            if (tid == 0) {                                                 \
                s_skip = 0;                                                 \
                if (stats && _sd_pg < nstats) {                             \
                    int32_t _pmin = stats[_sd_pg * 2];                      \
                    int32_t _pmax = stats[_sd_pg * 2 + 1];                  \
                    if (_pmax < 19940101 || _pmin > 19950100) s_skip = 1;   \
                }                                                           \
            }                                                               \
            __syncwarp();                                                    \
            if (s_skip) continue;                                           \
            /* Skip empty pages */                                          \
            uint64_t _ps0b = meta->d_prefix_sums[0][_sd_pg];               \
            uint64_t _ps0e = meta->d_prefix_sums[0][_sd_pg + 1];           \
            if (_ps0e == _ps0b) continue;                                   \
            /* Found valid page: submit using per-slot QP */                \
            uint32_t _s = (ring_head + ring_count) % io_multi;              \
            QueuePair* _qp = d_qps                                         \
                + ((block_id * io_multi + _s) % n_qps);                     \
            bam_q6_submit_slot(_s, _sd_pg, pc, _qp, meta,                  \
                               slot_meta, &ring_cid[_s],                    \
                               tid, block_id, io_multi, &s_skip);           \
            ring_count++;                                                   \
            _filled = true;                                                 \
        }                                                                   \
    } while(0)

    // ────── Initial fill: submit first slot ──────
    TRY_FILL_ONE_SLOT();

    // ────── Main loop: prefetch + process ──────
    while (ring_count > 0) {
        // Prefetch: submit one more slot if room
        if (ring_count < io_multi) {
            TRY_FILL_ONE_SLOT();
        }

        // Process oldest slot: poll → decompress → eval
        uint32_t s = ring_head;
        SlotMeta& sm = slot_meta[s];

        blk_page_count++;

        // 1. Poll I/O completion (using same per-slot QP as submit)
        long long t0 = clock64();
        QueuePair* qp_s = d_qps + ((block_id * io_multi + s) % n_qps);
        if ((int)tid < sm.n_ios) {
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp_s->cq, ring_cid[s], &poll_loc, &poll_head);
            cq_dequeue(&qp_s->cq, cq_pos, &qp_s->sq);
            put_cid(&qp_s->sq, ring_cid[s]);
        }
        __syncwarp();

        long long t1 = clock64();
        blk_io_cycles += (uint64_t)(t1 - t0);
        blk_io_count += sm.n_ios;
        blk_io_bytes += sm.io_bytes;
        for (int bf = 0; bf < 3; bf++) blk_boundary_count += sm.needs_boundary[bf];

        // 2. Fused decompress + Q6 eval
        long long td0 = clock64();

        unsigned long long entry_base =
            (unsigned long long)(block_id * io_multi + s) * ENTRIES_PER_BLOCK;
        int32_t* decomp = meta->d_decomp_buf
            + (uint64_t)(block_id * io_multi + s) * ENTRIES_PER_BLOCK * elems_per_slot;

        bool all_comp = (comp[0] != 0 && comp[1] != 0 &&
                         comp[2] != 0 && comp[3] != 0);
        bool has_boundary = (sm.needs_boundary[0] || sm.needs_boundary[1] || sm.needs_boundary[2]);

        uint64_t nrows_this = sm.row_end - sm.row_begin;
        int64_t local_rev = 0;
        uint64_t local_sum_sd = 0, local_sum_qt = 0, local_sum_ep = 0, local_sum_dc = 0;

        if (all_comp && !has_boundary) {
            // ── Case A: all compressed, no boundary ──
            char* sub_page_ptr[4] = {
                base + (entry_base + 0) * page_size,  // SD
                base + (entry_base + 1) * page_size,  // QT
                base + (entry_base + 3) * page_size,  // EP
                base + (entry_base + 5) * page_size,  // DC
            };

            bam_pag_head* hdr0 = (bam_pag_head*)sub_page_ptr[0];
            uint32_t nalloc = hdr0->nalloc;
            uint32_t nblocks_pfor = hdr0->watermark / 128;

            uint32_t* blk_start[4];
            uint32_t* data_ptr_f[4];
            for (int f = 0; f < 4; f++) {
                blk_start[f] = (uint32_t*)(sub_page_ptr[f] + BAM_PAG_HDR_BYTES);
                data_ptr_f[f] = blk_start[f] + (nblocks_pfor + 1);
            }

            for (uint32_t b = 0; b < nblocks_pfor; b++) {
                int32_t sd_v[4], qt_v[4], ep_v[4], dc_v[4];

                decode_pfor_block_smem(
                    data_ptr_f[0] + blk_start[0][b],
                    blk_start[0][b + 1] - blk_start[0][b],
                    s_comp_blk, shared_bws, shared_offs, tid, sd_v);
                decode_pfor_block_smem(
                    data_ptr_f[1] + blk_start[1][b],
                    blk_start[1][b + 1] - blk_start[1][b],
                    s_comp_blk, shared_bws, shared_offs, tid, qt_v);
                decode_pfor_block_smem(
                    data_ptr_f[2] + blk_start[2][b],
                    blk_start[2][b + 1] - blk_start[2][b],
                    s_comp_blk, shared_bws, shared_offs, tid, ep_v);
                decode_pfor_block_smem(
                    data_ptr_f[3] + blk_start[3][b],
                    blk_start[3][b + 1] - blk_start[3][b],
                    s_comp_blk, shared_bws, shared_offs, tid, dc_v);

                for (uint32_t ki = 0; ki < 4; ki++) {
                    uint32_t idx = b * 128 + ki * 32 + tid;
                    if (idx < nalloc) {
                        local_sum_sd += (uint32_t)sd_v[ki];
                        local_sum_qt += (uint32_t)qt_v[ki];
                        local_sum_ep += (uint32_t)ep_v[ki];
                        local_sum_dc += (uint32_t)dc_v[ki];

                        if (sd_v[ki] >= 19940101 && sd_v[ki] < 19950101 &&
                            dc_v[ki] >= 5 && dc_v[ki] <= 7 &&
                            qt_v[ki] < 2400) {
                            local_rev += (int64_t)ep_v[ki] * dc_v[ki];
                        }
                    }
                }
            }
        } else {
            // ── Case B: fallback (boundary or mixed compression) ──
            int32_t* sd_decomp = decomp + 0 * elems_per_slot;
            decompress_page_warp(base + (entry_base + 0) * page_size, sd_decomp,
                                 comp[0], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + (entry_base + 1) * page_size,
                                 decomp + 1 * elems_per_slot,
                                 comp[1], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + (entry_base + 3) * page_size,
                                 decomp + 3 * elems_per_slot,
                                 comp[2], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + (entry_base + 5) * page_size,
                                 decomp + 5 * elems_per_slot,
                                 comp[3], shared_bws, shared_offs, tid, nullptr);

            if (sm.needs_boundary[0]) {
                decompress_page_warp(base + (entry_base + 2) * page_size,
                                     decomp + 2 * elems_per_slot,
                                     comp[1], shared_bws, shared_offs, tid, nullptr);
            }
            if (sm.needs_boundary[1]) {
                decompress_page_warp(base + (entry_base + 4) * page_size,
                                     decomp + 4 * elems_per_slot,
                                     comp[2], shared_bws, shared_offs, tid, nullptr);
            }
            if (sm.needs_boundary[2]) {
                decompress_page_warp(base + (entry_base + 6) * page_size,
                                     decomp + 6 * elems_per_slot,
                                     comp[3], shared_bws, shared_offs, tid, nullptr);
            }

            int32_t* qt_a = decomp + 1 * elems_per_slot;
            int32_t* qt_b = decomp + 2 * elems_per_slot;
            int32_t* ep_a = decomp + 3 * elems_per_slot;
            int32_t* ep_b = decomp + 4 * elems_per_slot;
            int32_t* dc_a = decomp + 5 * elems_per_slot;
            int32_t* dc_b = decomp + 6 * elems_per_slot;

            uint64_t qt_base = ps1[sm.fi_pg[0]];
            uint64_t ep_base = ps2[sm.fi_pg[1]];
            uint64_t dc_base = ps3[sm.fi_pg[2]];

            for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 32) {
                uint64_t gr = sm.row_begin + i;
                int32_t l_shipdate = sd_decomp[i];
                int32_t l_quantity = (gr < sm.fi_bound[0])
                    ? qt_a[gr - qt_base] : qt_b[gr - sm.fi_bound[0]];
                int32_t l_extendedprice = (gr < sm.fi_bound[1])
                    ? ep_a[gr - ep_base] : ep_b[gr - sm.fi_bound[1]];
                int32_t l_discount = (gr < sm.fi_bound[2])
                    ? dc_a[gr - dc_base] : dc_b[gr - sm.fi_bound[2]];

                local_sum_sd += (uint32_t)l_shipdate;
                local_sum_qt += (uint32_t)l_quantity;
                local_sum_ep += (uint32_t)l_extendedprice;
                local_sum_dc += (uint32_t)l_discount;

                if (l_shipdate >= 19940101 && l_shipdate < 19950101 &&
                    l_discount >= 5 && l_discount <= 7 &&
                    l_quantity < 2400) {
                    local_rev += (int64_t)l_extendedprice * l_discount;
                }
            }
        }

        long long td1 = clock64();
        // Fused decomp+eval: report all cycles under decomp (eval = 0).
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        // 4. Warp-level reduction + atomicAdd
        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
            local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
            local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
            local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
            local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
        }

        if (tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(local_rev));
            if (d_diag) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                          (unsigned long long)local_sum_sd);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                          (unsigned long long)local_sum_qt);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                          (unsigned long long)local_sum_ep);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                          (unsigned long long)local_sum_dc);
            }
        }

        // Advance ring
        ring_head = (ring_head + 1) % io_multi;
        ring_count--;
    }

    #undef TRY_FILL_ONE_SLOT

    // === Flush per-block profiling counters to global memory ===
    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
    }
}

// ============================================================
// BAM Q6 kernel — sync, variable L_SHIPDATE predicate
//
// Identical to bam_q6_kernel_comp_sync except zone map pruning
// and Q6 eval use meta->sd_low / meta->sd_high instead of
// hardcoded 19940101 / 19950101.
// ============================================================
__global__ void bam_q6_kernel_comp_sync_vardate(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    QueuePair* qp = ctrls[0]->d_qps + (block_id % ctrls[0]->n_qps);

    const uint64_t npages = meta->field_npages;
    const uint32_t blocks_per_page = meta->blocks_per_page;
    const uint64_t page_size = meta->page_size;

    const uint64_t* ps0 = meta->d_prefix_sums[0];
    const uint64_t* ps1 = meta->d_prefix_sums[1];
    const uint64_t* ps2 = meta->d_prefix_sums[2];
    const uint64_t* ps3 = meta->d_prefix_sums[3];

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;

    const uint16_t* comp = meta->compression_method;

    const int32_t vd_low  = meta->sd_low;
    const int32_t vd_high = meta->sd_high;

    __shared__ uint shared_bws[4];
    __shared__ uint shared_offs[4];
    __shared__ int s_skip;
    __shared__ uint32_t s_comp_blk[136];  // for fused decode_pfor_block_smem

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;
    int32_t* decomp_base = meta->d_decomp_buf
        + (uint64_t)block_id * ENTRIES_PER_BLOCK * elems_per_slot;

    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    for (uint64_t sd_pg = block_id; sd_pg < npages; sd_pg += gridDim.x) {
        // === 0. Zone map pruning (variable date) ===
        if (tid == 0) {
            s_skip = 0;
            if (stats && sd_pg < nstats) {
                int32_t page_min = stats[sd_pg * 2];
                int32_t page_max = stats[sd_pg * 2 + 1];
                if (page_max < vd_low || page_min > (vd_high - 1)) {
                    s_skip = 1;
                }
            }
        }
        __syncwarp();
        if (s_skip) continue;

        blk_page_count++;

        uint64_t row_begin = ps0[sd_pg];
        uint64_t row_end   = ps0[sd_pg + 1];
        uint64_t nrows_this = row_end - row_begin;
        if (nrows_this == 0) continue;

        const uint64_t* ps_arr[3] = { ps1, ps2, ps3 };
        uint64_t fi_pg[3];
        uint64_t fi_bound[3];
        bool needs_boundary[3];

        for (int f = 0; f < 3; f++) {
            fi_pg[f] = find_page_for_row(ps_arr[f], npages, row_begin);
            fi_bound[f] = ps_arr[f][fi_pg[f] + 1];
            needs_boundary[f] = (fi_bound[f] < row_end && fi_pg[f] + 1 < npages);
            if (!needs_boundary[f]) fi_bound[f] = row_end;
        }

        uint64_t io_lba[7];
        unsigned long long io_entry[7];
        uint32_t io_nblocks[7];
        int n_ios = 0;

        if (comp[0] != 0) {
            io_lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[0][sd_pg] / 512;
            io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[0][sd_pg]);
        } else {
            uint64_t sd_page_id = meta->field_start_page_ids[0] + sd_pg;
            io_lba[n_ios] = meta->partition_start_lba + sd_page_id * blocks_per_page;
            io_nblocks[n_ios] = blocks_per_page;
        }
        io_entry[n_ios] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 0;
        n_ios++;

        unsigned long long fi_entry_a[3], fi_entry_b[3];
        for (int f = 0; f < 3; f++) {
            fi_entry_a[f] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 1 + f * 2;
            fi_entry_b[f] = fi_entry_a[f] + 1;

            if (comp[f + 1] != 0) {
                io_lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[f + 1][fi_pg[f]] / 512;
                io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f]]);
            } else {
                uint64_t pg_id = meta->field_start_page_ids[f + 1] + fi_pg[f];
                io_lba[n_ios] = meta->partition_start_lba + pg_id * blocks_per_page;
                io_nblocks[n_ios] = blocks_per_page;
            }
            io_entry[n_ios] = fi_entry_a[f];
            n_ios++;
        }

        int boundary_this_page = 0;
        for (int f = 0; f < 3; f++) {
            if (needs_boundary[f]) {
                if (comp[f + 1] != 0) {
                    io_lba[n_ios] = meta->partition_start_lba + meta->d_comp_offsets[f + 1][fi_pg[f] + 1] / 512;
                    io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f] + 1]);
                } else {
                    uint64_t pg_id = meta->field_start_page_ids[f + 1] + fi_pg[f] + 1;
                    io_lba[n_ios] = meta->partition_start_lba + pg_id * blocks_per_page;
                    io_nblocks[n_ios] = blocks_per_page;
                }
                io_entry[n_ios] = fi_entry_b[f];
                n_ios++;
                boundary_this_page++;
            }
        }
        blk_boundary_count += boundary_this_page;

        long long t0 = clock64();

        uint16_t my_cid = 0;
        uint16_t my_sq_pos = 0;

        if ((int)tid < n_ios) {
            access_data_async(pc, qp, io_lba[tid], io_nblocks[tid],
                              io_entry[tid], NVM_IO_READ, &my_cid, &my_sq_pos);
        }
        __syncwarp();

        if ((int)tid < n_ios) {
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, my_cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, my_cid);
        }
        __syncwarp();

        long long t1 = clock64();
        blk_io_cycles += (uint64_t)(t1 - t0);
        blk_io_count += n_ios;
        for (int ii = 0; ii < n_ios; ii++) blk_io_bytes += (uint64_t)io_nblocks[ii] * 512;

        // Fused decompress + Q6 eval (variable date)
        long long td0 = clock64();

        bool all_comp = (comp[0] != 0 && comp[1] != 0 &&
                         comp[2] != 0 && comp[3] != 0);
        bool has_boundary = (needs_boundary[0] || needs_boundary[1] || needs_boundary[2]);

        int64_t local_rev = 0;
        uint64_t local_sum_sd = 0, local_sum_qt = 0, local_sum_ep = 0, local_sum_dc = 0;

        if (all_comp && !has_boundary) {
            // ── Case A: all compressed, no boundary ──
            char* sub_page_ptr[4] = {
                base + io_entry[0]   * page_size,  // SD
                base + fi_entry_a[0] * page_size,  // QT
                base + fi_entry_a[1] * page_size,  // EP
                base + fi_entry_a[2] * page_size,  // DC
            };

            bam_pag_head* hdr0 = (bam_pag_head*)sub_page_ptr[0];
            uint32_t nalloc = hdr0->nalloc;
            uint32_t nblocks_pfor = hdr0->watermark / 128;

            uint32_t* blk_start[4];
            uint32_t* data_ptr_f[4];
            for (int f = 0; f < 4; f++) {
                blk_start[f] = (uint32_t*)(sub_page_ptr[f] + BAM_PAG_HDR_BYTES);
                data_ptr_f[f] = blk_start[f] + (nblocks_pfor + 1);
            }

            for (uint32_t b = 0; b < nblocks_pfor; b++) {
                int32_t sd_v[4], qt_v[4], ep_v[4], dc_v[4];

                decode_pfor_block_smem(
                    data_ptr_f[0] + blk_start[0][b],
                    blk_start[0][b + 1] - blk_start[0][b],
                    s_comp_blk, shared_bws, shared_offs, tid, sd_v);
                decode_pfor_block_smem(
                    data_ptr_f[1] + blk_start[1][b],
                    blk_start[1][b + 1] - blk_start[1][b],
                    s_comp_blk, shared_bws, shared_offs, tid, qt_v);
                decode_pfor_block_smem(
                    data_ptr_f[2] + blk_start[2][b],
                    blk_start[2][b + 1] - blk_start[2][b],
                    s_comp_blk, shared_bws, shared_offs, tid, ep_v);
                decode_pfor_block_smem(
                    data_ptr_f[3] + blk_start[3][b],
                    blk_start[3][b + 1] - blk_start[3][b],
                    s_comp_blk, shared_bws, shared_offs, tid, dc_v);

                for (uint32_t ki = 0; ki < 4; ki++) {
                    uint32_t idx = b * 128 + ki * 32 + tid;
                    if (idx < nalloc) {
                        local_sum_sd += (uint32_t)sd_v[ki];
                        local_sum_qt += (uint32_t)qt_v[ki];
                        local_sum_ep += (uint32_t)ep_v[ki];
                        local_sum_dc += (uint32_t)dc_v[ki];

                        if (sd_v[ki] >= vd_low && sd_v[ki] < vd_high &&
                            dc_v[ki] >= 5 && dc_v[ki] <= 7 &&
                            qt_v[ki] < 2400) {
                            local_rev += (int64_t)ep_v[ki] * dc_v[ki];
                        }
                    }
                }
            }
        } else {
            // ── Case B: fallback (boundary or mixed compression) ──
            int32_t* sd_decomp = decomp_base + 0 * elems_per_slot;
            decompress_page_warp(base + io_entry[0] * page_size, sd_decomp,
                                 comp[0], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + fi_entry_a[0] * page_size,
                                 decomp_base + 1 * elems_per_slot,
                                 comp[1], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + fi_entry_a[1] * page_size,
                                 decomp_base + 3 * elems_per_slot,
                                 comp[2], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + fi_entry_a[2] * page_size,
                                 decomp_base + 5 * elems_per_slot,
                                 comp[3], shared_bws, shared_offs, tid, nullptr);

            if (needs_boundary[0]) {
                decompress_page_warp(base + fi_entry_b[0] * page_size,
                                     decomp_base + 2 * elems_per_slot,
                                     comp[1], shared_bws, shared_offs, tid, nullptr);
            }
            if (needs_boundary[1]) {
                decompress_page_warp(base + fi_entry_b[1] * page_size,
                                     decomp_base + 4 * elems_per_slot,
                                     comp[2], shared_bws, shared_offs, tid, nullptr);
            }
            if (needs_boundary[2]) {
                decompress_page_warp(base + fi_entry_b[2] * page_size,
                                     decomp_base + 6 * elems_per_slot,
                                     comp[3], shared_bws, shared_offs, tid, nullptr);
            }

            int32_t* qt_a = decomp_base + 1 * elems_per_slot;
            int32_t* qt_b = decomp_base + 2 * elems_per_slot;
            int32_t* ep_a = decomp_base + 3 * elems_per_slot;
            int32_t* ep_b = decomp_base + 4 * elems_per_slot;
            int32_t* dc_a = decomp_base + 5 * elems_per_slot;
            int32_t* dc_b = decomp_base + 6 * elems_per_slot;

            uint64_t qt_base = ps1[fi_pg[0]];
            uint64_t ep_base = ps2[fi_pg[1]];
            uint64_t dc_base = ps3[fi_pg[2]];

            for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 32) {
                uint64_t gr = row_begin + i;
                int32_t l_shipdate = sd_decomp[i];
                int32_t l_quantity = (gr < fi_bound[0])
                    ? qt_a[gr - qt_base] : qt_b[gr - fi_bound[0]];
                int32_t l_extendedprice = (gr < fi_bound[1])
                    ? ep_a[gr - ep_base] : ep_b[gr - fi_bound[1]];
                int32_t l_discount = (gr < fi_bound[2])
                    ? dc_a[gr - dc_base] : dc_b[gr - fi_bound[2]];

                local_sum_sd += (uint32_t)l_shipdate;
                local_sum_qt += (uint32_t)l_quantity;
                local_sum_ep += (uint32_t)l_extendedprice;
                local_sum_dc += (uint32_t)l_discount;

                if (l_shipdate >= vd_low && l_shipdate < vd_high &&
                    l_discount >= 5 && l_discount <= 7 &&
                    l_quantity < 2400) {
                    local_rev += (int64_t)l_extendedprice * l_discount;
                }
            }
        }

        long long td1 = clock64();
        // Fused decomp+eval: report all cycles under decomp (eval = 0).
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
            local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
            local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
            local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
            local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
        }

        if (tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(local_rev));
            if (d_diag) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                          (unsigned long long)local_sum_sd);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                          (unsigned long long)local_sum_qt);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                          (unsigned long long)local_sum_ep);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                          (unsigned long long)local_sum_dc);
            }
        }
    }

    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
    }
}

// ============================================================
// BAM Q6 kernel — pipelined I/O, variable L_SHIPDATE predicate
//
// Identical to bam_q6_kernel_comp_io_multi except zone map pruning
// and Q6 eval use meta->sd_low / meta->sd_high instead of
// hardcoded 19940101 / 19950101.
// ============================================================
__global__ void bam_q6_kernel_comp_io_multi_vardate(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t io_multi = meta->io_multi;

    const uint64_t npages = meta->field_npages;
    const uint64_t page_size = meta->page_size;

    Controller* ctrl = ctrls[0];
    QueuePair* d_qps = ctrl->d_qps;
    uint32_t n_qps = ctrl->n_qps;

    const uint64_t* ps0 = meta->d_prefix_sums[0];
    const uint64_t* ps1 = meta->d_prefix_sums[1];
    const uint64_t* ps2 = meta->d_prefix_sums[2];
    const uint64_t* ps3 = meta->d_prefix_sums[3];

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;

    const uint16_t* comp = meta->compression_method;

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;

    const int32_t vd_low  = meta->sd_low;
    const int32_t vd_high = meta->sd_high;

    __shared__ uint shared_bws[4];
    __shared__ uint shared_offs[4];
    __shared__ int s_skip;
    __shared__ uint32_t s_comp_blk[136];  // for fused decode_pfor_block_smem

    __shared__ SlotMeta slot_meta[MAX_IO_MULTI];

    uint16_t ring_cid[MAX_IO_MULTI];

    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    uint64_t cursor = block_id;
    uint32_t ring_head = 0;
    uint32_t ring_count = 0;

    // Helper: try to fill one ring slot (variable date zone map)
    #define TRY_FILL_ONE_SLOT_VD() do {                                     \
        bool _filled = false;                                               \
        while (!_filled && cursor < npages) {                               \
            uint64_t _sd_pg = cursor;                                       \
            cursor += gridDim.x;                                            \
            if (tid == 0) {                                                 \
                s_skip = 0;                                                 \
                if (stats && _sd_pg < nstats) {                             \
                    int32_t _pmin = stats[_sd_pg * 2];                      \
                    int32_t _pmax = stats[_sd_pg * 2 + 1];                  \
                    if (_pmax < vd_low || _pmin > (vd_high - 1)) s_skip = 1;\
                }                                                           \
            }                                                               \
            __syncwarp();                                                    \
            if (s_skip) continue;                                           \
            uint64_t _ps0b = meta->d_prefix_sums[0][_sd_pg];               \
            uint64_t _ps0e = meta->d_prefix_sums[0][_sd_pg + 1];           \
            if (_ps0e == _ps0b) continue;                                   \
            uint32_t _s = (ring_head + ring_count) % io_multi;              \
            QueuePair* _qp = d_qps                                         \
                + ((block_id * io_multi + _s) % n_qps);                     \
            bam_q6_submit_slot(_s, _sd_pg, pc, _qp, meta,                  \
                               slot_meta, &ring_cid[_s],                    \
                               tid, block_id, io_multi, &s_skip);           \
            ring_count++;                                                   \
            _filled = true;                                                 \
        }                                                                   \
    } while(0)

    TRY_FILL_ONE_SLOT_VD();

    while (ring_count > 0) {
        if (ring_count < io_multi) {
            TRY_FILL_ONE_SLOT_VD();
        }

        uint32_t s = ring_head;
        SlotMeta& sm = slot_meta[s];

        blk_page_count++;

        long long t0 = clock64();
        QueuePair* qp_s = d_qps + ((block_id * io_multi + s) % n_qps);
        if ((int)tid < sm.n_ios) {
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp_s->cq, ring_cid[s], &poll_loc, &poll_head);
            cq_dequeue(&qp_s->cq, cq_pos, &qp_s->sq);
            put_cid(&qp_s->sq, ring_cid[s]);
        }
        __syncwarp();

        long long t1 = clock64();
        blk_io_cycles += (uint64_t)(t1 - t0);
        blk_io_count += sm.n_ios;
        blk_io_bytes += sm.io_bytes;
        for (int bf = 0; bf < 3; bf++) blk_boundary_count += sm.needs_boundary[bf];

        // Fused decompress + Q6 eval (variable date)
        long long td0 = clock64();

        unsigned long long entry_base =
            (unsigned long long)(block_id * io_multi + s) * ENTRIES_PER_BLOCK;
        int32_t* decomp = meta->d_decomp_buf
            + (uint64_t)(block_id * io_multi + s) * ENTRIES_PER_BLOCK * elems_per_slot;

        bool all_comp = (comp[0] != 0 && comp[1] != 0 &&
                         comp[2] != 0 && comp[3] != 0);
        bool has_boundary = (sm.needs_boundary[0] || sm.needs_boundary[1] || sm.needs_boundary[2]);

        uint64_t nrows_this = sm.row_end - sm.row_begin;
        int64_t local_rev = 0;
        uint64_t local_sum_sd = 0, local_sum_qt = 0, local_sum_ep = 0, local_sum_dc = 0;

        if (all_comp && !has_boundary) {
            // ── Case A: all compressed, no boundary ──
            char* sub_page_ptr[4] = {
                base + (entry_base + 0) * page_size,  // SD
                base + (entry_base + 1) * page_size,  // QT
                base + (entry_base + 3) * page_size,  // EP
                base + (entry_base + 5) * page_size,  // DC
            };

            bam_pag_head* hdr0 = (bam_pag_head*)sub_page_ptr[0];
            uint32_t nalloc = hdr0->nalloc;
            uint32_t nblocks_pfor = hdr0->watermark / 128;

            uint32_t* blk_start[4];
            uint32_t* data_ptr_f[4];
            for (int f = 0; f < 4; f++) {
                blk_start[f] = (uint32_t*)(sub_page_ptr[f] + BAM_PAG_HDR_BYTES);
                data_ptr_f[f] = blk_start[f] + (nblocks_pfor + 1);
            }

            for (uint32_t b = 0; b < nblocks_pfor; b++) {
                int32_t sd_v[4], qt_v[4], ep_v[4], dc_v[4];

                decode_pfor_block_smem(
                    data_ptr_f[0] + blk_start[0][b],
                    blk_start[0][b + 1] - blk_start[0][b],
                    s_comp_blk, shared_bws, shared_offs, tid, sd_v);
                decode_pfor_block_smem(
                    data_ptr_f[1] + blk_start[1][b],
                    blk_start[1][b + 1] - blk_start[1][b],
                    s_comp_blk, shared_bws, shared_offs, tid, qt_v);
                decode_pfor_block_smem(
                    data_ptr_f[2] + blk_start[2][b],
                    blk_start[2][b + 1] - blk_start[2][b],
                    s_comp_blk, shared_bws, shared_offs, tid, ep_v);
                decode_pfor_block_smem(
                    data_ptr_f[3] + blk_start[3][b],
                    blk_start[3][b + 1] - blk_start[3][b],
                    s_comp_blk, shared_bws, shared_offs, tid, dc_v);

                for (uint32_t ki = 0; ki < 4; ki++) {
                    uint32_t idx = b * 128 + ki * 32 + tid;
                    if (idx < nalloc) {
                        local_sum_sd += (uint32_t)sd_v[ki];
                        local_sum_qt += (uint32_t)qt_v[ki];
                        local_sum_ep += (uint32_t)ep_v[ki];
                        local_sum_dc += (uint32_t)dc_v[ki];

                        if (sd_v[ki] >= vd_low && sd_v[ki] < vd_high &&
                            dc_v[ki] >= 5 && dc_v[ki] <= 7 &&
                            qt_v[ki] < 2400) {
                            local_rev += (int64_t)ep_v[ki] * dc_v[ki];
                        }
                    }
                }
            }
        } else {
            // ── Case B: fallback (boundary or mixed compression) ──
            int32_t* sd_decomp = decomp + 0 * elems_per_slot;
            decompress_page_warp(base + (entry_base + 0) * page_size, sd_decomp,
                                 comp[0], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + (entry_base + 1) * page_size,
                                 decomp + 1 * elems_per_slot,
                                 comp[1], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + (entry_base + 3) * page_size,
                                 decomp + 3 * elems_per_slot,
                                 comp[2], shared_bws, shared_offs, tid, nullptr);
            decompress_page_warp(base + (entry_base + 5) * page_size,
                                 decomp + 5 * elems_per_slot,
                                 comp[3], shared_bws, shared_offs, tid, nullptr);

            if (sm.needs_boundary[0]) {
                decompress_page_warp(base + (entry_base + 2) * page_size,
                                     decomp + 2 * elems_per_slot,
                                     comp[1], shared_bws, shared_offs, tid, nullptr);
            }
            if (sm.needs_boundary[1]) {
                decompress_page_warp(base + (entry_base + 4) * page_size,
                                     decomp + 4 * elems_per_slot,
                                     comp[2], shared_bws, shared_offs, tid, nullptr);
            }
            if (sm.needs_boundary[2]) {
                decompress_page_warp(base + (entry_base + 6) * page_size,
                                     decomp + 6 * elems_per_slot,
                                     comp[3], shared_bws, shared_offs, tid, nullptr);
            }

            int32_t* qt_a = decomp + 1 * elems_per_slot;
            int32_t* qt_b = decomp + 2 * elems_per_slot;
            int32_t* ep_a = decomp + 3 * elems_per_slot;
            int32_t* ep_b = decomp + 4 * elems_per_slot;
            int32_t* dc_a = decomp + 5 * elems_per_slot;
            int32_t* dc_b = decomp + 6 * elems_per_slot;

            uint64_t qt_base = ps1[sm.fi_pg[0]];
            uint64_t ep_base = ps2[sm.fi_pg[1]];
            uint64_t dc_base = ps3[sm.fi_pg[2]];

            for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 32) {
                uint64_t gr = sm.row_begin + i;
                int32_t l_shipdate = sd_decomp[i];
                int32_t l_quantity = (gr < sm.fi_bound[0])
                    ? qt_a[gr - qt_base] : qt_b[gr - sm.fi_bound[0]];
                int32_t l_extendedprice = (gr < sm.fi_bound[1])
                    ? ep_a[gr - ep_base] : ep_b[gr - sm.fi_bound[1]];
                int32_t l_discount = (gr < sm.fi_bound[2])
                    ? dc_a[gr - dc_base] : dc_b[gr - sm.fi_bound[2]];

                local_sum_sd += (uint32_t)l_shipdate;
                local_sum_qt += (uint32_t)l_quantity;
                local_sum_ep += (uint32_t)l_extendedprice;
                local_sum_dc += (uint32_t)l_discount;

                if (l_shipdate >= vd_low && l_shipdate < vd_high &&
                    l_discount >= 5 && l_discount <= 7 &&
                    l_quantity < 2400) {
                    local_rev += (int64_t)l_extendedprice * l_discount;
                }
            }
        }

        long long td1 = clock64();
        // Fused decomp+eval: report all cycles under decomp (eval = 0).
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
            local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
            local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
            local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
            local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
        }

        if (tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(local_rev));
            if (d_diag) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                          (unsigned long long)local_sum_sd);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                          (unsigned long long)local_sum_qt);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                          (unsigned long long)local_sum_ep);
                atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                          (unsigned long long)local_sum_dc);
            }
        }

        ring_head = (ring_head + 1) % io_multi;
        ring_count--;
    }

    #undef TRY_FILL_ONE_SLOT_VD

    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
    }
}

// ============================================================
// Host wrapper — uses an externally-managed BAM Controller
// ============================================================
BAMRunResult bam_q6_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl_handle) {
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    uint32_t num_blocks = params.num_blocks;
    const uint64_t page_size = params.page_size;
    const uint32_t io_multi = std::max(1u, std::min(params.io_multiplicity, (uint32_t)MAX_IO_MULTI));

    // Per-slot QPs: within a block, each io_multi slot must map to a distinct QP
    // to avoid CQ interleaving deadlock (sequential slot processing within a warp).
    // Cross-block QP sharing is safe: different blocks run concurrently on separate
    // SMs and independently dequeue their completions.
    // Formula: qp_index = (block_id * io_multi + slot) % n_qps
    // Within-block uniqueness is guaranteed when io_multi <= n_qps (always true).
    uint32_t n_qps_avail = params.num_queues;
    if (io_multi > n_qps_avail) {
        fprintf(stderr, "[bam_q6] io_multi=%u exceeds n_qps=%u, capping\n",
                io_multi, n_qps_avail);
    }

    // ── Page cache: 7 entries per block × io_multi slots ──
    const uint64_t n_pc_pages = (uint64_t)num_blocks * ENTRIES_PER_BLOCK * io_multi;
    const uint64_t max_range = 64;

    // Check GPU memory before allocating
    size_t gpu_free = 0, gpu_total = 0;
    BAM_CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));
    fprintf(stderr, "[bam_q6] GPU memory: %.1f GB free / %.1f GB total\n",
            gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "[bam_q6] io_multi=%u, num_blocks=%u, pc_pages=%lu\n", io_multi, num_blocks, (unsigned long)n_pc_pages);

    page_cache_t h_pc(page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);
    page_cache_d_t* d_pc = h_pc.d_pc_ptr;

    // ── Copy prefix sum arrays to GPU ──
    const uint64_t ps_len = params.field_npages + 1;
    const size_t ps_bytes = ps_len * sizeof(uint64_t);
    uint64_t* d_prefix_sums[4];
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaMalloc(&d_prefix_sums[fi], ps_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_prefix_sums[fi], params.h_prefix_sums[fi],
                                   ps_bytes, cudaMemcpyHostToDevice));
    }

    // ── Copy zone map stats to GPU (if provided) ──
    int32_t* d_shipdate_stats = nullptr;
    if (params.h_shipdate_stats && params.nstats > 0) {
        const size_t stats_bytes = params.nstats * 2 * sizeof(int32_t);
        BAM_CUDA_CHECK(cudaMalloc(&d_shipdate_stats, stats_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_shipdate_stats, params.h_shipdate_stats,
                                   stats_bytes, cudaMemcpyHostToDevice));
    }

    // ── Copy compression metadata to GPU ──
    uint32_t* d_comp_sizes[4] = {};
    uint64_t* d_comp_offsets[4] = {};
    for (int fi = 0; fi < 4; fi++) {
        if (params.compression_method[fi] != 0 &&
            params.h_compressed_page_sizes[fi] && params.h_compressed_offsets[fi]) {
            size_t sz_sizes = params.field_npages * sizeof(uint32_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes[fi], sz_sizes));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes[fi], params.h_compressed_page_sizes[fi],
                                       sz_sizes, cudaMemcpyHostToDevice));

            size_t sz_offsets = params.field_npages * sizeof(uint64_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], sz_offsets));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], params.h_compressed_offsets[fi],
                                       sz_offsets, cudaMemcpyHostToDevice));
        }
    }

    // ── Decompression output buffer (scaled by io_multi) ──
    const uint32_t elems_per_slot = params.decomp_elems_per_slot;
    int32_t* d_decomp_buf = nullptr;
    size_t decomp_buf_size = (size_t)num_blocks * io_multi * ENTRIES_PER_BLOCK
                           * elems_per_slot * sizeof(int32_t);
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp_buf, decomp_buf_size));

    // ── GPU metadata ──
    BAMKernelMeta h_meta;
    for (int fi = 0; fi < 4; fi++) {
        h_meta.field_start_page_ids[fi] = params.field_start_page_ids[fi];
        h_meta.d_prefix_sums[fi] = d_prefix_sums[fi];
        h_meta.compression_method[fi] = params.compression_method[fi];
        h_meta.d_comp_sizes[fi] = d_comp_sizes[fi];
        h_meta.d_comp_offsets[fi] = d_comp_offsets[fi];
    }
    h_meta.field_npages = params.field_npages;
    h_meta.page_size = params.page_size;
    h_meta.blocks_per_page = params.blocks_per_page;
    h_meta.partition_start_lba = params.partition_start_lba;
    h_meta.n_devices = params.n_devices > 0 ? params.n_devices : 1;
    for (uint32_t d = 0; d < h_meta.n_devices; d++)
        h_meta.partition_start_lbas[d] = params.partition_start_lbas[d];
    h_meta.num_blocks = num_blocks;
    h_meta.nrows = params.nrows;
    h_meta.d_shipdate_stats = d_shipdate_stats;
    h_meta.nstats = params.nstats;
    h_meta.d_decomp_buf = d_decomp_buf;
    h_meta.decomp_elems_per_slot = elems_per_slot;
    h_meta.io_multi = io_multi;
    h_meta.sd_low = params.sd_low;
    h_meta.sd_high = params.sd_high;

    BAMKernelMeta* d_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_meta, sizeof(BAMKernelMeta)));
    BAM_CUDA_CHECK(cudaMemcpy(d_meta, &h_meta, sizeof(BAMKernelMeta),
                               cudaMemcpyHostToDevice));

    int64_t* d_revenue;
    BAM_CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    // ── Diagnostic counters ──
    BAMDiagCounters* d_diag;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, sizeof(BAMDiagCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, sizeof(BAMDiagCounters)));

    // ── Profiling counters ──
    BAMPerfCounters* d_perf;
    BAM_CUDA_CHECK(cudaMalloc(&d_perf, sizeof(BAMPerfCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_perf, 0, sizeof(BAMPerfCounters)));

    // ── Increase per-thread stack size for deeply-inlined BAM functions ──
    size_t old_stack_size = 0;
    cudaDeviceGetLimit(&old_stack_size, cudaLimitStackSize);
    fprintf(stderr, "[bam_q6] Default stack size: %zu bytes\n", old_stack_size);
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    // ── Kernel launch: 1 block = 32 threads (1 warp) ──
    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    const uint32_t threads_per_block = 32;

    fprintf(stderr, "[bam_q6] Launching %u blocks x %u threads, io_multi=%u, decomp_buf=%zu MB\n",
            num_blocks, threads_per_block, io_multi, decomp_buf_size / (1024 * 1024));

    const bool use_vardate = (params.sd_low != 19940101 || params.sd_high != 19950101);
    if (io_multi <= 1) {
        if (use_vardate) {
            bam_q6_kernel_comp_sync_vardate<<<num_blocks, threads_per_block, 0, stream>>>(
                h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
        } else {
            bam_q6_kernel_comp_sync<<<num_blocks, threads_per_block, 0, stream>>>(
                h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
        }
    } else {
        if (use_vardate) {
            bam_q6_kernel_comp_io_multi_vardate<<<num_blocks, threads_per_block, 0, stream>>>(
                h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
        } else {
            bam_q6_kernel_comp_io_multi<<<num_blocks, threads_per_block, 0, stream>>>(
                h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
        }
    }
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Result ──
    int64_t h_revenue = 0;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_revenue, d_revenue, sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, stream));

    BAMDiagCounters h_diag;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_diag, d_diag, sizeof(BAMDiagCounters),
                                    cudaMemcpyDeviceToHost, stream));

    BAMPerfCounters h_perf;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_perf, d_perf, sizeof(BAMPerfCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr, "[diag] sum_shipdate=%lu  sum_quantity=%lu\n",
            h_diag.sum_shipdate, h_diag.sum_quantity);
    fprintf(stderr, "[diag] sum_extprice=%lu  sum_discount=%lu\n",
            h_diag.sum_extprice, h_diag.sum_discount);

    // ── Print profiling breakdown ──
    // Get GPU clock rate to convert cycles to time
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, params.cuda_device);
    double clock_ghz = clock_khz / 1e6;  // GHz

    uint64_t total_cycles = h_perf.io_cycles + h_perf.decomp_cycles + h_perf.eval_cycles;
    fprintf(stderr, "\n[perf] GPU clock: %.3f GHz\n", clock_ghz);
    fprintf(stderr, "[perf] pages_processed: %lu  io_calls: %lu  io_bytes: %lu (%.2f MiB)\n",
            h_perf.page_count, h_perf.io_count,
            h_perf.io_bytes, h_perf.io_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[perf] boundary_page_reads: %lu (%.1f%% of pages)\n",
            h_perf.boundary_count,
            h_perf.page_count > 0 ? 100.0 * h_perf.boundary_count / h_perf.page_count : 0.0);
    fprintf(stderr, "[perf] Accumulated cycles across %u blocks (thread 0):\n", num_blocks);
    fprintf(stderr, "[perf]   io:      %12lu cycles  (%5.1f%%)\n",
            h_perf.io_cycles, 100.0 * h_perf.io_cycles / total_cycles);
    fprintf(stderr, "[perf]   decomp:  %12lu cycles  (%5.1f%%)\n",
            h_perf.decomp_cycles, 100.0 * h_perf.decomp_cycles / total_cycles);
    fprintf(stderr, "[perf]   eval:    %12lu cycles  (%5.1f%%)\n",
            h_perf.eval_cycles, 100.0 * h_perf.eval_cycles / total_cycles);
    fprintf(stderr, "[perf]   total:   %12lu cycles\n", total_cycles);
    // Average per-IO latency
    if (h_perf.io_count > 0) {
        double avg_io_us = (double)h_perf.io_cycles / h_perf.io_count / (clock_ghz * 1e3);
        fprintf(stderr, "[perf]   avg_io_latency: %.1f us/call  (%lu calls)\n",
                avg_io_us, h_perf.io_count);
        double avg_io_kb = (double)h_perf.io_bytes / h_perf.io_count / 1024.0;
        fprintf(stderr, "[perf]   avg_io_size: %.1f KiB/call\n", avg_io_kb);
    }

    BAM_CUDA_CHECK(cudaStreamDestroy(stream));

    // ── Cleanup ──
    BAM_CUDA_CHECK(cudaFree(d_meta));
    BAM_CUDA_CHECK(cudaFree(d_revenue));
    BAM_CUDA_CHECK(cudaFree(d_diag));
    BAM_CUDA_CHECK(cudaFree(d_perf));
    BAM_CUDA_CHECK(cudaFree(d_decomp_buf));
    if (d_shipdate_stats) BAM_CUDA_CHECK(cudaFree(d_shipdate_stats));
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaFree(d_prefix_sums[fi]));
        if (d_comp_sizes[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_sizes[fi]));
        if (d_comp_offsets[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_offsets[fi]));
    }
    // page_cache_t destructor handles page cache cleanup
    // Controller is NOT deleted here — caller owns it

    return BAMRunResult{h_revenue, h_perf.io_count, h_perf.io_bytes};
}

// ============================================================
// bam_q6_prescan_run: prescan + execution two-phase approach.
// Phase 1: prescan kernel pre-computes qualifying page list with I/O descriptors.
// Phase 2: execution kernel reads pre-computed descriptors and runs fused decomp+eval.
// ============================================================
BAMRunResult bam_q6_prescan_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl_handle) {
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    uint32_t num_blocks = params.num_blocks;
    const uint64_t page_size = params.page_size;
    const uint64_t npages = params.field_npages;

    // Prescan uses io_multi=1 (sync execution kernel)
    const uint32_t io_multi = 1;

    // ── Page cache: 7 entries per block ──
    const uint64_t n_pc_pages = (uint64_t)num_blocks * ENTRIES_PER_BLOCK;
    const uint64_t max_range = 64;

    size_t gpu_free = 0, gpu_total = 0;
    BAM_CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));
    fprintf(stderr, "[prescan] GPU memory: %.1f GB free / %.1f GB total\n",
            gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));

    page_cache_t h_pc(page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);
    page_cache_d_t* d_pc = h_pc.d_pc_ptr;

    // ── Copy prefix sum arrays to GPU ──
    const uint64_t ps_len = npages + 1;
    const size_t ps_bytes = ps_len * sizeof(uint64_t);
    uint64_t* d_prefix_sums[4];
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaMalloc(&d_prefix_sums[fi], ps_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_prefix_sums[fi], params.h_prefix_sums[fi],
                                   ps_bytes, cudaMemcpyHostToDevice));
    }

    // ── Copy zone map stats to GPU ──
    int32_t* d_shipdate_stats = nullptr;
    if (params.h_shipdate_stats && params.nstats > 0) {
        const size_t stats_bytes = params.nstats * 2 * sizeof(int32_t);
        BAM_CUDA_CHECK(cudaMalloc(&d_shipdate_stats, stats_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_shipdate_stats, params.h_shipdate_stats,
                                   stats_bytes, cudaMemcpyHostToDevice));
    }

    // ── Copy compression metadata to GPU ──
    uint32_t* d_comp_sizes[4] = {};
    uint64_t* d_comp_offsets[4] = {};
    for (int fi = 0; fi < 4; fi++) {
        if (params.compression_method[fi] != 0 &&
            params.h_compressed_page_sizes[fi] && params.h_compressed_offsets[fi]) {
            size_t sz_sizes = npages * sizeof(uint32_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes[fi], sz_sizes));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes[fi], params.h_compressed_page_sizes[fi],
                                       sz_sizes, cudaMemcpyHostToDevice));

            size_t sz_offsets = npages * sizeof(uint64_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], sz_offsets));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], params.h_compressed_offsets[fi],
                                       sz_offsets, cudaMemcpyHostToDevice));
        }
    }

    // ── Decompression output buffer (fallback path only) ──
    const uint32_t elems_per_slot = params.decomp_elems_per_slot;
    int32_t* d_decomp_buf = nullptr;
    size_t decomp_buf_size = (size_t)num_blocks * ENTRIES_PER_BLOCK
                           * elems_per_slot * sizeof(int32_t);
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp_buf, decomp_buf_size));

    // ── GPU metadata ──
    BAMKernelMeta h_meta;
    for (int fi = 0; fi < 4; fi++) {
        h_meta.field_start_page_ids[fi] = params.field_start_page_ids[fi];
        h_meta.d_prefix_sums[fi] = d_prefix_sums[fi];
        h_meta.compression_method[fi] = params.compression_method[fi];
        h_meta.d_comp_sizes[fi] = d_comp_sizes[fi];
        h_meta.d_comp_offsets[fi] = d_comp_offsets[fi];
    }
    h_meta.field_npages = npages;
    h_meta.page_size = params.page_size;
    h_meta.blocks_per_page = params.blocks_per_page;
    h_meta.partition_start_lba = params.partition_start_lba;
    h_meta.n_devices = params.n_devices > 0 ? params.n_devices : 1;
    for (uint32_t d = 0; d < h_meta.n_devices; d++)
        h_meta.partition_start_lbas[d] = params.partition_start_lbas[d];
    h_meta.num_blocks = num_blocks;
    h_meta.nrows = params.nrows;
    h_meta.d_shipdate_stats = d_shipdate_stats;
    h_meta.nstats = params.nstats;
    h_meta.d_decomp_buf = d_decomp_buf;
    h_meta.decomp_elems_per_slot = elems_per_slot;
    h_meta.io_multi = io_multi;
    h_meta.sd_low = params.sd_low;
    h_meta.sd_high = params.sd_high;

    BAMKernelMeta* d_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_meta, sizeof(BAMKernelMeta)));
    BAM_CUDA_CHECK(cudaMemcpy(d_meta, &h_meta, sizeof(BAMKernelMeta),
                               cudaMemcpyHostToDevice));

    int64_t* d_revenue;
    BAM_CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    BAMDiagCounters* d_diag;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, sizeof(BAMDiagCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, sizeof(BAMDiagCounters)));

    BAMPerfCounters* d_perf;
    BAM_CUDA_CHECK(cudaMalloc(&d_perf, sizeof(BAMPerfCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_perf, 0, sizeof(BAMPerfCounters)));

    size_t old_stack_size = 0;
    cudaDeviceGetLimit(&old_stack_size, cudaLimitStackSize);
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Phase 1: Prescan kernel ──
    QualPageIO* d_qual_pages;
    BAM_CUDA_CHECK(cudaMalloc(&d_qual_pages, npages * sizeof(QualPageIO)));
    uint32_t* d_num_qual;
    BAM_CUDA_CHECK(cudaMalloc(&d_num_qual, sizeof(uint32_t)));
    BAM_CUDA_CHECK(cudaMemset(d_num_qual, 0, sizeof(uint32_t)));

    cudaEvent_t prescan_start, prescan_end;
    BAM_CUDA_CHECK(cudaEventCreate(&prescan_start));
    BAM_CUDA_CHECK(cudaEventCreate(&prescan_end));

    const uint32_t prescan_threads = 256;
    const uint32_t prescan_blocks = (npages + prescan_threads - 1) / prescan_threads;

    BAM_CUDA_CHECK(cudaEventRecord(prescan_start, stream));
    bam_q6_prescan_kernel<<<prescan_blocks, prescan_threads, 0, stream>>>(
        d_meta, d_qual_pages, d_num_qual);
    BAM_CUDA_CHECK(cudaEventRecord(prescan_end, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    float prescan_ms = 0;
    BAM_CUDA_CHECK(cudaEventElapsedTime(&prescan_ms, prescan_start, prescan_end));

    uint32_t h_num_qual = 0;
    BAM_CUDA_CHECK(cudaMemcpy(&h_num_qual, d_num_qual, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

    fprintf(stderr, "[prescan] %u / %lu pages qualified (%.1f%%), prescan: %.3f ms\n",
            h_num_qual, (unsigned long)npages, 100.0 * h_num_qual / npages, prescan_ms);

    // ── Phase 2: Execution kernel ──
    const uint32_t threads_per_block = 32;
    fprintf(stderr, "[prescan] Launching %u blocks x %u threads, decomp_buf=%zu MB\n",
            num_blocks, threads_per_block, decomp_buf_size / (1024 * 1024));

    bam_q6_kernel_prescan_sync<<<num_blocks, threads_per_block, 0, stream>>>(
        h_pc.pdt.d_ctrls, d_pc, d_meta, d_qual_pages, h_num_qual,
        d_revenue, d_diag, d_perf);
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Result ──
    int64_t h_revenue = 0;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_revenue, d_revenue, sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, stream));

    BAMDiagCounters h_diag;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_diag, d_diag, sizeof(BAMDiagCounters),
                                    cudaMemcpyDeviceToHost, stream));

    BAMPerfCounters h_perf;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_perf, d_perf, sizeof(BAMPerfCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr, "[diag] sum_shipdate=%lu  sum_quantity=%lu\n",
            h_diag.sum_shipdate, h_diag.sum_quantity);
    fprintf(stderr, "[diag] sum_extprice=%lu  sum_discount=%lu\n",
            h_diag.sum_extprice, h_diag.sum_discount);

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, params.cuda_device);
    double clock_ghz = clock_khz / 1e6;

    uint64_t total_cycles = h_perf.io_cycles + h_perf.decomp_cycles + h_perf.eval_cycles;
    fprintf(stderr, "\n[perf] GPU clock: %.3f GHz\n", clock_ghz);
    fprintf(stderr, "[perf] pages_processed: %lu  io_calls: %lu  io_bytes: %lu (%.2f MiB)\n",
            h_perf.page_count, h_perf.io_count,
            h_perf.io_bytes, h_perf.io_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[perf] boundary_page_reads: %lu (%.1f%% of pages)\n",
            h_perf.boundary_count,
            h_perf.page_count > 0 ? 100.0 * h_perf.boundary_count / h_perf.page_count : 0.0);
    fprintf(stderr, "[perf] Accumulated cycles across %u blocks (thread 0):\n", num_blocks);
    fprintf(stderr, "[perf]   io:      %12lu cycles  (%5.1f%%)\n",
            h_perf.io_cycles, total_cycles > 0 ? 100.0 * h_perf.io_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   decomp:  %12lu cycles  (%5.1f%%)\n",
            h_perf.decomp_cycles, total_cycles > 0 ? 100.0 * h_perf.decomp_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   eval:    %12lu cycles  (%5.1f%%)\n",
            h_perf.eval_cycles, total_cycles > 0 ? 100.0 * h_perf.eval_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   total:   %12lu cycles\n", total_cycles);
    if (h_perf.io_count > 0) {
        double avg_io_us = (double)h_perf.io_cycles / h_perf.io_count / (clock_ghz * 1e3);
        fprintf(stderr, "[perf]   avg_io_latency: %.1f us/call  (%lu calls)\n",
                avg_io_us, h_perf.io_count);
        double avg_io_kb = (double)h_perf.io_bytes / h_perf.io_count / 1024.0;
        fprintf(stderr, "[perf]   avg_io_size: %.1f KiB/call\n", avg_io_kb);
    }

    BAM_CUDA_CHECK(cudaEventDestroy(prescan_start));
    BAM_CUDA_CHECK(cudaEventDestroy(prescan_end));
    BAM_CUDA_CHECK(cudaStreamDestroy(stream));

    BAM_CUDA_CHECK(cudaFree(d_qual_pages));
    BAM_CUDA_CHECK(cudaFree(d_num_qual));
    BAM_CUDA_CHECK(cudaFree(d_meta));
    BAM_CUDA_CHECK(cudaFree(d_revenue));
    BAM_CUDA_CHECK(cudaFree(d_diag));
    BAM_CUDA_CHECK(cudaFree(d_perf));
    BAM_CUDA_CHECK(cudaFree(d_decomp_buf));
    if (d_shipdate_stats) BAM_CUDA_CHECK(cudaFree(d_shipdate_stats));
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaFree(d_prefix_sums[fi]));
        if (d_comp_sizes[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_sizes[fi]));
        if (d_comp_offsets[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_offsets[fi]));
    }

    return BAMRunResult{h_revenue, h_perf.io_count, h_perf.io_bytes};
}

// ============================================================
// BAM Q6 Coalesced I/O kernel — pipelined (io_multi ring buffer)
//
// 1 warp processes k consecutive driving pages per ring slot.
// Each field is read in a single NVMe command covering k pages.
// Requires identical prefix sums across all 4 fields (no boundary pages).
//
// Page cache entries per slot: 4 (SD_group, QT_group, EP_group, DC_group).
// Each entry holds k consecutive pages (page_cache page_size = original * k).
// Ring buffer depth = io_multi for pipelined I/O.
// ============================================================
#define COALESCED_ENTRIES_PER_SLOT 4

// Per-slot metadata for coalesced ring buffer (shared memory).
struct CoalSlotMeta {
    uint64_t pg_base;     // first page index in group
    uint32_t actual_k;    // pages in this group
    uint32_t io_bytes;    // total NVMe bytes submitted
    int      is_contiguous; // 1 = coalesced I/O, 0 = per-page fallback (segment boundary)
};

__global__ void bam_q6_kernel_coalesced_io_multi(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;  // 0..31
    const uint32_t io_multi = meta->io_multi;

    const uint32_t n_qps = ctrls[0]->n_qps;
    QueuePair* d_qps = ctrls[0]->d_qps;

    const uint64_t npages = meta->field_npages;
    const uint32_t k = meta->coalesce_k;
    const uint32_t orig_page_size = meta->original_page_size;
    const uint64_t pc_page_size = meta->page_size;  // = orig_page_size * k
    const uint32_t blocks_per_page = meta->blocks_per_page;

    const uint64_t n_groups = (npages + k - 1) / k;

    const uint64_t* ps0 = meta->d_prefix_sums[0];
    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;
    const uint16_t* comp = meta->compression_method;

    const int32_t sd_low  = meta->sd_low;
    const int32_t sd_high = meta->sd_high;

    __shared__ uint shared_bws[4];
    __shared__ uint shared_offs[4];
    __shared__ int s_skip;
    __shared__ int s_contig;
    __shared__ uint32_t s_comp_blk[136];  // max PFOR block = 130 uint32

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;

    // Ring buffer slot metadata in shared memory
    __shared__ CoalSlotMeta coal_meta[MAX_IO_MULTI];

    // Per-thread per-slot CID (only lanes 0-3 store meaningful values)
    uint16_t ring_cid[MAX_IO_MULTI];

    // Per-block profiling accumulators
    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;

    // Ring buffer state
    uint64_t cursor = block_id;       // next group to submit (grid-stride)
    uint32_t ring_head = 0;
    uint32_t ring_count = 0;

    // ────── Helper: try to fill one ring slot from cursor ──────
    #define TRY_FILL_COAL() do {                                                \
        bool _filled = false;                                                   \
        while (!_filled && cursor < n_groups) {                                 \
            uint64_t _grp = cursor;                                             \
            cursor += gridDim.x;                                                \
            uint64_t _pg_base = _grp * k;                                       \
            uint64_t _pg_end = _pg_base + k;                                    \
            if (_pg_end > npages) _pg_end = npages;                             \
            uint32_t _actual_k = (uint32_t)(_pg_end - _pg_base);               \
            /* Group-level zone map: skip if ALL pages prunable */              \
            if (tid == 0) {                                                     \
                s_skip = 1;                                                     \
                if (!stats) {                                                   \
                    s_skip = 0;                                                 \
                } else {                                                        \
                    for (uint32_t _i = 0; _i < _actual_k; _i++) {              \
                        uint64_t _pg = _pg_base + _i;                           \
                        if (_pg < nstats) {                                     \
                            int32_t _pmin = stats[_pg * 2];                     \
                            int32_t _pmax = stats[_pg * 2 + 1];                \
                            if (!(_pmax < sd_low || _pmin > (sd_high - 1))) {  \
                                s_skip = 0; break;                              \
                            }                                                   \
                        } else { s_skip = 0; break; }                           \
                    }                                                           \
                }                                                               \
            }                                                                   \
            __syncwarp();                                                        \
            if (s_skip) continue;                                               \
            /* Skip empty groups */                                             \
            if (ps0[_pg_end] == ps0[_pg_base]) continue;                        \
            /* Check contiguity for compressed fields */                        \
            if (tid == 0) {                                                     \
                s_contig = 1;                                                   \
                for (int _f = 0; _f < 4; _f++) {                               \
                    if (comp[_f] != 0) {                                        \
                        uint64_t _span = meta->d_comp_offsets[_f][_pg_end]      \
                                       - meta->d_comp_offsets[_f][_pg_base];    \
                        if (_span > pc_page_size) {                             \
                            s_contig = 0; break;                                \
                        }                                                       \
                    }                                                           \
                }                                                               \
            }                                                                   \
            __syncwarp();                                                        \
            /* Allocate ring slot */                                            \
            uint32_t _s = (ring_head + ring_count) % io_multi;                  \
            QueuePair* _qp = d_qps                                             \
                + ((block_id * io_multi + _s) % n_qps);                         \
            unsigned long long _entry_base =                                    \
                (unsigned long long)(block_id * io_multi + _s)                  \
                * COALESCED_ENTRIES_PER_SLOT;                                    \
            /* Submit coalesced I/O only if contiguous */                       \
            uint32_t _total_bytes = 0;                                          \
            if (s_contig) {                                                     \
                uint64_t _io_lba[4];                                            \
                uint32_t _io_nblocks[4];                                        \
                for (int _f = 0; _f < 4; _f++) {                               \
                    if (comp[_f] != 0) {                                        \
                        _io_lba[_f] = meta->partition_start_lba                 \
                            + meta->d_comp_offsets[_f][_pg_base] / 512;         \
                        uint64_t _span = meta->d_comp_offsets[_f][_pg_end]      \
                                       - meta->d_comp_offsets[_f][_pg_base];    \
                        _io_nblocks[_f] = safe_io_nblocks((uint32_t)_span);     \
                    } else {                                                    \
                        uint64_t _pg_id = meta->field_start_page_ids[_f]        \
                                        + _pg_base;                             \
                        _io_lba[_f] = meta->partition_start_lba                 \
                            + _pg_id * blocks_per_page;                         \
                        _io_nblocks[_f] = _actual_k * blocks_per_page;          \
                    }                                                           \
                }                                                               \
                uint16_t _sq_pos = 0;                                           \
                if (tid < 4) {                                                  \
                    unsigned long long _ent = _entry_base + tid;                \
                    access_data_async(pc, _qp, _io_lba[tid], _io_nblocks[tid], \
                        _ent, NVM_IO_READ, &ring_cid[_s], &_sq_pos);           \
                }                                                               \
                __syncwarp();                                                    \
                for (int _ii = 0; _ii < 4; _ii++)                              \
                    _total_bytes += _io_nblocks[_ii] * 512;                     \
            }                                                                   \
            /* Save slot metadata */                                            \
            if (tid == 0) {                                                     \
                coal_meta[_s].pg_base = _pg_base;                               \
                coal_meta[_s].actual_k = _actual_k;                             \
                coal_meta[_s].io_bytes = _total_bytes;                          \
                coal_meta[_s].is_contiguous = s_contig;                         \
            }                                                                   \
            __syncwarp();                                                        \
            ring_count++;                                                       \
            _filled = true;                                                     \
        }                                                                       \
    } while(0)

    // ────── Initial fill ──────
    TRY_FILL_COAL();

    // ────── Main loop: prefetch + process ──────
    while (ring_count > 0) {
        // Prefetch: submit one more slot if room
        if (ring_count < io_multi) {
            TRY_FILL_COAL();
        }

        // Process oldest slot
        uint32_t s = ring_head;
        CoalSlotMeta& cm = coal_meta[s];

        // 1. Poll coalesced I/O completion (only if contiguous)
        QueuePair* qp_s = d_qps + ((block_id * io_multi + s) % n_qps);
        if (cm.is_contiguous) {
            long long t0 = clock64();
            if (tid < 4) {
                uint32_t poll_loc, poll_head;
                uint32_t cq_pos = cq_poll(&qp_s->cq, ring_cid[s], &poll_loc, &poll_head);
                cq_dequeue(&qp_s->cq, cq_pos, &qp_s->sq);
                put_cid(&qp_s->sq, ring_cid[s]);
            }
            __syncwarp();
            long long t1 = clock64();
            blk_io_cycles += (uint64_t)(t1 - t0);
            blk_io_count += 4;
            blk_io_bytes += cm.io_bytes;
        }

        // 2. Process each sub-page in the group
        unsigned long long entry_base =
            (unsigned long long)(block_id * io_multi + s) * COALESCED_ENTRIES_PER_SLOT;
        int32_t* decomp = meta->d_decomp_buf
            + (uint64_t)(block_id * io_multi + s) * COALESCED_ENTRIES_PER_SLOT * elems_per_slot;

        for (uint32_t sub = 0; sub < cm.actual_k; sub++) {
            uint64_t pg = cm.pg_base + sub;

            // Per-page zone map check
            if (tid == 0) {
                s_skip = 0;
                if (stats && pg < nstats) {
                    int32_t page_min = stats[pg * 2];
                    int32_t page_max = stats[pg * 2 + 1];
                    if (page_max < sd_low || page_min > (sd_high - 1)) {
                        s_skip = 1;
                    }
                }
            }
            __syncwarp();
            if (s_skip) continue;

            uint64_t row_begin = ps0[pg];
            uint64_t row_end   = ps0[pg + 1];
            uint64_t nrows_this = row_end - row_begin;
            if (nrows_this == 0) continue;

            blk_page_count++;

            // --- Per-page I/O fallback for non-contiguous groups ---
            if (!cm.is_contiguous) {
                uint16_t page_cid = 0;
                uint16_t sq_pos = 0;
                uint64_t io_lba = 0;
                uint32_t io_nblocks_lane = 0;
                if (tid < 4) {
                    if (comp[tid] != 0) {
                        io_lba = meta->partition_start_lba
                               + meta->d_comp_offsets[tid][pg] / 512;
                        io_nblocks_lane = safe_io_nblocks(meta->d_comp_sizes[tid][pg]);
                    } else {
                        uint64_t pg_id = meta->field_start_page_ids[tid] + pg;
                        io_lba = meta->partition_start_lba + pg_id * blocks_per_page;
                        io_nblocks_lane = blocks_per_page;
                    }
                    unsigned long long ent = entry_base + tid;
                    access_data_async(pc, qp_s, io_lba, io_nblocks_lane,
                                      ent, NVM_IO_READ, &page_cid, &sq_pos);
                }
                __syncwarp();
                long long t0 = clock64();
                if (tid < 4) {
                    uint32_t poll_loc, poll_head;
                    uint32_t cq_pos = cq_poll(&qp_s->cq, page_cid, &poll_loc, &poll_head);
                    cq_dequeue(&qp_s->cq, cq_pos, &qp_s->sq);
                    put_cid(&qp_s->sq, page_cid);
                }
                __syncwarp();
                long long t1 = clock64();
                blk_io_cycles += (uint64_t)(t1 - t0);
                blk_io_count += 4;
                // Sum per-lane I/O bytes via warp shuffle
                uint32_t my_bytes = (tid < 4) ? (io_nblocks_lane * 512) : 0;
                for (int off = 16; off > 0; off /= 2)
                    my_bytes += __shfl_down_sync(0xFFFFFFFF, my_bytes, off);
                if (tid == 0) blk_io_bytes += my_bytes;
            }

            // Compute sub-page buffer pointers
            char* sub_page_ptr[4];
            for (int f = 0; f < 4; f++) {
                char* group_buf = base + (entry_base + f) * pc_page_size;
                if (!cm.is_contiguous) {
                    // Per-page read: data at start of entry
                    sub_page_ptr[f] = group_buf;
                } else if (comp[f] != 0) {
                    uint64_t off = meta->d_comp_offsets[f][pg]
                                 - meta->d_comp_offsets[f][cm.pg_base];
                    sub_page_ptr[f] = group_buf + off;
                } else {
                    sub_page_ptr[f] = group_buf + (uint64_t)sub * orig_page_size;
                }
            }

            // ── Fused decompress + Q6 eval ──
            // Decode PFOR blocks via shared memory, evaluate Q6 predicates
            // directly on register values. Eliminates decomp_out global buffer.
            long long td0 = clock64();

            bool all_comp = (comp[0] != 0 && comp[1] != 0 &&
                             comp[2] != 0 && comp[3] != 0);
            bool none_comp = (comp[0] == 0 && comp[1] == 0 &&
                              comp[2] == 0 && comp[3] == 0);

            int64_t local_rev = 0;
            uint64_t local_sum_sd = 0, local_sum_qt = 0;
            uint64_t local_sum_ep = 0, local_sum_dc = 0;

            if (all_comp) {
                // ── Case A: all fields compressed ──
                // Decode via shared memory, eval on registers.
                bam_pag_head* hdr0 = (bam_pag_head*)sub_page_ptr[0];
                uint32_t wm = hdr0->watermark;
                uint32_t nalloc = hdr0->nalloc;
                uint32_t nblocks_pfor = wm / 128;

                uint32_t* blk_start[4];
                uint32_t* data_ptr_f[4];
                for (int f = 0; f < 4; f++) {
                    blk_start[f] = (uint32_t*)(sub_page_ptr[f] + BAM_PAG_HDR_BYTES);
                    data_ptr_f[f] = blk_start[f] + (nblocks_pfor + 1);
                }

                for (uint32_t b = 0; b < nblocks_pfor; b++) {
                    int32_t sd_v[4], qt_v[4], ep_v[4], dc_v[4];

                    decode_pfor_block_smem(
                        data_ptr_f[0] + blk_start[0][b],
                        blk_start[0][b + 1] - blk_start[0][b],
                        s_comp_blk, shared_bws, shared_offs, tid, sd_v);
                    decode_pfor_block_smem(
                        data_ptr_f[1] + blk_start[1][b],
                        blk_start[1][b + 1] - blk_start[1][b],
                        s_comp_blk, shared_bws, shared_offs, tid, qt_v);
                    decode_pfor_block_smem(
                        data_ptr_f[2] + blk_start[2][b],
                        blk_start[2][b + 1] - blk_start[2][b],
                        s_comp_blk, shared_bws, shared_offs, tid, ep_v);
                    decode_pfor_block_smem(
                        data_ptr_f[3] + blk_start[3][b],
                        blk_start[3][b + 1] - blk_start[3][b],
                        s_comp_blk, shared_bws, shared_offs, tid, dc_v);

                    // Q6 predicate evaluation on register values
                    for (uint32_t ki = 0; ki < 4; ki++) {
                        uint32_t idx = b * 128 + ki * 32 + tid;
                        if (idx < nalloc) {
                            local_sum_sd += (uint32_t)sd_v[ki];
                            local_sum_qt += (uint32_t)qt_v[ki];
                            local_sum_ep += (uint32_t)ep_v[ki];
                            local_sum_dc += (uint32_t)dc_v[ki];

                            if (sd_v[ki] >= sd_low && sd_v[ki] < sd_high &&
                                dc_v[ki] >= 5 && dc_v[ki] <= 7 &&
                                qt_v[ki] < 2400) {
                                local_rev += (int64_t)ep_v[ki] * dc_v[ki];
                            }
                        }
                    }
                }
            } else if (none_comp) {
                // ── Case B: all fields uncompressed ──
                // Read directly from page cache, no decomp buffer.
                int32_t* src_sd = (int32_t*)(sub_page_ptr[0] + BAM_PAG_HDR_BYTES);
                int32_t* src_qt = (int32_t*)(sub_page_ptr[1] + BAM_PAG_HDR_BYTES);
                int32_t* src_ep = (int32_t*)(sub_page_ptr[2] + BAM_PAG_HDR_BYTES);
                int32_t* src_dc = (int32_t*)(sub_page_ptr[3] + BAM_PAG_HDR_BYTES);

                for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 32) {
                    int32_t l_shipdate      = src_sd[i];
                    int32_t l_quantity      = src_qt[i];
                    int32_t l_extendedprice = src_ep[i];
                    int32_t l_discount      = src_dc[i];

                    local_sum_sd += (uint32_t)l_shipdate;
                    local_sum_qt += (uint32_t)l_quantity;
                    local_sum_ep += (uint32_t)l_extendedprice;
                    local_sum_dc += (uint32_t)l_discount;

                    if (l_shipdate >= sd_low && l_shipdate < sd_high &&
                        l_discount >= 5 && l_discount <= 7 &&
                        l_quantity < 2400) {
                        local_rev += (int64_t)l_extendedprice * l_discount;
                    }
                }
            } else {
                // ── Case C: mixed compression (fallback) ──
                // Decompress to global buffer, then eval separately.
                int32_t* sd_decomp = decomp + 0 * elems_per_slot;
                decompress_page_warp(sub_page_ptr[0], sd_decomp,
                                     comp[0], shared_bws, shared_offs, tid, nullptr);
                int32_t* qt_decomp = decomp + 1 * elems_per_slot;
                decompress_page_warp(sub_page_ptr[1], qt_decomp,
                                     comp[1], shared_bws, shared_offs, tid, nullptr);
                int32_t* ep_decomp = decomp + 2 * elems_per_slot;
                decompress_page_warp(sub_page_ptr[2], ep_decomp,
                                     comp[2], shared_bws, shared_offs, tid, nullptr);
                int32_t* dc_decomp = decomp + 3 * elems_per_slot;
                decompress_page_warp(sub_page_ptr[3], dc_decomp,
                                     comp[3], shared_bws, shared_offs, tid, nullptr);

                for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 32) {
                    int32_t l_shipdate      = sd_decomp[i];
                    int32_t l_quantity      = qt_decomp[i];
                    int32_t l_extendedprice = ep_decomp[i];
                    int32_t l_discount      = dc_decomp[i];

                    local_sum_sd += (uint32_t)l_shipdate;
                    local_sum_qt += (uint32_t)l_quantity;
                    local_sum_ep += (uint32_t)l_extendedprice;
                    local_sum_dc += (uint32_t)l_discount;

                    if (l_shipdate >= sd_low && l_shipdate < sd_high &&
                        l_discount >= 5 && l_discount <= 7 &&
                        l_quantity < 2400) {
                        local_rev += (int64_t)l_extendedprice * l_discount;
                    }
                }
            }

            // Warp-level reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
                local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
                local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
                local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
                local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
            }

            if (tid == 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                          static_cast<unsigned long long>(local_rev));
                if (d_diag) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                              (unsigned long long)local_sum_sd);
                    atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                              (unsigned long long)local_sum_qt);
                    atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                              (unsigned long long)local_sum_ep);
                    atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                              (unsigned long long)local_sum_dc);
                }
            }

            long long td1 = clock64();
            // Fused decomp+eval: report all cycles under decomp (eval = 0).
            blk_decomp_cycles += (uint64_t)(td1 - td0);
        }

        // Advance ring
        ring_head = (ring_head + 1) % io_multi;
        ring_count--;

        // Refill: when ring becomes empty, try to submit next group.
        // This is essential for io_multi=1 where the top-of-loop prefetch
        // condition (ring_count < io_multi → 1 < 1) never triggers.
        if (ring_count == 0) {
            TRY_FILL_COAL();
        }
    }

    #undef TRY_FILL_COAL

    // Flush profiling counters
    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
    }
}

// ============================================================
// Host wrapper for 128-thread Q6 kernel (4 warps, parallel field decode)
// ============================================================
BAMRunResult bam_q6_128t_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl_handle) {
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    uint32_t num_blocks = params.num_blocks;
    const uint64_t page_size = params.page_size;

    // 128t kernel uses io_multi=1 (sync I/O within warp 0)
    const uint32_t io_multi = 1;

    // ── Page cache: 7 entries per block ──
    const uint64_t n_pc_pages = (uint64_t)num_blocks * ENTRIES_PER_BLOCK;
    const uint64_t max_range = 64;

    size_t gpu_free = 0, gpu_total = 0;
    BAM_CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));
    fprintf(stderr, "[bam_q6_128t] GPU memory: %.1f GB free / %.1f GB total\n",
            gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));

    page_cache_t h_pc(page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);
    page_cache_d_t* d_pc = h_pc.d_pc_ptr;

    // ── Copy prefix sum arrays to GPU ──
    const uint64_t ps_len = params.field_npages + 1;
    const size_t ps_bytes = ps_len * sizeof(uint64_t);
    uint64_t* d_prefix_sums[4];
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaMalloc(&d_prefix_sums[fi], ps_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_prefix_sums[fi], params.h_prefix_sums[fi],
                                   ps_bytes, cudaMemcpyHostToDevice));
    }

    // ── Copy zone map stats to GPU ──
    int32_t* d_shipdate_stats = nullptr;
    if (params.h_shipdate_stats && params.nstats > 0) {
        const size_t stats_bytes = params.nstats * 2 * sizeof(int32_t);
        BAM_CUDA_CHECK(cudaMalloc(&d_shipdate_stats, stats_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_shipdate_stats, params.h_shipdate_stats,
                                   stats_bytes, cudaMemcpyHostToDevice));
    }

    // ── Copy compression metadata to GPU ──
    uint32_t* d_comp_sizes[4] = {};
    uint64_t* d_comp_offsets[4] = {};
    for (int fi = 0; fi < 4; fi++) {
        if (params.compression_method[fi] != 0 &&
            params.h_compressed_page_sizes[fi] && params.h_compressed_offsets[fi]) {
            size_t sz_sizes = params.field_npages * sizeof(uint32_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes[fi], sz_sizes));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes[fi], params.h_compressed_page_sizes[fi],
                                       sz_sizes, cudaMemcpyHostToDevice));

            size_t sz_offsets = params.field_npages * sizeof(uint64_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], sz_offsets));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], params.h_compressed_offsets[fi],
                                       sz_offsets, cudaMemcpyHostToDevice));
        }
    }

    // ── Decompression output buffer (for Case B fallback) ──
    const uint32_t elems_per_slot = params.decomp_elems_per_slot;
    int32_t* d_decomp_buf = nullptr;
    size_t decomp_buf_size = (size_t)num_blocks * ENTRIES_PER_BLOCK
                           * elems_per_slot * sizeof(int32_t);
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp_buf, decomp_buf_size));

    // ── GPU metadata ──
    BAMKernelMeta h_meta;
    for (int fi = 0; fi < 4; fi++) {
        h_meta.field_start_page_ids[fi] = params.field_start_page_ids[fi];
        h_meta.d_prefix_sums[fi] = d_prefix_sums[fi];
        h_meta.compression_method[fi] = params.compression_method[fi];
        h_meta.d_comp_sizes[fi] = d_comp_sizes[fi];
        h_meta.d_comp_offsets[fi] = d_comp_offsets[fi];
    }
    h_meta.field_npages = params.field_npages;
    h_meta.page_size = params.page_size;
    h_meta.blocks_per_page = params.blocks_per_page;
    h_meta.partition_start_lba = params.partition_start_lba;
    h_meta.n_devices = params.n_devices > 0 ? params.n_devices : 1;
    for (uint32_t d = 0; d < h_meta.n_devices; d++)
        h_meta.partition_start_lbas[d] = params.partition_start_lbas[d];
    h_meta.num_blocks = num_blocks;
    h_meta.nrows = params.nrows;
    h_meta.d_shipdate_stats = d_shipdate_stats;
    h_meta.nstats = params.nstats;
    h_meta.d_decomp_buf = d_decomp_buf;
    h_meta.decomp_elems_per_slot = elems_per_slot;
    h_meta.io_multi = io_multi;
    h_meta.sd_low = params.sd_low;
    h_meta.sd_high = params.sd_high;

    BAMKernelMeta* d_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_meta, sizeof(BAMKernelMeta)));
    BAM_CUDA_CHECK(cudaMemcpy(d_meta, &h_meta, sizeof(BAMKernelMeta),
                               cudaMemcpyHostToDevice));

    int64_t* d_revenue;
    BAM_CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    BAMDiagCounters* d_diag;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, sizeof(BAMDiagCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, sizeof(BAMDiagCounters)));

    BAMPerfCounters* d_perf;
    BAM_CUDA_CHECK(cudaMalloc(&d_perf, sizeof(BAMPerfCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_perf, 0, sizeof(BAMPerfCounters)));

    size_t old_stack_size = 0;
    cudaDeviceGetLimit(&old_stack_size, cudaLimitStackSize);
    fprintf(stderr, "[bam_q6_128t] Default stack size: %zu bytes, %u devices\n",
            old_stack_size, h_meta.n_devices);
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    // ── Kernel launch: 128 threads per block ──
    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    const uint32_t threads_per_block = 128;
    fprintf(stderr, "[bam_q6_128t] Launching %u blocks x %u threads (4 warps/block), decomp_buf=%zu MB\n",
            num_blocks, threads_per_block, decomp_buf_size / (1024 * 1024));

    bam_q6_kernel_128t_sync<<<num_blocks, threads_per_block, 0, stream>>>(
        h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Result ──
    int64_t h_revenue = 0;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_revenue, d_revenue, sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, stream));

    BAMDiagCounters h_diag;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_diag, d_diag, sizeof(BAMDiagCounters),
                                    cudaMemcpyDeviceToHost, stream));

    BAMPerfCounters h_perf;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_perf, d_perf, sizeof(BAMPerfCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr, "[diag] sum_shipdate=%lu  sum_quantity=%lu\n",
            h_diag.sum_shipdate, h_diag.sum_quantity);
    fprintf(stderr, "[diag] sum_extprice=%lu  sum_discount=%lu\n",
            h_diag.sum_extprice, h_diag.sum_discount);

    // ── Print profiling breakdown ──
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, params.cuda_device);
    double clock_ghz = clock_khz / 1e6;

    uint64_t total_cycles = h_perf.io_cycles + h_perf.decomp_cycles + h_perf.eval_cycles;
    fprintf(stderr, "\n[perf] GPU clock: %.3f GHz\n", clock_ghz);
    fprintf(stderr, "[perf] pages_processed: %lu  io_calls: %lu  io_bytes: %lu (%.2f MiB)\n",
            h_perf.page_count, h_perf.io_count,
            h_perf.io_bytes, h_perf.io_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[perf] boundary_page_reads: %lu (%.1f%% of pages)\n",
            h_perf.boundary_count,
            h_perf.page_count > 0 ? 100.0 * h_perf.boundary_count / h_perf.page_count : 0.0);
    fprintf(stderr, "[perf] Accumulated cycles across %u blocks (thread 0):\n", num_blocks);
    fprintf(stderr, "[perf]   io:      %12lu cycles  (%5.1f%%)\n",
            h_perf.io_cycles, total_cycles > 0 ? 100.0 * h_perf.io_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   decomp:  %12lu cycles  (%5.1f%%)\n",
            h_perf.decomp_cycles, total_cycles > 0 ? 100.0 * h_perf.decomp_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   eval:    %12lu cycles  (%5.1f%%)\n",
            h_perf.eval_cycles, total_cycles > 0 ? 100.0 * h_perf.eval_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   total:   %12lu cycles\n", total_cycles);
    if (h_perf.io_count > 0) {
        double avg_io_us = (double)h_perf.io_cycles / h_perf.io_count / (clock_ghz * 1e3);
        fprintf(stderr, "[perf]   avg_io_latency: %.1f us/call  (%lu calls)\n",
                avg_io_us, h_perf.io_count);
        double avg_io_kb = (double)h_perf.io_bytes / h_perf.io_count / 1024.0;
        fprintf(stderr, "[perf]   avg_io_size: %.1f KiB/call\n", avg_io_kb);
    }

    BAM_CUDA_CHECK(cudaStreamDestroy(stream));

    // ── Cleanup ──
    BAM_CUDA_CHECK(cudaFree(d_meta));
    BAM_CUDA_CHECK(cudaFree(d_revenue));
    BAM_CUDA_CHECK(cudaFree(d_diag));
    BAM_CUDA_CHECK(cudaFree(d_perf));
    BAM_CUDA_CHECK(cudaFree(d_decomp_buf));
    if (d_shipdate_stats) BAM_CUDA_CHECK(cudaFree(d_shipdate_stats));
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaFree(d_prefix_sums[fi]));
        if (d_comp_sizes[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_sizes[fi]));
        if (d_comp_offsets[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_offsets[fi]));
    }

    return BAMRunResult{h_revenue, h_perf.io_count, h_perf.io_bytes};
}

// ============================================================
// Host wrapper for 128t prefetch kernel (Pattern 2)
// ============================================================
BAMRunResult bam_q6_128t_pf_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl_handle) {
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    uint32_t num_blocks = params.num_blocks;
    const uint64_t page_size = params.page_size;

    // Double-buffered: 14 entries per block
    const uint64_t n_pc_pages = (uint64_t)num_blocks * ENTRIES_PER_BLOCK_DBL;
    const uint64_t max_range = 64;

    size_t gpu_free = 0, gpu_total = 0;
    BAM_CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));
    fprintf(stderr, "[bam_q6_128t_pf] GPU memory: %.1f GB free / %.1f GB total\n",
            gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));

    page_cache_t h_pc(page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);
    page_cache_d_t* d_pc = h_pc.d_pc_ptr;

    const uint64_t ps_len = params.field_npages + 1;
    const size_t ps_bytes = ps_len * sizeof(uint64_t);
    uint64_t* d_prefix_sums[4];
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaMalloc(&d_prefix_sums[fi], ps_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_prefix_sums[fi], params.h_prefix_sums[fi],
                                   ps_bytes, cudaMemcpyHostToDevice));
    }

    int32_t* d_shipdate_stats = nullptr;
    if (params.h_shipdate_stats && params.nstats > 0) {
        const size_t stats_bytes = params.nstats * 2 * sizeof(int32_t);
        BAM_CUDA_CHECK(cudaMalloc(&d_shipdate_stats, stats_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_shipdate_stats, params.h_shipdate_stats,
                                   stats_bytes, cudaMemcpyHostToDevice));
    }

    uint32_t* d_comp_sizes[4] = {};
    uint64_t* d_comp_offsets[4] = {};
    for (int fi = 0; fi < 4; fi++) {
        if (params.compression_method[fi] != 0 &&
            params.h_compressed_page_sizes[fi] && params.h_compressed_offsets[fi]) {
            size_t sz_sizes = params.field_npages * sizeof(uint32_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes[fi], sz_sizes));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes[fi], params.h_compressed_page_sizes[fi],
                                       sz_sizes, cudaMemcpyHostToDevice));

            size_t sz_offsets = params.field_npages * sizeof(uint64_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], sz_offsets));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], params.h_compressed_offsets[fi],
                                       sz_offsets, cudaMemcpyHostToDevice));
        }
    }

    const uint32_t elems_per_slot = params.decomp_elems_per_slot;
    int32_t* d_decomp_buf = nullptr;
    size_t decomp_buf_size = (size_t)num_blocks * ENTRIES_PER_BLOCK_DBL
                           * elems_per_slot * sizeof(int32_t);
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp_buf, decomp_buf_size));

    BAMKernelMeta h_meta;
    for (int fi = 0; fi < 4; fi++) {
        h_meta.field_start_page_ids[fi] = params.field_start_page_ids[fi];
        h_meta.d_prefix_sums[fi] = d_prefix_sums[fi];
        h_meta.compression_method[fi] = params.compression_method[fi];
        h_meta.d_comp_sizes[fi] = d_comp_sizes[fi];
        h_meta.d_comp_offsets[fi] = d_comp_offsets[fi];
    }
    h_meta.field_npages = params.field_npages;
    h_meta.page_size = params.page_size;
    h_meta.blocks_per_page = params.blocks_per_page;
    h_meta.partition_start_lba = params.partition_start_lba;
    h_meta.n_devices = params.n_devices > 0 ? params.n_devices : 1;
    for (uint32_t d = 0; d < h_meta.n_devices; d++)
        h_meta.partition_start_lbas[d] = params.partition_start_lbas[d];
    h_meta.num_blocks = num_blocks;
    h_meta.nrows = params.nrows;
    h_meta.d_shipdate_stats = d_shipdate_stats;
    h_meta.nstats = params.nstats;
    h_meta.d_decomp_buf = d_decomp_buf;
    h_meta.decomp_elems_per_slot = elems_per_slot;
    h_meta.io_multi = 1;
    h_meta.sd_low = params.sd_low;
    h_meta.sd_high = params.sd_high;

    BAMKernelMeta* d_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_meta, sizeof(BAMKernelMeta)));
    BAM_CUDA_CHECK(cudaMemcpy(d_meta, &h_meta, sizeof(BAMKernelMeta),
                               cudaMemcpyHostToDevice));

    int64_t* d_revenue;
    BAM_CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    BAMDiagCounters* d_diag;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, sizeof(BAMDiagCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, sizeof(BAMDiagCounters)));

    BAMPerfCounters* d_perf;
    BAM_CUDA_CHECK(cudaMalloc(&d_perf, sizeof(BAMPerfCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_perf, 0, sizeof(BAMPerfCounters)));

    size_t old_stack_size = 0;
    cudaDeviceGetLimit(&old_stack_size, cudaLimitStackSize);
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    const uint32_t threads_per_block = 128;
    fprintf(stderr, "[bam_q6_128t_pf] Launching %u blocks x %u threads (4 warps/block, prefetch), "
            "decomp_buf=%zu MB, pc_pages=%lu\n",
            num_blocks, threads_per_block, decomp_buf_size / (1024 * 1024), n_pc_pages);

    bam_q6_kernel_128t_pf_sync<<<num_blocks, threads_per_block, 0, stream>>>(
        h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t h_revenue = 0;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_revenue, d_revenue, sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, stream));
    BAMDiagCounters h_diag;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_diag, d_diag, sizeof(BAMDiagCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAMPerfCounters h_perf;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_perf, d_perf, sizeof(BAMPerfCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr, "[diag] sum_shipdate=%lu  sum_quantity=%lu\n",
            h_diag.sum_shipdate, h_diag.sum_quantity);
    fprintf(stderr, "[diag] sum_extprice=%lu  sum_discount=%lu\n",
            h_diag.sum_extprice, h_diag.sum_discount);

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, params.cuda_device);
    double clock_ghz = clock_khz / 1e6;

    uint64_t total_cycles = h_perf.io_cycles + h_perf.decomp_cycles + h_perf.eval_cycles;
    fprintf(stderr, "\n[perf] GPU clock: %.3f GHz\n", clock_ghz);
    fprintf(stderr, "[perf] pages_processed: %lu  io_calls: %lu  io_bytes: %lu (%.2f MiB)\n",
            h_perf.page_count, h_perf.io_count,
            h_perf.io_bytes, h_perf.io_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[perf] boundary_page_reads: %lu (%.1f%% of pages)\n",
            h_perf.boundary_count,
            h_perf.page_count > 0 ? 100.0 * h_perf.boundary_count / h_perf.page_count : 0.0);
    fprintf(stderr, "[perf] Accumulated cycles across %u blocks (thread 0):\n", num_blocks);
    fprintf(stderr, "[perf]   io:      %12lu cycles  (%5.1f%%)\n",
            h_perf.io_cycles, total_cycles > 0 ? 100.0 * h_perf.io_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   decomp:  %12lu cycles  (%5.1f%%)\n",
            h_perf.decomp_cycles, total_cycles > 0 ? 100.0 * h_perf.decomp_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   eval:    %12lu cycles  (%5.1f%%)\n",
            h_perf.eval_cycles, total_cycles > 0 ? 100.0 * h_perf.eval_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   total:   %12lu cycles\n", total_cycles);
    if (h_perf.io_count > 0) {
        double avg_io_us = (double)h_perf.io_cycles / h_perf.io_count / (clock_ghz * 1e3);
        fprintf(stderr, "[perf]   avg_io_latency: %.1f us/call  (%lu calls)\n",
                avg_io_us, h_perf.io_count);
    }
    fprintf(stderr, "[perf]   prefetch: %lu submitted, %lu hits (%.1f%% hit rate)\n",
            h_perf.pf_submit_count, h_perf.pf_hit_count,
            h_perf.pf_submit_count > 0 ? 100.0 * h_perf.pf_hit_count / h_perf.pf_submit_count : 0.0);

    BAM_CUDA_CHECK(cudaStreamDestroy(stream));
    BAM_CUDA_CHECK(cudaFree(d_meta));
    BAM_CUDA_CHECK(cudaFree(d_revenue));
    BAM_CUDA_CHECK(cudaFree(d_diag));
    BAM_CUDA_CHECK(cudaFree(d_perf));
    BAM_CUDA_CHECK(cudaFree(d_decomp_buf));
    if (d_shipdate_stats) BAM_CUDA_CHECK(cudaFree(d_shipdate_stats));
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaFree(d_prefix_sums[fi]));
        if (d_comp_sizes[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_sizes[fi]));
        if (d_comp_offsets[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_offsets[fi]));
    }

    return BAMRunResult{h_revenue, h_perf.io_count, h_perf.io_bytes};
}

// ============================================================
// Host wrapper for 160t IO-exclusive kernel (Pattern 1)
// ============================================================
BAMRunResult bam_q6_160t_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl_handle) {
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    uint32_t num_blocks = params.num_blocks;
    const uint64_t page_size = params.page_size;

    // Double-buffered: 14 entries per block
    const uint64_t n_pc_pages = (uint64_t)num_blocks * ENTRIES_PER_BLOCK_DBL;
    const uint64_t max_range = 64;

    size_t gpu_free = 0, gpu_total = 0;
    BAM_CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));
    fprintf(stderr, "[bam_q6_160t] GPU memory: %.1f GB free / %.1f GB total\n",
            gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));

    page_cache_t h_pc(page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);
    page_cache_d_t* d_pc = h_pc.d_pc_ptr;

    const uint64_t ps_len = params.field_npages + 1;
    const size_t ps_bytes = ps_len * sizeof(uint64_t);
    uint64_t* d_prefix_sums[4];
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaMalloc(&d_prefix_sums[fi], ps_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_prefix_sums[fi], params.h_prefix_sums[fi],
                                   ps_bytes, cudaMemcpyHostToDevice));
    }

    int32_t* d_shipdate_stats = nullptr;
    if (params.h_shipdate_stats && params.nstats > 0) {
        const size_t stats_bytes = params.nstats * 2 * sizeof(int32_t);
        BAM_CUDA_CHECK(cudaMalloc(&d_shipdate_stats, stats_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_shipdate_stats, params.h_shipdate_stats,
                                   stats_bytes, cudaMemcpyHostToDevice));
    }

    uint32_t* d_comp_sizes[4] = {};
    uint64_t* d_comp_offsets[4] = {};
    for (int fi = 0; fi < 4; fi++) {
        if (params.compression_method[fi] != 0 &&
            params.h_compressed_page_sizes[fi] && params.h_compressed_offsets[fi]) {
            size_t sz_sizes = params.field_npages * sizeof(uint32_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes[fi], sz_sizes));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes[fi], params.h_compressed_page_sizes[fi],
                                       sz_sizes, cudaMemcpyHostToDevice));

            size_t sz_offsets = params.field_npages * sizeof(uint64_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], sz_offsets));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], params.h_compressed_offsets[fi],
                                       sz_offsets, cudaMemcpyHostToDevice));
        }
    }

    const uint32_t elems_per_slot = params.decomp_elems_per_slot;
    int32_t* d_decomp_buf = nullptr;
    size_t decomp_buf_size = (size_t)num_blocks * ENTRIES_PER_BLOCK_DBL
                           * elems_per_slot * sizeof(int32_t);
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp_buf, decomp_buf_size));

    BAMKernelMeta h_meta;
    for (int fi = 0; fi < 4; fi++) {
        h_meta.field_start_page_ids[fi] = params.field_start_page_ids[fi];
        h_meta.d_prefix_sums[fi] = d_prefix_sums[fi];
        h_meta.compression_method[fi] = params.compression_method[fi];
        h_meta.d_comp_sizes[fi] = d_comp_sizes[fi];
        h_meta.d_comp_offsets[fi] = d_comp_offsets[fi];
    }
    h_meta.field_npages = params.field_npages;
    h_meta.page_size = params.page_size;
    h_meta.blocks_per_page = params.blocks_per_page;
    h_meta.partition_start_lba = params.partition_start_lba;
    h_meta.n_devices = params.n_devices > 0 ? params.n_devices : 1;
    for (uint32_t d = 0; d < h_meta.n_devices; d++)
        h_meta.partition_start_lbas[d] = params.partition_start_lbas[d];
    h_meta.num_blocks = num_blocks;
    h_meta.nrows = params.nrows;
    h_meta.d_shipdate_stats = d_shipdate_stats;
    h_meta.nstats = params.nstats;
    h_meta.d_decomp_buf = d_decomp_buf;
    h_meta.decomp_elems_per_slot = elems_per_slot;
    h_meta.io_multi = 1;
    h_meta.sd_low = params.sd_low;
    h_meta.sd_high = params.sd_high;

    BAMKernelMeta* d_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_meta, sizeof(BAMKernelMeta)));
    BAM_CUDA_CHECK(cudaMemcpy(d_meta, &h_meta, sizeof(BAMKernelMeta),
                               cudaMemcpyHostToDevice));

    int64_t* d_revenue;
    BAM_CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    BAMDiagCounters* d_diag;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, sizeof(BAMDiagCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, sizeof(BAMDiagCounters)));

    BAMPerfCounters* d_perf;
    BAM_CUDA_CHECK(cudaMalloc(&d_perf, sizeof(BAMPerfCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_perf, 0, sizeof(BAMPerfCounters)));

    size_t old_stack_size = 0;
    cudaDeviceGetLimit(&old_stack_size, cudaLimitStackSize);
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    const uint32_t threads_per_block = 160;
    fprintf(stderr, "[bam_q6_160t] Launching %u blocks x %u threads (5 warps/block, warp0=IO), "
            "decomp_buf=%zu MB, pc_pages=%lu\n",
            num_blocks, threads_per_block, decomp_buf_size / (1024 * 1024), n_pc_pages);

    bam_q6_kernel_160t_sync<<<num_blocks, threads_per_block, 0, stream>>>(
        h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t h_revenue = 0;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_revenue, d_revenue, sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, stream));
    BAMDiagCounters h_diag;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_diag, d_diag, sizeof(BAMDiagCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAMPerfCounters h_perf;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_perf, d_perf, sizeof(BAMPerfCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr, "[diag] sum_shipdate=%lu  sum_quantity=%lu\n",
            h_diag.sum_shipdate, h_diag.sum_quantity);
    fprintf(stderr, "[diag] sum_extprice=%lu  sum_discount=%lu\n",
            h_diag.sum_extprice, h_diag.sum_discount);

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, params.cuda_device);
    double clock_ghz = clock_khz / 1e6;

    uint64_t total_cycles = h_perf.io_cycles + h_perf.decomp_cycles + h_perf.eval_cycles;
    fprintf(stderr, "\n[perf] GPU clock: %.3f GHz\n", clock_ghz);
    fprintf(stderr, "[perf] pages_processed: %lu  io_calls: %lu  io_bytes: %lu (%.2f MiB)\n",
            h_perf.page_count, h_perf.io_count,
            h_perf.io_bytes, h_perf.io_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[perf] boundary_page_reads: %lu (%.1f%% of pages)\n",
            h_perf.boundary_count,
            h_perf.page_count > 0 ? 100.0 * h_perf.boundary_count / h_perf.page_count : 0.0);
    fprintf(stderr, "[perf] Accumulated cycles across %u blocks (thread 0):\n", num_blocks);
    fprintf(stderr, "[perf]   io:      %12lu cycles  (%5.1f%%)\n",
            h_perf.io_cycles, total_cycles > 0 ? 100.0 * h_perf.io_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   decomp:  %12lu cycles  (%5.1f%%)\n",
            h_perf.decomp_cycles, total_cycles > 0 ? 100.0 * h_perf.decomp_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   eval:    %12lu cycles  (%5.1f%%)\n",
            h_perf.eval_cycles, total_cycles > 0 ? 100.0 * h_perf.eval_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   total:   %12lu cycles\n", total_cycles);
    if (h_perf.io_count > 0) {
        double avg_io_us = (double)h_perf.io_cycles / h_perf.io_count / (clock_ghz * 1e3);
        fprintf(stderr, "[perf]   avg_io_latency: %.1f us/call  (%lu calls)\n",
                avg_io_us, h_perf.io_count);
    }
    fprintf(stderr, "[perf]   prefetch: %lu submitted, %lu hits (%.1f%% hit rate)\n",
            h_perf.pf_submit_count, h_perf.pf_hit_count,
            h_perf.pf_submit_count > 0 ? 100.0 * h_perf.pf_hit_count / h_perf.pf_submit_count : 0.0);

    BAM_CUDA_CHECK(cudaStreamDestroy(stream));
    BAM_CUDA_CHECK(cudaFree(d_meta));
    BAM_CUDA_CHECK(cudaFree(d_revenue));
    BAM_CUDA_CHECK(cudaFree(d_diag));
    BAM_CUDA_CHECK(cudaFree(d_perf));
    BAM_CUDA_CHECK(cudaFree(d_decomp_buf));
    if (d_shipdate_stats) BAM_CUDA_CHECK(cudaFree(d_shipdate_stats));
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaFree(d_prefix_sums[fi]));
        if (d_comp_sizes[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_sizes[fi]));
        if (d_comp_offsets[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_offsets[fi]));
    }

    return BAMRunResult{h_revenue, h_perf.io_count, h_perf.io_bytes};
}

// ============================================================
// Work Queue kernel: IO blocks + Compute blocks with ring buffer
// ============================================================
__global__ void bam_q6_kernel_wq_sync(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    uint32_t n_io_blocks,
    uint32_t ring_size,
    volatile uint32_t* d_slot_state,
    WQSlotMeta* d_slot_meta,
    uint32_t* d_page_head,
    uint32_t* d_qual_head,
    uint32_t* d_comp_head,
    uint32_t* d_io_done_count,
    uint32_t* d_n_qual_final,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const bool is_io_block = (block_id < n_io_blocks);

    const uint64_t npages = meta->field_npages;
    const uint32_t page_size = meta->page_size;
    const uint16_t* comp = meta->compression_method;
    const uint32_t blocks_per_page = meta->blocks_per_page;
    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;
    const uint64_t* ps0 = meta->d_prefix_sums[0];
    const uint64_t* ps1 = meta->d_prefix_sums[1];
    const uint64_t* ps2 = meta->d_prefix_sums[2];
    const uint64_t* ps3 = meta->d_prefix_sums[3];
    char* base = (char*)pc->base_addr;
    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;

    // Per-block profiling accumulators (thread 0 only)
    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    if (is_io_block) {
        // ════════════════════════════════════════════════
        // IO BLOCK: warp 0 only, scan pages and fill ring
        // ════════════════════════════════════════════════
        if (warp_id != 0) return;  // warps 1-3 exit immediately

        QueuePair* qp = ctrls[0]->d_qps + (block_id % ctrls[0]->n_qps);

        while (true) {
            // Claim next SD page
            uint32_t pg;
            if (lane_id == 0) pg = atomicAdd(d_page_head, 1);
            pg = __shfl_sync(0xFFFFFFFF, pg, 0);
            if (pg >= npages) break;

            // Zone map check
            int skip = 0;
            if (stats && pg < nstats) {
                int32_t page_min = stats[pg * 2];
                int32_t page_max = stats[pg * 2 + 1];
                if (page_max < meta->sd_low || page_min > meta->sd_high - 1)
                    skip = 1;
            }
            skip = __shfl_sync(0xFFFFFFFF, skip, 0);
            if (skip) continue;

            // Empty page check
            uint64_t row_begin_chk = ps0[pg];
            uint64_t row_end_chk = ps0[pg + 1];
            if (row_end_chk == row_begin_chk) continue;

            // Pack IO descriptors (entry_base=0 placeholder, io_entry will be rebuilt)
            uint64_t io_lba[7];
            unsigned long long io_entry[7];
            uint32_t io_nblocks[7];
            uint64_t fi_pg[3], fi_bound[3];
            int needs_boundary[3];
            int all_comp, has_boundary, boundary_cnt;
            uint64_t row_begin, row_end;

            int n_ios = pack_io_descriptors(
                pg, meta, ps0, ps1, ps2, ps3, 0ULL,
                io_lba, io_entry, io_nblocks,
                fi_pg, fi_bound, needs_boundary,
                &all_comp, &has_boundary, &boundary_cnt,
                &row_begin, &row_end);

            if (n_ios == 0) continue;

            // Claim qualifying page index
            uint32_t qi;
            if (lane_id == 0) qi = atomicAdd(d_qual_head, 1);
            qi = __shfl_sync(0xFFFFFFFF, qi, 0);
            uint32_t slot = qi % ring_size;

            // Wait for slot to be FREE
            if (lane_id == 0) {
                while (atomicAdd(&((uint32_t*)d_slot_state)[slot], 0) != 0) {}
            }
            __syncwarp();

            // Rebuild io_entry with correct entry_base for this slot
            unsigned long long entry_base = (unsigned long long)slot * ENTRIES_PER_BLOCK;
            io_entry[0] = entry_base + 0;  // SD
            io_entry[1] = entry_base + 1;  // QT_A
            io_entry[2] = entry_base + 3;  // EP_A
            io_entry[3] = entry_base + 5;  // DC_A
            int bi = 4;
            if (needs_boundary[0]) io_entry[bi++] = entry_base + 2;  // QT_B
            if (needs_boundary[1]) io_entry[bi++] = entry_base + 4;  // EP_B
            if (needs_boundary[2]) io_entry[bi++] = entry_base + 6;  // DC_B

            // Submit NVMe reads
            long long t0 = clock64();
            uint16_t cid = 0, sq_pos = 0;
            if ((int)lane_id < n_ios) {
                access_data_async(pc, qp, io_lba[lane_id], io_nblocks[lane_id],
                                  io_entry[lane_id], NVM_IO_READ, &cid, &sq_pos);
            }
            __syncwarp();

            // Poll for completion
            if ((int)lane_id < n_ios) {
                uint32_t poll_loc, poll_head;
                uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq);
                put_cid(&qp->sq, cid);
            }
            __syncwarp();
            long long t1 = clock64();
            blk_io_cycles += (uint64_t)(t1 - t0);
            blk_io_count += n_ios;
            for (int ii = 0; ii < n_ios; ii++)
                blk_io_bytes += (uint64_t)io_nblocks[ii] * 512;
            blk_boundary_count += boundary_cnt;
            blk_page_count++;

            // Store metadata for compute block
            if (lane_id == 0) {
                d_slot_meta[slot].row_begin = row_begin;
                d_slot_meta[slot].row_end = row_end;
                d_slot_meta[slot].all_comp = all_comp;
                d_slot_meta[slot].has_boundary = has_boundary;
                for (int f = 0; f < 3; f++) {
                    d_slot_meta[slot].fi_pg[f] = fi_pg[f];
                    d_slot_meta[slot].fi_bound[f] = fi_bound[f];
                    d_slot_meta[slot].needs_boundary[f] = needs_boundary[f];
                }
                __threadfence();
                atomicExch((uint32_t*)&d_slot_state[slot], 1);  // READY
            }
            __syncwarp();
        }

        // IO block done scanning
        if (lane_id == 0) {
            uint32_t done = atomicAdd(d_io_done_count, 1) + 1;
            if (done == n_io_blocks) {
                // Last IO block: publish final qualifying count
                uint32_t final_qual = atomicAdd(d_qual_head, 0);
                __threadfence();
                atomicExch(d_n_qual_final, final_qual);
            }
        }

        // Flush perf counters
        if (lane_id == 0 && d_perf) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                      (unsigned long long)blk_io_cycles);
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                      (unsigned long long)blk_page_count);
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                      (unsigned long long)blk_io_count);
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                      (unsigned long long)blk_io_bytes);
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                      (unsigned long long)blk_boundary_count);
        }
    } else {
        // ════════════════════════════════════════════════
        // COMPUTE BLOCK: 128 threads, dequeue from ring
        // ════════════════════════════════════════════════

        // Shared memory for Case A PFOR decode
        __shared__ uint32_t s_comp_blk[4 * 136];  // 4 warps × 136 uint32
        __shared__ uint s_bws[4 * 4];
        __shared__ uint s_offs[4 * 4];
        __shared__ int32_t s_vals[4 * 128];  // 4 fields × 128 elements

        // Shared metadata
        __shared__ uint32_t s_nblocks_pfor;
        __shared__ uint32_t s_nalloc;

        // Reduction
        __shared__ int64_t s_warp_rev[4];
        __shared__ uint64_t s_warp_sum[4 * 4];

        // Shared for work claim
        __shared__ uint32_t s_qi;
        __shared__ int s_terminated;

        // Field-to-entry mapping for Case A (same as 128t kernel)
        const int field_entry_off[4] = {0, 1, 3, 5};

        while (true) {
            // Claim next work item
            if (tid == 0) {
                s_qi = atomicAdd(d_comp_head, 1);
                s_terminated = 0;
            }
            __syncthreads();
            uint32_t qi = s_qi;

            uint32_t slot = qi % ring_size;

            // Wait for slot READY or termination
            if (tid == 0) {
                while (atomicAdd((uint32_t*)&d_slot_state[slot], 0) != 1) {
                    // Check termination
                    uint32_t nqf = atomicAdd(d_n_qual_final, 0);
                    if (nqf != 0xFFFFFFFF && qi >= nqf) {
                        s_terminated = 1;
                        break;
                    }
                }
                if (!s_terminated) __threadfence();
            }
            __syncthreads();
            if (s_terminated) break;

            // Read metadata
            unsigned long long entry_base = (unsigned long long)slot * ENTRIES_PER_BLOCK;
            int32_t* decomp_base = meta->d_decomp_buf
                + (uint64_t)slot * ENTRIES_PER_BLOCK * elems_per_slot;

            uint64_t row_begin = d_slot_meta[slot].row_begin;
            uint64_t row_end = d_slot_meta[slot].row_end;
            int all_comp = d_slot_meta[slot].all_comp;
            int has_boundary = d_slot_meta[slot].has_boundary;

            int64_t local_rev = 0;
            uint64_t local_sum_sd = 0, local_sum_qt = 0;
            uint64_t local_sum_ep = 0, local_sum_dc = 0;

            long long td0 = clock64();

            if (all_comp && !has_boundary) {
                // ── Case A: 4 warps decode 4 fields in parallel ──
                char* my_page_ptr = base + (entry_base + field_entry_off[warp_id]) * page_size;

                if (warp_id == 0 && lane_id == 0) {
                    bam_pag_head* hdr0 = (bam_pag_head*)(base + entry_base * page_size);
                    s_nblocks_pfor = hdr0->watermark / 128;
                    s_nalloc = hdr0->nalloc;
                }
                __syncthreads();

                uint32_t nblocks_pfor = s_nblocks_pfor;
                uint32_t nalloc = s_nalloc;

                bam_pag_head* my_hdr = (bam_pag_head*)my_page_ptr;
                uint32_t* my_blk_start = (uint32_t*)(my_page_ptr + BAM_PAG_HDR_BYTES);
                uint32_t* my_data_ptr = my_blk_start + (nblocks_pfor + 1);
                uint32_t* my_comp_blk = s_comp_blk + warp_id * 136;
                uint* my_bws = s_bws + warp_id * 4;
                uint* my_offs = s_offs + warp_id * 4;

                for (uint32_t b = 0; b < nblocks_pfor; b++) {
                    int32_t field_vals[4];
                    decode_pfor_block_smem(
                        my_data_ptr + my_blk_start[b],
                        my_blk_start[b + 1] - my_blk_start[b],
                        my_comp_blk, my_bws, my_offs,
                        lane_id, field_vals);

                    for (int k = 0; k < 4; k++) {
                        s_vals[warp_id * 128 + k * 32 + lane_id] = field_vals[k];
                    }
                    __syncthreads();

                    // Eval: all 128 threads
                    uint32_t idx = b * 128 + tid;
                    if (idx < nalloc) {
                        int32_t sd_v = s_vals[0 * 128 + tid];
                        int32_t qt_v = s_vals[1 * 128 + tid];
                        int32_t ep_v = s_vals[2 * 128 + tid];
                        int32_t dc_v = s_vals[3 * 128 + tid];

                        local_sum_sd += (uint32_t)sd_v;
                        local_sum_qt += (uint32_t)qt_v;
                        local_sum_ep += (uint32_t)ep_v;
                        local_sum_dc += (uint32_t)dc_v;

                        if (sd_v >= meta->sd_low && sd_v < meta->sd_high &&
                            dc_v >= 5 && dc_v <= 7 &&
                            qt_v < 2400) {
                            local_rev += (int64_t)ep_v * dc_v;
                        }
                    }
                    __syncthreads();
                }
            } else {
                // ── Case B: fallback with decompression to global buffer ──
                __shared__ uint shared_bws_b[4];
                __shared__ uint shared_offs_b[4];
                __shared__ uint64_t s_fi_pg[3];
                __shared__ uint64_t s_fi_bound[3];
                __shared__ int s_needs_boundary[3];

                if (tid == 0) {
                    for (int f = 0; f < 3; f++) {
                        s_fi_pg[f] = d_slot_meta[slot].fi_pg[f];
                        s_fi_bound[f] = d_slot_meta[slot].fi_bound[f];
                        s_needs_boundary[f] = d_slot_meta[slot].needs_boundary[f];
                    }
                }
                __syncthreads();

                // Warp 0 decompresses all fields
                if (warp_id == 0) {
                    decompress_page_warp(base + (entry_base + 0) * page_size,
                        decomp_base + 0 * elems_per_slot,
                        comp[0], shared_bws_b, shared_offs_b, lane_id, nullptr);
                    decompress_page_warp(base + (entry_base + 1) * page_size,
                        decomp_base + 1 * elems_per_slot,
                        comp[1], shared_bws_b, shared_offs_b, lane_id, nullptr);
                    decompress_page_warp(base + (entry_base + 3) * page_size,
                        decomp_base + 3 * elems_per_slot,
                        comp[2], shared_bws_b, shared_offs_b, lane_id, nullptr);
                    decompress_page_warp(base + (entry_base + 5) * page_size,
                        decomp_base + 5 * elems_per_slot,
                        comp[3], shared_bws_b, shared_offs_b, lane_id, nullptr);

                    if (s_needs_boundary[0]) {
                        decompress_page_warp(base + (entry_base + 2) * page_size,
                            decomp_base + 2 * elems_per_slot,
                            comp[1], shared_bws_b, shared_offs_b, lane_id, nullptr);
                    }
                    if (s_needs_boundary[1]) {
                        decompress_page_warp(base + (entry_base + 4) * page_size,
                            decomp_base + 4 * elems_per_slot,
                            comp[2], shared_bws_b, shared_offs_b, lane_id, nullptr);
                    }
                    if (s_needs_boundary[2]) {
                        decompress_page_warp(base + (entry_base + 6) * page_size,
                            decomp_base + 6 * elems_per_slot,
                            comp[3], shared_bws_b, shared_offs_b, lane_id, nullptr);
                    }
                }
                __syncthreads();

                // Eval: all 128 threads
                int32_t* sd_decomp = decomp_base + 0 * elems_per_slot;
                int32_t* qt_a = decomp_base + 1 * elems_per_slot;
                int32_t* qt_b = decomp_base + 2 * elems_per_slot;
                int32_t* ep_a = decomp_base + 3 * elems_per_slot;
                int32_t* ep_b = decomp_base + 4 * elems_per_slot;
                int32_t* dc_a = decomp_base + 5 * elems_per_slot;
                int32_t* dc_b = decomp_base + 6 * elems_per_slot;

                uint64_t qt_base_off = ps1[s_fi_pg[0]];
                uint64_t ep_base_off = ps2[s_fi_pg[1]];
                uint64_t dc_base_off = ps3[s_fi_pg[2]];

                uint64_t nrows_this = row_end - row_begin;
                for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 128) {
                    uint64_t gr = row_begin + i;
                    int32_t l_shipdate = sd_decomp[i];
                    int32_t l_quantity = (gr < s_fi_bound[0])
                        ? qt_a[gr - qt_base_off] : qt_b[gr - s_fi_bound[0]];
                    int32_t l_extendedprice = (gr < s_fi_bound[1])
                        ? ep_a[gr - ep_base_off] : ep_b[gr - s_fi_bound[1]];
                    int32_t l_discount = (gr < s_fi_bound[2])
                        ? dc_a[gr - dc_base_off] : dc_b[gr - s_fi_bound[2]];

                    local_sum_sd += (uint32_t)l_shipdate;
                    local_sum_qt += (uint32_t)l_quantity;
                    local_sum_ep += (uint32_t)l_extendedprice;
                    local_sum_dc += (uint32_t)l_discount;

                    if (l_shipdate >= meta->sd_low && l_shipdate < meta->sd_high &&
                        l_discount >= 5 && l_discount <= 7 &&
                        l_quantity < 2400) {
                        local_rev += (int64_t)l_extendedprice * l_discount;
                    }
                }
            }

            long long td1 = clock64();
            blk_decomp_cycles += (uint64_t)(td1 - td0);
            blk_page_count++;

            // Reduction: warp shuffle → shared → atomicAdd
            for (int offset = 16; offset > 0; offset /= 2) {
                local_rev    += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
                local_sum_sd += __shfl_down_sync(0xFFFFFFFF, local_sum_sd, offset);
                local_sum_qt += __shfl_down_sync(0xFFFFFFFF, local_sum_qt, offset);
                local_sum_ep += __shfl_down_sync(0xFFFFFFFF, local_sum_ep, offset);
                local_sum_dc += __shfl_down_sync(0xFFFFFFFF, local_sum_dc, offset);
            }
            if (lane_id == 0) {
                s_warp_rev[warp_id] = local_rev;
                s_warp_sum[warp_id * 4 + 0] = local_sum_sd;
                s_warp_sum[warp_id * 4 + 1] = local_sum_qt;
                s_warp_sum[warp_id * 4 + 2] = local_sum_ep;
                s_warp_sum[warp_id * 4 + 3] = local_sum_dc;
            }
            __syncthreads();

            if (tid == 0) {
                int64_t block_rev = s_warp_rev[0] + s_warp_rev[1] + s_warp_rev[2] + s_warp_rev[3];
                atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                          static_cast<unsigned long long>(block_rev));
                if (d_diag) {
                    uint64_t bsd = s_warp_sum[0] + s_warp_sum[4] + s_warp_sum[8] + s_warp_sum[12];
                    uint64_t bqt = s_warp_sum[1] + s_warp_sum[5] + s_warp_sum[9] + s_warp_sum[13];
                    uint64_t bep = s_warp_sum[2] + s_warp_sum[6] + s_warp_sum[10] + s_warp_sum[14];
                    uint64_t bdc = s_warp_sum[3] + s_warp_sum[7] + s_warp_sum[11] + s_warp_sum[15];
                    atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_shipdate),
                              (unsigned long long)bsd);
                    atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_quantity),
                              (unsigned long long)bqt);
                    atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_extprice),
                              (unsigned long long)bep);
                    atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag->sum_discount),
                              (unsigned long long)bdc);
                }
            }
            __syncthreads();

            // Mark slot FREE for reuse
            if (tid == 0) {
                __threadfence();
                atomicExch((uint32_t*)&d_slot_state[slot], 0);  // FREE
            }
            __syncthreads();
        }

        // Flush perf counters
        if (tid == 0 && d_perf) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                      (unsigned long long)blk_decomp_cycles);
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                      (unsigned long long)blk_page_count);
        }
    }
}

// ============================================================
// Host wrapper for work queue Q6 kernel
// ============================================================
BAMRunResult bam_q6_wq_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl_handle) {
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    uint32_t num_blocks = params.num_blocks;
    const uint64_t page_size = params.page_size;
    const uint32_t ring_size = WQ_RING_SIZE;
    uint32_t n_io_blocks = std::max(1u, (uint32_t)params.io_multiplicity);
    if (n_io_blocks >= num_blocks) n_io_blocks = num_blocks / 2;
    uint32_t n_comp_blocks = num_blocks - n_io_blocks;

    // ── Page cache: ring_size * 7 entries (shared across IO/compute) ──
    const uint64_t n_pc_pages = (uint64_t)ring_size * ENTRIES_PER_BLOCK;
    const uint64_t max_range = 64;

    size_t gpu_free = 0, gpu_total = 0;
    BAM_CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));
    fprintf(stderr, "[bam_q6_wq] GPU memory: %.1f GB free / %.1f GB total\n",
            gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));

    page_cache_t h_pc(page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);
    page_cache_d_t* d_pc = h_pc.d_pc_ptr;

    // ── Copy prefix sum arrays to GPU ──
    const uint64_t ps_len = params.field_npages + 1;
    const size_t ps_bytes = ps_len * sizeof(uint64_t);
    uint64_t* d_prefix_sums[4];
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaMalloc(&d_prefix_sums[fi], ps_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_prefix_sums[fi], params.h_prefix_sums[fi],
                                   ps_bytes, cudaMemcpyHostToDevice));
    }

    // ── Copy zone map stats to GPU ──
    int32_t* d_shipdate_stats = nullptr;
    if (params.h_shipdate_stats && params.nstats > 0) {
        const size_t stats_bytes = params.nstats * 2 * sizeof(int32_t);
        BAM_CUDA_CHECK(cudaMalloc(&d_shipdate_stats, stats_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_shipdate_stats, params.h_shipdate_stats,
                                   stats_bytes, cudaMemcpyHostToDevice));
    }

    // ── Copy compression metadata to GPU ──
    uint32_t* d_comp_sizes[4] = {};
    uint64_t* d_comp_offsets[4] = {};
    for (int fi = 0; fi < 4; fi++) {
        if (params.compression_method[fi] != 0 &&
            params.h_compressed_page_sizes[fi] && params.h_compressed_offsets[fi]) {
            size_t sz_sizes = params.field_npages * sizeof(uint32_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes[fi], sz_sizes));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes[fi], params.h_compressed_page_sizes[fi],
                                       sz_sizes, cudaMemcpyHostToDevice));

            size_t sz_offsets = params.field_npages * sizeof(uint64_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], sz_offsets));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], params.h_compressed_offsets[fi],
                                       sz_offsets, cudaMemcpyHostToDevice));
        }
    }

    // ── Decompression output buffer (for Case B fallback) ──
    const uint32_t elems_per_slot = params.decomp_elems_per_slot;
    int32_t* d_decomp_buf = nullptr;
    size_t decomp_buf_size = (size_t)ring_size * ENTRIES_PER_BLOCK
                           * elems_per_slot * sizeof(int32_t);
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp_buf, decomp_buf_size));

    // ── GPU metadata ──
    BAMKernelMeta h_meta;
    for (int fi = 0; fi < 4; fi++) {
        h_meta.field_start_page_ids[fi] = params.field_start_page_ids[fi];
        h_meta.d_prefix_sums[fi] = d_prefix_sums[fi];
        h_meta.compression_method[fi] = params.compression_method[fi];
        h_meta.d_comp_sizes[fi] = d_comp_sizes[fi];
        h_meta.d_comp_offsets[fi] = d_comp_offsets[fi];
    }
    h_meta.field_npages = params.field_npages;
    h_meta.page_size = params.page_size;
    h_meta.blocks_per_page = params.blocks_per_page;
    h_meta.partition_start_lba = params.partition_start_lba;
    h_meta.n_devices = params.n_devices > 0 ? params.n_devices : 1;
    for (uint32_t d = 0; d < h_meta.n_devices; d++)
        h_meta.partition_start_lbas[d] = params.partition_start_lbas[d];
    h_meta.num_blocks = num_blocks;
    h_meta.nrows = params.nrows;
    h_meta.d_shipdate_stats = d_shipdate_stats;
    h_meta.nstats = params.nstats;
    h_meta.d_decomp_buf = d_decomp_buf;
    h_meta.decomp_elems_per_slot = elems_per_slot;
    h_meta.io_multi = 1;
    h_meta.sd_low = params.sd_low;
    h_meta.sd_high = params.sd_high;

    BAMKernelMeta* d_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_meta, sizeof(BAMKernelMeta)));
    BAM_CUDA_CHECK(cudaMemcpy(d_meta, &h_meta, sizeof(BAMKernelMeta),
                               cudaMemcpyHostToDevice));

    // ── Result buffers ──
    int64_t* d_revenue;
    BAM_CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    BAMDiagCounters* d_diag;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, sizeof(BAMDiagCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, sizeof(BAMDiagCounters)));

    BAMPerfCounters* d_perf;
    BAM_CUDA_CHECK(cudaMalloc(&d_perf, sizeof(BAMPerfCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_perf, 0, sizeof(BAMPerfCounters)));

    // ── Ring buffer state ──
    volatile uint32_t* d_slot_state;
    BAM_CUDA_CHECK(cudaMalloc((void**)&d_slot_state, ring_size * sizeof(uint32_t)));
    BAM_CUDA_CHECK(cudaMemset((void*)d_slot_state, 0, ring_size * sizeof(uint32_t)));

    WQSlotMeta* d_slot_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_slot_meta, ring_size * sizeof(WQSlotMeta)));

    uint32_t* d_page_head;
    BAM_CUDA_CHECK(cudaMalloc(&d_page_head, sizeof(uint32_t)));
    BAM_CUDA_CHECK(cudaMemset(d_page_head, 0, sizeof(uint32_t)));

    uint32_t* d_qual_head;
    BAM_CUDA_CHECK(cudaMalloc(&d_qual_head, sizeof(uint32_t)));
    BAM_CUDA_CHECK(cudaMemset(d_qual_head, 0, sizeof(uint32_t)));

    uint32_t* d_comp_head;
    BAM_CUDA_CHECK(cudaMalloc(&d_comp_head, sizeof(uint32_t)));
    BAM_CUDA_CHECK(cudaMemset(d_comp_head, 0, sizeof(uint32_t)));

    uint32_t* d_io_done_count;
    BAM_CUDA_CHECK(cudaMalloc(&d_io_done_count, sizeof(uint32_t)));
    BAM_CUDA_CHECK(cudaMemset(d_io_done_count, 0, sizeof(uint32_t)));

    uint32_t* d_n_qual_final;
    BAM_CUDA_CHECK(cudaMalloc(&d_n_qual_final, sizeof(uint32_t)));
    uint32_t init_val = 0xFFFFFFFF;
    BAM_CUDA_CHECK(cudaMemcpy(d_n_qual_final, &init_val, sizeof(uint32_t),
                               cudaMemcpyHostToDevice));

    // ── Stack size for pack_io_descriptors ──
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    // ── Kernel launch ──
    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    const uint32_t threads_per_block = 128;
    fprintf(stderr, "[bam_q6_wq] Launching %u blocks x %u threads "
            "(%u IO blocks, %u compute blocks), ring_size=%u, "
            "decomp_buf=%zu MB, pc_pages=%lu\n",
            num_blocks, threads_per_block,
            n_io_blocks, n_comp_blocks, ring_size,
            decomp_buf_size / (1024 * 1024), n_pc_pages);

    bam_q6_kernel_wq_sync<<<num_blocks, threads_per_block, 0, stream>>>(
        h_pc.pdt.d_ctrls, d_pc, d_meta,
        n_io_blocks, ring_size,
        d_slot_state, d_slot_meta,
        d_page_head, d_qual_head, d_comp_head,
        d_io_done_count, d_n_qual_final,
        d_revenue, d_diag, d_perf);
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Results ──
    int64_t h_revenue = 0;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_revenue, d_revenue, sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, stream));
    BAMDiagCounters h_diag;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_diag, d_diag, sizeof(BAMDiagCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAMPerfCounters h_perf;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_perf, d_perf, sizeof(BAMPerfCounters),
                                    cudaMemcpyDeviceToHost, stream));

    uint32_t h_qual_count = 0;
    BAM_CUDA_CHECK(cudaMemcpy(&h_qual_count, d_n_qual_final, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr, "[diag] sum_shipdate=%lu  sum_quantity=%lu\n",
            h_diag.sum_shipdate, h_diag.sum_quantity);
    fprintf(stderr, "[diag] sum_extprice=%lu  sum_discount=%lu\n",
            h_diag.sum_extprice, h_diag.sum_discount);

    // Print profiling
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, params.cuda_device);
    double clock_ghz = clock_khz / 1e6;

    uint64_t total_cycles = h_perf.io_cycles + h_perf.decomp_cycles + h_perf.eval_cycles;
    fprintf(stderr, "\n[perf] GPU clock: %.3f GHz\n", clock_ghz);
    fprintf(stderr, "[perf] qualifying_pages: %u / %lu  (%.1f%%)\n",
            h_qual_count, params.field_npages,
            100.0 * h_qual_count / params.field_npages);
    fprintf(stderr, "[perf] pages_processed: %lu  io_calls: %lu  io_bytes: %lu (%.2f MiB)\n",
            h_perf.page_count, h_perf.io_count,
            h_perf.io_bytes, h_perf.io_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[perf] boundary_page_reads: %lu (%.1f%% of pages)\n",
            h_perf.boundary_count,
            h_perf.page_count > 0 ? 100.0 * h_perf.boundary_count / h_perf.page_count : 0.0);
    fprintf(stderr, "[perf] Accumulated cycles (%u IO blocks, %u compute blocks):\n",
            n_io_blocks, n_comp_blocks);
    fprintf(stderr, "[perf]   io:      %12lu cycles  (%5.1f%%)\n",
            h_perf.io_cycles, total_cycles > 0 ? 100.0 * h_perf.io_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   decomp:  %12lu cycles  (%5.1f%%)\n",
            h_perf.decomp_cycles, total_cycles > 0 ? 100.0 * h_perf.decomp_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   total:   %12lu cycles\n", total_cycles);
    if (h_perf.io_count > 0) {
        double avg_io_us = (double)h_perf.io_cycles / h_perf.io_count / (clock_ghz * 1e3);
        fprintf(stderr, "[perf]   avg_io_latency: %.1f us/call  (%lu calls)\n",
                avg_io_us, h_perf.io_count);
    }

    BAM_CUDA_CHECK(cudaStreamDestroy(stream));

    // ── Cleanup ──
    BAM_CUDA_CHECK(cudaFree(d_meta));
    BAM_CUDA_CHECK(cudaFree(d_revenue));
    BAM_CUDA_CHECK(cudaFree(d_diag));
    BAM_CUDA_CHECK(cudaFree(d_perf));
    BAM_CUDA_CHECK(cudaFree(d_decomp_buf));
    BAM_CUDA_CHECK(cudaFree((void*)d_slot_state));
    BAM_CUDA_CHECK(cudaFree(d_slot_meta));
    BAM_CUDA_CHECK(cudaFree(d_page_head));
    BAM_CUDA_CHECK(cudaFree(d_qual_head));
    BAM_CUDA_CHECK(cudaFree(d_comp_head));
    BAM_CUDA_CHECK(cudaFree(d_io_done_count));
    BAM_CUDA_CHECK(cudaFree(d_n_qual_final));
    if (d_shipdate_stats) BAM_CUDA_CHECK(cudaFree(d_shipdate_stats));
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaFree(d_prefix_sums[fi]));
        if (d_comp_sizes[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_sizes[fi]));
        if (d_comp_offsets[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_offsets[fi]));
    }

    return BAMRunResult{h_revenue, h_perf.io_count, h_perf.io_bytes};
}

// ============================================================
// Host wrapper for coalesced Q6 kernel (with io_multi pipelining)
// ============================================================
BAMRunResult bam_q6_coalesced_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl_handle) {
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    const uint32_t k = std::max(1u, params.coalesce_k);
    const uint64_t orig_page_size = params.page_size;
    const uint64_t pc_page_size = orig_page_size * k;
    const uint32_t io_multi = std::max(1u, std::min(params.io_multiplicity, (uint32_t)MAX_IO_MULTI));

    // Host-side prefix sum identity check (replaces GPU-side assertion)
    {
        const uint64_t ps_len = params.field_npages + 1;
        for (uint64_t p = 0; p < ps_len; p++) {
            for (int f = 1; f < 4; f++) {
                if (params.h_prefix_sums[f][p] != params.h_prefix_sums[0][p]) {
                    fprintf(stderr, "[FATAL] coalesced kernel requires identical prefix sums.\n"
                                    "  prefix_sum[%d][%lu] = %lu != prefix_sum[0][%lu] = %lu\n",
                            f, (unsigned long)p,
                            (unsigned long)params.h_prefix_sums[f][p],
                            (unsigned long)p,
                            (unsigned long)params.h_prefix_sums[0][p]);
                    exit(EXIT_FAILURE);
                }
            }
        }
        fprintf(stderr, "[bam_q6_coal] Prefix sum identity verified (%lu entries).\n",
                (unsigned long)ps_len);
    }

    // Cap num_blocks at number of groups
    uint32_t n_groups = (uint32_t)((params.field_npages + k - 1) / k);
    uint32_t num_blocks = std::min(params.num_blocks, n_groups);

    // Per-slot QPs: within a block, each io_multi slot must map to a distinct QP
    // to avoid CQ interleaving deadlock (sequential slot processing within a warp).
    // Cross-block QP sharing is safe: different blocks run concurrently on separate
    // SMs and independently dequeue their completions.
    // Formula: qp_index = (block_id * io_multi + slot) % n_qps
    // Within-block uniqueness is guaranteed when io_multi <= n_qps (always true).
    uint32_t n_qps_avail = params.num_queues;
    if (io_multi > n_qps_avail) {
        fprintf(stderr, "[bam_q6_coal] io_multi=%u exceeds n_qps=%u, capping\n",
                io_multi, n_qps_avail);
    }

    // Page cache: 4 entries per slot × io_multi slots per block
    const uint64_t n_pc_pages = (uint64_t)num_blocks * COALESCED_ENTRIES_PER_SLOT * io_multi;
    const uint64_t max_range = 64;

    size_t gpu_free = 0, gpu_total = 0;
    BAM_CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));
    fprintf(stderr, "[bam_q6_coal] GPU memory: %.1f GB free / %.1f GB total\n",
            gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "[bam_q6_coal] coalesce_k=%u, io_multi=%u, pc_page_size=%lu, pc_pages=%lu, blocks=%u, groups=%u\n",
            k, io_multi, (unsigned long)pc_page_size, (unsigned long)n_pc_pages, num_blocks, n_groups);

    page_cache_t h_pc(pc_page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);
    page_cache_d_t* d_pc = h_pc.d_pc_ptr;

    // Copy prefix sum arrays to GPU (npages+1 elements)
    const uint64_t ps_len = params.field_npages + 1;
    const size_t ps_bytes = ps_len * sizeof(uint64_t);
    uint64_t* d_prefix_sums[4];
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaMalloc(&d_prefix_sums[fi], ps_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_prefix_sums[fi], params.h_prefix_sums[fi],
                                   ps_bytes, cudaMemcpyHostToDevice));
    }

    // Copy zone map stats to GPU
    int32_t* d_shipdate_stats = nullptr;
    if (params.h_shipdate_stats && params.nstats > 0) {
        const size_t stats_bytes = params.nstats * 2 * sizeof(int32_t);
        BAM_CUDA_CHECK(cudaMalloc(&d_shipdate_stats, stats_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_shipdate_stats, params.h_shipdate_stats,
                                   stats_bytes, cudaMemcpyHostToDevice));
    }

    // Copy compression metadata — npages+1 for offsets (sentinel for coalesced span calc)
    uint32_t* d_comp_sizes[4] = {};
    uint64_t* d_comp_offsets[4] = {};
    for (int fi = 0; fi < 4; fi++) {
        if (params.compression_method[fi] != 0 &&
            params.h_compressed_page_sizes[fi] && params.h_compressed_offsets[fi]) {
            size_t sz_sizes = params.field_npages * sizeof(uint32_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes[fi], sz_sizes));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes[fi], params.h_compressed_page_sizes[fi],
                                       sz_sizes, cudaMemcpyHostToDevice));

            // npages + 1 offsets for coalesced span calculation.
            // The host array has npages+1 elements but the sentinel (index npages)
            // is set to 0 by calculate_compressed_offsets(). We must overwrite it
            // with the correct value = offset_past_last_page so that
            //   span = offsets[pg_end] - offsets[pg_base]
            // works for the last group where pg_end == npages.
            size_t sz_offsets = (params.field_npages + 1) * sizeof(uint64_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], sz_offsets));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], params.h_compressed_offsets[fi],
                                       params.field_npages * sizeof(uint64_t),
                                       cudaMemcpyHostToDevice));
            // Compute correct sentinel: byte offset past the last compressed page.
            // Pages are 4096-aligned in the column file layout.
            uint64_t last_off = params.h_compressed_offsets[fi][params.field_npages - 1];
            uint32_t last_sz  = params.h_compressed_page_sizes[fi][params.field_npages - 1];
            uint64_t sentinel = last_off + ((last_sz + 4095) & ~(uint64_t)4095);
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi] + params.field_npages,
                                       &sentinel, sizeof(uint64_t),
                                       cudaMemcpyHostToDevice));
            fprintf(stderr, "[bam_q6_coal] field %d: sentinel offset = %lu (last_off=%lu, last_sz=%u)\n",
                    fi, (unsigned long)sentinel, (unsigned long)last_off, last_sz);
        }
    }

    // Decompression buffer: 4 slots per ring entry, scaled by io_multi
    const uint32_t elems_per_slot = params.decomp_elems_per_slot;
    int32_t* d_decomp_buf = nullptr;
    size_t decomp_buf_size = (size_t)num_blocks * io_multi * COALESCED_ENTRIES_PER_SLOT
                           * elems_per_slot * sizeof(int32_t);
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp_buf, decomp_buf_size));

    // GPU metadata
    BAMKernelMeta h_meta;
    for (int fi = 0; fi < 4; fi++) {
        h_meta.field_start_page_ids[fi] = params.field_start_page_ids[fi];
        h_meta.d_prefix_sums[fi] = d_prefix_sums[fi];
        h_meta.compression_method[fi] = params.compression_method[fi];
        h_meta.d_comp_sizes[fi] = d_comp_sizes[fi];
        h_meta.d_comp_offsets[fi] = d_comp_offsets[fi];
    }
    h_meta.field_npages = params.field_npages;
    h_meta.page_size = (uint32_t)pc_page_size;
    h_meta.blocks_per_page = params.blocks_per_page;
    h_meta.partition_start_lba = params.partition_start_lba;
    h_meta.n_devices = params.n_devices > 0 ? params.n_devices : 1;
    for (uint32_t d = 0; d < h_meta.n_devices; d++)
        h_meta.partition_start_lbas[d] = params.partition_start_lbas[d];
    h_meta.num_blocks = num_blocks;
    h_meta.nrows = params.nrows;
    h_meta.d_shipdate_stats = d_shipdate_stats;
    h_meta.nstats = params.nstats;
    h_meta.d_decomp_buf = d_decomp_buf;
    h_meta.decomp_elems_per_slot = elems_per_slot;
    h_meta.io_multi = io_multi;
    h_meta.sd_low = params.sd_low;
    h_meta.sd_high = params.sd_high;
    h_meta.coalesce_k = k;
    h_meta.original_page_size = (uint32_t)orig_page_size;

    BAMKernelMeta* d_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_meta, sizeof(BAMKernelMeta)));
    BAM_CUDA_CHECK(cudaMemcpy(d_meta, &h_meta, sizeof(BAMKernelMeta),
                               cudaMemcpyHostToDevice));

    int64_t* d_revenue;
    BAM_CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    BAMDiagCounters* d_diag;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, sizeof(BAMDiagCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, sizeof(BAMDiagCounters)));

    BAMPerfCounters* d_perf;
    BAM_CUDA_CHECK(cudaMalloc(&d_perf, sizeof(BAMPerfCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_perf, 0, sizeof(BAMPerfCounters)));

    size_t old_stack_size = 0;
    cudaDeviceGetLimit(&old_stack_size, cudaLimitStackSize);
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    const uint32_t threads_per_block = 32;

    // Memory estimate for user reference
    size_t pc_mem = n_pc_pages * pc_page_size;
    fprintf(stderr, "[bam_q6_coal] Launching %u blocks x %u threads, k=%u, io_multi=%u\n",
            num_blocks, threads_per_block, k, io_multi);
    fprintf(stderr, "[bam_q6_coal] page_cache: %zu MiB, decomp_buf: %zu MiB\n",
            pc_mem / (1024 * 1024), decomp_buf_size / (1024 * 1024));

    bam_q6_kernel_coalesced_io_multi<<<num_blocks, threads_per_block, 0, stream>>>(
        h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Results
    int64_t h_revenue = 0;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_revenue, d_revenue, sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, stream));

    BAMDiagCounters h_diag;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_diag, d_diag, sizeof(BAMDiagCounters),
                                    cudaMemcpyDeviceToHost, stream));

    BAMPerfCounters h_perf;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_perf, d_perf, sizeof(BAMPerfCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    fprintf(stderr, "[diag] sum_shipdate=%lu  sum_quantity=%lu\n",
            h_diag.sum_shipdate, h_diag.sum_quantity);
    fprintf(stderr, "[diag] sum_extprice=%lu  sum_discount=%lu\n",
            h_diag.sum_extprice, h_diag.sum_discount);

    // Profiling output
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, params.cuda_device);
    double clock_ghz = clock_khz / 1e6;

    uint64_t total_cycles = h_perf.io_cycles + h_perf.decomp_cycles + h_perf.eval_cycles;
    fprintf(stderr, "\n[perf] GPU clock: %.3f GHz\n", clock_ghz);
    fprintf(stderr, "[perf] pages_processed: %lu  io_calls: %lu  io_bytes: %lu (%.2f MiB)\n",
            h_perf.page_count, h_perf.io_count,
            h_perf.io_bytes, h_perf.io_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[perf] Accumulated cycles across %u blocks (thread 0):\n", num_blocks);
    fprintf(stderr, "[perf]   io:      %12lu cycles  (%5.1f%%)\n",
            h_perf.io_cycles, total_cycles > 0 ? 100.0 * h_perf.io_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   decomp:  %12lu cycles  (%5.1f%%)\n",
            h_perf.decomp_cycles, total_cycles > 0 ? 100.0 * h_perf.decomp_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   eval:    %12lu cycles  (%5.1f%%)\n",
            h_perf.eval_cycles, total_cycles > 0 ? 100.0 * h_perf.eval_cycles / total_cycles : 0.0);
    fprintf(stderr, "[perf]   total:   %12lu cycles\n", total_cycles);
    if (h_perf.io_count > 0) {
        double avg_io_us = (double)h_perf.io_cycles / h_perf.io_count / (clock_ghz * 1e3);
        fprintf(stderr, "[perf]   avg_io_latency: %.1f us/call  (%lu calls)\n",
                avg_io_us, h_perf.io_count);
        double avg_io_kb = (double)h_perf.io_bytes / h_perf.io_count / 1024.0;
        fprintf(stderr, "[perf]   avg_io_size: %.1f KiB/call\n", avg_io_kb);
    }

    BAM_CUDA_CHECK(cudaStreamDestroy(stream));

    // Cleanup
    BAM_CUDA_CHECK(cudaFree(d_meta));
    BAM_CUDA_CHECK(cudaFree(d_revenue));
    BAM_CUDA_CHECK(cudaFree(d_diag));
    BAM_CUDA_CHECK(cudaFree(d_perf));
    BAM_CUDA_CHECK(cudaFree(d_decomp_buf));
    if (d_shipdate_stats) BAM_CUDA_CHECK(cudaFree(d_shipdate_stats));
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaFree(d_prefix_sums[fi]));
        if (d_comp_sizes[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_sizes[fi]));
        if (d_comp_offsets[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_offsets[fi]));
    }

    return BAMRunResult{h_revenue, h_perf.io_count, h_perf.io_bytes};
}

// ============================================================
// BAM Revenue kernel — sync (Q6 scan plan, shipdate-only predicate)
// Same I/O and decompression as Q6, different eval.
// ============================================================
__global__ void bam_revenue_kernel_comp_sync(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    const uint32_t ndev = meta->n_devices;

    const uint64_t npages = meta->field_npages;
    const uint32_t blocks_per_page = meta->blocks_per_page;
    const uint64_t page_size = meta->page_size;

    const uint64_t* ps0 = meta->d_prefix_sums[0];
    const uint64_t* ps1 = meta->d_prefix_sums[1];
    const uint64_t* ps2 = meta->d_prefix_sums[2];
    const uint64_t* ps3 = meta->d_prefix_sums[3];

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;

    const uint16_t* comp = meta->compression_method;

    __shared__ uint shared_bws[4];
    __shared__ uint shared_offs[4];
    __shared__ int s_skip;

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;
    int32_t* decomp_base = meta->d_decomp_buf
        + (uint64_t)block_id * ENTRIES_PER_BLOCK * elems_per_slot;

    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    for (uint64_t sd_pg = block_id; sd_pg < npages; sd_pg += gridDim.x) {
        // Zone map pruning (using sd_low/sd_high)
        if (tid == 0) {
            s_skip = 0;
            if (stats && sd_pg < nstats) {
                int32_t page_min = stats[sd_pg * 2];
                int32_t page_max = stats[sd_pg * 2 + 1];
                if (page_max < meta->sd_low || page_min > (meta->sd_high - 1)) {
                    s_skip = 1;
                }
            }
        }
        __syncwarp();
        if (s_skip) continue;

        blk_page_count++;

        uint64_t row_begin = ps0[sd_pg];
        uint64_t row_end   = ps0[sd_pg + 1];
        uint64_t nrows_this = row_end - row_begin;
        if (nrows_this == 0) continue;

        const uint64_t* ps_arr[3] = { ps1, ps2, ps3 };
        uint64_t fi_pg[3];
        uint64_t fi_bound[3];
        bool needs_boundary[3];

        for (int f = 0; f < 3; f++) {
            fi_pg[f] = find_page_for_row(ps_arr[f], npages, row_begin);
            fi_bound[f] = ps_arr[f][fi_pg[f] + 1];
            needs_boundary[f] = (fi_bound[f] < row_end && fi_pg[f] + 1 < npages);
            if (!needs_boundary[f]) fi_bound[f] = row_end;
        }

        // I/O: same as Q6 (7 entry slots) with RAID0 device routing
        uint64_t io_lba[7];
        unsigned long long io_entry[7];
        uint32_t io_nblocks[7];
        uint32_t io_dev[7];
        int n_ios = 0;

        // L_SHIPDATE
        {
            uint64_t sd_global_pg = meta->field_start_page_ids[0] + sd_pg;
            uint32_t dev = sd_global_pg % ndev;
            if (comp[0] != 0) {
                io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[0][sd_pg] / 512;
                io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[0][sd_pg]);
            } else {
                uint64_t local_pg = sd_global_pg / ndev;
                io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                io_nblocks[n_ios] = blocks_per_page;
            }
            io_dev[n_ios] = dev;
        }
        io_entry[n_ios] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 0;
        n_ios++;

        // Non-driving fields: primary pages
        unsigned long long fi_entry_a[3], fi_entry_b[3];
        for (int f = 0; f < 3; f++) {
            fi_entry_a[f] = (unsigned long long)block_id * ENTRIES_PER_BLOCK + 1 + f * 2;
            fi_entry_b[f] = fi_entry_a[f] + 1;

            uint64_t global_pg = meta->field_start_page_ids[f + 1] + fi_pg[f];
            uint32_t dev = global_pg % ndev;
            if (comp[f + 1] != 0) {
                io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[f + 1][fi_pg[f]] / 512;
                io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f]]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                io_nblocks[n_ios] = blocks_per_page;
            }
            io_dev[n_ios] = dev;
            io_entry[n_ios] = fi_entry_a[f];
            n_ios++;
        }

        // Boundary pages (packed after primary)
        int boundary_this_page = 0;
        for (int f = 0; f < 3; f++) {
            if (needs_boundary[f]) {
                uint64_t global_pg = meta->field_start_page_ids[f + 1] + fi_pg[f] + 1;
                uint32_t dev = global_pg % ndev;
                if (comp[f + 1] != 0) {
                    io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[f + 1][fi_pg[f] + 1] / 512;
                    io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg[f] + 1]);
                } else {
                    uint64_t local_pg = global_pg / ndev;
                    io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                    io_nblocks[n_ios] = blocks_per_page;
                }
                io_dev[n_ios] = dev;
                io_entry[n_ios] = fi_entry_b[f];
                n_ios++;
                boundary_this_page++;
            }
        }
        blk_boundary_count += boundary_this_page;

        long long t0 = clock64();
        uint16_t my_cid = 0;
        uint16_t my_sq_pos = 0;
        if ((int)tid < n_ios) {
            QueuePair* my_qp = ctrls[io_dev[tid]]->d_qps
                             + (block_id % ctrls[io_dev[tid]]->n_qps);
            access_data_async(pc, my_qp, io_lba[tid], io_nblocks[tid],
                              io_entry[tid], NVM_IO_READ, &my_cid, &my_sq_pos);
        }
        __syncwarp();
        if ((int)tid < n_ios) {
            QueuePair* my_qp = ctrls[io_dev[tid]]->d_qps
                             + (block_id % ctrls[io_dev[tid]]->n_qps);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&my_qp->cq, my_cid, &poll_loc, &poll_head);
            cq_dequeue(&my_qp->cq, cq_pos, &my_qp->sq);
            put_cid(&my_qp->sq, my_cid);
        }
        __syncwarp();

        long long t1 = clock64();
        blk_io_cycles += (uint64_t)(t1 - t0);
        blk_io_count += n_ios;
        for (int ii = 0; ii < n_ios; ii++) blk_io_bytes += (uint64_t)io_nblocks[ii] * 512;

        // Decompress: same as Q6 (all 4 fields)
        long long td0 = clock64();

        int32_t* sd_decomp = decomp_base + 0 * elems_per_slot;
        decompress_page_warp(base + io_entry[0] * page_size, sd_decomp,
                             comp[0], shared_bws, shared_offs, tid, nullptr);

        decompress_page_warp(base + fi_entry_a[0] * page_size,
                             decomp_base + 1 * elems_per_slot,
                             comp[1], shared_bws, shared_offs, tid, nullptr);
        decompress_page_warp(base + fi_entry_a[1] * page_size,
                             decomp_base + 3 * elems_per_slot,
                             comp[2], shared_bws, shared_offs, tid, nullptr);
        decompress_page_warp(base + fi_entry_a[2] * page_size,
                             decomp_base + 5 * elems_per_slot,
                             comp[3], shared_bws, shared_offs, tid, nullptr);

        if (needs_boundary[0]) {
            decompress_page_warp(base + fi_entry_b[0] * page_size,
                                 decomp_base + 2 * elems_per_slot,
                                 comp[1], shared_bws, shared_offs, tid, nullptr);
        }
        if (needs_boundary[1]) {
            decompress_page_warp(base + fi_entry_b[1] * page_size,
                                 decomp_base + 4 * elems_per_slot,
                                 comp[2], shared_bws, shared_offs, tid, nullptr);
        }
        if (needs_boundary[2]) {
            decompress_page_warp(base + fi_entry_b[2] * page_size,
                                 decomp_base + 6 * elems_per_slot,
                                 comp[3], shared_bws, shared_offs, tid, nullptr);
        }

        long long td1 = clock64();
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        // Revenue eval: shipdate-only predicate
        long long te0 = clock64();

        int32_t* qt_a = decomp_base + 1 * elems_per_slot;
        int32_t* qt_b = decomp_base + 2 * elems_per_slot;
        int32_t* ep_a = decomp_base + 3 * elems_per_slot;
        int32_t* ep_b = decomp_base + 4 * elems_per_slot;
        int32_t* dc_a = decomp_base + 5 * elems_per_slot;
        int32_t* dc_b = decomp_base + 6 * elems_per_slot;

        uint64_t qt_base = ps1[fi_pg[0]];
        uint64_t ep_base = ps2[fi_pg[1]];
        uint64_t dc_base = ps3[fi_pg[2]];

        const int32_t qt_max = meta->revenue_qt_max;
        int64_t local_rev = 0;

        for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 32) {
            uint64_t gr = row_begin + i;

            int32_t l_shipdate = sd_decomp[i];

            int32_t l_extendedprice = (gr < fi_bound[1])
                ? ep_a[gr - ep_base]
                : ep_b[gr - fi_bound[1]];

            int32_t l_discount = (gr < fi_bound[2])
                ? dc_a[gr - dc_base]
                : dc_b[gr - fi_bound[2]];

            bool pass = (l_shipdate >= meta->sd_low && l_shipdate < meta->sd_high);
            if (qt_max > 0) {
                int32_t l_quantity = (gr < fi_bound[0])
                    ? qt_a[gr - qt_base]
                    : qt_b[gr - fi_bound[0]];
                pass = pass && (l_quantity < qt_max);
            }
            if (pass) {
                local_rev += (int64_t)l_extendedprice * l_discount;
            }
        }

        long long te1 = clock64();
        blk_eval_cycles += (uint64_t)(te1 - te0);

        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
        }

        if (tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(local_rev));
        }
    }

    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
    }
}

// ============================================================
// BAM Revenue kernel — pipelined I/O (Q6 scan plan, shipdate-only predicate)
// ============================================================
__global__ void bam_revenue_kernel_comp_io_multi(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t io_multi = meta->io_multi;

    const uint32_t n_qps = ctrls[0]->n_qps;
    QueuePair* d_qps = ctrls[0]->d_qps;

    const uint64_t npages = meta->field_npages;
    const uint64_t page_size = meta->page_size;

    const uint64_t* ps2 = meta->d_prefix_sums[2]; // L_EXTENDEDPRICE
    const uint64_t* ps3 = meta->d_prefix_sums[3]; // L_DISCOUNT

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;

    const uint16_t* comp = meta->compression_method;

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;

    __shared__ uint shared_bws[4];
    __shared__ uint shared_offs[4];
    __shared__ int s_skip;

    __shared__ SlotMeta slot_meta[MAX_IO_MULTI];

    uint16_t ring_cid[MAX_IO_MULTI];

    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_eval_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    uint64_t cursor = block_id;
    uint32_t ring_head = 0;
    uint32_t ring_count = 0;

    // Fill macro: same as Q6 but zone map uses sd_low/sd_high
    #define REV_TRY_FILL_ONE_SLOT() do {                                    \
        bool _filled = false;                                               \
        while (!_filled && cursor < npages) {                               \
            uint64_t _sd_pg = cursor;                                       \
            cursor += gridDim.x;                                            \
            if (tid == 0) {                                                 \
                s_skip = 0;                                                 \
                if (stats && _sd_pg < nstats) {                             \
                    int32_t _pmin = stats[_sd_pg * 2];                      \
                    int32_t _pmax = stats[_sd_pg * 2 + 1];                  \
                    if (_pmax < meta->sd_low || _pmin > (meta->sd_high - 1)) \
                        s_skip = 1;                                         \
                }                                                           \
            }                                                               \
            __syncwarp();                                                    \
            if (s_skip) continue;                                           \
            uint64_t _ps0b = meta->d_prefix_sums[0][_sd_pg];               \
            uint64_t _ps0e = meta->d_prefix_sums[0][_sd_pg + 1];           \
            if (_ps0e == _ps0b) continue;                                   \
            uint32_t _s = (ring_head + ring_count) % io_multi;              \
            QueuePair* _qp = d_qps                                         \
                + ((block_id * io_multi + _s) % n_qps);                     \
            bam_q6_submit_slot(_s, _sd_pg, pc, _qp, meta,                  \
                               slot_meta, &ring_cid[_s],                    \
                               tid, block_id, io_multi, &s_skip);           \
            ring_count++;                                                   \
            _filled = true;                                                 \
        }                                                                   \
    } while(0)

    REV_TRY_FILL_ONE_SLOT();

    while (ring_count > 0) {
        if (ring_count < io_multi) {
            REV_TRY_FILL_ONE_SLOT();
        }

        uint32_t s = ring_head;
        SlotMeta& sm = slot_meta[s];

        blk_page_count++;

        // Poll I/O
        long long t0 = clock64();
        QueuePair* qp_s = d_qps + ((block_id * io_multi + s) % n_qps);
        if ((int)tid < sm.n_ios) {
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp_s->cq, ring_cid[s], &poll_loc, &poll_head);
            cq_dequeue(&qp_s->cq, cq_pos, &qp_s->sq);
            put_cid(&qp_s->sq, ring_cid[s]);
        }
        __syncwarp();

        long long t1 = clock64();
        blk_io_cycles += (uint64_t)(t1 - t0);
        blk_io_count += sm.n_ios;
        blk_io_bytes += sm.io_bytes;
        for (int bf = 0; bf < 3; bf++) blk_boundary_count += sm.needs_boundary[bf];

        // Decompress: same as Q6 (all 4 fields, 7 entry slots)
        long long td0 = clock64();

        unsigned long long entry_base =
            (unsigned long long)(block_id * io_multi + s) * ENTRIES_PER_BLOCK;
        int32_t* decomp = meta->d_decomp_buf
            + (uint64_t)(block_id * io_multi + s) * ENTRIES_PER_BLOCK * elems_per_slot;

        int32_t* sd_decomp = decomp + 0 * elems_per_slot;
        decompress_page_warp(base + (entry_base + 0) * page_size, sd_decomp,
                             comp[0], shared_bws, shared_offs, tid, nullptr);

        decompress_page_warp(base + (entry_base + 1) * page_size,
                             decomp + 1 * elems_per_slot,
                             comp[1], shared_bws, shared_offs, tid, nullptr);
        decompress_page_warp(base + (entry_base + 3) * page_size,
                             decomp + 3 * elems_per_slot,
                             comp[2], shared_bws, shared_offs, tid, nullptr);
        decompress_page_warp(base + (entry_base + 5) * page_size,
                             decomp + 5 * elems_per_slot,
                             comp[3], shared_bws, shared_offs, tid, nullptr);

        if (sm.needs_boundary[0]) {
            decompress_page_warp(base + (entry_base + 2) * page_size,
                                 decomp + 2 * elems_per_slot,
                                 comp[1], shared_bws, shared_offs, tid, nullptr);
        }
        if (sm.needs_boundary[1]) {
            decompress_page_warp(base + (entry_base + 4) * page_size,
                                 decomp + 4 * elems_per_slot,
                                 comp[2], shared_bws, shared_offs, tid, nullptr);
        }
        if (sm.needs_boundary[2]) {
            decompress_page_warp(base + (entry_base + 6) * page_size,
                                 decomp + 6 * elems_per_slot,
                                 comp[3], shared_bws, shared_offs, tid, nullptr);
        }

        long long td1 = clock64();
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        // Revenue eval: shipdate-only predicate
        long long te0 = clock64();

        int32_t* ep_a = decomp + 3 * elems_per_slot;
        int32_t* ep_b = decomp + 4 * elems_per_slot;
        int32_t* dc_a = decomp + 5 * elems_per_slot;
        int32_t* dc_b = decomp + 6 * elems_per_slot;

        uint64_t ep_base = ps2[sm.fi_pg[1]];
        uint64_t dc_base = ps3[sm.fi_pg[2]];

        uint64_t nrows_this = sm.row_end - sm.row_begin;
        int64_t local_rev = 0;

        for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 32) {
            uint64_t gr = sm.row_begin + i;

            int32_t l_shipdate = sd_decomp[i];

            int32_t l_extendedprice = (gr < sm.fi_bound[1])
                ? ep_a[gr - ep_base]
                : ep_b[gr - sm.fi_bound[1]];

            int32_t l_discount = (gr < sm.fi_bound[2])
                ? dc_a[gr - dc_base]
                : dc_b[gr - sm.fi_bound[2]];

            if (l_shipdate >= meta->sd_low && l_shipdate < meta->sd_high) {
                local_rev += (int64_t)l_extendedprice * l_discount;
            }
        }

        long long te1 = clock64();
        blk_eval_cycles += (uint64_t)(te1 - te0);

        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
        }

        if (tid == 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(local_rev));
        }

        ring_head = (ring_head + 1) % io_multi;
        ring_count--;
    }

    #undef REV_TRY_FILL_ONE_SLOT

    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->eval_cycles),
                  (unsigned long long)blk_eval_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
    }
}

// ============================================================
// BAM Revenue kernel — 128 threads (4 warps), parallel field decode
// Same IO/decomp as bam_q6_kernel_128t_sync, revenue predicate
// (shipdate-only: sd_low <= l_shipdate < sd_high).
// ============================================================
__global__ void bam_revenue_kernel_128t_sync(
    Controller** ctrls,
    page_cache_d_t* pc,
    BAMKernelMeta* meta,
    int64_t* d_revenue,
    BAMDiagCounters* d_diag,
    BAMPerfCounters* d_perf)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t tid = threadIdx.x;           // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t lane_id = tid % 32;          // 0..31

    // RAID0: per-IO device routing (ndev=1 degenerates to single device)
    const uint32_t ndev = meta->n_devices;

    const uint64_t npages = meta->field_npages;
    const uint32_t blocks_per_page = meta->blocks_per_page;
    const uint64_t page_size = meta->page_size;

    const uint64_t* ps0 = meta->d_prefix_sums[0];
    const uint64_t* ps1 = meta->d_prefix_sums[1];
    const uint64_t* ps2 = meta->d_prefix_sums[2];
    const uint64_t* ps3 = meta->d_prefix_sums[3];

    char* base = (char*)pc->base_addr;

    const int32_t* stats = meta->d_shipdate_stats;
    const uint64_t nstats = meta->nstats;

    const uint16_t* comp = meta->compression_method;

    const int field_entry_off[4] = {0, 1, 3, 5};

    __shared__ int s_skip;
    __shared__ uint32_t s_comp_blk[4 * 136];
    __shared__ uint shared_bws[4 * 4];
    __shared__ uint shared_offs[4 * 4];
    __shared__ int32_t s_vals[4 * 128];
    __shared__ uint32_t s_nblocks_pfor;
    __shared__ uint32_t s_nalloc;
    __shared__ int s_all_comp;
    __shared__ int s_has_boundary;
    __shared__ uint64_t s_row_begin;
    __shared__ uint64_t s_row_end;
    __shared__ uint64_t s_fi_pg[3];
    __shared__ uint64_t s_fi_bound[3];
    __shared__ int s_needs_boundary[3];
    __shared__ int64_t s_warp_rev[4];

    const uint32_t elems_per_slot = meta->decomp_elems_per_slot;
    int32_t* decomp_base = meta->d_decomp_buf
        + (uint64_t)block_id * ENTRIES_PER_BLOCK * elems_per_slot;

    uint64_t blk_io_cycles = 0;
    uint64_t blk_decomp_cycles = 0;
    uint64_t blk_page_count = 0;
    uint64_t blk_io_count = 0;
    uint64_t blk_io_bytes = 0;
    uint64_t blk_boundary_count = 0;

    for (uint64_t sd_pg = block_id; sd_pg < npages; sd_pg += gridDim.x) {

        // === Phase 0: Zone map pruning ===
        if (tid == 0) {
            s_skip = 0;
            if (stats && sd_pg < nstats) {
                int32_t page_min = stats[sd_pg * 2];
                int32_t page_max = stats[sd_pg * 2 + 1];
                if (page_max < meta->sd_low || page_min > meta->sd_high - 1) {
                    s_skip = 1;
                }
            }
        }
        __syncthreads();
        if (s_skip) continue;

        // === Phase 1: I/O — warp 0 only ===
        if (warp_id == 0) {
            uint64_t row_begin = ps0[sd_pg];
            uint64_t row_end   = ps0[sd_pg + 1];
            uint64_t nrows_this = row_end - row_begin;

            if (lane_id == 0) {
                if (nrows_this == 0) {
                    s_skip = 1;
                } else {
                    s_skip = 0;
                    s_row_begin = row_begin;
                    s_row_end = row_end;
                }
            }
            __syncwarp();

            if (!s_skip) {
                const uint64_t* ps_arr[3] = { ps1, ps2, ps3 };
                uint64_t fi_pg_local[3];
                uint64_t fi_bound_local[3];
                int needs_boundary_local[3];

                for (int f = 0; f < 3; f++) {
                    fi_pg_local[f] = find_page_for_row(ps_arr[f], npages, row_begin);
                    fi_bound_local[f] = ps_arr[f][fi_pg_local[f] + 1];
                    needs_boundary_local[f] = (fi_bound_local[f] < row_end && fi_pg_local[f] + 1 < npages) ? 1 : 0;
                    if (!needs_boundary_local[f]) fi_bound_local[f] = row_end;
                }

                if (lane_id == 0) {
                    for (int f = 0; f < 3; f++) {
                        s_fi_pg[f] = fi_pg_local[f];
                        s_fi_bound[f] = fi_bound_local[f];
                        s_needs_boundary[f] = needs_boundary_local[f];
                    }
                    s_all_comp = (comp[0] != 0 && comp[1] != 0 &&
                                  comp[2] != 0 && comp[3] != 0);
                    s_has_boundary = (needs_boundary_local[0] ||
                                     needs_boundary_local[1] ||
                                     needs_boundary_local[2]);
                }

                unsigned long long entry_base =
                    (unsigned long long)block_id * ENTRIES_PER_BLOCK;
                uint64_t io_lba[7];
                unsigned long long io_entry[7];
                uint32_t io_nblocks[7];
                uint32_t io_dev[7];      // RAID0: target device index
                int n_ios = 0;

                // L_SHIPDATE
                {
                    uint64_t sd_global_pg = meta->field_start_page_ids[0] + sd_pg;
                    uint32_t dev = sd_global_pg % ndev;
                    if (comp[0] != 0) {
                        io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[0][sd_pg] / 512;
                        io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[0][sd_pg]);
                    } else {
                        uint64_t local_pg = sd_global_pg / ndev;
                        io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                        io_nblocks[n_ios] = blocks_per_page;
                    }
                    io_dev[n_ios] = dev;
                }
                io_entry[n_ios] = entry_base + 0;
                n_ios++;

                unsigned long long fi_entry_a[3], fi_entry_b[3];
                for (int f = 0; f < 3; f++) {
                    fi_entry_a[f] = entry_base + 1 + f * 2;
                    fi_entry_b[f] = fi_entry_a[f] + 1;

                    uint64_t fi_global_pg = meta->field_start_page_ids[f + 1] + fi_pg_local[f];
                    uint32_t dev = fi_global_pg % ndev;
                    if (comp[f + 1] != 0) {
                        io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[f + 1][fi_pg_local[f]] / 512;
                        io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg_local[f]]);
                    } else {
                        uint64_t local_pg = fi_global_pg / ndev;
                        io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                        io_nblocks[n_ios] = blocks_per_page;
                    }
                    io_dev[n_ios] = dev;
                    io_entry[n_ios] = fi_entry_a[f];
                    n_ios++;
                }

                int boundary_this = 0;
                for (int f = 0; f < 3; f++) {
                    if (needs_boundary_local[f]) {
                        uint64_t fi_global_pg = meta->field_start_page_ids[f + 1] + fi_pg_local[f] + 1;
                        uint32_t dev = fi_global_pg % ndev;
                        if (comp[f + 1] != 0) {
                            io_lba[n_ios] = meta->partition_start_lbas[dev] + meta->d_comp_offsets[f + 1][fi_pg_local[f] + 1] / 512;
                            io_nblocks[n_ios] = safe_io_nblocks(meta->d_comp_sizes[f + 1][fi_pg_local[f] + 1]);
                        } else {
                            uint64_t local_pg = fi_global_pg / ndev;
                            io_lba[n_ios] = meta->partition_start_lbas[dev] + local_pg * blocks_per_page;
                            io_nblocks[n_ios] = blocks_per_page;
                        }
                        io_dev[n_ios] = dev;
                        io_entry[n_ios] = fi_entry_b[f];
                        n_ios++;
                        boundary_this++;
                    }
                }

                long long t0 = clock64();
                uint16_t my_cid = 0;
                uint16_t my_sq_pos = 0;
                QueuePair* my_qp = nullptr;
                if ((int)lane_id < n_ios) {
                    my_qp = ctrls[io_dev[lane_id]]->d_qps
                          + (block_id % ctrls[io_dev[lane_id]]->n_qps);
                    access_data_async(pc, my_qp, io_lba[lane_id], io_nblocks[lane_id],
                                      io_entry[lane_id], NVM_IO_READ, &my_cid, &my_sq_pos);
                }
                __syncwarp();

                if ((int)lane_id < n_ios) {
                    uint32_t poll_loc, poll_head;
                    uint32_t cq_pos = cq_poll(&my_qp->cq, my_cid, &poll_loc, &poll_head);
                    cq_dequeue(&my_qp->cq, cq_pos, &my_qp->sq);
                    put_cid(&my_qp->sq, my_cid);
                }
                __syncwarp();

                long long t1 = clock64();
                blk_io_cycles += (uint64_t)(t1 - t0);
                blk_io_count += n_ios;
                for (int ii = 0; ii < n_ios; ii++) blk_io_bytes += (uint64_t)io_nblocks[ii] * 512;
                blk_boundary_count += boundary_this;
                blk_page_count++;
            }
        }
        __syncthreads();
        if (s_skip) continue;

        // === Phase 2: Fused decompress + Revenue eval ===
        long long td0 = clock64();

        unsigned long long entry_base =
            (unsigned long long)block_id * ENTRIES_PER_BLOCK;

        int64_t local_rev = 0;

        if (s_all_comp && !s_has_boundary) {
            // ── Case A: 4 warps decode 4 fields in parallel ──
            char* my_page_ptr = base + (entry_base + field_entry_off[warp_id]) * page_size;

            if (warp_id == 0 && lane_id == 0) {
                bam_pag_head* hdr0 = (bam_pag_head*)(base + entry_base * page_size);
                s_nblocks_pfor = hdr0->watermark / 128;
                s_nalloc = hdr0->nalloc;
            }
            __syncthreads();

            uint32_t nblocks_pfor = s_nblocks_pfor;
            uint32_t nalloc = s_nalloc;

            uint32_t* my_blk_start = (uint32_t*)(my_page_ptr + BAM_PAG_HDR_BYTES);
            uint32_t* my_data_ptr = my_blk_start + (nblocks_pfor + 1);

            uint32_t* my_comp_blk = s_comp_blk + warp_id * 136;
            uint* my_bws = shared_bws + warp_id * 4;
            uint* my_offs = shared_offs + warp_id * 4;

            for (uint32_t b = 0; b < nblocks_pfor; b++) {
                int32_t field_vals[4];
                decode_pfor_block_smem(
                    my_data_ptr + my_blk_start[b],
                    my_blk_start[b + 1] - my_blk_start[b],
                    my_comp_blk, my_bws, my_offs,
                    lane_id, field_vals);

                for (int k = 0; k < 4; k++) {
                    s_vals[warp_id * 128 + k * 32 + lane_id] = field_vals[k];
                }
                __syncthreads();

                uint32_t idx = b * 128 + tid;
                if (idx < nalloc) {
                    int32_t sd_v = s_vals[0 * 128 + tid]; // SD
                    int32_t qt_v = s_vals[1 * 128 + tid]; // QT
                    int32_t ep_v = s_vals[2 * 128 + tid]; // EP
                    int32_t dc_v = s_vals[3 * 128 + tid]; // DC

                    bool pass = (sd_v >= meta->sd_low && sd_v < meta->sd_high);
                    if (meta->revenue_qt_max > 0)
                        pass = pass && (qt_v < meta->revenue_qt_max);
                    if (pass) {
                        local_rev += (int64_t)ep_v * dc_v;
                    }
                }
                __syncthreads();
            }
        } else {
            // ── Case B: fallback (boundary or mixed compression) ──
            if (warp_id == 0) {
                decompress_page_warp(base + (entry_base + 0) * page_size,
                    decomp_base + 0 * elems_per_slot,
                    comp[0], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 1) * page_size,
                    decomp_base + 1 * elems_per_slot,
                    comp[1], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 3) * page_size,
                    decomp_base + 3 * elems_per_slot,
                    comp[2], shared_bws, shared_offs, lane_id, nullptr);
                decompress_page_warp(base + (entry_base + 5) * page_size,
                    decomp_base + 5 * elems_per_slot,
                    comp[3], shared_bws, shared_offs, lane_id, nullptr);

                if (s_needs_boundary[0]) {
                    decompress_page_warp(base + (entry_base + 2) * page_size,
                        decomp_base + 2 * elems_per_slot,
                        comp[1], shared_bws, shared_offs, lane_id, nullptr);
                }
                if (s_needs_boundary[1]) {
                    decompress_page_warp(base + (entry_base + 4) * page_size,
                        decomp_base + 4 * elems_per_slot,
                        comp[2], shared_bws, shared_offs, lane_id, nullptr);
                }
                if (s_needs_boundary[2]) {
                    decompress_page_warp(base + (entry_base + 6) * page_size,
                        decomp_base + 6 * elems_per_slot,
                        comp[3], shared_bws, shared_offs, lane_id, nullptr);
                }
            }
            __syncthreads();

            int32_t* sd_decomp = decomp_base + 0 * elems_per_slot;
            int32_t* qt_a = decomp_base + 1 * elems_per_slot;
            int32_t* qt_b = decomp_base + 2 * elems_per_slot;
            int32_t* ep_a = decomp_base + 3 * elems_per_slot;
            int32_t* ep_b = decomp_base + 4 * elems_per_slot;
            int32_t* dc_a = decomp_base + 5 * elems_per_slot;
            int32_t* dc_b = decomp_base + 6 * elems_per_slot;

            uint64_t qt_base_off = ps1[s_fi_pg[0]];
            uint64_t ep_base_off = ps2[s_fi_pg[1]];
            uint64_t dc_base_off = ps3[s_fi_pg[2]];

            const int32_t qt_max = meta->revenue_qt_max;
            uint64_t nrows_this = s_row_end - s_row_begin;
            for (uint32_t i = tid; i < (uint32_t)nrows_this; i += 128) {
                uint64_t gr = s_row_begin + i;
                int32_t l_shipdate = sd_decomp[i];

                int32_t l_extendedprice = (gr < s_fi_bound[1])
                    ? ep_a[gr - ep_base_off] : ep_b[gr - s_fi_bound[1]];
                int32_t l_discount = (gr < s_fi_bound[2])
                    ? dc_a[gr - dc_base_off] : dc_b[gr - s_fi_bound[2]];

                bool pass = (l_shipdate >= meta->sd_low && l_shipdate < meta->sd_high);
                if (qt_max > 0) {
                    int32_t l_quantity = (gr < s_fi_bound[0])
                        ? qt_a[gr - qt_base_off] : qt_b[gr - s_fi_bound[0]];
                    pass = pass && (l_quantity < qt_max);
                }
                if (pass) {
                    local_rev += (int64_t)l_extendedprice * l_discount;
                }
            }
        }

        long long td1 = clock64();
        blk_decomp_cycles += (uint64_t)(td1 - td0);

        // === Phase 3: Block-level reduction ===
        for (int offset = 16; offset > 0; offset /= 2) {
            local_rev += __shfl_down_sync(0xFFFFFFFF, local_rev, offset);
        }

        if (lane_id == 0) {
            s_warp_rev[warp_id] = local_rev;
        }
        __syncthreads();

        if (tid == 0) {
            int64_t block_rev = s_warp_rev[0] + s_warp_rev[1] + s_warp_rev[2] + s_warp_rev[3];
            atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                      static_cast<unsigned long long>(block_rev));
        }
        local_rev = 0;
        __syncthreads();
    }

    if (tid == 0 && d_perf) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_cycles),
                  (unsigned long long)blk_io_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->decomp_cycles),
                  (unsigned long long)blk_decomp_cycles);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->page_count),
                  (unsigned long long)blk_page_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_count),
                  (unsigned long long)blk_io_count);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->io_bytes),
                  (unsigned long long)blk_io_bytes);
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_perf->boundary_count),
                  (unsigned long long)blk_boundary_count);
    }
}

// ============================================================
// BAM Revenue host wrapper — 128t (4 warps, parallel field decode)
// ============================================================
BAMRunResult bam_revenue_128t_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl_handle) {
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    uint32_t num_blocks = params.num_blocks;
    const uint64_t page_size = params.page_size;

    const uint32_t io_multi = 1;

    const uint64_t n_pc_pages = (uint64_t)num_blocks * ENTRIES_PER_BLOCK;
    const uint64_t max_range = 64;

    size_t gpu_free = 0, gpu_total = 0;
    BAM_CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));
    fprintf(stderr, "[bam_revenue_128t] GPU memory: %.1f GB free / %.1f GB total\n",
            gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));

    page_cache_t h_pc(page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);
    page_cache_d_t* d_pc = h_pc.d_pc_ptr;

    const uint64_t ps_len = params.field_npages + 1;
    const size_t ps_bytes = ps_len * sizeof(uint64_t);
    uint64_t* d_prefix_sums[4];
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaMalloc(&d_prefix_sums[fi], ps_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_prefix_sums[fi], params.h_prefix_sums[fi],
                                   ps_bytes, cudaMemcpyHostToDevice));
    }

    int32_t* d_shipdate_stats = nullptr;
    if (params.h_shipdate_stats && params.nstats > 0) {
        const size_t stats_bytes = params.nstats * 2 * sizeof(int32_t);
        BAM_CUDA_CHECK(cudaMalloc(&d_shipdate_stats, stats_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_shipdate_stats, params.h_shipdate_stats,
                                   stats_bytes, cudaMemcpyHostToDevice));
    }

    uint32_t* d_comp_sizes[4] = {};
    uint64_t* d_comp_offsets[4] = {};
    for (int fi = 0; fi < 4; fi++) {
        if (params.compression_method[fi] != 0 &&
            params.h_compressed_page_sizes[fi] && params.h_compressed_offsets[fi]) {
            size_t sz_sizes = params.field_npages * sizeof(uint32_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes[fi], sz_sizes));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes[fi], params.h_compressed_page_sizes[fi],
                                       sz_sizes, cudaMemcpyHostToDevice));

            size_t sz_offsets = params.field_npages * sizeof(uint64_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], sz_offsets));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], params.h_compressed_offsets[fi],
                                       sz_offsets, cudaMemcpyHostToDevice));
        }
    }

    const uint32_t elems_per_slot = params.decomp_elems_per_slot;
    int32_t* d_decomp_buf = nullptr;
    size_t decomp_buf_size = (size_t)num_blocks * ENTRIES_PER_BLOCK
                           * elems_per_slot * sizeof(int32_t);
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp_buf, decomp_buf_size));

    BAMKernelMeta h_meta;
    for (int fi = 0; fi < 4; fi++) {
        h_meta.field_start_page_ids[fi] = params.field_start_page_ids[fi];
        h_meta.d_prefix_sums[fi] = d_prefix_sums[fi];
        h_meta.compression_method[fi] = params.compression_method[fi];
        h_meta.d_comp_sizes[fi] = d_comp_sizes[fi];
        h_meta.d_comp_offsets[fi] = d_comp_offsets[fi];
    }
    h_meta.field_npages = params.field_npages;
    h_meta.page_size = params.page_size;
    h_meta.blocks_per_page = params.blocks_per_page;
    h_meta.partition_start_lba = params.partition_start_lba;
    h_meta.n_devices = params.n_devices > 0 ? params.n_devices : 1;
    for (uint32_t d = 0; d < h_meta.n_devices; d++)
        h_meta.partition_start_lbas[d] = params.partition_start_lbas[d];
    h_meta.num_blocks = num_blocks;
    h_meta.nrows = params.nrows;
    h_meta.d_shipdate_stats = d_shipdate_stats;
    h_meta.nstats = params.nstats;
    h_meta.d_decomp_buf = d_decomp_buf;
    h_meta.decomp_elems_per_slot = elems_per_slot;
    h_meta.io_multi = io_multi;
    h_meta.sd_low = params.sd_low;
    h_meta.sd_high = params.sd_high;
    h_meta.revenue_qt_max = params.revenue_qt_max;

    BAMKernelMeta* d_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_meta, sizeof(BAMKernelMeta)));
    BAM_CUDA_CHECK(cudaMemcpy(d_meta, &h_meta, sizeof(BAMKernelMeta),
                               cudaMemcpyHostToDevice));

    int64_t* d_revenue;
    BAM_CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    BAMDiagCounters* d_diag;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, sizeof(BAMDiagCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, sizeof(BAMDiagCounters)));

    BAMPerfCounters* d_perf;
    BAM_CUDA_CHECK(cudaMalloc(&d_perf, sizeof(BAMPerfCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_perf, 0, sizeof(BAMPerfCounters)));

    size_t old_stack_size = 0;
    cudaDeviceGetLimit(&old_stack_size, cudaLimitStackSize);
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    const uint32_t threads_per_block = 128;
    fprintf(stderr, "[bam_revenue_128t] Launching %u blocks x %u threads (4 warps/block), n_devices=%u, decomp_buf=%zu MB\n",
            num_blocks, threads_per_block, h_meta.n_devices, decomp_buf_size / (1024 * 1024));

    bam_revenue_kernel_128t_sync<<<num_blocks, threads_per_block, 0, stream>>>(
        h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t h_revenue = 0;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_revenue, d_revenue, sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, stream));

    BAMPerfCounters h_perf;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_perf, d_perf, sizeof(BAMPerfCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, params.cuda_device);
    double clock_ghz = clock_khz / 1e6;

    uint64_t total_cycles = h_perf.io_cycles + h_perf.decomp_cycles;
    fprintf(stderr, "\n[perf] GPU clock: %.3f GHz\n", clock_ghz);
    fprintf(stderr, "[perf] pages_processed: %lu  io_calls: %lu  io_bytes: %lu (%.2f MiB)\n",
            h_perf.page_count, h_perf.io_count,
            h_perf.io_bytes, h_perf.io_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[perf] boundary_page_reads: %lu (%.1f%% of pages)\n",
            h_perf.boundary_count,
            h_perf.page_count > 0 ? 100.0 * h_perf.boundary_count / h_perf.page_count : 0.0);
    if (total_cycles > 0) {
        fprintf(stderr, "[perf]   io:      %12lu cycles  (%5.1f%%)\n",
                h_perf.io_cycles, 100.0 * h_perf.io_cycles / total_cycles);
        fprintf(stderr, "[perf]   decomp:  %12lu cycles  (%5.1f%%)\n",
                h_perf.decomp_cycles, 100.0 * h_perf.decomp_cycles / total_cycles);
        fprintf(stderr, "[perf]   total:   %12lu cycles\n", total_cycles);
    }
    if (h_perf.io_count > 0) {
        double avg_io_us = (double)h_perf.io_cycles / h_perf.io_count / (clock_ghz * 1e3);
        fprintf(stderr, "[perf]   avg_io_latency: %.1f us/call  (%lu calls)\n",
                avg_io_us, h_perf.io_count);
    }

    BAM_CUDA_CHECK(cudaStreamDestroy(stream));

    BAM_CUDA_CHECK(cudaFree(d_meta));
    BAM_CUDA_CHECK(cudaFree(d_revenue));
    BAM_CUDA_CHECK(cudaFree(d_diag));
    BAM_CUDA_CHECK(cudaFree(d_perf));
    BAM_CUDA_CHECK(cudaFree(d_decomp_buf));
    if (d_shipdate_stats) BAM_CUDA_CHECK(cudaFree(d_shipdate_stats));
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaFree(d_prefix_sums[fi]));
        if (d_comp_sizes[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_sizes[fi]));
        if (d_comp_offsets[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_offsets[fi]));
    }

    return BAMRunResult{h_revenue, h_perf.io_count, h_perf.io_bytes};
}

// ============================================================
// BAM Revenue host wrapper — reuses Q6 infrastructure
// ============================================================
BAMRunResult bam_revenue_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl_handle) {
    BAMCtrlHandle* h = static_cast<BAMCtrlHandle*>(ctrl_handle);

    const uint32_t num_blocks = params.num_blocks;
    const uint64_t page_size = params.page_size;
    const uint32_t io_multi = std::max(1u, std::min(params.io_multiplicity, (uint32_t)MAX_IO_MULTI));

    const uint64_t n_pc_pages = (uint64_t)num_blocks * ENTRIES_PER_BLOCK * io_multi;
    const uint64_t max_range = 64;

    size_t gpu_free = 0, gpu_total = 0;
    BAM_CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));
    fprintf(stderr, "[bam_revenue] GPU memory: %.1f GB free / %.1f GB total\n",
            gpu_free / (1024.0 * 1024.0 * 1024.0), gpu_total / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "[bam_revenue] io_multi=%u, pc_pages=%lu\n", io_multi, (unsigned long)n_pc_pages);

    // Cross-block QP sharing is safe; only within-block slot uniqueness is needed.
    uint32_t n_qps_avail = params.num_queues;
    if (io_multi > n_qps_avail) {
        fprintf(stderr, "[bam_revenue] WARNING: io_multi=%u exceeds n_qps=%u.\n",
                io_multi, n_qps_avail);
    }

    page_cache_t h_pc(page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);
    page_cache_d_t* d_pc = h_pc.d_pc_ptr;

    const uint64_t ps_len = params.field_npages + 1;
    const size_t ps_bytes = ps_len * sizeof(uint64_t);
    uint64_t* d_prefix_sums[4];
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaMalloc(&d_prefix_sums[fi], ps_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_prefix_sums[fi], params.h_prefix_sums[fi],
                                   ps_bytes, cudaMemcpyHostToDevice));
    }

    int32_t* d_shipdate_stats = nullptr;
    if (params.h_shipdate_stats && params.nstats > 0) {
        const size_t stats_bytes = params.nstats * 2 * sizeof(int32_t);
        BAM_CUDA_CHECK(cudaMalloc(&d_shipdate_stats, stats_bytes));
        BAM_CUDA_CHECK(cudaMemcpy(d_shipdate_stats, params.h_shipdate_stats,
                                   stats_bytes, cudaMemcpyHostToDevice));
    }

    uint32_t* d_comp_sizes[4] = {};
    uint64_t* d_comp_offsets[4] = {};
    for (int fi = 0; fi < 4; fi++) {
        if (params.compression_method[fi] != 0 &&
            params.h_compressed_page_sizes[fi] && params.h_compressed_offsets[fi]) {
            size_t sz_sizes = params.field_npages * sizeof(uint32_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes[fi], sz_sizes));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes[fi], params.h_compressed_page_sizes[fi],
                                       sz_sizes, cudaMemcpyHostToDevice));

            size_t sz_offsets = params.field_npages * sizeof(uint64_t);
            BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], sz_offsets));
            BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], params.h_compressed_offsets[fi],
                                       sz_offsets, cudaMemcpyHostToDevice));
        }
    }

    const uint32_t elems_per_slot = params.decomp_elems_per_slot;
    int32_t* d_decomp_buf = nullptr;
    size_t decomp_buf_size = (size_t)num_blocks * io_multi * ENTRIES_PER_BLOCK
                           * elems_per_slot * sizeof(int32_t);
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp_buf, decomp_buf_size));

    BAMKernelMeta h_meta;
    for (int fi = 0; fi < 4; fi++) {
        h_meta.field_start_page_ids[fi] = params.field_start_page_ids[fi];
        h_meta.d_prefix_sums[fi] = d_prefix_sums[fi];
        h_meta.compression_method[fi] = params.compression_method[fi];
        h_meta.d_comp_sizes[fi] = d_comp_sizes[fi];
        h_meta.d_comp_offsets[fi] = d_comp_offsets[fi];
    }
    h_meta.field_npages = params.field_npages;
    h_meta.page_size = params.page_size;
    h_meta.blocks_per_page = params.blocks_per_page;
    h_meta.partition_start_lba = params.partition_start_lba;
    h_meta.n_devices = params.n_devices > 0 ? params.n_devices : 1;
    for (uint32_t d = 0; d < h_meta.n_devices; d++)
        h_meta.partition_start_lbas[d] = params.partition_start_lbas[d];
    h_meta.num_blocks = num_blocks;
    h_meta.nrows = params.nrows;
    h_meta.d_shipdate_stats = d_shipdate_stats;
    h_meta.nstats = params.nstats;
    h_meta.d_decomp_buf = d_decomp_buf;
    h_meta.decomp_elems_per_slot = elems_per_slot;
    h_meta.io_multi = io_multi;
    h_meta.sd_low = params.sd_low;
    h_meta.sd_high = params.sd_high;
    h_meta.revenue_qt_max = params.revenue_qt_max;

    BAMKernelMeta* d_meta;
    BAM_CUDA_CHECK(cudaMalloc(&d_meta, sizeof(BAMKernelMeta)));
    BAM_CUDA_CHECK(cudaMemcpy(d_meta, &h_meta, sizeof(BAMKernelMeta),
                               cudaMemcpyHostToDevice));

    int64_t* d_revenue;
    BAM_CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    BAMDiagCounters* d_diag;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, sizeof(BAMDiagCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, sizeof(BAMDiagCounters)));

    BAMPerfCounters* d_perf;
    BAM_CUDA_CHECK(cudaMalloc(&d_perf, sizeof(BAMPerfCounters)));
    BAM_CUDA_CHECK(cudaMemset(d_perf, 0, sizeof(BAMPerfCounters)));

    size_t old_stack_size = 0;
    cudaDeviceGetLimit(&old_stack_size, cudaLimitStackSize);
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    cudaStream_t stream;
    BAM_CUDA_CHECK(cudaStreamCreate(&stream));

    const uint32_t threads_per_block = 32;

    fprintf(stderr, "[bam_revenue] Launching %u blocks x %u threads, io_multi=%u, n_devices=%u\n",
            num_blocks, threads_per_block, io_multi, h_meta.n_devices);

    if (io_multi <= 1) {
        bam_revenue_kernel_comp_sync<<<num_blocks, threads_per_block, 0, stream>>>(
            h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
    } else {
        bam_revenue_kernel_comp_io_multi<<<num_blocks, threads_per_block, 0, stream>>>(
            h_pc.pdt.d_ctrls, d_pc, d_meta, d_revenue, d_diag, d_perf);
    }
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t h_revenue = 0;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_revenue, d_revenue, sizeof(int64_t),
                                    cudaMemcpyDeviceToHost, stream));

    BAMPerfCounters h_perf;
    BAM_CUDA_CHECK(cudaMemcpyAsync(&h_perf, d_perf, sizeof(BAMPerfCounters),
                                    cudaMemcpyDeviceToHost, stream));
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, params.cuda_device);
    double clock_ghz = clock_khz / 1e6;

    uint64_t total_cycles = h_perf.io_cycles + h_perf.decomp_cycles + h_perf.eval_cycles;
    fprintf(stderr, "\n[perf] GPU clock: %.3f GHz\n", clock_ghz);
    fprintf(stderr, "[perf] pages_processed: %lu  io_calls: %lu  io_bytes: %lu (%.2f MiB)\n",
            h_perf.page_count, h_perf.io_count,
            h_perf.io_bytes, h_perf.io_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "[perf] boundary_page_reads: %lu (%.1f%% of pages)\n",
            h_perf.boundary_count,
            h_perf.page_count > 0 ? 100.0 * h_perf.boundary_count / h_perf.page_count : 0.0);
    if (total_cycles > 0) {
        fprintf(stderr, "[perf]   io:      %12lu cycles  (%5.1f%%)\n",
                h_perf.io_cycles, 100.0 * h_perf.io_cycles / total_cycles);
        fprintf(stderr, "[perf]   decomp:  %12lu cycles  (%5.1f%%)\n",
                h_perf.decomp_cycles, 100.0 * h_perf.decomp_cycles / total_cycles);
        fprintf(stderr, "[perf]   eval:    %12lu cycles  (%5.1f%%)\n",
                h_perf.eval_cycles, 100.0 * h_perf.eval_cycles / total_cycles);
    }
    if (h_perf.io_count > 0) {
        double avg_io_us = (double)h_perf.io_cycles / h_perf.io_count / (clock_ghz * 1e3);
        fprintf(stderr, "[perf]   avg_io_latency: %.1f us/call  (%lu calls)\n",
                avg_io_us, h_perf.io_count);
    }

    BAM_CUDA_CHECK(cudaStreamDestroy(stream));

    BAM_CUDA_CHECK(cudaFree(d_meta));
    BAM_CUDA_CHECK(cudaFree(d_revenue));
    BAM_CUDA_CHECK(cudaFree(d_diag));
    BAM_CUDA_CHECK(cudaFree(d_perf));
    BAM_CUDA_CHECK(cudaFree(d_decomp_buf));
    if (d_shipdate_stats) BAM_CUDA_CHECK(cudaFree(d_shipdate_stats));
    for (int fi = 0; fi < 4; fi++) {
        BAM_CUDA_CHECK(cudaFree(d_prefix_sums[fi]));
        if (d_comp_sizes[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_sizes[fi]));
        if (d_comp_offsets[fi]) BAM_CUDA_CHECK(cudaFree(d_comp_offsets[fi]));
    }

    return BAMRunResult{h_revenue, h_perf.io_count, h_perf.io_bytes};
}

// ============================================================
// Test kernel: decompress a single compressed page on GPU
// ============================================================
__global__ void bam_test_decompress_kernel(
    char* d_page,
    int32_t* d_output,
    uint32_t* d_nalloc,
    uint16_t comp_method)
{
    __shared__ uint shared_buf[549];
    uint32_t tid = threadIdx.x;

    decompress_page_128(d_page, d_output, comp_method, shared_buf, tid, d_nalloc);
}

uint32_t bam_test_decompress_page(const void* page_buf, size_t page_buf_size,
                                   int32_t* decomp_out, uint32_t max_elems) {
    // Upload compressed page to GPU
    char* d_page;
    BAM_CUDA_CHECK(cudaMalloc(&d_page, page_buf_size));
    BAM_CUDA_CHECK(cudaMemcpy(d_page, page_buf, page_buf_size,
                               cudaMemcpyHostToDevice));

    // Allocate output buffer on GPU
    int32_t* d_output;
    BAM_CUDA_CHECK(cudaMalloc(&d_output, max_elems * sizeof(int32_t)));
    BAM_CUDA_CHECK(cudaMemset(d_output, 0, max_elems * sizeof(int32_t)));

    // Allocate nalloc output on GPU
    uint32_t* d_nalloc;
    BAM_CUDA_CHECK(cudaMalloc(&d_nalloc, sizeof(uint32_t)));

    // Launch: 1 block of 128 threads, comp_method=1 (PFOR)
    bam_test_decompress_kernel<<<1, 128>>>(d_page, d_output, d_nalloc, 1);
    BAM_CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    uint32_t h_nalloc = 0;
    BAM_CUDA_CHECK(cudaMemcpy(&h_nalloc, d_nalloc, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

    uint32_t copy_elems = (h_nalloc < max_elems) ? h_nalloc : max_elems;
    BAM_CUDA_CHECK(cudaMemcpy(decomp_out, d_output, copy_elems * sizeof(int32_t),
                               cudaMemcpyDeviceToHost));

    BAM_CUDA_CHECK(cudaFree(d_page));
    BAM_CUDA_CHECK(cudaFree(d_output));
    BAM_CUDA_CHECK(cudaFree(d_nalloc));

    return h_nalloc;
}

// ============================================================
// PFOR64 (int64) decompression
// ============================================================

__device__ void decompress_page64(
    char* page_ptr,
    int64_t* decomp_out,
    uint16_t comp_method,
    ulong* shared_buf,    // LoadBinPackTile64 uses ulong* shared buffer
    uint32_t tid,
    uint32_t* out_nalloc)
{
    volatile bam_pag_head* hdr = (volatile bam_pag_head*)page_ptr;
    uint32_t nalloc = hdr->nalloc;
    uint32_t watermark = hdr->watermark;

    if (out_nalloc && tid == 0) *out_nalloc = nalloc;

    if (comp_method != 0) {
        // PFOR64 compressed
        uint32_t nblocks = watermark / 128;
        uint32_t ntiles = nblocks / 4;
        uint32_t remaining_blocks = nblocks - ntiles * 4;
        uint32_t noffsets = nblocks + 1;

        uint32_t* block_start = (uint32_t*)(page_ptr + BAM_PAG_HDR_BYTES);

        // 8-byte alignment: if noffsets is even, skip 1 extra uint32
        uint32_t encoded_value_offset = noffsets;
        if ((noffsets & 1) == 0) encoded_value_offset++;
        ulong* data_ptr = (ulong*)(block_start + encoded_value_offset);

        // Process full tiles (4 PFOR blocks = 512 values per tile)
        // Write guard uses nalloc (not watermark) to prevent overflow into
        // adjacent pages when used by bam_flatten_pfor64_kernel with
        // prefix_sum-based contiguous output.
        for (uint32_t t = 0; t < ntiles; t++) {
            long items[4];
            LoadBinPackTile64<128, 4>(t, block_start, data_ptr,
                                      shared_buf, items, false, 512);
            for (int it = 0; it < 4; it++) {
                uint32_t idx = t * 512 + it * 128 + tid;
                if (idx < nalloc) {
                    decomp_out[idx] = items[it];
                }
            }
            __syncthreads();
        }

        // Process remaining blocks (1-3 blocks of 128 elements each).
        // Decode directly from global memory using decodeElement64.
        for (uint32_t b = 0; b < remaining_blocks; b++) {
            uint32_t block_idx = ntiles * 4 + b;
            ulong* blk_data = data_ptr + block_start[block_idx];

            // Extract per-miniblock bitwidths and offsets into shared memory
            uint* shared_buf32 = reinterpret_cast<uint*>(shared_buf);
            uint* rem_bws  = &shared_buf32[0];  // 4 entries
            uint* rem_offs = &shared_buf32[4];   // 4 entries
            if (tid < 4) {
                uint32_t mb_bw_packed = *(blk_data + 1);  // truncates ulong → uint

                // 64-bit miniblock offset: ceil(bw/2) words per miniblock
                uint bw0 = mb_bw_packed & 0xFF;
                uint bw1 = (mb_bw_packed >> 8) & 0xFF;
                uint bw2 = (mb_bw_packed >> 16) & 0xFF;
                uint h0 = (bw0 + 1) >> 1;
                uint h1 = (bw1 + 1) >> 1;
                uint h2 = (bw2 + 1) >> 1;
                uint packed_off = (h0 << 8) | ((h0 + h1) << 16) | ((h0 + h1 + h2) << 24);

                rem_bws[tid]  = (mb_bw_packed >> (tid << 3)) & 255;
                rem_offs[tid] = (packed_off >> (tid << 3)) & 255;
            }
            __syncthreads();

            ulong mb_idx = tid >> 5;
            ulong mb_pos = tid & 31;
            long val = decodeElement64(tid, mb_idx, mb_pos,
                                       blk_data, rem_bws, rem_offs);

            uint32_t idx = block_idx * 128 + tid;
            if (idx < nalloc) {
                decomp_out[idx] = val;
            }
            __syncthreads();
        }
    } else {
        // Uncompressed int64: the loader aligns int64 data to 8-byte boundary
        // (pag_head is 12B, rounded up to 16 for uint64 alignment).
        constexpr uint32_t data_off = (BAM_PAG_HDR_BYTES + 7) & ~7u;  // 16
        int64_t* src = (int64_t*)(page_ptr + data_off);
        for (uint32_t i = tid; i < nalloc; i += 128) {
            decomp_out[i] = src[i];
        }
        __syncthreads();
    }
}

// ============================================================
// Test kernel for PFOR64 decompression
// ============================================================
__global__ void bam_test_decompress_kernel64(
    char* d_page,
    int64_t* d_output,
    uint32_t* d_nalloc,
    uint16_t comp_method)
{
    __shared__ ulong shared_buf64[581];
    uint32_t tid = threadIdx.x;

    decompress_page64(d_page, d_output, comp_method, shared_buf64, tid, d_nalloc);
}

uint32_t bam_test_decompress_page64(const void* page_buf, size_t page_buf_size,
                                     int64_t* decomp_out, uint32_t max_elems) {
    char* d_page;
    BAM_CUDA_CHECK(cudaMalloc(&d_page, page_buf_size));
    BAM_CUDA_CHECK(cudaMemcpy(d_page, page_buf, page_buf_size,
                               cudaMemcpyHostToDevice));

    int64_t* d_output;
    BAM_CUDA_CHECK(cudaMalloc(&d_output, max_elems * sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_output, 0, max_elems * sizeof(int64_t)));

    uint32_t* d_nalloc;
    BAM_CUDA_CHECK(cudaMalloc(&d_nalloc, sizeof(uint32_t)));

    // Launch: 1 block of 128 threads, comp_method=2 (PFOR64)
    bam_test_decompress_kernel64<<<1, 128>>>(d_page, d_output, d_nalloc, 2);
    BAM_CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_nalloc = 0;
    BAM_CUDA_CHECK(cudaMemcpy(&h_nalloc, d_nalloc, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

    uint32_t copy_elems = (h_nalloc < max_elems) ? h_nalloc : max_elems;
    BAM_CUDA_CHECK(cudaMemcpy(decomp_out, d_output, copy_elems * sizeof(int64_t),
                               cudaMemcpyDeviceToHost));

    BAM_CUDA_CHECK(cudaFree(d_page));
    BAM_CUDA_CHECK(cudaFree(d_output));
    BAM_CUDA_CHECK(cudaFree(d_nalloc));

    return h_nalloc;
}

// ============================================================
// PFOR64 batch flatten: decompress all pages of an INT64 column
// into a contiguous flat array using prefix_sum for offsets.
// ============================================================

__global__ void bam_flatten_pfor64_kernel(
    const char* d_pages,
    const uint64_t* d_prefix_sum,
    int64_t* d_flat_output,
    uint32_t page_size,
    uint32_t npages,
    uint16_t comp_method)
{
    uint32_t page_idx = blockIdx.x;
    if (page_idx >= npages) return;

    uint32_t tid = threadIdx.x;
    char* page_ptr = (char*)(d_pages + (uint64_t)page_idx * page_size);

    // Destination offset in flat array
    uint64_t row_base = (page_idx == 0) ? 0 : d_prefix_sum[page_idx - 1];
    int64_t* out_ptr = d_flat_output + row_base;

    __shared__ ulong shared_buf64[581];
    uint32_t nalloc_dummy;
    decompress_page64(page_ptr, out_ptr, comp_method, shared_buf64, tid, &nalloc_dummy);
}

void bam_flatten_pfor64_pages(
    const char* d_pages,
    const uint64_t* d_prefix_sum,
    int64_t* d_flat_output,
    uint32_t page_size,
    uint32_t npages,
    uint16_t comp_method,
    cudaStream_t stream)
{
    if (npages == 0) return;
    bam_flatten_pfor64_kernel<<<npages, 128, 0, stream>>>(
        d_pages, d_prefix_sum, d_flat_output,
        page_size, npages, comp_method);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// VCHAR I/O kernel + context
//
// Reads LZ4-compressed VCHAR pages via BAM into a GPU staging
// buffer.  The decompress+scan step is in bam_vchar_kernel.cu
// (C++17 / nvCOMPdx) — kept separate to avoid header conflicts.
// ============================================================

#define NVM_CTRL_PAGE_BLOCKS_VCHAR_IO 8

__device__ __forceinline__ uint32_t safe_io_nblocks_vchar_io(uint32_t comp_bytes) {
    uint32_t nblk = (comp_bytes + 511) / 512;
    if (nblk > NVM_CTRL_PAGE_BLOCKS_VCHAR_IO && nblk <= NVM_CTRL_PAGE_BLOCKS_VCHAR_IO * 2)
        nblk = NVM_CTRL_PAGE_BLOCKS_VCHAR_IO * 2 + 1;
    return nblk;
}

__global__ void bam_vchar_io_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    char*           d_staging_buf,
    uint32_t        page_size,
    uint64_t        partition_start_lba,
    uint64_t        batch_start,
    const uint64_t* partition_start_lbas,
    uint32_t        n_devices,
    uint64_t        field_start_page_id)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t pg  = batch_start + bid;

    const uint32_t ndev = (n_devices > 1) ? n_devices : 1;

    char* base = (char*)pc->base_addr;

    // Phase 1: NVMe I/O (lane 0 only)
    if (tid == 0) {
        uint32_t comp_size = d_comp_sizes[pg];
        uint32_t nblk = safe_io_nblocks_vchar_io(comp_size);
        unsigned long long pc_entry = (unsigned long long)bid;

        uint64_t lba;
        uint32_t dev;
        if (ndev > 1) {
            uint64_t global_pg = field_start_page_id + pg;
            dev = global_pg % ndev;
            lba = partition_start_lbas[dev] + d_comp_offsets[pg] / 512;
        } else {
            dev = 0;
            lba = partition_start_lba + d_comp_offsets[pg] / 512;
        }

        QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);

        uint16_t cid = 0;
        uint16_t sq_pos = 0;
        access_data_async(pc, qp, lba, nblk, pc_entry,
                          NVM_IO_READ, &cid, &sq_pos);

        uint32_t poll_loc, poll_head;
        uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
        cq_dequeue(&qp->cq, cq_pos, &qp->sq);
        put_cid(&qp->sq, cid);
    }
    __syncthreads();

    // Phase 2: Copy from page cache to staging buffer (all threads)
    const char* src = base + (unsigned long long)bid * page_size;
    char*       dst = d_staging_buf + (unsigned long long)bid * page_size;
    uint32_t comp_size = d_comp_sizes[pg];

    for (uint32_t i = tid; i < comp_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ── Context struct ──
struct BAMVcharIOContext {
    page_cache_t* h_pc;
    char*         d_staging_buf;
    uint32_t      page_size;
    uint32_t      num_blocks;
    cudaStream_t  stream;
};

bam_vchar_io_ctx_t bam_vchar_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);

    const uint64_t n_pc_pages = (uint64_t)num_blocks;
    const uint64_t max_range = 64;

    auto* ctx = new BAMVcharIOContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // Create page cache
    ctx->h_pc = new page_cache_t(page_size, n_pc_pages, h->cuda_device,
                                  *h->ctrl, max_range, h->ctrls);

    // Allocate staging buffer
    size_t staging_size = (size_t)num_blocks * page_size;
    BAM_CUDA_CHECK(cudaMalloc(&ctx->d_staging_buf, staging_size));

    // Create stream
    BAM_CUDA_CHECK(cudaStreamCreate(&ctx->stream));

    // Set stack size for BAM I/O
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    return static_cast<bam_vchar_io_ctx_t>(ctx);
}

char* bam_vchar_io_staging_buf(bam_vchar_io_ctx_t ctx_handle) {
    auto* ctx = static_cast<BAMVcharIOContext*>(ctx_handle);
    return ctx->d_staging_buf;
}

void bam_vchar_io_read_batch(
    bam_vchar_io_ctx_t ctx_handle,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    uint64_t partition_start_lba,
    uint64_t batch_start,
    uint32_t batch_size,
    const uint64_t* partition_start_lbas,
    uint32_t n_devices,
    uint64_t field_start_page_id)
{
    auto* ctx = static_cast<BAMVcharIOContext*>(ctx_handle);

    const uint32_t threads_per_block = 32;  // 1 warp for I/O + copy

    bam_vchar_io_kernel<<<batch_size, threads_per_block, 0, ctx->stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        d_comp_sizes, d_comp_offsets, ctx->d_staging_buf,
        ctx->page_size, partition_start_lba, batch_start,
        partition_start_lbas, n_devices, field_start_page_id);

    BAM_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

void bam_vchar_io_destroy(bam_vchar_io_ctx_t ctx_handle) {
    auto* ctx = static_cast<BAMVcharIOContext*>(ctx_handle);
    BAM_CUDA_CHECK(cudaFree(ctx->d_staging_buf));
    BAM_CUDA_CHECK(cudaStreamDestroy(ctx->stream));
    delete ctx->h_pc;
    delete ctx;
}

// ============================================================
// VCHAR I/O kernel v2: 128 threads (4 warps), each warp reads 1 page.
// One block processes 4 pages.
// d_staging_buf layout: [num_blocks * 4 * page_size]
//   Slot = bid * 4 + warp_id
// ============================================================

__global__ void bam_vchar_io_kernel_v2(
    Controller**    ctrls,
    page_cache_d_t* pc,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    char*           d_staging_buf,
    uint32_t        page_size,
    uint64_t        partition_start_lba,
    uint64_t        batch_start,
    uint64_t        npages_total,
    const uint64_t* partition_start_lbas,
    uint32_t        n_devices,
    uint64_t        field_start_page_id)
{
    const uint32_t bid     = blockIdx.x;
    const uint32_t tid     = threadIdx.x;       // 0..127
    const uint32_t warp_id = tid / 32;          // 0..3
    const uint32_t lane    = tid % 32;

    const uint32_t ndev = (n_devices > 1) ? n_devices : 1;

    // Global page index for this warp
    const uint64_t pg = batch_start + (uint64_t)bid * 4 + warp_id;

    // Skip if this warp's page is out of range
    if (pg >= npages_total) return;

    // Page cache slot: bid * 4 + warp_id
    const uint32_t slot = bid * 4 + warp_id;

    char* base = (char*)pc->base_addr;

    // Phase 1: NVMe I/O (lane 0 of each warp)
    if (lane == 0) {
        uint32_t comp_size = d_comp_sizes[pg];
        uint32_t nblk = safe_io_nblocks_vchar_io(comp_size);
        unsigned long long pc_entry = (unsigned long long)slot;

        uint64_t lba;
        uint32_t dev;
        if (ndev > 1) {
            uint64_t global_pg = field_start_page_id + pg;
            dev = global_pg % ndev;
            lba = partition_start_lbas[dev] + d_comp_offsets[pg] / 512;
        } else {
            dev = 0;
            lba = partition_start_lba + d_comp_offsets[pg] / 512;
        }

        QueuePair* qp = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);

        uint16_t cid = 0;
        uint16_t sq_pos = 0;
        access_data_async(pc, qp, lba, nblk, pc_entry,
                          NVM_IO_READ, &cid, &sq_pos);

        uint32_t poll_loc, poll_head;
        uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
        cq_dequeue(&qp->cq, cq_pos, &qp->sq);
        put_cid(&qp->sq, cid);
    }
    // Sync within warp (all 32 threads wait for lane 0's I/O)
    __syncwarp();

    // Phase 2: Copy from page cache to staging buffer (all 32 threads in this warp)
    const char* src = base + (unsigned long long)slot * page_size;
    char*       dst = d_staging_buf + (unsigned long long)slot * page_size;
    uint32_t comp_size = d_comp_sizes[pg];

    for (uint32_t i = lane; i < comp_size; i += 32) {
        dst[i] = src[i];
    }
}

// ── v2 Context struct ──
struct BAMVcharIOContextV2 {
    page_cache_t* h_pc;
    char*         d_staging_buf;
    uint32_t      page_size;
    uint32_t      num_blocks;    // grid size (each block handles 4 pages)
    cudaStream_t  stream;
};

bam_vchar_io_ctx_t bam_vchar_io_v2_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);

    // Page cache needs num_blocks * 4 entries (4 pages per block)
    const uint64_t n_pc_pages = (uint64_t)num_blocks * 4;
    const uint64_t max_range = 64;

    auto* ctx = new BAMVcharIOContextV2();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // Create page cache
    ctx->h_pc = new page_cache_t(page_size, n_pc_pages, h->cuda_device,
                                  *h->ctrl, max_range, h->ctrls);

    // Allocate staging buffer: num_blocks * 4 * page_size
    size_t staging_size = (size_t)num_blocks * 4 * page_size;
    BAM_CUDA_CHECK(cudaMalloc(&ctx->d_staging_buf, staging_size));

    // Create stream
    BAM_CUDA_CHECK(cudaStreamCreate(&ctx->stream));

    // Set stack size for BAM I/O
    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    return static_cast<bam_vchar_io_ctx_t>(ctx);
}

char* bam_vchar_io_v2_staging_buf(bam_vchar_io_ctx_t ctx_handle) {
    auto* ctx = static_cast<BAMVcharIOContextV2*>(ctx_handle);
    return ctx->d_staging_buf;
}

void bam_vchar_io_v2_read_batch(
    bam_vchar_io_ctx_t ctx_handle,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    uint64_t partition_start_lba,
    uint64_t batch_start,
    uint32_t batch_blocks,
    uint64_t npages_total,
    const uint64_t* partition_start_lbas,
    uint32_t n_devices,
    uint64_t field_start_page_id)
{
    auto* ctx = static_cast<BAMVcharIOContextV2*>(ctx_handle);

    const uint32_t threads_per_block = 128;  // 4 warps

    bam_vchar_io_kernel_v2<<<batch_blocks, threads_per_block, 0, ctx->stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        d_comp_sizes, d_comp_offsets, ctx->d_staging_buf,
        ctx->page_size, partition_start_lba, batch_start, npages_total,
        partition_start_lbas, n_devices, field_start_page_id);

    BAM_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

void bam_vchar_io_v2_read_batch_async(
    bam_vchar_io_ctx_t ctx_handle,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    uint64_t partition_start_lba,
    uint64_t batch_start,
    uint32_t batch_blocks,
    uint64_t npages_total,
    const uint64_t* partition_start_lbas,
    uint32_t n_devices,
    uint64_t field_start_page_id)
{
    auto* ctx = static_cast<BAMVcharIOContextV2*>(ctx_handle);

    const uint32_t threads_per_block = 128;

    bam_vchar_io_kernel_v2<<<batch_blocks, threads_per_block, 0, ctx->stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        d_comp_sizes, d_comp_offsets, ctx->d_staging_buf,
        ctx->page_size, partition_start_lba, batch_start, npages_total,
        partition_start_lbas, n_devices, field_start_page_id);
    // No synchronize — caller controls timing via bam_vchar_io_v2_sync.
}

void bam_vchar_io_v2_sync(bam_vchar_io_ctx_t ctx_handle) {
    auto* ctx = static_cast<BAMVcharIOContextV2*>(ctx_handle);
    BAM_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

void bam_vchar_io_v2_destroy(bam_vchar_io_ctx_t ctx_handle) {
    auto* ctx = static_cast<BAMVcharIOContextV2*>(ctx_handle);
    BAM_CUDA_CHECK(cudaFree(ctx->d_staging_buf));
    BAM_CUDA_CHECK(cudaStreamDestroy(ctx->stream));
    delete ctx->h_pc;
    delete ctx;
}

// ============================================================
// PFOR64 GPU-initiated I/O + flatten
//
// Reads PFOR64-compressed (or uncompressed) INT64 pages via
// BAM NVMe I/O and decompresses directly into a flat int64_t
// output array.  Block-stride loop: each block reuses its
// page cache entry across multiple pages.
// ============================================================

__global__ void bam_pfor64_io_flatten_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMPfor64FlattenParams p,
    int64_t*        d_flat_output)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ ulong shared_buf64[581];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint64_t pg = bid; pg < p.npages; pg += gridDim.x) {
        // Phase 1: NVMe I/O (thread 0 only)
        if (tid == 0) {
            uint64_t global_pg = p.field_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            if (p.comp_method != 0 && p.d_comp_offsets) {
                lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[pg]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                nblk = p.blocks_per_page;
            }
            QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);
            uint16_t cid = 0;
            uint16_t sq_pos = 0;
            access_data_async(pc, qp, lba, nblk, (unsigned long long)bid,
                              NVM_IO_READ, &cid, &sq_pos);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, cid);
        }
        __syncthreads();

        // Phase 2: Decompress directly into flat output
        char* page_ptr = (char*)pc->base_addr
                       + (unsigned long long)bid * p.page_size;
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        int64_t* out_ptr = d_flat_output + row_base;

        uint32_t nalloc_dummy;
        decompress_page64(page_ptr, out_ptr, p.comp_method,
                          shared_buf64, tid, &nalloc_dummy);
        __syncthreads();
    }
}

// ── PFOR64 I/O Context ──
struct BAMPfor64IOContext {
    page_cache_t* h_pc;
    uint32_t      page_size;
    uint32_t      num_blocks;
};

bam_pfor64_io_ctx_t bam_pfor64_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    auto* ctx = new BAMPfor64IOContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // Page cache: one entry per block (block-stride reuse)
    const uint64_t n_pc_pages = (uint64_t)num_blocks;
    const uint64_t max_range = 64;
    ctx->h_pc = new page_cache_t(page_size, n_pc_pages, h->cuda_device,
                                  *h->ctrl, max_range, h->ctrls);

    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    return static_cast<bam_pfor64_io_ctx_t>(ctx);
}

void bam_pfor64_io_flatten(
    bam_pfor64_io_ctx_t ctx_handle,
    const BAMPfor64FlattenParams& params,
    int64_t* d_flat_output)
{
    auto* ctx = static_cast<BAMPfor64IOContext*>(ctx_handle);

    bam_pfor64_io_flatten_kernel<<<params.num_blocks, 128>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_flat_output);
    BAM_CUDA_CHECK(cudaGetLastError());
    BAM_CUDA_CHECK(cudaDeviceSynchronize());
}

void bam_pfor64_io_flatten_async(
    bam_pfor64_io_ctx_t ctx_handle,
    const BAMPfor64FlattenParams& params,
    int64_t* d_flat_output,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor64IOContext*>(ctx_handle);

    bam_pfor64_io_flatten_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_flat_output);
    BAM_CUDA_CHECK(cudaGetLastError());
}

void bam_pfor64_io_destroy(bam_pfor64_io_ctx_t ctx_handle) {
    auto* ctx = static_cast<BAMPfor64IOContext*>(ctx_handle);
    delete ctx->h_pc;
    delete ctx;
}

// ============================================================
// PFOR64 dual-field flatten: two INT64 fields in one kernel
// ============================================================

__global__ void bam_pfor64_dual_flatten_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMPfor64DualFlattenParams p)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ ulong shared_buf64[581];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint64_t pg = bid; pg < p.npages; pg += gridDim.x) {
        char* page_ptr = (char*)pc->base_addr
                       + (unsigned long long)bid * p.page_size;

        // ── Field 0: IO + decompress + flatten ──
        if (tid == 0) {
            uint64_t global_pg = p.field0_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            if (p.field0_comp_method != 0 && p.field0_d_comp_offsets) {
                lba = p.partition_start_lbas[dev] + p.field0_d_comp_offsets[pg] / 512;
                nblk = safe_io_nblocks_vchar_io(p.field0_d_comp_sizes[pg]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                nblk = p.blocks_per_page;
            }
            QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);
            uint16_t cid = 0, sq_pos = 0;
            access_data_async(pc, qp, lba, nblk, (unsigned long long)bid,
                              NVM_IO_READ, &cid, &sq_pos);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, cid);
        }
        __syncthreads();
        {
            uint64_t row_base = (pg == 0) ? 0 : p.field0_d_prefix_sum[pg - 1];
            uint32_t nalloc_dummy;
            decompress_page64(page_ptr, p.field0_d_output + row_base,
                              p.field0_comp_method, shared_buf64, tid, &nalloc_dummy);
        }
        __syncthreads();

        // ── Field 1: IO + decompress + flatten (reuse same page cache slot) ──
        if (tid == 0) {
            uint64_t global_pg = p.field1_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            if (p.field1_comp_method != 0 && p.field1_d_comp_offsets) {
                lba = p.partition_start_lbas[dev] + p.field1_d_comp_offsets[pg] / 512;
                nblk = safe_io_nblocks_vchar_io(p.field1_d_comp_sizes[pg]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                nblk = p.blocks_per_page;
            }
            QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);
            uint16_t cid = 0, sq_pos = 0;
            access_data_async(pc, qp, lba, nblk, (unsigned long long)bid,
                              NVM_IO_READ, &cid, &sq_pos);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, cid);
        }
        __syncthreads();
        {
            uint64_t row_base = (pg == 0) ? 0 : p.field1_d_prefix_sum[pg - 1];
            uint32_t nalloc_dummy;
            decompress_page64(page_ptr, p.field1_d_output + row_base,
                              p.field1_comp_method, shared_buf64, tid, &nalloc_dummy);
        }
        __syncthreads();
    }
}

void bam_pfor64_dual_flatten_async(
    bam_pfor64_io_ctx_t ctx_handle,
    const BAMPfor64DualFlattenParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor64IOContext*>(ctx_handle);

    bam_pfor64_dual_flatten_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr, params);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// PFOR32 GPU-initiated I/O + flatten (INT32 variant)
// ============================================================

__global__ void bam_pfor32_io_flatten_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMPfor32FlattenParams p,
    int32_t*        d_flat_output)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint64_t pg = bid; pg < p.npages; pg += gridDim.x) {
        // Phase 1: NVMe I/O (thread 0 only)
        if (tid == 0) {
            uint64_t global_pg = p.field_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            if (p.comp_method != 0 && p.d_comp_offsets) {
                lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[pg]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                nblk = p.blocks_per_page;
            }
            QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);
            uint16_t cid = 0;
            uint16_t sq_pos = 0;
            access_data_async(pc, qp, lba, nblk, (unsigned long long)bid,
                              NVM_IO_READ, &cid, &sq_pos);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, cid);
        }
        __syncthreads();

        // Phase 2: Decompress directly into flat output
        char* page_ptr = (char*)pc->base_addr
                       + (unsigned long long)bid * p.page_size;
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        int32_t* out_ptr = d_flat_output + row_base;

        uint32_t nalloc_dummy;
        decompress_page_128(page_ptr, out_ptr, p.comp_method,
                            shared_buf32, tid, &nalloc_dummy);
        __syncthreads();
    }
}

// ── PFOR32 I/O Context ──
struct BAMPfor32IOContext {
    page_cache_t* h_pc;
    uint32_t      page_size;
    uint32_t      num_blocks;
};

bam_pfor32_io_ctx_t bam_pfor32_io_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks)
{
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);
    auto* ctx = new BAMPfor32IOContext();
    ctx->page_size = page_size;
    ctx->num_blocks = num_blocks;

    // Page cache: one entry per block (block-stride reuse)
    const uint64_t n_pc_pages = (uint64_t)num_blocks;
    const uint64_t max_range = 64;
    ctx->h_pc = new page_cache_t(page_size, n_pc_pages, h->cuda_device,
                                  *h->ctrl, max_range, h->ctrls);

    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    return static_cast<bam_pfor32_io_ctx_t>(ctx);
}

void* bam_pfor32_io_get_d_ctrls(bam_pfor32_io_ctx_t h) {
    return static_cast<BAMPfor32IOContext*>(h)->h_pc->pdt.d_ctrls;
}
void* bam_pfor32_io_get_d_pc_ptr(bam_pfor32_io_ctx_t h) {
    return static_cast<BAMPfor32IOContext*>(h)->h_pc->d_pc_ptr;
}
const char* bam_pfor32_io_get_pc_base(bam_pfor32_io_ctx_t h) {
    return (const char*)static_cast<BAMPfor32IOContext*>(h)->h_pc->pdt.base_addr;
}
uint32_t bam_pfor32_io_get_num_slots(bam_pfor32_io_ctx_t h) {
    return static_cast<BAMPfor32IOContext*>(h)->num_blocks;
}

void bam_pfor32_io_flatten(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMPfor32FlattenParams& params,
    int32_t* d_flat_output)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);

    bam_pfor32_io_flatten_kernel<<<params.num_blocks, 128>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_flat_output);
    BAM_CUDA_CHECK(cudaGetLastError());
    BAM_CUDA_CHECK(cudaDeviceSynchronize());
}

void bam_pfor32_io_flatten_async(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMPfor32FlattenParams& params,
    int32_t* d_flat_output,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);

    bam_pfor32_io_flatten_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_flat_output);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ── PFOR32 I/O + flatten + widen to uint64_t (two-buffer) ──
// Decompresses INT32 pages into d_temp_i32, then widens to d_flat_output (uint64_t).
// Uses separate buffers to avoid the in-place widen race condition.
__global__ void bam_pfor32_io_flatten_widen_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMPfor32FlattenParams p,
    int32_t*        d_temp_i32,
    uint64_t*       d_flat_output)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint64_t pg = bid; pg < p.npages; pg += gridDim.x) {
        // Phase 1: NVMe I/O (thread 0 only)
        if (tid == 0) {
            uint64_t global_pg = p.field_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            if (p.comp_method != 0 && p.d_comp_offsets) {
                lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[pg]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                nblk = p.blocks_per_page;
            }
            QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);
            uint16_t cid = 0;
            uint16_t sq_pos = 0;
            access_data_async(pc, qp, lba, nblk, (unsigned long long)bid,
                              NVM_IO_READ, &cid, &sq_pos);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, cid);
        }
        __syncthreads();

        // Phase 2: Decompress PFOR32 into separate INT32 buffer
        char* page_ptr = (char*)pc->base_addr
                       + (unsigned long long)bid * p.page_size;
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        int32_t* out_i32 = d_temp_i32 + row_base;

        uint32_t nalloc_dummy;
        decompress_page_128(page_ptr, out_i32, p.comp_method,
                            shared_buf32, tid, &nalloc_dummy);
        __syncthreads();

        // Phase 3: Widen int32 → uint64 (separate buffers, no aliasing)
        uint32_t nalloc = *reinterpret_cast<const uint32_t*>(page_ptr);
        for (uint32_t i = tid; i < nalloc; i += blockDim.x)
            d_flat_output[row_base + i] = static_cast<uint64_t>(
                static_cast<uint32_t>(out_i32[i]));
        __syncthreads();
    }
}

void bam_pfor32_io_flatten_widen_async(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMPfor32FlattenParams& params,
    int32_t* d_temp_i32,
    uint64_t* d_flat_output,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);

    bam_pfor32_io_flatten_widen_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_temp_i32, d_flat_output);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ── PFOR32 I/O + decompress + nationkey filter + HT build (Q5 dim tables) ──
// Decompresses INT32 NK pages, filters by nationkey, reads KEY from pre-flattened
// uint64 buffer, and inserts into hash table — all in one kernel.
// Forward declarations for Q5 HT primitives (defined later in this file).
__device__ __forceinline__ uint32_t q5f_hash64(uint64_t key);
__device__ __forceinline__ void q5f_ht_insert(
    uint64_t *keys, int32_t *values, uint32_t mask,
    uint64_t key, int32_t value);

__global__ void bam_pfor32_io_nk_ht_build_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMPfor32FlattenParams p,
    int32_t*        d_temp_i32,
    const uint64_t* d_key_flat,
    const int8_t*   d_nationkey_to_idx,
    uint64_t*       ht_keys,
    int32_t*        ht_values,
    uint32_t        ht_mask)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint64_t pg = bid; pg < p.npages; pg += gridDim.x) {
        if (tid == 0) {
            uint64_t global_pg = p.field_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            if (p.comp_method != 0 && p.d_comp_offsets) {
                lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[pg]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                nblk = p.blocks_per_page;
            }
            QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);
            uint16_t cid = 0;
            uint16_t sq_pos = 0;
            access_data_async(pc, qp, lba, nblk, (unsigned long long)bid,
                              NVM_IO_READ, &cid, &sq_pos);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, cid);
        }
        __syncthreads();

        char* page_ptr = (char*)pc->base_addr
                       + (unsigned long long)bid * p.page_size;
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        int32_t* out_i32 = d_temp_i32 + row_base;

        uint32_t nalloc_dummy;
        decompress_page_128(page_ptr, out_i32, p.comp_method,
                            shared_buf32, tid, &nalloc_dummy);
        __syncthreads();

        uint32_t nalloc = *reinterpret_cast<const uint32_t*>(page_ptr);
        for (uint32_t i = tid; i < nalloc; i += blockDim.x) {
            int32_t nk = out_i32[i];
            if (nk < 0 || nk >= 25) continue;
            int8_t nation_idx = d_nationkey_to_idx[nk];
            if (nation_idx < 0) continue;
            uint64_t key = d_key_flat[row_base + i];
            q5f_ht_insert(ht_keys, ht_values, ht_mask,
                              key, (int32_t)nation_idx);
        }
        __syncthreads();
    }
}

void bam_pfor32_io_nk_ht_build_async(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMPfor32FlattenParams& params,
    int32_t* d_temp_i32,
    const uint64_t* d_key_flat,
    const int8_t* d_nationkey_to_idx,
    uint64_t* ht_keys,
    int32_t* ht_values,
    uint32_t ht_mask,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);

    bam_pfor32_io_nk_ht_build_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_temp_i32, d_key_flat, d_nationkey_to_idx,
        ht_keys, ht_values, ht_mask);
    BAM_CUDA_CHECK(cudaGetLastError());
}

void bam_pfor32_io_destroy(bam_pfor32_io_ctx_t ctx_handle) {
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    delete ctx->h_pc;
    delete ctx;
}

// ============================================================
// Masked PFOR64/PFOR32 flatten: zone-map IO pruning variants.
// Skip NVMe read for inactive pages; fill output with fill_value.
// ============================================================

__global__ void bam_pfor64_io_flatten_masked_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMPfor64FlattenParams p,
    int64_t*        d_flat_output,
    const uint8_t*  d_page_active,
    int64_t         fill_value)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ ulong shared_buf64[581];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint64_t pg = bid; pg < p.npages; pg += gridDim.x) {
        // Zone map check: skip NVMe read for inactive pages
        if (d_page_active && !d_page_active[pg]) {
            uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
            uint64_t row_end  = p.d_prefix_sum[pg];
            for (uint64_t r = row_base + tid; r < row_end; r += blockDim.x)
                d_flat_output[r] = fill_value;
            __syncthreads();
            continue;
        }

        // Phase 1: NVMe I/O (thread 0 only)
        if (tid == 0) {
            uint64_t global_pg = p.field_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            if (p.comp_method != 0 && p.d_comp_offsets) {
                lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[pg]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                nblk = p.blocks_per_page;
            }
            QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);
            uint16_t cid = 0;
            uint16_t sq_pos = 0;
            access_data_async(pc, qp, lba, nblk, (unsigned long long)bid,
                              NVM_IO_READ, &cid, &sq_pos);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, cid);
        }
        __syncthreads();

        // Phase 2: Decompress directly into flat output
        char* page_ptr = (char*)pc->base_addr
                       + (unsigned long long)bid * p.page_size;
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        int64_t* out_ptr = d_flat_output + row_base;

        uint32_t nalloc_dummy;
        decompress_page64(page_ptr, out_ptr, p.comp_method,
                          shared_buf64, tid, &nalloc_dummy);
        __syncthreads();
    }
}

void bam_pfor64_io_flatten_masked_async(
    bam_pfor64_io_ctx_t ctx_handle,
    const BAMPfor64FlattenParams& params,
    const uint8_t* d_page_active,
    int64_t fill_value,
    int64_t* d_flat_output,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor64IOContext*>(ctx_handle);

    bam_pfor64_io_flatten_masked_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_flat_output, d_page_active, fill_value);
    BAM_CUDA_CHECK(cudaGetLastError());
}

__global__ void bam_pfor32_io_flatten_masked_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMPfor32FlattenParams p,
    int32_t*        d_flat_output,
    const uint8_t*  d_page_active,
    int32_t         fill_value)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint64_t pg = bid; pg < p.npages; pg += gridDim.x) {
        // Zone map check: skip NVMe read for inactive pages
        if (d_page_active && !d_page_active[pg]) {
            uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
            uint64_t row_end  = p.d_prefix_sum[pg];
            for (uint64_t r = row_base + tid; r < row_end; r += blockDim.x)
                d_flat_output[r] = fill_value;
            __syncthreads();
            continue;
        }

        // Phase 1: NVMe I/O (thread 0 only)
        if (tid == 0) {
            uint64_t global_pg = p.field_start_page_id + pg;
            uint32_t dev = global_pg % ndev;
            uint64_t lba;
            uint32_t nblk;
            if (p.comp_method != 0 && p.d_comp_offsets) {
                lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[pg]);
            } else {
                uint64_t local_pg = global_pg / ndev;
                lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                nblk = p.blocks_per_page;
            }
            QueuePair* qp = ctrls[dev]->d_qps + (bid % ctrls[dev]->n_qps);
            uint16_t cid = 0;
            uint16_t sq_pos = 0;
            access_data_async(pc, qp, lba, nblk, (unsigned long long)bid,
                              NVM_IO_READ, &cid, &sq_pos);
            uint32_t poll_loc, poll_head;
            uint32_t cq_pos = cq_poll(&qp->cq, cid, &poll_loc, &poll_head);
            cq_dequeue(&qp->cq, cq_pos, &qp->sq);
            put_cid(&qp->sq, cid);
        }
        __syncthreads();

        // Phase 2: Decompress directly into flat output
        char* page_ptr = (char*)pc->base_addr
                       + (unsigned long long)bid * p.page_size;
        uint64_t row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
        int32_t* out_ptr = d_flat_output + row_base;

        uint32_t nalloc_dummy;
        decompress_page_128(page_ptr, out_ptr, p.comp_method,
                            shared_buf32, tid, &nalloc_dummy);
        __syncthreads();
    }
}

void bam_pfor32_io_flatten_masked_async(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMPfor32FlattenParams& params,
    const uint8_t* d_page_active,
    int32_t fill_value,
    int32_t* d_flat_output,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);

    bam_pfor32_io_flatten_masked_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_flat_output, d_page_active, fill_value);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Fused revenue kernel: batch-read 4 fields → decompress → evaluate
// ============================================================
__global__ void bam_revenue_pfor_fused_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMRevenueFusedParams p,
    int64_t*        d_revenue)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    int64_t local_rev = 0;

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;

        uint32_t nrows = (pg == 0) ? p.d_prefix_sum[0]
                                   : p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1];

        // ── Batch-submit 4 NVMe reads (one per field) ──
        uint16_t cids[4];
        QueuePair* qps_arr[4];

        if (tid == 0) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);

                uint64_t lba;
                uint32_t nblk;
                if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[fi][pg]);
                } else {
                    uint64_t local_pg = global_pg / ndev;
                    lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            // Poll all 4 completions
            for (int fi = 0; fi < 4; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
            __threadfence_system();
        }
        __syncthreads();

        // ── Decompress / read 4 fields ──
        int32_t* field_data[4];
        bool all_uncomp = (p.comp_methods[0] == 0 && p.comp_methods[1] == 0 &&
                           p.comp_methods[2] == 0 && p.comp_methods[3] == 0);

        if (all_uncomp) {
            // Uncompressed: point directly into page cache (no scratch copy)
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                char* page_ptr = (char*)pc->base_addr + slot * p.page_size;
                field_data[fi] = (int32_t*)(page_ptr + BAM_PAG_HDR_BYTES);
            }
        } else {
            // Compressed: decompress to per-block scratch
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                char* page_ptr = (char*)pc->base_addr + slot * p.page_size;
                field_data[fi] = p.d_scratch
                    + ((uint64_t)bid * 4 + fi) * p.scratch_stride;
                decompress_page_128(page_ptr, field_data[fi], p.comp_methods[fi],
                                    shared_buf32, tid, nullptr);
                __syncthreads();
            }
        }

        // ── Revenue/Q6 evaluation ──
        for (uint64_t i = tid; i < nrows; i += blockDim.x) {
            int32_t sd = field_data[0][i];
            bool pass = (sd >= p.sd_low && sd < p.sd_high);
            if (p.qt_max > 0) {
                pass = pass && (field_data[1][i] < p.qt_max);
            }
            if (p.dc_high > 0) {
                pass = pass && (field_data[3][i] >= p.dc_low
                             && field_data[3][i] <= p.dc_high);
            }
            if (pass) {
                local_rev += (int64_t)field_data[2][i] * field_data[3][i];
            }
        }
        __syncthreads();  // ensure all threads done before next iteration's DMA
    }

    // Block-level reduction
    __shared__ int64_t shared_rev[128];
    shared_rev[tid] = local_rev;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) shared_rev[tid] += shared_rev[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                  static_cast<unsigned long long>(shared_rev[0]));
    }
}

void bam_revenue_fused_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMRevenueFusedParams& params,
    int64_t* d_revenue,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);

    bam_revenue_pfor_fused_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_revenue);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Synchronous single-block Q6 test kernel
// 1 block, 128 threads. Reads pages one-by-one, decompresses
// field 0 (L_SHIPDATE) to check for DMA reliability.
// Processes ALL 4 fields per page, evaluates Q6 in-line.
// ============================================================
__global__ void bam_revenue_sync_test_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMRevenueFusedParams p,
    int64_t*        d_revenue,
    int64_t*        d_diag)   // [0]=count_sd, [1]=count_sd_qty, [2]=count_all, [3]=rev_no_disc
{
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    int64_t local_rev = 0;
    int64_t local_count_sd = 0, local_count_sd_qty = 0;
    int64_t local_count_all = 0, local_rev_no_disc = 0;

    if (tid == 0) {
        printf("[SYNC-DIAG] sd_low=%d sd_high=%d qt_max=%d dc_low=%d dc_high=%d\n",
               p.sd_low, p.sd_high, p.qt_max, p.dc_low, p.dc_high);
        printf("[SYNC-DIAG] npages=%lu scratch_stride=%u n_devices=%u\n",
               (unsigned long)p.npages, p.scratch_stride, p.n_devices);
        printf("[SYNC-DIAG] field_start_page_ids: %lu %lu %lu %lu\n",
               (unsigned long)p.field_start_page_ids[0],
               (unsigned long)p.field_start_page_ids[1],
               (unsigned long)p.field_start_page_ids[2],
               (unsigned long)p.field_start_page_ids[3]);
        printf("[SYNC-DIAG] comp_methods: %u %u %u %u\n",
               (unsigned)p.comp_methods[0], (unsigned)p.comp_methods[1],
               (unsigned)p.comp_methods[2], (unsigned)p.comp_methods[3]);
    }
    __syncthreads();

    const uint64_t _loop_n_sync = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = 0; _iter < _loop_n_sync; _iter++) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;

        uint64_t nrows = (pg == 0) ? p.d_prefix_sum[0]
                                   : p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1];

        // Read + decompress 4 fields sequentially into scratch
        int32_t* field_data[4];
        for (int fi = 0; fi < 4; fi++) {
            if (tid == 0) {
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                uint64_t local_pg = global_pg / ndev;
                uint64_t lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                uint32_t nblk = p.blocks_per_page;
                QueuePair* qp = ctrls[dev]->d_qps + 0;
                uint16_t cid = 0, sq_pos = 0;
                access_data_async(pc, qp, lba, nblk, 0ULL,
                                  NVM_IO_READ, &cid, &sq_pos);
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qp->cq, cid, &pl, &ph);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq);
                put_cid(&qp->sq, cid);
            }
            __syncthreads();

            char* page_ptr = (char*)pc->base_addr; // slot 0
            field_data[fi] = p.d_scratch + (uint64_t)fi * p.scratch_stride;
            uint32_t nalloc_out;
            decompress_page_128(page_ptr, field_data[fi], p.comp_methods[fi],
                                shared_buf32, tid, &nalloc_out);
            __syncthreads();

            if (fi == 0 && tid == 0) {
                nrows = nalloc_out;
            }
            __syncthreads();
        }

        // Print first qualifying page's first few values
        if (pg == 0 && tid == 0) {
            printf("[SYNC-DIAG] Page 0 nrows=%lu\n", (unsigned long)nrows);
            for (int i = 0; i < 5 && (uint64_t)i < nrows; i++) {
                printf("[SYNC-DIAG]   row %d: sd=%d qty=%d ext=%d disc=%d\n",
                       i, field_data[0][i], field_data[1][i],
                       field_data[2][i], field_data[3][i]);
            }
        }
        __syncthreads();

        // Q6 evaluation with diagnostics
        for (uint64_t i = tid; i < nrows; i += blockDim.x) {
            int32_t sd = field_data[0][i];
            int32_t qty = field_data[1][i];
            int32_t ext = field_data[2][i];
            int32_t disc = field_data[3][i];

            bool pass_sd = (sd >= p.sd_low && sd < p.sd_high);
            if (pass_sd) local_count_sd++;

            bool pass_sq = pass_sd && (qty < p.qt_max);
            if (pass_sq) local_count_sd_qty++;

            bool pass_all = pass_sq && (disc >= p.dc_low && disc <= p.dc_high);
            if (pass_all) local_count_all++;

            if (pass_sq)
                local_rev_no_disc += (int64_t)ext * disc;
            if (pass_all)
                local_rev += (int64_t)ext * disc;
        }
    }

    // Reduction for all counters
    __shared__ int64_t shared_rev[128];
    __shared__ int64_t shared_cnt[128];

    // Revenue
    shared_rev[tid] = local_rev;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) shared_rev[tid] += shared_rev[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                  static_cast<unsigned long long>(shared_rev[0]));

    // count_sd
    shared_cnt[tid] = local_count_sd;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) shared_cnt[tid] += shared_cnt[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag[0]),
                            static_cast<unsigned long long>(shared_cnt[0]));

    // count_sd_qty
    shared_cnt[tid] = local_count_sd_qty;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) shared_cnt[tid] += shared_cnt[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag[1]),
                            static_cast<unsigned long long>(shared_cnt[0]));

    // count_all (sd+qty+disc)
    shared_cnt[tid] = local_count_all;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) shared_cnt[tid] += shared_cnt[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag[2]),
                            static_cast<unsigned long long>(shared_cnt[0]));

    // rev_no_disc (sd+qty filter only)
    shared_cnt[tid] = local_rev_no_disc;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) shared_cnt[tid] += shared_cnt[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(reinterpret_cast<unsigned long long*>(&d_diag[3]),
                            static_cast<unsigned long long>(shared_cnt[0]));
}

void bam_revenue_sync_test_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMRevenueFusedParams& params,
    int64_t* d_revenue,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);

    int64_t* d_diag = nullptr;
    BAM_CUDA_CHECK(cudaMalloc(&d_diag, 4 * sizeof(int64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_diag, 0, 4 * sizeof(int64_t)));

    // 1 block, 128 threads — fully sequential page processing
    bam_revenue_sync_test_kernel<<<1, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_revenue, d_diag);
    BAM_CUDA_CHECK(cudaGetLastError());
    BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t h_diag[4];
    BAM_CUDA_CHECK(cudaMemcpy(h_diag, d_diag, 4 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    printf("[SYNC-DIAG] count_sd=%ld count_sd_qty=%ld count_all_filters=%ld\n",
           h_diag[0], h_diag[1], h_diag[2]);
    printf("[SYNC-DIAG] revenue_with_disc_filter=%ld\n", h_diag[2]); // placeholder
    printf("[SYNC-DIAG] revenue_no_disc_filter=%ld\n", h_diag[3]);

    BAM_CUDA_CHECK(cudaFree(d_diag));
}

// ============================================================
// Q5 hash table primitives (duplicated from q5_scan.cu)
// ============================================================
static constexpr uint64_t Q5F_HT_EMPTY = UINT64_MAX;

__device__ __forceinline__ uint32_t q5f_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

__device__ __forceinline__ void q5f_ht_insert(
    uint64_t *keys, int32_t *values, uint32_t mask,
    uint64_t key, int32_t value)
{
    uint32_t slot = q5f_hash64(key) & mask;
    while (true) {
        uint64_t old = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)Q5F_HT_EMPTY,
            (unsigned long long)key);
        if (old == Q5F_HT_EMPTY || old == key) {
            values[slot] = value;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

__device__ __forceinline__ int32_t q5f_ht_probe(
    const uint64_t *keys, const int32_t *values, uint32_t mask,
    uint64_t key)
{
    uint32_t slot = q5f_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return values[slot];
        if (k == Q5F_HT_EMPTY) return -1;
        slot = (slot + 1) & mask;
    }
}

// ============================================================
// Binary search in cumulative prefix sum.
// ps[0] = nrows_page_0, ps[i] = cumulative nrows up to page i.
// Returns page index containing row 'gid'.
// ============================================================
__device__ __forceinline__ uint32_t dpf_find_page(
    const uint64_t *ps, uint64_t npages, uint64_t gid)
{
    uint32_t lo = 0, hi = (uint32_t)npages;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (ps[mid] <= gid) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ============================================================
// Shared macro for INT64 BaM IO + decompress in fused kernels.
// Submits NVMe reads for INT64 pages, polls, decompresses to scratch.
// ============================================================

// Q5 ORDERS fused kernel: batch I/O + decompress + HT build
// ============================================================
__global__ void bam_q5_orders_fused_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMQ5OrdersFusedParams p)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ union { uint buf32[549]; ulong buf64[581]; } shared_decomp;
    __shared__ uint32_t s_i64_pg_lo, s_n_i64_pgs;
    __shared__ uint64_t s_i64_pg_lo_start, s_i64_pg_lo_nrows;
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    constexpr uint32_t N_I32 = 1;
    constexpr uint32_t N_I64 = 2;
    constexpr uint32_t SLOTS = N_I32 + N_I64 * 3;  // 7

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;

        uint64_t nrows = (pg == 0) ? p.d_prefix_sum[0]
                                   : p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1];
        uint64_t base_row = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

        // Thread 0: find INT64 page range
        if (tid == 0) {
            s_i64_pg_lo = dpf_find_page(p.d_prefix_sum_i64, p.npages_i64, base_row);
            uint32_t hi = dpf_find_page(p.d_prefix_sum_i64, p.npages_i64, base_row + nrows - 1);
            s_n_i64_pgs = hi - s_i64_pg_lo + 1;
            if (s_n_i64_pgs > 3) s_n_i64_pgs = 3;
            s_i64_pg_lo_start = (s_i64_pg_lo == 0) ? 0 : p.d_prefix_sum_i64[s_i64_pg_lo - 1];
            s_i64_pg_lo_nrows = p.d_prefix_sum_i64[s_i64_pg_lo] - s_i64_pg_lo_start;
        }
        __syncthreads();
        uint32_t i64_pg_lo = s_i64_pg_lo;
        uint32_t n_i64_pgs = s_n_i64_pgs;
        uint64_t i64_lo_start = s_i64_pg_lo_start;
        uint64_t i64_lo_nrows = s_i64_pg_lo_nrows;

        // Batch-submit NVMe reads: 1 INT32 + up to 6 INT64
        uint16_t cids[7];
        QueuePair* qps_arr[7];
        if (tid == 0) {
            // INT32: O_ORDERDATE
            {
                uint64_t slot = (uint64_t)bid * SLOTS;
                uint64_t global_pg = p.field_start_page_id + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[0] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba; uint32_t nblk;
                if (p.comp_method != 0 && p.d_comp_offsets) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[pg]);
                } else {
                    lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[0], lba, nblk, slot, NVM_IO_READ, &cids[0], &sq_pos);
            }
            // INT64: O_ORDERKEY, O_CUSTKEY
            uint32_t ri = N_I32;
            for (int fi = 0; fi < (int)N_I64; fi++) {
                for (uint32_t ipg = 0; ipg < n_i64_pgs; ipg++) {
                    uint64_t i64_pg = i64_pg_lo + ipg;
                    uint64_t slot = (uint64_t)bid * SLOTS + N_I32 + fi * 3 + ipg;
                    uint64_t global_pg = p.field_start_page_ids_i64[fi] + i64_pg;
                    uint32_t dev = global_pg % ndev;
                    qps_arr[ri] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                    uint64_t lba; uint32_t nblk;
                    if (p.comp_methods_i64[fi] != 0 && p.d_comp_offsets_i64[fi]) {
                        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i64[fi][i64_pg] / 512;
                        nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes_i64[fi][i64_pg]);
                    } else {
                        lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                        nblk = p.blocks_per_page;
                    }
                    uint16_t sq_pos = 0;
                    access_data_async(pc, qps_arr[ri], lba, nblk, slot, NVM_IO_READ, &cids[ri], &sq_pos);
                    ri++;
                }
            }
            // Poll all completions
            uint32_t total_reads = N_I32 + N_I64 * n_i64_pgs;
            for (uint32_t r = 0; r < total_reads; r++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[r]->cq, cids[r], &pl, &ph);
                cq_dequeue(&qps_arr[r]->cq, cq_pos, &qps_arr[r]->sq);
                put_cid(&qps_arr[r]->sq, cids[r]);
            }
        }
        __syncthreads();

        // Decompress INT32: O_ORDERDATE
        char* page_ptr = (char*)pc->base_addr + (uint64_t)bid * SLOTS * p.page_size;
        int32_t* odate_data = p.d_scratch + (uint64_t)bid * p.scratch_stride;
        uint32_t nalloc_dummy;
        decompress_page_128(page_ptr, odate_data, p.comp_method,
                           shared_decomp.buf32, tid, &nalloc_dummy);
        __syncthreads();

        // Decompress INT64 fields
        int64_t* i64_scratch[N_I64][3];
        for (int fi = 0; fi < (int)N_I64; fi++) {
            for (uint32_t ipg = 0; ipg < n_i64_pgs; ipg++) {
                uint64_t slot = (uint64_t)bid * SLOTS + N_I32 + fi * 3 + ipg;
                char* pp = (char*)pc->base_addr + slot * p.page_size;
                i64_scratch[fi][ipg] = p.d_scratch_i64
                    + ((uint64_t)bid * N_I64 * 3 + fi * 3 + ipg) * p.scratch_stride_i64;
                decompress_page64(pp, i64_scratch[fi][ipg], p.comp_methods_i64[fi],
                                 shared_decomp.buf64, tid, &nalloc_dummy);
                __syncthreads();
            }
        }

        // Date filter + CUSTOMER HT probe + ORDERS HT insert
        for (uint64_t i = tid; i < nrows; i += blockDim.x) {
            int32_t odate = odate_data[i];
            if (odate < p.date_low || odate >= p.date_high) continue;

            // Map to INT64 page using prefix sum
            uint64_t global_row = base_row + i;
            uint32_t wpg = 0;
            uint64_t off = global_row - i64_lo_start;
            if (off >= i64_lo_nrows) {
                wpg = 1;
                off = global_row - p.d_prefix_sum_i64[i64_pg_lo];
                if (n_i64_pgs >= 3) {
                    uint64_t p1_nrows = p.d_prefix_sum_i64[i64_pg_lo + 1] - p.d_prefix_sum_i64[i64_pg_lo];
                    if (off >= p1_nrows) {
                        wpg = 2;
                        off = global_row - p.d_prefix_sum_i64[i64_pg_lo + 1];
                    }
                }
            }

            uint64_t custkey = (uint64_t)i64_scratch[1][wpg][off];
            int32_t cust_nation_idx = q5f_ht_probe(
                p.d_ht_cust_keys, p.d_ht_cust_values, p.ht_cust_mask, custkey);
            if (cust_nation_idx < 0) continue;

            uint64_t orderkey = (uint64_t)i64_scratch[0][wpg][off];
            q5f_ht_insert(p.d_ht_ord_keys, p.d_ht_ord_values, p.ht_ord_mask,
                          orderkey, cust_nation_idx);
        }
    }
}

void bam_q5_orders_fused_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMQ5OrdersFusedParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    bam_q5_orders_fused_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr, params);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Q5 LINEITEM fused kernel: batch I/O + decompress + HT probe + revenue
// ============================================================
__global__ void bam_q5_lineitem_fused_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMQ5LineitemFusedParams p,
    int64_t*        d_revenue)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ union { uint buf32[549]; ulong buf64[581]; } shared_decomp;
    __shared__ uint32_t s_i64_pg_lo, s_n_i64_pgs;
    __shared__ uint64_t s_i64_pg_lo_start, s_i64_pg_lo_nrows;
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    constexpr uint32_t N_I32 = 2;
    constexpr uint32_t N_I64 = 2;
    constexpr uint32_t SLOTS = N_I32 + N_I64 * 3;  // 8

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;

        uint64_t nrows = (pg == 0) ? p.d_prefix_sum[0]
                                   : p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1];
        uint64_t base_row = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

        // Thread 0: find INT64 page range
        if (tid == 0) {
            s_i64_pg_lo = dpf_find_page(p.d_prefix_sum_i64, p.npages_i64, base_row);
            uint32_t hi = dpf_find_page(p.d_prefix_sum_i64, p.npages_i64, base_row + nrows - 1);
            s_n_i64_pgs = hi - s_i64_pg_lo + 1;
            if (s_n_i64_pgs > 3) s_n_i64_pgs = 3;
            s_i64_pg_lo_start = (s_i64_pg_lo == 0) ? 0 : p.d_prefix_sum_i64[s_i64_pg_lo - 1];
            s_i64_pg_lo_nrows = p.d_prefix_sum_i64[s_i64_pg_lo] - s_i64_pg_lo_start;
        }
        __syncthreads();
        uint32_t i64_pg_lo = s_i64_pg_lo;
        uint32_t n_i64_pgs = s_n_i64_pgs;
        uint64_t i64_lo_start = s_i64_pg_lo_start;
        uint64_t i64_lo_nrows = s_i64_pg_lo_nrows;

        // Batch-submit NVMe reads: 2 INT32 + up to 6 INT64
        uint16_t cids[8];
        QueuePair* qps_arr[8];
        if (tid == 0) {
            for (int fi = 0; fi < (int)N_I32; fi++) {
                uint64_t slot = (uint64_t)bid * SLOTS + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba; uint32_t nblk;
                if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[fi][pg]);
                } else {
                    lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot, NVM_IO_READ, &cids[fi], &sq_pos);
            }
            uint32_t ri = N_I32;
            for (int fi = 0; fi < (int)N_I64; fi++) {
                for (uint32_t ipg = 0; ipg < n_i64_pgs; ipg++) {
                    uint64_t i64_pg = i64_pg_lo + ipg;
                    uint64_t slot = (uint64_t)bid * SLOTS + N_I32 + fi * 3 + ipg;
                    uint64_t global_pg = p.field_start_page_ids_i64[fi] + i64_pg;
                    uint32_t dev = global_pg % ndev;
                    qps_arr[ri] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                    uint64_t lba; uint32_t nblk;
                    if (p.comp_methods_i64[fi] != 0 && p.d_comp_offsets_i64[fi]) {
                        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i64[fi][i64_pg] / 512;
                        nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes_i64[fi][i64_pg]);
                    } else {
                        lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                        nblk = p.blocks_per_page;
                    }
                    uint16_t sq_pos = 0;
                    access_data_async(pc, qps_arr[ri], lba, nblk, slot, NVM_IO_READ, &cids[ri], &sq_pos);
                    ri++;
                }
            }
            uint32_t total_reads = N_I32 + N_I64 * n_i64_pgs;
            for (uint32_t r = 0; r < total_reads; r++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[r]->cq, cids[r], &pl, &ph);
                cq_dequeue(&qps_arr[r]->cq, cq_pos, &qps_arr[r]->sq);
                put_cid(&qps_arr[r]->sq, cids[r]);
            }
        }
        __syncthreads();

        // Decompress 2 INT32 fields to per-block scratch
        int32_t* field_data[N_I32];
        for (int fi = 0; fi < (int)N_I32; fi++) {
            uint64_t slot = (uint64_t)bid * SLOTS + fi;
            char* pp = (char*)pc->base_addr + slot * p.page_size;
            field_data[fi] = p.d_scratch + ((uint64_t)bid * N_I32 + fi) * p.scratch_stride;
            uint32_t nalloc_dummy;
            decompress_page_128(pp, field_data[fi], p.comp_methods[fi],
                               shared_decomp.buf32, tid, &nalloc_dummy);
            __syncthreads();
        }

        // Decompress INT64 fields
        int64_t* i64_scratch[N_I64][3];
        for (int fi = 0; fi < (int)N_I64; fi++) {
            for (uint32_t ipg = 0; ipg < n_i64_pgs; ipg++) {
                uint64_t slot = (uint64_t)bid * SLOTS + N_I32 + fi * 3 + ipg;
                char* pp = (char*)pc->base_addr + slot * p.page_size;
                i64_scratch[fi][ipg] = p.d_scratch_i64
                    + ((uint64_t)bid * N_I64 * 3 + fi * 3 + ipg) * p.scratch_stride_i64;
                uint32_t nalloc_dummy;
                decompress_page64(pp, i64_scratch[fi][ipg], p.comp_methods_i64[fi],
                                 shared_decomp.buf64, tid, &nalloc_dummy);
                __syncthreads();
            }
        }

        // Probe ORDERS HT + SUPPLIER HT + revenue accumulation
        for (uint64_t i = tid; i < nrows; i += blockDim.x) {
            // Map to INT64 page using prefix sum
            uint64_t global_row = base_row + i;
            uint32_t wpg = 0;
            uint64_t off = global_row - i64_lo_start;
            if (off >= i64_lo_nrows) {
                wpg = 1;
                off = global_row - p.d_prefix_sum_i64[i64_pg_lo];
                if (n_i64_pgs >= 3) {
                    uint64_t p1_nrows = p.d_prefix_sum_i64[i64_pg_lo + 1] - p.d_prefix_sum_i64[i64_pg_lo];
                    if (off >= p1_nrows) {
                        wpg = 2;
                        off = global_row - p.d_prefix_sum_i64[i64_pg_lo + 1];
                    }
                }
            }

            uint64_t orderkey = (uint64_t)i64_scratch[0][wpg][off];
            int32_t cust_nation_idx = q5f_ht_probe(
                p.d_ht_ord_keys, p.d_ht_ord_values, p.ht_ord_mask, orderkey);
            if (cust_nation_idx < 0) continue;

            uint64_t suppkey = (uint64_t)i64_scratch[1][wpg][off];
            int32_t supp_nation_idx = q5f_ht_probe(
                p.d_ht_supp_keys, p.d_ht_supp_values, p.ht_supp_mask, suppkey);
            if (supp_nation_idx < 0) continue;

            if (cust_nation_idx != supp_nation_idx) continue;

            int32_t extprice = field_data[0][i];
            int32_t discount = field_data[1][i];
            int64_t contribution = (int64_t)extprice * (int64_t)(100 - discount);
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_revenue[cust_nation_idx]),
                      (unsigned long long)contribution);
        }
    }
}

void bam_q5_lineitem_fused_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMQ5LineitemFusedParams& params,
    int64_t* d_revenue,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    bam_q5_lineitem_fused_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr, params, d_revenue);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Q3 hash table primitives (reuse Q5 hash function)
// ============================================================
static constexpr uint64_t Q3F_HT_EMPTY = UINT64_MAX;

__device__ __forceinline__ uint32_t q3f_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

// Hash set probe (keys only, returns true if found)
__device__ __forceinline__ bool q3f_hashset_probe(
    const uint64_t *keys, uint32_t mask, uint64_t key)
{
    uint32_t slot = q3f_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return true;
        if (k == Q3F_HT_EMPTY) return false;
        slot = (slot + 1) & mask;
    }
}

// Hash table insert (key + uint64_t payload)
__device__ __forceinline__ void q3f_ht_insert_kv(
    uint64_t *keys, uint64_t *payloads, uint32_t mask,
    uint64_t key, uint64_t payload)
{
    uint32_t slot = q3f_hash64(key) & mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)Q3F_HT_EMPTY,
            (unsigned long long)key);
        if (prev == Q3F_HT_EMPTY || prev == key) {
            payloads[slot] = payload;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

// Hash table probe (returns payload, or Q3F_HT_EMPTY if not found)
__device__ __forceinline__ uint64_t q3f_ht_probe_kv(
    const uint64_t *keys, const uint64_t *payloads, uint32_t mask,
    uint64_t key)
{
    uint32_t slot = q3f_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return payloads[slot];
        if (k == Q3F_HT_EMPTY) return Q3F_HT_EMPTY;
        slot = (slot + 1) & mask;
    }
}

// ============================================================
// Q3 ORDERS fused kernel: batch I/O + decompress + HT build
// ============================================================
__global__ void bam_q3_orders_fused_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMQ3OrdersFusedParams p)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ union { uint buf32[549]; ulong buf64[581]; } shared_decomp;
    __shared__ uint32_t s_i64_pg_lo, s_n_i64_pgs;
    __shared__ uint64_t s_i64_pg_lo_start, s_i64_pg_lo_nrows;
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    constexpr uint32_t N_I32 = 2;
    constexpr uint32_t N_I64 = 2;
    constexpr uint32_t SLOTS = N_I32 + N_I64 * 3;  // 8

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;

        uint64_t nrows = (pg == 0) ? p.d_prefix_sum[0]
                                   : p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1];
        uint64_t base_row = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

        // Thread 0: find INT64 page range
        if (tid == 0) {
            s_i64_pg_lo = dpf_find_page(p.d_prefix_sum_i64, p.npages_i64, base_row);
            uint32_t hi = dpf_find_page(p.d_prefix_sum_i64, p.npages_i64, base_row + nrows - 1);
            s_n_i64_pgs = hi - s_i64_pg_lo + 1;
            if (s_n_i64_pgs > 3) s_n_i64_pgs = 3;
            s_i64_pg_lo_start = (s_i64_pg_lo == 0) ? 0 : p.d_prefix_sum_i64[s_i64_pg_lo - 1];
            s_i64_pg_lo_nrows = p.d_prefix_sum_i64[s_i64_pg_lo] - s_i64_pg_lo_start;
        }
        __syncthreads();
        uint32_t i64_pg_lo = s_i64_pg_lo;
        uint32_t n_i64_pgs = s_n_i64_pgs;
        uint64_t i64_lo_start = s_i64_pg_lo_start;
        uint64_t i64_lo_nrows = s_i64_pg_lo_nrows;

        // Batch-submit NVMe reads: 2 INT32 + up to 6 INT64
        uint16_t cids[8];
        QueuePair* qps_arr[8];
        if (tid == 0) {
            for (int fi = 0; fi < (int)N_I32; fi++) {
                uint64_t slot = (uint64_t)bid * SLOTS + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba; uint32_t nblk;
                if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[fi][pg]);
                } else {
                    lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot, NVM_IO_READ, &cids[fi], &sq_pos);
            }
            uint32_t ri = N_I32;
            for (int fi = 0; fi < (int)N_I64; fi++) {
                for (uint32_t ipg = 0; ipg < n_i64_pgs; ipg++) {
                    uint64_t i64_pg = i64_pg_lo + ipg;
                    uint64_t slot = (uint64_t)bid * SLOTS + N_I32 + fi * 3 + ipg;
                    uint64_t global_pg = p.field_start_page_ids_i64[fi] + i64_pg;
                    uint32_t dev = global_pg % ndev;
                    qps_arr[ri] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                    uint64_t lba; uint32_t nblk;
                    if (p.comp_methods_i64[fi] != 0 && p.d_comp_offsets_i64[fi]) {
                        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i64[fi][i64_pg] / 512;
                        nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes_i64[fi][i64_pg]);
                    } else {
                        lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                        nblk = p.blocks_per_page;
                    }
                    uint16_t sq_pos = 0;
                    access_data_async(pc, qps_arr[ri], lba, nblk, slot, NVM_IO_READ, &cids[ri], &sq_pos);
                    ri++;
                }
            }
            uint32_t total_reads = N_I32 + N_I64 * n_i64_pgs;
            for (uint32_t r = 0; r < total_reads; r++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[r]->cq, cids[r], &pl, &ph);
                cq_dequeue(&qps_arr[r]->cq, cq_pos, &qps_arr[r]->sq);
                put_cid(&qps_arr[r]->sq, cids[r]);
            }
        }
        __syncthreads();

        // Decompress 2 INT32 fields
        int32_t* field_data[N_I32];
        for (int fi = 0; fi < (int)N_I32; fi++) {
            uint64_t slot = (uint64_t)bid * SLOTS + fi;
            char* pp = (char*)pc->base_addr + slot * p.page_size;
            field_data[fi] = p.d_scratch + ((uint64_t)bid * N_I32 + fi) * p.scratch_stride;
            uint32_t nalloc_dummy;
            decompress_page_128(pp, field_data[fi], p.comp_methods[fi],
                               shared_decomp.buf32, tid, &nalloc_dummy);
            __syncthreads();
        }

        // Decompress INT64 fields
        int64_t* i64_scratch[N_I64][3];
        for (int fi = 0; fi < (int)N_I64; fi++) {
            for (uint32_t ipg = 0; ipg < n_i64_pgs; ipg++) {
                uint64_t slot = (uint64_t)bid * SLOTS + N_I32 + fi * 3 + ipg;
                char* pp = (char*)pc->base_addr + slot * p.page_size;
                i64_scratch[fi][ipg] = p.d_scratch_i64
                    + ((uint64_t)bid * N_I64 * 3 + fi * 3 + ipg) * p.scratch_stride_i64;
                uint32_t nalloc_dummy;
                decompress_page64(pp, i64_scratch[fi][ipg], p.comp_methods_i64[fi],
                                 shared_decomp.buf64, tid, &nalloc_dummy);
                __syncthreads();
            }
        }

        // Date filter + CUSTOMER set probe + ORDERS HT insert
        for (uint64_t i = tid; i < nrows; i += blockDim.x) {
            int32_t odate = field_data[0][i];
            if (!p.skip_date_filter && odate >= 19950315) continue;

            // Map to INT64 page using prefix sum
            uint64_t global_row = base_row + i;
            uint32_t wpg = 0;
            uint64_t off = global_row - i64_lo_start;
            if (off >= i64_lo_nrows) {
                wpg = 1;
                off = global_row - p.d_prefix_sum_i64[i64_pg_lo];
                if (n_i64_pgs >= 3) {
                    uint64_t p1_nrows = p.d_prefix_sum_i64[i64_pg_lo + 1] - p.d_prefix_sum_i64[i64_pg_lo];
                    if (off >= p1_nrows) {
                        wpg = 2;
                        off = global_row - p.d_prefix_sum_i64[i64_pg_lo + 1];
                    }
                }
            }

            uint64_t custkey = (uint64_t)i64_scratch[1][wpg][off];
            if (!q3f_hashset_probe(p.d_custkey_set, p.custkey_set_mask, custkey))
                continue;

            uint64_t orderkey = (uint64_t)i64_scratch[0][wpg][off];
            int32_t shippri = field_data[1][i];
            uint64_t payload = ((uint64_t)(uint32_t)odate << 32)
                             | (uint64_t)(uint32_t)shippri;
            q3f_ht_insert_kv(p.d_orders_ht_keys, p.d_orders_ht_payloads,
                             p.orders_ht_mask, orderkey, payload);
        }
    }
}

void bam_q3_orders_fused_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMQ3OrdersFusedParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    bam_q3_orders_fused_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr, params);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Q3 LINEITEM fused kernel: batch I/O + decompress + HT probe + aggregate
// ============================================================
__global__ void bam_q3_lineitem_fused_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMQ3LineitemFusedParams p)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ union { uint buf32[549]; ulong buf64[581]; } shared_decomp;
    __shared__ uint32_t s_i64_pg_lo, s_n_i64_pgs;
    __shared__ uint64_t s_i64_pg_lo_start, s_i64_pg_lo_nrows;
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    constexpr uint32_t N_I32 = 3;   // L_SHIPDATE, L_EXTPRICE, L_DISCOUNT
    constexpr uint32_t N_I64 = 1;   // L_ORDERKEY
    constexpr uint32_t SLOTS = N_I32 + N_I64 * 3;  // 6

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;

        uint64_t nrows = (pg == 0) ? p.d_prefix_sum[0]
                                   : p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1];
        uint64_t base_row = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

        // Thread 0: find INT64 page range
        if (tid == 0) {
            s_i64_pg_lo = dpf_find_page(p.d_prefix_sum_i64, p.npages_i64, base_row);
            uint32_t hi = dpf_find_page(p.d_prefix_sum_i64, p.npages_i64, base_row + nrows - 1);
            s_n_i64_pgs = hi - s_i64_pg_lo + 1;
            if (s_n_i64_pgs > 3) s_n_i64_pgs = 3;
            s_i64_pg_lo_start = (s_i64_pg_lo == 0) ? 0 : p.d_prefix_sum_i64[s_i64_pg_lo - 1];
            s_i64_pg_lo_nrows = p.d_prefix_sum_i64[s_i64_pg_lo] - s_i64_pg_lo_start;
        }
        __syncthreads();
        uint32_t i64_pg_lo = s_i64_pg_lo;
        uint32_t n_i64_pgs = s_n_i64_pgs;
        uint64_t i64_lo_start = s_i64_pg_lo_start;
        uint64_t i64_lo_nrows = s_i64_pg_lo_nrows;

        // Batch-submit NVMe reads: 3 INT32 + up to 3 INT64
        uint16_t cids[6];
        QueuePair* qps_arr[6];
        if (tid == 0) {
            for (int fi = 0; fi < (int)N_I32; fi++) {
                uint64_t slot = (uint64_t)bid * SLOTS + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba; uint32_t nblk;
                if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[fi][pg]);
                } else {
                    lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot, NVM_IO_READ, &cids[fi], &sq_pos);
            }
            uint32_t ri = N_I32;
            for (int fi = 0; fi < (int)N_I64; fi++) {
                for (uint32_t ipg = 0; ipg < n_i64_pgs; ipg++) {
                    uint64_t i64_pg = i64_pg_lo + ipg;
                    uint64_t slot = (uint64_t)bid * SLOTS + N_I32 + fi * 3 + ipg;
                    uint64_t global_pg = p.field_start_page_ids_i64[fi] + i64_pg;
                    uint32_t dev = global_pg % ndev;
                    qps_arr[ri] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                    uint64_t lba; uint32_t nblk;
                    if (p.comp_methods_i64[fi] != 0 && p.d_comp_offsets_i64[fi]) {
                        lba = p.partition_start_lbas[dev] + p.d_comp_offsets_i64[fi][i64_pg] / 512;
                        nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes_i64[fi][i64_pg]);
                    } else {
                        lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                        nblk = p.blocks_per_page;
                    }
                    uint16_t sq_pos = 0;
                    access_data_async(pc, qps_arr[ri], lba, nblk, slot, NVM_IO_READ, &cids[ri], &sq_pos);
                    ri++;
                }
            }
            uint32_t total_reads = N_I32 + N_I64 * n_i64_pgs;
            for (uint32_t r = 0; r < total_reads; r++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[r]->cq, cids[r], &pl, &ph);
                cq_dequeue(&qps_arr[r]->cq, cq_pos, &qps_arr[r]->sq);
                put_cid(&qps_arr[r]->sq, cids[r]);
            }
        }
        __syncthreads();

        // Decompress 3 INT32 fields
        int32_t* field_data[N_I32];
        for (int fi = 0; fi < (int)N_I32; fi++) {
            uint64_t slot = (uint64_t)bid * SLOTS + fi;
            char* pp = (char*)pc->base_addr + slot * p.page_size;
            field_data[fi] = p.d_scratch + ((uint64_t)bid * N_I32 + fi) * p.scratch_stride;
            uint32_t nalloc_dummy;
            decompress_page_128(pp, field_data[fi], p.comp_methods[fi],
                               shared_decomp.buf32, tid, &nalloc_dummy);
            __syncthreads();
        }

        // Decompress INT64 field (L_ORDERKEY)
        int64_t* i64_scratch[N_I64][3];
        for (int fi = 0; fi < (int)N_I64; fi++) {
            for (uint32_t ipg = 0; ipg < n_i64_pgs; ipg++) {
                uint64_t slot = (uint64_t)bid * SLOTS + N_I32 + fi * 3 + ipg;
                char* pp = (char*)pc->base_addr + slot * p.page_size;
                i64_scratch[fi][ipg] = p.d_scratch_i64
                    + ((uint64_t)bid * N_I64 * 3 + fi * 3 + ipg) * p.scratch_stride_i64;
                uint32_t nalloc_dummy;
                decompress_page64(pp, i64_scratch[fi][ipg], p.comp_methods_i64[fi],
                                 shared_decomp.buf64, tid, &nalloc_dummy);
                __syncthreads();
            }
        }

        // Shipdate filter + ORDERS HT probe + aggregation
        for (uint64_t i = tid; i < nrows; i += blockDim.x) {
            int32_t shipdate = field_data[0][i];
            if (!p.skip_shipdate_filter && shipdate <= 19950315) continue;

            // Map to INT64 page using prefix sum
            uint64_t global_row = base_row + i;
            uint32_t wpg = 0;
            uint64_t off = global_row - i64_lo_start;
            if (off >= i64_lo_nrows) {
                wpg = 1;
                off = global_row - p.d_prefix_sum_i64[i64_pg_lo];
                if (n_i64_pgs >= 3) {
                    uint64_t p1_nrows = p.d_prefix_sum_i64[i64_pg_lo + 1] - p.d_prefix_sum_i64[i64_pg_lo];
                    if (off >= p1_nrows) {
                        wpg = 2;
                        off = global_row - p.d_prefix_sum_i64[i64_pg_lo + 1];
                    }
                }
            }

            uint64_t orderkey = (uint64_t)i64_scratch[0][wpg][off];
            uint64_t ord_payload = q3f_ht_probe_kv(
                p.d_orders_ht_keys, p.d_orders_ht_payloads,
                p.orders_ht_mask, orderkey);
            if (ord_payload == Q3F_HT_EMPTY) continue;

            int32_t extprice = field_data[1][i];
            int32_t discount = field_data[2][i];
            int64_t revenue = (int64_t)extprice * (int64_t)(100 - discount);

            uint32_t aggr_slot = q3f_hash64(orderkey) & p.aggr_mask;
            while (true) {
                uint64_t prev = atomicCAS(
                    reinterpret_cast<unsigned long long *>(&p.d_aggr_keys[aggr_slot]),
                    (unsigned long long)Q3F_HT_EMPTY,
                    (unsigned long long)orderkey);
                if (prev == Q3F_HT_EMPTY || prev == orderkey) {
                    atomicAdd(
                        reinterpret_cast<unsigned long long *>(&p.d_aggr_revenues[aggr_slot]),
                        (unsigned long long)revenue);
                    break;
                }
                aggr_slot = (aggr_slot + 1) & p.aggr_mask;
            }
        }
    }
}

void bam_q3_lineitem_fused_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMQ3LineitemFusedParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    bam_q3_lineitem_fused_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr, params);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Q1 fused kernel: 5 INT32 fields per block iteration.
// Pre-flattened CHAR(1) arrays for returnflag/linestatus.
// ============================================================

__global__ void bam_q1_fused_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMQ1FusedParams p)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Per-thread local accumulators (register-allocated).
    // 6 groups × 6 aggs = 36 int64_t.  SUM_CHARGE_HI handled at final flush.
    constexpr int LA = 6;  // QTY, BASE_PRICE, DISC_PRICE, CHARGE, DISCOUNT, COUNT
    int64_t local_agg[Q1_NUM_GROUPS * LA];
    for (int k = 0; k < Q1_NUM_GROUPS * LA; k++) local_agg[k] = 0;

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;

        uint64_t nrows = (pg == 0) ? p.d_prefix_sum[0]
                                   : p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1];
        uint64_t base_row = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];

        // Batch-submit 5 NVMe reads
        uint16_t cids[5];
        QueuePair* qps_arr[5];
        if (tid == 0) {
            for (int fi = 0; fi < 5; fi++) {
                uint64_t slot = (uint64_t)bid * 5 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba;
                uint32_t nblk;
                if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[fi][pg]);
                } else {
                    uint64_t local_pg = global_pg / ndev;
                    lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            for (int fi = 0; fi < 5; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
        }
        __syncthreads();

        // Decompress 5 INT32 fields to per-block scratch
        int32_t* field_data[5];
        for (int fi = 0; fi < 5; fi++) {
            uint64_t slot = (uint64_t)bid * 5 + fi;
            char* page_ptr = (char*)pc->base_addr + slot * p.page_size;
            field_data[fi] = p.d_scratch
                + ((uint64_t)bid * 5 + fi) * p.scratch_stride;
            uint32_t nalloc_dummy;
            decompress_page_128(page_ptr, field_data[fi], p.comp_methods[fi],
                               shared_buf32, tid, &nalloc_dummy);
            __syncthreads();
        }

        // Evaluate: shipdate filter → group → local accumulate
        for (uint64_t i = tid; i < nrows; i += blockDim.x) {
            int32_t shipdate = field_data[0][i];
            if (shipdate > 19980902) continue;

            // CHAR(1) from pre-flattened arrays (global rowid)
            char returnflag = (char)(uint8_t)p.d_l_rf_flat[base_row + i];
            char linestatus = (char)(uint8_t)p.d_l_ls_flat[base_row + i];

            int row;
            switch (returnflag) {
                case 'A': row = 0; break;
                case 'N': row = 1; break;
                case 'R': row = 2; break;
                default: continue;
            }
            int col = (linestatus == 'F') ? 0 : 1;
            int gid = row * 2 + col;

            int32_t quantity      = field_data[1][i];
            int32_t extendedprice = field_data[2][i];
            int32_t discount      = field_data[3][i];
            int32_t tax           = field_data[4][i];

            int64_t disc_price = (int64_t)extendedprice * (int64_t)(100 - discount);
            int64_t charge = disc_price * (int64_t)(100 + tax);

            int64_t* la = local_agg + gid * LA;
            la[0] += quantity;
            la[1] += extendedprice;
            la[2] += disc_price;
            la[3] += charge;
            la[4] += discount;
            la[5] += 1;
        }
    }

    // Flush per-thread local accumulators → global aggregates
    for (int g = 0; g < Q1_NUM_GROUPS; g++) {
        int64_t* la = local_agg + g * LA;
        if (la[5] == 0) continue;
        int64_t* ga = p.d_agg + g * Q1_NUM_AGGS;
        atomicAdd((unsigned long long*)&ga[Q1_SUM_QTY],
                  (unsigned long long)la[0]);
        atomicAdd((unsigned long long*)&ga[Q1_SUM_BASE_PRICE],
                  (unsigned long long)la[1]);
        atomicAdd((unsigned long long*)&ga[Q1_SUM_DISC_PRICE],
                  (unsigned long long)la[2]);
        {
            unsigned long long old_lo = atomicAdd(
                (unsigned long long*)&ga[Q1_SUM_CHARGE],
                (unsigned long long)la[3]);
            if (old_lo + (unsigned long long)la[3] < old_lo) {
                atomicAdd((unsigned long long*)&ga[Q1_SUM_CHARGE_HI], 1ULL);
            }
        }
        atomicAdd((unsigned long long*)&ga[Q1_SUM_DISCOUNT],
                  (unsigned long long)la[4]);
        atomicAdd((unsigned long long*)&ga[Q1_COUNT],
                  (unsigned long long)la[5]);
    }
}

void bam_q1_fused_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const BAMQ1FusedParams& params,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    bam_q1_fused_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr, params);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Per-thread LZ4 decompression test
//
// Uses lz4_decompress_per_thread (single-thread, no nvCOMPdx).
// Uploads a host buffer, decompresses on GPU (1 thread),
// downloads the result.
// ============================================================

#include "lz4_decomp.cuh"

__global__ void bam_test_lz4_decompress_kernel(
    const uint8_t* input, uint32_t input_size,
    uint8_t* output, uint32_t max_output,
    uint32_t* out_decomp_size)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t size = lz4_decompress_per_thread(input, input_size,
                                                   output, max_output);
        *out_decomp_size = size;
    }
}

uint32_t bam_test_lz4_decompress_page(const void* comp_buf, size_t comp_size,
                                       void* decomp_out, uint32_t max_output) {
    // Upload compressed data to GPU
    uint8_t* d_input;
    BAM_CUDA_CHECK(cudaMalloc(&d_input, comp_size));
    BAM_CUDA_CHECK(cudaMemcpy(d_input, comp_buf, comp_size,
                               cudaMemcpyHostToDevice));

    // Allocate output buffer on GPU
    uint8_t* d_output;
    BAM_CUDA_CHECK(cudaMalloc(&d_output, max_output));
    BAM_CUDA_CHECK(cudaMemset(d_output, 0, max_output));

    // Allocate decomp size output
    uint32_t* d_size;
    BAM_CUDA_CHECK(cudaMalloc(&d_size, sizeof(uint32_t)));

    // Launch: 1 block of 1 thread
    bam_test_lz4_decompress_kernel<<<1, 1>>>(d_input, (uint32_t)comp_size,
                                              d_output, max_output, d_size);
    BAM_CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    uint32_t h_size = 0;
    BAM_CUDA_CHECK(cudaMemcpy(&h_size, d_size, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

    uint32_t copy_bytes = (h_size < max_output) ? h_size : max_output;
    BAM_CUDA_CHECK(cudaMemcpy(decomp_out, d_output, copy_bytes,
                               cudaMemcpyDeviceToHost));

    BAM_CUDA_CHECK(cudaFree(d_input));
    BAM_CUDA_CHECK(cudaFree(d_output));
    BAM_CUDA_CHECK(cudaFree(d_size));

    return h_size;
}

// ============================================================
// scan_o_comment v4: single-kernel IO + per-thread LZ4 + VCHAR scan
//
// 128 threads (4 warps) per block.
// Each warp: lane 0 reads 32 pages via BAM, then all 32 lanes
// each decompress 1 page with lz4_decompress_per_thread, then scan.
// Grid-stride loop processes all pages.
// ============================================================

__global__ void bam_scan_o_comment_v4_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        partition_start_lba,
    uint64_t        npages_total)
{
    const uint32_t tid      = threadIdx.x;      // 0..127
    const uint32_t warp_id  = tid / 32;         // 0..3
    const uint32_t lane     = tid % 32;
    const uint32_t bid      = blockIdx.x;

    const uint32_t global_warp  = bid * 4 + warp_id;
    const uint32_t total_warps  = gridDim.x * 4;
    const uint32_t global_tid   = bid * 128 + tid;

    char* base = (char*)pc->base_addr;
    char* my_decomp = d_decomp_buf + (uint64_t)global_tid * page_size;

    uint64_t my_records  = 0;
    uint64_t my_strlen   = 0;
    uint64_t my_byte_sum = 0;

    for (uint64_t warp_base = (uint64_t)global_warp * 32;
         warp_base < npages_total;
         warp_base += (uint64_t)total_warps * 32)
    {
        const uint64_t my_page = warp_base + lane;
        const bool valid = (my_page < npages_total);

        // ── Phase 1: IO (lane 0 reads up to 32 pages) ──
        if (lane == 0) {
            uint32_t n = 32;
            if (warp_base + 32 > npages_total)
                n = (uint32_t)(npages_total - warp_base);

            for (uint32_t k = 0; k < n; k++) {
                uint64_t pg = warp_base + k;
                uint32_t comp_size = d_comp_sizes[pg];
                uint64_t lba = partition_start_lba + d_comp_offsets[pg] / 512;
                uint32_t nblk = safe_io_nblocks_vchar_io(comp_size);

                unsigned long long pc_entry =
                    (unsigned long long)global_warp * 32 + k;
                QueuePair* qp = ctrls[0]->d_qps
                    + (pc_entry % ctrls[0]->n_qps);

                uint16_t cid = 0;
                uint16_t sq_pos = 0;
                access_data_async(pc, qp, lba, nblk, pc_entry,
                                  NVM_IO_READ, &cid, &sq_pos);
                uint32_t poll_loc, poll_head;
                uint32_t cq_pos = cq_poll(&qp->cq, cid,
                                           &poll_loc, &poll_head);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq);
                put_cid(&qp->sq, cid);
            }
        }
        __syncwarp();

        // ── Phase 2: Per-thread LZ4 decompress ──
        if (valid) {
            uint32_t comp_size = d_comp_sizes[my_page];
            const uint8_t* comp_src = (const uint8_t*)(
                base + (unsigned long long)(global_warp * 32 + lane) * page_size);

            lz4_decompress_per_thread(
                comp_src, comp_size,
                (uint8_t*)my_decomp, page_size);
        }

        // ── Phase 3: VCHAR scan ──
        if (valid) {
            uint32_t nalloc = *(uint32_t*)my_decomp;

            for (uint32_t slot = 0; slot < nalloc; slot++) {
                uint32_t oslt = *(uint32_t*)(
                    my_decomp + page_size
                    - sizeof(uint32_t) * (slot + 1));
                uint16_t vlen = *(uint16_t*)(my_decomp + oslt);
                const uint8_t* vdata = (const uint8_t*)(
                    my_decomp + oslt + sizeof(uint32_t));

                my_records++;
                my_strlen += vlen;
                for (uint16_t b = 0; b < vlen; b++) {
                    my_byte_sum += vdata[b];
                }
            }
        }
    }

    // ── Global accumulate ──
    atomicAdd((unsigned long long*)d_total_records, (unsigned long long)my_records);
    atomicAdd((unsigned long long*)d_total_strlen, (unsigned long long)my_strlen);
    atomicAdd((unsigned long long*)d_total_byte_sum, (unsigned long long)my_byte_sum);
}

void bam_scan_o_comment_v4_run(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint64_t npages,
    uint64_t partition_start_lba,
    const uint32_t* h_comp_sizes,
    const uint64_t* h_comp_offsets,
    uint32_t num_blocks,
    uint64_t* out_records,
    uint64_t* out_strlen,
    uint64_t* out_byte_sum)
{
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);

    const uint32_t threads_per_block = 128;
    const uint64_t total_threads = (uint64_t)num_blocks * threads_per_block;

    // Page cache: one entry per thread (32 pages/warp × 4 warps × num_blocks)
    const uint64_t n_pc_pages = total_threads;
    const uint64_t max_range = 64;

    page_cache_t h_pc(page_size, n_pc_pages, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);

    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    // Decompress buffer: one page per thread
    char* d_decomp = nullptr;
    size_t decomp_size = total_threads * (size_t)page_size;
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp, decomp_size));

    // Upload compressed metadata
    uint32_t* d_comp_sizes = nullptr;
    uint64_t* d_comp_offsets = nullptr;
    BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes, npages * sizeof(uint32_t)));
    BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes, h_comp_sizes,
                               npages * sizeof(uint32_t), cudaMemcpyHostToDevice));
    BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets, npages * sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets, h_comp_offsets,
                               npages * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Output accumulators
    uint64_t* d_total_records = nullptr;
    uint64_t* d_total_strlen = nullptr;
    uint64_t* d_total_byte_sum = nullptr;
    BAM_CUDA_CHECK(cudaMalloc(&d_total_records, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMalloc(&d_total_strlen, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMalloc(&d_total_byte_sum, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_total_records, 0, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_total_strlen, 0, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_total_byte_sum, 0, sizeof(uint64_t)));

    // Launch kernel
    bam_scan_o_comment_v4_kernel<<<num_blocks, threads_per_block>>>(
        h_pc.pdt.d_ctrls, h_pc.d_pc_ptr,
        d_comp_sizes, d_comp_offsets, d_decomp,
        d_total_records, d_total_strlen, d_total_byte_sum,
        page_size, partition_start_lba, npages);

    BAM_CUDA_CHECK(cudaDeviceSynchronize());

    // Read results
    BAM_CUDA_CHECK(cudaMemcpy(out_records, d_total_records,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost));
    BAM_CUDA_CHECK(cudaMemcpy(out_strlen, d_total_strlen,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost));
    BAM_CUDA_CHECK(cudaMemcpy(out_byte_sum, d_total_byte_sum,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Cleanup
    BAM_CUDA_CHECK(cudaFree(d_total_records));
    BAM_CUDA_CHECK(cudaFree(d_total_strlen));
    BAM_CUDA_CHECK(cudaFree(d_total_byte_sum));
    BAM_CUDA_CHECK(cudaFree(d_decomp));
    BAM_CUDA_CHECK(cudaFree(d_comp_sizes));
    BAM_CUDA_CHECK(cudaFree(d_comp_offsets));
}

// ============================================================
// scan_o_comment v5: v4 + coalesced IO (warp-independent)
//
// Same warp-independent structure as v4 — NO __syncthreads().
// Each warp independently processes 32 pages per iteration:
//   1. Lane 0: coalesced IO (groups ~3 pages per NVMe command)
//      Batch submit all groups, then batch poll.
//   2. __syncwarp()
//   3. All 32 lanes: decompress + scan (same as v4)
//
// Key improvement over v4:
//   NVMe commands reduced ~3x per warp (11 coalesced vs 32 individual).
//   Total: ~2,376 commands/iter vs ~6,912 in v4.
//
// Memory (54 blocks, 216 warps, coalesce_limit=2MiB):
//   Page cache:  216 × 12 × 2 MiB  ≈ 5.06 GB
//   Decompress:  216 × 32 × 1 MiB  ≈ 6.75 GB
//   Total:       ≈ 11.8 GB
// ============================================================

#define V5_GROUPS_PER_WARP 12

__global__ void bam_scan_o_comment_v5_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    char*           d_decomp_buf,
    uint64_t*       d_total_records,
    uint64_t*       d_total_strlen,
    uint64_t*       d_total_byte_sum,
    uint32_t        page_size,
    uint64_t        partition_start_lba,
    uint64_t        npages_total,
    uint32_t        coalesce_limit,
    uint32_t        max_groups_per_warp)
{
    const uint32_t tid      = threadIdx.x;      // 0..127
    const uint32_t warp_id  = tid / 32;         // 0..3
    const uint32_t lane     = tid % 32;
    const uint32_t bid      = blockIdx.x;

    const uint32_t global_warp  = bid * 4 + warp_id;
    const uint32_t total_warps  = gridDim.x * 4;

    char* base = (char*)pc->base_addr;
    char* my_decomp = d_decomp_buf + (uint64_t)global_warp * 32 * page_size
                    + (uint64_t)lane * page_size;

    // Shared memory: per-warp page mapping + IO tracking
    __shared__ uint32_t  s_page_pc_entry[4][32];
    __shared__ uint32_t  s_page_offset[4][32];
    __shared__ uint16_t  s_cids[4][V5_GROUPS_PER_WARP];
    __shared__ uintptr_t s_io_qps[4][V5_GROUPS_PER_WARP];

    uint64_t my_records  = 0;
    uint64_t my_strlen   = 0;
    uint64_t my_byte_sum = 0;

    for (uint64_t warp_base = (uint64_t)global_warp * 32;
         warp_base < npages_total;
         warp_base += (uint64_t)total_warps * 32)
    {
        uint32_t n = 32;
        if (warp_base + 32 > npages_total)
            n = (uint32_t)(npages_total - warp_base);

        // ── Coalesced IO (lane 0 only) ──
        if (lane == 0) {
            uint64_t entry_base = (uint64_t)global_warp * max_groups_per_warp;
            uint32_t n_groups = 0;
            uint32_t pg = 0;

            // Batch submit: group consecutive pages, submit coalesced reads
            while (pg < n && n_groups < max_groups_per_warp) {
                uint64_t group_disk_start = d_comp_offsets[warp_base + pg];
                uint32_t group_first = pg;
                uint32_t group_end = pg + 1;

                while (group_end < n) {
                    uint64_t end_byte =
                        d_comp_offsets[warp_base + group_end]
                        + d_comp_sizes[warp_base + group_end];
                    if (end_byte - group_disk_start > coalesce_limit)
                        break;
                    group_end++;
                }

                uint64_t pc_entry_id = entry_base + n_groups;

                // Record per-page mapping
                for (uint32_t p = group_first; p < group_end; p++) {
                    s_page_pc_entry[warp_id][p] = (uint32_t)pc_entry_id;
                    s_page_offset[warp_id][p] = (uint32_t)(
                        d_comp_offsets[warp_base + p] - group_disk_start);
                }

                // Submit coalesced NVMe read
                uint64_t last_end =
                    d_comp_offsets[warp_base + group_end - 1]
                    + d_comp_sizes[warp_base + group_end - 1];
                uint64_t total_bytes = last_end - group_disk_start;
                uint64_t lba = partition_start_lba + group_disk_start / 512;
                uint32_t nblk = (uint32_t)((total_bytes + 511) / 512);

                QueuePair* qp = ctrls[0]->d_qps
                    + (pc_entry_id % ctrls[0]->n_qps);
                uint16_t cid = 0;
                uint16_t sq_pos = 0;
                access_data_async(pc, qp, lba, nblk, pc_entry_id,
                                  NVM_IO_READ, &cid, &sq_pos);
                s_cids[warp_id][n_groups] = cid;
                s_io_qps[warp_id][n_groups] = (uintptr_t)qp;

                n_groups++;
                pg = group_end;
            }

            // Batch poll all completions
            for (uint32_t g = 0; g < n_groups; g++) {
                QueuePair* qp = (QueuePair*)s_io_qps[warp_id][g];
                uint16_t cid = s_cids[warp_id][g];
                uint32_t poll_loc, poll_head;
                uint32_t cq_pos = cq_poll(&qp->cq, cid,
                                           &poll_loc, &poll_head);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq);
                put_cid(&qp->sq, cid);
            }
        }
        __syncwarp();

        // ── Per-thread decompress + scan (same as v4) ──
        if (lane < n) {
            uint32_t comp_size = d_comp_sizes[warp_base + lane];

            // Source: find compressed data in coalesced cache entry
            uint32_t pc_entry = s_page_pc_entry[warp_id][lane];
            uint32_t offset   = s_page_offset[warp_id][lane];
            const uint8_t* comp_src = (const uint8_t*)(
                base + (uint64_t)pc_entry * coalesce_limit + offset);

            lz4_decompress_per_thread(
                comp_src, comp_size,
                (uint8_t*)my_decomp, page_size);

            // VCHAR scan
            uint32_t nalloc = *(uint32_t*)my_decomp;

            for (uint32_t slot = 0; slot < nalloc; slot++) {
                uint32_t oslt = *(uint32_t*)(
                    my_decomp + page_size
                    - sizeof(uint32_t) * (slot + 1));
                uint16_t vlen = *(uint16_t*)(my_decomp + oslt);
                const uint8_t* vdata = (const uint8_t*)(
                    my_decomp + oslt + sizeof(uint32_t));

                my_records++;
                my_strlen += vlen;
                for (uint16_t b = 0; b < vlen; b++) {
                    my_byte_sum += vdata[b];
                }
            }
        }
    }

    // ── Global accumulate ──
    atomicAdd((unsigned long long*)d_total_records,  (unsigned long long)my_records);
    atomicAdd((unsigned long long*)d_total_strlen,   (unsigned long long)my_strlen);
    atomicAdd((unsigned long long*)d_total_byte_sum, (unsigned long long)my_byte_sum);
}

void bam_scan_o_comment_v5_run(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint64_t npages,
    uint64_t partition_start_lba,
    const uint32_t* h_comp_sizes,
    const uint64_t* h_comp_offsets,
    uint32_t num_blocks,
    uint32_t coalesce_limit,
    uint64_t* out_records,
    uint64_t* out_strlen,
    uint64_t* out_byte_sum)
{
    auto* h = static_cast<BAMCtrlHandle*>(ctrl_handle);

    const uint32_t threads_per_block = 128;
    const uint32_t warps_per_block = 4;
    const uint32_t total_warps = num_blocks * warps_per_block;
    const uint32_t MAX_GROUPS = V5_GROUPS_PER_WARP;

    // Page cache: MAX_GROUPS entries per warp, each coalesce_limit bytes
    const uint64_t n_pc_entries = (uint64_t)total_warps * MAX_GROUPS;
    const uint64_t max_range = 64;

    page_cache_t h_pc(coalesce_limit, n_pc_entries, h->cuda_device,
                      *h->ctrl, max_range, h->ctrls);

    BAM_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    // Decompress buffer: 32 pages per warp (same as v4)
    size_t decomp_size = (uint64_t)total_warps * 32 * (size_t)page_size;
    char* d_decomp = nullptr;
    BAM_CUDA_CHECK(cudaMalloc(&d_decomp, decomp_size));

    // Upload compressed metadata
    uint32_t* d_comp_sizes = nullptr;
    uint64_t* d_comp_offsets = nullptr;
    BAM_CUDA_CHECK(cudaMalloc(&d_comp_sizes, npages * sizeof(uint32_t)));
    BAM_CUDA_CHECK(cudaMemcpy(d_comp_sizes, h_comp_sizes,
                               npages * sizeof(uint32_t), cudaMemcpyHostToDevice));
    BAM_CUDA_CHECK(cudaMalloc(&d_comp_offsets, npages * sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMemcpy(d_comp_offsets, h_comp_offsets,
                               npages * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Output accumulators
    uint64_t* d_total_records = nullptr;
    uint64_t* d_total_strlen  = nullptr;
    uint64_t* d_total_byte_sum = nullptr;
    BAM_CUDA_CHECK(cudaMalloc(&d_total_records, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMalloc(&d_total_strlen, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMalloc(&d_total_byte_sum, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_total_records, 0, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_total_strlen, 0, sizeof(uint64_t)));
    BAM_CUDA_CHECK(cudaMemset(d_total_byte_sum, 0, sizeof(uint64_t)));

    // Launch kernel
    bam_scan_o_comment_v5_kernel<<<num_blocks, threads_per_block>>>(
        h_pc.pdt.d_ctrls, h_pc.d_pc_ptr,
        d_comp_sizes, d_comp_offsets, d_decomp,
        d_total_records, d_total_strlen, d_total_byte_sum,
        page_size, partition_start_lba, npages,
        coalesce_limit, MAX_GROUPS);

    BAM_CUDA_CHECK(cudaDeviceSynchronize());

    // Read results
    BAM_CUDA_CHECK(cudaMemcpy(out_records, d_total_records,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost));
    BAM_CUDA_CHECK(cudaMemcpy(out_strlen, d_total_strlen,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost));
    BAM_CUDA_CHECK(cudaMemcpy(out_byte_sum, d_total_byte_sum,
                               sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Cleanup
    BAM_CUDA_CHECK(cudaFree(d_total_records));
    BAM_CUDA_CHECK(cudaFree(d_total_strlen));
    BAM_CUDA_CHECK(cudaFree(d_total_byte_sum));
    BAM_CUDA_CHECK(cudaFree(d_decomp));
    BAM_CUDA_CHECK(cudaFree(d_comp_sizes));
    BAM_CUDA_CHECK(cudaFree(d_comp_offsets));
}

// ============================================================
// Batch page reader: reads N pages via BaM using existing ctx
// ============================================================
__global__ void bam_batch_read_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    BAMBatchReadEntry* d_entries,
    uint32_t page_size,
    uint32_t n_pages)
{
    uint32_t pg = blockIdx.x;
    if (pg >= n_pages) return;
    uint32_t tid = threadIdx.x;
    if (tid == 0) {
        BAMBatchReadEntry e = d_entries[pg];
        QueuePair* qp = ctrls[e.dev]->d_qps + (pg % ctrls[e.dev]->n_qps);
        uint16_t cid = 0, sq_pos = 0;
        access_data_async(pc, qp, e.lba, e.nblk, (uint64_t)pg,
                          NVM_IO_READ, &cid, &sq_pos);
        uint32_t pl, ph;
        uint32_t cq_pos = cq_poll(&qp->cq, cid, &pl, &ph);
        cq_dequeue(&qp->cq, cq_pos, &qp->sq);
        put_cid(&qp->sq, cid);
        __threadfence_system();
    }
}

void bam_read_pages_batch_to_host(
    bam_pfor32_io_ctx_t ctx_handle,
    uint32_t page_size,
    const BAMBatchReadEntry* h_entries,
    uint32_t n_pages,
    char* h_output,
    cudaStream_t stream)
{
    if (n_pages == 0) return;
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    const uint32_t n_slots = ctx->num_blocks;  // page cache capacity

    BAMBatchReadEntry* d_entries = nullptr;
    uint32_t alloc_sz = std::min(n_pages, n_slots);
    BAM_CUDA_CHECK(cudaMalloc(&d_entries, alloc_sz * sizeof(BAMBatchReadEntry)));

    for (uint32_t off = 0; off < n_pages; off += n_slots) {
        uint32_t chunk = std::min(n_pages - off, n_slots);

        BAM_CUDA_CHECK(cudaMemcpyAsync(d_entries, h_entries + off,
            chunk * sizeof(BAMBatchReadEntry), cudaMemcpyHostToDevice, stream));

        bam_batch_read_kernel<<<chunk, 32, 0, stream>>>(
            ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
            d_entries, page_size, chunk);
        BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy from page_cache slots [0..chunk) to host [off..off+chunk)
        for (uint32_t i = 0; i < chunk; i++) {
            BAM_CUDA_CHECK(cudaMemcpyAsync(
                h_output + (uint64_t)(off + i) * page_size,
                (char*)ctx->h_pc->pdt.base_addr + (uint64_t)i * page_size,
                page_size, cudaMemcpyDeviceToHost, stream));
        }
        BAM_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    BAM_CUDA_CHECK(cudaFree(d_entries));
}

void bam_read_pages_batch_to_gpu(
    bam_pfor32_io_ctx_t ctx_handle,
    uint32_t page_size,
    const BAMBatchReadEntry* h_entries,
    uint32_t n_pages,
    char* d_output,
    cudaStream_t stream,
    size_t* out_kernel_launches)
{
    if (n_pages == 0) { if (out_kernel_launches) *out_kernel_launches = 0; return; }
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    const uint32_t n_slots = ctx->num_blocks;  // page cache capacity

    // Tile: process min(n_slots, remaining) pages per iteration
    BAMBatchReadEntry* d_entries = nullptr;
    uint32_t alloc_sz = std::min(n_pages, n_slots);
    BAM_CUDA_CHECK(cudaMalloc(&d_entries, alloc_sz * sizeof(BAMBatchReadEntry)));

    size_t kl = 0;
    for (uint32_t off = 0; off < n_pages; off += n_slots) {
        uint32_t chunk = std::min(n_pages - off, n_slots);

        BAM_CUDA_CHECK(cudaMemcpyAsync(d_entries, h_entries + off,
            chunk * sizeof(BAMBatchReadEntry), cudaMemcpyHostToDevice, stream));

        bam_batch_read_kernel<<<chunk, 32, 0, stream>>>(
            ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
            d_entries, page_size, chunk);
        kl++;
        BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy from page_cache slots [0..chunk) to output [off..off+chunk)
        for (uint32_t i = 0; i < chunk; i++) {
            BAM_CUDA_CHECK(cudaMemcpyAsync(
                d_output + (uint64_t)(off + i) * page_size,
                (char*)ctx->h_pc->pdt.base_addr + (uint64_t)i * page_size,
                page_size, cudaMemcpyDeviceToDevice, stream));
        }
        BAM_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    BAM_CUDA_CHECK(cudaFree(d_entries));
    if (out_kernel_launches) *out_kernel_launches = kl;
}

void bam_read_pages_batch_to_gpu_prealloc(
    bam_pfor32_io_ctx_t ctx_handle,
    uint32_t page_size,
    const BAMBatchReadEntry* h_entries,
    uint32_t n_pages,
    char* d_output,
    BAMBatchReadEntry* d_entries,
    cudaStream_t stream,
    size_t* out_kernel_launches)
{
    if (n_pages == 0) { if (out_kernel_launches) *out_kernel_launches = 0; return; }
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    const uint32_t n_slots = ctx->num_blocks;

    size_t kl = 0;
    for (uint32_t off = 0; off < n_pages; off += n_slots) {
        uint32_t chunk = std::min(n_pages - off, n_slots);

        BAM_CUDA_CHECK(cudaMemcpyAsync(d_entries, h_entries + off,
            chunk * sizeof(BAMBatchReadEntry), cudaMemcpyHostToDevice, stream));

        bam_batch_read_kernel<<<chunk, 32, 0, stream>>>(
            ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
            d_entries, page_size, chunk);
        kl++;
        BAM_CUDA_CHECK(cudaStreamSynchronize(stream));

        for (uint32_t i = 0; i < chunk; i++) {
            BAM_CUDA_CHECK(cudaMemcpyAsync(
                d_output + (uint64_t)(off + i) * page_size,
                (char*)ctx->h_pc->pdt.base_addr + (uint64_t)i * page_size,
                page_size, cudaMemcpyDeviceToDevice, stream));
        }
        BAM_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (out_kernel_launches) *out_kernel_launches = kl;
}

// ============================================================
// SSB fused kernels: BaM IO + PFOR decomp + scan in one kernel
// ============================================================

// Helper: HT probe (duplicated from q2x.cuh for use inside bam_kernel.cu)
static constexpr int32_t SSB_FUSED_HT_EMPTY = -1;
__device__ __forceinline__ uint32_t ssb_fused_hash32(uint32_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}
__device__ __forceinline__ int32_t ssb_fused_ht_probe(
    const int32_t *keys, const int32_t *values, uint32_t mask, int32_t key) {
    uint32_t slot = ssb_fused_hash32((uint32_t)key) & mask;
    while (true) {
        int32_t k = keys[slot];
        if (k == key) return values[slot];
        if (k == SSB_FUSED_HT_EMPTY) return -1;
        slot = (slot + 1) & mask;
    }
}

// ── SSB Q1x fused kernel ──
__global__ void ssb_dpf_q1x_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    SSBDpfQ1xParams p,
    int64_t*        d_revenue)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    int64_t local_rev = 0;

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;

        uint32_t nrows = (pg == 0) ? (uint32_t)p.d_prefix_sum[0]
                                   : (uint32_t)(p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1]);

        // Batch-submit 4 NVMe reads
        uint16_t cids[4];
        QueuePair* qps_arr[4];
        if (tid == 0) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba;
                uint32_t nblk;
                if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[fi][pg]);
                } else {
                    uint64_t local_pg = global_pg / ndev;
                    lba = p.partition_start_lbas[dev] + local_pg * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            for (int fi = 0; fi < 4; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
            __threadfence_system();
        }
        __syncthreads();

        // Decompress
        int32_t* field_data[4];
        bool all_uncomp = (p.comp_methods[0] == 0 && p.comp_methods[1] == 0 &&
                           p.comp_methods[2] == 0 && p.comp_methods[3] == 0);
        if (all_uncomp) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                char* pp = (char*)pc->base_addr + slot * p.page_size;
                field_data[fi] = (int32_t*)(pp + BAM_PAG_HDR_BYTES);
            }
        } else {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                char* pp = (char*)pc->base_addr + slot * p.page_size;
                field_data[fi] = p.d_scratch + ((uint64_t)bid * 4 + fi) * p.scratch_stride;
                decompress_page_128(pp, field_data[fi], p.comp_methods[fi],
                                    shared_buf32, tid, nullptr);
                __syncthreads();
            }
        }

        // Q1x evaluation
        for (uint32_t i = tid; i < nrows; i += blockDim.x) {
            int32_t od = field_data[0][i];
            int32_t qt = field_data[1][i];
            int32_t dc = field_data[2][i];
            int32_t ep = field_data[3][i];
            bool date_ok = (ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values,
                                                p.date_ht_mask, od) >= 0);
            if (date_ok && dc >= p.disc_lo && dc <= p.disc_hi &&
                qt >= p.qty_lo && qt < p.qty_hi) {
                local_rev += (int64_t)ep * dc;
            }
        }
        __syncthreads();
    }

    // Block-level reduction
    __shared__ int64_t shared_rev[128];
    shared_rev[tid] = local_rev;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) shared_rev[tid] += shared_rev[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                  static_cast<unsigned long long>(shared_rev[0]));
}

void ssb_dpf_q1x_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const SSBDpfQ1xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    ssb_dpf_q1x_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_revenue);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ── SSB Q2x fused kernel ──
__global__ void ssb_dpf_q2x_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    SSBDpfQ2xParams p,
    int64_t*        d_revenue)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    constexpr uint32_t HIST_SIZE = 7 * 40;  // SSB_MAX_YEARS * SSB_MAX_BRANDS = 280
    __shared__ int64_t s_hist[HIST_SIZE];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Initialize shared histogram (persists across all pages)
    for (uint32_t i = tid; i < HIST_SIZE; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;
        uint32_t nrows = (pg == 0) ? (uint32_t)p.d_prefix_sum[0]
                                   : (uint32_t)(p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1]);

        uint16_t cids[4]; QueuePair* qps_arr[4];
        if (tid == 0) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba; uint32_t nblk;
                if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[fi][pg]);
                } else {
                    lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            for (int fi = 0; fi < 4; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
            __threadfence_system();
        }
        __syncthreads();

        int32_t* field_data[4];
        bool all_uncomp = (p.comp_methods[0] == 0 && p.comp_methods[1] == 0 &&
                           p.comp_methods[2] == 0 && p.comp_methods[3] == 0);
        if (all_uncomp) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                field_data[fi] = (int32_t*)((char*)pc->base_addr + slot * p.page_size + BAM_PAG_HDR_BYTES);
            }
        } else {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                char* pp = (char*)pc->base_addr + slot * p.page_size;
                field_data[fi] = p.d_scratch + ((uint64_t)bid * 4 + fi) * p.scratch_stride;
                decompress_page_128(pp, field_data[fi], p.comp_methods[fi],
                                    shared_buf32, tid, nullptr);
                __syncthreads();
            }
        }

        // Q2x: date lookup + supp probe + part probe → shared histogram
        for (uint32_t i = tid; i < nrows; i += blockDim.x) {
            int32_t od = field_data[0][i];
            int32_t pk = field_data[1][i];
            int32_t sk = field_data[2][i];
            int32_t rv = field_data[3][i];

            int32_t yi = ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, od);
            if (yi < 0) continue;

            int32_t sv = ssb_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, sk);
            if (sv < 0) continue;
            int32_t bi = ssb_fused_ht_probe(p.d_part_ht_keys, p.d_part_ht_values, p.part_ht_mask, pk);
            if (bi < 0) continue;

            int32_t gi = yi * 40 + bi;
            atomicAdd(reinterpret_cast<unsigned long long*>(&s_hist[gi]),
                      static_cast<unsigned long long>((int64_t)rv));
        }
        __syncthreads();
    }

    // Flush shared histogram to global
    for (uint32_t i = tid; i < HIST_SIZE; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_revenue[i]),
                      static_cast<unsigned long long>(s_hist[i]));
    }
}

void ssb_dpf_q2x_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const SSBDpfQ2xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    ssb_dpf_q2x_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_revenue);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ── SSB Q3x fused kernel ──
__global__ void ssb_dpf_q3x_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    SSBDpfQ3xParams p,
    int64_t*        d_revenue)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    extern __shared__ int64_t s_hist[];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Initialize shared histogram (persists across page loop)
    for (uint32_t i = tid; i < p.hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;
        uint32_t nrows = (pg == 0) ? (uint32_t)p.d_prefix_sum[0]
                                   : (uint32_t)(p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1]);

        uint16_t cids[4]; QueuePair* qps_arr[4];
        if (tid == 0) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba; uint32_t nblk;
                if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[fi][pg]);
                } else {
                    lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            for (int fi = 0; fi < 4; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
            __threadfence_system();
        }
        __syncthreads();

        int32_t* field_data[4];
        bool all_uncomp = (p.comp_methods[0] == 0 && p.comp_methods[1] == 0 &&
                           p.comp_methods[2] == 0 && p.comp_methods[3] == 0);
        if (all_uncomp) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                field_data[fi] = (int32_t*)((char*)pc->base_addr + slot * p.page_size + BAM_PAG_HDR_BYTES);
            }
        } else {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                char* pp = (char*)pc->base_addr + slot * p.page_size;
                field_data[fi] = p.d_scratch + ((uint64_t)bid * 4 + fi) * p.scratch_stride;
                decompress_page_128(pp, field_data[fi], p.comp_methods[fi],
                                    shared_buf32, tid, nullptr);
                __syncthreads();
            }
        }

        for (uint32_t i = tid; i < nrows; i += blockDim.x) {
            int32_t od = field_data[0][i];
            int32_t ck = field_data[1][i];
            int32_t sk = field_data[2][i];
            int32_t rv = field_data[3][i];

            int32_t yi = ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, od);
            if (yi < 0) continue;

            int32_t cd = ssb_fused_ht_probe(p.d_cust_ht_keys, p.d_cust_ht_values, p.cust_ht_mask, ck);
            if (cd < 0) continue;
            int32_t sd = ssb_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, sk);
            if (sd < 0) continue;

            int32_t gi = cd * p.num_supp_dims * (int32_t)p.max_years + sd * (int32_t)p.max_years + yi;
            atomicAdd(reinterpret_cast<unsigned long long*>(&s_hist[gi]),
                      static_cast<unsigned long long>((int64_t)rv));
        }
        __syncthreads();
    }

    // Flush shared histogram to global
    for (uint32_t i = tid; i < p.hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_revenue[i]),
                      static_cast<unsigned long long>(s_hist[i]));
    }
}

void ssb_dpf_q3x_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const SSBDpfQ3xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    size_t hist_smem = (size_t)params.hist_size * sizeof(int64_t);
    ssb_dpf_q3x_kernel<<<params.num_blocks, 128, hist_smem, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_revenue);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ── SSB Q4x fused kernel (6 fields) ──
__global__ void ssb_dpf_q4x_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    SSBDpfQ4xParams p,
    int64_t*        d_profit)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint shared_buf32[549];
    extern __shared__ int64_t s_hist[];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    // Initialize shared histogram (persists across page loop)
    for (uint32_t i = tid; i < p.hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;
        uint32_t nrows = (pg == 0) ? (uint32_t)p.d_prefix_sum[0]
                                   : (uint32_t)(p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1]);

        uint16_t cids[6]; QueuePair* qps_arr[6];
        if (tid == 0) {
            for (int fi = 0; fi < 6; fi++) {
                uint64_t slot = (uint64_t)bid * 6 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba; uint32_t nblk;
                if (p.comp_methods[fi] != 0 && p.d_comp_offsets[fi]) {
                    lba = p.partition_start_lbas[dev] + p.d_comp_offsets[fi][pg] / 512;
                    nblk = safe_io_nblocks_vchar_io(p.d_comp_sizes[fi][pg]);
                } else {
                    lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                    nblk = p.blocks_per_page;
                }
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            for (int fi = 0; fi < 6; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
            __threadfence_system();
        }
        __syncthreads();

        int32_t* field_data[6];
        bool all_uncomp = true;
        for (int fi = 0; fi < 6; fi++) if (p.comp_methods[fi] != 0) all_uncomp = false;
        if (all_uncomp) {
            for (int fi = 0; fi < 6; fi++) {
                uint64_t slot = (uint64_t)bid * 6 + fi;
                field_data[fi] = (int32_t*)((char*)pc->base_addr + slot * p.page_size + BAM_PAG_HDR_BYTES);
            }
        } else {
            for (int fi = 0; fi < 6; fi++) {
                uint64_t slot = (uint64_t)bid * 6 + fi;
                char* pp = (char*)pc->base_addr + slot * p.page_size;
                field_data[fi] = p.d_scratch + ((uint64_t)bid * 6 + fi) * p.scratch_stride;
                decompress_page_128(pp, field_data[fi], p.comp_methods[fi],
                                    shared_buf32, tid, nullptr);
                __syncthreads();
            }
        }

        for (uint32_t i = tid; i < nrows; i += blockDim.x) {
            int32_t od = field_data[0][i];
            int32_t ck = field_data[1][i];
            int32_t pk = field_data[2][i];
            int32_t sk = field_data[3][i];
            int32_t rv = field_data[4][i];
            int32_t sc = field_data[5][i];

            int32_t yi = ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, od);
            if (yi < 0) continue;

            int32_t cv = ssb_fused_ht_probe(p.d_cust_ht_keys, p.d_cust_ht_values, p.cust_ht_mask, ck);
            if (cv < 0) continue;
            int32_t sv = ssb_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, sk);
            if (sv < 0) continue;
            int32_t pv = ssb_fused_ht_probe(p.d_part_ht_keys, p.d_part_ht_values, p.part_ht_mask, pk);
            if (pv < 0) continue;

            int32_t gi = yi * p.stride_year + cv * (p.supp_dims * p.part_dims) + sv * p.part_dims + pv;
            int64_t profit = (int64_t)rv - (int64_t)sc;
            atomicAdd(reinterpret_cast<unsigned long long*>(&s_hist[gi]),
                      static_cast<unsigned long long>(profit));
        }
        __syncthreads();
    }

    // Flush shared histogram to global
    for (uint32_t i = tid; i < p.hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_profit[i]),
                      static_cast<unsigned long long>(s_hist[i]));
    }
}

void ssb_dpf_q4x_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const SSBDpfQ4xParams& params,
    int64_t* d_profit,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    size_t hist_smem = (size_t)params.hist_size * sizeof(int64_t);
    ssb_dpf_q4x_kernel<<<params.num_blocks, 128, hist_smem, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_profit);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// SSB Uncompressed DPF kernels — no PFOR shared workspace
// ════════════════════════════════════════════════════════════════

__global__ void ssb_dpf_q1x_uncomp_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    SSBDpfQ1xParams p,
    int64_t*        d_revenue)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    int64_t local_rev = 0;

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;
        uint32_t nrows = (pg == 0) ? (uint32_t)p.d_prefix_sum[0]
                                   : (uint32_t)(p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1]);

        uint16_t cids[4]; QueuePair* qps_arr[4];
        if (tid == 0) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                uint32_t nblk = p.blocks_per_page;
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            for (int fi = 0; fi < 4; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
            __threadfence_system();
        }
        __syncthreads();

        int32_t* field_data[4];
        for (int fi = 0; fi < 4; fi++) {
            uint64_t slot = (uint64_t)bid * 4 + fi;
            field_data[fi] = (int32_t*)((char*)pc->base_addr + slot * p.page_size + BAM_PAG_HDR_BYTES);
        }

        for (uint32_t i = tid; i < nrows; i += blockDim.x) {
            int32_t od = field_data[0][i];
            int32_t qt = field_data[1][i];
            int32_t dc = field_data[2][i];
            int32_t ep = field_data[3][i];
            bool date_ok = (ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values,
                                                p.date_ht_mask, od) >= 0);
            if (date_ok && dc >= p.disc_lo && dc <= p.disc_hi &&
                qt >= p.qty_lo && qt < p.qty_hi) {
                local_rev += (int64_t)ep * dc;
            }
        }
        __syncthreads();
    }

    __shared__ int64_t shared_rev[128];
    shared_rev[tid] = local_rev;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if ((int)tid < s) shared_rev[tid] += shared_rev[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(d_revenue),
                  static_cast<unsigned long long>(shared_rev[0]));
}

void ssb_dpf_q1x_uncomp_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const SSBDpfQ1xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    ssb_dpf_q1x_uncomp_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_revenue);
    BAM_CUDA_CHECK(cudaGetLastError());
}

__global__ void ssb_dpf_q2x_uncomp_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    SSBDpfQ2xParams p,
    int64_t*        d_revenue)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    constexpr uint32_t HIST_SIZE = 7 * 40;
    __shared__ int64_t s_hist[HIST_SIZE];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint32_t i = tid; i < HIST_SIZE; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;
        uint32_t nrows = (pg == 0) ? (uint32_t)p.d_prefix_sum[0]
                                   : (uint32_t)(p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1]);

        uint16_t cids[4]; QueuePair* qps_arr[4];
        if (tid == 0) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                uint32_t nblk = p.blocks_per_page;
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            for (int fi = 0; fi < 4; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
            __threadfence_system();
        }
        __syncthreads();

        int32_t* field_data[4];
        for (int fi = 0; fi < 4; fi++) {
            uint64_t slot = (uint64_t)bid * 4 + fi;
            field_data[fi] = (int32_t*)((char*)pc->base_addr + slot * p.page_size + BAM_PAG_HDR_BYTES);
        }

        for (uint32_t i = tid; i < nrows; i += blockDim.x) {
            int32_t od = field_data[0][i];
            int32_t pk = field_data[1][i];
            int32_t sk = field_data[2][i];
            int32_t rv = field_data[3][i];
            int32_t yi = ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, od);
            if (yi < 0) continue;
            int32_t sv = ssb_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, sk);
            if (sv < 0) continue;
            int32_t bi = ssb_fused_ht_probe(p.d_part_ht_keys, p.d_part_ht_values, p.part_ht_mask, pk);
            if (bi < 0) continue;
            int32_t gi = yi * 40 + bi;
            atomicAdd(reinterpret_cast<unsigned long long*>(&s_hist[gi]),
                      static_cast<unsigned long long>((int64_t)rv));
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < HIST_SIZE; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_revenue[i]),
                      static_cast<unsigned long long>(s_hist[i]));
    }
}

void ssb_dpf_q2x_uncomp_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const SSBDpfQ2xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    ssb_dpf_q2x_uncomp_kernel<<<params.num_blocks, 128, 0, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_revenue);
    BAM_CUDA_CHECK(cudaGetLastError());
}

__global__ void ssb_dpf_q3x_uncomp_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    SSBDpfQ3xParams p,
    int64_t*        d_revenue)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    extern __shared__ int64_t s_hist[];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint32_t i = tid; i < p.hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;
        uint32_t nrows = (pg == 0) ? (uint32_t)p.d_prefix_sum[0]
                                   : (uint32_t)(p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1]);

        uint16_t cids[4]; QueuePair* qps_arr[4];
        if (tid == 0) {
            for (int fi = 0; fi < 4; fi++) {
                uint64_t slot = (uint64_t)bid * 4 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                uint32_t nblk = p.blocks_per_page;
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            for (int fi = 0; fi < 4; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
            __threadfence_system();
        }
        __syncthreads();

        int32_t* field_data[4];
        for (int fi = 0; fi < 4; fi++) {
            uint64_t slot = (uint64_t)bid * 4 + fi;
            field_data[fi] = (int32_t*)((char*)pc->base_addr + slot * p.page_size + BAM_PAG_HDR_BYTES);
        }

        for (uint32_t i = tid; i < nrows; i += blockDim.x) {
            int32_t od = field_data[0][i];
            int32_t ck = field_data[1][i];
            int32_t sk = field_data[2][i];
            int32_t rv = field_data[3][i];
            int32_t yi = ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, od);
            if (yi < 0) continue;
            int32_t cd = ssb_fused_ht_probe(p.d_cust_ht_keys, p.d_cust_ht_values, p.cust_ht_mask, ck);
            if (cd < 0) continue;
            int32_t sd = ssb_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, sk);
            if (sd < 0) continue;
            int32_t gi = cd * p.num_supp_dims * (int32_t)p.max_years + sd * (int32_t)p.max_years + yi;
            atomicAdd(reinterpret_cast<unsigned long long*>(&s_hist[gi]),
                      static_cast<unsigned long long>((int64_t)rv));
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < p.hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_revenue[i]),
                      static_cast<unsigned long long>(s_hist[i]));
    }
}

void ssb_dpf_q3x_uncomp_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const SSBDpfQ3xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    size_t hist_smem = (size_t)params.hist_size * sizeof(int64_t);
    ssb_dpf_q3x_uncomp_kernel<<<params.num_blocks, 128, hist_smem, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_revenue);
    BAM_CUDA_CHECK(cudaGetLastError());
}

__global__ void ssb_dpf_q4x_uncomp_kernel(
    Controller**    ctrls,
    page_cache_d_t* pc,
    SSBDpfQ4xParams p,
    int64_t*        d_profit)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    extern __shared__ int64_t s_hist[];
    const uint32_t ndev = (p.n_devices > 1) ? p.n_devices : 1;

    for (uint32_t i = tid; i < p.hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    const uint64_t _loop_n = p.d_active_page_ids ? p.num_active_pages : p.npages;
    for (uint64_t _iter = bid; _iter < _loop_n; _iter += gridDim.x) {
        const uint64_t pg = p.d_active_page_ids ? p.d_active_page_ids[_iter] : _iter;
        if (!p.d_active_page_ids && p.d_page_active && !p.d_page_active[pg]) continue;
        uint32_t nrows = (pg == 0) ? (uint32_t)p.d_prefix_sum[0]
                                   : (uint32_t)(p.d_prefix_sum[pg] - p.d_prefix_sum[pg - 1]);

        uint16_t cids[6]; QueuePair* qps_arr[6];
        if (tid == 0) {
            for (int fi = 0; fi < 6; fi++) {
                uint64_t slot = (uint64_t)bid * 6 + fi;
                uint64_t global_pg = p.field_start_page_ids[fi] + pg;
                uint32_t dev = global_pg % ndev;
                qps_arr[fi] = ctrls[dev]->d_qps + (slot % ctrls[dev]->n_qps);
                uint64_t lba = p.partition_start_lbas[dev] + (global_pg / ndev) * p.blocks_per_page;
                uint32_t nblk = p.blocks_per_page;
                uint16_t sq_pos = 0;
                access_data_async(pc, qps_arr[fi], lba, nblk, slot,
                                  NVM_IO_READ, &cids[fi], &sq_pos);
            }
            for (int fi = 0; fi < 6; fi++) {
                uint32_t pl, ph;
                uint32_t cq_pos = cq_poll(&qps_arr[fi]->cq, cids[fi], &pl, &ph);
                cq_dequeue(&qps_arr[fi]->cq, cq_pos, &qps_arr[fi]->sq);
                put_cid(&qps_arr[fi]->sq, cids[fi]);
            }
            __threadfence_system();
        }
        __syncthreads();

        int32_t* field_data[6];
        for (int fi = 0; fi < 6; fi++) {
            uint64_t slot = (uint64_t)bid * 6 + fi;
            field_data[fi] = (int32_t*)((char*)pc->base_addr + slot * p.page_size + BAM_PAG_HDR_BYTES);
        }

        for (uint32_t i = tid; i < nrows; i += blockDim.x) {
            int32_t od = field_data[0][i];
            int32_t ck = field_data[1][i];
            int32_t pk = field_data[2][i];
            int32_t sk = field_data[3][i];
            int32_t rv = field_data[4][i];
            int32_t sc = field_data[5][i];
            int32_t yi = ssb_fused_ht_probe(p.d_date_ht_keys, p.d_date_ht_values, p.date_ht_mask, od);
            if (yi < 0) continue;
            int32_t cv = ssb_fused_ht_probe(p.d_cust_ht_keys, p.d_cust_ht_values, p.cust_ht_mask, ck);
            if (cv < 0) continue;
            int32_t sv = ssb_fused_ht_probe(p.d_supp_ht_keys, p.d_supp_ht_values, p.supp_ht_mask, sk);
            if (sv < 0) continue;
            int32_t pv = ssb_fused_ht_probe(p.d_part_ht_keys, p.d_part_ht_values, p.part_ht_mask, pk);
            if (pv < 0) continue;
            int32_t gi = yi * p.stride_year + cv * (p.supp_dims * p.part_dims) + sv * p.part_dims + pv;
            int64_t profit = (int64_t)rv - (int64_t)sc;
            atomicAdd(reinterpret_cast<unsigned long long*>(&s_hist[gi]),
                      static_cast<unsigned long long>(profit));
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < p.hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long*>(&d_profit[i]),
                      static_cast<unsigned long long>(s_hist[i]));
    }
}

void ssb_dpf_q4x_uncomp_run(
    bam_pfor32_io_ctx_t ctx_handle,
    const SSBDpfQ4xParams& params,
    int64_t* d_profit,
    cudaStream_t stream)
{
    auto* ctx = static_cast<BAMPfor32IOContext*>(ctx_handle);
    size_t hist_smem = (size_t)params.hist_size * sizeof(int64_t);
    ssb_dpf_q4x_uncomp_kernel<<<params.num_blocks, 128, hist_smem, stream>>>(
        ctx->h_pc->pdt.d_ctrls, ctx->h_pc->d_pc_ptr,
        params, d_profit);
    BAM_CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// scan_o_comment v6: moved to bam_vchar_kernel.cu (PAR-32K-nvCOMPdx)
