#pragma once

#include "common/common.cu"
#include "common/page.cu"
#include "common/primitive_c.cu"
#include "common/primitive_cuda.cu"
#include "common/primitive_cufile.cu"
#include "metadata/ssb_metadata.h"
#include "schema/ssb_tables.cuh"
#include "common/filter.cuh"
#include "kernel/ssb/q1x.cuh"
#include "kernel/ssb/q2x.cuh"
#include "kernel/ssb/q3x.cuh"
#include "kernel/ssb/q4x.cuh"
#include "kernel/ssb/flatten.cuh"
#include "kernel/ssb/dim_build.cuh"
#include "common/pruning.cuh"

#include <span>
#include <numeric>
#include <algorithm>
#include <nvcomp/snappy.h>
#include <nvcomp/lz4.h>
#include <snappy.h>
#include <lz4.h>

#include <set>
#include <atomic>
#include <unordered_map>

namespace SsbGidp {

// Main-thread kernel launch counter (reset per query, aggregated with worker counts).
static size_t s_kernel_launches;

// Collect unique compression method names from FieldPageInfo vectors.
static std::string collect_compression_methods(
    std::initializer_list<std::reference_wrapper<const std::vector<FieldPageInfo>>> field_lists)
{
    std::set<std::string> methods;
    for (const auto &list_ref : field_lists) {
        for (const auto &fi : list_ref.get()) {
            methods.insert(compression_method_name(fi.compression_method));
        }
    }
    std::string result;
    for (const auto &m : methods) {
        if (!result.empty()) result += "+";
        result += m;
    }
    return result;
}

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

#define GDS_CHECK(call) do {                                           \
    CUfileError_t err = (call);                                        \
    if (err.err != CU_FILE_SUCCESS) {                                  \
        std::cerr << "GDS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

// ============================================================
// nvCOMP decompression context
// ============================================================
struct NvcompDecompCtx {
    void   **d_comp_ptrs   = nullptr;
    void   **d_decomp_ptrs = nullptr;
    size_t  *d_comp_sizes  = nullptr;
    size_t  *d_decomp_sizes = nullptr;
    size_t  *d_actual_sizes = nullptr;
    nvcompStatus_t *d_statuses = nullptr;
    void   *d_temp         = nullptr;
    size_t  temp_bytes     = 0;
    void   **h_comp_ptrs   = nullptr;
    void   **h_decomp_ptrs = nullptr;
    size_t  *h_comp_sizes  = nullptr;
    size_t  *h_decomp_sizes = nullptr;
    void   **h_comp_ptrs_1   = nullptr;
    void   **h_decomp_ptrs_1 = nullptr;
    size_t  *h_comp_sizes_1  = nullptr;
    size_t  *h_decomp_sizes_1 = nullptr;
};

static void nvcomp_decompctx_alloc(NvcompDecompCtx &ctx, size_t max_batch,
                                    size_t page_size,
                                    const std::vector<FieldPageInfo> &fields) {
    CUDA_CHECK(cudaMalloc(&ctx.d_comp_ptrs,    max_batch * sizeof(void *)));
    CUDA_CHECK(cudaMalloc(&ctx.d_decomp_ptrs,  max_batch * sizeof(void *)));
    CUDA_CHECK(cudaMalloc(&ctx.d_comp_sizes,   max_batch * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_decomp_sizes, max_batch * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_actual_sizes, max_batch * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_statuses,     max_batch * sizeof(nvcompStatus_t)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_comp_ptrs,    max_batch * sizeof(void *)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_decomp_ptrs,  max_batch * sizeof(void *)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_comp_sizes,   max_batch * sizeof(size_t)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_decomp_sizes, max_batch * sizeof(size_t)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_comp_ptrs_1,    max_batch * sizeof(void *)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_decomp_ptrs_1,  max_batch * sizeof(void *)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_comp_sizes_1,   max_batch * sizeof(size_t)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_decomp_sizes_1, max_batch * sizeof(size_t)));

    size_t max_total_uncompressed = max_batch * page_size;
    ctx.temp_bytes = 0;
    for (auto &fi : fields) {
        size_t temp_bytes = 0;
        if (fi.compression_method == CompressionMethod::SNAPPY) {
            nvcompBatchedSnappyDecompressGetTempSizeAsync(
                max_batch, page_size,
                nvcompBatchedSnappyDecompressDefaultOpts,
                &temp_bytes, max_total_uncompressed);
        } else if (fi.compression_method == CompressionMethod::LZ4) {
            nvcompBatchedLZ4DecompressGetTempSizeAsync(
                max_batch, page_size,
                nvcompBatchedLZ4DecompressDefaultOpts,
                &temp_bytes, max_total_uncompressed);
        }
        ctx.temp_bytes = std::max(ctx.temp_bytes, temp_bytes);
    }
    if (ctx.temp_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&ctx.d_temp, ctx.temp_bytes));
    }
}

static size_t query_nvcomp_temp_size(
    size_t max_batch, size_t page_size,
    const std::vector<FieldPageInfo> &fields)
{
    size_t max_total_uncompressed = max_batch * page_size;
    size_t temp = 0;
    for (auto &fi : fields) {
        size_t tb = 0;
        if (fi.compression_method == CompressionMethod::SNAPPY)
            nvcompBatchedSnappyDecompressGetTempSizeAsync(
                max_batch, page_size,
                nvcompBatchedSnappyDecompressDefaultOpts,
                &tb, max_total_uncompressed);
        else if (fi.compression_method == CompressionMethod::LZ4)
            nvcompBatchedLZ4DecompressGetTempSizeAsync(
                max_batch, page_size,
                nvcompBatchedLZ4DecompressDefaultOpts,
                &tb, max_total_uncompressed);
        temp = std::max(temp, tb);
    }
    return temp;
}

static void nvcomp_decompctx_free(NvcompDecompCtx &ctx) {
    if (ctx.d_comp_ptrs)    cudaFree(ctx.d_comp_ptrs);
    if (ctx.d_decomp_ptrs)  cudaFree(ctx.d_decomp_ptrs);
    if (ctx.d_comp_sizes)   cudaFree(ctx.d_comp_sizes);
    if (ctx.d_decomp_sizes) cudaFree(ctx.d_decomp_sizes);
    if (ctx.d_actual_sizes) cudaFree(ctx.d_actual_sizes);
    if (ctx.d_statuses)     cudaFree(ctx.d_statuses);
    if (ctx.d_temp)         cudaFree(ctx.d_temp);
    if (ctx.h_comp_ptrs)    cudaFreeHost(ctx.h_comp_ptrs);
    if (ctx.h_decomp_ptrs)  cudaFreeHost(ctx.h_decomp_ptrs);
    if (ctx.h_comp_sizes)   cudaFreeHost(ctx.h_comp_sizes);
    if (ctx.h_decomp_sizes) cudaFreeHost(ctx.h_decomp_sizes);
    if (ctx.h_comp_ptrs_1)    cudaFreeHost(ctx.h_comp_ptrs_1);
    if (ctx.h_decomp_ptrs_1)  cudaFreeHost(ctx.h_decomp_ptrs_1);
    if (ctx.h_comp_sizes_1)   cudaFreeHost(ctx.h_comp_sizes_1);
    if (ctx.h_decomp_sizes_1) cudaFreeHost(ctx.h_decomp_sizes_1);
}

static void nvcomp_decompctx_run(CompressionMethod method,
                                  NvcompDecompCtx &ctx,
                                  size_t num_chunks,
                                  size_t page_size,
                                  cudaStream_t stream,
                                  bool do_sync = true,
                                  bool skip_h2d = false,
                                  int h_set = 0) {
    if (num_chunks == 0) return;

    if (!skip_h2d) {
        void  **h_cp  = (h_set == 0) ? ctx.h_comp_ptrs    : ctx.h_comp_ptrs_1;
        void  **h_dp  = (h_set == 0) ? ctx.h_decomp_ptrs  : ctx.h_decomp_ptrs_1;
        size_t *h_cs  = (h_set == 0) ? ctx.h_comp_sizes   : ctx.h_comp_sizes_1;
        size_t *h_ds  = (h_set == 0) ? ctx.h_decomp_sizes : ctx.h_decomp_sizes_1;
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_comp_ptrs, h_cp,
                                    num_chunks * sizeof(void *),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_decomp_ptrs, h_dp,
                                    num_chunks * sizeof(void *),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_comp_sizes, h_cs,
                                    num_chunks * sizeof(size_t),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_decomp_sizes, h_ds,
                                    num_chunks * sizeof(size_t),
                                    cudaMemcpyHostToDevice, stream));
    }

    nvcompStatus_t nvstatus;
    if (method == CompressionMethod::SNAPPY) {
        nvstatus = nvcompBatchedSnappyDecompressAsync(
            (const void *const *)ctx.d_comp_ptrs,
            ctx.d_comp_sizes, ctx.d_decomp_sizes,
            ctx.d_actual_sizes, num_chunks,
            ctx.d_temp, ctx.temp_bytes,
            (void *const *)ctx.d_decomp_ptrs,
            nvcompBatchedSnappyDecompressDefaultOpts,
            ctx.d_statuses, stream);
    } else if (method == CompressionMethod::LZ4) {
        nvstatus = nvcompBatchedLZ4DecompressAsync(
            (const void *const *)ctx.d_comp_ptrs,
            ctx.d_comp_sizes, ctx.d_decomp_sizes,
            ctx.d_actual_sizes, num_chunks,
            ctx.d_temp, ctx.temp_bytes,
            (void *const *)ctx.d_decomp_ptrs,
            nvcompBatchedLZ4DecompressDefaultOpts,
            ctx.d_statuses, stream);
    } else {
        std::cerr << "Unsupported compression method: "
                  << static_cast<int>(method) << std::endl;
        return;
    }
    if (nvstatus != nvcompSuccess) {
        std::cerr << "nvCOMP batch decompress failed: " << nvstatus << std::endl;
    }
    if (do_sync) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

// ────────────────────────────────────────────────────────────
// Compute tile-local prefix sum from full prefix sum on GPU.
// batch_ps[i] = full_ps[pg + i + 1] - full_ps[pg]  (cumulative nrecs).
// ────────────────────────────────────────────────────────────
__global__ void compute_batch_ps_kernel(const uint64_t *full_ps, uint64_t *batch_ps,
                                         uint32_t pg, uint32_t bnp) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bnp) {
        batch_ps[i] = full_ps[pg + i + 1] - full_ps[pg];
    }
}

// ────────────────────────────────────────────────────────────
// Packed variant: data_buf contains only active pages in order of
// d_active_ids[tile_active_start..tile_active_start+n_active).
// batch_ps[i] = sum_{j<=i} nrecs(d_active_ids[tile_active_start+j])
// ────────────────────────────────────────────────────────────
__global__ void compute_packed_ps_from_active_ids_kernel(
    const uint64_t *full_ps, uint64_t *batch_ps,
    const uint32_t *d_active_ids,
    uint32_t tile_active_start, uint32_t n_active)
{
    // Single-thread scan; n_active is small (<= Q3X_TILE_PAGES).
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t cum = 0;
    for (uint32_t i = 0; i < n_active; i++) {
        uint32_t abs_pg = d_active_ids[tile_active_start + i];
        cum += full_ps[abs_pg + 1] - full_ps[abs_pg];
        batch_ps[i] = cum;
    }
}

// ────────────────────────────────────────────────────────────
// Selective variant: inactive pages contribute 0 nrecs.
// page_active_mask[pg] is 1 for active, 0 for inactive.
// Produces cumulative prefix sum suitable for flatten kernel.
// ────────────────────────────────────────────────────────────
__global__ void compute_selective_batch_ps_kernel(
    const uint64_t *full_ps, uint64_t *batch_ps,
    const uint8_t *page_active_mask,
    uint32_t tile_start, uint32_t tile_npages)
{
    // Single-thread scan (tile_npages is small, typically <= 3072).
    // Launched with <<<1, 1>>>.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t cum = 0;
    for (uint32_t i = 0; i < tile_npages; i++) {
        uint32_t abs_pg = tile_start + i;
        if (page_active_mask[abs_pg]) {
            cum += full_ps[abs_pg + 1] - full_ps[abs_pg];
        }
        batch_ps[i] = cum;
    }
}

// ────────────────────────────────────────────────────────────
// Zero only the 12-byte page header of every page in buf.
// ────────────────────────────────────────────────────────────
__global__ void ssb_zero_page_headers_kernel(char *buf, uint32_t npages, uint32_t page_size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < npages) {
        uint32_t *hdr = reinterpret_cast<uint32_t *>(buf + (uint64_t)tid * page_size);
        hdr[0] = 0;  // nalloc
        hdr[1] = 0;  // watermark
        hdr[2] = 0;  // lfreespace
    }
}

// ============================================================
// LO Worker: per-worker multi-column IO pipeline
// Each worker handles a page range across ALL columns with
// double-buffered IO-decomp overlap across columns.
// ============================================================
static constexpr size_t LO_MAX_DEVICES = 8;

struct LoWorkerCtx {
    CUfileHandle_t cufile_handles[LO_MAX_DEVICES];
    int dup_fds[LO_MAX_DEVICES];
    void *staging_buf[2];
    size_t staging_buf_npages;
    cudaStream_t io_stream;
    cudaStream_t decomp_stream;
    cudaEvent_t io_done;
    cudaEvent_t buf_done[2];
    NvcompDecompCtx nvctx;
    size_t bytes_read;
    size_t ios_completed;
    size_t kernel_launches;
};

static void lo_worker_alloc(
    LoWorkerCtx *workers, size_t nworkers,
    const int *fds, size_t num_devices, size_t page_size,
    size_t pages_per_worker, bool any_compressed,
    const std::vector<FieldPageInfo> &lo_page_infos,
    bool need_zonemap_db)
{
    for (size_t t = 0; t < nworkers; t++) {
        auto &w = workers[t];
        w.bytes_read = 0;
        w.ios_completed = 0;
        w.kernel_launches = 0;
        w.staging_buf_npages = pages_per_worker;
        for (size_t d = 0; d < num_devices; d++) {
            w.dup_fds[d] = dup(fds[d]);
            if (w.dup_fds[d] < 0) { std::cerr << "dup failed" << std::endl; exit(EXIT_FAILURE); }
            w.cufile_handles[d] = mb_cufile_handle_register(w.dup_fds[d]);
        }
        size_t buf_size = pages_per_worker * page_size;
        for (int b = 0; b < 2; b++) {
            w.staging_buf[b] = mb_cuda_alloc(buf_size);
            GDS_CHECK(cuFileBufRegister(w.staging_buf[b], buf_size, 0));
        }
        CUDA_CHECK(cudaStreamCreate(&w.io_stream));
        CUDA_CHECK(cudaStreamCreate(&w.decomp_stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&w.io_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&w.buf_done[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&w.buf_done[1], cudaEventDisableTiming));
        if (any_compressed) {
            nvcomp_decompctx_alloc(w.nvctx, pages_per_worker, page_size, lo_page_infos);
        }
    }
}

static void lo_worker_free(
    LoWorkerCtx *workers, size_t nworkers,
    size_t num_devices, bool any_compressed)
{
    for (size_t t = 0; t < nworkers; t++) {
        auto &w = workers[t];
        if (any_compressed) nvcomp_decompctx_free(w.nvctx);
        for (int b = 0; b < 2; b++) {
            cuFileBufDeregister(w.staging_buf[b]);
            mb_cuda_free(w.staging_buf[b]);
        }
        CUDA_CHECK(cudaStreamDestroy(w.io_stream));
        CUDA_CHECK(cudaStreamDestroy(w.decomp_stream));
        CUDA_CHECK(cudaEventDestroy(w.io_done));
        CUDA_CHECK(cudaEventDestroy(w.buf_done[0]));
        CUDA_CHECK(cudaEventDestroy(w.buf_done[1]));
        for (size_t d = 0; d < num_devices; d++) {
            cuFileHandleDeregister(w.cufile_handles[d]);
            close(w.dup_fds[d]);
        }
    }
}

// Read one column's pages for this worker into staging_buf[buf_idx],
// set up nvCOMP pointers, and scatter uncompressed pages via D→D copy.
// Returns decomp_count (number of pages needing nvCOMP decomp).
//
// packed_mode: when true, data_buf layout is active pages placed consecutively
//              at offsets k*page_size (k = packed_start + 0,1,...,num_pages-1),
//              instead of at physical page positions (page_rel - output_page_offset).
//              This is used by Q3x to avoid the "sparse active pages" anomaly.
static size_t lo_worker_read_column(
    LoWorkerCtx &w,
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const uint32_t *page_indices, size_t num_pages,
    char *data_buf, size_t output_page_offset,
    int buf_idx,
    bool packed_mode = false, size_t packed_start = 0)
{
    if (num_pages == 0) return 0;

    const bool is_compressed = (field.compression_method != CompressionMethod::NONE);
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    char *stg = static_cast<char *>(w.staging_buf[buf_idx]);

    // Compute per-device data sizes and cumulative offsets
    size_t dev_data_size[LO_MAX_DEVICES] = {};
    for (size_t k = 0; k < num_pages; k++) {
        size_t page_rel = page_indices[k];
        uint64_t page_id = field.start_page_id + page_rel;
        size_t d = page_id % num_devices;
        if (is_compressed)
            dev_data_size[d] += roundup4096(field.compressed_page_sizes[page_rel]);
        else
            dev_data_size[d] += page_size;
    }
    size_t dev_io_start[LO_MAX_DEVICES + 1] = {};
    for (size_t d = 0; d < num_devices; d++)
        dev_io_start[d + 1] = dev_io_start[d] + dev_data_size[d];

    // Coalesced cuFileRead per device
    for (size_t d = 0; d < num_devices; d++) {
        if (dev_data_size[d] == 0) continue;
        char *dev_base = stg + dev_io_start[d];
        size_t io_buf_off = 0;
        size_t run_file_offset = 0, run_io_start = 0, run_size = 0;
        bool in_run = false;

        for (size_t k = 0; k < num_pages; k++) {
            size_t page_rel = page_indices[k];
            uint64_t page_id = field.start_page_id + page_rel;
            if (page_id % num_devices != d) continue;

            size_t this_disk_size, this_file_offset;
            if (is_compressed) {
                this_disk_size = roundup4096(field.compressed_page_sizes[page_rel]);
                this_file_offset = field.compressed_offsets[page_rel];
            } else {
                this_disk_size = page_size;
                this_file_offset = (page_id / num_devices) * page_size;
            }

            if (in_run && this_file_offset == run_file_offset + run_size) {
                run_size += this_disk_size;
            } else {
                if (in_run) {
                    off_t buf_offset = (dev_base + run_io_start) - stg;
                    ssize_t nread = cuFileRead(w.cufile_handles[d],
                        w.staging_buf[buf_idx], run_size, run_file_offset, buf_offset);
                    if (nread < 0 || (size_t)nread != run_size)
                        std::cerr << "cuFileRead fail: dev=" << d << " size=" << run_size << " nread=" << nread << std::endl;
                    else { w.bytes_read += nread; w.ios_completed++; }
                }
                run_file_offset = this_file_offset;
                run_io_start = io_buf_off;
                run_size = this_disk_size;
                in_run = true;
            }
            io_buf_off += this_disk_size;
        }
        if (in_run) {
            off_t buf_offset = (dev_base + run_io_start) - stg;
            ssize_t nread = cuFileRead(w.cufile_handles[d],
                w.staging_buf[buf_idx], run_size, run_file_offset, buf_offset);
            if (nread < 0 || (size_t)nread != run_size)
                std::cerr << "cuFileRead fail: dev=" << d << " size=" << run_size << " nread=" << nread << std::endl;
            else { w.bytes_read += nread; w.ios_completed++; }
        }
    }

    // Scatter: setup nvCOMP pointers or D→D copy
    // Use h_set = buf_idx to avoid overwriting pinned host arrays
    // while a previous async H2D copy is still in-flight.
    void  **h_cp  = (buf_idx == 0) ? w.nvctx.h_comp_ptrs   : w.nvctx.h_comp_ptrs_1;
    void  **h_dp  = (buf_idx == 0) ? w.nvctx.h_decomp_ptrs : w.nvctx.h_decomp_ptrs_1;
    size_t *h_cs  = (buf_idx == 0) ? w.nvctx.h_comp_sizes  : w.nvctx.h_comp_sizes_1;
    size_t *h_ds  = (buf_idx == 0) ? w.nvctx.h_decomp_sizes: w.nvctx.h_decomp_sizes_1;

    size_t decomp_count = 0;
    size_t dev_off[LO_MAX_DEVICES] = {};
    for (size_t k = 0; k < num_pages; k++) {
        size_t page_rel = page_indices[k];
        uint64_t page_id = field.start_page_id + page_rel;
        size_t d = page_id % num_devices;

        char *io_src = stg + dev_io_start[d] + dev_off[d];
        char *dst = packed_mode
            ? data_buf + (packed_start + k) * page_size
            : data_buf + (page_rel - output_page_offset) * page_size;

        if (is_compressed) {
            size_t cs = field.compressed_page_sizes[page_rel];
            if (cs < page_size) {
                h_cp[decomp_count]  = io_src;
                h_cs[decomp_count]  = cs;
                h_dp[decomp_count]  = dst;
                h_ds[decomp_count]  = page_size;
                decomp_count++;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                    cudaMemcpyDeviceToDevice, w.io_stream));
            }
            dev_off[d] += roundup4096(cs);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                cudaMemcpyDeviceToDevice, w.io_stream));
            dev_off[d] += page_size;
        }
    }

    return decomp_count;
}

// Process one tile for this worker: read all columns with double-buffered IO-decomp.
static void lo_worker_process_tile(
    LoWorkerCtx &w,
    const FieldPageInfo *lo_page_infos, size_t num_cols,
    size_t page_size, size_t num_devices,
    const uint32_t *page_indices, size_t num_pages,
    void **tile_data_bufs,
    size_t output_page_offset,
    CUcontext cuda_ctx,
    bool packed_mode = false, size_t packed_start = 0)
{
    mb_cuda_set_context(cuda_ctx);

    int buf = 0;
    bool submitted[2] = {false, false};

    for (size_t fi = 0; fi < num_cols; fi++) {
        // Wait for staging_buf[buf] to be available
        if (submitted[buf])
            CUDA_CHECK(cudaEventSynchronize(w.buf_done[buf]));

        // IO: read column fi pages → staging_buf[buf]
        size_t decomp_count = lo_worker_read_column(
            w, lo_page_infos[fi], page_size, num_devices,
            page_indices, num_pages,
            static_cast<char *>(tile_data_bufs[fi]),
            output_page_offset, buf,
            packed_mode, packed_start);

        // Decomp: launch async on decomp_stream
        if (decomp_count > 0) {
            CUDA_CHECK(cudaEventRecord(w.io_done, w.io_stream));
            CUDA_CHECK(cudaStreamWaitEvent(w.decomp_stream, w.io_done));
            nvcomp_decompctx_run(lo_page_infos[fi].compression_method,
                w.nvctx, decomp_count, page_size, w.decomp_stream,
                /*do_sync=*/false, /*skip_h2d=*/false, /*h_set=*/buf);
            w.kernel_launches++;
        }

        // Record completion on the appropriate stream
        cudaStream_t done_stream = (decomp_count > 0) ? w.decomp_stream : w.io_stream;
        CUDA_CHECK(cudaEventRecord(w.buf_done[buf], done_stream));
        submitted[buf] = true;
        buf ^= 1;
    }

    // Wait for all columns to complete
    for (int b = 0; b < 2; b++)
        if (submitted[b])
            CUDA_CHECK(cudaEventSynchronize(w.buf_done[b]));
}

// ============================================================
// Zone map helper: read sideways stats for a given LINEORDER field + sideways field.
// Returns Stats<int32_t>* array (caller must free), or nullptr if not available.
// ============================================================
static Stats<int32_t> *ssb_read_sideways_stats(
    std::vector<int> &fds,
    const SSBTableMetadata &metadata,
    size_t lo_field_idx,
    size_t sideways_idx,
    uint64_t &out_nstats)
{
    uint64_t nstats = metadata.table_lineorder_sideways_nstats[lo_field_idx][sideways_idx];
    uint64_t stats_start = metadata.table_lineorder_sideways_stats_start_page_ids[lo_field_idx][sideways_idx];
    uint64_t stats_npg = metadata.table_lineorder_sideways_stats_npages[lo_field_idx][sideways_idx];
    out_nstats = nstats;

    if (nstats == 0 || stats_start == 0 || stats_npg == 0) return nullptr;

    void *buf = nullptr;
    if (posix_memalign(&buf, 512, stats_npg * metadata.page_size) != 0) return nullptr;
    for (uint64_t j = 0; j < stats_npg; j++)
        page_pread_host(fds, static_cast<char *>(buf) + j * metadata.page_size,
                        stats_start + j, metadata.page_size);
    return reinterpret_cast<Stats<int32_t> *>(buf);
}

// ============================================================
// Zone map helper: read column stats (non-sideways) for a given LINEORDER field.
// ============================================================
static Stats<int32_t> *ssb_read_column_stats(
    std::vector<int> &fds,
    const SSBTableMetadata &metadata,
    size_t lo_field_idx,
    uint64_t &out_nstats)
{
    uint64_t nstats = metadata.table_lineorder_nstats[lo_field_idx];
    uint64_t stats_start = metadata.table_lineorder_stats_start_page_ids[lo_field_idx];
    uint64_t stats_npg = metadata.table_lineorder_stats_npages[lo_field_idx];
    out_nstats = nstats;

    if (nstats == 0 || stats_start == 0 || stats_npg == 0) return nullptr;

    void *buf = nullptr;
    if (posix_memalign(&buf, 512, stats_npg * metadata.page_size) != 0) return nullptr;
    for (uint64_t j = 0; j < stats_npg; j++)
        page_pread_host(fds, static_cast<char *>(buf) + j * metadata.page_size,
                        stats_start + j, metadata.page_size);
    return reinterpret_cast<Stats<int32_t> *>(buf);
}

// ============================================================
// Align field size to 4 bytes (matches loader's alignment=4 for CHAR fields)
static constexpr size_t align4(size_t sz) { return (sz + 3) & ~(size_t)3; }

// CPU-side hash / GPU HT infrastructure (used by Q1x–Q4x)
// ============================================================

// CPU-side hash function (must match GPU version)
static uint32_t ssb_hash32_host(uint32_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

struct GpuHT {
    int32_t *d_keys = nullptr;
    int32_t *d_values = nullptr;
    uint32_t mask = 0;
    uint32_t count = 0;
    void free_all() {
        if (d_keys) cudaFree(d_keys);
        if (d_values) cudaFree(d_values);
        d_keys = d_values = nullptr;
    }
};

// Pre-allocate GPU HT buffers (no I/O). Call BEFORE timing.
static GpuHT alloc_gpu_ht(uint32_t max_nrows) {
    uint32_t sz = 1;
    while (sz < max_nrows * 2) sz <<= 1;
    if (sz < 64) sz = 64;
    GpuHT ht;
    ht.mask = sz - 1;
    ht.count = 0;
    CUDA_CHECK(cudaMalloc(&ht.d_keys, sz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ht.d_values, sz * sizeof(int32_t)));
    return ht;
}

// Upload host-built HT arrays to pre-allocated GPU HT.
// Frees h_keys/h_values after upload.
static void upload_gpu_ht(GpuHT &ht, int32_t *h_keys, int32_t *h_values, uint32_t count) {
    uint32_t sz = ht.mask + 1;
    ht.count = count;
    CUDA_CHECK(cudaMemcpy(ht.d_keys, h_keys, sz * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ht.d_values, h_values, sz * sizeof(int32_t), cudaMemcpyHostToDevice));
    free(h_keys);
    free(h_values);
}

// Legacy: allocate + upload in one call (for callers outside timing).
static GpuHT make_gpu_ht(int32_t *h_keys, int32_t *h_values, uint32_t mask, size_t count) {
    uint32_t sz = mask + 1;
    GpuHT ht;
    ht.mask = mask;
    ht.count = static_cast<uint32_t>(count);
    CUDA_CHECK(cudaMalloc(&ht.d_keys, sz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ht.d_values, sz * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(ht.d_keys, h_keys, sz * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ht.d_values, h_values, sz * sizeof(int32_t), cudaMemcpyHostToDevice));
    free(h_keys);
    free(h_values);
    return ht;
}

// ============================================================
// Forward declarations: GPU dim build infrastructure
// (full definitions follow after Q1x)
// ============================================================

struct GdsDimFieldMeta {
    size_t start_page = 0;
    size_t npages = 0;
    CompressionMethod comp_method = CompressionMethod::NONE;
    std::vector<uint32_t> comp_page_sizes;
    std::vector<size_t> file_offsets;
};

struct DimGpuBufs {
    char *d_buf_a;
    char *d_buf_b;
    char *d_buf_c;
    uint64_t *d_prefix_sum;
    int32_t *d_flat_keys;
    uint8_t *d_filter;
    int32_t *d_values;
    DimGpuDict dict;
    size_t max_nrows;
    char     *h_dict_strs;
    uint16_t *h_dict_lens;
    uint32_t *h_dict_ids;
};

static DimGpuBufs dim_gpu_bufs_alloc(size_t page_size, size_t max_nrows, size_t max_dim_pages);
static void dim_gpu_bufs_free(DimGpuBufs &b);
static uint32_t dim_field_key(SSB::common::Table table, size_t field_idx);
static GdsDimFieldMeta gds_precache_dim_field(
    std::vector<CUfileHandle_t> &dim_cufile_handles,
    const SSBTableMetadata &metadata,
    SSB::common::Table table, size_t field_idx,
    size_t page_size, size_t num_devices,
    void *gds_buf, size_t gds_buf_npages);
struct DimReadScratch {
    size_t *pg_indices;
    size_t *buf_offsets;
    size_t max_pages;
    size_t bytes_read;
};
static DimReadScratch dim_read_scratch_alloc(size_t max_pages);
static void dim_read_scratch_free(DimReadScratch &s);
static void gds_build_date_ht_ext_gpu(
    GpuHT &ht,
    const std::map<uint32_t, GdsDimFieldMeta> &meta_cache,
    std::vector<CUfileHandle_t> &handles, size_t num_devices, size_t page_size,
    void **per_dev_bufs, size_t per_dev_buf_npages,
    NvcompDecompCtx &nvctx, cudaStream_t stream, DimGpuBufs &db,
    int32_t filter_mode, int32_t filter_lo, int32_t filter_hi,
    DimReadScratch &scratch,
    CUcontext cuda_ctx);

// ============================================================
// SSB Q1.x GIDP implementation (tile-based streaming)
// ============================================================
static BenchmarkResult ssb_q1x_gidp(
    BenchmarkOptions &options,
    SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    SSBTableMetadata *metadatap;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    {
        metadatap = reinterpret_cast<SSBTableMetadata *>(ptr);
        SSBTableMetadata &metadata_pre = *metadatap;
        std::cout << "=== SSB Table Metadata ===" << std::endl;
        std::cout << "Page Size: " << metadata_pre.page_size << std::endl;
        const size_t page_size = metadata_pre.page_size;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, page_size) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, page_size);
    }

    metadatap = reinterpret_cast<SSBTableMetadata *>(ptr);
    SSBTableMetadata &metadata = *metadatap;
    superpage_set_constants_for(metadata.page_size, sizeof(SSBTableMetadata));

    SSB::metadata_print(metadata);

    mb_cufile_driver_open();
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);

    // Prepare LINEORDER field metadata
    constexpr size_t num_lo_cols = SSB::query::q1x::NUM_LO_ACTIVE_FIELDS;
    auto q1x_lo_cols = SSB::query::q1x::LO_FIELDS;
    std::vector<FieldPageInfo> lo_page_infos(num_lo_cols);
    ssb_prepare_fields_metadata(
        fds, metadata, SSB::common::Table::LINEORDER,
        metadata.page_size, q1x_lo_cols, lo_page_infos);

    size_t min_npages = SIZE_MAX;
    for (size_t i = 0; i < num_lo_cols; i++) {
        min_npages = std::min(lo_page_infos[i].npages, min_npages);
    }

    for (size_t fi = 0; fi < num_lo_cols; fi++) {
        const FieldPageInfo &info = lo_page_infos[fi];
        std::cout << "  LO Field " << info.field_index
                  << ": start_page=" << info.start_page_id
                  << " npages=" << info.npages
                  << " compression=" << static_cast<int>(info.compression_method)
                  << std::endl;
    }

    if (min_npages == 0) {
        std::cout << "No pages to read." << std::endl;
        free_fields_metadata(lo_page_infos);
        free(ptr);
        for (int fd : fds) close(fd);
        return BenchmarkResult{};
    }

    bool any_compressed = false;
    for (size_t fi = 0; fi < num_lo_cols; fi++) {
        if (lo_page_infos[fi].compression_method != CompressionMethod::NONE) {
            any_compressed = true;
            break;
        }
    }

    const size_t page_size = metadata.page_size;
    const size_t nthreads = options.nthreads;
    const size_t num_devices = fds.size();

    // ── Dim cuFile handles (separate from LO workers) ──
    std::vector<CUfileHandle_t> dim_cufile_handles(num_devices);
    std::vector<int> dim_dup_fds(num_devices);
    for (size_t d = 0; d < num_devices; d++) {
        dim_dup_fds[d] = dup(fds[d]);
        dim_cufile_handles[d] = mb_cufile_handle_register(dim_dup_fds[d]);
    }

    // ── Pre-allocate GPU HT (Rule 4) ──
    GpuHT date_ht = alloc_gpu_ht(metadata.table_date_nrows);

    // ── Dim field page budget ──
    uint32_t max_dim_pages = 128;
    {
        auto field_max = [&](const uint64_t *np, const std::vector<uint32_t> &fs) {
            for (uint32_t fi : fs)
                max_dim_pages = std::max(max_dim_pages, (uint32_t)np[fi]);
        };
        field_max(metadata.table_date_npages,
            {SSB::common::D_DATEKEY, SSB::common::D_YEAR,
             SSB::common::D_YEARMONTHNUM, SSB::common::D_WEEKNUMINYEAR});
    }

    // Dim IO buffers (Rule 4)
    void *per_dev_bufs[num_devices];
    for (size_t d = 0; d < num_devices; d++) {
        per_dev_bufs[d] = mb_cuda_alloc(max_dim_pages * page_size);
        GDS_CHECK(cuFileBufRegister(per_dev_bufs[d], max_dim_pages * page_size, 0));
    }

    // ── Pre-cache dim metadata (Rule 3) ──
    using DT = SSB::common::Table;
    std::map<uint32_t, GdsDimFieldMeta> dim_meta;
    {
        using DF = SSB::common::DateField;
        const std::vector<size_t> q1x_date_fields = {
            DF::D_DATEKEY, DF::D_YEAR, DF::D_YEARMONTHNUM, DF::D_WEEKNUMINYEAR};
        for (auto fi : q1x_date_fields)
            dim_meta[dim_field_key(DT::DDATE, fi)] = gds_precache_dim_field(
                dim_cufile_handles, metadata, DT::DDATE, fi, page_size, num_devices,
                per_dev_bufs[0], max_dim_pages);
    }
    DimGpuBufs dim_bufs = dim_gpu_bufs_alloc(page_size, metadata.table_date_nrows, max_dim_pages);

    // Dim nvCOMP context (Rule 4)
    NvcompDecompCtx dim_nvctx{};
    {
        size_t mb = max_dim_pages;
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_decomp_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_actual_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_statuses,     mb * sizeof(nvcompStatus_t)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_decomp_sizes, mb * sizeof(size_t)));
        size_t max_total = mb * page_size;
        size_t snappy_tmp = 0, lz4_tmp = 0;
        nvcompBatchedSnappyDecompressGetTempSizeAsync(
            mb, page_size, nvcompBatchedSnappyDecompressDefaultOpts, &snappy_tmp, max_total);
        nvcompBatchedLZ4DecompressGetTempSizeAsync(
            mb, page_size, nvcompBatchedLZ4DecompressDefaultOpts, &lz4_tmp, max_total);
        dim_nvctx.temp_bytes = std::max(snappy_tmp, lz4_tmp);
        if (dim_nvctx.temp_bytes > 0)
            CUDA_CHECK(cudaMalloc(&dim_nvctx.d_temp, dim_nvctx.temp_bytes));
    }
    cudaStream_t dim_stream;
    CUDA_CHECK(cudaStreamCreate(&dim_stream));
    DimReadScratch dim_scratch = dim_read_scratch_alloc(max_dim_pages);

    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    // ── Q1x DATE filter mode for GPU dim build ──
    int32_t date_fmode, date_flo, date_fhi;
    switch (query) {
        case SSB::Query::Q11:     date_fmode = 1; date_flo = 1993; date_fhi = 1993; break;
        case SSB::Query::Q12:     date_fmode = 2; date_flo = 199401; date_fhi = 0; break;
        case SSB::Query::Q13:     date_fmode = 3; date_flo = 1994; date_fhi = 6; break;
        case SSB::Query::REVENUE: date_fmode = 0; date_flo = 0; date_fhi = 0; break;
        default:                  date_fmode = 0; date_flo = 0; date_fhi = 0; break;
    }

    // Q1.x filter constants
    int32_t disc_lo, disc_hi, qty_lo, qty_hi;
    switch (query) {
        case SSB::Query::Q11:
            disc_lo = 1; disc_hi = 3; qty_lo = 0; qty_hi = 25;
            break;
        case SSB::Query::Q12:
            disc_lo = 4; disc_hi = 6; qty_lo = 26; qty_hi = 36;
            break;
        case SSB::Query::Q13:
            disc_lo = 5; disc_hi = 7; qty_lo = 26; qty_hi = 36;
            break;
        case SSB::Query::REVENUE:
            if (options.disable_other_filters) {
                disc_lo = 0; disc_hi = 99; qty_lo = 0;
                qty_hi = options.revenue_qt_max > 0 ? options.revenue_qt_max : 100;
            } else {
                disc_lo = 1; disc_hi = 3; qty_lo = 0; qty_hi = 25;
            }
            break;
        default:
            disc_lo = disc_hi = qty_lo = qty_hi = 0;
            break;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── GPU memory budget (40 GiB cap) ──
    constexpr size_t Q1X_TILE_PAGES_MAX = 7168;
    size_t Q1X_TILE_PAGES;
    {
        constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t app_fixed = gpu_free_start - gpu_free_now;
        uint64_t remaining = (GPU_MEM_BUDGET > app_fixed) ? GPU_MEM_BUDGET - app_fixed : 0;
        // Worker fixed costs: nvCOMP d_temp + staging rounding
        size_t trial_ppw = (Q1X_TILE_PAGES_MAX + nthreads - 1) / nthreads;
        size_t nvcomp_temp = any_compressed ?
            query_nvcomp_temp_size(trial_ppw, page_size, lo_page_infos) : 0;
        uint64_t worker_fixed = (uint64_t)nthreads * nvcomp_temp
                              + (uint64_t)(nthreads - 1) * 2 * page_size
                              + 64ULL * 1024 * 1024;
        remaining -= std::min(remaining, worker_fixed);
        size_t per_tp = (num_lo_cols + 2) * page_size;
        Q1X_TILE_PAGES = std::min(Q1X_TILE_PAGES_MAX,
                                   std::max((size_t)1, (size_t)(remaining / per_tp)));
        // Do not cap by min_npages: allocate full budget for consistent memory usage
    }

    // ── LO worker setup (nthreads workers × all columns, double-buffered) ──
    const size_t pages_per_worker = (Q1X_TILE_PAGES + nthreads - 1) / nthreads;

    LoWorkerCtx workers[nthreads];
    lo_worker_alloc(workers, nthreads, fds.data(), num_devices, page_size,
                    pages_per_worker, any_compressed, lo_page_infos, false);

    const size_t npages = min_npages;

    // Field index mapping for Q1x
    const size_t LO_ORDERDATE_IDX = 0;
    const size_t LO_QUANTITY_IDX = 1;
    const size_t LO_DISCOUNT_IDX = 2;
    const size_t LO_EXTPRICE_IDX = 3;

    void *tile_data_buf[num_lo_cols];
    for (size_t i = 0; i < num_lo_cols; i++)
        tile_data_buf[i] = mb_cuda_alloc(Q1X_TILE_PAGES * page_size);

    int64_t *d_revenue = static_cast<int64_t *>(mb_cuda_alloc(sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    // ── Zone map predicate constants (Q1x: LO_ORDERDATE) ──
    int32_t zm_dk_low, zm_dk_high;
    switch (query) {
        case SSB::Query::Q11: zm_dk_low = 19930101; zm_dk_high = 19931231; break;
        case SSB::Query::Q12: zm_dk_low = 19940101; zm_dk_high = 19940131; break;
        case SSB::Query::Q13: zm_dk_low = 19940201; zm_dk_high = 19940214; break;
        case SSB::Query::REVENUE: zm_dk_low = options.q6_sd_low; zm_dk_high = options.q6_sd_high; break;
        default: zm_dk_low = 19920101; zm_dk_high = 19981231; break;
    }

    // ── Zonemap GPU buffers (Rule 4: allocate outside timing) ──
    uint64_t odate_nstats = metadata.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
    uint64_t odate_stats_start = metadata.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
    uint64_t odate_stats_npg = metadata.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];

    void *d_zm_stats_buf = nullptr;
    uint8_t *d_zm_mask = nullptr;
    uint32_t *d_zm_active_ids = nullptr;
    uint32_t *d_zm_num_selected = nullptr;
    void *d_zm_cub_temp = nullptr;
    size_t zm_cub_temp_bytes = 0;
    uint32_t *h_zm_active_ids = nullptr;

    ZonemapPred *d_zm_preds = nullptr;

    if (options.enable_zonemap && odate_nstats > 0 && odate_stats_npg > 0) {
        CUDA_CHECK(cudaMalloc(&d_zm_stats_buf, odate_stats_npg * page_size));
        GDS_CHECK(cuFileBufRegister(d_zm_stats_buf, odate_stats_npg * page_size, 0));
    }
    CUDA_CHECK(cudaMalloc(&d_zm_mask, npages));
    CUDA_CHECK(cudaMalloc(&d_zm_active_ids, npages * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_zm_num_selected, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_zm_preds, kZonemapMaxPreds * sizeof(ZonemapPred)));
    zm_cub_temp_bytes = zonemap_compact_query_temp(npages);
    if (zm_cub_temp_bytes > 0)
        CUDA_CHECK(cudaMalloc(&d_zm_cub_temp, zm_cub_temp_bytes));
    h_zm_active_ids = static_cast<uint32_t *>(malloc(npages * sizeof(uint32_t)));

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // ════════════ START TIMING ════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // ── Phase 0: GPU dim build (Rule 1: IO inside timing, Rule 2: parallel IO) ──
    gds_build_date_ht_ext_gpu(date_ht, dim_meta,
        dim_cufile_handles, num_devices, page_size,
        per_dev_bufs, max_dim_pages,
        dim_nvctx, dim_stream, dim_bufs,
        date_fmode, date_flo, date_fhi, dim_scratch, cuda_ctx_handle);

    // ── Zonemap pruning on GPU (Rule 6: IO + mask inside timing, GPU computation) ──
    size_t total_active_pages = npages;
    if (options.enable_zonemap && odate_nstats > 0 && odate_stats_npg > 0) {
        // GDS read zonemap stats → GPU
        gds_read_zonemap(dim_cufile_handles.data(), num_devices,
            odate_stats_start, odate_stats_npg, page_size, d_zm_stats_buf);

        // GPU: evaluate predicate (single kernel, replaces init_mask + eval_range)
        ZonemapPred h_preds[1] = {{
            reinterpret_cast<Stats<int32_t>*>(d_zm_stats_buf),
            odate_nstats, zm_dk_low, zm_dk_high
        }};
        zonemap_eval_preds(npages, h_preds, 1, d_zm_preds, d_zm_mask, stream);
        s_kernel_launches++;

        // GPU: compact mask → active page ID list
        zonemap_compact_active(d_zm_mask, npages, d_zm_active_ids,
            d_zm_num_selected, d_zm_cub_temp, zm_cub_temp_bytes, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // D2H: active list + count
        uint32_t h_num_selected = 0;
        CUDA_CHECK(cudaMemcpy(&h_num_selected, d_zm_num_selected,
            sizeof(uint32_t), cudaMemcpyDeviceToHost));
        total_active_pages = h_num_selected;
        CUDA_CHECK(cudaMemcpy(h_zm_active_ids, d_zm_active_ids,
            total_active_pages * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        std::cout << "[ZONEMAP] LO_ORDERDATE pruning: " << total_active_pages
                  << " / " << npages << " pages active ("
                  << (npages - total_active_pages) << " pruned)" << std::endl;
    } else {
        // No pruning: fill sequential IDs on host
        for (size_t pg = 0; pg < npages; pg++)
            h_zm_active_ids[pg] = static_cast<uint32_t>(pg);
    }

    bool use_selective = options.enable_zonemap && total_active_pages < npages;

    size_t num_tiles = (npages + Q1X_TILE_PAGES - 1) / Q1X_TILE_PAGES;
    std::cout << "[Q1x] Tile execution: " << num_tiles << " tiles of "
              << Q1X_TILE_PAGES << " pages" << std::endl;

    size_t active_cursor = 0;
    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        size_t p_lo = tile_idx * Q1X_TILE_PAGES;
        size_t tile_np = std::min(Q1X_TILE_PAGES, npages - p_lo);

        // Advance cursor to this tile's range in h_zm_active_ids (sorted, O(1) amortized)
        while (active_cursor < total_active_pages && h_zm_active_ids[active_cursor] < (uint32_t)p_lo)
            active_cursor++;
        size_t tile_active_start = active_cursor;
        while (active_cursor < total_active_pages && h_zm_active_ids[active_cursor] < (uint32_t)(p_lo + tile_np))
            active_cursor++;
        size_t n_active = active_cursor - tile_active_start;
        if (n_active == 0) continue;

        bool selective_partial = use_selective && n_active < tile_np;

        // Zero page headers for selective loading (inactive pages → nalloc=0)
        if (selective_partial) {
            unsigned zblk = (tile_np + 255) / 256;
            for (size_t fi = 0; fi < num_lo_cols; fi++) {
                ssb_zero_page_headers_kernel<<<zblk, 256, 0, stream>>>(
                    static_cast<char *>(tile_data_buf[fi]), tile_np, page_size);
                s_kernel_launches++;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Parallel IO + decomp using LoWorkerCtx (nthreads workers × all columns)
        // Workers get slices of h_zm_active_ids directly (no intermediate buffer)
        {
            size_t ppw = (n_active + nthreads - 1) / nthreads;
            std::thread thr_buf[nthreads];
            size_t n_thr = 0;
            for (size_t t = 0; t < nthreads; t++) {
                size_t w_start = t * ppw;
                if (w_start >= n_active) break;
                size_t w_count = std::min(ppw, n_active - w_start);
                thr_buf[n_thr++] = std::thread([&, t, w_start, w_count, p_lo, tile_active_start]() {
                    lo_worker_process_tile(
                        workers[t], lo_page_infos.data(), num_lo_cols,
                        page_size, num_devices,
                        h_zm_active_ids + tile_active_start + w_start, w_count,
                        tile_data_buf, p_lo, cuda_ctx_handle);
                });
            }
            for (size_t i = 0; i < n_thr; i++) thr_buf[i].join();
        }

        // Kernel
        uint32_t capacity = (page_size - 12) / 4;
        ssb_q1x(
            tile_data_buf[LO_ORDERDATE_IDX],
            tile_data_buf[LO_QUANTITY_IDX],
            tile_data_buf[LO_DISCOUNT_IDX],
            tile_data_buf[LO_EXTPRICE_IDX],
            tile_np, page_size,
            (uint64_t)tile_np * capacity,
            date_ht.d_keys, date_ht.d_values, date_ht.mask,
            disc_lo, disc_hi, qty_lo, qty_hi,
            d_revenue, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    int64_t h_revenue = 0;
    CUDA_CHECK(cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::cout << "SSB Q1.x revenue: " << h_revenue << std::endl;

    // ════════════ END TIMING ════════════
    auto total_end = std::chrono::steady_clock::now();

    size_t nios = 0, total_bytes_read = 0;
    uint64_t total_kernel_launches = s_kernel_launches;
    for (size_t t = 0; t < nthreads; t++) {
        nios += workers[t].ios_completed;
        total_bytes_read += workers[t].bytes_read;
        total_kernel_launches += workers[t].kernel_launches;
    }
    total_bytes_read += dim_scratch.bytes_read;

    std::cout << "\n========================================"
              << "\nTotal elapsed: "
              << std::chrono::duration<double>(total_end - total_start).count()
              << " seconds\nTotal I/Os: " << nios
              << "\nTotal bytes read: " << total_bytes_read
              << "\n========================================" << std::endl;

    // Cleanup
    date_ht.free_all();
    for (size_t i = 0; i < num_lo_cols; i++)
        mb_cuda_free(tile_data_buf[i]);
    mb_cuda_free(d_revenue);

    // GDS dim handles + GPU dim build cleanup
    for (size_t d = 0; d < num_devices; d++) {
        cuFileHandleDeregister(dim_cufile_handles[d]);
        close(dim_dup_fds[d]);
    }
    for (size_t d = 0; d < num_devices; d++) {
        cuFileBufDeregister(per_dev_bufs[d]);
        mb_cuda_free(per_dev_bufs[d]);
    }
    dim_gpu_bufs_free(dim_bufs);
    nvcomp_decompctx_free(dim_nvctx);
    dim_read_scratch_free(dim_scratch);
    CUDA_CHECK(cudaStreamDestroy(dim_stream));

    CUDA_CHECK(cudaStreamDestroy(stream));

    size_t total_pages = 0;
    for (const auto &fi : lo_page_infos) total_pages += fi.npages;

    lo_worker_free(workers, nthreads, num_devices, any_compressed);

    // Zonemap GPU buffer cleanup
    if (d_zm_stats_buf) {
        cuFileBufDeregister(d_zm_stats_buf);
        CUDA_CHECK(cudaFree(d_zm_stats_buf));
    }
    CUDA_CHECK(cudaFree(d_zm_mask));
    CUDA_CHECK(cudaFree(d_zm_active_ids));
    CUDA_CHECK(cudaFree(d_zm_num_selected));
    if (d_zm_cub_temp) CUDA_CHECK(cudaFree(d_zm_cub_temp));
    CUDA_CHECK(cudaFree(d_zm_preds));
    free(h_zm_active_ids);

    free_fields_metadata(lo_page_infos);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = nios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = collect_compression_methods({lo_page_infos}),
        .gpu_mem_bytes = gpu_mem_bytes,
        .gpu_app_bytes = gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = total_kernel_launches,
    };
}

// ============================================================
// GPU dim table build infrastructure (shared by Q2x/Q3x/Q4x)
// ============================================================

// (DimGpuBufs struct defined above Q1x — forward declaration section)

static DimGpuBufs dim_gpu_bufs_alloc(size_t page_size, size_t max_nrows,
                                      size_t max_dim_pages) {
    DimGpuBufs b{};
    CUDA_CHECK(cudaMalloc(&b.d_buf_a, max_dim_pages * page_size));
    CUDA_CHECK(cudaMalloc(&b.d_buf_b, max_dim_pages * page_size));
    CUDA_CHECK(cudaMalloc(&b.d_buf_c, max_dim_pages * page_size));
    CUDA_CHECK(cudaMalloc(&b.d_prefix_sum, max_dim_pages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&b.d_flat_keys, max_nrows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&b.d_filter, max_nrows));
    CUDA_CHECK(cudaMalloc(&b.d_values, max_nrows * sizeof(int32_t)));
    b.dict.alloc();
    b.max_nrows = max_nrows;
    // Pre-allocate host dict download buffers (Rule 4)
    b.h_dict_strs = static_cast<char *>(malloc(DIM_DICT_CAP * DIM_DICT_MAX_STRLEN));
    b.h_dict_lens = static_cast<uint16_t *>(malloc(DIM_DICT_CAP * sizeof(uint16_t)));
    b.h_dict_ids  = static_cast<uint32_t *>(malloc(DIM_DICT_CAP * sizeof(uint32_t)));
    return b;
}

static void dim_gpu_bufs_free(DimGpuBufs &b) {
    if (b.d_buf_a) { cudaFree(b.d_buf_a); b.d_buf_a = nullptr; }
    if (b.d_buf_b) { cudaFree(b.d_buf_b); b.d_buf_b = nullptr; }
    if (b.d_buf_c) { cudaFree(b.d_buf_c); b.d_buf_c = nullptr; }
    if (b.d_prefix_sum) { cudaFree(b.d_prefix_sum); b.d_prefix_sum = nullptr; }
    if (b.d_flat_keys) { cudaFree(b.d_flat_keys); b.d_flat_keys = nullptr; }
    if (b.d_filter) { cudaFree(b.d_filter); b.d_filter = nullptr; }
    if (b.d_values) { cudaFree(b.d_values); b.d_values = nullptr; }
    b.dict.free_all();
    free(b.h_dict_strs); b.h_dict_strs = nullptr;
    free(b.h_dict_lens); b.h_dict_lens = nullptr;
    free(b.h_dict_ids);  b.h_dict_ids  = nullptr;
}

// Raw dict data (cudaMemcpy only, no string construction) — Rule 4 compliant
struct DimDictRaw {
    char     *h_strs;
    uint16_t *h_lens;
    uint32_t *h_ids;
    uint32_t n;
    const char *fallback;   // set when n==0 (need_dict=false case)
};

static DimDictRaw dim_dict_raw_alloc() {
    DimDictRaw r{};
    r.h_strs = (char *)malloc(DIM_DICT_CAP * DIM_DICT_MAX_STRLEN);
    r.h_lens = (uint16_t *)malloc(DIM_DICT_CAP * sizeof(uint16_t));
    r.h_ids  = (uint32_t *)malloc(DIM_DICT_CAP * sizeof(uint32_t));
    r.n = 0;
    r.fallback = nullptr;
    return r;
}

static void dim_dict_raw_free(DimDictRaw &r) {
    free(r.h_strs); r.h_strs = nullptr;
    free(r.h_lens); r.h_lens = nullptr;
    free(r.h_ids);  r.h_ids  = nullptr;
}

static void dim_download_dict_raw(const DimGpuDict &gd, DimDictRaw &out) {
    uint32_t n;
    cudaMemcpy(&n, gd.d_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    out.n = n;
    if (n == 0) return;
    cudaMemcpy(out.h_strs, gd.d_strs, DIM_DICT_CAP * DIM_DICT_MAX_STRLEN, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.h_lens, gd.d_lens, DIM_DICT_CAP * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.h_ids,  gd.d_type_ids, DIM_DICT_CAP * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

static std::vector<std::string> dim_build_dict_strings(const DimDictRaw &r) {
    if (r.n == 0) {
        if (r.fallback) return {r.fallback};
        return {};
    }
    std::vector<std::string> result(r.n);
    for (uint32_t slot = 0; slot < DIM_DICT_CAP; slot++) {
        if (r.h_ids[slot] != UINT32_MAX && r.h_ids[slot] < r.n)
            result[r.h_ids[slot]] = std::string(
                r.h_strs + slot * DIM_DICT_MAX_STRLEN, r.h_lens[slot]);
    }
    return result;
}

// Helper: run dim_char_filter_kernel with predicate setup
static void dim_run_char_filter(
    const char *d_pages, uint32_t npages, uint32_t page_size,
    uint32_t field_size, const uint64_t *d_prefix_sum,
    int32_t filter_mode, const char *const *preds, uint32_t n_preds,
    bool enable_dict, DimGpuDict *dict,
    const uint8_t *d_prefilter, uint8_t *d_filter, int32_t *d_values,
    cudaStream_t stream)
{
    DimCharFilterParams p{};
    p.pages = d_pages;
    p.page_size = page_size;
    p.npages = npages;
    p.field_size = field_size;
    p.aligned_field_size = (field_size + 3) & ~3u;
    p.d_prefix_sum = d_prefix_sum;
    p.filter_mode = filter_mode;
    p.n_preds = n_preds;
    for (uint32_t i = 0; i < n_preds && i < 4; i++) {
        size_t len = std::min(strlen(preds[i]), (size_t)DIM_DICT_MAX_STRLEN);
        p.pred_lens[i] = static_cast<uint32_t>(len);
        memcpy(p.pred_strs[i], preds[i], len);
    }
    p.enable_dict = enable_dict;
    if (dict) {
        p.d_dict_hashes   = dict->d_hashes;
        p.d_dict_strs     = dict->d_strs;
        p.d_dict_lens     = dict->d_lens;
        p.d_dict_type_ids = dict->d_type_ids;
        p.d_id_counter    = dict->d_counter;
    }
    p.d_prefilter = d_prefilter;
    p.d_filter = d_filter;
    p.d_values = d_values;
    dim_char_filter_kernel<<<npages, 256, 0, stream>>>(p);
    s_kernel_launches++;
}

// Helper: get npages for a dim table field
static uint32_t dim_npages(const SSBTableMetadata &m, SSB::common::Table table, size_t field) {
    switch (table) {
        case SSB::common::Table::DDATE:    return m.table_date_npages[field];
        case SSB::common::Table::CUSTOMER: return m.table_customer_npages[field];
        case SSB::common::Table::SUPPLIER: return m.table_supplier_npages[field];
        case SSB::common::Table::PART:     return m.table_part_npages[field];
        default: return 0;
    }
}

// ============================================================
// GDS dim field metadata + GPU read + GPU HT build
// ============================================================

// (GdsDimFieldMeta struct defined above Q1x — forward declaration section)

// Extract field metadata from SSBTableMetadata
static void gds_extract_field_meta(
    const SSBTableMetadata &metadata, SSB::common::Table table, size_t field_idx,
    size_t &out_start, size_t &out_npages, uint16_t &out_comp_raw,
    size_t &out_sizes_npages, size_t &out_sizes_start,
    size_t &out_nbase, size_t &out_base_start)
{
    switch (table) {
    case SSB::common::Table::DDATE:
        out_start = metadata.table_date_start_page_ids[field_idx];
        out_npages = metadata.table_date_npages[field_idx];
        out_comp_raw = metadata.table_date_compression_method[field_idx];
        out_sizes_npages = metadata.table_date_compressed_page_sizes_npages[field_idx];
        out_sizes_start = metadata.table_date_compressed_page_sizes_start_page_ids[field_idx];
        out_nbase = metadata.table_date_compression_nbases[field_idx];
        out_base_start = metadata.table_date_compression_base_start_page_ids[field_idx];
        break;
    case SSB::common::Table::CUSTOMER:
        out_start = metadata.table_customer_start_page_ids[field_idx];
        out_npages = metadata.table_customer_npages[field_idx];
        out_comp_raw = metadata.table_customer_compression_method[field_idx];
        out_sizes_npages = metadata.table_customer_compressed_page_sizes_npages[field_idx];
        out_sizes_start = metadata.table_customer_compressed_page_sizes_start_page_ids[field_idx];
        out_nbase = metadata.table_customer_compression_nbases[field_idx];
        out_base_start = metadata.table_customer_compression_base_start_page_ids[field_idx];
        break;
    case SSB::common::Table::SUPPLIER:
        out_start = metadata.table_supplier_start_page_ids[field_idx];
        out_npages = metadata.table_supplier_npages[field_idx];
        out_comp_raw = metadata.table_supplier_compression_method[field_idx];
        out_sizes_npages = metadata.table_supplier_compressed_page_sizes_npages[field_idx];
        out_sizes_start = metadata.table_supplier_compressed_page_sizes_start_page_ids[field_idx];
        out_nbase = metadata.table_supplier_compression_nbases[field_idx];
        out_base_start = metadata.table_supplier_compression_base_start_page_ids[field_idx];
        break;
    case SSB::common::Table::PART:
        out_start = metadata.table_part_start_page_ids[field_idx];
        out_npages = metadata.table_part_npages[field_idx];
        out_comp_raw = metadata.table_part_compression_method[field_idx];
        out_sizes_npages = metadata.table_part_compressed_page_sizes_npages[field_idx];
        out_sizes_start = metadata.table_part_compressed_page_sizes_start_page_ids[field_idx];
        out_nbase = metadata.table_part_compression_nbases[field_idx];
        out_base_start = metadata.table_part_compression_base_start_page_ids[field_idx];
        break;
    default: break;
    }
}

static uint32_t dim_field_key(SSB::common::Table table, size_t field_idx) {
    return (static_cast<uint32_t>(table) << 8) | static_cast<uint32_t>(field_idx);
}

// Pre-cache metadata for one dim field via GDS cuFileRead (Rule 3: before timing).
// For compressed: reads comp_page_sizes + base pages, computes file_offsets.
// For uncompressed: computes file_offsets from stripe layout.
static GdsDimFieldMeta gds_precache_dim_field(
    std::vector<CUfileHandle_t> &dim_cufile_handles,
    const SSBTableMetadata &metadata,
    SSB::common::Table table, size_t field_idx,
    size_t page_size, size_t num_devices,
    void *gds_buf, size_t gds_buf_npages)
{
    GdsDimFieldMeta meta;
    size_t sizes_npages = 0, sizes_start = 0, nbase = 0, base_start = 0;
    uint16_t comp_raw = 0;
    gds_extract_field_meta(metadata, table, field_idx,
        meta.start_page, meta.npages, comp_raw,
        sizes_npages, sizes_start, nbase, base_start);
    meta.comp_method = static_cast<CompressionMethod>(comp_raw);
    if (meta.npages == 0) return meta;

    if (meta.comp_method == CompressionMethod::NONE) {
        // Uncompressed: compute file offsets from stripe layout
        meta.file_offsets.resize(meta.npages);
        for (size_t pg = 0; pg < meta.npages; pg++) {
            size_t page_id = meta.start_page + pg;
            meta.file_offsets[pg] = (page_id / num_devices) * page_size;
        }
        return meta;
    }

    // Compressed: read comp_page_sizes metadata via GDS
    for (size_t pg = 0; pg < sizes_npages; pg++) {
        size_t page_id = sizes_start + pg;
        size_t dev = page_id % num_devices;
        size_t file_off = (page_id / num_devices) * page_size;
        cuFileRead(dim_cufile_handles[dev], gds_buf, page_size, file_off, pg * page_size);
    }
    void *sizes_host = malloc(sizes_npages * page_size);
    CUDA_CHECK(cudaMemcpy(sizes_host, gds_buf, sizes_npages * page_size, cudaMemcpyDeviceToHost));
    uint32_t *cps = reinterpret_cast<uint32_t *>(sizes_host);
    meta.comp_page_sizes.assign(cps, cps + meta.npages);

    // Read base page IDs
    size_t base_npages = SSB::nbase_to_npages(nbase, page_size);
    for (size_t pg = 0; pg < base_npages; pg++) {
        size_t page_id = base_start + pg;
        size_t dev = page_id % num_devices;
        size_t file_off = (page_id / num_devices) * page_size;
        cuFileRead(dim_cufile_handles[dev], gds_buf, page_size, file_off, pg * page_size);
    }
    void *bases_host = malloc(base_npages * page_size);
    CUDA_CHECK(cudaMemcpy(bases_host, gds_buf, base_npages * page_size, cudaMemcpyDeviceToHost));

    // Compute file offsets via calculate_compressed_offsets
    calculate_compressed_offsets(
        reinterpret_cast<size_t *>(bases_host),
        cps, nbase, meta.npages, page_size,
        meta.start_page, num_devices, meta.file_offsets);

    free(bases_host);
    free(sizes_host);
    return meta;
}

// ── Multi-threaded coalesced GDS read + GPU nvCOMP decomp ────
// Reads a dim field to GPU buffer using parallel cuFileRead per device,
// then GPU-side nvCOMP decompression. No D→H copies. (Rule 2)
static constexpr size_t DIM_MAX_DEVICES = 8;

// DimReadScratch defined above Q1x — forward declaration section

static DimReadScratch dim_read_scratch_alloc(size_t max_pages) {
    DimReadScratch s{};
    s.max_pages = max_pages;
    s.bytes_read = 0;
    s.pg_indices = static_cast<size_t *>(malloc(max_pages * sizeof(size_t)));
    s.buf_offsets = static_cast<size_t *>(malloc(max_pages * sizeof(size_t)));
    return s;
}

static void dim_read_scratch_free(DimReadScratch &s) {
    free(s.pg_indices);
    free(s.buf_offsets);
    s.pg_indices = nullptr;
    s.buf_offsets = nullptr;
}

static void gds_read_dim_field_to_gpu(
    const GdsDimFieldMeta &meta,
    std::vector<CUfileHandle_t> &dim_cufile_handles,
    size_t num_devices, size_t page_size,
    void **per_dev_bufs,  // per_dev_bufs[d] = registered buffer for device d
    size_t per_dev_buf_npages,
    NvcompDecompCtx &nvctx, cudaStream_t stream,
    char *d_out,
    DimReadScratch &scratch,
    CUcontext cuda_ctx = nullptr)
{
    if (meta.npages == 0) return;
    assert(num_devices <= DIM_MAX_DEVICES && meta.npages <= scratch.max_pages);
    auto roundup4096 = [](size_t v) -> size_t { return (v + 4095) & ~(size_t)4095; };

    // Group pages by device (pre-allocated scratch — Rule 4)
    size_t dev_pg_count[DIM_MAX_DEVICES] = {};
    size_t dev_pg_offset[DIM_MAX_DEVICES + 1] = {};
    for (size_t pg = 0; pg < meta.npages; pg++)
        dev_pg_count[(meta.start_page + pg) % num_devices]++;
    for (size_t d = 0; d < num_devices; d++)
        dev_pg_offset[d + 1] = dev_pg_offset[d] + dev_pg_count[d];

    size_t dev_fill[DIM_MAX_DEVICES] = {};
    for (size_t pg = 0; pg < meta.npages; pg++) {
        size_t dev = (meta.start_page + pg) % num_devices;
        scratch.pg_indices[dev_pg_offset[dev] + dev_fill[dev]++] = pg;
    }

    if (meta.comp_method == CompressionMethod::NONE) {
        // ── Uncompressed: parallel coalesced cuFileRead per device → scatter D→D ──
        std::thread io_threads[DIM_MAX_DEVICES];
        size_t n_io_threads = 0;
        for (size_t d = 0; d < num_devices; d++) {
            if (dev_pg_count[d] == 0) continue;
            io_threads[n_io_threads++] = std::thread([&, d]() {
                if (cuda_ctx) mb_cuda_set_context(cuda_ctx);
                size_t n = dev_pg_count[d];
                const size_t *pgs = &scratch.pg_indices[dev_pg_offset[d]];
                size_t first_file_off = meta.file_offsets[pgs[0]];
                ssize_t nread = cuFileRead(dim_cufile_handles[d],
                    per_dev_bufs[d], n * page_size, first_file_off, 0);
                if (nread < 0 || static_cast<size_t>(nread) != n * page_size) {
                    fprintf(stderr, "cuFileRead dim uncomp failed: dev=%zu nread=%zd errno=%d(%s)\n",
                        d, nread, errno, strerror(errno));
                }
            });
        }
        for (size_t i = 0; i < n_io_threads; i++) io_threads[i].join();
        for (size_t d = 0; d < num_devices; d++)
            scratch.bytes_read += dev_pg_count[d] * page_size;

        // Scatter: copy each page from per-device buffer to d_out in logical order
        for (size_t d = 0; d < num_devices; d++) {
            const size_t *pgs = &scratch.pg_indices[dev_pg_offset[d]];
            for (size_t i = 0; i < dev_pg_count[d]; i++) {
                CUDA_CHECK(cudaMemcpyAsync(
                    d_out + pgs[i] * page_size,
                    static_cast<char *>(per_dev_bufs[d]) + i * page_size,
                    page_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        // ── Compressed: parallel coalesced cuFileRead per device → nvCOMP GPU decomp ──

        // Compute per-device buffer layout (variable-size compressed pages)
        for (size_t d = 0; d < num_devices; d++) {
            const size_t *pgs = &scratch.pg_indices[dev_pg_offset[d]];
            size_t off = 0;
            for (size_t i = 0; i < dev_pg_count[d]; i++) {
                scratch.buf_offsets[dev_pg_offset[d] + i] = off;
                off += roundup4096(meta.comp_page_sizes[pgs[i]]);
            }
        }

        // Parallel cuFileRead: coalesce consecutive compressed pages per device
        std::thread io_threads[DIM_MAX_DEVICES];
        size_t n_io_threads = 0;
        for (size_t d = 0; d < num_devices; d++) {
            if (dev_pg_count[d] == 0) continue;
            io_threads[n_io_threads++] = std::thread([&, d]() {
                if (cuda_ctx) mb_cuda_set_context(cuda_ctx);
                const size_t *pgs = &scratch.pg_indices[dev_pg_offset[d]];
                const size_t *boff = &scratch.buf_offsets[dev_pg_offset[d]];
                size_t n = dev_pg_count[d];
                size_t i = 0;
                while (i < n) {
                    size_t run_start = i;
                    size_t run_file_off = meta.file_offsets[pgs[i]];
                    size_t run_size = roundup4096(meta.comp_page_sizes[pgs[i]]);
                    while (i + 1 < n &&
                           meta.file_offsets[pgs[i + 1]] == run_file_off + run_size) {
                        i++;
                        run_size += roundup4096(meta.comp_page_sizes[pgs[i]]);
                    }
                    ssize_t nread = cuFileRead(dim_cufile_handles[d],
                        per_dev_bufs[d], run_size, run_file_off, boff[run_start]);
                    if (nread < 0 || static_cast<size_t>(nread) != run_size) {
                        fprintf(stderr, "cuFileRead dim comp failed: dev=%zu nread=%zd "
                            "expected=%zu errno=%d(%s)\n",
                            d, nread, run_size, errno, strerror(errno));
                    }
                    i++;
                }
            });
        }
        for (size_t i = 0; i < n_io_threads; i++) io_threads[i].join();
        for (size_t pg = 0; pg < meta.npages; pg++)
            scratch.bytes_read += (meta.comp_page_sizes[pg] + 4095) & ~(size_t)4095;

        // Set nvCOMP pointers for batch decompression
        for (size_t d = 0; d < num_devices; d++) {
            const size_t *pgs = &scratch.pg_indices[dev_pg_offset[d]];
            const size_t *boff = &scratch.buf_offsets[dev_pg_offset[d]];
            for (size_t i = 0; i < dev_pg_count[d]; i++) {
                size_t pg = pgs[i];
                nvctx.h_comp_ptrs[pg]   = static_cast<char *>(per_dev_bufs[d]) + boff[i];
                nvctx.h_decomp_ptrs[pg] = d_out + pg * page_size;
                nvctx.h_comp_sizes[pg]  = meta.comp_page_sizes[pg];
                nvctx.h_decomp_sizes[pg] = page_size;
            }
        }
        nvcomp_decompctx_run(meta.comp_method, nvctx, meta.npages, page_size, stream);
        s_kernel_launches++;
    }
}

// ── GPU dim build: DATE HT (Q2x simple) ─────────────────────
static void gds_build_date_ht_gpu(
    GpuHT &ht, const GdsDimFieldMeta &dk_meta, const GdsDimFieldMeta &yr_meta,
    std::vector<CUfileHandle_t> &handles, size_t num_devices, size_t page_size,
    void **per_dev_bufs, size_t per_dev_buf_npages,
    NvcompDecompCtx &nvctx, cudaStream_t stream, DimGpuBufs &db,
    DimReadScratch &scratch,
    CUcontext cuda_ctx = nullptr)
{
    gds_read_dim_field_to_gpu(dk_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_a, scratch, cuda_ctx);
    gds_read_dim_field_to_gpu(yr_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_b, scratch, cuda_ctx);
    uint32_t np = dk_meta.npages;
    uint32_t cap = (page_size - 12) / sizeof(int32_t);
    CUDA_CHECK(cudaMemsetAsync(ht.d_keys, 0xFF, (ht.mask + 1) * sizeof(int32_t), stream));
    dim_build_date_ht_kernel<<<np, 256, 0, stream>>>(
        db.d_buf_a, db.d_buf_b, np, page_size, cap,
        SSB_YEAR_MIN, ht.d_keys, ht.d_values, ht.mask);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ── GPU dim build: Q2x SUPPLIER + PART ──────────────────────
static void gds_build_dim_q2x_gpu(
    GpuHT &supp_ht, GpuHT &part_ht,
    const std::map<uint32_t, GdsDimFieldMeta> &meta_cache,
    const SSBTableMetadata &metadata,
    std::vector<CUfileHandle_t> &handles, size_t num_devices, size_t page_size,
    void **per_dev_bufs, size_t per_dev_buf_npages,
    NvcompDecompCtx &nvctx, cudaStream_t stream,
    SSB::Query query, DimDictRaw &brand1_dict_raw,
    DimGpuBufs &db, const char *region,
    DimReadScratch &scratch,
    CUcontext cuda_ctx = nullptr)
{
    using SF = SSB::common::SupplierField;
    using PF = SSB::common::PartField;
    auto &m = metadata;

    // ── SUPPLIER: read + filtered HT build ──
    auto &sk_meta = meta_cache.at(dim_field_key(SSB::common::Table::SUPPLIER, SF::S_SUPPKEY));
    auto &sr_meta = meta_cache.at(dim_field_key(SSB::common::Table::SUPPLIER, SF::S_REGION));
    gds_read_dim_field_to_gpu(sk_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_a, scratch, cuda_ctx);
    gds_read_dim_field_to_gpu(sr_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_b, scratch, cuda_ctx);
    // 2-step: filter on S_REGION pages, then HT build on S_SUPPKEY pages
    // (fields have different per-page capacities at small page sizes like 64K)
    {
        const char *region_pred = region;
        uint32_t region_np = sr_meta.npages;
        dim_run_char_filter(
            db.d_buf_b, region_np, page_size, SSB::common::S_REGION_SIZE,
            nullptr, DIM_FILT_PREFIX, &region_pred, 1,
            false, nullptr,
            nullptr, db.d_filter, nullptr, stream);

        uint32_t key_np = sk_meta.npages;
        uint32_t key_cap = (page_size - 12) / sizeof(int32_t);
        CUDA_CHECK(cudaMemsetAsync(supp_ht.d_keys, 0xFF,
            (supp_ht.mask + 1) * sizeof(int32_t), stream));
        dim_build_ht_paged_kernel<<<key_np, 256, 0, stream>>>(
            db.d_buf_a, key_np, page_size, key_cap,
            db.d_filter, nullptr,
            supp_ht.d_keys, supp_ht.d_values, supp_ht.mask);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // ── PART: filter + dict + flatten + HT build ──
    auto &pb_meta = meta_cache.at(dim_field_key(SSB::common::Table::PART, PF::P_BRAND1));
    auto &pk_meta = meta_cache.at(dim_field_key(SSB::common::Table::PART, PF::P_PARTKEY));
    gds_read_dim_field_to_gpu(pb_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_a, scratch, cuda_ctx);
    gds_read_dim_field_to_gpu(pk_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_b, scratch, cuda_ctx);

    uint32_t pb_np = pb_meta.npages;
    dim_extract_prefix_sum_kernel<<<1, 1, 0, stream>>>(db.d_buf_a, page_size, pb_np, db.d_prefix_sum);
    s_kernel_launches++;
    db.dict.reset();
    int32_t filt_mode;
    bool need_dict;
    const char *preds[2];
    uint32_t n_preds;
    if (query == SSB::Query::Q21) {
        filt_mode = DIM_FILT_PREFIX; preds[0] = "MFGR#12"; n_preds = 1; need_dict = true;
    } else if (query == SSB::Query::Q22) {
        filt_mode = DIM_FILT_RANGE; preds[0] = "MFGR#2221"; preds[1] = "MFGR#2228"; n_preds = 2; need_dict = true;
    } else {
        filt_mode = DIM_FILT_EQ; preds[0] = "MFGR#2221"; n_preds = 1; need_dict = false;
    }
    dim_run_char_filter(
        db.d_buf_a, pb_np, page_size, SSB::common::P_BRAND1_SIZE,
        db.d_prefix_sum, filt_mode, preds, n_preds,
        need_dict, need_dict ? &db.dict : nullptr,
        nullptr, db.d_filter, need_dict ? db.d_values : nullptr, stream);

    // Flatten P_PARTKEY + HT build
    uint64_t nrows = m.table_part_nrows;
    uint32_t pk_np = pk_meta.npages;
    dim_extract_prefix_sum_kernel<<<1, 1, 0, stream>>>(db.d_buf_b, page_size, pk_np, db.d_prefix_sum);
    s_kernel_launches++;
    uint32_t grid = (nrows + 255) / 256;
    dim_flatten_int32_kernel<<<grid, 256, 0, stream>>>(
        db.d_buf_b, page_size, db.d_prefix_sum, pk_np, nrows, db.d_flat_keys);
    s_kernel_launches++;
    CUDA_CHECK(cudaMemsetAsync(part_ht.d_keys, 0xFF,
        (part_ht.mask + 1) * sizeof(int32_t), stream));
    dim_build_ht_flat_kernel<<<grid, 256, 0, stream>>>(
        db.d_flat_keys, db.d_filter, need_dict ? db.d_values : nullptr,
        nrows, part_ht.d_keys, part_ht.d_values, part_ht.mask);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Download dict raw (Rule 4: cudaMemcpy only, no string construction)
    if (need_dict) {
        dim_download_dict_raw(db.dict, brand1_dict_raw);
    } else {
        brand1_dict_raw.n = 0;
        brand1_dict_raw.fallback = preds[0];
    }
}

// ── GPU dim build: Extended DATE HT (Q3x/Q4x with filter modes) ──
static void gds_build_date_ht_ext_gpu(
    GpuHT &ht,
    const std::map<uint32_t, GdsDimFieldMeta> &meta_cache,
    std::vector<CUfileHandle_t> &handles, size_t num_devices, size_t page_size,
    void **per_dev_bufs, size_t per_dev_buf_npages,
    NvcompDecompCtx &nvctx, cudaStream_t stream, DimGpuBufs &db,
    int32_t filter_mode, int32_t filter_lo, int32_t filter_hi,
    DimReadScratch &scratch,
    CUcontext cuda_ctx = nullptr)
{
    using F = SSB::common::DateField;
    auto &dk_meta = meta_cache.at(dim_field_key(SSB::common::Table::DDATE, F::D_DATEKEY));
    auto &yr_meta = meta_cache.at(dim_field_key(SSB::common::Table::DDATE, F::D_YEAR));

    gds_read_dim_field_to_gpu(dk_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_a, scratch, cuda_ctx);
    gds_read_dim_field_to_gpu(yr_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_b, scratch, cuda_ctx);

    // For mode 2 (yearmonthnum) or mode 3 (year+weeknuminyear), read aux
    if (filter_mode == 2) {
        auto &aux_meta = meta_cache.at(dim_field_key(SSB::common::Table::DDATE, F::D_YEARMONTHNUM));
        gds_read_dim_field_to_gpu(aux_meta, handles, num_devices, page_size,
            per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_c, scratch, cuda_ctx);
    } else if (filter_mode == 3) {
        auto &aux_meta = meta_cache.at(dim_field_key(SSB::common::Table::DDATE, F::D_WEEKNUMINYEAR));
        gds_read_dim_field_to_gpu(aux_meta, handles, num_devices, page_size,
            per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_c, scratch, cuda_ctx);
    }

    uint32_t np = dk_meta.npages;
    uint32_t cap = (page_size - 12) / sizeof(int32_t);
    CUDA_CHECK(cudaMemsetAsync(ht.d_keys, 0xFF, (ht.mask + 1) * sizeof(int32_t), stream));
    dim_build_date_ht_ext_kernel<<<np, 256, 0, stream>>>(
        db.d_buf_a, db.d_buf_b,
        (filter_mode >= 2) ? db.d_buf_c : nullptr,
        np, page_size, cap,
        SSB_YEAR_MIN, filter_mode, filter_lo, filter_hi,
        ht.d_keys, ht.d_values, ht.mask);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ── GPU dim build: Generic table HT (Q3x/Q4x CUSTOMER/SUPPLIER/PART) ──
static void gds_build_table_ht_gpu(
    GpuHT &ht,
    const std::map<uint32_t, GdsDimFieldMeta> &meta_cache,
    const SSBTableMetadata &metadata,
    std::vector<CUfileHandle_t> &handles, size_t num_devices, size_t page_size,
    void **per_dev_bufs, size_t per_dev_buf_npages,
    NvcompDecompCtx &nvctx, cudaStream_t stream, DimGpuBufs &db,
    SSB::common::Table table,
    size_t key_field, size_t filter_field, uint32_t filter_field_size,
    int32_t filter_mode, const char *const *filter_preds, uint32_t n_filter_preds,
    bool has_separate_group, size_t group_field, uint32_t group_field_size,
    bool need_dict, uint64_t nrows,
    DimDictRaw &dim_dict_raw,
    DimReadScratch &scratch,
    CUcontext cuda_ctx = nullptr)
{
    // Read key + filter field
    auto &key_meta = meta_cache.at(dim_field_key(table, key_field));
    auto &filter_meta = meta_cache.at(dim_field_key(table, filter_field));
    gds_read_dim_field_to_gpu(key_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_a, scratch, cuda_ctx);
    gds_read_dim_field_to_gpu(filter_meta, handles, num_devices, page_size,
        per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_b, scratch, cuda_ctx);

    // If 3-field: read group field
    if (has_separate_group) {
        auto &group_meta = meta_cache.at(dim_field_key(table, group_field));
        gds_read_dim_field_to_gpu(group_meta, handles, num_devices, page_size,
            per_dev_bufs, per_dev_buf_npages, nvctx, stream, db.d_buf_c, scratch, cuda_ctx);
    }

    uint32_t filter_np = dim_npages(metadata, table, filter_field);
    dim_extract_prefix_sum_kernel<<<1, 1, 0, stream>>>(db.d_buf_b, page_size, filter_np, db.d_prefix_sum);
    s_kernel_launches++;

    if (!has_separate_group) {
        // 2-field: filter + dict in one pass on filter field
        db.dict.reset();
        dim_run_char_filter(
            db.d_buf_b, filter_np, page_size, filter_field_size,
            db.d_prefix_sum, filter_mode, filter_preds, n_filter_preds,
            need_dict, need_dict ? &db.dict : nullptr,
            nullptr, db.d_filter, need_dict ? db.d_values : nullptr, stream);
    } else {
        // 3-field: pass 1 — filter only
        dim_run_char_filter(
            db.d_buf_b, filter_np, page_size, filter_field_size,
            db.d_prefix_sum, filter_mode, filter_preds, n_filter_preds,
            false, nullptr,
            nullptr, db.d_filter, nullptr, stream);

        // Process group field with prefilter
        uint32_t group_np = dim_npages(metadata, table, group_field);
        dim_extract_prefix_sum_kernel<<<1, 1, 0, stream>>>(db.d_buf_c, page_size, group_np, db.d_prefix_sum);
        s_kernel_launches++;
        db.dict.reset();
        dim_run_char_filter(
            db.d_buf_c, group_np, page_size, group_field_size,
            db.d_prefix_sum, DIM_FILT_NONE, nullptr, 0,
            need_dict, need_dict ? &db.dict : nullptr,
            db.d_filter, nullptr, need_dict ? db.d_values : nullptr, stream);
    }

    // Flatten key pages → d_flat_keys
    uint32_t key_np = dim_npages(metadata, table, key_field);
    dim_extract_prefix_sum_kernel<<<1, 1, 0, stream>>>(db.d_buf_a, page_size, key_np, db.d_prefix_sum);
    s_kernel_launches++;
    uint32_t grid = (nrows + 255) / 256;
    dim_flatten_int32_kernel<<<grid, 256, 0, stream>>>(
        db.d_buf_a, page_size, db.d_prefix_sum, key_np, nrows, db.d_flat_keys);
    s_kernel_launches++;

    // HT build
    CUDA_CHECK(cudaMemsetAsync(ht.d_keys, 0xFF, (ht.mask + 1) * sizeof(int32_t), stream));
    dim_build_ht_flat_kernel<<<grid, 256, 0, stream>>>(
        db.d_flat_keys, db.d_filter, need_dict ? db.d_values : nullptr,
        nrows, ht.d_keys, ht.d_values, ht.mask);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Download dict raw (Rule 4: cudaMemcpy only, no string construction)
    if (need_dict) {
        dim_download_dict_raw(db.dict, dim_dict_raw);
    } else {
        dim_dict_raw.n = 0;
        dim_dict_raw.fallback = "_";
    }
}

// ============================================================
// SSB Q2.x GIDP implementation (shared staging + flatten + flat array)
// ============================================================
static BenchmarkResult ssb_q2x_gidp(
    BenchmarkOptions &options,
    SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    SSBTableMetadata *metadatap;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    {
        metadatap = reinterpret_cast<SSBTableMetadata *>(ptr);
        SSBTableMetadata &metadata_pre = *metadatap;
        std::cout << "=== SSB Table Metadata ===" << std::endl;
        std::cout << "Page Size: " << metadata_pre.page_size << std::endl;
        const size_t page_size = metadata_pre.page_size;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, page_size) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, page_size);
    }

    metadatap = reinterpret_cast<SSBTableMetadata *>(ptr);
    SSBTableMetadata &metadata = *metadatap;
    superpage_set_constants_for(metadata.page_size, sizeof(SSBTableMetadata));

    SSB::metadata_print(metadata);

    const size_t page_size = metadata.page_size;
    const size_t nthreads = options.nthreads;
    const size_t num_devices = fds.size();

    // ── Query parameters ──
    const char *supp_region = nullptr;
    switch (query) {
    case SSB::Query::Q21:
        supp_region = "AMERICA"; break;
    case SSB::Query::Q22:
        supp_region = "ASIA"; break;
    case SSB::Query::Q23:
        supp_region = "EUROPE"; break;
    default:
        std::cerr << "Invalid Q2x query" << std::endl; exit(EXIT_FAILURE);
    }

    // ── Open GDS driver and stream ──
    mb_cufile_driver_open();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Pre-allocate GPU buffers for dimension HTs (over-sized, based on nrows) ──
    // DATE HT
    // Group-by output buffer
    size_t group_buf_size = SSB_Q2X_GROUPS * sizeof(int64_t);
    int64_t *d_revenue;
    CUDA_CHECK(cudaMalloc(&d_revenue, group_buf_size));

    // ── cuFileHandles + dedicated GDS buffer for dimension table reads ──
    // Must be allocated BEFORE loaders to avoid GPU BAR exhaustion.
    std::vector<CUfileHandle_t> dim_cufile_handles(num_devices);
    std::vector<int> dim_dup_fds(num_devices);
    for (size_t d = 0; d < num_devices; d++) {
        dim_dup_fds[d] = dup(fds[d]);
        dim_cufile_handles[d] = mb_cufile_handle_register(dim_dup_fds[d]);
    }
    // per_dev_bufs: each device d gets its own registered buffer for parallel cuFileRead
    // — constructed after loader init (see below)

    // ── Prepare LINEORDER field metadata ──
    constexpr size_t num_lo_cols = SSB::query::q2x::NUM_LO_ACTIVE_FIELDS;
    auto q2x_lo_cols = SSB::query::q2x::LO_FIELDS;
    std::vector<FieldPageInfo> lo_page_infos(num_lo_cols);
    ssb_prepare_fields_metadata(
        fds, metadata, SSB::common::Table::LINEORDER,
        metadata.page_size, q2x_lo_cols, lo_page_infos);

    size_t min_npages = SIZE_MAX;
    for (size_t i = 0; i < num_lo_cols; i++) {
        min_npages = std::min(lo_page_infos[i].npages, min_npages);
    }

    for (size_t fi = 0; fi < num_lo_cols; fi++) {
        const FieldPageInfo &info = lo_page_infos[fi];
        std::cout << "  LO Field " << info.field_index
                  << ": start_page=" << info.start_page_id
                  << " npages=" << info.npages
                  << " compression=" << static_cast<int>(info.compression_method)
                  << std::endl;
    }

    if (min_npages == 0) {
        std::cout << "No pages to read." << std::endl;
        free_fields_metadata(lo_page_infos);
        free(ptr);
        for (int fd : fds) close(fd);
        return BenchmarkResult{};
    }

    bool any_compressed = false;
    for (size_t fi = 0; fi < num_lo_cols; fi++) {
        if (lo_page_infos[fi].compression_method != CompressionMethod::NONE)
            any_compressed = true;
    }

    // ── Pre-allocate GPU HTs (Rule 4: alloc outside timing) ──
    GpuHT date_ht = alloc_gpu_ht(metadata.table_date_nrows);
    GpuHT supp_ht = alloc_gpu_ht(metadata.table_supplier_nrows);
    GpuHT part_ht = alloc_gpu_ht(metadata.table_part_nrows);

    // ── Pre-allocate GPU dim build buffers (Rule 4) ──
    uint32_t max_dim_pages = 128;
    {
        auto field_max = [&](const uint64_t *np, const std::vector<uint32_t> &fs) {
            for (uint32_t fi : fs)
                max_dim_pages = std::max(max_dim_pages, (uint32_t)np[fi]);
        };
        field_max(metadata.table_date_npages,
            {SSB::common::D_DATEKEY, SSB::common::D_YEAR});
        field_max(metadata.table_supplier_npages,
            {SSB::common::S_SUPPKEY, SSB::common::S_REGION});
        field_max(metadata.table_part_npages,
            {SSB::common::P_PARTKEY, SSB::common::P_BRAND1});
    }

    // Dim IO buffers: one per device for parallel dim reads (Rule 4)
    void *per_dev_bufs[num_devices];
    for (size_t d = 0; d < num_devices; d++) {
        per_dev_bufs[d] = mb_cuda_alloc(max_dim_pages * page_size);
        GDS_CHECK(cuFileBufRegister(per_dev_bufs[d], max_dim_pages * page_size, 0));
    }

    // ── Pre-cache dim field metadata (Rule 3: metadata outside timing) ──
    using DT = SSB::common::Table;
    std::map<uint32_t, GdsDimFieldMeta> dim_meta;
    {
        const std::pair<DT, std::vector<size_t>> q2x_dim_fields[] = {
            {DT::DDATE, {SSB::common::D_DATEKEY, SSB::common::D_YEAR}},
            {DT::SUPPLIER, {SSB::common::S_SUPPKEY, SSB::common::S_REGION}},
            {DT::PART, {SSB::common::P_PARTKEY, SSB::common::P_BRAND1}},
        };
        for (auto &[tbl, fields] : q2x_dim_fields)
            for (auto fi : fields)
                dim_meta[dim_field_key(tbl, fi)] = gds_precache_dim_field(
                    dim_cufile_handles, metadata, tbl, fi, page_size, num_devices,
                    per_dev_bufs[0], max_dim_pages);
    }
    size_t max_dim_nrows = std::max({metadata.table_date_nrows,
        metadata.table_supplier_nrows, metadata.table_part_nrows});
    DimGpuBufs dim_bufs = dim_gpu_bufs_alloc(page_size, max_dim_nrows, max_dim_pages);

    // Dim nvCOMP context (Rule 4)
    NvcompDecompCtx dim_nvctx{};
    {
        size_t mb = max_dim_pages;
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_decomp_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_actual_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_statuses,     mb * sizeof(nvcompStatus_t)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_decomp_sizes, mb * sizeof(size_t)));
        size_t max_total = mb * page_size;
        size_t snappy_tmp = 0, lz4_tmp = 0;
        nvcompBatchedSnappyDecompressGetTempSizeAsync(
            mb, page_size, nvcompBatchedSnappyDecompressDefaultOpts, &snappy_tmp, max_total);
        nvcompBatchedLZ4DecompressGetTempSizeAsync(
            mb, page_size, nvcompBatchedLZ4DecompressDefaultOpts, &lz4_tmp, max_total);
        dim_nvctx.temp_bytes = std::max(snappy_tmp, lz4_tmp);
        if (dim_nvctx.temp_bytes > 0)
            CUDA_CHECK(cudaMalloc(&dim_nvctx.d_temp, dim_nvctx.temp_bytes));
    }
    cudaStream_t dim_stream;
    CUDA_CHECK(cudaStreamCreate(&dim_stream));
    DimReadScratch dim_scratch = dim_read_scratch_alloc(max_dim_pages);

    // ── GPU memory budget (40 GiB cap) ──
    constexpr size_t Q2X_TILE_PAGES_MAX = 3072;
    size_t Q2X_TILE_PAGES;
    {
        constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t app_fixed = gpu_free_start - gpu_free_now;
        uint64_t remaining = (GPU_MEM_BUDGET > app_fixed) ? GPU_MEM_BUDGET - app_fixed : 0;
        // Worker fixed costs: nvCOMP d_temp + staging rounding
        size_t trial_ppw = (Q2X_TILE_PAGES_MAX + nthreads - 1) / nthreads;
        size_t nvcomp_temp = any_compressed ?
            query_nvcomp_temp_size(trial_ppw, page_size, lo_page_infos) : 0;
        uint64_t worker_fixed = (uint64_t)nthreads * nvcomp_temp
                              + (uint64_t)(nthreads - 1) * 2 * page_size
                              + 64ULL * 1024 * 1024;
        remaining -= std::min(remaining, worker_fixed);
        uint32_t cap_est = (page_size - 12) / sizeof(int32_t);
        size_t per_tp = num_lo_cols * page_size
                      + num_lo_cols * cap_est * sizeof(uint64_t)
                      + sizeof(uint64_t)
                      + 2 * page_size;
        Q2X_TILE_PAGES = std::min(Q2X_TILE_PAGES_MAX,
                                   std::max((size_t)1, (size_t)(remaining / per_tp)));
        // Do not cap by min_npages: allocate full budget for consistent memory usage
    }

    // ── LO worker setup (nthreads workers × all columns, double-buffered) ──
    const size_t pages_per_worker = (Q2X_TILE_PAGES + nthreads - 1) / nthreads;

    LoWorkerCtx workers[nthreads];
    lo_worker_alloc(workers, nthreads, fds.data(), num_devices, page_size,
                    pages_per_worker, any_compressed, lo_page_infos, false);

    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    const size_t npages = min_npages;

    // Compute max tile nrows for flat array allocation
    uint64_t tile_nrows_max = 0;
    for (size_t tile_idx = 0; tile_idx < (npages + Q2X_TILE_PAGES - 1) / Q2X_TILE_PAGES; tile_idx++) {
        size_t p_lo = tile_idx * Q2X_TILE_PAGES;
        size_t tile_np = std::min(Q2X_TILE_PAGES, npages - p_lo);
        if (lo_page_infos[0].prefix_sum_nrecs) {
            uint64_t nr = lo_page_infos[0].prefix_sum_nrecs[p_lo + tile_np]
                        - lo_page_infos[0].prefix_sum_nrecs[p_lo];
            tile_nrows_max = std::max(tile_nrows_max, nr);
        }
    }
    if (tile_nrows_max == 0) {
        uint32_t capacity = (page_size - 12) / sizeof(int32_t);
        tile_nrows_max = Q2X_TILE_PAGES * capacity;
    }

    // Per-column staging buffers
    void *tile_data_buf[num_lo_cols];
    for (size_t i = 0; i < num_lo_cols; i++)
        tile_data_buf[i] = mb_cuda_alloc(Q2X_TILE_PAGES * page_size);

    // Per-column flat arrays
    uint64_t *d_flat[num_lo_cols];
    for (size_t i = 0; i < num_lo_cols; i++)
        CUDA_CHECK(cudaMalloc(&d_flat[i], tile_nrows_max * sizeof(uint64_t)));

    // Prefix sum shared buffer
    uint64_t *d_ps_shared;
    CUDA_CHECK(cudaMalloc(&d_ps_shared, Q2X_TILE_PAGES * sizeof(uint64_t)));

    // Pre-allocate h_tile_ps for flatten lambda (Rule 4)
    uint64_t *h_tile_ps = static_cast<uint64_t *>(malloc(Q2X_TILE_PAGES * sizeof(uint64_t)));

    // Pre-uploaded prefix_sum GPU arrays (Rule 3: populated before total_start)
    std::unordered_map<const uint64_t*, uint64_t*> d_full_ps_gpu;
    // GPU page-active mask for selective_partial prefix_sum (Rule 3)
    uint8_t *d_page_active_mask = nullptr;
    // Per-page nrecs pre-computed from prefix_sum (Rule 3: avoid metadata reads in measurement)
    std::vector<uint64_t> h_per_page_nrecs;

    // ── Flatten helper (h_tile_ps pre-allocated before timing, Rule 4) ──
    auto flatten_int32_field_tile = [&](const FieldPageInfo &fi, void *data_buf,
                                         uint64_t tile_nrows, uint64_t *d_out,
                                         size_t tile_start, size_t tile_npages,
                                         bool selective_partial_mode = false) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            if (selective_partial_mode && d_page_active_mask) {
                compute_selective_batch_ps_kernel<<<1, 1, 0, stream>>>(
                    it->second, d_ps_shared, d_page_active_mask,
                    (uint32_t)tile_start, (uint32_t)tile_npages);
                s_kernel_launches++;
            } else {
                compute_batch_ps_kernel<<<((uint32_t)tile_npages + 255) / 256, 256, 0, stream>>>(
                    it->second, d_ps_shared, (uint32_t)tile_start, (uint32_t)tile_npages);
                s_kernel_launches++;
            }
        } else {
#if 0
            uint64_t cum = 0;
            for (size_t j = 0; j < tile_npages; j++) {
                uint32_t nalloc = 0;
                CUDA_CHECK(cudaMemcpy(&nalloc,
                    static_cast<const char *>(data_buf) + j * page_size,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cum += nalloc;
                h_tile_ps[j] = cum;
            }
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps,
                                  tile_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        ssb_flatten_int32_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, tile_npages, tile_nrows, d_out, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // ── Zonemap GPU buffers (Rule 4: allocate outside timing) ──
    const size_t zm_ref_field = SSB::common::LO_ORDERDATE;

    int32_t zm_sr_pred = -1;
    size_t zm_sr_sw_idx = SSB::common::LSS_S_REGION;
    uint64_t sr_nstats = metadata.table_lineorder_sideways_nstats[zm_ref_field][zm_sr_sw_idx];
    uint64_t sr_stats_start = metadata.table_lineorder_sideways_stats_start_page_ids[zm_ref_field][zm_sr_sw_idx];
    uint64_t sr_stats_npg = metadata.table_lineorder_sideways_stats_npages[zm_ref_field][zm_sr_sw_idx];

    int32_t zm_part_lo = -1, zm_part_hi = -1;
    size_t zm_part_sw_idx = 0;
    uint64_t part_nstats = 0, part_stats_start = 0, part_stats_npg = 0;

    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);

        auto it_sr = dict_maps[SSB::common::LSS_S_REGION].find(std::string(supp_region));
        if (it_sr != dict_maps[SSB::common::LSS_S_REGION].end()) zm_sr_pred = it_sr->second;

        if (query == SSB::Query::Q21) {
            zm_part_sw_idx = SSB::common::LSS_P_CATEGORY;
            auto it = dict_maps[zm_part_sw_idx].find("MFGR#12");
            if (it != dict_maps[zm_part_sw_idx].end()) { zm_part_lo = zm_part_hi = it->second; }
        } else {
            zm_part_sw_idx = SSB::common::LSS_P_BRAND1;
            auto it_lo_b = dict_maps[zm_part_sw_idx].find(
                query == SSB::Query::Q22 ? "MFGR#2221" : "MFGR#2221");
            auto it_hi_b = dict_maps[zm_part_sw_idx].find(
                query == SSB::Query::Q22 ? "MFGR#2228" : "MFGR#2221");
            if (it_lo_b != dict_maps[zm_part_sw_idx].end()) zm_part_lo = it_lo_b->second;
            if (it_hi_b != dict_maps[zm_part_sw_idx].end()) zm_part_hi = it_hi_b->second;
        }
        part_nstats = metadata.table_lineorder_sideways_nstats[zm_ref_field][zm_part_sw_idx];
        part_stats_start = metadata.table_lineorder_sideways_stats_start_page_ids[zm_ref_field][zm_part_sw_idx];
        part_stats_npg = metadata.table_lineorder_sideways_stats_npages[zm_ref_field][zm_part_sw_idx];
    }

    void *d_zm_stats_sr = nullptr, *d_zm_stats_part = nullptr;
    uint8_t *d_zm_mask = nullptr;
    uint32_t *d_zm_active_ids = nullptr;
    uint32_t *d_zm_num_selected = nullptr;
    void *d_zm_cub_temp = nullptr;
    size_t zm_cub_temp_bytes = 0;
    ZonemapPred *d_zm_preds = nullptr;
    uint32_t *h_zm_active_ids = nullptr;

    if (options.enable_zonemap) {
        if (sr_nstats > 0 && sr_stats_npg > 0) {
            CUDA_CHECK(cudaMalloc(&d_zm_stats_sr, sr_stats_npg * page_size));
            GDS_CHECK(cuFileBufRegister(d_zm_stats_sr, sr_stats_npg * page_size, 0));
        }
        if (part_nstats > 0 && part_stats_npg > 0) {
            CUDA_CHECK(cudaMalloc(&d_zm_stats_part, part_stats_npg * page_size));
            GDS_CHECK(cuFileBufRegister(d_zm_stats_part, part_stats_npg * page_size, 0));
        }
    }
    CUDA_CHECK(cudaMalloc(&d_zm_mask, npages));
    CUDA_CHECK(cudaMalloc(&d_zm_active_ids, npages * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_zm_num_selected, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_zm_preds, kZonemapMaxPreds * sizeof(ZonemapPred)));
    zm_cub_temp_bytes = zonemap_compact_query_temp(npages);
    if (zm_cub_temp_bytes > 0)
        CUDA_CHECK(cudaMalloc(&d_zm_cub_temp, zm_cub_temp_bytes));
    h_zm_active_ids = static_cast<uint32_t *>(malloc(npages * sizeof(uint32_t)));

    // Pre-allocate brand1_dict raw (Rule 4: alloc outside timing)
    DimDictRaw brand1_dict_raw = dim_dict_raw_alloc();

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    CUDA_CHECK(cudaMemset(d_revenue, 0, group_buf_size));

    // Pre-upload LO field prefix_sums to GPU (Rule 3: before total_start)
    {
        auto upload_ps = [&](const FieldPageInfo &fi) {
            if (!fi.prefix_sum_nrecs || d_full_ps_gpu.count(fi.prefix_sum_nrecs)) return;
            uint64_t *d_ps = nullptr;
            CUDA_CHECK(cudaMalloc(&d_ps, (fi.npages + 1) * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_ps, fi.prefix_sum_nrecs,
                                  (fi.npages + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            d_full_ps_gpu[fi.prefix_sum_nrecs] = d_ps;
        };
        for (size_t i = 0; i < num_lo_cols; i++) upload_ps(lo_page_infos[i]);
    }

    // Pre-compute per-page nrecs from prefix_sum (Rule 3: avoid metadata reads in measurement)
    if (lo_page_infos[0].prefix_sum_nrecs) {
        h_per_page_nrecs.resize(npages);
        for (size_t pg = 0; pg < npages; pg++)
            h_per_page_nrecs[pg] = lo_page_infos[0].prefix_sum_nrecs[pg + 1]
                                 - lo_page_infos[0].prefix_sum_nrecs[pg];
    }

    // ════════════ START TIMING ════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // ── Phase 0: GPU dim build (Rule 1: I/O inside timing, Rule 2: parallel IO) ──
    auto &dk_meta = dim_meta.at(dim_field_key(DT::DDATE, SSB::common::D_DATEKEY));
    auto &yr_meta = dim_meta.at(dim_field_key(DT::DDATE, SSB::common::D_YEAR));
    gds_build_date_ht_gpu(date_ht, dk_meta, yr_meta,
        dim_cufile_handles, num_devices, page_size,
        per_dev_bufs, max_dim_pages,
        dim_nvctx, dim_stream, dim_bufs, dim_scratch, cuda_ctx_handle);

    gds_build_dim_q2x_gpu(supp_ht, part_ht,
        dim_meta, metadata,
        dim_cufile_handles, num_devices, page_size,
        per_dev_bufs, max_dim_pages,
        dim_nvctx, dim_stream,
        query, brand1_dict_raw, dim_bufs, supp_region, dim_scratch, cuda_ctx_handle);

    std::cout << "[Q2x] Supplier matches: " << supp_ht.count
              << " (HT size=" << (supp_ht.mask + 1) << ")" << std::endl;

    size_t num_brands = brand1_dict_raw.n > 0 ? brand1_dict_raw.n : 1;
    std::cout << "[Q2x] Part matches: " << part_ht.count
              << " (HT size=" << (part_ht.mask + 1) << ")"
              << " brands=" << num_brands << std::endl;

    // ── Zonemap pruning on GPU (Rule 6: IO + eval inside timing) ──
    size_t total_active_pages = npages;
    if (options.enable_zonemap) {
        uint32_t npreds = 0;
        ZonemapPred h_preds[2];

        if (d_zm_stats_sr && zm_sr_pred >= 0) {
            gds_read_zonemap(dim_cufile_handles.data(), num_devices,
                sr_stats_start, sr_stats_npg, page_size, d_zm_stats_sr);
            h_preds[npreds++] = {reinterpret_cast<Stats<int32_t>*>(d_zm_stats_sr),
                sr_nstats, zm_sr_pred, zm_sr_pred};
        }
        if (d_zm_stats_part && zm_part_lo >= 0) {
            gds_read_zonemap(dim_cufile_handles.data(), num_devices,
                part_stats_start, part_stats_npg, page_size, d_zm_stats_part);
            h_preds[npreds++] = {reinterpret_cast<Stats<int32_t>*>(d_zm_stats_part),
                part_nstats, zm_part_lo, zm_part_hi};
        }

        if (npreds > 0) {
            zonemap_eval_preds(npages, h_preds, npreds, d_zm_preds, d_zm_mask, stream);
            s_kernel_launches++;

            zonemap_compact_active(d_zm_mask, npages, d_zm_active_ids,
                d_zm_num_selected, d_zm_cub_temp, zm_cub_temp_bytes, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));

            uint32_t h_num_selected = 0;
            CUDA_CHECK(cudaMemcpy(&h_num_selected, d_zm_num_selected,
                sizeof(uint32_t), cudaMemcpyDeviceToHost));
            total_active_pages = h_num_selected;
            CUDA_CHECK(cudaMemcpy(h_zm_active_ids, d_zm_active_ids,
                total_active_pages * sizeof(uint32_t), cudaMemcpyDeviceToHost));

            d_page_active_mask = d_zm_mask;
        } else {
            for (size_t pg = 0; pg < npages; pg++)
                h_zm_active_ids[pg] = static_cast<uint32_t>(pg);
        }

        std::cout << "[ZONEMAP] Q2x GPU pruning: " << total_active_pages
                  << " / " << npages << " pages active ("
                  << (npages - total_active_pages) << " pruned)" << std::endl;
    } else {
        for (size_t pg = 0; pg < npages; pg++)
            h_zm_active_ids[pg] = static_cast<uint32_t>(pg);
    }

    bool use_selective = options.enable_zonemap && total_active_pages < npages;

    // Q21: active-packed tile partitioning avoids the "sparse active pages"
    // anomaly where physical-range tiling causes extreme imbalance (e.g. SF300:
    // tile 0 gets 1 active page, tile 1 gets ~all, tile 2 gets 0).
    // Packed layout places active pages consecutively in data_buf.
    const bool use_active_packed = (query == SSB::Query::Q21);

    // Ensure d_zm_active_ids is populated on GPU when zonemap pruning did not
    // write it (zonemap disabled or no predicates). Packed mode reads this.
    if (use_active_packed &&
        !(options.enable_zonemap && total_active_pages < npages)) {
        CUDA_CHECK(cudaMemcpyAsync(d_zm_active_ids, h_zm_active_ids,
            total_active_pages * sizeof(uint32_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    size_t num_tiles;
    if (use_active_packed) {
        num_tiles = (total_active_pages + Q2X_TILE_PAGES - 1) / Q2X_TILE_PAGES;
        if (num_tiles == 0) num_tiles = 1;
        std::cout << "[Q2x] Tile execution (active-packed): " << num_tiles
                  << " tiles of up to " << Q2X_TILE_PAGES << " active pages"
                  << " (total_active=" << total_active_pages << ")" << std::endl;
    } else {
        num_tiles = (npages + Q2X_TILE_PAGES - 1) / Q2X_TILE_PAGES;
        std::cout << "[Q2x] Tile execution: " << num_tiles << " tiles of "
                  << Q2X_TILE_PAGES << " pages" << std::endl;
    }

    size_t active_cursor = 0;
    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        size_t p_lo, tile_np, tile_active_start, n_active;

        if (use_active_packed) {
            tile_active_start = tile_idx * Q2X_TILE_PAGES;
            n_active = std::min((size_t)Q2X_TILE_PAGES,
                                total_active_pages - tile_active_start);
            if (n_active == 0) continue;
            p_lo = h_zm_active_ids[tile_active_start];
            uint32_t last_pg = h_zm_active_ids[tile_active_start + n_active - 1];
            tile_np = (size_t)last_pg - p_lo + 1;
        } else {
            p_lo = tile_idx * Q2X_TILE_PAGES;
            tile_np = std::min(Q2X_TILE_PAGES, npages - p_lo);

            // Cursor-based walk through h_zm_active_ids (sorted, O(1) amortized)
            while (active_cursor < total_active_pages && h_zm_active_ids[active_cursor] < (uint32_t)p_lo)
                active_cursor++;
            tile_active_start = active_cursor;
            while (active_cursor < total_active_pages && h_zm_active_ids[active_cursor] < (uint32_t)(p_lo + tile_np))
                active_cursor++;
            n_active = active_cursor - tile_active_start;
            if (n_active == 0) continue;
        }

        bool selective_partial = !use_active_packed && use_selective && n_active < tile_np;

        // Compute tile nrows from pre-computed per-page nrecs
        uint64_t tile_nrows = 0;
        if (!h_per_page_nrecs.empty()) {
            for (size_t a = 0; a < n_active; a++)
                tile_nrows += h_per_page_nrecs[h_zm_active_ids[tile_active_start + a]];
        } else {
            uint32_t capacity = (page_size - 12) / sizeof(int32_t);
            tile_nrows = n_active * capacity;
        }
        if (tile_nrows == 0) continue;

        // Zero page headers for selective loading (inactive pages → nalloc=0)
        // Not needed in packed mode: active pages are placed consecutively.
        if (selective_partial) {
            unsigned zblk = (tile_np + 255) / 256;
            for (size_t fi = 0; fi < num_lo_cols; fi++) {
                ssb_zero_page_headers_kernel<<<zblk, 256, 0, stream>>>(
                    static_cast<char *>(tile_data_buf[fi]), tile_np, page_size);
                s_kernel_launches++;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Parallel IO + decomp using LoWorkerCtx (nthreads workers × all columns)
        {
            size_t ppw = (n_active + nthreads - 1) / nthreads;
            std::thread thr_buf[nthreads];
            size_t n_thr = 0;
            for (size_t t = 0; t < nthreads; t++) {
                size_t w_start = t * ppw;
                if (w_start >= n_active) break;
                size_t w_count = std::min(ppw, n_active - w_start);
                thr_buf[n_thr++] = std::thread([&, t, w_start, w_count, p_lo,
                                                  tile_active_start, use_active_packed]() {
                    lo_worker_process_tile(
                        workers[t], lo_page_infos.data(), num_lo_cols,
                        page_size, num_devices,
                        h_zm_active_ids + tile_active_start + w_start, w_count,
                        tile_data_buf, p_lo, cuda_ctx_handle,
                        /*packed_mode=*/use_active_packed,
                        /*packed_start=*/use_active_packed ? w_start : 0);
                });
            }
            for (size_t i = 0; i < n_thr; i++) thr_buf[i].join();
        }

        // Flatten all fields
        if (use_active_packed) {
            for (size_t fi = 0; fi < num_lo_cols; fi++) {
                auto it = d_full_ps_gpu.find(lo_page_infos[fi].prefix_sum_nrecs);
                if (it == d_full_ps_gpu.end()) {
                    std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
                    abort();
                }
                compute_packed_ps_from_active_ids_kernel<<<1, 1, 0, stream>>>(
                    it->second, d_ps_shared, d_zm_active_ids,
                    (uint32_t)tile_active_start, (uint32_t)n_active);
                s_kernel_launches++;
                ssb_flatten_int32_pages_ps(
                    static_cast<const char *>(tile_data_buf[fi]),
                    page_size, d_ps_shared,
                    (uint32_t)n_active, tile_nrows, d_flat[fi], stream);
                s_kernel_launches++;
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        } else {
            for (size_t fi = 0; fi < num_lo_cols; fi++) {
                flatten_int32_field_tile(lo_page_infos[fi], tile_data_buf[fi], tile_nrows,
                                         d_flat[fi], p_lo, tile_np, selective_partial);
            }
        }

        // Flat array kernel
        ssb_q2x_probe_flat(
            d_flat[0], d_flat[1], d_flat[2], d_flat[3],
            tile_nrows,
            date_ht.d_keys, date_ht.d_values, date_ht.mask,
            supp_ht.d_keys, supp_ht.d_values, supp_ht.mask,
            part_ht.d_keys, part_ht.d_values, part_ht.mask,
            d_revenue, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // ── Read results ──
    int64_t h_revenue[SSB_Q2X_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_revenue, d_revenue, group_buf_size, cudaMemcpyDeviceToHost));

    // ════════════ END TIMING ════════════
    auto total_end = std::chrono::steady_clock::now();

    // Build dict strings outside timing (Rule 4)
    std::vector<std::string> brand1_dict = dim_build_dict_strings(brand1_dict_raw);

    std::cout << "\nSSB Q2.x results:" << std::endl;
    size_t result_count = 0;
    for (int32_t y = 0; y < SSB_NUM_YEARS; y++) {
        for (size_t b = 0; b < num_brands; b++) {
            int64_t rev = h_revenue[y * SSB_MAX_BRANDS + b];
            if (rev != 0) {
                std::cout << "  " << rev
                          << " | " << (SSB_YEAR_MIN + y)
                          << " | " << brand1_dict[b] << std::endl;
                result_count++;
            }
        }
    }
    std::cout << "Total result rows: " << result_count << std::endl;

    size_t nios = 0, total_bytes_read = 0;
    uint64_t total_kernel_launches = s_kernel_launches;
    for (size_t t = 0; t < nthreads; t++) {
        nios += workers[t].ios_completed;
        total_bytes_read += workers[t].bytes_read;
        total_kernel_launches += workers[t].kernel_launches;
    }
    total_bytes_read += dim_scratch.bytes_read;

    std::cout << "\n========================================"
              << "\nTotal elapsed: "
              << std::chrono::duration<double>(total_end - total_start).count()
              << " seconds\nTotal I/Os: " << nios
              << "\nTotal bytes read: " << total_bytes_read
              << "\n========================================" << std::endl;

    // ── Cleanup ──
    date_ht.free_all();
    supp_ht.free_all();
    part_ht.free_all();
    dim_dict_raw_free(brand1_dict_raw);
    CUDA_CHECK(cudaFree(d_revenue));
    CUDA_CHECK(cudaFree(d_ps_shared));
    for (auto &[k, v] : d_full_ps_gpu) cudaFree(v);
    for (size_t i = 0; i < num_lo_cols; i++)
        CUDA_CHECK(cudaFree(d_flat[i]));

    for (size_t i = 0; i < num_lo_cols; i++)
        mb_cuda_free(tile_data_buf[i]);

    // GDS dim handles + GPU dim build cleanup
    for (size_t d = 0; d < num_devices; d++) {
        cuFileHandleDeregister(dim_cufile_handles[d]);
        close(dim_dup_fds[d]);
    }
    for (size_t d = 0; d < num_devices; d++) {
        cuFileBufDeregister(per_dev_bufs[d]);
        mb_cuda_free(per_dev_bufs[d]);
    }
    dim_gpu_bufs_free(dim_bufs);
    nvcomp_decompctx_free(dim_nvctx);
    dim_read_scratch_free(dim_scratch);
    CUDA_CHECK(cudaStreamDestroy(dim_stream));

    // Zonemap GPU buffer cleanup
    if (d_zm_stats_sr) {
        cuFileBufDeregister(d_zm_stats_sr);
        CUDA_CHECK(cudaFree(d_zm_stats_sr));
    }
    if (d_zm_stats_part) {
        cuFileBufDeregister(d_zm_stats_part);
        CUDA_CHECK(cudaFree(d_zm_stats_part));
    }
    CUDA_CHECK(cudaFree(d_zm_mask));
    CUDA_CHECK(cudaFree(d_zm_active_ids));
    CUDA_CHECK(cudaFree(d_zm_num_selected));
    if (d_zm_cub_temp) CUDA_CHECK(cudaFree(d_zm_cub_temp));
    CUDA_CHECK(cudaFree(d_zm_preds));
    free(h_zm_active_ids);

    CUDA_CHECK(cudaStreamDestroy(stream));

    size_t total_pages = 0;
    for (const auto &fi : lo_page_infos) total_pages += fi.npages;

    lo_worker_free(workers, nthreads, num_devices, any_compressed);
    free(h_tile_ps);

    free_fields_metadata(lo_page_infos);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = nios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = collect_compression_methods({lo_page_infos}),
        .gpu_mem_bytes = gpu_mem_bytes,
        .gpu_app_bytes = gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = total_kernel_launches,
    };
}

// ============================================================
// SSB Q3.x GIDP implementation (shared staging + flatten + flat array)
// ============================================================
static BenchmarkResult ssb_q3x_gidp(
    BenchmarkOptions &options,
    SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    SSBTableMetadata *metadatap;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    {
        metadatap = reinterpret_cast<SSBTableMetadata *>(ptr);
        SSBTableMetadata &metadata_pre = *metadatap;
        std::cout << "=== SSB Table Metadata ===" << std::endl;
        std::cout << "Page Size: " << metadata_pre.page_size << std::endl;
        const size_t page_size = metadata_pre.page_size;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, page_size) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, page_size);
    }

    metadatap = reinterpret_cast<SSBTableMetadata *>(ptr);
    SSBTableMetadata &metadata = *metadatap;
    superpage_set_constants_for(metadata.page_size, sizeof(SSBTableMetadata));

    SSB::metadata_print(metadata);

    const size_t page_size = metadata.page_size;
    const size_t nthreads = options.nthreads;
    const size_t num_devices = fds.size();

    // ── Pre-allocate GPU HTs (Rule 2: alloc outside timing) ──
    GpuHT date_ht = alloc_gpu_ht(metadata.table_date_nrows);
    GpuHT cust_ht = alloc_gpu_ht(metadata.table_customer_nrows);
    GpuHT supp_ht = alloc_gpu_ht(metadata.table_supplier_nrows);

    mb_cufile_driver_open();

    // ── cuFileHandles for dim table GDS reads ──
    std::vector<CUfileHandle_t> dim_cufile_handles(num_devices);
    std::vector<int> dim_dup_fds(num_devices);
    for (size_t d = 0; d < num_devices; d++) {
        dim_dup_fds[d] = dup(fds[d]);
        dim_cufile_handles[d] = mb_cufile_handle_register(dim_dup_fds[d]);
    }
    // per_dev_bufs: each device d gets its own registered buffer for parallel cuFileRead
    // — constructed after loader init (see below)

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Group-by output buffer
    size_t group_buf_size = SSB_Q3X_MAX_GROUPS * sizeof(int64_t);
    int64_t *d_revenue;
    CUDA_CHECK(cudaMalloc(&d_revenue, group_buf_size));
    CUDA_CHECK(cudaMemset(d_revenue, 0, group_buf_size));

    // ── Prepare LINEORDER field metadata ──
    constexpr size_t num_lo_cols = SSB::query::q3x::NUM_LO_ACTIVE_FIELDS;
    auto q3x_lo_cols = SSB::query::q3x::LO_FIELDS;
    std::vector<FieldPageInfo> lo_page_infos(num_lo_cols);
    ssb_prepare_fields_metadata(
        fds, metadata, SSB::common::Table::LINEORDER,
        metadata.page_size, q3x_lo_cols, lo_page_infos);

    size_t min_npages = SIZE_MAX;
    for (size_t i = 0; i < num_lo_cols; i++) {
        min_npages = std::min(lo_page_infos[i].npages, min_npages);
    }

    for (size_t fi = 0; fi < num_lo_cols; fi++) {
        const FieldPageInfo &info = lo_page_infos[fi];
        std::cout << "  LO Field " << info.field_index
                  << ": start_page=" << info.start_page_id
                  << " npages=" << info.npages
                  << " compression=" << static_cast<int>(info.compression_method)
                  << std::endl;
    }

    if (min_npages == 0) {
        std::cout << "No pages to read." << std::endl;
        free_fields_metadata(lo_page_infos);
        free(ptr);
        for (int fd : fds) close(fd);
        return BenchmarkResult{};
    }

    bool any_compressed = false;
    for (size_t fi = 0; fi < num_lo_cols; fi++) {
        if (lo_page_infos[fi].compression_method != CompressionMethod::NONE)
            any_compressed = true;
    }

    // ── Pre-allocate GPU dim build buffers (Rule 4) ──
    using DT = SSB::common::Table;
    uint32_t max_dim_pages = 128;
    {
        auto field_max = [&](const uint64_t *np, const std::vector<uint32_t> &fs) {
            for (uint32_t fi : fs) max_dim_pages = std::max(max_dim_pages, (uint32_t)np[fi]);
        };
        field_max(metadata.table_date_npages,
            {SSB::common::D_DATEKEY, SSB::common::D_YEAR, SSB::common::D_YEARMONTHNUM, SSB::common::D_WEEKNUMINYEAR});
        field_max(metadata.table_customer_npages,
            {SSB::common::C_CUSTKEY, SSB::common::C_REGION, SSB::common::C_NATION, SSB::common::C_CITY});
        field_max(metadata.table_supplier_npages,
            {SSB::common::S_SUPPKEY, SSB::common::S_REGION, SSB::common::S_NATION, SSB::common::S_CITY});
    }
    void *per_dev_bufs[num_devices];
    for (size_t d = 0; d < num_devices; d++) {
        per_dev_bufs[d] = mb_cuda_alloc(max_dim_pages * page_size);
        GDS_CHECK(cuFileBufRegister(per_dev_bufs[d], max_dim_pages * page_size, 0));
    }

    // ── Pre-cache dim metadata (Rule 3) ──
    std::map<uint32_t, GdsDimFieldMeta> dim_meta;
    {
        using CF = SSB::common::CustomerField;
        using SF = SSB::common::SupplierField;
        using DF = SSB::common::DateField;
        const std::pair<DT, std::vector<size_t>> q3x_dim_fields[] = {
            {DT::DDATE, {DF::D_DATEKEY, DF::D_YEAR, DF::D_YEARMONTHNUM, DF::D_WEEKNUMINYEAR}},
            {DT::CUSTOMER, {CF::C_CUSTKEY, CF::C_REGION, CF::C_NATION, CF::C_CITY}},
            {DT::SUPPLIER, {SF::S_SUPPKEY, SF::S_REGION, SF::S_NATION, SF::S_CITY}},
        };
        for (auto &[tbl, fields] : q3x_dim_fields)
            for (auto fi : fields)
                dim_meta[dim_field_key(tbl, fi)] = gds_precache_dim_field(
                    dim_cufile_handles, metadata, tbl, fi, page_size, num_devices,
                    per_dev_bufs[0], max_dim_pages);
    }
    size_t max_dim_nrows = std::max({metadata.table_date_nrows,
        metadata.table_customer_nrows, metadata.table_supplier_nrows});
    DimGpuBufs dim_bufs = dim_gpu_bufs_alloc(page_size, max_dim_nrows, max_dim_pages);
    NvcompDecompCtx dim_nvctx{};
    {
        size_t mb = max_dim_pages;
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_decomp_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_actual_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_statuses,     mb * sizeof(nvcompStatus_t)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_decomp_sizes, mb * sizeof(size_t)));
        size_t max_total = mb * page_size;
        size_t snappy_tmp = 0, lz4_tmp = 0;
        nvcompBatchedSnappyDecompressGetTempSizeAsync(
            mb, page_size, nvcompBatchedSnappyDecompressDefaultOpts, &snappy_tmp, max_total);
        nvcompBatchedLZ4DecompressGetTempSizeAsync(
            mb, page_size, nvcompBatchedLZ4DecompressDefaultOpts, &lz4_tmp, max_total);
        dim_nvctx.temp_bytes = std::max(snappy_tmp, lz4_tmp);
        if (dim_nvctx.temp_bytes > 0)
            CUDA_CHECK(cudaMalloc(&dim_nvctx.d_temp, dim_nvctx.temp_bytes));
    }
    cudaStream_t dim_stream;
    CUDA_CHECK(cudaStreamCreate(&dim_stream));
    DimReadScratch dim_scratch = dim_read_scratch_alloc(max_dim_pages);

    // ── GPU memory budget (40 GiB cap) ──
    constexpr size_t Q3X_TILE_PAGES_MAX = 3072;
    size_t Q3X_TILE_PAGES;
    {
        constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t app_fixed = gpu_free_start - gpu_free_now;
        uint64_t remaining = (GPU_MEM_BUDGET > app_fixed) ? GPU_MEM_BUDGET - app_fixed : 0;
        // Worker fixed costs: nvCOMP d_temp + staging rounding
        size_t trial_ppw = (Q3X_TILE_PAGES_MAX + nthreads - 1) / nthreads;
        size_t nvcomp_temp = any_compressed ?
            query_nvcomp_temp_size(trial_ppw, page_size, lo_page_infos) : 0;
        uint64_t worker_fixed = (uint64_t)nthreads * nvcomp_temp
                              + (uint64_t)(nthreads - 1) * 2 * page_size
                              + 64ULL * 1024 * 1024;
        remaining -= std::min(remaining, worker_fixed);
        uint32_t cap_est = (page_size - 12) / sizeof(int32_t);
        size_t per_tp = num_lo_cols * page_size
                      + num_lo_cols * cap_est * sizeof(uint64_t)
                      + sizeof(uint64_t)
                      + 2 * page_size;
        Q3X_TILE_PAGES = std::min(Q3X_TILE_PAGES_MAX,
                                   std::max((size_t)1, (size_t)(remaining / per_tp)));
        // Do not cap by min_npages: allocate full budget for consistent memory usage
    }

    // ── LO worker setup (nthreads workers × all columns, double-buffered) ──
    const size_t pages_per_worker = (Q3X_TILE_PAGES + nthreads - 1) / nthreads;

    LoWorkerCtx workers[nthreads];
    lo_worker_alloc(workers, nthreads, fds.data(), num_devices, page_size,
                    pages_per_worker, any_compressed, lo_page_infos, false);

    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    const size_t npages = min_npages;

    // Compute max tile nrows for flat array allocation
    uint64_t tile_nrows_max = 0;
    for (size_t tile_idx = 0; tile_idx < (npages + Q3X_TILE_PAGES - 1) / Q3X_TILE_PAGES; tile_idx++) {
        size_t p_lo = tile_idx * Q3X_TILE_PAGES;
        size_t tile_np = std::min(Q3X_TILE_PAGES, npages - p_lo);
        if (lo_page_infos[0].prefix_sum_nrecs) {
            uint64_t nr = lo_page_infos[0].prefix_sum_nrecs[p_lo + tile_np]
                        - lo_page_infos[0].prefix_sum_nrecs[p_lo];
            tile_nrows_max = std::max(tile_nrows_max, nr);
        }
    }
    if (tile_nrows_max == 0) {
        uint32_t capacity = (page_size - 12) / sizeof(int32_t);
        tile_nrows_max = Q3X_TILE_PAGES * capacity;
    }

    void *tile_data_buf[num_lo_cols];
    for (size_t i = 0; i < num_lo_cols; i++)
        tile_data_buf[i] = mb_cuda_alloc(Q3X_TILE_PAGES * page_size);

    // Per-column flat arrays
    uint64_t *d_flat[num_lo_cols];
    for (size_t i = 0; i < num_lo_cols; i++)
        CUDA_CHECK(cudaMalloc(&d_flat[i], tile_nrows_max * sizeof(uint64_t)));

    // Prefix sum shared buffer
    uint64_t *d_ps_shared;
    CUDA_CHECK(cudaMalloc(&d_ps_shared, Q3X_TILE_PAGES * sizeof(uint64_t)));

    // Pre-allocate h_tile_ps for flatten lambda (Rule 4)
    uint64_t *h_tile_ps = static_cast<uint64_t *>(malloc(Q3X_TILE_PAGES * sizeof(uint64_t)));

    // Pre-uploaded prefix_sum GPU arrays (Rule 3: populated before total_start)
    std::unordered_map<const uint64_t*, uint64_t*> d_full_ps_gpu;
    // GPU page-active mask for selective_partial prefix_sum (Rule 3)
    uint8_t *d_page_active_mask = nullptr;
    // Per-page nrecs pre-computed from prefix_sum (Rule 3: avoid metadata reads in measurement)
    std::vector<uint64_t> h_per_page_nrecs;

    // ── Flatten helper ──
    auto flatten_int32_field_tile = [&](const FieldPageInfo &fi, void *data_buf,
                                         uint64_t tile_nrows, uint64_t *d_out,
                                         size_t tile_start, size_t tile_npages,
                                         bool selective_partial_mode = false) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            if (selective_partial_mode && d_page_active_mask) {
                compute_selective_batch_ps_kernel<<<1, 1, 0, stream>>>(
                    it->second, d_ps_shared, d_page_active_mask,
                    (uint32_t)tile_start, (uint32_t)tile_npages);
                s_kernel_launches++;
            } else {
                compute_batch_ps_kernel<<<((uint32_t)tile_npages + 255) / 256, 256, 0, stream>>>(
                    it->second, d_ps_shared, (uint32_t)tile_start, (uint32_t)tile_npages);
                s_kernel_launches++;
            }
        } else {
#if 0
            uint64_t cum = 0;
            for (size_t j = 0; j < tile_npages; j++) {
                uint32_t nalloc = 0;
                CUDA_CHECK(cudaMemcpy(&nalloc,
                    static_cast<const char *>(data_buf) + j * page_size,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cum += nalloc;
                h_tile_ps[j] = cum;
            }
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps,
                                  tile_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        ssb_flatten_int32_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, tile_npages, tile_nrows, d_out, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // ── Zonemap GPU buffers (Rule 4: allocate outside timing) ──
    const size_t zm_ref_field = SSB::common::LO_ORDERDATE;

    size_t zm_cust_sw_idx = 0;
    int32_t zm_cust_lo = -1, zm_cust_hi = -1;
    size_t zm_supp_sw_idx = 0;
    int32_t zm_supp_lo = -1, zm_supp_hi = -1;

    uint64_t cust_nstats = 0, cust_stats_start = 0, cust_stats_npg = 0;
    uint64_t supp_nstats = 0, supp_stats_start = 0, supp_stats_npg = 0;

    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);

        if (query == SSB::Query::Q31) {
            zm_cust_sw_idx = SSB::common::LSS_C_REGION;
            auto it = dict_maps[zm_cust_sw_idx].find("ASIA");
            if (it != dict_maps[zm_cust_sw_idx].end()) zm_cust_lo = zm_cust_hi = it->second;
        } else if (query == SSB::Query::Q32) {
            zm_cust_sw_idx = SSB::common::LSS_C_NATION;
            auto it = dict_maps[zm_cust_sw_idx].find("UNITED STATES");
            if (it != dict_maps[zm_cust_sw_idx].end()) zm_cust_lo = zm_cust_hi = it->second;
        } else {
            zm_cust_sw_idx = SSB::common::LSS_C_CITY;
            auto it1 = dict_maps[zm_cust_sw_idx].find("UNITED KI1");
            auto it2 = dict_maps[zm_cust_sw_idx].find("UNITED KI5");
            if (it1 != dict_maps[zm_cust_sw_idx].end() && it2 != dict_maps[zm_cust_sw_idx].end()) {
                zm_cust_lo = std::min(it1->second, it2->second);
                zm_cust_hi = std::max(it1->second, it2->second);
            }
        }
        cust_nstats = metadata.table_lineorder_sideways_nstats[zm_ref_field][zm_cust_sw_idx];
        cust_stats_start = metadata.table_lineorder_sideways_stats_start_page_ids[zm_ref_field][zm_cust_sw_idx];
        cust_stats_npg = metadata.table_lineorder_sideways_stats_npages[zm_ref_field][zm_cust_sw_idx];

        if (query == SSB::Query::Q31) {
            zm_supp_sw_idx = SSB::common::LSS_S_REGION;
            auto it = dict_maps[zm_supp_sw_idx].find("ASIA");
            if (it != dict_maps[zm_supp_sw_idx].end()) zm_supp_lo = zm_supp_hi = it->second;
        } else if (query == SSB::Query::Q32) {
            zm_supp_sw_idx = SSB::common::LSS_S_NATION;
            auto it = dict_maps[zm_supp_sw_idx].find("UNITED STATES");
            if (it != dict_maps[zm_supp_sw_idx].end()) zm_supp_lo = zm_supp_hi = it->second;
        } else {
            zm_supp_sw_idx = SSB::common::LSS_S_CITY;
            auto it1 = dict_maps[zm_supp_sw_idx].find("UNITED KI1");
            auto it2 = dict_maps[zm_supp_sw_idx].find("UNITED KI5");
            if (it1 != dict_maps[zm_supp_sw_idx].end() && it2 != dict_maps[zm_supp_sw_idx].end()) {
                zm_supp_lo = std::min(it1->second, it2->second);
                zm_supp_hi = std::max(it1->second, it2->second);
            }
        }
        supp_nstats = metadata.table_lineorder_sideways_nstats[zm_ref_field][zm_supp_sw_idx];
        supp_stats_start = metadata.table_lineorder_sideways_stats_start_page_ids[zm_ref_field][zm_supp_sw_idx];
        supp_stats_npg = metadata.table_lineorder_sideways_stats_npages[zm_ref_field][zm_supp_sw_idx];
    }

    void *d_zm_stats_cust = nullptr, *d_zm_stats_supp = nullptr;
    uint8_t *d_zm_mask = nullptr;
    uint32_t *d_zm_active_ids = nullptr;
    uint32_t *d_zm_num_selected = nullptr;
    void *d_zm_cub_temp = nullptr;
    size_t zm_cub_temp_bytes = 0;
    ZonemapPred *d_zm_preds = nullptr;
    uint32_t *h_zm_active_ids = nullptr;

    if (options.enable_zonemap) {
        if (cust_nstats > 0 && cust_stats_npg > 0) {
            CUDA_CHECK(cudaMalloc(&d_zm_stats_cust, cust_stats_npg * page_size));
            GDS_CHECK(cuFileBufRegister(d_zm_stats_cust, cust_stats_npg * page_size, 0));
        }
        if (supp_nstats > 0 && supp_stats_npg > 0) {
            CUDA_CHECK(cudaMalloc(&d_zm_stats_supp, supp_stats_npg * page_size));
            GDS_CHECK(cuFileBufRegister(d_zm_stats_supp, supp_stats_npg * page_size, 0));
        }
    }
    CUDA_CHECK(cudaMalloc(&d_zm_mask, npages));
    CUDA_CHECK(cudaMalloc(&d_zm_active_ids, npages * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_zm_num_selected, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_zm_preds, kZonemapMaxPreds * sizeof(ZonemapPred)));
    zm_cub_temp_bytes = zonemap_compact_query_temp(npages);
    if (zm_cub_temp_bytes > 0)
        CUDA_CHECK(cudaMalloc(&d_zm_cub_temp, zm_cub_temp_bytes));
    h_zm_active_ids = static_cast<uint32_t *>(malloc(npages * sizeof(uint32_t)));

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // ── Pre-declare dim dicts (Rule 4: alloc outside timing) ──
    DimDictRaw cust_dict_raw = dim_dict_raw_alloc();
    DimDictRaw supp_dict_raw = dim_dict_raw_alloc();

    // Pre-upload LO field prefix_sums to GPU (Rule 3: before total_start)
    {
        auto upload_ps = [&](const FieldPageInfo &fi) {
            if (!fi.prefix_sum_nrecs || d_full_ps_gpu.count(fi.prefix_sum_nrecs)) return;
            uint64_t *d_ps = nullptr;
            CUDA_CHECK(cudaMalloc(&d_ps, (fi.npages + 1) * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_ps, fi.prefix_sum_nrecs,
                                  (fi.npages + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            d_full_ps_gpu[fi.prefix_sum_nrecs] = d_ps;
        };
        for (size_t i = 0; i < num_lo_cols; i++) upload_ps(lo_page_infos[i]);
    }

    // Pre-compute per-page nrecs from prefix_sum (Rule 3: avoid metadata reads in measurement)
    if (lo_page_infos[0].prefix_sum_nrecs) {
        h_per_page_nrecs.resize(npages);
        for (size_t pg = 0; pg < npages; pg++)
            h_per_page_nrecs[pg] = lo_page_infos[0].prefix_sum_nrecs[pg + 1]
                                 - lo_page_infos[0].prefix_sum_nrecs[pg];
    }

    // ════════════ START TIMING ════════════
    auto total_start = std::chrono::steady_clock::now();
    auto phase_mark = total_start;
    s_kernel_launches = 0;

    // ── Phase 0: GPU dim build (Rule 1: I/O inside timing, Rule 2: parallel IO) ──

    auto dim_t0 = std::chrono::steady_clock::now();

    // DATE
    {
        int32_t date_fmode, date_lo, date_hi;
        if (query == SSB::Query::Q34) {
            date_fmode = 2; date_lo = 199712; date_hi = 0;
        } else {
            date_fmode = 1; date_lo = 1992; date_hi = 1997;
        }
        gds_build_date_ht_ext_gpu(date_ht, dim_meta,
            dim_cufile_handles, num_devices, page_size,
            per_dev_bufs, max_dim_pages,
            dim_nvctx, dim_stream, dim_bufs,
            date_fmode, date_lo, date_hi, dim_scratch, cuda_ctx_handle);
    }
    auto dim_t1 = std::chrono::steady_clock::now();

    // CUSTOMER
    {
        using CF = SSB::common::CustomerField;
        const char *p_asia[] = {"ASIA"};
        const char *p_us[] = {"UNITED STATES"};
        const char *p_uk[] = {"UNITED KI1", "UNITED KI5"};
        if (query == SSB::Query::Q31) {
            gds_build_table_ht_gpu(cust_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::CUSTOMER,
                CF::C_CUSTKEY, CF::C_REGION, SSB::common::C_REGION_SIZE,
                DIM_FILT_EQ, p_asia, 1,
                true, CF::C_NATION, SSB::common::C_NATION_SIZE,
                true, metadata.table_customer_nrows, cust_dict_raw, dim_scratch, cuda_ctx_handle);
        } else if (query == SSB::Query::Q32) {
            gds_build_table_ht_gpu(cust_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::CUSTOMER,
                CF::C_CUSTKEY, CF::C_NATION, SSB::common::C_NATION_SIZE,
                DIM_FILT_EQ, p_us, 1,
                true, CF::C_CITY, SSB::common::C_CITY_SIZE,
                true, metadata.table_customer_nrows, cust_dict_raw, dim_scratch, cuda_ctx_handle);
        } else {
            gds_build_table_ht_gpu(cust_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::CUSTOMER,
                CF::C_CUSTKEY, CF::C_CITY, SSB::common::C_CITY_SIZE,
                DIM_FILT_IN, p_uk, 2,
                false, 0, 0,
                true, metadata.table_customer_nrows, cust_dict_raw, dim_scratch, cuda_ctx_handle);
        }
    }
    auto dim_t2 = std::chrono::steady_clock::now();

    // SUPPLIER
    {
        using SF = SSB::common::SupplierField;
        const char *p_asia[] = {"ASIA"};
        const char *p_us[] = {"UNITED STATES"};
        const char *p_uk[] = {"UNITED KI1", "UNITED KI5"};
        if (query == SSB::Query::Q31) {
            gds_build_table_ht_gpu(supp_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::SUPPLIER,
                SF::S_SUPPKEY, SF::S_REGION, SSB::common::S_REGION_SIZE,
                DIM_FILT_EQ, p_asia, 1,
                true, SF::S_NATION, SSB::common::S_NATION_SIZE,
                true, metadata.table_supplier_nrows, supp_dict_raw, dim_scratch, cuda_ctx_handle);
        } else if (query == SSB::Query::Q32) {
            gds_build_table_ht_gpu(supp_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::SUPPLIER,
                SF::S_SUPPKEY, SF::S_NATION, SSB::common::S_NATION_SIZE,
                DIM_FILT_EQ, p_us, 1,
                true, SF::S_CITY, SSB::common::S_CITY_SIZE,
                true, metadata.table_supplier_nrows, supp_dict_raw, dim_scratch, cuda_ctx_handle);
        } else {
            gds_build_table_ht_gpu(supp_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::SUPPLIER,
                SF::S_SUPPKEY, SF::S_CITY, SSB::common::S_CITY_SIZE,
                DIM_FILT_IN, p_uk, 2,
                false, 0, 0,
                true, metadata.table_supplier_nrows, supp_dict_raw, dim_scratch, cuda_ctx_handle);
        }
    }
    auto dim_t3 = std::chrono::steady_clock::now();
    {
        double date_ms = std::chrono::duration<double, std::milli>(dim_t1 - dim_t0).count();
        double cust_ms = std::chrono::duration<double, std::milli>(dim_t2 - dim_t1).count();
        double supp_ms = std::chrono::duration<double, std::milli>(dim_t3 - dim_t2).count();
        std::cout << "[Q3x-TIMING] dim build: DATE=" << date_ms
                  << " CUSTOMER=" << cust_ms
                  << " SUPPLIER=" << supp_ms << " ms" << std::endl;
    }

    int32_t num_cust_dims = std::max(1u, cust_dict_raw.n);
    int32_t num_supp_dims = std::max(1u, supp_dict_raw.n);

    std::cout << "[Q3x] Customer matches: " << cust_ht.count
              << " (HT size=" << (cust_ht.mask + 1) << ")"
              << " dims=" << num_cust_dims << std::endl;
    std::cout << "[Q3x] Supplier matches: " << supp_ht.count
              << " (HT size=" << (supp_ht.mask + 1) << ")"
              << " dims=" << num_supp_dims << std::endl;

    {
        auto t = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t - phase_mark).count();
        std::cout << "[Q3x-TIMING] Phase 0 (dim build): " << ms << " ms" << std::endl;
        phase_mark = t;
    }

    // ── Zonemap pruning on GPU (Rule 6: IO + eval inside timing) ──
    size_t total_active_pages = npages;
    if (options.enable_zonemap) {
        uint32_t npreds = 0;
        ZonemapPred h_preds[2];

        if (d_zm_stats_cust && zm_cust_lo >= 0) {
            gds_read_zonemap(dim_cufile_handles.data(), num_devices,
                cust_stats_start, cust_stats_npg, page_size, d_zm_stats_cust);
            h_preds[npreds++] = {reinterpret_cast<Stats<int32_t>*>(d_zm_stats_cust),
                cust_nstats, zm_cust_lo, zm_cust_hi};
        }
        if (d_zm_stats_supp && zm_supp_lo >= 0) {
            gds_read_zonemap(dim_cufile_handles.data(), num_devices,
                supp_stats_start, supp_stats_npg, page_size, d_zm_stats_supp);
            h_preds[npreds++] = {reinterpret_cast<Stats<int32_t>*>(d_zm_stats_supp),
                supp_nstats, zm_supp_lo, zm_supp_hi};
        }

        if (npreds > 0) {
            zonemap_eval_preds(npages, h_preds, npreds, d_zm_preds, d_zm_mask, stream);
            s_kernel_launches++;

            zonemap_compact_active(d_zm_mask, npages, d_zm_active_ids,
                d_zm_num_selected, d_zm_cub_temp, zm_cub_temp_bytes, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));

            uint32_t h_num_selected = 0;
            CUDA_CHECK(cudaMemcpy(&h_num_selected, d_zm_num_selected,
                sizeof(uint32_t), cudaMemcpyDeviceToHost));
            total_active_pages = h_num_selected;
            CUDA_CHECK(cudaMemcpy(h_zm_active_ids, d_zm_active_ids,
                total_active_pages * sizeof(uint32_t), cudaMemcpyDeviceToHost));

            d_page_active_mask = d_zm_mask;
        } else {
            for (size_t pg = 0; pg < npages; pg++)
                h_zm_active_ids[pg] = static_cast<uint32_t>(pg);
        }

        std::cout << "[ZONEMAP] Q3x GPU pruning: " << total_active_pages
                  << " / " << npages << " pages active ("
                  << (npages - total_active_pages) << " pruned)" << std::endl;
    } else {
        for (size_t pg = 0; pg < npages; pg++)
            h_zm_active_ids[pg] = static_cast<uint32_t>(pg);
    }

    bool use_selective = options.enable_zonemap && total_active_pages < npages;

    // Q31: active-packed tile partitioning avoids the "sparse active pages"
    // anomaly where physical-range tiling causes extreme imbalance (e.g. SF300:
    // tile 0 gets 1 active page, tile 1 gets 278 active pages, tile 2 gets 0).
    // Packed layout places active pages consecutively in data_buf.
    const bool use_active_packed = (query == SSB::Query::Q31);

    // Ensure d_zm_active_ids is populated on GPU when zonemap pruning did not
    // write it (zonemap disabled or no predicates). Packed mode reads this.
    if (use_active_packed &&
        !(options.enable_zonemap && total_active_pages < npages)) {
        CUDA_CHECK(cudaMemcpyAsync(d_zm_active_ids, h_zm_active_ids,
            total_active_pages * sizeof(uint32_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    size_t num_tiles;
    if (use_active_packed) {
        num_tiles = (total_active_pages + Q3X_TILE_PAGES - 1) / Q3X_TILE_PAGES;
        if (num_tiles == 0) num_tiles = 1;
        std::cout << "[Q3x] Tile execution (active-packed): " << num_tiles
                  << " tiles of up to " << Q3X_TILE_PAGES << " active pages"
                  << " (total_active=" << total_active_pages << ")" << std::endl;
    } else {
        num_tiles = (npages + Q3X_TILE_PAGES - 1) / Q3X_TILE_PAGES;
        std::cout << "[Q3x] Tile execution: " << num_tiles << " tiles of "
                  << Q3X_TILE_PAGES << " pages" << std::endl;
    }

    {
        auto t = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t - phase_mark).count();
        std::cout << "[Q3x-TIMING] Phase 1 (zonemap pruning): " << ms << " ms" << std::endl;
        phase_mark = t;
    }

    double tile_io_ms_total = 0, tile_flatten_ms_total = 0, tile_probe_ms_total = 0;

    size_t active_cursor = 0;
    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        size_t p_lo, tile_np, tile_active_start, n_active;

        if (use_active_packed) {
            // Partition active page list into equal-sized chunks.
            tile_active_start = tile_idx * Q3X_TILE_PAGES;
            n_active = std::min((size_t)Q3X_TILE_PAGES,
                                total_active_pages - tile_active_start);
            if (n_active == 0) continue;
            // p_lo / tile_np are not used for IO in packed mode, but kept for
            // logging consistency: physical span of this tile's active pages.
            p_lo = h_zm_active_ids[tile_active_start];
            uint32_t last_pg = h_zm_active_ids[tile_active_start + n_active - 1];
            tile_np = (size_t)last_pg - p_lo + 1;
        } else {
            p_lo = tile_idx * Q3X_TILE_PAGES;
            tile_np = std::min(Q3X_TILE_PAGES, npages - p_lo);

            // Cursor-based walk through h_zm_active_ids (sorted, O(1) amortized)
            while (active_cursor < total_active_pages && h_zm_active_ids[active_cursor] < (uint32_t)p_lo)
                active_cursor++;
            tile_active_start = active_cursor;
            while (active_cursor < total_active_pages && h_zm_active_ids[active_cursor] < (uint32_t)(p_lo + tile_np))
                active_cursor++;
            n_active = active_cursor - tile_active_start;
            if (n_active == 0) continue;
        }

        bool selective_partial = !use_active_packed && use_selective && n_active < tile_np;

        // Compute tile nrows from pre-computed per-page nrecs
        uint64_t tile_nrows = 0;
        if (!h_per_page_nrecs.empty()) {
            for (size_t a = 0; a < n_active; a++)
                tile_nrows += h_per_page_nrecs[h_zm_active_ids[tile_active_start + a]];
        } else {
            uint32_t capacity = (page_size - 12) / sizeof(int32_t);
            tile_nrows = n_active * capacity;
        }
        if (tile_nrows == 0) continue;

        auto tile_t0 = std::chrono::steady_clock::now();

        // Zero page headers for selective loading (inactive pages → nalloc=0)
        // Not needed in packed mode: active pages are placed consecutively.
        if (selective_partial) {
            unsigned zblk = (tile_np + 255) / 256;
            for (size_t fi = 0; fi < num_lo_cols; fi++) {
                ssb_zero_page_headers_kernel<<<zblk, 256, 0, stream>>>(
                    static_cast<char *>(tile_data_buf[fi]), tile_np, page_size);
                s_kernel_launches++;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Parallel IO + decomp using LoWorkerCtx (nthreads workers × all columns)
        {
            size_t ppw = (n_active + nthreads - 1) / nthreads;
            std::thread thr_buf[nthreads];
            size_t n_thr = 0;
            for (size_t t = 0; t < nthreads; t++) {
                size_t w_start = t * ppw;
                if (w_start >= n_active) break;
                size_t w_count = std::min(ppw, n_active - w_start);
                thr_buf[n_thr++] = std::thread([&, t, w_start, w_count, p_lo,
                                                  tile_active_start, use_active_packed]() {
                    lo_worker_process_tile(
                        workers[t], lo_page_infos.data(), num_lo_cols,
                        page_size, num_devices,
                        h_zm_active_ids + tile_active_start + w_start, w_count,
                        tile_data_buf, p_lo, cuda_ctx_handle,
                        /*packed_mode=*/use_active_packed,
                        /*packed_start=*/use_active_packed ? w_start : 0);
                });
            }
            for (size_t i = 0; i < n_thr; i++) thr_buf[i].join();
        }

        auto tile_t1 = std::chrono::steady_clock::now();

        // Flatten all fields
        if (use_active_packed) {
            // Packed: active pages in data_buf[0..n_active), use new kernel that
            // consumes d_active_ids directly to build prefix_sum over active pages.
            for (size_t fi = 0; fi < num_lo_cols; fi++) {
                auto it = d_full_ps_gpu.find(lo_page_infos[fi].prefix_sum_nrecs);
                if (it == d_full_ps_gpu.end()) {
                    std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
                    abort();
                }
                compute_packed_ps_from_active_ids_kernel<<<1, 1, 0, stream>>>(
                    it->second, d_ps_shared, d_zm_active_ids,
                    (uint32_t)tile_active_start, (uint32_t)n_active);
                s_kernel_launches++;
                ssb_flatten_int32_pages_ps(
                    static_cast<const char *>(tile_data_buf[fi]),
                    page_size, d_ps_shared,
                    (uint32_t)n_active, tile_nrows, d_flat[fi], stream);
                s_kernel_launches++;
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        } else {
            for (size_t fi = 0; fi < num_lo_cols; fi++) {
                flatten_int32_field_tile(lo_page_infos[fi], tile_data_buf[fi], tile_nrows,
                                         d_flat[fi], p_lo, tile_np, selective_partial);
            }
        }

        auto tile_t2 = std::chrono::steady_clock::now();

        // Flat array kernel
        ssb_q3x_probe_flat(
            d_flat[0], d_flat[1], d_flat[2], d_flat[3],
            tile_nrows,
            date_ht.d_keys, date_ht.d_values, date_ht.mask,
            cust_ht.d_keys, cust_ht.d_values, cust_ht.mask,
            supp_ht.d_keys, supp_ht.d_values, supp_ht.mask,
            num_supp_dims, num_cust_dims,
            d_revenue, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto tile_t3 = std::chrono::steady_clock::now();

        double io_ms      = std::chrono::duration<double, std::milli>(tile_t1 - tile_t0).count();
        double flatten_ms = std::chrono::duration<double, std::milli>(tile_t2 - tile_t1).count();
        double probe_ms   = std::chrono::duration<double, std::milli>(tile_t3 - tile_t2).count();
        tile_io_ms_total      += io_ms;
        tile_flatten_ms_total += flatten_ms;
        tile_probe_ms_total   += probe_ms;
        std::cout << "[Q3x-TIMING] tile " << tile_idx << "/" << num_tiles
                  << " (tile_np=" << tile_np << " n_active=" << n_active
                  << " nrows=" << tile_nrows << "): "
                  << "io+decomp=" << io_ms
                  << " flatten=" << flatten_ms
                  << " probe=" << probe_ms << " ms" << std::endl;
    }

    {
        auto t = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t - phase_mark).count();
        std::cout << "[Q3x-TIMING] Phase 2 (tile loop total): " << ms
                  << " ms (io+decomp=" << tile_io_ms_total
                  << " flatten=" << tile_flatten_ms_total
                  << " probe=" << tile_probe_ms_total << ")" << std::endl;
        phase_mark = t;
    }

    int64_t h_revenue[SSB_Q3X_MAX_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_revenue, d_revenue, group_buf_size, cudaMemcpyDeviceToHost));

    // ════════════ END TIMING ════════════
    auto total_end = std::chrono::steady_clock::now();

    // Build dict strings outside timing (Rule 4)
    std::vector<std::string> cust_dim_dict = dim_build_dict_strings(cust_dict_raw);
    std::vector<std::string> supp_dim_dict = dim_build_dict_strings(supp_dict_raw);

    // ── Print results ──
    struct Q3xRow { std::string cust; std::string supp; int32_t year; int64_t rev; };
    std::vector<Q3xRow> q3x_rows;
    for (int32_t c = 0; c < num_cust_dims; c++) {
        for (int32_t s = 0; s < num_supp_dims; s++) {
            for (int32_t y = 0; y < SSB_Q3X_MAX_YEARS; y++) {
                int64_t rev = h_revenue[c * num_supp_dims * SSB_Q3X_MAX_YEARS
                                       + s * SSB_Q3X_MAX_YEARS + y];
                if (rev != 0)
                    q3x_rows.push_back({cust_dim_dict[c], supp_dim_dict[s], SSB_YEAR_MIN + y, rev});
            }
        }
    }
    std::sort(q3x_rows.begin(), q3x_rows.end(), [](const Q3xRow &a, const Q3xRow &b) {
        if (a.year != b.year) return a.year < b.year;
        return a.rev > b.rev;
    });
    std::cout << "\nSSB Q3.x results:" << std::endl;
    for (const auto &r : q3x_rows)
        std::cout << "  " << r.cust << " | " << r.supp << " | " << r.year << " | " << r.rev << std::endl;
    std::cout << "Total result rows: " << q3x_rows.size() << std::endl;

    size_t nios = 0, total_bytes_read = 0;
    uint64_t total_kernel_launches = s_kernel_launches;
    for (size_t t = 0; t < nthreads; t++) {
        nios += workers[t].ios_completed;
        total_bytes_read += workers[t].bytes_read;
        total_kernel_launches += workers[t].kernel_launches;
    }
    total_bytes_read += dim_scratch.bytes_read;

    std::cout << "\n========================================"
              << "\nTotal elapsed: "
              << std::chrono::duration<double>(total_end - total_start).count()
              << " seconds\nTotal I/Os: " << nios
              << "\nTotal bytes read: " << total_bytes_read
              << "\n========================================" << std::endl;

    // ── Cleanup ──
    date_ht.free_all();
    cust_ht.free_all();
    supp_ht.free_all();
    dim_dict_raw_free(cust_dict_raw);
    dim_dict_raw_free(supp_dict_raw);
    CUDA_CHECK(cudaFree(d_revenue));
    CUDA_CHECK(cudaFree(d_ps_shared));
    for (auto &[k, v] : d_full_ps_gpu) cudaFree(v);
    for (size_t i = 0; i < num_lo_cols; i++)
        CUDA_CHECK(cudaFree(d_flat[i]));

    for (size_t i = 0; i < num_lo_cols; i++)
        mb_cuda_free(tile_data_buf[i]);

    // GPU dim build cleanup
    for (size_t d = 0; d < num_devices; d++) {
        cuFileHandleDeregister(dim_cufile_handles[d]);
        close(dim_dup_fds[d]);
    }
    for (size_t d = 0; d < num_devices; d++) {
        cuFileBufDeregister(per_dev_bufs[d]);
        mb_cuda_free(per_dev_bufs[d]);
    }
    dim_gpu_bufs_free(dim_bufs);
    nvcomp_decompctx_free(dim_nvctx);
    dim_read_scratch_free(dim_scratch);
    CUDA_CHECK(cudaStreamDestroy(dim_stream));

    // Zonemap GPU buffer cleanup
    if (d_zm_stats_cust) {
        cuFileBufDeregister(d_zm_stats_cust);
        CUDA_CHECK(cudaFree(d_zm_stats_cust));
    }
    if (d_zm_stats_supp) {
        cuFileBufDeregister(d_zm_stats_supp);
        CUDA_CHECK(cudaFree(d_zm_stats_supp));
    }
    CUDA_CHECK(cudaFree(d_zm_mask));
    CUDA_CHECK(cudaFree(d_zm_active_ids));
    CUDA_CHECK(cudaFree(d_zm_num_selected));
    if (d_zm_cub_temp) CUDA_CHECK(cudaFree(d_zm_cub_temp));
    CUDA_CHECK(cudaFree(d_zm_preds));
    free(h_zm_active_ids);

    CUDA_CHECK(cudaStreamDestroy(stream));

    size_t total_pages = 0;
    for (const auto &fi : lo_page_infos) total_pages += fi.npages;

    lo_worker_free(workers, nthreads, num_devices, any_compressed);
    free(h_tile_ps);

    free_fields_metadata(lo_page_infos);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = nios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = collect_compression_methods({lo_page_infos}),
        .gpu_mem_bytes = gpu_mem_bytes,
        .gpu_app_bytes = gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = total_kernel_launches,
    };
}

// ============================================================
// SSB Q4.x GIDP implementation (shared staging + flatten + flat array)
// ============================================================
static BenchmarkResult ssb_q4x_gidp(
    BenchmarkOptions &options,
    SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    SSBTableMetadata *metadatap;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    {
        metadatap = reinterpret_cast<SSBTableMetadata *>(ptr);
        SSBTableMetadata &metadata_pre = *metadatap;
        std::cout << "=== SSB Table Metadata ===" << std::endl;
        std::cout << "Page Size: " << metadata_pre.page_size << std::endl;
        const size_t page_size = metadata_pre.page_size;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, page_size) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, page_size);
    }

    metadatap = reinterpret_cast<SSBTableMetadata *>(ptr);
    SSBTableMetadata &metadata = *metadatap;
    superpage_set_constants_for(metadata.page_size, sizeof(SSBTableMetadata));

    SSB::metadata_print(metadata);

    const size_t page_size = metadata.page_size;
    const size_t nthreads = options.nthreads;
    const size_t num_devices = fds.size();

    // ── Pre-allocate GPU HTs (Rule 2: alloc outside timing) ──
    GpuHT date_ht = alloc_gpu_ht(metadata.table_date_nrows);
    GpuHT cust_ht = alloc_gpu_ht(metadata.table_customer_nrows);
    GpuHT supp_ht = alloc_gpu_ht(metadata.table_supplier_nrows);
    GpuHT part_ht = alloc_gpu_ht(metadata.table_part_nrows);

    mb_cufile_driver_open();

    // ── Dim cuFile handles + dedicated GDS buffer (Rule 4: alloc outside timing) ──
    std::vector<CUfileHandle_t> dim_cufile_handles(num_devices);
    std::vector<int> dim_dup_fds(num_devices);
    for (size_t d = 0; d < num_devices; d++) {
        dim_dup_fds[d] = dup(fds[d]);
        dim_cufile_handles[d] = mb_cufile_handle_register(dim_dup_fds[d]);
    }
    // per_dev_bufs: each device d gets its own registered buffer for parallel cuFileRead
    // — constructed after loader init (see below)

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Group-by output buffer (worst-case size)
    size_t group_buf_size = SSB_Q4X_MAX_GROUPS * sizeof(int64_t);
    int64_t *d_profit;
    CUDA_CHECK(cudaMalloc(&d_profit, group_buf_size));

    // ── Prepare LINEORDER field metadata ──
    constexpr size_t num_lo_cols = SSB::query::q4x::NUM_LO_ACTIVE_FIELDS;
    auto q4x_lo_cols = SSB::query::q4x::LO_FIELDS;
    std::vector<FieldPageInfo> lo_page_infos(num_lo_cols);
    ssb_prepare_fields_metadata(
        fds, metadata, SSB::common::Table::LINEORDER,
        metadata.page_size, q4x_lo_cols, lo_page_infos);

    size_t min_npages = SIZE_MAX;
    for (size_t i = 0; i < num_lo_cols; i++) {
        min_npages = std::min(lo_page_infos[i].npages, min_npages);
    }

    for (size_t fi = 0; fi < num_lo_cols; fi++) {
        const FieldPageInfo &info = lo_page_infos[fi];
        std::cout << "  LO Field " << info.field_index
                  << ": start_page=" << info.start_page_id
                  << " npages=" << info.npages
                  << " compression=" << static_cast<int>(info.compression_method)
                  << std::endl;
    }

    if (min_npages == 0) {
        std::cout << "No pages to read." << std::endl;
        free_fields_metadata(lo_page_infos);
        free(ptr);
        for (int fd : fds) close(fd);
        return BenchmarkResult{};
    }

    bool any_compressed = false;
    for (size_t fi = 0; fi < num_lo_cols; fi++) {
        if (lo_page_infos[fi].compression_method != CompressionMethod::NONE)
            any_compressed = true;
    }

    // ── Pre-allocate GPU dim build buffers (Rule 4) ──
    using DT = SSB::common::Table;
    uint32_t max_dim_pages = 128;
    {
        auto field_max = [&](const uint64_t *np, const std::vector<uint32_t> &fs) {
            for (uint32_t fi : fs) max_dim_pages = std::max(max_dim_pages, (uint32_t)np[fi]);
        };
        field_max(metadata.table_date_npages,
            {SSB::common::D_DATEKEY, SSB::common::D_YEAR});
        field_max(metadata.table_customer_npages,
            {SSB::common::C_CUSTKEY, SSB::common::C_REGION, SSB::common::C_NATION});
        field_max(metadata.table_supplier_npages,
            {SSB::common::S_SUPPKEY, SSB::common::S_REGION, SSB::common::S_NATION, SSB::common::S_CITY});
        field_max(metadata.table_part_npages,
            {SSB::common::P_PARTKEY, SSB::common::P_MFGR, SSB::common::P_CATEGORY, SSB::common::P_BRAND1});
    }
    void *per_dev_bufs[num_devices];
    for (size_t d = 0; d < num_devices; d++) {
        per_dev_bufs[d] = mb_cuda_alloc(max_dim_pages * page_size);
        GDS_CHECK(cuFileBufRegister(per_dev_bufs[d], max_dim_pages * page_size, 0));
    }

    // ── Pre-cache dim metadata (Rule 3) ──
    std::map<uint32_t, GdsDimFieldMeta> dim_meta;
    {
        using CF = SSB::common::CustomerField;
        using SF = SSB::common::SupplierField;
        using PF = SSB::common::PartField;
        using DF = SSB::common::DateField;
        const std::pair<DT, std::vector<size_t>> q4x_dim_fields[] = {
            {DT::DDATE, {DF::D_DATEKEY, DF::D_YEAR}},
            {DT::CUSTOMER, {CF::C_CUSTKEY, CF::C_REGION, CF::C_NATION}},
            {DT::SUPPLIER, {SF::S_SUPPKEY, SF::S_REGION, SF::S_NATION, SF::S_CITY}},
            {DT::PART, {PF::P_PARTKEY, PF::P_MFGR, PF::P_CATEGORY, PF::P_BRAND1}},
        };
        for (auto &[tbl, fields] : q4x_dim_fields)
            for (auto fi : fields)
                dim_meta[dim_field_key(tbl, fi)] = gds_precache_dim_field(
                    dim_cufile_handles, metadata, tbl, fi, page_size, num_devices,
                    per_dev_bufs[0], max_dim_pages);
    }
    size_t max_dim_nrows = std::max({metadata.table_date_nrows,
        metadata.table_customer_nrows, metadata.table_supplier_nrows,
        metadata.table_part_nrows});
    DimGpuBufs dim_bufs = dim_gpu_bufs_alloc(page_size, max_dim_nrows, max_dim_pages);
    NvcompDecompCtx dim_nvctx{};
    {
        size_t mb = max_dim_pages;
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_decomp_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_actual_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dim_nvctx.d_statuses,     mb * sizeof(nvcompStatus_t)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMallocHost(&dim_nvctx.h_decomp_sizes, mb * sizeof(size_t)));
        size_t max_total = mb * page_size;
        size_t snappy_tmp = 0, lz4_tmp = 0;
        nvcompBatchedSnappyDecompressGetTempSizeAsync(
            mb, page_size, nvcompBatchedSnappyDecompressDefaultOpts, &snappy_tmp, max_total);
        nvcompBatchedLZ4DecompressGetTempSizeAsync(
            mb, page_size, nvcompBatchedLZ4DecompressDefaultOpts, &lz4_tmp, max_total);
        dim_nvctx.temp_bytes = std::max(snappy_tmp, lz4_tmp);
        if (dim_nvctx.temp_bytes > 0)
            CUDA_CHECK(cudaMalloc(&dim_nvctx.d_temp, dim_nvctx.temp_bytes));
    }
    cudaStream_t dim_stream;
    CUDA_CHECK(cudaStreamCreate(&dim_stream));
    DimReadScratch dim_scratch = dim_read_scratch_alloc(max_dim_pages);

    // ── GPU memory budget (40 GiB cap) ──
    constexpr size_t Q4X_TILE_PAGES_MAX = 2048;
    size_t Q4X_TILE_PAGES;
    {
        constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t app_fixed = gpu_free_start - gpu_free_now;
        uint64_t remaining = (GPU_MEM_BUDGET > app_fixed) ? GPU_MEM_BUDGET - app_fixed : 0;
        // Worker fixed costs: nvCOMP d_temp + staging rounding
        size_t trial_ppw = (Q4X_TILE_PAGES_MAX + nthreads - 1) / nthreads;
        size_t nvcomp_temp = any_compressed ?
            query_nvcomp_temp_size(trial_ppw, page_size, lo_page_infos) : 0;
        uint64_t worker_fixed = (uint64_t)nthreads * nvcomp_temp
                              + (uint64_t)(nthreads - 1) * 2 * page_size
                              + 64ULL * 1024 * 1024;
        remaining -= std::min(remaining, worker_fixed);
        uint32_t cap_est = (page_size - 12) / sizeof(int32_t);
        size_t per_tp = num_lo_cols * page_size
                      + num_lo_cols * cap_est * sizeof(uint64_t)
                      + sizeof(uint64_t)
                      + 2 * page_size;
        Q4X_TILE_PAGES = std::min(Q4X_TILE_PAGES_MAX,
                                   std::max((size_t)1, (size_t)(remaining / per_tp)));
        // Do not cap by min_npages: allocate full budget for consistent memory usage
    }

    // ── LO worker setup (nthreads workers × all columns, double-buffered) ──
    const size_t pages_per_worker = (Q4X_TILE_PAGES + nthreads - 1) / nthreads;

    LoWorkerCtx workers[nthreads];
    lo_worker_alloc(workers, nthreads, fds.data(), num_devices, page_size,
                    pages_per_worker, any_compressed, lo_page_infos, false);

    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    const size_t npages = min_npages;

    // Compute max tile nrows for flat array allocation
    uint64_t tile_nrows_max = 0;
    for (size_t tile_idx = 0; tile_idx < (npages + Q4X_TILE_PAGES - 1) / Q4X_TILE_PAGES; tile_idx++) {
        size_t p_lo = tile_idx * Q4X_TILE_PAGES;
        size_t tile_np = std::min(Q4X_TILE_PAGES, npages - p_lo);
        if (lo_page_infos[0].prefix_sum_nrecs) {
            uint64_t nr = lo_page_infos[0].prefix_sum_nrecs[p_lo + tile_np]
                        - lo_page_infos[0].prefix_sum_nrecs[p_lo];
            tile_nrows_max = std::max(tile_nrows_max, nr);
        }
    }
    if (tile_nrows_max == 0) {
        uint32_t capacity = (page_size - 12) / sizeof(int32_t);
        tile_nrows_max = Q4X_TILE_PAGES * capacity;
    }

    // Per-column staging buffers
    void *tile_data_buf[num_lo_cols];
    for (size_t i = 0; i < num_lo_cols; i++)
        tile_data_buf[i] = mb_cuda_alloc(Q4X_TILE_PAGES * page_size);

    // Per-column flat arrays
    uint64_t *d_flat[num_lo_cols];
    for (size_t i = 0; i < num_lo_cols; i++)
        CUDA_CHECK(cudaMalloc(&d_flat[i], tile_nrows_max * sizeof(uint64_t)));

    // Prefix sum shared buffer
    uint64_t *d_ps_shared;
    CUDA_CHECK(cudaMalloc(&d_ps_shared, Q4X_TILE_PAGES * sizeof(uint64_t)));

    // Pre-allocate h_tile_ps for flatten lambda (Rule 4)
    uint64_t *h_tile_ps = static_cast<uint64_t *>(malloc(Q4X_TILE_PAGES * sizeof(uint64_t)));

    // Pre-uploaded prefix_sum GPU arrays (Rule 3: populated before total_start)
    std::unordered_map<const uint64_t*, uint64_t*> d_full_ps_gpu;
    // GPU page-active mask for selective_partial prefix_sum (Rule 3)
    uint8_t *d_page_active_mask = nullptr;
    // Per-page nrecs pre-computed from prefix_sum (Rule 3: avoid metadata reads in measurement)
    std::vector<uint64_t> h_per_page_nrecs;

    // ── Flatten helper ──
    auto flatten_int32_field_tile = [&](const FieldPageInfo &fi, void *data_buf,
                                         uint64_t tile_nrows, uint64_t *d_out,
                                         size_t tile_start, size_t tile_npages,
                                         bool selective_partial_mode = false) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            if (selective_partial_mode && d_page_active_mask) {
                compute_selective_batch_ps_kernel<<<1, 1, 0, stream>>>(
                    it->second, d_ps_shared, d_page_active_mask,
                    (uint32_t)tile_start, (uint32_t)tile_npages);
                s_kernel_launches++;
            } else {
                compute_batch_ps_kernel<<<((uint32_t)tile_npages + 255) / 256, 256, 0, stream>>>(
                    it->second, d_ps_shared, (uint32_t)tile_start, (uint32_t)tile_npages);
                s_kernel_launches++;
            }
        } else {
#if 0
            uint64_t cum = 0;
            for (size_t j = 0; j < tile_npages; j++) {
                uint32_t nalloc = 0;
                CUDA_CHECK(cudaMemcpy(&nalloc,
                    static_cast<const char *>(data_buf) + j * page_size,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cum += nalloc;
                h_tile_ps[j] = cum;
            }
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps,
                                  tile_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        ssb_flatten_int32_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, tile_npages, tile_nrows, d_out, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // ── Zonemap GPU buffers (Rule 4: allocate outside timing) ──
    const size_t zm_ref_field = SSB::common::LO_ORDERDATE;

    int32_t zm_cr_pred = -1;
    size_t zm_cr_sw_idx = SSB::common::LSS_C_REGION;
    uint64_t cr_nstats = metadata.table_lineorder_sideways_nstats[zm_ref_field][zm_cr_sw_idx];
    uint64_t cr_stats_start = metadata.table_lineorder_sideways_stats_start_page_ids[zm_ref_field][zm_cr_sw_idx];
    uint64_t cr_stats_npg = metadata.table_lineorder_sideways_stats_npages[zm_ref_field][zm_cr_sw_idx];

    size_t zm_supp_sw_idx = 0;
    int32_t zm_supp_lo = -1, zm_supp_hi = -1;
    uint64_t supp_nstats = 0, supp_stats_start = 0, supp_stats_npg = 0;

    size_t zm_part_sw_idx = 0;
    int32_t zm_part_lo = -1, zm_part_hi = -1;
    uint64_t part_nstats = 0, part_stats_start = 0, part_stats_npg = 0;

    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);

        auto it_cr = dict_maps[SSB::common::LSS_C_REGION].find("AMERICA");
        if (it_cr != dict_maps[SSB::common::LSS_C_REGION].end()) zm_cr_pred = it_cr->second;

        if (query == SSB::Query::Q41 || query == SSB::Query::Q42) {
            zm_supp_sw_idx = SSB::common::LSS_S_REGION;
            auto it = dict_maps[zm_supp_sw_idx].find("AMERICA");
            if (it != dict_maps[zm_supp_sw_idx].end()) zm_supp_lo = zm_supp_hi = it->second;
        } else {
            zm_supp_sw_idx = SSB::common::LSS_S_NATION;
            auto it = dict_maps[zm_supp_sw_idx].find("UNITED STATES");
            if (it != dict_maps[zm_supp_sw_idx].end()) zm_supp_lo = zm_supp_hi = it->second;
        }
        supp_nstats = metadata.table_lineorder_sideways_nstats[zm_ref_field][zm_supp_sw_idx];
        supp_stats_start = metadata.table_lineorder_sideways_stats_start_page_ids[zm_ref_field][zm_supp_sw_idx];
        supp_stats_npg = metadata.table_lineorder_sideways_stats_npages[zm_ref_field][zm_supp_sw_idx];

        if (query == SSB::Query::Q41 || query == SSB::Query::Q42) {
            zm_part_sw_idx = SSB::common::LSS_P_MFGR;
            auto it1 = dict_maps[zm_part_sw_idx].find("MFGR#1");
            auto it2 = dict_maps[zm_part_sw_idx].find("MFGR#2");
            if (it1 != dict_maps[zm_part_sw_idx].end() && it2 != dict_maps[zm_part_sw_idx].end()) {
                zm_part_lo = std::min(it1->second, it2->second);
                zm_part_hi = std::max(it1->second, it2->second);
            }
        } else {
            zm_part_sw_idx = SSB::common::LSS_P_CATEGORY;
            auto it = dict_maps[zm_part_sw_idx].find("MFGR#14");
            if (it != dict_maps[zm_part_sw_idx].end()) zm_part_lo = zm_part_hi = it->second;
        }
        part_nstats = metadata.table_lineorder_sideways_nstats[zm_ref_field][zm_part_sw_idx];
        part_stats_start = metadata.table_lineorder_sideways_stats_start_page_ids[zm_ref_field][zm_part_sw_idx];
        part_stats_npg = metadata.table_lineorder_sideways_stats_npages[zm_ref_field][zm_part_sw_idx];
    }

    void *d_zm_stats_cr = nullptr, *d_zm_stats_supp = nullptr, *d_zm_stats_part = nullptr;
    uint8_t *d_zm_mask = nullptr;
    uint32_t *d_zm_active_ids = nullptr;
    uint32_t *d_zm_num_selected = nullptr;
    void *d_zm_cub_temp = nullptr;
    size_t zm_cub_temp_bytes = 0;
    ZonemapPred *d_zm_preds = nullptr;
    uint32_t *h_zm_active_ids = nullptr;

    if (options.enable_zonemap) {
        if (cr_nstats > 0 && cr_stats_npg > 0) {
            CUDA_CHECK(cudaMalloc(&d_zm_stats_cr, cr_stats_npg * page_size));
            GDS_CHECK(cuFileBufRegister(d_zm_stats_cr, cr_stats_npg * page_size, 0));
        }
        if (supp_nstats > 0 && supp_stats_npg > 0) {
            CUDA_CHECK(cudaMalloc(&d_zm_stats_supp, supp_stats_npg * page_size));
            GDS_CHECK(cuFileBufRegister(d_zm_stats_supp, supp_stats_npg * page_size, 0));
        }
        if (part_nstats > 0 && part_stats_npg > 0) {
            CUDA_CHECK(cudaMalloc(&d_zm_stats_part, part_stats_npg * page_size));
            GDS_CHECK(cuFileBufRegister(d_zm_stats_part, part_stats_npg * page_size, 0));
        }
    }
    CUDA_CHECK(cudaMalloc(&d_zm_mask, npages));
    CUDA_CHECK(cudaMalloc(&d_zm_active_ids, npages * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_zm_num_selected, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_zm_preds, kZonemapMaxPreds * sizeof(ZonemapPred)));
    zm_cub_temp_bytes = zonemap_compact_query_temp(npages);
    if (zm_cub_temp_bytes > 0)
        CUDA_CHECK(cudaMalloc(&d_zm_cub_temp, zm_cub_temp_bytes));
    h_zm_active_ids = static_cast<uint32_t *>(malloc(npages * sizeof(uint32_t)));

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    CUDA_CHECK(cudaMemset(d_profit, 0, group_buf_size));

    // ── Pre-declare dim dicts (Rule 4: alloc outside timing) ──
    DimDictRaw cust_dict_raw = dim_dict_raw_alloc();
    DimDictRaw supp_dict_raw = dim_dict_raw_alloc();
    DimDictRaw part_dict_raw = dim_dict_raw_alloc();

    // Pre-upload LO field prefix_sums to GPU (Rule 3: before total_start)
    {
        auto upload_ps = [&](const FieldPageInfo &fi) {
            if (!fi.prefix_sum_nrecs || d_full_ps_gpu.count(fi.prefix_sum_nrecs)) return;
            uint64_t *d_ps = nullptr;
            CUDA_CHECK(cudaMalloc(&d_ps, (fi.npages + 1) * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_ps, fi.prefix_sum_nrecs,
                                  (fi.npages + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            d_full_ps_gpu[fi.prefix_sum_nrecs] = d_ps;
        };
        for (size_t i = 0; i < num_lo_cols; i++) upload_ps(lo_page_infos[i]);
    }

    // Pre-compute per-page nrecs from prefix_sum (Rule 3: avoid metadata reads in measurement)
    if (lo_page_infos[0].prefix_sum_nrecs) {
        h_per_page_nrecs.resize(npages);
        for (size_t pg = 0; pg < npages; pg++)
            h_per_page_nrecs[pg] = lo_page_infos[0].prefix_sum_nrecs[pg + 1]
                                 - lo_page_infos[0].prefix_sum_nrecs[pg];
    }

    // ════════════ START TIMING ════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // ── Phase 0: GPU dim build (Rule 1: I/O inside timing, Rule 2: parallel IO) ──

    // DATE
    {
        if (query == SSB::Query::Q41)
            gds_build_date_ht_ext_gpu(date_ht, dim_meta,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                0, 0, 0, dim_scratch, cuda_ctx_handle);       // no filter
        else
            gds_build_date_ht_ext_gpu(date_ht, dim_meta,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                1, 1997, 1998, dim_scratch, cuda_ctx_handle);  // year range
    }

    // CUSTOMER
    {
        using CF = SSB::common::CustomerField;
        const char *p_america[] = {"AMERICA"};
        if (query == SSB::Query::Q41) {
            gds_build_table_ht_gpu(cust_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::CUSTOMER,
                CF::C_CUSTKEY, CF::C_REGION, SSB::common::C_REGION_SIZE,
                DIM_FILT_EQ, p_america, 1,
                true, CF::C_NATION, SSB::common::C_NATION_SIZE,
                true, metadata.table_customer_nrows, cust_dict_raw, dim_scratch, cuda_ctx_handle);
        } else {
            gds_build_table_ht_gpu(cust_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::CUSTOMER,
                CF::C_CUSTKEY, CF::C_REGION, SSB::common::C_REGION_SIZE,
                DIM_FILT_EQ, p_america, 1,
                false, 0, 0,
                false, metadata.table_customer_nrows, cust_dict_raw, dim_scratch, cuda_ctx_handle);
        }
    }

    // SUPPLIER
    {
        using SF = SSB::common::SupplierField;
        const char *p_america[] = {"AMERICA"};
        const char *p_us[] = {"UNITED STATES"};
        if (query == SSB::Query::Q41) {
            gds_build_table_ht_gpu(supp_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::SUPPLIER,
                SF::S_SUPPKEY, SF::S_REGION, SSB::common::S_REGION_SIZE,
                DIM_FILT_EQ, p_america, 1,
                false, 0, 0,
                false, metadata.table_supplier_nrows, supp_dict_raw, dim_scratch, cuda_ctx_handle);
        } else if (query == SSB::Query::Q42) {
            gds_build_table_ht_gpu(supp_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::SUPPLIER,
                SF::S_SUPPKEY, SF::S_REGION, SSB::common::S_REGION_SIZE,
                DIM_FILT_EQ, p_america, 1,
                true, SF::S_NATION, SSB::common::S_NATION_SIZE,
                true, metadata.table_supplier_nrows, supp_dict_raw, dim_scratch, cuda_ctx_handle);
        } else {
            gds_build_table_ht_gpu(supp_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::SUPPLIER,
                SF::S_SUPPKEY, SF::S_NATION, SSB::common::S_NATION_SIZE,
                DIM_FILT_EQ, p_us, 1,
                true, SF::S_CITY, SSB::common::S_CITY_SIZE,
                true, metadata.table_supplier_nrows, supp_dict_raw, dim_scratch, cuda_ctx_handle);
        }
    }

    // PART
    {
        using PF = SSB::common::PartField;
        const char *p_mfgr12[] = {"MFGR#1", "MFGR#2"};
        const char *p_mfgr14[] = {"MFGR#14"};
        if (query == SSB::Query::Q41) {
            gds_build_table_ht_gpu(part_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::PART,
                PF::P_PARTKEY, PF::P_MFGR, SSB::common::P_MFGR_SIZE,
                DIM_FILT_IN, p_mfgr12, 2,
                false, 0, 0,
                false, metadata.table_part_nrows, part_dict_raw, dim_scratch, cuda_ctx_handle);
        } else if (query == SSB::Query::Q42) {
            gds_build_table_ht_gpu(part_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::PART,
                PF::P_PARTKEY, PF::P_MFGR, SSB::common::P_MFGR_SIZE,
                DIM_FILT_IN, p_mfgr12, 2,
                true, PF::P_CATEGORY, SSB::common::P_CATEGORY_SIZE,
                true, metadata.table_part_nrows, part_dict_raw, dim_scratch, cuda_ctx_handle);
        } else {
            gds_build_table_ht_gpu(part_ht, dim_meta, metadata,
                dim_cufile_handles, num_devices, page_size,
                per_dev_bufs, max_dim_pages,
                dim_nvctx, dim_stream, dim_bufs,
                DT::PART,
                PF::P_PARTKEY, PF::P_CATEGORY, SSB::common::P_CATEGORY_SIZE,
                DIM_FILT_EQ, p_mfgr14, 1,
                true, PF::P_BRAND1, SSB::common::P_BRAND1_SIZE,
                true, metadata.table_part_nrows, part_dict_raw, dim_scratch, cuda_ctx_handle);
        }
    }

    int32_t num_cust_dims = std::max(1u, cust_dict_raw.n);
    int32_t num_supp_dims = std::max(1u, supp_dict_raw.n);
    int32_t num_part_dims = std::max(1u, part_dict_raw.n);
    int32_t stride_year = num_cust_dims * num_supp_dims * num_part_dims;
    int32_t total_groups = SSB_Q4X_MAX_YEARS * stride_year;

    std::cout << "[Q4x] Customer matches: " << cust_ht.count
              << " dims=" << num_cust_dims << std::endl;
    std::cout << "[Q4x] Supplier matches: " << supp_ht.count
              << " dims=" << num_supp_dims << std::endl;
    std::cout << "[Q4x] Part matches: " << part_ht.count
              << " dims=" << num_part_dims << std::endl;
    std::cout << "[Q4x] Total groups: " << total_groups << std::endl;

    // ── Zonemap pruning on GPU (Rule 6: IO + eval inside timing) ──
    size_t total_active_pages = npages;
    if (options.enable_zonemap) {
        uint32_t npreds = 0;
        ZonemapPred h_preds[3];

        if (d_zm_stats_cr && zm_cr_pred >= 0) {
            gds_read_zonemap(dim_cufile_handles.data(), num_devices,
                cr_stats_start, cr_stats_npg, page_size, d_zm_stats_cr);
            h_preds[npreds++] = {reinterpret_cast<Stats<int32_t>*>(d_zm_stats_cr),
                cr_nstats, zm_cr_pred, zm_cr_pred};
        }
        if (d_zm_stats_supp && zm_supp_lo >= 0) {
            gds_read_zonemap(dim_cufile_handles.data(), num_devices,
                supp_stats_start, supp_stats_npg, page_size, d_zm_stats_supp);
            h_preds[npreds++] = {reinterpret_cast<Stats<int32_t>*>(d_zm_stats_supp),
                supp_nstats, zm_supp_lo, zm_supp_hi};
        }
        if (d_zm_stats_part && zm_part_lo >= 0) {
            gds_read_zonemap(dim_cufile_handles.data(), num_devices,
                part_stats_start, part_stats_npg, page_size, d_zm_stats_part);
            h_preds[npreds++] = {reinterpret_cast<Stats<int32_t>*>(d_zm_stats_part),
                part_nstats, zm_part_lo, zm_part_hi};
        }

        if (npreds > 0) {
            zonemap_eval_preds(npages, h_preds, npreds, d_zm_preds, d_zm_mask, stream);
            s_kernel_launches++;

            zonemap_compact_active(d_zm_mask, npages, d_zm_active_ids,
                d_zm_num_selected, d_zm_cub_temp, zm_cub_temp_bytes, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));

            uint32_t h_num_selected = 0;
            CUDA_CHECK(cudaMemcpy(&h_num_selected, d_zm_num_selected,
                sizeof(uint32_t), cudaMemcpyDeviceToHost));
            total_active_pages = h_num_selected;
            CUDA_CHECK(cudaMemcpy(h_zm_active_ids, d_zm_active_ids,
                total_active_pages * sizeof(uint32_t), cudaMemcpyDeviceToHost));

            d_page_active_mask = d_zm_mask;
        } else {
            for (size_t pg = 0; pg < npages; pg++)
                h_zm_active_ids[pg] = static_cast<uint32_t>(pg);
        }

        std::cout << "[ZONEMAP] Q4x GPU pruning: " << total_active_pages
                  << " / " << npages << " pages active ("
                  << (npages - total_active_pages) << " pruned)" << std::endl;
    } else {
        for (size_t pg = 0; pg < npages; pg++)
            h_zm_active_ids[pg] = static_cast<uint32_t>(pg);
    }

    bool use_selective = options.enable_zonemap && total_active_pages < npages;

    size_t num_tiles = (npages + Q4X_TILE_PAGES - 1) / Q4X_TILE_PAGES;
    std::cout << "[Q4x] Tile execution: " << num_tiles << " tiles of "
              << Q4X_TILE_PAGES << " pages" << std::endl;

    size_t active_cursor = 0;
    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        size_t p_lo = tile_idx * Q4X_TILE_PAGES;
        size_t tile_np = std::min(Q4X_TILE_PAGES, npages - p_lo);

        // Cursor-based walk through h_zm_active_ids (sorted, O(1) amortized)
        while (active_cursor < total_active_pages && h_zm_active_ids[active_cursor] < (uint32_t)p_lo)
            active_cursor++;
        size_t tile_active_start = active_cursor;
        while (active_cursor < total_active_pages && h_zm_active_ids[active_cursor] < (uint32_t)(p_lo + tile_np))
            active_cursor++;
        size_t n_active = active_cursor - tile_active_start;
        if (n_active == 0) continue;

        bool selective_partial = use_selective && n_active < tile_np;

        // Compute tile nrows from pre-computed per-page nrecs
        uint64_t tile_nrows = 0;
        if (!h_per_page_nrecs.empty()) {
            for (size_t a = 0; a < n_active; a++)
                tile_nrows += h_per_page_nrecs[h_zm_active_ids[tile_active_start + a]];
        } else {
            uint32_t capacity = (page_size - 12) / sizeof(int32_t);
            tile_nrows = n_active * capacity;
        }
        if (tile_nrows == 0) continue;

        // Zero page headers for selective loading (inactive pages → nalloc=0)
        if (selective_partial) {
            unsigned zblk = (tile_np + 255) / 256;
            for (size_t fi = 0; fi < num_lo_cols; fi++) {
                ssb_zero_page_headers_kernel<<<zblk, 256, 0, stream>>>(
                    static_cast<char *>(tile_data_buf[fi]), tile_np, page_size);
                s_kernel_launches++;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Parallel IO + decomp using LoWorkerCtx (nthreads workers × all columns)
        {
            size_t ppw = (n_active + nthreads - 1) / nthreads;
            std::thread thr_buf[nthreads];
            size_t n_thr = 0;
            for (size_t t = 0; t < nthreads; t++) {
                size_t w_start = t * ppw;
                if (w_start >= n_active) break;
                size_t w_count = std::min(ppw, n_active - w_start);
                thr_buf[n_thr++] = std::thread([&, t, w_start, w_count, p_lo, tile_active_start]() {
                    lo_worker_process_tile(
                        workers[t], lo_page_infos.data(), num_lo_cols,
                        page_size, num_devices,
                        h_zm_active_ids + tile_active_start + w_start, w_count,
                        tile_data_buf, p_lo, cuda_ctx_handle);
                });
            }
            for (size_t i = 0; i < n_thr; i++) thr_buf[i].join();
        }

        // Flatten all fields
        for (size_t fi = 0; fi < num_lo_cols; fi++) {
            flatten_int32_field_tile(lo_page_infos[fi], tile_data_buf[fi], tile_nrows,
                                     d_flat[fi], p_lo, tile_np, selective_partial);
        }

        // Flat array kernel
        ssb_q4x_probe_flat(
            d_flat[0], d_flat[1], d_flat[2], d_flat[3],
            d_flat[4], d_flat[5],
            tile_nrows,
            date_ht.d_keys, date_ht.d_values, date_ht.mask,
            cust_ht.d_keys, cust_ht.d_values, cust_ht.mask,
            supp_ht.d_keys, supp_ht.d_values, supp_ht.mask,
            part_ht.d_keys, part_ht.d_values, part_ht.mask,
            num_supp_dims, num_part_dims, stride_year, total_groups,
            d_profit, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    int64_t h_profit[SSB_Q4X_MAX_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_profit, d_profit, total_groups * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // ════════════ END TIMING ════════════
    auto total_end = std::chrono::steady_clock::now();

    // Build dict strings outside timing (Rule 4)
    std::vector<std::string> cust_dim_dict = dim_build_dict_strings(cust_dict_raw);
    std::vector<std::string> supp_dim_dict = dim_build_dict_strings(supp_dict_raw);
    std::vector<std::string> part_dim_dict = dim_build_dict_strings(part_dict_raw);

    std::cout << "\nSSB Q4.x results:" << std::endl;
    size_t result_count = 0;
    for (int32_t y = 0; y < SSB_Q4X_MAX_YEARS; y++) {
        for (int32_t c = 0; c < num_cust_dims; c++) {
            for (int32_t s = 0; s < num_supp_dims; s++) {
                for (int32_t p = 0; p < num_part_dims; p++) {
                    int32_t idx = y * stride_year
                                + c * (num_supp_dims * num_part_dims)
                                + s * num_part_dims + p;
                    int64_t val = h_profit[idx];
                    if (val != 0) {
                        std::cout << "  " << (SSB_YEAR_MIN + y);
                        if (cust_dim_dict[0] != "_")
                            std::cout << " | " << cust_dim_dict[c];
                        if (supp_dim_dict[0] != "_")
                            std::cout << " | " << supp_dim_dict[s];
                        if (part_dim_dict[0] != "_")
                            std::cout << " | " << part_dim_dict[p];
                        std::cout << " | " << val << std::endl;
                        result_count++;
                    }
                }
            }
        }
    }
    std::cout << "Total result rows: " << result_count << std::endl;

    size_t nios = 0, total_bytes_read = 0;
    uint64_t total_kernel_launches = s_kernel_launches;
    for (size_t t = 0; t < nthreads; t++) {
        nios += workers[t].ios_completed;
        total_bytes_read += workers[t].bytes_read;
        total_kernel_launches += workers[t].kernel_launches;
    }
    total_bytes_read += dim_scratch.bytes_read;

    std::cout << "\n========================================"
              << "\nTotal elapsed: "
              << std::chrono::duration<double>(total_end - total_start).count()
              << " seconds\nTotal I/Os: " << nios
              << "\nTotal bytes read: " << total_bytes_read
              << "\n========================================" << std::endl;

    // ── Cleanup ──
    date_ht.free_all();
    cust_ht.free_all();
    supp_ht.free_all();
    part_ht.free_all();
    dim_dict_raw_free(cust_dict_raw);
    dim_dict_raw_free(supp_dict_raw);
    dim_dict_raw_free(part_dict_raw);
    CUDA_CHECK(cudaFree(d_profit));
    CUDA_CHECK(cudaFree(d_ps_shared));
    for (auto &[k, v] : d_full_ps_gpu) cudaFree(v);
    for (size_t i = 0; i < num_lo_cols; i++)
        CUDA_CHECK(cudaFree(d_flat[i]));

    for (size_t i = 0; i < num_lo_cols; i++)
        mb_cuda_free(tile_data_buf[i]);

    // GPU dim build cleanup
    for (size_t d = 0; d < num_devices; d++) {
        cuFileHandleDeregister(dim_cufile_handles[d]);
        close(dim_dup_fds[d]);
    }
    for (size_t d = 0; d < num_devices; d++) {
        cuFileBufDeregister(per_dev_bufs[d]);
        mb_cuda_free(per_dev_bufs[d]);
    }
    dim_gpu_bufs_free(dim_bufs);
    nvcomp_decompctx_free(dim_nvctx);
    dim_read_scratch_free(dim_scratch);
    CUDA_CHECK(cudaStreamDestroy(dim_stream));

    // Zonemap GPU buffer cleanup
    if (d_zm_stats_cr) {
        cuFileBufDeregister(d_zm_stats_cr);
        CUDA_CHECK(cudaFree(d_zm_stats_cr));
    }
    if (d_zm_stats_supp) {
        cuFileBufDeregister(d_zm_stats_supp);
        CUDA_CHECK(cudaFree(d_zm_stats_supp));
    }
    if (d_zm_stats_part) {
        cuFileBufDeregister(d_zm_stats_part);
        CUDA_CHECK(cudaFree(d_zm_stats_part));
    }
    CUDA_CHECK(cudaFree(d_zm_mask));
    CUDA_CHECK(cudaFree(d_zm_active_ids));
    CUDA_CHECK(cudaFree(d_zm_num_selected));
    if (d_zm_cub_temp) CUDA_CHECK(cudaFree(d_zm_cub_temp));
    CUDA_CHECK(cudaFree(d_zm_preds));
    free(h_zm_active_ids);

    CUDA_CHECK(cudaStreamDestroy(stream));

    size_t total_pages = 0;
    for (const auto &fi : lo_page_infos) total_pages += fi.npages;

    lo_worker_free(workers, nthreads, num_devices, any_compressed);

    free(h_tile_ps);
    free_fields_metadata(lo_page_infos);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = nios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = collect_compression_methods({lo_page_infos}),
        .gpu_mem_bytes = gpu_mem_bytes,
        .gpu_app_bytes = gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = total_kernel_launches,
    };
}

} // namespace SsbGidp

BenchmarkResult ssb_q11_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q1x_gidp(options, SSB::Query::Q11);
}

BenchmarkResult ssb_q12_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q1x_gidp(options, SSB::Query::Q12);
}

BenchmarkResult ssb_q13_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q1x_gidp(options, SSB::Query::Q13);
}

BenchmarkResult ssb_q21_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q2x_gidp(options, SSB::Query::Q21);
}

BenchmarkResult ssb_q22_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q2x_gidp(options, SSB::Query::Q22);
}

BenchmarkResult ssb_q23_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q2x_gidp(options, SSB::Query::Q23);
}

BenchmarkResult ssb_q31_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q3x_gidp(options, SSB::Query::Q31);
}

BenchmarkResult ssb_q32_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q3x_gidp(options, SSB::Query::Q32);
}

BenchmarkResult ssb_q33_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q3x_gidp(options, SSB::Query::Q33);
}

BenchmarkResult ssb_q34_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q3x_gidp(options, SSB::Query::Q34);
}

BenchmarkResult ssb_q41_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q4x_gidp(options, SSB::Query::Q41);
}

BenchmarkResult ssb_q42_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q4x_gidp(options, SSB::Query::Q42);
}

BenchmarkResult ssb_q43_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q4x_gidp(options, SSB::Query::Q43);
}

BenchmarkResult ssb_revenue_gidp(BenchmarkOptions &options) {
    return SsbGidp::ssb_q1x_gidp(options, SSB::Query::REVENUE);
}
