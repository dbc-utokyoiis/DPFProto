#pragma once

#include "common/common.cu"
#include "common/page.cu"
#include "common/primitive_c.cu"
#include "common/primitive_cuda.cu"
#include "common/primitive_cufile.cu"
#include "metadata/metadata.h"
//#include "schema/lineitem.cu"
#include "schema/table.h"
#include "schema/tpch_tables.cuh"
#include "common/filter.cuh"
#include "kernel/tpch/scan.cuh"
#include "kernel/tpch/revenue.cuh"
#include "kernel/tpch/q5.cuh"
#include "kernel/tpch/q13.cuh"
#include "kernel/tpch/q3.cuh"
#include "kernel/tpch/q1.cuh"
#include "kernel/tpch/q16.cuh"
//#include "kernel/tpch/gpu_pag.cuh"
#include "tpch/bam_lz4_fused_q5_dim.cuh"
#include "common/pruning.cuh"

#include <span>
#include <numeric>
#include <algorithm>
//#include <nvcomp.h>
#include <nvcomp/snappy.h>
#include <nvcomp/lz4.h>

#include <set>
#include <atomic>
#include <unordered_map>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/device/device_merge_sort.cuh>

constexpr size_t MAX_GDS_DEVICES = 64;

// Collect unique compression method names from FieldPageInfo vectors.
// Returns e.g. "NONE", "SNAPPY", "SNAPPY+LZ4" (sorted, deduplicated).
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

static const char *cufile_error_string(CUfileOpError err) {
    switch (err) {
        case CU_FILE_SUCCESS:                  return "CU_FILE_SUCCESS";
        case CU_FILE_DRIVER_NOT_INITIALIZED:   return "CU_FILE_DRIVER_NOT_INITIALIZED";
        case CU_FILE_INVALID_FILE_OPEN_FLAG:   return "CU_FILE_INVALID_FILE_OPEN_FLAG";
        case CU_FILE_DRIVER_VERSION_MISMATCH:  return "CU_FILE_DRIVER_VERSION_MISMATCH";
        default:                               return "UNKNOWN";
    }
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
        std::cerr << "GDS error: " << cufile_error_string(err.err)     \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

// cuFileBatchIOSetUp maximum entries (GDS internal limit = io_batchsize)
static constexpr size_t GDS_MAX_BATCH_ENTRIES = 128;

// Max I/O size per batch entry (matches per_buffer_cache_size_kb in cufile.json)
static constexpr size_t GDS_MAX_IO_SIZE = 16 * 1024 * 1024;  // 16 MiB

// Number of pages per sync I/O batch (-S mode)
static constexpr size_t GDS_SYNC_BATCH_PAGES = 1024;

struct GDSReaderConfig {
    std::span<int> fds;
    size_t page_size = 0;
    size_t large_page_size = 16 * 1024 * 1024;  // coalesced I/O unit
    size_t num_threads = 1;
    size_t io_multiplicity = 1;       // number of large-page I/Os in flight
    size_t accumulate_rounds = 8;     // number of cycles before Q6 kernel
    size_t handles_per_thread = 1;
    int32_t q6_sd_low = 19940101;     // Q6 L_SHIPDATE lower bound (inclusive)
    int32_t q6_sd_high = 19950101;    // Q6 L_SHIPDATE upper bound (exclusive)
    int32_t revenue_qt_max = 0;       // Revenue: L_QUANTITY < qt_max (0 = no filter)
};

struct GDSThreadContext {
    size_t thread_id;
    CUcontext cuda_ctx;
    GDSReaderConfig config;

    // Per-device file handles: fds[device][handle_idx]
    std::vector<std::vector<int>> fds;
    std::vector<std::vector<CUfileHandle_t>> cufile_handles;

    // Staging buffer for coalesced GDS I/O (cuFileBufRegister'd)
    void *io_buf = nullptr;
    // Per-field data buffer (NOT registered, decompressed pages accumulate here)
    void *data_buf[TPCH::common::TPCH_MAX_NFIELDS] = {};

    CUfileBatchHandle_t batch_handle;

    // nvCOMP decompression arrays (device)
    void   **d_nvcomp_comp_ptrs   = nullptr;
    void   **d_nvcomp_decomp_ptrs = nullptr;
    size_t  *d_nvcomp_comp_sizes  = nullptr;
    size_t  *d_nvcomp_decomp_sizes = nullptr;
    size_t  *d_nvcomp_actual_sizes = nullptr;
    nvcompStatus_t *d_nvcomp_statuses = nullptr;
    void   *d_nvcomp_temp         = nullptr;
    size_t  nvcomp_temp_bytes     = 0;

    // nvCOMP decompression arrays (pinned host)
    void   **h_nvcomp_comp_ptrs   = nullptr;
    void   **h_nvcomp_decomp_ptrs = nullptr;
    size_t  *h_nvcomp_comp_sizes  = nullptr;
    size_t  *h_nvcomp_decomp_sizes = nullptr;

    size_t bytes_read = 0;
    size_t ios_completed = 0;

    int64_t *d_q6_revenue = nullptr;
    uint64_t q6_nrecs_lineitem_total = 0;
};


// ============================================================
// Batch IO submit + wait
// ============================================================

static void submit_and_wait_batch(GDSThreadContext &ctx,
                                   CUfileIOParams_t *params,
                                   size_t count) {
    if (count == 0) return;

    std::vector<CUfileIOEvents_t> events(count);

    CUfileError_t status = cuFileBatchIOSubmit(ctx.batch_handle,
                                               static_cast<unsigned>(count),
                                               params, 0);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileBatchIOSubmit failed: thread=" << ctx.thread_id
                  << " err=" << cufile_error_string(status.err)
                  << " (code=" << static_cast<int>(status.err) << ")"
                  << " count=" << count << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t completed = 0;
    while (completed < count) {
        unsigned nr = static_cast<unsigned>(count - completed);
        struct timespec ts = {1, 0};
        status = cuFileBatchIOGetStatus(ctx.batch_handle, 1, &nr, events.data(), &ts);
        if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "cuFileBatchIOGetStatus failed: thread=" << ctx.thread_id << std::endl;
            return;
        }
        for (unsigned i = 0; i < nr; i++) {
            if (events[i].status == CUFILE_COMPLETE && events[i].ret > 0) {
                ctx.bytes_read += events[i].ret;
            }
            ctx.ios_completed++;
            completed++;
        }
    }
}

// ============================================================
// nvCOMP GPU decompression
// ============================================================

static void nvcomp_decompress_batch(
    CompressionMethod method,
    GDSThreadContext &ctx,
    size_t num_chunks,
    size_t page_size,
    cudaStream_t stream)
{
    if (num_chunks == 0) return;

    CUDA_CHECK(cudaMemcpyAsync(ctx.d_nvcomp_comp_ptrs, ctx.h_nvcomp_comp_ptrs,
                                num_chunks * sizeof(void *),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_nvcomp_decomp_ptrs, ctx.h_nvcomp_decomp_ptrs,
                                num_chunks * sizeof(void *),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_nvcomp_comp_sizes, ctx.h_nvcomp_comp_sizes,
                                num_chunks * sizeof(size_t),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_nvcomp_decomp_sizes, ctx.h_nvcomp_decomp_sizes,
                                num_chunks * sizeof(size_t),
                                cudaMemcpyHostToDevice, stream));

    nvcompStatus_t nvstatus;

    if (method == CompressionMethod::SNAPPY) {
        nvstatus = nvcompBatchedSnappyDecompressAsync(
            (const void *const *)ctx.d_nvcomp_comp_ptrs,
            ctx.d_nvcomp_comp_sizes,
            ctx.d_nvcomp_decomp_sizes,
            ctx.d_nvcomp_actual_sizes,
            num_chunks,
            ctx.d_nvcomp_temp,
            ctx.nvcomp_temp_bytes,
            (void *const *)ctx.d_nvcomp_decomp_ptrs,
            nvcompBatchedSnappyDecompressDefaultOpts,
            ctx.d_nvcomp_statuses,
            stream);
        if (nvstatus != nvcompSuccess) {
            std::cerr << "nvcompBatchedSnappyDecompressAsync failed: " << nvstatus << std::endl;
        }
    } else if (method == CompressionMethod::LZ4) {
        nvstatus = nvcompBatchedLZ4DecompressAsync(
            (const void *const *)ctx.d_nvcomp_comp_ptrs,
            ctx.d_nvcomp_comp_sizes,
            ctx.d_nvcomp_decomp_sizes,
            ctx.d_nvcomp_actual_sizes,
            num_chunks,
            ctx.d_nvcomp_temp,
            ctx.nvcomp_temp_bytes,
            (void *const *)ctx.d_nvcomp_decomp_ptrs,
            nvcompBatchedLZ4DecompressDefaultOpts,
            ctx.d_nvcomp_statuses,
            stream);
        if (nvstatus != nvcompSuccess) {
            std::cerr << "nvcompBatchedLZ4DecompressAsync failed: " << nvstatus << std::endl;
        }
    } else {
        std::cerr << "Unsupported compression method for GPU decompression: "
                  << static_cast<int>(method) << std::endl;
        return;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}


// ============================================================
// Worker thread — processes all N_FIELDS fields per cycle
// ============================================================
static void gds_worker_batch_thread(
    GDSThreadContext &ctx,
    const std::vector<FieldPageInfo> &fields,
    std::span<const size_t> page_indices)
{
    mb_cuda_set_context(ctx.cuda_ctx);

    const auto &config = ctx.config;
    const size_t num_devices = config.fds.size();
    const size_t io_multiplicity = config.io_multiplicity;  // number of large-page I/Os in flight
    const size_t page_size = config.page_size;
    const size_t large_page_size = config.large_page_size;
    const size_t pages_per_io = large_page_size / page_size;
    const size_t accumulate_pages = io_multiplicity * pages_per_io * config.accumulate_rounds;

    // sizeof(pag_head) == 12
    const uint32_t capacity_per_page = (page_size - 12) / sizeof(int32_t);

    // Field index mapping for Q6
    const size_t L_SHIPDATE_IDX = 0;
    const size_t L_QUANTITY_IDX = 1;
    const size_t L_EXTENDEDPRICE_IDX = 2;
    const size_t L_DISCOUNT_IDX = 3;

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    // Params vector: worst case = every page is a separate run (segment boundaries
    // in compressed data), and each run may need splitting into GDS_MAX_IO_SIZE chunks.
    const size_t chunks_per_page = (page_size + GDS_MAX_IO_SIZE - 1) / GDS_MAX_IO_SIZE;
    const size_t max_batch_entries = io_multiplicity * pages_per_io * chunks_per_page;
    const size_t batch_capacity = io_multiplicity;
    std::vector<CUfileIOParams_t> params(max_batch_entries);
    size_t next_fh[MAX_GDS_DEVICES] = {};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const size_t total_pages = page_indices.size();
    size_t current_idx = 0;

    while (current_idx < total_pages) {
        size_t pages_this_cycle = std::min(accumulate_pages, total_pages - current_idx);

        // ── I/O + decompress phase: each field in turn ──
        for (size_t fi = 0; fi < fields.size(); fi++) {
            const FieldPageInfo &finfo = fields[fi];
            bool is_compressed = (finfo.compression_method != CompressionMethod::NONE);

            size_t field_pages = pages_this_cycle;

            char *data_base = static_cast<char *>(ctx.data_buf[fi]);
            char *io_base = static_cast<char *>(ctx.io_buf);

            size_t pages_done = 0;
            while (pages_done < field_pages) {
                // How many pages to read in this submit
                size_t batch_pages = std::min(io_multiplicity * pages_per_io,
                                              field_pages - pages_done);
                size_t n_large_pages = (batch_pages + pages_per_io - 1) / pages_per_io;

                // ── Build batch I/O entries (split at segment boundaries) ──
                size_t entry_idx = 0;
                memset(next_fh, 0, num_devices * sizeof(size_t));

                // Helper: flush a contiguous run, splitting into GDS_MAX_IO_SIZE chunks
                auto flush_run = [&](size_t d, char *dest, size_t file_off, size_t size) {
                    for (size_t off = 0; off < size; off += GDS_MAX_IO_SIZE) {
                        size_t chunk = std::min(GDS_MAX_IO_SIZE, size - off);
                        size_t fh_idx = next_fh[d] % config.handles_per_thread;
                        next_fh[d]++;
                        std::memset(&params[entry_idx], 0, sizeof(CUfileIOParams_t));
                        params[entry_idx].mode = CUFILE_BATCH;
                        params[entry_idx].fh = ctx.cufile_handles[d][fh_idx];
                        params[entry_idx].opcode = CUFILE_READ;
                        params[entry_idx].u.batch.devPtr_base = dest + off;
                        params[entry_idx].u.batch.file_offset = file_off + off;
                        params[entry_idx].u.batch.devPtr_offset = 0;
                        params[entry_idx].u.batch.size = chunk;
                        params[entry_idx].cookie = nullptr;
                        entry_idx++;
                    }
                };

                for (size_t lp = 0; lp < n_large_pages; lp++) {
                    size_t lp_first = pages_done + lp * pages_per_io;
                    size_t lp_count = std::min(pages_per_io, batch_pages - lp * pages_per_io);

                    for (size_t d = 0; d < num_devices; d++) {
                        char *dev_io_base = io_base + lp * large_page_size
                                          + d * (large_page_size / num_devices);

                        size_t io_buf_off = 0;       // running offset within device sub-region
                        size_t run_file_offset = 0;
                        size_t run_io_start = 0;
                        size_t run_size = 0;
                        bool in_run = false;

                        for (size_t k = 0; k < lp_count; k++) {
                            size_t page_rel = page_indices[current_idx + lp_first + k];
                            uint64_t page_id = finfo.start_page_id + page_rel;
                            if (page_id % num_devices != d) continue;

                            size_t this_disk_size;
                            size_t this_file_offset;
                            if (is_compressed) {
                                this_disk_size = roundup4096(
                                    finfo.compressed_page_sizes[page_rel]);
                                this_file_offset = finfo.compressed_offsets[page_rel];
                            } else {
                                this_disk_size = page_size;
                                uint64_t device_page_id = page_id / num_devices;
                                this_file_offset = device_page_id * page_size;
                            }

                            if (in_run && this_file_offset == run_file_offset + run_size) {
                                // Contiguous with current run — extend
                                run_size += this_disk_size;
                            } else {
                                // Flush previous run (segment boundary or first page)
                                if (in_run) {
                                    flush_run(d, dev_io_base + run_io_start,
                                              run_file_offset, run_size);
                                }
                                // Start new run
                                run_file_offset = this_file_offset;
                                run_io_start = io_buf_off;
                                run_size = this_disk_size;
                                in_run = true;
                            }
                            io_buf_off += this_disk_size;
                        }

                        // Flush last run
                        if (in_run) {
                            flush_run(d, dev_io_base + run_io_start,
                                      run_file_offset, run_size);
                        }
                    }
                }

                // Submit in chunks (GDS batch handle limit)
                for (size_t si = 0; si < entry_idx; si += batch_capacity) {
                    size_t chunk = std::min(batch_capacity, entry_idx - si);
                    submit_and_wait_batch(ctx, params.data() + si, chunk);
                }

                // ── Scatter: io_buf → data_buf + decompress ──
                size_t decomp_count = 0;
                for (size_t lp = 0; lp < n_large_pages; lp++) {
                    size_t lp_first = pages_done + lp * pages_per_io;
                    size_t lp_count = std::min(pages_per_io, batch_pages - lp * pages_per_io);

                    // Track per-device read offset within io_buf sub-region
                    size_t dev_off[MAX_GDS_DEVICES] = {};

                    for (size_t k = 0; k < lp_count; k++) {
                        size_t page_rel = page_indices[current_idx + lp_first + k];
                        uint64_t page_id = finfo.start_page_id + page_rel;
                        size_t d = page_id % num_devices;

                        char *io_src = io_base + lp * large_page_size
                                     + d * (large_page_size / num_devices)
                                     + dev_off[d];
                        size_t dest_page_idx = lp_first + k;  // page index within cycle
                        char *dst = data_base + dest_page_idx * page_size;

                        if (is_compressed) {
                            size_t cs = finfo.compressed_page_sizes[page_rel];
                            if (cs < page_size) {
                                // Queue for nvCOMP batch decompression
                                ctx.h_nvcomp_comp_ptrs[decomp_count]    = io_src;
                                ctx.h_nvcomp_comp_sizes[decomp_count]   = cs;
                                ctx.h_nvcomp_decomp_ptrs[decomp_count]  = dst;
                                ctx.h_nvcomp_decomp_sizes[decomp_count] = page_size;
                                decomp_count++;
                            } else {
                                // Incompressible page: direct copy
                                CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                                    cudaMemcpyDeviceToDevice, stream));
                            }
                            dev_off[d] += roundup4096(cs);
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                                cudaMemcpyDeviceToDevice, stream));
                            dev_off[d] += page_size;
                        }
                    }
                }

                if (decomp_count > 0) {
                    nvcomp_decompress_batch(
                        finfo.compression_method, ctx, decomp_count, page_size, stream);
                } else {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }

                pages_done += batch_pages;
            }
        }

        // ── Q6 kernel (data_buf is contiguous per field) ──
        {
            uint64_t cycle_nrecs = (uint64_t)pages_this_cycle * capacity_per_page;
            q6_col_vardate(
                ctx.data_buf[L_SHIPDATE_IDX],
                ctx.data_buf[L_QUANTITY_IDX],
                ctx.data_buf[L_DISCOUNT_IDX],
                ctx.data_buf[L_EXTENDEDPRICE_IDX],
                pages_this_cycle,
                page_size,
                cycle_nrecs,
                ctx.d_q6_revenue,
                stream,
                config.q6_sd_low,
                config.q6_sd_high
            );
        }

        current_idx += pages_this_cycle;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ============================================================
// Worker thread (sync) — uses cuFileRead instead of batch API
// Fixed batch of GDS_SYNC_BATCH_PAGES (1024) pages per I/O round.
// io_multiplicity / large_page_size are ignored; io_buf is sized
// to hold GDS_SYNC_BATCH_PAGES pages with a flat per-device layout.
// ============================================================
static void gds_worker_sync_thread(
    GDSThreadContext &ctx,
    const std::vector<FieldPageInfo> &fields,
    std::span<const size_t> page_indices)
{
    mb_cuda_set_context(ctx.cuda_ctx);

    const auto &config = ctx.config;
    const size_t num_devices = config.fds.size();
    const size_t page_size = config.page_size;

    // Sync mode constants (independent of -i / -L)
    const size_t pages_per_batch = GDS_SYNC_BATCH_PAGES;
    const size_t io_region_size = pages_per_batch * page_size;  // io_buf total
    const size_t dev_region_size = io_region_size / num_devices; // per-device sub-region
    const size_t accumulate_pages = pages_per_batch * config.accumulate_rounds;

    // sizeof(pag_head) == 12
    const uint32_t capacity_per_page = (page_size - 12) / sizeof(int32_t);

    // Field index mapping for Q6
    const size_t L_SHIPDATE_IDX = 0;
    const size_t L_QUANTITY_IDX = 1;
    const size_t L_EXTENDEDPRICE_IDX = 2;
    const size_t L_DISCOUNT_IDX = 3;

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    size_t next_fh[MAX_GDS_DEVICES] = {};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const size_t total_pages = page_indices.size();
    size_t current_idx = 0;

    while (current_idx < total_pages) {
        size_t pages_this_cycle = std::min(accumulate_pages, total_pages - current_idx);

        // ── I/O + decompress phase: each field in turn ──
        for (size_t fi = 0; fi < fields.size(); fi++) {
            const FieldPageInfo &finfo = fields[fi];
            bool is_compressed = (finfo.compression_method != CompressionMethod::NONE);

            size_t field_pages = pages_this_cycle;

            char *data_base = static_cast<char *>(ctx.data_buf[fi]);
            char *io_base = static_cast<char *>(ctx.io_buf);

            size_t pages_done = 0;
            while (pages_done < field_pages) {
                size_t batch_pages = std::min(pages_per_batch, field_pages - pages_done);

                // ── Sync I/O: read contiguous runs via cuFileRead ──
                memset(next_fh, 0, num_devices * sizeof(size_t));

                // Helper: flush a contiguous run via synchronous cuFileRead
                auto flush_run = [&](size_t d, char *dest, size_t file_off, size_t size) {
                    size_t fh_idx = next_fh[d] % config.handles_per_thread;
                    next_fh[d]++;
                    off_t buf_offset = dest - io_base;
                    ssize_t nread = cuFileRead(ctx.cufile_handles[d][fh_idx],
                                               ctx.io_buf, size, file_off, buf_offset);
                    if (nread < 0 || static_cast<size_t>(nread) != size) {
                        std::cerr << "cuFileRead failed: thread=" << ctx.thread_id
                                  << " device=" << d
                                  << " size=" << size
                                  << " file_off=" << file_off
                                  << " buf_offset=" << buf_offset
                                  << " nread=" << nread << std::endl;
                    } else {
                        ctx.bytes_read += nread;
                        ctx.ios_completed++;
                    }
                };

                for (size_t d = 0; d < num_devices; d++) {
                    char *dev_io_base = io_base + d * dev_region_size;

                    size_t io_buf_off = 0;
                    size_t run_file_offset = 0;
                    size_t run_io_start = 0;
                    size_t run_size = 0;
                    bool in_run = false;

                    for (size_t k = 0; k < batch_pages; k++) {
                        size_t page_rel = page_indices[current_idx + pages_done + k];
                        uint64_t page_id = finfo.start_page_id + page_rel;
                        if (page_id % num_devices != d) continue;

                        size_t this_disk_size;
                        size_t this_file_offset;
                        if (is_compressed) {
                            this_disk_size = roundup4096(
                                finfo.compressed_page_sizes[page_rel]);
                            this_file_offset = finfo.compressed_offsets[page_rel];
                        } else {
                            this_disk_size = page_size;
                            uint64_t device_page_id = page_id / num_devices;
                            this_file_offset = device_page_id * page_size;
                        }

                        if (in_run && this_file_offset == run_file_offset + run_size) {
                            run_size += this_disk_size;
                        } else {
                            if (in_run) {
                                flush_run(d, dev_io_base + run_io_start,
                                          run_file_offset, run_size);
                            }
                            run_file_offset = this_file_offset;
                            run_io_start = io_buf_off;
                            run_size = this_disk_size;
                            in_run = true;
                        }
                        io_buf_off += this_disk_size;
                    }

                    if (in_run) {
                        flush_run(d, dev_io_base + run_io_start,
                                  run_file_offset, run_size);
                    }
                }

                // ── Scatter: io_buf → data_buf + decompress ──
                size_t decomp_count = 0;
                size_t dev_off[MAX_GDS_DEVICES] = {};

                for (size_t k = 0; k < batch_pages; k++) {
                    size_t page_rel = page_indices[current_idx + pages_done + k];
                    uint64_t page_id = finfo.start_page_id + page_rel;
                    size_t d = page_id % num_devices;

                    char *io_src = io_base + d * dev_region_size + dev_off[d];
                    size_t dest_page_idx = pages_done + k;
                    char *dst = data_base + dest_page_idx * page_size;

                    if (is_compressed) {
                        size_t cs = finfo.compressed_page_sizes[page_rel];
                        if (cs < page_size) {
                            ctx.h_nvcomp_comp_ptrs[decomp_count]    = io_src;
                            ctx.h_nvcomp_comp_sizes[decomp_count]   = cs;
                            ctx.h_nvcomp_decomp_ptrs[decomp_count]  = dst;
                            ctx.h_nvcomp_decomp_sizes[decomp_count] = page_size;
                            decomp_count++;
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                                cudaMemcpyDeviceToDevice, stream));
                        }
                        dev_off[d] += roundup4096(cs);
                    } else {
                        CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                            cudaMemcpyDeviceToDevice, stream));
                        dev_off[d] += page_size;
                    }
                }

                if (decomp_count > 0) {
                    nvcomp_decompress_batch(
                        finfo.compression_method, ctx, decomp_count, page_size, stream);
                } else {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }

                pages_done += batch_pages;
            }
        }

        // ── Q6 kernel (data_buf is contiguous per field) ──
        {
            uint64_t cycle_nrecs = (uint64_t)pages_this_cycle * capacity_per_page;
            q6_col_vardate(
                ctx.data_buf[L_SHIPDATE_IDX],
                ctx.data_buf[L_QUANTITY_IDX],
                ctx.data_buf[L_DISCOUNT_IDX],
                ctx.data_buf[L_EXTENDEDPRICE_IDX],
                pages_this_cycle,
                page_size,
                cycle_nrecs,
                ctx.d_q6_revenue,
                stream,
                config.q6_sd_low,
                config.q6_sd_high
            );
        }

        current_idx += pages_this_cycle;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}


// (Revenue query workers removed — tpch_revenue now uses tile execution)


namespace Gidp {

// Main-thread kernel launch counter (reset per query, aggregated with worker counts).
static size_t s_kernel_launches;

struct GolapBatchCookie
{
    void *buf_dev;
    size_t idx;
    size_t pagid;
};

struct GolapThreadArgs
{
    size_t thrid;
    CUcontext ctx;
    size_t page_size;
    //size_t sub_page_size;
    size_t page_start_index;
    size_t page_id_final;
    //size_t subpagid_final;
    //size_t nrows;
    //size_t nrows_final;
    struct TPCHTableMetadata &metadata;
    std::vector<CUfileHandle_t> cufile_handles;
    CUfileBatchHandle_t batch_idp;
    std::vector<uint64_t>::iterator page_vec_begin;
    std::vector<uint64_t>::iterator page_vec_end;
    std::vector<uint64_t> pagid_vec;
    std::vector<void *> buf_dev_vec;
    void * buf_out_dev;
    void * buf_out_host;
    void ** arr_buf_dev;
    void ** arr_buf_host;
    std::vector<CUfileIOParams_t> batch_params_vec;
    std::vector<CUfileIOEvents_t> batch_events_vec;
    std::vector<GolapBatchCookie> batch_cookie_vec;

    //std::vector<uint64_t> compressed_page_sizes_vec;
    //std::vector<uint64_t> compressed_subpage_sizes_vec;
    uint64_t kernel_exec_time = 0;
    uint64_t period_sec;
    size_t stats_nio;
};

struct Range {
    /* Represents a page range [start, end). */
    size_t start;
    size_t end;
};

using AlignedPtr = std::unique_ptr<void, void(*)(void*)>;

// ============================================================
// nvCOMP decompression context for gds_load_field_sync
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
    // Host staging arrays — set 0 (always allocated)
    void   **h_comp_ptrs   = nullptr;
    void   **h_decomp_ptrs = nullptr;
    size_t  *h_comp_sizes  = nullptr;
    size_t  *h_decomp_sizes = nullptr;
    // Host staging arrays — set 1 (double-buffer, nullptr if not used)
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

// Allocate double-buffer set 1 for host staging arrays (for ZM selective path)
static void nvcomp_decompctx_alloc_db(NvcompDecompCtx &ctx, size_t max_batch) {
    CUDA_CHECK(cudaMallocHost(&ctx.h_comp_ptrs_1,    max_batch * sizeof(void *)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_decomp_ptrs_1,  max_batch * sizeof(void *)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_comp_sizes_1,   max_batch * sizeof(size_t)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_decomp_sizes_1, max_batch * sizeof(size_t)));
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

// Decompress a batch of pages using the NvcompDecompCtx.
// h_set selects which host staging buffer set to use for H2D:
//   0 = h_comp_ptrs / h_decomp_ptrs / etc. (default, always available)
//   1 = h_comp_ptrs_1 / h_decomp_ptrs_1 / etc. (double-buffer set)
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
// Zero only the 12-byte page header (nalloc, watermark, lfreespace)
// of every page in buf.  Active pages will be overwritten by IO;
// inactive pages keep nalloc=0 so the scan kernel skips them.
// Replaces full-buffer cudaMemset (~9 GiB) with ~27 KB of writes.
// ────────────────────────────────────────────────────────────
__global__ void zero_page_headers_kernel(char *buf, uint32_t npages, uint32_t page_size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < npages) {
        uint32_t *hdr = reinterpret_cast<uint32_t *>(buf + (uint64_t)tid * page_size);
        hdr[0] = 0;  // nalloc
        hdr[1] = 0;  // watermark
        hdr[2] = 0;  // lfreespace
    }
}

// Compute batch-relative prefix_sum on GPU from a pre-uploaded full prefix_sum.
// batch_ps[i] = full_ps[pg + i + 1] - full_ps[pg]  for i in [0, bnp)
__global__ void compute_batch_ps_kernel(const uint64_t *full_ps, uint64_t *batch_ps,
                                         uint32_t pg, uint32_t bnp) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bnp) {
        batch_ps[i] = full_ps[pg + i + 1] - full_ps[pg];
    }
}

// ============================================================
// GDS selective page loader (sync mode) — loads only specified pages
// of a single field.  Pages are given as an index array (sorted).
// Batches up to GDS_SYNC_BATCH_PAGES pages for I/O coalescing
// and nvCOMP batch decompression.
// ============================================================
static void gds_load_field_sync_selective(
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const std::vector<std::vector<CUfileHandle_t>> &cufile_handles,
    void *io_buf,    // registered GPU buffer (batch_size * page_size)
    void *data_buf,  // destination GPU buffer (npages * page_size)
    NvcompDecompCtx *nvctx,  // nullptr if no compression support needed
    cudaStream_t stream,
    size_t &bytes_read, size_t &ios_completed,
    size_t &kernel_launches,
    const uint32_t *active_pages, size_t num_active_pages,
    size_t output_page_offset = 0,
    bool async_final = false,
    size_t batch_pages = GDS_SYNC_BATCH_PAGES);

static void gds_load_field_pipelined(
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const std::vector<std::vector<CUfileHandle_t>> &cufile_handles,
    void *set_bufs[2],
    void *data_buf,
    NvcompDecompCtx *nvctx,
    cudaStream_t io_stream,
    cudaStream_t nvcomp_stream,
    cudaEvent_t io_done_event,
    cudaEvent_t set_done_events[2],
    size_t &bytes_read, size_t &ios_completed,
    size_t &kernel_launches,
    size_t range_start, size_t range_npages,
    size_t output_page_offset,
    size_t io_sub_batch_pages,
    size_t nvcomp_batch_pages,
    size_t device_start = 0);

static void gds_load_field_pipelined_selective(
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const std::vector<std::vector<CUfileHandle_t>> &cufile_handles,
    void *set_bufs[2],
    void *data_buf,
    NvcompDecompCtx *nvctx,
    cudaStream_t io_stream,
    cudaStream_t nvcomp_stream,
    cudaEvent_t io_done_event,
    cudaEvent_t set_done_events[2],
    size_t &bytes_read, size_t &ios_completed,
    size_t &kernel_launches,
    const uint32_t *active_pages, size_t num_active_pages,
    size_t output_page_offset,
    size_t io_sub_batch_pages,
    size_t nvcomp_batch_pages,
    size_t device_start = 0);

// ============================================================
// GDS field page loader (sync mode) — loads all pages of a single field
// Follows Q6 gds_worker_sync_thread I/O pattern (RAID0, coalescing)
// Supports both compressed and uncompressed pages.
// ============================================================
static void gds_load_field_sync(
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const std::vector<std::vector<CUfileHandle_t>> &cufile_handles,
    void *io_buf,    // registered GPU buffer (batch_size * page_size)
    void *data_buf,  // destination GPU buffer (npages * page_size)
    NvcompDecompCtx *nvctx,  // nullptr if no compression support needed
    cudaStream_t stream,
    size_t &bytes_read, size_t &ios_completed,
    size_t &kernel_launches,
    size_t range_start = 0, size_t range_npages = 0,  // 0 = all pages
    size_t output_page_offset = 0,
    bool async_final = false,
    size_t batch_pages = GDS_SYNC_BATCH_PAGES);

BenchmarkResult tpch_q6(BenchmarkOptions &options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    /* Trying to read the head of superpage */
    void *ptr;
    TPCHTableMetadata *metadatap;
    if (posix_memalign((void**)&ptr, 512, metadata_head_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, (void*)ptr, 0, metadata_head_size);

    {
        metadatap = reinterpret_cast<TPCHTableMetadata*>(ptr);
        TPCHTableMetadata &metadata_pre = *metadatap;

        std::cout << "=== TPCH Table Metadata ===" << std::endl;
        std::cout << "Page Size: " << metadata_pre.page_size << std::endl;

        const size_t page_size = metadata_pre.page_size;
        free(ptr);

        if (posix_memalign((void**)&ptr, 512, page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        /* reload metadata page for consistency */
        page_pread_host(fds, (void*)ptr, 0, page_size);
    }

    /* Reload metadata pointer */
    metadatap = reinterpret_cast<TPCHTableMetadata*>(ptr);
    /* Start to use metadata*/
    TPCHTableMetadata &metadata = *metadatap;
    superpage_set_constants(metadata.page_size);

    mb_cufile_driver_open();
    /* Check io_depth value does not exceed its max value */
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);
    if (options.io_multiplicity > props.max_batch_io_size)
    {
#if 0
        last_error_line = __LINE__;
        for (auto cufile_handle : cufile_handles)
        {
            mb_cufile_handle_deregister(cufile_handle);
        }
#endif
        mb_cufile_driver_close();
        std::cerr << "io_multiplicity exceeds max_batch_io_size" << std::endl;
        exit(EXIT_FAILURE);
    }

    constexpr size_t num_lineitem_cols = TPCH::Query::Q6::NUM_SCAN_TARGET_COLS;
    auto q6_cols_idx = TPCH::Query::Q6::SCAN_TARGET_COLS;
    std::vector<FieldPageInfo> lineitem_page_infos(num_lineitem_cols);
    prepare_fields_metadata(
        fds,
        metadata,
        metadata.page_size,
        q6_cols_idx,
        lineitem_page_infos);

    size_t min_npages = SIZE_MAX;
    size_t max_npages = 0;
    std::vector<size_t> vec_npages_per_field {};
    for (size_t i = 0; i < num_lineitem_cols; i++) {
        min_npages = std::min(lineitem_page_infos[i].npages, min_npages);
        max_npages = std::max(lineitem_page_infos[i].npages, max_npages);
        vec_npages_per_field.push_back(lineitem_page_infos[i].npages);
    }

    for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
        const FieldPageInfo &info = lineitem_page_infos[fi];
        std::cout << "  Field " << info.field_index
                  << ": start_page=" << info.start_page_id
                  << " npages=" << info.npages
                  << " compression=" << static_cast<int>(info.compression_method)
                  << std::endl;
    }

    if (max_npages == 0)
    {
        std::cout << "No pages to read." << std::endl;
        free_fields_metadata(lineitem_page_infos);
        free(ptr);
        for (int fd : fds) {
            close(fd);
        }
        exit(EXIT_SUCCESS);
        return BenchmarkResult{};
    }

    // Determine if any field is compressed
    bool any_compressed = false;
    for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
        if (lineitem_page_infos[fi].compression_method != CompressionMethod::NONE) {
            any_compressed = true;
            break;
        }
    }

    // ── Zone map IO pruning: pre-allocate outside timing (Rule 4) ──
    std::vector<size_t> active_pages;
    bool zm_has_stats = false;
    size_t zm_sd_col = 0;
    uint64_t zm_stats_start_page = 0, zm_stats_npages_count = 0, zm_nstats = 0;
    int32_t *d_zm_sd_stats = nullptr;
    GdsZonemapCtx zm_ctx{};
    CUfileHandle_t zm_cufile_handles[MAX_GDS_DEVICES];
    int zm_dup_fds[MAX_GDS_DEVICES];
    bool zm_handles_open = false;

    if (options.enable_zonemap) {
        zm_sd_col = q6_cols_idx[0];
        zm_stats_start_page = metadata.table_lineitem_stats_start_page_ids[zm_sd_col];
        zm_stats_npages_count = metadata.table_lineitem_stats_npages[zm_sd_col];
        zm_nstats = metadata.table_lineitem_nstats[zm_sd_col];

        if (zm_nstats > 0 && zm_stats_start_page > 0) {
            zm_has_stats = true;
            CUDA_CHECK(cudaMalloc(&d_zm_sd_stats, zm_stats_npages_count * metadata.page_size));
            GDS_CHECK(cuFileBufRegister(d_zm_sd_stats, zm_stats_npages_count * metadata.page_size, 0));
            zm_ctx = gds_zonemap_ctx_create(min_npages);
            for (size_t d = 0; d < fds.size(); d++) {
                zm_dup_fds[d] = dup(fds[d]);
                zm_cufile_handles[d] = mb_cufile_handle_register(zm_dup_fds[d]);
            }
            zm_handles_open = true;
        } else {
            active_pages.resize(min_npages);
            std::iota(active_pages.begin(), active_pages.end(), size_t(0));
        }
    } else {
        active_pages.resize(min_npages);
        std::iota(active_pages.begin(), active_pages.end(), size_t(0));
    }

    const size_t nthreads = options.nthreads;
    const size_t page_size = metadata.page_size;
    const size_t num_devices = fds.size();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── GDS setup (per-column-group loader resources, v10 pattern) ──
    // Each column gets io_thr_per_col dedicated loaders with double-buffered
    // IO-decomp pipeline.  All columns loaded simultaneously.
    //
    // ZM selective: use more loaders (lower per-thread IO count) with smaller
    // nvcomp batches.  io_sub_batch = nvcomp_batch for the selective path to
    // maximise IO coalescing (no run splitting at sub-batch boundaries).
    // Total set_buf budget: num_loaders × 2 × batch × page_size ≈ 4096 MB.
    const bool zm_pruned = options.enable_zonemap;

    constexpr size_t Q6_MAX_IO_THR_PER_COL    = 2;
    constexpr size_t Q6_IO_SUB_BATCH_PAGES     = 32;
    constexpr size_t Q6_NVCOMP_BATCH_PAGES     = 1024;

    const size_t io_thr_per_col = std::min(nthreads, Q6_MAX_IO_THR_PER_COL);
    size_t eff_nvcomp_batch = Q6_NVCOMP_BATCH_PAGES;
    const size_t num_loaders = io_thr_per_col * num_lineitem_cols;
    std::cout << "[Q6] nthreads=" << nthreads
              << " io_thr_per_col=" << io_thr_per_col
              << " num_loaders=" << num_loaders << std::endl;

    struct Q6LoaderCtx {
        std::vector<std::vector<CUfileHandle_t>> cufile_handles;
        std::vector<std::vector<int>> dup_fds;
        void *set_buf[2] = {};               // 2 registered bufs for double-buffering
        cudaStream_t io_stream = nullptr;     // for D2D scatter
        cudaStream_t nvcomp_stream = nullptr; // for nvCOMP
        cudaEvent_t io_done_event;            // io→nvcomp sync
        cudaEvent_t set_done_events[2];       // per-set nvCOMP completion
        NvcompDecompCtx nvctx;
        size_t bytes_read = 0;
        size_t ios_completed = 0;
        size_t kernel_launches = 0;
    };
    std::vector<Q6LoaderCtx> loaders(num_loaders);
    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    // Phase 1: create CUfileHandles, streams, events (set_buf/nvcomp deferred for budget)
    for (size_t t = 0; t < num_loaders; t++) {
        auto &L = loaders[t];
        L.cufile_handles.resize(num_devices);
        L.dup_fds.resize(num_devices);
        for (size_t d = 0; d < num_devices; d++) {
            int duped = dup(fds[d]);
            if (duped < 0) { std::cerr << "dup failed" << std::endl; exit(EXIT_FAILURE); }
            L.dup_fds[d].push_back(duped);
            L.cufile_handles[d].push_back(mb_cufile_handle_register(duped));
        }
        CUDA_CHECK(cudaStreamCreate(&L.io_stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&L.io_done_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&L.set_done_events[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&L.set_done_events[1], cudaEventDisableTiming));
        if (any_compressed) {
            CUDA_CHECK(cudaStreamCreate(&L.nvcomp_stream));
        }
    }

    const size_t npages = min_npages;
    constexpr size_t Q6_TILE_PAGES = 512;  // Fixed batch size for streaming

    // Field index mapping for Q6
    const size_t L_SHIPDATE_IDX = 0;
    const size_t L_QUANTITY_IDX = 1;
    const size_t L_EXTENDEDPRICE_IDX = 2;
    const size_t L_DISCOUNT_IDX = 3;

    // ── Per-column data buffers (fixed batch size) ──
    void *tile_data_buf[num_lineitem_cols];
    for (size_t i = 0; i < num_lineitem_cols; i++)
        tile_data_buf[i] = mb_cuda_alloc(Q6_TILE_PAGES * page_size);

    int64_t *d_q6_revenue = static_cast<int64_t*>(mb_cuda_alloc(sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_q6_revenue, 0, sizeof(int64_t)));

    // Phase 2: allocate per-loader set_buf with dynamic budget (Rule 4: gpu_mem ≤ 40 GiB)
    {
        static constexpr size_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_before_loaders = 0;
        cudaMemGetInfo(&gpu_free_before_loaders, &gpu_total_dummy);
        size_t non_loader_bytes = gpu_free_start - gpu_free_before_loaders;
        size_t min_batch = Q6_IO_SUB_BATCH_PAGES;
        eff_nvcomp_batch = min_batch;
        if (GPU_MEM_BUDGET > non_loader_bytes) {
            size_t nvcomp_per_loader = 2 * 1024 * 1024;
            size_t budget_for_loaders = GPU_MEM_BUDGET - non_loader_bytes;
            size_t per_loader_budget = budget_for_loaders / num_loaders;
            if (per_loader_budget > nvcomp_per_loader) {
                eff_nvcomp_batch = (per_loader_budget - nvcomp_per_loader) / (2 * page_size);
            }
            eff_nvcomp_batch = std::max(eff_nvcomp_batch, min_batch);
            eff_nvcomp_batch = std::min(eff_nvcomp_batch, (size_t)Q6_NVCOMP_BATCH_PAGES);
            eff_nvcomp_batch = (eff_nvcomp_batch / Q6_IO_SUB_BATCH_PAGES) * Q6_IO_SUB_BATCH_PAGES;
        }
        for (size_t t = 0; t < num_loaders; t++) {
            auto &L = loaders[t];
            for (size_t b = 0; b < 2; b++) {
                L.set_buf[b] = mb_cuda_alloc(eff_nvcomp_batch * page_size);
                GDS_CHECK(cuFileBufRegister(L.set_buf[b], eff_nvcomp_batch * page_size, 0));
            }
            if (any_compressed) {
                nvcomp_decompctx_alloc(L.nvctx, eff_nvcomp_batch, page_size, lineitem_page_infos);
                if (zm_pruned) {
                    nvcomp_decompctx_alloc_db(L.nvctx, eff_nvcomp_batch);
                }
            }
        }
        std::cout << "[Q6] Per-loader IO: 2x" << eff_nvcomp_batch << " pages ("
                  << (2 * eff_nvcomp_batch * page_size / (1024*1024)) << " MiB), "
                  << num_loaders << " loaders" << std::endl;
    }

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t golap_gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // ── Zone map IO pruning (GDS + GPU, inside timing per Rule 6) ──
    if (zm_has_stats) {
        gds_read_zonemap(zm_cufile_handles, num_devices,
                         zm_stats_start_page, zm_stats_npages_count,
                         page_size, d_zm_sd_stats);

        zm_ctx.h_preds[0].d_stats  = reinterpret_cast<int32_t*>(d_zm_sd_stats);
        zm_ctx.h_preds[0].nstats   = zm_nstats;
        zm_ctx.h_preds[0].pred_lo  = options.q6_sd_low;
        zm_ctx.h_preds[0].pred_hi  = options.q6_sd_high - 1;

        gds_zonemap_eval_async(zm_ctx, min_npages, 1, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (size_t pg = 0; pg < min_npages; pg++)
            if (zm_ctx.h_mask[pg]) active_pages.push_back(pg);

        std::cout << "[ZONEMAP] L_SHIPDATE GPU pruning: " << active_pages.size()
                  << " / " << min_npages << " pages active ("
                  << (min_npages - active_pages.size()) << " pruned)" << std::endl;
    }

    bool use_zonemap = options.enable_zonemap;
    bool use_selective = use_zonemap && !active_pages.empty();

    size_t num_tiles = (npages + Q6_TILE_PAGES - 1) / Q6_TILE_PAGES;
    std::cout << "[Q6] Tile execution: " << num_tiles << " tiles of "
              << Q6_TILE_PAGES << " pages" << std::endl;

    std::vector<std::thread> all_threads;
    all_threads.reserve(num_loaders);

    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        size_t p_lo = tile_idx * Q6_TILE_PAGES;
        size_t tile_np = std::min(Q6_TILE_PAGES, npages - p_lo);

        // Zone map: check active pages in this tile
        std::vector<uint32_t> tile_active_pages;
        if (use_selective) {
            auto it_lo = std::lower_bound(active_pages.begin(), active_pages.end(), (uint32_t)p_lo);
            auto it_hi = std::lower_bound(active_pages.begin(), active_pages.end(), (uint32_t)(p_lo + tile_np));
            for (auto it = it_lo; it != it_hi; ++it)
                tile_active_pages.push_back(*it);
            if (tile_active_pages.empty()) continue;
        }

        bool selective_partial = use_selective && tile_active_pages.size() < tile_np;

        // Zero page headers for selective loading (inactive pages → nalloc=0).
        // Only the 12-byte header is needed; active pages are overwritten by IO.
        if (selective_partial) {
            unsigned zblk = (tile_np + 255) / 256;
            for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
                zero_page_headers_kernel<<<zblk, 256, 0, stream>>>(
                    static_cast<char *>(tile_data_buf[fi]), tile_np, page_size);
                s_kernel_launches++;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        if (selective_partial) {
            // ── Selective loading (zone map) — column-parallel, per-thread nvCOMP ──
            // Use nvcomp_batch_pages as io_sub_batch for full-batch IO coalescing:
            // sparse pages break runs at sub-batch boundaries, so a single large
            // sub-batch maximizes the chance of merging adjacent disk pages.
            all_threads.clear();
            for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
                size_t loader_base = fi * io_thr_per_col;
                size_t n_active = tile_active_pages.size();
                size_t pages_per_thr = (n_active + io_thr_per_col - 1) / io_thr_per_col;
                for (size_t t = 0; t < io_thr_per_col; t++) {
                    size_t pg_start = t * pages_per_thr;
                    if (pg_start >= n_active) break;
                    size_t pg_count = std::min(pages_per_thr, n_active - pg_start);
                    size_t li = loader_base + t;
                    all_threads.emplace_back([&, fi, li, pg_start, pg_count]() {
                        auto &L = loaders[li];
                        mb_cuda_set_context(cuda_ctx_handle);
                        NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                        gds_load_field_pipelined_selective(
                            lineitem_page_infos[fi], page_size, num_devices,
                            L.cufile_handles, L.set_buf, tile_data_buf[fi],
                            nv, L.io_stream, L.nvcomp_stream,
                            L.io_done_event, L.set_done_events,
                            L.bytes_read, L.ios_completed, L.kernel_launches,
                            tile_active_pages.data() + pg_start, pg_count, p_lo,
                            eff_nvcomp_batch,
                            eff_nvcomp_batch,
                            li);
                    });
                }
            }
            for (auto &th : all_threads) th.join();
            for (size_t i = 0; i < num_loaders; i++) {
                CUDA_CHECK(cudaStreamSynchronize(loaders[i].io_stream));
                if (loaders[i].nvcomp_stream)
                    CUDA_CHECK(cudaStreamSynchronize(loaders[i].nvcomp_stream));
            }
        } else {
            // ── Per-thread IO-decomp pipeline (column-parallel, v10 pattern) ──
            all_threads.clear();
            for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
                size_t loader_base = fi * io_thr_per_col;
                size_t pages_per_thr = (tile_np + io_thr_per_col - 1) / io_thr_per_col;
                for (size_t t = 0; t < io_thr_per_col; t++) {
                    size_t start = p_lo + t * pages_per_thr;
                    size_t end = std::min(start + pages_per_thr, p_lo + tile_np);
                    if (start >= end) break;
                    size_t count = end - start;
                    size_t li = loader_base + t;
                    all_threads.emplace_back([&, fi, li, start, count]() {
                        auto &L = loaders[li];
                        mb_cuda_set_context(cuda_ctx_handle);
                        NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                        gds_load_field_pipelined(
                            lineitem_page_infos[fi], page_size, num_devices,
                            L.cufile_handles, L.set_buf,
                            tile_data_buf[fi],
                            nv, L.io_stream, L.nvcomp_stream,
                            L.io_done_event, L.set_done_events,
                            L.bytes_read, L.ios_completed, L.kernel_launches,
                            start, count, p_lo,
                            Q6_IO_SUB_BATCH_PAGES,
                            eff_nvcomp_batch,
                            li);
                    });
                }
            }
            for (auto &th : all_threads) th.join();

            // Sync all loader streams
            for (size_t i = 0; i < num_loaders; i++) {
                CUDA_CHECK(cudaStreamSynchronize(loaders[i].io_stream));
                if (loaders[i].nvcomp_stream)
                    CUDA_CHECK(cudaStreamSynchronize(loaders[i].nvcomp_stream));
            }
        }

        // Kernel call (revenue accumulates via atomicAdd across tiles)
        // Use tile_np * capacity as iteration bound (not nrecs_lineitem):
        // the paged kernel maps total_idx → (page_idx, rec_idx) via capacity,
        // so we must iterate the full slot space to reach all pages.
        uint32_t capacity = (page_size - 12) / 4;
        q6_col_vardate(
            tile_data_buf[L_SHIPDATE_IDX],
            tile_data_buf[L_QUANTITY_IDX],
            tile_data_buf[L_DISCOUNT_IDX],
            tile_data_buf[L_EXTENDEDPRICE_IDX],
            tile_np, page_size, (uint64_t)tile_np * capacity,
            d_q6_revenue, stream,
            options.q6_sd_low, options.q6_sd_high);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    int64_t h_q6_revenue = 0;
    CUDA_CHECK(cudaMemcpy(&h_q6_revenue, d_q6_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::cout << "TPCH Q6 total revenue: " << h_q6_revenue << std::endl;

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(total_end - total_start).count();

    // ── Aggregate I/O stats ──
    size_t nios = 0, bytes_read = 0;
    for (size_t t = 0; t < num_loaders; t++) {
        nios += loaders[t].ios_completed;
        bytes_read += loaders[t].bytes_read;
        s_kernel_launches += loaders[t].kernel_launches;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << nios << std::endl;
    std::cout << "Total bytes read: " << bytes_read << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    for (size_t i = 0; i < num_lineitem_cols; i++)
        mb_cuda_free(tile_data_buf[i]);
    mb_cuda_free(d_q6_revenue);
    CUDA_CHECK(cudaStreamDestroy(stream));

    size_t total_pages = 0;
    for (const auto &fi : lineitem_page_infos) total_pages += fi.npages;

    for (size_t t = 0; t < num_loaders; t++) {
        auto &L = loaders[t];
        for (size_t b = 0; b < 2; b++) {
            cuFileBufDeregister(L.set_buf[b]);
            mb_cuda_free(L.set_buf[b]);
        }
        CUDA_CHECK(cudaStreamDestroy(L.io_stream));
        CUDA_CHECK(cudaEventDestroy(L.io_done_event));
        CUDA_CHECK(cudaEventDestroy(L.set_done_events[0]));
        CUDA_CHECK(cudaEventDestroy(L.set_done_events[1]));
        if (any_compressed) {
            nvcomp_decompctx_free(L.nvctx);
            CUDA_CHECK(cudaStreamDestroy(L.nvcomp_stream));
        }
        for (size_t d = 0; d < num_devices; d++) {
            for (size_t h = 0; h < L.cufile_handles[d].size(); h++) {
                cuFileHandleDeregister(L.cufile_handles[d][h]);
                close(L.dup_fds[d][h]);
            }
        }
    }

    if (zm_handles_open) {
        cuFileBufDeregister(d_zm_sd_stats);
        cudaFree(d_zm_sd_stats);
        gds_zonemap_ctx_destroy(zm_ctx);
        for (size_t d = 0; d < num_devices; d++) {
            cuFileHandleDeregister(zm_cufile_handles[d]);
            close(zm_dup_fds[d]);
        }
    }

    free_fields_metadata(lineitem_page_infos);
    mb_cufile_driver_close();

    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = nios,
        .read_bytes = (uint64_t)bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = collect_compression_methods({lineitem_page_infos}),
        .gpu_mem_bytes = golap_gpu_mem_bytes,
        .gpu_app_bytes = golap_gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// Revenue query — tile execution (same pattern as Q6)
// ============================================================
BenchmarkResult tpch_revenue(BenchmarkOptions &options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    TPCHTableMetadata *metadatap;
    if (posix_memalign((void**)&ptr, 512, metadata_head_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, (void*)ptr, 0, metadata_head_size);

    {
        metadatap = reinterpret_cast<TPCHTableMetadata*>(ptr);
        TPCHTableMetadata &metadata_pre = *metadatap;

        std::cout << "=== TPCH Table Metadata ===" << std::endl;
        std::cout << "Page Size: " << metadata_pre.page_size << std::endl;
        std::cout << "nrecs_lineitem: " << metadata_pre.table_lineitem_nrows << std::endl;

        const size_t page_size = metadata_pre.page_size;
        free(ptr);

        if (posix_memalign((void**)&ptr, 512, page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, (void*)ptr, 0, page_size);
    }

    metadatap = reinterpret_cast<TPCHTableMetadata*>(ptr);
    TPCHTableMetadata &metadata = *metadatap;
    superpage_set_constants(metadata.page_size);

    mb_cufile_driver_open();
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);
    if (options.io_multiplicity > props.max_batch_io_size)
    {
        mb_cufile_driver_close();
        std::cerr << "io_multiplicity exceeds max_batch_io_size" << std::endl;
        exit(EXIT_FAILURE);
    }

    constexpr size_t num_lineitem_cols = TPCH::Query::Q6::NUM_SCAN_TARGET_COLS;
    auto rev_cols_idx = TPCH::Query::Q6::SCAN_TARGET_COLS;
    std::vector<FieldPageInfo> lineitem_page_infos(num_lineitem_cols);
    prepare_fields_metadata(
        fds,
        metadata,
        metadata.page_size,
        rev_cols_idx,
        lineitem_page_infos);

    size_t min_npages = SIZE_MAX;
    size_t max_npages = 0;
    for (size_t i = 0; i < num_lineitem_cols; i++) {
        min_npages = std::min(lineitem_page_infos[i].npages, min_npages);
        max_npages = std::max(lineitem_page_infos[i].npages, max_npages);
    }

    for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
        const FieldPageInfo &info = lineitem_page_infos[fi];
        std::cout << "  Field " << info.field_index
                  << ": start_page=" << info.start_page_id
                  << " npages=" << info.npages
                  << " compression=" << static_cast<int>(info.compression_method)
                  << std::endl;
    }

    if (max_npages == 0)
    {
        std::cout << "No pages to read." << std::endl;
        free_fields_metadata(lineitem_page_infos);
        free(ptr);
        for (int fd : fds) {
            close(fd);
        }
        exit(EXIT_SUCCESS);
        return BenchmarkResult{};
    }

    bool any_compressed = false;
    for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
        if (lineitem_page_infos[fi].compression_method != CompressionMethod::NONE) {
            any_compressed = true;
            break;
        }
    }

    // ── Zone map IO pruning: pre-allocate outside timing (Rule 4) ──
    std::vector<size_t> active_pages;
    bool zm_has_stats = false;
    size_t zm_sd_col = 0;
    uint64_t zm_stats_start_page = 0, zm_stats_npages_count = 0, zm_nstats = 0;
    int32_t *d_zm_sd_stats = nullptr;
    GdsZonemapCtx zm_ctx{};
    CUfileHandle_t zm_cufile_handles[MAX_GDS_DEVICES];
    int zm_dup_fds[MAX_GDS_DEVICES];
    bool zm_handles_open = false;

    if (options.enable_zonemap) {
        zm_sd_col = rev_cols_idx[0];
        zm_stats_start_page = metadata.table_lineitem_stats_start_page_ids[zm_sd_col];
        zm_stats_npages_count = metadata.table_lineitem_stats_npages[zm_sd_col];
        zm_nstats = metadata.table_lineitem_nstats[zm_sd_col];

        if (zm_nstats > 0 && zm_stats_start_page > 0) {
            zm_has_stats = true;
            CUDA_CHECK(cudaMalloc(&d_zm_sd_stats, zm_stats_npages_count * metadata.page_size));
            GDS_CHECK(cuFileBufRegister(d_zm_sd_stats, zm_stats_npages_count * metadata.page_size, 0));
            zm_ctx = gds_zonemap_ctx_create(min_npages);
            for (size_t d = 0; d < fds.size(); d++) {
                zm_dup_fds[d] = dup(fds[d]);
                zm_cufile_handles[d] = mb_cufile_handle_register(zm_dup_fds[d]);
            }
            zm_handles_open = true;
        } else {
            std::cout << "[ZONEMAP] No stats available for L_SHIPDATE, processing all pages." << std::endl;
            active_pages.resize(min_npages);
            std::iota(active_pages.begin(), active_pages.end(), size_t(0));
        }
    } else {
        active_pages.resize(min_npages);
        std::iota(active_pages.begin(), active_pages.end(), size_t(0));
    }

    const size_t nthreads = options.nthreads;
    const size_t page_size = metadata.page_size;
    const size_t num_devices = fds.size();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── GDS setup (per-column-group loader, Q6 pipelined pattern) ──
    const bool zm_pruned = options.enable_zonemap;

    constexpr size_t REV_MAX_IO_THR_PER_COL    = 2;
    constexpr size_t REV_IO_SUB_BATCH_PAGES     = 32;
    constexpr size_t REV_NVCOMP_BATCH_PAGES     = 1024;

    const size_t io_thr_per_col = std::min(nthreads, REV_MAX_IO_THR_PER_COL);
    size_t eff_nvcomp_batch = REV_NVCOMP_BATCH_PAGES;
    const size_t num_loaders = io_thr_per_col * num_lineitem_cols;
    std::cout << "[Revenue] nthreads=" << nthreads
              << " io_thr_per_col=" << io_thr_per_col
              << " num_loaders=" << num_loaders
              << " num_devices=" << num_devices << std::endl;

    struct RevLoaderCtx {
        std::vector<std::vector<CUfileHandle_t>> cufile_handles;
        std::vector<std::vector<int>> dup_fds;
        void *set_buf[2] = {};               // double-buffer
        cudaStream_t io_stream = nullptr;     // IO + D2D scatter
        cudaStream_t nvcomp_stream = nullptr; // nvCOMP decompression
        cudaEvent_t io_done_event;            // io→nvcomp sync
        cudaEvent_t set_done_events[2];       // per-set completion
        NvcompDecompCtx nvctx;
        size_t bytes_read = 0;
        size_t ios_completed = 0;
        size_t kernel_launches = 0;
    };
    std::vector<RevLoaderCtx> loaders(num_loaders);
    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    // Phase 1: create CUfileHandles, streams, events (set_buf/nvcomp deferred for budget)
    for (size_t t = 0; t < num_loaders; t++) {
        auto &L = loaders[t];
        L.cufile_handles.resize(num_devices);
        L.dup_fds.resize(num_devices);
        for (size_t d = 0; d < num_devices; d++) {
            int duped = dup(fds[d]);
            if (duped < 0) { std::cerr << "dup failed" << std::endl; exit(EXIT_FAILURE); }
            L.dup_fds[d].push_back(duped);
            L.cufile_handles[d].push_back(mb_cufile_handle_register(duped));
        }
        CUDA_CHECK(cudaStreamCreate(&L.io_stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&L.io_done_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&L.set_done_events[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&L.set_done_events[1], cudaEventDisableTiming));
        if (any_compressed) {
            CUDA_CHECK(cudaStreamCreate(&L.nvcomp_stream));
        }
    }

    const size_t npages = min_npages;
    // Budget-based tile size: maximize tile while reserving loader budget
    size_t REV_TILE_PAGES;
    {
        constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        uint64_t loader_reserve = (uint64_t)num_loaders *
            (2 * REV_NVCOMP_BATCH_PAGES * page_size + 2 * 1024 * 1024);
        uint64_t tile_budget = (loader_reserve < GPU_MEM_BUDGET)
            ? GPU_MEM_BUDGET - loader_reserve : 0;
        REV_TILE_PAGES = std::min(npages,
            std::max((size_t)512, (size_t)(tile_budget / (num_lineitem_cols * page_size))));
    }

    const size_t L_SHIPDATE_IDX = 0;
    const size_t L_QUANTITY_IDX = 1;
    const size_t L_EXTENDEDPRICE_IDX = 2;
    const size_t L_DISCOUNT_IDX = 3;

    // ── Per-column data buffers (tile-sized) ──
    void *tile_data_buf[num_lineitem_cols];
    for (size_t i = 0; i < num_lineitem_cols; i++)
        tile_data_buf[i] = mb_cuda_alloc(REV_TILE_PAGES * page_size);

    int64_t *d_revenue = static_cast<int64_t*>(mb_cuda_alloc(sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    const uint64_t *ref_ps = lineitem_page_infos[0].prefix_sum_nrecs;

    // Phase 2: allocate per-loader set_buf with dynamic budget
    {
        static constexpr size_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_before_loaders = 0;
        cudaMemGetInfo(&gpu_free_before_loaders, &gpu_total_dummy);
        size_t non_loader_bytes = gpu_free_start - gpu_free_before_loaders;
        size_t min_batch = REV_IO_SUB_BATCH_PAGES;
        eff_nvcomp_batch = min_batch;
        if (GPU_MEM_BUDGET > non_loader_bytes) {
            size_t nvcomp_per_loader = 2 * 1024 * 1024;
            size_t budget_for_loaders = GPU_MEM_BUDGET - non_loader_bytes;
            size_t per_loader_budget = budget_for_loaders / num_loaders;
            if (per_loader_budget > nvcomp_per_loader) {
                eff_nvcomp_batch = (per_loader_budget - nvcomp_per_loader) / (2 * page_size);
            }
            eff_nvcomp_batch = std::max(eff_nvcomp_batch, min_batch);
            eff_nvcomp_batch = std::min(eff_nvcomp_batch, (size_t)REV_NVCOMP_BATCH_PAGES);
            eff_nvcomp_batch = (eff_nvcomp_batch / REV_IO_SUB_BATCH_PAGES) * REV_IO_SUB_BATCH_PAGES;
        }
        for (size_t t = 0; t < num_loaders; t++) {
            auto &L = loaders[t];
            for (size_t b = 0; b < 2; b++) {
                L.set_buf[b] = mb_cuda_alloc(eff_nvcomp_batch * page_size);
                GDS_CHECK(cuFileBufRegister(L.set_buf[b], eff_nvcomp_batch * page_size, 0));
            }
            if (any_compressed) {
                nvcomp_decompctx_alloc(L.nvctx, eff_nvcomp_batch, page_size, lineitem_page_infos);
                if (zm_pruned) {
                    nvcomp_decompctx_alloc_db(L.nvctx, eff_nvcomp_batch);
                }
            }
        }
        std::cout << "[Revenue] Per-loader IO: 2x" << eff_nvcomp_batch << " pages ("
                  << (2 * eff_nvcomp_batch * page_size / (1024*1024)) << " MiB), "
                  << num_loaders << " loaders" << std::endl;
    }

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t golap_gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // ── Zone map IO pruning (GDS + GPU, inside timing per Rule 6) ──
    if (zm_has_stats) {
        gds_read_zonemap(zm_cufile_handles, num_devices,
                         zm_stats_start_page, zm_stats_npages_count,
                         page_size, d_zm_sd_stats);

        zm_ctx.h_preds[0].d_stats  = reinterpret_cast<int32_t*>(d_zm_sd_stats);
        zm_ctx.h_preds[0].nstats   = zm_nstats;
        zm_ctx.h_preds[0].pred_lo  = options.q6_sd_low;
        zm_ctx.h_preds[0].pred_hi  = options.q6_sd_high - 1;

        gds_zonemap_eval_async(zm_ctx, min_npages, 1, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (size_t pg = 0; pg < min_npages; pg++)
            if (zm_ctx.h_mask[pg]) active_pages.push_back(pg);

        std::cout << "[ZONEMAP] L_SHIPDATE GPU pruning: " << active_pages.size()
                  << " / " << min_npages << " pages active ("
                  << (min_npages - active_pages.size()) << " pruned)" << std::endl;
    }

    bool use_zonemap = options.enable_zonemap;
    bool use_selective = use_zonemap && !active_pages.empty();

    size_t num_tiles = (npages + REV_TILE_PAGES - 1) / REV_TILE_PAGES;
    std::cout << "[Revenue] Tile execution: " << num_tiles << " tiles of "
              << REV_TILE_PAGES << " pages" << std::endl;

    std::vector<std::thread> all_threads;
    all_threads.reserve(num_loaders);

    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        size_t p_lo = tile_idx * REV_TILE_PAGES;
        size_t tile_np = std::min(REV_TILE_PAGES, npages - p_lo);

        // Zone map: check active pages in this tile
        std::vector<uint32_t> tile_active_pages;
        if (use_selective) {
            auto it_lo = std::lower_bound(active_pages.begin(), active_pages.end(), (uint32_t)p_lo);
            auto it_hi = std::lower_bound(active_pages.begin(), active_pages.end(), (uint32_t)(p_lo + tile_np));
            for (auto it = it_lo; it != it_hi; ++it)
                tile_active_pages.push_back(*it);
            if (tile_active_pages.empty()) continue;
        }

        bool selective_partial = use_selective && tile_active_pages.size() < tile_np;

        // Zero page headers for selective loading (inactive pages → nalloc=0).
        if (selective_partial) {
            unsigned zblk = (tile_np + 255) / 256;
            for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
                zero_page_headers_kernel<<<zblk, 256, 0, stream>>>(
                    static_cast<char *>(tile_data_buf[fi]), tile_np, page_size);
                s_kernel_launches++;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        if (selective_partial) {
            // ── Selective loading (zone map) — column-parallel pipelined ──
            all_threads.clear();
            for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
                size_t loader_base = fi * io_thr_per_col;
                size_t n_active = tile_active_pages.size();
                size_t pages_per_thr = (n_active + io_thr_per_col - 1) / io_thr_per_col;
                for (size_t t = 0; t < io_thr_per_col; t++) {
                    size_t pg_start = t * pages_per_thr;
                    if (pg_start >= n_active) break;
                    size_t pg_count = std::min(pages_per_thr, n_active - pg_start);
                    size_t li = loader_base + t;
                    all_threads.emplace_back([&, fi, li, pg_start, pg_count]() {
                        auto &L = loaders[li];
                        mb_cuda_set_context(cuda_ctx_handle);
                        NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                        gds_load_field_pipelined_selective(
                            lineitem_page_infos[fi], page_size, num_devices,
                            L.cufile_handles, L.set_buf, tile_data_buf[fi],
                            nv, L.io_stream, L.nvcomp_stream,
                            L.io_done_event, L.set_done_events,
                            L.bytes_read, L.ios_completed, L.kernel_launches,
                            tile_active_pages.data() + pg_start, pg_count, p_lo,
                            eff_nvcomp_batch,
                            eff_nvcomp_batch,
                            li);
                    });
                }
            }
            for (auto &th : all_threads) th.join();
            for (size_t i = 0; i < num_loaders; i++) {
                CUDA_CHECK(cudaStreamSynchronize(loaders[i].io_stream));
                if (loaders[i].nvcomp_stream)
                    CUDA_CHECK(cudaStreamSynchronize(loaders[i].nvcomp_stream));
            }
        } else {
            // ── Per-thread IO-decomp pipeline (column-parallel, pipelined) ──
            all_threads.clear();
            for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
                size_t loader_base = fi * io_thr_per_col;
                size_t pages_per_thr = (tile_np + io_thr_per_col - 1) / io_thr_per_col;
                for (size_t t = 0; t < io_thr_per_col; t++) {
                    size_t start = p_lo + t * pages_per_thr;
                    size_t end = std::min(start + pages_per_thr, p_lo + tile_np);
                    if (start >= end) break;
                    size_t count = end - start;
                    size_t li = loader_base + t;
                    all_threads.emplace_back([&, fi, li, start, count]() {
                        auto &L = loaders[li];
                        mb_cuda_set_context(cuda_ctx_handle);
                        NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                        gds_load_field_pipelined(
                            lineitem_page_infos[fi], page_size, num_devices,
                            L.cufile_handles, L.set_buf,
                            tile_data_buf[fi],
                            nv, L.io_stream, L.nvcomp_stream,
                            L.io_done_event, L.set_done_events,
                            L.bytes_read, L.ios_completed, L.kernel_launches,
                            start, count, p_lo,
                            REV_IO_SUB_BATCH_PAGES,
                            eff_nvcomp_batch,
                            li);
                    });
                }
            }
            for (auto &th : all_threads) th.join();

            // Sync all loader streams
            for (size_t i = 0; i < num_loaders; i++) {
                CUDA_CHECK(cudaStreamSynchronize(loaders[i].io_stream));
                if (loaders[i].nvcomp_stream)
                    CUDA_CHECK(cudaStreamSynchronize(loaders[i].nvcomp_stream));
            }
        }

        // Kernel call (revenue accumulates via atomicAdd across tiles)
        uint32_t capacity = (page_size - 12) / 4;
        q6_col_vardate(
            tile_data_buf[L_SHIPDATE_IDX],
            tile_data_buf[L_QUANTITY_IDX],
            tile_data_buf[L_DISCOUNT_IDX],
            tile_data_buf[L_EXTENDEDPRICE_IDX],
            tile_np, page_size, (uint64_t)tile_np * capacity,
            d_revenue, stream,
            options.q6_sd_low, options.q6_sd_high,
            options.disable_other_filters ? 0 : 5,
            options.disable_other_filters ? INT32_MAX : 7,
            options.disable_other_filters ? options.revenue_qt_max : 24);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    int64_t h_revenue = 0;
    CUDA_CHECK(cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::cout << "Revenue total: " << h_revenue << std::endl;

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(total_end - total_start).count();

    // ── Aggregate I/O stats ──
    size_t nios = 0, bytes_read = 0;
    for (size_t t = 0; t < num_loaders; t++) {
        nios += loaders[t].ios_completed;
        bytes_read += loaders[t].bytes_read;
        s_kernel_launches += loaders[t].kernel_launches;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << nios << std::endl;
    std::cout << "Total bytes read: " << bytes_read << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    for (size_t i = 0; i < num_lineitem_cols; i++)
        mb_cuda_free(tile_data_buf[i]);
    mb_cuda_free(d_revenue);
    CUDA_CHECK(cudaStreamDestroy(stream));

    size_t total_pages = 0;
    for (const auto &fi : lineitem_page_infos) total_pages += fi.npages;

    for (size_t t = 0; t < num_loaders; t++) {
        auto &L = loaders[t];
        for (size_t b = 0; b < 2; b++) {
            cuFileBufDeregister(L.set_buf[b]);
            mb_cuda_free(L.set_buf[b]);
        }
        CUDA_CHECK(cudaStreamDestroy(L.io_stream));
        CUDA_CHECK(cudaEventDestroy(L.io_done_event));
        CUDA_CHECK(cudaEventDestroy(L.set_done_events[0]));
        CUDA_CHECK(cudaEventDestroy(L.set_done_events[1]));
        if (any_compressed) {
            nvcomp_decompctx_free(L.nvctx);
            CUDA_CHECK(cudaStreamDestroy(L.nvcomp_stream));
        }
        for (size_t d = 0; d < num_devices; d++) {
            for (size_t h = 0; h < L.cufile_handles[d].size(); h++) {
                cuFileHandleDeregister(L.cufile_handles[d][h]);
                close(L.dup_fds[d][h]);
            }
        }
    }

    if (zm_handles_open) {
        cuFileBufDeregister(d_zm_sd_stats);
        cudaFree(d_zm_sd_stats);
        gds_zonemap_ctx_destroy(zm_ctx);
        for (size_t d = 0; d < num_devices; d++) {
            cuFileHandleDeregister(zm_cufile_handles[d]);
            close(zm_dup_fds[d]);
        }
    }

    free_fields_metadata(lineitem_page_infos);
    mb_cufile_driver_close();

    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = nios,
        .read_bytes = (uint64_t)bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = collect_compression_methods({lineitem_page_infos}),
        .gpu_mem_bytes = golap_gpu_mem_bytes,
        .gpu_app_bytes = golap_gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// GDS selective page loader (sync mode) — loads only specified pages
// of a single field.  Pages are given as an index array (sorted).
// Batches up to GDS_SYNC_BATCH_PAGES pages for I/O coalescing
// and nvCOMP batch decompression.
// ============================================================
static void gds_load_field_sync_selective(
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const std::vector<std::vector<CUfileHandle_t>> &cufile_handles,
    void *io_buf,    // registered GPU buffer (batch_size * page_size)
    void *data_buf,  // destination GPU buffer (npages * page_size)
    NvcompDecompCtx *nvctx,  // nullptr if no compression support needed
    cudaStream_t stream,
    size_t &bytes_read, size_t &ios_completed,
    size_t &kernel_launches,
    const uint32_t *active_pages, size_t num_active_pages,
    size_t output_page_offset,  // output offset: dst = data_base + (page_rel - offset) * page_size
    bool async_final,
    size_t batch_pages)
{
    if (num_active_pages == 0) return;

    const size_t pages_per_batch = batch_pages;
    const bool is_compressed = (field.compression_method != CompressionMethod::NONE);

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    char *io_base = static_cast<char *>(io_buf);
    char *data_base = static_cast<char *>(data_buf);

    size_t next_fh[MAX_GDS_DEVICES] = {};
    size_t batch_start = 0;

    while (batch_start < num_active_pages) {
        size_t batch_count = std::min(pages_per_batch, num_active_pages - batch_start);

        // ── Compute per-device data sizes for dynamic layout ──
        // Sparse pages may not distribute evenly across devices.
        size_t dev_data_size[MAX_GDS_DEVICES] = {};
        for (size_t k = 0; k < batch_count; k++) {
            size_t page_rel = active_pages[batch_start + k];
            uint64_t page_id = field.start_page_id + page_rel;
            size_t d = page_id % num_devices;
            if (is_compressed)
                dev_data_size[d] += roundup4096(field.compressed_page_sizes[page_rel]);
            else
                dev_data_size[d] += page_size;
        }
        size_t dev_io_start[MAX_GDS_DEVICES + 1] = {};
        for (size_t d = 0; d < num_devices; d++)
            dev_io_start[d + 1] = dev_io_start[d] + dev_data_size[d];

        // ── I/O: read by device, coalescing consecutive pages ──
        memset(next_fh, 0, num_devices * sizeof(size_t));

        auto flush_run = [&](size_t d, char *dest, size_t file_off, size_t size) {
            size_t fh_idx = next_fh[d] % cufile_handles[d].size();
            next_fh[d]++;
            off_t buf_offset = dest - io_base;
            ssize_t nread = cuFileRead(cufile_handles[d][fh_idx],
                                       io_buf, size, file_off, buf_offset);
            if (nread < 0 || static_cast<size_t>(nread) != size) {
                std::cerr << "cuFileRead failed: device=" << d
                          << " size=" << size << " file_off=" << file_off
                          << " nread=" << nread << std::endl;
            } else {
                bytes_read += nread;
                ios_completed++;
            }
        };

        for (size_t d = 0; d < num_devices; d++) {
            char *dev_io_base = io_base + dev_io_start[d];
            size_t io_buf_off = 0;
            size_t run_file_offset = 0;
            size_t run_io_start = 0;
            size_t run_size = 0;
            bool in_run = false;

            for (size_t k = 0; k < batch_count; k++) {
                size_t page_rel = active_pages[batch_start + k];
                uint64_t page_id = field.start_page_id + page_rel;
                if (page_id % num_devices != d) continue;

                size_t this_disk_size;
                size_t this_file_offset;
                if (is_compressed) {
                    this_disk_size = roundup4096(
                        field.compressed_page_sizes[page_rel]);
                    this_file_offset = field.compressed_offsets[page_rel];
                } else {
                    this_disk_size = page_size;
                    uint64_t device_page_id = page_id / num_devices;
                    this_file_offset = device_page_id * page_size;
                }

                if (in_run && this_file_offset == run_file_offset + run_size) {
                    run_size += this_disk_size;
                } else {
                    if (in_run) {
                        flush_run(d, dev_io_base + run_io_start,
                                  run_file_offset, run_size);
                    }
                    run_file_offset = this_file_offset;
                    run_io_start = io_buf_off;
                    run_size = this_disk_size;
                    in_run = true;
                }
                io_buf_off += this_disk_size;
            }
            if (in_run) {
                flush_run(d, dev_io_base + run_io_start,
                          run_file_offset, run_size);
            }
        }

        // ── Scatter: io_buf → data_buf (+ decompress if compressed) ──
        size_t decomp_count = 0;
        size_t dev_off[MAX_GDS_DEVICES] = {};

        for (size_t k = 0; k < batch_count; k++) {
            size_t page_rel = active_pages[batch_start + k];
            uint64_t page_id = field.start_page_id + page_rel;
            size_t d = page_id % num_devices;

            char *io_src = io_base + dev_io_start[d] + dev_off[d];
            char *dst = data_base + (page_rel - output_page_offset) * page_size;

            if (is_compressed) {
                size_t cs = field.compressed_page_sizes[page_rel];
                if (cs < page_size) {
                    nvctx->h_comp_ptrs[decomp_count]    = io_src;
                    nvctx->h_comp_sizes[decomp_count]   = cs;
                    nvctx->h_decomp_ptrs[decomp_count]  = dst;
                    nvctx->h_decomp_sizes[decomp_count] = page_size;
                    decomp_count++;
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                        cudaMemcpyDeviceToDevice, stream));
                }
                dev_off[d] += roundup4096(cs);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                    cudaMemcpyDeviceToDevice, stream));
                dev_off[d] += page_size;
            }
        }

        bool is_last_batch = (batch_start + batch_count >= num_active_pages);
        bool do_sync = !async_final || !is_last_batch;
        if (decomp_count > 0) {
            nvcomp_decompctx_run(field.compression_method, *nvctx,
                                  decomp_count, page_size, stream, do_sync);
            kernel_launches++;
        } else if (do_sync) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        batch_start += batch_count;
    }
}

// ============================================================
// GDS field page loader (sync mode) — loads all pages of a single field
// Follows Q6 gds_worker_sync_thread I/O pattern (RAID0, coalescing)
// Supports both compressed and uncompressed pages.
// ============================================================
static void gds_load_field_sync(
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const std::vector<std::vector<CUfileHandle_t>> &cufile_handles,
    void *io_buf,    // registered GPU buffer (batch_size * page_size)
    void *data_buf,  // destination GPU buffer (npages * page_size)
    NvcompDecompCtx *nvctx,  // nullptr if no compression support needed
    cudaStream_t stream,
    size_t &bytes_read, size_t &ios_completed,
    size_t &kernel_launches,
    size_t range_start, size_t range_npages,  // 0 = all pages
    size_t output_page_offset,  // output offset: dst = data_base + (page_rel - offset) * page_size
    bool async_final,
    size_t batch_pages)
{
    const size_t npages = (range_npages > 0) ? (range_start + range_npages) : field.npages;
    const size_t first_page = range_start;
    const size_t pages_per_batch = batch_pages;
    const size_t dev_region_size = pages_per_batch * page_size / num_devices;
    const bool is_compressed = (field.compression_method != CompressionMethod::NONE);

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    char *io_base = static_cast<char *>(io_buf);
    char *data_base = static_cast<char *>(data_buf);

    size_t next_fh[64] = {};
    size_t pages_done = first_page;

    while (pages_done < npages) {
        size_t batch_pages = std::min(pages_per_batch, npages - pages_done);

        // ── I/O: read by device, coalescing consecutive pages ──
        memset(next_fh, 0, sizeof(next_fh));

        auto flush_run = [&](size_t d, char *dest, size_t file_off, size_t size) {
            size_t fh_idx = next_fh[d] % cufile_handles[d].size();
            next_fh[d]++;
            off_t buf_offset = dest - io_base;
            ssize_t nread = cuFileRead(cufile_handles[d][fh_idx],
                                       io_buf, size, file_off, buf_offset);
            if (nread < 0 || static_cast<size_t>(nread) != size) {
                std::cerr << "cuFileRead failed: device=" << d
                          << " size=" << size << " file_off=" << file_off
                          << " nread=" << nread << std::endl;
            } else {
                bytes_read += nread;
                ios_completed++;
            }
        };

        for (size_t d = 0; d < num_devices; d++) {
            char *dev_io_base = io_base + d * dev_region_size;
            size_t io_buf_off = 0;
            size_t run_file_offset = 0;
            size_t run_io_start = 0;
            size_t run_size = 0;
            bool in_run = false;

            for (size_t k = 0; k < batch_pages; k++) {
                size_t page_rel = pages_done + k;
                uint64_t page_id = field.start_page_id + page_rel;
                if (page_id % num_devices != d) continue;

                size_t this_disk_size;
                size_t this_file_offset;
                if (is_compressed) {
                    this_disk_size = roundup4096(
                        field.compressed_page_sizes[page_rel]);
                    this_file_offset = field.compressed_offsets[page_rel];
                } else {
                    this_disk_size = page_size;
                    uint64_t device_page_id = page_id / num_devices;
                    this_file_offset = device_page_id * page_size;
                }

                if (in_run && this_file_offset == run_file_offset + run_size) {
                    run_size += this_disk_size;
                } else {
                    if (in_run) {
                        flush_run(d, dev_io_base + run_io_start,
                                  run_file_offset, run_size);
                    }
                    run_file_offset = this_file_offset;
                    run_io_start = io_buf_off;
                    run_size = this_disk_size;
                    in_run = true;
                }
                io_buf_off += this_disk_size;
            }
            if (in_run) {
                flush_run(d, dev_io_base + run_io_start,
                          run_file_offset, run_size);
            }
        }

        // ── Scatter: io_buf → data_buf (+ decompress if compressed) ──
        size_t decomp_count = 0;
        size_t dev_off[MAX_GDS_DEVICES] = {};

        for (size_t k = 0; k < batch_pages; k++) {
            size_t page_rel = pages_done + k;
            uint64_t page_id = field.start_page_id + page_rel;
            size_t d = page_id % num_devices;

            char *io_src = io_base + d * dev_region_size + dev_off[d];
            char *dst = data_base + (page_rel - output_page_offset) * page_size;

            if (is_compressed) {
                size_t cs = field.compressed_page_sizes[page_rel];
                if (cs < page_size) {
                    // Queue for nvCOMP batch decompression
                    nvctx->h_comp_ptrs[decomp_count]    = io_src;
                    nvctx->h_comp_sizes[decomp_count]   = cs;
                    nvctx->h_decomp_ptrs[decomp_count]  = dst;
                    nvctx->h_decomp_sizes[decomp_count] = page_size;
                    decomp_count++;
                } else {
                    // Incompressible page: direct copy
                    CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                        cudaMemcpyDeviceToDevice, stream));
                }
                dev_off[d] += roundup4096(cs);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                    cudaMemcpyDeviceToDevice, stream));
                dev_off[d] += page_size;
            }
        }

        bool is_last_batch = (pages_done + batch_pages >= npages);
        bool do_sync = !async_final || !is_last_batch;
        if (decomp_count > 0) {
            nvcomp_decompctx_run(field.compression_method, *nvctx,
                                  decomp_count, page_size, stream, do_sync);
            kernel_launches++;
        } else if (do_sync) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        pages_done += batch_pages;
    }
}


// ============================================================
// GDS field page loader — double-buffered I/O-decomp pipeline.
//
// 2 set_bufs (double-buffered).  Each set_buf holds
// nvcomp_batch_pages of data, filled by multiple io_sub_batch
// cuFileRead calls.  One set fills via cuFileRead while the
// other set's nvCOMP decomp runs on nvcomp_stream.
// nvCOMP reads directly from set_bufs (no staging copy).
//
// For uncompressed fields, nvcomp_stream/nvctx may be nullptr.
// Caller must sync io_stream and nvcomp_stream after return.
// ============================================================
static void gds_load_field_pipelined(
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const std::vector<std::vector<CUfileHandle_t>> &cufile_handles,
    void *set_bufs[2],           // 2 registered GPU buffers (one per double-buffer set)
    void *data_buf,              // destination GPU buffer
    NvcompDecompCtx *nvctx,      // nvCOMP context (sized for nvcomp_batch_pages; nullptr if uncompressed)
    cudaStream_t io_stream,      // for D2D scatter (uncompressed/incompressible pages)
    cudaStream_t nvcomp_stream,  // for nvCOMP decomp (nullptr if uncompressed)
    cudaEvent_t io_done_event,   // for io_stream → nvcomp_stream sync
    cudaEvent_t set_done_events[2], // signaled when nvCOMP finishes reading a set
    size_t &bytes_read, size_t &ios_completed,
    size_t &kernel_launches,
    size_t range_start, size_t range_npages,
    size_t output_page_offset,
    size_t io_sub_batch_pages,
    size_t nvcomp_batch_pages,
    size_t device_start)
{
    const size_t npages = (range_npages > 0) ? (range_start + range_npages) : field.npages;
    const size_t first_page = range_start;
    const size_t io_dev_region_size = io_sub_batch_pages * page_size / num_devices;
    const size_t io_sub_size = io_sub_batch_pages * page_size;
    const bool is_compressed = (field.compression_method != CompressionMethod::NONE);

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    char *data_base = static_cast<char *>(data_buf);
    size_t pages_done = first_page;
    int set_cur = 0;
    bool set_submitted[2] = {false, false};

    while (pages_done < npages) {
        // Wait for previous nvCOMP on this set to finish (bufs safe to reuse)
        if (set_submitted[set_cur]) {
            CUDA_CHECK(cudaEventSynchronize(set_done_events[set_cur]));
        }

        size_t nv_batch = std::min(nvcomp_batch_pages, npages - pages_done);
        size_t decomp_count = 0;

        // ── Fill this set's io_bufs with I/O sub-batches ──
        size_t batch_pages_loaded = 0;
        size_t sub_idx = 0;
        while (batch_pages_loaded < nv_batch) {
            size_t io_batch = std::min(io_sub_batch_pages, nv_batch - batch_pages_loaded);
            char *set_base = static_cast<char *>(set_bufs[set_cur]);
            char *io_base = set_base + sub_idx * io_sub_size;

            // ── cuFileRead: coalesce consecutive pages ──
            size_t next_fh[MAX_GDS_DEVICES] = {};
            auto flush_run = [&](size_t d, char *dest, size_t file_off, size_t size) {
                size_t fh_idx = next_fh[d] % cufile_handles[d].size();
                next_fh[d]++;
                off_t buf_offset = dest - set_base;
                ssize_t nread = cuFileRead(cufile_handles[d][fh_idx],
                                           set_bufs[set_cur], size, file_off, buf_offset);
                if (nread < 0 || static_cast<size_t>(nread) != size) {
                    std::cerr << "cuFileRead failed: device=" << d
                              << " size=" << size << " file_off=" << file_off
                              << " nread=" << nread << std::endl;
                } else {
                    bytes_read += nread;
                    ios_completed++;
                }
            };

            // Stagger device iteration order (device_start) so that
            // concurrent loader threads don't all read from the same
            // device simultaneously (avoids NVMe convoy effect).
            for (size_t di = 0; di < num_devices; di++) {
                size_t d = (device_start + di) % num_devices;
                char *dev_io_base = io_base + d * io_dev_region_size;
                size_t io_buf_off = 0;
                size_t run_file_offset = 0;
                size_t run_io_start = 0;
                size_t run_size = 0;
                bool in_run = false;

                for (size_t k = 0; k < io_batch; k++) {
                    size_t page_rel = pages_done + batch_pages_loaded + k;
                    uint64_t page_id = field.start_page_id + page_rel;
                    if (page_id % num_devices != d) continue;

                    size_t this_disk_size;
                    size_t this_file_offset;
                    if (is_compressed) {
                        this_disk_size = roundup4096(
                            field.compressed_page_sizes[page_rel]);
                        this_file_offset = field.compressed_offsets[page_rel];
                    } else {
                        this_disk_size = page_size;
                        uint64_t device_page_id = page_id / num_devices;
                        this_file_offset = device_page_id * page_size;
                    }

                    if (in_run && this_file_offset == run_file_offset + run_size) {
                        run_size += this_disk_size;
                    } else {
                        if (in_run) {
                            flush_run(d, dev_io_base + run_io_start,
                                      run_file_offset, run_size);
                        }
                        run_file_offset = this_file_offset;
                        run_io_start = io_buf_off;
                        run_size = this_disk_size;
                        in_run = true;
                    }
                    io_buf_off += this_disk_size;
                }
                if (in_run) {
                    flush_run(d, dev_io_base + run_io_start,
                              run_file_offset, run_size);
                }
            }

            // ── Scatter: set nvCOMP ptrs (compressed) or D2D copy (uncompressed) ──
            size_t dev_off[MAX_GDS_DEVICES] = {};
            for (size_t k = 0; k < io_batch; k++) {
                size_t page_rel = pages_done + batch_pages_loaded + k;
                uint64_t page_id = field.start_page_id + page_rel;
                size_t d = page_id % num_devices;

                char *io_src = io_base + d * io_dev_region_size + dev_off[d];
                char *dst = data_base + (page_rel - output_page_offset) * page_size;

                if (is_compressed) {
                    size_t cs = field.compressed_page_sizes[page_rel];
                    if (cs < page_size) {
                        // nvCOMP reads directly from io_buf (no staging copy)
                        nvctx->h_comp_ptrs[decomp_count]    = io_src;
                        nvctx->h_comp_sizes[decomp_count]   = cs;
                        nvctx->h_decomp_ptrs[decomp_count]  = dst;
                        nvctx->h_decomp_sizes[decomp_count] = page_size;
                        decomp_count++;
                    } else {
                        CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                            cudaMemcpyDeviceToDevice, io_stream));
                    }
                    dev_off[d] += roundup4096(cs);
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                        cudaMemcpyDeviceToDevice, io_stream));
                    dev_off[d] += page_size;
                }
            }

            batch_pages_loaded += io_batch;
            sub_idx++;
        }

        // ── Launch nvCOMP batch on nvcomp_stream ──
        if (decomp_count > 0 && nvctx && nvcomp_stream) {
            // Ensure any D2D copies (incompressible pages) are done
            CUDA_CHECK(cudaEventRecord(io_done_event, io_stream));
            CUDA_CHECK(cudaStreamWaitEvent(nvcomp_stream, io_done_event));

            nvcomp_decompctx_run(field.compression_method, *nvctx,
                                  decomp_count, page_size, nvcomp_stream,
                                  /*do_sync=*/false);
            kernel_launches++;
        }

        // Record event so we know when io_bufs in this set are safe to reuse.
        // When decomp_count > 0, nvcomp_stream already depends on io_stream
        // (via io_done_event), so recording on nvcomp_stream covers both.
        // When decomp_count == 0 (uncompressed field or all-incompressible batch),
        // all work is on io_stream, so we must record on io_stream.
        if (decomp_count > 0 && nvcomp_stream) {
            CUDA_CHECK(cudaEventRecord(set_done_events[set_cur], nvcomp_stream));
        } else {
            CUDA_CHECK(cudaEventRecord(set_done_events[set_cur], io_stream));
        }
        set_submitted[set_cur] = true;

        pages_done += nv_batch;
        set_cur ^= 1;
    }
    // Caller must sync io_stream and nvcomp_stream.
}

// ────────────────────────────────────────────────────────────
// gds_load_field_pipelined_selective
// Double-buffered IO-decomp pipeline for sparse (zone map) page indices.
// Same pipeline structure as gds_load_field_pipelined, but iterates
// active_pages[] instead of a contiguous page range.
// ────────────────────────────────────────────────────────────
static void gds_load_field_pipelined_selective(
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const std::vector<std::vector<CUfileHandle_t>> &cufile_handles,
    void *set_bufs[2],           // 2 registered GPU buffers (one per double-buffer set)
    void *data_buf,              // destination GPU buffer
    NvcompDecompCtx *nvctx,      // nvCOMP context (nullptr if uncompressed)
    cudaStream_t io_stream,      // for D2D scatter (uncompressed/incompressible pages)
    cudaStream_t nvcomp_stream,  // for nvCOMP decomp (nullptr if uncompressed)
    cudaEvent_t io_done_event,   // for io_stream → nvcomp_stream sync
    cudaEvent_t set_done_events[2], // signaled when nvCOMP finishes reading a set
    size_t &bytes_read, size_t &ios_completed,
    size_t &kernel_launches,
    const uint32_t *active_pages, size_t num_active_pages,
    size_t output_page_offset,
    size_t io_sub_batch_pages,
    size_t nvcomp_batch_pages,
    size_t device_start)
{
    if (num_active_pages == 0) return;

    const size_t io_sub_size = io_sub_batch_pages * page_size;
    const bool is_compressed = (field.compression_method != CompressionMethod::NONE);

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    char *data_base = static_cast<char *>(data_buf);
    size_t ap_done = 0;   // index into active_pages[]
    int set_cur = 0;
    bool set_submitted[2] = {false, false};

    while (ap_done < num_active_pages) {
        // Wait for previous batch on this set to finish (bufs safe to reuse)
        if (set_submitted[set_cur]) {
            CUDA_CHECK(cudaEventSynchronize(set_done_events[set_cur]));
        }

        size_t nv_batch = std::min(nvcomp_batch_pages, num_active_pages - ap_done);
        size_t decomp_count = 0;

        // ── Fill this set's io_bufs with I/O sub-batches ──
        size_t batch_pages_loaded = 0;
        size_t sub_idx = 0;
        while (batch_pages_loaded < nv_batch) {
            size_t io_batch = std::min(io_sub_batch_pages, nv_batch - batch_pages_loaded);
            char *set_base = static_cast<char *>(set_bufs[set_cur]);
            char *io_base = set_base + sub_idx * io_sub_size;

            // ── Compute per-device data sizes for dynamic layout ──
            // Sparse pages may not distribute evenly across devices,
            // so we compute actual per-device sizes instead of assuming
            // io_sub_batch_pages / num_devices pages per device.
            size_t dev_data_size[MAX_GDS_DEVICES] = {};
            for (size_t k = 0; k < io_batch; k++) {
                size_t page_rel = active_pages[ap_done + batch_pages_loaded + k];
                uint64_t page_id = field.start_page_id + page_rel;
                size_t d = page_id % num_devices;
                if (is_compressed)
                    dev_data_size[d] += roundup4096(field.compressed_page_sizes[page_rel]);
                else
                    dev_data_size[d] += page_size;
            }
            size_t dev_io_start[MAX_GDS_DEVICES + 1] = {};
            for (size_t d = 0; d < num_devices; d++)
                dev_io_start[d + 1] = dev_io_start[d] + dev_data_size[d];

            // ── cuFileRead: coalesce consecutive pages ──
            // Stagger device iteration order (device_start) so that
            // concurrent loader threads don't all read from the same
            // device simultaneously (avoids NVMe convoy effect).
            size_t next_fh[MAX_GDS_DEVICES] = {};

            auto flush_run = [&](size_t d, char *dest, size_t file_off, size_t size) {
                size_t fh_idx = next_fh[d] % cufile_handles[d].size();
                next_fh[d]++;
                off_t buf_offset = dest - set_base;
                ssize_t nread = cuFileRead(cufile_handles[d][fh_idx],
                                           set_bufs[set_cur], size, file_off, buf_offset);
                if (nread < 0 || static_cast<size_t>(nread) != size) {
                    std::cerr << "cuFileRead failed: device=" << d
                              << " size=" << size << " file_off=" << file_off
                              << " nread=" << nread << std::endl;
                } else {
                    bytes_read += nread;
                    ios_completed++;
                }
            };

            for (size_t di = 0; di < num_devices; di++) {
                size_t d = (device_start + di) % num_devices;
                char *dev_io_base = io_base + dev_io_start[d];
                size_t io_buf_off = 0;
                size_t run_file_offset = 0;
                size_t run_io_start = 0;
                size_t run_size = 0;
                bool in_run = false;

                for (size_t k = 0; k < io_batch; k++) {
                    size_t page_rel = active_pages[ap_done + batch_pages_loaded + k];
                    uint64_t page_id = field.start_page_id + page_rel;
                    if (page_id % num_devices != d) continue;

                    size_t this_disk_size;
                    size_t this_file_offset;
                    if (is_compressed) {
                        this_disk_size = roundup4096(
                            field.compressed_page_sizes[page_rel]);
                        this_file_offset = field.compressed_offsets[page_rel];
                    } else {
                        this_disk_size = page_size;
                        uint64_t device_page_id = page_id / num_devices;
                        this_file_offset = device_page_id * page_size;
                    }

                    if (in_run && this_file_offset == run_file_offset + run_size) {
                        run_size += this_disk_size;
                    } else {
                        if (in_run) {
                            flush_run(d, dev_io_base + run_io_start,
                                      run_file_offset, run_size);
                        }
                        run_file_offset = this_file_offset;
                        run_io_start = io_buf_off;
                        run_size = this_disk_size;
                        in_run = true;
                    }
                    io_buf_off += this_disk_size;
                }
                if (in_run) {
                    flush_run(d, dev_io_base + run_io_start,
                              run_file_offset, run_size);
                }
            }

            // ── Scatter: set nvCOMP ptrs (compressed) or D2D copy (uncompressed) ──
            // Use set_cur-indexed host arrays so that batch N (set 0) and
            // batch N+1 (set 1) never share host staging memory.  This
            // eliminates the h_comp_ptrs race condition without any sync.
            void  **h_cp  = nullptr;
            void  **h_dp  = nullptr;
            size_t *h_cs  = nullptr;
            size_t *h_ds  = nullptr;
            if (nvctx) {
                h_cp  = (set_cur == 0) ? nvctx->h_comp_ptrs    : nvctx->h_comp_ptrs_1;
                h_dp  = (set_cur == 0) ? nvctx->h_decomp_ptrs  : nvctx->h_decomp_ptrs_1;
                h_cs  = (set_cur == 0) ? nvctx->h_comp_sizes   : nvctx->h_comp_sizes_1;
                h_ds  = (set_cur == 0) ? nvctx->h_decomp_sizes : nvctx->h_decomp_sizes_1;
            }
            size_t dev_off[MAX_GDS_DEVICES] = {};
            for (size_t k = 0; k < io_batch; k++) {
                size_t page_rel = active_pages[ap_done + batch_pages_loaded + k];
                uint64_t page_id = field.start_page_id + page_rel;
                size_t d = page_id % num_devices;

                char *io_src = io_base + dev_io_start[d] + dev_off[d];
                char *dst = data_base + (page_rel - output_page_offset) * page_size;

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
                            cudaMemcpyDeviceToDevice, io_stream));
                    }
                    dev_off[d] += roundup4096(cs);
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                        cudaMemcpyDeviceToDevice, io_stream));
                    dev_off[d] += page_size;
                }
            }

            batch_pages_loaded += io_batch;
            sub_idx++;
        }

        // ── Launch nvCOMP batch on io_stream ──
        // Double-buffered host staging arrays (indexed by set_cur) eliminate
        // the h_comp_ptrs race: batch N (set 0) and batch N+1 (set 1) write
        // to separate pinned host arrays.  The cudaEventSynchronize on
        // set_done_events at the top of the loop ensures the SAME set's
        // previous batch has completed (including H2D) before we reuse it.
        // This allows fully async decomp (do_sync=false) like the non-ZM path.
        if (decomp_count > 0 && nvctx) {
            nvcomp_decompctx_run(field.compression_method, *nvctx,
                                  decomp_count, page_size, io_stream,
                                  /*do_sync=*/true, /*skip_h2d=*/false,
                                  /*h_set=*/set_cur);
            kernel_launches++;
        }

        // Record event for buffer reuse timing (always on io_stream)
        CUDA_CHECK(cudaEventRecord(set_done_events[set_cur], io_stream));
        set_submitted[set_cur] = true;

        ap_done += nv_batch;
        set_cur ^= 1;
    }
    // Caller must sync io_stream and nvcomp_stream.
}


// ============================================================
// TPC-H Q13 — GOLAP sync mode
// ============================================================
BenchmarkResult tpch_q13(BenchmarkOptions &options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    TPCHTableMetadata *metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    {
        const size_t page_size = metadatap->page_size;
        std::cout << "=== TPCH Q13 ===" << std::endl;
        std::cout << "Page Size: " << page_size << std::endl;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, page_size) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, page_size);
    }

    metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    TPCHTableMetadata &metadata = *metadatap;
    superpage_set_constants(metadata.page_size);
    const size_t page_size = metadata.page_size;
    const size_t num_devices = fds.size();

    std::cout << "nrecs_orders: " << metadata.table_orders_nrows
              << ", nrecs_customer: " << metadata.table_customer_nrows
              << ", num_devices: " << num_devices << std::endl;

    // ── Phase 0: Load metadata for ORDERS and CUSTOMER ──
    // ORDERS fields: O_CUSTKEY (idx 0), O_COMMENT (idx 1)
    constexpr size_t num_orders_cols = TPCH::Query::Q13::NUM_ORDERS_SCAN_COLS;
    auto orders_cols_idx = TPCH::Query::Q13::ORDERS_SCAN_COLS;
    std::vector<FieldPageInfo> orders_fields(num_orders_cols);

    uint32_t saved_column = metadata.column;
    metadata.column = TPCH::common::Table::ORDERS;
    prepare_fields_metadata(fds, metadata, page_size, orders_cols_idx, orders_fields);

    // CUSTOMER fields: C_CUSTKEY (idx 0)
    constexpr size_t num_customer_cols = TPCH::Query::Q13::NUM_CUSTOMER_SCAN_COLS;
    auto customer_cols_idx = TPCH::Query::Q13::CUSTOMER_SCAN_COLS;
    std::vector<FieldPageInfo> customer_fields(num_customer_cols);

    metadata.column = TPCH::common::Table::CUSTOMER;
    prepare_fields_metadata(fds, metadata, page_size, customer_cols_idx, customer_fields);

    metadata.column = saved_column;

    const FieldPageInfo &fi_o_custkey = orders_fields[0];   // O_CUSTKEY
    const FieldPageInfo &fi_o_comment = orders_fields[1];   // O_COMMENT
    const FieldPageInfo &fi_c_custkey = customer_fields[0];  // C_CUSTKEY

    std::cout << "  O_CUSTKEY: start_page=" << fi_o_custkey.start_page_id
              << " npages=" << fi_o_custkey.npages
              << " compression=" << compression_method_name(fi_o_custkey.compression_method)
              << std::endl;
    std::cout << "  O_COMMENT: start_page=" << fi_o_comment.start_page_id
              << " npages=" << fi_o_comment.npages
              << " compression=" << compression_method_name(fi_o_comment.compression_method)
              << std::endl;
    std::cout << "  C_CUSTKEY: start_page=" << fi_c_custkey.start_page_id
              << " npages=" << fi_c_custkey.npages
              << " compression=" << compression_method_name(fi_c_custkey.compression_method)
              << std::endl;

    const uint64_t nrecs_orders = metadata.table_orders_nrows;
    const uint64_t nrecs_customer = metadata.table_customer_nrows;

    // ── Check for compressed fields ──
    bool any_compressed = false;
    std::vector<FieldPageInfo> all_fields;
    for (auto &fi : orders_fields) {
        all_fields.push_back(fi);
        if (fi.compression_method != CompressionMethod::NONE) any_compressed = true;
    }
    for (auto &fi : customer_fields) {
        all_fields.push_back(fi);
        if (fi.compression_method != CompressionMethod::NONE) any_compressed = true;
    }

    if (any_compressed) {
        std::cout << "[Q13] Compressed fields detected." << std::endl;
    }

    // ── GDS setup (per-thread resources) ──
    mb_cufile_driver_open();

    const size_t nthreads = options.nthreads;
    const size_t handles_per_thread = options.gds_num_handlers_per_thread;

    struct Q13LoaderCtx {
        std::vector<std::vector<CUfileHandle_t>> cufile_handles;
        std::vector<std::vector<int>> dup_fds;
        void *io_buf = nullptr;
        size_t io_batch_pages = 0;
        cudaStream_t stream = nullptr;
        NvcompDecompCtx nvctx;
        size_t bytes_read = 0;
        size_t ios_completed = 0;
        size_t kernel_launches = 0;
    };

    std::vector<Q13LoaderCtx> loaders(nthreads);
    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    // Phase 1: create CUfileHandles and streams (io_buf/nvcomp deferred for budget control)
    for (size_t t = 0; t < nthreads; t++) {
        auto &L = loaders[t];
        L.cufile_handles.resize(num_devices);
        L.dup_fds.resize(num_devices);
        for (size_t d = 0; d < num_devices; d++) {
            for (size_t h = 0; h < handles_per_thread; h++) {
                int duped = dup(fds[d]);
                if (duped < 0) {
                    std::cerr << "dup failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                L.dup_fds[d].push_back(duped);
                L.cufile_handles[d].push_back(mb_cufile_handle_register(duped));
            }
        }
        CUDA_CHECK(cudaStreamCreate(&L.stream));
    }

    std::cout << "[Q13] nthreads=" << nthreads << std::endl;

    // Main stream for compute phases (flatten, Q13 pipeline)
    cudaStream_t stream = loaders[0].stream;

    size_t total_bytes_read = 0, total_ios = 0;

    // Pre-allocated thread vector shared across loader lambdas (R4: avoid heap alloc in timed section)
    std::vector<std::thread> _mt_threads;
    _mt_threads.reserve(nthreads);

    // Multi-threaded field loader: splits pages across nthreads I/O threads
    auto load_field_mt = [&](const FieldPageInfo &field, void *data_buf) {
        if (nthreads <= 1) {
            auto &L = loaders[0];
            NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
            gds_load_field_sync(field, page_size, num_devices,
                                L.cufile_handles, L.io_buf, data_buf,
                                nv, L.stream, L.bytes_read, L.ios_completed,
                                L.kernel_launches,
                                0, 0, 0, false, L.io_batch_pages);
            return;
        }
        size_t npages = field.npages;
        size_t pages_per_thr = (npages + nthreads - 1) / nthreads;
        auto &threads = _mt_threads; threads.clear();
        for (size_t t = 0; t < nthreads; t++) {
            size_t start = t * pages_per_thr;
            if (start >= npages) break;
            size_t count = std::min(pages_per_thr, npages - start);
            threads.emplace_back([&, t, start, count]() {
                auto &L = loaders[t];
                mb_cuda_set_context(cuda_ctx_handle);
                NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                gds_load_field_sync(field, page_size, num_devices,
                                    L.cufile_handles, L.io_buf, data_buf,
                                    nv, L.stream,
                                    L.bytes_read, L.ios_completed,
                                    L.kernel_launches,
                                    start, count, 0, false, L.io_batch_pages);
            });
        }
        for (auto &th : threads) th.join();
    };

    // Tile-based field loader: loads [tile_start, tile_start+tile_npages) into data_buf[0..]
    auto load_field_tile_mt = [&](const FieldPageInfo &field, void *data_buf,
                                   size_t tile_start, size_t tile_npages) {
        if (nthreads <= 1) {
            auto &L = loaders[0];
            NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
            gds_load_field_sync(field, page_size, num_devices,
                                L.cufile_handles, L.io_buf, data_buf,
                                nv, L.stream, L.bytes_read, L.ios_completed,
                                L.kernel_launches,
                                tile_start, tile_npages, tile_start, false, L.io_batch_pages);
            return;
        }
        size_t pages_per_thr = (tile_npages + nthreads - 1) / nthreads;
        auto &threads = _mt_threads; threads.clear();
        for (size_t t = 0; t < nthreads; t++) {
            size_t start = tile_start + t * pages_per_thr;
            if (start >= tile_start + tile_npages) break;
            size_t count = std::min(pages_per_thr, tile_start + tile_npages - start);
            threads.emplace_back([&, t, start, count, tile_start]() {
                auto &L = loaders[t];
                mb_cuda_set_context(cuda_ctx_handle);
                NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                gds_load_field_sync(field, page_size, num_devices,
                                    L.cufile_handles, L.io_buf, data_buf,
                                    nv, L.stream,
                                    L.bytes_read, L.ios_completed,
                                    L.kernel_launches,
                                    start, count, tile_start, false, L.io_batch_pages);
            });
        }
        for (auto &th : threads) th.join();
    };

    // ── Pre-allocate all GPU buffers before timer ──
    constexpr size_t Q13_BATCH_PAGES = 512;
    void *staging_buf = mb_cuda_alloc(Q13_BATCH_PAGES * page_size);

    uint64_t *d_o_custkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_custkey_flat, nrecs_orders * sizeof(uint64_t)));

    uint64_t *d_c_custkey = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c_custkey, nrecs_customer * sizeof(uint64_t)));

    // Aggregation array for batch scan (UINT64_MAX=LIKE match, real custkey=non-match)
    uint64_t *d_o_aggr_custkey = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_aggr_custkey, nrecs_orders * sizeof(uint64_t)));
    // Atomic counter for qualifying orders (shared across batches)
    uint64_t *d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint64_t)));

    // Prefix-sum pool (reused across batch flatten operations)
    size_t ps_pool_npages = std::max({fi_o_custkey.npages, fi_c_custkey.npages, Q13_BATCH_PAGES});
    uint64_t *d_ps_pool = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_pool, ps_pool_npages * sizeof(uint64_t)));

    // Pre-upload full prefix_sum to GPU and pre-compute batch metadata (Rule 3)
    struct Q13BatchMeta { uint64_t row_start, batch_nrecs; };
    uint64_t *d_full_ps_q13[3] = {};
    std::vector<Q13BatchMeta> q13_batch_metas[3];
    {
        const FieldPageInfo *q13_ps_fields[3] = {&fi_o_custkey, &fi_c_custkey, &fi_o_comment};
        for (int f = 0; f < 3; f++) {
            auto &fi = *q13_ps_fields[f];
            CUDA_CHECK(cudaMalloc(&d_full_ps_q13[f], (fi.npages + 1) * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_full_ps_q13[f], fi.prefix_sum_nrecs,
                                  (fi.npages + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            for (size_t pg = 0; pg < fi.npages; pg += Q13_BATCH_PAGES) {
                size_t bnp = std::min(Q13_BATCH_PAGES, fi.npages - pg);
                q13_batch_metas[f].push_back({fi.prefix_sum_nrecs[pg],
                    fi.prefix_sum_nrecs[pg + bnp] - fi.prefix_sum_nrecs[pg]});
            }
        }
    }

    // ── KMP setup (constants + CPU computation + GPU upload) ──
    const char *patterns_str = "specialrequests";
    const int pattern_offsets_h[] = {0, 7};
    const int pattern_lengths_h[] = {7, 8};
    const int total_pattern_chars = 15;
    const int num_patterns = 2;

    std::vector<int> next_h(total_pattern_chars, 0);
    for (int p = 0; p < num_patterns; p++) {
        int off = pattern_offsets_h[p];
        int len = pattern_lengths_h[p];
        for (int i = 1; i < len; i++) {
            int j = next_h[off + i - 1];
            while (j > 0 && patterns_str[off + i] != patterns_str[off + j])
                j = next_h[off + j - 1];
            if (patterns_str[off + i] == patterns_str[off + j])
                j++;
            next_h[off + i] = j;
        }
    }

    char *d_patterns = nullptr;
    int *d_next = nullptr, *d_pattern_offsets = nullptr, *d_pattern_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_patterns, total_pattern_chars));
    CUDA_CHECK(cudaMemcpy(d_patterns, patterns_str, total_pattern_chars,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_next, total_pattern_chars * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_next, next_h.data(), total_pattern_chars * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_pattern_offsets, num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pattern_offsets, pattern_offsets_h,
                          num_patterns * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_pattern_lengths, num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pattern_lengths, pattern_lengths_h,
                          num_patterns * sizeof(int), cudaMemcpyHostToDevice));

    // Pre-allocate Q13 pipeline buffers (sort+RLE, no malloc inside timed section)
    Q13PipelineBuffers q13_bufs{};
    {
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_sort_alt, nrecs_orders * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_rle_keys, nrecs_orders * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_rle_counts, nrecs_orders * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_num_rle, sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_c_count, nrecs_customer * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_c_count_alt, nrecs_customer * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_aggr2_keys, nrecs_customer * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_aggr2_counts, nrecs_customer * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_composite_keys, nrecs_customer * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_composite_keys_alt, nrecs_customer * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_composite_vals, nrecs_customer * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_composite_vals_alt, nrecs_customer * sizeof(uint64_t)));
        q13_bufs.cub_temp_bytes = q13_pipeline_cub_temp_size(nrecs_orders, nrecs_customer);
        CUDA_CHECK(cudaMalloc(&q13_bufs.d_cub_temp, q13_bufs.cub_temp_bytes));
        q13_bufs.h_composite_capacity = nrecs_customer;
        q13_bufs.h_composite = (uint64_t *)malloc(nrecs_customer * sizeof(uint64_t));
    }

    // Phase 2: allocate per-loader io_buf with dynamic sizing (gpu_mem <= 40 GiB)
    {
        static constexpr size_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_before_loaders = 0;
        cudaMemGetInfo(&gpu_free_before_loaders, &gpu_total_dummy);
        size_t non_loader_bytes = gpu_free_start - gpu_free_before_loaders;
        size_t q13_io_pages = 1;
        if (GPU_MEM_BUDGET > non_loader_bytes) {
            size_t nvcomp_per_loader = 2 * 1024 * 1024;  // ~2 MiB nvcomp overhead
            size_t budget_for_loaders = GPU_MEM_BUDGET - non_loader_bytes;
            size_t per_loader_budget = budget_for_loaders / nthreads;
            if (per_loader_budget > nvcomp_per_loader) {
                q13_io_pages = (per_loader_budget - nvcomp_per_loader) / page_size;
            }
            q13_io_pages = std::max(q13_io_pages, (size_t)1);
            q13_io_pages = std::min(q13_io_pages, (size_t)GDS_SYNC_BATCH_PAGES);
        }
        for (size_t t = 0; t < nthreads; t++) {
            auto &L = loaders[t];
            L.io_batch_pages = q13_io_pages;
            L.io_buf = mb_cuda_alloc(q13_io_pages * page_size);
            GDS_CHECK(cuFileBufRegister(L.io_buf, q13_io_pages * page_size, 0));
            if (any_compressed) {
                nvcomp_decompctx_alloc(L.nvctx, q13_io_pages, page_size, all_fields);
            }
        }
        std::cout << "[Q13] Per-loader IO: " << q13_io_pages << " pages ("
                  << (q13_io_pages * page_size / (1024*1024)) << " MiB), "
                  << nthreads << " loaders" << std::endl;
    }

    // Pre-allocate result vector (avoid heap alloc in timed section)
    std::vector<std::pair<uint32_t, uint32_t>> q13_result;
    q13_result.reserve(1024);

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t golap_gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // ════════════════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    // ════════════════════════════════════════════════════════

    // ── Phase 1: Batch flatten O_CUSTKEY ──
    std::cout << "[Q13] Loading O_CUSTKEY (" << fi_o_custkey.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_o_custkey.npages; pg += Q13_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q13_BATCH_PAGES, fi_o_custkey.npages - pg);
          load_field_tile_mt(fi_o_custkey, staging_buf, pg, bnp);

          auto &bm = q13_batch_metas[0][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q13[0], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;

          q13_flatten_int64_pages_ps(
              static_cast<const char *>(staging_buf),
              page_size, d_ps_pool, bnp,
              bm.batch_nrecs, d_o_custkey_flat + bm.row_start, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    // ── Phase 2: Batch flatten C_CUSTKEY ──
    std::cout << "[Q13] Loading C_CUSTKEY (" << fi_c_custkey.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_c_custkey.npages; pg += Q13_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q13_BATCH_PAGES, fi_c_custkey.npages - pg);
          load_field_tile_mt(fi_c_custkey, staging_buf, pg, bnp);

          auto &bm = q13_batch_metas[1][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q13[1], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;

          q13_flatten_int64_pages_ps(
              static_cast<const char *>(staging_buf),
              page_size, d_ps_pool, bnp,
              bm.batch_nrecs, d_c_custkey + bm.row_start, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    // ── Phase 3: Batch O_COMMENT scan + aggregation ──
    std::cout << "[Q13] Running Q13 pipeline (batch scan)..." << std::endl;
    CUDA_CHECK(cudaMemset(d_o_aggr_custkey, 0xFF, nrecs_orders * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint64_t)));

    std::cout << "[Q13] Loading O_COMMENT (" << fi_o_comment.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_o_comment.npages; pg += Q13_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q13_BATCH_PAGES, fi_o_comment.npages - pg);
          load_field_tile_mt(fi_o_comment, staging_buf, pg, bnp);

          auto &bm = q13_batch_metas[2][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q13[2], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;

          q13_scan_batch(
              static_cast<const char *>(staging_buf),
              d_ps_pool, (uint32_t)bnp, (uint32_t)page_size, bm.batch_nrecs,
              d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
              num_patterns, total_pattern_chars,
              d_o_custkey_flat + bm.row_start,
              d_o_aggr_custkey + bm.row_start,
              d_count, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    uint64_t h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    std::cout << "[Q13] Qualifying orders (NOT LIKE): " << h_count
              << " / " << nrecs_orders << std::endl;

    // ── Phase 4: Aggregation pipeline ──
    q13_pig_aggregate(q13_bufs, d_o_aggr_custkey, nrecs_orders, d_c_custkey, nrecs_customer,
                      q13_result, stream);
    s_kernel_launches++;

    auto total_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(total_end - total_start).count();

    // Aggregate I/O stats from all loader threads
    for (size_t t = 0; t < nthreads; t++) {
        total_bytes_read += loaders[t].bytes_read;
        total_ios += loaders[t].ios_completed;
    }

    for (size_t t = 0; t < nthreads; t++)
        s_kernel_launches += loaders[t].kernel_launches;

    // ── Print results ──
    std::cout << "\n=== TPC-H Q13 Result ===" << std::endl;
    std::cout << "c_count | custdist" << std::endl;
    std::cout << "--------+---------" << std::endl;
    for (auto &[c_count, custdist] : q13_result) {
        printf("%7u | %8u\n", c_count, custdist);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_ios << std::endl;
    std::cout << "Total bytes read: " << total_bytes_read << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    cudaFree(d_patterns);
    cudaFree(d_next);
    cudaFree(d_pattern_offsets);
    cudaFree(d_pattern_lengths);
    if (d_o_custkey_flat) cudaFree(d_o_custkey_flat);
    cudaFree(d_ps_pool);
    for (int f = 0; f < 3; f++) cudaFree(d_full_ps_q13[f]);
    cudaFree(d_c_custkey);
    cudaFree(d_o_aggr_custkey);
    cudaFree(d_count);
    cudaFree(q13_bufs.d_sort_alt);
    cudaFree(q13_bufs.d_rle_keys);
    cudaFree(q13_bufs.d_rle_counts);
    cudaFree(q13_bufs.d_num_rle);
    cudaFree(q13_bufs.d_c_count);
    cudaFree(q13_bufs.d_c_count_alt);
    cudaFree(q13_bufs.d_aggr2_keys);
    cudaFree(q13_bufs.d_aggr2_counts);
    cudaFree(q13_bufs.d_composite_keys);
    cudaFree(q13_bufs.d_composite_keys_alt);
    cudaFree(q13_bufs.d_composite_vals);
    cudaFree(q13_bufs.d_composite_vals_alt);
    cudaFree(q13_bufs.d_cub_temp);
    free(q13_bufs.h_composite);

    mb_cuda_free(staging_buf);

    // Free per-thread loader resources
    for (size_t t = 0; t < nthreads; t++) {
        auto &L = loaders[t];
        if (any_compressed) nvcomp_decompctx_free(L.nvctx);
        cuFileBufDeregister(L.io_buf);
        mb_cuda_free(L.io_buf);
        CUDA_CHECK(cudaStreamDestroy(L.stream));
        for (size_t d = 0; d < num_devices; d++) {
            for (size_t h = 0; h < L.cufile_handles[d].size(); h++) {
                cuFileHandleDeregister(L.cufile_handles[d][h]);
                close(L.dup_fds[d][h]);
            }
        }
    }

    size_t total_pages = 0;
    for (const auto &fi : orders_fields) total_pages += fi.npages;
    for (const auto &fi : customer_fields) total_pages += fi.npages;

    std::string comp_str = collect_compression_methods({orders_fields, customer_fields});
    free_fields_metadata(orders_fields);
    free_fields_metadata(customer_fields);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = total_ios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = comp_str,
        .gpu_mem_bytes = golap_gpu_mem_bytes,
        .gpu_app_bytes = golap_gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// TPC-H Q16 — GOLAP sync mode
// ============================================================

// Host-side VCHAR page access (column-store layout)
static uint32_t host_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}
static uint32_t host_pag_get_oslt(const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}
static uint16_t host_pagcol_vchar_len(const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = host_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}
static const char *host_pagcol_vchar_data(const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = host_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);  // skip [len u16 + pad u16]
}

// Host-side CHAR page access
static const char *host_pagcol_char_data(const char *page, uint32_t slotid, uint32_t padded_len) {
    return page + 12 /* sizeof(pag_head) */ + padded_len * slotid;
}

// Host-side KMP multi-pattern match (for '%Customer%Complaints%')
static bool host_kmp_multi_match(const char *text, uint16_t text_len,
    const char *patterns, const int *next_table,
    const int *pattern_offsets, const int *pattern_lengths, int num_patterns)
{
    int text_pos = 0;
    for (int p = 0; p < num_patterns; p++) {
        int poff = pattern_offsets[p];
        int plen = pattern_lengths[p];
        int j = 0;
        while (text_pos < text_len && j < plen) {
            if (text[text_pos] == patterns[poff + j]) {
                text_pos++;
                j++;
            } else if (j > 0) {
                j = next_table[poff + j - 1];
            } else {
                text_pos++;
            }
        }
        if (j < plen) return false;  // pattern p not found
    }
    return true;  // all patterns found in order
}

BenchmarkResult tpch_q16(BenchmarkOptions &options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    TPCHTableMetadata *metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    {
        const size_t page_size = metadatap->page_size;
        std::cout << "=== TPCH Q16 ===" << std::endl;
        std::cout << "Page Size: " << page_size << std::endl;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, page_size) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, page_size);
    }

    metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    TPCHTableMetadata &metadata = *metadatap;
    superpage_set_constants(metadata.page_size);
    const size_t page_size = metadata.page_size;
    const size_t num_devices = fds.size();

    const uint64_t nrecs_supplier = metadata.table_supplier_nrows;
    const uint64_t nrecs_part = metadata.table_part_nrows;
    const uint64_t nrecs_partsupp = metadata.table_partsupp_nrows;

    std::cout << "nrecs_supplier: " << nrecs_supplier
              << ", nrecs_part: " << nrecs_part
              << ", nrecs_partsupp: " << nrecs_partsupp
              << ", num_devices: " << num_devices << std::endl;

    // ── Phase 0: Load metadata for SUPPLIER, PART, PARTSUPP ──
    constexpr size_t num_supplier_cols = TPCH::Query::Q16::NUM_SUPPLIER_SCAN_COLS;
    auto supplier_cols_idx = TPCH::Query::Q16::SUPPLIER_SCAN_COLS;
    std::vector<FieldPageInfo> supplier_fields(num_supplier_cols);

    uint32_t saved_column = metadata.column;
    metadata.column = TPCH::common::Table::SUPPLIER;
    prepare_fields_metadata(fds, metadata, page_size, supplier_cols_idx, supplier_fields);

    constexpr size_t num_part_cols = TPCH::Query::Q16::NUM_PART_SCAN_COLS;
    auto part_cols_idx = TPCH::Query::Q16::PART_SCAN_COLS;
    std::vector<FieldPageInfo> part_fields(num_part_cols);

    metadata.column = TPCH::common::Table::PART;
    prepare_fields_metadata(fds, metadata, page_size, part_cols_idx, part_fields);

    constexpr size_t num_partsupp_cols = TPCH::Query::Q16::NUM_PARTSUPP_SCAN_COLS;
    auto partsupp_cols_idx = TPCH::Query::Q16::PARTSUPP_SCAN_COLS;
    std::vector<FieldPageInfo> partsupp_fields(num_partsupp_cols);

    metadata.column = TPCH::common::Table::PARTSUPP;
    prepare_fields_metadata(fds, metadata, page_size, partsupp_cols_idx, partsupp_fields);

    metadata.column = saved_column;

    const FieldPageInfo &fi_s_suppkey  = supplier_fields[0];
    const FieldPageInfo &fi_s_comment  = supplier_fields[1];
    const FieldPageInfo &fi_p_partkey  = part_fields[0];
    const FieldPageInfo &fi_p_brand    = part_fields[1];
    const FieldPageInfo &fi_p_type     = part_fields[2];
    const FieldPageInfo &fi_p_size     = part_fields[3];
    const FieldPageInfo &fi_ps_partkey = partsupp_fields[0];
    const FieldPageInfo &fi_ps_suppkey = partsupp_fields[1];

    std::cout << "  S_SUPPKEY: npages=" << fi_s_suppkey.npages
              << " compression=" << compression_method_name(fi_s_suppkey.compression_method) << std::endl;
    std::cout << "  S_COMMENT: npages=" << fi_s_comment.npages
              << " compression=" << compression_method_name(fi_s_comment.compression_method) << std::endl;
    std::cout << "  P_PARTKEY: npages=" << fi_p_partkey.npages
              << " compression=" << compression_method_name(fi_p_partkey.compression_method) << std::endl;
    std::cout << "  P_BRAND: npages=" << fi_p_brand.npages
              << " compression=" << compression_method_name(fi_p_brand.compression_method) << std::endl;
    std::cout << "  P_TYPE: npages=" << fi_p_type.npages
              << " compression=" << compression_method_name(fi_p_type.compression_method) << std::endl;
    std::cout << "  P_SIZE: npages=" << fi_p_size.npages
              << " compression=" << compression_method_name(fi_p_size.compression_method) << std::endl;
    std::cout << "  PS_PARTKEY: npages=" << fi_ps_partkey.npages
              << " compression=" << compression_method_name(fi_ps_partkey.compression_method) << std::endl;
    std::cout << "  PS_SUPPKEY: npages=" << fi_ps_suppkey.npages
              << " compression=" << compression_method_name(fi_ps_suppkey.compression_method) << std::endl;

    // ── Check for compressed fields ──
    bool any_compressed = false;
    std::vector<FieldPageInfo> all_fields;
    for (auto &fi : supplier_fields) {
        all_fields.push_back(fi);
        if (fi.compression_method != CompressionMethod::NONE) any_compressed = true;
    }
    for (auto &fi : part_fields) {
        all_fields.push_back(fi);
        if (fi.compression_method != CompressionMethod::NONE) any_compressed = true;
    }
    for (auto &fi : partsupp_fields) {
        all_fields.push_back(fi);
        if (fi.compression_method != CompressionMethod::NONE) any_compressed = true;
    }

    const bool use_prefix_sum = options.use_prefix_sum;

    // ── GDS setup (per-thread resources, reusing Q13 pattern) ──
    mb_cufile_driver_open();

    const size_t nthreads = options.nthreads;
    const size_t handles_per_thread = options.gds_num_handlers_per_thread;

    struct Q13LoaderCtx {
        std::vector<std::vector<CUfileHandle_t>> cufile_handles;
        std::vector<std::vector<int>> dup_fds;
        void *io_buf = nullptr;
        size_t io_batch_pages = 0;
        cudaStream_t stream = nullptr;
        NvcompDecompCtx nvctx;
        size_t bytes_read = 0;
        size_t ios_completed = 0;
        size_t kernel_launches = 0;
    };

    std::vector<Q13LoaderCtx> loaders(nthreads);
    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    // Phase 1: create CUfileHandles and streams (io_buf/nvcomp deferred for budget control)
    for (size_t t = 0; t < nthreads; t++) {
        auto &L = loaders[t];
        L.cufile_handles.resize(num_devices);
        L.dup_fds.resize(num_devices);
        for (size_t d = 0; d < num_devices; d++) {
            for (size_t h = 0; h < handles_per_thread; h++) {
                int duped = dup(fds[d]);
                if (duped < 0) { std::cerr << "dup failed" << std::endl; exit(EXIT_FAILURE); }
                L.dup_fds[d].push_back(duped);
                L.cufile_handles[d].push_back(mb_cufile_handle_register(duped));
            }
        }
        CUDA_CHECK(cudaStreamCreate(&L.stream));
    }

    std::cout << "[Q16] nthreads=" << nthreads << std::endl;
    cudaStream_t stream = loaders[0].stream;
    size_t total_bytes_read = 0, total_ios = 0;

    // Pre-allocated thread vector (avoid heap alloc in timed section)
    std::vector<std::thread> _mt_threads;
    _mt_threads.reserve(nthreads);

    // ── Tile-based multi-threaded loader ──
    auto load_field_tile_mt = [&](const FieldPageInfo &field, void *data_buf,
                                   size_t tile_start, size_t tile_npages) {
        if (nthreads <= 1) {
            auto &L = loaders[0];
            NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
            gds_load_field_sync(field, page_size, num_devices,
                                L.cufile_handles, L.io_buf, data_buf,
                                nv, L.stream, L.bytes_read, L.ios_completed,
                                L.kernel_launches,
                                tile_start, tile_npages, tile_start, false, L.io_batch_pages);
            return;
        }
        size_t pages_per_thr = (tile_npages + nthreads - 1) / nthreads;
        auto &threads = _mt_threads; threads.clear();
        for (size_t t = 0; t < nthreads; t++) {
            size_t start = tile_start + t * pages_per_thr;
            if (start >= tile_start + tile_npages) break;
            size_t count = std::min(pages_per_thr, tile_start + tile_npages - start);
            threads.emplace_back([&, t, start, count, tile_start]() {
                auto &L = loaders[t];
                mb_cuda_set_context(cuda_ctx_handle);
                NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                gds_load_field_sync(field, page_size, num_devices,
                                    L.cufile_handles, L.io_buf, data_buf,
                                    nv, L.stream,
                                    L.bytes_read, L.ios_completed,
                                    L.kernel_launches,
                                    start, count, tile_start, false, L.io_batch_pages);
            });
        }
        for (auto &th : threads) th.join();
    };

    // ── Pre-allocate all GPU buffers before timer ──

    // Single staging buffer for batch loading (fixed size)
    constexpr size_t Q16_BATCH_PAGES = 512;
    void *staging_buf = mb_cuda_alloc(Q16_BATCH_PAGES * page_size);

    // Flat pools (reused across phases)
    size_t flat_nrecs_max = std::max({nrecs_supplier, nrecs_part, nrecs_partsupp});
    uint64_t *flat_pool[2];
    CUDA_CHECK(cudaMalloc(&flat_pool[0], flat_nrecs_max * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&flat_pool[1], flat_nrecs_max * sizeof(uint64_t)));

    // Auxiliary pools for Phase 1 PART extraction (brand_ids, type_ids)
    uint32_t *aux_pool[2];
    CUDA_CHECK(cudaMalloc(&aux_pool[0], nrecs_part * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&aux_pool[1], nrecs_part * sizeof(uint32_t)));

    // Prefix-sum pool (reused for all flatten operations)
    size_t ps_max_npages = std::max({fi_s_suppkey.npages, fi_s_comment.npages,
        fi_p_partkey.npages, fi_p_size.npages, fi_p_brand.npages, fi_p_type.npages,
        fi_ps_partkey.npages, fi_ps_suppkey.npages});
    uint64_t *d_ps_pool = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_pool, ps_max_npages * sizeof(uint64_t)));

    // Pre-upload full prefix_sum to GPU and pre-compute batch metadata (Rule 3)
    struct Q16BatchMeta { uint64_t row_start, batch_nrecs; };
    constexpr int Q16_NUM_PS_FIELDS = 7;
    uint64_t *d_full_ps_q16[Q16_NUM_PS_FIELDS] = {};
    std::vector<Q16BatchMeta> q16_batch_metas[Q16_NUM_PS_FIELDS];
    {
        const FieldPageInfo *q16_ps_fields[Q16_NUM_PS_FIELDS] = {
            &fi_s_suppkey, &fi_p_partkey, &fi_p_size, &fi_p_brand,
            &fi_p_type, &fi_ps_partkey, &fi_ps_suppkey};
        for (int f = 0; f < Q16_NUM_PS_FIELDS; f++) {
            auto &fi = *q16_ps_fields[f];
            CUDA_CHECK(cudaMalloc(&d_full_ps_q16[f], (fi.npages + 1) * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_full_ps_q16[f], fi.prefix_sum_nrecs,
                                  (fi.npages + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            for (size_t pg = 0; pg < fi.npages; pg += Q16_BATCH_PAGES) {
                size_t bnp = std::min(Q16_BATCH_PAGES, fi.npages - pg);
                q16_batch_metas[f].push_back({fi.prefix_sum_nrecs[pg],
                    fi.prefix_sum_nrecs[pg + bnp] - fi.prefix_sum_nrecs[pg]});
            }
        }
    }

    // S_COMMENT batch metadata (separate field, not in the 7-field array)
    std::vector<Q16BatchMeta> q16_bm_s_comment;
    for (size_t pg = 0; pg < fi_s_comment.npages; pg += Q16_BATCH_PAGES) {
        size_t bnp = std::min(Q16_BATCH_PAGES, fi_s_comment.npages - pg);
        if (fi_s_comment.prefix_sum_nrecs) {
            q16_bm_s_comment.push_back({fi_s_comment.prefix_sum_nrecs[pg],
                fi_s_comment.prefix_sum_nrecs[pg + bnp] - fi_s_comment.prefix_sum_nrecs[pg]});
        } else {
            q16_bm_s_comment.push_back({0, 0});
        }
    }

    // S_COMMENT prefix_sum (separate, lives through supplier scan)
    uint64_t *d_ps_s_comment = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_s_comment, fi_s_comment.npages * sizeof(uint64_t)));

    // Excluded supplier resources
    uint32_t *d_excl_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_excl_count, sizeof(uint32_t)));
    uint32_t max_excl_capacity = 1;
    while (max_excl_capacity < nrecs_supplier * 4 + 16) max_excl_capacity <<= 1;
    uint64_t *d_excl_keys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_excl_keys, max_excl_capacity * sizeof(uint64_t)));

    // PART hash table
    uint32_t ht_capacity = 1;
    while (ht_capacity < nrecs_part * 2) ht_capacity <<= 1;
    uint32_t ht_mask = ht_capacity - 1;
    uint64_t *d_ht_keys = nullptr;
    uint32_t *d_ht_group_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_keys, ht_capacity * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_group_ids, ht_capacity * sizeof(uint32_t)));

    // Type dictionary
    uint64_t *d_dict_keys = nullptr;
    uint32_t *d_dict_type_ids = nullptr;
    char *d_dict_strs = nullptr;
    uint16_t *d_dict_lens = nullptr;
    uint32_t *d_type_id_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dict_keys, Q16_TYPE_DICT_CAP * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_dict_type_ids, Q16_TYPE_DICT_CAP * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dict_strs, (size_t)Q16_TYPE_DICT_CAP * Q16_TYPE_STR_MAX));
    CUDA_CHECK(cudaMalloc(&d_dict_lens, Q16_TYPE_DICT_CAP * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_type_id_counter, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_dict_keys, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dict_type_ids, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_type_id_counter, 0, sizeof(uint32_t), stream));

    // KMP tables for '%Customer%Complaints%'
    const char excl_patterns_str[] = "CustomerComplaints";
    const int excl_pattern_offsets[] = {0, 8};
    const int excl_pattern_lengths[] = {8, 10};
    constexpr int excl_total_chars = 18;
    constexpr int excl_num_patterns = 2;
    int excl_next[excl_total_chars] = {};
    for (int p = 0; p < excl_num_patterns; p++) {
        int off = excl_pattern_offsets[p];
        int len = excl_pattern_lengths[p];
        for (int i = 1; i < len; i++) {
            int j = excl_next[off + i - 1];
            while (j > 0 && excl_patterns_str[off + i] != excl_patterns_str[off + j])
                j = excl_next[off + j - 1];
            if (excl_patterns_str[off + i] == excl_patterns_str[off + j]) j++;
            excl_next[off + i] = j;
        }
    }
    char *d_kmp_patterns = nullptr;
    int *d_kmp_next = nullptr, *d_kmp_offsets = nullptr, *d_kmp_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kmp_patterns, excl_total_chars));
    CUDA_CHECK(cudaMemcpy(d_kmp_patterns, excl_patterns_str, excl_total_chars, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_kmp_next, excl_total_chars * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_kmp_next, excl_next, excl_total_chars * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_kmp_offsets, excl_num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_kmp_offsets, excl_pattern_offsets, excl_num_patterns * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_kmp_lengths, excl_num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_kmp_lengths, excl_pattern_lengths, excl_num_patterns * sizeof(int), cudaMemcpyHostToDevice));

    // Pre-allocate Q16 pipeline buffers (sort+RLE, no malloc inside timed section)
    Q16PipelineBuffers q16_bufs{};
    {
        size_t n = nrecs_partsupp;
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_emit_pairs, n * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_sort_alt, n * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_unique_keys, n * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_unique_counts, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_num_unique_ptr, sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_group_ids, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_group_ids_alt, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_result_gids, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_result_counts, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_num_groups_ptr, sizeof(uint64_t)));
        q16_bufs.cub_temp_bytes = q16_pipeline_cub_temp_size(n);
        CUDA_CHECK(cudaMalloc(&q16_bufs.d_cub_temp, q16_bufs.cub_temp_bytes));
        // Host result buffers (brand*type*size groups, 256K is a safe upper bound)
        constexpr size_t Q16_MAX_GROUPS = 256 * 1024;
        q16_bufs.h_result_capacity = Q16_MAX_GROUPS;
        q16_bufs.h_gids = (uint32_t *)malloc(Q16_MAX_GROUPS * sizeof(uint32_t));
        q16_bufs.h_counts = (uint32_t *)malloc(Q16_MAX_GROUPS * sizeof(uint32_t));
    }

    // Phase 2: allocate per-loader io_buf with dynamic sizing (gpu_mem <= 40 GiB)
    {
        static constexpr size_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_before_loaders = 0;
        cudaMemGetInfo(&gpu_free_before_loaders, &gpu_total_dummy);
        size_t non_loader_bytes = gpu_free_start - gpu_free_before_loaders;
        size_t q16_io_pages = 1;
        if (GPU_MEM_BUDGET > non_loader_bytes) {
            size_t nvcomp_per_loader = 2 * 1024 * 1024;  // ~2 MiB nvcomp overhead
            size_t budget_for_loaders = GPU_MEM_BUDGET - non_loader_bytes;
            size_t per_loader_budget = budget_for_loaders / nthreads;
            if (per_loader_budget > nvcomp_per_loader) {
                q16_io_pages = (per_loader_budget - nvcomp_per_loader) / page_size;
            }
            q16_io_pages = std::max(q16_io_pages, (size_t)1);
            q16_io_pages = std::min(q16_io_pages, (size_t)GDS_SYNC_BATCH_PAGES);
        }
        printf("[Q16-BUDGET] gpu_free_start=%zu gpu_free_before_loaders=%zu non_loader=%zu MiB\n",
               gpu_free_start, gpu_free_before_loaders, non_loader_bytes / (1024*1024));
        for (size_t t = 0; t < nthreads; t++) {
            auto &L = loaders[t];
            L.io_batch_pages = q16_io_pages;
            L.io_buf = mb_cuda_alloc(q16_io_pages * page_size);
            GDS_CHECK(cuFileBufRegister(L.io_buf, q16_io_pages * page_size, 0));
            if (any_compressed) {
                nvcomp_decompctx_alloc(L.nvctx, q16_io_pages, page_size, all_fields);
            }
        }
        std::cout << "[Q16] Per-loader IO: " << q16_io_pages << " pages ("
                  << (q16_io_pages * page_size / (1024*1024)) << " MiB), "
                  << nthreads << " loaders" << std::endl;
    }

    // Pre-allocate host vectors (avoid heap alloc in timed section)
    std::vector<uint64_t> h_ps_fallback(std::max(fi_s_comment.npages, (uint64_t)1));
    std::vector<std::pair<uint32_t, uint32_t>> q16_raw_result;
    q16_raw_result.reserve(4096);

    // Upload full S_COMMENT prefix_sum before measurement (Rule 3)
    if (fi_s_comment.prefix_sum_nrecs != nullptr) {
        CUDA_CHECK(cudaMemcpy(d_ps_s_comment, fi_s_comment.prefix_sum_nrecs + 1,
                              fi_s_comment.npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
    } else {
        // Fallback: scan pages in batches to collect nalloc, build prefix_sum on host
        uint64_t cum = 0;
        for (size_t pg = 0; pg < fi_s_comment.npages; pg += Q16_BATCH_PAGES) {
            size_t bnp = std::min(Q16_BATCH_PAGES, fi_s_comment.npages - pg);
            load_field_tile_mt(fi_s_comment, staging_buf, pg, bnp);
            for (size_t p = 0; p < bnp; p++) {
                uint32_t nalloc = 0;
                CUDA_CHECK(cudaMemcpy(&nalloc,
                    static_cast<const char *>(staging_buf) + p * page_size,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cum += nalloc;
                h_ps_fallback[pg + p] = cum;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_ps_s_comment, h_ps_fallback.data(),
                              fi_s_comment.npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t golap_gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // ════════════════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    // ════════════════════════════════════════════════════════

    // ── Phase 0: SUPPLIER anti-join (batch) ──

    // Batch flatten S_SUPPKEY → flat_pool[0]
    std::cout << "[Q16] Loading S_SUPPKEY (" << fi_s_suppkey.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_s_suppkey.npages; pg += Q16_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q16_BATCH_PAGES, fi_s_suppkey.npages - pg);
          load_field_tile_mt(fi_s_suppkey, staging_buf, pg, bnp);
          auto &bm = q16_batch_metas[0][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q16[0], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;
          q13_flatten_int64_pages_ps(
              static_cast<const char *>(staging_buf), page_size,
              d_ps_pool, bnp, bm.batch_nrecs, flat_pool[0] + bm.row_start, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    // Batch scan S_COMMENT → excluded suppkeys
    std::cout << "[Q16] Loading S_COMMENT (" << fi_s_comment.npages << " pages)..." << std::endl;

    // Batch scan S_COMMENT → excluded suppkeys
    CUDA_CHECK(cudaMemsetAsync(d_excl_count, 0, sizeof(uint32_t), stream));
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_s_comment.npages; pg += Q16_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q16_BATCH_PAGES, fi_s_comment.npages - pg);
          load_field_tile_mt(fi_s_comment, staging_buf, pg, bnp);
          auto &bm = q16_bm_s_comment[bi];
          uint64_t row_base = bm.row_start;
          uint64_t nrecs_batch = bm.batch_nrecs;
          q16_supplier_scan_batch(
              static_cast<const char *>(staging_buf),
              d_ps_s_comment, fi_s_comment.npages, pg, page_size,
              flat_pool[0], nrecs_batch, row_base,
              d_kmp_patterns, d_kmp_next, d_kmp_offsets, d_kmp_lengths,
              excl_num_patterns, flat_pool[1], d_excl_count, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    uint32_t h_excl_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_excl_count, d_excl_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "[Q16] Excluded suppliers: " << h_excl_count << std::endl;

    // Build excluded suppkey hash table
    uint32_t excl_capacity = 1;
    while (excl_capacity < (uint32_t)h_excl_count * 4 + 16)
        excl_capacity <<= 1;
    uint32_t excl_mask = excl_capacity - 1;
    CUDA_CHECK(cudaMemsetAsync(d_excl_keys, 0xFF, excl_capacity * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    q16_build_excl_ht(flat_pool[1], h_excl_count,
                      d_excl_keys, excl_mask, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Phase 1: PART scan + hash table (batch) ──

    // Batch flatten P_PARTKEY → flat_pool[0]
    std::cout << "[Q16] Loading P_PARTKEY (" << fi_p_partkey.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_p_partkey.npages; pg += Q16_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q16_BATCH_PAGES, fi_p_partkey.npages - pg);
          load_field_tile_mt(fi_p_partkey, staging_buf, pg, bnp);
          auto &bm = q16_batch_metas[1][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q16[1], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;
          q13_flatten_int64_pages_ps(
              static_cast<const char *>(staging_buf), page_size,
              d_ps_pool, bnp, bm.batch_nrecs, flat_pool[0] + bm.row_start, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    // Batch flatten P_SIZE → flat_pool[1]
    std::cout << "[Q16] Loading P_SIZE (" << fi_p_size.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_p_size.npages; pg += Q16_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q16_BATCH_PAGES, fi_p_size.npages - pg);
          load_field_tile_mt(fi_p_size, staging_buf, pg, bnp);
          auto &bm = q16_batch_metas[2][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q16[2], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;
          q13_flatten_int32_pages_ps(
              static_cast<const char *>(staging_buf), page_size,
              d_ps_pool, bnp, bm.batch_nrecs, flat_pool[1] + bm.row_start, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    // ── Phase 1b: Extract brand_ids, type_ids on GPU (batch) ──
    constexpr uint32_t BRAND45_ID = 19;
    constexpr uint32_t P_BRAND_PADDED_LEN = 12;

    // Batch extract P_BRAND → aux_pool[0] (brand_ids)
    std::cout << "[Q16] Loading P_BRAND (" << fi_p_brand.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_p_brand.npages; pg += Q16_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q16_BATCH_PAGES, fi_p_brand.npages - pg);
          load_field_tile_mt(fi_p_brand, staging_buf, pg, bnp);
          auto &bm = q16_batch_metas[3][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q16[3], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;
          q16_extract_brand_ids(
              static_cast<const char *>(staging_buf),
              d_ps_pool, bnp, page_size,
              P_BRAND_PADDED_LEN, bm.batch_nrecs, aux_pool[0] + bm.row_start, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    // Batch extract P_TYPE → aux_pool[1] (type_ids)
    std::cout << "[Q16] Loading P_TYPE (" << fi_p_type.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_p_type.npages; pg += Q16_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q16_BATCH_PAGES, fi_p_type.npages - pg);
          load_field_tile_mt(fi_p_type, staging_buf, pg, bnp);
          auto &bm = q16_batch_metas[4][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q16[4], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;
          q16_extract_type_ids(
              static_cast<const char *>(staging_buf),
              d_ps_pool, bnp, page_size, bm.batch_nrecs,
              d_dict_keys, d_dict_type_ids, d_dict_strs, d_dict_lens,
              d_type_id_counter, aux_pool[1] + bm.row_start, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    uint32_t num_types = 0;
    CUDA_CHECK(cudaMemcpy(&num_types, d_type_id_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "[Q16] P_TYPE distinct values (excluding MEDIUM POLISHED): " << num_types << std::endl;

    // Copy type dictionary strings to host for result decoding
    char h_dict_strs[Q16_TYPE_DICT_CAP * Q16_TYPE_STR_MAX];
    uint16_t h_dict_lens[Q16_TYPE_DICT_CAP];
    uint32_t h_dict_type_ids[Q16_TYPE_DICT_CAP];
    CUDA_CHECK(cudaMemcpy(h_dict_strs, d_dict_strs,
        (size_t)Q16_TYPE_DICT_CAP * Q16_TYPE_STR_MAX, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dict_lens, d_dict_lens,
        Q16_TYPE_DICT_CAP * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dict_type_ids, d_dict_type_ids,
        Q16_TYPE_DICT_CAP * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Build reverse lookup: type_id → string
    char type_str_pool[150][Q16_TYPE_STR_MAX];
    uint16_t type_str_lens[150];
    memset(type_str_pool, 0, sizeof(type_str_pool));
    memset(type_str_lens, 0, sizeof(type_str_lens));
    for (uint32_t s = 0; s < Q16_TYPE_DICT_CAP; s++) {
        uint32_t tid = h_dict_type_ids[s];
        if (tid != UINT32_MAX && tid < 150) {
            memcpy(type_str_pool[tid], h_dict_strs + (size_t)s * Q16_TYPE_STR_MAX,
                   h_dict_lens[s]);
            type_str_lens[tid] = h_dict_lens[s];
        }
    }

    // P_SIZE: cast uint64_t → uint32_t (output to upper half of flat_pool[1])
    uint32_t *d_p_size_u32 = reinterpret_cast<uint32_t *>(
        reinterpret_cast<char *>(flat_pool[1]) + nrecs_part * sizeof(uint64_t));
    q16_cast_u64_to_u32(flat_pool[1], d_p_size_u32, nrecs_part, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Build PART hash table
    uint64_t p_size_bitmask = (1ULL << 49) | (1ULL << 14) | (1ULL << 23) | (1ULL << 45)
                            | (1ULL << 19) | (1ULL <<  3) | (1ULL << 36) | (1ULL <<  9);

    std::cout << "[Q16] Building PART hash table..." << std::endl;
    CUDA_CHECK(cudaMemsetAsync(d_ht_keys, 0xFF, ht_capacity * sizeof(uint64_t), stream));
    q16_build_part_hashtable(flat_pool[0], aux_pool[0], aux_pool[1], d_p_size_u32,
        nrecs_part, p_size_bitmask, BRAND45_ID, num_types,
        d_ht_keys, d_ht_group_ids, ht_mask, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Phase 2: PARTSUPP load + flatten (batch) ──

    // Batch flatten PS_PARTKEY → flat_pool[0]
    std::cout << "[Q16] Loading PS_PARTKEY (" << fi_ps_partkey.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_ps_partkey.npages; pg += Q16_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q16_BATCH_PAGES, fi_ps_partkey.npages - pg);
          load_field_tile_mt(fi_ps_partkey, staging_buf, pg, bnp);
          auto &bm = q16_batch_metas[5][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q16[5], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;
          q13_flatten_int64_pages_ps(
              static_cast<const char *>(staging_buf), page_size,
              d_ps_pool, bnp, bm.batch_nrecs, flat_pool[0] + bm.row_start, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    // Batch flatten PS_SUPPKEY → flat_pool[1]
    std::cout << "[Q16] Loading PS_SUPPKEY (" << fi_ps_suppkey.npages << " pages)..." << std::endl;
    {
      size_t bi = 0;
      for (size_t pg = 0; pg < fi_ps_suppkey.npages; pg += Q16_BATCH_PAGES, bi++) {
          size_t bnp = std::min(Q16_BATCH_PAGES, fi_ps_suppkey.npages - pg);
          load_field_tile_mt(fi_ps_suppkey, staging_buf, pg, bnp);
          auto &bm = q16_batch_metas[6][bi];
          compute_batch_ps_kernel<<<(bnp + 255) / 256, 256, 0, stream>>>(
              d_full_ps_q16[6], d_ps_pool, (uint32_t)pg, (uint32_t)bnp);
          s_kernel_launches++;
          q13_flatten_int64_pages_ps(
              static_cast<const char *>(staging_buf), page_size,
              d_ps_pool, bnp, bm.batch_nrecs, flat_pool[1] + bm.row_start, stream);
          s_kernel_launches++;
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
    }

    // ── Phase 3: Run Q16 pipeline ──
    std::cout << "[Q16] Running Q16 pipeline..." << std::endl;

    q16_golap_pipeline(q16_bufs,
        d_ht_keys, d_ht_group_ids, ht_mask,
        d_excl_keys, excl_mask,
        flat_pool[0], flat_pool[1], nrecs_partsupp,
        q16_raw_result, stream);
    s_kernel_launches++;

    auto total_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(total_end - total_start).count();

    // Aggregate I/O stats
    for (size_t t = 0; t < nthreads; t++) {
        total_bytes_read += loaders[t].bytes_read;
        total_ios += loaders[t].ios_completed;
    }

    for (size_t t = 0; t < nthreads; t++)
        s_kernel_launches += loaders[t].kernel_launches;

    // ── Phase 4: Expand group_id → (brand, type, size) and sort ──
    std::vector<Q16ResultRow> q16_result;
    q16_result.reserve(q16_raw_result.size());

    for (auto &[gid, cnt] : q16_raw_result) {
        uint32_t size_val = (gid % 50) + 1;
        uint32_t rem = gid / 50;
        uint32_t type_id = rem % num_types;
        uint32_t brand_id = rem / num_types;

        char brand_buf[16];
        snprintf(brand_buf, sizeof(brand_buf), "Brand#%d%d",
                 (int)(brand_id / 5) + 1, (int)(brand_id % 5) + 1);

        Q16ResultRow row;
        row.p_brand = brand_buf;
        row.p_type = (type_id < num_types)
            ? std::string(type_str_pool[type_id], type_str_lens[type_id]) : "UNKNOWN";
        row.p_size = size_val;
        row.supplier_cnt = cnt;
        q16_result.push_back(row);
    }

    // Sort: supplier_cnt DESC, p_brand ASC, p_type ASC, p_size ASC
    std::sort(q16_result.begin(), q16_result.end(),
        [](const Q16ResultRow &a, const Q16ResultRow &b) {
            if (a.supplier_cnt != b.supplier_cnt) return a.supplier_cnt > b.supplier_cnt;
            if (a.p_brand != b.p_brand) return a.p_brand < b.p_brand;
            if (a.p_type != b.p_type) return a.p_type < b.p_type;
            return a.p_size < b.p_size;
        });

    // ── Print results ──
    std::cout << "\n=== TPC-H Q16 Result ===" << std::endl;
    std::cout << "p_brand    | p_type                    | p_size | supplier_cnt" << std::endl;
    std::cout << "-----------+---------------------------+--------+-------------" << std::endl;
    for (size_t i = 0; i < std::min(q16_result.size(), (size_t)50); i++) {
        auto &r = q16_result[i];
        printf("%-10s | %-25s | %6d | %12u\n",
               r.p_brand.c_str(), r.p_type.c_str(), r.p_size, r.supplier_cnt);
    }
    if (q16_result.size() > 50) {
        std::cout << "... (" << q16_result.size() << " total rows)" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total rows: " << q16_result.size() << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_ios << std::endl;
    std::cout << "Total bytes read: " << total_bytes_read << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    mb_cuda_free(staging_buf);
    for (int i = 0; i < 2; i++) cudaFree(flat_pool[i]);
    for (int i = 0; i < 2; i++) cudaFree(aux_pool[i]);
    cudaFree(d_ps_pool);
    for (int f = 0; f < Q16_NUM_PS_FIELDS; f++) cudaFree(d_full_ps_q16[f]);
    cudaFree(d_ps_s_comment);
    cudaFree(d_excl_count);
    cudaFree(d_excl_keys);
    cudaFree(d_ht_keys);
    cudaFree(d_ht_group_ids);
    cudaFree(d_dict_keys);
    cudaFree(d_dict_type_ids);
    cudaFree(d_dict_strs);
    cudaFree(d_dict_lens);
    cudaFree(d_type_id_counter);
    cudaFree(d_kmp_patterns);
    cudaFree(d_kmp_next);
    cudaFree(d_kmp_offsets);
    cudaFree(d_kmp_lengths);
    cudaFree(q16_bufs.d_emit_pairs);
    cudaFree(q16_bufs.d_sort_alt);
    cudaFree(q16_bufs.d_unique_keys);
    cudaFree(q16_bufs.d_unique_counts);
    cudaFree(q16_bufs.d_num_unique_ptr);
    cudaFree(q16_bufs.d_group_ids);
    cudaFree(q16_bufs.d_group_ids_alt);
    cudaFree(q16_bufs.d_result_gids);
    cudaFree(q16_bufs.d_result_counts);
    cudaFree(q16_bufs.d_num_groups_ptr);
    cudaFree(q16_bufs.d_cub_temp);
    free(q16_bufs.h_gids);
    free(q16_bufs.h_counts);

    for (size_t t = 0; t < nthreads; t++) {
        auto &L = loaders[t];
        if (any_compressed) nvcomp_decompctx_free(L.nvctx);
        cuFileBufDeregister(L.io_buf);
        mb_cuda_free(L.io_buf);
        CUDA_CHECK(cudaStreamDestroy(L.stream));
        for (size_t d = 0; d < num_devices; d++) {
            for (size_t h = 0; h < L.cufile_handles[d].size(); h++) {
                cuFileHandleDeregister(L.cufile_handles[d][h]);
                close(L.dup_fds[d][h]);
            }
        }
    }

    size_t total_pages = 0;
    for (const auto &fi : supplier_fields) total_pages += fi.npages;
    for (const auto &fi : part_fields) total_pages += fi.npages;
    for (const auto &fi : partsupp_fields) total_pages += fi.npages;

    std::string comp_str = collect_compression_methods({supplier_fields, part_fields, partsupp_fields});
    free_fields_metadata(supplier_fields);
    free_fields_metadata(part_fields);
    free_fields_metadata(partsupp_fields);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = total_ios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = comp_str,
        .gpu_mem_bytes = golap_gpu_mem_bytes,
        .gpu_app_bytes = golap_gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

// ═══════════════════════════════════════════════════════════════
// Cross-column pipelined loader (SSB LoWorkerCtx pattern)
//
// Each worker owns 2 staging buffers (double-buffered) and pipelines
// IO-decomp across columns: while GPU decompresses column N,
// the host thread reads column N+1 from NVMe via cuFileRead.
// ═══════════════════════════════════════════════════════════════

static constexpr size_t TPCH_PIPE_MAX_DEVICES = 8;

struct TpchPipelineCol {
    const FieldPageInfo *field;
    void *data_buf;                    // per-column destination GPU buffer
    const uint32_t *page_indices;      // active pages (nullptr = contiguous from page_start)
    size_t num_pages;                  // total active pages for this column
    size_t page_start;                 // first page (contiguous mode start / output_page_offset)
};

struct TpchWorkerCtx {
    CUfileHandle_t cufile_handles[TPCH_PIPE_MAX_DEVICES];
    int dup_fds[TPCH_PIPE_MAX_DEVICES];
    void *staging_buf[2];              // double-buffered IO staging (cuFile registered)
    cudaStream_t io_stream;
    cudaStream_t decomp_stream;
    cudaEvent_t io_done;
    cudaEvent_t buf_done[2];
    NvcompDecompCtx nvctx;
    uint32_t *h_contig_indices;        // pre-allocated for contiguous index generation
    size_t max_ppw;                    // max pages per worker (staging_buf capacity)
    size_t bytes_read;
    size_t ios_completed;
    size_t kernel_launches;
};

static void tpch_worker_alloc(
    TpchWorkerCtx *workers, size_t nworkers,
    const int *fds, size_t num_devices, size_t page_size,
    size_t max_ppw, bool any_compressed,
    const std::vector<FieldPageInfo> &all_fields)
{
    for (size_t t = 0; t < nworkers; t++) {
        auto &w = workers[t];
        w.bytes_read = 0;
        w.ios_completed = 0;
        w.kernel_launches = 0;
        w.max_ppw = max_ppw;
        for (size_t d = 0; d < num_devices; d++) {
            w.dup_fds[d] = dup(fds[d]);
            if (w.dup_fds[d] < 0) { std::cerr << "dup failed" << std::endl; exit(EXIT_FAILURE); }
            w.cufile_handles[d] = mb_cufile_handle_register(w.dup_fds[d]);
        }
        size_t buf_size = max_ppw * page_size;
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
            nvcomp_decompctx_alloc(w.nvctx, max_ppw, page_size, all_fields);
            nvcomp_decompctx_alloc_db(w.nvctx, max_ppw);
        }
        CUDA_CHECK(cudaMallocHost(&w.h_contig_indices, max_ppw * sizeof(uint32_t)));
    }
}

static void tpch_worker_free(
    TpchWorkerCtx *workers, size_t nworkers,
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
        cudaFreeHost(w.h_contig_indices);
        for (size_t d = 0; d < num_devices; d++) {
            cuFileHandleDeregister(w.cufile_handles[d]);
            close(w.dup_fds[d]);
        }
    }
}

// Read one column's pages for this worker into staging_buf[buf_idx],
// set up nvCOMP pointers, and scatter uncompressed pages via D→D copy.
// Returns decomp_count (number of pages needing nvCOMP decomp).
static size_t tpch_worker_read_column(
    TpchWorkerCtx &w,
    const FieldPageInfo &field,
    size_t page_size, size_t num_devices,
    const uint32_t *page_indices, size_t num_pages,
    char *data_buf, size_t output_page_offset,
    int buf_idx)
{
    if (num_pages == 0) return 0;

    const bool is_compressed = (field.compression_method != CompressionMethod::NONE);
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    char *stg = static_cast<char *>(w.staging_buf[buf_idx]);

    // Compute per-device data sizes and cumulative offsets
    size_t dev_data_size[TPCH_PIPE_MAX_DEVICES] = {};
    for (size_t k = 0; k < num_pages; k++) {
        size_t page_rel = page_indices[k];
        uint64_t page_id = field.start_page_id + page_rel;
        size_t d = page_id % num_devices;
        if (is_compressed)
            dev_data_size[d] += roundup4096(field.compressed_page_sizes[page_rel]);
        else
            dev_data_size[d] += page_size;
    }
    size_t dev_io_start[TPCH_PIPE_MAX_DEVICES + 1] = {};
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
                        std::cerr << "cuFileRead fail: dev=" << d << " size=" << run_size
                                  << " nread=" << nread << std::endl;
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
                std::cerr << "cuFileRead fail: dev=" << d << " size=" << run_size
                          << " nread=" << nread << std::endl;
            else { w.bytes_read += nread; w.ios_completed++; }
        }
    }

    // Scatter: setup nvCOMP pointers or D→D copy
    void  **h_cp  = (buf_idx == 0) ? w.nvctx.h_comp_ptrs   : w.nvctx.h_comp_ptrs_1;
    void  **h_dp  = (buf_idx == 0) ? w.nvctx.h_decomp_ptrs : w.nvctx.h_decomp_ptrs_1;
    size_t *h_cs  = (buf_idx == 0) ? w.nvctx.h_comp_sizes  : w.nvctx.h_comp_sizes_1;
    size_t *h_ds  = (buf_idx == 0) ? w.nvctx.h_decomp_sizes: w.nvctx.h_decomp_sizes_1;

    size_t decomp_count = 0;
    size_t dev_off[TPCH_PIPE_MAX_DEVICES] = {};
    for (size_t k = 0; k < num_pages; k++) {
        size_t page_rel = page_indices[k];
        uint64_t page_id = field.start_page_id + page_rel;
        size_t d = page_id % num_devices;

        char *io_src = stg + dev_io_start[d] + dev_off[d];
        char *dst = data_buf + (page_rel - output_page_offset) * page_size;

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

// Process multiple columns with double-buffered IO-decomp pipeline.
// Each column can have its own page_indices and page count.
// worker_idx/total_workers determine which pages this worker handles.
static void tpch_worker_pipeline(
    TpchWorkerCtx &w,
    const TpchPipelineCol *cols, size_t num_cols,
    size_t page_size, size_t num_devices,
    size_t worker_idx, size_t total_workers,
    CUcontext cuda_ctx)
{
    mb_cuda_set_context(cuda_ctx);

    int buf = 0;
    bool submitted[2] = {false, false};

    for (size_t fi = 0; fi < num_cols; fi++) {
        // Wait for staging_buf[buf] to be available
        if (submitted[buf])
            CUDA_CHECK(cudaEventSynchronize(w.buf_done[buf]));

        // Determine this worker's share of column fi
        size_t n = cols[fi].num_pages;
        size_t ppw = (n + total_workers - 1) / total_workers;
        size_t w_off = worker_idx * ppw;
        size_t w_count = (w_off >= n) ? 0 : std::min(ppw, n - w_off);

        // Build page indices for this worker's share
        const uint32_t *pi;
        if (cols[fi].page_indices) {
            pi = cols[fi].page_indices + w_off;
        } else {
            for (size_t i = 0; i < w_count; i++)
                w.h_contig_indices[i] = (uint32_t)(cols[fi].page_start + w_off + i);
            pi = w.h_contig_indices;
        }

        // IO: read column fi pages → staging_buf[buf]
        size_t decomp_count = tpch_worker_read_column(
            w, *cols[fi].field, page_size, num_devices,
            pi, w_count,
            static_cast<char *>(cols[fi].data_buf),
            cols[fi].page_start, buf);

        // Decomp: launch async on decomp_stream
        if (decomp_count > 0) {
            CUDA_CHECK(cudaEventRecord(w.io_done, w.io_stream));
            CUDA_CHECK(cudaStreamWaitEvent(w.decomp_stream, w.io_done));
            nvcomp_decompctx_run(cols[fi].field->compression_method,
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

// ════════════════════════════════════════════════════════════
// TPC-H Q5: Local Supplier Volume (GOLAP sync mode)
// ════════════════════════════════════════════════════════════
BenchmarkResult tpch_q5(BenchmarkOptions &options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    TPCHTableMetadata *metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    {
        const size_t ps = metadatap->page_size;
        std::cout << "=== TPCH Q5 ===" << std::endl;
        std::cout << "Page Size: " << ps << std::endl;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, ps) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, ps);
    }

    metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    TPCHTableMetadata &metadata = *metadatap;
    superpage_set_constants(metadata.page_size);
    const size_t page_size = metadata.page_size;
    const size_t num_devices = fds.size();

    const uint64_t nrecs_region   = metadata.table_region_nrows;
    const uint64_t nrecs_nation   = metadata.table_nation_nrows;
    const uint64_t nrecs_supplier = metadata.table_supplier_nrows;
    const uint64_t nrecs_customer = metadata.table_customer_nrows;
    const uint64_t nrecs_orders   = metadata.table_orders_nrows;
    const uint64_t nrecs_lineitem = metadata.table_lineitem_nrows;

    std::cout << "nrecs: region=" << nrecs_region
              << ", nation=" << nrecs_nation
              << ", supplier=" << nrecs_supplier
              << ", customer=" << nrecs_customer
              << ", orders=" << nrecs_orders
              << ", lineitem=" << nrecs_lineitem
              << ", num_devices=" << num_devices << std::endl;

    // ── Load metadata for all tables ──
    std::vector<FieldPageInfo> region_fields(TPCH::Query::Q5::NUM_REGION_SCAN_COLS);
    std::vector<FieldPageInfo> nation_fields(TPCH::Query::Q5::NUM_NATION_SCAN_COLS);
    std::vector<FieldPageInfo> supplier_fields(TPCH::Query::Q5::NUM_SUPPLIER_SCAN_COLS);
    std::vector<FieldPageInfo> customer_fields(TPCH::Query::Q5::NUM_CUSTOMER_SCAN_COLS);
    std::vector<FieldPageInfo> orders_fields(TPCH::Query::Q5::NUM_ORDERS_SCAN_COLS);
    std::vector<FieldPageInfo> lineitem_fields(TPCH::Query::Q5::NUM_LINEITEM_SCAN_COLS);

    uint32_t saved_column = metadata.column;
    metadata.column = TPCH::common::Table::REGION;
    prepare_fields_metadata(fds, metadata, page_size,
        TPCH::Query::Q5::REGION_SCAN_COLS, region_fields);
    metadata.column = TPCH::common::Table::NATION;
    prepare_fields_metadata(fds, metadata, page_size,
        TPCH::Query::Q5::NATION_SCAN_COLS, nation_fields);
    metadata.column = TPCH::common::Table::SUPPLIER;
    prepare_fields_metadata(fds, metadata, page_size,
        TPCH::Query::Q5::SUPPLIER_SCAN_COLS, supplier_fields);
    metadata.column = TPCH::common::Table::CUSTOMER;
    prepare_fields_metadata(fds, metadata, page_size,
        TPCH::Query::Q5::CUSTOMER_SCAN_COLS, customer_fields);
    metadata.column = TPCH::common::Table::ORDERS;
    prepare_fields_metadata(fds, metadata, page_size,
        TPCH::Query::Q5::ORDERS_SCAN_COLS, orders_fields);
    metadata.column = TPCH::common::Table::LINEITEM;
    prepare_fields_metadata(fds, metadata, page_size,
        TPCH::Query::Q5::LINEITEM_SCAN_COLS, lineitem_fields);
    metadata.column = saved_column;

    const FieldPageInfo &fi_r_regionkey = region_fields[0];
    const FieldPageInfo &fi_r_name      = region_fields[1];
    const FieldPageInfo &fi_n_nationkey = nation_fields[0];
    const FieldPageInfo &fi_n_name      = nation_fields[1];
    const FieldPageInfo &fi_n_regionkey = nation_fields[2];
    const FieldPageInfo &fi_s_suppkey   = supplier_fields[0];
    const FieldPageInfo &fi_s_nationkey = supplier_fields[1];
    const FieldPageInfo &fi_c_custkey   = customer_fields[0];
    const FieldPageInfo &fi_c_nationkey = customer_fields[1];
    const FieldPageInfo &fi_o_orderkey  = orders_fields[0];
    const FieldPageInfo &fi_o_custkey   = orders_fields[1];
    const FieldPageInfo &fi_o_orderdate = orders_fields[2];
    const FieldPageInfo &fi_l_orderkey  = lineitem_fields[0];
    const FieldPageInfo &fi_l_suppkey   = lineitem_fields[1];
    const FieldPageInfo &fi_l_extprice  = lineitem_fields[2];
    const FieldPageInfo &fi_l_discount  = lineitem_fields[3];

    std::cout << "  S_SUPPKEY: npages=" << fi_s_suppkey.npages
              << " compression=" << compression_method_name(fi_s_suppkey.compression_method) << std::endl;
    std::cout << "  S_NATIONKEY: npages=" << fi_s_nationkey.npages
              << " compression=" << compression_method_name(fi_s_nationkey.compression_method) << std::endl;
    std::cout << "  C_CUSTKEY: npages=" << fi_c_custkey.npages
              << " compression=" << compression_method_name(fi_c_custkey.compression_method) << std::endl;
    std::cout << "  C_NATIONKEY: npages=" << fi_c_nationkey.npages
              << " compression=" << compression_method_name(fi_c_nationkey.compression_method) << std::endl;
    std::cout << "  O_ORDERKEY: npages=" << fi_o_orderkey.npages
              << " compression=" << compression_method_name(fi_o_orderkey.compression_method) << std::endl;
    std::cout << "  O_CUSTKEY: npages=" << fi_o_custkey.npages
              << " compression=" << compression_method_name(fi_o_custkey.compression_method) << std::endl;
    std::cout << "  O_ORDERDATE: npages=" << fi_o_orderdate.npages
              << " compression=" << compression_method_name(fi_o_orderdate.compression_method) << std::endl;
    std::cout << "  L_ORDERKEY: npages=" << fi_l_orderkey.npages
              << " compression=" << compression_method_name(fi_l_orderkey.compression_method) << std::endl;
    std::cout << "  L_SUPPKEY: npages=" << fi_l_suppkey.npages
              << " compression=" << compression_method_name(fi_l_suppkey.compression_method) << std::endl;
    std::cout << "  L_EXTENDEDPRICE: npages=" << fi_l_extprice.npages
              << " compression=" << compression_method_name(fi_l_extprice.compression_method) << std::endl;
    std::cout << "  L_DISCOUNT: npages=" << fi_l_discount.npages
              << " compression=" << compression_method_name(fi_l_discount.compression_method) << std::endl;

    // ── Check for compressed fields ──
    bool any_compressed = false;
    std::vector<FieldPageInfo> all_fields;
    for (auto &fi : supplier_fields) all_fields.push_back(fi);
    for (auto &fi : customer_fields) all_fields.push_back(fi);
    for (auto &fi : orders_fields) all_fields.push_back(fi);
    for (auto &fi : lineitem_fields) all_fields.push_back(fi);
    for (auto &fi : all_fields) {
        if (fi.compression_method != CompressionMethod::NONE) any_compressed = true;
    }

    // ── GDS setup ──
    const size_t nthreads = options.nthreads;
    std::cout << "[Q5] nthreads=" << nthreads << std::endl;
    mb_cufile_driver_open();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // (Unified TpchWorkerCtx allocated after staging_pages computation below)

    // ── Pre-allocate shared prefix_sum buffer (reused across all flatten calls) ──
    uint32_t max_npages_small = std::max((uint32_t)fi_s_suppkey.npages,
                                          std::max((uint32_t)fi_s_nationkey.npages,
                                                   std::max((uint32_t)fi_c_custkey.npages,
                                                            (uint32_t)fi_c_nationkey.npages)));
    uint64_t *d_ps_shared = nullptr;
    // Will be allocated later after tile info computation

    // Pre-uploaded prefix_sum GPU arrays (Rule 3: populated before total_start)
    std::unordered_map<const uint64_t*, uint64_t*> d_full_ps_gpu;

    // Pre-allocated host prefix-sum buffer (Rule 4: avoid heap alloc in timed section).
    // Initially sized for Phase 1/2 fields; resized after tile info computation.
    std::vector<uint64_t> h_tile_ps(max_npages_small);

    // Helper: flatten INT32 pages → pre-allocated flat uint64_t array
    auto flatten_int32_field = [&](const FieldPageInfo &fi, void *data_buf, uint64_t nrecs, uint64_t *d_flat) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            compute_batch_ps_kernel<<<((uint32_t)fi.npages + 255) / 256, 256, 0, stream>>>(
                it->second, d_ps_shared, 0, (uint32_t)fi.npages);
            s_kernel_launches++;
        } else {
#if 0
            uint64_t cum = 0;
            for (size_t p = 0; p < fi.npages; p++) {
                uint32_t nalloc = 0;
                CUDA_CHECK(cudaMemcpy(&nalloc,
                    static_cast<const char *>(data_buf) + p * page_size,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cum += nalloc;
                h_tile_ps[p] = cum;
            }
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps.data(),
                                  fi.npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        q13_flatten_int32_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, fi.npages, nrecs, d_flat, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // Helper: flatten INT64 pages → pre-allocated flat uint64_t array
    auto flatten_int64_field = [&](const FieldPageInfo &fi, void *data_buf, uint64_t nrecs, uint64_t *d_flat) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            compute_batch_ps_kernel<<<((uint32_t)fi.npages + 255) / 256, 256, 0, stream>>>(
                it->second, d_ps_shared, 0, (uint32_t)fi.npages);
            s_kernel_launches++;
        } else {
#if 0
            uint64_t cum = 0;
            for (size_t p = 0; p < fi.npages; p++) {
                uint32_t nalloc = 0;
                CUDA_CHECK(cudaMemcpy(&nalloc,
                    static_cast<const char *>(data_buf) + p * page_size,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cum += nalloc;
                h_tile_ps[p] = cum;
            }
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps.data(),
                                  fi.npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        q13_flatten_int64_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, fi.npages, nrecs, d_flat, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // ── Host prefix sums ──
    auto build_host_ps = [](const FieldPageInfo &fi) -> std::vector<uint64_t> {
        std::vector<uint64_t> ps(fi.npages + 1, 0);
        if (fi.prefix_sum_nrecs != nullptr) {
            memcpy(ps.data(), fi.prefix_sum_nrecs, (fi.npages + 1) * sizeof(uint64_t));
        } else {
            std::cerr << "[Q5] WARNING: prefix_sum_nrecs not available" << std::endl;
        }
        return ps;
    };

    auto ps_o_i32 = build_host_ps(fi_o_orderdate);
    auto ps_o_i64 = build_host_ps(fi_o_orderkey);
    auto ps_l_i32 = build_host_ps(fi_l_extprice);
    auto ps_l_i64 = build_host_ps(fi_l_orderkey);

    // ── Tile execution constants ──
    constexpr size_t Q5_TILE_PAGES_MAX = 1024;
    const size_t o_npages_i32 = fi_o_orderdate.npages;
    const size_t Q5_TILE_PAGES_O = o_npages_i32;  // all ORDERS in one tile
    const size_t o_npages_i64 = fi_o_orderkey.npages;
    const size_t l_npages_i32 = fi_l_extprice.npages;
    const size_t l_npages_i64 = fi_l_orderkey.npages;

    if (o_npages_i64 != fi_o_custkey.npages)
        std::cerr << "[Q5] WARNING: O_ORDERKEY.npages != O_CUSTKEY.npages" << std::endl;
    if (l_npages_i64 != fi_l_suppkey.npages)
        std::cerr << "[Q5] WARNING: L_ORDERKEY.npages != L_SUPPKEY.npages" << std::endl;
    if (l_npages_i32 != fi_l_discount.npages)
        std::cerr << "[Q5] WARNING: LINEITEM INT32 npages mismatch" << std::endl;

    size_t max_npages_i64 = std::max(o_npages_i64, l_npages_i64);

    // ── Tile info computation ──
    size_t o_num_tiles = (o_npages_i32 + Q5_TILE_PAGES_O - 1) / Q5_TILE_PAGES_O;

    uint64_t tile_nrows_max_o = 0;
    size_t o_i64_tile_npages_max = 0;
    uint64_t o_i64_nrows_max = 0;
    for (size_t p_lo = 0; p_lo < o_npages_i32; p_lo += Q5_TILE_PAGES_O) {
        size_t tile_np = std::min(Q5_TILE_PAGES_O, o_npages_i32 - p_lo);
        uint64_t first_row = ps_o_i32[p_lo];
        uint64_t last_row = ps_o_i32[p_lo + tile_np];
        tile_nrows_max_o = std::max(tile_nrows_max_o, last_row - first_row);
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_o_i64.begin(), ps_o_i64.end(), first_row);
        size_t i64_s = (it_s == ps_o_i64.begin()) ? 0 : (size_t)(it_s - ps_o_i64.begin()) - 1;
        auto it_e = std::upper_bound(ps_o_i64.begin(), ps_o_i64.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_o_i64.begin());
        o_i64_tile_npages_max = std::max(o_i64_tile_npages_max, i64_e - i64_s);
        o_i64_nrows_max = std::max(o_i64_nrows_max, ps_o_i64[i64_e] - ps_o_i64[i64_s]);
    }

    // ── Budget-based Q5_TILE_PAGES (40 GiB cap) ──
    // staging_pages = max(Q5_TILE_PAGES_O, Q5_TILE_PAGES, o_i64_max, l_i64_max)
    // Column buffers and worker staging scale with staging_pages, which grows with
    // Q5_TILE_PAGES when l_i64_tile_npages_max > o_staging.  The formula accounts
    // for two cases:
    //   Case 1: T*i64_r ≤ o_staging → staging_pages fixed at o_staging
    //   Case 2: T*i64_r > o_staging → staging_pages ≈ T*i64_r, all costs scale
    size_t Q5_TILE_PAGES;
    {
        constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        uint64_t fixed_bytes = 256;
        fixed_bytes += (uint64_t)(fi_s_suppkey.npages + fi_s_nationkey.npages) * page_size;
        fixed_bytes += (uint64_t)nrecs_supplier * 16;
        { uint32_t c = 1; while (c < nrecs_supplier) c <<= 1;
          fixed_bytes += (uint64_t)c * 12; }
        fixed_bytes += (uint64_t)(fi_c_custkey.npages + fi_c_nationkey.npages) * page_size;
        fixed_bytes += (uint64_t)nrecs_customer * 16;
        { uint32_t c = 1; while (c < nrecs_customer) c <<= 1;
          fixed_bytes += (uint64_t)c * 12; }
        { uint64_t e = std::max((uint64_t)1024, nrecs_orders / 15);
          uint32_t c = 1; while (c < e * 2) c <<= 1;
          fixed_bytes += (uint64_t)c * 12; }
        fixed_bytes += tile_nrows_max_o * 8 + (uint64_t)o_i64_nrows_max * 16 + 512;
        // Worker nvcomp overhead is fixed (doesn't scale with staging)
        uint64_t worker_nvcomp_total = (uint64_t)nthreads * 2 * 1024 * 1024;
        fixed_bytes += worker_nvcomp_total;
        size_t o_staging = std::max(Q5_TILE_PAGES_O, o_i64_tile_npages_max);
        double i64_r = (l_npages_i32 > 0) ? (double)l_npages_i64 / l_npages_i32 : 2.0;
        i64_r = std::max(i64_r, 1.0) * 1.05;
        // Case 1: staging_pages = o_staging (fixed), worker ppw from o_staging
        size_t est_ppw = std::max((size_t)1, (o_staging + nthreads - 1) / nthreads);
        uint64_t worker_staging_fixed = (uint64_t)nthreads * est_ppw * 2 * page_size;
        uint64_t col_base = (uint64_t)4 * o_staging * page_size;
        uint64_t case1_fixed = fixed_bytes + worker_staging_fixed + col_base;
        size_t flat_per_tp = (size_t)((4.0 + 2.0 * i64_r) * page_size);
        size_t T;
        if (case1_fixed < GPU_MEM_BUDGET) {
            T = (GPU_MEM_BUDGET - case1_fixed) / flat_per_tp;
            if ((size_t)(T * std::max(1.0, i64_r)) > o_staging) {
                // Case 2: col_bufs + worker staging scale with T*i64_r
                // per_tp = flat(4+2r) + col_growth(4r) + worker_growth(2r) = (4+8r)*PS
                size_t full_per_tp = (size_t)((8.0 * i64_r + 4.0) * page_size);
                T = (fixed_bytes < GPU_MEM_BUDGET)
                    ? (GPU_MEM_BUDGET - fixed_bytes) / full_per_tp : 1;
            }
        } else {
            T = 1;
        }
        Q5_TILE_PAGES = std::min(Q5_TILE_PAGES_MAX, std::max((size_t)1, T));
    }
    Q5_TILE_PAGES = std::min(Q5_TILE_PAGES, l_npages_i32);  // never exceed actual column
    size_t l_num_tiles = (l_npages_i32 + Q5_TILE_PAGES - 1) / Q5_TILE_PAGES;

    uint64_t tile_nrows_max_l = 0;
    size_t l_i64_tile_npages_max = 0;
    uint64_t l_i64_nrows_max = 0;
    for (size_t p_lo = 0; p_lo < l_npages_i32; p_lo += Q5_TILE_PAGES) {
        size_t tile_np = std::min(Q5_TILE_PAGES, l_npages_i32 - p_lo);
        uint64_t first_row = ps_l_i32[p_lo];
        uint64_t last_row = ps_l_i32[p_lo + tile_np];
        tile_nrows_max_l = std::max(tile_nrows_max_l, last_row - first_row);
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_l_i64.begin(), ps_l_i64.end(), first_row);
        size_t i64_s = (it_s == ps_l_i64.begin()) ? 0 : (size_t)(it_s - ps_l_i64.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64.begin(), ps_l_i64.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_l_i64.begin());
        l_i64_tile_npages_max = std::max(l_i64_tile_npages_max, i64_e - i64_s);
        l_i64_nrows_max = std::max(l_i64_nrows_max, ps_l_i64[i64_e] - ps_l_i64[i64_s]);
    }

    size_t staging_pages = std::max({Q5_TILE_PAGES_O, Q5_TILE_PAGES,
                                      o_i64_tile_npages_max,
                                      l_i64_tile_npages_max});
    if (tile_nrows_max_o == 0) tile_nrows_max_o = 1;
    if (tile_nrows_max_l == 0) tile_nrows_max_l = 1;
    if (o_i64_nrows_max == 0) o_i64_nrows_max = 1;
    if (l_i64_nrows_max == 0) l_i64_nrows_max = 1;

    // d_ps_shared: sized for tile use (max of small-table pages and staging_pages)
    uint32_t ps_shared_size = std::max((uint32_t)staging_pages, max_npages_small);
    CUDA_CHECK(cudaMalloc(&d_ps_shared, ps_shared_size * sizeof(uint64_t)));

    // Resize h_tile_ps for tile-aware flatten (covers Phase 3/4 tile pages)
    h_tile_ps.resize(ps_shared_size);

    // ── Tile-aware flatten helpers ──
    auto flatten_int32_field_tile = [&](const FieldPageInfo &fi, void *data_buf,
                                         uint64_t tile_nrows, uint64_t *d_flat,
                                         size_t tile_start, size_t tile_npages) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            compute_batch_ps_kernel<<<((uint32_t)tile_npages + 255) / 256, 256, 0, stream>>>(
                it->second, d_ps_shared, (uint32_t)tile_start, (uint32_t)tile_npages);
            s_kernel_launches++;
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
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps.data(),
                                  tile_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        q13_flatten_int32_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, tile_npages, tile_nrows, d_flat, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    auto flatten_int64_field_tile = [&](const FieldPageInfo &fi, void *data_buf,
                                         uint64_t tile_nrows, uint64_t *d_flat,
                                         size_t tile_start, size_t tile_npages) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            compute_batch_ps_kernel<<<((uint32_t)tile_npages + 255) / 256, 256, 0, stream>>>(
                it->second, d_ps_shared, (uint32_t)tile_start, (uint32_t)tile_npages);
            s_kernel_launches++;
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
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps.data(),
                                  tile_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        q13_flatten_int64_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, tile_npages, tile_nrows, d_flat, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    auto flatten_int32_field_tile_masked = [&](const FieldPageInfo &fi, void *data_buf,
                                                 uint64_t tile_nrows, uint64_t *d_flat,
                                                 size_t tile_start, size_t tile_npages,
                                                 const uint8_t *d_mask, uint64_t fill_value) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            compute_batch_ps_kernel<<<((uint32_t)tile_npages + 255) / 256, 256, 0, stream>>>(
                it->second, d_ps_shared, (uint32_t)tile_start, (uint32_t)tile_npages);
            s_kernel_launches++;
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
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps.data(),
                                  tile_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        q13_flatten_int32_pages_ps_masked(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, tile_npages, tile_nrows,
            d_mask, fill_value, d_flat, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // (load_field_tile_mt / load_field_selective_tile_mt removed:
    //  all phases now use unified TpchWorkerCtx + tpch_worker_pipeline)

    // ════════════════════════════════════════════════════════
    // Pre-allocate ALL GPU and host memory
    // ════════════════════════════════════════════════════════

    // Phase 0: Host buffers for NATION (logging only; REGION read via GDS to GPU)
    char nation_names[25][26];
    void *h_n_nkey = nullptr, *h_n_name = nullptr, *h_n_rkey = nullptr;
    posix_memalign(&h_n_nkey, 512, fi_n_nationkey.npages * page_size);
    posix_memalign(&h_n_name, 512, fi_n_name.npages * page_size);
    posix_memalign(&h_n_rkey, 512, fi_n_regionkey.npages * page_size);
    int8_t *d_nationkey_to_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nationkey_to_idx, 25));
    int32_t *d_asia_regionkey = nullptr;
    CUDA_CHECK(cudaMalloc(&d_asia_regionkey, sizeof(int32_t)));

    // Phase 1: SUPPLIER data_bufs, flat arrays, HT
    void *data_buf_s_suppkey = mb_cuda_alloc(fi_s_suppkey.npages * page_size);
    void *data_buf_s_nationkey = mb_cuda_alloc(fi_s_nationkey.npages * page_size);
    uint64_t *d_s_suppkey_flat = nullptr, *d_s_nationkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_s_suppkey_flat, nrecs_supplier * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_s_nationkey_flat, nrecs_supplier * sizeof(uint64_t)));
    uint32_t ht_supp_cap = 1;
    while (ht_supp_cap < nrecs_supplier) ht_supp_cap <<= 1;
    uint32_t ht_supp_mask = ht_supp_cap - 1;
    uint64_t *d_ht_supp_keys = nullptr;
    int32_t  *d_ht_supp_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_supp_keys, ht_supp_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_supp_values, ht_supp_cap * sizeof(int32_t)));

    // Phase 2: CUSTOMER data_bufs, flat arrays, HT
    void *data_buf_c_custkey = mb_cuda_alloc(fi_c_custkey.npages * page_size);
    void *data_buf_c_nationkey = mb_cuda_alloc(fi_c_nationkey.npages * page_size);
    uint64_t *d_c_custkey_flat = nullptr, *d_c_nationkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c_custkey_flat, nrecs_customer * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_c_nationkey_flat, nrecs_customer * sizeof(uint64_t)));
    uint32_t ht_cust_cap = 1;
    while (ht_cust_cap < nrecs_customer) ht_cust_cap <<= 1;
    uint32_t ht_cust_mask = ht_cust_cap - 1;
    uint64_t *d_ht_cust_keys = nullptr;
    int32_t  *d_ht_cust_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_cust_keys, ht_cust_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_cust_values, ht_cust_cap * sizeof(int32_t)));

    // Phase 3: ORDERS HT + tile-sized flat arrays
    uint64_t est_ord_qual = std::max((uint64_t)1024, nrecs_orders / 15);
    uint32_t ht_ord_cap = 1;
    while (ht_ord_cap < est_ord_qual * 2) ht_ord_cap <<= 1;
    uint32_t ht_ord_mask = ht_ord_cap - 1;
    uint64_t *d_ht_ord_keys = nullptr;
    int32_t  *d_ht_ord_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_ord_keys, ht_ord_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_ord_values, ht_ord_cap * sizeof(int32_t)));

    // Per-column data buffers for cross-column pipeline
    // ORDERS needs 3 (1 INT32 + 2 INT64), LINEITEM needs 4 (2 INT32 + 2 INT64)
    static constexpr size_t Q5_NUM_COL_BUFS = 4;
    void *tile_col_bufs[Q5_NUM_COL_BUFS];
    for (size_t i = 0; i < Q5_NUM_COL_BUFS; i++)
        tile_col_bufs[i] = mb_cuda_alloc(staging_pages * page_size);

    uint64_t *d_o_orderdate_flat = nullptr;
    uint64_t *d_o_orderkey_flat = nullptr, *d_o_custkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_orderdate_flat, tile_nrows_max_o * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_o_orderkey_flat, o_i64_nrows_max * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_o_custkey_flat, o_i64_nrows_max * sizeof(uint64_t)));

    // Phase 4: LINEITEM tile-sized flat arrays + revenue
    uint64_t *d_l_extprice_flat = nullptr, *d_l_discount_flat = nullptr;
    uint64_t *d_l_orderkey_flat = nullptr, *d_l_suppkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l_extprice_flat, tile_nrows_max_l * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_l_discount_flat, tile_nrows_max_l * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_l_orderkey_flat, l_i64_nrows_max * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_l_suppkey_flat, l_i64_nrows_max * sizeof(uint64_t)));
    int64_t *d_revenue = nullptr;
    CUDA_CHECK(cudaMalloc(&d_revenue, 25 * sizeof(int64_t)));

    // Zone map tile mask
    uint8_t *d_tile_page_mask = nullptr;
    if (options.enable_zonemap) {
        CUDA_CHECK(cudaMalloc(&d_tile_page_mask, staging_pages * sizeof(uint8_t)));
    }

    // Unified TpchWorkerCtx: compute max_ppw to fill up to 40 GiB budget
    size_t gpu_free_before_workers = 0;
    cudaMemGetInfo(&gpu_free_before_workers, &gpu_total_dummy);
    size_t non_staging_bytes = gpu_free_start - gpu_free_before_workers;
    static constexpr size_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
    size_t min_ppw = (staging_pages + nthreads - 1) / nthreads;
    size_t max_ppw = min_ppw;
    if (GPU_MEM_BUDGET > non_staging_bytes) {
        // Per worker: 2 staging_bufs × max_ppw × page_size + nvCOMP (~2 MB)
        size_t nvcomp_per_worker = 2 * 1024 * 1024;
        size_t budget_for_staging = GPU_MEM_BUDGET - non_staging_bytes;
        size_t per_worker_budget = budget_for_staging / nthreads;
        if (per_worker_budget > nvcomp_per_worker) {
            max_ppw = (per_worker_budget - nvcomp_per_worker) / (2 * page_size);
        }
        max_ppw = std::max(max_ppw, min_ppw);
        max_ppw = std::min(max_ppw, staging_pages);  // cap: no benefit beyond max column pages
    }
    std::vector<TpchWorkerCtx> workers(nthreads);
    tpch_worker_alloc(workers.data(), nthreads, fds.data(), num_devices,
                       page_size, max_ppw, any_compressed, all_fields);

    std::cout << "[Q5] ORDERS tile execution: " << o_num_tiles << " tiles of "
              << Q5_TILE_PAGES_O << " pages, tile_nrows_max=" << tile_nrows_max_o << std::endl;
    std::cout << "[Q5] LINEITEM tile execution: " << l_num_tiles << " tiles of "
              << Q5_TILE_PAGES << " pages, tile_nrows_max=" << tile_nrows_max_l << std::endl;

    // Pre-upload prefix_sum arrays to GPU (Rule 3: before total_start)
    {
        auto upload_ps = [&](const FieldPageInfo &fi) {
            if (!fi.prefix_sum_nrecs || d_full_ps_gpu.count(fi.prefix_sum_nrecs)) return;
            uint64_t *d_ps = nullptr;
            CUDA_CHECK(cudaMalloc(&d_ps, (fi.npages + 1) * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_ps, fi.prefix_sum_nrecs,
                                  (fi.npages + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            d_full_ps_gpu[fi.prefix_sum_nrecs] = d_ps;
        };
        upload_ps(fi_s_suppkey); upload_ps(fi_s_nationkey);
        upload_ps(fi_c_custkey); upload_ps(fi_c_nationkey);
        upload_ps(fi_o_orderdate); upload_ps(fi_o_orderkey); upload_ps(fi_o_custkey);
        upload_ps(fi_l_extprice); upload_ps(fi_l_discount);
        upload_ps(fi_l_orderkey); upload_ps(fi_l_suppkey);
    }

    // ── Zone map: pre-allocate outside timing (Rule 4) ──
    const int32_t date_low  = options.q6_sd_low  ? options.q6_sd_low  : 19940101;
    const int32_t date_high = options.q6_sd_high ? options.q6_sd_high : 19950101;

    std::vector<bool> ord_page_active(o_npages_i32, true);
    std::vector<bool> li_page_active(l_npages_i32, true);
    uint8_t *d_mask_ord_i64 = nullptr;
    uint8_t *d_mask_li_i64 = nullptr;
    std::vector<uint8_t> h_mask_ord_i64(o_npages_i64, 1);
    std::vector<uint8_t> h_mask_li_i64(l_npages_i64, 1);

    // Zonemap stats metadata (resolved from metadata struct, outside timing per Rule 3)
    struct ZmStatsInfo {
        uint64_t nstats, stats_start, stats_npg;
    };
    ZmStatsInfo zm_ord_odate{}, zm_ord_rname{}, zm_li_odate{}, zm_li_rname{};
    void *d_zm_stats_buf[2] = {nullptr, nullptr};
    uint64_t zm_stats_buf_bytes = 0;
    GdsZonemapCtx zm_ctx{};
    CUfileHandle_t zm_cufile_handles[MAX_GDS_DEVICES];
    int zm_dup_fds[MAX_GDS_DEVICES];
    bool zm_handles_open = false;

    if (options.enable_zonemap) {
        CUDA_CHECK(cudaMalloc(&d_mask_ord_i64, o_npages_i64));
        CUDA_CHECK(cudaMalloc(&d_mask_li_i64, l_npages_i64));

        const size_t o_odate_field = TPCH::common::O_ORDERDATE;
        zm_ord_odate = { metadata.table_orders_nstats[o_odate_field],
                         metadata.table_orders_stats_start_page_ids[o_odate_field],
                         metadata.table_orders_stats_npages[o_odate_field] };

        const size_t sw_rname_idx = TPCH::common::OS_SIDEWAYS_R_NAME;
        zm_ord_rname = { metadata.table_orders_sideways_nstats[o_odate_field][sw_rname_idx],
                         metadata.table_orders_sideways_stats_start_page_ids[o_odate_field][sw_rname_idx],
                         metadata.table_orders_sideways_stats_npages[o_odate_field][sw_rname_idx] };

        const size_t li_ref_field = TPCH::common::L_EXTENDEDPRICE;
        const size_t sw_idx = TPCH::common::LS_SIDEWAYS_O_ORDERDATE;
        zm_li_odate = { metadata.table_lineitem_sideways_nstats[li_ref_field][sw_idx],
                        metadata.table_lineitem_sideways_stats_start_page_ids[li_ref_field][sw_idx],
                        metadata.table_lineitem_sideways_stats_npages[li_ref_field][sw_idx] };

        const size_t li_sw_rname_idx = TPCH::common::LS_SIDEWAYS_R_NAME;
        zm_li_rname = { metadata.table_lineitem_sideways_nstats[li_ref_field][li_sw_rname_idx],
                        metadata.table_lineitem_sideways_stats_start_page_ids[li_ref_field][li_sw_rname_idx],
                        metadata.table_lineitem_sideways_stats_npages[li_ref_field][li_sw_rname_idx] };

        // Allocate 2 GPU buffers for simultaneous stats (max 2 preds per eval)
        uint64_t max_stats_npg = std::max({zm_ord_odate.stats_npg, zm_ord_rname.stats_npg,
                                            zm_li_odate.stats_npg, zm_li_rname.stats_npg});
        zm_stats_buf_bytes = max_stats_npg * page_size;
        if (zm_stats_buf_bytes > 0) {
            for (int i = 0; i < 2; i++) {
                CUDA_CHECK(cudaMalloc(&d_zm_stats_buf[i], zm_stats_buf_bytes));
                GDS_CHECK(cuFileBufRegister(d_zm_stats_buf[i], zm_stats_buf_bytes, 0));
            }
        }

        uint64_t max_npages_zm = std::max((uint64_t)o_npages_i32, (uint64_t)l_npages_i32);
        zm_ctx = gds_zonemap_ctx_create(max_npages_zm);

        for (size_t d = 0; d < num_devices; d++) {
            zm_dup_fds[d] = dup(fds[d]);
            zm_cufile_handles[d] = mb_cufile_handle_register(zm_dup_fds[d]);
        }
        zm_handles_open = true;
    }

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t golap_gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // Pre-allocate tile loop buffers (Rule 4: avoid heap allocation inside timed section)
    std::vector<uint32_t> tile_active_i32;
    tile_active_i32.reserve(std::max(Q5_TILE_PAGES_O, Q5_TILE_PAGES));
    std::vector<uint8_t> h_mask(std::max(Q5_TILE_PAGES_O, Q5_TILE_PAGES));
    std::vector<uint32_t> tile_i64_active;
    tile_i64_active.reserve(std::max(o_npages_i64, l_npages_i64));

    // ════════════════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    auto phase_start = total_start;
    // ════════════════════════════════════════════════════════

    // ── Zone map IO + eval (GDS + GPU, inside timing per Rule 6) ──
    if (options.enable_zonemap && d_zm_stats_buf[0]) {
        const int32_t asia_dict_id = 2;

        // ORDERS: O_ORDERDATE + sideways R_NAME
        {
            int32_t *d_odate_stats = nullptr;
            if (zm_ord_odate.nstats > 0 && zm_ord_odate.stats_start > 0) {
                gds_read_zonemap(zm_cufile_handles, num_devices,
                                 zm_ord_odate.stats_start, zm_ord_odate.stats_npg,
                                 page_size, d_zm_stats_buf[0]);
                d_odate_stats = reinterpret_cast<int32_t*>(d_zm_stats_buf[0]);
            }

            int32_t *d_rname_stats = nullptr;
            if (zm_ord_rname.nstats > 0 && zm_ord_rname.stats_start > 0) {
                gds_read_zonemap(zm_cufile_handles, num_devices,
                                 zm_ord_rname.stats_start, zm_ord_rname.stats_npg,
                                 page_size, d_zm_stats_buf[1]);
                d_rname_stats = reinterpret_cast<int32_t*>(d_zm_stats_buf[1]);
            }

            uint32_t npreds = 0;
            if (d_odate_stats) {
                zm_ctx.h_preds[npreds].d_stats  = d_odate_stats;
                zm_ctx.h_preds[npreds].nstats   = zm_ord_odate.nstats;
                zm_ctx.h_preds[npreds].pred_lo  = date_low;
                zm_ctx.h_preds[npreds].pred_hi  = date_high - 1;
                npreds++;
            }
            if (d_rname_stats) {
                zm_ctx.h_preds[npreds].d_stats  = d_rname_stats;
                zm_ctx.h_preds[npreds].nstats   = zm_ord_rname.nstats;
                zm_ctx.h_preds[npreds].pred_lo  = asia_dict_id;
                zm_ctx.h_preds[npreds].pred_hi  = asia_dict_id;
                npreds++;
            }

            zm_ctx.d_ps_i32   = d_full_ps_gpu[fi_o_orderdate.prefix_sum_nrecs] + 1;
            zm_ctx.d_ps_i64   = d_full_ps_gpu[fi_o_orderkey.prefix_sum_nrecs] + 1;
            zm_ctx.d_mask_i64 = d_mask_ord_i64;
            zm_ctx.npages_i64 = (uint32_t)o_npages_i64;

            if (npreds > 0) {
                gds_zonemap_eval_async(zm_ctx, o_npages_i32, npreds, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                for (uint32_t pg = 0; pg < o_npages_i32; pg++)
                    ord_page_active[pg] = zm_ctx.h_mask[pg];
                CUDA_CHECK(cudaMemcpy(h_mask_ord_i64.data(), d_mask_ord_i64,
                                      o_npages_i64, cudaMemcpyDeviceToHost));
            }

            uint32_t active_count = 0;
            for (uint32_t pg = 0; pg < o_npages_i32; pg++)
                if (ord_page_active[pg]) active_count++;
            std::cout << "[ZONEMAP] ORDERS GPU pruning: active=" << active_count
                      << "/" << o_npages_i32 << std::endl;
        }

        // LINEITEM: sideways O_ORDERDATE + sideways R_NAME
        {
            int32_t *d_li_odate_stats = nullptr;
            if (zm_li_odate.nstats > 0 && zm_li_odate.stats_start > 0) {
                gds_read_zonemap(zm_cufile_handles, num_devices,
                                 zm_li_odate.stats_start, zm_li_odate.stats_npg,
                                 page_size, d_zm_stats_buf[0]);
                d_li_odate_stats = reinterpret_cast<int32_t*>(d_zm_stats_buf[0]);
            }

            int32_t *d_li_rname_stats = nullptr;
            if (zm_li_rname.nstats > 0 && zm_li_rname.stats_start > 0) {
                gds_read_zonemap(zm_cufile_handles, num_devices,
                                 zm_li_rname.stats_start, zm_li_rname.stats_npg,
                                 page_size, d_zm_stats_buf[1]);
                d_li_rname_stats = reinterpret_cast<int32_t*>(d_zm_stats_buf[1]);
            }

            uint32_t npreds = 0;
            if (d_li_odate_stats) {
                zm_ctx.h_preds[npreds].d_stats  = d_li_odate_stats;
                zm_ctx.h_preds[npreds].nstats   = zm_li_odate.nstats;
                zm_ctx.h_preds[npreds].pred_lo  = date_low;
                zm_ctx.h_preds[npreds].pred_hi  = date_high - 1;
                npreds++;
            }
            if (d_li_rname_stats) {
                zm_ctx.h_preds[npreds].d_stats  = d_li_rname_stats;
                zm_ctx.h_preds[npreds].nstats   = zm_li_rname.nstats;
                zm_ctx.h_preds[npreds].pred_lo  = asia_dict_id;
                zm_ctx.h_preds[npreds].pred_hi  = asia_dict_id;
                npreds++;
            }

            zm_ctx.d_ps_i32   = d_full_ps_gpu[fi_l_extprice.prefix_sum_nrecs] + 1;
            zm_ctx.d_ps_i64   = d_full_ps_gpu[fi_l_orderkey.prefix_sum_nrecs] + 1;
            zm_ctx.d_mask_i64 = d_mask_li_i64;
            zm_ctx.npages_i64 = (uint32_t)l_npages_i64;

            if (npreds > 0) {
                gds_zonemap_eval_async(zm_ctx, l_npages_i32, npreds, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                for (uint32_t pg = 0; pg < l_npages_i32; pg++)
                    li_page_active[pg] = zm_ctx.h_mask[pg];
                CUDA_CHECK(cudaMemcpy(h_mask_li_i64.data(), d_mask_li_i64,
                                      l_npages_i64, cudaMemcpyDeviceToHost));
            }

            uint32_t li_active_count = 0;
            for (uint32_t pg = 0; pg < l_npages_i32; pg++)
                if (li_page_active[pg]) li_active_count++;
            std::cout << "[ZONEMAP] LINEITEM GPU pruning: active=" << li_active_count
                      << "/" << l_npages_i32 << std::endl;
        }

        zm_ctx.d_ps_i32 = nullptr;
        zm_ctx.d_ps_i64 = nullptr;
        zm_ctx.d_mask_i64 = nullptr;
    }

    // Phase 0: REGION + NATION → d_nationkey_to_idx (GPU kernel via GDS)
    memset(nation_names, 0, sizeof(nation_names));
    int num_asia_nations = 0;
    int32_t asia_regionkey = -1;
    {
        // Read N_NAME to host for logging
        read_pages(fds, h_n_name, fi_n_name.start_page_id, fi_n_name.npages, page_size);
        // Read N_NKEY, N_RKEY to host for logging
        std::thread p0_thr[2];
        p0_thr[0] = std::thread([&]() { read_pages(fds, h_n_nkey, fi_n_nationkey.start_page_id, fi_n_nationkey.npages, page_size); });
        p0_thr[1] = std::thread([&]() { read_pages(fds, h_n_rkey, fi_n_regionkey.start_page_id, fi_n_regionkey.npages, page_size); });
        for (int i = 0; i < 2; i++) p0_thr[i].join();

        // GDS: read R_RKEY, R_NAME, N_NKEY, N_RKEY directly to GPU
        CUDA_CHECK(cudaMemsetAsync(d_nationkey_to_idx, 0xFF, 25, stream));
        char* d_staging = static_cast<char*>(workers[0].staging_buf[0]);
        size_t off = 0;

        auto gds_read_field = [&](const FieldPageInfo& fi, size_t base_off) {
            for (uint64_t p = 0; p < fi.npages; p++) {
                uint64_t pagid = fi.start_page_id + p;
                uint64_t dev = pagid % num_devices;
                off_t foff = (off_t)(pagid / num_devices) * page_size;
                ssize_t nread = cuFileRead(workers[0].cufile_handles[dev],
                    workers[0].staging_buf[0], page_size, foff,
                    base_off + p * page_size);
                if (nread < 0 || (size_t)nread != page_size) {
                    std::cerr << "cuFileRead Phase0 failed: pagid=" << pagid
                              << " nread=" << nread << std::endl;
                }
            }
        };

        char* d_r_rkey_stg = d_staging + off; gds_read_field(fi_r_regionkey, off); off += fi_r_regionkey.npages * page_size;
        char* d_r_name_stg = d_staging + off; gds_read_field(fi_r_name,      off); off += fi_r_name.npages * page_size;
        char* d_n_nkey_stg = d_staging + off; gds_read_field(fi_n_nationkey,  off); off += fi_n_nationkey.npages * page_size;
        char* d_n_rkey_stg = d_staging + off; gds_read_field(fi_n_regionkey,  off);

        q5_phase0_region_nation_launch(
            d_r_rkey_stg, d_r_name_stg, d_n_nkey_stg, d_n_rkey_stg,
            d_nationkey_to_idx, d_asia_regionkey, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemcpy(&asia_regionkey, d_asia_regionkey,
            sizeof(int32_t), cudaMemcpyDeviceToHost));
        std::cout << "[Q5] asia_regionkey=" << asia_regionkey << std::endl;

        // Host-side: build nation_names for logging
        uint32_t nalloc_nation = host_pag_get_nalloc((const char *)h_n_nkey);
        for (uint32_t i = 0; i < nalloc_nation; i++) {
            int32_t n_rkey = *reinterpret_cast<const int32_t *>(
                (const char *)h_n_rkey + 12 + sizeof(int32_t) * i);
            if (n_rkey != asia_regionkey) continue;
            int32_t n_nkey = *reinterpret_cast<const int32_t *>(
                (const char *)h_n_nkey + 12 + sizeof(int32_t) * i);
            const char *n_name = host_pagcol_char_data((const char *)h_n_name, i, 28);
            if (n_nkey >= 0 && n_nkey < 25) {
                int len = 25;
                while (len > 0 && n_name[len - 1] == ' ') len--;
                memcpy(nation_names[num_asia_nations], n_name, len);
                nation_names[num_asia_nations][len] = '\0';
                num_asia_nations++;
            }
        }
        std::cout << "[Q5] ASIA nations (" << num_asia_nations << "):";
        for (int i = 0; i < num_asia_nations; i++)
            std::cout << " " << nation_names[i];
        std::cout << std::endl;
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 0 (REGION+NATION): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // Phase 1: SUPPLIER hash table build
    std::cout << "[Q5] Loading S_SUPPKEY (" << fi_s_suppkey.npages << " pages)..." << std::endl;
    std::cout << "[Q5] Loading S_NATIONKEY (" << fi_s_nationkey.npages << " pages)..." << std::endl;
    {
        TpchPipelineCol supp_cols[2] = {
            {&fi_s_suppkey, data_buf_s_suppkey, nullptr, fi_s_suppkey.npages, 0},
            {&fi_s_nationkey, data_buf_s_nationkey, nullptr, fi_s_nationkey.npages, 0},
        };
        std::thread thr_buf[64];
        size_t nthr = 0;
        for (size_t t = 0; t < nthreads; t++) {
            thr_buf[nthr++] = std::thread([&, t]() {
                tpch_worker_pipeline(workers[t], supp_cols, 2,
                                     page_size, num_devices, t, nthreads, cuda_ctx_handle);
            });
        }
        for (size_t i = 0; i < nthr; i++) thr_buf[i].join();
    }

    flatten_int64_field(fi_s_suppkey, data_buf_s_suppkey, nrecs_supplier, d_s_suppkey_flat);
    flatten_int32_field(fi_s_nationkey, data_buf_s_nationkey, nrecs_supplier, d_s_nationkey_flat);

    CUDA_CHECK(cudaMemsetAsync(d_ht_supp_keys, 0xFF, ht_supp_cap * sizeof(uint64_t), stream));
    q5_build_supplier_ht(d_s_suppkey_flat, d_s_nationkey_flat, nrecs_supplier,
        d_nationkey_to_idx, d_ht_supp_keys, d_ht_supp_values, ht_supp_mask, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "[Q5] SUPPLIER HT built (capacity=" << ht_supp_cap << ")" << std::endl;

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 1 (SUPPLIER): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // Phase 2: CUSTOMER hash table build
    std::cout << "[Q5] Loading C_CUSTKEY (" << fi_c_custkey.npages << " pages)..." << std::endl;
    std::cout << "[Q5] Loading C_NATIONKEY (" << fi_c_nationkey.npages << " pages)..." << std::endl;
    {
        TpchPipelineCol cust_cols[2] = {
            {&fi_c_custkey, data_buf_c_custkey, nullptr, fi_c_custkey.npages, 0},
            {&fi_c_nationkey, data_buf_c_nationkey, nullptr, fi_c_nationkey.npages, 0},
        };
        std::thread thr_buf[64];
        size_t nthr = 0;
        for (size_t t = 0; t < nthreads; t++) {
            thr_buf[nthr++] = std::thread([&, t]() {
                tpch_worker_pipeline(workers[t], cust_cols, 2,
                                     page_size, num_devices, t, nthreads, cuda_ctx_handle);
            });
        }
        for (size_t i = 0; i < nthr; i++) thr_buf[i].join();
    }

    flatten_int64_field(fi_c_custkey, data_buf_c_custkey, nrecs_customer, d_c_custkey_flat);
    flatten_int32_field(fi_c_nationkey, data_buf_c_nationkey, nrecs_customer, d_c_nationkey_flat);

    CUDA_CHECK(cudaMemsetAsync(d_ht_cust_keys, 0xFF, ht_cust_cap * sizeof(uint64_t), stream));
    q5_build_customer_ht(d_c_custkey_flat, d_c_nationkey_flat, nrecs_customer,
        d_nationkey_to_idx, d_ht_cust_keys, d_ht_cust_values, ht_cust_mask, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "[Q5] CUSTOMER HT built (capacity=" << ht_cust_cap << ")" << std::endl;

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 2 (CUSTOMER): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // Phase 3: ORDERS hash table build (date_low/date_high, ord_page_active pre-computed above)
    CUDA_CHECK(cudaMemsetAsync(d_ht_ord_keys, 0xFF, ht_ord_cap * sizeof(uint64_t), stream));

    // ── ORDERS tile loop (cross-column pipelined) ──
    for (size_t tile_idx = 0; tile_idx < o_num_tiles; tile_idx++) {
        size_t p_lo = tile_idx * Q5_TILE_PAGES_O;
        size_t tile_np = std::min(Q5_TILE_PAGES_O, o_npages_i32 - p_lo);
        uint64_t tile_nrows = ps_o_i32[p_lo + tile_np] - ps_o_i32[p_lo];
        if (tile_nrows == 0) continue;

        // Collect active pages in this tile
        tile_active_i32.clear();
        for (size_t j = 0; j < tile_np; j++)
            if (ord_page_active[p_lo + j]) tile_active_i32.push_back((uint32_t)(p_lo + j));
        if (options.enable_zonemap && tile_active_i32.empty()) continue;

        bool use_selective = options.enable_zonemap && tile_active_i32.size() < tile_np;

        // Upload tile mask
        if (use_selective) {
            for (size_t j = 0; j < tile_np; j++)
                h_mask[j] = ord_page_active[p_lo + j] ? 1 : 0;
            CUDA_CHECK(cudaMemcpy(d_tile_page_mask, h_mask.data(), tile_np, cudaMemcpyHostToDevice));
        }

        // ── INT64 page range for this tile ──
        uint64_t first_row = ps_o_i32[p_lo];
        uint64_t last_row = ps_o_i32[p_lo + tile_np];
        auto it_s = std::upper_bound(ps_o_i64.begin(), ps_o_i64.end(), first_row);
        size_t i64_start = (it_s == ps_o_i64.begin()) ? 0 : (size_t)(it_s - ps_o_i64.begin()) - 1;
        auto it_e = std::upper_bound(ps_o_i64.begin(), ps_o_i64.end(), last_row - 1);
        size_t i64_end = (size_t)(it_e - ps_o_i64.begin());
        size_t i64_np = i64_end - i64_start;
        uint64_t i64_nrows = ps_o_i64[i64_end] - ps_o_i64[i64_start];
        uint64_t i64_offset = first_row - ps_o_i64[i64_start];

        // Compute needed INT64 pages for this tile (from GPU-derived mask)
        tile_i64_active.clear();
        if (options.enable_zonemap) {
            for (size_t pg = i64_start; pg < i64_end; pg++)
                if (h_mask_ord_i64[pg]) tile_i64_active.push_back((uint32_t)pg);
        }

        // Zero per-column bufs (inactive pages → nalloc=0)
        CUDA_CHECK(cudaMemset(tile_col_bufs[0], 0, tile_np * page_size));
        CUDA_CHECK(cudaMemset(tile_col_bufs[1], 0, i64_np * page_size));
        CUDA_CHECK(cudaMemset(tile_col_bufs[2], 0, i64_np * page_size));

        // Build pipeline column descriptors (3 columns: O_ORDERDATE, O_ORDERKEY, O_CUSTKEY)
        size_t i32_n = use_selective ? tile_active_i32.size() : tile_np;
        size_t i64_n = options.enable_zonemap ? tile_i64_active.size() : i64_np;
        TpchPipelineCol ord_cols[3] = {
            {&fi_o_orderdate, tile_col_bufs[0],
             use_selective ? tile_active_i32.data() : nullptr, i32_n, p_lo},
            {&fi_o_orderkey, tile_col_bufs[1],
             options.enable_zonemap ? tile_i64_active.data() : nullptr, i64_n, i64_start},
            {&fi_o_custkey, tile_col_bufs[2],
             options.enable_zonemap ? tile_i64_active.data() : nullptr, i64_n, i64_start},
        };

        // ── Cross-column pipeline: load all 3 columns with IO-decomp overlap ──
        {
            std::thread thr_buf[64];
            size_t nthr = 0;
            for (size_t t = 0; t < nthreads; t++) {
                thr_buf[nthr++] = std::thread([&, t]() {
                    tpch_worker_pipeline(workers[t], ord_cols, 3,
                                         page_size, num_devices,
                                         t, nthreads, cuda_ctx_handle);
                });
            }
            for (size_t i = 0; i < nthr; i++) thr_buf[i].join();
        }

        // ── Flatten all 3 columns ──
        if (use_selective) {
            flatten_int32_field_tile_masked(fi_o_orderdate, tile_col_bufs[0], tile_nrows,
                d_o_orderdate_flat, p_lo, tile_np, d_tile_page_mask, 0);
        } else {
            flatten_int32_field_tile(fi_o_orderdate, tile_col_bufs[0], tile_nrows,
                d_o_orderdate_flat, p_lo, tile_np);
        }
        flatten_int64_field_tile(fi_o_orderkey, tile_col_bufs[1], i64_nrows,
            d_o_orderkey_flat, i64_start, i64_np);
        flatten_int64_field_tile(fi_o_custkey, tile_col_bufs[2], i64_nrows,
            d_o_custkey_flat, i64_start, i64_np);

        // ── Kernel: build ORDERS HT ──
        q5_build_orders_ht(
            d_o_orderkey_flat + i64_offset, d_o_custkey_flat + i64_offset,
            d_o_orderdate_flat,
            tile_nrows, date_low, date_high,
            d_ht_cust_keys, d_ht_cust_values, ht_cust_mask,
            d_ht_ord_keys, d_ht_ord_values, ht_ord_mask, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::cout << "[Q5] ORDERS HT built (capacity=" << ht_ord_cap << ")" << std::endl;

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 3 (ORDERS): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // Phase 4: LINEITEM streaming probe + revenue aggregation (li_page_active pre-computed above)
    CUDA_CHECK(cudaMemsetAsync(d_revenue, 0, 25 * sizeof(int64_t), stream));

    // ── LINEITEM tile loop ──
    for (size_t tile_idx = 0; tile_idx < l_num_tiles; tile_idx++) {
        size_t p_lo = tile_idx * Q5_TILE_PAGES;
        size_t tile_np = std::min(Q5_TILE_PAGES, l_npages_i32 - p_lo);
        uint64_t tile_nrows = ps_l_i32[p_lo + tile_np] - ps_l_i32[p_lo];
        if (tile_nrows == 0) continue;

        // Collect active pages in this tile
        tile_active_i32.clear();
        for (size_t j = 0; j < tile_np; j++)
            if (li_page_active[p_lo + j]) tile_active_i32.push_back((uint32_t)(p_lo + j));
        if (options.enable_zonemap && tile_active_i32.empty()) continue;

        bool use_selective = options.enable_zonemap && tile_active_i32.size() < tile_np;

        // Upload tile mask
        if (use_selective) {
            for (size_t j = 0; j < tile_np; j++)
                h_mask[j] = li_page_active[p_lo + j] ? 1 : 0;
            CUDA_CHECK(cudaMemcpy(d_tile_page_mask, h_mask.data(), tile_np, cudaMemcpyHostToDevice));
        }

        // ── INT64 page range for this tile ──
        uint64_t first_row = ps_l_i32[p_lo];
        uint64_t last_row = ps_l_i32[p_lo + tile_np];
        auto it_s = std::upper_bound(ps_l_i64.begin(), ps_l_i64.end(), first_row);
        size_t i64_start = (it_s == ps_l_i64.begin()) ? 0 : (size_t)(it_s - ps_l_i64.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64.begin(), ps_l_i64.end(), last_row - 1);
        size_t i64_end = (size_t)(it_e - ps_l_i64.begin());
        size_t i64_np = i64_end - i64_start;
        uint64_t i64_nrows = ps_l_i64[i64_end] - ps_l_i64[i64_start];
        uint64_t i64_offset = first_row - ps_l_i64[i64_start];

        // Compute needed INT64 pages for this tile (from GPU-derived mask)
        tile_i64_active.clear();
        if (options.enable_zonemap) {
            for (size_t pg = i64_start; pg < i64_end; pg++)
                if (h_mask_li_i64[pg]) tile_i64_active.push_back((uint32_t)pg);
        }

        // Zero per-column bufs (inactive pages → nalloc=0)
        CUDA_CHECK(cudaMemset(tile_col_bufs[0], 0, tile_np * page_size));
        CUDA_CHECK(cudaMemset(tile_col_bufs[1], 0, tile_np * page_size));
        CUDA_CHECK(cudaMemset(tile_col_bufs[2], 0, i64_np * page_size));
        CUDA_CHECK(cudaMemset(tile_col_bufs[3], 0, i64_np * page_size));

        // Build pipeline column descriptors (4 columns: L_EXTPRICE, L_DISCOUNT, L_ORDERKEY, L_SUPPKEY)
        size_t i32_n = use_selective ? tile_active_i32.size() : tile_np;
        size_t i64_n = options.enable_zonemap ? tile_i64_active.size() : i64_np;
        TpchPipelineCol li_cols[4] = {
            {&fi_l_extprice, tile_col_bufs[0],
             use_selective ? tile_active_i32.data() : nullptr, i32_n, p_lo},
            {&fi_l_discount, tile_col_bufs[1],
             use_selective ? tile_active_i32.data() : nullptr, i32_n, p_lo},
            {&fi_l_orderkey, tile_col_bufs[2],
             options.enable_zonemap ? tile_i64_active.data() : nullptr, i64_n, i64_start},
            {&fi_l_suppkey, tile_col_bufs[3],
             options.enable_zonemap ? tile_i64_active.data() : nullptr, i64_n, i64_start},
        };

        // ── Cross-column pipeline: load all 4 columns with IO-decomp overlap ──
        {
            std::thread thr_buf[64];
            size_t nthr = 0;
            for (size_t t = 0; t < nthreads; t++) {
                thr_buf[nthr++] = std::thread([&, t]() {
                    tpch_worker_pipeline(workers[t], li_cols, 4,
                                         page_size, num_devices,
                                         t, nthreads, cuda_ctx_handle);
                });
            }
            for (size_t i = 0; i < nthr; i++) thr_buf[i].join();
        }

        // ── Flatten all 4 columns ──
        if (use_selective) {
            flatten_int32_field_tile_masked(fi_l_extprice, tile_col_bufs[0], tile_nrows,
                d_l_extprice_flat, p_lo, tile_np, d_tile_page_mask, 0);
            flatten_int32_field_tile_masked(fi_l_discount, tile_col_bufs[1], tile_nrows,
                d_l_discount_flat, p_lo, tile_np, d_tile_page_mask, 0);
        } else {
            flatten_int32_field_tile(fi_l_extprice, tile_col_bufs[0], tile_nrows,
                d_l_extprice_flat, p_lo, tile_np);
            flatten_int32_field_tile(fi_l_discount, tile_col_bufs[1], tile_nrows,
                d_l_discount_flat, p_lo, tile_np);
        }
        flatten_int64_field_tile(fi_l_orderkey, tile_col_bufs[2], i64_nrows,
            d_l_orderkey_flat, i64_start, i64_np);
        flatten_int64_field_tile(fi_l_suppkey, tile_col_bufs[3], i64_nrows,
            d_l_suppkey_flat, i64_start, i64_np);

        // ── Kernel: LINEITEM probe ──
        q5_lineitem_probe(
            d_l_orderkey_flat + i64_offset, d_l_suppkey_flat + i64_offset,
            d_l_extprice_flat, d_l_discount_flat,
            tile_nrows,
            d_ht_ord_keys, d_ht_ord_values, ht_ord_mask,
            d_ht_supp_keys, d_ht_supp_values, ht_supp_mask,
            d_revenue, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 4 (LINEITEM): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // Phase 5: Results
    int64_t h_revenue[25] = {};
    CUDA_CHECK(cudaMemcpy(h_revenue, d_revenue, 25 * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // ════════════════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    // ════════════════════════════════════════════════════════

    // Build result and sort
    struct Q5Row { int nation_idx; int64_t revenue; };
    std::vector<Q5Row> results;
    for (int i = 0; i < num_asia_nations; i++) {
        results.push_back({i, h_revenue[i]});
    }
    std::sort(results.begin(), results.end(),
        [](const Q5Row &a, const Q5Row &b) { return a.revenue > b.revenue; });

    std::cout << "\n=== TPC-H Q5 Result ===" << std::endl;
    std::cout << "n_name                    | revenue" << std::endl;
    std::cout << "--------------------------+-----------------" << std::endl;
    for (auto &r : results) {
        printf("%-25s | %20ld\n", nation_names[r.nation_idx], (long)r.revenue);
    }
    std::cout << std::endl;

    // ── Free all GPU and host memory ──
    if (zm_handles_open) {
        for (int i = 0; i < 2; i++) {
            if (d_zm_stats_buf[i]) { cuFileBufDeregister(d_zm_stats_buf[i]); cudaFree(d_zm_stats_buf[i]); }
        }
        gds_zonemap_ctx_destroy(zm_ctx);
        for (size_t d = 0; d < num_devices; d++) {
            cuFileHandleDeregister(zm_cufile_handles[d]);
            close(zm_dup_fds[d]);
        }
    }
    if (d_tile_page_mask) cudaFree(d_tile_page_mask);
    if (d_mask_ord_i64) cudaFree(d_mask_ord_i64);
    if (d_mask_li_i64) cudaFree(d_mask_li_i64);
    cudaFree(d_ps_shared);
    for (auto &[k, v] : d_full_ps_gpu) cudaFree(v);
    cudaFree(d_revenue);
    cudaFree(d_ht_ord_keys);
    cudaFree(d_ht_ord_values);
    cudaFree(d_ht_cust_keys);
    cudaFree(d_ht_cust_values);
    cudaFree(d_ht_supp_keys);
    cudaFree(d_ht_supp_values);
    cudaFree(d_nationkey_to_idx);
    cudaFree(d_s_suppkey_flat);
    cudaFree(d_s_nationkey_flat);
    cudaFree(d_c_custkey_flat);
    cudaFree(d_c_nationkey_flat);
    cudaFree(d_o_orderkey_flat);
    cudaFree(d_o_custkey_flat);
    cudaFree(d_o_orderdate_flat);
    cudaFree(d_l_orderkey_flat);
    cudaFree(d_l_suppkey_flat);
    cudaFree(d_l_extprice_flat);
    cudaFree(d_l_discount_flat);
    mb_cuda_free(data_buf_s_suppkey);
    mb_cuda_free(data_buf_s_nationkey);
    mb_cuda_free(data_buf_c_custkey);
    mb_cuda_free(data_buf_c_nationkey);
    for (size_t i = 0; i < Q5_NUM_COL_BUFS; i++)
        mb_cuda_free(tile_col_bufs[i]);
    free(h_n_nkey);
    free(h_n_name);
    free(h_n_rkey);
    cudaStreamDestroy(stream);

    size_t total_ios = 0, total_bytes_read = 0;
    uint64_t total_kernel_launches = s_kernel_launches;
    for (size_t t = 0; t < nthreads; t++) {
        total_ios += workers[t].ios_completed;
        total_bytes_read += workers[t].bytes_read;
        total_kernel_launches += workers[t].kernel_launches;
    }
    tpch_worker_free(workers.data(), nthreads, num_devices, any_compressed);

    size_t total_pages = 0;
    for (const auto &fi : supplier_fields) total_pages += fi.npages;
    for (const auto &fi : customer_fields) total_pages += fi.npages;
    for (const auto &fi : orders_fields) total_pages += fi.npages;
    for (const auto &fi : lineitem_fields) total_pages += fi.npages;

    std::string comp_str = collect_compression_methods(
        {supplier_fields, customer_fields, orders_fields, lineitem_fields});
    free_fields_metadata(region_fields);
    free_fields_metadata(nation_fields);
    free_fields_metadata(supplier_fields);
    free_fields_metadata(customer_fields);
    free_fields_metadata(orders_fields);
    free_fields_metadata(lineitem_fields);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = total_ios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = comp_str,
        .gpu_mem_bytes = golap_gpu_mem_bytes,
        .gpu_app_bytes = golap_gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = total_kernel_launches,
    };
}

// ============================================================
// TPC-H Q3 — GOLAP sync mode (Shipping Priority)
// ============================================================
BenchmarkResult tpch_q3(BenchmarkOptions &options) {
    constexpr bool debug_print = false;

    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    TPCHTableMetadata *metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    const bool is_q3sel = (options.q3sel_selectivity > 0);
    const int sel_pct = is_q3sel ? options.q3sel_selectivity : 20;

    {
        const size_t page_size = metadatap->page_size;
        if (is_q3sel)
            std::cout << "=== TPCH Q3SEL (sel=" << sel_pct << "%) ===" << std::endl;
        else
            std::cout << "=== TPCH Q3 ===" << std::endl;
        std::cout << "Page Size: " << page_size << std::endl;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, page_size) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, page_size);
    }

    metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    TPCHTableMetadata &metadata = *metadatap;
    superpage_set_constants(metadata.page_size);
    const size_t page_size = metadata.page_size;
    const size_t num_devices = fds.size();

    const uint64_t nrecs_customer = metadata.table_customer_nrows;
    const uint64_t nrecs_orders = metadata.table_orders_nrows;
    const uint64_t nrecs_lineitem = metadata.table_lineitem_nrows;

    if (debug_print) std::cout << "nrecs_customer: " << nrecs_customer
              << ", nrecs_orders: " << nrecs_orders
              << ", nrecs_lineitem: " << nrecs_lineitem
              << ", num_devices: " << num_devices << std::endl;

    // ── Load field metadata ──
    // CUSTOMER fields: C_CUSTKEY (idx 0), C_MKTSEGMENT (idx 6)
    constexpr size_t num_customer_cols = TPCH::Query::Q3::NUM_CUSTOMER_SCAN_COLS;
    auto customer_cols_idx = TPCH::Query::Q3::CUSTOMER_SCAN_COLS;
    std::vector<FieldPageInfo> customer_fields(num_customer_cols);

    uint32_t saved_column = metadata.column;
    metadata.column = TPCH::common::Table::CUSTOMER;
    prepare_fields_metadata(fds, metadata, page_size, customer_cols_idx, customer_fields);

    // ORDERS fields: O_ORDERKEY (0), O_CUSTKEY (1), O_ORDERDATE (4), O_SHIPPRIORITY (7)
    constexpr size_t num_orders_cols = TPCH::Query::Q3::NUM_ORDERS_SCAN_COLS;
    auto orders_cols_idx = TPCH::Query::Q3::ORDERS_SCAN_COLS;
    std::vector<FieldPageInfo> orders_fields(num_orders_cols);

    metadata.column = TPCH::common::Table::ORDERS;
    prepare_fields_metadata(fds, metadata, page_size, orders_cols_idx, orders_fields);

    // LINEITEM fields: L_ORDERKEY (0), L_EXTENDEDPRICE (5), L_DISCOUNT (6), L_SHIPDATE (10)
    constexpr size_t num_lineitem_cols = TPCH::Query::Q3::NUM_LINEITEM_SCAN_COLS;
    auto lineitem_cols_idx = TPCH::Query::Q3::LINEITEM_SCAN_COLS;
    std::vector<FieldPageInfo> lineitem_fields(num_lineitem_cols);

    metadata.column = TPCH::common::Table::LINEITEM;
    prepare_fields_metadata(fds, metadata, page_size, lineitem_cols_idx, lineitem_fields);

    metadata.column = saved_column;

    const FieldPageInfo &fi_c_custkey    = customer_fields[0];
    const FieldPageInfo &fi_c_mktsegment = customer_fields[1];
    const FieldPageInfo &fi_o_orderkey   = orders_fields[0];
    const FieldPageInfo &fi_o_custkey    = orders_fields[1];
    const FieldPageInfo &fi_o_orderdate  = orders_fields[2];
    const FieldPageInfo &fi_o_shippriority = orders_fields[3];
    const FieldPageInfo &fi_l_orderkey   = lineitem_fields[0];
    const FieldPageInfo &fi_l_extprice   = lineitem_fields[1];
    const FieldPageInfo &fi_l_discount   = lineitem_fields[2];
    const FieldPageInfo &fi_l_shipdate   = lineitem_fields[3];

    if (debug_print) {
        std::cout << "  C_CUSTKEY: npages=" << fi_c_custkey.npages
                  << " compression=" << compression_method_name(fi_c_custkey.compression_method) << std::endl;
        std::cout << "  C_MKTSEGMENT: npages=" << fi_c_mktsegment.npages
                  << " compression=" << compression_method_name(fi_c_mktsegment.compression_method) << std::endl;
        std::cout << "  O_ORDERKEY: npages=" << fi_o_orderkey.npages
                  << " compression=" << compression_method_name(fi_o_orderkey.compression_method) << std::endl;
        std::cout << "  O_CUSTKEY: npages=" << fi_o_custkey.npages
                  << " compression=" << compression_method_name(fi_o_custkey.compression_method) << std::endl;
        std::cout << "  O_ORDERDATE: npages=" << fi_o_orderdate.npages
                  << " compression=" << compression_method_name(fi_o_orderdate.compression_method) << std::endl;
        std::cout << "  O_SHIPPRIORITY: npages=" << fi_o_shippriority.npages
                  << " compression=" << compression_method_name(fi_o_shippriority.compression_method) << std::endl;
        std::cout << "  L_ORDERKEY: npages=" << fi_l_orderkey.npages
                  << " compression=" << compression_method_name(fi_l_orderkey.compression_method) << std::endl;
        std::cout << "  L_EXTENDEDPRICE: npages=" << fi_l_extprice.npages
                  << " compression=" << compression_method_name(fi_l_extprice.compression_method) << std::endl;
        std::cout << "  L_DISCOUNT: npages=" << fi_l_discount.npages
                  << " compression=" << compression_method_name(fi_l_discount.compression_method) << std::endl;
        std::cout << "  L_SHIPDATE: npages=" << fi_l_shipdate.npages
                  << " compression=" << compression_method_name(fi_l_shipdate.compression_method) << std::endl;
    }

    // ── Check for compressed fields ──
    bool any_compressed = false;
    std::vector<FieldPageInfo> all_fields;
    for (auto &fi : customer_fields) all_fields.push_back(fi);
    for (auto &fi : orders_fields) all_fields.push_back(fi);
    for (auto &fi : lineitem_fields) all_fields.push_back(fi);
    for (auto &fi : all_fields) {
        if (fi.compression_method != CompressionMethod::NONE) any_compressed = true;
    }

    // ── GDS setup ──
    const size_t nthreads = options.nthreads;
    if (debug_print) std::cout << "[Q3] nthreads=" << nthreads << std::endl;
    mb_cufile_driver_open();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    struct Q3LoaderCtx {
        std::vector<std::vector<CUfileHandle_t>> cufile_handles;
        std::vector<std::vector<int>> dup_fds;
        void *io_buf = nullptr;
        size_t io_batch_pages = 0;
        cudaStream_t stream = nullptr;
        NvcompDecompCtx nvctx;
        size_t bytes_read = 0;
        size_t ios_completed = 0;
        size_t kernel_launches = 0;
    };
    std::vector<Q3LoaderCtx> loaders(nthreads);
    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    // Phase 1: create CUfileHandles and streams (io_buf/nvcomp deferred for budget control)
    for (size_t t = 0; t < nthreads; t++) {
        auto &L = loaders[t];
        L.cufile_handles.resize(num_devices);
        L.dup_fds.resize(num_devices);
        for (size_t d = 0; d < num_devices; d++) {
            int duped = dup(fds[d]);
            if (duped < 0) { std::cerr << "dup failed" << std::endl; exit(EXIT_FAILURE); }
            L.dup_fds[d].push_back(duped);
            L.cufile_handles[d].push_back(mb_cufile_handle_register(duped));
        }
        CUDA_CHECK(cudaStreamCreate(&L.stream));
    }

    // Pre-allocated thread vector shared across loader lambdas (R4: avoid heap alloc in timed section)
    std::vector<std::thread> _mt_threads;
    _mt_threads.reserve(nthreads);

    auto load_field_mt = [&](const FieldPageInfo &field, void *data_buf) {
        if (nthreads <= 1) {
            auto &L = loaders[0];
            NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
            gds_load_field_sync(field, page_size, num_devices,
                                L.cufile_handles, L.io_buf, data_buf,
                                nv, L.stream, L.bytes_read, L.ios_completed,
                                L.kernel_launches,
                                0, 0, 0, false, L.io_batch_pages);
            return;
        }
        size_t npages = field.npages;
        size_t pages_per_thr = (npages + nthreads - 1) / nthreads;
        auto &threads = _mt_threads; threads.clear();
        for (size_t t = 0; t < nthreads; t++) {
            size_t start = t * pages_per_thr;
            if (start >= npages) break;
            size_t count = std::min(pages_per_thr, npages - start);
            threads.emplace_back([&, t, start, count]() {
                auto &L = loaders[t];
                mb_cuda_set_context(cuda_ctx_handle);
                NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                gds_load_field_sync(field, page_size, num_devices,
                                    L.cufile_handles, L.io_buf, data_buf,
                                    nv, L.stream,
                                    L.bytes_read, L.ios_completed,
                                    L.kernel_launches,
                                    start, count, 0, false, L.io_batch_pages);
            });
        }
        for (auto &th : threads) th.join();
    };

    // ── Selective page loading (zone map mode) ──
    // Sum ios_completed across all loaders (for coalescing ratio)
    auto sum_ios_completed = [&]() -> size_t {
        size_t sum = 0;
        for (size_t t = 0; t < nthreads; t++) sum += loaders[t].ios_completed;
        return sum;
    };

    struct CoalesceInfo {
        const char *field_name;
        size_t pages;
        size_t ios;
    };
    std::vector<CoalesceInfo> coalesce_stats;

    auto load_field_selective_mt = [&](const FieldPageInfo &field, void *data_buf,
                                       const std::vector<uint32_t> &active_pgs,
                                       const char *field_name = nullptr) {
        CUDA_CHECK(cudaMemset(data_buf, 0, field.npages * page_size));
        size_t n_active = active_pgs.size();
        if (n_active == 0) return;

        size_t ios_before = sum_ios_completed();

        auto load_selective = [&](size_t thr_start, size_t thr_count, Q3LoaderCtx &L) {
            NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
            gds_load_field_sync_selective(field, page_size, num_devices,
                                          L.cufile_handles, L.io_buf, data_buf,
                                          nv, L.stream, L.bytes_read, L.ios_completed,
                                          L.kernel_launches,
                                          active_pgs.data() + thr_start, thr_count,
                                          0, false, L.io_batch_pages);
        };

        if (nthreads <= 1) {
            load_selective(0, n_active, loaders[0]);
        } else {
            size_t per_thr = (n_active + nthreads - 1) / nthreads;
            auto &threads = _mt_threads; threads.clear();
            for (size_t t = 0; t < nthreads; t++) {
                size_t thr_start = t * per_thr;
                if (thr_start >= n_active) break;
                size_t thr_count = std::min(per_thr, n_active - thr_start);
                threads.emplace_back([&, t, thr_start, thr_count]() {
                    mb_cuda_set_context(cuda_ctx_handle);
                    load_selective(thr_start, thr_count, loaders[t]);
                });
            }
            for (auto &th : threads) th.join();
        }

        size_t ios_after = sum_ios_completed();
        size_t ios_delta = ios_after - ios_before;
        if (field_name && ios_delta > 0) {
            coalesce_stats.push_back({field_name, n_active, ios_delta});
        }
    };

    // ── Tile execution constants ──
    constexpr size_t Q3_TILE_PAGES_MAX = 1024;
    const size_t o_npages_i32 = fi_o_orderdate.npages;
    const size_t Q3_TILE_PAGES_O = o_npages_i32;  // all ORDERS in one tile
    const size_t l_npages_i32 = fi_l_shipdate.npages;
    const size_t o_npages_i64 = fi_o_orderkey.npages;
    const size_t l_npages_i64 = fi_l_orderkey.npages;

    // ── Host prefix sums ──
    auto build_host_ps = [](const FieldPageInfo &fi) -> std::vector<uint64_t> {
        std::vector<uint64_t> ps(fi.npages + 1, 0);
        if (fi.prefix_sum_nrecs != nullptr) {
            memcpy(ps.data(), fi.prefix_sum_nrecs, (fi.npages + 1) * sizeof(uint64_t));
        } else {
            std::cerr << "[Q3] WARNING: prefix_sum_nrecs not available" << std::endl;
        }
        return ps;
    };

    auto ps_o_i32 = build_host_ps(fi_o_orderdate);
    auto ps_o_i64 = build_host_ps(fi_o_orderkey);
    auto ps_l_i32 = build_host_ps(fi_l_shipdate);
    auto ps_l_i64 = build_host_ps(fi_l_orderkey);

    if (o_npages_i32 != fi_o_shippriority.npages)
        std::cerr << "[Q3] WARNING: O_ORDERDATE.npages != O_SHIPPRIORITY.npages" << std::endl;
    if (o_npages_i64 != fi_o_custkey.npages)
        std::cerr << "[Q3] WARNING: O_ORDERKEY.npages != O_CUSTKEY.npages" << std::endl;
    if (l_npages_i32 != fi_l_extprice.npages || l_npages_i32 != fi_l_discount.npages)
        std::cerr << "[Q3] WARNING: LINEITEM INT32 npages mismatch" << std::endl;

    size_t max_npages_i64_q3 = std::max(o_npages_i64, l_npages_i64);

    // ── Tile info computation ──
    size_t o_num_tiles = (o_npages_i32 + Q3_TILE_PAGES_O - 1) / Q3_TILE_PAGES_O;

    uint64_t tile_nrows_max_o = 0;
    size_t o_i64_tile_npages_max = 0;
    uint64_t o_i64_nrows_max = 0;
    for (size_t p_lo = 0; p_lo < o_npages_i32; p_lo += Q3_TILE_PAGES_O) {
        size_t tile_np = std::min(Q3_TILE_PAGES_O, o_npages_i32 - p_lo);
        uint64_t first_row = ps_o_i32[p_lo];
        uint64_t last_row = ps_o_i32[p_lo + tile_np];
        uint64_t tile_nr = last_row - first_row;
        tile_nrows_max_o = std::max(tile_nrows_max_o, tile_nr);
        if (tile_nr == 0) continue;
        auto it_s = std::upper_bound(ps_o_i64.begin(), ps_o_i64.end(), first_row);
        size_t i64_s = (it_s == ps_o_i64.begin()) ? 0 : (size_t)(it_s - ps_o_i64.begin()) - 1;
        auto it_e = std::upper_bound(ps_o_i64.begin(), ps_o_i64.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_o_i64.begin());
        o_i64_tile_npages_max = std::max(o_i64_tile_npages_max, i64_e - i64_s);
        o_i64_nrows_max = std::max(o_i64_nrows_max, ps_o_i64[i64_e] - ps_o_i64[i64_s]);
    }

    // ── Pre-compute HT cap for budget estimation ──
    uint64_t pre_est_orders_qual = is_q3sel
        ? std::max((uint64_t)1024, nrecs_orders * (uint64_t)sel_pct / 100)
        : std::max((uint64_t)1024, nrecs_orders / 8);
    uint32_t pre_orders_ht_cap = 1;
    while (pre_orders_ht_cap < pre_est_orders_qual * 2) pre_orders_ht_cap <<= 1;

    // ── Budget-based Q3_TILE_PAGES (40 GiB cap) ──
    size_t Q3_TILE_PAGES;
    {
        constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        uint64_t fixed_bytes = 0;
        // CUSTOMER flat + hash set
        fixed_bytes += (uint64_t)nrecs_customer * 8;
        { uint64_t e = is_q3sel
              ? std::max((uint64_t)1024, nrecs_customer * (uint64_t)sel_pct / 100)
              : std::max((uint64_t)1024, nrecs_customer / 4);
          uint32_t cl = is_q3sel ? 1 : 2;
          uint32_t c = 1; while (c < e * cl) c <<= 1;
          fixed_bytes += (uint64_t)c * 8; }
        // ORDERS flat (4 fields) + HT + aggr HT + results + sort temp
        fixed_bytes += tile_nrows_max_o * 16 + (uint64_t)o_i64_nrows_max * 16;
        {
          uint64_t ht_cap_for_budget = pre_orders_ht_cap;
          if (is_q3sel) {
              uint64_t ht_bytes_ideal = (uint64_t)pre_orders_ht_cap * 96;
              uint64_t ht_budget_limit = GPU_MEM_BUDGET / 2;
              if (ht_bytes_ideal > ht_budget_limit) {
                  uint32_t max_c = static_cast<uint32_t>(ht_budget_limit / 96);
                  uint32_t max_c_p2 = 1;
                  while (max_c_p2 * 2 <= max_c) max_c_p2 <<= 1;
                  uint32_t min_c = 1;
                  while (min_c < pre_est_orders_qual) min_c <<= 1;
                  if (max_c_p2 < min_c) max_c_p2 = min_c;
                  ht_cap_for_budget = max_c_p2;
              }
          }
          fixed_bytes += (uint64_t)ht_cap_for_budget * 96;
        }
        fixed_bytes += 1024;
        // Worker IO reserve
        size_t o_staging = std::max(Q3_TILE_PAGES_O, o_i64_tile_npages_max);
        size_t est_ppw = std::max((size_t)1, (o_staging + nthreads - 1) / nthreads);
        fixed_bytes += (uint64_t)nthreads * (est_ppw * page_size + 2 * 1024 * 1024);
        // Per LINEITEM tile page cost (1 staging buf, 3 INT32 + 1 INT64 flat)
        double i64_r = (l_npages_i32 > 0) ? (double)l_npages_i64 / l_npages_i32 : 2.0;
        i64_r = std::max(i64_r, 1.0) * 1.05;
        uint64_t stg_base = (uint64_t)o_staging * page_size;
        size_t flat_per_tp = (size_t)((6.0 + i64_r) * page_size);
        size_t T;
        if (fixed_bytes + stg_base < GPU_MEM_BUDGET) {
            T = (GPU_MEM_BUDGET - fixed_bytes - stg_base) / flat_per_tp;
            if ((size_t)(T * std::max(1.0, i64_r)) > o_staging) {
                size_t full_per_tp = (size_t)((6.0 + 2.0 * i64_r) * page_size);
                T = (fixed_bytes < GPU_MEM_BUDGET)
                    ? (GPU_MEM_BUDGET - fixed_bytes) / full_per_tp : 1;
            }
        } else {
            T = 1;
        }
        Q3_TILE_PAGES = std::min(Q3_TILE_PAGES_MAX, std::max((size_t)1, T));
    }
    Q3_TILE_PAGES = std::min(Q3_TILE_PAGES, l_npages_i32);  // never exceed actual column
    size_t l_num_tiles = (l_npages_i32 + Q3_TILE_PAGES - 1) / Q3_TILE_PAGES;

    uint64_t tile_nrows_max_l = 0;
    size_t l_i64_tile_npages_max = 0;
    uint64_t l_i64_nrows_max = 0;
    for (size_t p_lo = 0; p_lo < l_npages_i32; p_lo += Q3_TILE_PAGES) {
        size_t tile_np = std::min(Q3_TILE_PAGES, l_npages_i32 - p_lo);
        uint64_t first_row = ps_l_i32[p_lo];
        uint64_t last_row = ps_l_i32[p_lo + tile_np];
        uint64_t tile_nr = last_row - first_row;
        tile_nrows_max_l = std::max(tile_nrows_max_l, tile_nr);
        if (tile_nr == 0) continue;
        auto it_s = std::upper_bound(ps_l_i64.begin(), ps_l_i64.end(), first_row);
        size_t i64_s = (it_s == ps_l_i64.begin()) ? 0 : (size_t)(it_s - ps_l_i64.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64.begin(), ps_l_i64.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_l_i64.begin());
        l_i64_tile_npages_max = std::max(l_i64_tile_npages_max, i64_e - i64_s);
        l_i64_nrows_max = std::max(l_i64_nrows_max, ps_l_i64[i64_e] - ps_l_i64[i64_s]);
    }

    size_t staging_pages = std::max({Q3_TILE_PAGES_O,
                                      o_i64_tile_npages_max,
                                      l_i64_tile_npages_max});
    if (tile_nrows_max_o == 0) tile_nrows_max_o = 1;
    if (tile_nrows_max_l == 0) tile_nrows_max_l = 1;
    if (o_i64_nrows_max == 0) o_i64_nrows_max = 1;
    if (l_i64_nrows_max == 0) l_i64_nrows_max = 1;

    if (debug_print) {
        std::cout << "[Q3]   ORDERS: o_npages_i32=" << o_npages_i32
                  << " o_npages_i64=" << o_npages_i64
                  << " h_ps_odate[0]=" << ps_o_i32[1]
                  << " h_ps_okey[0]=" << ps_o_i64[1] << std::endl;
        std::cout << "[Q3]   LINEITEM: l_npages_i32=" << l_npages_i32
                  << " l_npages_i64=" << l_npages_i64
                  << " h_ps_lsd[0]=" << ps_l_i32[1]
                  << " h_ps_lokey[0]=" << ps_l_i64[1] << std::endl;
    }

    // ── Pre-allocate shared prefix_sum buffer (tile-sized) ──
    uint64_t *d_ps_shared = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_shared, staging_pages * sizeof(uint64_t)));
    std::vector<uint64_t> h_tile_ps(staging_pages);  // shared across flatten lambdas (Rule 4)

    // Pre-uploaded prefix_sum GPU arrays (Rule 3: populated before total_start)
    std::unordered_map<const uint64_t*, uint64_t*> d_full_ps_gpu;

    // ── Tile-aware flatten helpers ──
    auto flatten_int32_field_tile = [&](const FieldPageInfo &fi, void *data_buf,
                                         uint64_t tile_nrows, uint64_t *d_flat,
                                         size_t tile_start, size_t tile_npages) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            compute_batch_ps_kernel<<<((uint32_t)tile_npages + 255) / 256, 256, 0, stream>>>(
                it->second, d_ps_shared, (uint32_t)tile_start, (uint32_t)tile_npages);
            s_kernel_launches++;
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
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps.data(),
                                  tile_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        q13_flatten_int32_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, tile_npages, tile_nrows, d_flat, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    auto flatten_int64_field_tile = [&](const FieldPageInfo &fi, void *data_buf,
                                         uint64_t tile_nrows, uint64_t *d_flat,
                                         size_t tile_start, size_t tile_npages) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            compute_batch_ps_kernel<<<((uint32_t)tile_npages + 255) / 256, 256, 0, stream>>>(
                it->second, d_ps_shared, (uint32_t)tile_start, (uint32_t)tile_npages);
            s_kernel_launches++;
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
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps.data(),
                                  tile_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        q13_flatten_int64_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, tile_npages, tile_nrows, d_flat, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // INT32 flatten with page-active mask + fill_value
    auto flatten_int32_field_tile_masked = [&](const FieldPageInfo &fi, void *data_buf,
                                                 uint64_t tile_nrows, uint64_t *d_flat,
                                                 size_t tile_start, size_t tile_npages,
                                                 const uint8_t *d_mask, uint64_t fill_value) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            compute_batch_ps_kernel<<<((uint32_t)tile_npages + 255) / 256, 256, 0, stream>>>(
                it->second, d_ps_shared, (uint32_t)tile_start, (uint32_t)tile_npages);
            s_kernel_launches++;
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
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_tile_ps.data(),
                                  tile_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        q13_flatten_int32_pages_ps_masked(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, tile_npages, tile_nrows,
            d_mask, fill_value, d_flat, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // INT64 flatten (full-column, for CUSTOMER Phase 1)
    auto flatten_int64_field = [&](const FieldPageInfo &fi, void *data_buf, uint64_t nrecs, uint64_t *d_flat) {
        auto it = d_full_ps_gpu.find(fi.prefix_sum_nrecs);
        if (it != d_full_ps_gpu.end()) {
            compute_batch_ps_kernel<<<((uint32_t)fi.npages + 255) / 256, 256, 0, stream>>>(
                it->second, d_ps_shared, 0, (uint32_t)fi.npages);
            s_kernel_launches++;
        } else {
#if 0
            std::vector<uint64_t> h_ps(fi.npages);
            uint64_t cum = 0;
            for (size_t p = 0; p < fi.npages; p++) {
                uint32_t nalloc = 0;
                CUDA_CHECK(cudaMemcpy(&nalloc,
                    static_cast<const char *>(data_buf) + p * page_size,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cum += nalloc;
                h_ps[p] = cum;
            }
            CUDA_CHECK(cudaMemcpy(d_ps_shared, h_ps.data(),
                                  fi.npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
#else
            std::cerr << "FATAL: prefix_sum not uploaded to GPU for field" << std::endl;
            abort();
#endif
        }
        q13_flatten_int64_pages_ps(
            static_cast<const char *>(data_buf),
            page_size, d_ps_shared, fi.npages, nrecs, d_flat, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // ── Tile-aware multi-threaded page loader ──
    auto load_field_tile_mt = [&](const FieldPageInfo &field, void *data_buf,
                                  size_t tile_start, size_t tile_npages) {
        if (nthreads <= 1) {
            auto &L = loaders[0];
            NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
            gds_load_field_sync(field, page_size, num_devices,
                                L.cufile_handles, L.io_buf, data_buf,
                                nv, L.stream, L.bytes_read, L.ios_completed,
                                L.kernel_launches,
                                tile_start, tile_npages, tile_start, false, L.io_batch_pages);
            return;
        }
        size_t pages_per_thr = (tile_npages + nthreads - 1) / nthreads;
        auto &threads = _mt_threads; threads.clear();
        for (size_t t = 0; t < nthreads; t++) {
            size_t start = tile_start + t * pages_per_thr;
            size_t end = std::min(start + pages_per_thr, tile_start + tile_npages);
            if (start >= end) break;
            size_t count = end - start;
            threads.emplace_back([&, t, start, count]() {
                auto &L = loaders[t];
                mb_cuda_set_context(cuda_ctx_handle);
                NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                gds_load_field_sync(field, page_size, num_devices,
                                    L.cufile_handles, L.io_buf, data_buf,
                                    nv, L.stream, L.bytes_read, L.ios_completed,
                                    L.kernel_launches,
                                    start, count, tile_start, false, L.io_batch_pages);
            });
        }
        for (auto &th : threads) th.join();
    };

    // ── Selective tile loader (zone map) ──
    auto load_field_selective_tile_mt = [&](const FieldPageInfo &field, void *data_buf,
                                             const std::vector<uint32_t> &active_pgs,
                                             size_t output_page_offset,
                                             const char *field_name = nullptr) {
        size_t n_active = active_pgs.size();
        if (n_active == 0) return;

        size_t ios_before = sum_ios_completed();

        auto load_selective = [&](size_t thr_start, size_t thr_count, Q3LoaderCtx &L) {
            NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
            gds_load_field_sync_selective(field, page_size, num_devices,
                                          L.cufile_handles, L.io_buf, data_buf,
                                          nv, L.stream, L.bytes_read, L.ios_completed,
                                          L.kernel_launches,
                                          active_pgs.data() + thr_start, thr_count,
                                          output_page_offset, false, L.io_batch_pages);
        };

        if (nthreads <= 1) {
            load_selective(0, n_active, loaders[0]);
        } else {
            size_t per_thr = (n_active + nthreads - 1) / nthreads;
            auto &threads = _mt_threads; threads.clear();
            for (size_t t = 0; t < nthreads; t++) {
                size_t thr_start = t * per_thr;
                if (thr_start >= n_active) break;
                size_t thr_count = std::min(per_thr, n_active - thr_start);
                threads.emplace_back([&, t, thr_start, thr_count]() {
                    mb_cuda_set_context(cuda_ctx_handle);
                    load_selective(thr_start, thr_count, loaders[t]);
                });
            }
            for (auto &th : threads) th.join();
        }

        size_t ios_after = sum_ios_completed();
        size_t ios_delta = ios_after - ios_before;
        if (field_name && ios_delta > 0) {
            coalesce_stats.push_back({field_name, n_active, ios_delta});
        }
    };

    // ════════════════════════════════════════════════════════
    // Pre-allocate GPU memory (tile-sized)
    // ════════════════════════════════════════════════════════

    // Phase 1: CUSTOMER flat arrays + hash set (data loaded via tile_data_buf)
    uint64_t *d_c_custkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c_custkey_flat, nrecs_customer * sizeof(uint64_t)));

    // Hash set for CUSTOMER custkeys
    uint64_t est_building = is_q3sel
        ? std::max((uint64_t)1024, nrecs_customer * (uint64_t)sel_pct / 100)
        : std::max((uint64_t)1024, nrecs_customer / 4);
    uint32_t custset_cap = 1;
    uint32_t custset_load = is_q3sel ? 1 : 2;
    while (custset_cap < est_building * custset_load) custset_cap <<= 1;
    uint32_t custset_mask = custset_cap - 1;
    uint64_t *d_custkey_set = nullptr;
    CUDA_CHECK(cudaMalloc(&d_custkey_set, custset_cap * sizeof(uint64_t)));

    // Shared staging buffer (reused across columns and tiles)
    void *tile_data_buf = mb_cuda_alloc(staging_pages * page_size);

    // Phase 2: ORDERS flat arrays (tile-sized) + hash table
    uint64_t *d_o_orderdate_flat = nullptr, *d_o_shippriority_flat = nullptr;
    uint64_t *d_o_orderkey_flat = nullptr, *d_o_custkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_orderdate_flat, tile_nrows_max_o * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_o_shippriority_flat, tile_nrows_max_o * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_o_orderkey_flat, o_i64_nrows_max * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_o_custkey_flat, o_i64_nrows_max * sizeof(uint64_t)));

    // ORDERS HT: qualifying rows
    uint64_t est_orders_qual = is_q3sel
        ? std::max((uint64_t)1024, nrecs_orders * (uint64_t)sel_pct / 100)
        : std::max((uint64_t)1024, nrecs_orders / 8);
    uint32_t orders_ht_cap = 1;
    while (orders_ht_cap < est_orders_qual * 2) orders_ht_cap <<= 1;

    {
        static constexpr uint64_t GPU_MEM_BUDGET_Q3 = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t app_used = gpu_free_start - gpu_free_now;
        if (app_used < GPU_MEM_BUDGET_Q3) {
            uint64_t remaining = GPU_MEM_BUDGET_Q3 - app_used;
            uint64_t custset_bytes = (uint64_t)custset_cap * sizeof(uint64_t);
            if (remaining > custset_bytes) {
                uint64_t ht_budget = remaining - custset_bytes;
                constexpr uint64_t BYTES_PER_ENTRY =
                    8 + 8 + 8 + 8 + sizeof(Q3ResultRow) + sizeof(Q3ResultRow);
                uint32_t max_cap = static_cast<uint32_t>(ht_budget / BYTES_PER_ENTRY);
                uint32_t max_cap_p2 = 1;
                while (max_cap_p2 * 2 <= max_cap) max_cap_p2 <<= 1;
                uint32_t min_cap = 1;
                while (min_cap < est_orders_qual) min_cap <<= 1;
                if (max_cap_p2 < min_cap) max_cap_p2 = min_cap;
                if (orders_ht_cap > max_cap_p2) {
                    std::cout << "[Q3] Budget cap: orders_ht_cap " << orders_ht_cap
                              << " -> " << max_cap_p2
                              << " (remaining=" << (remaining >> 20) << " MiB)" << std::endl;
                    orders_ht_cap = max_cap_p2;
                }
            }
        }
    }

    uint32_t orders_ht_mask = orders_ht_cap - 1;
    uint64_t *d_orders_ht_keys = nullptr;
    uint64_t *d_orders_ht_payloads = nullptr;
    CUDA_CHECK(cudaMalloc(&d_orders_ht_keys, orders_ht_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_orders_ht_payloads, orders_ht_cap * sizeof(uint64_t)));

    // Phase 3: LINEITEM flat arrays (tile-sized) + aggregation hash map
    uint64_t *d_l_shipdate_flat = nullptr, *d_l_extprice_flat = nullptr;
    uint64_t *d_l_discount_flat = nullptr;
    uint64_t *d_l_orderkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l_shipdate_flat, tile_nrows_max_l * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_l_extprice_flat, tile_nrows_max_l * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_l_discount_flat, tile_nrows_max_l * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_l_orderkey_flat, l_i64_nrows_max * sizeof(uint64_t)));

    // Aggregation hash map: keys = l_orderkey, values = revenue (int64_t)
    uint32_t aggr_cap = orders_ht_cap;  // same capacity as orders HT
    uint32_t aggr_mask = aggr_cap - 1;
    uint64_t *d_aggr_keys = nullptr;
    int64_t  *d_aggr_revenues = nullptr;
    CUDA_CHECK(cudaMalloc(&d_aggr_keys, aggr_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_aggr_revenues, aggr_cap * sizeof(int64_t)));

    // Zone map tile mask
    uint8_t *d_tile_page_mask = nullptr;
    if (options.enable_zonemap) {
        CUDA_CHECK(cudaMalloc(&d_tile_page_mask, staging_pages * sizeof(uint8_t)));
    }

    // Phase 4: result arrays
    Q3ResultRow *d_results = nullptr;
    uint32_t *d_result_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, aggr_cap * sizeof(Q3ResultRow)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));

    // Pre-allocate CUB DeviceMergeSort temp buffer (Rule 4: avoid thrust hidden cudaMalloc)
    void *d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceMergeSort::SortKeys(nullptr, sort_temp_bytes,
        d_results, (int)aggr_cap, Q3ResultCmp{}, stream);
    CUDA_CHECK(cudaMalloc(&d_sort_temp, sort_temp_bytes));

    // GPU zonemap I/O mask buffers
    uint8_t *d_mask_ord_i64 = nullptr;
    uint8_t *d_mask_li_i64 = nullptr;
    std::vector<uint8_t> h_mask_ord_i64(o_npages_i64, 1);
    std::vector<uint8_t> h_mask_li_i64(l_npages_i64, 1);

    // Q3SEL mktseg dict_id range: AUTOMOBILE=0, BUILDING=1, FURNITURE=2, MACHINERY=3, HOUSEHOLD=4
    // sel=20% → {1}, sel=40% → {0,1}, sel=60% → {0,1,2}, sel=80% → {0,1,2,3}, sel=100% → all
    int32_t q3sel_mktseg_lo = 0, q3sel_mktseg_hi = 4;
    if (is_q3sel && sel_pct < 100) {
        static constexpr int32_t seg_dict_ids[] = {1, 0, 2, 3, 4}; // BUILDING, AUTOMOBILE, FURNITURE, MACHINERY, HOUSEHOLD
        int nseg = std::max(1, sel_pct / 20);
        q3sel_mktseg_lo = seg_dict_ids[0];
        q3sel_mktseg_hi = seg_dict_ids[0];
        for (int i = 1; i < nseg; i++) {
            q3sel_mktseg_lo = std::min(q3sel_mktseg_lo, seg_dict_ids[i]);
            q3sel_mktseg_hi = std::max(q3sel_mktseg_hi, seg_dict_ids[i]);
        }
    }
    const bool q3sel_do_mktseg_pruning = is_q3sel && sel_pct < 100 && options.enable_zonemap;

    std::vector<bool> ord_page_active(o_npages_i32, true);
    std::vector<bool> li_page_active(l_npages_i32, true);

    // ── Zonemap pre-alloc (outside timing): resolve metadata, alloc 3 stats buffers, ctx, cuFile handles ──
    // NOTE: must come BEFORE loader io_buf allocation to avoid cuFileBufRegister limit
    struct ZmStatsInfo {
        uint64_t nstats, stats_start, stats_npg;
    };
    ZmStatsInfo zm_ord_odate{}, zm_ord_mktseg{};
    ZmStatsInfo zm_li_lsd{}, zm_li_sw_odate{}, zm_li_sw_mktseg{};
    void *d_zm_stats_base = nullptr;  // single allocation, 3 slots at offsets
    void *d_zm_stats_buf[3] = {nullptr, nullptr, nullptr};
    uint64_t zm_stats_buf_bytes = 0;  // per-slot size
    GdsZonemapCtx zm_ctx{};
    CUfileHandle_t zm_cufile_handles[MAX_GDS_DEVICES];
    int zm_dup_fds[MAX_GDS_DEVICES];
    bool zm_handles_open = false;

    if (options.enable_zonemap) {
        CUDA_CHECK(cudaMalloc(&d_mask_ord_i64, o_npages_i64));
        CUDA_CHECK(cudaMalloc(&d_mask_li_i64, l_npages_i64));

        // ORDERS stats metadata
        const size_t o_odate_field = TPCH::common::O_ORDERDATE;
        if (!is_q3sel || !options.disable_other_filters) {
            zm_ord_odate = { metadata.table_orders_nstats[o_odate_field],
                             metadata.table_orders_stats_start_page_ids[o_odate_field],
                             metadata.table_orders_stats_npages[o_odate_field] };
        }
        const size_t sw_mktseg_idx = TPCH::common::OS_SIDEWAYS_C_MKTSEGMENT;
        zm_ord_mktseg = { metadata.table_orders_sideways_nstats[o_odate_field][sw_mktseg_idx],
                          metadata.table_orders_sideways_stats_start_page_ids[o_odate_field][sw_mktseg_idx],
                          metadata.table_orders_sideways_stats_npages[o_odate_field][sw_mktseg_idx] };

        // LINEITEM stats metadata
        const size_t l_sd_field = TPCH::common::L_SHIPDATE;
        if (!is_q3sel || !options.disable_other_filters) {
            zm_li_lsd = { metadata.table_lineitem_nstats[l_sd_field],
                          metadata.table_lineitem_stats_start_page_ids[l_sd_field],
                          metadata.table_lineitem_stats_npages[l_sd_field] };
            const size_t sw_odate_idx = TPCH::common::LS_SIDEWAYS_O_ORDERDATE;
            zm_li_sw_odate = { metadata.table_lineitem_sideways_nstats[l_sd_field][sw_odate_idx],
                               metadata.table_lineitem_sideways_stats_start_page_ids[l_sd_field][sw_odate_idx],
                               metadata.table_lineitem_sideways_stats_npages[l_sd_field][sw_odate_idx] };
        }
        const size_t li_sw_mktseg_idx = TPCH::common::LS_SIDEWAYS_C_MKTSEGMENT;
        zm_li_sw_mktseg = { metadata.table_lineitem_sideways_nstats[l_sd_field][li_sw_mktseg_idx],
                            metadata.table_lineitem_sideways_stats_start_page_ids[l_sd_field][li_sw_mktseg_idx],
                            metadata.table_lineitem_sideways_stats_npages[l_sd_field][li_sw_mktseg_idx] };

        // Allocate 1 contiguous GPU buffer with 3 slots (single cuFileBufRegister)
        uint64_t max_stats_npg = std::max({zm_ord_odate.stats_npg, zm_ord_mktseg.stats_npg,
                                            zm_li_lsd.stats_npg, zm_li_sw_odate.stats_npg,
                                            zm_li_sw_mktseg.stats_npg});
        zm_stats_buf_bytes = max_stats_npg * page_size;
        if (zm_stats_buf_bytes > 0) {
            uint64_t total_zm_bytes = zm_stats_buf_bytes * 3;
            CUDA_CHECK(cudaMalloc(&d_zm_stats_base, total_zm_bytes));
            GDS_CHECK(cuFileBufRegister(d_zm_stats_base, total_zm_bytes, 0));
            for (int i = 0; i < 3; i++)
                d_zm_stats_buf[i] = static_cast<char*>(d_zm_stats_base) + i * zm_stats_buf_bytes;
        }

        uint64_t max_npages_zm = std::max((uint64_t)o_npages_i32, (uint64_t)l_npages_i32);
        zm_ctx = gds_zonemap_ctx_create(max_npages_zm);

        for (size_t d = 0; d < num_devices; d++) {
            zm_dup_fds[d] = dup(fds[d]);
            zm_cufile_handles[d] = mb_cufile_handle_register(zm_dup_fds[d]);
        }
        zm_handles_open = true;
    }

    // Phase 2: allocate per-loader io_buf with dynamic sizing (Rule 4: gpu_mem ≤ 40 GiB)
    {
        static constexpr size_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_before_loaders = 0;
        cudaMemGetInfo(&gpu_free_before_loaders, &gpu_total_dummy);
        size_t non_loader_bytes = gpu_free_start - gpu_free_before_loaders;
        size_t min_io_pages = (staging_pages + nthreads - 1) / nthreads;
        size_t q3_io_pages = min_io_pages;
        if (GPU_MEM_BUDGET > non_loader_bytes) {
            size_t nvcomp_per_loader = 2 * 1024 * 1024;  // ~2 MiB nvcomp overhead
            size_t budget_for_loaders = GPU_MEM_BUDGET - non_loader_bytes;
            size_t per_loader_budget = budget_for_loaders / nthreads;
            if (per_loader_budget > nvcomp_per_loader) {
                q3_io_pages = (per_loader_budget - nvcomp_per_loader) / page_size;
            }
            q3_io_pages = std::max(q3_io_pages, min_io_pages);
            q3_io_pages = std::min(q3_io_pages, (size_t)GDS_SYNC_BATCH_PAGES);
        }
        for (size_t t = 0; t < nthreads; t++) {
            auto &L = loaders[t];
            L.io_batch_pages = q3_io_pages;
            L.io_buf = mb_cuda_alloc(q3_io_pages * page_size);
            GDS_CHECK(cuFileBufRegister(L.io_buf, q3_io_pages * page_size, 0));
            if (any_compressed) {
                nvcomp_decompctx_alloc(L.nvctx, q3_io_pages, page_size, all_fields);
            }
        }
        std::cout << "[Q3] Per-loader IO: " << q3_io_pages << " pages ("
                  << (q3_io_pages * page_size / (1024*1024)) << " MiB), "
                  << nthreads << " loaders" << std::endl;
    }

    std::cout << "[Q3] ORDERS tile execution: " << o_num_tiles << " tiles of "
              << Q3_TILE_PAGES_O << " pages, tile_nrows_max=" << tile_nrows_max_o << std::endl;
    std::cout << "[Q3] LINEITEM tile execution: " << l_num_tiles << " tiles of "
              << Q3_TILE_PAGES << " pages, tile_nrows_max=" << tile_nrows_max_l << std::endl;

    // Pre-upload prefix_sum arrays to GPU (Rule 3: before total_start)
    {
        auto upload_ps = [&](const FieldPageInfo &fi) {
            if (!fi.prefix_sum_nrecs || d_full_ps_gpu.count(fi.prefix_sum_nrecs)) return;
            uint64_t *d_ps = nullptr;
            CUDA_CHECK(cudaMalloc(&d_ps, (fi.npages + 1) * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_ps, fi.prefix_sum_nrecs,
                                  (fi.npages + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            d_full_ps_gpu[fi.prefix_sum_nrecs] = d_ps;
        };
        upload_ps(fi_c_custkey); upload_ps(fi_c_mktsegment);
        upload_ps(fi_o_orderdate); upload_ps(fi_o_shippriority);
        upload_ps(fi_o_orderkey); upload_ps(fi_o_custkey);
        upload_ps(fi_l_shipdate); upload_ps(fi_l_extprice);
        upload_ps(fi_l_discount); upload_ps(fi_l_orderkey);
    }

    // Pre-compute batch metadata for C_CUSTKEY (Rule 3: before total_start)
    struct Q3BatchMeta { uint64_t row_start, batch_nrecs; };
    std::vector<Q3BatchMeta> q3_bm_custkey;
    for (size_t pg = 0; pg < fi_c_custkey.npages; pg += Q3_TILE_PAGES) {
        size_t bnp = std::min(Q3_TILE_PAGES, fi_c_custkey.npages - pg);
        uint64_t rs = fi_c_custkey.prefix_sum_nrecs ? fi_c_custkey.prefix_sum_nrecs[pg] : 0;
        uint64_t nr = fi_c_custkey.prefix_sum_nrecs
            ? fi_c_custkey.prefix_sum_nrecs[pg + bnp] - rs : 0;
        q3_bm_custkey.push_back({rs, nr});
    }

    // mktseg batch meta + GPU upload deferred until mktseg_ps_ptr is known
    std::vector<Q3BatchMeta> q3_bm_mktseg;
    uint64_t *d_full_ps_mktseg = nullptr;

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t golap_gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // Pre-allocate tile loop buffers (Rule 4: avoid heap allocation inside timed section)
    size_t max_tile_pages = std::max(Q3_TILE_PAGES_O, Q3_TILE_PAGES);
    std::vector<uint32_t> tile_active_i32;
    tile_active_i32.reserve(max_tile_pages);
    std::vector<uint8_t> h_mask(max_tile_pages);
    std::vector<uint32_t> tile_i64_active;
    tile_i64_active.reserve(std::max(o_npages_i64, l_npages_i64));

    // Pre-allocate results buffer (pinned host memory for fast D2H)
    Q3ResultRow *results = nullptr;
    CUDA_CHECK(cudaMallocHost(&results, aggr_cap * sizeof(Q3ResultRow)));

    // Pre-build C_MKTSEGMENT prefix_sum if not available (Rule 3)
    std::vector<uint64_t> h_mktseg_ps_fallback;
    const uint64_t* mktseg_ps_ptr = fi_c_mktsegment.prefix_sum_nrecs;
    if (mktseg_ps_ptr == nullptr) {
        h_mktseg_ps_fallback.resize(fi_c_mktsegment.npages + 1);
        h_mktseg_ps_fallback[0] = 0;
        uint64_t cum = 0;
        for (size_t pg = 0; pg < fi_c_mktsegment.npages; pg += Q3_TILE_PAGES) {
            size_t bnp = std::min(Q3_TILE_PAGES, fi_c_mktsegment.npages - pg);
            load_field_tile_mt(fi_c_mktsegment, tile_data_buf, pg, bnp);
            for (size_t p = 0; p < bnp; p++) {
                uint32_t nalloc = 0;
                CUDA_CHECK(cudaMemcpy(&nalloc,
                    static_cast<const char *>(tile_data_buf) + p * page_size,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cum += nalloc;
                h_mktseg_ps_fallback[pg + p + 1] = cum;
            }
        }
        mktseg_ps_ptr = h_mktseg_ps_fallback.data();
    }

    // Pre-compute C_MKTSEGMENT batch metadata + upload to GPU (Rule 3: before total_start)
    for (size_t pg = 0; pg < fi_c_mktsegment.npages; pg += Q3_TILE_PAGES) {
        size_t bnp = std::min(Q3_TILE_PAGES, fi_c_mktsegment.npages - pg);
        q3_bm_mktseg.push_back({mktseg_ps_ptr[pg],
            mktseg_ps_ptr[pg + bnp] - mktseg_ps_ptr[pg]});
    }
    CUDA_CHECK(cudaMalloc(&d_full_ps_mktseg, (fi_c_mktsegment.npages + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_full_ps_mktseg, mktseg_ps_ptr,
                          (fi_c_mktsegment.npages + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // ════════════════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    auto phase_start = total_start;
    // ════════════════════════════════════════════════════════

    // ── Zone map IO + eval (GDS + GPU, inside timing) ──
    if (options.enable_zonemap && d_zm_stats_buf[0]) {
        // ORDERS: O_ORDERDATE + sideways C_MKTSEGMENT
        {
            uint32_t npreds = 0;
            int32_t *d_odate_stats = nullptr;
            int32_t *d_mktseg_stats = nullptr;

            if ((!is_q3sel || !options.disable_other_filters) && zm_ord_odate.nstats > 0 && zm_ord_odate.stats_start > 0) {
                gds_read_zonemap(zm_cufile_handles, num_devices,
                                 zm_ord_odate.stats_start, zm_ord_odate.stats_npg,
                                 page_size, d_zm_stats_buf[0]);
                d_odate_stats = reinterpret_cast<int32_t*>(d_zm_stats_buf[0]);
            }

            if (zm_ord_mktseg.nstats > 0 && zm_ord_mktseg.stats_start > 0) {
                int buf_idx = d_odate_stats ? 1 : 0;
                gds_read_zonemap(zm_cufile_handles, num_devices,
                                 zm_ord_mktseg.stats_start, zm_ord_mktseg.stats_npg,
                                 page_size, d_zm_stats_buf[buf_idx]);
                d_mktseg_stats = reinterpret_cast<int32_t*>(d_zm_stats_buf[buf_idx]);
            }

            if (d_odate_stats) {
                zm_ctx.h_preds[npreds].d_stats  = d_odate_stats;
                zm_ctx.h_preds[npreds].nstats   = zm_ord_odate.nstats;
                zm_ctx.h_preds[npreds].pred_lo  = INT32_MIN;
                zm_ctx.h_preds[npreds].pred_hi  = 19950314;
                npreds++;
            }
            if (d_mktseg_stats) {
                zm_ctx.h_preds[npreds].d_stats  = d_mktseg_stats;
                zm_ctx.h_preds[npreds].nstats   = zm_ord_mktseg.nstats;
                if (is_q3sel) {
                    zm_ctx.h_preds[npreds].pred_lo = q3sel_mktseg_lo;
                    zm_ctx.h_preds[npreds].pred_hi = q3sel_mktseg_hi;
                } else {
                    zm_ctx.h_preds[npreds].pred_lo = 1;
                    zm_ctx.h_preds[npreds].pred_hi = 1;
                }
                npreds++;
            }

            zm_ctx.d_ps_i32   = d_full_ps_gpu[fi_o_orderdate.prefix_sum_nrecs] + 1;
            zm_ctx.d_ps_i64   = d_full_ps_gpu[fi_o_orderkey.prefix_sum_nrecs] + 1;
            zm_ctx.d_mask_i64 = d_mask_ord_i64;
            zm_ctx.npages_i64 = (uint32_t)o_npages_i64;

            if (npreds > 0) {
                gds_zonemap_eval_async(zm_ctx, o_npages_i32, npreds, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (uint32_t pg = 0; pg < o_npages_i32; pg++)
                    ord_page_active[pg] = zm_ctx.h_mask[pg];
                CUDA_CHECK(cudaMemcpy(h_mask_ord_i64.data(), d_mask_ord_i64,
                                      o_npages_i64, cudaMemcpyDeviceToHost));
            }

            uint32_t active_count = 0;
            for (size_t i = 0; i < o_npages_i32; i++) if (ord_page_active[i]) active_count++;
            if (debug_print) std::cout << "[Q3-ZONEMAP] ORDERS GPU pruning: active="
                      << active_count << "/" << o_npages_i32 << std::endl;
        }

        // LINEITEM: L_SHIPDATE + sw O_ORDERDATE + sw C_MKTSEGMENT
        {
            uint32_t npreds = 0;
            int buf_idx = 0;
            int32_t *d_lsd_stats = nullptr;
            int32_t *d_sw_odate_stats = nullptr;
            int32_t *d_sw_mktseg_stats = nullptr;

            if ((!is_q3sel || !options.disable_other_filters) && zm_li_lsd.nstats > 0 && zm_li_lsd.stats_start > 0) {
                gds_read_zonemap(zm_cufile_handles, num_devices,
                                 zm_li_lsd.stats_start, zm_li_lsd.stats_npg,
                                 page_size, d_zm_stats_buf[buf_idx]);
                d_lsd_stats = reinterpret_cast<int32_t*>(d_zm_stats_buf[buf_idx]);
                buf_idx++;
            }

            if ((!is_q3sel || !options.disable_other_filters) && zm_li_sw_odate.nstats > 0 && zm_li_sw_odate.stats_start > 0) {
                gds_read_zonemap(zm_cufile_handles, num_devices,
                                 zm_li_sw_odate.stats_start, zm_li_sw_odate.stats_npg,
                                 page_size, d_zm_stats_buf[buf_idx]);
                d_sw_odate_stats = reinterpret_cast<int32_t*>(d_zm_stats_buf[buf_idx]);
                buf_idx++;
            }

            if (zm_li_sw_mktseg.nstats > 0 && zm_li_sw_mktseg.stats_start > 0) {
                gds_read_zonemap(zm_cufile_handles, num_devices,
                                 zm_li_sw_mktseg.stats_start, zm_li_sw_mktseg.stats_npg,
                                 page_size, d_zm_stats_buf[buf_idx]);
                d_sw_mktseg_stats = reinterpret_cast<int32_t*>(d_zm_stats_buf[buf_idx]);
            }

            if (d_lsd_stats) {
                zm_ctx.h_preds[npreds].d_stats  = d_lsd_stats;
                zm_ctx.h_preds[npreds].nstats   = zm_li_lsd.nstats;
                zm_ctx.h_preds[npreds].pred_lo  = 19950316;
                zm_ctx.h_preds[npreds].pred_hi  = INT32_MAX;
                npreds++;
            }
            if (d_sw_odate_stats) {
                zm_ctx.h_preds[npreds].d_stats  = d_sw_odate_stats;
                zm_ctx.h_preds[npreds].nstats   = zm_li_sw_odate.nstats;
                zm_ctx.h_preds[npreds].pred_lo  = INT32_MIN;
                zm_ctx.h_preds[npreds].pred_hi  = 19950314;
                npreds++;
            }
            if (d_sw_mktseg_stats) {
                zm_ctx.h_preds[npreds].d_stats  = d_sw_mktseg_stats;
                zm_ctx.h_preds[npreds].nstats   = zm_li_sw_mktseg.nstats;
                if (is_q3sel) {
                    zm_ctx.h_preds[npreds].pred_lo = q3sel_mktseg_lo;
                    zm_ctx.h_preds[npreds].pred_hi = q3sel_mktseg_hi;
                } else {
                    zm_ctx.h_preds[npreds].pred_lo = 1;
                    zm_ctx.h_preds[npreds].pred_hi = 1;
                }
                npreds++;
            }

            zm_ctx.d_ps_i32   = d_full_ps_gpu[fi_l_shipdate.prefix_sum_nrecs] + 1;
            zm_ctx.d_ps_i64   = d_full_ps_gpu[fi_l_orderkey.prefix_sum_nrecs] + 1;
            zm_ctx.d_mask_i64 = d_mask_li_i64;
            zm_ctx.npages_i64 = (uint32_t)l_npages_i64;

            if (npreds > 0) {
                gds_zonemap_eval_async(zm_ctx, l_npages_i32, npreds, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (uint32_t pg = 0; pg < l_npages_i32; pg++)
                    li_page_active[pg] = zm_ctx.h_mask[pg];
                CUDA_CHECK(cudaMemcpy(h_mask_li_i64.data(), d_mask_li_i64,
                                      l_npages_i64, cudaMemcpyDeviceToHost));
            }

            uint32_t li_active_count = 0;
            for (size_t i = 0; i < l_npages_i32; i++) if (li_page_active[i]) li_active_count++;
            if (debug_print) std::cout << "[Q3-ZONEMAP] LINEITEM GPU pruning: active="
                      << li_active_count << "/" << l_npages_i32 << std::endl;
        }

        zm_ctx.d_ps_i32 = nullptr;
        zm_ctx.d_ps_i64 = nullptr;
        zm_ctx.d_mask_i64 = nullptr;
    }

    // ── Phase 1: CUSTOMER scan (batch via tile_data_buf) ──
    // Step 1a: Batch flatten C_CUSTKEY → d_c_custkey_flat
    if (debug_print) std::cout << "[Q3] Loading C_CUSTKEY (" << fi_c_custkey.npages << " pages)..." << std::endl;
    {
        size_t bi = 0;
        for (size_t pg = 0; pg < fi_c_custkey.npages; pg += Q3_TILE_PAGES, bi++) {
            size_t bnp = std::min(Q3_TILE_PAGES, fi_c_custkey.npages - pg);
            load_field_tile_mt(fi_c_custkey, tile_data_buf, pg, bnp);

            auto &bm = q3_bm_custkey[bi];
            flatten_int64_field_tile(fi_c_custkey, tile_data_buf, bm.batch_nrecs,
                d_c_custkey_flat + bm.row_start, pg, bnp);
        }
    }

    // Step 1b: Batch scan C_MKTSEGMENT → build CUSTOMER hash set
    if (debug_print) std::cout << "[Q3] Loading C_MKTSEGMENT (" << fi_c_mktsegment.npages << " pages)..." << std::endl;
    CUDA_CHECK(cudaMemsetAsync(d_custkey_set, 0xFF, custset_cap * sizeof(uint64_t), stream));
    constexpr uint32_t CHAR_MKTSEG_PADDED_LEN = 12;

    // Q3SEL segment values (8-byte LE): ordered by cumulative selectivity
    static constexpr uint64_t Q3SEL_SEGMENTS[5] = {
        0x474E49444C495542ULL, // BUILDING
        0x49424F4D4F545541ULL, // AUTOMOBILE
        0x525554494E525546ULL, // FURNITURE
        0x52454E494843414DULL, // MACHINERY
        0x4C4F484553554F48ULL, // HOUSEHOLD
    };
    uint32_t q3sel_num_seg = 0;
    if (is_q3sel) {
        if (sel_pct >= 100) q3sel_num_seg = 0; // 0 = all pass
        else q3sel_num_seg = (uint32_t)(sel_pct / 20);
        if (q3sel_num_seg == 0 && sel_pct > 0 && sel_pct < 100) q3sel_num_seg = 1;
    }

    {
        size_t bi = 0;
        for (size_t pg = 0; pg < fi_c_mktsegment.npages; pg += Q3_TILE_PAGES, bi++) {
            size_t bnp = std::min(Q3_TILE_PAGES, fi_c_mktsegment.npages - pg);
            load_field_tile_mt(fi_c_mktsegment, tile_data_buf, pg, bnp);

            auto &bm = q3_bm_mktseg[bi];
            compute_batch_ps_kernel<<<((uint32_t)bnp + 255) / 256, 256, 0, stream>>>(
                d_full_ps_mktseg, d_ps_shared, (uint32_t)pg, (uint32_t)bnp);
            s_kernel_launches++;

            if (is_q3sel) {
                q3sel_customer_scan(
                    static_cast<const char *>(tile_data_buf),
                    d_ps_shared, (uint32_t)bnp,
                    page_size, CHAR_MKTSEG_PADDED_LEN,
                    d_c_custkey_flat + bm.row_start, bm.batch_nrecs,
                    d_custkey_set, custset_mask,
                    q3sel_num_seg, Q3SEL_SEGMENTS, stream);
            } else {
                q3_customer_scan(
                    static_cast<const char *>(tile_data_buf),
                    d_ps_shared, (uint32_t)bnp,
                    page_size, CHAR_MKTSEG_PADDED_LEN,
                    d_c_custkey_flat + bm.row_start, bm.batch_nrecs,
                    d_custkey_set, custset_mask, stream);
            }
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    if (debug_print) std::cout << "[Q3] CUSTOMER hash set built (capacity=" << custset_cap << ")" << std::endl;

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q3-TIMING] Phase 1 (CUSTOMER): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ── Phase 2: ORDERS probe + build (tile execution) ──
    CUDA_CHECK(cudaMemsetAsync(d_orders_ht_keys, 0xFF, orders_ht_cap * sizeof(uint64_t), stream));

    for (size_t tile_idx = 0; tile_idx < o_num_tiles; tile_idx++) {
        size_t p_lo = tile_idx * Q3_TILE_PAGES_O;
        size_t tile_np = std::min(Q3_TILE_PAGES_O, o_npages_i32 - p_lo);
        uint64_t tile_nrows = ps_o_i32[p_lo + tile_np] - ps_o_i32[p_lo];
        if (tile_nrows == 0) continue;

        // Check tile for active pages
        bool any_active = !options.enable_zonemap;
        tile_active_i32.clear();
        for (size_t j = 0; j < tile_np; j++) {
            if (ord_page_active[p_lo + j]) {
                any_active = true;
                tile_active_i32.push_back((uint32_t)(p_lo + j));
            }
        }
        if (!any_active) continue;

        bool use_selective = options.enable_zonemap && tile_active_i32.size() < tile_np;

        // Upload tile mask (shared across INT32 columns in this tile)
        if (use_selective) {
            for (size_t j = 0; j < tile_np; j++)
                h_mask[j] = ord_page_active[p_lo + j] ? 1 : 0;
            CUDA_CHECK(cudaMemcpy(d_tile_page_mask, h_mask.data(), tile_np, cudaMemcpyHostToDevice));
        }

        // ── INT32: O_ORDERDATE (fill=INT32_MAX → fails 'date < 19950315'; q3sel: fill=0) ──
        CUDA_CHECK(cudaMemset(tile_data_buf, 0, tile_np * page_size));
        if (use_selective) {
            load_field_selective_tile_mt(fi_o_orderdate, tile_data_buf, tile_active_i32, p_lo, "O_ORDERDATE");
            uint64_t odate_fill = is_q3sel ? 0 : (uint64_t)(uint32_t)INT32_MAX;
            flatten_int32_field_tile_masked(fi_o_orderdate, tile_data_buf, tile_nrows,
                d_o_orderdate_flat, p_lo, tile_np, d_tile_page_mask, odate_fill);
        } else {
            load_field_tile_mt(fi_o_orderdate, tile_data_buf, p_lo, tile_np);
            flatten_int32_field_tile(fi_o_orderdate, tile_data_buf, tile_nrows,
                d_o_orderdate_flat, p_lo, tile_np);
        }

        // ── INT32: O_SHIPPRIORITY (fill=0, filtered by O_ORDERDATE) ──
        CUDA_CHECK(cudaMemset(tile_data_buf, 0, tile_np * page_size));
        if (use_selective) {
            load_field_selective_tile_mt(fi_o_shippriority, tile_data_buf, tile_active_i32, p_lo, "O_SHIPPRIORITY");
            flatten_int32_field_tile_masked(fi_o_shippriority, tile_data_buf, tile_nrows,
                d_o_shippriority_flat, p_lo, tile_np, d_tile_page_mask, 0);
        } else {
            load_field_tile_mt(fi_o_shippriority, tile_data_buf, p_lo, tile_np);
            flatten_int32_field_tile(fi_o_shippriority, tile_data_buf, tile_nrows,
                d_o_shippriority_flat, p_lo, tile_np);
        }

        // ── INT64 page range for this tile ──
        uint64_t first_row = ps_o_i32[p_lo];
        uint64_t last_row = ps_o_i32[p_lo + tile_np];
        auto it_s = std::upper_bound(ps_o_i64.begin(), ps_o_i64.end(), first_row);
        size_t i64_start = (it_s == ps_o_i64.begin()) ? 0 : (size_t)(it_s - ps_o_i64.begin()) - 1;
        auto it_e = std::upper_bound(ps_o_i64.begin(), ps_o_i64.end(), last_row - 1);
        size_t i64_end = (size_t)(it_e - ps_o_i64.begin());
        size_t i64_np = i64_end - i64_start;
        uint64_t i64_nrows = ps_o_i64[i64_end] - ps_o_i64[i64_start];
        uint64_t i64_offset = first_row - ps_o_i64[i64_start];

        // Compute needed INT64 pages for this tile (from GPU-derived mask)
        tile_i64_active.clear();
        if (options.enable_zonemap) {
            for (size_t pg = i64_start; pg < i64_end; pg++)
                if (h_mask_ord_i64[pg]) tile_i64_active.push_back((uint32_t)pg);
        }

        // ── INT64: O_ORDERKEY ──
        CUDA_CHECK(cudaMemset(tile_data_buf, 0, i64_np * page_size));
        if (options.enable_zonemap) {
            load_field_selective_tile_mt(fi_o_orderkey, tile_data_buf, tile_i64_active, i64_start, "O_ORDERKEY");
        } else {
            load_field_tile_mt(fi_o_orderkey, tile_data_buf, i64_start, i64_np);
        }
        flatten_int64_field_tile(fi_o_orderkey, tile_data_buf, i64_nrows,
            d_o_orderkey_flat, i64_start, i64_np);

        // ── INT64: O_CUSTKEY ──
        CUDA_CHECK(cudaMemset(tile_data_buf, 0, i64_np * page_size));
        if (options.enable_zonemap) {
            load_field_selective_tile_mt(fi_o_custkey, tile_data_buf, tile_i64_active, i64_start, "O_CUSTKEY");
        } else {
            load_field_tile_mt(fi_o_custkey, tile_data_buf, i64_start, i64_np);
        }
        flatten_int64_field_tile(fi_o_custkey, tile_data_buf, i64_nrows,
            d_o_custkey_flat, i64_start, i64_np);

        // ── Kernel: flat probe + build ──
        if (is_q3sel) {
            q3sel_orders_probe_build(
                d_o_custkey_flat + i64_offset, d_o_orderdate_flat,
                d_o_orderkey_flat + i64_offset, d_o_shippriority_flat,
                tile_nrows,
                d_custkey_set, custset_mask,
                d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                options.disable_other_filters ? 0 : 19950315, stream);
        } else {
            q3_orders_probe_build(
                d_o_custkey_flat + i64_offset, d_o_orderdate_flat,
                d_o_orderkey_flat + i64_offset, d_o_shippriority_flat,
                tile_nrows,
                d_custkey_set, custset_mask,
                d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask, stream);
        }
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (debug_print) std::cout << "[Q3] ORDERS HT built (capacity=" << orders_ht_cap << ")" << std::endl;

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q3-TIMING] Phase 2 (ORDERS): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ── Phase 3: LINEITEM probe + aggregation (tile execution) ──
    CUDA_CHECK(cudaMemsetAsync(d_aggr_keys, 0xFF, aggr_cap * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_aggr_revenues, 0, aggr_cap * sizeof(int64_t), stream));

    for (size_t tile_idx = 0; tile_idx < l_num_tiles; tile_idx++) {
        size_t p_lo = tile_idx * Q3_TILE_PAGES;
        size_t tile_np = std::min(Q3_TILE_PAGES, l_npages_i32 - p_lo);
        uint64_t tile_nrows = ps_l_i32[p_lo + tile_np] - ps_l_i32[p_lo];
        if (tile_nrows == 0) continue;

        // Check tile for active pages
        bool any_active = !options.enable_zonemap;
        tile_active_i32.clear();
        for (size_t j = 0; j < tile_np; j++) {
            if (li_page_active[p_lo + j]) {
                any_active = true;
                tile_active_i32.push_back((uint32_t)(p_lo + j));
            }
        }
        if (!any_active) continue;

        bool use_selective = options.enable_zonemap && tile_active_i32.size() < tile_np;

        // Upload tile mask (shared across INT32 columns in this tile)
        if (use_selective) {
            for (size_t j = 0; j < tile_np; j++)
                h_mask[j] = li_page_active[p_lo + j] ? 1 : 0;
            CUDA_CHECK(cudaMemcpy(d_tile_page_mask, h_mask.data(), tile_np, cudaMemcpyHostToDevice));
        }

        // ── INT32: L_SHIPDATE (fill=0 → fails 'shipdate > 19950315') ──
        if (!is_q3sel || !options.disable_other_filters) {
            CUDA_CHECK(cudaMemset(tile_data_buf, 0, tile_np * page_size));
            if (use_selective) {
                load_field_selective_tile_mt(fi_l_shipdate, tile_data_buf, tile_active_i32, p_lo, "L_SHIPDATE");
                flatten_int32_field_tile_masked(fi_l_shipdate, tile_data_buf, tile_nrows,
                    d_l_shipdate_flat, p_lo, tile_np, d_tile_page_mask, 0);
            } else {
                load_field_tile_mt(fi_l_shipdate, tile_data_buf, p_lo, tile_np);
                flatten_int32_field_tile(fi_l_shipdate, tile_data_buf, tile_nrows,
                    d_l_shipdate_flat, p_lo, tile_np);
            }
        }

        // ── INT32: L_EXTENDEDPRICE (fill=0, filtered by L_SHIPDATE) ──
        CUDA_CHECK(cudaMemset(tile_data_buf, 0, tile_np * page_size));
        if (use_selective) {
            load_field_selective_tile_mt(fi_l_extprice, tile_data_buf, tile_active_i32, p_lo, "L_EXTENDEDPRICE");
            flatten_int32_field_tile_masked(fi_l_extprice, tile_data_buf, tile_nrows,
                d_l_extprice_flat, p_lo, tile_np, d_tile_page_mask, 0);
        } else {
            load_field_tile_mt(fi_l_extprice, tile_data_buf, p_lo, tile_np);
            flatten_int32_field_tile(fi_l_extprice, tile_data_buf, tile_nrows,
                d_l_extprice_flat, p_lo, tile_np);
        }

        // ── INT32: L_DISCOUNT (fill=0, filtered by L_SHIPDATE) ──
        CUDA_CHECK(cudaMemset(tile_data_buf, 0, tile_np * page_size));
        if (use_selective) {
            load_field_selective_tile_mt(fi_l_discount, tile_data_buf, tile_active_i32, p_lo, "L_DISCOUNT");
            flatten_int32_field_tile_masked(fi_l_discount, tile_data_buf, tile_nrows,
                d_l_discount_flat, p_lo, tile_np, d_tile_page_mask, 0);
        } else {
            load_field_tile_mt(fi_l_discount, tile_data_buf, p_lo, tile_np);
            flatten_int32_field_tile(fi_l_discount, tile_data_buf, tile_nrows,
                d_l_discount_flat, p_lo, tile_np);
        }

        // ── INT64 page range for this tile ──
        uint64_t first_row = ps_l_i32[p_lo];
        uint64_t last_row = ps_l_i32[p_lo + tile_np];
        auto it_s = std::upper_bound(ps_l_i64.begin(), ps_l_i64.end(), first_row);
        size_t i64_start = (it_s == ps_l_i64.begin()) ? 0 : (size_t)(it_s - ps_l_i64.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64.begin(), ps_l_i64.end(), last_row - 1);
        size_t i64_end = (size_t)(it_e - ps_l_i64.begin());
        size_t i64_np = i64_end - i64_start;
        uint64_t i64_nrows = ps_l_i64[i64_end] - ps_l_i64[i64_start];
        uint64_t i64_offset = first_row - ps_l_i64[i64_start];

        // ── INT64: L_ORDERKEY ──
        CUDA_CHECK(cudaMemset(tile_data_buf, 0, i64_np * page_size));
        if (options.enable_zonemap) {
            tile_i64_active.clear();
            for (size_t pg = i64_start; pg < i64_end; pg++)
                if (h_mask_li_i64[pg]) tile_i64_active.push_back((uint32_t)pg);
            load_field_selective_tile_mt(fi_l_orderkey, tile_data_buf, tile_i64_active, i64_start, "L_ORDERKEY");
        } else {
            load_field_tile_mt(fi_l_orderkey, tile_data_buf, i64_start, i64_np);
        }
        flatten_int64_field_tile(fi_l_orderkey, tile_data_buf, i64_nrows,
            d_l_orderkey_flat, i64_start, i64_np);

        // ── Kernel: flat probe + aggregate ──
        if (is_q3sel) {
            q3sel_lineitem_probe_aggr(
                d_l_orderkey_flat + i64_offset,
                d_l_extprice_flat, d_l_discount_flat, d_l_shipdate_flat,
                tile_nrows,
                d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                d_aggr_keys, d_aggr_revenues, aggr_mask,
                options.disable_other_filters ? 0 : 19950315, stream);
        } else {
            q3_lineitem_probe_aggr(
                d_l_orderkey_flat + i64_offset, d_l_shipdate_flat,
                d_l_extprice_flat, d_l_discount_flat,
                tile_nrows,
                d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                d_aggr_keys, d_aggr_revenues, aggr_mask, stream);
        }
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q3-TIMING] Phase 3 (LINEITEM): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ── Phase 4: Collect + GPU Sort + Top-10 ──
    CUDA_CHECK(cudaMemsetAsync(d_result_count, 0, sizeof(uint32_t), stream));

    q3_collect_results(d_aggr_keys, d_aggr_revenues, aggr_cap,
                       d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                       d_results, d_result_count, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto phase4_collect_end = std::chrono::steady_clock::now();

    uint32_t h_result_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (debug_print) std::cout << "[Q3] Aggregation groups: " << h_result_count << std::endl;

    // Sort on GPU: ORDER BY revenue DESC, o_orderdate ASC
    // Use CUB DeviceMergeSort with pre-allocated temp buffer (Rule 4: no hidden cudaMalloc)
    cub::DeviceMergeSort::SortKeys(d_sort_temp, sort_temp_bytes,
        d_results, (int)h_result_count, Q3ResultCmp{}, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto phase4_sort_end = std::chrono::steady_clock::now();

    // D2H: all results to pinned host memory
    CUDA_CHECK(cudaMemcpy(results, d_results,
                          h_result_count * sizeof(Q3ResultRow), cudaMemcpyDeviceToHost));

    // ════════════════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    // ════════════════════════════════════════════════════════

    {
        double ms_collect = std::chrono::duration<double, std::milli>(phase4_collect_end - phase_start).count();
        double ms_sort = std::chrono::duration<double, std::milli>(phase4_sort_end - phase4_collect_end).count();
        double ms_d2h = std::chrono::duration<double, std::milli>(total_end - phase4_sort_end).count();
        double ms_total = std::chrono::duration<double, std::milli>(total_end - phase_start).count();
        std::cout << "[Q3-TIMING] Phase 4 (COLLECT+SORT): " << ms_total << " ms"
                  << "  [collect=" << ms_collect
                  << " gpu_sort=" << ms_sort
                  << " d2h=" << ms_d2h << "]" << std::endl;
    }

    // Print IO coalescing stats
    if (!coalesce_stats.empty()) {
        std::cout << "\n=== IO Coalescing ===" << std::endl;
        size_t total_pages = 0, total_ios_sel = 0;
        for (auto &c : coalesce_stats) {
            double ratio = (double)c.pages / (double)c.ios;
            std::cout << "  " << c.field_name
                      << ": pages=" << c.pages
                      << " ios=" << c.ios
                      << " ratio=" << std::fixed << std::setprecision(2) << ratio
                      << std::endl;
            total_pages += c.pages;
            total_ios_sel += c.ios;
        }
        if (total_ios_sel > 0) {
            double total_ratio = (double)total_pages / (double)total_ios_sel;
            std::cout << "  TOTAL: pages=" << total_pages
                      << " ios=" << total_ios_sel
                      << " ratio=" << std::fixed << std::setprecision(2) << total_ratio
                      << std::endl;
        }
    }

    // Print top 10
    if (is_q3sel)
        std::cout << "\n=== TPC-H Q3SEL Result (sel=" << sel_pct << "%, Top 10) ===" << std::endl;
    else
        std::cout << "\n=== TPC-H Q3 Result (Top 10) ===" << std::endl;
    std::cout << "l_orderkey |       revenue | o_orderdate | o_shippriority" << std::endl;
    std::cout << "-----------+---------------+-------------+---------------" << std::endl;
    uint32_t limit = std::min(h_result_count, (uint32_t)10);
    for (uint32_t i = 0; i < limit; i++) {
        auto &r = results[i];
        printf("%10lu | %13ld | %11u | %14u\n",
               (unsigned long)r.l_orderkey, (long)r.revenue,
               r.o_orderdate, r.o_shippriority);
    }
    std::cout << std::endl;

    // ── Free all GPU memory ──
    // Phase 1 (CUSTOMER)
    cudaFree(d_c_custkey_flat);
    cudaFree(d_custkey_set);
    // Shared staging buffer
    mb_cuda_free(tile_data_buf);
    // Phase 2 (ORDERS) flat arrays
    cudaFree(d_o_orderkey_flat);
    cudaFree(d_o_custkey_flat);
    cudaFree(d_o_orderdate_flat);
    cudaFree(d_o_shippriority_flat);
    // Phase 3 (LINEITEM) flat arrays
    cudaFree(d_l_orderkey_flat);
    cudaFree(d_l_extprice_flat);
    cudaFree(d_l_discount_flat);
    cudaFree(d_l_shipdate_flat);
    // Hash tables
    cudaFree(d_orders_ht_keys);
    cudaFree(d_orders_ht_payloads);
    cudaFree(d_aggr_keys);
    cudaFree(d_aggr_revenues);
    // Zone map
    if (zm_handles_open) {
        if (d_zm_stats_base) { cuFileBufDeregister(d_zm_stats_base); cudaFree(d_zm_stats_base); }
        gds_zonemap_ctx_destroy(zm_ctx);
        for (size_t d = 0; d < num_devices; d++) {
            cuFileHandleDeregister(zm_cufile_handles[d]);
            close(zm_dup_fds[d]);
        }
    }
    if (d_tile_page_mask) cudaFree(d_tile_page_mask);
    if (d_mask_ord_i64) cudaFree(d_mask_ord_i64);
    if (d_mask_li_i64) cudaFree(d_mask_li_i64);
    // Remaining
    cudaFree(d_ps_shared);
    for (auto &[k, v] : d_full_ps_gpu) cudaFree(v);
    cudaFree(d_full_ps_mktseg);
    cudaFree(d_results);
    cudaFree(d_result_count);
    cudaFree(d_sort_temp);
    cudaFreeHost(results);
    cudaStreamDestroy(stream);

    size_t total_ios = 0, total_bytes_read = 0;
    for (size_t t = 0; t < nthreads; t++) {
        auto &L = loaders[t];
        total_ios += L.ios_completed;
        total_bytes_read += L.bytes_read;
        s_kernel_launches += L.kernel_launches;
        cuFileBufDeregister(L.io_buf);
        mb_cuda_free(L.io_buf);
        cudaStreamDestroy(L.stream);
        if (any_compressed) nvcomp_decompctx_free(L.nvctx);
        for (size_t d = 0; d < num_devices; d++) {
            for (size_t h = 0; h < L.cufile_handles[d].size(); h++) {
                cuFileHandleDeregister(L.cufile_handles[d][h]);
                close(L.dup_fds[d][h]);
            }
        }
    }

    size_t total_pages = 0;
    for (const auto &fi : customer_fields) total_pages += fi.npages;
    for (const auto &fi : orders_fields) total_pages += fi.npages;
    for (const auto &fi : lineitem_fields) total_pages += fi.npages;

    std::string comp_str = collect_compression_methods(
        {customer_fields, orders_fields, lineitem_fields});
    free_fields_metadata(customer_fields);
    free_fields_metadata(orders_fields);
    free_fields_metadata(lineitem_fields);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = total_ios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = comp_str,
        .gpu_mem_bytes = golap_gpu_mem_bytes,
        .gpu_app_bytes = golap_gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

BenchmarkResult tpch_q3sel(BenchmarkOptions &options) {
    return tpch_q3(options);
}

// ============================================================
// TPC-H Q1 — GOLAP sync mode
// ============================================================

BenchmarkResult tpch_q1(BenchmarkOptions &options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    CUcontext cuda_ctx_handle;
    cuCtxGetCurrent(&cuda_ctx_handle);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    TPCHTableMetadata *metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    {
        const size_t ps = metadatap->page_size;
        std::cout << "=== TPCH Q1 ===" << std::endl;
        std::cout << "Page Size: " << ps << std::endl;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, ps) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, ps);
    }

    metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    TPCHTableMetadata &metadata = *metadatap;
    superpage_set_constants(metadata.page_size);
    const size_t page_size = metadata.page_size;
    const size_t num_devices = fds.size();

    const uint64_t nrecs_lineitem = metadata.table_lineitem_nrows;
    std::cout << "nrecs_lineitem: " << nrecs_lineitem
              << ", num_devices: " << num_devices << std::endl;

    // ── Phase 0: Load metadata for LINEITEM (7 columns) ──
    constexpr size_t num_lineitem_cols = TPCH::Query::Q1::NUM_SCAN_TARGET_COLS;
    std::vector<FieldPageInfo> lineitem_fields(num_lineitem_cols);

    uint32_t saved_column = metadata.column;
    metadata.column = TPCH::common::Table::LINEITEM;
    prepare_fields_metadata(fds, metadata, page_size,
        TPCH::Query::Q1::SCAN_TARGET_COLS, lineitem_fields);
    metadata.column = saved_column;

    // Column references (order matches SCAN_TARGET_COLS)
    const FieldPageInfo &fi_l_quantity      = lineitem_fields[0];
    const FieldPageInfo &fi_l_extendedprice = lineitem_fields[1];
    const FieldPageInfo &fi_l_discount      = lineitem_fields[2];
    const FieldPageInfo &fi_l_tax           = lineitem_fields[3];
    const FieldPageInfo &fi_l_returnflag    = lineitem_fields[4];
    const FieldPageInfo &fi_l_linestatus    = lineitem_fields[5];
    const FieldPageInfo &fi_l_shipdate      = lineitem_fields[6];

    for (size_t i = 0; i < num_lineitem_cols; i++) {
        std::cout << "  field[" << i << "]: start_page=" << lineitem_fields[i].start_page_id
                  << " npages=" << lineitem_fields[i].npages
                  << " compression=" << compression_method_name(lineitem_fields[i].compression_method)
                  << std::endl;
    }

    // Verify all 7 columns have the same npages
    const size_t npages_ref = lineitem_fields[0].npages;
    for (size_t i = 1; i < num_lineitem_cols; i++) {
        if (lineitem_fields[i].npages != npages_ref) {
            std::cerr << "ERROR: field[" << i << "].npages=" << lineitem_fields[i].npages
                      << " != field[0].npages=" << npages_ref << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // ── Check for compressed fields ──
    bool any_compressed = false;
    for (auto &fi : lineitem_fields) {
        if (fi.compression_method != CompressionMethod::NONE) any_compressed = true;
    }

    // ── Zone map IO pruning: pre-allocate outside timing (Rule 4) ──
    std::vector<uint32_t> active_pages;
    bool use_selective = false;
    bool zm_has_stats = false;
    uint64_t zm_nstats = 0, zm_stats_start = 0, zm_stats_npg = 0;
    uint32_t zm_min_npages_li = npages_ref;
    int32_t *d_zm_sd_stats = nullptr;
    GdsZonemapCtx zm_ctx{};
    CUfileHandle_t zm_cufile_handles[MAX_GDS_DEVICES];
    int zm_dup_fds[MAX_GDS_DEVICES];
    bool zm_handles_open = false;

    if (options.enable_zonemap) {
        size_t sd_col = TPCH::Query::Q1::SCAN_TARGET_COLS[6];
        zm_nstats = metadata.table_lineitem_nstats[sd_col];
        zm_stats_start = metadata.table_lineitem_stats_start_page_ids[sd_col];
        zm_stats_npg = metadata.table_lineitem_stats_npages[sd_col];
        zm_min_npages_li = fi_l_shipdate.npages;

        if (zm_nstats == 0 || zm_stats_start == 0) {
            std::cout << "[ZONEMAP] No stats available for L_SHIPDATE, processing all pages." << std::endl;
        } else {
            zm_has_stats = true;
        }
    }

    // ── GDS setup (per-column-group loader resources) ──
    // Each column gets io_thr_per_col dedicated loaders (io_buf + stream + nvCOMP ctx).
    // Column-parallel loading: all columns loaded simultaneously.
    // async_final=true skips nvCOMP sync on the last batch so decomp
    // overlaps naturally with other threads' cuFileRead.
    const size_t nthreads = options.nthreads;
    constexpr size_t Q1_MAX_IO_THR_PER_COL = 2;
    const size_t io_thr_per_col = std::min(nthreads, Q1_MAX_IO_THR_PER_COL);
    const size_t num_loaders = io_thr_per_col * num_lineitem_cols;
    std::cout << "[Q1] nthreads=" << nthreads
              << " io_thr_per_col=" << io_thr_per_col
              << " num_loaders=" << num_loaders << std::endl;
    mb_cufile_driver_open();

    if (zm_has_stats) {
        CUDA_CHECK(cudaMalloc(&d_zm_sd_stats, zm_stats_npg * page_size));
        GDS_CHECK(cuFileBufRegister(d_zm_sd_stats, zm_stats_npg * page_size, 0));
        zm_ctx = gds_zonemap_ctx_create(zm_min_npages_li);
        for (size_t d = 0; d < num_devices; d++) {
            zm_dup_fds[d] = dup(fds[d]);
            zm_cufile_handles[d] = mb_cufile_handle_register(zm_dup_fds[d]);
        }
        zm_handles_open = true;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Double-buffered I/O-decomp pipeline: 2 large set_bufs.
    // Each set_buf holds nvcomp_batch_pages of data, filled by
    // multiple io_sub_batch cuFileRead calls.  While nvCOMP reads
    // from set A, cuFileRead fills set B.
    constexpr size_t Q1_IO_SUB_BATCH_PAGES = 32;       // I/O granularity (pages per cuFileRead round)
    constexpr size_t Q1_NVCOMP_BATCH_PAGES = 1024;     // nvCOMP decomp batch size

    struct Q1LoaderCtx {
        std::vector<std::vector<CUfileHandle_t>> cufile_handles;
        std::vector<std::vector<int>> dup_fds;
        void *set_buf[2] = {};               // 2 registered bufs, each nvcomp_batch * page_size
        cudaStream_t io_stream = nullptr;     // for D2D scatter
        cudaStream_t nvcomp_stream = nullptr; // for nvCOMP
        cudaEvent_t io_done_event;            // io→nvcomp sync
        cudaEvent_t set_done_events[2];       // per-set nvCOMP completion
        NvcompDecompCtx nvctx;
        size_t bytes_read = 0;
        size_t ios_completed = 0;
        size_t kernel_launches = 0;
    };
    std::vector<Q1LoaderCtx> loaders(num_loaders);

    // Phase 1: create CUfileHandles, streams, events (set_buf/nvcomp deferred for budget)
    for (size_t t = 0; t < num_loaders; t++) {
        auto &L = loaders[t];
        L.cufile_handles.resize(num_devices);
        L.dup_fds.resize(num_devices);
        for (size_t d = 0; d < num_devices; d++) {
            int duped = dup(fds[d]);
            if (duped < 0) { std::cerr << "dup failed" << std::endl; exit(EXIT_FAILURE); }
            L.dup_fds[d].push_back(duped);
            L.cufile_handles[d].push_back(mb_cufile_handle_register(duped));
        }
        CUDA_CHECK(cudaStreamCreate(&L.io_stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&L.io_done_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&L.set_done_events[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&L.set_done_events[1], cudaEventDisableTiming));
        if (any_compressed) {
            CUDA_CHECK(cudaStreamCreate(&L.nvcomp_stream));
        }
    }

    // Aggregate array (allocated outside tile loop — accumulates across tiles)
    constexpr size_t agg_size = Q1_NUM_GROUPS * Q1_NUM_AGGS * sizeof(int64_t);
    int64_t *d_agg = nullptr;
    CUDA_CHECK(cudaMalloc(&d_agg, agg_size));
    CUDA_CHECK(cudaMemset(d_agg, 0, agg_size));

    size_t total_bytes_read = 0, total_ios = 0;

    // ── Column index constants (order matches SCAN_TARGET_COLS) ──
    enum { L_QTY_IDX = 0, L_EPRICE_IDX, L_DISC_IDX, L_TAX_IDX,
           L_RFLAG_IDX, L_LSTATUS_IDX, L_SDATE_IDX };

    // ── Allocate per-column data buffers ──
    constexpr size_t Q1_TILE_PAGES = 512;  // Fixed batch size for streaming
    void *tile_data_buf[num_lineitem_cols];
    for (size_t i = 0; i < num_lineitem_cols; i++)
        tile_data_buf[i] = mb_cuda_alloc(Q1_TILE_PAGES * page_size);

    const uint64_t *ref_ps = fi_l_shipdate.prefix_sum_nrecs;
    const uint32_t capacity = (page_size - 12) / 4;  // (page_size - sizeof(pag_head)) / sizeof(int32_t)

    // Phase 2: allocate per-loader set_buf with dynamic budget
    size_t q1_nvcomp_batch = Q1_NVCOMP_BATCH_PAGES;
    {
        static constexpr size_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_before_loaders = 0;
        cudaMemGetInfo(&gpu_free_before_loaders, &gpu_total_dummy);
        size_t non_loader_bytes = gpu_free_start - gpu_free_before_loaders;
        size_t min_batch = Q1_IO_SUB_BATCH_PAGES;
        q1_nvcomp_batch = min_batch;
        if (GPU_MEM_BUDGET > non_loader_bytes) {
            size_t nvcomp_per_loader = 2 * 1024 * 1024;
            size_t budget_for_loaders = GPU_MEM_BUDGET - non_loader_bytes;
            size_t per_loader_budget = budget_for_loaders / num_loaders;
            if (per_loader_budget > nvcomp_per_loader) {
                q1_nvcomp_batch = (per_loader_budget - nvcomp_per_loader) / (2 * page_size);
            }
            q1_nvcomp_batch = std::max(q1_nvcomp_batch, min_batch);
            q1_nvcomp_batch = std::min(q1_nvcomp_batch, (size_t)Q1_NVCOMP_BATCH_PAGES);
            q1_nvcomp_batch = (q1_nvcomp_batch / Q1_IO_SUB_BATCH_PAGES) * Q1_IO_SUB_BATCH_PAGES;
        }
        for (size_t t = 0; t < num_loaders; t++) {
            auto &L = loaders[t];
            for (size_t b = 0; b < 2; b++) {
                L.set_buf[b] = mb_cuda_alloc(q1_nvcomp_batch * page_size);
                GDS_CHECK(cuFileBufRegister(L.set_buf[b], q1_nvcomp_batch * page_size, 0));
            }
            if (any_compressed) {
                nvcomp_decompctx_alloc(L.nvctx, q1_nvcomp_batch, page_size, lineitem_fields);
            }
        }
        std::cout << "[Q1] Per-loader IO: 2x" << q1_nvcomp_batch << " pages ("
                  << (2 * q1_nvcomp_batch * page_size / (1024*1024)) << " MiB), "
                  << num_loaders << " loaders" << std::endl;
    }

    // Pre-allocate result buffer on stack
    int64_t h_agg[Q1_NUM_GROUPS * Q1_NUM_AGGS];

    // Pre-allocate thread and zone map vectors (avoid heap alloc in timed section)
    std::vector<std::thread> all_threads;
    all_threads.reserve(num_loaders);
    std::vector<uint32_t> tile_active_pages;
    tile_active_pages.reserve(Q1_TILE_PAGES);

    // Pre-compute tile metadata from prefix_sum (Rule 3: before total_start)
    size_t num_tiles = (npages_ref + Q1_TILE_PAGES - 1) / Q1_TILE_PAGES;
    std::vector<uint64_t> q1_tile_nrecs(num_tiles);
    for (size_t t = 0; t < num_tiles; t++) {
        size_t p_lo = t * Q1_TILE_PAGES;
        size_t tile_np = std::min(Q1_TILE_PAGES, npages_ref - p_lo);
        q1_tile_nrecs[t] = ref_ps[p_lo + tile_np] - ref_ps[p_lo];
    }

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t golap_gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // ── Zone map IO pruning (GDS + GPU, inside timing per Rule 6) ──
    if (zm_has_stats) {
        gds_read_zonemap(zm_cufile_handles, num_devices,
                         zm_stats_start, zm_stats_npg,
                         page_size, d_zm_sd_stats);

        zm_ctx.h_preds[0].d_stats  = reinterpret_cast<int32_t*>(d_zm_sd_stats);
        zm_ctx.h_preds[0].nstats   = zm_nstats;
        zm_ctx.h_preds[0].pred_lo  = INT32_MIN;
        zm_ctx.h_preds[0].pred_hi  = (int32_t)19980902;

        gds_zonemap_eval_async(zm_ctx, zm_min_npages_li, 1, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (uint32_t pg = 0; pg < zm_min_npages_li; pg++)
            if (zm_ctx.h_mask[pg]) active_pages.push_back(pg);
        use_selective = true;

        std::cout << "[ZONEMAP] L_SHIPDATE GPU pruning: "
                  << active_pages.size() << " / " << zm_min_npages_li
                  << " pages active (" << (zm_min_npages_li - active_pages.size())
                  << " pruned)" << std::endl;
    }

    std::cout << "[Q1] Tile execution: " << num_tiles << " tiles of "
              << Q1_TILE_PAGES << " pages (page-direct, capacity=" << capacity << ")" << std::endl;

    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        size_t p_lo = tile_idx * Q1_TILE_PAGES;
        size_t tile_np = std::min(Q1_TILE_PAGES, npages_ref - p_lo);
        uint64_t tile_nrecs = q1_tile_nrecs[tile_idx];

        // Zone map: check active pages in this tile
        tile_active_pages.clear();
        if (use_selective) {
            auto it_lo = std::lower_bound(active_pages.begin(), active_pages.end(), (uint32_t)p_lo);
            auto it_hi = std::lower_bound(active_pages.begin(), active_pages.end(), (uint32_t)(p_lo + tile_np));
            for (auto it = it_lo; it != it_hi; ++it)
                tile_active_pages.push_back(*it);
            if (tile_active_pages.empty()) continue;
        }

        bool selective_partial = use_selective && tile_active_pages.size() < tile_np;

        // Zero data buffers for selective loading (inactive pages → nalloc=0)
        if (selective_partial) {
            for (size_t fi = 0; fi < num_lineitem_cols; fi++)
                CUDA_CHECK(cudaMemset(tile_data_buf[fi], 0, tile_np * page_size));
        }

        if (selective_partial) {
            // ── Selective loading (zone map) — per-thread nvCOMP (unchanged) ──
            all_threads.clear();
            for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
                size_t loader_base = fi * io_thr_per_col;
                size_t n_active = tile_active_pages.size();
                size_t pages_per_thr = (n_active + io_thr_per_col - 1) / io_thr_per_col;
                for (size_t t = 0; t < io_thr_per_col; t++) {
                    size_t pg_start = t * pages_per_thr;
                    if (pg_start >= n_active) break;
                    size_t pg_count = std::min(pages_per_thr, n_active - pg_start);
                    size_t li = loader_base + t;
                    all_threads.emplace_back([&, fi, li, pg_start, pg_count]() {
                        auto &L = loaders[li];
                        mb_cuda_set_context(cuda_ctx_handle);
                        NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                        gds_load_field_pipelined_selective(
                            lineitem_fields[fi], page_size, num_devices,
                            L.cufile_handles, L.set_buf, tile_data_buf[fi],
                            nv, L.io_stream, L.nvcomp_stream,
                            L.io_done_event, L.set_done_events,
                            L.bytes_read, L.ios_completed,
                            L.kernel_launches,
                            tile_active_pages.data() + pg_start, pg_count, p_lo,
                            Q1_IO_SUB_BATCH_PAGES,
                            q1_nvcomp_batch);
                    });
                }
            }
            for (auto &th : all_threads) th.join();
            for (size_t i = 0; i < num_loaders; i++) {
                CUDA_CHECK(cudaStreamSynchronize(loaders[i].io_stream));
                if (loaders[i].nvcomp_stream)
                    CUDA_CHECK(cudaStreamSynchronize(loaders[i].nvcomp_stream));
            }
        } else {
            // ── Per-thread IO-decomp pipeline (column-parallel) ──
            all_threads.clear();
            for (size_t fi = 0; fi < num_lineitem_cols; fi++) {
                size_t loader_base = fi * io_thr_per_col;
                size_t pages_per_thr = (tile_np + io_thr_per_col - 1) / io_thr_per_col;
                for (size_t t = 0; t < io_thr_per_col; t++) {
                    size_t start = p_lo + t * pages_per_thr;
                    size_t end = std::min(start + pages_per_thr, p_lo + tile_np);
                    if (start >= end) break;
                    size_t count = end - start;
                    size_t li = loader_base + t;
                    all_threads.emplace_back([&, fi, li, start, count]() {
                        auto &L = loaders[li];
                        mb_cuda_set_context(cuda_ctx_handle);
                        NvcompDecompCtx *nv = any_compressed ? &L.nvctx : nullptr;
                        gds_load_field_pipelined(
                            lineitem_fields[fi], page_size, num_devices,
                            L.cufile_handles, L.set_buf,
                            tile_data_buf[fi],
                            nv, L.io_stream, L.nvcomp_stream,
                            L.io_done_event, L.set_done_events,
                            L.bytes_read, L.ios_completed,
                            L.kernel_launches,
                            start, count, p_lo,
                            Q1_IO_SUB_BATCH_PAGES,
                            q1_nvcomp_batch);
                    });
                }
            }
            for (auto &th : all_threads) th.join();

            // Sync all loader streams
            for (size_t i = 0; i < num_loaders; i++) {
                CUDA_CHECK(cudaStreamSynchronize(loaders[i].io_stream));
                if (loaders[i].nvcomp_stream)
                    CUDA_CHECK(cudaStreamSynchronize(loaders[i].nvcomp_stream));
            }
        }

        // Page-direct scan + aggregate (no flatten needed)
        q1_scan_aggregate_paged(
            tile_data_buf[L_SDATE_IDX], tile_data_buf[L_QTY_IDX],
            tile_data_buf[L_EPRICE_IDX], tile_data_buf[L_DISC_IDX],
            tile_data_buf[L_TAX_IDX], tile_data_buf[L_RFLAG_IDX],
            tile_data_buf[L_LSTATUS_IDX],
            (uint64_t)tile_np * capacity, capacity, page_size, d_agg, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // ── Results: D→H copy (must be inside timed section per Rule 5) ──
    CUDA_CHECK(cudaMemcpy(h_agg, d_agg, agg_size, cudaMemcpyDeviceToHost));

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(total_end - total_start).count();

    // ── Aggregate I/O stats ──
    for (size_t t = 0; t < num_loaders; t++) {
        total_bytes_read += loaders[t].bytes_read;
        total_ios += loaders[t].ios_completed;
        s_kernel_launches += loaders[t].kernel_launches;
    }

    // Group labels
    const char returnflags[] = {'A', 'A', 'N', 'N', 'R', 'R'};
    const char linestatuses[] = {'F', 'O', 'F', 'O', 'F', 'O'};

    std::cout << "\n=== TPC-H Q1 Result ===" << std::endl;
    std::cout << "l_returnflag|l_linestatus|sum_qty|sum_base_price|sum_disc_price|sum_charge|avg_qty|avg_price|avg_disc|count_order" << std::endl;

    for (int g = 0; g < Q1_NUM_GROUPS; g++) {
        int64_t count = h_agg[g * Q1_NUM_AGGS + Q1_COUNT];
        if (count == 0) continue;

        int64_t sum_qty        = h_agg[g * Q1_NUM_AGGS + Q1_SUM_QTY];
        int64_t sum_base_price = h_agg[g * Q1_NUM_AGGS + Q1_SUM_BASE_PRICE];
        int64_t sum_disc_price = h_agg[g * Q1_NUM_AGGS + Q1_SUM_DISC_PRICE];
        uint64_t charge_lo     = (uint64_t)h_agg[g * Q1_NUM_AGGS + Q1_SUM_CHARGE];
        uint64_t charge_hi     = (uint64_t)h_agg[g * Q1_NUM_AGGS + Q1_SUM_CHARGE_HI];
        unsigned __int128 sum_charge = ((unsigned __int128)charge_hi << 64) | charge_lo;
        int64_t sum_discount   = h_agg[g * Q1_NUM_AGGS + Q1_SUM_DISCOUNT];

        /* NOTE: For consistency, "/ 100" is omitted. */
        double avg_qty   = sum_qty / (double) count;
        double avg_price = sum_base_price / (double) count;
        double avg_disc  = sum_discount / (double) count;

        printf("%c|%c|%ld|%ld|%ld|",
               returnflags[g], linestatuses[g],
               sum_qty, sum_base_price, sum_disc_price);
        q1_print_u128(sum_charge);
        printf("|%lf|%lf|%lf|%ld\n",
               avg_qty, avg_price, avg_disc,
               count);
    }

    std::cout << "\nElapsed: " << elapsed << " sec" << std::endl;
    std::cout << "Total I/O: " << total_bytes_read / (1024*1024) << " MiB, "
              << total_ios << " IOs" << std::endl;

    // ── Cleanup ──
    for (size_t i = 0; i < num_lineitem_cols; i++)
        mb_cuda_free(tile_data_buf[i]);
    CUDA_CHECK(cudaFree(d_agg));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // (Per-thread nvCOMP resources freed with loaders below.)

    for (size_t t = 0; t < num_loaders; t++) {
        auto &L = loaders[t];
        for (size_t b = 0; b < 2; b++) {
            cuFileBufDeregister(L.set_buf[b]);
            mb_cuda_free(L.set_buf[b]);
        }
        CUDA_CHECK(cudaStreamDestroy(L.io_stream));
        CUDA_CHECK(cudaEventDestroy(L.io_done_event));
        CUDA_CHECK(cudaEventDestroy(L.set_done_events[0]));
        CUDA_CHECK(cudaEventDestroy(L.set_done_events[1]));
        if (any_compressed) {
            nvcomp_decompctx_free(L.nvctx);
            CUDA_CHECK(cudaStreamDestroy(L.nvcomp_stream));
        }
        for (size_t d = 0; d < num_devices; d++) {
            for (size_t h = 0; h < L.cufile_handles[d].size(); h++) {
                cuFileHandleDeregister(L.cufile_handles[d][h]);
                close(L.dup_fds[d][h]);
            }
        }
    }

    size_t total_pages = 0;
    for (const auto &fi : lineitem_fields) total_pages += fi.npages;

    if (zm_handles_open) {
        cuFileBufDeregister(d_zm_sd_stats);
        cudaFree(d_zm_sd_stats);
        gds_zonemap_ctx_destroy(zm_ctx);
        for (size_t d = 0; d < num_devices; d++) {
            cuFileHandleDeregister(zm_cufile_handles[d]);
            close(zm_dup_fds[d]);
        }
    }

    std::string comp_str = collect_compression_methods({lineitem_fields});
    free_fields_metadata(lineitem_fields);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = total_ios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count(),
        .compression = comp_str,
        .gpu_mem_bytes = golap_gpu_mem_bytes,
        .gpu_app_bytes = golap_gpu_mem_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// VCHAR scan kernel — scan decompressed pages for checksum
// (no decompression, pages already decompressed by nvCOMP)
// ============================================================

// VCHAR page helpers (same as q13_scan.cu, inlined for this TU)
__device__ __forceinline__ uint32_t slc_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}

__device__ __forceinline__ uint32_t slc_pag_get_oslt(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}

__device__ __forceinline__ uint16_t slc_vchar_len(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = slc_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}

__device__ __forceinline__ const char *slc_vchar_data(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = slc_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);
}

__global__ void vchar_scan_checksum_kernel(
    const char *__restrict__ data_buf,   // decompressed pages at page_idx * page_size
    uint32_t    page_size,
    uint64_t    npages,
    uint64_t   *d_total_records,
    uint64_t   *d_total_strlen,
    uint64_t   *d_total_byte_sum)
{
    uint64_t my_records = 0, my_strlen = 0, my_byte_sum = 0;

    // Grid-stride loop over pages; all threads in block cooperate per page
    for (uint64_t pg = blockIdx.x; pg < npages; pg += gridDim.x) {
        const char *page = data_buf + pg * page_size;
        uint32_t nalloc = slc_pag_get_nalloc(page);

        for (uint32_t s = threadIdx.x; s < nalloc; s += blockDim.x) {
            uint16_t len = slc_vchar_len(page, s, page_size);
            const char *data = slc_vchar_data(page, s, page_size);

            my_records++;
            my_strlen += len;

            uint64_t bsum = 0;
            for (uint16_t b = 0; b < len; b++) {
                bsum += (uint8_t)data[b];
            }
            my_byte_sum += bsum;
        }
    }

    if (my_records > 0)
        atomicAdd((unsigned long long *)d_total_records, (unsigned long long)my_records);
    if (my_strlen > 0)
        atomicAdd((unsigned long long *)d_total_strlen, (unsigned long long)my_strlen);
    if (my_byte_sum > 0)
        atomicAdd((unsigned long long *)d_total_byte_sum, (unsigned long long)my_byte_sum);
}

// ============================================================
// scan_l_comment — GDS Sync read + nvCOMP LZ4 decompress + VCHAR scan
//
// Reads LINEITEM.L_COMMENT LZ4-compressed pages via cuFileRead (GDS sync),
// decompresses with nvCOMP batch LZ4 API, then runs VCHAR scan kernel.
// Reports per-phase timing (IO / Decomp / Scan / Total).
//
// Usage: ./tpchdb -q scan_l_comment -x golap -S /dev/nvme1n1p1
// ============================================================
BenchmarkResult scan_l_comment(BenchmarkOptions &options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    const size_t metadata_head_size = 4096;
    std::vector<int> fds;

    void *ptr;
    if (posix_memalign((void **)&ptr, 512, metadata_head_size) != 0) {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    open_files(options, fds);
    page_pread_host(fds, ptr, 0, metadata_head_size);

    TPCHTableMetadata *metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    {
        const size_t page_size = metadatap->page_size;
        free(ptr);
        if (posix_memalign((void **)&ptr, 512, page_size) != 0) {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        page_pread_host(fds, ptr, 0, page_size);
    }

    metadatap = reinterpret_cast<TPCHTableMetadata *>(ptr);
    TPCHTableMetadata &metadata = *metadatap;
    superpage_set_constants(metadata.page_size);
    const size_t page_size = metadata.page_size;
    const size_t num_devices = fds.size();

    std::cout << "=== scan_l_comment (GDS Sync + nvCOMP LZ4 + VCHAR scan) ===" << std::endl;
    std::cout << "Page Size: " << page_size << ", num_devices: " << num_devices << std::endl;

    // ── Load L_COMMENT field metadata ──
    uint32_t saved_column = metadata.column;
    metadata.column = TPCH::common::Table::LINEITEM;
    constexpr size_t NUM_COLS = 1;
    constexpr std::array<size_t, NUM_COLS> cols = { TPCH::common::L_COMMENT };
    std::vector<FieldPageInfo> fields(NUM_COLS);
    prepare_fields_metadata(fds, metadata, page_size, cols, fields);
    metadata.column = saved_column;

    const FieldPageInfo &fi = fields[0];
    const size_t npages = fi.npages;

    std::cout << "  L_COMMENT: start_page=" << fi.start_page_id
              << " npages=" << npages
              << " compression=" << compression_method_name(fi.compression_method)
              << " nrows=" << metadata.table_lineitem_nrows << std::endl;

    // Compressed page statistics
    uint64_t total_comp_bytes = 0;
    uint32_t max_comp_size = 0, min_comp_size = UINT32_MAX;
    for (size_t i = 0; i < npages; i++) {
        uint32_t cs = fi.compressed_page_sizes[i];
        total_comp_bytes += cs;
        if (cs > max_comp_size) max_comp_size = cs;
        if (cs < min_comp_size) min_comp_size = cs;
    }
    double avg_comp_size = (double)total_comp_bytes / npages;
    double comp_ratio = avg_comp_size / page_size;

    std::cout << "  IO stats: total_comp_bytes=" << total_comp_bytes
              << " (" << std::fixed << std::setprecision(2)
              << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB)"
              << " avg=" << (uint32_t)avg_comp_size
              << " min=" << min_comp_size
              << " max=" << max_comp_size
              << " ratio=" << std::setprecision(3) << comp_ratio
              << std::endl;

    // ── GDS setup ──
    mb_cufile_driver_open();

    const size_t handles_per_thread = options.gds_num_handlers_per_thread;
    std::vector<std::vector<CUfileHandle_t>> cufile_handles(num_devices);
    std::vector<std::vector<int>> dup_fds(num_devices);
    for (size_t d = 0; d < num_devices; d++) {
        for (size_t h = 0; h < handles_per_thread; h++) {
            int duped = dup(fds[d]);
            if (duped < 0) { std::cerr << "dup failed" << std::endl; exit(1); }
            dup_fds[d].push_back(duped);
            cufile_handles[d].push_back(mb_cufile_handle_register(duped));
        }
    }

    // ── GPU allocations ──
    const size_t batch_pages = GDS_SYNC_BATCH_PAGES;  // 1024 pages per batch

    // io_buf: GDS read target (registered with cuFile)
    const size_t io_buf_size = batch_pages * page_size;
    void *io_buf = mb_cuda_alloc(io_buf_size);
    GDS_CHECK(cuFileBufRegister(io_buf, io_buf_size, 0));

    // data_buf: decompressed pages (batch_pages * page_size)
    char *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, batch_pages * page_size));

    // nvCOMP decompression context
    NvcompDecompCtx nvctx;
    nvcomp_decompctx_alloc(nvctx, batch_pages, page_size, fields);

    // Global accumulators
    uint64_t *d_total_records = nullptr, *d_total_strlen = nullptr, *d_total_byte_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total_records, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_total_strlen, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_total_byte_sum, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_total_records, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_total_strlen, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_total_byte_sum, 0, sizeof(uint64_t)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Scan kernel config
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const uint32_t scan_blocks = static_cast<uint32_t>(sm_count);
    const uint32_t scan_threads = 128;

    std::cout << "  batch_pages=" << batch_pages
              << " scan_blocks=" << scan_blocks
              << " scan_threads=" << scan_threads
              << std::endl;

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    size_t gpu_free_alloc = 0;
    cudaMemGetInfo(&gpu_free_alloc, &gpu_total_dummy);
    uint64_t golap_gpu_mem_bytes = gpu_free_start - gpu_free_alloc;

    // ── Sequential pipeline with per-phase timing ──
    uint64_t io_ns_total = 0;
    uint64_t decomp_ns_total = 0;
    uint64_t scan_ns_total = 0;
    size_t total_bytes_read = 0;
    size_t total_ios = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    s_kernel_launches = 0;

    size_t next_fh[MAX_GDS_DEVICES] = {};
    const size_t dev_region_size = io_buf_size / num_devices;
    char *io_base = static_cast<char *>(io_buf);

    for (size_t batch_start = 0; batch_start < npages; batch_start += batch_pages) {
        size_t bp = std::min(batch_pages, npages - batch_start);

        // ── Phase 1: IO — cuFileRead with run coalescing ──
        auto io_start = std::chrono::high_resolution_clock::now();

        for (size_t d = 0; d < num_devices; d++) {
            char *dev_io_base = io_base + d * dev_region_size;
            size_t io_buf_off = 0;
            size_t run_file_offset = 0, run_io_start = 0, run_size = 0;
            bool in_run = false;

            auto flush_run = [&](char *dest, size_t file_off, size_t size) {
                size_t fh_idx = next_fh[d] % cufile_handles[d].size();
                next_fh[d]++;
                off_t buf_offset = dest - io_base;
                ssize_t nread = cuFileRead(cufile_handles[d][fh_idx],
                                           io_buf, size, file_off, buf_offset);
                if (nread < 0 || static_cast<size_t>(nread) != size) {
                    std::cerr << "cuFileRead failed: d=" << d
                              << " size=" << size << " off=" << file_off
                              << " nread=" << nread << std::endl;
                } else {
                    total_bytes_read += nread;
                    total_ios++;
                }
            };

            for (size_t k = 0; k < bp; k++) {
                size_t page_rel = batch_start + k;
                uint64_t page_id = fi.start_page_id + page_rel;
                if (page_id % num_devices != d) continue;

                size_t this_disk_size = roundup4096(fi.compressed_page_sizes[page_rel]);
                size_t this_file_offset = fi.compressed_offsets[page_rel];

                if (in_run && this_file_offset == run_file_offset + run_size) {
                    run_size += this_disk_size;
                } else {
                    if (in_run) {
                        flush_run(dev_io_base + run_io_start,
                                  run_file_offset, run_size);
                    }
                    run_file_offset = this_file_offset;
                    run_io_start = io_buf_off;
                    run_size = this_disk_size;
                    in_run = true;
                }
                io_buf_off += this_disk_size;
            }
            if (in_run) {
                flush_run(dev_io_base + run_io_start,
                          run_file_offset, run_size);
            }
        }

        auto io_end = std::chrono::high_resolution_clock::now();
        io_ns_total += std::chrono::duration_cast<std::chrono::nanoseconds>(
            io_end - io_start).count();

        // ── Phase 2: nvCOMP batch LZ4 decompress ──
        auto decomp_start = std::chrono::high_resolution_clock::now();

        // Scatter from io_buf → queue nvCOMP decompress to data_buf
        size_t decomp_count = 0;
        size_t dev_off[MAX_GDS_DEVICES] = {};

        for (size_t k = 0; k < bp; k++) {
            size_t page_rel = batch_start + k;
            uint64_t page_id = fi.start_page_id + page_rel;
            size_t d = page_id % num_devices;

            char *io_src = io_base + d * dev_region_size + dev_off[d];
            char *dst = d_data + k * page_size;
            size_t cs = fi.compressed_page_sizes[page_rel];

            if (cs < page_size) {
                nvctx.h_comp_ptrs[decomp_count]    = io_src;
                nvctx.h_comp_sizes[decomp_count]   = cs;
                nvctx.h_decomp_ptrs[decomp_count]  = dst;
                nvctx.h_decomp_sizes[decomp_count] = page_size;
                decomp_count++;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst, io_src, page_size,
                    cudaMemcpyDeviceToDevice, stream));
            }
            dev_off[d] += roundup4096(cs);
        }

        if (decomp_count > 0) {
            nvcomp_decompctx_run(fi.compression_method, nvctx,
                                  decomp_count, page_size, stream);
            s_kernel_launches++;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        auto decomp_end = std::chrono::high_resolution_clock::now();
        decomp_ns_total += std::chrono::duration_cast<std::chrono::nanoseconds>(
            decomp_end - decomp_start).count();

        // ── Phase 3: VCHAR scan kernel ──
        auto scan_start = std::chrono::high_resolution_clock::now();

        uint32_t scan_grid = std::min(scan_blocks, (uint32_t)bp);
        vchar_scan_checksum_kernel<<<scan_grid, scan_threads, 0, stream>>>(
            d_data, static_cast<uint32_t>(page_size), bp,
            d_total_records, d_total_strlen, d_total_byte_sum);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto scan_end = std::chrono::high_resolution_clock::now();
        scan_ns_total += std::chrono::duration_cast<std::chrono::nanoseconds>(
            scan_end - scan_start).count();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // Read results
    uint64_t h_records = 0, h_strlen = 0, h_byte_sum = 0;
    CUDA_CHECK(cudaMemcpy(&h_records, d_total_records,
                           sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_strlen, d_total_strlen,
                           sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_byte_sum, d_total_byte_sum,
                           sizeof(uint64_t), cudaMemcpyDeviceToHost));

    double io_ms     = io_ns_total / 1e6;
    double decomp_ms = decomp_ns_total / 1e6;
    double scan_ms   = scan_ns_total / 1e6;
    double total_ms  = elapsed_ns / 1e6;

    std::cout << std::dec
              << "\n=== scan_l_comment results ===\n"
              << "  total_records  = " << h_records << "\n"
              << "  total_strlen   = " << h_strlen << "\n"
              << "  total_byte_sum = " << h_byte_sum << "\n"
              << std::endl;

    std::cout << std::fixed << std::setprecision(2)
              << "  IO time        = " << io_ms << " ms\n"
              << "  Decomp (nvCOMP)= " << decomp_ms << " ms\n"
              << "  Scan           = " << scan_ms << " ms\n"
              << "  Total          = " << total_ms << " ms\n"
              << std::endl;

    {
        double io_s = io_ns_total / 1e9;
        double io_tput = (io_s > 0) ? (double)total_comp_bytes / io_s : 0;
        double total_s = elapsed_ns / 1e9;
        double total_tput = (total_s > 0) ? (double)total_comp_bytes / total_s : 0;
        std::cout << "  IO throughput  = " << std::fixed << std::setprecision(2)
                  << io_tput / (1024.0 * 1024 * 1024) << " GB/s"
                  << " (comp_bytes=" << std::setprecision(2)
                  << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB"
                  << " / " << std::setprecision(3) << io_s << " s)"
                  << "\n  End-to-end     = " << std::setprecision(2)
                  << total_tput / (1024.0 * 1024 * 1024) << " GB/s"
                  << " (" << std::setprecision(3) << total_s << " s)"
                  << std::endl;
    }

    if (h_records == (uint64_t)metadata.table_lineitem_nrows) {
        std::cout << "  Record count matches metadata nrows: OK" << std::endl;
    } else {
        std::cerr << "  WARNING: record count " << h_records
                  << " != metadata nrows " << metadata.table_lineitem_nrows << std::endl;
    }

    // ── Cleanup ──
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_total_records));
    CUDA_CHECK(cudaFree(d_total_strlen));
    CUDA_CHECK(cudaFree(d_total_byte_sum));
    CUDA_CHECK(cudaFree(d_data));
    nvcomp_decompctx_free(nvctx);
    GDS_CHECK(cuFileBufDeregister(io_buf));
    CUDA_CHECK(cudaFree(io_buf));

    for (size_t d = 0; d < num_devices; d++) {
        for (auto &fh : cufile_handles[d]) mb_cufile_handle_deregister(fh);
        for (auto &fd : dup_fds[d]) close(fd);
    }

    free_fields_metadata(fields);
    mb_cufile_driver_close();
    close_files(options, fds);
    free(metadatap);

    return BenchmarkResult{
        .nios = total_ios,
        .read_bytes = (uint64_t)total_bytes_read,
        .elapsed_nanoseconds = (int64_t)elapsed_ns,
        .compression = "LZ4",
        .gpu_mem_bytes = golap_gpu_mem_bytes,
        .gpu_app_bytes = golap_gpu_mem_bytes,
        .kernel_launches = s_kernel_launches,
    };
}

}
