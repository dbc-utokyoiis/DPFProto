#pragma once

// gidp_bam_fusion.cu — GIDP+BAM+FUSION execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMPdx (device-side LZ4), fused IO/decomp/query kernel
//
// Included AFTER gidp.cu and datapathfusion.cu in tpch_main.cu.
// Differs from gidp_bam.cu only in decompression: uses nvCOMPdx device-side
// LZ4 instead of host-launched nvCOMP.

#include "tpch/bam_bulk_read.cuh"
#include "tpch/bam_lz4_decomp.cuh"
#include "tpch/bam_lz4_fused_q6.cuh"
#include "tpch/bam_lz4_fused_q1.cuh"
#include "tpch/bam_lz4_fused_revenue.cuh"
#include "tpch/bam_lz4_fused_q5_lineitem.cuh"
#include "tpch/bam_lz4_fused_q5_lineitem_v2.cuh"
#include "tpch/bam_lz4_fused_q3_lineitem.cuh"
#include "tpch/bam_lz4_fused_q3_orders.cuh"
#include "tpch/bam_lz4_q3sel_scan.cuh"
#include "tpch/bam_lz4_fused_q5_orders.cuh"
#include "tpch/bam_lz4_fused_q16_partsupp.cuh"
#include "tpch/bam_lz4_fused_q16_ptype.cuh"
#include "tpch/bam_lz4_fused_q16_phase01.cuh"
#include "tpch/bam_lz4_fused_q3_mktseg.cuh"
#include "tpch/bam_lz4_fused_q5_dim.cuh"
#include "tpch/bam_q13_kernel.cuh"
#include "tpch/bam_lz4_fused_q13_comment.cuh"
#include "tpch/bam_lz4_fused_q13_comment_v2.cuh"
#include "tpch/bam_io_contention_bench.cuh"
#include "tpch/bam_lz4_decomp_bench.cuh"
#include "tpch/nvcompdx_lz4_bench.cuh"

#include <cub/device/device_merge_sort.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace GidpBamFusion {

static size_t s_kernel_launches;

constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;

// BaM page_cache per-slot overhead beyond the data DMA pages (Cond3 path):
//   PRP list DMA (ctrl_page_size=4096) + cache_page_t (32) + prp1+prp2 (16)
constexpr uint64_t BAM_PC_OVERHEAD_PER_SLOT = 4096 + 32 + 16;

// ════════════════════════════════════════════════════════════
// Utility: batched BaM page read with automatic LZ4 decompression
// ════════════════════════════════════════════════════════════

struct BamPageReadReq {
    uint64_t page_id;        // global page ID (for device routing)
    char *dest;              // final destination (must be within decomp_base buffer for compressed pages)
    uint32_t comp_size;      // 0 = uncompressed, >0 = compressed size in bytes
    uint64_t comp_offset;    // byte offset on disk (compressed pages only)
};

// Read N pages via BaM, internally batching at batch_size.
// Compressed pages: IO → io_staging → LZ4 decompress → dest
// Uncompressed pages: IO → dest directly
//
// For compressed pages, dest must satisfy: (dest - decomp_base) % page_size == 0.
// decomp_base is the base pointer of the contiguous output buffer.
static void batched_bam_read_pages(
    BamBulkReadCtx &io_ctx,
    const BamPageReadReq *reqs, size_t N,
    char *decomp_base,              // base of output buffer for decompress
    char *io_staging,               // temp buffer, batch_size * page_size
    uint32_t *d_decomp_scs,        // GPU temp, batch_size
    uint32_t *d_decomp_spi,        // GPU temp, batch_size
    uint32_t batch_size,
    uint32_t page_size,
    const uint64_t *partition_start_lbas,
    uint32_t n_devices,
    uint32_t blocks_per_page,
    cudaStream_t stream,
    uint64_t &total_io_count,
    uint64_t &total_io_bytes)
{
    constexpr uint32_t CTRL_PAGE_BLOCKS = 4096 / 512;
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + 4095) & ~(size_t)4095;
    };

    std::vector<uint32_t> h_scs(batch_size);
    std::vector<uint32_t> h_spi(batch_size);

    for (size_t batch_start = 0; batch_start < N; batch_start += batch_size) {
        uint32_t batch_count = static_cast<uint32_t>(
            std::min((size_t)batch_size, N - batch_start));
        uint32_t decomp_count = 0;
        uint32_t io_slot = 0;

        for (uint32_t i = 0; i < batch_count; i++) {
            const auto &req = reqs[batch_start + i];
            BamBulkReadDesc &desc = io_ctx.h_descs[0][i];
            desc = {};
            uint32_t dev = req.page_id % n_devices;

            if (req.comp_size > 0) {
                // Compressed: IO → io_staging slot → decompress → dest
                desc.lba = partition_start_lbas[dev] + req.comp_offset / 512;
                desc.nblocks = static_cast<uint32_t>((roundup4096(req.comp_size) + 511) / 512);
                if (desc.nblocks > CTRL_PAGE_BLOCKS && desc.nblocks <= 2 * CTRL_PAGE_BLOCKS)
                    desc.nblocks = 3 * CTRL_PAGE_BLOCKS;
                desc.device = dev;
                desc.dest = io_staging + io_slot * page_size;
                desc.copy_bytes = req.comp_size;
                h_scs[decomp_count] = req.comp_size;
                h_spi[decomp_count] = static_cast<uint32_t>((req.dest - decomp_base) / page_size);
                decomp_count++;
                io_slot++;
            } else {
                // Uncompressed: IO → dest directly
                uint64_t local_page_id = req.page_id / n_devices;
                desc.lba = partition_start_lbas[dev] + local_page_id * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = req.dest;
                desc.copy_bytes = page_size;
            }
        }

        bam_bulk_read_async(io_ctx, batch_count, 0, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (uint32_t i = 0; i < batch_count; i++) {
            total_io_count++;
            total_io_bytes += static_cast<uint64_t>(io_ctx.h_descs[0][i].nblocks) * 512;
        }

        if (decomp_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_scs, h_scs.data(),
                        decomp_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_spi, h_spi.data(),
                        decomp_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            bam_lz4_batch_decompress(
                io_staging, decomp_base,
                d_decomp_scs, d_decomp_spi,
                decomp_count, page_size, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
}

// Variant of batched_bam_read_pages that only decompresses pages [0, decomp_end).
// Pages [decomp_end, N) with comp_size > 0 receive compressed data at req.dest.
static void batched_bam_read_pages_partial_decomp(
    BamBulkReadCtx &io_ctx,
    const BamPageReadReq *reqs, size_t N,
    size_t decomp_end,
    char *decomp_base,
    char *io_staging,
    uint32_t *d_decomp_scs,
    uint32_t *d_decomp_spi,
    uint32_t batch_size,
    uint32_t page_size,
    const uint64_t *partition_start_lbas,
    uint32_t n_devices,
    uint32_t blocks_per_page,
    cudaStream_t stream,
    uint64_t &total_io_count,
    uint64_t &total_io_bytes)
{
    constexpr uint32_t CTRL_PAGE_BLOCKS = 4096 / 512;
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + 4095) & ~(size_t)4095;
    };

    std::vector<uint32_t> h_scs(batch_size);
    std::vector<uint32_t> h_spi(batch_size);

    for (size_t batch_start = 0; batch_start < N; batch_start += batch_size) {
        uint32_t batch_count = static_cast<uint32_t>(
            std::min((size_t)batch_size, N - batch_start));
        uint32_t decomp_count = 0;
        uint32_t io_slot = 0;

        for (uint32_t i = 0; i < batch_count; i++) {
            size_t global_idx = batch_start + i;
            const auto &req = reqs[global_idx];
            BamBulkReadDesc &desc = io_ctx.h_descs[0][i];
            desc = {};
            uint32_t dev = req.page_id % n_devices;
            bool should_decomp = (global_idx < decomp_end);

            if (req.comp_size > 0) {
                desc.lba = partition_start_lbas[dev] + req.comp_offset / 512;
                desc.nblocks = static_cast<uint32_t>((roundup4096(req.comp_size) + 511) / 512);
                if (desc.nblocks > CTRL_PAGE_BLOCKS && desc.nblocks <= 2 * CTRL_PAGE_BLOCKS)
                    desc.nblocks = 3 * CTRL_PAGE_BLOCKS;
                desc.device = dev;

                if (should_decomp) {
                    desc.dest = io_staging + io_slot * page_size;
                    desc.copy_bytes = req.comp_size;
                    h_scs[decomp_count] = req.comp_size;
                    h_spi[decomp_count] = static_cast<uint32_t>((req.dest - decomp_base) / page_size);
                    decomp_count++;
                    io_slot++;
                } else {
                    desc.dest = req.dest;
                    desc.copy_bytes = req.comp_size;
                }
            } else {
                uint64_t local_page_id = req.page_id / n_devices;
                desc.lba = partition_start_lbas[dev] + local_page_id * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = req.dest;
                desc.copy_bytes = page_size;
            }
        }

        bam_bulk_read_async(io_ctx, batch_count, 0, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (uint32_t i = 0; i < batch_count; i++) {
            total_io_count++;
            total_io_bytes += static_cast<uint64_t>(io_ctx.h_descs[0][i].nblocks) * 512;
        }

        if (decomp_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_scs, h_scs.data(),
                        decomp_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_spi, h_spi.data(),
                        decomp_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            bam_lz4_batch_decompress(
                io_staging, decomp_base,
                d_decomp_scs, d_decomp_spi,
                decomp_count, page_size, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
}

BenchmarkResult tpch_q6(BenchmarkOptions &options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    // ── 2. Open BaM controller(s) ──
    size_t gpu_free_pre_ctrl = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_dummy);

    const uint32_t bam_num_queues = 128;
    auto ds = DataPathFusion::bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_dummy);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;
    const uint32_t n_devices = ds.n_devices;

    // Helper: read a striped page to host
    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata page via BaM ──
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    if (rc != 0) {
        std::cerr << "bam_read_page failed (metadata header)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (full metadata)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    std::cout << "=== TPCH Table Metadata ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;

    // ── 4. Extract Q6 field info ──
    constexpr size_t NUM_FIELDS = TPCH::Query::Q6::NUM_SCAN_TARGET_COLS;
    auto q6_cols = TPCH::Query::Q6::SCAN_TARGET_COLS;
    const size_t blocks_per_page = page_size / 512;

    uint64_t field_start_page_ids[NUM_FIELDS];
    uint64_t field_npages_arr[NUM_FIELDS];
    CompressionMethod field_comp_methods[NUM_FIELDS];

    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        size_t col = q6_cols[fi];
        field_start_page_ids[fi] = metadata.table_lineitem_start_page_ids[col];
        field_npages_arr[fi] = metadata.table_lineitem_npages[col];
        field_comp_methods[fi] = static_cast<CompressionMethod>(
            metadata.table_lineitem_compression_method[col]);
        std::cout << "  Field " << col
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << static_cast<int>(field_comp_methods[fi])
                  << std::endl;
    }

    const uint64_t npages = field_npages_arr[0];
    for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
        if (field_npages_arr[fi] != npages) {
            std::cerr << "Error: field " << fi << " has different npages" << std::endl;
            bam_ctrl_close(ctrl);
            exit(EXIT_FAILURE);
        }
    }
    if (npages == 0) {
        std::cout << "No pages to read." << std::endl;
        bam_ctrl_close(ctrl);
        return BenchmarkResult{};
    }

    // ── 5. Read compression metadata ──
    std::vector<uint32_t> h_comp_sizes[NUM_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_FIELDS];

    bool any_compressed = false;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        any_compressed = true;

        size_t col = q6_cols[fi];
        uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
        uint64_t cs_npages_cnt = metadata.table_lineitem_compressed_page_sizes_npages[col];
        uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
        uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];

        // Read compressed page sizes
        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_sizes)" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        h_comp_sizes[fi].assign(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + npages);

        // Read compression base page IDs and compute offsets
        size_t bp_npages = TPCH::nbase_to_npages(nbase, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_bases)" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            h_comp_sizes[fi].data(), nbase, npages, page_size,
            field_start_page_ids[fi], n_devices, offsets_vec);
        h_comp_offsets[fi] = std::move(offsets_vec);
    }

    // Validate: device-side decomp only supports LZ4
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (field_comp_methods[fi] != CompressionMethod::NONE &&
            field_comp_methods[fi] != CompressionMethod::LZ4) {
            std::cerr << "gidp+bam+fusion only supports LZ4 or NONE compression, "
                      << "field " << fi << " uses method "
                      << static_cast<int>(field_comp_methods[fi]) << std::endl;
            bam_ctrl_close(ctrl);
            exit(EXIT_FAILURE);
        }
    }

    // ── 6. Zone map pruning metadata (Rule 3: metadata outside timing) ──
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    uint64_t zm_sd_nstats = 0, zm_sd_stats_start = 0, zm_sd_stats_npg = 0;
    if (options.enable_zonemap) {
        size_t sd_col = q6_cols[0];
        zm_sd_nstats      = metadata.table_lineitem_nstats[sd_col];
        zm_sd_stats_start = metadata.table_lineitem_stats_start_page_ids[sd_col];
        zm_sd_stats_npg   = metadata.table_lineitem_stats_npages[sd_col];
    }

    // ── 7. Warp-Specialized kernel setup ──
    // 8 warps (256 threads) per block:
    //   Warps 0-3: IO (BaM reads), Warps 4-7: Decomp (nvCOMPdx LZ4)
    //   All 256 threads: Q6 scan
    // Intra-block sync via __syncthreads() — no cross-block coherence issues.
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };
    auto safe_nblocks = [](uint32_t nblk) -> uint32_t {
        if (nblk > 8 && nblk <= 16) nblk = 17;
        return nblk;
    };

    // Upload per-field compression metadata to GPU
    uint32_t* d_comp_sizes_gpu[NUM_FIELDS] = {};
    uint64_t* d_comp_offsets_gpu[NUM_FIELDS] = {};
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        CUDA_CHECK(cudaMalloc(&d_comp_sizes_gpu[fi], npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_comp_sizes_gpu[fi], h_comp_sizes[fi].data(),
                    npages * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_comp_offsets_gpu[fi], npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_comp_offsets_gpu[fi], h_comp_offsets[fi].data(),
                    npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    // Grid size: capped by occupancy (full page count; kernel skips pruned pages)
    const uint32_t max_blocks = q6_warp_spec_max_blocks(static_cast<uint32_t>(page_size));
    const uint32_t num_blocks = std::min(static_cast<uint32_t>(npages), max_blocks);

    // Page cache: 56 slots per block (2 bufs × 7 batch × 4 fields)
    auto io_pc = bam_io_page_cache_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks * 56);
    void* d_ctrls_pc     = bam_io_page_cache_get_d_ctrls(io_pc);
    void* d_pc_ptr       = bam_io_page_cache_get_d_pc_ptr(io_pc);
    const char* pc_base  = (const char*)bam_io_page_cache_get_base_addr(io_pc);

    // BaM zonemap ctx (Rule 4: alloc outside timing; borrows page_cache)
    if (options.enable_zonemap && zm_sd_nstats > 0 && zm_sd_stats_start > 0 && zm_sd_stats_npg > 0) {
        zm_ctx = bam_zonemap_ctx_create(d_ctrls_pc, d_pc_ptr, (void*)pc_base,
            static_cast<uint32_t>(page_size), npages);
        for (uint64_t j = 0; j < zm_sd_stats_npg; j++) {
            uint64_t pg_id = zm_sd_stats_start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            zm_ctx.h_reads[zm_nreads++] = {
                ds.partition_start_lbas[dev] + local * blocks_per_page,
                static_cast<uint32_t>(blocks_per_page), dev};
        }
        zm_ctx.h_preds[zm_npreds++] = {0, zm_sd_nstats,
            options.q6_sd_low, options.q6_sd_high - 1};
        zm_valid = true;
    }

    // Decomp buffer: 56 pages per block (2 bufs × 7 batch × 4 fields)
    char* d_decomp_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_decomp_buf,
        (size_t)num_blocks * 56 * page_size));

    // Revenue output
    int64_t* d_q6_revenue = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q6_revenue, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_q6_revenue, 0, sizeof(int64_t)));

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // Fill params
    Q6WarpSpecParams wsp{};
    wsp.total_pages = static_cast<uint32_t>(npages);
    wsp.d_page_mask = nullptr;
    wsp.page_size = static_cast<uint32_t>(page_size);
    wsp.n_devices = n_devices;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        wsp.field_start_page_ids[fi] = field_start_page_ids[fi];
        wsp.d_comp_offsets[fi] = d_comp_offsets_gpu[fi];
        wsp.d_comp_sizes[fi] = d_comp_sizes_gpu[fi];
        wsp.is_compressed[fi] = (field_comp_methods[fi] != CompressionMethod::NONE);
    }
    for (uint32_t d = 0; d < n_devices && d < 4; d++)
        wsp.partition_start_lbas[d] = ds.partition_start_lbas[d];
    wsp.sd_low = options.q6_sd_low;
    wsp.sd_high = options.q6_sd_high;
    wsp.d_revenue = d_q6_revenue;

    std::cout << "[GIDP+BAM+WARPSPEC Q6] num_blocks=" << num_blocks
              << " (npages=" << npages << " zm=" << (zm_valid ? "on" : "off") << ")"
              << std::endl;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    // ════════════════════════════════════════════
    // total_start — warp-specialized kernel
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // Zone map eval (Rule 6: IO + eval inside timing, mask stays on GPU)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
        const uint32_t num_active = *zm_ctx.h_num_active;
        std::cout << "[ZONEMAP] L_SHIPDATE pruning: active=" << num_active
                  << "/" << npages << std::endl;
        wsp.d_active_page_ids = zm_ctx.d_active_ids;
        wsp.total_pages = num_active;
    }

    const uint32_t q6_launch_blocks = zm_valid
        ? std::min(*zm_ctx.h_num_active, num_blocks) : num_blocks;
    q6_warp_spec_launch(d_ctrls_pc, d_pc_ptr, pc_base,
                        d_decomp_buf, wsp, q6_launch_blocks, stream);
    s_kernel_launches++;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Result ──
    int64_t h_q6_revenue = 0;
    CUDA_CHECK(cudaMemcpy(&h_q6_revenue, d_q6_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::cout << "TPCH Q6 total revenue: " << h_q6_revenue << std::endl;

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

    // Compute IO stats (using h_mask for active pages only)
    uint64_t total_io_count = 0;
    uint64_t total_io_bytes = 0;
    {
        const uint8_t* h_mask = zm_valid ? zm_ctx.h_mask : nullptr;
        for (size_t pg = 0; pg < npages; pg++) {
            if (h_mask && !h_mask[pg]) continue;
            for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
                uint32_t nblk;
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    nblk = safe_nblocks(
                        static_cast<uint32_t>((roundup4096(comp_sz) + 511) / 512));
                } else {
                    nblk = static_cast<uint32_t>(page_size / 512);
                }
                total_io_bytes += (uint64_t)nblk * 512;
                total_io_count++;
            }
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    bam_io_page_cache_destroy(io_pc);
    cudaFree(d_decomp_buf);
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (d_comp_sizes_gpu[fi]) cudaFree(d_comp_sizes_gpu[fi]);
        if (d_comp_offsets_gpu[fi]) cudaFree(d_comp_offsets_gpu[fi]);
    }
    cudaFree(d_q6_revenue);
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_ctrl_close(ctrl);

    // Collect compression method string
    std::string comp_str;
    {
        std::set<std::string> methods;
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            methods.insert(compression_method_name(field_comp_methods[fi]));
        for (const auto &m : methods) {
            if (!comp_str.empty()) comp_str += "+";
            comp_str += m;
        }
    }

    return BenchmarkResult{
        .nios = total_io_count,
        .read_bytes = total_io_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = comp_str,
        .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NUM_FIELDS,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// TPC-H Q13 — GIDP+BAM+DECOMP execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMPdx (device-side LZ4)
// Pipeline: q13_golap (scan → sort → RLE → probe → sort → RLE → pack)
// ============================================================
BenchmarkResult tpch_q13(BenchmarkOptions &options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    // ── 2. Open BaM controller(s) ──
    size_t gpu_free_pre_ctrl = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_dummy);

    const uint32_t bam_num_queues = 128;
    auto ds = DataPathFusion::bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_dummy);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;
    const uint32_t n_devices = ds.n_devices;

    // Helper: read a striped page to host
    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata page via BaM ──
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    if (rc != 0) {
        std::cerr << "bam_read_page failed (metadata header)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (full metadata)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    std::cout << "=== TPCH Q13 (GIDP+BAM+DECOMP) ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;

    // ── 4. Extract Q13 field info ──
    constexpr size_t FI_O_CUSTKEY = 0;
    constexpr size_t FI_O_COMMENT = 1;
    constexpr size_t FI_C_CUSTKEY = 2;
    constexpr size_t NUM_Q13_FIELDS = 3;

    const size_t blocks_per_page = page_size / 512;
    const size_t o_custkey_col = TPCH::common::O_CUSTKEY;
    const size_t o_comment_col = TPCH::common::O_COMMENT;
    const size_t c_custkey_col = TPCH::common::C_CUSTKEY;

    uint64_t field_start_page_ids[NUM_Q13_FIELDS];
    uint64_t field_npages_arr[NUM_Q13_FIELDS];
    CompressionMethod field_comp_methods[NUM_Q13_FIELDS];

    // ORDERS columns
    field_start_page_ids[FI_O_CUSTKEY] = metadata.table_orders_start_page_ids[o_custkey_col];
    field_npages_arr[FI_O_CUSTKEY] = metadata.table_orders_npages[o_custkey_col];
    field_comp_methods[FI_O_CUSTKEY] = static_cast<CompressionMethod>(
        metadata.table_orders_compression_method[o_custkey_col]);

    field_start_page_ids[FI_O_COMMENT] = metadata.table_orders_start_page_ids[o_comment_col];
    field_npages_arr[FI_O_COMMENT] = metadata.table_orders_npages[o_comment_col];
    field_comp_methods[FI_O_COMMENT] = static_cast<CompressionMethod>(
        metadata.table_orders_compression_method[o_comment_col]);

    // CUSTOMER columns
    field_start_page_ids[FI_C_CUSTKEY] = metadata.table_customer_start_page_ids[c_custkey_col];
    field_npages_arr[FI_C_CUSTKEY] = metadata.table_customer_npages[c_custkey_col];
    field_comp_methods[FI_C_CUSTKEY] = static_cast<CompressionMethod>(
        metadata.table_customer_compression_method[c_custkey_col]);

    const char *field_names[NUM_Q13_FIELDS] = { "O_CUSTKEY", "O_COMMENT", "C_CUSTKEY" };
    for (size_t fi = 0; fi < NUM_Q13_FIELDS; fi++) {
        std::cout << "  " << field_names[fi]
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << static_cast<int>(field_comp_methods[fi])
                  << std::endl;
    }

    const uint64_t nrecs_orders = metadata.table_orders_nrows;
    const uint64_t nrecs_customer = metadata.table_customer_nrows;
    std::cout << "  nrecs_orders=" << nrecs_orders
              << " nrecs_customer=" << nrecs_customer << std::endl;

    // ── 5. Read compression metadata ──
    std::vector<uint32_t> h_comp_sizes[NUM_Q13_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_Q13_FIELDS];

    bool any_compressed = false;
    for (size_t fi = 0; fi < NUM_Q13_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        any_compressed = true;

        uint64_t cs_start, cs_npages_cnt, nbase, base_start;
        if (fi == FI_O_CUSTKEY || fi == FI_O_COMMENT) {
            size_t col = (fi == FI_O_CUSTKEY) ? o_custkey_col : o_comment_col;
            cs_start = metadata.table_orders_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_orders_compressed_page_sizes_npages[col];
            nbase = metadata.table_orders_compression_nbases[col];
            base_start = metadata.table_orders_compression_base_start_page_ids[col];
        } else {
            cs_start = metadata.table_customer_compressed_page_sizes_start_page_ids[c_custkey_col];
            cs_npages_cnt = metadata.table_customer_compressed_page_sizes_npages[c_custkey_col];
            nbase = metadata.table_customer_compression_nbases[c_custkey_col];
            base_start = metadata.table_customer_compression_base_start_page_ids[c_custkey_col];
        }

        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_sizes " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        h_comp_sizes[fi].assign(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages_arr[fi]);

        size_t bp_npages = TPCH::nbase_to_npages(nbase, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_bases " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            h_comp_sizes[fi].data(), nbase, field_npages_arr[fi], page_size,
            field_start_page_ids[fi], n_devices, offsets_vec);
        h_comp_offsets[fi] = std::move(offsets_vec);
    }

    // Validate compression methods per field
    // O_CUSTKEY / C_CUSTKEY: LZ4 or NONE (device-side LZ4 decomp + flatten)
    // O_COMMENT: LZ4, LZ4PAR, or NONE (batch scan; LZ4PAR uses fused PAR-32K kernel)
    for (size_t fi = 0; fi < NUM_Q13_FIELDS; fi++) {
        auto cm = field_comp_methods[fi];
        if (fi == FI_O_COMMENT) {
            if (cm != CompressionMethod::NONE && cm != CompressionMethod::LZ4
                && cm != CompressionMethod::LZ4PAR) {
                std::cerr << "gidp+bam+fusion Q13: O_COMMENT requires LZ4, LZ4PAR, or NONE, "
                          << "got method " << static_cast<int>(cm) << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        } else {
            if (cm != CompressionMethod::NONE && cm != CompressionMethod::LZ4) {
                std::cerr << "gidp+bam+fusion Q13: " << field_names[fi]
                          << " requires LZ4 or NONE, got method "
                          << static_cast<int>(cm) << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
    }

    // ── 6. Read prefix sums via BaM ──
    auto read_prefix_sum = [&](uint64_t ps_start, uint64_t ps_npages_cnt, uint64_t npages_field)
        -> std::vector<uint64_t>
    {
        std::vector<uint64_t> result_ps;
        if (ps_npages_cnt == 0) return result_ps;
        std::vector<char> ps_buf(ps_npages_cnt * page_size);
        for (uint64_t p = 0; p < ps_npages_cnt; p++) {
            rc = read_striped_page(ps_start + p, page_size, ps_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (prefix_sum)" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        auto *raw = reinterpret_cast<uint64_t*>(ps_buf.data());
        result_ps.assign(raw + 1, raw + 1 + npages_field);
        return result_ps;
    };

    std::vector<uint64_t> h_ps_o_custkey = read_prefix_sum(
        metadata.table_orders_prefix_sum_start_page_ids[o_custkey_col],
        metadata.table_orders_prefix_sum_npages[o_custkey_col],
        field_npages_arr[FI_O_CUSTKEY]);

    std::vector<uint64_t> h_ps_o_comment = read_prefix_sum(
        metadata.table_orders_prefix_sum_start_page_ids[o_comment_col],
        metadata.table_orders_prefix_sum_npages[o_comment_col],
        field_npages_arr[FI_O_COMMENT]);

    std::vector<uint64_t> h_ps_c_custkey = read_prefix_sum(
        metadata.table_customer_prefix_sum_start_page_ids[c_custkey_col],
        metadata.table_customer_prefix_sum_npages[c_custkey_col],
        field_npages_arr[FI_C_CUSTKEY]);

    // ── 7. GPU memory + stream ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    // Bench: 3024 warps (sm*7*4) → 31.84 GB/s decomp+KMP; up from sm*4 (1728 warps, 20 GB/s)
    uint32_t num_blocks = static_cast<uint32_t>(sm_count) * 7;
    if (num_blocks > static_cast<uint32_t>(field_npages_arr[FI_O_COMMENT]))
        num_blocks = static_cast<uint32_t>(field_npages_arr[FI_O_COMMENT]);

    auto roundup4096_q13 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    // ── Streaming buffer allocation ──
    constexpr size_t Q13_BATCH_PAGES = 512;
    // Max chunk size for split_batch fallback = BaM page_cache slot cap
    const uint32_t q13_max_split_chunk = (uint32_t)(sm_count * 8) * 4;
    const uint32_t q13_max_batch = std::max((uint32_t)Q13_BATCH_PAGES, q13_max_split_chunk);
    void *staging_data = mb_cuda_alloc(Q13_BATCH_PAGES * page_size);
    void *staging_io = mb_cuda_alloc(Q13_BATCH_PAGES * page_size);
    uint64_t *d_batch_ps = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_ps, q13_max_batch * sizeof(uint64_t)));
    std::vector<uint64_t> bps(q13_max_batch);

    uint64_t total_io_count = 0, total_io_bytes = 0;

    // Persistent BaM context for batch reads (no cudaMalloc/cudaFree per call)
    BamBulkReadCtx io_ctx = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(sm_count),
        static_cast<uint32_t>(Q13_BATCH_PAGES));

    // Pre-allocate decomp scratch buffers (Rule 4: no cudaMalloc inside measurement)
    uint32_t *d_decomp_scs_q13 = nullptr, *d_decomp_spi_q13 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_decomp_scs_q13, Q13_BATCH_PAGES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_decomp_spi_q13, Q13_BATCH_PAGES * sizeof(uint32_t)));

    // ── Batch helpers (device-side LZ4) ──
    auto batch_read_lz4_field = [&](size_t fi, size_t pg_start, size_t pg_count) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        auto* descs = io_ctx.h_descs[0];
        size_t io_off = 0;
        for (size_t pg = 0; pg < pg_count; pg++) {
            auto &desc = descs[pg];
            size_t abs_pg = pg_start + pg;
            if (is_compressed) {
                uint64_t page_id = field_start_page_ids[fi] + abs_pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + h_comp_offsets[fi][abs_pg] / 512;
                desc.nblocks = (roundup4096_q13(h_comp_sizes[fi][abs_pg]) + 511) / 512;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_io) + io_off;
                desc.copy_bytes = h_comp_sizes[fi][abs_pg];
                io_off += page_size;
            } else {
                uint64_t page_id = field_start_page_ids[fi] + abs_pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + (page_id / n_devices) * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_data) + pg * page_size;
                desc.copy_bytes = page_size;
            }
        }
        bam_bulk_read_async(io_ctx, (uint32_t)pg_count, 0, stream);
        s_kernel_launches++;
        for (size_t p = 0; p < pg_count; p++) {
            total_io_count++;
            total_io_bytes += static_cast<uint64_t>(descs[p].nblocks) * 512;
        }

        if (is_compressed) {
            std::vector<uint32_t> h_scs(pg_count), h_spi(pg_count);
            for (size_t pg = 0; pg < pg_count; pg++) {
                h_scs[pg] = h_comp_sizes[fi][pg_start + pg];
                h_spi[pg] = static_cast<uint32_t>(pg);
            }
            CUDA_CHECK(cudaMemcpy(d_decomp_scs_q13, h_scs.data(), pg_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_decomp_spi_q13, h_spi.data(), pg_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
            bam_lz4_batch_decompress(
                static_cast<const char*>(staging_io), static_cast<char*>(staging_data),
                d_decomp_scs_q13, d_decomp_spi_q13, (uint32_t)pg_count, (uint32_t)page_size, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    auto upload_batch_ps = [&](const std::vector<uint64_t> &h_ps, size_t pg, size_t bnp,
                               uint64_t &row_start, uint64_t &batch_nrecs) {
        row_start = (pg == 0) ? 0 : h_ps[pg - 1];
        batch_nrecs = h_ps[pg + bnp - 1] - row_start;
        for (size_t p = 0; p < bnp; p++)
            bps[p] = h_ps[pg + p] - row_start;
        CUDA_CHECK(cudaMemcpy(d_batch_ps, bps.data(), bnp * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
    };

    auto batch_flatten_int64 = [&](size_t fi, const std::vector<uint64_t> &h_ps,
                                   uint64_t nrecs_total, uint64_t *d_flat) {
        for (size_t pg = 0; pg < field_npages_arr[fi]; pg += Q13_BATCH_PAGES) {
            size_t bnp = std::min(Q13_BATCH_PAGES, (size_t)(field_npages_arr[fi] - pg));
            batch_read_lz4_field(fi, pg, bnp);
            uint64_t row_start, batch_nrecs;
            upload_batch_ps(h_ps, pg, bnp, row_start, batch_nrecs);
            q13_flatten_int64_pages_ps(
                static_cast<const char*>(staging_data), page_size, d_batch_ps,
                bnp, batch_nrecs, d_flat + row_start, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // ── 8. Pre-allocate flat arrays (cudaMalloc outside measurement) ──
    uint64_t *d_o_custkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_custkey_flat, nrecs_orders * sizeof(uint64_t)));
    uint64_t *d_c_custkey = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c_custkey, nrecs_customer * sizeof(uint64_t)));

    // ── 10. KMP setup ──
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

    // ── 11. Allocate scan output + O_COMMENT GPU metadata ──
    uint64_t *d_o_aggr_custkey = nullptr;
    uint64_t *d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_aggr_custkey, nrecs_orders * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemsetAsync(d_o_aggr_custkey, 0xFF,
                               nrecs_orders * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream));

    uint64_t *d_ps_o_comment = nullptr;
    uint32_t *d_comp_sizes = nullptr;
    uint64_t *d_comp_offsets_gpu = nullptr;
    const uint64_t o_comment_npages = field_npages_arr[FI_O_COMMENT];
    CUDA_CHECK(cudaMalloc(&d_ps_o_comment, o_comment_npages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_comp_sizes, o_comment_npages * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_comp_offsets_gpu, o_comment_npages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_o_comment, h_ps_o_comment.data(),
                          o_comment_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
    if (field_comp_methods[FI_O_COMMENT] != CompressionMethod::NONE) {
        CUDA_CHECK(cudaMemcpy(d_comp_sizes, h_comp_sizes[FI_O_COMMENT].data(),
                              o_comment_npages * sizeof(uint32_t), cudaMemcpyHostToDevice));
        std::vector<uint64_t> co64(o_comment_npages);
        for (uint64_t i = 0; i < o_comment_npages; i++)
            co64[i] = static_cast<uint64_t>(h_comp_offsets[FI_O_COMMENT][i]);
        CUDA_CHECK(cudaMemcpy(d_comp_offsets_gpu, co64.data(),
                              o_comment_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    // ── 11b. O_CUSTKEY / C_CUSTKEY GPU metadata (for fused flatten) ──
    auto upload_field_comp_meta = [&](size_t fi, uint32_t **d_cs, uint64_t **d_co, uint64_t **d_ps,
                                       const std::vector<uint64_t> &h_ps) {
        uint64_t np = field_npages_arr[fi];
        CUDA_CHECK(cudaMalloc(d_cs, np * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(d_co, np * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(d_ps, np * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(*d_ps, h_ps.data(), np * sizeof(uint64_t), cudaMemcpyHostToDevice));
        if (field_comp_methods[fi] != CompressionMethod::NONE) {
            CUDA_CHECK(cudaMemcpy(*d_cs, h_comp_sizes[fi].data(), np * sizeof(uint32_t), cudaMemcpyHostToDevice));
            std::vector<uint64_t> co64(np);
            for (uint64_t i = 0; i < np; i++)
                co64[i] = static_cast<uint64_t>(h_comp_offsets[fi][i]);
            CUDA_CHECK(cudaMemcpy(*d_co, co64.data(), np * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
    };

    uint32_t *d_ock_comp_sizes = nullptr;
    uint64_t *d_ock_comp_offsets = nullptr, *d_ps_o_custkey_gpu = nullptr;
    upload_field_comp_meta(FI_O_CUSTKEY, &d_ock_comp_sizes, &d_ock_comp_offsets,
                           &d_ps_o_custkey_gpu, h_ps_o_custkey);

    uint32_t *d_cck_comp_sizes = nullptr;
    uint64_t *d_cck_comp_offsets = nullptr, *d_ps_c_custkey_gpu = nullptr;
    upload_field_comp_meta(FI_C_CUSTKEY, &d_cck_comp_sizes, &d_cck_comp_offsets,
                           &d_ps_c_custkey_gpu, h_ps_c_custkey);

    // ── 12. O_COMMENT context setup ──
    const bool o_comment_is_lz4par =
        (field_comp_methods[FI_O_COMMENT] == CompressionMethod::LZ4PAR);
    const bool o_comment_compressed =
        (field_comp_methods[FI_O_COMMENT] != CompressionMethod::NONE);

    // Budget cap for fused contexts: pre-compute pipeline + P12 cost, cap O_COMMENT blocks
    {
        uint64_t pipeline_bytes =
            (uint64_t)nrecs_orders * (sizeof(uint64_t) * 2 + sizeof(uint32_t)) +
            sizeof(uint64_t) +
            (uint64_t)nrecs_customer * (sizeof(uint32_t) * 4 + sizeof(uint64_t) * 4) +
            q13_pipeline_cub_temp_size(nrecs_orders, nrecs_customer);
        // P12 context: page_cache(slots * PS) + decomp(slots * PS) + page_cache overhead
        uint32_t p12_slots = std::min(
            (uint32_t)(field_npages_arr[FI_O_CUSTKEY] + field_npages_arr[FI_C_CUSTKEY]),
            (uint32_t)(sm_count * 8 * 4));  // Q16P01_WARPS=4
        uint64_t p12_cost = (uint64_t)p12_slots * (2 * page_size + BAM_PC_OVERHEAD_PER_SLOT);

        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t app_used = gpu_free_before_app - gpu_free_now;
        uint64_t app_budget = (gpu_ctrl_bytes < GPU_MEM_BUDGET)
            ? GPU_MEM_BUDGET - gpu_ctrl_bytes : 0;
        uint64_t reserved = pipeline_bytes + p12_cost;
        if (app_used + reserved < app_budget) {
            uint64_t remaining = app_budget - app_used - reserved;
            // O_COMMENT fused: 4 page_cache slots + 4 decomp pages per block (both v2 and LZ4PAR)
            uint64_t per_block = 4 * (2 * page_size + BAM_PC_OVERHEAD_PER_SLOT);
            uint32_t max_blocks = static_cast<uint32_t>(remaining / per_block);
            if (max_blocks < num_blocks) {
                std::cout << "[Q13] Budget cap: fused blocks "
                          << num_blocks << " → " << max_blocks << std::endl;
                num_blocks = std::max(max_blocks, 1u);
            }
        } else {
            num_blocks = 1;
        }
    }

    // Fused context: LZ4PAR or standard LZ4
    bam_q13_fused_io_ctx_t fused_ctx{};
    uint64_t *d_phase_cycles = nullptr;
    bam_fused_q13c_v2_ctx_t fused_q13c_ctx{};
    if (o_comment_is_lz4par) {
        fused_ctx = bam_q13_fused_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks);
        CUDA_CHECK(cudaMalloc(&d_phase_cycles, 3 * sizeof(uint64_t)));
    } else {
        // Standard LZ4 / NONE: fused IO+LZ4+KMP v2 kernel (4-warp pattern)
        fused_q13c_ctx = bam_fused_q13c_v2_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks);
    }

    // ── Phase 1+2 batched split context (O_CUSTKEY + C_CUSTKEY) ──
    const uint32_t q13_p12_total_pages =
        (uint32_t)(field_npages_arr[FI_O_CUSTKEY] + field_npages_arr[FI_C_CUSTKEY]);
    auto q13_p12_ctx = bam_fused_q16p01_create(ctrl, page_size, q13_p12_total_pages);
    cudaStream_t stream_p12_io;
    CUDA_CHECK(cudaStreamCreate(&stream_p12_io));

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

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    total_io_count = 0; total_io_bytes = 0;  // reset at timing start

    // ── Phase 1+2: Fused IO+decomp+flatten for O_CUSTKEY + C_CUSTKEY ──
    {
        const uint32_t q13_p12_num_blocks = std::min((q13_p12_total_pages + 3u) / 4u,
                                                      (uint32_t)(sm_count * 8));

        auto fill_q13_io = [&](BAMFusedQ16IOBase &b, size_t fi,
                               uint64_t *d_co, uint32_t *d_cs, uint64_t *d_ps) {
            b.field_start_page_id = field_start_page_ids[fi];
            b.d_comp_offsets = d_co;
            b.d_comp_sizes = d_cs;
            b.is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
            for (uint32_t d = 0; d < n_devices; d++)
                b.partition_start_lbas[d] = ds.partition_start_lbas[d];
            b.n_devices = n_devices;
            b.page_size = page_size;
            b.npages = field_npages_arr[fi];
            b.num_blocks = q13_p12_num_blocks;
            b.d_prefix_sum = d_ps;
        };

        // O_CUSTKEY: fused IO+decomp+flatten INT64
        {
            BAMFusedQ16FlattenI64Params p{};
            fill_q13_io(p, FI_O_CUSTKEY, d_ock_comp_offsets, d_ock_comp_sizes, d_ps_o_custkey_gpu);
            p.d_output = d_o_custkey_flat;
            bam_fused_q16p01_flatten_i64_async(q13_p12_ctx, p, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // C_CUSTKEY: fused IO+decomp+flatten INT64
        {
            BAMFusedQ16FlattenI64Params p{};
            fill_q13_io(p, FI_C_CUSTKEY, d_cck_comp_offsets, d_cck_comp_sizes, d_ps_c_custkey_gpu);
            p.d_output = d_c_custkey;
            bam_fused_q16p01_flatten_i64_async(q13_p12_ctx, p, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        {
            auto now = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - total_start).count();
            printf("[Q13-TIMING] Phase 1+2 total: %.3f ms\n", ms);
        }

        // IO stats
        for (size_t fi : {FI_O_CUSTKEY, FI_C_CUSTKEY}) {
            const auto& cs = h_comp_sizes[fi];
            bool compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
            for (uint64_t pg = 0; pg < field_npages_arr[fi]; pg++) {
                total_io_count++;
                if (compressed) {
                    uint32_t comp_sz = cs[pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
    }

    // Destroy Phase 1+2 context (not needed after this point)
    bam_fused_q16p01_destroy(q13_p12_ctx);
    CUDA_CHECK(cudaStreamDestroy(stream_p12_io));

    auto phase2_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase2_end - total_start).count();
        printf("[Q13-TIMING] Phase 1+2 total: %.3f ms\n", ms);
    }

    // ── Phase 3: O_COMMENT scan ──
    if (o_comment_is_lz4par) {
        // ── Path A: LZ4PAR — Fused IO+PAR-32K Decomp+Scan ──
        std::cout << "[GIDP+BAM+DECOMP Q13] Fused IO+decomp+scan for O_COMMENT ("
                  << o_comment_npages << " pages, " << num_blocks << " blocks)..."
                  << std::endl;

        CUDA_CHECK(cudaMemsetAsync(d_phase_cycles, 0, 3 * sizeof(uint64_t), stream));

        BAMq13FusedParams fused_params{};
        fused_params.d_comp_sizes      = d_comp_sizes;
        fused_params.d_comp_offsets    = d_comp_offsets_gpu;
        fused_params.d_prefix_sum      = d_ps_o_comment;
        fused_params.d_o_custkey_flat  = d_o_custkey_flat;
        fused_params.d_o_aggr_custkey  = d_o_aggr_custkey;
        fused_params.d_count           = d_count;
        fused_params.d_patterns        = d_patterns;
        fused_params.d_next            = d_next;
        fused_params.d_pattern_offsets = d_pattern_offsets;
        fused_params.d_pattern_lengths = d_pattern_lengths;
        fused_params.num_patterns      = num_patterns;
        fused_params.partition_start_lba = ds.partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            fused_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        fused_params.n_devices          = n_devices;
        fused_params.field_start_page_id = field_start_page_ids[FI_O_COMMENT];
        fused_params.page_size          = static_cast<uint32_t>(page_size);
        fused_params.comp_method        = static_cast<uint16_t>(field_comp_methods[FI_O_COMMENT]);
        fused_params.npages             = o_comment_npages;
        fused_params.num_blocks         = num_blocks;
        fused_params.d_phase_cycles     = d_phase_cycles;

        bam_q13_fused_io_decomp_scan_async(fused_ctx, fused_params, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Count fused I/O stats for O_COMMENT
        for (uint64_t pg = 0; pg < o_comment_npages; pg++) {
            total_io_count++;
            if (o_comment_compressed) {
                uint32_t comp_sz = h_comp_sizes[FI_O_COMMENT][pg];
                uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                if (nblk > 8 && nblk <= 16) nblk = 24;
                total_io_bytes += (uint64_t)nblk * 512;
            } else {
                total_io_bytes += page_size;
            }
        }
    } else {
        // ── Path B: LZ4 / NONE — Fused IO+LZ4+KMP v2 kernel (4-warp pattern) ──
        std::cout << "[GIDP+BAM+DECOMP Q13] Fused IO+LZ4+KMP v2 for O_COMMENT ("
                  << o_comment_npages << " pages, " << num_blocks << " blocks)..."
                  << std::endl;

        BAMFusedQ13Cv2Params fq13c_params{};
        fq13c_params.field_start_page_id = field_start_page_ids[FI_O_COMMENT];
        fq13c_params.d_comp_offsets     = d_comp_offsets_gpu;
        fq13c_params.d_comp_sizes       = d_comp_sizes;
        fq13c_params.is_compressed      = o_comment_compressed;
        for (uint32_t d = 0; d < n_devices; d++)
            fq13c_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        fq13c_params.n_devices          = n_devices;
        fq13c_params.page_size          = static_cast<uint32_t>(page_size);
        fq13c_params.npages             = o_comment_npages;
        fq13c_params.num_blocks         = num_blocks;
        fq13c_params.d_prefix_sum       = d_ps_o_comment;
        fq13c_params.d_o_custkey_flat   = d_o_custkey_flat;
        fq13c_params.d_o_aggr_custkey   = d_o_aggr_custkey;
        fq13c_params.d_count            = d_count;
        fq13c_params.d_patterns         = d_patterns;
        fq13c_params.d_next             = d_next;
        fq13c_params.d_pattern_offsets  = d_pattern_offsets;
        fq13c_params.d_pattern_lengths  = d_pattern_lengths;
        fq13c_params.num_patterns       = num_patterns;

        bam_fused_q13c_v2_run_async(fused_q13c_ctx, fq13c_params, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Count fused I/O stats for O_COMMENT
        for (uint64_t pg = 0; pg < o_comment_npages; pg++) {
            total_io_count++;
            if (o_comment_compressed) {
                uint32_t comp_sz = h_comp_sizes[FI_O_COMMENT][pg];
                uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                if (nblk > 8 && nblk <= 16) nblk = 24;
                total_io_bytes += (uint64_t)nblk * 512;
            } else {
                total_io_bytes += page_size;
            }
        }
    }

    uint64_t h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    std::cout << "[Q13] Qualifying orders (NOT LIKE): " << h_count
              << " / " << nrecs_orders << std::endl;

    auto phase3_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase3_end - phase2_end).count();
        printf("[Q13-TIMING] Phase 3 (O_COMMENT scan, %lupg): %.3f ms\n",
               (unsigned long)field_npages_arr[FI_O_COMMENT], ms);
    }

    // ── Phase 4: Aggregation pipeline (sort → RLE → probe → sort → RLE → pack) ──
    std::cout << "[Q13] Running aggregation pipeline..." << std::endl;
    std::vector<std::pair<uint32_t, uint32_t>> q13_result;
    q13_pig_aggregate(q13_bufs, d_o_aggr_custkey, nrecs_orders,
                       d_c_custkey, nrecs_customer,
                       q13_result, stream);
    s_kernel_launches++;

    auto phase4_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase4_end - phase3_end).count();
        printf("[Q13-TIMING] Phase 4 (Aggregation): %.3f ms\n", ms);
    }

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

    // ── Print results ──
    std::cout << "\n=== TPC-H Q13 Result ===" << std::endl;
    std::cout << "c_count | custdist" << std::endl;
    std::cout << "--------+---------" << std::endl;
    for (auto &[c_count, custdist] : q13_result) {
        printf("%7u | %8u\n", c_count, custdist);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    if (fused_ctx) bam_q13_fused_io_destroy(fused_ctx);
    if (d_phase_cycles) cudaFree(d_phase_cycles);
    if (fused_q13c_ctx) bam_fused_q13c_v2_destroy(fused_q13c_ctx);
    cudaFree(d_comp_sizes);
    cudaFree(d_comp_offsets_gpu);
    cudaFree(d_ps_o_comment);
    cudaFree(d_patterns);
    cudaFree(d_next);
    cudaFree(d_pattern_offsets);
    cudaFree(d_pattern_lengths);
    cudaFree(d_o_custkey_flat);
    cudaFree(d_ock_comp_sizes);
    cudaFree(d_ock_comp_offsets);
    cudaFree(d_ps_o_custkey_gpu);
    cudaFree(d_cck_comp_sizes);
    cudaFree(d_cck_comp_offsets);
    cudaFree(d_ps_c_custkey_gpu);
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
    mb_cuda_free(staging_data);
    mb_cuda_free(staging_io);
    cudaFree(d_batch_ps);
    cudaFree(d_decomp_scs_q13);
    cudaFree(d_decomp_spi_q13);
    bam_bulk_read_ctx_destroy(io_ctx);
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_ctrl_close(ctrl);

    // Collect compression method string
    std::string comp_str;
    {
        std::set<std::string> methods;
        for (size_t fi = 0; fi < NUM_Q13_FIELDS; fi++)
            methods.insert(compression_method_name(field_comp_methods[fi]));
        for (const auto &m : methods) {
            if (!comp_str.empty()) comp_str += "+";
            comp_str += m;
        }
    }

    size_t total_pages = 0;
    for (size_t fi = 0; fi < NUM_Q13_FIELDS; fi++)
        total_pages += field_npages_arr[fi];

    return BenchmarkResult{
        .nios = total_io_count,
        .read_bytes = total_io_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = comp_str,
        .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// TPC-H Q16 — GIDP+BAM+DECOMP execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMPdx (device-side LZ4)
// Pipeline: supplier anti-join → PART HT → PARTSUPP probe → COUNT DISTINCT
// ============================================================
BenchmarkResult tpch_q16(BenchmarkOptions &options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    // ── 2. Open BaM controller(s) ──
    size_t gpu_free_pre_ctrl = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_dummy);

    const uint32_t bam_num_queues = 128;
    auto ds = DataPathFusion::bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_dummy);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;
    const uint32_t n_devices = ds.n_devices;

    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata page via BaM ──
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    if (rc != 0) {
        std::cerr << "bam_read_page failed (metadata header)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (full metadata)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    std::cout << "=== TPCH Q16 (GIDP+BAM+DECOMP) ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;

    const uint64_t nrecs_supplier = metadata.table_supplier_nrows;
    const uint64_t nrecs_part = metadata.table_part_nrows;
    const uint64_t nrecs_partsupp = metadata.table_partsupp_nrows;
    std::cout << "nrecs_supplier=" << nrecs_supplier
              << " nrecs_part=" << nrecs_part
              << " nrecs_partsupp=" << nrecs_partsupp << std::endl;

    // ── 4. Extract Q16 field info (8 fields, 3 tables) ──
    constexpr size_t FI_S_SUPPKEY  = 0;
    constexpr size_t FI_S_COMMENT  = 1;
    constexpr size_t FI_P_PARTKEY  = 2;
    constexpr size_t FI_P_BRAND    = 3;
    constexpr size_t FI_P_TYPE     = 4;
    constexpr size_t FI_P_SIZE     = 5;
    constexpr size_t FI_PS_PARTKEY = 6;
    constexpr size_t FI_PS_SUPPKEY = 7;
    constexpr size_t NUM_Q16_FIELDS = 8;

    const size_t blocks_per_page = page_size / 512;

    const size_t col_s_suppkey  = TPCH::common::S_SUPPKEY;
    const size_t col_s_comment  = TPCH::common::S_COMMENT;
    const size_t col_p_partkey  = TPCH::common::P_PARTKEY;
    const size_t col_p_brand    = TPCH::common::P_BRAND;
    const size_t col_p_type     = TPCH::common::P_TYPE;
    const size_t col_p_size     = TPCH::common::P_SIZE;
    const size_t col_ps_partkey = TPCH::common::PS_PARTKEY;
    const size_t col_ps_suppkey = TPCH::common::PS_SUPPKEY;

    uint64_t field_start_page_ids[NUM_Q16_FIELDS];
    uint64_t field_npages_arr[NUM_Q16_FIELDS];
    CompressionMethod field_comp_methods[NUM_Q16_FIELDS];

    // SUPPLIER columns
    field_start_page_ids[FI_S_SUPPKEY] = metadata.table_supplier_start_page_ids[col_s_suppkey];
    field_npages_arr[FI_S_SUPPKEY]     = metadata.table_supplier_npages[col_s_suppkey];
    field_comp_methods[FI_S_SUPPKEY]   = static_cast<CompressionMethod>(
        metadata.table_supplier_compression_method[col_s_suppkey]);

    field_start_page_ids[FI_S_COMMENT] = metadata.table_supplier_start_page_ids[col_s_comment];
    field_npages_arr[FI_S_COMMENT]     = metadata.table_supplier_npages[col_s_comment];
    field_comp_methods[FI_S_COMMENT]   = static_cast<CompressionMethod>(
        metadata.table_supplier_compression_method[col_s_comment]);

    // PART columns
    field_start_page_ids[FI_P_PARTKEY] = metadata.table_part_start_page_ids[col_p_partkey];
    field_npages_arr[FI_P_PARTKEY]     = metadata.table_part_npages[col_p_partkey];
    field_comp_methods[FI_P_PARTKEY]   = static_cast<CompressionMethod>(
        metadata.table_part_compression_method[col_p_partkey]);

    field_start_page_ids[FI_P_BRAND] = metadata.table_part_start_page_ids[col_p_brand];
    field_npages_arr[FI_P_BRAND]     = metadata.table_part_npages[col_p_brand];
    field_comp_methods[FI_P_BRAND]   = static_cast<CompressionMethod>(
        metadata.table_part_compression_method[col_p_brand]);

    field_start_page_ids[FI_P_TYPE] = metadata.table_part_start_page_ids[col_p_type];
    field_npages_arr[FI_P_TYPE]     = metadata.table_part_npages[col_p_type];
    field_comp_methods[FI_P_TYPE]   = static_cast<CompressionMethod>(
        metadata.table_part_compression_method[col_p_type]);

    field_start_page_ids[FI_P_SIZE] = metadata.table_part_start_page_ids[col_p_size];
    field_npages_arr[FI_P_SIZE]     = metadata.table_part_npages[col_p_size];
    field_comp_methods[FI_P_SIZE]   = static_cast<CompressionMethod>(
        metadata.table_part_compression_method[col_p_size]);

    // PARTSUPP columns
    field_start_page_ids[FI_PS_PARTKEY] = metadata.table_partsupp_start_page_ids[col_ps_partkey];
    field_npages_arr[FI_PS_PARTKEY]     = metadata.table_partsupp_npages[col_ps_partkey];
    field_comp_methods[FI_PS_PARTKEY]   = static_cast<CompressionMethod>(
        metadata.table_partsupp_compression_method[col_ps_partkey]);

    field_start_page_ids[FI_PS_SUPPKEY] = metadata.table_partsupp_start_page_ids[col_ps_suppkey];
    field_npages_arr[FI_PS_SUPPKEY]     = metadata.table_partsupp_npages[col_ps_suppkey];
    field_comp_methods[FI_PS_SUPPKEY]   = static_cast<CompressionMethod>(
        metadata.table_partsupp_compression_method[col_ps_suppkey]);

    const char *field_names[NUM_Q16_FIELDS] = {
        "S_SUPPKEY", "S_COMMENT", "P_PARTKEY", "P_BRAND",
        "P_TYPE", "P_SIZE", "PS_PARTKEY", "PS_SUPPKEY" };
    for (size_t fi = 0; fi < NUM_Q16_FIELDS; fi++) {
        std::cout << "  " << field_names[fi]
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << static_cast<int>(field_comp_methods[fi])
                  << std::endl;
    }

    // ── 5. Read compression metadata ──
    std::vector<uint32_t> h_comp_sizes[NUM_Q16_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_Q16_FIELDS];

    bool any_compressed = false;

    auto get_comp_meta = [&](size_t fi, uint64_t &cs_start, uint64_t &cs_npages_cnt,
                             uint64_t &nbase, uint64_t &base_start) {
        switch (fi) {
        case FI_S_SUPPKEY: case FI_S_COMMENT: {
            size_t col = (fi == FI_S_SUPPKEY) ? col_s_suppkey : col_s_comment;
            cs_start      = metadata.table_supplier_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_supplier_compressed_page_sizes_npages[col];
            nbase         = metadata.table_supplier_compression_nbases[col];
            base_start    = metadata.table_supplier_compression_base_start_page_ids[col];
            break;
        }
        case FI_P_PARTKEY: case FI_P_BRAND: case FI_P_TYPE: case FI_P_SIZE: {
            size_t col;
            switch (fi) {
            case FI_P_PARTKEY: col = col_p_partkey; break;
            case FI_P_BRAND:   col = col_p_brand; break;
            case FI_P_TYPE:    col = col_p_type; break;
            default:           col = col_p_size; break;
            }
            cs_start      = metadata.table_part_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_part_compressed_page_sizes_npages[col];
            nbase         = metadata.table_part_compression_nbases[col];
            base_start    = metadata.table_part_compression_base_start_page_ids[col];
            break;
        }
        case FI_PS_PARTKEY: case FI_PS_SUPPKEY: {
            size_t col = (fi == FI_PS_PARTKEY) ? col_ps_partkey : col_ps_suppkey;
            cs_start      = metadata.table_partsupp_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_partsupp_compressed_page_sizes_npages[col];
            nbase         = metadata.table_partsupp_compression_nbases[col];
            base_start    = metadata.table_partsupp_compression_base_start_page_ids[col];
            break;
        }
        }
    };

    for (size_t fi = 0; fi < NUM_Q16_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        any_compressed = true;

        uint64_t cs_start, cs_npages_cnt, nbase, base_start;
        get_comp_meta(fi, cs_start, cs_npages_cnt, nbase, base_start);

        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_sizes " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        h_comp_sizes[fi].assign(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages_arr[fi]);

        size_t bp_npages = TPCH::nbase_to_npages(nbase, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_bases " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        // DEBUG: print nbase, base_page_ids, and comp_sizes for root-cause analysis
        {
            auto *bp = reinterpret_cast<size_t*>(bases_buf.data());
            std::cout << "[DEBUG COMP META] " << field_names[fi]
                      << ": nbase=" << nbase
                      << " bp_npages=" << bp_npages
                      << " base_start=" << base_start
                      << " cs_start=" << cs_start
                      << " cs_npages_cnt=" << cs_npages_cnt
                      << std::endl;
            size_t nprint = std::min((uint64_t)5, nbase);
            std::cout << "  base_page_ids[0.." << (nprint > 0 ? nprint - 1 : 0) << "]:";
            for (uint64_t b = 0; b < nprint; b++)
                std::cout << " " << bp[b];
            std::cout << std::endl;
            std::cout << "  comp_sizes[0..2]: " << h_comp_sizes[fi][0]
                      << " " << h_comp_sizes[fi][1]
                      << " " << h_comp_sizes[fi][2] << std::endl;
        }

        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            h_comp_sizes[fi].data(), nbase, field_npages_arr[fi], page_size,
            field_start_page_ids[fi], n_devices, offsets_vec);
        h_comp_offsets[fi] = std::move(offsets_vec);

        // DEBUG: print first few offsets
        {
            std::cout << "  offsets[0..4]:";
            for (size_t k = 0; k < std::min((size_t)5, h_comp_offsets[fi].size()); k++)
                std::cout << " " << h_comp_offsets[fi][k];
            std::cout << std::endl;
        }
    }

    // Validate: device-side decomp only supports LZ4
    for (size_t fi = 0; fi < NUM_Q16_FIELDS; fi++) {
        if (field_comp_methods[fi] != CompressionMethod::NONE &&
            field_comp_methods[fi] != CompressionMethod::LZ4) {
            std::cerr << "gidp+bam+fusion only supports LZ4 or NONE compression, "
                      << field_names[fi] << " uses method "
                      << static_cast<int>(field_comp_methods[fi]) << std::endl;
            bam_ctrl_close(ctrl);
            exit(EXIT_FAILURE);
        }
    }

    // ── 6. Read prefix sums via BaM ──
    auto read_prefix_sum = [&](uint64_t ps_start, uint64_t ps_npages_cnt, uint64_t npages_field)
        -> std::vector<uint64_t>
    {
        std::vector<uint64_t> result_ps;
        if (ps_npages_cnt == 0) return result_ps;
        std::vector<char> ps_buf(ps_npages_cnt * page_size);
        for (uint64_t p = 0; p < ps_npages_cnt; p++) {
            rc = read_striped_page(ps_start + p, page_size, ps_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (prefix_sum)" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        auto *raw = reinterpret_cast<uint64_t*>(ps_buf.data());
        result_ps.assign(raw + 1, raw + 1 + npages_field);
        return result_ps;
    };

    auto get_ps_meta = [&](size_t fi, uint64_t &ps_start, uint64_t &ps_npages_cnt) {
        switch (fi) {
        case FI_S_SUPPKEY: case FI_S_COMMENT: {
            size_t col = (fi == FI_S_SUPPKEY) ? col_s_suppkey : col_s_comment;
            ps_start      = metadata.table_supplier_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_supplier_prefix_sum_npages[col];
            break;
        }
        case FI_P_PARTKEY: case FI_P_BRAND: case FI_P_TYPE: case FI_P_SIZE: {
            size_t col;
            switch (fi) {
            case FI_P_PARTKEY: col = col_p_partkey; break;
            case FI_P_BRAND:   col = col_p_brand; break;
            case FI_P_TYPE:    col = col_p_type; break;
            default:           col = col_p_size; break;
            }
            ps_start      = metadata.table_part_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_part_prefix_sum_npages[col];
            break;
        }
        case FI_PS_PARTKEY: case FI_PS_SUPPKEY: {
            size_t col = (fi == FI_PS_PARTKEY) ? col_ps_partkey : col_ps_suppkey;
            ps_start      = metadata.table_partsupp_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_partsupp_prefix_sum_npages[col];
            break;
        }
        }
    };

    std::vector<uint64_t> h_prefix_sum[NUM_Q16_FIELDS];
    for (size_t fi = 0; fi < NUM_Q16_FIELDS; fi++) {
        uint64_t ps_start, ps_npages_cnt;
        get_ps_meta(fi, ps_start, ps_npages_cnt);
        h_prefix_sum[fi] = read_prefix_sum(ps_start, ps_npages_cnt, field_npages_arr[fi]);
    }

    // ── 7. GPU memory allocation ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    uint64_t total_io_count = 0, total_io_bytes = 0;

    // ── Fused Q16 PARTSUPP kernel setup (before total_start) ──
    const uint32_t q16_ps_npages = field_npages_arr[FI_PS_PARTKEY];
    uint32_t q16_num_blocks = std::min((uint32_t)864, (uint32_t)(sm_count * 8));

    // Pre-compute P_TYPE and P01 block/slot counts for budget check
    const uint32_t q16_pt_npages = field_npages_arr[FI_P_TYPE];
    uint32_t q16_pt_num_blocks = std::min(q16_pt_npages, (uint32_t)(sm_count * 8));
    const uint32_t q16_p01_num_slots = std::max({
        (uint32_t)field_npages_arr[FI_S_SUPPKEY],
        (uint32_t)field_npages_arr[FI_S_COMMENT],
        (uint32_t)(field_npages_arr[FI_P_PARTKEY] + field_npages_arr[FI_P_SIZE]
                   + field_npages_arr[FI_P_BRAND])});

    // ── Budget cap for fused contexts ──
    {
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t app_used = gpu_free_before_app - gpu_free_now;
        uint64_t app_budget = (gpu_ctrl_bytes < GPU_MEM_BUDGET)
            ? GPU_MEM_BUDGET - gpu_ctrl_bytes : 0;

        // Estimate non-fused GPU allocation costs
        uint64_t nonfused_bytes = 0;
        // Metadata (prefix sums, comp offsets/sizes)
        nonfused_bytes += (q16_ps_npages + 1) * sizeof(uint64_t);
        for (size_t fi : {FI_PS_PARTKEY, FI_PS_SUPPKEY}) {
            if (field_comp_methods[fi] != CompressionMethod::NONE)
                nonfused_bytes += (uint64_t)field_npages_arr[fi] * (sizeof(uint64_t) + sizeof(uint32_t));
        }
        nonfused_bytes += (uint64_t)q16_pt_npages * (sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t));
        for (size_t fi : {FI_S_SUPPKEY, FI_S_COMMENT, FI_P_PARTKEY, FI_P_BRAND, FI_P_SIZE})
            nonfused_bytes += (uint64_t)field_npages_arr[fi] * (sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t));
        // Pipeline buffers (dominant non-fused cost)
        nonfused_bytes += (uint64_t)nrecs_partsupp * (sizeof(uint64_t) * 3 + sizeof(uint32_t) * 5) + 16;
        nonfused_bytes += q16_pipeline_cub_temp_size(nrecs_partsupp);
        // Phase 0+1 flat arrays and hash tables
        nonfused_bytes += (uint64_t)nrecs_supplier * sizeof(uint64_t) * 2;
        nonfused_bytes += (uint64_t)nrecs_part * (sizeof(uint64_t) * 2 + sizeof(uint32_t) * 3);
        uint32_t ht_cap_est = 1;
        while (ht_cap_est < nrecs_part * 2) ht_cap_est <<= 1;
        nonfused_bytes += (uint64_t)ht_cap_est * (sizeof(uint64_t) + sizeof(uint32_t));
        uint32_t excl_cap_est = 1;
        while (excl_cap_est < nrecs_supplier * 4 + 16) excl_cap_est <<= 1;
        nonfused_bytes += (uint64_t)excl_cap_est * sizeof(uint64_t);
        nonfused_bytes += (uint64_t)q16_p01_num_slots * sizeof(uint64_t);

        // P01 fused context cost (treated as fixed, not capped)
        uint64_t p01_cost = (uint64_t)q16_p01_num_slots * (2 * page_size + BAM_PC_OVERHEAD_PER_SLOT);
        uint64_t fixed_cost = nonfused_bytes + p01_cost;

        if (app_used + fixed_cost < app_budget) {
            uint64_t avail_for_ps_pt = app_budget - app_used - fixed_cost;
            // PARTSUPP: 4 page_cache slots + 8 decomp pages per block
            uint64_t ps_cost = (uint64_t)q16_num_blocks * (12 * page_size + 4 * BAM_PC_OVERHEAD_PER_SLOT);
            // P_TYPE: 4 page_cache slots + 4 decomp pages per block
            uint64_t pt_cost = (uint64_t)q16_pt_num_blocks * (8 * page_size + 4 * BAM_PC_OVERHEAD_PER_SLOT);
            if (ps_cost + pt_cost > avail_for_ps_pt) {
                double scale = (double)avail_for_ps_pt / (double)(ps_cost + pt_cost);
                uint32_t new_ps = std::max(1u, (uint32_t)(q16_num_blocks * scale));
                uint32_t new_pt = std::max(1u, (uint32_t)(q16_pt_num_blocks * scale));
                std::cout << "[Q16] Budget cap: PS blocks " << q16_num_blocks << " → " << new_ps
                          << ", PT blocks " << q16_pt_num_blocks << " → " << new_pt << std::endl;
                q16_num_blocks = new_ps;
                q16_pt_num_blocks = new_pt;
            }
        } else {
            q16_num_blocks = 1;
            q16_pt_num_blocks = 1;
        }
    }

    // Create fused context (page cache + decomp buffer)
    auto q16_fused_ctx = bam_fused_q16ps_create(ctrl, page_size, q16_num_blocks);

    // Upload exclusive prefix sum for PARTSUPP (npages+1 entries, ps[0]=0)
    uint64_t *d_q16_ps = nullptr;
    {
        std::vector<uint64_t> excl_ps(q16_ps_npages + 1);
        excl_ps[0] = 0;
        for (uint32_t i = 0; i < q16_ps_npages; i++)
            excl_ps[i + 1] = h_prefix_sum[FI_PS_PARTKEY][i];
        CUDA_CHECK(cudaMalloc(&d_q16_ps, (q16_ps_npages + 1) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_q16_ps, excl_ps.data(),
            (q16_ps_npages + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    // Upload compression metadata for PS_PARTKEY and PS_SUPPKEY to GPU
    uint64_t *d_q16_comp_offsets[2] = {nullptr, nullptr};
    uint32_t *d_q16_comp_sizes[2] = {nullptr, nullptr};
    for (int fi_idx = 0; fi_idx < 2; fi_idx++) {
        size_t fi = (fi_idx == 0) ? FI_PS_PARTKEY : FI_PS_SUPPKEY;
        if (field_comp_methods[fi] != CompressionMethod::NONE) {
            uint32_t np = field_npages_arr[fi];
            CUDA_CHECK(cudaMalloc(&d_q16_comp_offsets[fi_idx], np * sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&d_q16_comp_sizes[fi_idx], np * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_q16_comp_offsets[fi_idx], h_comp_offsets[fi].data(),
                np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_q16_comp_sizes[fi_idx], h_comp_sizes[fi].data(),
                np * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }

    // Pre-allocate Q16PipelineBuffers (sort+RLE pipeline, no malloc inside timed section)
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
        constexpr size_t Q16_MAX_GROUPS = 256 * 1024;
        q16_bufs.h_result_capacity = Q16_MAX_GROUPS;
        q16_bufs.h_gids = (uint32_t *)malloc(Q16_MAX_GROUPS * sizeof(uint32_t));
        q16_bufs.h_counts = (uint32_t *)malloc(Q16_MAX_GROUPS * sizeof(uint32_t));
    }

    // ── Fused P_TYPE context (separate BaM page cache for parallel stream) ──
    auto q16_pt_ctx = bam_fused_q16pt_create(ctrl, page_size, q16_pt_num_blocks);

    // Upload P_TYPE compression metadata to GPU
    uint64_t *d_pt_comp_offsets = nullptr;
    uint32_t *d_pt_comp_sizes = nullptr;
    if (field_comp_methods[FI_P_TYPE] != CompressionMethod::NONE) {
        CUDA_CHECK(cudaMalloc(&d_pt_comp_offsets, q16_pt_npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_pt_comp_sizes, q16_pt_npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_pt_comp_offsets, h_comp_offsets[FI_P_TYPE].data(),
            q16_pt_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pt_comp_sizes, h_comp_sizes[FI_P_TYPE].data(),
            q16_pt_npages * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // Upload P_TYPE inclusive prefix sum
    uint64_t *d_pt_prefix_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pt_prefix_sum, q16_pt_npages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_pt_prefix_sum, h_prefix_sum[FI_P_TYPE].data(),
        q16_pt_npages * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // ── Phase 0+1 context (shared page cache for all Phase 0+1 fields) ──
    auto q16_p01_ctx = bam_fused_q16p01_create(ctrl, page_size, q16_p01_num_slots);

    // Upload P01 field compression metadata to GPU
    uint64_t *d_p01_comp_offsets[NUM_Q16_FIELDS] = {};
    uint32_t *d_p01_comp_sizes[NUM_Q16_FIELDS] = {};
    uint64_t *d_p01_prefix_sum[NUM_Q16_FIELDS] = {};
    for (size_t fi : {FI_S_SUPPKEY, FI_S_COMMENT, FI_P_PARTKEY, FI_P_BRAND, FI_P_SIZE}) {
        uint32_t np = field_npages_arr[fi];
        if (field_comp_methods[fi] != CompressionMethod::NONE) {
            CUDA_CHECK(cudaMalloc(&d_p01_comp_offsets[fi], np * sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&d_p01_comp_sizes[fi], np * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_p01_comp_offsets[fi], h_comp_offsets[fi].data(),
                np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_p01_comp_sizes[fi], h_comp_sizes[fi].data(),
                np * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc(&d_p01_prefix_sum[fi], np * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_p01_prefix_sum[fi], h_prefix_sum[fi].data(),
            np * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    // num_blocks for non-split kernel launch (unused in split path, kept for API compat)
    const uint32_t q16_p01_num_blocks = std::min((uint32_t)(sm_count * 8),
        (q16_p01_num_slots + 3u) / 4u);

    // Helper to fill BAMFusedQ16IOBase for a field
    auto fill_io_base = [&](BAMFusedQ16IOBase &b, size_t fi, uint32_t nblk) {
        b.field_start_page_id = field_start_page_ids[fi];
        b.d_comp_offsets = d_p01_comp_offsets[fi];
        b.d_comp_sizes = d_p01_comp_sizes[fi];
        b.is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        for (uint32_t d = 0; d < n_devices; d++)
            b.partition_start_lbas[d] = ds.partition_start_lbas[d];
        b.n_devices = n_devices;
        b.page_size = page_size;
        b.npages = field_npages_arr[fi];
        b.num_blocks = nblk;
        b.d_prefix_sum = d_p01_prefix_sum[fi];
    };

    // ── Split IO infrastructure for Phase 0+1 ──
    cudaStream_t stream_p01_io;
    CUDA_CHECK(cudaStreamCreate(&stream_p01_io));
    char* decomp_buf = bam_fused_q16p01_get_decomp_buf(q16_p01_ctx);

    // ── Phase 0+1 flat arrays & hash tables ──
    uint64_t *d_s_suppkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_s_suppkey_flat, nrecs_supplier * sizeof(uint64_t)));

    uint64_t *d_p_partkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p_partkey_flat, nrecs_part * sizeof(uint64_t)));

    uint32_t *d_p_size_u32 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p_size_u32, nrecs_part * sizeof(uint32_t)));

    uint32_t *d_p_brand_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p_brand_ids, nrecs_part * sizeof(uint32_t)));

    uint32_t *d_type_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_type_ids, nrecs_part * sizeof(uint32_t)));

    // P_TYPE dictionary
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
    CUDA_CHECK(cudaMemset(d_dict_keys, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_dict_type_ids, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_type_id_counter, 0, sizeof(uint32_t)));

    // PART hash table
    uint32_t ht_capacity = 1;
    while (ht_capacity < nrecs_part * 2) ht_capacity <<= 1;
    uint32_t ht_mask = ht_capacity - 1;
    uint64_t *d_ht_keys = nullptr;
    uint32_t *d_ht_group_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_keys, ht_capacity * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_group_ids, ht_capacity * sizeof(uint32_t)));

    // KMP patterns for '%Customer%Complaints%'
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

    uint64_t *d_excl_suppkeys = nullptr;
    uint32_t *d_excl_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_excl_suppkeys, nrecs_supplier * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_excl_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_excl_count, 0, sizeof(uint32_t)));

    // Pre-allocate excluded-supplier hash table (max capacity: all suppliers excluded)
    uint32_t excl_max_cap = 1;
    while (excl_max_cap < nrecs_supplier * 4 + 16) excl_max_cap <<= 1;
    uint64_t *d_excl_keys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_excl_keys, excl_max_cap * sizeof(uint64_t)));

    // Pre-allocate P_SIZE temporary flatten buffer
    uint64_t *d_p_size_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p_size_tmp, nrecs_part * sizeof(uint64_t)));

    // Pre-allocate batch prefix sum buffer for sequential fallback path
    const uint32_t q16_p01_actual_slots = bam_fused_q16p01_get_num_slots(q16_p01_ctx);
    uint64_t *d_q16_batch_ps = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q16_batch_ps, q16_p01_actual_slots * sizeof(uint64_t)));

    // Pre-allocate Phase 1 P_TYPE stream + events
    cudaStream_t stream_pt;
    CUDA_CHECK(cudaStreamCreate(&stream_pt));
    cudaEvent_t ev_pt_start, ev_pt_end;
    CUDA_CHECK(cudaEventCreate(&ev_pt_start));
    CUDA_CHECK(cudaEventCreate(&ev_pt_end));

    // Pre-allocate PARTSUPP IO stream (before total_start per Rule 4)
    cudaStream_t stream_partsupp_io;
    CUDA_CHECK(cudaStreamCreate(&stream_partsupp_io));

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // ── Launch all Phase 0 + Phase 1 fields concurrently ──

    // ── Phase 0: SUPPLIER (fused IO+decomp+flatten / fused IO+decomp+scan) ──
    auto p0_field_start = std::chrono::steady_clock::now();
    {
        BAMFusedQ16FlattenI64Params p{};
        fill_io_base(p, FI_S_SUPPKEY, q16_p01_num_blocks);
        p.d_output = d_s_suppkey_flat;
        bam_fused_q16p01_flatten_i64_async(q16_p01_ctx, p, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - p0_field_start).count();
        printf("[Q16-TIMING]   S_SUPPKEY (%upg): %.3f ms\n", field_npages_arr[FI_S_SUPPKEY], ms);
        p0_field_start = now;
    }
    {
        BAMFusedQ16SupplierScanParams p{};
        fill_io_base(p, FI_S_COMMENT, q16_p01_num_blocks);
        p.d_s_suppkey_flat = d_s_suppkey_flat;
        p.d_patterns = d_kmp_patterns;
        p.d_next = d_kmp_next;
        p.d_pattern_offsets = d_kmp_offsets;
        p.d_pattern_lengths = d_kmp_lengths;
        p.num_patterns = excl_num_patterns;
        p.d_excl_suppkeys = d_excl_suppkeys;
        p.d_excl_count = d_excl_count;
        bam_fused_q16p01_supplier_scan_async(q16_p01_ctx, p, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - p0_field_start).count();
        printf("[Q16-TIMING]   S_COMMENT (%upg): %.3f ms\n", field_npages_arr[FI_S_COMMENT], ms);
    }

    // Build excluded suppkey hash table
    uint32_t h_excl_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_excl_count, d_excl_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "[Q16] Excluded suppliers: " << h_excl_count << std::endl;

    uint32_t excl_capacity = 1;
    while (excl_capacity < (uint32_t)h_excl_count * 4 + 16) excl_capacity <<= 1;
    uint32_t excl_mask = excl_capacity - 1;
    // d_excl_keys already pre-allocated with excl_max_cap >= excl_capacity
    CUDA_CHECK(cudaMemsetAsync(d_excl_keys, 0xFF, excl_capacity * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    q16_build_excl_ht(d_excl_suppkeys, h_excl_count, d_excl_keys, excl_mask, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto phase0_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase0_end - total_start).count();
        printf("[Q16-TIMING] Phase 0 (SUPPLIER): %.3f ms\n", ms);
    }


    // ── Phase 1: P_TYPE (fused, parallel stream) + P_PARTKEY/P_SIZE/P_BRAND (split, main stream) ──
    CUDA_CHECK(cudaEventRecord(ev_pt_start, stream_pt));
    {
        BAMFusedQ16PTypeParams p{};
        p.field_start_page_id = field_start_page_ids[FI_P_TYPE];
        p.d_comp_offsets = d_pt_comp_offsets;
        p.d_comp_sizes = d_pt_comp_sizes;
        p.is_compressed = (field_comp_methods[FI_P_TYPE] != CompressionMethod::NONE);
        p.d_prefix_sum = d_pt_prefix_sum;
        for (uint32_t d = 0; d < n_devices; d++)
            p.partition_start_lbas[d] = ds.partition_start_lbas[d];
        p.n_devices = n_devices;
        p.page_size = page_size;
        p.npages = q16_pt_npages;
        p.num_blocks = q16_pt_num_blocks;
        p.d_dict_keys = d_dict_keys;
        p.d_dict_type_ids = d_dict_type_ids;
        p.d_dict_strs = d_dict_strs;
        p.d_dict_lens = d_dict_lens;
        p.d_type_id_counter = d_type_id_counter;
        p.d_type_ids = d_type_ids;
        bam_fused_q16pt_run_async(q16_pt_ctx, p, stream_pt);
        s_kernel_launches++;
    }
    CUDA_CHECK(cudaEventRecord(ev_pt_end, stream_pt));

    // ── Phase 1: Batched IO+decomp for P_PARTKEY + P_SIZE + P_BRAND ──
    constexpr uint32_t BRAND45_ID = 19;
    constexpr uint32_t P_BRAND_PADDED_LEN = 12;

    auto p1_field_start = std::chrono::steady_clock::now();

    // P_PARTKEY: fused IO+decomp+flatten INT64
    {
        BAMFusedQ16FlattenI64Params p{};
        fill_io_base(p, FI_P_PARTKEY, q16_p01_num_blocks);
        p.d_output = d_p_partkey_flat;
        bam_fused_q16p01_flatten_i64_async(q16_p01_ctx, p, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // P_SIZE: fused IO+decomp+flatten INT32 → uint32_t directly (no cast needed)
    {
        BAMFusedQ16FlattenI32Params p{};
        fill_io_base(p, FI_P_SIZE, q16_p01_num_blocks);
        p.d_output = d_p_size_u32;
        bam_fused_q16p01_flatten_i32_async(q16_p01_ctx, p, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // P_BRAND: fused IO+decomp+brand extraction
    {
        BAMFusedQ16BrandParams p{};
        fill_io_base(p, FI_P_BRAND, q16_p01_num_blocks);
        p.padded_len = P_BRAND_PADDED_LEN;
        p.d_brand_ids = d_p_brand_ids;
        bam_fused_q16p01_brand_async(q16_p01_ctx, p, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - p1_field_start).count();
        printf("[Q16-TIMING]   P1 flatten/extract: %.3f ms\n", ms);
    }

    // Wait for P_TYPE
    CUDA_CHECK(cudaStreamSynchronize(stream_pt));
    {
        float pt_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&pt_ms, ev_pt_start, ev_pt_end));
        printf("[Q16-TIMING] P_TYPE fused (stream_pt): %.3f ms\n", pt_ms);
    }

    // Read type dictionary back to host
    uint32_t num_types = 0;
    CUDA_CHECK(cudaMemcpy(&num_types, d_type_id_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "[Q16] P_TYPE distinct values (excluding MEDIUM POLISHED): " << num_types << std::endl;

    char h_dict_strs[Q16_TYPE_DICT_CAP * Q16_TYPE_STR_MAX];
    uint16_t h_dict_lens_arr[Q16_TYPE_DICT_CAP];
    uint32_t h_dict_type_ids_arr[Q16_TYPE_DICT_CAP];
    CUDA_CHECK(cudaMemcpy(h_dict_strs, d_dict_strs,
        (size_t)Q16_TYPE_DICT_CAP * Q16_TYPE_STR_MAX, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dict_lens_arr, d_dict_lens,
        Q16_TYPE_DICT_CAP * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dict_type_ids_arr, d_dict_type_ids,
        Q16_TYPE_DICT_CAP * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    char type_str_pool[150][Q16_TYPE_STR_MAX];
    uint16_t type_str_lens[150];
    memset(type_str_pool, 0, sizeof(type_str_pool));
    memset(type_str_lens, 0, sizeof(type_str_lens));
    for (uint32_t s = 0; s < Q16_TYPE_DICT_CAP; s++) {
        uint32_t tid = h_dict_type_ids_arr[s];
        if (tid != UINT32_MAX && tid < 150) {
            memcpy(type_str_pool[tid], h_dict_strs + (size_t)s * Q16_TYPE_STR_MAX,
                   h_dict_lens_arr[s]);
            type_str_lens[tid] = h_dict_lens_arr[s];
        }
    }

    // Build PART hash table
    auto ht_build_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemsetAsync(d_ht_keys, 0xFF, ht_capacity * sizeof(uint64_t), stream));
    uint64_t p_size_bitmask = (1ULL << 49) | (1ULL << 14) | (1ULL << 23) | (1ULL << 45)
                            | (1ULL << 19) | (1ULL <<  3) | (1ULL << 36) | (1ULL <<  9);
    std::cout << "[Q16] Building PART hash table..." << std::endl;
    q16_build_part_hashtable(d_p_partkey_flat, d_p_brand_ids, d_type_ids, d_p_size_u32,
        nrecs_part, p_size_bitmask, BRAND45_ID, num_types,
        d_ht_keys, d_ht_group_ids, ht_mask, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - ht_build_start).count();
        printf("[Q16-TIMING]   HT build: %.3f ms\n", ms);
    }

    auto phase1_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase1_end - total_start).count();
        printf("[Q16-TIMING] Phase 0+1 (SUPPLIER+PART): %.3f ms\n", ms);
    }


    // ── Phase 0+1 I/O statistics ──
    {
        // Phase 0: S_SUPPKEY + S_COMMENT
        for (size_t fi : {FI_S_SUPPKEY, FI_S_COMMENT}) {
            for (uint32_t pg = 0; pg < field_npages_arr[fi]; pg++) {
                total_io_count++;
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
        // Phase 1: P_TYPE + P_PARTKEY + P_BRAND + P_SIZE
        for (size_t fi : {FI_P_TYPE, FI_P_PARTKEY, FI_P_BRAND, FI_P_SIZE}) {
            for (uint32_t pg = 0; pg < field_npages_arr[fi]; pg++) {
                total_io_count++;
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
    }

    // ── 12. Phase 2+3: PARTSUPP fused I/O + decompress + probe + pipeline ──
    std::cout << "[Q16] Running fused PARTSUPP kernel..." << std::endl;

    // Pre-fill emit_pairs with UINT64_MAX sentinel
    CUDA_CHECK(cudaMemsetAsync(q16_bufs.d_emit_pairs, 0xFF,
        nrecs_partsupp * sizeof(uint64_t), stream));

    // Build fused kernel params
    BAMFusedQ16PSParams q16ps_params{};
    q16ps_params.field_start_page_ids[0] = field_start_page_ids[FI_PS_PARTKEY];
    q16ps_params.field_start_page_ids[1] = field_start_page_ids[FI_PS_SUPPKEY];
    q16ps_params.d_comp_offsets[0] = d_q16_comp_offsets[0];
    q16ps_params.d_comp_offsets[1] = d_q16_comp_offsets[1];
    q16ps_params.d_comp_sizes[0] = d_q16_comp_sizes[0];
    q16ps_params.d_comp_sizes[1] = d_q16_comp_sizes[1];
    q16ps_params.is_compressed[0] = (field_comp_methods[FI_PS_PARTKEY] != CompressionMethod::NONE);
    q16ps_params.is_compressed[1] = (field_comp_methods[FI_PS_SUPPKEY] != CompressionMethod::NONE);
    q16ps_params.d_ps = d_q16_ps;
    q16ps_params.npages = q16_ps_npages;
    for (uint32_t d = 0; d < n_devices; d++)
        q16ps_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
    q16ps_params.n_devices = n_devices;
    q16ps_params.page_size = page_size;
    q16ps_params.num_blocks = q16_num_blocks;
    q16ps_params.d_ht_keys = d_ht_keys;
    q16ps_params.d_ht_group_ids = d_ht_group_ids;
    q16ps_params.ht_mask = ht_mask;
    q16ps_params.d_excl_keys = d_excl_keys;
    q16ps_params.excl_mask = excl_mask;
    q16ps_params.d_emit_pairs = q16_bufs.d_emit_pairs;

    printf("[GIDP+BAM+DECOMP Q16] Fused PARTSUPP kernel: %u persistent blocks, %u pages\n",
           q16_num_blocks, q16_ps_npages);

    // Use split IO/decomp/probe path with independent field-page decomp (2× parallelism)
    bam_fused_q16ps_run_split_probe_async(q16_fused_ctx, q16ps_params, stream_partsupp_io, stream);
    s_kernel_launches++;
    // split_probe_async syncs both streams internally

    // Post-probe pipeline: sort → RLE → extract → sort → RLE
    std::vector<std::pair<uint32_t, uint32_t>> q16_raw_result;
    q16_post_probe_pipeline(q16_bufs, nrecs_partsupp, q16_raw_result, stream);
    s_kernel_launches++;

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

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

    // Add fused PARTSUPP kernel I/O stats
    {
        const size_t ps_fi[2] = {FI_PS_PARTKEY, FI_PS_SUPPKEY};
        for (uint32_t pg = 0; pg < q16_ps_npages; pg++) {
            for (int k = 0; k < 2; k++) {
                total_io_count++;
                if (field_comp_methods[ps_fi[k]] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[ps_fi[k]][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total rows: " << q16_result.size() << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    // Phase 0 temporaries
    cudaFree(d_excl_suppkeys);
    cudaFree(d_excl_count);
    cudaFree(d_s_suppkey_flat);
    cudaFree(d_kmp_patterns);
    cudaFree(d_kmp_next);
    cudaFree(d_kmp_offsets);
    cudaFree(d_kmp_lengths);
    // Phase 1 temporaries
    cudaFree(d_p_partkey_flat);
    cudaFree(d_p_size_u32);
    cudaFree(d_p_brand_ids);
    cudaFree(d_type_ids);
    cudaFree(d_dict_keys);
    cudaFree(d_dict_type_ids);
    cudaFree(d_dict_strs);
    cudaFree(d_dict_lens);
    cudaFree(d_type_id_counter);
    if (d_pt_comp_offsets) cudaFree(d_pt_comp_offsets);
    if (d_pt_comp_sizes) cudaFree(d_pt_comp_sizes);
    cudaFree(d_pt_prefix_sum);
    for (size_t fi : {FI_S_SUPPKEY, FI_S_COMMENT, FI_P_PARTKEY, FI_P_BRAND, FI_P_SIZE}) {
        if (d_p01_comp_offsets[fi]) cudaFree(d_p01_comp_offsets[fi]);
        if (d_p01_comp_sizes[fi]) cudaFree(d_p01_comp_sizes[fi]);
        cudaFree(d_p01_prefix_sum[fi]);
    }
    bam_fused_q16pt_destroy(q16_pt_ctx);
    bam_fused_q16p01_destroy(q16_p01_ctx);
    CUDA_CHECK(cudaStreamDestroy(stream_pt));
    CUDA_CHECK(cudaStreamDestroy(stream_partsupp_io));
    CUDA_CHECK(cudaStreamDestroy(stream_p01_io));
    CUDA_CHECK(cudaEventDestroy(ev_pt_start));
    CUDA_CHECK(cudaEventDestroy(ev_pt_end));
    // Phase 2+3 temporaries
    bam_fused_q16ps_destroy(q16_fused_ctx);
    cudaFree(d_q16_ps);
    for (int fi_idx = 0; fi_idx < 2; fi_idx++) {
        if (d_q16_comp_offsets[fi_idx]) cudaFree(d_q16_comp_offsets[fi_idx]);
        if (d_q16_comp_sizes[fi_idx]) cudaFree(d_q16_comp_sizes[fi_idx]);
    }
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
    cudaFree(d_ht_keys);
    cudaFree(d_ht_group_ids);
    cudaFree(d_excl_keys);
    cudaFree(d_p_size_tmp);
    cudaFree(d_q16_batch_ps);

    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_ctrl_close(ctrl);

    // Collect compression method string
    std::string comp_str;
    {
        std::set<std::string> methods;
        for (size_t fi = 0; fi < NUM_Q16_FIELDS; fi++)
            methods.insert(compression_method_name(field_comp_methods[fi]));
        for (const auto &m : methods) {
            if (!comp_str.empty()) comp_str += "+";
            comp_str += m;
        }
    }

    size_t total_pages = 0;
    for (size_t fi = 0; fi < NUM_Q16_FIELDS; fi++)
        total_pages += field_npages_arr[fi];

    return BenchmarkResult{
        .nios = total_io_count,
        .read_bytes = total_io_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = comp_str,
        .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// TPC-H Q5 — GIDP+BAM+DECOMP execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMPdx (device-side LZ4)
// Pipeline: 4-phase hash join (SUPPLIER → CUSTOMER → ORDERS → LINEITEM)
// ============================================================
BenchmarkResult tpch_q5(BenchmarkOptions &options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    // ── 2. Open BaM controller(s) ──
    size_t gpu_free_pre_ctrl = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_dummy);

    const uint32_t bam_num_queues = 128;
    auto ds = DataPathFusion::bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_dummy);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;
    const uint32_t n_devices = ds.n_devices;

    // Helper: read a striped page to host
    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata page via BaM ──
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    if (rc != 0) {
        std::cerr << "bam_read_page failed (metadata header)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (full metadata)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    std::cout << "=== TPCH Q5 (GIDP+BAM+DECOMP) ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;

    const uint64_t nrecs_supplier = metadata.table_supplier_nrows;
    const uint64_t nrecs_customer = metadata.table_customer_nrows;
    const uint64_t nrecs_orders   = metadata.table_orders_nrows;
    const uint64_t nrecs_lineitem = metadata.table_lineitem_nrows;
    std::cout << "nrecs: supplier=" << nrecs_supplier
              << ", customer=" << nrecs_customer
              << ", orders=" << nrecs_orders
              << ", lineitem=" << nrecs_lineitem << std::endl;

    // ── 4. Extract Q5 field info (11 fields, 4 tables) ──
    constexpr size_t FI_S_SUPPKEY   = 0;
    constexpr size_t FI_S_NATIONKEY = 1;
    constexpr size_t FI_C_CUSTKEY   = 2;
    constexpr size_t FI_C_NATIONKEY = 3;
    constexpr size_t FI_O_ORDERKEY  = 4;
    constexpr size_t FI_O_CUSTKEY   = 5;
    constexpr size_t FI_O_ORDERDATE = 6;
    constexpr size_t FI_L_ORDERKEY  = 7;
    constexpr size_t FI_L_SUPPKEY   = 8;
    constexpr size_t FI_L_EXTPRICE  = 9;
    constexpr size_t FI_L_DISCOUNT  = 10;
    constexpr size_t NUM_Q5_FIELDS  = 11;

    const size_t blocks_per_page = page_size / 512;

    const size_t col_r_regionkey = TPCH::common::R_REGIONKEY;
    const size_t col_r_name      = TPCH::common::R_NAME;
    const size_t col_n_nationkey = TPCH::common::N_NATIONKEY;
    const size_t col_n_name      = TPCH::common::N_NAME;
    const size_t col_n_regionkey = TPCH::common::N_REGIONKEY;
    const size_t col_s_suppkey   = TPCH::common::S_SUPPKEY;
    const size_t col_s_nationkey = TPCH::common::S_NATIONKEY;
    const size_t col_c_custkey   = TPCH::common::C_CUSTKEY;
    const size_t col_c_nationkey = TPCH::common::C_NATIONKEY;
    const size_t col_o_orderkey  = TPCH::common::O_ORDERKEY;
    const size_t col_o_custkey   = TPCH::common::O_CUSTKEY;
    const size_t col_o_orderdate = TPCH::common::O_ORDERDATE;
    const size_t col_l_orderkey  = TPCH::common::L_ORDERKEY;
    const size_t col_l_suppkey   = TPCH::common::L_SUPPKEY;
    const size_t col_l_extprice  = TPCH::common::L_EXTENDEDPRICE;
    const size_t col_l_discount  = TPCH::common::L_DISCOUNT;

    uint64_t field_start_page_ids[NUM_Q5_FIELDS];
    uint64_t field_npages_arr[NUM_Q5_FIELDS];
    CompressionMethod field_comp_methods[NUM_Q5_FIELDS];

    // SUPPLIER columns
    field_start_page_ids[FI_S_SUPPKEY] = metadata.table_supplier_start_page_ids[col_s_suppkey];
    field_npages_arr[FI_S_SUPPKEY]     = metadata.table_supplier_npages[col_s_suppkey];
    field_comp_methods[FI_S_SUPPKEY]   = static_cast<CompressionMethod>(
        metadata.table_supplier_compression_method[col_s_suppkey]);

    field_start_page_ids[FI_S_NATIONKEY] = metadata.table_supplier_start_page_ids[col_s_nationkey];
    field_npages_arr[FI_S_NATIONKEY]     = metadata.table_supplier_npages[col_s_nationkey];
    field_comp_methods[FI_S_NATIONKEY]   = static_cast<CompressionMethod>(
        metadata.table_supplier_compression_method[col_s_nationkey]);

    // CUSTOMER columns
    field_start_page_ids[FI_C_CUSTKEY] = metadata.table_customer_start_page_ids[col_c_custkey];
    field_npages_arr[FI_C_CUSTKEY]     = metadata.table_customer_npages[col_c_custkey];
    field_comp_methods[FI_C_CUSTKEY]   = static_cast<CompressionMethod>(
        metadata.table_customer_compression_method[col_c_custkey]);

    field_start_page_ids[FI_C_NATIONKEY] = metadata.table_customer_start_page_ids[col_c_nationkey];
    field_npages_arr[FI_C_NATIONKEY]     = metadata.table_customer_npages[col_c_nationkey];
    field_comp_methods[FI_C_NATIONKEY]   = static_cast<CompressionMethod>(
        metadata.table_customer_compression_method[col_c_nationkey]);

    // ORDERS columns
    field_start_page_ids[FI_O_ORDERKEY] = metadata.table_orders_start_page_ids[col_o_orderkey];
    field_npages_arr[FI_O_ORDERKEY]     = metadata.table_orders_npages[col_o_orderkey];
    field_comp_methods[FI_O_ORDERKEY]   = static_cast<CompressionMethod>(
        metadata.table_orders_compression_method[col_o_orderkey]);

    field_start_page_ids[FI_O_CUSTKEY] = metadata.table_orders_start_page_ids[col_o_custkey];
    field_npages_arr[FI_O_CUSTKEY]     = metadata.table_orders_npages[col_o_custkey];
    field_comp_methods[FI_O_CUSTKEY]   = static_cast<CompressionMethod>(
        metadata.table_orders_compression_method[col_o_custkey]);

    field_start_page_ids[FI_O_ORDERDATE] = metadata.table_orders_start_page_ids[col_o_orderdate];
    field_npages_arr[FI_O_ORDERDATE]     = metadata.table_orders_npages[col_o_orderdate];
    field_comp_methods[FI_O_ORDERDATE]   = static_cast<CompressionMethod>(
        metadata.table_orders_compression_method[col_o_orderdate]);

    // LINEITEM columns
    field_start_page_ids[FI_L_ORDERKEY] = metadata.table_lineitem_start_page_ids[col_l_orderkey];
    field_npages_arr[FI_L_ORDERKEY]     = metadata.table_lineitem_npages[col_l_orderkey];
    field_comp_methods[FI_L_ORDERKEY]   = static_cast<CompressionMethod>(
        metadata.table_lineitem_compression_method[col_l_orderkey]);

    field_start_page_ids[FI_L_SUPPKEY] = metadata.table_lineitem_start_page_ids[col_l_suppkey];
    field_npages_arr[FI_L_SUPPKEY]     = metadata.table_lineitem_npages[col_l_suppkey];
    field_comp_methods[FI_L_SUPPKEY]   = static_cast<CompressionMethod>(
        metadata.table_lineitem_compression_method[col_l_suppkey]);

    field_start_page_ids[FI_L_EXTPRICE] = metadata.table_lineitem_start_page_ids[col_l_extprice];
    field_npages_arr[FI_L_EXTPRICE]     = metadata.table_lineitem_npages[col_l_extprice];
    field_comp_methods[FI_L_EXTPRICE]   = static_cast<CompressionMethod>(
        metadata.table_lineitem_compression_method[col_l_extprice]);

    field_start_page_ids[FI_L_DISCOUNT] = metadata.table_lineitem_start_page_ids[col_l_discount];
    field_npages_arr[FI_L_DISCOUNT]     = metadata.table_lineitem_npages[col_l_discount];
    field_comp_methods[FI_L_DISCOUNT]   = static_cast<CompressionMethod>(
        metadata.table_lineitem_compression_method[col_l_discount]);

    const char *field_names[NUM_Q5_FIELDS] = {
        "S_SUPPKEY", "S_NATIONKEY", "C_CUSTKEY", "C_NATIONKEY",
        "O_ORDERKEY", "O_CUSTKEY", "O_ORDERDATE",
        "L_ORDERKEY", "L_SUPPKEY", "L_EXTENDEDPRICE", "L_DISCOUNT" };
    for (size_t fi = 0; fi < NUM_Q5_FIELDS; fi++) {
        std::cout << "  " << field_names[fi]
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << static_cast<int>(field_comp_methods[fi])
                  << std::endl;
    }

    // Validate: device-side decomp only supports LZ4
    for (size_t fi = 0; fi < NUM_Q5_FIELDS; fi++) {
        if (field_comp_methods[fi] != CompressionMethod::NONE &&
            field_comp_methods[fi] != CompressionMethod::LZ4) {
            std::cerr << "gidp+bam+fusion only supports LZ4 or NONE compression, "
                      << "field " << field_names[fi] << " uses method "
                      << static_cast<int>(field_comp_methods[fi]) << std::endl;
            bam_ctrl_close(ctrl);
            exit(EXIT_FAILURE);
        }
    }

    // ── 5. Read compression metadata ──
    std::vector<uint32_t> h_comp_sizes[NUM_Q5_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_Q5_FIELDS];

    auto get_comp_meta = [&](size_t fi, uint64_t &cs_start, uint64_t &cs_npages_cnt,
                             uint64_t &nbase, uint64_t &base_start) {
        switch (fi) {
        case FI_S_SUPPKEY: case FI_S_NATIONKEY: {
            size_t col = (fi == FI_S_SUPPKEY) ? col_s_suppkey : col_s_nationkey;
            cs_start      = metadata.table_supplier_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_supplier_compressed_page_sizes_npages[col];
            nbase         = metadata.table_supplier_compression_nbases[col];
            base_start    = metadata.table_supplier_compression_base_start_page_ids[col];
            break;
        }
        case FI_C_CUSTKEY: case FI_C_NATIONKEY: {
            size_t col = (fi == FI_C_CUSTKEY) ? col_c_custkey : col_c_nationkey;
            cs_start      = metadata.table_customer_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_customer_compressed_page_sizes_npages[col];
            nbase         = metadata.table_customer_compression_nbases[col];
            base_start    = metadata.table_customer_compression_base_start_page_ids[col];
            break;
        }
        case FI_O_ORDERKEY: case FI_O_CUSTKEY: case FI_O_ORDERDATE: {
            size_t col;
            switch (fi) {
            case FI_O_ORDERKEY:  col = col_o_orderkey; break;
            case FI_O_CUSTKEY:   col = col_o_custkey; break;
            default:             col = col_o_orderdate; break;
            }
            cs_start      = metadata.table_orders_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_orders_compressed_page_sizes_npages[col];
            nbase         = metadata.table_orders_compression_nbases[col];
            base_start    = metadata.table_orders_compression_base_start_page_ids[col];
            break;
        }
        case FI_L_ORDERKEY: case FI_L_SUPPKEY: case FI_L_EXTPRICE: case FI_L_DISCOUNT: {
            size_t col;
            switch (fi) {
            case FI_L_ORDERKEY:  col = col_l_orderkey; break;
            case FI_L_SUPPKEY:   col = col_l_suppkey; break;
            case FI_L_EXTPRICE:  col = col_l_extprice; break;
            default:             col = col_l_discount; break;
            }
            cs_start      = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_lineitem_compressed_page_sizes_npages[col];
            nbase         = metadata.table_lineitem_compression_nbases[col];
            base_start    = metadata.table_lineitem_compression_base_start_page_ids[col];
            break;
        }
        }
    };

    for (size_t fi = 0; fi < NUM_Q5_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;

        uint64_t cs_start, cs_npages_cnt, nbase, base_start;
        get_comp_meta(fi, cs_start, cs_npages_cnt, nbase, base_start);

        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_sizes " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        h_comp_sizes[fi].assign(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages_arr[fi]);

        size_t bp_npages = TPCH::nbase_to_npages(nbase, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_bases " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            h_comp_sizes[fi].data(), nbase, field_npages_arr[fi], page_size,
            field_start_page_ids[fi], n_devices, offsets_vec);
        h_comp_offsets[fi] = std::move(offsets_vec);
    }

    // ── 6. Read prefix sums via BaM ──
    auto read_prefix_sum = [&](uint64_t ps_start, uint64_t ps_npages_cnt, uint64_t npages_field)
        -> std::vector<uint64_t>
    {
        std::vector<uint64_t> result_ps;
        if (ps_npages_cnt == 0) return result_ps;
        std::vector<char> ps_buf(ps_npages_cnt * page_size);
        for (uint64_t p = 0; p < ps_npages_cnt; p++) {
            rc = read_striped_page(ps_start + p, page_size, ps_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (prefix_sum)" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        auto *raw = reinterpret_cast<uint64_t*>(ps_buf.data());
        result_ps.assign(raw + 1, raw + 1 + npages_field);
        return result_ps;
    };

    auto get_ps_meta = [&](size_t fi, uint64_t &ps_start, uint64_t &ps_npages_cnt) {
        switch (fi) {
        case FI_S_SUPPKEY: case FI_S_NATIONKEY: {
            size_t col = (fi == FI_S_SUPPKEY) ? col_s_suppkey : col_s_nationkey;
            ps_start      = metadata.table_supplier_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_supplier_prefix_sum_npages[col];
            break;
        }
        case FI_C_CUSTKEY: case FI_C_NATIONKEY: {
            size_t col = (fi == FI_C_CUSTKEY) ? col_c_custkey : col_c_nationkey;
            ps_start      = metadata.table_customer_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_customer_prefix_sum_npages[col];
            break;
        }
        case FI_O_ORDERKEY: case FI_O_CUSTKEY: case FI_O_ORDERDATE: {
            size_t col;
            switch (fi) {
            case FI_O_ORDERKEY:  col = col_o_orderkey; break;
            case FI_O_CUSTKEY:   col = col_o_custkey; break;
            default:             col = col_o_orderdate; break;
            }
            ps_start      = metadata.table_orders_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_orders_prefix_sum_npages[col];
            break;
        }
        case FI_L_ORDERKEY: case FI_L_SUPPKEY: case FI_L_EXTPRICE: case FI_L_DISCOUNT: {
            size_t col;
            switch (fi) {
            case FI_L_ORDERKEY:  col = col_l_orderkey; break;
            case FI_L_SUPPKEY:   col = col_l_suppkey; break;
            case FI_L_EXTPRICE:  col = col_l_extprice; break;
            default:             col = col_l_discount; break;
            }
            ps_start      = metadata.table_lineitem_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_lineitem_prefix_sum_npages[col];
            break;
        }
        }
    };

    std::vector<uint64_t> h_prefix_sum[NUM_Q5_FIELDS];
    for (size_t fi = 0; fi < NUM_Q5_FIELDS; fi++) {
        uint64_t ps_start, ps_npages_cnt;
        get_ps_meta(fi, ps_start, ps_npages_cnt);
        h_prefix_sum[fi] = read_prefix_sum(ps_start, ps_npages_cnt, field_npages_arr[fi]);
    }

    // ── 7. Persistent GPU structures (hash tables, revenue, nationkey_to_idx) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // SUPPLIER hash table
    uint32_t ht_supp_cap = 1;
    while (ht_supp_cap < nrecs_supplier) ht_supp_cap <<= 1;
    uint32_t ht_supp_mask = ht_supp_cap - 1;
    uint64_t *d_ht_supp_keys = nullptr;
    int32_t  *d_ht_supp_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_supp_keys, ht_supp_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_supp_values, ht_supp_cap * sizeof(int32_t)));

    // CUSTOMER hash table
    uint32_t ht_cust_cap = 1;
    while (ht_cust_cap < nrecs_customer) ht_cust_cap <<= 1;
    uint32_t ht_cust_mask = ht_cust_cap - 1;
    uint64_t *d_ht_cust_keys = nullptr;
    int32_t  *d_ht_cust_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_cust_keys, ht_cust_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_cust_values, ht_cust_cap * sizeof(int32_t)));

    // ORDERS hash table
    uint64_t est_ord_qual = std::max((uint64_t)1024, nrecs_orders / 15);
    uint32_t ht_ord_cap = 1;
    while (ht_ord_cap < est_ord_qual * 2) ht_ord_cap <<= 1;
    uint32_t ht_ord_mask = ht_ord_cap - 1;
    uint64_t *d_ht_ord_keys = nullptr;
    int32_t  *d_ht_ord_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_ord_keys, ht_ord_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_ord_values, ht_ord_cap * sizeof(int32_t)));

    int64_t *d_revenue = nullptr;
    CUDA_CHECK(cudaMalloc(&d_revenue, 25 * sizeof(int64_t)));

    int8_t *d_nationkey_to_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nationkey_to_idx, 25));
    int32_t *d_asia_regionkey = nullptr;
    CUDA_CHECK(cudaMalloc(&d_asia_regionkey, sizeof(int32_t)));

    const uint64_t npages_l_i32 = field_npages_arr[FI_L_EXTPRICE];
    const uint64_t npages_l_i64 = field_npages_arr[FI_L_ORDERKEY];

    uint64_t gpu_app_bytes = 0;

    // ── 8. Streaming infrastructure ──
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    constexpr size_t Q5_BATCH_PAGES = 512;

    // LINEITEM tile metrics: INT32 tile → INT64 page range
    std::vector<uint64_t> ps_l_i32_full(npages_l_i32 + 1);
    ps_l_i32_full[0] = 0;
    for (uint64_t i = 0; i < npages_l_i32; i++)
        ps_l_i32_full[i + 1] = h_prefix_sum[FI_L_EXTPRICE][i];

    std::vector<uint64_t> ps_l_i64_full(npages_l_i64 + 1);
    ps_l_i64_full[0] = 0;
    for (uint64_t i = 0; i < npages_l_i64; i++)
        ps_l_i64_full[i + 1] = h_prefix_sum[FI_L_ORDERKEY][i];

    // ORDERS prefix sums (needed for zone map INT32→INT64 mapping)
    const uint64_t npages_o_i64 = field_npages_arr[FI_O_ORDERKEY];
    const uint64_t npages_o_i32 = field_npages_arr[FI_O_ORDERDATE];
    std::vector<uint64_t> ps_o_i32_full(npages_o_i32 + 1);
    ps_o_i32_full[0] = 0;
    for (uint64_t i = 0; i < npages_o_i32; i++)
        ps_o_i32_full[i + 1] = h_prefix_sum[FI_O_ORDERDATE][i];
    std::vector<uint64_t> ps_o_i64_full(npages_o_i64 + 1);
    ps_o_i64_full[0] = 0;
    for (uint64_t i = 0; i < npages_o_i64; i++)
        ps_o_i64_full[i + 1] = h_prefix_sum[FI_O_ORDERKEY][i];

    const int32_t date_low  = options.q6_sd_low  ? options.q6_sd_low  : 19940101;
    const int32_t date_high = options.q6_sd_high ? options.q6_sd_high : 19950101;

    // Staging buffers: sized for Phase 1+2 mega-batch or Phase 3 ORDERS batch
    const uint64_t npages_s_total = field_npages_arr[FI_S_SUPPKEY] + field_npages_arr[FI_S_NATIONKEY];
    const uint64_t npages_c_total = field_npages_arr[FI_C_CUSTKEY] + field_npages_arr[FI_C_NATIONKEY];
    const size_t max_staging_pages = std::max({npages_s_total, npages_c_total, (uint64_t)Q5_BATCH_PAGES});

    void *staging_data = mb_cuda_alloc(max_staging_pages * page_size);
    void *staging_io   = mb_cuda_alloc(max_staging_pages * page_size);
    uint64_t *d_batch_ps = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_ps, max_staging_pages * sizeof(uint64_t)));
    std::vector<uint64_t> bps(max_staging_pages);

    // Zone map mask buffer
    uint8_t *d_page_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_page_mask, Q5_BATCH_PAGES));

    // Persistent BaM context for batch reads (Phases 1-3, no cudaMalloc/cudaFree per call)
    BamBulkReadCtx io_ctx_batch = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(sm_count),
        static_cast<uint32_t>(max_staging_pages));

    // Pre-allocated decomp metadata buffers (no cudaMalloc/cudaFree during measurement)
    uint32_t *d_decomp_scs = nullptr, *d_decomp_spi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_decomp_scs, max_staging_pages * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_decomp_spi, max_staging_pages * sizeof(uint32_t)));

    // ── Warp-spec Q5 LINEITEM/ORDERS setup ──
    uint32_t ws_q5li_max = q5li_warp_spec_max_blocks(static_cast<uint32_t>(page_size));
    uint32_t ws_q5li_num_blocks = std::min(static_cast<uint32_t>(npages_l_i32), ws_q5li_max);
    uint32_t ws_q5ord_max = q5ord_warp_spec_max_blocks(static_cast<uint32_t>(page_size));
    uint32_t ws_q5ord_num_blocks = std::min(static_cast<uint32_t>(npages_o_i32), ws_q5ord_max);

    // Shared page_cache + decomp_buf for ORDERS (Phase 3) and LINEITEM (Phase 4).
    // Both phases run sequentially, so the same GPU memory is reused.
    constexpr uint32_t Q5ORD_WS_SLOTS_PER_BLOCK = 98;   // 2 * 7 * 7
    constexpr uint32_t Q5LI_WS_SLOTS_PER_BLOCK  = 112;  // 2 * 7 * 8
    uint32_t q5_ord_slots = ws_q5ord_num_blocks * Q5ORD_WS_SLOTS_PER_BLOCK;
    uint32_t q5_li_slots  = ws_q5li_num_blocks * Q5LI_WS_SLOTS_PER_BLOCK;
    uint32_t q5_shared_slots = std::max(q5_ord_slots, q5_li_slots);

    bam_io_page_cache_t q5_ws_pc = nullptr;
    void*       q5_ws_ctrls    = nullptr;
    void*       q5_ws_pc_ptr   = nullptr;
    const char* q5_ws_pc_base  = nullptr;
    char*       q5_ws_decomp   = nullptr;
    // Forward declarations; actual allocation after zone map (needs budget check)

    // Upload LINEITEM prefix sums to GPU
    uint64_t *d_ps_li_i32 = nullptr, *d_ps_li_i64 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_li_i32, (npages_l_i32 + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_li_i32, ps_l_i32_full.data(),
                (npages_l_i32 + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ps_li_i64, (npages_l_i64 + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_li_i64, ps_l_i64_full.data(),
                (npages_l_i64 + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Upload LINEITEM compression metadata to GPU
    // INT32 fields: FI_L_EXTPRICE=9, FI_L_DISCOUNT=10
    uint64_t* d_li_comp_offsets_i32[2] = {};
    uint32_t* d_li_comp_sizes_i32[2] = {};
    const size_t li_i32_fis[2] = {FI_L_EXTPRICE, FI_L_DISCOUNT};
    for (int k = 0; k < 2; k++) {
        size_t fi = li_i32_fis[k];
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        CUDA_CHECK(cudaMalloc(&d_li_comp_offsets_i32[k], npages_l_i32 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_li_comp_offsets_i32[k], h_comp_offsets[fi].data(),
                    npages_l_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_li_comp_sizes_i32[k], npages_l_i32 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_li_comp_sizes_i32[k], h_comp_sizes[fi].data(),
                    npages_l_i32 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    // INT64 fields: FI_L_ORDERKEY=7, FI_L_SUPPKEY=8
    uint64_t* d_li_comp_offsets_i64[2] = {};
    uint32_t* d_li_comp_sizes_i64[2] = {};
    const size_t li_i64_fis[2] = {FI_L_ORDERKEY, FI_L_SUPPKEY};
    for (int k = 0; k < 2; k++) {
        size_t fi = li_i64_fis[k];
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        CUDA_CHECK(cudaMalloc(&d_li_comp_offsets_i64[k], npages_l_i64 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_li_comp_offsets_i64[k], h_comp_offsets[fi].data(),
                    npages_l_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_li_comp_sizes_i64[k], npages_l_i64 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_li_comp_sizes_i64[k], h_comp_sizes[fi].data(),
                    npages_l_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    uint64_t total_io_count = 0, total_io_bytes = 0;

    // Batch read using persistent BaM context (no per-call page_cache create/destroy)
    auto batch_read_lz4_field = [&](size_t fi, size_t pg_start, size_t pg_count) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        size_t io_offset = 0;
        for (size_t pg = pg_start; pg < pg_start + pg_count; pg++) {
            BamBulkReadDesc &desc = io_ctx_batch.h_descs[0][pg - pg_start];
            desc = {};
            if (is_compressed) {
                uint64_t byte_offset = h_comp_offsets[fi][pg];
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                desc.nblocks = (roundup4096(comp_sz) + 511) / 512;
                constexpr uint32_t CTRL_PAGE_BLOCKS = 4096 / 512;
                if (desc.nblocks > CTRL_PAGE_BLOCKS && desc.nblocks <= 2 * CTRL_PAGE_BLOCKS)
                    desc.nblocks = 3 * CTRL_PAGE_BLOCKS;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_io) + io_offset;
                desc.copy_bytes = comp_sz;
                io_offset += page_size;
            } else {
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                uint64_t local_pg = page_id / n_devices;
                desc.lba = ds.partition_start_lbas[dev] + local_pg * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_data) + (pg - pg_start) * page_size;
                desc.copy_bytes = page_size;
            }
        }
        bam_bulk_read_async(io_ctx_batch, static_cast<uint32_t>(pg_count), 0, stream);
        s_kernel_launches++;
        for (size_t i = 0; i < pg_count; i++) {
            total_io_count++;
            total_io_bytes += static_cast<uint64_t>(io_ctx_batch.h_descs[0][i].nblocks) * 512;
        }
        if (is_compressed) {
            std::vector<uint32_t> h_scs(pg_count), h_spi(pg_count);
            for (size_t i = 0; i < pg_count; i++) {
                h_scs[i] = h_comp_sizes[fi][pg_start + i];
                h_spi[i] = static_cast<uint32_t>(i);
            }
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_scs, h_scs.data(), pg_count * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_spi, h_spi.data(), pg_count * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, stream));
            bam_lz4_batch_decompress(
                static_cast<const char*>(staging_io),
                static_cast<char*>(staging_data),
                d_decomp_scs, d_decomp_spi,
                static_cast<uint32_t>(pg_count),
                static_cast<uint32_t>(page_size), stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    auto upload_batch_ps = [&](const std::vector<uint64_t> &h_ps,
                               size_t pg_start, size_t pg_count) {
        uint64_t base = (pg_start == 0) ? 0 : h_ps[pg_start - 1];
        for (size_t i = 0; i < pg_count; i++)
            bps[i] = h_ps[pg_start + i] - base;
        CUDA_CHECK(cudaMemcpyAsync(d_batch_ps, bps.data(),
                                   pg_count * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, stream));
    };

    auto batch_flatten_int64 = [&](size_t fi, const std::vector<uint64_t> &h_ps,
                                   uint64_t total_nrecs, uint64_t *d_flat) {
        size_t npg = field_npages_arr[fi];
        for (size_t pg = 0; pg < npg; pg += Q5_BATCH_PAGES) {
            size_t bnp = std::min(Q5_BATCH_PAGES, npg - pg);
            batch_read_lz4_field(fi, pg, bnp);
            upload_batch_ps(h_ps, pg, bnp);
            uint64_t row_start = (pg == 0) ? 0 : h_ps[pg - 1];
            uint64_t batch_nrecs = h_ps[pg + bnp - 1] - row_start;
            q13_flatten_int64_pages_ps(
                static_cast<const char*>(staging_data),
                page_size, d_batch_ps, bnp,
                batch_nrecs, d_flat + row_start, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    auto batch_flatten_int32 = [&](size_t fi, const std::vector<uint64_t> &h_ps,
                                   uint64_t total_nrecs, uint64_t *d_flat) {
        size_t npg = field_npages_arr[fi];
        for (size_t pg = 0; pg < npg; pg += Q5_BATCH_PAGES) {
            size_t bnp = std::min(Q5_BATCH_PAGES, npg - pg);
            batch_read_lz4_field(fi, pg, bnp);
            upload_batch_ps(h_ps, pg, bnp);
            uint64_t row_start = (pg == 0) ? 0 : h_ps[pg - 1];
            uint64_t batch_nrecs = h_ps[pg + bnp - 1] - row_start;
            q13_flatten_int32_pages_ps(
                static_cast<const char*>(staging_data),
                page_size, d_batch_ps, bnp,
                batch_nrecs, d_flat + row_start, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // Selective batch read using persistent BaM context
    auto batch_read_lz4_field_selective = [&](size_t fi, size_t batch_start,
                                              const std::vector<uint32_t> &active_pages) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        size_t seq_idx = 0;
        for (uint32_t pg : active_pages) {
            size_t slot = pg - batch_start;
            BamBulkReadDesc &desc = io_ctx_batch.h_descs[0][seq_idx];
            desc = {};
            if (is_compressed) {
                uint64_t byte_offset = h_comp_offsets[fi][pg];
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                desc.nblocks = (roundup4096(comp_sz) + 511) / 512;
                constexpr uint32_t CTRL_PAGE_BLOCKS = 4096 / 512;
                if (desc.nblocks > CTRL_PAGE_BLOCKS && desc.nblocks <= 2 * CTRL_PAGE_BLOCKS)
                    desc.nblocks = 3 * CTRL_PAGE_BLOCKS;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_io) + seq_idx * page_size;
                desc.copy_bytes = comp_sz;
            } else {
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                uint64_t local_pg = page_id / n_devices;
                desc.lba = ds.partition_start_lbas[dev] + local_pg * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_data) + slot * page_size;
                desc.copy_bytes = page_size;
            }
            seq_idx++;
        }
        if (!active_pages.empty()) {
            bam_bulk_read_async(io_ctx_batch, static_cast<uint32_t>(active_pages.size()), 0, stream);
            s_kernel_launches++;
            for (size_t i = 0; i < active_pages.size(); i++) {
                total_io_count++;
                total_io_bytes += static_cast<uint64_t>(io_ctx_batch.h_descs[0][i].nblocks) * 512;
            }
        }
        if (is_compressed && !active_pages.empty()) {
            std::vector<uint32_t> h_scs(active_pages.size()), h_spi(active_pages.size());
            for (size_t i = 0; i < active_pages.size(); i++) {
                h_scs[i] = h_comp_sizes[fi][active_pages[i]];
                h_spi[i] = static_cast<uint32_t>(active_pages[i] - batch_start);
            }
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_scs, h_scs.data(), active_pages.size() * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_spi, h_spi.data(), active_pages.size() * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, stream));
            bam_lz4_batch_decompress(
                static_cast<const char*>(staging_io),
                static_cast<char*>(staging_data),
                d_decomp_scs, d_decomp_spi,
                static_cast<uint32_t>(active_pages.size()),
                static_cast<uint32_t>(page_size), stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // ── Zone map: pre-compute per-page active flags ──
    const bool enable_zonemap = options.enable_zonemap;
    // ── GPU-side zone map setup (metadata extraction outside timing) ──
    bam_io_page_cache_t zm_pc = nullptr;
    BamZonemapCtx zm_ord_ctx{}, zm_li_ctx{};
    uint32_t zm_ord_nreads = 0, zm_ord_npreds = 0;
    uint32_t zm_li_nreads = 0, zm_li_npreds = 0;
    bool zm_ord_valid = false, zm_li_valid = false;

    if (enable_zonemap) {
        zm_pc = bam_io_page_cache_create(ctrl, page_size, kBamZonemapMaxReads);
        void* zm_d_ctrls = bam_io_page_cache_get_d_ctrls(zm_pc);
        void* zm_d_pc_ptr = bam_io_page_cache_get_d_pc_ptr(zm_pc);
        const char* zm_pc_base = (const char*)bam_io_page_cache_get_base_addr(zm_pc);

        // ORDERS zone map: O_ORDERDATE + R_NAME sideways
        zm_ord_ctx = bam_zonemap_ctx_create(zm_d_ctrls, zm_d_pc_ptr, (void*)zm_pc_base,
            static_cast<uint32_t>(page_size), npages_o_i32);

        {
            const size_t o_odate_field = TPCH::common::O_ORDERDATE;
            uint64_t odate_nstats     = metadata.table_orders_nstats[o_odate_field];
            uint64_t odate_stats_start = metadata.table_orders_stats_start_page_ids[o_odate_field];
            uint64_t odate_stats_npg   = metadata.table_orders_stats_npages[o_odate_field];
            if (odate_nstats > 0 && odate_stats_start > 0) {
                for (uint64_t j = 0; j < odate_stats_npg; j++) {
                    uint64_t pg_id = odate_stats_start + j;
                    uint32_t dev = pg_id % n_devices;
                    uint64_t local = pg_id / n_devices;
                    zm_ord_ctx.h_reads[zm_ord_nreads++] = {
                        ds.partition_start_lbas[dev] + local * blocks_per_page,
                        static_cast<uint32_t>(blocks_per_page), dev};
                }
                zm_ord_ctx.h_preds[zm_ord_npreds++] = {0, odate_nstats,
                    date_low, date_high - 1};
            }

            const size_t sw_rname_idx = TPCH::common::OS_SIDEWAYS_R_NAME;
            const int32_t asia_dict_id = 2;
            uint64_t rname_nstats      = metadata.table_orders_sideways_nstats[o_odate_field][sw_rname_idx];
            uint64_t rname_stats_start = metadata.table_orders_sideways_stats_start_page_ids[o_odate_field][sw_rname_idx];
            uint64_t rname_stats_npg   = metadata.table_orders_sideways_stats_npages[o_odate_field][sw_rname_idx];
            if (rname_nstats > 0 && rname_stats_start > 0) {
                uint32_t stats_page_offset = zm_ord_nreads;
                for (uint64_t j = 0; j < rname_stats_npg; j++) {
                    uint64_t pg_id = rname_stats_start + j;
                    uint32_t dev = pg_id % n_devices;
                    uint64_t local = pg_id / n_devices;
                    zm_ord_ctx.h_reads[zm_ord_nreads++] = {
                        ds.partition_start_lbas[dev] + local * blocks_per_page,
                        static_cast<uint32_t>(blocks_per_page), dev};
                }
                zm_ord_ctx.h_preds[zm_ord_npreds++] = {stats_page_offset, rname_nstats,
                    asia_dict_id, asia_dict_id};
            }
        }
        zm_ord_valid = (zm_ord_npreds > 0);

        // LINEITEM zone map: sideways O_ORDERDATE + sideways R_NAME
        zm_li_ctx = bam_zonemap_ctx_create(zm_d_ctrls, zm_d_pc_ptr, (void*)zm_pc_base,
            static_cast<uint32_t>(page_size), npages_l_i32);

        {
            const size_t li_ref_field = TPCH::common::L_EXTENDEDPRICE;
            const size_t li_sw_odate_idx = TPCH::common::LS_SIDEWAYS_O_ORDERDATE;
            uint64_t li_odate_nstats      = metadata.table_lineitem_sideways_nstats[li_ref_field][li_sw_odate_idx];
            uint64_t li_odate_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[li_ref_field][li_sw_odate_idx];
            uint64_t li_odate_stats_npg   = metadata.table_lineitem_sideways_stats_npages[li_ref_field][li_sw_odate_idx];
            if (li_odate_nstats > 0 && li_odate_stats_start > 0) {
                for (uint64_t j = 0; j < li_odate_stats_npg; j++) {
                    uint64_t pg_id = li_odate_stats_start + j;
                    uint32_t dev = pg_id % n_devices;
                    uint64_t local = pg_id / n_devices;
                    zm_li_ctx.h_reads[zm_li_nreads++] = {
                        ds.partition_start_lbas[dev] + local * blocks_per_page,
                        static_cast<uint32_t>(blocks_per_page), dev};
                }
                zm_li_ctx.h_preds[zm_li_npreds++] = {0, li_odate_nstats,
                    date_low, date_high - 1};
            }

            const size_t li_sw_rname_idx = TPCH::common::LS_SIDEWAYS_R_NAME;
            const int32_t asia_dict_id = 2;
            uint64_t li_rname_nstats      = metadata.table_lineitem_sideways_nstats[li_ref_field][li_sw_rname_idx];
            uint64_t li_rname_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[li_ref_field][li_sw_rname_idx];
            uint64_t li_rname_stats_npg   = metadata.table_lineitem_sideways_stats_npages[li_ref_field][li_sw_rname_idx];
            if (li_rname_nstats > 0 && li_rname_stats_start > 0) {
                uint32_t stats_page_offset = zm_li_nreads;
                for (uint64_t j = 0; j < li_rname_stats_npg; j++) {
                    uint64_t pg_id = li_rname_stats_start + j;
                    uint32_t dev = pg_id % n_devices;
                    uint64_t local = pg_id / n_devices;
                    zm_li_ctx.h_reads[zm_li_nreads++] = {
                        ds.partition_start_lbas[dev] + local * blocks_per_page,
                        static_cast<uint32_t>(blocks_per_page), dev};
                }
                zm_li_ctx.h_preds[zm_li_npreds++] = {stats_page_offset, li_rname_nstats,
                    asia_dict_id, asia_dict_id};
            }
        }
        zm_li_valid = (zm_li_npreds > 0);
    }

    // Phase 3: ORDERS — upload compression metadata + prefix sums to GPU
    uint64_t *d_ps_o_i32 = nullptr, *d_ps_o_i64 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_o_i32, (npages_o_i32 + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_o_i32, ps_o_i32_full.data(),
                (npages_o_i32 + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ps_o_i64, (npages_o_i64 + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_o_i64, ps_o_i64_full.data(),
                (npages_o_i64 + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // ORDERS compression metadata
    uint64_t* d_o_comp_offsets_i32 = nullptr;
    uint32_t* d_o_comp_sizes_i32 = nullptr;
    {
        size_t fi = FI_O_ORDERDATE;
        if (field_comp_methods[fi] != CompressionMethod::NONE) {
            CUDA_CHECK(cudaMalloc(&d_o_comp_offsets_i32, npages_o_i32 * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_o_comp_offsets_i32, h_comp_offsets[fi].data(),
                        npages_o_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_o_comp_sizes_i32, npages_o_i32 * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_o_comp_sizes_i32, h_comp_sizes[fi].data(),
                        npages_o_i32 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }
    uint64_t* d_o_comp_offsets_i64[2] = {};
    uint32_t* d_o_comp_sizes_i64[2] = {};
    {
        const size_t o_i64_fis[2] = {FI_O_ORDERKEY, FI_O_CUSTKEY};
        for (int k = 0; k < 2; k++) {
            size_t fi = o_i64_fis[k];
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            CUDA_CHECK(cudaMalloc(&d_o_comp_offsets_i64[k], npages_o_i64 * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_o_comp_offsets_i64[k], h_comp_offsets[fi].data(),
                        npages_o_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_o_comp_sizes_i64[k], npages_o_i64 * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_o_comp_sizes_i64[k], h_comp_sizes[fi].data(),
                        npages_o_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }

    // ── Create shared warp-spec page_cache + decomp_buf (ORDERS + LINEITEM) ──
    q5_ws_pc = bam_io_page_cache_create(
        ctrl, static_cast<uint32_t>(page_size), q5_shared_slots);
    q5_ws_ctrls   = bam_io_page_cache_get_d_ctrls(q5_ws_pc);
    q5_ws_pc_ptr  = bam_io_page_cache_get_d_pc_ptr(q5_ws_pc);
    q5_ws_pc_base = (const char*)bam_io_page_cache_get_base_addr(q5_ws_pc);
    {
        size_t decomp_size = (size_t)q5_shared_slots * page_size;
        CUDA_CHECK(cudaMalloc(&q5_ws_decomp, decomp_size));
    }

    // ── Q5 LINEITEM v2 (Q1-style balanced pipeline, fixed 108 blocks) ──
    const uint32_t q5li_v2_num_blocks = static_cast<uint32_t>(sm_count);
    bam_fused_q5li_v2_ctx_t q5li_v2_ctx = bam_fused_q5li_v2_create(
        ctrl, static_cast<uint32_t>(page_size), q5li_v2_num_blocks);

    // ── Fused dim table context (SUPPLIER + CUSTOMER fields) ──
    const size_t q5_dim_fis[] = { FI_S_SUPPKEY, FI_S_NATIONKEY, FI_C_CUSTKEY, FI_C_NATIONKEY };
    uint64_t* d_dim_comp_offsets[NUM_Q5_FIELDS] = {};
    uint32_t* d_dim_comp_sizes[NUM_Q5_FIELDS] = {};
    uint64_t* d_dim_prefix_sum[NUM_Q5_FIELDS] = {};
    for (size_t fi : q5_dim_fis) {
        uint64_t np = field_npages_arr[fi];
        CUDA_CHECK(cudaMalloc(&d_dim_prefix_sum[fi], np * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_dim_prefix_sum[fi], h_prefix_sum[fi].data(),
                    np * sizeof(uint64_t), cudaMemcpyHostToDevice));
        if (field_comp_methods[fi] != CompressionMethod::NONE) {
            std::vector<uint64_t> co64(np);
            for (uint64_t i = 0; i < np; i++)
                co64[i] = static_cast<uint64_t>(h_comp_offsets[fi][i]);
            CUDA_CHECK(cudaMalloc(&d_dim_comp_offsets[fi], np * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_dim_comp_offsets[fi], co64.data(),
                        np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_dim_comp_sizes[fi], np * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_dim_comp_sizes[fi], h_comp_sizes[fi].data(),
                        np * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }

    // ── Nocopy page_cache for dim tables (SUPPLIER + CUSTOMER shared) ──
    const uint32_t q5_dim_total_supp = static_cast<uint32_t>(
        field_npages_arr[FI_S_SUPPKEY] + field_npages_arr[FI_S_NATIONKEY]);
    const uint32_t q5_dim_total_cust = static_cast<uint32_t>(
        field_npages_arr[FI_C_CUSTKEY] + field_npages_arr[FI_C_NATIONKEY]);
    const uint32_t q5_dim_max_descs = std::max(q5_dim_total_supp, q5_dim_total_cust);

    BamBulkReadCtx q5_dim_nocopy = bam_bulk_read_nocopy_ctx_create(
        ctrl, static_cast<uint32_t>(page_size), q5_dim_max_descs, q5_dim_max_descs);

    uint32_t q5_dim_k1_max = q3_cust_io_decomp_max_blocks(static_cast<uint32_t>(page_size));
    uint32_t q5_dim_k2_max = q5_dim_process_max_blocks();
    // Staging: reuse q5_ws_decomp (12+ GB, dim tables need at most ~240 MB @ SF100)
    char* q5_dim_staging = q5_ws_decomp;

    // Measure GPU memory (includes fused contexts + active page lists)
    {
        size_t gpu_free_after_app = 0;
        cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
        gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;
    }

    // Zone map INT64 mask buffers (allocated outside timing)
    uint8_t* d_mask_ord_i64 = nullptr;
    uint8_t* d_mask_li_i64 = nullptr;
    if (zm_ord_valid || zm_li_valid) {
        if (zm_ord_valid) cudaMalloc(&d_mask_ord_i64, npages_o_i64);
        if (zm_li_valid)  cudaMalloc(&d_mask_li_i64, npages_l_i64);
    }

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_ord_valid || zm_li_valid) {
        bam_pre_io(zm_ord_ctx.d_ctrls, zm_ord_ctx.d_pc, stream);
    }

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    total_io_count = 0; total_io_bytes = 0;

    // ── Phase 0: REGION + NATION → d_nationkey_to_idx (GPU kernel) ──
    char nation_names[25][26];
    memset(nation_names, 0, sizeof(nation_names));
    int num_asia_nations = 0;
    int32_t asia_regionkey = -1;
    {
        uint64_t r_rkey_start  = metadata.table_region_start_page_ids[col_r_regionkey];
        uint64_t r_rkey_npages = metadata.table_region_npages[col_r_regionkey];
        uint64_t r_name_start  = metadata.table_region_start_page_ids[col_r_name];
        uint64_t r_name_npages = metadata.table_region_npages[col_r_name];
        uint64_t n_nkey_start  = metadata.table_nation_start_page_ids[col_n_nationkey];
        uint64_t n_nkey_npages = metadata.table_nation_npages[col_n_nationkey];
        uint64_t n_name_start  = metadata.table_nation_start_page_ids[col_n_name];
        uint64_t n_name_npages = metadata.table_nation_npages[col_n_name];
        uint64_t n_rkey_start  = metadata.table_nation_start_page_ids[col_n_regionkey];
        uint64_t n_rkey_npages = metadata.table_nation_npages[col_n_regionkey];

        // Read all pages to host (REGION for GPU upload, NATION for GPU upload + logging)
        std::vector<char> h_r_rkey(r_rkey_npages * page_size);
        std::vector<char> h_r_name(r_name_npages * page_size);
        std::vector<char> h_n_nkey(n_nkey_npages * page_size);
        std::vector<char> h_n_name(n_name_npages * page_size);
        std::vector<char> h_n_rkey(n_rkey_npages * page_size);
        for (uint64_t p = 0; p < r_rkey_npages; p++)
            read_striped_page(r_rkey_start + p, page_size, h_r_rkey.data() + p * page_size);
        for (uint64_t p = 0; p < r_name_npages; p++)
            read_striped_page(r_name_start + p, page_size, h_r_name.data() + p * page_size);
        for (uint64_t p = 0; p < n_nkey_npages; p++)
            read_striped_page(n_nkey_start + p, page_size, h_n_nkey.data() + p * page_size);
        for (uint64_t p = 0; p < n_name_npages; p++)
            read_striped_page(n_name_start + p, page_size, h_n_name.data() + p * page_size);
        for (uint64_t p = 0; p < n_rkey_npages; p++)
            read_striped_page(n_rkey_start + p, page_size, h_n_rkey.data() + p * page_size);

        // Upload R_RKEY, R_NAME, N_NKEY, N_RKEY to GPU staging
        CUDA_CHECK(cudaMemsetAsync(d_nationkey_to_idx, 0xFF, 25, stream));
        size_t off = 0;
        char* d_r_rkey_stg = q5_dim_staging + off; off += r_rkey_npages * page_size;
        char* d_r_name_stg = q5_dim_staging + off; off += r_name_npages * page_size;
        char* d_n_nkey_stg = q5_dim_staging + off; off += n_nkey_npages * page_size;
        char* d_n_rkey_stg = q5_dim_staging + off;
        CUDA_CHECK(cudaMemcpyAsync(d_r_rkey_stg, h_r_rkey.data(),
            r_rkey_npages * page_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_r_name_stg, h_r_name.data(),
            r_name_npages * page_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_n_nkey_stg, h_n_nkey.data(),
            n_nkey_npages * page_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_n_rkey_stg, h_n_rkey.data(),
            n_rkey_npages * page_size, cudaMemcpyHostToDevice, stream));

        q5_phase0_region_nation_launch(
            d_r_rkey_stg, d_r_name_stg, d_n_nkey_stg, d_n_rkey_stg,
            d_nationkey_to_idx, d_asia_regionkey, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemcpy(&asia_regionkey, d_asia_regionkey,
            sizeof(int32_t), cudaMemcpyDeviceToHost));
        std::cout << "[Q5] asia_regionkey=" << asia_regionkey << std::endl;

        // Host-side: build nation_names for logging
        uint32_t nalloc_nation = DataPathFusion::bam_host_pag_get_nalloc(h_n_nkey.data());
        for (uint32_t i = 0; i < nalloc_nation; i++) {
            int32_t n_rkey = *reinterpret_cast<const int32_t *>(
                h_n_rkey.data() + 12 + sizeof(int32_t) * i);
            if (n_rkey != asia_regionkey) continue;
            int32_t n_nkey = *reinterpret_cast<const int32_t *>(
                h_n_nkey.data() + 12 + sizeof(int32_t) * i);
            const char *n_name = DataPathFusion::bam_host_pagcol_char_data(h_n_name.data(), i, 28);
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

    // Helper: build nocopy descriptors for a dim table field
    auto fill_dim_descs = [&](BamBulkReadDesc* h_descs, uint32_t base_idx,
                              size_t fi, uint32_t npages) {
        for (uint32_t pg = 0; pg < npages; pg++) {
            BamBulkReadDesc& d = h_descs[base_idx + pg];
            uint64_t global_pg = field_start_page_ids[fi] + pg;
            d.device = global_pg % n_devices;
            if (field_comp_methods[fi] != CompressionMethod::NONE) {
                d.lba = ds.partition_start_lbas[d.device]
                    + h_comp_offsets[fi][pg] / 512;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                if (nblk > 8 && nblk <= 16) nblk = 17;
                d.nblocks = nblk;
            } else {
                uint64_t local_pg = global_pg / n_devices;
                d.lba = ds.partition_start_lbas[d.device]
                    + local_pg * blocks_per_page;
                d.nblocks = static_cast<uint32_t>(blocks_per_page);
            }
            d.dest = nullptr;
            d.copy_bytes = 0;
        }
    };

    // ═══════════════════════════════════════════════════════
    // Phase 1: SUPPLIER — nocopy batch IO+decomp → HT build
    // ═══════════════════════════════════════════════════════
    {
        const uint32_t npages_sk = static_cast<uint32_t>(field_npages_arr[FI_S_SUPPKEY]);
        const uint32_t npages_snk = static_cast<uint32_t>(field_npages_arr[FI_S_NATIONKEY]);

        // Build nocopy descriptors: S_SUPPKEY pages [0..npages_sk), S_NATIONKEY [npages_sk..total)
        fill_dim_descs(q5_dim_nocopy.h_descs[0], 0, FI_S_SUPPKEY, npages_sk);
        fill_dim_descs(q5_dim_nocopy.h_descs[0], npages_sk, FI_S_NATIONKEY, npages_snk);

        CUDA_CHECK(cudaMemcpyAsync(q5_dim_nocopy.d_descs[0], q5_dim_nocopy.h_descs[0],
            q5_dim_total_supp * sizeof(BamBulkReadDesc), cudaMemcpyHostToDevice, stream));

        // Kernel 1: IO all pages + decomp to staging
        {
            Q3CustIODecompParams io_params = {};
            io_params.total_descs = q5_dim_total_supp;
            io_params.ck_npages = npages_sk;
            io_params.ck_comp_sizes = d_dim_comp_sizes[FI_S_SUPPKEY];
            io_params.mk_comp_sizes = d_dim_comp_sizes[FI_S_NATIONKEY];
            io_params.page_size = static_cast<uint32_t>(page_size);

            uint32_t k1_blocks = std::min((q5_dim_total_supp + 3u) / 4u, q5_dim_k1_max);
            q3_cust_io_decomp_launch(
                q5_dim_nocopy.d_ctrls, q5_dim_nocopy.d_pc,
                (const char*)q5_dim_nocopy.pc_base,
                q5_dim_nocopy.d_descs[0],
                q5_dim_staging, io_params, k1_blocks, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // IO stats for SUPPLIER fields
        for (size_t fi : {FI_S_SUPPKEY, FI_S_NATIONKEY}) {
            bool compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
            for (uint64_t pg = 0; pg < field_npages_arr[fi]; pg++) {
                total_io_count++;
                if (compressed) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }

        // Kernel 2: process staging → HT build (NK-first filter)
        CUDA_CHECK(cudaMemsetAsync(d_ht_supp_keys, 0xFF, ht_supp_cap * sizeof(uint64_t), stream));
        {
            Q5DimProcessParams proc = {};
            proc.d_staging = q5_dim_staging;
            proc.page_size = static_cast<uint32_t>(page_size);
            proc.key_prefix_sum = d_dim_prefix_sum[FI_S_SUPPKEY];
            proc.key_npages = npages_sk;
            proc.nk_page_offset = npages_sk;
            proc.nk_prefix_sum = d_dim_prefix_sum[FI_S_NATIONKEY];
            proc.nk_npages = npages_snk;
            proc.nrecs = nrecs_supplier;
            proc.d_nationkey_to_idx = d_nationkey_to_idx;
            proc.d_ht_keys = d_ht_supp_keys;
            proc.d_ht_values = d_ht_supp_values;
            proc.ht_mask = ht_supp_mask;

            q5_dim_process_launch(proc, q5_dim_k2_max, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        std::cout << "[Q5] SUPPLIER HT built (capacity=" << ht_supp_cap << ")" << std::endl;
    }

    auto phase1_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase1_end - total_start).count();
        std::cout << "[Q5-TIMING] Phase 1 (SUPPLIER): " << ms << " ms" << std::endl;
    }

    // ═══════════════════════════════════════════════════════
    // Phase 2: CUSTOMER — nocopy batch IO+decomp → HT build
    // ═══════════════════════════════════════════════════════
    {
        const uint32_t npages_ck = static_cast<uint32_t>(field_npages_arr[FI_C_CUSTKEY]);
        const uint32_t npages_cnk = static_cast<uint32_t>(field_npages_arr[FI_C_NATIONKEY]);

        // Build nocopy descriptors: C_CUSTKEY pages [0..npages_ck), C_NATIONKEY [npages_ck..total)
        fill_dim_descs(q5_dim_nocopy.h_descs[0], 0, FI_C_CUSTKEY, npages_ck);
        fill_dim_descs(q5_dim_nocopy.h_descs[0], npages_ck, FI_C_NATIONKEY, npages_cnk);

        CUDA_CHECK(cudaMemcpyAsync(q5_dim_nocopy.d_descs[0], q5_dim_nocopy.h_descs[0],
            q5_dim_total_cust * sizeof(BamBulkReadDesc), cudaMemcpyHostToDevice, stream));

        // Kernel 1: IO all pages + decomp to staging
        {
            Q3CustIODecompParams io_params = {};
            io_params.total_descs = q5_dim_total_cust;
            io_params.ck_npages = npages_ck;
            io_params.ck_comp_sizes = d_dim_comp_sizes[FI_C_CUSTKEY];
            io_params.mk_comp_sizes = d_dim_comp_sizes[FI_C_NATIONKEY];
            io_params.page_size = static_cast<uint32_t>(page_size);

            uint32_t k1_blocks = std::min((q5_dim_total_cust + 3u) / 4u, q5_dim_k1_max);
            q3_cust_io_decomp_launch(
                q5_dim_nocopy.d_ctrls, q5_dim_nocopy.d_pc,
                (const char*)q5_dim_nocopy.pc_base,
                q5_dim_nocopy.d_descs[0],
                q5_dim_staging, io_params, k1_blocks, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // IO stats for CUSTOMER fields
        for (size_t fi : {FI_C_CUSTKEY, FI_C_NATIONKEY}) {
            bool compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
            for (uint64_t pg = 0; pg < field_npages_arr[fi]; pg++) {
                total_io_count++;
                if (compressed) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }

        // Kernel 2: process staging → HT build (NK-first filter)
        CUDA_CHECK(cudaMemsetAsync(d_ht_cust_keys, 0xFF, ht_cust_cap * sizeof(uint64_t), stream));
        {
            Q5DimProcessParams proc = {};
            proc.d_staging = q5_dim_staging;
            proc.page_size = static_cast<uint32_t>(page_size);
            proc.key_prefix_sum = d_dim_prefix_sum[FI_C_CUSTKEY];
            proc.key_npages = npages_ck;
            proc.nk_page_offset = npages_ck;
            proc.nk_prefix_sum = d_dim_prefix_sum[FI_C_NATIONKEY];
            proc.nk_npages = npages_cnk;
            proc.nrecs = nrecs_customer;
            proc.d_nationkey_to_idx = d_nationkey_to_idx;
            proc.d_ht_keys = d_ht_cust_keys;
            proc.d_ht_values = d_ht_cust_values;
            proc.ht_mask = ht_cust_mask;

            q5_dim_process_launch(proc, q5_dim_k2_max, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        std::cout << "[Q5] CUSTOMER HT built (capacity=" << ht_cust_cap << ")" << std::endl;
    }

    auto phase2_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase2_end - phase1_end).count();
        std::cout << "[Q5-TIMING] Phase 2 (CUSTOMER): " << ms << " ms" << std::endl;
    }

    // ── Zone map eval (GPU-side, inside timing, with INT64 mask derivation) ──
    if (zm_ord_valid) {
        zm_ord_ctx.d_ps_i32   = d_ps_o_i32 + 1;
        zm_ord_ctx.d_ps_i64   = d_ps_o_i64 + 1;
        zm_ord_ctx.d_mask_i64 = d_mask_ord_i64;
        zm_ord_ctx.npages_i64 = static_cast<uint32_t>(npages_o_i64);
        bam_zonemap_eval_async(zm_ord_ctx, npages_o_i32, zm_ord_nreads, zm_ord_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
    }
    if (zm_li_valid) {
        zm_li_ctx.d_ps_i32   = d_ps_li_i32 + 1;
        zm_li_ctx.d_ps_i64   = d_ps_li_i64 + 1;
        zm_li_ctx.d_mask_i64 = d_mask_li_i64;
        zm_li_ctx.npages_i64 = static_cast<uint32_t>(npages_l_i64);
        bam_zonemap_eval_async(zm_li_ctx, npages_l_i32, zm_li_nreads, zm_li_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
    }

    // ═══════════════════════════════════════════════════════
    // Phase 3: ORDERS — warp-spec BaM I/O + nvCOMPdx LZ4 + probe CUSTOMER + build HT
    // ═══════════════════════════════════════════════════════
    {
        // Initialize ORDERS HT
        CUDA_CHECK(cudaMemsetAsync(d_ht_ord_keys, 0xFF, ht_ord_cap * sizeof(uint64_t), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        Q5OrdWarpSpecParams ws_q5ord_params = {};
        ws_q5ord_params.i32_field_start_page_id = field_start_page_ids[FI_O_ORDERDATE];
        ws_q5ord_params.d_comp_offsets_i32 = d_o_comp_offsets_i32;
        ws_q5ord_params.d_comp_sizes_i32 = d_o_comp_sizes_i32;
        ws_q5ord_params.is_compressed_i32 = (field_comp_methods[FI_O_ORDERDATE] != CompressionMethod::NONE);
        ws_q5ord_params.i64_field_start_page_ids[0] = field_start_page_ids[FI_O_ORDERKEY];
        ws_q5ord_params.i64_field_start_page_ids[1] = field_start_page_ids[FI_O_CUSTKEY];
        for (int k = 0; k < 2; k++) {
            ws_q5ord_params.d_comp_offsets_i64[k] = d_o_comp_offsets_i64[k];
            ws_q5ord_params.d_comp_sizes_i64[k] = d_o_comp_sizes_i64[k];
        }
        ws_q5ord_params.is_compressed_i64[0] = (field_comp_methods[FI_O_ORDERKEY] != CompressionMethod::NONE);
        ws_q5ord_params.is_compressed_i64[1] = (field_comp_methods[FI_O_CUSTKEY] != CompressionMethod::NONE);
        ws_q5ord_params.d_ps_i32 = d_ps_o_i32;
        ws_q5ord_params.d_ps_i64 = d_ps_o_i64;
        ws_q5ord_params.npages_i32 = static_cast<uint32_t>(npages_o_i32);
        ws_q5ord_params.npages_i64 = static_cast<uint32_t>(npages_o_i64);
        for (uint32_t d = 0; d < n_devices && d < 4; d++)
            ws_q5ord_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        ws_q5ord_params.n_devices = n_devices;
        ws_q5ord_params.page_size = static_cast<uint32_t>(page_size);
        ws_q5ord_params.date_low = date_low;
        ws_q5ord_params.date_high = date_high;
        ws_q5ord_params.d_ht_cust_keys = d_ht_cust_keys;
        ws_q5ord_params.d_ht_cust_values = d_ht_cust_values;
        ws_q5ord_params.ht_cust_mask = ht_cust_mask;
        ws_q5ord_params.d_ht_ord_keys = d_ht_ord_keys;
        ws_q5ord_params.d_ht_ord_values = d_ht_ord_values;
        ws_q5ord_params.ht_ord_mask = ht_ord_mask;
        ws_q5ord_params.d_active_page_ids = zm_ord_valid ? zm_ord_ctx.d_active_ids : nullptr;
        ws_q5ord_params.d_page_mask = zm_ord_valid ? zm_ord_ctx.d_mask : nullptr;

        uint32_t q5ord_launch_blocks = ws_q5ord_num_blocks;
        if (zm_ord_valid) {
            ws_q5ord_params.total_pages = *zm_ord_ctx.h_num_active;
            q5ord_launch_blocks = std::min(*zm_ord_ctx.h_num_active, ws_q5ord_num_blocks);
        } else {
            ws_q5ord_params.total_pages = static_cast<uint32_t>(npages_o_i32);
        }

        std::cout << "[GIDP+BAM+WARPSPEC Q5] ORDERS kernel: "
                  << q5ord_launch_blocks << " blocks, "
                  << ws_q5ord_params.total_pages << "/" << npages_o_i32 << " active INT32 pages (mask=" << (zm_ord_valid?"ON":"OFF") << "), "
                  << npages_o_i64 << " INT64 pages" << std::endl;

        q5ord_warp_spec_launch(q5_ws_ctrls, q5_ws_pc_ptr, q5_ws_pc_base,
                               q5_ws_decomp, ws_q5ord_params, q5ord_launch_blocks, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cout << "[Q5] ORDERS HT built (capacity=" << ht_ord_cap << ")" << std::endl;
    }

    auto phase3_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase3_end - phase2_end).count();
        std::cout << "[Q5-TIMING] Phase 3 (ORDERS): " << ms << " ms" << std::endl;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 4: LINEITEM — warp-spec BaM I/O + nvCOMPdx LZ4 + probe kernel
    // ═══════════════════════════════════════════════════════════════════
    CUDA_CHECK(cudaMemsetAsync(d_revenue, 0, 25 * sizeof(int64_t), stream));

    {
        BAMFusedQ5LIV2Params v2_params = {};
        v2_params.i32_field_start_page_ids[0] = field_start_page_ids[FI_L_EXTPRICE];
        v2_params.i32_field_start_page_ids[1] = field_start_page_ids[FI_L_DISCOUNT];
        v2_params.d_comp_offsets_i32[0] = d_li_comp_offsets_i32[0];
        v2_params.d_comp_offsets_i32[1] = d_li_comp_offsets_i32[1];
        v2_params.d_comp_sizes_i32[0] = d_li_comp_sizes_i32[0];
        v2_params.d_comp_sizes_i32[1] = d_li_comp_sizes_i32[1];
        v2_params.is_compressed_i32[0] = (field_comp_methods[FI_L_EXTPRICE] != CompressionMethod::NONE);
        v2_params.is_compressed_i32[1] = (field_comp_methods[FI_L_DISCOUNT] != CompressionMethod::NONE);
        v2_params.i64_field_start_page_ids[0] = field_start_page_ids[FI_L_ORDERKEY];
        v2_params.i64_field_start_page_ids[1] = field_start_page_ids[FI_L_SUPPKEY];
        v2_params.d_comp_offsets_i64[0] = d_li_comp_offsets_i64[0];
        v2_params.d_comp_offsets_i64[1] = d_li_comp_offsets_i64[1];
        v2_params.d_comp_sizes_i64[0] = d_li_comp_sizes_i64[0];
        v2_params.d_comp_sizes_i64[1] = d_li_comp_sizes_i64[1];
        v2_params.is_compressed_i64[0] = (field_comp_methods[FI_L_ORDERKEY] != CompressionMethod::NONE);
        v2_params.is_compressed_i64[1] = (field_comp_methods[FI_L_SUPPKEY] != CompressionMethod::NONE);
        v2_params.d_ps_i32 = d_ps_li_i32;
        v2_params.d_ps_i64 = d_ps_li_i64;
        v2_params.npages_i32 = static_cast<uint32_t>(npages_l_i32);
        v2_params.npages_i64 = static_cast<uint32_t>(npages_l_i64);
        for (uint32_t d = 0; d < n_devices && d < 4; d++)
            v2_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        v2_params.n_devices = n_devices;
        v2_params.page_size = static_cast<uint32_t>(page_size);
        v2_params.num_blocks = q5li_v2_num_blocks;
        v2_params.d_ht_ord_keys = d_ht_ord_keys;
        v2_params.d_ht_ord_values = d_ht_ord_values;
        v2_params.ht_ord_mask = ht_ord_mask;
        v2_params.d_ht_supp_keys = d_ht_supp_keys;
        v2_params.d_ht_supp_values = d_ht_supp_values;
        v2_params.ht_supp_mask = ht_supp_mask;
        v2_params.d_revenue = d_revenue;
        v2_params.d_active_page_ids = zm_li_valid ? zm_li_ctx.d_active_ids : nullptr;
        v2_params.d_page_mask = zm_li_valid ? zm_li_ctx.d_mask : nullptr;
        v2_params.total_pages = zm_li_valid
            ? *zm_li_ctx.h_num_active
            : static_cast<uint32_t>(npages_l_i32);

        std::cout << "[GIDP+BAM+FUSION Q5] LINEITEM v2 kernel: "
                  << q5li_v2_num_blocks << " blocks (fixed), "
                  << v2_params.total_pages << "/" << npages_l_i32
                  << " active INT32 pages (mask=" << (zm_li_valid?"ON":"OFF") << "), "
                  << npages_l_i64 << " INT64 pages" << std::endl;

        bam_fused_q5li_v2_run_async(q5li_v2_ctx, v2_params, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto phase4_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase4_end - phase3_end).count();
        std::cout << "[Q5-TIMING] Phase 4 (LINEITEM): " << ms << " ms" << std::endl;
    }

    // ── Phase 5: Results ──
    int64_t h_revenue[25] = {};
    CUDA_CHECK(cudaMemcpy(h_revenue, d_revenue, 25 * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

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

    // Add REGION/NATION IO stats (uncompressed host-side reads)
    {
        uint64_t r_rkey_npages = metadata.table_region_npages[col_r_regionkey];
        uint64_t r_name_npages = metadata.table_region_npages[col_r_name];
        uint64_t n_nkey_npages = metadata.table_nation_npages[col_n_nationkey];
        uint64_t n_name_npages = metadata.table_nation_npages[col_n_name];
        uint64_t n_rkey_npages = metadata.table_nation_npages[col_n_regionkey];
        uint64_t rn_pages = r_rkey_npages + r_name_npages
                          + n_nkey_npages + n_name_npages + n_rkey_npages;
        total_io_count += rn_pages;
        total_io_bytes += rn_pages * page_size;
    }

    // Add SUPPLIER + CUSTOMER IO stats
    for (size_t fi : {FI_S_SUPPKEY, FI_S_NATIONKEY, FI_C_CUSTKEY, FI_C_NATIONKEY}) {
        for (uint64_t pg = 0; pg < field_npages_arr[fi]; pg++) {
            total_io_count++;
            if (field_comp_methods[fi] != CompressionMethod::NONE) {
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                if (nblk > 8 && nblk <= 16) nblk = 24;
                total_io_bytes += (uint64_t)nblk * 512;
            } else {
                total_io_bytes += page_size;
            }
        }
    }

    // Read back INT64 masks from GPU for IO accounting
    std::vector<uint8_t> h_ord_i64(npages_o_i64, 0);
    std::vector<uint8_t> h_li_i64(npages_l_i64, 0);
    if (zm_ord_valid)
        cudaMemcpy(h_ord_i64.data(), d_mask_ord_i64, npages_o_i64, cudaMemcpyDeviceToHost);
    if (zm_li_valid)
        cudaMemcpy(h_li_i64.data(), d_mask_li_i64, npages_l_i64, cudaMemcpyDeviceToHost);

    // Add fused ORDERS kernel I/O stats
    {
        // INT32 field: O_ORDERDATE — count active pages
        for (uint32_t pg = 0; pg < npages_o_i32; pg++) {
            if (zm_ord_valid && !zm_ord_ctx.h_mask[pg]) continue;
            total_io_count++;
            size_t fi = FI_O_ORDERDATE;
            if (field_comp_methods[fi] != CompressionMethod::NONE) {
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                if (nblk > 8 && nblk <= 16) nblk = 24;
                total_io_bytes += (uint64_t)nblk * 512;
            } else {
                total_io_bytes += page_size;
            }
        }
        // INT64 fields: O_ORDERKEY, O_CUSTKEY — GPU-derived mask
        const size_t o_i64_fi[2] = {FI_O_ORDERKEY, FI_O_CUSTKEY};
        for (uint32_t pg = 0; pg < npages_o_i64; pg++) {
            if (zm_ord_valid && !h_ord_i64[pg]) continue;
            for (int k = 0; k < 2; k++) {
                total_io_count++;
                size_t fi = o_i64_fi[k];
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
    }

    // Add fused LINEITEM kernel I/O stats
    {
        // INT32 fields: L_EXTPRICE, L_DISCOUNT — count active pages
        const size_t li_i32_fi[2] = {FI_L_EXTPRICE, FI_L_DISCOUNT};
        for (uint32_t pg = 0; pg < npages_l_i32; pg++) {
            if (zm_li_valid && !zm_li_ctx.h_mask[pg]) continue;
            for (int k = 0; k < 2; k++) {
                total_io_count++;
                size_t fi = li_i32_fi[k];
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
        // INT64 fields: L_ORDERKEY, L_SUPPKEY — GPU-derived mask
        const size_t li_i64_fi[2] = {FI_L_ORDERKEY, FI_L_SUPPKEY};
        for (uint32_t pg = 0; pg < npages_l_i64; pg++) {
            if (zm_li_valid && !h_li_i64[pg]) continue;
            for (int k = 0; k < 2; k++) {
                total_io_count++;
                size_t fi = li_i64_fi[k];
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    // Zone map contexts
    if (zm_ord_valid) bam_zonemap_ctx_destroy(zm_ord_ctx);
    if (zm_li_valid) bam_zonemap_ctx_destroy(zm_li_ctx);
    if (zm_pc) bam_io_page_cache_destroy(zm_pc);
    if (d_mask_ord_i64) cudaFree(d_mask_ord_i64);
    if (d_mask_li_i64) cudaFree(d_mask_li_i64);
    // Hash tables
    if (d_ht_supp_keys) cudaFree(d_ht_supp_keys);
    if (d_ht_supp_values) cudaFree(d_ht_supp_values);
    if (d_ht_cust_keys) cudaFree(d_ht_cust_keys);
    if (d_ht_cust_values) cudaFree(d_ht_cust_values);
    cudaFree(d_ht_ord_keys);
    cudaFree(d_ht_ord_values);
    cudaFree(d_revenue);
    cudaFree(d_nationkey_to_idx);
    // Staging and batch infrastructure
    mb_cuda_free(staging_data);
    mb_cuda_free(staging_io);
    cudaFree(d_batch_ps);
    cudaFree(d_page_mask);
    cudaFree(d_decomp_scs);
    cudaFree(d_decomp_spi);
    bam_bulk_read_ctx_destroy(io_ctx_batch);
    // Warp-spec Q5 shared page_cache + decomp cleanup
    if (q5_ws_decomp) cudaFree(q5_ws_decomp);
    if (q5_ws_pc) bam_io_page_cache_destroy(q5_ws_pc);
    // Q5 LINEITEM v2 context cleanup
    if (q5li_v2_ctx) bam_fused_q5li_v2_destroy(q5li_v2_ctx);
    cudaFree(d_ps_li_i32);
    cudaFree(d_ps_li_i64);
    for (int k = 0; k < 2; k++) {
        if (d_li_comp_offsets_i32[k]) cudaFree(d_li_comp_offsets_i32[k]);
        if (d_li_comp_sizes_i32[k]) cudaFree(d_li_comp_sizes_i32[k]);
        if (d_li_comp_offsets_i64[k]) cudaFree(d_li_comp_offsets_i64[k]);
        if (d_li_comp_sizes_i64[k]) cudaFree(d_li_comp_sizes_i64[k]);
    }
    cudaFree(d_ps_o_i32);
    cudaFree(d_ps_o_i64);
    if (d_o_comp_offsets_i32) cudaFree(d_o_comp_offsets_i32);
    if (d_o_comp_sizes_i32) cudaFree(d_o_comp_sizes_i32);
    for (int k = 0; k < 2; k++) {
        if (d_o_comp_offsets_i64[k]) cudaFree(d_o_comp_offsets_i64[k]);
        if (d_o_comp_sizes_i64[k]) cudaFree(d_o_comp_sizes_i64[k]);
    }
    // Nocopy Q5 dim (SUPPLIER + CUSTOMER) cleanup
    bam_bulk_read_ctx_destroy(q5_dim_nocopy);
    for (size_t fi : q5_dim_fis) {
        if (d_dim_comp_offsets[fi]) cudaFree(d_dim_comp_offsets[fi]);
        if (d_dim_comp_sizes[fi]) cudaFree(d_dim_comp_sizes[fi]);
        if (d_dim_prefix_sum[fi]) cudaFree(d_dim_prefix_sum[fi]);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_ctrl_close(ctrl);

    // Collect compression method string
    std::string comp_str;
    {
        std::set<std::string> methods;
        for (size_t fi = 0; fi < NUM_Q5_FIELDS; fi++)
            methods.insert(compression_method_name(field_comp_methods[fi]));
        for (const auto &m : methods) {
            if (!comp_str.empty()) comp_str += "+";
            comp_str += m;
        }
    }

    size_t total_pages = 0;
    for (size_t fi = 0; fi < NUM_Q5_FIELDS; fi++)
        total_pages += field_npages_arr[fi];

    return BenchmarkResult{
        .nios = total_io_count,
        .read_bytes = total_io_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = comp_str,
        .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// TPC-H Q3 — GIDP+BAM+DECOMP execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMPdx (device-side LZ4)
// Pipeline: 4-phase hash join (CUSTOMER → ORDERS → LINEITEM → collect)
// ============================================================
BenchmarkResult tpch_q3(BenchmarkOptions &options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    // ── 2. Open BaM controller(s) ──
    size_t gpu_free_pre_ctrl = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_dummy);

    const uint32_t bam_num_queues = 128;
    auto ds = DataPathFusion::bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_dummy);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;
    const uint32_t n_devices = ds.n_devices;

    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata page via BaM ──
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    if (rc != 0) {
        std::cerr << "bam_read_page failed (metadata header)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (full metadata)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const bool is_q3sel = (options.q3sel_selectivity > 0);
    const int sel_pct = is_q3sel ? options.q3sel_selectivity : 20;

    if (is_q3sel)
        std::cout << "=== TPCH Q3SEL (GIDP+BAM+DECOMP, sel=" << sel_pct << "%) ===" << std::endl;
    else
        std::cout << "=== TPCH Q3 (GIDP+BAM+DECOMP) ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;

    const uint64_t nrecs_customer = metadata.table_customer_nrows;
    const uint64_t nrecs_orders   = metadata.table_orders_nrows;
    const uint64_t nrecs_lineitem = metadata.table_lineitem_nrows;
    std::cout << "nrecs: customer=" << nrecs_customer
              << ", orders=" << nrecs_orders
              << ", lineitem=" << nrecs_lineitem << std::endl;

    // ── 4. Extract Q3 field info (10 fields, 3 tables) ──
    constexpr size_t FI_C_CUSTKEY      = 0;
    constexpr size_t FI_C_MKTSEGMENT   = 1;
    constexpr size_t FI_O_ORDERKEY     = 2;
    constexpr size_t FI_O_CUSTKEY      = 3;
    constexpr size_t FI_O_ORDERDATE    = 4;
    constexpr size_t FI_O_SHIPPRIORITY = 5;
    constexpr size_t FI_L_ORDERKEY     = 6;
    constexpr size_t FI_L_EXTPRICE     = 7;
    constexpr size_t FI_L_DISCOUNT     = 8;
    constexpr size_t FI_L_SHIPDATE     = 9;
    constexpr size_t NUM_Q3_FIELDS     = 10;

    const size_t blocks_per_page = page_size / 512;

    const size_t col_c_custkey      = TPCH::common::C_CUSTKEY;
    const size_t col_c_mktseg       = TPCH::common::C_MKTSEGMENT;
    const size_t col_o_orderkey     = TPCH::common::O_ORDERKEY;
    const size_t col_o_custkey      = TPCH::common::O_CUSTKEY;
    const size_t col_o_orderdate    = TPCH::common::O_ORDERDATE;
    const size_t col_o_shippriority = TPCH::common::O_SHIPPRIORITY;
    const size_t col_l_orderkey     = TPCH::common::L_ORDERKEY;
    const size_t col_l_extprice     = TPCH::common::L_EXTENDEDPRICE;
    const size_t col_l_discount     = TPCH::common::L_DISCOUNT;
    const size_t col_l_shipdate     = TPCH::common::L_SHIPDATE;

    uint64_t field_start_page_ids[NUM_Q3_FIELDS];
    uint64_t field_npages_arr[NUM_Q3_FIELDS];
    CompressionMethod field_comp_methods[NUM_Q3_FIELDS];

    // CUSTOMER columns
    field_start_page_ids[FI_C_CUSTKEY] = metadata.table_customer_start_page_ids[col_c_custkey];
    field_npages_arr[FI_C_CUSTKEY]     = metadata.table_customer_npages[col_c_custkey];
    field_comp_methods[FI_C_CUSTKEY]   = static_cast<CompressionMethod>(
        metadata.table_customer_compression_method[col_c_custkey]);

    field_start_page_ids[FI_C_MKTSEGMENT] = metadata.table_customer_start_page_ids[col_c_mktseg];
    field_npages_arr[FI_C_MKTSEGMENT]     = metadata.table_customer_npages[col_c_mktseg];
    field_comp_methods[FI_C_MKTSEGMENT]   = static_cast<CompressionMethod>(
        metadata.table_customer_compression_method[col_c_mktseg]);

    // ORDERS columns
    field_start_page_ids[FI_O_ORDERKEY] = metadata.table_orders_start_page_ids[col_o_orderkey];
    field_npages_arr[FI_O_ORDERKEY]     = metadata.table_orders_npages[col_o_orderkey];
    field_comp_methods[FI_O_ORDERKEY]   = static_cast<CompressionMethod>(
        metadata.table_orders_compression_method[col_o_orderkey]);

    field_start_page_ids[FI_O_CUSTKEY] = metadata.table_orders_start_page_ids[col_o_custkey];
    field_npages_arr[FI_O_CUSTKEY]     = metadata.table_orders_npages[col_o_custkey];
    field_comp_methods[FI_O_CUSTKEY]   = static_cast<CompressionMethod>(
        metadata.table_orders_compression_method[col_o_custkey]);

    field_start_page_ids[FI_O_ORDERDATE] = metadata.table_orders_start_page_ids[col_o_orderdate];
    field_npages_arr[FI_O_ORDERDATE]     = metadata.table_orders_npages[col_o_orderdate];
    field_comp_methods[FI_O_ORDERDATE]   = static_cast<CompressionMethod>(
        metadata.table_orders_compression_method[col_o_orderdate]);

    field_start_page_ids[FI_O_SHIPPRIORITY] = metadata.table_orders_start_page_ids[col_o_shippriority];
    field_npages_arr[FI_O_SHIPPRIORITY]     = metadata.table_orders_npages[col_o_shippriority];
    field_comp_methods[FI_O_SHIPPRIORITY]   = static_cast<CompressionMethod>(
        metadata.table_orders_compression_method[col_o_shippriority]);

    // LINEITEM columns
    field_start_page_ids[FI_L_ORDERKEY] = metadata.table_lineitem_start_page_ids[col_l_orderkey];
    field_npages_arr[FI_L_ORDERKEY]     = metadata.table_lineitem_npages[col_l_orderkey];
    field_comp_methods[FI_L_ORDERKEY]   = static_cast<CompressionMethod>(
        metadata.table_lineitem_compression_method[col_l_orderkey]);

    field_start_page_ids[FI_L_EXTPRICE] = metadata.table_lineitem_start_page_ids[col_l_extprice];
    field_npages_arr[FI_L_EXTPRICE]     = metadata.table_lineitem_npages[col_l_extprice];
    field_comp_methods[FI_L_EXTPRICE]   = static_cast<CompressionMethod>(
        metadata.table_lineitem_compression_method[col_l_extprice]);

    field_start_page_ids[FI_L_DISCOUNT] = metadata.table_lineitem_start_page_ids[col_l_discount];
    field_npages_arr[FI_L_DISCOUNT]     = metadata.table_lineitem_npages[col_l_discount];
    field_comp_methods[FI_L_DISCOUNT]   = static_cast<CompressionMethod>(
        metadata.table_lineitem_compression_method[col_l_discount]);

    field_start_page_ids[FI_L_SHIPDATE] = metadata.table_lineitem_start_page_ids[col_l_shipdate];
    field_npages_arr[FI_L_SHIPDATE]     = metadata.table_lineitem_npages[col_l_shipdate];
    field_comp_methods[FI_L_SHIPDATE]   = static_cast<CompressionMethod>(
        metadata.table_lineitem_compression_method[col_l_shipdate]);

    const char *field_names[NUM_Q3_FIELDS] = {
        "C_CUSTKEY", "C_MKTSEGMENT",
        "O_ORDERKEY", "O_CUSTKEY", "O_ORDERDATE", "O_SHIPPRIORITY",
        "L_ORDERKEY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE" };
    for (size_t fi = 0; fi < NUM_Q3_FIELDS; fi++) {
        std::cout << "  " << field_names[fi]
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << static_cast<int>(field_comp_methods[fi])
                  << std::endl;
    }

    // Validate: device-side decomp only supports LZ4
    for (size_t fi = 0; fi < NUM_Q3_FIELDS; fi++) {
        if (field_comp_methods[fi] != CompressionMethod::NONE &&
            field_comp_methods[fi] != CompressionMethod::LZ4) {
            std::cerr << "gidp+bam+fusion only supports LZ4 or NONE compression, "
                      << "field " << field_names[fi] << " uses method "
                      << static_cast<int>(field_comp_methods[fi]) << std::endl;
            bam_ctrl_close(ctrl);
            exit(EXIT_FAILURE);
        }
    }

    // ── 5. Read compression metadata ──
    std::vector<uint32_t> h_comp_sizes[NUM_Q3_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_Q3_FIELDS];

    auto get_comp_meta = [&](size_t fi, uint64_t &cs_start, uint64_t &cs_npages_cnt,
                             uint64_t &nbase, uint64_t &base_start) {
        switch (fi) {
        case FI_C_CUSTKEY: case FI_C_MKTSEGMENT: {
            size_t col = (fi == FI_C_CUSTKEY) ? col_c_custkey : col_c_mktseg;
            cs_start      = metadata.table_customer_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_customer_compressed_page_sizes_npages[col];
            nbase         = metadata.table_customer_compression_nbases[col];
            base_start    = metadata.table_customer_compression_base_start_page_ids[col];
            break;
        }
        case FI_O_ORDERKEY: case FI_O_CUSTKEY: case FI_O_ORDERDATE: case FI_O_SHIPPRIORITY: {
            size_t col;
            switch (fi) {
            case FI_O_ORDERKEY:     col = col_o_orderkey; break;
            case FI_O_CUSTKEY:      col = col_o_custkey; break;
            case FI_O_ORDERDATE:    col = col_o_orderdate; break;
            default:                col = col_o_shippriority; break;
            }
            cs_start      = metadata.table_orders_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_orders_compressed_page_sizes_npages[col];
            nbase         = metadata.table_orders_compression_nbases[col];
            base_start    = metadata.table_orders_compression_base_start_page_ids[col];
            break;
        }
        case FI_L_ORDERKEY: case FI_L_EXTPRICE: case FI_L_DISCOUNT: case FI_L_SHIPDATE: {
            size_t col;
            switch (fi) {
            case FI_L_ORDERKEY:  col = col_l_orderkey; break;
            case FI_L_EXTPRICE:  col = col_l_extprice; break;
            case FI_L_DISCOUNT:  col = col_l_discount; break;
            default:             col = col_l_shipdate; break;
            }
            cs_start      = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
            cs_npages_cnt = metadata.table_lineitem_compressed_page_sizes_npages[col];
            nbase         = metadata.table_lineitem_compression_nbases[col];
            base_start    = metadata.table_lineitem_compression_base_start_page_ids[col];
            break;
        }
        }
    };

    for (size_t fi = 0; fi < NUM_Q3_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;

        uint64_t cs_start, cs_npages_cnt, nbase, base_start;
        get_comp_meta(fi, cs_start, cs_npages_cnt, nbase, base_start);

        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_sizes " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        h_comp_sizes[fi].assign(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages_arr[fi]);

        size_t bp_npages = TPCH::nbase_to_npages(nbase, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_bases " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            h_comp_sizes[fi].data(), nbase, field_npages_arr[fi], page_size,
            field_start_page_ids[fi], n_devices, offsets_vec);
        h_comp_offsets[fi] = std::move(offsets_vec);
    }

    // ── 6. Read prefix sums via BaM ──
    auto read_prefix_sum = [&](uint64_t ps_start, uint64_t ps_npages_cnt, uint64_t npages_field)
        -> std::vector<uint64_t>
    {
        std::vector<uint64_t> result_ps;
        if (ps_npages_cnt == 0) return result_ps;
        std::vector<char> ps_buf(ps_npages_cnt * page_size);
        for (uint64_t p = 0; p < ps_npages_cnt; p++) {
            rc = read_striped_page(ps_start + p, page_size, ps_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (prefix_sum)" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        auto *raw = reinterpret_cast<uint64_t*>(ps_buf.data());
        result_ps.assign(raw + 1, raw + 1 + npages_field);
        return result_ps;
    };

    auto get_ps_meta = [&](size_t fi, uint64_t &ps_start, uint64_t &ps_npages_cnt) {
        switch (fi) {
        case FI_C_CUSTKEY: case FI_C_MKTSEGMENT: {
            size_t col = (fi == FI_C_CUSTKEY) ? col_c_custkey : col_c_mktseg;
            ps_start      = metadata.table_customer_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_customer_prefix_sum_npages[col];
            break;
        }
        case FI_O_ORDERKEY: case FI_O_CUSTKEY: case FI_O_ORDERDATE: case FI_O_SHIPPRIORITY: {
            size_t col;
            switch (fi) {
            case FI_O_ORDERKEY:     col = col_o_orderkey; break;
            case FI_O_CUSTKEY:      col = col_o_custkey; break;
            case FI_O_ORDERDATE:    col = col_o_orderdate; break;
            default:                col = col_o_shippriority; break;
            }
            ps_start      = metadata.table_orders_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_orders_prefix_sum_npages[col];
            break;
        }
        case FI_L_ORDERKEY: case FI_L_EXTPRICE: case FI_L_DISCOUNT: case FI_L_SHIPDATE: {
            size_t col;
            switch (fi) {
            case FI_L_ORDERKEY:  col = col_l_orderkey; break;
            case FI_L_EXTPRICE:  col = col_l_extprice; break;
            case FI_L_DISCOUNT:  col = col_l_discount; break;
            default:             col = col_l_shipdate; break;
            }
            ps_start      = metadata.table_lineitem_prefix_sum_start_page_ids[col];
            ps_npages_cnt = metadata.table_lineitem_prefix_sum_npages[col];
            break;
        }
        }
    };

    std::vector<uint64_t> h_prefix_sum[NUM_Q3_FIELDS];
    for (size_t fi = 0; fi < NUM_Q3_FIELDS; fi++) {
        uint64_t ps_start, ps_npages_cnt;
        get_ps_meta(fi, ps_start, ps_npages_cnt);
        h_prefix_sum[fi] = read_prefix_sum(ps_start, ps_npages_cnt, field_npages_arr[fi]);
    }

    // ── 7. Persistent GPU structures (hash tables, aggregation, results) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // CUSTOMER hash set
    uint64_t est_building = is_q3sel
        ? std::max((uint64_t)1024, nrecs_customer * sel_pct / 100)
        : std::max((uint64_t)1024, nrecs_customer / 4);
    uint32_t custset_cap = 1;
    while (custset_cap < est_building * 2) custset_cap <<= 1;
    uint32_t custset_mask = custset_cap - 1;
    uint64_t *d_custkey_set = nullptr;
    CUDA_CHECK(cudaMalloc(&d_custkey_set, custset_cap * sizeof(uint64_t)));

    // ORDERS hash table
    uint64_t est_orders_qual = is_q3sel
        ? std::max((uint64_t)1024, nrecs_orders * sel_pct / 100)
        : std::max((uint64_t)1024, nrecs_orders / 8);
    uint32_t orders_ht_cap = 1;
    while (orders_ht_cap < est_orders_qual * 2) orders_ht_cap <<= 1;

    if (is_q3sel) {
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t app_used = gpu_free_before_app - gpu_free_now;
        uint64_t total_budget = (gpu_ctrl_bytes < GPU_MEM_BUDGET)
            ? GPU_MEM_BUDGET - gpu_ctrl_bytes : 0;
        if (app_used < total_budget) {
            uint64_t remaining = total_budget - app_used;
            uint64_t custset_bytes = (uint64_t)custset_cap * sizeof(uint64_t);
            if (remaining > custset_bytes) {
                uint64_t ht_budget = remaining - custset_bytes;
                constexpr uint64_t BYTES_PER_ENTRY = 8 + 8 + 8 + 8;
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

    // Aggregation hash map
    uint32_t aggr_cap = orders_ht_cap;
    uint32_t aggr_mask = aggr_cap - 1;
    uint64_t *d_aggr_keys = nullptr;
    int64_t  *d_aggr_revenues = nullptr;
    CUDA_CHECK(cudaMalloc(&d_aggr_keys, aggr_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_aggr_revenues, aggr_cap * sizeof(int64_t)));

    // Result arrays (allocated after Phase 3 to maximize Phase 2-3 memory budget)
    Q3ResultRow *d_results = nullptr;
    uint32_t *d_result_count = nullptr;

    // Pre-compute CUB DeviceMergeSort temp size (allocation deferred)
    void *d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceMergeSort::SortKeys(nullptr, sort_temp_bytes,
        d_results, (int)aggr_cap, Q3ResultCmp{}, stream);

    uint64_t gpu_app_bytes = 0;

    // ── 8. Streaming infrastructure ──
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    constexpr size_t Q3_BATCH_PAGES = 512;

    // LINEITEM tile metrics: INT32 vs INT64 page mapping
    const uint64_t npages_l_i32 = field_npages_arr[FI_L_SHIPDATE];
    const uint64_t npages_l_i64 = field_npages_arr[FI_L_ORDERKEY];

    // Full prefix sums for LINEITEM (leading-zero form for binary search)
    std::vector<uint64_t> ps_l_i32_full(npages_l_i32 + 1);
    ps_l_i32_full[0] = 0;
    for (uint64_t i = 0; i < npages_l_i32; i++)
        ps_l_i32_full[i + 1] = h_prefix_sum[FI_L_SHIPDATE][i];

    std::vector<uint64_t> ps_l_i64_full(npages_l_i64 + 1);
    ps_l_i64_full[0] = 0;
    for (uint64_t i = 0; i < npages_l_i64; i++)
        ps_l_i64_full[i + 1] = h_prefix_sum[FI_L_ORDERKEY][i];

    // ORDERS prefix sums (needed for zone map INT32→INT64 mapping)
    const uint64_t npages_o_i64 = field_npages_arr[FI_O_ORDERKEY];
    const uint64_t npages_o_i32 = field_npages_arr[FI_O_ORDERDATE];
    std::vector<uint64_t> ps_o_i32_full(npages_o_i32 + 1);
    ps_o_i32_full[0] = 0;
    for (uint64_t i = 0; i < npages_o_i32; i++)
        ps_o_i32_full[i + 1] = h_prefix_sum[FI_O_ORDERDATE][i];
    std::vector<uint64_t> ps_o_i64_full(npages_o_i64 + 1);
    ps_o_i64_full[0] = 0;
    for (uint64_t i = 0; i < npages_o_i64; i++)
        ps_o_i64_full[i + 1] = h_prefix_sum[FI_O_ORDERKEY][i];

    // Staging buffers: staging_data holds all Phase 1 pages (combined read),
    // staging_io is batch-sized (used by batched_bam_read_pages internally)
    const uint64_t npages_custkey = field_npages_arr[FI_C_CUSTKEY];
    const uint64_t npages_mktseg = field_npages_arr[FI_C_MKTSEGMENT];
    const size_t p1_total_pages = npages_custkey + npages_mktseg;
    const size_t staging_pages = std::max(p1_total_pages, (size_t)Q3_BATCH_PAGES);

    void *staging_data = mb_cuda_alloc(staging_pages * page_size);
    void *staging_io   = mb_cuda_alloc(Q3_BATCH_PAGES * page_size);
    uint64_t *d_batch_ps = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_ps, staging_pages * sizeof(uint64_t)));
    std::vector<uint64_t> bps(staging_pages);

    // Persistent BaM read context (shared by Phase 1 batched reads and later phases)
    BamBulkReadCtx io_ctx_p1 = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(sm_count),
        static_cast<uint32_t>(Q3_BATCH_PAGES));

    uint8_t *d_page_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_page_mask, Q3_BATCH_PAGES));

    // ── Warp-spec Q3 LINEITEM kernel setup ──
    // Deferred: page_cache + decomp_buf created after zone map to use active page count
    bam_io_page_cache_t q3li_ws_pc = nullptr;
    char* q3li_ws_decomp = nullptr;
    uint32_t fused_q3li_num_blocks = 0;

    // ── Fused Q3 ORDERS kernel setup ──
    // Deferred: context created after zone map to use active page count for num_blocks
    bam_fused_q3ord_ctx_t fused_q3ord_ctx = nullptr;
    uint32_t fused_q3ord_num_blocks = 0;

    // Upload ORDERS prefix sums to GPU
    uint64_t *d_ps_o_i32 = nullptr, *d_ps_o_i64 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_o_i32, (npages_o_i32 + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_o_i32, ps_o_i32_full.data(),
                (npages_o_i32 + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ps_o_i64, (npages_o_i64 + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_o_i64, ps_o_i64_full.data(),
                (npages_o_i64 + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Upload ORDERS compression metadata to GPU
    uint64_t* d_o_comp_offsets_i32[2] = {};
    uint32_t* d_o_comp_sizes_i32[2] = {};
    {
        const size_t o_i32_fis[2] = {FI_O_ORDERDATE, FI_O_SHIPPRIORITY};
        for (int k = 0; k < 2; k++) {
            size_t fi = o_i32_fis[k];
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            CUDA_CHECK(cudaMalloc(&d_o_comp_offsets_i32[k], npages_o_i32 * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_o_comp_offsets_i32[k], h_comp_offsets[fi].data(),
                        npages_o_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_o_comp_sizes_i32[k], npages_o_i32 * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_o_comp_sizes_i32[k], h_comp_sizes[fi].data(),
                        npages_o_i32 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }
    uint64_t* d_o_comp_offsets_i64[2] = {};
    uint32_t* d_o_comp_sizes_i64[2] = {};
    {
        const size_t o_i64_fis[2] = {FI_O_ORDERKEY, FI_O_CUSTKEY};
        for (int k = 0; k < 2; k++) {
            size_t fi = o_i64_fis[k];
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            CUDA_CHECK(cudaMalloc(&d_o_comp_offsets_i64[k], npages_o_i64 * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_o_comp_offsets_i64[k], h_comp_offsets[fi].data(),
                        npages_o_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_o_comp_sizes_i64[k], npages_o_i64 * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_o_comp_sizes_i64[k], h_comp_sizes[fi].data(),
                        npages_o_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }

    // ORDERS zone map mask (uploaded after zone map is computed)
    // Upload LINEITEM prefix sums to GPU
    uint64_t *d_ps_li_i32 = nullptr, *d_ps_li_i64 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_li_i32, (npages_l_i32 + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_li_i32, ps_l_i32_full.data(),
                (npages_l_i32 + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ps_li_i64, (npages_l_i64 + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_li_i64, ps_l_i64_full.data(),
                (npages_l_i64 + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Upload LINEITEM compression metadata to GPU
    // INT32 fields: FI_L_SHIPDATE=9, FI_L_EXTPRICE=7, FI_L_DISCOUNT=8
    uint64_t* d_li_comp_offsets_i32[3] = {};
    uint32_t* d_li_comp_sizes_i32[3] = {};
    const size_t li_i32_fis[3] = {FI_L_SHIPDATE, FI_L_EXTPRICE, FI_L_DISCOUNT};
    for (int k = 0; k < 3; k++) {
        size_t fi = li_i32_fis[k];
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        CUDA_CHECK(cudaMalloc(&d_li_comp_offsets_i32[k], npages_l_i32 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_li_comp_offsets_i32[k], h_comp_offsets[fi].data(),
                    npages_l_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_li_comp_sizes_i32[k], npages_l_i32 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_li_comp_sizes_i32[k], h_comp_sizes[fi].data(),
                    npages_l_i32 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    // INT64 field: FI_L_ORDERKEY=6
    uint64_t* d_li_comp_offsets_i64 = nullptr;
    uint32_t* d_li_comp_sizes_i64 = nullptr;
    if (field_comp_methods[FI_L_ORDERKEY] != CompressionMethod::NONE) {
        CUDA_CHECK(cudaMalloc(&d_li_comp_offsets_i64, npages_l_i64 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_li_comp_offsets_i64, h_comp_offsets[FI_L_ORDERKEY].data(),
                    npages_l_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_li_comp_sizes_i64, npages_l_i64 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_li_comp_sizes_i64, h_comp_sizes[FI_L_ORDERKEY].data(),
                    npages_l_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // Pre-allocate decompress helper buffers (sized for batch)
    uint32_t *d_decomp_scs = nullptr, *d_decomp_spi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_decomp_scs, Q3_BATCH_PAGES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_decomp_spi, Q3_BATCH_PAGES * sizeof(uint32_t)));

    // Read + LZ4 decompress a batch of pages for one field into staging
    auto batch_read_lz4_field = [&](size_t fi, uint64_t pg_start, uint64_t pg_count,
                                     size_t &io_count_out, size_t &io_bytes_out) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        size_t io_offset = 0;
        for (uint64_t pg = pg_start; pg < pg_start + pg_count; pg++) {
            uint64_t local_pg = pg - pg_start;
            BamBulkReadDesc &desc = io_ctx_p1.h_descs[0][local_pg];
            desc = {};
            if (is_compressed) {
                uint64_t byte_offset = h_comp_offsets[fi][pg];
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                desc.nblocks = (roundup4096(comp_sz) + 511) / 512;
                constexpr uint32_t CTRL_PAGE_BLOCKS = 4096 / 512;
                if (desc.nblocks > CTRL_PAGE_BLOCKS && desc.nblocks <= 2 * CTRL_PAGE_BLOCKS)
                    desc.nblocks = 3 * CTRL_PAGE_BLOCKS;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_io) + io_offset;
                desc.copy_bytes = comp_sz;
                io_offset += page_size;
            } else {
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                uint64_t local_page_id = page_id / n_devices;
                desc.lba = ds.partition_start_lbas[dev] + local_page_id * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_data) + local_pg * page_size;
                desc.copy_bytes = page_size;
            }
        }
        bam_bulk_read_async(io_ctx_p1, static_cast<uint32_t>(pg_count), 0, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Aggregate IO stats
        for (uint64_t pg = 0; pg < pg_count; pg++) {
            io_count_out++;
            io_bytes_out += static_cast<uint64_t>(io_ctx_p1.h_descs[0][pg].nblocks) * 512;
        }

        if (is_compressed) {
            std::vector<uint32_t> slot_comp_sizes(pg_count);
            std::vector<uint32_t> slot_page_indices(pg_count);
            for (uint64_t pg = 0; pg < pg_count; pg++) {
                slot_comp_sizes[pg] = h_comp_sizes[fi][pg_start + pg];
                slot_page_indices[pg] = static_cast<uint32_t>(pg);
            }
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_scs, slot_comp_sizes.data(),
                        pg_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_decomp_spi, slot_page_indices.data(),
                        pg_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            bam_lz4_batch_decompress(
                static_cast<const char*>(staging_io),
                static_cast<char*>(staging_data),
                d_decomp_scs, d_decomp_spi,
                static_cast<uint32_t>(pg_count),
                static_cast<uint32_t>(page_size), stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // Upload batch-relative prefix sum for pages [pg_start, pg_start+bnp)
    auto upload_batch_ps = [&](size_t fi, uint64_t pg_start, uint64_t bnp) {
        uint64_t base = (pg_start == 0) ? 0 : h_prefix_sum[fi][pg_start - 1];
        for (uint64_t i = 0; i < bnp; i++)
            bps[i] = h_prefix_sum[fi][pg_start + i] - base;
        CUDA_CHECK(cudaMemcpyAsync(d_batch_ps, bps.data(),
                                   bnp * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    };

    // Read + decompress + flatten INT64 field → pre-allocated flat array
    auto batch_flatten_int64 = [&](size_t fi, const std::vector<uint64_t> &ps,
                                    uint64_t nrecs, uint64_t *d_flat,
                                    size_t &io_cnt, size_t &io_bytes) {
        uint64_t np = field_npages_arr[fi];
        for (uint64_t pg = 0; pg < np; pg += Q3_BATCH_PAGES) {
            uint64_t bnp = std::min(Q3_BATCH_PAGES, np - pg);
            batch_read_lz4_field(fi, pg, bnp, io_cnt, io_bytes);
            upload_batch_ps(fi, pg, bnp);
            uint64_t row_start = (pg == 0) ? 0 : ps[pg - 1];
            uint64_t batch_nrecs = ps[pg + bnp - 1] - row_start;
            q13_flatten_int64_pages_ps(
                static_cast<const char*>(staging_data),
                page_size, d_batch_ps, bnp,
                batch_nrecs, d_flat + row_start, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // Read + decompress + flatten INT32 field → pre-allocated flat array
    auto batch_flatten_int32 = [&](size_t fi, const std::vector<uint64_t> &ps,
                                    uint64_t nrecs, uint64_t *d_flat,
                                    size_t &io_cnt, size_t &io_bytes) {
        uint64_t np = field_npages_arr[fi];
        for (uint64_t pg = 0; pg < np; pg += Q3_BATCH_PAGES) {
            uint64_t bnp = std::min(Q3_BATCH_PAGES, np - pg);
            batch_read_lz4_field(fi, pg, bnp, io_cnt, io_bytes);
            upload_batch_ps(fi, pg, bnp);
            uint64_t row_start = (pg == 0) ? 0 : ps[pg - 1];
            uint64_t batch_nrecs = ps[pg + bnp - 1] - row_start;
            q13_flatten_int32_pages_ps(
                static_cast<const char*>(staging_data),
                page_size, d_batch_ps, bnp,
                batch_nrecs, d_flat + row_start, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // ── Selective batch read with LZ4 (reads only active pages at correct slot positions) ──
    // bam_lz4_batch_decompress reads compressed data SEQUENTIALLY (slot 0,1,2,...) from
    // d_comp_pages, and writes decompressed output to d_decomp_pages[d_page_indices[i]].
    // So: BaM must pack compressed data sequentially in staging_io, and d_page_indices
    // maps from sequential index to the target slot position in staging_data.
    // ── Zone map: pre-compute per-page active flags ──
    // Q3SEL: compute mktseg dict ID range for sideways pruning
    int32_t q3sel_mktseg_lo = 0, q3sel_mktseg_hi = 4;
    if (is_q3sel && sel_pct < 100) {
        static constexpr int32_t seg_dict_ids[] = {1, 0, 2, 3, 4};
        int nseg = std::max(1, sel_pct / 20);
        q3sel_mktseg_lo = seg_dict_ids[0];
        q3sel_mktseg_hi = seg_dict_ids[0];
        for (int i = 1; i < nseg; i++) {
            q3sel_mktseg_lo = std::min(q3sel_mktseg_lo, seg_dict_ids[i]);
            q3sel_mktseg_hi = std::max(q3sel_mktseg_hi, seg_dict_ids[i]);
        }
    }

    const bool enable_zonemap = options.enable_zonemap;

    // ── GPU-side zone map setup (metadata extraction outside timing) ──
    bam_io_page_cache_t zm_pc = nullptr;
    BamZonemapCtx zm_ord_ctx{}, zm_li_ctx{};
    uint32_t zm_ord_nreads = 0, zm_ord_npreds = 0;
    uint32_t zm_li_nreads = 0, zm_li_npreds = 0;
    bool zm_ord_valid = false, zm_li_valid = false;

    if (enable_zonemap) {
        zm_pc = bam_io_page_cache_create(ctrl, page_size, kBamZonemapMaxReads);
        void* zm_d_ctrls = bam_io_page_cache_get_d_ctrls(zm_pc);
        void* zm_d_pc_ptr = bam_io_page_cache_get_d_pc_ptr(zm_pc);
        const char* zm_pc_base = (const char*)bam_io_page_cache_get_base_addr(zm_pc);

        // ORDERS zone map
        zm_ord_ctx = bam_zonemap_ctx_create(zm_d_ctrls, zm_d_pc_ptr, (void*)zm_pc_base,
            static_cast<uint32_t>(page_size), npages_o_i32);

        {
            const size_t o_odate_field = TPCH::common::O_ORDERDATE;

            // O_ORDERDATE direct stats (Q3, and Q3SEL when filters enabled)
            if (!is_q3sel || !options.disable_other_filters) {
                uint64_t odate_nstats     = metadata.table_orders_nstats[o_odate_field];
                uint64_t odate_stats_start = metadata.table_orders_stats_start_page_ids[o_odate_field];
                uint64_t odate_stats_npg   = metadata.table_orders_stats_npages[o_odate_field];
                if (odate_nstats > 0 && odate_stats_start > 0) {
                    for (uint64_t j = 0; j < odate_stats_npg; j++) {
                        uint64_t pg_id = odate_stats_start + j;
                        uint32_t dev = pg_id % n_devices;
                        uint64_t local = pg_id / n_devices;
                        zm_ord_ctx.h_reads[zm_ord_nreads++] = {
                            ds.partition_start_lbas[dev] + local * blocks_per_page,
                            static_cast<uint32_t>(blocks_per_page), dev};
                    }
                    zm_ord_ctx.h_preds[zm_ord_npreds++] = {0, odate_nstats,
                        INT32_MIN, 19950314};
                }
            }

            // Sideways C_MKTSEGMENT stats (both Q3 and Q3SEL)
            const size_t sw_mktseg_idx = TPCH::common::OS_SIDEWAYS_C_MKTSEGMENT;
            uint64_t mktseg_nstats      = metadata.table_orders_sideways_nstats[o_odate_field][sw_mktseg_idx];
            uint64_t mktseg_stats_start = metadata.table_orders_sideways_stats_start_page_ids[o_odate_field][sw_mktseg_idx];
            uint64_t mktseg_stats_npg   = metadata.table_orders_sideways_stats_npages[o_odate_field][sw_mktseg_idx];
            if (mktseg_nstats > 0 && mktseg_stats_start > 0) {
                uint32_t stats_page_offset = zm_ord_nreads;
                for (uint64_t j = 0; j < mktseg_stats_npg; j++) {
                    uint64_t pg_id = mktseg_stats_start + j;
                    uint32_t dev = pg_id % n_devices;
                    uint64_t local = pg_id / n_devices;
                    zm_ord_ctx.h_reads[zm_ord_nreads++] = {
                        ds.partition_start_lbas[dev] + local * blocks_per_page,
                        static_cast<uint32_t>(blocks_per_page), dev};
                }
                if (is_q3sel)
                    zm_ord_ctx.h_preds[zm_ord_npreds++] = {stats_page_offset, mktseg_nstats,
                        q3sel_mktseg_lo, q3sel_mktseg_hi};
                else
                    zm_ord_ctx.h_preds[zm_ord_npreds++] = {stats_page_offset, mktseg_nstats,
                        1, 1};
            }
        }
        zm_ord_valid = (zm_ord_npreds > 0);

        // LINEITEM zone map
        zm_li_ctx = bam_zonemap_ctx_create(zm_d_ctrls, zm_d_pc_ptr, (void*)zm_pc_base,
            static_cast<uint32_t>(page_size), npages_l_i32);

        {
            const size_t l_sd_field = TPCH::common::L_SHIPDATE;

            // L_SHIPDATE direct stats (Q3, and Q3SEL when filters enabled)
            if (!is_q3sel || !options.disable_other_filters) {
                uint64_t lsd_nstats = metadata.table_lineitem_nstats[l_sd_field];
                uint64_t lsd_stats_start = metadata.table_lineitem_stats_start_page_ids[l_sd_field];
                uint64_t lsd_stats_npg = metadata.table_lineitem_stats_npages[l_sd_field];
                if (lsd_nstats > 0 && lsd_stats_start > 0) {
                    for (uint64_t j = 0; j < lsd_stats_npg; j++) {
                        uint64_t pg_id = lsd_stats_start + j;
                        uint32_t dev = pg_id % n_devices;
                        uint64_t local = pg_id / n_devices;
                        zm_li_ctx.h_reads[zm_li_nreads++] = {
                            ds.partition_start_lbas[dev] + local * blocks_per_page,
                            static_cast<uint32_t>(blocks_per_page), dev};
                    }
                    zm_li_ctx.h_preds[zm_li_npreds++] = {0, lsd_nstats,
                        19950316, INT32_MAX};
                }
            }

            // Sideways O_ORDERDATE stats (Q3, and Q3SEL when filters enabled)
            if (!is_q3sel || !options.disable_other_filters) {
                const size_t sw_odate_idx = TPCH::common::LS_SIDEWAYS_O_ORDERDATE;
                uint64_t sw_odate_nstats      = metadata.table_lineitem_sideways_nstats[l_sd_field][sw_odate_idx];
                uint64_t sw_odate_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[l_sd_field][sw_odate_idx];
                uint64_t sw_odate_stats_npg   = metadata.table_lineitem_sideways_stats_npages[l_sd_field][sw_odate_idx];
                if (sw_odate_nstats > 0 && sw_odate_stats_start > 0) {
                    uint32_t stats_page_offset = zm_li_nreads;
                    for (uint64_t j = 0; j < sw_odate_stats_npg; j++) {
                        uint64_t pg_id = sw_odate_stats_start + j;
                        uint32_t dev = pg_id % n_devices;
                        uint64_t local = pg_id / n_devices;
                        zm_li_ctx.h_reads[zm_li_nreads++] = {
                            ds.partition_start_lbas[dev] + local * blocks_per_page,
                            static_cast<uint32_t>(blocks_per_page), dev};
                    }
                    zm_li_ctx.h_preds[zm_li_npreds++] = {stats_page_offset, sw_odate_nstats,
                        INT32_MIN, 19950314};
                }
            }

            // Sideways C_MKTSEGMENT stats (both Q3 and Q3SEL)
            const size_t li_sw_mktseg_idx = TPCH::common::LS_SIDEWAYS_C_MKTSEGMENT;
            uint64_t li_sw_mktseg_nstats      = metadata.table_lineitem_sideways_nstats[l_sd_field][li_sw_mktseg_idx];
            uint64_t li_sw_mktseg_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[l_sd_field][li_sw_mktseg_idx];
            uint64_t li_sw_mktseg_stats_npg   = metadata.table_lineitem_sideways_stats_npages[l_sd_field][li_sw_mktseg_idx];
            if (li_sw_mktseg_nstats > 0 && li_sw_mktseg_stats_start > 0) {
                uint32_t stats_page_offset = zm_li_nreads;
                for (uint64_t j = 0; j < li_sw_mktseg_stats_npg; j++) {
                    uint64_t pg_id = li_sw_mktseg_stats_start + j;
                    uint32_t dev = pg_id % n_devices;
                    uint64_t local = pg_id / n_devices;
                    zm_li_ctx.h_reads[zm_li_nreads++] = {
                        ds.partition_start_lbas[dev] + local * blocks_per_page,
                        static_cast<uint32_t>(blocks_per_page), dev};
                }
                if (is_q3sel)
                    zm_li_ctx.h_preds[zm_li_npreds++] = {stats_page_offset, li_sw_mktseg_nstats,
                        q3sel_mktseg_lo, q3sel_mktseg_hi};
                else
                    zm_li_ctx.h_preds[zm_li_npreds++] = {stats_page_offset, li_sw_mktseg_nstats,
                        1, 1};
            }
        }
        zm_li_valid = (zm_li_npreds > 0);
    }

    // Pre-allocate Phase 1 flat buffer (outside timing)
    uint64_t *d_c_custkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c_custkey_flat, nrecs_customer * sizeof(uint64_t)));

    // Upload CUSTOMER compression metadata to GPU (for fused kernels)
    uint64_t *d_ck_comp_offsets = nullptr;
    uint32_t *d_ck_comp_sizes = nullptr;
    if (field_comp_methods[FI_C_CUSTKEY] != CompressionMethod::NONE) {
        CUDA_CHECK(cudaMalloc(&d_ck_comp_offsets, npages_custkey * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_ck_comp_offsets, h_comp_offsets[FI_C_CUSTKEY].data(),
                    npages_custkey * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_ck_comp_sizes, npages_custkey * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_ck_comp_sizes, h_comp_sizes[FI_C_CUSTKEY].data(),
                    npages_custkey * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    uint64_t *d_mk_comp_offsets = nullptr;
    uint32_t *d_mk_comp_sizes = nullptr;
    if (field_comp_methods[FI_C_MKTSEGMENT] != CompressionMethod::NONE) {
        CUDA_CHECK(cudaMalloc(&d_mk_comp_offsets, npages_mktseg * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_mk_comp_offsets, h_comp_offsets[FI_C_MKTSEGMENT].data(),
                    npages_mktseg * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_mk_comp_sizes, npages_mktseg * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_mk_comp_sizes, h_comp_sizes[FI_C_MKTSEGMENT].data(),
                    npages_mktseg * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // Upload CUSTOMER prefix sums to GPU (for fused kernels)
    uint64_t *d_ps_custkey = nullptr, *d_ps_mktseg = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_custkey, npages_custkey * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_custkey, h_prefix_sum[FI_C_CUSTKEY].data(),
                npages_custkey * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ps_mktseg, npages_mktseg * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_mktseg, h_prefix_sum[FI_C_MKTSEGMENT].data(),
                npages_mktseg * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Q3 CUSTOMER Phase 1: nocopy page_cache + IO/CK decomp kernel + MK decomp kernel
    uint32_t q3_cust_total_pages = static_cast<uint32_t>(npages_custkey + npages_mktseg);
    BamBulkReadCtx q3_cust_nocopy = bam_bulk_read_nocopy_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        std::min(q3_cust_total_pages, (uint32_t)(sm_count * 16)),
        q3_cust_total_pages);

    uint32_t q3_k1_max = q3_cust_io_decomp_max_blocks(static_cast<uint32_t>(page_size));
    uint32_t q3_k2_max = q3_cust_process_max_blocks();
    // Kernel 1 grid: ceil(total_pages / 4 warps), capped by occupancy
    uint32_t q3_k1_blocks = std::min(
        (q3_cust_total_pages + 3) / 4, q3_k1_max);
    // Kernel 2 grid: cooperative launch — use max occupancy
    uint32_t q3_k2_blocks = q3_k2_max;
    // Staging buffer: page-indexed (total_pages * page_size)
    char* q3_p1_staging = nullptr;
    CUDA_CHECK(cudaMalloc(&q3_p1_staging, (size_t)q3_cust_total_pages * page_size));

    // ORDERS fused kernel blocks (full page count; kernel skips pruned pages)
    fused_q3ord_num_blocks = std::min(static_cast<uint32_t>(npages_o_i32),
        static_cast<uint32_t>(sm_count * 4));

    // Active page counts (computed from h_mask after zone map eval, for logging)
    uint32_t n_ord_active = 0;
    uint32_t n_li_active = 0;

    // Warp-spec LINEITEM: query max blocks (full page count; kernel skips pruned pages)
    uint32_t q3li_ws_max = q3li_warp_spec_max_blocks(static_cast<uint32_t>(page_size));
    fused_q3li_num_blocks = std::min(static_cast<uint32_t>(npages_l_i32), q3li_ws_max);

    // Shared page_cache + decomp_buf for ORDERS (Phase 2) and LINEITEM (Phase 3).
    // Both phases run sequentially, so the same GPU memory is reused.
    // Q3SEL: Phase 2 uses IO+decomp → separate scan (staging in decomp_buf),
    //        Phase 3 uses tiled IO+decomp → separate scan (staging in decomp_buf).
    // Q3:    Phase 2 uses fused IO+decomp+scan, Phase 3 uses fused warp-spec.
    constexpr uint32_t Q3LI_SLOTS_PER_BLOCK = 84;   // 2 * 7 * 6
    constexpr uint32_t ORD_SLOTS_PER_BLOCK  = 16;   // NBUF(2) * MAX_FIELDS(8)
    uint32_t ord_slots_needed = fused_q3ord_num_blocks * ORD_SLOTS_PER_BLOCK;
    uint32_t shared_slots;
    {
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t app_used = gpu_free_before_app - gpu_free_now;
        uint64_t app_budget = (gpu_ctrl_bytes < GPU_MEM_BUDGET)
            ? GPU_MEM_BUDGET - gpu_ctrl_bytes : 0;
        if (app_used < app_budget) {
            uint64_t remaining = app_budget - app_used;
            uint32_t max_shared_slots = static_cast<uint32_t>(
                remaining / (2 * page_size + BAM_PC_OVERHEAD_PER_SLOT));
            uint32_t max_li_blocks = max_shared_slots / Q3LI_SLOTS_PER_BLOCK;
            if (max_li_blocks < fused_q3li_num_blocks) {
                std::cout << "[Q3] Budget cap: LINEITEM fused blocks "
                          << fused_q3li_num_blocks << " → " << max_li_blocks
                          << std::endl;
                fused_q3li_num_blocks = max_li_blocks;
            }
            uint32_t q3li_pc_slots = fused_q3li_num_blocks * Q3LI_SLOTS_PER_BLOCK;
            shared_slots = std::max(ord_slots_needed, q3li_pc_slots);
        } else {
            fused_q3li_num_blocks = std::min(fused_q3li_num_blocks, 1u);
            shared_slots = std::max(ord_slots_needed, fused_q3li_num_blocks * Q3LI_SLOTS_PER_BLOCK);
        }
    }

    q3li_ws_pc = bam_io_page_cache_create(
        ctrl, static_cast<uint32_t>(page_size), shared_slots);
    void*       q3li_ws_ctrls    = bam_io_page_cache_get_d_ctrls(q3li_ws_pc);
    void*       q3li_ws_pc_ptr   = bam_io_page_cache_get_d_pc_ptr(q3li_ws_pc);
    const char* q3li_ws_pc_base  = (const char*)bam_io_page_cache_get_base_addr(q3li_ws_pc);
    size_t shared_decomp_size = (size_t)shared_slots * page_size;
    CUDA_CHECK(cudaMalloc(&q3li_ws_decomp, shared_decomp_size));

    // ORDERS context: borrow shared page_cache + decomp_buf (Q3 only; Q3SEL uses separate scan)
    if (!is_q3sel) {
        fused_q3ord_ctx = bam_fused_q3ord_create_shared(
            q3li_ws_pc, q3li_ws_decomp,
            static_cast<uint32_t>(page_size), fused_q3ord_num_blocks);
    }

    // Measure GPU memory after all allocs (Phase 1-3 buffers + fused contexts)
    {
        size_t gpu_free_after_app = 0;
        cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
        gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;
    }

    // Pre-allocate D2H result buffer (outside measurement interval, RULES.md)
    Q3ResultRow *h_results = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_results, aggr_cap * sizeof(Q3ResultRow)));

    // Pre-allocate Q3SEL GPU-side IO pruning buffers (outside measurement interval)
    constexpr uint32_t Q3SEL_MAX_TILES = 256;
    uint8_t*       d_q3sel_i64_mask     = nullptr;
    uint32_t*      d_q3sel_needed_i64   = nullptr;
    uint32_t*      d_q3sel_n_needed_i64 = nullptr;
    uint32_t*      d_q3sel_i64_remap    = nullptr;
    uint64_t*      d_q3sel_active_ps    = nullptr;
    uint64_t*      d_q3sel_nrecs        = nullptr;
    Q3SelTileInfo* d_q3sel_tiles        = nullptr;
    uint32_t*      d_q3sel_n_tiles      = nullptr;
    uint32_t*      d_all_ord_pages      = nullptr;
    uint32_t*      d_all_li_pages       = nullptr;
    uint32_t*      h_q3sel_n_needed_i64 = nullptr;
    uint64_t*      h_q3sel_nrecs        = nullptr;
    uint32_t*      h_q3sel_n_tiles      = nullptr;
    Q3SelTileInfo* h_q3sel_tiles        = nullptr;

    if (is_q3sel) {
        uint32_t max_npages_i64 = static_cast<uint32_t>(
            std::max(npages_o_i64, npages_l_i64));
        uint32_t max_npages_i32 = static_cast<uint32_t>(
            std::max(npages_o_i32, npages_l_i32));

        CUDA_CHECK(cudaMalloc(&d_q3sel_i64_mask, max_npages_i64 * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_q3sel_needed_i64, max_npages_i64 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_q3sel_n_needed_i64, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_q3sel_i64_remap, max_npages_i64 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_q3sel_active_ps, (max_npages_i32 + 1) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_q3sel_nrecs, sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_q3sel_tiles, Q3SEL_MAX_TILES * sizeof(Q3SelTileInfo)));
        CUDA_CHECK(cudaMalloc(&d_q3sel_n_tiles, sizeof(uint32_t)));

        CUDA_CHECK(cudaMallocHost(&h_q3sel_n_needed_i64, sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_q3sel_nrecs, sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocHost(&h_q3sel_n_tiles, sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_q3sel_tiles, Q3SEL_MAX_TILES * sizeof(Q3SelTileInfo)));

        // Fallback iota arrays for when zone map is disabled
        {
            std::vector<uint32_t> iota_ord(npages_o_i32), iota_li(npages_l_i32);
            for (uint32_t i = 0; i < npages_o_i32; i++) iota_ord[i] = i;
            for (uint32_t i = 0; i < npages_l_i32; i++) iota_li[i] = i;
            CUDA_CHECK(cudaMalloc(&d_all_ord_pages, npages_o_i32 * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_all_ord_pages, iota_ord.data(),
                npages_o_i32 * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_all_li_pages, npages_l_i32 * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_all_li_pages, iota_li.data(),
                npages_l_i32 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }

    // Zone map INT64 mask buffers (allocated outside timing)
    uint8_t* d_mask_ord_i64 = nullptr;
    uint8_t* d_mask_li_i64 = nullptr;
    if (zm_ord_valid || zm_li_valid) {
        if (zm_ord_valid) cudaMalloc(&d_mask_ord_i64, npages_o_i64);
        if (zm_li_valid)  cudaMalloc(&d_mask_li_i64, npages_l_i64);
    }

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_ord_valid || zm_li_valid) {
        bam_pre_io(zm_ord_ctx.d_ctrls, zm_ord_ctx.d_pc, stream);
    }

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    size_t total_io_count = 0, total_io_bytes = 0;

    // ═══════════════════════════════════════════════════════
    // Phase 1: CUSTOMER — IO(CK+MK) + CK decomp/flatten, then MK decomp/scan/hash
    // ═══════════════════════════════════════════════════════
    {
        // Build descriptors: CK first (slots 0..ck-1), then MK (slots ck..ck+mk-1)
        {
            BamBulkReadDesc* h_descs = q3_cust_nocopy.h_descs[0];
            for (uint32_t pg = 0; pg < npages_custkey; pg++) {
                BamBulkReadDesc& d = h_descs[pg];
                uint64_t global_pg = field_start_page_ids[FI_C_CUSTKEY] + pg;
                d.device = global_pg % n_devices;
                if (field_comp_methods[FI_C_CUSTKEY] != CompressionMethod::NONE) {
                    d.lba = ds.partition_start_lbas[d.device] + h_comp_offsets[FI_C_CUSTKEY][pg] / 512;
                    uint32_t comp_sz = h_comp_sizes[FI_C_CUSTKEY][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 17;
                    d.nblocks = nblk;
                } else {
                    uint64_t local_pg = global_pg / n_devices;
                    d.lba = ds.partition_start_lbas[d.device] + local_pg * (page_size / 512);
                    d.nblocks = page_size / 512;
                }
                d.dest = nullptr;
                d.copy_bytes = 0;
            }
            for (uint32_t pg = 0; pg < npages_mktseg; pg++) {
                BamBulkReadDesc& d = h_descs[npages_custkey + pg];
                uint64_t global_pg = field_start_page_ids[FI_C_MKTSEGMENT] + pg;
                d.device = global_pg % n_devices;
                if (field_comp_methods[FI_C_MKTSEGMENT] != CompressionMethod::NONE) {
                    d.lba = ds.partition_start_lbas[d.device] + h_comp_offsets[FI_C_MKTSEGMENT][pg] / 512;
                    uint32_t comp_sz = h_comp_sizes[FI_C_MKTSEGMENT][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 17;
                    d.nblocks = nblk;
                } else {
                    uint64_t local_pg = global_pg / n_devices;
                    d.lba = ds.partition_start_lbas[d.device] + local_pg * (page_size / 512);
                    d.nblocks = page_size / 512;
                }
                d.dest = nullptr;
                d.copy_bytes = 0;
            }
        }

        // Upload descriptors to GPU
        CUDA_CHECK(cudaMemcpyAsync(q3_cust_nocopy.d_descs[0], q3_cust_nocopy.h_descs[0],
            q3_cust_total_pages * sizeof(BamBulkReadDesc), cudaMemcpyHostToDevice, stream));

        // Kernel 1: IO all pages + decomp all pages to staging (1 kl)
        {
            Q3CustIODecompParams io_params = {};
            io_params.total_descs = q3_cust_total_pages;
            io_params.ck_npages = static_cast<uint32_t>(npages_custkey);
            io_params.ck_comp_sizes = d_ck_comp_sizes;
            io_params.mk_comp_sizes = d_mk_comp_sizes;
            io_params.page_size = static_cast<uint32_t>(page_size);

            q3_cust_io_decomp_launch(
                q3_cust_nocopy.d_ctrls, q3_cust_nocopy.d_pc,
                (const char*)q3_cust_nocopy.pc_base,
                q3_cust_nocopy.d_descs[0],
                q3_p1_staging, io_params, q3_k1_blocks, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Kernel 2: CK flatten + MK scan + hash insert (cooperative, 1 kl)
        CUDA_CHECK(cudaMemsetAsync(d_custkey_set, 0xFF, custset_cap * sizeof(uint64_t), stream));
        {
            constexpr uint32_t CHAR_MKTSEG_PADDED_LEN = 12;
            Q3CustProcessParams proc_params = {};
            proc_params.d_staging = q3_p1_staging;
            proc_params.page_size = static_cast<uint32_t>(page_size);
            proc_params.ck_prefix_sum = d_ps_custkey;
            proc_params.ck_npages = static_cast<uint32_t>(npages_custkey);
            proc_params.nrecs_customer = nrecs_customer;
            proc_params.d_c_custkey_flat = d_c_custkey_flat;
            proc_params.mk_page_offset = static_cast<uint32_t>(npages_custkey);
            proc_params.mk_prefix_sum = d_ps_mktseg;
            proc_params.mk_npages = static_cast<uint32_t>(npages_mktseg);
            proc_params.padded_len = CHAR_MKTSEG_PADDED_LEN;
            proc_params.d_custkey_set = d_custkey_set;
            proc_params.custkey_set_mask = custset_mask;

            // Q3SEL: multi-segment support
            static constexpr uint64_t Q3SEL_SEGMENTS[5] = {
                0x474E49444C495542ULL, // BUILDING
                0x49424F4D4F545541ULL, // AUTOMOBILE
                0x525554494E525546ULL, // FURNITURE
                0x52454E494843414DULL, // MACHINERY
                0x4C4F484553554F48ULL, // HOUSEHOLD
            };
            if (is_q3sel) {
                uint32_t nseg = (sel_pct >= 100) ? 0 : (uint32_t)(sel_pct / 20);
                if (nseg == 0 && sel_pct > 0 && sel_pct < 100) nseg = 1;
                proc_params.num_segments = nseg;
                for (uint32_t s = 0; s < 5; s++)
                    proc_params.segment_values[s] = Q3SEL_SEGMENTS[s];
            } else {
                proc_params.num_segments = 1;
                proc_params.segment_values[0] = Q3SEL_SEGMENTS[0]; // BUILDING only
            }

            q3_cust_process_launch(proc_params, q3_k2_blocks, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        std::cout << "[Q3] CUSTOMER hash set built (capacity=" << custset_cap << ")" << std::endl;
    }

    auto phase1_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase1_end - total_start).count();
        std::cout << "[Q3-TIMING] Phase 1 (CUSTOMER): " << ms << " ms" << std::endl;
    }


    // ── Zone map eval (GPU-side, inside timing, with INT64 mask derivation) ──
    if (zm_ord_valid) {
        zm_ord_ctx.d_ps_i32   = d_ps_o_i32 + 1;
        zm_ord_ctx.d_ps_i64   = d_ps_o_i64 + 1;
        zm_ord_ctx.d_mask_i64 = d_mask_ord_i64;
        zm_ord_ctx.npages_i64 = static_cast<uint32_t>(npages_o_i64);
        bam_zonemap_eval_async(zm_ord_ctx, npages_o_i32, zm_ord_nreads, zm_ord_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
    }
    if (zm_li_valid) {
        zm_li_ctx.d_ps_i32   = d_ps_li_i32 + 1;
        zm_li_ctx.d_ps_i64   = d_ps_li_i64 + 1;
        zm_li_ctx.d_mask_i64 = d_mask_li_i64;
        zm_li_ctx.npages_i64 = static_cast<uint32_t>(npages_l_i64);
        bam_zonemap_eval_async(zm_li_ctx, npages_l_i32, zm_li_nreads, zm_li_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
    }

    // GPU-side active page IDs for Q3SEL kernels
    uint32_t *d_ord_active_ids = zm_ord_valid ? zm_ord_ctx.d_active_ids : d_all_ord_pages;
    uint32_t *d_li_active_ids  = zm_li_valid  ? zm_li_ctx.d_active_ids  : d_all_li_pages;
    n_ord_active = zm_ord_valid ? *zm_ord_ctx.h_num_active : static_cast<uint32_t>(npages_o_i32);
    n_li_active  = zm_li_valid  ? *zm_li_ctx.h_num_active  : static_cast<uint32_t>(npages_l_i32);

    // ═══════════════════════════════════════════════════════════
    // Phase 2: ORDERS
    //   Q3SEL: IO+decomp → separate scan (full GPU parallelism for scan)
    //   Q3:    fused BaM I/O + nvCOMPdx LZ4 + probe CUSTOMER + build HT
    // ═══════════════════════════════════════════════════════════
    {
        // Initialize ORDERS HT
        CUDA_CHECK(cudaMemsetAsync(d_orders_ht_keys, 0xFF, orders_ht_cap * sizeof(uint64_t), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (is_q3sel) {
            // ── Q3SEL ORDERS: GPU-side IO pruning + IO+decomp → scan ──
            char* d_ord_staging = q3li_ws_decomp;

            // Derive needed INT64 pages (GPU kernel)
            q3sel_derive_i64_launch(
                d_ord_active_ids, n_ord_active,
                d_ps_o_i32, d_ps_o_i64, static_cast<uint32_t>(npages_o_i64),
                d_q3sel_i64_mask, d_q3sel_needed_i64, d_q3sel_n_needed_i64,
                stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            s_kernel_launches++;
            CUDA_CHECK(cudaMemcpy(h_q3sel_n_needed_i64, d_q3sel_n_needed_i64,
                sizeof(uint32_t), cudaMemcpyDeviceToHost));
            uint32_t n_needed_i64 = *h_q3sel_n_needed_i64;

            // Per-tile setup: i64_remap + active_ps + nrecs (GPU kernel)
            q3sel_tile_setup_launch(
                d_ord_active_ids, 0, n_ord_active,
                d_q3sel_needed_i64, 0, n_needed_i64,
                d_ps_o_i32, static_cast<uint32_t>(npages_o_i64),
                d_q3sel_active_ps, d_q3sel_i64_remap, d_q3sel_nrecs,
                stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            s_kernel_launches++;
            CUDA_CHECK(cudaMemcpy(h_q3sel_nrecs, d_q3sel_nrecs,
                sizeof(uint64_t), cudaMemcpyDeviceToHost));
            uint64_t nrecs_ord_active = *h_q3sel_nrecs;

            uint32_t odate_pg_off = 0;
            uint32_t sp_pg_off    = n_ord_active;
            uint32_t okey_pg_off  = n_ord_active * 2;
            uint32_t ckey_pg_off  = n_ord_active * 2 + n_needed_i64;
            uint32_t total_descs  = n_ord_active * 2 + n_needed_i64 * 2;

            uint32_t io_decomp_blocks = std::min(
                q3sel_io_decomp_max_blocks(static_cast<uint32_t>(page_size)),
                total_descs);

            std::cout << "[GIDP+BAM+DECOMP Q3SEL] ORDERS IO+decomp: "
                      << total_descs << " pages (" << n_ord_active << " i32 active, "
                      << n_needed_i64 << " i64 needed), "
                      << io_decomp_blocks << " blocks" << std::endl;

            Q3SelIODecompParams io_params = {};
            io_params.d_active_ids = d_ord_active_ids;
            io_params.n_active = n_ord_active;
            io_params.d_needed_i64 = d_q3sel_needed_i64;
            io_params.n_needed_i64 = n_needed_i64;
            io_params.n_i32_fields = 2;
            io_params.field_start_page_ids_i32[0] = field_start_page_ids[FI_O_ORDERDATE];
            io_params.field_start_page_ids_i32[1] = field_start_page_ids[FI_O_SHIPPRIORITY];
            io_params.d_comp_offsets_i32[0] = d_o_comp_offsets_i32[0];
            io_params.d_comp_offsets_i32[1] = d_o_comp_offsets_i32[1];
            io_params.d_comp_sizes_i32[0] = d_o_comp_sizes_i32[0];
            io_params.d_comp_sizes_i32[1] = d_o_comp_sizes_i32[1];
            io_params.is_compressed_i32[0] = (field_comp_methods[FI_O_ORDERDATE] != CompressionMethod::NONE);
            io_params.is_compressed_i32[1] = (field_comp_methods[FI_O_SHIPPRIORITY] != CompressionMethod::NONE);
            io_params.n_i64_fields = 2;
            io_params.field_start_page_ids_i64[0] = field_start_page_ids[FI_O_ORDERKEY];
            io_params.field_start_page_ids_i64[1] = field_start_page_ids[FI_O_CUSTKEY];
            io_params.d_comp_offsets_i64[0] = d_o_comp_offsets_i64[0];
            io_params.d_comp_offsets_i64[1] = d_o_comp_offsets_i64[1];
            io_params.d_comp_sizes_i64[0] = d_o_comp_sizes_i64[0];
            io_params.d_comp_sizes_i64[1] = d_o_comp_sizes_i64[1];
            io_params.is_compressed_i64[0] = (field_comp_methods[FI_O_ORDERKEY] != CompressionMethod::NONE);
            io_params.is_compressed_i64[1] = (field_comp_methods[FI_O_CUSTKEY] != CompressionMethod::NONE);
            for (uint32_t d = 0; d < n_devices && d < 4; d++)
                io_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
            io_params.n_devices = n_devices;
            io_params.page_size = static_cast<uint32_t>(page_size);

            q3sel_io_decomp_launch(
                q3li_ws_ctrls, q3li_ws_pc_ptr, q3li_ws_pc_base,
                d_ord_staging, io_params, io_decomp_blocks, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));

            Q3SelOrdersScanParams scan_params = {};
            scan_params.d_staging = d_ord_staging;
            scan_params.page_size = static_cast<uint32_t>(page_size);
            scan_params.odate_pg_off = odate_pg_off;
            scan_params.sp_pg_off = sp_pg_off;
            scan_params.okey_pg_off = okey_pg_off;
            scan_params.ckey_pg_off = ckey_pg_off;
            scan_params.n_active_i32 = n_ord_active;
            scan_params.d_active_ps_i32 = d_q3sel_active_ps;
            scan_params.d_active_pages_i32 = d_ord_active_ids;
            scan_params.d_ps_i32_full = d_ps_o_i32;
            scan_params.d_ps_i64_full = d_ps_o_i64;
            scan_params.npages_i64 = static_cast<uint32_t>(npages_o_i64);
            scan_params.d_i64_remap = d_q3sel_i64_remap;
            scan_params.d_custkey_set = d_custkey_set;
            scan_params.custkey_set_mask = custset_mask;
            scan_params.d_orders_ht_keys = d_orders_ht_keys;
            scan_params.d_orders_ht_payloads = d_orders_ht_payloads;
            scan_params.orders_ht_mask = orders_ht_mask;
            scan_params.o_orderdate_limit = options.disable_other_filters ? 0 : 19950315;
            scan_params.nrecs_active = nrecs_ord_active;

            q3sel_orders_scan_launch(scan_params, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));

            std::cout << "[Q3] ORDERS HT built (capacity=" << orders_ht_cap << ")" << std::endl;
        } else {
            // ── Q3: fused IO+decomp+scan ──
            BAMFusedQ3OrdParams fused_q3ord_params = {};
            fused_q3ord_params.i32_field_start_page_ids[0] = field_start_page_ids[FI_O_ORDERDATE];
            fused_q3ord_params.i32_field_start_page_ids[1] = field_start_page_ids[FI_O_SHIPPRIORITY];
            for (int k = 0; k < 2; k++) {
                fused_q3ord_params.d_comp_offsets_i32[k] = d_o_comp_offsets_i32[k];
                fused_q3ord_params.d_comp_sizes_i32[k] = d_o_comp_sizes_i32[k];
            }
            fused_q3ord_params.is_compressed_i32[0] = (field_comp_methods[FI_O_ORDERDATE] != CompressionMethod::NONE);
            fused_q3ord_params.is_compressed_i32[1] = (field_comp_methods[FI_O_SHIPPRIORITY] != CompressionMethod::NONE);
            fused_q3ord_params.i64_field_start_page_ids[0] = field_start_page_ids[FI_O_ORDERKEY];
            fused_q3ord_params.i64_field_start_page_ids[1] = field_start_page_ids[FI_O_CUSTKEY];
            for (int k = 0; k < 2; k++) {
                fused_q3ord_params.d_comp_offsets_i64[k] = d_o_comp_offsets_i64[k];
                fused_q3ord_params.d_comp_sizes_i64[k] = d_o_comp_sizes_i64[k];
            }
            fused_q3ord_params.is_compressed_i64[0] = (field_comp_methods[FI_O_ORDERKEY] != CompressionMethod::NONE);
            fused_q3ord_params.is_compressed_i64[1] = (field_comp_methods[FI_O_CUSTKEY] != CompressionMethod::NONE);
            fused_q3ord_params.d_ps_i32 = d_ps_o_i32;
            fused_q3ord_params.d_ps_i64 = d_ps_o_i64;
            fused_q3ord_params.npages_i32 = static_cast<uint32_t>(npages_o_i32);
            fused_q3ord_params.npages_i64 = static_cast<uint32_t>(npages_o_i64);
            for (uint32_t d = 0; d < n_devices && d < 4; d++)
                fused_q3ord_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
            fused_q3ord_params.n_devices = n_devices;
            fused_q3ord_params.page_size = static_cast<uint32_t>(page_size);
            fused_q3ord_params.num_blocks = fused_q3ord_num_blocks;
            fused_q3ord_params.d_custkey_set = d_custkey_set;
            fused_q3ord_params.custkey_set_mask = custset_mask;
            fused_q3ord_params.d_orders_ht_keys = d_orders_ht_keys;
            fused_q3ord_params.d_orders_ht_payloads = d_orders_ht_payloads;
            fused_q3ord_params.orders_ht_mask = orders_ht_mask;
            fused_q3ord_params.d_active_page_ids = zm_ord_valid ? zm_ord_ctx.d_active_ids : nullptr;
            fused_q3ord_params.d_page_mask = zm_ord_valid ? zm_ord_ctx.d_mask : nullptr;
            if (zm_ord_valid) {
                fused_q3ord_params.total_pages = *zm_ord_ctx.h_num_active;
                fused_q3ord_num_blocks = std::min(*zm_ord_ctx.h_num_active, fused_q3ord_num_blocks);
            } else {
                fused_q3ord_params.total_pages = npages_o_i32;
            }
            fused_q3ord_params.skip_date_filter = false;

            std::cout << "[GIDP+BAM+DECOMP Q3] Fused ORDERS kernel: "
                      << fused_q3ord_num_blocks << " persistent blocks, "
                      << n_ord_active << "/" << npages_o_i32 << " active INT32 pages, "
                      << npages_o_i64 << " INT64 pages" << std::endl;

            bam_fused_q3ord_run_async(fused_q3ord_ctx, fused_q3ord_params, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::cout << "[Q3] ORDERS HT built (capacity=" << orders_ht_cap << ")" << std::endl;
        }
    }


    auto phase2_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase2_end - phase1_end).count();
        std::cout << "[Q3-TIMING] Phase 2 (ORDERS): " << ms << " ms" << std::endl;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 3: LINEITEM
    //   Q3SEL: tiled IO+decomp → separate scan (full GPU parallelism)
    //   Q3:    fused warp-spec kernel (IO || Decomp → Scan)
    // ═══════════════════════════════════════════════════════════════════
    CUDA_CHECK(cudaMemsetAsync(d_aggr_keys, 0xFF, aggr_cap * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_aggr_revenues, 0, aggr_cap * sizeof(int64_t), stream));

    {
        Q3LIWarpSpecParams ws_params = {};
        ws_params.i32_field_start_page_ids[0] = field_start_page_ids[FI_L_SHIPDATE];
        ws_params.i32_field_start_page_ids[1] = field_start_page_ids[FI_L_EXTPRICE];
        ws_params.i32_field_start_page_ids[2] = field_start_page_ids[FI_L_DISCOUNT];
        for (int k = 0; k < 3; k++) {
            ws_params.d_comp_offsets_i32[k] = d_li_comp_offsets_i32[k];
            ws_params.d_comp_sizes_i32[k] = d_li_comp_sizes_i32[k];
        }
        ws_params.is_compressed_i32[0] = (field_comp_methods[FI_L_SHIPDATE] != CompressionMethod::NONE);
        ws_params.is_compressed_i32[1] = (field_comp_methods[FI_L_EXTPRICE] != CompressionMethod::NONE);
        ws_params.is_compressed_i32[2] = (field_comp_methods[FI_L_DISCOUNT] != CompressionMethod::NONE);
        ws_params.i64_field_start_page_ids[0] = field_start_page_ids[FI_L_ORDERKEY];
        ws_params.d_comp_offsets_i64[0] = d_li_comp_offsets_i64;
        ws_params.d_comp_sizes_i64[0] = d_li_comp_sizes_i64;
        ws_params.is_compressed_i64[0] = (field_comp_methods[FI_L_ORDERKEY] != CompressionMethod::NONE);
        ws_params.d_ps_i32 = d_ps_li_i32;
        ws_params.d_ps_i64 = d_ps_li_i64;
        ws_params.npages_i32 = static_cast<uint32_t>(npages_l_i32);
        ws_params.npages_i64 = static_cast<uint32_t>(npages_l_i64);
        for (uint32_t d = 0; d < n_devices && d < 4; d++)
            ws_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        ws_params.n_devices = n_devices;
        ws_params.page_size = static_cast<uint32_t>(page_size);
        ws_params.d_active_page_ids = zm_li_valid ? zm_li_ctx.d_active_ids : nullptr;
        ws_params.d_page_mask = zm_li_valid ? zm_li_ctx.d_mask : nullptr;
        if (zm_li_valid) {
            ws_params.total_pages = *zm_li_ctx.h_num_active;
            fused_q3li_num_blocks = std::min(*zm_li_ctx.h_num_active, fused_q3li_num_blocks);
        } else {
            ws_params.total_pages = static_cast<uint32_t>(npages_l_i32);
        }
        ws_params.d_orders_ht_keys = d_orders_ht_keys;
        ws_params.d_orders_ht_payloads = d_orders_ht_payloads;
        ws_params.orders_ht_mask = orders_ht_mask;
        ws_params.d_aggr_keys = d_aggr_keys;
        ws_params.d_aggr_revenues = d_aggr_revenues;
        ws_params.aggr_mask = aggr_mask;
        ws_params.skip_shipdate_filter = is_q3sel && options.disable_other_filters;

        std::cout << "[GIDP+BAM+WARPSPEC Q3] LINEITEM kernel: "
                  << fused_q3li_num_blocks << " blocks, "
                  << n_li_active << "/" << npages_l_i32 << " active INT32 pages, "
                  << npages_l_i64 << " INT64 pages" << std::endl;

        q3li_warp_spec_launch(
            q3li_ws_ctrls, q3li_ws_pc_ptr, q3li_ws_pc_base,
            q3li_ws_decomp, ws_params, fused_q3li_num_blocks, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto phase3_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(phase3_end - phase2_end).count();
        std::cout << "[Q3-TIMING] Phase 3 (LINEITEM): " << ms << " ms" << std::endl;
    }

    // Reuse Phase 2-3 buffers for Phase 4 (no alloc/free overhead):
    //   d_results + d_result_count → decomp_buf
    //   d_sort_temp               → page_cache data buffer
    {
        size_t results_bytes = (size_t)aggr_cap * sizeof(Q3ResultRow);
        size_t count_offset = (results_bytes + 3) & ~(size_t)3;
        d_results = reinterpret_cast<Q3ResultRow*>(q3li_ws_decomp);
        d_result_count = reinterpret_cast<uint32_t*>(q3li_ws_decomp + count_offset);
        d_sort_temp = const_cast<char*>(q3li_ws_pc_base);
    }

    // ── Phase 4: Collect + Top-10 ──
    CUDA_CHECK(cudaMemsetAsync(d_result_count, 0, sizeof(uint32_t), stream));

    q3_collect_results(d_aggr_keys, d_aggr_revenues, aggr_cap,
                       d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                       d_results, d_result_count, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint32_t h_result_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Sort on GPU (CUB DeviceMergeSort with pre-allocated temp)
    cub::DeviceMergeSort::SortKeys(d_sort_temp, sort_temp_bytes,
        d_results, (int)h_result_count, Q3ResultCmp{}, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_results, d_results,
                          h_result_count * sizeof(Q3ResultRow), cudaMemcpyDeviceToHost));

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(total_end - phase3_end).count();
        std::cout << "[Q3-TIMING] Phase 4 (Collect): " << ms << " ms" << std::endl;
    }
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

    // Print top 10
    if (is_q3sel)
        std::cout << "\n=== TPC-H Q3SEL Result (sel=" << sel_pct << "%, Top 10) ===" << std::endl;
    else
        std::cout << "\n=== TPC-H Q3 Result (Top 10) ===" << std::endl;
    std::cout << "l_orderkey |       revenue | o_orderdate | o_shippriority" << std::endl;
    std::cout << "-----------+---------------+-------------+---------------" << std::endl;
    uint32_t limit = std::min(h_result_count, (uint32_t)10);
    for (uint32_t i = 0; i < limit; i++) {
        auto &r = h_results[i];
        printf("%10lu | %13ld | %11u | %14u\n",
               (unsigned long)r.l_orderkey, (long)r.revenue,
               r.o_orderdate, r.o_shippriority);
    }
    std::cout << std::endl;

    // Read back INT64 masks from GPU for IO accounting
    std::vector<uint8_t> h_ord_i64(npages_o_i64, 0);
    std::vector<uint8_t> h_li_i64(npages_l_i64, 0);
    if (zm_ord_valid)
        cudaMemcpy(h_ord_i64.data(), d_mask_ord_i64, npages_o_i64, cudaMemcpyDeviceToHost);
    if (zm_li_valid)
        cudaMemcpy(h_li_i64.data(), d_mask_li_i64, npages_l_i64, cudaMemcpyDeviceToHost);

    // Add fused ORDERS kernel I/O stats (Q3)
    {
        // INT32 fields: O_ORDERDATE, O_SHIPPRIORITY — count active pages
        const size_t o_i32_fi[2] = {FI_O_ORDERDATE, FI_O_SHIPPRIORITY};
        for (uint32_t pg = 0; pg < npages_o_i32; pg++) {
            if (zm_ord_valid && !zm_ord_ctx.h_mask[pg]) continue;
            for (int k = 0; k < 2; k++) {
                total_io_count++;
                size_t fi = o_i32_fi[k];
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
        // INT64 fields: O_ORDERKEY, O_CUSTKEY — GPU-derived mask
        const size_t o_i64_fi[2] = {FI_O_ORDERKEY, FI_O_CUSTKEY};
        for (uint32_t pg = 0; pg < npages_o_i64; pg++) {
            if (zm_ord_valid && !h_ord_i64[pg]) continue;
            for (int k = 0; k < 2; k++) {
                total_io_count++;
                size_t fi = o_i64_fi[k];
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
    }

    // Add LINEITEM I/O stats
    {
        // Both Q3 and Q3SEL now read 3 INT32 fields (L_SHIPDATE, L_EXTPRICE, L_DISCOUNT)
        const size_t li_i32_fi[3] = {FI_L_SHIPDATE, FI_L_EXTPRICE, FI_L_DISCOUNT};
        for (uint32_t pg = 0; pg < npages_l_i32; pg++) {
            if (zm_li_valid && !zm_li_ctx.h_mask[pg]) continue;
            for (int k = 0; k < 3; k++) {
                total_io_count++;
                size_t fi = li_i32_fi[k];
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    if (nblk > 8 && nblk <= 16) nblk = 24;
                    total_io_bytes += (uint64_t)nblk * 512;
                } else {
                    total_io_bytes += page_size;
                }
            }
        }
        // INT64 field: L_ORDERKEY — GPU-derived mask
        for (uint32_t pg = 0; pg < npages_l_i64; pg++) {
            if (zm_li_valid && !h_li_i64[pg]) continue;
            total_io_count++;
            size_t fi = FI_L_ORDERKEY;
            if (field_comp_methods[fi] != CompressionMethod::NONE) {
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                uint32_t nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                if (nblk > 8 && nblk <= 16) nblk = 24;
                total_io_bytes += (uint64_t)nblk * 512;
            } else {
                total_io_bytes += page_size;
            }
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    cudaFreeHost(h_results);
    mb_cuda_free(staging_data);
    mb_cuda_free(staging_io);
    cudaFree(d_batch_ps);
    cudaFree(d_page_mask);
    cudaFree(d_custkey_set);
    if (d_c_custkey_flat) cudaFree(d_c_custkey_flat);
    if (d_decomp_scs) cudaFree(d_decomp_scs);
    if (d_decomp_spi) cudaFree(d_decomp_spi);
    bam_bulk_read_ctx_destroy(io_ctx_p1);
    cudaFree(d_orders_ht_keys);
    cudaFree(d_orders_ht_payloads);
    cudaFree(d_aggr_keys);
    cudaFree(d_aggr_revenues);
    // d_results, d_sort_temp, d_result_count are aliases into q3li_ws_decomp — do not free
    // Shared page_cache + decomp_buf
    if (q3li_ws_pc) bam_io_page_cache_destroy(q3li_ws_pc);
    if (q3li_ws_decomp) cudaFree(q3li_ws_decomp);
    cudaFree(d_ps_li_i32);
    cudaFree(d_ps_li_i64);
    for (int k = 0; k < 3; k++) {
        if (d_li_comp_offsets_i32[k]) cudaFree(d_li_comp_offsets_i32[k]);
        if (d_li_comp_sizes_i32[k]) cudaFree(d_li_comp_sizes_i32[k]);
    }
    if (d_li_comp_offsets_i64) cudaFree(d_li_comp_offsets_i64);
    if (d_li_comp_sizes_i64) cudaFree(d_li_comp_sizes_i64);
    // Q3 CUSTOMER Phase 1 cleanup
    bam_bulk_read_ctx_destroy(q3_cust_nocopy);
    if (q3_p1_staging) cudaFree(q3_p1_staging);
    if (d_ck_comp_offsets) cudaFree(d_ck_comp_offsets);
    if (d_ck_comp_sizes) cudaFree(d_ck_comp_sizes);
    if (d_mk_comp_offsets) cudaFree(d_mk_comp_offsets);
    if (d_mk_comp_sizes) cudaFree(d_mk_comp_sizes);
    cudaFree(d_ps_custkey);
    cudaFree(d_ps_mktseg);
    // Fused Q3 ORDERS cleanup
    if (fused_q3ord_ctx) bam_fused_q3ord_destroy(fused_q3ord_ctx);
    // d_ord_active_ids / d_li_active_ids are aliases to zm_ctx.d_active_ids or d_all_*_pages — do not free
    // Q3SEL GPU-side IO pruning buffers
    if (d_q3sel_i64_mask) cudaFree(d_q3sel_i64_mask);
    if (d_q3sel_needed_i64) cudaFree(d_q3sel_needed_i64);
    if (d_q3sel_n_needed_i64) cudaFree(d_q3sel_n_needed_i64);
    if (d_q3sel_i64_remap) cudaFree(d_q3sel_i64_remap);
    if (d_q3sel_active_ps) cudaFree(d_q3sel_active_ps);
    if (d_q3sel_nrecs) cudaFree(d_q3sel_nrecs);
    if (d_q3sel_tiles) cudaFree(d_q3sel_tiles);
    if (d_q3sel_n_tiles) cudaFree(d_q3sel_n_tiles);
    if (d_all_ord_pages) cudaFree(d_all_ord_pages);
    if (d_all_li_pages) cudaFree(d_all_li_pages);
    if (h_q3sel_n_needed_i64) cudaFreeHost(h_q3sel_n_needed_i64);
    if (h_q3sel_nrecs) cudaFreeHost(h_q3sel_nrecs);
    if (h_q3sel_n_tiles) cudaFreeHost(h_q3sel_n_tiles);
    if (h_q3sel_tiles) cudaFreeHost(h_q3sel_tiles);
    cudaFree(d_ps_o_i32);
    cudaFree(d_ps_o_i64);
    for (int k = 0; k < 2; k++) {
        if (d_o_comp_offsets_i32[k]) cudaFree(d_o_comp_offsets_i32[k]);
        if (d_o_comp_sizes_i32[k]) cudaFree(d_o_comp_sizes_i32[k]);
        if (d_o_comp_offsets_i64[k]) cudaFree(d_o_comp_offsets_i64[k]);
        if (d_o_comp_sizes_i64[k]) cudaFree(d_o_comp_sizes_i64[k]);
    }
    if (zm_ord_valid) bam_zonemap_ctx_destroy(zm_ord_ctx);
    if (zm_li_valid) bam_zonemap_ctx_destroy(zm_li_ctx);
    if (zm_pc) bam_io_page_cache_destroy(zm_pc);
    if (d_mask_ord_i64) cudaFree(d_mask_ord_i64);
    if (d_mask_li_i64) cudaFree(d_mask_li_i64);
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_ctrl_close(ctrl);

    // Collect compression method string
    std::string comp_str;
    {
        std::set<std::string> methods;
        for (size_t fi = 0; fi < NUM_Q3_FIELDS; fi++)
            methods.insert(compression_method_name(field_comp_methods[fi]));
        for (const auto &m : methods) {
            if (!comp_str.empty()) comp_str += "+";
            comp_str += m;
        }
    }

    size_t total_pages = 0;
    for (size_t fi = 0; fi < NUM_Q3_FIELDS; fi++)
        total_pages += field_npages_arr[fi];

    return BenchmarkResult{
        .nios = total_io_count,
        .read_bytes = total_io_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = comp_str,
        .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = total_pages,
        .kernel_launches = s_kernel_launches,
    };
}

BenchmarkResult tpch_q3sel(BenchmarkOptions &options) {
    return tpch_q3(options);
}

// ============================================================
// Q1: Pricing Summary Report (LINEITEM only, 7 fields)
// I/O: BaM GPU-initiated, Decomp: device-side LZ4 (bam_lz4_batch_decompress)
// Scan: flatten all 7 fields → q1_scan_aggregate (flat arrays)
// ============================================================

BenchmarkResult tpch_q1(BenchmarkOptions &options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    // ── 2. Open BaM controller(s) ──
    size_t gpu_free_pre_ctrl = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_dummy);

    const uint32_t bam_num_queues = 128;
    auto ds = DataPathFusion::bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_dummy);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;
    const uint32_t n_devices = ds.n_devices;

    // Helper: read a striped page to host
    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata page via BaM ──
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    if (rc != 0) {
        std::cerr << "bam_read_page failed (metadata header)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (full metadata)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    std::cout << "=== TPCH Q1 (GIDP+BAM+DECOMP) ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;

    const uint64_t nrecs_lineitem = metadata.table_lineitem_nrows;
    std::cout << "nrecs_lineitem: " << nrecs_lineitem << std::endl;

    // ── 4. Extract Q1 field info (7 LINEITEM columns) ──
    constexpr size_t FI_L_QUANTITY    = 0;
    constexpr size_t FI_L_EXTPRICE    = 1;
    constexpr size_t FI_L_DISCOUNT    = 2;
    constexpr size_t FI_L_TAX         = 3;
    constexpr size_t FI_L_RETURNFLAG  = 4;
    constexpr size_t FI_L_LINESTATUS  = 5;
    constexpr size_t FI_L_SHIPDATE    = 6;
    constexpr size_t NUM_Q1_FIELDS    = 7;

    auto q1_cols = TPCH::Query::Q1::SCAN_TARGET_COLS;
    const size_t blocks_per_page = page_size / 512;

    uint64_t field_start_page_ids[NUM_Q1_FIELDS];
    uint64_t field_npages_arr[NUM_Q1_FIELDS];
    CompressionMethod field_comp_methods[NUM_Q1_FIELDS];

    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        size_t col = q1_cols[fi];
        field_start_page_ids[fi] = metadata.table_lineitem_start_page_ids[col];
        field_npages_arr[fi]     = metadata.table_lineitem_npages[col];
        field_comp_methods[fi]   = static_cast<CompressionMethod>(
            metadata.table_lineitem_compression_method[col]);
    }

    const char *field_names[NUM_Q1_FIELDS] = {
        "L_QUANTITY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_TAX",
        "L_RETURNFLAG", "L_LINESTATUS", "L_SHIPDATE" };
    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        std::cout << "  " << field_names[fi]
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << static_cast<int>(field_comp_methods[fi])
                  << std::endl;
    }

    // ── 5. Read compression metadata ──
    std::vector<uint32_t> h_comp_sizes[NUM_Q1_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_Q1_FIELDS];

    bool any_compressed = false;

    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        any_compressed = true;

        size_t col = q1_cols[fi];
        uint64_t cs_start      = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
        uint64_t cs_npages_cnt = metadata.table_lineitem_compressed_page_sizes_npages[col];
        uint64_t nbase         = metadata.table_lineitem_compression_nbases[col];
        uint64_t base_start    = metadata.table_lineitem_compression_base_start_page_ids[col];

        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_sizes " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        h_comp_sizes[fi].assign(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages_arr[fi]);

        size_t bp_npages = TPCH::nbase_to_npages(nbase, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_bases " << field_names[fi] << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            h_comp_sizes[fi].data(), nbase, field_npages_arr[fi], page_size,
            field_start_page_ids[fi], n_devices, offsets_vec);
        h_comp_offsets[fi] = std::move(offsets_vec);
    }

    // ── 6. Verify all fields have same npages (all INT32 LINEITEM) ──
    const uint64_t npages = field_npages_arr[0];
    for (size_t fi = 1; fi < NUM_Q1_FIELDS; fi++) {
        if (field_npages_arr[fi] != npages) {
            std::cerr << "Error: Q1 field " << fi << " has different npages ("
                      << field_npages_arr[fi] << " vs " << npages << ")" << std::endl;
            bam_ctrl_close(ctrl);
            exit(EXIT_FAILURE);
        }
    }

    // ── 6a. Zone map pruning metadata (Rule 3: metadata outside timing) ──
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    uint64_t zm_sd_nstats = 0, zm_sd_stats_start = 0, zm_sd_stats_npg = 0;
    if (options.enable_zonemap) {
        size_t sd_col = q1_cols[FI_L_SHIPDATE];
        zm_sd_nstats      = metadata.table_lineitem_nstats[sd_col];
        zm_sd_stats_start = metadata.table_lineitem_stats_start_page_ids[sd_col];
        zm_sd_stats_npg   = metadata.table_lineitem_stats_npages[sd_col];
    }

    // ── 7. Fused kernel setup (all allocation before timed section) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    const uint32_t fused_num_blocks = std::min(
        static_cast<uint32_t>(npages),
        static_cast<uint32_t>(sm_count * 5));

    // Create fused Q1 context (page_cache + decomp_buf)
    auto fused_ctx = bam_fused_q1_create(ctrl, static_cast<uint32_t>(page_size), fused_num_blocks);

    // Upload per-field compression metadata to GPU
    uint64_t* d_comp_offsets[NUM_Q1_FIELDS] = {};
    uint32_t* d_comp_sizes_gpu[NUM_Q1_FIELDS] = {};

    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;

        CUDA_CHECK(cudaMalloc(&d_comp_offsets[fi], npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_comp_offsets[fi], h_comp_offsets[fi].data(),
                    npages * sizeof(uint64_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_comp_sizes_gpu[fi], npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_comp_sizes_gpu[fi], h_comp_sizes[fi].data(),
                    npages * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // BaM zonemap ctx (Rule 4: alloc outside timing; separate small page_cache)
    bam_io_page_cache_t zm_pc = nullptr;
    if (options.enable_zonemap && zm_sd_nstats > 0 && zm_sd_stats_start > 0 && zm_sd_stats_npg > 0) {
        zm_pc = bam_io_page_cache_create(ctrl, static_cast<uint32_t>(page_size),
            kBamZonemapMaxReads);
        void* zm_d_ctrls = bam_io_page_cache_get_d_ctrls(zm_pc);
        void* zm_d_pc    = bam_io_page_cache_get_d_pc_ptr(zm_pc);
        const char* zm_base = (const char*)bam_io_page_cache_get_base_addr(zm_pc);
        zm_ctx = bam_zonemap_ctx_create(zm_d_ctrls, zm_d_pc, (void*)zm_base,
            static_cast<uint32_t>(page_size), npages);
        for (uint64_t j = 0; j < zm_sd_stats_npg; j++) {
            uint64_t pg_id = zm_sd_stats_start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            zm_ctx.h_reads[zm_nreads++] = {
                ds.partition_start_lbas[dev] + local * blocks_per_page,
                static_cast<uint32_t>(blocks_per_page), dev};
        }
        zm_ctx.h_preds[zm_npreds++] = {0, zm_sd_nstats, INT32_MIN, (int32_t)19980902};
        zm_valid = true;
    }

    // Aggregate array
    constexpr size_t agg_size = Q1_NUM_GROUPS * Q1_NUM_AGGS * sizeof(int64_t);
    int64_t *d_agg = nullptr;
    CUDA_CHECK(cudaMalloc(&d_agg, agg_size));
    CUDA_CHECK(cudaMemset(d_agg, 0, agg_size));

    // Per-phase cycle accumulator (IO, decomp, scan, iters, total)
    unsigned long long* d_q1_cycles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q1_cycles, 5 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_q1_cycles, 0, 5 * sizeof(unsigned long long)));

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    std::cout << "[GIDP+BAM+DECOMP Q1] Fused kernel: " << fused_num_blocks
              << " persistent blocks, " << npages << " pages"
              << " (zm=" << (zm_valid ? "on" : "off") << ")" << std::endl;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // Zone map eval (Rule 6: IO + eval inside timing, mask stays on GPU)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
        uint32_t active_count = 0;
        for (uint64_t pg = 0; pg < npages; pg++)
            if (zm_ctx.h_mask[pg]) active_count++;
        std::cout << "[ZONEMAP] L_SHIPDATE pruning: active=" << active_count
                  << "/" << npages << std::endl;
    }

    // Build params struct (no cudaMalloc inside timed section)
    BAMFusedQ1Params fused_params{};
    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        fused_params.field_start_page_ids[fi] = field_start_page_ids[fi];
        fused_params.d_comp_offsets[fi] = d_comp_offsets[fi];
        fused_params.d_comp_sizes[fi] = d_comp_sizes_gpu[fi];
        fused_params.is_compressed[fi] = (field_comp_methods[fi] != CompressionMethod::NONE);
    }
    for (uint32_t d = 0; d < n_devices && d < 4; d++)
        fused_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
    fused_params.n_devices = n_devices;
    fused_params.page_size = static_cast<uint32_t>(page_size);
    fused_params.npages = npages;
    fused_params.num_blocks = fused_num_blocks;
    fused_params.d_agg = d_agg;
    fused_params.d_page_active = zm_valid ? zm_ctx.d_mask : nullptr;
    fused_params.d_active_page_ids = zm_valid ? zm_ctx.d_active_ids : nullptr;
    fused_params.num_active_pages = zm_valid ? *zm_ctx.h_num_active : 0;
    fused_params.d_cycles = d_q1_cycles;

    // Single fused kernel launch
    bam_fused_q1_run_async(fused_ctx, fused_params, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── 12. Results ──
    int64_t h_agg[Q1_NUM_GROUPS * Q1_NUM_AGGS];
    CUDA_CHECK(cudaMemcpy(h_agg, d_agg, agg_size, cudaMemcpyDeviceToHost));

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

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "========================================" << std::endl;

    // Cycle breakdown (outside timing)
    {
        unsigned long long h_cyc[5] = {0};
        CUDA_CHECK(cudaMemcpy(h_cyc, d_q1_cycles, sizeof(h_cyc), cudaMemcpyDeviceToHost));
        double blk = (double)fused_num_blocks;
        double iters = (double)h_cyc[3];
        double avg_io    = blk > 0 ? (double)h_cyc[0] / blk : 0.0;
        double avg_dec   = blk > 0 ? (double)h_cyc[1] / blk : 0.0;
        double avg_scan  = blk > 0 ? (double)h_cyc[2] / blk : 0.0;
        double avg_tot   = blk > 0 ? (double)h_cyc[4] / blk : 0.0;
        double per_iter_io   = iters > 0 ? avg_io   / (iters / blk) : 0.0;
        double per_iter_dec  = iters > 0 ? avg_dec  / (iters / blk) : 0.0;
        double per_iter_scan = iters > 0 ? avg_scan / (iters / blk) : 0.0;
        std::cout << "[Q1-CYC] per-block (Mcycles): IO=" << (avg_io / 1e6)
                  << " DEC=" << (avg_dec / 1e6)
                  << " SCAN=" << (avg_scan / 1e6)
                  << " TOTAL=" << (avg_tot / 1e6)
                  << " iters/blk=" << (iters / blk) << std::endl;
        std::cout << "[Q1-CYC] per-iter (cycles):  IO=" << per_iter_io
                  << " DEC=" << per_iter_dec
                  << " SCAN=" << per_iter_scan << std::endl;
    }

    // ── 14. Cleanup ──
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    if (zm_pc) bam_io_page_cache_destroy(zm_pc);
    bam_fused_q1_destroy(fused_ctx);
    if (d_q1_cycles) cudaFree(d_q1_cycles);
    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        if (d_comp_offsets[fi]) cudaFree(d_comp_offsets[fi]);
        if (d_comp_sizes_gpu[fi]) cudaFree(d_comp_sizes_gpu[fi]);
    }
    CUDA_CHECK(cudaFree(d_agg));
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_ctrl_close(ctrl);

    std::string comp_str;
    {
        std::set<std::string> methods;
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++)
            methods.insert(compression_method_name(field_comp_methods[fi]));
        for (const auto &m : methods) {
            if (!comp_str.empty()) comp_str += "+";
            comp_str += m;
        }
    }

    // Compute fused kernel I/O stats (active pages only when zone map enabled)
    uint64_t fused_io_count = 0, fused_io_bytes = 0;
    {
        uint64_t hist[5] = {0};        // [≤8, 9-16 raw (before bump), 17-32, 33-64, >64]
        uint64_t bumped_9_16 = 0;
        uint64_t bumped_bytes_delta = 0;
        uint64_t per_field_hist[NUM_Q1_FIELDS][5] = {{0}};
        uint64_t per_field_bump[NUM_Q1_FIELDS] = {0};
        uint64_t per_field_bytes[NUM_Q1_FIELDS] = {0};
        double per_field_sz_sum[NUM_Q1_FIELDS] = {0};
        uint64_t per_field_sz_min[NUM_Q1_FIELDS];
        uint64_t per_field_sz_max[NUM_Q1_FIELDS] = {0};
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) per_field_sz_min[fi] = UINT64_MAX;
        const uint8_t* h_mask = zm_valid ? zm_ctx.h_mask : nullptr;
        for (size_t pg = 0; pg < npages; pg++) {
            if (h_mask && !h_mask[pg]) continue;
            for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
                fused_io_count++;
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    uint32_t raw_nblk = ((comp_sz + 4095u) & ~4095u) / 512;
                    uint32_t nblk = raw_nblk;
                    if (nblk > 8 && nblk <= 16) { nblk = 24; bumped_9_16++; bumped_bytes_delta += (uint64_t)(24 - raw_nblk) * 512; per_field_bump[fi]++; }
                    fused_io_bytes += (uint64_t)nblk * 512;
                    per_field_bytes[fi] += (uint64_t)nblk * 512;
                    per_field_sz_sum[fi] += comp_sz;
                    if (comp_sz < per_field_sz_min[fi]) per_field_sz_min[fi] = comp_sz;
                    if (comp_sz > per_field_sz_max[fi]) per_field_sz_max[fi] = comp_sz;
                    int b;
                    if (raw_nblk <= 8) b = 0;
                    else if (raw_nblk <= 16) b = 1;
                    else if (raw_nblk <= 32) b = 2;
                    else if (raw_nblk <= 64) b = 3;
                    else b = 4;
                    hist[b]++;
                    per_field_hist[fi][b]++;
                } else {
                    fused_io_bytes += page_size;
                    per_field_bytes[fi] += page_size;
                }
            }
        }
        std::cout << "[Q1-IO-HIST] raw_nblk buckets: "
                  << "≤8=" << hist[0]
                  << " 9-16(danger)=" << hist[1]
                  << " 17-32=" << hist[2]
                  << " 33-64=" << hist[3]
                  << " >64=" << hist[4]
                  << " | bumped_9_16=" << bumped_9_16
                  << " extra_bytes=" << bumped_bytes_delta
                  << std::endl;
        const char* q1_field_names[NUM_Q1_FIELDS] = {
            "L_QUANTITY","L_EXTENDEDPRICE","L_DISCOUNT","L_TAX",
            "L_RETURNFLAG","L_LINESTATUS","L_SHIPDATE"
        };
        std::cout << "[Q1-FIELD] field comp_method"
                  << " ≤8 9-16 17-32 33-64 >64 | bump bytes avg min max" << std::endl;
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
            uint64_t cnt = per_field_hist[fi][0] + per_field_hist[fi][1]
                         + per_field_hist[fi][2] + per_field_hist[fi][3]
                         + per_field_hist[fi][4];
            double avg_sz = cnt > 0 ? per_field_sz_sum[fi] / (double)cnt : 0.0;
            std::cout << "[Q1-FIELD] " << q1_field_names[fi]
                      << " cm=" << (int)field_comp_methods[fi]
                      << " " << per_field_hist[fi][0]
                      << " " << per_field_hist[fi][1]
                      << " " << per_field_hist[fi][2]
                      << " " << per_field_hist[fi][3]
                      << " " << per_field_hist[fi][4]
                      << " | bump=" << per_field_bump[fi]
                      << " bytes=" << per_field_bytes[fi]
                      << " avg_sz=" << (uint64_t)avg_sz
                      << " min=" << (per_field_sz_min[fi] == UINT64_MAX ? 0 : per_field_sz_min[fi])
                      << " max=" << per_field_sz_max[fi]
                      << std::endl;
        }
    }

    return BenchmarkResult{
        .nios = fused_io_count,
        .read_bytes = fused_io_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = comp_str,
        .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NUM_Q1_FIELDS,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// Revenue — GIDP+BAM+DECOMP execution mode
// Same 4 fields as Q6 with fused BaM I/O + LZ4 decompress + Revenue scan.
// Single persistent kernel (same architecture as fused Q6).
// ============================================================
BenchmarkResult tpch_revenue(BenchmarkOptions &options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    size_t gpu_free_start = 0, gpu_total_dummy = 0;
    cudaMemGetInfo(&gpu_free_start, &gpu_total_dummy);

    // ── 2. Open BaM controller(s) ──
    size_t gpu_free_pre_ctrl = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_dummy);

    const uint32_t bam_num_queues = 128;
    auto ds = DataPathFusion::bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_dummy);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;
    const uint32_t n_devices = ds.n_devices;

    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata page via BaM ──
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    if (rc != 0) {
        std::cerr << "bam_read_page failed (metadata header)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (full metadata)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    std::cout << "=== TPCH Table Metadata ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;
    std::cout << "nrecs_lineitem: " << metadata.table_lineitem_nrows << std::endl;

    // ── 4. Extract field info (same 4 fields as Q6) ──
    constexpr size_t NUM_FIELDS = TPCH::Query::Q6::NUM_SCAN_TARGET_COLS;
    auto rev_cols = TPCH::Query::Q6::SCAN_TARGET_COLS;
    const size_t blocks_per_page = page_size / 512;

    uint64_t field_start_page_ids[NUM_FIELDS];
    uint64_t field_npages_arr[NUM_FIELDS];
    CompressionMethod field_comp_methods[NUM_FIELDS];

    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        size_t col = rev_cols[fi];
        field_start_page_ids[fi] = metadata.table_lineitem_start_page_ids[col];
        field_npages_arr[fi] = metadata.table_lineitem_npages[col];
        field_comp_methods[fi] = static_cast<CompressionMethod>(
            metadata.table_lineitem_compression_method[col]);
        std::cout << "  Field " << col
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << static_cast<int>(field_comp_methods[fi])
                  << std::endl;
    }

    const uint64_t npages = field_npages_arr[0];
    for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
        if (field_npages_arr[fi] != npages) {
            std::cerr << "Error: field " << fi << " has different npages" << std::endl;
            bam_ctrl_close(ctrl);
            exit(EXIT_FAILURE);
        }
    }
    if (npages == 0) {
        std::cout << "No pages to read." << std::endl;
        bam_ctrl_close(ctrl);
        return BenchmarkResult{};
    }

    // ── 5. Read compression metadata ──
    std::vector<uint32_t> h_comp_sizes[NUM_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_FIELDS];

    bool any_compressed = false;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        any_compressed = true;

        size_t col = rev_cols[fi];
        uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
        uint64_t cs_npages_cnt = metadata.table_lineitem_compressed_page_sizes_npages[col];
        uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
        uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];

        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_sizes)" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        h_comp_sizes[fi].assign(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + npages);

        size_t bp_npages = TPCH::nbase_to_npages(nbase, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_bases)" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            h_comp_sizes[fi].data(), nbase, npages, page_size,
            field_start_page_ids[fi], n_devices, offsets_vec);
        h_comp_offsets[fi] = std::move(offsets_vec);
    }

    // ── 6. Zone map pruning metadata (Rule 3: metadata outside timing) ──
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    uint64_t zm_sd_nstats = 0, zm_sd_stats_start = 0, zm_sd_stats_npg = 0;
    if (options.enable_zonemap) {
        size_t sd_col = rev_cols[0];
        zm_sd_nstats      = metadata.table_lineitem_nstats[sd_col];
        zm_sd_stats_start = metadata.table_lineitem_stats_start_page_ids[sd_col];
        zm_sd_stats_npg   = metadata.table_lineitem_stats_npages[sd_col];
    }

    // ── 7. Warp-Specialized kernel setup ──
    // 32 warps (1024 threads) per block:
    //   Warps 0-3: IO (BaM reads), Warps 4-31: Decomp (7 groups × 4 warps)
    //   All 1024 threads: Revenue scan
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };
    auto safe_nblocks = [](uint32_t nblk) -> uint32_t {
        if (nblk > 8 && nblk <= 16) nblk = 17;
        return nblk;
    };

    // Upload per-field compression metadata to GPU
    uint32_t* d_comp_sizes_gpu[NUM_FIELDS] = {};
    uint64_t* d_comp_offsets_gpu[NUM_FIELDS] = {};
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
        CUDA_CHECK(cudaMalloc(&d_comp_sizes_gpu[fi], npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_comp_sizes_gpu[fi], h_comp_sizes[fi].data(),
                    npages * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_comp_offsets_gpu[fi], npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_comp_offsets_gpu[fi], h_comp_offsets[fi].data(),
                    npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    // Grid size: capped by occupancy (full page count; kernel skips pruned pages)
    const uint32_t max_blocks = revenue_warp_spec_max_blocks(static_cast<uint32_t>(page_size));
    const uint32_t num_blocks = std::min(static_cast<uint32_t>(npages), max_blocks);

    // Page cache: 56 slots per block (2 bufs × 7 batch × 4 fields)
    auto io_pc = bam_io_page_cache_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks * 56);
    void* d_ctrls_pc     = bam_io_page_cache_get_d_ctrls(io_pc);
    void* d_pc_ptr       = bam_io_page_cache_get_d_pc_ptr(io_pc);
    const char* pc_base  = (const char*)bam_io_page_cache_get_base_addr(io_pc);

    // BaM zonemap ctx (Rule 4: alloc outside timing; borrows page_cache)
    if (options.enable_zonemap && zm_sd_nstats > 0 && zm_sd_stats_start > 0 && zm_sd_stats_npg > 0) {
        zm_ctx = bam_zonemap_ctx_create(d_ctrls_pc, d_pc_ptr, (void*)pc_base,
            static_cast<uint32_t>(page_size), npages);
        for (uint64_t j = 0; j < zm_sd_stats_npg; j++) {
            uint64_t pg_id = zm_sd_stats_start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            zm_ctx.h_reads[zm_nreads++] = {
                ds.partition_start_lbas[dev] + local * blocks_per_page,
                static_cast<uint32_t>(blocks_per_page), dev};
        }
        zm_ctx.h_preds[zm_npreds++] = {0, zm_sd_nstats,
            options.q6_sd_low, options.q6_sd_high - 1};
        zm_valid = true;
    }

    // Decomp buffer: 56 pages per block (2 bufs × 7 batch × 4 fields)
    char* d_decomp_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_decomp_buf,
        (size_t)num_blocks * 56 * page_size));

    // Revenue output
    int64_t *d_revenue = nullptr;
    CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // Fill params
    RevenueWarpSpecParams wsp{};
    wsp.total_pages = static_cast<uint32_t>(npages);
    wsp.d_page_mask = nullptr;
    wsp.page_size = static_cast<uint32_t>(page_size);
    wsp.n_devices = n_devices;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        wsp.field_start_page_ids[fi] = field_start_page_ids[fi];
        wsp.d_comp_offsets[fi] = d_comp_offsets_gpu[fi];
        wsp.d_comp_sizes[fi] = d_comp_sizes_gpu[fi];
        wsp.is_compressed[fi] = (field_comp_methods[fi] != CompressionMethod::NONE);
    }
    for (uint32_t d = 0; d < n_devices && d < 4; d++)
        wsp.partition_start_lbas[d] = ds.partition_start_lbas[d];
    wsp.sd_low = options.q6_sd_low;
    wsp.sd_high = options.q6_sd_high;
    wsp.disc_lo = options.disable_other_filters ? 0 : 5;
    wsp.disc_hi = options.disable_other_filters ? INT32_MAX : 7;
    wsp.qt_max = options.disable_other_filters ? options.revenue_qt_max : 24;
    wsp.d_revenue = d_revenue;

    std::cout << "[GIDP+BAM+WARPSPEC Revenue] num_blocks=" << num_blocks
              << " (npages=" << npages << " zm=" << (zm_valid ? "on" : "off") << ")"
              << std::endl;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    // ════════════════════════════════════════════
    // total_start — warp-specialized kernel
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // Zone map eval (Rule 6: IO + eval inside timing, mask stays on GPU)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
        const uint32_t num_active = *zm_ctx.h_num_active;
        std::cout << "[ZONEMAP] L_SHIPDATE pruning: active=" << num_active
                  << "/" << npages << std::endl;
        wsp.d_active_page_ids = zm_ctx.d_active_ids;
        wsp.total_pages = num_active;
    }

    const uint32_t rev_launch_blocks = zm_valid
        ? std::min(*zm_ctx.h_num_active, num_blocks) : num_blocks;
    revenue_warp_spec_launch(d_ctrls_pc, d_pc_ptr, pc_base,
                             d_decomp_buf, wsp, rev_launch_blocks, stream);
    s_kernel_launches++;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Result ──
    int64_t h_revenue = 0;
    CUDA_CHECK(cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::cout << "Revenue total: " << h_revenue << std::endl;

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

    // Compute IO stats (using h_mask for active pages only)
    uint64_t total_io_count = 0;
    uint64_t total_io_bytes = 0;
    {
        const uint8_t* h_mask = zm_valid ? zm_ctx.h_mask : nullptr;
        for (size_t pg = 0; pg < npages; pg++) {
            if (h_mask && !h_mask[pg]) continue;
            for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
                uint32_t nblk;
                if (field_comp_methods[fi] != CompressionMethod::NONE) {
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    nblk = safe_nblocks(
                        static_cast<uint32_t>((roundup4096(comp_sz) + 511) / 512));
                } else {
                    nblk = static_cast<uint32_t>(page_size / 512);
                }
                total_io_bytes += (uint64_t)nblk * 512;
                total_io_count++;
            }
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    bam_io_page_cache_destroy(io_pc);
    cudaFree(d_decomp_buf);
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (d_comp_sizes_gpu[fi]) cudaFree(d_comp_sizes_gpu[fi]);
        if (d_comp_offsets_gpu[fi]) cudaFree(d_comp_offsets_gpu[fi]);
    }
    cudaFree(d_revenue);
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_ctrl_close(ctrl);

    std::string comp_str;
    {
        std::set<std::string> methods;
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            methods.insert(compression_method_name(field_comp_methods[fi]));
        for (const auto &m : methods) {
            if (!comp_str.empty()) comp_str += "+";
            comp_str += m;
        }
    }

    return BenchmarkResult{
        .nios = total_io_count,
        .read_bytes = total_io_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = comp_str,
        .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NUM_FIELDS,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// I/O contention microbenchmark
// Reads L_EXTENDEDPRICE pages (uncompressed, fixed 1 MiB) with varying
// (num_warps, ios_per_warp) to measure QP contention effects.
// ============================================================
BenchmarkResult io_contention_bench(BenchmarkOptions &options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);

    // ── 2. Open BaM controller(s) ──
    const uint32_t bam_num_queues = 128;
    auto ds = DataPathFusion::bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;
    const uint32_t n_devices = ds.n_devices;

    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata ──
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    if (rc != 0) { std::cerr << "bam_read_page failed" << std::endl; exit(1); }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) { std::cerr << "metadata read failed" << std::endl; exit(1); }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());

    // Use L_EXTENDEDPRICE (uncompressed field → fixed page size, clean I/O benchmark)
    const size_t col = TPCH::common::L_EXTENDEDPRICE;
    uint64_t bench_start_page_id = metadata.table_lineitem_start_page_ids[col];
    uint32_t bench_npages = metadata.table_lineitem_npages[col];
    fprintf(stdout, "Page Size: %zu\n", page_size);
    fprintf(stdout, "L_EXTENDEDPRICE: start_page=%lu npages=%u compression=%d\n",
            bench_start_page_id, bench_npages,
            metadata.table_lineitem_compression_method[col]);

    // ── 4. Run benchmark configurations ──
    // Fixed total outstanding ≈ 1024, varying (warps, ios/warp)
    struct { uint32_t warps; uint32_t ios; } configs[] = {
        {1024, 1},
        {512,  2},
        {256,  4},
        {128,  8},
        {64,  16},
        {32,  32},
        // Also test with current fused kernel's warp count equivalents
        {3024, 1},   // current: 216 blks × 2 groups × 7 IO warps
        {1512, 1},   // half
        {432,  1},   // proposed: 216 blks × 2 groups × 1 IO warp
        {432,  4},
        {432,  8},
        {216,  4},
        {216,  8},
    };

    fprintf(stdout, "\n=== BaM I/O Contention Microbenchmark ===\n");
    fprintf(stdout, "Pages: %u × %lu B = %lu MiB\n",
            bench_npages, page_size, (uint64_t)bench_npages * page_size / (1024*1024));
    fprintf(stdout, "Devices: %u, QPs/device: 128, Total QPs: %u\n",
            n_devices, n_devices * 128);
    fprintf(stdout, "\n%-8s %-8s %-12s %-10s %-12s %-10s\n",
            "warps", "ios/w", "outstanding", "w/QP", "time(ms)", "IO GB/s");
    fprintf(stdout, "%-8s %-8s %-12s %-10s %-12s %-10s\n",
            "------", "------", "----------", "--------", "--------", "--------");

    for (auto& c : configs) {
        BamIoBenchConfig cfg;
        cfg.num_warps = c.warps;
        cfg.ios_per_warp = c.ios;

        auto r = bam_io_contention_bench_run(
            ctrl, page_size, bench_npages,
            bench_start_page_id, n_devices,
            ds.partition_start_lbas, cfg);

        fprintf(stdout, "%-8u %-8u %-12u %-10.2f %-12.1f %-10.2f\n",
                r.num_warps, r.ios_per_warp, r.total_outstanding,
                r.warps_per_qp, r.elapsed_ms, r.io_throughput_gbs);
    }
    fprintf(stdout, "\n");

    // Return dummy result
    bam_ctrl_close(ctrl);
    return BenchmarkResult{};
}

// ============================================================
// LZ4 decomp-only microbenchmark
// Pre-loads Q1 field pages via BaM sync I/O, then measures
// decomp-only throughput per field.
// ============================================================
BenchmarkResult lz4_decomp_bench(BenchmarkOptions &options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);

    // ── 2. Open BaM controller(s) ──
    const uint32_t bam_num_queues = 128;
    auto ds = DataPathFusion::bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;
    const uint32_t n_devices = ds.n_devices;

    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata ──
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    if (rc != 0) { std::cerr << "bam_read_page failed" << std::endl; exit(1); }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) { std::cerr << "metadata read failed" << std::endl; exit(1); }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());

    // ── 4. Extract Q1 field info (7 LINEITEM columns) ──
    constexpr size_t NUM_Q1_FIELDS = 7;
    auto q1_cols = TPCH::Query::Q1::SCAN_TARGET_COLS;

    const char* field_names[NUM_Q1_FIELDS] = {
        "L_QUANTITY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_TAX",
        "L_RETURNFLAG", "L_LINESTATUS", "L_SHIPDATE" };

    uint64_t field_start_page_ids[NUM_Q1_FIELDS];
    uint64_t field_npages_arr[NUM_Q1_FIELDS];
    CompressionMethod field_comp_methods[NUM_Q1_FIELDS];

    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        size_t col = q1_cols[fi];
        field_start_page_ids[fi] = metadata.table_lineitem_start_page_ids[col];
        field_npages_arr[fi]     = metadata.table_lineitem_npages[col];
        field_comp_methods[fi]   = static_cast<CompressionMethod>(
            metadata.table_lineitem_compression_method[col]);
    }

    const uint64_t npages = field_npages_arr[0];

    // ── 5. Read compression metadata for compressed fields ──
    std::vector<uint32_t> h_comp_sizes[NUM_Q1_FIELDS];
    std::vector<size_t> h_comp_offsets_sz[NUM_Q1_FIELDS];

    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;

        size_t col = q1_cols[fi];
        uint64_t cs_start      = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
        uint64_t cs_npages_cnt = metadata.table_lineitem_compressed_page_sizes_npages[col];
        uint64_t nbase         = metadata.table_lineitem_compression_nbases[col];
        uint64_t base_start    = metadata.table_lineitem_compression_base_start_page_ids[col];

        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) { std::cerr << "comp_sizes read failed" << std::endl; exit(1); }
        }
        h_comp_sizes[fi].assign(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + npages);

        size_t bp_npages = TPCH::nbase_to_npages(nbase, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) { std::cerr << "comp_bases read failed" << std::endl; exit(1); }
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            h_comp_sizes[fi].data(), nbase, npages, page_size,
            field_start_page_ids[fi], n_devices, offsets_vec);
        h_comp_offsets_sz[fi] = std::move(offsets_vec);
    }

    // ── 6. Load compressed pages into GPU memory and run pure nvCOMPdx bench ──
    printf("\n=== Pure nvCOMPdx LZ4 Decomp Benchmark (no BaM dependency) ===\n");
    printf("Pages per field: %lu × %zu B = %lu MiB (decompressed)\n",
            npages, page_size, npages * page_size / (1024*1024));
    printf("Devices: %u\n\n", n_devices);

    // Warp counts to test
    uint32_t warp_configs[] = { 48, 108, 216, 432, 864, 1512, 3024 };
    constexpr size_t n_warp_configs = sizeof(warp_configs) / sizeof(warp_configs[0]);

    // Allocate decomp output buffer (shared across fields, sized for max warps)
    uint32_t max_warps = warp_configs[n_warp_configs - 1];
    if (max_warps > npages) max_warps = npages;
    char* d_decomp_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_decomp_buf, (size_t)max_warps * page_size));

    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        fprintf(stdout, "--- Field %zu: %s (%s) ---\n",
                fi, field_names[fi], is_compressed ? "LZ4" : "uncompressed");

        if (!is_compressed) {
            fprintf(stdout, "  (skipped: uncompressed field)\n\n");
            continue;
        }

        // Print average compressed size
        uint64_t total_comp_bytes = 0;
        for (auto s : h_comp_sizes[fi]) total_comp_bytes += s;
        double avg_comp = (double)total_comp_bytes / h_comp_sizes[fi].size();
        double ratio = (double)page_size / avg_comp;
        fprintf(stdout, "  avg comp_sz: %.0f B (ratio: %.2fx)\n", avg_comp, ratio);

        // Read all compressed pages from disk into host buffer (packed)
        std::vector<uint64_t> h_offsets(npages);
        uint64_t packed_offset = 0;
        for (uint64_t p = 0; p < npages; p++) {
            h_offsets[p] = packed_offset;
            packed_offset += h_comp_sizes[fi][p];
        }
        size_t total_comp_buf_size = packed_offset;
        std::vector<char> h_comp_buf(total_comp_buf_size + 4096);  // pad for 4KB-aligned reads

        // Read each compressed page using BaM (host-side) and pack
        for (uint64_t p = 0; p < npages; p++) {
            uint64_t global_pg = field_start_page_ids[fi] + p;
            uint32_t dev = global_pg % n_devices;
            uint64_t local_pg = global_pg / n_devices;
            // Read from device-local offset in compressed layout
            uint64_t comp_off = h_comp_offsets_sz[fi][p];
            uint64_t lba = ds.partition_start_lbas[dev] + comp_off / 512;
            uint32_t read_sz = ((h_comp_sizes[fi][p] + 4095u) & ~4095u);
            rc = bam_read_page(ctrl, read_sz, lba, h_comp_buf.data() + h_offsets[p], dev);
            if (rc != 0) { fprintf(stdout, "read failed fi=%zu p=%lu\n", fi, p); break; }
        }

        // Copy to GPU
        char* d_comp_buf = nullptr;
        uint64_t* d_comp_offsets = nullptr;
        uint32_t* d_comp_sizes_gpu = nullptr;
        CUDA_CHECK(cudaMalloc(&d_comp_buf, total_comp_buf_size));
        CUDA_CHECK(cudaMalloc(&d_comp_offsets, npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_comp_sizes_gpu, npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_comp_buf, h_comp_buf.data(), total_comp_buf_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_comp_offsets, h_offsets.data(), npages * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_comp_sizes_gpu, h_comp_sizes[fi].data(), npages * sizeof(uint32_t), cudaMemcpyHostToDevice));

        fprintf(stdout, "  Loaded %.1f MiB compressed data to GPU\n",
                (double)total_comp_buf_size / (1024*1024));

        fprintf(stdout, "  %-8s %-12s %-12s %-14s\n",
                "warps", "time(ms)", "decomp GB/s", "us/page/warp");
        fprintf(stdout, "  %-8s %-12s %-12s %-14s\n",
                "------", "--------", "----------", "------------");

        for (size_t wi = 0; wi < n_warp_configs; wi++) {
            uint32_t nw = warp_configs[wi];
            if (nw > npages) nw = npages;

            auto r = nvcompdx_lz4_bench_run(
                d_comp_buf, d_comp_offsets, d_comp_sizes_gpu,
                d_decomp_buf, npages, page_size, nw);

            fprintf(stdout, "  %-8u %-12.1f %-12.2f %-14.1f\n",
                    r.num_warps, r.elapsed_ms,
                    r.decomp_throughput_gbs, r.us_per_page_per_warp);
        }
        fprintf(stdout, "\n");

        cudaFree(d_comp_buf);
        cudaFree(d_comp_offsets);
        cudaFree(d_comp_sizes_gpu);
    }

    cudaFree(d_decomp_buf);
    bam_ctrl_close(ctrl);
    return BenchmarkResult{};
}

} // namespace GidpBamFusion
