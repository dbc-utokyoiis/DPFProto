#pragma once

// gidp_bam.cu — GIDP+BAM execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMP (host-launched), Kernel: same as GIDP
//
// Included AFTER gidp.cu and datapathfusion.cu in tpch_main.cu, so all
// helpers are accessible via qualified namespace names.

#include "tpch/bam_bulk_read.cuh"
#include "tpch/bam_lz4_fused_q5_dim.cuh"
#include <cub/device/device_merge_sort.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// Aliases for helpers defined in gidp.cu (namespace Gidp) and
// datapathfusion.cu (namespace DataPathFusion).
using Gidp::NvcompDecompCtx;

// NVMe PRP2 danger zone fix: when a transfer spans exactly 2 controller pages
// (9-16 blocks), the NVMe controller interprets PRP2 as a direct data address,
// but BaM's page_cache sets PRP2 as a PRP list pointer. Bump to 17 blocks
// (≥3 pages) so PRP2 is correctly treated as a PRP list.
static inline uint32_t bam_safe_nblocks(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) nblk = 17;
    return nblk;
}

namespace GidpBam {

static size_t s_kernel_launches;

// BaM page_cache per-slot overhead beyond the data DMA pages (Cond3 path):
//   PRP list DMA (ctrl_page_size=4096) + cache_page_t (32) + prp1+prp2 (16)
constexpr uint64_t BAM_PC_OVERHEAD_PER_SLOT = 4096 + 32 + 16;

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
    // Per-field: compressed_page_sizes[], compressed_offsets[]
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

    // ── 6. Zone map metadata (ctx created after io_ctx; eval inside timing) ──
    std::vector<size_t> active_pages;
    bool use_zonemap = options.enable_zonemap;
    uint64_t zm_stats_start = 0, zm_stats_npg = 0, zm_nstats = 0;
    {
        size_t sd_col = q6_cols[0];  // L_SHIPDATE
        zm_stats_start = metadata.table_lineitem_stats_start_page_ids[sd_col];
        zm_stats_npg   = metadata.table_lineitem_stats_npages[sd_col];
        zm_nstats      = metadata.table_lineitem_nstats[sd_col];
        if (!(use_zonemap && zm_nstats > 0 && zm_stats_start > 0 && zm_stats_npg > 0))
            use_zonemap = false;
    }
    if (!use_zonemap) {
        active_pages.resize(npages);
        std::iota(active_pages.begin(), active_pages.end(), size_t(0));
    }

    // ── 7. GPU memory allocation (pipeline, batch-sized) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    constexpr int NPIPE = 4;
    constexpr size_t BATCH_PAGES_MAX = 1536;

    size_t n_compressed = 0;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
        if (field_comp_methods[fi] != CompressionMethod::NONE) n_compressed++;

    // Create BaM I/O context first (page_cache DMA has large fixed overhead)
    const uint32_t io_blocks = static_cast<uint32_t>(sm_count);
    BamBulkReadCtx io_ctx = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        io_blocks,
        static_cast<uint32_t>(BATCH_PAGES_MAX * NUM_FIELDS));

    // Zone map ctx: borrows io_ctx's page_cache
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (use_zonemap) {
        zm_ctx = bam_zonemap_ctx_create(
            io_ctx.d_ctrls, io_ctx.d_pc, io_ctx.pc_base,
            static_cast<uint32_t>(page_size), npages);
        for (uint64_t j = 0; j < zm_stats_npg; j++) {
            uint64_t pg_id = zm_stats_start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            zm_ctx.h_reads[j] = {
                ds.partition_start_lbas[dev] + local * blocks_per_page,
                static_cast<uint32_t>(blocks_per_page), dev};
        }
        zm_nreads = static_cast<uint32_t>(zm_stats_npg);
        zm_ctx.h_preds[0] = {0, zm_nstats,
            options.q6_sd_low, options.q6_sd_high - 1};
        zm_npreds = 1;
        zm_valid = true;
    }

    // Compute BATCH_PAGES from measured remaining GPU memory (40 GiB total cap)
    constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
    size_t gpu_free_after_io = 0;
    cudaMemGetInfo(&gpu_free_after_io, &gpu_total_dummy);
    uint64_t total_used = gpu_free_start - gpu_free_after_io;
    uint64_t remaining = (GPU_MEM_BUDGET > total_used) ? GPU_MEM_BUDGET - total_used : 0;
    size_t bytes_per_batch_page = (NPIPE * NUM_FIELDS + n_compressed) * page_size;
    size_t batch_by_mem = (bytes_per_batch_page > 0)
        ? remaining / bytes_per_batch_page : BATCH_PAGES_MAX;
    const size_t BATCH_PAGES = std::min(BATCH_PAGES_MAX, std::max(batch_by_mem, (size_t)1));

    // Data buffers: NPIPE × NUM_FIELDS ring buffer
    void *data_buf[NPIPE][NUM_FIELDS];
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            data_buf[p][fi] = mb_cuda_alloc(BATCH_PAGES * page_size);

    // Per-field IO buffers (compressed fields only, shared across NPIPE)
    void *io_buf[NUM_FIELDS] = {};
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            if (field_comp_methods[fi] != CompressionMethod::NONE)
                io_buf[fi] = mb_cuda_alloc(BATCH_PAGES * page_size);
    }

    int64_t *d_q6_revenue = static_cast<int64_t*>(mb_cuda_alloc(sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_q6_revenue, 0, sizeof(int64_t)));

    // Per-field nvCOMP contexts (shared across NPIPE — serialized on stream_comp)
    NvcompDecompCtx nvctx[NUM_FIELDS]{};
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            std::vector<FieldPageInfo> tf(1);
            tf[0].compression_method = field_comp_methods[fi];
            Gidp::nvcomp_decompctx_alloc(nvctx[fi], BATCH_PAGES, page_size, tf);
        }
    }

    // 1 IO stream + 1 compute stream
    cudaStream_t stream_io, stream_comp;
    CUDA_CHECK(cudaStreamCreate(&stream_io));
    CUDA_CHECK(cudaStreamCreate(&stream_comp));
    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // ── 8. Pipelined batch execution ──
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    const size_t L_SHIPDATE_IDX = 0;
    const size_t L_QUANTITY_IDX = 1;
    const size_t L_EXTENDEDPRICE_IDX = 2;
    const size_t L_DISCOUNT_IDX = 3;
    uint32_t capacity = (page_size - 12) / 4;

    // Pre-compute batch→active_pages mapping and filter empty batches
    const size_t num_batches_all = (npages + BATCH_PAGES - 1) / BATCH_PAGES;
    std::vector<std::vector<size_t>> batch_actives(num_batches_all);
    std::vector<size_t> non_empty_batches;
    size_t num_batches = 0;
    if (!use_zonemap) {
        size_t ap_idx = 0;
        for (size_t b = 0; b < num_batches_all; b++) {
            size_t pg_base = b * BATCH_PAGES;
            size_t pg_end = std::min(pg_base + BATCH_PAGES, npages);
            while (ap_idx < active_pages.size() && active_pages[ap_idx] < pg_base)
                ap_idx++;
            for (size_t i = ap_idx; i < active_pages.size() && active_pages[i] < pg_end; i++)
                batch_actives[b].push_back(active_pages[i]);
            if (!batch_actives[b].empty())
                non_empty_batches.push_back(b);
        }
        num_batches = non_empty_batches.size();
    }

    // Per-batch pinned host arrays for fully async nvcomp H2D.
    // Each batch gets its own slice so the host never overwrites pinned memory
    // that the GPU's cudaMemcpyAsync hasn't consumed yet.
    void   **pb_h_comp_ptrs[NUM_FIELDS]{};
    void   **pb_h_decomp_ptrs[NUM_FIELDS]{};
    size_t  *pb_h_comp_sizes[NUM_FIELDS]{};
    size_t  *pb_h_decomp_sizes[NUM_FIELDS]{};
    if (any_compressed && num_batches_all > 0) {
        size_t total_slots = num_batches_all * BATCH_PAGES;
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            CUDA_CHECK(cudaMallocHost(&pb_h_comp_ptrs[fi],    total_slots * sizeof(void*)));
            CUDA_CHECK(cudaMallocHost(&pb_h_decomp_ptrs[fi],  total_slots * sizeof(void*)));
            CUDA_CHECK(cudaMallocHost(&pb_h_comp_sizes[fi],   total_slots * sizeof(size_t)));
            CUDA_CHECK(cudaMallocHost(&pb_h_decomp_sizes[fi], total_slots * sizeof(size_t)));
        }
    }

    // Lambda: build I/O descriptors for all fields into io_ctx.h_descs[0]
    auto build_tile_descs = [&](size_t pipe_idx, int buf)
        -> std::pair<uint32_t, uint64_t>
    {
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        auto& ba = batch_actives[batch_idx];
        uint32_t ndescs = 0;
        uint64_t io_bytes = 0;

        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
            size_t io_offset = 0;
            for (size_t pg : ba) {
                size_t local_pg = pg - pg_base;
                BamBulkReadDesc desc{};
                if (is_compressed) {
                    uint64_t byte_offset = h_comp_offsets[fi][pg];
                    uint64_t page_id = field_start_page_ids[fi] + pg;
                    uint32_t dev = page_id % n_devices;
                    desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                    desc.device = dev;
                    desc.dest = static_cast<char*>(io_buf[fi]) + io_offset;
                    desc.copy_bytes = comp_sz;
                    io_offset += page_size;
                } else {
                    uint64_t page_id = field_start_page_ids[fi] + pg;
                    uint32_t dev = page_id % n_devices;
                    uint64_t local_pg_dev = page_id / n_devices;
                    desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                    desc.nblocks = blocks_per_page;
                    desc.device = dev;
                    desc.dest = static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size;
                    desc.copy_bytes = page_size;
                }
                io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                io_ctx.h_descs[0][ndescs++] = desc;
            }
        }
        return {ndescs, io_bytes};
    };

    // Lambda: decompress all fields from io_buf[fi] → data_buf[buf][fi]
    auto run_tile_decomp = [&](size_t pipe_idx, int buf) {
        if (!any_compressed) return;
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        auto& ba = batch_actives[batch_idx];
        size_t slot_base = pipe_idx * BATCH_PAGES;

        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            size_t decomp_count = 0;
            for (size_t idx = 0; idx < ba.size(); idx++) {
                size_t pg = ba[idx];
                size_t local_pg = pg - pg_base;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                if (comp_sz < page_size) {
                    pb_h_comp_ptrs[fi][slot_base + decomp_count] =
                        static_cast<char*>(io_buf[fi]) + idx * page_size;
                    pb_h_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                    pb_h_decomp_ptrs[fi][slot_base + decomp_count] =
                        static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size;
                    pb_h_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                    decomp_count++;
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size,
                        static_cast<char*>(io_buf[fi]) + idx * page_size,
                        page_size, cudaMemcpyDeviceToDevice, stream_comp));
                }
            }
            if (decomp_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_comp_ptrs,
                    pb_h_comp_ptrs[fi] + slot_base,
                    decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_decomp_ptrs,
                    pb_h_decomp_ptrs[fi] + slot_base,
                    decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_comp_sizes,
                    pb_h_comp_sizes[fi] + slot_base,
                    decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_decomp_sizes,
                    pb_h_decomp_sizes[fi] + slot_base,
                    decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                Gidp::nvcomp_decompctx_run(field_comp_methods[fi], nvctx[fi],
                                           decomp_count, page_size, stream_comp,
                                           /*do_sync=*/false, /*skip_h2d=*/true);
                s_kernel_launches++;
            }
        }
    };

    // Lambda: zero inactive page headers + Q6 scan
    auto run_scan = [&](size_t pipe_idx, int buf) {
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        size_t batch_np = std::min(BATCH_PAGES, npages - pg_base);
        auto& ba = batch_actives[batch_idx];

        // Zone map partial batch: zero inactive page headers (nalloc=0)
        bool batch_partial = use_zonemap && ba.size() < batch_np;
        if (batch_partial) {
            std::vector<bool> pg_active(batch_np, false);
            for (size_t pg : ba) pg_active[pg - pg_base] = true;
            for (size_t lp = 0; lp < batch_np; lp++) {
                if (!pg_active[lp]) {
                    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                        CUDA_CHECK(cudaMemsetAsync(
                            static_cast<char*>(data_buf[buf][fi]) + lp * page_size,
                            0, 12, stream_comp));
                }
            }
        }

        q6_col_vardate(
            data_buf[buf][L_SHIPDATE_IDX],
            data_buf[buf][L_QUANTITY_IDX],
            data_buf[buf][L_DISCOUNT_IDX],
            data_buf[buf][L_EXTENDEDPRICE_IDX],
            batch_np, page_size, (uint64_t)batch_np * capacity,
            d_q6_revenue, stream_comp,
            options.q6_sd_low, options.q6_sd_high);
        s_kernel_launches++;
    };

    // Pipeline events
    cudaEvent_t ev_io_done, ev_decomp_done;
    CUDA_CHECK(cudaEventCreate(&ev_io_done));
    CUDA_CHECK(cudaEventCreate(&ev_decomp_done));
    std::vector<cudaEvent_t> event_comp_vec(num_batches_all);
    for (size_t i = 0; i < num_batches_all; i++)
        CUDA_CHECK(cudaEventCreate(&event_comp_vec[i]));

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream_io);
    }

    // ════════════════════════════════════════════
    // total_start — pipelined execution
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    uint64_t total_io_count = 0;
    uint64_t total_io_bytes = 0;

    // ── Zone map GPU eval (fused IO + eval, single kernel) ──
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream_io);
        CUDA_CHECK(cudaStreamSynchronize(stream_io));
        s_kernel_launches++;
        for (size_t pg = 0; pg < npages; pg++)
            if (zm_ctx.h_mask[pg]) active_pages.push_back(pg);

        std::cout << "[ZONEMAP] L_SHIPDATE pruning: active="
                  << active_pages.size() << "/" << npages << std::endl;

        // Populate batch_actives
        {
            size_t ap_idx = 0;
            for (size_t b = 0; b < num_batches_all; b++) {
                size_t pg_base = b * BATCH_PAGES;
                size_t pg_end = std::min(pg_base + BATCH_PAGES, npages);
                while (ap_idx < active_pages.size() && active_pages[ap_idx] < pg_base)
                    ap_idx++;
                for (size_t i = ap_idx; i < active_pages.size() && active_pages[i] < pg_end; i++)
                    batch_actives[b].push_back(active_pages[i]);
                if (!batch_actives[b].empty())
                    non_empty_batches.push_back(b);
            }
            num_batches = non_empty_batches.size();
        }
    }

    std::cout << "[GIDP+BAM Q6] Pipeline: NPIPE=" << NPIPE
              << " BATCH_PAGES=" << BATCH_PAGES
              << " batches=" << num_batches << "/" << num_batches_all << std::endl;

    for (size_t tile = 0; tile < num_batches; tile++) {
        int buf = tile % NPIPE;

        // Wait for data_buf[buf] to be free (NPIPE tiles ago)
        if (tile >= (size_t)NPIPE)
            CUDA_CHECK(cudaStreamWaitEvent(stream_io, event_comp_vec[tile - NPIPE]));
        // Wait for io_buf to be free (previous tile's decomp)
        if (tile > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_io, ev_decomp_done));
            CUDA_CHECK(cudaEventSynchronize(io_ctx.h2d_done[0]));
        }

        // IO all fields into io_buf (compressed) / data_buf[buf] (uncompressed)
        auto [ndescs, io_bytes] = build_tile_descs(tile, buf);
        bam_bulk_read_async(io_ctx, ndescs, 0, stream_io);
        s_kernel_launches++;
        CUDA_CHECK(cudaEventRecord(ev_io_done, stream_io));
        total_io_count += ndescs;
        total_io_bytes += io_bytes;

        // Decomp: io_buf → data_buf[buf]
        CUDA_CHECK(cudaStreamWaitEvent(stream_comp, ev_io_done));
        run_tile_decomp(tile, buf);
        CUDA_CHECK(cudaEventRecord(ev_decomp_done, stream_comp));

        // Scan (all fields decompressed into data_buf[buf])
        run_scan(tile, buf);
        CUDA_CHECK(cudaEventRecord(event_comp_vec[tile], stream_comp));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_comp));

    // ── 12. Result ──
    int64_t h_q6_revenue = 0;
    CUDA_CHECK(cudaMemcpy(&h_q6_revenue, d_q6_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::cout << "TPCH Q6 total revenue: " << h_q6_revenue << std::endl;

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── 13. Cleanup ──
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            if (field_comp_methods[fi] != CompressionMethod::NONE)
                Gidp::nvcomp_decompctx_free(nvctx[fi]);
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (pb_h_comp_ptrs[fi])    cudaFreeHost(pb_h_comp_ptrs[fi]);
            if (pb_h_decomp_ptrs[fi])  cudaFreeHost(pb_h_decomp_ptrs[fi]);
            if (pb_h_comp_sizes[fi])   cudaFreeHost(pb_h_comp_sizes[fi]);
            if (pb_h_decomp_sizes[fi]) cudaFreeHost(pb_h_decomp_sizes[fi]);
        }
    }
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            mb_cuda_free(data_buf[p][fi]);
    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
        if (io_buf[fi]) mb_cuda_free(io_buf[fi]);
    mb_cuda_free(d_q6_revenue);
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    bam_bulk_read_ctx_destroy(io_ctx);
    CUDA_CHECK(cudaEventDestroy(ev_io_done));
    CUDA_CHECK(cudaEventDestroy(ev_decomp_done));
    for (size_t i = 0; i < num_batches_all; i++)
        CUDA_CHECK(cudaEventDestroy(event_comp_vec[i]));
    CUDA_CHECK(cudaStreamDestroy(stream_io));
    CUDA_CHECK(cudaStreamDestroy(stream_comp));
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
// TPC-H Q13 — GIDP+BAM execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMP (host-launched)
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

    std::cout << "=== TPCH Q13 (GIDP+BAM) ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;

    // ── 4. Extract Q13 field info ──
    constexpr size_t FI_O_CUSTKEY = 0;
    constexpr size_t FI_O_COMMENT = 1;
    constexpr size_t FI_C_CUSTKEY = 2;
    constexpr size_t NUM_FIELDS = 3;

    const size_t blocks_per_page = page_size / 512;
    const size_t o_custkey_col = TPCH::common::O_CUSTKEY;
    const size_t o_comment_col = TPCH::common::O_COMMENT;
    const size_t c_custkey_col = TPCH::common::C_CUSTKEY;

    uint64_t field_start_page_ids[NUM_FIELDS];
    uint64_t field_npages_arr[NUM_FIELDS];
    CompressionMethod field_comp_methods[NUM_FIELDS];

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

    const char *field_names[NUM_FIELDS] = { "O_CUSTKEY", "O_COMMENT", "C_CUSTKEY" };
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
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
    std::vector<uint32_t> h_comp_sizes[NUM_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_FIELDS];

    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) continue;

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
        // raw[0]=0, raw[1..npages_field]=cumulative → skip leading 0
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

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    // ── Batch infrastructure (shared across all Q13 fields, double-buffered) ──
    constexpr uint64_t Q13_BATCH_PAGES = 512;
    constexpr int Q13_NBUF = 2;
    void *staging_data[Q13_NBUF], *staging_io[Q13_NBUF];
    for (int b = 0; b < Q13_NBUF; b++) {
        staging_data[b] = mb_cuda_alloc(Q13_BATCH_PAGES * page_size);
        staging_io[b] = mb_cuda_alloc(Q13_BATCH_PAGES * page_size);
    }

    bool any_compressed = false;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
        if (field_comp_methods[fi] != CompressionMethod::NONE) any_compressed = true;
    NvcompDecompCtx nvctx[Q13_NBUF]{};
    if (any_compressed) {
        for (int b = 0; b < Q13_NBUF; b++) {
            std::vector<FieldPageInfo> tf(1);
            tf[0].compression_method = field_comp_methods[FI_O_CUSTKEY];
            Gidp::nvcomp_decompctx_alloc(nvctx[b], Q13_BATCH_PAGES, page_size, tf);
        }
    }

    uint64_t *d_batch_ps[Q13_NBUF];
    for (int b = 0; b < Q13_NBUF; b++)
        CUDA_CHECK(cudaMalloc(&d_batch_ps[b], Q13_BATCH_PAGES * sizeof(uint64_t)));
    std::vector<uint64_t> bps(Q13_BATCH_PAGES);

    // Persistent BaM I/O context (page_cache allocated once, measured in gpu_app_bytes)
    BamBulkReadCtx io_ctx = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(sm_count),
        static_cast<uint32_t>(Q13_BATCH_PAGES));

    cudaStream_t stream_io;
    CUDA_CHECK(cudaStreamCreate(&stream_io));

    // ── Split helpers for double-buffered IO + decompress ──
    auto batch_build_descs = [&](int buf, size_t fi, size_t pg_start, size_t batch_np) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        auto* descs = io_ctx.h_descs[buf];
        for (size_t pg = 0; pg < batch_np; pg++) {
            auto &desc = descs[pg];
            size_t abs_pg = pg_start + pg;
            if (is_compressed) {
                uint64_t page_id = field_start_page_ids[fi] + abs_pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + h_comp_offsets[fi][abs_pg] / 512;
                desc.nblocks = bam_safe_nblocks((roundup4096(h_comp_sizes[fi][abs_pg]) + 511) / 512);
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_io[buf]) + pg * page_size;
                desc.copy_bytes = h_comp_sizes[fi][abs_pg];
            } else {
                uint64_t page_id = field_start_page_ids[fi] + abs_pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + (page_id / n_devices) * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_data[buf]) + pg * page_size;
                desc.copy_bytes = page_size;
            }
        }
    };

    auto batch_decomp = [&](int buf, size_t fi, size_t pg_start, size_t batch_np) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        if (!is_compressed) return;
        size_t dc = 0;
        for (size_t pg = 0; pg < batch_np; pg++) {
            size_t abs_pg = pg_start + pg;
            uint32_t cs = h_comp_sizes[fi][abs_pg];
            if (cs < page_size) {
                nvctx[buf].h_comp_ptrs[dc]   = static_cast<char*>(staging_io[buf]) + pg * page_size;
                nvctx[buf].h_comp_sizes[dc]  = cs;
                nvctx[buf].h_decomp_ptrs[dc] = static_cast<char*>(staging_data[buf]) + pg * page_size;
                nvctx[buf].h_decomp_sizes[dc] = page_size;
                dc++;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(staging_data[buf]) + pg * page_size,
                    static_cast<char*>(staging_io[buf]) + pg * page_size,
                    page_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
        if (dc > 0) {
            Gidp::nvcomp_decompctx_run(field_comp_methods[fi], nvctx[buf], dc, page_size, stream);
            s_kernel_launches++;
        }
    };

    // Combined synchronous IO + decomp (buf=0, for non-pipelined callers)
    auto batch_read_field = [&](size_t fi, size_t pg_start, size_t batch_np) {
        batch_build_descs(0, fi, pg_start, batch_np);
        bam_bulk_read_async(io_ctx, (uint32_t)batch_np, 0, stream);
        s_kernel_launches++;
        batch_decomp(0, fi, pg_start, batch_np);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    auto upload_batch_ps = [&](int buf, const std::vector<uint64_t> &h_ps, size_t pg, size_t bnp,
                               uint64_t &row_start, uint64_t &batch_nrecs) {
        row_start = (pg == 0) ? 0 : h_ps[pg - 1];
        batch_nrecs = h_ps[pg + bnp - 1] - row_start;
        for (size_t p = 0; p < bnp; p++)
            bps[p] = h_ps[pg + p] - row_start;
        CUDA_CHECK(cudaMemcpy(d_batch_ps[buf], bps.data(), bnp * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
    };

    // Double-buffered flatten: buf alternates 0,1 across batches
    auto batch_flatten_int64 = [&](size_t fi, const std::vector<uint64_t> &h_ps,
                                   uint64_t nrecs_total, uint64_t *d_flat) {
        size_t npages = field_npages_arr[fi];
        size_t num_batches = (npages + Q13_BATCH_PAGES - 1) / Q13_BATCH_PAGES;
        for (size_t b = 0; b < num_batches; b++) {
            int buf = b % Q13_NBUF;
            size_t pg = b * Q13_BATCH_PAGES;
            size_t bnp = std::min(Q13_BATCH_PAGES, npages - pg);

            batch_build_descs(buf, fi, pg, bnp);
            bam_bulk_read_async(io_ctx, (uint32_t)bnp, buf, stream);
            s_kernel_launches++;
            batch_decomp(buf, fi, pg, bnp);

            uint64_t row_start, batch_nrecs;
            upload_batch_ps(buf, h_ps, pg, bnp, row_start, batch_nrecs);
            q13_flatten_int64_pages_ps(
                static_cast<const char*>(staging_data[buf]), page_size, d_batch_ps[buf],
                bnp, batch_nrecs, d_flat + row_start, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // ── 8. Allocate flatten outputs ──
    uint64_t *d_o_custkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_custkey_flat, nrecs_orders * sizeof(uint64_t)));
    uint64_t *d_c_custkey = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c_custkey, nrecs_customer * sizeof(uint64_t)));

    // ── 9. KMP setup ──
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

    // ── 11. Allocate scan output arrays ──
    uint64_t *d_o_aggr_custkey = nullptr;
    uint64_t *d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_aggr_custkey, nrecs_orders * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemsetAsync(d_o_aggr_custkey, 0xFF,
                               nrecs_orders * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream));

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
    uint64_t total_io_count = 0, total_io_bytes = 0;

    // ── 12a. Batch flatten O_CUSTKEY + C_CUSTKEY (inside timing) ──
    batch_flatten_int64(FI_O_CUSTKEY, h_ps_o_custkey, nrecs_orders, d_o_custkey_flat);
    batch_flatten_int64(FI_C_CUSTKEY, h_ps_c_custkey, nrecs_customer, d_c_custkey);

    // Count IO for O_CUSTKEY and C_CUSTKEY
    for (size_t fi : {FI_O_CUSTKEY, FI_C_CUSTKEY}) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        for (uint64_t pg = 0; pg < field_npages_arr[fi]; pg++) {
            total_io_count++;
            if (is_compressed)
                total_io_bytes += (roundup4096(h_comp_sizes[fi][pg]) + 511) / 512 * 512;
            else
                total_io_bytes += page_size;
        }
    }

    // ── 12b. Process O_COMMENT in batches (double-buffered IO || decomp+scan) ──
    const uint64_t o_comment_npages = field_npages_arr[FI_O_COMMENT];

    std::cout << "[GIDP+BAM Q13] Processing O_COMMENT in batches of "
              << Q13_BATCH_PAGES << " pages (" << o_comment_npages << " total)..."
              << std::endl;

    {
        size_t num_batches = (o_comment_npages + Q13_BATCH_PAGES - 1) / Q13_BATCH_PAGES;
        for (size_t b = 0; b < num_batches; b++) {
            int buf = b % Q13_NBUF;
            uint64_t pg_start = b * Q13_BATCH_PAGES;
            uint64_t batch_count = std::min(Q13_BATCH_PAGES, o_comment_npages - pg_start);

            batch_build_descs(buf, FI_O_COMMENT, pg_start, batch_count);
            bam_bulk_read_async(io_ctx, (uint32_t)batch_count, buf, stream);
            s_kernel_launches++;
            batch_decomp(buf, FI_O_COMMENT, pg_start, batch_count);

            // Upload batch-relative prefix_sum
            uint64_t rec_start, nrecs_batch;
            upload_batch_ps(buf, h_ps_o_comment, pg_start, batch_count, rec_start, nrecs_batch);

            // IO accounting
            {
                size_t batch_bytes = 0;
                bool is_compressed = (field_comp_methods[FI_O_COMMENT] != CompressionMethod::NONE);
                for (uint64_t pg = 0; pg < batch_count; pg++) {
                    size_t abs_pg = pg_start + pg;
                    if (is_compressed)
                        batch_bytes += (roundup4096(h_comp_sizes[FI_O_COMMENT][abs_pg]) + 511) / 512 * 512;
                    else
                        batch_bytes += page_size;
                }
                total_io_count += batch_count;
                total_io_bytes += batch_bytes;
            }

            // Scan batch → d_o_aggr_custkey[rec_start..]
            q13_scan_batch(
                static_cast<const char*>(staging_data[buf]),
                d_batch_ps[buf], (uint32_t)batch_count, (uint32_t)page_size, nrecs_batch,
                d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
                num_patterns, total_pattern_chars,
                d_o_custkey_flat + rec_start,
                d_o_aggr_custkey + rec_start,
                d_count, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    uint64_t h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    std::cout << "[Q13] Qualifying orders (NOT LIKE): " << h_count
              << " / " << nrecs_orders << std::endl;

    // ── 13. Aggregation pipeline (sort → RLE → probe → sort → RLE → pack) ──
    std::cout << "[Q13] Running aggregation pipeline..." << std::endl;
    std::vector<std::pair<uint32_t, uint32_t>> q13_result;
    q13_pig_aggregate(q13_bufs, d_o_aggr_custkey, nrecs_orders,
                       d_c_custkey, nrecs_customer,
                       q13_result, stream);
    s_kernel_launches++;

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
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    cudaFree(d_count);
    bam_bulk_read_ctx_destroy(io_ctx);
    for (int b = 0; b < Q13_NBUF; b++) {
        mb_cuda_free(staging_data[b]);
        mb_cuda_free(staging_io[b]);
        cudaFree(d_batch_ps[b]);
    }
    if (any_compressed) {
        for (int b = 0; b < Q13_NBUF; b++)
            Gidp::nvcomp_decompctx_free(nvctx[b]);
    }
    CUDA_CHECK(cudaStreamDestroy(stream_io));
    cudaFree(d_patterns);
    cudaFree(d_next);
    cudaFree(d_pattern_offsets);
    cudaFree(d_pattern_lengths);
    cudaFree(d_o_custkey_flat);
    cudaFree(d_c_custkey);
    cudaFree(d_o_aggr_custkey);
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

    size_t total_pages = 0;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
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
// TPC-H Q16 — GIDP+BAM execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMP (host-launched)
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

    std::cout << "=== TPCH Q16 (GIDP+BAM) ===" << std::endl;
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

    // Helper: get compression metadata pointers for a field
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

    // Helper: get prefix_sum metadata for a field
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

    // ── 7. GPU memory allocation (persistent structures) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr size_t Q16_BATCH_PAGES = 512;
    constexpr int Q16_NBUF = 2;

    // Staging buffers (double-buffered for IO || decomp+compute overlap)
    void *staging_data[Q16_NBUF], *staging_io[Q16_NBUF];
    for (int b = 0; b < Q16_NBUF; b++) {
        staging_data[b] = mb_cuda_alloc(Q16_BATCH_PAGES * page_size);
        staging_io[b] = mb_cuda_alloc(Q16_BATCH_PAGES * page_size);
    }

    NvcompDecompCtx nvctx[Q16_NBUF]{};
    if (any_compressed) {
        for (int b = 0; b < Q16_NBUF; b++) {
            std::vector<FieldPageInfo> temp_fields(NUM_Q16_FIELDS);
            for (size_t fi = 0; fi < NUM_Q16_FIELDS; fi++)
                temp_fields[fi].compression_method = field_comp_methods[fi];
            Gidp::nvcomp_decompctx_alloc(nvctx[b], Q16_BATCH_PAGES, page_size, temp_fields);
        }
    }

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    // Batch prefix-sum pool (double-buffered)
    uint64_t *d_ps_pool[Q16_NBUF];
    for (int b = 0; b < Q16_NBUF; b++)
        CUDA_CHECK(cudaMalloc(&d_ps_pool[b], Q16_BATCH_PAGES * sizeof(uint64_t)));

    // 2 large flat pools (reused: SUPP→PART→PARTSUPP)
    uint64_t max_flat_nrecs = std::max({nrecs_supplier, nrecs_part, nrecs_partsupp});
    uint64_t *flat_pool[2];
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaMalloc(&flat_pool[i], max_flat_nrecs * sizeof(uint64_t)));

    // 3 auxiliary uint32 pools for PART phase (brand_ids, type_ids, p_size_u32)
    uint32_t *aux_pool[3];
    for (int i = 0; i < 3; i++)
        CUDA_CHECK(cudaMalloc(&aux_pool[i], nrecs_part * sizeof(uint32_t)));

    // PART hash table (pre-allocated before timing)
    uint32_t ht_capacity = 1;
    while (ht_capacity < nrecs_part * 2)
        ht_capacity <<= 1;
    uint32_t ht_mask = ht_capacity - 1;
    uint64_t *d_ht_keys = nullptr;
    uint32_t *d_ht_group_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_keys, ht_capacity * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_group_ids, ht_capacity * sizeof(uint32_t)));

    uint64_t total_io_count = 0, total_io_bytes = 0;

    // Persistent BaM I/O context (page_cache allocated once, measured in gpu_app_bytes)
    BamBulkReadCtx io_ctx = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(sm_count),
        static_cast<uint32_t>(Q16_BATCH_PAGES));

    cudaStream_t stream_io;
    CUDA_CHECK(cudaStreamCreate(&stream_io));

    // ── Split helpers for double-buffered IO + decompress ──
    auto batch_build_descs = [&](int buf, size_t fi, size_t pg_start, size_t batch_np) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        auto* descs = io_ctx.h_descs[buf];
        for (size_t p = 0; p < batch_np; p++) {
            size_t abs_pg = pg_start + p;
            auto &desc = descs[p];
            if (is_compressed) {
                uint64_t byte_off = h_comp_offsets[fi][abs_pg];
                uint64_t page_id = field_start_page_ids[fi] + abs_pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + byte_off / 512;
                desc.nblocks = bam_safe_nblocks((roundup4096(h_comp_sizes[fi][abs_pg]) + 511) / 512);
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_io[buf]) + p * page_size;
                desc.copy_bytes = h_comp_sizes[fi][abs_pg];
            } else {
                uint64_t page_id = field_start_page_ids[fi] + abs_pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + (page_id / n_devices) * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_data[buf]) + p * page_size;
                desc.copy_bytes = page_size;
            }
        }
    };

    auto batch_decomp = [&](int buf, size_t fi, size_t pg_start, size_t batch_np) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        if (!is_compressed) return;
        size_t dc = 0;
        for (size_t p = 0; p < batch_np; p++) {
            uint32_t cs = h_comp_sizes[fi][pg_start + p];
            if (cs < page_size) {
                nvctx[buf].h_comp_ptrs[dc] = static_cast<char*>(staging_io[buf]) + p * page_size;
                nvctx[buf].h_comp_sizes[dc] = cs;
                nvctx[buf].h_decomp_ptrs[dc] = static_cast<char*>(staging_data[buf]) + p * page_size;
                nvctx[buf].h_decomp_sizes[dc] = page_size;
                dc++;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(staging_data[buf]) + p * page_size,
                    static_cast<char*>(staging_io[buf]) + p * page_size,
                    page_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
        if (dc > 0) {
            Gidp::nvcomp_decompctx_run(field_comp_methods[fi], nvctx[buf], dc, page_size, stream);
            s_kernel_launches++;
        }
    };

    // Combined synchronous IO + decomp (buf=0, for non-pipelined callers)
    auto batch_read_field = [&](size_t fi, size_t pg_start, size_t batch_np) {
        batch_build_descs(0, fi, pg_start, batch_np);
        bam_bulk_read_async(io_ctx, (uint32_t)batch_np, 0, stream);
        s_kernel_launches++;
        total_io_count += batch_np;
        for (size_t p = 0; p < batch_np; p++)
            total_io_bytes += static_cast<uint64_t>(io_ctx.h_descs[0][p].nblocks) * 512;
        batch_decomp(0, fi, pg_start, batch_np);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // Upload batch-relative prefix_sum for field fi, pages [pg, pg+bnp) to d_ps_pool[buf]
    auto upload_batch_ps = [&](int buf, size_t fi, size_t pg, size_t bnp,
                                uint64_t &row_start, uint64_t &batch_nrecs) {
        const auto &ps = h_prefix_sum[fi];
        std::vector<uint64_t> bps_local(bnp);
        row_start = (pg == 0) ? 0 : ps[pg - 1];
        batch_nrecs = ps[pg + bnp - 1] - row_start;
        for (size_t p = 0; p < bnp; p++)
            bps_local[p] = ps[pg + p] - row_start;
        CUDA_CHECK(cudaMemcpy(d_ps_pool[buf], bps_local.data(), bnp * sizeof(uint64_t), cudaMemcpyHostToDevice));
    };

    // Double-buffered flatten INT64: buf alternates 0,1 across batches
    auto batch_flatten_int64 = [&](size_t fi, uint64_t nrecs_total, uint64_t *d_flat) {
        size_t npages = field_npages_arr[fi];
        size_t num_batches = (npages + Q16_BATCH_PAGES - 1) / Q16_BATCH_PAGES;
        for (size_t b = 0; b < num_batches; b++) {
            int buf = b % Q16_NBUF;
            size_t pg = b * Q16_BATCH_PAGES;
            size_t bnp = std::min(Q16_BATCH_PAGES, npages - pg);

            batch_build_descs(buf, fi, pg, bnp);
            bam_bulk_read_async(io_ctx, (uint32_t)bnp, buf, stream);
            s_kernel_launches++;
            total_io_count += bnp;
            for (size_t p = 0; p < bnp; p++)
                total_io_bytes += static_cast<uint64_t>(io_ctx.h_descs[buf][p].nblocks) * 512;
            batch_decomp(buf, fi, pg, bnp);

            uint64_t row_start, batch_nrecs;
            upload_batch_ps(buf, fi, pg, bnp, row_start, batch_nrecs);
            q13_flatten_int64_pages_ps(
                static_cast<const char*>(staging_data[buf]),
                page_size, d_ps_pool[buf], bnp,
                batch_nrecs, d_flat + row_start, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // Double-buffered flatten INT32: buf alternates 0,1 across batches
    auto batch_flatten_int32 = [&](size_t fi, uint64_t nrecs_total, uint64_t *d_flat) {
        size_t npages = field_npages_arr[fi];
        size_t num_batches = (npages + Q16_BATCH_PAGES - 1) / Q16_BATCH_PAGES;
        for (size_t b = 0; b < num_batches; b++) {
            int buf = b % Q16_NBUF;
            size_t pg = b * Q16_BATCH_PAGES;
            size_t bnp = std::min(Q16_BATCH_PAGES, npages - pg);

            batch_build_descs(buf, fi, pg, bnp);
            bam_bulk_read_async(io_ctx, (uint32_t)bnp, buf, stream);
            s_kernel_launches++;
            total_io_count += bnp;
            for (size_t p = 0; p < bnp; p++)
                total_io_bytes += static_cast<uint64_t>(io_ctx.h_descs[buf][p].nblocks) * 512;
            batch_decomp(buf, fi, pg, bnp);

            uint64_t row_start, batch_nrecs;
            upload_batch_ps(buf, fi, pg, bnp, row_start, batch_nrecs);
            q13_flatten_int32_pages_ps(
                static_cast<const char*>(staging_data[buf]),
                page_size, d_ps_pool[buf], bnp,
                batch_nrecs, d_flat + row_start, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

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
        constexpr size_t Q16_MAX_GROUPS = 256 * 1024;
        q16_bufs.h_result_capacity = Q16_MAX_GROUPS;
        q16_bufs.h_gids = (uint32_t *)malloc(Q16_MAX_GROUPS * sizeof(uint32_t));
        q16_bufs.h_counts = (uint32_t *)malloc(Q16_MAX_GROUPS * sizeof(uint32_t));
    }

    // Pre-allocate Phase 0 buffers (SUPPLIER scan)
    uint64_t *d_ps_s_comment = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_s_comment, h_prefix_sum[FI_S_COMMENT].size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_ps_s_comment, h_prefix_sum[FI_S_COMMENT].data(),
                          h_prefix_sum[FI_S_COMMENT].size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaMemcpy(d_kmp_patterns, excl_patterns_str, excl_total_chars,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_kmp_next, excl_total_chars * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_kmp_next, excl_next, excl_total_chars * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_kmp_offsets, excl_num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_kmp_offsets, excl_pattern_offsets,
                          excl_num_patterns * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_kmp_lengths, excl_num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_kmp_lengths, excl_pattern_lengths,
                          excl_num_patterns * sizeof(int), cudaMemcpyHostToDevice));

    uint64_t *d_excl_suppkeys = nullptr;
    uint32_t *d_excl_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_excl_suppkeys, nrecs_supplier * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_excl_count, sizeof(uint32_t)));

    // Pre-allocate d_excl_keys with worst-case capacity (all suppliers excluded)
    uint64_t *d_excl_keys = nullptr;
    uint32_t excl_max_cap = 1;
    while (excl_max_cap < nrecs_supplier * 4 + 16) excl_max_cap <<= 1;
    CUDA_CHECK(cudaMalloc(&d_excl_keys, excl_max_cap * sizeof(uint64_t)));

    // Pre-allocate Phase 1 dict buffers
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

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // ── Phase 0: SUPPLIER (batch) ──
    // Flatten S_SUPPKEY → flat_pool[0]
    batch_flatten_int64(FI_S_SUPPKEY, nrecs_supplier, flat_pool[0]);

    // GPU scan S_COMMENT → excluded suppkeys (batch)
    CUDA_CHECK(cudaMemsetAsync(d_excl_count, 0, sizeof(uint32_t), stream));

    for (size_t pg = 0; pg < field_npages_arr[FI_S_COMMENT]; pg += Q16_BATCH_PAGES) {
        size_t bnp = std::min(Q16_BATCH_PAGES, (size_t)(field_npages_arr[FI_S_COMMENT] - pg));
        batch_read_field(FI_S_COMMENT, pg, bnp);
        const auto &ps = h_prefix_sum[FI_S_COMMENT];
        uint64_t row_base = (pg == 0) ? 0 : ps[pg - 1];
        uint64_t nrecs_batch = ps[pg + bnp - 1] - row_base;
        q16_supplier_scan_batch(
            static_cast<const char *>(staging_data[0]),
            d_ps_s_comment, field_npages_arr[FI_S_COMMENT], pg, page_size,
            flat_pool[0], nrecs_batch, row_base,
            d_kmp_patterns, d_kmp_next, d_kmp_offsets, d_kmp_lengths,
            excl_num_patterns, d_excl_suppkeys, d_excl_count, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    uint32_t h_excl_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_excl_count, d_excl_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "[Q16] Excluded suppliers: " << h_excl_count << std::endl;

    // Build excluded suppkey hash table (d_excl_keys pre-allocated with excl_max_cap)
    uint32_t excl_capacity = 1;
    while (excl_capacity < (uint32_t)h_excl_count * 4 + 16)
        excl_capacity <<= 1;
    uint32_t excl_mask = excl_capacity - 1;

    CUDA_CHECK(cudaMemsetAsync(d_excl_keys, 0xFF, excl_capacity * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    q16_build_excl_ht(d_excl_suppkeys, h_excl_count,
                      d_excl_keys, excl_mask, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Phase 1: PART (batch) ──
    // Flatten P_PARTKEY (INT64) → flat_pool[0]
    batch_flatten_int64(FI_P_PARTKEY, nrecs_part, flat_pool[0]);

    // Flatten P_SIZE (INT32) → flat_pool[1]
    batch_flatten_int32(FI_P_SIZE, nrecs_part, flat_pool[1]);

    // Extract brand_ids from P_BRAND (CHAR) → aux_pool[0]
    constexpr uint32_t BRAND45_ID = 19;
    constexpr uint32_t P_BRAND_PADDED_LEN = 12;
    for (size_t pg = 0; pg < field_npages_arr[FI_P_BRAND]; pg += Q16_BATCH_PAGES) {
        size_t bnp = std::min(Q16_BATCH_PAGES, (size_t)(field_npages_arr[FI_P_BRAND] - pg));
        batch_read_field(FI_P_BRAND, pg, bnp);
        uint64_t row_start, batch_nrecs;
        upload_batch_ps(0, FI_P_BRAND, pg, bnp, row_start, batch_nrecs);
        q16_extract_brand_ids(
            static_cast<const char *>(staging_data[0]),
            d_ps_pool[0], bnp, page_size,
            P_BRAND_PADDED_LEN, batch_nrecs, aux_pool[0] + row_start, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Extract type_ids from P_TYPE (VCHAR) → aux_pool[1]
    // (dict buffers pre-allocated before total_start)
    CUDA_CHECK(cudaMemsetAsync(d_dict_keys, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dict_type_ids, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_type_id_counter, 0, sizeof(uint32_t), stream));
    for (size_t pg = 0; pg < field_npages_arr[FI_P_TYPE]; pg += Q16_BATCH_PAGES) {
        size_t bnp = std::min(Q16_BATCH_PAGES, (size_t)(field_npages_arr[FI_P_TYPE] - pg));
        batch_read_field(FI_P_TYPE, pg, bnp);
        uint64_t row_start, batch_nrecs;
        upload_batch_ps(0, FI_P_TYPE, pg, bnp, row_start, batch_nrecs);
        q16_extract_type_ids(
            static_cast<const char *>(staging_data[0]),
            d_ps_pool[0], bnp, page_size, batch_nrecs,
            d_dict_keys, d_dict_type_ids, d_dict_strs, d_dict_lens,
            d_type_id_counter, aux_pool[1] + row_start, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    uint32_t num_types = 0;
    CUDA_CHECK(cudaMemcpy(&num_types, d_type_id_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "[Q16] P_TYPE distinct values (excluding MEDIUM POLISHED): " << num_types << std::endl;

    // Copy type dictionary to host for result decoding
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

    // Cast P_SIZE uint64_t → uint32_t → aux_pool[2]
    q16_cast_u64_to_u32(flat_pool[1], aux_pool[2], nrecs_part, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Build PART hash table
    CUDA_CHECK(cudaMemsetAsync(d_ht_keys, 0xFF, ht_capacity * sizeof(uint64_t), stream));

    uint64_t p_size_bitmask = (1ULL << 49) | (1ULL << 14) | (1ULL << 23) | (1ULL << 45)
                            | (1ULL << 19) | (1ULL <<  3) | (1ULL << 36) | (1ULL <<  9);

    std::cout << "[Q16] Building PART hash table..." << std::endl;
    q16_build_part_hashtable(flat_pool[0], aux_pool[0], aux_pool[1], aux_pool[2],
        nrecs_part, p_size_bitmask, BRAND45_ID, num_types,
        d_ht_keys, d_ht_group_ids, ht_mask, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Phase 2: PARTSUPP (batch) ──
    // Flatten PS_PARTKEY → flat_pool[0]
    batch_flatten_int64(FI_PS_PARTKEY, nrecs_partsupp, flat_pool[0]);
    // Flatten PS_SUPPKEY → flat_pool[1]
    batch_flatten_int64(FI_PS_SUPPKEY, nrecs_partsupp, flat_pool[1]);

    // ── Phase 3: Q16 pipeline ──
    std::cout << "[Q16] Running Q16 pipeline..." << std::endl;
    std::vector<std::pair<uint32_t, uint32_t>> q16_raw_result;

    q16_golap_pipeline(q16_bufs,
        d_ht_keys, d_ht_group_ids, ht_mask,
        d_excl_keys, excl_mask,
        flat_pool[0], flat_pool[1], nrecs_partsupp,
        q16_raw_result, stream);
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

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total rows: " << q16_result.size() << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    // Phase 0 temporaries (deferred from timed section)
    cudaFree(d_excl_suppkeys);
    cudaFree(d_excl_count);
    cudaFree(d_ps_s_comment);
    cudaFree(d_kmp_patterns);
    cudaFree(d_kmp_next);
    cudaFree(d_kmp_offsets);
    cudaFree(d_kmp_lengths);
    // Phase 1 temporaries (deferred from timed section)
    cudaFree(d_dict_keys);
    cudaFree(d_dict_type_ids);
    cudaFree(d_dict_strs);
    cudaFree(d_dict_lens);
    cudaFree(d_type_id_counter);
    bam_bulk_read_ctx_destroy(io_ctx);
    for (int b = 0; b < Q16_NBUF; b++) {
        mb_cuda_free(staging_data[b]);
        mb_cuda_free(staging_io[b]);
        cudaFree(d_ps_pool[b]);
    }
    for (int i = 0; i < 2; i++) cudaFree(flat_pool[i]);
    for (int i = 0; i < 3; i++) cudaFree(aux_pool[i]);
    cudaFree(d_ht_keys);
    cudaFree(d_ht_group_ids);
    cudaFree(d_excl_keys);
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

    if (any_compressed) {
        for (int b = 0; b < Q16_NBUF; b++)
            Gidp::nvcomp_decompctx_free(nvctx[b]);
    }
    CUDA_CHECK(cudaStreamDestroy(stream_io));
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
// TPC-H Q5 — GIDP+BAM execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMP (host-launched)
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

    std::cout << "=== TPCH Q5 (GIDP+BAM) ===" << std::endl;
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

    // ── 5. Read compression metadata ──
    std::vector<uint32_t> h_comp_sizes[NUM_Q5_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_Q5_FIELDS];

    bool any_compressed = false;

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

    // ── 7. GPU memory allocation (persistent structures only) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = 0;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Tile metrics for LINEITEM streaming ──
    constexpr size_t BATCH_PAGES = 512;
    const uint64_t npages_l_i64 = field_npages_arr[FI_L_ORDERKEY];
    const uint64_t npages_l_i32 = field_npages_arr[FI_L_EXTPRICE];

    // Build full prefix sums with leading 0 for tile computation
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

    // Date parameters (needed early for zone map computation)
    const int32_t date_low  = options.q6_sd_low  ? options.q6_sd_low  : 19940101;
    const int32_t date_high = options.q6_sd_high ? options.q6_sd_high : 19950101;

    uint64_t tile_nrows_max_l = 0;
    size_t l_i64_tile_npages_max = 0;
    uint64_t l_i64_nrows_max = 0;
    for (size_t p_lo = 0; p_lo < npages_l_i32; p_lo += BATCH_PAGES) {
        size_t tile_np = std::min(BATCH_PAGES, npages_l_i32 - p_lo);
        uint64_t first_row = ps_l_i32_full[p_lo];
        uint64_t last_row = ps_l_i32_full[p_lo + tile_np];
        tile_nrows_max_l = std::max(tile_nrows_max_l, last_row - first_row);
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), first_row);
        size_t i64_s = (it_s == ps_l_i64_full.begin()) ? 0 : (size_t)(it_s - ps_l_i64_full.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_l_i64_full.begin());
        l_i64_tile_npages_max = std::max(l_i64_tile_npages_max, i64_e - i64_s);
        l_i64_nrows_max = std::max(l_i64_nrows_max, ps_l_i64_full[i64_e] - ps_l_i64_full[i64_s]);
    }
    if (tile_nrows_max_l == 0) tile_nrows_max_l = 1;
    if (l_i64_nrows_max == 0) l_i64_nrows_max = 1;

    const size_t max_staging_pages = std::max(BATCH_PAGES, l_i64_tile_npages_max);

    // nvCOMP decompression context (sized for max staging pages)
    NvcompDecompCtx nvctx{};
    if (any_compressed) {
        std::vector<FieldPageInfo> temp_fields(NUM_Q5_FIELDS);
        for (size_t fi = 0; fi < NUM_Q5_FIELDS; fi++)
            temp_fields[fi].compression_method = field_comp_methods[fi];
        Gidp::nvcomp_decompctx_alloc(nvctx, max_staging_pages, page_size, temp_fields);
    }

    // Hash tables (persist across phases)
    uint32_t ht_supp_cap = 1;
    while (ht_supp_cap < nrecs_supplier) ht_supp_cap <<= 1;
    uint32_t ht_supp_mask = ht_supp_cap - 1;
    uint64_t *d_ht_supp_keys = nullptr;
    int32_t  *d_ht_supp_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_supp_keys, ht_supp_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_supp_values, ht_supp_cap * sizeof(int32_t)));

    uint32_t ht_cust_cap = 1;
    while (ht_cust_cap < nrecs_customer) ht_cust_cap <<= 1;
    uint32_t ht_cust_mask = ht_cust_cap - 1;
    uint64_t *d_ht_cust_keys = nullptr;
    int32_t  *d_ht_cust_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ht_cust_keys, ht_cust_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ht_cust_values, ht_cust_cap * sizeof(int32_t)));

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

    // ── 8. Helper lambdas ──
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    auto build_field_descs = [&](BamBulkReadDesc *descs, uint32_t &ndescs,
                                  size_t fi, void *data_dest, void *io_dest,
                                  uint64_t pg_start, uint64_t pg_count) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        size_t io_offset = 0;
        for (uint64_t p = 0; p < pg_count; p++) {
            uint64_t pg = pg_start + p;
            auto &desc = descs[ndescs];
            desc = {};
            if (is_compressed) {
                uint64_t byte_offset = h_comp_offsets[fi][pg];
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                desc.device = dev;
                desc.dest = static_cast<char*>(io_dest) + io_offset;
                desc.copy_bytes = comp_sz;
                io_offset += page_size;
            } else {
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                uint64_t local_pg_dev = page_id / n_devices;
                desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(data_dest) + p * page_size;
                desc.copy_bytes = page_size;
            }
            ndescs++;
        }
    };

    auto decompress_field = [&](size_t fi, void *io_buf_ptr, void *data_buf_ptr,
                                 uint64_t pg_start, uint64_t pg_count) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) return;
        size_t decomp_count = 0;
        for (uint64_t p = 0; p < pg_count; p++) {
            uint64_t pg = pg_start + p;
            uint32_t comp_sz = h_comp_sizes[fi][pg];
            if (comp_sz < page_size) {
                nvctx.h_comp_ptrs[decomp_count] =
                    static_cast<char*>(io_buf_ptr) + p * page_size;
                nvctx.h_comp_sizes[decomp_count] = comp_sz;
                nvctx.h_decomp_ptrs[decomp_count] =
                    static_cast<char*>(data_buf_ptr) + p * page_size;
                nvctx.h_decomp_sizes[decomp_count] = page_size;
                decomp_count++;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(data_buf_ptr) + p * page_size,
                    static_cast<char*>(io_buf_ptr) + p * page_size,
                    page_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
        if (decomp_count > 0) {
            Gidp::nvcomp_decompctx_run(field_comp_methods[fi], nvctx,
                                       decomp_count, page_size, stream);
            s_kernel_launches++;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // ── 9. Streaming buffer allocation (all large allocations before total_start) ──
    // Shared staging buffers (reused across all phases)
    void *staging_data = mb_cuda_alloc(max_staging_pages * page_size);
    void *staging_io = mb_cuda_alloc(max_staging_pages * page_size);

    uint64_t *d_batch_ps = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_ps, max_staging_pages * sizeof(uint64_t)));
    std::vector<uint64_t> bps(max_staging_pages);

    // 2 flat array pools (SUPPLIER/CUSTOMER only; ORDERS/LINEITEM use paged kernels)
    uint64_t max_flat_nrecs = std::max(nrecs_supplier, nrecs_customer);
    uint64_t *flat_pool[2];
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaMalloc(&flat_pool[i], max_flat_nrecs * sizeof(uint64_t)));

    // Zone map mask buffer (BATCH_PAGES bytes, reused per batch/tile)
    uint8_t *d_page_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_page_mask, BATCH_PAGES));

    // Persistent BaM I/O context for Phase 1-2 small-table reads
    BamBulkReadCtx io_ctx_small = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(sm_count),
        static_cast<uint32_t>(BATCH_PAGES));

    // I/O counters (declared before lambdas so batch helpers can accumulate)
    uint64_t total_io_count = 0, total_io_bytes = 0;

    // ── Batch helpers ──
    auto batch_read_field = [&](size_t fi, size_t pg_start, size_t pg_count) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        void *io_ptr = is_compressed ? staging_io : nullptr;
        uint32_t ndescs = 0;
        build_field_descs(io_ctx_small.h_descs[0], ndescs, fi, staging_data, io_ptr, pg_start, pg_count);
        bam_bulk_read_async(io_ctx_small, ndescs, 0, stream);
        s_kernel_launches++;
        total_io_count += ndescs;
        for (uint32_t i = 0; i < ndescs; i++)
            total_io_bytes += static_cast<uint64_t>(io_ctx_small.h_descs[0][i].nblocks) * 512;
        decompress_field(fi, io_ptr, staging_data, pg_start, pg_count);
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
        for (size_t pg = 0; pg < field_npages_arr[fi]; pg += BATCH_PAGES) {
            size_t bnp = std::min(BATCH_PAGES, field_npages_arr[fi] - pg);
            batch_read_field(fi, pg, bnp);
            uint64_t row_start, batch_nrecs;
            upload_batch_ps(h_ps, pg, bnp, row_start, batch_nrecs);
            q13_flatten_int64_pages_ps(
                static_cast<const char*>(staging_data), page_size, d_batch_ps,
                bnp, batch_nrecs, d_flat + row_start, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    auto batch_flatten_int32 = [&](size_t fi, const std::vector<uint64_t> &h_ps,
                                   uint64_t nrecs_total, uint64_t *d_flat) {
        for (size_t pg = 0; pg < field_npages_arr[fi]; pg += BATCH_PAGES) {
            size_t bnp = std::min(BATCH_PAGES, field_npages_arr[fi] - pg);
            batch_read_field(fi, pg, bnp);
            uint64_t row_start, batch_nrecs;
            upload_batch_ps(h_ps, pg, bnp, row_start, batch_nrecs);
            q13_flatten_int32_pages_ps(
                static_cast<const char*>(staging_data), page_size, d_batch_ps,
                bnp, batch_nrecs, d_flat + row_start, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // ── Selective batch read (reads only active pages at correct slot positions) ──
    auto batch_read_field_selective = [&](size_t fi, size_t batch_start,
                                          const std::vector<uint32_t> &active_pages) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        auto* descs = io_ctx_small.h_descs[0];
        uint32_t ndescs = 0;
        for (uint32_t pg : active_pages) {
            size_t slot = pg - batch_start;
            auto &desc = descs[ndescs];
            desc = {};
            if (is_compressed) {
                uint64_t byte_offset = h_comp_offsets[fi][pg];
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_io) + slot * page_size;
                desc.copy_bytes = comp_sz;
            } else {
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                uint64_t local_pg_dev = page_id / n_devices;
                desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_data) + slot * page_size;
                desc.copy_bytes = page_size;
            }
            ndescs++;
        }
        if (ndescs > 0) {
            bam_bulk_read_async(io_ctx_small, ndescs, 0, stream);
            s_kernel_launches++;
            total_io_count += ndescs;
            for (uint32_t i = 0; i < ndescs; i++)
                total_io_bytes += static_cast<uint64_t>(descs[i].nblocks) * 512;
        }
        if (is_compressed) {
            size_t decomp_count = 0;
            for (uint32_t pg : active_pages) {
                size_t slot = pg - batch_start;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                if (comp_sz < page_size) {
                    nvctx.h_comp_ptrs[decomp_count] =
                        static_cast<char*>(staging_io) + slot * page_size;
                    nvctx.h_comp_sizes[decomp_count] = comp_sz;
                    nvctx.h_decomp_ptrs[decomp_count] =
                        static_cast<char*>(staging_data) + slot * page_size;
                    nvctx.h_decomp_sizes[decomp_count] = page_size;
                    decomp_count++;
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<char*>(staging_data) + slot * page_size,
                        static_cast<char*>(staging_io) + slot * page_size,
                        page_size, cudaMemcpyDeviceToDevice, stream));
                }
            }
            if (decomp_count > 0) {
                Gidp::nvcomp_decompctx_run(field_comp_methods[fi], nvctx,
                                           decomp_count, page_size, stream);
                s_kernel_launches++;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // ── Zone map metadata (GPU pruning done inside timing, Rule 6) ──
    const bool enable_zonemap = options.enable_zonemap;
    const int32_t asia_dict_id = 2;

    // ORDERS stats metadata
    const size_t o_odate_field = TPCH::common::O_ORDERDATE;
    uint64_t zm_o_odate_nstats      = metadata.table_orders_nstats[o_odate_field];
    uint64_t zm_o_odate_stats_start = metadata.table_orders_stats_start_page_ids[o_odate_field];
    uint64_t zm_o_odate_stats_npg   = metadata.table_orders_stats_npages[o_odate_field];

    const size_t sw_rname_idx = TPCH::common::OS_SIDEWAYS_R_NAME;
    uint64_t zm_o_rname_nstats      = metadata.table_orders_sideways_nstats[o_odate_field][sw_rname_idx];
    uint64_t zm_o_rname_stats_start = metadata.table_orders_sideways_stats_start_page_ids[o_odate_field][sw_rname_idx];
    uint64_t zm_o_rname_stats_npg   = metadata.table_orders_sideways_stats_npages[o_odate_field][sw_rname_idx];

    // LINEITEM stats metadata
    const size_t li_ref_field = TPCH::common::L_EXTENDEDPRICE;
    const size_t li_sw_odate_idx = TPCH::common::LS_SIDEWAYS_O_ORDERDATE;
    uint64_t zm_l_odate_nstats      = metadata.table_lineitem_sideways_nstats[li_ref_field][li_sw_odate_idx];
    uint64_t zm_l_odate_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[li_ref_field][li_sw_odate_idx];
    uint64_t zm_l_odate_stats_npg   = metadata.table_lineitem_sideways_stats_npages[li_ref_field][li_sw_odate_idx];

    const size_t li_sw_rname_idx = TPCH::common::LS_SIDEWAYS_R_NAME;
    uint64_t zm_l_rname_nstats      = metadata.table_lineitem_sideways_nstats[li_ref_field][li_sw_rname_idx];
    uint64_t zm_l_rname_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[li_ref_field][li_sw_rname_idx];
    uint64_t zm_l_rname_stats_npg   = metadata.table_lineitem_sideways_stats_npages[li_ref_field][li_sw_rname_idx];

    // BamZonemapCtx (borrows io_ctx_small page_cache, eval inside timing)
    uint8_t *h_zm_ord_mask = nullptr, *h_zm_li_mask = nullptr;
    BamZonemapCtx zm_ctx_ord{}, zm_ctx_li{};
    uint32_t zm_nreads_ord = 0, zm_npreds_ord = 0;
    uint32_t zm_nreads_li = 0, zm_npreds_li = 0;
    bool zm_valid_ord = false, zm_valid_li = false;

    if (enable_zonemap) {
        // ORDERS ctx
        zm_ctx_ord = bam_zonemap_ctx_create(
            io_ctx_small.d_ctrls, io_ctx_small.d_pc, io_ctx_small.pc_base,
            static_cast<uint32_t>(page_size), npages_o_i32);
        h_zm_ord_mask = zm_ctx_ord.h_mask;
        if (zm_o_odate_nstats > 0 && zm_o_odate_stats_npg > 0) {
            uint32_t offset = zm_nreads_ord;
            for (uint64_t j = 0; j < zm_o_odate_stats_npg; j++) {
                uint64_t pg_id = zm_o_odate_stats_start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                zm_ctx_ord.h_reads[zm_nreads_ord++] = {
                    ds.partition_start_lbas[dev] + local * blocks_per_page,
                    static_cast<uint32_t>(blocks_per_page), dev};
            }
            zm_ctx_ord.h_preds[zm_npreds_ord++] = {
                offset, zm_o_odate_nstats,
                (int32_t)date_low, (int32_t)(date_high - 1)};
        }
        if (zm_o_rname_nstats > 0 && zm_o_rname_stats_npg > 0) {
            uint32_t offset = zm_nreads_ord;
            for (uint64_t j = 0; j < zm_o_rname_stats_npg; j++) {
                uint64_t pg_id = zm_o_rname_stats_start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                zm_ctx_ord.h_reads[zm_nreads_ord++] = {
                    ds.partition_start_lbas[dev] + local * blocks_per_page,
                    static_cast<uint32_t>(blocks_per_page), dev};
            }
            zm_ctx_ord.h_preds[zm_npreds_ord++] = {
                offset, zm_o_rname_nstats,
                asia_dict_id, asia_dict_id};
        }
        if (zm_npreds_ord > 0) zm_valid_ord = true;
        else memset(h_zm_ord_mask, 1, npages_o_i32);

        // LINEITEM ctx
        zm_ctx_li = bam_zonemap_ctx_create(
            io_ctx_small.d_ctrls, io_ctx_small.d_pc, io_ctx_small.pc_base,
            static_cast<uint32_t>(page_size), npages_l_i32);
        h_zm_li_mask = zm_ctx_li.h_mask;
        if (zm_l_odate_nstats > 0 && zm_l_odate_stats_npg > 0) {
            uint32_t offset = zm_nreads_li;
            for (uint64_t j = 0; j < zm_l_odate_stats_npg; j++) {
                uint64_t pg_id = zm_l_odate_stats_start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                zm_ctx_li.h_reads[zm_nreads_li++] = {
                    ds.partition_start_lbas[dev] + local * blocks_per_page,
                    static_cast<uint32_t>(blocks_per_page), dev};
            }
            zm_ctx_li.h_preds[zm_npreds_li++] = {
                offset, zm_l_odate_nstats,
                (int32_t)date_low, (int32_t)(date_high - 1)};
        }
        if (zm_l_rname_nstats > 0 && zm_l_rname_stats_npg > 0) {
            uint32_t offset = zm_nreads_li;
            for (uint64_t j = 0; j < zm_l_rname_stats_npg; j++) {
                uint64_t pg_id = zm_l_rname_stats_start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                zm_ctx_li.h_reads[zm_nreads_li++] = {
                    ds.partition_start_lbas[dev] + local * blocks_per_page,
                    static_cast<uint32_t>(blocks_per_page), dev};
            }
            zm_ctx_li.h_preds[zm_npreds_li++] = {
                offset, zm_l_rname_nstats,
                asia_dict_id, asia_dict_id};
        }
        if (zm_npreds_li > 0) zm_valid_li = true;
        else memset(h_zm_li_mask, 1, npages_l_i32);
    } else {
        h_zm_ord_mask = static_cast<uint8_t *>(malloc(npages_o_i32));
        h_zm_li_mask = static_cast<uint8_t *>(malloc(npages_l_i32));
        memset(h_zm_ord_mask, 1, npages_o_i32);
        memset(h_zm_li_mask, 1, npages_l_i32);
    }

    // ════════════════════════════════════════════
    // Pre-create BaM I/O contexts (outside timing)
    // ════════════════════════════════════════════
    constexpr size_t Q5_TILE_PAGES_O  = 1728;
    constexpr size_t Q5_TILE_PAGES_P4_MAX = 1024;

    // Pre-compute max INT64 tile pages for ORDERS
    size_t pre_o_i64_tile_npages_max = 0;
    for (size_t p_lo = 0; p_lo < npages_o_i32; p_lo += Q5_TILE_PAGES_O) {
        size_t tile_np = std::min(Q5_TILE_PAGES_O, (size_t)(npages_o_i32 - p_lo));
        uint64_t first_row = ps_o_i32_full[p_lo];
        uint64_t last_row  = ps_o_i32_full[p_lo + tile_np];
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_o_i64_full.begin(), ps_o_i64_full.end(), first_row);
        size_t i64_s = (it_s == ps_o_i64_full.begin()) ? 0 : (size_t)(it_s - ps_o_i64_full.begin()) - 1;
        auto it_e = std::upper_bound(ps_o_i64_full.begin(), ps_o_i64_full.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_o_i64_full.begin());
        pre_o_i64_tile_npages_max = std::max(pre_o_i64_tile_npages_max, i64_e - i64_s);
    }

    // Pre-compute max INT64 tile pages for LINEITEM (using MAX for io_ctx sizing)
    size_t pre_l_i64_tile_npages_max_p4 = 0;
    for (size_t p_lo = 0; p_lo < npages_l_i32; p_lo += Q5_TILE_PAGES_P4_MAX) {
        size_t tile_np = std::min(Q5_TILE_PAGES_P4_MAX, npages_l_i32 - p_lo);
        uint64_t first_row = ps_l_i32_full[p_lo];
        uint64_t last_row  = ps_l_i32_full[p_lo + tile_np];
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), first_row);
        size_t i64_s = (it_s == ps_l_i64_full.begin()) ? 0 : (size_t)(it_s - ps_l_i64_full.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_l_i64_full.begin());
        pre_l_i64_tile_npages_max_p4 = std::max(pre_l_i64_tile_npages_max_p4, i64_e - i64_s);
    }

    constexpr int Q5_BAM_NPIPE = 4;  // max(NPIPE_O=1, NPIPE_L=4)
    const uint32_t io_blocks_bam = static_cast<uint32_t>(sm_count);
    uint32_t pre_max_descs_o  = (uint32_t)Q5_TILE_PAGES_O  + 2 * (uint32_t)pre_o_i64_tile_npages_max;
    uint32_t pre_max_descs_p4 = 2 * (uint32_t)Q5_TILE_PAGES_P4_MAX + 2 * (uint32_t)pre_l_i64_tile_npages_max_p4;
    uint32_t max_descs_bam = std::max(pre_max_descs_o, pre_max_descs_p4);
    BamBulkReadCtx io_ctx_bam[Q5_BAM_NPIPE];
    for (int p = 0; p < Q5_BAM_NPIPE; p++)
        io_ctx_bam[p] = bam_bulk_read_ctx_create(
            ctrl, static_cast<uint32_t>(page_size), io_blocks_bam, max_descs_bam);

    // Pre-allocate Phase 3 (ORDERS) tile + paged-kernel buffers
    void *tile_data_o[1][3]{};
    void *tile_io_o[1][3]{};
    uint32_t *d_o_active_pg[1]{};
    uint64_t *d_o_ps_ref[1]{};
    uint64_t *d_o_ps_i64[1]{};
    {
        const size_t o_fi_pre[3] = { FI_O_ORDERDATE, FI_O_ORDERKEY, FI_O_CUSTKEY };
        tile_data_o[0][0] = mb_cuda_alloc(Q5_TILE_PAGES_O * page_size);
        if (field_comp_methods[o_fi_pre[0]] != CompressionMethod::NONE)
            tile_io_o[0][0] = mb_cuda_alloc(Q5_TILE_PAGES_O * page_size);
        for (int fi = 1; fi < 3; fi++) {
            tile_data_o[0][fi] = mb_cuda_alloc(pre_o_i64_tile_npages_max * page_size);
            if (field_comp_methods[o_fi_pre[fi]] != CompressionMethod::NONE)
                tile_io_o[0][fi] = mb_cuda_alloc(pre_o_i64_tile_npages_max * page_size);
        }
        CUDA_CHECK(cudaMalloc(&d_o_active_pg[0], Q5_TILE_PAGES_O * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_o_ps_ref[0], (Q5_TILE_PAGES_O + 1) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_o_ps_i64[0], (pre_o_i64_tile_npages_max + 1) * sizeof(uint64_t)));
    }

    // Compute TILE_PAGES_P4 from 40 GiB total budget (ctrl + app)
    size_t Q5_TILE_PAGES_P4;
    {
        size_t gpu_free_now = 0, gt = 0;
        cudaMemGetInfo(&gpu_free_now, &gt);
        uint64_t total_used = gpu_ctrl_bytes + (gpu_free_before_app - gpu_free_now);
        constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        uint64_t p4_budget = (total_used < GPU_MEM_BUDGET) ? GPU_MEM_BUDGET - total_used : 0;

        const size_t li_fi_tmp[4] = { FI_L_EXTPRICE, FI_L_DISCOUNT, FI_L_ORDERKEY, FI_L_SUPPKEY };
        int n_comp_i32 = 0, n_comp_i64 = 0;
        for (int fi = 0; fi < 2; fi++)
            if (field_comp_methods[li_fi_tmp[fi]] != CompressionMethod::NONE) n_comp_i32++;
        for (int fi = 2; fi < 4; fi++)
            if (field_comp_methods[li_fi_tmp[fi]] != CompressionMethod::NONE) n_comp_i64++;

        double i64_ratio = (npages_l_i32 > 0) ? (double)npages_l_i64 / npages_l_i32 : 2.0;
        i64_ratio = std::max(i64_ratio, 1.0) * 1.05;  // 5% margin for row distribution
        size_t bytes_per_tp = (size_t)(
            (Q5_BAM_NPIPE * 2 + n_comp_i32) * page_size +
            (Q5_BAM_NPIPE * 2 + n_comp_i64) * i64_ratio * page_size);
        Q5_TILE_PAGES_P4 = (bytes_per_tp > 0)
            ? std::min((size_t)Q5_TILE_PAGES_P4_MAX, (size_t)(p4_budget / bytes_per_tp))
            : Q5_TILE_PAGES_P4_MAX;
        if (Q5_TILE_PAGES_P4 == 0) Q5_TILE_PAGES_P4 = 1;
    }

    // Recompute i64 max for actual tile size
    pre_l_i64_tile_npages_max_p4 = 0;
    for (size_t p_lo = 0; p_lo < npages_l_i32; p_lo += Q5_TILE_PAGES_P4) {
        size_t tile_np = std::min(Q5_TILE_PAGES_P4, npages_l_i32 - p_lo);
        uint64_t first_row = ps_l_i32_full[p_lo];
        uint64_t last_row  = ps_l_i32_full[p_lo + tile_np];
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), first_row);
        size_t i64_s = (it_s == ps_l_i64_full.begin()) ? 0 : (size_t)(it_s - ps_l_i64_full.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_l_i64_full.begin());
        pre_l_i64_tile_npages_max_p4 = std::max(pre_l_i64_tile_npages_max_p4, i64_e - i64_s);
    }

    // Pre-allocate Phase 4 (LINEITEM) tile data + paged-kernel buffers
    void *tile_data[Q5_BAM_NPIPE][4]{};
    uint32_t *d_active_pg[Q5_BAM_NPIPE]{};
    uint64_t *d_ps_ref[Q5_BAM_NPIPE]{};
    uint64_t *d_ps_i64[Q5_BAM_NPIPE]{};
    {
        for (int p = 0; p < Q5_BAM_NPIPE; p++) {
            for (int fi = 0; fi < 2; fi++)
                tile_data[p][fi] = mb_cuda_alloc(Q5_TILE_PAGES_P4 * page_size);
            for (int fi = 2; fi < 4; fi++)
                tile_data[p][fi] = mb_cuda_alloc(pre_l_i64_tile_npages_max_p4 * page_size);
            CUDA_CHECK(cudaMalloc(&d_active_pg[p], Q5_TILE_PAGES_P4 * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_ps_ref[p], (Q5_TILE_PAGES_P4 + 1) * sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&d_ps_i64[p], (pre_l_i64_tile_npages_max_p4 + 1) * sizeof(uint64_t)));
        }
    }

    // Per-field IO buffers for Phase 4 LINEITEM (compressed fields only, shared across NPIPE)
    void *io_buf_l[4] = {};
    {
        const size_t li_fi_pre[4] = { FI_L_EXTPRICE, FI_L_DISCOUNT, FI_L_ORDERKEY, FI_L_SUPPKEY };
        for (int fi = 0; fi < 4; fi++) {
            if (field_comp_methods[li_fi_pre[fi]] != CompressionMethod::NONE) {
                size_t npg = (fi < 2) ? Q5_TILE_PAGES_P4 : pre_l_i64_tile_npages_max_p4;
                io_buf_l[fi] = mb_cuda_alloc(npg * page_size);
            }
        }
    }

    // Pre-create Phase 3/4 nvcomp contexts and pinned host arrays (before total_start per Rule 4)
    NvcompDecompCtx nvctx_o[1][3]{};
    void   **pb_comp_ptrs_o[3]{};
    void   **pb_decomp_ptrs_o[3]{};
    size_t  *pb_comp_sizes_o[3]{};
    size_t  *pb_decomp_sizes_o[3]{};
    NvcompDecompCtx nvctx_l[4]{};
    void   **pb_comp_ptrs[4]{};
    void   **pb_decomp_ptrs[4]{};
    size_t  *pb_comp_sizes[4]{};
    size_t  *pb_decomp_sizes[4]{};
    {
        const size_t o_fi_nvctx[3] = { FI_O_ORDERDATE, FI_O_ORDERKEY, FI_O_CUSTKEY };
        size_t n_o_tiles_max = (npages_o_i32 + Q5_TILE_PAGES_O - 1) / Q5_TILE_PAGES_O;
        if (any_compressed) {
            for (int fi = 0; fi < 3; fi++) {
                size_t gfi = o_fi_nvctx[fi];
                if (field_comp_methods[gfi] == CompressionMethod::NONE) continue;
                size_t max_pages = (fi < 1) ? (size_t)Q5_TILE_PAGES_O : pre_o_i64_tile_npages_max;
                std::vector<FieldPageInfo> tf(1);
                tf[0].compression_method = field_comp_methods[gfi];
                Gidp::nvcomp_decompctx_alloc(nvctx_o[0][fi], max_pages, page_size, tf);
                if (n_o_tiles_max > 0) {
                    size_t total_slots = n_o_tiles_max * max_pages;
                    CUDA_CHECK(cudaMallocHost(&pb_comp_ptrs_o[fi],    total_slots * sizeof(void*)));
                    CUDA_CHECK(cudaMallocHost(&pb_decomp_ptrs_o[fi],  total_slots * sizeof(void*)));
                    CUDA_CHECK(cudaMallocHost(&pb_comp_sizes_o[fi],   total_slots * sizeof(size_t)));
                    CUDA_CHECK(cudaMallocHost(&pb_decomp_sizes_o[fi], total_slots * sizeof(size_t)));
                }
            }
        }
        const size_t li_fi_nvctx[4] = { FI_L_EXTPRICE, FI_L_DISCOUNT, FI_L_ORDERKEY, FI_L_SUPPKEY };
        size_t n_l_tiles_max = (npages_l_i32 + Q5_TILE_PAGES_P4 - 1) / Q5_TILE_PAGES_P4;
        if (any_compressed) {
            for (int fi = 0; fi < 4; fi++) {
                size_t gfi = li_fi_nvctx[fi];
                if (field_comp_methods[gfi] == CompressionMethod::NONE) continue;
                size_t max_pages = (fi < 2) ? Q5_TILE_PAGES_P4 : pre_l_i64_tile_npages_max_p4;
                std::vector<FieldPageInfo> tf(1);
                tf[0].compression_method = field_comp_methods[gfi];
                Gidp::nvcomp_decompctx_alloc(nvctx_l[fi], max_pages, page_size, tf);
                if (n_l_tiles_max > 0) {
                    size_t total_slots = n_l_tiles_max * max_pages;
                    CUDA_CHECK(cudaMallocHost(&pb_comp_ptrs[fi],    total_slots * sizeof(void*)));
                    CUDA_CHECK(cudaMallocHost(&pb_decomp_ptrs[fi],  total_slots * sizeof(void*)));
                    CUDA_CHECK(cudaMallocHost(&pb_comp_sizes[fi],   total_slots * sizeof(size_t)));
                    CUDA_CHECK(cudaMallocHost(&pb_decomp_sizes[fi], total_slots * sizeof(size_t)));
                }
            }
        }
    }

    // Pre-create CUDA streams and events for Phase 3/4 (before total_start per Rule 4)
    cudaStream_t stream_io_o[1], stream_comp_o;
    CUDA_CHECK(cudaStreamCreate(&stream_io_o[0]));
    CUDA_CHECK(cudaStreamCreate(&stream_comp_o));
    cudaStream_t stream_io_l, stream_comp;
    CUDA_CHECK(cudaStreamCreate(&stream_io_l));
    CUDA_CHECK(cudaStreamCreate(&stream_comp));
    cudaEvent_t ev_io_done, ev_decomp_done;
    CUDA_CHECK(cudaEventCreate(&ev_io_done));
    CUDA_CHECK(cudaEventCreate(&ev_decomp_done));

    const size_t q5_max_o_tiles = (npages_o_i32 + Q5_TILE_PAGES_O - 1) / Q5_TILE_PAGES_O;
    std::vector<cudaEvent_t> event_io_o(q5_max_o_tiles);
    std::vector<cudaEvent_t> event_comp_o(q5_max_o_tiles);
    for (size_t i = 0; i < q5_max_o_tiles; i++) {
        CUDA_CHECK(cudaEventCreate(&event_io_o[i]));
        CUDA_CHECK(cudaEventCreate(&event_comp_o[i]));
    }
    const size_t q5_max_l_tiles = (npages_l_i32 + Q5_TILE_PAGES_P4 - 1) / Q5_TILE_PAGES_P4;
    std::vector<cudaEvent_t> event_comp_vec(q5_max_l_tiles);
    for (size_t i = 0; i < q5_max_l_tiles; i++)
        CUDA_CHECK(cudaEventCreate(&event_comp_vec[i]));

    // Measure GPU memory (includes BaM I/O contexts + tile buffers)
    {
        size_t gpu_free_after_app = 0;
        cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
        gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;
    }

    // GPU prefix sums for Phase 4 INT64 mask derivation (truncated format)
    uint64_t* d_ps_o_i32_zm = nullptr;
    uint64_t* d_ps_o_i64_zm = nullptr;
    uint64_t* d_ps_l_i32_zm = nullptr;
    uint64_t* d_ps_l_i64_zm = nullptr;
    if (enable_zonemap) {
        cudaMalloc(&d_ps_o_i32_zm, npages_o_i32 * sizeof(uint64_t));
        cudaMemcpy(d_ps_o_i32_zm, h_prefix_sum[FI_O_ORDERDATE].data(),
                   npages_o_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ps_o_i64_zm, npages_o_i64 * sizeof(uint64_t));
        cudaMemcpy(d_ps_o_i64_zm, h_prefix_sum[FI_O_ORDERKEY].data(),
                   npages_o_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ps_l_i32_zm, npages_l_i32 * sizeof(uint64_t));
        cudaMemcpy(d_ps_l_i32_zm, h_prefix_sum[FI_L_EXTPRICE].data(),
                   npages_l_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ps_l_i64_zm, npages_l_i64 * sizeof(uint64_t));
        cudaMemcpy(d_ps_l_i64_zm, h_prefix_sum[FI_L_ORDERKEY].data(),
                   npages_l_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // Zone map INT64 mask buffers (allocated outside timing)
    uint8_t* d_mask_ord_i64 = nullptr;
    uint8_t* d_mask_li_i64 = nullptr;
    if (enable_zonemap) {
        cudaMalloc(&d_mask_ord_i64, npages_o_i64);
        cudaMalloc(&d_mask_li_i64, npages_l_i64);
    }

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid_ord || zm_valid_li) {
        bam_pre_io(zm_ctx_ord.d_ctrls, zm_ctx_ord.d_pc, stream);
    }

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    total_io_count = 0; total_io_bytes = 0;  // reset at timing start

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
        char* d_staging = static_cast<char*>(staging_data);
        size_t off = 0;
        char* d_r_rkey_stg = d_staging + off; off += r_rkey_npages * page_size;
        char* d_r_name_stg = d_staging + off; off += r_name_npages * page_size;
        char* d_n_nkey_stg = d_staging + off; off += n_nkey_npages * page_size;
        char* d_n_rkey_stg = d_staging + off;
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

        // Count REGION/NATION IO
        uint64_t rn_pages = r_rkey_npages + r_name_npages
                          + n_nkey_npages + n_name_npages + n_rkey_npages;
        total_io_count += rn_pages;
        total_io_bytes += rn_pages * page_size;
    }

    auto phase_start = std::chrono::steady_clock::now();

    // ═══════════════════════════════════════════════════════
    // Phase 1: SUPPLIER — batch flatten, HT build
    // ═══════════════════════════════════════════════════════
    {
        batch_flatten_int64(FI_S_SUPPKEY, h_prefix_sum[FI_S_SUPPKEY], nrecs_supplier, flat_pool[0]);
        batch_flatten_int32(FI_S_NATIONKEY, h_prefix_sum[FI_S_NATIONKEY], nrecs_supplier, flat_pool[1]);

        CUDA_CHECK(cudaMemsetAsync(d_ht_supp_keys, 0xFF, ht_supp_cap * sizeof(uint64_t), stream));
        q5_build_supplier_ht(flat_pool[0], flat_pool[1], nrecs_supplier,
            d_nationkey_to_idx, d_ht_supp_keys, d_ht_supp_values, ht_supp_mask, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cout << "[Q5] SUPPLIER HT built (capacity=" << ht_supp_cap << ")" << std::endl;
    }
    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 1 (SUPPLIER): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ═══════════════════════════════════════════════════════
    // Phase 2: CUSTOMER — batch flatten, HT build
    // ═══════════════════════════════════════════════════════
    {
        batch_flatten_int64(FI_C_CUSTKEY, h_prefix_sum[FI_C_CUSTKEY], nrecs_customer, flat_pool[0]);
        batch_flatten_int32(FI_C_NATIONKEY, h_prefix_sum[FI_C_NATIONKEY], nrecs_customer, flat_pool[1]);

        CUDA_CHECK(cudaMemsetAsync(d_ht_cust_keys, 0xFF, ht_cust_cap * sizeof(uint64_t), stream));
        q5_build_customer_ht(flat_pool[0], flat_pool[1], nrecs_customer,
            d_nationkey_to_idx, d_ht_cust_keys, d_ht_cust_values, ht_cust_mask, stream);
        s_kernel_launches++;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cout << "[Q5] CUSTOMER HT built (capacity=" << ht_cust_cap << ")" << std::endl;
    }
    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 2 (CUSTOMER): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 2→3 transition: " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ── Zonemap pruning on GPU (fused IO + eval + INT64 mask derivation) ──
    std::vector<uint8_t> h_mask_ord_i64(npages_o_i64, 1);
    std::vector<uint8_t> h_mask_li_i64(npages_l_i64, 1);
    if (enable_zonemap) {
        if (zm_valid_ord) {
            zm_ctx_ord.d_ps_i32   = d_ps_o_i32_zm;
            zm_ctx_ord.d_ps_i64   = d_ps_o_i64_zm;
            zm_ctx_ord.d_mask_i64 = d_mask_ord_i64;
            zm_ctx_ord.npages_i64 = static_cast<uint32_t>(npages_o_i64);
            bam_zonemap_eval_async(zm_ctx_ord, npages_o_i32, zm_nreads_ord, zm_npreds_ord, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            s_kernel_launches++;
            cudaMemcpy(h_mask_ord_i64.data(), d_mask_ord_i64, npages_o_i64, cudaMemcpyDeviceToHost);
        }
        if (zm_valid_li) {
            zm_ctx_li.d_ps_i32   = d_ps_l_i32_zm;
            zm_ctx_li.d_ps_i64   = d_ps_l_i64_zm;
            zm_ctx_li.d_mask_i64 = d_mask_li_i64;
            zm_ctx_li.npages_i64 = static_cast<uint32_t>(npages_l_i64);
            bam_zonemap_eval_async(zm_ctx_li, npages_l_i32, zm_nreads_li, zm_npreds_li, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            s_kernel_launches++;
            cudaMemcpy(h_mask_li_i64.data(), d_mask_li_i64, npages_l_i64, cudaMemcpyDeviceToHost);
        }

        uint32_t ord_active = 0, li_active = 0;
        for (size_t i = 0; i < npages_o_i32; i++) if (h_zm_ord_mask[i]) ord_active++;
        for (size_t i = 0; i < npages_l_i32; i++) if (h_zm_li_mask[i]) li_active++;
        std::cout << "[ZONEMAP] ORDERS pruning: active=" << ord_active << "/" << npages_o_i32 << std::endl;
        std::cout << "[ZONEMAP] LINEITEM pruning: active=" << li_active << "/" << npages_l_i32 << std::endl;
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Zonemap pruning: " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ═══════════════════════════════════════════════════════════
    // Phase 3: ORDERS — pipelined BaM IO + paged kernel HT build
    // ═══════════════════════════════════════════════════════════
    {
        constexpr int NPIPE = 1;
        constexpr int O_ORDERDATE_F = 0, O_ORDERKEY_F = 1, O_CUSTKEY_F = 2;
        constexpr int O_NFIELDS = 3;
        const size_t o_fi[O_NFIELDS] = { FI_O_ORDERDATE, FI_O_ORDERKEY, FI_O_CUSTKEY };

        constexpr size_t TILE_PAGES_O = Q5_TILE_PAGES_O;
        const size_t o_i64_tile_npages_max = pre_o_i64_tile_npages_max;

        // tile_data_o, tile_io_o, d_o_active_pg, d_o_ps_ref, d_o_ps_i64
        // are pre-allocated before the measurement point.
        std::vector<uint32_t> h_o_apg;
        std::vector<uint64_t> h_o_ps_ref(TILE_PAGES_O + 1);
        std::vector<uint64_t> h_o_ps_i64(o_i64_tile_npages_max + 1);

        // Alias to pre-created shared BaM I/O contexts
        BamBulkReadCtx *io_ctx_o = io_ctx_bam;

        // Pre-compute tile info
        struct OTileInfo {
            size_t p_lo, tile_np;
            uint64_t first_row, last_row, tile_nrows;
            std::vector<uint32_t> active_i32;
            bool use_selective;
            size_t i64_start, i64_end, i64_np;
            uint64_t i64_offset;
            std::vector<uint32_t> i64_active;
            bool use_sel_i64;
        };
        const size_t n_o_tiles_all = (npages_o_i32 + TILE_PAGES_O - 1) / TILE_PAGES_O;
        std::vector<OTileInfo> o_tiles;
        o_tiles.reserve(n_o_tiles_all);
        for (size_t p_lo = 0; p_lo < npages_o_i32; p_lo += TILE_PAGES_O) {
            OTileInfo ti;
            ti.p_lo = p_lo;
            ti.tile_np = std::min(TILE_PAGES_O, (size_t)(npages_o_i32 - p_lo));
            ti.first_row = ps_o_i32_full[p_lo];
            ti.last_row  = ps_o_i32_full[p_lo + ti.tile_np];
            ti.tile_nrows = ti.last_row - ti.first_row;
            if (ti.tile_nrows == 0) continue;

            for (size_t j = 0; j < ti.tile_np; j++)
                if (h_zm_ord_mask[p_lo + j]) ti.active_i32.push_back((uint32_t)(p_lo + j));
            if (enable_zonemap && ti.active_i32.empty()) continue;
            ti.use_selective = enable_zonemap && ti.active_i32.size() < ti.tile_np;

            auto it_s = std::upper_bound(ps_o_i64_full.begin(), ps_o_i64_full.end(), ti.first_row);
            ti.i64_start = (it_s == ps_o_i64_full.begin()) ? 0 : (size_t)(it_s - ps_o_i64_full.begin()) - 1;
            auto it_e = std::upper_bound(ps_o_i64_full.begin(), ps_o_i64_full.end(), ti.last_row - 1);
            ti.i64_end = (size_t)(it_e - ps_o_i64_full.begin());
            ti.i64_np = ti.i64_end - ti.i64_start;
            ti.i64_offset = ti.first_row - ps_o_i64_full[ti.i64_start];

            if (enable_zonemap) {
                for (uint32_t pg = ti.i64_start; pg < ti.i64_end; pg++)
                    if (h_mask_ord_i64[pg]) ti.i64_active.push_back(pg);
            }
            ti.use_sel_i64 = enable_zonemap && ti.i64_active.size() < ti.i64_np;
            o_tiles.push_back(std::move(ti));
        }
        const size_t num_o_tiles = o_tiles.size();
        std::cout << "[GIDP+BAM Q5] ORDERS Pipeline: NPIPE=" << NPIPE
                  << " TILE_PAGES=" << TILE_PAGES_O
                  << " tiles=" << num_o_tiles << "/" << n_o_tiles_all
                  << " (zone map pruned)" << std::endl;

        // Lambda: build IO descriptors for all 3 ORDERS fields
        auto build_o_tile_descs = [&](size_t tile_idx, int buf) -> std::pair<uint32_t, uint64_t> {
            auto &ti = o_tiles[tile_idx];
            uint32_t ndescs = 0;
            uint64_t io_bytes = 0;

            // INT32 field (O_ORDERDATE, fi=0)
            {
                size_t gfi = o_fi[0];
                bool is_compressed = (field_comp_methods[gfi] != CompressionMethod::NONE);
                if (ti.use_selective) {
                    size_t io_offset = 0;
                    for (size_t idx = 0; idx < ti.active_i32.size(); idx++) {
                        uint32_t pg = ti.active_i32[idx];
                        size_t slot = pg - ti.p_lo;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_io_o[buf][0]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data_o[buf][0]) + slot * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_o[buf].h_descs[0][ndescs++] = desc;
                    }
                } else {
                    size_t io_offset = 0;
                    for (size_t j = 0; j < ti.tile_np; j++) {
                        uint64_t pg = ti.p_lo + j;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_io_o[buf][0]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data_o[buf][0]) + j * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_o[buf].h_descs[0][ndescs++] = desc;
                    }
                }
            }

            // INT64 fields (O_ORDERKEY, O_CUSTKEY: fi=1,2)
            for (int fi = 1; fi < 3; fi++) {
                size_t gfi = o_fi[fi];
                bool is_compressed = (field_comp_methods[gfi] != CompressionMethod::NONE);
                if (ti.use_sel_i64) {
                    size_t io_offset = 0;
                    for (size_t idx = 0; idx < ti.i64_active.size(); idx++) {
                        uint32_t pg = ti.i64_active[idx];
                        size_t slot = pg - ti.i64_start;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_io_o[buf][fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data_o[buf][fi]) + slot * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_o[buf].h_descs[0][ndescs++] = desc;
                    }
                } else {
                    size_t io_offset = 0;
                    for (size_t j = 0; j < ti.i64_np; j++) {
                        uint64_t pg = ti.i64_start + j;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_io_o[buf][fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data_o[buf][fi]) + j * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_o[buf].h_descs[0][ndescs++] = desc;
                    }
                }
            }
            return {ndescs, io_bytes};
        };

        // Lambda: decompress + paged probe_build for ORDERS tile
        auto run_o_tile_decomp_probe = [&](size_t tile_idx, int buf) {
            auto &ti = o_tiles[tile_idx];

            if (any_compressed) {
                // Decompress INT32 field (O_ORDERDATE, fi=0)
                {
                    size_t gfi = o_fi[0];
                    if (field_comp_methods[gfi] != CompressionMethod::NONE) {
                        size_t slot_base = tile_idx * TILE_PAGES_O;
                        size_t decomp_count = 0;
                        if (ti.use_selective) {
                            for (size_t idx = 0; idx < ti.active_i32.size(); idx++) {
                                uint32_t pg = ti.active_i32[idx];
                                size_t local_pg = pg - ti.p_lo;
                                uint32_t comp_sz = h_comp_sizes[gfi][pg];
                                if (comp_sz < page_size) {
                                    pb_comp_ptrs_o[0][slot_base + decomp_count] =
                                        static_cast<char*>(tile_io_o[buf][0]) + idx * page_size;
                                    pb_comp_sizes_o[0][slot_base + decomp_count] = comp_sz;
                                    pb_decomp_ptrs_o[0][slot_base + decomp_count] =
                                        static_cast<char*>(tile_data_o[buf][0]) + local_pg * page_size;
                                    pb_decomp_sizes_o[0][slot_base + decomp_count] = page_size;
                                    decomp_count++;
                                } else {
                                    CUDA_CHECK(cudaMemcpyAsync(
                                        static_cast<char*>(tile_data_o[buf][0]) + local_pg * page_size,
                                        static_cast<char*>(tile_io_o[buf][0]) + idx * page_size,
                                        page_size, cudaMemcpyDeviceToDevice, stream_comp_o));
                                }
                            }
                        } else {
                            for (size_t j = 0; j < ti.tile_np; j++) {
                                uint64_t pg = ti.p_lo + j;
                                uint32_t comp_sz = h_comp_sizes[gfi][pg];
                                if (comp_sz < page_size) {
                                    pb_comp_ptrs_o[0][slot_base + decomp_count] =
                                        static_cast<char*>(tile_io_o[buf][0]) + j * page_size;
                                    pb_comp_sizes_o[0][slot_base + decomp_count] = comp_sz;
                                    pb_decomp_ptrs_o[0][slot_base + decomp_count] =
                                        static_cast<char*>(tile_data_o[buf][0]) + j * page_size;
                                    pb_decomp_sizes_o[0][slot_base + decomp_count] = page_size;
                                    decomp_count++;
                                } else {
                                    CUDA_CHECK(cudaMemcpyAsync(
                                        static_cast<char*>(tile_data_o[buf][0]) + j * page_size,
                                        static_cast<char*>(tile_io_o[buf][0]) + j * page_size,
                                        page_size, cudaMemcpyDeviceToDevice, stream_comp_o));
                                }
                            }
                        }
                        if (decomp_count > 0) {
                            CUDA_CHECK(cudaMemcpyAsync(nvctx_o[buf][0].d_comp_ptrs,
                                pb_comp_ptrs_o[0] + slot_base,
                                decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp_o));
                            CUDA_CHECK(cudaMemcpyAsync(nvctx_o[buf][0].d_decomp_ptrs,
                                pb_decomp_ptrs_o[0] + slot_base,
                                decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp_o));
                            CUDA_CHECK(cudaMemcpyAsync(nvctx_o[buf][0].d_comp_sizes,
                                pb_comp_sizes_o[0] + slot_base,
                                decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp_o));
                            CUDA_CHECK(cudaMemcpyAsync(nvctx_o[buf][0].d_decomp_sizes,
                                pb_decomp_sizes_o[0] + slot_base,
                                decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp_o));
                            Gidp::nvcomp_decompctx_run(field_comp_methods[gfi], nvctx_o[buf][0],
                                                       decomp_count, page_size, stream_comp_o,
                                                       /*do_sync=*/false, /*skip_h2d=*/true);
                            s_kernel_launches++;
                        }
                    }
                }

                // Decompress INT64 fields (O_ORDERKEY, O_CUSTKEY: fi=1,2)
                for (int fi = 1; fi < 3; fi++) {
                    size_t gfi = o_fi[fi];
                    if (field_comp_methods[gfi] != CompressionMethod::NONE) {
                        size_t slot_base = tile_idx * o_i64_tile_npages_max;
                        size_t decomp_count = 0;
                        if (ti.use_sel_i64) {
                            for (size_t idx = 0; idx < ti.i64_active.size(); idx++) {
                                uint32_t pg = ti.i64_active[idx];
                                size_t slot = pg - ti.i64_start;
                                uint32_t comp_sz = h_comp_sizes[gfi][pg];
                                if (comp_sz < page_size) {
                                    pb_comp_ptrs_o[fi][slot_base + decomp_count] =
                                        static_cast<char*>(tile_io_o[buf][fi]) + idx * page_size;
                                    pb_comp_sizes_o[fi][slot_base + decomp_count] = comp_sz;
                                    pb_decomp_ptrs_o[fi][slot_base + decomp_count] =
                                        static_cast<char*>(tile_data_o[buf][fi]) + slot * page_size;
                                    pb_decomp_sizes_o[fi][slot_base + decomp_count] = page_size;
                                    decomp_count++;
                                } else {
                                    CUDA_CHECK(cudaMemcpyAsync(
                                        static_cast<char*>(tile_data_o[buf][fi]) + slot * page_size,
                                        static_cast<char*>(tile_io_o[buf][fi]) + idx * page_size,
                                        page_size, cudaMemcpyDeviceToDevice, stream_comp_o));
                                }
                            }
                        } else {
                            for (size_t j = 0; j < ti.i64_np; j++) {
                                uint64_t pg = ti.i64_start + j;
                                uint32_t comp_sz = h_comp_sizes[gfi][pg];
                                if (comp_sz < page_size) {
                                    pb_comp_ptrs_o[fi][slot_base + decomp_count] =
                                        static_cast<char*>(tile_io_o[buf][fi]) + j * page_size;
                                    pb_comp_sizes_o[fi][slot_base + decomp_count] = comp_sz;
                                    pb_decomp_ptrs_o[fi][slot_base + decomp_count] =
                                        static_cast<char*>(tile_data_o[buf][fi]) + j * page_size;
                                    pb_decomp_sizes_o[fi][slot_base + decomp_count] = page_size;
                                    decomp_count++;
                                } else {
                                    CUDA_CHECK(cudaMemcpyAsync(
                                        static_cast<char*>(tile_data_o[buf][fi]) + j * page_size,
                                        static_cast<char*>(tile_io_o[buf][fi]) + j * page_size,
                                        page_size, cudaMemcpyDeviceToDevice, stream_comp_o));
                                }
                            }
                        }
                        if (decomp_count > 0) {
                            CUDA_CHECK(cudaMemcpyAsync(nvctx_o[buf][fi].d_comp_ptrs,
                                pb_comp_ptrs_o[fi] + slot_base,
                                decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp_o));
                            CUDA_CHECK(cudaMemcpyAsync(nvctx_o[buf][fi].d_decomp_ptrs,
                                pb_decomp_ptrs_o[fi] + slot_base,
                                decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp_o));
                            CUDA_CHECK(cudaMemcpyAsync(nvctx_o[buf][fi].d_comp_sizes,
                                pb_comp_sizes_o[fi] + slot_base,
                                decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp_o));
                            CUDA_CHECK(cudaMemcpyAsync(nvctx_o[buf][fi].d_decomp_sizes,
                                pb_decomp_sizes_o[fi] + slot_base,
                                decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp_o));
                            Gidp::nvcomp_decompctx_run(field_comp_methods[gfi], nvctx_o[buf][fi],
                                                       decomp_count, page_size, stream_comp_o,
                                                       /*do_sync=*/false, /*skip_h2d=*/true);
                            s_kernel_launches++;
                        }
                    }
                }
            }

            // Paged probe_build: upload active pages + prefix sums, then call kernel
            h_o_apg.clear();
            if (ti.use_selective) {
                for (auto pg : ti.active_i32) h_o_apg.push_back(pg - ti.p_lo);
            } else {
                for (size_t j = 0; j < ti.tile_np; j++) h_o_apg.push_back((uint32_t)j);
            }
            uint32_t num_active = (uint32_t)h_o_apg.size();
            CUDA_CHECK(cudaMemcpyAsync(d_o_active_pg[buf], h_o_apg.data(),
                num_active * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_comp_o));

            for (size_t pg = 0; pg <= ti.tile_np; pg++)
                h_o_ps_ref[pg] = ps_o_i32_full[ti.p_lo + pg] - ps_o_i64_full[ti.i64_start];
            CUDA_CHECK(cudaMemcpyAsync(d_o_ps_ref[buf], h_o_ps_ref.data(),
                (ti.tile_np + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_comp_o));

            for (size_t pg = 0; pg <= ti.i64_np; pg++)
                h_o_ps_i64[pg] = ps_o_i64_full[ti.i64_start + pg] - ps_o_i64_full[ti.i64_start];
            CUDA_CHECK(cudaMemcpyAsync(d_o_ps_i64[buf], h_o_ps_i64.data(),
                (ti.i64_np + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_comp_o));

            uint32_t stride = (page_size - 12) / sizeof(int32_t);
            q5_build_orders_ht_paged(
                static_cast<const char*>(tile_data_o[buf][O_ORDERDATE_F]),
                static_cast<const char*>(tile_data_o[buf][O_ORDERKEY_F]),
                static_cast<const char*>(tile_data_o[buf][O_CUSTKEY_F]),
                d_o_active_pg[buf], num_active,
                (uint32_t)page_size, stride,
                d_o_ps_ref[buf], d_o_ps_i64[buf], (uint32_t)ti.i64_np,
                date_low, date_high,
                d_ht_cust_keys, d_ht_cust_values, ht_cust_mask,
                d_ht_ord_keys, d_ht_ord_values, ht_ord_mask,
                stream_comp_o);
            s_kernel_launches++;
        };

        // HT memset before pipeline
        CUDA_CHECK(cudaMemsetAsync(d_ht_ord_keys, 0xFF, ht_ord_cap * sizeof(uint64_t), stream_comp_o));
        CUDA_CHECK(cudaStreamSynchronize(stream_comp_o));

        // Pipeline loop
        if (num_o_tiles > 0) {
            {
                auto [ndescs, io_bytes] = build_o_tile_descs(0, 0);
                bam_bulk_read_async(io_ctx_o[0], ndescs, 0, stream_io_o[0]);
                s_kernel_launches++;
                CUDA_CHECK(cudaEventRecord(event_io_o[0], stream_io_o[0]));
                total_io_count += ndescs;
                total_io_bytes += io_bytes;
            }

            for (size_t pipe = 0; pipe < num_o_tiles; pipe++) {
                int buf = pipe % NPIPE;
                int next_buf = (pipe + 1) % NPIPE;

                // (b) Wait for IO, then decomp + paged probe_build
                CUDA_CHECK(cudaStreamWaitEvent(stream_comp_o, event_io_o[pipe]));
                run_o_tile_decomp_probe(pipe, buf);
                CUDA_CHECK(cudaEventRecord(event_comp_o[pipe], stream_comp_o));

                // (a) Launch IO for next tile (after comp event recorded)
                if (pipe + 1 < num_o_tiles) {
                    if (pipe + 1 >= (size_t)NPIPE) {
                        CUDA_CHECK(cudaStreamWaitEvent(stream_io_o[next_buf], event_comp_o[pipe + 1 - NPIPE]));
                        CUDA_CHECK(cudaEventSynchronize(io_ctx_o[next_buf].h2d_done[0]));
                    }
                    auto [ndescs, io_bytes] = build_o_tile_descs(pipe + 1, next_buf);
                    bam_bulk_read_async(io_ctx_o[next_buf], ndescs, 0, stream_io_o[next_buf]);
                    s_kernel_launches++;
                    CUDA_CHECK(cudaEventRecord(event_io_o[pipe + 1], stream_io_o[next_buf]));
                    total_io_count += ndescs;
                    total_io_bytes += io_bytes;
                }
            }

            CUDA_CHECK(cudaStreamSynchronize(stream_comp_o));
        }

        std::cout << "[Q5] ORDERS HT built (capacity=" << ht_ord_cap << ")" << std::endl;

        // Phase 3 pipeline resources freed after total_end
    }
    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 3 (ORDERS): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 4: LINEITEM — pipelined BaM IO + paged kernel probe
    // ═══════════════════════════════════════════════════════════════════
    CUDA_CHECK(cudaMemsetAsync(d_revenue, 0, 25 * sizeof(int64_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream)); stream = nullptr;

    {
        constexpr int NPIPE = 4;
        constexpr int LI_EXTPRICE_F = 0, LI_DISCOUNT_F = 1, LI_ORDERKEY_F = 2, LI_SUPPKEY_F = 3;
        constexpr int LI_NFIELDS = 4;
        const size_t li_fi[LI_NFIELDS] = { FI_L_EXTPRICE, FI_L_DISCOUNT, FI_L_ORDERKEY, FI_L_SUPPKEY };

        const size_t TILE_PAGES_P4 = Q5_TILE_PAGES_P4;
        const size_t l_i64_tile_npages_max_p4 = pre_l_i64_tile_npages_max_p4;

        // tile_data, io_buf_l, d_active_pg, d_ps_ref, d_ps_i64
        // are pre-allocated before the measurement point.
        std::vector<uint32_t> h_active_pg_buf(TILE_PAGES_P4);
        std::vector<uint64_t> h_ps_ref_buf(TILE_PAGES_P4 + 1);
        std::vector<uint64_t> h_ps_i64_buf(l_i64_tile_npages_max_p4 + 1);

        // Pre-compute tile info
        struct TileInfo {
            size_t p_lo, tile_np;
            uint64_t first_row, last_row, tile_nrows;
            std::vector<uint32_t> active_i32;
            bool use_selective;
            size_t i64_start, i64_end, i64_np;
            uint64_t i64_offset;
            std::vector<uint32_t> i64_active;
            bool use_sel_i64;
        };
        const size_t n_tiles_all = (npages_l_i32 + TILE_PAGES_P4 - 1) / TILE_PAGES_P4;
        std::vector<TileInfo> tiles;
        tiles.reserve(n_tiles_all);
        for (size_t p_lo = 0; p_lo < npages_l_i32; p_lo += TILE_PAGES_P4) {
            TileInfo ti;
            ti.p_lo = p_lo;
            ti.tile_np = std::min(TILE_PAGES_P4, (size_t)(npages_l_i32 - p_lo));
            ti.first_row = ps_l_i32_full[p_lo];
            ti.last_row = ps_l_i32_full[p_lo + ti.tile_np];
            ti.tile_nrows = ti.last_row - ti.first_row;
            if (ti.tile_nrows == 0) continue;

            for (size_t j = 0; j < ti.tile_np; j++)
                if (h_zm_li_mask[p_lo + j]) ti.active_i32.push_back((uint32_t)(p_lo + j));
            if (enable_zonemap && ti.active_i32.empty()) continue;
            ti.use_selective = enable_zonemap && ti.active_i32.size() < ti.tile_np;

            auto it_s = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), ti.first_row);
            ti.i64_start = (it_s == ps_l_i64_full.begin()) ? 0 : (size_t)(it_s - ps_l_i64_full.begin()) - 1;
            auto it_e = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), ti.last_row - 1);
            ti.i64_end = (size_t)(it_e - ps_l_i64_full.begin());
            ti.i64_np = ti.i64_end - ti.i64_start;
            ti.i64_offset = ti.first_row - ps_l_i64_full[ti.i64_start];

            if (enable_zonemap) {
                for (uint32_t pg = ti.i64_start; pg < ti.i64_end; pg++)
                    if (h_mask_li_i64[pg]) ti.i64_active.push_back(pg);
            }
            ti.use_sel_i64 = enable_zonemap && ti.i64_active.size() < ti.i64_np;
            tiles.push_back(std::move(ti));
        }
        const size_t num_tiles = tiles.size();
        std::cout << "[GIDP+BAM Q5] Pipeline: NPIPE=" << NPIPE
                  << " TILE_PAGES=" << TILE_PAGES_P4
                  << " tiles=" << num_tiles << "/" << n_tiles_all
                  << " (zone map pruned)" << std::endl;

        // Lambda: build IO descriptors for all 4 LINEITEM fields
        auto build_tile_descs = [&](size_t tile_idx, int buf) -> std::pair<uint32_t, uint64_t> {
            auto &ti = tiles[tile_idx];
            uint32_t ndescs = 0;
            uint64_t io_bytes = 0;

            // INT32 fields (L_EXTPRICE, L_DISCOUNT: fi=0,1)
            for (int fi = 0; fi < 2; fi++) {
                size_t gfi = li_fi[fi];
                bool is_compressed = (field_comp_methods[gfi] != CompressionMethod::NONE);
                if (ti.use_selective) {
                    size_t io_offset = 0;
                    for (size_t idx = 0; idx < ti.active_i32.size(); idx++) {
                        uint32_t pg = ti.active_i32[idx];
                        size_t slot = pg - ti.p_lo;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_buf_l[fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data[buf][fi]) + slot * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_bam[0].h_descs[0][ndescs++] = desc;
                    }
                } else {
                    size_t io_offset = 0;
                    for (size_t j = 0; j < ti.tile_np; j++) {
                        uint64_t pg = ti.p_lo + j;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_buf_l[fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data[buf][fi]) + j * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_bam[0].h_descs[0][ndescs++] = desc;
                    }
                }
            }

            // INT64 fields (L_ORDERKEY, L_SUPPKEY: fi=2,3)
            for (int fi = 2; fi < 4; fi++) {
                size_t gfi = li_fi[fi];
                bool is_compressed = (field_comp_methods[gfi] != CompressionMethod::NONE);
                if (ti.use_sel_i64) {
                    size_t io_offset = 0;
                    for (size_t idx = 0; idx < ti.i64_active.size(); idx++) {
                        uint32_t pg = ti.i64_active[idx];
                        size_t slot = pg - ti.i64_start;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_buf_l[fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data[buf][fi]) + slot * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_bam[0].h_descs[0][ndescs++] = desc;
                    }
                } else {
                    size_t io_offset = 0;
                    for (size_t j = 0; j < ti.i64_np; j++) {
                        uint64_t pg = ti.i64_start + j;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_buf_l[fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data[buf][fi]) + j * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_bam[0].h_descs[0][ndescs++] = desc;
                    }
                }
            }
            return {ndescs, io_bytes};
        };

        // Lambda: decompress all fields (io_buf_l → tile_data[buf])
        auto run_tile_decomp = [&](size_t tile_idx, int buf) {
            auto &ti = tiles[tile_idx];
            if (!any_compressed) return;

            // Decompress INT32 fields (L_EXTPRICE, L_DISCOUNT: fi=0,1)
            for (int fi = 0; fi < 2; fi++) {
                size_t gfi = li_fi[fi];
                if (field_comp_methods[gfi] == CompressionMethod::NONE) continue;
                size_t slot_base = tile_idx * TILE_PAGES_P4;
                size_t decomp_count = 0;
                if (ti.use_selective) {
                    for (size_t idx = 0; idx < ti.active_i32.size(); idx++) {
                        uint32_t pg = ti.active_i32[idx];
                        size_t local_pg = pg - ti.p_lo;
                        uint32_t comp_sz = h_comp_sizes[gfi][pg];
                        if (comp_sz < page_size) {
                            pb_comp_ptrs[fi][slot_base + decomp_count] =
                                static_cast<char*>(io_buf_l[fi]) + idx * page_size;
                            pb_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                            pb_decomp_ptrs[fi][slot_base + decomp_count] =
                                static_cast<char*>(tile_data[buf][fi]) + local_pg * page_size;
                            pb_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                            decomp_count++;
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                static_cast<char*>(tile_data[buf][fi]) + local_pg * page_size,
                                static_cast<char*>(io_buf_l[fi]) + idx * page_size,
                                page_size, cudaMemcpyDeviceToDevice, stream_comp));
                        }
                    }
                } else {
                    for (size_t j = 0; j < ti.tile_np; j++) {
                        uint64_t pg = ti.p_lo + j;
                        uint32_t comp_sz = h_comp_sizes[gfi][pg];
                        if (comp_sz < page_size) {
                            pb_comp_ptrs[fi][slot_base + decomp_count] =
                                static_cast<char*>(io_buf_l[fi]) + j * page_size;
                            pb_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                            pb_decomp_ptrs[fi][slot_base + decomp_count] =
                                static_cast<char*>(tile_data[buf][fi]) + j * page_size;
                            pb_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                            decomp_count++;
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                static_cast<char*>(tile_data[buf][fi]) + j * page_size,
                                static_cast<char*>(io_buf_l[fi]) + j * page_size,
                                page_size, cudaMemcpyDeviceToDevice, stream_comp));
                        }
                    }
                }
                if (decomp_count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_comp_ptrs,
                        pb_comp_ptrs[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_decomp_ptrs,
                        pb_decomp_ptrs[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_comp_sizes,
                        pb_comp_sizes[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_decomp_sizes,
                        pb_decomp_sizes[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                    Gidp::nvcomp_decompctx_run(field_comp_methods[gfi], nvctx_l[fi],
                                               decomp_count, page_size, stream_comp,
                                               /*do_sync=*/false, /*skip_h2d=*/true);
                    s_kernel_launches++;
                }
            }

            // Decompress INT64 fields (L_ORDERKEY, L_SUPPKEY: fi=2,3)
            for (int fi = 2; fi < 4; fi++) {
                size_t gfi = li_fi[fi];
                if (field_comp_methods[gfi] != CompressionMethod::NONE) {
                    size_t slot_base = tile_idx * l_i64_tile_npages_max_p4;
                    size_t decomp_count = 0;
                    if (ti.use_sel_i64) {
                        for (size_t idx = 0; idx < ti.i64_active.size(); idx++) {
                            uint32_t pg = ti.i64_active[idx];
                            size_t slot = pg - ti.i64_start;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            if (comp_sz < page_size) {
                                pb_comp_ptrs[fi][slot_base + decomp_count] =
                                    static_cast<char*>(io_buf_l[fi]) + idx * page_size;
                                pb_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                                pb_decomp_ptrs[fi][slot_base + decomp_count] =
                                    static_cast<char*>(tile_data[buf][fi]) + slot * page_size;
                                pb_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                                decomp_count++;
                            } else {
                                CUDA_CHECK(cudaMemcpyAsync(
                                    static_cast<char*>(tile_data[buf][fi]) + slot * page_size,
                                    static_cast<char*>(io_buf_l[fi]) + idx * page_size,
                                    page_size, cudaMemcpyDeviceToDevice, stream_comp));
                            }
                        }
                    } else {
                        for (size_t j = 0; j < ti.i64_np; j++) {
                            uint64_t pg = ti.i64_start + j;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            if (comp_sz < page_size) {
                                pb_comp_ptrs[fi][slot_base + decomp_count] =
                                    static_cast<char*>(io_buf_l[fi]) + j * page_size;
                                pb_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                                pb_decomp_ptrs[fi][slot_base + decomp_count] =
                                    static_cast<char*>(tile_data[buf][fi]) + j * page_size;
                                pb_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                                decomp_count++;
                            } else {
                                CUDA_CHECK(cudaMemcpyAsync(
                                    static_cast<char*>(tile_data[buf][fi]) + j * page_size,
                                    static_cast<char*>(io_buf_l[fi]) + j * page_size,
                                    page_size, cudaMemcpyDeviceToDevice, stream_comp));
                            }
                        }
                    }
                    if (decomp_count > 0) {
                        CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_comp_ptrs,
                            pb_comp_ptrs[fi] + slot_base,
                            decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                        CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_decomp_ptrs,
                            pb_decomp_ptrs[fi] + slot_base,
                            decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                        CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_comp_sizes,
                            pb_comp_sizes[fi] + slot_base,
                            decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                        CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_decomp_sizes,
                            pb_decomp_sizes[fi] + slot_base,
                            decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                        Gidp::nvcomp_decompctx_run(field_comp_methods[gfi], nvctx_l[fi],
                                                   decomp_count, page_size, stream_comp,
                                                   /*do_sync=*/false, /*skip_h2d=*/true);
                        s_kernel_launches++;
                    }
                }
            }
        };

        // Lambda: paged probe on stream_comp (tile_data[buf] only, io_buf_l not used)
        auto run_tile_probe = [&](size_t tile_idx, int buf) {
            auto &ti = tiles[tile_idx];
            {
                uint32_t num_active;
                if (ti.use_selective) {
                    num_active = (uint32_t)ti.active_i32.size();
                    for (uint32_t i = 0; i < num_active; i++)
                        h_active_pg_buf[i] = ti.active_i32[i] - (uint32_t)ti.p_lo;
                } else {
                    num_active = (uint32_t)ti.tile_np;
                    for (uint32_t i = 0; i < num_active; i++)
                        h_active_pg_buf[i] = i;
                }
                CUDA_CHECK(cudaMemcpyAsync(d_active_pg[buf], h_active_pg_buf.data(),
                    num_active * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_comp));

                uint64_t origin = ps_l_i64_full[ti.i64_start];
                for (size_t pg = 0; pg <= ti.tile_np; pg++)
                    h_ps_ref_buf[pg] = ps_l_i32_full[ti.p_lo + pg] - origin;
                CUDA_CHECK(cudaMemcpyAsync(d_ps_ref[buf], h_ps_ref_buf.data(),
                    (ti.tile_np + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_comp));

                for (size_t pg = 0; pg <= ti.i64_np; pg++)
                    h_ps_i64_buf[pg] = ps_l_i64_full[ti.i64_start + pg] - origin;
                CUDA_CHECK(cudaMemcpyAsync(d_ps_i64[buf], h_ps_i64_buf.data(),
                    (ti.i64_np + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_comp));

                uint32_t stride = (page_size - 12) / sizeof(int32_t);
                q5_lineitem_probe_paged(
                    static_cast<const char*>(tile_data[buf][LI_EXTPRICE_F]),
                    static_cast<const char*>(tile_data[buf][LI_DISCOUNT_F]),
                    static_cast<const char*>(tile_data[buf][LI_ORDERKEY_F]),
                    static_cast<const char*>(tile_data[buf][LI_SUPPKEY_F]),
                    d_active_pg[buf], num_active,
                    (uint32_t)page_size, stride,
                    d_ps_ref[buf], d_ps_i64[buf], (uint32_t)ti.i64_np,
                    d_ht_ord_keys, d_ht_ord_values, ht_ord_mask,
                    d_ht_supp_keys, d_ht_supp_values, ht_supp_mask,
                    d_revenue, stream_comp);
                s_kernel_launches++;
            }
        };

        // Pipeline loop: io_buf_l shared across NPIPE, tile_data[NPIPE] ring buffer
        for (size_t tile = 0; tile < num_tiles; tile++) {
            int buf = tile % NPIPE;
            // Wait for tile_data[buf] to be free (NPIPE tiles ago)
            if (tile >= (size_t)NPIPE)
                CUDA_CHECK(cudaStreamWaitEvent(stream_io_l, event_comp_vec[tile - NPIPE]));
            // Wait for io_buf_l to be free (previous tile's decomp)
            if (tile > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_io_l, ev_decomp_done));
                CUDA_CHECK(cudaEventSynchronize(io_ctx_bam[0].h2d_done[0]));
            }
            // IO all fields into io_buf_l (compressed) / tile_data[buf] (uncompressed)
            auto [ndescs, io_bytes] = build_tile_descs(tile, buf);
            bam_bulk_read_async(io_ctx_bam[0], ndescs, 0, stream_io_l);
            s_kernel_launches++;
            CUDA_CHECK(cudaEventRecord(ev_io_done, stream_io_l));
            total_io_count += ndescs;
            total_io_bytes += io_bytes;
            // Decomp: io_buf_l → tile_data[buf]
            CUDA_CHECK(cudaStreamWaitEvent(stream_comp, ev_io_done));
            run_tile_decomp(tile, buf);
            CUDA_CHECK(cudaEventRecord(ev_decomp_done, stream_comp));
            // Probe: tile_data[buf] only (io_buf_l is free after this point)
            run_tile_probe(tile, buf);
            CUDA_CHECK(cudaEventRecord(event_comp_vec[tile], stream_comp));
        }
        if (num_tiles > 0)
            CUDA_CHECK(cudaStreamSynchronize(stream_comp));

        // Phase 4 pipeline resources freed after total_end
    }
    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q5-TIMING] Phase 4 (LINEITEM): " << ms << " ms" << std::endl;
        phase_start = now;
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

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Deferred Phase 1-2 resource cleanup (moved from timing section) ──
    for (int i = 0; i < 2; i++) cudaFree(flat_pool[i]);
    mb_cuda_free(staging_data); staging_data = nullptr;
    mb_cuda_free(staging_io); staging_io = nullptr;
    cudaFree(d_batch_ps); d_batch_ps = nullptr;
    cudaFree(d_page_mask); d_page_mask = nullptr;
    if (any_compressed) { Gidp::nvcomp_decompctx_free(nvctx); nvctx = NvcompDecompCtx{}; }

    // ── Deferred Phase 3 pipeline resource cleanup ──
    {
        const size_t o_fi_cleanup[3] = { FI_O_ORDERDATE, FI_O_ORDERKEY, FI_O_CUSTKEY };
        if (any_compressed) {
            for (int fi = 0; fi < 3; fi++)
                if (field_comp_methods[o_fi_cleanup[fi]] != CompressionMethod::NONE)
                    Gidp::nvcomp_decompctx_free(nvctx_o[0][fi]);
            for (int fi = 0; fi < 3; fi++) {
                if (pb_comp_ptrs_o[fi])    cudaFreeHost(pb_comp_ptrs_o[fi]);
                if (pb_decomp_ptrs_o[fi])  cudaFreeHost(pb_decomp_ptrs_o[fi]);
                if (pb_comp_sizes_o[fi])   cudaFreeHost(pb_comp_sizes_o[fi]);
                if (pb_decomp_sizes_o[fi]) cudaFreeHost(pb_decomp_sizes_o[fi]);
            }
        }
    }

    // ── Deferred Phase 4 pipeline resource cleanup ──
    {
        const size_t li_fi_cleanup[4] = { FI_L_EXTPRICE, FI_L_DISCOUNT, FI_L_ORDERKEY, FI_L_SUPPKEY };
        if (any_compressed) {
            for (int fi = 0; fi < 4; fi++)
                if (field_comp_methods[li_fi_cleanup[fi]] != CompressionMethod::NONE)
                    Gidp::nvcomp_decompctx_free(nvctx_l[fi]);
            for (int fi = 0; fi < 4; fi++) {
                if (pb_comp_ptrs[fi])    cudaFreeHost(pb_comp_ptrs[fi]);
                if (pb_decomp_ptrs[fi])  cudaFreeHost(pb_decomp_ptrs[fi]);
                if (pb_comp_sizes[fi])   cudaFreeHost(pb_comp_sizes[fi]);
                if (pb_decomp_sizes[fi]) cudaFreeHost(pb_decomp_sizes[fi]);
            }
        }
    }

    // ── Cleanup pre-created streams and events ──
    for (size_t i = 0; i < q5_max_o_tiles; i++) {
        cudaEventDestroy(event_io_o[i]);
        cudaEventDestroy(event_comp_o[i]);
    }
    for (size_t i = 0; i < q5_max_l_tiles; i++)
        cudaEventDestroy(event_comp_vec[i]);
    cudaEventDestroy(ev_io_done);
    cudaEventDestroy(ev_decomp_done);
    cudaStreamDestroy(stream_io_o[0]);
    cudaStreamDestroy(stream_comp_o);
    cudaStreamDestroy(stream_io_l);
    cudaStreamDestroy(stream_comp);

    // ── Cleanup (pre-allocated tile buffers) ──
    for (int fi = 0; fi < 3; fi++) {
        mb_cuda_free(tile_data_o[0][fi]);
        if (tile_io_o[0][fi]) mb_cuda_free(tile_io_o[0][fi]);
    }
    cudaFree(d_o_active_pg[0]);
    cudaFree(d_o_ps_ref[0]);
    cudaFree(d_o_ps_i64[0]);
    for (int p = 0; p < Q5_BAM_NPIPE; p++) {
        for (int fi = 0; fi < 4; fi++)
            mb_cuda_free(tile_data[p][fi]);
        cudaFree(d_active_pg[p]);
        cudaFree(d_ps_ref[p]);
        cudaFree(d_ps_i64[p]);
    }
    for (int fi = 0; fi < 4; fi++)
        if (io_buf_l[fi]) mb_cuda_free(io_buf_l[fi]);

    // ── Cleanup zone map contexts ──
    if (enable_zonemap) {
        bam_zonemap_ctx_destroy(zm_ctx_ord);
        bam_zonemap_ctx_destroy(zm_ctx_li);
    } else {
        free(h_zm_ord_mask);
        free(h_zm_li_mask);
    }
    if (d_mask_ord_i64) cudaFree(d_mask_ord_i64);
    if (d_mask_li_i64) cudaFree(d_mask_li_i64);
    if (d_ps_o_i32_zm) cudaFree(d_ps_o_i32_zm);
    if (d_ps_o_i64_zm) cudaFree(d_ps_o_i64_zm);
    if (d_ps_l_i32_zm) cudaFree(d_ps_l_i32_zm);
    if (d_ps_l_i64_zm) cudaFree(d_ps_l_i64_zm);

    // ── Cleanup ──
    bam_bulk_read_ctx_destroy(io_ctx_small);
    for (int p = 0; p < Q5_BAM_NPIPE; p++)
        bam_bulk_read_ctx_destroy(io_ctx_bam[p]);
    cudaFree(d_ht_supp_keys);
    cudaFree(d_ht_supp_values);
    cudaFree(d_ht_cust_keys);
    cudaFree(d_ht_cust_values);
    cudaFree(d_ht_ord_keys);
    cudaFree(d_ht_ord_values);
    cudaFree(d_revenue);
    cudaFree(d_nationkey_to_idx);

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

    // Count only pages actually read (active pages) for total_pages
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
// TPC-H Q3 — GIDP+BAM execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMP (host-launched)
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

    const bool is_q3sel = (options.q3sel_selectivity > 0);
    const int sel_pct = is_q3sel ? options.q3sel_selectivity : 20;

    if (is_q3sel)
        std::cout << "=== TPCH Q3SEL (GIDP+BAM, sel=" << sel_pct << "%) ===" << std::endl;
    else
        std::cout << "=== TPCH Q3 (GIDP+BAM) ===" << std::endl;
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

    // ── 5. Read compression metadata ──
    std::vector<uint32_t> h_comp_sizes[NUM_Q3_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_Q3_FIELDS];

    bool any_compressed = false;

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

    // ── 7. GPU memory allocation (persistent structures only) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Tile metrics for LINEITEM streaming ──
    constexpr size_t BATCH_PAGES = 512;
    const uint64_t npages_l_orderkey = field_npages_arr[FI_L_ORDERKEY];  // INT64
    const uint64_t npages_l_i32 = field_npages_arr[FI_L_SHIPDATE];      // INT32

    std::vector<uint64_t> ps_l_i32_full(npages_l_i32 + 1);
    ps_l_i32_full[0] = 0;
    for (uint64_t i = 0; i < npages_l_i32; i++)
        ps_l_i32_full[i + 1] = h_prefix_sum[FI_L_SHIPDATE][i];

    std::vector<uint64_t> ps_l_i64_full(npages_l_orderkey + 1);
    ps_l_i64_full[0] = 0;
    for (uint64_t i = 0; i < npages_l_orderkey; i++)
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

    uint64_t tile_nrows_max_l = 0;
    size_t l_i64_tile_npages_max = 0;
    uint64_t l_i64_nrows_max = 0;
    for (size_t p_lo = 0; p_lo < npages_l_i32; p_lo += BATCH_PAGES) {
        size_t tile_np = std::min(BATCH_PAGES, npages_l_i32 - p_lo);
        uint64_t first_row = ps_l_i32_full[p_lo];
        uint64_t last_row = ps_l_i32_full[p_lo + tile_np];
        tile_nrows_max_l = std::max(tile_nrows_max_l, last_row - first_row);
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), first_row);
        size_t i64_s = (it_s == ps_l_i64_full.begin()) ? 0 : (size_t)(it_s - ps_l_i64_full.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_l_i64_full.begin());
        l_i64_tile_npages_max = std::max(l_i64_tile_npages_max, i64_e - i64_s);
        l_i64_nrows_max = std::max(l_i64_nrows_max, ps_l_i64_full[i64_e] - ps_l_i64_full[i64_s]);
    }
    if (tile_nrows_max_l == 0) tile_nrows_max_l = 1;
    if (l_i64_nrows_max == 0) l_i64_nrows_max = 1;

    const size_t max_staging_pages = std::max(BATCH_PAGES, l_i64_tile_npages_max);

    // nvCOMP decompression context (sized for max staging pages)
    NvcompDecompCtx nvctx{};
    if (any_compressed) {
        std::vector<FieldPageInfo> temp_fields(NUM_Q3_FIELDS);
        for (size_t fi = 0; fi < NUM_Q3_FIELDS; fi++)
            temp_fields[fi].compression_method = field_comp_methods[fi];
        Gidp::nvcomp_decompctx_alloc(nvctx, max_staging_pages, page_size, temp_fields);
    }

    // CUSTOMER hash set
    uint64_t est_building = is_q3sel
        ? std::max((uint64_t)1024, nrecs_customer * sel_pct / 100)
        : std::max((uint64_t)1024, nrecs_customer / 4);
    uint32_t custset_load = is_q3sel ? 1 : 2;
    uint32_t custset_cap = 1;
    while (custset_cap < est_building * custset_load) custset_cap <<= 1;
    uint32_t custset_mask = custset_cap - 1;
    uint64_t *d_custkey_set = nullptr;
    CUDA_CHECK(cudaMalloc(&d_custkey_set, custset_cap * sizeof(uint64_t)));

    // ORDERS hash table
    uint64_t est_orders_qual = is_q3sel
        ? std::max((uint64_t)1024, nrecs_orders * sel_pct / 100)
        : std::max((uint64_t)1024, nrecs_orders / 8);
    uint32_t orders_ht_cap = 1;
    while (orders_ht_cap < est_orders_qual * 2) orders_ht_cap <<= 1;

    {
        constexpr uint64_t GPU_MEM_BUDGET_Q3 = 40ULL * 1024 * 1024 * 1024;
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total_dummy);
        uint64_t total_used = gpu_ctrl_bytes + (gpu_free_before_app - gpu_free_now);
        if (total_used < GPU_MEM_BUDGET_Q3) {
            uint64_t remaining = GPU_MEM_BUDGET_Q3 - total_used;
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

    // Aggregation hash map
    uint32_t aggr_cap = orders_ht_cap;
    uint32_t aggr_mask = aggr_cap - 1;
    uint64_t *d_aggr_keys = nullptr;
    int64_t  *d_aggr_revenues = nullptr;
    CUDA_CHECK(cudaMalloc(&d_aggr_keys, aggr_cap * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_aggr_revenues, aggr_cap * sizeof(int64_t)));

    // Result arrays
    Q3ResultRow *d_results = nullptr;
    uint32_t *d_result_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, aggr_cap * sizeof(Q3ResultRow)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));

    // Pre-allocate CUB DeviceMergeSort temp buffer
    void *d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceMergeSort::SortKeys(nullptr, sort_temp_bytes,
        d_results, (int)aggr_cap, Q3ResultCmp{}, stream);
    CUDA_CHECK(cudaMalloc(&d_sort_temp, sort_temp_bytes));

    // Pre-allocate results buffer (pinned host memory for fast D2H)
    Q3ResultRow *results = nullptr;
    CUDA_CHECK(cudaMallocHost(&results, aggr_cap * sizeof(Q3ResultRow)));

    // ── 8. Helper lambdas ──
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    auto build_field_descs = [&](BamBulkReadDesc *descs, uint32_t &ndescs,
                                  size_t fi, void *data_dest, void *io_dest,
                                  uint64_t pg_start, uint64_t pg_count) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        size_t io_offset = 0;
        for (uint64_t p = 0; p < pg_count; p++) {
            uint64_t pg = pg_start + p;
            auto &desc = descs[ndescs];
            desc = {};
            if (is_compressed) {
                uint64_t byte_offset = h_comp_offsets[fi][pg];
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                desc.device = dev;
                desc.dest = static_cast<char*>(io_dest) + io_offset;
                desc.copy_bytes = comp_sz;
                io_offset += page_size;
            } else {
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                uint64_t local_pg_dev = page_id / n_devices;
                desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(data_dest) + p * page_size;
                desc.copy_bytes = page_size;
            }
            ndescs++;
        }
    };

    auto decompress_field = [&](size_t fi, void *io_buf_ptr, void *data_buf_ptr,
                                 uint64_t pg_start, uint64_t pg_count) {
        if (field_comp_methods[fi] == CompressionMethod::NONE) return;
        size_t decomp_count = 0;
        for (uint64_t p = 0; p < pg_count; p++) {
            uint64_t pg = pg_start + p;
            uint32_t comp_sz = h_comp_sizes[fi][pg];
            if (comp_sz < page_size) {
                nvctx.h_comp_ptrs[decomp_count] =
                    static_cast<char*>(io_buf_ptr) + p * page_size;
                nvctx.h_comp_sizes[decomp_count] = comp_sz;
                nvctx.h_decomp_ptrs[decomp_count] =
                    static_cast<char*>(data_buf_ptr) + p * page_size;
                nvctx.h_decomp_sizes[decomp_count] = page_size;
                decomp_count++;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(data_buf_ptr) + p * page_size,
                    static_cast<char*>(io_buf_ptr) + p * page_size,
                    page_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
        if (decomp_count > 0) {
            Gidp::nvcomp_decompctx_run(field_comp_methods[fi], nvctx,
                                       decomp_count, page_size, stream);
            s_kernel_launches++;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // ── 9. Streaming buffer allocation (all large allocations before total_start) ──
    void *staging_data = mb_cuda_alloc(max_staging_pages * page_size);
    void *staging_io = mb_cuda_alloc(max_staging_pages * page_size);

    uint64_t *d_batch_ps = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_ps, max_staging_pages * sizeof(uint64_t)));
    std::vector<uint64_t> bps(max_staging_pages);

    // Flat array for CUSTOMER C_CUSTKEY only (Phase 1)
    uint64_t *d_custkey_flat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_custkey_flat, nrecs_customer * sizeof(uint64_t)));

    // Zone map mask buffer (BATCH_PAGES bytes, reused per batch/tile)
    uint8_t *d_page_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_page_mask, BATCH_PAGES));

    // Persistent BaM I/O context for Phase 1 small-table reads
    BamBulkReadCtx io_ctx_small = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(sm_count),
        static_cast<uint32_t>(BATCH_PAGES));

    // I/O counters (declared before lambdas so batch helpers can accumulate)
    uint64_t total_io_count = 0, total_io_bytes = 0;

    // ── Batch helpers ──
    auto batch_read_field = [&](size_t fi, size_t pg_start, size_t pg_count) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        void *io_ptr = is_compressed ? staging_io : nullptr;
        uint32_t ndescs = 0;
        build_field_descs(io_ctx_small.h_descs[0], ndescs, fi, staging_data, io_ptr, pg_start, pg_count);
        bam_bulk_read_async(io_ctx_small, ndescs, 0, stream);
        s_kernel_launches++;
        total_io_count += ndescs;
        for (uint32_t i = 0; i < ndescs; i++)
            total_io_bytes += static_cast<uint64_t>(io_ctx_small.h_descs[0][i].nblocks) * 512;
        decompress_field(fi, io_ptr, staging_data, pg_start, pg_count);
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
        for (size_t pg = 0; pg < field_npages_arr[fi]; pg += BATCH_PAGES) {
            size_t bnp = std::min(BATCH_PAGES, (size_t)(field_npages_arr[fi] - pg));
            batch_read_field(fi, pg, bnp);
            uint64_t row_start, batch_nrecs;
            upload_batch_ps(h_ps, pg, bnp, row_start, batch_nrecs);
            q13_flatten_int64_pages_ps(
                static_cast<const char*>(staging_data), page_size, d_batch_ps,
                bnp, batch_nrecs, d_flat + row_start, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    auto batch_flatten_int32 = [&](size_t fi, const std::vector<uint64_t> &h_ps,
                                   uint64_t nrecs_total, uint64_t *d_flat) {
        for (size_t pg = 0; pg < field_npages_arr[fi]; pg += BATCH_PAGES) {
            size_t bnp = std::min(BATCH_PAGES, (size_t)(field_npages_arr[fi] - pg));
            batch_read_field(fi, pg, bnp);
            uint64_t row_start, batch_nrecs;
            upload_batch_ps(h_ps, pg, bnp, row_start, batch_nrecs);
            q13_flatten_int32_pages_ps(
                static_cast<const char*>(staging_data), page_size, d_batch_ps,
                bnp, batch_nrecs, d_flat + row_start, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // ── Selective batch read (reads only active pages at correct slot positions) ──
    auto batch_read_field_selective = [&](size_t fi, size_t batch_start,
                                          const std::vector<uint32_t> &active_pages) {
        bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
        auto* descs = io_ctx_small.h_descs[0];
        uint32_t ndescs = 0;
        for (uint32_t pg : active_pages) {
            size_t slot = pg - batch_start;
            auto &desc = descs[ndescs];
            desc = {};
            if (is_compressed) {
                uint64_t byte_offset = h_comp_offsets[fi][pg];
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_io) + slot * page_size;
                desc.copy_bytes = comp_sz;
            } else {
                uint64_t page_id = field_start_page_ids[fi] + pg;
                uint32_t dev = page_id % n_devices;
                uint64_t local_pg_dev = page_id / n_devices;
                desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                desc.nblocks = blocks_per_page;
                desc.device = dev;
                desc.dest = static_cast<char*>(staging_data) + slot * page_size;
                desc.copy_bytes = page_size;
            }
            ndescs++;
        }
        if (ndescs > 0) {
            bam_bulk_read_async(io_ctx_small, ndescs, 0, stream);
            s_kernel_launches++;
            total_io_count += ndescs;
            for (uint32_t i = 0; i < ndescs; i++)
                total_io_bytes += static_cast<uint64_t>(descs[i].nblocks) * 512;
        }
        if (is_compressed) {
            size_t decomp_count = 0;
            for (uint32_t pg : active_pages) {
                size_t slot = pg - batch_start;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                if (comp_sz < page_size) {
                    nvctx.h_comp_ptrs[decomp_count] =
                        static_cast<char*>(staging_io) + slot * page_size;
                    nvctx.h_comp_sizes[decomp_count] = comp_sz;
                    nvctx.h_decomp_ptrs[decomp_count] =
                        static_cast<char*>(staging_data) + slot * page_size;
                    nvctx.h_decomp_sizes[decomp_count] = page_size;
                    decomp_count++;
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<char*>(staging_data) + slot * page_size,
                        static_cast<char*>(staging_io) + slot * page_size,
                        page_size, cudaMemcpyDeviceToDevice, stream));
                }
            }
            if (decomp_count > 0) {
                Gidp::nvcomp_decompctx_run(field_comp_methods[fi], nvctx,
                                           decomp_count, page_size, stream);
                s_kernel_launches++;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    // ── Zone map metadata (GPU pruning done inside timing, Rule 6) ──
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

    // ORDERS stats metadata
    const size_t o_odate_field = TPCH::common::O_ORDERDATE;
    uint64_t zm_o_odate_nstats = 0, zm_o_odate_stats_start = 0, zm_o_odate_stats_npg = 0;
    if (!is_q3sel || !options.disable_other_filters) {
        zm_o_odate_nstats      = metadata.table_orders_nstats[o_odate_field];
        zm_o_odate_stats_start = metadata.table_orders_stats_start_page_ids[o_odate_field];
        zm_o_odate_stats_npg   = metadata.table_orders_stats_npages[o_odate_field];
    }

    const size_t sw_mktseg_idx = TPCH::common::OS_SIDEWAYS_C_MKTSEGMENT;
    uint64_t zm_o_mktseg_nstats      = metadata.table_orders_sideways_nstats[o_odate_field][sw_mktseg_idx];
    uint64_t zm_o_mktseg_stats_start = metadata.table_orders_sideways_stats_start_page_ids[o_odate_field][sw_mktseg_idx];
    uint64_t zm_o_mktseg_stats_npg   = metadata.table_orders_sideways_stats_npages[o_odate_field][sw_mktseg_idx];

    // LINEITEM stats metadata
    const size_t l_sd_field = TPCH::common::L_SHIPDATE;
    uint64_t zm_l_sd_nstats = 0, zm_l_sd_stats_start = 0, zm_l_sd_stats_npg = 0;
    if (!is_q3sel || !options.disable_other_filters) {
        zm_l_sd_nstats      = metadata.table_lineitem_nstats[l_sd_field];
        zm_l_sd_stats_start = metadata.table_lineitem_stats_start_page_ids[l_sd_field];
        zm_l_sd_stats_npg   = metadata.table_lineitem_stats_npages[l_sd_field];
    }

    const size_t li_sw_odate_idx = TPCH::common::LS_SIDEWAYS_O_ORDERDATE;
    uint64_t zm_l_odate_nstats = 0, zm_l_odate_stats_start = 0, zm_l_odate_stats_npg = 0;
    if (!is_q3sel || !options.disable_other_filters) {
        zm_l_odate_nstats      = metadata.table_lineitem_sideways_nstats[l_sd_field][li_sw_odate_idx];
        zm_l_odate_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[l_sd_field][li_sw_odate_idx];
        zm_l_odate_stats_npg   = metadata.table_lineitem_sideways_stats_npages[l_sd_field][li_sw_odate_idx];
    }

    const size_t li_sw_mktseg_idx = TPCH::common::LS_SIDEWAYS_C_MKTSEGMENT;
    uint64_t zm_l_mktseg_nstats      = metadata.table_lineitem_sideways_nstats[l_sd_field][li_sw_mktseg_idx];
    uint64_t zm_l_mktseg_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[l_sd_field][li_sw_mktseg_idx];
    uint64_t zm_l_mktseg_stats_npg   = metadata.table_lineitem_sideways_stats_npages[l_sd_field][li_sw_mktseg_idx];

    // BamZonemapCtx (borrows io_ctx_small page_cache, eval inside timing)
    uint8_t *h_zm_ord_mask = nullptr, *h_zm_li_mask = nullptr;
    BamZonemapCtx zm_ctx_ord{}, zm_ctx_li{};
    uint32_t zm_nreads_ord = 0, zm_npreds_ord = 0;
    uint32_t zm_nreads_li = 0, zm_npreds_li = 0;
    bool zm_valid_ord = false, zm_valid_li = false;

    if (enable_zonemap) {
        // ORDERS ctx (O_ORDERDATE + sideways C_MKTSEGMENT)
        zm_ctx_ord = bam_zonemap_ctx_create(
            io_ctx_small.d_ctrls, io_ctx_small.d_pc, io_ctx_small.pc_base,
            static_cast<uint32_t>(page_size), npages_o_i32);
        h_zm_ord_mask = zm_ctx_ord.h_mask;
        if (zm_o_odate_nstats > 0 && zm_o_odate_stats_npg > 0) {
            uint32_t offset = zm_nreads_ord;
            for (uint64_t j = 0; j < zm_o_odate_stats_npg; j++) {
                uint64_t pg_id = zm_o_odate_stats_start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                zm_ctx_ord.h_reads[zm_nreads_ord++] = {
                    ds.partition_start_lbas[dev] + local * blocks_per_page,
                    static_cast<uint32_t>(blocks_per_page), dev};
            }
            zm_ctx_ord.h_preds[zm_npreds_ord++] = {
                offset, zm_o_odate_nstats,
                INT32_MIN, (int32_t)19950314};
        }
        if (zm_o_mktseg_nstats > 0 && zm_o_mktseg_stats_npg > 0) {
            uint32_t offset = zm_nreads_ord;
            for (uint64_t j = 0; j < zm_o_mktseg_stats_npg; j++) {
                uint64_t pg_id = zm_o_mktseg_stats_start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                zm_ctx_ord.h_reads[zm_nreads_ord++] = {
                    ds.partition_start_lbas[dev] + local * blocks_per_page,
                    static_cast<uint32_t>(blocks_per_page), dev};
            }
            int32_t mlo = is_q3sel ? q3sel_mktseg_lo : (int32_t)1;
            int32_t mhi = is_q3sel ? q3sel_mktseg_hi : (int32_t)1;
            zm_ctx_ord.h_preds[zm_npreds_ord++] = {
                offset, zm_o_mktseg_nstats, mlo, mhi};
        }
        if (zm_npreds_ord > 0) zm_valid_ord = true;
        else memset(h_zm_ord_mask, 1, npages_o_i32);

        // LINEITEM ctx (L_SHIPDATE + sideways O_ORDERDATE + sideways C_MKTSEGMENT)
        zm_ctx_li = bam_zonemap_ctx_create(
            io_ctx_small.d_ctrls, io_ctx_small.d_pc, io_ctx_small.pc_base,
            static_cast<uint32_t>(page_size), npages_l_i32);
        h_zm_li_mask = zm_ctx_li.h_mask;
        if (zm_l_sd_nstats > 0 && zm_l_sd_stats_npg > 0) {
            uint32_t offset = zm_nreads_li;
            for (uint64_t j = 0; j < zm_l_sd_stats_npg; j++) {
                uint64_t pg_id = zm_l_sd_stats_start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                zm_ctx_li.h_reads[zm_nreads_li++] = {
                    ds.partition_start_lbas[dev] + local * blocks_per_page,
                    static_cast<uint32_t>(blocks_per_page), dev};
            }
            zm_ctx_li.h_preds[zm_npreds_li++] = {
                offset, zm_l_sd_nstats,
                (int32_t)19950316, INT32_MAX};
        }
        if (zm_l_odate_nstats > 0 && zm_l_odate_stats_npg > 0) {
            uint32_t offset = zm_nreads_li;
            for (uint64_t j = 0; j < zm_l_odate_stats_npg; j++) {
                uint64_t pg_id = zm_l_odate_stats_start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                zm_ctx_li.h_reads[zm_nreads_li++] = {
                    ds.partition_start_lbas[dev] + local * blocks_per_page,
                    static_cast<uint32_t>(blocks_per_page), dev};
            }
            zm_ctx_li.h_preds[zm_npreds_li++] = {
                offset, zm_l_odate_nstats,
                INT32_MIN, (int32_t)19950314};
        }
        if (zm_l_mktseg_nstats > 0 && zm_l_mktseg_stats_npg > 0) {
            uint32_t offset = zm_nreads_li;
            for (uint64_t j = 0; j < zm_l_mktseg_stats_npg; j++) {
                uint64_t pg_id = zm_l_mktseg_stats_start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                zm_ctx_li.h_reads[zm_nreads_li++] = {
                    ds.partition_start_lbas[dev] + local * blocks_per_page,
                    static_cast<uint32_t>(blocks_per_page), dev};
            }
            int32_t mlo = is_q3sel ? q3sel_mktseg_lo : (int32_t)1;
            int32_t mhi = is_q3sel ? q3sel_mktseg_hi : (int32_t)1;
            zm_ctx_li.h_preds[zm_npreds_li++] = {
                offset, zm_l_mktseg_nstats, mlo, mhi};
        }
        if (zm_npreds_li > 0) zm_valid_li = true;
        else memset(h_zm_li_mask, 1, npages_l_i32);
    } else {
        h_zm_ord_mask = static_cast<uint8_t *>(malloc(npages_o_i32));
        h_zm_li_mask = static_cast<uint8_t *>(malloc(npages_l_i32));
        memset(h_zm_ord_mask, 1, npages_o_i32);
        memset(h_zm_li_mask, 1, npages_l_i32);
    }

    // ════════════════════════════════════════════
    // Pre-create BaM I/O contexts (outside timing)
    // ════════════════════════════════════════════
    constexpr size_t Q3_TILE_PAGES_O  = 1728;
    constexpr size_t Q3_TILE_PAGES_P3_MAX = 1024;

    // Pre-compute max INT64 tile pages for ORDERS
    size_t pre_o_i64_tile_npages_max = 0;
    for (size_t p_lo = 0; p_lo < npages_o_i32; p_lo += Q3_TILE_PAGES_O) {
        size_t tile_np = std::min(Q3_TILE_PAGES_O, (size_t)(npages_o_i32 - p_lo));
        uint64_t first_row = ps_o_i32_full[p_lo];
        uint64_t last_row  = ps_o_i32_full[p_lo + tile_np];
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_o_i64_full.begin(), ps_o_i64_full.end(), first_row);
        size_t i64_s = (it_s == ps_o_i64_full.begin()) ? 0 : (size_t)(it_s - ps_o_i64_full.begin()) - 1;
        auto it_e = std::upper_bound(ps_o_i64_full.begin(), ps_o_i64_full.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_o_i64_full.begin());
        pre_o_i64_tile_npages_max = std::max(pre_o_i64_tile_npages_max, i64_e - i64_s);
    }

    // Pre-compute max INT64 tile pages for LINEITEM
    size_t pre_l_i64_tile_npages_max_p3 = 0;
    for (size_t p_lo = 0; p_lo < npages_l_i32; p_lo += Q3_TILE_PAGES_P3_MAX) {
        size_t tile_np = std::min(Q3_TILE_PAGES_P3_MAX, npages_l_i32 - p_lo);
        uint64_t first_row = ps_l_i32_full[p_lo];
        uint64_t last_row  = ps_l_i32_full[p_lo + tile_np];
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), first_row);
        size_t i64_s = (it_s == ps_l_i64_full.begin()) ? 0 : (size_t)(it_s - ps_l_i64_full.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_l_i64_full.begin());
        pre_l_i64_tile_npages_max_p3 = std::max(pre_l_i64_tile_npages_max_p3, i64_e - i64_s);
    }

    constexpr int Q3_BAM_NPIPE = 4;  // Phase 3 LINEITEM NPIPE
    const uint32_t io_blocks_bam = static_cast<uint32_t>(sm_count);
    // BaM I/O contexts (io_buf pattern: 1 for Phase 3 LINEITEM, Phase 2 uses io_staging_o)
    uint32_t max_descs_q3 = std::max({
        (uint32_t)Q3_TILE_PAGES_O, (uint32_t)pre_o_i64_tile_npages_max,
        3 * (uint32_t)Q3_TILE_PAGES_P3_MAX + (uint32_t)pre_l_i64_tile_npages_max_p3,
        (uint32_t)Q3_TILE_PAGES_P3_MAX, (uint32_t)pre_l_i64_tile_npages_max_p3});
    BamBulkReadCtx io_ctx_bam[2];
    for (int s = 0; s < 2; s++)
        io_ctx_bam[s] = bam_bulk_read_ctx_create(
            ctrl, static_cast<uint32_t>(page_size), io_blocks_bam, max_descs_q3);

    // Pre-allocate Phase 2 (ORDERS) tile data + paged-kernel buffers
    void *tile_data_o[1][4]{};
    uint32_t *d_o_active_pg[1]{};
    uint64_t *d_o_ps_ref[1]{};
    uint64_t *d_o_ps_i64[1]{};
    {
        const size_t o_fi_pre[4] = { FI_O_ORDERDATE, FI_O_SHIPPRIORITY, FI_O_ORDERKEY, FI_O_CUSTKEY };
        for (int fi = 0; fi < 2; fi++)
            tile_data_o[0][fi] = mb_cuda_alloc(Q3_TILE_PAGES_O * page_size);
        for (int fi = 2; fi < 4; fi++)
            tile_data_o[0][fi] = mb_cuda_alloc(pre_o_i64_tile_npages_max * page_size);
        CUDA_CHECK(cudaMalloc(&d_o_active_pg[0], Q3_TILE_PAGES_O * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_o_ps_ref[0], (Q3_TILE_PAGES_O + 1) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_o_ps_i64[0], (pre_o_i64_tile_npages_max + 1) * sizeof(uint64_t)));
    }

    // Shared IO staging for Phase 2 ORDERS (2 buffers)
    size_t io_staging_o_pages = std::max(Q3_TILE_PAGES_O, pre_o_i64_tile_npages_max);
    void *io_staging_o[2] = {nullptr, nullptr};
    {
        const size_t o_fi_pre[4] = { FI_O_ORDERDATE, FI_O_SHIPPRIORITY, FI_O_ORDERKEY, FI_O_CUSTKEY };
        bool any_o_compressed = false;
        for (int fi = 0; fi < 4; fi++)
            if (field_comp_methods[o_fi_pre[fi]] != CompressionMethod::NONE) any_o_compressed = true;
        if (any_o_compressed) {
            io_staging_o[0] = mb_cuda_alloc(io_staging_o_pages * page_size);
            io_staging_o[1] = mb_cuda_alloc(io_staging_o_pages * page_size);
        }
    }

    // Compute TILE_PAGES_P3 from 40 GiB total budget (ctrl + app)
    size_t Q3_TILE_PAGES_P3;
    {
        size_t gpu_free_now = 0, gt = 0;
        cudaMemGetInfo(&gpu_free_now, &gt);
        uint64_t total_used = gpu_ctrl_bytes + (gpu_free_before_app - gpu_free_now);
        constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
        uint64_t p3_budget = (total_used < GPU_MEM_BUDGET) ? GPU_MEM_BUDGET - total_used : 0;

        const size_t li_fi_tmp[4] = { FI_L_SHIPDATE, FI_L_EXTPRICE, FI_L_DISCOUNT, FI_L_ORDERKEY };
        int n_comp_i32 = 0, n_comp_i64 = 0;
        for (int fi = 0; fi < 3; fi++)
            if (field_comp_methods[li_fi_tmp[fi]] != CompressionMethod::NONE) n_comp_i32++;
        if (field_comp_methods[li_fi_tmp[3]] != CompressionMethod::NONE) n_comp_i64 = 1;

        double i64_ratio = (npages_l_i32 > 0) ? (double)npages_l_orderkey / npages_l_i32 : 2.0;
        i64_ratio = std::max(i64_ratio, 1.0) * 1.05;
        size_t bytes_per_tp = (size_t)(
            (Q3_BAM_NPIPE * 3 + n_comp_i32) * page_size +
            (Q3_BAM_NPIPE * 1 + n_comp_i64) * i64_ratio * page_size);
        Q3_TILE_PAGES_P3 = (bytes_per_tp > 0)
            ? std::min((size_t)Q3_TILE_PAGES_P3_MAX, (size_t)(p3_budget / bytes_per_tp))
            : Q3_TILE_PAGES_P3_MAX;
        if (Q3_TILE_PAGES_P3 == 0) Q3_TILE_PAGES_P3 = 1;
    }

    // Recompute i64 max for actual tile size
    pre_l_i64_tile_npages_max_p3 = 0;
    for (size_t p_lo = 0; p_lo < npages_l_i32; p_lo += Q3_TILE_PAGES_P3) {
        size_t tile_np = std::min(Q3_TILE_PAGES_P3, npages_l_i32 - p_lo);
        uint64_t first_row = ps_l_i32_full[p_lo];
        uint64_t last_row  = ps_l_i32_full[p_lo + tile_np];
        if (first_row == last_row) continue;
        auto it_s = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), first_row);
        size_t i64_s = (it_s == ps_l_i64_full.begin()) ? 0 : (size_t)(it_s - ps_l_i64_full.begin()) - 1;
        auto it_e = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), last_row - 1);
        size_t i64_e = (size_t)(it_e - ps_l_i64_full.begin());
        pre_l_i64_tile_npages_max_p3 = std::max(pre_l_i64_tile_npages_max_p3, i64_e - i64_s);
    }

    // Pre-allocate Phase 3 (LINEITEM) tile data + paged-kernel buffers
    void *tile_data[Q3_BAM_NPIPE][4]{};
    uint32_t *d_active_pg[Q3_BAM_NPIPE]{};
    uint64_t *d_ps_ref[Q3_BAM_NPIPE]{};
    uint64_t *d_ps_i64[Q3_BAM_NPIPE]{};
    {
        for (int p = 0; p < Q3_BAM_NPIPE; p++) {
            for (int fi = 0; fi < 3; fi++)
                tile_data[p][fi] = mb_cuda_alloc(Q3_TILE_PAGES_P3 * page_size);
            tile_data[p][3] = mb_cuda_alloc(pre_l_i64_tile_npages_max_p3 * page_size);
            CUDA_CHECK(cudaMalloc(&d_active_pg[p], Q3_TILE_PAGES_P3 * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_ps_ref[p], (Q3_TILE_PAGES_P3 + 1) * sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&d_ps_i64[p], (pre_l_i64_tile_npages_max_p3 + 1) * sizeof(uint64_t)));
        }
    }

    // Per-field IO buffers for Phase 3 LINEITEM (compressed fields only, shared across NPIPE)
    void *io_buf_l[4] = {};
    {
        const size_t li_fi_pre[4] = { FI_L_SHIPDATE, FI_L_EXTPRICE, FI_L_DISCOUNT, FI_L_ORDERKEY };
        for (int fi = 0; fi < 4; fi++) {
            if (field_comp_methods[li_fi_pre[fi]] != CompressionMethod::NONE) {
                size_t npg = (fi < 3) ? Q3_TILE_PAGES_P3 : pre_l_i64_tile_npages_max_p3;
                io_buf_l[fi] = mb_cuda_alloc(npg * page_size);
            }
        }
    }

    // Pre-allocate Phase 2+3 nvcomp decompression contexts and pinned host arrays
    NvcompDecompCtx nvctx_o[4]{};   // Phase 2 ORDERS: shared across NPIPE
    void   **pb_comp_ptrs_o[4]{};
    void   **pb_decomp_ptrs_o[4]{};
    size_t  *pb_comp_sizes_o[4]{};
    size_t  *pb_decomp_sizes_o[4]{};
    NvcompDecompCtx nvctx_l[4]{};   // Phase 3 LINEITEM: shared across NPIPE
    void   **pb_comp_ptrs[4]{};
    void   **pb_decomp_ptrs[4]{};
    size_t  *pb_comp_sizes[4]{};
    size_t  *pb_decomp_sizes[4]{};
    {
        const size_t o_fi_nvctx[4] = { FI_O_ORDERDATE, FI_O_SHIPPRIORITY, FI_O_ORDERKEY, FI_O_CUSTKEY };
        const size_t li_fi_nvctx[4] = { FI_L_SHIPDATE, FI_L_EXTPRICE, FI_L_DISCOUNT, FI_L_ORDERKEY };
        size_t n_o_tiles_max = (npages_o_i32 + Q3_TILE_PAGES_O - 1) / Q3_TILE_PAGES_O;
        size_t n_l_tiles_max = (npages_l_i32 + Q3_TILE_PAGES_P3 - 1) / Q3_TILE_PAGES_P3;
        if (any_compressed) {
            // Phase 2 nvctx + pb (shared across NPIPE — serialized on stream_comp)
            for (int fi = 0; fi < 4; fi++) {
                size_t gfi = o_fi_nvctx[fi];
                if (field_comp_methods[gfi] == CompressionMethod::NONE) continue;
                size_t max_pages = (fi < 2) ? (size_t)Q3_TILE_PAGES_O : pre_o_i64_tile_npages_max;
                std::vector<FieldPageInfo> tf(1);
                tf[0].compression_method = field_comp_methods[gfi];
                Gidp::nvcomp_decompctx_alloc(nvctx_o[fi], max_pages, page_size, tf);
                if (n_o_tiles_max > 0) {
                    size_t total_slots = n_o_tiles_max * max_pages;
                    CUDA_CHECK(cudaMallocHost(&pb_comp_ptrs_o[fi],    total_slots * sizeof(void*)));
                    CUDA_CHECK(cudaMallocHost(&pb_decomp_ptrs_o[fi],  total_slots * sizeof(void*)));
                    CUDA_CHECK(cudaMallocHost(&pb_comp_sizes_o[fi],   total_slots * sizeof(size_t)));
                    CUDA_CHECK(cudaMallocHost(&pb_decomp_sizes_o[fi], total_slots * sizeof(size_t)));
                }
            }
            // Phase 3 nvctx + pb (shared across NPIPE — serialized on stream_comp)
            for (int fi = 0; fi < 4; fi++) {
                size_t gfi = li_fi_nvctx[fi];
                if (field_comp_methods[gfi] == CompressionMethod::NONE) continue;
                size_t max_pages = (fi < 3) ? (size_t)Q3_TILE_PAGES_P3 : pre_l_i64_tile_npages_max_p3;
                std::vector<FieldPageInfo> tf(1);
                tf[0].compression_method = field_comp_methods[gfi];
                Gidp::nvcomp_decompctx_alloc(nvctx_l[fi], max_pages, page_size, tf);
                if (n_l_tiles_max > 0) {
                    size_t total_slots = n_l_tiles_max * max_pages;
                    CUDA_CHECK(cudaMallocHost(&pb_comp_ptrs[fi],    total_slots * sizeof(void*)));
                    CUDA_CHECK(cudaMallocHost(&pb_decomp_ptrs[fi],  total_slots * sizeof(void*)));
                    CUDA_CHECK(cudaMallocHost(&pb_comp_sizes[fi],   total_slots * sizeof(size_t)));
                    CUDA_CHECK(cudaMallocHost(&pb_decomp_sizes[fi], total_slots * sizeof(size_t)));
                }
            }
        }
    }

    // Measure GPU memory (includes BaM I/O contexts + tile buffers)
    uint64_t gpu_app_bytes = 0;
    {
        size_t gpu_free_after_app = 0;
        cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
        gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;
    }

    // Pre-create phase-specific CUDA streams and events (before total_start per Rule 4)
    cudaStream_t stream_io_o[2], stream_comp_o;
    for (int s = 0; s < 2; s++)
        CUDA_CHECK(cudaStreamCreate(&stream_io_o[s]));
    CUDA_CHECK(cudaStreamCreate(&stream_comp_o));
    cudaStream_t stream_io_l, stream_comp_p3;
    CUDA_CHECK(cudaStreamCreate(&stream_io_l));
    CUDA_CHECK(cudaStreamCreate(&stream_comp_p3));
    cudaEvent_t ev_io_done_l, ev_decomp_done_l;
    CUDA_CHECK(cudaEventCreate(&ev_io_done_l));
    CUDA_CHECK(cudaEventCreate(&ev_decomp_done_l));

    const size_t q3_max_o_tiles = (npages_o_i32 + Q3_TILE_PAGES_O - 1) / Q3_TILE_PAGES_O;
    const size_t q3_max_o_field_ev = q3_max_o_tiles * 4;  // O_NFIELDS=4
    std::vector<cudaEvent_t> ev_io_o(q3_max_o_field_ev);
    std::vector<cudaEvent_t> ev_decomp_o(q3_max_o_field_ev);
    std::vector<cudaEvent_t> event_comp_o(q3_max_o_tiles);
    for (size_t i = 0; i < q3_max_o_field_ev; i++) {
        CUDA_CHECK(cudaEventCreate(&ev_io_o[i]));
        CUDA_CHECK(cudaEventCreate(&ev_decomp_o[i]));
    }
    for (size_t i = 0; i < q3_max_o_tiles; i++)
        CUDA_CHECK(cudaEventCreate(&event_comp_o[i]));

    const size_t q3_max_l_tiles = (npages_l_i32 + Q3_TILE_PAGES_P3 - 1) / Q3_TILE_PAGES_P3;
    std::vector<cudaEvent_t> event_comp_vec(q3_max_l_tiles);
    for (size_t i = 0; i < q3_max_l_tiles; i++)
        CUDA_CHECK(cudaEventCreate(&event_comp_vec[i]));

    // GPU prefix sums for Phase 4 INT64 mask derivation (truncated format)
    uint64_t* d_ps_o_i32_zm = nullptr;
    uint64_t* d_ps_o_i64_zm = nullptr;
    uint64_t* d_ps_l_i32_zm = nullptr;
    uint64_t* d_ps_l_i64_zm = nullptr;
    if (enable_zonemap) {
        cudaMalloc(&d_ps_o_i32_zm, npages_o_i32 * sizeof(uint64_t));
        cudaMemcpy(d_ps_o_i32_zm, h_prefix_sum[FI_O_ORDERDATE].data(),
                   npages_o_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ps_o_i64_zm, npages_o_i64 * sizeof(uint64_t));
        cudaMemcpy(d_ps_o_i64_zm, h_prefix_sum[FI_O_ORDERKEY].data(),
                   npages_o_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ps_l_i32_zm, npages_l_i32 * sizeof(uint64_t));
        cudaMemcpy(d_ps_l_i32_zm, h_prefix_sum[FI_L_SHIPDATE].data(),
                   npages_l_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ps_l_i64_zm, npages_l_orderkey * sizeof(uint64_t));
        cudaMemcpy(d_ps_l_i64_zm, h_prefix_sum[FI_L_ORDERKEY].data(),
                   npages_l_orderkey * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // Zone map INT64 mask buffers (allocated outside timing)
    uint8_t* d_mask_ord_i64 = nullptr;
    uint8_t* d_mask_li_i64 = nullptr;
    if (enable_zonemap) {
        cudaMalloc(&d_mask_ord_i64, npages_o_i64);
        cudaMalloc(&d_mask_li_i64, npages_l_orderkey);
    }

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid_ord || zm_valid_li) {
        bam_pre_io(zm_ctx_ord.d_ctrls, zm_ctx_ord.d_pc, stream);
    }

    // ════════════════════════════════════════════
    // total_start
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    total_io_count = 0; total_io_bytes = 0;  // reset at timing start
    auto phase_start = total_start;

    // ═══════════════════════════════════════════════════════
    // Phase 1: CUSTOMER — batch flatten C_CUSTKEY, batch scan C_MKTSEGMENT
    // ═══════════════════════════════════════════════════════
    {
        auto p1_t0 = std::chrono::steady_clock::now();

        // 1a. Batch flatten C_CUSTKEY → d_custkey_flat
        batch_flatten_int64(FI_C_CUSTKEY, h_prefix_sum[FI_C_CUSTKEY], nrecs_customer, d_custkey_flat);

        auto p1_t1 = std::chrono::steady_clock::now();

        // 1b. Batch scan C_MKTSEGMENT → custkey_set
        CUDA_CHECK(cudaMemsetAsync(d_custkey_set, 0xFF, custset_cap * sizeof(uint64_t), stream));
        constexpr uint32_t CHAR_MKTSEG_PADDED_LEN = 12;

        static constexpr uint64_t Q3SEL_SEGMENTS[5] = {
            0x474E49444C495542ULL, // BUILDING
            0x49424F4D4F545541ULL, // AUTOMOBILE
            0x525554494E525546ULL, // FURNITURE
            0x52454E494843414DULL, // MACHINERY
            0x4C4F484553554F48ULL, // HOUSEHOLD
        };
        uint32_t q3sel_num_seg = 0;
        if (is_q3sel) {
            if (sel_pct >= 100) q3sel_num_seg = 0;
            else q3sel_num_seg = (uint32_t)(sel_pct / 20);
            if (q3sel_num_seg == 0 && sel_pct > 0 && sel_pct < 100) q3sel_num_seg = 1;
        }

        for (size_t pg = 0; pg < field_npages_arr[FI_C_MKTSEGMENT]; pg += BATCH_PAGES) {
            size_t bnp = std::min(BATCH_PAGES, (size_t)(field_npages_arr[FI_C_MKTSEGMENT] - pg));
            batch_read_field(FI_C_MKTSEGMENT, pg, bnp);
            uint64_t row_start, batch_nrecs;
            upload_batch_ps(h_prefix_sum[FI_C_MKTSEGMENT], pg, bnp, row_start, batch_nrecs);
            if (is_q3sel) {
                q3sel_customer_scan(
                    static_cast<const char*>(staging_data),
                    d_batch_ps, (uint32_t)bnp,
                    page_size, CHAR_MKTSEG_PADDED_LEN,
                    d_custkey_flat + row_start, batch_nrecs,
                    d_custkey_set, custset_mask,
                    q3sel_num_seg, Q3SEL_SEGMENTS, stream);
            } else {
                q3_customer_scan(
                    static_cast<const char*>(staging_data),
                    d_batch_ps, (uint32_t)bnp,
                    page_size, CHAR_MKTSEG_PADDED_LEN,
                    d_custkey_flat + row_start, batch_nrecs,
                    d_custkey_set, custset_mask, stream);
            }
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        auto p1_t2 = std::chrono::steady_clock::now();
        std::cout << "[Q3-P1-DETAIL] C_CUSTKEY flatten: "
                  << std::chrono::duration<double, std::milli>(p1_t1 - p1_t0).count() << " ms"
                  << ", C_MKTSEGMENT scan: "
                  << std::chrono::duration<double, std::milli>(p1_t2 - p1_t1).count() << " ms"
                  << std::endl;
        std::cout << "[Q3] CUSTOMER hash set built (capacity=" << custset_cap << ")" << std::endl;
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q3-TIMING] Phase 1 (CUSTOMER): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ── Zonemap pruning on GPU (fused IO + eval + INT64 mask derivation) ──
    std::vector<uint8_t> h_mask_ord_i64(npages_o_i64, 1);
    std::vector<uint8_t> h_mask_li_i64(npages_l_orderkey, 1);
    if (enable_zonemap) {
        if (zm_valid_ord) {
            zm_ctx_ord.d_ps_i32   = d_ps_o_i32_zm;
            zm_ctx_ord.d_ps_i64   = d_ps_o_i64_zm;
            zm_ctx_ord.d_mask_i64 = d_mask_ord_i64;
            zm_ctx_ord.npages_i64 = static_cast<uint32_t>(npages_o_i64);
            bam_zonemap_eval_async(zm_ctx_ord, npages_o_i32, zm_nreads_ord, zm_npreds_ord, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            s_kernel_launches++;
            cudaMemcpy(h_mask_ord_i64.data(), d_mask_ord_i64, npages_o_i64, cudaMemcpyDeviceToHost);
        }
        if (zm_valid_li) {
            zm_ctx_li.d_ps_i32   = d_ps_l_i32_zm;
            zm_ctx_li.d_ps_i64   = d_ps_l_i64_zm;
            zm_ctx_li.d_mask_i64 = d_mask_li_i64;
            zm_ctx_li.npages_i64 = static_cast<uint32_t>(npages_l_orderkey);
            bam_zonemap_eval_async(zm_ctx_li, npages_l_i32, zm_nreads_li, zm_npreds_li, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            s_kernel_launches++;
            cudaMemcpy(h_mask_li_i64.data(), d_mask_li_i64, npages_l_orderkey, cudaMemcpyDeviceToHost);
        }

        uint32_t ord_active = 0, li_active = 0;
        for (size_t i = 0; i < npages_o_i32; i++) if (h_zm_ord_mask[i]) ord_active++;
        for (size_t i = 0; i < npages_l_i32; i++) if (h_zm_li_mask[i]) li_active++;
        std::cout << "[Q3-ZONEMAP] ORDERS pruning: active=" << ord_active << "/" << npages_o_i32 << std::endl;
        std::cout << "[Q3-ZONEMAP] LINEITEM pruning: active=" << li_active << "/" << npages_l_i32 << std::endl;
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q3-TIMING] Zonemap pruning: " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ═══════════════════════════════════════════════════════════
    // Phase 2: ORDERS — pipelined tile loop (NPIPE=4, multi-stream IO)
    // ═══════════════════════════════════════════════════════════
    {
        constexpr int NPIPE = 1;  // single tile → no pipeline needed
        constexpr int O_ORDERDATE_F = 0, O_SHIPPRIORITY_F = 1, O_ORDERKEY_F = 2, O_CUSTKEY_F = 3;
        constexpr int O_NFIELDS = 4;
        const size_t o_fi[O_NFIELDS] = { FI_O_ORDERDATE, FI_O_SHIPPRIORITY, FI_O_ORDERKEY, FI_O_CUSTKEY };

        constexpr size_t TILE_PAGES_O = Q3_TILE_PAGES_O;
        const size_t o_i64_tile_npages_max = pre_o_i64_tile_npages_max;

        // tile_data_o, tile_io_o, d_o_active_pg, d_o_ps_ref, d_o_ps_i64
        // are pre-allocated before the measurement point.
        std::vector<uint32_t> h_o_apg;
        std::vector<uint64_t> h_o_ps_ref(TILE_PAGES_O + 1);
        std::vector<uint64_t> h_o_ps_i64(o_i64_tile_npages_max + 1);

        // nvctx_o + pb arrays are pre-allocated before total_start

        // Alias to pre-created shared BaM I/O contexts
        BamBulkReadCtx *io_ctx_o = io_ctx_bam;

        // stream_io_o, stream_comp_o pre-created before total_start

        // Pre-compute tile info + filter empty tiles
        struct OTileInfo {
            size_t p_lo, tile_np;
            uint64_t first_row, last_row, tile_nrows;
            std::vector<uint32_t> active_i32;
            bool use_selective;
            std::vector<uint8_t> h_mask;
            size_t i64_start, i64_end, i64_np;
            uint64_t i64_offset;
            std::vector<uint32_t> i64_active;
            bool use_sel_i64;
        };
        const size_t n_o_tiles_all = (npages_o_i32 + TILE_PAGES_O - 1) / TILE_PAGES_O;
        std::vector<OTileInfo> o_tiles;
        o_tiles.reserve(n_o_tiles_all);
        for (size_t p_lo = 0; p_lo < npages_o_i32; p_lo += TILE_PAGES_O) {
            OTileInfo ti;
            ti.p_lo = p_lo;
            ti.tile_np = std::min(TILE_PAGES_O, (size_t)(npages_o_i32 - p_lo));
            ti.first_row = ps_o_i32_full[p_lo];
            ti.last_row  = ps_o_i32_full[p_lo + ti.tile_np];
            ti.tile_nrows = ti.last_row - ti.first_row;
            if (ti.tile_nrows == 0) continue;

            for (size_t j = 0; j < ti.tile_np; j++)
                if (h_zm_ord_mask[p_lo + j]) ti.active_i32.push_back((uint32_t)(p_lo + j));
            if (enable_zonemap && ti.active_i32.empty()) continue;
            ti.use_selective = enable_zonemap && ti.active_i32.size() < ti.tile_np;
            if (ti.use_selective) {
                ti.h_mask.resize(ti.tile_np);
                for (size_t j = 0; j < ti.tile_np; j++)
                    ti.h_mask[j] = h_zm_ord_mask[p_lo + j];
            }

            auto it_s = std::upper_bound(ps_o_i64_full.begin(), ps_o_i64_full.end(), ti.first_row);
            ti.i64_start = (it_s == ps_o_i64_full.begin()) ? 0 : (size_t)(it_s - ps_o_i64_full.begin()) - 1;
            auto it_e = std::upper_bound(ps_o_i64_full.begin(), ps_o_i64_full.end(), ti.last_row - 1);
            ti.i64_end = (size_t)(it_e - ps_o_i64_full.begin());
            ti.i64_np = ti.i64_end - ti.i64_start;
            ti.i64_offset = ti.first_row - ps_o_i64_full[ti.i64_start];

            if (enable_zonemap) {
                for (uint32_t pg = ti.i64_start; pg < ti.i64_end; pg++)
                    if (h_mask_ord_i64[pg]) ti.i64_active.push_back(pg);
            }
            ti.use_sel_i64 = enable_zonemap && ti.i64_active.size() < ti.i64_np;
            o_tiles.push_back(std::move(ti));
        }
        const size_t num_o_tiles = o_tiles.size();
        std::cout << "[GIDP+BAM Q3] ORDERS Pipeline: NPIPE=" << NPIPE
                  << " TILE_PAGES=" << TILE_PAGES_O
                  << " tiles=" << num_o_tiles << "/" << n_o_tiles_all
                  << " (zone map pruned)" << std::endl;

        // Lambda: build IO descriptors for ONE ORDERS field
        auto build_o_field_descs = [&](size_t tile_idx, int fi, int buf, int s) -> std::pair<uint32_t, uint64_t> {
            auto &ti = o_tiles[tile_idx];
            uint32_t ndescs = 0;
            uint64_t io_bytes = 0;

            if (fi < 2) {
                // INT32 field (O_ORDERDATE fi=0, O_SHIPPRIORITY fi=1)
                size_t gfi = o_fi[fi];
                bool is_compressed = (field_comp_methods[gfi] != CompressionMethod::NONE);
                if (ti.use_selective) {
                    size_t io_offset = 0;
                    for (size_t idx = 0; idx < ti.active_i32.size(); idx++) {
                        uint32_t pg = ti.active_i32[idx];
                        size_t slot = pg - ti.p_lo;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_staging_o[s]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data_o[buf][fi]) + slot * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_o[s].h_descs[0][ndescs++] = desc;
                    }
                } else {
                    size_t io_offset = 0;
                    for (size_t j = 0; j < ti.tile_np; j++) {
                        uint64_t pg = ti.p_lo + j;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_staging_o[s]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data_o[buf][fi]) + j * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_o[s].h_descs[0][ndescs++] = desc;
                    }
                }
            } else {
                // INT64 fields (O_ORDERKEY fi=2, O_CUSTKEY fi=3)
                size_t gfi = o_fi[fi];
                bool is_compressed = (field_comp_methods[gfi] != CompressionMethod::NONE);
                if (ti.use_sel_i64) {
                    size_t io_offset = 0;
                    for (size_t idx = 0; idx < ti.i64_active.size(); idx++) {
                        uint32_t pg = ti.i64_active[idx];
                        size_t slot = pg - ti.i64_start;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_staging_o[s]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data_o[buf][fi]) + slot * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_o[s].h_descs[0][ndescs++] = desc;
                    }
                } else {
                    size_t io_offset = 0;
                    for (size_t j = 0; j < ti.i64_np; j++) {
                        uint64_t pg = ti.i64_start + j;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_staging_o[s]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data_o[buf][fi]) + j * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_o[s].h_descs[0][ndescs++] = desc;
                    }
                }
            }
            return {ndescs, io_bytes};
        };

        // Lambda: decompress ONE ORDERS field
        auto run_o_field_decomp = [&](size_t tile_idx, int fi, int buf, int s) {
            auto &ti = o_tiles[tile_idx];
            if (!any_compressed) return;

            size_t gfi = o_fi[fi];
            if (field_comp_methods[gfi] == CompressionMethod::NONE) return;

            if (fi < 2) {
                // Decompress INT32 field (O_ORDERDATE fi=0, O_SHIPPRIORITY fi=1)
                size_t slot_base = tile_idx * TILE_PAGES_O;
                size_t decomp_count = 0;
                if (ti.use_selective) {
                    for (size_t idx = 0; idx < ti.active_i32.size(); idx++) {
                        uint32_t pg = ti.active_i32[idx];
                        size_t local_pg = pg - ti.p_lo;
                        uint32_t comp_sz = h_comp_sizes[gfi][pg];
                        if (comp_sz < page_size) {
                            pb_comp_ptrs_o[fi][slot_base + decomp_count] =
                                static_cast<char*>(io_staging_o[s]) + idx * page_size;
                            pb_comp_sizes_o[fi][slot_base + decomp_count] = comp_sz;
                            pb_decomp_ptrs_o[fi][slot_base + decomp_count] =
                                static_cast<char*>(tile_data_o[buf][fi]) + local_pg * page_size;
                            pb_decomp_sizes_o[fi][slot_base + decomp_count] = page_size;
                            decomp_count++;
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                static_cast<char*>(tile_data_o[buf][fi]) + local_pg * page_size,
                                static_cast<char*>(io_staging_o[s]) + idx * page_size,
                                page_size, cudaMemcpyDeviceToDevice, stream_comp_o));
                        }
                    }
                } else {
                    for (size_t j = 0; j < ti.tile_np; j++) {
                        uint64_t pg = ti.p_lo + j;
                        uint32_t comp_sz = h_comp_sizes[gfi][pg];
                        if (comp_sz < page_size) {
                            pb_comp_ptrs_o[fi][slot_base + decomp_count] =
                                static_cast<char*>(io_staging_o[s]) + j * page_size;
                            pb_comp_sizes_o[fi][slot_base + decomp_count] = comp_sz;
                            pb_decomp_ptrs_o[fi][slot_base + decomp_count] =
                                static_cast<char*>(tile_data_o[buf][fi]) + j * page_size;
                            pb_decomp_sizes_o[fi][slot_base + decomp_count] = page_size;
                            decomp_count++;
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                static_cast<char*>(tile_data_o[buf][fi]) + j * page_size,
                                static_cast<char*>(io_staging_o[s]) + j * page_size,
                                page_size, cudaMemcpyDeviceToDevice, stream_comp_o));
                        }
                    }
                }
                if (decomp_count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_o[fi].d_comp_ptrs,
                        pb_comp_ptrs_o[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp_o));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_o[fi].d_decomp_ptrs,
                        pb_decomp_ptrs_o[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp_o));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_o[fi].d_comp_sizes,
                        pb_comp_sizes_o[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp_o));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_o[fi].d_decomp_sizes,
                        pb_decomp_sizes_o[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp_o));
                    Gidp::nvcomp_decompctx_run(field_comp_methods[gfi], nvctx_o[fi],
                                               decomp_count, page_size, stream_comp_o,
                                               /*do_sync=*/false, /*skip_h2d=*/true);
                    s_kernel_launches++;
                }
            } else {
                // Decompress INT64 field (O_ORDERKEY fi=2, O_CUSTKEY fi=3)
                size_t slot_base = tile_idx * o_i64_tile_npages_max;
                size_t decomp_count = 0;
                if (ti.use_sel_i64) {
                    for (size_t idx = 0; idx < ti.i64_active.size(); idx++) {
                        uint32_t pg = ti.i64_active[idx];
                        size_t slot = pg - ti.i64_start;
                        uint32_t comp_sz = h_comp_sizes[gfi][pg];
                        if (comp_sz < page_size) {
                            pb_comp_ptrs_o[fi][slot_base + decomp_count] =
                                static_cast<char*>(io_staging_o[s]) + idx * page_size;
                            pb_comp_sizes_o[fi][slot_base + decomp_count] = comp_sz;
                            pb_decomp_ptrs_o[fi][slot_base + decomp_count] =
                                static_cast<char*>(tile_data_o[buf][fi]) + slot * page_size;
                            pb_decomp_sizes_o[fi][slot_base + decomp_count] = page_size;
                            decomp_count++;
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                static_cast<char*>(tile_data_o[buf][fi]) + slot * page_size,
                                static_cast<char*>(io_staging_o[s]) + idx * page_size,
                                page_size, cudaMemcpyDeviceToDevice, stream_comp_o));
                        }
                    }
                } else {
                    for (size_t j = 0; j < ti.i64_np; j++) {
                        uint64_t pg = ti.i64_start + j;
                        uint32_t comp_sz = h_comp_sizes[gfi][pg];
                        if (comp_sz < page_size) {
                            pb_comp_ptrs_o[fi][slot_base + decomp_count] =
                                static_cast<char*>(io_staging_o[s]) + j * page_size;
                            pb_comp_sizes_o[fi][slot_base + decomp_count] = comp_sz;
                            pb_decomp_ptrs_o[fi][slot_base + decomp_count] =
                                static_cast<char*>(tile_data_o[buf][fi]) + j * page_size;
                            pb_decomp_sizes_o[fi][slot_base + decomp_count] = page_size;
                            decomp_count++;
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                static_cast<char*>(tile_data_o[buf][fi]) + j * page_size,
                                static_cast<char*>(io_staging_o[s]) + j * page_size,
                                page_size, cudaMemcpyDeviceToDevice, stream_comp_o));
                        }
                    }
                }
                if (decomp_count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_o[fi].d_comp_ptrs,
                        pb_comp_ptrs_o[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp_o));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_o[fi].d_decomp_ptrs,
                        pb_decomp_ptrs_o[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp_o));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_o[fi].d_comp_sizes,
                        pb_comp_sizes_o[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp_o));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_o[fi].d_decomp_sizes,
                        pb_decomp_sizes_o[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp_o));
                    Gidp::nvcomp_decompctx_run(field_comp_methods[gfi], nvctx_o[fi],
                                               decomp_count, page_size, stream_comp_o,
                                               /*do_sync=*/false, /*skip_h2d=*/true);
                    s_kernel_launches++;
                }
            }
        };

        // Lambda: paged probe_build for ORDERS tile
        auto run_o_probe = [&](size_t tile_idx, int buf) {
            auto &ti = o_tiles[tile_idx];

            // Paged probe_build: upload active pages + prefix sums, then call kernel
            h_o_apg.clear();
            if (ti.use_selective) {
                for (auto pg : ti.active_i32) h_o_apg.push_back(pg - ti.p_lo);
            } else {
                for (size_t j = 0; j < ti.tile_np; j++) h_o_apg.push_back((uint32_t)j);
            }
            uint32_t num_active = (uint32_t)h_o_apg.size();
            CUDA_CHECK(cudaMemcpyAsync(d_o_active_pg[buf], h_o_apg.data(),
                num_active * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_comp_o));

            // ps_ref[pg] = ps_o_i32_full[p_lo + pg] - ps_o_i64_full[i64_start]
            for (size_t pg = 0; pg <= ti.tile_np; pg++)
                h_o_ps_ref[pg] = ps_o_i32_full[ti.p_lo + pg] - ps_o_i64_full[ti.i64_start];
            CUDA_CHECK(cudaMemcpyAsync(d_o_ps_ref[buf], h_o_ps_ref.data(),
                (ti.tile_np + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_comp_o));

            // ps_i64[pg] = ps_o_i64_full[i64_start + pg] - ps_o_i64_full[i64_start]
            for (size_t pg = 0; pg <= ti.i64_np; pg++)
                h_o_ps_i64[pg] = ps_o_i64_full[ti.i64_start + pg] - ps_o_i64_full[ti.i64_start];
            CUDA_CHECK(cudaMemcpyAsync(d_o_ps_i64[buf], h_o_ps_i64.data(),
                (ti.i64_np + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_comp_o));

            uint32_t stride = (page_size - 12) / sizeof(int32_t);
            if (is_q3sel) {
                q3sel_orders_probe_build_paged(
                    static_cast<const char*>(tile_data_o[buf][O_ORDERDATE_F]),
                    static_cast<const char*>(tile_data_o[buf][O_ORDERKEY_F]),
                    static_cast<const char*>(tile_data_o[buf][O_CUSTKEY_F]),
                    static_cast<const char*>(tile_data_o[buf][O_SHIPPRIORITY_F]),
                    d_o_active_pg[buf], num_active,
                    (uint32_t)page_size, stride,
                    d_o_ps_ref[buf], d_o_ps_i64[buf], (uint32_t)ti.i64_np,
                    d_custkey_set, custset_mask,
                    d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                    options.disable_other_filters ? 0 : 19950315,
                    stream_comp_o);
            } else {
                q3_orders_probe_build_paged(
                    static_cast<const char*>(tile_data_o[buf][O_ORDERDATE_F]),
                    static_cast<const char*>(tile_data_o[buf][O_ORDERKEY_F]),
                    static_cast<const char*>(tile_data_o[buf][O_CUSTKEY_F]),
                    static_cast<const char*>(tile_data_o[buf][O_SHIPPRIORITY_F]),
                    d_o_active_pg[buf], num_active,
                    (uint32_t)page_size, stride,
                    d_o_ps_ref[buf], d_o_ps_i64[buf], (uint32_t)ti.i64_np,
                    d_custkey_set, custset_mask,
                    d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                    stream_comp_o);
            }
            s_kernel_launches++;
        };

        // HT memset before pipeline (paged kernel inserts per-tile)
        CUDA_CHECK(cudaMemsetAsync(d_orders_ht_keys, 0xFF, orders_ht_cap * sizeof(uint64_t), stream_comp_o));
        CUDA_CHECK(cudaStreamSynchronize(stream_comp_o));

        // Pipeline loop — events pre-created before total_start
        for (size_t tile = 0; tile < num_o_tiles; tile++) {
            int buf = tile % NPIPE;

            if (tile >= (size_t)NPIPE) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_io_o[0], event_comp_o[tile - NPIPE]));
                CUDA_CHECK(cudaStreamWaitEvent(stream_io_o[1], event_comp_o[tile - NPIPE]));
            }

            for (int fi = 0; fi < O_NFIELDS; fi++) {
                int s = fi % 2;
                size_t ev_idx = tile * O_NFIELDS + fi;

                if (fi >= 2) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream_io_o[s], ev_decomp_o[ev_idx - 2]));
                } else if (tile > 0) {
                    size_t last_fi = O_NFIELDS - 1;  // =3
                    if (last_fi % 2 != (size_t)s) last_fi--;
                    CUDA_CHECK(cudaStreamWaitEvent(stream_io_o[s],
                        ev_decomp_o[(tile - 1) * O_NFIELDS + last_fi]));
                }

                CUDA_CHECK(cudaEventSynchronize(io_ctx_o[s].h2d_done[0]));
                auto [ndescs, io_bytes] = build_o_field_descs(tile, fi, buf, s);
                bam_bulk_read_async(io_ctx_o[s], ndescs, 0, stream_io_o[s]);
                s_kernel_launches++;
                CUDA_CHECK(cudaEventRecord(ev_io_o[ev_idx], stream_io_o[s]));
                total_io_count += ndescs;
                total_io_bytes += io_bytes;

                CUDA_CHECK(cudaStreamWaitEvent(stream_comp_o, ev_io_o[ev_idx]));
                run_o_field_decomp(tile, fi, buf, s);
                CUDA_CHECK(cudaEventRecord(ev_decomp_o[ev_idx], stream_comp_o));
            }

            run_o_probe(tile, buf);
            CUDA_CHECK(cudaEventRecord(event_comp_o[tile], stream_comp_o));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_comp_o));

        std::cout << "[Q3] ORDERS HT built (capacity=" << orders_ht_cap << ")" << std::endl;

        // nvcomp + pb cleanup moved to after total_end
        for (size_t i = 0; i < q3_max_o_field_ev; i++) {
            CUDA_CHECK(cudaEventDestroy(ev_io_o[i]));
            CUDA_CHECK(cudaEventDestroy(ev_decomp_o[i]));
        }
        for (size_t i = 0; i < q3_max_o_tiles; i++)
            CUDA_CHECK(cudaEventDestroy(event_comp_o[i]));
        for (int s = 0; s < 2; s++)
            CUDA_CHECK(cudaStreamDestroy(stream_io_o[s]));
        CUDA_CHECK(cudaStreamDestroy(stream_comp_o));
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q3-TIMING] Phase 2 (ORDERS): " << ms << " ms" << std::endl;
        phase_start = now;
    }


    // ═══════════════════════════════════════════════════════════════════
    // Phase 3: LINEITEM — pipelined tile loop (NPIPE=4, multi-stream IO)
    // ═══════════════════════════════════════════════════════════════════
    CUDA_CHECK(cudaMemsetAsync(d_aggr_keys, 0xFF, aggr_cap * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_aggr_revenues, 0, aggr_cap * sizeof(int64_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream)); stream = nullptr;

    {
        constexpr int NPIPE = 4;
        constexpr int LI_SHIPDATE = 0, LI_EXTPRICE = 1, LI_DISCOUNT = 2, LI_ORDERKEY = 3;
        constexpr int LI_NFIELDS = 4;
        const size_t li_fi[LI_NFIELDS] = { FI_L_SHIPDATE, FI_L_EXTPRICE, FI_L_DISCOUNT, FI_L_ORDERKEY };

        const size_t TILE_PAGES_P3 = Q3_TILE_PAGES_P3;
        const size_t l_i64_tile_npages_max_p3 = pre_l_i64_tile_npages_max_p3;

        // tile_data, d_active_pg, d_ps_ref, d_ps_i64
        // are pre-allocated before the measurement point.
        std::vector<uint32_t> h_active_pg_buf(TILE_PAGES_P3);
        std::vector<uint64_t> h_ps_ref_buf(TILE_PAGES_P3 + 1);
        std::vector<uint64_t> h_ps_i64_buf(l_i64_tile_npages_max_p3 + 1);

        // nvctx_l + pb arrays are pre-allocated before total_start

        cudaStream_t &stream_comp = stream_comp_p3;

        // Pre-compute tile info + filter empty tiles
        struct TileInfo {
            size_t p_lo, tile_np;
            uint64_t first_row, last_row, tile_nrows;
            std::vector<uint32_t> active_i32;
            bool use_selective;
            std::vector<uint8_t> h_mask;
            size_t i64_start, i64_end, i64_np;
            uint64_t i64_offset;
            std::vector<uint32_t> i64_active;
            bool use_sel_i64;
        };
        const size_t n_tiles_all = (npages_l_i32 + TILE_PAGES_P3 - 1) / TILE_PAGES_P3;
        std::vector<TileInfo> tiles;
        tiles.reserve(n_tiles_all);
        for (size_t p_lo = 0; p_lo < npages_l_i32; p_lo += TILE_PAGES_P3) {
            TileInfo ti;
            ti.p_lo = p_lo;
            ti.tile_np = std::min(TILE_PAGES_P3, (size_t)(npages_l_i32 - p_lo));
            ti.first_row = ps_l_i32_full[p_lo];
            ti.last_row = ps_l_i32_full[p_lo + ti.tile_np];
            ti.tile_nrows = ti.last_row - ti.first_row;
            if (ti.tile_nrows == 0) continue;

            for (size_t j = 0; j < ti.tile_np; j++)
                if (h_zm_li_mask[p_lo + j]) ti.active_i32.push_back((uint32_t)(p_lo + j));
            if (enable_zonemap && ti.active_i32.empty()) continue;
            ti.use_selective = enable_zonemap && ti.active_i32.size() < ti.tile_np;
            if (ti.use_selective) {
                ti.h_mask.resize(ti.tile_np);
                for (size_t j = 0; j < ti.tile_np; j++)
                    ti.h_mask[j] = h_zm_li_mask[p_lo + j];
            }

            auto it_s = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), ti.first_row);
            ti.i64_start = (it_s == ps_l_i64_full.begin()) ? 0 : (size_t)(it_s - ps_l_i64_full.begin()) - 1;
            auto it_e = std::upper_bound(ps_l_i64_full.begin(), ps_l_i64_full.end(), ti.last_row - 1);
            ti.i64_end = (size_t)(it_e - ps_l_i64_full.begin());
            ti.i64_np = ti.i64_end - ti.i64_start;
            ti.i64_offset = ti.first_row - ps_l_i64_full[ti.i64_start];

            if (enable_zonemap) {
                for (uint32_t pg = ti.i64_start; pg < ti.i64_end; pg++)
                    if (h_mask_li_i64[pg]) ti.i64_active.push_back(pg);
            }
            ti.use_sel_i64 = enable_zonemap && ti.i64_active.size() < ti.i64_np;
            tiles.push_back(std::move(ti));
        }
        const size_t num_tiles = tiles.size();
        std::cout << "[GIDP+BAM Q3] Pipeline: NPIPE=" << NPIPE
                  << " TILE_PAGES=" << TILE_PAGES_P3
                  << " tiles=" << num_tiles << "/" << n_tiles_all
                  << " (zone map pruned)" << std::endl;

        // Lambda: build IO descriptors for all 4 LINEITEM fields
        auto build_tile_descs = [&](size_t tile_idx, int buf) -> std::pair<uint32_t, uint64_t> {
            auto &ti = tiles[tile_idx];
            uint32_t ndescs = 0;
            uint64_t io_bytes = 0;

            // INT32 fields (LI_SHIPDATE, LI_EXTPRICE, LI_DISCOUNT: fi=0,1,2)
            for (int fi = 0; fi < 3; fi++) {
                size_t gfi = li_fi[fi];
                bool is_compressed = (field_comp_methods[gfi] != CompressionMethod::NONE);
                if (ti.use_selective) {
                    size_t io_offset = 0;
                    for (size_t idx = 0; idx < ti.active_i32.size(); idx++) {
                        uint32_t pg = ti.active_i32[idx];
                        size_t slot = pg - ti.p_lo;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_buf_l[fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data[buf][fi]) + slot * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_bam[0].h_descs[0][ndescs++] = desc;
                    }
                } else {
                    size_t io_offset = 0;
                    for (size_t j = 0; j < ti.tile_np; j++) {
                        uint64_t pg = ti.p_lo + j;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_buf_l[fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data[buf][fi]) + j * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_bam[0].h_descs[0][ndescs++] = desc;
                    }
                }
            }

            // INT64 field (L_ORDERKEY: fi=3)
            {
                int fi = 3;
                size_t gfi = li_fi[fi];
                bool is_compressed = (field_comp_methods[gfi] != CompressionMethod::NONE);
                if (ti.use_sel_i64) {
                    size_t io_offset = 0;
                    for (size_t idx = 0; idx < ti.i64_active.size(); idx++) {
                        uint32_t pg = ti.i64_active[idx];
                        size_t slot = pg - ti.i64_start;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_buf_l[fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data[buf][LI_ORDERKEY]) + slot * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_bam[0].h_descs[0][ndescs++] = desc;
                    }
                } else {
                    size_t io_offset = 0;
                    for (size_t j = 0; j < ti.i64_np; j++) {
                        uint64_t pg = ti.i64_start + j;
                        BamBulkReadDesc desc{};
                        if (is_compressed) {
                            uint64_t byte_offset = h_comp_offsets[gfi][pg];
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                            desc.device = dev;
                            desc.dest = static_cast<char*>(io_buf_l[fi]) + io_offset;
                            desc.copy_bytes = comp_sz;
                            io_offset += page_size;
                        } else {
                            uint64_t page_id = field_start_page_ids[gfi] + pg;
                            uint32_t dev = page_id % n_devices;
                            uint64_t local_pg_dev = page_id / n_devices;
                            desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                            desc.nblocks = blocks_per_page;
                            desc.device = dev;
                            desc.dest = static_cast<char*>(tile_data[buf][LI_ORDERKEY]) + j * page_size;
                            desc.copy_bytes = page_size;
                        }
                        io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                        io_ctx_bam[0].h_descs[0][ndescs++] = desc;
                    }
                }
            }
            return {ndescs, io_bytes};
        };

        // Lambda: decompress all fields (io_buf_l → tile_data[buf])
        auto run_tile_decomp = [&](size_t tile_idx, int buf) {
            auto &ti = tiles[tile_idx];
            if (!any_compressed) return;

            // Decompress INT32 fields (LI_SHIPDATE, LI_EXTPRICE, LI_DISCOUNT: fi=0,1,2)
            for (int fi = 0; fi < 3; fi++) {
                size_t gfi = li_fi[fi];
                if (field_comp_methods[gfi] == CompressionMethod::NONE) continue;
                size_t slot_base = tile_idx * TILE_PAGES_P3;
                size_t decomp_count = 0;
                if (ti.use_selective) {
                    for (size_t idx = 0; idx < ti.active_i32.size(); idx++) {
                        uint32_t pg = ti.active_i32[idx];
                        size_t local_pg = pg - ti.p_lo;
                        uint32_t comp_sz = h_comp_sizes[gfi][pg];
                        if (comp_sz < page_size) {
                            pb_comp_ptrs[fi][slot_base + decomp_count] =
                                static_cast<char*>(io_buf_l[fi]) + idx * page_size;
                            pb_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                            pb_decomp_ptrs[fi][slot_base + decomp_count] =
                                static_cast<char*>(tile_data[buf][fi]) + local_pg * page_size;
                            pb_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                            decomp_count++;
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                static_cast<char*>(tile_data[buf][fi]) + local_pg * page_size,
                                static_cast<char*>(io_buf_l[fi]) + idx * page_size,
                                page_size, cudaMemcpyDeviceToDevice, stream_comp));
                        }
                    }
                } else {
                    for (size_t j = 0; j < ti.tile_np; j++) {
                        uint64_t pg = ti.p_lo + j;
                        uint32_t comp_sz = h_comp_sizes[gfi][pg];
                        if (comp_sz < page_size) {
                            pb_comp_ptrs[fi][slot_base + decomp_count] =
                                static_cast<char*>(io_buf_l[fi]) + j * page_size;
                            pb_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                            pb_decomp_ptrs[fi][slot_base + decomp_count] =
                                static_cast<char*>(tile_data[buf][fi]) + j * page_size;
                            pb_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                            decomp_count++;
                        } else {
                            CUDA_CHECK(cudaMemcpyAsync(
                                static_cast<char*>(tile_data[buf][fi]) + j * page_size,
                                static_cast<char*>(io_buf_l[fi]) + j * page_size,
                                page_size, cudaMemcpyDeviceToDevice, stream_comp));
                        }
                    }
                }
                if (decomp_count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_comp_ptrs,
                        pb_comp_ptrs[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_decomp_ptrs,
                        pb_decomp_ptrs[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_comp_sizes,
                        pb_comp_sizes[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_decomp_sizes,
                        pb_decomp_sizes[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                    Gidp::nvcomp_decompctx_run(field_comp_methods[gfi], nvctx_l[fi],
                                               decomp_count, page_size, stream_comp,
                                               /*do_sync=*/false, /*skip_h2d=*/true);
                    s_kernel_launches++;
                }
            }

            // Decompress INT64 field (L_ORDERKEY: fi=3)
            {
                int fi = 3;
                size_t gfi = li_fi[fi];
                if (field_comp_methods[gfi] != CompressionMethod::NONE) {
                    size_t slot_base = tile_idx * l_i64_tile_npages_max_p3;
                    size_t decomp_count = 0;
                    if (ti.use_sel_i64) {
                        for (size_t idx = 0; idx < ti.i64_active.size(); idx++) {
                            uint32_t pg = ti.i64_active[idx];
                            size_t slot = pg - ti.i64_start;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            if (comp_sz < page_size) {
                                pb_comp_ptrs[LI_ORDERKEY][slot_base + decomp_count] =
                                    static_cast<char*>(io_buf_l[fi]) + idx * page_size;
                                pb_comp_sizes[LI_ORDERKEY][slot_base + decomp_count] = comp_sz;
                                pb_decomp_ptrs[LI_ORDERKEY][slot_base + decomp_count] =
                                    static_cast<char*>(tile_data[buf][LI_ORDERKEY]) + slot * page_size;
                                pb_decomp_sizes[LI_ORDERKEY][slot_base + decomp_count] = page_size;
                                decomp_count++;
                            } else {
                                CUDA_CHECK(cudaMemcpyAsync(
                                    static_cast<char*>(tile_data[buf][LI_ORDERKEY]) + slot * page_size,
                                    static_cast<char*>(io_buf_l[fi]) + idx * page_size,
                                    page_size, cudaMemcpyDeviceToDevice, stream_comp));
                            }
                        }
                    } else {
                        for (size_t j = 0; j < ti.i64_np; j++) {
                            uint64_t pg = ti.i64_start + j;
                            uint32_t comp_sz = h_comp_sizes[gfi][pg];
                            if (comp_sz < page_size) {
                                pb_comp_ptrs[LI_ORDERKEY][slot_base + decomp_count] =
                                    static_cast<char*>(io_buf_l[fi]) + j * page_size;
                                pb_comp_sizes[LI_ORDERKEY][slot_base + decomp_count] = comp_sz;
                                pb_decomp_ptrs[LI_ORDERKEY][slot_base + decomp_count] =
                                    static_cast<char*>(tile_data[buf][LI_ORDERKEY]) + j * page_size;
                                pb_decomp_sizes[LI_ORDERKEY][slot_base + decomp_count] = page_size;
                                decomp_count++;
                            } else {
                                CUDA_CHECK(cudaMemcpyAsync(
                                    static_cast<char*>(tile_data[buf][LI_ORDERKEY]) + j * page_size,
                                    static_cast<char*>(io_buf_l[fi]) + j * page_size,
                                    page_size, cudaMemcpyDeviceToDevice, stream_comp));
                            }
                        }
                    }
                    if (decomp_count > 0) {
                        CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_comp_ptrs,
                            pb_comp_ptrs[fi] + slot_base,
                            decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                        CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_decomp_ptrs,
                            pb_decomp_ptrs[fi] + slot_base,
                            decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                        CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_comp_sizes,
                            pb_comp_sizes[fi] + slot_base,
                            decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                        CUDA_CHECK(cudaMemcpyAsync(nvctx_l[fi].d_decomp_sizes,
                            pb_decomp_sizes[fi] + slot_base,
                            decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                        Gidp::nvcomp_decompctx_run(field_comp_methods[gfi], nvctx_l[fi],
                                                   decomp_count, page_size, stream_comp,
                                                   /*do_sync=*/false, /*skip_h2d=*/true);
                        s_kernel_launches++;
                    }
                }
            }
        };

        // Lambda: paged kernel probe on stream_comp
        auto run_l_probe = [&](size_t tile_idx, int buf) {
            auto &ti = tiles[tile_idx];

            // Build active page list + prefix sums for paged kernel
            {
                uint32_t num_active;
                if (ti.use_selective) {
                    num_active = (uint32_t)ti.active_i32.size();
                    for (uint32_t i = 0; i < num_active; i++)
                        h_active_pg_buf[i] = ti.active_i32[i] - (uint32_t)ti.p_lo;
                } else {
                    num_active = (uint32_t)ti.tile_np;
                    for (uint32_t i = 0; i < num_active; i++)
                        h_active_pg_buf[i] = i;
                }
                CUDA_CHECK(cudaMemcpyAsync(d_active_pg[buf], h_active_pg_buf.data(),
                    num_active * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_comp));

                uint64_t origin = ps_l_i64_full[ti.i64_start];
                for (size_t pg = 0; pg <= ti.tile_np; pg++)
                    h_ps_ref_buf[pg] = ps_l_i32_full[ti.p_lo + pg] - origin;
                CUDA_CHECK(cudaMemcpyAsync(d_ps_ref[buf], h_ps_ref_buf.data(),
                    (ti.tile_np + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_comp));

                for (size_t pg = 0; pg <= ti.i64_np; pg++)
                    h_ps_i64_buf[pg] = ps_l_i64_full[ti.i64_start + pg] - origin;
                CUDA_CHECK(cudaMemcpyAsync(d_ps_i64[buf], h_ps_i64_buf.data(),
                    (ti.i64_np + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_comp));

                uint32_t stride = (page_size - 12) / sizeof(int32_t);
                if (is_q3sel) {
                    q3sel_lineitem_probe_aggr_paged(
                        static_cast<const char*>(tile_data[buf][LI_SHIPDATE]),
                        static_cast<const char*>(tile_data[buf][LI_EXTPRICE]),
                        static_cast<const char*>(tile_data[buf][LI_DISCOUNT]),
                        static_cast<const char*>(tile_data[buf][LI_ORDERKEY]),
                        d_active_pg[buf], num_active,
                        (uint32_t)page_size, stride,
                        d_ps_ref[buf], d_ps_i64[buf], (uint32_t)ti.i64_np,
                        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                        d_aggr_keys, d_aggr_revenues, aggr_mask,
                        options.disable_other_filters ? 0 : 19950315, stream_comp);
                } else {
                    q3_lineitem_probe_aggr_paged(
                        static_cast<const char*>(tile_data[buf][LI_SHIPDATE]),
                        static_cast<const char*>(tile_data[buf][LI_EXTPRICE]),
                        static_cast<const char*>(tile_data[buf][LI_DISCOUNT]),
                        static_cast<const char*>(tile_data[buf][LI_ORDERKEY]),
                        d_active_pg[buf], num_active,
                        (uint32_t)page_size, stride,
                        d_ps_ref[buf], d_ps_i64[buf], (uint32_t)ti.i64_np,
                        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                        d_aggr_keys, d_aggr_revenues, aggr_mask, stream_comp);
                }
                s_kernel_launches++;
            }
        };

        // Pipeline loop: io_buf_l shared across NPIPE, tile_data[NPIPE] ring buffer
        for (size_t tile = 0; tile < num_tiles; tile++) {
            int buf = tile % NPIPE;
            // Wait for tile_data[buf] to be free (NPIPE tiles ago)
            if (tile >= (size_t)NPIPE)
                CUDA_CHECK(cudaStreamWaitEvent(stream_io_l, event_comp_vec[tile - NPIPE]));
            // Wait for io_buf_l to be free (previous tile's decomp)
            if (tile > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_io_l, ev_decomp_done_l));
                CUDA_CHECK(cudaEventSynchronize(io_ctx_bam[0].h2d_done[0]));
            }
            // IO all fields into io_buf_l (compressed) / tile_data[buf] (uncompressed)
            auto [ndescs, io_bytes] = build_tile_descs(tile, buf);
            bam_bulk_read_async(io_ctx_bam[0], ndescs, 0, stream_io_l);
            s_kernel_launches++;
            CUDA_CHECK(cudaEventRecord(ev_io_done_l, stream_io_l));
            total_io_count += ndescs;
            total_io_bytes += io_bytes;
            // Decomp: io_buf_l → tile_data[buf]
            CUDA_CHECK(cudaStreamWaitEvent(stream_comp, ev_io_done_l));
            run_tile_decomp(tile, buf);
            CUDA_CHECK(cudaEventRecord(ev_decomp_done_l, stream_comp));
            // Probe: tile_data[buf] only (io_buf_l is free after this point)
            run_l_probe(tile, buf);
            CUDA_CHECK(cudaEventRecord(event_comp_vec[tile], stream_comp));
        }
        if (num_tiles > 0)
            CUDA_CHECK(cudaStreamSynchronize(stream_comp));

        for (size_t i = 0; i < num_tiles; i++)
            CUDA_CHECK(cudaEventDestroy(event_comp_vec[i]));
        CUDA_CHECK(cudaEventDestroy(ev_io_done_l));
        CUDA_CHECK(cudaEventDestroy(ev_decomp_done_l));
        CUDA_CHECK(cudaStreamDestroy(stream_io_l));
        stream = stream_comp;
    }

    {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q3-TIMING] Phase 3 (LINEITEM): " << ms << " ms" << std::endl;
        phase_start = now;
    }

    // ── Phase 4: Collect + Top-10 ──
    CUDA_CHECK(cudaMemsetAsync(d_result_count, 0, sizeof(uint32_t), stream));

    q3_collect_results(d_aggr_keys, d_aggr_revenues, aggr_cap,
                       d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                       d_results, d_result_count, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto phase4_collect_end = std::chrono::steady_clock::now();

    uint32_t h_result_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Sort on GPU (CUB DeviceMergeSort with pre-allocated temp)
    cub::DeviceMergeSort::SortKeys(d_sort_temp, sort_temp_bytes,
        d_results, (int)h_result_count, Q3ResultCmp{}, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto phase4_sort_end = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMemcpy(results, d_results,
                          h_result_count * sizeof(Q3ResultRow), cudaMemcpyDeviceToHost));

    {
        auto now = std::chrono::steady_clock::now();
        double ms_collect = std::chrono::duration<double, std::milli>(phase4_collect_end - phase_start).count();
        double ms_sort = std::chrono::duration<double, std::milli>(phase4_sort_end - phase4_collect_end).count();
        double ms_d2h = std::chrono::duration<double, std::milli>(now - phase4_sort_end).count();
        double ms_total = std::chrono::duration<double, std::milli>(now - phase_start).count();
        std::cout << "[Q3-TIMING] Phase 4 (COLLECT+SORT): " << ms_total << " ms"
                  << "  [collect=" << ms_collect
                  << " gpu_sort=" << ms_sort
                  << " d2h=" << ms_d2h << "]" << std::endl;
    }

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
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
        auto &r = results[i];
        printf("%10lu | %13ld | %11u | %14u\n",
               (unsigned long)r.l_orderkey, (long)r.revenue,
               r.o_orderdate, r.o_shippriority);
    }
    std::cout << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    cudaFreeHost(results);
    bam_bulk_read_ctx_destroy(io_ctx_small);
    for (int s = 0; s < 2; s++)
        bam_bulk_read_ctx_destroy(io_ctx_bam[s]);
    mb_cuda_free(staging_data);
    mb_cuda_free(staging_io);
    cudaFree(d_batch_ps);
    cudaFree(d_page_mask);
    cudaFree(d_custkey_flat);
    cudaFree(d_custkey_set);
    cudaFree(d_orders_ht_keys);
    cudaFree(d_orders_ht_payloads);
    cudaFree(d_aggr_keys);
    cudaFree(d_aggr_revenues);
    cudaFree(d_results);
    cudaFree(d_result_count);
    cudaFree(d_sort_temp);
    // Pre-allocated tile buffers
    for (int fi = 0; fi < 4; fi++)
        mb_cuda_free(tile_data_o[0][fi]);
    if (io_staging_o[0]) mb_cuda_free(io_staging_o[0]);
    if (io_staging_o[1]) mb_cuda_free(io_staging_o[1]);
    cudaFree(d_o_active_pg[0]);
    cudaFree(d_o_ps_ref[0]);
    cudaFree(d_o_ps_i64[0]);
    for (int p = 0; p < Q3_BAM_NPIPE; p++) {
        for (int fi = 0; fi < 4; fi++)
            mb_cuda_free(tile_data[p][fi]);
        cudaFree(d_active_pg[p]);
        cudaFree(d_ps_ref[p]);
        cudaFree(d_ps_i64[p]);
    }
    for (int fi = 0; fi < 4; fi++)
        if (io_buf_l[fi]) mb_cuda_free(io_buf_l[fi]);

    if (any_compressed) {
        Gidp::nvcomp_decompctx_free(nvctx);
        // Phase 2 (ORDERS) nvcomp + pb
        for (int fi = 0; fi < 4; fi++) {
            Gidp::nvcomp_decompctx_free(nvctx_o[fi]);
            if (pb_comp_ptrs_o[fi])    cudaFreeHost(pb_comp_ptrs_o[fi]);
            if (pb_decomp_ptrs_o[fi])  cudaFreeHost(pb_decomp_ptrs_o[fi]);
            if (pb_comp_sizes_o[fi])   cudaFreeHost(pb_comp_sizes_o[fi]);
            if (pb_decomp_sizes_o[fi]) cudaFreeHost(pb_decomp_sizes_o[fi]);
        }
        // Phase 3 (LINEITEM) nvcomp + pb
        for (int fi = 0; fi < 4; fi++)
            Gidp::nvcomp_decompctx_free(nvctx_l[fi]);
        for (int fi = 0; fi < 4; fi++) {
            if (pb_comp_ptrs[fi])    cudaFreeHost(pb_comp_ptrs[fi]);
            if (pb_decomp_ptrs[fi])  cudaFreeHost(pb_decomp_ptrs[fi]);
            if (pb_comp_sizes[fi])   cudaFreeHost(pb_comp_sizes[fi]);
            if (pb_decomp_sizes[fi]) cudaFreeHost(pb_decomp_sizes[fi]);
        }
    }
    // Cleanup zone map contexts
    if (enable_zonemap) {
        bam_zonemap_ctx_destroy(zm_ctx_ord);
        bam_zonemap_ctx_destroy(zm_ctx_li);
    } else {
        free(h_zm_ord_mask);
        free(h_zm_li_mask);
    }
    if (d_mask_ord_i64) cudaFree(d_mask_ord_i64);
    if (d_mask_li_i64) cudaFree(d_mask_li_i64);
    if (d_ps_o_i32_zm) cudaFree(d_ps_o_i32_zm);
    if (d_ps_o_i64_zm) cudaFree(d_ps_o_i64_zm);
    if (d_ps_l_i32_zm) cudaFree(d_ps_l_i32_zm);
    if (d_ps_l_i64_zm) cudaFree(d_ps_l_i64_zm);

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

// ============================================================
// Q1: Pricing Summary Report (LINEITEM only, 7 fields)
// I/O: BaM GPU-initiated, Decomp: nvCOMP host-launched
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

    std::cout << "=== TPCH Q1 (GIDP+BAM) ===" << std::endl;
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

    // ── 5a. Zone map metadata (ctx created after io_ctx; eval inside timing) ──
    std::vector<size_t> active_pages;
    bool use_zonemap = options.enable_zonemap;
    uint64_t zm_stats_start = 0, zm_stats_npg = 0, zm_nstats = 0;
    {
        size_t sd_col_zm = q1_cols[FI_L_SHIPDATE];
        zm_stats_start = metadata.table_lineitem_stats_start_page_ids[sd_col_zm];
        zm_stats_npg   = metadata.table_lineitem_stats_npages[sd_col_zm];
        zm_nstats      = metadata.table_lineitem_nstats[sd_col_zm];
        if (!(use_zonemap && zm_nstats > 0 && zm_stats_start > 0 && zm_stats_npg > 0))
            use_zonemap = false;
    }
    if (!use_zonemap) {
        active_pages.resize(field_npages_arr[FI_L_SHIPDATE]);
        std::iota(active_pages.begin(), active_pages.end(), size_t(0));
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

    std::vector<uint64_t> h_prefix_sum[NUM_Q1_FIELDS];
    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
        size_t col = q1_cols[fi];
        uint64_t ps_start      = metadata.table_lineitem_prefix_sum_start_page_ids[col];
        uint64_t ps_npages_cnt = metadata.table_lineitem_prefix_sum_npages[col];
        h_prefix_sum[fi] = read_prefix_sum(ps_start, ps_npages_cnt, field_npages_arr[fi]);
    }

    // ── 7. GPU memory allocation (double-buffered pipeline) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    constexpr int NPIPE = 2;
    constexpr size_t BATCH_PAGES_MAX = 1536;

    size_t n_compressed = 0;
    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++)
        if (field_comp_methods[fi] != CompressionMethod::NONE) n_compressed++;

    // Create BaM I/O context first (page_cache DMA has large fixed overhead)
    const uint32_t q1_io_blocks = static_cast<uint32_t>(sm_count);
    BamBulkReadCtx io_ctx = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        q1_io_blocks,
        static_cast<uint32_t>(BATCH_PAGES_MAX * NUM_Q1_FIELDS));

    // Zone map ctx: borrows io_ctx's page_cache
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (use_zonemap) {
        zm_ctx = bam_zonemap_ctx_create(
            io_ctx.d_ctrls, io_ctx.d_pc, io_ctx.pc_base,
            static_cast<uint32_t>(page_size), field_npages_arr[FI_L_SHIPDATE]);
        for (uint64_t j = 0; j < zm_stats_npg; j++) {
            uint64_t pg_id = zm_stats_start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            zm_ctx.h_reads[j] = {
                ds.partition_start_lbas[dev] + local * blocks_per_page,
                static_cast<uint32_t>(blocks_per_page), dev};
        }
        zm_nreads = static_cast<uint32_t>(zm_stats_npg);
        zm_ctx.h_preds[0] = {0, zm_nstats, INT32_MIN, (int32_t)19980902};
        zm_npreds = 1;
        zm_valid = true;
    }

    // Compute BATCH_PAGES from measured remaining GPU memory (40 GiB total cap)
    constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
    size_t gpu_free_after_io = 0;
    cudaMemGetInfo(&gpu_free_after_io, &gpu_total_dummy);
    uint64_t total_used = gpu_free_start - gpu_free_after_io;
    uint64_t remaining = (GPU_MEM_BUDGET > total_used) ? GPU_MEM_BUDGET - total_used : 0;
    size_t bytes_per_batch_page = (NPIPE * NUM_Q1_FIELDS + n_compressed) * page_size;
    size_t batch_by_mem = (bytes_per_batch_page > 0)
        ? remaining / bytes_per_batch_page : BATCH_PAGES_MAX;
    const size_t BATCH_PAGES = std::min(BATCH_PAGES_MAX, std::max(batch_by_mem, (size_t)1));

    // All Q1 fields are LINEITEM with the same npages
    const uint64_t npages = field_npages_arr[0];

    // Pre-compute batch→active_pages mapping and filter empty batches
    const size_t num_batches_all = (npages + BATCH_PAGES - 1) / BATCH_PAGES;
    std::vector<std::vector<size_t>> batch_actives(num_batches_all);
    std::vector<size_t> non_empty_batches;
    size_t num_batches = 0;
    if (!use_zonemap) {
        size_t ap_idx = 0;
        for (size_t b = 0; b < num_batches_all; b++) {
            size_t pg_base = b * BATCH_PAGES;
            size_t pg_end = std::min(pg_base + BATCH_PAGES, (size_t)npages);
            while (ap_idx < active_pages.size() && active_pages[ap_idx] < pg_base)
                ap_idx++;
            for (size_t i = ap_idx; i < active_pages.size() && active_pages[i] < pg_end; i++)
                batch_actives[b].push_back(active_pages[i]);
            if (!batch_actives[b].empty())
                non_empty_batches.push_back(b);
        }
        num_batches = non_empty_batches.size();
    }

    // Data buffers: NPIPE × NUM_Q1_FIELDS ring buffer
    void *data_buf[NPIPE][NUM_Q1_FIELDS];
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++)
            data_buf[p][fi] = mb_cuda_alloc(BATCH_PAGES * page_size);

    // Per-field IO buffers (compressed fields only, shared across NPIPE)
    void *io_buf[NUM_Q1_FIELDS] = {};
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++)
            if (field_comp_methods[fi] != CompressionMethod::NONE)
                io_buf[fi] = mb_cuda_alloc(BATCH_PAGES * page_size);
    }

    // Aggregate array (persistent across batches)
    constexpr size_t agg_size = Q1_NUM_GROUPS * Q1_NUM_AGGS * sizeof(int64_t);
    int64_t *d_agg = nullptr;
    CUDA_CHECK(cudaMalloc(&d_agg, agg_size));
    CUDA_CHECK(cudaMemset(d_agg, 0, agg_size));

    // Per-field nvCOMP contexts (shared across NPIPE — serialized on stream_comp)
    NvcompDecompCtx nvctx[NUM_Q1_FIELDS]{};
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            std::vector<FieldPageInfo> tf(1);
            tf[0].compression_method = field_comp_methods[fi];
            Gidp::nvcomp_decompctx_alloc(nvctx[fi], BATCH_PAGES, page_size, tf);
        }
    }

    // Per-batch pinned host arrays for fully async nvcomp H2D.
    void   **pb_h_comp_ptrs[NUM_Q1_FIELDS]{};
    void   **pb_h_decomp_ptrs[NUM_Q1_FIELDS]{};
    size_t  *pb_h_comp_sizes[NUM_Q1_FIELDS]{};
    size_t  *pb_h_decomp_sizes[NUM_Q1_FIELDS]{};
    if (any_compressed && num_batches_all > 0) {
        size_t total_slots = num_batches_all * BATCH_PAGES;
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            CUDA_CHECK(cudaMallocHost(&pb_h_comp_ptrs[fi],    total_slots * sizeof(void*)));
            CUDA_CHECK(cudaMallocHost(&pb_h_decomp_ptrs[fi],  total_slots * sizeof(void*)));
            CUDA_CHECK(cudaMallocHost(&pb_h_comp_sizes[fi],   total_slots * sizeof(size_t)));
            CUDA_CHECK(cudaMallocHost(&pb_h_decomp_sizes[fi], total_slots * sizeof(size_t)));
        }
    }

    // 1 IO stream + 1 compute stream
    cudaStream_t stream_io, stream_comp;
    CUDA_CHECK(cudaStreamCreate(&stream_io));
    CUDA_CHECK(cudaStreamCreate(&stream_comp));

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // ── 8. Pipelined batch execution ──
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    uint32_t capacity = (page_size - 12) / 4;

    std::cout << "[GIDP+BAM Q1] Batch execution: " << num_batches
              << " batches of " << BATCH_PAGES << " pages" << std::endl;

    // Lambda: build I/O descriptors for all fields into io_ctx.h_descs[0]
    auto build_tile_descs = [&](size_t pipe_idx, int buf)
        -> std::pair<uint32_t, uint64_t>
    {
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        auto& ba = batch_actives[batch_idx];
        uint32_t ndescs = 0;
        uint64_t io_bytes = 0;

        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
            bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
            size_t io_offset = 0;
            for (size_t pg : ba) {
                size_t local_pg = pg - pg_base;
                BamBulkReadDesc desc{};
                if (is_compressed) {
                    uint64_t byte_offset = h_comp_offsets[fi][pg];
                    uint64_t page_id = field_start_page_ids[fi] + pg;
                    uint32_t dev = page_id % n_devices;
                    desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                    desc.device = dev;
                    desc.dest = static_cast<char*>(io_buf[fi]) + io_offset;
                    desc.copy_bytes = comp_sz;
                    io_offset += page_size;
                } else {
                    uint64_t page_id = field_start_page_ids[fi] + pg;
                    uint32_t dev = page_id % n_devices;
                    uint64_t local_pg_dev = page_id / n_devices;
                    desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                    desc.nblocks = blocks_per_page;
                    desc.device = dev;
                    desc.dest = static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size;
                    desc.copy_bytes = page_size;
                }
                io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                io_ctx.h_descs[0][ndescs++] = desc;
            }
        }
        return {ndescs, io_bytes};
    };

    // Lambda: decompress all fields from io_buf[fi] → data_buf[buf][fi]
    auto run_tile_decomp = [&](size_t pipe_idx, int buf) {
        if (!any_compressed) return;
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        auto& ba = batch_actives[batch_idx];
        size_t slot_base = pipe_idx * BATCH_PAGES;

        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            size_t decomp_count = 0;
            for (size_t idx = 0; idx < ba.size(); idx++) {
                size_t pg = ba[idx];
                size_t local_pg = pg - pg_base;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                if (comp_sz < page_size) {
                    pb_h_comp_ptrs[fi][slot_base + decomp_count] =
                        static_cast<char*>(io_buf[fi]) + idx * page_size;
                    pb_h_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                    pb_h_decomp_ptrs[fi][slot_base + decomp_count] =
                        static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size;
                    pb_h_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                    decomp_count++;
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size,
                        static_cast<char*>(io_buf[fi]) + idx * page_size,
                        page_size, cudaMemcpyDeviceToDevice, stream_comp));
                }
            }
            if (decomp_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_comp_ptrs,
                    pb_h_comp_ptrs[fi] + slot_base,
                    decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_decomp_ptrs,
                    pb_h_decomp_ptrs[fi] + slot_base,
                    decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_comp_sizes,
                    pb_h_comp_sizes[fi] + slot_base,
                    decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_decomp_sizes,
                    pb_h_decomp_sizes[fi] + slot_base,
                    decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                Gidp::nvcomp_decompctx_run(field_comp_methods[fi], nvctx[fi],
                                           decomp_count, page_size, stream_comp,
                                           /*do_sync=*/false, /*skip_h2d=*/true);
                s_kernel_launches++;
            }
        }
    };

    // Lambda: zero inactive page headers + Q1 scan
    auto run_scan = [&](size_t pipe_idx, int buf) {
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        size_t batch_np = std::min(BATCH_PAGES, npages - pg_base);
        auto& ba = batch_actives[batch_idx];

        // Zone map partial batch: zero inactive page headers (nalloc=0)
        bool batch_partial = use_zonemap && ba.size() < batch_np;
        if (batch_partial) {
            std::vector<bool> pg_active(batch_np, false);
            for (size_t pg : ba) pg_active[pg - pg_base] = true;
            for (size_t lp = 0; lp < batch_np; lp++) {
                if (!pg_active[lp]) {
                    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++)
                        CUDA_CHECK(cudaMemsetAsync(
                            static_cast<char*>(data_buf[buf][fi]) + lp * page_size,
                            0, 12, stream_comp));
                }
            }
        }

        q1_scan_aggregate_paged(
            data_buf[buf][FI_L_SHIPDATE],
            data_buf[buf][FI_L_QUANTITY],
            data_buf[buf][FI_L_EXTPRICE],
            data_buf[buf][FI_L_DISCOUNT],
            data_buf[buf][FI_L_TAX],
            data_buf[buf][FI_L_RETURNFLAG],
            data_buf[buf][FI_L_LINESTATUS],
            (uint64_t)batch_np * capacity, capacity, page_size,
            d_agg, stream_comp, nullptr);
        s_kernel_launches++;
    };

    // Pipeline events
    cudaEvent_t ev_io_done, ev_decomp_done;
    CUDA_CHECK(cudaEventCreate(&ev_io_done));
    CUDA_CHECK(cudaEventCreate(&ev_decomp_done));
    std::vector<cudaEvent_t> event_comp_vec(num_batches_all);
    for (size_t i = 0; i < num_batches_all; i++)
        CUDA_CHECK(cudaEventCreate(&event_comp_vec[i]));

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream_io);
    }

    // ════════════════════════════════════════════
    // total_start — pipelined execution
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    uint64_t total_io_count = 0;
    uint64_t total_io_bytes = 0;

    // ── Zone map GPU eval (fused IO + eval, single kernel) ──
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream_io);
        CUDA_CHECK(cudaStreamSynchronize(stream_io));
        s_kernel_launches++;
        for (size_t pg = 0; pg < npages; pg++)
            if (zm_ctx.h_mask[pg]) active_pages.push_back(pg);

        std::cout << "[ZONEMAP] L_SHIPDATE pruning: active="
                  << active_pages.size() << "/" << npages << std::endl;

        // Populate batch_actives
        {
            size_t ap_idx = 0;
            for (size_t b = 0; b < num_batches_all; b++) {
                size_t pg_base = b * BATCH_PAGES;
                size_t pg_end = std::min(pg_base + BATCH_PAGES, (size_t)npages);
                while (ap_idx < active_pages.size() && active_pages[ap_idx] < pg_base)
                    ap_idx++;
                for (size_t i = ap_idx; i < active_pages.size() && active_pages[i] < pg_end; i++)
                    batch_actives[b].push_back(active_pages[i]);
                if (!batch_actives[b].empty())
                    non_empty_batches.push_back(b);
            }
            num_batches = non_empty_batches.size();
        }
    }

    std::cout << "[GIDP+BAM Q1] Pipeline: NPIPE=" << NPIPE
              << " BATCH_PAGES=" << BATCH_PAGES
              << " batches=" << num_batches << "/" << num_batches_all
              << " page_size=" << page_size << std::endl;

    for (size_t tile = 0; tile < num_batches; tile++) {
        int buf = tile % NPIPE;

        // Wait for data_buf[buf] to be free (NPIPE tiles ago)
        if (tile >= (size_t)NPIPE)
            CUDA_CHECK(cudaStreamWaitEvent(stream_io, event_comp_vec[tile - NPIPE]));
        // Wait for io_buf to be free (previous tile's decomp)
        if (tile > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_io, ev_decomp_done));
            CUDA_CHECK(cudaEventSynchronize(io_ctx.h2d_done[0]));
        }

        // IO all fields into io_buf (compressed) / data_buf[buf] (uncompressed)
        auto [ndescs, io_bytes] = build_tile_descs(tile, buf);
        bam_bulk_read_async(io_ctx, ndescs, 0, stream_io);
        s_kernel_launches++;
        CUDA_CHECK(cudaEventRecord(ev_io_done, stream_io));
        total_io_count += ndescs;
        total_io_bytes += io_bytes;

        // Decomp: io_buf → data_buf[buf]
        CUDA_CHECK(cudaStreamWaitEvent(stream_comp, ev_io_done));
        run_tile_decomp(tile, buf);
        CUDA_CHECK(cudaEventRecord(ev_decomp_done, stream_comp));

        // Scan (all fields decompressed into data_buf[buf])
        run_scan(tile, buf);
        CUDA_CHECK(cudaEventRecord(event_comp_vec[tile], stream_comp));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_comp));

    // ── 9. Results (inside measurement interval per Rule 5) ──
    int64_t h_agg[Q1_NUM_GROUPS * Q1_NUM_AGGS];
    CUDA_CHECK(cudaMemcpy(h_agg, d_agg, agg_size, cudaMemcpyDeviceToHost));

    // ════════════════════════════════════════════
    // total_end
    // ════════════════════════════════════════════
    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

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

    std::cout << "\nElapsed: " << elapsed << " sec" << std::endl;
    std::cout << "Total I/O: " << total_io_bytes / (1024*1024) << " MiB, "
              << total_io_count << " IOs" << std::endl;

    // ── 10. Cleanup ──
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++)
            if (field_comp_methods[fi] != CompressionMethod::NONE)
                Gidp::nvcomp_decompctx_free(nvctx[fi]);
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++) {
            if (pb_h_comp_ptrs[fi])    cudaFreeHost(pb_h_comp_ptrs[fi]);
            if (pb_h_decomp_ptrs[fi])  cudaFreeHost(pb_h_decomp_ptrs[fi]);
            if (pb_h_comp_sizes[fi])   cudaFreeHost(pb_h_comp_sizes[fi]);
            if (pb_h_decomp_sizes[fi]) cudaFreeHost(pb_h_decomp_sizes[fi]);
        }
    }
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++)
            mb_cuda_free(data_buf[p][fi]);
    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++)
        if (io_buf[fi]) mb_cuda_free(io_buf[fi]);
    CUDA_CHECK(cudaFree(d_agg));
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    bam_bulk_read_ctx_destroy(io_ctx);
    CUDA_CHECK(cudaEventDestroy(ev_io_done));
    CUDA_CHECK(cudaEventDestroy(ev_decomp_done));
    for (size_t i = 0; i < num_batches_all; i++)
        CUDA_CHECK(cudaEventDestroy(event_comp_vec[i]));
    CUDA_CHECK(cudaStreamDestroy(stream_io));
    CUDA_CHECK(cudaStreamDestroy(stream_comp));
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

    size_t total_pages = 0;
    for (size_t fi = 0; fi < NUM_Q1_FIELDS; fi++)
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
// Revenue — GIDP+BAM execution mode
// Same fields and kernel as Q6 (q6_col_vardate)
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

    // ── 6. Zone map metadata (ctx created after io_ctx; eval inside timing) ──
    std::vector<size_t> active_pages;
    bool use_zonemap = options.enable_zonemap;
    size_t sd_col = rev_cols[0];  // L_SHIPDATE
    uint64_t zm_stats_start = metadata.table_lineitem_stats_start_page_ids[sd_col];
    uint64_t zm_stats_npg   = metadata.table_lineitem_stats_npages[sd_col];
    uint64_t zm_nstats      = metadata.table_lineitem_nstats[sd_col];

    if (!(use_zonemap && zm_nstats > 0 && zm_stats_start > 0 && zm_stats_npg > 0))
        use_zonemap = false;
    if (!use_zonemap) {
        active_pages.resize(npages);
        std::iota(active_pages.begin(), active_pages.end(), size_t(0));
    }

    // ── 7. GPU memory allocation (pipeline, batch-sized) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    constexpr int NPIPE = 4;
    constexpr size_t BATCH_PAGES_MAX = 1536;

    size_t n_compressed = 0;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
        if (field_comp_methods[fi] != CompressionMethod::NONE) n_compressed++;

    // Create BaM I/O context first (page_cache DMA has large fixed overhead)
    const uint32_t io_blocks = static_cast<uint32_t>(sm_count);
    BamBulkReadCtx io_ctx = bam_bulk_read_ctx_create(
        ctrl, static_cast<uint32_t>(page_size),
        io_blocks,
        static_cast<uint32_t>(BATCH_PAGES_MAX * NUM_FIELDS));

    // Zone map context (borrows io_ctx page_cache, created outside timing)
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (use_zonemap) {
        zm_ctx = bam_zonemap_ctx_create(
            io_ctx.d_ctrls, io_ctx.d_pc, io_ctx.pc_base,
            static_cast<uint32_t>(page_size), npages);
        for (uint64_t j = 0; j < zm_stats_npg; j++) {
            uint64_t pg_id = zm_stats_start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            zm_ctx.h_reads[j] = {
                ds.partition_start_lbas[dev] + local * blocks_per_page,
                static_cast<uint32_t>(blocks_per_page), dev};
        }
        zm_nreads = static_cast<uint32_t>(zm_stats_npg);
        zm_ctx.h_preds[0] = {0, zm_nstats,
            options.q6_sd_low, options.q6_sd_high - 1};
        zm_npreds = 1;
        zm_valid = true;
    }

    // Compute BATCH_PAGES from measured remaining GPU memory (40 GiB total cap)
    constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
    size_t gpu_free_after_io = 0;
    cudaMemGetInfo(&gpu_free_after_io, &gpu_total_dummy);
    uint64_t total_used = gpu_free_start - gpu_free_after_io;
    uint64_t remaining = (GPU_MEM_BUDGET > total_used) ? GPU_MEM_BUDGET - total_used : 0;
    size_t bytes_per_batch_page = (NPIPE * NUM_FIELDS + n_compressed) * page_size;
    size_t batch_by_mem = (bytes_per_batch_page > 0)
        ? remaining / bytes_per_batch_page : BATCH_PAGES_MAX;
    const size_t BATCH_PAGES = std::min(BATCH_PAGES_MAX, std::max(batch_by_mem, (size_t)1));

    // Data buffers: NPIPE × NUM_FIELDS ring buffer
    void *data_buf[NPIPE][NUM_FIELDS];
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            data_buf[p][fi] = mb_cuda_alloc(BATCH_PAGES * page_size);

    // Per-field IO buffers (compressed fields only, shared across NPIPE)
    void *io_buf[NUM_FIELDS] = {};
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            if (field_comp_methods[fi] != CompressionMethod::NONE)
                io_buf[fi] = mb_cuda_alloc(BATCH_PAGES * page_size);
    }

    int64_t *d_revenue = static_cast<int64_t*>(mb_cuda_alloc(sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    // Per-field nvCOMP contexts (shared across NPIPE)
    NvcompDecompCtx nvctx[NUM_FIELDS]{};
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            std::vector<FieldPageInfo> tf(1);
            tf[0].compression_method = field_comp_methods[fi];
            Gidp::nvcomp_decompctx_alloc(nvctx[fi], BATCH_PAGES, page_size, tf);
        }
    }

    // 1 IO stream + 1 compute stream
    cudaStream_t stream_io, stream_comp;
    CUDA_CHECK(cudaStreamCreate(&stream_io));
    CUDA_CHECK(cudaStreamCreate(&stream_comp));
    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // ── 8. Pipelined batch execution ──
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    const size_t L_SHIPDATE_IDX = 0;
    const size_t L_QUANTITY_IDX = 1;
    const size_t L_EXTENDEDPRICE_IDX = 2;
    const size_t L_DISCOUNT_IDX = 3;
    uint32_t capacity = (page_size - 12) / 4;

    // Pre-compute batch→active_pages mapping and filter empty batches
    const size_t num_batches_all = (npages + BATCH_PAGES - 1) / BATCH_PAGES;
    std::vector<std::vector<size_t>> batch_actives(num_batches_all);
    std::vector<size_t> non_empty_batches;
    size_t num_batches = 0;
    if (!use_zonemap) {
        size_t ap_idx = 0;
        for (size_t b = 0; b < num_batches_all; b++) {
            size_t pg_base = b * BATCH_PAGES;
            size_t pg_end = std::min(pg_base + BATCH_PAGES, npages);
            while (ap_idx < active_pages.size() && active_pages[ap_idx] < pg_base)
                ap_idx++;
            for (size_t i = ap_idx; i < active_pages.size() && active_pages[i] < pg_end; i++)
                batch_actives[b].push_back(active_pages[i]);
            if (!batch_actives[b].empty())
                non_empty_batches.push_back(b);
        }
        num_batches = non_empty_batches.size();
    }

    // Per-batch pinned host arrays for fully async nvcomp H2D
    void   **pb_h_comp_ptrs[NUM_FIELDS]{};
    void   **pb_h_decomp_ptrs[NUM_FIELDS]{};
    size_t  *pb_h_comp_sizes[NUM_FIELDS]{};
    size_t  *pb_h_decomp_sizes[NUM_FIELDS]{};
    if (any_compressed && num_batches_all > 0) {
        size_t total_slots = num_batches_all * BATCH_PAGES;
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            CUDA_CHECK(cudaMallocHost(&pb_h_comp_ptrs[fi],    total_slots * sizeof(void*)));
            CUDA_CHECK(cudaMallocHost(&pb_h_decomp_ptrs[fi],  total_slots * sizeof(void*)));
            CUDA_CHECK(cudaMallocHost(&pb_h_comp_sizes[fi],   total_slots * sizeof(size_t)));
            CUDA_CHECK(cudaMallocHost(&pb_h_decomp_sizes[fi], total_slots * sizeof(size_t)));
        }
    }

    // Lambda: build I/O descriptors for all fields into io_ctx.h_descs[0]
    auto build_tile_descs = [&](size_t pipe_idx, int buf)
        -> std::pair<uint32_t, uint64_t>
    {
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        auto& ba = batch_actives[batch_idx];
        uint32_t ndescs = 0;
        uint64_t io_bytes = 0;

        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            bool is_compressed = (field_comp_methods[fi] != CompressionMethod::NONE);
            size_t io_offset = 0;
            for (size_t pg : ba) {
                size_t local_pg = pg - pg_base;
                BamBulkReadDesc desc{};
                if (is_compressed) {
                    uint64_t byte_offset = h_comp_offsets[fi][pg];
                    uint64_t page_id = field_start_page_ids[fi] + pg;
                    uint32_t dev = page_id % n_devices;
                    desc.lba = ds.partition_start_lbas[dev] + byte_offset / 512;
                    uint32_t comp_sz = h_comp_sizes[fi][pg];
                    desc.nblocks = bam_safe_nblocks((roundup4096(comp_sz) + 511) / 512);
                    desc.device = dev;
                    desc.dest = static_cast<char*>(io_buf[fi]) + io_offset;
                    desc.copy_bytes = comp_sz;
                    io_offset += page_size;
                } else {
                    uint64_t page_id = field_start_page_ids[fi] + pg;
                    uint32_t dev = page_id % n_devices;
                    uint64_t local_pg_dev = page_id / n_devices;
                    desc.lba = ds.partition_start_lbas[dev] + local_pg_dev * blocks_per_page;
                    desc.nblocks = blocks_per_page;
                    desc.device = dev;
                    desc.dest = static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size;
                    desc.copy_bytes = page_size;
                }
                io_bytes += static_cast<uint64_t>(desc.nblocks) * 512;
                io_ctx.h_descs[0][ndescs++] = desc;
            }
        }
        return {ndescs, io_bytes};
    };

    // Lambda: decompress all fields from io_buf[fi] → data_buf[buf][fi]
    auto run_tile_decomp = [&](size_t pipe_idx, int buf) {
        if (!any_compressed) return;
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        auto& ba = batch_actives[batch_idx];
        size_t slot_base = pipe_idx * BATCH_PAGES;

        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            size_t decomp_count = 0;
            for (size_t idx = 0; idx < ba.size(); idx++) {
                size_t pg = ba[idx];
                size_t local_pg = pg - pg_base;
                uint32_t comp_sz = h_comp_sizes[fi][pg];
                if (comp_sz < page_size) {
                    pb_h_comp_ptrs[fi][slot_base + decomp_count] =
                        static_cast<char*>(io_buf[fi]) + idx * page_size;
                    pb_h_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                    pb_h_decomp_ptrs[fi][slot_base + decomp_count] =
                        static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size;
                    pb_h_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                    decomp_count++;
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size,
                        static_cast<char*>(io_buf[fi]) + idx * page_size,
                        page_size, cudaMemcpyDeviceToDevice, stream_comp));
                }
            }
            if (decomp_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_comp_ptrs,
                    pb_h_comp_ptrs[fi] + slot_base,
                    decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_decomp_ptrs,
                    pb_h_decomp_ptrs[fi] + slot_base,
                    decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_comp_sizes,
                    pb_h_comp_sizes[fi] + slot_base,
                    decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                CUDA_CHECK(cudaMemcpyAsync(nvctx[fi].d_decomp_sizes,
                    pb_h_decomp_sizes[fi] + slot_base,
                    decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                Gidp::nvcomp_decompctx_run(field_comp_methods[fi], nvctx[fi],
                                           decomp_count, page_size, stream_comp,
                                           /*do_sync=*/false, /*skip_h2d=*/true);
                s_kernel_launches++;
            }
        }
    };

    // Lambda: zero inactive page headers + Revenue scan
    auto run_scan = [&](size_t pipe_idx, int buf) {
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        size_t batch_np = std::min(BATCH_PAGES, npages - pg_base);
        auto& ba = batch_actives[batch_idx];

        bool batch_partial = use_zonemap && ba.size() < batch_np;
        if (batch_partial) {
            std::vector<bool> pg_active(batch_np, false);
            for (size_t pg : ba) pg_active[pg - pg_base] = true;
            for (size_t lp = 0; lp < batch_np; lp++) {
                if (!pg_active[lp]) {
                    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                        CUDA_CHECK(cudaMemsetAsync(
                            static_cast<char*>(data_buf[buf][fi]) + lp * page_size,
                            0, 12, stream_comp));
                }
            }
        }

        q6_col_vardate(
            data_buf[buf][L_SHIPDATE_IDX],
            data_buf[buf][L_QUANTITY_IDX],
            data_buf[buf][L_DISCOUNT_IDX],
            data_buf[buf][L_EXTENDEDPRICE_IDX],
            batch_np, page_size, (uint64_t)batch_np * capacity,
            d_revenue, stream_comp,
            options.q6_sd_low, options.q6_sd_high,
            options.disable_other_filters ? 0 : 5,
            options.disable_other_filters ? INT32_MAX : 7,
            options.disable_other_filters ? options.revenue_qt_max : 24);
        s_kernel_launches++;
    };

    // Pipeline events
    cudaEvent_t ev_io_done, ev_decomp_done;
    CUDA_CHECK(cudaEventCreate(&ev_io_done));
    CUDA_CHECK(cudaEventCreate(&ev_decomp_done));
    std::vector<cudaEvent_t> event_comp_vec(num_batches_all);
    for (size_t i = 0; i < num_batches_all; i++)
        CUDA_CHECK(cudaEventCreate(&event_comp_vec[i]));

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream_io);
    }

    // ════════════════════════════════════════════
    // total_start — pipelined execution
    // ════════════════════════════════════════════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;
    uint64_t total_io_count = 0;
    uint64_t total_io_bytes = 0;

    // ── Zone map GPU eval (fused IO + eval, single kernel) ──
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream_io);
        CUDA_CHECK(cudaStreamSynchronize(stream_io));
        s_kernel_launches++;
        for (size_t pg = 0; pg < npages; pg++)
            if (zm_ctx.h_mask[pg]) active_pages.push_back(pg);

        std::cout << "[ZONEMAP] L_SHIPDATE pruning: active="
                  << active_pages.size() << "/" << npages << std::endl;

        // Populate batch_actives
        {
            size_t ap_idx = 0;
            for (size_t b = 0; b < num_batches_all; b++) {
                size_t pg_base = b * BATCH_PAGES;
                size_t pg_end = std::min(pg_base + BATCH_PAGES, npages);
                while (ap_idx < active_pages.size() && active_pages[ap_idx] < pg_base)
                    ap_idx++;
                for (size_t i = ap_idx; i < active_pages.size() && active_pages[i] < pg_end; i++)
                    batch_actives[b].push_back(active_pages[i]);
                if (!batch_actives[b].empty())
                    non_empty_batches.push_back(b);
            }
            num_batches = non_empty_batches.size();
        }
    }

    std::cout << "[GIDP+BAM Revenue] Pipeline: NPIPE=" << NPIPE
              << " BATCH_PAGES=" << BATCH_PAGES
              << " batches=" << num_batches << "/" << num_batches_all << std::endl;

    for (size_t tile = 0; tile < num_batches; tile++) {
        int buf = tile % NPIPE;

        if (tile >= (size_t)NPIPE)
            CUDA_CHECK(cudaStreamWaitEvent(stream_io, event_comp_vec[tile - NPIPE]));
        if (tile > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_io, ev_decomp_done));
            CUDA_CHECK(cudaEventSynchronize(io_ctx.h2d_done[0]));
        }

        auto [ndescs, io_bytes] = build_tile_descs(tile, buf);
        bam_bulk_read_async(io_ctx, ndescs, 0, stream_io);
        s_kernel_launches++;
        CUDA_CHECK(cudaEventRecord(ev_io_done, stream_io));
        total_io_count += ndescs;
        total_io_bytes += io_bytes;

        CUDA_CHECK(cudaStreamWaitEvent(stream_comp, ev_io_done));
        run_tile_decomp(tile, buf);
        CUDA_CHECK(cudaEventRecord(ev_decomp_done, stream_comp));

        run_scan(tile, buf);
        CUDA_CHECK(cudaEventRecord(event_comp_vec[tile], stream_comp));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_comp));

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

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── Cleanup ──
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            if (field_comp_methods[fi] != CompressionMethod::NONE)
                Gidp::nvcomp_decompctx_free(nvctx[fi]);
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (pb_h_comp_ptrs[fi])    cudaFreeHost(pb_h_comp_ptrs[fi]);
            if (pb_h_decomp_ptrs[fi])  cudaFreeHost(pb_h_decomp_ptrs[fi]);
            if (pb_h_comp_sizes[fi])   cudaFreeHost(pb_h_comp_sizes[fi]);
            if (pb_h_decomp_sizes[fi]) cudaFreeHost(pb_h_decomp_sizes[fi]);
        }
    }
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            mb_cuda_free(data_buf[p][fi]);
    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
        if (io_buf[fi]) mb_cuda_free(io_buf[fi]);
    mb_cuda_free(d_revenue);
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    bam_bulk_read_ctx_destroy(io_ctx);
    CUDA_CHECK(cudaEventDestroy(ev_io_done));
    CUDA_CHECK(cudaEventDestroy(ev_decomp_done));
    for (size_t i = 0; i < num_batches_all; i++)
        CUDA_CHECK(cudaEventDestroy(event_comp_vec[i]));
    CUDA_CHECK(cudaStreamDestroy(stream_io));
    CUDA_CHECK(cudaStreamDestroy(stream_comp));
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

BenchmarkResult tpch_q3sel(BenchmarkOptions &options) {
    return tpch_q3(options);
}

} // namespace GidpBam
