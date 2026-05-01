#pragma once

// ============================================================
// BAM TPC-H Q6 — host-side entry point (C++20, no BAM headers)
//
// Metadata loading runs on CPU using existing project utilities.
// The actual BAM kernel launch is delegated to bam_kernel.cu
// (compiled as a separate C++11 CUDA target).
// ============================================================

#include <cstring>
#include <set>
#include <cub/device/device_merge_sort.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "common/common.cu"
#include "common/page.cu"
#include "common/primitive_c.cu"
#include "common/primitive_cuda.cu"
#include "metadata/metadata.h"
#include "schema/tpch_tables.cuh"
#include "tpch/bam_kernel.cuh"
#include "tpch/bam_vchar_kernel.cuh"
#include "tpch/bam_q1_kernel.cuh"
#include "tpch/bam_q3_kernel.cuh"
#include "tpch/bam_q13_kernel.cuh"
#include "tpch/bam_q13_fsst_kernel.cuh"
#include "tpch/bam_q16_kernel.cuh"
#include "tpch/bam_q16_fsst_kernel.cuh"
#include "tpch/bam_bulk_read.cuh"
#include "kernel/tpch/q1.cuh"
#include "kernel/tpch/q3.cuh"
#include "kernel/tpch/q5.cuh"
#include "kernel/tpch/q6_scan.cuh"
#include "kernel/tpch/revenue_scan.cuh"
#include "kernel/tpch/q13.cuh"
#include "kernel/tpch/q16.cuh"
#include "tpch/bam_lz4_fused_q5_dim.cuh"

namespace DataPathFusion {

static constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;

static size_t s_kernel_launches;

// Result struct for PiG query functions (carries pre-computed I/O stats).
struct PigResult {
    double      elapsed_ms;
    uint64_t    nios;              // total NVMe read calls (pre-computed from metadata)
    uint64_t    read_bytes;        // total bytes read from NVMe (pre-computed from metadata)
    std::string compression;       // e.g. "PFOR+LZ4PAR"
    uint64_t    gpu_mem_bytes;     // total GPU memory consumed (ctrl + app)
    uint64_t    gpu_ctrl_bytes;    // GPU memory consumed by BAM ctrl/QPs
    uint64_t    gpu_app_bytes;     // GPU memory consumed by application buffers + page caches
    uint64_t    total_pages;       // total page reads across all columns, before IO pruning
    uint64_t    kernel_launches;   // number of host-side CUDA kernel launches
};

// Collect unique compression method names from a list of uint16_t comp codes.
static std::string collect_comp_methods(std::initializer_list<uint16_t> codes) {
    std::set<uint16_t> seen;
    for (auto c : codes) seen.insert(c);
    std::string result;
    for (auto c : seen) {
        if (!result.empty()) result += '+';
        result += compression_method_name(static_cast<CompressionMethod>(c));
    }
    return result;
}

// Host-side page access helpers (for REGION/NATION host reads in Q5)
static uint32_t bam_host_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}
static const char *bam_host_pagcol_char_data(const char *page, uint32_t slotid, uint32_t padded_len) {
    return page + 12 /* sizeof(pag_head) */ + padded_len * slotid;
}

// ============================================================
// Detect partition 1 start LBA by reading GPT/MBR from LBA 0.
// BAM accesses the raw NVMe namespace; data written to a
// partition needs this offset added to all LBA calculations.
// Returns 0 if no partition table is found.
// ============================================================
static uint64_t detect_partition_start_lba(bam_ctrl_handle_t ctrl) {
    const uint64_t read_size = 4096;
    std::vector<char> buf(read_size);
    int rc = bam_read_page(ctrl, read_size, 0, buf.data());
    if (rc != 0) {
        std::cerr << "Warning: failed to read LBA 0 for partition detection" << std::endl;
        return 0;
    }

    // Check for GPT: "EFI PART" signature at byte 512 (LBA 1)
    if (memcmp(buf.data() + 512, "EFI PART", 8) == 0) {
        // GPT header at byte 512; partition_entry_start_lba at offset 72
        uint64_t pe_start_lba;
        memcpy(&pe_start_lba, buf.data() + 512 + 72, sizeof(uint64_t));
        uint64_t pe_byte_off = pe_start_lba * 512;
        if (pe_byte_off + 128 <= read_size) {
            // First partition entry; starting_lba at offset 32
            uint64_t start_lba;
            memcpy(&start_lba, buf.data() + pe_byte_off + 32, sizeof(uint64_t));
            if (start_lba > 0 && start_lba < 0xFFFFFFFF) {
                std::cout << "GPT detected: partition 1 start LBA = " << start_lba << std::endl;
                return start_lba;
            }
        }
    }

    // Check for MBR: 0x55AA at byte 510
    if (static_cast<uint8_t>(buf[510]) == 0x55 &&
        static_cast<uint8_t>(buf[511]) == 0xAA) {
        // MBR partition entry 0 at byte 446; LBA start at offset 8
        uint32_t start_lba;
        memcpy(&start_lba, buf.data() + 446 + 8, sizeof(uint32_t));
        if (start_lba > 0) {
            std::cout << "MBR detected: partition 1 start LBA = " << start_lba << std::endl;
            return static_cast<uint64_t>(start_lba);
        }
    }

    std::cout << "No partition table detected, using LBA offset 0" << std::endl;
    return 0;
}

// ============================================================
// Common multi-device setup: parse paths, open controller(s),
// detect per-device partition offsets.
// ============================================================
struct BAMDeviceSetup {
    bam_ctrl_handle_t ctrl;
    uint32_t n_devices;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint64_t partition_start_lba;  // = partition_start_lbas[0]
};

static BAMDeviceSetup bam_open_devices(const char* file, uint32_t num_queues,
                                        uint32_t queue_depth) {
    BAMDeviceSetup s{};

    // 1. Parse comma-separated device paths
    std::string file_copy(file);
    std::vector<std::string> dev_paths_storage;
    std::vector<const char*> dev_paths;
    {
        char* tok = strtok(&file_copy[0], ",");
        while (tok) {
            dev_paths_storage.emplace_back(tok);
            tok = strtok(nullptr, ",");
        }
        for (auto& str : dev_paths_storage) dev_paths.push_back(str.c_str());
    }
    s.n_devices = static_cast<uint32_t>(dev_paths.size());

    for (uint32_t d = 0; d < s.n_devices; d++)
        std::cout << "BAM controller[" << d << "]: " << dev_paths[d] << std::endl;

    // 2. Open controller(s)
    if (s.n_devices == 1) {
        s.ctrl = bam_ctrl_open(dev_paths[0], 1, 0, queue_depth, num_queues);
    } else {
        s.ctrl = bam_ctrl_open_multi(dev_paths.data(), s.n_devices, 1, 0,
                                      queue_depth, num_queues);
    }

    // 3. Detect per-device partition offsets (GPT/MBR)
    for (uint32_t d = 0; d < s.n_devices; d++) {
        const uint64_t det_size = 4096;
        std::vector<char> det_buf(det_size);
        int drc = bam_read_page(s.ctrl, det_size, 0, det_buf.data(), d);
        if (drc != 0) {
            std::cerr << "Warning: failed to read LBA 0 from device " << d << std::endl;
            s.partition_start_lbas[d] = 0;
            continue;
        }
        // GPT detection
        if (memcmp(det_buf.data() + 512, "EFI PART", 8) == 0) {
            uint64_t pe_start_lba;
            memcpy(&pe_start_lba, det_buf.data() + 512 + 72, sizeof(uint64_t));
            uint64_t pe_byte_off = pe_start_lba * 512;
            if (pe_byte_off + 128 <= det_size) {
                uint64_t start_lba;
                memcpy(&start_lba, det_buf.data() + pe_byte_off + 32, sizeof(uint64_t));
                if (start_lba > 0 && start_lba < 0xFFFFFFFF) {
                    std::cout << "Device " << d << ": GPT partition 1 start LBA = "
                              << start_lba << std::endl;
                    s.partition_start_lbas[d] = start_lba;
                    continue;
                }
            }
        }
        // MBR detection
        if (static_cast<uint8_t>(det_buf[510]) == 0x55 &&
            static_cast<uint8_t>(det_buf[511]) == 0xAA) {
            uint32_t start_lba;
            memcpy(&start_lba, det_buf.data() + 446 + 8, sizeof(uint32_t));
            if (start_lba > 0) {
                std::cout << "Device " << d << ": MBR partition 1 start LBA = "
                          << start_lba << std::endl;
                s.partition_start_lbas[d] = static_cast<uint64_t>(start_lba);
                continue;
            }
        }
        std::cout << "Device " << d << ": No partition table, LBA offset 0" << std::endl;
        s.partition_start_lbas[d] = 0;
    }

    s.partition_start_lba = s.partition_start_lbas[0];
    return s;
}

BenchmarkResult tpch_q6(BenchmarkOptions& options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    // ── 2. Open BAM controller(s) and detect partition offsets ──
    size_t gpu_free_pre_ctrl = 0, gpu_total_mem = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_mem);

    const uint32_t bam_num_queues = 128;
    auto ds = bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_mem);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;
    const uint32_t n_devices = ds.n_devices;
    const uint64_t partition_start_lba = ds.partition_start_lba;

    // Helper: read a page given global page ID (handles striping)
    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata page via BAM ──
    // 3a. Read header (4096 bytes) to learn page_size
    // Header = page 0 → always on device 0
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

    // 4b. Read full metadata page (page 0, device 0)
    std::vector<char> meta_buf(page_size);
    rc = read_striped_page(0, page_size, meta_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (full metadata)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    std::cout << std::dec << "=== TPCH Table Metadata ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;

    // ── 5. Extract Q6 field info directly from metadata ──
    constexpr size_t NUM_FIELDS = TPCH::Query::Q6::NUM_SCAN_TARGET_COLS;
    auto q6_cols = TPCH::Query::Q6::SCAN_TARGET_COLS;

    uint64_t field_start_page_ids[NUM_FIELDS];
    uint64_t field_npages_arr[NUM_FIELDS];
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        size_t col = q6_cols[fi];
        field_start_page_ids[fi] = metadata.table_lineitem_start_page_ids[col];
        field_npages_arr[fi] = metadata.table_lineitem_npages[col];
        std::cout << "  Field " << col
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << metadata.table_lineitem_compression_method[col]
                  << std::endl;
    }

    const uint64_t npages = field_npages_arr[0];
    for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
        if (field_npages_arr[fi] != npages) {
            std::cerr << "Error: field " << fi << " has different npages ("
                      << field_npages_arr[fi] << " vs " << npages << ")" << std::endl;
            bam_ctrl_close(ctrl);
            exit(EXIT_FAILURE);
        }
    }

    if (npages == 0) {
        std::cout << "No pages to read." << std::endl;
        bam_ctrl_close(ctrl);
        return BenchmarkResult{};
    }

    // ── 5b. Read prefix sum and compression metadata (vector-based, Q5 pattern) ──
    const uint64_t nrows = metadata.table_lineitem_nrows;
    const uint64_t blocks_per_page = page_size / 512;

    auto read_prefix_sum_host = [&](uint64_t ps_start, uint64_t ps_npages_cnt,
                                     uint64_t field_npages_cnt) -> std::vector<uint64_t>
    {
        if (ps_npages_cnt == 0) return {};
        std::vector<char> h_buf(ps_npages_cnt * page_size);
        for (uint64_t p = 0; p < ps_npages_cnt; p++) {
            rc = read_striped_page(ps_start + p, page_size, h_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (prefix sum page=" << p << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        // Skip leading 0 element: raw layout is [0, ps[1], ps[2], ...]
        uint64_t* ps_raw = reinterpret_cast<uint64_t*>(h_buf.data()) + 1;
        return std::vector<uint64_t>(ps_raw, ps_raw + field_npages_cnt);
    };

    auto prepare_comp_metadata = [&](
        uint64_t field_start, uint64_t field_npages_cnt, uint16_t comp_method,
        uint64_t cs_start_page, uint64_t cs_npages_cnt,
        uint64_t nbase_val, uint64_t base_start_page)
        -> std::pair<std::vector<uint32_t>, std::vector<uint64_t>>
    {
        if (comp_method == 0) return {{}, {}};
        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start_page + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_sizes page=" << p << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<uint32_t> comp_sizes(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages_cnt);
        size_t bp_npages = TPCH::nbase_to_npages(nbase_val, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start_page + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_bases page=" << p << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            comp_sizes.data(), nbase_val, field_npages_cnt, page_size, field_start,
            n_devices, offsets_vec);
        std::vector<uint64_t> comp_offsets(field_npages_cnt);
        for (uint64_t i = 0; i < field_npages_cnt; i++)
            comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
        return {comp_sizes, comp_offsets};
    };

    // Read prefix sums and compression metadata for all 4 fields
    uint16_t comp_methods[NUM_FIELDS] = {};
    std::vector<uint64_t> h_ps[NUM_FIELDS];
    std::vector<uint32_t> h_cs[NUM_FIELDS];
    std::vector<uint64_t> h_co[NUM_FIELDS];

    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        size_t col = q6_cols[fi];
        comp_methods[fi] = metadata.table_lineitem_compression_method[col];

        h_ps[fi] = read_prefix_sum_host(
            metadata.table_lineitem_prefix_sum_start_page_ids[col],
            metadata.table_lineitem_prefix_sum_npages[col],
            npages);

        std::cout << "  Field " << col << " prefix_sum: total_rows=" << h_ps[fi].back()
                  << " compression=" << comp_methods[fi] << std::endl;

        auto [cs, co] = prepare_comp_metadata(
            field_start_page_ids[fi], npages, comp_methods[fi],
            metadata.table_lineitem_compressed_page_sizes_start_page_ids[col],
            metadata.table_lineitem_compressed_page_sizes_npages[col],
            metadata.table_lineitem_compression_nbases[col],
            metadata.table_lineitem_compression_base_start_page_ids[col]);
        h_cs[fi] = std::move(cs);
        h_co[fi] = std::move(co);
    }
    std::cout << std::dec << "  nrows=" << nrows << std::endl;

    // ── 5b-diag. Verify page headers (nalloc) vs prefix_sum ──
    {
        struct DiagHdr { uint32_t nalloc; uint32_t watermark; uint32_t lfreespace; };
        std::vector<char> diag_page(page_size);
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            size_t col = q6_cols[fi];
            uint64_t pg0_global = field_start_page_ids[fi];
            rc = read_striped_page(pg0_global, page_size, diag_page.data());
            if (rc != 0) {
                std::cerr << "[DIAG] Failed to read page 0 of field " << col << std::endl;
                continue;
            }
            DiagHdr hdr;
            std::memcpy(&hdr, diag_page.data(), sizeof(hdr));
            uint64_t ps_nrows = h_ps[fi][0];
            const int32_t* data = reinterpret_cast<const int32_t*>(diag_page.data() + sizeof(DiagHdr));
            std::cout << "[DIAG] Field " << col
                      << " page0: nalloc=" << hdr.nalloc
                      << " watermark=" << hdr.watermark
                      << " ps_nrows=" << ps_nrows
                      << " match=" << (hdr.nalloc == ps_nrows ? "YES" : "NO")
                      << std::endl;
            std::cout << "[DIAG]   data[0..4]:";
            for (int k = 0; k < 5 && (uint32_t)k < hdr.nalloc; k++)
                std::cout << " " << data[k];
            std::cout << std::endl;
        }
        // Last page of field 0
        {
            size_t col = q6_cols[0];
            uint64_t last_pg_global = field_start_page_ids[0] + npages - 1;
            rc = read_striped_page(last_pg_global, page_size, diag_page.data());
            if (rc == 0) {
                DiagHdr hdr;
                std::memcpy(&hdr, diag_page.data(), sizeof(hdr));
                uint64_t ps_nrows_last = (npages == 1) ? h_ps[0][0]
                                       : h_ps[0][npages - 1] - h_ps[0][npages - 2];
                std::cout << "[DIAG] Field " << col
                          << " last_page(" << npages - 1 << "): nalloc=" << hdr.nalloc
                          << " ps_nrows=" << ps_nrows_last
                          << " match=" << (hdr.nalloc == ps_nrows_last ? "YES" : "NO")
                          << std::endl;
            }
        }
        // Page 1: check alignment across fields
        if (npages > 1) {
            std::cout << "[DIAG] Page 1 cross-field alignment check:" << std::endl;
            for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
                size_t col = q6_cols[fi];
                rc = read_striped_page(field_start_page_ids[fi] + 1, page_size, diag_page.data());
                if (rc != 0) continue;
                DiagHdr hdr1;
                std::memcpy(&hdr1, diag_page.data(), sizeof(hdr1));
                const int32_t* data1 = reinterpret_cast<const int32_t*>(diag_page.data() + sizeof(DiagHdr));
                uint64_t ps_nrows1 = h_ps[fi][1] - h_ps[fi][0];
                std::cout << "[DIAG]   Field " << col
                          << " page1: nalloc=" << hdr1.nalloc
                          << " ps_nrows=" << ps_nrows1
                          << " data[0..2]:";
                for (int k = 0; k < 3 && (uint32_t)k < hdr1.nalloc; k++)
                    std::cout << " " << data1[k];
                std::cout << std::endl;
            }
        }
        // Prefix sum total check
        uint64_t ps_total = h_ps[0][npages - 1];
        std::cout << "[DIAG] prefix_sum total=" << ps_total
                  << " metadata nrows=" << nrows
                  << " match=" << (ps_total == nrows ? "YES" : "NO")
                  << std::endl;
    }

    // ── 5b-2. Detect per-field prefix_sum mismatch ──
    bool ps_mismatch = false;
    for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
        if (h_ps[fi] != h_ps[0]) { ps_mismatch = true; break; }
    }
    if (ps_mismatch) {
        std::cout << "[Q6] Per-field prefix_sum MISMATCH — using flatten+scan path" << std::endl;
        std::cout << "[Q6]   h_ps[0][0]=" << h_ps[0][0];
        for (size_t fi = 1; fi < NUM_FIELDS; fi++)
            std::cout << " h_ps[" << fi << "][0]=" << h_ps[fi][0];
        std::cout << std::endl;
    }

    // ── 5c. Zone map metadata (Rule 3: metadata outside timing) ──
    bool use_zonemap = options.enable_zonemap;
    uint64_t zm_sd_nstats = 0, zm_sd_stats_start = 0, zm_sd_stats_npg = 0;
    if (use_zonemap) {
        size_t sd_col = q6_cols[0];  // L_SHIPDATE
        zm_sd_nstats      = metadata.table_lineitem_nstats[sd_col];
        zm_sd_stats_start = metadata.table_lineitem_stats_start_page_ids[sd_col];
        zm_sd_stats_npg   = metadata.table_lineitem_stats_npages[sd_col];
        if (zm_sd_nstats == 0 || zm_sd_stats_start == 0) {
            std::cout << "[ZONEMAP] No stats available for L_SHIPDATE, processing all pages." << std::endl;
            use_zonemap = false;
        }
    }

    const int32_t sd_low  = options.q6_sd_low  ? options.q6_sd_low  : 19940101;
    const int32_t sd_high = options.q6_sd_high ? options.q6_sd_high : 19950101;

    // ── 6a. IO byte accounting helper ──
    uint64_t total_read_bytes = 0;
    auto q6_col_page_bytes = [&](size_t fi, uint64_t j) -> uint64_t {
        return (comp_methods[fi] != 0 && j < h_cs[fi].size()) ? h_cs[fi][j] : page_size;
    };
    if (!use_zonemap) {
        for (uint64_t j = 0; j < npages; j++)
            for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                total_read_bytes += q6_col_page_bytes(fi, j);
    }

    // ── 6b. Fused kernel setup ──
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const uint32_t safe_blocks = bam_num_queues;
    const uint32_t num_blocks_fused = std::min({static_cast<uint32_t>(sm_count),
                                                static_cast<uint32_t>(npages),
                                                n_devices * safe_blocks});

    // Compute max rows per page (for scratch buffer sizing)
    uint64_t max_rows_per_page = 0;
    for (uint64_t i = 0; i < npages; i++) {
        uint64_t nr = (i == 0) ? h_ps[0][0] : h_ps[0][i] - h_ps[0][i - 1];
        max_rows_per_page = std::max(max_rows_per_page, nr);
    }
    std::cout << "[Q6] Fused kernel: " << num_blocks_fused << " blocks, "
              << npages << " pages, max_rows_per_page=" << max_rows_per_page << std::endl;

    // ── 7. GPU memory allocation ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_mem);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // BamZonemapCtx (Rule 4: alloc outside timing; Rule 3: metadata outside timing)
    bam_pfor32_io_ctx_t pfor_ctx_zm = nullptr;
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (use_zonemap && zm_sd_nstats > 0 && zm_sd_stats_npg > 0) {
        pfor_ctx_zm = bam_pfor32_io_create(ctrl, static_cast<uint32_t>(page_size), kBamZonemapMaxReads);
        zm_ctx = bam_zonemap_ctx_create(
            bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
            bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
            (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
            static_cast<uint32_t>(page_size), npages);
        for (uint64_t j = 0; j < zm_sd_stats_npg; j++) {
            uint64_t pg_id = zm_sd_stats_start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            zm_ctx.h_reads[zm_nreads++] = {
                ds.partition_start_lbas[dev] + local * blocks_per_page,
                static_cast<uint32_t>(blocks_per_page), dev};
        }
        zm_ctx.h_preds[zm_npreds++] = {0, zm_sd_nstats, sd_low, sd_high - 1};
        zm_valid = true;
    }

    int64_t* d_revenue = nullptr;
    cudaMalloc(&d_revenue, sizeof(int64_t));
    cudaMemset(d_revenue, 0, sizeof(int64_t));

    // Compression metadata on GPU (shared by both paths)
    uint32_t* d_comp_sizes[NUM_FIELDS] = {};
    uint64_t* d_comp_offsets[NUM_FIELDS] = {};
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (comp_methods[fi] != 0 && !h_cs[fi].empty()) {
            cudaMalloc(&d_comp_sizes[fi], npages * sizeof(uint32_t));
            cudaMemcpy(d_comp_sizes[fi], h_cs[fi].data(),
                       npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMalloc(&d_comp_offsets[fi], npages * sizeof(uint64_t));
            cudaMemcpy(d_comp_offsets[fi], h_co[fi].data(),
                       npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
    }

    int64_t h_revenue = 0;
    int64_t elapsed_ns = 0;

    if (ps_mismatch) {
        // ── 7a. Flatten path: per-field flatten → revenue_scan_flat ──
        // Page cache: num_blocks entries (1 page per block per flatten call)
        bam_pfor32_io_ctx_t pfor32_ctx = bam_pfor32_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks_fused);

        // Per-field prefix_sum arrays on GPU
        uint64_t* d_ps[NUM_FIELDS] = {};
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            cudaMalloc(&d_ps[fi], npages * sizeof(uint64_t));
            cudaMemcpy(d_ps[fi], h_ps[fi].data(), npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }

        // Flat output arrays: nrows × int32_t per field
        int32_t* d_flat[NUM_FIELDS] = {};
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            cudaMalloc(&d_flat[fi], nrows * sizeof(int32_t));

        // Pre-allocate per-field zone map mask buffers (Rule 4: alloc outside timing)
        uint8_t* d_page_active_fi[NUM_FIELDS] = {};
        if (zm_valid) {
            for (size_t fi = 1; fi < NUM_FIELDS; fi++)
                cudaMalloc(&d_page_active_fi[fi], npages);
        }

        size_t gpu_free_after_app = 0;
        cudaMemGetInfo(&gpu_free_after_app, &gpu_total_mem);
        uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

        // Pre-issue IO to initialize BaM page_cache DMA registration
        if (zm_valid) {
            bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
        }

        auto total_start = std::chrono::steady_clock::now();
        s_kernel_launches = 0;

        // GPU zonemap eval (Rule 6: IO + eval inside timing, mask stays on GPU)
        if (zm_valid) {
            bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
            cudaStreamSynchronize(stream);
            s_kernel_launches++;

            // IO byte accounting from h_mask
            for (uint64_t j = 0; j < npages; j++) {
                if (zm_ctx.h_mask[j]) {
                    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                        total_read_bytes += q6_col_page_bytes(fi, j);
                }
            }

            // Derive per-field masks from h_mask
            d_page_active_fi[0] = zm_ctx.d_mask;
            for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
                std::vector<uint8_t> h_mask_fi(npages, 0);
                for (uint64_t pg0 = 0; pg0 < npages; pg0++) {
                    if (!zm_ctx.h_mask[pg0]) continue;
                    uint64_t row_lo = (pg0 == 0) ? 0 : h_ps[0][pg0 - 1];
                    uint64_t row_hi = h_ps[0][pg0];
                    auto it_lo = std::upper_bound(h_ps[fi].begin(), h_ps[fi].end(), row_lo);
                    uint64_t pg_lo = static_cast<uint64_t>(it_lo - h_ps[fi].begin());
                    auto it_hi = std::lower_bound(h_ps[fi].begin(), h_ps[fi].end(), row_hi);
                    uint64_t pg_hi = static_cast<uint64_t>(it_hi - h_ps[fi].begin());
                    for (uint64_t pg = pg_lo; pg <= pg_hi && pg < npages; pg++)
                        h_mask_fi[pg] = 1;
                }
                cudaMemcpy(d_page_active_fi[fi], h_mask_fi.data(), npages, cudaMemcpyHostToDevice);
            }
        }

        // Flatten each field with its own prefix_sum and per-field zone map mask
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            BAMPfor32FlattenParams fp{};
            fp.partition_start_lba = partition_start_lba;
            for (uint32_t d = 0; d < n_devices; d++)
                fp.partition_start_lbas[d] = ds.partition_start_lbas[d];
            fp.n_devices = n_devices;
            fp.page_size = static_cast<uint32_t>(page_size);
            fp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
            fp.comp_method = comp_methods[fi];
            fp.field_start_page_id = field_start_page_ids[fi];
            fp.npages = npages;
            fp.nrows = nrows;
            fp.num_blocks = num_blocks_fused;
            fp.d_prefix_sum = d_ps[fi];
            fp.d_comp_sizes = d_comp_sizes[fi];
            fp.d_comp_offsets = d_comp_offsets[fi];

            bam_pfor32_io_flatten_masked_async(
                pfor32_ctx, fp, d_page_active_fi[fi],
                /*fill_value=*/0, d_flat[fi], stream);
            s_kernel_launches++;
            cudaStreamSynchronize(stream);
        }

        // Revenue scan on flattened arrays (field order: shipdate, quantity, discount, extprice)
        // Q6 fields: [0]=L_SHIPDATE, [1]=L_QUANTITY, [2]=L_EXTENDEDPRICE, [3]=L_DISCOUNT
        revenue_scan_flat(
            d_flat[0], d_flat[1], d_flat[3], d_flat[2],
            nrows, d_revenue, sd_low, sd_high,
            /*qt_max=*/2400, stream,
            /*dc_low=*/5, /*dc_high=*/7);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);

        cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost);

        auto total_end = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

        std::cout << std::dec << "TPCH Q6 total revenue: " << h_revenue << std::endl;

        // Cleanup flatten path
        bam_pfor32_io_destroy(pfor32_ctx);
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            cudaFree(d_ps[fi]);
            cudaFree(d_flat[fi]);
        }
        // Free per-field zone map masks (fi=0 is zm_ctx.d_mask, freed by ctx_destroy)
        for (size_t fi = 1; fi < NUM_FIELDS; fi++)
            if (d_page_active_fi[fi]) cudaFree(d_page_active_fi[fi]);

        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (d_comp_sizes[fi]) cudaFree(d_comp_sizes[fi]);
            if (d_comp_offsets[fi]) cudaFree(d_comp_offsets[fi]);
        }
        cudaFree(d_revenue);
        if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
        if (pfor_ctx_zm) bam_pfor32_io_destroy(pfor_ctx_zm);
        cudaStreamDestroy(stream);
        bam_ctrl_close(ctrl);

        return BenchmarkResult{
            .read_bytes = total_read_bytes,
            .elapsed_nanoseconds = elapsed_ns,
            .compression = collect_comp_methods({comp_methods[0], comp_methods[1],
                                                 comp_methods[2], comp_methods[3]}),
            .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
            .gpu_ctrl_bytes = gpu_ctrl_bytes,
            .gpu_app_bytes = gpu_app_bytes,
            .total_pages = npages * NUM_FIELDS,
            .kernel_launches = s_kernel_launches,
        };
    }

    // ── 7b. Fused path (prefix_sums match): existing fused kernel ──
    // Page cache: num_blocks * 4 entries (one per field per block)
    bam_pfor32_io_ctx_t pfor32_ctx = bam_pfor32_io_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks_fused * 4);

    uint64_t* d_prefix_sum = nullptr;
    cudaMalloc(&d_prefix_sum, npages * sizeof(uint64_t));
    cudaMemcpy(d_prefix_sum, h_ps[0].data(), npages * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Per-block scratch buffer: num_blocks * 4 fields * max_rows_per_page int32_t
    int32_t* d_scratch = nullptr;
    uint64_t scratch_total = (uint64_t)num_blocks_fused * 4 * max_rows_per_page;
    cudaMalloc(&d_scratch, scratch_total * sizeof(int32_t));

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_mem);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // ── 8. Build fused params & run ──
    BAMRevenueFusedParams fp{};
    fp.partition_start_lba = partition_start_lba;
    for (uint32_t d = 0; d < n_devices; d++)
        fp.partition_start_lbas[d] = ds.partition_start_lbas[d];
    fp.n_devices = n_devices;
    fp.page_size = static_cast<uint32_t>(page_size);
    fp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
    fp.npages = npages;
    fp.num_blocks = num_blocks_fused;
    fp.sd_low = sd_low;
    fp.sd_high = sd_high;
    fp.qt_max = 2400;
    fp.dc_low = 5;
    fp.dc_high = 7;
    fp.d_prefix_sum = d_prefix_sum;
    fp.d_page_active = nullptr;  // set after zonemap eval
    fp.d_active_page_ids = nullptr;
    fp.num_active_pages = 0;
    fp.d_scratch = d_scratch;
    fp.scratch_stride = static_cast<uint32_t>(max_rows_per_page);
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        fp.field_start_page_ids[fi] = field_start_page_ids[fi];
        fp.comp_methods[fi] = comp_methods[fi];
        fp.d_comp_sizes[fi] = d_comp_sizes[fi];
        fp.d_comp_offsets[fi] = d_comp_offsets[fi];
    }

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    {
        auto total_start = std::chrono::steady_clock::now();
        s_kernel_launches = 0;

        // GPU zonemap eval (Rule 6: IO + eval inside timing, mask stays on GPU)
        if (zm_valid) {
            bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
            cudaStreamSynchronize(stream);
            s_kernel_launches++;

            fp.d_page_active = zm_ctx.d_mask;
            fp.d_active_page_ids = zm_ctx.d_active_ids;
            fp.num_active_pages = *zm_ctx.h_num_active;

            // IO byte accounting from h_mask
            for (uint64_t j = 0; j < npages; j++) {
                if (zm_ctx.h_mask[j]) {
                    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                        total_read_bytes += q6_col_page_bytes(fi, j);
                }
            }
        }

        bam_revenue_fused_run(pfor32_ctx, fp, d_revenue, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);

        cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost);

        auto total_end = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    }

    // ── 9. Result retrieval & stats ──
    std::cout << std::dec << "TPCH Q6 total revenue: " << h_revenue << std::endl;

    // ── 10. Cleanup ──
    bam_pfor32_io_destroy(pfor32_ctx);
    cudaFree(d_prefix_sum);
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (d_comp_sizes[fi]) cudaFree(d_comp_sizes[fi]);
        if (d_comp_offsets[fi]) cudaFree(d_comp_offsets[fi]);
    }
    cudaFree(d_scratch);
    cudaFree(d_revenue);
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    if (pfor_ctx_zm) bam_pfor32_io_destroy(pfor_ctx_zm);
    cudaStreamDestroy(stream);
    bam_ctrl_close(ctrl);

    return BenchmarkResult{
        .read_bytes = total_read_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = collect_comp_methods({comp_methods[0], comp_methods[1],
                                             comp_methods[2], comp_methods[3]}),
        .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NUM_FIELDS,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// Revenue query — same scan plan as Q6, shipdate-only predicate
// Tile execution: PFOR32 flatten + revenue_scan_flat kernel
// ============================================================
BenchmarkResult tpch_revenue(BenchmarkOptions& options) {
    // ── 1. CUDA init ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    // ── 2. Open BAM controller(s) and detect partition offsets ──
    size_t gpu_free_pre_ctrl = 0, gpu_total_mem = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_mem);

    const uint32_t bam_num_queues = 128;
    auto ds = bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_mem);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;
    const uint32_t n_devices = ds.n_devices;
    const uint64_t partition_start_lba = ds.partition_start_lba;

    // Helper: read a page given global page ID (handles striping)
    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // ── 3. Read metadata page via BAM ──
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

    std::cout << std::dec << "=== TPCH Table Metadata ===" << std::endl;
    std::cout << "Page Size: " << metadata.page_size << std::endl;
    std::cout << "nrecs_lineitem: " << metadata.table_lineitem_nrows << std::endl;

    // ── 5. Extract field info (same 4 fields as Q6) ──
    constexpr size_t NUM_FIELDS = TPCH::Query::Q6::NUM_SCAN_TARGET_COLS;
    auto q6_cols = TPCH::Query::Q6::SCAN_TARGET_COLS;

    uint64_t field_start_page_ids[NUM_FIELDS];
    uint64_t field_npages_arr[NUM_FIELDS];
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        size_t col = q6_cols[fi];
        field_start_page_ids[fi] = metadata.table_lineitem_start_page_ids[col];
        field_npages_arr[fi] = metadata.table_lineitem_npages[col];
        std::cout << "  Field " << col
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << metadata.table_lineitem_compression_method[col]
                  << std::endl;
    }

    const uint64_t npages = field_npages_arr[0];
    for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
        if (field_npages_arr[fi] != npages) {
            std::cerr << "Error: field " << fi << " has different npages ("
                      << field_npages_arr[fi] << " vs " << npages << ")" << std::endl;
            bam_ctrl_close(ctrl);
            exit(EXIT_FAILURE);
        }
    }

    if (npages == 0) {
        std::cout << "No pages to read." << std::endl;
        bam_ctrl_close(ctrl);
        return BenchmarkResult{};
    }

    // ── 5b. Read prefix sum and compression metadata (vector-based) ──
    const uint64_t nrows = metadata.table_lineitem_nrows;
    const uint64_t blocks_per_page = page_size / 512;

    auto read_prefix_sum_host = [&](uint64_t ps_start, uint64_t ps_npages_cnt,
                                     uint64_t field_npages_cnt) -> std::vector<uint64_t>
    {
        if (ps_npages_cnt == 0) return {};
        std::vector<char> h_buf(ps_npages_cnt * page_size);
        for (uint64_t p = 0; p < ps_npages_cnt; p++) {
            rc = read_striped_page(ps_start + p, page_size, h_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (prefix sum page=" << p << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        uint64_t* ps_raw = reinterpret_cast<uint64_t*>(h_buf.data()) + 1;
        return std::vector<uint64_t>(ps_raw, ps_raw + field_npages_cnt);
    };

    auto prepare_comp_metadata = [&](
        uint64_t field_start, uint64_t field_npages_cnt, uint16_t comp_method,
        uint64_t cs_start_page, uint64_t cs_npages_cnt,
        uint64_t nbase_val, uint64_t base_start_page)
        -> std::pair<std::vector<uint32_t>, std::vector<uint64_t>>
    {
        if (comp_method == 0) return {{}, {}};
        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            rc = read_striped_page(cs_start_page + p, page_size, sizes_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_sizes page=" << p << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<uint32_t> comp_sizes(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages_cnt);
        size_t bp_npages = TPCH::nbase_to_npages(nbase_val, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            rc = read_striped_page(base_start_page + p, page_size, bases_buf.data() + p * page_size);
            if (rc != 0) {
                std::cerr << "bam_read_page failed (comp_bases page=" << p << ")" << std::endl;
                bam_ctrl_close(ctrl);
                exit(EXIT_FAILURE);
            }
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            comp_sizes.data(), nbase_val, field_npages_cnt, page_size, field_start,
            n_devices, offsets_vec);
        std::vector<uint64_t> comp_offsets(field_npages_cnt);
        for (uint64_t i = 0; i < field_npages_cnt; i++)
            comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
        return {comp_sizes, comp_offsets};
    };

    uint16_t comp_methods[NUM_FIELDS] = {};
    std::vector<uint64_t> h_ps[NUM_FIELDS];
    std::vector<uint32_t> h_cs[NUM_FIELDS];
    std::vector<uint64_t> h_co[NUM_FIELDS];

    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        size_t col = q6_cols[fi];
        comp_methods[fi] = metadata.table_lineitem_compression_method[col];

        h_ps[fi] = read_prefix_sum_host(
            metadata.table_lineitem_prefix_sum_start_page_ids[col],
            metadata.table_lineitem_prefix_sum_npages[col],
            npages);

        std::cout << "  Field " << col << " prefix_sum: total_rows=" << h_ps[fi].back()
                  << " compression=" << comp_methods[fi] << std::endl;

        auto [cs, co] = prepare_comp_metadata(
            field_start_page_ids[fi], npages, comp_methods[fi],
            metadata.table_lineitem_compressed_page_sizes_start_page_ids[col],
            metadata.table_lineitem_compressed_page_sizes_npages[col],
            metadata.table_lineitem_compression_nbases[col],
            metadata.table_lineitem_compression_base_start_page_ids[col]);
        h_cs[fi] = std::move(cs);
        h_co[fi] = std::move(co);
    }
    std::cout << std::dec << "  nrows=" << nrows << std::endl;

    // ── 5b-2. Detect per-field prefix_sum mismatch ──
    bool ps_mismatch = false;
    for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
        if (h_ps[fi] != h_ps[0]) { ps_mismatch = true; break; }
    }
    if (ps_mismatch) {
        std::cout << "[Revenue] Per-field prefix_sum MISMATCH — using flatten+scan path" << std::endl;
        std::cout << "[Revenue]   h_ps[0][0]=" << h_ps[0][0];
        for (size_t fi = 1; fi < NUM_FIELDS; fi++)
            std::cout << " h_ps[" << fi << "][0]=" << h_ps[fi][0];
        std::cout << std::endl;
    }

    // ── 5c. Zone map metadata (Rule 3: metadata outside timing) ──
    bool use_zonemap = options.enable_zonemap;
    uint64_t zm_sd_nstats = 0, zm_sd_stats_start = 0, zm_sd_stats_npg = 0;
    if (use_zonemap) {
        size_t sd_col = q6_cols[0];
        zm_sd_nstats      = metadata.table_lineitem_nstats[sd_col];
        zm_sd_stats_start = metadata.table_lineitem_stats_start_page_ids[sd_col];
        zm_sd_stats_npg   = metadata.table_lineitem_stats_npages[sd_col];
        if (zm_sd_nstats == 0 || zm_sd_stats_start == 0) {
            std::cout << "[ZONEMAP] No stats available for L_SHIPDATE, processing all pages." << std::endl;
            use_zonemap = false;
        }
    }

    const int32_t sd_low  = options.q6_sd_low;
    const int32_t sd_high = options.q6_sd_high;
    const int32_t qt_max  = options.disable_other_filters ? options.revenue_qt_max : 24;

    // ── 6a. IO byte accounting helper ──
    uint64_t total_read_bytes = 0;
    auto col_page_bytes = [&](size_t fi, uint64_t j) -> uint64_t {
        return (comp_methods[fi] != 0 && j < h_cs[fi].size()) ? h_cs[fi][j] : page_size;
    };
    if (!use_zonemap) {
        for (uint64_t j = 0; j < npages; j++)
            for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                total_read_bytes += col_page_bytes(fi, j);
    }

    // ── 6b. Fused kernel setup ──
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    const uint32_t safe_blocks = bam_num_queues;
    const uint32_t num_blocks_fused = std::min({static_cast<uint32_t>(sm_count),
                                                static_cast<uint32_t>(npages),
                                                n_devices * safe_blocks});

    // Compute max rows per page (for scratch buffer sizing)
    uint64_t max_rows_per_page = 0;
    for (uint64_t i = 0; i < npages; i++) {
        uint64_t nr = (i == 0) ? h_ps[0][0] : h_ps[0][i] - h_ps[0][i - 1];
        max_rows_per_page = std::max(max_rows_per_page, nr);
    }
    std::cout << "[Revenue] Fused kernel: " << num_blocks_fused << " blocks, "
              << npages << " pages, max_rows_per_page=" << max_rows_per_page << std::endl;

    // ── 7. GPU memory allocation ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_mem);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // BamZonemapCtx (Rule 4: alloc outside timing; Rule 3: metadata outside timing)
    bam_pfor32_io_ctx_t pfor_ctx_zm = nullptr;
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (use_zonemap && zm_sd_nstats > 0 && zm_sd_stats_npg > 0) {
        pfor_ctx_zm = bam_pfor32_io_create(ctrl, static_cast<uint32_t>(page_size), kBamZonemapMaxReads);
        zm_ctx = bam_zonemap_ctx_create(
            bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
            bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
            (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
            static_cast<uint32_t>(page_size), npages);
        for (uint64_t j = 0; j < zm_sd_stats_npg; j++) {
            uint64_t pg_id = zm_sd_stats_start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            zm_ctx.h_reads[zm_nreads++] = {
                ds.partition_start_lbas[dev] + local * blocks_per_page,
                static_cast<uint32_t>(blocks_per_page), dev};
        }
        zm_ctx.h_preds[zm_npreds++] = {0, zm_sd_nstats, sd_low, sd_high - 1};
        zm_valid = true;
    }

    int64_t* d_revenue = nullptr;
    cudaMalloc(&d_revenue, sizeof(int64_t));
    cudaMemset(d_revenue, 0, sizeof(int64_t));

    // Compression metadata on GPU (shared by both paths)
    uint32_t* d_comp_sizes[NUM_FIELDS] = {};
    uint64_t* d_comp_offsets[NUM_FIELDS] = {};
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (comp_methods[fi] != 0 && !h_cs[fi].empty()) {
            cudaMalloc(&d_comp_sizes[fi], npages * sizeof(uint32_t));
            cudaMemcpy(d_comp_sizes[fi], h_cs[fi].data(),
                       npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMalloc(&d_comp_offsets[fi], npages * sizeof(uint64_t));
            cudaMemcpy(d_comp_offsets[fi], h_co[fi].data(),
                       npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
    }

    int64_t h_revenue = 0;
    int64_t elapsed_ns = 0;

    if (ps_mismatch) {
        // ── 7a. Flatten path: per-field flatten → revenue_scan_flat ──
        bam_pfor32_io_ctx_t pfor32_ctx = bam_pfor32_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks_fused);

        // Per-field prefix_sum arrays on GPU
        uint64_t* d_ps[NUM_FIELDS] = {};
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            cudaMalloc(&d_ps[fi], npages * sizeof(uint64_t));
            cudaMemcpy(d_ps[fi], h_ps[fi].data(), npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }

        // Flat output arrays: nrows × int32_t per field
        int32_t* d_flat[NUM_FIELDS] = {};
        for (size_t fi = 0; fi < NUM_FIELDS; fi++)
            cudaMalloc(&d_flat[fi], nrows * sizeof(int32_t));

        // Pre-allocate per-field zone map mask buffers (Rule 4: alloc outside timing)
        uint8_t* d_page_active_fi[NUM_FIELDS] = {};
        if (zm_valid) {
            for (size_t fi = 1; fi < NUM_FIELDS; fi++)
                cudaMalloc(&d_page_active_fi[fi], npages);
        }

        size_t gpu_free_after_app = 0;
        cudaMemGetInfo(&gpu_free_after_app, &gpu_total_mem);
        uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

        // Pre-issue IO to initialize BaM page_cache DMA registration
        if (zm_valid) {
            bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
        }

        auto total_start = std::chrono::steady_clock::now();
        s_kernel_launches = 0;

        // GPU zonemap eval (Rule 6: IO + eval inside timing, mask stays on GPU)
        if (zm_valid) {
            bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
            cudaStreamSynchronize(stream);
            s_kernel_launches++;

            // IO byte accounting from h_mask
            for (uint64_t j = 0; j < npages; j++) {
                if (zm_ctx.h_mask[j]) {
                    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                        total_read_bytes += col_page_bytes(fi, j);
                }
            }

            // Derive per-field masks from h_mask
            d_page_active_fi[0] = zm_ctx.d_mask;
            for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
                std::vector<uint8_t> h_mask_fi(npages, 0);
                for (uint64_t pg0 = 0; pg0 < npages; pg0++) {
                    if (!zm_ctx.h_mask[pg0]) continue;
                    uint64_t row_lo = (pg0 == 0) ? 0 : h_ps[0][pg0 - 1];
                    uint64_t row_hi = h_ps[0][pg0];
                    auto it_lo = std::upper_bound(h_ps[fi].begin(), h_ps[fi].end(), row_lo);
                    uint64_t pg_lo = static_cast<uint64_t>(it_lo - h_ps[fi].begin());
                    auto it_hi = std::lower_bound(h_ps[fi].begin(), h_ps[fi].end(), row_hi);
                    uint64_t pg_hi = static_cast<uint64_t>(it_hi - h_ps[fi].begin());
                    for (uint64_t pg = pg_lo; pg <= pg_hi && pg < npages; pg++)
                        h_mask_fi[pg] = 1;
                }
                cudaMemcpy(d_page_active_fi[fi], h_mask_fi.data(), npages, cudaMemcpyHostToDevice);
            }
        }

        // Flatten each field with its own prefix_sum and per-field zone map mask
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            BAMPfor32FlattenParams fp{};
            fp.partition_start_lba = partition_start_lba;
            for (uint32_t d = 0; d < n_devices; d++)
                fp.partition_start_lbas[d] = ds.partition_start_lbas[d];
            fp.n_devices = n_devices;
            fp.page_size = static_cast<uint32_t>(page_size);
            fp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
            fp.comp_method = comp_methods[fi];
            fp.field_start_page_id = field_start_page_ids[fi];
            fp.npages = npages;
            fp.nrows = nrows;
            fp.num_blocks = num_blocks_fused;
            fp.d_prefix_sum = d_ps[fi];
            fp.d_comp_sizes = d_comp_sizes[fi];
            fp.d_comp_offsets = d_comp_offsets[fi];

            bam_pfor32_io_flatten_masked_async(
                pfor32_ctx, fp, d_page_active_fi[fi],
                /*fill_value=*/0, d_flat[fi], stream);
            s_kernel_launches++;
            cudaStreamSynchronize(stream);
        }

        // Revenue scan on flattened arrays
        // Q6 fields: [0]=L_SHIPDATE, [1]=L_QUANTITY, [2]=L_EXTENDEDPRICE, [3]=L_DISCOUNT
        revenue_scan_flat(
            d_flat[0], d_flat[1], d_flat[3], d_flat[2],
            nrows, d_revenue, sd_low, sd_high,
            qt_max, stream,
            /*dc_low=*/options.disable_other_filters ? 0 : 5,
            /*dc_high=*/options.disable_other_filters ? 0 : 7);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);

        cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost);

        auto total_end = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

        std::cout << std::dec << "Revenue total: " << h_revenue << std::endl;

        // Cleanup flatten path
        bam_pfor32_io_destroy(pfor32_ctx);
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            cudaFree(d_ps[fi]);
            cudaFree(d_flat[fi]);
        }
        // fi=0 is zm_ctx.d_mask, freed by ctx_destroy
        for (size_t fi = 1; fi < NUM_FIELDS; fi++)
            if (d_page_active_fi[fi]) cudaFree(d_page_active_fi[fi]);

        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (d_comp_sizes[fi]) cudaFree(d_comp_sizes[fi]);
            if (d_comp_offsets[fi]) cudaFree(d_comp_offsets[fi]);
        }
        cudaFree(d_revenue);
        if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
        if (pfor_ctx_zm) bam_pfor32_io_destroy(pfor_ctx_zm);
        cudaStreamDestroy(stream);
        bam_ctrl_close(ctrl);

        return BenchmarkResult{
            .read_bytes = total_read_bytes,
            .elapsed_nanoseconds = elapsed_ns,
            .compression = collect_comp_methods({comp_methods[0], comp_methods[1],
                                                 comp_methods[2], comp_methods[3]}),
            .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
            .gpu_ctrl_bytes = gpu_ctrl_bytes,
            .gpu_app_bytes = gpu_app_bytes,
            .total_pages = npages * NUM_FIELDS,
            .kernel_launches = s_kernel_launches,
        };
    }

    // ── 7b. Fused path (prefix_sums match): existing fused kernel ──
    // Page cache: num_blocks * 4 entries (one per field per block)
    bam_pfor32_io_ctx_t pfor32_ctx = bam_pfor32_io_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks_fused * 4);

    // Per-field metadata on GPU (full column, not tile-sized)
    uint64_t* d_prefix_sum = nullptr;
    cudaMalloc(&d_prefix_sum, npages * sizeof(uint64_t));
    cudaMemcpy(d_prefix_sum, h_ps[0].data(), npages * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Per-block scratch buffer: num_blocks * 4 fields * max_rows_per_page int32_t
    int32_t* d_scratch = nullptr;
    uint64_t scratch_total = (uint64_t)num_blocks_fused * 4 * max_rows_per_page;
    cudaMalloc(&d_scratch, scratch_total * sizeof(int32_t));

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_mem);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // ── 8. Build fused params & run ──
    BAMRevenueFusedParams fp{};
    fp.partition_start_lba = partition_start_lba;
    for (uint32_t d = 0; d < n_devices; d++)
        fp.partition_start_lbas[d] = ds.partition_start_lbas[d];
    fp.n_devices = n_devices;
    fp.page_size = static_cast<uint32_t>(page_size);
    fp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
    fp.npages = npages;
    fp.num_blocks = num_blocks_fused;
    fp.sd_low = sd_low;
    fp.sd_high = sd_high;
    fp.qt_max = qt_max;
    fp.dc_low = options.disable_other_filters ? 0 : 5;
    fp.dc_high = options.disable_other_filters ? 0 : 7;
    fp.d_prefix_sum = d_prefix_sum;
    fp.d_page_active = nullptr;  // set after zonemap eval
    fp.d_active_page_ids = nullptr;
    fp.num_active_pages = 0;
    fp.d_scratch = d_scratch;
    fp.scratch_stride = static_cast<uint32_t>(max_rows_per_page);
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        fp.field_start_page_ids[fi] = field_start_page_ids[fi];
        fp.comp_methods[fi] = comp_methods[fi];
        fp.d_comp_sizes[fi] = d_comp_sizes[fi];
        fp.d_comp_offsets[fi] = d_comp_offsets[fi];
    }

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    {
        auto total_start = std::chrono::steady_clock::now();
        s_kernel_launches = 0;

        // GPU zonemap eval (Rule 6: IO + eval inside timing, mask stays on GPU)
        if (zm_valid) {
            bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
            cudaStreamSynchronize(stream);
            s_kernel_launches++;
            fp.d_page_active = zm_ctx.d_mask;
            fp.d_active_page_ids = zm_ctx.d_active_ids;
            fp.num_active_pages = *zm_ctx.h_num_active;

            // IO byte accounting from h_mask
            for (uint64_t j = 0; j < npages; j++) {
                if (zm_ctx.h_mask[j]) {
                    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                        total_read_bytes += col_page_bytes(fi, j);
                }
            }
        }

        bam_revenue_fused_run(pfor32_ctx, fp, d_revenue, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);

        cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost);

        auto total_end = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    }

    // ── 9. Result retrieval & stats ──
    std::cout << std::dec << "Revenue total: " << h_revenue << std::endl;

    // ── 10. Cleanup ──
    bam_pfor32_io_destroy(pfor32_ctx);
    cudaFree(d_prefix_sum);
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (d_comp_sizes[fi]) cudaFree(d_comp_sizes[fi]);
        if (d_comp_offsets[fi]) cudaFree(d_comp_offsets[fi]);
    }
    cudaFree(d_scratch);
    cudaFree(d_revenue);
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    if (pfor_ctx_zm) bam_pfor32_io_destroy(pfor_ctx_zm);
    cudaStreamDestroy(stream);
    bam_ctrl_close(ctrl);

    return BenchmarkResult{
        .read_bytes = total_read_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = collect_comp_methods({comp_methods[0], comp_methods[1],
                                             comp_methods[2], comp_methods[3]}),
        .gpu_mem_bytes = gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NUM_FIELDS,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// checkmeta — read page 0 via BAM and print metadata
// ============================================================
void bam_check_metadata(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    // The device path is the BAM controller path directly (e.g. /dev/libnvm2)
    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    // Open a single BAM controller for all reads
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 64, 1);

    // Detect partition offset
    uint64_t part_lba = detect_partition_start_lba(ctrl);

    // Step 1: Read first 4096 bytes from partition start to learn page_size
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    int rc = bam_read_page(ctrl, init_page_size, part_lba, head_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (head read)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const uint64_t page_size = meta_head->page_size;
    std::cout << "Page size (from header): " << page_size << std::endl;

    if (page_size == 0 || page_size > 1024 * 1024) {
        std::cerr << "Invalid page_size: " << page_size << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    // Step 2: Re-read with the correct page_size
    std::vector<char> page_buf(page_size);
    rc = bam_read_page(ctrl, page_size, part_lba, page_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed (full page read)" << std::endl;
        bam_ctrl_close(ctrl);
        exit(EXIT_FAILURE);
    }

    // Done with BAM reads
    bam_ctrl_close(ctrl);

    auto* metadata = reinterpret_cast<TPCHTableMetadata*>(page_buf.data());
    superpage_set_constants(metadata->page_size);

    // Print metadata
    std::cout << "\n=== TPCH Table Metadata (via BAM) ===" << std::endl;
    std::cout << "page_size:     " << metadata->page_size << std::endl;
    std::cout << "compressed:    " << metadata->compressed << std::endl;

    auto print_table = [&](const char* name, uint64_t nrows,
                           const uint64_t* start_page_ids,
                           const uint64_t* npages,
                           const uint16_t* compression_method,
                           size_t nfields) {
        std::cout << "\n--- " << name << " (nrows=" << nrows << ") ---" << std::endl;
        for (size_t i = 0; i < nfields; i++) {
            if (npages[i] == 0) continue;
            std::cout << "  field " << std::setw(2) << i
                      << ": start_page=" << std::setw(8) << start_page_ids[i]
                      << " npages=" << std::setw(6) << npages[i]
                      << " compression=" << compression_method[i]
                      << std::endl;
        }
    };

    print_table("LINEITEM",
                metadata->table_lineitem_nrows,
                metadata->table_lineitem_start_page_ids,
                metadata->table_lineitem_npages,
                metadata->table_lineitem_compression_method,
                TPCH::common::kLineitemFieldCount);

    print_table("ORDERS",
                metadata->table_orders_nrows,
                metadata->table_orders_start_page_ids,
                metadata->table_orders_npages,
                metadata->table_orders_compression_method,
                TPCH::common::kOrdersFieldCount);

    print_table("CUSTOMER",
                metadata->table_customer_nrows,
                metadata->table_customer_start_page_ids,
                metadata->table_customer_npages,
                metadata->table_customer_compression_method,
                TPCH::common::kCustomerFieldCount);

    std::cout << "\nfree_page_id:  " << metadata->free_page_id << std::endl;
    std::cout << "npage_used:    " << metadata->npage_used << std::endl;
    std::cout << "\ncheckmeta: OK" << std::endl;
}

// ============================================================
// test_pfor_page — read first compressed L_SHIPDATE page,
// verify on CPU (header + offsets), decompress on GPU, print results.
// ============================================================
void test_pfor_page(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::Query::Q6::SCAN_TARGET_COLS[0]; // L_SHIPDATE
    uint64_t field_start_page_id = metadata.table_lineitem_start_page_ids[col];
    uint64_t npages = metadata.table_lineitem_npages[col];
    uint16_t comp_method = metadata.table_lineitem_compression_method[col];

    std::cout << "L_SHIPDATE: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method << std::endl;

    if (comp_method == 0) {
        std::cout << "L_SHIPDATE is NOT compressed. Nothing to test." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes for this field
    uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_lineitem_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
    uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    // === Read the first compressed page ===
    uint32_t cs0 = comp_page_sizes[0];
    uint64_t offset0 = offsets_vec[0];
    uint64_t lba0 = partition_start_lba + offset0 / 512;
    uint32_t io_blocks = ((cs0 + 4095u) & ~4095u) / 512;

    std::cout << "\n=== Page 0 ===\n"
              << "  compressed_page_size = " << cs0 << " bytes\n"
              << "  disk_offset = " << offset0 << " bytes (LBA " << lba0 << ")\n"
              << "  io_blocks = " << io_blocks << " (" << io_blocks * 512 << " bytes)\n";

    // Read via bam_read_page (page_size = io_blocks * 512 to read exact amount)
    size_t read_size = (size_t)io_blocks * 512;
    std::vector<char> page_buf(read_size);
    int rc = bam_read_page(ctrl, read_size, lba0, page_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed for compressed page 0" << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // === CPU-side verification: parse header ===
    struct { uint32_t nalloc; uint32_t watermark; uint32_t lfreespace; } hdr;
    std::memcpy(&hdr, page_buf.data(), sizeof(hdr));

    std::cout << "\n--- pag_head ---\n"
              << "  nalloc     = " << hdr.nalloc << "\n"
              << "  watermark  = " << hdr.watermark << "\n"
              << "  lfreespace = " << hdr.lfreespace << "\n";

    uint32_t nblocks = hdr.watermark / 128;
    uint32_t noffsets = nblocks + 1;
    std::cout << "  nblocks    = " << nblocks << "\n"
              << "  noffsets   = " << noffsets << "\n";

    // Parse offset array
    uint32_t* offsets = reinterpret_cast<uint32_t*>(page_buf.data() + 12);
    std::cout << "\n--- Offset array (first 10) ---\n";
    for (uint32_t i = 0; i < std::min(noffsets, 10u); i++) {
        std::cout << "  offsets[" << i << "] = " << offsets[i] << "\n";
    }
    if (noffsets > 10) {
        std::cout << "  ...\n  offsets[" << noffsets - 1 << "] = " << offsets[noffsets - 1] << "\n";
    }

    // Verify offsets are monotonically increasing
    bool offsets_ok = true;
    for (uint32_t i = 1; i < noffsets; i++) {
        if (offsets[i] <= offsets[i - 1]) {
            std::cerr << "  ERROR: offsets[" << i << "]=" << offsets[i]
                      << " <= offsets[" << i-1 << "]=" << offsets[i-1] << "\n";
            offsets_ok = false;
            break;
        }
    }
    if (offsets_ok) std::cout << "  Offsets: monotonically increasing OK\n";

    // Parse compressed data header (first 4 uint32 of binPack output)
    uint32_t* comp_data = offsets + noffsets;
    std::cout << "\n--- Compressed data header ---\n"
              << "  [0] block_size      = " << comp_data[0] << "\n"
              << "  [1] miniblock_count = " << comp_data[1] << "\n"
              << "  [2] total_count     = " << comp_data[2] << "\n"
              << "  [3] first_val       = " << comp_data[3] << "\n";

    // Print first block's reference and bitwidths
    if (offsets[0] < offsets[noffsets - 1]) {
        uint32_t ref_val = comp_data[offsets[0]];
        uint32_t bw_packed = comp_data[offsets[0] + 1];
        std::cout << "\n--- Block 0 ---\n"
                  << "  reference (min_val) = " << ref_val << "\n"
                  << "  bitwidths (packed)  = 0x" << std::hex << bw_packed << std::dec
                  << " (bw0=" << (bw_packed & 0xFF)
                  << " bw1=" << ((bw_packed >> 8) & 0xFF)
                  << " bw2=" << ((bw_packed >> 16) & 0xFF)
                  << " bw3=" << ((bw_packed >> 24) & 0xFF) << ")\n";
    }

    // === GPU-side decompression test ===
    std::cout << "\n--- GPU decompression test ---\n";
    uint32_t max_elems = hdr.watermark;
    if (max_elems == 0) max_elems = 16384;
    std::vector<int32_t> decomp_result(max_elems, 0);

    uint32_t nalloc = bam_test_decompress_page(page_buf.data(), read_size,
                                                decomp_result.data(), max_elems);
    std::cout << "  nalloc returned = " << nalloc << "\n";
    std::cout << "  First 20 decompressed values:\n  ";
    for (uint32_t i = 0; i < std::min(nalloc, 20u); i++) {
        std::cout << decomp_result[i];
        if (i + 1 < std::min(nalloc, 20u)) std::cout << ", ";
    }
    std::cout << "\n";

    // Compute checksum
    uint64_t checksum = 0;
    for (uint32_t i = 0; i < nalloc; i++) {
        checksum += static_cast<uint32_t>(decomp_result[i]);
    }
    std::cout << "  Checksum (sum of " << nalloc << " values) = " << checksum << "\n";

    // Quick sanity: L_SHIPDATE values should be in range 19920102..19981201
    uint32_t out_of_range = 0;
    for (uint32_t i = 0; i < nalloc; i++) {
        if (decomp_result[i] < 19920101 || decomp_result[i] > 19981231) {
            out_of_range++;
        }
    }
    if (out_of_range > 0) {
        std::cout << "  WARNING: " << out_of_range << " / " << nalloc
                  << " values out of L_SHIPDATE range!\n";
    } else {
        std::cout << "  All values in valid L_SHIPDATE range: OK\n";
    }

    std::cout << "\ntest_pfor_page: done\n";
    bam_ctrl_close(ctrl);
}

// ============================================================
// test_pfor64_page — read first compressed L_ORDERKEY page,
// verify on CPU (header + offsets + 8byte alignment), decompress on GPU.
// ============================================================
void test_pfor64_page(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;

    // L_ORDERKEY = column 0
    constexpr size_t col = 0;  // TPCH::LineitemField::L_ORDERKEY
    uint64_t field_start_page_id = metadata.table_lineitem_start_page_ids[col];
    uint64_t npages = metadata.table_lineitem_npages[col];
    uint16_t comp_method = metadata.table_lineitem_compression_method[col];

    std::cout << "L_ORDERKEY: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method << std::endl;

    if (comp_method == 0) {
        std::cout << "L_ORDERKEY is NOT compressed. Nothing to test." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes
    uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_lineitem_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
    uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    // === Read the first compressed page ===
    uint32_t cs0 = comp_page_sizes[0];
    uint64_t offset0 = offsets_vec[0];
    uint64_t lba0 = partition_start_lba + offset0 / 512;
    uint32_t io_blocks = ((cs0 + 4095u) & ~4095u) / 512;

    std::cout << "\n=== Page 0 ===\n"
              << "  compressed_page_size = " << std::hex << cs0 << std::dec
              << " bytes (" << cs0 << ")\n"
              << "  disk_offset = " << offset0 << " bytes (LBA " << std::hex << lba0 << std::dec << ")\n"
              << "  io_blocks = " << io_blocks << " (" << io_blocks * 512 << " bytes)\n";

    size_t read_size = (size_t)io_blocks * 512;
    std::vector<char> page_buf(read_size);
    int rc = bam_read_page(ctrl, read_size, lba0, page_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed for compressed page 0" << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // === CPU-side verification: parse header ===
    struct { uint32_t nalloc; uint32_t watermark; uint32_t lfreespace; } hdr;
    std::memcpy(&hdr, page_buf.data(), sizeof(hdr));

    std::cout << "\n--- pag_head ---\n"
              << "  nalloc     = " << hdr.nalloc << " (0x" << std::hex << hdr.nalloc << std::dec << ")\n"
              << "  watermark  = " << hdr.watermark << " (0x" << std::hex << hdr.watermark << std::dec << ")\n"
              << "  lfreespace = " << hdr.lfreespace << "\n";

    uint32_t nblocks = hdr.watermark / 128;
    uint32_t noffsets = nblocks + 1;
    std::cout << "  nblocks    = " << nblocks << " (0x" << std::hex << nblocks << std::dec << ")\n"
              << "  noffsets   = " << noffsets << " (0x" << std::hex << noffsets << std::dec << ")\n";

    // Parse offset array
    uint32_t* offsets = reinterpret_cast<uint32_t*>(page_buf.data() + 12);
    std::cout << "\n--- Offset array (first 10) ---\n";
    for (uint32_t i = 0; i < std::min(noffsets, 10u); i++) {
        std::cout << "  offsets[" << i << "] = " << offsets[i]
                  << " (0x" << std::hex << offsets[i] << std::dec << ")\n";
    }
    if (noffsets > 10) {
        std::cout << "  ...\n  offsets[" << noffsets - 1 << "] = " << offsets[noffsets - 1] << "\n";
    }

    // Verify offsets monotonically increasing
    bool offsets_ok = true;
    for (uint32_t i = 1; i < noffsets; i++) {
        if (offsets[i] <= offsets[i - 1]) {
            std::cerr << "  ERROR: offsets[" << i << "]=" << offsets[i]
                      << " <= offsets[" << i-1 << "]=" << offsets[i-1] << "\n";
            offsets_ok = false;
            break;
        }
    }
    if (offsets_ok) std::cout << "  Offsets: monotonically increasing OK\n";

    // Check 8-byte alignment padding
    uint32_t encoded_value_offset = noffsets;
    bool has_padding = false;
    if ((noffsets & 1) == 0) {
        encoded_value_offset++;
        has_padding = true;
    }
    size_t data_byte_offset = 12 + encoded_value_offset * sizeof(uint32_t);
    std::cout << "\n--- 8-byte alignment ---\n"
              << "  noffsets even? " << ((noffsets & 1) == 0 ? "yes" : "no")
              << ", padding inserted: " << (has_padding ? "yes" : "no")
              << "\n  encoded data at byte offset " << data_byte_offset
              << " (8-byte aligned: " << ((data_byte_offset & 7) == 0 ? "OK" : "FAIL") << ")\n";

    // Parse compressed data header (first 4 ulong of binPack64 output)
    uint64_t* comp_data = reinterpret_cast<uint64_t*>(page_buf.data() + data_byte_offset);
    std::cout << "\n--- Compressed data header (ulong) ---\n"
              << "  [0] block_size      = " << comp_data[0] << "\n"
              << "  [1] miniblock_count = " << comp_data[1] << "\n"
              << "  [2] total_count     = " << comp_data[2] << "\n"
              << "  [3] first_val       = " << comp_data[3] << "\n";

    // Print first block's reference and bitwidths
    if (offsets[0] < offsets[noffsets - 1]) {
        uint64_t ref_val = comp_data[offsets[0]];
        uint32_t bw_packed = (uint32_t)comp_data[offsets[0] + 1];
        std::cout << "\n--- Block 0 ---\n"
                  << "  reference (min_val) = " << ref_val << "\n"
                  << "  bitwidths (packed)  = 0x" << std::hex << bw_packed << std::dec
                  << " (bw0=" << (bw_packed & 0xFF)
                  << " bw1=" << ((bw_packed >> 8) & 0xFF)
                  << " bw2=" << ((bw_packed >> 16) & 0xFF)
                  << " bw3=" << ((bw_packed >> 24) & 0xFF) << ")\n";
    }

    // === GPU-side decompression test ===
    std::cout << "\n--- GPU decompression test (PFOR64) ---\n";
    uint32_t max_elems = hdr.watermark;
    if (max_elems == 0) max_elems = 16384;
    std::vector<int64_t> decomp_result(max_elems, 0);

    uint32_t nalloc = bam_test_decompress_page64(page_buf.data(), read_size,
                                                  decomp_result.data(), max_elems);
    std::cout << "  nalloc returned = " << nalloc << "\n";
    std::cout << "  First 20 decompressed values:\n  ";
    for (uint32_t i = 0; i < std::min(nalloc, 20u); i++) {
        std::cout << decomp_result[i];
        if (i + 1 < std::min(nalloc, 20u)) std::cout << ", ";
    }
    std::cout << "\n";

    // Compute checksum
    uint64_t checksum = 0;
    for (uint32_t i = 0; i < nalloc; i++) {
        checksum += static_cast<uint64_t>(decomp_result[i]);
    }
    std::cout << "  Checksum (sum of " << nalloc << " values) = " << checksum << "\n";

    // Sanity: L_ORDERKEY values should be positive integers
    uint32_t out_of_range = 0;
    for (uint32_t i = 0; i < nalloc; i++) {
        if (decomp_result[i] <= 0) {
            out_of_range++;
        }
    }
    if (out_of_range > 0) {
        std::cout << "  WARNING: " << out_of_range << " / " << nalloc
                  << " values <= 0!\n";
    } else {
        std::cout << "  All values > 0: OK\n";
    }

    std::cout << "\ntest_pfor64_page: done\n";
    bam_ctrl_close(ctrl);
}

// ============================================================
// scan_o_comment — read LZ4-compressed ORDERS.O_COMMENT pages
// via BaM, decompress with nvCOMPdx on GPU, scan VCHAR records.
// ============================================================
void scan_o_comment(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::common::O_COMMENT;
    uint64_t field_start_page_id = metadata.table_orders_start_page_ids[col];
    uint64_t npages = metadata.table_orders_npages[col];
    uint16_t comp_method = metadata.table_orders_compression_method[col];

    std::cout << std::dec
              << "O_COMMENT: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method
              << " nrows=" << metadata.table_orders_nrows << std::endl;

    if (comp_method == 0) {
        std::cerr << "O_COMMENT is NOT compressed. Expected LZ4." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes for O_COMMENT
    uint64_t cs_start = metadata.table_orders_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_orders_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_orders_compression_nbases[col];
    uint64_t base_start = metadata.table_orders_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    // Convert offsets to uint64_t array for the kernel
    std::vector<uint64_t> comp_offsets(npages);
    for (uint64_t i = 0; i < npages; i++) {
        comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
    }

    // Determine num_blocks from GPU SM count
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks = static_cast<uint32_t>(sm_count) * 2;
    if (num_blocks > static_cast<uint32_t>(npages)) {
        num_blocks = static_cast<uint32_t>(npages);
    }

    std::cout << std::dec
              << "scan_o_comment: npages=" << npages
              << " sm_count=" << sm_count
              << " num_blocks=" << num_blocks
              << " page_size=" << page_size
              << " nbase=" << nbase << std::endl;

    // ── Upload compressed page metadata to GPU ──
    uint32_t* d_comp_sizes = nullptr;
    uint64_t* d_comp_offsets = nullptr;
    {
        size_t sz = npages * sizeof(uint32_t);
        cudaMalloc(&d_comp_sizes, sz);
        cudaMemcpy(d_comp_sizes, comp_page_sizes, sz, cudaMemcpyHostToDevice);
    }
    {
        size_t sz = npages * sizeof(uint64_t);
        cudaMalloc(&d_comp_offsets, sz);
        cudaMemcpy(d_comp_offsets, comp_offsets.data(), sz, cudaMemcpyHostToDevice);
    }

    // ── Create VCHAR I/O context (BAM page cache + staging buffer) ──
    bam_vchar_io_ctx_t io_ctx = bam_vchar_io_create(ctrl,
        static_cast<uint32_t>(page_size), num_blocks);
    char* d_staging_buf = bam_vchar_io_staging_buf(io_ctx);

    // ── Allocate decompress buffer + output counters ──
    char* d_decomp_buf = nullptr;
    cudaMalloc(&d_decomp_buf, (size_t)num_blocks * page_size);

    uint64_t* d_total_records = nullptr;
    uint64_t* d_total_strlen = nullptr;
    uint64_t* d_total_byte_sum = nullptr;
    cudaMalloc(&d_total_records, sizeof(uint64_t));
    cudaMalloc(&d_total_strlen, sizeof(uint64_t));
    cudaMalloc(&d_total_byte_sum, sizeof(uint64_t));
    cudaMemset(d_total_records, 0, sizeof(uint64_t));
    cudaMemset(d_total_strlen, 0, sizeof(uint64_t));
    cudaMemset(d_total_byte_sum, 0, sizeof(uint64_t));

    // ── Batch processing: I/O kernel (C++11) → decompress+scan kernel (C++17) ──
    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t batch = 0; batch < npages; batch += num_blocks) {
        uint32_t batch_size = static_cast<uint32_t>(
            std::min((uint64_t)num_blocks, npages - batch));

        // Step 1: Read compressed pages via BAM into staging buffer
        bam_vchar_io_read_batch(io_ctx, d_comp_sizes, d_comp_offsets,
                                 partition_start_lba, batch, batch_size);

        // Step 2: Decompress (nvCOMPdx LZ4) + scan VCHAR records
        bam_vchar_decomp_scan_batch(
            d_staging_buf, d_comp_sizes, d_decomp_buf,
            d_total_records, d_total_strlen, d_total_byte_sum,
            static_cast<uint32_t>(page_size), batch, batch_size);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // ── Read results ──
    BAMVcharResult result = {};
    cudaMemcpy(&result.total_records, d_total_records,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_strlen, d_total_strlen,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_byte_sum, d_total_byte_sum,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);

    std::cout << std::dec
              << "\n=== scan_o_comment results ===\n"
              << "  total_records  = " << result.total_records << "\n"
              << "  total_strlen   = " << result.total_strlen << "\n"
              << "  total_byte_sum = " << result.total_byte_sum << "\n"
              << "  elapsed        = " << elapsed_ns / 1'000'000 << "."
              << (elapsed_ns % 1'000'000) / 1'000 << " ms\n"
              << std::endl;

    // Verify record count against metadata
    if (result.total_records == metadata.table_lineitem_nrows) {
        std::cout << "  Record count matches metadata nrows: OK" << std::endl;
    } else {
        std::cerr << "  WARNING: record count " << result.total_records
                  << " != metadata nrows " << metadata.table_lineitem_nrows << std::endl;
    }

    // ── Cleanup ──
    cudaFree(d_total_records);
    cudaFree(d_total_strlen);
    cudaFree(d_total_byte_sum);
    cudaFree(d_decomp_buf);
    cudaFree(d_comp_sizes);
    cudaFree(d_comp_offsets);
    bam_vchar_io_destroy(io_ctx);
    bam_ctrl_close(ctrl);
}

// ============================================================
// scan_o_comment_v2 — 4 warps read 4 pages concurrently,
// each warp decompresses its page, all 128 threads scan.
// ============================================================
void scan_o_comment_v2(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::common::L_COMMENT;
    uint64_t field_start_page_id = metadata.table_lineitem_start_page_ids[col];
    uint64_t npages = metadata.table_lineitem_npages[col];
    uint16_t comp_method = metadata.table_lineitem_compression_method[col];

    std::cout << std::dec
              << "L_COMMENT: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method
              << " nrows=" << metadata.table_lineitem_nrows << std::endl;

    if (comp_method == 0) {
        std::cerr << "L_COMMENT is NOT compressed. Expected LZ4." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes for L_COMMENT
    uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_lineitem_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
    uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    // Convert offsets to uint64_t array for the kernel
    std::vector<uint64_t> comp_offsets(npages);
    for (uint64_t i = 0; i < npages; i++) {
        comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
    }

    // Determine num_blocks from GPU SM count
    // v2: each block uses 4x shared memory (4 warp decompressors), ~1 block/SM
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks = static_cast<uint32_t>(sm_count);
    // Cap to ceil(npages / 4)
    uint32_t max_blocks = static_cast<uint32_t>((npages + 3) / 4);
    if (num_blocks > max_blocks) {
        num_blocks = max_blocks;
    }

    // IO metrics
    uint64_t total_comp_bytes = 0;
    uint32_t max_comp_size = 0, min_comp_size = UINT32_MAX;
    for (uint64_t i = 0; i < npages; i++) {
        uint32_t cs = comp_page_sizes[i];
        total_comp_bytes += cs;
        if (cs > max_comp_size) max_comp_size = cs;
        if (cs < min_comp_size) min_comp_size = cs;
    }
    double avg_comp_size = (double)total_comp_bytes / npages;
    double comp_ratio = avg_comp_size / page_size;

    std::cout << std::dec
              << "scan_o_comment_v2: npages=" << npages
              << " sm_count=" << sm_count
              << " num_blocks=" << num_blocks
              << " pages_per_batch=" << num_blocks * 4
              << " page_size=" << page_size
              << " nbase=" << nbase << std::endl;
    std::cout << "  IO stats: total_comp_bytes=" << total_comp_bytes
              << " (" << std::fixed << std::setprecision(2)
              << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB)"
              << " avg=" << (uint32_t)avg_comp_size
              << " min=" << min_comp_size
              << " max=" << max_comp_size
              << " ratio=" << std::setprecision(3) << comp_ratio
              << std::endl;

    // ── Upload compressed page metadata to GPU ──
    uint32_t* d_comp_sizes = nullptr;
    uint64_t* d_comp_offsets = nullptr;
    {
        size_t sz = npages * sizeof(uint32_t);
        cudaMalloc(&d_comp_sizes, sz);
        cudaMemcpy(d_comp_sizes, comp_page_sizes, sz, cudaMemcpyHostToDevice);
    }
    {
        size_t sz = npages * sizeof(uint64_t);
        cudaMalloc(&d_comp_offsets, sz);
        cudaMemcpy(d_comp_offsets, comp_offsets.data(), sz, cudaMemcpyHostToDevice);
    }

    // ── Create VCHAR I/O v2 context (BAM page cache + staging buffer) ──
    bam_vchar_io_ctx_t io_ctx = bam_vchar_io_v2_create(ctrl,
        static_cast<uint32_t>(page_size), num_blocks);
    char* d_staging_buf = bam_vchar_io_v2_staging_buf(io_ctx);

    // ── Allocate decompress buffer + output counters ──
    // Decomp buffer: num_blocks * 4 pages
    char* d_decomp_buf = nullptr;
    cudaMalloc(&d_decomp_buf, (size_t)num_blocks * 4 * page_size);

    uint64_t* d_total_records = nullptr;
    uint64_t* d_total_strlen = nullptr;
    uint64_t* d_total_byte_sum = nullptr;
    cudaMalloc(&d_total_records, sizeof(uint64_t));
    cudaMalloc(&d_total_strlen, sizeof(uint64_t));
    cudaMalloc(&d_total_byte_sum, sizeof(uint64_t));
    cudaMemset(d_total_records, 0, sizeof(uint64_t));
    cudaMemset(d_total_strlen, 0, sizeof(uint64_t));
    cudaMemset(d_total_byte_sum, 0, sizeof(uint64_t));

    // ── Batch processing ──
    // Each batch: num_blocks blocks × 4 pages/block = num_blocks*4 pages
    const uint32_t pages_per_batch = num_blocks * 4;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t batch = 0; batch < npages; batch += pages_per_batch) {
        uint64_t remaining = npages - batch;
        uint32_t batch_blocks = static_cast<uint32_t>(
            std::min((uint64_t)num_blocks, (remaining + 3) / 4));

        // Step 1: Read compressed pages via BAM (4 pages per block)
        bam_vchar_io_v2_read_batch(io_ctx, d_comp_sizes, d_comp_offsets,
                                    partition_start_lba, batch, batch_blocks, npages);

        // Step 2: Decompress + scan (4 warps decompress, 128 threads scan)
        bam_vchar_decomp_scan_batch_v2(
            d_staging_buf, d_comp_sizes, d_decomp_buf,
            d_total_records, d_total_strlen, d_total_byte_sum,
            static_cast<uint32_t>(page_size), batch, batch_blocks, npages);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // ── Read results ──
    BAMVcharResult result = {};
    cudaMemcpy(&result.total_records, d_total_records,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_strlen, d_total_strlen,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_byte_sum, d_total_byte_sum,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);

    std::cout << std::dec
              << "\n=== scan_o_comment_v2 results ===\n"
              << "  total_records  = " << result.total_records << "\n"
              << "  total_strlen   = " << result.total_strlen << "\n"
              << "  total_byte_sum = " << result.total_byte_sum << "\n"
              << "  elapsed        = " << elapsed_ns / 1'000'000 << "."
              << (elapsed_ns % 1'000'000) / 1'000 << " ms\n"
              << std::endl;

    // IO throughput
    {
        double elapsed_s = (double)elapsed_ns / 1e9;
        double throughput = (double)total_comp_bytes / elapsed_s;
        std::cout << "  IO throughput  = " << std::fixed << std::setprecision(2)
                  << throughput / (1024.0 * 1024 * 1024) << " GB/s"
                  << " (comp_bytes=" << std::setprecision(2)
                  << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB"
                  << " / " << std::setprecision(3) << elapsed_s << " s)"
                  << std::endl;
    }

    // Verify record count against metadata
    if (result.total_records == metadata.table_lineitem_nrows) {
        std::cout << "  Record count matches metadata nrows: OK" << std::endl;
    } else {
        std::cerr << "  WARNING: record count " << result.total_records
                  << " != metadata nrows " << metadata.table_lineitem_nrows << std::endl;
    }

    // ── Cleanup ──
    cudaFree(d_total_records);
    cudaFree(d_total_strlen);
    cudaFree(d_total_byte_sum);
    cudaFree(d_decomp_buf);
    cudaFree(d_comp_sizes);
    cudaFree(d_comp_offsets);
    bam_vchar_io_v2_destroy(io_ctx);
    bam_ctrl_close(ctrl);
}

// ============================================================
// scan_o_comment_v3 — double-buffered pipeline + cp.async scan
//
// Two IO contexts (buf_a, buf_b) with separate page caches.
// IO(batch N+1) on io_stream overlaps with compute(batch N)
// on comp_stream. cp.async 4B prefetch for byte-sum scan.
// ============================================================
void scan_o_comment_v3(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::common::O_COMMENT;
    uint64_t field_start_page_id = metadata.table_orders_start_page_ids[col];
    uint64_t npages = metadata.table_orders_npages[col];
    uint16_t comp_method = metadata.table_orders_compression_method[col];

    std::cout << std::dec
              << "O_COMMENT: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method
              << " nrows=" << metadata.table_orders_nrows << std::endl;

    if (comp_method == 0) {
        std::cerr << "O_COMMENT is NOT compressed. Expected LZ4." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes for O_COMMENT
    uint64_t cs_start = metadata.table_orders_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_orders_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_orders_compression_nbases[col];
    uint64_t base_start = metadata.table_orders_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    std::vector<uint64_t> comp_offsets(npages);
    for (uint64_t i = 0; i < npages; i++) {
        comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
    }

    // Auto-tune num_blocks from SM count
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks = static_cast<uint32_t>(sm_count);
    uint32_t max_blocks = static_cast<uint32_t>((npages + 3) / 4);
    if (num_blocks > max_blocks) {
        num_blocks = max_blocks;
    }
    const uint32_t pages_per_batch = num_blocks * 4;

    std::cout << std::dec
              << "scan_o_comment_v3: npages=" << npages
              << " sm_count=" << sm_count
              << " num_blocks=" << num_blocks
              << " pages_per_batch=" << pages_per_batch
              << " page_size=" << page_size
              << " nbase=" << nbase << std::endl;

    // ── Upload compressed page metadata to GPU ──
    uint32_t* d_comp_sizes = nullptr;
    uint64_t* d_comp_offsets = nullptr;
    {
        size_t sz = npages * sizeof(uint32_t);
        cudaMalloc(&d_comp_sizes, sz);
        cudaMemcpy(d_comp_sizes, comp_page_sizes, sz, cudaMemcpyHostToDevice);
    }
    {
        size_t sz = npages * sizeof(uint64_t);
        cudaMalloc(&d_comp_offsets, sz);
        cudaMemcpy(d_comp_offsets, comp_offsets.data(), sz, cudaMemcpyHostToDevice);
    }

    // ── Create two IO contexts for double buffering ──
    bam_vchar_io_ctx_t io_ctx_a = bam_vchar_io_v2_create(ctrl,
        static_cast<uint32_t>(page_size), num_blocks);
    bam_vchar_io_ctx_t io_ctx_b = bam_vchar_io_v2_create(ctrl,
        static_cast<uint32_t>(page_size), num_blocks);
    char* d_staging_a = bam_vchar_io_v2_staging_buf(io_ctx_a);
    char* d_staging_b = bam_vchar_io_v2_staging_buf(io_ctx_b);

    // ── Two decompress buffers ──
    char* d_decomp_a = nullptr;
    char* d_decomp_b = nullptr;
    cudaMalloc(&d_decomp_a, (size_t)num_blocks * 4 * page_size);
    cudaMalloc(&d_decomp_b, (size_t)num_blocks * 4 * page_size);

    // ── Global accumulators ──
    uint64_t* d_total_records = nullptr;
    uint64_t* d_total_strlen = nullptr;
    uint64_t* d_total_byte_sum = nullptr;
    cudaMalloc(&d_total_records, sizeof(uint64_t));
    cudaMalloc(&d_total_strlen, sizeof(uint64_t));
    cudaMalloc(&d_total_byte_sum, sizeof(uint64_t));
    cudaMemset(d_total_records, 0, sizeof(uint64_t));
    cudaMemset(d_total_strlen, 0, sizeof(uint64_t));
    cudaMemset(d_total_byte_sum, 0, sizeof(uint64_t));

    // ── Compute stream for decompress+scan ──
    cudaStream_t comp_stream;
    cudaStreamCreate(&comp_stream);

    // ── Double-buffered pipeline ──
    //
    //   IO stream A/B:  [IO batch0]        [IO batch2]        ...
    //   Comp stream:              [Comp b0]           [Comp b2] ...
    //   IO stream B/A:         [IO batch1]        [IO batch3]
    //   Comp stream:                      [Comp b1]           [Comp b3]
    //
    auto t0 = std::chrono::high_resolution_clock::now();

    // Current/next buffer tracking
    bam_vchar_io_ctx_t cur_io  = io_ctx_a;
    bam_vchar_io_ctx_t nxt_io  = io_ctx_b;
    char*              cur_stg = d_staging_a;
    char*              nxt_stg = d_staging_b;
    char*              cur_dec = d_decomp_a;
    char*              nxt_dec = d_decomp_b;

    // Prime: IO batch 0 into buffer A
    uint64_t batch0_remaining = npages;
    uint32_t batch0_blocks = static_cast<uint32_t>(
        std::min((uint64_t)num_blocks, (batch0_remaining + 3) / 4));
    bam_vchar_io_v2_read_batch_async(cur_io, d_comp_sizes, d_comp_offsets,
                                      partition_start_lba, 0, batch0_blocks, npages);
    bam_vchar_io_v2_sync(cur_io);

    for (uint64_t batch = 0; batch < npages; batch += pages_per_batch) {
        uint64_t remaining = npages - batch;
        uint32_t batch_blocks = static_cast<uint32_t>(
            std::min((uint64_t)num_blocks, (remaining + 3) / 4));

        uint64_t next_batch = batch + pages_per_batch;
        bool has_next = (next_batch < npages);

        // Launch decompress+scan for current buffer (async on comp_stream)
        bam_vchar_decomp_scan_batch_v3_async(
            cur_stg, d_comp_sizes, cur_dec,
            d_total_records, d_total_strlen, d_total_byte_sum,
            static_cast<uint32_t>(page_size), batch, batch_blocks,
            npages, comp_stream);

        // If next batch exists, launch IO into next buffer (async)
        if (has_next) {
            uint64_t nxt_remaining = npages - next_batch;
            uint32_t nxt_blocks = static_cast<uint32_t>(
                std::min((uint64_t)num_blocks, (nxt_remaining + 3) / 4));
            bam_vchar_io_v2_read_batch_async(nxt_io, d_comp_sizes, d_comp_offsets,
                                              partition_start_lba, next_batch,
                                              nxt_blocks, npages);
        }

        // Wait for compute to finish
        cudaStreamSynchronize(comp_stream);

        // Wait for next IO to finish (data ready for next iteration)
        if (has_next) {
            bam_vchar_io_v2_sync(nxt_io);
        }

        // Swap buffers
        std::swap(cur_io,  nxt_io);
        std::swap(cur_stg, nxt_stg);
        std::swap(cur_dec, nxt_dec);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // ── Read results ──
    BAMVcharResult result = {};
    cudaMemcpy(&result.total_records, d_total_records,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_strlen, d_total_strlen,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_byte_sum, d_total_byte_sum,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);

    std::cout << std::dec
              << "\n=== scan_o_comment_v3 results ===\n"
              << "  total_records  = " << result.total_records << "\n"
              << "  total_strlen   = " << result.total_strlen << "\n"
              << "  total_byte_sum = " << result.total_byte_sum << "\n"
              << "  elapsed        = " << elapsed_ns / 1'000'000 << "."
              << (elapsed_ns % 1'000'000) / 1'000 << " ms\n"
              << std::endl;

    if (result.total_records == metadata.table_orders_nrows) {
        std::cout << "  Record count matches metadata nrows: OK" << std::endl;
    } else {
        std::cerr << "  WARNING: record count " << result.total_records
                  << " != metadata nrows " << metadata.table_orders_nrows << std::endl;
    }

    // ── Cleanup ──
    cudaStreamDestroy(comp_stream);
    cudaFree(d_total_records);
    cudaFree(d_total_strlen);
    cudaFree(d_total_byte_sum);
    cudaFree(d_decomp_a);
    cudaFree(d_decomp_b);
    cudaFree(d_comp_sizes);
    cudaFree(d_comp_offsets);
    bam_vchar_io_v2_destroy(io_ctx_a);
    bam_vchar_io_v2_destroy(io_ctx_b);
    bam_ctrl_close(ctrl);
}

// ============================================================
// scan_o_comment_v4 — single-kernel: BAM IO + per-thread LZ4 + VCHAR scan
//
// 128 threads (4 warps) per block.  Each warp: lane 0 reads
// 32 pages via BAM, then all 32 threads decompress + scan.
// ============================================================
void scan_o_comment_v4(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::common::L_COMMENT;
    uint64_t field_start_page_id = metadata.table_lineitem_start_page_ids[col];
    uint64_t npages = metadata.table_lineitem_npages[col];
    uint16_t comp_method = metadata.table_lineitem_compression_method[col];

    std::cout << std::dec
              << "L_COMMENT: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method
              << " nrows=" << metadata.table_lineitem_nrows << std::endl;

    if (comp_method == 0) {
        std::cerr << "L_COMMENT is NOT compressed. Expected LZ4." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes
    uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_lineitem_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
    uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    std::vector<uint64_t> comp_offsets(npages);
    for (uint64_t i = 0; i < npages; i++) {
        comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
    }

    // Determine num_blocks: sm_count / 2 (= 54 on A100, 6,912 threads)
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks = static_cast<uint32_t>(sm_count) / 2;
    if (num_blocks == 0) num_blocks = 1;

    // Cap so we don't allocate more than needed
    uint32_t max_blocks = static_cast<uint32_t>(
        (npages + 127) / 128);  // ceil(npages / 128)
    if (num_blocks > max_blocks) num_blocks = max_blocks;

    uint64_t total_threads = (uint64_t)num_blocks * 128;
    double pc_gb = (double)(total_threads * page_size) / (1024.0 * 1024 * 1024);
    double decomp_gb = pc_gb;

    // IO metrics: total compressed bytes and per-page stats
    uint64_t total_comp_bytes = 0;
    uint32_t max_comp_size = 0, min_comp_size = UINT32_MAX;
    for (uint64_t i = 0; i < npages; i++) {
        uint32_t cs = comp_page_sizes[i];
        total_comp_bytes += cs;
        if (cs > max_comp_size) max_comp_size = cs;
        if (cs < min_comp_size) min_comp_size = cs;
    }
    double avg_comp_size = (double)total_comp_bytes / npages;
    double comp_ratio = avg_comp_size / page_size;

    std::cout << std::dec
              << "scan_o_comment_v4: npages=" << npages
              << " sm_count=" << sm_count
              << " num_blocks=" << num_blocks
              << " total_threads=" << total_threads
              << " page_cache=" << std::fixed << std::setprecision(2)
              << pc_gb << " GB"
              << " decomp_buf=" << decomp_gb << " GB"
              << std::endl;
    std::cout << "  IO stats: total_comp_bytes=" << total_comp_bytes
              << " (" << std::fixed << std::setprecision(2)
              << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB)"
              << " avg=" << (uint32_t)avg_comp_size
              << " min=" << min_comp_size
              << " max=" << max_comp_size
              << " ratio=" << std::setprecision(3) << comp_ratio
              << std::endl;

    // Run
    uint64_t total_records = 0, total_strlen = 0, total_byte_sum = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    bam_scan_o_comment_v4_run(
        ctrl, static_cast<uint32_t>(page_size), npages,
        partition_start_lba,
        comp_page_sizes, comp_offsets.data(),
        num_blocks,
        &total_records, &total_strlen, &total_byte_sum);

    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << std::dec
              << "\n=== scan_o_comment_v4 results ===\n"
              << "  total_records  = " << total_records << "\n"
              << "  total_strlen   = " << total_strlen << "\n"
              << "  total_byte_sum = " << total_byte_sum << "\n"
              << "  elapsed        = " << elapsed_ns / 1'000'000 << "."
              << (elapsed_ns % 1'000'000) / 1'000 << " ms\n"
              << std::endl;

    // IO throughput
    {
        double elapsed_s = (double)elapsed_ns / 1e9;
        double throughput = (double)total_comp_bytes / elapsed_s;
        std::cout << "  IO throughput  = " << std::fixed << std::setprecision(2)
                  << throughput / (1024.0 * 1024 * 1024) << " GB/s"
                  << " (comp_bytes=" << std::setprecision(2)
                  << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB"
                  << " / " << std::setprecision(3) << elapsed_s << " s)"
                  << std::endl;
    }

    if (total_records == metadata.table_lineitem_nrows) {
        std::cout << "  Record count matches metadata nrows: OK" << std::endl;
    } else {
        std::cerr << "  WARNING: record count " << total_records
                  << " != metadata nrows " << metadata.table_lineitem_nrows << std::endl;
    }

    bam_ctrl_close(ctrl);
}

// ============================================================
// scan_o_comment_v5 — GPU-internal IO pipelining (double-buffered)
//
// 128 threads (4 warps) per block.  Each warp has 2×N=8 page cache
// slots for double-buffered async IO, N=4 decompress buffers.
// Pipeline: poll bank A → submit IO bank B → decompress+scan bank A → swap.
// ============================================================
void scan_o_comment_v5(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::common::L_COMMENT;
    uint64_t field_start_page_id = metadata.table_lineitem_start_page_ids[col];
    uint64_t npages = metadata.table_lineitem_npages[col];
    uint16_t comp_method = metadata.table_lineitem_compression_method[col];

    std::cout << std::dec
              << "L_COMMENT: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method
              << " nrows=" << metadata.table_lineitem_nrows << std::endl;

    if (comp_method == 0) {
        std::cerr << "L_COMMENT is NOT compressed. Expected LZ4." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes
    uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_lineitem_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
    uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    std::vector<uint64_t> comp_offsets(npages);
    for (uint64_t i = 0; i < npages; i++) {
        comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
    }

    // Coalesced IO parameters
    const uint32_t coalesce_limit = 2 * 1024 * 1024;  // 2 MiB max per NVMe command
    const uint32_t GROUPS_PER_WARP = 12;  // max coalesced groups per warp

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    // num_blocks = sm_count / 2 (same as v4)
    uint32_t num_blocks = static_cast<uint32_t>(sm_count) / 2;
    if (num_blocks == 0) num_blocks = 1;
    uint32_t max_blocks = static_cast<uint32_t>((npages + 127) / 128);
    if (num_blocks > max_blocks) num_blocks = max_blocks;

    uint32_t total_warps = num_blocks * 4;
    uint32_t iters_per_warp = static_cast<uint32_t>(
        (npages + (uint64_t)total_warps * 32 - 1) / ((uint64_t)total_warps * 32));

    double pc_gb = (double)((uint64_t)total_warps * GROUPS_PER_WARP * coalesce_limit)
                   / (1024.0 * 1024 * 1024);
    double decomp_gb = (double)((uint64_t)total_warps * 32 * page_size)
                       / (1024.0 * 1024 * 1024);

    // IO metrics
    uint64_t total_comp_bytes = 0;
    uint32_t max_comp_size = 0, min_comp_size = UINT32_MAX;
    for (uint64_t i = 0; i < npages; i++) {
        uint32_t cs = comp_page_sizes[i];
        total_comp_bytes += cs;
        if (cs > max_comp_size) max_comp_size = cs;
        if (cs < min_comp_size) min_comp_size = cs;
    }
    double avg_comp_size = (double)total_comp_bytes / npages;
    double comp_ratio = avg_comp_size / page_size;

    // Estimate coalesced groups per warp iteration (32 pages)
    uint32_t est_pages_per_group = coalesce_limit / (max_comp_size > 0 ? max_comp_size : 1);
    if (est_pages_per_group == 0) est_pages_per_group = 1;
    uint32_t est_groups = (32 + est_pages_per_group - 1) / est_pages_per_group;
    uint32_t est_cmds_per_iter = est_groups * total_warps;

    std::cout << std::dec
              << "scan_o_comment_v5: npages=" << npages
              << " sm_count=" << sm_count
              << " num_blocks=" << num_blocks
              << " total_warps=" << total_warps
              << " iters/warp=" << iters_per_warp
              << " coalesce=" << (coalesce_limit / 1024) << "KiB"
              << " groups/warp=" << est_groups
              << " cmds/iter=" << est_cmds_per_iter
              << "\n  page_cache=" << std::fixed << std::setprecision(2)
              << pc_gb << " GB"
              << " decomp_buf=" << decomp_gb << " GB"
              << std::endl;
    std::cout << "  IO stats: total_comp_bytes=" << total_comp_bytes
              << " (" << std::fixed << std::setprecision(2)
              << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB)"
              << " avg=" << (uint32_t)avg_comp_size
              << " min=" << min_comp_size
              << " max=" << max_comp_size
              << " ratio=" << std::setprecision(3) << comp_ratio
              << std::endl;

    // Run
    uint64_t total_records = 0, total_strlen = 0, total_byte_sum = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    bam_scan_o_comment_v5_run(
        ctrl, static_cast<uint32_t>(page_size), npages,
        partition_start_lba,
        comp_page_sizes, comp_offsets.data(),
        num_blocks, coalesce_limit,
        &total_records, &total_strlen, &total_byte_sum);

    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << std::dec
              << "\n=== scan_o_comment_v5 results ===\n"
              << "  total_records  = " << total_records << "\n"
              << "  total_strlen   = " << total_strlen << "\n"
              << "  total_byte_sum = " << total_byte_sum << "\n"
              << "  elapsed        = " << elapsed_ns / 1'000'000 << "."
              << (elapsed_ns % 1'000'000) / 1'000 << " ms\n"
              << std::endl;

    // IO throughput
    {
        double elapsed_s = (double)elapsed_ns / 1e9;
        double throughput = (double)total_comp_bytes / elapsed_s;
        std::cout << "  IO throughput  = " << std::fixed << std::setprecision(2)
                  << throughput / (1024.0 * 1024 * 1024) << " GB/s"
                  << " (comp_bytes=" << std::setprecision(2)
                  << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB"
                  << " / " << std::setprecision(3) << elapsed_s << " s)"
                  << std::endl;
    }

    if (total_records == metadata.table_lineitem_nrows) {
        std::cout << "  Record count matches metadata nrows: OK" << std::endl;
    } else {
        std::cerr << "  WARNING: record count " << total_records
                  << " != metadata nrows " << metadata.table_lineitem_nrows << std::endl;
    }

    bam_ctrl_close(ctrl);
}

// ============================================================
// scan_o_comment_v6 — PAR-32K nvCOMPdx warp-cooperative decompression
//
// Two-phase double-buffered pipeline (same as v3):
//   Phase 1: BAM I/O reads compressed PAR-32K pages into staging buffer
//   Phase 2: nvCOMPdx Warp() decompresses 32 × 32KiB sub-chunks + VCHAR scan
//
// 128 threads (4 warps) per block, each warp handles 1 page (4 pages/block).
// ============================================================
void scan_o_comment_v6(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::common::L_COMMENT;
    uint64_t field_start_page_id = metadata.table_lineitem_start_page_ids[col];
    uint64_t npages = metadata.table_lineitem_npages[col];
    uint16_t comp_method = metadata.table_lineitem_compression_method[col];

    std::cout << std::dec
              << "L_COMMENT: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method
              << " nrows=" << metadata.table_lineitem_nrows << std::endl;

    if (comp_method == 0) {
        std::cerr << "L_COMMENT is NOT compressed. Expected LZ4PAR." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes
    uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_lineitem_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
    uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    std::vector<uint64_t> comp_offsets(npages);
    for (uint64_t i = 0; i < npages; i++) {
        comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
    }

    // Auto-tune num_blocks from SM count
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks = static_cast<uint32_t>(sm_count);
    uint32_t max_blocks = static_cast<uint32_t>((npages + 3) / 4);
    if (num_blocks > max_blocks) num_blocks = max_blocks;
    const uint32_t pages_per_batch = num_blocks * 4;

    // IO metrics
    uint64_t total_comp_bytes = 0;
    uint32_t max_comp_size = 0, min_comp_size = UINT32_MAX;
    for (uint64_t i = 0; i < npages; i++) {
        uint32_t cs = comp_page_sizes[i];
        total_comp_bytes += cs;
        if (cs > max_comp_size) max_comp_size = cs;
        if (cs < min_comp_size) min_comp_size = cs;
    }
    double avg_comp_size = (double)total_comp_bytes / npages;
    double comp_ratio = avg_comp_size / page_size;

    std::cout << std::dec
              << "scan_o_comment_v6: npages=" << npages
              << " sm_count=" << sm_count
              << " num_blocks=" << num_blocks
              << " pages_per_batch=" << pages_per_batch
              << " (128 threads/block = 4 warps, PAR-32K nvCOMPdx)"
              << "\n  page_size=" << page_size
              << " nbase=" << nbase << std::endl;
    std::cout << "  IO stats: total_comp_bytes=" << total_comp_bytes
              << " (" << std::fixed << std::setprecision(2)
              << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB)"
              << " avg=" << (uint32_t)avg_comp_size
              << " min=" << min_comp_size
              << " max=" << max_comp_size
              << " ratio=" << std::setprecision(3) << comp_ratio
              << std::endl;

    // Upload compressed page metadata to GPU
    uint32_t* d_comp_sizes = nullptr;
    uint64_t* d_comp_offsets = nullptr;
    {
        size_t sz = npages * sizeof(uint32_t);
        cudaMalloc(&d_comp_sizes, sz);
        cudaMemcpy(d_comp_sizes, comp_page_sizes, sz, cudaMemcpyHostToDevice);
    }
    {
        size_t sz = npages * sizeof(uint64_t);
        cudaMalloc(&d_comp_offsets, sz);
        cudaMemcpy(d_comp_offsets, comp_offsets.data(), sz, cudaMemcpyHostToDevice);
    }

    // Single IO context + decompress buffer (sequential pipeline for timing)
    bam_vchar_io_ctx_t io_ctx = bam_vchar_io_v2_create(ctrl,
        static_cast<uint32_t>(page_size), num_blocks);
    char* d_staging = bam_vchar_io_v2_staging_buf(io_ctx);

    char* d_decomp = nullptr;
    cudaMalloc(&d_decomp, (size_t)num_blocks * 4 * page_size);

    // Global accumulators
    uint64_t* d_total_records = nullptr;
    uint64_t* d_total_strlen = nullptr;
    uint64_t* d_total_byte_sum = nullptr;
    cudaMalloc(&d_total_records, sizeof(uint64_t));
    cudaMalloc(&d_total_strlen, sizeof(uint64_t));
    cudaMalloc(&d_total_byte_sum, sizeof(uint64_t));
    cudaMemset(d_total_records, 0, sizeof(uint64_t));
    cudaMemset(d_total_strlen, 0, sizeof(uint64_t));
    cudaMemset(d_total_byte_sum, 0, sizeof(uint64_t));

    // Compute stream for decompress+scan
    cudaStream_t comp_stream;
    cudaStreamCreate(&comp_stream);

    // Sequential pipeline with per-phase timing
    uint64_t io_ns_total = 0;
    uint64_t comp_ns_total = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t batch = 0; batch < npages; batch += pages_per_batch) {
        uint64_t remaining = npages - batch;
        uint32_t batch_blocks = static_cast<uint32_t>(
            std::min((uint64_t)num_blocks, (remaining + 3) / 4));

        // Phase 1: IO — read compressed pages (sync)
        auto io_start = std::chrono::high_resolution_clock::now();
        bam_vchar_io_v2_read_batch(io_ctx, d_comp_sizes, d_comp_offsets,
                                    partition_start_lba, batch,
                                    batch_blocks, npages);
        auto io_end = std::chrono::high_resolution_clock::now();
        io_ns_total += std::chrono::duration_cast<std::chrono::nanoseconds>(
            io_end - io_start).count();

        // Phase 2: Decompress + Scan (async launch, then sync)
        auto comp_start = std::chrono::high_resolution_clock::now();
        bam_vchar_decomp_scan_par32k_async(
            d_staging, d_comp_sizes, d_decomp,
            d_total_records, d_total_strlen, d_total_byte_sum,
            static_cast<uint32_t>(page_size), batch, batch_blocks,
            npages, comp_stream);
        cudaStreamSynchronize(comp_stream);
        auto comp_end = std::chrono::high_resolution_clock::now();
        comp_ns_total += std::chrono::duration_cast<std::chrono::nanoseconds>(
            comp_end - comp_start).count();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // Read results
    BAMVcharResult result = {};
    cudaMemcpy(&result.total_records, d_total_records,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_strlen, d_total_strlen,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_byte_sum, d_total_byte_sum,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double io_ms     = io_ns_total / 1e6;
    double comp_ms   = comp_ns_total / 1e6;
    double total_ms  = elapsed_ns / 1e6;

    std::cout << std::dec
              << "\n=== scan_o_comment_v6 results ===\n"
              << "  total_records  = " << result.total_records << "\n"
              << "  total_strlen   = " << result.total_strlen << "\n"
              << "  total_byte_sum = " << result.total_byte_sum << "\n"
              << std::endl;

    std::cout << std::fixed << std::setprecision(2)
              << "  IO time        = " << io_ms << " ms\n"
              << "  Decomp+Scan    = " << comp_ms << " ms\n"
              << "  Total          = " << total_ms << " ms\n"
              << std::endl;

    // IO throughput (based on IO time only)
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

    if (result.total_records == metadata.table_lineitem_nrows) {
        std::cout << "  Record count matches metadata nrows: OK" << std::endl;
    } else {
        std::cerr << "  WARNING: record count " << result.total_records
                  << " != metadata nrows " << metadata.table_lineitem_nrows << std::endl;
    }

    // Cleanup
    cudaStreamDestroy(comp_stream);
    cudaFree(d_total_records);
    cudaFree(d_total_strlen);
    cudaFree(d_total_byte_sum);
    cudaFree(d_decomp);
    cudaFree(d_comp_sizes);
    cudaFree(d_comp_offsets);
    bam_vchar_io_v2_destroy(io_ctx);
    bam_ctrl_close(ctrl);
}

// ============================================================
// scan_l_comment_v7 — Double-buffered PAR-8K nvCOMPdx
//
// Same kernel as v6 but with IO/Decomp+Scan overlap:
//   Buffer A: IO(N)  → Decomp+Scan(N)
//   Buffer B: IO(N+1) overlaps with Decomp+Scan(N)
// ============================================================
void scan_l_comment_v7(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::common::L_COMMENT;
    uint64_t field_start_page_id = metadata.table_lineitem_start_page_ids[col];
    uint64_t npages = metadata.table_lineitem_npages[col];
    uint16_t comp_method = metadata.table_lineitem_compression_method[col];

    std::cout << std::dec
              << "L_COMMENT: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method
              << " nrows=" << metadata.table_lineitem_nrows << std::endl;

    if (comp_method == 0) {
        std::cerr << "L_COMMENT is NOT compressed. Expected LZ4PAR." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes
    uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_lineitem_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
    uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    std::vector<uint64_t> comp_offsets(npages);
    for (uint64_t i = 0; i < npages; i++) {
        comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
    }

    // Auto-tune num_blocks from SM count
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks = static_cast<uint32_t>(sm_count);
    uint32_t max_blocks = static_cast<uint32_t>((npages + 3) / 4);
    if (num_blocks > max_blocks) num_blocks = max_blocks;
    const uint32_t pages_per_batch = num_blocks * 4;

    // IO metrics
    uint64_t total_comp_bytes = 0;
    uint32_t max_comp_size = 0, min_comp_size = UINT32_MAX;
    for (uint64_t i = 0; i < npages; i++) {
        uint32_t cs = comp_page_sizes[i];
        total_comp_bytes += cs;
        if (cs > max_comp_size) max_comp_size = cs;
        if (cs < min_comp_size) min_comp_size = cs;
    }
    double avg_comp_size = (double)total_comp_bytes / npages;
    double comp_ratio = avg_comp_size / page_size;

    std::cout << std::dec
              << "scan_l_comment_v7: npages=" << npages
              << " sm_count=" << sm_count
              << " num_blocks=" << num_blocks
              << " pages_per_batch=" << pages_per_batch
              << " (128 threads/block = 4 warps, PAR-8K nvCOMPdx, double-buffered)"
              << "\n  page_size=" << page_size
              << " nbase=" << nbase << std::endl;
    std::cout << "  IO stats: total_comp_bytes=" << total_comp_bytes
              << " (" << std::fixed << std::setprecision(2)
              << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB)"
              << " avg=" << (uint32_t)avg_comp_size
              << " min=" << min_comp_size
              << " max=" << max_comp_size
              << " ratio=" << std::setprecision(3) << comp_ratio
              << std::endl;

    // Upload compressed page metadata to GPU
    uint32_t* d_comp_sizes = nullptr;
    uint64_t* d_comp_offsets = nullptr;
    {
        size_t sz = npages * sizeof(uint32_t);
        cudaMalloc(&d_comp_sizes, sz);
        cudaMemcpy(d_comp_sizes, comp_page_sizes, sz, cudaMemcpyHostToDevice);
    }
    {
        size_t sz = npages * sizeof(uint64_t);
        cudaMalloc(&d_comp_offsets, sz);
        cudaMemcpy(d_comp_offsets, comp_offsets.data(), sz, cudaMemcpyHostToDevice);
    }

    // Two IO contexts for double buffering
    bam_vchar_io_ctx_t io_ctx_a = bam_vchar_io_v2_create(ctrl,
        static_cast<uint32_t>(page_size), num_blocks);
    bam_vchar_io_ctx_t io_ctx_b = bam_vchar_io_v2_create(ctrl,
        static_cast<uint32_t>(page_size), num_blocks);
    char* d_staging_a = bam_vchar_io_v2_staging_buf(io_ctx_a);
    char* d_staging_b = bam_vchar_io_v2_staging_buf(io_ctx_b);

    // Two decompress buffers
    char* d_decomp_a = nullptr;
    char* d_decomp_b = nullptr;
    cudaMalloc(&d_decomp_a, (size_t)num_blocks * 4 * page_size);
    cudaMalloc(&d_decomp_b, (size_t)num_blocks * 4 * page_size);

    // Global accumulators
    uint64_t* d_total_records = nullptr;
    uint64_t* d_total_strlen = nullptr;
    uint64_t* d_total_byte_sum = nullptr;
    cudaMalloc(&d_total_records, sizeof(uint64_t));
    cudaMalloc(&d_total_strlen, sizeof(uint64_t));
    cudaMalloc(&d_total_byte_sum, sizeof(uint64_t));
    cudaMemset(d_total_records, 0, sizeof(uint64_t));
    cudaMemset(d_total_strlen, 0, sizeof(uint64_t));
    cudaMemset(d_total_byte_sum, 0, sizeof(uint64_t));

    // Compute stream for decompress+scan
    cudaStream_t comp_stream;
    cudaStreamCreate(&comp_stream);

    // ── Double-buffered pipeline ──
    // Prime: IO batch 0 into buffer A (synchronous)
    auto t0 = std::chrono::high_resolution_clock::now();

    bam_vchar_io_ctx_t cur_io  = io_ctx_a;
    bam_vchar_io_ctx_t nxt_io  = io_ctx_b;
    char*              cur_stg = d_staging_a;
    char*              nxt_stg = d_staging_b;
    char*              cur_dec = d_decomp_a;
    char*              nxt_dec = d_decomp_b;

    uint32_t batch0_blocks = static_cast<uint32_t>(
        std::min((uint64_t)num_blocks, (npages + 3) / 4));
    bam_vchar_io_v2_read_batch_async(cur_io, d_comp_sizes, d_comp_offsets,
                                      partition_start_lba, 0, batch0_blocks, npages);
    bam_vchar_io_v2_sync(cur_io);

    for (uint64_t batch = 0; batch < npages; batch += pages_per_batch) {
        uint64_t remaining = npages - batch;
        uint32_t batch_io_blocks = static_cast<uint32_t>(
            std::min((uint64_t)num_blocks, (remaining + 3) / 4));
        uint32_t batch_pages = static_cast<uint32_t>(
            std::min((uint64_t)pages_per_batch, remaining));

        uint64_t next_batch = batch + pages_per_batch;
        bool has_next = (next_batch < npages);

        // Launch Decomp+Scan for current buffer (async on comp_stream)
        // PAR-8K: 1 page/block, grid = batch_pages
        bam_vchar_decomp_scan_par8k_async(
            cur_stg, d_comp_sizes, cur_dec,
            d_total_records, d_total_strlen, d_total_byte_sum,
            static_cast<uint32_t>(page_size), batch, batch_pages,
            npages, comp_stream);

        // Launch IO for next batch into other buffer (async, overlaps with comp)
        if (has_next) {
            uint64_t nxt_remaining = npages - next_batch;
            uint32_t nxt_io_blocks = static_cast<uint32_t>(
                std::min((uint64_t)num_blocks, (nxt_remaining + 3) / 4));
            bam_vchar_io_v2_read_batch_async(nxt_io, d_comp_sizes, d_comp_offsets,
                                              partition_start_lba, next_batch,
                                              nxt_io_blocks, npages);
        }

        // Wait for Decomp+Scan to finish
        cudaStreamSynchronize(comp_stream);

        // Wait for next IO to finish
        if (has_next) {
            bam_vchar_io_v2_sync(nxt_io);
        }

        // Swap buffers
        std::swap(cur_io,  nxt_io);
        std::swap(cur_stg, nxt_stg);
        std::swap(cur_dec, nxt_dec);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // Read results
    BAMVcharResult result = {};
    cudaMemcpy(&result.total_records, d_total_records,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_strlen, d_total_strlen,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_byte_sum, d_total_byte_sum,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double total_ms = elapsed_ns / 1e6;

    std::cout << std::dec
              << "\n=== scan_l_comment_v7 results ===\n"
              << "  total_records  = " << result.total_records << "\n"
              << "  total_strlen   = " << result.total_strlen << "\n"
              << "  total_byte_sum = " << result.total_byte_sum << "\n"
              << std::endl;

    std::cout << std::fixed << std::setprecision(2)
              << "  Total          = " << total_ms << " ms"
              << "  (double-buffered: IO overlaps Decomp+Scan)\n"
              << std::endl;

    // Throughput
    {
        double total_s = elapsed_ns / 1e9;
        double tput = (total_s > 0) ? (double)total_comp_bytes / total_s : 0;
        std::cout << "  Throughput     = " << std::fixed << std::setprecision(2)
                  << tput / (1024.0 * 1024 * 1024) << " GB/s"
                  << " (comp_bytes=" << std::setprecision(2)
                  << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB"
                  << " / " << std::setprecision(3) << total_s << " s)"
                  << std::endl;
    }

    if (result.total_records == metadata.table_lineitem_nrows) {
        std::cout << "  Record count matches metadata nrows: OK" << std::endl;
    } else {
        std::cerr << "  WARNING: record count " << result.total_records
                  << " != metadata nrows " << metadata.table_lineitem_nrows << std::endl;
    }

    // Cleanup
    cudaStreamDestroy(comp_stream);
    cudaFree(d_total_records);
    cudaFree(d_total_strlen);
    cudaFree(d_total_byte_sum);
    cudaFree(d_decomp_a);
    cudaFree(d_decomp_b);
    cudaFree(d_comp_sizes);
    cudaFree(d_comp_offsets);
    bam_vchar_io_v2_destroy(io_ctx_a);
    bam_vchar_io_v2_destroy(io_ctx_b);
    bam_ctrl_close(ctrl);
}

// ============================================================
// scan_l_comment_v8 — Double-buffered PAR-32K nvCOMPdx
//
// Same pipeline as v7 but uses PAR-32K (1 page/block, 4 warps × 8 chunks)
// instead of PAR-8K.
// ============================================================
void scan_l_comment_v8(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::common::L_COMMENT;
    uint64_t field_start_page_id = metadata.table_lineitem_start_page_ids[col];
    uint64_t npages = metadata.table_lineitem_npages[col];
    uint16_t comp_method = metadata.table_lineitem_compression_method[col];

    std::cout << std::dec
              << "L_COMMENT: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method
              << " nrows=" << metadata.table_lineitem_nrows << std::endl;

    if (comp_method == 0) {
        std::cerr << "L_COMMENT is NOT compressed. Expected LZ4PAR." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes
    uint64_t cs_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_lineitem_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_lineitem_compression_nbases[col];
    uint64_t base_start = metadata.table_lineitem_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    std::vector<uint64_t> comp_offsets(npages);
    for (uint64_t i = 0; i < npages; i++) {
        comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
    }

    // Auto-tune num_blocks from SM count
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks = static_cast<uint32_t>(sm_count);
    uint32_t max_blocks = static_cast<uint32_t>((npages + 3) / 4);
    if (num_blocks > max_blocks) num_blocks = max_blocks;
    const uint32_t pages_per_batch = num_blocks * 4;

    // IO metrics
    uint64_t total_comp_bytes = 0;
    uint32_t max_comp_size = 0, min_comp_size = UINT32_MAX;
    for (uint64_t i = 0; i < npages; i++) {
        uint32_t cs = comp_page_sizes[i];
        total_comp_bytes += cs;
        if (cs > max_comp_size) max_comp_size = cs;
        if (cs < min_comp_size) min_comp_size = cs;
    }
    double avg_comp_size = (double)total_comp_bytes / npages;
    double comp_ratio = avg_comp_size / page_size;

    std::cout << std::dec
              << "scan_l_comment_v8: npages=" << npages
              << " sm_count=" << sm_count
              << " num_blocks=" << num_blocks
              << " pages_per_batch=" << pages_per_batch
              << " (128 threads/block = 4 warps, PAR-32K nvCOMPdx, double-buffered)"
              << "\n  page_size=" << page_size
              << " nbase=" << nbase << std::endl;
    std::cout << "  IO stats: total_comp_bytes=" << total_comp_bytes
              << " (" << std::fixed << std::setprecision(2)
              << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB)"
              << " avg=" << (uint32_t)avg_comp_size
              << " min=" << min_comp_size
              << " max=" << max_comp_size
              << " ratio=" << std::setprecision(3) << comp_ratio
              << std::endl;

    // Upload compressed page metadata to GPU
    uint32_t* d_comp_sizes = nullptr;
    uint64_t* d_comp_offsets = nullptr;
    {
        size_t sz = npages * sizeof(uint32_t);
        cudaMalloc(&d_comp_sizes, sz);
        cudaMemcpy(d_comp_sizes, comp_page_sizes, sz, cudaMemcpyHostToDevice);
    }
    {
        size_t sz = npages * sizeof(uint64_t);
        cudaMalloc(&d_comp_offsets, sz);
        cudaMemcpy(d_comp_offsets, comp_offsets.data(), sz, cudaMemcpyHostToDevice);
    }

    // Two IO contexts for double buffering
    bam_vchar_io_ctx_t io_ctx_a = bam_vchar_io_v2_create(ctrl,
        static_cast<uint32_t>(page_size), num_blocks);
    bam_vchar_io_ctx_t io_ctx_b = bam_vchar_io_v2_create(ctrl,
        static_cast<uint32_t>(page_size), num_blocks);
    char* d_staging_a = bam_vchar_io_v2_staging_buf(io_ctx_a);
    char* d_staging_b = bam_vchar_io_v2_staging_buf(io_ctx_b);

    // Two decompress buffers
    char* d_decomp_a = nullptr;
    char* d_decomp_b = nullptr;
    cudaMalloc(&d_decomp_a, (size_t)num_blocks * 4 * page_size);
    cudaMalloc(&d_decomp_b, (size_t)num_blocks * 4 * page_size);

    // Global accumulators
    uint64_t* d_total_records = nullptr;
    uint64_t* d_total_strlen = nullptr;
    uint64_t* d_total_byte_sum = nullptr;
    cudaMalloc(&d_total_records, sizeof(uint64_t));
    cudaMalloc(&d_total_strlen, sizeof(uint64_t));
    cudaMalloc(&d_total_byte_sum, sizeof(uint64_t));
    cudaMemset(d_total_records, 0, sizeof(uint64_t));
    cudaMemset(d_total_strlen, 0, sizeof(uint64_t));
    cudaMemset(d_total_byte_sum, 0, sizeof(uint64_t));

    // Compute stream for decompress+scan
    cudaStream_t comp_stream;
    cudaStreamCreate(&comp_stream);

    // ── Double-buffered pipeline ──
    // Prime: IO batch 0 into buffer A (synchronous)
    auto t0 = std::chrono::high_resolution_clock::now();

    bam_vchar_io_ctx_t cur_io  = io_ctx_a;
    bam_vchar_io_ctx_t nxt_io  = io_ctx_b;
    char*              cur_stg = d_staging_a;
    char*              nxt_stg = d_staging_b;
    char*              cur_dec = d_decomp_a;
    char*              nxt_dec = d_decomp_b;

    uint32_t batch0_blocks = static_cast<uint32_t>(
        std::min((uint64_t)num_blocks, (npages + 3) / 4));
    bam_vchar_io_v2_read_batch_async(cur_io, d_comp_sizes, d_comp_offsets,
                                      partition_start_lba, 0, batch0_blocks, npages);
    bam_vchar_io_v2_sync(cur_io);

    for (uint64_t batch = 0; batch < npages; batch += pages_per_batch) {
        uint64_t remaining = npages - batch;
        uint32_t batch_io_blocks = static_cast<uint32_t>(
            std::min((uint64_t)num_blocks, (remaining + 3) / 4));
        uint32_t batch_pages = static_cast<uint32_t>(
            std::min((uint64_t)pages_per_batch, remaining));

        uint64_t next_batch = batch + pages_per_batch;
        bool has_next = (next_batch < npages);

        // Launch Decomp+Scan for current buffer (async on comp_stream)
        // PAR-32K: 1 page/block, grid = batch_pages
        bam_vchar_decomp_scan_par32k_v8_async(
            cur_stg, d_comp_sizes, cur_dec,
            d_total_records, d_total_strlen, d_total_byte_sum,
            static_cast<uint32_t>(page_size), batch, batch_pages,
            npages, comp_stream);

        // Launch IO for next batch into other buffer (async, overlaps with comp)
        if (has_next) {
            uint64_t nxt_remaining = npages - next_batch;
            uint32_t nxt_io_blocks = static_cast<uint32_t>(
                std::min((uint64_t)num_blocks, (nxt_remaining + 3) / 4));
            bam_vchar_io_v2_read_batch_async(nxt_io, d_comp_sizes, d_comp_offsets,
                                              partition_start_lba, next_batch,
                                              nxt_io_blocks, npages);
        }

        // Wait for Decomp+Scan to finish
        cudaStreamSynchronize(comp_stream);

        // Wait for next IO to finish
        if (has_next) {
            bam_vchar_io_v2_sync(nxt_io);
        }

        // Swap buffers
        std::swap(cur_io,  nxt_io);
        std::swap(cur_stg, nxt_stg);
        std::swap(cur_dec, nxt_dec);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // Read results
    BAMVcharResult result = {};
    cudaMemcpy(&result.total_records, d_total_records,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_strlen, d_total_strlen,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result.total_byte_sum, d_total_byte_sum,
               sizeof(uint64_t), cudaMemcpyDeviceToHost);

    double total_ms = elapsed_ns / 1e6;

    std::cout << std::dec
              << "\n=== scan_l_comment_v8 results ===\n"
              << "  total_records  = " << result.total_records << "\n"
              << "  total_strlen   = " << result.total_strlen << "\n"
              << "  total_byte_sum = " << result.total_byte_sum << "\n"
              << std::endl;

    std::cout << std::fixed << std::setprecision(2)
              << "  Total          = " << total_ms << " ms"
              << "  (double-buffered: IO overlaps Decomp+Scan)\n"
              << std::endl;

    // Throughput
    {
        double total_s = elapsed_ns / 1e9;
        double tput = (total_s > 0) ? (double)total_comp_bytes / total_s : 0;
        std::cout << "  Throughput     = " << std::fixed << std::setprecision(2)
                  << tput / (1024.0 * 1024 * 1024) << " GB/s"
                  << " (comp_bytes=" << std::setprecision(2)
                  << (double)total_comp_bytes / (1024.0 * 1024 * 1024) << " GB"
                  << " / " << std::setprecision(3) << total_s << " s)"
                  << std::endl;
    }

    if (result.total_records == metadata.table_lineitem_nrows) {
        std::cout << "  Record count matches metadata nrows: OK" << std::endl;
    } else {
        std::cerr << "  WARNING: record count " << result.total_records
                  << " != metadata nrows " << metadata.table_lineitem_nrows << std::endl;
    }

    // Cleanup
    cudaStreamDestroy(comp_stream);
    cudaFree(d_total_records);
    cudaFree(d_total_strlen);
    cudaFree(d_total_byte_sum);
    cudaFree(d_decomp_a);
    cudaFree(d_decomp_b);
    cudaFree(d_comp_sizes);
    cudaFree(d_comp_offsets);
    bam_vchar_io_v2_destroy(io_ctx_a);
    bam_vchar_io_v2_destroy(io_ctx_b);
    bam_ctrl_close(ctrl);
}

// ============================================================
// TPC-H Q13 — PiG (BaM) implementation
//
// Phase 0: Preload integer columns (O_CUSTKEY, C_CUSTKEY) via bam_read_page
//          → GPU → PFOR64 decompress + flatten
// Phase 1: Double-buffered O_COMMENT scan (v8 IO pipeline)
//          PAR-32K nvCOMPdx decompress + KMP pattern matching
// Phase 2: Aggregation (Sort → RLE → Probe → Sort → RLE → Pack)
// ============================================================
PigResult tpch_q13(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "=== TPC-H Q13 (PiG) ===" << std::endl;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    size_t gpu_free_pre_ctrl = 0, gpu_total_pre = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_pre);

    const uint32_t bam_num_queues = 128;
    auto ds = bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;
    const uint32_t n_devices = ds.n_devices;
    const uint64_t partition_start_lba = ds.partition_start_lba;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_pre);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;

    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    read_striped_page(0, page_size, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;

    // ── Field metadata ──
    constexpr size_t col_o_custkey = TPCH::common::O_CUSTKEY;
    constexpr size_t col_o_comment = TPCH::common::O_COMMENT;
    constexpr size_t col_c_custkey = TPCH::common::C_CUSTKEY;

    uint64_t o_custkey_start  = metadata.table_orders_start_page_ids[col_o_custkey];
    uint64_t o_custkey_npages = metadata.table_orders_npages[col_o_custkey];
    uint16_t o_custkey_comp   = metadata.table_orders_compression_method[col_o_custkey];

    uint64_t o_comment_start  = metadata.table_orders_start_page_ids[col_o_comment];
    uint64_t o_comment_npages = metadata.table_orders_npages[col_o_comment];
    uint16_t o_comment_comp   = metadata.table_orders_compression_method[col_o_comment];

    uint64_t c_custkey_start  = metadata.table_customer_start_page_ids[col_c_custkey];
    uint64_t c_custkey_npages = metadata.table_customer_npages[col_c_custkey];
    uint16_t c_custkey_comp   = metadata.table_customer_compression_method[col_c_custkey];

    uint64_t nrecs_orders   = metadata.table_orders_nrows;
    uint64_t nrecs_customer = metadata.table_customer_nrows;

    std::cout << "nrecs_orders=" << nrecs_orders
              << " nrecs_customer=" << nrecs_customer << std::endl;
    std::cout << "  O_CUSTKEY: start_page=" << o_custkey_start
              << " npages=" << o_custkey_npages
              << " compression=" << o_custkey_comp << std::endl;
    std::cout << "  O_COMMENT: start_page=" << o_comment_start
              << " npages=" << o_comment_npages
              << " compression=" << o_comment_comp << std::endl;
    std::cout << "  C_CUSTKEY: start_page=" << c_custkey_start
              << " npages=" << c_custkey_npages
              << " compression=" << c_custkey_comp << std::endl;

    const bool is_fsst = (o_comment_comp == static_cast<uint16_t>(CompressionMethod::FSST)
                        || o_comment_comp == static_cast<uint16_t>(CompressionMethod::FSST_ROWID));

    // ================================================================
    // Host-side metadata reads (before total_start)
    // ================================================================

    // ── Helper: read prefix_sum from metadata pages → host vector ──
    auto read_prefix_sum_host = [&](uint64_t ps_start, uint64_t ps_npages,
                                     uint64_t field_npages) -> std::vector<uint64_t>
    {
        if (ps_npages == 0) return {};
        std::vector<char> h_buf(ps_npages * page_size);
        for (uint64_t p = 0; p < ps_npages; p++) {
            read_striped_page(ps_start + p, page_size, h_buf.data() + p * page_size);
        }
        // prefix_sum format: [0, cum[0], cum[1], ..., cum[npages-1]=total]
        uint64_t* ps_raw = reinterpret_cast<uint64_t*>(h_buf.data()) + 1;
        return std::vector<uint64_t>(ps_raw, ps_raw + field_npages);
    };

    // ── Helper: read comp_sizes + compute comp_offsets → host vectors ──
    auto prepare_comp_metadata = [&](
        uint64_t field_start, uint64_t field_npages, uint16_t comp_method,
        uint64_t cs_start_page, uint64_t cs_npages_cnt,
        uint64_t nbase_val, uint64_t base_start_page)
        -> std::pair<std::vector<uint32_t>, std::vector<uint64_t>>
    {
        if (comp_method == 0) return {{}, {}};

        // 1. Read comp_sizes
        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            read_striped_page(cs_start_page + p, page_size, sizes_buf.data() + p * page_size);
        }
        std::vector<uint32_t> comp_sizes(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages);

        // 2. Read bases + calculate_compressed_offsets
        size_t bp_npages = TPCH::nbase_to_npages(nbase_val, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            read_striped_page(base_start_page + p, page_size, bases_buf.data() + p * page_size);
        }

        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            comp_sizes.data(), nbase_val, field_npages, page_size, field_start,
            n_devices, offsets_vec);

        std::vector<uint64_t> comp_offsets(field_npages);
        for (uint64_t i = 0; i < field_npages; i++)
            comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);

        return {comp_sizes, comp_offsets};
    };

    // ── Read O_CUSTKEY metadata ──
    std::cout << "[Q13] Reading metadata..." << std::endl;
    auto h_ps_o_custkey = read_prefix_sum_host(
        metadata.table_orders_prefix_sum_start_page_ids[col_o_custkey],
        metadata.table_orders_prefix_sum_npages[col_o_custkey],
        o_custkey_npages);
    if (h_ps_o_custkey.empty()) {
        std::cerr << "ERROR: O_CUSTKEY prefix_sum metadata not found." << std::endl;
        bam_ctrl_close(ctrl);
        return PigResult{};
    }

    auto [h_cs_o_custkey, h_co_o_custkey] = prepare_comp_metadata(
        o_custkey_start, o_custkey_npages, o_custkey_comp,
        metadata.table_orders_compressed_page_sizes_start_page_ids[col_o_custkey],
        metadata.table_orders_compressed_page_sizes_npages[col_o_custkey],
        metadata.table_orders_compression_nbases[col_o_custkey],
        metadata.table_orders_compression_base_start_page_ids[col_o_custkey]);

    // ── Read C_CUSTKEY metadata ──
    auto h_ps_c_custkey = read_prefix_sum_host(
        metadata.table_customer_prefix_sum_start_page_ids[col_c_custkey],
        metadata.table_customer_prefix_sum_npages[col_c_custkey],
        c_custkey_npages);
    if (h_ps_c_custkey.empty()) {
        std::cerr << "ERROR: C_CUSTKEY prefix_sum metadata not found." << std::endl;
        bam_ctrl_close(ctrl);
        return PigResult{};
    }

    auto [h_cs_c_custkey, h_co_c_custkey] = prepare_comp_metadata(
        c_custkey_start, c_custkey_npages, c_custkey_comp,
        metadata.table_customer_compressed_page_sizes_start_page_ids[col_c_custkey],
        metadata.table_customer_compressed_page_sizes_npages[col_c_custkey],
        metadata.table_customer_compression_nbases[col_c_custkey],
        metadata.table_customer_compression_base_start_page_ids[col_c_custkey]);

    auto [h_cs_o_comment, h_co_o_comment] = prepare_comp_metadata(
        o_comment_start, o_comment_npages, o_comment_comp,
        metadata.table_orders_compressed_page_sizes_start_page_ids[col_o_comment],
        metadata.table_orders_compressed_page_sizes_npages[col_o_comment],
        metadata.table_orders_compression_nbases[col_o_comment],
        metadata.table_orders_compression_base_start_page_ids[col_o_comment]);

    auto h_ps_o_comment = read_prefix_sum_host(
        metadata.table_orders_prefix_sum_start_page_ids[col_o_comment],
        metadata.table_orders_prefix_sum_npages[col_o_comment],
        o_comment_npages);
    if (h_ps_o_comment.empty()) {
        std::cerr << "ERROR: O_COMMENT prefix_sum metadata not found." << std::endl;
        bam_ctrl_close(ctrl);
        return PigResult{};
    }

    // ── KMP pattern tables (host-side computation) ──
    const char* patterns_str = "specialrequests";
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

    // ================================================================
    // GPU memory allocation (before total_start)
    // ================================================================

    size_t gpu_free_before = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free_before, &gpu_total);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Auto-tune num_blocks from SM count (4 blocks/SM for occupancy)
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks = static_cast<uint32_t>(sm_count) * 4;
    if (num_blocks > static_cast<uint32_t>(o_comment_npages))
        num_blocks = static_cast<uint32_t>(o_comment_npages);

    // O_CUSTKEY / C_CUSTKEY flat output arrays
    int64_t* d_o_custkey_flat_i64 = nullptr;
    int64_t* d_c_custkey_i64 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_custkey_flat_i64, nrecs_orders * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_c_custkey_i64, nrecs_customer * sizeof(int64_t)));

    // O_CUSTKEY GPU metadata
    uint64_t* d_ps_o_custkey = nullptr;
    uint32_t* d_cs_o_custkey = nullptr;
    uint64_t* d_co_o_custkey = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_o_custkey, o_custkey_npages * sizeof(uint64_t)));
    if (o_custkey_comp != 0) {
        CUDA_CHECK(cudaMalloc(&d_cs_o_custkey, o_custkey_npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_co_o_custkey, o_custkey_npages * sizeof(uint64_t)));
    }

    // C_CUSTKEY GPU metadata
    uint64_t* d_ps_c_custkey = nullptr;
    uint32_t* d_cs_c_custkey = nullptr;
    uint64_t* d_co_c_custkey = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_c_custkey, c_custkey_npages * sizeof(uint64_t)));
    if (c_custkey_comp != 0) {
        CUDA_CHECK(cudaMalloc(&d_cs_c_custkey, c_custkey_npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_co_c_custkey, c_custkey_npages * sizeof(uint64_t)));
    }

    // O_COMMENT GPU metadata (both LZ4PAR and FSST streaming paths)
    uint64_t* d_ps_o_comment = nullptr;
    uint32_t* d_comp_sizes = nullptr;
    uint64_t* d_comp_offsets_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ps_o_comment, o_comment_npages * sizeof(uint64_t)));
    if (o_comment_comp != 0) {
        CUDA_CHECK(cudaMalloc(&d_comp_sizes, o_comment_npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_comp_offsets_gpu, o_comment_npages * sizeof(uint64_t)));
    }

    // Aggregation output
    uint64_t* d_o_aggr_custkey = nullptr;
    uint64_t* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o_aggr_custkey, nrecs_orders * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint64_t)));

    // KMP pattern tables
    char* d_patterns = nullptr;
    int* d_next = nullptr;
    int* d_pattern_offsets = nullptr;
    int* d_pattern_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_patterns, total_pattern_chars));
    CUDA_CHECK(cudaMalloc(&d_next, total_pattern_chars * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pattern_offsets, num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pattern_lengths, num_patterns * sizeof(int)));

    // ── PFOR64 I/O context (page cache for O_CUSTKEY / C_CUSTKEY) ──
    bam_pfor64_io_ctx_t pfor64_io_ctx = bam_pfor64_io_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks);

    // ── Fused IO+Decomp+Scan context for O_COMMENT ──
    bam_q13_fused_io_ctx_t fused_ctx{};
    bam_q13_fsst_io_ctx_t fsst_ctx{};
    if (is_fsst) {
        fsst_ctx = bam_q13_fsst_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks);
    } else {
        fused_ctx = bam_q13_fused_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks);
    }

    std::cout << "[Q13] O_COMMENT scan: npages=" << o_comment_npages
              << " sm_count=" << sm_count
              << " num_blocks=" << num_blocks
              << (is_fsst ? " (FSST, BaM+decomp+scan)" : " (PAR-32K, fused IO+decomp+scan)")
              << std::endl;

    // Phase timing counters (clock64 cycles: io, symtab/decomp, scan)
    uint64_t* d_phase_cycles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_phase_cycles, 3 * sizeof(uint64_t)));

    uint64_t h_count = 0;
    std::vector<std::pair<uint32_t, uint32_t>> q13_result;

    // ── Pre-compute I/O statistics from metadata ──
    uint64_t q13_nios = o_custkey_npages + o_comment_npages + c_custkey_npages;
    uint64_t q13_read_bytes = 0;
    auto add_col_io_q13 = [&](uint64_t npages, uint16_t comp, const std::vector<uint32_t>& h_cs) {
        for (uint64_t pg = 0; pg < npages; pg++)
            q13_read_bytes += (comp != 0 && pg < h_cs.size()) ? h_cs[pg] : page_size;
    };
    add_col_io_q13(o_custkey_npages, o_custkey_comp, h_cs_o_custkey);
    add_col_io_q13(o_comment_npages, o_comment_comp, h_cs_o_comment);
    add_col_io_q13(c_custkey_npages, c_custkey_comp, h_cs_c_custkey);

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

    // Upload metadata before measurement (Rule 3)
    cudaMemcpy(d_ps_o_custkey, h_ps_o_custkey.data(),
               o_custkey_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (o_custkey_comp != 0) {
        cudaMemcpy(d_cs_o_custkey, h_cs_o_custkey.data(),
                   o_custkey_npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_co_o_custkey, h_co_o_custkey.data(),
                   o_custkey_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_ps_c_custkey, h_ps_c_custkey.data(),
               c_custkey_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (c_custkey_comp != 0) {
        cudaMemcpy(d_cs_c_custkey, h_cs_c_custkey.data(),
                   c_custkey_npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_co_c_custkey, h_co_c_custkey.data(),
                   c_custkey_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // Upload KMP tables + O_COMMENT metadata before measurement (Rule 3)
    cudaMemcpy(d_patterns, patterns_str, total_pattern_chars, cudaMemcpyHostToDevice);
    cudaMemcpy(d_next, next_h.data(), total_pattern_chars * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern_offsets, pattern_offsets_h,
               num_patterns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern_lengths, pattern_lengths_h,
               num_patterns * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_ps_o_comment, h_ps_o_comment.data(),
               o_comment_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (o_comment_comp != 0) {
        cudaMemcpy(d_comp_sizes, h_cs_o_comment.data(),
                   o_comment_npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_comp_offsets_gpu, h_co_o_comment.data(),
                   o_comment_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    size_t gpu_free_after = 0;
    cudaMemGetInfo(&gpu_free_after, &gpu_total);
    uint64_t q13_gpu_mem_bytes = gpu_free_before - gpu_free_after;

    // ================================================================
    // total_start — query processing begins
    // ================================================================
    auto total_start = std::chrono::high_resolution_clock::now();
    s_kernel_launches = 0;
    using hrc = std::chrono::high_resolution_clock;
    auto dur_ms = [](hrc::time_point a, hrc::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // ── Phase 0a: O_CUSTKEY GPU transfer + BaM flatten ──
    std::cout << "[Q13] Loading O_CUSTKEY (" << o_custkey_npages
              << " pages, GPU-initiated I/O)..." << std::endl;

    BAMPfor64FlattenParams params_o{};
    params_o.partition_start_lba = partition_start_lba;
    for (uint32_t d = 0; d < n_devices; d++)
        params_o.partition_start_lbas[d] = ds.partition_start_lbas[d];
    params_o.n_devices = n_devices;
    params_o.page_size = static_cast<uint32_t>(page_size);
    params_o.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
    params_o.comp_method = o_custkey_comp;
    params_o.field_start_page_id = o_custkey_start;
    params_o.npages = o_custkey_npages;
    params_o.nrows = nrecs_orders;
    params_o.num_blocks = num_blocks;
    params_o.d_prefix_sum = d_ps_o_custkey;
    params_o.d_comp_sizes = d_cs_o_custkey;
    params_o.d_comp_offsets = d_co_o_custkey;

    bam_pfor64_io_flatten_async(pfor64_io_ctx, params_o, d_o_custkey_flat_i64, stream);
    s_kernel_launches++;
    cudaStreamSynchronize(stream);
    uint64_t* d_o_custkey_flat = reinterpret_cast<uint64_t*>(d_o_custkey_flat_i64);
    auto t_o_custkey = hrc::now();

    // ── Phase 0b: C_CUSTKEY BaM flatten ──
    std::cout << "[Q13] Loading C_CUSTKEY (" << c_custkey_npages
              << " pages, GPU-initiated I/O)..." << std::endl;

    BAMPfor64FlattenParams params_c{};
    params_c.partition_start_lba = partition_start_lba;
    for (uint32_t d = 0; d < n_devices; d++)
        params_c.partition_start_lbas[d] = ds.partition_start_lbas[d];
    params_c.n_devices = n_devices;
    params_c.page_size = static_cast<uint32_t>(page_size);
    params_c.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
    params_c.comp_method = c_custkey_comp;
    params_c.field_start_page_id = c_custkey_start;
    params_c.npages = c_custkey_npages;
    params_c.nrows = nrecs_customer;
    params_c.num_blocks = num_blocks;
    params_c.d_prefix_sum = d_ps_c_custkey;
    params_c.d_comp_sizes = d_cs_c_custkey;
    params_c.d_comp_offsets = d_co_c_custkey;

    bam_pfor64_io_flatten_async(pfor64_io_ctx, params_c, d_c_custkey_i64, stream);
    s_kernel_launches++;
    cudaStreamSynchronize(stream);
    uint64_t* d_c_custkey = reinterpret_cast<uint64_t*>(d_c_custkey_i64);
    auto t_c_custkey = hrc::now();

    // ── Phase 1: O_COMMENT fused IO+decomp+scan ──
    cudaMemsetAsync(d_o_aggr_custkey, 0xFF, nrecs_orders * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream);

    // Ensure memset is done before scan starts
    cudaStreamSynchronize(stream);
    auto t_metadata = hrc::now();

    auto scan_start = std::chrono::high_resolution_clock::now();

    if (is_fsst) {
        // ── Phase 2 (FSST): Fused IO+Decomp+Scan (streaming) ──
        cudaMemsetAsync(d_phase_cycles, 0, 3 * sizeof(uint64_t), stream);

        BAMq13FsstParams fsst_params{};
        fsst_params.d_comp_sizes       = d_comp_sizes;
        fsst_params.d_comp_offsets     = d_comp_offsets_gpu;
        fsst_params.d_prefix_sum       = d_ps_o_comment;
        fsst_params.d_o_custkey_flat   = d_o_custkey_flat;
        fsst_params.d_o_aggr_custkey   = d_o_aggr_custkey;
        fsst_params.d_count            = d_count;
        fsst_params.d_patterns         = d_patterns;
        fsst_params.d_next             = d_next;
        fsst_params.d_pattern_offsets  = d_pattern_offsets;
        fsst_params.d_pattern_lengths  = d_pattern_lengths;
        fsst_params.num_patterns       = num_patterns;
        for (uint32_t d = 0; d < n_devices; d++)
            fsst_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        fsst_params.n_devices          = n_devices;
        fsst_params.field_start_page_id = o_comment_start;
        fsst_params.page_size          = static_cast<uint32_t>(page_size);
        fsst_params.npages             = o_comment_npages;
        fsst_params.num_blocks         = num_blocks;
        fsst_params.nrecs_total        = nrecs_orders;
        fsst_params.d_phase_cycles     = d_phase_cycles;

        bam_q13_fsst_o_comment_async(fsst_ctx, fsst_params, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    } else {
        // ── Phase 2 (LZ4PAR): Fused IO+Decomp+Scan for O_COMMENT ──
        cudaMemsetAsync(d_phase_cycles, 0, 3 * sizeof(uint64_t), stream);

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
        fused_params.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            fused_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        fused_params.n_devices = n_devices;
        fused_params.field_start_page_id = o_comment_start;
        fused_params.page_size         = static_cast<uint32_t>(page_size);
        fused_params.comp_method       = o_comment_comp;
        fused_params.npages            = o_comment_npages;
        fused_params.num_blocks        = num_blocks;
        fused_params.d_phase_cycles    = d_phase_cycles;

        bam_q13_fused_io_decomp_scan_async(fused_ctx, fused_params, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }

    auto scan_end = std::chrono::high_resolution_clock::now();

    // Read qualifying count
    h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    std::cout << "[Q13] Qualifying orders (NOT LIKE): " << h_count
              << " / " << nrecs_orders << std::endl;

    // ── Phase 3: Aggregation ──
    std::cout << "[Q13] Running aggregation pipeline..." << std::endl;
    q13_result.clear();
    q13_pig_aggregate(q13_bufs, d_o_aggr_custkey, nrecs_orders,
                       d_c_custkey, nrecs_customer,
                       q13_result, stream);
    s_kernel_launches++;
    auto t_aggr = hrc::now();

    // ================================================================
    // total_end — query processing ends
    // ================================================================
    auto total_end = std::chrono::high_resolution_clock::now();
    double scan_ms = std::chrono::duration<double, std::milli>(scan_end - scan_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // ── Timing breakdown ──
    {
        std::cout << "\n--- Q13 PiG Phase Timing Breakdown ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  O_CUSTKEY  (pfor64 flatten):   " << dur_ms(total_start, t_o_custkey) << " ms" << std::endl;
        std::cout << "  C_CUSTKEY  (pfor64 flatten):   " << dur_ms(t_o_custkey, t_c_custkey) << " ms" << std::endl;
        std::cout << "  Metadata   (H->D + KMP):       " << dur_ms(t_c_custkey, t_metadata) << " ms" << std::endl;
        if (is_fsst) {
            std::cout << "  O_COMMENT  (FSST fused IO+decomp+scan): " << scan_ms << " ms" << std::endl;
            {
                uint64_t h_phase_cycles[3] = {};
                cudaMemcpy(h_phase_cycles, d_phase_cycles, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                int clock_rate_khz = 0;
                cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
                uint64_t pages_per_block = (o_comment_npages + num_blocks - 1) / num_blocks;
                double io_ms_     = (double)h_phase_cycles[0] / num_blocks / clock_rate_khz;
                double symtab_ms  = (double)h_phase_cycles[1] / num_blocks / clock_rate_khz;
                double scan_ms_   = (double)h_phase_cycles[2] / num_blocks / clock_rate_khz;
                std::cout << "    GPU clock: " << clock_rate_khz << " kHz"
                          << "  num_blocks=" << num_blocks
                          << "  pages_per_block=" << pages_per_block << std::endl;
                std::cout << "    Phase IO:      " << io_ms_     << " ms  (avg/page: "
                          << io_ms_ / pages_per_block << " ms)" << std::endl;
                std::cout << "    Phase Symtab:  " << symtab_ms  << " ms  (avg/page: "
                          << symtab_ms / pages_per_block << " ms)" << std::endl;
                std::cout << "    Phase Scan:    " << scan_ms_   << " ms  (avg/page: "
                          << scan_ms_ / pages_per_block << " ms)" << std::endl;
                std::cout << "    Phase Total:   " << (io_ms_ + symtab_ms + scan_ms_) << " ms" << std::endl;
            }
        } else {
            std::cout << "  O_COMMENT  (fused IO+decomp+scan): " << scan_ms << " ms" << std::endl;
            {
                uint64_t h_phase_cycles[3] = {};
                cudaMemcpy(h_phase_cycles, d_phase_cycles, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                int clock_rate_khz = 0;
                cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
                uint32_t total_warps = num_blocks * 4;
                double io_ms_    = (double)h_phase_cycles[0] / total_warps / clock_rate_khz;
                double decomp_ms = (double)h_phase_cycles[1] / total_warps / clock_rate_khz;
                double scan_ms_  = (double)h_phase_cycles[2] / total_warps / clock_rate_khz;
                uint64_t pages_per_warp = (o_comment_npages + total_warps - 1) / total_warps;
                std::cout << "    GPU clock: " << clock_rate_khz << " kHz"
                          << "  total_warps=" << total_warps
                          << "  pages_per_warp=" << pages_per_warp << std::endl;
                std::cout << "    Phase IO:     " << io_ms_    << " ms  (avg/page: "
                          << io_ms_ / pages_per_warp << " ms)" << std::endl;
                std::cout << "    Phase Decomp: " << decomp_ms << " ms  (avg/page: "
                          << decomp_ms / pages_per_warp << " ms)" << std::endl;
                std::cout << "    Phase Scan:   " << scan_ms_  << " ms  (avg/page: "
                          << scan_ms_ / pages_per_warp << " ms)" << std::endl;
                std::cout << "    Phase Total:  " << (io_ms_ + decomp_ms + scan_ms_) << " ms" << std::endl;
            }
        }
        std::cout << "  Aggregate  (sort+RLE+probe):   " << dur_ms(scan_end, t_aggr) << " ms" << std::endl;
        std::cout << "  ─────────────────────────────────────" << std::endl;
        double pfor_total = dur_ms(total_start, t_o_custkey) + dur_ms(t_o_custkey, t_c_custkey);
        std::cout << "  PFOR total (2x pfor64):        " << pfor_total << " ms" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
    }

    // ── Print results ──
    std::cout << "\n=== TPC-H Q13 Result ===" << std::endl;
    std::cout << "c_count | custdist" << std::endl;
    std::cout << "--------+---------" << std::endl;
    for (auto& [c_count, custdist] : q13_result) {
        printf("%7u | %8u\n", c_count, custdist);
    }

    // ================================================================
    // Cleanup (after total_end)
    // ================================================================
    bam_pfor64_io_destroy(pfor64_io_ctx);
    if (is_fsst) {
        bam_q13_fsst_io_destroy(fsst_ctx);
    } else {
        bam_q13_fused_io_destroy(fused_ctx);
    }
    cudaFree(d_phase_cycles);
    cudaFree(d_comp_sizes);
    cudaFree(d_comp_offsets_gpu);
    cudaFree(d_ps_o_comment);
    cudaFree(d_ps_o_custkey);
    cudaFree(d_cs_o_custkey);
    cudaFree(d_co_o_custkey);
    cudaFree(d_ps_c_custkey);
    cudaFree(d_cs_c_custkey);
    cudaFree(d_co_c_custkey);
    cudaFree(d_o_aggr_custkey);
    cudaFree(d_count);
    cudaFree(d_patterns);
    cudaFree(d_next);
    cudaFree(d_pattern_offsets);
    cudaFree(d_pattern_lengths);
    cudaFree(d_o_custkey_flat_i64);
    cudaFree(d_c_custkey_i64);
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
    cudaStreamDestroy(stream);
    bam_ctrl_close(ctrl);
    return PigResult{total_ms, q13_nios, q13_read_bytes,
                     collect_comp_methods({o_custkey_comp, o_comment_comp, c_custkey_comp}),
                     gpu_ctrl_bytes + q13_gpu_mem_bytes, gpu_ctrl_bytes, q13_gpu_mem_bytes,
                     o_custkey_npages + o_comment_npages + c_custkey_npages,
                     s_kernel_launches};
}

// ============================================================
// TPC-H Q5 (PiG) — GPU-initiated I/O via BaM
// ============================================================

PigResult tpch_q5(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "=== TPC-H Q5 (PiG) ===" << std::endl;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    size_t gpu_free_pre_ctrl = 0, gpu_total_pre = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_pre);

    const uint32_t bam_num_queues = 128;
    auto ds = bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;
    const uint32_t n_devices = ds.n_devices;
    const uint64_t partition_start_lba = ds.partition_start_lba;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_pre);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;

    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    read_striped_page(0, page_size, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;

    // ── Field metadata ──
    constexpr size_t col_r_regionkey = TPCH::common::R_REGIONKEY;
    constexpr size_t col_r_name      = TPCH::common::R_NAME;
    constexpr size_t col_n_nationkey = TPCH::common::N_NATIONKEY;
    constexpr size_t col_n_name      = TPCH::common::N_NAME;
    constexpr size_t col_n_regionkey = TPCH::common::N_REGIONKEY;
    constexpr size_t col_s_suppkey   = TPCH::common::S_SUPPKEY;
    constexpr size_t col_s_nationkey = TPCH::common::S_NATIONKEY;
    constexpr size_t col_c_custkey   = TPCH::common::C_CUSTKEY;
    constexpr size_t col_c_nationkey = TPCH::common::C_NATIONKEY;
    constexpr size_t col_o_orderkey  = TPCH::common::O_ORDERKEY;
    constexpr size_t col_o_custkey   = TPCH::common::O_CUSTKEY;
    constexpr size_t col_o_orderdate = TPCH::common::O_ORDERDATE;
    constexpr size_t col_l_orderkey  = TPCH::common::L_ORDERKEY;
    constexpr size_t col_l_suppkey   = TPCH::common::L_SUPPKEY;
    constexpr size_t col_l_extprice  = TPCH::common::L_EXTENDEDPRICE;
    constexpr size_t col_l_discount  = TPCH::common::L_DISCOUNT;

    // REGION fields (host-read)
    uint64_t r_rkey_start  = metadata.table_region_start_page_ids[col_r_regionkey];
    uint64_t r_rkey_npages = metadata.table_region_npages[col_r_regionkey];
    uint64_t r_name_start  = metadata.table_region_start_page_ids[col_r_name];
    uint64_t r_name_npages = metadata.table_region_npages[col_r_name];

    // NATION fields (host-read)
    uint64_t n_nkey_start  = metadata.table_nation_start_page_ids[col_n_nationkey];
    uint64_t n_nkey_npages = metadata.table_nation_npages[col_n_nationkey];
    uint64_t n_name_start  = metadata.table_nation_start_page_ids[col_n_name];
    uint64_t n_name_npages = metadata.table_nation_npages[col_n_name];
    uint64_t n_rkey_start  = metadata.table_nation_start_page_ids[col_n_regionkey];
    uint64_t n_rkey_npages = metadata.table_nation_npages[col_n_regionkey];

    // SUPPLIER fields
    uint64_t s_suppkey_start  = metadata.table_supplier_start_page_ids[col_s_suppkey];
    uint64_t s_suppkey_npages = metadata.table_supplier_npages[col_s_suppkey];
    uint16_t s_suppkey_comp   = metadata.table_supplier_compression_method[col_s_suppkey];

    uint64_t s_nationkey_start  = metadata.table_supplier_start_page_ids[col_s_nationkey];
    uint64_t s_nationkey_npages = metadata.table_supplier_npages[col_s_nationkey];
    uint16_t s_nationkey_comp   = metadata.table_supplier_compression_method[col_s_nationkey];

    // CUSTOMER fields
    uint64_t c_custkey_start  = metadata.table_customer_start_page_ids[col_c_custkey];
    uint64_t c_custkey_npages = metadata.table_customer_npages[col_c_custkey];
    uint16_t c_custkey_comp   = metadata.table_customer_compression_method[col_c_custkey];

    uint64_t c_nationkey_start  = metadata.table_customer_start_page_ids[col_c_nationkey];
    uint64_t c_nationkey_npages = metadata.table_customer_npages[col_c_nationkey];
    uint16_t c_nationkey_comp   = metadata.table_customer_compression_method[col_c_nationkey];

    // ORDERS fields
    uint64_t o_orderkey_start  = metadata.table_orders_start_page_ids[col_o_orderkey];
    uint64_t o_orderkey_npages = metadata.table_orders_npages[col_o_orderkey];
    uint16_t o_orderkey_comp   = metadata.table_orders_compression_method[col_o_orderkey];

    uint64_t o_custkey_start  = metadata.table_orders_start_page_ids[col_o_custkey];
    uint64_t o_custkey_npages = metadata.table_orders_npages[col_o_custkey];
    uint16_t o_custkey_comp   = metadata.table_orders_compression_method[col_o_custkey];

    uint64_t o_orderdate_start  = metadata.table_orders_start_page_ids[col_o_orderdate];
    uint64_t o_orderdate_npages = metadata.table_orders_npages[col_o_orderdate];
    uint16_t o_orderdate_comp   = metadata.table_orders_compression_method[col_o_orderdate];

    // LINEITEM fields
    uint64_t l_orderkey_start  = metadata.table_lineitem_start_page_ids[col_l_orderkey];
    uint64_t l_orderkey_npages = metadata.table_lineitem_npages[col_l_orderkey];
    uint16_t l_orderkey_comp   = metadata.table_lineitem_compression_method[col_l_orderkey];

    uint64_t l_suppkey_start  = metadata.table_lineitem_start_page_ids[col_l_suppkey];
    uint64_t l_suppkey_npages = metadata.table_lineitem_npages[col_l_suppkey];
    uint16_t l_suppkey_comp   = metadata.table_lineitem_compression_method[col_l_suppkey];

    uint64_t l_extprice_start  = metadata.table_lineitem_start_page_ids[col_l_extprice];
    uint64_t l_extprice_npages = metadata.table_lineitem_npages[col_l_extprice];
    uint16_t l_extprice_comp   = metadata.table_lineitem_compression_method[col_l_extprice];

    uint64_t l_discount_start  = metadata.table_lineitem_start_page_ids[col_l_discount];
    uint64_t l_discount_npages = metadata.table_lineitem_npages[col_l_discount];
    uint16_t l_discount_comp   = metadata.table_lineitem_compression_method[col_l_discount];

    uint64_t nrecs_supplier = metadata.table_supplier_nrows;
    uint64_t nrecs_customer = metadata.table_customer_nrows;
    uint64_t nrecs_orders   = metadata.table_orders_nrows;
    uint64_t nrecs_lineitem = metadata.table_lineitem_nrows;

    std::cout << "nrecs: supplier=" << nrecs_supplier
              << ", customer=" << nrecs_customer
              << ", orders=" << nrecs_orders
              << ", lineitem=" << nrecs_lineitem << std::endl;

    // ================================================================
    // Host-side metadata reads (before total_start)
    // ================================================================

    auto read_prefix_sum_host = [&](uint64_t ps_start, uint64_t ps_npages,
                                     uint64_t field_npages) -> std::vector<uint64_t>
    {
        if (ps_npages == 0) return {};
        std::vector<char> h_buf(ps_npages * page_size);
        for (uint64_t p = 0; p < ps_npages; p++) {
            read_striped_page(ps_start + p, page_size, h_buf.data() + p * page_size);
        }
        uint64_t* ps_raw = reinterpret_cast<uint64_t*>(h_buf.data()) + 1;
        return std::vector<uint64_t>(ps_raw, ps_raw + field_npages);
    };

    auto prepare_comp_metadata = [&](
        uint64_t field_start, uint64_t field_npages, uint16_t comp_method,
        uint64_t cs_start_page, uint64_t cs_npages_cnt,
        uint64_t nbase_val, uint64_t base_start_page)
        -> std::pair<std::vector<uint32_t>, std::vector<uint64_t>>
    {
        if (comp_method == 0) return {{}, {}};
        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            read_striped_page(cs_start_page + p, page_size, sizes_buf.data() + p * page_size);
        }
        std::vector<uint32_t> comp_sizes(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages);
        size_t bp_npages = TPCH::nbase_to_npages(nbase_val, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            read_striped_page(base_start_page + p, page_size, bases_buf.data() + p * page_size);
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            comp_sizes.data(), nbase_val, field_npages, page_size, field_start,
            n_devices, offsets_vec);
        std::vector<uint64_t> comp_offsets(field_npages);
        for (uint64_t i = 0; i < field_npages; i++)
            comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
        return {comp_sizes, comp_offsets};
    };

    std::cout << "[Q5] Reading metadata..." << std::endl;

    // SUPPLIER metadata
    auto h_ps_s_suppkey = read_prefix_sum_host(
        metadata.table_supplier_prefix_sum_start_page_ids[col_s_suppkey],
        metadata.table_supplier_prefix_sum_npages[col_s_suppkey],
        s_suppkey_npages);
    auto [h_cs_s_suppkey, h_co_s_suppkey] = prepare_comp_metadata(
        s_suppkey_start, s_suppkey_npages, s_suppkey_comp,
        metadata.table_supplier_compressed_page_sizes_start_page_ids[col_s_suppkey],
        metadata.table_supplier_compressed_page_sizes_npages[col_s_suppkey],
        metadata.table_supplier_compression_nbases[col_s_suppkey],
        metadata.table_supplier_compression_base_start_page_ids[col_s_suppkey]);

    auto h_ps_s_nationkey = read_prefix_sum_host(
        metadata.table_supplier_prefix_sum_start_page_ids[col_s_nationkey],
        metadata.table_supplier_prefix_sum_npages[col_s_nationkey],
        s_nationkey_npages);
    auto [h_cs_s_nationkey, h_co_s_nationkey] = prepare_comp_metadata(
        s_nationkey_start, s_nationkey_npages, s_nationkey_comp,
        metadata.table_supplier_compressed_page_sizes_start_page_ids[col_s_nationkey],
        metadata.table_supplier_compressed_page_sizes_npages[col_s_nationkey],
        metadata.table_supplier_compression_nbases[col_s_nationkey],
        metadata.table_supplier_compression_base_start_page_ids[col_s_nationkey]);

    // CUSTOMER metadata
    auto h_ps_c_custkey = read_prefix_sum_host(
        metadata.table_customer_prefix_sum_start_page_ids[col_c_custkey],
        metadata.table_customer_prefix_sum_npages[col_c_custkey],
        c_custkey_npages);
    auto [h_cs_c_custkey, h_co_c_custkey] = prepare_comp_metadata(
        c_custkey_start, c_custkey_npages, c_custkey_comp,
        metadata.table_customer_compressed_page_sizes_start_page_ids[col_c_custkey],
        metadata.table_customer_compressed_page_sizes_npages[col_c_custkey],
        metadata.table_customer_compression_nbases[col_c_custkey],
        metadata.table_customer_compression_base_start_page_ids[col_c_custkey]);

    auto h_ps_c_nationkey = read_prefix_sum_host(
        metadata.table_customer_prefix_sum_start_page_ids[col_c_nationkey],
        metadata.table_customer_prefix_sum_npages[col_c_nationkey],
        c_nationkey_npages);
    auto [h_cs_c_nationkey, h_co_c_nationkey] = prepare_comp_metadata(
        c_nationkey_start, c_nationkey_npages, c_nationkey_comp,
        metadata.table_customer_compressed_page_sizes_start_page_ids[col_c_nationkey],
        metadata.table_customer_compressed_page_sizes_npages[col_c_nationkey],
        metadata.table_customer_compression_nbases[col_c_nationkey],
        metadata.table_customer_compression_base_start_page_ids[col_c_nationkey]);

    // ORDERS metadata
    auto h_ps_o_orderkey = read_prefix_sum_host(
        metadata.table_orders_prefix_sum_start_page_ids[col_o_orderkey],
        metadata.table_orders_prefix_sum_npages[col_o_orderkey],
        o_orderkey_npages);
    auto [h_cs_o_orderkey, h_co_o_orderkey] = prepare_comp_metadata(
        o_orderkey_start, o_orderkey_npages, o_orderkey_comp,
        metadata.table_orders_compressed_page_sizes_start_page_ids[col_o_orderkey],
        metadata.table_orders_compressed_page_sizes_npages[col_o_orderkey],
        metadata.table_orders_compression_nbases[col_o_orderkey],
        metadata.table_orders_compression_base_start_page_ids[col_o_orderkey]);

    auto h_ps_o_custkey = read_prefix_sum_host(
        metadata.table_orders_prefix_sum_start_page_ids[col_o_custkey],
        metadata.table_orders_prefix_sum_npages[col_o_custkey],
        o_custkey_npages);
    auto [h_cs_o_custkey, h_co_o_custkey] = prepare_comp_metadata(
        o_custkey_start, o_custkey_npages, o_custkey_comp,
        metadata.table_orders_compressed_page_sizes_start_page_ids[col_o_custkey],
        metadata.table_orders_compressed_page_sizes_npages[col_o_custkey],
        metadata.table_orders_compression_nbases[col_o_custkey],
        metadata.table_orders_compression_base_start_page_ids[col_o_custkey]);

    auto h_ps_o_orderdate = read_prefix_sum_host(
        metadata.table_orders_prefix_sum_start_page_ids[col_o_orderdate],
        metadata.table_orders_prefix_sum_npages[col_o_orderdate],
        o_orderdate_npages);
    auto [h_cs_o_orderdate, h_co_o_orderdate] = prepare_comp_metadata(
        o_orderdate_start, o_orderdate_npages, o_orderdate_comp,
        metadata.table_orders_compressed_page_sizes_start_page_ids[col_o_orderdate],
        metadata.table_orders_compressed_page_sizes_npages[col_o_orderdate],
        metadata.table_orders_compression_nbases[col_o_orderdate],
        metadata.table_orders_compression_base_start_page_ids[col_o_orderdate]);

    // LINEITEM metadata
    auto h_ps_l_orderkey = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_orderkey],
        metadata.table_lineitem_prefix_sum_npages[col_l_orderkey],
        l_orderkey_npages);
    auto [h_cs_l_orderkey, h_co_l_orderkey] = prepare_comp_metadata(
        l_orderkey_start, l_orderkey_npages, l_orderkey_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_orderkey],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_orderkey],
        metadata.table_lineitem_compression_nbases[col_l_orderkey],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_orderkey]);

    auto h_ps_l_suppkey = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_suppkey],
        metadata.table_lineitem_prefix_sum_npages[col_l_suppkey],
        l_suppkey_npages);
    auto [h_cs_l_suppkey, h_co_l_suppkey] = prepare_comp_metadata(
        l_suppkey_start, l_suppkey_npages, l_suppkey_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_suppkey],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_suppkey],
        metadata.table_lineitem_compression_nbases[col_l_suppkey],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_suppkey]);

    auto h_ps_l_extprice = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_extprice],
        metadata.table_lineitem_prefix_sum_npages[col_l_extprice],
        l_extprice_npages);
    auto [h_cs_l_extprice, h_co_l_extprice] = prepare_comp_metadata(
        l_extprice_start, l_extprice_npages, l_extprice_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_extprice],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_extprice],
        metadata.table_lineitem_compression_nbases[col_l_extprice],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_extprice]);

    auto h_ps_l_discount = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_discount],
        metadata.table_lineitem_prefix_sum_npages[col_l_discount],
        l_discount_npages);
    auto [h_cs_l_discount, h_co_l_discount] = prepare_comp_metadata(
        l_discount_start, l_discount_npages, l_discount_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_discount],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_discount],
        metadata.table_lineitem_compression_nbases[col_l_discount],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_discount]);

    // ================================================================
    // Zone map metadata extraction (before total_start, only if -Z)
    // ================================================================

    bool zonemap_enabled = options.enable_zonemap;

    // Date range (needed for zone map pruning logic too)
    const int32_t date_low  = options.q6_sd_low  ? options.q6_sd_low  : 19940101;
    const int32_t date_high = options.q6_sd_high ? options.q6_sd_high : 19950101;

    // Extract metadata numbers for zone map (no CPU reads of stats pages)
    const size_t o_odate_field = TPCH::common::O_ORDERDATE;
    uint64_t odate_nstats      = metadata.table_orders_nstats[o_odate_field];
    uint64_t odate_stats_start = metadata.table_orders_stats_start_page_ids[o_odate_field];
    uint64_t odate_stats_npg   = metadata.table_orders_stats_npages[o_odate_field];

    const size_t sw_rname_idx = TPCH::common::OS_SIDEWAYS_R_NAME;
    const int32_t asia_dict_id = 2;
    uint64_t rname_nstats      = metadata.table_orders_sideways_nstats[o_odate_field][sw_rname_idx];
    uint64_t rname_stats_start = metadata.table_orders_sideways_stats_start_page_ids[o_odate_field][sw_rname_idx];
    uint64_t rname_stats_npg   = metadata.table_orders_sideways_stats_npages[o_odate_field][sw_rname_idx];

    const size_t li_ref_field = TPCH::common::L_EXTENDEDPRICE;
    const size_t li_sw_odate_idx = TPCH::common::LS_SIDEWAYS_O_ORDERDATE;
    uint64_t li_odate_nstats      = metadata.table_lineitem_sideways_nstats[li_ref_field][li_sw_odate_idx];
    uint64_t li_odate_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[li_ref_field][li_sw_odate_idx];
    uint64_t li_odate_stats_npg   = metadata.table_lineitem_sideways_stats_npages[li_ref_field][li_sw_odate_idx];

    const size_t li_sw_rname_idx = TPCH::common::LS_SIDEWAYS_R_NAME;
    uint64_t li_rname_nstats      = metadata.table_lineitem_sideways_nstats[li_ref_field][li_sw_rname_idx];
    uint64_t li_rname_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[li_ref_field][li_sw_rname_idx];
    uint64_t li_rname_stats_npg   = metadata.table_lineitem_sideways_stats_npages[li_ref_field][li_sw_rname_idx];

    // Check if any stats are available; if not, disable zonemap
    if (zonemap_enabled) {
        bool has_ord_stats = (odate_nstats > 0 && odate_stats_start > 0) ||
                             (rname_nstats > 0 && rname_stats_start > 0);
        bool has_li_stats  = (li_odate_nstats > 0 && li_odate_stats_start > 0) ||
                             (li_rname_nstats > 0 && li_rname_stats_start > 0);
        if (!has_ord_stats && !has_li_stats) {
            std::cout << "[Q5-ZONEMAP] No stats available, disabling zonemap" << std::endl;
            zonemap_enabled = false;
        }
    }

    // ================================================================
    // Fused kernel setup
    // ================================================================

    const uint64_t o_npages_i32 = o_orderdate_npages;
    const uint64_t o_npages_i64 = o_orderkey_npages;  // = o_custkey_npages
    const uint64_t l_npages_i32 = l_extprice_npages;
    const uint64_t l_npages_i64 = l_orderkey_npages;  // = l_suppkey_npages

    // Compute max rows per INT32 page (for scratch buffer sizing)
    uint64_t o_max_rows_per_page = 0;
    for (uint64_t i = 0; i < o_npages_i32; i++) {
        uint64_t nr = (i == 0) ? h_ps_o_orderdate[0] : h_ps_o_orderdate[i] - h_ps_o_orderdate[i - 1];
        o_max_rows_per_page = std::max(o_max_rows_per_page, nr);
    }
    uint64_t l_max_rows_per_page = 0;
    for (uint64_t i = 0; i < l_npages_i32; i++) {
        uint64_t nr = (i == 0) ? h_ps_l_extprice[0] : h_ps_l_extprice[i] - h_ps_l_extprice[i - 1];
        l_max_rows_per_page = std::max(l_max_rows_per_page, nr);
    }
    uint64_t max_rows_per_page = std::max(o_max_rows_per_page, l_max_rows_per_page);

    // Compute max rows per INT64 page (for scratch_i64 sizing)
    uint64_t o_max_rows_per_i64_page = 0;
    for (uint64_t i = 0; i < o_npages_i64; i++) {
        uint64_t nr = (i == 0) ? h_ps_o_orderkey[0] : h_ps_o_orderkey[i] - h_ps_o_orderkey[i - 1];
        o_max_rows_per_i64_page = std::max(o_max_rows_per_i64_page, nr);
    }
    uint64_t l_max_rows_per_i64_page = 0;
    for (uint64_t i = 0; i < l_npages_i64; i++) {
        uint64_t nr = (i == 0) ? h_ps_l_orderkey[0] : h_ps_l_orderkey[i] - h_ps_l_orderkey[i - 1];
        l_max_rows_per_i64_page = std::max(l_max_rows_per_i64_page, nr);
    }
    uint64_t max_rows_per_i64_page = std::max(o_max_rows_per_i64_page, l_max_rows_per_i64_page);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks_pfor = static_cast<uint32_t>(sm_count);

    // ORDERS fused: 5 slots/block (1 INT32 + 2 INT64 × 2 pages)
    const uint32_t num_blocks_orders = std::min(
        static_cast<uint32_t>(sm_count),
        static_cast<uint32_t>(o_npages_i32));
    // LINEITEM fused: 6 slots/block (2 INT32 + 2 INT64 × 2 pages)
    const uint32_t num_blocks_lineitem = std::min(
        static_cast<uint32_t>(sm_count),
        static_cast<uint32_t>(l_npages_i32));
    // Page cache: max(orders_slots, lineitem_slots)
    const uint32_t fused_pc_pages = std::max(num_blocks_orders * 7,
                                              num_blocks_lineitem * 8);

    std::cout << "[Q5] Fused kernel: ORDERS " << num_blocks_orders << " blocks, "
              << o_npages_i32 << " pages; LINEITEM " << num_blocks_lineitem << " blocks, "
              << l_npages_i32 << " pages; max_rows_per_page=" << max_rows_per_page
              << " max_rows_per_i64_page=" << max_rows_per_i64_page << std::endl;

    // ================================================================
    // GPU memory allocation (before total_start)
    // ================================================================

    size_t gpu_free_before = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free_before, &gpu_total);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // GPU-side zone map setup (allocation + metadata outside timing)
    bam_pfor32_io_ctx_t pfor_ctx_zm = nullptr;
    BamZonemapCtx zm_ord_ctx{}, zm_li_ctx{};
    uint32_t zm_ord_nreads = 0, zm_ord_npreds = 0;
    uint32_t zm_li_nreads = 0, zm_li_npreds = 0;
    bool zm_ord_valid = false, zm_li_valid = false;

    if (zonemap_enabled) {
        pfor_ctx_zm = bam_pfor32_io_create(ctrl, static_cast<uint32_t>(page_size), kBamZonemapMaxReads);

        // ORDERS zone map: O_ORDERDATE + R_NAME sideways
        zm_ord_ctx = bam_zonemap_ctx_create(
            bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
            bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
            (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
            static_cast<uint32_t>(page_size), static_cast<uint32_t>(o_orderdate_npages));

        {
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
        zm_li_ctx = bam_zonemap_ctx_create(
            bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
            bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
            (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
            static_cast<uint32_t>(page_size), static_cast<uint32_t>(l_extprice_npages));

        {
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

    // ── Helper: allocate GPU metadata arrays for a compressed field ──
    struct GpuFieldMeta {
        uint64_t* d_prefix_sum;
        uint32_t* d_comp_sizes;
        uint64_t* d_comp_offsets;
    };
    auto alloc_gpu_field_meta = [](uint64_t npages, uint16_t comp) -> GpuFieldMeta {
        GpuFieldMeta m{};
        cudaMalloc(&m.d_prefix_sum, npages * sizeof(uint64_t));
        if (comp != 0) {
            cudaMalloc(&m.d_comp_sizes, npages * sizeof(uint32_t));
            cudaMalloc(&m.d_comp_offsets, npages * sizeof(uint64_t));
        }
        return m;
    };

    // GPU metadata for each field
    auto gm_s_suppkey   = alloc_gpu_field_meta(s_suppkey_npages, s_suppkey_comp);
    auto gm_s_nationkey = alloc_gpu_field_meta(s_nationkey_npages, s_nationkey_comp);
    auto gm_c_custkey   = alloc_gpu_field_meta(c_custkey_npages, c_custkey_comp);
    auto gm_c_nationkey = alloc_gpu_field_meta(c_nationkey_npages, c_nationkey_comp);
    // Flat output arrays (SUPPLIER/CUSTOMER KEY only — ORDERS/LINEITEM use fused BaM)
    uint64_t* d_s_suppkey_flat = nullptr;
    uint64_t* d_c_custkey_flat = nullptr;
    cudaMalloc(&d_s_suppkey_flat, nrecs_supplier * sizeof(uint64_t));
    cudaMalloc(&d_c_custkey_flat, nrecs_customer * sizeof(uint64_t));

    // Temp INT32 buffer for PFOR32 NK IO+decomp+HT build fused kernel
    uint64_t temp_nrows = std::max(nrecs_supplier, nrecs_customer);
    int32_t* d_temp_i32 = nullptr;
    cudaMalloc(&d_temp_i32, temp_nrows * sizeof(int32_t));

    // Full-column GPU metadata for fused INT32 fields
    // O_ORDERDATE
    uint64_t* d_ps_o_orderdate = nullptr;
    uint32_t* d_cs_o_orderdate = nullptr;
    uint64_t* d_co_o_orderdate = nullptr;
    cudaMalloc(&d_ps_o_orderdate, o_orderdate_npages * sizeof(uint64_t));
    cudaMemcpy(d_ps_o_orderdate, h_ps_o_orderdate.data(),
               o_orderdate_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (o_orderdate_comp != 0) {
        cudaMalloc(&d_cs_o_orderdate, o_orderdate_npages * sizeof(uint32_t));
        cudaMemcpy(d_cs_o_orderdate, h_cs_o_orderdate.data(),
                   o_orderdate_npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_o_orderdate, o_orderdate_npages * sizeof(uint64_t));
        cudaMemcpy(d_co_o_orderdate, h_co_o_orderdate.data(),
                   o_orderdate_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    // L_EXTENDEDPRICE (prefix sum shared with L_DISCOUNT)
    uint64_t* d_ps_l_extprice = nullptr;
    uint32_t* d_cs_l_extprice = nullptr;
    uint64_t* d_co_l_extprice = nullptr;
    cudaMalloc(&d_ps_l_extprice, l_extprice_npages * sizeof(uint64_t));
    cudaMemcpy(d_ps_l_extprice, h_ps_l_extprice.data(),
               l_extprice_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (l_extprice_comp != 0) {
        cudaMalloc(&d_cs_l_extprice, l_extprice_npages * sizeof(uint32_t));
        cudaMemcpy(d_cs_l_extprice, h_cs_l_extprice.data(),
                   l_extprice_npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_l_extprice, l_extprice_npages * sizeof(uint64_t));
        cudaMemcpy(d_co_l_extprice, h_co_l_extprice.data(),
                   l_extprice_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    // L_DISCOUNT
    uint32_t* d_cs_l_discount = nullptr;
    uint64_t* d_co_l_discount = nullptr;
    if (l_discount_comp != 0) {
        cudaMalloc(&d_cs_l_discount, l_discount_npages * sizeof(uint32_t));
        cudaMemcpy(d_cs_l_discount, h_cs_l_discount.data(),
                   l_discount_npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_l_discount, l_discount_npages * sizeof(uint64_t));
        cudaMemcpy(d_co_l_discount, h_co_l_discount.data(),
                   l_discount_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // ORDERS INT64 GPU metadata: O_ORDERKEY, O_CUSTKEY
    uint64_t* d_ps_o_i64 = nullptr;
    uint32_t* d_cs_o_orderkey = nullptr;
    uint64_t* d_co_o_orderkey = nullptr;
    uint32_t* d_cs_o_custkey = nullptr;
    uint64_t* d_co_o_custkey = nullptr;
    cudaMalloc(&d_ps_o_i64, o_npages_i64 * sizeof(uint64_t));
    cudaMemcpy(d_ps_o_i64, h_ps_o_orderkey.data(),
               o_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (o_orderkey_comp != 0) {
        cudaMalloc(&d_cs_o_orderkey, o_npages_i64 * sizeof(uint32_t));
        cudaMemcpy(d_cs_o_orderkey, h_cs_o_orderkey.data(),
                   o_npages_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_o_orderkey, o_npages_i64 * sizeof(uint64_t));
        cudaMemcpy(d_co_o_orderkey, h_co_o_orderkey.data(),
                   o_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    if (o_custkey_comp != 0) {
        cudaMalloc(&d_cs_o_custkey, o_npages_i64 * sizeof(uint32_t));
        cudaMemcpy(d_cs_o_custkey, h_cs_o_custkey.data(),
                   o_npages_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_o_custkey, o_npages_i64 * sizeof(uint64_t));
        cudaMemcpy(d_co_o_custkey, h_co_o_custkey.data(),
                   o_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // LINEITEM INT64 GPU metadata: L_ORDERKEY, L_SUPPKEY
    uint64_t* d_ps_l_i64 = nullptr;
    uint32_t* d_cs_l_orderkey = nullptr;
    uint64_t* d_co_l_orderkey = nullptr;
    uint32_t* d_cs_l_suppkey = nullptr;
    uint64_t* d_co_l_suppkey = nullptr;
    cudaMalloc(&d_ps_l_i64, l_npages_i64 * sizeof(uint64_t));
    cudaMemcpy(d_ps_l_i64, h_ps_l_orderkey.data(),
               l_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (l_orderkey_comp != 0) {
        cudaMalloc(&d_cs_l_orderkey, l_npages_i64 * sizeof(uint32_t));
        cudaMemcpy(d_cs_l_orderkey, h_cs_l_orderkey.data(),
                   l_npages_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_l_orderkey, l_npages_i64 * sizeof(uint64_t));
        cudaMemcpy(d_co_l_orderkey, h_co_l_orderkey.data(),
                   l_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    if (l_suppkey_comp != 0) {
        cudaMalloc(&d_cs_l_suppkey, l_npages_i64 * sizeof(uint32_t));
        cudaMemcpy(d_cs_l_suppkey, h_cs_l_suppkey.data(),
                   l_npages_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_l_suppkey, l_npages_i64 * sizeof(uint64_t));
        cudaMemcpy(d_co_l_suppkey, h_co_l_suppkey.data(),
                   l_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // Per-block INT32 scratch buffer for fused kernels
    // ORDERS: num_blocks_orders * 1 * max_rows_per_page
    // LINEITEM: num_blocks_lineitem * 2 * max_rows_per_page
    uint64_t scratch_elems = std::max(
        (uint64_t)num_blocks_orders * max_rows_per_page,
        (uint64_t)num_blocks_lineitem * 2 * max_rows_per_page);
    int32_t* d_scratch = nullptr;
    cudaMalloc(&d_scratch, scratch_elems * sizeof(int32_t));

    // Per-block INT64 scratch buffer
    // ORDERS: num_blocks * 2 fields * 3 pages * max_rows_per_i64_page
    // LINEITEM: num_blocks * 2 fields * 3 pages * max_rows_per_i64_page
    uint64_t scratch_i64_elems = std::max(
        (uint64_t)num_blocks_orders * 2 * 3 * max_rows_per_i64_page,
        (uint64_t)num_blocks_lineitem * 2 * 3 * max_rows_per_i64_page);
    int64_t* d_scratch_i64 = nullptr;
    cudaMalloc(&d_scratch_i64, scratch_i64_elems * sizeof(int64_t));

    // Hash tables
    uint32_t ht_supp_cap = 1;
    while (ht_supp_cap < nrecs_supplier) ht_supp_cap <<= 1;
    uint32_t ht_supp_mask = ht_supp_cap - 1;
    uint64_t* d_ht_supp_keys = nullptr;
    int32_t*  d_ht_supp_values = nullptr;
    cudaMalloc(&d_ht_supp_keys, ht_supp_cap * sizeof(uint64_t));
    cudaMalloc(&d_ht_supp_values, ht_supp_cap * sizeof(int32_t));

    uint32_t ht_cust_cap = 1;
    while (ht_cust_cap < nrecs_customer) ht_cust_cap <<= 1;
    uint32_t ht_cust_mask = ht_cust_cap - 1;
    uint64_t* d_ht_cust_keys = nullptr;
    int32_t*  d_ht_cust_values = nullptr;
    cudaMalloc(&d_ht_cust_keys, ht_cust_cap * sizeof(uint64_t));
    cudaMalloc(&d_ht_cust_values, ht_cust_cap * sizeof(int32_t));

    uint64_t est_ord_qual = std::max((uint64_t)1024, nrecs_orders / 15);
    uint32_t ht_ord_cap = 1;
    while (ht_ord_cap < est_ord_qual * 2) ht_ord_cap <<= 1;
    uint32_t ht_ord_mask = ht_ord_cap - 1;
    uint64_t* d_ht_ord_keys = nullptr;
    int32_t*  d_ht_ord_values = nullptr;
    cudaMalloc(&d_ht_ord_keys, ht_ord_cap * sizeof(uint64_t));
    cudaMalloc(&d_ht_ord_values, ht_ord_cap * sizeof(int32_t));

    // Revenue + nationkey_to_idx
    int64_t* d_revenue = nullptr;
    cudaMalloc(&d_revenue, 25 * sizeof(int64_t));
    int8_t* d_nationkey_to_idx = nullptr;
    cudaMalloc(&d_nationkey_to_idx, 25);
    int32_t* d_asia_regionkey = nullptr;
    cudaMalloc(&d_asia_regionkey, sizeof(int32_t));

    // Host buffers for REGION/NATION
    char nation_names[25][26];
    std::vector<char> h_r_rkey(r_rkey_npages * page_size);
    std::vector<char> h_r_name(r_name_npages * page_size);
    std::vector<char> h_n_nkey(n_nkey_npages * page_size);
    std::vector<char> h_n_name(n_name_npages * page_size);
    std::vector<char> h_n_rkey(n_rkey_npages * page_size);

    // BaM IO contexts
    bam_pfor64_io_ctx_t pfor64_ctx = bam_pfor64_io_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks_pfor);
    bam_pfor32_io_ctx_t pfor32_ctx = bam_pfor32_io_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks_pfor);
    bam_pfor32_io_ctx_t fused_ctx = bam_pfor32_io_create(
        ctrl, static_cast<uint32_t>(page_size), fused_pc_pages);

    std::cout << "[Q5] sm_count=" << sm_count
              << " num_blocks_pfor=" << num_blocks_pfor << std::endl;

    // ── Helper: upload GPU metadata for a field ──
    auto upload_field_meta = [](const GpuFieldMeta& gm,
        const std::vector<uint64_t>& h_ps,
        const std::vector<uint32_t>& h_cs,
        const std::vector<uint64_t>& h_co,
        uint64_t npages, uint16_t comp)
    {
        cudaMemcpy(gm.d_prefix_sum, h_ps.data(),
                   npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (comp != 0) {
            cudaMemcpy(gm.d_comp_sizes, h_cs.data(),
                       npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(gm.d_comp_offsets, h_co.data(),
                       npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
    };

    // ── Helper: setup BAMPfor64FlattenParams ──
    auto make_pfor64_params = [&](uint64_t field_start, uint64_t npages,
        uint16_t comp, uint64_t nrows, const GpuFieldMeta& gm) -> BAMPfor64FlattenParams
    {
        BAMPfor64FlattenParams p{};
        p.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            p.partition_start_lbas[d] = ds.partition_start_lbas[d];
        p.n_devices = n_devices;
        p.page_size = static_cast<uint32_t>(page_size);
        p.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        p.comp_method = comp;
        p.field_start_page_id = field_start;
        p.npages = npages;
        p.nrows = nrows;
        p.num_blocks = num_blocks_pfor;
        p.d_prefix_sum = gm.d_prefix_sum;
        p.d_comp_sizes = gm.d_comp_sizes;
        p.d_comp_offsets = gm.d_comp_offsets;
        return p;
    };

    // ── Helper: PFOR64 flatten (flatten + sync; metadata must be pre-uploaded) ──
    auto flatten_pfor64 = [&](uint64_t field_start, uint64_t npages, uint16_t comp,
        uint64_t nrows, const GpuFieldMeta& gm,
        uint64_t* d_output)
    {
        bam_pfor64_io_flatten_async(pfor64_ctx,
            make_pfor64_params(field_start, npages, comp, nrows, gm),
            reinterpret_cast<int64_t*>(d_output), stream);
        cudaStreamSynchronize(stream);
    };

    int64_t h_revenue[25] = {};
    int num_asia_nations = 0;

    // IO accounting (SUPPLIER/CUSTOMER/REGION/NATION counted here; ORDERS/LINEITEM after zonemap eval)
    uint64_t q5_nios = 0;
    uint64_t q5_read_bytes = 0;
    auto add_col_io_q5 = [&](uint64_t npages, uint16_t comp,
                              const std::vector<uint32_t>& h_cs,
                              const std::vector<bool>& mask) {
        for (uint64_t pg = 0; pg < npages; pg++) {
            if (!mask.empty() && !mask[pg]) continue;
            q5_nios++;
            q5_read_bytes += (comp != 0 && pg < h_cs.size()) ? h_cs[pg] : page_size;
        }
    };
    static const std::vector<bool> no_mask;
    // SUPPLIER (no mask)
    add_col_io_q5(s_suppkey_npages, s_suppkey_comp, h_cs_s_suppkey, no_mask);
    add_col_io_q5(s_nationkey_npages, s_nationkey_comp, h_cs_s_nationkey, no_mask);
    // CUSTOMER (no mask)
    add_col_io_q5(c_custkey_npages, c_custkey_comp, h_cs_c_custkey, no_mask);
    add_col_io_q5(c_nationkey_npages, c_nationkey_comp, h_cs_c_nationkey, no_mask);
    // REGION + NATION (uncompressed host-side reads)
    {
        uint64_t rn_pages = r_rkey_npages + r_name_npages
                          + n_nkey_npages + n_name_npages + n_rkey_npages;
        q5_nios += rn_pages;
        q5_read_bytes += rn_pages * page_size;
    }

    size_t gpu_free_after = 0;
    // Upload Phase 1-2 field metadata before measurement
    upload_field_meta(gm_s_suppkey, h_ps_s_suppkey, h_cs_s_suppkey, h_co_s_suppkey,
                      s_suppkey_npages, s_suppkey_comp);
    upload_field_meta(gm_s_nationkey, h_ps_s_nationkey, h_cs_s_nationkey, h_co_s_nationkey,
                      s_nationkey_npages, s_nationkey_comp);
    upload_field_meta(gm_c_custkey, h_ps_c_custkey, h_cs_c_custkey, h_co_c_custkey,
                      c_custkey_npages, c_custkey_comp);
    upload_field_meta(gm_c_nationkey, h_ps_c_nationkey, h_cs_c_nationkey, h_co_c_nationkey,
                      c_nationkey_npages, c_nationkey_comp);

    // Phase 0 staging (allocated outside measurement interval, Rule 4)
    size_t p0_staging_bytes = (r_rkey_npages + r_name_npages + n_nkey_npages + n_rkey_npages) * page_size;
    char* d_p0_staging = nullptr;
    CUDA_CHECK(cudaMalloc(&d_p0_staging, p0_staging_bytes));

    // Zone map INT64 mask buffers (allocated outside timing)
    uint8_t* d_mask_ord_i64 = nullptr;
    uint8_t* d_mask_li_i64 = nullptr;
    if (zonemap_enabled) {
        cudaMalloc(&d_mask_ord_i64, o_npages_i64);
        cudaMalloc(&d_mask_li_i64, l_npages_i64);
    }

    cudaMemGetInfo(&gpu_free_after, &gpu_total);
    uint64_t q5_gpu_mem_bytes = gpu_free_before - gpu_free_after;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_ord_valid || zm_li_valid) {
        bam_pre_io(zm_ord_ctx.d_ctrls, zm_ord_ctx.d_pc, stream);
    }

    // ════════════════════════════════════════════════════════
    auto total_start = std::chrono::high_resolution_clock::now();
    s_kernel_launches = 0;
    // ════════════════════════════════════════════════════════

    // GPU-side zone map eval (IO + eval + INT64 mask derivation, all GPU-side)
    if (zm_ord_valid) {
        zm_ord_ctx.d_ps_i32   = d_ps_o_orderdate;
        zm_ord_ctx.d_ps_i64   = d_ps_o_i64;
        zm_ord_ctx.d_mask_i64 = d_mask_ord_i64;
        zm_ord_ctx.npages_i64 = static_cast<uint32_t>(o_npages_i64);
        bam_zonemap_eval_async(zm_ord_ctx, static_cast<uint32_t>(o_orderdate_npages),
                               zm_ord_nreads, zm_ord_npreds, stream);
        cudaStreamSynchronize(stream);
        s_kernel_launches++;
    }
    if (zm_li_valid) {
        zm_li_ctx.d_ps_i32   = d_ps_l_extprice;
        zm_li_ctx.d_ps_i64   = d_ps_l_i64;
        zm_li_ctx.d_mask_i64 = d_mask_li_i64;
        zm_li_ctx.npages_i64 = static_cast<uint32_t>(l_npages_i64);
        bam_zonemap_eval_async(zm_li_ctx, static_cast<uint32_t>(l_extprice_npages),
                               zm_li_nreads, zm_li_npreds, stream);
        cudaStreamSynchronize(stream);
        s_kernel_launches++;
    }

    // ORDERS/LINEITEM IO accounting (uses h_mask from zonemap eval)
    {
        // Read back INT64 masks from GPU for accounting
        std::vector<uint8_t> h_ord_i64(o_npages_i64, 0);
        std::vector<uint8_t> h_li_i64(l_npages_i64, 0);
        if (zm_ord_valid)
            cudaMemcpy(h_ord_i64.data(), d_mask_ord_i64, o_npages_i64, cudaMemcpyDeviceToHost);
        if (zm_li_valid)
            cudaMemcpy(h_li_i64.data(), d_mask_li_i64, l_npages_i64, cudaMemcpyDeviceToHost);

        // ORDERS INT32: O_ORDERDATE
        std::vector<uint32_t> active_odate_pages;
        std::vector<bool> mask_odate_pages(o_orderdate_npages, true);
        for (uint32_t pg = 0; pg < o_orderdate_npages; pg++) {
            if (zm_ord_valid && !zm_ord_ctx.h_mask[pg]) {
                mask_odate_pages[pg] = false;
                continue;
            }
            active_odate_pages.push_back(pg);
        }
        add_col_io_q5(o_orderdate_npages, o_orderdate_comp, h_cs_o_orderdate, mask_odate_pages);

        // ORDERS INT64: GPU-derived mask
        std::vector<bool> mask_ord_i64_pages(o_npages_i64, true);
        if (zm_ord_valid) {
            for (uint64_t pg = 0; pg < o_npages_i64; pg++)
                mask_ord_i64_pages[pg] = h_ord_i64[pg] != 0;
        }
        add_col_io_q5(o_orderkey_npages, o_orderkey_comp, h_cs_o_orderkey, mask_ord_i64_pages);
        add_col_io_q5(o_custkey_npages, o_custkey_comp, h_cs_o_custkey, mask_ord_i64_pages);

        // LINEITEM INT32: L_EXTPRICE, L_DISCOUNT
        std::vector<uint32_t> active_li_int32_pages;
        std::vector<bool> mask_li_int32_pages(l_extprice_npages, true);
        for (uint32_t pg = 0; pg < l_extprice_npages; pg++) {
            if (zm_li_valid && !zm_li_ctx.h_mask[pg]) {
                mask_li_int32_pages[pg] = false;
                continue;
            }
            active_li_int32_pages.push_back(pg);
        }
        add_col_io_q5(l_extprice_npages, l_extprice_comp, h_cs_l_extprice, mask_li_int32_pages);
        add_col_io_q5(l_discount_npages, l_discount_comp, h_cs_l_discount, mask_li_int32_pages);

        // LINEITEM INT64: GPU-derived mask
        std::vector<bool> mask_li_i64_pages(l_npages_i64, true);
        if (zm_li_valid) {
            for (uint64_t pg = 0; pg < l_npages_i64; pg++)
                mask_li_i64_pages[pg] = h_li_i64[pg] != 0;
        }
        add_col_io_q5(l_orderkey_npages, l_orderkey_comp, h_cs_l_orderkey, mask_li_i64_pages);
        add_col_io_q5(l_suppkey_npages, l_suppkey_comp, h_cs_l_suppkey, mask_li_i64_pages);

        if (zm_ord_valid) {
            std::cout << "[Q5-ZONEMAP] ORDERS active=" << active_odate_pages.size()
                      << "/" << o_orderdate_npages << std::endl;
        }
        if (zm_li_valid) {
            std::cout << "[Q5-ZONEMAP] LINEITEM active=" << active_li_int32_pages.size()
                      << "/" << l_extprice_npages << std::endl;
        }
    }

    // Phase 0: REGION + NATION → d_nationkey_to_idx (GPU kernel)
    memset(nation_names, 0, sizeof(nation_names));
    num_asia_nations = 0;
    int32_t asia_regionkey = -1;
    {
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
        char* d_staging = d_p0_staging;
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
        uint32_t nalloc_nation = bam_host_pag_get_nalloc(h_n_nkey.data());
        for (uint32_t i = 0; i < nalloc_nation; i++) {
            int32_t n_rkey = *reinterpret_cast<const int32_t *>(
                h_n_rkey.data() + 12 + sizeof(int32_t) * i);
            if (n_rkey != asia_regionkey) continue;
            int32_t n_nkey = *reinterpret_cast<const int32_t *>(
                h_n_nkey.data() + 12 + sizeof(int32_t) * i);
            const char *n_name = bam_host_pagcol_char_data(h_n_name.data(), i, 28);
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

    // Phase 1: SUPPLIER hash table build (KEY flatten → NK IO+decomp+HT build fused)
    std::cout << "[Q5] Loading S_SUPPKEY (" << s_suppkey_npages << " pages)..." << std::endl;
    flatten_pfor64(s_suppkey_start, s_suppkey_npages, s_suppkey_comp,
                   nrecs_supplier, gm_s_suppkey,
                   d_s_suppkey_flat);
    s_kernel_launches++;

    std::cout << "[Q5] Loading S_NATIONKEY (" << s_nationkey_npages << " pages) + HT build..." << std::endl;
    cudaMemsetAsync(d_ht_supp_keys, 0xFF, ht_supp_cap * sizeof(uint64_t), stream);
    {
        BAMPfor32FlattenParams p32{};
        p32.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            p32.partition_start_lbas[d] = ds.partition_start_lbas[d];
        p32.n_devices = n_devices;
        p32.page_size = static_cast<uint32_t>(page_size);
        p32.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        p32.comp_method = s_nationkey_comp;
        p32.field_start_page_id = s_nationkey_start;
        p32.npages = s_nationkey_npages;
        p32.nrows = nrecs_supplier;
        p32.num_blocks = num_blocks_pfor;
        p32.d_prefix_sum = gm_s_nationkey.d_prefix_sum;
        p32.d_comp_sizes = gm_s_nationkey.d_comp_sizes;
        p32.d_comp_offsets = gm_s_nationkey.d_comp_offsets;
        bam_pfor32_io_nk_ht_build_async(pfor32_ctx, p32, d_temp_i32,
            d_s_suppkey_flat, d_nationkey_to_idx,
            d_ht_supp_keys, d_ht_supp_values, ht_supp_mask, stream);
    }
    s_kernel_launches++;
    cudaStreamSynchronize(stream);
    std::cout << "[Q5] SUPPLIER HT built (capacity=" << ht_supp_cap << ")" << std::endl;

    // Phase 2: CUSTOMER hash table build (KEY flatten → NK IO+decomp+HT build fused)
    std::cout << "[Q5] Loading C_CUSTKEY (" << c_custkey_npages << " pages)..." << std::endl;
    flatten_pfor64(c_custkey_start, c_custkey_npages, c_custkey_comp,
                   nrecs_customer, gm_c_custkey,
                   d_c_custkey_flat);
    s_kernel_launches++;

    std::cout << "[Q5] Loading C_NATIONKEY (" << c_nationkey_npages << " pages) + HT build..." << std::endl;
    cudaMemsetAsync(d_ht_cust_keys, 0xFF, ht_cust_cap * sizeof(uint64_t), stream);
    {
        BAMPfor32FlattenParams p32{};
        p32.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            p32.partition_start_lbas[d] = ds.partition_start_lbas[d];
        p32.n_devices = n_devices;
        p32.page_size = static_cast<uint32_t>(page_size);
        p32.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        p32.comp_method = c_nationkey_comp;
        p32.field_start_page_id = c_nationkey_start;
        p32.npages = c_nationkey_npages;
        p32.nrows = nrecs_customer;
        p32.num_blocks = num_blocks_pfor;
        p32.d_prefix_sum = gm_c_nationkey.d_prefix_sum;
        p32.d_comp_sizes = gm_c_nationkey.d_comp_sizes;
        p32.d_comp_offsets = gm_c_nationkey.d_comp_offsets;
        bam_pfor32_io_nk_ht_build_async(pfor32_ctx, p32, d_temp_i32,
            d_c_custkey_flat, d_nationkey_to_idx,
            d_ht_cust_keys, d_ht_cust_values, ht_cust_mask, stream);
    }
    s_kernel_launches++;
    cudaStreamSynchronize(stream);
    std::cout << "[Q5] CUSTOMER HT built (capacity=" << ht_cust_cap << ")" << std::endl;

    // Phase 3: ORDERS fused kernel (INT32 + INT64 via BaM, no flatten)
    cudaMemsetAsync(d_ht_ord_keys, 0xFF, ht_ord_cap * sizeof(uint64_t), stream);
    {
        BAMQ5OrdersFusedParams op{};
        op.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            op.partition_start_lbas[d] = ds.partition_start_lbas[d];
        op.n_devices = n_devices;
        op.page_size = static_cast<uint32_t>(page_size);
        op.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        op.npages = o_npages_i32;
        op.num_blocks = num_blocks_orders;
        op.date_low = date_low;
        op.date_high = date_high;
        // INT32: O_ORDERDATE
        op.field_start_page_id = o_orderdate_start;
        op.comp_method = o_orderdate_comp;
        op.d_prefix_sum = d_ps_o_orderdate;
        op.d_comp_sizes = d_cs_o_orderdate;
        op.d_comp_offsets = d_co_o_orderdate;
        op.d_page_active = zm_ord_valid ? zm_ord_ctx.d_mask : nullptr;
        op.d_active_page_ids = zm_ord_valid ? zm_ord_ctx.d_active_ids : nullptr;
        op.num_active_pages = zm_ord_valid ? *zm_ord_ctx.h_num_active : 0;
        // INT64: O_ORDERKEY, O_CUSTKEY
        op.field_start_page_ids_i64[0] = o_orderkey_start;
        op.comp_methods_i64[0] = o_orderkey_comp;
        op.d_comp_sizes_i64[0] = d_cs_o_orderkey;
        op.d_comp_offsets_i64[0] = d_co_o_orderkey;
        op.field_start_page_ids_i64[1] = o_custkey_start;
        op.comp_methods_i64[1] = o_custkey_comp;
        op.d_comp_sizes_i64[1] = d_cs_o_custkey;
        op.d_comp_offsets_i64[1] = d_co_o_custkey;
        op.d_prefix_sum_i64 = d_ps_o_i64;
        op.npages_i64 = o_npages_i64;
        // CUSTOMER HT
        op.d_ht_cust_keys = d_ht_cust_keys;
        op.d_ht_cust_values = d_ht_cust_values;
        op.ht_cust_mask = ht_cust_mask;
        op.d_ht_ord_keys = d_ht_ord_keys;
        op.d_ht_ord_values = d_ht_ord_values;
        op.ht_ord_mask = ht_ord_mask;
        op.d_scratch = d_scratch;
        op.d_scratch_i64 = d_scratch_i64;
        op.scratch_stride = static_cast<uint32_t>(max_rows_per_page);
        op.scratch_stride_i64 = static_cast<uint32_t>(max_rows_per_i64_page);
        op.d_dbg_counters = nullptr;
        bam_q5_orders_fused_run(fused_ctx, op, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }
    std::cout << "[Q5] ORDERS HT built (capacity=" << ht_ord_cap << ")" << std::endl;

    // Phase 4: LINEITEM fused kernel (all fields via BaM, no flatten)
    cudaMemsetAsync(d_revenue, 0, 25 * sizeof(int64_t), stream);
    {
        BAMQ5LineitemFusedParams lp{};
        lp.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            lp.partition_start_lbas[d] = ds.partition_start_lbas[d];
        lp.n_devices = n_devices;
        lp.page_size = static_cast<uint32_t>(page_size);
        lp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        lp.npages = l_npages_i32;
        lp.num_blocks = num_blocks_lineitem;
        // INT32: L_EXTPRICE, L_DISCOUNT
        lp.field_start_page_ids[0] = l_extprice_start;
        lp.field_start_page_ids[1] = l_discount_start;
        lp.comp_methods[0] = l_extprice_comp;
        lp.comp_methods[1] = l_discount_comp;
        lp.d_prefix_sum = d_ps_l_extprice;
        lp.d_comp_sizes[0] = d_cs_l_extprice;
        lp.d_comp_sizes[1] = d_cs_l_discount;
        lp.d_comp_offsets[0] = d_co_l_extprice;
        lp.d_comp_offsets[1] = d_co_l_discount;
        lp.d_page_active = zm_li_valid ? zm_li_ctx.d_mask : nullptr;
        lp.d_active_page_ids = zm_li_valid ? zm_li_ctx.d_active_ids : nullptr;
        lp.num_active_pages = zm_li_valid ? *zm_li_ctx.h_num_active : 0;
        // INT64: L_ORDERKEY, L_SUPPKEY
        lp.field_start_page_ids_i64[0] = l_orderkey_start;
        lp.comp_methods_i64[0] = l_orderkey_comp;
        lp.d_comp_sizes_i64[0] = d_cs_l_orderkey;
        lp.d_comp_offsets_i64[0] = d_co_l_orderkey;
        lp.field_start_page_ids_i64[1] = l_suppkey_start;
        lp.comp_methods_i64[1] = l_suppkey_comp;
        lp.d_comp_sizes_i64[1] = d_cs_l_suppkey;
        lp.d_comp_offsets_i64[1] = d_co_l_suppkey;
        lp.d_prefix_sum_i64 = d_ps_l_i64;
        lp.npages_i64 = l_npages_i64;
        // ORDERS HT
        lp.d_ht_ord_keys = d_ht_ord_keys;
        lp.d_ht_ord_values = d_ht_ord_values;
        lp.ht_ord_mask = ht_ord_mask;
        // SUPPLIER HT
        lp.d_ht_supp_keys = d_ht_supp_keys;
        lp.d_ht_supp_values = d_ht_supp_values;
        lp.ht_supp_mask = ht_supp_mask;
        // Scratch
        lp.d_scratch = d_scratch;
        lp.d_scratch_i64 = d_scratch_i64;
        lp.scratch_stride = static_cast<uint32_t>(max_rows_per_page);
        lp.scratch_stride_i64 = static_cast<uint32_t>(max_rows_per_i64_page);
        lp.d_dbg_counters = nullptr;
        bam_q5_lineitem_fused_run(fused_ctx, lp, d_revenue, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }
    // Phase 5: Results
    memset(h_revenue, 0, sizeof(h_revenue));
    cudaMemcpy(h_revenue, d_revenue, 25 * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // ════════════════════════════════════════════════════════
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
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

    // ── Cleanup ──
    auto free_gm = [](const GpuFieldMeta& gm, uint16_t comp) {
        cudaFree(gm.d_prefix_sum);
        if (comp != 0) {
            cudaFree(gm.d_comp_sizes);
            cudaFree(gm.d_comp_offsets);
        }
    };
    free_gm(gm_s_suppkey, s_suppkey_comp);
    free_gm(gm_s_nationkey, s_nationkey_comp);
    free_gm(gm_c_custkey, c_custkey_comp);
    free_gm(gm_c_nationkey, c_nationkey_comp);

    cudaFree(d_p0_staging);
    cudaFree(d_s_suppkey_flat);
    cudaFree(d_c_custkey_flat);
    cudaFree(d_temp_i32);
    cudaFree(d_ht_supp_keys);
    cudaFree(d_ht_supp_values);
    cudaFree(d_ht_cust_keys);
    cudaFree(d_ht_cust_values);
    cudaFree(d_ht_ord_keys);
    cudaFree(d_ht_ord_values);
    cudaFree(d_revenue);
    cudaFree(d_nationkey_to_idx);
    if (zm_ord_valid) bam_zonemap_ctx_destroy(zm_ord_ctx);
    if (zm_li_valid) bam_zonemap_ctx_destroy(zm_li_ctx);
    if (pfor_ctx_zm) bam_pfor32_io_destroy(pfor_ctx_zm);
    if (d_mask_ord_i64) cudaFree(d_mask_ord_i64);
    if (d_mask_li_i64) cudaFree(d_mask_li_i64);
    // Fused INT32 metadata
    cudaFree(d_ps_o_orderdate);
    if (d_cs_o_orderdate) cudaFree(d_cs_o_orderdate);
    if (d_co_o_orderdate) cudaFree(d_co_o_orderdate);
    cudaFree(d_ps_l_extprice);
    if (d_cs_l_extprice) cudaFree(d_cs_l_extprice);
    if (d_co_l_extprice) cudaFree(d_co_l_extprice);
    if (d_cs_l_discount) cudaFree(d_cs_l_discount);
    if (d_co_l_discount) cudaFree(d_co_l_discount);
    // Fused INT64 metadata
    cudaFree(d_ps_o_i64);
    if (d_cs_o_orderkey) cudaFree(d_cs_o_orderkey);
    if (d_co_o_orderkey) cudaFree(d_co_o_orderkey);
    if (d_cs_o_custkey) cudaFree(d_cs_o_custkey);
    if (d_co_o_custkey) cudaFree(d_co_o_custkey);
    cudaFree(d_ps_l_i64);
    if (d_cs_l_orderkey) cudaFree(d_cs_l_orderkey);
    if (d_co_l_orderkey) cudaFree(d_co_l_orderkey);
    if (d_cs_l_suppkey) cudaFree(d_cs_l_suppkey);
    if (d_co_l_suppkey) cudaFree(d_co_l_suppkey);
    cudaFree(d_scratch);
    cudaFree(d_scratch_i64);
    bam_pfor64_io_destroy(pfor64_ctx);
    bam_pfor32_io_destroy(pfor32_ctx);
    bam_pfor32_io_destroy(fused_ctx);
    cudaStreamDestroy(stream);
    bam_ctrl_close(ctrl);
    uint64_t q5_total_pages = s_suppkey_npages + s_nationkey_npages
                            + c_custkey_npages + c_nationkey_npages
                            + o_orderkey_npages + o_custkey_npages + o_orderdate_npages
                            + l_orderkey_npages + l_suppkey_npages + l_extprice_npages + l_discount_npages;
    return PigResult{total_ms, q5_nios, q5_read_bytes,
                     collect_comp_methods({s_suppkey_comp, s_nationkey_comp,
                                           c_custkey_comp, c_nationkey_comp,
                                           o_orderkey_comp, o_custkey_comp, o_orderdate_comp,
                                           l_orderkey_comp, l_suppkey_comp, l_extprice_comp, l_discount_comp}),
                     gpu_ctrl_bytes + q5_gpu_mem_bytes, gpu_ctrl_bytes, q5_gpu_mem_bytes,
                     q5_total_pages,
                     s_kernel_launches};
}

// ============================================================
// TPC-H Q16 (PiG) — GPU-initiated I/O via BaM
// ============================================================

PigResult tpch_q16(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "=== TPC-H Q16 (PiG) ===" << std::endl;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    size_t gpu_free_pre_ctrl = 0, gpu_total_pre = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_pre);

    const uint32_t bam_num_queues = 128;
    auto ds = bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;
    const uint32_t n_devices = ds.n_devices;
    const uint64_t partition_start_lba = ds.partition_start_lba;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_pre);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;

    // Helper: read a page given global page ID (handles striping)
    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    read_striped_page(0, page_size, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;

    // ── Field metadata ──
    constexpr size_t col_s_suppkey = TPCH::common::S_SUPPKEY;
    constexpr size_t col_s_comment = TPCH::common::S_COMMENT;
    constexpr size_t col_p_partkey = TPCH::common::P_PARTKEY;
    constexpr size_t col_p_brand   = TPCH::common::P_BRAND;
    constexpr size_t col_p_type    = TPCH::common::P_TYPE;
    constexpr size_t col_p_size    = TPCH::common::P_SIZE;
    constexpr size_t col_ps_partkey = TPCH::common::PS_PARTKEY;
    constexpr size_t col_ps_suppkey = TPCH::common::PS_SUPPKEY;

    // SUPPLIER fields
    uint64_t s_suppkey_start  = metadata.table_supplier_start_page_ids[col_s_suppkey];
    uint64_t s_suppkey_npages = metadata.table_supplier_npages[col_s_suppkey];
    uint16_t s_suppkey_comp   = metadata.table_supplier_compression_method[col_s_suppkey];

    uint64_t s_comment_start  = metadata.table_supplier_start_page_ids[col_s_comment];
    uint64_t s_comment_npages = metadata.table_supplier_npages[col_s_comment];
    uint16_t s_comment_comp   = metadata.table_supplier_compression_method[col_s_comment];

    // PART fields
    uint64_t p_partkey_start  = metadata.table_part_start_page_ids[col_p_partkey];
    uint64_t p_partkey_npages = metadata.table_part_npages[col_p_partkey];
    uint16_t p_partkey_comp   = metadata.table_part_compression_method[col_p_partkey];

    uint64_t p_brand_start  = metadata.table_part_start_page_ids[col_p_brand];
    uint64_t p_brand_npages = metadata.table_part_npages[col_p_brand];
    uint16_t p_brand_comp   = metadata.table_part_compression_method[col_p_brand];

    uint64_t p_type_start  = metadata.table_part_start_page_ids[col_p_type];
    uint64_t p_type_npages = metadata.table_part_npages[col_p_type];
    uint16_t p_type_comp   = metadata.table_part_compression_method[col_p_type];

    uint64_t p_size_start  = metadata.table_part_start_page_ids[col_p_size];
    uint64_t p_size_npages = metadata.table_part_npages[col_p_size];
    uint16_t p_size_comp   = metadata.table_part_compression_method[col_p_size];

    // PARTSUPP fields
    uint64_t ps_partkey_start  = metadata.table_partsupp_start_page_ids[col_ps_partkey];
    uint64_t ps_partkey_npages = metadata.table_partsupp_npages[col_ps_partkey];
    uint16_t ps_partkey_comp   = metadata.table_partsupp_compression_method[col_ps_partkey];

    uint64_t ps_suppkey_start  = metadata.table_partsupp_start_page_ids[col_ps_suppkey];
    uint64_t ps_suppkey_npages = metadata.table_partsupp_npages[col_ps_suppkey];
    uint16_t ps_suppkey_comp   = metadata.table_partsupp_compression_method[col_ps_suppkey];

    uint64_t nrecs_supplier = metadata.table_supplier_nrows;
    uint64_t nrecs_part     = metadata.table_part_nrows;
    uint64_t nrecs_partsupp = metadata.table_partsupp_nrows;

    std::cout << "nrecs_supplier=" << nrecs_supplier
              << " nrecs_part=" << nrecs_part
              << " nrecs_partsupp=" << nrecs_partsupp << std::endl;
    std::cout << "  S_SUPPKEY: start=" << s_suppkey_start << " npages=" << s_suppkey_npages
              << " comp=" << s_suppkey_comp << std::endl;
    std::cout << "  S_COMMENT: start=" << s_comment_start << " npages=" << s_comment_npages
              << " comp=" << s_comment_comp << std::endl;
    std::cout << "  P_PARTKEY: start=" << p_partkey_start << " npages=" << p_partkey_npages
              << " comp=" << p_partkey_comp << std::endl;
    std::cout << "  P_BRAND:   start=" << p_brand_start << " npages=" << p_brand_npages
              << " comp=" << p_brand_comp << std::endl;
    std::cout << "  P_TYPE:    start=" << p_type_start << " npages=" << p_type_npages
              << " comp=" << p_type_comp << std::endl;
    std::cout << "  P_SIZE:    start=" << p_size_start << " npages=" << p_size_npages
              << " comp=" << p_size_comp << std::endl;
    std::cout << "  PS_PARTKEY: start=" << ps_partkey_start << " npages=" << ps_partkey_npages
              << " comp=" << ps_partkey_comp << std::endl;
    std::cout << "  PS_SUPPKEY: start=" << ps_suppkey_start << " npages=" << ps_suppkey_npages
              << " comp=" << ps_suppkey_comp << std::endl;

    // ── FSST detection for fused kernel dispatch ──
    const bool is_fsst_s_comment = (s_comment_comp == static_cast<uint16_t>(CompressionMethod::FSST)
                                  || s_comment_comp == static_cast<uint16_t>(CompressionMethod::FSST_ROWID));
    const bool is_fsst_p_type    = (p_type_comp == static_cast<uint16_t>(CompressionMethod::FSST)
                                  || p_type_comp == static_cast<uint16_t>(CompressionMethod::FSST_ROWID));
    const bool is_fsst_p_brand   = (p_brand_comp == static_cast<uint16_t>(CompressionMethod::FSST)
                                  || p_brand_comp == static_cast<uint16_t>(CompressionMethod::FSST_ROWID));
    const bool any_lz4par_needed = !is_fsst_s_comment || !is_fsst_p_type || !is_fsst_p_brand;
    const bool any_fsst_needed   = is_fsst_s_comment || is_fsst_p_type || is_fsst_p_brand;

    if (any_fsst_needed) {
        std::cout << "[Q16] FSST columns detected:";
        if (is_fsst_s_comment) std::cout << " S_COMMENT";
        if (is_fsst_p_brand) std::cout << " P_BRAND";
        if (is_fsst_p_type) std::cout << " P_TYPE";
        std::cout << std::endl;
    }

    // ================================================================
    // Host-side metadata reads (before total_start)
    // ================================================================

    auto read_prefix_sum_host = [&](uint64_t ps_start, uint64_t ps_npages,
                                     uint64_t field_npages) -> std::vector<uint64_t>
    {
        if (ps_npages == 0) return {};
        std::vector<char> h_buf(ps_npages * page_size);
        for (uint64_t p = 0; p < ps_npages; p++) {
            read_striped_page(ps_start + p, page_size, h_buf.data() + p * page_size);
        }
        uint64_t* ps_raw = reinterpret_cast<uint64_t*>(h_buf.data()) + 1;
        return std::vector<uint64_t>(ps_raw, ps_raw + field_npages);
    };

    auto prepare_comp_metadata = [&](
        uint64_t field_start, uint64_t field_npages, uint16_t comp_method,
        uint64_t cs_start_page, uint64_t cs_npages_cnt,
        uint64_t nbase_val, uint64_t base_start_page)
        -> std::pair<std::vector<uint32_t>, std::vector<uint64_t>>
    {
        if (comp_method == 0) return {{}, {}};
        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            read_striped_page(cs_start_page + p, page_size, sizes_buf.data() + p * page_size);
        }
        std::vector<uint32_t> comp_sizes(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages);
        size_t bp_npages = TPCH::nbase_to_npages(nbase_val, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            read_striped_page(base_start_page + p, page_size, bases_buf.data() + p * page_size);
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            comp_sizes.data(), nbase_val, field_npages, page_size, field_start,
            n_devices, offsets_vec);
        std::vector<uint64_t> comp_offsets(field_npages);
        for (uint64_t i = 0; i < field_npages; i++)
            comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
        return {comp_sizes, comp_offsets};
    };

    std::cout << "[Q16] Reading metadata..." << std::endl;

    // S_SUPPKEY
    auto h_ps_s_suppkey = read_prefix_sum_host(
        metadata.table_supplier_prefix_sum_start_page_ids[col_s_suppkey],
        metadata.table_supplier_prefix_sum_npages[col_s_suppkey],
        s_suppkey_npages);
    auto [h_cs_s_suppkey, h_co_s_suppkey] = prepare_comp_metadata(
        s_suppkey_start, s_suppkey_npages, s_suppkey_comp,
        metadata.table_supplier_compressed_page_sizes_start_page_ids[col_s_suppkey],
        metadata.table_supplier_compressed_page_sizes_npages[col_s_suppkey],
        metadata.table_supplier_compression_nbases[col_s_suppkey],
        metadata.table_supplier_compression_base_start_page_ids[col_s_suppkey]);

    // S_COMMENT
    auto h_ps_s_comment = read_prefix_sum_host(
        metadata.table_supplier_prefix_sum_start_page_ids[col_s_comment],
        metadata.table_supplier_prefix_sum_npages[col_s_comment],
        s_comment_npages);
    auto [h_cs_s_comment, h_co_s_comment] = prepare_comp_metadata(
        s_comment_start, s_comment_npages, s_comment_comp,
        metadata.table_supplier_compressed_page_sizes_start_page_ids[col_s_comment],
        metadata.table_supplier_compressed_page_sizes_npages[col_s_comment],
        metadata.table_supplier_compression_nbases[col_s_comment],
        metadata.table_supplier_compression_base_start_page_ids[col_s_comment]);

    // P_PARTKEY
    auto h_ps_p_partkey = read_prefix_sum_host(
        metadata.table_part_prefix_sum_start_page_ids[col_p_partkey],
        metadata.table_part_prefix_sum_npages[col_p_partkey],
        p_partkey_npages);
    auto [h_cs_p_partkey, h_co_p_partkey] = prepare_comp_metadata(
        p_partkey_start, p_partkey_npages, p_partkey_comp,
        metadata.table_part_compressed_page_sizes_start_page_ids[col_p_partkey],
        metadata.table_part_compressed_page_sizes_npages[col_p_partkey],
        metadata.table_part_compression_nbases[col_p_partkey],
        metadata.table_part_compression_base_start_page_ids[col_p_partkey]);

    // P_BRAND
    auto h_ps_p_brand = read_prefix_sum_host(
        metadata.table_part_prefix_sum_start_page_ids[col_p_brand],
        metadata.table_part_prefix_sum_npages[col_p_brand],
        p_brand_npages);
    auto [h_cs_p_brand, h_co_p_brand] = prepare_comp_metadata(
        p_brand_start, p_brand_npages, p_brand_comp,
        metadata.table_part_compressed_page_sizes_start_page_ids[col_p_brand],
        metadata.table_part_compressed_page_sizes_npages[col_p_brand],
        metadata.table_part_compression_nbases[col_p_brand],
        metadata.table_part_compression_base_start_page_ids[col_p_brand]);

    // P_TYPE
    auto h_ps_p_type = read_prefix_sum_host(
        metadata.table_part_prefix_sum_start_page_ids[col_p_type],
        metadata.table_part_prefix_sum_npages[col_p_type],
        p_type_npages);
    auto [h_cs_p_type, h_co_p_type] = prepare_comp_metadata(
        p_type_start, p_type_npages, p_type_comp,
        metadata.table_part_compressed_page_sizes_start_page_ids[col_p_type],
        metadata.table_part_compressed_page_sizes_npages[col_p_type],
        metadata.table_part_compression_nbases[col_p_type],
        metadata.table_part_compression_base_start_page_ids[col_p_type]);

    // P_SIZE
    auto h_ps_p_size = read_prefix_sum_host(
        metadata.table_part_prefix_sum_start_page_ids[col_p_size],
        metadata.table_part_prefix_sum_npages[col_p_size],
        p_size_npages);
    auto [h_cs_p_size, h_co_p_size] = prepare_comp_metadata(
        p_size_start, p_size_npages, p_size_comp,
        metadata.table_part_compressed_page_sizes_start_page_ids[col_p_size],
        metadata.table_part_compressed_page_sizes_npages[col_p_size],
        metadata.table_part_compression_nbases[col_p_size],
        metadata.table_part_compression_base_start_page_ids[col_p_size]);

    // PS_PARTKEY
    auto h_ps_ps_partkey = read_prefix_sum_host(
        metadata.table_partsupp_prefix_sum_start_page_ids[col_ps_partkey],
        metadata.table_partsupp_prefix_sum_npages[col_ps_partkey],
        ps_partkey_npages);
    auto [h_cs_ps_partkey, h_co_ps_partkey] = prepare_comp_metadata(
        ps_partkey_start, ps_partkey_npages, ps_partkey_comp,
        metadata.table_partsupp_compressed_page_sizes_start_page_ids[col_ps_partkey],
        metadata.table_partsupp_compressed_page_sizes_npages[col_ps_partkey],
        metadata.table_partsupp_compression_nbases[col_ps_partkey],
        metadata.table_partsupp_compression_base_start_page_ids[col_ps_partkey]);

    // PS_SUPPKEY
    auto h_ps_ps_suppkey = read_prefix_sum_host(
        metadata.table_partsupp_prefix_sum_start_page_ids[col_ps_suppkey],
        metadata.table_partsupp_prefix_sum_npages[col_ps_suppkey],
        ps_suppkey_npages);
    auto [h_cs_ps_suppkey, h_co_ps_suppkey] = prepare_comp_metadata(
        ps_suppkey_start, ps_suppkey_npages, ps_suppkey_comp,
        metadata.table_partsupp_compressed_page_sizes_start_page_ids[col_ps_suppkey],
        metadata.table_partsupp_compressed_page_sizes_npages[col_ps_suppkey],
        metadata.table_partsupp_compression_nbases[col_ps_suppkey],
        metadata.table_partsupp_compression_base_start_page_ids[col_ps_suppkey]);

    // ── KMP pattern tables for '%Customer%Complaints%' ──
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

    // ================================================================
    // GPU memory allocation (before total_start)
    // ================================================================

    size_t gpu_free_before = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free_before, &gpu_total);

    // GPU memory tracking helper
    auto gpu_track = [](const char* label, size_t& prev_free) {
        size_t cur_free = 0, cur_total = 0;
        cudaMemGetInfo(&cur_free, &cur_total);
        int64_t delta_mb = static_cast<int64_t>(prev_free - cur_free) / (1024*1024);
        int64_t cum_mb = static_cast<int64_t>(cur_total - cur_free) / (1024*1024);
        std::cerr << "[Q16 GPU] " << label << ": +" << delta_mb << " MiB (total used: " << cum_mb << " MiB)" << std::endl;
        prev_free = cur_free;
    };
    size_t gpu_track_free = gpu_free_before;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
#if 0  // Debug: double-buffered pipeline resources
    cudaStream_t comp_stream;
    cudaStreamCreate(&comp_stream);
#endif

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks_pfor = static_cast<uint32_t>(sm_count);       // PFOR flatten: 1 block/SM (≤ n_qps)
    uint32_t num_blocks      = static_cast<uint32_t>(sm_count) * 4;   // Fused kernels: 4 blocks/SM
#if 0  // Debug: double-buffered batch size
    const uint32_t pages_per_batch = num_blocks * 4;  // 4 blocks/SM for better occupancy
#endif

    // ── Helper lambda: allocate GPU metadata arrays for a compressed field ──
    struct GpuFieldMeta {
        uint64_t* d_prefix_sum;
        uint32_t* d_comp_sizes;
        uint64_t* d_comp_offsets;
    };
    auto alloc_gpu_field_meta = [](uint64_t npages, uint16_t comp) -> GpuFieldMeta {
        GpuFieldMeta m{};
        cudaMalloc(&m.d_prefix_sum, npages * sizeof(uint64_t));
        if (comp != 0) {
            cudaMalloc(&m.d_comp_sizes, npages * sizeof(uint32_t));
            cudaMalloc(&m.d_comp_offsets, npages * sizeof(uint64_t));
        }
        return m;
    };

    // INT64 flat output arrays
    int64_t* d_s_suppkey_i64 = nullptr;
    int64_t* d_p_partkey_i64 = nullptr;
    int64_t* d_ps_partkey_i64 = nullptr;
    int64_t* d_ps_suppkey_i64 = nullptr;
    cudaMalloc(&d_s_suppkey_i64, nrecs_supplier * sizeof(int64_t));
    cudaMalloc(&d_p_partkey_i64, nrecs_part * sizeof(int64_t));
    cudaMalloc(&d_ps_partkey_i64, nrecs_partsupp * sizeof(int64_t));
    cudaMalloc(&d_ps_suppkey_i64, nrecs_partsupp * sizeof(int64_t));

    // INT32 flat output
    int32_t* d_p_size_i32 = nullptr;
    cudaMalloc(&d_p_size_i32, nrecs_part * sizeof(int32_t));
    gpu_track("flat_output_arrays", gpu_track_free);

#if 0  // Debug: VCHAR decompressed page buffer for double-buffered pipeline
    // S_COMMENT uses this as a batch temp buffer (pages_per_batch pages),
    // P_BRAND / P_TYPE load all pages at once.
    uint64_t max_vchar_npages = std::max({(uint64_t)pages_per_batch, p_brand_npages, p_type_npages});
    char* d_vchar_pages = nullptr;
    cudaMalloc(&d_vchar_pages, max_vchar_npages * page_size);
#endif

    // GPU metadata for each field
    auto gm_s_suppkey  = alloc_gpu_field_meta(s_suppkey_npages, s_suppkey_comp);
    auto gm_s_comment  = alloc_gpu_field_meta(s_comment_npages, s_comment_comp);
    auto gm_p_partkey  = alloc_gpu_field_meta(p_partkey_npages, p_partkey_comp);
    auto gm_p_brand    = alloc_gpu_field_meta(p_brand_npages, p_brand_comp);
    auto gm_p_type     = alloc_gpu_field_meta(p_type_npages, p_type_comp);
    auto gm_p_size     = alloc_gpu_field_meta(p_size_npages, p_size_comp);
    auto gm_ps_partkey = alloc_gpu_field_meta(ps_partkey_npages, ps_partkey_comp);
    auto gm_ps_suppkey = alloc_gpu_field_meta(ps_suppkey_npages, ps_suppkey_comp);
    gpu_track("field_metadata_8cols", gpu_track_free);

    // KMP tables
    char* d_kmp_patterns = nullptr;
    int* d_kmp_next = nullptr;
    int* d_kmp_offsets = nullptr;
    int* d_kmp_lengths = nullptr;
    cudaMalloc(&d_kmp_patterns, excl_total_chars);
    cudaMalloc(&d_kmp_next, excl_total_chars * sizeof(int));
    cudaMalloc(&d_kmp_offsets, excl_num_patterns * sizeof(int));
    cudaMalloc(&d_kmp_lengths, excl_num_patterns * sizeof(int));

    // Exclusion: suppkey collector + HT
    uint64_t* d_excl_suppkeys = nullptr;
    uint32_t* d_excl_count = nullptr;
    cudaMalloc(&d_excl_suppkeys, nrecs_supplier * sizeof(uint64_t));
    cudaMalloc(&d_excl_count, sizeof(uint32_t));

    constexpr uint32_t EXCL_HT_PRE_CAP = 16384;
    uint64_t* d_excl_keys = nullptr;
    cudaMalloc(&d_excl_keys, EXCL_HT_PRE_CAP * sizeof(uint64_t));
    gpu_track("kmp+exclusion", gpu_track_free);

    // PART hash table
    uint32_t ht_capacity = 1;
    while (ht_capacity < nrecs_part * 2)
        ht_capacity <<= 1;
    uint32_t ht_mask = ht_capacity - 1;

    uint64_t* d_ht_keys = nullptr;
    uint32_t* d_ht_group_ids = nullptr;
    cudaMalloc(&d_ht_keys, ht_capacity * sizeof(uint64_t));
    cudaMalloc(&d_ht_group_ids, ht_capacity * sizeof(uint32_t));

    // Q16 intermediate arrays
    uint32_t* d_brand_ids = nullptr;
    uint32_t* d_type_ids = nullptr;
    cudaMalloc(&d_brand_ids, nrecs_part * sizeof(uint32_t));
    cudaMalloc(&d_type_ids, nrecs_part * sizeof(uint32_t));
    gpu_track("ht+brand_type_ids", gpu_track_free);

    // Type dictionary
    uint64_t* d_dict_keys = nullptr;
    uint32_t* d_dict_type_ids = nullptr;
    char* d_dict_strs = nullptr;
    uint16_t* d_dict_lens = nullptr;
    uint32_t* d_type_id_counter = nullptr;
    cudaMalloc(&d_dict_keys, Q16_TYPE_DICT_CAP * sizeof(uint64_t));
    cudaMalloc(&d_dict_type_ids, Q16_TYPE_DICT_CAP * sizeof(uint32_t));
    cudaMalloc(&d_dict_strs, (size_t)Q16_TYPE_DICT_CAP * Q16_TYPE_STR_MAX);
    cudaMalloc(&d_dict_lens, Q16_TYPE_DICT_CAP * sizeof(uint16_t));
    cudaMalloc(&d_type_id_counter, sizeof(uint32_t));
    gpu_track("type_dict", gpu_track_free);

    // Q16 pipeline buffers (pre-allocated)
    Q16PipelineBuffers pl_bufs{};
    cudaMalloc(&pl_bufs.d_emit_pairs, nrecs_partsupp * sizeof(uint64_t));
    cudaMalloc(&pl_bufs.d_sort_alt, nrecs_partsupp * sizeof(uint64_t));
    cudaMalloc(&pl_bufs.d_unique_keys, nrecs_partsupp * sizeof(uint64_t));
    cudaMalloc(&pl_bufs.d_unique_counts, nrecs_partsupp * sizeof(uint32_t));
    cudaMalloc(&pl_bufs.d_num_unique_ptr, sizeof(uint64_t));
    cudaMalloc(&pl_bufs.d_group_ids, nrecs_partsupp * sizeof(uint32_t));
    cudaMalloc(&pl_bufs.d_group_ids_alt, nrecs_partsupp * sizeof(uint32_t));
    cudaMalloc(&pl_bufs.d_result_gids, nrecs_partsupp * sizeof(uint32_t));
    cudaMalloc(&pl_bufs.d_result_counts, nrecs_partsupp * sizeof(uint32_t));
    cudaMalloc(&pl_bufs.d_num_groups_ptr, sizeof(uint64_t));
    pl_bufs.cub_temp_bytes = q16_pipeline_cub_temp_size(nrecs_partsupp);
    cudaMalloc(&pl_bufs.d_cub_temp, pl_bufs.cub_temp_bytes);
    constexpr size_t Q16_MAX_GROUPS = 256 * 1024;
    pl_bufs.h_result_capacity = Q16_MAX_GROUPS;
    pl_bufs.h_gids = (uint32_t *)malloc(Q16_MAX_GROUPS * sizeof(uint32_t));
    pl_bufs.h_counts = (uint32_t *)malloc(Q16_MAX_GROUPS * sizeof(uint32_t));
    std::cerr << "[Q16 GPU] cub_temp_bytes=" << pl_bufs.cub_temp_bytes / (1024*1024) << " MiB" << std::endl;
    gpu_track("pipeline_buffers", gpu_track_free);

    // ── BaM contexts ──
    bam_pfor64_io_ctx_t pfor64_ctx = bam_pfor64_io_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks_pfor);
    gpu_track("bam_pfor64_ctx", gpu_track_free);
    bam_pfor32_io_ctx_t pfor32_ctx = bam_pfor32_io_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks_pfor);
    gpu_track("bam_pfor32_ctx", gpu_track_free);
#if 0  // Debug: double-buffered VCHAR IO contexts
    bam_vchar_io_ctx_t vchar_io_a = bam_vchar_io_v2_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks);
    bam_vchar_io_ctx_t vchar_io_b = bam_vchar_io_v2_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks);
#endif

    // Fused IO+decomp+filter context (LZ4PAR — shared by S_COMMENT, P_BRAND, P_TYPE)
    bam_q16_fused_io_ctx_t fused_ctx{};
    if (any_lz4par_needed) {
        fused_ctx = bam_q16_fused_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks);
        gpu_track("bam_q16_fused_ctx", gpu_track_free);
    }

    // FSST fused IO context (no decomp_buf — shared by FSST columns)
    bam_q16_fsst_io_ctx_t fsst_fused_ctx{};
    if (any_fsst_needed) {
        fsst_fused_ctx = bam_q16_fsst_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks);
        gpu_track("bam_q16_fsst_ctx", gpu_track_free);
    }

    std::cout << "[Q16] sm_count=" << sm_count
              << " num_blocks_pfor=" << num_blocks_pfor
              << " num_blocks_fused=" << num_blocks << std::endl;

    // ── Helper: upload GPU metadata for a field ──
    auto upload_field_meta = [](const GpuFieldMeta& gm,
        const std::vector<uint64_t>& h_ps,
        const std::vector<uint32_t>& h_cs,
        const std::vector<uint64_t>& h_co,
        uint64_t npages, uint16_t comp)
    {
        cudaMemcpy(gm.d_prefix_sum, h_ps.data(),
                   npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (comp != 0) {
            cudaMemcpy(gm.d_comp_sizes, h_cs.data(),
                       npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(gm.d_comp_offsets, h_co.data(),
                       npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
    };

    // ── Helper: setup BAMPfor64FlattenParams ──
    auto make_pfor64_params = [&](uint64_t field_start, uint64_t npages,
        uint16_t comp, uint64_t nrows, const GpuFieldMeta& gm) -> BAMPfor64FlattenParams
    {
        BAMPfor64FlattenParams p{};
        p.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            p.partition_start_lbas[d] = ds.partition_start_lbas[d];
        p.n_devices = n_devices;
        p.page_size = static_cast<uint32_t>(page_size);
        p.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        p.comp_method = comp;
        p.field_start_page_id = field_start;
        p.npages = npages;
        p.nrows = nrows;
        p.num_blocks = num_blocks_pfor;
        p.d_prefix_sum = gm.d_prefix_sum;
        p.d_comp_sizes = gm.d_comp_sizes;
        p.d_comp_offsets = gm.d_comp_offsets;
        return p;
    };

#if 0  // Debug: double-buffered VCHAR I/O + decompress lambda
    double vchar_io_ms_accum = 0, vchar_decomp_ms_accum = 0, vchar_wait_ms_accum = 0;

    auto vchar_load_pages = [&](uint64_t npages_field, const GpuFieldMeta& gm,
                                 char* d_output_pages)
    {
        using hrc_inner = std::chrono::high_resolution_clock;
        bam_vchar_io_ctx_t cur_io  = vchar_io_a;
        bam_vchar_io_ctx_t nxt_io  = vchar_io_b;
        char* cur_stg = bam_vchar_io_v2_staging_buf(vchar_io_a);
        char* nxt_stg = bam_vchar_io_v2_staging_buf(vchar_io_b);

        double io_ms = 0, decomp_ms = 0, wait_ms = 0;

        // Prime: IO batch 0
        auto t0 = hrc_inner::now();
        uint64_t batch0_remaining = std::min((uint64_t)pages_per_batch, npages_field);
        uint32_t batch0_blocks = static_cast<uint32_t>(
            std::min((uint64_t)num_blocks, (batch0_remaining + 3) / 4));
        bam_vchar_io_v2_read_batch_async(cur_io, gm.d_comp_sizes, gm.d_comp_offsets,
                                          partition_start_lba, 0, batch0_blocks,
                                          npages_field);
        bam_vchar_io_v2_sync(cur_io);
        auto t1 = hrc_inner::now();
        io_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        for (uint64_t batch = 0; batch < npages_field; batch += pages_per_batch) {
            uint64_t remaining = npages_field - batch;
            uint32_t batch_pages = static_cast<uint32_t>(
                std::min((uint64_t)pages_per_batch, remaining));

            uint64_t next_batch = batch + pages_per_batch;
            bool has_next = (next_batch < npages_field);

            auto td0 = hrc_inner::now();
            bam_vchar_decomp_par32k_async(
                cur_stg, gm.d_comp_sizes, d_output_pages,
                static_cast<uint32_t>(page_size), batch, batch_pages,
                npages_field, comp_stream);

            if (has_next) {
                uint64_t nxt_remaining = npages_field - next_batch;
                uint32_t nxt_blocks = static_cast<uint32_t>(
                    std::min((uint64_t)num_blocks, (nxt_remaining + 3) / 4));
                bam_vchar_io_v2_read_batch_async(nxt_io, gm.d_comp_sizes,
                    gm.d_comp_offsets, partition_start_lba, next_batch,
                    nxt_blocks, npages_field);
            }

            auto td1 = hrc_inner::now();
            cudaStreamSynchronize(comp_stream);
            auto td2 = hrc_inner::now();
            if (has_next) {
                bam_vchar_io_v2_sync(nxt_io);
            }
            auto td3 = hrc_inner::now();

            decomp_ms += std::chrono::duration<double, std::milli>(td2 - td0).count();
            wait_ms   += std::chrono::duration<double, std::milli>(td3 - td2).count();

            std::swap(cur_io, nxt_io);
            std::swap(cur_stg, nxt_stg);
        }

        vchar_io_ms_accum = io_ms;
        vchar_decomp_ms_accum = decomp_ms;
        vchar_wait_ms_accum = wait_ms;
    };
#endif

    // Phase cycle profiling buffers
    uint64_t* d_sc_phase = nullptr;   // S_COMMENT
    uint64_t* d_br_phase = nullptr;   // P_BRAND
    uint64_t* d_ty_phase = nullptr;   // P_TYPE
    cudaMalloc(&d_sc_phase, 3 * sizeof(uint64_t));
    cudaMalloc(&d_br_phase, 3 * sizeof(uint64_t));
    cudaMalloc(&d_ty_phase, 3 * sizeof(uint64_t));

    uint32_t h_excl_count = 0;
    uint32_t num_types = 0;
    char h_dict_strs[Q16_TYPE_DICT_CAP * Q16_TYPE_STR_MAX];
    uint16_t h_dict_lens[Q16_TYPE_DICT_CAP];
    uint32_t h_dict_type_ids_h[Q16_TYPE_DICT_CAP];
    std::vector<std::pair<uint32_t, uint32_t>> q16_raw_result;

    // ── Pre-compute I/O statistics from metadata ──
    uint64_t q16_nios = s_suppkey_npages + s_comment_npages
                      + p_partkey_npages + p_brand_npages + p_type_npages + p_size_npages
                      + ps_partkey_npages + ps_suppkey_npages;
    uint64_t q16_read_bytes = 0;
    auto add_col_io_q16 = [&](uint64_t npages, uint16_t comp, const std::vector<uint32_t>& h_cs) {
        for (uint64_t pg = 0; pg < npages; pg++)
            q16_read_bytes += (comp != 0 && pg < h_cs.size()) ? h_cs[pg] : page_size;
    };
    add_col_io_q16(s_suppkey_npages, s_suppkey_comp, h_cs_s_suppkey);
    add_col_io_q16(s_comment_npages, s_comment_comp, h_cs_s_comment);
    add_col_io_q16(p_partkey_npages, p_partkey_comp, h_cs_p_partkey);
    add_col_io_q16(p_brand_npages, p_brand_comp, h_cs_p_brand);
    add_col_io_q16(p_type_npages, p_type_comp, h_cs_p_type);
    add_col_io_q16(p_size_npages, p_size_comp, h_cs_p_size);
    add_col_io_q16(ps_partkey_npages, ps_partkey_comp, h_cs_ps_partkey);
    add_col_io_q16(ps_suppkey_npages, ps_suppkey_comp, h_cs_ps_suppkey);

    // Upload all field metadata before measurement (Rule 3)
    upload_field_meta(gm_s_suppkey, h_ps_s_suppkey, h_cs_s_suppkey, h_co_s_suppkey,
                      s_suppkey_npages, s_suppkey_comp);
    upload_field_meta(gm_s_comment, h_ps_s_comment, h_cs_s_comment, h_co_s_comment,
                      s_comment_npages, s_comment_comp);
    upload_field_meta(gm_p_partkey, h_ps_p_partkey, h_cs_p_partkey, h_co_p_partkey,
                      p_partkey_npages, p_partkey_comp);
    upload_field_meta(gm_p_size, h_ps_p_size, h_cs_p_size, h_co_p_size,
                      p_size_npages, p_size_comp);
    upload_field_meta(gm_p_brand, h_ps_p_brand, h_cs_p_brand, h_co_p_brand,
                      p_brand_npages, p_brand_comp);
    upload_field_meta(gm_p_type, h_ps_p_type, h_cs_p_type, h_co_p_type,
                      p_type_npages, p_type_comp);
    upload_field_meta(gm_ps_partkey, h_ps_ps_partkey, h_cs_ps_partkey, h_co_ps_partkey,
                      ps_partkey_npages, ps_partkey_comp);
    upload_field_meta(gm_ps_suppkey, h_ps_ps_suppkey, h_cs_ps_suppkey, h_co_ps_suppkey,
                      ps_suppkey_npages, ps_suppkey_comp);

    // Upload KMP tables before measurement (Rule 3)
    cudaMemcpy(d_kmp_patterns, excl_patterns_str, excl_total_chars, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kmp_next, excl_next, excl_total_chars * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kmp_offsets, excl_pattern_offsets,
               excl_num_patterns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kmp_lengths, excl_pattern_lengths,
               excl_num_patterns * sizeof(int), cudaMemcpyHostToDevice);

    size_t gpu_free_after = 0;
    cudaMemGetInfo(&gpu_free_after, &gpu_total);
    uint64_t q16_gpu_mem_bytes = gpu_free_before - gpu_free_after;

    // ================================================================
    // total_start — query processing begins
    // ================================================================
    auto total_start = std::chrono::high_resolution_clock::now();
    s_kernel_launches = 0;
    using hrc = std::chrono::high_resolution_clock;
    auto dur_ms = [](hrc::time_point a, hrc::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    cudaMemset(d_sc_phase, 0, 3 * sizeof(uint64_t));
    cudaMemset(d_br_phase, 0, 3 * sizeof(uint64_t));
    cudaMemset(d_ty_phase, 0, 3 * sizeof(uint64_t));

    // ── Phase 0: SUPPLIER anti-join ──
    // 0a: S_SUPPKEY (pfor64, sync — 48 pages, tiny)
    std::cout << "[Q16] Loading S_SUPPKEY (" << s_suppkey_npages << " pages)..." << std::endl;
    bam_pfor64_io_flatten_async(pfor64_ctx,
        make_pfor64_params(s_suppkey_start, s_suppkey_npages, s_suppkey_comp,
                           nrecs_supplier, gm_s_suppkey),
        d_s_suppkey_i64, stream);
    s_kernel_launches++;
    cudaStreamSynchronize(stream);
    auto t_s_suppkey = hrc::now();

    // 0b: S_COMMENT pipelined IO + decomp + scan
    std::cout << "[Q16] Scanning S_COMMENT (" << s_comment_npages
              << " pages, pipelined IO+decomp+scan)..." << std::endl;

    cudaMemsetAsync(d_excl_count, 0, sizeof(uint32_t), stream);
    cudaStreamSynchronize(stream);

    // Fused IO+decomp+KMP scan for S_COMMENT
    if (is_fsst_s_comment) {
        // FSST fused kernel path
        uint32_t sc_num_blocks = num_blocks;
        if (sc_num_blocks > static_cast<uint32_t>(s_comment_npages))
            sc_num_blocks = static_cast<uint32_t>(s_comment_npages);

        BAMq16FsstSCommentParams fsc{};
        fsc.d_comp_sizes      = gm_s_comment.d_comp_sizes;
        fsc.d_comp_offsets    = gm_s_comment.d_comp_offsets;
        fsc.d_prefix_sum      = gm_s_comment.d_prefix_sum;
        fsc.d_s_suppkey_flat  = reinterpret_cast<const uint64_t*>(d_s_suppkey_i64);
        fsc.d_excl_suppkeys   = d_excl_suppkeys;
        fsc.d_excl_count      = d_excl_count;
        fsc.d_patterns        = d_kmp_patterns;
        fsc.d_next            = d_kmp_next;
        fsc.d_pattern_offsets = d_kmp_offsets;
        fsc.d_pattern_lengths = d_kmp_lengths;
        fsc.num_patterns      = excl_num_patterns;
        fsc.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            fsc.partition_start_lbas[d] = ds.partition_start_lbas[d];
        fsc.n_devices = n_devices;
        fsc.field_start_page_id = s_comment_start;
        fsc.page_size         = static_cast<uint32_t>(page_size);
        fsc.npages            = s_comment_npages;
        fsc.num_blocks        = sc_num_blocks;
        fsc.d_phase_cycles    = d_sc_phase;

        bam_q16_fsst_s_comment_async(fsst_fused_ctx, fsc, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    } else {
        // LZ4PAR fused kernel path
        uint32_t sc_num_blocks = num_blocks;
        if (sc_num_blocks > static_cast<uint32_t>(s_comment_npages))
            sc_num_blocks = static_cast<uint32_t>(s_comment_npages);

        BAMq16FusedSCommentParams sc_params{};
        sc_params.d_comp_sizes      = gm_s_comment.d_comp_sizes;
        sc_params.d_comp_offsets    = gm_s_comment.d_comp_offsets;
        sc_params.d_prefix_sum      = gm_s_comment.d_prefix_sum;
        sc_params.d_s_suppkey_flat  = reinterpret_cast<const uint64_t*>(d_s_suppkey_i64);
        sc_params.d_excl_suppkeys   = d_excl_suppkeys;
        sc_params.d_excl_count      = d_excl_count;
        sc_params.d_patterns        = d_kmp_patterns;
        sc_params.d_next            = d_kmp_next;
        sc_params.d_pattern_offsets = d_kmp_offsets;
        sc_params.d_pattern_lengths = d_kmp_lengths;
        sc_params.num_patterns      = excl_num_patterns;
        sc_params.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            sc_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        sc_params.n_devices = n_devices;
        sc_params.field_start_page_id = s_comment_start;
        sc_params.page_size         = static_cast<uint32_t>(page_size);
        sc_params.npages            = s_comment_npages;
        sc_params.num_blocks        = sc_num_blocks;
        sc_params.comp_method       = s_comment_comp;
        sc_params.d_phase_cycles    = d_sc_phase;

        bam_q16_fused_s_comment_async(fused_ctx, sc_params, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }

#if 0  // Debug: double-buffered IO + decomp + batch scan pipeline
    {
        bam_vchar_io_ctx_t cur_io = vchar_io_a;
        bam_vchar_io_ctx_t nxt_io = vchar_io_b;
        char* cur_stg = bam_vchar_io_v2_staging_buf(vchar_io_a);
        char* nxt_stg = bam_vchar_io_v2_staging_buf(vchar_io_b);

        uint64_t batch0_remaining = std::min((uint64_t)pages_per_batch, s_comment_npages);
        uint32_t batch0_blocks = static_cast<uint32_t>(
            std::min((uint64_t)num_blocks, (batch0_remaining + 3) / 4));
        bam_vchar_io_v2_read_batch_async(cur_io, gm_s_comment.d_comp_sizes,
            gm_s_comment.d_comp_offsets, partition_start_lba, 0, batch0_blocks,
            s_comment_npages);
        bam_vchar_io_v2_sync(cur_io);

        for (uint64_t batch = 0; batch < s_comment_npages; batch += pages_per_batch) {
            uint64_t remaining = s_comment_npages - batch;
            uint32_t batch_pages = static_cast<uint32_t>(
                std::min((uint64_t)pages_per_batch, remaining));

            uint64_t next_batch = batch + pages_per_batch;
            bool has_next = (next_batch < s_comment_npages);

            bam_vchar_decomp_par32k_async(
                cur_stg, gm_s_comment.d_comp_sizes, d_vchar_pages,
                static_cast<uint32_t>(page_size),
                0, batch_pages, batch_pages, comp_stream);

            if (has_next) {
                uint64_t nxt_remaining = s_comment_npages - next_batch;
                uint32_t nxt_blocks = static_cast<uint32_t>(
                    std::min((uint64_t)num_blocks, (nxt_remaining + 3) / 4));
                bam_vchar_io_v2_read_batch_async(nxt_io, gm_s_comment.d_comp_sizes,
                    gm_s_comment.d_comp_offsets, partition_start_lba, next_batch,
                    nxt_blocks, s_comment_npages);
            }

            cudaStreamSynchronize(comp_stream);

            uint64_t row_base = (batch == 0) ? 0 : h_ps_s_comment[batch - 1];
            uint64_t row_end  = h_ps_s_comment[batch + batch_pages - 1];
            uint64_t nrecs_batch = row_end - row_base;

            q16_supplier_scan_batch(
                d_vchar_pages, gm_s_comment.d_prefix_sum,
                static_cast<uint32_t>(s_comment_npages),
                static_cast<uint32_t>(batch),
                static_cast<uint32_t>(page_size),
                reinterpret_cast<uint64_t*>(d_s_suppkey_i64),
                nrecs_batch, row_base,
                d_kmp_patterns, d_kmp_next, d_kmp_offsets, d_kmp_lengths,
                excl_num_patterns, d_excl_suppkeys, d_excl_count, stream);
            cudaStreamSynchronize(stream);

            if (has_next) {
                bam_vchar_io_v2_sync(nxt_io);
            }

            std::swap(cur_io, nxt_io);
            std::swap(cur_stg, nxt_stg);
        }
    }
#endif
    auto t_s_comment = hrc::now();

    h_excl_count = 0;
    cudaMemcpy(&h_excl_count, d_excl_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "[Q16] Excluded suppliers: " << h_excl_count << std::endl;

    // 0d: Build exclusion HT
    uint32_t excl_capacity = 1;
    while (excl_capacity < (uint32_t)h_excl_count * 4 + 16)
        excl_capacity <<= 1;
    if (excl_capacity > EXCL_HT_PRE_CAP) {
        std::cerr << "ERROR: excl_count exceeds pre-allocated HT capacity" << std::endl;
        excl_capacity = EXCL_HT_PRE_CAP;
    }
    uint32_t excl_mask = excl_capacity - 1;
    cudaMemsetAsync(d_excl_keys, 0xFF, excl_capacity * sizeof(uint64_t), stream);
    cudaStreamSynchronize(stream);

    q16_build_excl_ht(d_excl_suppkeys, h_excl_count,
                      d_excl_keys, excl_mask, stream);
    s_kernel_launches++;
    cudaStreamSynchronize(stream);
    auto t_excl_ht = hrc::now();

    // ── Phase 1: PART scan + HT_PART ──

    std::cout << "[Q16] Loading P_PARTKEY (" << p_partkey_npages << " pages)..." << std::endl;
    bam_pfor64_io_flatten_async(pfor64_ctx,
        make_pfor64_params(p_partkey_start, p_partkey_npages, p_partkey_comp,
                           nrecs_part, gm_p_partkey),
        d_p_partkey_i64, stream);
    s_kernel_launches++;
    cudaStreamSynchronize(stream);
    auto t_p_partkey = hrc::now();

    std::cout << "[Q16] Loading P_SIZE (" << p_size_npages << " pages)..." << std::endl;
    {
        BAMPfor32FlattenParams p32{};
        p32.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            p32.partition_start_lbas[d] = ds.partition_start_lbas[d];
        p32.n_devices = n_devices;
        p32.page_size = static_cast<uint32_t>(page_size);
        p32.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        p32.comp_method = p_size_comp;
        p32.field_start_page_id = p_size_start;
        p32.npages = p_size_npages;
        p32.nrows = nrecs_part;
        p32.num_blocks = num_blocks_pfor;
        p32.d_prefix_sum = gm_p_size.d_prefix_sum;
        p32.d_comp_sizes = gm_p_size.d_comp_sizes;
        p32.d_comp_offsets = gm_p_size.d_comp_offsets;
        bam_pfor32_io_flatten_async(pfor32_ctx, p32, d_p_size_i32, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }
    auto t_p_size = hrc::now();

    // 1b: P_BRAND — fused IO + decomp + brand_id extraction
    std::cout << "[Q16] Loading P_BRAND (" << p_brand_npages
              << " pages, fused IO+decomp+extract)..." << std::endl;

    constexpr uint32_t BRAND45_ID = 19;
    constexpr uint32_t P_BRAND_PADDED_LEN = 12;

    if (is_fsst_p_brand) {
        // FSST fused kernel path
        uint32_t br_num_blocks = num_blocks;
        if (br_num_blocks > static_cast<uint32_t>(p_brand_npages))
            br_num_blocks = static_cast<uint32_t>(p_brand_npages);

        BAMq16FsstBrandParams fbr{};
        fbr.d_comp_sizes      = gm_p_brand.d_comp_sizes;
        fbr.d_comp_offsets    = gm_p_brand.d_comp_offsets;
        fbr.d_prefix_sum      = gm_p_brand.d_prefix_sum;
        fbr.d_brand_ids       = d_brand_ids;
        fbr.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            fbr.partition_start_lbas[d] = ds.partition_start_lbas[d];
        fbr.n_devices = n_devices;
        fbr.field_start_page_id = p_brand_start;
        fbr.page_size         = static_cast<uint32_t>(page_size);
        fbr.npages            = p_brand_npages;
        fbr.num_blocks        = br_num_blocks;
        fbr.d_phase_cycles    = d_br_phase;

        bam_q16_fsst_p_brand_async(fsst_fused_ctx, fbr, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    } else {
        // LZ4PAR fused kernel path
        uint32_t br_num_blocks = num_blocks;
        if (br_num_blocks > static_cast<uint32_t>(p_brand_npages))
            br_num_blocks = static_cast<uint32_t>(p_brand_npages);

        BAMq16FusedBrandParams br_params{};
        br_params.d_comp_sizes      = gm_p_brand.d_comp_sizes;
        br_params.d_comp_offsets    = gm_p_brand.d_comp_offsets;
        br_params.d_prefix_sum      = gm_p_brand.d_prefix_sum;
        br_params.padded_len        = P_BRAND_PADDED_LEN;
        br_params.d_brand_ids       = d_brand_ids;
        br_params.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            br_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        br_params.n_devices = n_devices;
        br_params.field_start_page_id = p_brand_start;
        br_params.page_size         = static_cast<uint32_t>(page_size);
        br_params.npages            = p_brand_npages;
        br_params.num_blocks        = br_num_blocks;
        br_params.comp_method       = p_brand_comp;
        br_params.d_phase_cycles    = d_br_phase;

        bam_q16_fused_p_brand_async(fused_ctx, br_params, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }
    auto t_p_brand = hrc::now();

#if 0  // Debug: double-buffered P_BRAND IO + extract
    auto t_p_brand_io_start = hrc::now();
    vchar_load_pages(p_brand_npages, gm_p_brand, d_vchar_pages);
    auto t_p_brand_io_done = hrc::now();
    double p_brand_prime_io = vchar_io_ms_accum;
    double p_brand_decomp = vchar_decomp_ms_accum;
    double p_brand_io_wait = vchar_wait_ms_accum;

    q16_extract_brand_ids(
        d_vchar_pages, gm_p_brand.d_prefix_sum,
        static_cast<uint32_t>(p_brand_npages), static_cast<uint32_t>(page_size),
        P_BRAND_PADDED_LEN, nrecs_part, d_brand_ids, stream);
    cudaStreamSynchronize(stream);
    auto t_p_brand_old = hrc::now();
#endif

    // 1c: P_TYPE — fused IO + decomp + type_id dictionary extraction
    std::cout << "[Q16] Loading P_TYPE (" << p_type_npages
              << " pages, fused IO+decomp+extract)..." << std::endl;

    cudaMemsetAsync(d_dict_keys, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_dict_type_ids, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint32_t), stream);
    cudaMemsetAsync(d_type_id_counter, 0, sizeof(uint32_t), stream);
    cudaStreamSynchronize(stream);

    if (is_fsst_p_type) {
        // FSST fused kernel path
        uint32_t ty_num_blocks = num_blocks;
        if (ty_num_blocks > static_cast<uint32_t>(p_type_npages))
            ty_num_blocks = static_cast<uint32_t>(p_type_npages);

        BAMq16FsstTypeParams fty{};
        fty.d_comp_sizes      = gm_p_type.d_comp_sizes;
        fty.d_comp_offsets    = gm_p_type.d_comp_offsets;
        fty.d_prefix_sum      = gm_p_type.d_prefix_sum;
        fty.d_dict_keys       = d_dict_keys;
        fty.d_dict_type_ids   = d_dict_type_ids;
        fty.d_dict_strs       = d_dict_strs;
        fty.d_dict_lens       = d_dict_lens;
        fty.d_type_id_counter = d_type_id_counter;
        fty.d_type_ids        = d_type_ids;
        fty.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            fty.partition_start_lbas[d] = ds.partition_start_lbas[d];
        fty.n_devices = n_devices;
        fty.field_start_page_id = p_type_start;
        fty.page_size         = static_cast<uint32_t>(page_size);
        fty.npages            = p_type_npages;
        fty.num_blocks        = ty_num_blocks;
        fty.d_phase_cycles    = d_ty_phase;

        bam_q16_fsst_p_type_async(fsst_fused_ctx, fty, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    } else {
        // LZ4PAR fused kernel path
        uint32_t ty_num_blocks = num_blocks;
        if (ty_num_blocks > static_cast<uint32_t>(p_type_npages))
            ty_num_blocks = static_cast<uint32_t>(p_type_npages);

        BAMq16FusedTypeParams ty_params{};
        ty_params.d_comp_sizes      = gm_p_type.d_comp_sizes;
        ty_params.d_comp_offsets    = gm_p_type.d_comp_offsets;
        ty_params.d_prefix_sum      = gm_p_type.d_prefix_sum;
        ty_params.d_dict_keys       = d_dict_keys;
        ty_params.d_dict_type_ids   = d_dict_type_ids;
        ty_params.d_dict_strs       = d_dict_strs;
        ty_params.d_dict_lens       = d_dict_lens;
        ty_params.d_type_id_counter = d_type_id_counter;
        ty_params.d_type_ids        = d_type_ids;
        ty_params.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            ty_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
        ty_params.n_devices = n_devices;
        ty_params.field_start_page_id = p_type_start;
        ty_params.page_size         = static_cast<uint32_t>(page_size);
        ty_params.npages            = p_type_npages;
        ty_params.num_blocks        = ty_num_blocks;
        ty_params.comp_method       = p_type_comp;
        ty_params.d_phase_cycles    = d_ty_phase;

        bam_q16_fused_p_type_async(fused_ctx, ty_params, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }
    auto t_p_type_extract_done = hrc::now();

#if 0  // Debug: double-buffered P_TYPE IO + extract
    auto t_p_type_io_start = hrc::now();
    vchar_load_pages(p_type_npages, gm_p_type, d_vchar_pages);
    auto t_p_type_io_done = hrc::now();
    double p_type_prime_io = vchar_io_ms_accum;
    double p_type_decomp = vchar_decomp_ms_accum;
    double p_type_io_wait = vchar_wait_ms_accum;

    cudaMemsetAsync(d_dict_keys, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_dict_type_ids, 0xFF, Q16_TYPE_DICT_CAP * sizeof(uint32_t), stream);
    cudaMemsetAsync(d_type_id_counter, 0, sizeof(uint32_t), stream);
    cudaStreamSynchronize(stream);

    q16_extract_type_ids(
        d_vchar_pages, gm_p_type.d_prefix_sum,
        static_cast<uint32_t>(p_type_npages), static_cast<uint32_t>(page_size),
        nrecs_part, d_dict_keys, d_dict_type_ids, d_dict_strs, d_dict_lens,
        d_type_id_counter, d_type_ids, stream);
    cudaStreamSynchronize(stream);
    auto t_p_type_extract_done_old = hrc::now();
#endif

    num_types = 0;
    cudaMemcpy(&num_types, d_type_id_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "[Q16] P_TYPE distinct values (excluding MEDIUM POLISHED): "
              << num_types << std::endl;

    // Copy type dictionary to host (for result decoding later)
    cudaMemcpy(h_dict_strs, d_dict_strs,
        (size_t)Q16_TYPE_DICT_CAP * Q16_TYPE_STR_MAX, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dict_lens, d_dict_lens,
        Q16_TYPE_DICT_CAP * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dict_type_ids_h, d_dict_type_ids,
        Q16_TYPE_DICT_CAP * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    auto t_p_type = hrc::now();

    // 1g: Build PART hash table
    uint64_t p_size_bitmask = (1ULL << 49) | (1ULL << 14) | (1ULL << 23) | (1ULL << 45)
                            | (1ULL << 19) | (1ULL <<  3) | (1ULL << 36) | (1ULL <<  9);

    cudaMemsetAsync(d_ht_keys, 0xFF, ht_capacity * sizeof(uint64_t), stream);
    cudaStreamSynchronize(stream);

    q16_build_part_hashtable(
        reinterpret_cast<uint64_t*>(d_p_partkey_i64), d_brand_ids, d_type_ids,
        reinterpret_cast<uint32_t*>(d_p_size_i32),
        nrecs_part, p_size_bitmask, BRAND45_ID, num_types,
        d_ht_keys, d_ht_group_ids, ht_mask, stream);
    s_kernel_launches++;
    cudaStreamSynchronize(stream);
    auto t_part_ht = hrc::now();

    // ── Phase 2: PARTSUPP probe + COUNT DISTINCT ──

    std::cout << "[Q16] Loading PS_PARTKEY + PS_SUPPKEY (" << ps_partkey_npages
              << " pages each, dual flatten)..." << std::endl;
    {
        BAMPfor64DualFlattenParams dp{};
        for (uint32_t d = 0; d < n_devices; d++)
            dp.partition_start_lbas[d] = ds.partition_start_lbas[d];
        dp.n_devices = n_devices;
        dp.page_size = static_cast<uint32_t>(page_size);
        dp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        dp.num_blocks = num_blocks_pfor;
        dp.npages = ps_partkey_npages;

        dp.field0_start_page_id = ps_partkey_start;
        dp.field0_comp_method = ps_partkey_comp;
        dp.field0_d_comp_sizes = gm_ps_partkey.d_comp_sizes;
        dp.field0_d_comp_offsets = gm_ps_partkey.d_comp_offsets;
        dp.field0_d_prefix_sum = gm_ps_partkey.d_prefix_sum;
        dp.field0_d_output = d_ps_partkey_i64;

        dp.field1_start_page_id = ps_suppkey_start;
        dp.field1_comp_method = ps_suppkey_comp;
        dp.field1_d_comp_sizes = gm_ps_suppkey.d_comp_sizes;
        dp.field1_d_comp_offsets = gm_ps_suppkey.d_comp_offsets;
        dp.field1_d_prefix_sum = gm_ps_suppkey.d_prefix_sum;
        dp.field1_d_output = d_ps_suppkey_i64;

        bam_pfor64_dual_flatten_async(pfor64_ctx, dp, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }
    auto t_ps_suppkey = hrc::now();

    // 2c: Pipeline (probe + sort-based COUNT DISTINCT)
    std::cout << "[Q16] Running Q16 pipeline..." << std::endl;
    q16_raw_result.clear();

    q16_golap_pipeline(pl_bufs, d_ht_keys, d_ht_group_ids, ht_mask,
        d_excl_keys, excl_mask,
        reinterpret_cast<uint64_t*>(d_ps_partkey_i64),
        reinterpret_cast<uint64_t*>(d_ps_suppkey_i64),
        nrecs_partsupp, q16_raw_result, stream);
    s_kernel_launches++;

    // ================================================================
    // total_end — query processing ends
    // ================================================================
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // ── Timing breakdown ──
    {
        std::cout << "\n--- Q16 PiG Phase Timing Breakdown ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  S_SUPPKEY  (pfor64 flatten):   " << dur_ms(total_start, t_s_suppkey) << " ms" << std::endl;
        std::cout << "  S_COMMENT  (" << (is_fsst_s_comment ? "FSST" : "LZ4PAR") << " fused IO+decomp+scan): " << dur_ms(t_s_suppkey, t_s_comment) << " ms" << std::endl;
        std::cout << "  Excl HT    (build):            " << dur_ms(t_s_comment, t_excl_ht) << " ms" << std::endl;
        std::cout << "  P_PARTKEY  (pfor64 flatten):   " << dur_ms(t_excl_ht, t_p_partkey) << " ms" << std::endl;
        std::cout << "  P_SIZE     (pfor32 flatten):   " << dur_ms(t_p_partkey, t_p_size) << " ms" << std::endl;
        std::cout << "  P_BRAND    (" << (is_fsst_p_brand ? "FSST" : "LZ4PAR") << " fused IO+decomp+extract): " << dur_ms(t_p_size, t_p_brand) << " ms" << std::endl;
        std::cout << "  P_TYPE     (" << (is_fsst_p_type ? "FSST" : "LZ4PAR") << " fused IO+decomp+extract): " << dur_ms(t_p_brand, t_p_type_extract_done) << " ms"
                  << "  [fused=" << dur_ms(t_p_brand, t_p_type_extract_done)
                  << ", dictD2H=" << dur_ms(t_p_type_extract_done, t_p_type) << "]" << std::endl;
        std::cout << "  Part HT    (build):            " << dur_ms(t_p_type, t_part_ht) << " ms" << std::endl;
        std::cout << "  PS_PARTKEY+PS_SUPPKEY (dual flatten): " << dur_ms(t_part_ht, t_ps_suppkey) << " ms" << std::endl;
        std::cout << "  Pipeline   (probe+COUNT DIST): " << dur_ms(t_ps_suppkey, total_end) << " ms" << std::endl;
        std::cout << "  ─────────────────────────────────────" << std::endl;
        double pfor_total = dur_ms(total_start, t_s_suppkey) + dur_ms(t_excl_ht, t_p_partkey)
                          + dur_ms(t_p_partkey, t_p_size) + dur_ms(t_part_ht, t_ps_suppkey);
        std::cout << "  PFOR total (4x pfor64 + 1x pfor32 + 1x dual): " << pfor_total
                  << " ms  (incl. cudaDeviceSync)" << std::endl;
        std::cout << "--------------------------------------" << std::endl;

        // ── Per-phase GPU cycle profiling for fused kernels ──
        {
            int clock_rate_khz = 0;
            cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);

            auto print_phase = [&](const char* label, uint64_t* d_phase,
                                   uint32_t nblocks, uint64_t npages) {
                uint64_t h[3] = {};
                cudaMemcpy(h, d_phase, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                uint64_t ppb = (npages + nblocks - 1) / nblocks;
                double dec_ms    = (double)h[1] / nblocks / clock_rate_khz;
                double fltio_ms  = (double)h[2] / nblocks / clock_rate_khz;
                std::cout << "  [" << label << "] GPU clock=" << clock_rate_khz
                          << "kHz  blocks=" << nblocks
                          << "  pages/block=" << ppb << std::endl;
                std::cout << "    Decomp:        " << dec_ms   << " ms  (avg/pg: "
                          << dec_ms / std::max(ppb, (uint64_t)1) << ")" << std::endl;
                std::cout << "    Filter+IO(nxt): " << fltio_ms << " ms  (avg/pg: "
                          << fltio_ms / std::max(ppb, (uint64_t)1) << ")" << std::endl;
                std::cout << "    Total:         " << (dec_ms + fltio_ms) << " ms" << std::endl;
            };

            uint32_t sc_nb = std::min(num_blocks, static_cast<uint32_t>(s_comment_npages));
            uint32_t br_nb = std::min(num_blocks, static_cast<uint32_t>(p_brand_npages));
            uint32_t ty_nb = std::min(num_blocks, static_cast<uint32_t>(p_type_npages));
            print_phase("S_COMMENT", d_sc_phase, sc_nb, s_comment_npages);
            print_phase("P_BRAND",   d_br_phase, br_nb, p_brand_npages);
            print_phase("P_TYPE",    d_ty_phase, ty_nb, p_type_npages);
        }
    }

    cudaFree(d_sc_phase);
    cudaFree(d_br_phase);
    cudaFree(d_ty_phase);

    // ── Result expansion + sort ──
    // Build reverse lookup: type_id → string
    char type_str_pool[150][Q16_TYPE_STR_MAX];
    uint16_t type_str_lens[150];
    memset(type_str_pool, 0, sizeof(type_str_pool));
    memset(type_str_lens, 0, sizeof(type_str_lens));
    for (uint32_t s = 0; s < Q16_TYPE_DICT_CAP; s++) {
        uint32_t tid = h_dict_type_ids_h[s];
        if (tid != UINT32_MAX && tid < 150) {
            memcpy(type_str_pool[tid], h_dict_strs + (size_t)s * Q16_TYPE_STR_MAX,
                   h_dict_lens[s]);
            type_str_lens[tid] = h_dict_lens[s];
        }
    }

    std::vector<Q16ResultRow> q16_result;
    q16_result.reserve(q16_raw_result.size());

    for (auto& [gid, cnt] : q16_raw_result) {
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
        [](const Q16ResultRow& a, const Q16ResultRow& b) {
            if (a.supplier_cnt != b.supplier_cnt) return a.supplier_cnt > b.supplier_cnt;
            if (a.p_brand != b.p_brand) return a.p_brand < b.p_brand;
            if (a.p_type != b.p_type) return a.p_type < b.p_type;
            return a.p_size < b.p_size;
        });

    // Print results
    std::cout << "\n=== TPC-H Q16 Result ===" << std::endl;
    std::cout << "p_brand    | p_type                    | p_size | supplier_cnt" << std::endl;
    std::cout << "-----------+---------------------------+--------+-------------" << std::endl;
    for (auto& r : q16_result) {
        printf("%-10s | %-25s | %6d | %12u\n",
               r.p_brand.c_str(), r.p_type.c_str(), r.p_size, r.supplier_cnt);
    }
    std::cout << "\nTotal rows: " << q16_result.size() << std::endl;

    // ================================================================
    // Cleanup (after total_end)
    // ================================================================
    bam_pfor64_io_destroy(pfor64_ctx);
    bam_pfor32_io_destroy(pfor32_ctx);
    if (any_lz4par_needed) bam_q16_fused_io_destroy(fused_ctx);
    if (any_fsst_needed)   bam_q16_fsst_io_destroy(fsst_fused_ctx);
#if 0  // Debug: double-buffered VCHAR IO contexts
    bam_vchar_io_v2_destroy(vchar_io_a);
    bam_vchar_io_v2_destroy(vchar_io_b);
#endif

    auto free_field_meta = [](const GpuFieldMeta& gm) {
        cudaFree(gm.d_prefix_sum);
        cudaFree(gm.d_comp_sizes);
        cudaFree(gm.d_comp_offsets);
    };
    free_field_meta(gm_s_suppkey);
    free_field_meta(gm_s_comment);
    free_field_meta(gm_p_partkey);
    free_field_meta(gm_p_brand);
    free_field_meta(gm_p_type);
    free_field_meta(gm_p_size);
    free_field_meta(gm_ps_partkey);
    free_field_meta(gm_ps_suppkey);

    cudaFree(d_s_suppkey_i64);
    cudaFree(d_p_partkey_i64);
    cudaFree(d_ps_partkey_i64);
    cudaFree(d_ps_suppkey_i64);
    cudaFree(d_p_size_i32);
#if 0  // Debug
    cudaFree(d_vchar_pages);
#endif
    cudaFree(d_kmp_patterns);
    cudaFree(d_kmp_next);
    cudaFree(d_kmp_offsets);
    cudaFree(d_kmp_lengths);
    cudaFree(d_excl_suppkeys);
    cudaFree(d_excl_count);
    cudaFree(d_excl_keys);
    cudaFree(d_ht_keys);
    cudaFree(d_ht_group_ids);
    cudaFree(d_brand_ids);
    cudaFree(d_type_ids);
    cudaFree(d_dict_keys);
    cudaFree(d_dict_type_ids);
    cudaFree(d_dict_strs);
    cudaFree(d_dict_lens);
    cudaFree(d_type_id_counter);
    cudaFree(pl_bufs.d_emit_pairs);
    cudaFree(pl_bufs.d_sort_alt);
    cudaFree(pl_bufs.d_unique_keys);
    cudaFree(pl_bufs.d_unique_counts);
    cudaFree(pl_bufs.d_num_unique_ptr);
    cudaFree(pl_bufs.d_group_ids);
    cudaFree(pl_bufs.d_group_ids_alt);
    cudaFree(pl_bufs.d_result_gids);
    cudaFree(pl_bufs.d_result_counts);
    cudaFree(pl_bufs.d_num_groups_ptr);
    cudaFree(pl_bufs.d_cub_temp);
    free(pl_bufs.h_gids);
    free(pl_bufs.h_counts);

    cudaStreamDestroy(stream);
#if 0  // Debug
    cudaStreamDestroy(comp_stream);
#endif
    bam_ctrl_close(ctrl);
    return PigResult{total_ms, q16_nios, q16_read_bytes,
                     collect_comp_methods({s_suppkey_comp, s_comment_comp,
                                           p_partkey_comp, p_brand_comp, p_type_comp, p_size_comp,
                                           ps_partkey_comp, ps_suppkey_comp}),
                     gpu_ctrl_bytes + q16_gpu_mem_bytes, gpu_ctrl_bytes, q16_gpu_mem_bytes,
                     q16_nios,
                     s_kernel_launches};
}

// ============================================================
// test_lz4_page — read first compressed O_COMMENT page,
// decompress with per-thread LZ4 (new) and nvCOMPdx (existing),
// compare byte-for-byte.
// ============================================================
void test_lz4_page(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    const uint32_t bam_num_queues = 128;
    bam_ctrl_handle_t ctrl = bam_ctrl_open(bam_ctrl_path, 1, 0, 1024, bam_num_queues);
    uint64_t partition_start_lba = detect_partition_start_lba(ctrl);

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, partition_start_lba, head_buf.data());
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    bam_read_page(ctrl, page_size, partition_start_lba, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;
    constexpr size_t col = TPCH::common::O_COMMENT;
    uint64_t field_start_page_id = metadata.table_orders_start_page_ids[col];
    uint64_t npages = metadata.table_orders_npages[col];
    uint16_t comp_method = metadata.table_orders_compression_method[col];

    std::cout << "O_COMMENT: start_page=" << field_start_page_id
              << " npages=" << npages
              << " compression=" << comp_method << std::endl;

    if (comp_method == 0) {
        std::cerr << "O_COMMENT is NOT compressed. Expected LZ4." << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // Read compressed_page_sizes
    uint64_t cs_start = metadata.table_orders_compressed_page_sizes_start_page_ids[col];
    uint64_t cs_npages = metadata.table_orders_compressed_page_sizes_npages[col];
    std::vector<char> sizes_buf(cs_npages * page_size);
    for (uint64_t p = 0; p < cs_npages; p++) {
        uint64_t lba = partition_start_lba + (cs_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, sizes_buf.data() + p * page_size);
    }
    uint32_t* comp_page_sizes = reinterpret_cast<uint32_t*>(sizes_buf.data());

    // Read base_page_ids and compute compressed offsets
    uint64_t nbase = metadata.table_orders_compression_nbases[col];
    uint64_t base_start = metadata.table_orders_compression_base_start_page_ids[col];
    size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    for (size_t p = 0; p < base_npages; p++) {
        uint64_t lba = partition_start_lba + (base_start + p) * blocks_per_page;
        bam_read_page(ctrl, page_size, lba, bases_buf.data() + p * page_size);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size, field_start_page_id,
        1, offsets_vec);

    // === Read the first compressed page ===
    uint32_t cs0 = comp_page_sizes[0];
    uint64_t offset0 = offsets_vec[0];
    uint64_t lba0 = partition_start_lba + offset0 / 512;
    uint32_t io_blocks = ((cs0 + 4095u) & ~4095u) / 512;
    size_t read_size = (size_t)io_blocks * 512;

    std::cout << "\n=== Page 0 ===\n"
              << "  compressed_size = " << cs0 << " bytes\n"
              << "  disk_offset     = " << offset0 << " bytes (LBA " << lba0 << ")\n"
              << "  read_size       = " << read_size << " bytes\n";

    std::vector<char> page_buf(read_size);
    int rc = bam_read_page(ctrl, read_size, lba0, page_buf.data());
    if (rc != 0) {
        std::cerr << "bam_read_page failed for compressed page 0" << std::endl;
        bam_ctrl_close(ctrl);
        return;
    }

    // === Method 1: per-thread LZ4 decompression ===
    std::cout << "\n--- Method 1: per-thread LZ4 ---\n";
    std::vector<char> decomp_lz4(page_size, 0);
    uint32_t lz4_size = bam_test_lz4_decompress_page(
        page_buf.data(), cs0, decomp_lz4.data(), (uint32_t)page_size);
    std::cout << "  decompressed_size = " << lz4_size << " bytes\n";

    if (lz4_size == 0) {
        std::cerr << "  ERROR: per-thread LZ4 decompression failed (returned 0)\n";
        bam_ctrl_close(ctrl);
        return;
    }

    // === Method 2: nvCOMPdx decompression ===
    std::cout << "\n--- Method 2: nvCOMPdx ---\n";
    // Set up staging buffer (GPU) with compressed data
    char* d_staging = nullptr;
    cudaMalloc(&d_staging, page_size);
    cudaMemcpy(d_staging, page_buf.data(), cs0, cudaMemcpyHostToDevice);

    // Set up comp_sizes on GPU (single element, page index 0)
    uint32_t* d_comp_sizes = nullptr;
    cudaMalloc(&d_comp_sizes, sizeof(uint32_t));
    cudaMemcpy(d_comp_sizes, &cs0, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Decompress buffer
    char* d_decomp = nullptr;
    cudaMalloc(&d_decomp, page_size);
    cudaMemset(d_decomp, 0, page_size);

    // Dummy accumulators (we only care about the decompressed buffer)
    uint64_t* d_dummy_records = nullptr;
    uint64_t* d_dummy_strlen = nullptr;
    uint64_t* d_dummy_bytesum = nullptr;
    cudaMalloc(&d_dummy_records, sizeof(uint64_t));
    cudaMalloc(&d_dummy_strlen, sizeof(uint64_t));
    cudaMalloc(&d_dummy_bytesum, sizeof(uint64_t));
    cudaMemset(d_dummy_records, 0, sizeof(uint64_t));
    cudaMemset(d_dummy_strlen, 0, sizeof(uint64_t));
    cudaMemset(d_dummy_bytesum, 0, sizeof(uint64_t));

    // Call nvCOMPdx decompress+scan (batch_start=0, batch_size=1)
    bam_vchar_decomp_scan_batch(
        d_staging, d_comp_sizes, d_decomp,
        d_dummy_records, d_dummy_strlen, d_dummy_bytesum,
        (uint32_t)page_size, /*batch_start=*/0, /*batch_size=*/1);

    // Download nvCOMPdx result
    std::vector<char> decomp_nvcomp(page_size, 0);
    cudaMemcpy(decomp_nvcomp.data(), d_decomp, page_size, cudaMemcpyDeviceToHost);

    // Read nvCOMPdx scan results for reference
    uint64_t ref_records = 0, ref_strlen = 0, ref_bytesum = 0;
    cudaMemcpy(&ref_records, d_dummy_records, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ref_strlen, d_dummy_strlen, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ref_bytesum, d_dummy_bytesum, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    std::cout << "  nvCOMPdx scan: records=" << ref_records
              << " strlen=" << ref_strlen
              << " bytesum=" << ref_bytesum << "\n";

    // === Byte-for-byte comparison ===
    std::cout << "\n--- Byte comparison ---\n";
    uint32_t cmp_bytes = std::min(lz4_size, (uint32_t)page_size);
    uint32_t mismatches = 0;
    uint32_t first_mismatch = UINT32_MAX;
    for (uint32_t i = 0; i < cmp_bytes; i++) {
        if (decomp_lz4[i] != decomp_nvcomp[i]) {
            if (mismatches < 10) {
                std::cout << "  MISMATCH at byte " << i << ": lz4=0x"
                          << std::hex << (unsigned)(uint8_t)decomp_lz4[i]
                          << " nvcomp=0x" << (unsigned)(uint8_t)decomp_nvcomp[i]
                          << std::dec << "\n";
            }
            if (first_mismatch == UINT32_MAX) first_mismatch = i;
            mismatches++;
        }
    }

    if (mismatches == 0) {
        std::cout << "  All " << cmp_bytes << " bytes match: OK\n";
    } else {
        std::cout << "  Total mismatches: " << mismatches << " / " << cmp_bytes
                  << " (first at byte " << first_mismatch << ")\n";
    }

    // === Parse VCHAR structure from per-thread LZ4 output ===
    std::cout << "\n--- VCHAR parse (per-thread LZ4 output) ---\n";
    uint32_t nalloc = *reinterpret_cast<uint32_t*>(decomp_lz4.data());
    std::cout << "  nalloc = " << nalloc << "\n";

    if (nalloc > 0 && nalloc < 100000) {
        uint64_t lz4_records = 0, lz4_strlen = 0, lz4_bytesum = 0;
        for (uint32_t slot = 0; slot < nalloc; slot++) {
            // oslt: slot table at end of page, growing downward
            uint32_t oslt = *reinterpret_cast<uint32_t*>(
                decomp_lz4.data() + page_size - sizeof(uint32_t) * (slot + 1));
            uint16_t vchar_len = *reinterpret_cast<uint16_t*>(
                decomp_lz4.data() + oslt);
            const uint8_t* vchar_data = reinterpret_cast<const uint8_t*>(
                decomp_lz4.data() + oslt + sizeof(uint32_t));

            lz4_records++;
            lz4_strlen += vchar_len;
            for (uint16_t b = 0; b < vchar_len; b++) {
                lz4_bytesum += vchar_data[b];
            }
        }
        std::cout << "  records  = " << lz4_records << "\n"
                  << "  strlen   = " << lz4_strlen << "\n"
                  << "  bytesum  = " << lz4_bytesum << "\n";

        if (lz4_records == ref_records && lz4_strlen == ref_strlen
            && lz4_bytesum == ref_bytesum) {
            std::cout << "  Matches nvCOMPdx scan results: OK\n";
        } else {
            std::cout << "  WARNING: does NOT match nvCOMPdx scan results!\n";
        }
    }

    // Cleanup
    cudaFree(d_staging);
    cudaFree(d_comp_sizes);
    cudaFree(d_decomp);
    cudaFree(d_dummy_records);
    cudaFree(d_dummy_strlen);
    cudaFree(d_dummy_bytesum);

    std::cout << "\ntest_lz4_page: done\n";
    bam_ctrl_close(ctrl);
}

// ============================================================
// TPC-H Q3 (PiG) — GPU-initiated I/O via BaM
// ============================================================

PigResult tpch_q3(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const bool is_q3sel = (options.q3sel_selectivity > 0);
    const int sel_pct = is_q3sel ? options.q3sel_selectivity : 20;

    if (is_q3sel)
        std::cout << "=== TPC-H Q3SEL (PiG, sel=" << sel_pct << "%) ===" << std::endl;
    else
        std::cout << "=== TPC-H Q3 (PiG) ===" << std::endl;

    size_t gpu_free_pre_ctrl = 0, gpu_total_pre = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_pre);

    const uint32_t bam_num_queues = 128;
    auto ds = bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;
    const uint32_t n_devices = ds.n_devices;
    const uint64_t partition_start_lba = ds.partition_start_lba;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_pre);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;

    // Helper: read a page given global page ID (handles striping)
    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    read_striped_page(0, page_size, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;

    // ── Field metadata ──
    constexpr size_t col_c_custkey    = TPCH::common::C_CUSTKEY;
    constexpr size_t col_c_mktsegment = TPCH::common::C_MKTSEGMENT;
    constexpr size_t col_o_orderkey   = TPCH::common::O_ORDERKEY;
    constexpr size_t col_o_custkey    = TPCH::common::O_CUSTKEY;
    constexpr size_t col_o_orderdate  = TPCH::common::O_ORDERDATE;
    constexpr size_t col_o_shippriority = TPCH::common::O_SHIPPRIORITY;
    constexpr size_t col_l_orderkey   = TPCH::common::L_ORDERKEY;
    constexpr size_t col_l_extprice   = TPCH::common::L_EXTENDEDPRICE;
    constexpr size_t col_l_discount   = TPCH::common::L_DISCOUNT;
    constexpr size_t col_l_shipdate   = TPCH::common::L_SHIPDATE;

    // CUSTOMER fields
    uint64_t c_custkey_start  = metadata.table_customer_start_page_ids[col_c_custkey];
    uint64_t c_custkey_npages = metadata.table_customer_npages[col_c_custkey];
    uint16_t c_custkey_comp   = metadata.table_customer_compression_method[col_c_custkey];

    uint64_t c_mktseg_start  = metadata.table_customer_start_page_ids[col_c_mktsegment];
    uint64_t c_mktseg_npages = metadata.table_customer_npages[col_c_mktsegment];
    uint16_t c_mktseg_comp   = metadata.table_customer_compression_method[col_c_mktsegment];

    // ORDERS fields
    uint64_t o_orderkey_start  = metadata.table_orders_start_page_ids[col_o_orderkey];
    uint64_t o_orderkey_npages = metadata.table_orders_npages[col_o_orderkey];
    uint16_t o_orderkey_comp   = metadata.table_orders_compression_method[col_o_orderkey];

    uint64_t o_custkey_start  = metadata.table_orders_start_page_ids[col_o_custkey];
    uint64_t o_custkey_npages = metadata.table_orders_npages[col_o_custkey];
    uint16_t o_custkey_comp   = metadata.table_orders_compression_method[col_o_custkey];

    uint64_t o_orderdate_start  = metadata.table_orders_start_page_ids[col_o_orderdate];
    uint64_t o_orderdate_npages = metadata.table_orders_npages[col_o_orderdate];
    uint16_t o_orderdate_comp   = metadata.table_orders_compression_method[col_o_orderdate];

    uint64_t o_shippri_start  = metadata.table_orders_start_page_ids[col_o_shippriority];
    uint64_t o_shippri_npages = metadata.table_orders_npages[col_o_shippriority];
    uint16_t o_shippri_comp   = metadata.table_orders_compression_method[col_o_shippriority];

    // LINEITEM fields
    uint64_t l_orderkey_start  = metadata.table_lineitem_start_page_ids[col_l_orderkey];
    uint64_t l_orderkey_npages = metadata.table_lineitem_npages[col_l_orderkey];
    uint16_t l_orderkey_comp   = metadata.table_lineitem_compression_method[col_l_orderkey];

    uint64_t l_extprice_start  = metadata.table_lineitem_start_page_ids[col_l_extprice];
    uint64_t l_extprice_npages = metadata.table_lineitem_npages[col_l_extprice];
    uint16_t l_extprice_comp   = metadata.table_lineitem_compression_method[col_l_extprice];

    uint64_t l_discount_start  = metadata.table_lineitem_start_page_ids[col_l_discount];
    uint64_t l_discount_npages = metadata.table_lineitem_npages[col_l_discount];
    uint16_t l_discount_comp   = metadata.table_lineitem_compression_method[col_l_discount];

    uint64_t l_shipdate_start  = metadata.table_lineitem_start_page_ids[col_l_shipdate];
    uint64_t l_shipdate_npages = metadata.table_lineitem_npages[col_l_shipdate];
    uint16_t l_shipdate_comp   = metadata.table_lineitem_compression_method[col_l_shipdate];

    uint64_t nrecs_customer = metadata.table_customer_nrows;
    uint64_t nrecs_orders   = metadata.table_orders_nrows;
    uint64_t nrecs_lineitem = metadata.table_lineitem_nrows;

    std::cout << "nrecs: customer=" << nrecs_customer
              << ", orders=" << nrecs_orders
              << ", lineitem=" << nrecs_lineitem << std::endl;

    // ================================================================
    // Host-side metadata reads (before total_start)
    // ================================================================

    auto read_prefix_sum_host = [&](uint64_t ps_start, uint64_t ps_npages,
                                     uint64_t field_npages) -> std::vector<uint64_t>
    {
        if (ps_npages == 0) return {};
        std::vector<char> h_buf(ps_npages * page_size);
        for (uint64_t p = 0; p < ps_npages; p++) {
            read_striped_page(ps_start + p, page_size, h_buf.data() + p * page_size);
        }
        uint64_t* ps_raw = reinterpret_cast<uint64_t*>(h_buf.data()) + 1;
        return std::vector<uint64_t>(ps_raw, ps_raw + field_npages);
    };

    auto prepare_comp_metadata = [&](
        uint64_t field_start, uint64_t field_npages, uint16_t comp_method,
        uint64_t cs_start_page, uint64_t cs_npages_cnt,
        uint64_t nbase_val, uint64_t base_start_page)
        -> std::pair<std::vector<uint32_t>, std::vector<uint64_t>>
    {
        if (comp_method == 0) return {{}, {}};
        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            read_striped_page(cs_start_page + p, page_size, sizes_buf.data() + p * page_size);
        }
        std::vector<uint32_t> comp_sizes(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages);
        size_t bp_npages = TPCH::nbase_to_npages(nbase_val, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            read_striped_page(base_start_page + p, page_size, bases_buf.data() + p * page_size);
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            comp_sizes.data(), nbase_val, field_npages, page_size, field_start,
            n_devices, offsets_vec);
        std::vector<uint64_t> comp_offsets(field_npages);
        for (uint64_t i = 0; i < field_npages; i++)
            comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
        return {comp_sizes, comp_offsets};
    };

    std::cout << "[Q3] Reading metadata..." << std::endl;

    // CUSTOMER metadata
    auto h_ps_c_custkey = read_prefix_sum_host(
        metadata.table_customer_prefix_sum_start_page_ids[col_c_custkey],
        metadata.table_customer_prefix_sum_npages[col_c_custkey],
        c_custkey_npages);
    auto [h_cs_c_custkey, h_co_c_custkey] = prepare_comp_metadata(
        c_custkey_start, c_custkey_npages, c_custkey_comp,
        metadata.table_customer_compressed_page_sizes_start_page_ids[col_c_custkey],
        metadata.table_customer_compressed_page_sizes_npages[col_c_custkey],
        metadata.table_customer_compression_nbases[col_c_custkey],
        metadata.table_customer_compression_base_start_page_ids[col_c_custkey]);

    // C_MKTSEGMENT metadata (lz4par — same comp_metadata structure)
    auto h_ps_c_mktseg = read_prefix_sum_host(
        metadata.table_customer_prefix_sum_start_page_ids[col_c_mktsegment],
        metadata.table_customer_prefix_sum_npages[col_c_mktsegment],
        c_mktseg_npages);
    auto [h_cs_c_mktseg, h_co_c_mktseg] = prepare_comp_metadata(
        c_mktseg_start, c_mktseg_npages, c_mktseg_comp,
        metadata.table_customer_compressed_page_sizes_start_page_ids[col_c_mktsegment],
        metadata.table_customer_compressed_page_sizes_npages[col_c_mktsegment],
        metadata.table_customer_compression_nbases[col_c_mktsegment],
        metadata.table_customer_compression_base_start_page_ids[col_c_mktsegment]);

    // ORDERS metadata
    auto h_ps_o_orderkey = read_prefix_sum_host(
        metadata.table_orders_prefix_sum_start_page_ids[col_o_orderkey],
        metadata.table_orders_prefix_sum_npages[col_o_orderkey],
        o_orderkey_npages);
    auto [h_cs_o_orderkey, h_co_o_orderkey] = prepare_comp_metadata(
        o_orderkey_start, o_orderkey_npages, o_orderkey_comp,
        metadata.table_orders_compressed_page_sizes_start_page_ids[col_o_orderkey],
        metadata.table_orders_compressed_page_sizes_npages[col_o_orderkey],
        metadata.table_orders_compression_nbases[col_o_orderkey],
        metadata.table_orders_compression_base_start_page_ids[col_o_orderkey]);

    auto h_ps_o_custkey = read_prefix_sum_host(
        metadata.table_orders_prefix_sum_start_page_ids[col_o_custkey],
        metadata.table_orders_prefix_sum_npages[col_o_custkey],
        o_custkey_npages);
    auto [h_cs_o_custkey, h_co_o_custkey] = prepare_comp_metadata(
        o_custkey_start, o_custkey_npages, o_custkey_comp,
        metadata.table_orders_compressed_page_sizes_start_page_ids[col_o_custkey],
        metadata.table_orders_compressed_page_sizes_npages[col_o_custkey],
        metadata.table_orders_compression_nbases[col_o_custkey],
        metadata.table_orders_compression_base_start_page_ids[col_o_custkey]);

    auto h_ps_o_orderdate = read_prefix_sum_host(
        metadata.table_orders_prefix_sum_start_page_ids[col_o_orderdate],
        metadata.table_orders_prefix_sum_npages[col_o_orderdate],
        o_orderdate_npages);
    auto [h_cs_o_orderdate, h_co_o_orderdate] = prepare_comp_metadata(
        o_orderdate_start, o_orderdate_npages, o_orderdate_comp,
        metadata.table_orders_compressed_page_sizes_start_page_ids[col_o_orderdate],
        metadata.table_orders_compressed_page_sizes_npages[col_o_orderdate],
        metadata.table_orders_compression_nbases[col_o_orderdate],
        metadata.table_orders_compression_base_start_page_ids[col_o_orderdate]);

    auto h_ps_o_shippri = read_prefix_sum_host(
        metadata.table_orders_prefix_sum_start_page_ids[col_o_shippriority],
        metadata.table_orders_prefix_sum_npages[col_o_shippriority],
        o_shippri_npages);
    auto [h_cs_o_shippri, h_co_o_shippri] = prepare_comp_metadata(
        o_shippri_start, o_shippri_npages, o_shippri_comp,
        metadata.table_orders_compressed_page_sizes_start_page_ids[col_o_shippriority],
        metadata.table_orders_compressed_page_sizes_npages[col_o_shippriority],
        metadata.table_orders_compression_nbases[col_o_shippriority],
        metadata.table_orders_compression_base_start_page_ids[col_o_shippriority]);

    // LINEITEM metadata
    auto h_ps_l_orderkey = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_orderkey],
        metadata.table_lineitem_prefix_sum_npages[col_l_orderkey],
        l_orderkey_npages);
    auto [h_cs_l_orderkey, h_co_l_orderkey] = prepare_comp_metadata(
        l_orderkey_start, l_orderkey_npages, l_orderkey_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_orderkey],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_orderkey],
        metadata.table_lineitem_compression_nbases[col_l_orderkey],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_orderkey]);

    auto h_ps_l_extprice = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_extprice],
        metadata.table_lineitem_prefix_sum_npages[col_l_extprice],
        l_extprice_npages);
    auto [h_cs_l_extprice, h_co_l_extprice] = prepare_comp_metadata(
        l_extprice_start, l_extprice_npages, l_extprice_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_extprice],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_extprice],
        metadata.table_lineitem_compression_nbases[col_l_extprice],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_extprice]);

    auto h_ps_l_discount = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_discount],
        metadata.table_lineitem_prefix_sum_npages[col_l_discount],
        l_discount_npages);
    auto [h_cs_l_discount, h_co_l_discount] = prepare_comp_metadata(
        l_discount_start, l_discount_npages, l_discount_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_discount],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_discount],
        metadata.table_lineitem_compression_nbases[col_l_discount],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_discount]);

    auto h_ps_l_shipdate = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_shipdate],
        metadata.table_lineitem_prefix_sum_npages[col_l_shipdate],
        l_shipdate_npages);
    auto [h_cs_l_shipdate, h_co_l_shipdate] = prepare_comp_metadata(
        l_shipdate_start, l_shipdate_npages, l_shipdate_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_shipdate],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_shipdate],
        metadata.table_lineitem_compression_nbases[col_l_shipdate],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_shipdate]);

    // Verify same-type columns share npages
    const uint64_t o_npages_i32 = o_orderdate_npages;
    if (o_shippri_npages != o_npages_i32) {
        std::cerr << "ERROR: ORDERS INT32 npages mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }
    const uint64_t o_npages_i64 = o_orderkey_npages;
    if (o_custkey_npages != o_npages_i64) {
        std::cerr << "ERROR: ORDERS INT64 npages mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }
    const uint64_t l_npages_i32 = l_shipdate_npages;
    if (l_extprice_npages != l_npages_i32 || l_discount_npages != l_npages_i32) {
        std::cerr << "ERROR: LINEITEM INT32 npages mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }
    const uint64_t l_npages_i64 = l_orderkey_npages;

    // Diagnostic: check if INT64 and INT32 prefix_sums differ
    {
        bool o_ps_match = (o_npages_i32 == o_npages_i64);
        if (o_ps_match) {
            for (size_t i = 0; i < o_npages_i32 && o_ps_match; i++)
                if (h_ps_o_orderdate[i] != h_ps_o_orderkey[i]) o_ps_match = false;
        }
        bool l_ps_match = (l_npages_i32 == l_npages_i64);
        if (l_ps_match) {
            for (size_t i = 0; i < l_npages_i32 && l_ps_match; i++)
                if (h_ps_l_shipdate[i] != h_ps_l_orderkey[i]) l_ps_match = false;
        }
        if (!o_ps_match || !l_ps_match) {
            std::cout << "[Q3] NOTE: INT64/INT32 prefix_sum mismatch — using full INT64 flatten" << std::endl;
            if (!o_ps_match)
                std::cout << "[Q3]   ORDERS: o_npages_i32=" << o_npages_i32
                          << " o_npages_i64=" << o_npages_i64
                          << " h_ps_odate[0]=" << h_ps_o_orderdate[0]
                          << " h_ps_okey[0]=" << h_ps_o_orderkey[0] << std::endl;
            if (!l_ps_match)
                std::cout << "[Q3]   LINEITEM: l_npages_i32=" << l_npages_i32
                          << " l_npages_i64=" << l_npages_i64
                          << " h_ps_lsd[0]=" << h_ps_l_shipdate[0]
                          << " h_ps_lokey[0]=" << h_ps_l_orderkey[0] << std::endl;
        } else {
            std::cout << "[Q3] INT64/INT32 prefix_sums match" << std::endl;
        }
    }

    // ================================================================
    // Zone map metadata extraction (before total_start, only if -Z)
    // ================================================================

    bool zonemap_enabled = options.enable_zonemap;

    // Q3SEL: mktseg dict_id range for zone map pruning
    // AUTOMOBILE=0, BUILDING=1, FURNITURE=2, MACHINERY=3, HOUSEHOLD=4
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

    // GPU-side zone map setup (metadata extraction outside timing)
    bam_pfor32_io_ctx_t pfor_ctx_zm = nullptr;
    BamZonemapCtx zm_ord_ctx{}, zm_li_ctx{};
    uint32_t zm_ord_nreads = 0, zm_ord_npreds = 0;
    uint32_t zm_li_nreads = 0, zm_li_npreds = 0;
    bool zm_ord_valid = false, zm_li_valid = false;

    // Prefix sums for INT32->INT64 page mapping (read outside timing)
    if (zonemap_enabled) {
        std::cout << "[Q3-ZONEMAP] Setting up GPU-side zone map..." << std::endl;

        pfor_ctx_zm = bam_pfor32_io_create(ctrl, static_cast<uint32_t>(page_size), kBamZonemapMaxReads);

        // ORDERS zone map
        zm_ord_ctx = bam_zonemap_ctx_create(
            bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
            bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
            (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
            static_cast<uint32_t>(page_size), o_orderdate_npages);

        {
            const size_t o_odate_field = TPCH::common::O_ORDERDATE;

            // O_ORDERDATE direct stats (Q3, and Q3SEL when filters enabled)
            if (!is_q3sel || !options.disable_other_filters) {
                uint64_t odate_nstats      = metadata.table_orders_nstats[o_odate_field];
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
            if (!(is_q3sel && sel_pct >= 100) && mktseg_nstats > 0 && mktseg_stats_start > 0) {
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
        zm_li_ctx = bam_zonemap_ctx_create(
            bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
            bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
            (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
            static_cast<uint32_t>(page_size), l_shipdate_npages);

        {
            const size_t l_sd_field = TPCH::common::L_SHIPDATE;

            // L_SHIPDATE direct stats (Q3, and Q3SEL when filters enabled)
            if (!is_q3sel || !options.disable_other_filters) {
                uint64_t lsd_nstats      = metadata.table_lineitem_nstats[l_sd_field];
                uint64_t lsd_stats_start = metadata.table_lineitem_stats_start_page_ids[l_sd_field];
                uint64_t lsd_stats_npg   = metadata.table_lineitem_stats_npages[l_sd_field];
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
            const size_t sw_mktseg_li_idx = TPCH::common::LS_SIDEWAYS_C_MKTSEGMENT;
            uint64_t sw_mktseg_li_nstats      = metadata.table_lineitem_sideways_nstats[l_sd_field][sw_mktseg_li_idx];
            uint64_t sw_mktseg_li_stats_start = metadata.table_lineitem_sideways_stats_start_page_ids[l_sd_field][sw_mktseg_li_idx];
            uint64_t sw_mktseg_li_stats_npg   = metadata.table_lineitem_sideways_stats_npages[l_sd_field][sw_mktseg_li_idx];
            if (!(is_q3sel && sel_pct >= 100) && sw_mktseg_li_nstats > 0 && sw_mktseg_li_stats_start > 0) {
                uint32_t stats_page_offset = zm_li_nreads;
                for (uint64_t j = 0; j < sw_mktseg_li_stats_npg; j++) {
                    uint64_t pg_id = sw_mktseg_li_stats_start + j;
                    uint32_t dev = pg_id % n_devices;
                    uint64_t local = pg_id / n_devices;
                    zm_li_ctx.h_reads[zm_li_nreads++] = {
                        ds.partition_start_lbas[dev] + local * blocks_per_page,
                        static_cast<uint32_t>(blocks_per_page), dev};
                }
                if (is_q3sel)
                    zm_li_ctx.h_preds[zm_li_npreds++] = {stats_page_offset, sw_mktseg_li_nstats,
                        q3sel_mktseg_lo, q3sel_mktseg_hi};
                else
                    zm_li_ctx.h_preds[zm_li_npreds++] = {stats_page_offset, sw_mktseg_li_nstats,
                        1, 1};
            }
        }
        zm_li_valid = (zm_li_npreds > 0);

    }

    // ================================================================
    // GPU memory allocation (before total_start)
    // ================================================================

    size_t gpu_free_before = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free_before, &gpu_total);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint32_t num_blocks_pfor = static_cast<uint32_t>(sm_count);
    // MKTSEG fused (LZ4PAR CHAR): cap to actual page count
    uint32_t num_blocks_mktseg = std::min(
        static_cast<uint32_t>(sm_count) * 4,
        static_cast<uint32_t>(c_mktseg_npages));

    // ── GPU metadata arrays ──
    struct GpuFieldMeta {
        uint64_t* d_prefix_sum;
        uint32_t* d_comp_sizes;
        uint64_t* d_comp_offsets;
    };
    auto alloc_gpu_field_meta = [](uint64_t npages, uint16_t comp) -> GpuFieldMeta {
        GpuFieldMeta m{};
        cudaMalloc(&m.d_prefix_sum, npages * sizeof(uint64_t));
        if (comp != 0) {
            cudaMalloc(&m.d_comp_sizes, npages * sizeof(uint32_t));
            cudaMalloc(&m.d_comp_offsets, npages * sizeof(uint64_t));
        }
        return m;
    };

    // GPU metadata: CUSTOMER full-sized (for flatten)
    auto gm_c_custkey   = alloc_gpu_field_meta(c_custkey_npages, c_custkey_comp);
    auto gm_c_mktseg    = alloc_gpu_field_meta(c_mktseg_npages, c_mktseg_comp);
    // ── Fused kernel setup ──
    // Compute max rows per INT32 page (for scratch buffer sizing)
    uint64_t o_max_rows_per_page = 0;
    for (uint64_t i = 0; i < o_npages_i32; i++) {
        uint64_t nr = (i == 0) ? h_ps_o_orderdate[0] : h_ps_o_orderdate[i] - h_ps_o_orderdate[i - 1];
        o_max_rows_per_page = std::max(o_max_rows_per_page, nr);
    }
    uint64_t l_max_rows_per_page = 0;
    for (uint64_t i = 0; i < l_npages_i32; i++) {
        uint64_t nr = (i == 0) ? h_ps_l_shipdate[0] : h_ps_l_shipdate[i] - h_ps_l_shipdate[i - 1];
        l_max_rows_per_page = std::max(l_max_rows_per_page, nr);
    }
    uint64_t max_rows_per_page = std::max(o_max_rows_per_page, l_max_rows_per_page);

    // Compute max rows per INT64 page (for scratch_i64 sizing)
    uint64_t o_max_rows_per_i64_page = 0;
    for (uint64_t i = 0; i < o_npages_i64; i++) {
        uint64_t nr = (i == 0) ? h_ps_o_orderkey[0] : h_ps_o_orderkey[i] - h_ps_o_orderkey[i - 1];
        o_max_rows_per_i64_page = std::max(o_max_rows_per_i64_page, nr);
    }
    uint64_t l_max_rows_per_i64_page = 0;
    for (uint64_t i = 0; i < l_npages_i64; i++) {
        uint64_t nr = (i == 0) ? h_ps_l_orderkey[0] : h_ps_l_orderkey[i] - h_ps_l_orderkey[i - 1];
        l_max_rows_per_i64_page = std::max(l_max_rows_per_i64_page, nr);
    }
    uint64_t max_rows_per_i64_page = std::max(o_max_rows_per_i64_page, l_max_rows_per_i64_page);

    // ORDERS fused: 6 slots/block (2 INT32 + 2 INT64 × 2 pages)
    const uint32_t fused_block_mult = is_q3sel ? 4 : 1;
    const uint32_t num_blocks_orders_fused = std::min(
        static_cast<uint32_t>(sm_count) * fused_block_mult,
        static_cast<uint32_t>(o_npages_i32));
    // LINEITEM fused: 5 slots/block (3 INT32 + 1 INT64 × 2 pages)
    const uint32_t num_blocks_lineitem_fused = std::min(
        static_cast<uint32_t>(sm_count) * fused_block_mult,
        static_cast<uint32_t>(l_npages_i32));
    // Page cache: max(orders_slots, lineitem_slots)
    const uint32_t fused_pc_pages = std::max(num_blocks_orders_fused * 8,
                                              num_blocks_lineitem_fused * 6);

    std::cout << "[Q3] Fused kernel: ORDERS " << num_blocks_orders_fused << " blocks, "
              << o_npages_i32 << " pages; LINEITEM " << num_blocks_lineitem_fused << " blocks, "
              << l_npages_i32 << " pages; max_rows_per_page=" << max_rows_per_page
              << " max_rows_per_i64_page=" << max_rows_per_i64_page << std::endl;

    // Flat output arrays (only CUSTOMER — ORDERS/LINEITEM use fused BaM)
    uint64_t* d_c_custkey_flat = nullptr;
    cudaMalloc(&d_c_custkey_flat, nrecs_customer * sizeof(uint64_t));

    // Full-column GPU metadata for fused INT32 fields
    // ORDERS: O_ORDERDATE + O_SHIPPRIORITY (shared prefix sum)
    uint64_t* d_ps_o_orderdate_q3 = nullptr;
    uint32_t* d_cs_o_orderdate_q3 = nullptr;
    uint64_t* d_co_o_orderdate_q3 = nullptr;
    uint32_t* d_cs_o_shippri_q3 = nullptr;
    uint64_t* d_co_o_shippri_q3 = nullptr;
    cudaMalloc(&d_ps_o_orderdate_q3, o_npages_i32 * sizeof(uint64_t));
    cudaMemcpy(d_ps_o_orderdate_q3, h_ps_o_orderdate.data(),
               o_npages_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (o_orderdate_comp != 0) {
        cudaMalloc(&d_cs_o_orderdate_q3, o_npages_i32 * sizeof(uint32_t));
        cudaMemcpy(d_cs_o_orderdate_q3, h_cs_o_orderdate.data(),
                   o_npages_i32 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_o_orderdate_q3, o_npages_i32 * sizeof(uint64_t));
        cudaMemcpy(d_co_o_orderdate_q3, h_co_o_orderdate.data(),
                   o_npages_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    if (o_shippri_comp != 0) {
        cudaMalloc(&d_cs_o_shippri_q3, o_npages_i32 * sizeof(uint32_t));
        cudaMemcpy(d_cs_o_shippri_q3, h_cs_o_shippri.data(),
                   o_npages_i32 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_o_shippri_q3, o_npages_i32 * sizeof(uint64_t));
        cudaMemcpy(d_co_o_shippri_q3, h_co_o_shippri.data(),
                   o_npages_i32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    // LINEITEM: L_SHIPDATE + L_EXTPRICE + L_DISCOUNT fused kernel metadata
    uint64_t* d_ps_l_shipdate = nullptr;
    uint32_t* d_cs_l_shipdate_q3 = nullptr;
    uint64_t* d_co_l_shipdate_q3 = nullptr;
    uint32_t* d_cs_l_extprice_q3 = nullptr;
    uint64_t* d_co_l_extprice_q3 = nullptr;
    uint32_t* d_cs_l_discount_q3 = nullptr;
    uint64_t* d_co_l_discount_q3 = nullptr;
    cudaMalloc(&d_ps_l_shipdate, l_shipdate_npages * sizeof(uint64_t));
    cudaMemcpy(d_ps_l_shipdate, h_ps_l_shipdate.data(),
               l_shipdate_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (l_shipdate_comp != 0) {
        cudaMalloc(&d_cs_l_shipdate_q3, l_shipdate_npages * sizeof(uint32_t));
        cudaMemcpy(d_cs_l_shipdate_q3, h_cs_l_shipdate.data(),
                   l_shipdate_npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_l_shipdate_q3, l_shipdate_npages * sizeof(uint64_t));
        cudaMemcpy(d_co_l_shipdate_q3, h_co_l_shipdate.data(),
                   l_shipdate_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    if (l_extprice_comp != 0) {
        cudaMalloc(&d_cs_l_extprice_q3, l_extprice_npages * sizeof(uint32_t));
        cudaMemcpy(d_cs_l_extprice_q3, h_cs_l_extprice.data(),
                   l_extprice_npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_l_extprice_q3, l_extprice_npages * sizeof(uint64_t));
        cudaMemcpy(d_co_l_extprice_q3, h_co_l_extprice.data(),
                   l_extprice_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    if (l_discount_comp != 0) {
        cudaMalloc(&d_cs_l_discount_q3, l_discount_npages * sizeof(uint32_t));
        cudaMemcpy(d_cs_l_discount_q3, h_cs_l_discount.data(),
                   l_discount_npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_l_discount_q3, l_discount_npages * sizeof(uint64_t));
        cudaMemcpy(d_co_l_discount_q3, h_co_l_discount.data(),
                   l_discount_npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // ORDERS INT64 GPU metadata: O_ORDERKEY, O_CUSTKEY
    uint64_t* d_ps_o_i64_q3 = nullptr;
    uint32_t* d_cs_o_orderkey_q3 = nullptr;
    uint64_t* d_co_o_orderkey_q3 = nullptr;
    uint32_t* d_cs_o_custkey_q3 = nullptr;
    uint64_t* d_co_o_custkey_q3 = nullptr;
    cudaMalloc(&d_ps_o_i64_q3, o_npages_i64 * sizeof(uint64_t));
    cudaMemcpy(d_ps_o_i64_q3, h_ps_o_orderkey.data(),
               o_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (o_orderkey_comp != 0) {
        cudaMalloc(&d_cs_o_orderkey_q3, o_npages_i64 * sizeof(uint32_t));
        cudaMemcpy(d_cs_o_orderkey_q3, h_cs_o_orderkey.data(),
                   o_npages_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_o_orderkey_q3, o_npages_i64 * sizeof(uint64_t));
        cudaMemcpy(d_co_o_orderkey_q3, h_co_o_orderkey.data(),
                   o_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    if (o_custkey_comp != 0) {
        cudaMalloc(&d_cs_o_custkey_q3, o_npages_i64 * sizeof(uint32_t));
        cudaMemcpy(d_cs_o_custkey_q3, h_cs_o_custkey.data(),
                   o_npages_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_o_custkey_q3, o_npages_i64 * sizeof(uint64_t));
        cudaMemcpy(d_co_o_custkey_q3, h_co_o_custkey.data(),
                   o_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // LINEITEM INT64 GPU metadata: L_ORDERKEY
    uint64_t* d_ps_l_i64_q3 = nullptr;
    uint32_t* d_cs_l_orderkey_q3 = nullptr;
    uint64_t* d_co_l_orderkey_q3 = nullptr;
    cudaMalloc(&d_ps_l_i64_q3, l_npages_i64 * sizeof(uint64_t));
    cudaMemcpy(d_ps_l_i64_q3, h_ps_l_orderkey.data(),
               l_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (l_orderkey_comp != 0) {
        cudaMalloc(&d_cs_l_orderkey_q3, l_npages_i64 * sizeof(uint32_t));
        cudaMemcpy(d_cs_l_orderkey_q3, h_cs_l_orderkey.data(),
                   l_npages_i64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_co_l_orderkey_q3, l_npages_i64 * sizeof(uint64_t));
        cudaMemcpy(d_co_l_orderkey_q3, h_co_l_orderkey.data(),
                   l_npages_i64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // Per-block INT32 scratch buffer for fused kernels
    // ORDERS: num_blocks * 2 * max_rows_per_page
    // LINEITEM: num_blocks * 3 * max_rows_per_page
    uint64_t scratch_elems = std::max(
        (uint64_t)num_blocks_orders_fused * 2 * max_rows_per_page,
        (uint64_t)num_blocks_lineitem_fused * 3 * max_rows_per_page);
    int32_t* d_scratch = nullptr;
    cudaMalloc(&d_scratch, scratch_elems * sizeof(int32_t));

    // Per-block INT64 scratch buffer
    // ORDERS: num_blocks * 2 fields * 3 pages * max_rows_per_i64_page
    // LINEITEM: num_blocks * 1 field * 3 pages * max_rows_per_i64_page
    uint64_t scratch_i64_elems = std::max(
        (uint64_t)num_blocks_orders_fused * 2 * 3 * max_rows_per_i64_page,
        (uint64_t)num_blocks_lineitem_fused * 1 * 3 * max_rows_per_i64_page);
    int64_t* d_scratch_i64 = nullptr;
    cudaMalloc(&d_scratch_i64, scratch_i64_elems * sizeof(int64_t));

    // BaM IO contexts (allocated before HT so budget measurement is accurate)
    bam_pfor64_io_ctx_t pfor64_ctx = bam_pfor64_io_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks_pfor);
    const bool is_fsst_mktseg = (c_mktseg_comp == static_cast<uint16_t>(CompressionMethod::FSST)
                                || c_mktseg_comp == static_cast<uint16_t>(CompressionMethod::FSST_ROWID));

    // LZ4PAR fused context (only when not FSST)
    bam_q3_fused_io_ctx_t fused_ctx = nullptr;
    if (!is_fsst_mktseg) {
        fused_ctx = bam_q3_fused_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks_mktseg);
    }
    // FSST fused context (only when FSST)
    bam_q3_fsst_io_ctx_t fsst_ctx = nullptr;
    if (is_fsst_mktseg) {
        fsst_ctx = bam_q3_fsst_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks_mktseg);
    }
    bam_pfor32_io_ctx_t pfor32_fused_ctx = bam_pfor32_io_create(
        ctrl, static_cast<uint32_t>(page_size), fused_pc_pages);

    // Hash tables
    uint64_t est_building = is_q3sel
        ? std::max((uint64_t)1024, nrecs_customer * sel_pct / 100)
        : std::max((uint64_t)1024, nrecs_customer / 4);
    uint32_t custset_cap = 1;
    while (custset_cap < est_building * 2) custset_cap <<= 1;
    uint32_t custset_mask = custset_cap - 1;
    uint64_t* d_custkey_set = nullptr;
    cudaMalloc(&d_custkey_set, custset_cap * sizeof(uint64_t));

    uint64_t est_orders_qual = is_q3sel
        ? std::max((uint64_t)1024, nrecs_orders * sel_pct / 100)
        : std::max((uint64_t)1024, nrecs_orders / 8);
    uint32_t orders_ht_cap = 1;
    while (orders_ht_cap < est_orders_qual * 2) orders_ht_cap <<= 1;

    if (is_q3sel) {
        size_t gpu_free_now = 0;
        cudaMemGetInfo(&gpu_free_now, &gpu_total);
        uint64_t app_used = gpu_free_before - gpu_free_now;
        uint64_t total_budget = (gpu_ctrl_bytes < GPU_MEM_BUDGET)
            ? GPU_MEM_BUDGET - gpu_ctrl_bytes : 0;
        if (app_used < total_budget) {
            uint64_t remaining = total_budget - app_used;
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
    uint64_t* d_orders_ht_keys = nullptr;
    uint64_t* d_orders_ht_payloads = nullptr;
    cudaMalloc(&d_orders_ht_keys, orders_ht_cap * sizeof(uint64_t));
    cudaMalloc(&d_orders_ht_payloads, orders_ht_cap * sizeof(uint64_t));

    // Aggregation hash map
    uint32_t aggr_cap = orders_ht_cap;
    uint32_t aggr_mask = aggr_cap - 1;
    uint64_t* d_aggr_keys = nullptr;
    int64_t*  d_aggr_revenues = nullptr;
    cudaMalloc(&d_aggr_keys, aggr_cap * sizeof(uint64_t));
    cudaMalloc(&d_aggr_revenues, aggr_cap * sizeof(int64_t));

    // Result arrays
    Q3ResultRow* d_results = nullptr;
    uint32_t* d_result_count = nullptr;
    cudaMalloc(&d_results, aggr_cap * sizeof(Q3ResultRow));
    cudaMalloc(&d_result_count, sizeof(uint32_t));

    // Pre-allocate CUB DeviceMergeSort temp buffer
    void *d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceMergeSort::SortKeys(nullptr, sort_temp_bytes,
        d_results, (int)aggr_cap, Q3ResultCmp{}, stream);
    cudaMalloc(&d_sort_temp, sort_temp_bytes);

    std::cout << "[Q3] sm_count=" << sm_count
              << " num_blocks_pfor=" << num_blocks_pfor
              << " num_blocks_mktseg=" << num_blocks_mktseg
              << " fused_pc_pages=" << fused_pc_pages << std::endl;

    // ── Helper lambdas ──
    auto upload_field_meta = [](const GpuFieldMeta& gm,
        const std::vector<uint64_t>& h_ps,
        const std::vector<uint32_t>& h_cs,
        const std::vector<uint64_t>& h_co,
        uint64_t npages, uint16_t comp)
    {
        cudaMemcpy(gm.d_prefix_sum, h_ps.data(),
                   npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (comp != 0) {
            cudaMemcpy(gm.d_comp_sizes, h_cs.data(),
                       npages * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(gm.d_comp_offsets, h_co.data(),
                       npages * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
    };

    auto make_pfor64_params = [&](uint64_t field_start, uint64_t npages,
        uint16_t comp, uint64_t nrows, const GpuFieldMeta& gm) -> BAMPfor64FlattenParams
    {
        BAMPfor64FlattenParams p{};
        p.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            p.partition_start_lbas[d] = ds.partition_start_lbas[d];
        p.n_devices = n_devices;
        p.page_size = static_cast<uint32_t>(page_size);
        p.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        p.comp_method = comp;
        p.field_start_page_id = field_start;
        p.npages = npages;
        p.nrows = nrows;
        p.num_blocks = num_blocks_pfor;
        p.d_prefix_sum = gm.d_prefix_sum;
        p.d_comp_sizes = gm.d_comp_sizes;
        p.d_comp_offsets = gm.d_comp_offsets;
        return p;
    };

    auto flatten_pfor64 = [&](uint64_t field_start, uint64_t npages, uint16_t comp,
        uint64_t nrows, const GpuFieldMeta& gm,
        uint64_t* d_output)
    {
        bam_pfor64_io_flatten_async(pfor64_ctx,
            make_pfor64_params(field_start, npages, comp, nrows, gm),
            reinterpret_cast<int64_t*>(d_output), stream);
        cudaStreamSynchronize(stream);
    };

    // Prepare fused MKTSEG params once
    upload_field_meta(gm_c_mktseg, h_ps_c_mktseg, h_cs_c_mktseg, h_co_c_mktseg,
                      c_mktseg_npages, c_mktseg_comp);

    BAMq3FusedMktsegParams mk_params{};
    mk_params.d_comp_sizes      = gm_c_mktseg.d_comp_sizes;
    mk_params.d_comp_offsets    = gm_c_mktseg.d_comp_offsets;
    mk_params.d_prefix_sum      = gm_c_mktseg.d_prefix_sum;
    mk_params.padded_len        = 12;  // CHAR(10) padded to 12
    mk_params.d_c_custkey_flat  = d_c_custkey_flat;
    mk_params.d_custkey_set     = d_custkey_set;
    mk_params.custkey_set_mask  = custset_mask;
    mk_params.partition_start_lba = partition_start_lba;
    for (uint32_t d = 0; d < n_devices; d++)
        mk_params.partition_start_lbas[d] = ds.partition_start_lbas[d];
    mk_params.n_devices = n_devices;
    mk_params.field_start_page_id = c_mktseg_start;
    mk_params.page_size         = static_cast<uint32_t>(page_size);
    mk_params.npages            = c_mktseg_npages;
    mk_params.num_blocks        = num_blocks_mktseg;
    mk_params.comp_method       = c_mktseg_comp;
    // Q3SEL: multi-segment support
    static constexpr uint64_t Q3SEL_SEGMENTS[5] = {
        0x474E49444C495542ULL, // BUILDING
        0x49424F4D4F545541ULL, // AUTOMOBILE
        0x525554494E525546ULL, // FURNITURE
        0x52454E494843414DULL, // MACHINERY
        0x4C4F484553554F48ULL, // HOUSEHOLD
    };
    if (is_q3sel) {
        uint32_t nseg = (sel_pct >= 100) ? 5 : (uint32_t)(sel_pct / 20);
        if (nseg == 0 && sel_pct > 0 && sel_pct < 100) nseg = 1;
        mk_params.num_segments = nseg;
        for (uint32_t s = 0; s < nseg; s++)
            mk_params.segment_values[s] = Q3SEL_SEGMENTS[s];
    } else {
        mk_params.num_segments = 0;
    }

    uint32_t h_result_count = 0;

    // ── Pre-compute I/O statistics from metadata ──
    // Page-level zone map pruning: each column counted with its own mask.
    uint64_t q3_nios = 0;
    uint64_t q3_read_bytes = 0;
    auto add_col_io_q3 = [&](uint64_t npages, uint16_t comp,
                              const std::vector<uint32_t>& h_cs) {
        for (uint64_t pg = 0; pg < npages; pg++) {
            q3_nios++;
            q3_read_bytes += (comp != 0 && pg < h_cs.size()) ? h_cs[pg] : page_size;
        }
    };
    auto add_col_io_q3_masked_u8 = [&](uint64_t npages, uint16_t comp,
                                        const std::vector<uint32_t>& h_cs,
                                        const uint8_t* mask) {
        for (uint64_t pg = 0; pg < npages; pg++) {
            if (mask && !mask[pg]) continue;
            q3_nios++;
            q3_read_bytes += (comp != 0 && pg < h_cs.size()) ? h_cs[pg] : page_size;
        }
    };
    // CUSTOMER IO (always full — no zone map)
    add_col_io_q3(c_custkey_npages, c_custkey_comp, h_cs_c_custkey);
    add_col_io_q3(c_mktseg_npages, c_mktseg_comp, h_cs_c_mktseg);

    // Zone map masks (d_mask from BamZonemapCtx, INT64 masks allocated here)
    uint8_t* d_mask_ord_i64 = nullptr;
    uint8_t* d_mask_li_i64 = nullptr;
    if (zonemap_enabled) {
        cudaMalloc(&d_mask_ord_i64, o_orderkey_npages);
        cudaMalloc(&d_mask_li_i64, l_orderkey_npages);
    }

    // Upload C_CUSTKEY metadata before measurement (Rule 3)
    upload_field_meta(gm_c_custkey, h_ps_c_custkey, h_cs_c_custkey, h_co_c_custkey,
                      c_custkey_npages, c_custkey_comp);

    // Pre-allocate pinned host buffer for result copy (Rule 4: outside measurement)
    Q3ResultRow* h_results_pinned = nullptr;
    cudaMallocHost(&h_results_pinned, aggr_cap * sizeof(Q3ResultRow));

    size_t gpu_free_after = 0;
    cudaMemGetInfo(&gpu_free_after, &gpu_total);
    uint64_t q3_gpu_mem_bytes = gpu_free_before - gpu_free_after;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_ord_valid || zm_li_valid) {
        bam_pre_io(zm_ord_ctx.d_ctrls, zm_ord_ctx.d_pc, stream);
    }

    auto total_start = std::chrono::high_resolution_clock::now();
    s_kernel_launches = 0;

    // GPU zone map eval (IO + eval + INT64 mask derivation, all GPU-side)
    if (zm_ord_valid) {
        zm_ord_ctx.d_ps_i32   = d_ps_o_orderdate_q3;
        zm_ord_ctx.d_ps_i64   = d_ps_o_i64_q3;
        zm_ord_ctx.d_mask_i64 = d_mask_ord_i64;
        zm_ord_ctx.npages_i64 = static_cast<uint32_t>(o_orderkey_npages);
        bam_zonemap_eval_async(zm_ord_ctx, o_orderdate_npages, zm_ord_nreads, zm_ord_npreds, stream);
        cudaStreamSynchronize(stream);
        s_kernel_launches++;
    }
    if (zm_li_valid) {
        zm_li_ctx.d_ps_i32   = d_ps_l_shipdate;
        zm_li_ctx.d_ps_i64   = d_ps_l_i64_q3;
        zm_li_ctx.d_mask_i64 = d_mask_li_i64;
        zm_li_ctx.npages_i64 = static_cast<uint32_t>(l_orderkey_npages);
        bam_zonemap_eval_async(zm_li_ctx, l_shipdate_npages, zm_li_nreads, zm_li_npreds, stream);
        cudaStreamSynchronize(stream);
        s_kernel_launches++;
    }

    // IO accounting (read back masks from GPU for logging)
    if (zm_ord_valid || zm_li_valid) {
        const uint8_t* ord_mask = zm_ord_valid ? zm_ord_ctx.h_mask : nullptr;
        const uint8_t* li_mask = zm_li_valid ? zm_li_ctx.h_mask : nullptr;

        // Read back INT64 masks from GPU for accounting
        std::vector<uint8_t> h_ord_i64(o_orderkey_npages, 0);
        std::vector<uint8_t> h_li_i64(l_orderkey_npages, 0);
        if (zm_ord_valid)
            cudaMemcpy(h_ord_i64.data(), d_mask_ord_i64, o_orderkey_npages, cudaMemcpyDeviceToHost);
        if (zm_li_valid)
            cudaMemcpy(h_li_i64.data(), d_mask_li_i64, l_orderkey_npages, cudaMemcpyDeviceToHost);

        // ORDERS INT64 (derived INT64 mask)
        for (uint64_t pg = 0; pg < o_orderkey_npages; pg++) {
            if (!h_ord_i64[pg]) continue;
            q3_nios++;
            q3_read_bytes += (o_orderkey_comp != 0 && pg < h_cs_o_orderkey.size()) ? h_cs_o_orderkey[pg] : page_size;
        }
        for (uint64_t pg = 0; pg < o_custkey_npages; pg++) {
            if (!h_ord_i64[pg]) continue;
            q3_nios++;
            q3_read_bytes += (o_custkey_comp != 0 && pg < h_cs_o_custkey.size()) ? h_cs_o_custkey[pg] : page_size;
        }
        // ORDERS INT32 (zone map mask)
        add_col_io_q3_masked_u8(o_orderdate_npages, o_orderdate_comp, h_cs_o_orderdate, ord_mask);
        add_col_io_q3_masked_u8(o_shippri_npages, o_shippri_comp, h_cs_o_shippri, ord_mask);
        // LINEITEM INT64 (derived INT64 mask)
        for (uint64_t pg = 0; pg < l_orderkey_npages; pg++) {
            if (!h_li_i64[pg]) continue;
            q3_nios++;
            q3_read_bytes += (l_orderkey_comp != 0 && pg < h_cs_l_orderkey.size()) ? h_cs_l_orderkey[pg] : page_size;
        }
        // LINEITEM INT32 (zone map mask)
        add_col_io_q3_masked_u8(l_shipdate_npages, l_shipdate_comp, h_cs_l_shipdate, li_mask);
        add_col_io_q3_masked_u8(l_extprice_npages, l_extprice_comp, h_cs_l_extprice, li_mask);
        add_col_io_q3_masked_u8(l_discount_npages, l_discount_comp, h_cs_l_discount, li_mask);

        uint32_t n_ord_active = 0, n_li_active = 0;
        if (ord_mask) for (uint64_t j = 0; j < o_orderdate_npages; j++) if (ord_mask[j]) n_ord_active++;
        if (li_mask) for (uint64_t j = 0; j < l_shipdate_npages; j++) if (li_mask[j]) n_li_active++;
        uint32_t n_ord_i64 = 0, n_li_i64 = 0;
        for (uint8_t b : h_ord_i64) if (b) n_ord_i64++;
        for (uint8_t b : h_li_i64) if (b) n_li_i64++;
        std::cout << "[Q3-ZONEMAP] ORDERS: " << n_ord_active << "/" << o_orderdate_npages
                  << " INT32, " << n_ord_i64 << "/" << o_orderkey_npages << " INT64" << std::endl;
        std::cout << "[Q3-ZONEMAP] LINEITEM: " << n_li_active << "/" << l_shipdate_npages
                  << " INT32, " << n_li_i64 << "/" << l_orderkey_npages << " INT64" << std::endl;
    } else if (zonemap_enabled) {
        // Zonemap enabled but no valid predicates — count all pages
        add_col_io_q3(o_orderkey_npages, o_orderkey_comp, h_cs_o_orderkey);
        add_col_io_q3(o_custkey_npages, o_custkey_comp, h_cs_o_custkey);
        add_col_io_q3(o_orderdate_npages, o_orderdate_comp, h_cs_o_orderdate);
        add_col_io_q3(o_shippri_npages, o_shippri_comp, h_cs_o_shippri);
        add_col_io_q3(l_orderkey_npages, l_orderkey_comp, h_cs_l_orderkey);
        add_col_io_q3(l_shipdate_npages, l_shipdate_comp, h_cs_l_shipdate);
        add_col_io_q3(l_extprice_npages, l_extprice_comp, h_cs_l_extprice);
        add_col_io_q3(l_discount_npages, l_discount_comp, h_cs_l_discount);
    } else {
        // No zonemap — count all pages
        add_col_io_q3(o_orderkey_npages, o_orderkey_comp, h_cs_o_orderkey);
        add_col_io_q3(o_custkey_npages, o_custkey_comp, h_cs_o_custkey);
        add_col_io_q3(o_orderdate_npages, o_orderdate_comp, h_cs_o_orderdate);
        add_col_io_q3(o_shippri_npages, o_shippri_comp, h_cs_o_shippri);
        add_col_io_q3(l_orderkey_npages, l_orderkey_comp, h_cs_l_orderkey);
        add_col_io_q3(l_shipdate_npages, l_shipdate_comp, h_cs_l_shipdate);
        add_col_io_q3(l_extprice_npages, l_extprice_comp, h_cs_l_extprice);
        add_col_io_q3(l_discount_npages, l_discount_comp, h_cs_l_discount);
    }

    // Phase 1: CUSTOMER — flatten C_CUSTKEY, fused MKTSEG IO+decomp+filter → hash set
    std::cout << "[Q3] Loading C_CUSTKEY (" << c_custkey_npages << " pages)..." << std::endl;
    flatten_pfor64(c_custkey_start, c_custkey_npages, c_custkey_comp,
                   nrecs_customer, gm_c_custkey,
                   d_c_custkey_flat);
    s_kernel_launches++;

    cudaMemsetAsync(d_custkey_set, 0xFF, custset_cap * sizeof(uint64_t), stream);

    std::cout << "[Q3] Loading C_MKTSEGMENT ("
              << (c_mktseg_comp == 0 ? "UNCOMP" : is_fsst_mktseg ? "FSST" : "LZ4PAR")
              << " fused, comp_method=" << c_mktseg_comp
              << ", " << c_mktseg_npages << " pages)..." << std::endl;

    if (is_fsst_mktseg) {
        BAMq3FsstMktsegParams fmk{};
        fmk.d_comp_sizes      = gm_c_mktseg.d_comp_sizes;
        fmk.d_comp_offsets    = gm_c_mktseg.d_comp_offsets;
        fmk.d_prefix_sum      = gm_c_mktseg.d_prefix_sum;
        fmk.d_c_custkey_flat  = d_c_custkey_flat;
        fmk.d_custkey_set     = d_custkey_set;
        fmk.custkey_set_mask  = custset_mask;
        for (uint32_t d = 0; d < n_devices; d++)
            fmk.partition_start_lbas[d] = ds.partition_start_lbas[d];
        fmk.n_devices = n_devices;
        fmk.field_start_page_id = c_mktseg_start;
        fmk.page_size         = static_cast<uint32_t>(page_size);
        fmk.npages            = c_mktseg_npages;
        fmk.num_blocks        = num_blocks_mktseg;
        fmk.num_segments      = mk_params.num_segments;
        for (uint32_t s = 0; s < mk_params.num_segments; s++)
            fmk.segment_values[s] = mk_params.segment_values[s];
        bam_q3_fused_mktseg_fsst_async(fsst_ctx, fmk, stream);
        s_kernel_launches++;
    } else {
        bam_q3_fused_mktseg_async(fused_ctx, mk_params, stream);
        s_kernel_launches++;
    }
    cudaStreamSynchronize(stream);
    std::cout << "[Q3] CUSTOMER hash set built (capacity=" << custset_cap << ")" << std::endl;

    // Phase 2: ORDERS fused kernel (INT32 + INT64 via BaM, no flatten)
    cudaMemsetAsync(d_orders_ht_keys, 0xFF, orders_ht_cap * sizeof(uint64_t), stream);
    {
        BAMQ3OrdersFusedParams ofp{};
        ofp.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            ofp.partition_start_lbas[d] = ds.partition_start_lbas[d];
        ofp.n_devices = n_devices;
        ofp.page_size = static_cast<uint32_t>(page_size);
        ofp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        ofp.npages = o_npages_i32;
        ofp.num_blocks = num_blocks_orders_fused;
        // INT32: O_ORDERDATE, O_SHIPPRIORITY
        ofp.field_start_page_ids[0] = o_orderdate_start;
        ofp.comp_methods[0] = o_orderdate_comp;
        ofp.d_comp_sizes[0] = d_cs_o_orderdate_q3;
        ofp.d_comp_offsets[0] = d_co_o_orderdate_q3;
        ofp.field_start_page_ids[1] = o_shippri_start;
        ofp.comp_methods[1] = o_shippri_comp;
        ofp.d_comp_sizes[1] = d_cs_o_shippri_q3;
        ofp.d_comp_offsets[1] = d_co_o_shippri_q3;
        ofp.d_prefix_sum = d_ps_o_orderdate_q3;
        ofp.d_page_active = zm_ord_valid ? zm_ord_ctx.d_mask : nullptr;
        ofp.d_active_page_ids = zm_ord_valid ? zm_ord_ctx.d_active_ids : nullptr;
        ofp.num_active_pages = zm_ord_valid ? *zm_ord_ctx.h_num_active : 0;
        // INT64: O_ORDERKEY, O_CUSTKEY
        ofp.field_start_page_ids_i64[0] = o_orderkey_start;
        ofp.comp_methods_i64[0] = o_orderkey_comp;
        ofp.d_comp_sizes_i64[0] = d_cs_o_orderkey_q3;
        ofp.d_comp_offsets_i64[0] = d_co_o_orderkey_q3;
        ofp.field_start_page_ids_i64[1] = o_custkey_start;
        ofp.comp_methods_i64[1] = o_custkey_comp;
        ofp.d_comp_sizes_i64[1] = d_cs_o_custkey_q3;
        ofp.d_comp_offsets_i64[1] = d_co_o_custkey_q3;
        ofp.d_prefix_sum_i64 = d_ps_o_i64_q3;
        ofp.npages_i64 = o_npages_i64;
        // Customer hash set
        ofp.d_custkey_set = d_custkey_set;
        ofp.custkey_set_mask = custset_mask;
        // Orders HT
        ofp.d_orders_ht_keys = d_orders_ht_keys;
        ofp.d_orders_ht_payloads = d_orders_ht_payloads;
        ofp.orders_ht_mask = orders_ht_mask;
        // Scratch
        ofp.d_scratch = d_scratch;
        ofp.d_scratch_i64 = d_scratch_i64;
        ofp.scratch_stride = static_cast<uint32_t>(max_rows_per_page);
        ofp.scratch_stride_i64 = static_cast<uint32_t>(max_rows_per_i64_page);
        ofp.skip_date_filter = is_q3sel && options.disable_other_filters;

        bam_q3_orders_fused_run(pfor32_fused_ctx, ofp, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }

    // Phase 3: LINEITEM fused kernel (all fields via BaM, no flatten)
    cudaMemsetAsync(d_aggr_keys, 0xFF, aggr_cap * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_aggr_revenues, 0, aggr_cap * sizeof(int64_t), stream);
    {
        BAMQ3LineitemFusedParams lfp{};
        lfp.partition_start_lba = partition_start_lba;
        for (uint32_t d = 0; d < n_devices; d++)
            lfp.partition_start_lbas[d] = ds.partition_start_lbas[d];
        lfp.n_devices = n_devices;
        lfp.page_size = static_cast<uint32_t>(page_size);
        lfp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        lfp.npages = l_npages_i32;
        lfp.num_blocks = num_blocks_lineitem_fused;
        // INT32: L_SHIPDATE, L_EXTPRICE, L_DISCOUNT
        lfp.field_start_page_ids[0] = l_shipdate_start;
        lfp.comp_methods[0] = l_shipdate_comp;
        lfp.d_comp_sizes[0] = d_cs_l_shipdate_q3;
        lfp.d_comp_offsets[0] = d_co_l_shipdate_q3;
        lfp.field_start_page_ids[1] = l_extprice_start;
        lfp.comp_methods[1] = l_extprice_comp;
        lfp.d_comp_sizes[1] = d_cs_l_extprice_q3;
        lfp.d_comp_offsets[1] = d_co_l_extprice_q3;
        lfp.field_start_page_ids[2] = l_discount_start;
        lfp.comp_methods[2] = l_discount_comp;
        lfp.d_comp_sizes[2] = d_cs_l_discount_q3;
        lfp.d_comp_offsets[2] = d_co_l_discount_q3;
        lfp.d_prefix_sum = d_ps_l_shipdate;
        lfp.d_page_active = zm_li_valid ? zm_li_ctx.d_mask : nullptr;
        lfp.d_active_page_ids = zm_li_valid ? zm_li_ctx.d_active_ids : nullptr;
        lfp.num_active_pages = zm_li_valid ? *zm_li_ctx.h_num_active : 0;
        // INT64: L_ORDERKEY
        lfp.field_start_page_ids_i64[0] = l_orderkey_start;
        lfp.comp_methods_i64[0] = l_orderkey_comp;
        lfp.d_comp_sizes_i64[0] = d_cs_l_orderkey_q3;
        lfp.d_comp_offsets_i64[0] = d_co_l_orderkey_q3;
        lfp.d_prefix_sum_i64 = d_ps_l_i64_q3;
        lfp.npages_i64 = l_npages_i64;
        // Orders HT (probe)
        lfp.d_orders_ht_keys = d_orders_ht_keys;
        lfp.d_orders_ht_payloads = d_orders_ht_payloads;
        lfp.orders_ht_mask = orders_ht_mask;
        // Aggregation hash map
        lfp.d_aggr_keys = d_aggr_keys;
        lfp.d_aggr_revenues = d_aggr_revenues;
        lfp.aggr_mask = aggr_mask;
        // Scratch
        lfp.d_scratch = d_scratch;
        lfp.d_scratch_i64 = d_scratch_i64;
        lfp.scratch_stride = static_cast<uint32_t>(max_rows_per_page);
        lfp.scratch_stride_i64 = static_cast<uint32_t>(max_rows_per_i64_page);
        lfp.skip_shipdate_filter = is_q3sel && options.disable_other_filters;
        lfp.d_dbg_counters = nullptr;

        bam_q3_lineitem_fused_run(pfor32_fused_ctx, lfp, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);
    }

    // Phase 4: Collect results
    cudaMemsetAsync(d_result_count, 0, sizeof(uint32_t), stream);
    q3_collect_results(d_aggr_keys, d_aggr_revenues, aggr_cap,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
        d_results, d_result_count, stream);
    s_kernel_launches++;
    cudaStreamSynchronize(stream);

    cudaMemcpy(&h_result_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // GPU sort (CUB DeviceMergeSort with pre-allocated temp)
    cub::DeviceMergeSort::SortKeys(d_sort_temp, sort_temp_bytes,
        d_results, (int)h_result_count, Q3ResultCmp{}, stream);
    s_kernel_launches++;
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_results_pinned, d_results,
               h_result_count * sizeof(Q3ResultRow), cudaMemcpyDeviceToHost);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Print top 10
    if (is_q3sel)
        std::cout << "\n=== TPC-H Q3SEL Result (sel=" << sel_pct << "%, Top 10) ===" << std::endl;
    else
        std::cout << "\n=== TPC-H Q3 Result (Top 10) ===" << std::endl;
    std::cout << "l_orderkey |       revenue | o_orderdate | o_shippriority" << std::endl;
    std::cout << "-----------+---------------+-------------+---------------" << std::endl;
    uint32_t limit = std::min(h_result_count, (uint32_t)10);
    for (uint32_t i = 0; i < limit; i++) {
        auto &r = h_results_pinned[i];
        printf("%10lu | %13ld | %11u | %14u\n",
               (unsigned long)r.l_orderkey, (long)r.revenue,
               r.o_orderdate, r.o_shippriority);
    }
    std::cout << std::endl;
    // ── Cleanup ──
    auto free_gm = [](const GpuFieldMeta& gm, uint16_t comp) {
        cudaFree(gm.d_prefix_sum);
        if (comp != 0) {
            cudaFree(gm.d_comp_sizes);
            cudaFree(gm.d_comp_offsets);
        }
    };
    free_gm(gm_c_custkey, c_custkey_comp);
    free_gm(gm_c_mktseg, c_mktseg_comp);

    if (d_mask_ord_i64) cudaFree(d_mask_ord_i64);
    if (d_mask_li_i64) cudaFree(d_mask_li_i64);
    if (zm_ord_valid) bam_zonemap_ctx_destroy(zm_ord_ctx);
    if (zm_li_valid) bam_zonemap_ctx_destroy(zm_li_ctx);
    if (pfor_ctx_zm) bam_pfor32_io_destroy(pfor_ctx_zm);
    cudaFree(d_c_custkey_flat);
    // Fused INT32 metadata
    cudaFree(d_ps_o_orderdate_q3);
    if (d_cs_o_orderdate_q3) cudaFree(d_cs_o_orderdate_q3);
    if (d_co_o_orderdate_q3) cudaFree(d_co_o_orderdate_q3);
    if (d_cs_o_shippri_q3) cudaFree(d_cs_o_shippri_q3);
    if (d_co_o_shippri_q3) cudaFree(d_co_o_shippri_q3);
    cudaFree(d_ps_l_shipdate);
    if (d_cs_l_shipdate_q3) cudaFree(d_cs_l_shipdate_q3);
    if (d_co_l_shipdate_q3) cudaFree(d_co_l_shipdate_q3);
    if (d_cs_l_extprice_q3) cudaFree(d_cs_l_extprice_q3);
    if (d_co_l_extprice_q3) cudaFree(d_co_l_extprice_q3);
    if (d_cs_l_discount_q3) cudaFree(d_cs_l_discount_q3);
    if (d_co_l_discount_q3) cudaFree(d_co_l_discount_q3);
    // Fused INT64 metadata
    cudaFree(d_ps_o_i64_q3);
    if (d_cs_o_orderkey_q3) cudaFree(d_cs_o_orderkey_q3);
    if (d_co_o_orderkey_q3) cudaFree(d_co_o_orderkey_q3);
    if (d_cs_o_custkey_q3) cudaFree(d_cs_o_custkey_q3);
    if (d_co_o_custkey_q3) cudaFree(d_co_o_custkey_q3);
    cudaFree(d_ps_l_i64_q3);
    if (d_cs_l_orderkey_q3) cudaFree(d_cs_l_orderkey_q3);
    if (d_co_l_orderkey_q3) cudaFree(d_co_l_orderkey_q3);
    cudaFree(d_scratch);
    cudaFree(d_scratch_i64);
    cudaFree(d_custkey_set);
    cudaFree(d_orders_ht_keys);
    cudaFree(d_orders_ht_payloads);
    cudaFree(d_aggr_keys);
    cudaFree(d_aggr_revenues);
    cudaFree(d_results);
    cudaFree(d_result_count);
    cudaFree(d_sort_temp);
    cudaFreeHost(h_results_pinned);
    bam_pfor64_io_destroy(pfor64_ctx);
    bam_pfor32_io_destroy(pfor32_fused_ctx);
    if (fused_ctx) bam_q3_fused_io_destroy(fused_ctx);
    if (fsst_ctx) bam_q3_fsst_io_destroy(fsst_ctx);
    cudaStreamDestroy(stream);
    bam_ctrl_close(ctrl);
    uint64_t q3_total_pages = c_custkey_npages + c_mktseg_npages
                            + o_orderkey_npages + o_custkey_npages + o_orderdate_npages + o_shippri_npages
                            + l_orderkey_npages + l_extprice_npages + l_discount_npages + l_shipdate_npages;
    return PigResult{total_ms, q3_nios, q3_read_bytes,
                     collect_comp_methods({c_custkey_comp, c_mktseg_comp,
                                           o_orderkey_comp, o_custkey_comp, o_orderdate_comp, o_shippri_comp,
                                           l_orderkey_comp, l_extprice_comp, l_discount_comp, l_shipdate_comp}),
                     gpu_ctrl_bytes + q3_gpu_mem_bytes, gpu_ctrl_bytes, q3_gpu_mem_bytes,
                     q3_total_pages,
                     s_kernel_launches};
}

PigResult tpch_q3sel(BenchmarkOptions& options) {
    return tpch_q3(options);
}

// ============================================================
// TPC-H Q1 — PiG (BaM) mode
// ============================================================

PigResult tpch_q1(BenchmarkOptions& options) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);

    const char* bam_ctrl_path = options.file;
    std::cout << "=== TPC-H Q1 (PiG) ===" << std::endl;
    std::cout << "BAM controller: " << bam_ctrl_path << std::endl;

    size_t gpu_free_pre_ctrl = 0, gpu_total_pre = 0;
    cudaMemGetInfo(&gpu_free_pre_ctrl, &gpu_total_pre);

    const uint32_t bam_num_queues = 128;
    auto ds = bam_open_devices(options.file, bam_num_queues, 1024);
    auto ctrl = ds.ctrl;
    const uint32_t n_devices = ds.n_devices;
    const uint64_t partition_start_lba = ds.partition_start_lba;

    size_t gpu_free_post_ctrl = 0;
    cudaMemGetInfo(&gpu_free_post_ctrl, &gpu_total_pre);
    uint64_t gpu_ctrl_bytes = gpu_free_pre_ctrl - gpu_free_post_ctrl;

    // Helper: read a page given global page ID (handles striping)
    auto read_striped_page = [&](uint64_t global_pg_id, uint64_t pg_size, void* dst) -> int {
        uint32_t dev = global_pg_id % n_devices;
        uint64_t local_pg = global_pg_id / n_devices;
        uint64_t lba = ds.partition_start_lbas[dev] + local_pg * (pg_size / 512);
        return bam_read_page(ctrl, pg_size, lba, dst, dev);
    };

    // Read metadata
    const uint64_t init_page_size = 4096;
    std::vector<char> head_buf(init_page_size);
    bam_read_page(ctrl, init_page_size, ds.partition_start_lbas[0], head_buf.data(), 0);
    auto* meta_head = reinterpret_cast<TPCHTableMetadata*>(head_buf.data());
    const size_t page_size = meta_head->page_size;

    std::vector<char> meta_buf(page_size);
    read_striped_page(0, page_size, meta_buf.data());
    TPCHTableMetadata& metadata = *reinterpret_cast<TPCHTableMetadata*>(meta_buf.data());
    superpage_set_constants(metadata.page_size);

    const uint64_t blocks_per_page = page_size / 512;

    // ── LINEITEM field metadata ──
    constexpr size_t col_l_quantity   = TPCH::common::L_QUANTITY;
    constexpr size_t col_l_extprice   = TPCH::common::L_EXTENDEDPRICE;
    constexpr size_t col_l_discount   = TPCH::common::L_DISCOUNT;
    constexpr size_t col_l_tax        = TPCH::common::L_TAX;
    constexpr size_t col_l_returnflag = TPCH::common::L_RETURNFLAG;
    constexpr size_t col_l_linestatus = TPCH::common::L_LINESTATUS;
    constexpr size_t col_l_shipdate   = TPCH::common::L_SHIPDATE;

    // Start pages and page counts
    uint64_t l_qty_start   = metadata.table_lineitem_start_page_ids[col_l_quantity];
    uint64_t l_qty_npages  = metadata.table_lineitem_npages[col_l_quantity];
    uint16_t l_qty_comp    = metadata.table_lineitem_compression_method[col_l_quantity];

    uint64_t l_ext_start   = metadata.table_lineitem_start_page_ids[col_l_extprice];
    uint64_t l_ext_npages  = metadata.table_lineitem_npages[col_l_extprice];
    uint16_t l_ext_comp    = metadata.table_lineitem_compression_method[col_l_extprice];

    uint64_t l_disc_start  = metadata.table_lineitem_start_page_ids[col_l_discount];
    uint64_t l_disc_npages = metadata.table_lineitem_npages[col_l_discount];
    uint16_t l_disc_comp   = metadata.table_lineitem_compression_method[col_l_discount];

    uint64_t l_tax_start   = metadata.table_lineitem_start_page_ids[col_l_tax];
    uint64_t l_tax_npages  = metadata.table_lineitem_npages[col_l_tax];
    uint16_t l_tax_comp    = metadata.table_lineitem_compression_method[col_l_tax];

    uint64_t l_rf_start    = metadata.table_lineitem_start_page_ids[col_l_returnflag];
    uint64_t l_rf_npages   = metadata.table_lineitem_npages[col_l_returnflag];
    uint16_t l_rf_comp     = metadata.table_lineitem_compression_method[col_l_returnflag];

    uint64_t l_ls_start    = metadata.table_lineitem_start_page_ids[col_l_linestatus];
    uint64_t l_ls_npages   = metadata.table_lineitem_npages[col_l_linestatus];
    uint16_t l_ls_comp     = metadata.table_lineitem_compression_method[col_l_linestatus];

    uint64_t l_sd_start    = metadata.table_lineitem_start_page_ids[col_l_shipdate];
    uint64_t l_sd_npages   = metadata.table_lineitem_npages[col_l_shipdate];
    uint16_t l_sd_comp     = metadata.table_lineitem_compression_method[col_l_shipdate];

    uint64_t nrecs_lineitem = metadata.table_lineitem_nrows;
    std::cout << "nrecs: lineitem=" << nrecs_lineitem << std::endl;

    // ================================================================
    // Host-side metadata reads (before total_start)
    // ================================================================

    auto read_prefix_sum_host = [&](uint64_t ps_start, uint64_t ps_npages,
                                     uint64_t field_npages) -> std::vector<uint64_t>
    {
        if (ps_npages == 0) return {};
        std::vector<char> h_buf(ps_npages * page_size);
        for (uint64_t p = 0; p < ps_npages; p++) {
            read_striped_page(ps_start + p, page_size, h_buf.data() + p * page_size);
        }
        uint64_t* ps_raw = reinterpret_cast<uint64_t*>(h_buf.data()) + 1;
        return std::vector<uint64_t>(ps_raw, ps_raw + field_npages);
    };

    auto prepare_comp_metadata = [&](
        uint64_t field_start, uint64_t field_npages, uint16_t comp_method,
        uint64_t cs_start_page, uint64_t cs_npages_cnt,
        uint64_t nbase_val, uint64_t base_start_page)
        -> std::pair<std::vector<uint32_t>, std::vector<uint64_t>>
    {
        if (comp_method == 0) return {{}, {}};
        std::vector<char> sizes_buf(cs_npages_cnt * page_size);
        for (uint64_t p = 0; p < cs_npages_cnt; p++) {
            read_striped_page(cs_start_page + p, page_size, sizes_buf.data() + p * page_size);
        }
        std::vector<uint32_t> comp_sizes(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + field_npages);
        size_t bp_npages = TPCH::nbase_to_npages(nbase_val, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++) {
            read_striped_page(base_start_page + p, page_size, bases_buf.data() + p * page_size);
        }
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            comp_sizes.data(), nbase_val, field_npages, page_size, field_start,
            n_devices, offsets_vec);
        std::vector<uint64_t> comp_offsets(field_npages);
        for (uint64_t i = 0; i < field_npages; i++)
            comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);
#if 0  // Debug: dump segment base page IDs and key params
        {
            const size_t* bases = reinterpret_cast<const size_t*>(bases_buf.data());
            printf("[Q1-COMP-META] field_start=%lu npages=%lu comp=%u nbase=%lu bp_npages=%zu base_start_page=%lu\n",
                   (unsigned long)field_start, (unsigned long)field_npages,
                   comp_method, (unsigned long)nbase_val, bp_npages, (unsigned long)base_start_page);
            for (size_t s = 0; s < std::min(nbase_val, (uint64_t)4); s++)
                printf("  segment_base[%zu]=%zu\n", s, bases[s]);
            printf("  comp_sizes[0..3]=%u %u %u %u\n",
                   comp_sizes[0], comp_sizes.size()>1?comp_sizes[1]:0,
                   comp_sizes.size()>2?comp_sizes[2]:0, comp_sizes.size()>3?comp_sizes[3]:0);
        }
#endif
        return {comp_sizes, comp_offsets};
    };

    auto read_stats_host = [&](uint64_t stats_start, uint64_t stats_npg) -> std::vector<char> {
        std::vector<char> buf(stats_npg * page_size);
        for (uint64_t p = 0; p < stats_npg; p++) {
            read_striped_page(stats_start + p, page_size, buf.data() + p * page_size);
        }
        return buf;
    };

    std::cout << "[Q1] Reading metadata..." << std::endl;

    // Prefix sums for all 7 fields
    auto h_ps_qty = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_quantity],
        metadata.table_lineitem_prefix_sum_npages[col_l_quantity], l_qty_npages);
    auto [h_cs_qty, h_co_qty] = prepare_comp_metadata(
        l_qty_start, l_qty_npages, l_qty_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_quantity],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_quantity],
        metadata.table_lineitem_compression_nbases[col_l_quantity],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_quantity]);

    auto h_ps_ext = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_extprice],
        metadata.table_lineitem_prefix_sum_npages[col_l_extprice], l_ext_npages);
    auto [h_cs_ext, h_co_ext] = prepare_comp_metadata(
        l_ext_start, l_ext_npages, l_ext_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_extprice],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_extprice],
        metadata.table_lineitem_compression_nbases[col_l_extprice],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_extprice]);

    auto h_ps_disc = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_discount],
        metadata.table_lineitem_prefix_sum_npages[col_l_discount], l_disc_npages);
    auto [h_cs_disc, h_co_disc] = prepare_comp_metadata(
        l_disc_start, l_disc_npages, l_disc_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_discount],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_discount],
        metadata.table_lineitem_compression_nbases[col_l_discount],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_discount]);

    auto h_ps_tax = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_tax],
        metadata.table_lineitem_prefix_sum_npages[col_l_tax], l_tax_npages);
    auto [h_cs_tax, h_co_tax] = prepare_comp_metadata(
        l_tax_start, l_tax_npages, l_tax_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_tax],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_tax],
        metadata.table_lineitem_compression_nbases[col_l_tax],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_tax]);

    auto h_ps_rf = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_returnflag],
        metadata.table_lineitem_prefix_sum_npages[col_l_returnflag], l_rf_npages);
    auto [h_cs_rf, h_co_rf] = prepare_comp_metadata(
        l_rf_start, l_rf_npages, l_rf_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_returnflag],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_returnflag],
        metadata.table_lineitem_compression_nbases[col_l_returnflag],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_returnflag]);

    auto h_ps_ls = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_linestatus],
        metadata.table_lineitem_prefix_sum_npages[col_l_linestatus], l_ls_npages);
    auto [h_cs_ls, h_co_ls] = prepare_comp_metadata(
        l_ls_start, l_ls_npages, l_ls_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_linestatus],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_linestatus],
        metadata.table_lineitem_compression_nbases[col_l_linestatus],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_linestatus]);

    auto h_ps_sd = read_prefix_sum_host(
        metadata.table_lineitem_prefix_sum_start_page_ids[col_l_shipdate],
        metadata.table_lineitem_prefix_sum_npages[col_l_shipdate], l_sd_npages);
    auto [h_cs_sd, h_co_sd] = prepare_comp_metadata(
        l_sd_start, l_sd_npages, l_sd_comp,
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[col_l_shipdate],
        metadata.table_lineitem_compressed_page_sizes_npages[col_l_shipdate],
        metadata.table_lineitem_compression_nbases[col_l_shipdate],
        metadata.table_lineitem_compression_base_start_page_ids[col_l_shipdate]);

    // ================================================================
    // Zone map metadata (Rule 3: metadata outside timing)
    // ================================================================
    bool zonemap_enabled = options.enable_zonemap;
    uint64_t zm_sd_nstats = 0, zm_sd_stats_start = 0, zm_sd_stats_npg = 0;
    if (zonemap_enabled) {
        zm_sd_nstats      = metadata.table_lineitem_nstats[col_l_shipdate];
        zm_sd_stats_start = metadata.table_lineitem_stats_start_page_ids[col_l_shipdate];
        zm_sd_stats_npg   = metadata.table_lineitem_stats_npages[col_l_shipdate];
        if (zm_sd_nstats == 0 || zm_sd_stats_start == 0) {
            std::cout << "[Q1-ZONEMAP] No stats available for L_SHIPDATE." << std::endl;
            zonemap_enabled = false;
        }
    }

    const uint64_t npages_ref = l_qty_npages;
    {
        uint64_t all_np[] = {l_qty_npages, l_ext_npages, l_disc_npages, l_tax_npages,
                             l_rf_npages, l_ls_npages, l_sd_npages};
        for (auto np : all_np) {
            if (np != npages_ref) {
                std::cerr << "ERROR: LINEITEM column npages mismatch: " << np
                          << " != " << npages_ref << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    // Check if prefix_sums differ across fields (common for uncomp INT32 vs CHAR)
    // Build per-page max nrows across all 7 fields for correct scratch sizing
    const std::vector<uint64_t>* all_ps[7] = {
        &h_ps_sd, &h_ps_qty, &h_ps_ext, &h_ps_disc, &h_ps_tax, &h_ps_rf, &h_ps_ls};
    bool q1_ps_all_match = true;
    for (size_t i = 0; i < npages_ref; i++) {
        for (int fi = 1; fi < 7; fi++) {
            if ((*all_ps[fi])[i] != (*all_ps[0])[i]) {
                q1_ps_all_match = false;
                break;
            }
        }
        if (!q1_ps_all_match) break;
    }
    // Also check if INT32 paged fields (sd, qty, ext, disc, tax) differ among themselves
    bool q1_int32_ps_mismatch = false;
    {
        const std::vector<uint64_t>* int32_ps[5] = {
            &h_ps_sd, &h_ps_qty, &h_ps_ext, &h_ps_disc, &h_ps_tax};
        for (int fi = 1; fi < 5 && !q1_int32_ps_mismatch; fi++) {
            if (*int32_ps[fi] != *int32_ps[0]) q1_int32_ps_mismatch = true;
        }
    }
    if (!q1_ps_all_match) {
        std::cout << "[Q1] NOTE: prefix_sum differs across fields" << std::endl;
        std::cout << "[Q1]   h_ps_sd[0]=" << h_ps_sd[0]
                  << " h_ps_qty[0]=" << h_ps_qty[0]
                  << " h_ps_rf[0]=" << h_ps_rf[0]
                  << " h_ps_ls[0]=" << h_ps_ls[0] << std::endl;
        if (q1_int32_ps_mismatch)
            std::cout << "[Q1]   INT32 fields also differ — using full flatten+scan path" << std::endl;
    } else {
        std::cout << "[Q1] All 7 fields share identical prefix_sums" << std::endl;
    }

    // ================================================================
    // GPU memory allocation (before total_start)
    // ================================================================

    size_t gpu_free_before = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free_before, &gpu_total);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Aggregation: 6 groups × 7 aggregates
    constexpr size_t agg_size = Q1_NUM_GROUPS * Q1_NUM_AGGS * sizeof(int64_t);
    int64_t* d_agg = nullptr;
    cudaMalloc(&d_agg, agg_size);

    // ── Unified kernel setup (7 fields: all INT32 PFOR) ──
    // Compute max rows per page across all 7 fields (for scratch buffer sizing)
    uint64_t q1_max_rows_per_page = 0;
    for (int fi = 0; fi < 7; fi++) {
        for (uint64_t i = 0; i < npages_ref; i++) {
            uint64_t nr = (i == 0) ? (*all_ps[fi])[0] : (*all_ps[fi])[i] - (*all_ps[fi])[i - 1];
            q1_max_rows_per_page = std::max(q1_max_rows_per_page, nr);
        }
    }

    // Slot-reuse: 1 page_cache slot per block, 4 blocks/SM for IO concurrency.
    // page_cache slots can exceed bam_num_queues (QP sharing via slot % n_qps).
    int sm_count_q1 = 0;
    cudaDeviceGetAttribute(&sm_count_q1, cudaDevAttrMultiProcessorCount, 0);
    const uint32_t num_blocks_q1_unified = std::min(
        static_cast<uint32_t>(sm_count_q1 * 4),
        static_cast<uint32_t>(npages_ref));

    std::cout << "[Q1] Unified kernel: " << num_blocks_q1_unified << " blocks, "
              << npages_ref << " pages; max_rows_per_page=" << q1_max_rows_per_page << std::endl;

    // GPU metadata for all 7 fields
    uint64_t* d_ps_q1 = nullptr;  // shared prefix sum
    cudaMalloc(&d_ps_q1, npages_ref * sizeof(uint64_t));
    cudaMemcpy(d_ps_q1, h_ps_sd.data(), npages_ref * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Helper: upload comp_sizes and comp_offsets to GPU
    auto upload_comp_meta = [&](uint16_t comp, const std::vector<uint32_t>& h_cs,
                                 const std::vector<uint64_t>& h_co,
                                 uint32_t*& d_cs, uint64_t*& d_co) {
        d_cs = nullptr; d_co = nullptr;
        if (comp != 0) {
            cudaMalloc(&d_cs, npages_ref * sizeof(uint32_t));
            cudaMemcpy(d_cs, h_cs.data(), npages_ref * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMalloc(&d_co, npages_ref * sizeof(uint64_t));
            cudaMemcpy(d_co, h_co.data(), npages_ref * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
    };

    uint32_t* d_cs_q1_sd = nullptr;   uint64_t* d_co_q1_sd = nullptr;
    uint32_t* d_cs_q1_qty = nullptr;  uint64_t* d_co_q1_qty = nullptr;
    uint32_t* d_cs_q1_ext = nullptr;  uint64_t* d_co_q1_ext = nullptr;
    uint32_t* d_cs_q1_disc = nullptr; uint64_t* d_co_q1_disc = nullptr;
    uint32_t* d_cs_q1_tax = nullptr;  uint64_t* d_co_q1_tax = nullptr;
    uint32_t* d_cs_q1_rf = nullptr;   uint64_t* d_co_q1_rf = nullptr;
    uint32_t* d_cs_q1_ls = nullptr;   uint64_t* d_co_q1_ls = nullptr;

    upload_comp_meta(l_sd_comp, h_cs_sd, h_co_sd, d_cs_q1_sd, d_co_q1_sd);
    upload_comp_meta(l_qty_comp, h_cs_qty, h_co_qty, d_cs_q1_qty, d_co_q1_qty);
    upload_comp_meta(l_ext_comp, h_cs_ext, h_co_ext, d_cs_q1_ext, d_co_q1_ext);
    upload_comp_meta(l_disc_comp, h_cs_disc, h_co_disc, d_cs_q1_disc, d_co_q1_disc);
    upload_comp_meta(l_tax_comp, h_cs_tax, h_co_tax, d_cs_q1_tax, d_co_q1_tax);
    upload_comp_meta(l_rf_comp, h_cs_rf, h_co_rf, d_cs_q1_rf, d_co_q1_rf);
    upload_comp_meta(l_ls_comp, h_cs_ls, h_co_ls, d_cs_q1_ls, d_co_q1_ls);

    // Per-block scratch buffer: 5 or 7 paged fields × max_rows_per_page
    const uint32_t q1_n_paged = q1_ps_all_match ? 7 : 5;
    // When not all match, CHAR fields need extra scratch for fallback (paged decomp)
    // but max across 7 fields is already in q1_max_rows_per_page
    uint64_t q1_scratch_elems = (uint64_t)num_blocks_q1_unified * 7 * q1_max_rows_per_page;
    int32_t* d_q1_scratch = nullptr;
    cudaMalloc(&d_q1_scratch, q1_scratch_elems * sizeof(int32_t));

    // Pre-allocate flatten buffers before measurement point
    int32_t* d_rf_flat = nullptr;
    int32_t* d_ls_flat = nullptr;
    int32_t* d_flat_all[7] = {};
    uint8_t* d_zm_fi[7] = {};
    uint64_t* d_ps_rf_q1 = nullptr;
    if (!q1_ps_all_match) {
        cudaMalloc(&d_rf_flat, nrecs_lineitem * sizeof(int32_t));
        cudaMalloc(&d_ls_flat, nrecs_lineitem * sizeof(int32_t));
        cudaMalloc(&d_ps_rf_q1, npages_ref * sizeof(uint64_t));
    }
    uint64_t* d_ps_fi[5] = {};  // per-field prefix sum buffers (uploaded before measurement)
    if (q1_int32_ps_mismatch) {
        for (int fi = 0; fi < 5; fi++)
            cudaMalloc(&d_flat_all[fi], nrecs_lineitem * sizeof(int32_t));
        d_flat_all[5] = d_rf_flat;
        d_flat_all[6] = d_ls_flat;
        if (zonemap_enabled) {
            for (int fi = 1; fi < 7; fi++)
                cudaMalloc(&d_zm_fi[fi], npages_ref);
        }
        const std::vector<uint64_t>* q1_ps_vecs[5] = {
            &h_ps_sd, &h_ps_qty, &h_ps_ext, &h_ps_disc, &h_ps_tax};
        for (int fi = 0; fi < 5; fi++) {
            cudaMalloc(&d_ps_fi[fi], npages_ref * sizeof(uint64_t));
            cudaMemcpy(d_ps_fi[fi], q1_ps_vecs[fi]->data(),
                       npages_ref * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
    }

    // Unified IO context (1 slot/block with slot reuse)
    bam_q1_fused_io_ctx_t unified_ctx_q1 = bam_q1_unified_create(
        ctrl, static_cast<uint32_t>(page_size), num_blocks_q1_unified);

    std::cout << "[Q1] num_blocks_unified=" << num_blocks_q1_unified
              << (!q1_ps_all_match ? " (CHAR fields pre-flattened)" : " (all paged)")
              << std::endl;

    int64_t h_agg[Q1_NUM_GROUPS * Q1_NUM_AGGS];

    // IO accounting helper (computed after zonemap eval)
    uint64_t q1_nios = 0;
    uint64_t q1_read_bytes = 0;

    // BamZonemapCtx (Rule 4: alloc outside timing; Rule 3: metadata outside timing)
    bam_pfor32_io_ctx_t pfor_ctx_zm = nullptr;
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (zonemap_enabled && zm_sd_nstats > 0 && zm_sd_stats_npg > 0) {
        pfor_ctx_zm = bam_pfor32_io_create(ctrl, static_cast<uint32_t>(page_size), kBamZonemapMaxReads);
        zm_ctx = bam_zonemap_ctx_create(
            bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
            bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
            (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
            static_cast<uint32_t>(page_size), npages_ref);
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

#if 0  // Host-side verification: read first few compressed pages via multi-device LBA
    {
        std::vector<char> vbuf(page_size);
        auto verify_field = [&](const char* name, uint64_t field_start,
                                const std::vector<uint32_t>& h_cs,
                                const std::vector<uint64_t>& h_co, int count) {
            for (int pg = 0; pg < count; pg++) {
                uint64_t global_pg = field_start + pg;
                uint32_t dev = global_pg % n_devices;
                uint64_t lba = ds.partition_start_lbas[dev] + h_co[pg] / 512;
                bam_read_page(ctrl, page_size, lba, vbuf.data(), dev);
                const uint32_t* hdr = reinterpret_cast<const uint32_t*>(vbuf.data());
                printf("[Q1-VERIFY-%s] page[%d]: dev=%u lba=%lu co=%lu cs=%u  "
                       "hdr[0..3]=%u %u %u %u\n",
                       name, pg, dev, (unsigned long)lba, (unsigned long)h_co[pg],
                       h_cs[pg], hdr[0], hdr[1], hdr[2], hdr[3]);
            }
        };
        int n = std::min((int)npages_ref, 4);
        verify_field("RF", l_rf_start, h_cs_rf, h_co_rf, n);
        verify_field("SD", l_sd_start, h_cs_sd, h_co_sd, n);

        // Also read L_RETURNFLAG page 0 using read_striped_page (uncompressed offset)
        read_striped_page(l_rf_start, page_size, vbuf.data());
        const uint32_t* uhdr = reinterpret_cast<const uint32_t*>(vbuf.data());
        printf("[Q1-VERIFY-RF-UNCOMPRESSED] read_striped_page(%lu): hdr[0..3]=%u %u %u %u\n",
               (unsigned long)l_rf_start, uhdr[0], uhdr[1], uhdr[2], uhdr[3]);

        // Try reading L_RETURNFLAG page 0 as if single-device (all data on device 0)
        {
            uint64_t single_lba = ds.partition_start_lbas[0] + l_rf_start * blocks_per_page;
            bam_read_page(ctrl, page_size, single_lba, vbuf.data(), 0);
            const uint32_t* shdr = reinterpret_cast<const uint32_t*>(vbuf.data());
            printf("[Q1-VERIFY-RF-SINGLE-DEV0] lba=%lu: hdr[0..3]=%u %u %u %u\n",
                   (unsigned long)single_lba, shdr[0], shdr[1], shdr[2], shdr[3]);
        }
        // Also try L_SHIPDATE page 0 as-if single-device for comparison
        {
            uint64_t single_lba = ds.partition_start_lbas[0] + l_sd_start * blocks_per_page;
            bam_read_page(ctrl, page_size, single_lba, vbuf.data(), 0);
            const uint32_t* shdr = reinterpret_cast<const uint32_t*>(vbuf.data());
            printf("[Q1-VERIFY-SD-SINGLE-DEV0] lba=%lu: hdr[0..3]=%u %u %u %u\n",
                   (unsigned long)single_lba, shdr[0], shdr[1], shdr[2], shdr[3]);
        }
    }
#endif

    // Pre-create IO contexts before measurement (Rule 4)
    bam_pfor32_io_ctx_t pfor32_ctx_q1 = nullptr;
    bam_pfor32_io_ctx_t flat_ctx = nullptr;
    if (!q1_ps_all_match) {
        pfor32_ctx_q1 = bam_pfor32_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks_q1_unified);
    }
    if (q1_int32_ps_mismatch) {
        flat_ctx = bam_pfor32_io_create(
            ctrl, static_cast<uint32_t>(page_size), num_blocks_q1_unified);
    }

    // Upload metadata before measurement (Rule 3)
    if (!q1_ps_all_match) {
        cudaMemcpy(d_ps_rf_q1, h_ps_rf.data(), npages_ref * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    size_t gpu_free_after = 0;
    cudaMemGetInfo(&gpu_free_after, &gpu_total);
    uint64_t q1_gpu_mem_bytes = gpu_free_before - gpu_free_after;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    auto total_start = std::chrono::high_resolution_clock::now();
    s_kernel_launches = 0;
    double total_ms = 0;

    // GPU zonemap eval (Rule 6: IO + eval inside timing, mask stays on GPU)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages_ref, zm_nreads, zm_npreds, stream);
        cudaStreamSynchronize(stream);
        s_kernel_launches++;
    }

    // IO accounting (after zonemap eval so h_mask is available)
    {
        auto add_col_io = [&](uint64_t npg, uint16_t comp, const std::vector<uint32_t>& h_cs) {
            for (uint64_t pg = 0; pg < npg; pg++) {
                if (zm_valid && !zm_ctx.h_mask[pg]) continue;
                q1_nios++;
                q1_read_bytes += (comp != 0 && pg < h_cs.size()) ? h_cs[pg] : page_size;
            }
        };
        add_col_io(l_qty_npages, l_qty_comp, h_cs_qty);
        add_col_io(l_ext_npages, l_ext_comp, h_cs_ext);
        add_col_io(l_disc_npages, l_disc_comp, h_cs_disc);
        add_col_io(l_tax_npages, l_tax_comp, h_cs_tax);
        add_col_io(l_rf_npages, l_rf_comp, h_cs_rf);
        add_col_io(l_ls_npages, l_ls_comp, h_cs_ls);
        add_col_io(l_sd_npages, l_sd_comp, h_cs_sd);
    }

    // Pre-flatten L_RETURNFLAG and L_LINESTATUS if prefix sums differ
    if (!q1_ps_all_match) {
        std::cout << "[Q1] Pre-flattening L_RETURNFLAG and L_LINESTATUS..." << std::endl;

        BAMPfor32FlattenParams p_rf{};
        for (uint32_t d = 0; d < n_devices; d++)
            p_rf.partition_start_lbas[d] = ds.partition_start_lbas[d];
        p_rf.n_devices = n_devices;
        p_rf.page_size = static_cast<uint32_t>(page_size);
        p_rf.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
        p_rf.comp_method = l_rf_comp;
        p_rf.field_start_page_id = l_rf_start;
        p_rf.npages = l_rf_npages;
        p_rf.nrows = nrecs_lineitem;
        p_rf.num_blocks = num_blocks_q1_unified;
        p_rf.d_prefix_sum = d_ps_rf_q1;
        p_rf.d_comp_sizes = d_cs_q1_rf;
        p_rf.d_comp_offsets = d_co_q1_rf;
        bam_pfor32_io_flatten_async(pfor32_ctx_q1, p_rf, d_rf_flat, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);

        BAMPfor32FlattenParams p_ls = p_rf;
        p_ls.comp_method = l_ls_comp;
        p_ls.field_start_page_id = l_ls_start;
        p_ls.d_comp_sizes = d_cs_q1_ls;
        p_ls.d_comp_offsets = d_co_q1_ls;
        bam_pfor32_io_flatten_async(pfor32_ctx_q1, p_ls, d_ls_flat, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);

        std::cout << "[Q1] Pre-flatten done." << std::endl;
    }

    if (q1_int32_ps_mismatch) {
        // ── Full flatten path: all 7 fields flattened → q1_scan_aggregate_flat_i32 ──
        // d_rf_flat and d_ls_flat already allocated and flattened above.
        // Flatten the remaining 5 INT32 fields.
        // flat_ctx already created before total_start (Rule 4).

        // Field mapping: [0]=SD [1]=QTY [2]=EXT [3]=DISC [4]=TAX [5]=RF [6]=LS
        struct FieldInfo {
            uint64_t start; uint16_t comp; uint32_t* d_cs; uint64_t* d_co;
            const std::vector<uint64_t>* h_ps;
        };
        FieldInfo finfo[7] = {
            {l_sd_start,   l_sd_comp,   d_cs_q1_sd,   d_co_q1_sd,   &h_ps_sd},
            {l_qty_start,  l_qty_comp,  d_cs_q1_qty,  d_co_q1_qty,  &h_ps_qty},
            {l_ext_start,  l_ext_comp,  d_cs_q1_ext,  d_co_q1_ext,  &h_ps_ext},
            {l_disc_start, l_disc_comp, d_cs_q1_disc, d_co_q1_disc, &h_ps_disc},
            {l_tax_start,  l_tax_comp,  d_cs_q1_tax,  d_co_q1_tax,  &h_ps_tax},
            {l_rf_start,   l_rf_comp,   d_cs_q1_rf,   d_co_q1_rf,   &h_ps_rf},
            {l_ls_start,   l_ls_comp,   d_cs_q1_ls,   d_co_q1_ls,   &h_ps_ls},
        };

        // d_flat_all, d_zm_fi are pre-allocated before the measurement point.
        d_flat_all[5] = d_rf_flat;
        d_flat_all[6] = d_ls_flat;

        // Per-field zone map masks (same approach as Q6)
        if (zm_valid) {
            d_zm_fi[0] = zm_ctx.d_mask;
            for (int fi = 1; fi < 7; fi++) {
                std::vector<uint8_t> h_mask_fi(npages_ref, 0);
                for (uint64_t pg0 = 0; pg0 < npages_ref; pg0++) {
                    if (!zm_ctx.h_mask[pg0]) continue;
                    uint64_t row_lo = (pg0 == 0) ? 0 : h_ps_sd[pg0 - 1];
                    uint64_t row_hi = h_ps_sd[pg0];
                    auto it_lo = std::upper_bound(finfo[fi].h_ps->begin(), finfo[fi].h_ps->end(), row_lo);
                    uint64_t pg_lo = static_cast<uint64_t>(it_lo - finfo[fi].h_ps->begin());
                    auto it_hi = std::lower_bound(finfo[fi].h_ps->begin(), finfo[fi].h_ps->end(), row_hi);
                    uint64_t pg_hi = static_cast<uint64_t>(it_hi - finfo[fi].h_ps->begin());
                    for (uint64_t pg = pg_lo; pg <= pg_hi && pg < npages_ref; pg++)
                        h_mask_fi[pg] = 1;
                }
                cudaMemcpy(d_zm_fi[fi], h_mask_fi.data(), npages_ref, cudaMemcpyHostToDevice);
            }
        }

        // Flatten all 7 fields
        for (int fi = 0; fi < 7; fi++) {
            // RF/LS (fi=5,6) already flattened with their own prefix_sums
            if (fi >= 5) continue;

            BAMPfor32FlattenParams fp{};
            for (uint32_t d = 0; d < n_devices; d++)
                fp.partition_start_lbas[d] = ds.partition_start_lbas[d];
            fp.n_devices = n_devices;
            fp.page_size = static_cast<uint32_t>(page_size);
            fp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
            fp.comp_method = finfo[fi].comp;
            fp.field_start_page_id = finfo[fi].start;
            fp.npages = npages_ref;
            fp.nrows = nrecs_lineitem;
            fp.num_blocks = num_blocks_q1_unified;
            fp.d_prefix_sum = d_ps_fi[fi];
            fp.d_comp_sizes = finfo[fi].d_cs;
            fp.d_comp_offsets = finfo[fi].d_co;

            bam_pfor32_io_flatten_masked_async(
                flat_ctx, fp, d_zm_fi[fi],
                /*fill_value=*/0, d_flat_all[fi], stream);
            s_kernel_launches++;
            cudaStreamSynchronize(stream);
        }

        // Run flat INT32 Q1 scan
        cudaMemsetAsync(d_agg, 0, agg_size, stream);
        q1_scan_aggregate_flat_i32(
            d_flat_all[0], d_flat_all[1], d_flat_all[2],
            d_flat_all[3], d_flat_all[4], d_flat_all[5], d_flat_all[6],
            nrecs_lineitem, d_agg, stream);
        s_kernel_launches++;
        cudaStreamSynchronize(stream);

        cudaMemcpy(h_agg, d_agg, agg_size, cudaMemcpyDeviceToHost);

        auto total_end = std::chrono::high_resolution_clock::now();
        total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        // Cleanup flatten-specific resources (d_flat_all, d_zm_fi freed in main cleanup)
        bam_pfor32_io_destroy(flat_ctx);

    } else {
        // ── Unified fused kernel (original path) ──
        cudaMemsetAsync(d_agg, 0, agg_size, stream);
        {
            BAMq1UnifiedParams fp{};
            for (uint32_t d = 0; d < n_devices; d++)
                fp.partition_start_lbas[d] = ds.partition_start_lbas[d];
            fp.n_devices = n_devices;
            fp.page_size = static_cast<uint32_t>(page_size);
            fp.blocks_per_page = static_cast<uint32_t>(blocks_per_page);
            fp.npages = npages_ref;
            fp.num_blocks = num_blocks_q1_unified;
            fp.field_start_page_ids[0] = l_sd_start;
            fp.comp_methods[0] = l_sd_comp;
            fp.d_comp_sizes[0] = d_cs_q1_sd;
            fp.d_comp_offsets[0] = d_co_q1_sd;
            fp.field_start_page_ids[1] = l_qty_start;
            fp.comp_methods[1] = l_qty_comp;
            fp.d_comp_sizes[1] = d_cs_q1_qty;
            fp.d_comp_offsets[1] = d_co_q1_qty;
            fp.field_start_page_ids[2] = l_ext_start;
            fp.comp_methods[2] = l_ext_comp;
            fp.d_comp_sizes[2] = d_cs_q1_ext;
            fp.d_comp_offsets[2] = d_co_q1_ext;
            fp.field_start_page_ids[3] = l_disc_start;
            fp.comp_methods[3] = l_disc_comp;
            fp.d_comp_sizes[3] = d_cs_q1_disc;
            fp.d_comp_offsets[3] = d_co_q1_disc;
            fp.field_start_page_ids[4] = l_tax_start;
            fp.comp_methods[4] = l_tax_comp;
            fp.d_comp_sizes[4] = d_cs_q1_tax;
            fp.d_comp_offsets[4] = d_co_q1_tax;
            fp.field_start_page_ids[5] = l_rf_start;
            fp.comp_methods[5] = l_rf_comp;
            fp.d_comp_sizes[5] = d_cs_q1_rf;
            fp.d_comp_offsets[5] = d_co_q1_rf;
            fp.field_start_page_ids[6] = l_ls_start;
            fp.comp_methods[6] = l_ls_comp;
            fp.d_comp_sizes[6] = d_cs_q1_ls;
            fp.d_comp_offsets[6] = d_co_q1_ls;
            fp.d_prefix_sum = d_ps_q1;
            fp.d_flat_rf = d_rf_flat;
            fp.d_flat_ls = d_ls_flat;
            fp.d_page_active = zm_valid ? zm_ctx.d_mask : nullptr;
            fp.d_active_page_ids = zm_valid ? zm_ctx.d_active_ids : nullptr;
            fp.num_active_pages = zm_valid ? *zm_ctx.h_num_active : 0;
            fp.d_agg = d_agg;
            fp.d_scratch = d_q1_scratch;
            fp.scratch_stride = static_cast<uint32_t>(q1_max_rows_per_page);
            bam_q1_unified_run(unified_ctx_q1, fp, stream);
            s_kernel_launches++;
            cudaStreamSynchronize(stream);
        }

        cudaMemcpy(h_agg, d_agg, agg_size, cudaMemcpyDeviceToHost);

        auto total_end = std::chrono::high_resolution_clock::now();
        total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    }

    // Print results (same format as GOLAP Q1)
    const char returnflags[]  = {'A', 'A', 'N', 'N', 'R', 'R'};
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

        double avg_qty   = sum_qty / (double)count;
        double avg_price = sum_base_price / (double)count;
        double avg_disc  = sum_discount / (double)count;

        printf("%c|%c|%ld|%ld|%ld|",
               returnflags[g], linestatuses[g],
               sum_qty, sum_base_price, sum_disc_price);
        q1_print_u128(sum_charge);
        printf("|%lf|%lf|%lf|%ld\n",
               avg_qty, avg_price, avg_disc, count);
    }
    std::cout << std::endl;

    // ── Cleanup ──
    if (d_ps_rf_q1) cudaFree(d_ps_rf_q1);
    if (pfor32_ctx_q1) bam_pfor32_io_destroy(pfor32_ctx_q1);
    cudaFree(d_agg);
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    if (pfor_ctx_zm) bam_pfor32_io_destroy(pfor_ctx_zm);
    // Unified metadata
    cudaFree(d_ps_q1);
    if (d_cs_q1_sd) cudaFree(d_cs_q1_sd);
    if (d_co_q1_sd) cudaFree(d_co_q1_sd);
    if (d_cs_q1_qty) cudaFree(d_cs_q1_qty);
    if (d_co_q1_qty) cudaFree(d_co_q1_qty);
    if (d_cs_q1_ext) cudaFree(d_cs_q1_ext);
    if (d_co_q1_ext) cudaFree(d_co_q1_ext);
    if (d_cs_q1_disc) cudaFree(d_cs_q1_disc);
    if (d_co_q1_disc) cudaFree(d_co_q1_disc);
    if (d_cs_q1_tax) cudaFree(d_cs_q1_tax);
    if (d_co_q1_tax) cudaFree(d_co_q1_tax);
    if (d_cs_q1_rf) cudaFree(d_cs_q1_rf);
    if (d_co_q1_rf) cudaFree(d_co_q1_rf);
    if (d_cs_q1_ls) cudaFree(d_cs_q1_ls);
    if (d_co_q1_ls) cudaFree(d_co_q1_ls);
    cudaFree(d_q1_scratch);
    if (d_rf_flat) cudaFree(d_rf_flat);
    if (d_ls_flat) cudaFree(d_ls_flat);
    // Pre-allocated flatten buffers
    for (int fi = 0; fi < 5; fi++)
        if (d_flat_all[fi]) cudaFree(d_flat_all[fi]);
    for (int fi = 1; fi < 7; fi++)
        if (d_zm_fi[fi]) cudaFree(d_zm_fi[fi]);
    for (int fi = 0; fi < 5; fi++)
        if (d_ps_fi[fi]) cudaFree(d_ps_fi[fi]);
    bam_q1_unified_destroy(unified_ctx_q1);
    cudaStreamDestroy(stream);
    bam_ctrl_close(ctrl);
    uint64_t q1_total_pages = l_qty_npages + l_ext_npages + l_disc_npages + l_tax_npages
                            + l_rf_npages + l_ls_npages + l_sd_npages;
    return PigResult{total_ms, q1_nios, q1_read_bytes,
                     collect_comp_methods({l_qty_comp, l_ext_comp, l_disc_comp, l_tax_comp,
                                           l_rf_comp, l_ls_comp, l_sd_comp}),
                     gpu_ctrl_bytes + q1_gpu_mem_bytes, gpu_ctrl_bytes, q1_gpu_mem_bytes,
                     q1_total_pages,
                     s_kernel_launches};
}

} // namespace BAM
