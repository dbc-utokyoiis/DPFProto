#pragma once

// gidp_bam.cu — SSB GIDP+BAM execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMP (host-launched), Kernel: same as GIDP
//
// Included AFTER gidp.cu in ssb_main.cu, so SsbGidp:: helpers are accessible.

#include "tpch/bam_bulk_read.cuh"
#include "kernel/ssb/dim_build.cuh"
#include "ssb/bam_lz4_fused_ssb.cuh"

// Aliases from SsbGidp namespace (defined in gidp.cu)
using SsbGidp::NvcompDecompCtx;
using SsbGidp::nvcomp_decompctx_alloc;
using SsbGidp::nvcomp_decompctx_run;
using SsbGidp::nvcomp_decompctx_free;
using SsbGidp::ssb_zero_page_headers_kernel;
using SsbGidp::collect_compression_methods;
using SsbGidp::GpuHT;
using SsbGidp::make_gpu_ht;
using SsbGidp::alloc_gpu_ht;
using SsbGidp::ssb_hash32_host;
using SsbGidp::align4;

// NVMe PRP2 danger zone fix (same as TPC-H gidp_bam.cu)
static inline uint32_t bam_safe_nblocks(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) nblk = 17;
    return nblk;
}

namespace SsbGidpBam {

static size_t s_kernel_launches;

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

// ============================================================
// BaM device setup (copied from TPC-H datapathfusion.cu)
// ============================================================
struct BAMDeviceSetup {
    bam_ctrl_handle_t ctrl;
    uint32_t n_devices;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint64_t partition_start_lba;
};

static BAMDeviceSetup bam_open_devices(const char* file, uint32_t num_queues,
                                        uint32_t queue_depth) {
    BAMDeviceSetup s{};
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

    if (s.n_devices == 1)
        s.ctrl = bam_ctrl_open(dev_paths[0], 1, 0, queue_depth, num_queues);
    else
        s.ctrl = bam_ctrl_open_multi(dev_paths.data(), s.n_devices, 1, 0,
                                      queue_depth, num_queues);

    for (uint32_t d = 0; d < s.n_devices; d++) {
        const uint64_t det_size = 4096;
        std::vector<char> det_buf(det_size);
        int drc = bam_read_page(s.ctrl, det_size, 0, det_buf.data(), d);
        if (drc != 0) {
            s.partition_start_lbas[d] = 0;
            continue;
        }
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

// ============================================================
// Pre-cached compression metadata for a single dim field.
// Built once at session open, reused across query executions.
// ============================================================
struct DimFieldCompMeta {
    size_t start_page = 0;
    size_t npages = 0;
    CompressionMethod comp_method = CompressionMethod::NONE;
    std::vector<BAMBatchReadEntry> data_entries;   // pre-computed BaM LBA entries
    std::vector<uint32_t> comp_page_sizes;         // per-page compressed sizes
};

// ============================================================
// Dimension table read context: shared page_cache + GPU buffers
// for BaM batch read + nvCOMP GPU decompression.
// Pre-allocated in bam_session_open, freed in bam_session_close.
// Two pipes enable overlapping BaM I/O with GPU decompression.
// ============================================================
struct DimReadCtx {
    bam_pfor32_io_ctx_t io_ctx = nullptr;
    // Pipe 0 (also used by legacy CPU path)
    cudaStream_t stream = nullptr;
    NvcompDecompCtx nvctx{};
    char *d_comp = nullptr;     // GPU buffer for compressed pages
    char *d_decomp = nullptr;   // GPU buffer for decompressed pages (CPU path)
    // Pipe 1 (for overlapping)
    cudaStream_t stream2 = nullptr;
    NvcompDecompCtx nvctx2{};
    char *d_comp2 = nullptr;
    size_t max_pages = 0;       // pre-allocated capacity (pages)
    // Pre-allocated batch entry buffers (per-pipe to avoid race on 2-pipe overlap)
    BAMBatchReadEntry *d_batch_entries = nullptr;   // pipe 0
    BAMBatchReadEntry *d_batch_entries2 = nullptr;  // pipe 1
    uint32_t *d_comp_sizes = nullptr;               // pipe 0 (fused IO+decomp)
    uint32_t *d_comp_sizes2 = nullptr;              // pipe 1 (fused IO+decomp)
    // Pre-cached dim field metadata (key = table<<8 | field_idx)
    std::map<uint32_t, DimFieldCompMeta> field_cache;
};

// ============================================================
// BaM-based page reader for small tables (dimension tables)
// Uses shared page_cache + GPU nvCOMP decompression.
// ============================================================
static void *bam_read_field_pages_cpu(
    DimReadCtx &dim,
    uint32_t n_devices, const uint64_t *partition_start_lbas,
    const SSBTableMetadata &metadata,
    SSB::common::Table table, size_t field_idx, size_t page_size)
{
    auto make_striped_entries = [&](uint64_t start, size_t np) {
        uint32_t bpp = static_cast<uint32_t>(page_size / 512);
        std::vector<BAMBatchReadEntry> entries(np);
        for (size_t j = 0; j < np; j++) {
            uint64_t pg_id = start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            entries[j] = {partition_start_lbas[dev] + local * bpp, dev, bpp};
        }
        return entries;
    };

    size_t start_page = 0, npages = 0;
    uint16_t comp_method_raw = 0;
    size_t comp_sizes_npages = 0, comp_sizes_start = 0;
    size_t nbase = 0, base_start = 0;

    switch (table) {
    case SSB::common::Table::DDATE:
        start_page = metadata.table_date_start_page_ids[field_idx];
        npages = metadata.table_date_npages[field_idx];
        comp_method_raw = metadata.table_date_compression_method[field_idx];
        comp_sizes_npages = metadata.table_date_compressed_page_sizes_npages[field_idx];
        comp_sizes_start = metadata.table_date_compressed_page_sizes_start_page_ids[field_idx];
        nbase = metadata.table_date_compression_nbases[field_idx];
        base_start = metadata.table_date_compression_base_start_page_ids[field_idx];
        break;
    case SSB::common::Table::CUSTOMER:
        start_page = metadata.table_customer_start_page_ids[field_idx];
        npages = metadata.table_customer_npages[field_idx];
        comp_method_raw = metadata.table_customer_compression_method[field_idx];
        comp_sizes_npages = metadata.table_customer_compressed_page_sizes_npages[field_idx];
        comp_sizes_start = metadata.table_customer_compressed_page_sizes_start_page_ids[field_idx];
        nbase = metadata.table_customer_compression_nbases[field_idx];
        base_start = metadata.table_customer_compression_base_start_page_ids[field_idx];
        break;
    case SSB::common::Table::SUPPLIER:
        start_page = metadata.table_supplier_start_page_ids[field_idx];
        npages = metadata.table_supplier_npages[field_idx];
        comp_method_raw = metadata.table_supplier_compression_method[field_idx];
        comp_sizes_npages = metadata.table_supplier_compressed_page_sizes_npages[field_idx];
        comp_sizes_start = metadata.table_supplier_compressed_page_sizes_start_page_ids[field_idx];
        nbase = metadata.table_supplier_compression_nbases[field_idx];
        base_start = metadata.table_supplier_compression_base_start_page_ids[field_idx];
        break;
    case SSB::common::Table::PART:
        start_page = metadata.table_part_start_page_ids[field_idx];
        npages = metadata.table_part_npages[field_idx];
        comp_method_raw = metadata.table_part_compression_method[field_idx];
        comp_sizes_npages = metadata.table_part_compressed_page_sizes_npages[field_idx];
        comp_sizes_start = metadata.table_part_compressed_page_sizes_start_page_ids[field_idx];
        nbase = metadata.table_part_compression_nbases[field_idx];
        base_start = metadata.table_part_compression_base_start_page_ids[field_idx];
        break;
    default:
        break;
    }

    if (npages == 0) return nullptr;

    auto comp_method = static_cast<CompressionMethod>(comp_method_raw);

    // Uncompressed: batch read all pages via shared page_cache
    if (comp_method == CompressionMethod::NONE) {
        void *buf = mb_alloc(npages * page_size);
        auto entries = make_striped_entries(start_page, npages);
        bam_read_pages_batch_to_host(dim.io_ctx, static_cast<uint32_t>(page_size),
            entries.data(), static_cast<uint32_t>(npages),
            static_cast<char *>(buf), dim.stream);
        return buf;
    }

    // Compressed: batch read metadata pages to host (for offset calculation)
    std::vector<char> sizes_buf(comp_sizes_npages * page_size);
    {
        auto entries = make_striped_entries(comp_sizes_start, comp_sizes_npages);
        bam_read_pages_batch_to_host(dim.io_ctx, static_cast<uint32_t>(page_size),
            entries.data(), static_cast<uint32_t>(comp_sizes_npages),
            sizes_buf.data(), dim.stream);
    }
    uint32_t *comp_page_sizes = reinterpret_cast<uint32_t *>(sizes_buf.data());

    size_t base_npages = SSB::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    {
        auto entries = make_striped_entries(base_start, base_npages);
        bam_read_pages_batch_to_host(dim.io_ctx, static_cast<uint32_t>(page_size),
            entries.data(), static_cast<uint32_t>(base_npages),
            bases_buf.data(), dim.stream);
    }

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t *>(bases_buf.data()),
        comp_page_sizes, nbase, npages, page_size,
        start_page, n_devices, offsets_vec);

    // Batch read compressed data pages to GPU
    std::vector<BAMBatchReadEntry> data_entries(npages);
    for (size_t pg = 0; pg < npages; pg++) {
        size_t logical_page = start_page + pg;
        uint32_t dev = logical_page % n_devices;
        uint64_t lba = partition_start_lbas[dev] + offsets_vec[pg] / 512;
        uint32_t nblk = static_cast<uint32_t>((roundup4096(comp_page_sizes[pg]) + 511) / 512);
        if (nblk == 0) nblk = 1;
        nblk = bam_safe_nblocks(nblk);
        data_entries[pg] = {lba, dev, nblk};
    }
    bam_read_pages_batch_to_gpu(dim.io_ctx, static_cast<uint32_t>(page_size),
        data_entries.data(), static_cast<uint32_t>(npages),
        dim.d_comp, dim.stream);

    // GPU nvCOMP decompression
    auto &nv = dim.nvctx;
    for (size_t pg = 0; pg < npages; pg++) {
        nv.h_comp_ptrs[pg]    = dim.d_comp + pg * page_size;
        nv.h_decomp_ptrs[pg]  = dim.d_decomp + pg * page_size;
        nv.h_comp_sizes[pg]   = comp_page_sizes[pg];
        nv.h_decomp_sizes[pg] = page_size;
    }
    nvcomp_decompctx_run(comp_method, nv, npages, page_size, dim.stream);
    s_kernel_launches++;

    // Copy decompressed pages from GPU to host
    void *out_buf = mb_alloc(npages * page_size);
    CUDA_CHECK(cudaMemcpy(out_buf, dim.d_decomp, npages * page_size, cudaMemcpyDeviceToHost));
    return out_buf;
}

// ============================================================
// Extract field metadata from SSBTableMetadata (shared helper)
// ============================================================
static void bam_extract_field_meta(
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

// ============================================================
// Pre-cache compression metadata for one dim field.
// Reads comp_page_sizes + bases from NVMe, computes BaM entries.
// Called once at session open time (outside timing).
// ============================================================
static DimFieldCompMeta bam_precache_field(
    DimReadCtx &dim, uint32_t n_devices, const uint64_t *partition_start_lbas,
    const SSBTableMetadata &metadata,
    SSB::common::Table table, size_t field_idx, size_t page_size)
{
    DimFieldCompMeta meta;
    size_t sizes_npages = 0, sizes_start = 0, nbase = 0, base_start = 0;
    uint16_t comp_raw = 0;
    bam_extract_field_meta(metadata, table, field_idx,
        meta.start_page, meta.npages, comp_raw,
        sizes_npages, sizes_start, nbase, base_start);
    meta.comp_method = static_cast<CompressionMethod>(comp_raw);
    if (meta.npages == 0) return meta;

    uint32_t bpp = static_cast<uint32_t>(page_size / 512);
    auto make_striped_entries = [&](uint64_t start, size_t np) {
        std::vector<BAMBatchReadEntry> entries(np);
        for (size_t j = 0; j < np; j++) {
            uint64_t pg_id = start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            entries[j] = {partition_start_lbas[dev] + local * bpp, dev, bpp};
        }
        return entries;
    };

    if (meta.comp_method == CompressionMethod::NONE) {
        meta.data_entries = make_striped_entries(meta.start_page, meta.npages);
        meta.comp_page_sizes.assign(meta.npages, page_size);  // fused path: comp_sz==page_size → direct copy
        return meta;
    }

    // Read comp_page_sizes from NVMe to host
    std::vector<char> sizes_buf(sizes_npages * page_size);
    {
        auto entries = make_striped_entries(sizes_start, sizes_npages);
        bam_read_pages_batch_to_host(dim.io_ctx, static_cast<uint32_t>(page_size),
            entries.data(), static_cast<uint32_t>(sizes_npages),
            sizes_buf.data(), dim.stream);
    }
    uint32_t *cps = reinterpret_cast<uint32_t *>(sizes_buf.data());
    meta.comp_page_sizes.assign(cps, cps + meta.npages);

    // Read bases from NVMe to host
    size_t base_npages = SSB::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(base_npages * page_size);
    {
        auto entries = make_striped_entries(base_start, base_npages);
        bam_read_pages_batch_to_host(dim.io_ctx, static_cast<uint32_t>(page_size),
            entries.data(), static_cast<uint32_t>(base_npages),
            bases_buf.data(), dim.stream);
    }

    // Compute compressed offsets → BaM entries
    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t *>(bases_buf.data()),
        cps, nbase, meta.npages, page_size,
        meta.start_page, n_devices, offsets_vec);

    meta.data_entries.resize(meta.npages);
    for (size_t pg = 0; pg < meta.npages; pg++) {
        size_t logical_page = meta.start_page + pg;
        uint32_t dev = logical_page % n_devices;
        uint64_t lba = partition_start_lbas[dev] + offsets_vec[pg] / 512;
        uint32_t nblk = static_cast<uint32_t>((roundup4096(meta.comp_page_sizes[pg]) + 511) / 512);
        if (nblk == 0) nblk = 1;
        nblk = bam_safe_nblocks(nblk);
        meta.data_entries[pg] = {lba, dev, nblk};
    }
    return meta;
}

// ============================================================
// Fast dim field read using pre-cached metadata (no NVMe metadata reads).
// pipe_idx: 0 = use stream/nvctx/d_comp, 1 = use stream2/nvctx2/d_comp2
// ============================================================
static void bam_dim_read_fast(
    DimReadCtx &dim, uint32_t pipe_idx,
    const DimFieldCompMeta &meta, size_t page_size,
    char *d_out, size_t &kl, uint64_t &io_bytes)
{
    if (meta.npages == 0) return;
    for (size_t pg = 0; pg < meta.npages; pg++)
        io_bytes += (uint64_t)meta.data_entries[pg].nblk * 512;
    cudaStream_t s     = (pipe_idx == 0) ? dim.stream  : dim.stream2;
    NvcompDecompCtx &nv = (pipe_idx == 0) ? dim.nvctx   : dim.nvctx2;
    char *d_comp_buf   = (pipe_idx == 0) ? dim.d_comp  : dim.d_comp2;

    BAMBatchReadEntry *d_ent = (pipe_idx == 0) ? dim.d_batch_entries : dim.d_batch_entries2;

    if (meta.comp_method == CompressionMethod::NONE) {
        size_t batch_kl = 0;
        bam_read_pages_batch_to_gpu_prealloc(dim.io_ctx, static_cast<uint32_t>(page_size),
            meta.data_entries.data(), static_cast<uint32_t>(meta.npages),
            d_out, d_ent, s, &batch_kl);
        kl += batch_kl;
        return;
    }

    // BaM batch read compressed → d_comp_buf
    size_t batch_kl = 0;
    bam_read_pages_batch_to_gpu_prealloc(dim.io_ctx, static_cast<uint32_t>(page_size),
        meta.data_entries.data(), static_cast<uint32_t>(meta.npages),
        d_comp_buf, d_ent, s, &batch_kl);
    kl += batch_kl;

    // nvCOMP decompress → d_out
    for (size_t pg = 0; pg < meta.npages; pg++) {
        nv.h_comp_ptrs[pg]    = d_comp_buf + pg * page_size;
        nv.h_decomp_ptrs[pg]  = d_out + pg * page_size;
        nv.h_comp_sizes[pg]   = meta.comp_page_sizes[pg];
        nv.h_decomp_sizes[pg] = page_size;
    }
    nvcomp_decompctx_run(meta.comp_method, nv, meta.npages, page_size, s);
    kl++;
}

// ============================================================
// Fused dim field read: BaM IO + nvCOMPdx LZ4 decomp in 1 kernel.
// Falls back to bam_dim_read_fast for uncompressed fields.
// ============================================================
static void bam_dim_read_fused(
    DimReadCtx &dim, uint32_t pipe_idx,
    const DimFieldCompMeta &meta, size_t page_size,
    char *d_out, size_t &kl, uint64_t &io_bytes)
{
    if (meta.npages == 0) return;
    for (size_t pg = 0; pg < meta.npages; pg++)
        io_bytes += (uint64_t)meta.data_entries[pg].nblk * 512;

    cudaStream_t s = (pipe_idx == 0) ? dim.stream : dim.stream2;
    BAMBatchReadEntry *d_ent = (pipe_idx == 0) ? dim.d_batch_entries : dim.d_batch_entries2;

    void *d_ctrls   = bam_pfor32_io_get_d_ctrls(dim.io_ctx);
    void *d_pc_ptr  = bam_pfor32_io_get_d_pc_ptr(dim.io_ctx);
    const char *pc_base = bam_pfor32_io_get_pc_base(dim.io_ctx);
    uint32_t num_slots  = bam_pfor32_io_get_num_slots(dim.io_ctx);
    uint32_t half = num_slots / 2;
    uint32_t slot_base = (pipe_idx == 0) ? 0 : half;

    cudaMemcpyAsync(d_ent, meta.data_entries.data(),
        meta.npages * sizeof(BAMBatchReadEntry), cudaMemcpyHostToDevice, s);

    if (meta.comp_method == CompressionMethod::NONE) {
        bam_dim_io_copy_launch(
            d_ctrls, d_pc_ptr, pc_base,
            slot_base, half, static_cast<uint32_t>(page_size),
            d_out, d_ent,
            static_cast<uint32_t>(meta.npages), s);
        kl++;
        return;
    }

    uint32_t *d_csz = (pipe_idx == 0) ? dim.d_comp_sizes : dim.d_comp_sizes2;
    cudaMemcpyAsync(d_csz, meta.comp_page_sizes.data(),
        meta.npages * sizeof(uint32_t), cudaMemcpyHostToDevice, s);

    bam_dim_io_decomp_launch(
        d_ctrls, d_pc_ptr, pc_base,
        slot_base, half, static_cast<uint32_t>(page_size),
        d_out,
        d_ent, d_csz,
        static_cast<uint32_t>(meta.npages), s);
    kl++;
}

// ============================================================
// BaM-based zone map stats reader
// ============================================================
static Stats<int32_t> *bam_ssb_read_column_stats(
    bam_ctrl_handle_t ctrl, uint32_t n_devices, const uint64_t *partition_start_lbas,
    const SSBTableMetadata &metadata, size_t page_size,
    size_t lo_field_idx, uint64_t &out_nstats)
{
    uint64_t nstats = metadata.table_lineorder_nstats[lo_field_idx];
    uint64_t stats_start = metadata.table_lineorder_stats_start_page_ids[lo_field_idx];
    uint64_t stats_npg = metadata.table_lineorder_stats_npages[lo_field_idx];
    out_nstats = nstats;

    if (nstats == 0 || stats_start == 0 || stats_npg == 0) return nullptr;

    void *buf = mb_alloc(stats_npg * page_size);
    for (uint64_t j = 0; j < stats_npg; j++) {
        uint64_t pg_id = stats_start + j;
        uint32_t dev = pg_id % n_devices;
        uint64_t local = pg_id / n_devices;
        uint64_t lba = partition_start_lbas[dev] + local * (page_size / 512);
        bam_read_page(ctrl, page_size, lba, static_cast<char *>(buf) + j * page_size, dev);
    }
    return reinterpret_cast<Stats<int32_t> *>(buf);
}

static Stats<int32_t> *bam_ssb_read_sideways_stats(
    bam_ctrl_handle_t ctrl, uint32_t n_devices, const uint64_t *partition_start_lbas,
    const SSBTableMetadata &metadata, size_t page_size,
    size_t lo_field_idx, size_t sideways_idx, uint64_t &out_nstats)
{
    uint64_t nstats = metadata.table_lineorder_sideways_nstats[lo_field_idx][sideways_idx];
    uint64_t stats_start = metadata.table_lineorder_sideways_stats_start_page_ids[lo_field_idx][sideways_idx];
    uint64_t stats_npg = metadata.table_lineorder_sideways_stats_npages[lo_field_idx][sideways_idx];
    out_nstats = nstats;

    if (nstats == 0 || stats_start == 0 || stats_npg == 0) return nullptr;

    void *buf = mb_alloc(stats_npg * page_size);
    for (uint64_t j = 0; j < stats_npg; j++) {
        uint64_t pg_id = stats_start + j;
        uint32_t dev = pg_id % n_devices;
        uint64_t local = pg_id / n_devices;
        uint64_t lba = partition_start_lbas[dev] + local * (page_size / 512);
        bam_read_page(ctrl, page_size, lba, static_cast<char *>(buf) + j * page_size, dev);
    }
    return reinterpret_cast<Stats<int32_t> *>(buf);
}

// ============================================================
// BaM session context (shared init for all query flights)
// ============================================================
struct BamSessionCtx {
    BAMDeviceSetup ds;
    bam_ctrl_handle_t ctrl;
    size_t page_size;
    uint32_t n_devices;
    std::vector<char> meta_buf;
    uint64_t gpu_ctrl_bytes;
    size_t gpu_free_baseline;  // cudaMemGetInfo before any app alloc (for 40 GiB budget)
    mutable DimReadCtx dim;  // shared page_cache + GPU buffers for dim reads
    bool dim_fused = false;  // true → use fused IO+decomp for compressed dim reads

    const SSBTableMetadata &metadata() const {
        return *reinterpret_cast<const SSBTableMetadata*>(meta_buf.data());
    }
    void *read_dim_pages(SSB::common::Table table, size_t field_idx) const {
        return bam_read_field_pages_cpu(dim,
                                         n_devices, ds.partition_start_lbas,
                                         metadata(), table, field_idx, page_size);
    }
    // Fast read using pre-cached metadata (no NVMe metadata reads during timing)
    const DimFieldCompMeta &cached_meta(SSB::common::Table table, size_t field_idx) const {
        return dim.field_cache.at(dim_field_key(table, field_idx));
    }
    void read_dim_fast(SSB::common::Table table, size_t field_idx,
                       uint32_t pipe, char *d_out, size_t &kl, uint64_t &io_bytes) const {
        bam_dim_read_fast(dim, pipe, cached_meta(table, field_idx), page_size, d_out, kl, io_bytes);
    }
    void read_dim_fused(SSB::common::Table table, size_t field_idx,
                        uint32_t pipe, char *d_out, size_t &kl, uint64_t &io_bytes) const {
        bam_dim_read_fused(dim, pipe, cached_meta(table, field_idx), page_size, d_out, kl, io_bytes);
    }
    void read_dim(SSB::common::Table table, size_t field_idx,
                  uint32_t pipe, char *d_out, size_t &kl, uint64_t &io_bytes) const {
        if (dim_fused)
            bam_dim_read_fused(dim, pipe, cached_meta(table, field_idx), page_size, d_out, kl, io_bytes);
        else
            bam_dim_read_fast(dim, pipe, cached_meta(table, field_idx), page_size, d_out, kl, io_bytes);
    }
};

static BamSessionCtx bam_session_open(BenchmarkOptions &options) {
    BamSessionCtx s{};
    size_t gpu_pre = 0, dummy = 0;
    cudaMemGetInfo(&gpu_pre, &dummy);
    s.gpu_free_baseline = gpu_pre;
    s.ds = bam_open_devices(options.file, 128, 1024);
    s.ctrl = s.ds.ctrl;
    s.n_devices = s.ds.n_devices;
    std::vector<char> hdr(4096);
    bam_read_page(s.ctrl, 4096, s.ds.partition_start_lbas[0], hdr.data(), 0);
    s.page_size = reinterpret_cast<SSBTableMetadata*>(hdr.data())->page_size;
    s.meta_buf.resize(s.page_size);
    // Page 0 on device 0
    bam_read_page(s.ctrl, s.page_size, s.ds.partition_start_lbas[0], s.meta_buf.data(), 0);
    superpage_set_constants_for(s.page_size, sizeof(SSBTableMetadata));
    SSB::metadata_print(s.metadata());
    // ── DimReadCtx: shared page_cache + GPU buffers for dim table reads ──
    // Compute max dim pages from metadata — only for fields used in SSB queries.
    // d_comp/nvctx must hold the largest single field's pages.
    // Using max of query-used fields per table to avoid over-allocating from
    // unused fields like P_TYPE, P_NAME.
    uint32_t DIM_MAX_PAGES = 128;
    {
        using DF = SSB::common::DateField;
        using CF = SSB::common::CustomerField;
        using SF = SSB::common::SupplierField;
        using PF = SSB::common::PartField;
        auto &m = s.metadata();
        auto field_max = [&](const uint64_t *npages, const std::vector<uint32_t> &fields) {
            for (uint32_t fi : fields)
                DIM_MAX_PAGES = std::max(DIM_MAX_PAGES, (uint32_t)npages[fi]);
        };
        field_max(m.table_date_npages,
            {DF::D_DATEKEY, DF::D_YEAR, DF::D_YEARMONTHNUM, DF::D_WEEKNUMINYEAR});
        field_max(m.table_customer_npages,
            {CF::C_CUSTKEY, CF::C_REGION, CF::C_NATION, CF::C_CITY});
        field_max(m.table_supplier_npages,
            {SF::S_SUPPKEY, SF::S_REGION, SF::S_NATION, SF::S_CITY});
        field_max(m.table_part_npages,
            {PF::P_PARTKEY, PF::P_MFGR, PF::P_CATEGORY, PF::P_BRAND1});
    }
    // page_cache only needs enough slots for one batch tile — bam_read_pages_batch_to_gpu
    // tiles reads in chunks of n_slots, so a fixed size (256) works for any DIM_MAX_PAGES.
    constexpr uint32_t DIM_PC_SLOTS = 256;
    auto &d = s.dim;
    d.max_pages = DIM_MAX_PAGES;
    d.io_ctx = bam_pfor32_io_create(s.ctrl, static_cast<uint32_t>(s.page_size), DIM_PC_SLOTS);
    // Pre-allocate per-pipe batch entry buffers (avoid race on 2-pipe overlap)
    uint32_t entries_cap = std::max((uint32_t)DIM_PC_SLOTS, DIM_MAX_PAGES);
    CUDA_CHECK(cudaMalloc(&d.d_batch_entries,  entries_cap * sizeof(BAMBatchReadEntry)));
    CUDA_CHECK(cudaMalloc(&d.d_batch_entries2, entries_cap * sizeof(BAMBatchReadEntry)));
    CUDA_CHECK(cudaMalloc(&d.d_comp_sizes,  DIM_MAX_PAGES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.d_comp_sizes2, DIM_MAX_PAGES * sizeof(uint32_t)));
    // Pipe 0
    CUDA_CHECK(cudaStreamCreate(&d.stream));
    CUDA_CHECK(cudaMalloc(&d.d_comp,   DIM_MAX_PAGES * s.page_size));
    CUDA_CHECK(cudaMalloc(&d.d_decomp, DIM_MAX_PAGES * s.page_size));
    // Pipe 1
    CUDA_CHECK(cudaStreamCreate(&d.stream2));
    CUDA_CHECK(cudaMalloc(&d.d_comp2,  DIM_MAX_PAGES * s.page_size));
    // nvCOMP decompression contexts for both pipes
    auto alloc_nvctx = [&](NvcompDecompCtx &nv) {
        size_t mb = DIM_MAX_PAGES;
        CUDA_CHECK(cudaMalloc(&nv.d_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&nv.d_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMalloc(&nv.d_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&nv.d_decomp_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&nv.d_actual_sizes, mb * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&nv.d_statuses,     mb * sizeof(nvcompStatus_t)));
        CUDA_CHECK(cudaMallocHost(&nv.h_comp_ptrs,    mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&nv.h_decomp_ptrs,  mb * sizeof(void *)));
        CUDA_CHECK(cudaMallocHost(&nv.h_comp_sizes,   mb * sizeof(size_t)));
        CUDA_CHECK(cudaMallocHost(&nv.h_decomp_sizes, mb * sizeof(size_t)));
        size_t max_total = mb * s.page_size;
        size_t snappy_tmp = 0, lz4_tmp = 0;
        nvcompBatchedSnappyDecompressGetTempSizeAsync(
            mb, s.page_size, nvcompBatchedSnappyDecompressDefaultOpts, &snappy_tmp, max_total);
        nvcompBatchedLZ4DecompressGetTempSizeAsync(
            mb, s.page_size, nvcompBatchedLZ4DecompressDefaultOpts, &lz4_tmp, max_total);
        nv.temp_bytes = std::max(snappy_tmp, lz4_tmp);
        if (nv.temp_bytes > 0)
            CUDA_CHECK(cudaMalloc(&nv.d_temp, nv.temp_bytes));
    };
    alloc_nvctx(d.nvctx);
    alloc_nvctx(d.nvctx2);
    // ── Pre-cache compression metadata for all dim fields ──
    {
        auto &m = s.metadata();
        using T = SSB::common::Table;
        const std::pair<T, uint32_t> dim_tables[] = {
            {T::DDATE,    SSB::common::kDateFieldCount},
            {T::CUSTOMER, SSB::common::kCustomerFieldCount},
            {T::SUPPLIER, SSB::common::kSupplierFieldCount},
            {T::PART,     SSB::common::kPartFieldCount},
        };
        for (auto &[tbl, nf] : dim_tables) {
            for (uint32_t fi = 0; fi < nf; fi++) {
                d.field_cache[dim_field_key(tbl, fi)] =
                    bam_precache_field(d, s.n_devices, s.ds.partition_start_lbas,
                                       m, tbl, fi, s.page_size);
            }
        }
    }
    size_t gpu_post = 0;
    cudaMemGetInfo(&gpu_post, &dummy);
    s.gpu_ctrl_bytes = gpu_pre - gpu_post;
    return s;
}

static void bam_session_close(BamSessionCtx &s) {
    auto &d = s.dim;
    nvcomp_decompctx_free(d.nvctx);
    nvcomp_decompctx_free(d.nvctx2);
    if (d.d_batch_entries)  { cudaFree(d.d_batch_entries);  d.d_batch_entries = nullptr; }
    if (d.d_batch_entries2) { cudaFree(d.d_batch_entries2); d.d_batch_entries2 = nullptr; }
    if (d.d_comp_sizes)    { cudaFree(d.d_comp_sizes);    d.d_comp_sizes = nullptr; }
    if (d.d_comp_sizes2)   { cudaFree(d.d_comp_sizes2);   d.d_comp_sizes2 = nullptr; }
    if (d.d_comp)   { cudaFree(d.d_comp);   d.d_comp = nullptr; }
    if (d.d_comp2)  { cudaFree(d.d_comp2);  d.d_comp2 = nullptr; }
    if (d.d_decomp) { cudaFree(d.d_decomp); d.d_decomp = nullptr; }
    if (d.stream)    { cudaStreamDestroy(d.stream); d.stream = nullptr; }
    if (d.stream2)   { cudaStreamDestroy(d.stream2); d.stream2 = nullptr; }
    if (d.io_ctx)    { bam_pfor32_io_destroy(d.io_ctx); d.io_ctx = nullptr; }
    d.field_cache.clear();
    bam_ctrl_close(s.ctrl);
}

// ============================================================
// Dim table GPU buffers (moved before Q1x for shared use)
// ============================================================
struct DimGpuBufs {
    char *d_buf_a;           // reusable page buffer (max_dim_pages)
    char *d_buf_b;           // reusable page buffer (max_dim_pages)
    char *d_buf_c;           // 3rd buffer for 3-field tables
    uint64_t *d_prefix_sum;  // prefix sum (max_dim_pages entries)
    int32_t *d_flat_keys;    // flat INT32 key array
    uint8_t *d_filter;       // flat filter bitmap [nrows]
    int32_t *d_values;       // flat dict values [nrows]
    DimGpuDict dict;         // GPU dict for group-by encoding
    size_t max_nrows;        // allocated capacity
};

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
}

// Forward declarations for DATE HT builders (defined after bam_lo_pipeline)
static void bam_build_date_ht_ext_gpu(GpuHT &ht, const BamSessionCtx &s, DimGpuBufs &db,
    int32_t filter_mode, int32_t filter_lo, int32_t filter_hi, size_t &kl, uint64_t &io_bytes);
static void bam_build_date_ht_q1x_gpu(GpuHT &ht, const BamSessionCtx &s, DimGpuBufs &db,
    SSB::Query query, size_t &kl, uint64_t &io_bytes);

// ============================================================
// SSB Q1.x GIDP+BAM implementation
// Pattern: TPC-H Q6 gidp+bam (NPIPE pipelined BaM IO + nvCOMP decomp + paged kernel)
// ============================================================
static BenchmarkResult ssb_q1x_gidp_bam(
    BenchmarkOptions &options,
    SSB::Query query)
{
    // ── 1. CUDA init + BaM session (shared with Q2x/Q3x/Q4x) ──
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);

    // Aliases for readability (Q1x was written before BamSessionCtx)
    const SSBTableMetadata &metadata = sess.metadata();
    const size_t page_size = sess.page_size;
    const uint32_t n_devices = sess.n_devices;
    const auto &ds = sess.ds;
    auto ctrl = sess.ctrl;
    uint64_t gpu_ctrl_bytes = sess.gpu_ctrl_bytes;
    size_t gpu_total_dummy = 0;

    // ── 4. Extract Q1x LINEORDER field info ──
    constexpr size_t NUM_FIELDS = SSB::query::q1x::NUM_LO_ACTIVE_FIELDS;
    auto q1x_lo_cols = SSB::query::q1x::LO_FIELDS;
    const size_t blocks_per_page = page_size / 512;

    uint64_t field_start_page_ids[NUM_FIELDS];
    uint64_t field_npages_arr[NUM_FIELDS];
    CompressionMethod field_comp_methods[NUM_FIELDS];

    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        size_t col = q1x_lo_cols[fi];
        field_start_page_ids[fi] = metadata.table_lineorder_start_page_ids[col];
        field_npages_arr[fi] = metadata.table_lineorder_npages[col];
        field_comp_methods[fi] = static_cast<CompressionMethod>(
            metadata.table_lineorder_compression_method[col]);
        std::cout << "  LO Field " << col
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << field_npages_arr[fi]
                  << " compression=" << static_cast<int>(field_comp_methods[fi])
                  << std::endl;
    }

    const uint64_t npages = field_npages_arr[0];
    for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
        if (field_npages_arr[fi] != npages) {
            std::cerr << "Error: field " << fi << " has different npages" << std::endl;
            bam_session_close(sess);
            exit(EXIT_FAILURE);
        }
    }
    if (npages == 0) {
        bam_session_close(sess);
        return BenchmarkResult{};
    }

    // ── 5. Read compression metadata via batch BaM reads ──
    std::vector<uint32_t> h_comp_sizes[NUM_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_FIELDS];
    bool any_compressed = false;
    {
        auto make_striped_entries = [&](uint64_t start, size_t np) {
            uint32_t bpp = static_cast<uint32_t>(page_size / 512);
            std::vector<BAMBatchReadEntry> entries(np);
            for (size_t j = 0; j < np; j++) {
                uint64_t pg_id = start + j;
                uint32_t dev = pg_id % n_devices;
                uint64_t local = pg_id / n_devices;
                entries[j] = {ds.partition_start_lbas[dev] + local * bpp, dev, bpp};
            }
            return entries;
        };
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            any_compressed = true;
            size_t col = q1x_lo_cols[fi];
            uint64_t cs_start = metadata.table_lineorder_compressed_page_sizes_start_page_ids[col];
            uint64_t cs_npages_cnt = metadata.table_lineorder_compressed_page_sizes_npages[col];
            uint64_t nbase = metadata.table_lineorder_compression_nbases[col];
            uint64_t base_start = metadata.table_lineorder_compression_base_start_page_ids[col];
            std::vector<char> sizes_buf(cs_npages_cnt * page_size);
            {
                auto entries = make_striped_entries(cs_start, cs_npages_cnt);
                bam_read_pages_batch_to_host(sess.dim.io_ctx, static_cast<uint32_t>(page_size),
                    entries.data(), static_cast<uint32_t>(cs_npages_cnt),
                    sizes_buf.data(), sess.dim.stream);
            }
            h_comp_sizes[fi].assign(
                reinterpret_cast<uint32_t*>(sizes_buf.data()),
                reinterpret_cast<uint32_t*>(sizes_buf.data()) + npages);

            size_t bp_npages = SSB::nbase_to_npages(nbase, page_size);
            std::vector<char> bases_buf(bp_npages * page_size);
            {
                auto entries = make_striped_entries(base_start, bp_npages);
                bam_read_pages_batch_to_host(sess.dim.io_ctx, static_cast<uint32_t>(page_size),
                    entries.data(), static_cast<uint32_t>(bp_npages),
                    bases_buf.data(), sess.dim.stream);
            }
            std::vector<size_t> offsets_vec;
            calculate_compressed_offsets(
                reinterpret_cast<size_t*>(bases_buf.data()),
                h_comp_sizes[fi].data(), nbase, npages, page_size,
                field_start_page_ids[fi], n_devices, offsets_vec);
            h_comp_offsets[fi] = std::move(offsets_vec);
        }
    }

    // ── 6. Allocate DATE hash table + dim bufs (Rule 4: outside timing) ──
    GpuHT date_ht = alloc_gpu_ht(metadata.table_date_nrows);
    uint32_t date_max_np = 0;
    for (size_t i = 0; i < SSB::common::kDateFieldCount; i++)
        date_max_np = std::max(date_max_np, (uint32_t)metadata.table_date_npages[i]);
    if (date_max_np == 0) date_max_np = 1;
    DimGpuBufs dim_bufs_q1 = dim_gpu_bufs_alloc(page_size, metadata.table_date_nrows, date_max_np);
    // Q1.x filter constants
    int32_t disc_lo, disc_hi, qty_lo, qty_hi;
    switch (query) {
        case SSB::Query::Q11: disc_lo = 1; disc_hi = 3; qty_lo = 0; qty_hi = 25; break;
        case SSB::Query::Q12: disc_lo = 4; disc_hi = 6; qty_lo = 26; qty_hi = 36; break;
        case SSB::Query::Q13: disc_lo = 5; disc_hi = 7; qty_lo = 26; qty_hi = 36; break;
        case SSB::Query::REVENUE:
            if (options.disable_other_filters) {
                disc_lo = 0; disc_hi = 99; qty_lo = 0;
                qty_hi = options.revenue_qt_max > 0 ? options.revenue_qt_max : 100;
            } else {
                disc_lo = 1; disc_hi = 3; qty_lo = 0; qty_hi = 25;
            }
            break;
        default: disc_lo = disc_hi = qty_lo = qty_hi = 0; break;
    }

    // ── 7. Zone map metadata (Rule 3: metadata read outside timing) ──
    std::vector<size_t> active_pages;
    bool use_zonemap = options.enable_zonemap;
    int32_t zm_dk_low = 0, zm_dk_high = 0;
    uint64_t zm_odate_nstats = 0, zm_stats_start = 0, zm_stats_npg = 0;
    if (use_zonemap) {
        switch (query) {
            case SSB::Query::Q11: zm_dk_low = 19930101; zm_dk_high = 19931231; break;
            case SSB::Query::Q12: zm_dk_low = 19940101; zm_dk_high = 19940131; break;
            case SSB::Query::Q13: zm_dk_low = 19940201; zm_dk_high = 19940214; break;
            case SSB::Query::REVENUE: zm_dk_low = options.q6_sd_low; zm_dk_high = options.q6_sd_high; break;
            default: zm_dk_low = 19920101; zm_dk_high = 19981231; break;
        }
        zm_odate_nstats = metadata.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
        zm_stats_start = metadata.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
        zm_stats_npg = metadata.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];
    }

    // ── 8. GPU memory allocation (NPIPE pipeline) ──
    size_t gpu_free_before_app = 0;
    cudaMemGetInfo(&gpu_free_before_app, &gpu_total_dummy);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

    constexpr int NPIPE = 4;
    constexpr size_t BATCH_PAGES_MAX = 1536;

    constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
    uint64_t total_used = sess.gpu_free_baseline - gpu_free_before_app;
    uint64_t remaining = (GPU_MEM_BUDGET > total_used) ? GPU_MEM_BUDGET - total_used : 0;
    size_t n_compressed = 0;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
        if (field_comp_methods[fi] != CompressionMethod::NONE) n_compressed++;
    uint64_t fixed_overhead = n_compressed * NPIPE * 2ULL * 1024 * 1024  // nvcomp temp
                            + (uint64_t)NPIPE * sm_count * page_size;    // BaM IO page_cache
    remaining -= std::min(remaining, fixed_overhead);
    size_t bytes_per_batch_page = NPIPE * (NUM_FIELDS + n_compressed) * page_size;
    size_t batch_by_mem = (bytes_per_batch_page > 0) ? (size_t)(remaining / bytes_per_batch_page) : BATCH_PAGES_MAX;
    const size_t BATCH_PAGES = std::min(BATCH_PAGES_MAX, std::max(batch_by_mem, (size_t)1));

    void *data_buf[NPIPE][NUM_FIELDS];
    void *io_buf[NPIPE][NUM_FIELDS];
    memset(io_buf, 0, sizeof(io_buf));
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            data_buf[p][fi] = mb_cuda_alloc(BATCH_PAGES * page_size);
            if (field_comp_methods[fi] != CompressionMethod::NONE)
                io_buf[p][fi] = mb_cuda_alloc(BATCH_PAGES * page_size);
        }

    int64_t *d_revenue = static_cast<int64_t*>(mb_cuda_alloc(sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    // Per-field nvCOMP contexts
    NvcompDecompCtx nvctx[NPIPE][NUM_FIELDS]{};
    if (any_compressed) {
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
            std::vector<FieldPageInfo> tf(1);
            tf[0].compression_method = field_comp_methods[fi];
            for (int p = 0; p < NPIPE; p++)
                nvcomp_decompctx_alloc(nvctx[p][fi], BATCH_PAGES, page_size, tf);
        }
    }

    // Per-pipe BaM I/O contexts
    const uint32_t io_blocks = static_cast<uint32_t>(sm_count);
    BamBulkReadCtx io_ctx[NPIPE];
    for (int p = 0; p < NPIPE; p++)
        io_ctx[p] = bam_bulk_read_ctx_create(
            ctrl, static_cast<uint32_t>(page_size),
            io_blocks,
            static_cast<uint32_t>(BATCH_PAGES * NUM_FIELDS));

    // Zone map ctx: borrows io_ctx[0]'s page_cache (no extra BaM page_cache)
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (use_zonemap && zm_odate_nstats > 0 && zm_stats_start > 0 && zm_stats_npg > 0) {
        zm_ctx = bam_zonemap_ctx_create(
            io_ctx[0].d_ctrls, io_ctx[0].d_pc, io_ctx[0].pc_base,
            static_cast<uint32_t>(page_size), npages);
        for (uint64_t j = 0; j < zm_stats_npg; j++) {
            uint64_t pg_id = zm_stats_start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            zm_ctx.h_reads[j] = {
                ds.partition_start_lbas[dev] + local * (page_size / 512),
                static_cast<uint32_t>(page_size / 512), dev
            };
        }
        zm_nreads = static_cast<uint32_t>(zm_stats_npg);
        zm_ctx.h_preds[0] = {0, zm_odate_nstats, zm_dk_low, zm_dk_high};
        zm_npreds = 1;
        zm_valid = true;
    }

    cudaStream_t stream_io[NPIPE], stream_comp;
    for (int p = 0; p < NPIPE; p++)
        CUDA_CHECK(cudaStreamCreate(&stream_io[p]));
    CUDA_CHECK(cudaStreamCreate(&stream_comp));

    size_t gpu_free_after_app = 0;
    cudaMemGetInfo(&gpu_free_after_app, &gpu_total_dummy);
    uint64_t gpu_app_bytes = gpu_free_before_app - gpu_free_after_app;

    // ── 9. Batch execution setup ──
    auto roundup4096 = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    const size_t LO_ORDERDATE_IDX = 0;
    const size_t LO_QUANTITY_IDX = 1;
    const size_t LO_DISCOUNT_IDX = 2;
    const size_t LO_EXTPRICE_IDX = 3;
    uint32_t capacity = (page_size - 12) / 4;

    const size_t num_batches_all = (npages + BATCH_PAGES - 1) / BATCH_PAGES;
    std::vector<std::vector<size_t>> batch_actives(num_batches_all);
    for (size_t b = 0; b < num_batches_all; b++)
        batch_actives[b].reserve(BATCH_PAGES);
    std::vector<size_t> non_empty_batches;
    non_empty_batches.reserve(num_batches_all);
    active_pages.reserve(npages);
    size_t num_batches = 0;  // set inside timing after zonemap eval

    // Per-batch pinned host arrays for async nvcomp H2D
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

    // build_descs lambda
    auto build_descs = [&](size_t pipe_idx, int buf) -> std::pair<uint32_t, uint64_t> {
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
                    desc.dest = static_cast<char*>(io_buf[buf][fi]) + io_offset;
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
                io_ctx[buf].h_descs[0][ndescs++] = desc;
            }
        }
        return {ndescs, io_bytes};
    };

    // run_decomp_scan lambda
    auto run_decomp_scan = [&](size_t pipe_idx, int buf) {
        size_t batch_idx = non_empty_batches[pipe_idx];
        size_t pg_base = batch_idx * BATCH_PAGES;
        size_t batch_np = std::min(BATCH_PAGES, (size_t)npages - pg_base);
        auto& ba = batch_actives[batch_idx];

        bool batch_partial = use_zonemap && ba.size() < batch_np;
        if (batch_partial && any_compressed) {
            unsigned zblk = (batch_np + 255) / 256;
            for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
                if (field_comp_methods[fi] == CompressionMethod::NONE) continue;
                ssb_zero_page_headers_kernel<<<zblk, 256, 0, stream_comp>>>(
                    static_cast<char*>(data_buf[buf][fi]), batch_np, page_size);
                s_kernel_launches++;
            }
        }

        // nvCOMP decompression
        if (any_compressed) {
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
                            static_cast<char*>(io_buf[buf][fi]) + idx * page_size;
                        pb_h_comp_sizes[fi][slot_base + decomp_count] = comp_sz;
                        pb_h_decomp_ptrs[fi][slot_base + decomp_count] =
                            static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size;
                        pb_h_decomp_sizes[fi][slot_base + decomp_count] = page_size;
                        decomp_count++;
                    } else {
                        CUDA_CHECK(cudaMemcpyAsync(
                            static_cast<char*>(data_buf[buf][fi]) + local_pg * page_size,
                            static_cast<char*>(io_buf[buf][fi]) + idx * page_size,
                            page_size, cudaMemcpyDeviceToDevice, stream_comp));
                    }
                }
                if (decomp_count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(nvctx[buf][fi].d_comp_ptrs,
                        pb_h_comp_ptrs[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx[buf][fi].d_decomp_ptrs,
                        pb_h_decomp_ptrs[fi] + slot_base,
                        decomp_count * sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx[buf][fi].d_comp_sizes,
                        pb_h_comp_sizes[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx[buf][fi].d_decomp_sizes,
                        pb_h_decomp_sizes[fi] + slot_base,
                        decomp_count * sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                    nvcomp_decompctx_run(field_comp_methods[fi], nvctx[buf][fi],
                                         decomp_count, page_size, stream_comp,
                                         /*do_sync=*/false, /*skip_h2d=*/true);
                    s_kernel_launches++;
                }
            }
        }

        // Zero inactive page headers for uncompressed + partial batch
        if (batch_partial && !any_compressed) {
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

        // SSB Q1.x kernel
        ssb_q1x(
            data_buf[buf][LO_ORDERDATE_IDX],
            data_buf[buf][LO_QUANTITY_IDX],
            data_buf[buf][LO_DISCOUNT_IDX],
            data_buf[buf][LO_EXTPRICE_IDX],
            batch_np, page_size,
            (uint64_t)batch_np * capacity,
            date_ht.d_keys, date_ht.d_values, date_ht.mask,
            disc_lo, disc_hi, qty_lo, qty_hi,
            d_revenue, stream_comp);
        s_kernel_launches++;
    };

    // ── 10. Pipelined execution ──
    std::vector<cudaEvent_t> event_io_vec(num_batches_all), event_comp_vec(num_batches_all);
    for (size_t i = 0; i < num_batches_all; i++) {
        CUDA_CHECK(cudaEventCreate(&event_io_vec[i]));
        CUDA_CHECK(cudaEventCreate(&event_comp_vec[i]));
    }


    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream_io[0]);
    }

    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // ── Zone map eval (inside timing, Rule 6) ──
    if (use_zonemap && zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream_io[0]);
        CUDA_CHECK(cudaStreamSynchronize(stream_io[0]));
        s_kernel_launches += 1;
        for (size_t pg = 0; pg < npages; pg++)
            if (zm_ctx.h_mask[pg]) active_pages.push_back(pg);
        std::cout << "[ZONEMAP] LO_ORDERDATE pruning: " << active_pages.size()
                  << " / " << npages << " pages active" << std::endl;
    } else {
        active_pages.resize(npages);
        std::iota(active_pages.begin(), active_pages.end(), size_t(0));
    }

    // Build batch_actives (depends on active_pages from zonemap)
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
    }
    num_batches = non_empty_batches.size();

    std::cout << "[GIDP+BAM Q1x] Pipeline: NPIPE=" << NPIPE
              << " BATCH_PAGES=" << BATCH_PAGES
              << " batches=" << num_batches << "/" << num_batches_all
              << " (zone map pruned)" << std::endl;

    // ── DATE HT build (inside timing interval per Rule 1, GPU path) ──
    uint64_t total_io_count = 0;
    uint64_t total_io_bytes = 0;
    {
        size_t dim_kl = 0;
        uint64_t dim_io = 0;
        bam_build_date_ht_q1x_gpu(date_ht, sess, dim_bufs_q1, query, dim_kl, dim_io);
        s_kernel_launches += dim_kl;
        total_io_bytes += dim_io;
    }

    if (num_batches > 0) {
        // Prime: launch IO for first batch
        {
            auto [ndescs, io_bytes] = build_descs(0, 0);
            bam_bulk_read_async(io_ctx[0], ndescs, 0, stream_io[0]);
            CUDA_CHECK(cudaEventRecord(event_io_vec[0], stream_io[0]));
            total_io_count += ndescs;
            total_io_bytes += io_bytes;
        }

        for (size_t pipe = 0; pipe < num_batches; pipe++) {
            int buf = pipe % NPIPE;
            int next_buf = (pipe + 1) % NPIPE;

            // (a) Launch IO for next batch
            if (pipe + 1 < num_batches) {
                if (pipe + 1 >= (size_t)NPIPE) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream_io[next_buf], event_comp_vec[pipe + 1 - NPIPE]));
                    CUDA_CHECK(cudaEventSynchronize(io_ctx[next_buf].h2d_done[0]));
                }
                auto [ndescs, io_bytes] = build_descs(pipe + 1, next_buf);
                bam_bulk_read_async(io_ctx[next_buf], ndescs, 0, stream_io[next_buf]);
                CUDA_CHECK(cudaEventRecord(event_io_vec[pipe + 1], stream_io[next_buf]));
                total_io_count += ndescs;
                total_io_bytes += io_bytes;
            }

            // (b) Wait for IO, decomp + scan
            CUDA_CHECK(cudaStreamWaitEvent(stream_comp, event_io_vec[pipe]));
            run_decomp_scan(pipe, buf);
            CUDA_CHECK(cudaEventRecord(event_comp_vec[pipe], stream_comp));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_comp));
    }

    // ── 11. Result ──
    int64_t h_revenue = 0;
    CUDA_CHECK(cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::cout << "SSB Q1.x revenue: " << h_revenue << std::endl;

    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total elapsed: " << elapsed << " seconds" << std::endl;
    std::cout << "Total I/Os: " << total_io_count << std::endl;
    std::cout << "Total bytes read: " << total_io_bytes << std::endl;
    std::cout << "========================================" << std::endl;

    // ── 12. Cleanup ──
    date_ht.free_all();
    if (any_compressed) {
        for (int p = 0; p < NPIPE; p++)
            for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                if (field_comp_methods[fi] != CompressionMethod::NONE)
                    nvcomp_decompctx_free(nvctx[p][fi]);
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (pb_h_comp_ptrs[fi])    cudaFreeHost(pb_h_comp_ptrs[fi]);
            if (pb_h_decomp_ptrs[fi])  cudaFreeHost(pb_h_decomp_ptrs[fi]);
            if (pb_h_comp_sizes[fi])   cudaFreeHost(pb_h_comp_sizes[fi]);
            if (pb_h_decomp_sizes[fi]) cudaFreeHost(pb_h_decomp_sizes[fi]);
        }
    }
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            mb_cuda_free(data_buf[p][fi]);
            if (io_buf[p][fi]) mb_cuda_free(io_buf[p][fi]);
        }
    mb_cuda_free(d_revenue);
    for (int p = 0; p < NPIPE; p++)
        bam_bulk_read_ctx_destroy(io_ctx[p]);
    for (size_t i = 0; i < num_batches_all; i++) {
        CUDA_CHECK(cudaEventDestroy(event_io_vec[i]));
        CUDA_CHECK(cudaEventDestroy(event_comp_vec[i]));
    }
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    dim_gpu_bufs_free(dim_bufs_q1);
    for (int p = 0; p < NPIPE; p++)
        CUDA_CHECK(cudaStreamDestroy(stream_io[p]));
    CUDA_CHECK(cudaStreamDestroy(stream_comp));
    bam_session_close(sess);

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
// Paged probe kernels for Q2x/Q3x/Q4x
// ============================================================

static __global__ void ssb_q2x_paged_kernel(
    void *lo_od, void *lo_pk, void *lo_sk, void *lo_rev,
    uint64_t nrecs, uint32_t cap, uint32_t ps,
    const int32_t *d_dk, const int32_t *d_dv, uint32_t dm,
    const int32_t *dsk, const int32_t *dsv, uint32_t sm,
    const int32_t *dpk, const int32_t *dpv, uint32_t pm, int64_t *d_out)
{
    constexpr uint32_t HIST_SIZE = SSB_NUM_YEARS * SSB_MAX_BRANDS;  // 280
    __shared__ int64_t s_hist[HIST_SIZE];
    for (uint32_t i = threadIdx.x; i < HIST_SIZE; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrecs) {
        size_t pi = idx / cap, ri = idx % cap;
        uint32_t na = reinterpret_cast<pag_head*>((char*)lo_od + pi*ps)->nalloc;
        if (ri < na) {
            int32_t od = ((int32_t*)((uint8_t*)lo_od + pi*ps))[3+ri];
            int32_t pk = ((int32_t*)((uint8_t*)lo_pk + pi*ps))[3+ri];
            int32_t sk = ((int32_t*)((uint8_t*)lo_sk + pi*ps))[3+ri];
            int32_t rv = ((int32_t*)((uint8_t*)lo_rev + pi*ps))[3+ri];
            int32_t yi = ssb_ht_probe(d_dk, d_dv, dm, od);
            if (yi >= 0) {
                int32_t sv = ssb_ht_probe(dsk, dsv, sm, sk);
                if (sv >= 0) {
                    int32_t bi = ssb_ht_probe(dpk, dpv, pm, pk);
                    if (bi >= 0) {
                        atomicAdd((unsigned long long int*)&s_hist[yi * SSB_MAX_BRANDS + bi],
                                  (unsigned long long int)(int64_t)rv);
                    }
                }
            }
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < HIST_SIZE; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd((unsigned long long int*)&d_out[i],
                      (unsigned long long int)s_hist[i]);
    }
}

static void ssb_q2x_paged(void *d0, void *d1, void *d2, void *d3,
    uint32_t np, uint32_t ps,
    const int32_t *dk, const int32_t *dv, uint32_t dm,
    const int32_t *sk, const int32_t *sv, uint32_t sm,
    const int32_t *pk, const int32_t *pv, uint32_t pm,
    int64_t *out, cudaStream_t s) {
    uint32_t cap = (ps - 12) / 4;
    uint64_t nr = (uint64_t)np * cap;
    ssb_q2x_paged_kernel<<<(nr+127)/128, 128, 0, s>>>(
        d0, d1, d2, d3, nr, cap, ps, dk, dv, dm, sk, sv, sm, pk, pv, pm, out);
    s_kernel_launches++;
}

static __global__ void ssb_q3x_paged_kernel(
    void *lo_od, void *lo_ck, void *lo_sk, void *lo_rev,
    uint64_t nrecs, uint32_t cap, uint32_t ps,
    const int32_t *d_dk, const int32_t *d_dv, uint32_t d_dm,
    const int32_t *dck, const int32_t *dcv, uint32_t cm,
    const int32_t *dsk, const int32_t *dsv, uint32_t sm,
    int32_t nsd, uint32_t hist_size, int64_t *d_out)
{
    extern __shared__ int64_t s_hist[];
    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrecs) {
        size_t pi = idx / cap, ri = idx % cap;
        uint32_t na = reinterpret_cast<pag_head*>((char*)lo_od + pi*ps)->nalloc;
        if (ri < na) {
            int32_t od = ((int32_t*)((uint8_t*)lo_od + pi*ps))[3+ri];
            int32_t ck = ((int32_t*)((uint8_t*)lo_ck + pi*ps))[3+ri];
            int32_t sk = ((int32_t*)((uint8_t*)lo_sk + pi*ps))[3+ri];
            int32_t rv = ((int32_t*)((uint8_t*)lo_rev + pi*ps))[3+ri];
            int32_t yi = ssb_ht_probe(d_dk, d_dv, d_dm, od);
            if (yi >= 0) {
                int32_t cv = ssb_ht_probe(dck, dcv, cm, ck);
                if (cv >= 0) {
                    int32_t sv = ssb_ht_probe(dsk, dsv, sm, sk);
                    if (sv >= 0) {
                        atomicAdd((unsigned long long int*)&s_hist[cv * nsd * SSB_Q3X_MAX_YEARS + sv * SSB_Q3X_MAX_YEARS + yi],
                                  (unsigned long long int)(int64_t)rv);
                    }
                }
            }
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd((unsigned long long int*)&d_out[i],
                      (unsigned long long int)s_hist[i]);
    }
}

static void ssb_q3x_paged(void *d0, void *d1, void *d2, void *d3,
    uint32_t np, uint32_t ps,
    const int32_t *dk, const int32_t *dv, uint32_t dm,
    const int32_t *ck, const int32_t *cv, uint32_t cm,
    const int32_t *sk, const int32_t *sv, uint32_t sm,
    int32_t nsd, int32_t ncd, int64_t *out, cudaStream_t s) {
    uint32_t cap = (ps - 12) / 4;
    uint64_t nr = (uint64_t)np * cap;
    uint32_t hist_size = (uint32_t)(ncd * nsd * SSB_Q3X_MAX_YEARS);
    size_t smem = hist_size * sizeof(int64_t);
    ssb_q3x_paged_kernel<<<(nr+127)/128, 128, smem, s>>>(
        d0, d1, d2, d3, nr, cap, ps, dk, dv, dm, ck, cv, cm, sk, sv, sm, nsd, hist_size, out);
    s_kernel_launches++;
}

static __global__ void ssb_q4x_paged_kernel(
    void *lo_od, void *lo_ck, void *lo_pk, void *lo_sk,
    void *lo_rev, void *lo_sc,
    uint64_t nrecs, uint32_t cap, uint32_t ps,
    const int32_t *d_dk, const int32_t *d_dv, uint32_t d_dm,
    const int32_t *dck, const int32_t *dcv, uint32_t cm,
    const int32_t *dsk, const int32_t *dsv, uint32_t sm,
    const int32_t *dpk, const int32_t *dpv, uint32_t pm,
    int32_t nsd, int32_t npd, int32_t sy, uint32_t hist_size, int64_t *d_out)
{
    extern __shared__ int64_t s_hist[];
    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrecs) {
        size_t pi = idx / cap, ri = idx % cap;
        uint32_t na = reinterpret_cast<pag_head*>((char*)lo_od + pi*ps)->nalloc;
        if (ri < na) {
            int32_t od = ((int32_t*)((uint8_t*)lo_od + pi*ps))[3+ri];
            int32_t ck = ((int32_t*)((uint8_t*)lo_ck + pi*ps))[3+ri];
            int32_t pk = ((int32_t*)((uint8_t*)lo_pk + pi*ps))[3+ri];
            int32_t sk = ((int32_t*)((uint8_t*)lo_sk + pi*ps))[3+ri];
            int32_t rv = ((int32_t*)((uint8_t*)lo_rev + pi*ps))[3+ri];
            int32_t sc = ((int32_t*)((uint8_t*)lo_sc + pi*ps))[3+ri];
            int32_t yi = ssb_ht_probe(d_dk, d_dv, d_dm, od);
            if (yi >= 0) {
                int32_t cvl = ssb_ht_probe(dck, dcv, cm, ck);
                if (cvl >= 0) {
                    int32_t svl = ssb_ht_probe(dsk, dsv, sm, sk);
                    if (svl >= 0) {
                        int32_t pvl = ssb_ht_probe(dpk, dpv, pm, pk);
                        if (pvl >= 0) {
                            int32_t gi = yi * sy + cvl * (nsd * npd) + svl * npd + pvl;
                            atomicAdd((unsigned long long int*)&s_hist[gi],
                                      (unsigned long long int)((int64_t)rv - (int64_t)sc));
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd((unsigned long long int*)&d_out[i],
                      (unsigned long long int)s_hist[i]);
    }
}

static void ssb_q4x_paged(void *d0, void *d1, void *d2, void *d3, void *d4, void *d5,
    uint32_t np, uint32_t ps,
    const int32_t *dk, const int32_t *dv, uint32_t dm,
    const int32_t *ck, const int32_t *cv, uint32_t cm,
    const int32_t *sk, const int32_t *sv, uint32_t sm,
    const int32_t *pk, const int32_t *pv, uint32_t pm,
    int32_t nsd, int32_t npd, int32_t sy, int32_t total_groups,
    int64_t *out, cudaStream_t s) {
    uint32_t cap = (ps - 12) / 4;
    uint64_t nr = (uint64_t)np * cap;
    uint32_t hist_size = (uint32_t)total_groups;
    size_t smem = (size_t)hist_size * sizeof(int64_t);
    ssb_q4x_paged_kernel<<<(nr+127)/128, 128, smem, s>>>(
        d0, d1, d2, d3, d4, d5, nr, cap, ps, dk, dv, dm,
        ck, cv, cm, sk, sv, sm, pk, pv, pm, nsd, npd, sy, hist_size, out);
    s_kernel_launches++;
}

// ============================================================
// BaM LO pipeline — pre-allocated context (Rule 4)
// ============================================================
template <size_t NUM_FIELDS>
struct BamLoPipeCtx {
    static constexpr int NPIPE = 4;
    static constexpr size_t BATCH_MAX = 1536;

    size_t batch_pages = 0;

    void *data_buf[NPIPE][NUM_FIELDS] = {};
    void *io_buf[NPIPE][NUM_FIELDS] = {};
    NvcompDecompCtx nvctx[NPIPE][NUM_FIELDS] = {};
    BamBulkReadCtx io_ctx[NPIPE] = {};
    cudaStream_t stream_io[NPIPE] = {};
    cudaStream_t stream_comp = nullptr;

    void **pb_cp[NUM_FIELDS] = {};
    void **pb_dp[NUM_FIELDS] = {};
    size_t *pb_cs[NUM_FIELDS] = {};
    size_t *pb_ds[NUM_FIELDS] = {};
    size_t pb_total_slots = 0;

    cudaEvent_t *ev_io = nullptr;
    cudaEvent_t *ev_comp = nullptr;
    size_t max_events = 0;

    bool any_compressed = false;
    CompressionMethod field_comp[NUM_FIELDS] = {};
    uint64_t gpu_app_bytes = 0;

    // Compression metadata (Rule 3: pre-read outside timing)
    std::vector<uint32_t> h_comp_sizes[NUM_FIELDS];
    std::vector<size_t> h_comp_offsets[NUM_FIELDS];

    // pg_act buffer (Rule 4: pre-allocated outside timing)
    bool *pg_act_buf = nullptr;
};

template <size_t NUM_FIELDS>
static void bam_lo_pipe_ctx_alloc(
    BamLoPipeCtx<NUM_FIELDS> &ctx,
    const BamSessionCtx &sess,
    const std::array<SSB::common::LineOrderField, NUM_FIELDS> &lo_cols,
    uint64_t npages, uint64_t extra_gpu_bytes)
{
    constexpr int NPIPE = BamLoPipeCtx<NUM_FIELDS>::NPIPE;
    constexpr size_t BATCH_MAX = BamLoPipeCtx<NUM_FIELDS>::BATCH_MAX;
    const size_t page_size = sess.page_size;
    auto ctrl = sess.ctrl;
    auto &metadata = sess.metadata();

    // Determine field compression
    ctx.any_compressed = false;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        size_t col = lo_cols[fi];
        ctx.field_comp[fi] = static_cast<CompressionMethod>(
            metadata.table_lineorder_compression_method[col]);
        if (ctx.field_comp[fi] != CompressionMethod::NONE) ctx.any_compressed = true;
    }

    // Compute BATCH_PAGES (40 GiB budget)
    size_t gpu_free_before = 0, dummy = 0;
    cudaMemGetInfo(&gpu_free_before, &dummy);
    constexpr uint64_t GPU_MEM_BUDGET = 40ULL * 1024 * 1024 * 1024;
    uint64_t total_used = sess.gpu_free_baseline - gpu_free_before;
    uint64_t remaining = (GPU_MEM_BUDGET > total_used) ? GPU_MEM_BUDGET - total_used : 0;
    size_t n_comp = 0;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
        if (ctx.field_comp[fi] != CompressionMethod::NONE) n_comp++;
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    uint64_t fixed_overhead = n_comp * NPIPE * 2ULL * 1024 * 1024  // nvcomp temp
                            + (uint64_t)NPIPE * sm_count * page_size;  // BaM IO page_cache
    remaining -= std::min(remaining, fixed_overhead);
    size_t per_batch = NPIPE * (NUM_FIELDS + n_comp) * page_size;
    size_t batch_by_mem = (per_batch > 0) ? (size_t)(remaining / per_batch) : BATCH_MAX;
    ctx.batch_pages = std::min(BATCH_MAX, std::max(batch_by_mem, (size_t)1));

    // Data + IO buffers
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            ctx.data_buf[p][fi] = mb_cuda_alloc(ctx.batch_pages * page_size);
            if (ctx.field_comp[fi] != CompressionMethod::NONE)
                ctx.io_buf[p][fi] = mb_cuda_alloc(ctx.batch_pages * page_size);
        }

    // nvCOMP contexts
    if (ctx.any_compressed)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (ctx.field_comp[fi] == CompressionMethod::NONE) continue;
            std::vector<FieldPageInfo> tf(1);
            tf[0].compression_method = ctx.field_comp[fi];
            for (int p = 0; p < NPIPE; p++)
                nvcomp_decompctx_alloc(ctx.nvctx[p][fi], ctx.batch_pages, page_size, tf);
        }

    // BaM bulk read contexts
    for (int p = 0; p < NPIPE; p++)
        ctx.io_ctx[p] = bam_bulk_read_ctx_create(ctrl, (uint32_t)page_size,
            (uint32_t)sm_count, (uint32_t)(ctx.batch_pages * NUM_FIELDS));

    // Streams
    for (int p = 0; p < NPIPE; p++) CUDA_CHECK(cudaStreamCreate(&ctx.stream_io[p]));
    CUDA_CHECK(cudaStreamCreate(&ctx.stream_comp));

    // Pre-batch nvcomp pointer arrays (worst-case: all batches)
    size_t max_batches = (npages + ctx.batch_pages - 1) / ctx.batch_pages;
    ctx.pb_total_slots = max_batches * ctx.batch_pages;
    if (ctx.any_compressed && ctx.pb_total_slots > 0)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (ctx.field_comp[fi] == CompressionMethod::NONE) continue;
            CUDA_CHECK(cudaMallocHost(&ctx.pb_cp[fi], ctx.pb_total_slots * sizeof(void*)));
            CUDA_CHECK(cudaMallocHost(&ctx.pb_dp[fi], ctx.pb_total_slots * sizeof(void*)));
            CUDA_CHECK(cudaMallocHost(&ctx.pb_cs[fi], ctx.pb_total_slots * sizeof(size_t)));
            CUDA_CHECK(cudaMallocHost(&ctx.pb_ds[fi], ctx.pb_total_slots * sizeof(size_t)));
        }

    // Events (worst-case)
    ctx.max_events = max_batches;
    if (max_batches > 0) {
        ctx.ev_io = new cudaEvent_t[max_batches];
        ctx.ev_comp = new cudaEvent_t[max_batches];
        for (size_t i = 0; i < max_batches; i++) {
            CUDA_CHECK(cudaEventCreate(&ctx.ev_io[i]));
            CUDA_CHECK(cudaEventCreate(&ctx.ev_comp[i]));
        }
    }

    // pg_act buffer for zone map partial batches (Rule 4)
    ctx.pg_act_buf = (bool *)malloc(BATCH_MAX * sizeof(bool));

    // GPU memory usage
    size_t gpu_free_after = 0;
    cudaMemGetInfo(&gpu_free_after, &dummy);
    ctx.gpu_app_bytes = gpu_free_before - gpu_free_after + extra_gpu_bytes;
}

template <size_t NUM_FIELDS>
static void bam_lo_pipe_ctx_free(BamLoPipeCtx<NUM_FIELDS> &ctx)
{
    constexpr int NPIPE = BamLoPipeCtx<NUM_FIELDS>::NPIPE;
    if (ctx.any_compressed) {
        for (int p = 0; p < NPIPE; p++)
            for (size_t fi = 0; fi < NUM_FIELDS; fi++)
                if (ctx.field_comp[fi] != CompressionMethod::NONE)
                    nvcomp_decompctx_free(ctx.nvctx[p][fi]);
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            if (ctx.pb_cp[fi]) cudaFreeHost(ctx.pb_cp[fi]);
            if (ctx.pb_dp[fi]) cudaFreeHost(ctx.pb_dp[fi]);
            if (ctx.pb_cs[fi]) cudaFreeHost(ctx.pb_cs[fi]);
            if (ctx.pb_ds[fi]) cudaFreeHost(ctx.pb_ds[fi]);
        }
    }
    for (int p = 0; p < NPIPE; p++)
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            mb_cuda_free(ctx.data_buf[p][fi]);
            if (ctx.io_buf[p][fi]) mb_cuda_free(ctx.io_buf[p][fi]);
        }
    for (int p = 0; p < NPIPE; p++) bam_bulk_read_ctx_destroy(ctx.io_ctx[p]);
    for (size_t i = 0; i < ctx.max_events; i++) {
        CUDA_CHECK(cudaEventDestroy(ctx.ev_io[i]));
        CUDA_CHECK(cudaEventDestroy(ctx.ev_comp[i]));
    }
    delete[] ctx.ev_io;
    delete[] ctx.ev_comp;
    for (int p = 0; p < NPIPE; p++) CUDA_CHECK(cudaStreamDestroy(ctx.stream_io[p]));
    CUDA_CHECK(cudaStreamDestroy(ctx.stream_comp));
    if (ctx.pg_act_buf) { free(ctx.pg_act_buf); ctx.pg_act_buf = nullptr; }
}

// Read compression metadata into ctx (Rule 3: outside timing)
template <size_t NUM_FIELDS>
static void bam_lo_pipe_ctx_read_comp_meta(
    BamLoPipeCtx<NUM_FIELDS> &ctx,
    const BamSessionCtx &sess,
    const std::array<SSB::common::LineOrderField, NUM_FIELDS> &lo_cols)
{
    if (!ctx.any_compressed) return;
    auto &metadata = sess.metadata();
    const size_t page_size = sess.page_size;
    const uint32_t n_devices = sess.n_devices;
    auto &ds = sess.ds;

    uint64_t field_start[NUM_FIELDS];
    for (size_t fi = 0; fi < NUM_FIELDS; fi++)
        field_start[fi] = metadata.table_lineorder_start_page_ids[lo_cols[fi]];
    uint64_t npages = metadata.table_lineorder_npages[lo_cols[0]];

    auto make_striped_entries = [&](uint64_t start, size_t np) {
        uint32_t bpp = static_cast<uint32_t>(page_size / 512);
        std::vector<BAMBatchReadEntry> entries(np);
        for (size_t j = 0; j < np; j++) {
            uint64_t pg_id = start + j;
            uint32_t dev = pg_id % n_devices;
            uint64_t local = pg_id / n_devices;
            entries[j] = {ds.partition_start_lbas[dev] + local * bpp, dev, bpp};
        }
        return entries;
    };

    auto meta_start = std::chrono::steady_clock::now();
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        if (ctx.field_comp[fi] == CompressionMethod::NONE) continue;
        size_t col = lo_cols[fi];
        uint64_t cs_start = metadata.table_lineorder_compressed_page_sizes_start_page_ids[col];
        uint64_t cs_np = metadata.table_lineorder_compressed_page_sizes_npages[col];
        uint64_t nbase = metadata.table_lineorder_compression_nbases[col];
        uint64_t base_start = metadata.table_lineorder_compression_base_start_page_ids[col];
        std::vector<char> sb(cs_np * page_size);
        {
            auto entries = make_striped_entries(cs_start, cs_np);
            bam_read_pages_batch_to_host(sess.dim.io_ctx, static_cast<uint32_t>(page_size),
                entries.data(), static_cast<uint32_t>(cs_np),
                sb.data(), sess.dim.stream);
        }
        ctx.h_comp_sizes[fi].assign(reinterpret_cast<uint32_t*>(sb.data()),
                                     reinterpret_cast<uint32_t*>(sb.data()) + npages);
        size_t bp_np = SSB::nbase_to_npages(nbase, page_size);
        std::vector<char> bb(bp_np * page_size);
        {
            auto entries = make_striped_entries(base_start, bp_np);
            bam_read_pages_batch_to_host(sess.dim.io_ctx, static_cast<uint32_t>(page_size),
                entries.data(), static_cast<uint32_t>(bp_np),
                bb.data(), sess.dim.stream);
        }
        std::vector<size_t> offsets;
        calculate_compressed_offsets(reinterpret_cast<size_t*>(bb.data()),
            ctx.h_comp_sizes[fi].data(), nbase, npages, page_size,
            field_start[fi], n_devices, offsets);
        ctx.h_comp_offsets[fi] = std::move(offsets);
    }
    auto meta_end = std::chrono::steady_clock::now();
    std::cout << "[LO-TIMING] comp_metadata="
              << std::chrono::duration<double, std::milli>(meta_end - meta_start).count()
              << "ms" << std::endl;
}

// ============================================================
// BaM LINEORDER pipeline template
// ============================================================
template <size_t NUM_FIELDS, typename ScanBatchFn>
static BenchmarkResult bam_lo_pipeline(
    BenchmarkOptions &options,
    const BamSessionCtx &sess,
    const std::array<SSB::common::LineOrderField, NUM_FIELDS> &lo_cols,
    const std::vector<bool> &caller_page_active,
    BamLoPipeCtx<NUM_FIELDS> &ctx,
    ScanBatchFn scan_batch_fn)
{
    auto ctrl = sess.ctrl;
    auto &ds = sess.ds;
    auto &metadata = sess.metadata();
    const size_t page_size = sess.page_size;
    const uint32_t n_devices = sess.n_devices;
    const size_t blocks_per_page = page_size / 512;

    // ── Extract LO field info (field_comp/any_compressed from pre-allocated ctx) ──
    uint64_t field_start[NUM_FIELDS], field_np[NUM_FIELDS];
    CompressionMethod (&field_comp)[NUM_FIELDS] = ctx.field_comp;
    const bool any_compressed = ctx.any_compressed;
    for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
        size_t col = lo_cols[fi];
        field_start[fi] = metadata.table_lineorder_start_page_ids[col];
        field_np[fi] = metadata.table_lineorder_npages[col];
        std::cout << "  LO Field " << col
                  << ": start=" << field_start[fi] << " np=" << field_np[fi]
                  << " comp=" << (int)field_comp[fi] << std::endl;
    }
    uint64_t npages = field_np[0];
    for (size_t fi = 1; fi < NUM_FIELDS; fi++) {
        if (field_np[fi] != npages) {
            std::cerr << "Error: field " << fi << " npages mismatch" << std::endl;
            return BenchmarkResult{};
        }
    }
    if (npages == 0) return BenchmarkResult{};

    // Compression metadata already in ctx (read by bam_lo_pipe_ctx_read_comp_meta)

    // ── Zone map (caller-provided page_active) ──
    std::vector<size_t> active_pages;
    bool use_zonemap = options.enable_zonemap;
    if (use_zonemap && caller_page_active.size() == npages) {
        for (size_t pg = 0; pg < npages; pg++)
            if (caller_page_active[pg]) active_pages.push_back(pg);
        std::cout << "[ZONEMAP] " << active_pages.size() << "/" << npages << std::endl;
    } else {
        active_pages.resize(npages);
        std::iota(active_pages.begin(), active_pages.end(), size_t(0));
    }
    if (active_pages.empty()) return BenchmarkResult{};

    // ── Use pre-allocated context (Rule 4) ──
    constexpr int NPIPE = BamLoPipeCtx<NUM_FIELDS>::NPIPE;
    const size_t BATCH_PAGES = ctx.batch_pages;
    void* (&data_buf)[NPIPE][NUM_FIELDS] = ctx.data_buf;
    void* (&io_buf)[NPIPE][NUM_FIELDS] = ctx.io_buf;
    NvcompDecompCtx (&nvctx)[NPIPE][NUM_FIELDS] = ctx.nvctx;
    BamBulkReadCtx (&io_ctx)[NPIPE] = ctx.io_ctx;
    cudaStream_t (&stream_io)[NPIPE] = ctx.stream_io;
    cudaStream_t &stream_comp = ctx.stream_comp;
    void** (&pb_cp)[NUM_FIELDS] = ctx.pb_cp;
    void** (&pb_dp)[NUM_FIELDS] = ctx.pb_dp;
    size_t* (&pb_cs)[NUM_FIELDS] = ctx.pb_cs;
    size_t* (&pb_ds)[NUM_FIELDS] = ctx.pb_ds;
    uint64_t gpu_app_bytes = ctx.gpu_app_bytes;

    // ── Batch setup ──
    auto roundup4096_fn = [](size_t v) -> size_t {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };
    uint32_t capacity = (page_size - 12) / 4;

    size_t total_active = active_pages.size();
    size_t num_batches = (total_active + BATCH_PAGES - 1) / BATCH_PAGES;
    std::vector<std::vector<size_t>> batch_actives(num_batches);
    for (size_t b = 0; b < num_batches; b++) {
        size_t start = b * BATCH_PAGES;
        size_t end = std::min(start + BATCH_PAGES, total_active);
        for (size_t i = start; i < end; i++)
            batch_actives[b].push_back(active_pages[i]);
    }
    std::vector<size_t> non_empty_batches(num_batches);
    std::iota(non_empty_batches.begin(), non_empty_batches.end(), size_t(0));

    std::cout << "[GIDP+BAM] NPIPE=" << NPIPE << " BATCH=" << BATCH_PAGES
              << " batches=" << num_batches << " active=" << total_active << "/" << npages << std::endl;

    // ── Lambdas ──
    auto build_descs = [&](size_t pipe_idx, int buf) -> std::pair<uint32_t, uint64_t> {
        size_t bi = non_empty_batches[pipe_idx];
        auto &ba = batch_actives[bi];
        uint32_t ndescs = 0; uint64_t io_bytes = 0;
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
            bool is_comp = (field_comp[fi] != CompressionMethod::NONE);
            size_t io_off = 0;
            for (size_t idx = 0; idx < ba.size(); idx++) {
                size_t pg = ba[idx];
                BamBulkReadDesc desc{};
                if (is_comp) {
                    uint64_t boff = ctx.h_comp_offsets[fi][pg];
                    uint64_t pid = field_start[fi] + pg;
                    uint32_t dev = pid % n_devices;
                    desc.lba = ds.partition_start_lbas[dev] + boff / 512;
                    uint32_t csz = ctx.h_comp_sizes[fi][pg];
                    desc.nblocks = bam_safe_nblocks((roundup4096_fn(csz) + 511) / 512);
                    desc.device = dev;
                    desc.dest = static_cast<char*>(io_buf[buf][fi]) + io_off;
                    desc.copy_bytes = csz;
                    io_off += page_size;
                } else {
                    uint64_t pid = field_start[fi] + pg;
                    uint32_t dev = pid % n_devices;
                    uint64_t lpd = pid / n_devices;
                    desc.lba = ds.partition_start_lbas[dev] + lpd * blocks_per_page;
                    desc.nblocks = blocks_per_page;
                    desc.device = dev;
                    desc.dest = static_cast<char*>(data_buf[buf][fi]) + idx * page_size;
                    desc.copy_bytes = page_size;
                }
                io_bytes += (uint64_t)desc.nblocks * 512;
                io_ctx[buf].h_descs[0][ndescs++] = desc;
            }
        }
        return {ndescs, io_bytes};
    };

    auto run_decomp_scan = [&](size_t pipe_idx, int buf) {
        size_t bi = non_empty_batches[pipe_idx];
        auto &ba = batch_actives[bi];
        size_t batch_np = ba.size();

        if (any_compressed) {
            size_t slot_base = pipe_idx * BATCH_PAGES;
            for (size_t fi = 0; fi < NUM_FIELDS; fi++) {
                if (field_comp[fi] == CompressionMethod::NONE) continue;
                size_t cnt = 0;
                for (size_t idx = 0; idx < ba.size(); idx++) {
                    size_t pg = ba[idx];
                    uint32_t csz = ctx.h_comp_sizes[fi][pg];
                    if (csz < page_size) {
                        pb_cp[fi][slot_base+cnt] = static_cast<char*>(io_buf[buf][fi]) + idx*page_size;
                        pb_cs[fi][slot_base+cnt] = csz;
                        pb_dp[fi][slot_base+cnt] = static_cast<char*>(data_buf[buf][fi]) + idx*page_size;
                        pb_ds[fi][slot_base+cnt] = page_size;
                        cnt++;
                    } else {
                        CUDA_CHECK(cudaMemcpyAsync(
                            static_cast<char*>(data_buf[buf][fi]) + idx*page_size,
                            static_cast<char*>(io_buf[buf][fi]) + idx*page_size,
                            page_size, cudaMemcpyDeviceToDevice, stream_comp));
                    }
                }
                if (cnt > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(nvctx[buf][fi].d_comp_ptrs,
                        pb_cp[fi]+slot_base, cnt*sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx[buf][fi].d_decomp_ptrs,
                        pb_dp[fi]+slot_base, cnt*sizeof(void*), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx[buf][fi].d_comp_sizes,
                        pb_cs[fi]+slot_base, cnt*sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                    CUDA_CHECK(cudaMemcpyAsync(nvctx[buf][fi].d_decomp_sizes,
                        pb_ds[fi]+slot_base, cnt*sizeof(size_t), cudaMemcpyHostToDevice, stream_comp));
                    nvcomp_decompctx_run(field_comp[fi], nvctx[buf][fi], cnt, page_size, stream_comp, false, true);
                    s_kernel_launches++;
                }
            }
        }

        void *db[NUM_FIELDS];
        for (size_t fi = 0; fi < NUM_FIELDS; fi++) db[fi] = data_buf[buf][fi];
        scan_batch_fn(db, batch_np, (uint64_t)batch_np * capacity, stream_comp);
    };

    // ── Pipeline execution ──
    auto *ev_io = ctx.ev_io;
    auto *ev_comp = ctx.ev_comp;

    auto total_start = std::chrono::steady_clock::now();
    uint64_t total_io_count = 0, total_io_bytes = 0;

    if (num_batches > 0) {
        { auto [nd, ib] = build_descs(0, 0);
          bam_bulk_read_async(io_ctx[0], nd, 0, stream_io[0]);
          CUDA_CHECK(cudaEventRecord(ev_io[0], stream_io[0]));
          total_io_count += nd; total_io_bytes += ib; }
        for (size_t pipe = 0; pipe < num_batches; pipe++) {
            int buf = pipe % NPIPE, next_buf = (pipe + 1) % NPIPE;
            if (pipe + 1 < num_batches) {
                if (pipe + 1 >= (size_t)NPIPE) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream_io[next_buf], ev_comp[pipe+1-NPIPE]));
                    CUDA_CHECK(cudaEventSynchronize(io_ctx[next_buf].h2d_done[0]));
                }
                auto [nd, ib] = build_descs(pipe+1, next_buf);
                bam_bulk_read_async(io_ctx[next_buf], nd, 0, stream_io[next_buf]);
                CUDA_CHECK(cudaEventRecord(ev_io[pipe+1], stream_io[next_buf]));
                total_io_count += nd; total_io_bytes += ib;
            }
            CUDA_CHECK(cudaStreamWaitEvent(stream_comp, ev_io[pipe]));
            run_decomp_scan(pipe, buf);
            CUDA_CHECK(cudaEventRecord(ev_comp[pipe], stream_comp));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_comp));
    }

    auto total_end = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();
    double elapsed = elapsed_ns / 1e9;
    std::string comp_str;
    { std::set<std::string> methods;
      for (size_t fi = 0; fi < NUM_FIELDS; fi++)
          methods.insert(compression_method_name(field_comp[fi]));
      for (const auto &m : methods) { if (!comp_str.empty()) comp_str += "+"; comp_str += m; }
    }

    return BenchmarkResult{
        .nios = total_io_count, .read_bytes = total_io_bytes,
        .elapsed_nanoseconds = elapsed_ns, .compression = comp_str,
        .gpu_mem_bytes = sess.gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = sess.gpu_ctrl_bytes, .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NUM_FIELDS,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// GPU dim builders for Q2x — BaM read → GPU decomp → GPU kernel
// All use flat row indexing via prefix_sum. No D→H copies.
// ============================================================
// Per-dict raw download buffer (Rule 4: pre-alloc outside timing,
// cudaMemcpy inside timing, string construction outside timing)
struct DimDictRaw {
    char     *h_strs;
    uint16_t *h_lens;
    uint32_t *h_ids;
    uint32_t n;
    const char *fallback;    // set when n==0 (need_dict=false case)
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
    free(r.h_strs); free(r.h_lens); free(r.h_ids);
    r.h_strs = nullptr; r.h_lens = nullptr; r.h_ids = nullptr;
}

// Phase 1: cudaMemcpy only — safe inside timing (no heap allocation)
static void dim_download_dict_raw(const DimGpuDict &gd, DimDictRaw &out) {
    cudaMemcpy(&out.n, gd.d_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (out.n == 0) return;
    cudaMemcpy(out.h_strs, gd.d_strs, DIM_DICT_CAP * DIM_DICT_MAX_STRLEN, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.h_lens, gd.d_lens, DIM_DICT_CAP * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out.h_ids,  gd.d_type_ids, DIM_DICT_CAP * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

// Phase 2: construct strings — call OUTSIDE timing
static std::vector<std::string> dim_build_dict_strings(const DimDictRaw &r) {
    if (r.n == 0) {
        if (r.fallback) return {r.fallback};
        return {"_"};
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
    cudaStream_t stream, size_t &kl)
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
    kl++;
}

// ============================================================
// Optimized dim builders: pre-cached metadata + batched I/O +
// overlapping BaM reads with GPU decomp via 2-pipe architecture.
//
// Timeline for Q2x:
//   pipe0: [DATE 2pg read+decomp] [DATE kernel]
//   pipe0: [SUPP 96pg batched read+decomp] [SUPP kernel]
//          nvcomp_decompctx_run returns immediately after kernel launch
//   pipe1:  [PART P_BRAND1 96pg read]  ← overlaps with SUPP decomp on pipe0
//   pipe0: sync SUPP → SUPP kernel
//   pipe1: [P_BRAND1 decomp] [filter+dict kernel]
//          nvcomp_decompctx_run returns immediately
//   pipe0:  [PART P_PARTKEY 48pg read]  ← overlaps with P_BRAND1 decomp on pipe1
//   pipe1: sync → done
//   pipe0: [P_PARTKEY decomp] [flatten] [HT build]
// ============================================================

static void bam_build_date_ht_gpu(GpuHT &ht, const BamSessionCtx &s, DimGpuBufs &db, size_t &kl, uint64_t &io_bytes) {
    auto &m = s.metadata();
    using F = SSB::common::DateField;
    s.read_dim(SSB::common::Table::DDATE, F::D_DATEKEY, 0, db.d_buf_a, kl, io_bytes);
    s.read_dim(SSB::common::Table::DDATE, F::D_YEAR, 0, db.d_buf_b, kl, io_bytes);
    uint32_t np = m.table_date_npages[F::D_DATEKEY];
    uint32_t cap = (s.page_size - 12) / sizeof(int32_t);
    CUDA_CHECK(cudaMemsetAsync(ht.d_keys, 0xFF, (ht.mask + 1) * sizeof(int32_t), s.dim.stream));
    dim_build_date_ht_kernel<<<np, 256, 0, s.dim.stream>>>(
        db.d_buf_a, db.d_buf_b, np, s.page_size, cap,
        SSB_YEAR_MIN, ht.d_keys, ht.d_values, ht.mask);
    kl++;
    CUDA_CHECK(cudaStreamSynchronize(s.dim.stream));
}

static void bam_build_dim_q2x_gpu(
    GpuHT &supp_ht, GpuHT &part_ht,
    const BamSessionCtx &s, const char *region,
    SSB::Query query, DimDictRaw &brand_dict_raw,
    DimGpuBufs &db, size_t &kl, uint64_t &io_bytes)
{
    auto &m = s.metadata();
    const size_t ps = s.page_size;
    using SF = SSB::common::SupplierField;
    using PF = SSB::common::PartField;
    cudaStream_t s0 = s.dim.stream, s1 = s.dim.stream2;

    // ── Step 1: SUPPLIER read+decomp (pipe 0) ──
    s.read_dim(SSB::common::Table::SUPPLIER, SF::S_SUPPKEY, 0, db.d_buf_a, kl, io_bytes);
    s.read_dim(SSB::common::Table::SUPPLIER, SF::S_REGION, 0, db.d_buf_b, kl, io_bytes);
    // Build SUPPLIER HT: 2-step (filter on S_REGION pages, HT build on S_SUPPKEY pages)
    // to handle different per-field page counts at small page sizes (e.g. 64K).
    {
        const char *region_pred = region;
        uint32_t region_np = m.table_supplier_npages[SF::S_REGION];
        dim_run_char_filter(
            db.d_buf_b, region_np, ps, SSB::common::S_REGION_SIZE,
            nullptr, DIM_FILT_PREFIX, &region_pred, 1,
            false, nullptr,
            nullptr, db.d_filter, nullptr, s0, kl);

        uint32_t key_np = m.table_supplier_npages[SF::S_SUPPKEY];
        uint32_t key_cap = (ps - 12) / sizeof(int32_t);
        CUDA_CHECK(cudaMemsetAsync(supp_ht.d_keys, 0xFF,
            (supp_ht.mask + 1) * sizeof(int32_t), s0));
        dim_build_ht_paged_kernel<<<key_np, 256, 0, s0>>>(
            db.d_buf_a, key_np, ps, key_cap,
            db.d_filter, nullptr,
            supp_ht.d_keys, supp_ht.d_values, supp_ht.mask);
        kl++;
        CUDA_CHECK(cudaStreamSynchronize(s0));
    }

    // ── Step 2: PART P_BRAND1 read (pipe 0, BaM blocks host) ──
    // After read, nvCOMP decomp kernel is launched on s0 and returns immediately.
    s.read_dim(SSB::common::Table::PART, PF::P_BRAND1, 0, db.d_buf_a, kl, io_bytes);
    // Decomp kernel running on s0 (GPU), host is free now.

    // ── Step 3: PART P_PARTKEY read (pipe 1) — overlaps with P_BRAND1 decomp on s0 ──
    s.read_dim(SSB::common::Table::PART, PF::P_PARTKEY, 1, db.d_buf_b, kl, io_bytes);
    // P_PARTKEY decomp kernel launched on s1. Both decomps may still be running.

    // ── Step 4: Wait for P_BRAND1 (pipe 0), run filter+dict ──
    CUDA_CHECK(cudaStreamSynchronize(s0));
    uint32_t pb_np = m.table_part_npages[PF::P_BRAND1];
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
        db.d_buf_a, pb_np, ps, SSB::common::P_BRAND1_SIZE,
        nullptr, filt_mode, preds, n_preds,
        need_dict, need_dict ? &db.dict : nullptr,
        nullptr, db.d_filter, need_dict ? db.d_values : nullptr, s0, kl);

    // ── Step 5: Wait for P_PARTKEY (pipe 1), paged HT build ──
    CUDA_CHECK(cudaStreamSynchronize(s1));
    uint32_t pk_np = m.table_part_npages[PF::P_PARTKEY];
    uint32_t key_cap = (ps - 12) / sizeof(int32_t);
    CUDA_CHECK(cudaMemsetAsync(part_ht.d_keys, 0xFF,
        (part_ht.mask + 1) * sizeof(int32_t), s0));
    dim_build_ht_paged_kernel<<<pk_np, 256, 0, s0>>>(
        db.d_buf_b, pk_np, ps, key_cap,
        db.d_filter, need_dict ? db.d_values : nullptr,
        part_ht.d_keys, part_ht.d_values, part_ht.mask);
    kl++;
    CUDA_CHECK(cudaStreamSynchronize(s0));

    // Raw dict download (cudaMemcpy only, no string alloc — Rule 4)
    if (need_dict) {
        dim_download_dict_raw(db.dict, brand_dict_raw);
    } else {
        brand_dict_raw.n = 0;
        brand_dict_raw.fallback = preds[0];
    }
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
// Generic GPU dim table builder: INT32 key + CHAR filter + optional CHAR group-by
// Handles 2-field (filter==group) and 3-field (filter!=group) patterns.
// Uses 2-pipe overlap for 3-field case.
// ============================================================
static void bam_build_table_ht_gpu(
    GpuHT &ht, const BamSessionCtx &s, DimGpuBufs &db,
    SSB::common::Table table,
    size_t key_field, size_t filter_field, uint32_t filter_field_size,
    int32_t filter_mode, const char *const *filter_preds, uint32_t n_filter_preds,
    bool has_separate_group, size_t group_field, uint32_t group_field_size,
    bool need_dict, uint64_t nrows,
    DimDictRaw &dict_raw, size_t &kl, uint64_t &io_bytes)
{
    auto &m = s.metadata();
    const size_t ps = s.page_size;
    cudaStream_t s0 = s.dim.stream, s1 = s.dim.stream2;

    // ── Read key (pipe 0) + filter field (pipe 1) via legacy path (LZ4/SNAPPY both) ──
    s.read_dim_fast(table, key_field, 0, db.d_buf_a, kl, io_bytes);
    s.read_dim_fast(table, filter_field, 1, db.d_buf_b, kl, io_bytes);
    CUDA_CHECK(cudaStreamSynchronize(s0));
    CUDA_CHECK(cudaStreamSynchronize(s1));

    // ── If 3-field: read group field (pipe 0, after parallel IO completes) ──
    if (has_separate_group) {
        s.read_dim_fast(table, group_field, 0, db.d_buf_c, kl, io_bytes);
        CUDA_CHECK(cudaStreamSynchronize(s0));
    }
    uint32_t filter_np = dim_npages(m, table, filter_field);

    if (!has_separate_group) {
        // 2-field: filter + dict in one pass on filter field
        db.dict.reset();
        dim_run_char_filter(
            db.d_buf_b, filter_np, ps, filter_field_size,
            nullptr, filter_mode, filter_preds, n_filter_preds,
            need_dict, need_dict ? &db.dict : nullptr,
            nullptr, db.d_filter, need_dict ? db.d_values : nullptr, s0, kl);
    } else {
        // 3-field: pass 1 — filter only (no dict)
        dim_run_char_filter(
            db.d_buf_b, filter_np, ps, filter_field_size,
            nullptr, filter_mode, filter_preds, n_filter_preds,
            false, nullptr,
            nullptr, db.d_filter, nullptr, s0, kl);

        // ── Process group field with prefilter (already read on pipe 0) ──
        uint32_t group_np = dim_npages(m, table, group_field);
        db.dict.reset();
        dim_run_char_filter(
            db.d_buf_c, group_np, ps, group_field_size,
            nullptr, DIM_FILT_NONE, nullptr, 0,
            need_dict, need_dict ? &db.dict : nullptr,
            db.d_filter, nullptr, need_dict ? db.d_values : nullptr, s0, kl);
    }

    // ── Paged HT build (no prefix_sum/flatten needed) ──
    uint32_t key_np = dim_npages(m, table, key_field);
    uint32_t key_cap = (ps - 12) / sizeof(int32_t);
    CUDA_CHECK(cudaMemsetAsync(ht.d_keys, 0xFF, (ht.mask + 1) * sizeof(int32_t), s0));
    dim_build_ht_paged_kernel<<<key_np, 256, 0, s0>>>(
        db.d_buf_a, key_np, ps, key_cap,
        db.d_filter, need_dict ? db.d_values : nullptr,
        ht.d_keys, ht.d_values, ht.mask);
    kl++;
    CUDA_CHECK(cudaStreamSynchronize(s0));

    // Raw dict download (cudaMemcpy only — Rule 4)
    if (need_dict) {
        dim_download_dict_raw(db.dict, dict_raw);
    } else {
        dict_raw.n = 0;
        dict_raw.fallback = nullptr;
    }
}

// ── Extended DATE HT builder (with optional year/ymn filter) ──
static void bam_build_date_ht_ext_gpu(
    GpuHT &ht, const BamSessionCtx &s, DimGpuBufs &db,
    int32_t filter_mode, int32_t filter_lo, int32_t filter_hi, size_t &kl, uint64_t &io_bytes)
{
    auto &m = s.metadata();
    using F = SSB::common::DateField;
    cudaStream_t s0 = s.dim.stream, s1 = s.dim.stream2;

    // Read D_DATEKEY (pipe 0) + D_YEAR (pipe 1) via legacy path for correctness
    s.read_dim_fast(SSB::common::Table::DDATE, F::D_DATEKEY, 0, db.d_buf_a, kl, io_bytes);
    s.read_dim_fast(SSB::common::Table::DDATE, F::D_YEAR, 1, db.d_buf_b, kl, io_bytes);
    CUDA_CHECK(cudaStreamSynchronize(s0));
    CUDA_CHECK(cudaStreamSynchronize(s1));

    // For mode 2/3, read aux field after parallel IO completes
    if (filter_mode == 2) {
        s.read_dim_fast(SSB::common::Table::DDATE, F::D_YEARMONTHNUM, 0, db.d_buf_c, kl, io_bytes);
        CUDA_CHECK(cudaStreamSynchronize(s0));
    } else if (filter_mode == 3) {
        s.read_dim_fast(SSB::common::Table::DDATE, F::D_WEEKNUMINYEAR, 0, db.d_buf_c, kl, io_bytes);
        CUDA_CHECK(cudaStreamSynchronize(s0));
    }

    uint32_t np = m.table_date_npages[F::D_DATEKEY];
    uint32_t cap = (s.page_size - 12) / sizeof(int32_t);
    CUDA_CHECK(cudaMemsetAsync(ht.d_keys, 0xFF, (ht.mask + 1) * sizeof(int32_t), s0));
    dim_build_date_ht_ext_kernel<<<np, 256, 0, s0>>>(
        db.d_buf_a, db.d_buf_b,
        (filter_mode >= 2) ? db.d_buf_c : nullptr,
        np, s.page_size, cap, SSB_YEAR_MIN,
        filter_mode, filter_lo, filter_hi,
        ht.d_keys, ht.d_values, ht.mask);
    kl++;
    CUDA_CHECK(cudaStreamSynchronize(s0));
}

// ============================================================
// Q1x DATE HT builder (membership-only: datekey → 0)
// filter_mode: 1=year eq, 2=yearmonthnum eq, 3=year+weeknum
// ============================================================
static void bam_build_date_ht_q1x_gpu(
    GpuHT &ht, const BamSessionCtx &s, DimGpuBufs &db,
    SSB::Query query, size_t &kl, uint64_t &io_bytes)
{
    switch (query) {
        case SSB::Query::Q11:
            bam_build_date_ht_ext_gpu(ht, s, db, 1, 1993, 1993, kl, io_bytes);
            break;
        case SSB::Query::Q12:
            bam_build_date_ht_ext_gpu(ht, s, db, 2, 199401, 0, kl, io_bytes);
            break;
        case SSB::Query::Q13:
            bam_build_date_ht_ext_gpu(ht, s, db, 3, 1994, 6, kl, io_bytes);
            break;
        default:
            bam_build_date_ht_ext_gpu(ht, s, db, 0, 0, 0, kl, io_bytes);
            break;
    }
}

// ============================================================
// Q3x combined dim builder (DATE + CUSTOMER + SUPPLIER)
// ============================================================
static void bam_build_dim_q3x_gpu(
    GpuHT &date_ht, GpuHT &cust_ht, GpuHT &supp_ht,
    const BamSessionCtx &s, SSB::Query query,
    DimDictRaw &cust_dict_raw, DimDictRaw &supp_dict_raw,
    DimGpuBufs &db, size_t &kl, uint64_t &io_bytes)
{
    using CF = SSB::common::CustomerField;
    using SF = SSB::common::SupplierField;
    auto &m = s.metadata();

    // ── DATE ──
    if (query == SSB::Query::Q34)
        bam_build_date_ht_ext_gpu(date_ht, s, db, 2, 199712, 0, kl, io_bytes);
    else
        bam_build_date_ht_ext_gpu(date_ht, s, db, 1, 1992, 1997, kl, io_bytes);

    // ── CUSTOMER ──
    if (query == SSB::Query::Q31) {
        const char *p_asia[] = {"ASIA"};
        bam_build_table_ht_gpu(cust_ht, s, db,
            SSB::common::Table::CUSTOMER,
            CF::C_CUSTKEY, CF::C_REGION, SSB::common::C_REGION_SIZE,
            DIM_FILT_EQ, p_asia, 1,
            true, CF::C_NATION, SSB::common::C_NATION_SIZE,
            true, m.table_customer_nrows, cust_dict_raw, kl, io_bytes);
    } else if (query == SSB::Query::Q32) {
        const char *p_us[] = {"UNITED STATES"};
        bam_build_table_ht_gpu(cust_ht, s, db,
            SSB::common::Table::CUSTOMER,
            CF::C_CUSTKEY, CF::C_NATION, SSB::common::C_NATION_SIZE,
            DIM_FILT_EQ, p_us, 1,
            true, CF::C_CITY, SSB::common::C_CITY_SIZE,
            true, m.table_customer_nrows, cust_dict_raw, kl, io_bytes);
    } else {
        const char *p_cities[] = {"UNITED KI1", "UNITED KI5"};
        bam_build_table_ht_gpu(cust_ht, s, db,
            SSB::common::Table::CUSTOMER,
            CF::C_CUSTKEY, CF::C_CITY, SSB::common::C_CITY_SIZE,
            DIM_FILT_IN, p_cities, 2,
            false, 0, 0,
            true, m.table_customer_nrows, cust_dict_raw, kl, io_bytes);
    }

    // ── SUPPLIER (same pattern as CUSTOMER) ──
    if (query == SSB::Query::Q31) {
        const char *p_asia[] = {"ASIA"};
        bam_build_table_ht_gpu(supp_ht, s, db,
            SSB::common::Table::SUPPLIER,
            SF::S_SUPPKEY, SF::S_REGION, SSB::common::S_REGION_SIZE,
            DIM_FILT_EQ, p_asia, 1,
            true, SF::S_NATION, SSB::common::S_NATION_SIZE,
            true, m.table_supplier_nrows, supp_dict_raw, kl, io_bytes);
    } else if (query == SSB::Query::Q32) {
        const char *p_us[] = {"UNITED STATES"};
        bam_build_table_ht_gpu(supp_ht, s, db,
            SSB::common::Table::SUPPLIER,
            SF::S_SUPPKEY, SF::S_NATION, SSB::common::S_NATION_SIZE,
            DIM_FILT_EQ, p_us, 1,
            true, SF::S_CITY, SSB::common::S_CITY_SIZE,
            true, m.table_supplier_nrows, supp_dict_raw, kl, io_bytes);
    } else {
        const char *p_cities[] = {"UNITED KI1", "UNITED KI5"};
        bam_build_table_ht_gpu(supp_ht, s, db,
            SSB::common::Table::SUPPLIER,
            SF::S_SUPPKEY, SF::S_CITY, SSB::common::S_CITY_SIZE,
            DIM_FILT_IN, p_cities, 2,
            false, 0, 0,
            true, m.table_supplier_nrows, supp_dict_raw, kl, io_bytes);
    }
}

// ============================================================
// Q4x combined dim builder (DATE + CUSTOMER + SUPPLIER + PART)
// ============================================================
static void bam_build_dim_q4x_gpu(
    GpuHT &date_ht, GpuHT &cust_ht, GpuHT &supp_ht, GpuHT &part_ht,
    const BamSessionCtx &s, SSB::Query query,
    DimDictRaw &cust_dict_raw, DimDictRaw &supp_dict_raw,
    DimDictRaw &part_dict_raw,
    DimGpuBufs &db, size_t &kl, uint64_t &io_bytes)
{
    using CF = SSB::common::CustomerField;
    using SF = SSB::common::SupplierField;
    using PF = SSB::common::PartField;
    auto &m = s.metadata();

    auto t0 = std::chrono::high_resolution_clock::now();
    // ── DATE ──
    if (query == SSB::Query::Q41)
        bam_build_date_ht_ext_gpu(date_ht, s, db, 0, 0, 0, kl, io_bytes);
    else
        bam_build_date_ht_ext_gpu(date_ht, s, db, 1, 1997, 1998, kl, io_bytes);
    auto t1 = std::chrono::high_resolution_clock::now();

    // ── CUSTOMER: filter=C_REGION EQ "AMERICA" ──
    {
        const char *p_america[] = {"AMERICA"};
        if (query == SSB::Query::Q41) {
            bam_build_table_ht_gpu(cust_ht, s, db,
                SSB::common::Table::CUSTOMER,
                CF::C_CUSTKEY, CF::C_REGION, SSB::common::C_REGION_SIZE,
                DIM_FILT_EQ, p_america, 1,
                true, CF::C_NATION, SSB::common::C_NATION_SIZE,
                true, m.table_customer_nrows, cust_dict_raw, kl, io_bytes);
        } else {
            bam_build_table_ht_gpu(cust_ht, s, db,
                SSB::common::Table::CUSTOMER,
                CF::C_CUSTKEY, CF::C_REGION, SSB::common::C_REGION_SIZE,
                DIM_FILT_EQ, p_america, 1,
                false, 0, 0,
                false, m.table_customer_nrows, cust_dict_raw, kl, io_bytes);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    // ── SUPPLIER ──
    if (query == SSB::Query::Q41) {
        const char *p_america[] = {"AMERICA"};
        bam_build_table_ht_gpu(supp_ht, s, db,
            SSB::common::Table::SUPPLIER,
            SF::S_SUPPKEY, SF::S_REGION, SSB::common::S_REGION_SIZE,
            DIM_FILT_EQ, p_america, 1,
            false, 0, 0,
            false, m.table_supplier_nrows, supp_dict_raw, kl, io_bytes);
    } else if (query == SSB::Query::Q42) {
        const char *p_america[] = {"AMERICA"};
        bam_build_table_ht_gpu(supp_ht, s, db,
            SSB::common::Table::SUPPLIER,
            SF::S_SUPPKEY, SF::S_REGION, SSB::common::S_REGION_SIZE,
            DIM_FILT_EQ, p_america, 1,
            true, SF::S_NATION, SSB::common::S_NATION_SIZE,
            true, m.table_supplier_nrows, supp_dict_raw, kl, io_bytes);
    } else {
        const char *p_us[] = {"UNITED STATES"};
        bam_build_table_ht_gpu(supp_ht, s, db,
            SSB::common::Table::SUPPLIER,
            SF::S_SUPPKEY, SF::S_NATION, SSB::common::S_NATION_SIZE,
            DIM_FILT_EQ, p_us, 1,
            true, SF::S_CITY, SSB::common::S_CITY_SIZE,
            true, m.table_supplier_nrows, supp_dict_raw, kl, io_bytes);
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    // ── PART ──
    if (query == SSB::Query::Q41) {
        const char *p_mfgr12[] = {"MFGR#1", "MFGR#2"};
        bam_build_table_ht_gpu(part_ht, s, db,
            SSB::common::Table::PART,
            PF::P_PARTKEY, PF::P_MFGR, SSB::common::P_MFGR_SIZE,
            DIM_FILT_IN, p_mfgr12, 2,
            false, 0, 0,
            false, m.table_part_nrows, part_dict_raw, kl, io_bytes);
    } else if (query == SSB::Query::Q42) {
        const char *p_mfgr12[] = {"MFGR#1", "MFGR#2"};
        bam_build_table_ht_gpu(part_ht, s, db,
            SSB::common::Table::PART,
            PF::P_PARTKEY, PF::P_MFGR, SSB::common::P_MFGR_SIZE,
            DIM_FILT_IN, p_mfgr12, 2,
            true, PF::P_CATEGORY, SSB::common::P_CATEGORY_SIZE,
            true, m.table_part_nrows, part_dict_raw, kl, io_bytes);
    } else {
        const char *p_mfgr14[] = {"MFGR#14"};
        bam_build_table_ht_gpu(part_ht, s, db,
            SSB::common::Table::PART,
            PF::P_PARTKEY, PF::P_CATEGORY, SSB::common::P_CATEGORY_SIZE,
            DIM_FILT_EQ, p_mfgr14, 1,
            true, PF::P_BRAND1, SSB::common::P_BRAND1_SIZE,
            true, m.table_part_nrows, part_dict_raw, kl, io_bytes);
    }
}

// ============================================================
// SSB Q2.x GIDP+BAM
// ============================================================
static BenchmarkResult ssb_q2x_gidp_bam(BenchmarkOptions &options, SSB::Query query) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    const size_t ps = sess.page_size;

    const char *supp_region;
    switch (query) {
        case SSB::Query::Q21: supp_region = "AMERICA"; break;
        case SSB::Query::Q22: supp_region = "ASIA"; break;
        case SSB::Query::Q23: supp_region = "EUROPE"; break;
        default: exit(EXIT_FAILURE);
    }
    // Pre-allocate GPU HTs (Rule 3: alloc outside timing)
    auto &m_pre = sess.metadata();
    GpuHT date = alloc_gpu_ht(m_pre.table_date_nrows);
    GpuHT supp = alloc_gpu_ht(m_pre.table_supplier_nrows);
    GpuHT part = alloc_gpu_ht(m_pre.table_part_nrows);

    // Pre-allocate GPU dim build buffers (Rule 3)
    size_t max_nrows = m_pre.table_part_nrows;  // PART is largest dim table
    DimGpuBufs dim_bufs = dim_gpu_bufs_alloc(ps, max_nrows, sess.dim.max_pages);

    size_t gpu_pre_dim = 0, gd = 0; cudaMemGetInfo(&gpu_pre_dim, &gd);
    int64_t *d_rev; CUDA_CHECK(cudaMalloc(&d_rev, SSB_Q2X_GROUPS*sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_rev, 0, SSB_Q2X_GROUPS*sizeof(int64_t)));
    size_t gpu_post_dim = 0; cudaMemGetInfo(&gpu_post_dim, &gd);

    // Sideways pruning — metadata + dict IDs (Rule 3: outside timing)
    auto &m = sess.metadata();
    uint64_t npages = m.table_lineorder_npages[SSB::common::LO_ORDERDATE];
    std::vector<bool> page_active(npages, true);
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    int32_t zm_sr_dict = -1, zm_p_lo = -1, zm_p_hi = -1;
    uint64_t zm_sr_nstats = 0, zm_sr_start = 0, zm_sr_npg = 0;
    uint64_t zm_part_nstats = 0, zm_part_start = 0, zm_part_npg = 0;
    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);
        const size_t ref_field = SSB::common::LO_ORDERDATE;

        auto it_sr = dict_maps[SSB::common::LSS_S_REGION].find(std::string(supp_region));
        if (it_sr != dict_maps[SSB::common::LSS_S_REGION].end()) zm_sr_dict = it_sr->second;
        zm_sr_nstats = m.table_lineorder_sideways_nstats[ref_field][SSB::common::LSS_S_REGION];
        zm_sr_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][SSB::common::LSS_S_REGION];
        zm_sr_npg = m.table_lineorder_sideways_stats_npages[ref_field][SSB::common::LSS_S_REGION];

        size_t part_sw_idx = 0;
        if (query == SSB::Query::Q21) {
            part_sw_idx = SSB::common::LSS_P_CATEGORY;
            auto it = dict_maps[part_sw_idx].find("MFGR#12");
            if (it != dict_maps[part_sw_idx].end()) zm_p_lo = zm_p_hi = it->second;
        } else {
            part_sw_idx = SSB::common::LSS_P_BRAND1;
            auto it_lo_b = dict_maps[part_sw_idx].find("MFGR#2221");
            auto it_hi_b = dict_maps[part_sw_idx].find(query == SSB::Query::Q22 ? "MFGR#2228" : "MFGR#2221");
            if (it_lo_b != dict_maps[part_sw_idx].end()) zm_p_lo = it_lo_b->second;
            if (it_hi_b != dict_maps[part_sw_idx].end()) zm_p_hi = it_hi_b->second;
        }
        zm_part_nstats = m.table_lineorder_sideways_nstats[ref_field][part_sw_idx];
        zm_part_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][part_sw_idx];
        zm_part_npg = m.table_lineorder_sideways_stats_npages[ref_field][part_sw_idx];
    }

    // Pre-allocate LO pipeline context (Rule 4: outside timing)
    BamLoPipeCtx<SSB::query::q2x::NUM_LO_ACTIVE_FIELDS> lo_ctx{};
    bam_lo_pipe_ctx_alloc(lo_ctx, sess, SSB::query::q2x::LO_FIELDS,
        npages, gpu_pre_dim - gpu_post_dim);
    bam_lo_pipe_ctx_read_comp_meta(lo_ctx, sess, SSB::query::q2x::LO_FIELDS);

    // BaM zonemap ctx: borrows io_ctx[0]'s page_cache (Rule 4: alloc outside timing)
    if (options.enable_zonemap) {
        bool has_sr = (zm_sr_dict >= 0 && zm_sr_nstats > 0 && zm_sr_start > 0 && zm_sr_npg > 0);
        bool has_part = (zm_p_lo >= 0 && zm_part_nstats > 0 && zm_part_start > 0 && zm_part_npg > 0);
        if (has_sr || has_part) {
            zm_ctx = bam_zonemap_ctx_create(
                lo_ctx.io_ctx[0].d_ctrls, lo_ctx.io_ctx[0].d_pc, lo_ctx.io_ctx[0].pc_base,
                static_cast<uint32_t>(ps), npages);
            uint32_t ri = 0;
            if (has_sr) {
                uint32_t off = ri;
                for (uint64_t j = 0; j < zm_sr_npg; j++) {
                    uint64_t pg_id = zm_sr_start + j;
                    uint32_t dev = pg_id % sess.n_devices;
                    uint64_t local = pg_id / sess.n_devices;
                    zm_ctx.h_reads[ri++] = {
                        sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                        static_cast<uint32_t>(ps / 512), dev};
                }
                zm_ctx.h_preds[zm_npreds++] = {off, zm_sr_nstats, zm_sr_dict, zm_sr_dict};
            }
            if (has_part) {
                uint32_t off = ri;
                for (uint64_t j = 0; j < zm_part_npg; j++) {
                    uint64_t pg_id = zm_part_start + j;
                    uint32_t dev = pg_id % sess.n_devices;
                    uint64_t local = pg_id / sess.n_devices;
                    zm_ctx.h_reads[ri++] = {
                        sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                        static_cast<uint32_t>(ps / 512), dev};
                }
                zm_ctx.h_preds[zm_npreds++] = {off, zm_part_nstats, zm_p_lo, zm_p_hi};
            }
            zm_nreads = ri;
            zm_valid = true;
        }
    }

    // Pre-allocate dict download buffer (Rule 4: outside timing)
    DimDictRaw brand_dict_raw = dim_dict_raw_alloc();

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, lo_ctx.stream_io[0]);
    }

    // Timing starts here to include dim I/O (Rule 1)
    s_kernel_launches = 0;
    uint64_t dim_io_bytes = 0;
    auto outer_start = std::chrono::steady_clock::now();
    bam_build_date_ht_gpu(date, sess, dim_bufs, s_kernel_launches, dim_io_bytes);
    bam_build_dim_q2x_gpu(supp, part, sess, supp_region, query, brand_dict_raw, dim_bufs, s_kernel_launches, dim_io_bytes);
    auto dim_end = std::chrono::steady_clock::now();

    // Fused BaM zonemap eval (Rule 6: IO + eval inside timing)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, lo_ctx.stream_io[0]);
        CUDA_CHECK(cudaStreamSynchronize(lo_ctx.stream_io[0]));
        s_kernel_launches++;
        uint32_t active_count = 0;
        for (size_t pg = 0; pg < npages; pg++) {
            page_active[pg] = zm_ctx.h_mask[pg];
            if (zm_ctx.h_mask[pg]) active_count++;
        }
        std::cout << "[ZONEMAP] Q2x pruning: active=" << active_count << "/" << npages << std::endl;
    }

    auto result = bam_lo_pipeline<SSB::query::q2x::NUM_LO_ACTIVE_FIELDS>(
        options, sess, SSB::query::q2x::LO_FIELDS, page_active,
        lo_ctx,
        [&](void **db, size_t np, uint64_t, cudaStream_t s) {
            ssb_q2x_paged(db[0], db[1], db[2], db[3], np, ps,
                date.d_keys, date.d_values, date.mask,
                supp.d_keys, supp.d_values, supp.mask,
                part.d_keys, part.d_values, part.mask, d_rev, s);
        });
    int64_t h_rev[SSB_Q2X_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_rev, d_rev, sizeof(h_rev), cudaMemcpyDeviceToHost));
    auto outer_end = std::chrono::steady_clock::now();

    // Construct dict strings outside timing (Rule 4)
    std::vector<std::string> brand1_dict = dim_build_dict_strings(brand_dict_raw);
    {
        auto dim_ms = std::chrono::duration<double, std::milli>(dim_end - outer_start).count();
        auto lo_ms  = std::chrono::duration<double, std::milli>(outer_end - dim_end).count();
        auto tot_ms = std::chrono::duration<double, std::milli>(outer_end - outer_start).count();
        std::cout << "[Q2x-TIMING] dim=" << dim_ms << "ms  lo_pipeline=" << lo_ms
                  << "ms  total=" << tot_ms << "ms" << std::endl;
    }
    result.elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(outer_end - outer_start).count();
    result.read_bytes += dim_io_bytes;
    {
        double elapsed = result.elapsed_nanoseconds / 1e9;
        std::cout << "\n========================================\n"
                  << "Total elapsed: " << elapsed << " seconds\n"
                  << "Total I/Os: " << result.nios << "\n"
                  << "Total bytes read: " << result.read_bytes
                  << "\n========================================" << std::endl;
    }
    std::cout << "\nSSB Q2.x results:" << std::endl;
    for (int32_t y = 0; y < SSB_NUM_YEARS; y++)
        for (size_t b = 0; b < brand1_dict.size(); b++) {
            int64_t v = h_rev[y * SSB_MAX_BRANDS + b];
            if (v != 0) std::cout << "  " << v << " | " << (SSB_YEAR_MIN+y) << " | " << brand1_dict[b] << std::endl;
        }

    date.free_all(); supp.free_all(); part.free_all();
    dim_gpu_bufs_free(dim_bufs);
    dim_dict_raw_free(brand_dict_raw);
    bam_lo_pipe_ctx_free(lo_ctx);
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    CUDA_CHECK(cudaFree(d_rev));
    bam_session_close(sess);
    return result;
}

// ============================================================
// SSB Q3.x GIDP+BAM
// ============================================================
static BenchmarkResult ssb_q3x_gidp_bam(BenchmarkOptions &options, SSB::Query query) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    const size_t ps = sess.page_size;

    // Pre-allocate GPU HTs (Rule 2: alloc outside timing)
    auto &m_pre3 = sess.metadata();
    GpuHT date = alloc_gpu_ht(m_pre3.table_date_nrows);
    GpuHT cust = alloc_gpu_ht(m_pre3.table_customer_nrows);
    GpuHT supp = alloc_gpu_ht(m_pre3.table_supplier_nrows);

    size_t max_nrows3 = std::max(m_pre3.table_customer_nrows, m_pre3.table_supplier_nrows);
    DimGpuBufs dim_bufs3 = dim_gpu_bufs_alloc(ps, max_nrows3, sess.dim.max_pages);

    size_t gpu_pre = 0, gd = 0; cudaMemGetInfo(&gpu_pre, &gd);
    size_t grp_sz = SSB_Q3X_MAX_GROUPS * sizeof(int64_t);
    int64_t *d_rev; CUDA_CHECK(cudaMalloc(&d_rev, grp_sz)); CUDA_CHECK(cudaMemset(d_rev, 0, grp_sz));
    size_t gpu_post = 0; cudaMemGetInfo(&gpu_post, &gd);

    // Sideways pruning — metadata + dict IDs (Rule 3: outside timing)
    auto &m = sess.metadata();
    uint64_t npages = m.table_lineorder_npages[SSB::common::LO_ORDERDATE];
    std::vector<bool> page_active(npages, true);
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    int32_t zm_cust_lo = -1, zm_cust_hi = -1;
    int32_t zm_supp_lo = -1, zm_supp_hi = -1;
    uint64_t zm_dk_nstats = 0, zm_dk_start = 0, zm_dk_npg = 0;
    uint64_t zm_cust_nstats = 0, zm_cust_start = 0, zm_cust_npg = 0;
    uint64_t zm_supp_nstats = 0, zm_supp_start = 0, zm_supp_npg = 0;
    bool zm_has_dk = false;
    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);
        const size_t ref_field = SSB::common::LO_ORDERDATE;

        // Date range stats (Q3.4: Dec 1997 only)
        if (query == SSB::Query::Q34) {
            zm_dk_nstats = m.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
            zm_dk_start = m.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
            zm_dk_npg = m.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];
            zm_has_dk = (zm_dk_nstats > 0 && zm_dk_start > 0 && zm_dk_npg > 0);
        }

        // Customer sideways pruning
        size_t cust_sw_idx = 0;
        if (query == SSB::Query::Q31) {
            cust_sw_idx = SSB::common::LSS_C_REGION;
            auto it = dict_maps[cust_sw_idx].find("ASIA");
            if (it != dict_maps[cust_sw_idx].end()) zm_cust_lo = zm_cust_hi = it->second;
        } else if (query == SSB::Query::Q32) {
            cust_sw_idx = SSB::common::LSS_C_NATION;
            auto it = dict_maps[cust_sw_idx].find("UNITED STATES");
            if (it != dict_maps[cust_sw_idx].end()) zm_cust_lo = zm_cust_hi = it->second;
        } else {
            cust_sw_idx = SSB::common::LSS_C_CITY;
            auto it1 = dict_maps[cust_sw_idx].find("UNITED KI1");
            auto it2 = dict_maps[cust_sw_idx].find("UNITED KI5");
            if (it1 != dict_maps[cust_sw_idx].end() && it2 != dict_maps[cust_sw_idx].end()) {
                zm_cust_lo = std::min(it1->second, it2->second);
                zm_cust_hi = std::max(it1->second, it2->second);
            }
        }
        zm_cust_nstats = m.table_lineorder_sideways_nstats[ref_field][cust_sw_idx];
        zm_cust_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][cust_sw_idx];
        zm_cust_npg = m.table_lineorder_sideways_stats_npages[ref_field][cust_sw_idx];

        // Supplier sideways pruning
        size_t supp_sw_idx = 0;
        if (query == SSB::Query::Q31) {
            supp_sw_idx = SSB::common::LSS_S_REGION;
            auto it = dict_maps[supp_sw_idx].find("ASIA");
            if (it != dict_maps[supp_sw_idx].end()) zm_supp_lo = zm_supp_hi = it->second;
        } else if (query == SSB::Query::Q32) {
            supp_sw_idx = SSB::common::LSS_S_NATION;
            auto it = dict_maps[supp_sw_idx].find("UNITED STATES");
            if (it != dict_maps[supp_sw_idx].end()) zm_supp_lo = zm_supp_hi = it->second;
        } else {
            supp_sw_idx = SSB::common::LSS_S_CITY;
            auto it1 = dict_maps[supp_sw_idx].find("UNITED KI1");
            auto it2 = dict_maps[supp_sw_idx].find("UNITED KI5");
            if (it1 != dict_maps[supp_sw_idx].end() && it2 != dict_maps[supp_sw_idx].end()) {
                zm_supp_lo = std::min(it1->second, it2->second);
                zm_supp_hi = std::max(it1->second, it2->second);
            }
        }
        zm_supp_nstats = m.table_lineorder_sideways_nstats[ref_field][supp_sw_idx];
        zm_supp_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][supp_sw_idx];
        zm_supp_npg = m.table_lineorder_sideways_stats_npages[ref_field][supp_sw_idx];
    }

    // Pre-allocate LO pipeline context (Rule 4: outside timing)
    BamLoPipeCtx<SSB::query::q3x::NUM_LO_ACTIVE_FIELDS> lo_ctx{};
    bam_lo_pipe_ctx_alloc(lo_ctx, sess, SSB::query::q3x::LO_FIELDS,
        npages, gpu_pre - gpu_post);
    bam_lo_pipe_ctx_read_comp_meta(lo_ctx, sess, SSB::query::q3x::LO_FIELDS);

    // BaM zonemap ctx: borrows io_ctx[0]'s page_cache (Rule 4: alloc outside timing)
    if (options.enable_zonemap) {
        bool has_cust = (zm_cust_lo >= 0 && zm_cust_nstats > 0 && zm_cust_start > 0 && zm_cust_npg > 0);
        bool has_supp = (zm_supp_lo >= 0 && zm_supp_nstats > 0 && zm_supp_start > 0 && zm_supp_npg > 0);
        if (zm_has_dk || has_cust || has_supp) {
            zm_ctx = bam_zonemap_ctx_create(
                lo_ctx.io_ctx[0].d_ctrls, lo_ctx.io_ctx[0].d_pc, lo_ctx.io_ctx[0].pc_base,
                static_cast<uint32_t>(ps), npages);
            uint32_t ri = 0;
            if (zm_has_dk) {
                uint32_t off = ri;
                for (uint64_t j = 0; j < zm_dk_npg; j++) {
                    uint64_t pg_id = zm_dk_start + j;
                    uint32_t dev = pg_id % sess.n_devices;
                    uint64_t local = pg_id / sess.n_devices;
                    zm_ctx.h_reads[ri++] = {
                        sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                        static_cast<uint32_t>(ps / 512), dev};
                }
                zm_ctx.h_preds[zm_npreds++] = {off, zm_dk_nstats, 19971201, 19971231};
            }
            if (has_cust) {
                uint32_t off = ri;
                for (uint64_t j = 0; j < zm_cust_npg; j++) {
                    uint64_t pg_id = zm_cust_start + j;
                    uint32_t dev = pg_id % sess.n_devices;
                    uint64_t local = pg_id / sess.n_devices;
                    zm_ctx.h_reads[ri++] = {
                        sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                        static_cast<uint32_t>(ps / 512), dev};
                }
                zm_ctx.h_preds[zm_npreds++] = {off, zm_cust_nstats, zm_cust_lo, zm_cust_hi};
            }
            if (has_supp) {
                uint32_t off = ri;
                for (uint64_t j = 0; j < zm_supp_npg; j++) {
                    uint64_t pg_id = zm_supp_start + j;
                    uint32_t dev = pg_id % sess.n_devices;
                    uint64_t local = pg_id / sess.n_devices;
                    zm_ctx.h_reads[ri++] = {
                        sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                        static_cast<uint32_t>(ps / 512), dev};
                }
                zm_ctx.h_preds[zm_npreds++] = {off, zm_supp_nstats, zm_supp_lo, zm_supp_hi};
            }
            zm_nreads = ri;
            zm_valid = true;
        }
    }

    // Pre-allocate dict download buffers (Rule 4: outside timing)
    DimDictRaw cust_dict_raw = dim_dict_raw_alloc();
    DimDictRaw supp_dict_raw = dim_dict_raw_alloc();

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, lo_ctx.stream_io[0]);
    }

    // Timing starts here to include dim I/O (Rule 1)
    s_kernel_launches = 0;
    uint64_t dim_io_bytes = 0;
    auto outer_start = std::chrono::steady_clock::now();
    bam_build_dim_q3x_gpu(date, cust, supp, sess, query, cust_dict_raw, supp_dict_raw, dim_bufs3, s_kernel_launches, dim_io_bytes);
    int32_t ncd = (int32_t)cust_dict_raw.n;
    int32_t nsd = (int32_t)supp_dict_raw.n;
    if (ncd == 0) ncd = 1;  // fallback "_"
    if (nsd == 0) nsd = 1;
    auto dim_end = std::chrono::steady_clock::now();

    // Fused BaM zonemap eval (Rule 6: IO + eval inside timing)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, lo_ctx.stream_io[0]);
        CUDA_CHECK(cudaStreamSynchronize(lo_ctx.stream_io[0]));
        s_kernel_launches++;
        uint32_t active_count = 0;
        for (size_t pg = 0; pg < npages; pg++) {
            page_active[pg] = zm_ctx.h_mask[pg];
            if (zm_ctx.h_mask[pg]) active_count++;
        }
        std::cout << "[ZONEMAP] Q3x pruning: active=" << active_count << "/" << npages << std::endl;
    }

    auto result = bam_lo_pipeline<SSB::query::q3x::NUM_LO_ACTIVE_FIELDS>(
        options, sess, SSB::query::q3x::LO_FIELDS, page_active,
        lo_ctx,
        [&](void **db, size_t np, uint64_t, cudaStream_t s) {
            ssb_q3x_paged(db[0], db[1], db[2], db[3], np, ps,
                date.d_keys, date.d_values, date.mask,
                cust.d_keys, cust.d_values, cust.mask,
                supp.d_keys, supp.d_values, supp.mask, nsd, ncd, d_rev, s);
        });
    int64_t h_rev[SSB_Q3X_MAX_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_rev, d_rev, sizeof(h_rev), cudaMemcpyDeviceToHost));
    auto outer_end = std::chrono::steady_clock::now();

    // Construct dict strings outside timing (Rule 4)
    std::vector<std::string> cust_dict = dim_build_dict_strings(cust_dict_raw);
    std::vector<std::string> supp_dict = dim_build_dict_strings(supp_dict_raw);
    {
        auto dim_ms = std::chrono::duration<double, std::milli>(dim_end - outer_start).count();
        auto lo_ms  = std::chrono::duration<double, std::milli>(outer_end - dim_end).count();
        auto tot_ms = std::chrono::duration<double, std::milli>(outer_end - outer_start).count();
        std::cout << "[Q3x-TIMING] dim=" << dim_ms << "ms  lo_pipeline=" << lo_ms
                  << "ms  total=" << tot_ms << "ms" << std::endl;
    }
    result.elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(outer_end - outer_start).count();
    result.read_bytes += dim_io_bytes;
    {
        double elapsed = result.elapsed_nanoseconds / 1e9;
        std::cout << "\n========================================\n"
                  << "Total elapsed: " << elapsed << " seconds\n"
                  << "Total I/Os: " << result.nios << "\n"
                  << "Total bytes read: " << result.read_bytes
                  << "\n========================================" << std::endl;
    }
    struct Q3xRow { std::string c, s; int32_t y; int64_t r; };
    std::vector<Q3xRow> rows;
    for (int32_t c = 0; c < ncd; c++)
        for (int32_t si = 0; si < nsd; si++)
            for (int32_t y = 0; y < SSB_Q3X_MAX_YEARS; y++) {
                int64_t v = h_rev[c * nsd * SSB_Q3X_MAX_YEARS + si * SSB_Q3X_MAX_YEARS + y];
                if (v != 0) rows.push_back({cust_dict[c], supp_dict[si], SSB_YEAR_MIN+y, v});
            }
    std::sort(rows.begin(), rows.end(), [](const Q3xRow &a, const Q3xRow &b) {
        return a.y != b.y ? a.y < b.y : a.r > b.r; });
    std::cout << "\nSSB Q3.x results:" << std::endl;
    for (auto &r : rows) std::cout << "  " << r.c << " | " << r.s << " | " << r.y << " | " << r.r << std::endl;

    date.free_all(); cust.free_all(); supp.free_all();
    dim_gpu_bufs_free(dim_bufs3);
    dim_dict_raw_free(cust_dict_raw); dim_dict_raw_free(supp_dict_raw);
    bam_lo_pipe_ctx_free(lo_ctx);
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    CUDA_CHECK(cudaFree(d_rev));
    bam_session_close(sess);
    return result;
}

// ============================================================
// SSB Q4.x GIDP+BAM
// ============================================================
static BenchmarkResult ssb_q4x_gidp_bam(BenchmarkOptions &options, SSB::Query query) {
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    const size_t ps = sess.page_size;

    // Pre-allocate GPU HTs (Rule 2: alloc outside timing)
    auto &m_pre4 = sess.metadata();
    GpuHT date = alloc_gpu_ht(m_pre4.table_date_nrows);
    GpuHT cust = alloc_gpu_ht(m_pre4.table_customer_nrows);
    GpuHT supp = alloc_gpu_ht(m_pre4.table_supplier_nrows);
    GpuHT part = alloc_gpu_ht(m_pre4.table_part_nrows);

    size_t max_nrows4 = std::max({(uint64_t)m_pre4.table_customer_nrows,
                                  (uint64_t)m_pre4.table_supplier_nrows,
                                  (uint64_t)m_pre4.table_part_nrows});
    DimGpuBufs dim_bufs4 = dim_gpu_bufs_alloc(ps, max_nrows4, sess.dim.max_pages);

    size_t gpu_pre = 0, gd = 0; cudaMemGetInfo(&gpu_pre, &gd);
    size_t grp_sz = SSB_Q4X_MAX_GROUPS * sizeof(int64_t);
    int64_t *d_prof; CUDA_CHECK(cudaMalloc(&d_prof, grp_sz)); CUDA_CHECK(cudaMemset(d_prof, 0, grp_sz));
    size_t gpu_post = 0; cudaMemGetInfo(&gpu_post, &gd);

    // Sideways pruning — metadata + dict IDs (Rule 3: outside timing)
    auto &m4 = sess.metadata();
    uint64_t npages4 = m4.table_lineorder_npages[SSB::common::LO_ORDERDATE];
    std::vector<bool> page_active(npages4, true);
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    int32_t zm_cr_dict = -1, zm_supp_lo = -1, zm_supp_hi = -1;
    int32_t zm_part_lo = -1, zm_part_hi = -1;
    uint64_t zm_dk_nstats = 0, zm_dk_start = 0, zm_dk_npg = 0;
    uint64_t zm_cr_nstats = 0, zm_cr_start = 0, zm_cr_npg = 0;
    uint64_t zm_supp_nstats = 0, zm_supp_start = 0, zm_supp_npg = 0;
    uint64_t zm_part_nstats = 0, zm_part_start = 0, zm_part_npg = 0;
    bool zm_has_dk = false;
    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);
        const size_t ref_field = SSB::common::LO_ORDERDATE;

        // Date range stats (Q4.2/Q4.3: 1997-1998 only)
        if (query != SSB::Query::Q41) {
            zm_dk_nstats = m4.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
            zm_dk_start = m4.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
            zm_dk_npg = m4.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];
            zm_has_dk = (zm_dk_nstats > 0 && zm_dk_start > 0 && zm_dk_npg > 0);
        }

        // C_REGION = 'AMERICA' for all Q4x
        {
            auto it = dict_maps[SSB::common::LSS_C_REGION].find("AMERICA");
            if (it != dict_maps[SSB::common::LSS_C_REGION].end()) zm_cr_dict = it->second;
        }
        zm_cr_nstats = m4.table_lineorder_sideways_nstats[ref_field][SSB::common::LSS_C_REGION];
        zm_cr_start = m4.table_lineorder_sideways_stats_start_page_ids[ref_field][SSB::common::LSS_C_REGION];
        zm_cr_npg = m4.table_lineorder_sideways_stats_npages[ref_field][SSB::common::LSS_C_REGION];

        // Supplier sideways pruning
        size_t supp_sw_idx = 0;
        if (query == SSB::Query::Q41 || query == SSB::Query::Q42) {
            supp_sw_idx = SSB::common::LSS_S_REGION;
            auto it = dict_maps[supp_sw_idx].find("AMERICA");
            if (it != dict_maps[supp_sw_idx].end()) zm_supp_lo = zm_supp_hi = it->second;
        } else {
            supp_sw_idx = SSB::common::LSS_S_NATION;
            auto it = dict_maps[supp_sw_idx].find("UNITED STATES");
            if (it != dict_maps[supp_sw_idx].end()) zm_supp_lo = zm_supp_hi = it->second;
        }
        zm_supp_nstats = m4.table_lineorder_sideways_nstats[ref_field][supp_sw_idx];
        zm_supp_start = m4.table_lineorder_sideways_stats_start_page_ids[ref_field][supp_sw_idx];
        zm_supp_npg = m4.table_lineorder_sideways_stats_npages[ref_field][supp_sw_idx];

        // Part sideways pruning
        size_t part_sw_idx = 0;
        if (query == SSB::Query::Q41 || query == SSB::Query::Q42) {
            part_sw_idx = SSB::common::LSS_P_MFGR;
            auto it1 = dict_maps[part_sw_idx].find("MFGR#1");
            auto it2 = dict_maps[part_sw_idx].find("MFGR#2");
            if (it1 != dict_maps[part_sw_idx].end() && it2 != dict_maps[part_sw_idx].end()) {
                zm_part_lo = std::min(it1->second, it2->second);
                zm_part_hi = std::max(it1->second, it2->second);
            }
        } else {
            part_sw_idx = SSB::common::LSS_P_CATEGORY;
            auto it = dict_maps[part_sw_idx].find("MFGR#14");
            if (it != dict_maps[part_sw_idx].end()) zm_part_lo = zm_part_hi = it->second;
        }
        zm_part_nstats = m4.table_lineorder_sideways_nstats[ref_field][part_sw_idx];
        zm_part_start = m4.table_lineorder_sideways_stats_start_page_ids[ref_field][part_sw_idx];
        zm_part_npg = m4.table_lineorder_sideways_stats_npages[ref_field][part_sw_idx];
    }

    // Pre-allocate LO pipeline context (Rule 4: outside timing)
    BamLoPipeCtx<SSB::query::q4x::NUM_LO_ACTIVE_FIELDS> lo_ctx{};
    bam_lo_pipe_ctx_alloc(lo_ctx, sess, SSB::query::q4x::LO_FIELDS,
        npages4, gpu_pre - gpu_post);
    bam_lo_pipe_ctx_read_comp_meta(lo_ctx, sess, SSB::query::q4x::LO_FIELDS);

    // BaM zonemap ctx: borrows io_ctx[0]'s page_cache (Rule 4: alloc outside timing)
    if (options.enable_zonemap) {
        bool has_cr = (zm_cr_dict >= 0 && zm_cr_nstats > 0 && zm_cr_start > 0 && zm_cr_npg > 0);
        bool has_supp = (zm_supp_lo >= 0 && zm_supp_nstats > 0 && zm_supp_start > 0 && zm_supp_npg > 0);
        bool has_part = (zm_part_lo >= 0 && zm_part_nstats > 0 && zm_part_start > 0 && zm_part_npg > 0);
        if (zm_has_dk || has_cr || has_supp || has_part) {
            zm_ctx = bam_zonemap_ctx_create(
                lo_ctx.io_ctx[0].d_ctrls, lo_ctx.io_ctx[0].d_pc, lo_ctx.io_ctx[0].pc_base,
                static_cast<uint32_t>(ps), npages4);
            uint32_t ri = 0;
            if (zm_has_dk) {
                uint32_t off = ri;
                for (uint64_t j = 0; j < zm_dk_npg; j++) {
                    uint64_t pg_id = zm_dk_start + j;
                    uint32_t dev = pg_id % sess.n_devices;
                    uint64_t local = pg_id / sess.n_devices;
                    zm_ctx.h_reads[ri++] = {
                        sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                        static_cast<uint32_t>(ps / 512), dev};
                }
                zm_ctx.h_preds[zm_npreds++] = {off, zm_dk_nstats, 19970101, 19981231};
            }
            if (has_cr) {
                uint32_t off = ri;
                for (uint64_t j = 0; j < zm_cr_npg; j++) {
                    uint64_t pg_id = zm_cr_start + j;
                    uint32_t dev = pg_id % sess.n_devices;
                    uint64_t local = pg_id / sess.n_devices;
                    zm_ctx.h_reads[ri++] = {
                        sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                        static_cast<uint32_t>(ps / 512), dev};
                }
                zm_ctx.h_preds[zm_npreds++] = {off, zm_cr_nstats, zm_cr_dict, zm_cr_dict};
            }
            if (has_supp) {
                uint32_t off = ri;
                for (uint64_t j = 0; j < zm_supp_npg; j++) {
                    uint64_t pg_id = zm_supp_start + j;
                    uint32_t dev = pg_id % sess.n_devices;
                    uint64_t local = pg_id / sess.n_devices;
                    zm_ctx.h_reads[ri++] = {
                        sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                        static_cast<uint32_t>(ps / 512), dev};
                }
                zm_ctx.h_preds[zm_npreds++] = {off, zm_supp_nstats, zm_supp_lo, zm_supp_hi};
            }
            if (has_part) {
                uint32_t off = ri;
                for (uint64_t j = 0; j < zm_part_npg; j++) {
                    uint64_t pg_id = zm_part_start + j;
                    uint32_t dev = pg_id % sess.n_devices;
                    uint64_t local = pg_id / sess.n_devices;
                    zm_ctx.h_reads[ri++] = {
                        sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                        static_cast<uint32_t>(ps / 512), dev};
                }
                zm_ctx.h_preds[zm_npreds++] = {off, zm_part_nstats, zm_part_lo, zm_part_hi};
            }
            zm_nreads = ri;
            zm_valid = true;
        }
    }

    // Pre-allocate dict download buffers (Rule 4: outside timing)
    DimDictRaw cust_dr = dim_dict_raw_alloc();
    DimDictRaw supp_dr = dim_dict_raw_alloc();
    DimDictRaw part_dr = dim_dict_raw_alloc();

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, lo_ctx.stream_io[0]);
    }

    // Timing starts here to include dim I/O (Rule 1)
    s_kernel_launches = 0;
    uint64_t dim_io_bytes = 0;
    auto outer_start = std::chrono::steady_clock::now();
    bam_build_dim_q4x_gpu(date, cust, supp, part, sess, query,
                          cust_dr, supp_dr, part_dr, dim_bufs4, s_kernel_launches, dim_io_bytes);
    int32_t ncd = std::max(1u, cust_dr.n), nsd = std::max(1u, supp_dr.n), npd = std::max(1u, part_dr.n);
    int32_t sy = ncd * nsd * npd;
    int32_t total_groups = SSB_Q4X_MAX_YEARS * sy;
    auto dim_end = std::chrono::steady_clock::now();

    // Fused BaM zonemap eval (Rule 6: IO + eval inside timing)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages4, zm_nreads, zm_npreds, lo_ctx.stream_io[0]);
        CUDA_CHECK(cudaStreamSynchronize(lo_ctx.stream_io[0]));
        s_kernel_launches++;
        uint32_t active_count = 0;
        for (size_t pg = 0; pg < npages4; pg++) {
            page_active[pg] = zm_ctx.h_mask[pg];
            if (zm_ctx.h_mask[pg]) active_count++;
        }
        std::cout << "[ZONEMAP] Q4x pruning: active=" << active_count << "/" << npages4 << std::endl;
    }

    auto result = bam_lo_pipeline<SSB::query::q4x::NUM_LO_ACTIVE_FIELDS>(
        options, sess, SSB::query::q4x::LO_FIELDS, page_active,
        lo_ctx,
        [&](void **db, size_t np, uint64_t, cudaStream_t s) {
            ssb_q4x_paged(db[0], db[1], db[2], db[3], db[4], db[5], np, ps,
                date.d_keys, date.d_values, date.mask,
                cust.d_keys, cust.d_values, cust.mask,
                supp.d_keys, supp.d_values, supp.mask,
                part.d_keys, part.d_values, part.mask,
                nsd, npd, sy, total_groups, d_prof, s);
        });
    int64_t h_prof[SSB_Q4X_MAX_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_prof, d_prof, sizeof(h_prof), cudaMemcpyDeviceToHost));
    auto outer_end = std::chrono::steady_clock::now();

    // Construct dict strings outside timing (Rule 4)
    std::vector<std::string> cust_dict = dim_build_dict_strings(cust_dr);
    std::vector<std::string> supp_dict = dim_build_dict_strings(supp_dr);
    std::vector<std::string> part_dict = dim_build_dict_strings(part_dr);

    {
        auto dim_ms = std::chrono::duration<double, std::milli>(dim_end - outer_start).count();
        auto lo_ms  = std::chrono::duration<double, std::milli>(outer_end - dim_end).count();
        auto tot_ms = std::chrono::duration<double, std::milli>(outer_end - outer_start).count();
        std::cout << "[Q4x-TIMING] dim=" << dim_ms << "ms  lo_pipeline=" << lo_ms
                  << "ms  total=" << tot_ms << "ms" << std::endl;
    }
    result.elapsed_nanoseconds = std::chrono::duration<int64_t, std::nano>(outer_end - outer_start).count();
    result.read_bytes += dim_io_bytes;
    {
        double elapsed = result.elapsed_nanoseconds / 1e9;
        std::cout << "\n========================================\n"
                  << "Total elapsed: " << elapsed << " seconds\n"
                  << "Total I/Os: " << result.nios << "\n"
                  << "Total bytes read: " << result.read_bytes
                  << "\n========================================" << std::endl;
    }
    std::cout << "\nSSB Q4.x results:" << std::endl;
    for (int32_t y = 0; y < SSB_Q4X_MAX_YEARS; y++)
        for (int32_t c = 0; c < ncd; c++)
            for (int32_t si = 0; si < nsd; si++)
                for (int32_t p = 0; p < npd; p++) {
                    int64_t v = h_prof[y*sy + c*(nsd*npd) + si*npd + p];
                    if (v != 0) {
                        std::cout << "  " << (SSB_YEAR_MIN+y);
                        if (cust_dict[0] != "_") std::cout << " | " << cust_dict[c];
                        if (supp_dict[0] != "_") std::cout << " | " << supp_dict[si];
                        if (part_dict[0] != "_") std::cout << " | " << part_dict[p];
                        std::cout << " | " << v << std::endl;
                    }
                }

    date.free_all(); cust.free_all(); supp.free_all(); part.free_all();
    dim_gpu_bufs_free(dim_bufs4);
    dim_dict_raw_free(cust_dr); dim_dict_raw_free(supp_dr); dim_dict_raw_free(part_dr);
    bam_lo_pipe_ctx_free(lo_ctx);
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    CUDA_CHECK(cudaFree(d_prof));
    bam_session_close(sess);
    return result;
}

} // namespace SsbGidpBam

// ============================================================
// Public wrapper functions
// ============================================================
BenchmarkResult ssb_q11_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q1x_gidp_bam(options, SSB::Query::Q11);
}
BenchmarkResult ssb_q12_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q1x_gidp_bam(options, SSB::Query::Q12);
}
BenchmarkResult ssb_q13_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q1x_gidp_bam(options, SSB::Query::Q13);
}
BenchmarkResult ssb_q21_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q2x_gidp_bam(options, SSB::Query::Q21);
}
BenchmarkResult ssb_q22_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q2x_gidp_bam(options, SSB::Query::Q22);
}
BenchmarkResult ssb_q23_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q2x_gidp_bam(options, SSB::Query::Q23);
}
BenchmarkResult ssb_q31_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q3x_gidp_bam(options, SSB::Query::Q31);
}
BenchmarkResult ssb_q32_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q3x_gidp_bam(options, SSB::Query::Q32);
}
BenchmarkResult ssb_q33_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q3x_gidp_bam(options, SSB::Query::Q33);
}
BenchmarkResult ssb_q34_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q3x_gidp_bam(options, SSB::Query::Q34);
}
BenchmarkResult ssb_q41_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q4x_gidp_bam(options, SSB::Query::Q41);
}
BenchmarkResult ssb_q42_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q4x_gidp_bam(options, SSB::Query::Q42);
}
BenchmarkResult ssb_q43_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q4x_gidp_bam(options, SSB::Query::Q43);
}
BenchmarkResult ssb_revenue_gidp_bam(BenchmarkOptions &options) {
    return SsbGidpBam::ssb_q1x_gidp_bam(options, SSB::Query::REVENUE);
}
