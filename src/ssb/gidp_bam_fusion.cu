#pragma once

// gidp_bam_fusion.cu — SSB GIDP+BAM+FUSION execution mode
// I/O: BaM (GPU-initiated), Decompression: nvCOMPdx (device-side LZ4)
// Single persistent warp-specialized kernel: IO+Decomp+Scan fused
//
// Included AFTER gidp_bam.cu in ssb_main.cu, so SsbGidpBam:: helpers are accessible.

#include "ssb/bam_lz4_fused_ssb.cuh"

namespace SsbGidpBamFusion {

static size_t s_kernel_launches;

using SsbGidpBam::BAMDeviceSetup;
using SsbGidpBam::bam_open_devices;
using SsbGidpBam::DimReadCtx;
using SsbGidpBam::bam_read_field_pages_cpu;
using SsbGidpBam::bam_ssb_read_column_stats;
using SsbGidpBam::bam_ssb_read_sideways_stats;
using SsbGidpBam::BamSessionCtx;
using SsbGidpBam::bam_session_open;
using SsbGidpBam::bam_session_close;
using SsbGidp::GpuHT;
using SsbGidp::make_gpu_ht;
using SsbGidp::alloc_gpu_ht;
using SsbGidp::upload_gpu_ht;
using SsbGidp::ssb_hash32_host;
using SsbGidpBam::DimGpuBufs;
using SsbGidpBam::dim_gpu_bufs_alloc;
using SsbGidpBam::dim_gpu_bufs_free;
using SsbGidpBam::bam_build_date_ht_gpu;
using SsbGidpBam::bam_build_date_ht_ext_gpu;
using SsbGidpBam::bam_build_date_ht_q1x_gpu;
using SsbGidpBam::bam_build_dim_q3x_gpu;
using SsbGidpBam::bam_build_dim_q4x_gpu;
using SsbGidpBam::DimDictRaw;
using SsbGidpBam::dim_dict_raw_alloc;
using SsbGidpBam::dim_dict_raw_free;
using SsbGidpBam::dim_build_dict_strings;
using SsbGidpBam::bam_ssb_read_column_stats;
// collect_compression_methods from gidp.cu takes FieldPageInfo lists;
// we build it directly from metadata instead.

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

// ============================================================
// Per-field GPU prefix sum cache for fused dim kernels.
// Loaded once at session open (Rule 3: metadata outside timing),
// kept on GPU and reused across queries.
// ============================================================
struct FusionDimPrefixSumCache {
    std::map<uint32_t, uint64_t*> d_ps;  // key = table<<8|field → GPU array
    std::map<uint32_t, uint32_t>  np;    // key → npages

    void load(const BamSessionCtx &s) {
        using T = SSB::common::Table;
        auto &m = s.metadata();
        uint32_t bpp = static_cast<uint32_t>(s.page_size / 512);

        auto make_entries = [&](uint64_t start, size_t n) {
            std::vector<BAMBatchReadEntry> e(n);
            for (size_t j = 0; j < n; j++) {
                uint64_t pg = start + j;
                uint32_t dev = pg % s.n_devices;
                uint64_t local = pg / s.n_devices;
                e[j] = {s.ds.partition_start_lbas[dev] + local * bpp, dev, bpp};
            }
            return e;
        };

        auto load_field = [&](T tbl, uint32_t fi,
                              uint64_t ps_start, uint64_t ps_npg, uint64_t data_npg) {
            if (ps_npg == 0 || data_npg == 0) return;
            std::vector<char> buf(ps_npg * s.page_size);
            auto ent = make_entries(ps_start, ps_npg);
            bam_read_pages_batch_to_host(s.dim.io_ctx,
                static_cast<uint32_t>(s.page_size),
                ent.data(), static_cast<uint32_t>(ps_npg),
                buf.data(), s.dim.stream);
            uint64_t *raw = reinterpret_cast<uint64_t*>(buf.data()) + 1;
            uint32_t key = (static_cast<uint32_t>(tbl) << 8) | fi;
            uint64_t *d_ptr;
            CUDA_CHECK(cudaMalloc(&d_ptr, data_npg * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_ptr, raw, data_npg * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice));
            d_ps[key] = d_ptr;
            np[key] = static_cast<uint32_t>(data_npg);
        };

        // SUPPLIER fields used in SSB queries
        using SF = SSB::common::SupplierField;
        for (uint32_t fi : {(uint32_t)SF::S_SUPPKEY, (uint32_t)SF::S_REGION,
                            (uint32_t)SF::S_NATION, (uint32_t)SF::S_CITY})
            load_field(T::SUPPLIER, fi,
                       m.table_supplier_prefix_sum_start_page_ids[fi],
                       m.table_supplier_prefix_sum_npages[fi],
                       m.table_supplier_npages[fi]);
        // CUSTOMER
        using CF = SSB::common::CustomerField;
        for (uint32_t fi : {(uint32_t)CF::C_CUSTKEY, (uint32_t)CF::C_REGION,
                            (uint32_t)CF::C_NATION, (uint32_t)CF::C_CITY})
            load_field(T::CUSTOMER, fi,
                       m.table_customer_prefix_sum_start_page_ids[fi],
                       m.table_customer_prefix_sum_npages[fi],
                       m.table_customer_npages[fi]);
        // PART
        using PF = SSB::common::PartField;
        for (uint32_t fi : {(uint32_t)PF::P_PARTKEY, (uint32_t)PF::P_MFGR,
                            (uint32_t)PF::P_CATEGORY, (uint32_t)PF::P_BRAND1})
            load_field(T::PART, fi,
                       m.table_part_prefix_sum_start_page_ids[fi],
                       m.table_part_prefix_sum_npages[fi],
                       m.table_part_npages[fi]);
    }

    const uint64_t* get(SSB::common::Table tbl, size_t fi) const {
        uint32_t key = (static_cast<uint32_t>(tbl) << 8) | static_cast<uint32_t>(fi);
        auto it = d_ps.find(key);
        return (it != d_ps.end()) ? it->second : nullptr;
    }

    void free_all() {
        for (auto &[k, p] : d_ps) cudaFree(p);
        d_ps.clear();
        np.clear();
    }
};

// ============================================================
// Fused CHAR filter + HT build kernel for gidp+bam+fusion.
// Iterates over INT32 key pages, binary-searches pre-loaded
// prefix sums to locate filter/group records on CHAR pages.
// 1 block per key page, 256 threads.
// ============================================================
struct FusionDimFusedParams {
    const char *key_pages;
    uint32_t key_npages;
    const uint64_t *key_ps;      // GPU prefix sum [key_npages]
    const char *filt_pages;
    uint32_t filt_npages;
    const uint64_t *filt_ps;     // GPU prefix sum [filt_npages]
    uint32_t filt_field_size;
    uint32_t filt_aligned_size;
    int32_t filter_mode;
    char pred_strs[4][DIM_DICT_MAX_STRLEN];
    uint32_t pred_lens[4];
    uint32_t n_preds;
    const char *group_pages;     // nullptr for 2-field
    uint32_t group_npages;
    const uint64_t *group_ps;    // GPU prefix sum, nullptr for 2-field
    uint32_t group_field_size;
    uint32_t group_aligned_size;
    bool enable_dict;
    uint64_t *d_dict_hashes;
    char     *d_dict_strs;
    uint16_t *d_dict_lens;
    uint32_t *d_dict_type_ids;
    uint32_t *d_id_counter;
    int32_t *ht_keys;
    int32_t *ht_values;
    uint32_t ht_mask;
    int32_t match_value;
    uint32_t page_size;
};

__global__ void fusion_dim_fused_filter_ht_kernel(FusionDimFusedParams p)
{
    uint32_t key_pg = blockIdx.x;
    if (key_pg >= p.key_npages) return;

    uint64_t key_row_base = (key_pg == 0) ? 0 : p.key_ps[key_pg - 1];
    uint32_t nalloc = (uint32_t)(p.key_ps[key_pg] - key_row_base);

    const int32_t *keys = reinterpret_cast<const int32_t *>(
        p.key_pages + (uint64_t)key_pg * p.page_size + sizeof(pag_head));

    for (uint32_t r = threadIdx.x; r < nalloc; r += blockDim.x) {
        uint64_t grow = key_row_base + r;

        // Binary search filter prefix sum
        uint32_t lo = 0, hi = p.filt_npages;
        while (lo < hi) {
            uint32_t mid = (lo + hi) >> 1;
            if (p.filt_ps[mid] <= grow) lo = mid + 1; else hi = mid;
        }
        uint32_t filt_pg = lo;
        uint32_t filt_local = (filt_pg == 0) ? (uint32_t)grow
                                             : (uint32_t)(grow - p.filt_ps[filt_pg - 1]);

        const char *rec = p.filt_pages + (uint64_t)filt_pg * p.page_size
                          + sizeof(pag_head)
                          + (uint64_t)filt_local * p.filt_aligned_size;
        uint32_t dlen = p.filt_field_size;
        while (dlen > 0 && rec[dlen - 1] == ' ') dlen--;

        bool pass = true;
        if (p.filter_mode == DIM_FILT_PREFIX) {
            pass = (dlen >= p.pred_lens[0]);
            for (uint32_t k = 0; pass && k < p.pred_lens[0]; k++)
                if (rec[k] != p.pred_strs[0][k]) pass = false;
        } else if (p.filter_mode == DIM_FILT_EQ) {
            pass = (dlen == p.pred_lens[0]);
            for (uint32_t k = 0; pass && k < dlen; k++)
                if (rec[k] != p.pred_strs[0][k]) pass = false;
        } else if (p.filter_mode == DIM_FILT_IN) {
            pass = false;
            for (uint32_t px = 0; px < p.n_preds && !pass; px++) {
                if (dlen != p.pred_lens[px]) continue;
                bool eq = true;
                for (uint32_t k = 0; k < dlen; k++)
                    if (rec[k] != p.pred_strs[px][k]) { eq = false; break; }
                if (eq) pass = true;
            }
        } else if (p.filter_mode == DIM_FILT_RANGE) {
            int c0 = dim_strcmp_dev(rec, dlen, p.pred_strs[0], p.pred_lens[0]);
            int c1 = dim_strcmp_dev(rec, dlen, p.pred_strs[1], p.pred_lens[1]);
            pass = (c0 >= 0 && c1 <= 0);
        }
        if (!pass) continue;

        int32_t val = p.match_value;
        if (p.enable_dict) {
            const char *dict_rec;
            uint32_t dict_dlen;
            if (p.group_pages) {
                uint32_t glo = 0, ghi = p.group_npages;
                while (glo < ghi) {
                    uint32_t mid = (glo + ghi) >> 1;
                    if (p.group_ps[mid] <= grow) glo = mid + 1; else ghi = mid;
                }
                uint32_t grp_pg = glo;
                uint32_t grp_local = (grp_pg == 0) ? (uint32_t)grow
                                                    : (uint32_t)(grow - p.group_ps[grp_pg - 1]);
                dict_rec = p.group_pages + (uint64_t)grp_pg * p.page_size
                           + sizeof(pag_head)
                           + (uint64_t)grp_local * p.group_aligned_size;
                dict_dlen = p.group_field_size;
            } else {
                dict_rec = rec;
                dict_dlen = p.filt_field_size;
            }
            while (dict_dlen > 0 && dict_rec[dict_dlen - 1] == ' ') dict_dlen--;

            uint64_t h = dim_fnv1a64(dict_rec, dict_dlen);
            uint32_t ds = (uint32_t)h & DIM_DICT_MASK;
            while (true) {
                uint64_t prev = atomicCAS(
                    reinterpret_cast<unsigned long long *>(&p.d_dict_hashes[ds]),
                    (unsigned long long)UINT64_MAX,
                    (unsigned long long)h);
                if (prev == UINT64_MAX) {
                    uint32_t nid = atomicAdd(p.d_id_counter, 1);
                    char *dst = p.d_dict_strs + (uint64_t)ds * DIM_DICT_MAX_STRLEN;
                    for (uint32_t k = 0; k < dict_dlen; k++) dst[k] = dict_rec[k];
                    p.d_dict_lens[ds] = (uint16_t)dict_dlen;
                    __threadfence();
                    p.d_dict_type_ids[ds] = nid;
                    val = (int32_t)nid;
                    break;
                }
                if (prev == h) {
                    uint32_t eid;
                    do { __threadfence(); eid = p.d_dict_type_ids[ds]; }
                    while (eid == UINT32_MAX);
                    val = (int32_t)eid;
                    break;
                }
                ds = (ds + 1) & DIM_DICT_MASK;
            }
        }

        ssb_ht_insert(p.ht_keys, p.ht_values, p.ht_mask, keys[r], val);
    }
}

// Host-side launcher for fused dim filter+HT kernel
static void fusion_dim_fused_filter_ht(
    const char *key_pages, uint32_t key_npages, const uint64_t *d_key_ps,
    const char *filt_pages, uint32_t filt_npages, const uint64_t *d_filt_ps,
    uint32_t page_size, uint32_t filt_field_size,
    int32_t filter_mode, const char *const *preds, uint32_t n_preds,
    bool enable_dict, DimGpuDict *dict,
    const char *group_pages, uint32_t group_npages, const uint64_t *d_group_ps,
    uint32_t group_field_size,
    int32_t *ht_keys, int32_t *ht_values, uint32_t ht_mask,
    int32_t match_value,
    cudaStream_t stream, size_t &kl)
{
    FusionDimFusedParams p{};
    p.key_pages = key_pages;
    p.key_npages = key_npages;
    p.key_ps = d_key_ps;
    p.filt_pages = filt_pages;
    p.filt_npages = filt_npages;
    p.filt_ps = d_filt_ps;
    p.filt_field_size = filt_field_size;
    p.filt_aligned_size = (filt_field_size + 3) & ~3u;
    p.page_size = page_size;
    p.filter_mode = filter_mode;
    p.n_preds = n_preds;
    for (uint32_t i = 0; i < n_preds && i < 4; i++) {
        size_t len = std::min(strlen(preds[i]), (size_t)DIM_DICT_MAX_STRLEN);
        p.pred_lens[i] = (uint32_t)len;
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
    p.group_pages = group_pages;
    p.group_npages = group_npages;
    p.group_ps = d_group_ps;
    p.group_field_size = group_field_size;
    p.group_aligned_size = group_pages ? ((group_field_size + 3) & ~3u) : 0;
    p.ht_keys = ht_keys;
    p.ht_values = ht_values;
    p.ht_mask = ht_mask;
    p.match_value = match_value;

    fusion_dim_fused_filter_ht_kernel<<<key_npages, 256, 0, stream>>>(p);
    kl++;
}

// Wrapper for dim_char_filter_kernel (same as gidp_bam.cu's dim_run_char_filter)
static void fusion_dim_run_char_filter(
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
// Q2x fused dim builder (SUPPLIER fused, PART 2-kernel)
// SUPPLIER: fused filter+HT (1 kernel)
// PART: char_filter + ht_paged (2 kernels, same as gidp+bam)
// ============================================================
static void fusion_build_dim_q2x_gpu(
    GpuHT &supp_ht, GpuHT &part_ht,
    const BamSessionCtx &s, const char *region,
    SSB::Query query, DimDictRaw &brand_dict_raw,
    const FusionDimPrefixSumCache &ps_cache,
    DimGpuBufs &db, size_t &kl, uint64_t &io_bytes)
{
    auto &m = s.metadata();
    const size_t ps = s.page_size;
    using SF = SSB::common::SupplierField;
    using PF = SSB::common::PartField;
    cudaStream_t s0 = s.dim.stream, s1 = s.dim.stream2;

    // SUPPLIER: read + fused filter+HT (1 kernel)
    s.read_dim(SSB::common::Table::SUPPLIER, SF::S_SUPPKEY, 0, db.d_buf_a, kl, io_bytes);
    s.read_dim(SSB::common::Table::SUPPLIER, SF::S_REGION, 0, db.d_buf_b, kl, io_bytes);
    {
        uint32_t key_np = m.table_supplier_npages[SF::S_SUPPKEY];
        uint32_t filt_np = m.table_supplier_npages[SF::S_REGION];
        CUDA_CHECK(cudaMemsetAsync(supp_ht.d_keys, 0xFF,
            (supp_ht.mask + 1) * sizeof(int32_t), s0));
        const char *region_pred = region;
        fusion_dim_fused_filter_ht(
            db.d_buf_a, key_np, ps_cache.get(SSB::common::Table::SUPPLIER, SF::S_SUPPKEY),
            db.d_buf_b, filt_np, ps_cache.get(SSB::common::Table::SUPPLIER, SF::S_REGION),
            ps, SSB::common::S_REGION_SIZE,
            DIM_FILT_PREFIX, &region_pred, 1,
            false, nullptr, nullptr, 0, nullptr, 0,
            supp_ht.d_keys, supp_ht.d_values, supp_ht.mask, 0, s0, kl);
        CUDA_CHECK(cudaStreamSynchronize(s0));
    }

    // PART: 2-kernel path (same as gidp+bam)
    // Read P_BRAND1 (pipe 0) + P_PARTKEY (pipe 1) with overlap
    s.read_dim(SSB::common::Table::PART, PF::P_BRAND1, 0, db.d_buf_a, kl, io_bytes);
    s.read_dim(SSB::common::Table::PART, PF::P_PARTKEY, 1, db.d_buf_b, kl, io_bytes);
    CUDA_CHECK(cudaStreamSynchronize(s0));
    CUDA_CHECK(cudaStreamSynchronize(s1));

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
    fusion_dim_run_char_filter(
        db.d_buf_a, pb_np, ps, SSB::common::P_BRAND1_SIZE,
        nullptr, filt_mode, preds, n_preds,
        need_dict, need_dict ? &db.dict : nullptr,
        nullptr, db.d_filter, need_dict ? db.d_values : nullptr, s0, kl);

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

    if (need_dict) {
        SsbGidpBam::dim_download_dict_raw(db.dict, brand_dict_raw);
    } else {
        brand_dict_raw.n = 0;
        brand_dict_raw.fallback = preds[0];
    }
}


// ============================================================
// Common compression metadata reader + uploader
// ============================================================
struct FusionCompMeta {
    uint32_t* d_comp_sizes[6];
    uint64_t* d_comp_offsets[6];
    uint32_t* h_comp_sizes[6];      // host copy for IO recount after zonemap
    bool      is_compressed[6];
    uint64_t  total_io_bytes;
    uint64_t  total_io_count;
    uint64_t  npages;
    uint32_t  page_size;
};

static FusionCompMeta fusion_read_comp_metadata(
    const BamSessionCtx &sess, const size_t *lo_cols, size_t num_fields,
    uint64_t npages, const std::vector<size_t> &active_pages)
{
    FusionCompMeta cm{};
    memset(cm.d_comp_sizes, 0, sizeof(cm.d_comp_sizes));
    memset(cm.d_comp_offsets, 0, sizeof(cm.d_comp_offsets));
    memset(cm.h_comp_sizes, 0, sizeof(cm.h_comp_sizes));
    memset(cm.is_compressed, 0, sizeof(cm.is_compressed));
    cm.npages = npages;

    auto &m = sess.metadata();
    size_t page_size = sess.page_size;
    cm.page_size = static_cast<uint32_t>(page_size);
    auto read_striped = [&](uint64_t pg_id, void *dst) {
        uint32_t dev = pg_id % sess.n_devices;
        uint64_t local = pg_id / sess.n_devices;
        uint64_t lba = sess.ds.partition_start_lbas[dev] + local * (page_size / 512);
        return bam_read_page(sess.ctrl, page_size, lba, dst, dev);
    };

    auto roundup4096 = [](size_t v) -> size_t {
        return (v + 4095) & ~(size_t)4095;
    };
    auto safe_nblocks = [](uint32_t nblk) -> uint32_t {
        if (nblk > 8 && nblk <= 16) nblk = 24;
        return nblk;
    };

    for (size_t fi = 0; fi < num_fields; fi++) {
        size_t col = lo_cols[fi];
        auto comp_method = static_cast<CompressionMethod>(
            m.table_lineorder_compression_method[col]);
        if (comp_method == CompressionMethod::NONE) {
            cm.is_compressed[fi] = false;
            continue;
        }
        cm.is_compressed[fi] = true;

        uint64_t cs_start = m.table_lineorder_compressed_page_sizes_start_page_ids[col];
        uint64_t cs_npages = m.table_lineorder_compressed_page_sizes_npages[col];
        uint64_t nbase = m.table_lineorder_compression_nbases[col];
        uint64_t base_start = m.table_lineorder_compression_base_start_page_ids[col];
        uint64_t field_start = m.table_lineorder_start_page_ids[col];

        // Read compressed page sizes
        std::vector<char> sizes_buf(cs_npages * page_size);
        for (uint64_t p = 0; p < cs_npages; p++)
            read_striped(cs_start + p, sizes_buf.data() + p * page_size);
        std::vector<uint32_t> h_comp_sizes(
            reinterpret_cast<uint32_t*>(sizes_buf.data()),
            reinterpret_cast<uint32_t*>(sizes_buf.data()) + npages);

        // Read compression bases → compute offsets
        size_t bp_npages = SSB::nbase_to_npages(nbase, page_size);
        std::vector<char> bases_buf(bp_npages * page_size);
        for (size_t p = 0; p < bp_npages; p++)
            read_striped(base_start + p, bases_buf.data() + p * page_size);
        std::vector<size_t> offsets_vec;
        calculate_compressed_offsets(
            reinterpret_cast<size_t*>(bases_buf.data()),
            h_comp_sizes.data(), nbase, npages, page_size,
            field_start, sess.n_devices, offsets_vec);

        // Save host copy for IO recount after zonemap
        cm.h_comp_sizes[fi] = static_cast<uint32_t*>(malloc(npages * sizeof(uint32_t)));
        memcpy(cm.h_comp_sizes[fi], h_comp_sizes.data(), npages * sizeof(uint32_t));

        // Upload to GPU
        CUDA_CHECK(cudaMalloc(&cm.d_comp_sizes[fi], npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(cm.d_comp_sizes[fi], h_comp_sizes.data(),
                    npages * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Convert size_t offsets to uint64_t for GPU
        std::vector<uint64_t> h_offsets64(npages);
        for (size_t i = 0; i < npages; i++)
            h_offsets64[i] = static_cast<uint64_t>(offsets_vec[i]);
        CUDA_CHECK(cudaMalloc(&cm.d_comp_offsets[fi], npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(cm.d_comp_offsets[fi], h_offsets64.data(),
                    npages * sizeof(uint64_t), cudaMemcpyHostToDevice));

        // Compute IO stats for active pages
        for (size_t pg_idx = 0; pg_idx < active_pages.size(); pg_idx++) {
            uint32_t pg = static_cast<uint32_t>(active_pages[pg_idx]);
            uint32_t comp_sz = h_comp_sizes[pg];
            uint32_t nblk = safe_nblocks(
                static_cast<uint32_t>((roundup4096(comp_sz) + 511) / 512));
            cm.total_io_bytes += (uint64_t)nblk * 512;
            cm.total_io_count++;
        }
    }

    // Add uncompressed field IO stats
    size_t blocks_per_page = page_size / 512;
    for (size_t fi = 0; fi < num_fields; fi++) {
        if (cm.is_compressed[fi]) continue;
        for (size_t pg_idx = 0; pg_idx < active_pages.size(); pg_idx++) {
            cm.total_io_bytes += (uint64_t)blocks_per_page * 512;
            cm.total_io_count++;
        }
    }

    return cm;
}

static void fusion_free_comp_meta(FusionCompMeta &cm, size_t num_fields) {
    for (size_t fi = 0; fi < num_fields; fi++) {
        if (cm.d_comp_sizes[fi]) cudaFree(cm.d_comp_sizes[fi]);
        if (cm.d_comp_offsets[fi]) cudaFree(cm.d_comp_offsets[fi]);
        if (cm.h_comp_sizes[fi]) free(cm.h_comp_sizes[fi]);
    }
}

// Recompute IO counts using zonemap mask (only active pages)
static void fusion_recount_io(FusionCompMeta &cm, size_t num_fields, const uint8_t *h_mask) {
    cm.total_io_bytes = 0;
    cm.total_io_count = 0;
    auto roundup4096 = [](size_t v) -> size_t { return (v + 4095) & ~(size_t)4095; };
    auto safe_nblocks = [](uint32_t nblk) -> uint32_t {
        if (nblk > 8 && nblk <= 16) nblk = 24;
        return nblk;
    };
    size_t blocks_per_page = cm.page_size / 512;
    for (uint64_t pg = 0; pg < cm.npages; pg++) {
        if (!h_mask[pg]) continue;
        for (size_t fi = 0; fi < num_fields; fi++) {
            if (cm.is_compressed[fi]) {
                uint32_t comp_sz = cm.h_comp_sizes[fi][pg];
                uint32_t nblk = safe_nblocks(
                    static_cast<uint32_t>((roundup4096(comp_sz) + 511) / 512));
                cm.total_io_bytes += (uint64_t)nblk * 512;
            } else {
                cm.total_io_bytes += (uint64_t)blocks_per_page * 512;
            }
            cm.total_io_count++;
        }
    }
}

// ============================================================
// Upload active page IDs to GPU
// ============================================================
static uint32_t* fusion_upload_active_page_ids(
    const std::vector<size_t> &active_pages)
{
    size_t n = active_pages.size();
    std::vector<uint32_t> h_ids(n);
    for (size_t i = 0; i < n; i++)
        h_ids[i] = static_cast<uint32_t>(active_pages[i]);
    uint32_t *d_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ids, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return d_ids;
}

// ============================================================
// Upload HT to GPU (returns GPU key/value pointers)
// ============================================================
static std::string fusion_collect_comp_str(
    const SSBTableMetadata &m, const size_t *lo_cols, size_t nf)
{
    std::set<std::string> methods;
    for (size_t fi = 0; fi < nf; fi++) {
        auto cm = static_cast<CompressionMethod>(m.table_lineorder_compression_method[lo_cols[fi]]);
        methods.insert(compression_method_name(cm));
    }
    std::string result;
    for (const auto &s : methods) {
        if (!result.empty()) result += "+";
        result += s;
    }
    return result;
}

// fusion_upload_ht removed — make_gpu_ht now handles upload inside builders

// ============================================================
// SSB Q1.x GIDP+BAM+FUSION
// Fields: LO_ORDERDATE, LO_QUANTITY, LO_DISCOUNT, LO_EXTENDEDPRICE (4)
// ============================================================
static BenchmarkResult ssb_q1x_fusion(BenchmarkOptions &options, SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    sess.dim_fused = true;
    const size_t ps = sess.page_size;
    auto &m = sess.metadata();

    // Extract LINEORDER field info
    constexpr size_t NF = 4;
    const size_t lo_cols[NF] = {
        SSB::common::LO_ORDERDATE, SSB::common::LO_QUANTITY,
        SSB::common::LO_DISCOUNT, SSB::common::LO_EXTENDEDPRICE
    };
    uint64_t npages = m.table_lineorder_npages[lo_cols[0]];
    uint64_t field_start_page_ids[NF];
    for (size_t fi = 0; fi < NF; fi++) {
        field_start_page_ids[fi] = m.table_lineorder_start_page_ids[lo_cols[fi]];
        std::cout << "  LO Field " << lo_cols[fi]
                  << ": start_page=" << field_start_page_ids[fi]
                  << " npages=" << m.table_lineorder_npages[lo_cols[fi]]
                  << std::endl;
    }
    if (npages == 0) { bam_session_close(sess); return BenchmarkResult{}; }

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

    // Zonemap metadata extraction (Rule 3: metadata outside timing)
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    int32_t zm_lo = 0, zm_hi = 0;
    uint64_t zm_dk_nstats = 0, zm_dk_start = 0, zm_dk_npg = 0;
    if (options.enable_zonemap) {
        switch (query) {
            case SSB::Query::Q11: zm_lo = 19930101; zm_hi = 19931231; break;
            case SSB::Query::Q12: zm_lo = 19940101; zm_hi = 19940131; break;
            case SSB::Query::Q13: zm_lo = 19940201; zm_hi = 19940214; break;
            case SSB::Query::REVENUE:
                zm_lo = options.q6_sd_low ? options.q6_sd_low : 19920101;
                zm_hi = options.q6_sd_high ? options.q6_sd_high : 19981231; break;
            default: zm_lo = 19920101; zm_hi = 19981231; break;
        }
        zm_dk_nstats = m.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
        zm_dk_start = m.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
        zm_dk_npg = m.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];
    }

    // GPU setup
    size_t gpu_free_before = 0, gd = 0;
    cudaMemGetInfo(&gpu_free_before, &gd);

    // Build DATE hash table
    GpuHT date_ht = alloc_gpu_ht(m.table_date_nrows);
    DimGpuBufs dim_bufs_q1 = dim_gpu_bufs_alloc(ps, m.table_date_nrows, sess.dim.max_pages);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Compression metadata (Rule 3: metadata outside timing)
    std::vector<size_t> all_pages(npages);
    std::iota(all_pages.begin(), all_pages.end(), size_t(0));
    auto cm = fusion_read_comp_metadata(sess, lo_cols, NF, npages, all_pages);

    // Grid + fused context
    uint32_t max_blocks = ssb_fused_q1x_max_blocks(static_cast<uint32_t>(ps));
    uint32_t num_blocks = std::min(static_cast<uint32_t>(npages), max_blocks);
    auto fctx = ssb_fused_create(sess.ctrl, static_cast<uint32_t>(ps),
                                  num_blocks, SSB_WS4_SLOTS_PER_BLOCK);
    void *d_ctrls = ssb_fused_get_d_ctrls(fctx);
    void *d_pc    = ssb_fused_get_d_pc_ptr(fctx);
    const char *pc_base = ssb_fused_get_pc_base(fctx);
    char *d_decomp = ssb_fused_get_decomp_buf(fctx);

    // BaM zonemap ctx: borrows fctx's page_cache (Rule 4: alloc outside timing)
    if (options.enable_zonemap && zm_dk_nstats > 0 && zm_dk_start > 0 && zm_dk_npg > 0) {
        zm_ctx = bam_zonemap_ctx_create(d_ctrls, d_pc, (void*)pc_base,
            static_cast<uint32_t>(ps), npages);
        uint32_t ri = 0;
        for (uint64_t j = 0; j < zm_dk_npg; j++) {
            uint64_t pg_id = zm_dk_start + j;
            uint32_t dev = pg_id % sess.n_devices;
            uint64_t local = pg_id / sess.n_devices;
            zm_ctx.h_reads[ri++] = {
                sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                static_cast<uint32_t>(ps / 512), dev};
        }
        zm_ctx.h_preds[zm_npreds++] = {0, zm_dk_nstats, zm_lo, zm_hi};
        zm_nreads = ri;
        zm_valid = true;
    }

    // Revenue output
    int64_t *d_revenue = nullptr;
    CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

    size_t gpu_free_after = 0;
    cudaMemGetInfo(&gpu_free_after, &gd);
    uint64_t gpu_app_bytes = gpu_free_before - gpu_free_after;

    // Fill params
    SSBFusedQ1xParams p{};
    for (size_t fi = 0; fi < NF; fi++) {
        p.field_start_page_ids[fi] = field_start_page_ids[fi];
        p.d_comp_offsets[fi] = cm.d_comp_offsets[fi];
        p.d_comp_sizes[fi] = cm.d_comp_sizes[fi];
        p.is_compressed[fi] = cm.is_compressed[fi];
    }
    for (uint32_t d = 0; d < sess.n_devices && d < 4; d++)
        p.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
    p.n_devices = sess.n_devices;
    p.page_size = static_cast<uint32_t>(ps);
    p.total_pages = static_cast<uint32_t>(npages);
    p.d_active_page_ids = nullptr;
    p.d_page_mask = nullptr;
    p.d_date_ht_keys = date_ht.d_keys;
    p.d_date_ht_values = date_ht.d_values;
    p.date_ht_mask = date_ht.mask;
    p.disc_lo = disc_lo; p.disc_hi = disc_hi;
    p.qty_lo = qty_lo; p.qty_hi = qty_hi;
    p.d_revenue = d_revenue;

    std::cout << "[FUSION Q1x] num_blocks=" << num_blocks
              << " (npages=" << npages << " zm=" << (zm_valid ? "on" : "off") << ")" << std::endl;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    // ═══════ Launch ═══════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // Fused BaM zonemap eval (Rule 6: IO + eval inside timing, mask stays on GPU)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
        const uint32_t num_active = *zm_ctx.h_num_active;
        std::cout << "[ZONEMAP] Q1x pruning: active=" << num_active << "/" << npages << std::endl;
        p.d_active_page_ids = zm_ctx.d_active_ids;
        p.total_pages = num_active;
        num_blocks = std::min(num_active, num_blocks);
    }

    // Phase 0: Build DATE HT (Rule 1: I/O inside timing)
    uint64_t dim_io_bytes = 0;
    bam_build_date_ht_q1x_gpu(date_ht, sess, dim_bufs_q1, query, s_kernel_launches, dim_io_bytes);

    ssb_fused_q1x_launch(d_ctrls, d_pc, pc_base, d_decomp, p, num_blocks, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t h_revenue = 0;
    CUDA_CHECK(cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost));
    std::cout << "SSB Q1.x revenue: " << h_revenue << std::endl;

    auto total_end = std::chrono::steady_clock::now();

    // Recompute IO counts for active pages only
    if (zm_valid)
        fusion_recount_io(cm, NF, zm_ctx.h_mask);

    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

    std::cout << "\n========================================"
              << "\nTotal elapsed: " << elapsed_ns / 1e9 << " seconds"
              << "\nTotal I/Os: " << cm.total_io_count
              << "\nTotal bytes read: " << (cm.total_io_bytes + dim_io_bytes)
              << "\n========================================" << std::endl;

    // Cleanup
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    date_ht.free_all();
    dim_gpu_bufs_free(dim_bufs_q1);
    ssb_fused_destroy(fctx);
    fusion_free_comp_meta(cm, NF);
    cudaFree(d_revenue);
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_session_close(sess);

    std::string comp_str = fusion_collect_comp_str(m, lo_cols, NF);
    return BenchmarkResult{
        .nios = cm.total_io_count,
        .read_bytes = cm.total_io_bytes + dim_io_bytes,
        .elapsed_nanoseconds = elapsed_ns,
        .compression = comp_str,
        .gpu_mem_bytes = sess.gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = sess.gpu_ctrl_bytes,
        .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NF,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// SSB Q2.x GIDP+BAM+FUSION
// Fields: LO_ORDERDATE, LO_PARTKEY, LO_SUPPKEY, LO_REVENUE (4)
// ============================================================
static BenchmarkResult ssb_q2x_fusion(BenchmarkOptions &options, SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    sess.dim_fused = true;
    const size_t ps = sess.page_size;
    auto &m = sess.metadata();

    constexpr size_t NF = 4;
    const size_t lo_cols[NF] = {
        SSB::common::LO_ORDERDATE, SSB::common::LO_PARTKEY,
        SSB::common::LO_SUPPKEY, SSB::common::LO_REVENUE
    };
    uint64_t npages = m.table_lineorder_npages[lo_cols[0]];
    uint64_t field_start_page_ids[NF];
    for (size_t fi = 0; fi < NF; fi++)
        field_start_page_ids[fi] = m.table_lineorder_start_page_ids[lo_cols[fi]];
    if (npages == 0) { bam_session_close(sess); return BenchmarkResult{}; }

    // Build dimension tables
    const char *supp_region;
    switch (query) {
        case SSB::Query::Q21: supp_region = "AMERICA"; break;
        case SSB::Query::Q22: supp_region = "ASIA"; break;
        case SSB::Query::Q23: supp_region = "EUROPE"; break;
        default: supp_region = "AMERICA"; break;
    }
    // Zonemap metadata extraction (Rule 3: metadata outside timing)
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

    // GPU setup
    size_t gpu_free_before = 0, gd = 0;
    cudaMemGetInfo(&gpu_free_before, &gd);

    GpuHT date_ht = alloc_gpu_ht(m.table_date_nrows);
    GpuHT supp = alloc_gpu_ht(m.table_supplier_nrows);
    GpuHT part = alloc_gpu_ht(m.table_part_nrows);
    DimGpuBufs dim_bufs = dim_gpu_bufs_alloc(ps, m.table_part_nrows, sess.dim.max_pages);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Revenue output
    int64_t *d_rev = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rev, SSB_Q2X_GROUPS * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_rev, 0, SSB_Q2X_GROUPS * sizeof(int64_t)));

    // Compression metadata (Rule 3: metadata outside timing)
    std::vector<size_t> all_pages(npages);
    std::iota(all_pages.begin(), all_pages.end(), size_t(0));
    auto cm = fusion_read_comp_metadata(sess, lo_cols, NF, npages, all_pages);

    // Grid + fused context (use max_blocks; launch_blocks adjusted after zonemap eval)
    uint32_t max_blocks = ssb_fused_q1x_max_blocks(static_cast<uint32_t>(ps));
    uint32_t num_blocks = std::min(static_cast<uint32_t>(npages), max_blocks);
    auto fctx = ssb_fused_create(sess.ctrl, static_cast<uint32_t>(ps),
                                  num_blocks, SSB_WS4_SLOTS_PER_BLOCK);

    // BaM zonemap ctx: borrows fctx's page_cache (Rule 4: alloc outside timing)
    if (options.enable_zonemap) {
        bool has_sr = (zm_sr_dict >= 0 && zm_sr_nstats > 0 && zm_sr_start > 0 && zm_sr_npg > 0);
        bool has_part = (zm_p_lo >= 0 && zm_part_nstats > 0 && zm_part_start > 0 && zm_part_npg > 0);
        if (has_sr || has_part) {
            zm_ctx = bam_zonemap_ctx_create(
                ssb_fused_get_d_ctrls(fctx), ssb_fused_get_d_pc_ptr(fctx),
                (void*)ssb_fused_get_pc_base(fctx),
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

    size_t gpu_free_after = 0;
    cudaMemGetInfo(&gpu_free_after, &gd);
    uint64_t gpu_app_bytes = gpu_free_before - gpu_free_after;

    // Fill params
    SSBFusedQ2xParams p{};
    for (size_t fi = 0; fi < NF; fi++) {
        p.field_start_page_ids[fi] = field_start_page_ids[fi];
        p.d_comp_offsets[fi] = cm.d_comp_offsets[fi];
        p.d_comp_sizes[fi] = cm.d_comp_sizes[fi];
        p.is_compressed[fi] = cm.is_compressed[fi];
    }
    for (uint32_t d = 0; d < sess.n_devices && d < 4; d++)
        p.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
    p.n_devices = sess.n_devices;
    p.page_size = static_cast<uint32_t>(ps);
    p.total_pages = static_cast<uint32_t>(npages);
    p.d_active_page_ids = nullptr;
    p.d_page_mask = nullptr;
    p.d_date_ht_keys = date_ht.d_keys; p.d_date_ht_values = date_ht.d_values; p.date_ht_mask = date_ht.mask;
    p.d_supp_ht_keys = supp.d_keys; p.d_supp_ht_values = supp.d_values; p.supp_ht_mask = supp.mask;
    p.d_part_ht_keys = part.d_keys; p.d_part_ht_values = part.d_values; p.part_ht_mask = part.mask;
    p.d_revenue = d_rev;

    // Pre-allocate dict download buffer (Rule 4: outside timing)
    DimDictRaw brand_dict_raw = dim_dict_raw_alloc();

    std::cout << "[FUSION Q2x] num_blocks=" << num_blocks
              << " (npages=" << npages << " zm=" << (zm_valid ? "on" : "off") << ")" << std::endl;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    // Load prefix sums for fused dim kernels (Rule 3: metadata outside timing)
    FusionDimPrefixSumCache ps_cache;
    ps_cache.load(sess);

    // ═══════ Launch ═══════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // Fused BaM zonemap eval (Rule 6: IO + eval inside timing, mask stays on GPU)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
        const uint32_t num_active = *zm_ctx.h_num_active;
        std::cout << "[ZONEMAP] Q2x pruning: active=" << num_active << "/" << npages << std::endl;
        p.d_active_page_ids = zm_ctx.d_active_ids;
        p.total_pages = num_active;
        num_blocks = std::min(num_active, num_blocks);
    }

    // Phase 0: Read dimension tables + build HTs (Rule 1: I/O inside timing)
    uint64_t dim_io_bytes = 0;
    bam_build_date_ht_gpu(date_ht, sess, dim_bufs, s_kernel_launches, dim_io_bytes);
    fusion_build_dim_q2x_gpu(supp, part, sess, supp_region, query, brand_dict_raw, ps_cache, dim_bufs, s_kernel_launches, dim_io_bytes);

    ssb_fused_q2x_launch(ssb_fused_get_d_ctrls(fctx), ssb_fused_get_d_pc_ptr(fctx),
                          ssb_fused_get_pc_base(fctx), ssb_fused_get_decomp_buf(fctx),
                          p, num_blocks, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t h_rev[SSB_Q2X_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_rev, d_rev, sizeof(h_rev), cudaMemcpyDeviceToHost));

    auto total_end = std::chrono::steady_clock::now();

    // Recompute IO counts for active pages only
    if (zm_valid)
        fusion_recount_io(cm, NF, zm_ctx.h_mask);

    // Construct dict strings outside timing (Rule 4)
    std::vector<std::string> brand_dict = dim_build_dict_strings(brand_dict_raw);

    std::cout << "\nSSB Q2.x results:" << std::endl;
    for (int32_t y = 0; y < SSB_NUM_YEARS; y++)
        for (size_t b = 0; b < brand_dict.size(); b++) {
            int64_t v = h_rev[y * SSB_MAX_BRANDS + b];
            if (v != 0) std::cout << "  " << v << " | " << (SSB_YEAR_MIN+y) << " | " << brand_dict[b] << std::endl;
        }
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

    std::cout << "\n========================================\nTotal elapsed: " << elapsed_ns / 1e9 << " seconds"
              << "\nTotal I/Os: " << cm.total_io_count << "\nTotal bytes read: " << (cm.total_io_bytes + dim_io_bytes)
              << "\n========================================" << std::endl;

    // Cleanup
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    dim_dict_raw_free(brand_dict_raw);
    ssb_fused_destroy(fctx);
    fusion_free_comp_meta(cm, NF);
    ps_cache.free_all();
    date_ht.free_all(); supp.free_all(); part.free_all();
    dim_gpu_bufs_free(dim_bufs);
    cudaFree(d_rev);
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_session_close(sess);

    std::string comp_str = fusion_collect_comp_str(m, lo_cols, NF);
    return BenchmarkResult{
        .nios = cm.total_io_count, .read_bytes = cm.total_io_bytes + dim_io_bytes,
        .elapsed_nanoseconds = elapsed_ns, .compression = comp_str,
        .gpu_mem_bytes = sess.gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = sess.gpu_ctrl_bytes, .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NF,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// SSB Q3.x GIDP+BAM+FUSION
// Fields: LO_ORDERDATE, LO_CUSTKEY, LO_SUPPKEY, LO_REVENUE (4)
// ============================================================
static BenchmarkResult ssb_q3x_fusion(BenchmarkOptions &options, SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    sess.dim_fused = true;
    const size_t ps = sess.page_size;
    auto &m = sess.metadata();

    constexpr size_t NF = 4;
    const size_t lo_cols[NF] = {
        SSB::common::LO_ORDERDATE, SSB::common::LO_CUSTKEY,
        SSB::common::LO_SUPPKEY, SSB::common::LO_REVENUE
    };
    uint64_t npages = m.table_lineorder_npages[lo_cols[0]];
    uint64_t field_start_page_ids[NF];
    for (size_t fi = 0; fi < NF; fi++)
        field_start_page_ids[fi] = m.table_lineorder_start_page_ids[lo_cols[fi]];
    if (npages == 0) { bam_session_close(sess); return BenchmarkResult{}; }

    // Zonemap metadata extraction (Rule 3: metadata outside timing)
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    uint64_t zm_dk_nstats = 0, zm_dk_start = 0, zm_dk_npg = 0;
    int32_t zm_cust_lo = -1, zm_cust_hi = -1;
    uint64_t zm_cust_nstats = 0, zm_cust_start = 0, zm_cust_npg = 0;
    size_t cust_sw_idx = 0;
    int32_t zm_supp_lo = -1, zm_supp_hi = -1;
    uint64_t zm_supp_nstats = 0, zm_supp_start = 0, zm_supp_npg = 0;
    size_t supp_sw_idx = 0;
    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);
        const size_t ref_field = SSB::common::LO_ORDERDATE;

        if (query == SSB::Query::Q34) {
            zm_dk_nstats = m.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
            zm_dk_start = m.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
            zm_dk_npg = m.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];
        }

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

    // GPU setup
    size_t gpu_free_before = 0, gd = 0;
    cudaMemGetInfo(&gpu_free_before, &gd);

    GpuHT date_ht = alloc_gpu_ht(m.table_date_nrows);
    GpuHT cust = alloc_gpu_ht(m.table_customer_nrows);
    GpuHT supp = alloc_gpu_ht(m.table_supplier_nrows);
    DimGpuBufs dim_bufs = dim_gpu_bufs_alloc(ps, std::max(m.table_customer_nrows, m.table_supplier_nrows), sess.dim.max_pages);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int64_t *d_rev = nullptr;
    size_t grp_sz = SSB_Q3X_MAX_GROUPS * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&d_rev, grp_sz));
    CUDA_CHECK(cudaMemset(d_rev, 0, grp_sz));

    // Compression metadata (Rule 3: metadata outside timing)
    std::vector<size_t> all_pages(npages);
    std::iota(all_pages.begin(), all_pages.end(), size_t(0));
    auto cm = fusion_read_comp_metadata(sess, lo_cols, NF, npages, all_pages);

    // Grid + fused context (use max_blocks; launch_blocks adjusted after zonemap eval)
    uint32_t max_blocks = ssb_fused_q1x_max_blocks(static_cast<uint32_t>(ps));
    uint32_t num_blocks = std::min(static_cast<uint32_t>(npages), max_blocks);
    auto fctx = ssb_fused_create(sess.ctrl, static_cast<uint32_t>(ps),
                                  num_blocks, SSB_WS4_SLOTS_PER_BLOCK);

    // BaM zonemap ctx: borrows fctx's page_cache (Rule 4: alloc outside timing)
    if (options.enable_zonemap) {
        bool has_dk = (query == SSB::Query::Q34 && zm_dk_nstats > 0 && zm_dk_start > 0 && zm_dk_npg > 0);
        bool has_cust = (zm_cust_lo >= 0 && zm_cust_nstats > 0 && zm_cust_start > 0 && zm_cust_npg > 0);
        bool has_supp = (zm_supp_lo >= 0 && zm_supp_nstats > 0 && zm_supp_start > 0 && zm_supp_npg > 0);
        if (has_dk || has_cust || has_supp) {
            zm_ctx = bam_zonemap_ctx_create(
                ssb_fused_get_d_ctrls(fctx), ssb_fused_get_d_pc_ptr(fctx),
                (void*)ssb_fused_get_pc_base(fctx),
                static_cast<uint32_t>(ps), npages);
            uint32_t ri = 0;
            if (has_dk) {
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

    size_t gpu_free_after = 0;
    cudaMemGetInfo(&gpu_free_after, &gd);
    uint64_t gpu_app_bytes = gpu_free_before - gpu_free_after;

    // Fill params
    SSBFusedQ3xParams p{};
    for (size_t fi = 0; fi < NF; fi++) {
        p.field_start_page_ids[fi] = field_start_page_ids[fi];
        p.d_comp_offsets[fi] = cm.d_comp_offsets[fi];
        p.d_comp_sizes[fi] = cm.d_comp_sizes[fi];
        p.is_compressed[fi] = cm.is_compressed[fi];
    }
    for (uint32_t d = 0; d < sess.n_devices && d < 4; d++)
        p.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
    p.n_devices = sess.n_devices;
    p.page_size = static_cast<uint32_t>(ps);
    p.total_pages = static_cast<uint32_t>(npages);
    p.d_active_page_ids = nullptr;
    p.d_page_mask = nullptr;
    p.d_date_ht_keys = date_ht.d_keys; p.d_date_ht_values = date_ht.d_values; p.date_ht_mask = date_ht.mask;
    p.d_cust_ht_keys = cust.d_keys; p.d_cust_ht_values = cust.d_values; p.cust_ht_mask = cust.mask;
    p.d_supp_ht_keys = supp.d_keys; p.d_supp_ht_values = supp.d_values; p.supp_ht_mask = supp.mask;
    p.d_revenue = d_rev;

    // Pre-allocate dict download buffers (Rule 4: outside timing)
    DimDictRaw cust_dict_raw = dim_dict_raw_alloc();
    DimDictRaw supp_dict_raw = dim_dict_raw_alloc();

    std::cout << "[FUSION Q3x] num_blocks=" << num_blocks
              << " (npages=" << npages << " zm=" << (zm_valid ? "on" : "off") << ")" << std::endl;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    // ═══════ Launch ═══════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // Fused BaM zonemap eval (Rule 6: IO + eval inside timing, mask stays on GPU)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
        const uint32_t num_active = *zm_ctx.h_num_active;
        std::cout << "[ZONEMAP] Q3x pruning: active=" << num_active << "/" << npages << std::endl;
        p.d_active_page_ids = zm_ctx.d_active_ids;
        p.total_pages = num_active;
        num_blocks = std::min(num_active, num_blocks);
    }

    // Phase 0: Read dimension tables + build HTs (Rule 1: I/O inside timing)
    uint64_t dim_io_bytes = 0;
    bam_build_dim_q3x_gpu(date_ht, cust, supp, sess, query, cust_dict_raw, supp_dict_raw, dim_bufs, s_kernel_launches, dim_io_bytes);
    int32_t ncd = std::max(1u, cust_dict_raw.n);
    int32_t nsd = std::max(1u, supp_dict_raw.n);
    p.num_supp_dims = nsd;
    p.hist_size = (uint32_t)(ncd * nsd * SSB_Q3X_MAX_YEARS);

    ssb_fused_q3x_launch(ssb_fused_get_d_ctrls(fctx), ssb_fused_get_d_pc_ptr(fctx),
                          ssb_fused_get_pc_base(fctx), ssb_fused_get_decomp_buf(fctx),
                          p, num_blocks, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t h_rev[SSB_Q3X_MAX_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_rev, d_rev, grp_sz, cudaMemcpyDeviceToHost));

    auto total_end = std::chrono::steady_clock::now();

    // Recompute IO counts for active pages only
    if (zm_valid)
        fusion_recount_io(cm, NF, zm_ctx.h_mask);

    // Construct dict strings outside timing (Rule 4)
    std::vector<std::string> cust_dict = dim_build_dict_strings(cust_dict_raw);
    std::vector<std::string> supp_dict = dim_build_dict_strings(supp_dict_raw);
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
    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

    std::cout << "\n========================================\nTotal elapsed: " << elapsed_ns / 1e9 << " seconds"
              << "\nTotal I/Os: " << cm.total_io_count << "\nTotal bytes read: " << (cm.total_io_bytes + dim_io_bytes)
              << "\n========================================" << std::endl;

    // Cleanup
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    dim_dict_raw_free(cust_dict_raw); dim_dict_raw_free(supp_dict_raw);
    ssb_fused_destroy(fctx);
    fusion_free_comp_meta(cm, NF);
    date_ht.free_all(); cust.free_all(); supp.free_all();
    dim_gpu_bufs_free(dim_bufs);
    cudaFree(d_rev);
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_session_close(sess);

    std::string comp_str = fusion_collect_comp_str(m, lo_cols, NF);
    return BenchmarkResult{
        .nios = cm.total_io_count, .read_bytes = cm.total_io_bytes + dim_io_bytes,
        .elapsed_nanoseconds = elapsed_ns, .compression = comp_str,
        .gpu_mem_bytes = sess.gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = sess.gpu_ctrl_bytes, .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NF,
        .kernel_launches = s_kernel_launches,
    };
}

// ============================================================
// SSB Q4.x GIDP+BAM+FUSION
// Fields: LO_ORDERDATE, LO_CUSTKEY, LO_PARTKEY, LO_SUPPKEY,
//         LO_REVENUE, LO_SUPPLYCOST (6)
// ============================================================
static BenchmarkResult ssb_q4x_fusion(BenchmarkOptions &options, SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    sess.dim_fused = true;
    const size_t ps = sess.page_size;
    auto &m = sess.metadata();

    constexpr size_t NF = 6;
    const size_t lo_cols[NF] = {
        SSB::common::LO_ORDERDATE, SSB::common::LO_CUSTKEY,
        SSB::common::LO_PARTKEY, SSB::common::LO_SUPPKEY,
        SSB::common::LO_REVENUE, SSB::common::LO_SUPPLYCOST
    };
    uint64_t npages = m.table_lineorder_npages[lo_cols[0]];
    uint64_t field_start_page_ids[NF];
    for (size_t fi = 0; fi < NF; fi++)
        field_start_page_ids[fi] = m.table_lineorder_start_page_ids[lo_cols[fi]];
    if (npages == 0) { bam_session_close(sess); return BenchmarkResult{}; }

    // Zonemap metadata extraction (Rule 3: metadata outside timing)
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    uint64_t zm_dk_nstats = 0, zm_dk_start = 0, zm_dk_npg = 0;
    int32_t zm_cr_id = -1;
    uint64_t zm_cr_nstats = 0, zm_cr_start = 0, zm_cr_npg = 0;
    int32_t zm_supp_lo = -1, zm_supp_hi = -1;
    uint64_t zm_supp_nstats = 0, zm_supp_start = 0, zm_supp_npg = 0;
    size_t supp_sw = 0;
    int32_t zm_part_lo = -1, zm_part_hi = -1;
    uint64_t zm_part_nstats = 0, zm_part_start = 0, zm_part_npg = 0;
    size_t part_sw = 0;
    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);
        const size_t ref_field = SSB::common::LO_ORDERDATE;

        if (query != SSB::Query::Q41) {
            zm_dk_nstats = m.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
            zm_dk_start = m.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
            zm_dk_npg = m.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];
        }

        {
            auto it = dict_maps[SSB::common::LSS_C_REGION].find("AMERICA");
            if (it != dict_maps[SSB::common::LSS_C_REGION].end()) zm_cr_id = it->second;
        }
        zm_cr_nstats = m.table_lineorder_sideways_nstats[ref_field][SSB::common::LSS_C_REGION];
        zm_cr_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][SSB::common::LSS_C_REGION];
        zm_cr_npg = m.table_lineorder_sideways_stats_npages[ref_field][SSB::common::LSS_C_REGION];

        if (query == SSB::Query::Q41 || query == SSB::Query::Q42) {
            supp_sw = SSB::common::LSS_S_REGION;
            auto it = dict_maps[supp_sw].find("AMERICA");
            if (it != dict_maps[supp_sw].end()) zm_supp_lo = zm_supp_hi = it->second;
        } else {
            supp_sw = SSB::common::LSS_S_NATION;
            auto it = dict_maps[supp_sw].find("UNITED STATES");
            if (it != dict_maps[supp_sw].end()) zm_supp_lo = zm_supp_hi = it->second;
        }
        zm_supp_nstats = m.table_lineorder_sideways_nstats[ref_field][supp_sw];
        zm_supp_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][supp_sw];
        zm_supp_npg = m.table_lineorder_sideways_stats_npages[ref_field][supp_sw];

        if (query == SSB::Query::Q41 || query == SSB::Query::Q42) {
            part_sw = SSB::common::LSS_P_MFGR;
            auto it1 = dict_maps[part_sw].find("MFGR#1");
            auto it2 = dict_maps[part_sw].find("MFGR#2");
            if (it1 != dict_maps[part_sw].end() && it2 != dict_maps[part_sw].end()) {
                zm_part_lo = std::min(it1->second, it2->second);
                zm_part_hi = std::max(it1->second, it2->second);
            }
        } else {
            part_sw = SSB::common::LSS_P_CATEGORY;
            auto it = dict_maps[part_sw].find("MFGR#14");
            if (it != dict_maps[part_sw].end()) zm_part_lo = zm_part_hi = it->second;
        }
        zm_part_nstats = m.table_lineorder_sideways_nstats[ref_field][part_sw];
        zm_part_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][part_sw];
        zm_part_npg = m.table_lineorder_sideways_stats_npages[ref_field][part_sw];
    }

    // GPU setup
    size_t gpu_free_before = 0, gd = 0;
    cudaMemGetInfo(&gpu_free_before, &gd);

    GpuHT date_ht = alloc_gpu_ht(m.table_date_nrows);
    GpuHT cust = alloc_gpu_ht(m.table_customer_nrows);
    GpuHT supp = alloc_gpu_ht(m.table_supplier_nrows);
    GpuHT part = alloc_gpu_ht(m.table_part_nrows);
    uint64_t max_dim_nrows = std::max({m.table_customer_nrows, m.table_supplier_nrows, m.table_part_nrows});
    DimGpuBufs dim_bufs = dim_gpu_bufs_alloc(ps, max_dim_nrows, sess.dim.max_pages);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int64_t *d_profit = nullptr;
    size_t grp_sz = SSB_Q4X_MAX_GROUPS * sizeof(int64_t);
    CUDA_CHECK(cudaMalloc(&d_profit, grp_sz));
    CUDA_CHECK(cudaMemset(d_profit, 0, grp_sz));

    // Compression metadata (Rule 3: metadata outside timing)
    std::vector<size_t> all_pages(npages);
    std::iota(all_pages.begin(), all_pages.end(), size_t(0));
    auto cm = fusion_read_comp_metadata(sess, lo_cols, NF, npages, all_pages);

    // Q4x uses 6-field kernel with different max_blocks
    uint32_t max_blocks = ssb_fused_q4x_max_blocks(static_cast<uint32_t>(ps));
    uint32_t num_blocks = std::min(static_cast<uint32_t>(npages), max_blocks);
    auto fctx = ssb_fused_create(sess.ctrl, static_cast<uint32_t>(ps),
                                  num_blocks, SSB_WS6_SLOTS_PER_BLOCK);

    // BaM zonemap ctx: borrows fctx's page_cache (Rule 4: alloc outside timing)
    if (options.enable_zonemap) {
        bool has_dk = (query != SSB::Query::Q41 && zm_dk_nstats > 0 && zm_dk_start > 0 && zm_dk_npg > 0);
        bool has_cr = (zm_cr_id >= 0 && zm_cr_nstats > 0 && zm_cr_start > 0 && zm_cr_npg > 0);
        bool has_supp = (zm_supp_lo >= 0 && zm_supp_nstats > 0 && zm_supp_start > 0 && zm_supp_npg > 0);
        bool has_part = (zm_part_lo >= 0 && zm_part_nstats > 0 && zm_part_start > 0 && zm_part_npg > 0);
        if (has_dk || has_cr || has_supp || has_part) {
            zm_ctx = bam_zonemap_ctx_create(
                ssb_fused_get_d_ctrls(fctx), ssb_fused_get_d_pc_ptr(fctx),
                (void*)ssb_fused_get_pc_base(fctx),
                static_cast<uint32_t>(ps), npages);
            uint32_t ri = 0;
            if (has_dk) {
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
                zm_ctx.h_preds[zm_npreds++] = {off, zm_cr_nstats, zm_cr_id, zm_cr_id};
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

    size_t gpu_free_after = 0;
    cudaMemGetInfo(&gpu_free_after, &gd);
    uint64_t gpu_app_bytes = gpu_free_before - gpu_free_after;

    // Fill params (HT pointers valid from alloc, dimension counts set after builds)
    SSBFusedQ4xParams p{};
    for (size_t fi = 0; fi < NF; fi++) {
        p.field_start_page_ids[fi] = field_start_page_ids[fi];
        p.d_comp_offsets[fi] = cm.d_comp_offsets[fi];
        p.d_comp_sizes[fi] = cm.d_comp_sizes[fi];
        p.is_compressed[fi] = cm.is_compressed[fi];
    }
    for (uint32_t d = 0; d < sess.n_devices && d < 4; d++)
        p.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
    p.n_devices = sess.n_devices;
    p.page_size = static_cast<uint32_t>(ps);
    p.total_pages = static_cast<uint32_t>(npages);
    p.d_active_page_ids = nullptr;
    p.d_page_mask = nullptr;
    p.d_date_ht_keys = date_ht.d_keys; p.d_date_ht_values = date_ht.d_values; p.date_ht_mask = date_ht.mask;
    p.d_cust_ht_keys = cust.d_keys; p.d_cust_ht_values = cust.d_values; p.cust_ht_mask = cust.mask;
    p.d_supp_ht_keys = supp.d_keys; p.d_supp_ht_values = supp.d_values; p.supp_ht_mask = supp.mask;
    p.d_part_ht_keys = part.d_keys; p.d_part_ht_values = part.d_values; p.part_ht_mask = part.mask;
    p.d_profit = d_profit;

    // Pre-allocate dict download buffers (Rule 4: outside timing)
    DimDictRaw cust_dr = dim_dict_raw_alloc();
    DimDictRaw supp_dr = dim_dict_raw_alloc();
    DimDictRaw part_dr = dim_dict_raw_alloc();

    std::cout << "[FUSION Q4x] num_blocks=" << num_blocks
              << " (npages=" << npages << " zm=" << (zm_valid ? "on" : "off") << ")" << std::endl;

    // Pre-issue IO to initialize BaM page_cache DMA registration
    if (zm_valid) {
        bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
    }

    // ═══════ Launch ═══════
    auto total_start = std::chrono::steady_clock::now();
    s_kernel_launches = 0;

    // Fused BaM zonemap eval (Rule 6: IO + eval inside timing, mask stays on GPU)
    if (zm_valid) {
        bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        s_kernel_launches++;
        const uint32_t num_active = *zm_ctx.h_num_active;
        std::cout << "[ZONEMAP] Q4x pruning: active=" << num_active << "/" << npages << std::endl;
        p.d_active_page_ids = zm_ctx.d_active_ids;
        p.total_pages = num_active;
        num_blocks = std::min(num_active, num_blocks);
    }

    // Phase 0: Read dimension tables + build HTs (Rule 1: I/O inside timing)
    uint64_t dim_io_bytes = 0;
    bam_build_dim_q4x_gpu(date_ht, cust, supp, part, sess, query, cust_dr, supp_dr, part_dr, dim_bufs, s_kernel_launches, dim_io_bytes);
    int32_t ncd = std::max(1u, cust_dr.n), nsd = std::max(1u, supp_dr.n), npd = std::max(1u, part_dr.n);
    int32_t stride_year = ncd * nsd * npd;
    p.supp_dims = nsd;
    p.part_dims = npd;
    p.stride_year = stride_year;
    p.hist_size = (uint32_t)(SSB_Q4X_MAX_YEARS * stride_year);

    std::cout << "[FUSION Q4x] stride_year=" << stride_year << std::endl;

    ssb_fused_q4x_launch(ssb_fused_get_d_ctrls(fctx), ssb_fused_get_d_pc_ptr(fctx),
                          ssb_fused_get_pc_base(fctx), ssb_fused_get_decomp_buf(fctx),
                          p, num_blocks, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t h_prof[SSB_Q4X_MAX_GROUPS];
    CUDA_CHECK(cudaMemcpy(h_prof, d_profit, grp_sz, cudaMemcpyDeviceToHost));

    auto total_end = std::chrono::steady_clock::now();

    // Recompute IO counts for active pages only
    if (zm_valid)
        fusion_recount_io(cm, NF, zm_ctx.h_mask);

    // Construct dict strings outside timing (Rule 4)
    std::vector<std::string> cust_dict = dim_build_dict_strings(cust_dr);
    std::vector<std::string> supp_dict = dim_build_dict_strings(supp_dr);
    std::vector<std::string> part_dict = dim_build_dict_strings(part_dr);
    std::cout << "\nSSB Q4.x results:" << std::endl;
    for (int32_t y = 0; y < SSB_Q4X_MAX_YEARS; y++)
        for (int32_t c = 0; c < ncd; c++)
            for (int32_t si = 0; si < nsd; si++)
                for (int32_t pi = 0; pi < npd; pi++) {
                    int64_t v = h_prof[y * stride_year + c * (nsd * npd) + si * npd + pi];
                    if (v != 0) {
                        std::cout << "  " << (SSB_YEAR_MIN + y);
                        if (cust_dict[0] != "_") std::cout << " | " << cust_dict[c];
                        if (supp_dict[0] != "_") std::cout << " | " << supp_dict[si];
                        if (part_dict[0] != "_") std::cout << " | " << part_dict[pi];
                        std::cout << " | " << v << std::endl;
                    }
                }

    auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

    std::cout << "\n========================================\nTotal elapsed: " << elapsed_ns / 1e9 << " seconds"
              << "\nTotal I/Os: " << cm.total_io_count << "\nTotal bytes read: " << (cm.total_io_bytes + dim_io_bytes)
              << "\n========================================" << std::endl;

    // Cleanup
    if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
    dim_dict_raw_free(cust_dr); dim_dict_raw_free(supp_dr); dim_dict_raw_free(part_dr);
    ssb_fused_destroy(fctx);
    fusion_free_comp_meta(cm, NF);
    date_ht.free_all(); cust.free_all(); supp.free_all(); part.free_all();
    dim_gpu_bufs_free(dim_bufs);
    cudaFree(d_profit);
    CUDA_CHECK(cudaStreamDestroy(stream));
    bam_session_close(sess);

    std::string comp_str = fusion_collect_comp_str(m, lo_cols, NF);
    return BenchmarkResult{
        .nios = cm.total_io_count, .read_bytes = cm.total_io_bytes + dim_io_bytes,
        .elapsed_nanoseconds = elapsed_ns, .compression = comp_str,
        .gpu_mem_bytes = sess.gpu_ctrl_bytes + gpu_app_bytes,
        .gpu_ctrl_bytes = sess.gpu_ctrl_bytes, .gpu_app_bytes = gpu_app_bytes,
        .total_pages = npages * NF,
        .kernel_launches = s_kernel_launches,
    };
}

} // namespace SsbGidpBamFusion

// ============================================================
// Public wrapper functions
// ============================================================
BenchmarkResult ssb_q11_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q1x_fusion(options, SSB::Query::Q11);
}
BenchmarkResult ssb_q12_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q1x_fusion(options, SSB::Query::Q12);
}
BenchmarkResult ssb_q13_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q1x_fusion(options, SSB::Query::Q13);
}
BenchmarkResult ssb_q21_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q2x_fusion(options, SSB::Query::Q21);
}
BenchmarkResult ssb_q22_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q2x_fusion(options, SSB::Query::Q22);
}
BenchmarkResult ssb_q23_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q2x_fusion(options, SSB::Query::Q23);
}
BenchmarkResult ssb_q31_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q3x_fusion(options, SSB::Query::Q31);
}
BenchmarkResult ssb_q32_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q3x_fusion(options, SSB::Query::Q32);
}
BenchmarkResult ssb_q33_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q3x_fusion(options, SSB::Query::Q33);
}
BenchmarkResult ssb_q34_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q3x_fusion(options, SSB::Query::Q34);
}
BenchmarkResult ssb_q41_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q4x_fusion(options, SSB::Query::Q41);
}
BenchmarkResult ssb_q42_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q4x_fusion(options, SSB::Query::Q42);
}
BenchmarkResult ssb_q43_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q4x_fusion(options, SSB::Query::Q43);
}
BenchmarkResult ssb_revenue_gidp_bam_fusion(BenchmarkOptions &options) {
    return SsbGidpBamFusion::ssb_q1x_fusion(options, SSB::Query::REVENUE);
}
