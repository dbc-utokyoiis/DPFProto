#pragma once

// datapathfusion.cu — SSB DATAPATHFUSION execution mode
// I/O + Decomp: BaM GPU-initiated NVMe + PFOR decompression (bam_pfor32_io_flatten)
// Scan: flat int32 kernels
//
// Uses the same loaded data as gidp/gidp+bam (GOLAP compression → PFOR for INT32).
// Reads PFOR-compressed pages directly via GPU-initiated I/O and decompresses
// into flat int32 arrays using the PFOR32 fused flatten kernel, then runs
// simple flat-array scan/probe kernels.
//
// Included AFTER gidp_bam_fusion.cu in ssb_main.cu.

#include "tpch/bam_kernel.cuh"
#include "common/fsst_page.h"

namespace SsbDatapathFusion {

using SsbGidpBam::BAMDeviceSetup;
using SsbGidpBam::bam_open_devices;
using SsbGidpBam::BamSessionCtx;
using SsbGidpBam::bam_session_open;
using SsbGidpBam::bam_session_close;
using SsbGidpBam::bam_ssb_read_column_stats;
using SsbGidpBam::bam_ssb_read_sideways_stats;
using SsbGidpBam::bam_build_date_ht_q1x_gpu;
using SsbGidpBam::DimGpuBufs;
using SsbGidpBam::dim_gpu_bufs_alloc;
using SsbGidpBam::dim_gpu_bufs_free;
using SsbGidp::GpuHT;
using SsbGidp::alloc_gpu_ht;

// Zone map helpers
using SsbGidpBam::bam_build_date_ht_ext_gpu;
using SsbGidpBamFusion::fusion_collect_comp_str;
using SsbGidpBamFusion::fusion_recount_io;

// Warp-spec PFOR helpers from fusion
using SsbGidpBamFusion::fusion_read_comp_metadata;
using SsbGidpBamFusion::fusion_free_comp_meta;

} // temporarily close namespace for global-scope header
#include "bam_pfor_fused_ssb.cuh"
#include "bam_lz4_fused_ssb.cuh"
namespace SsbDatapathFusion {

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
// Prefix sum reader (host-side, via BaM CPU reads)
// ============================================================
static std::vector<uint64_t> dpf_read_prefix_sum(
    const BamSessionCtx &sess, size_t lo_col)
{
    auto &m = sess.metadata();
    uint64_t ps_start = m.table_lineorder_prefix_sum_start_page_ids[lo_col];
    uint64_t ps_npg   = m.table_lineorder_prefix_sum_npages[lo_col];
    uint64_t npages   = m.table_lineorder_npages[lo_col];

    if (ps_npg == 0 || npages == 0) return {};

    size_t page_size = sess.page_size;
    std::vector<char> buf(ps_npg * page_size);
    for (uint64_t p = 0; p < ps_npg; p++) {
        uint64_t pg_id = ps_start + p;
        uint32_t dev = pg_id % sess.n_devices;
        uint64_t local = pg_id / sess.n_devices;
        uint64_t lba = sess.ds.partition_start_lbas[dev] + local * (page_size / 512);
        bam_read_page(sess.ctrl, page_size, lba, buf.data() + p * page_size, dev);
    }

    // Layout: [0, ps[0], ps[1], ..., ps[npages-1]] — skip leading 0
    uint64_t *raw = reinterpret_cast<uint64_t*>(buf.data()) + 1;
    return std::vector<uint64_t>(raw, raw + npages);
}

// ============================================================
// Compression metadata reader for PFOR flatten
// ============================================================
struct DpfCompMeta {
    std::vector<uint32_t> h_comp_sizes;
    std::vector<uint64_t> h_comp_offsets;
    uint16_t comp_method;
    bool is_compressed;
};

static DpfCompMeta dpf_read_field_comp(
    const BamSessionCtx &sess, size_t lo_col, uint64_t npages)
{
    auto &m = sess.metadata();
    size_t page_size = sess.page_size;
    DpfCompMeta cm{};

    cm.comp_method = m.table_lineorder_compression_method[lo_col];
    if (cm.comp_method == 0) {
        cm.is_compressed = false;
        return cm;
    }
    cm.is_compressed = true;

    uint64_t cs_start = m.table_lineorder_compressed_page_sizes_start_page_ids[lo_col];
    uint64_t cs_npg   = m.table_lineorder_compressed_page_sizes_npages[lo_col];
    uint64_t nbase    = m.table_lineorder_compression_nbases[lo_col];
    uint64_t base_start = m.table_lineorder_compression_base_start_page_ids[lo_col];
    uint64_t field_start = m.table_lineorder_start_page_ids[lo_col];

    auto read_striped = [&](uint64_t pg_id, void *dst) {
        uint32_t dev = pg_id % sess.n_devices;
        uint64_t local = pg_id / sess.n_devices;
        uint64_t lba = sess.ds.partition_start_lbas[dev] + local * (page_size / 512);
        bam_read_page(sess.ctrl, page_size, lba, dst, dev);
    };

    // Read compressed page sizes
    std::vector<char> sizes_buf(cs_npg * page_size);
    for (uint64_t p = 0; p < cs_npg; p++)
        read_striped(cs_start + p, sizes_buf.data() + p * page_size);
    cm.h_comp_sizes.assign(
        reinterpret_cast<uint32_t*>(sizes_buf.data()),
        reinterpret_cast<uint32_t*>(sizes_buf.data()) + npages);

    // Read compression bases → compute offsets
    size_t bp_npg = SSB::nbase_to_npages(nbase, page_size);
    std::vector<char> bases_buf(bp_npg * page_size);
    for (size_t p = 0; p < bp_npg; p++)
        read_striped(base_start + p, bases_buf.data() + p * page_size);

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        cm.h_comp_sizes.data(), nbase, npages, page_size,
        field_start, sess.n_devices, offsets_vec);

    cm.h_comp_offsets.resize(npages);
    for (size_t i = 0; i < npages; i++)
        cm.h_comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);

    return cm;
}

// ============================================================
// Compute I/O stats for reporting
// ============================================================
static void dpf_compute_io_stats(
    const std::vector<uint8_t> &mask, uint64_t npages,
    const DpfCompMeta *cms, size_t nf, size_t page_size,
    uint64_t &out_ios, uint64_t &out_bytes)
{
    out_ios = 0; out_bytes = 0;
    auto roundup4096 = [](uint64_t v) -> uint64_t {
        return (v + 4095) & ~(uint64_t)4095;
    };
    for (uint64_t pg = 0; pg < npages; pg++) {
        if (!mask[pg]) continue;
        for (size_t fi = 0; fi < nf; fi++) {
            out_ios++;
            if (cms[fi].is_compressed)
                out_bytes += roundup4096(cms[fi].h_comp_sizes[pg]);
            else
                out_bytes += page_size;
        }
    }
}

// ============================================================
// Flat int32 scan kernels for datapathfusion
// ============================================================

// Q1x: SUM(extprice * discount) with date HT + range filters
static constexpr int DPF_Q1X_BLK = 256;

__global__ void dpf_q1x_scan_kernel(
    const int32_t *__restrict__ lo_orderdate,
    const int32_t *__restrict__ lo_quantity,
    const int32_t *__restrict__ lo_discount,
    const int32_t *__restrict__ lo_extendedprice,
    uint64_t nrows,
    const int32_t *d_date_ht_keys, const int32_t *d_date_ht_values, uint32_t date_ht_mask,
    int32_t disc_lo, int32_t disc_hi,
    int32_t qty_lo, int32_t qty_hi,
    int64_t *d_revenue)
{
    uint64_t idx = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;

    using BlockReduce = cub::BlockReduce<int64_t, DPF_Q1X_BLK>;
    __shared__ typename BlockReduce::TempStorage temp;

    int64_t val = 0;
    if (idx < nrows) {
        int32_t od = lo_orderdate[idx];
        int32_t qt = lo_quantity[idx];
        int32_t dc = lo_discount[idx];
        int32_t ep = lo_extendedprice[idx];

        bool date_ok = (ssb_ht_probe(d_date_ht_keys, d_date_ht_values,
                                      date_ht_mask, od) >= 0);
        if (date_ok && dc >= disc_lo && dc <= disc_hi && qt >= qty_lo && qt < qty_hi)
            val = (int64_t)ep * dc;
    }

    int64_t agg = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0)
        atomicAdd(reinterpret_cast<unsigned long long int*>(d_revenue),
                  static_cast<unsigned long long int>(agg));
}

// Q2x: Date lookup + Supp HT + Part HT → group-by revenue
static constexpr int DPF_Q2X_BLK = 256;

__global__ void dpf_q2x_scan_kernel(
    const int32_t *__restrict__ lo_orderdate,
    const int32_t *__restrict__ lo_partkey,
    const int32_t *__restrict__ lo_suppkey,
    const int32_t *__restrict__ lo_revenue,
    uint64_t nrows,
    const int32_t *d_date_ht_keys, const int32_t *d_date_ht_values, uint32_t date_ht_mask,
    const int32_t *d_supp_ht_keys, const int32_t *d_supp_ht_values, uint32_t supp_ht_mask,
    const int32_t *d_part_ht_keys, const int32_t *d_part_ht_values, uint32_t part_ht_mask,
    int64_t *d_revenue_out)
{
    constexpr uint32_t HIST_SIZE = SSB_NUM_YEARS * SSB_MAX_BRANDS;  // 280
    __shared__ int64_t s_hist[HIST_SIZE];
    for (uint32_t i = threadIdx.x; i < HIST_SIZE; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    uint64_t idx = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrows) {
        int32_t od = lo_orderdate[idx];
        int32_t pk = lo_partkey[idx];
        int32_t sk = lo_suppkey[idx];
        int32_t rv = lo_revenue[idx];

        int32_t year_idx = ssb_ht_probe(d_date_ht_keys, d_date_ht_values, date_ht_mask, (int32_t)od);
        if (year_idx >= 0) {
            int32_t sv = ssb_ht_probe(d_supp_ht_keys, d_supp_ht_values, supp_ht_mask, sk);
            if (sv >= 0) {
                int32_t bi = ssb_ht_probe(d_part_ht_keys, d_part_ht_values, part_ht_mask, pk);
                if (bi >= 0) {
                    int32_t gi = year_idx * SSB_MAX_BRANDS + bi;
                    atomicAdd(reinterpret_cast<unsigned long long int*>(&s_hist[gi]),
                              static_cast<unsigned long long int>((int64_t)rv));
                }
            }
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < HIST_SIZE; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int*>(&d_revenue_out[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
}

// Q3x: Date lookup + Cust HT + Supp HT → group-by revenue
static constexpr int DPF_Q3X_BLK = 256;

__global__ void dpf_q3x_scan_kernel(
    const int32_t *__restrict__ lo_orderdate,
    const int32_t *__restrict__ lo_custkey,
    const int32_t *__restrict__ lo_suppkey,
    const int32_t *__restrict__ lo_revenue,
    uint64_t nrows,
    const int32_t *d_date_ht_keys, const int32_t *d_date_ht_values, uint32_t date_ht_mask,
    const int32_t *d_cust_ht_keys, const int32_t *d_cust_ht_values, uint32_t cust_ht_mask,
    const int32_t *d_supp_ht_keys, const int32_t *d_supp_ht_values, uint32_t supp_ht_mask,
    int32_t num_supp_dims,
    uint32_t hist_size,
    int64_t *d_revenue_out)
{
    extern __shared__ int64_t s_hist[];
    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    uint64_t idx = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrows) {
        int32_t od = lo_orderdate[idx];
        int32_t ck = lo_custkey[idx];
        int32_t sk = lo_suppkey[idx];
        int32_t rv = lo_revenue[idx];

        int32_t yi = ssb_ht_probe(d_date_ht_keys, d_date_ht_values, date_ht_mask, (int32_t)od);
        if (yi >= 0) {
            int32_t cd = ssb_ht_probe(d_cust_ht_keys, d_cust_ht_values, cust_ht_mask, ck);
            if (cd >= 0) {
                int32_t sd = ssb_ht_probe(d_supp_ht_keys, d_supp_ht_values, supp_ht_mask, sk);
                if (sd >= 0) {
                    int32_t gi = cd * num_supp_dims * SSB_Q3X_MAX_YEARS + sd * SSB_Q3X_MAX_YEARS + yi;
                    atomicAdd(reinterpret_cast<unsigned long long int*>(&s_hist[gi]),
                              static_cast<unsigned long long int>((int64_t)rv));
                }
            }
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int*>(&d_revenue_out[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
}

// Q4x: Date lookup + Cust + Supp + Part HT → group-by profit
static constexpr int DPF_Q4X_BLK = 256;

__global__ void dpf_q4x_scan_kernel(
    const int32_t *__restrict__ lo_orderdate,
    const int32_t *__restrict__ lo_custkey,
    const int32_t *__restrict__ lo_partkey,
    const int32_t *__restrict__ lo_suppkey,
    const int32_t *__restrict__ lo_revenue,
    const int32_t *__restrict__ lo_supplycost,
    uint64_t nrows,
    const int32_t *d_date_ht_keys, const int32_t *d_date_ht_values, uint32_t date_ht_mask,
    const int32_t *d_cust_ht_keys, const int32_t *d_cust_ht_values, uint32_t cust_ht_mask,
    const int32_t *d_supp_ht_keys, const int32_t *d_supp_ht_values, uint32_t supp_ht_mask,
    const int32_t *d_part_ht_keys, const int32_t *d_part_ht_values, uint32_t part_ht_mask,
    int32_t supp_dims, int32_t part_dims, int32_t stride_year,
    uint32_t hist_size,
    int64_t *d_profit)
{
    extern __shared__ int64_t s_hist[];
    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    uint64_t idx = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nrows) {
        int32_t od = lo_orderdate[idx];
        int32_t ck = lo_custkey[idx];
        int32_t pk = lo_partkey[idx];
        int32_t sk = lo_suppkey[idx];
        int32_t rv = lo_revenue[idx];
        int32_t sc = lo_supplycost[idx];

        int32_t yi = ssb_ht_probe(d_date_ht_keys, d_date_ht_values, date_ht_mask, (int32_t)od);
        if (yi >= 0) {
            int32_t cv = ssb_ht_probe(d_cust_ht_keys, d_cust_ht_values, cust_ht_mask, ck);
            if (cv >= 0) {
                int32_t sv = ssb_ht_probe(d_supp_ht_keys, d_supp_ht_values, supp_ht_mask, sk);
                if (sv >= 0) {
                    int32_t pv = ssb_ht_probe(d_part_ht_keys, d_part_ht_values, part_ht_mask, pk);
                    if (pv >= 0) {
                        int32_t gi = yi * stride_year + cv * (supp_dims * part_dims) + sv * part_dims + pv;
                        int64_t profit = (int64_t)rv - (int64_t)sc;
                        atomicAdd(reinterpret_cast<unsigned long long int*>(&s_hist[gi]),
                                  static_cast<unsigned long long int>(profit));
                    }
                }
            }
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
        if (s_hist[i] != 0)
            atomicAdd(reinterpret_cast<unsigned long long int*>(&d_profit[i]),
                      static_cast<unsigned long long int>(s_hist[i]));
    }
}

// ============================================================
// Common PFOR32 flatten helper
// ============================================================
static void dpf_flatten_field(
    bam_pfor32_io_ctx_t pfor_ctx,
    const BamSessionCtx &sess,
    size_t lo_col,
    uint64_t npages, uint64_t nrows,
    const std::vector<uint64_t> &h_prefix_sum,
    const DpfCompMeta &cm,
    const uint8_t *d_page_active,
    uint64_t *d_prefix_sum_gpu,
    int32_t *d_flat_output,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    auto &m = sess.metadata();

    // Upload prefix sum
    CUDA_CHECK(cudaMemcpyAsync(d_prefix_sum_gpu, h_prefix_sum.data(),
                npages * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

    // Upload comp metadata if needed
    uint32_t *d_cs = nullptr;
    uint64_t *d_co = nullptr;
    if (cm.is_compressed) {
        CUDA_CHECK(cudaMalloc(&d_cs, npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_cs, cm.h_comp_sizes.data(),
                    npages * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMalloc(&d_co, npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_co, cm.h_comp_offsets.data(),
                    npages * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    BAMPfor32FlattenParams fp{};
    fp.partition_start_lba = sess.ds.partition_start_lba;
    for (uint32_t d = 0; d < sess.n_devices && d < MAX_BAM_DEVICES; d++)
        fp.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
    fp.n_devices = sess.n_devices;
    fp.page_size = static_cast<uint32_t>(sess.page_size);
    fp.blocks_per_page = static_cast<uint32_t>(sess.page_size / 512);
    fp.comp_method = cm.comp_method;
    fp.field_start_page_id = m.table_lineorder_start_page_ids[lo_col];
    fp.npages = npages;
    fp.nrows = nrows;
    fp.num_blocks = num_blocks;
    fp.d_prefix_sum = d_prefix_sum_gpu;
    fp.d_comp_sizes = d_cs;
    fp.d_comp_offsets = d_co;

    bam_pfor32_io_flatten_masked_async(pfor_ctx, fp, d_page_active, 0,
                                        d_flat_output, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (d_cs) CUDA_CHECK(cudaFree(d_cs));
    if (d_co) CUDA_CHECK(cudaFree(d_co));
}

// ============================================================
// Dimension table infrastructure for DATAPATHFUSION mode
//
// All processing is GPU-native:
//   INT32 key columns: bam_pfor32_io_flatten → GPU flat array
//   FSST_ROWID string columns: BaM IO → GPU FSST decode kernel
//   HT build: GPU atomicCAS kernel
// ============================================================

// Metadata accessor for any dimension table field
struct DpfDimMeta {
    uint64_t start_page_id, npages, nrows;
    uint16_t comp_method;
    uint64_t cs_start, cs_npg;    // compressed page sizes
    uint64_t nbase, base_start;   // compression bases
    uint64_t ps_start, ps_npg;    // prefix sum
};

static DpfDimMeta dpf_dim_meta(const SSBTableMetadata &m, SSB::common::Table tbl, size_t fi)
{
    DpfDimMeta d{};
    switch (tbl) {
    case SSB::common::Table::SUPPLIER:
        d.start_page_id = m.table_supplier_start_page_ids[fi];
        d.npages = m.table_supplier_npages[fi];
        d.nrows = m.table_supplier_nrows;
        d.comp_method = m.table_supplier_compression_method[fi];
        d.cs_start = m.table_supplier_compressed_page_sizes_start_page_ids[fi];
        d.cs_npg = m.table_supplier_compressed_page_sizes_npages[fi];
        d.nbase = m.table_supplier_compression_nbases[fi];
        d.base_start = m.table_supplier_compression_base_start_page_ids[fi];
        d.ps_start = m.table_supplier_prefix_sum_start_page_ids[fi];
        d.ps_npg = m.table_supplier_prefix_sum_npages[fi];
        break;
    case SSB::common::Table::CUSTOMER:
        d.start_page_id = m.table_customer_start_page_ids[fi];
        d.npages = m.table_customer_npages[fi];
        d.nrows = m.table_customer_nrows;
        d.comp_method = m.table_customer_compression_method[fi];
        d.cs_start = m.table_customer_compressed_page_sizes_start_page_ids[fi];
        d.cs_npg = m.table_customer_compressed_page_sizes_npages[fi];
        d.nbase = m.table_customer_compression_nbases[fi];
        d.base_start = m.table_customer_compression_base_start_page_ids[fi];
        d.ps_start = m.table_customer_prefix_sum_start_page_ids[fi];
        d.ps_npg = m.table_customer_prefix_sum_npages[fi];
        break;
    case SSB::common::Table::PART:
        d.start_page_id = m.table_part_start_page_ids[fi];
        d.npages = m.table_part_npages[fi];
        d.nrows = m.table_part_nrows;
        d.comp_method = m.table_part_compression_method[fi];
        d.cs_start = m.table_part_compressed_page_sizes_start_page_ids[fi];
        d.cs_npg = m.table_part_compressed_page_sizes_npages[fi];
        d.nbase = m.table_part_compression_nbases[fi];
        d.base_start = m.table_part_compression_base_start_page_ids[fi];
        d.ps_start = m.table_part_prefix_sum_start_page_ids[fi];
        d.ps_npg = m.table_part_prefix_sum_npages[fi];
        break;
    case SSB::common::Table::DDATE:
        d.start_page_id = m.table_date_start_page_ids[fi];
        d.npages = m.table_date_npages[fi];
        d.nrows = m.table_date_nrows;
        d.comp_method = m.table_date_compression_method[fi];
        d.cs_start = m.table_date_compressed_page_sizes_start_page_ids[fi];
        d.cs_npg = m.table_date_compressed_page_sizes_npages[fi];
        d.nbase = m.table_date_compression_nbases[fi];
        d.base_start = m.table_date_compression_base_start_page_ids[fi];
        d.ps_start = m.table_date_prefix_sum_start_page_ids[fi];
        d.ps_npg = m.table_date_prefix_sum_npages[fi];
        break;
    default: break;
    }
    return d;
}

// Read a single page via striped BaM
static void dpf_read_striped(const BamSessionCtx &sess, uint64_t pg_id,
                              size_t read_size, void *dst)
{
    uint32_t dev = pg_id % sess.n_devices;
    uint64_t local = pg_id / sess.n_devices;
    uint64_t lba = sess.ds.partition_start_lbas[dev] + local * (sess.page_size / 512);
    bam_read_page(sess.ctrl, read_size, lba, dst, dev);
}

// PRP2 danger zone: NVMe interprets PRP2 as direct data address for 9-16 blocks.
// BaM always sets PRP2 as PRP list pointer → corruption. Bump to 17 blocks.
static inline uint32_t dpf_safe_nblocks(uint32_t nblk) {
    if (nblk > 8 && nblk <= 16) return 17;
    return nblk;
}

// Batch-read N striped pages to host using existing pfor_ctx page_cache
static void dpf_batch_read_striped_to_host(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    uint64_t start_page_id, uint64_t n_pages,
    char *h_output, cudaStream_t stream)
{
    if (n_pages == 0) return;
    size_t ps = sess.page_size;
    uint32_t bpp = static_cast<uint32_t>(ps / 512);
    std::vector<BAMBatchReadEntry> entries(n_pages);
    for (uint64_t p = 0; p < n_pages; p++) {
        uint64_t pg_id = start_page_id + p;
        uint32_t dev = pg_id % sess.n_devices;
        uint64_t local = pg_id / sess.n_devices;
        entries[p].lba = sess.ds.partition_start_lbas[dev] + local * bpp;
        entries[p].dev = dev;
        entries[p].nblk = bpp;
    }
    bam_read_pages_batch_to_host(pfor_ctx, static_cast<uint32_t>(ps),
                                  entries.data(), static_cast<uint32_t>(n_pages),
                                  h_output, stream);
}

// Read prefix sum for a dimension table column
static std::vector<uint64_t> dpf_dim_read_prefix_sum(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    const DpfDimMeta &dm, cudaStream_t stream)
{
    if (dm.ps_npg == 0 || dm.npages == 0) return {};
    size_t ps = sess.page_size;
    std::vector<char> buf(dm.ps_npg * ps);
    dpf_batch_read_striped_to_host(pfor_ctx, sess, dm.ps_start, dm.ps_npg,
                                    buf.data(), stream);
    uint64_t *raw = reinterpret_cast<uint64_t*>(buf.data()) + 1;
    return std::vector<uint64_t>(raw, raw + dm.npages);
}

// Read compression metadata for a dimension table column
static DpfCompMeta dpf_dim_read_comp_meta(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    const DpfDimMeta &dm, cudaStream_t stream)
{
    DpfCompMeta cm{};
    cm.comp_method = dm.comp_method;
    if (dm.comp_method == 0) { cm.is_compressed = false; return cm; }
    cm.is_compressed = true;
    size_t ps = sess.page_size;

    std::vector<char> sizes_buf(dm.cs_npg * ps);
    dpf_batch_read_striped_to_host(pfor_ctx, sess, dm.cs_start, dm.cs_npg,
                                    sizes_buf.data(), stream);
    cm.h_comp_sizes.assign(
        reinterpret_cast<uint32_t*>(sizes_buf.data()),
        reinterpret_cast<uint32_t*>(sizes_buf.data()) + dm.npages);

    size_t bp_npg = SSB::nbase_to_npages(dm.nbase, ps);
    std::vector<char> bases_buf(bp_npg * ps);
    dpf_batch_read_striped_to_host(pfor_ctx, sess, dm.base_start, bp_npg,
                                    bases_buf.data(), stream);

    std::vector<size_t> offsets_vec;
    calculate_compressed_offsets(
        reinterpret_cast<size_t*>(bases_buf.data()),
        cm.h_comp_sizes.data(), dm.nbase, dm.npages, ps,
        dm.start_page_id, sess.n_devices, offsets_vec);

    cm.h_comp_offsets.resize(dm.npages);
    for (size_t i = 0; i < dm.npages; i++)
        cm.h_comp_offsets[i] = static_cast<uint64_t>(offsets_vec[i]);

    return cm;
}

// ────────────────────────────────────────────────────────────
// GPU FSST constants and device helpers
// ────────────────────────────────────────────────────────────
static constexpr uint32_t DPF_DIM_MAX_STRLEN = 64;
static constexpr uint32_t DPF_DIM_DICT_CAP = 1024;
static constexpr uint32_t DPF_DIM_DICT_MASK = DPF_DIM_DICT_CAP - 1;
static constexpr uint32_t DPF_FSST_BLOCK_DIM = 256;

static constexpr int32_t DPF_FILT_NONE  = 0;
static constexpr int32_t DPF_FILT_EQ    = 1;
static constexpr int32_t DPF_FILT_IN    = 2;
static constexpr int32_t DPF_FILT_RANGE = 3;
static constexpr int32_t DPF_FILT_PREFIX = 4;

__device__ __forceinline__ uint64_t dpf_fnv1a64(const char *s, uint32_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

__device__ __forceinline__ int dpf_strcmp_dev(
    const char *a, uint32_t alen, const char *b, uint32_t blen)
{
    uint32_t minlen = alen < blen ? alen : blen;
    for (uint32_t i = 0; i < minlen; i++) {
        if ((uint8_t)a[i] != (uint8_t)b[i])
            return (uint8_t)a[i] < (uint8_t)b[i] ? -1 : 1;
    }
    if (alen < blen) return -1;
    if (alen > blen) return 1;
    return 0;
}

// ────────────────────────────────────────────────────────────
// GPU FSST kernel parameters
// ────────────────────────────────────────────────────────────
struct DpfFsstPred {
    char str[DPF_DIM_MAX_STRLEN];
    uint32_t len;
};

struct DpfDimFsstParams {
    const char* staging_buf;
    uint32_t page_size;
    uint32_t npages;
    const uint64_t* d_prefix_sum;
    int32_t filter_mode;
    DpfFsstPred preds[4];
    uint32_t n_preds;
    bool enable_dict;
    uint64_t* d_dict_hashes;
    char* d_dict_strs;
    uint16_t* d_dict_lens;
    uint32_t* d_dict_type_ids;
    uint32_t* d_id_counter;
    const uint8_t* d_prefilter;
    uint8_t* d_filter;
    int32_t* d_values;
    // Optional: direct HT insertion (skip separate build_ht)
    const int32_t* d_flat_keys;  // nullptr = no HT insertion
    int32_t* d_ht_keys;
    int32_t* d_ht_values;
    uint32_t ht_mask;
};

// ────────────────────────────────────────────────────────────
// GPU FSST decode + filter + dict kernel
// One block per page, DPF_FSST_BLOCK_DIM threads per block.
// Shared memory: FSST symbol table (2296 bytes, padded to 2496).
// ────────────────────────────────────────────────────────────
__global__ void dpf_dim_fsst_kernel(DpfDimFsstParams p)
{
    extern __shared__ char smem[];
    uint8_t* s_len = (uint8_t*)smem;
    uint64_t* s_val = (uint64_t*)(smem + 256);

    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (bid >= p.npages) return;

    const char* page = p.staging_buf + (uint64_t)bid * p.page_size;
    const pag_head* hdr = (const pag_head*)page;
    uint32_t nalloc = hdr->nalloc;
    if (nalloc == 0) return;

    uint32_t n_cb = *(const uint32_t*)(page + FSST_PAGE_NCB_OFFSET);
    const FsstCompBlockDirEntry* dir =
        (const FsstCompBlockDirEntry*)(page + FSST_PAGE_NCB_OFFSET + sizeof(uint32_t));
    const char* sym_base = (const char*)(dir + n_cb);

    for (uint32_t i = tid; i < 256; i += blockDim.x)
        s_len[i] = ((const uint8_t*)sym_base)[i];
    for (uint32_t i = tid; i < 255; i += blockDim.x)
        memcpy(&s_val[i], sym_base + FSST_SYMTAB_LEN_BYTES + i * 8, 8);
    __syncthreads();

    uint64_t row_base = (bid == 0) ? 0 : p.d_prefix_sum[bid - 1];

    uint32_t page_row_off = 0;
    for (uint32_t cb = 0; cb < n_cb; cb++) {
        uint32_t cb_off = dir[cb].offset;
        uint32_t cb_nr  = dir[cb].nrecs;
        const uint8_t* cb_data = (const uint8_t*)(page + cb_off);
        const uint16_t* otbl = (const uint16_t*)cb_data;
        const uint8_t* cdata = cb_data + (cb_nr + 1) * sizeof(uint16_t);

        for (uint32_t r = tid; r < cb_nr; r += blockDim.x) {
            uint64_t grow = row_base + page_row_off + r;

            if (p.d_prefilter && !p.d_prefilter[grow]) {
                if (p.d_filter) p.d_filter[grow] = 0;
                if (p.d_values) p.d_values[grow] = -1;
                continue;
            }

            uint16_t cs = otbl[r];
            uint16_t cl = otbl[r + 1] - cs;
            const uint8_t* cptr = cdata + cs;

            char dec[DPF_DIM_MAX_STRLEN];
            uint32_t pi2 = 0, po = 0;
            while (pi2 < cl && po < DPF_DIM_MAX_STRLEN) {
                uint8_t code = cptr[pi2++];
                uint64_t sv; uint8_t sl;
                if (code < 255) { sl = s_len[code]; sv = s_val[code]; }
                else { sv = (uint64_t)cptr[pi2++]; sl = 1; }
                for (uint8_t j = 0; j < sl && po < DPF_DIM_MAX_STRLEN; j++) {
                    dec[po++] = (char)(sv & 0xFF); sv >>= 8;
                }
            }
            while (po > 0 && dec[po - 1] == ' ') po--;
            uint32_t dlen = po;

            bool pass = true;
            if (p.filter_mode == DPF_FILT_EQ) {
                pass = (dlen == p.preds[0].len);
                for (uint32_t k = 0; pass && k < dlen; k++)
                    if (dec[k] != p.preds[0].str[k]) pass = false;
            } else if (p.filter_mode == DPF_FILT_IN) {
                pass = false;
                for (uint32_t px = 0; px < p.n_preds && !pass; px++) {
                    if (dlen != p.preds[px].len) continue;
                    bool eq = true;
                    for (uint32_t k = 0; k < dlen; k++)
                        if (dec[k] != p.preds[px].str[k]) { eq = false; break; }
                    if (eq) pass = true;
                }
            } else if (p.filter_mode == DPF_FILT_RANGE) {
                int c0 = dpf_strcmp_dev(dec, dlen, p.preds[0].str, p.preds[0].len);
                int c1 = dpf_strcmp_dev(dec, dlen, p.preds[1].str, p.preds[1].len);
                pass = (c0 >= 0 && c1 <= 0);
            } else if (p.filter_mode == DPF_FILT_PREFIX) {
                pass = (dlen >= p.preds[0].len);
                for (uint32_t k = 0; pass && k < p.preds[0].len; k++)
                    if (dec[k] != p.preds[0].str[k]) pass = false;
            }

            if (p.d_filter) p.d_filter[grow] = pass ? 1 : 0;

            if (p.d_values) {
                if (!pass) {
                    p.d_values[grow] = -1;
                } else if (p.enable_dict) {
                    uint64_t h = dpf_fnv1a64(dec, dlen);
                    uint32_t ds = (uint32_t)h & DPF_DIM_DICT_MASK;
                    while (true) {
                        uint64_t prev = atomicCAS(
                            reinterpret_cast<unsigned long long*>(&p.d_dict_hashes[ds]),
                            (unsigned long long)UINT64_MAX,
                            (unsigned long long)h);
                        if (prev == UINT64_MAX) {
                            uint32_t nid = atomicAdd(p.d_id_counter, 1);
                            char* dst = p.d_dict_strs + (uint64_t)ds * DPF_DIM_MAX_STRLEN;
                            for (uint32_t k = 0; k < dlen; k++) dst[k] = dec[k];
                            p.d_dict_lens[ds] = (uint16_t)dlen;
                            __threadfence();
                            p.d_dict_type_ids[ds] = nid;
                            p.d_values[grow] = (int32_t)nid;
                            break;
                        }
                        if (prev == h) {
                            uint32_t eid;
                            do { __threadfence(); eid = p.d_dict_type_ids[ds]; }
                            while (eid == UINT32_MAX);
                            p.d_values[grow] = (int32_t)eid;
                            break;
                        }
                        ds = (ds + 1) & DPF_DIM_DICT_MASK;
                    }
                } else {
                    p.d_values[grow] = 0;
                }
            }

            if (pass && p.d_ht_keys != nullptr) {
                int32_t key = p.d_flat_keys[grow];
                int32_t val = p.d_values ? p.d_values[grow] : 0;
                ssb_ht_insert(p.d_ht_keys, p.d_ht_values, p.ht_mask, key, val);
            }
        }
        page_row_off += cb_nr;
    }
}

// ────────────────────────────────────────────────────────────
// GPU HT build kernel (atomicCAS linear probing)
// ────────────────────────────────────────────────────────────
__global__ void dpf_dim_build_ht_kernel(
    const int32_t* __restrict__ d_keys,
    const uint8_t* __restrict__ d_filter,
    const int32_t* __restrict__ d_values,
    uint64_t nrows,
    int32_t* __restrict__ ht_keys,
    int32_t* __restrict__ ht_values,
    uint32_t ht_mask)
{
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nrows) return;
    if (d_filter && !d_filter[gid]) return;
    int32_t val = d_values ? d_values[gid] : 0;
    if (val < 0) return;
    ssb_ht_insert(ht_keys, ht_values, ht_mask, d_keys[gid], val);
}

// ────────────────────────────────────────────────────────────
// GPU dict: FNV-1a dictionary on GPU for group-by value encoding
// ────────────────────────────────────────────────────────────
struct DpfGpuDict {
    uint64_t* d_hashes = nullptr;
    char* d_strs = nullptr;
    uint16_t* d_lens = nullptr;
    uint32_t* d_type_ids = nullptr;
    uint32_t* d_counter = nullptr;
    void alloc() {
        CUDA_CHECK(cudaMalloc(&d_hashes, DPF_DIM_DICT_CAP * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_strs, DPF_DIM_DICT_CAP * DPF_DIM_MAX_STRLEN));
        CUDA_CHECK(cudaMalloc(&d_lens, DPF_DIM_DICT_CAP * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_type_ids, DPF_DIM_DICT_CAP * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_counter, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_hashes, 0xFF, DPF_DIM_DICT_CAP * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_type_ids, 0xFF, DPF_DIM_DICT_CAP * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(uint32_t)));
    }
    void free_all() {
        cudaFree(d_hashes); cudaFree(d_strs); cudaFree(d_lens);
        cudaFree(d_type_ids); cudaFree(d_counter);
    }
};

// ─── Dict raw download (Rule 4: no heap alloc inside timing) ───
struct DpfDictRaw {
    char     *h_strs;
    uint16_t *h_lens;
    uint32_t *h_ids;
    uint32_t n;
    const char *fallback;
};

static DpfDictRaw dpf_dict_raw_alloc() {
    DpfDictRaw r{};
    r.h_strs = (char *)malloc(DPF_DIM_DICT_CAP * DPF_DIM_MAX_STRLEN);
    r.h_lens = (uint16_t *)malloc(DPF_DIM_DICT_CAP * sizeof(uint16_t));
    r.h_ids  = (uint32_t *)malloc(DPF_DIM_DICT_CAP * sizeof(uint32_t));
    r.n = 0;
    r.fallback = nullptr;
    return r;
}

static void dpf_dict_raw_free(DpfDictRaw &r) {
    free(r.h_strs); free(r.h_lens); free(r.h_ids);
    r.h_strs = nullptr; r.h_lens = nullptr; r.h_ids = nullptr;
}

static void dpf_download_dict_raw(const DpfGpuDict &gd, DpfDictRaw &out) {
    CUDA_CHECK(cudaMemcpy(&out.n, gd.d_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (out.n == 0) return;
    CUDA_CHECK(cudaMemcpy(out.h_strs, gd.d_strs, DPF_DIM_DICT_CAP * DPF_DIM_MAX_STRLEN, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.h_lens, gd.d_lens, DPF_DIM_DICT_CAP * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.h_ids,  gd.d_type_ids, DPF_DIM_DICT_CAP * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

static std::vector<std::string> dpf_build_dict_strings(const DpfDictRaw &r) {
    if (r.n == 0) {
        if (r.fallback) return {r.fallback};
        return {"_"};
    }
    std::vector<std::string> result(r.n);
    for (uint32_t slot = 0; slot < DPF_DIM_DICT_CAP; slot++) {
        if (r.h_ids[slot] != UINT32_MAX && r.h_ids[slot] < r.n)
            result[r.h_ids[slot]] = std::string(
                r.h_strs + slot * DPF_DIM_MAX_STRLEN, r.h_lens[slot]);
    }
    return result;
}

// ────────────────────────────────────────────────────────────
// GPU PFOR32 flatten for dim table (result stays on GPU)
// ────────────────────────────────────────────────────────────
static int32_t* dpf_flatten_dim_pfor32_gpu(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    SSB::common::Table tbl, size_t fi,
    uint64_t &out_nrows, uint32_t num_blocks, cudaStream_t stream)
{
    auto dm = dpf_dim_meta(sess.metadata(), tbl, fi);
    if (dm.npages == 0) { out_nrows = 0; return nullptr; }

    auto h_ps = dpf_dim_read_prefix_sum(pfor_ctx, sess, dm, stream);
    out_nrows = h_ps.empty() ? dm.nrows : h_ps.back();
    auto cm = dpf_dim_read_comp_meta(pfor_ctx, sess, dm, stream);

    int32_t *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, out_nrows * sizeof(int32_t)));
    uint64_t *d_ps;
    CUDA_CHECK(cudaMalloc(&d_ps, dm.npages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_ps, h_ps.data(),
                dm.npages * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

    uint32_t *d_cs = nullptr;
    uint64_t *d_co = nullptr;
    if (cm.is_compressed) {
        CUDA_CHECK(cudaMalloc(&d_cs, dm.npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_cs, cm.h_comp_sizes.data(),
                    dm.npages * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMalloc(&d_co, dm.npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_co, cm.h_comp_offsets.data(),
                    dm.npages * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    BAMPfor32FlattenParams fp{};
    fp.partition_start_lba = sess.ds.partition_start_lba;
    for (uint32_t d = 0; d < sess.n_devices && d < MAX_BAM_DEVICES; d++)
        fp.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
    fp.n_devices = sess.n_devices;
    fp.page_size = static_cast<uint32_t>(sess.page_size);
    fp.blocks_per_page = static_cast<uint32_t>(sess.page_size / 512);
    fp.comp_method = cm.comp_method;
    fp.field_start_page_id = dm.start_page_id;
    fp.npages = dm.npages;
    fp.nrows = out_nrows;
    fp.num_blocks = std::min(num_blocks, static_cast<uint32_t>(dm.npages));
    fp.d_prefix_sum = d_ps;
    fp.d_comp_sizes = d_cs;
    fp.d_comp_offsets = d_co;

    bam_pfor32_io_flatten_async(pfor_ctx, fp, d_output, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_ps);
    if (d_cs) cudaFree(d_cs);
    if (d_co) cudaFree(d_co);
    return d_output;
}

// ────────────────────────────────────────────────────────────
// Read FSST pages from BaM to GPU device memory
// ────────────────────────────────────────────────────────────
static char* dpf_dim_read_fsst_to_gpu(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    SSB::common::Table tbl, size_t fi,
    uint64_t &out_npages, uint64_t &out_nrows, cudaStream_t stream)
{
    auto dm = dpf_dim_meta(sess.metadata(), tbl, fi);
    auto cm = dpf_dim_read_comp_meta(pfor_ctx, sess, dm, stream);
    out_npages = dm.npages;
    auto h_ps = dpf_dim_read_prefix_sum(pfor_ctx, sess, dm, stream);
    out_nrows = h_ps.empty() ? dm.nrows : h_ps.back();

    if (dm.npages == 0) return nullptr;
    size_t ps = sess.page_size;
    uint32_t bpp = static_cast<uint32_t>(ps / 512);

    char *d_pages;
    CUDA_CHECK(cudaMalloc(&d_pages, dm.npages * ps));

    // Build batch entries for all pages
    std::vector<BAMBatchReadEntry> entries(dm.npages);
    for (uint64_t pg = 0; pg < dm.npages; pg++) {
        uint64_t logical_page = dm.start_page_id + pg;
        uint32_t dev = logical_page % sess.n_devices;
        if (cm.is_compressed) {
            uint64_t lba = sess.ds.partition_start_lbas[dev] + cm.h_comp_offsets[pg] / 512;
            size_t comp_sz = cm.h_comp_sizes[pg];
            uint32_t nblk = static_cast<uint32_t>((comp_sz + 511) / 512);
            if (nblk == 0) nblk = 1;
            nblk = dpf_safe_nblocks(nblk);
            entries[pg].lba = lba;
            entries[pg].dev = dev;
            entries[pg].nblk = nblk;
        } else {
            uint64_t local = logical_page / sess.n_devices;
            entries[pg].lba = sess.ds.partition_start_lbas[dev] + local * bpp;
            entries[pg].dev = dev;
            entries[pg].nblk = bpp;
        }
    }
    bam_read_pages_batch_to_gpu(pfor_ctx, static_cast<uint32_t>(ps),
                                 entries.data(), static_cast<uint32_t>(dm.npages),
                                 d_pages, stream);
    return d_pages;
}

// ────────────────────────────────────────────────────────────
// Upload prefix sum for an FSST column to GPU
// ────────────────────────────────────────────────────────────
static uint64_t* dpf_dim_upload_prefix_sum(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    SSB::common::Table tbl, size_t fi,
    uint64_t npages, cudaStream_t stream)
{
    auto dm = dpf_dim_meta(sess.metadata(), tbl, fi);
    auto h_ps = dpf_dim_read_prefix_sum(pfor_ctx, sess, dm, stream);
    if (h_ps.empty()) return nullptr;
    uint64_t *d_ps;
    CUDA_CHECK(cudaMalloc(&d_ps, npages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_ps, h_ps.data(),
                npages * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return d_ps;
}

// ────────────────────────────────────────────────────────────
// Run FSST kernel on GPU pages
// ────────────────────────────────────────────────────────────
static void dpf_dim_fsst_run(
    const char* d_pages, uint64_t npages, uint64_t nrows,
    const uint64_t* d_prefix_sum, uint32_t page_size,
    int32_t filter_mode, const std::vector<std::string> &preds,
    bool enable_dict, DpfGpuDict *dict,
    const uint8_t* d_prefilter,
    uint8_t* d_filter, int32_t* d_values,
    const int32_t* d_flat_keys = nullptr,
    int32_t* d_ht_keys = nullptr, int32_t* d_ht_values = nullptr,
    uint32_t ht_mask = 0,
    cudaStream_t stream = nullptr)
{
    DpfDimFsstParams p{};
    p.staging_buf = d_pages;
    p.page_size = page_size;
    p.npages = static_cast<uint32_t>(npages);
    p.d_prefix_sum = d_prefix_sum;
    p.filter_mode = filter_mode;
    p.n_preds = static_cast<uint32_t>(preds.size());
    for (size_t i = 0; i < preds.size() && i < 4; i++) {
        size_t len = std::min(preds[i].size(), (size_t)DPF_DIM_MAX_STRLEN);
        p.preds[i].len = static_cast<uint32_t>(len);
        memcpy(p.preds[i].str, preds[i].data(), len);
    }
    p.enable_dict = enable_dict;
    if (dict) {
        p.d_dict_hashes = dict->d_hashes;
        p.d_dict_strs = dict->d_strs;
        p.d_dict_lens = dict->d_lens;
        p.d_dict_type_ids = dict->d_type_ids;
        p.d_id_counter = dict->d_counter;
    }
    p.d_prefilter = d_prefilter;
    p.d_filter = d_filter;
    p.d_values = d_values;
    p.d_flat_keys = d_flat_keys;
    p.d_ht_keys = d_ht_keys;
    p.d_ht_values = d_ht_values;
    p.ht_mask = ht_mask;

    uint32_t smem_sz = FSST_SMEM_SYMTAB_OFFSET;
    dpf_dim_fsst_kernel<<<npages, DPF_FSST_BLOCK_DIM, smem_sz, stream>>>(p);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ────────────────────────────────────────────────────────────
// Plain CHAR kernel: uncomp dimension pages (pag_head + aligned records)
// Same predicate/dict logic as FSST kernel, no FSST decode.
// ────────────────────────────────────────────────────────────
struct DpfDimPlainCharParams {
    const char* pages;
    uint32_t page_size;
    uint32_t npages;
    uint32_t field_size;          // schema field size (bytes)
    uint32_t aligned_field_size;  // 4-byte aligned
    const uint64_t* d_prefix_sum;
    int32_t filter_mode;
    DpfFsstPred preds[4];
    uint32_t n_preds;
    bool enable_dict;
    uint64_t* d_dict_hashes;
    char* d_dict_strs;
    uint16_t* d_dict_lens;
    uint32_t* d_dict_type_ids;
    uint32_t* d_id_counter;
    const uint8_t* d_prefilter;
    uint8_t* d_filter;
    int32_t* d_values;
    // Optional: direct HT insertion (skip separate build_ht)
    const int32_t* d_flat_keys;  // nullptr = no HT insertion
    int32_t* d_ht_keys;
    int32_t* d_ht_values;
    uint32_t ht_mask;
};

__global__ void dpf_dim_plain_char_kernel(DpfDimPlainCharParams p)
{
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (bid >= p.npages) return;

    const char* page = p.pages + (uint64_t)bid * p.page_size;
    const pag_head* hdr = (const pag_head*)page;
    uint32_t nalloc = hdr->nalloc;
    if (nalloc == 0) return;

    uint64_t row_base = (bid == 0) ? 0 : p.d_prefix_sum[bid - 1];
    const char* rec_base = page + sizeof(pag_head);

    for (uint32_t r = tid; r < nalloc; r += blockDim.x) {
        uint64_t grow = row_base + r;

        if (p.d_prefilter && !p.d_prefilter[grow]) {
            if (p.d_filter) p.d_filter[grow] = 0;
            if (p.d_values) p.d_values[grow] = -1;
            continue;
        }

        const char* rec = rec_base + (uint64_t)r * p.aligned_field_size;

        // Trim trailing spaces
        uint32_t dlen = p.field_size;
        while (dlen > 0 && rec[dlen - 1] == ' ') dlen--;

        bool pass = true;
        if (p.filter_mode == DPF_FILT_EQ) {
            pass = (dlen == p.preds[0].len);
            for (uint32_t k = 0; pass && k < dlen; k++)
                if (rec[k] != p.preds[0].str[k]) pass = false;
        } else if (p.filter_mode == DPF_FILT_IN) {
            pass = false;
            for (uint32_t px = 0; px < p.n_preds && !pass; px++) {
                if (dlen != p.preds[px].len) continue;
                bool eq = true;
                for (uint32_t k = 0; k < dlen; k++)
                    if (rec[k] != p.preds[px].str[k]) { eq = false; break; }
                if (eq) pass = true;
            }
        } else if (p.filter_mode == DPF_FILT_RANGE) {
            int c0 = dpf_strcmp_dev(rec, dlen, p.preds[0].str, p.preds[0].len);
            int c1 = dpf_strcmp_dev(rec, dlen, p.preds[1].str, p.preds[1].len);
            pass = (c0 >= 0 && c1 <= 0);
        } else if (p.filter_mode == DPF_FILT_PREFIX) {
            pass = (dlen >= p.preds[0].len);
            for (uint32_t k = 0; pass && k < p.preds[0].len; k++)
                if (rec[k] != p.preds[0].str[k]) pass = false;
        }

        if (p.d_filter) p.d_filter[grow] = pass ? 1 : 0;

        if (p.d_values) {
            if (!pass) {
                p.d_values[grow] = -1;
            } else if (p.enable_dict) {
                uint64_t h = dpf_fnv1a64(rec, dlen);
                uint32_t ds = (uint32_t)h & DPF_DIM_DICT_MASK;
                while (true) {
                    uint64_t prev = atomicCAS(
                        reinterpret_cast<unsigned long long*>(&p.d_dict_hashes[ds]),
                        (unsigned long long)UINT64_MAX,
                        (unsigned long long)h);
                    if (prev == UINT64_MAX) {
                        uint32_t nid = atomicAdd(p.d_id_counter, 1);
                        char* dst = p.d_dict_strs + (uint64_t)ds * DPF_DIM_MAX_STRLEN;
                        for (uint32_t k = 0; k < dlen; k++) dst[k] = rec[k];
                        p.d_dict_lens[ds] = (uint16_t)dlen;
                        __threadfence();
                        p.d_dict_type_ids[ds] = nid;
                        p.d_values[grow] = (int32_t)nid;
                        break;
                    }
                    if (prev == h) {
                        uint32_t eid;
                        do { __threadfence(); eid = p.d_dict_type_ids[ds]; }
                        while (eid == UINT32_MAX);
                        p.d_values[grow] = (int32_t)eid;
                        break;
                    }
                    ds = (ds + 1) & DPF_DIM_DICT_MASK;
                }
            } else {
                p.d_values[grow] = 0;
            }
        }

        if (pass && p.d_ht_keys != nullptr) {
            int32_t key = p.d_flat_keys[grow];
            int32_t val = p.d_values ? p.d_values[grow] : 0;
            ssb_ht_insert(p.d_ht_keys, p.d_ht_values, p.ht_mask, key, val);
        }
    }
}

static size_t dpf_dim_field_size(SSB::common::Table tbl, size_t fi)
{
    using namespace SSB::common;
    switch (tbl) {
    case Table::CUSTOMER:  return kCustomerFieldSizes[fi];
    case Table::SUPPLIER:  return kSupplierFieldSizes[fi];
    case Table::PART:      return kPartFieldSizes[fi];
    case Table::DDATE:     return kDateFieldSizes[fi];
    default: return 0;
    }
}

static void dpf_dim_plain_char_run(
    const char* d_pages, uint64_t npages, uint64_t nrows,
    const uint64_t* d_prefix_sum, uint32_t page_size,
    uint32_t field_size,
    int32_t filter_mode, const std::vector<std::string> &preds,
    bool enable_dict, DpfGpuDict *dict,
    const uint8_t* d_prefilter,
    uint8_t* d_filter, int32_t* d_values,
    const int32_t* d_flat_keys = nullptr,
    int32_t* d_ht_keys = nullptr, int32_t* d_ht_values = nullptr,
    uint32_t ht_mask = 0,
    cudaStream_t stream = nullptr)
{
    constexpr uint32_t alignment = 4;
    DpfDimPlainCharParams p{};
    p.pages = d_pages;
    p.page_size = page_size;
    p.npages = static_cast<uint32_t>(npages);
    p.field_size = field_size;
    p.aligned_field_size = (field_size + alignment - 1) & ~(alignment - 1);
    p.d_prefix_sum = d_prefix_sum;
    p.filter_mode = filter_mode;
    p.n_preds = static_cast<uint32_t>(preds.size());
    for (size_t i = 0; i < preds.size() && i < 4; i++) {
        size_t len = std::min(preds[i].size(), (size_t)DPF_DIM_MAX_STRLEN);
        p.preds[i].len = static_cast<uint32_t>(len);
        memcpy(p.preds[i].str, preds[i].data(), len);
    }
    p.enable_dict = enable_dict;
    if (dict) {
        p.d_dict_hashes = dict->d_hashes;
        p.d_dict_strs = dict->d_strs;
        p.d_dict_lens = dict->d_lens;
        p.d_dict_type_ids = dict->d_type_ids;
        p.d_id_counter = dict->d_counter;
    }
    p.d_prefilter = d_prefilter;
    p.d_filter = d_filter;
    p.d_values = d_values;
    p.d_flat_keys = d_flat_keys;
    p.d_ht_keys = d_ht_keys;
    p.d_ht_values = d_ht_values;
    p.ht_mask = ht_mask;

    dpf_dim_plain_char_kernel<<<npages, DPF_FSST_BLOCK_DIM, 0, stream>>>(p);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ────────────────────────────────────────────────────────────
// Build GPU HT from d_keys + d_filter + d_values
// ────────────────────────────────────────────────────────────
static GpuHT dpf_build_ht_gpu(
    const int32_t* d_keys, const uint8_t* d_filter,
    const int32_t* d_values, uint64_t nrows, cudaStream_t stream)
{
    uint32_t match_count = 0;
    if (d_filter) {
        std::vector<uint8_t> h_filt(nrows);
        CUDA_CHECK(cudaMemcpy(h_filt.data(), d_filter, nrows, cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < nrows; i++) if (h_filt[i]) match_count++;
    } else {
        match_count = static_cast<uint32_t>(nrows);
    }

    uint32_t ht_sz = 64;
    while (ht_sz < match_count * 2) ht_sz <<= 1;

    GpuHT ht{};
    ht.mask = ht_sz - 1;
    ht.count = match_count;
    CUDA_CHECK(cudaMalloc(&ht.d_keys, ht_sz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ht.d_values, ht_sz * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(ht.d_keys, 0xFF, ht_sz * sizeof(int32_t)));

    uint32_t grid = (nrows + 255) / 256;
    dpf_dim_build_ht_kernel<<<grid, 256, 0, stream>>>(
        d_keys, d_filter, d_values, nrows,
        ht.d_keys, ht.d_values, ht.mask);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return ht;
}

// ════════════════════════════════════════════════════════════
// DpfDimCtx: Pre-allocated workspace for dim HT build.
//
// Usage (QIMPL §9: cudaMalloc/Free must be outside timed section,
//        table reads must be inside):
//
//   Before timed section:  ctx = dpf_dim_prepare_*(...)
//   Inside timed section:  ht  = dpf_dim_run_*(ctx, ...)
//   After timed section:   ctx.free_all(); ht.free_all();
// ════════════════════════════════════════════════════════════
struct DpfDimCtx {
    // Key column (PFOR32 flatten)
    int32_t *d_keys = nullptr;
    uint64_t *d_key_ps = nullptr;
    uint32_t *d_key_cs = nullptr;
    uint64_t *d_key_co = nullptr;
    BAMPfor32FlattenParams key_fp{};

    // String column(s): [0]=filter/single, [1]=dict (two-col case)
    struct StrCol {
        char *d_pages = nullptr;
        uint64_t *d_ps = nullptr;
        void *d_entries = nullptr;   // device copy of entries for persistent kernel
        uint64_t npages = 0;
        bool uncomp = false;
        uint32_t field_size = 0;
        std::vector<BAMBatchReadEntry> entries;
    } str[2];
    int n_str = 0;

    uint8_t *d_filter = nullptr;
    int32_t *d_values = nullptr;

    // HT (pre-allocated with worst-case size)
    int32_t *d_ht_keys = nullptr;
    int32_t *d_ht_values = nullptr;
    uint32_t ht_max_sz = 0;

    uint8_t *h_filt = nullptr;   // pre-allocated host filter buffer (nrows bytes)

    uint64_t nrows = 0;
    uint32_t page_size = 0;
    bool owns_bufs = true;  // false when using DpfDimSharedBufs
    uint64_t io_bytes = 0;  // accumulated dim IO bytes

    void free_all() {
        // Per-table metadata (always owned, tiny)
        if (d_key_ps) cudaFree(d_key_ps);
        if (d_key_cs) cudaFree(d_key_cs);
        if (d_key_co) cudaFree(d_key_co);
        for (int i = 0; i < 2; i++)
            if (str[i].d_ps) cudaFree(str[i].d_ps);
        // Large buffers (only free if owned, not shared)
        if (owns_bufs) {
            if (h_filt) { free(h_filt); h_filt = nullptr; }
            if (d_keys) cudaFree(d_keys);
            for (int i = 0; i < 2; i++) {
                if (str[i].d_pages) cudaFree(str[i].d_pages);
                if (str[i].d_entries) cudaFree(str[i].d_entries);
            }
            if (d_filter) cudaFree(d_filter);
            if (d_values) cudaFree(d_values);
        }
        // HT is always per-table
        if (d_ht_keys) cudaFree(d_ht_keys);
        if (d_ht_values) cudaFree(d_ht_values);
    }
};

// Shared large GPU buffers for dim tables (reused sequentially, Rule 4: alloc outside timing)
struct DpfDimSharedBufs {
    int32_t  *d_keys;           // max_nrows * 4
    char     *d_str_pages[2];   // max_pages * page_size (per str col)
    void     *d_str_entries[2]; // max_pages * sizeof(BAMBatchReadEntry)
    uint8_t  *d_filter;         // max_nrows
    int32_t  *d_values;         // max_nrows * 4
    uint8_t  *h_filt;           // max_nrows (host)
    size_t max_nrows;
    size_t max_pages;
};

static DpfDimSharedBufs dpf_dim_shared_bufs_alloc(
    size_t page_size, size_t max_nrows, size_t max_pages, int max_str_cols = 2)
{
    DpfDimSharedBufs b{};
    b.max_nrows = max_nrows;
    b.max_pages = max_pages;
    CUDA_CHECK(cudaMalloc(&b.d_keys, max_nrows * sizeof(int32_t)));
    for (int i = 0; i < max_str_cols; i++) {
        CUDA_CHECK(cudaMalloc(&b.d_str_pages[i], max_pages * page_size));
        CUDA_CHECK(cudaMalloc(&b.d_str_entries[i], max_pages * sizeof(BAMBatchReadEntry)));
    }
    CUDA_CHECK(cudaMalloc(&b.d_filter, max_nrows));
    CUDA_CHECK(cudaMalloc(&b.d_values, max_nrows * sizeof(int32_t)));
    b.h_filt = (uint8_t *)malloc(max_nrows);
    return b;
}

static void dpf_dim_shared_bufs_free(DpfDimSharedBufs &b) {
    if (b.d_keys) cudaFree(b.d_keys);
    for (int i = 0; i < 2; i++) {
        if (b.d_str_pages[i]) cudaFree(b.d_str_pages[i]);
        if (b.d_str_entries[i]) cudaFree(b.d_str_entries[i]);
    }
    if (b.d_filter) cudaFree(b.d_filter);
    if (b.d_values) cudaFree(b.d_values);
    if (b.h_filt) { free(b.h_filt); b.h_filt = nullptr; }
}

// Prepare one string column: read metadata, alloc page buffer, build batch entries
static void dpf_dim_prepare_str(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    SSB::common::Table tbl, size_t col,
    DpfDimCtx::StrCol &sc, cudaStream_t stream,
    uint64_t &io_bytes,
    DpfDimSharedBufs *shared = nullptr, int str_idx = 0)
{
    size_t ps = sess.page_size;
    auto dm = dpf_dim_meta(sess.metadata(), tbl, col);
    auto cm = dpf_dim_read_comp_meta(pfor_ctx, sess, dm, stream);
    auto h_ps = dpf_dim_read_prefix_sum(pfor_ctx, sess, dm, stream);
    sc.npages = dm.npages;
    sc.uncomp = (dm.comp_method == 0);
    if (sc.uncomp) sc.field_size = static_cast<uint32_t>(dpf_dim_field_size(tbl, col));
    if (dm.npages == 0) return;

    if (shared) {
        sc.d_pages = shared->d_str_pages[str_idx];
        sc.d_entries = shared->d_str_entries[str_idx];
    } else {
        CUDA_CHECK(cudaMalloc(&sc.d_pages, dm.npages * ps));
        CUDA_CHECK(cudaMalloc(&sc.d_entries, dm.npages * sizeof(BAMBatchReadEntry)));
    }

    uint32_t bpp = static_cast<uint32_t>(ps / 512);
    sc.entries.resize(dm.npages);
    for (uint64_t pg = 0; pg < dm.npages; pg++) {
        uint64_t lp = dm.start_page_id + pg;
        uint32_t dev = lp % sess.n_devices;
        if (cm.is_compressed) {
            uint64_t lba = sess.ds.partition_start_lbas[dev] + cm.h_comp_offsets[pg] / 512;
            uint32_t nblk = static_cast<uint32_t>((cm.h_comp_sizes[pg] + 511) / 512);
            if (nblk == 0) nblk = 1;
            nblk = dpf_safe_nblocks(nblk);
            sc.entries[pg] = {lba, dev, nblk};
        } else {
            uint64_t local = lp / sess.n_devices;
            sc.entries[pg] = {sess.ds.partition_start_lbas[dev] + local * bpp, dev, bpp};
        }
    }
    // Accumulate string column IO bytes
    for (uint64_t pg = 0; pg < dm.npages; pg++)
        io_bytes += (uint64_t)sc.entries[pg].nblk * 512;
    if (!h_ps.empty()) {
        CUDA_CHECK(cudaMalloc(&sc.d_ps, dm.npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(sc.d_ps, h_ps.data(),
                    dm.npages * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    }
}

// Prepare key column: read metadata, alloc flatten buffers, fill params
static void dpf_dim_prepare_key(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    SSB::common::Table tbl, size_t key_col, uint32_t num_blocks,
    DpfDimCtx &ctx, cudaStream_t stream,
    DpfDimSharedBufs *shared = nullptr)
{
    size_t ps = sess.page_size;
    auto dm = dpf_dim_meta(sess.metadata(), tbl, key_col);
    if (dm.npages == 0) return;
    auto cm = dpf_dim_read_comp_meta(pfor_ctx, sess, dm, stream);
    auto h_ps = dpf_dim_read_prefix_sum(pfor_ctx, sess, dm, stream);
    ctx.nrows = h_ps.empty() ? dm.nrows : h_ps.back();
    if (ctx.nrows == 0) return;

    if (shared)
        ctx.d_keys = shared->d_keys;
    else
        CUDA_CHECK(cudaMalloc(&ctx.d_keys, ctx.nrows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_key_ps, dm.npages * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_key_ps, h_ps.data(),
                dm.npages * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    if (cm.is_compressed) {
        CUDA_CHECK(cudaMalloc(&ctx.d_key_cs, dm.npages * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_key_cs, cm.h_comp_sizes.data(),
                    dm.npages * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMalloc(&ctx.d_key_co, dm.npages * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_key_co, cm.h_comp_offsets.data(),
                    dm.npages * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    }

    auto &fp = ctx.key_fp;
    fp.partition_start_lba = sess.ds.partition_start_lba;
    for (uint32_t d = 0; d < sess.n_devices && d < MAX_BAM_DEVICES; d++)
        fp.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
    fp.n_devices = sess.n_devices;
    fp.page_size = static_cast<uint32_t>(ps);
    fp.blocks_per_page = static_cast<uint32_t>(ps / 512);
    fp.comp_method = cm.comp_method;
    fp.field_start_page_id = dm.start_page_id;
    fp.npages = dm.npages;
    fp.nrows = ctx.nrows;
    fp.num_blocks = std::min(num_blocks, static_cast<uint32_t>(dm.npages));
    fp.d_prefix_sum = ctx.d_key_ps;
    fp.d_comp_sizes = ctx.d_key_cs;
    fp.d_comp_offsets = ctx.d_key_co;

    // Accumulate key IO bytes
    if (cm.is_compressed) {
        for (uint64_t pg = 0; pg < dm.npages; pg++) {
            uint32_t nblk = static_cast<uint32_t>((cm.h_comp_sizes[pg] + 511) / 512);
            if (nblk == 0) nblk = 1;
            nblk = dpf_safe_nblocks(nblk);
            ctx.io_bytes += (uint64_t)nblk * 512;
        }
    } else {
        ctx.io_bytes += (uint64_t)dm.npages * ps;
    }
}

static DpfDimCtx dpf_dim_prepare_single(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    SSB::common::Table tbl, size_t key_col, size_t str_col,
    bool need_values, uint32_t num_blocks, cudaStream_t stream,
    DpfDimSharedBufs *shared = nullptr)
{
    DpfDimCtx ctx{};
    ctx.page_size = static_cast<uint32_t>(sess.page_size);
    dpf_dim_prepare_key(pfor_ctx, sess, tbl, key_col, num_blocks, ctx, stream, shared);
    if (ctx.nrows == 0) return ctx;

    dpf_dim_prepare_str(pfor_ctx, sess, tbl, str_col, ctx.str[0], stream, ctx.io_bytes, shared, 0);
    ctx.n_str = 1;

    if (shared) {
        ctx.d_filter = shared->d_filter;
        if (need_values) ctx.d_values = shared->d_values;
        ctx.h_filt = shared->h_filt;
        ctx.owns_bufs = false;
    } else {
        CUDA_CHECK(cudaMalloc(&ctx.d_filter, ctx.nrows));
        if (need_values) CUDA_CHECK(cudaMalloc(&ctx.d_values, ctx.nrows * sizeof(int32_t)));
        ctx.h_filt = (uint8_t *)malloc(ctx.nrows);
    }

    ctx.ht_max_sz = 64;
    while (ctx.ht_max_sz < ctx.nrows * 2) ctx.ht_max_sz <<= 1;
    CUDA_CHECK(cudaMalloc(&ctx.d_ht_keys, ctx.ht_max_sz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_ht_values, ctx.ht_max_sz * sizeof(int32_t)));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return ctx;
}

static DpfDimCtx dpf_dim_prepare_two(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    SSB::common::Table tbl, size_t key_col, size_t filt_col, size_t dict_col,
    uint32_t num_blocks, cudaStream_t stream,
    DpfDimSharedBufs *shared = nullptr)
{
    DpfDimCtx ctx{};
    ctx.page_size = static_cast<uint32_t>(sess.page_size);
    dpf_dim_prepare_key(pfor_ctx, sess, tbl, key_col, num_blocks, ctx, stream, shared);
    if (ctx.nrows == 0) return ctx;

    dpf_dim_prepare_str(pfor_ctx, sess, tbl, filt_col, ctx.str[0], stream, ctx.io_bytes, shared, 0);
    dpf_dim_prepare_str(pfor_ctx, sess, tbl, dict_col, ctx.str[1], stream, ctx.io_bytes, shared, 1);
    ctx.n_str = 2;

    if (shared) {
        ctx.d_filter = shared->d_filter;
        ctx.d_values = shared->d_values;
        ctx.h_filt = shared->h_filt;
        ctx.owns_bufs = false;
    } else {
        CUDA_CHECK(cudaMalloc(&ctx.d_filter, ctx.nrows));
        CUDA_CHECK(cudaMalloc(&ctx.d_values, ctx.nrows * sizeof(int32_t)));
        ctx.h_filt = (uint8_t *)malloc(ctx.nrows);
    }

    ctx.ht_max_sz = 64;
    while (ctx.ht_max_sz < ctx.nrows * 2) ctx.ht_max_sz <<= 1;
    CUDA_CHECK(cudaMalloc(&ctx.d_ht_keys, ctx.ht_max_sz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_ht_values, ctx.ht_max_sz * sizeof(int32_t)));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return ctx;
}

// Build HT from pre-allocated buffers (no cudaMalloc)
static GpuHT dpf_dim_build_ht(DpfDimCtx &ctx, cudaStream_t stream)
{
    uint32_t match_count = 0;
    {
        CUDA_CHECK(cudaMemcpy(ctx.h_filt, ctx.d_filter, ctx.nrows, cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < ctx.nrows; i++) if (ctx.h_filt[i]) match_count++;
    }
    uint32_t ht_sz = 64;
    while (ht_sz < match_count * 2) ht_sz <<= 1;
    if (ht_sz > ctx.ht_max_sz) ht_sz = ctx.ht_max_sz;

    GpuHT ht{};
    ht.mask = ht_sz - 1;
    ht.count = match_count;
    ht.d_keys = ctx.d_ht_keys;
    ht.d_values = ctx.d_ht_values;
    ctx.d_ht_keys = nullptr;    // ownership transferred
    ctx.d_ht_values = nullptr;
    CUDA_CHECK(cudaMemset(ht.d_keys, 0xFF, ht_sz * sizeof(int32_t)));

    uint32_t grid = (ctx.nrows + 255) / 256;
    dpf_dim_build_ht_kernel<<<grid, 256, 0, stream>>>(
        ctx.d_keys, ctx.d_filter, ctx.d_values, ctx.nrows,
        ht.d_keys, ht.d_values, ht.mask);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return ht;
}

// Read string column pages via BaM (IO only, no alloc)
static void dpf_dim_read_str_pages(
    DpfDimCtx::StrCol &sc, bam_pfor32_io_ctx_t pfor_ctx,
    uint32_t page_size, cudaStream_t stream)
{
    if (sc.npages == 0) return;
    void *d_ctrls   = bam_pfor32_io_get_d_ctrls(pfor_ctx);
    void *d_pc_ptr  = bam_pfor32_io_get_d_pc_ptr(pfor_ctx);
    const char *pc_base = bam_pfor32_io_get_pc_base(pfor_ctx);
    uint32_t num_slots  = bam_pfor32_io_get_num_slots(pfor_ctx);

    cudaMemcpyAsync(sc.d_entries, sc.entries.data(),
        sc.npages * sizeof(BAMBatchReadEntry), cudaMemcpyHostToDevice, stream);

    bam_dim_io_copy_launch(d_ctrls, d_pc_ptr, pc_base,
        0, num_slots, page_size,
        sc.d_pages, sc.d_entries,
        static_cast<uint32_t>(sc.npages), stream);
    s_kernel_launches++;
}

// Run filter/dict kernel on a string column (FSST or plain CHAR)
static void dpf_dim_filter_str(
    DpfDimCtx::StrCol &sc, uint64_t nrows, uint32_t page_size,
    int32_t filter_mode, const std::vector<std::string> &preds,
    bool enable_dict, DpfGpuDict *dict,
    const uint8_t *d_prefilter, uint8_t *d_filter, int32_t *d_values,
    const int32_t *d_flat_keys = nullptr,
    int32_t *d_ht_keys = nullptr, int32_t *d_ht_values = nullptr,
    uint32_t ht_mask = 0,
    cudaStream_t stream = nullptr)
{
    if (sc.uncomp) {
        dpf_dim_plain_char_run(sc.d_pages, sc.npages, nrows, sc.d_ps,
                               page_size, sc.field_size,
                               filter_mode, preds, enable_dict, dict,
                               d_prefilter, d_filter, d_values,
                               d_flat_keys, d_ht_keys, d_ht_values, ht_mask,
                               stream);
    } else {
        dpf_dim_fsst_run(sc.d_pages, sc.npages, nrows, sc.d_ps,
                          page_size,
                          filter_mode, preds, enable_dict, dict,
                          d_prefilter, d_filter, d_values,
                          d_flat_keys, d_ht_keys, d_ht_values, ht_mask,
                          stream);
    }
}

// Return accumulated dim IO bytes (key + string columns, computed during prepare)
static uint64_t dpf_dim_ctx_io_bytes(const DpfDimCtx &ctx) {
    return ctx.io_bytes;
}

// Run single-column dim HT build (IO + kernels only, no malloc)
static GpuHT dpf_dim_run_single(
    DpfDimCtx &ctx, bam_pfor32_io_ctx_t pfor_ctx,
    int32_t filter_mode, const std::vector<std::string> &preds,
    bool enable_dict, DpfGpuDict *dict,
    cudaStream_t stream)
{
    if (ctx.nrows == 0 || !ctx.d_keys) return GpuHT{};

    bam_pfor32_io_flatten_async(pfor_ctx, ctx.key_fp, ctx.d_keys, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    dpf_dim_read_str_pages(ctx.str[0], pfor_ctx, ctx.page_size, stream);
    dpf_dim_filter_str(ctx.str[0], ctx.nrows, ctx.page_size,
                        filter_mode, preds, enable_dict, dict,
                        nullptr, ctx.d_filter, ctx.d_values,
                        nullptr, nullptr, nullptr, 0, stream);

    return dpf_dim_build_ht(ctx, stream);
}

// Run single-column dim HT build with fused filter+HT (1 kernel fewer)
static GpuHT dpf_dim_run_single_ht(
    DpfDimCtx &ctx, bam_pfor32_io_ctx_t pfor_ctx,
    int32_t filter_mode, const std::vector<std::string> &preds,
    bool enable_dict, DpfGpuDict *dict,
    cudaStream_t stream)
{
    if (ctx.nrows == 0 || !ctx.d_keys) return GpuHT{};

    bam_pfor32_io_flatten_async(pfor_ctx, ctx.key_fp, ctx.d_keys, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    GpuHT ht{};
    ht.mask = ctx.ht_max_sz - 1;
    ht.d_keys = ctx.d_ht_keys;
    ht.d_values = ctx.d_ht_values;
    ctx.d_ht_keys = nullptr;
    ctx.d_ht_values = nullptr;
    CUDA_CHECK(cudaMemsetAsync(ht.d_keys, 0xFF,
        ctx.ht_max_sz * sizeof(int32_t), stream));

    dpf_dim_read_str_pages(ctx.str[0], pfor_ctx, ctx.page_size, stream);
    dpf_dim_filter_str(ctx.str[0], ctx.nrows, ctx.page_size,
                        filter_mode, preds, enable_dict, dict,
                        nullptr, ctx.d_filter, ctx.d_values,
                        ctx.d_keys, ht.d_keys, ht.d_values, ht.mask,
                        stream);

    return ht;
}

// Run two-column dim HT build (IO + kernels only, no malloc)
static GpuHT dpf_dim_run_two(
    DpfDimCtx &ctx, bam_pfor32_io_ctx_t pfor_ctx,
    int32_t filter_mode, const std::vector<std::string> &preds,
    DpfGpuDict *dict,
    cudaStream_t stream)
{
    if (ctx.nrows == 0 || !ctx.d_keys) return GpuHT{};

    bam_pfor32_io_flatten_async(pfor_ctx, ctx.key_fp, ctx.d_keys, stream);
    s_kernel_launches++;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pass 1: filter on str[0]
    dpf_dim_read_str_pages(ctx.str[0], pfor_ctx, ctx.page_size, stream);
    dpf_dim_filter_str(ctx.str[0], ctx.nrows, ctx.page_size,
                        filter_mode, preds, false, nullptr,
                        nullptr, ctx.d_filter, nullptr,
                        nullptr, nullptr, nullptr, 0, stream);

    // Pass 2: dict on str[1] (with prefilter)
    dpf_dim_read_str_pages(ctx.str[1], pfor_ctx, ctx.page_size, stream);
    dpf_dim_filter_str(ctx.str[1], ctx.nrows, ctx.page_size,
                        DPF_FILT_NONE, {}, true, dict,
                        ctx.d_filter, nullptr, ctx.d_values,
                        nullptr, nullptr, nullptr, 0, stream);

    return dpf_dim_build_ht(ctx, stream);
}

// ────────────────────────────────────────────────────────────
// Legacy single-shot functions (alloc + IO + free in one call).
// NOT for use inside timed sections — use prepare/run pattern instead.
// ────────────────────────────────────────────────────────────
static GpuHT dpf_dim_single_col_ht(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    SSB::common::Table tbl, size_t key_col, size_t fsst_col,
    int32_t filter_mode, const std::vector<std::string> &preds,
    bool enable_dict, DpfGpuDict *dict,
    uint32_t num_blocks, cudaStream_t stream)
{
    uint64_t nrows = 0;
    int32_t *d_keys = dpf_flatten_dim_pfor32_gpu(
        pfor_ctx, sess, tbl, key_col, nrows, num_blocks, stream);
    if (!d_keys) return GpuHT{};

    auto dm = dpf_dim_meta(sess.metadata(), tbl, fsst_col);
    bool uncomp = (dm.comp_method == 0);

    uint64_t fsst_np = 0, fsst_nr = 0;
    char *d_pages = dpf_dim_read_fsst_to_gpu(pfor_ctx, sess, tbl, fsst_col,
                                              fsst_np, fsst_nr, stream);
    uint64_t *d_ps = dpf_dim_upload_prefix_sum(pfor_ctx, sess, tbl, fsst_col, fsst_np, stream);

    uint8_t *d_filter;
    CUDA_CHECK(cudaMalloc(&d_filter, nrows));
    int32_t *d_values = nullptr;
    if (enable_dict) CUDA_CHECK(cudaMalloc(&d_values, nrows * sizeof(int32_t)));

    if (uncomp) {
        uint32_t fsz = static_cast<uint32_t>(dpf_dim_field_size(tbl, fsst_col));
        dpf_dim_plain_char_run(d_pages, fsst_np, nrows, d_ps,
                               static_cast<uint32_t>(sess.page_size), fsz,
                               filter_mode, preds, enable_dict, dict,
                               nullptr, d_filter, d_values,
                               nullptr, nullptr, nullptr, 0, stream);
    } else {
        dpf_dim_fsst_run(d_pages, fsst_np, nrows, d_ps,
                          static_cast<uint32_t>(sess.page_size),
                          filter_mode, preds, enable_dict, dict,
                          nullptr, d_filter, d_values,
                          nullptr, nullptr, nullptr, 0, stream);
    }

    cudaFree(d_pages); cudaFree(d_ps);

    GpuHT ht = dpf_build_ht_gpu(d_keys, d_filter, d_values, nrows, stream);
    cudaFree(d_keys); cudaFree(d_filter);
    if (d_values) cudaFree(d_values);
    return ht;
}

// ────────────────────────────────────────────────────────────
// Two-column: filter on col A, dict on col B → HT
// Uncomp (comp_method==0): plain CHAR kernel
// Compressed: FSST kernel (unchanged)
// ────────────────────────────────────────────────────────────
static GpuHT dpf_dim_two_col_ht(
    bam_pfor32_io_ctx_t pfor_ctx, const BamSessionCtx &sess,
    SSB::common::Table tbl, size_t key_col,
    size_t filter_col, int32_t filter_mode, const std::vector<std::string> &preds,
    size_t dict_col, DpfGpuDict *dict,
    uint32_t num_blocks, cudaStream_t stream)
{
    uint64_t nrows = 0;
    int32_t *d_keys = dpf_flatten_dim_pfor32_gpu(
        pfor_ctx, sess, tbl, key_col, nrows, num_blocks, stream);
    if (!d_keys) return GpuHT{};

    auto dm_filt = dpf_dim_meta(sess.metadata(), tbl, filter_col);
    auto dm_dict = dpf_dim_meta(sess.metadata(), tbl, dict_col);
    bool filt_uncomp = (dm_filt.comp_method == 0);
    bool dict_uncomp = (dm_dict.comp_method == 0);

    // Pass 1: Filter on col A
    uint64_t filt_np = 0, filt_nr = 0;
    char *d_filt_pages = dpf_dim_read_fsst_to_gpu(pfor_ctx, sess, tbl, filter_col,
                                                    filt_np, filt_nr, stream);
    uint64_t *d_filt_ps = dpf_dim_upload_prefix_sum(pfor_ctx, sess, tbl, filter_col, filt_np, stream);

    uint8_t *d_filter;
    CUDA_CHECK(cudaMalloc(&d_filter, nrows));
    if (filt_uncomp) {
        uint32_t fsz = static_cast<uint32_t>(dpf_dim_field_size(tbl, filter_col));
        dpf_dim_plain_char_run(d_filt_pages, filt_np, nrows, d_filt_ps,
                               static_cast<uint32_t>(sess.page_size), fsz,
                               filter_mode, preds, false, nullptr,
                               nullptr, d_filter, nullptr,
                               nullptr, nullptr, nullptr, 0, stream);
    } else {
        dpf_dim_fsst_run(d_filt_pages, filt_np, nrows, d_filt_ps,
                          static_cast<uint32_t>(sess.page_size),
                          filter_mode, preds, false, nullptr,
                          nullptr, d_filter, nullptr,
                          nullptr, nullptr, nullptr, 0, stream);
    }
    cudaFree(d_filt_pages); cudaFree(d_filt_ps);

    // Pass 2: Dict on col B (with prefilter)
    uint64_t dict_np = 0, dict_nr = 0;
    char *d_dict_pages = dpf_dim_read_fsst_to_gpu(pfor_ctx, sess, tbl, dict_col,
                                                    dict_np, dict_nr, stream);
    uint64_t *d_dict_ps = dpf_dim_upload_prefix_sum(pfor_ctx, sess, tbl, dict_col, dict_np, stream);

    int32_t *d_values;
    CUDA_CHECK(cudaMalloc(&d_values, nrows * sizeof(int32_t)));
    if (dict_uncomp) {
        uint32_t fsz = static_cast<uint32_t>(dpf_dim_field_size(tbl, dict_col));
        dpf_dim_plain_char_run(d_dict_pages, dict_np, nrows, d_dict_ps,
                               static_cast<uint32_t>(sess.page_size), fsz,
                               DPF_FILT_NONE, {}, true, dict,
                               d_filter, nullptr, d_values,
                               nullptr, nullptr, nullptr, 0, stream);
    } else {
        dpf_dim_fsst_run(d_dict_pages, dict_np, nrows, d_dict_ps,
                          static_cast<uint32_t>(sess.page_size),
                          DPF_FILT_NONE, {}, true, dict,
                          d_filter, nullptr, d_values,
                          nullptr, nullptr, nullptr, 0, stream);
    }
    cudaFree(d_dict_pages); cudaFree(d_dict_ps);

    GpuHT ht = dpf_build_ht_gpu(d_keys, d_filter, d_values, nrows, stream);
    cudaFree(d_keys); cudaFree(d_filter); cudaFree(d_values);
    return ht;
}

// ============================================================
// SSB Q1.x DATAPATHFUSION
// Fields: LO_ORDERDATE, LO_QUANTITY, LO_DISCOUNT, LO_EXTENDEDPRICE (4)
// ============================================================
static BenchmarkResult ssb_q1x_dpf(BenchmarkOptions &options, SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    const size_t ps = sess.page_size;
    auto &m = sess.metadata();

    constexpr size_t NF = 4;
    const size_t lo_cols[NF] = {
        SSB::common::LO_ORDERDATE, SSB::common::LO_QUANTITY,
        SSB::common::LO_DISCOUNT, SSB::common::LO_EXTENDEDPRICE
    };
    uint64_t npages = m.table_lineorder_npages[lo_cols[0]];
    if (npages == 0) { bam_session_close(sess); return BenchmarkResult{}; }

    // Build DATE hash table
    GpuHT date_ht = alloc_gpu_ht(m.table_date_nrows);
    DimGpuBufs dim_bufs_q1 = dim_gpu_bufs_alloc(ps, m.table_date_nrows, sess.dim.max_pages);

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

    // GPU-side zonemap pruning via BamZonemapCtx
    auto pfor_ctx_zm = bam_pfor32_io_create(sess.ctrl, static_cast<uint32_t>(ps), kBamZonemapMaxReads);
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (options.enable_zonemap) {
        int32_t zm_lo, zm_hi;
        switch (query) {
            case SSB::Query::Q11: zm_lo = 19930101; zm_hi = 19931231; break;
            case SSB::Query::Q12: zm_lo = 19940101; zm_hi = 19940131; break;
            case SSB::Query::Q13: zm_lo = 19940201; zm_hi = 19940214; break;
            case SSB::Query::REVENUE:
                zm_lo = options.q6_sd_low ? options.q6_sd_low : 19920101;
                zm_hi = options.q6_sd_high ? options.q6_sd_high : 19981231; break;
            default: zm_lo = 19920101; zm_hi = 19981231; break;
        }
        uint64_t dk_nstats = m.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
        uint64_t dk_start = m.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
        uint64_t dk_npg = m.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];
        if (dk_nstats > 0 && dk_start > 0 && dk_npg > 0) {
            zm_ctx = bam_zonemap_ctx_create(
                bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
                bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
                (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
                static_cast<uint32_t>(ps), npages);
            uint32_t ri = 0;
            for (uint64_t j = 0; j < dk_npg; j++) {
                uint64_t pg_id = dk_start + j;
                uint32_t dev = pg_id % sess.n_devices;
                uint64_t local = pg_id / sess.n_devices;
                zm_ctx.h_reads[ri++] = {
                    sess.ds.partition_start_lbas[dev] + local * (ps / 512),
                    static_cast<uint32_t>(ps / 512), dev};
            }
            zm_ctx.h_preds[zm_npreds++] = {0, dk_nstats, zm_lo, zm_hi};
            zm_nreads = ri;
            zm_valid = true;
        }
    }
    // Warp-spec PFOR path (kernel handles pruning internally via zonemap_compact_block_pages)
    {
        uint32_t max_blocks = ssb_pfor_fused_q1x_max_blocks(static_cast<uint32_t>(ps));
        uint32_t num_blocks = max_blocks;

            size_t gpu_free_before_ws = 0, gpu_total_ws = 0;
            cudaMemGetInfo(&gpu_free_before_ws, &gpu_total_ws);

            // Compression metadata (warp-spec format, all pages)
            std::vector<size_t> all_pages(npages);
            std::iota(all_pages.begin(), all_pages.end(), size_t(0));
            auto cm = fusion_read_comp_metadata(sess, lo_cols, NF, npages, all_pages);

            // Create fused context (page_cache + decomp_buf)
            auto fctx = ssb_fused_create(sess.ctrl, static_cast<uint32_t>(ps),
                                          num_blocks, SSB_WS4_SLOTS_PER_BLOCK);
            void *d_ctrls = ssb_fused_get_d_ctrls(fctx);
            void *d_pc    = ssb_fused_get_d_pc_ptr(fctx);
            const char *pc_base = ssb_fused_get_pc_base(fctx);
            char *d_decomp = ssb_fused_get_decomp_buf(fctx);

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            int64_t *d_revenue = nullptr;
            CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(int64_t)));
            CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(int64_t)));

            // Fill warp-spec params
            SSBFusedQ1xParams p{};
            for (size_t fi = 0; fi < NF; fi++) {
                p.field_start_page_ids[fi] = m.table_lineorder_start_page_ids[lo_cols[fi]];
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
            p.d_page_mask = zm_valid ? zm_ctx.d_mask : nullptr;
            p.d_date_ht_keys = date_ht.d_keys;
            p.d_date_ht_values = date_ht.d_values;
            p.date_ht_mask = date_ht.mask;
            p.disc_lo = disc_lo; p.disc_hi = disc_hi;
            p.qty_lo = qty_lo; p.qty_hi = qty_hi;
            p.d_revenue = d_revenue;

            size_t gpu_free_after_ws = 0;
            cudaMemGetInfo(&gpu_free_after_ws, &gpu_total_ws);
            uint64_t gpu_app_bytes_ws = gpu_free_before_ws - gpu_free_after_ws;

            // Pre-issue IO to initialize BaM page_cache DMA registration
            if (zm_valid) {
                bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
            }

            // ═══════ Timed section (zonemap eval + date HT + fact kernel) ═══════
            s_kernel_launches = 0;
            auto total_start = std::chrono::steady_clock::now();

            // GPU zonemap eval (Rule 6: inside timed section)
            if (zm_valid) {
                bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                s_kernel_launches++;
                const uint32_t num_active = *zm_ctx.h_num_active;
                p.d_active_page_ids = zm_ctx.d_active_ids;
                p.total_pages = num_active;
                num_blocks = std::min(num_active, num_blocks);
            }

            uint64_t dim_io_bytes = 0;
            bam_build_date_ht_q1x_gpu(date_ht, sess, dim_bufs_q1, query, s_kernel_launches, dim_io_bytes);
            ssb_pfor_fused_q1x_launch(d_ctrls, d_pc, pc_base, d_decomp, p, num_blocks, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int64_t h_revenue = 0;
            CUDA_CHECK(cudaMemcpy(&h_revenue, d_revenue, sizeof(int64_t), cudaMemcpyDeviceToHost));
            std::cout << "SSB Q1.x revenue: " << h_revenue << std::endl;

            auto total_end = std::chrono::steady_clock::now();

            uint32_t active_count = zm_valid ? *zm_ctx.h_num_active : static_cast<uint32_t>(npages);
            std::cout << "[DPF-PFOR Q1x] num_blocks=" << num_blocks
                      << " (" << active_count << " active / " << npages << " total)"
                      << std::endl;

            if (zm_valid) fusion_recount_io(cm, NF, zm_ctx.h_mask);
            auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

            std::cout << "\n========================================\nTotal elapsed: " << elapsed_ns / 1e9 << " seconds"
                      << "\nTotal I/Os: " << cm.total_io_count << "\nTotal bytes read: " << (cm.total_io_bytes + dim_io_bytes)
                      << "\n========================================" << std::endl;

            // Cleanup
            if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
            bam_pfor32_io_destroy(pfor_ctx_zm);
            date_ht.free_all(); dim_gpu_bufs_free(dim_bufs_q1);
            cudaFree(d_revenue);
            ssb_fused_destroy(fctx);
            fusion_free_comp_meta(cm, NF);
            CUDA_CHECK(cudaStreamDestroy(stream));
            bam_session_close(sess);

            std::string comp_str = fusion_collect_comp_str(m, lo_cols, NF);
            return BenchmarkResult{
                .nios = cm.total_io_count, .read_bytes = cm.total_io_bytes + dim_io_bytes,
                .elapsed_nanoseconds = elapsed_ns, .compression = comp_str,
                .gpu_mem_bytes = sess.gpu_ctrl_bytes + gpu_app_bytes_ws,
                .gpu_ctrl_bytes = sess.gpu_ctrl_bytes, .gpu_app_bytes = gpu_app_bytes_ws,
                .total_pages = npages * NF,
                .kernel_launches = s_kernel_launches,
            };

    }
}

// ============================================================
// SSB Q2.x DATAPATHFUSION (fused kernel)
// Fields: LO_ORDERDATE, LO_PARTKEY, LO_SUPPKEY, LO_REVENUE (4)
// ============================================================
static BenchmarkResult ssb_q2x_dpf(BenchmarkOptions &options, SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    const size_t ps = sess.page_size;
    auto &m = sess.metadata();

    constexpr size_t NF = 4;
    const size_t lo_cols[NF] = {
        SSB::common::LO_ORDERDATE, SSB::common::LO_PARTKEY,
        SSB::common::LO_SUPPKEY, SSB::common::LO_REVENUE
    };
    uint64_t npages = m.table_lineorder_npages[lo_cols[0]];
    if (npages == 0) { bam_session_close(sess); return BenchmarkResult{}; }

    // Pre-allocate Date HT + dim bufs (Rule 2: alloc outside timing)
    GpuHT date_ht = alloc_gpu_ht(m.table_date_nrows);
    uint32_t date_max_np = 0;
    for (size_t i = 0; i < SSB::common::kDateFieldCount; i++)
        date_max_np = std::max(date_max_np, (uint32_t)m.table_date_npages[i]);
    if (date_max_np == 0) date_max_np = 1;
    DimGpuBufs dim_bufs_date = dim_gpu_bufs_alloc(ps, m.table_date_nrows, date_max_np);

    // GPU-side zonemap pruning via BamZonemapCtx
    auto pfor_ctx_zm = bam_pfor32_io_create(sess.ctrl, static_cast<uint32_t>(ps), kBamZonemapMaxReads);
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);
        const size_t ref_field = SSB::common::LO_ORDERDATE;
        const char *supp_region_zm;
        switch (query) {
            case SSB::Query::Q21: supp_region_zm = "AMERICA"; break;
            case SSB::Query::Q22: supp_region_zm = "ASIA"; break;
            case SSB::Query::Q23: supp_region_zm = "EUROPE"; break;
            default: supp_region_zm = "AMERICA"; break;
        }
        int32_t zm_sr_dict = -1, zm_p_lo = -1, zm_p_hi = -1;
        auto it_sr = dict_maps[SSB::common::LSS_S_REGION].find(std::string(supp_region_zm));
        if (it_sr != dict_maps[SSB::common::LSS_S_REGION].end()) zm_sr_dict = it_sr->second;
        uint64_t zm_sr_nstats = m.table_lineorder_sideways_nstats[ref_field][SSB::common::LSS_S_REGION];
        uint64_t zm_sr_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][SSB::common::LSS_S_REGION];
        uint64_t zm_sr_npg = m.table_lineorder_sideways_stats_npages[ref_field][SSB::common::LSS_S_REGION];
        size_t part_sw_idx = 0;
        if (query == SSB::Query::Q21) {
            part_sw_idx = SSB::common::LSS_P_CATEGORY;
            auto it = dict_maps[part_sw_idx].find("MFGR#12");
            if (it != dict_maps[part_sw_idx].end()) zm_p_lo = zm_p_hi = it->second;
        } else {
            part_sw_idx = SSB::common::LSS_P_BRAND1;
            auto it_lo = dict_maps[part_sw_idx].find("MFGR#2221");
            auto it_hi = dict_maps[part_sw_idx].find(query == SSB::Query::Q22 ? "MFGR#2228" : "MFGR#2221");
            if (it_lo != dict_maps[part_sw_idx].end()) zm_p_lo = it_lo->second;
            if (it_hi != dict_maps[part_sw_idx].end()) zm_p_hi = it_hi->second;
        }
        uint64_t zm_part_nstats = m.table_lineorder_sideways_nstats[ref_field][part_sw_idx];
        uint64_t zm_part_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][part_sw_idx];
        uint64_t zm_part_npg = m.table_lineorder_sideways_stats_npages[ref_field][part_sw_idx];
        bool has_sr = (zm_sr_dict >= 0 && zm_sr_nstats > 0 && zm_sr_start > 0 && zm_sr_npg > 0);
        bool has_part = (zm_p_lo >= 0 && zm_part_nstats > 0 && zm_part_start > 0 && zm_part_npg > 0);
        if (has_sr || has_part) {
            zm_ctx = bam_zonemap_ctx_create(
                bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
                bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
                (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
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
    // Warp-spec PFOR path (kernel handles pruning internally via zonemap_compact_block_pages)
    {
        uint32_t max_blocks = ssb_pfor_fused_q1x_max_blocks(static_cast<uint32_t>(ps));
        uint32_t num_blocks = max_blocks;

            size_t gpu_free_before_ws = 0, gpu_total_ws = 0;
            cudaMemGetInfo(&gpu_free_before_ws, &gpu_total_ws);

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            // ── Dim table prepare (alloc + metadata — outside timed section) ──
            int sm_count = 0;
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
            uint32_t dim_nblk = static_cast<uint32_t>(sm_count);
            auto pfor_ctx = sess.dim.io_ctx;

            const char *supp_region;
            switch (query) {
                case SSB::Query::Q21: supp_region = "AMERICA"; break;
                case SSB::Query::Q22: supp_region = "ASIA"; break;
                case SSB::Query::Q23: supp_region = "EUROPE"; break;
                default: supp_region = "AMERICA"; break;
            }

            // Shared dim bufs: large buffers reused across SUPPLIER + PART (all single-col)
            size_t q2_max_nrows = std::max(
                (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, SSB::common::S_SUPPKEY).nrows,
                (size_t)dpf_dim_meta(m, SSB::common::Table::PART, SSB::common::P_PARTKEY).nrows);
            size_t q2_max_pages = 0;
            for (size_t c : {SSB::common::S_SUPPKEY, SSB::common::S_REGION})
                q2_max_pages = std::max(q2_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, c).npages);
            for (size_t c : {SSB::common::P_PARTKEY, SSB::common::P_BRAND1})
                q2_max_pages = std::max(q2_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::PART, c).npages);
            DpfDimSharedBufs dim_shared = dpf_dim_shared_bufs_alloc(ps, q2_max_nrows, q2_max_pages, 1);

            std::cout << "[Q2x ws] preparing supp_ctx..." << std::flush;
            auto supp_ctx = dpf_dim_prepare_single(
                pfor_ctx, sess, SSB::common::Table::SUPPLIER,
                SSB::common::S_SUPPKEY, SSB::common::S_REGION,
                false, dim_nblk, stream, &dim_shared);
            std::cout << " done" << std::endl;

            DpfGpuDict brand_dict_gpu;
            brand_dict_gpu.alloc();
            DpfDictRaw brand_dr = dpf_dict_raw_alloc();
            std::cout << "[Q2x ws] preparing part_ctx..." << std::flush;
            DpfDimCtx part_ctx{};
            switch (query) {
            case SSB::Query::Q21:
                part_ctx = dpf_dim_prepare_single(
                    pfor_ctx, sess, SSB::common::Table::PART,
                    SSB::common::P_PARTKEY, SSB::common::P_BRAND1,
                    true, dim_nblk, stream, &dim_shared);
                break;
            case SSB::Query::Q22:
                part_ctx = dpf_dim_prepare_single(
                    pfor_ctx, sess, SSB::common::Table::PART,
                    SSB::common::P_PARTKEY, SSB::common::P_BRAND1,
                    true, dim_nblk, stream, &dim_shared);
                break;
            case SSB::Query::Q23:
                part_ctx = dpf_dim_prepare_single(
                    pfor_ctx, sess, SSB::common::Table::PART,
                    SSB::common::P_PARTKEY, SSB::common::P_BRAND1,
                    true, dim_nblk, stream, &dim_shared);
                break;
            default: break;
            }
            std::cout << " done" << std::endl;

            // Compression metadata (warp-spec format, all pages)
            std::vector<size_t> all_pages(npages);
            std::iota(all_pages.begin(), all_pages.end(), size_t(0));
            std::cout << "[Q2x ws] reading comp metadata..." << std::flush;
            auto cm = fusion_read_comp_metadata(sess, lo_cols, NF, npages, all_pages);
            std::cout << " done" << std::endl;

            std::cout << "[Q2x ws] creating fused ctx (num_blocks=" << num_blocks
                      << " slots_per_block=" << SSB_WS4_SLOTS_PER_BLOCK << ")..." << std::flush;
            auto fctx = ssb_fused_create(sess.ctrl, static_cast<uint32_t>(ps),
                                          num_blocks, SSB_WS4_SLOTS_PER_BLOCK);
            std::cout << " done" << std::endl;
            void *d_ctrls = ssb_fused_get_d_ctrls(fctx);
            void *d_pc    = ssb_fused_get_d_pc_ptr(fctx);
            const char *pc_base = ssb_fused_get_pc_base(fctx);
            char *d_decomp = ssb_fused_get_decomp_buf(fctx);

            int64_t *d_rev = nullptr;
            CUDA_CHECK(cudaMalloc(&d_rev, SSB_Q2X_GROUPS * sizeof(int64_t)));
            CUDA_CHECK(cudaMemset(d_rev, 0, SSB_Q2X_GROUPS * sizeof(int64_t)));

            size_t gpu_free_after_ws = 0;
            cudaMemGetInfo(&gpu_free_after_ws, &gpu_total_ws);
            uint64_t gpu_app_bytes_ws = gpu_free_before_ws - gpu_free_after_ws;

            // Pre-declare predicate vectors (Rule 4: avoid allocation inside timing)
            std::vector<std::string> pred_supp_region = {std::string(supp_region)};
            std::vector<std::string> pred_part;
            if (query == SSB::Query::Q21) pred_part = {"MFGR#12"};
            else if (query == SSB::Query::Q22) pred_part = {"MFGR#2221", "MFGR#2228"};
            else pred_part = {"MFGR#2221"};

            // Pre-issue IO to initialize BaM page_cache DMA registration
            if (zm_valid) {
                bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
            }

            // ═══════ Timed section (dim IO + kernel + fact table kernel) ═══════
            s_kernel_launches = 0;
            auto total_start = std::chrono::steady_clock::now();

            // GPU zonemap eval (Rule 6: inside timed section)
            if (zm_valid) {
                bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                s_kernel_launches++;
            }

            // Date HT: GPU build (Rule 1: I/O inside timing)
            uint64_t dim_io_bytes = 0;
            bam_build_date_ht_ext_gpu(date_ht, sess, dim_bufs_date, 0, 0, 0, s_kernel_launches, dim_io_bytes);

            // Dim table IO + filter + HT build
            GpuHT supp_ht = dpf_dim_run_single_ht(supp_ctx, pfor_ctx,
                DPF_FILT_EQ, pred_supp_region, false, nullptr, stream);
            dim_io_bytes += dpf_dim_ctx_io_bytes(supp_ctx);

            GpuHT part_ht{};
            switch (query) {
            case SSB::Query::Q21:
                part_ht = dpf_dim_run_single(part_ctx, pfor_ctx,
                    DPF_FILT_PREFIX, pred_part, true, &brand_dict_gpu, stream);
                break;
            case SSB::Query::Q22:
                part_ht = dpf_dim_run_single(part_ctx, pfor_ctx,
                    DPF_FILT_RANGE, pred_part, true, &brand_dict_gpu, stream);
                break;
            case SSB::Query::Q23:
                part_ht = dpf_dim_run_single(part_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_part, true, &brand_dict_gpu, stream);
                break;
            default: break;
            }
            dpf_download_dict_raw(brand_dict_gpu, brand_dr);
            dim_io_bytes += dpf_dim_ctx_io_bytes(part_ctx);

            // Fill warp-spec params
            SSBFusedQ2xParams p{};
            for (size_t fi = 0; fi < NF; fi++) {
                p.field_start_page_ids[fi] = m.table_lineorder_start_page_ids[lo_cols[fi]];
                p.d_comp_offsets[fi] = cm.d_comp_offsets[fi];
                p.d_comp_sizes[fi] = cm.d_comp_sizes[fi];
                p.is_compressed[fi] = cm.is_compressed[fi];
            }
            for (uint32_t d = 0; d < sess.n_devices && d < 4; d++)
                p.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
            p.n_devices = sess.n_devices;
            p.page_size = static_cast<uint32_t>(ps);
            p.total_pages = static_cast<uint32_t>(npages);
            p.d_active_page_ids = zm_valid ? zm_ctx.d_active_ids : nullptr;
            p.d_page_mask = zm_valid ? zm_ctx.d_mask : nullptr;
            if (zm_valid) {
                p.total_pages = *zm_ctx.h_num_active;
                num_blocks = std::min(*zm_ctx.h_num_active, num_blocks);
            }
            p.d_date_ht_keys = date_ht.d_keys; p.d_date_ht_values = date_ht.d_values; p.date_ht_mask = date_ht.mask;
            p.d_supp_ht_keys = supp_ht.d_keys; p.d_supp_ht_values = supp_ht.d_values; p.supp_ht_mask = supp_ht.mask;
            p.d_part_ht_keys = part_ht.d_keys; p.d_part_ht_values = part_ht.d_values; p.part_ht_mask = part_ht.mask;
            p.d_revenue = d_rev;

            ssb_pfor_fused_q2x_launch(d_ctrls, d_pc, pc_base, d_decomp, p, num_blocks, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int64_t h_rev[SSB_Q2X_GROUPS];
            CUDA_CHECK(cudaMemcpy(h_rev, d_rev, sizeof(h_rev), cudaMemcpyDeviceToHost));

            auto total_end = std::chrono::steady_clock::now();

            uint32_t active_count = zm_valid ? *zm_ctx.h_num_active : static_cast<uint32_t>(npages);
            std::cout << "[DPF-PFOR Q2x] num_blocks=" << num_blocks
                      << " (" << active_count << " active / " << npages << " total)"
                      << std::endl;

            // Result collection outside timing (Rule 4)
            auto brand_dict = dpf_build_dict_strings(brand_dr);
            std::cout << "\nSSB Q2.x results:" << std::endl;
            for (int32_t y = 0; y < SSB_NUM_YEARS; y++)
                for (size_t b = 0; b < brand_dict.size(); b++) {
                    int64_t v = h_rev[y * SSB_MAX_BRANDS + b];
                    if (v != 0) std::cout << "  " << v << " | " << (SSB_YEAR_MIN+y) << " | " << brand_dict[b] << std::endl;
                }
            if (zm_valid) fusion_recount_io(cm, NF, zm_ctx.h_mask);
            auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

            std::cout << "\n========================================\nTotal elapsed: " << elapsed_ns / 1e9 << " seconds"
                      << "\nTotal I/Os: " << cm.total_io_count << "\nTotal bytes read: " << (cm.total_io_bytes + dim_io_bytes)
                      << "\n========================================" << std::endl;

            // Cleanup (outside timed section)
            if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
            bam_pfor32_io_destroy(pfor_ctx_zm);
            dpf_dict_raw_free(brand_dr);
            supp_ht.free_all(); part_ht.free_all();
            supp_ctx.free_all(); part_ctx.free_all();
            dpf_dim_shared_bufs_free(dim_shared);
            brand_dict_gpu.free_all();

            ssb_fused_destroy(fctx);
            fusion_free_comp_meta(cm, NF);
            date_ht.free_all(); dim_gpu_bufs_free(dim_bufs_date); cudaFree(d_rev);
            CUDA_CHECK(cudaStreamDestroy(stream));
            bam_session_close(sess);

            std::string comp_str = fusion_collect_comp_str(m, lo_cols, NF);
            return BenchmarkResult{
                .nios = cm.total_io_count, .read_bytes = cm.total_io_bytes + dim_io_bytes,
                .elapsed_nanoseconds = elapsed_ns, .compression = comp_str,
                .gpu_mem_bytes = sess.gpu_ctrl_bytes + gpu_app_bytes_ws,
                .gpu_ctrl_bytes = sess.gpu_ctrl_bytes, .gpu_app_bytes = gpu_app_bytes_ws,
                .total_pages = npages * NF,
                .kernel_launches = s_kernel_launches,
            };
    }
}

// ============================================================
// SSB Q3.x DATAPATHFUSION (fused kernel)
// Fields: LO_ORDERDATE, LO_CUSTKEY, LO_SUPPKEY, LO_REVENUE (4)
// ============================================================
static BenchmarkResult ssb_q3x_dpf(BenchmarkOptions &options, SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    const size_t ps = sess.page_size;
    auto &m = sess.metadata();

    constexpr size_t NF = 4;
    const size_t lo_cols[NF] = {
        SSB::common::LO_ORDERDATE, SSB::common::LO_CUSTKEY,
        SSB::common::LO_SUPPKEY, SSB::common::LO_REVENUE
    };
    uint64_t npages = m.table_lineorder_npages[lo_cols[0]];
    if (npages == 0) { bam_session_close(sess); return BenchmarkResult{}; }

    // Pre-allocate Date HT + dim bufs (Rule 2: alloc outside timing)
    GpuHT date_ht = alloc_gpu_ht(m.table_date_nrows);
    uint32_t date_max_np = 0;
    for (size_t i = 0; i < SSB::common::kDateFieldCount; i++)
        date_max_np = std::max(date_max_np, (uint32_t)m.table_date_npages[i]);
    if (date_max_np == 0) date_max_np = 1;
    DimGpuBufs dim_bufs_date = dim_gpu_bufs_alloc(ps, m.table_date_nrows, date_max_np);

    // GPU-side zonemap pruning via BamZonemapCtx
    auto pfor_ctx_zm = bam_pfor32_io_create(sess.ctrl, static_cast<uint32_t>(ps), kBamZonemapMaxReads);
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);
        const size_t ref_field = SSB::common::LO_ORDERDATE;

        uint64_t zm_dk_nstats = 0, zm_dk_start = 0, zm_dk_npg = 0;
        if (query == SSB::Query::Q34) {
            zm_dk_nstats = m.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
            zm_dk_start = m.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
            zm_dk_npg = m.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];
        }

        int32_t zm_cust_lo = -1, zm_cust_hi = -1;
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
        uint64_t zm_cust_nstats = m.table_lineorder_sideways_nstats[ref_field][cust_sw_idx];
        uint64_t zm_cust_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][cust_sw_idx];
        uint64_t zm_cust_npg = m.table_lineorder_sideways_stats_npages[ref_field][cust_sw_idx];

        int32_t zm_supp_lo = -1, zm_supp_hi = -1;
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
        uint64_t zm_supp_nstats = m.table_lineorder_sideways_nstats[ref_field][supp_sw_idx];
        uint64_t zm_supp_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][supp_sw_idx];
        uint64_t zm_supp_npg = m.table_lineorder_sideways_stats_npages[ref_field][supp_sw_idx];

        bool has_dk = (query == SSB::Query::Q34 && zm_dk_nstats > 0 && zm_dk_start > 0 && zm_dk_npg > 0);
        bool has_cust = (zm_cust_lo >= 0 && zm_cust_nstats > 0 && zm_cust_start > 0 && zm_cust_npg > 0);
        bool has_supp = (zm_supp_lo >= 0 && zm_supp_nstats > 0 && zm_supp_start > 0 && zm_supp_npg > 0);
        if (has_dk || has_cust || has_supp) {
            zm_ctx = bam_zonemap_ctx_create(
                bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
                bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
                (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
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
    // Warp-spec PFOR path (kernel handles pruning internally via zonemap_compact_block_pages)
    {
        uint32_t max_blocks = ssb_pfor_fused_q1x_max_blocks(static_cast<uint32_t>(ps));
        uint32_t num_blocks = max_blocks;

            size_t gpu_free_before_ws = 0, gpu_total_ws = 0;
            cudaMemGetInfo(&gpu_free_before_ws, &gpu_total_ws);

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            // ── Dim table prepare (alloc — outside timed section) ──
            int sm_count = 0;
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
            uint32_t dim_nblk = static_cast<uint32_t>(sm_count);
            auto pfor_ctx = sess.dim.io_ctx;

            DpfGpuDict cust_dict_gpu, supp_dict_gpu;
            cust_dict_gpu.alloc(); supp_dict_gpu.alloc();
            DpfDictRaw cust_dr = dpf_dict_raw_alloc();
            DpfDictRaw supp_dr = dpf_dict_raw_alloc();
            DpfDimCtx cust_ctx{}, supp_ctx{};
            bool cust_is_two = false, supp_is_two = false;

            // Shared dim bufs: large buffers reused across CUSTOMER + SUPPLIER
            size_t q3_max_nrows = 0, q3_max_pages = 0;
            int q3_max_str = 1;
            switch (query) {
            case SSB::Query::Q31:
                for (auto c : {SSB::common::C_CUSTKEY, SSB::common::C_REGION, SSB::common::C_NATION})
                    q3_max_pages = std::max(q3_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::CUSTOMER, c).npages);
                for (auto c : {SSB::common::S_SUPPKEY, SSB::common::S_REGION, SSB::common::S_NATION})
                    q3_max_pages = std::max(q3_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, c).npages);
                q3_max_str = 2;
                break;
            case SSB::Query::Q32:
                for (auto c : {SSB::common::C_CUSTKEY, SSB::common::C_NATION, SSB::common::C_CITY})
                    q3_max_pages = std::max(q3_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::CUSTOMER, c).npages);
                for (auto c : {SSB::common::S_SUPPKEY, SSB::common::S_NATION, SSB::common::S_CITY})
                    q3_max_pages = std::max(q3_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, c).npages);
                q3_max_str = 2;
                break;
            default: // Q33, Q34
                for (auto c : {SSB::common::C_CUSTKEY, SSB::common::C_CITY})
                    q3_max_pages = std::max(q3_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::CUSTOMER, c).npages);
                for (auto c : {SSB::common::S_SUPPKEY, SSB::common::S_CITY})
                    q3_max_pages = std::max(q3_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, c).npages);
                break;
            }
            q3_max_nrows = std::max(
                (size_t)dpf_dim_meta(m, SSB::common::Table::CUSTOMER, SSB::common::C_CUSTKEY).nrows,
                (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, SSB::common::S_SUPPKEY).nrows);
            DpfDimSharedBufs dim_shared = dpf_dim_shared_bufs_alloc(ps, q3_max_nrows, q3_max_pages, q3_max_str);

            switch (query) {
            case SSB::Query::Q31:
                cust_ctx = dpf_dim_prepare_two(pfor_ctx, sess, SSB::common::Table::CUSTOMER,
                    SSB::common::C_CUSTKEY, SSB::common::C_REGION,
                    SSB::common::C_NATION, dim_nblk, stream, &dim_shared);
                supp_ctx = dpf_dim_prepare_two(pfor_ctx, sess, SSB::common::Table::SUPPLIER,
                    SSB::common::S_SUPPKEY, SSB::common::S_REGION,
                    SSB::common::S_NATION, dim_nblk, stream, &dim_shared);
                cust_is_two = true; supp_is_two = true;
                break;
            case SSB::Query::Q32:
                cust_ctx = dpf_dim_prepare_two(pfor_ctx, sess, SSB::common::Table::CUSTOMER,
                    SSB::common::C_CUSTKEY, SSB::common::C_NATION,
                    SSB::common::C_CITY, dim_nblk, stream, &dim_shared);
                supp_ctx = dpf_dim_prepare_two(pfor_ctx, sess, SSB::common::Table::SUPPLIER,
                    SSB::common::S_SUPPKEY, SSB::common::S_NATION,
                    SSB::common::S_CITY, dim_nblk, stream, &dim_shared);
                cust_is_two = true; supp_is_two = true;
                break;
            case SSB::Query::Q33:
            case SSB::Query::Q34:
                cust_ctx = dpf_dim_prepare_single(pfor_ctx, sess, SSB::common::Table::CUSTOMER,
                    SSB::common::C_CUSTKEY, SSB::common::C_CITY,
                    true, dim_nblk, stream, &dim_shared);
                supp_ctx = dpf_dim_prepare_single(pfor_ctx, sess, SSB::common::Table::SUPPLIER,
                    SSB::common::S_SUPPKEY, SSB::common::S_CITY,
                    true, dim_nblk, stream, &dim_shared);
                break;
            default: break;
            }

            // Compression metadata (warp-spec format, all pages)
            std::vector<size_t> all_pages(npages);
            std::iota(all_pages.begin(), all_pages.end(), size_t(0));
            auto cm = fusion_read_comp_metadata(sess, lo_cols, NF, npages, all_pages);

            auto fctx = ssb_fused_create(sess.ctrl, static_cast<uint32_t>(ps),
                                          num_blocks, SSB_WS4_SLOTS_PER_BLOCK);
            void *d_ctrls = ssb_fused_get_d_ctrls(fctx);
            void *d_pc    = ssb_fused_get_d_pc_ptr(fctx);
            const char *pc_base = ssb_fused_get_pc_base(fctx);
            char *d_decomp = ssb_fused_get_decomp_buf(fctx);

            int64_t *d_rev = nullptr;
            size_t grp_sz = SSB_Q3X_MAX_GROUPS * sizeof(int64_t);
            CUDA_CHECK(cudaMalloc(&d_rev, grp_sz));
            CUDA_CHECK(cudaMemset(d_rev, 0, grp_sz));

            size_t gpu_free_after_ws = 0;
            cudaMemGetInfo(&gpu_free_after_ws, &gpu_total_ws);
            uint64_t gpu_app_bytes_ws = gpu_free_before_ws - gpu_free_after_ws;

            // Pre-declare predicate vectors (Rule 4: avoid allocation inside timing)
            std::vector<std::string> pred_q3;
            if (query == SSB::Query::Q31) pred_q3 = {"ASIA"};
            else if (query == SSB::Query::Q32) pred_q3 = {"UNITED STATES"};
            else pred_q3 = {"UNITED KI1", "UNITED KI5"};

            // Pre-issue IO to initialize BaM page_cache DMA registration
            if (zm_valid) {
                bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
            }

            // ═══════ Timed section (zonemap eval + dim IO + fact kernel) ═══════
            s_kernel_launches = 0;
            auto total_start = std::chrono::steady_clock::now();

            // GPU zonemap eval (Rule 6: inside timed section)
            if (zm_valid) {
                bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                s_kernel_launches++;
            }

            // Date HT: GPU build (Rule 1: I/O inside timing)
            uint64_t dim_io_bytes = 0;
            if (query == SSB::Query::Q34)
                bam_build_date_ht_ext_gpu(date_ht, sess, dim_bufs_date, 2, 199712, 0, s_kernel_launches, dim_io_bytes);
            else
                bam_build_date_ht_ext_gpu(date_ht, sess, dim_bufs_date, 1, 1992, 1997, s_kernel_launches, dim_io_bytes);

            // Dim table IO + filter + HT build
            GpuHT cust_ht{}, supp_ht{};
            switch (query) {
            case SSB::Query::Q31:
                cust_ht = dpf_dim_run_two(cust_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_q3, &cust_dict_gpu, stream);
                supp_ht = dpf_dim_run_two(supp_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_q3, &supp_dict_gpu, stream);
                break;
            case SSB::Query::Q32:
                cust_ht = dpf_dim_run_two(cust_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_q3, &cust_dict_gpu, stream);
                supp_ht = dpf_dim_run_two(supp_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_q3, &supp_dict_gpu, stream);
                break;
            case SSB::Query::Q33:
            case SSB::Query::Q34:
                cust_ht = dpf_dim_run_single(cust_ctx, pfor_ctx,
                    DPF_FILT_IN, pred_q3, true, &cust_dict_gpu, stream);
                supp_ht = dpf_dim_run_single(supp_ctx, pfor_ctx,
                    DPF_FILT_IN, pred_q3, true, &supp_dict_gpu, stream);
                break;
            default: break;
            }
            dpf_download_dict_raw(cust_dict_gpu, cust_dr);
            dpf_download_dict_raw(supp_dict_gpu, supp_dr);
            dim_io_bytes += dpf_dim_ctx_io_bytes(cust_ctx);
            dim_io_bytes += dpf_dim_ctx_io_bytes(supp_ctx);
            int32_t ncd = std::max(1u, cust_dr.n);
            int32_t nsd = std::max(1u, supp_dr.n);

            // Fill warp-spec params
            SSBFusedQ3xParams p{};
            for (size_t fi = 0; fi < NF; fi++) {
                p.field_start_page_ids[fi] = m.table_lineorder_start_page_ids[lo_cols[fi]];
                p.d_comp_offsets[fi] = cm.d_comp_offsets[fi];
                p.d_comp_sizes[fi] = cm.d_comp_sizes[fi];
                p.is_compressed[fi] = cm.is_compressed[fi];
            }
            for (uint32_t d = 0; d < sess.n_devices && d < 4; d++)
                p.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
            p.n_devices = sess.n_devices;
            p.page_size = static_cast<uint32_t>(ps);
            p.total_pages = static_cast<uint32_t>(npages);
            p.d_active_page_ids = zm_valid ? zm_ctx.d_active_ids : nullptr;
            p.d_page_mask = zm_valid ? zm_ctx.d_mask : nullptr;
            if (zm_valid) {
                p.total_pages = *zm_ctx.h_num_active;
                num_blocks = std::min(*zm_ctx.h_num_active, num_blocks);
            }
            p.d_date_ht_keys = date_ht.d_keys; p.d_date_ht_values = date_ht.d_values; p.date_ht_mask = date_ht.mask;
            p.d_cust_ht_keys = cust_ht.d_keys; p.d_cust_ht_values = cust_ht.d_values; p.cust_ht_mask = cust_ht.mask;
            p.d_supp_ht_keys = supp_ht.d_keys; p.d_supp_ht_values = supp_ht.d_values; p.supp_ht_mask = supp_ht.mask;
            p.num_supp_dims = nsd;
            p.hist_size = (uint32_t)(ncd * nsd * SSB_Q3X_MAX_YEARS);
            p.d_revenue = d_rev;

            ssb_pfor_fused_q3x_launch(d_ctrls, d_pc, pc_base, d_decomp, p, num_blocks, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int64_t h_rev[SSB_Q3X_MAX_GROUPS];
            CUDA_CHECK(cudaMemcpy(h_rev, d_rev, grp_sz, cudaMemcpyDeviceToHost));

            auto total_end = std::chrono::steady_clock::now();

            uint32_t active_count = zm_valid ? *zm_ctx.h_num_active : static_cast<uint32_t>(npages);
            std::cout << "[DPF-PFOR Q3x] num_blocks=" << num_blocks
                      << " (" << active_count << " active / " << npages << " total)"
                      << std::endl;

            // Result collection outside timing (Rule 4)
            auto cust_dict = dpf_build_dict_strings(cust_dr);
            auto supp_dict = dpf_build_dict_strings(supp_dr);
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

            if (zm_valid) fusion_recount_io(cm, NF, zm_ctx.h_mask);
            auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

            std::cout << "\n========================================\nTotal elapsed: " << elapsed_ns / 1e9 << " seconds"
                      << "\nTotal I/Os: " << cm.total_io_count << "\nTotal bytes read: " << (cm.total_io_bytes + dim_io_bytes)
                      << "\n========================================" << std::endl;

            // Cleanup (outside timed section)
            if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
            bam_pfor32_io_destroy(pfor_ctx_zm);
            dpf_dict_raw_free(cust_dr); dpf_dict_raw_free(supp_dr);
            cust_ht.free_all(); supp_ht.free_all();
            cust_ctx.free_all(); supp_ctx.free_all();
            dpf_dim_shared_bufs_free(dim_shared);
            cust_dict_gpu.free_all(); supp_dict_gpu.free_all();

            ssb_fused_destroy(fctx);
            fusion_free_comp_meta(cm, NF);
            date_ht.free_all(); dim_gpu_bufs_free(dim_bufs_date); cudaFree(d_rev);
            CUDA_CHECK(cudaStreamDestroy(stream));
            bam_session_close(sess);

            std::string comp_str = fusion_collect_comp_str(m, lo_cols, NF);
            return BenchmarkResult{
                .nios = cm.total_io_count, .read_bytes = cm.total_io_bytes + dim_io_bytes,
                .elapsed_nanoseconds = elapsed_ns, .compression = comp_str,
                .gpu_mem_bytes = sess.gpu_ctrl_bytes + gpu_app_bytes_ws,
                .gpu_ctrl_bytes = sess.gpu_ctrl_bytes, .gpu_app_bytes = gpu_app_bytes_ws,
                .total_pages = npages * NF,
                .kernel_launches = s_kernel_launches,
            };
    }
}

// ============================================================
// SSB Q4.x DATAPATHFUSION (fused kernel)
// Fields: LO_ORDERDATE, LO_CUSTKEY, LO_PARTKEY, LO_SUPPKEY,
//         LO_REVENUE, LO_SUPPLYCOST (6)
// ============================================================
static BenchmarkResult ssb_q4x_dpf(BenchmarkOptions &options, SSB::Query query)
{
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto cuda_ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(cuda_ctx);
    auto sess = bam_session_open(options);
    const size_t ps = sess.page_size;
    auto &m = sess.metadata();

    constexpr size_t NF = 6;
    const size_t lo_cols[NF] = {
        SSB::common::LO_ORDERDATE, SSB::common::LO_CUSTKEY,
        SSB::common::LO_PARTKEY, SSB::common::LO_SUPPKEY,
        SSB::common::LO_REVENUE, SSB::common::LO_SUPPLYCOST
    };
    uint64_t npages = m.table_lineorder_npages[lo_cols[0]];
    if (npages == 0) { bam_session_close(sess); return BenchmarkResult{}; }

    // Pre-allocate Date HT + dim bufs (Rule 2: alloc outside timing)
    GpuHT date_ht = alloc_gpu_ht(m.table_date_nrows);
    uint32_t date_max_np = 0;
    for (size_t i = 0; i < SSB::common::kDateFieldCount; i++)
        date_max_np = std::max(date_max_np, (uint32_t)m.table_date_npages[i]);
    if (date_max_np == 0) date_max_np = 1;
    DimGpuBufs dim_bufs_date = dim_gpu_bufs_alloc(ps, m.table_date_nrows, date_max_np);

    // GPU-side zonemap pruning via BamZonemapCtx
    auto pfor_ctx_zm = bam_pfor32_io_create(sess.ctrl, static_cast<uint32_t>(ps), kBamZonemapMaxReads);
    BamZonemapCtx zm_ctx{};
    uint32_t zm_nreads = 0, zm_npreds = 0;
    bool zm_valid = false;
    if (options.enable_zonemap) {
        std::array<std::map<std::string, int32_t>, SSB::common::kSidewaysDictMapCount> dict_maps;
        SSB::common::ssb_build_sideways_dict_encoding_maps(dict_maps);
        const size_t ref_field = SSB::common::LO_ORDERDATE;

        uint64_t zm_dk_nstats = 0, zm_dk_start = 0, zm_dk_npg = 0;
        if (query != SSB::Query::Q41) {
            zm_dk_nstats = m.table_lineorder_nstats[SSB::common::LO_ORDERDATE];
            zm_dk_start = m.table_lineorder_stats_start_page_ids[SSB::common::LO_ORDERDATE];
            zm_dk_npg = m.table_lineorder_stats_npages[SSB::common::LO_ORDERDATE];
        }

        int32_t zm_cr_id = -1;
        {
            auto it = dict_maps[SSB::common::LSS_C_REGION].find("AMERICA");
            if (it != dict_maps[SSB::common::LSS_C_REGION].end()) zm_cr_id = it->second;
        }
        uint64_t zm_cr_nstats = m.table_lineorder_sideways_nstats[ref_field][SSB::common::LSS_C_REGION];
        uint64_t zm_cr_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][SSB::common::LSS_C_REGION];
        uint64_t zm_cr_npg = m.table_lineorder_sideways_stats_npages[ref_field][SSB::common::LSS_C_REGION];

        int32_t zm_supp_lo = -1, zm_supp_hi = -1;
        size_t supp_sw = 0;
        if (query == SSB::Query::Q41 || query == SSB::Query::Q42) {
            supp_sw = SSB::common::LSS_S_REGION;
            auto it = dict_maps[supp_sw].find("AMERICA");
            if (it != dict_maps[supp_sw].end()) zm_supp_lo = zm_supp_hi = it->second;
        } else {
            supp_sw = SSB::common::LSS_S_NATION;
            auto it = dict_maps[supp_sw].find("UNITED STATES");
            if (it != dict_maps[supp_sw].end()) zm_supp_lo = zm_supp_hi = it->second;
        }
        uint64_t zm_supp_nstats = m.table_lineorder_sideways_nstats[ref_field][supp_sw];
        uint64_t zm_supp_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][supp_sw];
        uint64_t zm_supp_npg = m.table_lineorder_sideways_stats_npages[ref_field][supp_sw];

        int32_t zm_part_lo = -1, zm_part_hi = -1;
        size_t part_sw = 0;
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
        uint64_t zm_part_nstats = m.table_lineorder_sideways_nstats[ref_field][part_sw];
        uint64_t zm_part_start = m.table_lineorder_sideways_stats_start_page_ids[ref_field][part_sw];
        uint64_t zm_part_npg = m.table_lineorder_sideways_stats_npages[ref_field][part_sw];

        bool has_dk = (query != SSB::Query::Q41 && zm_dk_nstats > 0 && zm_dk_start > 0 && zm_dk_npg > 0);
        bool has_cr = (zm_cr_id >= 0 && zm_cr_nstats > 0 && zm_cr_start > 0 && zm_cr_npg > 0);
        bool has_supp = (zm_supp_lo >= 0 && zm_supp_nstats > 0 && zm_supp_start > 0 && zm_supp_npg > 0);
        bool has_part = (zm_part_lo >= 0 && zm_part_nstats > 0 && zm_part_start > 0 && zm_part_npg > 0);
        if (has_dk || has_cr || has_supp || has_part) {
            zm_ctx = bam_zonemap_ctx_create(
                bam_pfor32_io_get_d_ctrls(pfor_ctx_zm),
                bam_pfor32_io_get_d_pc_ptr(pfor_ctx_zm),
                (void*)bam_pfor32_io_get_pc_base(pfor_ctx_zm),
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
    // Warp-spec PFOR path (kernel handles pruning internally via zonemap_compact_block_pages)
    {
        uint32_t max_blocks = ssb_pfor_fused_q4x_max_blocks(static_cast<uint32_t>(ps));
        uint32_t num_blocks = max_blocks;

            size_t gpu_free_before_ws = 0, gpu_total_ws = 0;
            cudaMemGetInfo(&gpu_free_before_ws, &gpu_total_ws);

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            // ── Dim table prepare (alloc — outside timed section) ──
            int sm_count = 0;
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
            uint32_t dim_nblk = static_cast<uint32_t>(sm_count);
            auto pfor_ctx = sess.dim.io_ctx;

            DpfGpuDict cust_dict_gpu, supp_dict_gpu, part_dict_gpu;
            cust_dict_gpu.alloc(); supp_dict_gpu.alloc(); part_dict_gpu.alloc();
            DpfDictRaw cust_dr = dpf_dict_raw_alloc();
            DpfDictRaw supp_dr = dpf_dict_raw_alloc();
            DpfDictRaw part_dr = dpf_dict_raw_alloc();
            DpfDimCtx cust_ctx{}, supp_ctx{}, part_ctx{};

            // Shared dim bufs: large buffers reused across CUSTOMER + SUPPLIER + PART
            size_t q4_max_nrows = std::max({
                (size_t)dpf_dim_meta(m, SSB::common::Table::CUSTOMER, SSB::common::C_CUSTKEY).nrows,
                (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, SSB::common::S_SUPPKEY).nrows,
                (size_t)dpf_dim_meta(m, SSB::common::Table::PART, SSB::common::P_PARTKEY).nrows});
            size_t q4_max_pages = 0;
            int q4_max_str = 1;
            switch (query) {
            case SSB::Query::Q41:
                for (auto c : {SSB::common::C_CUSTKEY, SSB::common::C_REGION, SSB::common::C_NATION})
                    q4_max_pages = std::max(q4_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::CUSTOMER, c).npages);
                for (auto c : {SSB::common::S_SUPPKEY, SSB::common::S_REGION})
                    q4_max_pages = std::max(q4_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, c).npages);
                for (auto c : {SSB::common::P_PARTKEY, SSB::common::P_MFGR})
                    q4_max_pages = std::max(q4_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::PART, c).npages);
                q4_max_str = 2; // CUSTOMER uses two-col
                break;
            case SSB::Query::Q42:
                for (auto c : {SSB::common::C_CUSTKEY, SSB::common::C_REGION})
                    q4_max_pages = std::max(q4_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::CUSTOMER, c).npages);
                for (auto c : {SSB::common::S_SUPPKEY, SSB::common::S_REGION, SSB::common::S_NATION})
                    q4_max_pages = std::max(q4_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, c).npages);
                for (auto c : {SSB::common::P_PARTKEY, SSB::common::P_MFGR, SSB::common::P_CATEGORY})
                    q4_max_pages = std::max(q4_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::PART, c).npages);
                q4_max_str = 2;
                break;
            case SSB::Query::Q43:
                for (auto c : {SSB::common::C_CUSTKEY, SSB::common::C_REGION})
                    q4_max_pages = std::max(q4_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::CUSTOMER, c).npages);
                for (auto c : {SSB::common::S_SUPPKEY, SSB::common::S_NATION, SSB::common::S_CITY})
                    q4_max_pages = std::max(q4_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::SUPPLIER, c).npages);
                for (auto c : {SSB::common::P_PARTKEY, SSB::common::P_CATEGORY, SSB::common::P_BRAND1})
                    q4_max_pages = std::max(q4_max_pages, (size_t)dpf_dim_meta(m, SSB::common::Table::PART, c).npages);
                q4_max_str = 2;
                break;
            default: break;
            }
            DpfDimSharedBufs dim_shared = dpf_dim_shared_bufs_alloc(ps, q4_max_nrows, q4_max_pages, q4_max_str);

            switch (query) {
            case SSB::Query::Q41:
                cust_ctx = dpf_dim_prepare_two(pfor_ctx, sess, SSB::common::Table::CUSTOMER,
                    SSB::common::C_CUSTKEY, SSB::common::C_REGION,
                    SSB::common::C_NATION, dim_nblk, stream, &dim_shared);
                supp_ctx = dpf_dim_prepare_single(pfor_ctx, sess, SSB::common::Table::SUPPLIER,
                    SSB::common::S_SUPPKEY, SSB::common::S_REGION,
                    false, dim_nblk, stream, &dim_shared);
                part_ctx = dpf_dim_prepare_single(pfor_ctx, sess, SSB::common::Table::PART,
                    SSB::common::P_PARTKEY, SSB::common::P_MFGR,
                    false, dim_nblk, stream, &dim_shared);
                break;
            case SSB::Query::Q42:
                cust_ctx = dpf_dim_prepare_single(pfor_ctx, sess, SSB::common::Table::CUSTOMER,
                    SSB::common::C_CUSTKEY, SSB::common::C_REGION,
                    false, dim_nblk, stream, &dim_shared);
                supp_ctx = dpf_dim_prepare_two(pfor_ctx, sess, SSB::common::Table::SUPPLIER,
                    SSB::common::S_SUPPKEY, SSB::common::S_REGION,
                    SSB::common::S_NATION, dim_nblk, stream, &dim_shared);
                part_ctx = dpf_dim_prepare_two(pfor_ctx, sess, SSB::common::Table::PART,
                    SSB::common::P_PARTKEY, SSB::common::P_MFGR,
                    SSB::common::P_CATEGORY, dim_nblk, stream, &dim_shared);
                break;
            case SSB::Query::Q43:
                cust_ctx = dpf_dim_prepare_single(pfor_ctx, sess, SSB::common::Table::CUSTOMER,
                    SSB::common::C_CUSTKEY, SSB::common::C_REGION,
                    false, dim_nblk, stream, &dim_shared);
                supp_ctx = dpf_dim_prepare_two(pfor_ctx, sess, SSB::common::Table::SUPPLIER,
                    SSB::common::S_SUPPKEY, SSB::common::S_NATION,
                    SSB::common::S_CITY, dim_nblk, stream, &dim_shared);
                part_ctx = dpf_dim_prepare_two(pfor_ctx, sess, SSB::common::Table::PART,
                    SSB::common::P_PARTKEY, SSB::common::P_CATEGORY,
                    SSB::common::P_BRAND1, dim_nblk, stream, &dim_shared);
                break;
            default: break;
            }

            // Compression metadata (warp-spec format, all pages)
            std::vector<size_t> all_pages(npages);
            std::iota(all_pages.begin(), all_pages.end(), size_t(0));
            auto cm = fusion_read_comp_metadata(sess, lo_cols, NF, npages, all_pages);

            auto fctx = ssb_fused_create(sess.ctrl, static_cast<uint32_t>(ps),
                                          num_blocks, SSB_WS6_SLOTS_PER_BLOCK);
            void *d_ctrls = ssb_fused_get_d_ctrls(fctx);
            void *d_pc    = ssb_fused_get_d_pc_ptr(fctx);
            const char *pc_base = ssb_fused_get_pc_base(fctx);
            char *d_decomp = ssb_fused_get_decomp_buf(fctx);

            int64_t *d_profit = nullptr;
            size_t grp_sz = SSB_Q4X_MAX_GROUPS * sizeof(int64_t);
            CUDA_CHECK(cudaMalloc(&d_profit, grp_sz));
            CUDA_CHECK(cudaMemset(d_profit, 0, grp_sz));

            size_t gpu_free_after_ws = 0;
            cudaMemGetInfo(&gpu_free_after_ws, &gpu_total_ws);
            uint64_t gpu_app_bytes_ws = gpu_free_before_ws - gpu_free_after_ws;

            // Pre-declare predicate vectors (Rule 4: avoid allocation inside timing)
            std::vector<std::string> pred_america = {"AMERICA"};
            std::vector<std::string> pred_supp;
            std::vector<std::string> pred_part;
            if (query == SSB::Query::Q43) {
                pred_supp = {"UNITED STATES"};
                pred_part = {"MFGR#14"};
            } else {
                pred_supp = {"AMERICA"};
                pred_part = {"MFGR#1", "MFGR#2"};
            }

            // Pre-issue IO to initialize BaM page_cache DMA registration
            if (zm_valid) {
                bam_pre_io(zm_ctx.d_ctrls, zm_ctx.d_pc, stream);
            }

            // ═══════ Timed section (dim IO + kernel + fact kernel) ═══════
            s_kernel_launches = 0;
            auto total_start = std::chrono::steady_clock::now();

            // GPU zonemap eval (Rule 6: inside timed section)
            if (zm_valid) {
                bam_zonemap_eval_async(zm_ctx, npages, zm_nreads, zm_npreds, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                s_kernel_launches++;
            }

            // Date HT: GPU build (Rule 1: I/O inside timing)
            uint64_t dim_io_bytes = 0;
            if (query == SSB::Query::Q41)
                bam_build_date_ht_ext_gpu(date_ht, sess, dim_bufs_date, 0, 0, 0, s_kernel_launches, dim_io_bytes);
            else
                bam_build_date_ht_ext_gpu(date_ht, sess, dim_bufs_date, 1, 1997, 1998, s_kernel_launches, dim_io_bytes);

            // Dim table IO + filter + HT build
            GpuHT cust_ht{}, supp_ht{}, part_ht{};
            switch (query) {
            case SSB::Query::Q41:
                cust_ht = dpf_dim_run_two(cust_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_america, &cust_dict_gpu, stream);
                supp_ht = dpf_dim_run_single(supp_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_supp, false, nullptr, stream);
                part_ht = dpf_dim_run_single(part_ctx, pfor_ctx,
                    DPF_FILT_IN, pred_part, false, nullptr, stream);
                break;
            case SSB::Query::Q42:
                cust_ht = dpf_dim_run_single(cust_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_america, false, nullptr, stream);
                supp_ht = dpf_dim_run_two(supp_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_supp, &supp_dict_gpu, stream);
                part_ht = dpf_dim_run_two(part_ctx, pfor_ctx,
                    DPF_FILT_IN, pred_part, &part_dict_gpu, stream);
                break;
            case SSB::Query::Q43:
                cust_ht = dpf_dim_run_single(cust_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_america, false, nullptr, stream);
                supp_ht = dpf_dim_run_two(supp_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_supp, &supp_dict_gpu, stream);
                part_ht = dpf_dim_run_two(part_ctx, pfor_ctx,
                    DPF_FILT_EQ, pred_part, &part_dict_gpu, stream);
                break;
            default: break;
            }
            dpf_download_dict_raw(cust_dict_gpu, cust_dr);
            dpf_download_dict_raw(supp_dict_gpu, supp_dr);
            dpf_download_dict_raw(part_dict_gpu, part_dr);
            dim_io_bytes += dpf_dim_ctx_io_bytes(cust_ctx);
            dim_io_bytes += dpf_dim_ctx_io_bytes(supp_ctx);
            dim_io_bytes += dpf_dim_ctx_io_bytes(part_ctx);
            int32_t ncd = std::max(1u, cust_dr.n), nsd = std::max(1u, supp_dr.n), npd = std::max(1u, part_dr.n);
            int32_t stride_year = ncd * nsd * npd;

            // Fill warp-spec params
            SSBFusedQ4xParams p{};
            for (size_t fi = 0; fi < NF; fi++) {
                p.field_start_page_ids[fi] = m.table_lineorder_start_page_ids[lo_cols[fi]];
                p.d_comp_offsets[fi] = cm.d_comp_offsets[fi];
                p.d_comp_sizes[fi] = cm.d_comp_sizes[fi];
                p.is_compressed[fi] = cm.is_compressed[fi];
            }
            for (uint32_t d = 0; d < sess.n_devices && d < 4; d++)
                p.partition_start_lbas[d] = sess.ds.partition_start_lbas[d];
            p.n_devices = sess.n_devices;
            p.page_size = static_cast<uint32_t>(ps);
            p.total_pages = static_cast<uint32_t>(npages);
            p.d_active_page_ids = zm_valid ? zm_ctx.d_active_ids : nullptr;
            p.d_page_mask = zm_valid ? zm_ctx.d_mask : nullptr;
            if (zm_valid) {
                p.total_pages = *zm_ctx.h_num_active;
                num_blocks = std::min(*zm_ctx.h_num_active, num_blocks);
            }
            p.d_date_ht_keys = date_ht.d_keys; p.d_date_ht_values = date_ht.d_values; p.date_ht_mask = date_ht.mask;
            p.d_cust_ht_keys = cust_ht.d_keys; p.d_cust_ht_values = cust_ht.d_values; p.cust_ht_mask = cust_ht.mask;
            p.d_supp_ht_keys = supp_ht.d_keys; p.d_supp_ht_values = supp_ht.d_values; p.supp_ht_mask = supp_ht.mask;
            p.d_part_ht_keys = part_ht.d_keys; p.d_part_ht_values = part_ht.d_values; p.part_ht_mask = part_ht.mask;
            p.supp_dims = nsd;
            p.part_dims = npd;
            p.stride_year = stride_year;
            p.hist_size = (uint32_t)(SSB_Q4X_MAX_YEARS * stride_year);
            p.d_profit = d_profit;

            ssb_pfor_fused_q4x_launch(d_ctrls, d_pc, pc_base, d_decomp, p, num_blocks, stream);
            s_kernel_launches++;
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int64_t h_prof[SSB_Q4X_MAX_GROUPS];
            CUDA_CHECK(cudaMemcpy(h_prof, d_profit, grp_sz, cudaMemcpyDeviceToHost));

            auto total_end = std::chrono::steady_clock::now();

            uint32_t active_count = zm_valid ? *zm_ctx.h_num_active : static_cast<uint32_t>(npages);
            std::cout << "[DPF-PFOR Q4x] num_blocks=" << num_blocks
                      << " (" << active_count << " active / " << npages << " total)"
                      << std::endl;

            // Result collection outside timing (Rule 4)
            auto cust_dict = dpf_build_dict_strings(cust_dr);
            auto supp_dict = dpf_build_dict_strings(supp_dr);
            auto part_dict = dpf_build_dict_strings(part_dr);
            struct Q4xRow { int32_t y; std::string c, s, p; int64_t v; };
            std::vector<Q4xRow> rows;
            for (int32_t y = 0; y < SSB_Q4X_MAX_YEARS; y++)
                for (int32_t c = 0; c < ncd; c++)
                    for (int32_t si = 0; si < nsd; si++)
                        for (int32_t pi = 0; pi < npd; pi++) {
                            int64_t v = h_prof[y * stride_year + c * (nsd * npd) + si * npd + pi];
                            if (v != 0) rows.push_back({SSB_YEAR_MIN + y, cust_dict[c], supp_dict[si], part_dict[pi], v});
                        }
            std::sort(rows.begin(), rows.end(), [](const Q4xRow &a, const Q4xRow &b) {
                if (a.y != b.y) return a.y < b.y;
                if (a.c != b.c) return a.c < b.c;
                if (a.s != b.s) return a.s < b.s;
                return a.p < b.p; });
            std::cout << "\nSSB Q4.x results:" << std::endl;
            for (auto &r : rows) {
                std::cout << "  " << r.y;
                if (r.c != "_") std::cout << " | " << r.c;
                if (r.s != "_") std::cout << " | " << r.s;
                if (r.p != "_") std::cout << " | " << r.p;
                std::cout << " | " << r.v << std::endl;
            }

            if (zm_valid) fusion_recount_io(cm, NF, zm_ctx.h_mask);
            auto elapsed_ns = std::chrono::duration<int64_t, std::nano>(total_end - total_start).count();

            std::cout << "\n========================================\nTotal elapsed: " << elapsed_ns / 1e9 << " seconds"
                      << "\nTotal I/Os: " << cm.total_io_count << "\nTotal bytes read: " << (cm.total_io_bytes + dim_io_bytes)
                      << "\n========================================" << std::endl;

            // Cleanup (outside timed section)
            if (zm_valid) bam_zonemap_ctx_destroy(zm_ctx);
            bam_pfor32_io_destroy(pfor_ctx_zm);
            dpf_dict_raw_free(cust_dr); dpf_dict_raw_free(supp_dr); dpf_dict_raw_free(part_dr);
            cust_ht.free_all(); supp_ht.free_all(); part_ht.free_all();
            cust_ctx.free_all(); supp_ctx.free_all(); part_ctx.free_all();
            dpf_dim_shared_bufs_free(dim_shared);
            cust_dict_gpu.free_all(); supp_dict_gpu.free_all(); part_dict_gpu.free_all();

            ssb_fused_destroy(fctx);
            fusion_free_comp_meta(cm, NF);
            date_ht.free_all(); dim_gpu_bufs_free(dim_bufs_date); cudaFree(d_profit);
            CUDA_CHECK(cudaStreamDestroy(stream));
            bam_session_close(sess);

            std::string comp_str = fusion_collect_comp_str(m, lo_cols, NF);
            return BenchmarkResult{
                .nios = cm.total_io_count, .read_bytes = cm.total_io_bytes + dim_io_bytes,
                .elapsed_nanoseconds = elapsed_ns, .compression = comp_str,
                .gpu_mem_bytes = sess.gpu_ctrl_bytes + gpu_app_bytes_ws,
                .gpu_ctrl_bytes = sess.gpu_ctrl_bytes, .gpu_app_bytes = gpu_app_bytes_ws,
                .total_pages = npages * NF,
                .kernel_launches = s_kernel_launches,
            };
    }
}

} // namespace SsbDatapathFusion

// ============================================================
// Public wrapper functions
// ============================================================
BenchmarkResult ssb_q11_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q1x_dpf(options, SSB::Query::Q11);
}
BenchmarkResult ssb_q12_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q1x_dpf(options, SSB::Query::Q12);
}
BenchmarkResult ssb_q13_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q1x_dpf(options, SSB::Query::Q13);
}
BenchmarkResult ssb_q21_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q2x_dpf(options, SSB::Query::Q21);
}
BenchmarkResult ssb_q22_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q2x_dpf(options, SSB::Query::Q22);
}
BenchmarkResult ssb_q23_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q2x_dpf(options, SSB::Query::Q23);
}
BenchmarkResult ssb_q31_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q3x_dpf(options, SSB::Query::Q31);
}
BenchmarkResult ssb_q32_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q3x_dpf(options, SSB::Query::Q32);
}
BenchmarkResult ssb_q33_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q3x_dpf(options, SSB::Query::Q33);
}
BenchmarkResult ssb_q34_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q3x_dpf(options, SSB::Query::Q34);
}
BenchmarkResult ssb_q41_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q4x_dpf(options, SSB::Query::Q41);
}
BenchmarkResult ssb_q42_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q4x_dpf(options, SSB::Query::Q42);
}
BenchmarkResult ssb_q43_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q4x_dpf(options, SSB::Query::Q43);
}
BenchmarkResult ssb_revenue_datapathfusion(BenchmarkOptions &options) {
    return SsbDatapathFusion::ssb_q1x_dpf(options, SSB::Query::REVENUE);
}
