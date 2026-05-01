#pragma once

#include <cstdint>

#define MAX_BAM_DEVICES 4

// Metadata passed from the C++20 host code to the C++11 BAM kernel wrapper.
// Kept POD so it can cross the ABI boundary without BAM headers.
struct BAMQueryParams {
    uint64_t field_start_page_ids[4];
    uint64_t field_npages;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint32_t num_warps;
    uint32_t num_queues;
    uint64_t queue_depth;
    uint32_t nvm_ns_id;
    int      cuda_device;
    uint64_t partition_start_lba; // partition offset in 512-byte sectors (device 0)
    uint64_t partition_start_lbas[MAX_BAM_DEVICES]; // per-device partition LBA (RAID0)
    uint32_t n_devices;          // 1 = single device (default), >1 = RAID0 striping
    uint64_t nrows;              // total number of rows

    // Per-field prefix sum arrays (host pointers, npages+1 elements each).
    // prefix_sum[fi][p] = cumulative rows before page p for field fi.
    // Caller must ensure these are valid until bam_q6_run returns.
    const uint64_t* h_prefix_sums[4];

    // Zone map IO pruning: per-page min/max stats for L_SHIPDATE.
    // Array of (min_val, max_val) pairs as int32_t[nstats * 2].
    // nullptr means no pruning.
    const int32_t* h_shipdate_stats;
    uint64_t nstats;

    // Compression support.
    // Per-field compression method: 0 = uncompressed, 1 = PFOR.
    uint16_t compression_method[4];

    // Per-field compressed page sizes (host pointers, field_npages elements each).
    // compressed_page_sizes[fi][pg] = actual byte count of compressed page pg.
    // nullptr if field is uncompressed.
    const uint32_t* h_compressed_page_sizes[4];

    // Per-field compressed file offsets (host pointers, field_npages elements each).
    // compressed_offsets[fi][pg] = byte offset on disk for page pg.
    // LBA = partition_start_lba + compressed_offsets[fi][pg] / 512
    // nullptr if field is uncompressed.
    const uint64_t* h_compressed_offsets[4];

    // Number of CUDA blocks for kernel launch.
    uint32_t num_blocks;

    // Decompression buffer: max decompressed elements per page slot.
    // Must be >= max(watermark) across all pages of all Q6 fields.
    // Computed from prefix sums: ceil(max_rows_per_page / 128) * 128.
    uint32_t decomp_elems_per_slot;

    // I/O multiplicity: ring buffer depth for pipelined I/O.
    // 1 = synchronous (no pipelining), >1 = overlap I/O with compute.
    uint32_t io_multiplicity;

    // Q6 L_SHIPDATE bounds for selectivity experiments.
    // sd_low: inclusive lower bound (default 19940101)
    // sd_high: exclusive upper bound (default 19950101)
    int32_t sd_low;
    int32_t sd_high;

    // I/O coalescing: number of consecutive driving pages per NVMe command.
    // 1 = per-page I/O (default), >1 = coalesced reads.
    uint32_t coalesce_k;

    // Prescan mode: pre-compute qualifying page list + I/O descriptors
    // in a separate kernel, then execute with a streamlined kernel.
    bool use_prescan;

    // Thread block size: 32 (default, 1 warp) or 128 (4 warps, parallel field decode).
    uint32_t block_size;

    // Revenue query: L_QUANTITY < revenue_qt_max.
    // 0 = no quantity filter (default), >0 = apply filter.
    int32_t revenue_qt_max;
};

// Result returned from BAM kernel runs (carries I/O stats back to caller).
struct BAMRunResult {
    int64_t  revenue;
    uint64_t io_count;   // total NVMe read calls
    uint64_t io_bytes;   // total bytes read from NVMe
};

// Opaque handle to a BAM Controller (hides BAM headers from C++20 code).
typedef void* bam_ctrl_handle_t;

// Open / close a BAM NVMe controller.
bam_ctrl_handle_t bam_ctrl_open(const char* path, uint32_t ns_id,
                                 int cuda_device, uint32_t queue_depth,
                                 uint32_t num_queues);

// Open multiple BAM NVMe controllers (RAID0 striping).
bam_ctrl_handle_t bam_ctrl_open_multi(const char** paths, uint32_t n_devices,
                                       uint32_t ns_id, int cuda_device,
                                       uint32_t queue_depth, uint32_t num_queues);

void bam_ctrl_close(bam_ctrl_handle_t h);

// Read `page_size` bytes starting at `lba` (512-byte sectors) into `out_buf`.
// Uses the given controller; creates a temporary page cache internally.
// dev_idx selects which controller to use (0 = default).
int bam_read_page(bam_ctrl_handle_t ctrl, uint64_t page_size,
                  uint64_t lba, void* out_buf, uint32_t dev_idx = 0);

// Run the BAM Q6 kernel using an existing controller.
BAMRunResult bam_q6_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl);

// Run the coalesced BAM Q6 kernel: each NVMe command reads k consecutive
// pages per field (params.coalesce_k > 1). Requires identical prefix sums.
BAMRunResult bam_q6_coalesced_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl);

// Run the prescan BAM Q6 kernel: pre-computes qualifying page list + I/O descriptors,
// then executes with a streamlined kernel that skips zone-map checks and binary searches.
BAMRunResult bam_q6_prescan_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl);

// Run the 128-thread BAM Q6 kernel: 4 warps per block, parallel field decode.
// Warp 0 handles I/O; all 4 warps decode 4 Q6 fields in parallel.
BAMRunResult bam_q6_128t_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl);

// Run the 128-thread prefetch kernel (Pattern 2): 4 warps, double-buffered page cache.
// After IO, warp 0 submits prefetch for next page; polled on next iteration.
BAMRunResult bam_q6_128t_pf_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl);

// Run the work queue kernel: IO blocks + compute blocks with lock-free ring buffer.
// IO blocks scan pages and submit NVMe reads; compute blocks dequeue and process.
// Dispatch: -B 128 -R -Z -i <n_io_blocks>
BAMRunResult bam_q6_wq_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl);

// Run the 160-thread IO-exclusive kernel (Pattern 1): 5 warps, warp 0 = IO only.
// During decode (warps 1-4), warp 0 computes and submits prefetch in parallel.
BAMRunResult bam_q6_160t_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl);

// Run the BAM Revenue kernel (Q6 scan plan, shipdate-only predicate).
// Uses same 4-field infrastructure as Q6 but only applies shipdate filter.
BAMRunResult bam_revenue_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl);

// Run the 128-thread BAM Revenue kernel: 4 warps per block, parallel field decode.
// Same IO/decomp structure as bam_q6_128t but with revenue (shipdate-only) predicate.
BAMRunResult bam_revenue_128t_run(const BAMQueryParams& params, bam_ctrl_handle_t ctrl);

// Test PFOR decompression on GPU: upload a compressed page buffer,
// decompress via LoadBinPackTile (128 threads), download result.
// page_buf: host buffer containing the raw compressed page data.
// page_buf_size: number of bytes in page_buf.
// decomp_out: host buffer to receive decompressed int32 values (nalloc_aligned elements).
// Returns nalloc (actual valid element count).
uint32_t bam_test_decompress_page(const void* page_buf, size_t page_buf_size,
                                   int32_t* decomp_out, uint32_t max_elems);

// Test PFOR64 decompression on GPU (64-bit variant for L_ORDERKEY etc.).
// Returns nalloc (actual valid element count).
uint32_t bam_test_decompress_page64(const void* page_buf, size_t page_buf_size,
                                     int64_t* decomp_out, uint32_t max_elems);

// ============================================================
// PFOR64 batch flatten: decompress PFOR64-compressed pages into
// a contiguous int64_t flat array, using prefix_sum for row offsets.
// Handles both compressed (comp_method != 0) and uncompressed pages.
// ============================================================
void bam_flatten_pfor64_pages(
    const char* d_pages,
    const uint64_t* d_prefix_sum,
    int64_t* d_flat_output,
    uint32_t page_size,
    uint32_t npages,
    uint16_t comp_method,
    cudaStream_t stream);

// ============================================================
// PFOR64 GPU-initiated I/O + flatten: reads PFOR64-compressed
// pages via BAM and decompresses directly into a flat int64_t
// output array.  Managed as an opaque handle.
// ============================================================

// Parameters for PFOR64 GPU-initiated I/O flatten kernel.
// POD struct shared between C++20 host code and C++11 BAM kernel wrapper.
// All GPU pointers must be pre-allocated by the caller before total_start.
struct BAMPfor64FlattenParams {
    uint64_t partition_start_lba;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES]; // per-device partition LBA (RAID0)
    uint32_t n_devices;             // 1 = single device (default), >1 = RAID0 striping
    uint32_t page_size;
    uint32_t blocks_per_page;       // page_size / 512
    uint16_t comp_method;           // 0=uncompressed, 2=PFOR64
    uint64_t field_start_page_id;   // uncompressed: LBA = partition_start_lba + (field_start_page_id + pg) * blocks_per_page
    uint64_t npages;
    uint64_t nrows;                 // total number of rows (output array size)
    uint32_t num_blocks;            // kernel grid size

    // GPU pointers (caller must cudaMalloc before total_start,
    // cudaMemcpy H→D after total_start).
    const uint64_t* d_prefix_sum;   // [npages] cumulative row counts
    const uint32_t* d_comp_sizes;   // [npages] or nullptr (uncompressed)
    const uint64_t* d_comp_offsets; // [npages] or nullptr (uncompressed)
};

typedef void* bam_pfor64_io_ctx_t;

// Create a PFOR64 I/O context (page cache allocation).
// Call before total_start.
bam_pfor64_io_ctx_t bam_pfor64_io_create(
    bam_ctrl_handle_t ctrl,
    uint32_t page_size,
    uint32_t num_blocks);

// Run the PFOR64 I/O + flatten kernel.
// Call after total_start (no alloc/free inside).
void bam_pfor64_io_flatten(
    bam_pfor64_io_ctx_t ctx,
    const BAMPfor64FlattenParams& params,
    int64_t* d_flat_output);

// Async variant: launches kernel on the given stream without synchronizing.
// Caller must cudaStreamSynchronize(stream) before using d_flat_output.
void bam_pfor64_io_flatten_async(
    bam_pfor64_io_ctx_t ctx,
    const BAMPfor64FlattenParams& params,
    int64_t* d_flat_output,
    cudaStream_t stream);

// Destroy the PFOR64 I/O context (free page cache).
// Call after total_end.
void bam_pfor64_io_destroy(bam_pfor64_io_ctx_t ctx);

// ============================================================
// PFOR64 dual-field flatten: reads two INT64 fields from the
// same table in a single kernel launch.  Each block processes
// both fields sequentially per page, reusing the page cache slot.
// Both fields must have the same number of pages.
// ============================================================
struct BAMPfor64DualFlattenParams {
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint32_t num_blocks;
    uint64_t npages;                    // same for both fields

    uint64_t field0_start_page_id;
    uint16_t field0_comp_method;
    const uint32_t* field0_d_comp_sizes;
    const uint64_t* field0_d_comp_offsets;
    const uint64_t* field0_d_prefix_sum;
    int64_t* field0_d_output;

    uint64_t field1_start_page_id;
    uint16_t field1_comp_method;
    const uint32_t* field1_d_comp_sizes;
    const uint64_t* field1_d_comp_offsets;
    const uint64_t* field1_d_prefix_sum;
    int64_t* field1_d_output;
};

void bam_pfor64_dual_flatten_async(
    bam_pfor64_io_ctx_t ctx,
    const BAMPfor64DualFlattenParams& params,
    cudaStream_t stream);

// ============================================================
// PFOR32 GPU-initiated I/O + flatten: reads PFOR-compressed
// INT32 pages via BAM and decompresses directly into a flat
// int32_t output array.  Mirrors PFOR64 variant.
// ============================================================

// Parameters for PFOR32 GPU-initiated I/O flatten kernel.
// POD struct shared between C++20 host code and C++11 BAM kernel wrapper.
// All GPU pointers must be pre-allocated by the caller before total_start.
struct BAMPfor32FlattenParams {
    uint64_t partition_start_lba;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES]; // per-device partition LBA (RAID0)
    uint32_t n_devices;             // 1 = single device (default), >1 = RAID0 striping
    uint32_t page_size;
    uint32_t blocks_per_page;       // page_size / 512
    uint16_t comp_method;           // 0=uncompressed, 1=PFOR
    uint64_t field_start_page_id;   // uncompressed: LBA = partition_start_lba + (field_start_page_id + pg) * blocks_per_page
    uint64_t npages;
    uint64_t nrows;                 // total number of rows (output array size)
    uint32_t num_blocks;            // kernel grid size

    // GPU pointers (caller must cudaMalloc before total_start,
    // cudaMemcpy H→D after total_start).
    const uint64_t* d_prefix_sum;   // [npages] cumulative row counts
    const uint32_t* d_comp_sizes;   // [npages] or nullptr (uncompressed)
    const uint64_t* d_comp_offsets; // [npages] or nullptr (uncompressed)
};

typedef void* bam_pfor32_io_ctx_t;

// Create a PFOR32 I/O context (page cache allocation).
// Call before total_start.
bam_pfor32_io_ctx_t bam_pfor32_io_create(
    bam_ctrl_handle_t ctrl,
    uint32_t page_size,
    uint32_t num_blocks);

// Run the PFOR32 I/O + flatten kernel.
// Call after total_start (no alloc/free inside).
void bam_pfor32_io_flatten(
    bam_pfor32_io_ctx_t ctx,
    const BAMPfor32FlattenParams& params,
    int32_t* d_flat_output);

// Async variant: launches kernel on the given stream without synchronizing.
// Caller must cudaStreamSynchronize(stream) before using d_flat_output.
void bam_pfor32_io_flatten_async(
    bam_pfor32_io_ctx_t ctx,
    const BAMPfor32FlattenParams& params,
    int32_t* d_flat_output,
    cudaStream_t stream);

// Async PFOR32 flatten + widen: IO + decompress + zero-extend INT32 → uint64_t.
// Decompresses into d_temp_i32, then widens to d_flat_output (separate buffers).
void bam_pfor32_io_flatten_widen_async(
    bam_pfor32_io_ctx_t ctx,
    const BAMPfor32FlattenParams& params,
    int32_t* d_temp_i32,
    uint64_t* d_flat_output,
    cudaStream_t stream);

// Async PFOR32 IO + decompress NK + nationkey filter + HT build (Q5 dim tables).
// Decompresses NK pages into d_temp_i32, filters by nationkey, reads KEY from
// d_key_flat, and inserts into hash table — single kernel.
void bam_pfor32_io_nk_ht_build_async(
    bam_pfor32_io_ctx_t ctx,
    const BAMPfor32FlattenParams& params,
    int32_t* d_temp_i32,
    const uint64_t* d_key_flat,
    const int8_t* d_nationkey_to_idx,
    uint64_t* ht_keys,
    int32_t* ht_values,
    uint32_t ht_mask,
    cudaStream_t stream);

// Destroy the PFOR32 I/O context (free page cache).
// Call after total_end.
void bam_pfor32_io_destroy(bam_pfor32_io_ctx_t ctx);

// Accessors: expose page_cache internals for fused dim IO+decomp kernel.
void* bam_pfor32_io_get_d_ctrls(bam_pfor32_io_ctx_t ctx);
void* bam_pfor32_io_get_d_pc_ptr(bam_pfor32_io_ctx_t ctx);
const char* bam_pfor32_io_get_pc_base(bam_pfor32_io_ctx_t ctx);
uint32_t bam_pfor32_io_get_num_slots(bam_pfor32_io_ctx_t ctx);

// ============================================================
// Masked PFOR64/PFOR32 flatten: zone-map IO pruning variant.
//
// Identical to the standard flatten except it accepts a per-page
// active mask (d_page_active).  Inactive pages (mask==0) skip the
// NVMe read and their output range is filled with fill_value.
// If d_page_active==nullptr, all pages are treated as active.
// ============================================================

void bam_pfor64_io_flatten_masked_async(
    bam_pfor64_io_ctx_t ctx,
    const BAMPfor64FlattenParams& params,
    const uint8_t* d_page_active,
    int64_t fill_value,
    int64_t* d_flat_output,
    cudaStream_t stream);

void bam_pfor32_io_flatten_masked_async(
    bam_pfor32_io_ctx_t ctx,
    const BAMPfor32FlattenParams& params,
    const uint8_t* d_page_active,
    int32_t fill_value,
    int32_t* d_flat_output,
    cudaStream_t stream);

// ============================================================
// Fused revenue kernel: batch-reads 4 fields per page,
// decompresses to per-block scratch, evaluates revenue in one pass.
// Page cache must have num_blocks * 4 entries.
// ============================================================
struct BAMRevenueFusedParams {
    uint64_t partition_start_lba;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;
    uint32_t num_blocks;
    int32_t sd_low, sd_high, qt_max;
    int32_t dc_low, dc_high;           // discount bounds (0,0 = no filter)
    // Per-field metadata (device pointers)
    uint64_t field_start_page_ids[4];
    uint16_t comp_methods[4];
    uint64_t* d_prefix_sum;        // shared across fields (page-level cumulative row count)
    uint32_t* d_comp_sizes[4];
    uint64_t* d_comp_offsets[4];
    // Zone map
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    // Per-block scratch buffer (num_blocks * 4 * scratch_stride int32_t)
    int32_t* d_scratch;
    uint32_t scratch_stride;       // max rows per page
};

void bam_revenue_fused_run(
    bam_pfor32_io_ctx_t ctx,
    const BAMRevenueFusedParams& params,
    int64_t* d_revenue,
    cudaStream_t stream);

// Synchronous single-block test: reads all pages one-by-one, decompresses, evaluates Q6.
// Used for diagnosing DMA reliability issues.
void bam_revenue_sync_test_run(
    bam_pfor32_io_ctx_t ctx,
    const BAMRevenueFusedParams& params,
    int64_t* d_revenue,
    cudaStream_t stream);

// ============================================================
// Q5 ORDERS fused kernel: reads O_ORDERDATE INT32 page +
// O_ORDERKEY/O_CUSTKEY INT64 pages via BAM, decompresses to
// per-block scratch, evaluates date filter + CUSTOMER HT probe,
// inserts into ORDERS HT.
// Page cache: num_blocks * 5 entries (1 INT32 + 2 INT64 × 2 pages).
// ============================================================
struct BAMQ5OrdersFusedParams {
    uint64_t partition_start_lba;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;             // O_ORDERDATE INT32 page count
    uint32_t num_blocks;
    int32_t date_low, date_high;
    // O_ORDERDATE field metadata (1 INT32 field)
    uint64_t field_start_page_id;
    uint16_t comp_method;
    uint64_t* d_prefix_sum;      // INT32 page cumulative row counts
    uint32_t* d_comp_sizes;
    uint64_t* d_comp_offsets;
    // Zone map
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    // INT64 fields via BaM: [0]=O_ORDERKEY, [1]=O_CUSTKEY
    uint64_t field_start_page_ids_i64[2];
    uint16_t comp_methods_i64[2];
    uint64_t* d_prefix_sum_i64;  // INT64 page cumulative row counts
    uint64_t  npages_i64;
    uint32_t* d_comp_sizes_i64[2];
    uint64_t* d_comp_offsets_i64[2];
    // CUSTOMER HT (input)
    const uint64_t* d_ht_cust_keys;
    const int32_t*  d_ht_cust_values;
    uint32_t ht_cust_mask;
    // ORDERS HT (output)
    uint64_t* d_ht_ord_keys;
    int32_t*  d_ht_ord_values;
    uint32_t ht_ord_mask;
    // Per-block scratch
    int32_t* d_scratch;          // num_blocks * 1 * scratch_stride int32_t
    int64_t* d_scratch_i64;      // num_blocks * 2 * 2 * scratch_stride_i64 int64_t
    uint32_t scratch_stride;     // max rows per INT32 page
    uint32_t scratch_stride_i64; // max rows per INT64 page
    uint64_t* d_dbg_counters;    // [0]=total_rows [1]=date_pass [2]=cust_hit [3]=inserted
};

void bam_q5_orders_fused_run(
    bam_pfor32_io_ctx_t ctx,
    const BAMQ5OrdersFusedParams& params,
    cudaStream_t stream);

// ============================================================
// Q5 LINEITEM fused kernel: reads L_EXTENDEDPRICE + L_DISCOUNT
// INT32 pages + L_ORDERKEY/L_SUPPKEY INT64 pages via BAM,
// decompresses to per-block scratch, probes ORDERS + SUPPLIER
// HTs, accumulates revenue per nation.
// Page cache: num_blocks * 6 entries (2 INT32 + 2 INT64 × 2 pages).
// ============================================================
struct BAMQ5LineitemFusedParams {
    uint64_t partition_start_lba;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;             // L_EXTENDEDPRICE INT32 page count
    uint32_t num_blocks;
    // 2 INT32 fields: [0]=L_EXTENDEDPRICE, [1]=L_DISCOUNT
    uint64_t field_start_page_ids[2];
    uint16_t comp_methods[2];
    uint64_t* d_prefix_sum;      // shared prefix sum (INT32)
    uint32_t* d_comp_sizes[2];
    uint64_t* d_comp_offsets[2];
    // Zone map
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    // INT64 fields via BaM: [0]=L_ORDERKEY, [1]=L_SUPPKEY
    uint64_t field_start_page_ids_i64[2];
    uint16_t comp_methods_i64[2];
    uint64_t* d_prefix_sum_i64;  // INT64 page cumulative row counts
    uint64_t  npages_i64;
    uint32_t* d_comp_sizes_i64[2];
    uint64_t* d_comp_offsets_i64[2];
    // ORDERS HT
    const uint64_t* d_ht_ord_keys;
    const int32_t*  d_ht_ord_values;
    uint32_t ht_ord_mask;
    // SUPPLIER HT
    const uint64_t* d_ht_supp_keys;
    const int32_t*  d_ht_supp_values;
    uint32_t ht_supp_mask;
    // Per-block scratch
    int32_t* d_scratch;          // num_blocks * 2 * scratch_stride int32_t
    int64_t* d_scratch_i64;      // num_blocks * 2 * 2 * scratch_stride_i64 int64_t
    uint32_t scratch_stride;     // max rows per INT32 page
    uint32_t scratch_stride_i64; // max rows per INT64 page
    uint64_t* d_dbg_counters;    // [0]=total_rows [1]=ord_hit [2]=supp_hit [3]=nation_match
};

void bam_q5_lineitem_fused_run(
    bam_pfor32_io_ctx_t ctx,
    const BAMQ5LineitemFusedParams& params,
    int64_t* d_revenue,
    cudaStream_t stream);

// ============================================================
// Q3 ORDERS fused kernel: reads O_ORDERDATE + O_SHIPPRIORITY
// INT32 pages + O_ORDERKEY/O_CUSTKEY INT64 pages via BAM,
// decompresses to per-block scratch, evaluates date filter +
// CUSTOMER hash set probe, inserts into ORDERS HT.
// Page cache: num_blocks * 6 entries (2 INT32 + 2 INT64 × 2 pages).
// ============================================================
struct BAMQ3OrdersFusedParams {
    uint64_t partition_start_lba;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;             // O_ORDERDATE INT32 page count
    uint32_t num_blocks;
    // 2 INT32 fields: [0]=O_ORDERDATE, [1]=O_SHIPPRIORITY
    uint64_t field_start_page_ids[2];
    uint16_t comp_methods[2];
    uint64_t* d_prefix_sum;      // shared prefix sum (INT32 pages)
    uint32_t* d_comp_sizes[2];
    uint64_t* d_comp_offsets[2];
    // Zone map
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    // INT64 fields via BaM: [0]=O_ORDERKEY, [1]=O_CUSTKEY
    uint64_t field_start_page_ids_i64[2];
    uint16_t comp_methods_i64[2];
    uint64_t* d_prefix_sum_i64;  // INT64 page cumulative row counts
    uint64_t  npages_i64;
    uint32_t* d_comp_sizes_i64[2];
    uint64_t* d_comp_offsets_i64[2];
    // CUSTOMER hash set (probe only)
    const uint64_t* d_custkey_set;
    uint32_t custkey_set_mask;
    // ORDERS HT (output, key→payload)
    uint64_t* d_orders_ht_keys;
    uint64_t* d_orders_ht_payloads;
    uint32_t orders_ht_mask;
    // Per-block scratch
    int32_t* d_scratch;          // num_blocks * 2 * scratch_stride int32_t
    int64_t* d_scratch_i64;      // num_blocks * 2 * 2 * scratch_stride_i64 int64_t
    uint32_t scratch_stride;
    uint32_t scratch_stride_i64;
    // Q3SEL: skip date filter when true
    bool skip_date_filter;
};

void bam_q3_orders_fused_run(
    bam_pfor32_io_ctx_t ctx,
    const BAMQ3OrdersFusedParams& params,
    cudaStream_t stream);

// ============================================================
// Q3 LINEITEM fused kernel: reads L_SHIPDATE + L_EXTENDEDPRICE
// + L_DISCOUNT INT32 pages + L_ORDERKEY INT64 pages via BAM,
// decompresses to per-block scratch, evaluates shipdate filter
// + ORDERS HT probe, aggregates revenue into GROUP BY hash map.
// Page cache: num_blocks * 5 entries (3 INT32 + 1 INT64 × 2 pages).
// ============================================================
struct BAMQ3LineitemFusedParams {
    uint64_t partition_start_lba;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;             // INT32 page count (shared)
    uint32_t num_blocks;
    // 3 INT32 fields: [0]=L_SHIPDATE, [1]=L_EXTPRICE, [2]=L_DISCOUNT
    uint64_t field_start_page_ids[3];
    uint16_t comp_methods[3];
    uint64_t* d_prefix_sum;      // shared prefix sum (INT32 pages)
    uint32_t* d_comp_sizes[3];
    uint64_t* d_comp_offsets[3];
    // Zone map
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    // 1 INT64 field via BaM: [0]=L_ORDERKEY
    uint64_t field_start_page_ids_i64[1];
    uint16_t comp_methods_i64[1];
    uint64_t* d_prefix_sum_i64;  // INT64 page cumulative row counts
    uint64_t  npages_i64;
    uint32_t* d_comp_sizes_i64[1];
    uint64_t* d_comp_offsets_i64[1];
    // ORDERS HT (probe)
    const uint64_t* d_orders_ht_keys;
    const uint64_t* d_orders_ht_payloads;
    uint32_t orders_ht_mask;
    // Aggregation hash map (GROUP BY l_orderkey)
    uint64_t* d_aggr_keys;
    int64_t*  d_aggr_revenues;
    uint32_t aggr_mask;
    // Per-block scratch
    int32_t* d_scratch;          // num_blocks * 3 * scratch_stride int32_t
    int64_t* d_scratch_i64;      // num_blocks * 1 * 2 * scratch_stride_i64 int64_t
    uint32_t scratch_stride;
    uint32_t scratch_stride_i64;
    // Q3SEL: skip shipdate filter when true
    bool skip_shipdate_filter;
    // Debug counters
    uint64_t* d_dbg_counters;
};

void bam_q3_lineitem_fused_run(
    bam_pfor32_io_ctx_t ctx,
    const BAMQ3LineitemFusedParams& params,
    cudaStream_t stream);

// ============================================================
// Q1 fused kernel: reads 5 LINEITEM INT32 pages via BAM
// (L_SHIPDATE + L_QUANTITY + L_EXTENDEDPRICE + L_DISCOUNT
// + L_TAX), decompresses to per-block scratch, evaluates
// shipdate filter, looks up returnflag/linestatus from
// pre-flattened CHAR(1) arrays, and aggregates.
// Page cache: num_blocks * 5 entries.
// ============================================================
struct BAMQ1FusedParams {
    uint64_t partition_start_lba;
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;             // INT32 page count (shared across all 7 cols)
    uint32_t num_blocks;
    // 5 INT32 fields: [0]=L_SHIPDATE, [1]=L_QUANTITY, [2]=L_EXTENDEDPRICE,
    //                 [3]=L_DISCOUNT, [4]=L_TAX
    uint64_t field_start_page_ids[5];
    uint16_t comp_methods[5];
    uint64_t* d_prefix_sum;      // shared prefix sum
    uint32_t* d_comp_sizes[5];
    uint64_t* d_comp_offsets[5];
    // Zone map
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    // Pre-flattened CHAR(1) arrays (global rowid indexed)
    const uint64_t* d_l_rf_flat;
    const uint64_t* d_l_ls_flat;
    // Aggregation output
    int64_t* d_agg;              // [Q1_NUM_GROUPS * Q1_NUM_AGGS]
    // Per-block scratch (num_blocks * 5 * scratch_stride int32_t)
    int32_t* d_scratch;
    uint32_t scratch_stride;
};

void bam_q1_fused_run(
    bam_pfor32_io_ctx_t ctx,
    const BAMQ1FusedParams& params,
    cudaStream_t stream);

// ============================================================
// VCHAR I/O context: reads compressed VCHAR pages via BAM
// into a GPU staging buffer.  Managed as an opaque handle.
// ============================================================
typedef void* bam_vchar_io_ctx_t;

// Create a VCHAR I/O context (page cache + staging buffer).
bam_vchar_io_ctx_t bam_vchar_io_create(
    bam_ctrl_handle_t ctrl,
    uint32_t page_size,
    uint32_t num_blocks);

// Get the staging buffer pointer (device memory, num_blocks * page_size bytes).
char* bam_vchar_io_staging_buf(bam_vchar_io_ctx_t ctx);

// Read a batch of compressed VCHAR pages into the staging buffer.
// batch_start: index of first page in this batch.
// batch_size: number of pages to read (must be <= num_blocks).
// Blocks until all I/O completes.
void bam_vchar_io_read_batch(
    bam_vchar_io_ctx_t ctx,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    uint64_t partition_start_lba,
    uint64_t batch_start,
    uint32_t batch_size,
    const uint64_t* partition_start_lbas = nullptr,
    uint32_t n_devices = 1,
    uint64_t field_start_page_id = 0);

// Destroy the VCHAR I/O context (frees page cache + staging buffer).
void bam_vchar_io_destroy(bam_vchar_io_ctx_t ctx);

// ============================================================
// VCHAR I/O context v2: 128 threads (4 warps), each warp reads 1 page.
// Page cache has num_blocks * 4 entries.
// ============================================================

// Create a VCHAR I/O v2 context (page cache + staging buffer).
bam_vchar_io_ctx_t bam_vchar_io_v2_create(
    bam_ctrl_handle_t ctrl,
    uint32_t page_size,
    uint32_t num_blocks);

// Get the staging buffer pointer (device memory, num_blocks * 4 * page_size bytes).
char* bam_vchar_io_v2_staging_buf(bam_vchar_io_ctx_t ctx);

// Read a batch of compressed VCHAR pages into the staging buffer (v2).
// batch_start: index of first page in this batch.
// batch_blocks: number of thread blocks to launch (each handles 4 pages).
// npages_total: total page count (for bounds checking at tail).
void bam_vchar_io_v2_read_batch(
    bam_vchar_io_ctx_t ctx,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    uint64_t partition_start_lba,
    uint64_t batch_start,
    uint32_t batch_blocks,
    uint64_t npages_total,
    const uint64_t* partition_start_lbas = nullptr,
    uint32_t n_devices = 1,
    uint64_t field_start_page_id = 0);

// Async read batch (launches kernel, does NOT synchronize).
void bam_vchar_io_v2_read_batch_async(
    bam_vchar_io_ctx_t ctx,
    const uint32_t* d_comp_sizes,
    const uint64_t* d_comp_offsets,
    uint64_t partition_start_lba,
    uint64_t batch_start,
    uint32_t batch_blocks,
    uint64_t npages_total,
    const uint64_t* partition_start_lbas = nullptr,
    uint32_t n_devices = 1,
    uint64_t field_start_page_id = 0);

// Synchronize the IO context's stream (wait for async IO to complete).
void bam_vchar_io_v2_sync(bam_vchar_io_ctx_t ctx);

// Destroy the VCHAR I/O v2 context.
void bam_vchar_io_v2_destroy(bam_vchar_io_ctx_t ctx);

// ============================================================
// Per-thread LZ4 decompression test
// ============================================================

// Test per-thread LZ4 decompression on GPU.
// comp_buf: host buffer with LZ4 compressed data.
// comp_size: compressed size in bytes.
// decomp_out: host buffer to receive decompressed bytes.
// max_output: max output buffer size in bytes.
// Returns actual decompressed size, or 0 on error.
uint32_t bam_test_lz4_decompress_page(const void* comp_buf, size_t comp_size,
                                       void* decomp_out, uint32_t max_output);

// ============================================================
// scan_o_comment v4: single-kernel IO + per-thread LZ4 + VCHAR scan
// ============================================================

// Run single-kernel scan: BAM IO + per-thread LZ4 decompress + VCHAR scan.
// num_blocks: grid size (128 threads/block, 32 pages/warp × 4 warps).
// Page cache = num_blocks * 128 entries, decompress buf = num_blocks * 128 * page_size.
void bam_scan_o_comment_v4_run(
    bam_ctrl_handle_t ctrl,
    uint32_t page_size,
    uint64_t npages,
    uint64_t partition_start_lba,
    const uint32_t* h_comp_sizes,
    const uint64_t* h_comp_offsets,
    uint32_t num_blocks,
    uint64_t* out_records,
    uint64_t* out_strlen,
    uint64_t* out_byte_sum);

// ============================================================
// scan_o_comment v5: GPU-internal IO pipelining (double-buffered)
// ============================================================

// Run double-buffered single-kernel scan: BAM async IO + per-thread LZ4 + VCHAR scan.
// Coalesced IO + block-cooperative scan.
// Thread 0 submits coalesced NVMe reads (up to coalesce_limit bytes each).
// Threads 0..31 decompress (1 page/thread).
// All 128 threads cooperatively scan (4 threads/page).
void bam_scan_o_comment_v5_run(
    bam_ctrl_handle_t ctrl,
    uint32_t page_size,
    uint64_t npages,
    uint64_t partition_start_lba,
    const uint32_t* h_comp_sizes,
    const uint64_t* h_comp_offsets,
    uint32_t num_blocks,
    uint32_t coalesce_limit,
    uint64_t* out_records,
    uint64_t* out_strlen,
    uint64_t* out_byte_sum);

// scan_o_comment v6: moved to bam_vchar_kernel.cu (PAR-32K-nvCOMPdx)
// See bam_vchar_kernel.cuh for the new API.

// ============================================================
// Batch page reader: reads N pages via BaM using existing pfor32_io
// context.  Eliminates per-call page_cache creation overhead.
// ============================================================
struct BAMBatchReadEntry {
    uint64_t lba;
    uint32_t dev;
    uint32_t nblk;
};

// Read n_pages via BaM and copy to host buffer (h_output must be n_pages * page_size).
// ctx must have at least n_pages page cache slots.
void bam_read_pages_batch_to_host(
    bam_pfor32_io_ctx_t ctx,
    uint32_t page_size,
    const BAMBatchReadEntry* h_entries,
    uint32_t n_pages,
    char* h_output,
    cudaStream_t stream);

// Read n_pages via BaM and copy to GPU buffer (d_output must be n_pages * page_size).
void bam_read_pages_batch_to_gpu(
    bam_pfor32_io_ctx_t ctx,
    uint32_t page_size,
    const BAMBatchReadEntry* h_entries,
    uint32_t n_pages,
    char* d_output,
    cudaStream_t stream,
    size_t* out_kernel_launches = nullptr);

// Same as above but uses a pre-allocated device buffer for entries
// (avoids cudaMalloc/cudaFree per call — Rule 4 compliance).
void bam_read_pages_batch_to_gpu_prealloc(
    bam_pfor32_io_ctx_t ctx,
    uint32_t page_size,
    const BAMBatchReadEntry* h_entries,
    uint32_t n_pages,
    char* d_output,
    BAMBatchReadEntry* d_entries,   // pre-allocated, capacity >= min(n_pages, num_blocks)
    cudaStream_t stream,
    size_t* out_kernel_launches = nullptr);

// ============================================================
// SSB Q1x fused kernel: BaM IO + PFOR decomp + scan in one kernel
// 4 INT32 fields (orderdate, quantity, discount, extendedprice)
// ============================================================
struct SSBDpfQ1xParams {
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;
    uint32_t num_blocks;
    uint64_t field_start_page_ids[4];
    uint16_t comp_methods[4];
    uint64_t* d_prefix_sum;
    uint32_t* d_comp_sizes[4];
    uint64_t* d_comp_offsets[4];
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    int32_t* d_scratch;
    uint32_t scratch_stride;
    // Q1x filter
    const int32_t* d_date_ht_keys;
    const int32_t* d_date_ht_values;
    uint32_t date_ht_mask;
    int32_t disc_lo, disc_hi, qty_lo, qty_hi;
};

void ssb_dpf_q1x_run(
    bam_pfor32_io_ctx_t ctx,
    const SSBDpfQ1xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream);

// ============================================================
// SSB Q2x fused kernel: 4 INT32 fields + 2 HT probes → group-by
// ============================================================
struct SSBDpfQ2xParams {
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;
    uint32_t num_blocks;
    uint64_t field_start_page_ids[4]; // orderdate, partkey, suppkey, revenue
    uint16_t comp_methods[4];
    uint64_t* d_prefix_sum;
    uint32_t* d_comp_sizes[4];
    uint64_t* d_comp_offsets[4];
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    int32_t* d_scratch;
    uint32_t scratch_stride;
    // Date hash table
    const int32_t* d_date_ht_keys;
    const int32_t* d_date_ht_values;
    uint32_t date_ht_mask;
    // Supplier HT
    const int32_t* d_supp_ht_keys;
    const int32_t* d_supp_ht_values;
    uint32_t supp_ht_mask;
    // Part HT
    const int32_t* d_part_ht_keys;
    const int32_t* d_part_ht_values;
    uint32_t part_ht_mask;
};

void ssb_dpf_q2x_run(
    bam_pfor32_io_ctx_t ctx,
    const SSBDpfQ2xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream);

// ============================================================
// SSB Q3x fused kernel: 4 INT32 fields + 2 HT probes → 3D group-by
// ============================================================
struct SSBDpfQ3xParams {
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;
    uint32_t num_blocks;
    uint64_t field_start_page_ids[4]; // orderdate, custkey, suppkey, revenue
    uint16_t comp_methods[4];
    uint64_t* d_prefix_sum;
    uint32_t* d_comp_sizes[4];
    uint64_t* d_comp_offsets[4];
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    int32_t* d_scratch;
    uint32_t scratch_stride;
    const int32_t* d_date_ht_keys;
    const int32_t* d_date_ht_values;
    uint32_t date_ht_mask;
    const int32_t* d_cust_ht_keys;
    const int32_t* d_cust_ht_values;
    uint32_t cust_ht_mask;
    const int32_t* d_supp_ht_keys;
    const int32_t* d_supp_ht_values;
    uint32_t supp_ht_mask;
    int32_t num_supp_dims;
    uint32_t max_years;
    uint32_t hist_size;  // = num_cust_dims * num_supp_dims * max_years
};

void ssb_dpf_q3x_run(
    bam_pfor32_io_ctx_t ctx,
    const SSBDpfQ3xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream);

// ============================================================
// SSB Q4x fused kernel: 6 INT32 fields + 3 HT probes → 4D group-by
// ============================================================
struct SSBDpfQ4xParams {
    uint64_t partition_start_lbas[MAX_BAM_DEVICES];
    uint32_t n_devices;
    uint32_t page_size;
    uint32_t blocks_per_page;
    uint64_t npages;
    uint32_t num_blocks;
    uint64_t field_start_page_ids[6]; // orderdate, custkey, partkey, suppkey, revenue, supplycost
    uint16_t comp_methods[6];
    uint64_t* d_prefix_sum;
    uint32_t* d_comp_sizes[6];
    uint64_t* d_comp_offsets[6];
    const uint8_t* d_page_active;
    const uint32_t* d_active_page_ids;
    uint32_t num_active_pages;
    int32_t* d_scratch;
    uint32_t scratch_stride;
    const int32_t* d_date_ht_keys;
    const int32_t* d_date_ht_values;
    uint32_t date_ht_mask;
    const int32_t* d_cust_ht_keys;
    const int32_t* d_cust_ht_values;
    uint32_t cust_ht_mask;
    const int32_t* d_supp_ht_keys;
    const int32_t* d_supp_ht_values;
    uint32_t supp_ht_mask;
    const int32_t* d_part_ht_keys;
    const int32_t* d_part_ht_values;
    uint32_t part_ht_mask;
    int32_t supp_dims, part_dims, stride_year;
    uint32_t year_min, max_years;
    uint32_t hist_size;  // = max_years * stride_year
};

void ssb_dpf_q4x_run(
    bam_pfor32_io_ctx_t ctx,
    const SSBDpfQ4xParams& params,
    int64_t* d_profit,
    cudaStream_t stream);

// ── SSB Uncompressed DPF kernels (no PFOR shared workspace) ──
void ssb_dpf_q1x_uncomp_run(
    bam_pfor32_io_ctx_t ctx,
    const SSBDpfQ1xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream);

void ssb_dpf_q2x_uncomp_run(
    bam_pfor32_io_ctx_t ctx,
    const SSBDpfQ2xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream);

void ssb_dpf_q3x_uncomp_run(
    bam_pfor32_io_ctx_t ctx,
    const SSBDpfQ3xParams& params,
    int64_t* d_revenue,
    cudaStream_t stream);

void ssb_dpf_q4x_uncomp_run(
    bam_pfor32_io_ctx_t ctx,
    const SSBDpfQ4xParams& params,
    int64_t* d_profit,
    cudaStream_t stream);
