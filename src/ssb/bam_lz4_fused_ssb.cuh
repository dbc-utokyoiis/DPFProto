#pragma once

// ============================================================
// bam_lz4_fused_ssb.cuh — SSB Fused BaM I/O + nvCOMPdx LZ4 + scan
//
// Warp-specialized persistent kernels:
//   Q1x/Q2x/Q3x: 1024 threads (32 warps), 4 IO + 28 decomp, BATCH=7
//   Q4x:          1024 threads (32 warps), 6 IO + 24 decomp, BATCH=4
//
// Pipeline: Prolog(IO batch 0) → Main(IO‖Decomp→Scan) → Epilog
// ============================================================

#include <cstdint>
#include <cuda_runtime.h>

typedef void* bam_ctrl_handle_t;

// ── Warp-spec constants: Q1x/Q2x/Q3x (4 fields) ──
//   4 IO warps + 7 decomp groups × 4 fields = 32 warps = 1024 threads
static constexpr uint32_t SSB_WS4_BATCH            = 7;
static constexpr uint32_t SSB_WS4_NUM_FIELDS        = 4;
static constexpr uint32_t SSB_WS4_IO_WARPS          = 4;
static constexpr uint32_t SSB_WS4_DECOMP_GROUPS     = 7;
static constexpr uint32_t SSB_WS4_SLOTS_PER_BUF     = 28;   // BATCH × NUM_FIELDS
static constexpr uint32_t SSB_WS4_SLOTS_PER_BLOCK   = 56;   // 2 × SLOTS_PER_BUF

// ── Warp-spec constants: Q4x (6 fields) ──
//   6 IO warps + 4 decomp groups × 6 fields = 30 warps (+2 idle) = 1024 threads
static constexpr uint32_t SSB_WS6_BATCH            = 4;
static constexpr uint32_t SSB_WS6_NUM_FIELDS        = 6;
static constexpr uint32_t SSB_WS6_IO_WARPS          = 6;
static constexpr uint32_t SSB_WS6_DECOMP_GROUPS     = 4;
static constexpr uint32_t SSB_WS6_SLOTS_PER_BUF     = 24;   // BATCH × NUM_FIELDS
static constexpr uint32_t SSB_WS6_SLOTS_PER_BLOCK   = 48;   // 2 × SLOTS_PER_BUF

// ── SSB Q1x Warp-Spec Parameters ──
// Fields: 0=LO_ORDERDATE, 1=LO_QUANTITY, 2=LO_DISCOUNT, 3=LO_EXTENDEDPRICE
struct SSBFusedQ1xParams {
    uint64_t  field_start_page_ids[4];
    uint64_t* d_comp_offsets[4];
    uint32_t* d_comp_sizes[4];
    bool      is_compressed[4];
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;
    uint32_t  page_size;
    uint32_t  total_pages;
    const uint32_t* d_active_page_ids;
    const uint8_t*  d_page_mask;       // GPU zonemap mask (nullptr = no pruning)
    // Q1x predicates
    const int32_t* d_date_ht_keys;
    const int32_t* d_date_ht_values;
    uint32_t  date_ht_mask;
    int32_t   disc_lo, disc_hi;
    int32_t   qty_lo, qty_hi;
    // Output
    int64_t*  d_revenue;
};

// ── SSB Q2x Warp-Spec Parameters ──
// Fields: 0=LO_ORDERDATE, 1=LO_PARTKEY, 2=LO_SUPPKEY, 3=LO_REVENUE
struct SSBFusedQ2xParams {
    uint64_t  field_start_page_ids[4];
    uint64_t* d_comp_offsets[4];
    uint32_t* d_comp_sizes[4];
    bool      is_compressed[4];
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;
    uint32_t  page_size;
    uint32_t  total_pages;
    const uint32_t* d_active_page_ids;
    const uint8_t*  d_page_mask;       // GPU zonemap mask (nullptr = no pruning)
    // Date hash table: datekey → year_idx (0-based)
    const int32_t* d_date_ht_keys;
    const int32_t* d_date_ht_values;
    uint32_t  date_ht_mask;
    // Supplier hash table
    const int32_t* d_supp_ht_keys;
    const int32_t* d_supp_ht_values;
    uint32_t  supp_ht_mask;
    // Part hash table: partkey → brand_idx
    const int32_t* d_part_ht_keys;
    const int32_t* d_part_ht_values;
    uint32_t  part_ht_mask;
    // Output: revenue[year_idx * MAX_BRANDS + brand_idx]
    int64_t*  d_revenue;
};

// ── SSB Q3x Warp-Spec Parameters ──
// Fields: 0=LO_ORDERDATE, 1=LO_CUSTKEY, 2=LO_SUPPKEY, 3=LO_REVENUE
struct SSBFusedQ3xParams {
    uint64_t  field_start_page_ids[4];
    uint64_t* d_comp_offsets[4];
    uint32_t* d_comp_sizes[4];
    bool      is_compressed[4];
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;
    uint32_t  page_size;
    uint32_t  total_pages;
    const uint32_t* d_active_page_ids;
    const uint8_t*  d_page_mask;       // GPU zonemap mask (nullptr = no pruning)
    // Date hash table: datekey → year_idx (0-based)
    const int32_t* d_date_ht_keys;
    const int32_t* d_date_ht_values;
    uint32_t  date_ht_mask;
    // Customer hash table
    const int32_t* d_cust_ht_keys;
    const int32_t* d_cust_ht_values;
    uint32_t  cust_ht_mask;
    // Supplier hash table
    const int32_t* d_supp_ht_keys;
    const int32_t* d_supp_ht_values;
    uint32_t  supp_ht_mask;
    int32_t   num_supp_dims;
    uint32_t  hist_size;  // = num_cust_dims * num_supp_dims * MAX_YEARS
    // Output: revenue[cust_dim * num_supp_dims * MAX_YEARS + supp_dim * MAX_YEARS + year_idx]
    int64_t*  d_revenue;
};

// ── SSB Q4x Warp-Spec Parameters ──
// Fields: 0=LO_ORDERDATE, 1=LO_CUSTKEY, 2=LO_PARTKEY, 3=LO_SUPPKEY,
//         4=LO_REVENUE, 5=LO_SUPPLYCOST
struct SSBFusedQ4xParams {
    uint64_t  field_start_page_ids[6];
    uint64_t* d_comp_offsets[6];
    uint32_t* d_comp_sizes[6];
    bool      is_compressed[6];
    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;
    uint32_t  page_size;
    uint32_t  total_pages;
    const uint32_t* d_active_page_ids;
    const uint8_t*  d_page_mask;       // GPU zonemap mask (nullptr = no pruning)
    // Date hash table: datekey → year_idx (0-based)
    const int32_t* d_date_ht_keys;
    const int32_t* d_date_ht_values;
    uint32_t  date_ht_mask;
    // Customer hash table
    const int32_t* d_cust_ht_keys;
    const int32_t* d_cust_ht_values;
    uint32_t  cust_ht_mask;
    // Supplier hash table
    const int32_t* d_supp_ht_keys;
    const int32_t* d_supp_ht_values;
    uint32_t  supp_ht_mask;
    // Part hash table
    const int32_t* d_part_ht_keys;
    const int32_t* d_part_ht_values;
    uint32_t  part_ht_mask;
    // Group-by strides
    int32_t   supp_dims;
    int32_t   part_dims;
    int32_t   stride_year;  // cust_dims * supp_dims * part_dims
    uint32_t  hist_size;    // = MAX_YEARS * stride_year
    // Output: profit[year_idx * stride_year + ...]
    int64_t*  d_profit;
};

// ── Fused context (page_cache + decomp buffer) ──
typedef void* ssb_fused_ctx_t;

// Create context. slots_per_block = SSB_WS4_SLOTS_PER_BLOCK (56) or SSB_WS6_SLOTS_PER_BLOCK (48).
ssb_fused_ctx_t ssb_fused_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks,
    uint32_t slots_per_block);

void* ssb_fused_get_d_ctrls(ssb_fused_ctx_t ctx);
void* ssb_fused_get_d_pc_ptr(ssb_fused_ctx_t ctx);
const char* ssb_fused_get_pc_base(ssb_fused_ctx_t ctx);
char* ssb_fused_get_decomp_buf(ssb_fused_ctx_t ctx);
void ssb_fused_destroy(ssb_fused_ctx_t ctx);

// Max co-resident blocks (for grid size computation).
uint32_t ssb_fused_q1x_max_blocks(uint32_t page_size);  // also for Q2x/Q3x
uint32_t ssb_fused_q4x_max_blocks(uint32_t page_size);

// Kernel launch functions.
void ssb_fused_q1x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ1xParams& p,
    uint32_t num_blocks, cudaStream_t stream);

void ssb_fused_q2x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ2xParams& p,
    uint32_t num_blocks, cudaStream_t stream);

void ssb_fused_q3x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ3xParams& p,
    uint32_t num_blocks, cudaStream_t stream);

void ssb_fused_q4x_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    char* d_decomp_buf, const SSBFusedQ4xParams& p,
    uint32_t num_blocks, cudaStream_t stream);

// ── Dim table fused IO + nvCOMPdx decomp (1 kernel per field) ──
// Replaces BaM batch_read + nvCOMP host decomp (2 kernels per field).
// d_entries: device buffer with {lba, dev, nblk} per page (layout of BAMBatchReadEntry).
// d_comp_sizes: device buffer with compressed size per page.
// Caller must cudaMemcpyAsync both to device before calling.
void bam_dim_io_decomp_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    uint32_t slot_base, uint32_t max_slots, uint32_t page_size,
    char* d_output,
    const void* d_entries,
    const uint32_t* d_comp_sizes,
    uint32_t total_pages,
    cudaStream_t stream);

// ── Dim table fused IO + block-wide copy (for NONE/uncompressed fields) ──
// 256 threads per block. Thread 0 does BaM IO, all threads copy slot → d_output.
// Slot-split aware (slot_base/max_slots), no nvCOMPdx needed.
void bam_dim_io_copy_launch(
    void* d_ctrls, void* d_pc_ptr, const char* pc_base,
    uint32_t slot_base, uint32_t max_slots, uint32_t page_size,
    char* d_output,
    const void* d_entries,
    uint32_t total_pages,
    cudaStream_t stream);
