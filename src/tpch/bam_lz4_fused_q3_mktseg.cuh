#pragma once

#include <cstdint>
#include <cuda_runtime.h>

struct BamBulkReadDesc;  // forward declaration (defined in bam_bulk_read.cuh)

// ============================================================
// bam_lz4_fused_q3_mktseg.cuh — CUSTOMER Phase 1 kernels
// for Q3 gidp+bam+fusion.
//
// Two-kernel approach:
//   Kernel 1 (IO+decomp): Warp-stride IO for all CUSTOMER pages
//             (CK+MK) into page_cache, then nvCOMPdx LZ4 decomp
//             to a page-indexed staging buffer.
//   Kernel 2 (process):   Cooperative kernel — record-level parallel
//             CK flatten → grid.sync() → MK BUILDING scan + hash insert.
//             No shared memory, maximum parallelism.
// ============================================================

// Kernel 1: IO all pages + decomp all pages → staging buffer
struct Q3CustIODecompParams {
    uint32_t total_descs;            // CK + MK total page count
    uint32_t ck_npages;              // first ck_npages descriptors are CK
    const uint32_t* ck_comp_sizes;   // [ck_npages], NULL if uncompressed
    const uint32_t* mk_comp_sizes;   // [mk_npages], NULL if uncompressed
    uint32_t page_size;
};

// Kernel 2: CK flatten + grid.sync() + MK scan + hash insert
struct Q3CustProcessParams {
    const char* d_staging;           // decompressed pages [total_pages * page_size]
    uint32_t page_size;
    // CK flatten
    const uint64_t* ck_prefix_sum;   // [ck_npages] inclusive
    uint32_t ck_npages;
    uint64_t nrecs_customer;
    uint64_t* d_c_custkey_flat;      // output
    // MK scan + hash insert
    uint32_t mk_page_offset;        // MK pages at d_staging + mk_page_offset * page_size
    const uint64_t* mk_prefix_sum;   // [mk_npages] inclusive
    uint32_t mk_npages;
    uint32_t padded_len;             // C_MKTSEGMENT padded length (12)
    uint64_t* d_custkey_set;         // hash set output
    uint32_t custkey_set_mask;       // hash set capacity - 1
    // Q3SEL: multi-segment support (0 = all pass, >0 = check segment_values)
    uint32_t num_segments;
    uint64_t segment_values[5];
};

// Kernel 1: max blocks query
uint32_t q3_cust_io_decomp_max_blocks(uint32_t page_size);

// Kernel 1: IO all pages + decomp → d_staging (page-indexed)
void q3_cust_io_decomp_launch(
    void* d_ctrls, void* d_pc, const char* pc_base_addr,
    const BamBulkReadDesc* d_descs,
    char* d_staging,
    const Q3CustIODecompParams& params,
    uint32_t num_blocks,
    cudaStream_t stream);

// Kernel 2: max blocks query (cooperative launch)
uint32_t q3_cust_process_max_blocks();

// Kernel 2: cooperative flatten + scan (cudaLaunchCooperativeKernel)
void q3_cust_process_launch(
    const Q3CustProcessParams& params,
    uint32_t num_blocks,
    cudaStream_t stream);
