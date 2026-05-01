#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================
// bam_lz4_fused_q16_ptype.cuh — Fused BaM I/O + nvCOMPdx LZ4 + Q16 P_TYPE scan
//
// Single persistent kernel for Q16 P_TYPE VCHAR phase.
// Reads P_TYPE pages, decompresses, checks NOT LIKE 'MEDIUM POLISHED%',
// and assigns type_ids via GPU-side dictionary (FNV-1a hash, open addressing).
// ============================================================

typedef void* bam_fused_q16pt_ctx_t;
typedef void* bam_ctrl_handle_t;

struct BAMFusedQ16PTypeParams {
    uint64_t  field_start_page_id;
    uint64_t* d_comp_offsets;
    uint32_t* d_comp_sizes;
    bool      is_compressed;

    // Inclusive cumulative prefix sum [npages]
    const uint64_t* d_prefix_sum;

    uint64_t  partition_start_lbas[4];
    uint32_t  n_devices;
    uint32_t  page_size;
    uint64_t  npages;
    uint32_t  num_blocks;

    // GPU type dictionary (shared across all warps)
    uint64_t* d_dict_keys;        // [Q16_TYPE_DICT_CAP]
    uint32_t* d_dict_type_ids;    // [Q16_TYPE_DICT_CAP]
    char*     d_dict_strs;        // [Q16_TYPE_DICT_CAP * Q16_TYPE_STR_MAX]
    uint16_t* d_dict_lens;        // [Q16_TYPE_DICT_CAP]
    uint32_t* d_type_id_counter;  // [1]

    // Output: d_type_ids[nrecs_part], UINT32_MAX for excluded
    uint32_t* d_type_ids;
};

bam_fused_q16pt_ctx_t bam_fused_q16pt_create(
    bam_ctrl_handle_t ctrl_handle,
    uint32_t page_size,
    uint32_t num_blocks);

void bam_fused_q16pt_run_async(
    bam_fused_q16pt_ctx_t ctx,
    const BAMFusedQ16PTypeParams& params,
    cudaStream_t stream);

void bam_fused_q16pt_destroy(bam_fused_q16pt_ctx_t ctx);

// ============================================================
// Batch IO + fused nvCOMPdx decomp + P_TYPE scan
// Reads compressed pages from staging_io, decompresses with nvCOMPdx,
// and scans VCHAR in one kernel (no staging_data intermediate).
// ============================================================

struct BAMBatchQ16PTypeParams {
    const char* d_staging_io;      // compressed pages [batch_npages * page_size]
    char*       d_decomp_buf;      // decomp buffer pool [num_slots * page_size]
    const uint32_t* d_comp_sizes;  // compressed sizes [batch_npages]
    uint32_t    batch_npages;      // pages in this batch
    uint32_t    global_pg_offset;  // first page index (for prefix_sum lookup)

    // Global inclusive prefix sum for P_TYPE [total_npages]
    const uint64_t* d_prefix_sum;
    uint32_t    page_size;

    // GPU type dictionary (shared across batches)
    uint64_t*   d_dict_keys;
    uint32_t*   d_dict_type_ids;
    char*       d_dict_strs;
    uint16_t*   d_dict_lens;
    uint32_t*   d_type_id_counter;

    // Output
    uint32_t*   d_type_ids;
};

void bam_batch_q16pt_run(
    const BAMBatchQ16PTypeParams& params,
    cudaStream_t stream);
