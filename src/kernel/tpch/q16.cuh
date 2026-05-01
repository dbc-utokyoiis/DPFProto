#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>

// Result row for Q16
struct Q16ResultRow {
    std::string p_brand;
    std::string p_type;
    int32_t p_size;
    uint32_t supplier_cnt;
};

// GPU type dictionary constants
constexpr uint32_t Q16_TYPE_DICT_CAP = 512;
constexpr uint32_t Q16_TYPE_STR_MAX = 32;

// Build PART hash table from pre-processed columns.
cudaError_t q16_build_part_hashtable(
    const uint64_t *d_p_partkey,
    const uint32_t *d_p_brand_ids,
    const uint32_t *d_p_type_ids,
    const uint32_t *d_p_size,
    uint64_t nrecs_part,
    uint64_t p_size_bitmask,
    uint32_t brand_exclude_id,
    uint32_t num_types,
    uint64_t *d_ht_keys,
    uint32_t *d_ht_group_ids,
    uint32_t ht_mask,
    cudaStream_t stream);

// Fix partial group_ids in PART HT after P_TYPE completes (Stage 2).
// Applies type_id filter and computes final group_id from
// partial_gid encoding: (brand_id << 8) | (size_val - 1).
cudaError_t q16_fix_partial_group_ids(
    uint64_t *d_ht_keys,
    uint32_t *d_ht_group_ids,
    const uint32_t *d_ht_row_idx,
    uint32_t ht_capacity,
    const uint32_t *d_type_ids,
    uint32_t num_types,
    cudaStream_t stream);

// Probe PARTSUPP rows against PART hash table + excluded suppkey set.
cudaError_t q16_partsupp_probe(
    const uint64_t *d_ps_partkey,
    const uint64_t *d_ps_suppkey,
    uint64_t nrecs_partsupp,
    const uint64_t *d_ht_keys,
    const uint32_t *d_ht_group_ids,
    uint32_t ht_mask,
    const uint64_t *d_excl_keys,
    uint32_t excl_mask,
    uint64_t *d_emit_pairs,
    cudaStream_t stream);

// ============================================================
// Q16 pipeline: pre-allocated buffers (no malloc/free).
// ============================================================

// Pre-allocated scratch buffers for Q16 pipeline.
// Caller must cudaMalloc all fields before the measurement point.
struct Q16PipelineBuffers {
    uint64_t *d_emit_pairs;      // [nrecs_partsupp]
    uint64_t *d_sort_alt;        // [nrecs_partsupp]
    uint64_t *d_unique_keys;     // [nrecs_partsupp]
    uint32_t *d_unique_counts;   // [nrecs_partsupp]
    uint64_t *d_num_unique_ptr;  // [1]
    uint32_t *d_group_ids;       // [nrecs_partsupp]
    uint32_t *d_group_ids_alt;   // [nrecs_partsupp]
    uint32_t *d_result_gids;     // [nrecs_partsupp]
    uint32_t *d_result_counts;   // [nrecs_partsupp]
    uint64_t *d_num_groups_ptr;  // [1]
    void     *d_cub_temp;        // CUB scratch
    size_t    cub_temp_bytes;    // size of d_cub_temp
    // Host result buffers (pre-allocated, avoids heap alloc in timed section)
    uint32_t *h_gids;            // [h_result_capacity]
    uint32_t *h_counts;          // [h_result_capacity]
    size_t    h_result_capacity;
};

// Compute required CUB scratch size for Q16 pipeline.
size_t q16_pipeline_cub_temp_size(uint64_t nrecs_partsupp);

// Full Q16 pipeline (sort-based COUNT DISTINCT).
// Caller must pre-allocate all fields in bufs before the measurement point.
cudaError_t q16_golap_pipeline(
    const Q16PipelineBuffers &bufs,
    uint64_t *d_ht_keys,
    uint32_t *d_ht_group_ids,
    uint32_t ht_mask,
    uint64_t *d_excl_keys,
    uint32_t excl_mask,
    const uint64_t *d_ps_partkey,
    const uint64_t *d_ps_suppkey,
    uint64_t nrecs_partsupp,
    std::vector<std::pair<uint32_t, uint32_t>> &result,
    cudaStream_t stream);

// Scan S_COMMENT VCHAR pages on GPU with KMP, output excluded suppkeys.
cudaError_t q16_supplier_scan(
    const char *d_s_comment_pages,
    const uint64_t *d_prefix_sum,
    uint32_t npages, uint32_t page_size,
    const uint64_t *d_s_suppkey_flat,
    uint64_t nrecs_supplier,
    const char *d_patterns, const int *d_kmp_next,
    const int *d_pattern_offsets, const int *d_pattern_lengths,
    int num_patterns,
    uint64_t *d_excl_suppkeys, uint32_t *d_excl_count,
    cudaStream_t stream);

// Batch variant: scan a batch of S_COMMENT pages against a full prefix_sum.
// d_batch_pages: decompressed pages for this batch (batch_npages * page_size at offset 0).
// d_full_prefix_sum: full prefix sum array (total_npages elements, cumulative-inclusive).
// row_base / nrecs_batch: global row range for this batch.
cudaError_t q16_supplier_scan_batch(
    const char *d_batch_pages,
    const uint64_t *d_full_prefix_sum,
    uint32_t total_npages,
    uint32_t batch_start_page,
    uint32_t page_size,
    const uint64_t *d_s_suppkey_flat,
    uint64_t nrecs_batch,
    uint64_t row_base,
    const char *d_patterns, const int *d_kmp_next,
    const int *d_pattern_offsets, const int *d_pattern_lengths,
    int num_patterns,
    uint64_t *d_excl_suppkeys, uint32_t *d_excl_count,
    cudaStream_t stream);

// Build excluded suppkey hash table on GPU.
cudaError_t q16_build_excl_ht(
    const uint64_t *d_excl_suppkeys, uint32_t excl_count,
    uint64_t *d_excl_ht_keys, uint32_t excl_ht_mask,
    cudaStream_t stream);

// Extract brand_ids from P_BRAND CHAR pages on GPU.
cudaError_t q16_extract_brand_ids(
    const char *d_brand_pages,
    const uint64_t *d_prefix_sum,
    uint32_t npages, uint32_t page_size,
    uint32_t padded_len, uint64_t nrecs,
    uint32_t *d_brand_ids, cudaStream_t stream);

// Extract type_ids from P_TYPE VCHAR pages on GPU with dictionary.
cudaError_t q16_extract_type_ids(
    const char *d_type_pages,
    const uint64_t *d_prefix_sum,
    uint32_t npages, uint32_t page_size,
    uint64_t nrecs,
    uint64_t *d_dict_keys, uint32_t *d_dict_type_ids,
    char *d_dict_strs, uint16_t *d_dict_lens,
    uint32_t *d_type_id_counter,
    uint32_t *d_type_ids, cudaStream_t stream);

// Cast uint64_t array to uint32_t on GPU.
cudaError_t q16_cast_u64_to_u32(
    const uint64_t *d_in, uint32_t *d_out,
    uint64_t n, cudaStream_t stream);

// Run Q16 post-probe pipeline with pre-allocated buffers.
// Assumes d_emit_pairs is already populated (by fused kernel).
// Does: sort → RLE → extract → sort → RLE.
cudaError_t q16_post_probe_pipeline(
    const Q16PipelineBuffers &bufs,
    uint64_t nrecs_partsupp,
    std::vector<std::pair<uint32_t, uint32_t>> &result,
    cudaStream_t stream);
