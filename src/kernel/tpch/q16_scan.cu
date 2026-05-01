// Q16 GOLAP kernels: PART hash table build, PARTSUPP probe + anti-join,
// sort-based COUNT DISTINCT pipeline.
//
// TPC-H Q16: Parts/Supplier Relationship
// SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt
// FROM partsupp, part
// WHERE p_partkey = ps_partkey
//   AND p_brand <> 'Brand#45'
//   AND p_type NOT LIKE 'MEDIUM POLISHED%'
//   AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
//   AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier
//                          WHERE s_comment LIKE '%Customer%Complaints%')
// GROUP BY p_brand, p_type, p_size
// ORDER BY supplier_cnt DESC, p_brand, p_type, p_size;

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <utility>

#include "q16.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                               \
        return _e;                                                     \
    }                                                                  \
} while (0)
#endif

// ============================================================
// Hash table device functions (open addressing, linear probing)
// ============================================================
static constexpr uint64_t HT_EMPTY = UINT64_MAX;

__device__ __forceinline__ uint32_t hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

// Insert into hash table (no duplicates expected for partkey)
__device__ __forceinline__ void ht_insert(
    uint64_t *keys, uint32_t *group_ids, uint32_t mask,
    uint64_t key, uint32_t group_id)
{
    uint32_t slot = hash64(key) & mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)HT_EMPTY,
            (unsigned long long)key);
        if (prev == HT_EMPTY || prev == key) {
            group_ids[slot] = group_id;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

// Probe hash table: returns group_id or UINT32_MAX if not found
__device__ __forceinline__ uint32_t ht_probe(
    const uint64_t *keys, const uint32_t *group_ids, uint32_t mask,
    uint64_t key)
{
    uint32_t slot = hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return group_ids[slot];
        if (k == HT_EMPTY) return UINT32_MAX;
        slot = (slot + 1) & mask;
    }
}

// Probe excluded suppkey set: returns true if key is excluded
__device__ __forceinline__ bool excl_probe(
    const uint64_t *keys, uint32_t mask, uint64_t key)
{
    uint32_t slot = hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return true;
        if (k == HT_EMPTY) return false;
        slot = (slot + 1) & mask;
    }
}

// ============================================================
// Kernel: Build PART hash table
// ============================================================
__global__ void q16_build_part_ht_kernel(
    const uint64_t *__restrict__ p_partkey,
    const uint32_t *__restrict__ p_brand_ids,
    const uint32_t *__restrict__ p_type_ids,
    const uint32_t *__restrict__ p_size,
    uint64_t nrecs_part,
    uint64_t p_size_bitmask,
    uint32_t brand_exclude_id,
    uint32_t num_types,
    uint64_t *__restrict__ ht_keys,
    uint32_t *__restrict__ ht_group_ids,
    uint32_t ht_mask)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_part) return;

    uint32_t brand_id = p_brand_ids[idx];
    uint32_t type_id = p_type_ids[idx];
    uint32_t size_val = p_size[idx];

    // Predicate 1: p_brand <> 'Brand#45' (brand_id != brand_exclude_id)
    if (brand_id == brand_exclude_id) return;

    // Predicate 2: p_type NOT LIKE 'MEDIUM POLISHED%'
    // type_id == UINT32_MAX means the type string matches 'MEDIUM POLISHED%' (set by host)
    if (type_id == UINT32_MAX) return;

    // Predicate 3: p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
    if (size_val >= 64 || !((p_size_bitmask >> size_val) & 1)) return;

    // All predicates passed — compute group_id and insert
    // group_id = brand_id * num_types * 50 + type_id * 50 + (size_val - 1)
    // size_val is 1-based (1..50), so size_val-1 gives 0..49
    uint32_t group_id = brand_id * (num_types * 50) + type_id * 50 + (size_val - 1);

    uint64_t partkey = p_partkey[idx];
    ht_insert(ht_keys, ht_group_ids, ht_mask, partkey, group_id);
}

// ============================================================
// Kernel: Probe PARTSUPP against PART HT + excluded suppkey set
// ============================================================
__global__ void q16_partsupp_probe_kernel(
    const uint64_t *__restrict__ ps_partkey,
    const uint64_t *__restrict__ ps_suppkey,
    uint64_t nrecs_partsupp,
    const uint64_t *__restrict__ ht_keys,
    const uint32_t *__restrict__ ht_group_ids,
    uint32_t ht_mask,
    const uint64_t *__restrict__ excl_keys,
    uint32_t excl_mask,
    uint64_t *__restrict__ d_emit_pairs)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_partsupp) return;

    uint64_t pk = ps_partkey[idx];
    uint64_t sk = ps_suppkey[idx];

    // Probe PART hash table
    uint32_t group_id = ht_probe(ht_keys, ht_group_ids, ht_mask, pk);
    if (group_id == UINT32_MAX) {
        d_emit_pairs[idx] = HT_EMPTY;
        return;
    }

    // Anti-join: check ps_suppkey NOT IN excluded set
    if (excl_probe(excl_keys, excl_mask, sk)) {
        d_emit_pairs[idx] = HT_EMPTY;
        return;
    }

    // Emit composite: group_id in upper 32 bits, ps_suppkey in lower 32 bits
    d_emit_pairs[idx] = ((uint64_t)group_id << 32) | (uint64_t)(uint32_t)sk;
}

// ============================================================
// Kernel: Extract group_id from unique composite keys
// ============================================================
__global__ void q16_extract_group_ids_kernel(
    const uint64_t *__restrict__ unique_keys,
    uint32_t *__restrict__ group_ids,
    uint64_t n)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    group_ids[idx] = (uint32_t)(unique_keys[idx] >> 32);
}

// ============================================================
// Host wrappers
// ============================================================

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
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_part + BLOCK - 1) / BLOCK);
    q16_build_part_ht_kernel<<<grid, BLOCK, 0, stream>>>(
        d_p_partkey, d_p_brand_ids, d_p_type_ids, d_p_size,
        nrecs_part, p_size_bitmask, brand_exclude_id, num_types,
        d_ht_keys, d_ht_group_ids, ht_mask);
    return cudaGetLastError();
}

// ============================================================
// Kernel: Fix partial group_ids in PART HT (Stage 2)
// Applies type_id filter and computes final group_id.
// partial_gid encoding: (brand_id << 8) | (size_val - 1)
// ============================================================
__global__ void q16_fix_partial_gids_kernel(
    uint64_t *__restrict__ ht_keys,
    uint32_t *__restrict__ ht_group_ids,
    const uint32_t *__restrict__ ht_row_idx,
    uint32_t ht_capacity,
    const uint32_t *__restrict__ type_ids,
    uint32_t num_types)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ht_capacity) return;

    uint64_t key = ht_keys[idx];
    if (key == HT_EMPTY) return;

    uint32_t row_idx  = ht_row_idx[idx];
    uint32_t type_id  = type_ids[row_idx];

    if (type_id == UINT32_MAX) {
        // Type matches 'MEDIUM POLISHED%' — remove entry
        ht_keys[idx] = HT_EMPTY;
        return;
    }

    uint32_t partial  = ht_group_ids[idx];
    uint32_t brand_id = partial >> 8;
    uint32_t size_idx = partial & 0xFF;

    uint32_t group_id = brand_id * (num_types * 50) + type_id * 50 + size_idx;
    ht_group_ids[idx] = group_id;
}

cudaError_t q16_fix_partial_group_ids(
    uint64_t *d_ht_keys,
    uint32_t *d_ht_group_ids,
    const uint32_t *d_ht_row_idx,
    uint32_t ht_capacity,
    const uint32_t *d_type_ids,
    uint32_t num_types,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((ht_capacity + BLOCK - 1) / BLOCK);
    q16_fix_partial_gids_kernel<<<grid, BLOCK, 0, stream>>>(
        d_ht_keys, d_ht_group_ids, d_ht_row_idx, ht_capacity,
        d_type_ids, num_types);
    return cudaGetLastError();
}

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
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_partsupp + BLOCK - 1) / BLOCK);
    q16_partsupp_probe_kernel<<<grid, BLOCK, 0, stream>>>(
        d_ps_partkey, d_ps_suppkey, nrecs_partsupp,
        d_ht_keys, d_ht_group_ids, ht_mask,
        d_excl_keys, excl_mask,
        d_emit_pairs);
    return cudaGetLastError();
}

// Predicate for DeviceSelect::If — selects non-sentinel entries
struct Q16NotSentinel {
    __host__ __device__ __forceinline__
    bool operator()(const uint64_t &val) const {
        return val != UINT64_MAX;
    }
};

// ============================================================
// Full Q16 pipeline: probe → compact → sort → RLE → extract → sort → RLE
// ============================================================
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
    cudaStream_t stream)
{
    cudaError_t err;

    // ── Step 1: Probe PARTSUPP ──
    CUDA_CHECK(cudaMemsetAsync(bufs.d_emit_pairs, 0xFF,
        nrecs_partsupp * sizeof(uint64_t), stream));

    err = q16_partsupp_probe(d_ps_partkey, d_ps_suppkey, nrecs_partsupp,
        d_ht_keys, d_ht_group_ids, ht_mask,
        d_excl_keys, excl_mask,
        bufs.d_emit_pairs, stream);
    if (err != cudaSuccess) return err;

    // ── Step 1.5: Compact — remove UINT64_MAX sentinels ──
    size_t select_temp = bufs.cub_temp_bytes;
    cub::DeviceSelect::If(bufs.d_cub_temp, select_temp,
        bufs.d_emit_pairs, bufs.d_sort_alt, bufs.d_num_unique_ptr,
        (int)nrecs_partsupp, Q16NotSentinel(), stream);

    uint64_t h_num_compacted = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_compacted, bufs.d_num_unique_ptr,
        sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Step 2: Sort compacted pairs (much smaller than nrecs_partsupp) ──
    cub::DoubleBuffer<uint64_t> db_pairs(bufs.d_sort_alt, bufs.d_emit_pairs);

    size_t sort_temp = bufs.cub_temp_bytes;
    cub::DeviceRadixSort::SortKeys(bufs.d_cub_temp, sort_temp, db_pairs,
        (int)h_num_compacted, 0, 64, stream);

    // ── Step 3: RLE → unique (group_id, ps_suppkey) pairs ──
    size_t rle_temp = bufs.cub_temp_bytes;
    cub::DeviceRunLengthEncode::Encode(bufs.d_cub_temp, rle_temp,
        db_pairs.Current(), bufs.d_unique_keys, bufs.d_unique_counts,
        bufs.d_num_unique_ptr, (int)h_num_compacted, stream);

    uint64_t h_num_unique = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_unique, bufs.d_num_unique_ptr,
        sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("[Q16] Unique (group_id, ps_suppkey) pairs: %lu\n", h_num_unique);

    // ── Step 4: Extract group_ids from unique keys ──
    {
        constexpr int BLOCK = 256;
        int grid = (int)((h_num_unique + BLOCK - 1) / BLOCK);
        q16_extract_group_ids_kernel<<<grid, BLOCK, 0, stream>>>(
            bufs.d_unique_keys, bufs.d_group_ids, h_num_unique);
    }

    // ── Step 5: Sort group_ids ──
    cub::DoubleBuffer<uint32_t> db_gids(bufs.d_group_ids, bufs.d_group_ids_alt);

    size_t sort2_temp = bufs.cub_temp_bytes;
    cub::DeviceRadixSort::SortKeys(bufs.d_cub_temp, sort2_temp, db_gids,
        (int)h_num_unique, 0, 32, stream);

    // ── Step 6: RLE on group_ids → (group_id, supplier_cnt) ──
    size_t rle2_temp = bufs.cub_temp_bytes;
    cub::DeviceRunLengthEncode::Encode(bufs.d_cub_temp, rle2_temp,
        db_gids.Current(), bufs.d_result_gids, bufs.d_result_counts,
        bufs.d_num_groups_ptr, (int)h_num_unique, stream);

    uint64_t h_num_groups = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_groups, bufs.d_num_groups_ptr,
        sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("[Q16] Distinct groups: %lu\n", h_num_groups);

    // ── Step 7: Copy results to host (use pre-allocated buffers) ──
    assert(bufs.h_gids != nullptr && bufs.h_counts != nullptr && bufs.h_result_capacity >= h_num_groups);
    CUDA_CHECK(cudaMemcpy(bufs.h_gids, bufs.d_result_gids,
        h_num_groups * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bufs.h_counts, bufs.d_result_counts,
        h_num_groups * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    result.clear();
    result.reserve(h_num_groups);
    for (uint64_t i = 0; i < h_num_groups; i++) {
        result.emplace_back(bufs.h_gids[i], bufs.h_counts[i]);
    }

    return cudaSuccess;
}

// ============================================================
// Device helpers: VCHAR/CHAR page access + binary search
// ============================================================
__device__ __forceinline__ uint32_t q16_dev_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}
__device__ __forceinline__ uint32_t q16_dev_pag_get_oslt(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}
__device__ __forceinline__ uint16_t q16_dev_pagcol_vchar_len(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = q16_dev_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}
__device__ __forceinline__ const char *q16_dev_pagcol_vchar_data(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = q16_dev_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);
}

template <typename T>
__device__ __forceinline__ int q16_upper_bound(
    const T *__restrict__ data, int n, const T &val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (data[mid] <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ============================================================
// KMP multi-pattern match (device-side)
// ============================================================
__device__ bool q16_dev_kmp_multi_match(
    const char *text, uint16_t text_len,
    const char *patterns, const int *kmp_next,
    const int *pattern_offsets, const int *pattern_lengths,
    int num_patterns)
{
    int text_pos = 0;
    for (int p = 0; p < num_patterns; p++) {
        int off = pattern_offsets[p];
        int plen = pattern_lengths[p];
        int j = 0;
        bool found = false;
        for (int i = text_pos; i < (int)text_len; i++) {
            while (j > 0 && text[i] != patterns[off + j])
                j = kmp_next[off + j - 1];
            if (text[i] == patterns[off + j]) j++;
            if (j == plen) {
                text_pos = i + 1;
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// ============================================================
// FNV-1a 64-bit hash for type strings
// ============================================================
__device__ __forceinline__ uint64_t q16_dev_fnv1a64(const char *s, uint16_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (uint16_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ============================================================
// Kernel: Scan S_COMMENT for '%Customer%Complaints%'
// ============================================================
__global__ void q16_supplier_scan_kernel(
    const char *__restrict__ s_comment_pages,
    const uint64_t *__restrict__ prefix_sum,
    uint32_t npages,
    uint32_t page_size,
    const uint64_t *__restrict__ s_suppkey_flat,
    uint64_t nrecs_supplier,
    const char *__restrict__ patterns,
    const int *__restrict__ kmp_next,
    const int *__restrict__ pattern_offsets,
    const int *__restrict__ pattern_lengths,
    int num_patterns,
    uint64_t *__restrict__ excl_suppkeys,
    uint32_t *__restrict__ excl_count)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_supplier) return;

    int page_idx = q16_upper_bound(prefix_sum, (int)npages, (uint64_t)idx);
    uint32_t local_slot = (page_idx == 0) ? (uint32_t)idx
        : (uint32_t)(idx - prefix_sum[page_idx - 1]);

    const char *page = s_comment_pages + (uint64_t)page_idx * page_size;
    uint16_t vlen = q16_dev_pagcol_vchar_len(page, local_slot, page_size);
    const char *vdata = q16_dev_pagcol_vchar_data(page, local_slot, page_size);

    if (q16_dev_kmp_multi_match(vdata, vlen,
            patterns, kmp_next, pattern_offsets, pattern_lengths, num_patterns)) {
        uint32_t pos = atomicAdd(excl_count, 1);
        excl_suppkeys[pos] = s_suppkey_flat[idx];
    }
}

// ============================================================
// Kernel: Batch variant of supplier scan — scans a batch of
// decompressed S_COMMENT pages using the full prefix_sum.
// ============================================================
__global__ void q16_supplier_scan_batch_kernel(
    const char *__restrict__ batch_pages,
    const uint64_t *__restrict__ full_prefix_sum,
    uint32_t total_npages,
    uint32_t batch_start_page,
    uint32_t page_size,
    const uint64_t *__restrict__ s_suppkey_flat,
    uint64_t nrecs_batch,
    uint64_t row_base,
    const char *__restrict__ patterns,
    const int *__restrict__ kmp_next,
    const int *__restrict__ pattern_offsets,
    const int *__restrict__ pattern_lengths,
    int num_patterns,
    uint64_t *__restrict__ excl_suppkeys,
    uint32_t *__restrict__ excl_count)
{
    uint64_t local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= nrecs_batch) return;

    uint64_t global_row = row_base + local_idx;

    // Find global page using full prefix sum
    int global_page = q16_upper_bound(full_prefix_sum, (int)total_npages, global_row);
    int local_page = global_page - (int)batch_start_page;

    uint32_t local_slot = (global_page == 0) ? (uint32_t)global_row
        : (uint32_t)(global_row - full_prefix_sum[global_page - 1]);

    const char *page = batch_pages + (uint64_t)local_page * page_size;
    uint16_t vlen = q16_dev_pagcol_vchar_len(page, local_slot, page_size);
    const char *vdata = q16_dev_pagcol_vchar_data(page, local_slot, page_size);

    if (q16_dev_kmp_multi_match(vdata, vlen,
            patterns, kmp_next, pattern_offsets, pattern_lengths, num_patterns)) {
        uint32_t pos = atomicAdd(excl_count, 1);
        excl_suppkeys[pos] = s_suppkey_flat[global_row];
    }
}

// ============================================================
// Kernel: Build excluded suppkey hash table (GPU-side)
// ============================================================
__global__ void q16_build_excl_ht_kernel(
    const uint64_t *__restrict__ excl_suppkeys,
    uint32_t excl_count,
    uint64_t *__restrict__ excl_ht_keys,
    uint32_t excl_ht_mask)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= excl_count) return;

    uint64_t sk = excl_suppkeys[idx];
    uint32_t slot = hash64(sk) & excl_ht_mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&excl_ht_keys[slot]),
            (unsigned long long)HT_EMPTY,
            (unsigned long long)sk);
        if (prev == HT_EMPTY || prev == sk) return;
        slot = (slot + 1) & excl_ht_mask;
    }
}

// ============================================================
// Kernel: Extract brand_ids from P_BRAND CHAR pages
// ============================================================
__global__ void q16_extract_brand_ids_kernel(
    const char *__restrict__ brand_pages,
    const uint64_t *__restrict__ prefix_sum,
    uint32_t npages,
    uint32_t page_size,
    uint32_t padded_len,
    uint64_t nrecs,
    uint32_t *__restrict__ brand_ids)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs) return;

    int page_idx = q16_upper_bound(prefix_sum, (int)npages, (uint64_t)idx);
    uint32_t local_slot = (page_idx == 0) ? (uint32_t)idx
        : (uint32_t)(idx - prefix_sum[page_idx - 1]);

    const char *brand = brand_pages + (uint64_t)page_idx * page_size
                        + 12 /* pag_head */ + padded_len * local_slot;
    uint32_t d1 = brand[6] - '1';
    uint32_t d2 = brand[7] - '1';
    brand_ids[idx] = d1 * 5 + d2;
}

// ============================================================
// Kernel: Extract type_ids from P_TYPE VCHAR pages
//   + GPU-side dictionary (open-addressing, FNV-1a 64-bit keys)
// ============================================================
// type_dict_keys[dict_capacity]: hash64 keys (init UINT64_MAX)
// type_dict_type_ids[dict_capacity]: type_id (init UINT32_MAX)
// type_dict_strs[dict_capacity * 32]: packed string data
// type_dict_lens[dict_capacity]: string lengths
// type_id_counter: atomic counter for sequential type_id assignment
constexpr uint32_t Q16_TYPE_DICT_CAPACITY = 512;
constexpr uint32_t Q16_TYPE_DICT_MASK = Q16_TYPE_DICT_CAPACITY - 1;
constexpr uint32_t Q16_TYPE_MAX_LEN = 32;

__global__ void q16_extract_type_ids_kernel(
    const char *__restrict__ type_pages,
    const uint64_t *__restrict__ prefix_sum,
    uint32_t npages,
    uint32_t page_size,
    uint64_t nrecs,
    uint64_t *__restrict__ dict_keys,
    uint32_t *__restrict__ dict_type_ids,
    char *__restrict__ dict_strs,
    uint16_t *__restrict__ dict_lens,
    uint32_t *__restrict__ type_id_counter,
    uint32_t *__restrict__ type_ids)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs) return;

    int page_idx = q16_upper_bound(prefix_sum, (int)npages, (uint64_t)idx);
    uint32_t local_slot = (page_idx == 0) ? (uint32_t)idx
        : (uint32_t)(idx - prefix_sum[page_idx - 1]);

    const char *page = type_pages + (uint64_t)page_idx * page_size;
    uint16_t vlen = q16_dev_pagcol_vchar_len(page, local_slot, page_size);
    const char *vdata = q16_dev_pagcol_vchar_data(page, local_slot, page_size);

    // Check NOT LIKE 'MEDIUM POLISHED%'
    if (vlen >= 15) {
        const char mp[] = "MEDIUM POLISHED";
        bool match = true;
        for (int i = 0; i < 15; i++) {
            if (vdata[i] != mp[i]) { match = false; break; }
        }
        if (match) {
            type_ids[idx] = UINT32_MAX;
            return;
        }
    }

    // Hash type string
    uint64_t h = q16_dev_fnv1a64(vdata, vlen);

    // Probe/insert into dictionary
    uint32_t slot = (uint32_t)h & Q16_TYPE_DICT_MASK;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&dict_keys[slot]),
            (unsigned long long)UINT64_MAX,
            (unsigned long long)h);

        if (prev == UINT64_MAX) {
            // We inserted — assign new type_id
            uint32_t tid = atomicAdd(type_id_counter, 1);
            // Store string for reverse lookup
            char *dst = dict_strs + (uint64_t)slot * Q16_TYPE_MAX_LEN;
            for (uint16_t i = 0; i < vlen; i++) dst[i] = vdata[i];
            dict_lens[slot] = vlen;
            __threadfence();  // ensure string is visible before type_id
            dict_type_ids[slot] = tid;
            type_ids[idx] = tid;
            return;
        }
        if (prev == h) {
            // Key exists — wait for type_id to be written
            uint32_t tid;
            do {
                __threadfence();
                tid = dict_type_ids[slot];
            } while (tid == UINT32_MAX);
            type_ids[idx] = tid;
            return;
        }
        slot = (slot + 1) & Q16_TYPE_DICT_MASK;
    }
}

// ============================================================
// Kernel: Cast uint64_t → uint32_t
// ============================================================
__global__ void q16_cast_u64_to_u32_kernel(
    const uint64_t *__restrict__ in,
    uint32_t *__restrict__ out,
    uint64_t n)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = (uint32_t)in[idx];
}

// ============================================================
// Host wrappers for new kernels
// ============================================================

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
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_supplier + BLOCK - 1) / BLOCK);
    q16_supplier_scan_kernel<<<grid, BLOCK, 0, stream>>>(
        d_s_comment_pages, d_prefix_sum, npages, page_size,
        d_s_suppkey_flat, nrecs_supplier,
        d_patterns, d_kmp_next, d_pattern_offsets, d_pattern_lengths,
        num_patterns, d_excl_suppkeys, d_excl_count);
    return cudaGetLastError();
}

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
    cudaStream_t stream)
{
    if (nrecs_batch == 0) return cudaSuccess;
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_batch + BLOCK - 1) / BLOCK);
    q16_supplier_scan_batch_kernel<<<grid, BLOCK, 0, stream>>>(
        d_batch_pages, d_full_prefix_sum, total_npages, batch_start_page, page_size,
        d_s_suppkey_flat, nrecs_batch, row_base,
        d_patterns, d_kmp_next, d_pattern_offsets, d_pattern_lengths,
        num_patterns, d_excl_suppkeys, d_excl_count);
    return cudaGetLastError();
}

cudaError_t q16_build_excl_ht(
    const uint64_t *d_excl_suppkeys, uint32_t excl_count,
    uint64_t *d_excl_ht_keys, uint32_t excl_ht_mask,
    cudaStream_t stream)
{
    if (excl_count == 0) return cudaSuccess;
    constexpr int BLOCK = 256;
    int grid = (int)((excl_count + BLOCK - 1) / BLOCK);
    q16_build_excl_ht_kernel<<<grid, BLOCK, 0, stream>>>(
        d_excl_suppkeys, excl_count, d_excl_ht_keys, excl_ht_mask);
    return cudaGetLastError();
}

cudaError_t q16_extract_brand_ids(
    const char *d_brand_pages,
    const uint64_t *d_prefix_sum,
    uint32_t npages, uint32_t page_size,
    uint32_t padded_len, uint64_t nrecs,
    uint32_t *d_brand_ids, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs + BLOCK - 1) / BLOCK);
    q16_extract_brand_ids_kernel<<<grid, BLOCK, 0, stream>>>(
        d_brand_pages, d_prefix_sum, npages, page_size,
        padded_len, nrecs, d_brand_ids);
    return cudaGetLastError();
}

cudaError_t q16_extract_type_ids(
    const char *d_type_pages,
    const uint64_t *d_prefix_sum,
    uint32_t npages, uint32_t page_size,
    uint64_t nrecs,
    uint64_t *d_dict_keys, uint32_t *d_dict_type_ids,
    char *d_dict_strs, uint16_t *d_dict_lens,
    uint32_t *d_type_id_counter,
    uint32_t *d_type_ids, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs + BLOCK - 1) / BLOCK);
    q16_extract_type_ids_kernel<<<grid, BLOCK, 0, stream>>>(
        d_type_pages, d_prefix_sum, npages, page_size, nrecs,
        d_dict_keys, d_dict_type_ids, d_dict_strs, d_dict_lens,
        d_type_id_counter, d_type_ids);
    return cudaGetLastError();
}

cudaError_t q16_cast_u64_to_u32(
    const uint64_t *d_in, uint32_t *d_out,
    uint64_t n, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((n + BLOCK - 1) / BLOCK);
    q16_cast_u64_to_u32_kernel<<<grid, BLOCK, 0, stream>>>(d_in, d_out, n);
    return cudaGetLastError();
}

// ============================================================
// PiG Q16 pipeline: pre-allocated buffers (no malloc/free)
// ============================================================

size_t q16_pipeline_cub_temp_size(uint64_t nrecs_partsupp)
{
    size_t max_temp = 0;
    size_t temp;

    // DeviceSelect::If u64 (compact non-sentinel entries)
    temp = 0;
    cub::DeviceSelect::If(nullptr, temp,
        (uint64_t*)nullptr, (uint64_t*)nullptr, (uint64_t*)nullptr,
        (int)nrecs_partsupp, Q16NotSentinel());
    if (temp > max_temp) max_temp = temp;

    // Sort u64 keys (nrecs_partsupp)
    cub::DoubleBuffer<uint64_t> db64(nullptr, nullptr);
    temp = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp, db64,
        (int)nrecs_partsupp, 0, 64);
    if (temp > max_temp) max_temp = temp;

    // RLE u64 (nrecs_partsupp)
    temp = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp,
        (uint64_t*)nullptr, (uint64_t*)nullptr, (uint32_t*)nullptr,
        (uint64_t*)nullptr, (int)nrecs_partsupp);
    if (temp > max_temp) max_temp = temp;

    // Sort u32 keys (nrecs_partsupp — upper bound for h_num_unique)
    cub::DoubleBuffer<uint32_t> db32(nullptr, nullptr);
    temp = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp, db32,
        (int)nrecs_partsupp, 0, 32);
    if (temp > max_temp) max_temp = temp;

    // RLE u32 (nrecs_partsupp — upper bound)
    temp = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp,
        (uint32_t*)nullptr, (uint32_t*)nullptr, (uint32_t*)nullptr,
        (uint64_t*)nullptr, (int)nrecs_partsupp);
    if (temp > max_temp) max_temp = temp;

    return max_temp;
}

// ============================================================
// Post-probe pipeline: sort → RLE → extract → sort → RLE
// Assumes d_emit_pairs already populated (by fused kernel).
// ============================================================
cudaError_t q16_post_probe_pipeline(
    const Q16PipelineBuffers &bufs,
    uint64_t nrecs_partsupp,
    std::vector<std::pair<uint32_t, uint32_t>> &result,
    cudaStream_t stream)
{
    // ── Step 1.5: Compact — remove UINT64_MAX sentinels ──
    // d_emit_pairs → d_sort_alt (compacted qualifying pairs only)
    size_t select_temp = bufs.cub_temp_bytes;
    cub::DeviceSelect::If(bufs.d_cub_temp, select_temp,
        bufs.d_emit_pairs, bufs.d_sort_alt, bufs.d_num_unique_ptr,
        (int)nrecs_partsupp, Q16NotSentinel(), stream);

    uint64_t h_num_compacted = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_compacted, bufs.d_num_unique_ptr,
        sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Step 2: Sort compacted pairs (much smaller than nrecs_partsupp) ──
    cub::DoubleBuffer<uint64_t> db_pairs(bufs.d_sort_alt, bufs.d_emit_pairs);

    size_t sort_temp = bufs.cub_temp_bytes;
    cub::DeviceRadixSort::SortKeys(bufs.d_cub_temp, sort_temp, db_pairs,
        (int)h_num_compacted, 0, 64, stream);

    // ── Step 3: RLE → unique (group_id, ps_suppkey) pairs ──
    size_t rle_temp = bufs.cub_temp_bytes;
    cub::DeviceRunLengthEncode::Encode(bufs.d_cub_temp, rle_temp,
        db_pairs.Current(), bufs.d_unique_keys, bufs.d_unique_counts,
        bufs.d_num_unique_ptr, (int)h_num_compacted, stream);

    uint64_t h_num_unique = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_unique, bufs.d_num_unique_ptr,
        sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("[Q16] Unique (group_id, ps_suppkey) pairs: %lu\n", h_num_unique);

    // ── Step 4: Extract group_ids from unique keys ──
    {
        constexpr int BLOCK = 256;
        int grid = (int)((h_num_unique + BLOCK - 1) / BLOCK);
        q16_extract_group_ids_kernel<<<grid, BLOCK, 0, stream>>>(
            bufs.d_unique_keys, bufs.d_group_ids, h_num_unique);
    }

    // ── Step 5: Sort group_ids ──
    cub::DoubleBuffer<uint32_t> db_gids(bufs.d_group_ids, bufs.d_group_ids_alt);

    size_t sort2_temp = bufs.cub_temp_bytes;
    cub::DeviceRadixSort::SortKeys(bufs.d_cub_temp, sort2_temp, db_gids,
        (int)h_num_unique, 0, 32, stream);

    // ── Step 6: RLE on group_ids → (group_id, supplier_cnt) ──
    size_t rle2_temp = bufs.cub_temp_bytes;
    cub::DeviceRunLengthEncode::Encode(bufs.d_cub_temp, rle2_temp,
        db_gids.Current(), bufs.d_result_gids, bufs.d_result_counts,
        bufs.d_num_groups_ptr, (int)h_num_unique, stream);

    uint64_t h_num_groups = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_groups, bufs.d_num_groups_ptr,
        sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("[Q16] Distinct groups: %lu\n", h_num_groups);

    // ── Step 7: Copy results to host (use pre-allocated buffers) ──
    assert(bufs.h_gids != nullptr && bufs.h_counts != nullptr && bufs.h_result_capacity >= h_num_groups);
    CUDA_CHECK(cudaMemcpy(bufs.h_gids, bufs.d_result_gids,
        h_num_groups * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bufs.h_counts, bufs.d_result_counts,
        h_num_groups * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    result.clear();
    result.reserve(h_num_groups);
    for (uint64_t i = 0; i < h_num_groups; i++) {
        result.emplace_back(bufs.h_gids[i], bufs.h_counts[i]);
    }

    return cudaSuccess;
}
