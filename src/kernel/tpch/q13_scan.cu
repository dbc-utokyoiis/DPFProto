// Q13 GOLAP kernels: KMP scan, flatten, probe, pack, CUB pipeline
// Ported from notes/LbC/src/tpch_q13/kernel/scan.cu (tpch_q13_v4)
// Adapted for column-store page layout with slot tables.

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <utility>

#include "q13.cuh"

// ============================================================
// PTX helpers for cp.async prefetch
// ============================================================
template <int Bytes>
__device__ __forceinline__ void q13_cp_async_ca(void *dst_smem, const void *src_gmem) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp.async supports 4/8/16 B only");
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
        "r"(smem_addr), "l"(src_gmem), "n"(Bytes));
}

__device__ __forceinline__ void q13_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int NGroup>
__device__ __forceinline__ void q13_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NGroup) : "memory");
}

__device__ __forceinline__ void q13_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

static __device__ __host__ constexpr int alignTo4(int n) {
    return (n + 3) / 4 * 4;
}

// ============================================================
// VCHAR page access (column-store layout)
// ============================================================
// pag_head: [uint32_t nalloc][uint32_t watermark][uint32_t lfreespace] = 12 bytes
// Slot table grows from page end: page + page_size - sizeof(uint32_t) * (slotid + 1)
// VCHAR record: [uint16_t len][uint16_t pad][data aligned to 4B]

__device__ __forceinline__ uint32_t dev_pag_get_nalloc(const char *page) {
    return *reinterpret_cast<const uint32_t *>(page);
}

__device__ __forceinline__ uint32_t dev_pag_get_oslt(
    const char *page, uint32_t slotid, uint32_t page_size) {
    return *reinterpret_cast<const uint32_t *>(
        page + page_size - sizeof(uint32_t) * (slotid + 1));
}

__device__ __forceinline__ uint16_t dev_pagcol_vchar_len(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = dev_pag_get_oslt(page, slotid, page_size);
    return *reinterpret_cast<const uint16_t *>(page + oslt);
}

__device__ __forceinline__ const char *dev_pagcol_vchar_data(
    const char *page, uint32_t slotid, uint32_t page_size) {
    uint32_t oslt = dev_pag_get_oslt(page, slotid, page_size);
    return page + oslt + sizeof(uint32_t);  // skip [len u16 + pad u16]
}

// ============================================================
// Binary search (upper_bound) for prefix_sum mapping
// ============================================================
template <typename T>
__device__ __forceinline__ int upper_bound_device(
    const T *__restrict__ data, int n, const T &val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (data[mid] <= val)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ============================================================
// Binary search for probe kernel (returns value or 0)
// ============================================================
__device__ __forceinline__ uint64_t device_binary_search(
    const uint64_t *keys, const uint32_t *values,
    int num_items, uint64_t target_key) {
    int left = 0, right = num_items - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        uint64_t mid_key = keys[mid];
        if (mid_key == target_key)
            return values[mid];
        else if (mid_key < target_key)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return 0;  // LEFT OUTER JOIN: 0 for non-matches
}

// ============================================================
// KMP multi-pattern matching with cp.async prefetch
// ============================================================
static constexpr int KMP_PREFETCH_BYTES = 4;

__device__ void kmp_match_prefetch(
    bool do_match,
    const char *__restrict__ string_base,
    int string_len,
    const char *__restrict__ patterns_global,
    const int *__restrict__ next_global,
    const int *__restrict__ pattern_offsets,
    const int *__restrict__ pattern_lengths,
    int num_patterns, int total_pattern_chars,
    bool *__restrict__ result,
    char *smem_buf1, char *smem_buf2,
    const char *smem_pattern_base)
{
    string_len = alignTo4(string_len);
    int pat_size_aligned = alignTo4(total_pattern_chars);

    char *pat_shared = (char *)smem_pattern_base;
    int *next_shared = (int *)(pat_shared + pat_size_aligned);
    int *offsets_shared = (int *)(next_shared + total_pattern_chars);
    int *lens_shared = (int *)(offsets_shared + num_patterns);

    int tid = threadIdx.x;

    for (int i = tid; i < total_pattern_chars; i += blockDim.x)
        pat_shared[i] = patterns_global[i];
    for (int i = tid; i < total_pattern_chars; i += blockDim.x)
        next_shared[i] = next_global[i];
    for (int i = tid; i < num_patterns; i += blockDim.x) {
        offsets_shared[i] = pattern_offsets[i];
        lens_shared[i] = pattern_lengths[i];
    }
    __syncthreads();

    if (!do_match) return;

    constexpr int PIVOT = KMP_PREFETCH_BYTES;
    int num_tiles = (string_len + PIVOT - 1) / PIVOT;

    int current_pat_idx = 0;
    int l = 0;

    int p_offset = offsets_shared[current_pat_idx];
    int p_len = lens_shared[current_pat_idx];
    const char *current_pat_ptr = pat_shared + p_offset;
    const int *current_next_ptr = next_shared + p_offset;

    char *buf = (char *)string_base;

    // Prefetch first tile
    q13_cp_async_ca<KMP_PREFETCH_BYTES>(smem_buf1, &buf[0]);
    q13_cp_async_commit();
    char *rbuf = smem_buf1;
    int bufidx = 1;

    for (int j = 1; j < num_tiles; j++) {
        if (bufidx == 0) {
            q13_cp_async_ca<KMP_PREFETCH_BYTES>(smem_buf1, &buf[j * PIVOT]);
            rbuf = smem_buf2;
        } else {
            q13_cp_async_ca<KMP_PREFETCH_BYTES>(smem_buf2, &buf[j * PIVOT]);
            rbuf = smem_buf1;
        }
        bufidx = (bufidx + 1) & 1;
        q13_cp_async_commit();
        q13_cp_async_wait_group<1>();

        for (int k = 0; k < PIVOT; ++k) {
            if (current_pat_idx >= num_patterns) break;
            char c = rbuf[k];
            bool match = (current_pat_ptr[l] == c);
            bool at_zero = (l == 0);
            if (match || at_zero) {
                if (match) l++;
                if (l == p_len) {
                    current_pat_idx++;
                    l = 0;
                    if (current_pat_idx < num_patterns) {
                        p_offset = offsets_shared[current_pat_idx];
                        p_len = lens_shared[current_pat_idx];
                        current_pat_ptr = pat_shared + p_offset;
                        current_next_ptr = next_shared + p_offset;
                    } else break;
                }
            } else {
                l = current_next_ptr[l - 1];
                k--;
            }
        }
        if (current_pat_idx >= num_patterns) break;
    }

    // Process final tile
    rbuf = (bufidx == 0) ? smem_buf2 : smem_buf1;
    q13_cp_async_wait_all();
    for (int k = 0; k < PIVOT; ++k) {
        if (current_pat_idx >= num_patterns) break;
        char c = rbuf[k];
        bool match = (current_pat_ptr[l] == c);
        bool at_zero = (l == 0);
        if (match || at_zero) {
            if (match) l++;
            if (l == p_len) {
                current_pat_idx++;
                l = 0;
                if (current_pat_idx < num_patterns) {
                    p_offset = offsets_shared[current_pat_idx];
                    p_len = lens_shared[current_pat_idx];
                    current_pat_ptr = pat_shared + p_offset;
                    current_next_ptr = next_shared + p_offset;
                } else break;
            }
        } else {
            l = current_next_ptr[l - 1];
            k--;
        }
    }

    *result = (current_pat_idx >= num_patterns);
}

// ============================================================
// Q13 scan kernel
// ============================================================
static constexpr int Q13_BLOCK_SIZE = 128;

template <bool USE_PREFIX_SUM>
__global__ void q13_scan_kernel(
    const char *__restrict__ o_comment_pages,
    const uint64_t *__restrict__ prefix_sum,   // only when USE_PREFIX_SUM=true
    uint32_t npages,
    uint32_t page_size,
    uint32_t max_capacity,                     // only when USE_PREFIX_SUM=false
    uint64_t nrecs_total,
    // KMP tables
    const char *__restrict__ patterns,
    const int *__restrict__ next,
    const int *__restrict__ pattern_offsets,
    const int *__restrict__ pattern_lengths,
    int num_patterns, int total_pattern_chars,
    // O_CUSTKEY access
    const uint64_t *__restrict__ o_custkey_flat,  // USE_PREFIX_SUM=true
    const char *__restrict__ o_custkey_pages,      // USE_PREFIX_SUM=false
    uint32_t o_custkey_page_size,                  // USE_PREFIX_SUM=false
    uint32_t o_custkey_capacity,                   // USE_PREFIX_SUM=false
    // Output
    uint64_t *__restrict__ o_aggr_custkey,
    uint64_t *__restrict__ count)
{
    extern __shared__ __align__(16) char smem[];

    uint64_t global_rec_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    char *smem_base = smem + warp_id * (2 * 32 * KMP_PREFETCH_BYTES);
    char *smem_buf1 = smem_base + lane * KMP_PREFETCH_BYTES;
    char *smem_buf2 = smem_base + 32 * KMP_PREFETCH_BYTES + lane * KMP_PREFETCH_BYTES;
    char *smem_pattern_base = smem + 2 * blockDim.x * KMP_PREFETCH_BYTES;

    uint32_t page_idx, local_rec_idx;
    bool do_match = true;

    if constexpr (USE_PREFIX_SUM) {
        if (global_rec_id >= nrecs_total) {
            do_match = false;
        } else {
            page_idx = upper_bound_device(prefix_sum, (int)npages, global_rec_id);
            local_rec_idx = (page_idx == 0) ? (uint32_t)global_rec_id
                            : (uint32_t)(global_rec_id - prefix_sum[page_idx - 1]);
        }
    } else {
        page_idx = (uint32_t)(global_rec_id / max_capacity);
        local_rec_idx = (uint32_t)(global_rec_id % max_capacity);
        if (page_idx >= npages) {
            do_match = false;
        } else {
            uint32_t nalloc = dev_pag_get_nalloc(
                o_comment_pages + (uint64_t)page_idx * page_size);
            if (local_rec_idx >= nalloc)
                do_match = false;
        }
    }

    // Get VCHAR data for this record
    const char *vchar_data = nullptr;
    int vchar_len = 0;
    if (do_match) {
        const char *page = o_comment_pages + (uint64_t)page_idx * page_size;
        vchar_len = (int)dev_pagcol_vchar_len(page, local_rec_idx, page_size);
        vchar_data = dev_pagcol_vchar_data(page, local_rec_idx, page_size);
    }

    // KMP pattern matching
    bool matched = false;
    kmp_match_prefetch(
        do_match, vchar_data, vchar_len,
        patterns, next, pattern_offsets, pattern_lengths,
        num_patterns, total_pattern_chars,
        &matched, smem_buf1, smem_buf2, smem_pattern_base);

    // Compute actual row ID (contiguous across pages)
    uint64_t actual_row_id;
    if constexpr (USE_PREFIX_SUM) {
        actual_row_id = global_rec_id;
    } else {
        // In non-prefix_sum mode, global_rec_id is in slot-space (page*max_capacity+slot).
        // Compute actual row ID from O_COMMENT prefix_sum (always available).
        if (do_match) {
            actual_row_id = (page_idx == 0) ? local_rec_idx
                            : prefix_sum[page_idx - 1] + local_rec_idx;
        } else {
            actual_row_id = UINT64_MAX;  // sentinel, won't be used
        }
    }

    // Write output: NOT LIKE → write custkey, LIKE → write UINT64_MAX
    // O_CUSTKEY is always flattened to o_custkey_flat (both modes).
    if (do_match) {
        uint64_t custkey_val = o_custkey_flat[actual_row_id];
        o_aggr_custkey[actual_row_id] = matched ? UINT64_MAX : custkey_val;
    } else {
        if constexpr (USE_PREFIX_SUM) {
            if (global_rec_id < nrecs_total) {
                o_aggr_custkey[global_rec_id] = UINT64_MAX;
            }
        }
        // In non-prefix_sum mode, empty slots don't correspond to real rows.
        // o_aggr_custkey is pre-filled with UINT64_MAX via cudaMemset.
    }

    // Count non-matched (qualifying) records via block reduction
    using BlockReduce = cub::BlockReduce<uint64_t, Q13_BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    uint64_t qualifying = (do_match && !matched) ? 1 : 0;
    uint64_t aggregate = BlockReduce(temp_storage).Sum(qualifying);
    if (threadIdx.x == 0 && aggregate > 0) {
        atomicAdd(reinterpret_cast<unsigned long long *>(count),
                  static_cast<unsigned long long>(aggregate));
    }
}

// ============================================================
// Flatten kernels
// ============================================================
static constexpr int FLATTEN_BLOCK = 256;

__global__ void flatten_int32_pages_kernel(
    const char *__restrict__ pages,
    uint32_t page_size,
    uint32_t capacity,
    uint64_t nrecs_total,
    uint64_t *__restrict__ out)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_total) return;

    uint64_t page_idx = idx / capacity;
    uint32_t rec_idx = idx % capacity;

    const char *page = pages + page_idx * page_size;
    uint32_t nalloc = *reinterpret_cast<const uint32_t *>(page);
    if (rec_idx >= nalloc) {
        out[idx] = 0;
        return;
    }

    const int32_t *values = reinterpret_cast<const int32_t *>(page + 12);
    out[idx] = (uint64_t)(uint32_t)values[rec_idx];
}

__global__ void flatten_int64_pages_kernel(
    const char *__restrict__ pages,
    uint32_t page_size,
    uint32_t capacity,
    uint64_t nrecs_total,
    uint64_t *__restrict__ out)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_total) return;

    uint64_t page_idx = idx / capacity;
    uint32_t rec_idx = idx % capacity;

    const char *page = pages + page_idx * page_size;
    uint32_t nalloc = *reinterpret_cast<const uint32_t *>(page);
    if (rec_idx >= nalloc) {
        out[idx] = 0;
        return;
    }

    // int64_t: pag_head(12B) + 4B padding → data starts at offset 16 (8-byte aligned)
    const int64_t *values = reinterpret_cast<const int64_t *>(page + 16);
    out[idx] = (uint64_t)values[rec_idx];
}

// ============================================================
// Flatten kernels (prefix_sum-based, contiguous output)
// ============================================================
__global__ void flatten_int32_pages_ps_kernel(
    const char *__restrict__ pages,
    uint32_t page_size,
    const uint64_t *__restrict__ prefix_sum,
    uint32_t npages,
    uint64_t nrecs_total,
    uint64_t *__restrict__ out)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_total) return;

    uint32_t page_idx = upper_bound_device(prefix_sum, npages, idx);
    uint32_t local_idx = (page_idx == 0) ? (uint32_t)idx
                                         : (uint32_t)(idx - prefix_sum[page_idx - 1]);

    const char *page = pages + (uint64_t)page_idx * page_size;
    const int32_t *values = reinterpret_cast<const int32_t *>(page + 12);
    out[idx] = (uint64_t)(uint32_t)values[local_idx];
}

__global__ void flatten_int64_pages_ps_kernel(
    const char *__restrict__ pages,
    uint32_t page_size,
    const uint64_t *__restrict__ prefix_sum,
    uint32_t npages,
    uint64_t nrecs_total,
    uint64_t *__restrict__ out)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_total) return;

    uint32_t page_idx = upper_bound_device(prefix_sum, npages, idx);
    uint32_t local_idx = (page_idx == 0) ? (uint32_t)idx
                                         : (uint32_t)(idx - prefix_sum[page_idx - 1]);

    const char *page = pages + (uint64_t)page_idx * page_size;
    // int64_t: pag_head(12B) + 4B padding → data starts at offset 16 (8-byte aligned)
    const int64_t *values = reinterpret_cast<const int64_t *>(page + 16);
    out[idx] = (uint64_t)values[local_idx];
}

// ============================================================
// Masked flatten kernel (prefix_sum-based, fill_value for inactive pages)
// ============================================================
__global__ void flatten_int32_pages_ps_masked_kernel(
    const char *__restrict__ pages,
    uint32_t page_size,
    const uint64_t *__restrict__ prefix_sum,
    uint32_t npages,
    uint64_t nrecs_total,
    const uint8_t *__restrict__ page_active,  // nullptr = all active
    uint64_t fill_value,
    uint64_t *__restrict__ out)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_total) return;

    uint32_t page_idx = upper_bound_device(prefix_sum, (int)npages, idx);
    if (page_active != nullptr && !page_active[page_idx]) {
        out[idx] = fill_value;
        return;
    }
    uint32_t local_idx = (page_idx == 0) ? (uint32_t)idx
                                         : (uint32_t)(idx - prefix_sum[page_idx - 1]);

    const char *page = pages + (uint64_t)page_idx * page_size;
    const int32_t *values = reinterpret_cast<const int32_t *>(page + 12);
    out[idx] = (uint64_t)(uint32_t)values[local_idx];
}

// ============================================================
// Probe kernel (LEFT OUTER JOIN via binary search)
// ============================================================
__global__ void probe_customer_orders_kernel(
    const uint64_t *__restrict__ d_c_custkey,
    uint64_t num_customers,
    const uint64_t *__restrict__ d_o_rle_keys,
    const uint32_t *__restrict__ d_o_rle_counts,
    const uint64_t *__restrict__ d_num_rle_items,
    uint32_t *__restrict__ d_results)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_customers) return;

    uint64_t my_custkey = d_c_custkey[idx];
    int num_rle = (int)*d_num_rle_items;
    uint64_t my_count = device_binary_search(
        d_o_rle_keys, d_o_rle_counts, num_rle, my_custkey);
    d_results[idx] = (uint32_t)my_count;
}

// ============================================================
// Pack kernel for SortPairsDescending
// ============================================================
__global__ void pack_keys_kernel(
    const uint32_t *__restrict__ custdist,
    const uint32_t *__restrict__ c_count,
    uint64_t *__restrict__ composite_keys,
    uint64_t *__restrict__ values,
    uint64_t num_items)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    uint64_t upper = (uint64_t)custdist[idx];
    uint64_t lower = (uint64_t)c_count[idx];
    composite_keys[idx] = (upper << 32) | lower;
    values[idx] = idx;
}

// ============================================================
// Host wrapper: flatten
// ============================================================
cudaError_t q13_flatten_int32_pages(
    const char *pages, uint32_t page_size, uint32_t capacity,
    uint64_t nrecs_total, uint64_t *out, cudaStream_t stream)
{
    int grid = (int)((nrecs_total + FLATTEN_BLOCK - 1) / FLATTEN_BLOCK);
    flatten_int32_pages_kernel<<<grid, FLATTEN_BLOCK, 0, stream>>>(
        pages, page_size, capacity, nrecs_total, out);
    return cudaGetLastError();
}

cudaError_t q13_flatten_int64_pages(
    const char *pages, uint32_t page_size, uint32_t capacity,
    uint64_t nrecs_total, uint64_t *out, cudaStream_t stream)
{
    int grid = (int)((nrecs_total + FLATTEN_BLOCK - 1) / FLATTEN_BLOCK);
    flatten_int64_pages_kernel<<<grid, FLATTEN_BLOCK, 0, stream>>>(
        pages, page_size, capacity, nrecs_total, out);
    return cudaGetLastError();
}

cudaError_t q13_flatten_int32_pages_ps(
    const char *pages, uint32_t page_size,
    const uint64_t *prefix_sum, uint32_t npages,
    uint64_t nrecs_total, uint64_t *out, cudaStream_t stream)
{
    int grid = (int)((nrecs_total + FLATTEN_BLOCK - 1) / FLATTEN_BLOCK);
    flatten_int32_pages_ps_kernel<<<grid, FLATTEN_BLOCK, 0, stream>>>(
        pages, page_size, prefix_sum, npages, nrecs_total, out);
    return cudaGetLastError();
}

cudaError_t q13_flatten_int64_pages_ps(
    const char *pages, uint32_t page_size,
    const uint64_t *prefix_sum, uint32_t npages,
    uint64_t nrecs_total, uint64_t *out, cudaStream_t stream)
{
    int grid = (int)((nrecs_total + FLATTEN_BLOCK - 1) / FLATTEN_BLOCK);
    flatten_int64_pages_ps_kernel<<<grid, FLATTEN_BLOCK, 0, stream>>>(
        pages, page_size, prefix_sum, npages, nrecs_total, out);
    return cudaGetLastError();
}

cudaError_t q13_flatten_int32_pages_ps_masked(
    const char *pages, uint32_t page_size,
    const uint64_t *prefix_sum, uint32_t npages,
    uint64_t nrecs_total,
    const uint8_t *page_active, uint64_t fill_value,
    uint64_t *out, cudaStream_t stream)
{
    int grid = (int)((nrecs_total + FLATTEN_BLOCK - 1) / FLATTEN_BLOCK);
    flatten_int32_pages_ps_masked_kernel<<<grid, FLATTEN_BLOCK, 0, stream>>>(
        pages, page_size, prefix_sum, npages, nrecs_total,
        page_active, fill_value, out);
    return cudaGetLastError();
}

// ============================================================
// Host wrapper: Q13 GOLAP pipeline
// ============================================================
cudaError_t q13_golap(
    const char *o_comment_pages,
    uint32_t o_comment_npages,
    const char *o_custkey_pages,
    uint32_t o_custkey_capacity,
    const uint64_t *d_prefix_sum,
    const uint64_t *d_o_custkey_flat,
    uint32_t page_size,
    uint32_t max_capacity_vchar,
    uint64_t nrecs_orders,
    uint64_t nrecs_customer,
    const uint64_t *d_c_custkey,
    bool use_prefix_sum,
    const char *d_patterns,
    const int *d_next,
    const int *d_pattern_offsets,
    const int *d_pattern_lengths,
    int num_patterns,
    int total_pattern_chars,
    std::vector<std::pair<uint32_t, uint32_t>> &result,
    cudaStream_t stream)
{
    cudaError_t err;

    // ── Phase 1: Scan ORDERS with KMP ──
    uint64_t *d_o_aggr_custkey = nullptr;
    uint64_t *d_count = nullptr;
    cudaMalloc(&d_o_aggr_custkey, nrecs_orders * sizeof(uint64_t));
    cudaMalloc(&d_count, sizeof(uint64_t));
    cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream);

    // Grid size depends on prefix_sum mode
    uint64_t total_threads;
    if (use_prefix_sum) {
        total_threads = nrecs_orders;
    } else {
        total_threads = (uint64_t)o_comment_npages * max_capacity_vchar;
    }
    int grid_dim = (int)((total_threads + Q13_BLOCK_SIZE - 1) / Q13_BLOCK_SIZE);

    size_t smem_size = 2 * Q13_BLOCK_SIZE * KMP_PREFETCH_BYTES +
                       alignTo4(total_pattern_chars) +
                       sizeof(int) * total_pattern_chars +
                       sizeof(int) * num_patterns +
                       sizeof(int) * num_patterns;

    // O_CUSTKEY page size may differ from O_COMMENT page size (same in practice)
    uint32_t o_custkey_page_size = page_size;

    if (use_prefix_sum) {
        q13_scan_kernel<true><<<grid_dim, Q13_BLOCK_SIZE, smem_size, stream>>>(
            o_comment_pages, d_prefix_sum, o_comment_npages, page_size,
            0 /* unused */, nrecs_orders,
            d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
            num_patterns, total_pattern_chars,
            d_o_custkey_flat, nullptr, 0, 0,
            d_o_aggr_custkey, d_count);
    } else {
        // Pre-fill with UINT64_MAX: empty slots (not real rows) won't be written by the kernel.
        // After sort, UINT64_MAX entries go to end and are excluded by RLE.
        cudaMemsetAsync(d_o_aggr_custkey, 0xFF, nrecs_orders * sizeof(uint64_t), stream);

        // Pass d_prefix_sum (O_COMMENT prefix_sum, built at query time in golap.cu)
        // so that the kernel can compute actual_row_id for cross-column O_CUSTKEY access.
        // d_o_custkey_flat is always available (flattened in both modes).
        q13_scan_kernel<false><<<grid_dim, Q13_BLOCK_SIZE, smem_size, stream>>>(
            o_comment_pages, d_prefix_sum, o_comment_npages, page_size,
            max_capacity_vchar, nrecs_orders,
            d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
            num_patterns, total_pattern_chars,
            d_o_custkey_flat, o_custkey_pages, o_custkey_page_size, o_custkey_capacity,
            d_o_aggr_custkey, d_count);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "q13_scan_kernel launch failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    uint64_t h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("[Q13] Qualifying orders (NOT LIKE): %lu / %lu\n", h_count, nrecs_orders);

    // ── Phase 2: Sort o_aggr_custkey ──
    // All nrecs_orders entries: qualifying have real custkeys, non-qualifying have UINT64_MAX
    uint64_t *d_sort_alt = nullptr;
    cudaMalloc(&d_sort_alt, nrecs_orders * sizeof(uint64_t));
    cub::DoubleBuffer<uint64_t> db_sort_keys(d_o_aggr_custkey, d_sort_alt);

    void *d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_temp_bytes, db_sort_keys,
                                    (int)nrecs_orders, 0, 64, stream);
    cudaMalloc(&d_sort_temp, sort_temp_bytes);
    cub::DeviceRadixSort::SortKeys(d_sort_temp, sort_temp_bytes, db_sort_keys,
                                    (int)nrecs_orders, 0, 64, stream);

    // ── Phase 3: RLE on sorted custkeys ──
    uint64_t *d_rle_keys = nullptr;
    uint32_t *d_rle_counts = nullptr;
    uint64_t *d_num_rle = nullptr;
    cudaMalloc(&d_rle_keys, nrecs_orders * sizeof(uint64_t));
    cudaMalloc(&d_rle_counts, nrecs_orders * sizeof(uint32_t));
    cudaMalloc(&d_num_rle, sizeof(uint64_t));

    void *d_rle_temp = nullptr;
    size_t rle_temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, rle_temp_bytes,
        db_sort_keys.Current(), d_rle_keys, d_rle_counts,
        d_num_rle, (int)nrecs_orders, stream);
    cudaMalloc(&d_rle_temp, rle_temp_bytes);
    cub::DeviceRunLengthEncode::Encode(d_rle_temp, rle_temp_bytes,
        db_sort_keys.Current(), d_rle_keys, d_rle_counts,
        d_num_rle, (int)nrecs_orders, stream);

    uint64_t h_num_rle = 0;
    cudaMemcpyAsync(&h_num_rle, d_num_rle, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("[Q13] Unique custkeys after RLE: %lu\n", h_num_rle);

    // ── Phase 4: Probe CUSTOMER → c_count[] ──
    uint32_t *d_c_count = nullptr;
    cudaMalloc(&d_c_count, nrecs_customer * sizeof(uint32_t));

    {
        int probe_grid = (int)((nrecs_customer + 128 - 1) / 128);
        probe_customer_orders_kernel<<<probe_grid, 128, 0, stream>>>(
            d_c_custkey, nrecs_customer,
            d_rle_keys, d_rle_counts, d_num_rle,
            d_c_count);
    }

    // ── Phase 5: Sort c_count ──
    uint32_t *d_c_count_alt = nullptr;
    cudaMalloc(&d_c_count_alt, nrecs_customer * sizeof(uint32_t));
    cub::DoubleBuffer<uint32_t> db_sort2_keys(d_c_count, d_c_count_alt);

    size_t sort2_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort2_temp_bytes, db_sort2_keys,
                                    (int)nrecs_customer, 0, 32, stream);
    // Reuse sort_temp if large enough, else realloc
    if (sort2_temp_bytes > sort_temp_bytes) {
        cudaFree(d_sort_temp);
        cudaMalloc(&d_sort_temp, sort2_temp_bytes);
    }
    cub::DeviceRadixSort::SortKeys(d_sort_temp, sort2_temp_bytes, db_sort2_keys,
                                    (int)nrecs_customer, 0, 32, stream);

    // ── Phase 6: RLE on c_count → (c_count_value, frequency) ──
    uint32_t *d_aggr2_keys = nullptr;
    uint32_t *d_aggr2_counts = nullptr;
    cudaMalloc(&d_aggr2_keys, nrecs_customer * sizeof(uint32_t));
    cudaMalloc(&d_aggr2_counts, nrecs_customer * sizeof(uint32_t));

    size_t rle2_temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, rle2_temp_bytes,
        db_sort2_keys.Current(), d_aggr2_keys, d_aggr2_counts,
        d_num_rle, (int)nrecs_customer, stream);
    if (rle2_temp_bytes > rle_temp_bytes) {
        cudaFree(d_rle_temp);
        cudaMalloc(&d_rle_temp, rle2_temp_bytes);
    }
    cub::DeviceRunLengthEncode::Encode(d_rle_temp, rle2_temp_bytes,
        db_sort2_keys.Current(), d_aggr2_keys, d_aggr2_counts,
        d_num_rle, (int)nrecs_customer, stream);

    uint64_t h_num_dist = 0;
    cudaMemcpyAsync(&h_num_dist, d_num_rle, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("[Q13] Distinct c_count values: %lu\n", h_num_dist);

    // ── Phase 7: Pack + SortPairsDescending ──
    uint64_t *d_composite_keys = nullptr;
    uint64_t *d_composite_keys_alt = nullptr;
    uint64_t *d_composite_vals = nullptr;
    uint64_t *d_composite_vals_alt = nullptr;
    cudaMalloc(&d_composite_keys, h_num_dist * sizeof(uint64_t));
    cudaMalloc(&d_composite_keys_alt, h_num_dist * sizeof(uint64_t));
    cudaMalloc(&d_composite_vals, h_num_dist * sizeof(uint64_t));
    cudaMalloc(&d_composite_vals_alt, h_num_dist * sizeof(uint64_t));

    {
        int pack_grid = (int)((h_num_dist + 256 - 1) / 256);
        pack_keys_kernel<<<pack_grid, 256, 0, stream>>>(
            d_aggr2_counts, d_aggr2_keys,
            d_composite_keys, d_composite_vals, h_num_dist);
    }

    cub::DoubleBuffer<uint64_t> db_sort3_keys(d_composite_keys, d_composite_keys_alt);
    cub::DoubleBuffer<uint64_t> db_sort3_vals(d_composite_vals, d_composite_vals_alt);

    size_t sort3_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, sort3_temp_bytes,
        db_sort3_keys, db_sort3_vals, (int)h_num_dist, 0, 64, stream);
    if (sort3_temp_bytes > sort_temp_bytes && sort3_temp_bytes > sort2_temp_bytes) {
        cudaFree(d_sort_temp);
        cudaMalloc(&d_sort_temp, sort3_temp_bytes);
    }
    cub::DeviceRadixSort::SortPairsDescending(d_sort_temp, sort3_temp_bytes,
        db_sort3_keys, db_sort3_vals, (int)h_num_dist, 0, 64, stream);

    // ── Copy results to host ──
    std::vector<uint64_t> h_composite(h_num_dist);
    cudaMemcpyAsync(h_composite.data(), db_sort3_keys.Current(),
                    h_num_dist * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    result.clear();
    result.reserve(h_num_dist);
    for (uint64_t i = 0; i < h_num_dist; i++) {
        uint32_t custdist_val = (uint32_t)(h_composite[i] >> 32);
        uint32_t c_count_val = (uint32_t)(h_composite[i] & 0xFFFFFFFF);
        result.push_back({c_count_val, custdist_val});
    }

    // ── Cleanup ──
    cudaFree(d_o_aggr_custkey);
    cudaFree(d_count);
    cudaFree(d_sort_alt);
    cudaFree(d_sort_temp);
    cudaFree(d_rle_keys);
    cudaFree(d_rle_counts);
    cudaFree(d_num_rle);
    cudaFree(d_rle_temp);
    cudaFree(d_c_count);
    cudaFree(d_c_count_alt);
    cudaFree(d_aggr2_keys);
    cudaFree(d_aggr2_counts);
    cudaFree(d_composite_keys);
    cudaFree(d_composite_keys_alt);
    cudaFree(d_composite_vals);
    cudaFree(d_composite_vals_alt);

    return cudaSuccess;
}

// ============================================================
// Host wrapper: Q13 scan batch
// Scans a batch of decompressed O_COMMENT pages (prefix_sum mode).
// Writes to caller-provided d_o_aggr_custkey/d_count (batch-offset).
// ============================================================
cudaError_t q13_scan_batch(
    const char *o_comment_pages,
    const uint64_t *d_prefix_sum,
    uint32_t npages,
    uint32_t page_size,
    uint64_t nrecs_batch,
    const char *d_patterns,
    const int *d_next,
    const int *d_pattern_offsets,
    const int *d_pattern_lengths,
    int num_patterns,
    int total_pattern_chars,
    const uint64_t *d_o_custkey_flat,
    uint64_t *d_o_aggr_custkey,
    uint64_t *d_count,
    cudaStream_t stream)
{
    if (nrecs_batch == 0) return cudaSuccess;

    int grid_dim = (int)((nrecs_batch + Q13_BLOCK_SIZE - 1) / Q13_BLOCK_SIZE);
    size_t smem_size = 2 * Q13_BLOCK_SIZE * KMP_PREFETCH_BYTES +
                       alignTo4(total_pattern_chars) +
                       sizeof(int) * total_pattern_chars +
                       sizeof(int) * num_patterns +
                       sizeof(int) * num_patterns;

    q13_scan_kernel<true><<<grid_dim, Q13_BLOCK_SIZE, smem_size, stream>>>(
        o_comment_pages, d_prefix_sum, npages, page_size,
        0 /* unused */, nrecs_batch,
        d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
        num_patterns, total_pattern_chars,
        d_o_custkey_flat, nullptr, 0, 0,
        d_o_aggr_custkey, d_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "q13_scan_batch launch failed: %s\n", cudaGetErrorString(err));
    return err;
}

// ============================================================
// Compute required CUB scratch size for Q13 pipeline.
// ============================================================
size_t q13_pipeline_cub_temp_size(uint64_t nrecs_orders, uint64_t nrecs_customer)
{
    size_t max_temp = 0;
    size_t temp;

    // Sort u64 keys (nrecs_orders)
    cub::DoubleBuffer<uint64_t> db64(nullptr, nullptr);
    temp = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp, db64,
        (int)nrecs_orders, 0, 64);
    if (temp > max_temp) max_temp = temp;

    // RLE u64 (nrecs_orders)
    temp = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp,
        (uint64_t*)nullptr, (uint64_t*)nullptr, (uint32_t*)nullptr,
        (uint64_t*)nullptr, (int)nrecs_orders);
    if (temp > max_temp) max_temp = temp;

    // Sort u32 keys (nrecs_customer)
    cub::DoubleBuffer<uint32_t> db32(nullptr, nullptr);
    temp = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp, db32,
        (int)nrecs_customer, 0, 32);
    if (temp > max_temp) max_temp = temp;

    // RLE u32 (nrecs_customer)
    temp = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp,
        (uint32_t*)nullptr, (uint32_t*)nullptr, (uint32_t*)nullptr,
        (uint64_t*)nullptr, (int)nrecs_customer);
    if (temp > max_temp) max_temp = temp;

    // SortPairsDescending u64 (nrecs_customer — upper bound for h_num_dist)
    cub::DoubleBuffer<uint64_t> db64k(nullptr, nullptr);
    cub::DoubleBuffer<uint64_t> db64v(nullptr, nullptr);
    temp = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, temp,
        db64k, db64v, (int)nrecs_customer, 0, 64);
    if (temp > max_temp) max_temp = temp;

    return max_temp;
}

// ============================================================
// Host wrapper: Q13 aggregation (phases 2-7)
// Takes a pre-populated d_o_aggr_custkey array and runs
// Sort → RLE → Probe → Sort → RLE → Pack to produce final result.
// ============================================================
cudaError_t q13_pig_aggregate(
    const Q13PipelineBuffers &bufs,
    uint64_t *d_o_aggr_custkey,
    uint64_t nrecs_orders,
    const uint64_t *d_c_custkey,
    uint64_t nrecs_customer,
    std::vector<std::pair<uint32_t, uint32_t>> &result,
    cudaStream_t stream)
{
    // ── Phase 2: Sort o_aggr_custkey ──
    cub::DoubleBuffer<uint64_t> db_sort_keys(d_o_aggr_custkey, bufs.d_sort_alt);

    size_t sort_temp = bufs.cub_temp_bytes;
    cub::DeviceRadixSort::SortKeys(bufs.d_cub_temp, sort_temp, db_sort_keys,
                                    (int)nrecs_orders, 0, 64, stream);

    // ── Phase 3: RLE on sorted custkeys ──
    size_t rle_temp = bufs.cub_temp_bytes;
    cub::DeviceRunLengthEncode::Encode(bufs.d_cub_temp, rle_temp,
        db_sort_keys.Current(), bufs.d_rle_keys, bufs.d_rle_counts,
        bufs.d_num_rle, (int)nrecs_orders, stream);

    uint64_t h_num_rle = 0;
    cudaMemcpyAsync(&h_num_rle, bufs.d_num_rle, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("[Q13] Unique custkeys after RLE: %lu\n", h_num_rle);

    // ── Phase 4: Probe CUSTOMER → c_count[] ──
    {
        int probe_grid = (int)((nrecs_customer + 128 - 1) / 128);
        probe_customer_orders_kernel<<<probe_grid, 128, 0, stream>>>(
            d_c_custkey, nrecs_customer,
            bufs.d_rle_keys, bufs.d_rle_counts, bufs.d_num_rle,
            bufs.d_c_count);
    }

    // ── Phase 5: Sort c_count ──
    cub::DoubleBuffer<uint32_t> db_sort2_keys(bufs.d_c_count, bufs.d_c_count_alt);

    size_t sort2_temp = bufs.cub_temp_bytes;
    cub::DeviceRadixSort::SortKeys(bufs.d_cub_temp, sort2_temp, db_sort2_keys,
                                    (int)nrecs_customer, 0, 32, stream);

    // ── Phase 6: RLE on c_count → (c_count_value, frequency) ──
    size_t rle2_temp = bufs.cub_temp_bytes;
    cub::DeviceRunLengthEncode::Encode(bufs.d_cub_temp, rle2_temp,
        db_sort2_keys.Current(), bufs.d_aggr2_keys, bufs.d_aggr2_counts,
        bufs.d_num_rle, (int)nrecs_customer, stream);

    uint64_t h_num_dist = 0;
    cudaMemcpyAsync(&h_num_dist, bufs.d_num_rle, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("[Q13] Distinct c_count values: %lu\n", h_num_dist);

    // ── Phase 7: Pack + SortPairsDescending ──
    {
        int pack_grid = (int)((h_num_dist + 256 - 1) / 256);
        pack_keys_kernel<<<pack_grid, 256, 0, stream>>>(
            bufs.d_aggr2_counts, bufs.d_aggr2_keys,
            bufs.d_composite_keys, bufs.d_composite_vals, h_num_dist);
    }

    cub::DoubleBuffer<uint64_t> db_sort3_keys(bufs.d_composite_keys, bufs.d_composite_keys_alt);
    cub::DoubleBuffer<uint64_t> db_sort3_vals(bufs.d_composite_vals, bufs.d_composite_vals_alt);

    size_t sort3_temp = bufs.cub_temp_bytes;
    cub::DeviceRadixSort::SortPairsDescending(bufs.d_cub_temp, sort3_temp,
        db_sort3_keys, db_sort3_vals, (int)h_num_dist, 0, 64, stream);

    // ── Copy results to host (use pre-allocated buffer) ──
    assert(bufs.h_composite != nullptr && (uint64_t)bufs.h_composite_capacity >= h_num_dist);
    cudaMemcpyAsync(bufs.h_composite, db_sort3_keys.Current(),
                    h_num_dist * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    result.clear();
    result.reserve(h_num_dist);
    for (uint64_t i = 0; i < h_num_dist; i++) {
        uint32_t custdist_val = (uint32_t)(bufs.h_composite[i] >> 32);
        uint32_t c_count_val = (uint32_t)(bufs.h_composite[i] & 0xFFFFFFFF);
        result.push_back({c_count_val, custdist_val});
    }

    return cudaSuccess;
}
