// Q3 GOLAP kernels: CUSTOMER hash set build, ORDERS probe + HT build,
// LINEITEM probe + revenue aggregation, result collection.
//
// TPC-H Q3: Shipping Priority
// SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue,
//        o_orderdate, o_shippriority
// FROM   customer, orders, lineitem
// WHERE  c_mktsegment = 'BUILDING'
//   AND  c_custkey = o_custkey
//   AND  l_orderkey = o_orderkey
//   AND  o_orderdate < DATE '1995-03-15'
//   AND  l_shipdate  > DATE '1995-03-15'
// GROUP BY l_orderkey, o_orderdate, o_shippriority
// ORDER BY revenue DESC, o_orderdate
// LIMIT 10;

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "q3.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { cudaError_t err = (call); if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error: %s at %s:%d\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); std::exit(1); } \
    } while(0)
#endif

// ============================================================
// Hash table primitives (same hash64 as Q5/Q16)
// ============================================================
static constexpr uint64_t HT_EMPTY = UINT64_MAX;

__device__ __forceinline__ uint32_t q3_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

// Hash set insert (keys only, no payload)
__device__ __forceinline__ void q3_hashset_insert(
    uint64_t *keys, uint32_t mask, uint64_t key)
{
    uint32_t slot = q3_hash64(key) & mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)HT_EMPTY,
            (unsigned long long)key);
        if (prev == HT_EMPTY || prev == key) return;
        slot = (slot + 1) & mask;
    }
}

// Hash set probe (returns true if key exists)
__device__ __forceinline__ bool q3_hashset_probe(
    const uint64_t *keys, uint32_t mask, uint64_t key)
{
    uint32_t slot = q3_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return true;
        if (k == HT_EMPTY) return false;
        slot = (slot + 1) & mask;
    }
}

// Hash table insert (key + payload)
__device__ __forceinline__ void q3_ht_insert_kv(
    uint64_t *keys, uint64_t *payloads, uint32_t mask,
    uint64_t key, uint64_t payload)
{
    uint32_t slot = q3_hash64(key) & mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)HT_EMPTY,
            (unsigned long long)key);
        if (prev == HT_EMPTY || prev == key) {
            payloads[slot] = payload;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

// Hash table probe (returns payload, or HT_EMPTY if not found)
__device__ __forceinline__ uint64_t q3_ht_probe_kv(
    const uint64_t *keys, const uint64_t *payloads, uint32_t mask,
    uint64_t key)
{
    uint32_t slot = q3_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return payloads[slot];
        if (k == HT_EMPTY) return HT_EMPTY;
        slot = (slot + 1) & mask;
    }
}

// Prefix-sum binary search: find page P such that ps[P] <= gid < ps[P+1].
__device__ __forceinline__ uint32_t q3_ps_find_page(
    const uint64_t *ps, uint32_t n_entries, uint64_t gid)
{
    uint32_t lo = 0, hi = n_entries;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (ps[mid] <= gid) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

// ============================================================
// Phase 1: CUSTOMER scan — filter c_mktsegment = 'BUILDING'
// ============================================================

// "BUILDING" as uint64_t in little-endian: 'B','U','I','L','D','I','N','G'
// B=0x42 U=0x55 I=0x49 L=0x4C D=0x44 I=0x49 N=0x4E G=0x47
// LE: 0x474E49444C495542
static constexpr uint64_t BUILDING_U64 = 0x474E49444C495542ULL;

// upper_bound on prefix_sum: find the page such that ps[page-1] <= idx < ps[page]
__device__ __forceinline__ uint32_t q3_upper_bound(
    const uint64_t *ps, uint32_t n, uint64_t val)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (ps[mid] <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void q3_customer_scan_kernel(
    const char *__restrict__ c_mktseg_pages,
    const uint64_t *__restrict__ d_prefix_sum_mktseg,
    uint32_t npages_mktseg,
    uint32_t page_size,
    uint32_t padded_len,
    const uint64_t *__restrict__ d_c_custkey_flat,
    uint64_t nrecs_customer,
    uint64_t *__restrict__ d_custkey_set,
    uint32_t set_mask)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_customer) return;

    // Find page via upper_bound on prefix_sum
    uint32_t page_idx = q3_upper_bound(d_prefix_sum_mktseg, npages_mktseg, idx);
    uint64_t base = (page_idx > 0) ? d_prefix_sum_mktseg[page_idx - 1] : 0;
    uint32_t local_slot = (uint32_t)(idx - base);

    // Read C_MKTSEGMENT: pag_head (12B) + padded_len * local_slot
    const char *page = c_mktseg_pages + (uint64_t)page_idx * page_size;
    const char *rec = page + 12 + padded_len * local_slot;

    // Compare first 8 bytes: "BUILDING" (8 chars, uniquely identifies among 5 mktsegment values)
    // rec is at offset 12 + 12*N which is 4-byte aligned but not always 8-byte aligned.
    // Use two 4-byte reads to avoid misaligned uint64_t access.
    uint32_t lo = *reinterpret_cast<const uint32_t *>(rec);
    uint32_t hi = *reinterpret_cast<const uint32_t *>(rec + 4);
    uint64_t val8 = ((uint64_t)hi << 32) | (uint64_t)lo;
    if (val8 != BUILDING_U64) return;

    // Filter passed → insert custkey into hash set
    uint64_t custkey = d_c_custkey_flat[idx];
    q3_hashset_insert(d_custkey_set, set_mask, custkey);
}

cudaError_t q3_customer_scan(
    const char *d_mktseg_pages,
    const uint64_t *d_prefix_sum_mktseg,
    uint32_t npages_mktseg,
    uint32_t page_size,
    uint32_t padded_len,
    const uint64_t *d_c_custkey_flat,
    uint64_t nrecs_customer,
    uint64_t *d_custkey_set,
    uint32_t set_mask,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_customer + BLOCK - 1) / BLOCK);
    q3_customer_scan_kernel<<<grid, BLOCK, 0, stream>>>(
        d_mktseg_pages, d_prefix_sum_mktseg, npages_mktseg,
        page_size, padded_len, d_c_custkey_flat, nrecs_customer,
        d_custkey_set, set_mask);
    return cudaGetLastError();
}

// ============================================================
// Phase 2: ORDERS probe + filter + hash table build
// ============================================================
__global__ void q3_orders_probe_build_kernel(
    const uint64_t *__restrict__ d_o_custkey,
    const uint64_t *__restrict__ d_o_orderdate_u64,
    const uint64_t *__restrict__ d_o_orderkey,
    const uint64_t *__restrict__ d_o_shippriority_u64,
    uint64_t nrecs_orders,
    const uint64_t *__restrict__ d_custkey_set,
    uint32_t custkey_set_mask,
    uint64_t *__restrict__ d_orders_ht_keys,
    uint64_t *__restrict__ d_orders_ht_payloads,
    uint32_t orders_ht_mask)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_orders) return;

    // Filter 1: o_orderdate < 19950315 (cheap INT32 comparison first)
    uint32_t o_orderdate = (uint32_t)d_o_orderdate_u64[idx];
    if (o_orderdate >= 19950315u) return;

    // Filter 2: probe customer hash set (c_custkey = o_custkey)
    uint64_t o_custkey = d_o_custkey[idx];
    if (!q3_hashset_probe(d_custkey_set, custkey_set_mask, o_custkey)) return;

    // Both filters passed → insert into ORDERS hash table
    uint64_t o_orderkey = d_o_orderkey[idx];
    uint32_t o_shippriority = (uint32_t)d_o_shippriority_u64[idx];
    uint64_t payload = ((uint64_t)o_orderdate << 32) | (uint64_t)o_shippriority;

    q3_ht_insert_kv(d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                     o_orderkey, payload);
}

cudaError_t q3_orders_probe_build(
    const uint64_t *d_o_custkey,
    const uint64_t *d_o_orderdate_u64,
    const uint64_t *d_o_orderkey,
    const uint64_t *d_o_shippriority_u64,
    uint64_t nrecs_orders,
    const uint64_t *d_custkey_set,
    uint32_t custkey_set_mask,
    uint64_t *d_orders_ht_keys,
    uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_orders + BLOCK - 1) / BLOCK);
    q3_orders_probe_build_kernel<<<grid, BLOCK, 0, stream>>>(
        d_o_custkey, d_o_orderdate_u64, d_o_orderkey, d_o_shippriority_u64,
        nrecs_orders,
        d_custkey_set, custkey_set_mask,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask);
    return cudaGetLastError();
}

// ============================================================
// Phase 3: LINEITEM probe + revenue aggregation
// ============================================================
__global__ void q3_lineitem_probe_aggr_kernel(
    const uint64_t *__restrict__ d_l_orderkey,
    const uint64_t *__restrict__ d_l_shipdate_u64,
    const uint64_t *__restrict__ d_l_extendedprice_u64,
    const uint64_t *__restrict__ d_l_discount_u64,
    uint64_t nrecs_lineitem,
    const uint64_t *__restrict__ d_orders_ht_keys,
    const uint64_t *__restrict__ d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    uint64_t *__restrict__ d_aggr_keys,
    int64_t  *__restrict__ d_aggr_revenues,
    uint32_t aggr_mask)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_lineitem) return;

    // Filter: l_shipdate > 19950315
    uint32_t l_shipdate = (uint32_t)d_l_shipdate_u64[idx];
    if (l_shipdate <= 19950315u) return;

    // Probe ORDERS hash table
    uint64_t l_orderkey = d_l_orderkey[idx];
    uint64_t payload = q3_ht_probe_kv(d_orders_ht_keys, d_orders_ht_payloads,
                                       orders_ht_mask, l_orderkey);
    if (payload == HT_EMPTY) return;

    // Revenue: l_extendedprice * (100 - l_discount)
    int32_t l_extendedprice = (int32_t)d_l_extendedprice_u64[idx];
    int32_t l_discount = (int32_t)d_l_discount_u64[idx];
    int64_t revenue = (int64_t)l_extendedprice * (int64_t)(100 - l_discount);

    // Aggregate into GROUP BY hash map (key: l_orderkey)
    uint32_t aggr_slot = q3_hash64(l_orderkey) & aggr_mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&d_aggr_keys[aggr_slot]),
            (unsigned long long)HT_EMPTY,
            (unsigned long long)l_orderkey);
        if (prev == HT_EMPTY || prev == l_orderkey) {
            atomicAdd(reinterpret_cast<unsigned long long *>(&d_aggr_revenues[aggr_slot]),
                      (unsigned long long)revenue);
            return;
        }
        aggr_slot = (aggr_slot + 1) & aggr_mask;
    }
}

cudaError_t q3_lineitem_probe_aggr(
    const uint64_t *d_l_orderkey,
    const uint64_t *d_l_shipdate_u64,
    const uint64_t *d_l_extendedprice_u64,
    const uint64_t *d_l_discount_u64,
    uint64_t nrecs_lineitem,
    const uint64_t *d_orders_ht_keys,
    const uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    uint64_t *d_aggr_keys,
    int64_t  *d_aggr_revenues,
    uint32_t aggr_mask,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_lineitem + BLOCK - 1) / BLOCK);
    q3_lineitem_probe_aggr_kernel<<<grid, BLOCK, 0, stream>>>(
        d_l_orderkey, d_l_shipdate_u64, d_l_extendedprice_u64, d_l_discount_u64,
        nrecs_lineitem,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
        d_aggr_keys, d_aggr_revenues, aggr_mask);
    return cudaGetLastError();
}

// ============================================================
// Phase 4: Collect non-empty entries from aggregation hash map
// ============================================================
__global__ void q3_collect_results_kernel(
    const uint64_t *__restrict__ d_aggr_keys,
    const int64_t  *__restrict__ d_aggr_revenues,
    uint32_t aggr_capacity,
    const uint64_t *__restrict__ d_orders_ht_keys,
    const uint64_t *__restrict__ d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    Q3ResultRow *__restrict__ d_results,
    uint32_t *__restrict__ d_result_count)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= aggr_capacity) return;

    uint64_t key = d_aggr_keys[idx];
    if (key == HT_EMPTY) return;

    // Probe orders HT on GPU to retrieve (o_orderdate, o_shippriority)
    uint64_t payload = q3_ht_probe_kv(d_orders_ht_keys, d_orders_ht_payloads,
                                       orders_ht_mask, key);
    uint32_t o_orderdate    = (uint32_t)(payload >> 32);
    uint32_t o_shippriority = (uint32_t)(payload & 0xFFFFFFFF);

    uint32_t pos = atomicAdd(d_result_count, 1);
    d_results[pos] = { key, d_aggr_revenues[idx], o_orderdate, o_shippriority };
}

cudaError_t q3_collect_results(
    const uint64_t *d_aggr_keys,
    const int64_t  *d_aggr_revenues,
    uint32_t aggr_capacity,
    const uint64_t *d_orders_ht_keys,
    const uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    Q3ResultRow *d_results,
    uint32_t *d_result_count,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((aggr_capacity + BLOCK - 1) / BLOCK);
    q3_collect_results_kernel<<<grid, BLOCK, 0, stream>>>(
        d_aggr_keys, d_aggr_revenues, aggr_capacity,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
        d_results, d_result_count);
    return cudaGetLastError();
}

// ============================================================
// Page-based ORDERS HT build (zone map mode)
// Iterates over active O_ORDERDATE (INT32) pages.  Uses prefix-
// sum binary search to map global record IDs to INT64 pages
// (O_ORDERKEY, O_CUSTKEY) which may have a different page count.
// O_SHIPPRIORITY is INT32 so shares the same page_id directly.
// ============================================================
__global__ void q3_orders_probe_build_paged_kernel(
    const char *__restrict__ o_orderdate_pages,
    const char *__restrict__ o_orderkey_pages,
    const char *__restrict__ o_custkey_pages,
    const char *__restrict__ o_shippriority_pages,
    const uint32_t *__restrict__ active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *__restrict__ ps_ref,
    const uint64_t *__restrict__ ps_i64,
    uint32_t npages_i64,
    const uint64_t *__restrict__ d_custkey_set,
    uint32_t custkey_set_mask,
    uint64_t *__restrict__ ht_ord_keys,
    uint64_t *__restrict__ ht_ord_payloads,
    uint32_t ht_ord_mask)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t apg_idx = (uint32_t)(tid / stride);
    uint32_t local_idx = (uint32_t)(tid % stride);
    if (apg_idx >= num_active_pages) return;

    uint32_t page_id = active_pages[apg_idx];
    const char *odate_page = o_orderdate_pages + (uint64_t)page_id * page_size;
    uint32_t nalloc = *(const uint32_t *)odate_page;
    if (local_idx >= nalloc) return;

    // Filter: o_orderdate < 19950315
    int32_t odate = *(const int32_t *)(odate_page + 12 + (uint64_t)local_idx * 4);
    if (odate >= 19950315) return;

    // O_SHIPPRIORITY is INT32 → same page_id
    int32_t shippriority = *(const int32_t *)(
        o_shippriority_pages + (uint64_t)page_id * page_size + 12 + (uint64_t)local_idx * 4);

    // Map to INT64 page via prefix_sum binary search
    uint64_t gid = ps_ref[page_id] + local_idx;
    uint32_t i64pg = q3_ps_find_page(ps_i64, npages_i64 + 1, gid);
    uint32_t i64lc = (uint32_t)(gid - ps_i64[i64pg]);

    // Probe CUSTOMER hash set
    uint64_t custkey = *(const uint64_t *)(
        o_custkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);
    if (!q3_hashset_probe(d_custkey_set, custkey_set_mask, custkey)) return;

    // Read O_ORDERKEY from INT64 page
    uint64_t orderkey = *(const uint64_t *)(
        o_orderkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);

    uint64_t payload = ((uint64_t)(uint32_t)odate << 32) | (uint64_t)(uint32_t)shippriority;
    q3_ht_insert_kv(ht_ord_keys, ht_ord_payloads, ht_ord_mask, orderkey, payload);
}

cudaError_t q3_orders_probe_build_paged(
    const char *o_orderdate_pages,
    const char *o_orderkey_pages,
    const char *o_custkey_pages,
    const char *o_shippriority_pages,
    const uint32_t *d_active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *d_ps_ref,
    const uint64_t *d_ps_i64,
    uint32_t npages_i64,
    const uint64_t *d_custkey_set,
    uint32_t custkey_set_mask,
    uint64_t *d_orders_ht_keys,
    uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    cudaStream_t stream)
{
    if (num_active_pages == 0) return cudaSuccess;
    constexpr int BLOCK = 256;
    uint64_t total_threads = (uint64_t)num_active_pages * stride;
    int grid = (int)((total_threads + BLOCK - 1) / BLOCK);
    q3_orders_probe_build_paged_kernel<<<grid, BLOCK, 0, stream>>>(
        o_orderdate_pages, o_orderkey_pages, o_custkey_pages,
        o_shippriority_pages,
        d_active_pages, num_active_pages, page_size, stride,
        d_ps_ref, d_ps_i64, npages_i64,
        d_custkey_set, custkey_set_mask,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask);
    return cudaGetLastError();
}

// ============================================================
// Page-based LINEITEM probe (zone map mode)
// Iterates over active L_SHIPDATE (INT32) pages.  Uses prefix-
// sum binary search to map to INT64 pages (L_ORDERKEY).
// L_EXTENDEDPRICE and L_DISCOUNT are INT32 so share the same
// page_id directly as the reference (L_SHIPDATE).
// ============================================================
__global__ void q3_lineitem_probe_aggr_paged_kernel(
    const char *__restrict__ l_shipdate_pages,
    const char *__restrict__ l_extprice_pages,
    const char *__restrict__ l_discount_pages,
    const char *__restrict__ l_orderkey_pages,
    const uint32_t *__restrict__ active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *__restrict__ ps_ref,
    const uint64_t *__restrict__ ps_i64,
    uint32_t npages_i64,
    const uint64_t *__restrict__ ht_ord_keys,
    const uint64_t *__restrict__ ht_ord_payloads,
    uint32_t ht_ord_mask,
    uint64_t *__restrict__ d_aggr_keys,
    int64_t  *__restrict__ d_aggr_revenues,
    uint32_t aggr_mask)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t apg_idx = (uint32_t)(tid / stride);
    uint32_t local_idx = (uint32_t)(tid % stride);
    if (apg_idx >= num_active_pages) return;

    uint32_t page_id = active_pages[apg_idx];
    const char *sd_page = l_shipdate_pages + (uint64_t)page_id * page_size;
    uint32_t nalloc = *(const uint32_t *)sd_page;
    if (local_idx >= nalloc) return;

    // Filter: l_shipdate > 19950315
    int32_t shipdate = *(const int32_t *)(sd_page + 12 + (uint64_t)local_idx * 4);
    if (shipdate <= 19950315) return;

    // Map to INT64 page via prefix_sum binary search
    uint64_t gid = ps_ref[page_id] + local_idx;
    uint32_t i64pg = q3_ps_find_page(ps_i64, npages_i64 + 1, gid);
    uint32_t i64lc = (uint32_t)(gid - ps_i64[i64pg]);

    // Probe ORDERS hash table
    uint64_t orderkey = *(const uint64_t *)(
        l_orderkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);
    uint64_t ord_payload = q3_ht_probe_kv(ht_ord_keys, ht_ord_payloads,
                                            ht_ord_mask, orderkey);
    if (ord_payload == HT_EMPTY) return;

    // Revenue: l_extendedprice * (100 - l_discount)
    int32_t extprice = *(const int32_t *)(
        l_extprice_pages + (uint64_t)page_id * page_size + 12 + (uint64_t)local_idx * 4);
    int32_t discount = *(const int32_t *)(
        l_discount_pages + (uint64_t)page_id * page_size + 12 + (uint64_t)local_idx * 4);
    int64_t revenue = (int64_t)extprice * (int64_t)(100 - discount);

    // Aggregate
    uint32_t aggr_slot = q3_hash64(orderkey) & aggr_mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&d_aggr_keys[aggr_slot]),
            (unsigned long long)HT_EMPTY,
            (unsigned long long)orderkey);
        if (prev == HT_EMPTY || prev == orderkey) {
            atomicAdd(reinterpret_cast<unsigned long long *>(&d_aggr_revenues[aggr_slot]),
                      (unsigned long long)revenue);
            return;
        }
        aggr_slot = (aggr_slot + 1) & aggr_mask;
    }
}

cudaError_t q3_lineitem_probe_aggr_paged(
    const char *l_shipdate_pages,
    const char *l_extprice_pages,
    const char *l_discount_pages,
    const char *l_orderkey_pages,
    const uint32_t *d_active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *d_ps_ref,
    const uint64_t *d_ps_i64,
    uint32_t npages_i64,
    const uint64_t *d_orders_ht_keys,
    const uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    uint64_t *d_aggr_keys,
    int64_t  *d_aggr_revenues,
    uint32_t aggr_mask,
    cudaStream_t stream)
{
    if (num_active_pages == 0) return cudaSuccess;
    constexpr int BLOCK = 256;
    uint64_t total_threads = (uint64_t)num_active_pages * stride;
    int grid = (int)((total_threads + BLOCK - 1) / BLOCK);
    q3_lineitem_probe_aggr_paged_kernel<<<grid, BLOCK, 0, stream>>>(
        l_shipdate_pages, l_extprice_pages, l_discount_pages, l_orderkey_pages,
        d_active_pages, num_active_pages, page_size, stride,
        d_ps_ref, d_ps_i64, npages_i64,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
        d_aggr_keys, d_aggr_revenues, aggr_mask);
    return cudaGetLastError();
}

// ============================================================
// Q3SEL: CUSTOMER scan with variable selectivity
// num_segments == 0 → all customers pass (100% selectivity)
// num_segments > 0 → match any of segment_values[0..num_segments-1]
// ============================================================

__global__ void q3sel_customer_scan_kernel(
    const char *__restrict__ c_mktseg_pages,
    const uint64_t *__restrict__ d_prefix_sum_mktseg,
    uint32_t npages_mktseg,
    uint32_t page_size,
    uint32_t padded_len,
    const uint64_t *__restrict__ d_c_custkey_flat,
    uint64_t nrecs_customer,
    uint64_t *__restrict__ d_custkey_set,
    uint32_t set_mask,
    uint32_t num_segments,
    uint64_t seg0, uint64_t seg1, uint64_t seg2, uint64_t seg3, uint64_t seg4)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_customer) return;

    if (num_segments > 0) {
        uint32_t page_idx = q3_upper_bound(d_prefix_sum_mktseg, npages_mktseg, idx);
        uint64_t base = (page_idx > 0) ? d_prefix_sum_mktseg[page_idx - 1] : 0;
        uint32_t local_slot = (uint32_t)(idx - base);

        const char *page = c_mktseg_pages + (uint64_t)page_idx * page_size;
        const char *rec = page + 12 + padded_len * local_slot;

        uint32_t lo = *reinterpret_cast<const uint32_t *>(rec);
        uint32_t hi = *reinterpret_cast<const uint32_t *>(rec + 4);
        uint64_t val8 = ((uint64_t)hi << 32) | (uint64_t)lo;

        bool match = (val8 == seg0);
        if (num_segments >= 2) match = match || (val8 == seg1);
        if (num_segments >= 3) match = match || (val8 == seg2);
        if (num_segments >= 4) match = match || (val8 == seg3);
        if (num_segments >= 5) match = match || (val8 == seg4);
        if (!match) return;
    }

    uint64_t custkey = d_c_custkey_flat[idx];
    q3_hashset_insert(d_custkey_set, set_mask, custkey);
}

cudaError_t q3sel_customer_scan(
    const char *d_mktseg_pages,
    const uint64_t *d_prefix_sum_mktseg,
    uint32_t npages_mktseg,
    uint32_t page_size,
    uint32_t padded_len,
    const uint64_t *d_c_custkey_flat,
    uint64_t nrecs_customer,
    uint64_t *d_custkey_set,
    uint32_t set_mask,
    uint32_t num_segments,
    const uint64_t *segment_values,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_customer + BLOCK - 1) / BLOCK);
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    if (num_segments >= 1) s0 = segment_values[0];
    if (num_segments >= 2) s1 = segment_values[1];
    if (num_segments >= 3) s2 = segment_values[2];
    if (num_segments >= 4) s3 = segment_values[3];
    if (num_segments >= 5) s4 = segment_values[4];
    q3sel_customer_scan_kernel<<<grid, BLOCK, 0, stream>>>(
        d_mktseg_pages, d_prefix_sum_mktseg, npages_mktseg,
        page_size, padded_len, d_c_custkey_flat, nrecs_customer,
        d_custkey_set, set_mask,
        num_segments, s0, s1, s2, s3, s4);
    return cudaGetLastError();
}

// ============================================================
// Q3SEL: ORDERS probe + build (no date filter)
// ============================================================

__global__ void q3sel_orders_probe_build_kernel(
    const uint64_t *__restrict__ d_o_custkey,
    const uint64_t *__restrict__ d_o_orderdate_u64,
    const uint64_t *__restrict__ d_o_orderkey,
    const uint64_t *__restrict__ d_o_shippriority_u64,
    uint64_t nrecs_orders,
    const uint64_t *__restrict__ d_custkey_set,
    uint32_t custkey_set_mask,
    uint64_t *__restrict__ d_orders_ht_keys,
    uint64_t *__restrict__ d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    int32_t o_orderdate_limit)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_orders) return;

    uint64_t o_custkey = d_o_custkey[idx];
    if (!q3_hashset_probe(d_custkey_set, custkey_set_mask, o_custkey)) return;

    uint32_t o_orderdate = (uint32_t)d_o_orderdate_u64[idx];
    if (o_orderdate_limit != 0 && (int32_t)o_orderdate >= o_orderdate_limit) return;

    uint64_t o_orderkey = d_o_orderkey[idx];
    uint32_t o_shippriority = (uint32_t)d_o_shippriority_u64[idx];
    uint64_t payload = ((uint64_t)o_orderdate << 32) | (uint64_t)o_shippriority;

    q3_ht_insert_kv(d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
                     o_orderkey, payload);
}

cudaError_t q3sel_orders_probe_build(
    const uint64_t *d_o_custkey,
    const uint64_t *d_o_orderdate_u64,
    const uint64_t *d_o_orderkey,
    const uint64_t *d_o_shippriority_u64,
    uint64_t nrecs_orders,
    const uint64_t *d_custkey_set,
    uint32_t custkey_set_mask,
    uint64_t *d_orders_ht_keys,
    uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    int32_t o_orderdate_limit,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_orders + BLOCK - 1) / BLOCK);
    q3sel_orders_probe_build_kernel<<<grid, BLOCK, 0, stream>>>(
        d_o_custkey, d_o_orderdate_u64, d_o_orderkey, d_o_shippriority_u64,
        nrecs_orders,
        d_custkey_set, custkey_set_mask,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
        o_orderdate_limit);
    return cudaGetLastError();
}

// ============================================================
// Q3SEL: LINEITEM probe + aggregate (no shipdate filter)
// ============================================================

__global__ void q3sel_lineitem_probe_aggr_kernel(
    const uint64_t *__restrict__ d_l_orderkey,
    const uint64_t *__restrict__ d_l_extendedprice_u64,
    const uint64_t *__restrict__ d_l_discount_u64,
    const uint64_t *__restrict__ d_l_shipdate_u64,
    uint64_t nrecs_lineitem,
    const uint64_t *__restrict__ d_orders_ht_keys,
    const uint64_t *__restrict__ d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    uint64_t *__restrict__ d_aggr_keys,
    int64_t  *__restrict__ d_aggr_revenues,
    uint32_t aggr_mask,
    int32_t l_shipdate_limit)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrecs_lineitem) return;

    if (l_shipdate_limit != 0) {
        int32_t sd = (int32_t)d_l_shipdate_u64[idx];
        if (sd <= l_shipdate_limit) return;
    }

    uint64_t l_orderkey = d_l_orderkey[idx];
    uint64_t payload = q3_ht_probe_kv(d_orders_ht_keys, d_orders_ht_payloads,
                                       orders_ht_mask, l_orderkey);
    if (payload == HT_EMPTY) return;

    int32_t l_extendedprice = (int32_t)d_l_extendedprice_u64[idx];
    int32_t l_discount = (int32_t)d_l_discount_u64[idx];
    int64_t revenue = (int64_t)l_extendedprice * (int64_t)(100 - l_discount);

    uint32_t aggr_slot = q3_hash64(l_orderkey) & aggr_mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&d_aggr_keys[aggr_slot]),
            (unsigned long long)HT_EMPTY,
            (unsigned long long)l_orderkey);
        if (prev == HT_EMPTY || prev == l_orderkey) {
            atomicAdd(reinterpret_cast<unsigned long long *>(&d_aggr_revenues[aggr_slot]),
                      (unsigned long long)revenue);
            return;
        }
        aggr_slot = (aggr_slot + 1) & aggr_mask;
    }
}

cudaError_t q3sel_lineitem_probe_aggr(
    const uint64_t *d_l_orderkey,
    const uint64_t *d_l_extendedprice_u64,
    const uint64_t *d_l_discount_u64,
    const uint64_t *d_l_shipdate_u64,
    uint64_t nrecs_lineitem,
    const uint64_t *d_orders_ht_keys,
    const uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    uint64_t *d_aggr_keys,
    int64_t  *d_aggr_revenues,
    uint32_t aggr_mask,
    int32_t l_shipdate_limit,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_lineitem + BLOCK - 1) / BLOCK);
    q3sel_lineitem_probe_aggr_kernel<<<grid, BLOCK, 0, stream>>>(
        d_l_orderkey, d_l_extendedprice_u64, d_l_discount_u64, d_l_shipdate_u64,
        nrecs_lineitem,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
        d_aggr_keys, d_aggr_revenues, aggr_mask,
        l_shipdate_limit);
    return cudaGetLastError();
}

// ============================================================
// Q3SEL Page-based ORDERS HT build (no date filter)
// Same as q3_orders_probe_build_paged but without the
// o_orderdate < 19950315 filter.
// ============================================================
__global__ void q3sel_orders_probe_build_paged_kernel(
    const char *__restrict__ o_orderdate_pages,
    const char *__restrict__ o_orderkey_pages,
    const char *__restrict__ o_custkey_pages,
    const char *__restrict__ o_shippriority_pages,
    const uint32_t *__restrict__ active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *__restrict__ ps_ref,
    const uint64_t *__restrict__ ps_i64,
    uint32_t npages_i64,
    const uint64_t *__restrict__ d_custkey_set,
    uint32_t custkey_set_mask,
    uint64_t *__restrict__ ht_ord_keys,
    uint64_t *__restrict__ ht_ord_payloads,
    uint32_t ht_ord_mask,
    int32_t o_orderdate_limit)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t apg_idx = (uint32_t)(tid / stride);
    uint32_t local_idx = (uint32_t)(tid % stride);
    if (apg_idx >= num_active_pages) return;

    uint32_t page_id = active_pages[apg_idx];
    const char *odate_page = o_orderdate_pages + (uint64_t)page_id * page_size;
    uint32_t nalloc = *(const uint32_t *)odate_page;
    if (local_idx >= nalloc) return;

    int32_t odate = *(const int32_t *)(odate_page + 12 + (uint64_t)local_idx * 4);
    if (o_orderdate_limit != 0 && odate >= o_orderdate_limit) return;

    int32_t shippriority = *(const int32_t *)(
        o_shippriority_pages + (uint64_t)page_id * page_size + 12 + (uint64_t)local_idx * 4);

    uint64_t gid = ps_ref[page_id] + local_idx;
    uint32_t i64pg = q3_ps_find_page(ps_i64, npages_i64 + 1, gid);
    uint32_t i64lc = (uint32_t)(gid - ps_i64[i64pg]);

    uint64_t custkey = *(const uint64_t *)(
        o_custkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);
    if (!q3_hashset_probe(d_custkey_set, custkey_set_mask, custkey)) return;

    uint64_t orderkey = *(const uint64_t *)(
        o_orderkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);

    uint64_t payload = ((uint64_t)(uint32_t)odate << 32) | (uint64_t)(uint32_t)shippriority;
    q3_ht_insert_kv(ht_ord_keys, ht_ord_payloads, ht_ord_mask, orderkey, payload);
}

cudaError_t q3sel_orders_probe_build_paged(
    const char *o_orderdate_pages,
    const char *o_orderkey_pages,
    const char *o_custkey_pages,
    const char *o_shippriority_pages,
    const uint32_t *d_active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *d_ps_ref,
    const uint64_t *d_ps_i64,
    uint32_t npages_i64,
    const uint64_t *d_custkey_set,
    uint32_t custkey_set_mask,
    uint64_t *d_orders_ht_keys,
    uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    int32_t o_orderdate_limit,
    cudaStream_t stream)
{
    if (num_active_pages == 0) return cudaSuccess;
    constexpr int BLOCK = 256;
    uint64_t total_threads = (uint64_t)num_active_pages * stride;
    int grid = (int)((total_threads + BLOCK - 1) / BLOCK);
    q3sel_orders_probe_build_paged_kernel<<<grid, BLOCK, 0, stream>>>(
        o_orderdate_pages, o_orderkey_pages, o_custkey_pages,
        o_shippriority_pages,
        d_active_pages, num_active_pages, page_size, stride,
        d_ps_ref, d_ps_i64, npages_i64,
        d_custkey_set, custkey_set_mask,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
        o_orderdate_limit);
    return cudaGetLastError();
}

// ============================================================
// Q3SEL Page-based LINEITEM probe + aggregate (no shipdate filter)
// Same as q3_lineitem_probe_aggr_paged but without the
// l_shipdate <= 19950315 filter.
// ============================================================
__global__ void q3sel_lineitem_probe_aggr_paged_kernel(
    const char *__restrict__ l_shipdate_pages,
    const char *__restrict__ l_extprice_pages,
    const char *__restrict__ l_discount_pages,
    const char *__restrict__ l_orderkey_pages,
    const uint32_t *__restrict__ active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *__restrict__ ps_ref,
    const uint64_t *__restrict__ ps_i64,
    uint32_t npages_i64,
    const uint64_t *__restrict__ ht_ord_keys,
    const uint64_t *__restrict__ ht_ord_payloads,
    uint32_t ht_ord_mask,
    uint64_t *__restrict__ d_aggr_keys,
    int64_t  *__restrict__ d_aggr_revenues,
    uint32_t aggr_mask,
    int32_t l_shipdate_limit)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t apg_idx = (uint32_t)(tid / stride);
    uint32_t local_idx = (uint32_t)(tid % stride);
    if (apg_idx >= num_active_pages) return;

    uint32_t page_id = active_pages[apg_idx];
    const char *sd_page = l_shipdate_pages + (uint64_t)page_id * page_size;
    uint32_t nalloc = *(const uint32_t *)sd_page;
    if (local_idx >= nalloc) return;

    if (l_shipdate_limit != 0) {
        int32_t shipdate = *(const int32_t *)(sd_page + 12 + (uint64_t)local_idx * 4);
        if (shipdate <= l_shipdate_limit) return;
    }

    uint64_t gid = ps_ref[page_id] + local_idx;
    uint32_t i64pg = q3_ps_find_page(ps_i64, npages_i64 + 1, gid);
    uint32_t i64lc = (uint32_t)(gid - ps_i64[i64pg]);

    uint64_t orderkey = *(const uint64_t *)(
        l_orderkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);
    uint64_t ord_payload = q3_ht_probe_kv(ht_ord_keys, ht_ord_payloads,
                                            ht_ord_mask, orderkey);
    if (ord_payload == HT_EMPTY) return;

    int32_t extprice = *(const int32_t *)(
        l_extprice_pages + (uint64_t)page_id * page_size + 12 + (uint64_t)local_idx * 4);
    int32_t discount = *(const int32_t *)(
        l_discount_pages + (uint64_t)page_id * page_size + 12 + (uint64_t)local_idx * 4);
    int64_t revenue = (int64_t)extprice * (int64_t)(100 - discount);

    uint32_t aggr_slot = q3_hash64(orderkey) & aggr_mask;
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long *>(&d_aggr_keys[aggr_slot]),
            (unsigned long long)HT_EMPTY,
            (unsigned long long)orderkey);
        if (prev == HT_EMPTY || prev == orderkey) {
            atomicAdd(reinterpret_cast<unsigned long long *>(&d_aggr_revenues[aggr_slot]),
                      (unsigned long long)revenue);
            return;
        }
        aggr_slot = (aggr_slot + 1) & aggr_mask;
    }
}

cudaError_t q3sel_lineitem_probe_aggr_paged(
    const char *l_shipdate_pages,
    const char *l_extprice_pages,
    const char *l_discount_pages,
    const char *l_orderkey_pages,
    const uint32_t *d_active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *d_ps_ref,
    const uint64_t *d_ps_i64,
    uint32_t npages_i64,
    const uint64_t *d_orders_ht_keys,
    const uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    uint64_t *d_aggr_keys,
    int64_t  *d_aggr_revenues,
    uint32_t aggr_mask,
    int32_t l_shipdate_limit,
    cudaStream_t stream)
{
    if (num_active_pages == 0) return cudaSuccess;
    constexpr int BLOCK = 256;
    uint64_t total_threads = (uint64_t)num_active_pages * stride;
    int grid = (int)((total_threads + BLOCK - 1) / BLOCK);
    q3sel_lineitem_probe_aggr_paged_kernel<<<grid, BLOCK, 0, stream>>>(
        l_shipdate_pages, l_extprice_pages, l_discount_pages, l_orderkey_pages,
        d_active_pages, num_active_pages, page_size, stride,
        d_ps_ref, d_ps_i64, npages_i64,
        d_orders_ht_keys, d_orders_ht_payloads, orders_ht_mask,
        d_aggr_keys, d_aggr_revenues, aggr_mask,
        l_shipdate_limit);
    return cudaGetLastError();
}
