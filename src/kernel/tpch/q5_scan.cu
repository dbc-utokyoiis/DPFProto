#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "q5.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { cudaError_t err = (call); if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error: %s at %s:%d\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); std::exit(1); } \
    } while(0)
#endif

// ============================================================
// Hash table primitives (same hash64 as Q16)
// ============================================================
static constexpr uint64_t HT_EMPTY = UINT64_MAX;

__device__ __forceinline__ uint32_t q5_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

__device__ __forceinline__ void q5_ht_insert(
    uint64_t *keys, int32_t *values, uint32_t mask,
    uint64_t key, int32_t value)
{
    uint32_t slot = q5_hash64(key) & mask;
    while (true) {
        uint64_t old = atomicCAS(
            reinterpret_cast<unsigned long long *>(&keys[slot]),
            (unsigned long long)HT_EMPTY,
            (unsigned long long)key);
        if (old == HT_EMPTY || old == key) {
            values[slot] = value;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

__device__ __forceinline__ int32_t q5_ht_probe(
    const uint64_t *keys, const int32_t *values, uint32_t mask,
    uint64_t key)
{
    uint32_t slot = q5_hash64(key) & mask;
    while (true) {
        uint64_t k = keys[slot];
        if (k == key) return values[slot];
        if (k == HT_EMPTY) return -1;
        slot = (slot + 1) & mask;
    }
}

// ============================================================
// Phase 1: Build SUPPLIER hash table
// ============================================================
__global__ void q5_build_supplier_ht_kernel(
    const uint64_t *__restrict__ d_s_suppkey,
    const uint64_t *__restrict__ d_s_nationkey,
    uint64_t nrecs,
    const int8_t *__restrict__ d_nationkey_to_idx,
    uint64_t *__restrict__ ht_keys,
    int32_t  *__restrict__ ht_values,
    uint32_t  ht_mask)
{
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nrecs) return;

    int32_t nationkey = (int32_t)d_s_nationkey[gid];
    if (nationkey < 0 || nationkey >= 25) return;
    int8_t nation_idx = d_nationkey_to_idx[nationkey];
    if (nation_idx < 0) return;

    uint64_t suppkey = d_s_suppkey[gid];
    q5_ht_insert(ht_keys, ht_values, ht_mask, suppkey, (int32_t)nation_idx);
}

cudaError_t q5_build_supplier_ht(
    const uint64_t *d_s_suppkey,
    const uint64_t *d_s_nationkey,
    uint64_t nrecs_supplier,
    const int8_t *d_nationkey_to_idx,
    uint64_t *d_ht_supp_keys,
    int32_t  *d_ht_supp_values,
    uint32_t  ht_supp_mask,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_supplier + BLOCK - 1) / BLOCK);
    q5_build_supplier_ht_kernel<<<grid, BLOCK, 0, stream>>>(
        d_s_suppkey, d_s_nationkey, nrecs_supplier,
        d_nationkey_to_idx,
        d_ht_supp_keys, d_ht_supp_values, ht_supp_mask);
    return cudaGetLastError();
}

// ============================================================
// Phase 2: Build CUSTOMER hash table
// ============================================================
__global__ void q5_build_customer_ht_kernel(
    const uint64_t *__restrict__ d_c_custkey,
    const uint64_t *__restrict__ d_c_nationkey,
    uint64_t nrecs,
    const int8_t *__restrict__ d_nationkey_to_idx,
    uint64_t *__restrict__ ht_keys,
    int32_t  *__restrict__ ht_values,
    uint32_t  ht_mask)
{
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nrecs) return;

    int32_t nationkey = (int32_t)d_c_nationkey[gid];
    if (nationkey < 0 || nationkey >= 25) return;
    int8_t nation_idx = d_nationkey_to_idx[nationkey];
    if (nation_idx < 0) return;

    uint64_t custkey = d_c_custkey[gid];
    q5_ht_insert(ht_keys, ht_values, ht_mask, custkey, (int32_t)nation_idx);
}

cudaError_t q5_build_customer_ht(
    const uint64_t *d_c_custkey,
    const uint64_t *d_c_nationkey,
    uint64_t nrecs_customer,
    const int8_t *d_nationkey_to_idx,
    uint64_t *d_ht_cust_keys,
    int32_t  *d_ht_cust_values,
    uint32_t  ht_cust_mask,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_customer + BLOCK - 1) / BLOCK);
    q5_build_customer_ht_kernel<<<grid, BLOCK, 0, stream>>>(
        d_c_custkey, d_c_nationkey, nrecs_customer,
        d_nationkey_to_idx,
        d_ht_cust_keys, d_ht_cust_values, ht_cust_mask);
    return cudaGetLastError();
}

// ============================================================
// Phase 3: Build ORDERS hash table (date filter + CUSTOMER probe)
// ============================================================
__global__ void q5_build_orders_ht_kernel(
    const uint64_t *__restrict__ d_o_orderkey,
    const uint64_t *__restrict__ d_o_custkey,
    const uint64_t *__restrict__ d_o_orderdate,
    uint64_t nrecs,
    int32_t date_low, int32_t date_high,
    const uint64_t *__restrict__ ht_cust_keys,
    const int32_t  *__restrict__ ht_cust_values,
    uint32_t ht_cust_mask,
    uint64_t *__restrict__ ht_ord_keys,
    int32_t  *__restrict__ ht_ord_values,
    uint32_t  ht_ord_mask)
{
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nrecs) return;

    // Date filter first (cheap)
    int32_t odate = (int32_t)d_o_orderdate[gid];
    if (odate < date_low || odate >= date_high) return;

    // Probe CUSTOMER hash table
    uint64_t custkey = d_o_custkey[gid];
    int32_t cust_nation_idx = q5_ht_probe(ht_cust_keys, ht_cust_values, ht_cust_mask, custkey);
    if (cust_nation_idx < 0) return;

    // Insert into ORDERS hash table
    uint64_t orderkey = d_o_orderkey[gid];
    q5_ht_insert(ht_ord_keys, ht_ord_values, ht_ord_mask, orderkey, cust_nation_idx);
}

cudaError_t q5_build_orders_ht(
    const uint64_t *d_o_orderkey,
    const uint64_t *d_o_custkey,
    const uint64_t *d_o_orderdate,
    uint64_t nrecs_batch,
    int32_t date_low, int32_t date_high,
    const uint64_t *d_ht_cust_keys,
    const int32_t  *d_ht_cust_values,
    uint32_t ht_cust_mask,
    uint64_t *d_ht_ord_keys,
    int32_t  *d_ht_ord_values,
    uint32_t  ht_ord_mask,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_batch + BLOCK - 1) / BLOCK);
    q5_build_orders_ht_kernel<<<grid, BLOCK, 0, stream>>>(
        d_o_orderkey, d_o_custkey, d_o_orderdate,
        nrecs_batch, date_low, date_high,
        d_ht_cust_keys, d_ht_cust_values, ht_cust_mask,
        d_ht_ord_keys, d_ht_ord_values, ht_ord_mask);
    return cudaGetLastError();
}

// ============================================================
// Phase 4: LINEITEM probe + revenue aggregation
// ============================================================
__global__ void q5_lineitem_probe_kernel(
    const uint64_t *__restrict__ d_l_orderkey,
    const uint64_t *__restrict__ d_l_suppkey,
    const uint64_t *__restrict__ d_l_extendedprice,
    const uint64_t *__restrict__ d_l_discount,
    uint64_t nrecs,
    const uint64_t *__restrict__ ht_ord_keys,
    const int32_t  *__restrict__ ht_ord_values,
    uint32_t ht_ord_mask,
    const uint64_t *__restrict__ ht_supp_keys,
    const int32_t  *__restrict__ ht_supp_values,
    uint32_t ht_supp_mask,
    int64_t *__restrict__ d_revenue)
{
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nrecs) return;

    // Probe ORDERS hash table
    uint64_t orderkey = d_l_orderkey[gid];
    int32_t cust_nation_idx = q5_ht_probe(ht_ord_keys, ht_ord_values, ht_ord_mask, orderkey);
    if (cust_nation_idx < 0) return;

    // Probe SUPPLIER hash table
    uint64_t suppkey = d_l_suppkey[gid];
    int32_t supp_nation_idx = q5_ht_probe(ht_supp_keys, ht_supp_values, ht_supp_mask, suppkey);
    if (supp_nation_idx < 0) return;

    // Same-nation constraint: c_nationkey = s_nationkey
    if (cust_nation_idx != supp_nation_idx) return;

    // Compute revenue contribution: l_extendedprice * (100 - l_discount)
    int32_t extprice = (int32_t)d_l_extendedprice[gid];
    int32_t discount = (int32_t)d_l_discount[gid];
    int64_t contribution = (int64_t)extprice * (int64_t)(100 - discount);

    atomicAdd(reinterpret_cast<unsigned long long *>(&d_revenue[cust_nation_idx]),
              (unsigned long long)contribution);
}

cudaError_t q5_lineitem_probe(
    const uint64_t *d_l_orderkey,
    const uint64_t *d_l_suppkey,
    const uint64_t *d_l_extendedprice,
    const uint64_t *d_l_discount,
    uint64_t nrecs_batch,
    const uint64_t *d_ht_ord_keys,
    const int32_t  *d_ht_ord_values,
    uint32_t ht_ord_mask,
    const uint64_t *d_ht_supp_keys,
    const int32_t  *d_ht_supp_values,
    uint32_t ht_supp_mask,
    int64_t *d_revenue,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (int)((nrecs_batch + BLOCK - 1) / BLOCK);
    q5_lineitem_probe_kernel<<<grid, BLOCK, 0, stream>>>(
        d_l_orderkey, d_l_suppkey, d_l_extendedprice, d_l_discount,
        nrecs_batch,
        d_ht_ord_keys, d_ht_ord_values, ht_ord_mask,
        d_ht_supp_keys, d_ht_supp_values, ht_supp_mask,
        d_revenue);
    return cudaGetLastError();
}

// ============================================================
// Prefix-sum binary search: find page P such that ps[P] <= gid < ps[P+1].
// ps is an exclusive prefix sum with (npages + 1) entries:
//   ps[0] = 0, ps[i] = sum of nalloc for pages 0..i-1.
// ============================================================
__device__ __forceinline__ uint32_t q5_ps_find_page(
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
// Page-based ORDERS HT build (zone map mode)
// Iterates over active O_ORDERDATE (INT32) pages.  Uses prefix-
// sum binary search to map global record IDs to INT64 pages
// (O_ORDERKEY, O_CUSTKEY) which may have a different page count.
// ============================================================
__global__ void q5_build_orders_ht_paged_kernel(
    const char *__restrict__ o_orderdate_pages,
    const char *__restrict__ o_orderkey_pages,
    const char *__restrict__ o_custkey_pages,
    const uint32_t *__restrict__ active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *__restrict__ ps_ref,
    const uint64_t *__restrict__ ps_i64,
    uint32_t npages_i64,
    int32_t date_low, int32_t date_high,
    const uint64_t *__restrict__ ht_cust_keys,
    const int32_t  *__restrict__ ht_cust_values,
    uint32_t ht_cust_mask,
    uint64_t *__restrict__ ht_ord_keys,
    int32_t  *__restrict__ ht_ord_values,
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

    int32_t odate = *(const int32_t *)(odate_page + 12 + (uint64_t)local_idx * 4);
    if (odate < date_low || odate >= date_high) return;

    // Map to INT64 page via prefix_sum binary search
    uint64_t gid = ps_ref[page_id] + local_idx;
    uint32_t i64pg = q5_ps_find_page(ps_i64, npages_i64 + 1, gid);
    uint32_t i64lc = (uint32_t)(gid - ps_i64[i64pg]);

    uint64_t custkey = *(const uint64_t *)(
        o_custkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);
    int32_t cust_nation_idx = q5_ht_probe(ht_cust_keys, ht_cust_values,
                                           ht_cust_mask, custkey);
    if (cust_nation_idx < 0) return;

    uint64_t orderkey = *(const uint64_t *)(
        o_orderkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);
    q5_ht_insert(ht_ord_keys, ht_ord_values, ht_ord_mask,
                 orderkey, cust_nation_idx);
}

cudaError_t q5_build_orders_ht_paged(
    const char *o_orderdate_pages,
    const char *o_orderkey_pages,
    const char *o_custkey_pages,
    const uint32_t *d_active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *d_ps_ref,
    const uint64_t *d_ps_i64,
    uint32_t npages_i64,
    int32_t date_low, int32_t date_high,
    const uint64_t *d_ht_cust_keys,
    const int32_t  *d_ht_cust_values,
    uint32_t ht_cust_mask,
    uint64_t *d_ht_ord_keys,
    int32_t  *d_ht_ord_values,
    uint32_t ht_ord_mask,
    cudaStream_t stream)
{
    if (num_active_pages == 0) return cudaSuccess;
    constexpr int BLOCK = 256;
    uint64_t total_threads = (uint64_t)num_active_pages * stride;
    int grid = (int)((total_threads + BLOCK - 1) / BLOCK);
    q5_build_orders_ht_paged_kernel<<<grid, BLOCK, 0, stream>>>(
        o_orderdate_pages, o_orderkey_pages, o_custkey_pages,
        d_active_pages, num_active_pages, page_size, stride,
        d_ps_ref, d_ps_i64, npages_i64,
        date_low, date_high,
        d_ht_cust_keys, d_ht_cust_values, ht_cust_mask,
        d_ht_ord_keys, d_ht_ord_values, ht_ord_mask);
    return cudaGetLastError();
}

// ============================================================
// Page-based LINEITEM probe (zone map mode)
// Iterates over active L_EXTENDEDPRICE (INT32) pages.  Uses
// prefix-sum binary search to map to INT64 pages (L_ORDERKEY,
// L_SUPPKEY).  L_DISCOUNT shares the same page structure as
// L_EXTENDEDPRICE so uses the same page_id directly.
// ============================================================
__global__ void q5_lineitem_probe_paged_kernel(
    const char *__restrict__ l_extprice_pages,
    const char *__restrict__ l_discount_pages,
    const char *__restrict__ l_orderkey_pages,
    const char *__restrict__ l_suppkey_pages,
    const uint32_t *__restrict__ active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *__restrict__ ps_ref,
    const uint64_t *__restrict__ ps_i64,
    uint32_t npages_i64,
    const uint64_t *__restrict__ ht_ord_keys,
    const int32_t  *__restrict__ ht_ord_values,
    uint32_t ht_ord_mask,
    const uint64_t *__restrict__ ht_supp_keys,
    const int32_t  *__restrict__ ht_supp_values,
    uint32_t ht_supp_mask,
    int64_t *__restrict__ d_revenue)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t apg_idx = (uint32_t)(tid / stride);
    uint32_t local_idx = (uint32_t)(tid % stride);
    if (apg_idx >= num_active_pages) return;

    uint32_t page_id = active_pages[apg_idx];
    const char *ep_page = l_extprice_pages + (uint64_t)page_id * page_size;
    uint32_t nalloc = *(const uint32_t *)ep_page;
    if (local_idx >= nalloc) return;

    // Map to INT64 page via prefix_sum binary search
    uint64_t gid = ps_ref[page_id] + local_idx;
    uint32_t i64pg = q5_ps_find_page(ps_i64, npages_i64 + 1, gid);
    uint32_t i64lc = (uint32_t)(gid - ps_i64[i64pg]);

    uint64_t orderkey = *(const uint64_t *)(
        l_orderkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);
    int32_t cust_nation_idx = q5_ht_probe(ht_ord_keys, ht_ord_values,
                                           ht_ord_mask, orderkey);
    if (cust_nation_idx < 0) return;

    uint64_t suppkey = *(const uint64_t *)(
        l_suppkey_pages + (uint64_t)i64pg * page_size + 16 + (uint64_t)i64lc * 8);
    int32_t supp_nation_idx = q5_ht_probe(ht_supp_keys, ht_supp_values,
                                           ht_supp_mask, suppkey);
    if (supp_nation_idx < 0) return;

    if (cust_nation_idx != supp_nation_idx) return;

    // L_EXTENDEDPRICE and L_DISCOUNT share the same page structure
    int32_t extprice = *(const int32_t *)(ep_page + 12 + (uint64_t)local_idx * 4);
    int32_t discount = *(const int32_t *)(
        l_discount_pages + (uint64_t)page_id * page_size + 12 + (uint64_t)local_idx * 4);
    int64_t contribution = (int64_t)extprice * (int64_t)(100 - discount);
    atomicAdd(reinterpret_cast<unsigned long long *>(&d_revenue[cust_nation_idx]),
              (unsigned long long)contribution);
}

cudaError_t q5_lineitem_probe_paged(
    const char *l_extprice_pages,
    const char *l_discount_pages,
    const char *l_orderkey_pages,
    const char *l_suppkey_pages,
    const uint32_t *d_active_pages,
    uint32_t num_active_pages,
    uint32_t page_size,
    uint32_t stride,
    const uint64_t *d_ps_ref,
    const uint64_t *d_ps_i64,
    uint32_t npages_i64,
    const uint64_t *d_ht_ord_keys,
    const int32_t  *d_ht_ord_values,
    uint32_t ht_ord_mask,
    const uint64_t *d_ht_supp_keys,
    const int32_t  *d_ht_supp_values,
    uint32_t ht_supp_mask,
    int64_t *d_revenue,
    cudaStream_t stream)
{
    if (num_active_pages == 0) return cudaSuccess;
    constexpr int BLOCK = 256;
    uint64_t total_threads = (uint64_t)num_active_pages * stride;
    int grid = (int)((total_threads + BLOCK - 1) / BLOCK);
    q5_lineitem_probe_paged_kernel<<<grid, BLOCK, 0, stream>>>(
        l_extprice_pages, l_discount_pages,
        l_orderkey_pages, l_suppkey_pages,
        d_active_pages, num_active_pages, page_size, stride,
        d_ps_ref, d_ps_i64, npages_i64,
        d_ht_ord_keys, d_ht_ord_values, ht_ord_mask,
        d_ht_supp_keys, d_ht_supp_values, ht_supp_mask,
        d_revenue);
    return cudaGetLastError();
}
