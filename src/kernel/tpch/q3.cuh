#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct Q3ResultRow {
    uint64_t l_orderkey;
    int64_t  revenue;        // fixed-point: actual * 10000
    uint32_t o_orderdate;
    uint32_t o_shippriority;
};

struct Q3ResultCmp {
    __host__ __device__ bool operator()(const Q3ResultRow &a, const Q3ResultRow &b) const {
        if (a.revenue != b.revenue) return a.revenue > b.revenue;
        return a.o_orderdate < b.o_orderdate;
    }
};

// Phase 1: CUSTOMER scan — filter c_mktsegment = 'BUILDING', insert custkey into hash set.
// c_mktsegment_pages: raw CHAR(10) padded-12B pages in GPU memory.
// d_prefix_sum_mktseg: exclusive prefix sum (cumulative nalloc) for CHAR pages.
// d_c_custkey_flat: flattened C_CUSTKEY (INT64) array.
cudaError_t q3_customer_scan(
    const char *d_mktseg_pages,
    const uint64_t *d_prefix_sum_mktseg,
    uint32_t npages_mktseg,
    uint32_t page_size,
    uint32_t padded_len,           // CHAR(10) padded to 12
    const uint64_t *d_c_custkey_flat,
    uint64_t nrecs_customer,
    uint64_t *d_custkey_set,
    uint32_t set_mask,
    cudaStream_t stream);

// Phase 2: ORDERS probe + build — probe custkey set, filter o_orderdate < 19950315,
// build orders HT with (o_orderkey → packed(o_orderdate, o_shippriority)).
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
    cudaStream_t stream);

// Phase 3: LINEITEM probe + aggregate — probe orders HT, filter l_shipdate > 19950315,
// aggregate revenue by l_orderkey.
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
    cudaStream_t stream);

// Phase 4: Collect non-empty entries from aggregation hash map.
// Probes orders HT on GPU to attach (o_orderdate, o_shippriority) per row.
cudaError_t q3_collect_results(
    const uint64_t *d_aggr_keys,
    const int64_t  *d_aggr_revenues,
    uint32_t aggr_capacity,
    const uint64_t *d_orders_ht_keys,
    const uint64_t *d_orders_ht_payloads,
    uint32_t orders_ht_mask,
    Q3ResultRow *d_results,
    uint32_t *d_result_count,
    cudaStream_t stream);

// ── Q3SEL variants (no date/shipdate filters, variable mktsegment selectivity) ──

// Q3SEL Phase 1: CUSTOMER scan with variable selectivity.
// num_segments == 0 → all customers pass; > 0 → match any of segment_values[0..N-1].
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
    cudaStream_t stream);

// Q3SEL Phase 2: ORDERS probe + build.
// o_orderdate_limit: 0 = no date filter, >0 = filter o_orderdate < limit.
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
    cudaStream_t stream);

// Q3SEL Phase 3: LINEITEM probe + aggregate.
// l_shipdate_limit: 0 = no shipdate filter, >0 = filter l_shipdate > limit.
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
    cudaStream_t stream);

// Q3SEL Page-based ORDERS HT build.
// o_orderdate_limit: 0 = no date filter, >0 = filter o_orderdate < limit.
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
    cudaStream_t stream);

// Q3SEL Page-based LINEITEM probe + aggregate.
// l_shipdate_limit: 0 = no shipdate filter, >0 = filter l_shipdate > limit.
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
    cudaStream_t stream);

// Page-based ORDERS HT build (zone map mode): iterate over active O_ORDERDATE pages.
// Uses prefix_sum binary search to map INT32 page records to INT64 pages.
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
    cudaStream_t stream);

// Page-based LINEITEM probe (zone map mode): iterate over active pages.
// Uses prefix_sum binary search to map INT32 page records to INT64 pages.
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
    cudaStream_t stream);
