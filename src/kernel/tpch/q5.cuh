#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Build SUPPLIER hash table: filter by ASIA nations, insert (suppkey → nation_idx).
cudaError_t q5_build_supplier_ht(
    const uint64_t *d_s_suppkey,
    const uint64_t *d_s_nationkey,
    uint64_t nrecs_supplier,
    const int8_t *d_nationkey_to_idx,
    uint64_t *d_ht_supp_keys,
    int32_t  *d_ht_supp_values,
    uint32_t  ht_supp_mask,
    cudaStream_t stream);

// Build CUSTOMER hash table: filter by ASIA nations, insert (custkey → nation_idx).
cudaError_t q5_build_customer_ht(
    const uint64_t *d_c_custkey,
    const uint64_t *d_c_nationkey,
    uint64_t nrecs_customer,
    const int8_t *d_nationkey_to_idx,
    uint64_t *d_ht_cust_keys,
    int32_t  *d_ht_cust_values,
    uint32_t  ht_cust_mask,
    cudaStream_t stream);

// Build ORDERS hash table: date filter + probe CUSTOMER HT, insert (orderkey → nation_idx).
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
    cudaStream_t stream);

// LINEITEM streaming probe: probe ORDERS + SUPPLIER HTs, same-nation check, aggregate revenue.
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
    cudaStream_t stream);

// Page-based ORDERS HT build (zone map mode): iterate over active O_ORDERDATE pages.
// Uses prefix_sum binary search to map INT32 page records to INT64 pages.
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
    cudaStream_t stream);

// Page-based LINEITEM probe (zone map mode): iterate over active pages.
// Uses prefix_sum binary search to map INT32 page records to INT64 pages.
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
    cudaStream_t stream);
