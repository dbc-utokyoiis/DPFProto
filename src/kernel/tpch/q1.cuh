#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

constexpr int Q1_NUM_GROUPS = 6;   // 3 returnflag (A/N/R) x 2 linestatus (F/O)
constexpr int Q1_NUM_AGGS = 7;

// Aggregate index constants
enum Q1AggIdx {
    Q1_SUM_QTY = 0,
    Q1_SUM_BASE_PRICE = 1,
    Q1_SUM_DISC_PRICE = 2,
    Q1_SUM_CHARGE = 3,        // low 64 bits
    Q1_SUM_DISCOUNT = 4,
    Q1_COUNT = 5,
    Q1_SUM_CHARGE_HI = 6,     // high 64 bits (carry count)
};

// Group ordering: (returnflag, linestatus)
// 'A','F' = 0; 'A','O' = 1; 'N','F' = 2; 'N','O' = 3; 'R','F' = 4; 'R','O' = 5

// Q1 scan + aggregate kernel: filter on L_SHIPDATE, group by (returnflag, linestatus)
cudaError_t q1_scan_aggregate(
    const uint64_t *d_l_shipdate,
    const uint64_t *d_l_quantity,
    const uint64_t *d_l_extendedprice,
    const uint64_t *d_l_discount,
    const uint64_t *d_l_tax,
    const uint64_t *d_l_returnflag,
    const uint64_t *d_l_linestatus,
    uint64_t nrecs_lineitem,
    int64_t *d_agg,              // [Q1_NUM_GROUPS * Q1_NUM_AGGS], pre-zeroed
    cudaStream_t stream);

// Flat INT32 variant: operates on flattened int32_t arrays (datapathfusion path)
cudaError_t q1_scan_aggregate_flat_i32(
    const int32_t *d_l_shipdate,
    const int32_t *d_l_quantity,
    const int32_t *d_l_extendedprice,
    const int32_t *d_l_discount,
    const int32_t *d_l_tax,
    const int32_t *d_l_returnflag,
    const int32_t *d_l_linestatus,
    uint64_t nrecs_lineitem,
    int64_t *d_agg,              // [Q1_NUM_GROUPS * Q1_NUM_AGGS], pre-zeroed
    cudaStream_t stream);

// Page-direct variant: reads INT32 values directly from page-structured buffers
// d_page_active: optional per-page mask [npages] (1=active, 0=skip). nullptr → all active.
cudaError_t q1_scan_aggregate_paged(
    const void *l_shipdate_pages,
    const void *l_quantity_pages,
    const void *l_extendedprice_pages,
    const void *l_discount_pages,
    const void *l_tax_pages,
    const void *l_returnflag_pages,
    const void *l_linestatus_pages,
    uint64_t nrecs_total,
    uint32_t capacity,
    uint32_t page_size,
    int64_t *d_agg,              // [Q1_NUM_GROUPS * Q1_NUM_AGGS], pre-zeroed
    cudaStream_t stream,
    const uint8_t *d_page_active = nullptr);

// Helper to print unsigned __int128 value (host-only)
static inline void q1_print_u128(unsigned __int128 val) {
    if (val == 0) { printf("0"); return; }
    char buf[40];
    int pos = 39;
    buf[pos] = '\0';
    while (val > 0) {
        buf[--pos] = '0' + (int)(val % 10);
        val /= 10;
    }
    printf("%s", buf + pos);
}
