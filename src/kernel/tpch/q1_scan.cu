// Q1 GOLAP kernel: scan + aggregate on flattened LINEITEM arrays
// Filter: l_shipdate <= 19980902
// Group by: (l_returnflag, l_linestatus) → max 6 groups
// Aggregates: sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_discount, count
//
// Optimization: per-thread local accumulation with grid-stride loop.
// Reduces atomicAdd from ~3.5B (per-row) to ~166K (per-thread flush).

#include <cuda_runtime.h>
#include <cstdint>

#include "q1.cuh"

static constexpr int Q1_BLOCK_SIZE = 256;
static constexpr int Q1_LOCAL_AGGS = 6;  // QTY, BASE_PRICE, DISC_PRICE, CHARGE, DISCOUNT, COUNT

__global__ void q1_scan_aggregate_kernel(
    const uint64_t *__restrict__ d_l_shipdate,
    const uint64_t *__restrict__ d_l_quantity,
    const uint64_t *__restrict__ d_l_extendedprice,
    const uint64_t *__restrict__ d_l_discount,
    const uint64_t *__restrict__ d_l_tax,
    const uint64_t *__restrict__ d_l_returnflag,
    const uint64_t *__restrict__ d_l_linestatus,
    uint64_t nrecs_lineitem,
    int64_t *__restrict__ d_agg)
{
    // Per-thread local accumulation (registers)
    // 6 groups × 6 aggs = 36 int64_t
    int64_t local_agg[Q1_NUM_GROUPS * Q1_LOCAL_AGGS] = {};

    // Grid-stride loop: each thread processes multiple rows
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nrecs_lineitem; idx += stride) {

        // Filter: l_shipdate <= 19980902
        int32_t shipdate = (int32_t)d_l_shipdate[idx];
        if (shipdate > 19980902) continue;

        // Group key
        char returnflag = (char)(uint8_t)d_l_returnflag[idx];
        char linestatus = (char)(uint8_t)d_l_linestatus[idx];

        int row;
        switch (returnflag) {
            case 'A': row = 0; break;
            case 'N': row = 1; break;
            case 'R': row = 2; break;
            default: continue;
        }
        int col = (linestatus == 'F') ? 0 : 1;
        int gid = row * 2 + col;

        // Read values (all stored as fixed-point x100)
        int32_t quantity      = (int32_t)d_l_quantity[idx];
        int32_t extendedprice = (int32_t)d_l_extendedprice[idx];
        int32_t discount      = (int32_t)d_l_discount[idx];
        int32_t tax           = (int32_t)d_l_tax[idx];

        // Compute aggregates
        int64_t disc_price = (int64_t)extendedprice * (int64_t)(100 - discount);
        int64_t charge = disc_price * (int64_t)(100 + tax);

        // Accumulate into thread-local registers
        int64_t *la = local_agg + gid * Q1_LOCAL_AGGS;
        la[0] += quantity;
        la[1] += extendedprice;
        la[2] += disc_price;
        la[3] += charge;
        la[4] += discount;
        la[5] += 1;
    }

    // Flush local accumulators to global aggregates
    // With sm_count=108, 256 threads/block: max 108×256×6×6 = 995K atomicAdds
    // (vs ~3.5B in the per-row version)
    for (int g = 0; g < Q1_NUM_GROUPS; g++) {
        int64_t *la = local_agg + g * Q1_LOCAL_AGGS;
        if (la[5] == 0) continue;  // this thread had no rows for this group
        int64_t *ga = d_agg + g * Q1_NUM_AGGS;
        atomicAdd((unsigned long long *)&ga[Q1_SUM_QTY],        (unsigned long long)la[0]);
        atomicAdd((unsigned long long *)&ga[Q1_SUM_BASE_PRICE], (unsigned long long)la[1]);
        atomicAdd((unsigned long long *)&ga[Q1_SUM_DISC_PRICE], (unsigned long long)la[2]);
        {
            unsigned long long old_lo = atomicAdd(
                (unsigned long long *)&ga[Q1_SUM_CHARGE], (unsigned long long)la[3]);
            if (old_lo + (unsigned long long)la[3] < old_lo) {
                atomicAdd((unsigned long long *)&ga[Q1_SUM_CHARGE_HI], 1ULL);
            }
        }
        atomicAdd((unsigned long long *)&ga[Q1_SUM_DISCOUNT],   (unsigned long long)la[4]);
        atomicAdd((unsigned long long *)&ga[Q1_COUNT],          (unsigned long long)la[5]);
    }
}

// Host wrapper
cudaError_t q1_scan_aggregate(
    const uint64_t *d_l_shipdate,
    const uint64_t *d_l_quantity,
    const uint64_t *d_l_extendedprice,
    const uint64_t *d_l_discount,
    const uint64_t *d_l_tax,
    const uint64_t *d_l_returnflag,
    const uint64_t *d_l_linestatus,
    uint64_t nrecs_lineitem,
    int64_t *d_agg,
    cudaStream_t stream)
{
    // Use sm_count-based grid for efficient local accumulation
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int grid = sm_count;
    // Cap to data size for small inputs
    int data_grid = (int)((nrecs_lineitem + Q1_BLOCK_SIZE - 1) / Q1_BLOCK_SIZE);
    if (data_grid < grid) grid = data_grid;

    q1_scan_aggregate_kernel<<<grid, Q1_BLOCK_SIZE, 0, stream>>>(
        d_l_shipdate, d_l_quantity, d_l_extendedprice,
        d_l_discount, d_l_tax, d_l_returnflag, d_l_linestatus,
        nrecs_lineitem, d_agg);
    return cudaGetLastError();
}

// ============================================================
// Flat INT32 variant: operates on flattened int32_t arrays.
// Used by datapathfusion path when per-field prefix_sums differ.
// ============================================================

__global__ void q1_scan_aggregate_flat_i32_kernel(
    const int32_t *__restrict__ d_l_shipdate,
    const int32_t *__restrict__ d_l_quantity,
    const int32_t *__restrict__ d_l_extendedprice,
    const int32_t *__restrict__ d_l_discount,
    const int32_t *__restrict__ d_l_tax,
    const int32_t *__restrict__ d_l_returnflag,
    const int32_t *__restrict__ d_l_linestatus,
    uint64_t nrecs_lineitem,
    int64_t *__restrict__ d_agg)
{
    int64_t local_agg[Q1_NUM_GROUPS * Q1_LOCAL_AGGS] = {};

    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nrecs_lineitem; idx += stride) {

        int32_t shipdate = d_l_shipdate[idx];
        if (shipdate > 19980902) continue;

        char returnflag = (char)(uint8_t)d_l_returnflag[idx];
        char linestatus = (char)(uint8_t)d_l_linestatus[idx];

        int row;
        switch (returnflag) {
            case 'A': row = 0; break;
            case 'N': row = 1; break;
            case 'R': row = 2; break;
            default: continue;
        }
        int col = (linestatus == 'F') ? 0 : 1;
        int gid = row * 2 + col;

        int32_t quantity      = d_l_quantity[idx];
        int32_t extendedprice = d_l_extendedprice[idx];
        int32_t discount      = d_l_discount[idx];
        int32_t tax           = d_l_tax[idx];

        int64_t disc_price = (int64_t)extendedprice * (int64_t)(100 - discount);
        int64_t charge = disc_price * (int64_t)(100 + tax);

        int64_t *la = local_agg + gid * Q1_LOCAL_AGGS;
        la[0] += quantity;
        la[1] += extendedprice;
        la[2] += disc_price;
        la[3] += charge;
        la[4] += discount;
        la[5] += 1;
    }

    for (int g = 0; g < Q1_NUM_GROUPS; g++) {
        int64_t *la = local_agg + g * Q1_LOCAL_AGGS;
        if (la[5] == 0) continue;
        int64_t *ga = d_agg + g * Q1_NUM_AGGS;
        atomicAdd((unsigned long long *)&ga[Q1_SUM_QTY],        (unsigned long long)la[0]);
        atomicAdd((unsigned long long *)&ga[Q1_SUM_BASE_PRICE], (unsigned long long)la[1]);
        atomicAdd((unsigned long long *)&ga[Q1_SUM_DISC_PRICE], (unsigned long long)la[2]);
        {
            unsigned long long old_lo = atomicAdd(
                (unsigned long long *)&ga[Q1_SUM_CHARGE], (unsigned long long)la[3]);
            if (old_lo + (unsigned long long)la[3] < old_lo) {
                atomicAdd((unsigned long long *)&ga[Q1_SUM_CHARGE_HI], 1ULL);
            }
        }
        atomicAdd((unsigned long long *)&ga[Q1_SUM_DISCOUNT],   (unsigned long long)la[4]);
        atomicAdd((unsigned long long *)&ga[Q1_COUNT],          (unsigned long long)la[5]);
    }
}

cudaError_t q1_scan_aggregate_flat_i32(
    const int32_t *d_l_shipdate,
    const int32_t *d_l_quantity,
    const int32_t *d_l_extendedprice,
    const int32_t *d_l_discount,
    const int32_t *d_l_tax,
    const int32_t *d_l_returnflag,
    const int32_t *d_l_linestatus,
    uint64_t nrecs_lineitem,
    int64_t *d_agg,
    cudaStream_t stream)
{
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int grid = sm_count;
    int data_grid = (int)((nrecs_lineitem + Q1_BLOCK_SIZE - 1) / Q1_BLOCK_SIZE);
    if (data_grid < grid) grid = data_grid;

    q1_scan_aggregate_flat_i32_kernel<<<grid, Q1_BLOCK_SIZE, 0, stream>>>(
        d_l_shipdate, d_l_quantity, d_l_extendedprice,
        d_l_discount, d_l_tax, d_l_returnflag, d_l_linestatus,
        nrecs_lineitem, d_agg);
    return cudaGetLastError();
}

// ============================================================
// Page-direct variant: reads INT32 values directly from page buffers
// without flatten step.  Same accumulation logic as above.
// ============================================================

struct q1_pag_head {
    uint32_t nalloc;
    uint32_t watermark;
    uint32_t lfreespace;
};

__global__ void q1_scan_aggregate_paged_kernel(
    const void *__restrict__ l_shipdate_pages,
    const void *__restrict__ l_quantity_pages,
    const void *__restrict__ l_extendedprice_pages,
    const void *__restrict__ l_discount_pages,
    const void *__restrict__ l_tax_pages,
    const void *__restrict__ l_returnflag_pages,
    const void *__restrict__ l_linestatus_pages,
    uint64_t nrecs_total,
    uint32_t capacity,
    uint32_t page_size,
    int64_t *__restrict__ d_agg,
    const uint8_t *__restrict__ d_page_active)
{
    int64_t local_agg[Q1_NUM_GROUPS * Q1_LOCAL_AGGS] = {};

    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nrecs_total; idx += stride) {

        uint32_t page_idx = (uint32_t)(idx / capacity);
        uint32_t rec_idx  = (uint32_t)(idx % capacity);

        // Zone map: skip inactive pages (direct filter)
        if (d_page_active && !d_page_active[page_idx]) continue;

        // nalloc check (read from shipdate column — all columns share the same page structure)
        uint32_t nalloc = reinterpret_cast<const q1_pag_head*>(
            reinterpret_cast<const char*>(l_shipdate_pages) + (uint64_t)page_idx * page_size
        )->nalloc;
        if (rec_idx >= nalloc) continue;

        // Filter: l_shipdate <= 19980902
        int32_t shipdate = reinterpret_cast<const int32_t*>(
            reinterpret_cast<const char*>(l_shipdate_pages) + (uint64_t)page_idx * page_size)[3 + rec_idx];
        if (shipdate > 19980902) continue;

        // Group key
        char returnflag = (char)(uint8_t)reinterpret_cast<const int32_t*>(
            reinterpret_cast<const char*>(l_returnflag_pages) + (uint64_t)page_idx * page_size)[3 + rec_idx];
        char linestatus = (char)(uint8_t)reinterpret_cast<const int32_t*>(
            reinterpret_cast<const char*>(l_linestatus_pages) + (uint64_t)page_idx * page_size)[3 + rec_idx];

        int row;
        switch (returnflag) {
            case 'A': row = 0; break;
            case 'N': row = 1; break;
            case 'R': row = 2; break;
            default: continue;
        }
        int col = (linestatus == 'F') ? 0 : 1;
        int gid = row * 2 + col;

        // Read values
        int32_t quantity = reinterpret_cast<const int32_t*>(
            reinterpret_cast<const char*>(l_quantity_pages) + (uint64_t)page_idx * page_size)[3 + rec_idx];
        int32_t extendedprice = reinterpret_cast<const int32_t*>(
            reinterpret_cast<const char*>(l_extendedprice_pages) + (uint64_t)page_idx * page_size)[3 + rec_idx];
        int32_t discount = reinterpret_cast<const int32_t*>(
            reinterpret_cast<const char*>(l_discount_pages) + (uint64_t)page_idx * page_size)[3 + rec_idx];
        int32_t tax = reinterpret_cast<const int32_t*>(
            reinterpret_cast<const char*>(l_tax_pages) + (uint64_t)page_idx * page_size)[3 + rec_idx];

        // Compute aggregates
        int64_t disc_price = (int64_t)extendedprice * (int64_t)(100 - discount);
        int64_t charge = disc_price * (int64_t)(100 + tax);

        // Accumulate into thread-local registers
        int64_t *la = local_agg + gid * Q1_LOCAL_AGGS;
        la[0] += quantity;
        la[1] += extendedprice;
        la[2] += disc_price;
        la[3] += charge;
        la[4] += discount;
        la[5] += 1;
    }

    // Flush local accumulators to global aggregates
    for (int g = 0; g < Q1_NUM_GROUPS; g++) {
        int64_t *la = local_agg + g * Q1_LOCAL_AGGS;
        if (la[5] == 0) continue;
        int64_t *ga = d_agg + g * Q1_NUM_AGGS;
        atomicAdd((unsigned long long *)&ga[Q1_SUM_QTY],        (unsigned long long)la[0]);
        atomicAdd((unsigned long long *)&ga[Q1_SUM_BASE_PRICE], (unsigned long long)la[1]);
        atomicAdd((unsigned long long *)&ga[Q1_SUM_DISC_PRICE], (unsigned long long)la[2]);
        {
            unsigned long long old_lo = atomicAdd(
                (unsigned long long *)&ga[Q1_SUM_CHARGE], (unsigned long long)la[3]);
            if (old_lo + (unsigned long long)la[3] < old_lo) {
                atomicAdd((unsigned long long *)&ga[Q1_SUM_CHARGE_HI], 1ULL);
            }
        }
        atomicAdd((unsigned long long *)&ga[Q1_SUM_DISCOUNT],   (unsigned long long)la[4]);
        atomicAdd((unsigned long long *)&ga[Q1_COUNT],          (unsigned long long)la[5]);
    }
}

// Host wrapper for page-direct variant
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
    int64_t *d_agg,
    cudaStream_t stream,
    const uint8_t *d_page_active)
{
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int grid = sm_count;
    int data_grid = (int)((nrecs_total + Q1_BLOCK_SIZE - 1) / Q1_BLOCK_SIZE);
    if (data_grid < grid) grid = data_grid;

    q1_scan_aggregate_paged_kernel<<<grid, Q1_BLOCK_SIZE, 0, stream>>>(
        l_shipdate_pages, l_quantity_pages, l_extendedprice_pages,
        l_discount_pages, l_tax_pages, l_returnflag_pages, l_linestatus_pages,
        nrecs_total, capacity, page_size, d_agg, d_page_active);
    return cudaGetLastError();
}
