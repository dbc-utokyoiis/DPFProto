#pragma once

#if 0
extern "C" cudaError_t q6_shared(
    const Lineitem *lineitems,
    size_t len,
    int64_t *revenue,
    cudaStream_t stream);

extern "C" cudaError_t q6_shared_debug(
    const Lineitem *lineitems,
    size_t len,
    int64_t *revenue,
    cudaStream_t stream);


// extern "C" cudaError_t q6_subpage_shared(
//     const Lineitem *lineitems,
//     size_t len,
//     int64_t *revenue,
//     cudaStream_t stream);

extern "C" cudaError_t q6_shared_subpage(
    Lineitem *lineitems_buf,
    size_t siz_page,
    size_t siz_subpage,
    size_t nlineitem,
    size_t nlineitem_per_subpage,
    size_t nlineitem_per_subpage_final,
    size_t nsubpage,
    int64_t *revenue,
    cudaStream_t stream);
#endif

//extern "C" cudaError_t q6_col(
//    /* Continuous PAG ARRAY */
//    void *l_shipdate,
//    void *l_discount,
//    void *l_extendedprice,
//    void *l_quantity,
//    size_t siz_page,
//    size_t nrecs_lineitem,
//    int64_t *revenue,
//    cudaStream_t stream);

cudaError_t q6_col(
    /* Continuous PAG ARRAY */
    void *l_shipdate,
    void *l_quantity,
    void *l_discount,
    void *l_extendedprice,
    uint64_t npages,
    uint32_t page_size,
    uint64_t nrecs_lineitem,
    int64_t *d_revenue,
    cudaStream_t stream);


cudaError_t q6_col_indirect(
    void **d_l_shipdate_pages,
    void **d_l_quantity_pages,
    void **d_l_discount_pages,
    void **d_l_extendedprice_pages,
    uint32_t page_size,
    uint64_t nrecs_lineitem,
    int64_t *d_revenue,
    cudaStream_t stream);

cudaError_t q6_col_vardate(
    void *l_shipdate,
    void *l_quantity,
    void *l_discount,
    void *l_extendedprice,
    uint64_t npages,
    uint32_t page_size,
    uint64_t nrecs_lineitem,
    int64_t *d_revenue,
    cudaStream_t stream,
    int32_t sd_low,
    int32_t sd_high,
    int32_t disc_low = 5,
    int32_t disc_high = 7,
    int32_t qt_max = 2400);