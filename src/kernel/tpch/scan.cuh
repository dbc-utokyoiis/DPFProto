#pragma once

extern "C" cudaError_t scan_customer_row(
    void **pags,
    size_t npages,
    uint attridx,
    uint nattrs,
    size_t page_size,
    uint32_t max_nrecs,
    int64_t *count,
    bool use_prefetch,
    cudaStream_t stream);

//extern "C" cudaError_t q6_shared_debug(
//    const Lineitem *lineitems,
//    size_t len,
//    int64_t *revenue,
//    cudaStream_t stream);


// extern "C" cudaError_t q6_subpage_shared(
//     const Lineitem *lineitems,
//     size_t len,
//     int64_t *revenue,
//     cudaStream_t stream);

//extern "C" cudaError_t q6_shared_subpage(
//    Lineitem *lineitems_buf,
//    size_t siz_page,
//    size_t siz_subpage,
//    size_t nlineitem,
//    size_t nlineitem_per_subpage,
//    size_t nlineitem_per_subpage_final,
//    size_t nsubpage,
//    int64_t *revenue,
//    cudaStream_t stream);
