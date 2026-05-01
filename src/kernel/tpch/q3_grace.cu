#pragma once

#include <algorithm>
#include <utility>

#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "common/error.cu"
#include "hash/murmur2.cu"
#include "hash/table.cu"
#include "schema/customer.cu"
#include "schema/lineitem.cu"
#include "schema/orders.cu"

#define RECORD_LEVEL_PARALLELISM

//const unsigned int NUM_THREADS_PER_BLOCK = 512U;
const unsigned int NUM_THREADS_PER_BLOCK = 1024U;

std::pair<unsigned int, unsigned int> get_dims(size_t len)
{
    unsigned int block_dim = std::min(static_cast<unsigned int>(len), NUM_THREADS_PER_BLOCK);
    unsigned int grid_dim = (static_cast<unsigned int>(len) + block_dim - 1) / block_dim;
    return {grid_dim, block_dim};
}

std::pair<unsigned int, unsigned int> get_dims_subpage(size_t nrecs, size_t nrecs_per_subpage)
{
    //unsigned int block_dim = std::min(static_cast<unsigned int>(nrecs), static_cast<unsigned int>(nrecs_per_subpage));
    unsigned int block_dim = std::min(static_cast<unsigned int>(nrecs_per_subpage), NUM_THREADS_PER_BLOCK);
    unsigned int grid_dim = (static_cast<unsigned int>(nrecs) + block_dim - 1) / block_dim;
    return {grid_dim, block_dim};
}

struct CustomerReduced {
    int64_t custkey;
};

struct OrdersReduced {
    int64_t orderkey;
    int64_t custkey;
    int32_t orderdate;
    int64_t shippriority;
};

struct LineitemReduced {
    int64_t orderkey;
    int64_t extendedprice;
    int64_t discount;
};

template <typename Input, typename Output, typename Selection, typename GetKey, typename Projection = cuda::std::identity>
__global__ void q3_grace_method0_kernel(
    const Input *rows,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    Output **meta_addrs,
    size_t page_size,
    Selection select,
    GetKey get_key,
    Projection project = {})
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < len) {
        Input row = rows[idx];

        if (select(row)) {
            auto partition_key = get_key(row);
            uint64_t hash = MurmurHash64A(&partition_key, sizeof(partition_key), 0);
            uint64_t key = hash % nbuckets;

            size_t out_cap = page_size / sizeof(Output);

            // There is no 64-bit version of atomicInc,
            // so we assume little-endian and increment by 32-bit.
            size_t out_offset = atomicInc((unsigned int *)&meta_lens[key], static_cast<unsigned int>((multiplicity + 1) * out_cap - 1));

            Output *out_addr = meta_addrs[(multiplicity + 1) * key + (out_offset / out_cap)];
            size_t out_idx = out_offset % out_cap;

            out_addr[out_idx] = project(row);
        }
    }
}

__device__ int strncmp_device(const char *s1, const char *s2, size_t n)
{
    unsigned char u1, u2;

    while (n-- > 0) {
        u1 = (unsigned char)*s1++;
        u2 = (unsigned char)*s2++;
        if (u1 != u2)
            return u1 - u2;
        if (u1 == '\0')
            return 0;
    }
    return 0;
}

extern "C" cudaError_t q3_grace_method0_c_custkey(
    const Customer *customers,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    Customer **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_method0_kernel<<<grid_dim, block_dim, 0, stream>>>(
        customers, len, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Customer &c) { return true; },
        [] __device__(const Customer &c) { return c.custkey; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_method0_c_custkey_s(
    const Customer *customers,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    Customer **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_method0_kernel<<<grid_dim, block_dim, 0, stream>>>(
        customers, len, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Customer &c) { return strncmp_device(c.mktsegment, "BUILDING", sizeof(c.mktsegment) / sizeof(c.mktsegment[0])) == 0; },
        [] __device__(const Customer &c) { return c.custkey; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_method0_c_custkey_sp(
    const Customer *customers,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    CustomerReduced **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
#ifdef RECORD_LEVEL_PARALLELISM
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_method0_kernel<<<grid_dim, block_dim, 0, stream>>>(
        customers, len, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Customer &c) { return strncmp_device(c.mktsegment, "BUILDING", sizeof(c.mktsegment) / sizeof(c.mktsegment[0])) == 0; },
        [] __device__(const Customer &c) { return c.custkey; },
        [] __device__(const Customer &c) { return CustomerReduced{.custkey = c.custkey}; });
#else
    for (size_t i = 0; i < len; i++) {
        q3_grace_method0_kernel<<<1, 1, 0, stream>>>(
            &customers[i], 1, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
            [] __device__(const Customer &c) { return strncmp_device(c.mktsegment, "BUILDING", sizeof(c.mktsegment) / sizeof(c.mktsegment[0])) == 0; },
            [] __device__(const Customer &c) { return c.custkey; },
            [] __device__(const Customer &c) { return CustomerReduced{.custkey = c.custkey}; });
    }
#endif

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_method0_o_custkey(
    const Orders *orders,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    Orders **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_method0_kernel<<<grid_dim, block_dim, 0, stream>>>(
        orders, len, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Orders &o) { return true; },
        [] __device__(const Orders &o) { return o.custkey; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_method0_o_custkey_s(
    const Orders *orders,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    Orders **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_method0_kernel<<<grid_dim, block_dim, 0, stream>>>(
        orders, len, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Orders &o) { return o.orderdate < 19950315; },
        [] __device__(const Orders &o) { return o.custkey; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_method0_o_custkey_sp(
    const Orders *orders,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    OrdersReduced **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{

#ifdef RECORD_LEVEL_PARALLELISM
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_method0_kernel<<<grid_dim, block_dim, 0, stream>>>(
        orders, len, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Orders &o) { return o.orderdate < 19950315; },
        [] __device__(const Orders &o) { return o.custkey; },
        [] __device__(const Orders &o) { return OrdersReduced{.orderkey = o.orderkey, .custkey = o.custkey, .orderdate = o.orderdate, .shippriority = o.shippriority}; });
#else
    for (size_t i = 0; i < len; i++) {
        q3_grace_method0_kernel<<<1, 1, 0, stream>>>(
            &orders[i], 1, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
            [] __device__(const Orders &o) { return o.orderdate < 19950315; },
            [] __device__(const Orders &o) { return o.custkey; },
            [] __device__(const Orders &o) { return OrdersReduced{.orderkey = o.orderkey, .custkey = o.custkey, .orderdate = o.orderdate, .shippriority = o.shippriority}; });
    }
#endif

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_method0_l_orderkey(
    const Lineitem *lineitems,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    Lineitem **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_method0_kernel<<<grid_dim, block_dim, 0, stream>>>(
        lineitems, len, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Lineitem &l) { return true; },
        [] __device__(const Lineitem &l) { return l.orderkey; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_method0_l_orderkey_s(
    const Lineitem *lineitems,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    Lineitem **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_method0_kernel<<<grid_dim, block_dim, 0, stream>>>(
        lineitems, len, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Lineitem &l) { return l.shipdate > 19950315; },
        [] __device__(const Lineitem &l) { return l.orderkey; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_method0_l_orderkey_sp(
    const Lineitem *lineitems,
    size_t len,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    LineitemReduced **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
#ifdef RECORD_LEVEL_PARALLELISM
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_method0_kernel<<<grid_dim, block_dim, 0, stream>>>(
        lineitems, len, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Lineitem &l) { return l.shipdate > 19950315; },
        [] __device__(const Lineitem &l) { return l.orderkey; },
        [] __device__(const Lineitem &l) { return LineitemReduced{.orderkey = l.orderkey, .extendedprice = l.extendedprice, .discount = l.discount}; });
#else
    for (size_t i = 0; i < len; i++) {
        q3_grace_method0_kernel<<<1, 1, 0, stream>>>(
            &lineitems[i], 1, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
            [] __device__(const Lineitem &l) { return l.shipdate > 19950315; },
            [] __device__(const Lineitem &l) { return l.orderkey; },
            [] __device__(const Lineitem &l) { return LineitemReduced{.orderkey = l.orderkey, .extendedprice = l.extendedprice, .discount = l.discount}; });
    }
#endif

    return cudaSuccess;
}

// Equivalent to `q3_step0`
template <typename Customer, typename Selection>
__global__ void q3_grace_build_custkey_kernel(
    const Customer *customers,
    size_t len,
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<>> *table_slots,
    size_t table_capacity,
    Selection select)
{
    StaticHashTable<uint64_t, cuda::std::tuple<>> table{table_ctrl, table_slots, table_capacity};

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len) {
        Customer c = customers[idx];

        if (select(c)) {
            table.insert(c.custkey, {}, [](const int64_t &key) { return static_cast<uint64_t>(key); });
        }
    }
}

extern "C" cudaError_t q3_grace_build_custkey(
    const Customer *customers,
    size_t len,
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<>> *table_slots,
    size_t table_capacity,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_build_custkey_kernel<<<grid_dim, block_dim, 0, stream>>>(
        customers, len, table_ctrl, table_slots, table_capacity,
        [] __device__(const Customer &c) { return strncmp_device(c.mktsegment, "BUILDING", sizeof(c.mktsegment) / sizeof(c.mktsegment[0])) == 0; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_build_custkey_s(
    const Customer *customers,
    size_t len,
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<>> *table_slots,
    size_t table_capacity,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_build_custkey_kernel<<<grid_dim, block_dim, 0, stream>>>(
        customers, len, table_ctrl, table_slots, table_capacity,
        [] __device__(const Customer &c) { return true; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_build_custkey_sp(
    const CustomerReduced *customers,
    size_t len,
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<>> *table_slots,
    size_t table_capacity,
    cudaStream_t stream)
{
#ifdef RECORD_LEVEL_PARALLELISM
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_build_custkey_kernel<<<grid_dim, block_dim, 0, stream>>>(
        customers, len, table_ctrl, table_slots, table_capacity,
        [] __device__(const CustomerReduced &c) { return true; });
#else
    for (size_t i = 0; i < len; i++) {
        q3_grace_build_custkey_kernel<<<1, 1, 0, stream>>>(
            &customers[i], 1, table_ctrl, table_slots, table_capacity,
            [] __device__(const CustomerReduced &c) { return true; });
    }
#endif

    return cudaSuccess;
}

// Equivalent to the probe part of `q3_step1` + partition
template <typename Orders, typename Selection>
__global__ void q3_grace_probe_custkey_kernel(
    const Orders *orders,
    size_t len,

    // probe
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<>> *table_slots,
    size_t table_capacity,

    // parition
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    cuda::std::tuple<int64_t, int32_t, int64_t> **meta_addrs,
    size_t page_size,

    Selection select)
{
    StaticHashTable<uint64_t, cuda::std::tuple<>> table{table_ctrl, table_slots, table_capacity};

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len) {
        Orders o = orders[idx];

        if (select(o)) {
            if (auto value = table.find(o.custkey, [](const int64_t &key) { return static_cast<uint64_t>(key); }); value != nullptr) {
                using Output = cuda::std::tuple<int64_t, int32_t, int64_t>;

                uint64_t hash = MurmurHash64A(&o.orderkey, sizeof(o.orderkey), 0);
                uint64_t key = hash % nbuckets;

                size_t out_cap = page_size / sizeof(Output);

                // There is no 64-bit version of atomicInc,
                // so we assume little-endian and increment by 32-bit.
                size_t out_offset = atomicInc((unsigned int *)&meta_lens[key], static_cast<unsigned int>((multiplicity + 1) * out_cap - 1));

                Output *out_addr = meta_addrs[(multiplicity + 1) * key + (out_offset / out_cap)];
                size_t out_idx = out_offset % out_cap;

                out_addr[out_idx] = {o.orderkey, o.orderdate, o.shippriority};
            }
        }
    }
}

extern "C" cudaError_t q3_grace_probe_custkey(
    const Orders *orders,
    size_t len,
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<>> *table_slots,
    size_t table_capacity,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    cuda::std::tuple<int64_t, int32_t, int64_t> **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_probe_custkey_kernel<<<grid_dim, block_dim, 0, stream>>>(
        orders, len, table_ctrl, table_slots, table_capacity, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Orders &o) { return o.orderdate < 19950315; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_probe_custkey_s(
    const Orders *orders,
    size_t len,
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<>> *table_slots,
    size_t table_capacity,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    cuda::std::tuple<int64_t, int32_t, int64_t> **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_probe_custkey_kernel<<<grid_dim, block_dim, 0, stream>>>(
        orders, len, table_ctrl, table_slots, table_capacity, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const Orders &o) { return true; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_probe_custkey_sp(
    const OrdersReduced *orders,
    size_t len,
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<>> *table_slots,
    size_t table_capacity,
    size_t nbuckets,
    size_t multiplicity,
    size_t *meta_lens,
    cuda::std::tuple<int64_t, int32_t, int64_t> **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
#ifdef RECORD_LEVEL_PARALLELISM
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_probe_custkey_kernel<<<grid_dim, block_dim, 0, stream>>>(
        orders, len, table_ctrl, table_slots, table_capacity, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
        [] __device__(const OrdersReduced &o) { return true; });
#else
    for (size_t i = 0; i < len; i++) {
        q3_grace_probe_custkey_kernel<<<1, 1, 0, stream>>>(
            &orders[i], 1, table_ctrl, table_slots, table_capacity, nbuckets, multiplicity, meta_lens, meta_addrs, page_size,
            [] __device__(const OrdersReduced &o) { return true; });
    }
#endif

    return cudaSuccess;
}

// Equivalent to the build part of `q3_step1`
__global__ void q3_grace_build_orderkey_kernel(
    const cuda::std::tuple<int64_t, int32_t, int64_t> *rows,
    size_t len,
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<int32_t, int64_t>> *table_slots,
    size_t table_capacity)
{
    StaticHashTable<uint64_t, cuda::std::tuple<int32_t, int64_t>> table{table_ctrl, table_slots, table_capacity};

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len) {
        auto [o_orderkey, o_orderdate, o_shippriority] = rows[idx];
        table.insert(o_orderkey, {o_orderdate, o_shippriority}, [](const int64_t &key) { return static_cast<uint64_t>(key); });
    }
}

extern "C" cudaError_t q3_grace_build_orderkey(
    const cuda::std::tuple<int64_t, int32_t, int64_t> *rows,
    size_t len,
    uint16_t *table_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<int32_t, int64_t>> *table_slots,
    size_t table_capacity,
    cudaStream_t stream)
{
#ifdef RECORD_LEVEL_PARALLELISM
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_build_orderkey_kernel<<<grid_dim, block_dim, 0, stream>>>(
        rows, len, table_ctrl, table_slots, table_capacity);
#else
    for (size_t i = 0; i < len; i++) {
        q3_grace_build_orderkey_kernel<<<1, 1, 0, stream>>>(
            &rows[i], 1, table_ctrl, table_slots, table_capacity);
    }
#endif

    return cudaSuccess;
}

// Equivalent to `q3_step2_hash`
template <typename Lineitem, typename Selection>
__global__ void q3_grace_probe_orderkey_aggregate_kernel(
    const Lineitem *lineitems,
    size_t len,

    // probe
    uint16_t *table_orderkey_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<int32_t, int64_t>> *table_orderkey_slots,
    size_t table_orderkey_capacity,

    // aggregate
    uint16_t *table_aggregate_ctrl,
    cuda::std::pair<cuda::std::tuple<int64_t, int32_t, int64_t>, int64_t> *table_aggregate_slots,
    size_t table_aggregate_capacity,

    Selection select)
{
    StaticHashTable<uint64_t, cuda::std::tuple<int32_t, int64_t>> table_order{table_orderkey_ctrl, table_orderkey_slots, table_orderkey_capacity};
    StaticHashTable<cuda::std::tuple<int64_t, int32_t, int64_t>, int64_t> table_aggregate{table_aggregate_ctrl, table_aggregate_slots, table_aggregate_capacity};

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len) {
        Lineitem l = lineitems[idx];

        if (select(l)) {
            if (auto value = table_order.find(l.orderkey, [](const int64_t &key) { return static_cast<uint64_t>(key); }); value != nullptr) {
                auto [o_orderdate, o_shippriority] = *value;

                auto revenue = table_aggregate.find_or_insert(
                    {l.orderkey, o_orderdate, o_shippriority},
                    0,
                    [](const cuda::std::tuple<int64_t, int32_t, int64_t> &key) {
                        auto [l_orderkey, o_orderdate, o_shippriority] = key;
                        return static_cast<uint64_t>(l_orderkey) ^ static_cast<uint64_t>(o_orderdate) ^ static_cast<uint64_t>(o_shippriority);
                    });

                // There is no int64_t version of atomicAdd...
                atomicAdd(
                    reinterpret_cast<unsigned long long *>(revenue),
                    static_cast<unsigned long long>(l.extendedprice * (100 - l.discount)));
            }
        }
    }
}

extern "C" cudaError_t q3_grace_probe_orderkey_aggregate(
    const Lineitem *lineitems,
    size_t len,
    uint16_t *table_orderkey_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<int32_t, int64_t>> *table_orderkey_slots,
    size_t table_orderkey_capacity,
    uint16_t *table_aggregate_ctrl,
    cuda::std::pair<cuda::std::tuple<int64_t, int32_t, int64_t>, int64_t> *table_aggregate_slots,
    size_t table_aggregate_capacity,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_probe_orderkey_aggregate_kernel<<<grid_dim, block_dim, 0, stream>>>(
        lineitems, len, table_orderkey_ctrl, table_orderkey_slots, table_orderkey_capacity, table_aggregate_ctrl, table_aggregate_slots, table_aggregate_capacity,
        [] __device__(const Lineitem &l) { return l.shipdate > 19950315; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_probe_orderkey_aggregate_s(
    const Lineitem *lineitems,
    size_t len,
    uint16_t *table_orderkey_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<int32_t, int64_t>> *table_orderkey_slots,
    size_t table_orderkey_capacity,
    uint16_t *table_aggregate_ctrl,
    cuda::std::pair<cuda::std::tuple<int64_t, int32_t, int64_t>, int64_t> *table_aggregate_slots,
    size_t table_aggregate_capacity,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_probe_orderkey_aggregate_kernel<<<grid_dim, block_dim, 0, stream>>>(
        lineitems, len, table_orderkey_ctrl, table_orderkey_slots, table_orderkey_capacity, table_aggregate_ctrl, table_aggregate_slots, table_aggregate_capacity,
        [] __device__(const Lineitem &l) { return true; });

    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_probe_orderkey_aggregate_sp(
    const LineitemReduced *lineitems,
    size_t len,
    uint16_t *table_orderkey_ctrl,
    cuda::std::pair<uint64_t, cuda::std::tuple<int32_t, int64_t>> *table_orderkey_slots,
    size_t table_orderkey_capacity,
    uint16_t *table_aggregate_ctrl,
    cuda::std::pair<cuda::std::tuple<int64_t, int32_t, int64_t>, int64_t> *table_aggregate_slots,
    size_t table_aggregate_capacity,
    cudaStream_t stream)
{
#ifdef RECORD_LEVEL_PARALLELISM
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_probe_orderkey_aggregate_kernel<<<grid_dim, block_dim, 0, stream>>>(
        lineitems, len, table_orderkey_ctrl, table_orderkey_slots, table_orderkey_capacity, table_aggregate_ctrl, table_aggregate_slots, table_aggregate_capacity,
        [] __device__(const LineitemReduced &l) { return true; });
#else
    for (size_t i = 0; i < len; i++) {
        q3_grace_probe_orderkey_aggregate_kernel<<<1, 1, 0, stream>>>(
            &lineitems[i], 1, table_orderkey_ctrl, table_orderkey_slots, table_orderkey_capacity, table_aggregate_ctrl, table_aggregate_slots, table_aggregate_capacity,
            [] __device__(const LineitemReduced &l) { return true; });
    }
#endif

    return cudaSuccess;
}

__global__ void q3_grace_step0_kernel(
    const Customer *customers,
    size_t len,
    uint64_t *keys,
    size_t *indices,
    size_t nbuckets)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < len) {
        Customer c = customers[idx];
        uint64_t hash = MurmurHash64A(&c.custkey, sizeof(c.custkey), 0);
        uint64_t key = hash % nbuckets;
        keys[idx] = key;
        indices[idx] = idx;
    }
}

extern "C" cudaError_t q3_grace_step0(
    const Customer *customers,
    size_t len,
    uint64_t *keys,
    size_t *indices,
    size_t nbuckets,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_step0_kernel<<<grid_dim, block_dim, 0, stream>>>(customers, len, keys, indices, nbuckets);
    return cudaSuccess;
}

struct Q3GraceStep1CompareOp {
    __host__ __device__ __forceinline__ bool operator()(const uint64_t &lhs, const uint64_t &rhs) const
    {
        return lhs < rhs;
    }
};

extern "C" cudaError_t q3_grace_step1_merge(
    uint64_t *d_keys,
    size_t *d_items,
    size_t num_items,
    cudaStream_t stream)
{
    Q3GraceStep1CompareOp compare_op;

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    TRY(cub::DeviceMergeSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys,
        d_items,
        num_items,
        compare_op,
        stream));
    TRY(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
    TRY(cub::DeviceMergeSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys,
        d_items,
        num_items,
        compare_op,
        stream));
    TRY(cudaFreeAsync(d_temp_storage, stream));
    return cudaSuccess;
}

extern "C" cudaError_t q3_grace_step1_radix(
    const uint64_t *d_keys_in,
    uint64_t *d_keys_out,
    const size_t *d_values_in,
    size_t *d_values_out,
    size_t num_items,
    cudaStream_t stream)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    TRY(cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_items,
        0,
        // sizeof(d_keys_in[0]) * 8,
        8,
        stream));
    TRY(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
    TRY(cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_items,
        0,
        // sizeof(d_keys_in[0]) * 8,
        8,
        stream));
    TRY(cudaFreeAsync(d_temp_storage, stream));
    return cudaSuccess;
}

template <typename T>
__device__ size_t lower_bound(const T *array, size_t begin, size_t end, T value)
{
    size_t lo = begin;
    size_t hi = end;

    while (lo < hi) {
        size_t mi = lo + (hi - lo) / 2;
        if (array[mi] >= value) {
            hi = mi;
        }
        else {
            lo = mi + 1;
        }
    }

    return lo;
}

template <typename T>
__device__ size_t upper_bound(const T *array, size_t begin, size_t end, T value)
{
    size_t lo = begin;
    size_t hi = end;

    while (lo < hi) {
        size_t mi = lo + (hi - lo) / 2;
        if (array[mi] > value) {
            hi = mi;
        }
        else {
            lo = mi + 1;
        }
    }

    return lo;
}

__global__ void q3_grace_step2_kernel(
    const uint64_t *keys,
    const Customer *customers,
    const size_t *indices,
    size_t len,
    size_t *offsets,
    size_t *meta_lens,
    Customer **meta_addrs,
    size_t page_size)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < len) {
        uint64_t key = keys[idx];

        size_t lb = lower_bound(keys, 0, len, key);
        size_t offset = idx - lb;
        // offsets[idx] = offset;

        size_t out_len = meta_lens[key];
        size_t out_cap = page_size / sizeof(Customer);

        size_t out_offset = (out_len + offset) % (2 * out_cap);

        Customer *out_addr = meta_addrs[2 * key + (out_offset < out_cap ? 0 : 1)];
        size_t out_idx = out_offset < out_cap ? out_offset : out_offset - out_cap;

        out_addr[out_idx] = customers[indices[idx]];

        // if (offset == 0) {
        //     size_t prev_idx = (idx == 0 ? len : idx) - 1;
        //     size_t prev_key = keys[prev_idx];
        //     size_t prev_size = offsets[prev_idx] + 1;
        //     meta_lens[prev_key] += prev_size;
        // }

        if (offset == 0) {
            size_t ub = upper_bound(keys, lb + 1, len, key);
            size_t size = ub - lb;
            meta_lens[key] += size;
        }

        // atomicAdd((unsigned long long *)&meta_lens[key], 1ULL);
    }
}

extern "C" cudaError_t q3_grace_step2(
    const uint64_t *keys,
    const Customer *customers,
    const size_t *indices,
    size_t len,
    size_t *offsets,
    size_t *meta_lens,
    Customer **meta_addrs,
    size_t page_size,
    cudaStream_t stream)
{
    auto [grid_dim, block_dim] = get_dims(len);

    q3_grace_step2_kernel<<<grid_dim, block_dim, 0, stream>>>(keys, customers, indices, len, offsets, meta_lens, meta_addrs, page_size);
    return cudaSuccess;
}
