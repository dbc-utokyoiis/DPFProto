// bam_lz4_fused_q5_dim.cu — Q5 Dimension Table Process Kernel
//
// Non-cooperative grid-stride kernel that reads KEY (INT64) and
// NATIONKEY (INT32) from page-indexed staging buffer, filters by
// nationkey_to_idx, and builds the hash table directly.
//
// Reads NK first to skip ~80% of records (only 5/25 nations are
// in the target region at SF100 ASIA).

#include "bam_lz4_fused_q5_dim.cuh"

#include <cstdio>
#include <cstdlib>

#define Q5DIM_CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                           \
                cudaGetErrorString(err), __FILE__, __LINE__);                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

static constexpr uint32_t Q5DIM_THREADS = 256;
static constexpr uint64_t HT_EMPTY = 0xFFFFFFFFFFFFFFFFULL;

// Binary search upper_bound (same as q3p1_upper_bound / q5_scan.cu)
__device__ __forceinline__ uint32_t q5dim_upper_bound(
    const uint64_t* __restrict__ data, uint32_t n, uint64_t val)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (data[mid] <= val)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Hash function (same as q5_scan.cu)
__device__ __forceinline__ uint32_t q5dim_hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (uint32_t)key;
}

// HT insert (same as q5_scan.cu)
__device__ __forceinline__ void q5dim_ht_insert(
    uint64_t* __restrict__ keys,
    int32_t*  __restrict__ values,
    uint32_t mask,
    uint64_t key,
    int32_t value)
{
    uint32_t slot = q5dim_hash64(key) & mask;
    while (true) {
        uint64_t prev = atomicCAS(
            (unsigned long long*)&keys[slot],
            (unsigned long long)HT_EMPTY,
            (unsigned long long)key);
        if (prev == HT_EMPTY || prev == key) {
            values[slot] = value;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

// ════════════════════════════════════════════════════════════════
// Kernel: grid-stride over records, NK-first filter → HT build
// ════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(Q5DIM_THREADS)
void q5_dim_process_kernel(Q5DimProcessParams p)
{
    const uint64_t tid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t idx = tid; idx < p.nrecs; idx += stride) {
        // Step 1: Read NATIONKEY (INT32) first — skip 80% of records
        uint32_t nk_page = q5dim_upper_bound(p.nk_prefix_sum, p.nk_npages, idx);
        uint32_t nk_local = (nk_page == 0)
            ? (uint32_t)idx
            : (uint32_t)(idx - p.nk_prefix_sum[nk_page - 1]);

        const char* nk_page_data = p.d_staging
            + (uint64_t)(p.nk_page_offset + nk_page) * p.page_size;
        int32_t nationkey = reinterpret_cast<const int32_t*>(nk_page_data + 12)[nk_local];

        if (nationkey < 0 || nationkey >= 25) continue;
        int8_t nation_idx = p.d_nationkey_to_idx[nationkey];
        if (nation_idx < 0) continue;

        // Step 2: Read KEY (INT64) — only for matching nations
        uint32_t key_page = q5dim_upper_bound(p.key_prefix_sum, p.key_npages, idx);
        uint32_t key_local = (key_page == 0)
            ? (uint32_t)idx
            : (uint32_t)(idx - p.key_prefix_sum[key_page - 1]);

        const char* key_page_data = p.d_staging
            + (uint64_t)key_page * p.page_size;
        uint64_t key = (uint64_t)reinterpret_cast<const int64_t*>(key_page_data + 16)[key_local];

        // Step 3: Insert into HT
        q5dim_ht_insert(p.d_ht_keys, p.d_ht_values, p.ht_mask, key, (int32_t)nation_idx);
    }
}

// ════════════════════════════════════════════════════════════════
// max_blocks query
// ════════════════════════════════════════════════════════════════

uint32_t q5_dim_process_max_blocks()
{
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, q5_dim_process_kernel, Q5DIM_THREADS, 0);

    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    uint32_t total = (uint32_t)max_blocks_per_sm * (uint32_t)sm_count;
    fprintf(stderr, "[q5_dim_proc] max_blocks_per_sm=%d sm_count=%d max_total=%u\n",
            max_blocks_per_sm, sm_count, total);
    return total;
}

// ════════════════════════════════════════════════════════════════
// Launch function
// ════════════════════════════════════════════════════════════════

void q5_dim_process_launch(
    const Q5DimProcessParams& params,
    uint32_t num_blocks,
    cudaStream_t stream)
{
    q5_dim_process_kernel<<<num_blocks, Q5DIM_THREADS, 0, stream>>>(params);
    Q5DIM_CUDA_CHECK(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════
// Phase 0: Combined REGION + NATION processing on GPU
//
// Single-thread kernel (5 regions + 25 nations — serial is optimal).
//   1. Scans R_NAME for "ASIA" → asia_regionkey
//   2. Filters NATION by asia_regionkey → d_nationkey_to_idx[25]
//
// Page format: offset 0 = nalloc (uint32_t), offset 12 = data.
// R_NAME: CHAR padded to 28 bytes per record.
// R_REGIONKEY, N_NATIONKEY, N_REGIONKEY: INT32 (4 bytes per record).
// ════════════════════════════════════════════════════════════════

__global__ void q5_phase0_region_nation_kernel(
    const char* __restrict__ d_r_rkey_page,
    const char* __restrict__ d_r_name_page,
    const char* __restrict__ d_n_nkey_page,
    const char* __restrict__ d_n_rkey_page,
    int8_t* __restrict__ d_nationkey_to_idx,
    int32_t* __restrict__ d_asia_regionkey)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Step 1: Find asia_regionkey from REGION pages
    uint32_t nalloc_region = *reinterpret_cast<const uint32_t*>(d_r_rkey_page);
    const int32_t* r_rkeys = reinterpret_cast<const int32_t*>(d_r_rkey_page + 12);

    int32_t asia_regionkey = -1;
    for (uint32_t i = 0; i < nalloc_region; i++) {
        const char* rn = d_r_name_page + 12 + 28 * i;
        if (rn[0] == 'A' && rn[1] == 'S' && rn[2] == 'I' && rn[3] == 'A') {
            asia_regionkey = r_rkeys[i];
            break;
        }
    }
    *d_asia_regionkey = asia_regionkey;

    // Step 2: Build nationkey_to_idx from NATION pages
    uint32_t nalloc_nation = *reinterpret_cast<const uint32_t*>(d_n_nkey_page);
    const int32_t* nkeys = reinterpret_cast<const int32_t*>(d_n_nkey_page + 12);
    const int32_t* n_rkeys = reinterpret_cast<const int32_t*>(d_n_rkey_page + 12);

    int8_t nation_count = 0;
    for (uint32_t i = 0; i < nalloc_nation; i++) {
        if (n_rkeys[i] != asia_regionkey) continue;
        int32_t nkey = nkeys[i];
        if (nkey >= 0 && nkey < 25) {
            d_nationkey_to_idx[nkey] = nation_count++;
        }
    }
}

void q5_phase0_region_nation_launch(
    const char* d_r_rkey_page,
    const char* d_r_name_page,
    const char* d_n_nkey_page,
    const char* d_n_rkey_page,
    int8_t* d_nationkey_to_idx,
    int32_t* d_asia_regionkey,
    cudaStream_t stream)
{
    q5_phase0_region_nation_kernel<<<1, 32, 0, stream>>>(
        d_r_rkey_page, d_r_name_page,
        d_n_nkey_page, d_n_rkey_page,
        d_nationkey_to_idx, d_asia_regionkey);
    Q5DIM_CUDA_CHECK(cudaGetLastError());
}
