#include "helper.cuh"
#include "gpu_rec.cuh"
#include "gpu_pag.cuh"
//#include "common/pag.h"
#include <cub/cub.cuh>

/* helper for prefetch */
template <int Bytes>
__device__ __forceinline__ void cp_async_ca_shared_global(void* dst_smem, const void* src_gmem) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp.async supports 4/8/16 B only");
  unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst_smem));
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
      "r"(smem_addr),     // %0: shared-space address (32-bit)
      "l"(src_gmem),      // %1: global-space address (64-bit OK)
      "n"(Bytes)          // %2: immediate copy size
  );
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int NGroup>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(NGroup) : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

#define ITEMS_PER_THREAD_1 (1)
#define ITEMS_PER_THREAD_4 (4)
#define ITEMS_PER_THREAD_8 (8)
#define ITEMS_PER_THREAD_16 (16)
#define ITEMS_PER_THREAD_32 (32)
#define ITEMS_PER_THREAD_48 (48)
#define ITEMS_PER_THREAD_64 (64)
#define ITEMS_PER_THREAD_96 (96)
#define ITEMS_PER_THREAD_128 (128)
#define ITEMS_PER_THREAD_192 (192)

#define NCALC_LOOP (1)

template <int ALIGNMENT>
static __device__ __host__ constexpr int align_up(int n) {
    return (n + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
}

template<uint nbytes_prefetch>
static __device__ void kmp_multi_pattern_kernel_aligned_with_prefetch_and_pivot(
    bool do_match,
    int nrecs_in_page,
    const char* __restrict__ string_base,
    int string_len,
    /* KMP-specific arguments */
    const char* __restrict__ patterns_global,
    const int* __restrict__ next_global,
    const int* __restrict__ pattern_offsets,
    const int* __restrict__ pattern_lengths,
    int num_patterns,
    int total_pattern_chars,
    bool* __restrict__ result,
    /* shared memory address for KMP */
    const char* __restrict__ shared_mem_buf1,
    const char* __restrict__ shared_mem_buf2,
    const char* __restrict__ shared_mem_pattern_base)
{
    char *smem_buf1 = (char*)shared_mem_buf1;
    char *smem_buf2 = (char*)shared_mem_buf2;
    // --- ポインタ計算の修正 (アライメント対応) ---
    // char配列の直後にint配列を置くため、オフセットを4バイト境界に切り上げる
    string_len = align_up<4>(string_len);
    int pat_size_aligned = align_up<4>(total_pattern_chars);

    char* pat_shared = (char*)shared_mem_pattern_base;
    // これで int* へのキャストが安全になる
    int* next_shared = (int*)(pat_shared + pat_size_aligned);

    // next_shared は int配列なので、サイズ(バイト数)は常に4の倍数。
    // したがって、後続の offsets_shared も自動的にアライメントされる。
    int* offsets_shared = (int*)(next_shared + total_pattern_chars);
    int* lens_shared = (int*)(offsets_shared + num_patterns);

    int tid = threadIdx.x;

    // パターン文字
    for (int i = tid; i < total_pattern_chars; i += blockDim.x) {
        pat_shared[i] = patterns_global[i];
    }
    // Nextテーブル (要素数 = total_pattern_chars)
    for (int i = tid; i < total_pattern_chars; i += blockDim.x) {
        next_shared[i] = next_global[i];
    }
    // メタデータ
    for (int i = tid; i < num_patterns; i += blockDim.x) {
        offsets_shared[i] = pattern_offsets[i];
        lens_shared[i] = pattern_lengths[i];
    }
    __syncthreads();
    if (!do_match) {
        return;
    }

    //const uint32_t* my_string_u32 = (const uint32_t*)string_base;
    constexpr int PIVOT_WIDTH = nbytes_prefetch;
    int num_tiles = (string_len + PIVOT_WIDTH - 1)/ PIVOT_WIDTH;

    int current_pat_idx = 0;
    int l = 0;

    int p_offset = offsets_shared[current_pat_idx];
    int p_len = lens_shared[current_pat_idx];
    const char* current_pat_ptr = pat_shared + p_offset;
    const int* current_next_ptr = next_shared + p_offset;

    char *buf = (char*)string_base;

    // for (int i = 0; i < num_tiles; ++i) {
    //     uint32_t t = my_string_u32[i];
    //     ...
    // }
#if 0
    for (int i = 0; i < num_tiles; ++i) {
        uint32_t t = my_string_u32[i];

        for (int k = 0; k < PIVOT_WIDTH; ++k) {
            if (current_pat_idx >= num_patterns) break;

            char c = (char)(t & 0xFF);

            bool match = (current_pat_ptr[j] == c);
            bool at_zero = (j == 0);
            bool advance_input = match || at_zero;

            if (advance_input) {
                t >>= 8;
                if (match) j++;

                if (j == p_len) {
                    current_pat_idx++;
                    j = 0;

                    if (current_pat_idx < num_patterns) {
                        p_offset = offsets_shared[current_pat_idx];
                        p_len = lens_shared[current_pat_idx];
                        current_pat_ptr = pat_shared + p_offset;
                        current_next_ptr = next_shared + p_offset;
                    } else {
                        break;
                    }
                }
            } else {
                j = current_next_ptr[j - 1];
                k--;
            }
        }
        if (current_pat_idx >= num_patterns) break;
    }
#endif
    assert(num_tiles > 0);

    // int nrecs_in_page,
    // const char* __restrict__ string_base,
    {
        cp_async_ca_shared_global<nbytes_prefetch>(smem_buf1, &buf[0]);
        cp_async_commit_group();
        char *rbuf = smem_buf1;
        int bufidx = 1;
        int j = 1;

        for (; j < num_tiles; j++) {
            if (bufidx == 0) {
                cp_async_ca_shared_global<nbytes_prefetch>(smem_buf1, &buf[j * nbytes_prefetch * nrecs_in_page]);
                rbuf = smem_buf2;
            } else {
                cp_async_ca_shared_global<nbytes_prefetch>(smem_buf2, &buf[j * nbytes_prefetch * nrecs_in_page]);
                rbuf = smem_buf1;
            }
            bufidx++;
            bufidx = bufidx & 1;
            cp_async_commit_group();
            cp_async_wait_group<1>();

            // for (int k = 0; k < nbytes_prefetch; k++) {
            //     char ch = rbuf[k];
            //     for (int l = 0; l < ncalc_loop; l++) {
            //         checksum += static_cast<uint64_t>(ch);
            //     }
            // }
            for (int k = 0; k < nbytes_prefetch; ++k) {
                if (current_pat_idx >= num_patterns) break;

                char c = rbuf[k];

                bool match = (current_pat_ptr[l] == c);
                bool at_zero = (l == 0);
                bool advance_input = match || at_zero;

                if (advance_input) {
                    if (match) l++;

                    if (l == p_len) {
                        current_pat_idx++;
                        l = 0;

                        if (current_pat_idx < num_patterns) {
                            p_offset = offsets_shared[current_pat_idx];
                            p_len = lens_shared[current_pat_idx];
                            current_pat_ptr = pat_shared + p_offset;
                            current_next_ptr = next_shared + p_offset;
                        } else {
                            break;
                        }
                    }
                } else {
                    l = current_next_ptr[l - 1];
                    k--;
                }
            }
            if (current_pat_idx >= num_patterns) break;
        }
        if (bufidx == 0) {
            rbuf = smem_buf2;
        } else {
            rbuf = smem_buf1;
        }
        cp_async_wait_all();

        // process final tile
        for (int k = 0; k < nbytes_prefetch; ++k) {
            if (current_pat_idx >= num_patterns) break;

            char c = rbuf[k];

            bool match = (current_pat_ptr[l] == c);
            bool at_zero = (l == 0);
            bool advance_input = match || at_zero;

            if (advance_input) {
                if (match) l++;

                if (l == p_len) {
                    current_pat_idx++;
                    l = 0;

                    if (current_pat_idx < num_patterns) {
                        p_offset = offsets_shared[current_pat_idx];
                        p_len = lens_shared[current_pat_idx];
                        current_pat_ptr = pat_shared + p_offset;
                        current_next_ptr = next_shared + p_offset;
                    } else {
                        break;
                    }
                }
            } else {
                l = current_next_ptr[l - 1];
                k--;
            }
        }
    }

#if 0
    if (string_len % nbytes_prefetch) {
        for (int k = num_tiles * nbytes_prefetch; k < string_len; k++) {
            if (current_pat_idx >= num_patterns) break;

            char c = buf[k];
            //char c = rbuf[k];

            bool match = (current_pat_ptr[l] == c);
            bool at_zero = (l == 0);
            bool advance_input = match || at_zero;

            if (advance_input) {
                if (match) l++;

                if (l == p_len) {
                    current_pat_idx++;
                    l = 0;

                    if (current_pat_idx < num_patterns) {
                        p_offset = offsets_shared[current_pat_idx];
                        p_len = lens_shared[current_pat_idx];
                        current_pat_ptr = pat_shared + p_offset;
                        current_next_ptr = next_shared + p_offset;
                    } else {
                        break;
                    }
                }
            } else {
                l = current_next_ptr[l - 1];
                k--;
            }
        }
    }
#endif

    *result = (current_pat_idx == num_patterns);
}

template<uint ITEMS_PER_THREAD>
__global__ void scan_customer_page_kernel(
    PAG **pags,
    uint page_size,
    uint nattrs,
    uint attridx,
    int64_t *count)
{
    extern __shared__ __align__(16) char smem[];

    uint pagidx = blockIdx.x;
    uint recidx_base = threadIdx.x;
    uint t = threadIdx.x;

    size_t ncalc_loop = NCALC_LOOP;
    // size_t chrbufidx = blockDim.x * blockIdx.x + threadIdx.x;
    // size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    using BlockReduceInt = cub::BlockReduce<uint64_t, NUM_THREADS_PER_BLOCK>;
    __shared__ typename BlockReduceInt::TempStorage temp_storage[2];

    //printf("addr %p\n", pags);
    PAG* pag = pags[pagidx];
    //printf("%x", pags[0]);
    uint32_t nalloc = pag_get_nalloc(pag);
    //printf("nalloc: %d (blockIdx.x=%d, threadIdx.x=%d)\n", nalloc, blockIdx.x, threadIdx.x);
    if (nalloc == 0) {
        return;
    }

    uint64_t nrecs = 0;
    uint64_t checksum = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        uint recidx = recidx_base + i * blockDim.x;
        if (recidx < nalloc) {
            REC* rec = pag_fetch_rec(pag, recidx, page_size);
            int vchrlen = rec_get_size_vchr(rec, nattrs, attridx);
            char *buf = rec_get_attr_vchr_aligned(rec, nattrs, attridx, 4);
            for (int j = 0; j < vchrlen; j++) {
                volatile char ch = buf[j];
                for (int l = 0; l < ncalc_loop; l++) {
                    checksum += static_cast<uint64_t>(ch);
                }
            }
            nrecs++;
        }
    }
    __syncthreads();

    // printf("count: %ld, nalloc: %u, smem[%u]: %ld\n", *count, nalloc, t, smem[t]);
    unsigned long long int aggregate1 = BlockReduceInt(temp_storage[0]).Sum(nrecs);
    unsigned long long int aggregate2 = BlockReduceInt(temp_storage[1]).Sum(checksum);
    __syncthreads();

    if (t == 0) {
        atomicAdd(reinterpret_cast<unsigned long long int *>(&count[0]), static_cast<unsigned long long int>(aggregate1));
        atomicAdd(reinterpret_cast<unsigned long long int *>(&count[1]), static_cast<unsigned long long int>(aggregate2));
        // printf("(blockIdx.x=%d) count: %ld, nalloc: %u, aggregate: %ld\n", blockIdx.x, *count, nalloc, aggregate);
    }
}

template<uint ITEMS_PER_THREAD>
__global__ void scan_customer_page_kernel_with_prefetch(
    PAG **pags,
    uint page_size,
    uint nattrs,
    uint attridx,
    int64_t *count)
{
    extern __shared__ __align__(16) char smem[];

    uint pagidx = blockIdx.x;
    uint recidx_base = threadIdx.x;
    uint t = threadIdx.x;

    using BlockReduceInt = cub::BlockReduce<uint64_t, NUM_THREADS_PER_BLOCK>;
    __shared__ typename BlockReduceInt::TempStorage temp_storage[2];

    constexpr int nbytes_prefetch = 4;
    char *smem_buf1 = &smem[threadIdx.x * nbytes_prefetch * 2];
    char *smem_buf2 = &smem[threadIdx.x * nbytes_prefetch * 2 + nbytes_prefetch];

    //printf("addr %p\n", pags);
    PAG* pag = pags[pagidx];
    //printf("%x", pags[0]);
    uint32_t nalloc = pag_get_nalloc(pag);
    //printf("nalloc: %d (blockIdx.x=%d, threadIdx.x=%d)\n", nalloc, blockIdx.x, threadIdx.x);
    if (nalloc == 0) {
        return;
    }

    int ncalc_loop = NCALC_LOOP;
    int bufidx = 0;
    uint64_t nrecs = 0;
    uint64_t checksum = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        uint recidx = recidx_base + i * blockDim.x;
        if (recidx < nalloc) {
            REC* rec = pag_fetch_rec(pag, recidx, page_size);
            int vchrlen = rec_get_size_vchr(rec, nattrs, attridx);
            char *buf = rec_get_attr_vchr_aligned(rec, nattrs, attridx, 4);
#if 0
            for (int j = 0; j < vchrlen; j++) {
                volatile char ch = buf[j];
                checksum += static_cast<uint64_t>(ch);
            }
#else
            int ntiles = vchrlen / nbytes_prefetch;
            if (ntiles) {
                cp_async_ca_shared_global<nbytes_prefetch>(smem_buf1, &buf[0]);
                cp_async_commit_group();
                char *rbuf = smem_buf1;
                bufidx = 1;
                int j = 1;
                for (; j < ntiles; j++) {
                    if (bufidx == 0) {
                        cp_async_ca_shared_global<nbytes_prefetch>(smem_buf1, &buf[j * nbytes_prefetch]);
                        rbuf = smem_buf2;
                    } else {
                        cp_async_ca_shared_global<nbytes_prefetch>(smem_buf2, &buf[j * nbytes_prefetch]);
                        rbuf = smem_buf1;
                    }
                    bufidx++;
                    bufidx = bufidx & 1;
                    cp_async_commit_group();
                    cp_async_wait_group<1>();

                    for (int k = 0; k < nbytes_prefetch; k++) {
                        char ch = rbuf[k];
                        for (int l = 0; l < ncalc_loop; l++) {
                            checksum += static_cast<uint64_t>(ch);
                        }
                    }
                }
                if (bufidx == 0) {
                    rbuf = smem_buf2;
                } else {
                    rbuf = smem_buf1;
                }
                cp_async_wait_all();
                for (int k = 0; k < nbytes_prefetch; k++) {
                    char ch = rbuf[k];
                    for (int l = 0; l < ncalc_loop; l++) {
                        checksum += static_cast<uint64_t>(ch);
                    }
                }
            }
            if (vchrlen % nbytes_prefetch) {
                for (int k = ntiles * nbytes_prefetch; k < vchrlen; k++) {
                    char ch = buf[k];
                    for (int l = 0; l < ncalc_loop; l++) {
                        checksum += static_cast<uint64_t>(ch);
                    }
                }
            }
#endif
            nrecs++;
        }
    }
    __syncthreads();

    // printf("count: %ld, nalloc: %u, smem[%u]: %ld\n", *count, nalloc, t, smem[t]);
    unsigned long long int aggregate1 = BlockReduceInt(temp_storage[0]).Sum(nrecs);
    unsigned long long int aggregate2 = BlockReduceInt(temp_storage[1]).Sum(checksum);
    __syncthreads();

    if (t == 0) {
        atomicAdd(reinterpret_cast<unsigned long long int *>(&count[0]), static_cast<unsigned long long int>(aggregate1));
        atomicAdd(reinterpret_cast<unsigned long long int *>(&count[1]), static_cast<unsigned long long int>(aggregate2));
        // printf("(blockIdx.x=%d) count: %ld, nalloc: %u, aggregate: %ld\n", blockIdx.x, *count, nalloc, aggregate);
    }
}


extern "C" cudaError_t scan_customer_row(
    PAG **arr_pag_dev,
    uint npages,
    uint nattrs, /* number of attributes in a page. This is always 1 if the page is a part of column store. */
    uint attridx,  /* index of attributes in a page. This is always 0 if the page is a part of column store. */
    size_t page_size,
    uint32_t max_nrecs,
    int64_t *count,
    bool use_prefetch,
    cudaStream_t stream)
{
    //auto [grid_dim, block_dim] = get_dims_generic(npages);
    const unsigned int grid_dim = npages;
    constexpr unsigned int block_dim = NUM_THREADS_PER_BLOCK;

    #if 0
    printf("grid_dim: %d, block_dim: %d, npages: %d\n", grid_dim, block_dim, npages);
    #endif

    // size_t shared_mem_size = sizeof(int64_t) * block_dim * npages;
    size_t shared_mem_size = 0;

    /* 6144 / 512 threads */
    // size_t num_work_per_thread = max_nrecs / NUM_THREADS_PER_BLOCK;
    size_t num_work_per_thread = max_nrecs / NUM_THREADS_PER_BLOCK;

    #if 0
    printf("num_work_per_thread: %zu\n", num_work_per_thread);
    #endif
    if (use_prefetch) {
        /* 4B * 2 (double buffering) * threads */
        shared_mem_size = 4 * 2 * block_dim;
        if (num_work_per_thread == 0) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_1><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 4) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_4><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 8) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_8><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 16) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_16><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 32) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_32><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 48) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_48><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 64) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_64><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 96) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_96><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 128) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_128><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 128) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_128><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 192) {
            scan_customer_page_kernel_with_prefetch<ITEMS_PER_THREAD_128><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else {
            fprintf(stderr, "Please define ITEMS_PER_THREAD_X to ensure performance. (max_nrecs=%u, num_work_per_thread=%zu)",
                max_nrecs, num_work_per_thread);
            abort();
        }
    } else {
        if (num_work_per_thread == 0) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_1><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 4) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_4><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 8) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_8><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 16) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_16><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 32) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_32><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 48) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_48><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 64) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_64><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 96) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_96><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 128) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_128><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else if (num_work_per_thread <= 192) {
            scan_customer_page_kernel<ITEMS_PER_THREAD_192><<<grid_dim, block_dim, shared_mem_size, stream>>>(
                arr_pag_dev, page_size, nattrs, attridx, count);
        } else {
            fprintf(stderr, "Please define ITEMS_PER_THREAD_X to ensure performance. (max_nrecs=%u, num_work_per_thread=%zu)",
                max_nrecs, num_work_per_thread);
            abort();
        }
    }
    /* this is large enough for SF = 1000 */


    return cudaSuccess;
}
