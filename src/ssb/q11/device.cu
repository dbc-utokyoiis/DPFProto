// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include "common/common.cu"
#include "common/page.cu"
#include "common/primitive_c.cu"
#include "schema/ssb_tables.cuh"

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

// #include "cub/test/test_util.h"
// #include "ssb_gpu_utils.h"
#include "ssb/q11.cu"

#include "common/primitive_c.cu"
#include "common/primitive_cuda.cu"
#include "common/primitive_cufile.cu"
#include "common/gpu_utils.h"
#include "econfig.h"

using namespace std;
using namespace cub;

struct DeviceSyncQ11Args
{
    size_t thrid;
    CUcontext ctx;
    std::vector<int> fds;
    std::array<std::vector<uint64_t>::iterator, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> pagid_vec_begin_arr;
    std::array<std::vector<uint64_t>::iterator, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> pagid_vec_end_arr;
    std::array<std::vector<char *>, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> buf_dev_vec_arr;
    /* shared between threads*/
    unsigned long long *revenue_dev;
    int64_t q11_sum;
    std::vector<CUfileHandle_t> cufile_handles;
    CUfileBatchHandle_t batch_idp;
    std::vector<CUfileIOParams_t> batch_params_vec;
    std::vector<CUfileIOEvents_t> batch_events_vec;
    uint64_t period_sec;
    size_t stats_nio;
    size_t nrows_processed;
};

#if 0
/**
 * Globals, constants and typedefs
 */
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

template<typename T>
T* loadToGPU(T* src, int numEntries, CachingDeviceAllocator& g_allocator) {
  T* dest;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&dest, sizeof(T) * numEntries));
  CubDebugExit(cudaMemcpy(dest, src, sizeof(T) * numEntries, cudaMemcpyHostToDevice));
  return dest;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernel(
    uint* lo_orderdate_val_block_start, uint* lo_orderdate_val_data,
    uint* lo_orderdate_rl_block_start, uint* lo_orderdate_rl_data,
    uint* lo_discount_block_start, uint* lo_discount_data,
    uint* lo_quantity_block_start, uint* lo_quantity_data,
    uint* lo_extendedprice_block_start, uint* lo_extendedprice_data,
    int lo_num_entries, unsigned long long* revenue) {
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduceInt;

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;    // Current tile index
  int tile_offset = tile_idx * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    typename BlockReduceInt::TempStorage reduce;
    uint shared_buffer[BLOCK_THREADS * ITEMS_PER_THREAD * 2 + 128];
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  long long sum = 0;

  int num_tiles = (lo_num_entries + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
    is_last_tile = true;
  }

    RENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(
        lo_orderdate_val_block_start, lo_orderdate_rl_block_start, lo_orderdate_val_data, lo_orderdate_rl_data,
        temp_storage.shared_buffer, items, items2, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();


  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    selection_flags[ITEM] = 1;

    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      selection_flags[ITEM] = (items[ITEM] > 19930000 && items[ITEM] < 19940000); 
  }

  __syncthreads();

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_quantity_block_start, lo_quantity_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      selection_flags[ITEM] = selection_flags[ITEM] && items[ITEM] < 25;
  }

  __syncthreads();

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_discount_block_start, lo_discount_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      selection_flags[ITEM] = selection_flags[ITEM] && items[ITEM] >= 1 && items[ITEM ] <= 3;
  }

  __syncthreads();

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_extendedprice_block_start, lo_extendedprice_data, temp_storage.shared_buffer, items2, is_last_tile, num_tile_items);

  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM];
  }

  __syncthreads();

  unsigned long long aggregate = BlockReduceInt(temp_storage.reduce).Sum(sum);

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}

float runQuery(encoded_column lo_orderdate_val, encoded_column lo_orderdate_rl, 
  encoded_column lo_discount, encoded_column lo_quantity, 
    encoded_column lo_extendedprice,
    int lo_num_entries, CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  unsigned long long* d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(long long)));
  cudaMemset(d_sum, 0, sizeof(long long));

  // Run
  const int num_threads = 128;
  const int items_per_thread = 4;
  int tile_size = num_threads * items_per_thread;
  QueryKernel<num_threads, items_per_thread><<<(lo_num_entries + tile_size - 1)/tile_size, 128>>>(
          lo_orderdate_val.block_start, lo_orderdate_val.data,
          lo_orderdate_rl.block_start, lo_orderdate_rl.data,
          lo_discount.block_start, lo_discount.data,
          lo_quantity.block_start, lo_quantity.data,
          lo_extendedprice.block_start, lo_extendedprice.data,
          lo_num_entries, d_sum);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start,stop);

  unsigned long long revenue;
  CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));

  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  cout << "Revenue: " << revenue << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  CLEANUP(d_sum);

  return time_query;
}
#endif

BenchmarkResult deviceSyncSSBQ11(BenchmarkOptions &options)
{
  int num_trials  = 5;
  string encoding = ENCODING;

  /* initialize CUDA context */
  mb_cuda_init();
  auto device = mb_cuda_get_device(0);
  auto ctx = mb_cuda_new_context(device);
  mb_cuda_set_context(ctx);


  if (options.page_size % 4096 != 0 || options.page_size < 4096)
  {
      std::cerr << "page_size should be multiple of 4096" << std::endl;
      exit(EXIT_FAILURE);
  }
  const size_t page_size = options.page_size;

  std::vector<int> fds;
  std::vector<CUfileHandle_t> cufile_handles;
  open_files(options, fds);

  size_t size_meta = 4096;
  SSBTableMetadata *metadata_ptr = nullptr;
  metadata_ptr = static_cast<SSBTableMetadata*>(mb_alloc(size_meta));
  std::cout << "metadata_ptr: " << metadata_ptr << std::endl; 
  page_pread_host(fds, metadata_ptr, 0, size_meta);

  auto &metadata = *metadata_ptr;
  assert(metadata.page_size == page_size);
  SSB::metadata_print(metadata);

  const size_t nrows_page_int32 = page_size / sizeof(int32_t);
  const size_t nrows_page_int64 = page_size / sizeof(int64_t);

  size_t start_pagid;
  size_t nios_issued = 0;
  const size_t nrows_lo = metadata.table_lineorder_nrows;
  start_pagid = metadata.table_lineorder_start_page_ids[0];
  size_t npages_lo = (nrows_lo * sizeof(int32_t) - 1) / page_size + 1;
  size_t nthreads = std::min(npages_lo, options.nthreads);
  if (nthreads < options.nthreads)
  {
      std::cout << "npages_lo is smaller than nthreads, so changed nthread to " 
          << nthreads << std::endl;
  }

  std::cout
      << "nrows: " << nrows_lo << std::endl
      << "nrows_page_int32: " << nrows_page_int32 << std::endl
      << "nrows_page_int64: " << nrows_page_int64 << std::endl
      << "npages_lo: " << npages_lo << std::endl;

  constexpr size_t Q11_LO_NUM_ACTIVE_FIELDS = SSB::query::q11::NUM_LO_ACTIVE_FIELDS;
  constexpr auto &active_field_index = SSB::query::q1x::LO_FIELDS;

  std::array<std::vector<uint64_t>, Q11_LO_NUM_ACTIVE_FIELDS> pagid_vec_arr;
  for (size_t f = 0; f < Q11_LO_NUM_ACTIVE_FIELDS; f++)
  {
      pagid_vec_arr[f].reserve(npages_lo);
      start_pagid = metadata.table_lineorder_start_page_ids[active_field_index[f]];
      pagid_vec_arr[f] = generate_pagid_table(start_pagid, start_pagid + npages_lo);
  }

  mb_cufile_driver_open();
  cufile_handles.reserve(fds.size());
  for (auto fd : fds)
  {
      CUfileHandle_t cufile_handle = mb_cufile_handle_register(fd);
      cufile_handles.push_back(cufile_handle);
      // std::cout << " fd " << fd << " is registered to a cufile handle object" << std::endl;
  }
  CUfileDrvProps_t props;
  cuFileDriverGetProperties(&props);
  if (Q11_LO_NUM_ACTIVE_FIELDS * options.io_multiplicity > props.max_batch_io_size)
  {
#if 0
      for (auto cufile_handle : cufile_handles)
      {
          mb_cufile_handle_deregister(cufile_handle);
      }
      mb_cufile_driver_close();
      close_files(options, fds);
      exit(EXIT_FAILURE);
#else
      std::cout << "[WARN] io_multiplicity is too large. "
          << "io_multiplicity=" << options.io_multiplicity
          << "was shrinked to " << props.max_batch_io_size / Q11_LO_NUM_ACTIVE_FIELDS
          << std::endl;
#endif
  }
  assert(Q11_LO_NUM_ACTIVE_FIELDS * options.io_multiplicity < props.max_batch_io_size);
  size_t io_multiplicity = std::min(
    options.io_multiplicity,
    props.max_batch_io_size / Q11_LO_NUM_ACTIVE_FIELDS);
  
  // Only support io_multiplicity = 1 for now
  assert(io_multiplicity == 1);
  assert(io_multiplicity > 0);

  char *buf_dev_head = reinterpret_cast<char*>(mb_cuda_alloc((io_multiplicity * Q11_LO_NUM_ACTIVE_FIELDS * nthreads + 1) * page_size));
  char *buf_dev_aligned = (char *)(((size_t)buf_dev_head + page_size - 1) & ~(page_size - 1));
  std::cout << "buf_dev_head: " << (void *)buf_dev_head << std::endl;
  std::cout << "buf_dev_aligned: " << (void *)buf_dev_aligned << std::endl;
  std::cout << "size_buf: " << (io_multiplicity * Q11_LO_NUM_ACTIVE_FIELDS * nthreads + 1) * page_size << std::endl;

  unsigned long long *revenue_dev = reinterpret_cast<unsigned long long*>(mb_cuda_alloc(sizeof(unsigned long long) * nthreads));
  for (size_t i = 0; i < nthreads; i++)
  {
      cudaMemset(revenue_dev + i, 0, sizeof(unsigned long long));
  }

  /* nthreads = std::min(npages_lo, options.nthreads) confirms that npages_lo / nthreads >= 1 */
  size_t tasks_per_thread_base = npages_lo / nthreads;
  size_t tasks_remainder = npages_lo % nthreads;
  std::vector<DeviceSyncQ11Args> thread_args_vec;
  thread_args_vec.reserve(nthreads);

  size_t tasks_start = 0;
  for (size_t i = 0; i < nthreads; i++)
  {
    CUfileBatchHandle_t batch_idp = nullptr;
    checkCuFileErrors(cuFileBatchIOSetUp(&batch_idp, io_multiplicity * Q11_LO_NUM_ACTIVE_FIELDS));

    std::array<std::vector<uint64_t>::iterator, Q11_LO_NUM_ACTIVE_FIELDS> pagid_vec_begin_arr;
    std::array<std::vector<uint64_t>::iterator, Q11_LO_NUM_ACTIVE_FIELDS> pagid_vec_end_arr;
    std::array<std::vector<char *>, Q11_LO_NUM_ACTIVE_FIELDS> buf_dev_vec_arr;
    size_t tasks_count_per_thread = tasks_per_thread_base + (i < tasks_remainder ? 1 : 0);
    for (size_t f = 0; f < Q11_LO_NUM_ACTIVE_FIELDS; f++)
    {
      auto &pagid_vec = pagid_vec_arr[f];
      auto begin = pagid_vec_arr[f].begin() + tasks_start;
      auto end = pagid_vec_arr[f].begin() + tasks_start + tasks_count_per_thread;
      pagid_vec_begin_arr[f] = begin;
      pagid_vec_end_arr[f] = end;

      buf_dev_vec_arr[f].resize(io_multiplicity);
      for (size_t j = 0; j < io_multiplicity; j++)
      {
        size_t offset = (i * Q11_LO_NUM_ACTIVE_FIELDS * io_multiplicity 
          + f * io_multiplicity + j) * page_size;
        //f * io_multiplicity * nthreads * page_size + (j * page_size);   
        char *buf = buf_dev_aligned + offset;
        buf_dev_vec_arr[f][j] = buf;
        mb_cufile_buf_register(buf, page_size);
        std::cout << "offset[" << i << "][" << f << "]["  << j << "]: " << offset << std::endl;   
        std::cout << "buf[" << i << "][" << f << "]["  << j << "]: " << static_cast<const void*>(buf) << std::endl;   
      }
      // std::cout << "buf_arr[" << f << "]: " << (void *)buf_arr[f] << std::endl;   
    }

    std::vector<CUfileIOParams_t> batch_params_vec(Q11_LO_NUM_ACTIVE_FIELDS * io_multiplicity);
    std::vector<CUfileIOEvents_t> batch_events_vec(Q11_LO_NUM_ACTIVE_FIELDS * io_multiplicity);
    // batch_params = mb_cufile_io_params_init(batch_idp, cufile_handles[0], options.io_multiplicity, page_size, 0);
    #if 0
    for (size_t j = 0; j < options.io_multiplicity; j++)
    {
        //void *buf_dev = (void*)mb_cuda_alloc_v2(options.io_size);
        size_t offset = (i * options.io_multiplicity + j) * options.io_size;
        void *buf_dev = (void *)(
            &reinterpret_cast<uint8_t*>(buf_dev_aligned)[offset]);

        mb_cufile_buf_register(buf_dev, options.io_size);
        #if 0
        std::cout << "buf_dev:" << buf_dev << std::endl;
        #endif

        buf_dev_vec[j] = buf_dev;
    }
    #endif
    tasks_start += tasks_count_per_thread;
    thread_args_vec.push_back(DeviceSyncQ11Args{
        .thrid = i,
        .ctx = ctx,
        .fds = fds,
        .pagid_vec_begin_arr = pagid_vec_begin_arr,
        .pagid_vec_end_arr = pagid_vec_end_arr,
        .buf_dev_vec_arr = buf_dev_vec_arr,
        .revenue_dev = &revenue_dev[i],
        .q11_sum = 0,
        .cufile_handles = cufile_handles,
        .batch_idp = batch_idp,
        .batch_params_vec = batch_params_vec,
        .batch_events_vec = batch_events_vec,
        .period_sec = options.period_sec,
        .stats_nio = 0,
        .nrows_processed = 0,
    });
  }

  /* Verification */
  size_t _verify_npages = 0;
  for (size_t i = 0; i < nthreads; i++)
  {
      auto &args = thread_args_vec[i];
      ssize_t distance = std::distance(
          args.pagid_vec_begin_arr[0],
          args.pagid_vec_end_arr[0]);
      _verify_npages += distance;
  }
  if (_verify_npages != npages_lo) {
      std::cerr << "[ERROR][BUG] Task allocation fails. npages_lo: " << npages_lo << " != " << _verify_npages << std::endl;
      exit(EXIT_FAILURE);
  }

  std::vector<std::thread> threads;
  threads.reserve(nthreads);

  auto start_cpu_usage = read_cpu_usage();
  auto start = chrono::system_clock::now();

  for (size_t i = 0; i < nthreads; i++)
  {
    DeviceSyncQ11Args &args = thread_args_vec[i];
    threads.emplace_back(
        [&options, &args, nthreads, page_size, io_multiplicity,
          nrows_page_int32, nrows_page_int64, nrows_lo]()
        {
          cpu_set_affinity(args.thrid);
          mb_cuda_set_context(args.ctx);
          CUfileOpcode_t opcode = CUFILE_READ;
          CUfileBatchHandle_t batch_idp = args.batch_idp;

          args.pagid_vec_begin_arr[0];
          args.pagid_vec_end_arr[0];
          size_t npages = std::distance(
              args.pagid_vec_begin_arr[0],
              args.pagid_vec_end_arr[0]);

          unsigned long long *revenue_dev = args.revenue_dev;

#if 0
          const size_t nstreams = io_multiplicity;
          std::vector<cudaStream_t> streams;
          for (size_t i = 0; i < nstreams; i++)
          {
            auto stream = mb_cuda_stream_create();
            streams.push_back(stream);
          }
#else
          // Using a single stream for now
          cudaStream_t stream = mb_cuda_stream_create();
#endif
          assert(args.buf_dev_vec_arr[0][0] != nullptr);
          assert(args.buf_dev_vec_arr[1][0] != nullptr);
          assert(args.buf_dev_vec_arr[2][0] != nullptr);
          assert(args.buf_dev_vec_arr[3][0] != nullptr);

          size_t i = 0;
          uint64_t time_start = gettime();
          while (i < npages)
          {
            size_t page_id_item1 = args.pagid_vec_begin_arr[0][i];
            size_t page_id_item2 = args.pagid_vec_begin_arr[1][i];
            size_t page_id_item3 = args.pagid_vec_begin_arr[2][i];
            size_t page_id_item4 = args.pagid_vec_begin_arr[3][i];

            CUfileIOParams_t *params1 = &args.batch_params_vec[0];
            CUfileIOParams_t *params2 = &args.batch_params_vec[1];
            CUfileIOParams_t *params3 = &args.batch_params_vec[2];
            CUfileIOParams_t *params4 = &args.batch_params_vec[3];

            uint64_t file_offset1 = page_id_item1 * options.io_size;
            uint64_t file_offset2 = page_id_item2 * options.io_size;
            uint64_t file_offset3 = page_id_item3 * options.io_size;
            uint64_t file_offset4 = page_id_item4 * options.io_size;

            size_t io_size = page_size;
            params1->mode = CUFILE_BATCH;
            params1->opcode = opcode;
            params1->fh = args.cufile_handles[0];
            params1->u.batch.devPtr_base = args.buf_dev_vec_arr[0][0];
            params1->u.batch.file_offset = file_offset1;
            params1->u.batch.devPtr_offset = 0;
            params1->u.batch.size = io_size;

            params2->mode = CUFILE_BATCH;
            params2->opcode = opcode;
            params2->fh = args.cufile_handles[0];
            params2->u.batch.devPtr_base = args.buf_dev_vec_arr[1][0];
            params2->u.batch.file_offset = file_offset2;
            params2->u.batch.devPtr_offset = 0;
            params2->u.batch.size = io_size;

            params3->mode = CUFILE_BATCH;
            params3->opcode = opcode;
            params3->fh = args.cufile_handles[0];
            params3->u.batch.devPtr_base = args.buf_dev_vec_arr[2][0];
            params3->u.batch.file_offset = file_offset3;
            params3->u.batch.devPtr_offset = 0;
            params3->u.batch.size = io_size;

            params4->mode = CUFILE_BATCH;
            params4->opcode = opcode;
            params4->fh = args.cufile_handles[0];
            params4->u.batch.devPtr_base = args.buf_dev_vec_arr[3][0];
            params4->u.batch.file_offset = file_offset4;
            params4->u.batch.devPtr_offset = 0;
            params4->u.batch.size = io_size;

            // std::cout << "page_id_item1: " << page_id_item1 << std::endl;
            // std::cout << "page_id_item2: " << page_id_item2 << std::endl;
            // std::cout << "page_id_item3: " << page_id_item3 << std::endl;
            // std::cout << "page_id_item4: " << page_id_item4 << std::endl;

            // page_pread_host(args.fds, args.buf_arr[0], page_id_item1, page_size);
            // page_pread_host(args.fds, args.buf_arr[1], page_id_item2, page_size);
            // page_pread_host(args.fds, args.buf_arr[2], page_id_item3, page_size);
            // page_pread_host(args.fds, args.buf_arr[3], page_id_item4, page_size);

            checkCuFileErrors(
              cuFileBatchIOSubmit(batch_idp, Q11_LO_NUM_ACTIVE_FIELDS, &args.batch_params_vec[0], 0)
            );
            args.stats_nio += Q11_LO_NUM_ACTIVE_FIELDS;

            unsigned nmax_requests = 4;
            checkCuFileErrors(
              cuFileBatchIOGetStatus(batch_idp, Q11_LO_NUM_ACTIVE_FIELDS,
                &nmax_requests, &args.batch_events_vec[0], nullptr));

            bool is_last_page = false;
            size_t nrows_page = nrows_page_int32;
            if (i == npages - 1 && args.thrid == nthreads - 1)
            {
              size_t nrows_final_page = nrows_lo % nrows_page_int32;
              is_last_page = true;
              nrows_page = nrows_final_page;
              // std::cout << "last_page: " << npages - 1 << std::endl;
              // std::cout << "nrows_final_page: " << nrows_final_page << std::endl;
            }
#if 1
            // Launch kernel here.
            q11_kernel(
                reinterpret_cast<uint32_t *>(args.buf_dev_vec_arr[0][0]),
                reinterpret_cast<uint32_t *>(args.buf_dev_vec_arr[1][0]),
                reinterpret_cast<uint32_t *>(args.buf_dev_vec_arr[2][0]),
                reinterpret_cast<uint32_t *>(args.buf_dev_vec_arr[3][0]),
                nrows_page,
                revenue_dev,
                is_last_page, stream);
#else
            int32_t *items1 = reinterpret_cast<int32_t *>(args.buf_arr[0]);
            int32_t *items2 = reinterpret_cast<int32_t *>(args.buf_arr[1]);
            int32_t *items3 = reinterpret_cast<int32_t *>(args.buf_arr[2]);
            int32_t *items4 = reinterpret_cast<int32_t *>(args.buf_arr[3]);

            size_t n = is_last_page ? nrows_final_page : nrows_page_int32;

            for (size_t j = 0; j < n; j++)
            {
              if (items1[j] > 19930000 && items1[j] < 19940000 && items2[j] < 25 && items3[j] >= 1 && items3[j] <= 3)
              {
                args.q11_sum += items4[j] * items3[j];
              }
            }

#endif
            cudaStreamSynchronize(stream);
            args.nrows_processed += nrows_page;
            i++;
          }
          cudaMemcpyAsync(
              reinterpret_cast<ulong*>(&args.q11_sum),
              args.revenue_dev,
              sizeof(ulong),
              cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          uint64_t time_end = gettime();
        }

    );
  }

  int64_t q11_sum = 0;
  size_t nrows_total_host = 0;
  for (size_t i = 0; i < nthreads; i++)
  {
    threads[i].join();
    q11_sum += thread_args_vec[i].q11_sum;
    nrows_total_host += thread_args_vec[i].nrows_processed;
  }

  auto end = chrono::system_clock::now();
  auto end_cpu_usage = read_cpu_usage();

  std::cout << "q11_sum: " << q11_sum << std::endl;
  std::cout << "nrows_total_host: " << nrows_total_host << std::endl;


  for (size_t i = 0; i < nthreads; i++)
  {
    auto& args = thread_args_vec[i];
    nios_issued = thread_args_vec[i].stats_nio;

    (void)cuFileBatchIODestroy(args.batch_idp);

    for (size_t j = 0; j < Q11_LO_NUM_ACTIVE_FIELDS; j++)
    {
        auto& buf_dev_vec = args.buf_dev_vec_arr[j];
        for (size_t k = 0; k < io_multiplicity; k++)
        {
            auto& buf_dev = buf_dev_vec[k];
            mb_cufile_buf_deregister(buf_dev);
        }
    }
  }
  mb_cuda_free(revenue_dev);
  mb_cuda_free(buf_dev_head);

  for (auto cufile_handle : cufile_handles)
  {
      mb_cufile_handle_deregister(cufile_handle);
  }
  mb_cufile_driver_close();

  close_files(options, fds);

  return BenchmarkResult{
      .nios = nios_issued,
      .elapsed_nanoseconds = (end - start).count(),
      .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
      .gpu_usage = GpuUsage{},
  };
}