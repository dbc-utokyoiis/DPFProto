#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

// constexpr unsigned int NUM_THREADS_PER_BLOCK = 512U;

/**
 * Globals, constants and typedefs
 */
// bool                    g_verbose = false;  // Whether to display input/output to console
// CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

// NOTE: BLOCK_THREADS is a number of threads per thread block (128).
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void Q11Kernel(
    uint* lo_orderdate,
    uint* lo_quantity,
    uint* lo_discount,
    uint* lo_extendedprice,
    int lo_num_entries, unsigned long long* revenue, int contain_last_tile) {
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
#if 1
  bool is_last_tile = false;
  if (contain_last_tile) {
    if (tile_idx == num_tiles - 1) {
      num_tile_items = lo_num_entries - tile_offset;
      is_last_tile = true;
    }
  }
#else
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
    is_last_tile = true;
  }
#endif

#if 1
#ifdef ENABLE_ENCODING
  RENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(
      lo_orderdate_val_block_start, lo_orderdate_rl_block_start, lo_orderdate_val_data, lo_orderdate_rl_data,
      temp_storage.shared_buffer, items, items2, is_last_tile, num_tile_items);
#else
  // Load the lo_orderdate into registers (items[ITEM])
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      items[ITEM] = lo_orderdate[tile_idx * tile_size + threadIdx.x + (BLOCK_THREADS * ITEM)];
  }
#endif

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

#ifdef ENABLE_ENCODING
  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_quantity_block_start, lo_quantity_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);
#else
  // Load the lo_quantity into registers (items[ITEM])
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      items[ITEM] = lo_quantity[tile_idx * tile_size + threadIdx.x + (BLOCK_THREADS * ITEM)];
  }
#endif
  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      selection_flags[ITEM] = selection_flags[ITEM] && items[ITEM] < 25;
  }

  __syncthreads();

#ifdef ENABLE_ENCODING
  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_discount_block_start, lo_discount_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);

#else
  // Load the lo_quantity into registers (items[ITEM])
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      items[ITEM] = lo_discount[tile_idx * tile_size + threadIdx.x + (BLOCK_THREADS * ITEM)];
  }
#endif
  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      selection_flags[ITEM] = selection_flags[ITEM] && items[ITEM] >= 1 && items[ITEM ] <= 3;
  }

  __syncthreads();

#ifdef ENABLE_ENCODING
  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_extendedprice_block_start, lo_extendedprice_data, temp_storage.shared_buffer, items2, is_last_tile, num_tile_items);
#else
  // Load the lo_quantity into registers (items2[ITEM])
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      items2[ITEM] = lo_extendedprice[tile_idx * tile_size + threadIdx.x + (BLOCK_THREADS * ITEM)];
  }
#endif
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
#else
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      items[ITEM] = lo_orderdate[tile_idx * tile_size + threadIdx.x + (BLOCK_THREADS * ITEM)];
  }
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
#if 0
    // count(*) --> OK
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      selection_flags[ITEM] = 1;
#else
    // count(*) where lo_orderdate > 19930000 and lo_orderdate < 19940000 --> OK
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
    selection_flags[ITEM] = 1;
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      selection_flags[ITEM] = (items[ITEM] > 19930000 && items[ITEM] < 19940000); 
#endif
  }
  __syncthreads();

#if 1
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      items[ITEM] = lo_quantity[tile_idx * tile_size + threadIdx.x + (BLOCK_THREADS * ITEM)];
  }
  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      selection_flags[ITEM] = selection_flags[ITEM] && (items[ITEM] > 25);
  }
  __syncthreads();
#endif
 
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += 1;
  }
  __syncthreads();

  // カーネルの立ち上げ数が間違ってそう？ --> そんなことはなかった。kernel 内部に不具合がありそう
  unsigned long long aggregate = BlockReduceInt(temp_storage.reduce).Sum(sum);
  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
#endif
}

extern "C" cudaError_t q11_kernel(
    uint* lo_orderdate, 
    uint* lo_quantity,
    uint* lo_discount,
    uint* lo_extendedprice,
    int lo_num_entries, unsigned long long* revenue,
    const int contain_last_tile, cudaStream_t stream) {
  constexpr int block_threads = 128;
  constexpr int items_per_thread = 4;
  constexpr unsigned int tile_size = block_threads * items_per_thread;

  // auto [grid_dim, block_dim] = get_dims(lo_num_entries);
  // unsigned int block_dim = std::min(static_cast<unsigned int>(lo_num_entries), tile_size);
  // unsigned int grid_dim = (static_cast<unsigned int>(lo_num_entries) + block_dim - 1) / block_dim;
  // return {grid_dim, block_dim};

  Q11Kernel<block_threads, items_per_thread><<<(lo_num_entries + tile_size - 1)/tile_size, block_threads>>>(
  //Q11Kernel<block_threads, items_per_thread><<<grid_dim, block_dim>>>(
      lo_orderdate,
      lo_quantity,
      lo_discount,
      lo_extendedprice,
      lo_num_entries, revenue, contain_last_tile);

  return cudaSuccess;
}