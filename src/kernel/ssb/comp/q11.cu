#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

// constexpr unsigned int NUM_THREADS_PER_BLOCK = 512U;
#define ENABLE_Q11_ENCODING
/**
 * Globals, constants and typedefs
 */
// bool                    g_verbose = false;  // Whether to display input/output to console
// CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

// NOTE: BLOCK_THREADS is a number of threads per thread block (128).
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void Q11CompKernel(
    uint* lo_orderdate_block_start, uint* lo_orderdate_data,
    uint* lo_quantity_block_start, uint* lo_quantity_data,
    uint* lo_discount_block_start, uint* lo_discount_data,
    uint* lo_extendedprice_block_start, uint* lo_extendedprice_data,
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
#ifdef ENABLE_Q11_ENCODING
  // RENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(
  //     lo_orderdate_val_block_start, lo_orderdate_rl_block_start, lo_orderdate_val_data, lo_orderdate_rl_data,
  //     temp_storage.shared_buffer, items, items2, is_last_tile, num_tile_items);
  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_orderdate_block_start, lo_orderdate_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);
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

#ifdef ENABLE_Q11_ENCODING
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

#ifdef ENABLE_Q11_ENCODING
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

#ifdef ENABLE_Q11_ENCODING
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

  // ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_discount_block_start, lo_discount_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);
  //__syncthreads();
  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_orderdate_block_start, lo_orderdate_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);
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

    if (selection_flags[ITEM])
      printf("[OK]items[%d] = %d\n", ITEM, items[ITEM]);
    else
      printf("[NG]items[%d] = %d\n", ITEM, items[ITEM]);
#endif
  }
  __syncthreads();

#if 0
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

  unsigned long long aggregate = BlockReduceInt(temp_storage.reduce).Sum(sum);
  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
#endif
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void Q11CompCountKernel(
  uint* lo_orderdate_block_start, uint* lo_orderdate_data,
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
  long long sum = 0;

  int num_tiles = (lo_num_entries + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (contain_last_tile) {
    if (tile_idx == num_tiles - 1) {
      num_tile_items = lo_num_entries - tile_offset;
      is_last_tile = true;
    }
  }
 
  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_orderdate_block_start, lo_orderdate_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    selection_flags[ITEM] = 1;
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      printf("items[%d] = %d\n", ITEM, items[ITEM]); 
  }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += 1;
  }
  __syncthreads();

  unsigned long long aggregate = BlockReduceInt(temp_storage.reduce).Sum(sum);
  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }

  return;
}

extern "C" cudaError_t q11_comp_kernel(
    uint* lo_orderdate_block_start, uint* lo_orderdate_data, 
    uint* lo_quantity_block_start, uint* lo_quantity_data,
    uint* lo_discount_block_start, uint* lo_discount_data,
    uint* lo_extendedprice_block_start, uint* lo_extendedprice_data,
    int lo_num_entries, unsigned long long* revenue,
    const int contain_last_tile, cudaStream_t stream) {
  constexpr int block_threads = 128;
  constexpr int items_per_thread = 4;
  constexpr unsigned int tile_size = block_threads * items_per_thread;

  // auto [grid_dim, block_dim] = get_dims(lo_num_entries);
  // unsigned int block_dim = std::min(static_cast<unsigned int>(lo_num_entries), tile_size);
  // unsigned int grid_dim = (static_cast<unsigned int>(lo_num_entries) + block_dim - 1) / block_dim;
  // return {grid_dim, block_dim};

  #if 1
  Q11CompKernel<block_threads, items_per_thread><<<(lo_num_entries + tile_size - 1)/tile_size, block_threads>>>(
      lo_orderdate_block_start, lo_orderdate_data,
      lo_quantity_block_start, lo_quantity_data,
      lo_discount_block_start, lo_discount_data,
      lo_extendedprice_block_start, lo_extendedprice_data,
      lo_num_entries, revenue, contain_last_tile);
  #else
  Q11CompCountKernel<block_threads, items_per_thread><<<(lo_num_entries + tile_size - 1)/tile_size, block_threads>>>(
      lo_orderdate_block_start, lo_orderdate_data,
      lo_num_entries, revenue, contain_last_tile);

  #endif

  return cudaSuccess;
}

// __global__ void PrintKernel(uint *offsets)
// {
//     // printf("offset: %x\n", offsets);
//     printf("offsets[0]: %x\n", offsets[0]);
//     printf("offsets[1]: %x\n", offsets[1]);
//     printf("offsets[2]: %x\n", offsets[2]);
//     printf("offsets[3]: %x\n", offsets[3]);
// }
// 
// extern "C" void print_offset_value(uint *offsets) {
//     PrintKernel<<<1, 1>>>(offsets);
//     cudaDeviceSynchronize();
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Error in PrintKernel: %s\n", cudaGetErrorString(err));
//     }
// }
