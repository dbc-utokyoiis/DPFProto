#pragma once

struct  __attribute__ ((__packed__)) binpack_hdr
{
  union {
    struct {
      uint block_size;
      uint miniblock_count;
      uint total_count;
      uint dummy;
    };
    ulong dummy_long[2];
  };
  ulong first_val;
  ulong body[];
};

__forceinline__ __device__ int decodeElement(int i, uint miniblock_index, uint index_into_miniblock, uint* data_block, uint* bitwidths, uint* offsets) {
  // Reference for the frame
  int reference = reinterpret_cast<int*>(data_block)[0];

  uint miniblock_offset = offsets[miniblock_index];
  uint bitwidth = bitwidths[miniblock_index];

  uint start_bitindex = (bitwidth * index_into_miniblock);
  uint start_intindex = 2 + (start_bitindex >> 5);

  start_bitindex = start_bitindex & (32-1);

  unsigned long long element_block = (((unsigned long long)data_block[miniblock_offset + start_intindex + 1]) << 32) | data_block[miniblock_offset + start_intindex];
  uint element = (element_block >> start_bitindex) & ((1LL<<bitwidth) - 1LL);

#if 0
  //if (blockIdx.x == 0 && threadIdx.x < 16) {
  //if (element + reference == 424238335) {
    printf("[DEBUG] (%d, %d) i=%d miniblock_index=%d, index_into_miniblock=%d miniblock_offset=%x\n"
      "\tbitwidth=%x start_bitindex=%d start_intindex=%d\n"
      "\tidx1=%d idx0=%d\n"
      "\telement_block[1]=%x element_block[0]=%x element=%u val=%u\n",
      blockIdx.x, threadIdx.x, i, miniblock_index, index_into_miniblock, miniblock_offset,
      bitwidth, start_bitindex, start_intindex,
      miniblock_offset + start_intindex + 1, miniblock_offset + start_intindex,
      (uint)(element_block >> 32), (uint)element_block, element, element+reference);

  //}
#endif

  return reference + element;
}

__forceinline__ __device__ long decodeElement64(int i, ulong miniblock_index, ulong index_into_miniblock, ulong* data_block, uint* bitwidths, uint* offsets) {
  // Reference for the frame
  long reference = reinterpret_cast<long*>(data_block)[0];

  uint miniblock_offset = offsets[miniblock_index];
  // Convert 32bit-version offset to 64bit-version offset
  // if ((miniblock_offset << 3) % 8 == 0) {
  //   miniblock_offset >>= 1;
  // } else {
  //   miniblock_offset >>= 1;
  // }
  // miniblock_offset = ((miniblock_offset >> 2) + 1) << 2;
  // miniblock_offset = ((miniblock_offset >> 1) + 1) << 1;

  uint bitwidth = bitwidths[miniblock_index];

  ulong start_bitindex = (bitwidth * index_into_miniblock);
  // int version (32-bit) - divide by 32 because 32 bits per int
  // ulong start_intindex = 2 + (start_bitindex >> 5);
  // long version (64-bit) - divide by 64 (x >> 6) because 64 bits per int
  //   first 2 longs are for reference and bitwidths
  ulong start_intindex = 2 + (start_bitindex >> 6);

  start_bitindex = start_bitindex & (64-1);

  // if (start_bitindex > 0 && miniblock_index > 0) {
  //   miniblock_offset += 1;
  // }
  //if (miniblock_index > 0) {
  // miniblock_offset += miniblock_index;
  //}

  __uint128_t element_block =
    (((__uint128_t)data_block[miniblock_offset + start_intindex + 1]) << 64)
    | data_block[miniblock_offset + start_intindex];
  ulong element = (element_block >> start_bitindex) & ((1LL<<bitwidth) - 1LL);

  //if (blockIdx.x == 0 && threadIdx.x  16) {
  //if (miniblock_index == 1 && index_into_miniblock == 0) {
#if 0
  if (1) {
    //printf("[DEBUG] (%d, %d) i=%d miniblock_index=%ld, miniblock_offset=%x\n\tbitwidth=%x start_bitindex=%ld start_intindex=%ld\n\telement_block[1]=%lx element_block[0]=%lx element=%ld\n",
    //  blockIdx.x, threadIdx.x, i, miniblock_index, miniblock_offset, bitwidth, start_bitindex, start_intindex,
    //  (ulong)(element_block >> 64), (ulong)element_block, element);

    printf("[DEBUG] (%d, %d) i=%d miniblock_index=%ld, index_into_miniblock=%ld miniblock_offset=%x\n"
      "\tbitwidth=%x start_bitindex=%ld start_intindex=%ld\n"
      "\tidx1=%ld idx0=%ld\n"
      "\telement_block[1]=%lx element_block[0]=%lx element=%lu\n",
      blockIdx.x, threadIdx.x, i, miniblock_index, index_into_miniblock, miniblock_offset,
      bitwidth, start_bitindex, start_intindex,
      miniblock_offset + start_intindex + 1, miniblock_offset + start_intindex,
      (ulong)(element_block >> 64), (ulong)element_block, element);
  }
#endif

  return reference + element;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPack(uint* block_start,
    uint* data, uint* shared_buffer, int (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {
  int tile_idx = blockIdx.x;
  int threadId = threadIdx.x;

  // Block start indices of 5 blocks converted into integer offsets.
  uint *block_starts = &shared_buffer[0];
  if (threadId < ITEMS_PER_THREAD + 1) {
    block_starts[threadIdx.x] = block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x];
  }
#if 0
  if (tile_idx == 0 && threadId == 0) {
    printf("block_starts[%u]=%u (block_start[%u]=%u)\n",
      threadIdx.x, block_starts[threadIdx.x],
      tile_idx * ITEMS_PER_THREAD + threadIdx.x, block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x]);

    for (int i = 0; i < ITEMS_PER_THREAD + 1; i++) {
      printf("\tblock_starts[%u]=%u\n", i, block_starts[i]);
    }
  }
#endif
  __syncthreads();

  // Shared memory for 4 blocks of encoded l_shipdate data 
  // 5 + 32
  uint* data_block = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];

  // Lets load 4 blocks from the encoded column
  uint start_offset = block_starts[0];
  uint end_offset = block_starts[ITEMS_PER_THREAD];
  // printf("start_offset=%u end_offset=%u\n", start_offset, end_offset);
  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    uint index = start_offset + threadIdx.x + (i << 7); // i * 128
    if (index < end_offset)
      data_block[threadIdx.x + (i << 7)] = data[index];
  }
#if 0
  if (tile_idx == 0 && threadId == 0) {
    printf("data_block[%u]=%u end_offset=%u\n", threadIdx.x, data_block[threadIdx.x], end_offset);
  }
#endif
  __syncthreads();

  uint* bitwidths = &shared_buffer[ITEMS_PER_THREAD + 1];
  uint* offsets = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)];

  if (threadId < (ITEMS_PER_THREAD << 2)) {
    int i = threadId >> 2;
    int miniblock_index = threadId & 3;

    // Miniblock bitwidths
    uint miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);
    // miniblock 0: reference, 1: bitwidths

    // Miniblock bitwidth
    uint miniblock_offsets = (miniblock_bitwidths << 8) + (miniblock_bitwidths << 16) + (miniblock_bitwidths << 24);
    uint miniblock_offset = (miniblock_offsets >> (miniblock_index << 3)) & 255;
    uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 3)) & 255;

    offsets[threadId] = miniblock_offset;
    bitwidths[threadId] = bitwidth;
#if 0
    if (1) {
      // printf("(%u, %u) miniblock_offsets=%x (miniblock_offset=%x) bitwidth=%x\n",
      //   tile_idx, threadId, miniblock_offsets, miniblock_offset, bitwidth);
      printf("(%u, %u) miniblock_offsets=%x (miniblock_offset=%x, miniblock_index=%d) bitwidth=%x\n",
        tile_idx, threadId, miniblock_offsets, miniblock_offset, miniblock_index, bitwidth);
    }
#endif
  }
#if 0
  if (tile_idx == 0 && threadId == 0) {
    printf("offsets[%u]=%u bitwidths[%u]=0x%x\n", threadId, offsets[threadId], threadId, bitwidths[threadId]);
    printf("\toffsets[%u]=%x bitwidths[%u]=%x\n",
      threadId >> 2, block_starts[threadId >> 2], 0, block_starts[0]);
    // uint miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);
  }
#endif
  __syncthreads();

  // Index of miniblock containing i
  uint miniblock_index = threadIdx.x >> 5; // i / 32

  // Entry index in the miniblock
  uint index_into_miniblock = threadIdx.x & (32 - 1);

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    /*if (is_last_tile) {*/
      /*if (threadIdx.x + i*128 < num_tile_items) {*/
        /*items[i] = decodeElement(threadIdx.x, data_block + block_starts[i] - block_starts[0]);*/
      /*}*/
    /*}*/
    /*else {*/
      items[i] = decodeElement(threadIdx.x, miniblock_index, index_into_miniblock, data_block + block_starts[i] - block_starts[0], bitwidths + (i<<2), offsets + (i<<2));
    /*}*/
  }
}

// Variant of LoadBinPack with explicit tile_idx parameter (instead of blockIdx.x).
// Used by the BAM kernel where one CUDA block processes multiple tiles per page.
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPackTile(int tile_idx,
    uint* block_start, uint* data, uint* shared_buffer,
    int (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {
  int threadId = threadIdx.x;

  uint *block_starts = &shared_buffer[0];
  if (threadId < ITEMS_PER_THREAD + 1) {
    block_starts[threadIdx.x] = block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x];
  }
  __syncthreads();

  uint* data_block = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];

  uint start_offset = block_starts[0];
  uint end_offset = block_starts[ITEMS_PER_THREAD];
  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    uint index = start_offset + threadIdx.x + (i << 7);
    if (index < end_offset)
      data_block[threadIdx.x + (i << 7)] = data[index];
  }
  __syncthreads();

  uint* bitwidths = &shared_buffer[ITEMS_PER_THREAD + 1];
  uint* offsets = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)];

  if (threadId < (ITEMS_PER_THREAD << 2)) {
    int i = threadId >> 2;
    int miniblock_index = threadId & 3;

    uint miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);

    uint miniblock_offsets = (miniblock_bitwidths << 8) + (miniblock_bitwidths << 16) + (miniblock_bitwidths << 24);
    uint miniblock_offset = (miniblock_offsets >> (miniblock_index << 3)) & 255;
    uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 3)) & 255;

    offsets[threadId] = miniblock_offset;
    bitwidths[threadId] = bitwidth;
  }
  __syncthreads();

  uint miniblock_index = threadIdx.x >> 5;
  uint index_into_miniblock = threadIdx.x & (32 - 1);

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    items[i] = decodeElement(threadIdx.x, miniblock_index, index_into_miniblock, data_block + block_starts[i] - block_starts[0], bitwidths + (i<<2), offsets + (i<<2));
  }
}

#if 0
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPack64(uint* block_start,
    ulong* data, ulong* shared_buffer, long (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {
  int tile_idx = blockIdx.x;
  int threadId = threadIdx.x;

  printf("(%d, %d)\n", blockIdx.x, threadIdx.x);
  // if (tile_idx == 0 && threadId == 0) {
  //   printf("(%d, %d)\n", blockIdx.x, threadIdx.x);
  // }

  // Block start indices of 5 blocks converted into integer offsets.
  // BLOCK_THREADS = 128, ITEM_PER_THREAD = 4
  // uint *block_starts = &shared_buffer[0];
  uint *shared_buffer32 = reinterpret_cast<uint*>(shared_buffer);
  uint *block_starts = reinterpret_cast<uint*>(shared_buffer);
  if (threadId < ITEMS_PER_THREAD + 1) {
    block_starts[threadIdx.x] = block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x];
  }
  __syncthreads();
  #if 0
  // === DEBUG ===
  if (tile_idx == 0 && threadId == 0) {
    printf("shared_buffer info\n");
    printf("\tshared_buffer32=%lx\n", shared_buffer32);
    printf("\tshared_buffer=%lx\n", shared_buffer);
    printf("\t\t(for offsets) shared_buffer[37]=%lx\n",
      &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)]);
    printf("\t\t(for bitwidths) shared_buffer[ITEMS_PER_THREAD + 1]=%lx\n",
      &shared_buffer[ITEMS_PER_THREAD + 1]);
    printf("\tblock_starts=%lx\n", block_starts);
    printf("\t\tblock_starts[ITEMS_PER_THREAD << 3]=%lx\n", &shared_buffer[ITEMS_PER_THREAD << 3]);
  }
  __syncthreads();
  #endif

  #if 0
  if (tile_idx == 0 && threadId == 0) {
    printf("block_starts[%u]=%u (block_start[%u]=%u)\n",
      threadIdx.x, block_starts[threadIdx.x],
      tile_idx * ITEMS_PER_THREAD + threadIdx.x, block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x]);

    // for (int i = 0; i < ITEMS_PER_THREAD + 1; i++) {
    //   block_starts[i] = block_start[i];
    //   printf("\tblock_starts[%u]=%u block_start[%u]=%u block_starts[%d]=%lx\n",
    //     i, block_starts[i], i, block_start[i], i, &block_starts[i]);
    // }
    printf("\tblock_starts[%u]=%u shared_buffer=%lx\n",
      ITEMS_PER_THREAD, block_starts[ITEMS_PER_THREAD], shared_buffer);
  }
  __syncthreads();
  #endif
  // === DEBUG ===

  // This is just a pointer for the region of the data block.
  // The first 5 elements are used for storing block_starts[0] - block_starts[4].
  // The next 32 elements are used for storing bitwidths and offsets.
  // 32 + 5 = 37 is the start of the data block.
  ulong* data_block = reinterpret_cast<ulong*>(&shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)]);
  // struct binpack_hdr *hdr = (struct binpack_hdr*)data_block;

  // Lets load 4 blocks from the encoded column
  uint start_offset = block_starts[0];
  uint end_offset = block_starts[ITEMS_PER_THREAD];
  if (tile_idx == 0 && threadId == 0) {
    printf("1: data_block=%lx start_offset=%u end_offset=%u, block_starts[%u]=%u (%lx) \n",
      data_block, start_offset, end_offset, ITEMS_PER_THREAD, block_starts[ITEMS_PER_THREAD],
      &block_starts[ITEMS_PER_THREAD]);
  }

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    uint index = start_offset + threadIdx.x + (i << 7); // i * 128
    if (index < end_offset)
      data_block[threadIdx.x + (i << 7)] = data[index];
  }
  __syncthreads();
  if (tile_idx == 0 && threadId == 0) {
    if (ITEMS_PER_THREAD < 4) {
      printf("2: start to load data array to shared_buffer.\n");
      //for (int i = 0; i < 128 + 2; i++) {
      // for (int i = 0; i < end_offset; i++) {
      //   uint index = start_offset + threadIdx.x + (i << 7); // i * 128
      //   if (index < end_offset) {
      //     data_block[index + (i << 7)] = data[index];
      //     printf("\tdata_block[%u]=%lu data[%u]=%lu\n",
      //       threadIdx.x, data_block[threadIdx.x], index, data[index]);
      //   }
      // }
    }
    printf("3: data_block[%u]=%lu end_offset=%u\n",
      threadIdx.x, data_block[threadIdx.x], end_offset);
    // printf("\tdata_block[%u]=%lu end_offset=%u\n",
    //   threadIdx.x + (1 << 7), data_block[threadIdx.x + (1 << 7)], end_offset);
    // printf("\tdata_block[%u]=%lu end_offset=%u\n",
    //   threadIdx.x + (2 << 7), data_block[threadIdx.x + (2 << 7)], end_offset);
    // printf("\tdata_block[%u]=%lu end_offset=%u\n",
    //   threadIdx.x + (3 << 7), data_block[threadIdx.x + (3 << 7)], end_offset);
  }
  __syncthreads();

  uint* bitwidths = &shared_buffer32[ITEMS_PER_THREAD + 1];
  uint* offsets = reinterpret_cast<uint*>(&shared_buffer32[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)]);

  if (threadId < (ITEMS_PER_THREAD << 2)) {
    int i = threadId >> 2;
    int miniblock_index = threadId & 3;

    // Miniblock bitwidths
    uint miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);
    // ulong miniblock_offsets = (miniblock_bitwidths << 8) + (miniblock_bitwidths << 16) + (miniblock_bitwidths << 24);
    uint miniblock_offsets = (uint)((miniblock_bitwidths << 8) + (miniblock_bitwidths << 16) + (miniblock_bitwidths << 24));
    uint miniblock_offset = (miniblock_offsets >> (miniblock_index << 3)) & 255;
    uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 3)) & 255;

    offsets[threadId] = miniblock_offset;
    bitwidths[threadId] = bitwidth;

    // if (tile_idx == 0 && threadId == 0) {
    if (1) {
      printf("(%u, %u) miniblock_offsets=%x (miniblock_offset=%x, miniblock_index=%d) bitwidth=%x\n",
        tile_idx, threadId, miniblock_offsets, miniblock_offset, miniblock_index, bitwidth);
    }
  }
  __syncthreads();
#if 1
  if (tile_idx == 0 && threadId == 0) {
    printf("offsets[%u]=%u bitwidths[%u]=%x\n", threadId, offsets[threadId], threadId, bitwidths[threadId]);
  }
#endif

  // Index of miniblock containing i
  ulong miniblock_index = threadIdx.x >> 5; // i / 32

  // Entry index in the miniblock
  ulong index_into_miniblock = threadIdx.x & (32 - 1);

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    /*if (is_last_tile) {*/
      /*if (threadIdx.x + i*128 < num_tile_items) {*/
        /*items[i] = decodeElement(threadIdx.x, data_block + block_starts[i] - block_starts[0]);*/
      /*}*/
    /*}*/
    /*else {*/
      #if 1
      items[i] = decodeElement64(threadIdx.x, miniblock_index, index_into_miniblock, data_block + block_starts[i] - block_starts[0], bitwidths + (i<<2), offsets + (i<<2));
      #else
      items[i] = 0;
      #endif
    /*}*/
  }
}
#else
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPack64(uint* block_start,
    ulong* data, ulong* shared_buffer, long (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {
  int tile_idx = blockIdx.x;
  int threadId = threadIdx.x;

  // Block start indices of 5 blocks converted into integer offsets.
  uint *shared_buffer32 = reinterpret_cast<uint*>(shared_buffer);
  uint *block_starts = &shared_buffer32[0];
  if (threadId < ITEMS_PER_THREAD + 1) {
    block_starts[threadIdx.x] = block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x];
  }
#if 0
  if (tile_idx == 0 && threadId == 0) {
    printf("block_starts[%u]=%u (block_start[%u]=%u)\n",
      threadIdx.x, block_starts[threadIdx.x],
      tile_idx * ITEMS_PER_THREAD + threadIdx.x, block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x]);

    for (int i = 0; i < ITEMS_PER_THREAD + 1; i++) {
      printf("\tblock_starts[%u]=%u\n", i, block_starts[i]);
    }
  }
#endif
  __syncthreads();

  // Shared memory for 4 blocks of encoded l_shipdate data 
  // 5 + 32
  ulong* data_block = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];

  // Lets load 4 blocks from the encoded column
  uint start_offset = block_starts[0];
  uint end_offset = block_starts[ITEMS_PER_THREAD];
  // printf("start_offset=%u end_offset=%u\n", start_offset, end_offset);
  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    uint index = start_offset + threadIdx.x + (i << 7); // i * 128
    if (index < end_offset)
      data_block[threadIdx.x + (i << 7)] = data[index];
  }
#if 0
  if (tile_idx == 0 && threadId == 0) {
    printf("data_block[%u]=%u end_offset=%u\n", threadIdx.x, data_block[threadIdx.x], end_offset);
  }
#endif
  __syncthreads();

  uint* bitwidths = &shared_buffer32[ITEMS_PER_THREAD + 1];
  uint* offsets = &shared_buffer32[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)];

  if (threadId < (ITEMS_PER_THREAD << 2)) {
    int i = threadId >> 2;
    int miniblock_index = threadId & 3;

    // Miniblock bitwidths
    uint miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);
    // miniblock 0: reference, 1: bitwidths

    // uint miniblock_offset_base = ((miniblock_bitwidths >> 3 + 1)) << 3;
    // if (miniblock_bitwidths % 8 == 0) {
    //   miniblock_offset_base = miniblock_bitwidths;
    // }  else {
    //   miniblock_bitwidths = miniblock_bitwidths >> 3
    // }

    // uint bw = ((((0xff0000 & miniblock_bitwidths) >> (16 + 3)) + 1) << (16 + 3))
    //   + ((((0x00ff00 & miniblock_bitwidths) >> (8 + 3)) + 1) << (8 + 3))
    //   + (((0x0000ff & miniblock_bitwidths) >> 3) + 1) << 3;

#if 1
    // uint bw =
    //   ((((0xff0000 & miniblock_bitwidths) >> (16+3)) + 1) << (16 + 3))
    //   + ((((0x00ff00 & miniblock_bitwidths) >> (8+3)) + 1) << (8 + 3))
    //   + ((((0x0000ff & miniblock_bitwidths) >> 3) + 1) << 3);
    // printf("[] bw:%x %x %x %x %x %x %x\n", bw,
    //   0xff0000 & miniblock_bitwidths >> 16, (((0xff0000 & miniblock_bitwidths) >> (16+3)) + 1) << (16 + 3),
    //   0x00ff00 & miniblock_bitwidths >> 8, (((0x00ff00 & miniblock_bitwidths) >> (8+3)) + 1) << (8 + 3),
    //   0x0000ff & miniblock_bitwidths, (((0x0000ff & miniblock_bitwidths) >> 3) + 1) << 3);
    // uint miniblock_offsets = (bw << 8) + (bw << 16) + (bw << 24);
    // When 64 bit elements are processed, the modulo of 8 bytes is 0 or 4 bytes.
    // If the module is 4, the offset should be carried up to 8 bytes.
    uint shift = 2;
    uint bw =
      ((((0xff0000 & miniblock_bitwidths) >> (16+shift)) + 1) << (16 + shift - 1))
      + ((((0x00ff00 & miniblock_bitwidths) >> (8+shift)) + 1) << (8 + shift - 1))
      + ((((0x0000ff & miniblock_bitwidths) >> shift) + 1) << shift - 1);
#if 0
    printf("[] bw:%x %x %x %x %x %x %x\n", bw,
      0xff0000 & miniblock_bitwidths >> 16, (((0xff0000 & miniblock_bitwidths) >> (16+shift)) + 1) << (16 + shift),
      0x00ff00 & miniblock_bitwidths >> 8, (((0x00ff00 & miniblock_bitwidths) >> (8+shift)) + 1) << (8 + shift),
      0x0000ff & miniblock_bitwidths, (((0x0000ff & miniblock_bitwidths) >> shift) + 1) << shift);
#endif
    uint miniblock_offsets = (bw << 8) + (bw << 16) + (bw << 24);
#else
    uint miniblock_offsets = (miniblock_bitwidths << 8) + (miniblock_bitwidths << 16) + (miniblock_bitwidths << 24);
#endif

    // Miniblock bitwidth
    // uint miniblock_offsets = (miniblock_bitwidths << 8) + (miniblock_bitwidths << 16) + (miniblock_bitwidths << 24);
    uint miniblock_offset = (miniblock_offsets >> (miniblock_index << 3)) & 255;
    // miniblock_offset = ((miniblock_offset >> 3) + 1) << 3;
    // if (miniblock_index > 0 && miniblock_offset % 8 == 0) {
    // } else {
    //   miniblock_offset = miniblock_offset + 1;
    // }

    uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 3)) & 255;

    offsets[threadId] = miniblock_offset;
    bitwidths[threadId] = bitwidth;
#if 0
    if (1) {
      // printf("(%u, %u) miniblock_offsets=%x (miniblock_offset=%x) bitwidth=%x\n",
      //   tile_idx, threadId, miniblock_offsets, miniblock_offset, bitwidth);
      printf("(%u, %u) miniblock_offsets=%x (miniblock_offset=%x, miniblock_index=%d) bitwidth=%x\n",
        tile_idx, threadId, miniblock_offsets, miniblock_offset, miniblock_index, bitwidth);
    }
#endif
  }
#if 0
  if (tile_idx == 0 && threadId == 0) {
    printf("offsets[%u]=%u bitwidths[%u]=0x%x\n", threadId, offsets[threadId], threadId, bitwidths[threadId]);
    printf("\toffsets[%u]=%x bitwidths[%u]=%x\n",
      threadId >> 2, block_starts[threadId >> 2], 0, block_starts[0]);
    // uint miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);
  }
#endif
  __syncthreads();

  // Index of miniblock containing i
  uint miniblock_index = threadIdx.x >> 5; // i / 32

  // Entry index in the miniblock
  uint index_into_miniblock = threadIdx.x & (32 - 1);

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    /*if (is_last_tile) {*/
      /*if (threadIdx.x + i*128 < num_tile_items) {*/
        /*items[i] = decodeElement(threadIdx.x, data_block + block_starts[i] - block_starts[0]);*/
      /*}*/
    /*}*/
    /*else {*/
      items[i] = decodeElement64(threadIdx.x, miniblock_index, index_into_miniblock, data_block + block_starts[i] - block_starts[0], bitwidths + (i<<2), offsets + (i<<2));
    /*}*/
  }
}
#endif

// Variant of LoadBinPack64 with explicit tile_idx parameter (instead of blockIdx.x).
// Used by the BAM kernel where one CUDA block processes multiple tiles per page.
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPackTile64(int tile_idx,
    uint* block_start, ulong* data, ulong* shared_buffer,
    long (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {
  int threadId = threadIdx.x;

  uint *shared_buffer32 = reinterpret_cast<uint*>(shared_buffer);
  uint *block_starts = &shared_buffer32[0];
  if (threadId < ITEMS_PER_THREAD + 1) {
    block_starts[threadIdx.x] = block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x];
  }
  __syncthreads();

  ulong* data_block = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];

  uint start_offset = block_starts[0];
  uint end_offset = block_starts[ITEMS_PER_THREAD];
  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    uint index = start_offset + threadIdx.x + (i << 7);
    if (index < end_offset)
      data_block[threadIdx.x + (i << 7)] = data[index];
  }
  __syncthreads();

  uint* bitwidths = &shared_buffer32[ITEMS_PER_THREAD + 1];
  uint* offsets = &shared_buffer32[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)];

  if (threadId < (ITEMS_PER_THREAD << 2)) {
    int i = threadId >> 2;
    int miniblock_index = threadId & 3;

    uint miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);

    // 64-bit miniblock offset: each miniblock of 32 values × bw bits
    // occupies ceil(bw/2) 64-bit words (host packs per-miniblock with
    // word-boundary reset between miniblocks).
    uint bw0 = miniblock_bitwidths & 0xFF;
    uint bw1 = (miniblock_bitwidths >> 8) & 0xFF;
    uint bw2 = (miniblock_bitwidths >> 16) & 0xFF;
    uint h0 = (bw0 + 1) >> 1;
    uint h1 = (bw1 + 1) >> 1;
    uint h2 = (bw2 + 1) >> 1;
    uint miniblock_offsets = (h0 << 8) | ((h0 + h1) << 16) | ((h0 + h1 + h2) << 24);

    uint miniblock_offset = (miniblock_offsets >> (miniblock_index << 3)) & 255;
    uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 3)) & 255;

    offsets[threadId] = miniblock_offset;
    bitwidths[threadId] = bitwidth;
  }
  __syncthreads();

  uint miniblock_index = threadIdx.x >> 5;
  uint index_into_miniblock = threadIdx.x & (32 - 1);

  for (int i=0; i<ITEMS_PER_THREAD; i++) {
    items[i] = decodeElement64(threadIdx.x, miniblock_index, index_into_miniblock, data_block + block_starts[i] - block_starts[0], bitwidths + (i<<2), offsets + (i<<2));
  }
}