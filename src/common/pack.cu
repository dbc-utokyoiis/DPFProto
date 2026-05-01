#pragma once

#include <iostream>
#include <stdint.h>

#if 0
struct binpack_hdr64 {
  ulong block_size;
  ulong miniblock_count;
  ulong total_count;
  ulong body[];
};
#endif

// #define ENABLE_PACK_DEBUG 

static int delta(int*& in, int*& out, int num_entries) {
  for (int i = num_entries - 1; i > 0; i--) {
    out[i] = in[i] - in[i-1];
  }

  out[0] = 0;
  return 0;
}

uint binPack(uint*&in, uint*& out, uint*& block_offsets, uint num_entries, uint *workspace32) {
  uint offset = 0;

  uint block_size = 128;
  uint miniblock_count = 4;
  uint total_count = num_entries;
  uint first_val = in[0];

  out[0] = block_size;
  out[1] = miniblock_count;
  out[2] = total_count;
  out[3] = first_val;

  offset += 4;

#ifdef ENABLE_PACK_DEBUG
  uint miniblock_size = uint(block_size / miniblock_count);
  uint num_blocks = (num_entries + block_size - 1) / block_size;
#endif

  for (uint block_start=0; block_start<num_entries; block_start += block_size) {
    uint block_index = block_start / block_size;
    block_offsets[block_index] = offset;

    // Find min val
    uint min_val = in[0];
    for (int i = 1; i < block_size; i++) {
      if (in[i] < min_val) min_val = in[i];
    }

    for (int i = 0; i < block_size; i++) {
      in[i] = in[i] - min_val;
    }

    uint miniblock_size = block_size / miniblock_count;
    // uint* miniblock_bitwidths = new uint[miniblock_count];
    uint *miniblock_bitwidths = workspace32;
    for (int i=0; i<miniblock_count; i++) miniblock_bitwidths[i] = 0;

    for (uint miniblock = 0; miniblock < miniblock_count; miniblock++) {
      for (uint i = 0; i < miniblock_size; i++) {
        uint bitwidth = uint(ceil(log2(in[miniblock * miniblock_size + i] + 1)));
        if (bitwidth > miniblock_bitwidths[miniblock]) miniblock_bitwidths[miniblock] = bitwidth;
      }
    }

    // Extra for Simple BinPack
    uint max_bitwidth = miniblock_bitwidths[0];
    for (int i=1; i<miniblock_count; i++) max_bitwidth = max(max_bitwidth, miniblock_bitwidths[i]);
    for (int i=0; i<miniblock_count; i++) miniblock_bitwidths[i] = max_bitwidth;
#ifdef ENABLE_PACK_DEBUG 
    if (block_start == 0) std::cout << "max_bitwidth " << max_bitwidth << std::endl;
#endif

    out[offset] = min_val;
    offset++;

    out[offset] = miniblock_bitwidths[0] + (miniblock_bitwidths[1] << 8) +
      (miniblock_bitwidths[2] << 16) + (miniblock_bitwidths[3] << 24);
    offset++;

    for (int miniblock = 0; miniblock < miniblock_count; miniblock++) {
      uint bitwidth = miniblock_bitwidths[miniblock];
      uint shift = 0;
      for (int i = 0; i < miniblock_size; i++) {
        if (shift + bitwidth > 32) {
          if (shift != 32) out[offset] += in[miniblock * miniblock_size + i] << shift;
          offset++;
          shift = (shift + bitwidth) & (32-1);
          out[offset] = in[miniblock * miniblock_size + i] >> (bitwidth - shift);
        } else {
          out[offset] += in[miniblock * miniblock_size + i] << shift;
          shift += bitwidth;
        }
      }
      offset++;
    }

    // Increment the input pointer by block size
    in += block_size;
  }

  block_offsets[num_entries / block_size] = offset;
  #if 0
  std::cout << " ==== " << std::endl;
  for (uint i = 0; i < offset; i++) {
    if (i > 0 && i % 8 == 0) std::cout << std::endl;
    std::cout << out[i] << " ";
  }
  std::cout << " ==== " << std::endl;
  #endif

  return offset;
}

uint binPack64(ulong*&in, ulong* out, uint*& block_offsets, uint num_entries, uint *workspacke32) {
  bool constexpr enable_debug_print = false;
  uint offset = 0;

  uint block_size = 128;
  uint miniblock_count = 4;
  uint total_count = num_entries;
  ulong first_val = in[0];

  /* Keeping the header size as uint */
#if 1
  out[0] = block_size;
  out[1] = miniblock_count;
  out[2] = total_count;
  out[3] = first_val;
#else
  out_hdr->block_size = block_size;
  out_hdr->miniblock_count = miniblock_count;
  out_hdr->total_count = total_count;
  ulong *out = out_hdr->body;
  out[0] = first_val;
#endif

#ifdef ENABLE_PACK_DEBUG 
  std::cout << "block_size " << block_size << " miniblock_count " << miniblock_count << " total_count " << total_count << " first_val " << first_val << std::endl;
#endif

  // start from 4
  offset = 4;

#ifdef ENABLE_PACK_DEBUG
  // miniblock_size is 32 by default (128 / 4)
  uint miniblock_size = uint(block_size / miniblock_count);
  // adjusted_len is the multiple of tile_size (512 records). 
  // block size is 128, so 512 is the multiple of 128.
  uint num_blocks = (num_entries + block_size - 1) / block_size;
#endif

  for (uint block_start=0; block_start<num_entries; block_start += block_size) {
    uint block_index = block_start / block_size;
    block_offsets[block_index] = offset;
#if 0
    printf("new block offset=%u\n", offset);
#endif

    // Find min val
    ulong min_val = in[0];
    for (int i = 1; i < block_size; i++) {
      if (in[i] < min_val) min_val = in[i];
    }

    // Use min val as the reference value of the block
    for (int i = 0; i < block_size; i++) {
      in[i] = in[i] - min_val;
    }

    // mini block size is 32, by default (128 / 4).
    uint miniblock_size = block_size / miniblock_count;
    //uint* miniblock_bitwidths = new uint[miniblock_count];
    uint* miniblock_bitwidths = workspacke32;
    for (int i=0; i<miniblock_count; i++) miniblock_bitwidths[i] = 0;

    // Calculate required bidwidth per miniblock
    for (uint miniblock = 0; miniblock < miniblock_count; miniblock++) {
      for (uint i = 0; i < miniblock_size; i++) {
        // uint bitwidth = uint(ceil(log2(in[miniblock * miniblock_size + i] + 1)));
        uint bitwidth = std::bit_width(in[miniblock * miniblock_size + i]);
        if (bitwidth > miniblock_bitwidths[miniblock]) miniblock_bitwidths[miniblock] = bitwidth;
      }
    }

    // Extra for Simple BinPack
    uint max_bitwidth = miniblock_bitwidths[0];
    for (int i=1; i<miniblock_count; i++) max_bitwidth = max(max_bitwidth, miniblock_bitwidths[i]);
    for (int i=0; i<miniblock_count; i++) miniblock_bitwidths[i] = max_bitwidth;
#ifdef ENABLE_PACK_DEBUG
    if (block_start == 0) std::cout << "max_bitwidth " << max_bitwidth << std::endl;
#endif

    // Store the reference value of the block
    out[offset] = min_val;
#if 0
    printf("[refval] out[%u]=%lx\n", offset, out[offset]);
#endif
    offset++;

    // NOTE: the maximum bitwidth is just 64, so 8 bits are enough to store the bitwidth of each miniblock
    // TODO: we can use 8 mini blocks, but we use 4 mini blocks currently(8 * 4).
#if 0
    out[offset] = 0;
    for (int i=0; i<miniblock_count; i++) {
      if (out[offset] != 0) out[offset] <<= 8;
      out[offset] |= miniblock_bitwidths[i];
    }
#else
    out[offset] = miniblock_bitwidths[0] + (miniblock_bitwidths[1] << 8) +
      (miniblock_bitwidths[2] << 16) + (miniblock_bitwidths[3] << 24);
#endif

#if 0
    printf("[bitwidth] out[%u]=%lx\n", offset, out[offset]);
#endif
    offset++;

    // The main bit-packing processing
    for (int miniblock = 0; miniblock < miniblock_count; miniblock++) {
      uint bitwidth = miniblock_bitwidths[miniblock];
      uint shift = 0;
      out[offset] = 0;
      for (int i = 0; i < miniblock_size; i++) {
        if (shift + bitwidth > 64) {
          // handling to store the bit-packed value over 64-bit boundary
          if (shift != 64) out[offset] += in[miniblock * miniblock_size + i] << shift;
          offset++;
#if 0
          printf("\tincremented during packing offset=%u val=%lu\n", offset, in[miniblock * miniblock_size + i]);
#endif
          // resume the next loop not to overwrite the previous value
          shift = (shift + bitwidth) & (64-1);
          out[offset] = in[miniblock * miniblock_size + i] >> (bitwidth - shift);
        } else {
          out[offset] += in[miniblock * miniblock_size + i] << shift;
          shift += bitwidth;
        }
      }
      if (offset == UINT32_MAX) {
        std::cout << "offset is overflowed" << std::endl;
        exit(1);
      }
      offset++;
    }
#if 0
    printf("endof miniblock offset=%u\n", offset);
#endif

    // Increment the input pointer by block size
    in += block_size;
  }

  block_offsets[num_entries / block_size] = offset;
  if constexpr (enable_debug_print) {
    std::cout << "num_entries:" << num_entries << ", final_index" << num_entries / block_size << std::endl;
  }

  // return sizeof(struct binpack_hdr) + offset * sizeof(ulong);
  return offset;
}



std::pair<uint, uint> rleBinPack(uint*&in, uint*& value, uint*& run_length, uint*& val_offsets, uint*& rl_offsets, uint num_entries) {
  uint val_offset = 0;
  uint rl_offset = 0;

  uint block_size = 512;
  uint elem_per_thread = 1;
  uint tile_size = block_size * elem_per_thread;

  //nonblock 
  block_size = tile_size;
  
  uint miniblock_count = 4;
  uint total_count = num_entries;
  uint first_val = in[0];

  value[0] = block_size;
  value[1] = miniblock_count;
  value[2] = total_count;
  value[3] = first_val;

  run_length[0] = block_size;
  run_length[1] = miniblock_count;
  run_length[2] = total_count;
  run_length[3] = first_val;

  val_offset += 4;
  rl_offset += 4;

#ifdef ENABLE_PACK_DEBUG
  uint num_tiles = (num_entries + tile_size - 1) / tile_size;
#endif

  uint* val = new uint[tile_size]();
  uint* rl = new uint[tile_size]();

  for (uint tile_start=0; tile_start<num_entries; tile_start += tile_size) {
    uint block_index = tile_start / block_size;

    uint count = 0;
    val[count] = in[0];
    uint run = 1;
    for (int i = 1; i < tile_size; i++) {
      if (in[i] != in[i-1]) {
        rl[count] = run;
        count++;
        val[count] = in[i]; 
        run = 1;
      } else {
        run++;
      }
    }
    rl[count] = run;
    count++;

    // non block
    int bl_size = count;
    int block_start = 0;

    rl_offsets[block_index] = rl_offset;
    val_offsets[block_index] = val_offset;

    uint min_val = val[block_start];
    uint min_rl = rl[block_start];
    for (int i = 1; i < bl_size; i++) {
      if (val[block_start + i] < min_val) min_val = val[block_start + i];
      if (rl[block_start + i] < min_rl) min_rl = rl[block_start + i];
    }

    uint val_bitwidth = 0;
    uint rl_bitwidth = 0;

    for (int i = block_start; i < block_start + bl_size; i++) {
      val[i] = val[i] - min_val;
      rl[i] = rl[i] - min_rl;
      uint bitwidth = uint(ceil(log2(val[i] + 1)));
      val_bitwidth = max(val_bitwidth, bitwidth);
      bitwidth = uint(ceil(log2(rl[i] + 1)));
      rl_bitwidth = max(rl_bitwidth, bitwidth);
    }

    value[val_offset] = min_val;
    run_length[rl_offset] = min_rl;
    val_offset++; rl_offset++;

    value[val_offset] = val_bitwidth + (val_bitwidth << 8) +
      (val_bitwidth << 16) + (val_bitwidth << 24);
    run_length[rl_offset] = rl_bitwidth + (rl_bitwidth << 8) +
      (rl_bitwidth << 16) + (rl_bitwidth << 24);
    val_offset++; rl_offset++;

    if (block_start == (bl_size * (elem_per_thread - 1))) { // if last block
      value[val_offset] = count - bl_size * (elem_per_thread - 1);
      run_length[rl_offset] = count - bl_size * (elem_per_thread - 1);
    } else {
      value[val_offset] = bl_size;
      run_length[rl_offset] = bl_size;
    }
    val_offset++; rl_offset++;

    uint bitwidth = val_bitwidth;
    uint shift = 0;
    for (int i = block_start; i < block_start + bl_size; i++) {
      if (shift + bitwidth > 32) {
        if (shift != 32) value[val_offset] += val[i] << shift;
        val_offset++;
        shift = (shift + bitwidth) & (32-1);
        value[val_offset] = val[i] >> (bitwidth - shift);
      } else {
        value[val_offset] += val[i] << shift;
        shift += bitwidth;
      }
    }
    val_offset++;

    bitwidth = rl_bitwidth;
    shift = 0;
    for (int i = block_start; i < block_start + bl_size; i++) {
      if (shift + bitwidth > 32) {
        if (shift != 32) run_length[rl_offset] += rl[i] << shift;
        rl_offset++;
        shift = (shift + bitwidth) & (32-1);
        run_length[rl_offset] = rl[i] >> (bitwidth - shift);
      } else {
        run_length[rl_offset] += rl[i] << shift;
        shift += bitwidth;
      }
    }
    rl_offset++;

    in += tile_size;

  }

  val_offsets[num_entries / block_size] = val_offset;
  rl_offsets[num_entries / block_size] = rl_offset;

  return std::make_pair(val_offset, rl_offset);
}

