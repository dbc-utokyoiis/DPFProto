#pragma once

#include <cstdint>
#include <cstdio>
#include "helper.cuh"

typedef void REC;

struct rec_head {
    uint16_t lenrec;   /* length of record*/
    uint16_t arofst[]; /* array of offsets to attributes */
};

enum rec_type {
    REC_ATTR_INT16 = 0, // 2 bytes
    REC_ATTR_INT32,     // 4 bytes
    REC_ATTR_INT64,     // 8 bytes
    REC_ATTR_CHAR,      // fixed length CHAR(n) -> n bytes
    REC_ATTR_VCHAR,     // variable length VCHAR(n) -> max n bytes
};

const unsigned int NUM_THREADS_PER_BLOCK = 128U;
//const unsigned int NUM_THREADS_PER_BLOCK = 512U;
//const unsigned int NUM_THREADS_PER_BLOCK = 1024U;

/* host-side helper function */
static std::pair<unsigned int, unsigned int> get_dims_generic(size_t len)
{
    unsigned int block_dim = std::min(static_cast<unsigned int>(len), NUM_THREADS_PER_BLOCK);
    unsigned int grid_dim = (static_cast<unsigned int>(len) + block_dim - 1) / block_dim;
    return {grid_dim, block_dim};
}

/* NOTE: gpu version */
static inline __device__ uint16_t rec_get_attr_int16(REC *rp, const uint N, uint idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  //if(idx>=N){ PANIC_PERROR("rec_get_attr_int16"); }
  DEBUG_ASSERT(idx < N);
  return(*(uint16_t *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
}

static inline __device__ uint32_t rec_get_attr_int32(REC *rp, const uint N, uint idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  //if(idx>=N){ PANIC_PERROR("rec_get_attr_int32"); }
  DEBUG_ASSERT(idx < N);
  return(*(uint32_t *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
}

static inline __device__ uint64_t rec_get_attr_int64(REC *rp, const uint N, uint idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  //if(idx>=N){ PANIC_PERROR("rec_get_attr_int64"); }
  DEBUG_ASSERT(idx < N);
  return(*(uint64_t *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
}

static inline __device__ char *rec_get_attr_chr(REC *rp, const uint N, uint idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  //if(idx>=N){ PANIC_PERROR("rec_get_attr_chr"); }
  DEBUG_ASSERT(idx < N);
  return((char *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
}
//#define rec_get_attr_vchr rec_get_attr_chr
//
// static inline __device__ int rec_get_size_vchr(REC *rp, const uint N, uint idx)
// {
//   int l;
//   struct rec_head *rhp = (struct rec_head *)rp;
//   //if(idx>=N){ PANIC_PERROR("rec_get_size_vchr"); }
//   //printf("rec_get_size_vchr: idx=%u, N=%u\n", idx, N);
//   DEBUG_ASSERT(idx < N);
//   if(idx+1 == N){ l = rhp->lenrec - rhp->arofst[idx]; }
//   else{ l = rhp->arofst[idx+1] - rhp->arofst[idx]; };
//   return(l);
// }

static inline __device__ char *rec_get_attr_vchr(REC *rp, const uint N, int idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  DEBUG_ASSERT(idx < N);
  return((char *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx] + sizeof(uint16_t)));
}

static inline __device__ int rec_get_size_vchr(REC *rp, const uint N, uint idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  uint16_t len = *(reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
  return static_cast<int>(len);
}

static inline __device__ char *rec_get_attr_vchr_aligned(REC *rp, const uint N, int idx, int alignment)
{
  const int offset = alignment;
  /* NOTE: assuming that alignment have to be 4, basically */
  struct rec_head *rhp = (struct rec_head *)rp;
  return((char *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx] + offset));
}