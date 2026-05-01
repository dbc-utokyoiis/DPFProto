#pragma once

#include <cstdint>
#include <string>
#include "gpu_rec.cuh"

/* Basically same to pag.h */
/* Read-only functions are only defined here. */
struct pag_head {
    // char eyecatch; // magic number for page identification
    uint32_t nalloc; // number of allocated records
    uint32_t watermark; // free space watermark
    uint32_t lfreespace;
};

#define PAG_HDR_SIZ (sizeof(struct pag_head))
#define PAG_FTR_SIZ (0)

#define PAG_PAGEID_MASK_PAGEID  (0x0000ffffffffffff)
#define PAG_OSLT_MASK_OFFSET    (0x4fffffff)

#define PAG_SLOTID_MASK_SLOTID  (0x4fffffff)
#define PAG_SLOTID_MASK_ERROR   (0x80000000)

typedef void PAG;

static __device__ inline uint32_t pag_get_oslt(PAG *pp, uint32_t slotid, uint32_t page_size)
{
  return(*(uint32_t *)(reinterpret_cast<char*>(pp) + page_size - PAG_FTR_SIZ - sizeof(uint32_t) * (slotid + 1)));
}

static __device__ inline uint32_t pag_get_nalloc(PAG *pp)
{
  struct pag_head *php = (struct pag_head *)pp;
  return(php->nalloc);
}

static __device__ inline REC *pag_fetch_rec(PAG *pp, uint32_t slotid, uint32_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  /* check slot existence */
  if(slotid >= php->nalloc)
    return NULL;

  uint32_t oslt = pag_get_oslt(pp, slotid, page_size);

  return((REC *)(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET)));
}

static __device__ inline uint64_t pagvc_get_varchar_rowid64(PAG *pp, uint32_t slotid, size_t page_size)
{
  return(*(uint64_t *)(reinterpret_cast<char*>(pp) + page_size - PAG_FTR_SIZ - sizeof(uint64_t) * (slotid + 1)));
}

static __device__ inline uint32_t pagvc_get_varchar_len(PAG *pp, uint32_t slotid)
{
  return (*(uint16_t *)(
    reinterpret_cast<char*>(pp)
      + sizeof(struct pag_head) + sizeof(uint32_t) * slotid));
}

static __device__ inline char* pagvc_get_string_base(PAG *pp, uint32_t slotid)
{
  constexpr int alignment = 4;
  struct pag_head *php = (struct pag_head *)pp;
  const uint32_t nalloc = php->nalloc;
  const size_t pivotid = 1;
  return (reinterpret_cast<char*>(pp)
     + sizeof(struct pag_head)
     + sizeof(uint32_t) * (slotid + nalloc * pivotid));
}