#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "rec.h"
#include "pag_core.h"

#define INVALID_ROWID (UINT64_MAX)
#define PAG_HDR_SIZ (sizeof(struct pag_head))
#define PAG_FTR_SIZ (0)

#define PAG_PAGEID_MASK_PAGEID  (0x0000ffffffffffff)
#define PAG_OSLT_MASK_OFFSET    (0x4fffffff)

#define PAG_SLOTID_MASK_SLOTID  (0x4fffffff)
#define PAG_SLOTID_MASK_ERROR   (0x80000000)

static inline uint32_t pag_get_oslt(PAG *pp, uint32_t slotid, size_t page_size)
{ 
  return(*(uint32_t *)(reinterpret_cast<char*>(pp) + page_size - PAG_FTR_SIZ - sizeof(uint32_t) * (slotid + 1)));
}

static inline uint64_t pag_get_varchar_rowid64(PAG *pp, uint32_t slotid, size_t page_size)
{ 
  return(*(uint64_t *)(reinterpret_cast<char*>(pp) + page_size - PAG_FTR_SIZ - sizeof(uint64_t) * (slotid + 1)));
}

static inline void pag_set_oslt(PAG *pp, uint32_t slotid, uint32_t oslt, size_t page_size)
{
  *(uint32_t *)(reinterpret_cast<char*>(pp) + page_size - PAG_FTR_SIZ - sizeof(uint32_t) * (slotid + 1)) = oslt;
}

static inline void pag_set_varchar_rowid64(PAG *pp, uint32_t slotid, uint64_t rowid64, size_t page_size)
{
  *(uint64_t *)(reinterpret_cast<char*>(pp) + page_size - PAG_FTR_SIZ - sizeof(uint64_t) * (slotid + 1)) = rowid64;
}

inline uint64_t pagcol_get_rowid(PAG *pp, uint32_t slotid, size_t page_size)
{
  return *(uint64_t *)(reinterpret_cast<char*>(pp) + page_size - PAG_FTR_SIZ - sizeof(uint64_t) * (slotid + 1));
}


template<typename ColType>
ColType pagcol_v2_get_rowid(PAG *pp, uint32_t slotid, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  const uint32_t nalloc = php->nalloc;
  /* check slot existence */
  if(slotid >= nalloc) {
    if constexpr (std::is_same_v<ColType, int32_t>) {
      return(INT32_MIN);
    } else if constexpr (std::is_same_v<ColType, int64_t>) {
      return(INT64_MIN);
    } else {
      static_assert("pagcol_v2_get_rowid: unsupported type.");
    }
  }

  return(*(uint64_t *)(reinterpret_cast<char*>(pp)
    + sizeof(struct pag_head) + sizeof(ColType) * nalloc + sizeof(uint64_t) * slotid));
}

uint* pagcol_v2_get_value_comp_offset_ptr(PAG *pp, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  const uint32_t nalloc = php->nalloc;
  return ((uint *)(reinterpret_cast<char*>(pp) + sizeof(struct pag_head)));
}

uint* pagcol_v2_get_rowid_comp_offset_ptr(PAG *pp, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  const uint32_t nalloc = php->nalloc;
  return ((uint *)(reinterpret_cast<char*>(pp) + sizeof(struct pag_head) + sizeof(uint) * (nalloc + 1)));
}

/* row-store API */
uint32_t pag_append_rec_unordered(PAG *pag, REC *rec, size_t page_size)
{
    pag_head *php = (pag_head *)pag;
    rec_head *rhp = (rec_head *)rec;
    // assert(php->watermark <= page_size);

    uint32_t len = rhp->lenrec;
    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((len & 3) == 0); // record length must be 4-byte aligned
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    /* check space availability */
    if (len + sizeof(uint32_t) >= php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)rec, len);
    pag_set_oslt(pag, slotid, php->watermark, page_size);
    php->watermark += len;
    php->nalloc++;
    php->lfreespace -= len + sizeof(uint32_t);
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned
    // php->watermark = (php->watermark + 1) & ~1; // align to even address

    return(slotid);
}

/* column-store API v2 */
int32_t pagcol_append_batch_unordered_column_int32(PAG *pag, std::span<int32_t> values32, size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    if (php->nalloc) {
      std::cerr << "pagcol_append_batch_unordered_column_int32: pag->nalloc is not zero. nalloc=" << php->nalloc << std::endl;
      exit(EXIT_FAILURE);
    }

    /* check space availability */
    size_t len_total = (sizeof(uint32_t)) * values32.size() ;
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    size_t nalloc = values32.size();
    php->nalloc = nalloc;

    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)values32.data(), sizeof(int32_t) * values32.size());
    php->watermark += (nalloc * sizeof(int32_t));
    php->lfreespace -= (nalloc * sizeof(int32_t));

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    return(slotid);
}

int32_t pagcol_append_batch_unordered_column_int64(PAG *pag, std::span<int64_t> values64, size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    if (php->nalloc) {
      std::cerr << "pagcol_append_batch_unordered_column_int32: pag->nalloc is not zero. nalloc=" << php->nalloc << std::endl;
      exit(EXIT_FAILURE);
    }

    /* check space availability */
    size_t len_total = (sizeof(uint64_t)) * values64.size() ;
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);
    if (php->watermark == sizeof(struct pag_head)) {
      /* if the page is empty, align the watermark to 8 bytes for 64-bit values */
      php->watermark = (php->watermark + 7) & ~7;
    }
    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    size_t nalloc = values64.size();
    php->nalloc = nalloc;

    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)values64.data(), sizeof(int64_t) * values64.size());
    php->watermark += (nalloc * sizeof(int64_t));
    php->lfreespace -= (nalloc * sizeof(int64_t));

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    return(slotid);
}

int32_t pagcol_append_batch_unordered_column_int32_comp(PAG *pag,
  uint nvalues_aligned, uint nvalues_actual, std::span<uint> compressed_values,
  const size_t compressed_value_bytes,
  std::span<uint> block_offsets_values,
  uint32_t &compressed_page_bytes,
  size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    if (php->nalloc) {
      std::cerr << "pagcol_append_rec_unordered_column_int32: pag->nalloc is not zero. nalloc=" << php->nalloc << std::endl;
      exit(EXIT_FAILURE);
    }
    size_t nblocks = nvalues_aligned / 128;
    if (nblocks != block_offsets_values.size() - 1) {
      std::cerr << "pagcol_append_rec_unordered_column_int32: nvalues and block_offsets_values.size() mismatch. "
        << "nvalues_aligned=" << nvalues_aligned << ", nvalues_actual=" << nvalues_actual
        << ", block_offsets_values.size()=" << block_offsets_values.size() << std::endl;
      exit(EXIT_FAILURE);
    }

    /* check space availability */
    size_t len_total = (compressed_value_bytes + sizeof(uint32_t) * block_offsets_values.size());
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    /* set nalloc */
    php->nalloc = nvalues_actual;

    size_t noffsets = block_offsets_values.size();

    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)block_offsets_values.data(), sizeof(uint) * noffsets);
    php->watermark += (noffsets * sizeof(uint));
    php->lfreespace -= (noffsets * sizeof(uint));
    if (php->watermark & 3) {
      std::cerr << "pagcol_append_batch_unordered_column_int64_comp: watermark is not 8-byte aligned. watermark=" << php->watermark << std::endl;
      exit(EXIT_FAILURE);
    }
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)compressed_values.data(), compressed_value_bytes);
    php->watermark += (compressed_value_bytes);
    php->lfreespace -= (compressed_value_bytes);

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    /* this assertion should be passed because all data is 4-byte aligned */
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    compressed_page_bytes = php->watermark;

    /* set the aligned nvalues to watermark because the page is full. */
    php->watermark = nvalues_aligned;

    return(slotid);
}

int32_t pagcol_append_batch_unordered_column_int64_comp(PAG *pag,
  uint nvalues_aligned, uint nvalues_actual,
  std::span<ulong> compressed_values,
  size_t compressed_value_bytes,
  std::span<uint> block_offsets_values,
  uint32_t &compressed_page_bytes,
  size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    if (php->nalloc) {
      std::cerr << "pagcol_append_rec_unordered_column_int64: pag->nalloc is not zero. nalloc=" << php->nalloc << std::endl;
      exit(EXIT_FAILURE);
    }
    size_t nblocks = nvalues_aligned / 128;
    if (nblocks != block_offsets_values.size() - 1) {
      std::cerr << "pagcol_append_rec_unordered_column_int64: nvalues and block_offsets_values.size() mismatch. "
        << "nvalues_aligned=" << nvalues_aligned << ", nvalues_actual=" << nvalues_actual
        << ", block_offsets_values.size()=" << block_offsets_values.size() << std::endl;
      exit(EXIT_FAILURE);
    }

    /* check space availability */
    size_t len_total = (compressed_value_bytes + sizeof(uint32_t) * block_offsets_values.size());
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    /* set nalloc */
    php->nalloc = nvalues_actual;

    size_t noffsets = block_offsets_values.size();

    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)block_offsets_values.data(), sizeof(uint) * noffsets);
    php->watermark += (noffsets * sizeof(uint));
    php->lfreespace -= (noffsets * sizeof(uint));
    /* noffsets is even? */
    if (php->watermark & 7) {
      /* NOTE: header is 12B (3 integers), so padding is required if noffset is even */
      /*  This is only required for 64-bit fields */
      memset(reinterpret_cast<char*>(pag) + php->watermark, 0, sizeof(uint));
      // align to 4-byte boundary
      php->watermark += sizeof(uint);
      php->lfreespace -= sizeof(uint);
    }
    if (php->watermark & 7) {
      std::cerr << "pagcol_append_batch_unordered_column_int64_comp: watermark is not 8-byte aligned. watermark=" << php->watermark << std::endl;
      exit(EXIT_FAILURE);
    }
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)compressed_values.data(), compressed_value_bytes);
    size_t offset = php->watermark;
    php->watermark += (compressed_value_bytes);
    php->lfreespace -= (compressed_value_bytes);

#if 0
    for (size_t i = 0; i < 10; ++i) {
      ulong v = reinterpret_cast<ulong*>(reinterpret_cast<char*>(pag) + offset)[i];
      std::cout << v << "\n";
    }
    std::cout << "offset" << offset << "\n";
    exit(0);
#endif

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    /* this assertion should be passed because all data is 4-byte aligned */
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    compressed_page_bytes = php->watermark;

    /* set the aligned nvalues to watermark because the page is full. */
    php->watermark = nvalues_aligned;

    return(slotid);
}

int32_t pagcol_append_batch_unordered_column_int32_with_rowid(PAG *pag, std::span<int32_t> values32, std::span<uint64_t> rowids, size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    if (php->nalloc) {
      std::cerr << "pagcol_append_batch_unordered_column_int32: pag->nalloc is not zero. nalloc=" << php->nalloc << std::endl;
      exit(EXIT_FAILURE);
    }
    if (values32.size() != rowids.size()) {
      std::cerr << "pagcol_append_batch_unordered_column_int32: values32.size() and rowids.size() mismatch. "
        "values32.size()=" << values32.size()
        << ", rowids.size()=" << rowids.size() << std::endl;
      exit(EXIT_FAILURE);
    }

    /* check space availability */
    size_t len_total = (sizeof(int32_t) + sizeof(uint64_t)) * values32.size() ;
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    size_t nalloc = values32.size();
    php->nalloc = nalloc;

    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)values32.data(), sizeof(int32_t) * values32.size());
    php->watermark += (nalloc * sizeof(int32_t));
    php->lfreespace -= (nalloc * sizeof(int32_t));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)rowids.data(), sizeof(uint64_t) * rowids.size());
    php->watermark += (nalloc * sizeof(uint64_t));
    php->lfreespace -= (nalloc * sizeof(uint64_t));

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    return(slotid);
}

int32_t pagcol_append_batch_unordered_column_int64_with_rowid(PAG *pag, std::span<int64_t> values64, std::span<uint64_t> rowids, size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    if (php->nalloc) {
      std::cerr << "pagcol_append_batch_unordered_column_int32: pag->nalloc is not zero. nalloc=" << php->nalloc << std::endl;
      exit(EXIT_FAILURE);
    }
    if (values64.size() != rowids.size()) {
      std::cerr << "pagcol_append_batch_unordered_column_int64: values64.size() and rowids.size() mismatch. "
        "values64.size()=" << values64.size()
        << ", rowids.size()=" << rowids.size() << std::endl;
      exit(EXIT_FAILURE);
    }

    /* check space availability */
    size_t len_total = (sizeof(int64_t) + sizeof(uint64_t)) * values64.size() ;
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    size_t nalloc = values64.size();
    php->nalloc = nalloc;

    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)values64.data(), sizeof(int64_t) * values64.size());
    php->watermark += (nalloc * sizeof(int64_t));
    php->lfreespace -= (nalloc * sizeof(int64_t));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)rowids.data(), sizeof(uint64_t) * rowids.size());
    php->watermark += (nalloc * sizeof(uint64_t));
    php->lfreespace -= (nalloc * sizeof(uint64_t));
    
    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    return(slotid);
}

int32_t pagcol_append_batch_unordered_column_int32_with_rowid_comp(PAG *pag,
  std::span<uint> compressed_values, uint compressed_values_size, std::span<uint> block_offsets_values,
  std::span<ulong> compressed_rowids, uint compressed_rowids_size, std::span<uint> block_offsets_rowids,
  size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    if (php->nalloc) {
      std::cerr << "pagcol_append_rec_unordered_column_int32: pag->nalloc is not zero. nalloc=" << php->nalloc << std::endl;
      exit(EXIT_FAILURE);
    }
    if (block_offsets_values.size() != block_offsets_rowids.size()) {
      std::cerr << "pagcol_append_rec_unordered_column_int64: block_offsets_values.size() and block_offsets_rowids.size() mismatch. "
        "block_offsets_values.size()=" << block_offsets_values.size()
        << ", block_offsets_rowids.size()=" << block_offsets_rowids.size() << std::endl;
      exit(EXIT_FAILURE);
    }

    /* check space availability */
    size_t len_total = (sizeof(int32_t) + sizeof(uint64_t)) * compressed_values.size() ;
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    size_t noffsets = block_offsets_values.size();
    size_t nalloc = block_offsets_values.size() - 1;
    php->nalloc = nalloc;

    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)&compressed_values_size, sizeof(uint));
    php->watermark += (sizeof(uint));
    php->lfreespace -= (sizeof(uint));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)&compressed_rowids_size, sizeof(uint));
    php->watermark += (sizeof(uint));
    php->lfreespace -= (sizeof(uint));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)block_offsets_values.data(), sizeof(uint) * noffsets);
    php->watermark += (noffsets * sizeof(uint));
    php->lfreespace -= (noffsets * sizeof(uint));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)block_offsets_rowids.data(), sizeof(uint) * noffsets);
    php->watermark += (noffsets * sizeof(uint));
    php->lfreespace -= (noffsets * sizeof(uint));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)compressed_rowids.data(), compressed_values_size);
    php->watermark += (compressed_values_size);
    php->lfreespace -= (compressed_values_size);
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)compressed_rowids.data(), compressed_rowids_size);
    php->watermark += (nalloc * sizeof(uint64_t));
    php->lfreespace -= (nalloc * sizeof(uint64_t));
    
    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    return(slotid);
}

int32_t pagcol_append_batch_unordered_column_int64_with_rowid_comp(PAG *pag,
  std::span<ulong> compressed_values, uint compressed_values_size, std::span<uint> block_offsets_values,
  std::span<ulong> compressed_rowids, uint compressed_rowids_size, std::span<uint> block_offsets_rowids,
  size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    if (php->nalloc) {
      std::cerr << "pagcol_append_rec_unordered_column_int32: pag->nalloc is not zero. nalloc=" << php->nalloc << std::endl;
      exit(EXIT_FAILURE);
    }
    if (block_offsets_values.size() != block_offsets_rowids.size()) {
      std::cerr << "pagcol_append_rec_unordered_column_int64: block_offsets_values.size() and block_offsets_rowids.size() mismatch. "
        "block_offsets_values.size()=" << block_offsets_values.size()
        << ", block_offsets_rowids.size()=" << block_offsets_rowids.size() << std::endl;
      exit(EXIT_FAILURE);
    }

    /* check space availability */
    size_t len_total = (sizeof(int32_t) + sizeof(uint64_t)) * compressed_values.size() ;
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    size_t noffsets = block_offsets_values.size();
    size_t nalloc = block_offsets_values.size() - 1;
    php->nalloc = nalloc;

    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)&compressed_values_size, sizeof(uint));
    php->watermark += (sizeof(uint));
    php->lfreespace -= (sizeof(uint));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)&compressed_rowids_size, sizeof(uint));
    php->watermark += (sizeof(uint));
    php->lfreespace -= (sizeof(uint));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)block_offsets_values.data(), sizeof(uint) * noffsets);
    php->watermark += (noffsets * sizeof(uint));
    php->lfreespace -= (noffsets * sizeof(uint));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)block_offsets_rowids.data(), sizeof(uint) * noffsets);
    php->watermark += (noffsets * sizeof(uint));
    php->lfreespace -= (noffsets * sizeof(uint));
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)compressed_rowids.data(), compressed_values_size);
    php->watermark += (compressed_values_size);
    php->lfreespace -= (compressed_values_size);
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)compressed_rowids.data(), compressed_rowids_size);
    php->watermark += (nalloc * sizeof(uint64_t));
    php->lfreespace -= (nalloc * sizeof(uint64_t));
    
    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    return(slotid);
}


#if 0
/* Not yet implemented. */
int32_t pagcol_append_batch_unordered_column_char(PAG *pag, size_t len_schema, std::span<std::string> batch, std::span<uint64_t> rowids, size_t page_size)
{
    constexpr int alignment = 4; // align to 4 bytes
    constexpr int alignment_mask = alignment - 1; // align to 4 bytes
    pag_head *php = (pag_head *)pag;

    assert((php->watermark & (alignment - 1)) == 0); // insert destination pointer must be 4-byte aligned

    if (php->nalloc) {
      std::cerr << "pagcol_append_batch_unordered_column_int32: pag->nalloc is not zero. nalloc=" << php->nalloc << std::endl;
      exit(EXIT_FAILURE);
    }
    if (batch.size() != rowids.size()) {
      std::cerr << "pagcol_append_batch_unordered_column_char: batch.size() and rowids.size() mismatch. "
        "batch.size()=" << batch.size()
        << ", rowids.size()=" << rowids.size() << std::endl;
      exit(EXIT_FAILURE);
    }

    const size_t len_field = len_schema;
    // bool padding_required = false;

    size_t len_field_aligned = (len_field + alignment_mask) & ~alignment_mask; // align to 4 bytes
    assert((len_field_aligned & alignment_mask) == 0); // record length must be 4-byte aligned
    /* check space availability */
    size_t len_total = len_field_aligned * batch.size() + sizeof(uint64_t);
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    size_t slotid = php->nalloc;
    size_t nalloc = batch.size();
    php->nalloc = nalloc;

    for (size_t i = 0; i < batch.size(); ++i) {
      const std::string &str = batch[i];
      const char *valchar = str.c_str();
      size_t len_char = str.size();

      // std::cout << "[DEBUG] valchar='" << std::string(valchar, len_char) << "', len_char=" << len_char << std::endl;
      memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)valchar, len_char);
      memset(reinterpret_cast<char*>(pag) + php->watermark + len_char, ' ', len_field_aligned - len_char);
      pag_set_varchar_rowid64(pag, slotid + i, rowids[i], page_size);
      php->watermark += len_field_aligned;
      // std::cout << "[DEBUG][AFTER]  watermark" << php->watermark << ", len_total=" << len_total << std::endl;
      php->lfreespace -= len_total;
    }

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned


    return(slotid);
}
#endif


/* column-store API v2 without rowid, assuming implicit ordering */
int32_t pagcol_append_rec_unordered_column_int32(PAG *pag, int32_t val32, size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    /* check space availability */
    size_t len_total = sizeof(val32);
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)&val32, sizeof(val32));
    //pag_set_varchar_rowid64(pag, slotid, rowid, page_size);
    php->watermark += sizeof(val32);
    php->nalloc++;
    php->lfreespace -= len_total;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned
    // php->watermark = (php->watermark + 1) & ~1; // align to even address

    return(slotid);
}

int32_t pagcol_append_rec_unordered_column_int64(PAG *pag, int64_t val64, size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    /* check space availability */
    size_t len_total = sizeof(val64);
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));
    if (php->watermark == sizeof(struct pag_head)) {
      /* if the page is empty, align the watermark to 8 bytes for 64-bit values */
      php->watermark = (php->watermark + 7) & ~7;
    }

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)&val64, sizeof(val64));
    php->watermark += sizeof(val64);
    php->nalloc++;
    php->lfreespace -= len_total;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned
    // php->watermark = (php->watermark + 1) & ~1; // align to even address

    return(slotid);
}

int32_t pagcol_append_rec_unordered_column_char(PAG *pag, size_t len_schema, char *valchar, size_t len_char, size_t page_size)
{
    constexpr int alignment = 4; // align to 4 bytes
    constexpr int alignment_mask = alignment - 1; // align to 4 bytes
    pag_head *php = (pag_head *)pag;

    assert((php->watermark & (alignment - 1)) == 0); // insert destination pointer must be 4-byte aligned

    const size_t len_field = len_schema;
    // bool padding_required = false;

    size_t len_field_aligned = (len_field + alignment_mask) & ~alignment_mask; // align to 4 bytes
    assert((len_field_aligned & alignment_mask) == 0); // record length must be 4-byte aligned
    /* check space availability */
    size_t len_total = len_field_aligned;
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    // std::cout << "[DEBUG][BEFORE] watermark" << php->watermark << ", len_total=" << len_total << std::endl;

    //size_t padding_size = (schema_len % alignment) ? (alignment - (schema_len % alignment)) : 0;
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)valchar, len_char);
    memset(reinterpret_cast<char*>(pag) + php->watermark + len_char, ' ', len_field_aligned - len_char);
    //if (lenchar_aligned > lenchar) {
    //  memset(reinterpret_cast<char*>(pag) + php->watermark + lenchar, ' ', lenchar_aligned - lenchar);
    //}
    php->watermark += len_field_aligned;
    // std::cout << "[DEBUG][AFTER]  watermark" << php->watermark << ", len_total=" << len_total << std::endl;
    php->nalloc++;
    php->lfreespace -= len_total;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned
    // php->watermark = (php->watermark + 1) & ~1; // align to even address

    return(slotid);
}

int32_t pagcol_append_rec_unordered_column_vchar(PAG *pag, size_t len_schema, char *valchar, uint16_t len_vchar, size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    constexpr uint32_t alignment = 4; // align to 4 bytes
    // bool padding_required = false;
    size_t padding_size = 0;

    size_t l = std::min(static_cast<size_t>(len_vchar), len_schema);
    if (l & 3) {
      // padding_required = true;
      padding_size = (alignment - (l % alignment));
    }
    //*(uint16_t *)attrp = len_varchar;
    //memcpy(attrp+sizeof(uint16_t), val, len_varchar);
    //if (padding_required) {
    //  /* padding */
    //  memset(attrp+sizeof(uint16_t)+len_varchar, 0, padding_size);
    //}
    //*ofst = attrp - rp;
    //// assert((*ofst & 1) == 0);
    //attrp += (sizeof(uint16_t)+(alignment - sizeof(uint16_t))+(len_varchar));
    //if (padding_required) {
    //  attrp += padding_size;
    //}
    //rhp->lenrec = attrp - rp;

    size_t len_vchar_aligned = l + padding_size;
    /* check space availability */
    size_t len_total = sizeof(uint32_t) + 2 * sizeof(uint16_t) + len_vchar_aligned;
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    /* Confirmed that the page has the enough space */
    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

#if 0
    std::cout << "[DEBUG] "
      << " watermark=" << php->watermark
      << ", len_schema=" << len_schema
      << ", len_vchar=" << len_vchar
      << ", len_vchar_aligned=" << len_vchar_aligned << ", len_total=" << len_total << std::endl;
#endif
    /* a vchar record has to have slot pointer */
    uint32_t slotid = php->nalloc;
    pag_set_oslt(pag, slotid, php->watermark, page_size);

    char *dst_base = reinterpret_cast<char*>(pag) + php->watermark;
    /* set length */
    uint16_t *vcharlenp = reinterpret_cast<uint16_t*>(dst_base);
    *vcharlenp = len_vchar;

    /* append record content */
    memcpy(dst_base + 2 * sizeof(uint16_t), (void *)valchar, len_vchar);
    if (len_vchar_aligned > len_vchar) {
      memset(dst_base + 2 * sizeof(uint16_t) + len_vchar, 0, len_vchar_aligned - len_vchar);
    }
    assert((reinterpret_cast<uintptr_t>(dst_base) & 3) == 0);
    php->watermark += (sizeof(uint16_t) * 2 + len_vchar_aligned);
    php->nalloc++;
    php->lfreespace -= len_total;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    /* This is useful only for the pivoted layout */
    // pag_set_varchar_rowid64(pag, slotid, rowid, page_size);

    return(slotid);
}


/* column-store API with rowid */
#if 1
int32_t pagcol_append_rec_unordered_column_int32_with_rowid(PAG *pag, int32_t val32, uint64_t rowid, size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    /* check space availability */
    size_t len_total = sizeof(val32) + sizeof(uint64_t);
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)&val32, sizeof(val32));
    pag_set_varchar_rowid64(pag, slotid, rowid, page_size);
    php->watermark += sizeof(val32);
    php->nalloc++;
    php->lfreespace -= len_total;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned
    // php->watermark = (php->watermark + 1) & ~1; // align to even address

    return(slotid);
}

int32_t pagcol_append_rec_unordered_column_int64_with_rowid(PAG *pag, int64_t val64, uint64_t rowid, size_t page_size)
{
    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned
    if (php->watermark == sizeof(struct pag_head)) {
      /* if the page is empty, align the watermark to 8 bytes for 64-bit values */
      php->watermark = (php->watermark + 7) & ~7;
    }

    /* check space availability */
    size_t len_total = sizeof(val64) + sizeof(uint64_t);
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)&val64, sizeof(val64));
    pag_set_varchar_rowid64(pag, slotid, rowid, page_size);
    php->watermark += sizeof(val64);
    php->nalloc++;
    php->lfreespace -= len_total;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned
    // php->watermark = (php->watermark + 1) & ~1; // align to even address

    return(slotid);
}
#endif

int32_t pagcol_append_rec_unordered_column_char_with_rowid(PAG *pag, size_t len_schema, char *valchar, size_t len_char, uint64_t rowid, size_t page_size)
{
#if 0
    REC *_rp = (REC *)buf;
    uint8_t *rp = (uint8_t *)_rp;
    struct rec_head *rhp = (struct rec_head *)rp;
    uint16_t *ofst = &(rhp->arofst[idx]);
    uint8_t *attrp = rp + rhp->lenrec;
    int padding_size = 0;
    uint16_t char_len;
    const int alignment = 4; // align to 4 bytes

    if (reinterpret_cast<uintptr_t>(attrp) & 3 == 0) { // confirming that attrp is 4-byte aligned
      PANIC_ERROR("rec_set_attr_char:attrp is not 4-byte aligned.\n");
    }

    padding_size = (sizes[idx] % alignment) ? (alignment - (sizes[idx] % alignment)) : 0;
    char_len = sizes[idx] + padding_size;

    if(attrp + siz - rp > len) {
      PANIC_ERROR("rec_set_attr_chr:buffer overflow.\n");
    }
    if(*ofst) {
      PANIC_ERROR("rec_set_attr_chr:attribute exists.\n");
    }

    memset(attrp, ' ', sizes[idx] + padding_size);
    memcpy(attrp, val, siz < sizes[idx] ? siz : sizes[idx]);
    *ofst = attrp - rp;
    attrp += char_len;
    rhp->lenrec = attrp - rp;

    return(rp);
#endif

    constexpr int alignment = 4; // align to 4 bytes
    constexpr int alignment_mask = alignment - 1; // align to 4 bytes
    pag_head *php = (pag_head *)pag;

    assert((php->watermark & (alignment - 1)) == 0); // insert destination pointer must be 4-byte aligned

    const size_t len_field = len_schema;
    // bool padding_required = false;

    size_t len_field_aligned = (len_field + alignment_mask) & ~alignment_mask; // align to 4 bytes
    assert((len_field_aligned & alignment_mask) == 0); // record length must be 4-byte aligned
    /* check space availability */
    size_t len_total = len_field_aligned + sizeof(uint64_t);
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

    /* append record content and slot */
    uint32_t slotid = php->nalloc;
    // std::cout << "[DEBUG][BEFORE] watermark" << php->watermark << ", len_total=" << len_total << std::endl;

    //size_t padding_size = (schema_len % alignment) ? (alignment - (schema_len % alignment)) : 0;
    memcpy(reinterpret_cast<char*>(pag) + php->watermark, (void *)valchar, len_char);
    memset(reinterpret_cast<char*>(pag) + php->watermark + len_char, ' ', len_field_aligned - len_char);
    //if (lenchar_aligned > lenchar) {
    //  memset(reinterpret_cast<char*>(pag) + php->watermark + lenchar, ' ', lenchar_aligned - lenchar);
    //}
    pag_set_varchar_rowid64(pag, slotid, rowid, page_size);
    php->watermark += len_field_aligned;
    // std::cout << "[DEBUG][AFTER]  watermark" << php->watermark << ", len_total=" << len_total << std::endl;
    php->nalloc++;
    php->lfreespace -= len_total;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned
    // php->watermark = (php->watermark + 1) & ~1; // align to even address

    return(slotid);
}

/* for varchar field: non-pivoted layout */
int32_t pagcol_append_rec_unordered_column_vchar_with_rowid(PAG *pag, size_t len_schema, char *valchar, uint16_t len_vchar, uint64_t rowid, size_t page_size)
{
#if 0
    REC *_rp = (REC *)buf;
    uint8_t *rp = (uint8_t *)_rp;
    struct rec_head *rhp = (struct rec_head *)rp;
    uint16_t *ofst = &(rhp->arofst[idx]);
    // void *attrp = rp + rhp->lenrec;
    uint8_t *attrp = rp + rhp->lenrec;
    uint16_t len_varchar;
    const int alignment = 4; // align to 4 bytes
    int padding_required = 0;
    int padding_size = 0;

    if (reinterpret_cast<uintptr_t>(attrp) & 3 == 0) { // confirming that attrp is 4-byte aligned
      PANIC_ERROR("rec_set_attr_char:attrp is not 4-byte aligned.\n");
    }
    if(attrp + siz - rp > len) {
      PANIC_ERROR("rec_set_attr_vchr:buffer overflow.\n");
    }
    if(*ofst) {
      PANIC_ERROR("rec_set_attr_vchr:attribute exists.\n");
    }

    if (siz < sizes[idx] ) {
      len_varchar = siz;
    } else {
      len_varchar = sizes[idx];
    }
    if (len_varchar & 3) {
      padding_required = 1;
      padding_size = (alignment - (len_varchar % alignment));
    }

    *(uint16_t *)attrp = len_varchar; 
    memcpy(attrp+sizeof(uint16_t), val, len_varchar);
    if (padding_required) {
      /* padding */
      memset(attrp+sizeof(uint16_t)+len_varchar, 0, padding_size);
    }
    *ofst = attrp - rp;
    // assert((*ofst & 1) == 0);
    attrp += (sizeof(uint16_t)+(alignment - sizeof(uint16_t))+(len_varchar));
    if (padding_required) {
      attrp += padding_size;
    }
    rhp->lenrec = attrp - rp;

    return(rp);
#endif

    pag_head *php = (pag_head *)pag;

    //printf("pag_append_rec_unordered: len=%u\n", len);
    assert((php->watermark & 3) == 0); // insert destination pointer must be 4-byte aligned

    constexpr uint32_t alignment = 4; // align to 4 bytes
    // bool padding_required = false;
    size_t padding_size = 0;

    size_t l = std::min(static_cast<size_t>(len_vchar), len_schema);
    if (l & 3) {
      // padding_required = true;
      padding_size = (alignment - (l % alignment));
    }
    //*(uint16_t *)attrp = len_varchar; 
    //memcpy(attrp+sizeof(uint16_t), val, len_varchar);
    //if (padding_required) {
    //  /* padding */
    //  memset(attrp+sizeof(uint16_t)+len_varchar, 0, padding_size);
    //}
    //*ofst = attrp - rp;
    //// assert((*ofst & 1) == 0);
    //attrp += (sizeof(uint16_t)+(alignment - sizeof(uint16_t))+(len_varchar));
    //if (padding_required) {
    //  attrp += padding_size;
    //}
    //rhp->lenrec = attrp - rp;

    size_t len_vchar_aligned = l + padding_size;
    /* check space availability */
    size_t len_total = sizeof(uint32_t) + 2 * sizeof(uint16_t) + len_vchar_aligned + sizeof(uint64_t);
    if (len_total > php->lfreespace)
        return (PAG_SLOTID_MASK_ERROR);

    /* Confirmed that the page has the enough space */
    assert(php->lfreespace <= page_size - sizeof(struct pag_head));

#if 0
    std::cout << "[DEBUG] " 
      << " watermark=" << php->watermark
      << ", slotid=" << php->nalloc << ", rowid=" << rowid
      << ", len_schema=" << len_schema
      << ", len_vchar=" << len_vchar
      << ", len_vchar_aligned=" << len_vchar_aligned << ", len_total=" << len_total << std::endl;
#endif
    /* a vchar record has to have slot pointer */
    uint32_t slotid = php->nalloc;
    pag_set_oslt(pag, slotid, php->watermark, page_size);

    char *dst_base = reinterpret_cast<char*>(pag) + php->watermark;
    /* set length */
    uint16_t *vcharlenp = reinterpret_cast<uint16_t*>(dst_base);
    *vcharlenp = len_vchar;

    /* append record content */
    memcpy(dst_base + 2 * sizeof(uint16_t), (void *)valchar, len_vchar);
    if (len_vchar_aligned > len_vchar) {
      memset(dst_base + 2 * sizeof(uint16_t) + len_vchar, 0, len_vchar_aligned - len_vchar);
    }
    /* append rowid */
    uint64_t *rowid_dst = reinterpret_cast<uint64_t*>(
        dst_base + 2 * sizeof(uint16_t) + len_vchar_aligned
      );
    *rowid_dst = rowid;
    assert(reinterpret_cast<uintptr_t>(rowid_dst) % 4 == 0);
    php->watermark += (sizeof(uint16_t) * 2 + len_vchar_aligned + sizeof(uint64_t));
    php->nalloc++;
    php->lfreespace -= len_total;
    assert((php->watermark & 3) == 0); // record length must be 4-byte aligned

    /* This is useful only for the pivoted layout */
    // pag_set_varchar_rowid64(pag, slotid, rowid, page_size);

    return(slotid);
}

/* Pivoting helper functions */
static void __pag_print_string_pivoted(PAG *pag, int slotid, size_t page_size, FILE *fp)
{
    pag_head *php = (pag_head *)pag;
    const int nalloc = php->nalloc;

    /* check slot existence */
    if(slotid >= php->nalloc) {
      fprintf(fp, "pag_print_string_pivoted: slotid %d does not exist.\n", slotid);
      return;
    }

    char *buf_base = reinterpret_cast<char*>(pag) + sizeof(struct pag_head) + sizeof(uint32_t) * slotid;
    uint16_t *vcharlenp = reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(pag) + sizeof(struct pag_head) + sizeof(uint32_t) * slotid);
    uint16_t vcharlen = *vcharlenp;
    // align to 4 bytes
    uint16_t last_tile_size = (vcharlen % 4 == 0) ? 4 : vcharlen % 4;
    uint16_t vcharlen_aligned =
      vcharlen + (vcharlen % 4 == 0 ? 0 : (4 - (vcharlen % 4)));
    // uint16_t n_tiles = vcharlen_aligned / 4;
    assert(vcharlen_aligned % sizeof(uint32_t) == 0);
    size_t n_u32 = vcharlen_aligned / sizeof(uint32_t);
    char *vcharbuf_base = buf_base + nalloc * sizeof(uint32_t);
    uint64_t pk = pag_get_varchar_rowid64(pag, slotid, page_size);

    fprintf(fp, "[DEBUG](%s) slotid=%d, len=%u, pk=%lu str=",
      __func__, slotid, vcharlen, pk);
    
    //char *vcharbuf = vcharbuf_base;
    for (size_t i = 0; i < n_u32; i++) {
      const size_t n = (i == n_u32 - 1) ? last_tile_size : 4;
      for (size_t j = 0; j < n; j++) {
        size_t k = i * sizeof(uint32_t) * nalloc + j;
        fprintf(fp, "%c", vcharbuf_base[k]);
      }
    }
    fprintf(fp, "\n");
}

void pag_print_string_pivoted(PAG *pag, size_t nrecs, size_t page_size, FILE *fp)
{
  for (size_t i = 0; i < nrecs; i++) {
    __pag_print_string_pivoted(pag, i, page_size, fp);
  }
}

uint32_t pag_append_vchar_pivoted_with_pk(PAG *pag,
  size_t max_vcharlen, std::vector<std::string_view> &src_vec,
  std::vector<size_t> &aligned_lens_vec,
  std::vector<size_t> &len_orig_lens_vec,
  std::vector<uint64_t> &pk_vec,
  const size_t page_size,
  const size_t alignment)
{
    struct pag_head *php = (struct pag_head *)pag;
    // const size_t mask = alignment - 1;
    const size_t nrecs = src_vec.size();
    uint32_t slotid = 0;

    #if 0
    fprintf(stdout, "[DEBUG] pag=%p page_size=%lu\n", pag, page_size);
    fprintf(stdout, "[DEBUG] slotid=0 rowid64=%p\n",
      reinterpret_cast<char*>(pag) + page_size - PAG_FTR_SIZ - sizeof(uint64_t) * (0 + 1)
    );
    fprintf(stdout, "[DEBUG] splotid=1 rowid64=%p\n",
      reinterpret_cast<char*>(pag) + page_size - PAG_FTR_SIZ - sizeof(uint64_t) * (1 + 1)
    );
    #endif

    size_t bytes_estimated_vcharlen
      = (max_vcharlen % alignment)
          ? max_vcharlen + (alignment - (max_vcharlen % alignment)) : max_vcharlen;

    if (bytes_estimated_vcharlen % alignment > 0 ) {
      PANIC_ERROR("pag_append_vchar_pivoted_with_pk: max_vcharlen is not aligned.\n");
    }
    size_t __debug_max_vcharlen_n_u32 = bytes_estimated_vcharlen / alignment;

    //char *__debug = (reinterpret_cast<char*>(pag) + sizeof(struct pag_head) + nrecs * sizeof(uint32_t));
    if (page_size < sizeof(struct pag_head)
        + (__debug_max_vcharlen_n_u32 * nrecs * sizeof(uint32_t))) {
      fprintf(stderr, "pag_append_vchar_pivoted_with_pk: page size too small. "
        "page_size=%lu, required=%lu\n",
        page_size,
        sizeof(struct pag_head) + (max_vcharlen * nrecs * sizeof(uint32_t))
      );
      PANIC_ERROR("stop.\n");
    }

    for (size_t i = 0; i < src_vec.size(); i++) {
      const std::string_view &src = src_vec[i];
      size_t vcharlen_aligned = aligned_lens_vec[i];
      size_t vcharlen_orig = len_orig_lens_vec[i];
      uint64_t pk = pk_vec[i];

      if (vcharlen_aligned + sizeof(uint32_t) + sizeof(uint64_t) > php->lfreespace)
          return (PAG_SLOTID_MASK_ERROR);


      assert(vcharlen_aligned % alignment == 0);
      size_t n_u32 = vcharlen_aligned / sizeof(uint32_t);
      if(reinterpret_cast<uintptr_t>(reinterpret_cast<char*>(pag) + php->watermark)
          != reinterpret_cast<uintptr_t>(reinterpret_cast<char*>(pag) + i * sizeof(uint32_t) + sizeof(pag_head))
       ) {
         fprintf(stderr, "pag_append_vchar_pivoted_with_pk: "
           "watermark mismatch. watermark=%u, expected=%lu (pag_head=%lu nalloc=%lu)\n",
           php->watermark,
           i * sizeof(uint32_t) + sizeof(struct pag_head),
           sizeof(struct pag_head),
           nrecs);
         PANIC_ERROR("stop.\n");
         // php->watermark = sizeof(struct pag_head);
      }

      char *dst_base = 
        reinterpret_cast<char*>(reinterpret_cast<char*>(pag) + php->watermark);
      uint16_t * dst_lenp = reinterpret_cast<uint16_t*>(dst_base);
      
      /* copy len */
      *dst_lenp = static_cast<uint16_t>(vcharlen_orig);

#if 0
      std::cout << "[DEBUG] src=" << src << " len=" << vcharlen_orig << 
        ", len2=" << src.length() << ", len_aligned=" << vcharlen_aligned << std::endl; 
#endif
      const char *src_char = src.data();
      // char *dst_char = reinterpret_cast<char*>(dst_base + nrecs * sizeof(uint32_t));
      const uint32_t *src_u32 = reinterpret_cast<const uint32_t*>(src.data());
      uint32_t *dst_u32 = reinterpret_cast<uint32_t*>(dst_base + nrecs * sizeof(uint32_t));
#if 0
      for (size_t j = 0; j < src.size(); j++) {
        //dst_char[j] = src_char[j];
        printf("%c", src_char[j]);
      }
      printf("\n");
#endif
      for (size_t j = 0; j < n_u32; j++) {
        char *dst_str = reinterpret_cast<char*>(&dst_u32[j * nrecs]);
        for (size_t k = 0; k < 4; k++) {
          size_t idx = j * sizeof(uint32_t) + k;
          if (idx < src.size()) {
            dst_str[k] = src_char[idx];
          } else {
            dst_str[k] = 0;
          }
#if 0
          printf("\tj=%lu, k=%lu, dst=%c, src=%c\n",
            j, k,
            dst_str[k],
            (idx < src.size()) ? src_char[idx] : '-'
          );
#endif
        }
      }

      /* copy pk */
      pag_set_varchar_rowid64(pag, slotid, pk, page_size);
#if 0 
      printf("[DEBUG](%s) slotid=%u, watermark=%u, len=%u, pk=%lu rowid64=%lu src_vec=%s\n",
        __func__, slotid, php->watermark, vcharlen_orig, pk,
        pag_get_varchar_rowid64(pag, slotid, page_size), src.data());
#endif

      php->watermark += sizeof(uint32_t);
      if (php->lfreespace < vcharlen_aligned + sizeof(uint32_t) + sizeof(uint64_t)) {
        fprintf(stderr, "pag_append_vchar_pivoted_with_pk: "
          "lfreespace underflow. lfreespace=%u, required=%lu\n",
          php->lfreespace,
          vcharlen_aligned + sizeof(uint32_t) + sizeof(uint64_t));
        PANIC_ERROR("stop.\n");
      }

#if 0
      php->nalloc = src_vec.size();
      __pag_print_string_pivoted(pag, slotid, 1024*1024, stdout);
#endif
      php->lfreespace -= (vcharlen_aligned + sizeof(uint32_t) + sizeof(uint64_t));
      slotid++;
    }
    php->nalloc = src_vec.size();

    return(php->nalloc);
}


uint32_t pag_check(PAG *pp)
{
  struct pag_head *php = (struct pag_head *)pp;
  size_t i;
  for (i = 0; i < php->nalloc; i++) {
    uint32_t oslt = pag_get_oslt(pp, i, 0); 
    if (oslt & PAG_SLOTID_MASK_ERROR) {
      return 0;
    }
    REC *rp = (REC *)(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET));
    if ((reinterpret_cast<uintptr_t>(rp) & 1) != 0) {
      fprintf(stderr, "pag_check: record %lu is not aligned properly.\n", i);
      exit(EXIT_FAILURE);
    }
  }
  return(php->nalloc);
}

uint32_t pag_get_nalloc(PAG *pp)
{
  struct pag_head *php = (struct pag_head *)pp;
  return(php->nalloc);
}

uint32_t pag_get_nalloc_aligned(PAG *pp)
{
  struct pag_head *php = (struct pag_head *)pp;
  /* for decompression */
  return(php->watermark);
}

REC *pag_fetch_rec(PAG *pp, uint32_t slotid, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  /* check slot existence */
  if(slotid >= php->nalloc)
    return(NULL);

  uint32_t oslt = pag_get_oslt(pp, slotid, page_size);

  return((REC *)(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET)));
}

template<size_t N>
void pag_print_pag_data(PAG *pp, const std::array<enum rec_type, N> &types, size_t page_size, FILE *fp)
{
  struct pag_head *php = (struct pag_head *)pp;
  // struct pag_ftr_t *pfp = (struct pag_ftr_t *)(pp + page_size - PAG_FTR_SIZ);
  
  int i;
  
  fprintf(fp, "pag_print_pag_data():\n");
  fprintf(fp, "=== HDR (%lu) ===\n", PAG_HDR_SIZ);
  fprintf(fp, "nallloc: %u, owtmrk: %u, lfrspc: %u\n",
          php->nalloc, php->watermark, php->lfreespace);
  
  //fprintf(fp, "=== RCS (%lu) ===\n", PAG_SIZ - PAG_HDR_SIZ - PAG_FTR_SIZ);
  //if(rfp)
  //  recfmt_print_recfmt(rfp, fp);
  for(i=0; i<php->nalloc; i++){
    uint32_t oslt = pag_get_oslt(pp, i, page_size); 
    fprintf(fp, "  [%d] ofst: %u, ",
            i,  
            oslt & PAG_OSLT_MASK_OFFSET);
    REC *rp = (REC *)(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET));
    rec_print_rec(rp, types, fp);
  }
  
  //fprintf(fp, "=== FTR (%lu) ===\n", PAG_FTR_SIZ);
  //fprintf(fp, "  pagtype: %c, eyctch: %c\n",
  //        pfp->pagtyp, pfp->eyctch);
};

template<typename ColType>
ColType pagcol_fetch_int(PAG *pp, uint32_t slotid, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  /* check slot existence */
  if(slotid >= php->nalloc) {
    if constexpr (std::is_same_v<ColType, int32_t>) {
      return(INT32_MIN);
    } else if constexpr (std::is_same_v<ColType, int64_t>) {
      return(INT64_MIN);
    } else {
        static_assert("pagcol_fetch_int: unsupported type.");
    }
  }

  if constexpr (std::is_same_v<ColType, int32_t>) {
    return(*(ColType *)(reinterpret_cast<char*>(pp)
      + sizeof(struct pag_head) + sizeof(ColType) * slotid));
  } else if constexpr (std::is_same_v<ColType, int64_t>) {
    /* 8-byte aligned */
    return(*(ColType *)(reinterpret_cast<char*>(pp)
      + sizeof(struct pag_head) + 4 + sizeof(ColType) * slotid));
  } else {
      static_assert("pagcol_fetch_int: unsupported type.");
  }
}

/* for compression layout */
template<typename ColType>
ColType pagcol_fetch_rowid_from_head(PAG *pp, uint32_t slotid, size_t N, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  /* check slot existence */
  if(slotid >= php->nalloc)
    if constexpr (std::is_same_v<ColType, int32_t>) {
      return(INT32_MIN);
    } else if constexpr (std::is_same_v<ColType, int64_t>) {
      return(INT64_MIN);
    } else {
        static_assert("pagcol_fetch_int: unsupported type.");
    }

  return(*(ColType *)(reinterpret_cast<char*>(pp)
    + sizeof(struct pag_head) + N * sizeof(ColType) + sizeof(uint64_t) * slotid));
}

char *pagcol_fetch_char(PAG *pp, uint32_t slotid, size_t len_char, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  /* check slot existence */
  if(slotid >= php->nalloc)
    return(NULL);

  constexpr size_t alignment = 4;
  len_char = (len_char % alignment)
      ? len_char + (alignment - (len_char % alignment)) : len_char;

  return(reinterpret_cast<char*>(pp) + sizeof(struct pag_head) + len_char * slotid);
}

char *pagcol_fetch_vchar_base(PAG *pp, uint32_t slotid, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  /* check slot existence */
  if(slotid >= php->nalloc)
    return(NULL);

  uint32_t oslt = pag_get_oslt(pp, slotid, page_size);

  return(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET));
}

uint16_t pagcol_fetch_vchar_len(PAG *pp, uint32_t slotid, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  /* check slot existence */
  if(slotid >= php->nalloc)
    return(0);

  uint32_t oslt = pag_get_oslt(pp, slotid, page_size);

  return(*(uint16_t *)(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET)));
}

char *pagcol_fetch_vchar(PAG *pp, uint32_t slotid, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  /* check slot existence */
  if(slotid >= php->nalloc)
    return(NULL);

  uint32_t oslt = pag_get_oslt(pp, slotid, page_size);

  return(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET) + sizeof(uint32_t));
}

uint64_t pagcol_fetch_vchar_rowid(PAG *pp, uint32_t slotid, size_t len, size_t page_size)
{
  struct pag_head *php = (struct pag_head *)pp;

  /* check slot existence */
  if(slotid >= php->nalloc)
    return(INVALID_ROWID);

  uint32_t oslt = pag_get_oslt(pp, slotid, page_size);

  constexpr size_t alignment = 4;
  len = (len % alignment == 0) ? len : (len + (alignment - (len % alignment)));

  //return(*(uint64_t *)(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET) + sizeof(uint32_t) + len));
  return *reinterpret_cast<uint64_t*>(pagcol_fetch_vchar(pp, slotid, page_size) + len);
}

void pagcol_print_pag_data_varchar(PAG *pp, size_t nrecs, size_t page_size, FILE *fp)
{
  struct pag_head *php = (struct pag_head *)pp;
  
  int i;
  
  fprintf(fp, "pagcol_print_pag_data_varchar():\n");
  fprintf(fp, "=== HDR (%lu) ===\n", PAG_HDR_SIZ);
  fprintf(fp, "nallloc: %u, owtmrk: %u, lfrspc: %u\n",
          php->nalloc, php->watermark, php->lfreespace);
  
  for(i=0; i<nrecs; i++){
    uint64_t rowid64 = pag_get_varchar_rowid64(pp, i, page_size); 
    fprintf(fp, "  [%d] rowid64: %lu\n",
            i,  
            rowid64);
    // REC *rp = (REC *)(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET));
    // rec_print_rec(rp, types, fp);
  }
};

/* for fixed len data - varchar is not supported */
template<typename T>
void pagcol_print_pag_data_int(PAG *pp, size_t page_size, FILE *fp)
{
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>,
    "T must be int32_t or int64_t!");

  struct pag_head *php = (struct pag_head *)pp;
  
  int i;
  
  fprintf(fp, "pagcol_print_pag_data_varchar():\n");
  fprintf(fp, "=== HDR (%lu) ===\n", PAG_HDR_SIZ);
  fprintf(fp, "nallloc: %u, owtmrk: %u, lfrspc: %u\n",
          php->nalloc, php->watermark, php->lfreespace);
  
  size_t nrecs = php->nalloc;

  for(i=0; i<nrecs; i++){
    if constexpr (std::is_same_v<T, int32_t>) {
      int32_t *valp = reinterpret_cast<int32_t*>(
        reinterpret_cast<char*>(pp) + sizeof(struct pag_head) + sizeof(uint32_t) * i
      );
      fprintf(fp, "  [%d] value: %d, ", i, *valp);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      int64_t *valp = reinterpret_cast<int64_t*>(
        reinterpret_cast<char*>(pp) + sizeof(struct pag_head) + sizeof(uint64_t) * i
      );
      fprintf(fp, "  [%d] value: %ld, ", i, *valp);
    } else {
      static_assert("pagcol_print_pag_data: unsupported type.");
    }
 
    uint64_t rowid64 = pag_get_varchar_rowid64(pp, i, page_size); 
    fprintf(fp, "  [%d] rowid64: %lu\n",
            i,  
            rowid64);
    // REC *rp = (REC *)(reinterpret_cast<char*>(pp) + (oslt & PAG_OSLT_MASK_OFFSET));
    // rec_print_rec(rp, types, fp);
  }
};