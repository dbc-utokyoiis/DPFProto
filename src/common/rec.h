#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "common/err.h"

typedef void REC;

struct rec_head {
    uint16_t lenrec;   /* length of record*/
    uint16_t arofst[]; /* array of offsets to attributes */
};

enum rec_type {
    REC_ATTR_INT16 = 0, // 2 bytes
    REC_ATTR_INT32,     // 4 bytes
    REC_ATTR_INT64,     // 8 bytes
    REC_ATTR_DATE,      // 4 bytes
    REC_ATTR_DECIMAL,   // 4 bytes
    REC_ATTR_CHAR,      // fixed length CHAR(n) -> n bytes
    REC_ATTR_VCHAR,     // variable length VCHAR(n) -> max n bytes
};


/* Get the size of the record header */
/* These code can be used on GPU and CPU */
template <size_t N>
inline uint16_t rec_get_attr_int16(REC *rp, const std::array<size_t, N> &sizes, int idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  if(idx>=N){ PANIC_PERROR("rec_get_attr_int16"); }
  return(*(uint16_t *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
}

template <size_t N>
inline uint32_t rec_get_attr_int32(REC *rp, const std::array<size_t, N> &sizes, int idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  if(idx>=N){ PANIC_PERROR("rec_get_attr_int32"); }
  return(*(uint32_t *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
}

template <size_t N>
inline uint64_t rec_get_attr_int64(REC *rp, const std::array<size_t, N> &sizes, int idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  if(idx>=N){ PANIC_PERROR("rec_get_attr_int64"); }
  return(*(uint64_t *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
}

template <size_t N>
inline char *rec_get_attr_chr(REC *rp, const std::array<size_t, N> &sizes, int idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  if(idx>=N){ PANIC_PERROR("rec_get_attr_chr"); }
  return((char *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
}

template <size_t N>
inline char *rec_get_attr_vchr(REC *rp, const std::array<size_t, N> &sizes, int idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  if(idx>=N){ PANIC_PERROR("rec_get_attr_chr"); }
  return((char *)(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx] + sizeof(uint16_t)));
}

template <size_t N>
inline int rec_get_size_vchr(REC *rp, const std::array<size_t, N> &sizes, int idx)
{
  struct rec_head *rhp = (struct rec_head *)rp;
  return(reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t *>(rp) + rhp->arofst[idx]));
}

#if 0
template <size_t N>
inline int rec_get_attr_nlflag(REC *rp, std::array<size_t, N> &sizes, int idx)
{
  uint16_t *nlflag = reinterpret_cast<uint16_t*>((reinterpret_cast<uint8_t*>(rp) + sizeof(uint16_t) + N * sizeof(uint16_t) + (idx / 16) * sizeof(uint16_t)));
  return(*nlflag & (1 << (idx % 16)));
}
#endif

/* CPU-only set functions */
template <size_t N>
REC *rec_init(const std::array<size_t, N> &sizes, char *buf, int len) 
{
  REC *rp = reinterpret_cast<REC *>(buf);
  struct rec_head *rhp = reinterpret_cast<struct rec_head *>(rp);
  const int alignment = 4; // align to 4 bytes

  memset(buf, 0, len);
  rhp->lenrec = sizeof(uint16_t) + N * sizeof(uint16_t) + ((N + 15) / 16) * sizeof(uint16_t);

  if (rhp->lenrec % alignment) {
    rhp->lenrec += (alignment - (rhp->lenrec % alignment));
  }

  return(rp);
}

template <size_t N>
REC *rec_set_attr_int16(const std::array<size_t, N> &sizes, char *buf, int len, int idx, uint16_t val) 
{
  REC *_rp = (REC *)buf;
  uint8_t *rp = (uint8_t *)_rp;
  struct rec_head *rhp = (struct rec_head *)rp;
  uint16_t *ofst = &(rhp->arofst[idx]);
  uint8_t *attrp = rp + rhp->lenrec;

  if (reinterpret_cast<uintptr_t>(attrp) & 1 == 0) { // confirming that attrp is even address
    PANIC_ERROR("rec_reset_attr_int32:attrp is not even address.\n");
  }
  if(attrp + sizeof(uint16_t) - rp > len) {
    PANIC_ERROR("rec_set_attr_int16:buffer overflow.\n");
  }
  if(*ofst) {
    PANIC_ERROR("rec_set_attr_int16:attribute exists.\n");
  }
  *(uint16_t *)attrp = val; 
  *ofst = attrp - rp;
  attrp += sizeof(uint16_t);
  rhp->lenrec = attrp - rp;

  return(rp);
}

template <size_t N>
REC *rec_set_attr_int32(const std::array<size_t, N> &sizes, char *buf, int len, int idx, uint32_t val) 
{
  REC *_rp = (REC *)buf;
  uint8_t *rp = (uint8_t *)_rp;
  struct rec_head *rhp = (struct rec_head *)rp;
  uint16_t *ofst = &(rhp->arofst[idx]);
  uint8_t *attrp = rp + rhp->lenrec;

  if (reinterpret_cast<uintptr_t>(attrp) & 1 == 0) { // confirming that attrp is even address
    PANIC_ERROR("rec_reset_attr_int32:attrp is not even address.\n");
  }
  if(attrp + sizeof(uint32_t) - rp > len) {
    PANIC_ERROR("rec_set_attr_int32:buffer overflow.\n");
  }
  if(*ofst)
    PANIC_ERROR("rec_set_attr_int32:attribute exists.\n");
  *(uint32_t *)attrp = val; 
  *ofst = attrp - rp;
  attrp += sizeof(uint32_t);
  rhp->lenrec = attrp - rp;

  return(rp);
}

template <size_t N>
REC *rec_reset_attr_int32(const std::array<size_t, N> &sizes, char *buf, int len, int idx, uint32_t val) 
{
  REC *_rp = (REC *)buf;
  uint8_t *rp = (uint8_t *)_rp;
  struct rec_head *rhp = (struct rec_head *)rp;
  uint16_t *ofst = &(rhp->arofst[idx]);
  uint8_t *attrp = reinterpret_cast<uint8_t*>(rp) + rhp->arofst[idx];

  if (reinterpret_cast<uintptr_t>(attrp) & 1 == 0) { // confirming that attrp is even address
    PANIC_ERROR("rec_reset_attr_int32:attrp is not even address.\n");
  }
  if(!*ofst) {
    PANIC_ERROR("rec_reset_attr_int32:attribute does not exist.\n");
  }

  *(uint32_t *)attrp = val; 

  return(rp);
}


template <size_t N>
REC *rec_set_attr_int64(const std::array<size_t, N> &sizes, char *buf, int len, int idx, uint64_t val)
{
  REC *_rp = (REC *)buf;
  uint8_t *rp = (uint8_t *)_rp;
  struct rec_head *rhp = (struct rec_head *)rp;
  uint16_t *ofst = &(rhp->arofst[idx]);
  uint8_t *attrp = rp + rhp->lenrec;

  if (reinterpret_cast<uintptr_t>(attrp) & 1 == 0) { // confirming that attrp is even address
    PANIC_ERROR("rec_set_attr_int64:attrp is not even address.\n");
  }
  if(attrp + sizeof(uint64_t) - rp > len) {
    PANIC_ERROR("rec_set_attr_int64:buffer overflow.\n");
  }
  if(*ofst) {
    PANIC_ERROR("rec_set_attr_int64:attribute exists.\n");
  }
  *(uint64_t *)attrp = val;
  *ofst = attrp - rp;
  attrp += sizeof(uint64_t);
  rhp->lenrec = attrp - rp;

  return(rp);
}

template <size_t N>
REC *rec_set_attr_char(const std::array<size_t, N> &sizes, char *buf, int len, int idx, char *val, size_t siz)
{
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
}

template <size_t N>
REC *rec_set_attr_vchar(const std::array<size_t, N> &sizes, char *buf, int len, int idx, char *val, size_t siz) 
{
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
}

template <size_t N>
REC *rec_set_attr_nlflag(std::array<size_t, N> &sizes, char *buf, int len, int idx)
{
  REC *_rp = (REC *)buf;
  uint8_t *rp = (uint8_t *)_rp;
  struct rec_head *rhp = (struct rec_head *)rp;
  uint16_t *nlflag;

  if(rhp->arofst[idx] == 0) {
    PANIC_ERROR("rec_set_attr_nlflag: attribute not yet set.\n");
  }

  //uint16_t *nlflag = reinterpret_cast<uint16_t*>((reinterpret_cast<uint8_t*>(rp) + sizeof(uint16_t) + N * sizeof(uint16_t) + (idx / 16) * sizeof(uint16_t)));
  nlflag = reinterpret_cast<uint16_t*>(rp + sizeof(uint16_t) + N * sizeof(uint16_t) + (idx / 16) * sizeof(uint16_t));
  *nlflag |= (1 << idx % 16);

  return(rp);
}

template <size_t N>
void rec_print_rec(REC *rp, const std::array<enum rec_type, N> &types, FILE *fp) 
{
  int i, l;
  uint16_t vchar_len;
  const int alignment = 4; /* only for vchar */
  struct rec_head *rhp = reinterpret_cast<struct rec_head *>(rp);

  fprintf(fp, "REC: ");
  for(i=0; i<N; i++){
    fprintf(fp, "[%02d]", i);
    //if(rec_get_attr_nlflag(rp, rfp, i)){ 
    //  fprintf(fp, "(null)|");
    //  continue;
    //}    
    switch(types[i]){
    case REC_ATTR_INT16:
      // fprintf(fp, "<+%u>%u ", rhp->arofst[i], *(uint16_t *)(rp + rhp->arofst[i]));
      fprintf(fp, "%u|", *reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t*>(rp) + rhp->arofst[i]));
      break;
    case REC_ATTR_INT32:
      // fprintf(fp, "<+%u>%u ", rhp->arofst[i], *(uint32_t *)(rp + rhp->arofst[i]));
      fprintf(fp, "%u|", *reinterpret_cast<uint32_t *>(reinterpret_cast<uint8_t*>(rp) + rhp->arofst[i]));
      break;
    case REC_ATTR_INT64:
      // fprintf(fp, "<+%u>%lu ", rhp->arofst[i], *(uint64_t *)(rp + rhp->arofst[i]));
      fprintf(fp, "%lu|", *reinterpret_cast<uint64_t *>(reinterpret_cast<uint8_t*>(rp) + rhp->arofst[i]));
      break;
    case REC_ATTR_CHAR:
      if(i+1 == N){ l = rhp->lenrec - rhp->arofst[i]; }
      else{ l = rhp->arofst[i+1] - rhp->arofst[i]; };
      // fprintf(fp, "<+%u>%.*s ", rhp->arofst[i], l, (char *)(rp + rhp->arofst[i]));
      /* NOTE and FIXME: use sizes[] array to print CHAR correctly. */
      fprintf(fp, "%.*s|", l, reinterpret_cast<char*>(reinterpret_cast<uint8_t*>(rp) + rhp->arofst[i]));
      break;
    case REC_ATTR_VCHAR:
      vchar_len = *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(rp) + rhp->arofst[i]);
      // fprintf(fp, "<+%u>%.*s ", rhp->arofst[i], l, (char *)(rp + rhp->arofst[i]));
      fprintf(fp, "%.*s|", vchar_len, reinterpret_cast<char*>(reinterpret_cast<uint8_t*>(rp) + rhp->arofst[i] + alignment));
      break;
    default:
        fprintf(fp, "unknown type %d|", types[i]);
    }
  }
  fprintf(fp, "\n");
}

//REC *rec_set_attr_int16(RECFMT *rfp, char *buf, int len, int idx, uint16_t val);
//REC *rec_set_attr_int32(RECFMT *rfp, char *buf, int len, int idx, uint32_t val);
//REC *rec_set_attr_int64(RECFMT *rfp, char *buf, int len, int idx, uint64_t val);
//REC *rec_set_attr_chr(RECFMT *rfp, char *buf, int len, int idx, char *val, int siz);
//REC *rec_set_attr_vchr(RECFMT *rfp, char *buf, int len, int idx, char *val, int siz);
//REC *rec_set_attr_nlflag(RECFMT *rfp, char *buf, int len, int idx);