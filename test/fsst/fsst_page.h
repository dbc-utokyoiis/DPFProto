#pragma once
#include <cstdint>
#include <cstring>

// pag_head and PAG from common/pag.h (copied here to avoid include-chain issues)
struct pag_head {
    uint32_t nalloc;
    uint32_t watermark;
    uint32_t lfreespace;
};

typedef void PAG;

inline void pag_init(PAG *pag, size_t page_size)
{
    pag_head *php = (pag_head *)pag;
    php->nalloc = 0;
    php->watermark = sizeof(pag_head);
    php->lfreespace = page_size - sizeof(pag_head);
}

// ============================================================
// FSST String Page Format
//
//  +-------------------------------------+
//  | pag_head (12B)                      |
//  |   nalloc, watermark, lfreespace     |
//  +-------------------------------------+
//  | uint32_t n_comp_blocks (4B)         |
//  +-------------------------------------+
//  | FsstCompBlockDirEntry[n_comp_blocks]|
//  |   (8B each)                         |
//  +-------------------------------------+
//  | FSST Symbol Table (2296B)           |
//  |   len[256] + symbol[255] (u64x255) |
//  +-------------------------------------+
//  | Comp Block 0:                       |
//  |   uint16_t offset_table[nrecs+1]   |
//  |   compressed records (packed)       |
//  +-------------------------------------+
//  | Comp Block 1: ...                   |
//  +-------------------------------------+
//
// pag_head.nalloc     = total records in this page
// pag_head.watermark  = byte offset of end of data
// pag_head.lfreespace = remaining free space
// ============================================================

struct FsstCompBlockDirEntry {
    uint32_t offset;       // byte offset from page start
    uint32_t nrecs;        // records in this comp block
};

// GPU kernel launch metadata (precomputed per comp block)
struct FsstCompBlockKernelMeta {
    uint32_t page_byte_offset;        // offset of page within d_pages buffer
    uint32_t symtab_byte_offset;      // page-relative offset to symbol table
    uint32_t comp_block_byte_offset;  // page-relative offset to comp block data
    uint32_t nrecs;
    uint32_t comp_block_data_size;    // offset_table + compressed data bytes
    uint32_t rec_base;                // global record index of first record
};

// n_comp_blocks is stored at offset sizeof(pag_head) in the page
static constexpr uint32_t FSST_PAGE_NCB_OFFSET = sizeof(pag_head);

// Symbol table layout in page: len[256] + symbol[255*8] = 2296 bytes
static constexpr uint32_t FSST_SYMTAB_LEN_BYTES = 256;   // 255 used + 1 pad
static constexpr uint32_t FSST_SYMTAB_SYM_BYTES = 255 * 8;
static constexpr uint32_t FSST_SYMTAB_TOTAL     = FSST_SYMTAB_LEN_BYTES + FSST_SYMTAB_SYM_BYTES;  // 2296B

// Shared memory: symbol table occupies [0..2295], padded to 2496 for alignment
static constexpr uint32_t FSST_SMEM_SYMTAB_OFFSET = 2496;

// FNV-1a checksum (usable from both CPU and GPU)
#ifdef __CUDACC__
__host__ __device__
#endif
inline uint32_t fnv1a(const char* data, uint32_t len) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < len; i++) {
        h ^= (uint8_t)data[i];
        h *= 16777619u;
    }
    return h;
}
