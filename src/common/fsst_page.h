#pragma once
#include "pag_core.h"
#include <cstdint>
#include <cstring>

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
    uint64_t page_byte_offset;        // offset of page within d_pages buffer
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

// ============================================================
// FSST_ROWID Page Format (FSST compressed strings + PFOR64 rowids)
//
//  +-------------------------------------+
//  | pag_head (12B)                      |
//  +-------------------------------------+
//  | uint32_t n_comp_blocks (4B)         |
//  +-------------------------------------+
//  | FsstCompBlockDirEntry[n_comp_blocks]|
//  +-------------------------------------+
//  | FSST Symbol Table (2296B)           |
//  +-------------------------------------+
//  | uint32_t pfor64_rowid_offset (4B)   |  <-- byte offset to PFOR64 section (0=none)
//  +-------------------------------------+
//  | Comp Block 0: offset_table + data   |
//  | Comp Block 1: ...                   |
//  +-------------------------------------+
//  | PFOR64 rowid section:               |
//  |   uint32_t block_start[nblocks+1]   |
//  |   [padding to 8B alignment]         |
//  |   PFOR64 encoded rowid data         |
//  +-------------------------------------+
//
// pfor64_rowid_offset: byte offset from page start to the PFOR64 section.
//   When 0, no rowid data is present (backward compatible with plain FSST).
// nblocks = ceil(nalloc / 128)
// ============================================================

// Offset of the pfor64_rowid_offset field relative to symtab_end
// (i.e., it's placed right after the symbol table, before comp blocks)
static constexpr uint32_t FSST_PFOR64_OFFSET_SIZE = sizeof(uint32_t);  // 4B

// Compute the effective end-of-FSST-data boundary for a page.
// For plain FSST pages: returns watermark (whole page is FSST data).
// For FSST_ROWID pages: returns pfor64_rowid_offset (PFOR64 section follows).
//
// Detection: the 4 bytes at the pfor64_offset field position are read.
// For FSST_ROWID, this value is a valid byte offset (> its own position).
// For plain FSST, those bytes are part of the first comp block's offset table
// (a small value, always less than the field's position in the page).
#ifdef __CUDACC__
__device__ __host__
#endif
inline uint32_t fsst_page_string_data_end(
    const char* page_data,
    uint32_t watermark,
    uint32_t n_comp_blocks)
{
    uint32_t pfor64_off_pos = FSST_PAGE_NCB_OFFSET + sizeof(uint32_t)
        + n_comp_blocks * sizeof(FsstCompBlockDirEntry)
        + FSST_SYMTAB_TOTAL;
    uint32_t val = *reinterpret_cast<const uint32_t*>(page_data + pfor64_off_pos);
    // FSST_ROWID: val is a large offset pointing past comp blocks
    // Plain FSST: val is the first comp block's offset_table[0..1] (tiny value)
    if (val > pfor64_off_pos && val <= watermark)
        return val;
    return watermark;
}
