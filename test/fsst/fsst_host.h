#pragma once
#include "fsst_page.h"
#include "fsst.h"
#include <cstdint>
#include <vector>

// ============================================================
// CPU Compression API for FSST String Pages
//
// Usage:
//   // 1. Train FSST + compress strings
//   fsst_encoder_t* enc = fsst_create(n, lens, ptrs, 0);
//   fsst_decoder_t  st  = fsst_decoder(enc);
//   fsst_compress(enc, n, lens, ptrs, ...);
//
//   // 2. Serialize symbol table
//   uint8_t raw_symtab[FSST_SYMTAB_TOTAL];
//   fsst_serialize_symbol_table(st, raw_symtab);
//
//   // 3. Pack into pages (loop)
//   uint32_t offset = 0;
//   while (offset < nrecs) {
//       pag_init(pag, page_size);
//       uint32_t n = pagcol_append_batch_unordered_column_vchar_fsst(
//           pag, raw_symtab, nrecs - offset,
//           comp_ptrs + offset, comp_lens + offset,
//           max_comp_block_payload, comp_bytes, page_size);
//       // store page ...
//       offset += n;
//   }
// ============================================================

// Serialize fsst_decoder_t to raw bytes for page embedding.
void fsst_serialize_symbol_table(
    const fsst_decoder_t& symbol_table,
    uint8_t*              raw_symtab);    // [FSST_SYMTAB_TOTAL]

// Pack FSST-compressed strings into one page using pag_head format.
// Returns the number of records actually packed.
// Records that don't fit should be packed into the next page.
uint32_t pagcol_append_batch_unordered_column_vchar_fsst(
    PAG*                        pag,
    const uint8_t*              raw_symtab,          // [FSST_SYMTAB_TOTAL]
    uint32_t                    nrecs,               // records available
    const unsigned char* const* comp_ptrs,
    const size_t*               comp_lens,
    uint32_t                    max_comp_block_payload,
    uint32_t&                   compressed_page_bytes,  // out
    size_t                      page_size);

// Build FsstCompBlockKernelMeta from page data on CPU.
// Appends comp block metas to out_metas, starting global record index at rec_base.
// Returns number of records in this page (pag_head.nalloc).
uint32_t fsst_build_comp_block_metas_from_page(
    const uint8_t*                        page_data,
    uint32_t                              page_byte_offset,
    uint32_t                              rec_base,
    std::vector<FsstCompBlockKernelMeta>& out_metas);
