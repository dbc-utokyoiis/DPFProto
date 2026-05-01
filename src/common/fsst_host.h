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

// Result of compress_strings_with_fsst().
struct FsstCompressResult {
    uint32_t packed_count;          // records actually packed into the page
    uint32_t compressed_page_bytes; // total bytes written to the page
};

// Train FSST on the given strings, compress, serialize symbol table,
// and pack compressed records into one FSST page in dst_buffer.
// Returns packed_count (may be < nrecs if page is full) and compressed_page_bytes.
// Similar pattern to compress_int_with_pfor + pagcol_append_batch_unordered_column_int32_comp.
FsstCompressResult compress_strings_with_fsst(
    const unsigned char* const* ptrs,
    const size_t*               lens,
    size_t                      nrecs,
    void*                       dst_buffer,    // page buffer (>= page_size)
    size_t                      page_size,
    uint32_t                    max_comp_block_payload = 9000);

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
    uint64_t                              page_byte_offset,
    uint32_t                              rec_base,
    std::vector<FsstCompBlockKernelMeta>& out_metas);

// CPU-side decompression of an FSST page.
// Decompresses all comp blocks and writes records to flat output buffer.
// Returns total number of records decompressed.
uint32_t fsst_decompress_page_cpu(
    const uint8_t* page_data,
    size_t         page_size,
    uint8_t*       output,           // flat output buffer
    size_t*        output_lens,      // per-record decompressed length (optional, may be nullptr)
    size_t         max_output_size);

// ============================================================
// FSST + PFOR64 Rowid API (FSST_ROWID compression method)
// ============================================================

// Pre-compressed PFOR64 rowid data, passed to page packing functions.
// The caller (tpch_loader.cu) compresses rowids using binPack64() and
// passes the result here for page embedding.
struct Pfor64CompressedRowids {
    const uint64_t* encoded_data;   // binPack64 output (nulong words)
    const uint32_t* block_starts;   // block_start[nblocks+1]
    uint32_t        nulong;         // number of uint64_t words in encoded_data
    uint32_t        nblocks;        // number of PFOR64 blocks (= ceil(nrecs_padded/128))
    uint32_t        nrecs_padded;   // nrecs rounded up to multiple of 128
};

// Pack FSST-compressed strings + pre-compressed PFOR64 rowids into one page.
// Returns the number of records actually packed.
// The pfor64_rowid_offset field is written between symtab and first comp block.
uint32_t pagcol_append_batch_unordered_column_vchar_fsst_rowid(
    PAG*                        pag,
    const uint8_t*              raw_symtab,
    uint32_t                    nrecs,
    const unsigned char* const* comp_ptrs,
    const size_t*               comp_lens,
    const Pfor64CompressedRowids& pfor64_rowids,
    uint32_t                    max_comp_block_payload,
    uint32_t&                   compressed_page_bytes,
    size_t                      page_size);

// Locate the PFOR64 rowid section in an FSST_ROWID page.
// Returns true if found, filling out encoded_data/block_starts pointers and counts.
// The caller can then use binUnpack64() to decode.
bool fsst_rowid_locate_pfor64_section(
    const uint8_t* page_data,
    size_t         page_size,
    const uint64_t** out_encoded_data,    // pointer to PFOR64 encoded words
    const uint32_t** out_block_starts,    // pointer to block_start[nblocks+1]
    uint32_t*        out_nrecs_padded);   // nalloc rounded to 128
