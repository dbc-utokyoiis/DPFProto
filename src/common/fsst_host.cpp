#include "fsst_host.h"
#include <cassert>
#include <cstring>
#include <vector>

FsstCompressResult compress_strings_with_fsst(
    const unsigned char* const* ptrs,
    const size_t*               lens,
    size_t                      nrecs,
    void*                       dst_buffer,
    size_t                      page_size,
    uint32_t                    max_comp_block_payload)
{
    // 1. FSST train + compress
    fsst_encoder_t* enc = fsst_create(nrecs, lens, const_cast<const unsigned char**>(ptrs), 0);
    fsst_decoder_t  dec = fsst_decoder(enc);

    size_t total_raw = 0;
    for (size_t i = 0; i < nrecs; i++) total_raw += lens[i];

    size_t comp_buf_size = nrecs * 7 + 2 * total_raw;
    std::vector<unsigned char>  comp_buf(comp_buf_size);
    std::vector<size_t>         comp_lens(nrecs);
    std::vector<unsigned char*> comp_ptrs(nrecs);
    fsst_compress(enc, nrecs, lens, const_cast<const unsigned char**>(ptrs),
                  comp_buf_size, comp_buf.data(),
                  comp_lens.data(), comp_ptrs.data());
    fsst_destroy(enc);

    // 2. Serialize symbol table
    uint8_t raw_symtab[FSST_SYMTAB_TOTAL];
    fsst_serialize_symbol_table(dec, raw_symtab);

    // 3. Pack into FSST page
    pag_init(dst_buffer, page_size);
    uint32_t compressed_page_bytes = 0;
    uint32_t packed = pagcol_append_batch_unordered_column_vchar_fsst(
        dst_buffer, raw_symtab, nrecs,
        (const unsigned char* const*)comp_ptrs.data(),
        comp_lens.data(),
        max_comp_block_payload,
        compressed_page_bytes, page_size);

    return {packed, compressed_page_bytes};
}

void fsst_serialize_symbol_table(
    const fsst_decoder_t& symbol_table,
    uint8_t*              raw_symtab)
{
    memset(raw_symtab, 0, FSST_SYMTAB_TOTAL);
    memcpy(raw_symtab, symbol_table.len, 255);
    memcpy(raw_symtab + FSST_SYMTAB_LEN_BYTES,
           symbol_table.symbol, 255 * sizeof(uint64_t));
}

uint32_t pagcol_append_batch_unordered_column_vchar_fsst(
    PAG*                        pag,
    const uint8_t*              raw_symtab,
    uint32_t                    nrecs,
    const unsigned char* const* comp_ptrs,
    const size_t*               comp_lens,
    uint32_t                    max_comp_block_payload,
    uint32_t&                   compressed_page_bytes,
    size_t                      page_size)
{
    pag_head* php = (pag_head*)pag;
    char* base = (char*)pag;

    assert(php->nalloc == 0);
    assert((php->watermark & 3) == 0);

    // ── Phase 1: Assign records to comp blocks ──
    struct CbInfo { uint32_t first; uint32_t n; uint32_t data_bytes; };

    CbInfo cbs[4096];
    uint32_t ncbs = 0;

    // Stop creating comp blocks once total data exceeds page capacity
    uint32_t fixed_overhead_est = sizeof(pag_head) + sizeof(uint32_t) + FSST_SYMTAB_TOTAL;
    uint32_t cumulative_data = 0;

    uint32_t cb_first = 0, cb_data = 0, cb_n = 0;

    for (uint32_t i = 0; i < nrecs; i++) {
        uint32_t overhead = (cb_n + 2) * sizeof(uint16_t);
        uint32_t total = overhead + cb_data + (uint32_t)comp_lens[i];
        if (cb_n > 0 && total > max_comp_block_payload) {
            assert(ncbs < 4096);
            cbs[ncbs++] = {cb_first, cb_n, cb_data};
            cumulative_data += sizeof(FsstCompBlockDirEntry) + (cb_n + 1) * sizeof(uint16_t) + cb_data;
            if (fixed_overhead_est + cumulative_data >= page_size) break;
            cb_first = i;
            cb_data = 0;
            cb_n = 0;
        }
        cb_data += (uint32_t)comp_lens[i];
        cb_n++;
    }
    if (cb_n > 0 && ncbs < 4096) {
        cbs[ncbs++] = {cb_first, cb_n, cb_data};
    }

    // ── Phase 2: Determine how many comp blocks fit in this page ──
    // Layout: pag_head(12) + n_comp_blocks(4) + dir(8*N) + symtab(2296) + comp blocks
    uint32_t fixed_overhead = sizeof(pag_head) + sizeof(uint32_t) + FSST_SYMTAB_TOTAL;
    uint32_t used = fixed_overhead;
    uint32_t cbs_in_page = 0;
    uint32_t recs_in_page = 0;

    for (uint32_t ci = 0; ci < ncbs; ci++) {
        uint32_t cb_bytes = (cbs[ci].n + 1) * sizeof(uint16_t) + cbs[ci].data_bytes;
        uint32_t tentative = used + sizeof(FsstCompBlockDirEntry) + cb_bytes;
        tentative = (tentative + 3) & ~3u;
        if (tentative > page_size) break;
        used = tentative;
        cbs_in_page++;
        recs_in_page += cbs[ci].n;
    }

    if (cbs_in_page == 0) return 0;

    // ── Phase 3: Write page ──
    uint32_t wpos = sizeof(pag_head);

    // n_comp_blocks
    *(uint32_t*)(base + wpos) = cbs_in_page;
    wpos += sizeof(uint32_t);

    // Directory entries (placeholder, fill offsets later)
    uint32_t dir_offset = wpos;
    wpos += cbs_in_page * sizeof(FsstCompBlockDirEntry);

    // Symbol table
    uint32_t symtab_offset = wpos;
    memcpy(base + wpos, raw_symtab, FSST_SYMTAB_TOTAL);
    wpos += FSST_SYMTAB_TOTAL;

    // Comp blocks
    for (uint32_t ci = 0; ci < cbs_in_page; ci++) {
        auto& cb = cbs[ci];

        // Directory entry
        FsstCompBlockDirEntry* dir = (FsstCompBlockDirEntry*)(base + dir_offset) + ci;
        dir->offset = wpos;
        dir->nrecs = cb.n;

        // Offset table
        uint16_t* otbl = (uint16_t*)(base + wpos);
        uint16_t acc = 0;
        for (uint32_t r = 0; r < cb.n; r++) {
            otbl[r] = acc;
            acc += (uint16_t)comp_lens[cb.first + r];
        }
        otbl[cb.n] = acc;

        uint32_t otbl_bytes = (cb.n + 1) * sizeof(uint16_t);
        uint32_t data_start = wpos + otbl_bytes;

        // Compressed records
        uint32_t dpos = 0;
        for (uint32_t r = 0; r < cb.n; r++) {
            memcpy(base + data_start + dpos,
                   comp_ptrs[cb.first + r], comp_lens[cb.first + r]);
            dpos += (uint32_t)comp_lens[cb.first + r];
        }

        wpos = data_start + dpos;
        wpos = (wpos + 3) & ~3u;
        assert(wpos <= page_size);
    }

    // Update pag_head
    php->nalloc = recs_in_page;
    php->watermark = wpos;
    php->lfreespace = page_size - wpos;

    compressed_page_bytes = wpos;

    return recs_in_page;
}

uint32_t fsst_build_comp_block_metas_from_page(
    const uint8_t*                        page_data,
    uint64_t                              page_byte_offset,
    uint32_t                              rec_base,
    std::vector<FsstCompBlockKernelMeta>& out_metas)
{
    const pag_head* php = (const pag_head*)page_data;
    uint32_t n_comp_blocks = *(const uint32_t*)(page_data + FSST_PAGE_NCB_OFFSET);

    uint32_t dir_offset = sizeof(pag_head) + sizeof(uint32_t);
    uint32_t symtab_offset = dir_offset + n_comp_blocks * sizeof(FsstCompBlockDirEntry);

    uint32_t cb_rec_base = rec_base;
    for (uint32_t ci = 0; ci < n_comp_blocks; ci++) {
        const FsstCompBlockDirEntry* dir =
            (const FsstCompBlockDirEntry*)(page_data + dir_offset) + ci;

        FsstCompBlockKernelMeta meta;
        meta.page_byte_offset = page_byte_offset;
        meta.symtab_byte_offset = symtab_offset;
        meta.comp_block_byte_offset = dir->offset;
        meta.nrecs = dir->nrecs;

        uint32_t cb_end;
        if (ci + 1 < n_comp_blocks) {
            const FsstCompBlockDirEntry* next =
                (const FsstCompBlockDirEntry*)(page_data + dir_offset) + ci + 1;
            cb_end = next->offset;
        } else {
            cb_end = php->watermark;
        }
        meta.comp_block_data_size = cb_end - dir->offset;
        meta.rec_base = cb_rec_base;

        out_metas.push_back(meta);
        cb_rec_base += dir->nrecs;
    }
    return php->nalloc;
}

// ============================================================
// FSST_ROWID: FSST strings + PFOR64 rowids in one page
// ============================================================

uint32_t pagcol_append_batch_unordered_column_vchar_fsst_rowid(
    PAG*                        pag,
    const uint8_t*              raw_symtab,
    uint32_t                    nrecs,
    const unsigned char* const* comp_ptrs,
    const size_t*               comp_lens,
    const Pfor64CompressedRowids& pfor64_rowids,
    uint32_t                    max_comp_block_payload,
    uint32_t&                   compressed_page_bytes,
    size_t                      page_size)
{
    pag_head* php = (pag_head*)pag;
    char* base = (char*)pag;

    assert(php->nalloc == 0);
    assert((php->watermark & 3) == 0);

    // Pre-compute PFOR64 section size
    uint32_t pfor64_block_starts_bytes = (pfor64_rowids.nblocks + 1) * sizeof(uint32_t);
    uint32_t pfor64_block_starts_aligned = (pfor64_block_starts_bytes + 7) & ~7u;  // 8B align
    uint32_t pfor64_data_bytes = pfor64_rowids.nulong * sizeof(uint64_t);
    uint32_t pfor64_total = pfor64_block_starts_aligned + pfor64_data_bytes;

    // ── Phase 1: Assign records to comp blocks ──
    struct CbInfo { uint32_t first; uint32_t n; uint32_t data_bytes; };

    CbInfo cbs[4096];
    uint32_t ncbs = 0;

    // fixed_overhead includes: pag_head + n_comp_blocks + symtab + pfor64_offset
    uint32_t fixed_overhead_est = sizeof(pag_head) + sizeof(uint32_t)
        + FSST_SYMTAB_TOTAL + FSST_PFOR64_OFFSET_SIZE;
    uint32_t cumulative_data = 0;

    uint32_t cb_first = 0, cb_data = 0, cb_n = 0;

    for (uint32_t i = 0; i < nrecs; i++) {
        uint32_t overhead = (cb_n + 2) * sizeof(uint16_t);
        uint32_t total = overhead + cb_data + (uint32_t)comp_lens[i];
        if (cb_n > 0 && total > max_comp_block_payload) {
            assert(ncbs < 4096);
            cbs[ncbs++] = {cb_first, cb_n, cb_data};
            cumulative_data += sizeof(FsstCompBlockDirEntry) + (cb_n + 1) * sizeof(uint16_t) + cb_data;
            if (fixed_overhead_est + cumulative_data + pfor64_total >= page_size) break;
            cb_first = i;
            cb_data = 0;
            cb_n = 0;
        }
        cb_data += (uint32_t)comp_lens[i];
        cb_n++;
    }
    if (cb_n > 0 && ncbs < 4096) {
        cbs[ncbs++] = {cb_first, cb_n, cb_data};
    }

    // ── Phase 2: Determine how many comp blocks fit (including PFOR64 at end) ──
    // Layout: pag_head(12) + n_comp_blocks(4) + dir(8*N) + symtab(2296) + pfor64_offset(4) + comp blocks + pfor64_section
    uint32_t fixed_overhead = sizeof(pag_head) + sizeof(uint32_t)
        + FSST_SYMTAB_TOTAL + FSST_PFOR64_OFFSET_SIZE;
    uint32_t used = fixed_overhead;
    uint32_t cbs_in_page = 0;
    uint32_t recs_in_page = 0;

    for (uint32_t ci = 0; ci < ncbs; ci++) {
        uint32_t cb_bytes = (cbs[ci].n + 1) * sizeof(uint16_t) + cbs[ci].data_bytes;
        uint32_t tentative = used + sizeof(FsstCompBlockDirEntry) + cb_bytes;
        tentative = (tentative + 3) & ~3u;
        // Must also fit the PFOR64 section at the end
        if (tentative + pfor64_total > page_size) break;
        used = tentative;
        cbs_in_page++;
        recs_in_page += cbs[ci].n;
    }

    if (cbs_in_page == 0) return 0;

    // ── Phase 3: Write page ──
    uint32_t wpos = sizeof(pag_head);

    // n_comp_blocks
    *(uint32_t*)(base + wpos) = cbs_in_page;
    wpos += sizeof(uint32_t);

    // Directory entries (placeholder, fill offsets later)
    uint32_t dir_offset = wpos;
    wpos += cbs_in_page * sizeof(FsstCompBlockDirEntry);

    // Symbol table
    memcpy(base + wpos, raw_symtab, FSST_SYMTAB_TOTAL);
    wpos += FSST_SYMTAB_TOTAL;

    // pfor64_rowid_offset placeholder (fill after comp blocks)
    uint32_t pfor64_offset_pos = wpos;
    wpos += FSST_PFOR64_OFFSET_SIZE;

    // Comp blocks
    for (uint32_t ci = 0; ci < cbs_in_page; ci++) {
        auto& cb = cbs[ci];

        // Directory entry
        FsstCompBlockDirEntry* dir = (FsstCompBlockDirEntry*)(base + dir_offset) + ci;
        dir->offset = wpos;
        dir->nrecs = cb.n;

        // Offset table
        uint16_t* otbl = (uint16_t*)(base + wpos);
        uint16_t acc = 0;
        for (uint32_t r = 0; r < cb.n; r++) {
            otbl[r] = acc;
            acc += (uint16_t)comp_lens[cb.first + r];
        }
        otbl[cb.n] = acc;

        uint32_t otbl_bytes = (cb.n + 1) * sizeof(uint16_t);
        uint32_t data_start = wpos + otbl_bytes;

        // Compressed records
        uint32_t dpos = 0;
        for (uint32_t r = 0; r < cb.n; r++) {
            memcpy(base + data_start + dpos,
                   comp_ptrs[cb.first + r], comp_lens[cb.first + r]);
            dpos += (uint32_t)comp_lens[cb.first + r];
        }

        wpos = data_start + dpos;
        wpos = (wpos + 3) & ~3u;
        assert(wpos <= page_size);
    }

    // ── Phase 4: Write PFOR64 rowid section ──
    uint32_t pfor64_start = wpos;

    // Write pfor64_rowid_offset back to its placeholder
    *(uint32_t*)(base + pfor64_offset_pos) = pfor64_start;

    // block_start array
    memcpy(base + wpos, pfor64_rowids.block_starts, pfor64_block_starts_bytes);
    wpos += pfor64_block_starts_bytes;
    // Pad to 8B alignment
    while (wpos & 7u) {
        *(base + wpos) = 0;
        wpos++;
    }

    // PFOR64 encoded data
    memcpy(base + wpos, pfor64_rowids.encoded_data, pfor64_data_bytes);
    wpos += pfor64_data_bytes;

    assert(wpos <= page_size);

    // Update pag_head
    php->nalloc = recs_in_page;
    php->watermark = wpos;
    php->lfreespace = page_size - wpos;

    compressed_page_bytes = wpos;

    return recs_in_page;
}

bool fsst_rowid_locate_pfor64_section(
    const uint8_t* page_data,
    size_t         page_size,
    const uint64_t** out_encoded_data,
    const uint32_t** out_block_starts,
    uint32_t*        out_nrecs_padded)
{
    const pag_head* php = (const pag_head*)page_data;
    uint32_t nalloc = php->nalloc;
    if (nalloc == 0) return false;

    uint32_t n_comp_blocks = *(const uint32_t*)(page_data + FSST_PAGE_NCB_OFFSET);
    uint32_t pfor64_offset_pos = sizeof(pag_head) + sizeof(uint32_t)
        + n_comp_blocks * sizeof(FsstCompBlockDirEntry)
        + FSST_SYMTAB_TOTAL;

    uint32_t pfor64_start = *(const uint32_t*)(page_data + pfor64_offset_pos);
    if (pfor64_start == 0 || pfor64_start >= page_size) return false;

    uint32_t nrecs_padded = (nalloc + 127) & ~127u;
    uint32_t nblocks = nrecs_padded / 128;

    uint32_t block_starts_bytes = (nblocks + 1) * sizeof(uint32_t);
    // Encoded data starts at the next 8B-aligned position after block_starts
    // (alignment is in absolute page coordinates, matching the write path)
    uint32_t encoded_data_start = (pfor64_start + block_starts_bytes + 7) & ~7u;

    *out_block_starts = (const uint32_t*)(page_data + pfor64_start);
    *out_encoded_data = (const uint64_t*)(page_data + encoded_data_start);
    *out_nrecs_padded = nrecs_padded;

    return true;
}

uint32_t fsst_decompress_page_cpu(
    const uint8_t* page_data,
    size_t         page_size,
    uint8_t*       output,
    size_t*        output_lens,
    size_t         max_output_size)
{
    const pag_head* php = (const pag_head*)page_data;
    uint32_t nalloc = php->nalloc;
    if (nalloc == 0) return 0;

    uint32_t n_comp_blocks = *(const uint32_t*)(page_data + FSST_PAGE_NCB_OFFSET);
    uint32_t dir_offset = sizeof(pag_head) + sizeof(uint32_t);
    uint32_t symtab_offset = dir_offset + n_comp_blocks * sizeof(FsstCompBlockDirEntry);

    // Read symbol table
    const uint8_t* raw_symtab = page_data + symtab_offset;
    fsst_decoder_t decoder;
    memset(&decoder, 0, sizeof(decoder));
    memcpy(decoder.len, raw_symtab, 255);
    memcpy(decoder.symbol, raw_symtab + FSST_SYMTAB_LEN_BYTES, 255 * sizeof(uint64_t));

    uint32_t total_recs = 0;
    size_t out_pos = 0;

    for (uint32_t ci = 0; ci < n_comp_blocks; ci++) {
        const FsstCompBlockDirEntry* dir =
            (const FsstCompBlockDirEntry*)(page_data + dir_offset) + ci;

        uint32_t cb_offset = dir->offset;
        uint32_t cb_nrecs = dir->nrecs;

        const uint16_t* otbl = (const uint16_t*)(page_data + cb_offset);
        const uint8_t* comp_data = page_data + cb_offset + (cb_nrecs + 1) * sizeof(uint16_t);

        for (uint32_t r = 0; r < cb_nrecs; r++) {
            uint16_t comp_start = otbl[r];
            uint16_t comp_len = otbl[r + 1] - comp_start;
            const uint8_t* comp_ptr = comp_data + comp_start;

            // Decompress one record using FSST symbol table
            size_t decomp_len = 0;
            uint32_t posIn = 0;
            while (posIn < comp_len) {
                uint8_t code = comp_ptr[posIn++];
                if (code < 255) {
                    uint8_t sym_len = decoder.len[code];
                    if (out_pos + decomp_len + sym_len <= max_output_size) {
                        memcpy(output + out_pos + decomp_len,
                               &decoder.symbol[code], sym_len);
                    }
                    decomp_len += sym_len;
                } else {
                    // Escape byte: next byte is literal
                    if (posIn < comp_len && out_pos + decomp_len + 1 <= max_output_size) {
                        output[out_pos + decomp_len] = comp_ptr[posIn];
                    }
                    posIn++;
                    decomp_len++;
                }
            }

            if (output_lens) {
                output_lens[total_recs] = decomp_len;
            }
            out_pos += decomp_len;
            total_recs++;
        }
    }

    return total_recs;
}
