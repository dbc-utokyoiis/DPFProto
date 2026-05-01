#include "fsst_host.h"
#include <cassert>
#include <cstring>

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
    uint32_t                              page_byte_offset,
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
