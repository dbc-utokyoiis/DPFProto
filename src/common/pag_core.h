#pragma once
#include <cstddef>
#include <cstdint>

struct pag_head {
    uint32_t nalloc;
    uint32_t watermark;
    uint32_t lfreespace;
};

typedef void PAG;

static inline void pag_init(PAG *pag, size_t page_size)
{
    pag_head *php = (pag_head *)pag;
    php->nalloc = 0;
    php->watermark = sizeof(pag_head);
    php->lfreespace = page_size - sizeof(pag_head);
}
