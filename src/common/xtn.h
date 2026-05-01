#pragma once

#include <iostream>
#include <cstddef>
#include <cstdint>
#include <cassert>

#include "tpch_tables.cuh"

// struct Metadata is defined in tpch_tables.cuh

constexpr size_t XTNID_INVALID = UINT64_MAX;

namespace SuperPage {
    constexpr size_t GIBI = 1024 * 1024 * 1024UL;

    size_t NumPagesForSuperPages;
    // size_t npages_used_for_compressed_page_sizes;
    // size_t nfilters;

    constexpr uint64_t InvalidXtnId = XTNID_INVALID;
    constexpr uint64_t InvalidPageId = UINT64_MAX;

    bool Inited = false;
};

uint64_t superpage_get_base_page_id(void) {
    return SuperPage::NumPagesForSuperPages;
}

uint64_t superpage_get_base_super_page_id(void) {
    return 0;
}

size_t superpage_calc_npages_for_prefix_sum(size_t npages, size_t page_size)
{
    /* 8B */
    return ((npages + 1) * sizeof(uint64_t) + page_size - 1) / page_size;
}

size_t superpage_calc_npages_for_compressed_page_sizes(size_t npages, size_t page_size)
{
    /* 4B per page */
    return (npages * sizeof(uint32_t) + page_size - 1) / page_size;
}

size_t superpage_calc_npages_for_stats(size_t npages, size_t valsize, size_t page_size)
{
    /* 2 * <valsize>B per page */
    return (npages * 2 * valsize + page_size - 1) / page_size;
}

size_t superpage_get_super_npage(void) {
    return SuperPage::NumPagesForSuperPages;
}

/* Generic version: caller provides metadata struct size directly */
inline void superpage_set_constants_for(size_t page_size, size_t metadata_struct_size) {
    size_t num_super_pages = (metadata_struct_size + page_size - 1) / page_size;
    SuperPage::NumPagesForSuperPages = num_super_pages;
    SuperPage::Inited = true;
    std::cout << "Size of metadata [B]             : " << metadata_struct_size << std::endl;
    std::cout << "Num super pages                  : " << num_super_pages << std::endl;
}

void superpage_set_constants(size_t page_size, size_t MAX_DATA_GB=10'000UL) {
    // page_size = 16384;
    /* Assuming enough space */
    //size_t MaxDataSizeGB = MAX_DATA_GB * SuperPage::GIBI;
    size_t size_super_pages_bytes = sizeof(struct TPCHTableMetadata);
    size_t num_super_pages = (size_super_pages_bytes + page_size - 1) / page_size;
    SuperPage::NumPagesForSuperPages = num_super_pages;

    SuperPage::Inited = true;

    std::cout << "Assumed MAX_DATA_GB [GB]         : " << MAX_DATA_GB << std::endl;
    std::cout << "Size of metadata [B]             : " << size_super_pages_bytes << std::endl;
    std::cout << "Num super pages                  : " << num_super_pages << std::endl;

    // XTN::NumPagesForSuperXTNs =
    //     (max_num_xtn_entries + nxtn_entries_per_page - 1) / nxtn_entries_per_page;
    // XTN::NumXTNsForSuperXTNs =
    //     (max_num_xtn_entries + nxtn_entries_per_xtn - 1) / nxtn_entries_per_xtn;
 
#if 0
    std::cout << "Assumed MAX_DATA_GB [GB]         : " << MAX_DATA_GB << std::endl;
    std::cout << "XTN sizes for super XTN [MB]     : "
        << size_total_xtn_entries / XTN::MIBI << std::endl;
    std::cout << "XTN sizes for meta dict XTN [MB] : "
        << size_total_dct_meta / XTN::MIBI << std::endl;
    std::cout << "Number of XTN entries per page   : " << nxtn_entries_per_page << std::endl;
    std::cout << "Number of XTN entries per XTN    : " << nxtn_entries_per_xtn << std::endl;
    std::cout << "Number of super XTN              : " << XTN::MaxNumXTNs << std::endl;
    std::cout << "Number of pages for super XTN    : " << XTN::NumPagesForSuperXTNs << std::endl;
    std::cout << "Number of XTNs for super XTN     : " << XTN::NumXTNsForSuperXTNs << std::endl;
    std::cout << "Number of pages for dct meta XTN : " << XTN::NumPagesForDctMetaXTNs << std::endl;
    std::cout << "Number of XTNs for dct meta XTN  : " << XTN::NumXTNsForDctMetaXTNs << std::endl;
    std::cout << "Root XTN id                      : " << xtn_get_root_xtnid() << std::endl;
    std::cout << "Super XTN id                     : " << xtn_get_super_xtnid() << std::endl;
    std::cout << "Dctmeta XTN id                   : " << xtn_get_dct_meta_xtnid() << std::endl;
    std::cout << "Base XTN id for normal XTNs      : " << xtn_get_base_xtnid() << std::endl;
#endif
}


/* NOTE: current version should not depend on XTN */
#if 0

struct xtn_entry {
  uint8_t apag; /* number of allocated pages */
  // uint32_t id;   /* extent id */
  // uint16_t table;  /* file id --> enum TPCH::common::Table */
  // uint32_t nxtn; /* next extent id */
};

struct page_filter_entry {
  uint64_t value_min; // 8B min value
  uint64_t value_max; // 8B max value
};

/* 16B per 64MB if kXTN = 64 and page_size = 1MB */
/* SF=1      -> 1B * 1GB / 64MB = 1B * 16 = 16B */
/* SF=10     -> 1B * 10GB / 64MB = 1B * 160 = 160B = 0.16KB */
/* SF=100    -> 1B * 100GB / 64MB = 1B * 1600 = 1600B = 1.6KB */
/* SF=1000   -> 1B * 1000GB / 64MB = 1B * 16000 = 16000B = 16KB */
/* SF=10000  -> 1B * 10000GB / 64MB = 1B * 160000 = 160000B = 160KB = < 64MB */
/* SF=100000 -> 1B * 100000GB / 64MB = 1B * 1600000 = 1600000B = 1600KB = 1.6MB < 64MB */

/* 16B per 4MB if kXTN = 64 and page_size = 64KB */
/* SF=1     -> 1B * 1GB / 4MB = 1B * 250 = 0.25KB */
/* SF=10    -> 1B * 10GB / 4MB = 1B * 2500 = 2.5KB */
/* SF=100   -> 1B * 100GB / 4MB = 1B * 25000 = 25KB */
/* SF=1000  -> 1B * 1000GB / 4MB = 1B * 250000 = 250KB */
/* SF=10000 -> 1B * 10000GB / 4MB = 1B * 2500000 = 2500KB = 2.5MB < 4MB */

namespace XTN {
    /* 64 pages per XTN */
    constexpr size_t kNumPagesPerXTN = 64;

    /* 1 XTN are allocated to save Metadata */
    constexpr size_t kNumXTNForRootXTN = 1;
    /* 1 XTN are allocated to save struct xtn_entry */
    // constexpr size_t kNumXTNForSuperXTN = 1;

    /* Normal data XTN is started from XTN 9- */
    // constexpr size_t kXTNOFffsetForData = kNumXTNForRootXTN + kNumXTNForSuperXTN;

    constexpr size_t MIBI = 1024 * 1024UL;
    constexpr size_t GIBI = MIBI * 1024UL;

    // constexpr size_t SF_MAX = 100'000UL; // Max scalefactor
    // constexpr size_t kMaxDataSizeGB = SF_MAX * XTN::GIBI;

    size_t MaxNumXTNs = 0;
    size_t NumPagesForSuperXTNs;
    size_t NumXTNsForSuperXTNs;
    size_t NumPagesForDctMetaXTNs;
    size_t NumXTNsForDctMetaXTNs;

    constexpr uint64_t InvalidXtnId = XTNID_INVALID;
    constexpr uint64_t InvalidPageId = UINT64_MAX;
    bool Inited = false;
};

namespace DCT {
    constexpr uint32_t kNumBitsForClusterId = 8;
    constexpr uint32_t kNumMaxDictsInXtn = (1 << kNumBitsForClusterId);
    constexpr uint32_t kMaskForClusterId = (kNumMaxDictsInXtn - 1);
    constexpr uint32_t kMaskForVarCharId = (1 << (32 - kNumBitsForClusterId)) - 1;

    #if 0
    uint32_t encode_dict_id(const uint32_t cluster_id, const uint32_t varchar_id) {
        if (varchar_id > kMaskForVarCharId) {
            std::cerr << "Error: varchar_id exceeds the limit: " << varchar_id << " > " << kMaskForVarCharId << std::endl;
            exit(1);
        }
        //assert(cluster_id <= kMaskForClusterId);
        return (varchar_id << kNumBitsForClusterId) | (cluster_id & kMaskForClusterId);
    }

    std::pair<uint32_t, uint32_t> decode_dict_id(const uint32_t encoded_id) {
        //uint32_t varchar_id = (encoded_id >> kNumBitsForClusterId) & kMaskForVarCharId;
        uint32_t varchar_id = (encoded_id >> kNumBitsForClusterId);
        uint32_t cluster_id = encoded_id & kMaskForClusterId;
        return {cluster_id, varchar_id};
    }
    #endif

    constexpr uint32_t kNumMaxVarCharFields = 3;
}

struct dct_meta_entry {
  uint64_t page_ids[DCT::kNumMaxVarCharFields][DCT::kNumMaxDictsInXtn]; // 8B * 8 * 3 = 64B * 3
  uint32_t npages[DCT::kNumMaxVarCharFields][DCT::kNumMaxDictsInXtn]; // 4B * 8 * 3 = 32B * 3
}; // 96B * 3 per dictionary entry == per XTN entry

constexpr size_t xtn_get_root_xtnid(void);
constexpr size_t xtn_get_super_xtnid(void);
size_t xtn_get_dct_meta_xtnid(void);
size_t xtn_get_base_xtnid(void);

void xtn_set_constants(size_t page_size, size_t MAX_DATA_GB=10'000UL) {
    // page_size = 16384;
    /* Assuming enough space */
    size_t MaxDataSizeGB = MAX_DATA_GB * XTN::GIBI;
    size_t size_xtn = XTN::kNumPagesPerXTN * page_size;
    size_t max_num_xtn_entries = std::max(MaxDataSizeGB / size_xtn, 256UL);
    size_t size_xtn_entry_per_xtn = sizeof(struct xtn_entry);
    size_t size_total_xtn_entries = max_num_xtn_entries * size_xtn_entry_per_xtn;
    size_t nxtn_entries_per_page = page_size / size_xtn_entry_per_xtn;
    size_t nxtn_entries_per_xtn = nxtn_entries_per_page * XTN::kNumPagesPerXTN;

    /* Number of pages for super XTN */
    XTN::MaxNumXTNs = max_num_xtn_entries;
    XTN::NumPagesForSuperXTNs =
        (max_num_xtn_entries + nxtn_entries_per_page - 1) / nxtn_entries_per_page;
    XTN::NumXTNsForSuperXTNs =
        (max_num_xtn_entries + nxtn_entries_per_xtn - 1) / nxtn_entries_per_xtn;


    /* Number of pages for storing dict meta XTN */
    size_t size_dct_meta_per_xtn = sizeof(struct dct_meta_entry);
    size_t size_total_dct_meta = max_num_xtn_entries * size_dct_meta_per_xtn;
    size_t nxtn_dct_meta_entries_per_page = page_size / size_dct_meta_per_xtn;
    size_t nxtn_dct_meta_entries_per_xtn = nxtn_dct_meta_entries_per_page * XTN::kNumPagesPerXTN;

    XTN::NumPagesForDctMetaXTNs =
        (max_num_xtn_entries + nxtn_dct_meta_entries_per_page - 1) / nxtn_dct_meta_entries_per_page;
    XTN::NumXTNsForDctMetaXTNs =
        (max_num_xtn_entries + nxtn_dct_meta_entries_per_xtn - 1) / nxtn_dct_meta_entries_per_xtn;
    //XTN::NumXTNsForDctMetaXTNs =

    XTN::Inited = true;

    // XTN::NumPagesForSuperXTNs =
    //     (max_num_xtn_entries + nxtn_entries_per_page - 1) / nxtn_entries_per_page;
    // XTN::NumXTNsForSuperXTNs =
    //     (max_num_xtn_entries + nxtn_entries_per_xtn - 1) / nxtn_entries_per_xtn;
 
    std::cout << "Assumed MAX_DATA_GB [GB]         : " << MAX_DATA_GB << std::endl;
    std::cout << "sizeof(struct xtn_entry) [B]     : " << sizeof(struct xtn_entry) << std::endl;
    std::cout << "XTN sizes for super XTN [MB]     : "
        << size_total_xtn_entries / XTN::MIBI << std::endl;
    std::cout << "XTN sizes for meta dict XTN [MB] : "
        << size_total_dct_meta / XTN::MIBI << std::endl;
    std::cout << "Number of XTN entries per page   : " << nxtn_entries_per_page << std::endl;
    std::cout << "Number of XTN entries per XTN    : " << nxtn_entries_per_xtn << std::endl;
    std::cout << "Number of super XTN              : " << XTN::MaxNumXTNs << std::endl;
    std::cout << "Number of pages for super XTN    : " << XTN::NumPagesForSuperXTNs << std::endl;
    std::cout << "Number of XTNs for super XTN     : " << XTN::NumXTNsForSuperXTNs << std::endl;
    std::cout << "Number of pages for dct meta XTN : " << XTN::NumPagesForDctMetaXTNs << std::endl;
    std::cout << "Number of XTNs for dct meta XTN  : " << XTN::NumXTNsForDctMetaXTNs << std::endl;
    std::cout << "Root XTN id                      : " << xtn_get_root_xtnid() << std::endl;
    std::cout << "Super XTN id                     : " << xtn_get_super_xtnid() << std::endl;
    std::cout << "Dctmeta XTN id                   : " << xtn_get_dct_meta_xtnid() << std::endl;
    std::cout << "Base XTN id for normal XTNs      : " << xtn_get_base_xtnid() << std::endl;
}

size_t xtn_get_num_xtn_entry_per_page(size_t page_size) {
    return page_size / sizeof(struct xtn_entry);
}

constexpr size_t xtn_get_npages(void) {
    return XTN::kNumPagesPerXTN;;
}

/* root XTN = 0: saving table metadata */
constexpr size_t xtn_get_root_xtnid(void) {
    return 0; // Super extent id is always 0
}

/* super XTN : 1 Metadata XTN */
/*  page 0: Metadata page  */
constexpr size_t xtn_get_super_xtnid(void) {
    return XTN::kNumXTNForRootXTN;
}

/* super XTN = 1 + kNXTN: saving extent metadata */
size_t xtn_get_super_nxtn(void) {
    return XTN::NumXTNsForSuperXTNs;
}

size_t xtn_get_dct_meta_nxtn(void) {
    return XTN::NumXTNsForDctMetaXTNs;
}

// constexpr size_t xtn_get_super_xtn_page_id(void) {
//     return 1;
// }
size_t xtn_get_dct_meta_xtnid(void) {
    return XTN::kNumXTNForRootXTN + XTN::NumXTNsForSuperXTNs;
}

size_t xtn_get_base_xtnid(void) {
    return XTN::kNumXTNForRootXTN + XTN::NumXTNsForSuperXTNs + XTN::NumXTNsForDctMetaXTNs;
}

constexpr size_t xtn_get_size(void) {
    return sizeof(struct xtn_entry);
}

size_t xtn_alloc_next_id(size_t &next_xtn_id)
{
    return next_xtn_id++;
}

// size_t xtn_calc_super_page_id_from_xtnid(const size_t page_size, const size_t xtn_id)
// {
//     return xtn_get_super_xtn_page_id() + xtn_id / xtn_get_num_xtn_entry_per_page(page_size);
// }

void xtn_init(struct xtn_entry &xtn)
{
    // xtn.id = xtn_id;
    xtn.apag = 0; // Initially no pages allocated
    // xtn.nxtn = 0;
}

uint64_t xtn_alloc_page(struct xtn_entry *xtn_head, uint64_t xtn_id)
{
    uint64_t new_page_id = xtn_id * XTN::kNumPagesPerXTN + xtn_head[xtn_id].apag;
    if (xtn_head[xtn_id].apag >= XTN::kNumPagesPerXTN) {
        return XTN::InvalidPageId; // Invalid page id
    }
    // std::cout << "[XTNID]" << xtn_id << ": " << static_cast<int32_t>(xtn_head[xtn_id].apag) << std::endl;
    xtn_head[xtn_id].apag++;
    return new_page_id;
}

void xtn_dealloc_page(struct xtn_entry *xtn_head, uint64_t xtn_id)
{
    assert(xtn_head[xtn_id].apag > 0);
    xtn_head[xtn_id].apag--;
    return;
}

// struct xtn_entry* xtn_get_xtn(struct xtn_entry *xtn_head, const size_t xtn_id)
// {
//     if (xtn_id >= XTN::NumXTNsForSuperXTNs) {
//         return nullptr; // Invalid XTN id
//     }
//     return xtn_head[xtn_id];
// }

uint64_t xtn_get_npages_in_xtn(void)
{
    return XTN::kNumPagesPerXTN;
}

uint64_t xtn_calc_xtn_size(const size_t page_size)
{
    return xtn_get_npages_in_xtn() * page_size;
}

uint64_t xtn_get_allocated_npages(const struct xtn_entry *xtn, const uint64_t xtn_id)
{
    return xtn[xtn_id].apag;
}

uint64_t xtn_calc_page_id_from_xtn_id(const uint64_t xtn_id)
{
    return xtn_id * XTN::kNumPagesPerXTN;
}

uint64_t xtn_calc_page_id(const struct xtn_entry *xtn, const size_t xtn_id, const size_t page_id)
{
    if (page_id >= xtn[xtn_id].apag) {
        return XTN::InvalidPageId; // Invalid page id
    }
    return xtn_id * XTN::kNumPagesPerXTN + page_id;
}
#endif