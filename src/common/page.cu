#pragma once

#include "common/common.cu"
#include "common/primitive_c.cu"

#include <fcntl.h>

#include <cstdio>


inline void page_pread_host(std::vector<int> &fds, void *buf, size_t pagid, size_t sizpag)
{
    uint8_t *bufp = (uint8_t *)buf;
    size_t ndev = fds.size();

    // std::cout << "page_pread_host: " << pagid << ", " << sizpag << std::endl;
    int idevid = pagid_to_idev(pagid, ndev);
    int ipagid = pagid_to_ipagid(pagid, ndev);
    mb_pread(fds[idevid], bufp, sizpag, ipagid * sizpag);
}

inline void page_pwrite_host(std::vector<int> &fds, void *buf, size_t pagid, size_t sizpag)
{
    uint8_t *bufp = (uint8_t *)buf;
    size_t ndev = fds.size();

    // std::cout << "page_pwrite_host: " << pagid << ", " << sizpag << std::endl;
    int idevid = pagid_to_idev(pagid, ndev);
    int ipagid = pagid_to_ipagid(pagid, ndev);
    mb_pwrite(fds[idevid], bufp, sizpag, ipagid * sizpag);
}

inline void page_pread_comp_host(std::vector<int> &fds, void *buf, size_t pagid, size_t sizio, size_t sizpag)
{
    uint8_t *bufp = (uint8_t *)buf;
    size_t ndev = fds.size();

    // std::cout << "page_pread_comp_host: " << pagid << ", " << sizpag << std::endl;
    int idevid = pagid_to_idev(pagid, ndev);
    int ipagid = pagid_to_ipagid(pagid, ndev);
    mb_pread(fds[idevid], bufp, sizio, ipagid * sizpag);
}

inline void page_pwrite_comp_host(std::vector<int> &fds, void *buf, size_t pagid, size_t sizio, size_t sizpag)
{
    uint8_t *bufp = (uint8_t *)buf;
    size_t ndev = fds.size();

    // std::cout << "page_pwrite_comp_host: " << pagid << ", " << sizpag << std::endl;
    int idevid = pagid_to_idev(pagid, ndev);
    int ipagid = pagid_to_ipagid(pagid, ndev);
    mb_pwrite(fds[idevid], bufp, sizio, ipagid * sizpag);
}

void write_compressed_page_host(const std::vector<int>& fds, void *buf, size_t pagid, size_t sizio, size_t sizpag,
    std::vector<uint64_t> &dev_write_pos, size_t base_pagid)
{
    size_t ndev = fds.size();
    // size_t d = pagid % ndev;
    int idevid = pagid_to_idev(pagid, ndev);
    //size_t local = pagid - base_pagid;

    if (sizio <= 0) {
        fprintf(stderr, "[%s](%d)STOP\n", __func__, __LINE__);
        exit(EXIT_FAILURE);
    }

    uint32_t stored_size = static_cast<uint32_t>(sizio);
    size_t aligned_stored = roundup4096(stored_size);

    mb_pwrite(fds[idevid], buf, aligned_stored, static_cast<off_t>(dev_write_pos[idevid]));

    /* compressed_sizes is saved in the caller side. */
    /* compressed_sizes[local] = stored_size; */
    dev_write_pos[idevid] += aligned_stored;
}

/* helper method for read_compressed_page_host */
uint64_t calc_compressed_page_offset(size_t page_id, uint64_t *offsets, size_t base_page_id)
{
    if (page_id < base_page_id) {
        fprintf(stderr, "[%s](%d)STOP\n", __func__, __LINE__);
        exit(EXIT_FAILURE);
    }
    return offsets[page_id - base_page_id];
}

void read_compressed_page_host(const std::vector<int>& fds, void * buf, size_t pagid, size_t sizio, size_t page_size,
    const uint64_t page_start_offset) {
    size_t ndev = fds.size();
    //size_t d = pagid % ndev;
    int idevid = pagid_to_idev(pagid, ndev);
    //size_t aligned_comp = roundup512(comp_size);

    mb_pread(fds[idevid], buf, sizio, static_cast<off_t>(page_start_offset));
}

void calculate_compressed_offsets(size_t *compressed_base_page_ids,
                                  uint32_t *compressed_page_sizes,
                                  size_t num_segments,
                                  size_t total_npages,
                                  size_t page_size,
                                  size_t start_page_id,
                                  size_t num_devices,
                                  std::vector<size_t> &offsets_out)
{
    auto roundup4096 = [](size_t v) -> size_t
    {
        return (v + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
    };

    offsets_out.resize(total_npages + 1);
    size_t npages_sum = 0;

    std::vector<size_t> dev_pos(num_devices);

    for (size_t s = 0; s < num_segments; s++)
    {
        // Number of pages in this segment
        size_t npages_seg = (s + 1 < num_segments)
            ? compressed_base_page_ids[s + 1] - compressed_base_page_ids[s]
            : total_npages - npages_sum;

        // Bounds check: npages_seg must not exceed remaining pages
        if (npages_sum + npages_seg > total_npages) {
            fprintf(stderr, "[%s] ERROR: segment %zu: npages_seg=%zu would exceed total_npages=%zu "
                "(npages_sum=%zu, base[s]=%zu, base[s+1]=%zu)\n",
                __func__, s, npages_seg, total_npages, npages_sum,
                compressed_base_page_ids[s],
                (s + 1 < num_segments) ? compressed_base_page_ids[s + 1] : 0);
            exit(EXIT_FAILURE);
        }

        // Initialize per-device write positions for this segment
        // (mirrors the loader's compressed_page_write_offsets initialization)
        for (size_t d = 0; d < num_devices; d++)
        {
            size_t p = compressed_base_page_ids[s] + d;
            size_t idevid = p % num_devices;
            size_t lpagid = p / num_devices;
            dev_pos[idevid] = lpagid * page_size;
        }

        // Per-device prefix sum within segment
        for (size_t k = npages_sum; k < npages_sum + npages_seg; k++)
        {
            size_t page_id = start_page_id + k;
            size_t d = page_id % num_devices;
            offsets_out[k] = dev_pos[d];
            dev_pos[d] += roundup4096(compressed_page_sizes[k]);
        }

        npages_sum += npages_seg;
    }
    offsets_out[total_npages] = 0; // sentinel
}