#pragma once

#include <libaio.h>

#include "common/common.cu"
#include "common/page.cu"
#include "common/primitive_c.cu"
#include "schema/ssb_tables.cuh"

// #define COLUMN_FIRST

// buffer
struct HostSyncCompressQ11Args
{
    size_t thrid;
    std::vector<int> fds;
    io_context_t ctx;
    std::array<std::vector<uint64_t>::iterator, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> pagid_vec_begin_arr;
    std::array<std::vector<uint64_t>::iterator, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> pagid_vec_end_arr;
    std::array<char *, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> buf_arr;
    int32_t *selection_flags;
    int64_t q11_sum;
    uint64_t period_sec;
    size_t stats_nio;
};


BenchmarkResult hostSyncCompressSSBQ11(BenchmarkOptions &options)
{
    std::vector<int> fds;
    open_files(options, fds);

    SSBTableMetadata *metadata_ptr = nullptr;
    auto page_size = options.page_size;
    metadata_ptr = static_cast<SSBTableMetadata*>(mb_alloc(options.page_size));
    std::cout << "metadata_ptr: " << metadata_ptr << std::endl; 
    page_pread_host(fds, metadata_ptr, 0, page_size);

    auto &metadata = *metadata_ptr;
    assert(metadata.page_size == options.page_size);
    SSB::metadata_print(metadata);

    uint32_t *offsets_comp = static_cast<uint32_t*>(mb_alloc(page_size));
    uint64_t noffsets_lo_orderdate = metadata_ptr->compress_table_lineorder_noffsets[SSB::common::LO_ORDERDATE];
    std::cout << "compress_table_lineorder_noffsets[LO_ORDERDATE]: " 
        << metadata_ptr->compress_table_lineorder_noffsets[SSB::common::LO_ORDERDATE] << std::endl;

    size_t npages_for_offsets_lo_orderdate = metadata_calc_npages_for_compression_offsets(noffsets_lo_orderdate, page_size);
    uint64_t compsize_pagid_lo_orderdate  = metadata_ptr->compress_table_lineorder_compressed_page_size_start_page_ids[SSB::common::LO_ORDERDATE];
    uint64_t offsets_pagid_lo_orderdate = metadata_ptr->compress_table_lineorder_offset_start_page_ids[SSB::common::LO_ORDERDATE];
    std::cout << "npages_for_offsets_lo_orderdate: " << npages_for_offsets_lo_orderdate << std::endl;
    std::cout << "noffsets_lo_orderdate: " << noffsets_lo_orderdate << std::endl;
    std::cout << "offsets_pagid_lo_orderdate: " << offsets_pagid_lo_orderdate << std::endl;
    std::cout << "compsize_pagid_lo_orderdate: " << compsize_pagid_lo_orderdate << std::endl;
    // this must be
    // if N_LINEORDER % (size_miniblock=128) == 0 -> N_LINEORDER / (size_miniblock=128);
    // else N_LINEORDER / (size_miniblock=128) + 1;
    // sizeof(uint32_t)

    // metadata.compress_table_lineorder_compressed_page_size_start_page_ids
    // metadata.compress_table_lineorder_offset_start_page_ids[
    page_pread_host(fds, offsets_comp, offsets_pagid_lo_orderdate, page_size);
    for (size_t i = 0; i < noffsets_lo_orderdate; i++)
    {
        std::cout << "offsets_comp[" << i << "]: " << offsets_comp[i] << std::endl;
    }
    exit(1);

    const size_t nrows_page_int32 = options.page_size / sizeof(int32_t);
    const size_t nrows_page_int64 = options.page_size / sizeof(int64_t);

    size_t start_pagid;
    size_t nios_issued = 0;
    const size_t nrows_lo = metadata.table_lineorder_nrows;
    start_pagid = metadata.table_lineorder_start_page_ids[0];
    size_t npages_lo = (nrows_lo * sizeof(int32_t) - 1) / options.page_size + 1;
    size_t nthreads = std::min(npages_lo, options.nthreads);
    if (nthreads < options.nthreads)
    {
        std::cout << "npages_lo is smaller than nthreads, so changed nthread to " 
            << nthreads << std::endl;
    }

    std::cout
        << "nrows: " << nrows_lo << std::endl
        << "nrows_page_int32: " << nrows_page_int32 << std::endl
        << "nrows_page_int64: " << nrows_page_int64 << std::endl
        << "npages_lo: " << npages_lo << std::endl;

    constexpr auto &active_field_index = SSB::query::q1x::LO_FIELDS;

    std::array<std::vector<uint64_t>, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> pagid_vec_arr;
    for (size_t f = 0; f < SSB::query::q11::NUM_LO_ACTIVE_FIELDS; f++)
    {
        pagid_vec_arr[f].reserve(npages_lo);
        start_pagid = metadata.table_lineorder_start_page_ids[active_field_index[f]];
        pagid_vec_arr[f] = generate_pagid_table(start_pagid, start_pagid + npages_lo);
    }

    /* nthreads = std::min(npages_lo, options.nthreads) confirms that npages_lo / nthreads >= 1 */
    size_t jobs_per_thread_base = npages_lo / nthreads;
    size_t jobs_remainder = npages_lo % nthreads;
    std::vector<HostSyncCompressQ11Args> thread_args_vec;
    thread_args_vec.reserve(nthreads);

    size_t jobs_start = 0;
    for (size_t i = 0; i < nthreads; i++)
    {
        char *buf = static_cast<char*>(mb_alloc(SSB::query::q11::NUM_LO_ACTIVE_FIELDS * options.page_size));
        std::array<std::vector<uint64_t>::iterator, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> pagid_vec_begin_arr;
        std::array<std::vector<uint64_t>::iterator, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> pagid_vec_end_arr;
        std::array<char *, SSB::query::q11::NUM_LO_ACTIVE_FIELDS> buf_arr;
        int32_t *selection_flags = (int32_t *)malloc(sizeof(int32_t) * nrows_page_int32);
        size_t jobs_count = jobs_per_thread_base + (i < jobs_remainder ? 1 : 0);
        for (size_t f = 0; f < SSB::query::q11::NUM_LO_ACTIVE_FIELDS; f++)
        {
            auto &pagid_vec = pagid_vec_arr[f];
            auto begin = pagid_vec_arr[f].begin() + jobs_start;
            auto end = pagid_vec_arr[f].begin() + jobs_start + jobs_count;
            pagid_vec_begin_arr[f] = begin;
            pagid_vec_end_arr[f] = end;

            buf_arr[f] = buf + f * options.page_size;
            std::cout << "buf_arr[" << f << "]: " << (void *)buf_arr[f] << std::endl;   

        }
        jobs_start += jobs_count;
        thread_args_vec.push_back(HostSyncCompressQ11Args{
            .thrid = i,
            .fds = fds,
            .ctx = {},
            .pagid_vec_begin_arr = pagid_vec_begin_arr,
            .pagid_vec_end_arr = pagid_vec_end_arr,
            .buf_arr = buf_arr,
            .selection_flags = selection_flags,
            .q11_sum = 0,
            .period_sec = options.period_sec,
            .stats_nio = 0,
        });

        for (size_t f = 0; f < SSB::query::q11::NUM_LO_ACTIVE_FIELDS; f++)
        {
            for (size_t j = 0; j < nrows_page_int32; j++)
            {
                thread_args_vec[i].selection_flags[j] = 0;
            }
        }
    }

    size_t _verify_npages = 0;
    for (size_t i = 0; i < nthreads; i++)
    {
        auto &args = thread_args_vec[i];
        ssize_t distance = std::distance(
            args.pagid_vec_begin_arr[0],
            args.pagid_vec_end_arr[0]);
        _verify_npages += distance;
    }
    if (_verify_npages != npages_lo) {
        std::cerr << "[ERROR][BUG] Task allocation fails. npages_lo: " << npages_lo << " != " << _verify_npages << std::endl;
        exit(EXIT_FAILURE);
    }

    // size_t lo_count = 0;
    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    for (size_t i = 0; i < nthreads; i++)
    {
        HostSyncCompressQ11Args &args = thread_args_vec[i];
        threads.emplace_back(
            [&options, &args, nthreads, nrows_page_int32, nrows_page_int64, nrows_lo]()
            {
                cpu_set_affinity(args.thrid);
                args.pagid_vec_begin_arr[0];
                args.pagid_vec_end_arr[0];
                size_t npages = std::distance(
                    args.pagid_vec_begin_arr[0],
                    args.pagid_vec_end_arr[0]);

                assert(args.buf_arr[0] != nullptr);
                assert(args.buf_arr[1] != nullptr);
                assert(args.buf_arr[2] != nullptr);
                assert(args.buf_arr[3] != nullptr);

#ifdef COLUMN_FIRST
                int32_t *selection_flags = args.selection_flags;
#endif

                size_t i = 0;
                uint64_t time_start = gettime();
#if 0
                size_t nscan = 0;
                size_t nscan1 = 0; size_t nscan2 = 0;
                size_t nscan3 = 0; size_t nscan4 = 0;
                size_t nscan5 = 0; size_t nscan6 = 0;
#endif
                while (i < npages)
                {
                    size_t page_id_item1 = args.pagid_vec_begin_arr[0][i];
                    size_t page_id_item2 = args.pagid_vec_begin_arr[1][i];
                    size_t page_id_item3 = args.pagid_vec_begin_arr[2][i];
                    size_t page_id_item4 = args.pagid_vec_begin_arr[3][i];

                    std::cout << "page_id_item1: " << page_id_item1 << std::endl;
                    // std::cout << "page_id_item2: " << page_id_item2 << std::endl;
                    // std::cout << "page_id_item3: " << page_id_item3 << std::endl;
                    // std::cout << "page_id_item4: " << page_id_item4 << std::endl;

                    page_pread_host(args.fds, args.buf_arr[0], page_id_item1, options.page_size);
                    page_pread_host(args.fds, args.buf_arr[1], page_id_item2, options.page_size);
                    page_pread_host(args.fds, args.buf_arr[2], page_id_item3, options.page_size);
                    page_pread_host(args.fds, args.buf_arr[3], page_id_item4, options.page_size);
                    args.stats_nio += SSB::query::q11::NUM_LO_ACTIVE_FIELDS;

                    bool is_last_page = false;
                    size_t nrows_final_page = nrows_lo % nrows_page_int32;
                    if (i == npages - 1 && args.thrid == nthreads - 1)
                    {
                        is_last_page = true;
                        std::cout << "last_page: " << npages - 1 << std::endl;
                        std::cout << "nrows_final_page: " << nrows_final_page << std::endl;
                    }

                    #if 0
                    std::cout << "args.buf_arr[0]: " << (void *)args.buf_arr[0] << std::endl;
                    std::cout << "args.buf_arr[1]: " << (void *)args.buf_arr[1] << std::endl;
                    std::cout << "args.buf_arr[2]: " << (void *)args.buf_arr[2] << std::endl;
                    std::cout << "args.buf_arr[3]: " << (void *)args.buf_arr[3] << std::endl;
                    #endif

                    int32_t *items1 = reinterpret_cast<int32_t*>(args.buf_arr[0]);
                    int32_t *items2 = reinterpret_cast<int32_t*>(args.buf_arr[1]);
                    int32_t *items3 = reinterpret_cast<int32_t*>(args.buf_arr[2]);
                    int32_t *items4 = reinterpret_cast<int32_t*>(args.buf_arr[3]);

                    size_t n = is_last_page ? nrows_final_page : nrows_page_int32;
#if 0
                    std::cout << "n: " << n << std::endl;
#endif

#ifdef COLUMN_FIRST
                    for (size_t j = 0; j < n; j++)
                    {
                        // std::cout << "items1[" << j << "]: " << items1[j] << std::endl;
                        // std::cout << "items2[" << j << "]: " << items2[j] << std::endl;
                        // std::cout << "items3[" << j << "]: " << items3[j] << std::endl;
                        if (items1[j] > 19930000 && items1[j] < 19940000) {
                            selection_flags[j] = 1;
                        }
                    }


                    for (size_t j = 0; j < n; j++)
                    {
                        if (selection_flags[j] && items2[j] < 25) {
                            selection_flags[j] = 1;
                        }
                    }

                    for (size_t j = 0; j < n; j++) {
                        if (items3[j] >= 1 && items3[j] <= 3) {
                           args.q11_sum += items4[j] * items3[j];
                        }
                    }
#else
                    for (size_t j = 0; j < n; j++)
                    {
                        // std::cout << "items1[" << j << "]: " << items1[j] << std::endl;
                        // std::cout << "items2[" << j << "]: " << items2[j] << std::endl;
                        // std::cout << "items3[" << j << "]: " << items3[j] << std::endl;
                        if (items1[j] > 19930000 && items1[j] < 19940000 
                          && items2[j] < 25
                          && items3[j] >= 1 && items3[j] <= 3) {
                           args.q11_sum += items4[j] * items3[j];
                        }
                    }
#endif

                    i++;
                }
                uint64_t time_end = gettime();
#if 0
                std::cout << "(total) " << nscan << std::endl;
                std::cout << "(lineorder > 19920000) " << nscan1 << std::endl;
                std::cout << "(lineorder < 19940000) " << nscan2 << std::endl;
                std::cout << "(lineorder > 19910000) " << nscan3 << std::endl;
                std::cout << "(lineorder < 19930000) " << nscan4 << std::endl;
                std::cout << "(cond1 && cond2) " << nscan3 << std::endl;
#endif
                // " " << nscan3 << " " << nscan4 << " " << nscan5 << std::endl;
            }
 
        );
    }
    
    int64_t q11_sum = 0;
    for (size_t i = 0; i < nthreads; i++)
    {
        threads[i].join();
        q11_sum += thread_args_vec[i].q11_sum;
        nios_issued = thread_args_vec[i].stats_nio;
    }

    auto end = chrono::system_clock::now();
    auto end_cpu_usage = read_cpu_usage();

    std::cout << "q11_sum: " << q11_sum << std::endl;

    close_files(options, fds);

    return BenchmarkResult{
        .nios = nios_issued,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = GpuUsage{},
    };
}
