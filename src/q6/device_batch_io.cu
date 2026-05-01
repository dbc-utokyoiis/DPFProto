#pragma once

#include "common/common.cu"
#include "common/primitive_c.cu"
#include "common/primitive_cuda.cu"
#include "common/primitive_cufile.cu"
// #include <nvcomp.h>

struct DeviceBatchIOCookie
{
    void *buf_dev;
    size_t idx;
};

struct DeviceBatchIOThreadArgs
{
    size_t thrid;
    std::vector<CUfileHandle_t> cufile_handles;
    CUfileBatchHandle_t batch_idp;
    std::vector<uint64_t>::iterator pagid_vec_begin;
    std::vector<uint64_t>::iterator pagid_vec_end;
    std::queue<void *> buf_dev_queue;
    std::vector<CUfileIOParams_t> batch_params_vec;
    std::vector<CUfileIOEvents_t> batch_events_vec;
    std::vector<DeviceBatchIOCookie> batch_cookie_vec;
    uint64_t period_sec;
    size_t stats_nio;
};

BenchmarkResult bench_device_batch_io(BenchmarkOptions &options)
{
    std::vector<int> fds;
    std::vector<CUfileHandle_t> cufile_handles;
    open_files(options, fds);

    mb_cufile_driver_open();

    cufile_handles.reserve(fds.size());
    for (auto fd : fds)
    {
        CUfileHandle_t cufile_handle = mb_cufile_handle_register(fd);
        cufile_handles.push_back(cufile_handle);
    }

    std::vector<uint64_t> pagid_vec = generate_pagid_vec(options);

    size_t nios = pagid_vec.size();
    std::cout << "file_size: " << options.file_size << std::endl;

    // if (nios % options.nthreads != 0)
    // {
    //     std::cerr << "nios must be a multiple of nthreads" << std::endl;
    //     exit(EXIT_FAILURE);
    // };

    size_t nios_per_thread = (nios + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<DeviceBatchIOThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    /* Check io_depth value does not exceed its max value */
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);
    if (options.io_depth > props.max_batch_io_size)
    {
        for (auto cufile_handle : cufile_handles)
        {
            mb_cufile_handle_deregister(cufile_handle);
        }
        mb_cufile_driver_close();
        close_files(options, fds);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < options.nthreads; i++)
    {
        CUfileBatchHandle_t batch_idp = nullptr;
        checkCuFileErrors(cuFileBatchIOSetUp(&batch_idp, std::min<size_t>(options.io_depth, props.max_batch_io_size)));

        std::queue<void *> buf_dev_queue;

        std::vector<CUfileIOParams_t> batch_params_vec(nios_per_thread);

        std::vector<CUfileIOEvents_t> batch_events_vec(nios_per_thread);

        std::vector<DeviceBatchIOCookie> batch_cookie_vec(nios_per_thread);

        for (size_t j = 0; j < options.io_depth; j++)
        {
            void *buf_dev = mb_cuda_alloc(options.io_size);

            mb_cufile_buf_register(buf_dev, options.io_size);

            buf_dev_queue.push(buf_dev);
        }

        auto begin = pagid_vec.begin() + nios_per_thread * i;
        auto end = pagid_vec.begin() + std::min(nios_per_thread * (i + 1), nios);

        thread_args_vec.push_back(DeviceBatchIOThreadArgs{
            .thrid = i,
            .cufile_handles = cufile_handles,
            .batch_idp = batch_idp,
            .pagid_vec_begin = begin,
            .pagid_vec_end = end,
            .buf_dev_queue = buf_dev_queue,
            .batch_params_vec = batch_params_vec,
            .batch_events_vec = batch_events_vec,
            .batch_cookie_vec = batch_cookie_vec,
            .period_sec = options.period_sec,
            .stats_nio = 0,
        });
    }

    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    for (size_t i = 0; i < options.nthreads; i++)
    {
        DeviceBatchIOThreadArgs &args = thread_args_vec[i];
        threads.emplace_back(
            [&options, &args]()
            {
                cpu_set_affinity(args.thrid);
                CUfileOpcode_t opcode = CUFILE_READ;
                if (options.benchmark_type == BenchmarkType::SEQUENTIAL_READ ||
                    options.benchmark_type == BenchmarkType::RANDOM_READ)
                {
                    opcode = CUFILE_READ;
                }
                else if (options.benchmark_type == BenchmarkType::SEQUENTIAL_WRITE ||
                         options.benchmark_type == BenchmarkType::RANDOM_WRITE)
                {
                    opcode = CUFILE_WRITE;
                }

                size_t nios = std::distance(args.pagid_vec_begin, args.pagid_vec_end);

                size_t done = 0;
                size_t ongoing = 0;
                uint64_t time_start = gettime();

                while (done < nios)
                {
                    if (CALC_SEC(gettime() - time_start) >= args.period_sec)
                    {
                        break;
                    }
                    if (ongoing < options.io_depth && done + ongoing < nios)
                    {
                        // Send a batch
                        size_t j = 0;
                        for (; j < std::min(options.io_depth - ongoing, nios - (done + ongoing)); j++)
                        {
                            CUfileIOParams_t *params = &args.batch_params_vec[done + ongoing + j];
                            params->mode = CUFILE_BATCH;
                            void *buf_dev = args.buf_dev_queue.front();
                            args.buf_dev_queue.pop();

                            params->u.batch.devPtr_base = buf_dev;
                            uint64_t pagid = args.pagid_vec_begin[done + ongoing + j];
                            uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                            uint64_t idev = pagid_to_idev(pagid, options.ndev);
                            uint64_t file_offset = ipagid * options.io_size;
                            params->u.batch.file_offset = file_offset;
                            params->u.batch.devPtr_offset = 0;
                            params->u.batch.size = options.io_size;
                            params->fh = args.cufile_handles[idev];
                            params->opcode = opcode;
                            DeviceBatchIOCookie *cookie = &args.batch_cookie_vec[done + ongoing + j];

                            cookie->buf_dev = buf_dev;
                            cookie->idx = done + ongoing + j;
                            params->cookie = (void *)cookie;
                            args.stats_nio++;
                        }
                        checkCuFileErrors(cuFileBatchIOSubmit(args.batch_idp, j, &args.batch_params_vec[done + ongoing], 0));
                        ongoing += j;
                    }
                    else
                    {
                        // Wait for a batch
                        unsigned int nrequests = ongoing;
                        checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, nrequests, &nrequests, &args.batch_events_vec[done], nullptr));

                        size_t j = 0;
                        for (; j < nrequests; j++)
                        {
                            CUfileIOEvents_t *events = &args.batch_events_vec[done + j];
                            if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                            {
                                std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                                std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                                std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                                std::cerr << "  events->cookie: " << events->cookie << std::endl;
                                DeviceBatchIOCookie *cookie = (DeviceBatchIOCookie *)events->cookie;
                                uint64_t pagid = args.pagid_vec_begin[cookie->idx];
                                std::cerr << "  params: pagid=" << pagid
                                          << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                          << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                                std::cerr << "  options.io_depth=" << options.io_depth << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            DeviceBatchIOCookie *cookie = (DeviceBatchIOCookie *)events->cookie;

                            args.buf_dev_queue.push(cookie->buf_dev);
                        }
                        ongoing -= j;
                        done += j;
                    }
                }
                while (ongoing > 0)
                {
                    // Wait for a batch
                    unsigned int nrequests = ongoing;
                    checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, nrequests, &nrequests, &args.batch_events_vec[done], nullptr));

                    size_t j = 0;
                    for (; j < nrequests; j++)
                    {
                        CUfileIOEvents_t *events = &args.batch_events_vec[done + j];
                        if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                        {
                            std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                            std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                            std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                            std::cerr << "  events->cookie: " << events->cookie << std::endl;
                            DeviceBatchIOCookie *cookie = (DeviceBatchIOCookie *)events->cookie;
                            uint64_t pagid = args.pagid_vec_begin[cookie->idx];
                            std::cerr << "  params: pagid=" << pagid
                                      << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                      << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                            std::cerr << "  options.io_depth=" << options.io_depth << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        DeviceBatchIOCookie *cookie = (DeviceBatchIOCookie *)events->cookie;

                        args.buf_dev_queue.push(cookie->buf_dev);
                    }
                    ongoing -= j;
                    done += j;
                }
            });
    }

    for (size_t i = 0; i < options.nthreads; i++)
    {
        threads[i].join();
    }

    auto end = chrono::system_clock::now();
    auto end_cpu_usage = read_cpu_usage();

    uint64_t naios_issued = 0;
    for (size_t i = 0; i < options.nthreads; i++)
    {
        DeviceBatchIOThreadArgs &args = thread_args_vec[i];
        naios_issued += thread_args_vec[i].stats_nio;

        (void)cuFileBatchIODestroy(args.batch_idp);

        for (size_t j = 0; j < options.io_depth; j++)
        {
            void *buf_dev = args.buf_dev_queue.front();
            args.buf_dev_queue.pop();

            mb_cufile_buf_deregister(buf_dev);
            mb_cuda_free(buf_dev);
        }
    }

    for (auto cufile_handle : cufile_handles)
    {
        mb_cufile_handle_deregister(cufile_handle);
    }

    mb_cufile_driver_close();

    close_files(options, fds);

    return BenchmarkResult{
        .nios = naios_issued,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = get_gpu_usage(),
    };
}
