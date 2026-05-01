#pragma once

#include <libaio.h>

#include "common/common.cu"
#include "common/primitive_c.cu"

struct HostAsyncThreadArgs
{
    size_t thrid;
    std::vector<int> fds;
    io_context_t ctx;
    std::vector<uint64_t>::iterator pagid_vec_begin;
    std::vector<uint64_t>::iterator pagid_vec_end;
    iocb *iocb_vec;
    iocb **iocbp_vec;
    std::queue<void *> buf_queue;
    io_event *events_vec;
    uint64_t period_sec;
    size_t stats_nio;
};

BenchmarkResult bench_host_async_io(BenchmarkOptions &options)
{
    std::vector<int> fds;
    open_files(options, fds);

    std::vector<uint64_t> pagid_vec = generate_pagid_vec(options);

    size_t nios = pagid_vec.size();

    // if (nios % options.nthreads != 0)
    // {
    //     std::cerr << "nios must be a multiple of nthreads" << std::endl;
    //     exit(EXIT_FAILURE);
    // };

    size_t jobs_per_thread = (nios + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<HostAsyncThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    for (size_t i = 0; i < options.nthreads; i++)
    {
        iocb *iocb_vec = (iocb *)malloc(sizeof(iocb) * jobs_per_thread);
        iocb **iocbp_vec = (iocb **)malloc(sizeof(iocb *) * jobs_per_thread);
        for (size_t j = 0; j < jobs_per_thread; j++)
        {
            iocbp_vec[j] = &iocb_vec[j];
        }

        io_context_t ctx{};

        // Set up the context
        int ret = io_setup(options.io_depth, &ctx);
        if (ret != 0)
        {
            errno = -ret;
            perror("io_setup");
            exit(EXIT_FAILURE);
        }

        std::queue<void *> buf_queue;

        io_event *events_vec = (io_event *)malloc(sizeof(io_event) * jobs_per_thread);

        for (size_t j = 0; j < options.io_depth; j++)
        {
            void *buf = mb_alloc(options.io_size);
            buf_queue.push(buf);
        }

        auto begin = pagid_vec.begin() + jobs_per_thread * i;
        auto end = pagid_vec.begin() + std::min(jobs_per_thread * (i + 1), nios);

        thread_args_vec.push_back(HostAsyncThreadArgs{
            .thrid = i,
            .fds = fds,
            .ctx = ctx,
            .pagid_vec_begin = begin,
            .pagid_vec_end = end,
            .iocb_vec = iocb_vec,
            .iocbp_vec = iocbp_vec,
            .buf_queue = buf_queue,
            .events_vec = events_vec,
            .period_sec = options.period_sec,
            .stats_nio = 0,
        });
    }

    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    for (size_t i = 0; i < options.nthreads; i++)
    {
        HostAsyncThreadArgs &args = thread_args_vec[i];
        threads.emplace_back(
            [&options, &args]()
            {
                cpu_set_affinity(args.thrid);
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
                        size_t j = 0;
                        for (; j < std::min(options.io_depth - ongoing, nios - (done + ongoing)); j++)
                        {
                            iocb *iocbp = args.iocbp_vec[done + ongoing + j];

                            void *buf = args.buf_queue.front();
                            args.buf_queue.pop();

                            uint64_t pagid = args.pagid_vec_begin[done + ongoing + j];
                            uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                            uint64_t idev = pagid_to_idev(pagid, options.ndev);
                            uint64_t offset = ipagid * options.io_size;
                            args.stats_nio++;

                            if (options.benchmark_type == BenchmarkType::SEQUENTIAL_READ ||
                                options.benchmark_type == BenchmarkType::RANDOM_READ)
                            {
                                io_prep_pread(iocbp, args.fds[idev], buf, options.io_size, offset);
                            }
                            else if (options.benchmark_type == BenchmarkType::SEQUENTIAL_WRITE ||
                                     options.benchmark_type == BenchmarkType::RANDOM_WRITE)
                            {
                                io_prep_pwrite(iocbp, args.fds[idev], buf, options.io_size, offset);
                            }
                        }

                        // Submit I/Os
                        int ret = io_submit(args.ctx, j, &args.iocbp_vec[done + ongoing]);
                        if (ret < 0)
                        {
                            errno = -ret;
                            perror("io_submit");
                            exit(EXIT_FAILURE);
                        }
                        if (ret != j)
                        {
                            fprintf(stderr, "io_submit: expected %zu, actual %d\n", j, ret);
                            exit(EXIT_FAILURE);
                        }

                        ongoing += j;
                    }
                    else
                    {
                        unsigned int nrequests = ongoing;
                        nrequests = io_getevents(args.ctx, 1, nrequests, &args.events_vec[done], nullptr);

                        size_t j = 0;
                        for (; j < nrequests; j++)
                        {
                            io_event *events = &args.events_vec[done + j];
                            args.buf_queue.push(events->obj->u.c.buf);
                        }
                        ongoing -= j;
                        done += j;
                    }
                }
                while (ongoing)
                {
                    unsigned int nrequests = ongoing;
                    nrequests = io_getevents(args.ctx, 1, nrequests, &args.events_vec[done], nullptr);

                    size_t j = 0;
                    for (; j < nrequests; j++)
                    {
                        io_event *events = &args.events_vec[done + j];
                        args.buf_queue.push(events->obj->u.c.buf);
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
        HostAsyncThreadArgs &args = thread_args_vec[i];
        naios_issued += thread_args_vec[i].stats_nio;
        for (size_t j = 0; j < options.io_depth; j++)
        {
            void *buf = args.buf_queue.front();
            args.buf_queue.pop();
            free(buf);
        }
        free(args.events_vec);
        free(args.iocbp_vec);
        free(args.iocb_vec);
        io_destroy(args.ctx);
    }

    close_files(options, fds);

    return BenchmarkResult{
        .nios = naios_issued,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = GpuUsage{},
    };
}
