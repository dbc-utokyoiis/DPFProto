#pragma once

#include <libaio.h>

#include "./common.cu"
#include "./primitive_c.cu"
#include "./primitive_cuda.cu"

struct HostAsyncDeviceThreadArgs
{
    size_t thrid;
    std::vector<int> fds;
    io_context_t ctx;
    std::vector<uint64_t>::iterator pagid_vec_begin;
    std::vector<uint64_t>::iterator pagid_vec_end;
    iocb *iocb_vec;
    iocb **iocbp_vec;
    std::queue<void *> buf_host_queue;
    std::queue<void *> buf_dev_queue;
    io_event *events_vec;
    uint64_t period_sec;
    size_t stats_nio;
};

BenchmarkResult bench_host_async_device(BenchmarkOptions &options)
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

    std::vector<HostAsyncDeviceThreadArgs> thread_args_vec;
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

        std::queue<void *> buf_host_queue;
        std::queue<void *> buf_dev_queue;

        io_event *events_vec = (io_event *)malloc(sizeof(io_event) * jobs_per_thread);

        for (size_t j = 0; j < options.io_depth; j++)
        {
            void *buf_host = mb_alloc(options.io_size);
            checkCudaErrors(cudaHostRegister(buf_host, options.io_size, 0));
            buf_host_queue.push(buf_host);

            void *buf_dev = mb_cuda_alloc(options.io_size);
            buf_dev_queue.push(buf_dev);
        }

        auto begin = pagid_vec.begin() + jobs_per_thread * i;
        auto end = pagid_vec.begin() + std::min(jobs_per_thread * (i + 1), nios);

        thread_args_vec.push_back(HostAsyncDeviceThreadArgs{
            .thrid = i,
            .fds = fds,
            .ctx = ctx,
            .pagid_vec_begin = begin,
            .pagid_vec_end = end,
            .iocb_vec = iocb_vec,
            .iocbp_vec = iocbp_vec,
            .buf_host_queue = buf_host_queue,
            .buf_dev_queue = buf_dev_queue,
            .events_vec = events_vec,
            .period_sec = options.period_sec,
            .stats_nio = 0,
        });
    }

    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    for (size_t i = 0; i < options.nthreads; i++)
    {
        HostAsyncDeviceThreadArgs &args = thread_args_vec[i];
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

                            void *buf_host = args.buf_host_queue.front();
                            args.buf_host_queue.pop();

                            uint64_t pagid = args.pagid_vec_begin[done + ongoing + j];
                            uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                            uint64_t idev = pagid_to_idev(pagid, options.ndev);
                            uint64_t offset = ipagid * options.io_size;
                            args.stats_nio++;

                            if (options.benchmark_type == BenchmarkType::SEQUENTIAL_READ ||
                                options.benchmark_type == BenchmarkType::RANDOM_READ)
                            {
                                io_prep_pread(iocbp, args.fds[idev], buf_host, options.io_size, offset);
                            }
                            else if (options.benchmark_type == BenchmarkType::SEQUENTIAL_WRITE ||
                                     options.benchmark_type == BenchmarkType::RANDOM_WRITE)
                            {
                                void *buf_dev = args.buf_dev_queue.front();
                                args.buf_dev_queue.pop();

                                mb_cuda_memcpy_device_to_host(buf_host, buf_dev, options.io_size);
                                io_prep_pwrite(iocbp, args.fds[idev], buf_host, options.io_size, offset);

                                args.buf_dev_queue.push(buf_dev);
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
                            void *buf_host = events->obj->u.c.buf;
                            if (options.benchmark_type == BenchmarkType::SEQUENTIAL_READ ||
                                options.benchmark_type == BenchmarkType::RANDOM_READ)
                            {
                                void *buf_dev = args.buf_dev_queue.front();
                                args.buf_dev_queue.pop();

                                mb_cuda_memcpy_host_to_device(buf_dev, buf_host, options.io_size);

                                args.buf_dev_queue.push(buf_dev);
                            }
                            args.buf_host_queue.push(buf_host);
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
                        void *buf_host = events->obj->u.c.buf;
                        if (options.benchmark_type == BenchmarkType::SEQUENTIAL_READ ||
                            options.benchmark_type == BenchmarkType::RANDOM_READ)
                        {
                            void *buf_dev = args.buf_dev_queue.front();
                            args.buf_dev_queue.pop();

                            mb_cuda_memcpy_host_to_device(buf_dev, buf_host, options.io_size);

                            args.buf_dev_queue.push(buf_dev);
                        }
                        args.buf_host_queue.push(buf_host);
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
        HostAsyncDeviceThreadArgs &args = thread_args_vec[i];
        naios_issued += thread_args_vec[i].stats_nio;
        for (size_t j = 0; j < options.io_depth; j++)
        {
            void *buf_host = args.buf_host_queue.front();
            args.buf_host_queue.pop();
            checkCudaErrors(cudaHostUnregister(buf_host));
            free(buf_host);

            void *buf_dev = args.buf_dev_queue.front();
            args.buf_dev_queue.pop();
            mb_cuda_free(buf_dev);
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
