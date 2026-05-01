#pragma once

#include "./common.cu"
#include "./primitive_c.cu"
#include "./primitive_cuda.cu"

struct HostDeviceAsyncThreadArgs
{
    size_t thrid;
    std::vector<int> &fds;
    void *buf_host;
    void *buf_dev;
    size_t io_size;
    size_t ndev;
    uint64_t period_sec;
    size_t stats_nio;
    cudaStream_t stream;
    std::vector<uint64_t>::iterator pagid_vec_begin;
    std::vector<uint64_t>::iterator pagid_vec_end;
};

BenchmarkResult bench_host_device_async(BenchmarkOptions &options)
{
    std::vector<int> fds;
    open_files(options, fds);

    std::vector<uint64_t> pagid_vec = generate_pagid_vec(options);

    size_t nios = pagid_vec.size();
    size_t nios_per_thread = (nios + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<HostDeviceAsyncThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    for (size_t i = 0; i < options.nthreads; i++)
    {
        void *buf_host = mb_alloc(options.io_size);
        checkCudaErrors(cudaHostRegister(buf_host, options.io_size, 0));
        void *buf_dev = mb_cuda_alloc(options.io_size);

        cudaStream_t stream = mb_cuda_stream_create();

        auto begin = pagid_vec.begin() + nios_per_thread * i;
        auto end = pagid_vec.begin() + std::min(nios_per_thread * (i + 1), nios);

        thread_args_vec.push_back(HostDeviceAsyncThreadArgs{
            .thrid = i,
            .fds = fds,
            .buf_host = buf_host,
            .buf_dev = buf_dev,
            .io_size = options.io_size,
            .ndev = options.ndev,
            .period_sec = options.period_sec,
            .stats_nio = 0,
            .stream = stream,
            .pagid_vec_begin = begin,
            .pagid_vec_end = end,
        });
    }

    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    if (options.benchmark_type == BenchmarkType::SEQUENTIAL_READ ||
        options.benchmark_type == BenchmarkType::RANDOM_READ)
    {
        for (size_t i = 0; i < options.nthreads; i++)
        {
            HostDeviceAsyncThreadArgs &args = thread_args_vec[i];
            threads.emplace_back(
                [&args]()
                {
                    cpu_set_affinity(args.thrid);
                    uint64_t time_read = 0;
                    uint64_t time_start = gettime();
                    uint64_t time_transfer = 0;
                    for (auto it = args.pagid_vec_begin; it != args.pagid_vec_end; it++)
                    {
                        if (CALC_SEC(gettime() - time_start) >= args.period_sec)
                        {
                            break;
                        }
                        uint64_t time_read_start = gettime();
                        uint64_t pagid = *it;
                        uint64_t ipagid = pagid_to_ipagid(pagid, args.ndev);
                        uint64_t idev = pagid_to_idev(pagid, args.ndev);
                        off_t offset = ipagid * args.io_size;
                        mb_pread(args.fds[idev], args.buf_host, args.io_size, offset);
                        uint64_t time_read_end = gettime();
                        time_read += time_read_end - time_read_start;
                        uint64_t time_transfer_start = gettime();
                        mb_cuda_memcpy_host_to_device_async(args.buf_dev, args.buf_host, args.io_size, args.stream);
                        mb_cuda_stream_synchronize(args.stream);
                        uint64_t time_transfer_end = gettime();
                        time_transfer += time_transfer_end - time_transfer_start;
                        args.stats_nio++;
                    }
                    // std::cout << time_read << "\t" << time_transfer << std::endl;
                });
        }
    }
    else if (options.benchmark_type == BenchmarkType::SEQUENTIAL_WRITE ||
             options.benchmark_type == BenchmarkType::RANDOM_WRITE)
    {
        for (size_t i = 0; i < options.nthreads; i++)
        {
            HostDeviceAsyncThreadArgs &args = thread_args_vec[i];
            threads.emplace_back(
                [&args]()
                {
                    cpu_set_affinity(args.thrid);
                    uint64_t time_start = gettime();
                    for (auto it = args.pagid_vec_begin; it != args.pagid_vec_end; it++)
                    {
                        if (CALC_SEC(gettime() - time_start) >= args.period_sec)
                        {
                            break;
                        }
                        uint64_t pagid = *it;
                        uint64_t ipagid = pagid_to_ipagid(pagid, args.ndev);
                        uint64_t idev = pagid_to_idev(pagid, args.ndev);
                        off_t offset = ipagid * args.io_size;
                        mb_cuda_memcpy_device_to_host_async(args.buf_host, args.buf_dev, args.io_size, args.stream);
                        mb_cuda_stream_synchronize(args.stream);
                        mb_pwrite(args.fds[idev], args.buf_host, args.io_size, offset);
                        args.stats_nio++;
                    }
                });
        }
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
        HostDeviceAsyncThreadArgs &args = thread_args_vec[i];
        naios_issued += thread_args_vec[i].stats_nio;
        checkCudaErrors(cudaHostUnregister(args.buf_host));
        free(args.buf_host);
        mb_cuda_free(args.buf_dev);
        mb_cuda_stream_destroy(args.stream);
    }
    close_files(options, fds);

    return BenchmarkResult{
        .nios = naios_issued,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = get_gpu_usage(),
    };
}
