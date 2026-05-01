#pragma once

#include "./common.cu"
#include "./primitive_c.cu"
#include "./primitive_cuda.cu"
#include "./primitive_cufile.cu"

struct DeviceThreadArgs
{
    size_t thrid;
    std::vector<CUfileHandle_t> cufile_handles;
    void *buf_dev;
    size_t io_size;
    size_t ndev;
    uint64_t period_sec;
    size_t stats_nio;
    std::vector<uint64_t>::iterator pagid_vec_begin;
    std::vector<uint64_t>::iterator pagid_vec_end;
};

BenchmarkResult bench_device(BenchmarkOptions &options)
{
    std::vector<int> fds;
    std::vector<CUfileHandle_t> cufile_handles;
    open_files(options, fds);

    mb_cufile_driver_open();

    for (auto fd : fds)
    {
        CUfileHandle_t cufile_handle = mb_cufile_handle_register(fd);
        cufile_handles.push_back(cufile_handle);
    }

    std::vector<uint64_t> pagid_vec = generate_pagid_vec(options);

    size_t nios = pagid_vec.size();
    size_t nios_per_thread = (nios + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<DeviceThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    for (size_t i = 0; i < options.nthreads; i++)
    {
        void *buf_dev = mb_cuda_alloc(options.io_size);

        mb_cufile_buf_register(buf_dev, options.io_size);

        auto begin = pagid_vec.begin() + nios_per_thread * i;
        auto end = pagid_vec.begin() + std::min(nios_per_thread * (i + 1), nios);

        thread_args_vec.push_back(DeviceThreadArgs{
            .thrid = i,
            .cufile_handles = cufile_handles,
            .buf_dev = buf_dev,
            .io_size = options.io_size,
            .ndev = options.ndev,
            .period_sec = options.period_sec,
            .stats_nio = 0,
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
            DeviceThreadArgs &args = thread_args_vec[i];
            threads.emplace_back(
                [&args]()
                {
                    cpu_set_affinity(args.thrid);
                    uint64_t time_read = 0;
                    uint64_t time_start = gettime();
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
                        off_t file_offset = ipagid * args.io_size;
                        off_t buf_offset = 0;
                        ssize_t nread = cuFileRead(args.cufile_handles[idev], args.buf_dev, args.io_size, file_offset, buf_offset);
                        if (nread < 0 || nread != args.io_size)
                        {
                            std::cerr << "cuFileRead failed (io_size: " << args.io_size << ", nread: " << nread << ")" << std::endl;
                            checkCuFileErrors(cuFileBufDeregister(args.buf_dev));
                            checkCudaErrors(cudaFree(args.buf_dev));
                            (void)cuFileHandleDeregister(args.cufile_handles[idev]);
                            checkCuFileErrors(cuFileDriverClose());
                            exit(EXIT_FAILURE);
                        }
                        args.stats_nio++;
                        uint64_t time_read_end = gettime();
                        time_read += time_read_end - time_read_start;
                    }
                    // std::cout << time_read << std::endl;
                });
        }
    }
    else if (options.benchmark_type == BenchmarkType::SEQUENTIAL_WRITE ||
             options.benchmark_type == BenchmarkType::RANDOM_WRITE)
    {
        for (size_t i = 0; i < options.nthreads; i++)
        {
            DeviceThreadArgs &args = thread_args_vec[i];
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
                        off_t file_offset = ipagid * args.io_size;
                        off_t buf_offset = 0;
                        ssize_t nwritten = cuFileWrite(args.cufile_handles[idev], args.buf_dev, args.io_size, file_offset, buf_offset);
                        if (nwritten < 0 || nwritten != args.io_size)
                        {
                            std::cerr << "cuFileWrite failed (io_size: " << args.io_size << ", nwritten: " << nwritten << ")" << std::endl;
                            checkCuFileErrors(cuFileBufDeregister(args.buf_dev));
                            checkCudaErrors(cudaFree(args.buf_dev));
                            (void)cuFileHandleDeregister(args.cufile_handles[idev]);
                            checkCuFileErrors(cuFileDriverClose());
                            exit(EXIT_FAILURE);
                        }
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
        DeviceThreadArgs &args = thread_args_vec[i];
        naios_issued += thread_args_vec[i].stats_nio;
        mb_cufile_buf_deregister(args.buf_dev);
        mb_cuda_free(args.buf_dev);
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
