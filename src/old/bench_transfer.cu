#pragma once

#include <ranges>

#include "./common.cu"
#include "./primitive_c.cu"
#include "./primitive_cuda.cu"

struct TransferThreadArgs
{
    size_t thrid;
    void *buf_host;
    void *buf_dev;
    size_t io_size;
    size_t nios;
};

BenchmarkResult bench_transfer(const BenchmarkOptions &options)
{
    size_t nios = options.file_size / options.io_size;
    size_t nios_per_thread = (nios + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<TransferThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    for (size_t i = 0; i < options.nthreads; i++)
    {
        void *buf_host = mb_alloc(options.io_size);
        checkCudaErrors(cudaHostRegister(buf_host, options.io_size, 0));
        void *buf_dev = mb_cuda_alloc(options.io_size);

        thread_args_vec.push_back(TransferThreadArgs{
            .thrid = i,
            .buf_host = buf_host,
            .buf_dev = buf_dev,
            .io_size = options.io_size,
            .nios = nios_per_thread,
        });
    }

    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    if (options.benchmark_type == BenchmarkType::HOST_TO_DEVICE)
    {
        for (size_t i = 0; i < options.nthreads; i++)
        {
            TransferThreadArgs &args = thread_args_vec[i];
            threads.emplace_back(
                [&args]()
                {
                    cpu_set_affinity(args.thrid);
                    uint64_t time_transfer = 0;
                    for (size_t j = 0; j < args.nios; j++)
                    {
                        uint64_t time_transfer_start = gettime();
                        mb_cuda_memcpy_host_to_device(args.buf_dev, args.buf_host, args.io_size);
                        uint64_t time_transfer_end = gettime();
                        time_transfer += time_transfer_end - time_transfer_start;
                    }
                    std::cout << time_transfer << std::endl;
                });
        }
    }
    else if (options.benchmark_type == BenchmarkType::DEVICE_TO_HOST)
    {
        for (size_t i = 0; i < options.nthreads; i++)
        {
            TransferThreadArgs &args = thread_args_vec[i];
            threads.emplace_back(
                [&args]()
                {
                    cpu_set_affinity(args.thrid);
                    for (size_t j = 0; j < args.nios; j++)
                    {
                        mb_cuda_memcpy_device_to_host(args.buf_host, args.buf_dev, args.io_size);
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

    for (size_t i = 0; i < options.nthreads; i++)
    {
        TransferThreadArgs &args = thread_args_vec[i];
        checkCudaErrors(cudaHostUnregister(args.buf_host));
        free(args.buf_host);
        mb_cuda_free(args.buf_dev);
    }

    return BenchmarkResult{
        .nios = nios,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = get_gpu_usage(),
    };
}
