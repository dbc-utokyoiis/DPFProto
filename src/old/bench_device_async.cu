#pragma once

#include "./common.cu"
#include "./primitive_cuda.cu"
#include "./primitive_cufile.cu"

// NOTE: only 4 IOs seems to be accepted.
struct DeviceAsyncCookie
{
    void *buf_dev;
};

struct DeviceAsyncThreadArgs
{
    size_t thrid;
    std::vector<CUfileHandle_t> cufile_handles;
    std::vector<uint64_t>::iterator pagid_vec_begin;
    std::vector<uint64_t>::iterator pagid_vec_end;
    std::vector<void *> buf_dev_vec;
    std::vector<size_t> async_io_size;
    std::vector<off_t> async_file_offset;
    std::vector<off_t> async_buf_offset;
    std::vector<ssize_t> async_bytes_read;
    CUstream stream;
};

void stream_cb(void *userData)
{
    // callback from device driver thread.
    std::cout << "test" << std::endl;
}

BenchmarkResult bench_device_async(BenchmarkOptions &options)
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

    size_t nios_per_thread = (nios + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::cout << "1" << std::endl;
    std::vector<DeviceAsyncThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

#if 0
    // sample code
    std::cout << "1" << std::endl;
    void *buf_dev = mb_cuda_alloc(options.io_size);
    std::cout << "2" << std::endl;
    mb_cufile_buf_register(buf_dev, options.io_size);
    std::cout << "3" << std::endl;
    size_t io_size = options.io_size;
    off_t file_offset = 0 * options.io_size;
    off_t buf_offset = 0;
    ssize_t bytes_read = 0;
    std::cout << "4" << std::endl;
    cudaStream_t stream = mb_cuda_stream_create();
    std::cout << "5" << std::endl;

    checkCuFileErrors(cuFileReadAsync(cufile_handles[0],
                                      buf_dev,
                                      &io_size,
                                      &file_offset,
                                      &buf_offset,
                                      &bytes_read,
                                      stream));
    cudaLaunchHostFunc(stream, stream_cb, NULL);
    mb_cuda_stream_synchronize(stream);
    if (bytes_read == -1)
    {
        std::cout << "IOError: " << bytes_read << std::endl;
    }
    else
    {
        std::cout << "Successed to call cuFileReadyAsync!" << std::endl;
    }
    mb_cufile_buf_deregister(buf_dev);
    mb_cuda_free(buf_dev);
#endif

    std::cout << "2" << std::endl;
    for (size_t i = 0; i < options.nthreads; i++)
    {
        std::vector<void *> buf_dev_vec;
        std::vector<size_t> async_io_size;
        std::vector<off_t> async_file_offset;
        std::vector<off_t> async_buf_offset;
        std::vector<ssize_t> async_bytes_read;

        buf_dev_vec.reserve(options.io_depth);
        async_io_size.reserve(options.io_depth);
        async_file_offset.reserve(options.io_depth);
        async_buf_offset.reserve(options.io_depth);
        async_bytes_read.reserve(options.io_depth);
        for (size_t j = 0; j < options.io_depth; j++)
        {
            void *buf_dev = mb_cuda_alloc(options.io_size);

            mb_cufile_buf_register(buf_dev, options.io_size);

            buf_dev_vec.push_back(buf_dev);

            async_io_size.push_back(options.io_size);
            async_file_offset.push_back(0);
            async_buf_offset.push_back(0);
            async_bytes_read.push_back(0);
        }

        cudaStream_t stream = mb_cuda_stream_create_with_nonbloking_flag();
        // cudaStream_t stream = mb_cuda_stream_create();
        auto begin = pagid_vec.begin() + nios_per_thread * i;
        auto end = pagid_vec.begin() + std::min(nios_per_thread * (i + 1), nios);

        thread_args_vec.push_back(DeviceAsyncThreadArgs{
            .thrid = i,
            .cufile_handles = cufile_handles,
            .pagid_vec_begin = begin,
            .pagid_vec_end = end,
            .buf_dev_vec = buf_dev_vec,
            .async_io_size = async_io_size,
            .async_file_offset = async_file_offset,
            .async_buf_offset = async_buf_offset,
            .async_bytes_read = async_bytes_read,
            .stream = stream,
        });
    }
    std::cout << "3" << std::endl;

    for (size_t i = 0; i < options.nthreads; i++)
    {
        DeviceAsyncThreadArgs &args = thread_args_vec[i];
        threads.emplace_back(
            [&options, &args]()
            {
                std::cout << "thrid: " << args.thrid << std::endl;
                cpu_set_affinity(args.thrid);
                size_t nios = std::distance(args.pagid_vec_begin, args.pagid_vec_end);
#if 1
                // size_t done = 0;
                size_t *async_io_size_p = args.async_io_size.data();
                off_t *async_file_offset_p = args.async_file_offset.data();
                off_t *async_buf_offset_p = args.async_buf_offset.data();
                ssize_t *async_bytes_read_p = args.async_bytes_read.data();

                assert(args.async_io_size.size() == options.io_depth);
                assert(args.async_file_offset.size() == options.io_depth);
                assert(args.async_buf_offset.size() == options.io_depth);
                assert(args.async_bytes_read.size() == options.io_depth);

                for (size_t j = 0; j < nios; j++)
                {
#if 0
                    uint64_t pagid = args.pagid_vec_begin[done + j];
                    uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                    uint64_t idev = pagid_to_idev(pagid, options.ndev);
                    size_t io_size = 0;
                    off_t file_offset = 0;
                    off_t buf_offset = 0;
                    ssize_t bytes_read = 0;

                    void *buf_dev = args.buf_dev_vec[j];
                    checkCuFileErrors(cuFileReadAsync(args.cufile_handles[idev],
                                                      buf_dev,
                                                      &io_size,
                                                      &file_offset,
                                                      &buf_offset,
                                                      &bytes_read,
                                                      args.stream));
#else
                    if (j > 0 && (j % options.io_depth) == 0)
                    {
                        mb_cuda_stream_synchronize(args.stream);
                    }
                    uint64_t pagid = args.pagid_vec_begin[j];

                    uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                    uint64_t idev = pagid_to_idev(pagid, options.ndev);
                    size_t k = j % options.io_depth;
                    size_t *io_sizep = &async_io_size_p[k];
                    off_t *file_offsetp = &async_file_offset_p[k];
                    off_t *buf_offsetp = &async_buf_offset_p[k];
                    ssize_t *bytes_readp = &async_bytes_read_p[k];

                    std::cout << "idev: " << idev << " ipagid: " << ipagid << std::endl;
                    *io_sizep = options.io_size;
                    *file_offsetp = ipagid * options.io_size;
                    *buf_offsetp = 0;
                    *bytes_readp = 0;

                    // std::cout
                    //     << "pagid  :" << pagid << '\n'
                    //     << "idev   :" << idev << '\n'
                    //     << "ipagid :" << ipagid << '\n'
                    //     << "io_size:" << io_size << '\n'
                    //     << "file_of:" << file_offset << '\n'
                    //     << "buf_off:" << buf_offset << '\n'
                    //     << "byte_rd:" << bytes_read << '\n'
                    //     << std::endl;
                    void *buf_dev = args.buf_dev_vec[k];
                    checkCuFileErrors(cuFileReadAsync(args.cufile_handles[idev],
                                                      buf_dev,
                                                      io_sizep,
                                                      file_offsetp,
                                                      buf_offsetp,
                                                      bytes_readp,
                                                      args.stream));
#endif
                }
                mb_cuda_stream_synchronize(args.stream);
#else
                size_t done = 0;
                size_t ongoing = 0;

                while (done < nios)
                {
                    if (ongoing < options.io_depth && done + ongoing < nios)
                    {
                        // Send a batch
                        size_t j = 0;
                        for (; j < std::min(options.io_depth - ongoing, nios - (done + ongoing)); j++)
                        {
                            CUfileIOParams_t *params = &args.async_params_vec[done + ongoing + j];
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
                            DeviceAsyncCookie *cookie = &args.async_cookie_vec[done + ongoing + j];

                            checkCuFileErrors(cuFileReadAsync(args.cufile_handles[idev],
                                                              buf_dev,
                                                              options.io_size,
                                                              file_offset,
                                                              0,
                                                              int *bytes_read_p,
                                                              CUstream stream));

                            cookie->buf_dev = buf_dev;
                            params->cookie = (void *)cookie;
                        }
                        // checkCuFileErrors(cuFileBatchIOSubmit(args.batch_idp, j, &args.async_params_vec[done + ongoing], 0));
                        ongoing += j;
                    }
                    else
                    {
                        // Wait for a batch
                        unsigned int nrequests = ongoing;
                        // checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, 1, &nrequests, &args.batch_events_vec[done], nullptr));

                        size_t j = 0;
                        for (; j < nrequests; j++)
                        {
                            CUfileIOEvents_t *events = &args.async_events_vec[done + j];
                            if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                            {
                                std::cerr << "  events->ret: " << events->ret << std::endl;
                                std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                                std::cerr << "  events->cookie: " << events->cookie << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            // DeviceBatchCookie *cookie = (DeviceBatchCookie *)events->cookie;

                            args.buf_dev_queue.push(cookie->buf_dev);
                        }
                        ongoing -= j;
                        done += j;
                    }
                }
#endif
            });
    }

    for (size_t i = 0; i < options.nthreads; i++)
    {
        threads[i].join();
    }

    auto end = chrono::system_clock::now();
    auto end_cpu_usage = read_cpu_usage();

    for (auto cufile_handle : cufile_handles)
    {
        mb_cufile_handle_deregister(cufile_handle);
    }

    mb_cufile_driver_close();

    close_files(options, fds);

    return BenchmarkResult{
        .nios = nios,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = get_gpu_usage(),
    };
}
