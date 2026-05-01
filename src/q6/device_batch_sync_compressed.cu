#pragma once

#include "common/common.cu"
#include "common/page.cu"
#include "common/primitive_c.cu"
#include "common/primitive_cuda.cu"
#include "common/primitive_cufile.cu"
#include "metadata/metadata.h"
#include "schema/lineitem.cu"
#include "schema/table.h"
#include "kernel/tpch/q6.cuh"

#include "nvcomp/decompressor.cuh"


struct DeviceBatchSyncCompressedCookie
{
    void *buf_dev;
    size_t idx;
    size_t pagid;
};

struct DeviceBatchSyncCompressedThreadArgs
{
    size_t thrid;
    CUcontext ctx;
    size_t page_size;
    size_t sub_page_size;
    size_t pagid_start_index;
    size_t pagid_final;
    size_t subpagid_final;
    size_t nrows;
    size_t nrows_final;
    struct Metadata *metadata;
    std::vector<CUfileHandle_t> cufile_handles;
    CUfileBatchHandle_t batch_idp;
    std::vector<uint64_t>::iterator pagid_vec_begin;
    std::vector<uint64_t>::iterator pagid_vec_end;
    std::vector<uint64_t> pagid_vec;
    std::vector<void*> buf_compressed_dev_vec;
    std::vector<void*> buf_uncompressed_dev_vec;
    void * buf_out_dev;
    void * buf_out_host;
    std::vector<CUfileIOParams_t> batch_params_vec;
    std::vector<CUfileIOEvents_t> batch_events_vec;
    std::vector<DeviceBatchSyncCompressedCookie> batch_cookie_vec;
    std::vector<uint64_t> compressed_page_sizes_vec;
    std::vector<uint64_t> compressed_subpage_sizes_vec;
    uint64_t period_sec;
    size_t stats_nio;
};

BenchmarkResult bench_device_batch_sync_compressed(BenchmarkOptions &options)
{
    size_t i, j, n;
    std::vector<int> fds;
    std::vector<CUfileHandle_t> cufile_handles;

    // sine_t nrows_of_comp_sizes_per_page = options.page_size / size_of(usize_t);
    // size_t comp_page_sizes_npages = 
    //     if npages % nrows_of_comp_sizes_per_page == 0 {
    //         npages / nrows_of_comp_sizes_per_page
    //     } else {
    //         npages / nrows_of_comp_sizes_per_page + 1
    //     };
    /* initialize CUDA context */
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(ctx);


    open_files(options, fds);

    //std::vector<DeviceBatchSyncThreadArgs> thread_args_vec;
    //thread_args_vec.reserve(options.nthreads);
    struct Metadata *metadata = static_cast<Metadata*>(mb_alloc(options.page_size));
    mb_pread(fds[0], metadata, options.page_size, 0);

    std::cout << "page_size:                " << metadata->page_size << std::endl;
    std::cout << "compressed:               " << metadata->compressed << std::endl;
    std::cout << "sub_page_size:            " << metadata->sub_page_size << std::endl;
    std::cout << "table_customer_page_id:   " << metadata->table_customer_page_id << std::endl;
    std::cout << "table_customer_nrows:     " << metadata->table_customer_nrows   << std::endl;
    std::cout << "table_customer_npages:    " << metadata->table_customer_npages << std::endl;
    std::cout << "table_customer_nsubpages: " << metadata->table_customer_nsubpages << std::endl;
    std::cout << "table_orders_page_id:     " << metadata->table_orders_page_id << std::endl;
    std::cout << "table_orders_nrows:       " << metadata->table_orders_nrows << std::endl;
    std::cout << "table_orders_npages:      " << metadata->table_orders_npages << std::endl;
    std::cout << "table_orders_nsubpages:   " << metadata->table_orders_nsubpages << std::endl;
    std::cout << "table_lineitem_page_id:   " << metadata->table_lineitem_page_id << std::endl;
    std::cout << "table_lineitem_nrows:     " << metadata->table_lineitem_nrows << std::endl;
    std::cout << "table_lineitem_npages:    " << metadata->table_lineitem_npages << std::endl;
    std::cout << "table_lineitem_nsubpages: " << metadata->table_lineitem_nsubpages << std::endl;
    std::cout << "free_page_id:             " << metadata->free_page_id << std::endl;

    std::cout << "sizeof(CUfileIOEvents_t)       " << sizeof(CUfileIOEvents_t) << std::endl;
    std::cout << "sizeof(CUfileIOParams_t)       " << sizeof(CUfileIOParams_t) << std::endl;
    std::cout << "sizeof(DeviceBatchSyncCookie)  " << sizeof(DeviceBatchSyncCookie) << std::endl;

    auto table_lineitem = Table<Lineitem>(
        metadata->page_size,
        metadata->table_lineitem_page_id,
        metadata->table_lineitem_nrows,
        metadata->table_lineitem_npages,
        metadata->table_lineitem_nsubpages,
        metadata->compressed);

    Table<size_t> table_compressed_page_sizes = table_lineitem.compressed_page_sizes_table().value();
    Table<size_t> table_compressed_sub_page_sizes_table = table_lineitem.compressed_sub_page_sizes_table().value();

    std::cout << "=== lineitem table stats ===" << std::endl;
    std::cout << "lrec:             " << table_lineitem.get_rec_size() << std::endl;

#if 0
    struct Lineitem *lineitems = static_cast<Lineitem*>(mb_alloc(options.page_size));
    mb_pread(fds[0], lineitems, options.page_size, 38 * options.page_size);
    for (i = 0; i < 10; i++) {
        std::cout << "Lineitem[" << i << "]:" << lineitems[i].orderkey << "," << lineitems[i].linenumber << std::endl;
    }
    exit(1);
#endif
    //std::vector<cudaStream_t> streams;
    //size_t nstreams = options.nthreads * options.io_multiplicity;
    //for (i = 0; i < nstreams; i++) {
    //    auto stream = mb_cuda_stream_create();
    //    streams.push_back(stream);
    //}

    size_t nsubpages_per_page = options.page_size / options.sub_page_size;
    std::vector<uint64_t> pagid_vec = table_lineitem.generate_page_ids();
    size_t pagid_final = pagid_vec[pagid_vec.size() - 1];
    size_t subpagid_final;
    if (metadata->table_lineitem_nsubpages % nsubpages_per_page == 0) {
        subpagid_final = nsubpages_per_page - 1;
    } else {
        subpagid_final = metadata->table_lineitem_nsubpages % nsubpages_per_page;
    }
    std::cout << "nsubpage:" << nsubpages_per_page << std::endl;
    //size_t subpagid_final = metadata->table_lineitem_nsubpages - 1;
    std::cout << "=== table_lineitem ===" << std::endl;
    //auto pagid_vec_chunkded = chunk_vector(pagid_vec, options.nthreads);
    //auto pagid_chunked_vec = chunk_vector(pagid_vec, 2);
    auto pagid_start_index_vec = chunk_vector_start_indexes(pagid_vec, options.nthreads);;
    //if () {
    //    pagid_start_index_vec = chunk_vector_start_indexes(pagid_vec, options.nthreads);
    //} else {
    //    pagid_start_index_vec = chunk_vector_start_indexes(pagid_vec, options.nthreads);
    //}
    i = 0;
    //for (auto pagids : pagid_chunked_vec) {
    //    std::cout << "==" << i << ", len=" << pagids.size() << "==" << std::endl;
    //    for (auto pagid : pagids) {
    //        std::cout << pagid;
    //        if (pagid % 16 == 0) {
    //            std::cout << std::endl;
    //        } else {
    //            std::cout << " ";
    //        }
    //    }
    //    i += 1;
    //}

    std::cout << "=== table_lineitem start indexes ===" << std::endl;
    for (auto index : pagid_start_index_vec) {
        std::cout << index << std::endl;
    }

    std::vector<uint64_t> pagid_compressed_page_sizes = table_compressed_page_sizes.generate_page_ids();
    std::vector<uint64_t> compressed_page_sizes;
    std::cout << "=== table_compressed_page_sizes ===" << std::endl;
    std::cout << "table_compressed_page_sizes" << pagid_compressed_page_sizes.size() << std::endl;
    {
        size_t npages = table_compressed_page_sizes.get_npages();
        std::cout << "npages:" << npages << std::endl;
        std::cout << "nrows :" << table_compressed_page_sizes.get_nrows() << std::endl;
        char *buf = (char *)mb_alloc(options.page_size);
        size_t npage_sizes_per_page;
        size_t npage_sizes_last_page;
        size_t *ary;

        if (options.page_size % sizeof(size_t) == 0) {
            npage_sizes_per_page = options.page_size / sizeof(size_t);
        } else {
            npage_sizes_per_page = options.page_size / sizeof(size_t) + 1;
        }
        npage_sizes_last_page = metadata->table_lineitem_npages % npage_sizes_per_page;
        if (npage_sizes_last_page == 0) {
            npage_sizes_last_page = npage_sizes_per_page;
        }
        std::cout << "npage_sizes_per_page:" << npage_sizes_per_page << std::endl;
        std::cout << "npage_sizes_last_page:" << npage_sizes_last_page << std::endl;

        i = 0; j = 0;
        for (auto pagid : pagid_compressed_page_sizes) {
            page_pread_host(fds, buf, pagid, options.page_size);
            ary = reinterpret_cast<size_t*>(buf);

            n = 0;
            if (i < npages - 1) {
                n = npage_sizes_per_page;
            } else {
                n = npage_sizes_last_page;
            }
            for (size_t j = 0; j < n; j++) {
                compressed_page_sizes.push_back(ary[j]);
                // std::cout << ary[i];
                // if (i > 0 && i % 16 == 15) {
                //     std::cout << std::endl;
                // } else {
                //     std::cout << " ";
                // }   
            }
            // std::cout << pagid;
            // if (i > 0 && pagid % 16 == 15) {
            //     std::cout << std::endl;
            // } else {
            //     std::cout << " ";
            // }
            i++;
        }

        std::cout << "compressed_page_sizes.size():"<< compressed_page_sizes.size()
            << ",table_lineitem_npages.size():" << metadata->table_lineitem_npages << std::endl;
        assert(compressed_page_sizes.size() == metadata->table_lineitem_npages);
        std::cout << std::endl;
        free(buf);
    }

    std::vector<uint64_t> pagid_compressed_sub_page_sizes_table = table_compressed_sub_page_sizes_table.generate_page_ids();
    std::vector<uint64_t> compressed_subpage_sizes;
    std::cout << "=== table_compressed_page_sizes ===" << std::endl;
    {
        size_t npages = table_compressed_sub_page_sizes_table.get_npages();
        std::cout << "npages:" << npages << std::endl;
        std::cout << "nrows :" << table_compressed_sub_page_sizes_table.get_nrows() << std::endl;
        size_t *ary;
        char *buf = (char *)mb_alloc(options.page_size);
        size_t nsubpage_sizes_per_page;
        size_t nsubpage_sizes_last_page;

        if (options.page_size % sizeof(size_t) == 0) {
            nsubpage_sizes_per_page = options.page_size / sizeof(size_t);
        } else {
            nsubpage_sizes_per_page = options.page_size / sizeof(size_t) + 1;
        }
        nsubpage_sizes_last_page = metadata->table_lineitem_nsubpages % nsubpage_sizes_per_page;
        if (nsubpage_sizes_last_page == 0) {
            nsubpage_sizes_last_page = nsubpage_sizes_per_page;
        }
        std::cout << "nsubpage_sizes_per_page:" << nsubpage_sizes_per_page << std::endl;
        std::cout << "nsubpage_sizes_last_page:" << nsubpage_sizes_last_page << std::endl;

        // std::cout << pagid;
        // if (i > 0 && pagid % 16 == 15) {
        //     std::cout << std::endl;
        // } else {
        //     std::cout << " ";
        // }
        i = 0; j = 0;
        for (auto pagid : pagid_compressed_sub_page_sizes_table) {
            page_pread_host(fds, buf, pagid, options.page_size);
            ary = reinterpret_cast<size_t*>(buf);

            n = 0;
            if (i < npages - 1) {
                n = nsubpage_sizes_per_page;
            } else {
                n = nsubpage_sizes_last_page;
            }
            for (j = 0; j < n; j++) {
                compressed_subpage_sizes.push_back(ary[j]);
                // std::cout << ary[i];
                // if (i > 0 && i % 16 == 15) {
                //     std::cout << std::endl;
                // } else {
                //     std::cout << " ";
                // }   
            }
            i++;
        }
        std::cout << "compressed_subpage_sizes.size():"<< compressed_subpage_sizes.size()
            << ",table_lineitem_nsubpages.size():" << metadata->table_lineitem_nsubpages << std::endl;
        assert(compressed_subpage_sizes.size() == metadata->table_lineitem_nsubpages);
        // i += j;
        // while (i < metadata->table_lineitem_nsubpages) {
        // }
        std::cout << std::endl;
        free(buf);
    }


    mb_cufile_driver_open();

    cufile_handles.reserve(fds.size());
    for (auto fd : fds)
    {
        CUfileHandle_t cufile_handle = mb_cufile_handle_register(fd);
        cufile_handles.push_back(cufile_handle);
    }

    size_t nrows, nrows_final;
    nrows = options.sub_page_size / sizeof(Lineitem);
    nrows_final = table_lineitem.get_nrows() % nrows;
    size_t nios = pagid_vec.size();

    // if (nios % options.nthreads != 0)
    // {
    //     std::cerr << "nios must be a multiple of nthreads" << std::endl;
    //     exit(EXIT_FAILURE);
    // };

    size_t nios_per_thread = (nios + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<DeviceBatchSyncCompressedThreadArgs> thread_args_vec;
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

    void *buf_dev_head = mb_cuda_alloc(options.io_size * (options.io_depth * options.nthreads + 1));
    void *buf_dev_aligned = (void *)(((size_t)buf_dev_head + options.page_size - 1) & ~(options.page_size - 1));
    void *buf_uncompressed_dev_head = mb_cuda_alloc(options.io_size * (options.io_depth * options.nthreads + 1));
    #if 0
    std::cout << "buf_dev_head:" << buf_dev_head << ", buf_dev_aligned:" << buf_dev_aligned << std::endl;
    std::cout << "buf_uncompressed_dev_head:" << buf_uncompressed_dev_head << std::endl;
    #endif
    for (size_t i = 0; i < options.nthreads; i++)
    {
        CUfileBatchHandle_t batch_idp = nullptr;
        checkCuFileErrors(cuFileBatchIOSetUp(&batch_idp, std::min<size_t>(options.io_depth, props.max_batch_io_size)));

        std::vector<void *> buf_compressed_dev_vec;
        std::vector<void *> buf_uncompressed_dev_vec;

        std::vector<CUfileIOParams_t> batch_params_vec(nios_per_thread);

        std::vector<CUfileIOEvents_t> batch_events_vec(nios_per_thread);

        std::vector<DeviceBatchSyncCompressedCookie> batch_cookie_vec(nios_per_thread);

        //size_t nstreams = options.io_depth;
        //std::vector<cudaStream_t> streams(nstreams);
        //std::vector<cudaStream_t> streams;
        // for (i = 0; i < nstreams; i++) {
        //     auto stream = mb_cuda_stream_create();
        //     streams.push_back(stream);
        // }


        for (size_t j = 0; j < options.io_depth; j++)
        {
            // void *buf_dev = mb_cuda_alloc(options.io_size);
            size_t offset = (i * options.io_depth + j) * options.io_size;
            void *buf_dev = (void *)(
                &reinterpret_cast<uint8_t*>(buf_dev_aligned)[offset]);

            mb_cufile_buf_register(buf_dev, options.io_size);

            buf_compressed_dev_vec.push_back(buf_dev);

            // void *buf_uncompressed_dev = mb_cuda_alloc(options.io_size);
            void *buf_uncompressed_dev = (void *)(
                &reinterpret_cast<uint8_t*>(buf_uncompressed_dev_head)[offset]);
            buf_uncompressed_dev_vec.push_back(buf_uncompressed_dev);

            #if 0
            std::cout << "buf_compressed_dev_vec[" << j << "]:" << buf_compressed_dev_vec[j] << std::endl;
            std::cout << "buf_uncompressed_dev_vec[" << j << "]:" << buf_uncompressed_dev_vec[j] << std::endl;
            #endif
        }

        void *buf_out_dev = mb_cuda_alloc(4096);
        void *buf_out_host = mb_alloc(options.page_size);

        auto begin = pagid_vec.begin() + nios_per_thread * i;
        auto end = pagid_vec.begin() + std::min(nios_per_thread * (i + 1), nios);

        thread_args_vec.push_back(DeviceBatchSyncCompressedThreadArgs{
            .thrid = i,
            .ctx = ctx,
            .page_size = options.page_size,
            .sub_page_size = options.sub_page_size,
            .pagid_start_index = pagid_start_index_vec[i],
            .pagid_final = pagid_final,
            .subpagid_final = subpagid_final,
            .nrows = nrows,
            .nrows_final = nrows_final,
            .metadata = metadata,
            .cufile_handles = cufile_handles,
            .batch_idp = batch_idp,
            .pagid_vec_begin = begin,
            .pagid_vec_end = end,
            .buf_compressed_dev_vec = buf_compressed_dev_vec,
            .buf_uncompressed_dev_vec = buf_uncompressed_dev_vec,
            .buf_out_dev = buf_out_dev,
            .buf_out_host = buf_out_host,
            .batch_params_vec = batch_params_vec,
            .batch_events_vec = batch_events_vec,
            .batch_cookie_vec = batch_cookie_vec,
            .compressed_page_sizes_vec = compressed_page_sizes,
            .compressed_subpage_sizes_vec = compressed_subpage_sizes,
            .period_sec = options.period_sec,
            .stats_nio = 0,
        });
    }

    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    for (size_t i = 0; i < options.nthreads; i++)
    {
        DeviceBatchSyncCompressedThreadArgs &args = thread_args_vec[i];
        threads.emplace_back(
            [&options, &args]()
            {
                size_t i, j;
                cpu_set_affinity(args.thrid);
                mb_cuda_set_context(args.ctx);
                CUfileOpcode_t opcode = CUFILE_READ;

                size_t nios = std::distance(args.pagid_vec_begin, args.pagid_vec_end);

                const size_t nstreams = options.io_depth;
                std::vector<cudaStream_t> streams;
                for (i = 0; i < nstreams; i++) {
                    auto stream = mb_cuda_stream_create();
                    streams.push_back(stream);
                }

                //decompressor.set_output_buffer_async(
                //    std::vector<void*>(1, args.buf_out_dev),
                //    streams[0]); 

                // size_t page_size = args.page_size;
                // size_t sub_page_size = args.sub_page_size;
                size_t nsubpages = args.page_size / args.sub_page_size;
                size_t pagid_final = args.pagid_final;
                size_t subpagid_final = args.subpagid_final;
                size_t pagid_start_index = args.pagid_start_index;
                size_t pagid_start = 0;
                size_t npage_processed = 0;
                size_t nsubpage_processed = 0;
                size_t nlineitem_processed = 0;
                size_t nrows_normal = args.nrows;
                size_t nrows_final = args.nrows_final;
                size_t io_multiplicity = options.io_depth;
                size_t nbatch_max_decompression = nsubpages * io_multiplicity;
                std::vector<size_t> pagid_array(io_multiplicity);
                void *buf_in_decompressed_dev_head = args.buf_uncompressed_dev_vec[0];

                #if 0
                std::cout << "nbatch_max_decompression:" << nbatch_max_decompression
                    << ", nsubpages:" << nsubpages
                    << ", io_multiplicity:" << io_multiplicity << std::endl;
                #endif

                /* initialize decomressor */
                std::unique_ptr<Decompressor> decompressor = 
                    std::unique_ptr<Decompressor>(new Decompressor(nbatch_max_decompression, options.sub_page_size));

                std::vector<void*> device_output_ptrs;
                for (i = 0; i < io_multiplicity; i++) {
                    void *buf_dev = args.buf_uncompressed_dev_vec[i];
                    for (j = 0; j < nsubpages; j++) {
                        void *b = &reinterpret_cast<uint8_t*>(buf_dev)[j * args.sub_page_size];
                        device_output_ptrs.push_back(b);
                        #if 0
                        std::cout << "\t[" << j << "]:" << b << std::endl;
                        #endif
                    }
                }
                decompressor->set_output_buffer_async(
                    device_output_ptrs,
                    streams[0]);

                mb_cuda_stream_synchronize(streams[0]);

                std::vector<void*> device_input_ptrs(nbatch_max_decompression);
                std::vector<size_t> device_input_bytes(nbatch_max_decompression);

                //auto = &args.compressed_page_sizes_vec;
                //     = args.compressed_subpage_sizes_vec;
                size_t done = 0;
                size_t ongoing = 0;
                uint64_t time_start = gettime();
                CUevent barrier_event;

                while (done < nios)
                {
                    // if (CALC_SEC(gettime() - time_start) >= args.period_sec)
                    // {
                    //     break;
                    // }
                    if (ongoing < options.io_depth && done + ongoing < nios)
                    {
                        pagid_start = args.pagid_vec_begin[done];
                        // Send a batch
                        size_t j = 0;
                        for (; j < std::min(options.io_depth - ongoing, nios - (done + ongoing)); j++)
                        {

                            CUfileIOParams_t *params = &args.batch_params_vec[done + ongoing + j];
                            params->mode = CUFILE_BATCH;
                            void *buf_dev = args.buf_compressed_dev_vec[j];

                            size_t compressed_page_size = args.compressed_page_sizes_vec[(done + ongoing + j) + pagid_start_index];
                            // std::cout << "compressed_page_size[" << (done + ongoing + j) << "]:" << compressed_page_size << std::endl;

                            params->u.batch.devPtr_base = buf_dev;
                            uint64_t pagid = args.pagid_vec_begin[done + ongoing + j];
                            uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                            uint64_t idev = pagid_to_idev(pagid, options.ndev);
                            // std::cout << "pagid=" << pagid << ", ipagid=" << ipagid << ", idev=" << idev << std::endl;
                            uint64_t file_offset = ipagid * options.io_size;
                            params->u.batch.file_offset = file_offset;
                            params->u.batch.devPtr_offset = 0;
                            #if 1
                            params->u.batch.size = std::min(
                                options.io_size,
                                compressed_page_size % CUFILE_IO_ALIGN == 0 ?
                                    compressed_page_size : (compressed_page_size / CUFILE_IO_ALIGN + 1) * CUFILE_IO_ALIGN
                            );
                            #else
                            params->u.batch.size = options.io_size;
                            #endif
                            // std::cout << "params->u.batch.size:" << params->u.batch.size << std::endl;
                            params->fh = args.cufile_handles[idev];
                            params->opcode = opcode;
                            DeviceBatchSyncCompressedCookie *cookie = &args.batch_cookie_vec[done + ongoing + j];

                            cookie->buf_dev = buf_dev;
                            cookie->idx = done + ongoing + j;
                            cookie->pagid = pagid;
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

                        #if 0
                        std::cout << "nrequests: " << nrequests << std::endl;
                        #endif

                        device_input_ptrs.clear();
                        device_input_bytes.clear();

                        device_input_ptrs.resize(nbatch_max_decompression);
                        device_input_bytes.resize(nbatch_max_decompression);

                        size_t j = 0, k = 0;
                        size_t nbatch = 0;
                        for (; j < nrequests; j++)
                        {
                            CUfileIOEvents_t *events = &args.batch_events_vec[done + j];
                            //if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                            if (events->status != CUFILE_COMPLETE)
                            {
                                std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                                std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                                std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                                std::cerr << "  events->cookie: " << events->cookie << std::endl;
                                DeviceBatchSyncCompressedCookie *cookie = (DeviceBatchSyncCompressedCookie *)events->cookie;
                                uint64_t pagid = args.pagid_vec_begin[cookie->idx];
                                std::cerr << "  params: pagid=" << pagid
                                          << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                          << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                                std::cerr << "  options.io_depth=" << options.io_depth << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            DeviceBatchSyncCompressedCookie *cookie = (DeviceBatchSyncCompressedCookie *)events->cookie;

                            void *buf_dev = cookie->buf_dev;
                            size_t pagid = cookie->pagid;
                            //Lineitem *lineitems = (Lineitem *)buf_dev;
                            size_t n;
                            if (pagid == pagid_final) {
                                n = args.metadata->table_lineitem_nsubpages % nsubpages;
                            } else {
                                n = nsubpages;
                            }
 
                            size_t pagidx = pagid - pagid_start;
                            #if 0
                            std::cout << "pagidx: " << pagidx;
                            std::cout << ", pagid: " << pagid << ", pagid_start:" << pagid_start << std::endl;
                            #endif
                            size_t start_offset = 0;
                            for (k = 0; k < n; k++) {
                                size_t siz_compressed = args.compressed_subpage_sizes_vec[(pagid_start_index + done + pagidx) * nsubpages + k];
                                device_input_ptrs[pagidx * nsubpages + k] = &reinterpret_cast<uint8_t*>(buf_dev)[start_offset];
                                device_input_bytes[pagidx * nsubpages + k] = siz_compressed;
                                start_offset += siz_compressed;
                                #if 0
                                std::cout << 
                                    "comp_bytes[" << ((done + j) * nsubpages + k) << "]: " <<
                                    "compressed_bytes:" << siz_compressed <<
                                    ", start_offset:" << start_offset << std::endl;
                                #endif
                            }
                            nbatch += n;
                            pagid_array[j] = pagid;
                        }
                        #if 0
                        std::cout << "nbatch: " << nbatch << std::endl;
                        #endif

                        decompressor->set_input_async(
                            device_input_ptrs,
                            device_input_bytes,
                            streams[0]);
                        // for (size_t k = 0; k < nstreams; k++) {
                        //     mb_cuda_stream_synchronize(streams[k]);
                        // }
                        decompressor->decompress_async(nbatch, streams[0]);
                        mb_cuda_event_create(&barrier_event);
                        mb_cuda_event_wait_event(streams[0], barrier_event);
                        //for (size_t k = 0; k < nstreams; k++) {
                        //    mb_cuda_stream_synchronize(streams[k]);
                        //}
                        // exit(1);
                        // OK
                        // TODO: Use event record instread of synchronization.

                        #if 1
                        size_t nlineitem_total = 0;
                        size_t nsubpage_process_total = 0;
                        for (j = 0; j < nrequests; j++) {
                            //void *buf_dev = args.buf_uncompressed_dev_vec[j % io_multiplicity];
                            size_t pagid = pagid_array[j];
                            size_t bufidx = pagid - pagid_start;
                            void *buf_dev = args.buf_uncompressed_dev_vec[bufidx];

                            size_t nlineitem;
                            size_t nsubpages_process;
                            if (pagid == pagid_final) {
                                nlineitem = args.metadata->table_lineitem_nrows % (nrows_normal * nsubpages);
                                nsubpages_process = args.metadata->table_lineitem_nsubpages % nsubpages;
                                nsubpage_processed += args.metadata->table_lineitem_nsubpages % nsubpages;
                            } else {
                                nlineitem = nrows_normal * nsubpages;
                                nsubpages_process = nsubpages;
                                nsubpage_processed += nsubpages;
                            }
                            nlineitem_total += nlineitem;
                            nsubpage_process_total += nsubpages_process;
                        }
                        // std::cout << "pagid:" << pagid << ", nlineitem:" << nlineitem << std::endl;
                        Lineitem *lineitems = reinterpret_cast<Lineitem*>(reinterpret_cast<uint8_t*>(buf_in_decompressed_dev_head));
                        // buf_uncompressed_dev_head
                        q6_shared_subpage(
                            lineitems,
                            args.page_size,
                            args.sub_page_size,
                            nlineitem_total,
                            nrows_normal,
                            nrows_final,
                            nsubpage_process_total,
                            reinterpret_cast<int64_t*>(args.buf_out_dev),
                            streams[0]);
                            // q6_shared_subpage(
                            //     Lineitem *lineitems_buf,
                            //     size_t siz_page,
                            //     size_t siz_subpage,
                            //     size_t nlineitem,
                            //     size_t nlineitem_per_subpage,
                            //     size_t nlineitem_per_subpage_final,
                            //     size_t nsubpage,
                            //     int64_t *revenue,
                            //     cudaStream_t stream)

                        nlineitem_processed += nlineitem_total;
                        nsubpage_processed += nsubpage_process_total;

                        #if 0
                        mb_cuda_memcpy_device_to_host_async(args.buf_out_host, args.buf_out_dev, sizeof(int64_t), streams[0]);
                        mb_cuda_stream_synchronize(streams[0]);

                        std::cout << "revenue:" << *reinterpret_cast<int64_t*>(args.buf_out_host) << std::endl;
                        #endif

                        #else
                        for (j = 0; j < nrequests; j++) {
                            //void *buf_dev = args.buf_uncompressed_dev_vec[j % io_multiplicity];
                            size_t pagid = pagid_array[j];
                            size_t bufidx = pagid - pagid_start;
                            void *buf_dev = args.buf_uncompressed_dev_vec[bufidx];
                            #if 0
                            size_t nrows;

                            for (size_t k = 0; k < nsubpages; k++) {
                                Lineitem *lineitems = reinterpret_cast<Lineitem*>(&reinterpret_cast<uint8_t*>(buf_dev)[k * args.sub_page_size]);
                                if (pagid == pagid_final && k == subpagid_final - 1) {
                                    // std::cout << "[A][final] pagid:" << pagid << ", k:" << k << std::endl;
                                    nrows = nrows_final;
                                } else {
                                    nrows = nrows_normal;
                                }
                                q6_shared(lineitems, nrows, (int64_t*)args.buf_out_dev, streams[j % nstreams]);
                                nsubpage_processed++;
                                // std::cout << "nsubpage_processed:" << nsubpage_processed << std::endl;

                                #if 1
                                mb_cuda_memcpy_device_to_host_async(args.buf_out_host, args.buf_out_dev, sizeof(int64_t), streams[0]);
                                mb_cuda_stream_synchronize(streams[0]);
                                if (k == nsubpages - 1) {
                                    //std::cout << "pagid:" << pagid << ", nsubpage_processed:" << nsubpage_processed << std::endl;
                                    std::cout << "revenue:" << *reinterpret_cast<int64_t*>(args.buf_out_host) << std::endl;
                                } else if (pagid == pagid_final && k == subpagid_final - 1) {
                                    //std::cout << "pagid:" << pagid << ", nsubpage_processed:" << nsubpage_processed << std::endl;
                                    std::cout << "revenue:" << *reinterpret_cast<int64_t*>(args.buf_out_host) << std::endl;
                                }
                                #endif


                                if (nsubpage_processed == args.metadata->table_lineitem_nsubpages){
                                    break;
                                }
                            }
                            #else

                            size_t nlineitem;
                            size_t nsubpages_process;
                            if (pagid == pagid_final) {
                                nlineitem = args.metadata->table_lineitem_nrows % (nrows_normal * nsubpages);
                                nsubpages_process = args.metadata->table_lineitem_nsubpages % nsubpages;
                                nsubpage_processed += args.metadata->table_lineitem_nsubpages % nsubpages;
                            } else {
                                nlineitem = nrows_normal * nsubpages;
                                nsubpages_process = nsubpages;
                                nsubpage_processed += nsubpages;
                            }
                            // std::cout << "pagid:" << pagid << ", nlineitem:" << nlineitem << std::endl;
                            Lineitem *lineitems = reinterpret_cast<Lineitem*>(reinterpret_cast<uint8_t*>(buf_dev));
                            q6_shared_subpage(
                                lineitems,
                                args.page_size,
                                args.sub_page_size,
                                nlineitem,
                                nrows_normal,
                                nrows_final,
                                nsubpages_process,
                                reinterpret_cast<int64_t*>(args.buf_out_dev),
                                streams[j]);
                            // q6_shared_subpage(
                            //     Lineitem *lineitems_buf,
                            //     size_t siz_page,
                            //     size_t siz_subpage,
                            //     size_t nlineitem,
                            //     size_t nlineitem_per_subpage,
                            //     size_t nlineitem_per_subpage_final,
                            //     size_t nsubpage,
                            //     int64_t *revenue,
                            //     cudaStream_t stream)

                            nlineitem_processed += nlineitem;

                            #if 0
                            mb_cuda_memcpy_device_to_host_async(args.buf_out_host, args.buf_out_dev, sizeof(int64_t), streams[0]);
                            mb_cuda_stream_synchronize(streams[0]);

                            std::cout << "revenue:" << *reinterpret_cast<int64_t*>(args.buf_out_host) << std::endl;
                            #endif

                            //exit(1);
                            #endif

                            // Maybe we can dispatch the kernel.
                            // q6_shared_debug(lineitems, nrows, (int64_t*)args.buf_out_dev, streams[j]);

                            // args.buf_dev_queue.push(cookie->buf_dev);
                        }
                        #endif
                        ongoing -= j;
                        done += j;

                        mb_cuda_stream_synchronize(streams[0]);
                        //for (size_t k = 0; k < nstreams; k++) {
                        //    mb_cuda_stream_synchronize(streams[k]);
                        //}
                    }
                }
                #if 0
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
                            DeviceBatchSyncCookie *cookie = (DeviceBatchSyncCookie *)events->cookie;
                            uint64_t pagid = args.pagid_vec_begin[cookie->idx];
                            std::cerr << "  params: pagid=" << pagid
                                      << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                      << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                            std::cerr << "  options.io_depth=" << options.io_depth << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        DeviceBatchSyncCompressedCookie *cookie = (DeviceBatchSyncCompressedCookie *)events->cookie;

                        void *buf_dev = cookie->buf_dev;
                        size_t pagid = cookie->pagid;
                        // Lineitem *lineitems = (Lineitem *)buf_dev;
                        size_t nrows;

#if 0
                        for (size_t k = 0; k < nsubpages; k++) {
                            //Lineitem *lineitems = reinterpret_cast<uint8_t*>(buf_dev)[k * args.sub_page_size];
                            Lineitem *lineitems = reinterpret_cast<Lineitem*>(&reinterpret_cast<uint8_t*>(buf_dev)[k * args.sub_page_size]);
                            if (pagid == pagid_final && k == subpagid_final - 1) {
                                std::cout << "[B][final] pagid:" << pagid << ", k:" << k << std::endl;
                                nrows = nrows_final;
                            } else {
                                nrows = nrows_normal;
                            }
                            q6_shared(lineitems, nrows, (int64_t*)args.buf_out_dev, streams[j % nstreams]);
                            nsubpage_processed++;
                            std::cout << "nsubpage_processed:" << nsubpage_processed << std::endl;

                            if (nsubpage_processed == args.metadata->table_lineitem_nsubpages){
                                break;
                            }
                        }
                        #else
                            size_t nlineitem;
                            size_t nsubpages_process;
                            if (pagid == pagid_final) {
                                nlineitem = args.metadata->table_lineitem_nrows % (nrows_normal * nsubpages);
                                nsubpages_process = args.metadata->table_lineitem_nsubpages % nsubpages;
                                nsubpage_processed += args.metadata->table_lineitem_nsubpages % nsubpages;
                            } else {
                                nlineitem = nrows_normal * nsubpages;
                                nsubpages_process = nsubpages;
                                nsubpage_processed += nsubpages;
                            }
                            Lineitem *lineitems = reinterpret_cast<Lineitem*>(reinterpret_cast<uint8_t*>(buf_dev));
                            q6_shared_subpage(
                                lineitems,
                                args.page_size,
                                args.sub_page_size,
                                nlineitem,
                                nrows_normal,
                                nrows_final,
                                nsubpages_process,
                                reinterpret_cast<int64_t*>(args.buf_out_dev),
                                streams[j % nstreams]);
                            nlineitem_processed += nlineitem;
                        #endif

                        // void *buf_dev = cookie->buf_dev;
                        // Lineitem *lineitems = (Lineitem *)buf_dev;
                        // size_t nrows;
                        // if (cookie->pagid == args.pagid_final)
                        // {
                        //     nrows = args.nrows_final;
                        // }
                        // else
                        // {
                        //     nrows = args.nrows;
                        // }
                        // q6_shared_debug(lineitems, nrows, (int64_t*)args.buf_out_dev, streams[j]);
 

                        // args.buf_dev_queue.push(cookie->buf_dev);
                    }
                    ongoing -= j;
                    done += j;

                    for (size_t k = 0; k < nstreams; k++) {
                        cudaStreamSynchronize(streams[k]);
                    }
                }
                #endif

                mb_cuda_memcpy_device_to_host_async(args.buf_out_host, args.buf_out_dev, sizeof(int64_t), streams[0]);
                mb_cuda_stream_synchronize(streams[0]);

                std::cout << "revenue:" << *reinterpret_cast<int64_t*>(args.buf_out_host) << std::endl;
                std::cout << "nsubpage_processed:" << nsubpage_processed << std::endl;
                std::cout << "nlineitem_processed:" << nlineitem_processed << std::endl;
                
                for (i = 0; i < nstreams; i++) {
                    mb_cuda_stream_destroy(streams[i]);
                }
            });
    }

    for (size_t i = 0; i < options.nthreads; i++)
    {
        threads[i].join();
    }

    int64_t revenue = 0;
    for (size_t i = 0; i < options.nthreads; i++) {
        int64_t r = *reinterpret_cast<int64_t*>(thread_args_vec[i].buf_out_host);
        revenue += r;
    }
    std::cout << "revenue:" << revenue << std::endl;


    auto end = chrono::system_clock::now();
    auto end_cpu_usage = read_cpu_usage();

    uint64_t naios_issued = 0;
    for (size_t i = 0; i < options.nthreads; i++)
    {
        DeviceBatchSyncCompressedThreadArgs &args = thread_args_vec[i];
        naios_issued += thread_args_vec[i].stats_nio;

        (void)cuFileBatchIODestroy(args.batch_idp);

        for (size_t j = 0; j < options.io_depth; j++)
        {
            void *buf_dev = args.buf_compressed_dev_vec[j];
            mb_cufile_buf_deregister(buf_dev);
            // mb_cuda_free(buf_dev);

            // buf_dev = args.buf_uncompressed_dev_vec[j];
            // mb_cuda_free(buf_dev);
        }
        free(args.buf_out_host);
        mb_cuda_free(args.buf_out_dev);
    }
    mb_cuda_free(buf_uncompressed_dev_head);
    mb_cuda_free(buf_dev_head);

    for (auto cufile_handle : cufile_handles)
    {
        mb_cufile_handle_deregister(cufile_handle);
    }

    mb_cufile_driver_close();

    close_files(options, fds);

    //for (i = 0; i < nstreams; i++) {
    //    mb_cuda_stream_destroy(streams[i]);
    //}

    free(metadata);

    return BenchmarkResult{
        .nios = naios_issued,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = get_gpu_usage(),
    };
}
