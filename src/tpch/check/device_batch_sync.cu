#pragma once

#include "common/common.cu"
#include "common/page.cu"
#include "common/primitive_c.cu"
#include "common/primitive_cuda.cu"
#include "common/primitive_cufile.cu"
//#include "metadata/metadata.h"
#include "schema/lineitem.cu"
#include "schema/table.h"
#include "schema/tpch_tables.cuh"
#include "kernel/tpch/scan.cuh"
//#include "kernel/tpch/gpu_pag.cuh"

#include <nvcomp.h>

#if 0
struct ScanDeviceBatchSyncCookie
{
    void *buf_dev;
    size_t idx;
    size_t pagid;
};

struct ScanDeviceBatchSyncThreadArgs
{
    size_t thrid;
    CUcontext ctx;
    size_t page_size;
    //size_t sub_page_size;
    size_t page_start_index;
    size_t page_id_final;
    //size_t subpagid_final;
    //size_t nrows;
    //size_t nrows_final;
    struct TPCHTableMetadata &metadata;
    std::vector<CUfileHandle_t> cufile_handles;
    CUfileBatchHandle_t batch_idp;
    std::vector<uint64_t>::iterator page_vec_begin;
    std::vector<uint64_t>::iterator page_vec_end;
    std::vector<uint64_t> pagid_vec;
    std::vector<void *> buf_dev_vec;
    void * buf_out_dev;
    void * buf_out_host;
    void ** arr_buf_dev;
    void ** arr_buf_host;
    std::vector<CUfileIOParams_t> batch_params_vec;
    std::vector<CUfileIOEvents_t> batch_events_vec;
    std::vector<ScanDeviceBatchSyncCookie> batch_cookie_vec;

    //std::vector<uint64_t> compressed_page_sizes_vec;
    //std::vector<uint64_t> compressed_subpage_sizes_vec;
    uint64_t kernel_exec_time = 0;
    uint64_t period_sec;
    size_t stats_nio;
};

#if 0
struct ScanWithDictDeviceBatchSyncThreadArgs
{
    size_t thrid;
    CUcontext ctx;
    size_t page_size;
    //size_t sub_page_size;
    size_t xtn_start_index;
    size_t xtn_id_final;
    //size_t subpagid_final;
    //size_t nrows;
    //size_t nrows_final;
    struct TPCHTableMetadata &metadata;
    std::vector<CUfileHandle_t> cufile_handles;
    CUfileBatchHandle_t batch_idp;
    std::vector<uint64_t>::iterator xtn_vec_begin;
    std::vector<uint64_t>::iterator xtn_vec_end;
    std::vector<uint64_t> pagid_vec;
    std::vector<void *> buf_dev_vec;
    std::vector<void *> buf_dict_dev_vec;
    std::vector<std::vector<size_t>> &vec_vec_max_npages_required;
    void * buf_out_dev;
    void * buf_out_host;
    void ** arr_buf_dev;
    void ** arr_buf_host;
    void ** arr_buf_dict_dev;
    void ** arr_buf_dict_host;
    std::vector<CUfileIOParams_t> batch_params_vec;
    std::vector<CUfileIOEvents_t> batch_events_vec;
    std::vector<ScanDeviceBatchSyncCookie> batch_cookie_vec;

    //std::vector<uint64_t> compressed_page_sizes_vec;
    //std::vector<uint64_t> compressed_subpage_sizes_vec;
    uint64_t kernel_exec_time = 0;
    uint64_t period_sec;
    size_t stats_nio;
};
#endif


static const char* memTypeStr(cudaMemoryType t) {
    switch (t) {
        case cudaMemoryTypeUnregistered: return "Unregistered (pageable host)";
        case cudaMemoryTypeHost:         return "Host (pinned)";
        case cudaMemoryTypeDevice:       return "Device";
        case cudaMemoryTypeManaged:      return "Unified/Managed";
        default:                         return "Unknown";
    }
}

void printPointerInfo(const void* p, const char* name) {
    cudaPointerAttributes attr{};
    cudaError_t e = cudaPointerGetAttributes(&attr, p);
    if (e != cudaSuccess) {
        std::cerr << name << ": cudaPointerGetAttributes failed: "
                  << cudaGetErrorString(e) << std::endl;
        return;
    }

    std::cout << name << " pointer info\n"
              << "  type:   " << memTypeStr(attr.type) << "\n"
              << "  device: " << attr.device << "\n"
              << "  hostPointer:   " << attr.hostPointer << "\n"
              << "  devicePointer: " << attr.devicePointer << "\n";
}

BenchmarkResult __tpch_scan_customer_device_batch_sync_column(
    BenchmarkOptions &options, TPCHTableMetadata &metadata, std::vector<int> &fds)
{
    std::vector<CUfileHandle_t> cufile_handles;
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(ctx);


    //Table<size_t> table_compressed_page_sizes = table_lineitem.compressed_page_sizes_table().value();
    //Table<size_t> table_compressed_sub_page_sizes_table = table_lineitem.compressed_sub_page_sizes_table().value();

    //std::cout << "=== lineitem table stats ===" << std::endl;
    //std::cout << "lrec:             " << table_lineitem.get_rec_size() << std::endl;

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

    //size_t nsubpages_per_page = options.page_size / options.sub_page_size;

    auto &varchar_field_indexes = TPCH::common::fmt.customer_varchar_field_indexes;

    size_t varchar_field_index;
    switch (options.query) {
        case TPCH::QueryId::CHECK11:
        {
            varchar_field_index = TPCH::Query::Check::Customer::CHECK11::SCAN_TARGET_COL_VARCHAR_IDX;
            break;
        }
        case TPCH::QueryId::CHECK12:
        {
            varchar_field_index = TPCH::Query::Check::Customer::CHECK12::SCAN_TARGET_COL_VARCHAR_IDX;
            break;
        }
        case TPCH::QueryId::CHECK13:
        {
            varchar_field_index = TPCH::Query::Check::Customer::CHECK13::SCAN_TARGET_COL_VARCHAR_IDX;
            break;
        }
        default:
        {
            std::cerr << "Unsupported query id:" << static_cast<int>(options.query) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    auto table_customer = TPCHTable(
        metadata.page_size,
        metadata.table_customer_start_page_ids[varchar_field_indexes[varchar_field_index]],
        metadata.table_customer_npages[varchar_field_indexes[varchar_field_index]],
        metadata.table_customer_nrows,
        metadata.compressed);

    std::vector<uint64_t> page_vec = table_customer.generate_page_ids();
    size_t page_id_final = page_vec[page_vec.size() - 1];
    auto page_start_index_vec = chunk_vector_start_indexes(page_vec, options.nthreads);

    std::cout << "=== table_customer start indexes ===" << std::endl;
    for (auto index : page_start_index_vec) {
        std::cout << index << std::endl;
    }

    #if 0
    std::vector<uint64_t> xtn_compressed_page_sizes = table_compressed_page_sizes.generate_page_ids();
    std::vector<uint64_t> compressed_page_sizes;
    std::cout << "=== table_compressed_page_sizes ===" << std::endl;
    std::cout << "table_compressed_page_sizes" << xtn_compressed_page_sizes.size() << std::endl;
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
    #endif
    constexpr size_t size_warp = 32;
    uint32_t max_nrecs_per_page_tbl;
    {
        size_t n = metadata.table_customer_max_nrows_in_page[varchar_field_indexes[varchar_field_index]];
        max_nrecs_per_page_tbl = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
    }

    #if 0
    for (size_t i = 0; i < TPCH::common::kCustomerMaxNClustersInDCT + 1; i++) {
        if (i == 0) {
            size_t n = metadata.table_customer_max_nrows_in_page[0];
            nrecs_per_page_size[i] = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
        } else if (metadata.dict_encoded) {
            size_t n = metadata.table_customer_max_nrows_in_dict[i - 1];
            nrecs_per_page_size[i] = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
        } else {
            break;
        }
    }
    #endif

    mb_cufile_driver_open();

    cufile_handles.reserve(fds.size());
    for (auto fd : fds)
    {
        CUfileHandle_t cufile_handle = mb_cufile_handle_register(fd);
        cufile_handles.push_back(cufile_handle);
    }

    size_t npages = page_vec.size();
    size_t npages_per_thread = (npages + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<ScanDeviceBatchSyncThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    /* Check io_depth value does not exceed its max value */
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);
    if (options.io_multiplicity > props.max_batch_io_size)
    {
        for (auto cufile_handle : cufile_handles)
        {
            mb_cufile_handle_deregister(cufile_handle);
        }
        mb_cufile_driver_close();
        close_files(options, fds);
        exit(EXIT_FAILURE);
    }

    /* for alignment */
    void *buf_dev_head = mb_cuda_alloc(options.io_size * (options.io_multiplicity * options.nthreads + 1));
    void *buf_dev_aligned = (void *)(((size_t)buf_dev_head + options.page_size - 1) & ~(options.page_size - 1));

    void **arr_buf_dev_base = reinterpret_cast<void**>(mb_cuda_alloc(sizeof(void*) * options.nthreads * options.io_multiplicity));
    void **arr_buf_host_base = reinterpret_cast<void**>(mb_cuda_host_alloc(sizeof(void*) * options.nthreads * options.io_multiplicity));
    cudaHostRegister(arr_buf_host_base, sizeof(void*) * options.nthreads * options.io_multiplicity, cudaHostRegisterDefault);

    // std::cout << "buf_dev_head:" << buf_dev_head << "buf_dev_aligned:" << buf_dev_aligned << std::endl;

    for (size_t i = 0; i < options.nthreads; i++)
    {
        CUfileBatchHandle_t batch_idp = nullptr;
        checkCuFileErrors(cuFileBatchIOSetUp(&batch_idp, std::min<size_t>(options.io_multiplicity, props.max_batch_io_size)));

        std::vector<void *> buf_dev_vec(options.io_multiplicity);

        std::vector<CUfileIOParams_t> batch_params_vec(options.io_multiplicity);

        std::vector<CUfileIOEvents_t> batch_events_vec(options.io_multiplicity);

        std::vector<ScanDeviceBatchSyncCookie> batch_cookie_vec(options.io_multiplicity);

        void **arr_buf_dev = arr_buf_dev_base + i * options.io_multiplicity;
        void **arr_buf_host = arr_buf_host_base + i * options.io_multiplicity;
        std::cout << "arr_buf_dev_base:" << arr_buf_dev_base << std::endl;
        std::cout << "arr_buf_host_base:" << arr_buf_host_base << std::endl;
        // exit(0);

        for (size_t j = 0; j < options.io_multiplicity; j++)
        {
            //void *buf_dev = (void*)mb_cuda_alloc_v2(options.io_size);
            size_t offset = (i * options.io_multiplicity + j) * options.io_size;
            void *buf_dev = (void *)(
                &reinterpret_cast<uint8_t*>(buf_dev_aligned)[offset]);

            mb_cufile_buf_register(buf_dev, options.io_size);
            #if 1
            std::cout << "buf_dev:" << buf_dev << std::endl;
            #endif

            buf_dev_vec[j] = buf_dev;
        }

        void *buf_out_dev = mb_cuda_alloc(4096);
        void *buf_out_host = mb_alloc(options.page_size);

        auto begin = page_vec.begin() + npages_per_thread * i;
        auto end = page_vec.begin() + std::min(npages_per_thread * (i + 1), npages);

        thread_args_vec.push_back(ScanDeviceBatchSyncThreadArgs{
            .thrid = i,
            .ctx = ctx,
            .page_size = options.page_size,
            .page_start_index = page_start_index_vec[i],
            .page_id_final = page_id_final,
            .metadata = metadata,
            .cufile_handles = cufile_handles,
            .batch_idp = batch_idp,
            .page_vec_begin = begin,
            .page_vec_end = end,
            .buf_dev_vec = buf_dev_vec,
            .buf_out_dev = buf_out_dev,
            .buf_out_host = buf_out_host,
            .arr_buf_dev = arr_buf_dev,
            .arr_buf_host = arr_buf_host,
            .batch_params_vec = batch_params_vec,
            .batch_events_vec = batch_events_vec,
            .batch_cookie_vec = batch_cookie_vec,
            //.nrecs_per_page_size = std::ref(nrecs_per_page_size),
            //.compressed_page_sizes_vec = compressed_page_sizes,
            //.compressed_subpage_sizes_vec = compressed_subpage_sizes,
            .period_sec = options.period_sec,
            .stats_nio = 0,
        });
    }

    
    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    for (size_t i = 0; i < options.nthreads; i++)
    {
        ScanDeviceBatchSyncThreadArgs &args = thread_args_vec[i];
        threads.emplace_back(
            [&options, &args, &max_nrecs_per_page_tbl]()
            {
                size_t i;
                cpu_set_affinity(args.thrid);
                mb_cuda_set_context(args.ctx);
                CUfileOpcode_t opcode = CUFILE_READ;

                void **arr_buf_dev = args.arr_buf_dev;
                void **arr_buf_host = args.arr_buf_host;
                size_t npages = std::distance(args.page_vec_begin, args.page_vec_end);

                size_t io_multiplicity = options.io_multiplicity;
                size_t nstreams = options.io_multiplicity;
                std::vector<cudaStream_t> streams;
                for (i = 0; i < nstreams; i++) {
                    auto stream = mb_cuda_stream_create();
                    streams.push_back(stream);
                }

                uint customer_scan_attr = 0;
                uint customer_nattrs = 1;

                size_t done = 0;
                size_t ongoing = 0;
                size_t nios;
                uint64_t time_start = gettime();
                uint64_t kernel_exec_time = 0;
                struct TPCHTableMetadata &metadata = args.metadata;

                std::vector<uint64_t> pagid_vec {};

                void *buf_in_dev_head = args.buf_dev_vec[0];
                for (size_t i = 0; i < npages; i++) {
                    uint64_t page_id = args.page_vec_begin[i];
                    //size_t npages_in_xtn = xtn_get_allocated_npages(metadata.xtn_head, xtn_id);
                    pagid_vec.push_back(page_id);

                    //for (size_t j = 0; j < npages_in_xtn; j++)
                    //{
                    //    uint64_t pagid = xtn_calc_page_id(metadata.xtn_head, xtn_id, j);
                    //    pagid_vec.push_back(pagid);
                    //}
                }

                //pag_buffers.reserve(io_multiplicity);
                nios = pagid_vec.size();
                while (done < nios)
                {
                    if (ongoing < io_multiplicity && done + ongoing < nios)
                    {
                        // Send a batch
                        size_t j = 0;
                        for (; j < std::min(options.io_multiplicity - ongoing, nios - (done + ongoing)); j++)
                        {
                            CUfileIOParams_t *params = &args.batch_params_vec[j];
                            memset(params, 0, sizeof(CUfileIOParams_t));
                            params->mode = CUFILE_BATCH;
                            void *buf_dev = args.buf_dev_vec[j];
                            // if (buf_in_dev_head == nullptr) {
                            //     buf_in_dev_head = buf_dev;
                            // } else {
                            //     buf_in_dev_head = std::min(buf_dev, buf_in_dev_head);
                            // }

                            params->u.batch.devPtr_base = buf_dev;
                            uint64_t pagid = pagid_vec[done + ongoing + j];
                            //metadata.
                            //uint64_t pagid = args.pagid_vec_begin[done + ongoing + j];
                            //uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                            //uint64_t idev = pagid_to_idev(pagid, options.ndev);

                            //uint64_t pagid = 3;
                            uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                            uint64_t idev = pagid_to_idev(pagid, options.ndev);
                            // std::cout << "pagid=" << pagid << ", ipagid=" << ipagid << ", idev=" << idev << std::endl;
                            uint64_t file_offset = ipagid * options.io_size;
                            params->u.batch.file_offset = file_offset;
                            params->u.batch.devPtr_offset = 0;
                            params->u.batch.size = options.io_size;
                            params->fh = args.cufile_handles[idev];
                            params->opcode = opcode;
                            ScanDeviceBatchSyncCookie *cookie = &args.batch_cookie_vec[j];

                            cookie->buf_dev = buf_dev;
                            cookie->idx = done + ongoing + j;
                            cookie->pagid = pagid;
                            params->cookie = (void *)cookie;
                            args.stats_nio++;
                        }
                        // std::cout << "nFileBatchIOSubmit:" << j << std::endl;
                        checkCuFileErrors(cuFileBatchIOSubmit(args.batch_idp, j, &args.batch_params_vec[0], 0));
                        ongoing += j;
                    }
                    else
                    {
                        // Wait for a batch
                        unsigned int nrequests = ongoing;
                        checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, nrequests, &nrequests, &args.batch_events_vec[0], nullptr));

                        size_t j = 0;
                        for (; j < nrequests; j++)
                        {
                            CUfileIOEvents_t *events = &args.batch_events_vec[j];
                            if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                            {
                                std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                                std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                                std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                                std::cerr << "  events->cookie: " << events->cookie << std::endl;
                                ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                                uint64_t pagid = pagid_vec[cookie->idx];
                                std::cerr << "  params: pagid=" << pagid
                                          << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                          << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                                std::cerr << "  options.io_multiplicity=" << options.io_multiplicity << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;

                            // size_t pagid = cookie->pagid;
                            void *buf_dev = cookie->buf_dev;
                            //pag_buffers.push_back(buf_dev);
                            arr_buf_host[j] = buf_dev;
                        }

                        //printPointerInfo(arr_buf_host, "src");
                        // exit(1);
                        #if 1
                        mb_cuda_memcpy_host_to_device_async(arr_buf_dev, arr_buf_host, j * sizeof(void*), streams[0]);
                        //void **pags = pag_buffers.data();
                        size_t npags = j;
                        //for (size_t i = 0; i < npags; i++) {
                        //    std::cout << "  pags[" << i << "]: " << std::hex << pags[i] << std::endl;
                        //}

                        uint64_t kernel_exec_time_start = gettime();
                        scan_customer_row(
                                arr_buf_dev,
                                npags,
                                customer_nattrs,
                                customer_scan_attr,
                                args.page_size,
                                max_nrecs_per_page_tbl,
                                reinterpret_cast<int64_t*>(args.buf_out_dev),
                                options.enable_prefetch,
                                streams[0]);
                        uint64_t kernel_exec_time_end = gettime();
                        kernel_exec_time += (kernel_exec_time_end - kernel_exec_time_start);
                        #endif

                        ongoing -= j;
                        done += j;

                        //for (size_t k = 0; k < nrequests; k++) {
                        //    mb_cuda_stream_synchronize(streams[k % nstreams]);
                        //}
                        mb_cuda_stream_synchronize(streams[0]);
                    }
                }
                while (ongoing > 0)
                {
                    // Wait for a batch
                    unsigned int nrequests = ongoing;
                    checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, nrequests, &nrequests, &args.batch_events_vec[0], nullptr));

                    size_t j = 0;
                    for (; j < nrequests; j++)
                    {
                        CUfileIOEvents_t *events = &args.batch_events_vec[j];
                        if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                        {
                            std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                            std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                            std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                            std::cerr << "  events->cookie: " << events->cookie << std::endl;
                            ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                            uint64_t pagid = pagid_vec[cookie->idx];
                            std::cerr << "  params: pagid=" << pagid
                                      << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                      << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                            std::cerr << "  options.io_multiplicity=" << io_multiplicity << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                        void *buf_dev = cookie->buf_dev;
                        // pag_buffers.push_back(buf_dev);
                        arr_buf_host[j] = buf_dev;
                    }

                    #if 1
                    mb_cuda_memcpy_host_to_device_async(arr_buf_dev, arr_buf_host, j * sizeof(void*), streams[0]);
                    size_t npags = j;
                    uint64_t kernel_exec_time_start = gettime();
                    scan_customer_row(
                            arr_buf_dev,
                            npags,
                            customer_nattrs,
                            customer_scan_attr,
                            args.page_size,
                            max_nrecs_per_page_tbl,
                            reinterpret_cast<int64_t*>(args.buf_out_dev),
                            options.enable_prefetch,
                            streams[0]);
                    #endif
                    ongoing -= j;
                    done += j;

                    mb_cuda_stream_synchronize(streams[0]);

                    uint64_t kernel_exec_time_end = gettime();
                    kernel_exec_time += (kernel_exec_time_end - kernel_exec_time_start);
                }

                mb_cuda_memcpy_device_to_host_async(args.buf_out_host, args.buf_out_dev, sizeof(int64_t), streams[0]);
                mb_cuda_stream_synchronize(streams[0]);

                std::cout << "nlines_scanned:" << *reinterpret_cast<int64_t*>(args.buf_out_host) << std::endl;
                //std::cout << "nsubpage_processed:" << nsubpage_processed << std::endl;
                //std::cout << "nlineitem_processed:" << nlineitem_processed << std::endl;
                
                args.kernel_exec_time = kernel_exec_time;
                for (i = 0; i < nstreams; i++) {
                    mb_cuda_stream_destroy(streams[i]);
                }
            });
    }

    for (size_t i = 0; i < options.nthreads; i++)
    {
        threads[i].join();
    }

    int64_t count = 0;
    uint64_t kernel_exec_time = 0;
    for (size_t i = 0; i < options.nthreads; i++) {
        int64_t r = *reinterpret_cast<int64_t*>(thread_args_vec[i].buf_out_host);
        count += r;
        kernel_exec_time = std::max(kernel_exec_time, thread_args_vec[i].kernel_exec_time);
    }
    std::cout << "count:" << count << std::endl;
    std::cout << "kernel_exec_time:" << kernel_exec_time << std::endl;

    auto end = chrono::system_clock::now();
    auto end_cpu_usage = read_cpu_usage();

    uint64_t naios_issued = 0;
    for (size_t i = 0; i < options.nthreads; i++)
    {
        ScanDeviceBatchSyncThreadArgs &args = thread_args_vec[i];
        naios_issued += thread_args_vec[i].stats_nio;

        (void)cuFileBatchIODestroy(args.batch_idp);

        for (size_t j = 0; j < options.io_multiplicity; j++)
        {
            void *buf_dev = args.buf_dev_vec[j];

            mb_cufile_buf_deregister(buf_dev);
            // mb_cuda_free(buf_dev);
        }
        free(args.buf_out_host);
        mb_cuda_free(args.buf_out_dev);
    }
    cudaHostUnregister(arr_buf_host_base);
    mb_cuda_host_free(arr_buf_host_base);
    mb_cuda_free(arr_buf_dev_base);
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

    //free(metadata.dct_meta_head);
    //free(metadata.xtn_head);

    //std::cout << "[DEBUG] start: "<< start << "end: " << end << std::endl;
    //std::cout << end - start << " sec" << std::endl;
    std::cout << "[DEBUG] diff:" << (end - start).count() << std::endl;

    return BenchmarkResult{
        .nios = naios_issued,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = get_gpu_usage(),
    };
}


#if 0
BenchmarkResult __tpch_scan_customer_device_batch_sync_dict(
    BenchmarkOptions &options, TPCHTableMetadata &metadata, std::vector<int> &fds)
{
    std::vector<CUfileHandle_t> cufile_handles;

    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(ctx);

    auto table_customer = TPCHTable(
        metadata.page_size,
        metadata.table_customer_start_page_ids[0],
        metadata.table_customer_npages[0],
        metadata.table_customer_nrows,
        metadata.compressed);

    std::vector<uint64_t> page_vec = table_customer.generate_page_ids();
    size_t page_id_final = page_vec[page_vec.size() - 1];
    auto page_start_index_vec = chunk_vector_start_indexes(page_vec, options.nthreads);

    std::cout << "=== table_customer start indexes ===" << std::endl;
    for (auto index : page_start_index_vec) {
        std::cout << index << std::endl;
    }

    auto &varchar_field_indexes = TPCH::common::fmt.customer_varchar_field_indexes;
    size_t varchar_field_index;
    switch (options.query) {
        case TPCH::QueryId::CHECK11:
        {
            varchar_field_index = TPCH::Query::Check::Customer::CHECK11::SCAN_TARGET_COL_VARCHAR_IDX;
            break;
        }
        case TPCH::QueryId::CHECK12:
        {
            varchar_field_index = TPCH::Query::Check::Customer::CHECK12::SCAN_TARGET_COL_VARCHAR_IDX;
            break;
        }
        case TPCH::QueryId::CHECK13:
        {
            varchar_field_index = TPCH::Query::Check::Customer::CHECK13::SCAN_TARGET_COL_VARCHAR_IDX;
            break;
        }
        default:
        {
            std::cerr << "Unsupported query id:" << static_cast<int>(options.query) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    constexpr size_t size_warp = 32;
    uint32_t max_nrecs_per_page_tbl;
    {
        size_t n = metadata.table_customer_max_nrows_in_page[varchar_field_indexes[varchar_field_index]];
        max_nrecs_per_page_tbl = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
    }

    //std::array<std::array<uint32_t, DCT::kNumMaxDictsInXtn>, TPCH::common::fmt.customer_varchar_field_count> max_nrecs_per_page_dct{};
    exit(0);
#if 0
    if (metadata.dict_encoded) {
        for (size_t i = 0; i < TPCH::common::fmt.customer_varchar_field_count; ++i) {
            for (size_t j = 0; j < DCT::kNumMaxDictsInXtn; ++j) {
                size_t n = metadata.table_customer_max_nrows_in_dict[i][j];
                std::cout << "table_customer_max_nrows_in_dict[" << i << "][" << j << "]: " << n << std::endl;
                max_nrecs_per_page_dct[i][j] = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
            }
        }
    }
#endif
    //exit(1);

    /* calculate dict size first */
    size_t customer_num_varchar_fields = TPCH::common::fmt.customer_varchar_field_count;
    // auto &customer_varchar_field_indexes = TPCH::common::fmt.customer_varchar_field_indexes;
    size_t npages_for_dicts = 0;

    std::vector<std::vector<size_t>> vec_vec_max_npages_required(customer_num_varchar_fields);
    {
#if 0
        const size_t num_varchar_clusters = 8;
        size_t customer_start_page_id = metadata.table_customer_start_page_ids[0];
        size_t customer_npages = metadata.table_customer_npages[0];

        vec_vec_max_npages_required.resize(customer_num_varchar_fields);
        for (size_t j = 0; j < customer_num_varchar_fields; j++) {
            vec_vec_max_npages_required[j].resize(num_varchar_clusters);
            std::fill(vec_vec_max_npages_required[j].begin(), vec_vec_max_npages_required[j].end(), 0);
        }

        for (size_t i = 0; i < customer_npages; i++) {
            const uint64_t page_id = customer_start_page_id + i;
            //struct dct_meta_entry *dmp = &metadata.dct_meta_head[page_id];
            for (size_t j = 0; j < customer_num_varchar_fields; j++) {
                std::vector<size_t> &vec_max_npages_required = vec_vec_max_npages_required[j];
                for (size_t k = 0; k < num_varchar_clusters; k++) {
                    size_t n = dmp->npages[j][k];
                    vec_max_npages_required[k] = std::max(n, vec_max_npages_required[k]);
                    // std::cout << "Number of pages for customer XTN " << xtn_id << ", field " << j << ", cluster " << k << ": " << n << std::endl;
                }
            }
        }
        for (size_t j = 0; j < customer_num_varchar_fields; j++) {
            std::vector<size_t> &vec_max_npages_required = vec_vec_max_npages_required[j];
            for (size_t k = 0; k < num_varchar_clusters; k++) {
                size_t n = vec_max_npages_required[k];
                npages_for_dicts += n;
            }
        }
#endif
    }
    std::cout << npages_for_dicts << std::endl;


    mb_cufile_driver_open();

    cufile_handles.reserve(fds.size());
    for (auto fd : fds)
    {
        CUfileHandle_t cufile_handle = mb_cufile_handle_register(fd);
        cufile_handles.push_back(cufile_handle);
    }

    size_t npages = page_vec.size();
    size_t npages_per_thread = (npages + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<ScanWithDictDeviceBatchSyncThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    /* Check io_depth value does not exceed its max value */
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);
    if (options.io_multiplicity > props.max_batch_io_size)
    {
        for (auto cufile_handle : cufile_handles)
        {
            mb_cufile_handle_deregister(cufile_handle);
        }
        mb_cufile_driver_close();
        close_files(options, fds);
        exit(EXIT_FAILURE);
    }

    /* for alignment */
    void *buf_dev_head = mb_cuda_alloc(options.io_size * (options.io_multiplicity * options.nthreads + 1));
    void *buf_dev_aligned = (void *)(((size_t)buf_dev_head + options.page_size - 1) & ~(options.page_size - 1));

    /* for dict */
    // const size_t num_varchar_clusters = 8;
    void *buf_dict_dev_head = mb_cuda_alloc(options.io_size * (npages_for_dicts * options.nthreads + 1));
    void *buf_dict_dev_aligned = (void *)(((size_t)buf_dict_dev_head + options.page_size - 1) & ~(options.page_size - 1));

    void **arr_buf_dev_base = reinterpret_cast<void**>(mb_cuda_alloc(sizeof(void*) * options.nthreads * options.io_multiplicity));
    void **arr_buf_host_base = reinterpret_cast<void**>(mb_cuda_host_alloc(sizeof(void*) * options.nthreads * options.io_multiplicity));
    cudaHostRegister(arr_buf_host_base, sizeof(void*) * options.nthreads * options.io_multiplicity, cudaHostRegisterDefault);

    void **arr_buf_dict_dev_base = reinterpret_cast<void**>(mb_cuda_alloc(sizeof(void*) * options.nthreads * npages_for_dicts));
    void **arr_buf_dict_host_base = reinterpret_cast<void**>(mb_cuda_host_alloc(sizeof(void*) * options.nthreads * npages_for_dicts));
    cudaHostRegister(arr_buf_dict_host_base, sizeof(void*) * options.nthreads * npages_for_dicts, cudaHostRegisterDefault);


    // std::cout << "buf_dev_head:" << buf_dev_head << "buf_dev_aligned:" << buf_dev_aligned << std::endl;

    for (size_t i = 0; i < options.nthreads; i++)
    {
        CUfileBatchHandle_t batch_idp = nullptr;
        checkCuFileErrors(cuFileBatchIOSetUp(&batch_idp, std::min<size_t>(options.io_multiplicity, props.max_batch_io_size)));

        std::vector<void *> buf_dev_vec(options.io_multiplicity);
        std::vector<void *> buf_dict_dev_vec(npages_for_dicts);

        std::vector<CUfileIOParams_t> batch_params_vec(options.io_multiplicity);

        std::vector<CUfileIOEvents_t> batch_events_vec(options.io_multiplicity);

        std::vector<ScanDeviceBatchSyncCookie> batch_cookie_vec(options.io_multiplicity);

        void **arr_buf_dev = arr_buf_dev_base + i * options.io_multiplicity;
        void **arr_buf_host = arr_buf_host_base + i * options.io_multiplicity;
        //std::cout << "arr_buf_dev_base:" << arr_buf_dev_base << std::endl;
        //std::cout << "arr_buf_host_base:" << arr_buf_host_base << std::endl;

        void **arr_buf_dict_dev = arr_buf_dict_dev_base + i * npages_for_dicts;
        void **arr_buf_dict_host = arr_buf_dict_host_base + i * npages_for_dicts;
        // exit(0);

        for (size_t j = 0; j < options.io_multiplicity; j++)
        {
            //void *buf_dev = (void*)mb_cuda_alloc_v2(options.io_size);
            size_t offset = (i * options.io_multiplicity + j) * options.io_size;
            void *buf_dev = (void *)(
                &reinterpret_cast<uint8_t*>(buf_dev_aligned)[offset]);

            mb_cufile_buf_register(buf_dev, options.io_size);
            #if 0
            std::cout << "buf_dev:" << buf_dev << std::endl;
            #endif

            buf_dev_vec[j] = buf_dev;
        }
        for (size_t j = 0; j < npages_for_dicts; j++)
        {
            size_t offset = (i * npages_for_dicts + j) * options.io_size;
            void *buf_dict_dev = (void *)(
                &reinterpret_cast<uint8_t*>(buf_dict_dev_aligned)[offset]);
            #if 0
            std::cout << "buf_dict_dev:" << buf_dict_dev << std::endl;
            #endif
            buf_dict_dev_vec[j] = buf_dict_dev;
        }

        void *buf_out_dev = mb_cuda_alloc(4096);
        void *buf_out_host = mb_alloc(options.page_size);

        auto begin = page_vec.begin() + npages_per_thread * i;
        auto end = page_vec.begin() + std::min(npages_per_thread * (i + 1), npages);

        thread_args_vec.push_back(ScanWithDictDeviceBatchSyncThreadArgs{
            .thrid = i,
            .ctx = ctx,
            .page_size = options.page_size,
            .page_start_index = page_start_index_vec[i],
            .page_id_final = page_id_final,
            .metadata = metadata,
            .cufile_handles = cufile_handles,
            .batch_idp = batch_idp,
            .page_vec_begin = begin,
            .page_vec_end = end,
            .buf_dev_vec = buf_dev_vec,
            .buf_dict_dev_vec = buf_dict_dev_vec,
            .vec_vec_max_npages_required = vec_vec_max_npages_required,
            .buf_out_dev = buf_out_dev,
            .buf_out_host = buf_out_host,
            .arr_buf_dev = arr_buf_dev,
            .arr_buf_host = arr_buf_host,
            .arr_buf_dict_dev = arr_buf_dict_dev,
            .arr_buf_dict_host = arr_buf_dict_host,
            .batch_params_vec = batch_params_vec,
            .batch_events_vec = batch_events_vec,
            .batch_cookie_vec = batch_cookie_vec,
            //.nrecs_per_page_size = std::ref(nrecs_per_page_size),
            //.compressed_page_sizes_vec = compressed_page_sizes,
            //.compressed_subpage_sizes_vec = compressed_subpage_sizes,
            .period_sec = options.period_sec,
            .stats_nio = 0,
        });
    }

    
    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    for (size_t i = 0; i < options.nthreads; i++)
    {
        ScanWithDictDeviceBatchSyncThreadArgs &args = thread_args_vec[i];
        threads.emplace_back(
            [&options, &args, &max_nrecs_per_page_tbl, &max_nrecs_per_page_dct, &varchar_field_index, &fds]()
            {
                size_t i;
                cpu_set_affinity(args.thrid);
                mb_cuda_set_context(args.ctx);
                CUfileOpcode_t opcode = CUFILE_READ;

                // void **arr_buf_dev = args.arr_buf_dev;
                void **arr_buf_host = args.arr_buf_host;

                void **arr_buf_dict_dev = args.arr_buf_dict_dev;
                void **arr_buf_dict_host = args.arr_buf_dict_host;

                size_t nxtns = std::distance(args.xtn_vec_begin, args.xtn_vec_end);

                size_t io_multiplicity = options.io_multiplicity;
                /* same to num_varchar_clusters */
                size_t nstreams = 8;
                std::vector<cudaStream_t> streams;
                for (i = 0; i < nstreams; i++) {
                    auto stream = mb_cuda_stream_create();
                    streams.push_back(stream);
                }

                uint customer_scan_varchar_dict_attr = 0; //TPCH::Query::Check::Customer::Q1::SCAN_TARGET_COL;
                uint customer_varchar_dict_nattrs = 1;

                size_t done = 0;
                size_t ongoing = 0;
                size_t nios;
                uint64_t time_start = gettime();

                uint64_t kernel_exec_time = 0;
                struct TPCHTableMetadata &metadata = args.metadata;
                // auto &max_nrecs_per_page_dct = metadata.table_customer_max_nrows_in_dict;
                // std::vector<std::vector<uint64_t>> &vec_vec_max_npages_required = args.vec_vec_max_npages_required;

                std::vector<void**> vec_dct_page_buf_dev {};
                std::vector<uint64_t> vec_dct_npages_host {};
                std::vector<uint64_t> dct_pagid_vec {};
                std::vector<uint64_t> pagid_vec {};
                std::vector<void*> vec_dct_page_buf {};

                // DEBUG
                // #define DEBUG_CUDA_KERNEL
                #ifdef DEBUG_CUDA_KERNEL
                void *ptr;
                if (posix_memalign((void**)&ptr, 512,
                    options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                void *debug_buf = ptr;
                memset(debug_buf, 0, options.page_size);
                struct dbg_pag_head {
                    uint32_t nalloc; // number of allocated records
                    uint32_t watermark; // free space watermark
                    uint32_t lfreespace;
                } *dbg_pag_head;
                size_t dbg_nrecs_sum = 0;
                #endif

                // bool debug_should_stop;
                // void *buf_in_dev_head = args.buf_dev_vec[0];

                const size_t num_clusters = 8;
                // const size_t c_comment_idx = 2;
                const size_t c_varchar_idx = varchar_field_index;
                for (size_t i = 0; i < nxtns; i++) {
                    const uint64_t xtn_id = args.xtn_vec_begin[i];
                    /* a XTN for a table */
                    size_t npages_in_xtn = xtn_get_allocated_npages(metadata.xtn_head, xtn_id);

                    pagid_vec.clear();
                    dct_pagid_vec.clear();
                    vec_dct_page_buf_dev.clear();
                    vec_dct_npages_host.clear();
                    vec_dct_page_buf.clear();
                    /* When switching the XTN, then trying to fetch the related dictionary first*/
                    // uint64_t page_ids[DCT::kNumMaxVarCharFields][DCT::kNumMaxDictsInXtn]; // 8B * 8 * 3 = 64B * 3
                    // uint32_t npages[DCT::kNumMaxVarCharFields][DCT::kNumMaxDictsInXtn]; // 4B * 8 * 3 = 32B * 3
                    // TODO: scan based on the IO scheduler
                    auto &dct_pagids = metadata.dct_meta_head[xtn_id].page_ids;
                    auto &dct_npages = metadata.dct_meta_head[xtn_id].npages;
                    // size_t npages_sum = 0;

                    void **arr_buf_dict_dev_base = arr_buf_dict_dev;
                    void **arr_buf_dict_host_base = arr_buf_dict_host;
                    size_t npages_used = 0;
                    for (size_t j = 0; j < num_clusters; j++)
                    {
                        // std::vector<void*> vec_dct_buf {};
                        size_t npags = dct_npages[c_varchar_idx][j];
                        void **src_dev = arr_buf_dict_host_base + npages_used;
                        void **dst_dev = arr_buf_dict_dev_base + npages_used;;
                        if (max_nrecs_per_page_dct[c_varchar_idx][j] > 0) {
                            for (size_t k = 0; k < npags; k++) {
                                uint64_t pagid = dct_pagids[c_varchar_idx][j] + k;
                                //TODO: skip needless IO
                                //if (max_nrecs_per_page_dct[c_varchar_idx][k] == 0) {
                                //}
                                /* -- DEBUG code -- */
                                #ifdef DEBUG_CUDA_KERNEL
                                page_pread_host(fds, debug_buf, pagid, options.page_size);
                                dbg_pag_head = (struct dbg_pag_head *)debug_buf;
                                uint32_t nalloc = dbg_pag_head->nalloc;
                                if (pagid == 15579UL) {
                                    debug_should_stop = true;
                                }
                                dbg_nrecs_sum += nalloc;
                                #endif
                                /* -- DEBUG code -- */
                                dct_pagid_vec.push_back(pagid);
                                void *buf_dev = args.buf_dict_dev_vec[npages_used];
                                arr_buf_dict_host[npages_used] = buf_dev;
                                // vec_dct_page_buf.push_back(args.buf_dict_dev_vec[npages_used]);
                                vec_dct_page_buf.push_back(buf_dev);
                                npages_used++;
                                /* -- DEBUG code -- */
                                #ifdef DEBUG_CUDA_KERNEL
                                std::cout << "Prefetched dictionary page " << pagid << " for customer, cluster " << j << ", page " << k
                                    << ", nalloc:" << nalloc << std::endl;
                                #endif
                                /* -- DEBUG code -- */
                            }
                        }
                        // vec_dct_buf.push_back(buf_dev);
                        // vec_vec_dct_buf.push_back(vec_dct_buf);
                        vec_dct_npages_host.push_back(npags);

                        /* set dst_dev */
                        mb_cuda_memcpy_host_to_device_async(dst_dev, src_dev, npags * sizeof(void*), streams[j]);
                        vec_dct_page_buf_dev.push_back(dst_dev);
                    }
                    #ifdef DEBUG_CUDA_KERNEL
                    std::cout << dbg_nrecs_sum << std::endl;
                    #endif
                    // sleep(1);

                    //mb_cuda_memcpy_host_to_device_async(arr_buf_dev, arr_buf_host, num_clusters * sizeof(void*), streams[0]);

                    /* prefetch dcts */
                    size_t dctbufidx = 0;
                    ongoing = 0;
                    done = 0;
                    nios = dct_pagid_vec.size();
                    while (done < nios) {
                        if (ongoing < io_multiplicity && done + ongoing < nios)
                        {
                            // Send a batch
                            size_t j = 0;
                            for (; j < std::min(options.io_multiplicity - ongoing, nios - (done + ongoing)); j++)
                            {
                                CUfileIOParams_t *params = &args.batch_params_vec[j];
                                memset(params, 0, sizeof(CUfileIOParams_t));
                                params->mode = CUFILE_BATCH;
                                //void *buf_dev = args.buf_dict_dev_vec[dctbufidx];
                                void *buf_dev = vec_dct_page_buf[dctbufidx];
                                dctbufidx++;

                                params->u.batch.devPtr_base = buf_dev;
                                uint64_t pagid = dct_pagid_vec[done + ongoing + j];

                                uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                                uint64_t idev = pagid_to_idev(pagid, options.ndev);
                                // std::cout << "pagid=" << pagid << ", ipagid=" << ipagid << ", idev=" << idev << std::endl;
                                uint64_t file_offset = ipagid * options.io_size;
                                params->u.batch.file_offset = file_offset;
                                params->u.batch.devPtr_offset = 0;
                                params->u.batch.size = options.io_size;
                                params->fh = args.cufile_handles[idev];
                                params->opcode = opcode;
                                ScanDeviceBatchSyncCookie *cookie = &args.batch_cookie_vec[j];

                                cookie->buf_dev = buf_dev;
                                cookie->idx = done + ongoing + j;
                                cookie->pagid = pagid;
                                params->cookie = (void *)cookie;
                                args.stats_nio++;
                            }
                            // std::cout << "nFileBatchIOSubmit:" << j << std::endl;
                            checkCuFileErrors(cuFileBatchIOSubmit(args.batch_idp, j, &args.batch_params_vec[0], 0));
                            ongoing += j;
                            // std::cout << "[DEBUG][1](" << args.thrid << ") ongoing: " << ongoing << "done: " << done << std::endl;
                        }
                        else
                        {
                            // Wait for a batch
                            unsigned int nrequests = ongoing;
                            checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, nrequests, &nrequests, &args.batch_events_vec[0], nullptr));

                            size_t j = 0;
                            for (; j < nrequests; j++)
                            {
                                /* Just checcking errors here*/
                                CUfileIOEvents_t *events = &args.batch_events_vec[j];
                                if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                                {
                                    std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                                    std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                                    std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                                    std::cerr << "  events->cookie: " << events->cookie << std::endl;
                                    ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                                    uint64_t pagid = pagid_vec[cookie->idx];
                                    std::cerr << "  params: pagid=" << pagid
                                              << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                              << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                                    std::cerr << "  options.io_multiplicity=" << options.io_multiplicity << std::endl;
                                    exit(EXIT_FAILURE);
                                }
                                // ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                            }
                            ongoing -= j;
                            done += j;
                            // std::cout << "[DEBUG][2](" << args.thrid << ") ongoing: " << ongoing << "done: " << done << std::endl;
                        }
                    }

                    #if 0
                    for (size_t j = 0; j < npages_in_xtn; j++)
                    {
                        uint64_t pagid = xtn_calc_page_id(metadata.xtn_head, xtn_id, j);
                        pagid_vec.push_back(pagid);
                    }
                    //pag_buffers.reserve(io_multiplicity);
                    done = 0;
                    nios = pagid_vec.size();
                    while (done < nios)
                    {
                        if (ongoing < io_multiplicity && done + ongoing < nios)
                        {
                            // Send a batch
                            size_t j = 0;
                            for (; j < std::min(options.io_multiplicity - ongoing, nios - (done + ongoing)); j++)
                            {
                                CUfileIOParams_t *params = &args.batch_params_vec[j];
                                memset(params, 0, sizeof(CUfileIOParams_t));
                                params->mode = CUFILE_BATCH;
                                void *buf_dev = args.buf_dev_vec[j];

                                params->u.batch.devPtr_base = buf_dev;
                                uint64_t pagid = pagid_vec[done + ongoing + j];

                                uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                                uint64_t idev = pagid_to_idev(pagid, options.ndev);
                                // std::cout << "pagid=" << pagid << ", ipagid=" << ipagid << ", idev=" << idev << std::endl;
                                uint64_t file_offset = ipagid * options.io_size;
                                params->u.batch.file_offset = file_offset;
                                params->u.batch.devPtr_offset = 0;
                                params->u.batch.size = options.io_size;
                                params->fh = args.cufile_handles[idev];
                                params->opcode = opcode;
                                ScanDeviceBatchSyncCookie *cookie = &args.batch_cookie_vec[j];

                                cookie->buf_dev = buf_dev;
                                cookie->idx = done + ongoing + j;
                                cookie->pagid = pagid;
                                params->cookie = (void *)cookie;
                                args.stats_nio++;
                            }
                            // std::cout << "nFileBatchIOSubmit:" << j << std::endl;
                            checkCuFileErrors(cuFileBatchIOSubmit(args.batch_idp, j, &args.batch_params_vec[0], 0));
                            ongoing += j;
                        }
                        else
                        {
                            // Wait for a batch
                            unsigned int nrequests = ongoing;
                            checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, nrequests, &nrequests, &args.batch_events_vec[0], nullptr));

                            size_t j = 0;
                            for (; j < nrequests; j++)
                            {
                                CUfileIOEvents_t *events = &args.batch_events_vec[j];
                                if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                                {
                                    std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                                    std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                                    std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                                    std::cerr << "  events->cookie: " << events->cookie << std::endl;
                                    ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                                    uint64_t pagid = pagid_vec[cookie->idx];
                                    std::cerr << "  params: pagid=" << pagid
                                              << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                              << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                                    std::cerr << "  options.io_multiplicity=" << options.io_multiplicity << std::endl;
                                    exit(EXIT_FAILURE);
                                }
                                ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;

                                // size_t pagid = cookie->pagid;
                                void *buf_dev = cookie->buf_dev;
                                //pag_buffers.push_back(buf_dev);
                                arr_buf_host[j] = buf_dev;
                            }

                            //printPointerInfo(arr_buf_host, "src");
                            // exit(1);
                            #if 1
                            //for (size_t k = 0; k < nrequests; k++) {
                            //    mb_cuda_stream_synchronize(streams[k]);
                            //}
                            size_t npags = j;
                            //for (size_t i = 0; i < npags; i++) {
                            //    std::cout << "  pags[" << i << "]: " << std::hex << pags[i] << std::endl;
                            //}
                            // sleep(1);
                            for (size_t k = 0; k < nstreams; k++) {
                                /* pag: a pointer array of dict pages */
                                /* npags: size of array of the 1st argument */

                                if (max_nrecs_per_page_dct[c_varchar_idx][k]) {
                                    scan_customer_row(
                                            vec_dct_page_buf_dev[k],
                                            vec_dct_npages_host[k],
                                            customer_varchar_dict_nattrs,
                                            customer_scan_varchar_dict_attr,
                                            args.page_size,
                                            max_nrecs_per_page_dct[c_varchar_idx][k],
                                            reinterpret_cast<int64_t*>(args.buf_out_dev),
                                            streams[k]);
                                }

                                // std::cout << max_nrecs_per_page_dct[c_varchar_idx][k] << std::endl;
                            }
                            #endif

                            ongoing -= j;
                            done += j;

                            for (size_t k = 0; k < nrequests; k++) {
                                mb_cuda_stream_synchronize(streams[k]);
                            }
                        }
                        if (debug_should_stop)
                        {
                            std::cout << std::endl;
                            std::cerr << "[DEBUG] stopped." << std::endl;
                            sleep(3);
                            exit(1);
                        }
                    }
                    #else
                        #if 0
                        for (size_t k = 0; k < nstreams; k++) {
                            mb_cuda_stream_synchronize(streams[k]);
                        }
                        #endif

                        uint64_t kernel_exec_time_start = gettime();
                        for (size_t k = 0; k < nstreams; k++) {
                            /* pag: a pointer array of dict pages */
                            /* npags: size of array of the 1st argument */

                            if (max_nrecs_per_page_dct[c_varchar_idx][k]) {
                                scan_customer_row(
                                        vec_dct_page_buf_dev[k],
                                        vec_dct_npages_host[k],
                                        customer_varchar_dict_nattrs,
                                        customer_scan_varchar_dict_attr,
                                        args.page_size,
                                        max_nrecs_per_page_dct[c_varchar_idx][k],
                                        reinterpret_cast<int64_t*>(args.buf_out_dev),
                                        options.enable_prefetch,
                                        streams[k]);
                            }

                            // std::cout << max_nrecs_per_page_dct[c_varchar_idx][k] << std::endl;

                        }

                        for (size_t k = 0; k < nstreams; k++) {
                            mb_cuda_stream_synchronize(streams[k]);
                        }
                        uint64_t kernel_exec_time_end = gettime();
                        kernel_exec_time += (kernel_exec_time_end - kernel_exec_time_start);
                        #ifdef DEBUG_CUDA_KERNEL
                        if (debug_should_stop)
                        {
                            std::cout << std::endl;
                            std::cerr << "[DEBUG] stopped." << std::endl;
                            sleep(3);
                            exit(1);
                        }
                        #endif
                    #endif
                }
                // std::cout << "[DEBUG][3](" << args.thrid << ") ongoing: " << ongoing << "done: " << done << std::endl;
                assert(ongoing == 0);
                while (ongoing > 0)
                {
                    // Wait for a batch
                    unsigned int nrequests = ongoing;
                    checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, nrequests, &nrequests, &args.batch_events_vec[0], nullptr));

                    size_t j = 0;
                    for (; j < nrequests; j++)
                    {
                        CUfileIOEvents_t *events = &args.batch_events_vec[j];
                        if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                        {
                            std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                            std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                            std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                            std::cerr << "  events->cookie: " << events->cookie << std::endl;
                            ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                            uint64_t pagid = pagid_vec[cookie->idx];
                            std::cerr << "  params: pagid=" << pagid
                                      << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                      << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                            std::cerr << "  options.io_multiplicity=" << io_multiplicity << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                        void *buf_dev = cookie->buf_dev;
                        // pag_buffers.push_back(buf_dev);
                        arr_buf_host[j] = buf_dev;
                    }

                    #if 1
                    uint64_t kernel_exec_time_start = gettime();
                    for (size_t k = 0; k < nstreams; k++) {
                        //size_t dctidx = c_varchar_idx * num_clusters + k;
                        /* pag: a pointer array of dict pages */
                        /* npags: size of array of the 1st argument */
                        //if (max_nrecs_per_page_dct[c_varchar_idx][k]) {
                            scan_customer_row(
                                    vec_dct_page_buf_dev[k],
                                    vec_dct_npages_host[k],
                                    customer_varchar_dict_nattrs,
                                    customer_scan_varchar_dict_attr,
                                args.page_size,
                                max_nrecs_per_page_dct[c_varchar_idx][k],
                                reinterpret_cast<int64_t*>(args.buf_out_dev),
                                options.enable_prefetch,
                                streams[k]);
                        //}

                        std::cout << max_nrecs_per_page_dct[c_varchar_idx][k] << std::endl;
                    }
                    uint64_t kernel_exec_time_end = gettime();
                    kernel_exec_time += (kernel_exec_time_end - kernel_exec_time_start);
                    #endif
                    ongoing -= j;
                    done += j;

                    // mb_cuda_stream_synchronize(streams[0]);
                    for (size_t k = 0; k < nrequests; k++) {
                        mb_cuda_stream_synchronize(streams[k % nstreams]);
                    }
                }

                mb_cuda_memcpy_device_to_host_async(args.buf_out_host, args.buf_out_dev, sizeof(int64_t), streams[0]);
                mb_cuda_stream_synchronize(streams[0]);

                std::cout << "nlines_scanned:" << *reinterpret_cast<int64_t*>(args.buf_out_host) << std::endl;
                args.kernel_exec_time = kernel_exec_time;
                #ifdef DEBUG_CUDA_KERNEL
                std::cout << "dbg_nrecs_sum:" << dbg_nrecs_sum << std::endl;
                #endif
                //std::cout << "nsubpage_processed:" << nsubpage_processed << std::endl;
                //std::cout << "nlineitem_processed:" << nlineitem_processed << std::endl;
                
                for (i = 0; i < nstreams; i++) {
                    mb_cuda_stream_destroy(streams[i]);
                }
            });
    }

    for (size_t i = 0; i < options.nthreads; i++)
    {
        threads[i].join();
    }

    int64_t count = 0;
    uint64_t kernel_exec_time = 0;
    for (size_t i = 0; i < options.nthreads; i++) {
        int64_t r = *reinterpret_cast<int64_t*>(thread_args_vec[i].buf_out_host);
        count += r;
        kernel_exec_time = std::max(kernel_exec_time, thread_args_vec[i].kernel_exec_time);
    }
    std::cout << "count:" << count << std::endl;


    auto end = chrono::system_clock::now();
    auto end_cpu_usage = read_cpu_usage();

    std::cout << "kernel_exec_time:" << kernel_exec_time << std::endl;

    uint64_t naios_issued = 0;
    for (size_t i = 0; i < options.nthreads; i++)
    {
        ScanWithDictDeviceBatchSyncThreadArgs &args = thread_args_vec[i];
        naios_issued += thread_args_vec[i].stats_nio;

        (void)cuFileBatchIODestroy(args.batch_idp);

        for (size_t j = 0; j < options.io_multiplicity; j++)
        {
            void *buf_dev = args.buf_dev_vec[j];

            mb_cufile_buf_deregister(buf_dev);
            // mb_cuda_free(buf_dev);
        }
        free(args.buf_out_host);
        mb_cuda_free(args.buf_out_dev);
    }
    cudaHostUnregister(arr_buf_host_base);
    mb_cuda_host_free(arr_buf_host_base);
    mb_cuda_free(arr_buf_dev_base);
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

    free(metadata.dct_meta_head);
    free(metadata.xtn_head);

    //std::cout << "[DEBUG] start: "<< start << "end: " << end << std::endl;
    //std::cout << end - start << " sec" << std::endl;
    std::cout << "[DEBUG] diff:" << (end - start).count() << std::endl;

    return BenchmarkResult{
        .nios = naios_issued,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = get_gpu_usage(),
    };
}
#endif

BenchmarkResult tpch_scan_customer_device_batch_sync(BenchmarkOptions &options)
{
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

    open_files(options, fds);

    //std::vector<DeviceBatchSyncThreadArgs> thread_args_vec;
    //thread_args_vec.reserve(options.nthreads);
    xtn_set_constants(options.page_size);

    struct TPCHTableMetadata *metadatap = static_cast<TPCHTableMetadata*>(mb_alloc(options.page_size));
    auto &metadata = *metadatap;
    mb_pread(fds[0], metadatap, options.page_size, 0);

    std::cout << "page_size:                        " << metadata.page_size << std::endl;
    std::cout << "compressed:                       " << metadata.compressed << std::endl;
    std::cout << "column:                           " << metadata.column << std::endl;
    std::cout << "dict_encoded:                     " << metadata.dict_encoded << std::endl;
    std::cout << "table_customer_start_xtn_ids:     " << metadata.table_customer_start_xtn_ids[0] << std::endl;
    std::cout << "table_customer_nxtn:              " << metadata.table_customer_nrows   << std::endl;
    std::cout << "table_customer_nxtns:             " << metadata.table_customer_nxtns[0] << std::endl;
    std::cout << "table_customer_max_nrows_in_page: " << metadata.table_customer_max_nrows_in_page[0] << std::endl;
    std::cout << "table_nation_start_xtn_ids:       " << metadata.table_nation_start_xtn_ids[0] << std::endl;
    std::cout << "table_nation_nrows:               " << metadata.table_nation_nrows << std::endl;
    std::cout << "table_nation_nxtns:               " << metadata.table_nation_nxtns[0] << std::endl;
    std::cout << "table_lineitem_start_page_ids:    " << metadata.table_lineitem_start_xtn_ids[0] << std::endl;
    std::cout << "table_lineitem_nrows:             " << metadata.table_lineitem_nrows << std::endl;
    std::cout << "table_lineitem_nxtns:             " << metadata.table_lineitem_nxtns[0] << std::endl;
    std::cout << "free_xtn_id:                      " << metadata.free_xtn_id << std::endl;
    std::cout << "sizeof(CUfileIOEvents_t)          " << sizeof(CUfileIOEvents_t) << std::endl;
    std::cout << "sizeof(CUfileIOParams_t)          " << sizeof(CUfileIOParams_t) << std::endl;
    std::cout << "sizeof(DeviceBatchSyncCookie)     " << sizeof(ScanDeviceBatchSyncCookie) << std::endl;

    void *ptr;
    size_t num_super_nxtns = xtn_get_super_nxtn();
    if (posix_memalign((void**)&ptr, 512,
        options.page_size * XTN::NumPagesForSuperXTNs) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    metadata.xtn_head = new(ptr) struct xtn_entry[XTN::MaxNumXTNs];
    memset(metadata.xtn_head, 0, sizeof(struct xtn_entry) * XTN::MaxNumXTNs);

    size_t num_dct_meta_nxtns = xtn_get_dct_meta_nxtn();
    // Now, XTN::NumPagesForDctMetaXTNs is removed.
    exit(1);
#if 0
    if (posix_memalign((void**)&ptr, 512,
        options.page_size * XTN::NumPagesForDctMetaXTNs) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
#endif
    metadata.dct_meta_head = new(ptr) struct dct_meta_entry[XTN::MaxNumXTNs];
    memset(metadata.dct_meta_head, 0, sizeof(struct dct_meta_entry) * XTN::MaxNumXTNs);

    {
        size_t super_xtn_id = xtn_get_super_xtnid();
        uint64_t base_page_id = xtn_calc_page_id_from_xtn_id(super_xtn_id);
        char *ptr_base = reinterpret_cast<char*>(metadata.xtn_head);
        for (size_t i = 0; i < XTN::NumPagesForSuperXTNs; ++i) {
            page_pread_host(fds, (void*)&ptr_base[options.page_size * i], base_page_id + i, options.page_size);
        }
    }
    {
        size_t dct_meta_xtn_id = xtn_get_dct_meta_xtnid();
        uint64_t base_page_id = xtn_calc_page_id_from_xtn_id(dct_meta_xtn_id);
        char *ptr_base = reinterpret_cast<char*>(metadata.dct_meta_head);
        //for (size_t i = 0; i < XTN::NumPagesForDctMetaXTNs; ++i) {
        // NOTE: Now, XTN::NumPagesForDctMetaXTNs is removed.
        exit(1);
        for (size_t i = 0; i < 1; ++i) {
            page_pread_host(fds, (void*)&ptr_base[options.page_size * i], base_page_id + i, options.page_size);
        }
    }

    if (metadata.column) {
        auto result = __tpch_scan_customer_device_batch_sync_column(options, metadata, fds);
        free(metadatap);
        return result;
    }

    if (metadata.dict_encoded) {
        auto result = __tpch_scan_customer_device_batch_sync_dict(options, metadata, fds);
        free(metadatap);
        return result;
    }

    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(ctx);


    //Table<size_t> table_compressed_page_sizes = table_lineitem.compressed_page_sizes_table().value();
    //Table<size_t> table_compressed_sub_page_sizes_table = table_lineitem.compressed_sub_page_sizes_table().value();

    //std::cout << "=== lineitem table stats ===" << std::endl;
    //std::cout << "lrec:             " << table_lineitem.get_rec_size() << std::endl;

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

    //size_t nsubpages_per_page = options.page_size / options.sub_page_size;

    auto table_customer = TPCHTable(
        metadata.page_size,
        metadata.table_customer_start_xtn_ids[0],
        metadata.table_customer_nxtns[0],
        metadata.table_customer_nrows,
        metadata.compressed);

    std::vector<uint64_t> xtn_vec = table_customer.generate_xtn_ids();
    size_t xtn_id_final = xtn_vec[xtn_vec.size() - 1];
    auto xtn_start_index_vec = chunk_vector_start_indexes(xtn_vec, options.nthreads);

    std::cout << "=== table_customer start indexes ===" << std::endl;
    for (auto index : xtn_start_index_vec) {
        std::cout << index << std::endl;
    }

    #if 0
    std::vector<uint64_t> xtn_compressed_page_sizes = table_compressed_page_sizes.generate_page_ids();
    std::vector<uint64_t> compressed_page_sizes;
    std::cout << "=== table_compressed_page_sizes ===" << std::endl;
    std::cout << "table_compressed_page_sizes" << xtn_compressed_page_sizes.size() << std::endl;
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
    #endif
    constexpr size_t size_warp = 32;
    uint32_t max_nrecs_per_page_tbl;
    {
        size_t n = metadata.table_customer_max_nrows_in_page[0];
        max_nrecs_per_page_tbl = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
    }

    #if 0
    for (size_t i = 0; i < TPCH::common::kCustomerMaxNClustersInDCT + 1; i++) {
        if (i == 0) {
            size_t n = metadata.table_customer_max_nrows_in_page[0];
            nrecs_per_page_size[i] = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
        } else if (metadata.dict_encoded) {
            size_t n = metadata.table_customer_max_nrows_in_dict[i - 1];
            nrecs_per_page_size[i] = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
        } else {
            break;
        }
    }
    #endif

    mb_cufile_driver_open();

    cufile_handles.reserve(fds.size());
    for (auto fd : fds)
    {
        CUfileHandle_t cufile_handle = mb_cufile_handle_register(fd);
        cufile_handles.push_back(cufile_handle);
    }

    size_t nxtns = xtn_vec.size();
    size_t nxtns_per_thread = (nxtns + options.nthreads - 1) / options.nthreads;

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<ScanDeviceBatchSyncThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    /* Check io_depth value does not exceed its max value */
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);
    if (options.io_multiplicity > props.max_batch_io_size)
    {
        for (auto cufile_handle : cufile_handles)
        {
            mb_cufile_handle_deregister(cufile_handle);
        }
        mb_cufile_driver_close();
        close_files(options, fds);
        exit(EXIT_FAILURE);
    }

    /* for alignment */
    void *buf_dev_head = mb_cuda_alloc(options.io_size * (options.io_multiplicity * options.nthreads + 1));
    void *buf_dev_aligned = (void *)(((size_t)buf_dev_head + options.page_size - 1) & ~(options.page_size - 1));

    void **arr_buf_dev_base = reinterpret_cast<void**>(mb_cuda_alloc(sizeof(void*) * options.nthreads * options.io_multiplicity));
    void **arr_buf_host_base = reinterpret_cast<void**>(mb_cuda_host_alloc(sizeof(void*) * options.nthreads * options.io_multiplicity));
    cudaHostRegister(arr_buf_host_base, sizeof(void*) * options.nthreads * options.io_multiplicity, cudaHostRegisterDefault);

    // std::cout << "buf_dev_head:" << buf_dev_head << "buf_dev_aligned:" << buf_dev_aligned << std::endl;

    for (size_t i = 0; i < options.nthreads; i++)
    {
        CUfileBatchHandle_t batch_idp = nullptr;
        checkCuFileErrors(cuFileBatchIOSetUp(&batch_idp, std::min<size_t>(options.io_multiplicity, props.max_batch_io_size)));

        std::vector<void *> buf_dev_vec(options.io_multiplicity);

        std::vector<CUfileIOParams_t> batch_params_vec(options.io_multiplicity);

        std::vector<CUfileIOEvents_t> batch_events_vec(options.io_multiplicity);

        std::vector<ScanDeviceBatchSyncCookie> batch_cookie_vec(options.io_multiplicity);

        void **arr_buf_dev = arr_buf_dev_base + i * options.io_multiplicity;
        void **arr_buf_host = arr_buf_host_base + i * options.io_multiplicity;
        std::cout << "arr_buf_dev_base:" << arr_buf_dev_base << std::endl;
        std::cout << "arr_buf_host_base:" << arr_buf_host_base << std::endl;
        // exit(0);

        for (size_t j = 0; j < options.io_multiplicity; j++)
        {
            //void *buf_dev = (void*)mb_cuda_alloc_v2(options.io_size);
            size_t offset = (i * options.io_multiplicity + j) * options.io_size;
            void *buf_dev = (void *)(
                &reinterpret_cast<uint8_t*>(buf_dev_aligned)[offset]);

            mb_cufile_buf_register(buf_dev, options.io_size);
            #if 1
            std::cout << "buf_dev:" << buf_dev << std::endl;
            #endif

            buf_dev_vec[j] = buf_dev;
        }

        void *buf_out_dev = mb_cuda_alloc(4096);
        void *buf_out_host = mb_alloc(options.page_size);

        auto begin = xtn_vec.begin() + nxtns_per_thread * i;
        auto end = xtn_vec.begin() + std::min(nxtns_per_thread * (i + 1), nxtns);

        thread_args_vec.push_back(ScanDeviceBatchSyncThreadArgs{
            .thrid = i,
            .ctx = ctx,
            .page_size = options.page_size,
            .xtn_start_index = xtn_start_index_vec[i],
            .xtn_id_final = xtn_id_final,
            .metadata = metadata,
            .cufile_handles = cufile_handles,
            .batch_idp = batch_idp,
            .xtn_vec_begin = begin,
            .xtn_vec_end = end,
            .buf_dev_vec = buf_dev_vec,
            .buf_out_dev = buf_out_dev,
            .buf_out_host = buf_out_host,
            .arr_buf_dev = arr_buf_dev,
            .arr_buf_host = arr_buf_host,
            .batch_params_vec = batch_params_vec,
            .batch_events_vec = batch_events_vec,
            .batch_cookie_vec = batch_cookie_vec,
            //.nrecs_per_page_size = std::ref(nrecs_per_page_size),
            //.compressed_page_sizes_vec = compressed_page_sizes,
            //.compressed_subpage_sizes_vec = compressed_subpage_sizes,
            .period_sec = options.period_sec,
            .stats_nio = 0,
        });
    }

    
    auto start_cpu_usage = read_cpu_usage();
    auto start = chrono::system_clock::now();

    for (size_t i = 0; i < options.nthreads; i++)
    {
        ScanDeviceBatchSyncThreadArgs &args = thread_args_vec[i];
        threads.emplace_back(
            [&options, &args, &max_nrecs_per_page_tbl]()
            {
                size_t i;
                cpu_set_affinity(args.thrid);
                mb_cuda_set_context(args.ctx);
                CUfileOpcode_t opcode = CUFILE_READ;

                void **arr_buf_dev = args.arr_buf_dev;
                void **arr_buf_host = args.arr_buf_host;
                size_t nxtns = std::distance(args.xtn_vec_begin, args.xtn_vec_end);

                size_t io_multiplicity = options.io_multiplicity;
                size_t nstreams = options.io_multiplicity;
                std::vector<cudaStream_t> streams;
                for (i = 0; i < nstreams; i++) {
                    auto stream = mb_cuda_stream_create();
                    streams.push_back(stream);
                }

                uint customer_scan_attr;
                switch (options.query) {
                    case TPCH::QueryId::CHECK11:
                    {
                        customer_scan_attr = TPCH::Query::Check::Customer::CHECK11::SCAN_TARGET_COL_VARCHAR_IDX;
                        break;
                    }
                    case TPCH::QueryId::CHECK12:
                    {
                        customer_scan_attr = TPCH::Query::Check::Customer::CHECK12::SCAN_TARGET_COL_VARCHAR_IDX;
                        break;
                    }
                    case TPCH::QueryId::CHECK13:
                    {
                        customer_scan_attr = TPCH::Query::Check::Customer::CHECK13::SCAN_TARGET_COL_VARCHAR_IDX;
                        break;
                    }
                    default:
                    {
                        std::cerr << "Unsupported query id:" << static_cast<int>(options.query) << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                uint customer_nattrs = TPCH::common::kCustomerFieldCount;

                size_t done = 0;
                size_t ongoing = 0;
                size_t nios;
                uint64_t time_start = gettime();
                uint64_t kernel_exec_time = 0;
                struct TPCHTableMetadata &metadata = args.metadata;

                std::vector<uint64_t> pagid_vec {};

                void *buf_in_dev_head = args.buf_dev_vec[0];
                for (size_t i = 0; i < nxtns; i++) {
                    uint64_t xtn_id = args.xtn_vec_begin[i];
                    size_t npages_in_xtn = xtn_get_allocated_npages(metadata.xtn_head, xtn_id);

                    for (size_t j = 0; j < npages_in_xtn; j++)
                    {
                        uint64_t pagid = xtn_calc_page_id(metadata.xtn_head, xtn_id, j);
                        pagid_vec.push_back(pagid);
                    }
                }

                //pag_buffers.reserve(io_multiplicity);
                nios = pagid_vec.size();
                while (done < nios)
                {
                    if (ongoing < io_multiplicity && done + ongoing < nios)
                    {
                        // Send a batch
                        size_t j = 0;
                        for (; j < std::min(options.io_multiplicity - ongoing, nios - (done + ongoing)); j++)
                        {
                            CUfileIOParams_t *params = &args.batch_params_vec[j];
                            memset(params, 0, sizeof(CUfileIOParams_t));
                            params->mode = CUFILE_BATCH;
                            void *buf_dev = args.buf_dev_vec[j];
                            // if (buf_in_dev_head == nullptr) {
                            //     buf_in_dev_head = buf_dev;
                            // } else {
                            //     buf_in_dev_head = std::min(buf_dev, buf_in_dev_head);
                            // }

                            params->u.batch.devPtr_base = buf_dev;
                            uint64_t pagid = pagid_vec[done + ongoing + j];
                            //metadata.
                            //uint64_t pagid = args.pagid_vec_begin[done + ongoing + j];
                            //uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                            //uint64_t idev = pagid_to_idev(pagid, options.ndev);

                            //uint64_t pagid = 3;
                            uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                            uint64_t idev = pagid_to_idev(pagid, options.ndev);
                            // std::cout << "pagid=" << pagid << ", ipagid=" << ipagid << ", idev=" << idev << std::endl;
                            uint64_t file_offset = ipagid * options.io_size;
                            params->u.batch.file_offset = file_offset;
                            params->u.batch.devPtr_offset = 0;
                            params->u.batch.size = options.io_size;
                            params->fh = args.cufile_handles[idev];
                            params->opcode = opcode;
                            ScanDeviceBatchSyncCookie *cookie = &args.batch_cookie_vec[j];

                            cookie->buf_dev = buf_dev;
                            cookie->idx = done + ongoing + j;
                            cookie->pagid = pagid;
                            params->cookie = (void *)cookie;
                            args.stats_nio++;
                        }
                        // std::cout << "nFileBatchIOSubmit:" << j << std::endl;
                        checkCuFileErrors(cuFileBatchIOSubmit(args.batch_idp, j, &args.batch_params_vec[0], 0));
                        ongoing += j;
                    }
                    else
                    {
                        // Wait for a batch
                        unsigned int nrequests = ongoing;
                        checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, nrequests, &nrequests, &args.batch_events_vec[0], nullptr));

                        size_t j = 0;
                        for (; j < nrequests; j++)
                        {
                            CUfileIOEvents_t *events = &args.batch_events_vec[j];
                            if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                            {
                                std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                                std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                                std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                                std::cerr << "  events->cookie: " << events->cookie << std::endl;
                                ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                                uint64_t pagid = pagid_vec[cookie->idx];
                                std::cerr << "  params: pagid=" << pagid
                                          << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                          << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                                std::cerr << "  options.io_multiplicity=" << options.io_multiplicity << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;

                            // size_t pagid = cookie->pagid;
                            void *buf_dev = cookie->buf_dev;
                            //pag_buffers.push_back(buf_dev);
                            arr_buf_host[j] = buf_dev;
                        }

                        //printPointerInfo(arr_buf_host, "src");
                        // exit(1);
                        #if 1
                        mb_cuda_memcpy_host_to_device_async(arr_buf_dev, arr_buf_host, j * sizeof(void*), streams[0]);
                        //void **pags = pag_buffers.data();
                        size_t npags = j;
                        //for (size_t i = 0; i < npags; i++) {
                        //    std::cout << "  pags[" << i << "]: " << std::hex << pags[i] << std::endl;
                        //}

                        uint64_t kernel_exec_time_start = gettime();
                        scan_customer_row(
                                arr_buf_dev,
                                npags,
                                customer_nattrs,
                                customer_scan_attr,
                                args.page_size,
                                max_nrecs_per_page_tbl,
                                reinterpret_cast<int64_t*>(args.buf_out_dev),
                                options.enable_prefetch,
                                streams[0]);
                        uint64_t kernel_exec_time_end = gettime();
                        kernel_exec_time += (kernel_exec_time_end - kernel_exec_time_start);
                        #endif

                        ongoing -= j;
                        done += j;

                        //for (size_t k = 0; k < nrequests; k++) {
                        //    mb_cuda_stream_synchronize(streams[k % nstreams]);
                        //}
                        mb_cuda_stream_synchronize(streams[0]);
                    }
                }
                while (ongoing > 0)
                {
                    // Wait for a batch
                    unsigned int nrequests = ongoing;
                    checkCuFileErrors(cuFileBatchIOGetStatus(args.batch_idp, nrequests, &nrequests, &args.batch_events_vec[0], nullptr));

                    size_t j = 0;
                    for (; j < nrequests; j++)
                    {
                        CUfileIOEvents_t *events = &args.batch_events_vec[j];
                        if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                        {
                            std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                            std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                            std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                            std::cerr << "  events->cookie: " << events->cookie << std::endl;
                            ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                            uint64_t pagid = pagid_vec[cookie->idx];
                            std::cerr << "  params: pagid=" << pagid
                                      << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                      << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                            std::cerr << "  options.io_multiplicity=" << io_multiplicity << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        ScanDeviceBatchSyncCookie *cookie = (ScanDeviceBatchSyncCookie *)events->cookie;
                        void *buf_dev = cookie->buf_dev;
                        // pag_buffers.push_back(buf_dev);
                        arr_buf_host[j] = buf_dev;
                    }

                    #if 1
                    mb_cuda_memcpy_host_to_device_async(arr_buf_dev, arr_buf_host, j * sizeof(void*), streams[0]);
                    size_t npags = j;
                    uint64_t kernel_exec_time_start = gettime();
                    scan_customer_row(
                            arr_buf_dev,
                            npags,
                            customer_nattrs,
                            customer_scan_attr,
                            args.page_size,
                            max_nrecs_per_page_tbl,
                            reinterpret_cast<int64_t*>(args.buf_out_dev),
                            options.enable_prefetch,
                            streams[0]);
                    #endif
                    ongoing -= j;
                    done += j;

                    mb_cuda_stream_synchronize(streams[0]);

                    uint64_t kernel_exec_time_end = gettime();
                    kernel_exec_time += (kernel_exec_time_end - kernel_exec_time_start);
                }

                mb_cuda_memcpy_device_to_host_async(args.buf_out_host, args.buf_out_dev, sizeof(int64_t), streams[0]);
                mb_cuda_stream_synchronize(streams[0]);

                std::cout << "nlines_scanned:" << *reinterpret_cast<int64_t*>(args.buf_out_host) << std::endl;
                //std::cout << "nsubpage_processed:" << nsubpage_processed << std::endl;
                //std::cout << "nlineitem_processed:" << nlineitem_processed << std::endl;
                
                args.kernel_exec_time = kernel_exec_time;
                for (i = 0; i < nstreams; i++) {
                    mb_cuda_stream_destroy(streams[i]);
                }
            });
    }

    for (size_t i = 0; i < options.nthreads; i++)
    {
        threads[i].join();
    }

    int64_t count = 0;
    uint64_t kernel_exec_time = 0;
    for (size_t i = 0; i < options.nthreads; i++) {
        int64_t r = *reinterpret_cast<int64_t*>(thread_args_vec[i].buf_out_host);
        count += r;
        kernel_exec_time = std::max(kernel_exec_time, thread_args_vec[i].kernel_exec_time);
    }
    std::cout << "count:" << count << std::endl;
    std::cout << "kernel_exec_time:" << kernel_exec_time << std::endl;

    auto end = chrono::system_clock::now();
    auto end_cpu_usage = read_cpu_usage();

    uint64_t naios_issued = 0;
    for (size_t i = 0; i < options.nthreads; i++)
    {
        ScanDeviceBatchSyncThreadArgs &args = thread_args_vec[i];
        naios_issued += thread_args_vec[i].stats_nio;

        (void)cuFileBatchIODestroy(args.batch_idp);

        for (size_t j = 0; j < options.io_multiplicity; j++)
        {
            void *buf_dev = args.buf_dev_vec[j];

            mb_cufile_buf_deregister(buf_dev);
            // mb_cuda_free(buf_dev);
        }
        free(args.buf_out_host);
        mb_cuda_free(args.buf_out_dev);
    }
    cudaHostUnregister(arr_buf_host_base);
    mb_cuda_host_free(arr_buf_host_base);
    mb_cuda_free(arr_buf_dev_base);
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

    free(metadata.dct_meta_head);
    free(metadata.xtn_head);
    free(metadatap);

    //std::cout << "[DEBUG] start: "<< start << "end: " << end << std::endl;
    //std::cout << end - start << " sec" << std::endl;
    std::cout << "[DEBUG] diff:" << (end - start).count() << std::endl;

    return BenchmarkResult{
        .nios = naios_issued,
        .elapsed_nanoseconds = (end - start).count(),
        .cpu_usage = diff_cpu_usages(start_cpu_usage, end_cpu_usage),
        .gpu_usage = get_gpu_usage(),
    };
}
#else
BenchmarkResult tpch_scan_customer_device_batch_sync(BenchmarkOptions &options) {
    return BenchmarkResult{};
}
#endif