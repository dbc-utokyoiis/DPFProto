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
struct ScanDeviceMemCookie
{
    void *buf_dev;
    size_t idx;
    size_t pagid;
};

struct ScanDeviceMemThreadArgs
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
    void * buf_out_dev;
    void * buf_out_host;
    void ** arr_buf_dev;
    void ** arr_buf_host;
    std::vector<CUfileIOParams_t> batch_params_vec;
    std::vector<CUfileIOEvents_t> batch_events_vec;
    std::vector<ScanDeviceMemCookie> batch_cookie_vec;

    //std::vector<uint64_t> compressed_page_sizes_vec;
    //std::vector<uint64_t> compressed_subpage_sizes_vec;
    uint64_t kernel_exec_time = 0;
    uint64_t period_sec;
    size_t stats_nio;
};

struct ScanWithDictDeviceMemThreadArgs
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
    std::vector<ScanDeviceMemCookie> batch_cookie_vec;

    //std::vector<uint64_t> compressed_page_sizes_vec;
    //std::vector<uint64_t> compressed_subpage_sizes_vec;
    uint64_t kernel_exec_time = 0;
    uint64_t period_sec;
    size_t stats_nio;
};


#if 0
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
#endif

BenchmarkResult __tpch_scan_customer_device_mem_column(
    BenchmarkOptions &options, TPCHTableMetadata &metadata, std::vector<int> &fds)
{
    std::vector<CUfileHandle_t> cufile_handles;
    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(ctx);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);


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
        metadata.table_customer_start_xtn_ids[varchar_field_indexes[varchar_field_index]],
        metadata.table_customer_nxtns[varchar_field_indexes[varchar_field_index]],
        metadata.table_customer_nrows,
        metadata.compressed);

    std::vector<uint64_t> xtn_vec = table_customer.generate_xtn_ids();

    std::vector<uint64_t> pagid_vec {};
    size_t npages_sum = 0;
    for (auto xtn_id : xtn_vec) {
        size_t npages_in_xtn = xtn_get_allocated_npages(metadata.xtn_head, xtn_id);

        for (size_t i = 0; i < npages_in_xtn; i++)
        {
            uint64_t pagid = xtn_calc_page_id(metadata.xtn_head, xtn_id, i);
            pagid_vec.push_back(pagid);
        }
        npages_sum += npages_in_xtn;
    }

    if (deviceProp.totalGlobalMem < npages_sum * options.page_size) {
        std::cerr << "Insufficient device memory." << std::endl;
        std::cerr << "Required: " << npages_sum * options.page_size << " bytes" << std::endl;
        std::cerr << "Available: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto xtn_start_index_vec = chunk_vector_start_indexes(xtn_vec, options.nthreads);
    std::cout << "=== table_customer start indexes ===" << std::endl;
    for (auto index : xtn_start_index_vec) {
        std::cout << index << std::endl;
    }

    constexpr size_t size_warp = 32;
    uint32_t max_nrecs_per_page_tbl;
    {
        size_t n = metadata.table_customer_max_nrows_in_page[varchar_field_indexes[varchar_field_index]];
        max_nrecs_per_page_tbl = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
    }

    mb_cufile_driver_open();

    cufile_handles.reserve(fds.size());
    for (auto fd : fds)
    {
        CUfileHandle_t cufile_handle = mb_cufile_handle_register(fd);
        cufile_handles.push_back(cufile_handle);
    }

    size_t nxtns = xtn_vec.size();

    std::vector<std::thread> threads;
    threads.reserve(options.nthreads);

    std::vector<ScanDeviceMemThreadArgs> thread_args_vec;
    thread_args_vec.reserve(options.nthreads);

    /* Check io_depth value does not exceed its max value */
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);

    /* for alignment */
    void *buf_dev_head = mb_cuda_alloc(options.io_size * (npages_sum + 1));
    void *buf_dev_aligned = (void *)(((size_t)buf_dev_head + options.page_size - 1) & ~(options.page_size - 1));

    void **arr_buf_dev_base = reinterpret_cast<void**>(mb_cuda_alloc(sizeof(void*) * npages_sum));
    void **arr_buf_host_base = reinterpret_cast<void**>(mb_cuda_host_alloc(sizeof(void*) * npages_sum));
    // cudaHostRegister(arr_buf_host_base, sizeof(void*) * options.nthreads * options.io_multiplicity, cudaHostRegisterDefault);

    for (size_t i = 0; i < npages_sum; ++i)
    {
        //void *buf_dev = (void*)mb_cuda_alloc_v2(options.io_size);
        size_t offset = i * options.io_size;
        void *buf_dev = (void *)(
            &reinterpret_cast<uint8_t*>(buf_dev_aligned)[offset]);

        mb_cufile_buf_register(buf_dev, options.io_size);
        arr_buf_host_base[i] = buf_dev;
    }

    std::vector<cudaStream_t> streams;
    size_t nstreams = 8;
    for (size_t i = 0; i < nstreams; ++i) {
        auto stream = mb_cuda_stream_create();
        streams.push_back(stream);
    }
    mb_cuda_memcpy_host_to_device_async(arr_buf_dev_base, arr_buf_host_base, sizeof(void*) * npages_sum, streams[0]);

    size_t io_multiplicity = props.max_batch_io_size;
    CUfileBatchHandle_t batch_idp = nullptr;
    checkCuFileErrors(cuFileBatchIOSetUp(&batch_idp, io_multiplicity));

    std::vector<CUfileIOParams_t> batch_params_vec(io_multiplicity);
    std::vector<CUfileIOEvents_t> batch_events_vec(io_multiplicity);
    std::vector<ScanDeviceMemCookie> batch_cookie_vec(io_multiplicity);

    void *buf_out_dev = mb_cuda_alloc(4096);
    void *buf_out_host = mb_alloc(options.page_size);

    /* load data into storage */
    {
        size_t done = 0;
        size_t ongoing = 0;
        const size_t nios = npages_sum;
        CUfileOpcode_t opcode = CUFILE_READ;
        while (done < nios) {
            if (ongoing < io_multiplicity && done + ongoing < nios)
            {
                // Send a batch
                size_t i = 0;
                for (; i < std::min(io_multiplicity - ongoing, nios - (done + ongoing)); i++)
                {
                    CUfileIOParams_t *params = &batch_params_vec[i];
                    memset(params, 0, sizeof(CUfileIOParams_t));
                    params->mode = CUFILE_BATCH;
                    void *buf_dev = arr_buf_host_base[i + ongoing + done];
                    params->u.batch.devPtr_base = buf_dev;
                    uint64_t pagid = pagid_vec[done + ongoing + i];

                    uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                    uint64_t idev = pagid_to_idev(pagid, options.ndev);
                    uint64_t file_offset = ipagid * options.io_size;
                    params->u.batch.file_offset = file_offset;
                    params->u.batch.devPtr_offset = 0;
                    params->u.batch.size = options.io_size;
                    params->fh = cufile_handles[idev];
                    params->opcode = opcode;
                    ScanDeviceMemCookie *cookie = &batch_cookie_vec[i];

                    cookie->buf_dev = buf_dev;
                    cookie->idx = done + ongoing + i;
                    cookie->pagid = pagid;
                    params->cookie = (void *)cookie;
                }
                // std::cout << "nFileBatchIOSubmit:" << j << std::endl;
                checkCuFileErrors(cuFileBatchIOSubmit(batch_idp, i, &batch_params_vec[0], 0));
                ongoing += i;
            }
            else
            {
                // Wait for a batch
                unsigned int nrequests = ongoing;
                checkCuFileErrors(cuFileBatchIOGetStatus(batch_idp, nrequests, &nrequests, &batch_events_vec[0], nullptr));

                size_t i = 0;
                for (; i < nrequests; i++)
                {
                    CUfileIOEvents_t *events = &batch_events_vec[i];
                    if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                    {
                        std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                        std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                        std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                        std::cerr << "  events->cookie: " << events->cookie << std::endl;
                        ScanDeviceMemCookie *cookie = (ScanDeviceMemCookie *)events->cookie;
                        uint64_t pagid = pagid_vec[cookie->idx];
                        std::cerr << "  params: pagid=" << pagid
                                  << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                  << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                        std::cerr << "  options.io_multiplicity=" << options.io_multiplicity << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                ongoing -= i;
                done += i;
            }
        }
    }

    uint customer_scan_attr = 0;
    uint customer_nattrs = 1;
    

    #if 1
    uint64_t val = 0;
    size_t nrepeat = 32;

    std::vector<uint64_t> kernel_exec_times;
    for (size_t i = 0; i < nrepeat; ++i) {
        cudaMemset(buf_out_dev, 0, 4096);
        auto start_cpu_usage = read_cpu_usage();
        auto start = chrono::system_clock::now();
        uint64_t kernel_exec_time_start = gettime();
        scan_customer_row(
                arr_buf_dev_base,
                npages_sum,
                customer_nattrs,
                customer_scan_attr,
                options.page_size,
                max_nrecs_per_page_tbl,
                reinterpret_cast<int64_t*>(buf_out_dev),
                options.enable_prefetch,
                streams[0]);
        mb_cuda_stream_synchronize(streams[0]);
        uint64_t kernel_exec_time_end = gettime();
        #else
        std::vector<size_t> npages_kernel {};
        size_t npages_remainder = npages_sum % nstreams;
        for (size_t i = 0; i < nstreams; ++i) {
            npages_kernel.push_back(npages_sum / nstreams);
            if (i < npages_remainder) {
                npages_kernel[i]++;
            }
        }

        size_t offset = 0;
        auto start_cpu_usage = read_cpu_usage();
        auto start = chrono::system_clock::now();
        uint64_t kernel_exec_time_start = gettime();
        for (size_t i = 0; i < nstreams; ++i) {
            scan_customer_row(
                    &arr_buf_dev_base[offset],
                    npages_kernel[i],
                    customer_nattrs,
                    customer_scan_attr,
                    options.page_size,
                    max_nrecs_per_page_tbl,
                    reinterpret_cast<int64_t*>(buf_out_dev),
                    streams[i]);
            offset += npages_kernel[i];
        }
        for (size_t i = 0; i < nstreams; ++i) {
            mb_cuda_stream_synchronize(streams[i]);
        }
        uint64_t kernel_exec_time_end = gettime();
        #endif
        uint64_t kernel_exec_time = (kernel_exec_time_end - kernel_exec_time_start);

        mb_cuda_memcpy_device_to_host_async(buf_out_host, buf_out_dev, sizeof(int64_t) * 2, streams[0]);
        mb_cuda_stream_synchronize(streams[0]);
        auto end = chrono::system_clock::now();
        auto end_cpu_usage = read_cpu_usage();

        int64_t checksum = 0;
        int64_t count = 0;
        {
            int64_t r = *reinterpret_cast<int64_t*>(buf_out_host);
            count += r;

            int64_t r2 = *(reinterpret_cast<int64_t*>(buf_out_host) + 1);
            checksum += r2;
        }
        std::cout << "count(" << i << "):" << count << std::endl;
        std::cout << "checksum(" << i << "):" << checksum << std::endl;
        std::cout << "kernel_exec_time(" << i << "):" << kernel_exec_time << std::endl;
        kernel_exec_times.push_back(kernel_exec_time);
    }
    std::sort(kernel_exec_times.begin(), kernel_exec_times.end());
    uint64_t avg_kernel_exec_time = 0;
    avg_kernel_exec_time = std::accumulate(kernel_exec_times.begin(), kernel_exec_times.end() - 1, uint64_t(0));
    avg_kernel_exec_time /= (kernel_exec_times.size() - 1);
    std::cout << "avg_kernel_exec_time:" << avg_kernel_exec_time << std::endl;


    for (size_t i = 0; i < nstreams; i++) {
        mb_cuda_stream_destroy(streams[i]);
    }
    (void)cuFileBatchIODestroy(batch_idp);
    for (size_t i = 0; i < npages_sum; i++)
    {
        void *buf_dev = arr_buf_host_base[i];
        mb_cufile_buf_deregister(buf_dev);
    }
 
    // free(args.buf_out_host);
    // mb_cuda_free(args.buf_out_dev);
    //cudaHostUnregister(arr_buf_host_base);
    mb_cuda_host_free(arr_buf_host_base);
    mb_cuda_free(arr_buf_dev_base);
    mb_cuda_free(buf_dev_head);

    for (auto cufile_handle : cufile_handles)
    {
        mb_cufile_handle_deregister(cufile_handle);
    }

    mb_cufile_driver_close();

    close_files(options, fds);

    free(metadata.dct_meta_head);
    free(metadata.xtn_head);

    //std::cout << "[DEBUG] start: "<< start << "end: " << end << std::endl;
    //std::cout << end - start << " sec" << std::endl;
    //std::cout << "[DEBUG] diff:" << (end - start).count() << std::endl;

    return BenchmarkResult{
        .nios = 0,
        .elapsed_nanoseconds = 0,
        .cpu_usage = 0,
        .gpu_usage = get_gpu_usage(),
    };
}


BenchmarkResult __tpch_scan_customer_device_mem_dict(
    BenchmarkOptions &options, TPCHTableMetadata &metadata, std::vector<int> &fds)
{
    std::vector<CUfileHandle_t> cufile_handles;

    mb_cuda_init();
    auto device = mb_cuda_get_device(0);
    auto ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(ctx);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

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

    // auto &varchar_field_indexes = TPCH::common::fmt.customer_varchar_field_indexes;
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

    size_t c_varchar_idx = varchar_field_index;
    constexpr size_t size_warp = 32;
    // uint32_t max_nrecs_per_page_tbl;
    // {
    //     size_t n = metadata.table_customer_max_nrows_in_page[varchar_field_indexes[varchar_field_index]];
    //     max_nrecs_per_page_tbl = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
    // }

    std::array<std::array<uint32_t, DCT::kNumMaxDictsInXtn>, TPCH::common::fmt.customer_varchar_field_count> max_nrecs_per_page_dct{};
    if (metadata.dict_encoded) {
        for (size_t i = 0; i < TPCH::common::fmt.customer_varchar_field_count; ++i) {
            for (size_t j = 0; j < DCT::kNumMaxDictsInXtn; ++j) {
                size_t n = metadata.table_customer_max_nrows_in_dict[i][j];
                std::cout << "table_customer_max_nrows_in_dict[" << i << "][" << j << "]: " << n << std::endl;
                max_nrecs_per_page_dct[i][j] = n % size_warp == 0 ? n : (n / size_warp + 1) * size_warp;
            }
        }
    }

    /* calculate dict size first */
    // size_t customer_num_varchar_fields = TPCH::common::fmt.customer_varchar_field_count;
    // auto &customer_varchar_field_indexes = TPCH::common::fmt.customer_varchar_field_indexes;
    // size_t npages_for_dicts = 0;

    const size_t num_varchar_clusters = metadata.num_varchar_clusters;
    std::vector<std::vector<size_t>> vec_vec_pagids_for_dicts {};
    size_t npages_for_dicts_sum = 0;
    {
        vec_vec_pagids_for_dicts.reserve(num_varchar_clusters);
        for (size_t i = 0; i < num_varchar_clusters; i++) {
            vec_vec_pagids_for_dicts.push_back(std::vector<size_t>{});
        }

        size_t customer_start_xtn_id = metadata.table_customer_start_xtn_ids[0];
        size_t customer_nxtns = metadata.table_customer_nxtns[0];
        for (size_t i = 0; i < customer_nxtns; i++) {
            const uint64_t xtn_id = customer_start_xtn_id + i;
            struct dct_meta_entry *dmp = &metadata.dct_meta_head[xtn_id];
            for (size_t j = 0; j < num_varchar_clusters; j++) {
                //if (max_nrecs_per_page_dct[c_varchar_idx][j] > 0) {
                    size_t n = dmp->npages[c_varchar_idx][j];
                    size_t start_pagid = dmp->page_ids[c_varchar_idx][j];
                    npages_for_dicts_sum += n;
                    // std::cout << "xtn_id:" << xtn_id << "-->" << start_pagid << ", " << n << std::endl;

                    for (size_t k = 0; k < n; k++) {
                        vec_vec_pagids_for_dicts[j].push_back(start_pagid + k);
                    }
                    // std::cout << "xtn_id:" << xtn_id << "["<< j << "]-->" << start_pagid << ", " << n << std::endl;
                //}
            }
        }
    }
    std::cout << npages_for_dicts_sum << std::endl;

    if (deviceProp.totalGlobalMem < npages_for_dicts_sum * options.page_size) {
        std::cerr << "Insufficient device memory." << std::endl;
        std::cerr << "Required: " << npages_for_dicts_sum * options.page_size << " bytes" << std::endl;
        std::cerr << "Available: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < vec_vec_pagids_for_dicts.size(); i++) {
        std::cout << "vec_vec_pagids_for_dicts[" << i << "]:" << vec_vec_pagids_for_dicts[i].size() << std::endl;
    }

    mb_cufile_driver_open();

    cufile_handles.reserve(fds.size());
    for (auto fd : fds)
    {
        CUfileHandle_t cufile_handle = mb_cufile_handle_register(fd);
        cufile_handles.push_back(cufile_handle);
    }

    size_t nxtns = xtn_vec.size();
    // size_t nxtns_per_thread = (nxtns + options.nthreads - 1) / options.nthreads;

    /* Check io_depth value does not exceed its max value */
    CUfileDrvProps_t props;
    cuFileDriverGetProperties(&props);
    size_t io_multiplicity = props.max_batch_io_size;

    /* prepare CUDA streams */
    size_t nstreams = num_varchar_clusters;
    std::vector<cudaStream_t> streams{};
    for (size_t i = 0; i < nstreams; i++) {
        auto stream = mb_cuda_stream_create();
        streams.push_back(stream);
    }

     /* for dict */
    void *buf_dict_dev_head = mb_cuda_alloc(options.io_size * (npages_for_dicts_sum + 1));
    void *buf_dict_dev_aligned = (void *)(((size_t)buf_dict_dev_head + options.page_size - 1) & ~(options.page_size - 1));

    void **arr_buf_dict_dev_base = reinterpret_cast<void**>(mb_cuda_alloc(sizeof(void*) * npages_for_dicts_sum));
    void **arr_buf_dict_host_base = reinterpret_cast<void**>(mb_cuda_host_alloc(sizeof(void*) * npages_for_dicts_sum));
    //cudaHostRegister(arr_buf_dict_host_base, sizeof(void*) * npages_for_dicts_sum, cudaHostRegisterDefault);

    for (size_t i = 0; i < npages_for_dicts_sum; ++i)
    {
        size_t offset = i * options.io_size;
        void *buf_dev = (void *)(
            &reinterpret_cast<uint8_t*>(buf_dict_dev_aligned)[offset]);

        mb_cufile_buf_register(buf_dev, options.io_size);
        arr_buf_dict_host_base[i] = buf_dev;
    }

    mb_cuda_memcpy_host_to_device_async(arr_buf_dict_dev_base, arr_buf_dict_host_base, sizeof(void*) * npages_for_dicts_sum, streams[0]);
    mb_cuda_stream_synchronize(streams[0]);

    CUfileBatchHandle_t batch_idp = nullptr;
    checkCuFileErrors(cuFileBatchIOSetUp(&batch_idp, io_multiplicity));

    std::vector<CUfileIOParams_t> batch_params_vec(io_multiplicity);
    std::vector<CUfileIOEvents_t> batch_events_vec(io_multiplicity);
    std::vector<ScanDeviceMemCookie> batch_cookie_vec(io_multiplicity);

    void *buf_out_dev = mb_cuda_alloc(4096);
    void *buf_out_host = mb_alloc(options.page_size);

    std::vector<size_t> npages_for_dict_clusters {};
    {
        CUfileOpcode_t opcode = CUFILE_READ;
        size_t base = 0;

        for (size_t clusterid = 0; clusterid < num_varchar_clusters; clusterid++) {
            auto &vec_pagids_for_dicts = vec_vec_pagids_for_dicts[clusterid];
            size_t done = 0;
            size_t ongoing = 0;
            size_t nios = vec_pagids_for_dicts.size();
            while (done < nios) {
                if (ongoing < io_multiplicity && done + ongoing < nios)
                {
                    // Send a batch
                    size_t i = 0;
                    for (; i < std::min(io_multiplicity - ongoing, nios - (done + ongoing)); i++)
                    {

                        CUfileIOParams_t *params = &batch_params_vec[i];
                        memset(params, 0, sizeof(CUfileIOParams_t));
                        params->mode = CUFILE_BATCH;
                        void *buf_dev = arr_buf_dict_host_base[i + ongoing + done + base];
                        params->u.batch.devPtr_base = buf_dev;
                        uint64_t pagid = vec_pagids_for_dicts[done + ongoing + i];

                        uint64_t ipagid = pagid_to_ipagid(pagid, options.ndev);
                        uint64_t idev = pagid_to_idev(pagid, options.ndev);
                        uint64_t file_offset = ipagid * options.io_size;
                        params->u.batch.file_offset = file_offset;
                        params->u.batch.devPtr_offset = 0;
                        params->u.batch.size = options.io_size;
                        params->fh = cufile_handles[idev];
                        params->opcode = opcode;
                        ScanDeviceMemCookie *cookie = &batch_cookie_vec[i];

                        cookie->buf_dev = buf_dev;
                        cookie->idx = done + ongoing + i;
                        cookie->pagid = pagid;
                        params->cookie = (void *)cookie;
                    }
                    // std::cout << "nFileBatchIOSubmit:" << j << std::endl;
                    checkCuFileErrors(cuFileBatchIOSubmit(batch_idp, i, &batch_params_vec[0], 0));
                    ongoing += i;
                }
                else
                {
                    // Wait for a batch
                    unsigned int nrequests = ongoing;
                    checkCuFileErrors(cuFileBatchIOGetStatus(batch_idp, nrequests, &nrequests, &batch_events_vec[0], nullptr));

                    size_t i = 0;
                    for (; i < nrequests; i++)
                    {
                        CUfileIOEvents_t *events = &batch_events_vec[i];
                        if (events->ret != options.io_size || events->status != CUFILE_COMPLETE)
                        {
                            std::cerr << "  [FATAL] LINE: " << __LINE__ << std::endl;
                            std::cerr << "  events->ret: " << (ssize_t)events->ret << std::endl;
                            std::cerr << "  events->status: " << cufile_status_to_string(events->status) << std::endl;
                            std::cerr << "  events->cookie: " << events->cookie << std::endl;
                            ScanDeviceMemCookie *cookie = (ScanDeviceMemCookie *)events->cookie;
                            uint64_t pagid = vec_pagids_for_dicts[cookie->idx];
                            std::cerr << "  params: pagid=" << pagid
                                      << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                      << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                            std::cerr << "  options.io_multiplicity=" << options.io_multiplicity << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                    ongoing -= i;
                    done += i;
                }
            }
            base += nios;
            npages_for_dict_clusters.push_back(nios);
        }
    }
    
    uint customer_varchar_dict_nattrs = 1;
    uint customer_scan_varchar_dict_attr = 0;
    size_t nrepeat = 16;

    std::vector<uint64_t> kernel_exec_times;
    for (size_t i = 0; i < nrepeat; ++i) {
        cudaMemset(buf_out_dev, 0, 4096);
        auto start_cpu_usage = read_cpu_usage();
        auto start = chrono::system_clock::now();

        uint64_t kernel_exec_time_start = gettime();
        size_t offset = 0;
        for (size_t j = 0; j < num_varchar_clusters; j++) {
            /* pag: a pointer array of dict pages */
            /* npags: size of array of the 1st argument */
            if (max_nrecs_per_page_dct[c_varchar_idx][j]) {
                // std::cout << "offset: " << offset << std::endl;
                scan_customer_row(
                        &arr_buf_dict_dev_base[offset],
                        // vec_vec_pagids_for_dicts[i].size(),
                        npages_for_dict_clusters[j],
                        customer_varchar_dict_nattrs,
                        customer_scan_varchar_dict_attr,
                        options.page_size,
                        max_nrecs_per_page_dct[c_varchar_idx][j],
                        reinterpret_cast<int64_t*>(buf_out_dev),
                        options.enable_prefetch,
                        streams[j]);
                // std::cout << "npages: " << vec_vec_pagids_for_dicts[i].size() << ", "
                //     << npages_for_dict_clusters[i]
                //     << ", max_nrecs: " << max_nrecs_per_page_dct[c_varchar_idx][i] << std::endl;
            }
            // offset += vec_vec_pagids_for_dicts[i].size();
            offset += npages_for_dict_clusters[j];
        }
        for (size_t j = 0; j < num_varchar_clusters; j++) {
            if (max_nrecs_per_page_dct[c_varchar_idx][j]) {
                mb_cuda_stream_synchronize(streams[j]);
            }
        }
        uint64_t kernel_exec_time_end = gettime();
        uint64_t kernel_exec_time = (kernel_exec_time_end - kernel_exec_time_start);
 
        mb_cuda_memcpy_device_to_host_async(buf_out_host, buf_out_dev, sizeof(int64_t) * 2, streams[0]);
        mb_cuda_stream_synchronize(streams[0]);

        int64_t checksum = 0;
        int64_t count = 0;
        {
            int64_t r = *reinterpret_cast<int64_t*>(buf_out_host);
            count += r;

            int64_t r2 = *(reinterpret_cast<int64_t*>(buf_out_host) + 1);
            checksum += r2;
        }
 
        std::cout << "count(" << i << "):" << count << std::endl;
        std::cout << "checksum(" << i << "):" << checksum << std::endl;
        std::cout << "kernel_exec_time(" << i << "):" << kernel_exec_time << std::endl;
        kernel_exec_times.push_back(kernel_exec_time);

        auto end = chrono::system_clock::now();
        auto end_cpu_usage = read_cpu_usage();
    }
    std::sort(kernel_exec_times.begin(), kernel_exec_times.end());
    uint64_t avg_kernel_exec_time = 0;
    avg_kernel_exec_time = std::accumulate(kernel_exec_times.begin(), kernel_exec_times.end() - 1, uint64_t(0));
    avg_kernel_exec_time /= (kernel_exec_times.size() - 1);

    std::cout << "avg_kernel_exec_time:" << 
        ns_to_msec(avg_kernel_exec_time) <<
        "." << ns_to_sub_msec(avg_kernel_exec_time) <<
        "ms" << std::endl;
    std::cout << "nlines_scanned:" << *reinterpret_cast<int64_t*>(buf_out_host) << std::endl;
                
    for (size_t i = 0; i < nstreams; i++) {
        mb_cuda_stream_destroy(streams[i]);
    }

    uint64_t naios_issued = 0;
    (void)cuFileBatchIODestroy(batch_idp);

    // cudaHostUnregister(buf_out_host);
    free(buf_out_host);
    mb_cuda_free(buf_out_dev);

    for (size_t i = 0; i < npages_for_dicts_sum; i++)
    {
        void *buf_dev = arr_buf_dict_host_base[i];
        mb_cufile_buf_deregister(buf_dev);
    }
    mb_cuda_host_free(arr_buf_dict_host_base);
    mb_cuda_free(arr_buf_dict_dev_base);
    mb_cuda_free(buf_dict_dev_head);

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
    //std::cout << "[DEBUG] diff:" << (end - start).count() << std::endl;

    return BenchmarkResult{
        .nios = naios_issued,
        .elapsed_nanoseconds = 0,
        .cpu_usage = 0,
        .gpu_usage = get_gpu_usage(),
    };
}

BenchmarkResult tpch_scan_customer_device_mem(BenchmarkOptions &options)
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
    std::cout << "num_varchar_clusters:             " << metadata.num_varchar_clusters << std::endl;
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
    std::cout << "sizeof(DeviceBatchSyncCookie)     " << sizeof(ScanDeviceMemCookie) << std::endl;
    std::cout << "=== option ===                    " << std::endl;
    std::cout << "prefetch                          " <<
        (options.enable_prefetch ? "enabled" : "disabled") << std::endl;

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
#if 0
    if (posix_memalign((void**)&ptr, 512,
        options.page_size * XTN::NumPagesForDctMetaXTNs) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
#else
    /* XTN::NumPagesForDctMetaXTNs is removed */
    exit(1);
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
#if 0
        for (size_t i = 0; i < XTN::NumPagesForDctMetaXTNs; ++i) {
            page_pread_host(fds, (void*)&ptr_base[options.page_size * i], base_page_id + i, options.page_size);
        }
#else
        /* XTN::NumPagesForDctMetaXTNs is removed */
        exit(1);
#endif
    }

    if (metadata.column) {
        auto result = __tpch_scan_customer_device_mem_column(options, metadata, fds);
        free(metadatap);
        return result;
    }

    if (metadata.dict_encoded) {
        auto result = __tpch_scan_customer_device_mem_dict(options, metadata, fds);
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

    std::vector<ScanDeviceMemThreadArgs> thread_args_vec;
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
    // cudaHostRegister(arr_buf_host_base, sizeof(void*) * options.nthreads * options.io_multiplicity, cudaHostRegisterDefault);

    // std::cout << "buf_dev_head:" << buf_dev_head << "buf_dev_aligned:" << buf_dev_aligned << std::endl;

    for (size_t i = 0; i < options.nthreads; i++)
    {
        CUfileBatchHandle_t batch_idp = nullptr;
        checkCuFileErrors(cuFileBatchIOSetUp(&batch_idp, std::min<size_t>(options.io_multiplicity, props.max_batch_io_size)));

        std::vector<void *> buf_dev_vec(options.io_multiplicity);

        std::vector<CUfileIOParams_t> batch_params_vec(options.io_multiplicity);

        std::vector<CUfileIOEvents_t> batch_events_vec(options.io_multiplicity);

        std::vector<ScanDeviceMemCookie> batch_cookie_vec(options.io_multiplicity);

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

        thread_args_vec.push_back(ScanDeviceMemThreadArgs{
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
        ScanDeviceMemThreadArgs &args = thread_args_vec[i];
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
                            ScanDeviceMemCookie *cookie = &args.batch_cookie_vec[j];

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
                                ScanDeviceMemCookie *cookie = (ScanDeviceMemCookie *)events->cookie;
                                uint64_t pagid = pagid_vec[cookie->idx];
                                std::cerr << "  params: pagid=" << pagid
                                          << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                          << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                                std::cerr << "  options.io_multiplicity=" << options.io_multiplicity << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            ScanDeviceMemCookie *cookie = (ScanDeviceMemCookie *)events->cookie;

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
                            ScanDeviceMemCookie *cookie = (ScanDeviceMemCookie *)events->cookie;
                            uint64_t pagid = pagid_vec[cookie->idx];
                            std::cerr << "  params: pagid=" << pagid
                                      << ", ipagid=" << pagid_to_ipagid(pagid, options.ndev)
                                      << ", idev=" << pagid_to_idev(pagid, options.ndev) << std::endl;
                            std::cerr << "  options.io_multiplicity=" << io_multiplicity << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        ScanDeviceMemCookie *cookie = (ScanDeviceMemCookie *)events->cookie;
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
        ScanDeviceMemThreadArgs &args = thread_args_vec[i];
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
    // cudaHostUnregister(arr_buf_host_base);
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
BenchmarkResult tpch_scan_customer_device_mem(BenchmarkOptions &options) {
    return BenchmarkResult{};
}
#endif