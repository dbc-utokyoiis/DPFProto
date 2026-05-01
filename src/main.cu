// #include "./bench_device.cu"
// #include "./bench_device_async.cu"
// #include "./bench_device_batch.cu"
// #include "./bench_device_batch_sync.cu"
// #include "./bench_host.cu"
// #include "./bench_host_async.cu"
// #include "./bench_host_async_device.cu"
// #include "./bench_host_device.cu"
// #include "./bench_host_device_async.cu"
// #include "./bench_transfer.cu"
// #include "./bench_transfer_async.cu"
// #include "q6/device_batch_sync.cu"
// #include "q6/device_batch_sync_compressed.cu"
// #include "q6/device_batch_io.cu"
#include "common/shim_nvml.cu"
#include "./cli.cu"
#include "ssb/check/host_count.cu"
#include "ssb/q11/host.cu"
#include "ssb/q11/host_compress.cu"
#include "ssb/q11/device.cu"
#include "ssb/q11/device_compress.cu"

int main(int argc, char *const *argv)
{
    std::cerr << "migmatite" << std::endl;

    BenchmarkOptions options = parse_benchmark_options(argc, argv);

    validate_benchmark_options(options);

#if 0
    nvml::init();
    size_t n = nvml::device_get_count();
    for (size_t i = 0; i < n; i++)
    {
        auto handle = nvml::device_get_handle_by_index(i);
        nvml::device_set_accounting_mode(handle, NVML_FEATURE_ENABLED);
    }
#endif

    BenchmarkResult result;
    if (Benchmark::is_query(options.job, SSB::Query::CHECK))
    {
        if (!Benchmark::is_transfertype(options.job, Benchmark::TransferType::HOST))
        {
            std::cerr << "Unsupported transfer type for check" << std::endl;
            exit(EXIT_FAILURE);
        }
        result = check_host(options);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q11))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::HOST))
        {
            //result = hostSyncSSBQ11(options);
            std::cout << "hostSyncCompressSSBQ11" << std::endl;
            result = hostSyncCompressSSBQ11(options);
            std::cout << "end" << std::endl;
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DEVICE))
        {
            // if (options.compress)
            // {
            //     result = deviceSyncCompressSSBQ11(options);
            // }
            // else
            {
                //result = deviceSyncSSBQ11(options);
                result = deviceSyncCompressSSBQ11(options);
            }
        }
        // Not yet implemented
        // else if (options.transfer_type == Benchmark::TransferType::DEVICE_BATCH)
        // {
        //     result = deviceBatchSSBQ11(options);
        // }
        // else if (options.transfer_type == Benchmark::TransferType::DEVICE_BATCH_SYNC)
        // {
        //     result = deviceBatchSyncSSBQ11(options);
        // }
        else
        {
            std::cerr << "Unsupported transfer type for Q11" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cerr << "Unsupported benchmark type" << std::endl;
        exit(EXIT_FAILURE);
    }
    output_benchmark_result(options, result);

#if 0
    nvml::shutdown();
#endif
}
