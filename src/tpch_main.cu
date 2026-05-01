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
#include "common/shim_nvml.cu"
#include "./tpch_cli.cu"

#include "common/xtn.h"
#include "schema/tpch_tables.cuh"

#include "kernel/tpch/q6.cuh"
#include "tpch/check/device_batch_sync.cu"
#include "tpch/check/mem.cu"
#include "tpch/gidp.cu"
#include "tpch/datapathfusion.cu"
#include "tpch/gidp_bam.cu"
#include "tpch/gidp_bam_fusion.cu"
//#include "tpch/check/host_count.cu"
//#include "tpch/check/customer.cu"

static void dpf_to_result(const DataPathFusion::PigResult& pig, BenchmarkResult& result,
                          const CpuUsage& start_cpu) {
    result.elapsed_nanoseconds = static_cast<int64_t>(pig.elapsed_ms * 1e6);
    result.nios = pig.nios;
    result.read_bytes = pig.read_bytes;
    result.compression = pig.compression;
    result.num_thread_blocks = 0;
    result.gpu_mem_bytes = pig.gpu_mem_bytes;
    result.gpu_ctrl_bytes = pig.gpu_ctrl_bytes;
    result.gpu_app_bytes = pig.gpu_app_bytes;
    result.total_pages = pig.total_pages;
    result.kernel_launches = pig.kernel_launches;
    result.cpu_usage = diff_cpu_usages(start_cpu, read_cpu_usage());
    result.gpu_usage = get_gpu_usage();
}

int main(int argc, char *const *argv)
{
    std::cerr << "migmatite" << std::endl;

    BenchmarkOptions options = parse_benchmark_options(argc, argv);

    validate_benchmark_options(options);

    nvml::init();

    BenchmarkResult result;
#if 0
    if (Benchmark::is_query(options.job, TPCH::QueryId::CHECK11)
        || Benchmark::is_query(options.job, TPCH::QueryId::CHECK12)
        || Benchmark::is_query(options.job, TPCH::QueryId::CHECK13))
    {
        // result = check_host(options);
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DEVICE))
        {
            //result = tpch_scan_customer_device_batch_sync(options);
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DEVICE_PRELOAD))
        {
            //result = tpch_scan_customer_device_mem(options);
        }
    }
#endif
    if (Benchmark::is_query(options.job, TPCH::QueryId::CHECKMETA))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::bam_check_metadata(options);
            return 0;
        }
        else
        {
            std::cerr << "checkmeta only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::TEST_PFOR_PAGE))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::test_pfor_page(options);
            return 0;
        }
        else
        {
            std::cerr << "test_pfor_page only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::TEST_PFOR64_PAGE))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::test_pfor64_page(options);
            return 0;
        }
        else
        {
            std::cerr << "test_pfor64_page only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::TEST_LZ4_PAGE))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::test_lz4_page(options);
            return 0;
        }
        else
        {
            std::cerr << "test_lz4_page only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::SCAN_O_COMMENT))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::scan_o_comment(options);
            return 0;
        }
        else
        {
            std::cerr << "scan_o_comment only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::SCAN_O_COMMENT_V2))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::scan_o_comment_v2(options);
            return 0;
        }
        else
        {
            std::cerr << "scan_o_comment_v2 only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::SCAN_O_COMMENT_V3))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::scan_o_comment_v3(options);
            return 0;
        }
        else
        {
            std::cerr << "scan_o_comment_v3 only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::SCAN_O_COMMENT_V4))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::scan_o_comment_v4(options);
            return 0;
        }
        else
        {
            std::cerr << "scan_o_comment_v4 only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::SCAN_O_COMMENT_V5))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::scan_o_comment_v5(options);
            return 0;
        }
        else
        {
            std::cerr << "scan_o_comment_v5 only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::SCAN_O_COMMENT_V6))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::scan_o_comment_v6(options);
            return 0;
        }
        else
        {
            std::cerr << "scan_o_comment_v6 only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::SCAN_L_COMMENT_V7))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::scan_l_comment_v7(options);
            return 0;
        }
        else
        {
            std::cerr << "scan_l_comment_v7 only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::SCAN_L_COMMENT_V8))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            DataPathFusion::scan_l_comment_v8(options);
            return 0;
        }
        else
        {
            std::cerr << "scan_l_comment_v8 only supports -x datapathfusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::SCAN_L_COMMENT))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
        {
            auto sc0 = read_cpu_usage();
            result = Gidp::scan_l_comment(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else
        {
            std::cerr << "scan_l_comment only supports -x gidp" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::Q1))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
        {
            auto sc0 = read_cpu_usage();
            result = Gidp::tpch_q1(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBam::tpch_q1(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBamFusion::tpch_q1(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            auto start_cpu = read_cpu_usage();
            dpf_to_result(DataPathFusion::tpch_q1(options), result, start_cpu);
        }
        else
        {
            std::cerr << "Unsupported transfer type for Q1" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::Q6))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
        {
            auto sc0 = read_cpu_usage();
            result = Gidp::tpch_q6(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBam::tpch_q6(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBamFusion::tpch_q6(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            auto sc0 = read_cpu_usage();
            result = DataPathFusion::tpch_q6(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else
        {
            std::cerr << "Unsupported transfer type for Q6" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::REVENUE))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
        {
            auto sc0 = read_cpu_usage();
            result = Gidp::tpch_revenue(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBam::tpch_revenue(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBamFusion::tpch_revenue(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            auto sc0 = read_cpu_usage();
            result = DataPathFusion::tpch_revenue(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else
        {
            std::cerr << "Unsupported transfer type for revenue" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::Q5))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
        {
            auto sc0 = read_cpu_usage();
            result = Gidp::tpch_q5(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBam::tpch_q5(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBamFusion::tpch_q5(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            auto start_cpu = read_cpu_usage();
            dpf_to_result(DataPathFusion::tpch_q5(options), result, start_cpu);
        }
        else
        {
            std::cerr << "Unsupported transfer type for Q5" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::Q3))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
        {
            auto sc0 = read_cpu_usage();
            result = Gidp::tpch_q3(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBam::tpch_q3(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBamFusion::tpch_q3(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            auto start_cpu = read_cpu_usage();
            dpf_to_result(DataPathFusion::tpch_q3(options), result, start_cpu);
        }
        else
        {
            std::cerr << "Unsupported transfer type for Q3" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::Q3SEL))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
        {
            auto sc0 = read_cpu_usage();
            result = Gidp::tpch_q3sel(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBam::tpch_q3sel(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBamFusion::tpch_q3sel(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            auto start_cpu = read_cpu_usage();
            dpf_to_result(DataPathFusion::tpch_q3sel(options), result, start_cpu);
        }
        else
        {
            std::cerr << "Unsupported transfer type for Q3SEL" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::Q13))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
        {
            auto sc0 = read_cpu_usage();
            result = Gidp::tpch_q13(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBam::tpch_q13(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBamFusion::tpch_q13(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            auto start_cpu = read_cpu_usage();
            dpf_to_result(DataPathFusion::tpch_q13(options), result, start_cpu);
        }
        else
        {
            std::cerr << "Unsupported transfer type for Q13" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::Q16))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
        {
            auto sc0 = read_cpu_usage();
            result = Gidp::tpch_q16(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBam::tpch_q16(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            auto sc0 = read_cpu_usage();
            result = GidpBamFusion::tpch_q16(options);
            result.cpu_usage = diff_cpu_usages(sc0, read_cpu_usage());
            result.gpu_usage = get_gpu_usage();
        }
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
        {
            auto start_cpu = read_cpu_usage();
            dpf_to_result(DataPathFusion::tpch_q16(options), result, start_cpu);
        }
        else
        {
            std::cerr << "Unsupported transfer type for Q16" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::IO_BENCH))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            result = GidpBamFusion::io_contention_bench(options);
            return 0;  // benchmark prints its own results
        }
        else
        {
            std::cerr << "io_bench only supports -x gidp+bam+fusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if (Benchmark::is_query(options.job, TPCH::QueryId::DECOMP_BENCH))
    {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
        {
            result = GidpBamFusion::lz4_decomp_bench(options);
            return 0;  // benchmark prints its own results
        }
        else
        {
            std::cerr << "decomp_bench only supports -x gidp+bam+fusion" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cerr << "Unsupported benchmark type" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cerr << "Elapsed time (ns): " << result.elapsed_nanoseconds << std::endl;
    output_benchmark_result(options, result);

    nvml::shutdown();
}
