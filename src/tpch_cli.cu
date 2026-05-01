#pragma once

#include <getopt.h>

#include <cstring>
#include <iomanip>
#include <sstream>

#include "schema/tpch_tables.cuh"
#include "common/common.cu"

void scan_size(const char *str, size_t *size)
{
    char unit[2] = { 0 };

    int scanned = sscanf(str, "%zu%c", size, &unit[0]);

    if (scanned == 1)
    {
    }
    else if (scanned == 2)
    {
        char u = unit[0];
        *size *= u == 'K'   ? KIBI
                 : u == 'M' ? MEBI
                 : u == 'G' ? GIBI
                 : u == 'T' ? TEBI
                            : 1;
    }
    else if (scanned == EOF)
    {
        perror("sscanf");
        exit(EXIT_FAILURE);
    }
    else
    {
        std::cerr << "Invalid argument" << std::endl;
        exit(EXIT_FAILURE);
    }
    // std::cout << *size << std::endl;
}

void print_benchmark_usage()
{
    std::cout
        << "Usage:\n"
           "  tpchdb [OPTIONS] file\n"
           "Options:\n"
           "  -x <mode>                     execution mode [gidp, gidp+bam, gidp+bam+fusion, datapathfusion]\n"
           "  -w <nthreads>                 number of threads [default: 1]\n"
           "  -l <period_sec>               experiment period [default: 30]\n"
           "  -i <io_multiplicity(K|M|G|T)> io multiplicity[examples: 1; default: 1]\n"
           "  -L <large_page_size(K|M|G)>   large page size for coalesced I/O [default: 16M]\n"
           "  -q <benchmark_type>           benchmark type [values: 6]\n"
           "  -e <output_format>            output format [values: text, json; default: text]\n"
           "  -a <sd_low>                   Q6 L_SHIPDATE lower bound (inclusive) [default: 19940101]\n"
           "  -b <sd_high>                  Q6 L_SHIPDATE upper bound (exclusive) [default: 19950101]\n"
           "  -K <coalesce_k>               BaM I/O coalescing factor [default: 1 (no coalescing)]\n"
           "  -Z                            enable zone map IO pruning (L_SHIPDATE)\n"
           "  -S                            use synchronous cuFileRead instead of batch API\n"
           "  -R                            enable prescan mode (pre-compute I/O plan)\n"
           "  -Q <qt_max>                   Revenue query: L_QUANTITY < qt_max (0 = no filter) [default: 0]\n"
           "  -p                            use prefix_sum for VCHAR page mapping (GOLAP only) [default: off]\n"
           "  -h                            display help\n";
}

BenchmarkOptions parse_benchmark_options(int argc, char *const *argv)
{
    char const *file = "";
    size_t nthreads = 1;
    size_t io_multiplicity = 1;
    size_t period_sec = 15;
    bool enable_prefetch = false;
    bool enable_zonemap = false;
    bool use_sync_io = false;
    bool use_prescan = false;
    bool use_prefix_sum = false;
    uint32_t block_size = 32;
    int32_t q6_sd_low = 19940101;
    int32_t q6_sd_high = 19950101;
    int32_t revenue_qt_max = 5100;
    int32_t q3sel_selectivity = 0;
    bool disable_other_filters = false;
    uint32_t coalesce_k = 1;
    size_t io_size = 1 * MEBI;
    size_t large_page_size = 16 * MEBI;
    size_t gds_num_handlers_per_thread = 1;
    char const *benchmark_kind_str = "TPCH";
    // Benchmark::Kind benchmark_kind = Benchmark::Kind::NONE;
    char const *query_str = "none";
    TPCH::QueryId query = TPCH::QueryId::NONE;
    char const *transfer_type_str = "host";
    Benchmark::TransferType transfer_type = Benchmark::TransferType::DEVICE;
    OutputFormat output_format = OutputFormat::TEXT;

    int opt;
    while ((opt = getopt(argc, argv, "i:w:l:x:q:e:H:L:a:b:K:B:Q:s:FPhZSRp")) != EOF)
    {
        switch (opt)
        {
       case 'i':
        {
            scan_size(optarg, &io_multiplicity);
            break;
        }
        case 'w':
        {
            sscanf(optarg, "%zu", &nthreads);
            break;
        }
        case 'l':
        {
            sscanf(optarg, "%zu", &period_sec);
            break;
        }
        case 'x':
        {
            transfer_type_str = optarg;
            if (strcmp(optarg, "host") == 0)
            {
                transfer_type = Benchmark::TransferType::HOST;
                // benchmark_kind = Benchmark::Kind::SSB;
            }
#if 0
            else if (strcmp(optarg, "host_async") == 0)
            {
                transfer_type = Benchmark::TransferType::HOST_ASYNC;
            }
            else if (strcmp(optarg, "host_device") == 0)
            {
                transfer_type = Benchmark::TransferType::HOST_DEVICE;
            }
            else if (strcmp(optarg, "host_device_async") == 0)
            {
                transfer_type = Benchmark::TransferType::HOST_DEVICE_ASYNC;
            }
            else if (strcmp(optarg, "host_async_device") == 0)
            {
                transfer_type = Benchmark::TransferType::HOST_ASYNC_DEVICE;
            }
            else if (strcmp(optarg, "device") == 0)
            {
                transfer_type = Benchmark::TransferType::DEVICE;
            }
            else if (strcmp(optarg, "device_batch") == 0)
            {
                transfer_type = Benchmark::TransferType::DEVICE_BATCH;
            }
 #endif
            if (strcmp(optarg, "gidp") == 0)
            {
                transfer_type = Benchmark::TransferType::GIDP;
                use_sync_io = true;  // sync IO + coalescing is default for gidp
            }
            else if (strcmp(optarg, "gidp+bam") == 0)
            {
                transfer_type = Benchmark::TransferType::GIDP_BAM;
            }
            else if (strcmp(optarg, "gidp+bam+fusion") == 0)
            {
                transfer_type = Benchmark::TransferType::GIDP_BAM_FUSION;
            }
            else if (strcmp(optarg, "datapathfusion") == 0)
            {
                transfer_type = Benchmark::TransferType::DATAPATHFUSION;
            }
            else
            {
                std::cerr << "Unknown execution mode: " << optarg << std::endl;
                std::cerr << "Valid modes: gidp, gidp+bam, gidp+bam+fusion, datapathfusion" << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        }
        case 'q':
        {
            query_str = optarg;
            /* check 1: customer table */
            /* check 11: varchar field 11 - C_NAME */
#if 0
            if (strcmp(optarg, "check11") == 0)
            {
                query = TPCH::QueryId::CHECK11;
            }
            /* check 12: varchar field 12 - C_ADDRESS */
            else if (strcmp(optarg, "check12") == 0)
            {
                query = TPCH::QueryId::CHECK12;
            }
            /* check 13: varchar field 13 - C_COMMENT */
            else if (strcmp(optarg, "check13") == 0)
            {
                query = TPCH::QueryId::CHECK13;
            }
#endif
            if (strcmp(optarg, "q1") == 0)
            {
                query = TPCH::QueryId::Q1;
            }
            else if (strcmp(optarg, "q3") == 0)
            {
                query = TPCH::QueryId::Q3;
            }
            else if (strcmp(optarg, "q6") == 0)
            {
                query = TPCH::QueryId::Q6;
            }
            else if (strcmp(optarg, "checkmeta") == 0)
            {
                query = TPCH::QueryId::CHECKMETA;
            }
            else if (strcmp(optarg, "test_pfor_page") == 0)
            {
                query = TPCH::QueryId::TEST_PFOR_PAGE;
            }
            else if (strcmp(optarg, "test_pfor64_page") == 0)
            {
                query = TPCH::QueryId::TEST_PFOR64_PAGE;
            }
            else if (strcmp(optarg, "test_lz4_page") == 0)
            {
                query = TPCH::QueryId::TEST_LZ4_PAGE;
            }
            else if (strcmp(optarg, "scan_o_comment_v4") == 0)
            {
                query = TPCH::QueryId::SCAN_O_COMMENT_V4;
            }
            else if (strcmp(optarg, "scan_o_comment_v5") == 0)
            {
                query = TPCH::QueryId::SCAN_O_COMMENT_V5;
            }
            else if (strcmp(optarg, "scan_o_comment_v6") == 0)
            {
                query = TPCH::QueryId::SCAN_O_COMMENT_V6;
            }
            else if (strcmp(optarg, "scan_l_comment_v7") == 0)
            {
                query = TPCH::QueryId::SCAN_L_COMMENT_V7;
            }
            else if (strcmp(optarg, "scan_l_comment_v8") == 0)
            {
                query = TPCH::QueryId::SCAN_L_COMMENT_V8;
            }
            else if (strcmp(optarg, "scan_l_comment") == 0)
            {
                query = TPCH::QueryId::SCAN_L_COMMENT;
            }
            else if (strcmp(optarg, "revenue") == 0)
            {
                query = TPCH::QueryId::REVENUE;
            }
            else if (strcmp(optarg, "q5") == 0)
            {
                query = TPCH::QueryId::Q5;
            }
            else if (strcmp(optarg, "q13") == 0)
            {
                query = TPCH::QueryId::Q13;
            }
            else if (strcmp(optarg, "q16") == 0)
            {
                query = TPCH::QueryId::Q16;
            }
            else if (strcmp(optarg, "q3sel") == 0)
            {
                query = TPCH::QueryId::Q3SEL;
            }
            else if (strcmp(optarg, "io_bench") == 0)
            {
                query = TPCH::QueryId::IO_BENCH;
            }
            else if (strcmp(optarg, "decomp_bench") == 0)
            {
                query = TPCH::QueryId::DECOMP_BENCH;
            }
            else if (strcmp(optarg, "scan_o_comment") == 0
                     || strcmp(optarg, "scan_o_comment_v1") == 0)
            {
                query = TPCH::QueryId::SCAN_O_COMMENT;
            }
            else if (strcmp(optarg, "scan_o_comment_v2") == 0)
            {
                query = TPCH::QueryId::SCAN_O_COMMENT_V2;
            }
            else if (strcmp(optarg, "scan_o_comment_v3") == 0)
            {
                query = TPCH::QueryId::SCAN_O_COMMENT_V3;
            }
            else
            {
                std::cerr << "unknown benchmark type " << optarg << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        }
        case 'e':
        {
            if (strcmp(optarg, "text") == 0)
            {
                output_format = OutputFormat::TEXT;
            }
            else if (strcmp(optarg, "json") == 0)
            {
                output_format = OutputFormat::JSON;
            }
            else
            {
                std::cerr << "unknown output format " << optarg << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        }
        case 'P':
        {
            enable_prefetch = true;
            break;
        }
        case 'H':
        {
            sscanf(optarg, "%zu", &gds_num_handlers_per_thread);
            break;
        }
        case 'L':
        {
            scan_size(optarg, &large_page_size);
            break;
        }
        case 'h':
        {
            print_benchmark_usage();
            exit(EXIT_SUCCESS);
            break;
        }
        case 'a':
        {
            q6_sd_low = atoi(optarg);
            break;
        }
        case 'b':
        {
            q6_sd_high = atoi(optarg);
            break;
        }
        case 'Z':
        {
            enable_zonemap = true;
            break;
        }
        case 'S':
        {
            use_sync_io = true;
            break;
        }
        case 'R':
        {
            use_prescan = true;
            break;
        }
        case 'p':
        {
            use_prefix_sum = true;
            break;
        }
        case 'K':
        {
            coalesce_k = atoi(optarg);
            break;
        }
        case 'B':
        {
            block_size = atoi(optarg);
            break;
        }
        case 'Q':
        {
            revenue_qt_max = atoi(optarg);
            break;
        }
        case 's':
        {
            q3sel_selectivity = atoi(optarg);
            break;
        }
        case 'F':
        {
            disable_other_filters = true;
            break;
        }
        }
    }

    if (optind < argc)
    {
        file = argv[optind++];
    }
    else
    {
        print_benchmark_usage();
        exit(EXIT_SUCCESS);
    }

    /* Q3SEL: default selectivity to 20% if not specified */
    if (query == TPCH::QueryId::Q3SEL && q3sel_selectivity == 0) {
        q3sel_selectivity = 20;
    }

    /* automatic thread configuration */
    if (nthreads == 0) {
        nthreads = std::max(1u, std::thread::hardware_concurrency());
    }

    auto job = Benchmark::Job(query, transfer_type);
    return BenchmarkOptions {
        .file = file,
        .nthreads = nthreads,
        .io_multiplicity = io_multiplicity,
        .io_size = io_size,
        .period_sec = period_sec,
        .large_page_size = large_page_size,
        .gds_num_handlers_per_thread = gds_num_handlers_per_thread,
        .enable_prefetch = enable_prefetch,
        .benchmark_kind_str = benchmark_kind_str,
        .query_str = query_str,
        .query = query,
        .transfer_type_str = transfer_type_str,
        .job = job,
        .output_format = output_format,
        .enable_zonemap = enable_zonemap,
        .use_sync_io = use_sync_io,
        .q6_sd_low = q6_sd_low,
        .q6_sd_high = q6_sd_high,
        .coalesce_k = coalesce_k,
        .use_prescan = use_prescan,
        .use_prefix_sum = use_prefix_sum,
        .block_size = block_size,
        .revenue_qt_max = revenue_qt_max,
        .q3sel_selectivity = q3sel_selectivity,
        .disable_other_filters = disable_other_filters,
    };
}

void validate_benchmark_options(const BenchmarkOptions &options)
{
    if (options.file == nullptr || strlen(options.file) == 0)
    {
        std::cerr << "file path must be given" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!(options.nthreads >= 1))
    {
        std::cerr << "number of threads must be greater than or equal to 1" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (Benchmark::has_none(options.job))
    {
        std::cerr << "benchmark type should be given" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!(options.io_size % 512 == 0))
    {
        std::cerr << "io_size must be a multiple of 512 bytes" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "OK" << std::endl;
}

std::string format_elapsed_msec(uint64_t elapsed_nanoseconds) {
    uint64_t msec = elapsed_nanoseconds / 1'000'000;
    uint64_t frac = (elapsed_nanoseconds % 1'000'000) / 1'000;

    std::ostringstream oss;
    oss << msec << '.' << std::setw(3) << std::setfill('0') << frac << " msec";
    return oss.str();
}

void output_benchmark_result(const BenchmarkOptions &options, const BenchmarkResult &result)
{
    if (options.output_format == OutputFormat::TEXT)
    {
        // std::cout << "written: " << options.file_size << " bytes" << std::endl;
        // std::cout << "time: " << time_format_ms(result.elapsed_nanoseconds) << " ns" << std::endl;
        uint64_t read_bytes = result.read_bytes ? result.read_bytes
                             : (result.nios * options.io_size);
        std::cout
            << "time: " << format_elapsed_msec(result.elapsed_nanoseconds) << "\n"
            << "nios: " << result.nios << "\n"
            << "read_mb: " << read_bytes / MEBI << "\n";
        if (result.total_pages > 0) {
            uint64_t uncompressed_bytes = result.total_pages * options.io_size;
            uint64_t uncompressed_mb = uncompressed_bytes / MEBI;
            double io_reduction = (double)read_bytes / (double)uncompressed_bytes;
            double elapsed_sec = result.elapsed_nanoseconds / 1e9;
            double throughput_gbs = elapsed_sec > 0
                ? ((double)uncompressed_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec : 0;
            double io_throughput_gbs = elapsed_sec > 0
                ? ((double)read_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec : 0;
            std::cout
                << "uncompressed_read_mb: " << uncompressed_mb << "\n"
                << std::fixed << std::setprecision(3)
                << "io_reduction_ratio: " << io_reduction << "\n"
                << std::setprecision(2)
                << "effective_throughput_gbs: " << throughput_gbs << "\n"
                << "io_throughput_gbs: " << io_throughput_gbs << "\n"
                << std::defaultfloat;
        }
        std::cout
            << "benchmark_kind: " << options.benchmark_kind_str << "\n"
            << "query: " << options.query_str << "\n"
            << "transfer_type: " << options.transfer_type_str << "\n"
            << "sync_io: " << (options.use_sync_io ? "true" : "false") << "\n"
            << "zonemap: " << (options.enable_zonemap ? "true" : "false") << "\n"
            << "file: " << options.file << "\n"
            << "number_of_threads: " << options.nthreads << "\n"
            << "num_thread_blocks: " << result.num_thread_blocks << "\n"
            << "io_mutiplicity: " << options.io_multiplicity << "\n"
            << "io_size: " << options.io_size << "\n"
            << "number_of_ios: " << result.nios << "\n"
            << "elapsed_nanoseconds: " << result.elapsed_nanoseconds << "\n"
            << "compression: " << (result.compression.empty() ? "N/A" : result.compression) << "\n"
            << "gpu_mem_mb: " << result.gpu_mem_bytes / MEBI << "\n"
            << "gpu_ctrl_mb: " << result.gpu_ctrl_bytes / MEBI << "\n"
            << "gpu_app_mb: " << result.gpu_app_bytes / MEBI << "\n"
            << "kernel_launches: " << result.kernel_launches << "\n"
            << "cpu_usage[\n"
                << "\tuser: " << result.cpu_usage.user << "\n"
                << "\tnice: " << result.cpu_usage.nice << "\n"
                << "\tsystem: " << result.cpu_usage.system << "\n"
                << "\tidle " << result.cpu_usage.idle << ",\n"
                << "\tiowait: " << result.cpu_usage.iowait << "\n"
            << "]\n";
        for (const auto &device : result.gpu_usage.devices)
        {
            std::cout
                << "gpu_usage[\n"
                    << "\tdevices: [\n"
                        << "\t\t\tgpu: " << device.gpu << "\n"
                        << "\t\t\tmemory: " << device.memory << "\n"
                    << "\t]\n"
                << "]\n";
        }
        std::cout << std::endl;
    }
    else if (options.output_format == OutputFormat::JSON)
    {
        std::cout
            << "{\n"
            << "  \"time\": " << format_elapsed_msec(result.elapsed_nanoseconds) << ",\n"
            << "  \"nios\": " << result.nios << ",\n"
            << "  \"read_mb\": " << (result.nios * options.io_size) / MEBI << ",\n";
        if (result.total_pages > 0) {
            uint64_t ub = result.total_pages * options.io_size;
            double elapsed_sec = result.elapsed_nanoseconds / 1e9;
            std::cout
                << "  \"uncompressed_read_mb\": " << ub / MEBI << ",\n"
                << std::fixed << std::setprecision(3)
                << "  \"io_reduction_ratio\": " << (double)(result.nios * options.io_size) / (double)ub << ",\n"
                << std::setprecision(2)
                << "  \"effective_throughput_gbs\": "
                << (elapsed_sec > 0 ? ((double)ub / (1024.0*1024.0*1024.0)) / elapsed_sec : 0) << ",\n"
                << "  \"io_throughput_gbs\": "
                << (elapsed_sec > 0 ? ((double)(result.nios * options.io_size) / (1024.0*1024.0*1024.0)) / elapsed_sec : 0) << ",\n"
                << std::defaultfloat;
        }
        std::cout
            << "  \"benchmark_kind\": \"" << options.benchmark_kind_str << "\",\n"
            << "  \"query\": \"" << options.query_str << "\",\n"
            << "  \"transfer_type\": \"" << options.transfer_type_str << "\",\n"
            << "  \"sync_io\": " << (options.use_sync_io ? "true" : "false") << ",\n"
            << "  \"zonemap\": " << (options.enable_zonemap ? "true" : "false") << ",\n"
            << "  \"file\": \"" << options.file << "\",\n"
            << "  \"number_of_threads\": " << options.nthreads << ",\n"
            << "  \"io_mutiplicity\": " << options.io_multiplicity << ",\n"
            << "  \"io_size\": " << options.io_size << ",\n"
            << "  \"number_of_ios\": " << result.nios << ",\n"
            << "  \"elapsed_nanoseconds\": " << result.elapsed_nanoseconds << ",\n"
            << "  \"compression\": \"" << (result.compression.empty() ? "N/A" : result.compression) << "\",\n"
            << "  \"gpu_mem_mb\": " << result.gpu_mem_bytes / MEBI << ",\n"
            << "  \"gpu_ctrl_mb\": " << result.gpu_ctrl_bytes / MEBI << ",\n"
            << "  \"gpu_app_mb\": " << result.gpu_app_bytes / MEBI << ",\n"
            << "  \"cpu_usage\": {\n"
            << "    \"user\": " << result.cpu_usage.user << ",\n"
            << "    \"nice\": " << result.cpu_usage.nice << ",\n"
            << "    \"system\": " << result.cpu_usage.system << ",\n"
            << "    \"idle\": " << result.cpu_usage.idle << ",\n"
            << "    \"iowait\": " << result.cpu_usage.iowait << "\n"
            << "  },\n"
            << "  \"gpu_usage\": {\n"
            << "    \"devices\": [\n";
        bool first = true;
        for (const auto &device : result.gpu_usage.devices)
        {
            if (!first)
            {
                std::cout << "      },\n";
            }
            std::cout
                << "      {\n"
                << "        \"gpu\": " << device.gpu << ",\n"
                << "        \"memory\": " << device.memory << "\n";
            first = false;
        }
        if (!first)
        {
            std::cout << "      }\n";
        }
        std::cout
            << "    ]\n"
            << "  }\n"
            << "}\n";
    }
}
