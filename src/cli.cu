#pragma once

#include <getopt.h>

#include <cstring>
#include <iomanip>
#include <sstream>


#include "common/common.cu"

void scan_size(const char *str, size_t *size)
{
    char unit = '\0';

    int scanned = sscanf(str, "%zu%[KMGT]", size, &unit);

    if (scanned == 1)
    {
    }
    else if (scanned == 2)
    {
        *size *= unit == 'K'   ? KIBI
                 : unit == 'M' ? MEBI
                 : unit == 'G' ? GIBI
                 : unit == 'T' ? TEBI
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
}

void print_benchmark_usage()
{
    std::cout
        << "Usage:\n"
           "  migmatite [OPTIONS] file\n"
           "Options:\n"
           "  -x <transfer_type>            transfer type [values: host, host_async, host_device, host_device_async,\n"
           "                                device, device_batch, device_async; default: host]\n"
           "  -w <nthreads>                 number of threads [default: 1]\n"
           "  -d <io_depth>                 I/O depth [default: 1]\n"
           "  -l <period_sec>               experiment period [default: 30]\n"
           "  -p <page_size(K|M|G|T)>       I/O size [examples: 1K, 234M, 2G; default: 1M]\n"
           "  -s <subpage_size(K|M|G|T)>    file size [examples: 1K, 234M, 2G; default: 1G]\n"
           "  -i <io_multiplicity(K|M|G|T)> io multiplicity[examples: 1; default: 1]\n"
           "  -q <benchmark_type>           benchmark type [values: 6]\n"
           "  -e <output_format>            output format [values: text, json; default: text]\n"
           "  -h                            display help\n";
}

BenchmarkOptions parse_benchmark_options(int argc, char *const *argv)
{
    char const *file = "";
    size_t nthreads = 1;
    size_t io_multiplicity = 1;
    size_t period_sec = 15;
    size_t file_size = 100 * GIBI;
    size_t page_size = 1 * MEBI;
    size_t sub_page_size = 64 * KIBI;
    size_t io_size = 1 * MEBI;
    char const *benchmark_kind_str = "SSB";
    // Benchmark::Kind benchmark_kind = Benchmark::Kind::NONE;
    char const *query_str = "none";
    SSB::Query query = SSB::Query::NONE;
    char const *transfer_type_str = "host";
    Benchmark::TransferType transfer_type = Benchmark::TransferType::HOST;
    OutputFormat output_format = OutputFormat::TEXT;

    int opt;
    while ((opt = getopt(argc, argv, "p:s:f:d:i:w:l:x:q:e:h")) != EOF)
    {
        switch (opt)
        {
        case 'p':
        {
            scan_size(optarg, &page_size);
            break;
        }
        case 's':
        {
            scan_size(optarg, &sub_page_size);
            break;
        }
        case 'f':
        {
            scan_size(optarg, &file_size);
            break;
        }
       case 'i':
       case 'd':
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
#endif
            else if (strcmp(optarg, "host_async_device") == 0)
            {
                transfer_type = Benchmark::TransferType::HOST_ASYNC_DEVICE;
                // benchmark_kind = Benchmark::Kind::SSB;
            }
            else if (strcmp(optarg, "device") == 0)
            {
                transfer_type = Benchmark::TransferType::DEVICE;
                // benchmark_kind = Benchmark::Kind::SSB;
            }
            else if (strcmp(optarg, "device_batch") == 0)
            {
                transfer_type = Benchmark::TransferType::DEVICE_BATCH;
                // benchmark_kind = Benchmark::Kind::SSB;
            }
            else if (strcmp(optarg, "device_batch_sync") == 0)
            {
                transfer_type = Benchmark::TransferType::DEVICE_BATCH_SYNC;
                // benchmark_kind = Benchmark::Kind::SSB;
            }
#if 0
            else if (strcmp(optarg, "transfer") == 0)
            {
                transfer_type = TransferType::TRANSFER;
            }
            else if (strcmp(optarg, "transfer_async") == 0)
            {
                transfer_type = TransferType::TRANSFER_ASYNC;
            }
#endif
            else
            {
                std::cerr << "unknown transfer type " << optarg << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        }
        case 'q':
        {
            query_str = optarg;
            if (strcmp(optarg, "q11") == 0)
            {
                query = SSB::Query(SSB::Query::Q11);
            }
            else if (strcmp(optarg, "q21") == 0)
            {
                query = SSB::Query(SSB::Query::Q21);
            }
            else if (strcmp(optarg, "check") == 0)
            {
                query = SSB::Query(SSB::Query::CHECK);
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
        case 'h':
        {
            print_benchmark_usage();
            exit(EXIT_SUCCESS);
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

    auto job = Benchmark::Job(query, transfer_type);
    return BenchmarkOptions{
        .file = file,
        .nthreads = nthreads,
        .io_multiplicity = io_multiplicity,
        .io_size = io_size,
        .page_size = page_size,
        .sub_page_size = sub_page_size,
        .period_sec = period_sec,
        .file_size = file_size,
        .benchmark_kind_str = benchmark_kind_str,
        .query_str = query_str,
        .transfer_type_str = transfer_type_str,
        .job = job,
        .output_format = output_format,
    };
}

void validate_benchmark_options(const BenchmarkOptions &options)
{
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
        std::cout
            << "time: " << format_elapsed_msec(result.elapsed_nanoseconds) << "\n"
            << "nios: " << result.nios << "\n"
            << "read_mb: " << (result.nios * options.io_size) / MEBI << "\n"
            << "benchmark_kind: " << options.benchmark_kind_str << "\n"
            << "query: " << options.query_str << "\n"
            << "transfer_type: " << options.transfer_type_str << "\n"
            << "file: " << options.file << "\n"
            << "number_of_threads: " << options.nthreads << "\n"
            << "io_mutiplicity: " << options.io_multiplicity << "\n"
            << "io_size: " << options.io_size << "\n"
            << "number_of_ios: " << result.nios << "\n"
            << "elapsed_nanoseconds: " << result.elapsed_nanoseconds << "\n"
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
            << "  \"read_mb\": " << (result.nios * options.io_size) / MEBI << ",\n"
            << "  \"benchmark_kind\": \"" << options.benchmark_kind_str << "\",\n"
            << "  \"query\": \"" << options.query_str << "\",\n"
            << "  \"transfer_type\": \"" << options.transfer_type_str << "\",\n"
            << "  \"file\": \"" << options.file << "\",\n"
            << "  \"number_of_threads\": " << options.nthreads << ",\n"
            << "  \"io_mutiplicity\": " << options.io_multiplicity << ",\n"
            << "  \"io_size\": " << options.io_size << ",\n"
            << "  \"number_of_ios\": " << result.nios << ",\n"
            << "  \"elapsed_nanoseconds\": " << result.elapsed_nanoseconds << ",\n"
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
