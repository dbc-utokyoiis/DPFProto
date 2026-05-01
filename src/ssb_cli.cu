#pragma once

#include <getopt.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "common/common.cu"
#include "schema/ssb_tables.cuh"

void print_ssb_benchmark_usage()
{
    std::cout
        << "Usage:\n"
           "  ssbdb [OPTIONS] device[,device,...]\n"
           "Options:\n"
           "  -x <mode>                     execution mode [gidp, gidp+bam, gidp+bam+fusion, datapathfusion]\n"
           "  -w <nthreads>                 number of threads [default: 1]\n"
           "  -q <query>                    SSB query [q11,q12,q13,q21,q22,q23,q31,q32,q33,q34,q41,q42,q43]\n"
           "  -e <output_format>            output format [values: text, json; default: text]\n"
           "  -a <sd_low>                   Revenue query: LO_ORDERDATE >= sd_low [default: 19920101]\n"
           "  -b <sd_high>                  Revenue query: LO_ORDERDATE <= sd_high [default: 19981231]\n"
           "  -Q <qt_max>                   Revenue query: LO_QUANTITY < qt_max (0 = no filter) [default: 0]\n"
           "  -K <coalesce_k>               BaM I/O coalescing factor [default: 1]\n"
           "  -Z                            enable sideways information passing (zone map pruning)\n"
           "  -S                            use synchronous cuFileRead instead of batch API\n"
           "  -h                            display help\n";
}

static void ssb_scan_size(const char *str, size_t *size)
{
    char unit[2] = { 0 };
    int scanned = sscanf(str, "%zu%c", size, &unit[0]);
    if (scanned == 1) {
    } else if (scanned == 2) {
        char u = unit[0];
        *size *= u == 'K'   ? KIBI
                 : u == 'M' ? MEBI
                 : u == 'G' ? GIBI
                 : u == 'T' ? TEBI
                            : 1;
    } else {
        std::cerr << "Invalid argument" << std::endl;
        exit(EXIT_FAILURE);
    }
}

BenchmarkOptions parse_ssb_benchmark_options(int argc, char *const *argv)
{
    char const *file = "";
    size_t nthreads = 1;
    size_t io_multiplicity = 1;
    size_t period_sec = 15;
    bool enable_zonemap = false;
    bool use_sync_io = false;
    uint32_t coalesce_k = 1;
    size_t io_size = 1 * MEBI;
    size_t large_page_size = 16 * MEBI;
    char const *benchmark_kind_str = "SSB";
    char const *query_str = "none";
    SSB::Query ssb_query = SSB::Query::NONE;
    char const *transfer_type_str = "gidp";
    Benchmark::TransferType transfer_type = Benchmark::TransferType::GIDP;
    OutputFormat output_format = OutputFormat::TEXT;
    int32_t q6_sd_low = 19920101;
    int32_t q6_sd_high = 19981231;
    int32_t revenue_qt_max = 0;
    bool disable_other_filters = false;

    int opt;
    while ((opt = getopt(argc, argv, "w:x:q:e:K:L:a:b:Q:hZSF")) != EOF)
    {
        switch (opt)
        {
        case 'w':
            sscanf(optarg, "%zu", &nthreads);
            break;
        case 'x':
        {
            transfer_type_str = optarg;
            if (strcmp(optarg, "gidp") == 0) {
                transfer_type = Benchmark::TransferType::GIDP;
                use_sync_io = true;
            } else if (strcmp(optarg, "gidp+bam") == 0) {
                transfer_type = Benchmark::TransferType::GIDP_BAM;
            } else if (strcmp(optarg, "gidp+bam+fusion") == 0) {
                transfer_type = Benchmark::TransferType::GIDP_BAM_FUSION;
            } else if (strcmp(optarg, "datapathfusion") == 0) {
                transfer_type = Benchmark::TransferType::DATAPATHFUSION;
            } else {
                std::cerr << "Unknown execution mode: " << optarg << std::endl;
                std::cerr << "Valid modes: gidp, gidp+bam, gidp+bam+fusion, datapathfusion" << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        }
        case 'q':
        {
            query_str = optarg;
            if (strcmp(optarg, "q11") == 0) ssb_query = SSB::Query::Q11;
            else if (strcmp(optarg, "q12") == 0) ssb_query = SSB::Query::Q12;
            else if (strcmp(optarg, "q13") == 0) ssb_query = SSB::Query::Q13;
            else if (strcmp(optarg, "q21") == 0) ssb_query = SSB::Query::Q21;
            else if (strcmp(optarg, "q22") == 0) ssb_query = SSB::Query::Q22;
            else if (strcmp(optarg, "q23") == 0) ssb_query = SSB::Query::Q23;
            else if (strcmp(optarg, "q31") == 0) ssb_query = SSB::Query::Q31;
            else if (strcmp(optarg, "q32") == 0) ssb_query = SSB::Query::Q32;
            else if (strcmp(optarg, "q33") == 0) ssb_query = SSB::Query::Q33;
            else if (strcmp(optarg, "q34") == 0) ssb_query = SSB::Query::Q34;
            else if (strcmp(optarg, "q41") == 0) ssb_query = SSB::Query::Q41;
            else if (strcmp(optarg, "q42") == 0) ssb_query = SSB::Query::Q42;
            else if (strcmp(optarg, "q43") == 0) ssb_query = SSB::Query::Q43;
            else if (strcmp(optarg, "revenue") == 0) ssb_query = SSB::Query::REVENUE;
            else {
                std::cerr << "Unknown SSB query: " << optarg << std::endl;
                std::cerr << "Valid queries: q11,...,q43,revenue" << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        }
        case 'e':
        {
            if (strcmp(optarg, "text") == 0) output_format = OutputFormat::TEXT;
            else if (strcmp(optarg, "json") == 0) output_format = OutputFormat::JSON;
            else {
                std::cerr << "Unknown output format: " << optarg << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        }
        case 'K':
            coalesce_k = atoi(optarg);
            break;
        case 'L':
            ssb_scan_size(optarg, &large_page_size);
            break;
        case 'Z':
            enable_zonemap = true;
            break;
        case 'a':
            q6_sd_low = atoi(optarg);
            break;
        case 'b':
            q6_sd_high = atoi(optarg);
            break;
        case 'Q':
            revenue_qt_max = atoi(optarg);
            break;
        case 'S':
            use_sync_io = true;
            break;
        case 'F':
            disable_other_filters = true;
            break;
        case 'h':
            print_ssb_benchmark_usage();
            exit(EXIT_SUCCESS);
            break;
        }
    }

    if (optind < argc) {
        file = argv[optind++];
    } else {
        print_ssb_benchmark_usage();
        exit(EXIT_SUCCESS);
    }

    if (nthreads == 0) {
        nthreads = std::max(1u, std::thread::hardware_concurrency());
    }

    auto job = Benchmark::Job(ssb_query, transfer_type);
    return BenchmarkOptions {
        .file = file,
        .nthreads = nthreads,
        .io_multiplicity = io_multiplicity,
        .io_size = io_size,
        .period_sec = period_sec,
        .large_page_size = large_page_size,
        .gds_num_handlers_per_thread = 1,
        .enable_prefetch = false,
        .benchmark_kind_str = benchmark_kind_str,
        .query_str = query_str,
        .query = TPCH::QueryId::NONE,
        .transfer_type_str = transfer_type_str,
        .job = job,
        .output_format = output_format,
        .enable_zonemap = enable_zonemap,
        .use_sync_io = use_sync_io,
        .q6_sd_low = q6_sd_low,
        .q6_sd_high = q6_sd_high,
        .coalesce_k = coalesce_k,
        .use_prescan = false,
        .use_prefix_sum = false,
        .block_size = 32,
        .revenue_qt_max = revenue_qt_max,
        .disable_other_filters = disable_other_filters,
    };
}

void validate_ssb_benchmark_options(const BenchmarkOptions &options)
{
    if (options.file == nullptr || strlen(options.file) == 0) {
        std::cerr << "device path(s) must be given" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!(options.nthreads >= 1)) {
        std::cerr << "number of threads must be >= 1" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (Benchmark::has_none(options.job)) {
        std::cerr << "SSB query (-q) must be specified" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "OK" << std::endl;
}
