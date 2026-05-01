#pragma once

#include <getopt.h>

#include <cstring>

#include "common/tpch_loader.cu"

void scan_size(const char *str, size_t *size)
{
    char unit[2] = {0};

    int scanned = sscanf(str, "%zu%[KMGT]", size, unit);

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
}

void print_loader_usage()
{
    std::cout
        << "Usage:\n"
           "  loader [OPTIONS]\n"
           "Options:\n"
           "  -i, --inputdir=DIR            input directory\n"
           "  -o, --outputfiles=FILE        output files (commma-separated regular or block device files) \n"
           "  -A, --all                     load all tables\n"
           "  -C, --customer                load customer table\n"
           "  -N, --nation                  load nation table\n"
           "  -R, --region                  load region table\n"
           "  -L, --lineitem                load lineitem table\n"
           "  -O, --orders                  load orders table\n"
           "  -S, --supplier                load supplier table\n"
           "  -P, --part                    load part table\n"
           "  -T, --partsupp                load partsupp table\n"
           "  -x, --execmode=MODE           execution mode [gidp, gidp+bam, gidp+bam+fusion, datapathfusion]\n"
           "  -G, --gidpcomp                load as gidp compression mode (legacy)\n"
           "  -s, --sidewaysstats           enable sideways statistics (legacy)\n"
           "  -l, --lbc_enabled             enable lbc mode\n"
           "  -u, --lbc_numclusters         number of varchar clusters for lbc mode (default:128)\n"
           "  -D, --dictencoding            enable dictionary encoding\n"
           "  -V, --vchar2fixedchar         enable varchar to fixedchar conversion\n"
           "  -c, --compress                enable compression. \n"
           "  -d, --dryrun                  enable dryrun (no data write). \n"
           "  -p, --pagesize=SIZE           page size (default:1M) K,M,G,T can be used as an unit \n"
           "  -h                            display help\n";
}

LoaderOptions tpch_parse_loader_options(int argc, char *const *argv)
{
    size_t page_size = 1024 * 1024;
    OutputFormat output_format = OutputFormat::TEXT;
    struct option longopts[] = {
        { "all",                    no_argument, NULL, 'A' },
        { "customer",               no_argument, NULL, 'C' },
        { "lineitem",               no_argument, NULL, 'L' },
        { "nation",                 no_argument, NULL, 'N' },
        { "orders",                 no_argument, NULL, 'O' },
        { "part",                   no_argument, NULL, 'P' },
        /* partsupp is enabled by specifying -T (parTsupp) */
        { "partsupp",               no_argument, NULL, 'T' },
        { "region",                 no_argument, NULL, 'R' },
        { "supplier",               no_argument, NULL, 'S' },
        { "inputdir",         required_argument, NULL, 'i' },
        { "outputfiles",      required_argument, NULL, 'd' },
        { "vchar2fixedchar",  optional_argument, NULL, 'V' },
        { "sideways_stats",         no_argument, NULL, 's' },
        { "gidp_compression",       no_argument, NULL, 'G' },
        { "dictencoding",     optional_argument, NULL, 'D' },
        { "compress",               no_argument, NULL, 'c' },
        { "lbc_enabled",            no_argument, NULL, 'l' },
        { "lbc_numclusters",  optional_argument, NULL, 'u' },
        { "test",             optional_argument, NULL, 't' },
        { "dryrun",           optional_argument, NULL, 'n' },
        { "pagesize",         optional_argument, NULL, 'p' },
        { "verify",           optional_argument, NULL, 'f' },
        { "verbose",          optional_argument, NULL, 'v' },
        { "outputformat",     optional_argument, NULL, 'e' },
        { "execmode",         required_argument, NULL, 'x' },
        { 0,        0,                 0,     0  },
    };
    const char *dirname = "";
    const char *filname = "";
    bool enable_varchar_to_fixedchar = false;
    bool enable_dict_encoding = false;
    bool enable_lz4par = false;
    bool enable_fsst = false;
    bool load_customer = false;
    bool load_lineitem = false;
    bool load_nation = false;
    bool load_orders = false;
    bool load_part = false;
    bool load_partsupp = false;
    bool load_region = false;
    bool load_supplier = false;
    bool enable_sideways_stats = false;
    bool enable_golap_compression_mode = false;
    bool test = false;
    bool verify = false;
    bool compress = false;
    bool dryrun = false;
    bool verbose = false;
    bool lbc_enabled = false;
    size_t lbc_num_varchar_clusters = 128;
    ExecutionMode execution_mode = ExecutionMode::LEGACY;

    int opt;
    int longindex;

    //while ((opt = getopt(argc, argv, "e:p:ah")) != EOF)
    while ((opt = getopt_long(argc, argv, "i:d:e:p:u:x:cdvACDLNOPTStfhrVlsGZ", longopts, &longindex)) != EOF)
    {
        // std::cout << "Debug: found option " << (char)opt << std::endl;
        switch (opt)
        {
            case 'i':
            {
                dirname = optarg;
                break;
            }
            case 'd':
            {
                filname = optarg;
                break;
            }
            case 'l':
            {
                lbc_enabled = true;
                break;
            }
            case 'u':
            {
                lbc_num_varchar_clusters = atoi(optarg);
                if (lbc_num_varchar_clusters <= 0)
                {
                    std::cerr << "Invalid number of clusters " << optarg << std::endl;
                    std::cerr << "Valid range is positive integer value " << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 'c':
            {
                compress = true;
                break;
            }
            case 'n':
            {
                dryrun = true;
                break;
            }
            case 'f':
            {
                verify = true;
                break;
            }
            case 'v':
            {
                verbose = true;
                break;
            }
            // ACLNOPTRS
            case 'A':
            {
                // bulk load all tables
                load_customer = true;
                load_lineitem = true;
                load_nation = true;
                load_orders = true;
                load_part = true;
                load_partsupp = true;
                load_region = true;
                load_supplier = true;
 
                break;
            }
            case 'C':
            {
                load_customer = true;
                break;
            }
            case 'Z':
            {
                enable_fsst = true;
                break;
            }
            case 'L':
            {
                load_lineitem = true;
                break;
            }
            case 'N':
            {
                load_nation = true;
                break;
            }
            case 'O':
            {
                load_orders = true;
                break;
            }
            case 'R':
            {
                load_region = true;
                break;
            }
            case 'P':
            {
                load_part = true;
                break;
            }
            case 'T':
            {
                load_partsupp = true;
                break;
            }
            case 'S':
            {
                load_supplier = true;
                break;
            }
            case 's':
            {
                enable_sideways_stats = true;
                break;
            }
             case 'G':
            {
                enable_golap_compression_mode = true;
                enable_sideways_stats = true;
                break;
            }
            case 't':
            {
                test = true;
                break;
            }
            case 'V':
            {
                enable_varchar_to_fixedchar = true;
                break;
            }
             case 'D':
            {
                enable_dict_encoding = true;
                break;
            }
            case 'p':
            {
                scan_size(optarg, &page_size);
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
            case 'x':
            {
                if (strcmp(optarg, "gidp") == 0) {
                    execution_mode = ExecutionMode::GIDP;
                } else if (strcmp(optarg, "gidp+bam") == 0) {
                    execution_mode = ExecutionMode::GIDP_BAM;
                } else if (strcmp(optarg, "gidp+bam+fusion") == 0) {
                    execution_mode = ExecutionMode::GIDP_BAM_FUSION;
                } else if (strcmp(optarg, "datapathfusion") == 0) {
                    execution_mode = ExecutionMode::DATAPATHFUSION;
                } else {
                    std::cerr << "Unknown execution mode: " << optarg << std::endl;
                    std::cerr << "Valid modes: gidp, gidp+bam, gidp+bam+fusion, datapathfusion" << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            }
            case 'h':
            default:
            {
                print_loader_usage();
                exit(EXIT_SUCCESS);
                break;
            }
        }
    }

    #if 0
    if (optind < argc)
    {
        dirname = argv[optind++];
    }
    else
    {
        print_loader_usage();
        exit(EXIT_SUCCESS);
    }
    #endif
    // -x execution mode overrides -G/-s flags
    switch (execution_mode) {
    case ExecutionMode::GIDP:
    case ExecutionMode::GIDP_BAM:
        enable_golap_compression_mode = true;
        enable_sideways_stats = true;
        break;
    case ExecutionMode::GIDP_BAM_FUSION:
        enable_golap_compression_mode = true;  // throughput estimation
        enable_sideways_stats = true;
        break;
    case ExecutionMode::DATAPATHFUSION:
        enable_sideways_stats = true;
        enable_fsst = true;
        break;
    case ExecutionMode::LEGACY:
        break; // use -G/-s flags as-is
    }

    if (enable_golap_compression_mode)
    {
        std::cout << "Golap compression mode is enabled. Sideways statistics is also enabled." << std::endl;
        enable_sideways_stats = true;
    }

    return LoaderOptions{
        .input_dirname = dirname,
        .output_files = filname,
        .lbc_enabled = lbc_enabled,
        .lbc_num_varchar_clusters = lbc_num_varchar_clusters,
        .compress = compress,
        .dryrun = dryrun,
        .verbose = verbose,
        .page_size = page_size,
        .enable_column = true,
        .enable_varchar_to_fixedchar = enable_varchar_to_fixedchar,
        .enable_dict_encoding = enable_dict_encoding,
        .enable_sideways_stats = enable_sideways_stats,
        .enable_golap_compression_mode = enable_golap_compression_mode,
        .enable_lz4par = enable_lz4par,
        .enable_fsst = enable_fsst,
        .execution_mode = execution_mode,
        .load_customer = load_customer,
        .load_lineitem = load_lineitem,
        .load_nation = load_nation,
        .load_orders = load_orders,
        .load_part = load_part,
        .load_partsupp = load_partsupp,
        .load_region = load_region,
        .load_supplier = load_supplier,
        .test = test,
        .verify = verify,
        .output_format = output_format,
    };
}

void validate_loader_options(const LoaderOptions &options)
{
    if (!options.test) {
        if (std::strncmp((const char*)options.input_dirname, "", 1) == 0)
        {
            std::cerr << "dirname should be given" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (std::strncmp((const char*)options.output_files, "", 1) == 0)
        {
            std::cerr << "output_files should be given" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "dirname: " << options.input_dirname << std::endl;
    std::cout << "files: " << options.output_files << std::endl;
    std::cout << "page_size: " << options.page_size << std::endl;
    std::cout << "enable_column: " << options.enable_column << std::endl;
    std::cout << "load_customer: " << options.load_customer << std::endl;
    std::cout << "load_lineitem: " << options.load_lineitem << std::endl;
    std::cout << "load_supplier: " << options.load_supplier << std::endl;
    std::cout << "load_part: " << options.load_part << std::endl;
    std::cout << "load_partsupp: " << options.load_partsupp << std::endl;
    std::cout << "load_region: " << options.load_region << std::endl;
    std::cout << "load_nation: " << options.load_nation << std::endl;
    std::cout << "enable_golap_compression_mode: " << options.enable_golap_compression_mode << std::endl;
    std::cout << "enable_sideways_stats: " << options.enable_sideways_stats << std::endl;
    {
        const char* mode_str = "legacy";
        switch (options.execution_mode) {
        case ExecutionMode::GIDP:              mode_str = "gidp"; break;
        case ExecutionMode::GIDP_BAM:          mode_str = "gidp+bam"; break;
        case ExecutionMode::GIDP_BAM_FUSION:   mode_str = "gidp+bam+fusion"; break;
        case ExecutionMode::DATAPATHFUSION:     mode_str = "datapathfusion"; break;
        default: break;
        }
        std::cout << "execution_mode: " << mode_str << std::endl;
    }
    std::cout << "OK" << std::endl;
}

void output_loader_result(const LoaderOptions &options, const LoaderResult &result)
{
    if (options.output_format == OutputFormat::TEXT)
    {
        std::cout << "dirname: " << options.input_dirname << std::endl;
        std::cout << "files: " << options.output_files << std::endl;
        std::cout << "page_size: " << options.page_size << " bytes" << std::endl;
        std::cout << "enable_column: " << options.enable_column << std::endl;
        std::cout << "load_customer: " << options.load_customer << std::endl;
        std::cout << "load_lineitem: " << options.load_lineitem << std::endl;
        std::cout << "load_supplier: " << options.load_supplier << std::endl;
        std::cout << "load_part: " << options.load_part << std::endl;
        std::cout << "load_partsupp: " << options.load_partsupp << std::endl;
        std::cout << "load_region: " << options.load_region << std::endl;
        std::cout << "load_nation: " << options.load_nation << std::endl;
        std::cout << "time: " << result.elapsed_nanoseconds << " ns" << std::endl;
    }
    else if (options.output_format == OutputFormat::JSON)
    {
#if 0
        std::cout
            << "{\n"
            << "  \"transfer_type\": \"" << options.transfer_type_str << "\",\n"
            << "  \"benchmark_type\": \"" << options.benchmark_type_str << "\",\n"
            << "  \"file\": \"" << options.file << "\",\n"
            << "  \"number_of_threads\": " << options.nthreads << ",\n"
            << "  \"io_depth\": " << options.io_depth << ",\n"
            << "  \"io_mutiplicity\": " << options.io_multiplicity << ",\n"
            << "  \"io_size\": " << options.io_size << ",\n"
            << "  \"file_size\": " << options.file_size << ",\n"
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
#endif
    }
}
