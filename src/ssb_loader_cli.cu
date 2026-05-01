#pragma once

#include <getopt.h>

#include <cstring>

#include "common/ssb_loader.cu"

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
           "  ssbloader [OPTIONS]\n"
           "Options:\n"
           "  -i, --inputdir=DIR            input directory\n"
           "  -d, --outputfiles=FILE        output files (comma-separated regular or block device files) \n"
           "  -A, --all                     load all tables\n"
           "  -L, --lineorder               load lineorder table\n"
           "  -C, --customer                load customer table\n"
           "  -S, --supplier                load supplier table\n"
           "  -P, --part                    load part table\n"
           "  -D, --ddate                   load ddate table\n"
           "  -x, --execmode=MODE           execution mode [gidp, gidp+bam, gidp+bam+fusion, datapathfusion, uncomp]\n"
           "  -G, --golapcomp              load as golap compression mode (legacy)\n"
           "  -s, --sidewaysstats           enable sideways statistics (legacy)\n"
           "  -l, --lbc_enabled             enable lbc mode (for PiG)\n"
           "  -u, --lbc_numclusters=N       number of varchar clusters for lbc mode (default:128)\n"
           "  -c, --compress                enable compression\n"
           "  -n, --dryrun                  enable dryrun (no data write)\n"
           "  -p, --pagesize=SIZE           page size (default:1M) K,M,G,T can be used as a unit\n"
           "  -f, --verify                  verify after loading\n"
           "  -v, --verbose                 verbose output\n"
           "  -h                            display help\n";
}

LoaderOptions ssb_parse_loader_options(int argc, char *const *argv)
{
    size_t page_size = 1024 * 1024;
    OutputFormat output_format = OutputFormat::TEXT;
    struct option longopts[] = {
        { "all",                    no_argument, NULL, 'A' },
        { "lineorder",              no_argument, NULL, 'L' },
        { "customer",               no_argument, NULL, 'C' },
        { "supplier",               no_argument, NULL, 'S' },
        { "part",                   no_argument, NULL, 'P' },
        { "ddate",                  no_argument, NULL, 'D' },
        { "inputdir",         required_argument, NULL, 'i' },
        { "outputfiles",      required_argument, NULL, 'd' },
        { "execmode",         required_argument, NULL, 'x' },
        { "golap_compression",      no_argument, NULL, 'G' },
        { "sideways_stats",         no_argument, NULL, 's' },
        { "lbc_enabled",            no_argument, NULL, 'l' },
        { "lbc_numclusters",  required_argument, NULL, 'u' },
        { "compress",               no_argument, NULL, 'c' },
        { "test",             optional_argument, NULL, 't' },
        { "dryrun",                 no_argument, NULL, 'n' },
        { "pagesize",         required_argument, NULL, 'p' },
        { "verify",                 no_argument, NULL, 'f' },
        { "verbose",                no_argument, NULL, 'v' },
        { "outputformat",    required_argument, NULL, 'e' },
        { 0,        0,                 0,     0  },
    };
    const char *dirname = "";
    const char *filname = "";
    bool load_lineorder = false;
    bool load_customer = false;
    bool load_supplier = false;
    bool load_part = false;
    bool load_ddate = false;
    bool enable_sideways_stats = false;
    bool enable_golap_compression_mode = false;
    bool enable_fsst = false;
    ExecutionMode execution_mode = ExecutionMode::LEGACY;
    bool compress = false;
    bool dryrun = false;
    bool verbose = false;
    bool test = false;
    bool verify = false;
    bool lbc_enabled = false;
    size_t lbc_num_varchar_clusters = 128;

    int opt;
    int longindex;

    while ((opt = getopt_long(argc, argv, "i:d:e:p:u:x:cndvACLPSDstfhlGZ", longopts, &longindex)) != EOF)
    {
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
            case 'A':
            {
                load_lineorder = true;
                load_customer = true;
                load_supplier = true;
                load_part = true;
                load_ddate = true;
                break;
            }
            case 'L':
            {
                load_lineorder = true;
                break;
            }
            case 'C':
            {
                load_customer = true;
                break;
            }
            case 'S':
            {
                load_supplier = true;
                break;
            }
            case 'P':
            {
                load_part = true;
                break;
            }
            case 'D':
            {
                load_ddate = true;
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
                } else if (strcmp(optarg, "uncomp") == 0) {
                    execution_mode = ExecutionMode::UNCOMP;
                } else {
                    std::cerr << "Unknown execution mode: " << optarg << std::endl;
                    std::cerr << "Valid modes: gidp, gidp+bam, gidp+bam+fusion, datapathfusion, uncomp" << std::endl;
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

    // -x execution mode overrides -G/-s flags
    switch (execution_mode) {
    case ExecutionMode::GIDP:
    case ExecutionMode::GIDP_BAM:
        enable_golap_compression_mode = true;
        enable_sideways_stats = true;
        compress = true;
        break;
    case ExecutionMode::GIDP_BAM_FUSION:
        enable_golap_compression_mode = true;
        enable_sideways_stats = true;
        compress = true;
        break;
    case ExecutionMode::DATAPATHFUSION:
        enable_sideways_stats = true;
        enable_fsst = true;
        compress = true;
        break;
    case ExecutionMode::UNCOMP:
        enable_golap_compression_mode = false;
        enable_sideways_stats = true;
        compress = false;
        break;
    case ExecutionMode::LEGACY:
        break; // use -G/-s/-c flags as-is
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
        .enable_varchar_to_fixedchar = false,
        .enable_dict_encoding = false,
        .enable_sideways_stats = enable_sideways_stats,
        .enable_golap_compression_mode = enable_golap_compression_mode,
        .enable_lz4par = false,
        .enable_fsst = enable_fsst,
        .execution_mode = execution_mode,
        .load_lineorder = load_lineorder,
        .load_customer = load_customer,
        .load_supplier = load_supplier,
        .load_part = load_part,
        .load_ddate = load_ddate,
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
    std::cout << "load_lineorder: " << options.load_lineorder << std::endl;
    std::cout << "load_customer: " << options.load_customer << std::endl;
    std::cout << "load_supplier: " << options.load_supplier << std::endl;
    std::cout << "load_part: " << options.load_part << std::endl;
    std::cout << "load_ddate: " << options.load_ddate << std::endl;
    std::cout << "enable_golap_compression_mode: " << options.enable_golap_compression_mode << std::endl;
    std::cout << "enable_sideways_stats: " << options.enable_sideways_stats << std::endl;
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
        std::cout << "load_lineorder: " << options.load_lineorder << std::endl;
        std::cout << "load_customer: " << options.load_customer << std::endl;
        std::cout << "load_supplier: " << options.load_supplier << std::endl;
        std::cout << "load_part: " << options.load_part << std::endl;
        std::cout << "load_ddate: " << options.load_ddate << std::endl;
        std::cout << "time: " << result.elapsed_nanoseconds << " ns" << std::endl;
    }
}
