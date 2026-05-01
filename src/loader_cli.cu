#pragma once

#include <getopt.h>

#include <cstring>

#include "common/loader.cu"

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

void print_loader_usage()
{
    std::cout
        << "Usage:\n"
           "  loader [OPTIONS]\n"
           "Options:\n"
           "  -i, --inputdir=DIR            input directory\n"
           "  -o, --outputfiles=FILE        output files (commma-separated regular or block device files) \n"
           "  -A, --all                     load all tables\n"
           "  -L, --lineorder               load lineorder table\n"
           "  -S, --supplier                load supplier table\n"
           "  -P, --part                    load part table\n"
           "  -C, --customer                load customer table\n"
           "  -D, --ddate                   load ddate table\n"
           "  -c, --compress                enable compression. \n"
           "  -d, --dryrun                  enable dryrun (no data write). \n"
           "  -p, --pagesize=SIZE           page size (default:1M) K,M,G,T can be used as an unit \n"
           "  -h                            display help\n";
}

LoaderOptions parse_loader_options(int argc, char *const *argv)
{
    size_t page_size = 1024 * 1024;
    OutputFormat output_format = OutputFormat::TEXT;
    struct option longopts[] = {
        { "all",                no_argument, NULL, 'A' },
        { "customer",           no_argument, NULL, 'C' },
        { "lineorder",          no_argument, NULL, 'L' },
        { "part",               no_argument, NULL, 'P' },
        { "supplier",           no_argument, NULL, 'S' },
        { "ddate",              no_argument, NULL, 'D' },
        { "inputdir",     required_argument, NULL, 'i' },
        { "outputfiles",  required_argument, NULL, 'd' },
        { "compress",     optional_argument, NULL, 'c' },
        { "test",         optional_argument, NULL, 't' },
        { "dryrun",       optional_argument, NULL, 'n' },
        { "scale_factor", optional_argument, NULL, 's' },
        { "pagesize",     optional_argument, NULL, 'p' },
        { "verbose",      optional_argument, NULL, 'v' },
        { "outputformat", optional_argument, NULL, 'e' },
        { 0,        0,                 0,     0  },
    };
    const char *dirname = "";
    const char *filname = "";
    bool load_lineorder = false;
    bool load_supplier = false;
    bool load_part = false;
    bool load_customer = false;
    bool load_ddate = false;
    bool test = false;
    bool compress = false;
    bool dryrun = false;
    bool verbose = false;
    size_t scale_factor = 1;

    int opt;
    int longindex;

    //while ((opt = getopt(argc, argv, "e:p:ah")) != EOF)
    while ((opt = getopt_long(argc, argv, "i:d:e:p:s:cdvALPSDth", longopts, &longindex)) != EOF)
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
        case 'v':
        {
            verbose = true;
            break;
        }
        case 's':
        {
            scale_factor = atoi(optarg);
            if (scale_factor <= 0 ||
                (scale_factor != 1 && scale_factor != 10 && scale_factor != 100 && scale_factor != 1000))
            {
                std::cerr << "invalid scale factor " << optarg << std::endl;
                exit(EXIT_FAILURE);
            }
            break;
        }
        case 'A':
        {
            // bulk load all tables
            load_lineorder = true;
            load_supplier = true;
            load_part = true;
            load_customer = true;
            load_ddate = true;
            break;
        }
        case 'C':
        {
            load_customer = true;
            break;
        }
        case 'L':
        {
            load_lineorder = true;
            break;
        }
        case 'P':
        {
            load_part = true;
            break;
        }
        case 'S':
        {
            load_supplier = true;
            break;
        }
        case 'D':
        {
            load_ddate = true;
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

    return LoaderOptions{
        .input_dirname = dirname,
        .output_files = filname,
        .compress = compress,
        .dryrun = dryrun,
        .verbose = verbose,
        .scale_factor = scale_factor,
        .page_size = page_size,
        .load_lineorder = load_lineorder,
        .load_supplier = load_supplier,
        .load_part = load_part,
        .load_customer = load_customer,
        .load_ddate = load_ddate,
        .test = test,
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
    std::cout << "load_lineorder: " << options.load_lineorder << std::endl;
    std::cout << "load_supplier: " << options.load_supplier << std::endl;
    std::cout << "load_part: " << options.load_part << std::endl;
    std::cout << "load_customer: " << options.load_customer << std::endl;
    std::cout << "load_ddate: " << options.load_ddate << std::endl;
    // std::cout << "output_format: " << options.output_format << std::endl;
    std::cout << "OK" << std::endl;
}

void output_loader_result(const LoaderOptions &options, const LoaderResult &result)
{
    if (options.output_format == OutputFormat::TEXT)
    {
        std::cout << "dirname: " << options.input_dirname << std::endl;
        std::cout << "files: " << options.output_files << std::endl;
        std::cout << "page_size: " << options.page_size << " bytes" << std::endl;
        std::cout << "load_lineorder: " << options.load_lineorder << std::endl;
        std::cout << "load_supplier: " << options.load_supplier << std::endl;
        std::cout << "load_part: " << options.load_part << std::endl;
        std::cout << "load_customer: " << options.load_customer << std::endl;
        std::cout << "load_ddate: " << options.load_ddate << std::endl;
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
