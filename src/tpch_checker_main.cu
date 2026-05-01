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
#include <fstream>
#include <filesystem>
#include <future>
#include <vector>
#include <regex>
#include <unordered_map>
#include <string>
#include <string_view>
#include <iostream>

#include <getopt.h>
#include <cstring>

#include "schema/tpch_tables.cuh"

namespace chrono = std::chrono;
namespace fs = std::filesystem;

constexpr size_t KIBI = 1024UL;
constexpr size_t MEBI = 1024UL * 1024UL;
constexpr size_t GIBI = 1024UL * 1024UL * 1024UL;
constexpr size_t TEBI = 1024UL * 1024UL * 1024UL * 1024UL;

struct CheckerOptions
{
    // Reuse 
    // char const *input_dirname;
    // char const *output_dirname;

    std::string input_dirname;
    std::string output_dirname;
    bool dryrun;
    bool verbose;

    char *devname[128];
    size_t ndev;
    std::vector<int> output_fds;

    bool check_customer;
    bool check_lineitem;
    bool check_nation;
    bool check_orders;
    bool check_part;
    bool check_partsupp;
    bool check_region;
    bool check_supplier;
};

struct Metric {
    std::string field_name;
    size_t fields_written = 0;
    size_t fields_written_compressed = 0;
    size_t fields_written_offset_for_compression = 0;
    size_t fields_written_sizcomp_for_compression = 0;
};

struct StatEntry {
    std::string name;
    std::vector<Metric> metrics;
};

struct CheckerStats {
    std::vector<StatEntry> entries;
};

struct CheckerThreadStats {
    ssize_t nlines_processed;
};

template <size_t NumVarCharFields>
struct CheckTask {
    size_t file_id; // 1 to nfiles
    std::array<std::unordered_map<size_t, size_t>, NumVarCharFields> arr_histgram_map;

    std::array<size_t, NumVarCharFields> target_varchar_field_indexes;
};

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

void print_checker_usage()
{
    std::cout
        << "Usage:\n"
           "  checker [OPTIONS]\n"
           "Options:\n"
           "  -i, --inputdir=DIR            input directory\n"
           "  -A, --all                     load all tables\n"
           "  -C, --customer                load customer table\n"
           "  -N, --nation                  load nation table\n"
           "  -R, --region                  load region table\n"
           "  -L, --lineitem                load lineitem table\n"
           "  -O, --orders                  load orders table\n"
           "  -S, --supplier                load supplier table\n"
           "  -P, --part                    load part table\n"
           "  -T, --partsupp                load partsupp table\n"
           "  -h                            display help\n";
}

CheckerOptions checker_tpch_parse_checker_options(int argc, char *const *argv)
{
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
        { "verbose",          optional_argument, NULL, 'v' },
        { 0,        0,                 0,     0  },
    };
    std::string input_dirname("");
    std::string output_dirname("");
    bool check_customer = false;
    bool check_lineitem = false;
    bool check_nation = false;
    bool check_orders = false;
    bool check_part = false;
    bool check_partsupp = false;
    bool check_region = false;
    bool check_supplier = false;

    int opt;
    int longindex;

    //while ((opt = getopt(argc, argv, "e:p:ah")) != EOF)
    while ((opt = getopt_long(argc, argv, "i:o:dACLNOPTRSfh", longopts, &longindex)) != EOF)
    {
        switch (opt)
        {
        case 'i':
        {
            input_dirname = optarg;
            break;
        }
        case 'o':
        {
            output_dirname = optarg;
            break;
        }
        case 'A':
        {
            // bulk load all tables
            check_customer = true;
            check_lineitem = true;
            check_nation = true;
            check_orders = true;
            check_part = true;
            check_partsupp = true;
            check_region = true;
            check_supplier = true;
 
            break;
        }
        case 'C':
        {
            check_customer = true;
            break;
        }
        case 'L':
        {
            check_lineitem = true;
            break;
        }
        case 'N':
        {
            check_nation = true;
            break;
        }
        case 'O':
        {
            check_orders = true;
            break;
        }
        case 'R':
        {
            check_region = true;
            break;
        }
        case 'P':
        {
            check_part = true;
            break;
        }
        case 'T':
        {
            check_partsupp = true;
            break;
        }
        case 'S':
        {
            check_supplier = true;
            break;
        }
        case 'h':
        default:
        {
            print_checker_usage();
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

    return CheckerOptions{
        .input_dirname = input_dirname,
        .output_dirname = output_dirname,
        .check_customer = check_customer,
        .check_lineitem = check_lineitem,
        .check_nation = check_nation,
        .check_orders = check_orders,
        .check_part = check_part,
        .check_partsupp = check_partsupp,
        .check_region = check_region,
        .check_supplier = check_supplier,
    };
}

void validate_checker_options(const CheckerOptions &options)
{
    const char *input_dirname = options.input_dirname.c_str();
    if (std::strncmp(input_dirname, "", 1) == 0)
    {
        std::cerr << "dirname should be given" << std::endl;
        exit(EXIT_FAILURE);
    }
    const char *output_dirname = options.output_dirname.c_str();
    if (std::strncmp(output_dirname, "", 1) == 0)
    {
        std::cerr << "dirname should be given" << std::endl;
        exit(EXIT_FAILURE);
    }
 
    // std::cout << "dirname: " << options.input_dirname << std::endl;
    // std::cout << "load_customer: " << options.load_customer << std::endl;
    // std::cout << "load_lineitem: " << options.load_lineitem << std::endl;
    // std::cout << "load_supplier: " << options.load_supplier << std::endl;
    // std::cout << "load_part: " << options.load_part << std::endl;
    // std::cout << "load_partsupp: " << options.load_partsupp << std::endl;
    // std::cout << "load_region: " << options.load_region << std::endl;
    // std::cout << "load_nation: " << options.load_nation << std::endl;
}

static std::vector<std::string> list_regex_matched_files(const std::string& directory, const std::string& pattern) {
    std::vector<std::string> matched_files;
    std::regex regex_pattern(pattern);

    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                const std::string filename = entry.path().filename().string();
                if (std::regex_match(filename, regex_pattern)) {
#ifdef DEBUG_PRINT
                    std::cout << "Matched file: " << entry.path().string() << std::endl;
#endif
                    matched_files.push_back(entry.path().string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    } catch (const std::regex_error& e) {
        std::cerr << "Regex error: " << e.what() << std::endl;
    }

    return matched_files;
}

static std::vector<std::string> list_input_files(const std::string& directory, const std::string &prefix) {
#ifdef DEBUG_PRINT
    std::cout << "Regex rule: " << "customer\\.tbl\\**" << std::endl;
#endif
    std::stringstream ss;
    ss << prefix;
    ss << ".tbl.?[0-9]*";
    auto regex_pattern = ss.str();

    //std::cout << "Regex rule: " << regex_pattern << std::endl;
    //return count_regex_matched_files(directory, "customer.tbl.?[0-9]*");
    return list_regex_matched_files(directory, regex_pattern);
}

static std::vector<std::string_view> tpch_split_row(std::string_view row, char delimiter = '|') {
    std::vector<std::string_view> fields;
    size_t start = 0;
    while (start < row.size()) {
        size_t end = row.find(delimiter, start);
        if (end == std::string_view::npos) {
            fields.emplace_back(row.substr(start));
            break;
        }
        fields.emplace_back(row.substr(start, end - start));
        start = end + 1;
    }
    return fields;
}

template <size_t NumVarCharFields>
static CheckerThreadStats collect_stats_input_files_stats(
    CheckTask<NumVarCharFields>& task, const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open file.\n";
        return { -1 };
    }

    //CheckTask& task, std::vector<std::string>& paths,

    std::string line;
    ssize_t lines_processed = 0;
    while (std::getline(file, line)) {
        size_t f = 0;
        size_t vf = 0;
        // char *valuechar;
        auto fields = tpch_split_row(line);
        for (const auto& part : fields) {
            if (f == task.target_varchar_field_indexes[vf]) {
                size_t len = part.length();
                auto it = task.arr_histgram_map[vf].find(len);
                if (it != task.arr_histgram_map[vf].end()) {
                    it->second += 1;
                } else {
                    task.arr_histgram_map[vf][len] = 1;
                }
                vf++;
            }
            f++;
        }
        lines_processed++;
    }
 
    return { lines_processed };
}

template <size_t NumVarCharFields>
std::array<std::unordered_map<size_t, size_t>, NumVarCharFields> merge_histograms(
    const std::array<std::vector<std::unordered_map<size_t, size_t>>, NumVarCharFields>& histograms)
{
    std::array<std::unordered_map<size_t, size_t>, NumVarCharFields> merged;
    for (size_t i = 0; i < NumVarCharFields; i++) {
        for (const auto& map : histograms[i]) {
            for (const auto& [key, value] : map) {
                merged[i][key] += value;
            }
        }
    }
    return merged;
}

template <size_t NumVarCharFields, size_t NFields, typename BulkLoadTargetField, typename TableType>
void check_files(CheckerOptions &options, TableType table, const std::vector<std::string>& input_files,
                 const std::array<size_t, NFields>& field_sizes,
                 const std::array<enum rec_type, NFields> &field_types,
                 const std::array<BulkLoadTargetField, NFields>& target_fields,
                 const std::array<size_t, NumVarCharFields>& target_varchar_field_indexes,
                 StatEntry &stat_entry)
{
    std::vector<CheckTask<NumVarCharFields>> tasks;
    std::vector<std::future<CheckerThreadStats>> futures;
    std::vector<std::thread> threads;

    size_t ntasks = input_files.size();
    for (size_t i = 0; i < ntasks; i++) {
        CheckTask<NumVarCharFields> task;
        task.file_id = i;
        task.arr_histgram_map = std::array<std::unordered_map<size_t, size_t>, NumVarCharFields>{};
        for (size_t j = 0; j < NumVarCharFields; j++) {
            task.arr_histgram_map[j] = std::unordered_map<size_t, size_t>{};
        }
        task.target_varchar_field_indexes = target_varchar_field_indexes;
        tasks.push_back(task);
    }

    // Check the file contents
    for (size_t i = 0; i < ntasks; i++) {
        auto &filename = input_files[i];

        auto check_func = static_cast<CheckerThreadStats(*)(CheckTask<NumVarCharFields>&, const std::string&)>(
            &collect_stats_input_files_stats<NumVarCharFields>);

        std::packaged_task<CheckerThreadStats()> task(
            std::bind(check_func, std::ref(tasks[i]), std::cref(filename))
        );

        futures.push_back(task.get_future());
        threads.emplace_back(std::move(task));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::array<std::vector<std::unordered_map<size_t, size_t>>, NumVarCharFields> histograms;
    ssize_t count_sum = 0;
    for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        ssize_t count = result.nlines_processed;
        if (count == -1) {
            std::cerr << "Error processing file: " << tasks[i].file_id << std::endl;
            std::exit(EXIT_FAILURE);
        }
        count_sum += count;

        for (size_t j = 0; j < NumVarCharFields; j++) {
            histograms[j].push_back(tasks[i].arr_histgram_map[j]);
        }
    }
    auto merged_histograms = merge_histograms(histograms);
    

    std::string table_name = std::string(TPCH::common::table_name(table));
    size_t vf = 0;
    for (auto &merged : merged_histograms) {
        auto field_name = TPCH::common::metric_field_name(target_fields[target_varchar_field_indexes[vf]]);

        std::vector<std::pair<size_t, size_t>> sorted(merged.begin(), merged.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });


        std::stringstream ss;
        ss << table_name << "_" << field_name;
        std::string output_filename = options.output_dirname + "/" + ss.str();

        /* Confirming the parent directory exists */
        fs::path p(output_filename);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }

        std::ofstream ofs(output_filename);
        if (!ofs) {
            std::cerr << "Failed to open an output file: " << output_filename << "\n";
            exit(EXIT_FAILURE);
        }


        for (const auto& [key, count] : sorted) {
            // std::cout << key << " " << count << std::endl;
            ofs << key << " " << count << "\n";
        }
        vf++;
    }

    std::cout << count_sum << " lines processed." << std::endl;
}

template <size_t NumVarCharFields, size_t NFields, typename BulkLoadTargetField, typename TableType>
void checker_check_file__(
    TableType table,
    const std::array<enum rec_type, NFields> &field_types,
    const std::array<size_t, NFields>& field_sizes,
    const std::array<BulkLoadTargetField, NFields>& target_fields,
    const std::array<size_t, NumVarCharFields>& target_varchar_field_indexes,
    CheckerOptions &options, CheckerStats &stats)
{
    std::string s = std::string(TPCH::common::table_name(table));
    std::transform(s.begin(), s.end(), s.begin(), [](char c) { return std::toupper(c); });
    auto table_name = s;
    StatEntry stat_entry;
    stat_entry.name = table_name;
     
    // std::string table_name = TPCH::common::table_name_upper(table)
    // Load the ddate table
    std::cout << "Loading " << table_name << " table..." << std::endl;

    /* buffer and per-field pagids */
    std::string col = TPCH::common::table_name(table);
    std::stringstream ss_basedir;
    ss_basedir << options.input_dirname << "/" << col;
    auto basedir = ss_basedir.str();
    std::vector<std::string> input_files = list_input_files(basedir, col);
    if (input_files.empty()) {
        std::cerr << "No input files found for table: " << col << std::endl;
        exit(EXIT_FAILURE);
    }

    check_files(
        options, table, input_files,
        field_sizes, field_types,
        target_fields,
        target_varchar_field_indexes,
        stat_entry);

    // std::cout << table_name << " table loaded successfully." << std::endl;
    //stats.entries.push_back(stat_entry);
}

void checker_check_file_(TPCH::common::Table table, CheckerOptions &options, CheckerStats &stats)
{
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        checker_check_file__(
            table,
            TPCH::common::fmt.customer_field_types,
            TPCH::common::fmt.customer_field_sizes,
            TPCH::common::fmt.customer_fields,
            TPCH::common::fmt.customer_varchar_field_indexes,
            options, stats);
        break;
     case TPCH::common::Table::NATION:
        checker_check_file__(
            table,
            TPCH::common::fmt.nation_field_types,
            TPCH::common::fmt.nation_field_sizes,
            TPCH::common::fmt.nation_fields,
            TPCH::common::fmt.nation_varchar_field_indexes,
            options, stats);
        break;
    case TPCH::common::Table::REGION:
        checker_check_file__(
            table,
            TPCH::common::fmt.region_field_types,
            TPCH::common::fmt.region_field_sizes,
            TPCH::common::fmt.region_fields,
            TPCH::common::fmt.region_varchar_field_indexes,
            options, stats);
        break;
    default:
        break;
    }
    return;
}

void checker_check_nation(CheckerOptions &options, CheckerStats &stats)
{
    checker_check_file_(TPCH::common::Table::NATION, options, stats);
}

void checker_check_region(CheckerOptions &options, CheckerStats &stats)
{
    checker_check_file_(TPCH::common::Table::REGION, options, stats);
}

void checker_check_customer(CheckerOptions &options, CheckerStats &stats)
{
    checker_check_file_(TPCH::common::Table::CUSTOMER, options, stats);
}

int main(int argc, char *const *argv)
{
    std::cerr << "checker" << std::endl;

    CheckerOptions options = checker_tpch_parse_checker_options(argc, argv);

    validate_checker_options(options);

    std::cout << "options.input_dirname: " << options.input_dirname << std::endl;
    std::cout << "options.output_dirname: " << options.output_dirname << std::endl;

    // open_output_files(options, options.output_fds);

    struct CheckerStats stats;

    if (options.check_nation)
    {
        checker_check_nation(options, stats);
    }
    if (options.check_region)
    {
        checker_check_region(options, stats);
    }
    if (options.check_customer)
    {
        checker_check_customer(options, stats);
    }
    if (options.check_lineitem)
    {
        // next_xtn_id = loader_load_lineitem(options, metadata, next_xtn_id, stats);
    }
    if (options.check_orders)
    {
        // next_xtn_id = loader_load_orders(options, metadata, next_xtn_id, stats);
    }
    if (options.check_supplier)
    {
        //next_xtn_id = loader_load_supplier(options, metadata, next_xtn_id, stats);
        // metadata_print(metadata);
    }
    if (options.check_part)
    {
        //next_xtn_id = loader_load_supplier(options, metadata, next_xtn_id, stats);
        // metadata_print(metadata);
    }
    if (options.check_partsupp)
    {
        //next_xtn_id = loader_load_supplier(options, metadata, next_xtn_id, stats);
        // metadata_print(metadata);
    }

    //loader_stats_print(options.page_size, options.compress, stats);
}
