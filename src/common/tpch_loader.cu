#pragma once

#include <fcntl.h>
#include <linux/fs.h>
#include <numa.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <barrier>
#include <cassert>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <future>
#include <functional>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <queue>
#include <random>
#include <thread>
#include <utility>
#include <vector>
#include <ranges>
#include <regex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <span>
#include <memory_resource>


#include "cpu_usage.cu"
#include "page.cu"
#include "tpch_tables.cuh"
#include "pack.cu"
#include "common.cu"

#include "common/pag.h"
#include "common/rec.h"
#include "common/xtn.h"
#include "common/filter.cuh"
#include "common/compression.cuh"
#include "common/fsst_host.h"
#include "fsst.h"

#include <lz4.h>
#include <zlib.h>
#include <snappy.h>

// #define DEBUG_PRINT

namespace chrono = std::chrono;
namespace fs = std::filesystem;

struct ZlibContext {
    z_stream strm;
};

#if 0
const size_t KIBI = 1024UL;
const size_t MEBI = 1024UL * 1024UL;
const size_t GIBI = 1024UL * 1024UL * 1024UL;
const size_t TEBI = 1024UL * 1024UL * 1024UL * 1024UL;
#endif

constexpr size_t ITEMS_PER_THREAD = 4;
constexpr size_t BLOCK_THREADS = 128;
constexpr size_t NUM_ITEMS_PER_TILE = ITEMS_PER_THREAD * BLOCK_THREADS;


#if 0
enum class OutputFormat
{
    TEXT,
    JSON,
};
#endif

enum class ExecutionMode {
    LEGACY = 0,         // -x not specified (legacy -G/-s flags)
    GIDP,               // GPU In-Data-Path
    GIDP_BAM,           // GIDP + BAM
    GIDP_BAM_FUSION,    // GIDP + BAM + device-side decompression
    DATAPATHFUSION,      // Data Path Fusion
};

struct LoaderOptions
{
    // Reuse
    char const *input_dirname;
    char const *output_files;
    char *output_filename_orig;  /* for free memory */

    bool lbc_enabled;
    size_t lbc_num_varchar_clusters;
    size_t _lbc_num_varchar_clusters_original;
    bool compress;
    bool dryrun;
    bool verbose;
    size_t scale_factor;

    char *devname[128];
    size_t ndev;
    std::vector<int> output_fds;

    size_t page_size;
    size_t buffer_size_per_field;


    /* if enable_column is false, then
     * the rowstore is used.
     * To enable this option,
     * -r option is required.
     */
    bool enable_column;

    bool enable_varchar_to_fixedchar;
    bool enable_dict_encoding;
    bool enable_sideways_stats;
    bool enable_golap_compression_mode;
    bool enable_lz4par;
    bool enable_fsst;
    ExecutionMode execution_mode;

    bool load_customer;
    bool load_lineitem;
    bool load_nation;
    bool load_orders;
    bool load_part;
    bool load_partsupp;
    bool load_region;
    bool load_supplier;

    bool test;
    bool verify;

    OutputFormat output_format;
};

std::array<CompressionMethod, TPCH::common::TPCH_MAX_NFIELDS> golap_compression_methods = {};

struct Metric {
    std::string field_name;
    size_t fields_written = 0;
    size_t fields_written_compressed = 0;
    size_t fields_written_offset_for_compression = 0;
    size_t fields_written_sizcomp_for_compression = 0;
    CompressionMethod compression_method = CompressionMethod::NONE;
};

struct StatEntry {
    std::string name;
    std::vector<Metric> metrics;
};

struct LoaderStats {
    std::vector<StatEntry> entries;
};

template <size_t N>
struct LoaderThreadStats {
    ssize_t nlines_processed;
    std::array<size_t, N> nios;
    std::array<size_t, N> nwritten_sectors;
};

template <size_t NBufs, size_t NumRecBufs>
struct TPCHLoaderThreadStats {
    ssize_t nlines_processed;
    std::array<size_t, NBufs> nios;
    std::array<size_t, NBufs> nwritten_sectors;
    //std::array<size_t, NBufs> nwritten_sectors_data;
    std::array<uint32_t, NBufs> max_nrecs_in_page {};
    std::array<size_t, NumRecBufs> nrecs_inserted_total {};
};

struct LoaderResult
{
    size_t nalloc;
    size_t nios;
    int64_t elapsed_nanoseconds;
    CpuUsage cpu_usage;
};

struct LoadResult {
    bool success;
    std::uint32_t next_page_id;
};

template <size_t NFields, size_t NumVarCharFields>
struct SamplingResult {
    ssize_t nlines;
    std::array<size_t, NumVarCharFields> min_lens;
    std::array<size_t, NumVarCharFields> max_lens;
    std::array<size_t, NFields> npages_estimated;
    std::array<std::vector<size_t>, NumVarCharFields> npages_estimated_varchar_per_cluster;
    /* GOLAP: per-field, per-codec compressed sizes from this thread's sample page.
     * golap_compressed_page_sizes[field][codec_index]. Empty if non-GOLAP. */
    std::array<std::vector<size_t>, NFields> golap_compressed_page_sizes;

#if 1
    SamplingResult() = default;
    /* Move constructor */
    SamplingResult(SamplingResult&&) = default;
    SamplingResult& operator=(SamplingResult&&) = default;
    SamplingResult(const SamplingResult&) = delete;
    SamplingResult& operator=(const SamplingResult&) = delete;

    SamplingResult(ssize_t n,
                   std::array<size_t, NumVarCharFields> min_l,
                   std::array<size_t, NumVarCharFields> max_l,
                   std::array<size_t, NFields> np,
                   std::array<std::vector<size_t>, NumVarCharFields> np_vc,
                   std::array<std::vector<size_t>, NFields> golap_cps = {})
        : nlines(n),
          min_lens(min_l),
          max_lens(max_l),
          npages_estimated(np),
          npages_estimated_varchar_per_cluster(std::move(np_vc)),
          golap_compressed_page_sizes(std::move(golap_cps))
    {}
#endif
};

// N: number of fields for column store
template <size_t N>
struct TableLoadTask {
    size_t base_id; // 1 to nproc
    size_t nfiles;
    size_t nskip_lines_curr;
    size_t nlines_to_read;
    size_t start_page_offset;
    size_t start_page_offset64;
    bool compress;
    char *buf_compress;
    std::array<std::vector<uint>, N> offsets;
    uint *offsets_local;
    std::array<std::vector<uint>, N> compressed_page_sizes;
    // /* For RLE */
    // std::array<std::vector<int32_t>, N> vals;
    // uint *val32_local;
};

template <size_t NFields, size_t NumVarCharFields>
struct SamplingTask {
    std::string &path;
    std::size_t id;
    std::size_t page_size;
    std::span<const enum rec_type, NFields> field_types;
    std::span<const size_t, NFields> field_sizes;
    std::span<const size_t, NumVarCharFields> varchar_field_indexes;
    std::span<const bool, NFields> enable_stats;
    bool enable_lbc;
    size_t lbc_nclusters;
    bool golap_compression_mode;
    bool enable_sideways_stats;
    ExecutionMode execution_mode;
    bool verbose;

    std::barrier<> *barrier;
    struct SamplingTask<NFields, NumVarCharFields> *all_tasks; // Pointer to all tasks for synchronization
    size_t num_tasks;
    std::array<size_t, NumVarCharFields> lbc_min_lens;
    std::array<size_t, NumVarCharFields> lbc_max_lens;
};

#if 0
template <size_t NFields, size_t NumStartXtns, size_t NumVarCharFields>
struct RowLoadTask {
    size_t file_id; // 1 to nfiles
    size_t start_page_id;
    size_t num_buffers_used;
    size_t num_varchar_clusters;
    size_t nfiles;
    size_t nrows;
    std::array<uint64_t, NumStartXtns> start_page_ids;
    bool compress;
    bool dict_encoding;
    bool varchar_to_fixedchar;
    TPCHTableMetadata *metadata;
    char *buf_compress;
    // std::array<std::vector<bool>, NumBuffers> dirtyflags;
    uint *offsets_local;
    // for binpack
    //std::array<std::vector<uint>, NumBuffers> offsets;
    //std::array<std::vector<uint>, NumBuffers> compressed_page_sizes;
    // /* For RLE */
    // std::array<std::vector<int32_t>, N> vals;
    // uint *val32_local;

    std::span<const size_t, NFields> field_sizes;
    std::span<const enum rec_type, NFields> field_types;
    std::span<const size_t, NFields> field_encoded_sizes;
    std::span<const enum rec_type, NFields> field_encoded_types;
    std::span<const size_t, NumVarCharFields> varchar_field_indexes;
    std::span<const std::vector<size_t>, NumVarCharFields> varchar_cluster_thresholds;
    std::array<std::vector<size_t>, NumVarCharFields> arr_vec_npages_for_varchar_clusters;
};
#endif

struct CompressionContext {
    bool enable_golap_compression_mode;
    size_t max_compress_dst_buffer_size;
    ZlibContext *zlib_ctx;
    char *dst_buffer;
    std::array<void*, 3> workspaces;
    uint32_t compressed_size;
    bool inited;
};

struct ClusterContext {
    std::pmr::monotonic_buffer_resource resource;
    std::pmr::vector<std::pmr::string> storage;

    // Recieve shared_pool
    ClusterContext(std::pmr::memory_resource* upstream)
        : resource(upstream), // No initial buffer, use upstream directly
          storage(&resource)
    {}

    /* reset is same function name to unique_ptr->reset, so avoid the name here */
    void recycle() {
        storage.clear();
        resource.release(); // Return memory to shared_pool (available for reuse)

        storage.~vector();
        new (&storage) std::pmr::vector<std::pmr::string>(&resource);
    }
};


/* Basic types for database systems */
struct DateType {
    int32_t value;

    // Comparison for sorting
    auto operator<=>(const DateType&) const = default;

    friend std::ostream& operator<<(std::ostream& os, const DateType& d) {
        os << d.value;
        return os;
    }
};

struct CharType {
    std::string value;
    auto operator<=>(const CharType&) const = default;

    friend std::ostream& operator<<(std::ostream& os, const CharType& d) {
        os << d.value;
        return os;
    }
};

struct VCharType {
    std::string value;
    auto operator<=>(const VCharType&) const = default;

    friend std::ostream& operator<<(std::ostream& os, const VCharType& d) {
        os << d.value;
        return os;
    }
};

struct DecimalType {
    int32_t value;
    auto operator<=>(const DecimalType&) const = default;

    friend std::ostream& operator<<(std::ostream& os, const DecimalType& d) {
        os << d.value;
        return os;
    }
};

struct CharAsInt {
    int32_t value;
    auto operator<=>(const CharAsInt&) const = default;

    friend std::ostream& operator<<(std::ostream& os, const CharAsInt& d) {
        os << d.value;
        return os;
    }
};

using StatsVariant = std::variant<
    ColumnStats<int32_t>,
    ColumnStats<int64_t>,
    ColumnStats<DateType>,
    ColumnStats<DecimalType>,
    ColumnStats<CharType>,
    ColumnStats<VCharType>,
    ColumnStats<CharAsInt>
>;

using StatsHistoryVariant = std::variant<
    std::vector<ColumnStats<int32_t>>,
    std::vector<ColumnStats<int64_t>>,
    std::vector<ColumnStats<DateType>>,
    std::vector<ColumnStats<DecimalType>>,
    std::vector<ColumnStats<CharType>>,
    std::vector<ColumnStats<VCharType>>,
    std::vector<ColumnStats<CharAsInt>>
>;

using SidewayStatsVariant = std::variant<
    std::vector<ColumnStats<int32_t>>,
    std::vector<ColumnStats<int64_t>>,
    std::vector<ColumnStats<DateType>>,
    std::vector<ColumnStats<DecimalType>>,
    std::vector<ColumnStats<CharType>>,
    std::vector<ColumnStats<VCharType>>,
    std::vector<ColumnStats<CharAsInt>>
>;

// TODO: Use variant for different filter types
// using FilterVariant = std::variant<ColumnStats<int32_t>, ColumnStats<int64_t>>;
template <size_t NFields, size_t NumStartXtns, size_t NumVarCharFields, size_t NumFilters, size_t NumSidewaysFilters>
struct ColumnLoadTask {
    size_t file_id; // 1 to nfiles
    size_t num_buffers_used;
    size_t lbc_num_varchar_clusters;
    size_t nfiles;
    size_t base_row_id;
    size_t nrows;
    std::array<uint64_t, NumStartXtns> start_page_ids;
    std::array<uint64_t, NumStartXtns> npages;
    std::array<std::vector<size_t>, NumVarCharFields> start_page_ids_varchar;
    std::array<std::vector<size_t>, NumVarCharFields> npages_varchar;

    bool compress;
    bool enable_lbc;
    bool enable_sideways_stats;
    bool enable_golap_compression_mode;
    bool enable_lz4par;
    bool enable_fsst;
    ExecutionMode execution_mode;
    bool dict_encoding;
    bool varchar_to_fixedchar;
    TPCHTableMetadata *metadata;
    char *buf_compress;
    // std::array<std::vector<bool>, NumBuffers> dirtyflags;
    uint *offsets_local;
    // for binpack
    //std::array<std::vector<uint>, NumBuffers> offsets;
    //std::array<std::vector<uint>, NumBuffers> compressed_page_sizes;
    // /* For RLE */
    // std::array<std::vector<int32_t>, N> vals;
    // uint *val32_local;

    std::span<const size_t, NFields> field_sizes;
    std::span<const enum rec_type, NFields> field_types;
    std::span<const CompressionMethod, NFields> compression_methods;
    //std::span<const size_t, NFields> field_encoded_sizes;
    //std::span<const enum rec_type, NFields> field_encoded_types;
    //std::span<const size_t, NFields> column_sizes;
    std::span<const size_t, NumVarCharFields> varchar_field_indexes;
    std::span<const size_t, NumFilters> filter_field_indexes;
    std::span<const std::vector<size_t>, NumVarCharFields> lbc_varchar_cluster_thresholds;
    std::array<std::vector<size_t>, NumVarCharFields> arr_vec_npages_for_varchar_clusters;

    /* assuming 32bit min-max filter */
    std::array<std::vector<ColumnStats<int32_t>>, NumFilters> arr_vec_filters;

    std::span<const bool, NFields> enable_stats;
    std::array<StatsHistoryVariant, NFields> all_stats_history_per_page;

    /* 32bit integer only for simpilicity. */
    std::array<std::array<std::vector<ColumnStats<int32_t>>, NumSidewaysFilters>, NFields> all_sideways_stats_per_page;
    /* all_sideways_stats_per_page_varchar[varchar_id][clusterid][sideways_stats_idx][k]: */
    std::array<std::vector<std::array<std::vector<ColumnStats<int32_t>>, NumSidewaysFilters>>, NumVarCharFields> all_sideways_stats_per_page_varchar;

    std::array<std::vector<uint64_t>, NFields> nrecs_per_page;
    /* nrecs_per_page_varchar[varchar_id][clusterid][k]: */
    std::array<std::vector<std::vector<uint64_t>>, NumVarCharFields> nrecs_per_page_varchar;
    /* NOTE: nblocks means the number of 512-byte blocks */
    std::array<std::vector<uint32_t>, NFields> compressed_sizes_per_page;
    std::array<std::vector<uint64_t>, NFields> compressed_page_write_offsets;
    std::array<std::vector<std::vector<uint32_t>>, NumVarCharFields> compressed_sizes_per_page_varchar;
    std::array<std::vector<std::vector<uint64_t>>, NumVarCharFields> compressed_page_write_offsets_varchar;
};

static uint64_t binUnpack(uint* out, uint* block_offsets, uint num_entries, uint *decoded_values);
static uint64_t binUnpack64(ulong* out, uint* block_offsets, uint num_entries, ulong *decoded_values);


/* generic data type parser */
int32_t date_to_int32(std::string_view sv) {
    if (sv.size() < 10) return -1;

    int year = 0, month = 0, day = 0;

    std::from_chars(sv.data(), sv.data() + 4, year);
    std::from_chars(sv.data() + 5, sv.data() + 7, month);
    std::from_chars(sv.data() + 8, sv.data() + 10, day);

    return year * 10000LL + month * 100 + day;
}

constexpr std::optional<int32_t> decimal_to_int32(std::string_view sv, int scale = 2) {
    if (sv.empty()) return std::nullopt;

    size_t i = 0;
 
    while (i < sv.size() && sv[i] == ' ') i++;
    if (i == sv.size()) return std::nullopt;

    int32_t sign = 1;
    if (sv[i] == '-') {
        sign = -1;
        i++;
    } else if (sv[i] == '+') {
        i++;
    }

    int32_t result = 0;
    bool seen_dot = false;
    int decimals_processed = 0;
    bool has_digits = false;

    // 3. Parsing digits
    for (; i < sv.size(); ++i) {
        char c = sv[i];

        if (c >= '0' && c <= '9') {
            if (seen_dot) {
                if (decimals_processed < scale) {
                    result = result * 10 + (c - '0');
                    decimals_processed++;
                } else {
                    std::cerr << "[FATAL] Too many decimal places in decimal value: " << sv << std::endl;
                    exit(EXIT_FAILURE);
                }
            } else {
                result = result * 10 + (c - '0');
            }
            has_digits = true;
        } else if (c == '.') {
            if (seen_dot) return std::nullopt; // error: multiple dots
            seen_dot = true;
        } else {
            break; // error on non-numerical character
        }
    }

    if (!has_digits) return std::nullopt;

    // 4. Padding
    // If "711.5" (scale=2), then result = 7115
    while (decimals_processed < scale) {
        result *= 10;
        decimals_processed++;
    }

    return result * sign;
}


/* type definitions for data loader */
template <typename T>
T parse_value(std::string_view sv);

template <> int32_t parse_value<int32_t>(std::string_view sv) {
    int32_t val = 0;
    std::from_chars(sv.data(), sv.data() + sv.size(), val);
    return val;
}

template <> int64_t parse_value<int64_t>(std::string_view sv) {
    int64_t val = 0;
    std::from_chars(sv.data(), sv.data() + sv.size(), val);
    return val;
}

template <> DecimalType parse_value<DecimalType>(std::string_view sv) {
    auto valopt = decimal_to_int32(sv);
    if (!valopt.has_value()) {
        std::cerr << "[FATAL] Failed to parse decimal value: " << sv << std::endl;
        exit(EXIT_FAILURE);
    }
    return { *valopt };
}

template <> DateType parse_value<DateType>(std::string_view sv) {
    return { date_to_int32(sv) };
}

template <> CharType parse_value<CharType>(std::string_view sv) {
    return {std::string(sv)};
}

template <> VCharType parse_value<VCharType>(std::string_view sv) {
    return {std::string(sv)};
}

template <> CharAsInt parse_value<CharAsInt>(std::string_view sv) {
    return { static_cast<int32_t>(static_cast<uint8_t>(sv[0])) };
}

// Helper for static_assert in if constexpr
template<class> inline constexpr bool always_false_v = false;

// Use std::pair for buffering: <Value, RowID>
template <typename T>
using SortPair = std::pair<T, uint64_t>;

using BufferVariant = std::variant<
    std::vector<SortPair<int32_t>>,
    std::vector<SortPair<int64_t>>,
    std::vector<SortPair<DateType>>,
    std::vector<SortPair<DecimalType>>,
    std::vector<SortPair<CharType>>,
    std::vector<SortPair<VCharType>>,
    std::vector<SortPair<CharAsInt>>
>;


std::vector<std::string_view> tpch_split_row(std::string_view row, char delimiter = '|') {
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

void metadata_set_free_page_id(TPCHTableMetadata &metadata, size_t next_page_id)
{
    metadata.free_page_id = next_page_id;
}

void metadata_set_nrows(TPCHTableMetadata &metadata, TPCH::common::Table table, uint64_t num_rows)
{
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_nrows = num_rows;
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_nrows = num_rows;
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_nrows = num_rows;
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_nrows = num_rows;
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_nrows = num_rows;
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_nrows = num_rows;
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_nrows = num_rows;
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_nrows = num_rows;
        break;
    default:
        break;
    }
}

void metadata_set_prefix_sum_chunk_size(TPCHTableMetadata &metadata, TPCH::common::Table table, uint64_t npages)
{
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_prefix_sum_chunk_size = npages;
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_prefix_sum_chunk_size = npages;
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_prefix_sum_chunk_size = npages;
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_prefix_sum_chunk_size = npages;
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_prefix_sum_chunk_size = npages;
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_prefix_sum_chunk_size = npages;
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_prefix_sum_chunk_size = npages;
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_prefix_sum_chunk_size = npages;
        break;
    default:
        break;
    }
}

size_t metadata_set_prefix_sum_nrecs_per_page(TPCHTableMetadata &metadata, TPCH::common::Table table, 
    std::vector<int> &output_fds,
    size_t page_size,
    size_t i,
    const uint64_t start_page_id,
    std::vector<uint64_t>& prefix_sum_nrecs_per_page)
{
    size_t npages_for_prefix_sum = (prefix_sum_nrecs_per_page.size() * sizeof(uint64_t) + page_size - 1) / page_size;

    uint64_t *ptr_base = prefix_sum_nrecs_per_page.data();
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_prefix_sum_start_page_ids[i] = start_page_id;
        metadata.table_customer_prefix_sum_npages[i] = npages_for_prefix_sum;
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_prefix_sum_start_page_ids[i] = start_page_id;
        metadata.table_lineitem_prefix_sum_npages[i] = npages_for_prefix_sum;
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_prefix_sum_start_page_ids[i] = start_page_id;
        metadata.table_orders_prefix_sum_npages[i] = npages_for_prefix_sum;
        // metadata.table_orders_nrecs_per_page_prefix_sum[i] = nrecs_per_page;
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_prefix_sum_start_page_ids[i] = start_page_id;
        metadata.table_part_prefix_sum_npages[i] = npages_for_prefix_sum;
        // metadata.table_part_nrecs_per_page_prefix_sum[i] = nrecs_per_page;
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_prefix_sum_start_page_ids[i] = start_page_id;
        metadata.table_supplier_prefix_sum_npages[i] = npages_for_prefix_sum;
        // metadata.table_supplier_nrecs_per_page_prefix_sum[i] = nrecs_per_page;
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_prefix_sum_start_page_ids[i] = start_page_id;
        metadata.table_partsupp_prefix_sum_npages[i] = npages_for_prefix_sum;
        // metadata.table_partsupp_nrecs_per_page_prefix_sum[i] = nrecs_per_page;
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_prefix_sum_start_page_ids[i] = start_page_id;
        metadata.table_nation_prefix_sum_npages[i] = npages_for_prefix_sum;
        // metadata.table_nation_nrecs_per_page_prefix_sum[i] = nrecs_per_page;
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_prefix_sum_start_page_ids[i] = start_page_id;
        metadata.table_region_prefix_sum_npages[i] = npages_for_prefix_sum;
        // metadata.table_region_nrecs_per_page_prefix_sum[i] = nrecs_per_page;
        break;
    default:
        break;
    }


    void *ptr_write_buf = nullptr;
    if (posix_memalign((void**)&ptr_write_buf, 512, page_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    memset(ptr_write_buf, 0, page_size);
    for (size_t j = 0; j < npages_for_prefix_sum; j++) {
        if (j == npages_for_prefix_sum - 1 && (prefix_sum_nrecs_per_page.size() * sizeof(uint64_t)) % page_size != 0) {
            size_t remaining_bytes = (prefix_sum_nrecs_per_page.size() * sizeof(uint64_t)) - j * page_size;
            memset(ptr_write_buf, 0, page_size);
            memcpy(ptr_write_buf, reinterpret_cast<char*>(ptr_base) + j * page_size, remaining_bytes);
        } else {
            memcpy(ptr_write_buf, reinterpret_cast<char*>(ptr_base) + j * page_size, page_size);
        }
        page_pwrite_host(output_fds, ptr_write_buf, start_page_id + j, page_size);
    }
    free(ptr_write_buf);

    return npages_for_prefix_sum;
}

void metadata_set_page_id(TPCHTableMetadata &metadata, TPCH::common::Table table, int i, uint64_t start_page_id, uint64_t npages)
{
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_start_page_ids[i] = start_page_id;
        metadata.table_customer_npages[i] = npages;
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_start_page_ids[i] = start_page_id;
        metadata.table_lineitem_npages[i] = npages;
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_start_page_ids[i] = start_page_id;
        metadata.table_orders_npages[i] = npages;
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_start_page_ids[i] = start_page_id;
        metadata.table_part_npages[i] = npages;
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_start_page_ids[i] = start_page_id;
        metadata.table_supplier_npages[i] = npages;
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_start_page_ids[i] = start_page_id;
        metadata.table_partsupp_npages[i] = npages;
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_start_page_ids[i] = start_page_id;
        metadata.table_nation_npages[i] = npages;
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_start_page_ids[i] = start_page_id;
        metadata.table_region_npages[i] = npages;
        break;
    default:
        break;
    }
}

void metadata_set_max_nrows_in_page(TPCHTableMetadata &metadata, TPCH::common::Table table, size_t i, uint32_t nrows)
{
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_max_nrows_in_page[i] = nrows;
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_max_nrows_in_page[i] = nrows;
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_max_nrows_in_page[i] = nrows;
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_max_nrows_in_page[i] = nrows;
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_max_nrows_in_page[i] = nrows;
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_max_nrows_in_page[i] = nrows;
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_max_nrows_in_page[i] = nrows;
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_max_nrows_in_page[i] = nrows;
        break;
    default:
        break;
    }
}

void metadata_set_prefix_sum_start_page_ids(TPCHTableMetadata &metadata, TPCH::common::Table table, size_t i, uint64_t start_page_id)
{
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_prefix_sum_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_prefix_sum_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_prefix_sum_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_prefix_sum_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_prefix_sum_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_prefix_sum_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_prefix_sum_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_prefix_sum_start_page_ids[i] = start_page_id;
        break;
    default:
        break;
    }
}

void metadata_set_prefix_sum_npages(TPCHTableMetadata &metadata, TPCH::common::Table table, size_t i, uint64_t start_page_id)
{
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_prefix_sum_npages[i] = start_page_id;
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_prefix_sum_npages[i] = start_page_id;
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_prefix_sum_npages[i] = start_page_id;
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_prefix_sum_npages[i] = start_page_id;
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_prefix_sum_npages[i] = start_page_id;
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_prefix_sum_npages[i] = start_page_id;
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_prefix_sum_npages[i] = start_page_id;
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_prefix_sum_npages[i] = start_page_id;
        break;
    default:
        break;
    }
}

size_t metadata_set_no_compression(TPCHTableMetadata &metadata, TPCH::common::Table table, size_t i) {
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_compression_method[i] = static_cast<uint16_t>(CompressionMethod::NONE);
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_compression_method[i] = static_cast<uint16_t>(CompressionMethod::NONE);
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_compression_method[i] = static_cast<uint16_t>(CompressionMethod::NONE);
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_compression_method[i] = static_cast<uint16_t>(CompressionMethod::NONE);
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_compression_method[i] = static_cast<uint16_t>(CompressionMethod::NONE);
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_compression_method[i] = static_cast<uint16_t>(CompressionMethod::NONE);
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_compression_method[i] = static_cast<uint16_t>(CompressionMethod::NONE);
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_compression_method[i] = static_cast<uint16_t>(CompressionMethod::NONE);
        break;
    default:
        break;
    }
    return 0;
}
size_t metadata_set_compressed_page_sizes(TPCHTableMetadata &metadata, TPCH::common::Table table,
    std::vector<int> &output_fds,
    size_t page_size,
    size_t i, CompressionMethod compression,
    const uint64_t start_page_id,
    std::vector<uint32_t>& compressed_page_sizes)
{
    size_t npages_for_compressed_pages = (compressed_page_sizes.size() * sizeof(uint32_t) + page_size - 1) / page_size;

    uint32_t *ptr_base = compressed_page_sizes.data();
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_compressed_page_sizes_start_page_ids[i] = start_page_id;
        metadata.table_customer_compressed_page_sizes_npages[i] = npages_for_compressed_pages;
        metadata.table_customer_compression_method[i] = static_cast<uint16_t>(compression);
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_compressed_page_sizes_start_page_ids[i] = start_page_id;
        metadata.table_lineitem_compressed_page_sizes_npages[i] = npages_for_compressed_pages;
        metadata.table_lineitem_compression_method[i] = static_cast<uint16_t>(compression);
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_compressed_page_sizes_start_page_ids[i] = start_page_id;
        metadata.table_orders_compressed_page_sizes_npages[i] = npages_for_compressed_pages;
        metadata.table_orders_compression_method[i] = static_cast<uint16_t>(compression);
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_compressed_page_sizes_start_page_ids[i] = start_page_id;
        metadata.table_part_compressed_page_sizes_npages[i] = npages_for_compressed_pages;
        metadata.table_part_compression_method[i] = static_cast<uint16_t>(compression);
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_compressed_page_sizes_start_page_ids[i] = start_page_id;
        metadata.table_supplier_compressed_page_sizes_npages[i] = npages_for_compressed_pages;
        metadata.table_supplier_compression_method[i] = static_cast<uint16_t>(compression);
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_compressed_page_sizes_start_page_ids[i] = start_page_id;
        metadata.table_partsupp_compressed_page_sizes_npages[i] = npages_for_compressed_pages;
        metadata.table_partsupp_compression_method[i] = static_cast<uint16_t>(compression);
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_compressed_page_sizes_start_page_ids[i] = start_page_id;
        metadata.table_nation_compressed_page_sizes_npages[i] = npages_for_compressed_pages;
        metadata.table_nation_compression_method[i] = static_cast<uint16_t>(compression);
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_compressed_page_sizes_start_page_ids[i] = start_page_id;
        metadata.table_region_compressed_page_sizes_npages[i] = npages_for_compressed_pages;
        metadata.table_region_compression_method[i] = static_cast<uint16_t>(compression);
        break;
    default:
        break;
    }

    void *ptr_write_buf = nullptr;
    if (posix_memalign((void**)&ptr_write_buf, 512, page_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    memset(ptr_write_buf, 0, page_size);
    for (size_t j = 0; j < npages_for_compressed_pages; j++) {
        if (j == npages_for_compressed_pages - 1 && (compressed_page_sizes.size() * sizeof(uint32_t)) % page_size != 0) {
            size_t remaining_bytes = (compressed_page_sizes.size() * sizeof(uint32_t)) - j * page_size;
            memset(ptr_write_buf, 0, page_size);
            memcpy(ptr_write_buf, reinterpret_cast<char*>(ptr_base) + j * page_size, remaining_bytes);
        } else {
            memcpy(ptr_write_buf, reinterpret_cast<char*>(ptr_base) + j * page_size, page_size);
        }
        page_pwrite_host(output_fds, ptr_write_buf, start_page_id + j, page_size);
    }
    free(ptr_write_buf);

    return npages_for_compressed_pages;
}

template<typename T>
size_t metadata_set_stats(TPCHTableMetadata &metadata, TPCH::common::Table table, 
    std::vector<int> &output_fds,
    size_t page_size,
    size_t i,
    const uint64_t start_page_id,
    std::vector<Stats<T>>& stats)
{
    size_t npages_for_stats = (stats.size() * sizeof(Stats<T>) + page_size - 1) / page_size;
    Stats<T> *ptr_base = stats.data();
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        // metadata.table_customer_enable_stats[i] = 1;
        metadata.table_customer_nstats[i] = stats.size();
        metadata.table_customer_stats_npages[i] = npages_for_stats;
        metadata.table_customer_stats_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::LINEITEM:
        // metadata.table_lineitem_enable_stats[i] = 1;
        metadata.table_lineitem_nstats[i] = stats.size();
        metadata.table_lineitem_stats_npages[i] = npages_for_stats;
        metadata.table_lineitem_stats_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::ORDERS:
        // metadata.table_orders_enable_stats[i] = 1;
        metadata.table_orders_nstats[i] = stats.size();
        metadata.table_orders_stats_npages[i] = npages_for_stats;
        metadata.table_orders_stats_start_page_ids[i] = start_page_id;
        break;
    case TPCH::common::Table::PART:
        // metadata.table_part_enable_stats[i] = 1;
        metadata.table_part_nstats[i] = stats.size();
        metadata.table_part_stats_start_page_ids[i] = start_page_id;
        metadata.table_part_stats_npages[i] = npages_for_stats;
        break;
    case TPCH::common::Table::SUPPLIER:
        // metadata.table_supplier_enable_stats[i] = 1;
        metadata.table_supplier_nstats[i] = stats.size();
        metadata.table_supplier_stats_start_page_ids[i] = start_page_id;
        metadata.table_supplier_stats_npages[i] = npages_for_stats;
        break;
    case TPCH::common::Table::PARTSUPP:
        // metadata.table_partsupp_enable_stats[i] = 1;
        metadata.table_partsupp_nstats[i] = stats.size();
        metadata.table_partsupp_stats_start_page_ids[i] = start_page_id;
        metadata.table_partsupp_stats_npages[i] = npages_for_stats;
        break;
    case TPCH::common::Table::NATION:
        // metadata.table_nation_enable_stats[i] = 1;
        metadata.table_nation_nstats[i] = stats.size();
        metadata.table_nation_stats_start_page_ids[i] = start_page_id;
        metadata.table_nation_stats_npages[i] = npages_for_stats;
        break;
    case TPCH::common::Table::REGION:
        // metadata.table_region_enable_stats[i] = 1;
        metadata.table_region_nstats[i] = stats.size();
        metadata.table_region_stats_start_page_ids[i] = start_page_id;
        metadata.table_region_stats_npages[i] = npages_for_stats;
        break;
    default:
        break;
    }
    if (npages_for_stats == 0) {
        std::cerr << "[Unexpected] npages_for_stats is 0!"
            << " table=" << static_cast<int>(table)
            << " field=" << i
            << " stats.size()=" << stats.size()
            << std::endl;
        exit(EXIT_FAILURE);
    }

    void *ptr_write_buf = nullptr;
    if (posix_memalign((void**)&ptr_write_buf, 512, page_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    memset(ptr_write_buf, 0, page_size);
    for (size_t j = 0; j < npages_for_stats; j++) {
        if (j == npages_for_stats - 1 && (stats.size() * sizeof(Stats<T>)) % page_size != 0) {
            size_t remaining_bytes = (stats.size() * sizeof(Stats<T>)) - j * page_size;
            memset(ptr_write_buf, 0, page_size);
            memcpy(ptr_write_buf, reinterpret_cast<char*>(ptr_base) + j * page_size, remaining_bytes);
        } else {
            memcpy(ptr_write_buf, reinterpret_cast<char*>(ptr_base) + j * page_size, page_size);
        }
        page_pwrite_host(output_fds, ptr_write_buf, start_page_id + j, page_size);
    }
    free(ptr_write_buf);

    return npages_for_stats;
}

template<typename T>
size_t metadata_set_sideways_stats(TPCHTableMetadata &metadata, TPCH::common::Table table,
    std::vector<int> &output_fds,
    size_t page_size,
    size_t i, size_t j,
    const uint64_t start_page_id,
    std::vector<Stats<T>>& stats)
{
    size_t npages_for_stats = (stats.size() * sizeof(Stats<T>) + page_size - 1) / page_size;
    Stats<T> *ptr_base = stats.data();
    switch (table)
    {
    case TPCH::common::Table::LINEITEM:
        if (i > TPCH::common::kLineitemFieldCount) {
            std::cerr << "[FATAL] i exceeds kLineitemFieldCount!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (j > TPCH::common::kLineitemSidewaysCount) {
            std::cerr << "[FATAL] j exceeds kLineitemSidewaysCount!" << std::endl;
            exit(EXIT_FAILURE);
        }
        metadata.table_lineitem_sideways_nstats[i][j] = stats.size();
        metadata.table_lineitem_sideways_stats_npages[i][j] = npages_for_stats;
        metadata.table_lineitem_sideways_stats_start_page_ids[i][j] = start_page_id;
        break;
    case TPCH::common::Table::ORDERS:
        if (i > TPCH::common::kOrdersFieldCount) {
            std::cerr << "[FATAL] i exceeds kOrdersFieldCount!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (j > TPCH::common::kOrdersSidewaysCount) {
            std::cerr << "[FATAL] j exceeds kOrdersSidewaysCount!" << std::endl;
            exit(EXIT_FAILURE);
        }
        metadata.table_orders_sideways_nstats[i][j] = stats.size();
        metadata.table_orders_sideways_stats_npages[i][j] = npages_for_stats;
        metadata.table_orders_sideways_stats_start_page_ids[i][j] = start_page_id;
        break;
    default:
        std::cerr << "[FATAL] Sideways stats are only supported for LINEITEM and ORDERS tables." << std::endl;
        exit(EXIT_FAILURE);
        break;
    }
    if (npages_for_stats == 0) {
        std::cerr << "[Unexpected] npages_for_stats is 0! (sideways)"
            << " table=" << static_cast<int>(table)
            << " field=" << i
            << " sideways=" << j
            << " stats.size()=" << stats.size()
            << std::endl;
        exit(EXIT_FAILURE);
    }

    void *ptr_write_buf = nullptr;
    if (posix_memalign((void**)&ptr_write_buf, 512, page_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    memset(ptr_write_buf, 0, page_size);
    for (size_t k = 0; k < npages_for_stats; k++) {
        if (k == npages_for_stats - 1 && (stats.size() * sizeof(Stats<T>)) % page_size != 0) {
            size_t remaining_bytes = (stats.size() * sizeof(Stats<T>)) - k * page_size;
            memset(ptr_write_buf, 0, page_size);
            memcpy(ptr_write_buf, reinterpret_cast<char*>(ptr_base) + k * page_size, remaining_bytes);
        } else {
            memcpy(ptr_write_buf, reinterpret_cast<char*>(ptr_base) + k * page_size, page_size);
        }
        page_pwrite_host(output_fds, ptr_write_buf, start_page_id + k, page_size);
    }
    free(ptr_write_buf);

    return npages_for_stats;
}


void metadata_set_no_stats(TPCHTableMetadata &metadata, TPCH::common::Table table, size_t i)
{
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_nstats[i] = 0;
        metadata.table_customer_stats_npages[i] = 0;
        metadata.table_customer_stats_start_page_ids[i] = 0;
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_nstats[i] = 0;
        metadata.table_lineitem_stats_npages[i] = 0;
        metadata.table_lineitem_stats_start_page_ids[i] = 0;
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_nstats[i] = 0;
        metadata.table_orders_stats_npages[i] = 0;
        metadata.table_orders_stats_start_page_ids[i] = 0;
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_nstats[i] = 0;
        metadata.table_part_stats_start_page_ids[i] = 0;
        metadata.table_part_stats_npages[i] = 0;
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_nstats[i] = 0;
        metadata.table_supplier_stats_start_page_ids[i] = 0;
        metadata.table_supplier_stats_npages[i] = 0;
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_nstats[i] = 0;
        metadata.table_partsupp_stats_start_page_ids[i] = 0;
        metadata.table_partsupp_stats_npages[i] = 0;
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_nstats[i] = 0;
        metadata.table_nation_stats_start_page_ids[i] = 0;
        metadata.table_nation_stats_npages[i] = 0;
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_nstats[i] = 0;
        metadata.table_region_stats_start_page_ids[i] = 0;
        metadata.table_region_stats_npages[i] = 0;
        break;
    default:
        break;
    }
}


size_t metadata_set_compression_base_page_id(TPCHTableMetadata &metadata, TPCH::common::Table table,
    std::vector<int> &output_fds,
    size_t page_size,
    int i,
    uint64_t start_page_id,
    std::vector<uint64_t> &base_start_page_ids)
{
    // std::vector<int> &output_fds,
    // size_t page_size,
    // size_t i, size_t j,
    // const uint64_t start_page_id,
    // std::vector<Stats<T>>& stats)

    const uint64_t nbases = base_start_page_ids.size();
    size_t npages_for_bases = (nbases * sizeof(uint64_t) + page_size - 1) / page_size;
    void *ptr_base = base_start_page_ids.data();
    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
        metadata.table_customer_compression_base_start_page_ids[i] = start_page_id;
        metadata.table_customer_compression_nbases[i] = nbases;
        break;
    case TPCH::common::Table::LINEITEM:
        metadata.table_lineitem_compression_base_start_page_ids[i] = start_page_id;
        metadata.table_lineitem_compression_nbases[i] = nbases;
        break;
    case TPCH::common::Table::ORDERS:
        metadata.table_orders_compression_base_start_page_ids[i] = start_page_id;
        metadata.table_orders_compression_nbases[i] = nbases;
        break;
    case TPCH::common::Table::PART:
        metadata.table_part_compression_base_start_page_ids[i] = start_page_id;
        metadata.table_part_compression_nbases[i] = nbases;
        break;
    case TPCH::common::Table::SUPPLIER:
        metadata.table_supplier_compression_base_start_page_ids[i] = start_page_id;
        metadata.table_supplier_compression_nbases[i] = nbases;
        break;
    case TPCH::common::Table::PARTSUPP:
        metadata.table_partsupp_compression_base_start_page_ids[i] = start_page_id;
        metadata.table_partsupp_compression_nbases[i] = nbases;
        break;
    case TPCH::common::Table::NATION:
        metadata.table_nation_compression_base_start_page_ids[i] = start_page_id;
        metadata.table_nation_compression_nbases[i] = nbases;
        break;
    case TPCH::common::Table::REGION:
        metadata.table_region_compression_base_start_page_ids[i] = start_page_id;
        metadata.table_region_compression_nbases[i] = nbases;
        break;
    default:
        break;
    }
    size_t npages = TPCH::nbase_to_npages(nbases, page_size);

    if (npages_for_bases != npages)
    {
        std::cerr << "[Unexpected] npages_for_bases != npages!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (npages_for_bases == 0)
    {
        std::cerr << "[Unexpected] npages_for_bases is 0!" << std::endl;
        exit(EXIT_FAILURE);
    }

    void *ptr_write_buf = nullptr;
    if (posix_memalign((void **)&ptr_write_buf, 512, page_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    memset(ptr_write_buf, 0, page_size);
    for (size_t k = 0; k < npages_for_bases; k++)
    {
        if (k == npages_for_bases - 1 && (nbases * sizeof(uint64_t)) % page_size != 0)
        {
            size_t remaining_bytes = (nbases * sizeof(uint64_t)) - k * page_size;
            memset(ptr_write_buf, 0, page_size);
            memcpy(ptr_write_buf, reinterpret_cast<char *>(ptr_base) + k * page_size, remaining_bytes);
        }
        else
        {
            memcpy(ptr_write_buf, reinterpret_cast<char *>(ptr_base) + k * page_size, page_size);
        }
        page_pwrite_host(output_fds, ptr_write_buf, start_page_id + k, page_size);
    }
    free(ptr_write_buf);

    return npages_for_bases;
}


struct Stats<DateType> columnstats_get_from_page_id(std::span<Stats<DateType>> stats, uint64_t pagid, uint64_t base_pagid)
{
    uint64_t idx = pagid - base_pagid;

    return Stats<DateType> {
        .min_val = stats[idx].min_val,
        .max_val = stats[idx].max_val,
    };
}

static int compress_page_with_lz4(PAG *pag, size_t page_size, char *dst_compressed, size_t max_compressed_size, uint32_t &compressed_size) {
    const char* src = reinterpret_cast<const char*>(pag);
    char* dst = reinterpret_cast<char*>(dst_compressed);

    int comp_len = LZ4_compress_default(src, dst, (int)page_size, (int)max_compressed_size);

    if (comp_len <= 0) return -1;

    size_t actual_len = static_cast<size_t>(comp_len);

    compressed_size = actual_len;

    size_t aligned_len = roundup4096(actual_len);
    if (aligned_len > actual_len) {
        if (aligned_len <= max_compressed_size) {
            std::memset(dst + actual_len, 0, aligned_len - actual_len);
        }
    }

    return 0;
}


// LZ4-PAR: split page into N sub-chunks of 4KiB each, compress independently.
// N = page_size / 4096 (e.g. 256 for 1MiB pages).
// Disk layout:
//   [comp_sizes: N x uint32_t][chunk0][chunk1]...[chunk_{N-1}]
// Each chunk_i = LZ4(src[i*4096 .. (i+1)*4096)).
// compressed_size is set to the total byte count (header + all chunks),
// NOT 4096-aligned (caller handles alignment).
//
// GPU kernel design: 32 threads/warp, each decompresses one 4KiB sub-chunk
// into shared memory (32 * 4KiB = 128KiB).  N/32 iterations per page.
static int compress_page_with_lz4par(PAG *pag, size_t page_size, char *dst_compressed, size_t max_compressed_size, uint32_t &compressed_size) {
    //constexpr size_t CHUNK_SZ = 4096;
    //constexpr size_t CHUNK_SZ = 8192;
    constexpr size_t CHUNK_SZ = 32768;
    const uint32_t n_chunks = (uint32_t)(page_size / CHUNK_SZ);

    const char* src = reinterpret_cast<const char*>(pag);
    char* dst = reinterpret_cast<char*>(dst_compressed);

    // Header: n_chunks x uint32_t compressed sizes.
    const size_t hdr_size = n_chunks * sizeof(uint32_t);
    uint32_t* hdr = reinterpret_cast<uint32_t*>(dst);

    size_t write_pos = hdr_size;

    for (uint32_t i = 0; i < n_chunks; i++) {
        const char* chunk_src = src + (size_t)i * CHUNK_SZ;
        size_t remaining = max_compressed_size - write_pos;
        if (remaining == 0) return -1;

        int comp_len = LZ4_compress_default(
            chunk_src, dst + write_pos,
            (int)CHUNK_SZ, (int)remaining);
        if (comp_len <= 0) return -1;

        hdr[i] = (uint32_t)comp_len;
        write_pos += (size_t)comp_len;
    }

    compressed_size = (uint32_t)write_pos;

    // Zero-pad to 4096 alignment.
    size_t aligned_len = roundup4096(write_pos);
    if (aligned_len > write_pos && aligned_len <= max_compressed_size) {
        std::memset(dst + write_pos, 0, aligned_len - write_pos);
    }

    return 0;
}

// --- Zlib (Deflate) ---
static int compress_page_with_deflate(ZlibContext *ctx, PAG *pag, size_t page_size, char *dst_compressed, size_t max_compressed_size, uint32_t &compressed_size) {
    if (!ctx) return -1;
    if (deflateReset(&ctx->strm) != Z_OK) return -1;

    ctx->strm.next_in = reinterpret_cast<Bytef*>(pag);
    ctx->strm.avail_in = (uInt)page_size;
    ctx->strm.next_out = reinterpret_cast<Bytef*>(dst_compressed);
    ctx->strm.avail_out = (uInt)max_compressed_size;

    int ret = deflate(&ctx->strm, Z_FINISH);
    if (ret != Z_STREAM_END) return -1;

    size_t actual_len = ctx->strm.total_out;

    compressed_size = actual_len;

    size_t aligned_len = roundup4096(actual_len);
    if (aligned_len > actual_len && aligned_len <= max_compressed_size) {
        std::memset(dst_compressed + actual_len, 0, aligned_len - actual_len);
    }

    return 0;
}

// --- Snappy ---
static int compress_page_with_snappy(PAG *pag, size_t page_size, char *dst_compressed, size_t max_compressed_size, uint32_t &compressed_size) {
    const char* src = reinterpret_cast<const char*>(pag);
    char* dst = reinterpret_cast<char*>(dst_compressed);

    const size_t input_len = page_size;
    size_t output_len = max_compressed_size;

    snappy::RawCompress(src, input_len, dst, &output_len);

    if (output_len > max_compressed_size) return -1;

    compressed_size = output_len;

    size_t aligned_len = roundup4096(output_len);
    if (aligned_len > output_len && aligned_len <= max_compressed_size) {
        std::memset(dst + output_len, 0, aligned_len - output_len);
    }

#if 0
    constexpr bool debug_snappy = false;
    if constexpr (debug_snappy) {
        size_t len = output_len;
        std::cout << "compressed_size: " << compressed_size << ", output_len: " << output_len << std::endl;
        printf("Tail bytes: %02x %02x %02x %02x %02x\n",
            dst_compressed[len-5], dst_compressed[len-4], dst_compressed[len-3], dst_compressed[len-2], dst_compressed[len-1]);

        char *decomp_buf = reinterpret_cast<char*>(malloc(page_size));
        size_t decomp_len = 0;
        if (snappy::GetUncompressedLength(dst, output_len, &decomp_len)) {
            std::cout << "[Snappy] Uncompressed length: " << decomp_len << std::endl;
            if (decomp_len == page_size) {
                if (snappy::RawUncompress(dst, output_len, decomp_buf)) {
                    if (std::memcmp(decomp_buf, src, page_size) == 0) {
                        std::cout << "[Snappy] Decompression verified successfully." << std::endl;
                    } else {
                        std::cerr << "[Snappy] Decompressed data does NOT match original!" << std::endl;
                    }
                } else {
                    std::cerr << "[Snappy] RawUncompress failed!" << std::endl;
                }
            } else {
                std::cerr << "[Snappy] Uncompressed length does NOT match page size!" << std::endl;
            }
        }
        free(decomp_buf);
    }
#endif

    return 0;
}

static int compress_int_with_pfor(uint *src, size_t n, size_t page_size, std::array<void*, 3> &workspaces, uint32_t &compressed_size) {
    uint *in_values_array = nullptr;
    constexpr bool debug_pfor = false;
    if (debug_pfor) {
        in_values_array = reinterpret_cast<uint*>(malloc(n * sizeof(uint)));
        memcpy(in_values_array, src, n * sizeof(uint));
    }

    uint *in_values = src;
    uint *out_values = reinterpret_cast<uint*>(workspaces[0]);
    uint *offsets = reinterpret_cast<uint*>(workspaces[1]);
    uint *workspace = reinterpret_cast<uint*>(workspaces[2]);
    memset(out_values, 0, page_size);
    memset(offsets, 0, page_size);
    memset(workspace, 0, page_size);

    /* NOTE: binPack function modifies the in_values */
    size_t nuint = binPack(in_values, out_values, offsets, n, workspace);
    size_t comp_len = nuint * sizeof(uint32_t);

    if (comp_len <= 0) return -1;

    compressed_size = comp_len;

#if 0
    /* NOTE: propagete the actual compressed size */
    /* 0 padding should be done in pagcol_function */
    compressed_size = actual_len;
    size_t aligned_len = roundup4096(actual_len);
    if (aligned_len > actual_len) {
        if (aligned_len <= max_compressed_size) {
            std::memset(dst + actual_len, 0, aligned_len - actual_len);
        }
    }
#endif

    /* NOTE add decompression test here, first. */
    if (debug_pfor) {
        std::cout << "DEBUG: compress_int_with_pfor: n=" << n << ", nuint=" << nuint << ", comp_len=" << comp_len << std::endl;
        uint *decoded_values = reinterpret_cast<uint*>(malloc(n * sizeof(uint)));
        size_t checksum = binUnpack(out_values, offsets, n, decoded_values);
        //binUnpack64Print(out_values, offsets, N);
        for (size_t i = 0; i < n; ++i) {
            if (in_values_array[i] != decoded_values[i]) {
                std::cerr << "Mismatch at index " << i << ": in=" << in_values_array[i]
                    << ", decoded=" << decoded_values[i] << std::endl;
                exit(1);
            }
        }
        free(decoded_values);
        free(in_values_array);
    }

    return 0;
}

static int compress_ulong_with_pfor64(ulong *src, size_t n, size_t page_size, std::array<void*, 3> &workspaces, uint32_t &compressed_size) {
    ulong *in_values_array = nullptr;
    constexpr bool debug_pfor64 = false;
    if (debug_pfor64) {
        in_values_array = reinterpret_cast<ulong*>(malloc(n * sizeof(ulong)));
        memcpy(in_values_array, src, n * sizeof(ulong));
    }

    ulong *in_values = src;
    ulong *out_values = reinterpret_cast<ulong*>(workspaces[0]);
    uint *offsets = reinterpret_cast<uint*>(workspaces[1]);
    uint *workspace = reinterpret_cast<uint*>(workspaces[2]);

    /* NOTE: binPack function modifiess the in_values */
    size_t nulong = binPack64(in_values, out_values, offsets, n, workspace);
    size_t comp_len = nulong * sizeof(uint64_t);

    if (comp_len <= 0) return -1;

    compressed_size = comp_len;
#if 0
    size_t actual_len = static_cast<size_t>(comp_len);
    /* NOTE: propagete the actual compressed size */
    memcpy(dst, src, sizeof(pag_head));
    memcpy(dst + sizeof(pag_head), offsets, (n + 1) * sizeof(uint32_t));
    memcpy(dst + sizeof(pag_head) + (n + 1) * sizeof(uint32_t), out_values, nulong * sizeof(uint64_t));

    /* 0 padding should be done in pagcol_function */
    compressed_size = actual_len;
    size_t aligned_len = roundup4096(actual_len);
    if (aligned_len > actual_len) {
        if (aligned_len <= max_compressed_size) {
            std::memset(dst + actual_len, 0, aligned_len - actual_len);
        }
    }
#endif

    /* NOTE: add decompression test here, first. */
    if (debug_pfor64) {
        for (ulong i = 0; i < 10; ++i) {
            std::cout << "DEBUG: compress_ulong_with_pfor64: out_values[" << i << "]=" << out_values[i] << std::endl;
        }
        ulong *decoded_values = reinterpret_cast<ulong*>(malloc(n * sizeof(ulong)));
        size_t checksum = binUnpack64(out_values, offsets, n, decoded_values);
        std:: cout << "DEBUG: compress_ulong_with_pfor64: checksum=" << checksum << std::endl;
        //binUnpack64Print(out_values, offsets, N);
        for (ulong i = 0; i < n; ++i) {
            if (in_values_array[i] != decoded_values[i]) {
                std::cerr << "Mismatch at index " << i << ": in=" << in_values_array[i]
                    << ", decoded=" << decoded_values[i] << std::endl;
                exit(1);
            }
        }
        for (ulong i = 0; i < 10; ++i) {
            std::cout << "DEBUG: compress_ulong_with_pfor64: out_values[" << i << "]=" << out_values[i] << std::endl;
        }
        free(decoded_values);
        free(in_values_array);
    }

    return 0;
}


int decompress_page_with_lz4(PAG *src_compressed, size_t exact_compressed_size, PAG *dst_original, size_t page_size) {
    const char* src = reinterpret_cast<const char*>(src_compressed);
    char* dst = reinterpret_cast<char*>(dst_original);

    int result = LZ4_decompress_safe(src, dst, (int)exact_compressed_size, (int)page_size);
    if (result == (int)page_size) {
        return 0;
    } else {
        return result;
    }
}

// LZ4-PAR decompression: read header (N x uint32_t), decompress each 4KiB sub-chunk.
// N = page_size / 32768. Reconstructs the original page_size bytes in dst_original.
static int decompress_page_with_lz4par(PAG *src_compressed, size_t exact_compressed_size, PAG *dst_original, size_t page_size) {
    //constexpr size_t CHUNK_SZ = 8192;
    constexpr size_t CHUNK_SZ = 32768;
    const uint32_t n_chunks = (uint32_t)(page_size / CHUNK_SZ);

    const char* src = reinterpret_cast<const char*>(src_compressed);
    char* dst = reinterpret_cast<char*>(dst_original);

    const size_t hdr_size = n_chunks * sizeof(uint32_t);
    if (exact_compressed_size < hdr_size) return -1;

    const uint32_t* hdr = reinterpret_cast<const uint32_t*>(src);
    size_t read_pos = hdr_size;

    for (uint32_t i = 0; i < n_chunks; i++) {
        uint32_t comp_len = hdr[i];
        if (read_pos + comp_len > exact_compressed_size) return -1;

        int result = LZ4_decompress_safe(
            src + read_pos,
            dst + (size_t)i * CHUNK_SZ,
            (int)comp_len, (int)CHUNK_SZ);
        if (result != (int)CHUNK_SZ) return -2;

        read_pos += comp_len;
    }

    return 0;
}

// FSST page verify: decompress all records and compute byte-sum checksum.
// Returns number of records in the page.
static uint32_t fsst_verify_page(
    const char* compressed_buffer, size_t compressed_page_size,
    size_t& out_checksum)
{
    const pag_head* php = (const pag_head*)compressed_buffer;
    uint32_t nalloc = php->nalloc;
    if (nalloc == 0) { out_checksum = 0; return 0; }

    size_t max_output = compressed_page_size * 8;
    std::vector<uint8_t> flat_output(max_output);
    std::vector<size_t> output_lens(nalloc);
    uint32_t nrecs = fsst_decompress_page_cpu(
        (const uint8_t*)compressed_buffer, compressed_page_size,
        flat_output.data(), output_lens.data(), max_output);

    size_t checksum = 0;
    size_t offset = 0;
    for (uint32_t r = 0; r < nrecs; r++) {
        for (size_t m = 0; m < output_lens[r]; m++) {
            checksum += flat_output[offset + m];
        }
        offset += output_lens[r];
    }
    out_checksum = checksum;
    return nrecs;
}

int decompress_page_with_zlib(PAG *src_compressed, size_t exact_compressed_size, PAG *dst_original, size_t page_size) {
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.next_in = reinterpret_cast<Bytef*>(src_compressed);
    strm.avail_in = (uInt)exact_compressed_size;
    strm.next_out = reinterpret_cast<Bytef*>(dst_original);
    strm.avail_out = (uInt)page_size;

    if (inflateInit(&strm) != Z_OK) return false;
    int ret = inflate(&strm, Z_FINISH);
    inflateEnd(&strm);

    if (ret == Z_STREAM_END && strm.total_out == page_size) {
        return 0;
    } else {
        return -1;
    }
}

int decompress_page_with_snappy(PAG *src_compressed, size_t exact_compressed_size, PAG *dst_original, size_t page_size) {
    const char* src = reinterpret_cast<const char*>(src_compressed);
    char* dst = reinterpret_cast<char*>(dst_original);

#if 0
    constexpr bool debug_snappy = true;
    if constexpr (debug_snappy) {
        printf("DEBUG: Read Header bytes: %02x %02x %02x %02x\n",
               (unsigned char)src[0], (unsigned char)src[1], (unsigned char)src[2], (unsigned char)src[3]);
        if (!snappy::IsValidCompressedBuffer(src, exact_compressed_size)) {
            std::cerr << "[ERROR] Data is NOT a valid Snappy stream." << std::endl;
        } else {
            std::cout << "[OK] Stream is valid." << std::endl;
        }
    }
#endif

    size_t uncompressed_len = 0;
    if (!snappy::GetUncompressedLength(src, exact_compressed_size, &uncompressed_len)) {
        return false;
    }
    if (uncompressed_len != page_size) return false;

    // std::cout << "compressed_size: " << exact_compressed_size << ", uncompressed_size: " << uncompressed_len << std::endl;
    if (snappy::RawUncompress(src, exact_compressed_size, dst)) {
        return 0;
    } else {
        return -1;
    }
}

int decompress_page_with_pfor(PAG *src_compressed, size_t exact_compressed_size, PAG *dst_original, char *workspace, size_t page_size) {
    pag_head *srchdr = reinterpret_cast<pag_head*>(src_compressed);
    char *dst = reinterpret_cast<char*>(dst_original);

    uint nalloc_aligned = srchdr->watermark;
    uint *offsets = reinterpret_cast<uint*>(reinterpret_cast<char*>(src_compressed) + sizeof(pag_head));
    uint nblocks = nalloc_aligned / 128;
    uint noffsets = nblocks + 1;
    uint *encoded_values = offsets + noffsets;
    uint *decoded_values = reinterpret_cast<uint*>(workspace);

    binUnpack(encoded_values, offsets, nalloc_aligned, decoded_values);
    memcpy(dst_original, srchdr, sizeof(pag_head));
    memcpy(reinterpret_cast<char*>(dst_original) + sizeof(pag_head), decoded_values, nalloc_aligned * sizeof(uint));
    return 0;
}

int decompress_page_with_pfor64(PAG *src_compressed, size_t exact_compressed_size, PAG *dst_original, char *workspace, size_t page_size) {
    pag_head *srchdr = reinterpret_cast<pag_head*>(src_compressed);
    char *dst = reinterpret_cast<char*>(dst_original);

    uint nalloc_aligned = srchdr->watermark;
    uint *offsets = reinterpret_cast<uint*>(reinterpret_cast<char*>(src_compressed) + sizeof(pag_head));
    uint nblocks = nalloc_aligned / 128;
    uint noffsets = nblocks + 1;
    uint encoded_value_offset = noffsets;
    if ((noffsets & 1) == 0) {
        // pag_head includes 3 integers, so padding is required only when noffsets is even.
        encoded_value_offset++; // align to 8byte
    }
    ulong *encoded_values = reinterpret_cast<ulong*>(offsets + encoded_value_offset);
    ulong *decoded_values = reinterpret_cast<ulong*>(workspace);

    binUnpack64(encoded_values, offsets, nalloc_aligned, decoded_values);
    memcpy(dst_original, srchdr, sizeof(pag_head));
    /* 4-byte padding after pag_head (12B) for 8-byte alignment of int64 values,
       matching the layout produced by pagcol_append_batch_unordered_column_int64. */
    memset(reinterpret_cast<char*>(dst_original) + sizeof(pag_head), 0, 4);
    memcpy(reinterpret_cast<char*>(dst_original) + sizeof(pag_head) + 4, decoded_values, nalloc_aligned * sizeof(ulong));

    return 0;
}

static ZlibContext* compress_init_zlib(int compression_level = Z_DEFAULT_COMPRESSION) {
    ZlibContext* ctx = new(std::nothrow) ZlibContext();
    if (!ctx) return nullptr;

    // z_streamの初期設定
    ctx->strm.zalloc = Z_NULL;
    ctx->strm.zfree = Z_NULL;
    ctx->strm.opaque = Z_NULL;

    // deflateInit で内部メモリ(約256KB)が確保されます
    if (deflateInit(&ctx->strm, compression_level) != Z_OK) {
        delete ctx;
        return nullptr;
    }

    return ctx;
}

static void compress_free_zlib(ZlibContext* ctx) {
    if (ctx) {
        deflateEnd(&ctx->strm); // zlib内部バッファの解放
        delete ctx;             // コンテキスト自体の解放
    }
}

static size_t compress_max_size(CompressionMethod method, size_t page_size) {
    switch (method) {
        case CompressionMethod::SNAPPY:
            return roundup4096(snappy::MaxCompressedLength(page_size));
        case CompressionMethod::DEFLATE:
            return roundup4096(deflateBound(nullptr, (uLong)page_size));
        case CompressionMethod::LZ4:
            return roundup4096(LZ4_compressBound((int)page_size));
        case CompressionMethod::NONE:
            return roundup4096(page_size);
        default:
            fprintf(stderr, "[FATAL] Unsupported compression method for max size calculation\n");
            exit(EXIT_FAILURE);
    }
}

static bool init_compression_context(struct CompressionContext *ctx, const size_t page_size, bool enable_golap_compression_mode) {
    if (ctx == NULL) return false;

    ctx->enable_golap_compression_mode = enable_golap_compression_mode;

    if (enable_golap_compression_mode) {
        std::array<CompressionMethod, 3> methods = {
            CompressionMethod::LZ4,
            CompressionMethod::DEFLATE,
            CompressionMethod::SNAPPY
        };
        size_t buffer_size = 0;
        for (const auto& method : methods) {
            buffer_size = std::max(buffer_size, compress_max_size(method, page_size));
        }
        ctx->max_compress_dst_buffer_size = buffer_size;
    } else {
        /* PiG only depends on LZ4 */
        size_t buffer_size = compress_max_size(CompressionMethod::LZ4, page_size);
        ctx->max_compress_dst_buffer_size = std::max(buffer_size, page_size);

        char *workspace_base = reinterpret_cast<char*>(malloc(3 * page_size));
        ctx->workspaces[0] = reinterpret_cast<uint*>(workspace_base);
        ctx->workspaces[1] = reinterpret_cast<uint*>(workspace_base + page_size);
        ctx->workspaces[2] = reinterpret_cast<uint*>(workspace_base + 2 * page_size);

        /* NOTE: PFOR also uses ctx->dst_buffer. */
    }

    if (posix_memalign((void**)&ctx->dst_buffer, 512, ctx->max_compress_dst_buffer_size) != 0) {
        return false;
    }

    if (enable_golap_compression_mode) {
        ctx->zlib_ctx = compress_init_zlib();
    } else {
        ctx->zlib_ctx = NULL;
    }

    ctx->inited = true;
    return true;
}

static void free_compression_context(struct CompressionContext *ctx) {
    if (ctx == NULL) return;

    ctx->inited = false;
    if (ctx->enable_golap_compression_mode == false) {
        if (ctx->workspaces[0]) {
            free(ctx->workspaces[0]);
            ctx->workspaces[0] = NULL;
            ctx->workspaces[1] = NULL;
            ctx->workspaces[2] = NULL;
        }
    }

    if (ctx->dst_buffer) {
        free(ctx->dst_buffer);
        ctx->dst_buffer = NULL;
    }

    if (ctx->zlib_ctx) {
        compress_free_zlib(ctx->zlib_ctx);
        ctx->zlib_ctx = NULL;
    }
}

static double calculate_decomp_cost(size_t comp_size, size_t original_size, CompressionMethod method) {
    int decomp_throughput_100mbps = 0;
    switch (method) {
        case CompressionMethod::NONE:
            break;
        case CompressionMethod::SNAPPY:
            decomp_throughput_100mbps = static_cast<int>(DecompressionSpeed::SNAPPY);
            break;
        case CompressionMethod::DEFLATE:
            decomp_throughput_100mbps = static_cast<int>(DecompressionSpeed::DEFLATE);
            break;
        case CompressionMethod::LZ4:
            decomp_throughput_100mbps = static_cast<int>(DecompressionSpeed::LZ4);
            break;
        default:
            fprintf(stderr, "[FATAL] Unsupported compression method for cost calculation\n");
            exit(EXIT_FAILURE);
    }

    double comp_ratio = (double)comp_size / (double)original_size;
    size_t assumed_original_size = 100 * 10; // Assuming 100GB in 100MB
    size_t assumed_compressed_size = (size_t)(assumed_original_size * comp_ratio);
    size_t disk_throughput_100mbps = 20 * 10; // Assuing 20GB/s in 100MB
    double disk_time_sec = (double)assumed_compressed_size / (disk_throughput_100mbps);
    double decomp_time_sec = (method == CompressionMethod::NONE) ? 0.0 : (double)assumed_original_size / (decomp_throughput_100mbps);
    return std::max(disk_time_sec, decomp_time_sec);
}

#if 0
/* moved to common.cu */
static uint roundup512(size_t n) {
    return (n + 512 - 1) & ~(512 - 1);
}
#endif

std::string format_bytes_from_nsectors(size_t nsectors, size_t UNIT) {
    uint64_t val = nsectors * 512 / UNIT;
    uint64_t frac = ((nsectors * 512) % UNIT) / (UNIT/1024);

    std::ostringstream oss;
    oss << val << '.' << std::setw(3) << std::setfill('0') << frac;
    return oss.str();
}

std::string format_bytes(size_t bytes, size_t UNIT) {
    uint64_t val = bytes / UNIT;
    uint64_t frac = (bytes % UNIT) / (UNIT/1024);

    std::ostringstream oss;
    oss << val << '.' << std::setw(3) << std::setfill('0') << frac;
    return oss.str();
}

size_t bytes_from_nsectors(size_t nsectors) {
    return nsectors * 512;
}

void loader_stats_print(size_t page_size, bool compressed, LoaderStats stats)
{
    for (const auto& entry : stats.entries) {
        if (compressed) {
            std::cout << "TABLE(compressed): " << entry.name << std::endl;
        } else {
            std::cout << "TABLE(uncompressed): " << entry.name << std::endl;
        }
        size_t field_total = 0;
        size_t field_total_compressed = 0;
        size_t field_total_meta_offset = 0;
        size_t field_total_meta_sizpag = 0;

        for (const auto& metric : entry.metrics) {
            if (compressed) {
                auto orig_mb = metric.fields_written * page_size / MEBI;
                // auto comp_mb_str = format_bytes_from_nsectors(metric.fields_written_compressed, MEBI);
                auto comp_bytes = bytes_from_nsectors(metric.fields_written_compressed);
                auto comp_meta_offset_mb = metric.fields_written_offset_for_compression * page_size / MEBI;
                auto comp_meta_sizcomp_mb = metric.fields_written_sizcomp_for_compression * page_size / MEBI;

                std::cout << "  Field: " << metric.field_name
                          << ", Codec: " << compression_method_name(metric.compression_method)
                          << ", Fields Orignal: " << orig_mb << " MB"
                          << ", Fields Written Compressed: " << format_bytes(comp_bytes, MEBI) << " MB"
                          << ", Fields Written Meta: Offset " << comp_meta_offset_mb << " MB"
                          << " (" << metric.fields_written_compressed  << " sectors) "
                          << ", CompressedPageSizes " << comp_meta_sizcomp_mb << " MB"
                          // << std::setw(5) << std::setfill(' ')
                          << " ratio(data): " << ((double) comp_bytes / MEBI) / (double)orig_mb
                          << " ratio(data+offset+sizpag): " << ((double) comp_bytes / MEBI + comp_meta_offset_mb + comp_meta_sizcomp_mb) / (double)orig_mb
                          << std::endl;
                field_total_compressed += metric.fields_written_compressed;
                field_total_meta_offset += metric.fields_written_offset_for_compression;
                field_total_meta_sizpag += metric.fields_written_sizcomp_for_compression;
            } else {
                if (metric.fields_written * page_size / MEBI == 0) {
                    std::cout << "  Field: " << metric.field_name
                              << ", Fields Written: " << metric.fields_written * page_size / KIBI << " KB"
                              << std::endl;
                } else {
                    std::cout << "  Field: " << metric.field_name
                              << ", Fields Written: " << metric.fields_written * page_size / MEBI << " MB"
                              << std::endl;
                }
            }
            field_total += metric.fields_written;
        }
        if (compressed) {
            auto orig_mb = field_total * page_size / MEBI;
            // auto comp_mb_str = format_bytes_from_nsectors(field_total_compressed, MEBI);
            auto comp_bytes = bytes_from_nsectors(field_total);
            auto comp_meta_offset_mb = field_total_meta_offset * page_size / MEBI;
            auto comp_meta_sizcomp_mb = field_total_meta_sizpag * page_size / MEBI;

            std::cout << "  Total Fields Orignal: " << orig_mb << " MB"
                      << ", Total Fields Written Compressed: " << (double)comp_bytes / MEBI << " MB"
                      << " (" << field_total_compressed << " sectors) "
                      << " ratio(data): " << ((double) comp_bytes / MEBI) / (double)orig_mb
                      << " ratio(data+offset+sizpag): " << ((double) comp_bytes / MEBI + comp_meta_offset_mb + comp_meta_sizcomp_mb) / (double)orig_mb
                      << std::endl;
        } else {
            size_t field_total_bytes = field_total * page_size;
            if (field_total_bytes / MEBI == 0) {
                std::cout << "  Total Fields Written: " << field_total_bytes / KIBI << " KB" << std::endl;
            } else {
                std::cout << "  Total Fields Written: " << field_total_bytes / MEBI << " MB" << std::endl;
            }
        }
    }
}

int loader_validate_field_size(uint64_t page_size, size_t field_size)
{
    if (field_size == 0)
    {
        std::cerr << "[FATAL] Unexpected: field size is 0" << std::endl;
        return -1;
    }
    if (page_size % field_size)
    {
        std::cerr << "[FATAL] Unexpected: page size is not a multiple of field size" << std::endl;
        return -1;
    }
    return 0;
}

int count_regex_matched_files(const std::string& directory, const std::string& pattern) {
    int i = 0;
    std::regex regex_pattern(pattern);

    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                const std::string filename = entry.path().filename().string();
                if (std::regex_match(filename, regex_pattern)) {
#ifdef DEBUG_PRINT
                    std::cout << "Matched file: " << entry.path().string() << std::endl;
#endif
                    i++;
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    } catch (const std::regex_error& e) {
        std::cerr << "Regex error: " << e.what() << std::endl;
    }

    return i;
}

int count_input_files(const std::string& directory, const std::string &prefix) {
#ifdef DEBUG_PRINT
    std::cout << "Regex rule: " << "customer\\.tbl\\**" << std::endl;
#endif
    std::stringstream ss;
    ss << prefix;
    ss << ".tbl.?[0-9]*";
    auto regex_pattern = ss.str();

#ifdef DEBUG_PRINT
    std::cout << "Regex rule: " << regex_pattern << std::endl;
#endif
    //return count_regex_matched_files(directory, "customer.tbl.?[0-9]*");
    return count_regex_matched_files(directory, regex_pattern);
}

static uint64_t binUnpack(uint* out, uint* block_offsets, uint num_entries, uint *decoded_values) {
    constexpr bool enable_debug_print = false;
    // struct binpack_hdr* out_hdr = reinterpret_cast<struct binpack_hdr*>(out);
    if constexpr (enable_debug_print) {
        std::cout << "=== Start binUnpack32 ===" << std::endl;
    }
    uint offset = 0;

    uint block_count = 4;
    uint block_size = 128;
    uint miniblock_count = 4;
    uint total_count = num_entries;
    ulong first_val;
 
    /* Using ulong header for keeping simplicity */
    // out[0] = block_size;
    // out[1] = miniblock_count;
    // out[2] = total_count;
    // out[3] = first_val;
#if 0
    // Playground for reading the header
    block_size = out[0];
    miniblock_count = out[1];
    total_count = out[2];
    first_val = out[3];
 
    // ulong out = out_hdr->body;

    std::cout << "block_size " << block_size << " miniblock_count " << miniblock_count
      << " total_count " << total_count << " first_val " << first_val << std::endl;
 
    offset = 4 + 1;
    ulong bitwidths = out[offset];
    printf("bitwidths: %lx(%u)\n", bitwidths, offset);
    uint bitwidth1 = bitwidths & 255;
    uint bitwidth2 = (bitwidths >> 8) & 255;
    uint bitwidth3 = (bitwidths >> 16) & 255;
    uint bitwidth4 = (bitwidths >> 24) & 255;
    printf("\tbitwidths: %x %x %x %x\n", bitwidth1, bitwidth2, bitwidth3, bitwidth4);

    offset++;
    ulong packed = out[offset];
    ulong val1 = out[offset];
    printf("%lu %lu %lu %lu\n",
      packed & bitwidth1,
      (packed >> bitwidth1) & bitwidth2,
      (packed >> (bitwidth1 + bitwidth2)) & bitwidth3,
      (packed >> (bitwidth1 + bitwidth2 + bitwidth3)) & bitwidth4);
#else
    uint miniblock_size = uint(block_size / miniblock_count);

    uint nblock = num_entries % block_size == 0 ?
      (num_entries / block_size) : ((num_entries / block_size) + 1);

    if constexpr (enable_debug_print) {
        printf("nblock=%u ", nblock);
        for (uint k = 0; k < nblock; k++) {
          printf("offsets[%u]=%u ", k, block_offsets[k]);
        }
        printf("\n");
    }

    block_size = out[0];
    miniblock_count = out[1];
    total_count = out[2];
    // Start to prase data block.
    first_val = out[3];

    if constexpr (enable_debug_print) {
        std::cout << "=== Header Info ===" << std::endl;
        std::cout << "block_size " << block_size << " miniblock_count " << miniblock_count
          << " total_count " << total_count << " first_val " << first_val << std::endl;
    }
  
    offset = 4;
    uint min_val = out[offset];
    offset = 5;
    uint bitwidths = out[offset];
    if constexpr (enable_debug_print) {
        printf("bitwidths: %x(%u)\n", bitwidths, offset);
    }
    uint bitwidth1 = bitwidths & 255;
    uint bitwidth2 = (bitwidths >> 8) & 255;
    uint bitwidth3 = (bitwidths >> 16) & 255;
    uint bitwidth4 = (bitwidths >> 24) & 255;
    if constexpr (enable_debug_print) {
        printf("\tbitwidths: %x %x %x %x (%u)\n", bitwidth1, bitwidth2, bitwidth3, bitwidth4, bitwidth1);
    }

    offset++;
    uint bitwidth;
    // uint8_t* data_baseaddr = reinterpret_cast<uint8_t*>(&out[offset]);
    uint* data_baseaddr = reinterpret_cast<uint*>(&out[offset]);
    // uint packed;
    // uint k;

    // const uint MASK = (1UL << bitwidth) - 1; // 0x7F

    uint current_word = 0;    // 現在処理中の32bit word のインデックス
    uint bits_available = 32; // 現在のwordで残っているビット数
    // uint shift_current = 0; // 現在のwordで残っているビット数
  
    uint mask;
    uint current_value = data_baseaddr[current_word];
    uint unpacked;
  
  
    if constexpr (enable_debug_print) {
        printf("block_count %d\n", block_count);
        printf("miniblock_count %d\n", miniblock_count);
    }
  
    uint cnt = 0;
    uint64_t checksum = 0;
    //for (uint i = 0; i < block_count; i++) {
    for (uint i = 0; i < nblock; i++) {
      offset = block_offsets[i];
      min_val = out[offset];
      offset++;
      bitwidths = out[offset];
      offset++;
      data_baseaddr = reinterpret_cast<uint*>(&out[offset]);
      #if 0
      if (i == 1) {
        printf("offset=%u min_val: %d, bitwidths: %x\n", offset, min_val, bitwidths);
        exit(1);
      }
      #endif
      if constexpr (enable_debug_print) {
        printf("\toffset=%u min_val: %d, bitwidths: %x\n", offset, min_val, bitwidths);
      }
  
      // reset context
      current_word = 0;
      bits_available = 32;
      current_value = data_baseaddr[current_word];

      for (uint j = 0; j < miniblock_count; j++) {
        bitwidth = bitwidths & 255;
        mask = (1U << bitwidth) - 1;
        for (int k=0; k<miniblock_size; k++) {
          // printf("current_pos: %u current_value: %lx\n", current_word, current_value);
          if constexpr (enable_debug_print) {
            printf("\tcurrent_value: %u (%x)\n", current_value, current_value);
          }
          if (bits_available < bitwidth) {
            current_word++;
            uint next_value = data_baseaddr[current_word];
            assert(current_word > 0);
            if constexpr (enable_debug_print) {
              printf("\tcurrent_word: %u -> %u, bits_available=%u next_value=%u (%x)\n", current_word - 1, current_word, bits_available, next_value, next_value);
            }
  
            if (bits_available > 0) {
              if constexpr (enable_debug_print) {
                printf("\t\tcurrent_value: %x\n", current_value & ((1U << bits_available) - 1));
              }
              unpacked = ((current_value & ((1U << bits_available) - 1)));
              if constexpr (enable_debug_print) {
                printf("\t\tunpacking: %x %x\n", unpacked, next_value);
              }
              unpacked += ((next_value & ((1U << (bitwidth - bits_available)) - 1)) << bits_available);
              // printf("unpacked2: %x\n", unpacked);
              if constexpr (enable_debug_print) {
                printf("\t\tunpacking: %x %u\n", unpacked, bits_available);
              }
              // current_value = next_value >> (bitwidth - bits_available);
              // bits_available = 64 - (bitwidth - bits_available);
              // current_value = next_value >> bits_available;
              current_value = next_value >> (bitwidth - bits_available);
            } else {
              unpacked = next_value & mask;
              current_value = next_value >> bitwidth;
            }
  
            // current_value = next_value;
            bits_available = 32 - (bitwidth - bits_available);
            decoded_values[cnt++] = unpacked + min_val;
            checksum += (unpacked + min_val);
            if constexpr (enable_debug_print) {
              printf("\tunpacked3: %u (%u + %u (%x)) %u %u\n", unpacked + min_val, min_val, unpacked, unpacked, bits_available, bitwidth);
            }
            // printf("\t%u", unpacked);
          } else {
            unpacked = current_value & mask;
            decoded_values[cnt++] = unpacked + min_val;
            checksum += (unpacked + min_val);
            if constexpr (enable_debug_print) {
              printf("\tunpacked4: %u (unpacked:%u min_val:%u) (%u %u)\n", unpacked + min_val, unpacked, min_val, bits_available, bitwidth);
            }
            current_value >>= bitwidth;
            bits_available -= bitwidth;
            // printf("(%lx %u) ", current_value, bits_available);
          }
        }
        // exit(1);
        bitwidths >>= 8;
        current_word++;
        bits_available = 32;
        current_value = data_baseaddr[current_word];
 
        if constexpr (enable_debug_print) {
          printf("\nNOTE: finished processing miniblock %d pos(%u->%u)\n"
            "\tnew current value %x\n",
            j, current_word - 1, current_word, current_value);
          }
      }
 
      // The first 2 longs are header value, so plus + 2 is required
      if constexpr (enable_debug_print) {
        printf("\noffset=%d, block_offsets[%d]=%d\n",block_offsets[i]+current_word+2, i+1, block_offsets[i+1]);
      }
      // for (int j=block_offsets[i]+current_word; j<block_offsets[i+1]; j++) {
      //   printf("\nout[%d] %lx\n", j, out[j]);
      // }
      // exit(1);
    }
    return checksum;
#endif
}

#if 0
static void binUnpackPrint(uint* out, uint* block_offsets, uint num_entries) {
    size_t cnt = 0;

    uint offset = 0;

    uint block_count = 4;
    uint block_size = 128;
    uint miniblock_count = 4;
    uint total_count = num_entries;
    ulong first_val;

    /* Using ulong header for keeping simplicity */
    // out[0] = block_size;
    // out[1] = miniblock_count;
    // out[2] = total_count;
    // out[3] = first_val;
    uint miniblock_size = uint(block_size / miniblock_count);

    uint nblock = num_entries % block_size == 0 ?
      (num_entries / block_size) : ((num_entries / block_size) + 1);

    block_size = out[0];
    miniblock_count = out[1];
    total_count = out[2];
    // Start to prase data block.
    first_val = out[3];

    offset = 4;
    uint min_val = out[offset];
    offset = 5;
    uint bitwidths = out[offset];
    uint bitwidth1 = bitwidths & 255;
    uint bitwidth2 = (bitwidths >> 8) & 255;
    uint bitwidth3 = (bitwidths >> 16) & 255;
    uint bitwidth4 = (bitwidths >> 24) & 255;

    offset++;
    uint bitwidth;
    // uint8_t* data_baseaddr = reinterpret_cast<uint8_t*>(&out[offset]);
    uint* data_baseaddr = reinterpret_cast<uint*>(&out[offset]);
    uint packed;
    uint k;

    // const uint MASK = (1UL << bitwidth) - 1; // 0x7F

    uint current_word = 0;    // 現在処理中の32bit word のインデックス
    uint bits_available = 32; // 現在のwordで残っているビット数
    // uint shift_current = 0; // 現在のwordで残っているビット数

    uint mask;
    uint current_value = data_baseaddr[current_word];
    uint unpacked;

    // printf("block_count %d\n", block_count);
    // printf("miniblock_count %d\n", miniblock_count);

    //for (uint i = 0; i < block_count; i++) {
    for (uint i = 0; i < nblock; i++) {
      offset = block_offsets[i];
      min_val = out[offset];
      offset++;
      bitwidths = out[offset];
      offset++;
      data_baseaddr = reinterpret_cast<uint*>(&out[offset]);

      // reset context
      current_word = 0;
      bits_available = 32;
      current_value = data_baseaddr[current_word];

      for (uint j = 0; j < miniblock_count; j++) {
        bitwidth = bitwidths & 255;
        mask = (1U << bitwidth) - 1;
        for (int k=0; k<miniblock_size; k++) {
          if (bits_available < bitwidth) {
            current_word++;
            uint next_value = data_baseaddr[current_word];
            assert(current_word > 0);

            if (bits_available > 0) {
              unpacked = ((current_value & ((1U << bits_available) - 1)));
              unpacked += ((next_value & ((1U << (bitwidth - bits_available)) - 1)) << bits_available);
              current_value = next_value >> (bitwidth - bits_available);
            } else {
              unpacked = next_value & mask;
              current_value = next_value >> bitwidth;
            }

            // current_value = next_value;
            bits_available = 32 - (bitwidth - bits_available);
            // printf("\tunpacked3: %u (%u + %u (%x)) %u %u\n", unpacked + min_val, min_val, unpacked, unpacked, bits_available, bitwidth);
            printf("%lu\t%u\n", cnt, unpacked + min_val);
            cnt++;
          } else {
            unpacked = current_value & mask;
            // printf("\tunpacked4: %u (unpacked:%u min_val:%u) (%u %u)\n", unpacked + min_val, unpacked, min_val, bits_available, bitwidth);
            printf("%lu\t%u\n", cnt, unpacked + min_val);
            cnt++;
            current_value >>= bitwidth;
            bits_available -= bitwidth;
          }
        }
        // exit(1);
        bitwidths >>= 8;
        current_word++;
        bits_available = 32;
        current_value = data_baseaddr[current_word];
      }
    }
}
#endif

uint64_t binUnpack64(ulong* out, uint* block_offsets, uint num_entries, uint64_t* unpacked_output) {
  // struct binpack_hdr* out_hdr = reinterpret_cast<struct binpack_hdr*>(out);
  constexpr bool enable_debug_print = false;
  if constexpr (enable_debug_print) {
    std::cout << "=== Start to binUnpack64 ===" << std::endl;
  }
  uint offset = 0;

  uint block_count = 4;
  uint block_size = 128;
  uint miniblock_count = 4;
  uint total_count = num_entries;
  ulong first_val;

  /* Using ulong header for keeping simplicity */
  // out[0] = block_size;
  // out[1] = miniblock_count;
  // out[2] = total_count;
  // out[3] = first_val;
#if 0
  // Playground for reading the header
  block_size = out[0];
  miniblock_count = out[1];
  total_count = out[2];
  first_val = out[3];

  // ulong out = out_hdr->body;
  cout << "block_size " << block_size << " miniblock_count " << miniblock_count
    << " total_count " << total_count << " first_val " << first_val << endl;
  
  offset = 4 + 1;
  ulong bitwidths = out[offset];
  printf("bitwidths: %lx(%u)\n", bitwidths, offset);
  uint bitwidth1 = bitwidths & 255;
  uint bitwidth2 = (bitwidths >> 8) & 255;
  uint bitwidth3 = (bitwidths >> 16) & 255;
  uint bitwidth4 = (bitwidths >> 24) & 255;
  printf("\tbitwidths: %x %x %x %x\n", bitwidth1, bitwidth2, bitwidth3, bitwidth4);

  offset++;
  ulong packed = out[offset];
  ulong val1 = out[offset];
  printf("%lu %lu %lu %lu\n",
    packed & bitwidth1,
    (packed >> bitwidth1) & bitwidth2,
    (packed >> (bitwidth1 + bitwidth2)) & bitwidth3,
    (packed >> (bitwidth1 + bitwidth2 + bitwidth3)) & bitwidth4);
#else
  uint miniblock_size = uint(block_size / miniblock_count);

  uint nblock = num_entries % block_size == 0 ?
    (num_entries / block_size) : ((num_entries / block_size) + 1);

  if constexpr (enable_debug_print) {
    for (uint k = 0; k < nblock; k++) {
      printf("offsets[%u]=%u ", k, block_offsets[k]);
    }
    printf("\n");
  }

  block_size = out[0];
  miniblock_count = out[1];
  total_count = out[2];
  // Start to prase data block.
  first_val = out[3];

  // ulong out = out_hdr->body;
  if constexpr (enable_debug_print) {
    std::cout << "block_size " << block_size << " miniblock_count " << miniblock_count
      << " total_count " << total_count << " first_val " << first_val << std::endl;
  }

  offset = 4;
  ulong min_val = out[offset];
  offset = 5;
  ulong bitwidths = out[offset];
  if constexpr (enable_debug_print) {
    printf("bitwidths: %lx(%u)\n", bitwidths, offset);
  }
  ulong bitwidth1 = bitwidths & 255;
  ulong bitwidth2 = (bitwidths >> 8) & 255;
  ulong bitwidth3 = (bitwidths >> 16) & 255;
  ulong bitwidth4 = (bitwidths >> 24) & 255;
  if constexpr (enable_debug_print) {
    printf("\tbitwidths: %lx %lx %lx %lx (%lu)\n", bitwidth1, bitwidth2, bitwidth3, bitwidth4, bitwidth1);
  }

  offset++;
  ulong bitwidth;
  // uint8_t* data_baseaddr = reinterpret_cast<uint8_t*>(&out[offset]);
  ulong* data_baseaddr = reinterpret_cast<ulong*>(&out[offset]);
  // ulong packed;
  // ulong k;

  // const ulong MASK = (1UL << bitwidth) - 1; // 0x7F

  uint current_word = 0;    // 現在処理中の64bit word のインデックス
  uint bits_available = 64; // 現在のwordで残っているビット数
  // uint shift_current = 0; // 現在のwordで残っているビット数

  ulong mask;
  ulong current_value = data_baseaddr[current_word];
  ulong unpacked;

  if constexpr (enable_debug_print) {
    printf("block_count %d\n", block_count);
    printf("miniblock_count %d\n", miniblock_count);
  }

  ulong cnt = 0;
  uint64_t checksum = 0;
  // for (uint i = 0; i < block_count; i++) {
  for (uint i = 0; i < nblock; i++) {
    offset = block_offsets[i];
    min_val = out[offset];
    offset++;
    bitwidths = out[offset];
    offset++;
    data_baseaddr = reinterpret_cast<ulong*>(&out[offset]);

    if constexpr (enable_debug_print) {
      if (i == 1) {
        printf("offset=%u min_val: %ld, bitwidths: %lx\n", offset, min_val, bitwidths);
        //exit(1);
      }
    }

    // reset context
    current_word = 0;
    bits_available = 64;
    current_value = data_baseaddr[current_word];

    for (uint j = 0; j < miniblock_count; j++) {
      bitwidth = bitwidths & 255;
      mask = (1UL << bitwidth) - 1;
      for (int k=0; k<miniblock_size; k++) {
        // printf("current_pos: %u current_value: %lx\n", current_word, current_value);
        if (bits_available < bitwidth) {
          if constexpr (enable_debug_print) {
            printf("current_word: %u -> %u, bits_available=%u\n", current_word, current_word + 1, bits_available);
          }
          current_word++;
          ulong next_value = data_baseaddr[current_word];

          if (bits_available > 0) {
            if constexpr (enable_debug_print) {
              printf("current_value: %lx\n", current_value & ((1UL << bits_available) - 1));
            }
            unpacked = ((current_value & ((1UL << bits_available) - 1)));
            if constexpr (enable_debug_print) {
              printf("unpacked1: %lx %lx\n", unpacked, next_value);
            }
            unpacked += ((next_value & ((1UL << (bitwidth - bits_available)) - 1)) << bits_available);
            // printf("unpacked2: %lx\n", unpacked);
            if constexpr (enable_debug_print) {
              printf("unpacked2: %lx %u\n", unpacked, bits_available);
            }
            // current_value = next_value >> (bitwidth - bits_available);
            // bits_available = 64 - (bitwidth - bits_available);
            // current_value = next_value >> bits_available;
            current_value = next_value >> (bitwidth - bits_available);
          } else {
            unpacked = next_value & mask;
            current_value = next_value >> bitwidth;
          }

          // current_value = next_value;
          bits_available = 64 - (bitwidth - bits_available);
          // printf("unpacked3: %lx %u %lu\n", unpacked + min_val, bits_available, bitwidth);
          // printf("unpacked3: %lu (%lu + %lu (%lx)) %u %lu\n", unpacked + min_val, min_val, unpacked, unpacked, bits_available, bitwidth);
          // printf("%lu ", unpacked);
          checksum += (unpacked + min_val);
          unpacked_output[cnt] = unpacked + min_val;
          if constexpr (enable_debug_print) {
            printf("%lu\t%lu\n", cnt, unpacked + min_val);
          }
          cnt++;
        } else {
          unpacked = current_value & mask;
          // printf("unpacked4: %lu\n", unpacked + min_val);
          checksum += (unpacked + min_val);
          unpacked_output[cnt] = unpacked + min_val;
          if constexpr (enable_debug_print) {
            printf("%lu\t%lu\n", cnt, unpacked + min_val);
          }
          cnt++;
          current_value >>= bitwidth;
          bits_available -= bitwidth;
          // printf("(%lx %u) ", current_value, bits_available);
        }
      }
      // exit(1);
      bitwidths >>= 8;
      current_word++;
      bits_available = 64;
      current_value = data_baseaddr[current_word];

      if constexpr (enable_debug_print) {
        printf("\nNOTE: finished processing miniblock %d pos(%u->%u)\n"
          "\tnew current value %lx\n",
          j, current_word - 1, current_word, current_value);
      }
    }

    // The first 2 longs are header value, so plus + 2 is required
    if constexpr (enable_debug_print) {
      printf("\noffset=%d, block_offsets[%d]=%d\n",block_offsets[i]+current_word+2, i+1, block_offsets[i+1]);
    }
    // for (int j=block_offsets[i]+current_word; j<block_offsets[i+1]; j++) {
    //   printf("\nout[%d] %lx\n", j, out[j]);
    // }
    // exit(1);

  }
  if constexpr (enable_debug_print) {
    std::cout << "Checksum: " << checksum << std::endl;
  }
  return checksum;
#endif
}

#if 0
static void binUnpack64Print(ulong* out, uint* block_offsets, uint num_entries) {
    size_t cnt = 0;

    uint offset = 0;

    uint block_count = 4;
    ulong block_size = 128;
    ulong miniblock_count = 4;
    uint total_count = num_entries;
    ulong first_val;

    /* Using ulong header for keeping simplicity */
    // out[0] = block_size;
    // out[1] = miniblock_count;
    // out[2] = total_count;
    // out[3] = first_val;
    uint miniblock_size = uint(block_size / miniblock_count);

    uint nblock = num_entries % block_size == 0 ?
      (num_entries / block_size) : ((num_entries / block_size) + 1);
 
    block_size = out[0];
    miniblock_count = out[1];
    total_count = out[2];
    // Start to prase data block.
    first_val = out[3];

    offset = 4;
    ulong min_val = out[offset];
    offset = 5;
    ulong bitwidths = out[offset];
    ulong bitwidth1 = bitwidths & 255;
    ulong bitwidth2 = (bitwidths >> 8) & 255;
    ulong bitwidth3 = (bitwidths >> 16) & 255;
    ulong bitwidth4 = (bitwidths >> 24) & 255;
 
    std::cout << "bitwidths: " << bitwidth1 << ", " << bitwidth2 << ", "
      << bitwidth3 << ", " << bitwidth4 << " (" << offset << ")\n";
    // exit(1);

    offset++;
    uint bitwidth;
    // uint8_t* data_baseaddr = reinterpret_cast<uint8_t*>(&out[offset]);
    ulong* data_baseaddr = reinterpret_cast<ulong*>(&out[offset]);
    uint packed;
    uint k;

    // const uint MASK = (1UL << bitwidth) - 1; // 0x7F

    uint current_word = 0;    // 現在処理中の32bit word のインデックス
    uint bits_available = 64; // 現在のwordで残っているビット数
    // uint shift_current = 0; // 現在のwordで残っているビット数

    ulong mask;
    ulong current_value = data_baseaddr[current_word];
    ulong unpacked;

    // printf("block_count %d\n", block_count);
    // printf("miniblock_count %d\n", miniblock_count);

    for (uint i = 0; i < block_count; i++) {
    //for (uint i = 0; i < nblock; i++) {
      offset = block_offsets[i];
      min_val = out[offset];
      offset++;
      bitwidths = out[offset];
      offset++;
      data_baseaddr = reinterpret_cast<ulong*>(&out[offset]);

      // reset context
      current_word = 0;
      bits_available = 64;
      for (uint j = 0; j < miniblock_count; j++) {
        bitwidth = bitwidths & 255;
        mask = (1U << bitwidth) - 1;
        for (int k=0; k<miniblock_size; k++) {
          std::cout << "k=" << k << " current_value: " << current_value
            << " (" << std::hex << current_value << std::dec << ")"
            << " bits_available: " << bits_available << " bitwidth: " << bitwidth << "\n";
          
          if (bits_available < bitwidth) {
            current_word++;
            ulong next_value = data_baseaddr[current_word];
            assert(current_word > 0);

            if (bits_available > 0) {
              unpacked = ((current_value & ((1U << bits_available) - 1)));
              std::cout << "\t\tcurrent_value: " << std::hex << (current_value & ((1U << bits_available) - 1))
                << " next_value: " << std::hex << next_value << std::dec << "\n";
              unpacked += ((next_value & ((1U << (bitwidth - bits_available)) - 1)) << bits_available);
              std::cout << "\t\tunpacking: " << std::hex << unpacked << std::dec << " " 
                << (next_value & ((1U << (bitwidth - bits_available)) - 1)) << " "
                << bits_available << "\n";
              current_value = next_value >> (bitwidth - bits_available);
              std::cout << "\t\tunpacking: " << std::hex << unpacked << std::dec << " "
                << (next_value & ((1U << (bitwidth - bits_available)) - 1)) << " "
                << bits_available << "\n";
            } else {
              unpacked = next_value & mask;
              current_value = next_value;
            }

            // current_value = next_value;
            bits_available = 64 - (bitwidth - bits_available);
            printf("\tunpacked3: %lu (%lu + %lu (%lx)) %u %u\n", unpacked + min_val, min_val, unpacked, unpacked, bits_available, bitwidth);
            printf("%lu\t%lu\n", cnt, unpacked + min_val);
            cnt++;
          } else {
            unpacked = current_value & mask;
            printf("\tunpacked4: %lu (unpacked:%lu min_val:%lu) (%u %u)\n", unpacked + min_val, unpacked, min_val, bits_available, bitwidth);
            printf("%lu\t%lu\n", cnt, unpacked + min_val);
            cnt++;
            current_value >>= bitwidth;
            bits_available -= bitwidth;
          }
        }
        //exit(1);
        bitwidth >>= 8;
        current_word++;
        bits_available = 64;
        current_value = data_baseaddr[current_word];
      }
    }
}
#endif

static void compress_page_golap(CompressionContext &ctx, char *src, size_t page_size,
    char *dst, const CompressionMethod compression, uint32_t& compressed_size)
{
    compressed_size = 0;
    switch (compression) {
        case CompressionMethod::SNAPPY:
        {
            int ret = compress_page_with_snappy(
                src, page_size, dst, ctx.max_compress_dst_buffer_size, compressed_size);
            if (ret < 0) {
                std::cerr << "Snappy compression failed\n";
                exit(EXIT_FAILURE);
                return;
            }
            break;
        }
        case CompressionMethod::DEFLATE:
        {
            int ret = compress_page_with_deflate(
                ctx.zlib_ctx, src, page_size, dst, ctx.max_compress_dst_buffer_size, compressed_size);
            if (ret < 0) {
                std::cerr << "Deflate compression failed\n";
                exit(EXIT_FAILURE);
                return;
            }
            break;
        }
        case CompressionMethod::LZ4:
        {
            int ret = compress_page_with_lz4(
                src, page_size, dst, ctx.max_compress_dst_buffer_size, compressed_size);
            if (ret < 0) {
                std::cerr << "LZ4 compression failed\n";
                exit(EXIT_FAILURE);
                return;
            }
            break;
        }
        default:
        {
            std::cerr << "Unsupported compression method\n";
            free(dst);
            exit(EXIT_FAILURE);
            return;
        }
    }

    return;
}

#if 0
static void decompress_page(char *src, uint32_t compressed_size,
    char *dst, size_t page_size, const CompressionMethod compression)
{
    compressed_size = 0;
    switch (compression) {
        case CompressionMethod::SNAPPY:
        {
            int ret = compress_page_with_snappy(
                src, page_size, dst, ctx.max_compress_dst_buffer_size, compressed_size);
            if (ret < 0) {
                std::cerr << "Snappy compression failed\n";
                free(dst);
                exit(EXIT_FAILURE);
                return;
            }
            break;
        }
        case CompressionMethod::DEFLATE:
        {
            int ret = compress_page_with_deflate(
                ctx.zlib_ctx, src, page_size, dst, ctx.max_compress_dst_buffer_size, compressed_size);
            if (ret < 0) {
                std::cerr << "Deflate compression failed\n";
                free(dst);
                exit(EXIT_FAILURE);
                return;
            }
            break;
        }
        case CompressionMethod::LZ4:
        {
            int ret = compress_page_with_lz4(
                src, page_size, dst, ctx.max_compress_dst_buffer_size, compressed_size);
            if (ret < 0) {
                std::cerr << "LZ4 compression failed\n";
                free(dst);
                exit(EXIT_FAILURE);
                return;
            }
            break;
        }
        default:
        {
            std::cerr << "Unsupported compression method\n";
            free(dst);
            exit(EXIT_FAILURE);
            return;
        }
    }

    return;
}
#endif

void flush_page_to_storage(std::vector<int>& output_fds,
    char* page_buffer, size_t page_id, size_t page_size,
    const uint32_t& nrecs_in_page_count, std::vector<uint64_t> &nrecs_in_page,
    uint64_t& nios_data,
    uint64_t& nwritten_sectors,
    const CompressionMethod compression,
    std::vector<uint32_t> &compressed_sizes,
    CompressionContext &ctx,
    const uint64_t compressed_write_base_page_id,
    std::vector<uint64_t> &compressed_page_write_offsets,
    const size_t field_id, const int line_from)
{
    constexpr bool debug_flush = false;
    if constexpr (debug_flush) {
        std::cout << "    [DiskIO] Flushing Page Buffer to Disk (PageID: " << page_id << ")\n";
    }

    char *target_buffer = page_buffer;
    size_t size_write = page_size;
    if (compression == CompressionMethod::NONE)  {
        page_pwrite_host(output_fds, target_buffer, page_id, size_write);
    } else {
        if (!ctx.inited) {
            std::cerr << "[ERROR](" << line_from << ") Compression context is not initialized\n";
            exit(EXIT_FAILURE);
            return;
        }
        uint32_t compressed_size = 0;
        if (ctx.enable_golap_compression_mode) {
            memset(ctx.dst_buffer, 0, ctx.max_compress_dst_buffer_size);
            compress_page_golap(ctx, page_buffer, page_size, ctx.dst_buffer, compression, compressed_size);
            compressed_sizes.push_back(compressed_size);
            if constexpr (debug_flush) {
                std::cout << "    [Compression] Compressed Page Size: " << compressed_size << " bytes\n";
            }
        } else {
            switch (compression) {
                /* PFOR/PFOR64/FSST/FSST_ROWID compression is done in flush_staging_buffer */
                case CompressionMethod::PFOR:
                case CompressionMethod::PFOR64:
                case CompressionMethod::FSST:
                case CompressionMethod::FSST_ROWID:
                {
                    // ctx.dst_buffer should be used
                    compressed_size = ctx.compressed_size;
                    break;
                }
                case CompressionMethod::LZ4:
                {
                    memset(ctx.dst_buffer, 0, ctx.max_compress_dst_buffer_size);
                    int ret = compress_page_with_lz4(
                        page_buffer, page_size, ctx.dst_buffer, ctx.max_compress_dst_buffer_size, compressed_size);
                    if (ret < 0) {
                        std::cerr << "[ERROR](" << line_from << ") LZ4 compression failed\n";
                        exit(EXIT_FAILURE);
                        return;
                    }

                    /* DEBUG */
#if 0
                    char *dst_original = new char[page_size];
                    if (page_id == 9925) {
                        decompress_page_with_lz4(ctx.dst_buffer, compressed_size, dst_original, page_size);
                        std::cout << "[DEBUG](" << line_from << ") LZ4 Decompression Check for page_id == 9925\n";
                        std::cout << "\tOriginal Page Data: "
                            << std::hex
                            << reinterpret_cast<uint64_t*>(page_buffer)[0] << " "
                            << reinterpret_cast<uint64_t*>(page_buffer)[1] << " "
                            << reinterpret_cast<uint64_t*>(page_buffer)[2] << " "
                            << reinterpret_cast<uint64_t*>(page_buffer)[3] << " "
                            << std::dec << "\n";
                        std::cout << "\tCompressed Size: " << compressed_size << "\n";
                        exit(0);
                    }
#endif
                    break;
                }
                case CompressionMethod::LZ4PAR:
                {
                    memset(ctx.dst_buffer, 0, ctx.max_compress_dst_buffer_size);
                    int ret = compress_page_with_lz4par(
                        page_buffer, page_size, ctx.dst_buffer, ctx.max_compress_dst_buffer_size, compressed_size);
                    if (ret < 0) {
                        std::cerr << "[ERROR](" << line_from << ") LZ4PAR compression failed\n";
                        exit(EXIT_FAILURE);
                        return;
                    }
                    if (compressed_size > page_size) {
                        std::cerr << "[ERROR](" << line_from << ") LZ4PAR compressed size is larger than page size\n";
                        exit(EXIT_FAILURE);
                        return;
                    }

                    /* DEBUG */
#if 0
                    char *dst_original = new char[page_size];
                    if (page_id == 9925) {
                        int ret = decompress_page_with_lz4par(ctx.dst_buffer, compressed_size, dst_original, page_size);
                        if (ret != 0) {
                            std::cerr << "[ERROR](" << line_from << ") LZ4PAR decompression failed\n";
                            exit(EXIT_FAILURE);
                        }
                        std::cout << "[DEBUG](" << line_from << ") LZ4PAR Decompression Check for page_id == 9925\n";
                        std::cout << "\tOriginal Page Data: "
                            << std::hex
                            << reinterpret_cast<uint64_t*>(page_buffer)[0] << " "
                            << reinterpret_cast<uint64_t*>(page_buffer)[1] << " "
                            << reinterpret_cast<uint64_t*>(page_buffer)[2] << " "
                            << reinterpret_cast<uint64_t*>(page_buffer)[3] << " "
                            << std::dec << "\n";
                        std::cout << "\tCompressed Size: " << compressed_size << "\n";
                        exit(0);
                    }
#endif
                    break;
                }
                case CompressionMethod::SNAPPY:
                case CompressionMethod::DEFLATE:
                default:
                {
                    std::cerr << "[ERROR](" << line_from << ") Unsupported compression method\n";
                    std::cerr << "\tCompressionMethod: " << compression_method_name(compression) << "\n";
                    exit(EXIT_FAILURE);
                    return;
                }
            }
            compressed_sizes.push_back(compressed_size);
        }
        if (compressed_size == 0 || compressed_size > page_size) {
            std::cerr << "[ERROR](" << line_from << ") Invalid compressed size: " << compressed_size << "\n";
            exit(1);
        }
        size_write = roundup4096(compressed_size);
        target_buffer = ctx.dst_buffer;
        // V2
        write_compressed_page_host(output_fds, target_buffer, page_id, size_write, page_size,
                compressed_page_write_offsets,  compressed_write_base_page_id
            );
        // V1
        // page_pwrite_comp_host(output_fds, target_buffer, page_id, size_write, page_size);

    }

    const size_t nsectors_per_page = size_write / 512;

    // Reset page buffer mock logic here if needed
    nios_data++;
    nwritten_sectors+=nsectors_per_page;
    // std::cout << "[DEBUG][CHECK] xtn_id:"
    //     << dbg_xtn_id  << ", "
    //     << nrecs_in_page_count[i] << " records are written to dictionary page: " << pagids[i] << "\n";
    if (compression == CompressionMethod::PFOR ||
        compression == CompressionMethod::PFOR64 ||
        compression == CompressionMethod::FSST ||
        compression == CompressionMethod::FSST_ROWID) {
        // For PFOR/PFOR64/FSST/FSST_ROWID, nalloc is set during compression
        if (pag_get_nalloc(target_buffer) != nrecs_in_page_count) {
            std::cerr << "[ERROR](" << line_from << ") pag_get_nalloc(pagbuf) != nrecs_in_page_count[i] "
                << pag_get_nalloc(target_buffer) << " != " << nrecs_in_page_count << std::endl;
            exit(1);
        }
 
    } else {
        // For other compression methods, uncompressed page_buffer contains readable nalloc
        if (pag_get_nalloc(page_buffer) != nrecs_in_page_count) {
            std::cerr << "[ERROR](" << line_from << ") pag_get_nalloc(pagbuf) != nrecs_in_page_count[i] "
                << pag_get_nalloc(page_buffer) << " != " << nrecs_in_page_count << std::endl;

            exit(1);
        }
    }
    // std::cout << "[DEBUG][CHECK] FieldID:"
    //     << field_id  << ", "
    //     << nrecs_in_page_count << " records are written to page: " << page_id << "\n";
    nrecs_in_page.push_back(nrecs_in_page_count);
}

// Generic Flush Helper
template <typename T, typename S, size_t NumSidewaysFields>
void flush_staging_buffer(
    std::vector<SortPair<T>>& buffer, 
    char* page_buffer,
    size_t& page_id,
    const size_t page_size,
    size_t& npages_used,
    const size_t npages_limit,
    const size_t field_size,
    CompressionMethod compression,
    CompressionContext& comp_ctx,
    const size_t compression_base_page_id,
    std::vector<uint64_t>& compressed_write_offsets,
    std::vector<uint32_t>& compressed_sizes,
    bool enable_stats,
    ColumnStats<T>& current_stats,
    std::vector<ColumnStats<T>>& stats_history,
    bool enable_sideways_stats,
    std::vector<ColumnStats<S>>& sideways_stats,
    std::array<std::vector<ColumnStats<S>>, NumSidewaysFields>& sideways_stats_history,
    // New Args
    std::vector<int>& output_fds,
    const size_t nrecs_capacity_per_page,
    uint32_t& nrecs_in_page_count,
    uint32_t& max_nrecs_in_page_count,
    std::vector<uint64_t> &nrecs_in_page,
    uint64_t& nios,
    uint64_t& nios_data,
    size_t field_id,
    const bool is_final_flush = false
) {
    if (buffer.empty()) return;

    constexpr bool debug_flush = false;
#if 1
    /* Only for varchar insertion */
    std::vector<uint64_t> rowids;
    /* NOTE: Avoid sorting for now to keep insertion order */
    size_t processed = 0;
    while (processed < buffer.size())
    {
        // Simplified Logic: Assume nrecs_in_page_count == 0 at start of flush (or
        // at start of new page). For Fixed Types, capacity_per_page is the limit.
        // For Variable, limit is effectively infinite (byte check handles it).

        size_t available_slots =
            (nrecs_capacity_per_page > 0) ? nrecs_capacity_per_page : std::numeric_limits<size_t>::max();
        size_t batch_size = std::min(buffer.size() - processed, available_slots);

        if constexpr (debug_flush) {
            std::cout << "[Flush] Preparing to flush batch of size " << batch_size
                << " (processed " << processed << " of " << buffer.size() << ")\n";
        }

        rowids.clear();
        rowids.reserve(batch_size);

        // Prepare Batch Vectors
        if (batch_size > 0)
        {
            int ret;
            // Update Stats
            for (size_t k = 0; k < batch_size; ++k)
            {
                if (enable_stats) {
                    update_stats(current_stats, buffer[processed + k].first);
                }
            }
            if constexpr (std::is_same_v<T, int32_t>)
            {
                std::vector<int32_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k) {
                    vals.push_back(buffer[processed + k].first);
                }

                if (compression != CompressionMethod::NONE) {
                    size_t n_aligned = 0;
                    size_t n_actual = vals.size();
                    if (vals.size() % 128 > 0) {
                        size_t padding_needed = 128 - (n_actual % 128);
                        uint32_t _v = vals.back();
                        /* padding_needed elements are inserted */
                        vals.insert(vals.end(), padding_needed, _v);
                        
                        if (sizeof(pag_head) + vals.size() * sizeof(uint32_t) > page_size) {
                            std::cerr << "[ERROR] PFOR compression: Page size exceeded after padding\n";
                            exit(EXIT_FAILURE);
                        }
                    }
                    n_aligned = vals.size();

                    size_t nblocks = vals.size() / 128;
                    comp_ctx.compressed_size = 0;
                    ret = compress_int_with_pfor(
                        reinterpret_cast<uint*>(vals.data()), n_aligned, page_size,
                        comp_ctx.workspaces, comp_ctx.compressed_size);
                    if (ret < 0) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR compression failed\n";
                        exit(EXIT_FAILURE);
                        return;
                    }

                    uint* pfor_comp_values = reinterpret_cast<uint*>(comp_ctx.workspaces[0]);
                    uint* pfor_offsets = reinterpret_cast<uint*>(comp_ctx.workspaces[1]);

                    std::span<uint> pfor_comp_values_span(pfor_comp_values, comp_ctx.compressed_size / sizeof(uint));
                    std::span<uint> pfor_offsets_span(pfor_offsets, nblocks + 1);

                    pag_init(comp_ctx.dst_buffer, page_size);
                    uint32_t compressed_page_size = 0;
                    ret = pagcol_append_batch_unordered_column_int32_comp(comp_ctx.dst_buffer,
                        n_aligned, n_actual,
                        pfor_comp_values_span, comp_ctx.compressed_size, pfor_offsets_span,
                        compressed_page_size, page_size);
                    comp_ctx.compressed_size = compressed_page_size;
                    if (ret == PAG_SLOTID_MASK_ERROR) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR pagcol append failed\n";
                        exit(EXIT_FAILURE);
                    }
                } else {
                    pagcol_append_batch_unordered_column_int32(page_buffer, vals, page_size);
                }
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            else if constexpr (std::is_same_v<T, int64_t>)
            {
                std::vector<int64_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k) {
                    vals.push_back(buffer[processed + k].first);
                }

                if (compression != CompressionMethod::NONE) {
                    size_t n_aligned = 0;
                    size_t n_actual = vals.size();
                    if (vals.size() % 128 > 0) {
                        size_t padding_needed = 128 - (vals.size() % 128);
                        uint64_t _v = vals.back();
                        /* padding_needed elements are inserted */
                        vals.insert(vals.end(), padding_needed, _v);
                        
                        if (sizeof(pag_head) + vals.size() * sizeof(uint64_t) > page_size) {
                            std::cerr << "[ERROR] PFOR compression: Page size exceeded after padding\n";
                            exit(EXIT_FAILURE);
                        }
                    }
                    n_aligned = vals.size();

                    size_t nblocks = vals.size() / 128;
                    comp_ctx.compressed_size = 0;
                    ret = compress_ulong_with_pfor64(
                        reinterpret_cast<ulong*>(vals.data()), n_aligned, page_size,
                        comp_ctx.workspaces, comp_ctx.compressed_size);
                    if (ret < 0) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR64 compression failed\n";
                        exit(EXIT_FAILURE);
                    }

                    const size_t n = vals.size();
                    ulong* pfor64_comp_values = reinterpret_cast<ulong*>(comp_ctx.workspaces[0]);
                    uint* pfor64_offsets = reinterpret_cast<uint*>(comp_ctx.workspaces[1]);

                    std::span<ulong> pfor64_comp_values_span(pfor64_comp_values, comp_ctx.compressed_size / sizeof(ulong));
                    std::span<uint> pfor64_offsets_span(pfor64_offsets, nblocks + 1);

                    pag_init(comp_ctx.dst_buffer, page_size);
                    uint32_t compressed_page_size = 0;
                    int32_t ret = pagcol_append_batch_unordered_column_int64_comp(comp_ctx.dst_buffer,
                        n_aligned, n_actual,
                        pfor64_comp_values_span, comp_ctx.compressed_size, pfor64_offsets_span,
                        compressed_page_size, page_size);
                    
                    
#if 0
                    /* DEBUG */
                    char *comp_src = new char[page_size];
                    memcpy(comp_src, comp_ctx.dst_buffer, compressed_page_size);
                    char *decomp_dst = new char[page_size];
                    char *workspace = new char[page_size];
                    decompress_page_with_pfor64(comp_src, compressed_page_size,
                        decomp_dst, workspace, page_size);
                    delete[] workspace;
                    delete[] decomp_dst;
                    delete[] comp_src;
                    exit(0);
                    /* END DEBUG */
#endif

                    comp_ctx.compressed_size = compressed_page_size;
                    if (ret == PAG_SLOTID_MASK_ERROR) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR64 pagcol append failed\n";
                        exit(EXIT_FAILURE);
                    }
                } else {
                    std::vector<int64_t> vals;
                    vals.reserve(batch_size);
                    for (size_t k = 0; k < batch_size; ++k) {
                        vals.push_back(buffer[processed + k].first);
                    }

                    pagcol_append_batch_unordered_column_int64(page_buffer, vals, page_size);
                }
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            else if constexpr (std::is_same_v<T, DecimalType>)
            {
                std::vector<int32_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k) {
                    vals.push_back(buffer[processed + k].first.value);
                }

                if (compression != CompressionMethod::NONE) {
                    size_t n_aligned = 0;
                    size_t n_actual = vals.size();
                    if (vals.size() % 128 > 0) {
                        size_t padding_needed = 128 - (vals.size() % 128);
                        uint32_t _v = vals.back();
                        /* padding_needed elements are inserted */
                        vals.insert(vals.end(), padding_needed, _v);
                        
                        if (sizeof(pag_head) + vals.size() * sizeof(uint32_t) > page_size) {
                            std::cerr << "[ERROR] PFOR compression: Page size exceeded after padding\n";
                            exit(EXIT_FAILURE);
                        }
                    }
                    n_aligned = vals.size();

                    size_t nblocks = vals.size() / 128;
                    comp_ctx.compressed_size = 0;
                    ret = compress_int_with_pfor(
                        reinterpret_cast<uint*>(vals.data()), n_aligned, page_size,
                        comp_ctx.workspaces, comp_ctx.compressed_size);
                    if (ret < 0) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR compression failed\n";
                        exit(EXIT_FAILURE);
                        return;
                    }

                    const size_t n = vals.size();
                    uint* pfor_comp_values = reinterpret_cast<uint*>(comp_ctx.workspaces[0]);
                    uint* pfor_offsets = reinterpret_cast<uint*>(comp_ctx.workspaces[1]);

                    std::span<uint> pfor_comp_values_span(pfor_comp_values, comp_ctx.compressed_size / sizeof(uint));
                    std::span<uint> pfor_offsets_span(pfor_offsets, nblocks + 1);

                    pag_init(comp_ctx.dst_buffer, page_size);
                    uint32_t compressed_page_size = 0;
                    ret = pagcol_append_batch_unordered_column_int32_comp(comp_ctx.dst_buffer,
                        n_aligned, n_actual,
                        pfor_comp_values_span, comp_ctx.compressed_size, pfor_offsets_span,
                        compressed_page_size, page_size);
                    comp_ctx.compressed_size = compressed_page_size;
                    if (ret == PAG_SLOTID_MASK_ERROR) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR64 pagcol append failed\n";
                        exit(EXIT_FAILURE);
                    }
                } else {
                    pagcol_append_batch_unordered_column_int32(page_buffer, vals, page_size);
                }
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            else if constexpr (std::is_same_v<T, DateType>)
            {
                std::vector<int32_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k) {
                    vals.push_back(buffer[processed + k].first.value);
                }

                if (compression != CompressionMethod::NONE) {
                    size_t n_aligned = 0;
                    size_t n_actual = vals.size();
                    if (vals.size() % 128 > 0) {
                        size_t padding_needed = 128 - (vals.size() % 128);
                        uint32_t _v = vals.back();
                        /* padding_needed elements are inserted */
                        vals.insert(vals.end(), padding_needed, _v);
                        
                        if (sizeof(pag_head) + vals.size() * sizeof(uint32_t) > page_size) {
                            std::cerr << "[ERROR] PFOR compression: Page size exceeded after padding\n";
                            exit(EXIT_FAILURE);
                        }
                    }
                    n_aligned = vals.size();

                    size_t nblocks = n_aligned / 128;
                    comp_ctx.compressed_size = 0;
                    ret = compress_int_with_pfor(
                        reinterpret_cast<uint*>(vals.data()), n_aligned, page_size,
                        comp_ctx.workspaces, comp_ctx.compressed_size);
                    if (ret < 0) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR compression failed\n";
                        exit(EXIT_FAILURE);
                        return;
                    }

                    const size_t n = vals.size();
                    uint* pfor_comp_values = reinterpret_cast<uint*>(comp_ctx.workspaces[0]);
                    uint* pfor_offsets = reinterpret_cast<uint*>(comp_ctx.workspaces[1]);

                    std::span<uint> pfor_comp_values_span(pfor_comp_values, comp_ctx.compressed_size / sizeof(uint));
                    std::span<uint> pfor_offsets_span(pfor_offsets, nblocks + 1);

                    pag_init(comp_ctx.dst_buffer, page_size);
                    uint32_t compressed_page_size = 0;
                    ret = pagcol_append_batch_unordered_column_int32_comp(comp_ctx.dst_buffer,
                        n_aligned, n_actual,
                        pfor_comp_values_span, comp_ctx.compressed_size, pfor_offsets_span,
                        compressed_page_size, page_size);
                    comp_ctx.compressed_size = compressed_page_size;
                    if (ret == PAG_SLOTID_MASK_ERROR) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR64 pagcol append failed\n";
                        exit(EXIT_FAILURE);
                    }
                } else {
                    pagcol_append_batch_unordered_column_int32(page_buffer, vals, page_size);
                }
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            else if constexpr (std::is_same_v<T, CharAsInt>)
            {
                std::vector<int32_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k) {
                    vals.push_back(buffer[processed + k].first.value);
                }
                if (compression == CompressionMethod::PFOR || compression == CompressionMethod::PFOR64) {
                    size_t n_aligned = 0;
                    size_t n_actual = vals.size();
                    if (vals.size() % 128 > 0) {
                        size_t padding_needed = 128 - (vals.size() % 128);
                        uint32_t _v = vals.back();
                        vals.insert(vals.end(), padding_needed, _v);

                        if (sizeof(pag_head) + vals.size() * sizeof(uint32_t) > page_size) {
                            std::cerr << "[ERROR] PFOR compression: Page size exceeded after padding\n";
                            exit(EXIT_FAILURE);
                        }
                    }
                    n_aligned = vals.size();

                    size_t nblocks = vals.size() / 128;
                    comp_ctx.compressed_size = 0;
                    ret = compress_int_with_pfor(
                        reinterpret_cast<uint*>(vals.data()), n_aligned, page_size,
                        comp_ctx.workspaces, comp_ctx.compressed_size);
                    if (ret < 0) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR compression failed\n";
                        exit(EXIT_FAILURE);
                        return;
                    }

                    const size_t n = vals.size();
                    uint* pfor_comp_values = reinterpret_cast<uint*>(comp_ctx.workspaces[0]);
                    uint* pfor_offsets = reinterpret_cast<uint*>(comp_ctx.workspaces[1]);

                    std::span<uint> pfor_comp_values_span(pfor_comp_values, comp_ctx.compressed_size / sizeof(uint));
                    std::span<uint> pfor_offsets_span(pfor_offsets, nblocks + 1);

                    pag_init(comp_ctx.dst_buffer, page_size);
                    uint32_t compressed_page_size = 0;
                    ret = pagcol_append_batch_unordered_column_int32_comp(comp_ctx.dst_buffer,
                        n_aligned, n_actual,
                        pfor_comp_values_span, comp_ctx.compressed_size, pfor_offsets_span,
                        compressed_page_size, page_size);
                    comp_ctx.compressed_size = compressed_page_size;
                    if (ret == PAG_SLOTID_MASK_ERROR) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR pagcol append failed\n";
                        exit(EXIT_FAILURE);
                    }
                } else {
                    /* Non-PFOR compression (LZ4, LZ4PAR, etc.) or no compression:
                       append to page_buffer; flush_page_to_storage handles compression */
                    pagcol_append_batch_unordered_column_int32(page_buffer, vals, page_size);
                }
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            // FSST / FSST_ROWID batch compression for CharType/VCharType
            else if constexpr (std::is_same_v<T, CharType> || std::is_same_v<T, VCharType>)
            {
                if (compression == CompressionMethod::FSST_ROWID) {
                    // ── FSST_ROWID: FSST strings + PFOR64 rowids ──
                    // Build string arrays
                    // For CHAR types, pad to field_size with spaces (0x20)
                    // to match pagcol_append_rec_unordered_column_char semantics.
                    std::vector<std::string> padded_strings;
                    std::vector<size_t> lens(batch_size);
                    std::vector<const unsigned char*> ptrs(batch_size);
                    if constexpr (std::is_same_v<T, CharType>) {
                        padded_strings.resize(batch_size);
                        for (size_t k = 0; k < batch_size; k++) {
                            auto& val = buffer[processed + k].first.value;
                            padded_strings[k] = val;
                            if (padded_strings[k].size() < field_size) {
                                padded_strings[k].append(field_size - padded_strings[k].size(), ' ');
                            }
                            lens[k] = padded_strings[k].size();
                            ptrs[k] = (const unsigned char*)padded_strings[k].data();
                        }
                    } else {
                        for (size_t k = 0; k < batch_size; k++) {
                            auto& val = buffer[processed + k].first.value;
                            lens[k] = val.size();
                            ptrs[k] = (const unsigned char*)val.data();
                        }
                    }

                    // Step 1: FSST train + compress
                    fsst_encoder_t* enc = fsst_create(batch_size, lens.data(),
                        const_cast<const unsigned char**>(ptrs.data()), 0);
                    fsst_decoder_t dec = fsst_decoder(enc);

                    size_t total_raw = 0;
                    for (size_t k = 0; k < batch_size; k++) total_raw += lens[k];

                    size_t comp_buf_size = batch_size * 7 + 2 * total_raw;
                    std::vector<unsigned char> comp_buf(comp_buf_size);
                    std::vector<size_t> comp_lens(batch_size);
                    std::vector<unsigned char*> comp_ptrs(batch_size);
                    fsst_compress(enc, batch_size, lens.data(),
                        const_cast<const unsigned char**>(ptrs.data()),
                        comp_buf_size, comp_buf.data(),
                        comp_lens.data(), comp_ptrs.data());
                    fsst_destroy(enc);

                    // Serialize symbol table
                    uint8_t raw_symtab[FSST_SYMTAB_TOTAL];
                    fsst_serialize_symbol_table(dec, raw_symtab);

                    // Step 2: Collect and PFOR64-compress rowids
                    std::vector<uint64_t> batch_rowids(batch_size);
                    for (size_t k = 0; k < batch_size; k++) {
                        batch_rowids[k] = buffer[processed + k].second;
                    }
                    uint32_t nrecs_padded = (batch_size + 127) & ~127u;
                    batch_rowids.resize(nrecs_padded, batch_rowids.back());

                    uint32_t pfor64_compressed_size = 0;
                    int pfor64_ret = compress_ulong_with_pfor64(
                        batch_rowids.data(), nrecs_padded, page_size,
                        comp_ctx.workspaces, pfor64_compressed_size);
                    if (pfor64_ret < 0) {
                        std::cerr << "[ERROR](" << __LINE__ << ") PFOR64 rowid compression failed\n";
                        exit(EXIT_FAILURE);
                    }

                    uint32_t nulong = pfor64_compressed_size / sizeof(uint64_t);
                    uint32_t nblocks = nrecs_padded / 128;

                    Pfor64CompressedRowids pfor64_rowids;
                    pfor64_rowids.encoded_data = reinterpret_cast<const uint64_t*>(comp_ctx.workspaces[0]);
                    pfor64_rowids.block_starts = reinterpret_cast<const uint32_t*>(comp_ctx.workspaces[1]);
                    pfor64_rowids.nulong = nulong;
                    pfor64_rowids.nblocks = nblocks;
                    pfor64_rowids.nrecs_padded = nrecs_padded;

                    // Step 3: Pack FSST + PFOR64 into page
                    pag_init(comp_ctx.dst_buffer, page_size);
                    uint32_t compressed_page_bytes = 0;
                    uint32_t packed = pagcol_append_batch_unordered_column_vchar_fsst_rowid(
                        comp_ctx.dst_buffer, raw_symtab, batch_size,
                        (const unsigned char* const*)comp_ptrs.data(),
                        comp_lens.data(),
                        pfor64_rowids,
                        9000,
                        compressed_page_bytes, page_size);
                    comp_ctx.compressed_size = compressed_page_bytes;

                    if (packed == 0) {
                        std::cerr << "[ERROR](" << __LINE__ << ") FSST_ROWID page packing failed: 0 records packed\n";
                        exit(EXIT_FAILURE);
                    }
                    nrecs_in_page_count += packed;
                    processed += packed;
                } else if (compression == CompressionMethod::FSST) {
#if 0
                    // ── Old FSST path (before FSST_ROWID) ──
                    // Kept for reference. Identical to the active path below.
#endif
                    // Build string arrays
                    // For CHAR types, pad to field_size with spaces (0x20).
                    std::vector<std::string> padded_strings;
                    std::vector<size_t> lens(batch_size);
                    std::vector<const unsigned char*> ptrs(batch_size);
                    if constexpr (std::is_same_v<T, CharType>) {
                        padded_strings.resize(batch_size);
                        for (size_t k = 0; k < batch_size; k++) {
                            auto& val = buffer[processed + k].first.value;
                            padded_strings[k] = val;
                            if (padded_strings[k].size() < field_size) {
                                padded_strings[k].append(field_size - padded_strings[k].size(), ' ');
                            }
                            lens[k] = padded_strings[k].size();
                            ptrs[k] = (const unsigned char*)padded_strings[k].data();
                        }
                    } else {
                        for (size_t k = 0; k < batch_size; k++) {
                            auto& val = buffer[processed + k].first.value;
                            lens[k] = val.size();
                            ptrs[k] = (const unsigned char*)val.data();
                        }
                    }

                    auto result = compress_strings_with_fsst(
                        ptrs.data(), lens.data(), batch_size,
                        comp_ctx.dst_buffer, page_size);
                    comp_ctx.compressed_size = result.compressed_page_bytes;

                    if (result.packed_count == 0) {
                        std::cerr << "[ERROR](" << __LINE__ << ") FSST page packing failed: 0 records packed\n";
                        exit(EXIT_FAILURE);
                    }
                    nrecs_in_page_count += result.packed_count;
                    processed += result.packed_count;
                } else {
                    // Classic per-record append (LZ4, LZ4PAR, NONE)
                    for (size_t k = 0; k < batch_size; ++k)
                    {
                        auto &item = buffer[processed + k];
                        int res = PAG_SLOTID_MASK_ERROR;

                        if constexpr (std::is_same_v<T, CharType>) {
                            auto value = item.first.value;
                            res = pagcol_append_rec_unordered_column_char(page_buffer, field_size, value.data(), value.size(), page_size);
                        } else {
                            auto value = item.first.value;
                            res = pagcol_append_rec_unordered_column_vchar_with_rowid(page_buffer, field_size, value.data(), value.size(), rowids[k], page_size);
                        }

                        if (res == PAG_SLOTID_MASK_ERROR) {
                            flush_page_to_storage(output_fds, page_buffer, page_id,
                                page_size, nrecs_in_page_count, nrecs_in_page,
                                nios, nios_data, compression, compressed_sizes, comp_ctx,
                                compression_base_page_id,
                                compressed_write_offsets,
                                field_id, __LINE__);
                            if (enable_stats) {
                                save_and_reset_stats(stats_history, current_stats);
                            }
                            page_id++;
                            if (is_final_flush == false && npages_used >= npages_limit) {
                                std::cerr << "[ERROR] Exceeded allocated page limit during flush: "
                                    << npages_used << " > " << npages_limit << std::endl;
                                exit(1);
                            }
                            max_nrecs_in_page_count = std::max(nrecs_in_page_count, max_nrecs_in_page_count);
                            pag_init(page_buffer, page_size);
                            nrecs_in_page_count = 0;
                            k--;
                            continue;
                        }
                        nrecs_in_page_count++;
                    }
                    processed += batch_size;
                }
            }
            // Fallback for others (CharAsInt, etc.)
            else
            {
                for (size_t k = 0; k < batch_size; ++k)
                {
                    auto &item = buffer[processed + k];
                    int res = PAG_SLOTID_MASK_ERROR;

                    if constexpr (std::is_same_v<T, CharAsInt>)
                    {
                        res = pagcol_append_rec_unordered_column_int32(page_buffer, item.first.value, page_size);
                    }
                    else
                    {
                        fprintf(stderr, "Unsupported type for page append in flush_buffer\n");
                        exit(EXIT_FAILURE);
                    }

                    if (res == PAG_SLOTID_MASK_ERROR)
                    {
                        flush_page_to_storage(output_fds, page_buffer, page_id,
                            page_size, nrecs_in_page_count, nrecs_in_page,
                            nios, nios_data, compression, compressed_sizes, comp_ctx,
                            compression_base_page_id,
                            compressed_write_offsets,
                            field_id, __LINE__);
                        if (enable_stats) {
                            save_and_reset_stats(stats_history, current_stats);
                        }
                        page_id++;
                        if (is_final_flush == false && npages_used >= npages_limit) {
                            std::cerr << "[ERROR] Exceeded allocated page limit during flush: "
                                << npages_used << " > " << npages_limit << std::endl;
                            exit(1);
                        }
                        max_nrecs_in_page_count = std::max(nrecs_in_page_count, max_nrecs_in_page_count);
                        pag_init(page_buffer, page_size);
                        nrecs_in_page_count = 0;
                        k--;
                        continue;
                    }
                    nrecs_in_page_count++;
                }
                processed += batch_size;
            }
        }

        // bool fixed_full = (nrecs_capacity_per_page > 0 && nrecs_in_page_count >= nrecs_capacity_per_page);
        // Actually variable logic in loop handles flush.

        //if (fixed_full || (force_flush && nrecs_in_page_count > 0))
        if (nrecs_in_page_count > 0) {
            flush_page_to_storage(output_fds, page_buffer, page_id,
                page_size, nrecs_in_page_count, nrecs_in_page,
                nios, nios_data,
                compression, compressed_sizes, comp_ctx,
                compression_base_page_id, compressed_write_offsets,
                field_id, __LINE__);
            if (enable_stats) {
                save_and_reset_stats(stats_history, current_stats);
            }
            if (enable_sideways_stats) {
                if constexpr (NumSidewaysFields > 0) {
                    for (size_t i = 0; i < NumSidewaysFields; ++i) {
                        save_and_reset_stats(sideways_stats_history[i], sideways_stats[i]);
                    }
                }
            }
            page_id++;
            npages_used++;
            if (is_final_flush == false && npages_used >= npages_limit) {
                std::cerr << "[error] exceeded allocated page limit during flush: "
                    << npages_used << " > " << npages_limit << std::endl;
                exit(1);
            }
            if (npages_used != nrecs_in_page.size()){
                std::cerr << "Error(" << __LINE__ << "): Field ID " << field_id
                    << " npages_used " << npages_used
                    << " != nrecs_in_page.size() " << nrecs_in_page.size() << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (compression != CompressionMethod::NONE){
                if (npages_used != compressed_sizes.size()){
                    std::cerr << "Error(" << __LINE__ << "): Field ID " << field_id
                        << " npages_used " << npages_used
                        << " != compressed_sizes.size() " << compressed_sizes.size() << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }


            max_nrecs_in_page_count = std::max(nrecs_in_page_count, max_nrecs_in_page_count);

            pag_init(page_buffer, page_size);
            nrecs_in_page_count = 0;
        }
    }

#else
    if (compression == CompressionMethod::RLE || compression == CompressionMethod::PFOR) {
        /* Currently, PFOR is only supported */
        std::sort(buffer.begin(), buffer.end());
        if constexpr (debug_flush) {
            std::cout << "[Flush] RLE Sorted " << buffer.size() << " elements.\n";
        }
    } else if (enable_stats) {
        std::sort(buffer.begin(), buffer.end());
        if constexpr (debug_flush) {
            std::cout << "[Flush] For better stats, sorted " << buffer.size() << " elements.\n";
        }
    } else {
        if constexpr (debug_flush) {
            std::cout << "[Flush] Unordered Flush of " << buffer.size() << " elements.\n";
        }
    }

    std::vector<uint64_t> rowids;
    size_t processed = 0;
    while (processed < buffer.size())
    {
        // Simplified Logic: Assume nrecs_in_page_count == 0 at start of flush (or
        // at start of new page). For Fixed Types, capacity_per_page is the limit.
        // For Variable, limit is effectively infinite (byte check handles it).

        size_t available_slots =
            (nrecs_capacity_per_page > 0) ? nrecs_capacity_per_page : std::numeric_limits<size_t>::max();
        size_t batch_size = std::min(buffer.size() - processed, available_slots);

        if constexpr (debug_flush) {
            std::cout << "[Flush] Preparing to flush batch of size " << batch_size
                << " (processed " << processed << " of " << buffer.size() << ")\n";
        }

        rowids.clear();
        rowids.reserve(batch_size);

        // Prepare Batch Vectors
        if (batch_size > 0)
        {
            // Update Stats
            for (size_t k = 0; k < batch_size; ++k)
            {
                if (enable_stats) {
                    update_stats(current_stats, buffer[processed + k].first);
                }
                rowids.push_back(buffer[processed + k].second);
            }

            //  case CompressionMethod::PFOR:
            //  {
            //      int ret = compress_page_with_pfor(
            //          page_buffer, page_size, ctx.dst_buffer, ctx.workspaces, ctx.max_compress_dst_buffer_size, compressed_size);
            //      if (ret < 0) {
            //          std::cerr << "[ERROR](" << line_from << ") PFOR compression failed\n";
            //          exit(EXIT_FAILURE);
            //          return;
            //      }
            //      break;
            //  }
            //  case CompressionMethod::PFOR64:
            //  {
            //      int ret = compress_page_with_pfor64(
            //          page_buffer, page_size, ctx.dst_buffer, ctx.workspaces, ctx.max_compress_dst_buffer_size, compressed_size);
            //      if (ret < 0) {
            //          std::cerr << "[ERROR](" << line_from << ") PFOR64 compression failed\n";
            //          exit(EXIT_FAILURE);
            //          return;
            //      }
            //      break;
            //  }

            if constexpr (std::is_same_v<T, int32_t>)
            {
                if ()
                std::vector<int32_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k)
                    vals.push_back(buffer[processed + k].first);

                pagcol_append_batch_unordered_column_int32_with_rowid(page_buffer, vals, rowids,
                                                           page_size);
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            else if constexpr (std::is_same_v<T, int64_t>)
            {
                std::vector<int64_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k)
                    vals.push_back(buffer[processed + k].first);

                pagcol_append_batch_unordered_column_int64_with_rowid(page_buffer, vals, rowids,
                                                           page_size);
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            else if constexpr (std::is_same_v<T, DecimalType>)
            {
                std::vector<int32_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k)
                    vals.push_back(buffer[processed + k].first.value);

                pagcol_append_batch_unordered_column_int32_with_rowid(page_buffer, vals, rowids,
                                                           page_size);
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            else if constexpr (std::is_same_v<T, DateType>)
            {
                std::vector<int32_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k)
                    vals.push_back(buffer[processed + k].first.value);

                pagcol_append_batch_unordered_column_int32_with_rowid(page_buffer, vals, rowids,
                                                           page_size);
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            else if constexpr (std::is_same_v<T, CharAsInt>)
            {
                std::vector<int32_t> vals;
                vals.reserve(batch_size);
                for (size_t k = 0; k < batch_size; ++k)
                    vals.push_back(buffer[processed + k].first.value);

                pagcol_append_batch_unordered_column_int32_with_rowid(page_buffer, vals, rowids,
                                                           page_size);
                nrecs_in_page_count += batch_size;
                processed += batch_size;
            }
            // Fallback for others or if simple loop needed
            else
            {
                // Classic Loop for Date, Char, VarChar or if Batch logic not applied
                for (size_t k = 0; k < batch_size; ++k)
                {
                    auto &item = buffer[processed + k];
                    int res = PAG_SLOTID_MASK_ERROR;

                    // Retry label inside loop? No, simplified logic:
                    // If Var/Char, we check capability precisely per record.
                    // Fixed logic above calculated `batch_size` based on count.

                    // Perform append
                    if constexpr (std::is_same_v<T, CharAsInt>)
                    {
                        res = pagcol_append_rec_unordered_column_int32(page_buffer, item.first.value, page_size);
                    }
                    else if constexpr (std::is_same_v<T, CharType>)
                    {
                        // res = pagcol_append_rec_unordered_column_char(
                        //     page_buffer, item.first.value, item.second, ps);
                        auto value = item.first.value;
                        res = pagcol_append_rec_unordered_column_char_with_rowid(page_buffer, field_size, value.data(), value.size(), rowids[k], page_size);
                    }
                    else if constexpr (std::is_same_v<T, VCharType>)
                    {
                        // res = pagcol_append_rec_unordered_column_varchar(
                        //     page_buffer, item.first.value, item.second, ps);
                        auto value = item.first.value;
                        res = pagcol_append_rec_unordered_column_vchar_with_rowid(page_buffer, field_size, value.data(), value.size(), rowids[k], page_size);
                    }
                    else
                    {
                        // static_assert(always_false_v<T>, "Unsupported type for page append in flush_buffer");
                        fprintf(stderr, "Unsupported type for page append in flush_buffer\n");
                        exit(EXIT_FAILURE);
                    }

                    if (res == PAG_SLOTID_MASK_ERROR)
                    {
                        // Page Full
                        // Flush current page
                        flush_page_to_storage(output_fds, page_buffer, page_id,
                            page_size, nrecs_in_page_count, nrecs_in_page,
                            nios, nios_data, compression, compressed_sizes, comp_ctx,
                            field_id, __LINE__);
                        save_and_reset_stats(stats_history, current_stats);
                        page_id++;

                        if (is_final_flush == false && npages_used >= npages_limit) {
                            std::cerr << "[ERROR] Exceeded allocated page limit during flush: "
                                << npages_used << " > " << npages_limit << std::endl;
                            exit(1);
                        }
                        
                        max_nrecs_in_page_count = std::max(nrecs_in_page_count, max_nrecs_in_page_count);

                        pag_init(page_buffer, page_size);
                        nrecs_in_page_count = 0;

                        // Retry this item
                        k--; // Decrement to retry
                        continue;
                    }
                    nrecs_in_page_count++;
                }
                processed += batch_size;
            }
        }

        // bool fixed_full = (nrecs_capacity_per_page > 0 && nrecs_in_page_count >= nrecs_capacity_per_page);
        // Actually variable logic in loop handles flush.

        //if (fixed_full || (force_flush && nrecs_in_page_count > 0))
        if (nrecs_in_page_count > 0) {
            flush_page_to_storage(output_fds, page_buffer, page_id,
                page_size, nrecs_in_page_count, nrecs_in_page,
                nios, nios_data,
                compression, compressed_sizes, comp_ctx,
                field_id, __LINE__);
            if (enable_stats) {
                save_and_reset_stats(stats_history, current_stats);
            }
            page_id++;
            npages_used++;
            if (is_final_flush == false && npages_used >= npages_limit) {
                std::cerr << "[error] exceeded allocated page limit during flush: "
                    << npages_used << " > " << npages_limit << std::endl;
                exit(1);
            }
            max_nrecs_in_page_count = std::max(nrecs_in_page_count, max_nrecs_in_page_count);

            pag_init(page_buffer, page_size);
            nrecs_in_page_count = 0;
        }
    }
#endif
    // exit(1);

    /* now, all data is persisted. Clear the buffer. */
    buffer.clear();
}

/**
 * process_column_logic:
 * Parses, Updates Stats, buffers if needed, and Flushes if full.
 */
template <typename T, typename S, typename BoolType, size_t NumSidewaysFields>
void process_column_logic(
    // 1. Data & Identity
    std::string_view part,
    size_t row_id,
    const size_t field_id,
    const rec_type type,
    const size_t field_size,
    BoolType&& is_dirty,

    // 2. Storage & Page State
    char* page_buffer,
    const size_t page_size,
    size_t& page_id,
    size_t& npages_used,
    const size_t npages_limit,
    std::vector<int>& output_fds,

    // 3. Buffer State
    std::vector<SortPair<T>>* record_buffer_vec,
    const size_t max_buffered_page_count,
    size_t& buffered_page_count,
    size_t& freespace,

    // 4. Processing Config
    const CompressionMethod compression,
    CompressionContext& comp_ctx,
    uint64_t compression_base_page_id,
    std::vector<uint64_t> &compressed_write_offsets,

    // 5. Stats & Counters
    bool enable_stats,
    ColumnStats<T>& stats,
    std::vector<ColumnStats<T>>& stats_history,

    bool enable_sideways_stats,
    std::vector<ColumnStats<S>>& sideways_stats,
    std::array<std::vector<ColumnStats<S>>, NumSidewaysFields>& sideways_stats_history,

    const size_t nrecs_capacity_per_page,
    uint32_t& nrecs_in_page_count,
    uint32_t& max_nrecs_in_page_count,
    // Per-page nrecs prefix sum
    std::vector<uint64_t>& task_nrecs_per_page,
    // Compressioned page sizes - 512 bytes aligned
    std::vector<uint32_t>& compressed_sizes,
    // 6. IO counters
    uint64_t& nios_data,
    uint64_t& nsectors_written
) {
    // Assertion (Page Limit Check)
    if (npages_used >= npages_limit) {
        std::cerr << "Error(" << __LINE__ << "): Field ID: " << field_id
            << " Output Page ID " << page_id << " exceeded limit " << npages_limit << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // A. Parse
    T val = parse_value<T>(part);
    
    if (record_buffer_vec) {
        /* 1. Buffred Write Logic */

        record_buffer_vec->push_back({val, row_id});

        /* Strict Separation: Fixed Capacity vs Variable Freespace */
        if constexpr (!std::is_same_v<T, VCharType>) {
             // Fixed Type Logic (INT32, INT64, DATE, CHAR)
            // Check strictly based on Item Count vs Page Capacity
            size_t total_nrecs_limit = max_buffered_page_count * nrecs_capacity_per_page;
            if (record_buffer_vec->size() >= total_nrecs_limit) {
                flush_staging_buffer(*record_buffer_vec, page_buffer, page_id, page_size,
                    npages_used, npages_limit,
                    field_size, compression, comp_ctx, compression_base_page_id, compressed_write_offsets, compressed_sizes,
                    enable_stats, stats, stats_history,
                    enable_sideways_stats, sideways_stats, sideways_stats_history,
                    output_fds,
                    nrecs_capacity_per_page, nrecs_in_page_count, max_nrecs_in_page_count, task_nrecs_per_page, nios_data, nsectors_written, field_id);

                // Reset Buffer State --> really?
                // buffered_page_count = 0;
                // freespace = page_size - sizeof(pag_head);
                // Reset not needed for count logic as buffer is cleared in flush.
                // We don't touch freespace or buffered_page_count for Fixed types.
            }
        } else {
            // Variable Type Logic (VChar)
            // Use existing Freespace (Byte-based) tracking
            // Must match estimation formula: align4(len) + overhead
            constexpr size_t vchar_alignment = 4;
#if 0
            // Old: fixed 16B overhead for all VChar (LZ4/NONE with inline rowid)
            constexpr size_t vchar_overhead = sizeof(uint64_t) + 2 * sizeof(uint16_t) + sizeof(uint32_t); // 16
#endif
            // Must match the sampling phase overhead (L4804) exactly: 16B.
            // Sampling doesn't know the compression type, so it always uses 16B.
            // Using a smaller value here would accumulate more records per buffer
            // than sampling estimated per page, causing a page count mismatch.
            const bool is_fsst_variant = (compression == CompressionMethod::FSST
                                       || compression == CompressionMethod::FSST_ROWID);
            constexpr size_t vchar_overhead = sizeof(uint64_t) + 2 * sizeof(uint16_t) + sizeof(uint32_t); // 16B
            size_t raw_len = val.value.size();
            size_t rec_base_size = (raw_len % vchar_alignment == 0)
                ? raw_len : (raw_len + (vchar_alignment - (raw_len % vchar_alignment)));
            rec_base_size += vchar_overhead;

            bool is_overflow = (rec_base_size > freespace);
            if (is_overflow) {
              buffered_page_count++;
              freespace = (page_size - sizeof(pag_head)) - rec_base_size;
            } else {
              freespace -= rec_base_size;
            }

            if (buffered_page_count >= max_buffered_page_count) {
                // For FSST/FSST_ROWID: the overflow record that triggered the page boundary
                // must be excluded from this flush. FSST packs compressed records
                // directly, so including the overflow record would fit more records
                // per page than the uncompressed estimation, causing page count
                // mismatch between metadata and actual pages written.
                // Non-FSST (LZ4) handles overflow via pagcol_append retry loop.
                bool defer_overflow = (is_fsst_variant && is_overflow);
                if (defer_overflow) {
                    record_buffer_vec->pop_back();
                }

                flush_staging_buffer(*record_buffer_vec, page_buffer, page_id, page_size,
                    npages_used, npages_limit,
                    field_size, compression, comp_ctx, compression_base_page_id, compressed_write_offsets, compressed_sizes,
                    enable_stats, stats, stats_history,
                    enable_sideways_stats, sideways_stats, sideways_stats_history,
                    output_fds,
                    nrecs_capacity_per_page, nrecs_in_page_count, max_nrecs_in_page_count, task_nrecs_per_page, nios_data, nsectors_written, field_id);
                buffered_page_count = 0;
                freespace = page_size - sizeof(pag_head);

                if (defer_overflow) {
                    // Re-add the overflow record to the now-empty buffer
                    record_buffer_vec->push_back({val, row_id});
                    freespace -= rec_base_size;
                }
            }

        }
    } else {
        /* 2. Direct Write Logic */

        if (enable_stats) {
             update_stats(stats, val);
        }

retry_direct:
        int slot_id = PAG_SLOTID_MASK_ERROR;
        size_t ps = page_size;

        // Fixed Types: trying direct append
        if constexpr (std::is_same_v<T, int32_t>) {
            slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val, ps);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            slot_id = pagcol_append_rec_unordered_column_int64(page_buffer, val, ps);
        } else if constexpr (std::is_same_v<T, DecimalType>) {
            slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val.value, ps);
        } else if constexpr (std::is_same_v<T, DateType>) {
            slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val.value, ps);
        } else if constexpr (std::is_same_v<T, CharAsInt>) {
            slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val.value, ps);
        } else if constexpr (std::is_same_v<T, CharType>) {
            auto v = val.value;
            slot_id = pagcol_append_rec_unordered_column_char(page_buffer, field_size, v.data(), v.size(), ps);
        } else if constexpr (std::is_same_v<T, VCharType>) {
            // Variable Type (VChar): Append returns status
            auto v = val.value;
            slot_id = pagcol_append_rec_unordered_column_vchar_with_rowid(page_buffer, field_size, v.data(), v.size(), row_id, ps);
        } else {
            static_assert(always_false_v<T>, "Unsupported type for direct page append");
        }

        if (slot_id == PAG_SLOTID_MASK_ERROR) {
            flush_page_to_storage(output_fds, page_buffer, page_id, ps,
                nrecs_in_page_count, task_nrecs_per_page,
                nios_data, nsectors_written,
                compression, compressed_sizes, comp_ctx,
                compression_base_page_id, compressed_write_offsets,
                field_id, __LINE__);
             
            // Save stats on page full
            if (enable_stats) {
                save_and_reset_stats(stats_history, stats);
            }
            /* sideways stats */
            if (enable_sideways_stats) {
                if constexpr (NumSidewaysFields > 0) {
                    for (size_t i = 0; i < NumSidewaysFields; ++i) {
                        save_and_reset_stats(sideways_stats_history[i], sideways_stats[i]);
                    }
                }
            }

            page_id++;
            npages_used++;
            if (npages_used >= npages_limit) {
                std::cerr << "Error(" << __LINE__ << "): Field ID " << field_id
                    << " Output Page ID " << page_id << " exceeded limit " << npages_limit << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (npages_used != task_nrecs_per_page.size()){
                std::cerr << "Error(" << __LINE__ << "): Field ID " << field_id
                    << " npages_used " << npages_used
                    << " != task_nrecs_per_page.size() " << task_nrecs_per_page.size() << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (compression != CompressionMethod::NONE){
                if (npages_used != compressed_sizes.size()){
                    std::cerr << "Error(" << __LINE__ << "): Field ID " << field_id
                        << " npages_used " << npages_used
                        << " != compressed_sizes.size() " << compressed_sizes.size() << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }

            max_nrecs_in_page_count = std::max(nrecs_in_page_count, max_nrecs_in_page_count);

            pag_init(page_buffer, page_size);
            is_dirty = false;
            nrecs_in_page_count = 0; // Reset
            goto retry_direct;
        } else {
            // Direct append success
            nrecs_in_page_count++;
            is_dirty = true;
        }
    }

}

template <typename T>
int32_t process_column_simple_sample_mode(
    // 1. Data & Identity
    std::string_view part,
    size_t row_id,
    const size_t field_id,
    const rec_type type,
    const size_t field_size,
    bool& is_dirty,

    // 2. Storage & Page State
    char* page_buffer,
    const size_t page_size
) {
    T val = parse_value<T>(part);
    
    /* 2. Direct Write Logic */

    int slot_id = PAG_SLOTID_MASK_ERROR;
     // Fixed Types: trying direct append
    if constexpr (std::is_same_v<T, int32_t>) {
        slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val, page_size);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        slot_id = pagcol_append_rec_unordered_column_int64(page_buffer, val, page_size);
    } else if constexpr (std::is_same_v<T, DecimalType>) {
        slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val.value, page_size);
    } else if constexpr (std::is_same_v<T, DateType>) {
        slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val.value, page_size);
    } else if constexpr (std::is_same_v<T, CharAsInt>) {
        slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val.value, page_size);
    } else if constexpr (std::is_same_v<T, CharType>) {
        auto v = val.value;
        slot_id = pagcol_append_rec_unordered_column_char(page_buffer, field_size, v.data(), v.size(), page_size);
    } else if constexpr (std::is_same_v<T, VCharType>) {
        auto v = val.value;
        slot_id = pagcol_append_rec_unordered_column_vchar_with_rowid(
            page_buffer, field_size, v.data(), v.size(), row_id, page_size);
    } else {
        static_assert(always_false_v<T>, "Unsupported type for direct page append");
    }

    if (slot_id == PAG_SLOTID_MASK_ERROR) {
        //flush_page_to_storage(output_fds, page_buffer, page_id, ps,
        //    nrecs_in_page_count, task_nrecs_per_page,
        //    nios, nios_data, field_id, __LINE__);
        /* notify that sampling is finised. */
        return slot_id;
    } else {
        // Direct append success
        is_dirty = true;
    }

    return slot_id;
}

template <typename T, typename S, typename BoolType, size_t NumSidewaysFields>
int32_t process_column_simple_logic(
    // 1. Data & Identity
    std::string_view part,
    size_t row_id,
    const size_t field_id,
    const rec_type type,
    const size_t field_size,
    BoolType&& is_dirty,

    // 2. Storage & Page State
    char* page_buffer,
    const size_t page_size,
    size_t& page_id,
    size_t& npages_used,
    const size_t npages_limit,
    std::vector<int>& output_fds,

    // 3. Processing Config
    const CompressionMethod compression,
    CompressionContext& comp_ctx,
    uint64_t compression_base_page_id,
    std::vector<uint64_t> &compression_write_offsets,

    bool enable_stats,
    // 4. Stats & Counters
    ColumnStats<T>& stats,
    std::vector<ColumnStats<T>>& stats_history,

    std::vector<ColumnStats<S>>& sideways_stats,
    std::array<std::vector<ColumnStats<S>>, NumSidewaysFields>& sideways_stats_history,

    uint32_t& nrecs_in_page_count,
    uint32_t& max_nrecs_in_page_count,
    // Per-page nrecs prefix sum
    std::vector<uint64_t>& task_nrecs_per_page,
    // Compressioned page sizes - 512 bytes aligned
    std::vector<uint32_t>& compressed_sizes,
    // 6. IO counters
    uint64_t& nios,
    uint64_t& nios_data

) {
    T val = parse_value<T>(part);

    if (enable_stats) {
         update_stats(stats, val);
    }

    /* 2. Direct Write Logic */
retry_direct:
    int slot_id = PAG_SLOTID_MASK_ERROR;
     // Fixed Types: trying direct append
    if constexpr (std::is_same_v<T, int32_t>) {
        slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val, page_size);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        slot_id = pagcol_append_rec_unordered_column_int64(page_buffer, val, page_size);
    } else if constexpr (std::is_same_v<T, DecimalType>) {
        slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val.value, page_size);
    } else if constexpr (std::is_same_v<T, DateType>) {
        slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val.value, page_size);
    } else if constexpr (std::is_same_v<T, CharAsInt>) {
        slot_id = pagcol_append_rec_unordered_column_int32(page_buffer, val.value, page_size);
    } else if constexpr (std::is_same_v<T, CharType>) {
        auto v = val.value;
        slot_id = pagcol_append_rec_unordered_column_char(page_buffer, field_size, v.data(), v.size(), page_size);
    } else if constexpr (std::is_same_v<T, VCharType>) {
        auto v = val.value;
        slot_id = pagcol_append_rec_unordered_column_vchar_with_rowid(page_buffer, field_size, v.data(), v.size(), row_id, page_size);
    } else {
        static_assert(always_false_v<T>, "Unsupported type for direct page append");
    }

    if (slot_id == PAG_SLOTID_MASK_ERROR) {
        flush_page_to_storage(output_fds, page_buffer, page_id, page_size,
            nrecs_in_page_count, task_nrecs_per_page,
            nios, nios_data, compression, compressed_sizes, comp_ctx,
            compression_base_page_id,
            compression_write_offsets, field_id, __LINE__);

        if (enable_stats) {
            /* per-column stats */
            save_and_reset_stats(stats_history, stats);
        }
        /* sideways stats */
        for (size_t i = 0; i < NumSidewaysFields; ++i) {
            save_and_reset_stats(sideways_stats_history[i], sideways_stats[i]);
        }

        page_id++;
        npages_used++;
        if (npages_used >= npages_limit) {
            std::cerr << "[ERROR] Exceeded allocated page limit during flush: "
                << npages_used << " > " << npages_limit << std::endl;
            exit(1);
        }
        max_nrecs_in_page_count = std::max(nrecs_in_page_count, max_nrecs_in_page_count);

        pag_init(page_buffer, page_size);
        nrecs_in_page_count = 0;
        is_dirty = false;
        /* notify that sampling is finised. */
        goto retry_direct;
    } else {
        // Direct append success
        is_dirty = true;
        nrecs_in_page_count++;
    }

    return slot_id;
}

static std::vector<std::string> prep_input_files(LoaderOptions &options, const std::string &col)
{
    std::vector<std::string> files;
    std::stringstream ss_basedir;
    ss_basedir << options.input_dirname << "/" << col;
    auto basedir = ss_basedir.str();
    int n = count_input_files(basedir, col);
    int i;
    for (i = 1; i <= n; i ++) {
        std::stringstream ss;
        if (n == 1)
        {
            ss << basedir << "/" << col << ".tbl";
        }
        else
        {
            ss << basedir << "/" << col << ".tbl." << i;
        }
        std::string path = ss.str();
        if (!fs::exists(fs::status(path))) {
            std::cerr << "File does not exist: " << path << std::endl;
            exit(EXIT_FAILURE);
        }
        files.push_back(path);
    }
    return files;
}

static ssize_t count_file_lines(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open file.\n";
        return -1;
    }

    std::string line;
    size_t count = 0;
    while (std::getline(file, line)) {
        count++;
    }
 
    return count;
}

static std::vector<size_t> create_lbc_thresholds(bool enable_lbc, size_t nclusters, size_t len_min, size_t len_max) {
    std::vector<size_t> lbc_thresholds {};
    if (enable_lbc) {
        size_t diff = (len_max + 1 - len_min);
        if (diff < nclusters) {
            nclusters = diff;
            std::cout << "[WARN] Adjusted lbc_nclusters to " << nclusters << "\n";
        }
        lbc_thresholds.resize(nclusters);
        std::cout << "LBC Length range: [" << len_min << ", " << len_max << "]\n";

        size_t len_step_base = std::max(1UL, diff / nclusters);
        const size_t len_mod = diff % nclusters;
        size_t len_base = len_min;
        for (size_t i = 0; i < nclusters; ++i) {
            size_t len = len_base + len_step_base;
            lbc_thresholds[i] = len;
            if (i < len_mod) len++;
            len_base = len;

            std::cout << "LBC Threshold[" << i << "] = " << lbc_thresholds[i] << "\n";
        }

        if (lbc_thresholds.back() <= len_max) {
            lbc_thresholds.back() = len_max + 1;
            std::cout << "LBC Threshold[" << nclusters - 1 << "] = " << lbc_thresholds.back() << "\n";
        }

    } else {
        lbc_thresholds.push_back(SIZE_MAX);
    }
    return std::move(lbc_thresholds);
}

template <size_t NFields, size_t NumVarCharFields>
static struct SamplingResult<NFields, NumVarCharFields> sampling_lines(
    struct SamplingTask<NFields, NumVarCharFields> &task)
{
    std::array<size_t, NumVarCharFields> min_lens {};
    std::array<size_t, NumVarCharFields> max_lens {};
    std::array<size_t, NFields> npages_estimated {};
    std::array<std::vector<size_t>, NumVarCharFields> npages_estimated_varchar_per_cluster {};

    std::array<size_t, NFields> freespaces_init {};
    std::array<size_t, NFields> freespaces {};

    const bool enable_lbc = task.enable_lbc;
    const size_t lbc_nclusters  = task.lbc_nclusters;

    std::ifstream file(task.path);
    std::array<std::vector<uint16_t>, NumVarCharFields> varchar_lengths;
    if (!file) {
        std::cerr << "Failed to open file: " << task.path << "\n";
        return {-1, min_lens, max_lens, npages_estimated, npages_estimated_varchar_per_cluster};
    }

    std::array<enum rec_type, NFields> field_types;
    std::array<size_t, NFields> field_sizes;
    std::array<size_t, NumVarCharFields> varchar_field_indexes;
    //std::array<std::vector<size_t>, NumVarCharFields> varchar_length_histogram;

    // check maximum memory usage: assuming SF = 1000 (1000 * 60M rows)
    // const size_t assumed_max_rows = 60000000000ULL;
    const size_t assumed_max_rows = 4096 * 1024UL;
    for (size_t vf = 0; vf < NumVarCharFields; vf++) {
        /* avoid memory allocation */
        varchar_lengths[vf].reserve(assumed_max_rows);
    }


    constexpr size_t alignment = 4;
    // std::span<const enum rec_type, NFields> field_types;
    // std::span<const size_t, NFields> field_sizes;
    // std::span<const size_t, NumVarCharFields> varchar_field_indexes;

    std::copy(task.field_sizes.begin(), task.field_sizes.end(), field_sizes.begin());
    std::copy(task.field_types.begin(), task.field_types.end(), field_types.begin());
    std::copy(task.varchar_field_indexes.begin(), task.varchar_field_indexes.end(), varchar_field_indexes.begin());

    
    for (size_t i = 0; i < NFields; i++) {
        npages_estimated[i] = 0;
        auto attr_type = field_types[i];
        switch (attr_type) {
            /* Fixed length jtypes use freespaces[i] as number of records */
            /* Variable-length types use freespaces[i] as page space */
            case rec_type::REC_ATTR_INT16:
            /* NOTE: REC_ATTR_INT16 is hanlded as INT32 */
            case rec_type::REC_ATTR_INT32:
            case rec_type::REC_ATTR_INT64:
            case rec_type::REC_ATTR_DATE:
            case rec_type::REC_ATTR_DECIMAL:
            {
                const size_t freespace = task.page_size - sizeof(struct pag_head);
                const size_t rec_base_size = field_sizes[i];
                size_t rec_size = rec_base_size;
                size_t nrecs_per_page = 0;
                /* +overhead (rowid size) */
                if (task.golap_compression_mode) {
                    // nothing to do
                    nrecs_per_page = freespace / rec_size;
                } else {
                    /* NO rowid is required */
                    // rec_size += sizeof(uint64_t);

                    /* only for baseline method */
                    nrecs_per_page = freespace / rec_size;
                    /* number of records should be aligned to the mutiples number of 128 */
                    if (nrecs_per_page >= 128) {
                        /* for bin packing */
                        nrecs_per_page = (nrecs_per_page / 128) * 128;
                    } else {
                        nrecs_per_page = 128;
                    }
                }
                //std::cout << "Field " << i << ": rec_size=" << rec_size << ", nrecs_per_page=" << nrecs_per_page << "\n";
                // for fixed columns, counting number of recrds
                freespaces_init[i] = nrecs_per_page;
                freespaces[i] = freespaces_init[i];
                break;
            }
            case rec_type::REC_ATTR_CHAR:
            {
                if (field_sizes[i] < 3) {
                    /* CharAsInt: treat as int32_t, same as INT32/DATE/DECIMAL */
                    size_t rec_size = sizeof(int32_t);
                    const size_t freespace = task.page_size - sizeof(struct pag_head);
                    size_t nrecs_per_page = freespace / rec_size;
                    if (!task.golap_compression_mode) {
                        /* 128-alignment only for non-golap (PFOR) mode */
                        if (nrecs_per_page >= 128) {
                            nrecs_per_page = (nrecs_per_page / 128) * 128;
                        } else {
                            nrecs_per_page = 128;
                        }
                    }
                    freespaces_init[i] = nrecs_per_page;
                    freespaces[i] = freespaces_init[i];
                } else {
                    /* char type does not require 128-element alignment */
                    const size_t rec_base_size = field_sizes[i];
                    size_t rec_size = (rec_base_size % alignment == 0)
                        ? rec_base_size : rec_base_size + (alignment - (rec_base_size % alignment));
                    const size_t freespace = task.page_size - sizeof(struct pag_head);
                    const size_t nrecs_per_page = freespace / rec_size;
                    freespaces_init[i] = nrecs_per_page;
                    freespaces[i] = freespaces_init[i];
                }
                break;
            }
            case rec_type::REC_ATTR_VCHAR:
            {
                freespaces_init[i] = task.page_size - sizeof(struct pag_head);
                freespaces[i] = freespaces_init[i];
                break;
            }
            default:
            {
                std::cerr << "Unsupported field type: " << static_cast<int>(attr_type) << "\n";
                return { -1, min_lens, max_lens, npages_estimated, npages_estimated_varchar_per_cluster };
            }
        }
        if (task.verbose) std::cout << "Field " << i << ": freespace=" << freespaces_init[i] << "\n";
    }
    for (size_t vf = 0; vf < NumVarCharFields; vf++) {
        size_t len_schema = field_sizes[varchar_field_indexes[vf]];
        min_lens[vf] = len_schema;
        max_lens[vf] = 0;
    }

    std::string line;
    ssize_t count = 0;

    /* GOLAP compression sampling: every thread samples its own file.
     * Sample 3 positions (beginning, middle, end) to capture data
     * distribution variation across the file. For each field × codec,
     * we keep the worst (max) compressed size across all 3 samples. */
    std::array<std::vector<size_t>, NFields> golap_cps {};
    if (task.golap_compression_mode) {
        const size_t page_size = task.page_size;
        char *pages = (char *)malloc(NFields * page_size);
        std::vector<char*> page_buffers(NFields);
        for (size_t i = 0; i < NFields; i++) {
            page_buffers[i] = pages + (i * page_size);
        }

        // Build candidate codec list based on execution mode
        std::vector<CompressionMethod> candidate_codecs;
        if (task.execution_mode == ExecutionMode::GIDP_BAM_FUSION) {
            candidate_codecs = { CompressionMethod::LZ4 };
        } else {
            candidate_codecs = { CompressionMethod::SNAPPY, CompressionMethod::DEFLATE, CompressionMethod::LZ4 };
        }
        const size_t NCompressionMethods = candidate_codecs.size();

        ZlibContext* zlibcontext = compress_init_zlib();
        size_t max_compress_dst_buffer_size = 0;
        for (size_t i = 0; i < NCompressionMethods; i++) {
            max_compress_dst_buffer_size = std::max(max_compress_dst_buffer_size, compress_max_size(candidate_codecs[i], page_size));
        }
        if (task.verbose) {
            std::cout << "NOTE: Max compress tmp buffer size for candidate codecs:\n";
            for (size_t i = 0; i < NCompressionMethods; i++) {
                std::cout << "  " << compression_method_name(candidate_codecs[i]) << ":\t"
                          << compress_max_size(candidate_codecs[i], page_size) << "\n";
            }
            std::cout << "using max buffer size: " << max_compress_dst_buffer_size << std::endl;
        }

        char *compressed_page = (char *)malloc(max_compress_dst_buffer_size);
        if (!compressed_page) {
            std::cerr << "Failed to allocate memory for compressed page buffer\n";
            free(pages);
            return { -1, min_lens, max_lens, npages_estimated, npages_estimated_varchar_per_cluster };
        }

        // Determine file size for multi-position sampling
        file.seekg(0, std::ios::end);
        const std::streamoff file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Sample positions: beginning (0), middle (50%), tail (75%)
        const std::streamoff sample_offsets[] = { 0, file_size / 2, file_size * 3 / 4 };
        constexpr size_t NUM_SAMPLE_POSITIONS = 3;

        // Initialize worst compressed sizes: [field][codec] = 0
        std::array<std::vector<size_t>, NFields> compressed_page_sizes;
        for (size_t i = 0; i < NFields; i++) {
            compressed_page_sizes[i].resize(NCompressionMethods, 0);
        }

        for (size_t sp = 0; sp < NUM_SAMPLE_POSITIONS; sp++) {
            // Seek to sample position
            file.clear();
            file.seekg(sample_offsets[sp], std::ios::beg);
            if (sample_offsets[sp] > 0) {
                // Discard partial line after seeking to mid-file
                std::getline(file, line);
            }

            // Re-initialize page buffers for this sample position
            for (size_t i = 0; i < NFields; i++) {
                pag_init(page_buffers[i], page_size);
            }
            std::array<bool, NFields> is_dirty {};
            std::array<bool, NFields> sampling_done {};
            size_t nfilled = 0;
            size_t row_id = 0;

            // Fill one page per field from this position
            while (std::getline(file, line)) {
                if (nfilled >= NFields) break;

                auto fields = tpch_split_row(line);
                size_t f = 0;
                for (const auto& part : fields) {
                    if (f >= NFields) break;

                    enum rec_type attr_type = field_types[f];
                    int slotid = 0;
                    if (sampling_done[f]) { f++; continue; }

                    switch (attr_type) {
                        case rec_type::REC_ATTR_INT16:
                        case rec_type::REC_ATTR_INT32:
                            slotid = process_column_simple_sample_mode<int32_t>(
                                part, row_id, f, attr_type,
                                field_sizes[f], is_dirty[f], page_buffers[f], page_size);
                            break;
                        case rec_type::REC_ATTR_DATE:
                            slotid = process_column_simple_sample_mode<DateType>(
                                part, row_id, f, attr_type,
                                field_sizes[f], is_dirty[f], page_buffers[f], page_size);
                            break;
                        case rec_type::REC_ATTR_DECIMAL:
                            slotid = process_column_simple_sample_mode<DecimalType>(
                                part, row_id, f, attr_type,
                                field_sizes[f], is_dirty[f], page_buffers[f], page_size);
                            break;
                        case rec_type::REC_ATTR_INT64:
                            slotid = process_column_simple_sample_mode<int64_t>(
                                part, row_id, f, attr_type,
                                field_sizes[f], is_dirty[f], page_buffers[f], page_size);
                            break;
                        case rec_type::REC_ATTR_CHAR:
                            if (field_sizes[f] < 3) {
                                slotid = process_column_simple_sample_mode<CharAsInt>(
                                    part, row_id, f, attr_type,
                                    sizeof(int32_t), is_dirty[f], page_buffers[f], page_size);
                            } else {
                                slotid = process_column_simple_sample_mode<CharType>(
                                    part, row_id, f, attr_type,
                                    field_sizes[f], is_dirty[f], page_buffers[f], page_size);
                            }
                            break;
                        case rec_type::REC_ATTR_VCHAR:
                            slotid = process_column_simple_sample_mode<VCharType>(
                                part, row_id, f, attr_type,
                                field_sizes[f], is_dirty[f], page_buffers[f], page_size);
                            break;
                        default:
                            std::cerr << "Unsupported field type: " << static_cast<int>(attr_type) << "\n";
                            free(compressed_page);
                            free(pages);
                            return { -1, min_lens, max_lens, npages_estimated, npages_estimated_varchar_per_cluster };
                    }

                    if (slotid == PAG_SLOTID_MASK_ERROR) {
                        sampling_done[f] = true;
                        nfilled++;
                    }
                    f++;
                }
                row_id++;
            }

            // Compress each field's page with each codec; update worst (max) size
            const char *pos_label[] = { "begin", "middle", "tail" };
            for (size_t i = 0; i < NFields; i++) {
                for (size_t j = 0; j < NCompressionMethods; j++) {
                    CompressionMethod method = candidate_codecs[j];
                    memset(compressed_page, 0, max_compress_dst_buffer_size);
                    uint32_t compressed_size = 0;
                    int ret = -1;
                    switch (method) {
                        case CompressionMethod::SNAPPY:
                            ret = compress_page_with_snappy(
                                page_buffers[i], page_size, compressed_page, max_compress_dst_buffer_size, compressed_size);
                            break;
                        case CompressionMethod::DEFLATE:
                            ret = compress_page_with_deflate(
                                zlibcontext, page_buffers[i], page_size, compressed_page, max_compress_dst_buffer_size, compressed_size);
                            break;
                        case CompressionMethod::LZ4:
                            ret = compress_page_with_lz4(
                                page_buffers[i], page_size, compressed_page, max_compress_dst_buffer_size, compressed_size);
                            break;
                        default:
                            std::cerr << "Unsupported compression method\n";
                            free(compressed_page);
                            free(pages);
                            return { -1, min_lens, max_lens, npages_estimated, npages_estimated_varchar_per_cluster };
                    }
                    if (ret < 0) {
                        std::cerr << compression_method_name(method) << " compression failed\n";
                        free(compressed_page);
                        free(pages);
                        return { -1, min_lens, max_lens, npages_estimated, npages_estimated_varchar_per_cluster };
                    }
                    // Keep the worst (max) compressed size across sample positions
                    compressed_page_sizes[i][j] = std::max(compressed_page_sizes[i][j], (size_t)compressed_size);

                    double comp_ratio = (double)compressed_size / (double)page_size;
                    if (task.verbose)
                        std::cout << "[Sampling][File " << task.id << "][" << pos_label[sp]
                                  << "] Field " << i
                                  << ", Method " << compression_method_name(method)
                                  << ", Compressed Size: " << compressed_size
                                  << ", Ratio: " << comp_ratio << std::endl;
                }
            }
        }

        // Report worst-case per field × codec
        golap_cps = compressed_page_sizes;
        for (size_t i = 0; i < NFields; i++) {
            for (size_t j = 0; j < NCompressionMethods; j++) {
                double worst_ratio = (double)compressed_page_sizes[i][j] / (double)page_size;
                if (task.verbose)
                    std::cout << "[Sampling][File " << task.id << "][worst] Field " << i
                              << ", Method " << compression_method_name(candidate_codecs[j])
                              << ", Compressed Size: " << compressed_page_sizes[i][j]
                              << ", Ratio: " << worst_ratio << std::endl;
            }
        }

        file.clear();
        file.seekg(0, std::ios::beg);
        compress_free_zlib(zlibcontext);
        free(compressed_page);
        free(pages);
    }

    {
        while (std::getline(file, line)) {
            size_t f = 0;
            size_t vf = 0;
            enum rec_type attr_type;
            size_t len_schema;
            size_t len;

            auto fields = tpch_split_row(line);

            for (const auto& part : fields) {
                if (f >= NFields) {
                    // Denormialized attributes are ignored.
                    break;
                }

                attr_type = field_types[f];
                size_t base_size;
                size_t overhead;
                switch (attr_type) {
                    case rec_type::REC_ATTR_INT16:
                    case rec_type::REC_ATTR_INT32:
                    case rec_type::REC_ATTR_INT64:
                    case rec_type::REC_ATTR_DATE:
                    case rec_type::REC_ATTR_DECIMAL:
                    case rec_type::REC_ATTR_CHAR:
                    {
                        if (freespaces[f] == 0) {
                            // Need a new page
                            npages_estimated[f]++;
                            freespaces[f] = freespaces_init[f];
                        }
                        freespaces[f]--;
                        break;
                    }
                    case rec_type::REC_ATTR_VCHAR:
                    {
                        // const_cast<char*>(part.data());
                        len_schema = field_sizes[varchar_field_indexes[vf]];
                        len = std::min(len_schema, part.size());

                        varchar_lengths[vf].push_back(static_cast<uint16_t>(len));

                        /* aligned length of varchar string */
                        base_size = len % alignment == 0 ? len : (len + (alignment - (len % alignment)));
                        // sizeof rowid, length(uint16_t), padding(uint16_t), and slotid(uint32_t)
                        // NOTE: For FSST_ROWID, the actual per-record overhead is smaller (8B rowid
                        // stored in a separate PFOR64 section), but sampling doesn't know the
                        // compression type yet. Using the larger 16B overhead is safe (over-estimates
                        // page count, which is always a conservative direction).
                        overhead = sizeof(uint64_t) + 2 * sizeof(uint16_t) + sizeof(uint32_t);

                        min_lens[vf] = std::min(min_lens[vf], len);
                        max_lens[vf] = std::max(max_lens[vf], len);

                        vf++;
#if 0
                        if (task.id == 0) {
                            std::cout << "Field " << f << ": len=" << base_size + overhead << ", base_size=" << base_size << ", overhead=" << overhead << std::endl;
                        }
#endif
                        if (!enable_lbc) {
                            if (freespaces[f] < (base_size + overhead)) {
                                // Need a new page
                                npages_estimated[f]++;
                                freespaces[f] = freespaces_init[f];
                            }
                            freespaces[f] -= (base_size + overhead);
                        }
                        break;
                    }
                    default:
                    {
                        std::cerr << "Unsupported field type: " << static_cast<int>(attr_type) << "\n";
                        return { -1, min_lens, max_lens, npages_estimated, npages_estimated_varchar_per_cluster };
                    }
                }
#if 0
                if (task.id == 0 && f == 15) {
                    std::cout << "Record " << count << ", Field " << f << ": part.size()=" << part.size()
                              << ", base_size=" << base_size << ", overhead=" << overhead
                              << ", freespaces[" << f << "]=" << freespaces[f] << "\n";
                }
#endif
                f++;
            }
            count++;
        }
        for (size_t i = 0; i < NFields; i++) {
            auto attr_type = field_types[i];
            if (attr_type == rec_type::REC_ATTR_VCHAR && enable_lbc) {
                // Skip VARCHAR
                continue;
            }
            /* dirty page. */
            if (freespaces[i] < freespaces_init[i]) {
                npages_estimated[i]++;
            }
            if (task.id == 0 && task.verbose) {
                std::cout << "Path: " << task.path
                          << ", Field Index: " << i
                          << ", Npages: " << npages_estimated[i] << "\n";
            }
        }

        if constexpr (NumVarCharFields > 0) {
            if (enable_lbc) {
                for (size_t vf = 0; vf < NumVarCharFields; ++vf) {
                    task.lbc_min_lens[vf] = min_lens[vf];
                    task.lbc_max_lens[vf] = max_lens[vf];
                }
                /* Exchange min/max len */
                task.barrier->arrive_and_wait();
                for (size_t i = 0; i < task.num_tasks; ++i) {
                    for (size_t vf = 0; vf < NumVarCharFields; vf++) {
                        min_lens[vf] = std::min(min_lens[vf], task.all_tasks[i].lbc_min_lens[vf]);
                        max_lens[vf] = std::max(max_lens[vf], task.all_tasks[i].lbc_max_lens[vf]);
                    }
                }

                if (task.id == 0) {
                    std::cout << "LBC is enabled. Number of clusters: " << lbc_nclusters << "\n";
                    std::cout << "Recalculating the estimated storage size." << lbc_nclusters << "\n";
                }

                for (size_t vf = 0; vf < NumVarCharFields; vf++) {
                    size_t field_index = varchar_field_indexes[vf];
                    enum rec_type attr_type = rec_type::REC_ATTR_VCHAR;

                    const size_t len_min = min_lens[vf];
                    const size_t len_max = max_lens[vf];

                    const size_t f = field_index;
                    // std::cout << "Varchar Field Index: " << field_index
                    //           << ", Min Length: " << len_min
                    //           << ", Max Length: " << len_max
                    //           << ", Npages: " << npages_estimated[f] << "\n";
                    // exit(EXIT_FAILURE);

                    npages_estimated[f] = 0;
                    auto lbc_thresholds = create_lbc_thresholds(enable_lbc, lbc_nclusters, len_min, len_max);

                    std::vector<size_t> freespaces_varchar(lbc_nclusters);
                    npages_estimated_varchar_per_cluster[vf].resize(lbc_nclusters);

                    for (size_t cid = 0; cid < lbc_nclusters; ++cid) {
                        npages_estimated_varchar_per_cluster[vf][cid] = 0;
                        freespaces_varchar[cid] = freespaces_init[f];
                    }

                    if (attr_type != rec_type::REC_ATTR_VCHAR) {
                        std::cerr << "Internal Error: Expected VCHAR type for varchar field index "
                                  << field_index << ", but got type " << static_cast<int>(attr_type) << "\n";
                        exit(EXIT_FAILURE);
                        return { -1, min_lens, max_lens, npages_estimated, npages_estimated_varchar_per_cluster };
                    }

                    for (const size_t len_orig : varchar_lengths[vf]) {
                        const size_t len_aligned =
                            (len_orig % alignment == 0) ? len_orig : (len_orig + (alignment - (len_orig % alignment)));
                        /* sizeof rowid, length(uint16_t), padding(uint16_t), and slotid(uint32_t) */
                        const size_t overhead = sizeof(uint64_t) + 2 * sizeof(uint16_t) + sizeof(uint32_t);

                        /* Determine which cluster this string belongs to */
                        auto it = std::upper_bound(lbc_thresholds.begin(), lbc_thresholds.end(), len_orig);
                        size_t cid = std::distance(lbc_thresholds.begin(), it);
                        const size_t len_total_aligned_size = len_aligned + overhead;

                        // For this cluster, check if it fits in the current page
                        if (freespaces_varchar[cid] < len_total_aligned_size) {
                            // Need a new page
                            npages_estimated_varchar_per_cluster[vf][cid]++;
                            freespaces_varchar[cid] = freespaces_init[f];
                        }
                        freespaces_varchar[cid] -= len_total_aligned_size;
                    }

                    for (size_t cid = 0; cid < lbc_nclusters; ++cid) {
                        /* check dirty page. */
                        if (freespaces_varchar[cid] < freespaces_init[f]) {
                            npages_estimated_varchar_per_cluster[vf][cid]++;
                        }
                        if (task.id == 0 && task.verbose) {
                            std::cout << "Path: " << task.path
                                      << ", Varchar Field Index: " << field_index
                                      << ", Cluster ID: " << cid
                                      << ", Threshold: " << lbc_thresholds[cid]
                                      << ", Npages: " << npages_estimated_varchar_per_cluster[vf][cid] << "\n";
                        }
                    }
                }
            }
        }
    }
 
    return { count, min_lens, max_lens, npages_estimated, npages_estimated_varchar_per_cluster, std::move(golap_cps) };
}


template <size_t NumSidewaysFilters, size_t NumStartXtns, size_t NFields, size_t NumVarCharFields, size_t NumFilters>
static std::vector<ColumnLoadTask<NFields, NumStartXtns, NumVarCharFields, NumFilters, NumSidewaysFilters>> generate_column_tasks(
    LoaderOptions &options,
    TPCHTableMetadata &metadata,
    const TPCH::common::Table table,
    uint64_t &start_page_id_base,
    std::array<uint64_t, NumStartXtns> &start_page_ids_,
    std::vector<std::string> &paths,
    const std::array<size_t, NFields>& field_sizes,
    const std::array<enum rec_type, NFields>& field_types,
    const std::array<size_t, NumVarCharFields>& varchar_field_indexes,
    std::array<std::vector<size_t>, NumVarCharFields> &varchar_cluster_thresholds,
    const std::array<size_t, NumFilters>& filter_field_indexes,
    const std::array<bool, NFields>& enable_stats,
    const std::array<CompressionMethod, NFields>& compression_methods,
    size_t *npages_expected)
{
    if (options.verbose) std::cout << "NumStartXtns: " << NumStartXtns << ", NFields: " << NFields << ", NumVarCharFields: " << NumVarCharFields << std::endl;
    static_assert(NumStartXtns == NFields || NumStartXtns == NFields + NumVarCharFields);
    // constexpr size_t NumRecBufs = NumVarCharFields + 1;
    const uint64_t start_page_id = start_page_id_base;
    
    const size_t page_size = options.page_size;
    const size_t dict_encoding = options.enable_dict_encoding;
    const size_t varchar_to_fixedchar = options.enable_varchar_to_fixedchar;
    const bool compress = options.compress;
    const bool enable_lz4par = options.enable_lz4par;
    const bool enable_fsst = options.enable_fsst;
    const bool enable_lbc = options.lbc_enabled;

    const size_t lbc_num_varchar_clusters = options.lbc_num_varchar_clusters;
    const size_t num_buffers_used = options.enable_dict_encoding ?
        NFields + NumVarCharFields * lbc_num_varchar_clusters : NFields;

    std::vector<SamplingTask<NFields, NumVarCharFields>> sampling_tasks;
    /* The object of this function is to create load_tasks */
    std::vector<ColumnLoadTask<NFields, NumStartXtns, NumVarCharFields, NumFilters, NumSidewaysFilters>> load_tasks;

    std::vector<std::packaged_task<ssize_t()>> tasks;
    std::vector<std::future<SamplingResult<NFields, NumVarCharFields>>> futures;
    std::vector<std::thread> threads;
    std::vector<size_t> nrecs {};


    std::array<size_t, NumVarCharFields> max_varchar_lens;
    std::array<size_t, NumVarCharFields> min_varchar_lens;
    for (size_t i = 0; i < NumVarCharFields; ++i) {
        max_varchar_lens[i] = 0;
        min_varchar_lens[i] = SIZE_MAX;
    }

    {
        size_t i = 0;
        for (auto& path : paths) {
            struct SamplingTask<NFields, NumVarCharFields> task{
                .path = std::ref(path),
                .id = i,
                .page_size = options.page_size,
                .field_types = std::span(field_types),
                .field_sizes = std::span(field_sizes),
                .varchar_field_indexes = std::span(varchar_field_indexes),
                .enable_stats = std::span(enable_stats),
                .enable_lbc = options.lbc_enabled,
                .lbc_nclusters = options.lbc_num_varchar_clusters,
                .golap_compression_mode = options.enable_golap_compression_mode,
                .enable_sideways_stats = options.enable_sideways_stats,
                .execution_mode = options.execution_mode,
                .verbose = (bool)options.verbose
            };
            sampling_tasks.push_back(task);
            i++;
        }
    }

    {
        std::barrier sync_point(sampling_tasks.size());
        size_t i = 0;
        for (auto& task : sampling_tasks) {
            //std::packaged_task<ssize_t()> task(std::bind(count_file_lines, std::ref(path)));
            //futures.push_back(task.get_future());
            //threads.emplace_back(std::move(task));
            task.all_tasks = sampling_tasks.data();
            task.num_tasks = sampling_tasks.size();
            task.barrier = &sync_point;
            auto sampling_func = static_cast<SamplingResult<NFields, NumVarCharFields>(*)(
                struct SamplingTask<NFields, NumVarCharFields> &task)>(&sampling_lines);

            std::packaged_task<SamplingResult<NFields, NumVarCharFields>()> packaged_task(
                std::bind(sampling_func, std::ref(task)));
            futures.push_back(packaged_task.get_future());
            threads.emplace_back(std::move(packaged_task));
            ++i;
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    const size_t ntasks = threads.size();
    size_t count_sum = 0;
    std::vector<size_t> nrecs_prefix_sum(ntasks + 1);
    std::vector<std::array<size_t, NFields>> npages_estimated(ntasks);
    for (size_t i = 0; i < ntasks; ++i) {
        for (size_t j = 0; j < NFields; ++j) {
            npages_estimated[i][j] = 0;
        }
    }

    assert(futures.size() == ntasks);
    assert(npages_estimated.size() == ntasks);

    std::vector<std::array<std::vector<size_t>, NumVarCharFields>> npages_estimated_varchar_per_cluster(ntasks);
    // npages_estimated_varchar_per_cluster[taskid][varchar_field_index][cluster_id]

    {
        std::vector<SamplingResult<NFields, NumVarCharFields>> results;
        for (size_t i = 0; i < futures.size(); ++i) {
            auto result = futures[i].get();
            ssize_t count = result.nlines;

            if (count < 0) {
                std::cerr << "[FATAL] Failed to count lines in file: " << paths[i] << std::endl;
                exit(EXIT_FAILURE);
            }
#ifdef DEBUG_PRINT
            std::cout << "Linecount of \"" << paths[i] << "\" is "
                      << count << std::endl;
#endif
            nrecs.push_back(count);
            nrecs_prefix_sum[i + 1] = nrecs_prefix_sum[i] + count;
            count_sum += count;

            for (size_t j = 0; j < NFields; ++j) {
                npages_estimated[i][j] += result.npages_estimated[j];
                //std::cout << "File " << paths[i] << ", Field " << j
                //          << ", Estimated pages: " << result.npages_estimated[j]
                //          << std::endl;
            }

            for (size_t j = 0; j < NumVarCharFields; ++j) {
                max_varchar_lens[j] = std::max(max_varchar_lens[j], result.max_lens[j]);
                min_varchar_lens[j] = std::min(min_varchar_lens[j], result.min_lens[j]);

                if (max_varchar_lens[j] < min_varchar_lens[j]) {
                    std::cerr << "[FATAL] Inconsistent varchar lengths in file: " << paths[i] << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            /* for npages_estimated_varchar_per_cluster */
            results.push_back(std::move(result));
        }

        for (size_t i = 0; i < ntasks; ++i) {
            for (size_t j = 0; j < NumVarCharFields; ++j) {
                /* set vector by moving from results */
                npages_estimated_varchar_per_cluster[i][j] = results[i].npages_estimated_varchar_per_cluster[j];
            }
        }

        /* GOLAP compression: merge per-thread sampling results.
         * For each field × codec, take the worst (max) compression ratio across
         * all threads. If worst ratio > 0.95 → exclude that codec.
         * Among remaining codecs, pick the one with lowest effective cost
         * (calculate_decomp_cost at worst-case compressed size). */
        if (options.enable_golap_compression_mode) {
            std::vector<CompressionMethod> candidate_codecs;
            if (options.execution_mode == ExecutionMode::GIDP_BAM_FUSION) {
                candidate_codecs = { CompressionMethod::LZ4 };
            } else {
                candidate_codecs = { CompressionMethod::SNAPPY, CompressionMethod::DEFLATE, CompressionMethod::LZ4 };
            }
            const size_t NCodecs = candidate_codecs.size();
            const size_t page_size = options.page_size;

            /* Count how many threads actually produced GOLAP sampling data */
            size_t n_sampled = 0;
            for (size_t t = 0; t < ntasks; t++) {
                if (!results[t].golap_compressed_page_sizes[0].empty())
                    n_sampled++;
            }

            for (size_t i = 0; i < NFields; i++) {
                /* Baseline: NONE */
                double best_cost = calculate_decomp_cost(page_size, page_size, CompressionMethod::NONE);
                CompressionMethod best_method = CompressionMethod::NONE;

                for (size_t j = 0; j < NCodecs; j++) {
                    /* Find worst (max) compressed size across all threads */
                    size_t worst_compressed = 0;
                    bool have_data = false;
                    for (size_t t = 0; t < ntasks; t++) {
                        auto &cps = results[t].golap_compressed_page_sizes[i];
                        if (j < cps.size()) {
                            worst_compressed = std::max(worst_compressed, cps[j]);
                            have_data = true;
                        }
                    }
                    if (!have_data) continue;

                    double worst_ratio = (double)worst_compressed / (double)page_size;
                    double cost = calculate_decomp_cost(worst_compressed, page_size, candidate_codecs[j]);

                    if (options.verbose) std::cout << "[GOLAP Merge] Field " << i
                              << ", Method " << compression_method_name(candidate_codecs[j])
                              << ", Worst Compressed: " << worst_compressed
                              << ", Worst Ratio: " << worst_ratio
                              << ", Effective Cost: " << cost << " us"
                              << " (from " << n_sampled << " samples)" << std::endl;

                    if (worst_ratio > 0.95) continue;  /* threshold: worst case */
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_method = candidate_codecs[j];
                    }
                }

                golap_compression_methods[i] = best_method;
                std::cout << "[GOLAP Merge] Field " << i << ": Selected "
                          << compression_method_name(best_method)
                          << " (cost: " << best_cost << " us)" << std::endl;
            }
        }
    }

    std::cout << "Total linecount is " << count_sum << std::endl;
    if (options.verbose) {
        for (size_t i = 0; i < ntasks; ++i) {
            for (size_t j = 0; j < NFields; ++j) {
                std::cout << "File " << paths[i] << ", Field " << j
                          << ", Estimated pages: " << npages_estimated[i][j]
                          << std::endl;
                // std::cout << "Estimated number of pages for field " << i << ": " << npages_estimated[j][i] << std::endl;
            }
        }
    }

    std::array<std::vector<size_t>, NumVarCharFields> lbc_thresholds {};
    for (size_t i = 0; i < NumVarCharFields; ++i) {
       lbc_thresholds[i] = create_lbc_thresholds(
            options.lbc_enabled, options.lbc_num_varchar_clusters, min_varchar_lens[i], max_varchar_lens[i]
        );

        /* Ensuring the lifetime of vectors for multi-threaded processing */
        varchar_cluster_thresholds[i].assign(
            lbc_thresholds[i].begin(), lbc_thresholds[i].end()
        );
    }

    for (size_t i = 0; i < NumVarCharFields; ++i) {
        for (const auto& len : varchar_cluster_thresholds[i]) {
            if (options.verbose) std::cout << "LBC Thresholds for VarChar Field " << i << ": " << len << "\n";
        }
    }

    /* size_row_base includes the afost (sizeof(uint16_t)) */
    //std::array<size_t, NFields + 1> prefix_sum_npages {};
    //prefix_sum_npages[0] = 0;
    //
    //for (size_t i = 1; i < NFields + 1; ++i) {
    //    prefix_sum_npages[i] = prefix_sum_npages[i - 1];
    //}
    std::vector<std::array<size_t, NFields>> start_page_ids {};
    start_page_ids.resize(ntasks);

    std::vector<std::array<std::vector<size_t>, NumVarCharFields>> start_page_ids_varchar {};
    std::vector<std::array<std::vector<size_t>, NumVarCharFields>> npages_varchar_estimated {};
    start_page_ids_varchar.resize(ntasks);
    npages_varchar_estimated.resize(ntasks);
    for (size_t j = 0; j < ntasks; ++j) {
        for (size_t vf = 0; vf < NumVarCharFields; ++vf) {
            start_page_ids_varchar[j][vf].resize(varchar_cluster_thresholds[vf].size());
            npages_varchar_estimated[j][vf].resize(varchar_cluster_thresholds[vf].size());
        }
    }

    // std::cerr << "Enable LbC:" << enable_lbc << std::endl;
    size_t npages_col_sum = 0;
#if 1
    {
        size_t vf = 0;
        for (size_t i = 0; i < NFields; i++) {
            /* npages for this field[i] */
            size_t npages_sum = 0;

            if (enable_lbc) {
                /* check if this field is varchar field */
                bool is_varchar_field = false;
                size_t varchar_field_idx = 0;
                if (vf < NumVarCharFields && varchar_field_indexes[vf] == i) {
                    is_varchar_field = true;
                    varchar_field_idx = vf;
                    vf++;
                }
                if (is_varchar_field) {
                    /* sum up npages for varchar clusters */
                    const size_t nclusters = varchar_cluster_thresholds[varchar_field_idx].size();
                    for (size_t k = 0; k < nclusters; ++k) {
                        size_t npages_for_varchar_clusters = 0;
                        for (size_t j = 0; j < ntasks; ++j) {
                            /* NOTE: start_page_ids_varchar_fields[task_id][varchar_field_id][cluster_id] */
                            // Trying to allocate sequential page ids for each cluster
                            size_t page_id = start_page_id + npages_col_sum + npages_sum + npages_for_varchar_clusters;
                            start_page_ids_varchar[j][varchar_field_idx][k] = page_id;

                            /* npages_estimated_varchar_per_cluster[taskid][varchar_field_index][cluster_id] */
                            const auto& npages_cluster = npages_estimated_varchar_per_cluster[j][varchar_field_idx][k];
                            npages_for_varchar_clusters += npages_cluster;
                        }
                        npages_sum += npages_for_varchar_clusters;
                    }
                    /* Debug print */
                    if (options.verbose) {
                        for (size_t k = 0; k < nclusters; ++k) {
                            for (size_t j = 0; j < ntasks; ++j) {
                                std::cout << "\tTask[" << j << "], start_page_id: " << start_page_ids_varchar[j][varchar_field_idx][k]
                                    << ", npages: " << npages_estimated_varchar_per_cluster[j][varchar_field_idx][k] << std::endl;
                             }
                        }
                    }
                } else {
                    /* non-varchar field */
                    for (size_t j = 0; j < ntasks; ++j) {
                        /* NOTE: npages[taskid][fieldid] */

                        /* preallocate n pages here */
                        size_t page_id = start_page_id + npages_col_sum + npages_sum;
                        start_page_ids[j][i] = page_id;

                        npages_sum += npages_estimated[j][i];
                    }
#if 0
                    /* Debug print */
                    for (size_t j = 0; j < ntasks; ++j) {
                        std::cout << "\tTask[" << j << "], start_page_id: " << start_page_ids[j][i] << std::endl;
                    }
#endif
                }
            } else {
                /* lbc is disabled */
                for (size_t j = 0; j < ntasks; ++j) {
                    /* NOTE: npages[taskid][fieldid] */

                    /* preallocate n pages here */
                    size_t page_id = start_page_id + npages_col_sum + npages_sum;
                    start_page_ids[j][i] = page_id;

                    npages_sum += npages_estimated[j][i];
                }

#if 0
                /* Debug print */
                for (size_t j = 0; j < ntasks; ++j) {
                    std::cout << "\tTask[" << j << "], start_page_id: " << start_page_ids[j][i] << std::endl;
                }
#endif
            }

            /* calcurate npages based on the result of sampling */
            size_t npages_estimated = npages_sum;
            if (options.verbose) std::cout << "Expected number of pages for field " << i << ": " << npages_estimated << std::endl;

            if (options.verbose) std::cout << "\tfield size: " << field_sizes[i] << "\tstart_page_id: " << start_page_id + npages_col_sum
                    << "\tnpages_sum: " << npages_sum << std::endl;

            metadata_set_page_id(metadata, table, i, start_page_id + npages_col_sum, npages_estimated);
            npages_col_sum += npages_estimated;
        }
    }
    // exit(0);
#else
    for (size_t i = 0; i < NFields; i++) {
        /* npages for this field[i]*/
        size_t npages_sum = 0;

        for (size_t j = 0; j < ntasks; ++j) {
            /* NOTE: npages[taskid][fieldid] */

            /* preallocate n pages here */
            size_t page_id = start_page_id + npages_col_sum + npages_sum;
            start_page_ids[j][i] = page_id;

            npages_sum += npages_estimated[j][i];
        }
        /* calcurate npages based on the result of sampling */
        size_t npages_estimated = npages_sum;
        if (options.verbose) std::cout << "Expected number of pages for field " << i << ": " << npages_estimated << std::endl;
#if 0
        /* Debug print */
        for (size_t j = 0; j < ntasks; ++j) {
            std::cout << "\tTask[" << j << "], start_page_id: " << start_page_ids[j][i] << std::endl;
        }
#endif
        if (options.verbose) std::cout << "\tfield size: " << field_sizes[i] << "\tstart_page_id: " << start_page_id + npages_col_sum
                << "\tnpages_sum: " << npages_sum << std::endl;

        metadata_set_page_id(metadata, table, i, start_page_id + npages_col_sum, npages_estimated);
        npages_col_sum += npages_estimated;
    }
#endif
    // exit(1);

    // if (page_size % NUM_ITEMS_PER_TILE != 0) {
    //     std::cerr << "[FATAL] page_size is not a multiple of TILE_SIZE" << std::endl;
    //     exit(EXIT_FAILURE);
    // }
    /* prepare structure */
    for (size_t i = 0; i < paths.size(); ++i) {
        /* handling the smaller files */
        ColumnLoadTask<NFields, NumStartXtns, NumVarCharFields, NumFilters, NumSidewaysFilters> task{
            .field_sizes = std::span(field_sizes),
            .field_types = std::span(field_types),
            .compression_methods = std::span(compression_methods),
            .varchar_field_indexes = std::span(varchar_field_indexes),
            .filter_field_indexes = std::span(filter_field_indexes),
            .lbc_varchar_cluster_thresholds = std::span(varchar_cluster_thresholds),
            .enable_stats = std::span(enable_stats)
        };
        task.file_id = i;
        task.num_buffers_used = num_buffers_used;
        // task.num_varchar_clusters = num_varchar_clusters;
        task.metadata = &metadata;
        task.nfiles = 1;
        task.base_row_id = nrecs_prefix_sum[i];
        task.nrows = nrecs[i];
        task.field_sizes = field_sizes;
        task.field_types = field_types;
        task.varchar_field_indexes = varchar_field_indexes;

        task.dict_encoding = dict_encoding;
        task.varchar_to_fixedchar = varchar_to_fixedchar;
        task.compress = compress;
        task.enable_lz4par = enable_lz4par;
        task.enable_fsst = enable_fsst;
        task.execution_mode = options.execution_mode;
        task.enable_lbc = enable_lbc;
        task.enable_golap_compression_mode = options.enable_golap_compression_mode;
        task.enable_sideways_stats = options.enable_sideways_stats;

        size_t vf = 0;
        for (size_t j = 0; j < NFields; ++j) {
            if (vf < NumVarCharFields && enable_lbc && varchar_field_indexes[vf] == j) {
                task.start_page_ids_varchar[vf] = start_page_ids_varchar[i][vf];
                task.npages_varchar[vf] = npages_estimated_varchar_per_cluster[i][vf];
                vf++;
            } else {
                task.start_page_ids[j] = start_page_ids[i][j];
                task.npages[j] = npages_estimated[i][j];
            }
        }
        if (compress) {
            size_t vf = 0;
            for (size_t j = 0 ; j < NFields; j++) {
                if (enable_lbc && field_types[j] == REC_ATTR_VCHAR) {
                    const size_t nclusters = varchar_cluster_thresholds[vf].size();
                    task.compressed_page_write_offsets_varchar[vf].resize(nclusters);
                    for (size_t k = 0; k < nclusters; ++k) {
                        task.compressed_page_write_offsets_varchar[vf][k] = std::vector<uint64_t> {};
                        task.compressed_page_write_offsets_varchar[vf][k].resize(options.output_fds.size());
                        const size_t ndevs = options.output_fds.size();
                        for (size_t l = 0; l < ndevs; ++l) {
                            size_t p = start_page_ids_varchar[i][vf][k] + l;
                            uint64_t idevid = p % ndevs;
                            uint64_t lpagid = p / ndevs;
                            task.compressed_page_write_offsets_varchar[vf][k][idevid] = lpagid * page_size;
                            if (options.verbose) std::cout << "Task " << i << ", VarChar Field " << j << ", Cluster " << k
                                      << ", Device " << idevid << ", Local Page ID " << lpagid
                                      << ", Write Offset: " << task.compressed_page_write_offsets_varchar[vf][k][idevid]
                                      << std::endl;
                            // here!
                        }
                    }
                    vf++;
                } else {
                    task.compressed_page_write_offsets[j].resize(options.output_fds.size());
                    for (size_t l = 0; l < options.output_fds.size(); ++l) {
                        // task.compressed_page_write_offsets[j][l] = start_page_ids[i][j] * page_size;
                        const size_t ndevs = options.output_fds.size();
                        for (size_t k = 0; k < ndevs; ++k) {
                            size_t p = start_page_ids[i][j] + k;
                            uint64_t idevid = p % ndevs;
                            uint64_t lpagid = p / ndevs;
                            task.compressed_page_write_offsets[j][idevid] = lpagid * page_size;
                            if (options.verbose) std::cout << "Task " << i << ", Field " << j
                                      << ", Device " << idevid << ", Local Page ID " << lpagid
                                      << ", Write Offset: " << task.compressed_page_write_offsets[j][idevid]
                                      << std::endl;
                        }
                    }
                }
            }
            //task.buf_compress = static_cast<char*>(mb_alloc(page_size));
            /* TODO: use max number of records */
            // task.offsets_local = static_cast<uint*>(
            //     mb_alloc(sizeof(uint) *
            //         ((page_size - sizeof(struct pag_hdr)/(sizeof(uint32_t) + sizeof(uint16_t))) + 1)));
        }
        load_tasks.push_back(task);
    }

    if (options.verbose) std::cout << "Total number of pages: " << npages_col_sum << std::endl;
    assert(npages_col_sum >= NFields);

    size_t npages_dict_sum = 0;
    /* Keep using start_page_id to track npages */
    #if 0
    if (dict_encoding) {
        for (std::size_t i = 0; i < load_tasks.size(); ++i) {
            auto &task = load_tasks[i];
            auto &arr_vec_nrecs_per_page_varchar = vec_arr_vec_nrecs_per_page_varchar[i];
            for (size_t j = 0; j < NumVarCharFields; ++j) {
                auto &vec_nrecs_per_page_varchar = arr_vec_nrecs_per_page_varchar[j];
                auto &vec_npages_for_varchar_clusters = task.arr_vec_npages_for_varchar_clusters[j];
                
                const size_t k = j + 1;
                task.start_page_ids[k] = start_page_id;

                size_t npages_total = 0;
                for (size_t l = 0; l < options.num_varchar_clusters; ++l) {
                    size_t nrecs_per_page_varchar = vec_nrecs_per_page_varchar[l];
                    size_t nrecs = task.nrows;
                    size_t v1 = (nrecs + nrecs_per_page_varchar - 1);
                    size_t npages = (v1 % nrecs_per_page_varchar) ? v1 / nrecs_per_page_varchar + 1 : v1 / nrecs_per_page_varchar;
                    if (npages > 50) {
                        npages += 16; // TODO: this is for avoiding the shortage of pages
                    }
                    vec_npages_for_varchar_clusters.push_back(npages);

                    #if 1
                    std::cout << "Number of records per page: VarChar field " << j << ", cluster " << l << ": "
                        << nrecs_per_page_varchar << ", nrecs: " << nrecs << ", ";
                    std::cout << "VarChar field " << j << ", cluster " << l << ": " << npages << " pages" << std::endl;
                    #endif

                    npages_total += npages;
                }
                size_t npages = (npages_total + XTN::kNumPagesPerXTN - 1) / XTN::kNumPagesPerXTN;
                start_page_id += npages;
                npages_dict_sum += npages;
            }
        }
    }
    #endif
    //exit(10);
    /* returns the number of used pages */
    *npages_expected = npages_col_sum + npages_dict_sum;
    if (options.verbose) std::cout << "Expected number of pages: " << *npages_expected << std::endl;

    size_t count_verify = 0;
    for (size_t i = 0; i < load_tasks.size(); ++i) {
        if (options.verbose) {
            std::cout << "Task " << i << ": fileid = " << load_tasks[i].file_id
                      << ", nrows = " << load_tasks[i].nrows << std::endl;
            for (size_t j = 0; j < NFields; ++j) {
                std::cout << "\tstart_page_id[" << j << "] = " << load_tasks[i].start_page_ids[j];
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        count_verify += load_tasks[i].nrows;
    }
    if (count_verify != count_sum) {
        std::cerr << "[BUG][FATAL] Count verify failed: " << count_verify
                  << " != " << count_sum << std::endl;
        exit(EXIT_FAILURE);
    }
    if (options.verbose) std::cout << "Total tasks: " << load_tasks.size() << std::endl;
    assert(load_tasks.size() <= paths.size());

    return load_tasks;
}


template<typename ReturnType>
ReturnType loader_aligned_alloc(size_t size) {
    size_t alignment = 4096; // 4KB alignment
    void* ptr = nullptr;

    int res = posix_memalign(&ptr, alignment, size);
    if (res != 0) {
        std::cerr << "posix_memalign failed: " << res << "\n";
        std::exit(EXIT_FAILURE);
    }
    return static_cast<ReturnType>(ptr);
}

template <typename Type>
static void save_and_reset_filter(
    rec_type attr_type,
    std::vector<ColumnStats<Type>>& vec_filters,
    ColumnStats<Type>& current_filter
) {
    switch (attr_type) {
        case rec_type::REC_ATTR_INT16:
        case rec_type::REC_ATTR_INT32:
        case rec_type::REC_ATTR_DATE:
        case rec_type::REC_ATTR_DECIMAL:
            vec_filters.push_back(current_filter);
            current_filter.reset();
            return;
        case rec_type::REC_ATTR_INT64:
            fprintf(stderr, "[DEBUG] not yet implemented\n");
            exit(EXIT_FAILURE);

            return;
        default:
            std::cerr << "[ERROR] Unsupported filter attribute type: " 
                      << static_cast<int>(attr_type) << "\n";
            exit(EXIT_FAILURE);
            return;
    }
}

template <typename Type>
static void update_filter(
    rec_type attr_type,
    std::vector<ColumnStats<Type>>& vec_filters,
    ColumnStats<Type>& current_filter,
    const Type value
) {
    switch (attr_type) {
        case rec_type::REC_ATTR_INT16:
        case rec_type::REC_ATTR_INT32:
        case rec_type::REC_ATTR_DATE:
        case rec_type::REC_ATTR_DECIMAL:
            current_filter.update(value);
            return;
        case rec_type::REC_ATTR_INT64:
            fprintf(stderr, "[DEBUG] not yet implemented\n");
            exit(EXIT_FAILURE);

            return;
        default:
            std::cerr << "[ERROR] Unsupported filter attribute type: " 
                      << static_cast<int>(attr_type) << "\n";
            exit(EXIT_FAILURE);
            return;
    }
}

template <typename T>
static void update_stats(ColumnStats<T>& stats, const T& val) {
    stats.update(val);
}

template <typename T>
static void save_and_reset_stats(
    std::vector<ColumnStats<T>>& stats_history,
    ColumnStats<T>& current_stats
) {
    // Only save if initialized (or do we save empty stats for empty pages? user snippet implies just push)
    // User snippet does straight push_back.
    stats_history.push_back(current_stats);
    current_stats.reset();
}

#if 0
template <typename T>
static void save_stats(
    std::vector<ColumnStats<T>>& stats_history,
    ColumnStats<T>& current_stats
) {
    // Only save if initialized (or do we save empty stats for empty pages? user snippet implies just push)
    stats_history.push_back(current_stats);
}

template <typename T>
static void reset_stats(
    ColumnStats<T>& current_stats
) {
    current_stats.reset();
}
#endif


/* Buffering is simplified */
template <typename EnumType, size_t MaxNumBuffers, size_t NumStartXtns, size_t NFields,
    size_t NumVarCharFields, size_t NumFilters, size_t NumSidewaysFields>
static TPCHLoaderThreadStats<MaxNumBuffers, NFields> load_lines_to_table_as_column_with_sideways_stats_golap(
    ColumnLoadTask<NFields, NumStartXtns, NumVarCharFields, NumFilters, NumSidewaysFields>& task,
    std::vector<std::string>& paths,
    std::vector<int>& output_fds,
    const std::array<char*, MaxNumBuffers>& buffers,
    const size_t page_size,
    const std::array<enum rec_type, NumSidewaysFields>& sideways_field_types,
    const std::array<size_t, NumSidewaysFields>& sideways_field_sizes,
    const int dryrun, const int verbose)
{
    /* temporal implementation */
    constexpr size_t NumRecBufs = NFields;
    size_t num_buffers_used = task.num_buffers_used;
    size_t num_varchar_clusters = task.lbc_num_varchar_clusters;
    const size_t base_row_id = task.base_row_id;
    const bool compress = task.compress;
    const bool enable_lbc = task.enable_lbc;
    const bool enable_golap_compression_mode = task.enable_golap_compression_mode;

    TPCHTableMetadata &metadata = *task.metadata;

    std::array<uint64_t, NumStartXtns> start_page_ids{};
    std::array<uint64_t, NumStartXtns> npages_limit{};
    std::array<uint64_t, NFields> field_sizes{};
    std::array<enum rec_type, NFields> field_types{};
    std::array<bool, NFields> enable_stats{};

    std::copy(task.start_page_ids.begin(), task.start_page_ids.end(), start_page_ids.begin());
    std::copy(task.npages.begin(), task.npages.end(), npages_limit.begin());
    std::copy(task.field_sizes.begin(), task.field_sizes.end(), field_sizes.begin());
    std::copy(task.field_types.begin(), task.field_types.end(), field_types.begin());
    std::copy(task.enable_stats.begin(), task.enable_stats.end(), enable_stats.begin());

    std::array<bool, MaxNumBuffers> dirtyflags {};
    std::array<uint64_t, MaxNumBuffers> pagids {};
    /* Tracking free space in pages for varchar fields */
    // std::array<uint64_t, MaxNumBuffers> page_freespace {};

    /* tracking number of pages */
    std::array<size_t, MaxNumBuffers> nios_empty {};
    std::array<size_t, MaxNumBuffers> nios_data {};
    std::array<size_t, MaxNumBuffers> nwritten_sectors {};
    std::array<size_t, MaxNumBuffers> npages_used {};

    std::array<uint32_t, MaxNumBuffers> nrecs_in_page_count {};
    std::array<uint32_t, MaxNumBuffers> max_nrecs_in_page_count {};
    std::array<uint64_t, NumRecBufs> nrecs_inserted_total {};

    for (size_t i = 0; i < num_buffers_used; ++i) {
        // written[i] = false;
        dirtyflags[i] = false;
        pagids[i] = 0;
        nios_empty[i] = 0;
        nios_data[i] = 0;
        nwritten_sectors[i] = 0;
        npages_used[i] = 0;
        nrecs_in_page_count[i] = 0;
        max_nrecs_in_page_count[i] = 0;
        pag_init(buffers[i], page_size);
    }

    for (size_t i = 0; i < num_buffers_used; ++i) {
        pagids[i] = start_page_ids[i];
        for (size_t j = 0; j < npages_limit[i]; ++j) {
            page_pwrite_host(output_fds, buffers[i], pagids[i] + j, page_size);
        }
    }

    std::array<std::map<std::string_view, size_t>, 2> dict_encoding_maps {};
    for (size_t i = 0; i < 2; ++i) {
        dict_encoding_maps[i] = std::map<std::string_view, size_t>{};
        size_t j = 0;
        if (i == 0) {
            for (auto s : TPCH::common::dict_r_name_for_load) {
                dict_encoding_maps[i][s] = j;
                j++;
            }
        } else if (i == 1) {
            for (auto s : TPCH::common::dict_c_mktsegment_for_load) {
                dict_encoding_maps[i][s] = j;
                j++;
            }
        } else {
            std::cerr << "Error: Unsupported dictionary encoding map index." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    auto& task_compressed_sizes = task.compressed_sizes_per_page;
    auto& compressed_page_write_offsets = task.compressed_page_write_offsets;
    auto& task_nrecs_per_page = task.nrecs_per_page;

    std::array<SidewayStatsVariant, NFields> sideways_stats;
    auto& all_sideways_stats_history = task.all_sideways_stats_per_page;
    {
        for (size_t i = 0; i < NFields; i++) {
            /* FIXME: this part should not check the type */
            for (size_t j = 0; j < NumSidewaysFields; j++) {
                all_sideways_stats_history[i][j] = std::vector<ColumnStats<int32_t>>{};
                auto t = sideways_field_types[j];
                switch (t) {
                    case rec_type::REC_ATTR_INT32:
                    case rec_type::REC_ATTR_DATE:
                    case rec_type::REC_ATTR_CHAR: {
                        /* CHAR field is treated as INT32 for stats */
                        sideways_stats[i]= std::vector<ColumnStats<int32_t>>(NumSidewaysFields);
                        for (size_t j = 0; j < NumSidewaysFields; j++) {
                            all_sideways_stats_history[i][j] = std::vector<ColumnStats<int32_t>>{};
                        }
                        break;
                    }
#if 0
                case rec_type::REC_ATTR_INT64:
                    sideways_stats.emplace_back(ColumnStats<int64_t>{});
                    all_sideways_stats_history[f] = std::vector<ColumnStats<int64_t>>{};
                    break;
                case rec_type::REC_ATTR_DECIMAL: // DECIMAL -> int32
                    sideways_stats.emplace_back(ColumnStats<DecimalType>{});
                    all_sideways_stats_history[f] = std::vector<ColumnStats<DecimalType>>{};
                    break;
                case rec_type::REC_ATTR_DATE:  
                    sideways_stats.emplace_back(ColumnStats<DateType>{});
                    all_sideways_stats_history[f] = std::vector<ColumnStats<DateType>>{};
                    break;
                case rec_type::REC_ATTR_VCHAR: 
                    sideways_stats.emplace_back(ColumnStats<VCharType>{});
                    all_sideways_stats_history[f] = std::vector<ColumnStats<VCharType>>{};
                    break;
#endif
                default: 
                    std::cerr << "Error: Unknown rec_type encountered during stats initialization." << std::endl;
                    std::exit(EXIT_FAILURE);
                    break;
                }
            }
        }
    }

    auto& all_stats_history = task.all_stats_history_per_page;
    auto& compression_base_page_ids = start_page_ids;
    std::vector<StatsVariant> all_stats;
    all_stats.reserve(NFields);
    {
        for (size_t f = 0; f < NFields; f++) {
            task_nrecs_per_page[f] = std::vector<uint64_t>{};
            task_compressed_sizes[f] = std::vector<uint32_t>{};
            // auto t : field_types
            auto t = field_types[f];
            switch (t) {
                case rec_type::REC_ATTR_INT32:
                    all_stats.emplace_back(ColumnStats<int32_t>{});
                    all_stats_history[f] = std::vector<ColumnStats<int32_t>>{};
                    break;
                case rec_type::REC_ATTR_INT64:
                    all_stats.emplace_back(ColumnStats<int64_t>{});
                    all_stats_history[f] = std::vector<ColumnStats<int64_t>>{};
                    break;
                case rec_type::REC_ATTR_DECIMAL: // DECIMAL -> int32
                    all_stats.emplace_back(ColumnStats<DecimalType>{});
                    all_stats_history[f] = std::vector<ColumnStats<DecimalType>>{};
                    break;
                case rec_type::REC_ATTR_DATE:  
                    all_stats.emplace_back(ColumnStats<DateType>{});
                    all_stats_history[f] = std::vector<ColumnStats<DateType>>{};
                    break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[f] < 3) {
                        all_stats.emplace_back(ColumnStats<CharAsInt>{});
                        all_stats_history[f] = std::vector<ColumnStats<CharAsInt>>{};
                    } else {
                        all_stats.emplace_back(ColumnStats<CharType>{});
                        all_stats_history[f] = std::vector<ColumnStats<CharType>>{};
                    }
                    break;
                case rec_type::REC_ATTR_VCHAR: 
                    all_stats.emplace_back(ColumnStats<VCharType>{});
                    all_stats_history[f] = std::vector<ColumnStats<VCharType>>{};
                    break;
                default: 
                    std::cerr << "Error: Unknown rec_type encountered during stats initialization." << std::endl;
                    std::exit(EXIT_FAILURE);
                    break;
            }
        }
    }

    CompressionContext comp_ctx {};
    if (compress) {
        init_compression_context(&comp_ctx, page_size, enable_golap_compression_mode /* true */ );
    }

    /* Use sampling-based codec (NONE by default) */
    std::array<CompressionMethod, NFields> compression {};
    for (size_t i = 0; i < NFields; ++i) {
        if (compress) {
            if (enable_golap_compression_mode) {
                compression[i] = golap_compression_methods[i];
            } else {
                compression[i] = task.compression_methods[i];
            }
            if (verbose) std::cout << "Field " << i << " compression method: "
                      << compression_method_name(compression[i]) << std::endl;
        }
    }

    ssize_t nrecs_loaded = 0;
    std::string &path = paths[task.file_id];
    // std::cout << "Processing file: " << path << std::endl;
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open file.\n";
        return { -1, nios_empty, nwritten_sectors, max_nrecs_in_page_count, nrecs_inserted_total };
    }

    std::string line;
    nrecs_inserted_total.fill(0);

    while (std::getline(file, line)) {
        const size_t row_id = base_row_id + nrecs_loaded;
        std::string_view sv(line);
        size_t fi = 0;
        size_t scf = 0; /* sideways char field index */
        size_t f = 0;
        char *pagbuf;
        enum rec_type attr_type;

        auto fields = tpch_split_row(line);
        pagbuf = reinterpret_cast<char*>(buffers[0]);

        if constexpr (NumSidewaysFields > 0) {
            if (fields.size() != NFields + NumSidewaysFields && fields.size() != NFields) {
                std::cerr << "Error: row has " << fields.size()
                          << " fields, expected " << NFields << " or "
                          << NFields + NumSidewaysFields << "\n";
                exit(EXIT_FAILURE);
            }
        } else {
            if (fields.size() < NFields) {
                std::cerr << "Error: row has " << fields.size()
                          << " fields, expected " << NFields << "\n";
                exit(EXIT_FAILURE);
            }
        }

        for (const auto& part : fields) {
            if (fi < NFields) {
                attr_type = field_types[fi];
            }

            // Dispatch
            if (fi < NFields) {
                f = fi;
                switch (field_types[f]) {
                    case rec_type::REC_ATTR_INT32: {
                        auto& stats = std::get<ColumnStats<int32_t>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<int32_t>>>(all_stats_history[f]);
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        process_column_simple_logic<int32_t, int32_t>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            compression[f], comp_ctx,  compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist, sw_stats, sw_stats_hist,
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_INT64: {
                        auto& stats = std::get<ColumnStats<int64_t>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<int64_t>>>(all_stats_history[f]);
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        process_column_simple_logic<int64_t, int32_t>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist, sw_stats, all_sideways_stats_history[f],
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_DECIMAL: {
                        // DECIMAL -> INT32 Logic
                        auto& stats = std::get<ColumnStats<DecimalType>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<DecimalType>>>(all_stats_history[f]);
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        process_column_simple_logic<DecimalType, int32_t>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist, sw_stats, sw_stats_hist,
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_DATE: {
                        auto& stats = std::get<ColumnStats<DateType>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<DateType>>>(all_stats_history[f]);
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        process_column_simple_logic<DateType, int32_t>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist, sw_stats, sw_stats_hist,
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_CHAR: {
                        if (field_sizes[f] < 3) {
                            auto& stats = std::get<ColumnStats<CharAsInt>>(all_stats[f]);
                            auto& hist = std::get<std::vector<ColumnStats<CharAsInt>>>(all_stats_history[f]);
                            auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                            auto& sw_stats_hist = all_sideways_stats_history[f];
                            process_column_simple_logic<CharAsInt, int32_t>(
                                fields[f], row_id,
                                f, field_types[f], sizeof(int32_t), dirtyflags[f],
                                buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                                output_fds,
                                compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                                enable_stats[f], stats, hist, sw_stats, sw_stats_hist,
                                nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                                task_compressed_sizes[f],
                                nios_data[f], nwritten_sectors[f]
                            );
                        } else {
                            auto& stats = std::get<ColumnStats<CharType>>(all_stats[f]);
                            auto& hist = std::get<std::vector<ColumnStats<CharType>>>(all_stats_history[f]);
                            auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                            auto& sw_stats_hist = all_sideways_stats_history[f];
                            process_column_simple_logic<CharType, int32_t>(
                                fields[f], row_id,
                                f, field_types[f], field_sizes[f], dirtyflags[f],
                                buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                                output_fds,
                                compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                                enable_stats[f], stats, hist, sw_stats, sw_stats_hist,
                                nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                                task_compressed_sizes[f],
                                nios_data[f], nwritten_sectors[f]
                            );
                        }
                        break;
                    }
                    case rec_type::REC_ATTR_VCHAR: {
                        auto& stats = std::get<ColumnStats<VCharType>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<VCharType>>>(all_stats_history[f]);
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        process_column_simple_logic<VCharType, int32_t>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist, sw_stats, sw_stats_hist,
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                        break;
                    }
                    default: break;
                }
                ++nrecs_inserted_total[f];
            } else if (fi < NFields + NumSidewaysFields) {
                // Sideways column, skip for now
                size_t f_sw = fi - NFields;
                switch (sideways_field_types[f_sw]) {
                case rec_type::REC_ATTR_INT32: {
                    int32_t value = parse_value<int32_t>(part);
                    for (size_t j = 0; j < NFields; j++) {
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[j]);
                        update_stats(sw_stats[f_sw], value);
                    }
                    break;
                }
                case rec_type::REC_ATTR_DATE: {
                    /* avoid using the abstraction as workaround */
                    int32_t value = date_to_int32(part);
                    for (size_t j = 0; j < NFields; j++) {
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[j]);
                        update_stats(sw_stats[f_sw], value);
                    }
                    break;
                }
                case rec_type::REC_ATTR_CHAR: {
                    //autl& value = parse_value<CharType>(part).value();
                    for (size_t j = 0; j < NFields; j++) {
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[j]);
                        /* confirming that the mapped id is preserved */
                        if (dict_encoding_maps[scf].contains(part)) {
                            update_stats<int32_t>(sw_stats[f_sw], static_cast<int32_t>(dict_encoding_maps[scf][part]) );
                        } else {
                            std::cerr << "[ERROR] Dictionary encoding map is empty for sideways char field index: " << scf << "\n";
                            exit(EXIT_FAILURE);
                        }
                    }
                    scf++;
                    break;
                }
                default: {
                    std::cerr << "[ERROR] Unsupported sideways attribute type: "
                              << static_cast<int32_t>(sideways_field_types[fi]) << "\n";
                    exit(EXIT_FAILURE);
                }
                }
            } else {
                std::cerr << "[ERROR] Field index out of range: " << fi << "\n";
                exit(EXIT_FAILURE);
            }

            ++fi;
        }
        nrecs_loaded++;
    }

    for (size_t f = 0; f < NFields; ++f) {
        if (dirtyflags[f]) {
            char *buf = reinterpret_cast<char*>(buffers[f]);
            flush_page_to_storage(output_fds,
                buf, pagids[f], page_size,
                nrecs_in_page_count[f],
                task_nrecs_per_page[f],
                nios_data[f],
                nwritten_sectors[f],
                compression[f], task_compressed_sizes[f], comp_ctx, compression_base_page_ids[f],
                compressed_page_write_offsets[f],
                f,
                __LINE__
            );
            max_nrecs_in_page_count[f] = std::max(nrecs_in_page_count[f], max_nrecs_in_page_count[f]);
            pagids[f]++;
            npages_used[f]++;

            if (npages_used[f] > npages_limit[f]) {
                std::cerr << "[ERROR](" << __LINE__ << ") Exceeded the allocated number of pages for field " << f
                          << ": used " << npages_used[f]
                          << " > limit " << npages_limit[f] << "\n";
                exit(EXIT_FAILURE);
            }
            if (npages_used[f] != task_nrecs_per_page[f].size()) {
                std::cerr << "[ERROR](" << __LINE__ << ") Mismatch in number of records per page recorded for field " << f
                          << ": npages_used " << npages_used[f]
                          << " != task_nrecs_per_page.size() " << task_nrecs_per_page[f].size() << "\n";
                exit(EXIT_FAILURE);
            }
            if (compression[f] != CompressionMethod::NONE && npages_used[f] != task_compressed_sizes[f].size()) {
                std::cerr << "[ERROR](" << __LINE__ << ") Mismatch in number of compressed sizes recorded for field " << f
                          << ": npages_used " << npages_used[f]
                          << " != compressed_sizes.size() " << task_compressed_sizes[f].size() << "\n";
                exit(EXIT_FAILURE);
            }

            if (verbose) std::cout << "[DEBUG] Final flush: field " << f << ", page id " << pagids[f]-1
                      << ", nrecs in page: " << nrecs_in_page_count[f] << std::endl;

            if (enable_stats[f]) {
                switch (field_types[f]) {
                case rec_type::REC_ATTR_INT32: {
                    auto& stats = std::get<ColumnStats<int32_t>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<int32_t>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                case rec_type::REC_ATTR_INT64: {
                    auto& stats = std::get<ColumnStats<int64_t>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<int64_t>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                case rec_type::REC_ATTR_DECIMAL: {
                    auto& stats = std::get<ColumnStats<DecimalType>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<DecimalType>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                case rec_type::REC_ATTR_DATE: {
                    auto& stats = std::get<ColumnStats<DateType>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<DateType>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                case rec_type::REC_ATTR_CHAR: {
                    if (field_sizes[f] < 3) {
                        auto& stats = std::get<ColumnStats<CharAsInt>>(all_stats[f]);
                        auto& stats_history = std::get<std::vector<ColumnStats<CharAsInt>>>(all_stats_history[f]);
                        save_and_reset_stats(stats_history, stats);
                    } else {
                        auto& stats = std::get<ColumnStats<CharType>>(all_stats[f]);
                        auto& stats_history = std::get<std::vector<ColumnStats<CharType>>>(all_stats_history[f]);
                        save_and_reset_stats(stats_history, stats);
                    }
                    break;
                }
                case rec_type::REC_ATTR_VCHAR: {
                    auto& stats = std::get<ColumnStats<VCharType>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<VCharType>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                default:
                    std::cerr << "Error: Unknown rec_type encountered during stats saving." << std::endl;
                    std::exit(EXIT_FAILURE);
                    break;
                }
                /* per-column stats */
            }
            /* sideways stats */
            for (size_t i = 0; i < NumSidewaysFields; ++i) {
                auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                save_and_reset_stats<int32_t>(all_sideways_stats_history[f][i], sw_stats[i]);
            }

            dirtyflags[f] = false; // Reset offset after writing
        }
    }
 
    if (compress) {
        free_compression_context(&comp_ctx);
    }

    return { nrecs_loaded, nios_data, nwritten_sectors, max_nrecs_in_page_count, nrecs_inserted_total };
}


template <typename EnumType, size_t MaxNumBuffers, size_t NumStartXtns, size_t NFields,
    size_t NumVarCharFields, size_t NumFilters, size_t NumSidewaysFields>
static TPCHLoaderThreadStats<MaxNumBuffers, NFields> load_lines_to_table_as_column_with_sideways_stats(
    ColumnLoadTask<NFields, NumStartXtns, NumVarCharFields, NumFilters, NumSidewaysFields>& task,
    std::vector<std::string>& paths,
    std::vector<int>& output_fds,
    const std::array<char*, MaxNumBuffers>& buffers,
    const size_t page_size,
    const std::array<enum rec_type, NumSidewaysFields>& sideways_field_types,
    const std::array<size_t, NumSidewaysFields>& sideways_field_sizes,
    const int dryrun, const int verbose)
{
    /* temporal implementation */
    constexpr size_t NumRecBufs = NFields;
    size_t num_buffers_used = task.num_buffers_used;
    size_t num_varchar_clusters = task.lbc_num_varchar_clusters;
    const size_t base_row_id = task.base_row_id;
    const bool compress = task.compress;
    const bool enable_lz4par = task.enable_lz4par;
    const bool enable_fsst = task.enable_fsst;
    const bool enable_lbc = task.enable_lbc;
    const bool enable_golap_compression_mode = task.enable_golap_compression_mode;
    const bool enable_sideways_stats = true;

    TPCHTableMetadata &metadata = *task.metadata;

    std::array<uint64_t, NumStartXtns> start_page_ids{};
    std::array<uint64_t, NumStartXtns> npages_limit{};
    std::array<std::vector<uint64_t>, NumVarCharFields> start_page_ids_varchar{};
    std::array<std::vector<uint64_t>, NumVarCharFields> compression_base_page_ids_varchar{};
    std::array<std::vector<uint64_t>, NumVarCharFields> lbc_npages_limit{};
    std::array<uint64_t, NFields> field_sizes{};
    std::array<enum rec_type, NFields> field_types{};
    std::array<uint64_t, NumVarCharFields> varchar_field_indexes{};
    std::array<std::vector<size_t>, NumVarCharFields> varchar_cluster_thresholds{};
    std::array<bool, NFields> enable_stats{};

    std::copy(task.start_page_ids.begin(), task.start_page_ids.end(), start_page_ids.begin());
    std::copy(task.npages.begin(), task.npages.end(), npages_limit.begin());
    if (enable_lbc) {
        lbc_npages_limit = task.npages_varchar;
        start_page_ids_varchar = task.start_page_ids_varchar;
        compression_base_page_ids_varchar = start_page_ids_varchar;
    }
    std::copy(task.varchar_field_indexes.begin(), task.varchar_field_indexes.end(), varchar_field_indexes.begin());
    std::copy(task.lbc_varchar_cluster_thresholds.begin(), task.lbc_varchar_cluster_thresholds.end(), varchar_cluster_thresholds.begin());
    std::copy(task.enable_stats.begin(), task.enable_stats.end(), enable_stats.begin());
    std::copy(task.field_sizes.begin(), task.field_sizes.end(), field_sizes.begin());
    std::copy(task.field_types.begin(), task.field_types.end(), field_types.begin());
    std::copy(task.enable_stats.begin(), task.enable_stats.end(), enable_stats.begin());

    std::array<bool, MaxNumBuffers> dirtyflags {};
    std::array<std::vector<bool>, NumVarCharFields> dirtyflags_varchar {};
    std::array<uint64_t, MaxNumBuffers> pagids {};
    std::array<std::vector<uint64_t>, MaxNumBuffers> lbc_pagids {};
    /* Tracking free space in pages for varchar fields */
    std::array<uint64_t, MaxNumBuffers> page_freespace {};

    /* tracking number of pages */
    std::array<size_t, MaxNumBuffers> nios_empty {};
    std::array<size_t, MaxNumBuffers> nios_data {};
    std::array<size_t, MaxNumBuffers> nwritten_sectors {};
    std::array<size_t, MaxNumBuffers> npages_used {};
    std::array<std::vector<size_t>, NumVarCharFields> lbc_npages_used {};
    std::array<std::vector<size_t>, NumVarCharFields> arr_vec_start_pagids_for_varchar_clusters {};

    std::array<uint32_t, MaxNumBuffers> nrecs_in_page_count {};
    std::array<uint32_t, MaxNumBuffers> max_nrecs_in_page_count {};
    std::array<uint64_t, NumRecBufs> nrecs_inserted_total {};

    std::array<std::vector<uint32_t>, NumVarCharFields> nrecs_in_page_count_varchar {};
    std::array<uint32_t, NumVarCharFields> max_nrecs_in_page_count_varchar {};

    for (size_t i = 0; i < num_buffers_used; ++i) {
        // written[i] = false;
        dirtyflags[i] = false;
        pagids[i] = 0;
        nios_empty[i] = 0;
        nios_data[i] = 0;
        nwritten_sectors[i] = 0;
        npages_used[i] = 0;
        nrecs_in_page_count[i] = 0;
        max_nrecs_in_page_count[i] = 0;
        pag_init(buffers[i], page_size);
        page_freespace[i] = page_size - sizeof(struct pag_head);
    }

    for (size_t i = 0; i < num_buffers_used; ++i) {
        pagids[i] = start_page_ids[i];
        for (size_t j = 0; j < npages_limit[i]; ++j) {
            page_pwrite_host(output_fds, buffers[i], pagids[i] + j, page_size);
        }
    }
    auto& compression_base_page_ids = start_page_ids;

    std::array<std::map<std::string_view, size_t>, 2> dict_encoding_maps {};
    for (size_t i = 0; i < 2; ++i) {
        dict_encoding_maps[i] = std::map<std::string_view, size_t>{};
        size_t j = 0;
        if (i == 0) {
            for (auto s : TPCH::common::dict_r_name_for_load) {
                dict_encoding_maps[i][s] = j;
                j++;
            }
        } else if (i == 1) {
            for (auto s : TPCH::common::dict_c_mktsegment_for_load) {
                dict_encoding_maps[i][s] = j;
                j++;
            }
        } else {
            std::cerr << "Error: Unsupported dictionary encoding map index." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }


    /* init LbC */
    //std::vector<std::array<size_t, 1>> varchar_field_sizes {};
    //for (size_t i = 0; i < NumVarCharFields; ++i) {
    //    assert(NumVarCharFields == varchar_field_indexes.size());
    //    varchar_field_sizes.push_back(std::array<size_t, 1> {
    //        field_sizes[varchar_field_indexes[i]]
    //    });
    //}
    std::array<std::vector<char*>, NumVarCharFields> lbc_buffers {};
    std::array<char*, NumVarCharFields> lbc_buffer_starts {};
    std::array<std::vector<std::vector<uint64_t>>, NumVarCharFields> &task_nrecs_per_page_varchar = task.nrecs_per_page_varchar;
    auto& task_compressed_sizes_varchar = task.compressed_sizes_per_page_varchar;
    auto& compressed_page_write_offsets_varchar = task.compressed_page_write_offsets_varchar;
    std::array<std::vector<SidewayStatsVariant>, NumVarCharFields> sideways_stats_varchar;
    //;std::vector<std::array<ColumnStats<int32_t>, NumVarCharFields>> sideways_stats_varchar;
    auto& task_sideways_stats_per_page_varchar = task.all_sideways_stats_per_page_varchar;
    // std::vector<uint64_t> task_nrecs_per_page
    if (enable_lbc) {
        for (size_t i = 0; i < NumVarCharFields; ++i) {
            const size_t nclusters = varchar_cluster_thresholds[i].size();
            lbc_buffer_starts[i] = loader_aligned_alloc<char*>(page_size * nclusters);
            lbc_buffers[i] = std::vector<char*>{};
            dirtyflags_varchar[i] = std::vector<bool>(nclusters, false);
            for (size_t k = 0; k < nclusters; ++k) {
                lbc_buffers[i].push_back(&lbc_buffer_starts[i][k * page_size]);
                pag_init(lbc_buffers[i].back(), page_size);

            }
            lbc_pagids[i] = start_page_ids_varchar[i];
            lbc_npages_used[i] = std::vector<size_t>(nclusters, 0);

            for (size_t k = 0; k < nclusters; ++k) {
                for (size_t j = 0; j < lbc_npages_limit[i][k]; ++j) {
                    page_pwrite_host(output_fds, lbc_buffers[i][k], lbc_pagids[i][k] + j, page_size);
                }
            }

            nrecs_in_page_count_varchar[i] = std::vector<uint32_t>(nclusters, 0);
            task_nrecs_per_page_varchar[i].resize(nclusters);
            for (size_t k = 0; k < nclusters; ++k) {
                task_nrecs_per_page_varchar[i][k] = std::vector<uint64_t> {};
            }

            if (compress) {
                task_compressed_sizes_varchar[i].resize(nclusters);
                for (size_t k = 0; k < nclusters; ++k) {
                    task_compressed_sizes_varchar[i][k] = std::vector<uint32_t> {};
                }
            }
            for (size_t j = 0; j < NumSidewaysFields; ++j) {
                sideways_stats_varchar[i].resize(nclusters);
                // sideways_stats_varchar[vf][cid][sideways_field_index]

                for (size_t k = 0; k < nclusters; ++k) {
                    sideways_stats_varchar[i][k] = std::vector<ColumnStats<int32_t>>(NumSidewaysFields);
                }
                task_sideways_stats_per_page_varchar[i].resize(nclusters);
                //for (size_t k = 0; k < nclusters; ++k) {
                //    task_sideways_stats_per_page_varchar[i][j][k] = std::vector<ColumnStats<int32_t>>{};
                //}
                /* initialize sideways stats per varchar cluster */
            }
            max_nrecs_in_page_count_varchar[i] = 0;
        }

        for (size_t i = 0; i < NumVarCharFields; ++i) {
            const size_t nclusters = varchar_cluster_thresholds[i].size();
            for (size_t k = 0; k < nclusters; ++k) {
                    std::cout << "\tTask start_page_id: " << start_page_ids_varchar[i][k]
                        << ", npages: " << lbc_npages_limit[i][k] << std::endl;
            }
        }
        // exit(1);
    }


    std::copy(task.start_page_ids.begin(), task.start_page_ids.end(), start_page_ids.begin());
    std::copy(task.npages.begin(), task.npages.end(), npages_limit.begin());
    if (enable_lbc) {
        lbc_npages_limit = task.npages_varchar;
        start_page_ids_varchar = task.start_page_ids_varchar;
        //std::copy(task.start_page_ids_varchar.begin(), task.start_page_ids_varchar.end(), start_page_ids_varchar.begin());
        //std::copy(task.npages_varchar.begin(), task.npages_varchar.end(), .begin());
        //std::copy(task.start_page_ids_varchar.begin(), task.start_page_ids_varchar.end(), start_page_ids_varchar.begin());
    }
    std::copy(task.field_sizes.begin(), task.field_sizes.end(), field_sizes.begin());
    std::copy(task.field_types.begin(), task.field_types.end(), field_types.begin());
    std::copy(task.varchar_field_indexes.begin(), task.varchar_field_indexes.end(), varchar_field_indexes.begin());
    std::copy(task.lbc_varchar_cluster_thresholds.begin(), task.lbc_varchar_cluster_thresholds.end(), varchar_cluster_thresholds.begin());
    std::copy(task.enable_stats.begin(), task.enable_stats.end(), enable_stats.begin());
    auto &arr_vec_npages_for_varchar_clusters = task.arr_vec_npages_for_varchar_clusters;
    // std::copy(task.arr_vec_npages_for_varchar_clusters.begin(), task.arr_vec_npages_for_varchar_clusters.end(), arr_vec_npages_for_varchar_clusters.begin());

    auto& task_compressed_sizes = task.compressed_sizes_per_page;
    auto& compressed_page_write_offsets = task.compressed_page_write_offsets;
    auto& task_nrecs_per_page = task.nrecs_per_page;

    std::array<SidewayStatsVariant, NFields> sideways_stats;
    auto& all_sideways_stats_history = task.all_sideways_stats_per_page;
    {
        for (size_t i = 0; i < NFields; i++) {
            /* FIXME: this part should not check the type */
            for (size_t j = 0; j < NumSidewaysFields; j++) {
                all_sideways_stats_history[i][j] = std::vector<ColumnStats<int32_t>>{};
                auto t = sideways_field_types[j];
                switch (t) {
                    case rec_type::REC_ATTR_INT32:
                    case rec_type::REC_ATTR_DATE:
                    case rec_type::REC_ATTR_CHAR: {
                        /* CHAR field is treated as INT32 for stats */
                        sideways_stats[i]= std::vector<ColumnStats<int32_t>>(NumSidewaysFields);
                        for (size_t j = 0; j < NumSidewaysFields; j++) {
                            all_sideways_stats_history[i][j] = std::vector<ColumnStats<int32_t>>{};
                        }
                        break;
                    }
#if 0
                case rec_type::REC_ATTR_INT64:
                    sideways_stats.emplace_back(ColumnStats<int64_t>{});
                    all_sideways_stats_history[f] = std::vector<ColumnStats<int64_t>>{};
                    break;
                case rec_type::REC_ATTR_DECIMAL: // DECIMAL -> int32
                    sideways_stats.emplace_back(ColumnStats<DecimalType>{});
                    all_sideways_stats_history[f] = std::vector<ColumnStats<DecimalType>>{};
                    break;
                case rec_type::REC_ATTR_DATE:  
                    sideways_stats.emplace_back(ColumnStats<DateType>{});
                    all_sideways_stats_history[f] = std::vector<ColumnStats<DateType>>{};
                    break;
                case rec_type::REC_ATTR_VCHAR: 
                    sideways_stats.emplace_back(ColumnStats<VCharType>{});
                    all_sideways_stats_history[f] = std::vector<ColumnStats<VCharType>>{};
                    break;
#endif
                default: 
                    std::cerr << "Error: Unknown rec_type encountered during stats initialization." << std::endl;
                    std::exit(EXIT_FAILURE);
                    break;
                }
            }
        }
    }

    auto& all_stats_history = task.all_stats_history_per_page;
    std::vector<StatsVariant> all_stats;
    all_stats.reserve(NFields);
    {
        for (size_t f = 0; f < NFields; f++) {
            task_nrecs_per_page[f] = std::vector<uint64_t>{};
            task_compressed_sizes[f] = std::vector<uint32_t>{};
            // auto t : field_types
            auto t = field_types[f];
            switch (t) {
                case rec_type::REC_ATTR_INT32:
                    all_stats.emplace_back(ColumnStats<int32_t>{});
                    all_stats_history[f] = std::vector<ColumnStats<int32_t>>{};
                    break;
                case rec_type::REC_ATTR_INT64:
                    all_stats.emplace_back(ColumnStats<int64_t>{});
                    all_stats_history[f] = std::vector<ColumnStats<int64_t>>{};
                    break;
                case rec_type::REC_ATTR_DECIMAL: // DECIMAL -> int32
                    all_stats.emplace_back(ColumnStats<DecimalType>{});
                    all_stats_history[f] = std::vector<ColumnStats<DecimalType>>{};
                    break;
                case rec_type::REC_ATTR_DATE:  
                    all_stats.emplace_back(ColumnStats<DateType>{});
                    all_stats_history[f] = std::vector<ColumnStats<DateType>>{};
                    break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[f] < 3) {
                        all_stats.emplace_back(ColumnStats<CharAsInt>{});
                        all_stats_history[f] = std::vector<ColumnStats<CharAsInt>>{};
                    } else {
                        all_stats.emplace_back(ColumnStats<CharType>{});
                        all_stats_history[f] = std::vector<ColumnStats<CharType>>{};
                    }
                    break;
                case rec_type::REC_ATTR_VCHAR: 
                    all_stats.emplace_back(ColumnStats<VCharType>{});
                    all_stats_history[f] = std::vector<ColumnStats<VCharType>>{};
                    break;
                default: 
                    std::cerr << "Error: Unknown rec_type encountered during stats initialization." << std::endl;
                    std::exit(EXIT_FAILURE);
                    break;
            }
        }
    }

    CompressionContext comp_ctx {};
    if (compress) {
        init_compression_context(&comp_ctx, page_size, enable_golap_compression_mode /* false */ );
    }

    /* Use sampling-based codec (NONE by default) */
    std::array<CompressionMethod, NFields> compression {};
    for (size_t i = 0; i < NFields; ++i) {
        if (compress) {
            if (enable_golap_compression_mode) {
                std::cerr << "[ERROR] Unexpected call of " << __func__ << ". Stop.\n";
                exit(EXIT_FAILURE); // Not supported yet
            } else {
                compression[i] = task.compression_methods[i];
                if (compression[i] == CompressionMethod::LZ4 && enable_lz4par) {
                    if (verbose) std::cout << "Field " << i << " compression method: LZ4-PAR\n";
                    compression[i] = CompressionMethod::LZ4PAR;
                }
                if (compression[i] == CompressionMethod::LZ4 && enable_fsst
                    && ((field_types[i] == rec_type::REC_ATTR_CHAR && field_sizes[i] >= 3)
                        || field_types[i] == rec_type::REC_ATTR_VCHAR)) {
                    compression[i] = CompressionMethod::FSST_ROWID;
                }
            }
            if (verbose) std::cout << "Field " << i << " compression method: "
                      << compression_method_name(compression[i]) << std::endl;
        }
    }

    /* buffering configuration */
    std::array<size_t, NFields> max_buffered_page_counts{};
    std::array<size_t, NFields> buffered_page_counts{};
    for(size_t i = 0; i < NFields; ++i) {
        // Tiered Buffering:
        // 32 Pages for Sort/Stats columns (High Pruning/Compression potential)
        // 1 Page for Streaming columns (Low latency/overhead)
        //if (compression[i] == CompressionMethod::RLE || enable_stats[i]) {
        //    max_buffered_page_counts[i] = 32;
        //} else {
        //    max_buffered_page_counts[i] = 1;
        //}
        max_buffered_page_counts[i] = 1;

        if (verbose) std::cout << "Col " << i << ": Max Buffered Pages=" << max_buffered_page_counts[i]
                  << ((max_buffered_page_counts[i] > 1) ? " [Sort Mode]" : " [Streaming Mode]")
                  << "\n";
    }

    /* Defining thresholds for fields */
    std::array<uint64_t, NFields> nrecs_per_page_capacity {};
    for (size_t i = 0; i < NFields; ++i) {
        size_t freespace = page_size - sizeof(struct pag_head);
        auto attrtype = field_types[i];
        size_t rec_size;
        constexpr size_t alignment = 4;
        switch (attrtype) {
            /* NOTE: implicit ordering for fixed lengith attributes */
            case rec_type::REC_ATTR_INT16:
            {
                rec_size = sizeof(int32_t);
                break;
            }
            case rec_type::REC_ATTR_INT32:
            {
                rec_size = sizeof(int32_t);
                break;
            }
            case rec_type::REC_ATTR_INT64:
            {
                rec_size = sizeof(int64_t);
                break;
            }
            case rec_type::REC_ATTR_DATE:
            case rec_type::REC_ATTR_DECIMAL:
            {
                rec_size = sizeof(int32_t);
                break;
            }
            case rec_type::REC_ATTR_CHAR:
            {
                if (field_sizes[i] < 3) {
                    rec_size = sizeof(int32_t);
                } else {
                    size_t rec_size_base = field_sizes[i];
                    rec_size = (rec_size_base % alignment == 0) ? rec_size_base : (rec_size_base + (alignment - (rec_size_base % alignment)));
                    rec_size += sizeof(int64_t);
                }
                break;
            }
            case rec_type::REC_ATTR_VCHAR:
            {
                /* VCHAR does not use the field size directly */
                rec_size = 0;
                break;
            }
            default:
                std::cerr << "Unsupported field type: " << static_cast<int>(attrtype) << "\n";
                std::exit(EXIT_FAILURE);
        }

        /* This function is non-golap only; all fixed-size fields need
           128-aligned capacity to match the sampling-phase estimate.
           Without this, uncomp fields without stats would use natural
           capacity, creating prefix-sum mismatches across fields. */
        if ((attrtype == rec_type::REC_ATTR_INT16
            || attrtype == rec_type::REC_ATTR_INT32
            || attrtype == rec_type::REC_ATTR_INT64
            || attrtype == rec_type::REC_ATTR_DATE
            || attrtype == rec_type::REC_ATTR_CHAR
            || attrtype == rec_type::REC_ATTR_DECIMAL)) {
            /* For fixed types */
            size_t nrecs_per_sort_buffer = freespace / (rec_size);

            switch (attrtype) {
                case rec_type::REC_ATTR_INT16:
                case rec_type::REC_ATTR_INT32:
                case rec_type::REC_ATTR_INT64:
                case rec_type::REC_ATTR_DATE:
                case rec_type::REC_ATTR_DECIMAL:
                    nrecs_per_sort_buffer = freespace / rec_size;
                    /* number of records should be aligned to the mutiples number of 128 */
                    if (nrecs_per_sort_buffer >= 128) {
                        /* for bin packing */
                        nrecs_per_sort_buffer = (nrecs_per_sort_buffer / 128) * 128;
                    } else {
                        nrecs_per_sort_buffer = 128;
                    }
                    break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[i] < 3) {
                        nrecs_per_sort_buffer = freespace / rec_size;
                        if (nrecs_per_sort_buffer >= 128) {
                            nrecs_per_sort_buffer = (nrecs_per_sort_buffer / 128) * 128;
                        } else {
                            nrecs_per_sort_buffer = 128;
                        }
                    } else {
                        if (compression[i] == CompressionMethod::FSST || compression[i] == CompressionMethod::FSST_ROWID) {
                            /* FSST CHAR(>=3): match estimation formula exactly.
                             * Use aligned field size (no rowid overhead) so that
                             * nrecs_per_page matches the sampling estimate.
                             * FSST compression only reduces storage size per page. */
                            size_t rec_size_est = (field_sizes[i] % alignment == 0)
                                ? field_sizes[i] : field_sizes[i] + (alignment - (field_sizes[i] % alignment));
                            nrecs_per_sort_buffer = freespace / rec_size_est;
                        } else {
                            nrecs_per_sort_buffer = freespace / rec_size;
                        }
                    }
                    break;
                case rec_type::REC_ATTR_VCHAR:
                    nrecs_per_sort_buffer = 0;
                    /* VCHAR does not use the field size directly */
                    break;
                default:
                    std::cerr << "Unsupported field type: " << static_cast<int>(attrtype) << "\n";
                    std::exit(EXIT_FAILURE);
            }
            nrecs_per_page_capacity[i] = nrecs_per_sort_buffer;
        } else {
            /* For VCHAR types */
            nrecs_per_page_capacity[i] = 0;
        }

        if (verbose) std::cout << "Col " << i << ": nrecs_per_page_capacity = "
                  << nrecs_per_page_capacity[i] << "\n";
    }



    std::vector<std::optional<BufferVariant>> staging_line_buffers(NFields);
    for (size_t i = 0; i < NFields; ++i) {
        // should_buffer ->
        // when compression is enabled or stats collection is enabled
#if 0
        if ((compression[i] != CompressionMethod::NONE) || enable_stats[i]) {
            switch (field_types[i]) {
                case rec_type::REC_ATTR_INT32: staging_line_buffers[i] = std::vector<SortPair<int32_t>>{}; break;
                case rec_type::REC_ATTR_INT64: staging_line_buffers[i] = std::vector<SortPair<int64_t>>{}; break;
                case rec_type::REC_ATTR_DECIMAL: staging_line_buffers[i] = std::vector<SortPair<int32_t>>{}; break; // DECIMAL -> int32
                case rec_type::REC_ATTR_DATE: staging_line_buffers[i] = std::vector<SortPair<DateType>>{}; break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[i] < 3)
                        staging_line_buffers[i] = std::vector<SortPair<CharAsInt>>{};
                    else
                        staging_line_buffers[i] = std::vector<SortPair<CharType>>{};
                    break;
                case rec_type::REC_ATTR_VCHAR: staging_line_buffers[i] = std::vector<SortPair<VCharType>>{}; break;
                default: break;
            }
        }
#else
        auto attrtype = field_types[i];
        /* Non-golap: all fixed-size fields need staging buffers to
           enforce 128-aligned page capacity (matching nrecs_per_page_capacity). */
        bool need_staging =
            (attrtype == REC_ATTR_INT16 ||
              attrtype == REC_ATTR_INT32 ||
              attrtype == REC_ATTR_INT64 ||
              attrtype == REC_ATTR_DECIMAL ||
              attrtype == REC_ATTR_DATE ||
              (attrtype == REC_ATTR_CHAR && field_sizes[i] < 3))
            || ((compression[i] == CompressionMethod::FSST || compression[i] == CompressionMethod::FSST_ROWID)
                && (attrtype == REC_ATTR_CHAR || attrtype == REC_ATTR_VCHAR));
        if (need_staging) {
            switch (attrtype) {
                case rec_type::REC_ATTR_INT32: staging_line_buffers[i] = std::vector<SortPair<int32_t>>{}; break;
                case rec_type::REC_ATTR_INT64: staging_line_buffers[i] = std::vector<SortPair<int64_t>>{}; break;
                case rec_type::REC_ATTR_DECIMAL: staging_line_buffers[i] = std::vector<SortPair<DecimalType>>{}; break; // DECIMAL -> int32
                case rec_type::REC_ATTR_DATE: staging_line_buffers[i] = std::vector<SortPair<DateType>>{}; break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[i] < 3)
                        staging_line_buffers[i] = std::vector<SortPair<CharAsInt>>{};
                    else
                        staging_line_buffers[i] = std::vector<SortPair<CharType>>{};
                    break;
                case rec_type::REC_ATTR_VCHAR: staging_line_buffers[i] = std::vector<SortPair<VCharType>>{}; break;
                default: break;
            }
        }
#endif
    }

    ssize_t nrecs_loaded = 0;
    std::string &path = paths[task.file_id];
    // std::cout << "Processing file: " << path << std::endl;
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open file.\n";
        return { -1, nios_empty, nwritten_sectors, max_nrecs_in_page_count, nrecs_inserted_total };
    }

    std::string line;
    nrecs_inserted_total.fill(0);

    while (std::getline(file, line)) {
        const size_t row_id = base_row_id + nrecs_loaded;
        std::string_view sv(line);
        size_t fi = 0;
        size_t scf = 0; /* sideways char field index */
        size_t f = 0;
        size_t vf = 0;
        char *pagbuf;
        enum rec_type attr_type;

        auto fields = tpch_split_row(line);
        pagbuf = reinterpret_cast<char*>(buffers[0]);

        if constexpr (NumSidewaysFields > 0) {
            if (fields.size() != NFields + NumSidewaysFields && fields.size() != NFields) {
                std::cerr << "Error: row has " << fields.size()
                          << " fields, expected " << NFields << " or "
                          << NFields + NumSidewaysFields << "\n";
                exit(EXIT_FAILURE);
            }
        } else {
            if (fields.size() < NFields) {
                std::cerr << "Error: row has " << fields.size()
                          << " fields, expected " << NFields << "\n";
                exit(EXIT_FAILURE);
            }
        }

        for (const auto& part : fields) {
            if (fi < NFields) {
                attr_type = field_types[fi];
            }

            // Dispatch
            if (fi < NFields) {
                f = fi;
                switch (field_types[f]) {
                    case rec_type::REC_ATTR_INT32: {
                        auto& stats = std::get<ColumnStats<int32_t>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<int32_t>>>(all_stats_history[f]);
                        std::vector<SortPair<int32_t>>* buf_ptr = nullptr;
                        if (staging_line_buffers[f].has_value()) {
                            buf_ptr = &std::get<std::vector<SortPair<int32_t>>>(staging_line_buffers[f].value());
                        }
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        process_column_logic<int32_t>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist,
                            enable_sideways_stats, sw_stats, sw_stats_hist,
                            nrecs_per_page_capacity[f],
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_INT64: {
                        auto& stats = std::get<ColumnStats<int64_t>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<int64_t>>>(all_stats_history[f]);
                        std::vector<SortPair<int64_t>>* buf_ptr = nullptr;
                        if (staging_line_buffers[f].has_value()) {
                            buf_ptr = &std::get<std::vector<SortPair<int64_t>>>(staging_line_buffers[f].value());
                        }
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        process_column_logic<int64_t>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist,
                            enable_sideways_stats, sw_stats, sw_stats_hist,
                            nrecs_per_page_capacity[f],
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_DECIMAL: {
                        // DECIMAL -> INT32 Logic
                        auto& stats = std::get<ColumnStats<DecimalType>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<DecimalType>>>(all_stats_history[f]);
                        std::vector<SortPair<DecimalType>>* buf_ptr = nullptr;
                        if (staging_line_buffers[f].has_value()) {
                            buf_ptr = &std::get<std::vector<SortPair<DecimalType>>>(staging_line_buffers[f].value());
                        }
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        process_column_logic<DecimalType>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist,
                            enable_sideways_stats, sw_stats, sw_stats_hist,
                            nrecs_per_page_capacity[f],
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_DATE: {
                        auto& stats = std::get<ColumnStats<DateType>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<DateType>>>(all_stats_history[f]);
                        std::vector<SortPair<DateType>>* buf_ptr = nullptr;
                        if (staging_line_buffers[f].has_value()) {
                            buf_ptr = &std::get<std::vector<SortPair<DateType>>>(staging_line_buffers[f].value());
                        }
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        process_column_logic<DateType>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist,
                            enable_sideways_stats, sw_stats, sw_stats_hist,
                            nrecs_per_page_capacity[f],
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_CHAR: {
                        if (field_sizes[f] < 3) {
                            auto& stats = std::get<ColumnStats<CharAsInt>>(all_stats[f]);
                            auto& hist = std::get<std::vector<ColumnStats<CharAsInt>>>(all_stats_history[f]);
                            std::vector<SortPair<CharAsInt>>* buf_ptr = nullptr;
                            if (staging_line_buffers[f].has_value()) {
                                buf_ptr = &std::get<std::vector<SortPair<CharAsInt>>>(staging_line_buffers[f].value());
                            }
                            auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                            auto& sw_stats_hist = all_sideways_stats_history[f];
                            process_column_logic<CharAsInt>(
                                fields[f], row_id,
                                f, field_types[f], sizeof(int32_t), dirtyflags[f],
                                buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                                output_fds,
                                buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                                compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                                enable_stats[f], stats, hist,
                                enable_sideways_stats, sw_stats, sw_stats_hist,
                                nrecs_per_page_capacity[f],
                                nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                                task_compressed_sizes[f],
                                nios_data[f], nwritten_sectors[f]
                            );
                        } else {
                            auto& stats = std::get<ColumnStats<CharType>>(all_stats[f]);
                            auto& hist = std::get<std::vector<ColumnStats<CharType>>>(all_stats_history[f]);
                            std::vector<SortPair<CharType>>* buf_ptr = nullptr;
                            if (staging_line_buffers[f].has_value()) {
                                buf_ptr = &std::get<std::vector<SortPair<CharType>>>(staging_line_buffers[f].value());
                            }
                            auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                            auto& sw_stats_hist = all_sideways_stats_history[f];
                            process_column_logic<CharType>(
                                fields[f], row_id,
                                f, field_types[f], field_sizes[f], dirtyflags[f],
                                buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                                output_fds,
                                buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                                compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                                enable_stats[f], stats, hist,
                                enable_sideways_stats, sw_stats, sw_stats_hist,
                                nrecs_per_page_capacity[f],
                                nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                                task_compressed_sizes[f],
                                nios_data[f], nwritten_sectors[f]
                            );
                        }
                        break;
                    }
                    case rec_type::REC_ATTR_VCHAR: {
                        auto& stats = std::get<ColumnStats<VCharType>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<VCharType>>>(all_stats_history[f]);
                        std::vector<SortPair<VCharType>>* buf_ptr = nullptr;
                        if (staging_line_buffers[f].has_value()) {
                            buf_ptr = &std::get<std::vector<SortPair<VCharType>>>(staging_line_buffers[f].value());
                        }
                        if (enable_lbc) {
                            /* FIXME: using stats for varchar */
                            if (vf < NumVarCharFields) {
                                //size_t nclusters = varchar_cluster_thresholds[vf].size();
                                //for (size_t k = 0; k < nclusters; ++k) {
                                //    std::cout << "\t[VCHAR][LbC] start_page_id: "
                                //        << task.start_page_ids_varchar[vf][k]
                                //        << ", npages: "
                                //        << task.npages_varchar[vf][k] << std::endl;
                                //}

                                size_t len_orig = fields[f].size();
                                auto it = std::upper_bound(varchar_cluster_thresholds[vf].begin(), varchar_cluster_thresholds[vf].end(),
                                    len_orig);
                                size_t cid = std::distance(varchar_cluster_thresholds[vf].begin(), it);

                                /*std::vector<ColumnStats<int32_t>*/
                                auto& sw_stats_varchar = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats_varchar[vf][cid]);
                                /* NOTE pagids[f], npages_used[f], and npages_limit[f] should be used for lbc */
                                process_column_logic<VCharType>(
                                    fields[f], row_id,
                                    f, field_types[f], field_sizes[f], dirtyflags_varchar[vf][cid],
                                    lbc_buffers[vf][cid], page_size, lbc_pagids[vf][cid], lbc_npages_used[vf][cid], lbc_npages_limit[vf][cid],
                                    output_fds,
                                    buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                                    compression[f], comp_ctx, compression_base_page_ids_varchar[vf][cid],
                                    compressed_page_write_offsets_varchar[vf][cid],
                                    enable_stats[f], stats, hist,
                                    enable_sideways_stats, sw_stats_varchar, task_sideways_stats_per_page_varchar[vf][cid],
                                    nrecs_per_page_capacity[f],
                                    nrecs_in_page_count_varchar[vf][cid], max_nrecs_in_page_count_varchar[vf], task_nrecs_per_page_varchar[vf][cid],
                                    task_compressed_sizes_varchar[vf][cid],
                                    nios_data[f], nwritten_sectors[f]
                                );
 
                            }
                            vf++;
                            // exit(1);
                        } else {
                            auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                            auto& sw_stats_hist = all_sideways_stats_history[f];
                            process_column_logic<VCharType>(
                                fields[f], row_id,
                                f, field_types[f], field_sizes[f], dirtyflags[f],
                                buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                                output_fds,
                                buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                                compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                                enable_stats[f], stats, hist,
                                enable_sideways_stats, sw_stats, sw_stats_hist,
                                nrecs_per_page_capacity[f],
                                nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                                task_compressed_sizes[f],
                                nios_data[f], nwritten_sectors[f]
                            );
                        }
                        break;
                    }
                    default: break;
                }
                ++nrecs_inserted_total[f];
            } else if (fi < NFields + NumSidewaysFields) {
                // Sideways column, skip for now
                size_t f_sw = fi - NFields;
                switch (sideways_field_types[f_sw]) {
                case rec_type::REC_ATTR_INT32: {
                    int32_t value = parse_value<int32_t>(part);
                    for (size_t j = 0; j < NFields; j++) {
                        if (field_types[j] == rec_type::REC_ATTR_VCHAR && enable_lbc) {
                            for (size_t vf = 0; vf < NumVarCharFields; vf++) {
                                for (size_t cid = 0; cid < varchar_cluster_thresholds[vf].size(); cid++) {
                                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats_varchar[vf][cid]);
                                    update_stats(sw_stats[f_sw], value);
                                }
                            }
                        } else {
                            auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[j]);
                            update_stats(sw_stats[f_sw], value);
                        }
                    }
                    break;
                }
                case rec_type::REC_ATTR_DATE: {
                    /* avoid using the abstraction as workaround */
                    int32_t value = date_to_int32(part);
                    for (size_t j = 0; j < NFields; j++) {
                        if (field_types[j] == rec_type::REC_ATTR_VCHAR && enable_lbc) {
                            for (size_t vf = 0; vf < NumVarCharFields; vf++) {
                                for (size_t cid = 0; cid < varchar_cluster_thresholds[vf].size(); cid++) {
                                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats_varchar[vf][cid]);
                                    update_stats(sw_stats[f_sw], value);
                                }
                            }
                        } else {
                            auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[j]);
                            update_stats(sw_stats[f_sw], value);
                        }
                    }
                    break;
                }
                case rec_type::REC_ATTR_CHAR: {
                    //autl& value = parse_value<CharType>(part).value();
                    for (size_t j = 0; j < NFields; j++) {
                        if (field_types[j] == rec_type::REC_ATTR_VCHAR && enable_lbc) {
                            /* confirming that the mapped id is preserved */
                            for (size_t vf = 0; vf < NumVarCharFields; vf++) {
                                for (size_t cid = 0; cid < varchar_cluster_thresholds[vf].size(); cid++) {
                                    if (dict_encoding_maps[scf].contains(part)) {
                                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats_varchar[vf][cid]);
                                        update_stats<int32_t>(sw_stats[f_sw], static_cast<int32_t>(dict_encoding_maps[scf][part]) );
                                    } else {
                                        std::cerr << "[ERROR] Dictionary encoding map is empty for sideways char field index: " << scf << "\n";
                                        exit(EXIT_FAILURE);
                                    }
                                }
                            }
                        } else {
                            auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[j]);
                            /* confirming that the mapped id is preserved */
                            if (dict_encoding_maps[scf].contains(part)) {
                                update_stats<int32_t>(sw_stats[f_sw], static_cast<int32_t>(dict_encoding_maps[scf][part]) );
                            } else {
                                std::cerr << "[ERROR] Dictionary encoding map is empty for sideways char field index: " << scf << "\n";
                                exit(EXIT_FAILURE);
                            }
                        }
                    }
                    scf++;
                    break;
                }
                default: {
                    std::cerr << "[ERROR] Unsupported sideways attribute type: "
                              << static_cast<int32_t>(sideways_field_types[fi]) << "\n";
                    exit(EXIT_FAILURE);
                }
                }
            } else {
                std::cerr << "[ERROR] Field index out of range: " << fi << "\n";
                exit(EXIT_FAILURE);
            }

            ++fi;
        }
        nrecs_loaded++;
    }

    for (size_t f = 0; f < NFields; ++f) {
        const bool final_flush = true;
        if (staging_line_buffers[f].has_value()) {
            switch (field_types[f]) {
                case rec_type::REC_ATTR_INT32: {
                    auto& buf = std::get<std::vector<SortPair<int32_t>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<int32_t>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<int32_t>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto& sw_stats_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                        npages_used[f], npages_limit[f],
                        field_sizes[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f], task_compressed_sizes[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats, sw_stats, sw_stats_hist,
                        output_fds,
                        nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                        final_flush
                    );
                    // max_nrecs_in_page_count = std::max(nrecs_in_page_count, max_nrecs_in_page_count);
                    break;
                }
                case rec_type::REC_ATTR_INT64: {
                    auto& buf = std::get<std::vector<SortPair<int64_t>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<int64_t>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<int64_t>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto& sw_stats_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                        npages_used[f], npages_limit[f], field_sizes[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f], task_compressed_sizes[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats, sw_stats, sw_stats_hist,
                        output_fds,
                        nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                        final_flush);
                    break;
                }
                case rec_type::REC_ATTR_DECIMAL: {
                    auto& buf = std::get<std::vector<SortPair<DecimalType>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<DecimalType>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<DecimalType>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto& sw_stats_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                        npages_used[f], npages_limit[f], field_sizes[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f], task_compressed_sizes[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats, sw_stats, sw_stats_hist,
                        output_fds,
                        nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                        final_flush);
                    break;
                }
                case rec_type::REC_ATTR_DATE: {
                    auto& buf = std::get<std::vector<SortPair<DateType>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<DateType>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<DateType>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto& sw_stats_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                         npages_used[f], npages_limit[f], field_sizes[f],
                         compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f], task_compressed_sizes[f],
                         enable_stats[f], stats, hist,
                         enable_sideways_stats, sw_stats, sw_stats_hist,
                         output_fds,
                         nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                         final_flush);
                    break;
                }
                case rec_type::REC_ATTR_CHAR: {
                    if (field_sizes[f] < 3) {
                        auto& buf = std::get<std::vector<SortPair<CharAsInt>>>(staging_line_buffers[f].value());
                        auto& stats = std::get<ColumnStats<CharAsInt>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<CharAsInt>>>(all_stats_history[f]);
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                             npages_used[f], npages_limit[f], sizeof(int32_t),
                             compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f], task_compressed_sizes[f],
                             enable_stats[f], stats, hist,
                             enable_sideways_stats, sw_stats, sw_stats_hist,
                             output_fds,
                             nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                             final_flush);
                    } else {
                        auto& buf = std::get<std::vector<SortPair<CharType>>>(staging_line_buffers[f].value());
                        auto& stats = std::get<ColumnStats<CharType>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<CharType>>>(all_stats_history[f]);
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto& sw_stats_hist = all_sideways_stats_history[f];
                        flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                             npages_used[f], npages_limit[f], field_sizes[f],
                             compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f], task_compressed_sizes[f],
                             enable_stats[f], stats, hist,
                             enable_sideways_stats, sw_stats, sw_stats_hist,
                             output_fds,
                             nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                             final_flush);
                    }
                    break;
                }
                case rec_type::REC_ATTR_VCHAR: {
                    if (compression[f] != CompressionMethod::FSST && compression[f] != CompressionMethod::FSST_ROWID) {
                        std::cerr << "[ERROR](" << __FILE__ << ":" << __LINE__
                                  << ") Unexpected buffering on VCHAR field at final flush"
                                  << " (compression=" << compression_method_name(compression[f]) << "). Stop.\n";
                        exit(EXIT_FAILURE);
                    }
                    auto& buf = std::get<std::vector<SortPair<VCharType>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<VCharType>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<VCharType>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto& sw_stats_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                        npages_used[f], npages_limit[f], field_sizes[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f], task_compressed_sizes[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats, sw_stats, sw_stats_hist,
                        output_fds,
                        nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                        final_flush);
                    break;
                }
                default: break;
            }
        } else if (dirtyflags[f]) {
            char *buf = reinterpret_cast<char*>(buffers[f]);
            flush_page_to_storage(output_fds,
                buf, pagids[f], page_size,
                nrecs_in_page_count[f],
                task_nrecs_per_page[f],
                nios_data[f],
                nwritten_sectors[f],
                compression[f], task_compressed_sizes[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                f, __LINE__
            );
            max_nrecs_in_page_count[f] = std::max(nrecs_in_page_count[f], max_nrecs_in_page_count[f]);
            pagids[f]++;
            npages_used[f]++;
            if (verbose) std::cout << "[DEBUG] Final flush: field " << f << ", page id " << pagids[f]-1
                      << ", nrecs in page: " << nrecs_in_page_count[f] << std::endl;

            if (enable_stats[f]) {
                switch (field_types[f]) {
                case rec_type::REC_ATTR_INT32: {
                    auto& stats = std::get<ColumnStats<int32_t>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<int32_t>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                case rec_type::REC_ATTR_INT64: {
                    auto& stats = std::get<ColumnStats<int64_t>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<int64_t>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                case rec_type::REC_ATTR_DECIMAL: {
                    auto& stats = std::get<ColumnStats<DecimalType>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<DecimalType>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                case rec_type::REC_ATTR_DATE: {
                    auto& stats = std::get<ColumnStats<DateType>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<DateType>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                case rec_type::REC_ATTR_CHAR: {
                    if (field_sizes[f] < 3) {
                        auto& stats = std::get<ColumnStats<CharAsInt>>(all_stats[f]);
                        auto& stats_history = std::get<std::vector<ColumnStats<CharAsInt>>>(all_stats_history[f]);
                        save_and_reset_stats(stats_history, stats);
                    } else {
                        auto& stats = std::get<ColumnStats<CharType>>(all_stats[f]);
                        auto& stats_history = std::get<std::vector<ColumnStats<CharType>>>(all_stats_history[f]);
                        save_and_reset_stats(stats_history, stats);
                    }
                    break;
                }
                case rec_type::REC_ATTR_VCHAR: {
                    auto& stats = std::get<ColumnStats<VCharType>>(all_stats[f]);
                    auto& stats_history = std::get<std::vector<ColumnStats<VCharType>>>(all_stats_history[f]);
                    save_and_reset_stats(stats_history, stats);
                    break;
                }
                default:
                    std::cerr << "Error: Unknown rec_type encountered during stats saving." << std::endl;
                    std::exit(EXIT_FAILURE);
                    break;
                }
                /* per-column stats */
            }
            /* sideways stats */
            if (enable_lbc && field_types[f] == REC_ATTR_VCHAR) {
                /* flush sideways stats for lbc is done in the next after this loop */
            } else {
                auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                auto& sw_stats_hist = all_sideways_stats_history[f];
                for (size_t i = 0; i < NumSidewaysFields; ++i) {
                    save_and_reset_stats(sw_stats_hist[i], sw_stats[i]);
                }
            }

            dirtyflags[f] = false; // Reset offset after writing
        }
    }
 
    if (enable_lbc) {
        for (size_t i = 0; i < NumVarCharFields; ++i) {
            const size_t nclusters = varchar_cluster_thresholds[i].size();
            const size_t f = task.varchar_field_indexes[i];
            for (size_t k = 0; k < nclusters; ++k) {
                if (dirtyflags_varchar[i][k])
                {
                    char *buf = reinterpret_cast<char*>(lbc_buffers[i][k]);
                    flush_page_to_storage(output_fds,
                        buf, lbc_pagids[i][k], page_size,
                        // FIXME: next
                        nrecs_in_page_count_varchar[i][k],
                        task_nrecs_per_page_varchar[i][k],
                        nios_data[f],
                        nwritten_sectors[f],
                        compression[f],
                        task_compressed_sizes_varchar[i][k],
                        comp_ctx,
                        compression_base_page_ids_varchar[i][k],
                        compressed_page_write_offsets_varchar[i][k],
                        f, __LINE__
                    );
                    max_nrecs_in_page_count[f] = std::max(nrecs_in_page_count_varchar[i][k], max_nrecs_in_page_count[f]);
                    dirtyflags_varchar[i][k] = false; // Reset offset after writing

                    if (lbc_npages_used[i][k] + 1 != task_nrecs_per_page_varchar[i][k].size()) {
                        std::cerr << "Error(" << __LINE__ << "): Field ID " << f
                                  << " npages_used " << lbc_npages_used[i][k] + 1
                                  << " != task_nrecs_per_page.size() " << task_nrecs_per_page_varchar[i][k].size() << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    if (compression[f] != CompressionMethod::NONE) {
                        if (lbc_npages_used[i][k] + 1 != task_compressed_sizes_varchar[i][k].size())
                        {
                            std::cerr << "Error(" << __LINE__ << "): Field ID " << f
                                      << " npages_used " << lbc_npages_used[i][k] + 1
                                      << " != compressed_sizes.size() " << task_compressed_sizes_varchar[i][k].size() << std::endl;
                            std::exit(EXIT_FAILURE);
                        }
                    }

                    if constexpr (NumSidewaysFields > 0) {
                        auto& sw_stats_varchar = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats_varchar[i][k]);
                        auto& sw_stats_hist_varchar = task_sideways_stats_per_page_varchar[i][k];
                        for (size_t i = 0; i < NumSidewaysFields; ++i) {
                            save_and_reset_stats(sw_stats_hist_varchar[i], sw_stats_varchar[i]);
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < NumVarCharFields; ++i) {
            free(lbc_buffer_starts[i]);
        }
    }
    // exit(0);

    if (compress) {
        free_compression_context(&comp_ctx);
    }

    if (verbose) std::cout << "Finished loading file: " << path << ", total records loaded: " << nrecs_loaded << std::endl;
    for (size_t f = 0; f < num_buffers_used; ++f) {
        // std::cout << "(bufferid="<< f << ") used " << npages_used[f] << ", allocated " << npages[f] << std::endl;
        auto attrtype = field_types[f];
        switch (attrtype)
        {
        case rec_type::REC_ATTR_INT16:
        case rec_type::REC_ATTR_INT32:
        case rec_type::REC_ATTR_INT64:
        case rec_type::REC_ATTR_DATE:
        case rec_type::REC_ATTR_DECIMAL:
        case rec_type::REC_ATTR_CHAR:
            if (npages_used[f] > npages_limit[f]) {
                std::cerr << "[ERROR] Exceeded the allocated number of pages for field " << f
                          << ": used " << npages_used[f]
                          << ", allocated " << npages_limit[f] << std::endl;
                exit(1);
            }
            break;
        case rec_type::REC_ATTR_VCHAR:
            /* code */
            if (npages_used[f] > npages_limit[f]) {
                std::cerr << "[ERROR] Exceeded the allocated number of pages for field " << f
                          << ": used " << npages_used[f]
                          << ", allocated " << npages_limit[f] << std::endl;
                exit(1);
            }
 
            break;
        
        default:
            break;
        }
    }


    return { nrecs_loaded, nios_data, nwritten_sectors, max_nrecs_in_page_count, nrecs_inserted_total };
}
 
 

template <typename EnumType, size_t MaxNumBuffers, size_t NumStartXtns, size_t NFields,
    size_t NumVarCharFields, size_t NumFilters, size_t NumSidewaysFilters = 0>
static TPCHLoaderThreadStats<MaxNumBuffers, NFields> load_lines_to_table_as_column(
    ColumnLoadTask<NFields, NumStartXtns, NumVarCharFields, NumFilters, NumSidewaysFilters>& task,
    std::vector<std::string>& paths,
    std::vector<int>& output_fds,
    const std::array<char*, MaxNumBuffers>& buffers,
    const size_t page_size, const int dryrun, const int verbose)
{
    /* temporal implementation */
    constexpr size_t NumRecBufs = NFields;
    size_t num_buffers_used = task.num_buffers_used;
    size_t num_varchar_clusters = task.lbc_num_varchar_clusters;
    const size_t base_row_id = task.base_row_id;
    const bool compress = task.compress;
    const bool enable_lz4par = task.enable_lz4par;
    const bool enable_fsst = task.enable_fsst;
    const bool enable_lbc = task.enable_lbc;
    bool enable_golap_compression_mode = task.enable_golap_compression_mode;
    const bool enable_sideways_stats = task.enable_sideways_stats;

    TPCHTableMetadata &metadata = *task.metadata;

    std::array<uint64_t, NumStartXtns> start_page_ids{};
    std::array<uint64_t, NumStartXtns> npages_limit{};
    std::array<std::vector<uint64_t>, NumVarCharFields> start_page_ids_varchar{};
    std::array<std::vector<uint64_t>, NumVarCharFields> lbc_npages_limit{};
    std::array<uint64_t, NFields> field_sizes{};
    std::array<enum rec_type, NFields> field_types{};
    std::array<size_t, NFields> field_encoded_sizes{};
    std::array<enum rec_type, NFields> field_encoded_types{};
    std::array<size_t, NFields> arr_column_sizes{};
    std::array<uint64_t, NumVarCharFields> varchar_field_indexes{};
    std::array<std::vector<size_t>, NumVarCharFields> varchar_cluster_thresholds{};
    /* Disable to collect stats on all columns */
    std::array<bool, NFields> enable_stats{};
    // std::array<std::vector<size_t>, NumVarCharFields> arr_vec_npages_for_varchar_clusters{};

    if (enable_sideways_stats) {
        if (enable_golap_compression_mode) {
            if constexpr (std::is_same_v<EnumType, TPCH::common::LineitemField>) {
                return load_lines_to_table_as_column_with_sideways_stats_golap<EnumType, MaxNumBuffers, NumStartXtns, NFields,
                    NumVarCharFields, NumFilters, 3>(
                        task, paths, output_fds, buffers, page_size, 
                        TPCH::common::fmt.lineitem_sideways_information_types,
                        TPCH::common::fmt.lineitem_sideways_information_sizes,
                        dryrun, verbose);
            } else if constexpr (std::is_same_v<EnumType, TPCH::common::OrdersField>) {
                return load_lines_to_table_as_column_with_sideways_stats_golap<EnumType, MaxNumBuffers, NumStartXtns, NFields,
                    NumVarCharFields, NumFilters, 2>(
                        task, paths, output_fds, buffers, page_size, 
                        TPCH::common::fmt.orders_sideways_information_types,
                        TPCH::common::fmt.orders_sideways_information_sizes,
                        dryrun, verbose);
            }
        } else {
            if constexpr (std::is_same_v<EnumType, TPCH::common::LineitemField>) {
                return load_lines_to_table_as_column_with_sideways_stats<EnumType, MaxNumBuffers, NumStartXtns, NFields,
                    NumVarCharFields, NumFilters, 3>(
                        task, paths, output_fds, buffers, page_size, 
                        TPCH::common::fmt.lineitem_sideways_information_types,
                        TPCH::common::fmt.lineitem_sideways_information_sizes,
                        dryrun, verbose);
            } else if constexpr (std::is_same_v<EnumType, TPCH::common::OrdersField>) {
                return load_lines_to_table_as_column_with_sideways_stats<EnumType, MaxNumBuffers, NumStartXtns, NFields,
                    NumVarCharFields, NumFilters, 2>(
                        task, paths, output_fds, buffers, page_size, 
                        TPCH::common::fmt.orders_sideways_information_types,
                        TPCH::common::fmt.orders_sideways_information_sizes,
                        dryrun, verbose);
            }
        }
        /* Other fields will be loaded as normal */
    }
    /*
     else {
        enable_golap_compression_mode = false;
    }*/

    std::copy(task.start_page_ids.begin(), task.start_page_ids.end(), start_page_ids.begin());
    std::copy(task.npages.begin(), task.npages.end(), npages_limit.begin());
    if (enable_lbc) {
        lbc_npages_limit = task.npages_varchar;
        start_page_ids_varchar = task.start_page_ids_varchar;

        //std::copy(task.start_page_ids_varchar.begin(), task.start_page_ids_varchar.end(), start_page_ids_varchar.begin());
        //std::copy(task.npages_varchar.begin(), task.npages_varchar.end(), .begin());
        //std::copy(task.start_page_ids_varchar.begin(), task.start_page_ids_varchar.end(), start_page_ids_varchar.begin());
    }


    std::copy(task.field_sizes.begin(), task.field_sizes.end(), field_sizes.begin());
    std::copy(task.field_types.begin(), task.field_types.end(), field_types.begin());
    //std::copy(task.field_encoded_sizes.begin(), task.field_encoded_sizes.end(), field_encoded_sizes.begin());
    //std::copy(task.field_encoded_types.begin(), task.field_encoded_types.end(), field_encoded_types.begin());
    //std::copy(task.column_sizes.begin(), task.column_sizes.end(), arr_column_sizes.begin());
    std::copy(task.varchar_field_indexes.begin(), task.varchar_field_indexes.end(), varchar_field_indexes.begin());
    std::copy(task.lbc_varchar_cluster_thresholds.begin(), task.lbc_varchar_cluster_thresholds.end(), varchar_cluster_thresholds.begin());
    std::copy(task.enable_stats.begin(), task.enable_stats.end(), enable_stats.begin());
    auto &arr_vec_npages_for_varchar_clusters = task.arr_vec_npages_for_varchar_clusters;
    // std::copy(task.arr_vec_npages_for_varchar_clusters.begin(), task.arr_vec_npages_for_varchar_clusters.end(), arr_vec_npages_for_varchar_clusters.begin());

    // bool dict_encoding = task.dict_encoding;
    bool dict_encoding = false;

    CompressionContext comp_ctx {};
    if (compress) {
        /* NOTE: enable_golap_compression mode can be true and false */
        init_compression_context(&comp_ctx, page_size, enable_golap_compression_mode);
    }

#if 0
    constexpr bool enable_sort_on_date = true;
    size_t sort_buffer_size = 32 * MEBI;
    size_t npages_sort_buffer = sort_buffer_size / page_size;
    size_t freespace_date =  page_size - sizeof(struct pag_head);
    size_t nrecs_per_sort_buffer = freespace_date / (sizeof(uint32_t) + sizeof(uint64_t));
    size_t nrecs_for_sort = nrecs_per_sort_buffer * npages_sort_buffer;
    using DateRecord = std::pair<int32_t, uint64_t>; // <date_value, rowid>
    //std::array<std::vector<DateRecord>, NumFilters> arr_vec_sort_buffer {};
    std::array<std::vector<DateRecord>, NumFilters> arr_vec_sort_buffer {};

    std::cout << "sort_buffer_size = " << sort_buffer_size << " bytes, npages_sort_buffer = "
              << npages_sort_buffer << " pages, nrecs_per_sort_buffer = "
              << nrecs_per_sort_buffer << ", nrecs_for_sort = "
              << nrecs_for_sort << std::endl;
#else
    std::array<SidewayStatsVariant, NFields> sideways_stats;
    auto& all_sideways_stats_history = task.all_sideways_stats_per_page;
    {
        for (size_t i = 0; i < NFields; i++) {
            sideways_stats[i] = std::vector<ColumnStats<int32_t>>{};
        }
    }

    std::vector<StatsVariant> all_stats;
    /* NOTE: sideways_stats are just a stub because that is disabled in this code pass */
    //std::vector<StatsHistoryVariant> all_stats_history;
    auto& all_stats_history = task.all_stats_history_per_page;
    auto& task_nrecs_per_page = task.nrecs_per_page;

    auto& task_compressed_sizes = task.compressed_sizes_per_page;
    auto& compressed_page_write_offsets = task.compressed_page_write_offsets;
    // auto& task_nrecs_per_page_varchar = task.nrecs_per_page_varchar;
    // auto& task_nrecs_per_page_varchar = task.nrecs_per_page_varchar;
    
    all_stats.reserve(NFields);
    {
        for (size_t f = 0; f < NFields; f++) {
            task_nrecs_per_page[f] = std::vector<uint64_t>{};
            task_compressed_sizes[f] = std::vector<uint32_t>{};
            // auto t : field_types
            auto t = field_types[f];
            switch (t) {
                case rec_type::REC_ATTR_INT32:
                    all_stats.emplace_back(ColumnStats<int32_t>{});
                    all_stats_history[f] = std::vector<ColumnStats<int32_t>>{};
                    break;
                case rec_type::REC_ATTR_INT64:
                    all_stats.emplace_back(ColumnStats<int64_t>{});
                    all_stats_history[f] = std::vector<ColumnStats<int64_t>>{};
                    break;
                case rec_type::REC_ATTR_DECIMAL: // DECIMAL -> int32
                    all_stats.emplace_back(ColumnStats<DecimalType>{});
                    all_stats_history[f] = std::vector<ColumnStats<DecimalType>>{};
                    break;
                case rec_type::REC_ATTR_DATE:  
                    all_stats.emplace_back(ColumnStats<DateType>{});
                    all_stats_history[f] = std::vector<ColumnStats<DateType>>{};
                    break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[f] < 3) {
                        all_stats.emplace_back(ColumnStats<CharAsInt>{});
                        all_stats_history[f] = std::vector<ColumnStats<CharAsInt>>{};
                    } else {
                        all_stats.emplace_back(ColumnStats<CharType>{});
                        all_stats_history[f] = std::vector<ColumnStats<CharType>>{};
                    }
                    break;
                case rec_type::REC_ATTR_VCHAR: 
                    all_stats.emplace_back(ColumnStats<VCharType>{});
                    all_stats_history[f] = std::vector<ColumnStats<VCharType>>{};
                    break;
                default: 
                    std::cerr << "Error: Unknown rec_type encountered during stats initialization." << std::endl;
                    std::exit(EXIT_FAILURE);
                    break;
            }
        }
    }

    /* Use default compression method (NONE) */
    std::array<CompressionMethod, NFields> compression {};
    for (size_t i = 0; i < NFields; ++i) {
        if (compress) {
            if (enable_golap_compression_mode) {
                compression[i] = golap_compression_methods[i];
            } else {
                compression[i] = task.compression_methods[i];
                if (compression[i] == CompressionMethod::LZ4 && enable_lz4par) {
                    compression[i] = CompressionMethod::LZ4PAR;
                }
                if (compression[i] == CompressionMethod::LZ4 && enable_fsst
                    && ((field_types[i] == rec_type::REC_ATTR_CHAR && field_sizes[i] >= 3)
                        || field_types[i] == rec_type::REC_ATTR_VCHAR)) {
                    compression[i] = CompressionMethod::FSST_ROWID;
                }
            }
            if (verbose) std::cout << "Field " << i << " compression method: "
                      << compression_method_name(compression[i]) << std::endl;
        }
    }
    auto& compression_base_page_ids = start_page_ids;

    std::vector<std::optional<BufferVariant>> staging_line_buffers(NFields);
    for (size_t i = 0; i < NFields; ++i) {
        // should_buffer ->
        // when compression is enabled or stats collection is enabled
#if 0
        if ((compression[i] != CompressionMethod::NONE) || enable_stats[i]) {
            switch (field_types[i]) {
                case rec_type::REC_ATTR_INT32: staging_line_buffers[i] = std::vector<SortPair<int32_t>>{}; break;
                case rec_type::REC_ATTR_INT64: staging_line_buffers[i] = std::vector<SortPair<int64_t>>{}; break;
                case rec_type::REC_ATTR_DECIMAL: staging_line_buffers[i] = std::vector<SortPair<int32_t>>{}; break; // DECIMAL -> int32
                case rec_type::REC_ATTR_DATE: staging_line_buffers[i] = std::vector<SortPair<DateType>>{}; break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[i] < 3)
                        staging_line_buffers[i] = std::vector<SortPair<CharAsInt>>{};
                    else
                        staging_line_buffers[i] = std::vector<SortPair<CharType>>{};
                    break;
                case rec_type::REC_ATTR_VCHAR: staging_line_buffers[i] = std::vector<SortPair<VCharType>>{}; break;
                default: break;
            }
        }
#else
        auto attrtype = field_types[i];
        bool need_staging =
            ((attrtype == REC_ATTR_INT16 ||
              attrtype == REC_ATTR_INT32 ||
              attrtype == REC_ATTR_INT64 ||
              attrtype == REC_ATTR_DECIMAL ||
              attrtype == REC_ATTR_DATE)
             && (compression[i] == CompressionMethod::PFOR || compression[i] == CompressionMethod::PFOR64))
            || ((compression[i] == CompressionMethod::FSST || compression[i] == CompressionMethod::FSST_ROWID)
                && (attrtype == REC_ATTR_CHAR || attrtype == REC_ATTR_VCHAR));
        if (need_staging) {
            switch (attrtype) {
                case rec_type::REC_ATTR_INT32: staging_line_buffers[i] = std::vector<SortPair<int32_t>>{}; break;
                case rec_type::REC_ATTR_INT64: staging_line_buffers[i] = std::vector<SortPair<int64_t>>{}; break;
                case rec_type::REC_ATTR_DECIMAL: staging_line_buffers[i] = std::vector<SortPair<DecimalType>>{}; break; // DECIMAL -> int32
                case rec_type::REC_ATTR_DATE: staging_line_buffers[i] = std::vector<SortPair<DateType>>{}; break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[i] < 3)
                        staging_line_buffers[i] = std::vector<SortPair<CharAsInt>>{};
                    else
                        staging_line_buffers[i] = std::vector<SortPair<CharType>>{};
                    break;
                case rec_type::REC_ATTR_VCHAR: staging_line_buffers[i] = std::vector<SortPair<VCharType>>{}; break;
                default: break;
            }
        }
#endif
    }

    // max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
    std::array<size_t, NFields> max_buffered_page_counts{};
    std::array<size_t, NFields> buffered_page_counts{};

    for(size_t i = 0; i < NFields; ++i) {
        // Tiered Buffering: 
        // 32 Pages for Sort/Stats columns (High Pruning/Compression potential)
        // 1 Page for Streaming columns (Low latency/overhead)
        max_buffered_page_counts[i] = 1;

        if (verbose) std::cout << "Col " << i << ": Max Buffered Pages=" << max_buffered_page_counts[i]
                  << ((max_buffered_page_counts[i] > 1) ? " [Sort Mode]" : " [Streaming Mode]")
                  << "\n";
    }

    /* Defining thresholds for fields */
    std::array<uint64_t, NFields> nrecs_per_page_capacity {};
    for (size_t i = 0; i < NFields; ++i) {
        size_t freespace = page_size - sizeof(struct pag_head);
        auto attrtype = field_types[i];
        size_t rec_size;
        constexpr size_t alignment = 4;
        switch (attrtype) {
            case rec_type::REC_ATTR_INT16:
            {
                rec_size = sizeof(int32_t);
                break;
            }
            case rec_type::REC_ATTR_INT32:
            {
                rec_size = sizeof(int32_t);
                break;
            }
            case rec_type::REC_ATTR_INT64:
            {
                rec_size = sizeof(int64_t);
                break;
            }
            case rec_type::REC_ATTR_DATE:
            case rec_type::REC_ATTR_DECIMAL:
            {
                rec_size = sizeof(int32_t);
                break;
            }
            case rec_type::REC_ATTR_CHAR:
            {
                if (field_sizes[i] < 3) {
                    rec_size = sizeof(int32_t);
                } else {
                    size_t rec_size_base = field_sizes[i];
                    rec_size = (rec_size_base % alignment == 0) ? rec_size_base : (rec_size_base + (alignment - (rec_size_base % alignment)));
                }
                break;
            }
            case rec_type::REC_ATTR_VCHAR:
            {
                /* VCHAR does not use the field size directly */
                rec_size = 0;
                break;
            }
            default:
                std::cerr << "Unsupported field type: " << static_cast<int>(attrtype) << "\n";
                std::exit(EXIT_FAILURE);
        }

        if ((compress || enable_stats[i])
            && (attrtype == rec_type::REC_ATTR_INT16
                || attrtype == rec_type::REC_ATTR_INT32
                || attrtype == rec_type::REC_ATTR_INT64
                || attrtype == rec_type::REC_ATTR_DATE
                || attrtype == rec_type::REC_ATTR_CHAR
                || attrtype == rec_type::REC_ATTR_DECIMAL)) {
            /* For fixed types */
            size_t nrecs_per_sort_buffer = freespace / (rec_size);

            switch (attrtype) {
                case rec_type::REC_ATTR_INT16:
                case rec_type::REC_ATTR_INT32:
                case rec_type::REC_ATTR_INT64:
                case rec_type::REC_ATTR_DATE:
                case rec_type::REC_ATTR_DECIMAL:
                    nrecs_per_sort_buffer = freespace / rec_size;
                    /* number of records should be aligned to the mutiples number of 128 */
                    if (nrecs_per_sort_buffer >= 128) {
                        /* for bin packing */
                        nrecs_per_sort_buffer = (nrecs_per_sort_buffer / 128) * 128;
                    } else {
                        nrecs_per_sort_buffer = 128;
                    }
                    break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[i] < 3) {
                        nrecs_per_sort_buffer = freespace / rec_size;
                        if (nrecs_per_sort_buffer >= 128) {
                            nrecs_per_sort_buffer = (nrecs_per_sort_buffer / 128) * 128;
                        } else {
                            nrecs_per_sort_buffer = 128;
                        }
                    } else {
                        /* FSST and non-compressed CHAR(>=3): use raw record size */
                        nrecs_per_sort_buffer = freespace / rec_size;
                    }
                    break;
                case rec_type::REC_ATTR_VCHAR:
                    nrecs_per_sort_buffer = 0;
                    /* VCHAR does not use the field size directly */
                    break;
                default:
                    std::cerr << "Unsupported field type: " << static_cast<int>(attrtype) << "\n";
                    std::exit(EXIT_FAILURE);
            }
            nrecs_per_page_capacity[i] = nrecs_per_sort_buffer;
        } else {
            /* For VCHAR types */
            nrecs_per_page_capacity[i] = 0;
        }

        if (verbose) std::cout << "Col " << i << ": nrecs_per_page_capacity = "
                  << nrecs_per_page_capacity[i] << "\n";
    }
#endif

#if 0
    size_t siz_recbuf = 16384; // 16KB, should be enough for most records
    std::array<char*, NumRecBufs> recbufs;
    // std::array<REC*, NumBuffers> bufs;
    char *recbuf_base = reinterpret_cast<char*>(malloc(siz_recbuf * NumRecBufs));
    memset(recbuf_base, 0, siz_recbuf * NumRecBufs);
    for (size_t i = 0; i < NumRecBufs; ++i) {
        recbufs[i] = &recbuf_base[i * siz_recbuf];
    }
#endif
    REC *buf;

    /* initalize arrays and vectors */
    // std::array<bool, MaxNumBuffers> written {};
    std::array<bool, MaxNumBuffers> dirtyflags {};
    std::array<std::vector<bool>, NumVarCharFields> dirtyflags_varchar {};
    std::array<uint64_t, MaxNumBuffers> pagids {};
    std::array<std::vector<uint64_t>, MaxNumBuffers> lbc_pagids {};
    /* Tracking free space in pages for varchar fields */
    std::array<uint64_t, MaxNumBuffers> page_freespace {};

    /* tracking number of pages */
    std::array<size_t, MaxNumBuffers> nios_empty {};
    std::array<size_t, MaxNumBuffers> nios_data {};
    std::array<size_t, MaxNumBuffers> nwritten_sectors {};
    std::array<std::vector<size_t>, NumVarCharFields> arr_vec_start_pagids_for_varchar_clusters {};
    std::array<uint32_t, NumVarCharFields> last_cluster_ids {};
    std::array<size_t, MaxNumBuffers> npages_used {};
    std::array<std::vector<size_t>, NumVarCharFields> lbc_npages_used {};
    std::array<std::vector<size_t>, NumVarCharFields> arr_vec_npages_for_vc_metadata{};

    std::array<uint32_t, MaxNumBuffers> nrecs_in_page_count {};
    std::array<uint32_t, MaxNumBuffers> max_nrecs_in_page_count {};
    std::array<uint64_t, NumRecBufs> nrecs_inserted_total {};

    std::array<std::vector<uint32_t>, NumVarCharFields> nrecs_in_page_count_varchar {};
    std::array<uint32_t, NumVarCharFields> max_nrecs_in_page_count_varchar {};

    #if 0
    auto dict_types = TPCH::common::fmt.dict_types;
    #endif

    /* num_buffers_used is NumVarCharFields * options.num_varchar_clusters + 1 */
    std::vector<std::array<size_t, 1>> varchar_field_sizes {};
    std::vector<std::vector<std::unordered_map<std::string, uint32_t>>> vec_dict_maps {};
    //std::vector<std::map<uint32_t, std::string>> sort_maps {};
    for (size_t i = 0; i < num_buffers_used; ++i) {
        // written[i] = false;
        dirtyflags[i] = false;
        pagids[i] = 0;
        nios_empty[i] = 0;
        nios_data[i] = 0;
        nwritten_sectors[i] = 0;
        npages_used[i] = 0;
        nrecs_in_page_count[i] = 0;
        max_nrecs_in_page_count[i] = 0;
        pag_init(buffers[i], page_size);
    }
    for (size_t i = 0; i < num_buffers_used; ++i) { 
        pagids[i] = start_page_ids[i];
        page_freespace[i] = page_size - sizeof(struct pag_head);
        // nios[i] = 0;
        for (size_t j = 0; j < npages_limit[i]; ++j) {
            page_pwrite_host(output_fds, buffers[i], pagids[i] + j, page_size);
        }
    }

    for (size_t i = 0; i < NumVarCharFields; ++i) {
        assert(NumVarCharFields == varchar_field_indexes.size());
        varchar_field_sizes.push_back(std::array<size_t, 1> {
            field_sizes[varchar_field_indexes[i]]
        });
    }

    std::array<std::vector<char*>, NumVarCharFields> lbc_buffers {};
    std::array<char*, NumVarCharFields> lbc_buffer_starts {};
    std::array<std::vector<std::vector<uint64_t>>, NumVarCharFields> &task_nrecs_per_page_varchar = task.nrecs_per_page_varchar;
    auto& task_compressed_sizes_varchar = task.compressed_sizes_per_page_varchar;
    auto& compression_base_page_ids_varchar = start_page_ids_varchar;
    auto& compressed_page_write_offsets_varchar = task.compressed_page_write_offsets_varchar;
    // std::vector<uint64_t> task_nrecs_per_page
    if (enable_lbc) {
        for (size_t i = 0; i < NumVarCharFields; ++i) {
            const size_t nclusters = varchar_cluster_thresholds[i].size();
            lbc_buffer_starts[i] = loader_aligned_alloc<char*>(page_size * nclusters);
            lbc_buffers[i] = std::vector<char*>{};
            dirtyflags_varchar[i] = std::vector<bool>(nclusters, false);
            for (size_t k = 0; k < nclusters; ++k) {
                lbc_buffers[i].push_back(&lbc_buffer_starts[i][k * page_size]);
                pag_init(lbc_buffers[i].back(), page_size);

            }
            lbc_pagids[i] = start_page_ids_varchar[i];
            lbc_npages_used[i] = std::vector<size_t>(nclusters, 0);

            for (size_t k = 0; k < nclusters; ++k) {
                for (size_t j = 0; j < lbc_npages_limit[i][k]; ++j) {
                    page_pwrite_host(output_fds, lbc_buffers[i][k], lbc_pagids[i][k] + j, page_size);
                }
            }

            nrecs_in_page_count_varchar[i] = std::vector<uint32_t>(nclusters, 0);
            task_nrecs_per_page_varchar[i].resize(nclusters);
            for (size_t k = 0; k < nclusters; ++k) {
                task_nrecs_per_page_varchar[i][k] = std::vector<uint64_t> {};
            }

            if (compress) {
                task_compressed_sizes_varchar[i].resize(nclusters);
                for (size_t k = 0; k < nclusters; ++k) {
                    task_compressed_sizes_varchar[i][k] = std::vector<uint32_t> {};
                }
            }
            max_nrecs_in_page_count_varchar[i] = 0;
        }

        for (size_t i = 0; i < NumVarCharFields; ++i) {
            const size_t nclusters = varchar_cluster_thresholds[i].size();
            for (size_t k = 0; k < nclusters; ++k) {
                    std::cout << "\tTask start_page_id: " << start_page_ids_varchar[i][k]
                        << ", npages: " << lbc_npages_limit[i][k] << std::endl;
            }
        }
        // exit(1);
    }

    /* return values */
    /* nrecs_loaded will be negative if an error occurs. */
    ssize_t nrecs_loaded = 0;

    std::string &path = paths[task.file_id];
    // std::cout << "Processing file: " << path << std::endl;
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open file.\n";
        return { -1, nios_empty, nwritten_sectors, max_nrecs_in_page_count, nrecs_inserted_total };
    }

    std::string line;
    nrecs_inserted_total.fill(0);

    while (std::getline(file, line)) {
        const size_t row_id = base_row_id + nrecs_loaded;
        std::string_view sv(line);

        size_t f = 0;
        // size_t filteridx = 0;
        size_t vf = 0; // varchar field index
        // int16_t value16;
        // int32_t value32;
        // int64_t value64;
        // char *valuechar;
        // uint32_t slotid;
        char *pagbuf;
        enum rec_type attr_type;

        // bool debug_print = false;

        auto fields = tpch_split_row(line);
        pagbuf = reinterpret_cast<char*>(buffers[0]);

        // buffer_appended.fill(false);
        // update_buffer_ids.fill(0);

        // std::cout << "Processing line: " << line << "\n";

        /* only c_comment field is the target */
        assert(NFields == fields.size());
        for (const auto& part : fields) {
            buf = buffers[f];
            attr_type = field_types[f];

            // Dispatch
            switch (field_types[f]) {
                case rec_type::REC_ATTR_INT32: {
                    auto& stats = std::get<ColumnStats<int32_t>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<int32_t>>>(all_stats_history[f]);
                    std::vector<SortPair<int32_t>>* buf_ptr = nullptr;
                    if (staging_line_buffers[f].has_value()) {
                        buf_ptr = &std::get<std::vector<SortPair<int32_t>>>(staging_line_buffers[f].value());
                    }
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto& sw_hist = all_sideways_stats_history[f];
                    process_column_logic<int32_t>(
                        fields[f], row_id,
                        f, field_types[f], field_sizes[f], dirtyflags[f],
                        buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                        output_fds,
                        buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats,  sw_stats, sw_hist,
                        nrecs_per_page_capacity[f],
                        nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                        task_compressed_sizes[f],
                        nios_data[f], nwritten_sectors[f]
                    );
                    break;
                }
                case rec_type::REC_ATTR_INT64: {
                    auto& stats = std::get<ColumnStats<int64_t>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<int64_t>>>(all_stats_history[f]);
                    std::vector<SortPair<int64_t>>* buf_ptr = nullptr;
                    if (staging_line_buffers[f].has_value()) {
                        buf_ptr = &std::get<std::vector<SortPair<int64_t>>>(staging_line_buffers[f].value());
                    }
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto& sw_hist = all_sideways_stats_history[f];
                    process_column_logic<int64_t>(
                        fields[f], row_id,
                        f, field_types[f], field_sizes[f], dirtyflags[f],
                        buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                        output_fds,
                        buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats,  sw_stats, sw_hist,
                        nrecs_per_page_capacity[f],
                        nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                        task_compressed_sizes[f],
                        nios_data[f], nwritten_sectors[f]
                    );
                    break;
                }
                case rec_type::REC_ATTR_DECIMAL: {
                    // DECIMAL -> INT32 Logic
                    auto& stats = std::get<ColumnStats<DecimalType>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<DecimalType>>>(all_stats_history[f]);
                    std::vector<SortPair<DecimalType>>* buf_ptr = nullptr;
                    if (staging_line_buffers[f].has_value()) {
                        buf_ptr = &std::get<std::vector<SortPair<DecimalType>>>(staging_line_buffers[f].value());
                    }
                    auto &sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto& sw_hist = all_sideways_stats_history[f];
                    process_column_logic<DecimalType>(
                        fields[f], row_id,
                        f, field_types[f], field_sizes[f], dirtyflags[f],
                        buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                        output_fds,
                        buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats,  sw_stats, sw_hist,
                        nrecs_per_page_capacity[f],
                        nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                        task_compressed_sizes[f],
                        nios_data[f], nwritten_sectors[f]
                    );
                    break;
                }
                case rec_type::REC_ATTR_DATE: {
                    auto& stats = std::get<ColumnStats<DateType>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<DateType>>>(all_stats_history[f]);
                    std::vector<SortPair<DateType>>* buf_ptr = nullptr;
                    if (staging_line_buffers[f].has_value()) {
                        buf_ptr = &std::get<std::vector<SortPair<DateType>>>(staging_line_buffers[f].value());
                    }
                    auto &sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto& sw_hist = all_sideways_stats_history[f];
                    process_column_logic<DateType>(
                        fields[f], row_id,
                        f, field_types[f], field_sizes[f], dirtyflags[f],
                        buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                        output_fds,
                        buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats,  sw_stats, sw_hist,
                        nrecs_per_page_capacity[f],
                        nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                        task_compressed_sizes[f],
                        nios_data[f], nwritten_sectors[f]
                    );
                    break;
                }
                case rec_type::REC_ATTR_CHAR: {
                    if (field_sizes[f] < 3) {
                        auto& stats = std::get<ColumnStats<CharAsInt>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<CharAsInt>>>(all_stats_history[f]);
                        std::vector<SortPair<CharAsInt>>* buf_ptr = nullptr;
                        if (staging_line_buffers[f].has_value()) {
                            buf_ptr = &std::get<std::vector<SortPair<CharAsInt>>>(staging_line_buffers[f].value());
                        }
                        auto &sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto &sw_hist = all_sideways_stats_history[f];
                        process_column_logic<CharAsInt>(
                            fields[f], row_id,
                            f, field_types[f], sizeof(int32_t), dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist,
                            enable_sideways_stats,  sw_stats, sw_hist,
                            nrecs_per_page_capacity[f],
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                    } else {
                        auto& stats = std::get<ColumnStats<CharType>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<CharType>>>(all_stats_history[f]);
                        std::vector<SortPair<CharType>>* buf_ptr = nullptr;
                        if (staging_line_buffers[f].has_value()) {
                            buf_ptr = &std::get<std::vector<SortPair<CharType>>>(staging_line_buffers[f].value());
                        }
                        auto &sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto &sw_hist = all_sideways_stats_history[f];
                        process_column_logic<CharType>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist,
                            enable_sideways_stats,  sw_stats, sw_hist,
                            nrecs_per_page_capacity[f],
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                    }
                    break;
                }
                case rec_type::REC_ATTR_VCHAR: {
                    auto& stats = std::get<ColumnStats<VCharType>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<VCharType>>>(all_stats_history[f]);
                    std::vector<SortPair<VCharType>>* buf_ptr = nullptr;
                    if (staging_line_buffers[f].has_value()) {
                        buf_ptr = &std::get<std::vector<SortPair<VCharType>>>(staging_line_buffers[f].value());
                    }
                    auto &sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto &sw_hist = all_sideways_stats_history[f];
                    if (enable_lbc) {
                        if (vf < NumVarCharFields) {
                            //size_t nclusters = varchar_cluster_thresholds[vf].size();
                            //for (size_t k = 0; k < nclusters; ++k) {
                            //    std::cout << "\t[VCHAR][LbC] start_page_id: "
                            //        << task.start_page_ids_varchar[vf][k]
                            //        << ", npages: "
                            //        << task.npages_varchar[vf][k] << std::endl;
                            //}

                            size_t len_orig = fields[f].size();
                            auto it = std::upper_bound(varchar_cluster_thresholds[vf].begin(), varchar_cluster_thresholds[vf].end(),
                                len_orig);
                            size_t cid = std::distance(varchar_cluster_thresholds[vf].begin(), it);

                            /* NOTE pagids[f], npages_used[f], and npages_limit[f] should be used for lbc */
                            process_column_logic<VCharType>(
                                fields[f], row_id,
                                f, field_types[f], field_sizes[f], dirtyflags_varchar[vf][cid],
                                lbc_buffers[vf][cid], page_size, lbc_pagids[vf][cid], lbc_npages_used[vf][cid], lbc_npages_limit[vf][cid],
                                output_fds,
                                buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                                compression[f], comp_ctx,  compression_base_page_ids_varchar[vf][cid], compressed_page_write_offsets_varchar[vf][cid],
                                enable_stats[f], stats, hist,
                                enable_sideways_stats,  sw_stats, sw_hist,
                                nrecs_per_page_capacity[f],
                                nrecs_in_page_count_varchar[vf][cid], max_nrecs_in_page_count_varchar[vf], task_nrecs_per_page_varchar[vf][cid],
                                task_compressed_sizes_varchar[vf][cid],
                                nios_data[f], nwritten_sectors[f]
                            );
 
                        }
                        vf++;
                        // exit(1);
                    } else {
                        process_column_logic<VCharType>(
                            fields[f], row_id,
                            f, field_types[f], field_sizes[f], dirtyflags[f],
                            buffers[f], page_size, pagids[f], npages_used[f], npages_limit[f],
                            output_fds,
                            buf_ptr, max_buffered_page_counts[f], buffered_page_counts[f], page_freespace[f],
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],
                            enable_stats[f], stats, hist,
                            enable_sideways_stats,  sw_stats, sw_hist,
                            nrecs_per_page_capacity[f],
                            nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f],
                            task_compressed_sizes[f],
                            nios_data[f], nwritten_sectors[f]
                        );
                    }
                    break;
                }
                default: break;
            }
            ++nrecs_inserted_total[f];
            ++f;
        }
        ++nrecs_loaded;
    }

    // Final Flush
    for (size_t f = 0; f < NFields; ++f) {
        const bool final_flush = true;
        if (staging_line_buffers[f].has_value()) {
            switch (field_types[f]) {
                case rec_type::REC_ATTR_INT32: {
                    auto& buf = std::get<std::vector<SortPair<int32_t>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<int32_t>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<int32_t>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto &sw_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                        npages_used[f], npages_limit[f],
                        field_sizes[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f], task_compressed_sizes[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats, sw_stats, sw_hist,
                        output_fds,
                        nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                        final_flush
                    );

                    // max_nrecs_in_page_count = std::max(nrecs_in_page_count, max_nrecs_in_page_count);

                    break;
                }
                case rec_type::REC_ATTR_INT64: {
                    auto& buf = std::get<std::vector<SortPair<int64_t>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<int64_t>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<int64_t>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto &sw_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                        npages_used[f], npages_limit[f], field_sizes[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f], task_compressed_sizes[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats, sw_stats, sw_hist,
                        output_fds,
                        nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                        final_flush);
                    break;
                }
                case rec_type::REC_ATTR_DECIMAL: {
                    auto& buf = std::get<std::vector<SortPair<DecimalType>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<DecimalType>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<DecimalType>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto &sw_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                        npages_used[f], npages_limit[f], field_sizes[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],  task_compressed_sizes[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats, sw_stats, sw_hist,
                        output_fds,
                        nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                        final_flush);
                    break;
                }
                case rec_type::REC_ATTR_DATE: {
                    auto& buf = std::get<std::vector<SortPair<DateType>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<DateType>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<DateType>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto &sw_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                        npages_used[f], npages_limit[f], field_sizes[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],  task_compressed_sizes[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats, sw_stats, sw_hist,
                        output_fds,
                        nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                        final_flush);
                    break;
                }
                case rec_type::REC_ATTR_CHAR: {
                    if (field_sizes[f] < 3) {
                        auto& buf = std::get<std::vector<SortPair<CharAsInt>>>(staging_line_buffers[f].value());
                        auto& stats = std::get<ColumnStats<CharAsInt>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<CharAsInt>>>(all_stats_history[f]);
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto &sw_hist = all_sideways_stats_history[f];
                        flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                            npages_used[f], npages_limit[f], sizeof(int32_t),
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],  task_compressed_sizes[f],
                            enable_stats[f], stats, hist,
                            enable_sideways_stats, sw_stats, sw_hist,
                            output_fds,
                            nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                            final_flush);
                    } else {
                        auto& buf = std::get<std::vector<SortPair<CharType>>>(staging_line_buffers[f].value());
                        auto& stats = std::get<ColumnStats<CharType>>(all_stats[f]);
                        auto& hist = std::get<std::vector<ColumnStats<CharType>>>(all_stats_history[f]);
                        auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                        auto &sw_hist = all_sideways_stats_history[f];
                        flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                            npages_used[f], npages_limit[f], field_sizes[f],
                            compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],  task_compressed_sizes[f],
                            enable_stats[f], stats, hist,
                            enable_sideways_stats, sw_stats, sw_hist,
                            output_fds,
                            nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                            final_flush);
                    }
                    break;
                }
                case rec_type::REC_ATTR_VCHAR: {
                    auto& buf = std::get<std::vector<SortPair<VCharType>>>(staging_line_buffers[f].value());
                    auto& stats = std::get<ColumnStats<VCharType>>(all_stats[f]);
                    auto& hist = std::get<std::vector<ColumnStats<VCharType>>>(all_stats_history[f]);
                    auto& sw_stats = std::get<std::vector<ColumnStats<int32_t>>>(sideways_stats[f]);
                    auto &sw_hist = all_sideways_stats_history[f];
                    flush_staging_buffer(buf, buffers[f], pagids[f], page_size,
                        npages_used[f], npages_limit[f], field_sizes[f],
                        compression[f], comp_ctx, compression_base_page_ids[f], compressed_page_write_offsets[f],  task_compressed_sizes[f],
                        enable_stats[f], stats, hist,
                        enable_sideways_stats, sw_stats, sw_hist,
                        output_fds,
                        nrecs_per_page_capacity[f], nrecs_in_page_count[f], max_nrecs_in_page_count[f], task_nrecs_per_page[f], nios_data[f], nwritten_sectors[f], f,
                        final_flush);
                    break;
                }
                default: break;
            }
        } else if (dirtyflags[f]) {
            char *buf = reinterpret_cast<char*>(buffers[f]);
            flush_page_to_storage(output_fds,
                buf, pagids[f], page_size,
                nrecs_in_page_count[f],
                task_nrecs_per_page[f],
                nios_data[f],
                nwritten_sectors[f],
                compression[f],
                task_compressed_sizes[f],
                comp_ctx,
                compression_base_page_ids[f],
                compressed_page_write_offsets[f],
                f,
                __LINE__
            );
            max_nrecs_in_page_count[f] = std::max(nrecs_in_page_count[f], max_nrecs_in_page_count[f]);
            dirtyflags[f] = false; // Reset offset after writing
        }
    }

    if (enable_lbc) {
        for (size_t i = 0; i < NumVarCharFields; ++i) {
            const size_t nclusters = varchar_cluster_thresholds[i].size();
            const size_t f = task.varchar_field_indexes[i];
            for (size_t k = 0; k < nclusters; ++k) {
                if (dirtyflags_varchar[i][k])
                {
                    char *buf = reinterpret_cast<char*>(lbc_buffers[i][k]);
                    flush_page_to_storage(output_fds,
                        buf, lbc_pagids[i][k], page_size,
                        // FIXME: next
                        nrecs_in_page_count_varchar[i][k],
                        task_nrecs_per_page_varchar[i][k],
                        nios_data[f],
                        nwritten_sectors[f],
                        compression[f],
                        task_compressed_sizes_varchar[i][k],
                        comp_ctx,
                        compression_base_page_ids_varchar[i][k],
                        compressed_page_write_offsets_varchar[i][k],
                        f,
                        __LINE__
                    );
                    max_nrecs_in_page_count[f] = std::max(nrecs_in_page_count_varchar[i][k], max_nrecs_in_page_count[f]);
                    dirtyflags_varchar[i][k] = false; // Reset offset after writing
                }
            }
        }
    }
    // exit(0);


    if (verbose) std::cout << "Finished loading file: " << path << ", total records loaded: " << nrecs_loaded << std::endl;
    for (size_t f = 0; f < num_buffers_used; ++f) {
        // std::cout << "(bufferid="<< f << ") used " << npages_used[f] << ", allocated " << npages[f] << std::endl;
        if (npages_used[f] > npages_limit[f]) {
            std::cerr << "[ERROR] Exceeded the allocated number of pages for field " << f
                      << ": used " << npages_used[f]
                      << ", allocated " << npages_limit[f] << std::endl;
            exit(1);
        }
    }

    /* use buffers[0] */
    #if 0
    {
        buf = buffers[0];
        memset(buf, 0, page_size);
        pag_init(buf, page_size);
        for (size_t i = 0; i < NumVarCharFields; ++i) {
            for (size_t j = 0; j < num_varchar_clusters; ++j) {
                size_t pagid_base = arr_vec_npages_for_varchar_clusters[i][j];
                size_t npages = arr_vec_npages_for_varchar_clusters[i][j];

                size_t bi = i * num_varchar_clusters + j + 1;
                size_t pid = pagids[bi];
                assert(pagid_base <= pid);
                for (size_t k = pid - pagid_base; k < npages; ++k) {
                    page_pwrite_host(output_fds, buf, pagid_base + k, page_size);
                }
            }
        }
    }
    #endif

    // free(recbuf_base);

    #if 1
    if (enable_lbc) {
        for (size_t i = 0; i < NumVarCharFields; ++i) {
            free(lbc_buffer_starts[i]);
        }
    }

    if (compress) {
        free_compression_context(&comp_ctx);
    }

    if (dict_encoding) {
        /* TODO add test */
        for (size_t i = 0; i < NumRecBufs; ++i) {
            if (i < 3 && nrecs_inserted_total[0] >= nrecs_inserted_total[i]) {
                std::cout << "[WARNING] FileID: " << task.file_id << ", Field " << i << " - Expected: " << nrecs_inserted_total[0]
                          << ", Actual: " << nrecs_inserted_total[i] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
     } else {
        for (size_t i = 0; i < NumRecBufs; ++i) {
            if (nrecs_inserted_total[0] != nrecs_inserted_total[i]) {
                std::cout << "[WARNING] FileID: " << task.file_id << ", Field " << i << " - Expected: " << nrecs_inserted_total[0]
                          << ", Actual: " << nrecs_inserted_total[i] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    #endif
    return { nrecs_loaded, nios_data, nwritten_sectors, max_nrecs_in_page_count, nrecs_inserted_total };
}


template <typename EnumType, size_t MaxNumBuffers, size_t NFields, size_t NumStartXtns,
    size_t NumVarCharFields, size_t NumFilters, size_t NumSidewaysFilters = 0>
uint64_t load_multiple_files_to_table_column(
    LoaderOptions &options, TPCHTableMetadata &metadata, TPCH::common::Table table,
    std::vector<ColumnLoadTask<NFields, NumStartXtns, NumVarCharFields, NumFilters, NumSidewaysFilters>>& tasks,
    std::vector<std::string>& paths,
    const std::array<char*, MaxNumBuffers>& buffers,
    const size_t start_page_id, size_t *npages_used_sum,
    StatEntry &entry)
{
    constexpr size_t NumRecBufs = NFields;

    std::vector<std::future<TPCHLoaderThreadStats<MaxNumBuffers, NumRecBufs>>> futures;
    std::vector<std::thread> threads;
    std::vector<size_t> nitems_loaded;
    std::vector<std::array<char*, MaxNumBuffers>> thread_specific_buffers;

    /* prep buffers for each threads */
#if 0
    for (size_t i = 0; i < tasks.size(); ++i) {
        std::array<char*, MaxNumBuffers> buf;
        for (size_t j = 0; j < MaxNumBuffers; ++j) {
            buf[j] = &buffers[j][i * options.page_size];
            printf("Thread %zu, Buffer %zu: %p\n", i, j, buf[j]);
        }
        thread_specific_buffers.push_back(buf);
    }
#else
    for (size_t i = 0; i < tasks.size(); ++i) {
        std::array<char*, MaxNumBuffers> buf;
        for (size_t j = 0; j < NFields; ++j) {
            buf[j] = &buffers[j][i * options.page_size];
            if (options.verbose) printf("Thread %zu, Buffer %zu: %p\n", i, j, buf[j]);
        }
        for (size_t j = NFields; j < MaxNumBuffers; ++j) {
            buf[j] = nullptr;
        }
        thread_specific_buffers.push_back(buf);
    }
#endif
    // For debugging, please change the number of tasks here
    // DEBUG
    const size_t num_tasks = tasks.size();
    //const size_t num_tasks = 1;
    //const size_t num_tasks = 2;

    for (size_t i = 0; i < num_tasks; ++i) {
        auto load_func = static_cast<TPCHLoaderThreadStats<MaxNumBuffers, NumRecBufs>(*)(
            ColumnLoadTask<NFields, NumStartXtns, NumVarCharFields, NumFilters, NumSidewaysFilters>&,
            std::vector<std::string>&,
            std::vector<int>&,
            const std::array<char*, MaxNumBuffers>&,
            const size_t, const int, const int)>
            (&load_lines_to_table_as_column<EnumType>);

        std::packaged_task<TPCHLoaderThreadStats<MaxNumBuffers, NumRecBufs>()> task(
            std::bind(load_func,
                std::ref(tasks[i]), std::ref(paths), std::ref(options.output_fds),
                std::ref(thread_specific_buffers[i]),
                options.page_size, options.dryrun, options.verbose
        ));

        futures.push_back(task.get_future());
        threads.emplace_back(std::move(task));
    }

    for (auto& t : threads) {
        t.join();
    }

    size_t count_sum = 0;
    std::array<size_t, MaxNumBuffers> total_nios = {};
    std::array<size_t, MaxNumBuffers> total_nwritten_sectors = {};
    std::array<uint32_t, MaxNumBuffers> max_nrecs_in_page = {};
    std::array<size_t, NumRecBufs> nrecs_inserted_total = {};

    //for (size_t i = 0; i < futures.size(); ++i) {
    for (size_t i = 0; i < num_tasks; ++i) {
        auto result = futures[i].get();
        ssize_t count = result.nlines_processed;
        if (count < 0) {
            std::cerr << "[FATAL] Failed to count lines in file: " << paths[i] << std::endl;
            exit(EXIT_FAILURE);
        }
        count_sum += count;
        for (size_t j = 0; j < NumRecBufs; ++j) {
            total_nios[j] += result.nios[j];
            total_nwritten_sectors[j] += result.nwritten_sectors[j];
            max_nrecs_in_page[j] = std::max(max_nrecs_in_page[j], result.max_nrecs_in_page[j]);
        }
        for (size_t j = 0; j < NumRecBufs; ++j) {
            nrecs_inserted_total[j] += result.nrecs_inserted_total[j];
        }
    }

    /* Persist prefix sum array */
    const size_t table_first_page_id = tasks[0].start_page_ids[0];
    size_t start_base_page_id = start_page_id;
    std::array<std::vector<uint64_t>, NFields> prefix_sum_nrecs_per_page = {};
    /* LbC only */
    std::array<std::vector<uint64_t>, NumVarCharFields> prefix_sum_nrecs_per_page_varchar = {};
    {
        size_t npages_sum = 0;
        size_t k = 0;
        size_t vf = 0;
        for (size_t i = 0 ; i < NFields; ++i) {
            const enum rec_type field_type = tasks[0].field_types[i];
            const bool enable_lbc = tasks[0].enable_lbc;
            if (field_type == rec_type::REC_ATTR_VCHAR && tasks[0].enable_lbc) {
                prefix_sum_nrecs_per_page_varchar[vf].clear();
                prefix_sum_nrecs_per_page_varchar[vf].push_back(0);
                const size_t nclusters = tasks[0].lbc_varchar_cluster_thresholds[vf].size();
                for (size_t j = 0; j < nclusters; ++j) {
                    for (size_t l = 0; l < num_tasks; ++l) {
                        auto &task_nrecs_per_page_varchar = tasks[l].nrecs_per_page_varchar[vf][j];
                        for (const auto &rec_count : task_nrecs_per_page_varchar) {
                            uint64_t next_sum = prefix_sum_nrecs_per_page_varchar[vf].back() + rec_count;
                            prefix_sum_nrecs_per_page_varchar[vf].push_back(next_sum);
                            if (options.verbose) std::cout << "Field " << i << " (LbC), PageID " << table_first_page_id + k
                                << ", Task " << l << ", Cluster " << j << ", rec_count: " << rec_count << ", next_sum: " << next_sum << std::endl;
                            ++k;
                        }
                    }
                }
                prefix_sum_nrecs_per_page[i] = prefix_sum_nrecs_per_page_varchar[vf];
                vf++;
            } else {
                prefix_sum_nrecs_per_page[i].clear();
                prefix_sum_nrecs_per_page[i].push_back(0);
                for (size_t j = 0; j < num_tasks; ++j) {
                    auto& task_nrecs_per_page = tasks[j].nrecs_per_page[i];
                    for (const auto& rec_count : task_nrecs_per_page) {
                        uint64_t next_sum = prefix_sum_nrecs_per_page[i].back() + rec_count;
                        prefix_sum_nrecs_per_page[i].push_back(next_sum);
                        if (options.verbose) std::cout << "Field " << i << ", PageID " << table_first_page_id + k
                            << ", Task " << j << ", rec_count: " << rec_count << ", next_sum: " << next_sum << std::endl;
                        ++k;
                    }
                }
            }

            npages_sum += metadata_set_prefix_sum_nrecs_per_page(
                metadata, table, options.output_fds, options.page_size,
                i, start_base_page_id + npages_sum,
                prefix_sum_nrecs_per_page[i]
            );
        }
        if (options.verbose) std::cout << "Total pages used for prefix sum nrecs: from " << *npages_used_sum;
        *npages_used_sum += npages_sum;
        if (options.verbose) std::cout << " to " << *npages_used_sum << std::endl;
        start_base_page_id += npages_sum;
    }
    // exit(1);

    /* Persist stats */
    if constexpr (NumFilters > 0) {
        size_t npages_sum = 0;
        for (size_t i = 0 ; i < NFields; ++i) {
            enum rec_type field_type = tasks[0].field_types[i];
            bool enable_stats = tasks[0].enable_stats[i];
            if (enable_stats) {
                /* only three types are assumeted to be used for now */
                switch (field_type) {
                    case rec_type::REC_ATTR_INT32: {
                        std::vector<Stats<int32_t>> converted_hist {};
                        for (size_t j = 0; j < num_tasks; ++j) {
                            auto& hist = std::get<std::vector<ColumnStats<int32_t>>>(tasks[j].all_stats_history_per_page[i]);
                            for (auto& stat : hist) {
                                converted_hist.emplace_back(
                                    Stats<int32_t>{
                                        .min_val = stat.min_val,
                                        .max_val = stat.max_val
                                    }
                                );
                            }
                        }
                        npages_sum += metadata_set_stats(
                            metadata, table,
                            options.output_fds, options.page_size,
                            i,
                            start_base_page_id + npages_sum,
                            converted_hist
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_INT64: {
                        std::vector<Stats<int64_t>> converted_hist {};
                        for (size_t j = 0; j < num_tasks; ++j) {
                            auto& hist = std::get<std::vector<ColumnStats<int64_t>>>(tasks[j].all_stats_history_per_page[i]);
                            for (auto& stat : hist) {
                                converted_hist.emplace_back(
                                    Stats<int64_t>{
                                    .min_val = stat.min_val,
                                    .max_val = stat.max_val
                                });
                            }
                        }
                        npages_sum += metadata_set_stats(
                            metadata, table,
                            options.output_fds, options.page_size,
                            i,
                            start_base_page_id + npages_sum,
                            converted_hist
                        );
                        break;
                    }
                    case rec_type::REC_ATTR_DATE: {
                        std::vector<Stats<DateType>> converted_hist {};
                        for (size_t j = 0; j < num_tasks; ++j) {
                            auto& hist = std::get<std::vector<ColumnStats<DateType>>>(tasks[j].all_stats_history_per_page[i]);
                            for (auto& stat : hist) {
                                converted_hist.emplace_back(
                                    Stats<DateType>{
                                    .min_val = stat.min_val,
                                    .max_val = stat.max_val
                                });
                            };
                        }
                        npages_sum += metadata_set_stats(
                            metadata, table,
                            options.output_fds, options.page_size,
                            i,
                            start_base_page_id + npages_sum,
                            converted_hist
                        );
                        break;
                    }
                    default:
                    {
                        std::cerr << "[ERROR] Unsupported field type for statistics: " << static_cast<int>(field_type) << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
            } else {
                metadata_set_no_stats(metadata, table, i);
            }
        }

#if 0
        /* currently, NumFilters must be one */
        for (size_t fidx = 0; fidx < NumFilters; ++fidx) {
            size_t field_index = tasks[0].filter_field_indexes[fidx];
            std::cout << "Filter statistics for field " << field_index << ":\n";
            for (size_t t = 0; t < tasks.size(); ++t) {
                // auto& filters = tasks[t].arr_vec_filters[fidx];
                //auto& filters = tasks[t].arr_vec_filters_per_page[fidx];

                enum rec_type field_type = tasks[0].field_types[field_index];
                switch (field_type) {
                    case rec_type::REC_ATTR_DATE: {
                        auto& hist = std::get<std::vector<ColumnStats<DateType>>>(tasks[t].all_stats_history_per_page[field_index]);
                        std::vector<Stats<DateType>> converted_hist {};
                        for (auto& stat : hist) {
                            converted_hist.emplace_back(
                                Stats<DateType>{
                                    .min_val = stat.min_val,
                                    .max_val = stat.max_val
                                }
                            );
                        }
                        metadata_set_stats(
                            metadata, table,
                            options.output_fds, options.page_size,
                            field_index,
                            start_base_page_id + npages_sum,
                            converted_hist
                        );
#if 0
                        std::cout << "  Task " << t << " has " << hist.size() << " filter pages.\n";
                        for (size_t k = 0; k < hist.size(); ++k) {
                            std::cout << "    Filter " << k
                                      << ": min=" << hist[k].min_val
                                      << ", max=" << hist[k].max_val << "\n";
                        }
#endif
                        if constexpr (std::is_same_v<EnumType, TPCH::common::LineitemField>) {
                            if (field_index == TPCH::common::LineitemField::L_SHIPDATE) {
                                g_date_columnstats.insert(
                                    g_date_columnstats.end(),
                                    hist.begin(),
                                    hist.end()
                                );
#if 0
                                size_t count_overlapping = 0;
                                for (auto& filter : hist) {
                                    if (filter.overlaps({19940101}, {19950100})) {
                                        count_overlapping++;
                                    }
                                }
                                std::cout << "    Overlapping filters with [1994-01-01, 1995-01-01): " << count_overlapping << " / " << hist.size() << std::endl;
                                std::cout << "Ratio: " << static_cast<double>(count_overlapping) / hist.size() << std::endl;
#endif
                            }
                        }
                       break;
                    }
                    default:
                        std::cerr << "[ERROR] Unsupported field type for filter statistics: " << static_cast<int>(field_type) << std::endl;
                        exit(EXIT_FAILURE);
                }
                //std::copy(filters.begin(), filters.end(), std::back_inserter(g_date_columnstats));
            }

            if constexpr (std::is_same_v<EnumType, TPCH::common::LineitemField>) {
                if (field_index == TPCH::common::LineitemField::L_SHIPDATE) {
                    size_t count_overlapping = 0;
                    for (auto& filter : g_date_columnstats) {
                        if (filter.overlaps({19940101}, {19950100})) {
                            count_overlapping++;
                        }
                    }
                    std::cout << "Overlapping filters with [1994-01-01, 1995-01-01): " << count_overlapping << " / " << g_date_columnstats.size() << std::endl;
                    std::cout << "Ratio: " << static_cast<double>(count_overlapping) / g_date_columnstats.size() << std::endl;
                }
            }
        }
#endif
        if (options.verbose) std::cout << "Total pages used for prefix sum nrecs: from " << *npages_used_sum;
        *npages_used_sum += npages_sum;
        if (options.verbose) std::cout << " to " << *npages_used_sum << std::endl;
        start_base_page_id += npages_sum;
    }

    if (options.enable_sideways_stats) {
        // TODO: collect and persist golap stats
        if constexpr (NumSidewaysFilters > 0) {
            size_t npages_sum = 0;
            size_t vf = 0;
            for (size_t i = 0 ; i < NFields; ++i) {
                enum rec_type field_type = tasks[0].field_types[i];
                bool enable_lbc = tasks[0].enable_lbc;
                bool varchar_stats = (field_type == rec_type::REC_ATTR_VCHAR) && enable_lbc;
                for (size_t j = 0 ; j < NumSidewaysFilters; ++j) {
                    if (varchar_stats) {
                        enum rec_type stats_field_type = rec_type::REC_ATTR_INT32;
                        if constexpr (std::is_same_v<EnumType, TPCH::common::LineitemField>) {
                            stats_field_type = TPCH::common::fmt.lineitem_sideways_information_types[j];
                        } else if constexpr (std::is_same_v<EnumType, TPCH::common::OrdersField>) {
                            stats_field_type = TPCH::common::fmt.orders_sideways_information_types[j];
                        } else {
                            std::cerr << "[ERROR] Unsupported EnumType for sideways statistics." << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        std::vector<Stats<int32_t>> converted_hist {};
                        /* only three types are assumeted to be used for now */
                        switch (stats_field_type) {
                            case rec_type::REC_ATTR_INT32:
                            case rec_type::REC_ATTR_DATE:
                            case rec_type::REC_ATTR_CHAR: {
                                /* VChar sideways stats */
                                const size_t nclusters = tasks[0].lbc_varchar_cluster_thresholds[vf].size();
                                for (size_t k = 0; k < nclusters; ++k) {
                                    for (size_t l = 0; l < num_tasks; ++l) {
                                        /* NOTE: vf is incremeted after this loop */
                                        auto& hist = tasks[l].all_sideways_stats_per_page_varchar[vf][k][j];
                                        for (auto& stat : hist) {
                                            converted_hist.emplace_back(
                                                Stats<int32_t>{
                                                    .min_val = stat.min_val,
                                                    .max_val = stat.max_val
                                                }
                                            );
                                        }
                                    }
                                }
                                break;
                            }
                            default:
                            {
                                std::cerr << "[ERROR] Unsupported field type for statistics: " << static_cast<int>(stats_field_type) << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            npages_sum += metadata_set_sideways_stats(
                                metadata, table,
                                options.output_fds, options.page_size,
                                i, j,
                                start_base_page_id + npages_sum,
                                converted_hist
                            );
                            if (options.verbose) {
                                for (const auto& s : converted_hist) {
                                    std::cout << "Field " << i << ", Sideways Filter " << j
                                        << ": min=" << s.min_val << ", max=" << s.max_val << std::endl;
                                }
                            }
                        }
                    } else {
                        enum rec_type stats_field_type = rec_type::REC_ATTR_INT32;
                        if constexpr (std::is_same_v<EnumType, TPCH::common::LineitemField>) {
                            stats_field_type = TPCH::common::fmt.lineitem_sideways_information_types[j];
                        } else if constexpr (std::is_same_v<EnumType, TPCH::common::OrdersField>) {
                            stats_field_type = TPCH::common::fmt.orders_sideways_information_types[j];
                        } else {
                            std::cerr << "[ERROR] Unsupported EnumType for sideways statistics." << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        /* only three types are assumeted to be used for now */
                        switch (stats_field_type) {
                            case rec_type::REC_ATTR_INT32:
                            case rec_type::REC_ATTR_DATE:
                            case rec_type::REC_ATTR_CHAR: {
                                std::vector<Stats<int32_t>> converted_hist {};
                                for (size_t k = 0; k < num_tasks; ++k) {
                                    auto& hist = tasks[k].all_sideways_stats_per_page[i][j];
                                    for (auto& stat : hist) {
                                        converted_hist.emplace_back(
                                            Stats<int32_t>{
                                                .min_val = stat.min_val,
                                                .max_val = stat.max_val
                                            }
                                        );
                                    }
                                }
                                npages_sum += metadata_set_sideways_stats(
                                    metadata, table,
                                    options.output_fds, options.page_size,
                                    i, j,
                                    start_base_page_id + npages_sum,
                                    converted_hist
                                );
                                if (options.verbose) {
                                    for (const auto& s : converted_hist) {
                                        std::cout << "Field " << i << ", Sideways Filter " << j
                                            << ": min=" << s.min_val << ", max=" << s.max_val << std::endl;
                                    }
                                }

                               break;
                            }
                            default:
                            {
                                std::cerr << "[ERROR] Unsupported field type for statistics: " << static_cast<int>(stats_field_type) << std::endl;
                                exit(EXIT_FAILURE);
                            }
                        }
                    }

                    if (options.verbose) std::cout << "Total pages used for sideways information - "
                        << " SidewaysFilter: " << j << " for NField: " << i << " : from " << *npages_used_sum;
                    *npages_used_sum += npages_sum;
                    if (options.verbose) std::cout << " to " << *npages_used_sum << std::endl;
                    start_base_page_id += npages_sum;
                }
                if (varchar_stats) {
                    vf++;
                }
            }
        }
    }
    // exit(0);

    if (options.compress) {
        /* persist compressed page sizes */
        {
            size_t npages_sum = 0;
            size_t k = 0;
            size_t vf = 0;
            std::array<std::vector<uint32_t>, NFields> compressed_page_sizes = {};
            for (size_t i = 0; i < NFields; ++i)
            {
                CompressionMethod compression = CompressionMethod::NONE;
                if (options.enable_golap_compression_mode)
                {
                    compression = golap_compression_methods[i];
                }
                else
                {
                    /* PiG */
                    compression = tasks[0].compression_methods[i];
                    if (compression == CompressionMethod::LZ4 && options.enable_lz4par) {
                        compression = CompressionMethod::LZ4PAR;
                    }
                    if (compression == CompressionMethod::LZ4 && options.enable_fsst
                        && ((tasks[0].field_types[i] == rec_type::REC_ATTR_CHAR && tasks[0].field_sizes[i] >= 3)
                            || tasks[0].field_types[i] == rec_type::REC_ATTR_VCHAR)) {
                        compression = CompressionMethod::FSST_ROWID;
                    }
                }

                entry.metrics[i].compression_method = compression;

                if (compression == CompressionMethod::NONE)
                {
                    if (options.verbose) std::cout << "Field " << i << " has no compression. Skipping persisting compressed page sizes." << std::endl;
                    metadata_set_no_compression(metadata, table, i);
                }
                else
                {
                    enum rec_type field_type = tasks[0].field_types[i];
                    bool enable_lbc = tasks[0].enable_lbc;
                    const bool varchar_stats = (field_type == rec_type::REC_ATTR_VCHAR) && enable_lbc;

                    if (varchar_stats)
                    {
                        const size_t nclusters = tasks[0].lbc_varchar_cluster_thresholds[vf].size();
                        for (size_t cid = 0; cid < nclusters; ++cid)
                        {
                            for (size_t l = 0; l < num_tasks; ++l)
                            {
                                auto &task_compressed_sizes = tasks[l].compressed_sizes_per_page_varchar[vf][cid];
                                for (const auto &compressed_size : task_compressed_sizes)
                                {
                                    compressed_page_sizes[i].push_back(compressed_size);
                                    if (options.verbose) std::cout << "Field " << i << ", VChar Cluster " << cid
                                              << ", PageID " << table_first_page_id + k
                                              << ", Task " << l << ", compressed_size: " << compressed_size << std::endl;
                                    ++k;
                                }
                            }
                        }
                        vf++;
                    }
                    else
                    {
                        for (size_t j = 0; j < num_tasks; ++j)
                        {
                            auto &task_compressed_sizes = tasks[j].compressed_sizes_per_page[i];
                            for (const auto &compressed_size : task_compressed_sizes)
                            {
                                compressed_page_sizes[i].push_back(compressed_size);
                                if (options.verbose) std::cout << "Field " << i << ", PageID " << table_first_page_id + k
                                          << ", Task " << j << ", compressed_size: " << compressed_size << std::endl;
                                ++k;
                            }
                        }
                    }

                    // metadata_set_compressed_sizes_page_ids
                    npages_sum += metadata_set_compressed_page_sizes(
                        metadata, table,
                        options.output_fds, options.page_size,
                        i, compression,
                        start_base_page_id + npages_sum,
                        compressed_page_sizes[i]);
                }
            }
            if (options.verbose) std::cout << "Total pages used for compressed page sizes: from " << *npages_used_sum;
            *npages_used_sum += npages_sum;
            if (options.verbose) std::cout << " to " << *npages_used_sum << std::endl;
            start_base_page_id += npages_sum;
        }

        /* persist compression_base */
        {
            size_t npages_sum = 0;
            size_t k = 0;
            size_t vf = 0;
            std::array<std::vector<uint64_t>, NFields> compressed_base_start_page_ids = {};
            for (size_t i = 0; i < NFields; ++i)
            {
                CompressionMethod compression = CompressionMethod::NONE;
                if (options.enable_golap_compression_mode)
                {
                    compression = golap_compression_methods[i];
                }
                else
                {
                    /* PiG */
                    compression = tasks[0].compression_methods[i];
                }

		        if (compression == CompressionMethod::GOLAP) {
                          std::cout << "Field " << i << " has's compression is set to GOLAP. Unexpected." << std::endl;
		            exit(EXIT_FAILURE);
		        }
                else if (compression == CompressionMethod::NONE)
                {
                    if (options.verbose) std::cout << "Field " << i << " has no compression. Skipping persisting compressed base." << std::endl;
                }
                else
                {
                    enum rec_type field_type = tasks[0].field_types[i];
                    bool enable_lbc = tasks[0].enable_lbc;
                    const bool varchar_stats = (field_type == rec_type::REC_ATTR_VCHAR) && enable_lbc;

                    uint64_t nbases = 0;
                    if (varchar_stats)
                    {
                        const size_t nclusters = tasks[0].lbc_varchar_cluster_thresholds[vf].size();
                        for (size_t cid = 0; cid < nclusters; ++cid)
                        {
                            for (size_t j = 0; j < num_tasks; ++j)
                            {
                                auto start_page_id = tasks[j].start_page_ids_varchar[vf][cid];
                                compressed_base_start_page_ids[i].push_back(start_page_id);
                                if (options.verbose) std::cout << "Field " << i << ", VChar Cluster " << cid
                                    << ", PageID " << table_first_page_id + k
                                    << ", Task " << j << ", compressed_base_start_page_ids: " << compressed_base_start_page_ids[i][j] << std::endl;
                                ++nbases;
                                ++k;
                            }
                        }
                        vf++;
                    }
                    else
                    {
                        for (size_t j = 0; j < num_tasks; ++j)
                        {
                            auto start_page_id = tasks[j].start_page_ids[i];
                            compressed_base_start_page_ids[i].push_back(start_page_id);
                            if (options.verbose) std::cout << "Field " << i
                                << ", PageID " << table_first_page_id + k
                                << ", Task " << j << ", compressed_base_start_page_ids: " << compressed_base_start_page_ids[i][j] << std::endl;
                            ++nbases;
                            ++k;
                        }
                    }
                    if (nbases != compressed_base_start_page_ids[i].size()) {
                        std::cerr << "[ERROR] Mismatch in number of compressed bases for field " << i
                                  << ": expected " << nbases
                                  << ", got " << compressed_base_start_page_ids[i].size() << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    npages_sum += metadata_set_compression_base_page_id(metadata, table,
                            options.output_fds, options.page_size, i,
                            start_base_page_id + npages_sum,
                            compressed_base_start_page_ids[i]
                        );
                }
            }

            if (options.verbose) std::cout << "Total pages used for compressed page sizes: from " << *npages_used_sum;
            *npages_used_sum += npages_sum;
            if (options.verbose) std::cout << " to " << *npages_used_sum << std::endl;
            start_base_page_id += npages_sum;
        }
    }
    // exit(0);

    for (int i = 0; i < NumRecBufs; ++i) {
        //if (i == 0 || options.enable_dict_encoding) {
        entry.metrics[i].fields_written = total_nios[i];
        entry.metrics[i].fields_written_compressed = total_nwritten_sectors[i];
        metadata_set_max_nrows_in_page(metadata, table, i, max_nrecs_in_page[i]);
        if (options.verbose) std::cout << "Max rows in page for buffer " << i << ": " << max_nrecs_in_page[i] << std::endl;
    }

    std::cout << "Loaded " << count_sum << " records." << std::endl;
    std::cout << "Detail" << std::endl;
    for (size_t i = 0; i < NFields; ++i) {
        std::cout << "Field " << i;
        std::cout << ": " << nrecs_inserted_total[i] << " fields written." << std::endl;
    }
    // exit(1);

    return count_sum;
}


template <typename EnumType, size_t MaxNumBuffers, size_t NFields, size_t NumVarCharFields>
uint64_t load_single_file_to_table_column(LoaderOptions &options, TPCHTableMetadata &metadata, std::string& filename,
    const std::array<char*, MaxNumBuffers>& buffers,
    const std::array<uint64_t, NFields>& start_page_ids,
    const std::array<size_t, NFields>& field_sizes,
    const std::array<enum rec_type, NFields>& field_types,
    const std::array<size_t, NFields>& field_encoded_sizes,
    const std::array<enum rec_type, NFields>& field_encoded_types,
    const std::array<size_t, NFields>& arr_column_sizes,
    const std::array<EnumType, NFields>& target_fields,
    const std::array<size_t, NumVarCharFields>& varchar_field_indexes,
    const std::array<std::vector<size_t>, NumVarCharFields> varchar_cluster_thresholds,
    const std::array<std::vector<size_t>, NumVarCharFields> arr_vec_npages_for_varchar_clusters,
    StatEntry &entry)
{
    size_t num_buffers_used;
    if (options.enable_dict_encoding) {
        std::cerr << "Dictionary encoding is not supported for columnar loading." << std::endl;
        exit(1);
    } else {
        num_buffers_used = NFields;
    }

    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file.\n";
        return 0;
    }
    if (options.verbose) std::cerr << "Opened file:" << filename << std::endl;

    size_t siz_recbuf = 16384; // 16KB, should be enough for most records
    std::vector<char*> recbufs;
    char *recbuf_base = reinterpret_cast<char*>(malloc(siz_recbuf * num_buffers_used));
    for (size_t i = 0; i < num_buffers_used; ++i) {
        recbufs.push_back(&recbuf_base[i * siz_recbuf]);
    }
    REC *buf;

    // auto dict_types = TPCH::common::fmt.dict_types;
    
    uint64_t base_row_id = 0;
    /* initalize vectors */
    std::vector<bool> written {};
    std::vector<bool> dirtyflags {};
    std::vector<uint64_t> pagids {};
    /* tracking number of pages */
    std::vector<size_t> nios {};
    std::vector<size_t> nios_data {};
    std::vector<size_t> npages {};
    std::array<std::vector<size_t>, NumVarCharFields> arr_vec_start_pagids_for_varchar_clusters {};

    /* num_buffers_used is NumVarCharFields * options.num_varchar_clusters + 1 */
    std::vector<std::array<size_t, 1>> varchar_field_sizes {};
    std::vector<std::vector<std::unordered_map<std::string, uint32_t>>> vec_dict_maps {};
    std::vector<std::map<uint32_t, std::string>> sort_maps {};
    for (size_t i = 0; i < num_buffers_used; ++i) {
        written.push_back(false);
        dirtyflags.push_back(false);
        pagids.push_back(0);
        nios.push_back(0);
        nios_data.push_back(0);
        npages.push_back(0);
        pag_init(buffers[i], options.page_size);
    }

    for (size_t i = 0; i < NumVarCharFields; ++i) {
        assert(NumVarCharFields == varchar_field_indexes.size());
        if (options.verbose) std::cout << "VarcharFieldSize[" << i << "] = " << field_sizes[varchar_field_indexes[i]] << std::endl;
        vec_dict_maps.push_back(std::vector<std::unordered_map<std::string, uint32_t>> {});
        for (size_t j = 0; j < options.lbc_num_varchar_clusters; ++j) {
            vec_dict_maps[i].push_back(std::unordered_map<std::string, uint32_t> {});
        }
        varchar_field_sizes.push_back(std::array<size_t, 1> {
            field_sizes[varchar_field_indexes[i]]
        });
    }

#if 0
    size_t size_buffer = 128 * page_size;
    std::pmr::pool_options options;
    options.largest_required_pool_block = 256;
    options.max_blocks_per_chunk = size_buffer / options.largest_required_pool_block;
    std::pmr::unsynchronized_pool_resource shared_pool(options);
#endif

    std::array<uint64_t, NFields> xtns {};
    for (size_t i = 0; i < NFields; ++i) {
        /* preallocates pages here */
        
        pagids[i] = start_page_ids[i];
        if (options.verbose) std::cout << "Allocation from pagids[" << i << "] = " << pagids[i] << std::endl;
        nios[i] = 0;
    }

    size_t page_size = options.page_size;
    std::string line;

    uint64_t nrecs_loaded = 0;
    std::vector<bool> buffer_appended(num_buffers_used, false);
    std::vector<size_t> update_buffer_ids(num_buffers_used, 0);
    while (std::getline(file, line)) {
        size_t row_id = base_row_id + nrecs_loaded;
        // std::vector<std::string_view> fields;
        std::string_view sv(line);
        size_t f = 0;
        //size_t vf = 0; // varchar field index
        int16_t value16;
        int32_t value32;
        int64_t value64;
        char *valuechar;
        uint32_t slotid;
        char *pagbuf;
        enum rec_type attr_type;

        buffer_appended.assign(num_buffers_used, false);
        std::fill(buffer_appended.begin(), buffer_appended.end(), false);
        std::fill(update_buffer_ids.begin(), update_buffer_ids.end(), 0);

        auto fields = tpch_split_row(line);
        //pagbuf = reinterpret_cast<char*>(buffers[0]);

        if (options.verbose) std::cout << "Processing line: " << line << "\n";

        /* create record*/
        for (size_t i = 0; i < num_buffers_used; ++i) {
            // rec_init(arr_column_sizes[i], recbufs[i], siz_recbuf);
            memset(recbufs[i], 0, arr_column_sizes[i]);
        }
        if (options.enable_dict_encoding) {
            // for (size_t i = 1; i < num_buffers_used; ++i) {
            // }
        }

        for (const auto& part : fields) {
            buf = buffers[f];
            attr_type = field_types[f];

pag_append_retry:
            switch (attr_type) {
                case rec_type::REC_ATTR_INT16:
                    value16 = std::stoi(part.data());
                    //rec_set_attr_int16(arr_column_sizes[f], recbufs[f], siz_recbuf, 0, value16);
                    slotid = pagcol_append_rec_unordered_column_int32_with_rowid(buffers[f], value16, row_id, page_size);
                    break;
                case rec_type::REC_ATTR_INT32:
                    value32 = std::stoi(part.data());
                    //rec_set_attr_int32(arr_column_sizes[f], recbufs[f], siz_recbuf, 0, value32);
                    slotid = pagcol_append_rec_unordered_column_int32_with_rowid(buffers[f], value32, row_id, page_size);
                    break;
                case rec_type::REC_ATTR_INT64:
                    value64 = std::stoll(part.data());
                    //rec_set_attr_int64(arr_column_sizes[f], recbufs[f], siz_recbuf, 0, value64);
                    slotid = pagcol_append_rec_unordered_column_int64_with_rowid(buffers[f], value64, row_id, page_size);
                    break;
                case rec_type::REC_ATTR_CHAR:
                    if (field_sizes[f] < 3) {
                        value32 = static_cast<int32_t>(static_cast<uint8_t>(part[0]));
                        slotid = pagcol_append_rec_unordered_column_int32(buffers[f], value32, page_size);
                    } else {
                        valuechar = const_cast<char*>(part.data());
                        slotid = pagcol_append_rec_unordered_column_char_with_rowid(buffers[f], arr_column_sizes[f], valuechar, part.size(), row_id, page_size);
                        if (options.verbose) std::cout << "len[" << f << "]:" << arr_column_sizes[f] << std::endl;
                    }
                    break;
                case rec_type::REC_ATTR_VCHAR:
                    valuechar = const_cast<char*>(part.data());
                    // std::cout << "VCHAR Append: field=" << f << ", value=" << std::string(valuechar, part.size()) << ", size=" << part.size() << std::endl;
                    slotid = pagcol_append_rec_unordered_column_vchar_with_rowid(buffers[f], arr_column_sizes[f], valuechar, part.size(), row_id, page_size);
                    break;
                default:
                    std::cerr << "Unsupported field type: " << static_cast<int>(attr_type) << "\n";
                    return 0;
            }

            if (slotid == PAG_SLOTID_MASK_ERROR) {
                fprintf(stderr, "[%s](%d) Page is full - stop. (not yet implemented)", __func__, __LINE__);
                exit(1);
                pagbuf = reinterpret_cast<char*>(buffers[f]);

                uint64_t pagid = pagids[f];
                if (options.dryrun) {
                    if (options.verbose) std::cout << "Buffer[0] is full, writing to disk. (pageid=" << pagid << ")\n";
                } else {
                    if (options.verbose) {
                        std::cout << "Writing buffer[0] to disk. (pageid=" << pagid << ")\n";
                    }
                    page_pwrite_host(options.output_fds, pagbuf, pagid, options.page_size);
                    written[f] = true;
                    dirtyflags[f] = false; // Reset dirty flag after writing
                    nios[f]++;
                    nios_data[f]++;
                }
                /* Update next page */
                uint64_t next_pagid = pagids[f] + 1;
                pagids[f] = next_pagid;
                pag_init(pagbuf, options.page_size);
                /* try inserting page again here */
                goto pag_append_retry;
            } else {
                dirtyflags[f] = true;
            }

            f++;
        }

        nrecs_loaded++;
    }

    for (size_t i = 0; i < num_buffers_used; ++i) {
        uint64_t pagid = pagids[i];
        if (dirtyflags[i] || !written[i]) {
            if (options.verbose) std::cout << "Writing remaining data for field " << i << "...\n";
            // Write remaining data to disk or process it as needed
            char *buf = reinterpret_cast<char*>(buffers[i]);
            if (options.dryrun) {
                if (options.verbose) std::cout << "Remaining data in a buffer[" << i << "], writing to disk. (pageid=" << pagid << ")\n";
            } else {
                if (options.verbose) {
                    std::cout << "Remaining data in a buffer[" << i << "], writing to disk. (pageid=" << pagid << ")\n";
                }
                page_pwrite_host(options.output_fds, buf, pagid, options.page_size);
                nios[i]++;
                if (dirtyflags[i]) {
                    nios_data[i]++;
                }
            }
            dirtyflags[i] = false; // Reset offset after writing
        }
    }
    free(recbuf_base);
    if (options.verbose) std::cout << "Loaded " << nrecs_loaded << " records from file: " << filename << std::endl;

    // Update stats
    for (size_t i = 0; i < num_buffers_used; ++i) {
        assert(nios[i] == 1);
        assert(nios_data[i] == 1 || nios_data[i] == 0);
        assert(entry.metrics.size() == num_buffers_used);
        entry.metrics[i].fields_written += ((nios_data[i] * options.page_size) / MEBI);
        entry.metrics[i].fields_written_compressed = nios_data[i] * options.page_size / MEBI;
    }

    return nrecs_loaded;
}


template <typename EnumType, size_t NFields, size_t NumVarCharFields, size_t MaxNumBuffers>
uint64_t bulkload_single_file_column(
    LoaderOptions &options, TPCHTableMetadata &metadata, const TPCH::common::Table table,
    uint64_t start_page_id,
    const std::array<char*, MaxNumBuffers>& buffers,
    const std::array<size_t, NFields>& field_sizes,
    const std::array<enum rec_type, NFields>& field_types,
    const std::array<size_t, NFields>& field_encoded_sizes,
    const std::array<enum rec_type, NFields>& field_encoded_types,
    const std::array<size_t, NFields>& arr_column_sizes,
    const std::array<EnumType, NFields>& target_fields,
    const std::array<size_t, NumVarCharFields>& varchar_field_indexes,
    const std::array<std::vector<size_t>, NumVarCharFields> varchar_cluster_thresholds,
    const std::array<CompressionMethod, NFields>& compression_types,
    StatEntry &stat_entry) {

    std::stringstream ss;
    ss << options.input_dirname << "/" << TPCH::common::table_name(table) << "/" << TPCH::common::table_name(table) << ".tbl";
    std::string path = ss.str();

    std::string s = std::string(TPCH::common::table_name(table));
    std::transform(s.begin(), s.end(), s.begin(), [](char c) { return std::toupper(c); });
    auto table_name = s;

    for (size_t i = 0; i < NFields; ++i) {
        auto field_name = TPCH::common::metric_field_name(target_fields[i]);
        std::stringstream ss;
        ss << table_name << "_" << field_name;
        Metric metric = { ss.str(), 0, 0, 0, 0};
        stat_entry.metrics.push_back(metric);
        if (options.verbose) std::cout << ss.str() << std::endl;
    }

    uint64_t nrecs_loaded;
    std::array<uint64_t, NFields> start_page_ids;
    constexpr size_t alignment = 4;

    /* Calculate field_start_page_ids */
    //uint64_t npages_sum;
    uint64_t npages[MaxNumBuffers];
    uint64_t sizes_total[MaxNumBuffers];
    /* std::array of the following elements : npage[0] */
    /* std::vector<size_t> : num_clusters */
    std::array<std::vector<size_t>, NumVarCharFields> arr_vec_npages_per_dict_for_varchar_clusters {};

    size_t nrecs_total, size_total;
    /* For stats */
    size_t sum_size_total; // , sum_size_total_base;

    /* set nrecs_total from constants */
    switch (table) {
    case TPCH::common::NATION:
        nrecs_total = 25;
        break;
    case TPCH::common::REGION:
        nrecs_total = 5;
        break;
    default:
        std::cerr << "Unknown table type.\n";
        /* failed, then not advance pagid pointer */
        return start_page_id;
    }

    // size_t nrecs_per_xtn_row_format = 0;
    // size_t max_npages_for_rows = 0;

    sum_size_total = 0;
    for (size_t i = 0; i < NFields; ++i) {
        size_t siz_rowid = sizeof(uint64_t);
        size_t siz_field;
        /**
         * sampling (or 1pass) can be used to estimate
         * the size of each field with alignment, overheads, and stats.
         * Here, we use fixed size as estimation for each field.
         */
        siz_field = field_sizes[i];
        siz_field = siz_field % alignment == 0
            ? siz_field : (siz_field + (alignment - (siz_field % alignment)));

        if (field_types[i] == rec_type::REC_ATTR_VCHAR) {
            /**
             * column store layout's overhead for varchar:
             *   slotid (footer): sizeof(uint32_t) 
             *   varcharlen: 2 * sizeof(uint16_t)
             *   rowid: sizeof(uint64_t)
             **/
            siz_field += sizeof(uint32_t) + 2 * sizeof(uint16_t) + sizeof(uint64_t) + siz_rowid;
        } else {
            /**
             * column store layout's overhead:
             *   slotid (footer): sizeof(uint32_t) 
             *   rowid: sizeof(uint64_t)
             **/
            siz_field += siz_rowid;
        }

        size_t nrecs_per_page = (options.page_size - sizeof(struct pag_head)) / siz_field;
        size_t npages_estimated = (nrecs_total + nrecs_per_page - 1) / nrecs_per_page;

        size_total = npages_estimated * options.page_size;
        sizes_total[i] = size_total;
        start_page_ids[i] = start_page_id;
        npages[i] = npages_estimated;

        if (options.verbose) {
            std::cout << "Field " << i << " size per record: " << siz_field << " bytes "
              << " total size (record): " << nrecs_total * siz_field << " bytes, "
              << " total size (page): " << size_total << " bytes.\n";
            std::cout << "\tstart_page_id: " << start_page_ids[i]
              << ", npages: " << npages[i] << "\n";
        }

        metadata_set_page_id(metadata, table, i, start_page_ids[i], npages[i]);
        /* update stats */
        sum_size_total += size_total;
        start_page_id += npages[i];
    }

    if (options.enable_dict_encoding) {
#if 0
        const auto num_clusters = options.num_varchar_clusters;
        for (size_t i = 0; i < NumVarCharFields; ++i) {
            assert(max_npages_for_rows > 0);
            assert(nrecs_per_xtn_row_format > 0);
            const std::vector<size_t>& thresholds = varchar_cluster_thresholds[i];
            std::vector<size_t>& vec_npages_per_dict_for_varchar_clusters
                = arr_vec_npages_per_dict_for_varchar_clusters[i];
            size_t npages_total = 0;

            std::vector<size_t> &vec_npages_per_dict_for_varchar = 
                arr_vec_npages_per_dict_for_varchar_clusters[i - 1];

            for (size_t j = 0; j < num_clusters; ++j) {
                size_t nrecs_per_page = (options.page_size - sizeof(struct pag_head))
                     / (thresholds[j] + sizeof(uint32_t) + sizeof(uint16_t));
                size_t nrecs = nrecs_total;
                size_t npages = (nrecs + nrecs_per_page - 1) / nrecs_per_page;

                vec_npages_per_dict_for_varchar.push_back(npages);
                std::cout << "npages: " << npages << std::endl;
                npages_total += npages;
            }
            assert(arr_vec_npages_per_dict_for_varchar_clusters[i - 1].size() == num_clusters);

            size_total = npages_total * options.page_size;
            sizes_total[i] = size_total;

            start_page_ids[i] = start_page_id;
            npages[i] = (size_total + size_xtn - 1) / size_xtn;
            if (i == 0) {
                max_npages_for_rows = npages[i];
                metadata_set_page_id(metadata, table, 0, start_page_ids[i], npages[i]);
            }
            /* update stats */
            sum_size_total += size_total;
            start_page_id += npages[i];
        }
#endif
    }

    if (options.verbose) {
        std::cout << "Number of pages for " << TPCH::common::table_name(table) << " table, "
            << npages << " pages, start_page_ids: " << start_page_ids[0]
            << ", size_total: " << sizes_total[0] << " bytes. "
            << std::endl;
    }
    if (options.enable_dict_encoding) {
        #if 0
        std::cout << "Storage overhead for " << TPCH::common::table_name(table) << " table with dictionary encoding: "
            << sum_size_total_base << " bytes (base size) + "
            << sum_size_total - sum_size_total_base << " bytes (additional size) = "
            << sum_size_total << " bytes." << std::endl;
        #endif
    }

#ifdef DEBUG_PRINT
    std::cout << "Loading " << TPCH::common::table_name(table) << " table...\n";
#endif
    std::cout << "Loading " << TPCH::common::table_name(table) << " table...\n";
    nrecs_loaded = load_single_file_to_table_column<EnumType>(options, metadata, path, buffers,
        start_page_ids, field_sizes, field_types, field_encoded_sizes, field_encoded_types,
        arr_column_sizes,
        target_fields, varchar_field_indexes,
        varchar_cluster_thresholds, arr_vec_npages_per_dict_for_varchar_clusters,
        stat_entry);
    std::cout << "Done (" << TPCH::common::table_name(table) << " table).\n";

    // metadata_set_page_id(metadata, table, 0, start_page_id, npages);
    metadata_set_nrows(metadata, table, nrecs_loaded);

    // exit(1);

#ifdef DEBUG_PRINT
    std::cout << "Done (" << TPCH::common::table_name(table) << " table).\n";
#endif
    /* start_page_id is incremened at the end of loop */
    return start_page_id;
}

template <typename EnumType, size_t NFields, size_t NumVarCharFields, size_t MaxNumBuffers,
    size_t NumFilters>
uint64_t bulkload_multiple_files_column(
    LoaderOptions &options, TPCHTableMetadata &metadata, const TPCH::common::Table table,
    uint64_t start_page_id,
    const std::array<char*, MaxNumBuffers>& buffers,
    const std::array<size_t, NFields>& field_sizes,
    const std::array<enum rec_type, NFields>& field_types,
    const std::array<EnumType, NFields>& target_fields,
    const std::array<size_t, NumVarCharFields>& varchar_field_indexes,
    std::array<std::vector<size_t>, NumVarCharFields> &varchar_cluster_thresholds,
    const std::array<size_t, NumFilters>& filter_columns,
    const std::array<bool, NFields>& enable_stats_columns,
    const std::array<CompressionMethod, NFields>& compression_methods,
    // std::array<std::vector<int32_t>, N>& field_compression_offsets,
    // std::array<std::vector<int32_t>, N>& field_compressed_page_size,
    StatEntry &stat_entry)
{
    // std::array<uint64_t, NumVarCharFields> start_page_ids;
    if (table != TPCH::common::CUSTOMER && table != TPCH::common::LINEITEM
        && table != TPCH::common::ORDERS && table != TPCH::common::PARTSUPP
        && table != TPCH::common::PART && table != TPCH::common::SUPPLIER
    ) {
        std::cerr << "Unknown table type.\n";
        return 0;
    }
#ifdef DEBUG_PRINT
    std::cout << "Loading " << TPCH::common::table_name(table) << " table...\n";
#endif

    std::stringstream ss;
    ss << options.input_dirname << "/" << TPCH::common::table_name(table) << "/" << TPCH::common::table_name(table) << ".tbl";
    std::string path = ss.str();

    std::string s = std::string(TPCH::common::table_name(table));
    std::transform(s.begin(), s.end(), s.begin(), [](char c) { return std::toupper(c); });
    auto table_name = s;

    for (size_t i = 0; i < NFields; ++i) {
        auto field_name = TPCH::common::metric_field_name(target_fields[i]);
        std::stringstream ss;
        ss << table_name << "_" << field_name;
        Metric metric = { ss.str(), 0, 0, 0, 0};
        stat_entry.metrics.push_back(metric);
        if (options.verbose) std::cout << ss.str() << std::endl;
    }

    std::array<uint64_t, NFields> start_page_ids {};
    // uint64_t npages[MaxNumBuffers];
    uint64_t npages;
    // uint64_t sizes_total[MaxNumBuffers];
    uint64_t nrecs_loaded;

    /* This vector is initialized in generate_row_tasks function */
    std::vector<std::array<std::vector<size_t>, NumVarCharFields>>
        vec_arr_vec_npages_for_varchar_clusters;

    auto paths = prep_input_files(options, TPCH::common::table_name(table));

    // TPCH::common::fmt.customer_varchar_field_count + 1
    std::cout << options.enable_column << std::endl;
    std::cout << "Number of " << table_name << " files: " << paths.size() << std::endl;
#if 0
    auto tasks = generate_column_tasks(
        options, metadata, table,
        start_page_id, start_page_ids, paths,
        field_sizes, field_types,
        varchar_field_indexes, varchar_cluster_thresholds,
        filter_columns,
        enable_stats_columns,
        &npages);
#else
    auto tasks = [&]() {
        if constexpr (std::is_same_v<EnumType, TPCH::common::LineitemField>) {
            return generate_column_tasks<3>(
                options, metadata, table,
                start_page_id, start_page_ids, paths,
                field_sizes, field_types,
                varchar_field_indexes, varchar_cluster_thresholds,
                filter_columns,
                enable_stats_columns,
                compression_methods,
                &npages);
        } else if constexpr (std::is_same_v<EnumType, TPCH::common::OrdersField>) {
            return generate_column_tasks<2>(
                options, metadata, table,
                start_page_id, start_page_ids, paths,
                field_sizes, field_types,
                varchar_field_indexes, varchar_cluster_thresholds,
                filter_columns,
                enable_stats_columns,
                compression_methods,
                &npages);
        } else {
            return generate_column_tasks<0>(
                options, metadata, table,
                start_page_id, start_page_ids, paths,
                field_sizes, field_types,
                varchar_field_indexes, varchar_cluster_thresholds,
                filter_columns,
                enable_stats_columns,
                compression_methods,
                &npages);
        }
    }();

#endif
    if (options.verbose) std::cout << "Number of tasks files: " << tasks.size() << std::endl;

    if (options.verbose) {
        for (auto &path : paths) {
            std::cout << "Processing file: " << path << std::endl;
        }
    }

    //nrecs_loaded = load_multiple_files_to_table_column<TPCH::common::CustomerField>(
    nrecs_loaded = load_multiple_files_to_table_column<EnumType>(
         options, metadata, table, tasks, paths, buffers, start_page_id + npages, &npages, stat_entry);

    //metadata_set_page_id(metadata, table, 0, start_page_id, npages);
    metadata_set_nrows(metadata, table, nrecs_loaded);

    /* Compression pass (not yet implemented) */
#if 0
    if (options.compress) {
        for (auto task : tasks) {
                for (size_t f = 0; f < N; ++f) {
                    // std::cout << "\ttask.offsets[" << f << "] len: " << task.offsets[f].size() << std::endl;
                    auto& offsets = field_compression_offsets[f];
                    /* Extend the internal buffer of std::vector */
                    offsets.reserve(offsets.size() + task.offsets[f].size());
                    /* Merge each task into the single vector in bulk */
                    offsets.insert(offsets.end(), task.offsets[f].begin(), task.offsets[f].end());
                    // std::cout << "\t\t-->offsets[" << f << "] len: " << offsets.size() << std::endl;
                    task.offsets[f].clear();
                    task.offsets[f].shrink_to_fit();
                }
                for (size_t f = 0; f < N; ++f) {
                    auto& compressed_page_size = field_compressed_page_size[f];
                    compressed_page_size.reserve(compressed_page_size.size() + task.compressed_page_sizes[f].size());
                    compressed_page_size.insert(compressed_page_size.end(), task.compressed_page_sizes[f].begin(), task.compressed_page_sizes[f].end());
                    // std::cout << "\t\t-->compressed_page_size[" << f << "] len: " << compressed_page_size.size() << std::endl;
                }
                free(task.offsets_local);
                free(task.buf_compress);
        }
        if (options.verbose) {
            std::cout << "[INFO] Compression offsets and compressed page sizes collected." << std::endl;
            for (size_t f = 0; f < N; ++f) {
                auto& compressed_page_size = field_compressed_page_size[f];
                for (size_t i = 0; i < compressed_page_size.size(); ++i) {
                    std::cout << "[INFO] compressed_page_sizes[" << f << "][" << i << "] = " << compressed_page_size[i] << std::endl;
                }
            }
            for (size_t f = 0; f < N; ++f) {
                auto& offsets = field_compression_offsets[f];
                std::cout << "[INFO] offsets[" << f << "] len: " << offsets.size() << std::endl;
                for (size_t i = 0; i < offsets.size(); ++i) {
                    std::cout << "[INFO] offsets[" << f << "][" << i << "] = " << offsets[i] << std::endl;
                }
            }
        }
    }

    if (options.compress) {
        // Allocate a write buffer for offsets
        char *buf = static_cast<char*>(mb_alloc(options.page_size));
        uint *buf_u32 = reinterpret_cast<uint*>(buf);
        // Persist the offsets to disk
        for (size_t f = 0; f < N; ++f) {
            auto& offsets = field_compression_offsets[f];
            if (buf == nullptr) {
                std::cerr << "[FATAL] Failed to allocate memory for offsets buffer.\n";
                exit(EXIT_FAILURE);
            }
            char *buf_src = reinterpret_cast<char*>(offsets.data());

            size_t npages = TPCH::metadata_noffsets_to_nmetapages(offsets.size(), options.page_size);
            size_t size_offsets = offsets.size() * sizeof(uint32_t);
            std::cout << "[INFO] offsets[" << f << "] len: " << offsets.size() << ", npages: " << npages << std::endl;
            for (size_t i = 0; i < npages; ++i) {
                size_t page_id = metadata.compress_table_lineorder_offset_start_page_ids[f] + i;
                if (i == npages - 1) {
                    size_t last_page_size = size_offsets % options.page_size;
                    if (last_page_size == 0) {
                        last_page_size = options.page_size;
                    }
                    memcpy(buf, buf_src + i * options.page_size, last_page_size);
                    memset(buf + last_page_size, 0, options.page_size - last_page_size);
                } else {
                    memcpy(buf, buf_src + i * options.page_size, options.page_size);
                }
                if (options.dryrun) {
                    std::cout << "[INFO] offsets[" << f << "] writing to page id " << page_id << std::endl;
                } else {
                    if (options.verbose) {
                        std::cout << "[INFO] offsets[" << f << "] writing to page id " << page_id << std::endl;
                    }
                    std::cout << "[INFO] offsets[" << f << "] writing to page id " << page_id << std::endl;
                    #if 0
                    std::cout << "==="  << std::endl;
                    for (uint j = 0; j < options.page_size / sizeof(uint32_t); ++j) {
                        std::cout << buf_u32[j] << " ";
                        if (j > 0 && j % 16 == 0) {
                            std::cout << std::endl;
                        }
                    }
                    std::cout << "==="  << std::endl;
                    #endif
                    page_pwrite_host(options.output_fds, buf, page_id, options.page_size);
                }
            }
            stat_entry.metrics[f].fields_written_offset_for_compression+=npages;
        }

        // Persist the compressed page size to disk
        for (size_t f = 0; f < N; ++f) {
            auto& compressed_page_sizes = field_compressed_page_size[f];
            if (buf == nullptr) {
                std::cerr << "[FATAL] Failed to allocate memory for offsets buffer.\n";
                exit(EXIT_FAILURE);
            }
            char *buf_src = reinterpret_cast<char*>(compressed_page_sizes.data());

            size_t npages = TPCH::metadata_ncomp_pages_to_nmetapages(compressed_page_sizes.size(), options.page_size);
            size_t size_src_buf = compressed_page_sizes.size() * sizeof(uint32_t);
            std::cout << "[INFO] compressed_page_sizes[" << f << "] len: " << compressed_page_sizes.size() << ", npages: " << npages << std::endl;
            for (size_t i = 0; i < npages; ++i) {
                size_t page_id = metadata.compress_table_lineorder_compressed_page_size_start_page_ids[f] + i;
                if (i == npages - 1) {
                    size_t last_page_size = size_src_buf % options.page_size;
                    if (last_page_size == 0) {
                        last_page_size = options.page_size;
                    }
                    memcpy(buf, buf_src + i * options.page_size, last_page_size);
                    memset(buf + last_page_size, 0, options.page_size - last_page_size);
                } else {
                    memcpy(buf, buf_src + i * options.page_size, options.page_size);
                }
                if (options.dryrun) {
                    std::cout << "[INFO] compressed_page_sizes[" << f << "] writing to page id " << page_id << std::endl;
                } else {
                    if (options.verbose) {
                        std::cout << "[INFO] compressed_page_sizes[" << f << "] writing to page id " << page_id << std::endl;
                    }
                    page_pwrite_host(options.output_fds, buf, page_id, options.page_size);
                }
            }
            stat_entry.metrics[f].fields_written_sizcomp_for_compression+=npages;
        }
        free(buf);
    }
#endif

#ifdef DEBUG_PRINT
    std::cout << "done.n";
#endif
    return start_page_id + npages;
}

int open_output_file(const char *file)
{
    int oflag = 0;
    char *files = strdup(file);
    oflag = O_CREAT | O_WRONLY | O_DIRECT;

    int fd = open(file, oflag, 0644);
    if (fd < 0)
    {
        std::cerr << "failed to open file " << file << std::endl;
        perror("open");
        close(fd);
        exit(EXIT_FAILURE);
    }

    return fd;
}

void open_output_files(LoaderOptions &options, std::vector<int> &fds)
{
    int oflag = 0;
    oflag = O_RDWR | O_DIRECT;

    int i = 0;
    options.output_filename_orig = strndup(options.output_files, 16384);
    char *devarg = options.output_filename_orig;
    char *tp = strtok(devarg, ",");
    while (tp != NULL)
    {
        options.devname[i] = tp;
        i++;
        tp = strtok(NULL, ",");
    }
    if (tp == NULL && i == 0 && devarg != NULL)
    {
        options.devname[0] = devarg;
        i = 1;
    }
    options.ndev = i;

    fds.reserve(options.ndev);
    for (i = 0; i < options.ndev; i++)
    {
        int fd = open(options.devname[i], oflag, 0644);
        if (fd < 0)
        {
            std::cerr << "failed to open file " << options.output_files << std::endl;
            perror("open");
            close(fd);
            exit(EXIT_FAILURE);
        }
        fds.push_back(fd);
    }
    if (fds.size() != options.ndev)
    {
        perror("open");
        exit(EXIT_FAILURE);
    }

    return;
}

void close_output_files(LoaderOptions &options, std::vector<int> &fds)
{
    for (auto fd : fds)
    {
        if (fd >= 0)
        {
            close(fd);
        }
    }

    free(options.output_filename_orig);
}

std::pair<uint64_t, size_t> load_test_binpack(ulong N, uint *in_values, uint *out_values, uint *offsets, uint *decoded_values)
{
    if (N % 128 != 0) {
        std::cerr << "N must be a multiple of 128.\n";
        return {0, 0};
    }
    std::vector<ulong> in_values_array(N);
    /* NOTE: preserve the input values for verification */
    std::copy(in_values, in_values + N, in_values_array.begin());

    /* block_size = 128, mini block size is 32 */
    std::array<uint, 32> workspace;

    size_t size = binPack(in_values, out_values, offsets, N, workspace.data());
    uint64_t checksum = binUnpack(out_values, offsets, N, decoded_values);
    //binUnpackPrint(out_values, offsets, N, decoded_values);

    size_t checksum2 = 0;
    for (ulong i = 0; i < N; ++i) {
        if (in_values_array[i] != decoded_values[i]) {
            std::cerr << "Mismatch at index " << i << ": in=" << in_values_array[i]
                << ", decoded=" << decoded_values[i] << std::endl;
            exit(1);
        }
        checksum2 += decoded_values[i];
    }

    //std::cout << "Checksum1: " << checksum << ", Checksum2: " << checksum2 << std::endl;

    return {checksum, size};
}

std::pair<uint64_t, size_t> load_test_binpack64(ulong N, ulong *in_values, ulong *out_values, uint *offsets, ulong *decoded_values)
{
    if (N % 128 != 0) {
        std::cerr << "N must be a multiple of 128.\n";
        return {0, 0};
    }
    std::array<uint, 32> workspace;
    std::vector<ulong> in_values_array(N);
    std::copy(in_values, in_values + N, in_values_array.begin());
    size_t size = binPack64(in_values, out_values, offsets, N, workspace.data());
    // binUnpack(out_values, offsets, N);
    size_t checksum = binUnpack64(out_values, offsets, N, decoded_values);
    //binUnpack64Print(out_values, offsets, N);
    for (ulong i = 0; i < N; ++i) {
        if (in_values_array[i] != decoded_values[i]) {
            std::cerr << "Mismatch at index " << i << ": in=" << in_values_array[i]
                << ", decoded=" << decoded_values[i] << std::endl;
            exit(1);
        }
    }

    return {checksum, size};
}

void test_rec()
{
    char *buf = reinterpret_cast<char*>(malloc(16384));
    REC *rp = reinterpret_cast<REC*>(buf);

    char country[] = "ARGENTINA";
    char comment[] = "al foxes promise slyly according to the regular accounts. bold requests alon";

    rec_init(TPCH::common::fmt.nation_field_sizes, buf, 16384);
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, buf, 16384, 0, 1); // NATIONKEY
    rec_set_attr_char(TPCH::common::fmt.nation_field_sizes, buf, 16384, 1, country, 9); // NATIONKEY
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, buf, 16384, 2, 1); // NATIONKEY
    rec_set_attr_vchar(TPCH::common::fmt.nation_field_sizes, buf, 16384, 3, comment, 76); // COMMENT

    rec_print_rec(rp, TPCH::common::fmt.nation_field_types, stdout);
    //TPCH::common::fmt.nation_field_sizes;
    // REC *rec_set_attr_char(std::array<size_t, N> &sizes, char *buf, int len, int idx, char *val, int siz)
}


void test_pag()
{
    //const size_t page_size = 1 * MEBI;
    //char *pagbuf = reinterpret_cast<char*>(malloc(1 * MEBI));
    const size_t page_size = 512;
    char *pagbuf = reinterpret_cast<char*>(malloc(512));
    REC *pp = reinterpret_cast<PAG*>(pagbuf);

    char *recbuf = reinterpret_cast<char*>(malloc(16384));
    REC *rp = reinterpret_cast<REC*>(recbuf);
    // struct rec_head *rhp = reinterpret_cast<struct rec_head*>(recbuf);

    char country1[] = "ARGENTINA";
    char comment1[] = "al foxes promise slyly according to the regular accounts. bold requests alon";
    char country2[] = "BRAZIL";
    char comment2[] = "y alongside of the pending deposits. carefully special packages are about the ironic forges. slyly special ";
    char country3[] = "CANADA";
    char comment3[] = "eas hang ironic, silent packages. slyly regular packages are furiously over the tithes. fluffily bold";
    char country4[] = "EGYPT";
    char comment4[] = "y above the carefully unusual theodolites. final dugouts are quickly across the furiously regular d";

    pag_init(pp, 1 * MEBI);

    // INSERT1
    rec_init(TPCH::common::fmt.nation_field_sizes, recbuf, 16384);
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 0, 1); // NATIONKEY
    rec_set_attr_char(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 1, country1, 9); // NAME
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 2, 1); // REGIONKEY
    rec_set_attr_vchar(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 3, comment1, 76); // COMMENT

    rec_print_rec(rp, TPCH::common::fmt.nation_field_types, stdout);
  
    pag_append_rec_unordered(pp, rp, page_size);
    pag_print_pag_data(pp, TPCH::common::fmt.nation_field_types, page_size, stdout);

    // INSERT2
    rec_init(TPCH::common::fmt.nation_field_sizes, recbuf, 16384);
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 0, 2); // NATIONKEY
    rec_set_attr_char(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 1, country2, 6); // NAME
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 2, 1); // REGIONKEY
    rec_set_attr_vchar(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 3, comment2, 107); // COMMENT

    rec_print_rec(rp, TPCH::common::fmt.nation_field_types, stdout);

    pag_append_rec_unordered(pp, rp, page_size);
    pag_print_pag_data(pp, TPCH::common::fmt.nation_field_types, page_size, stdout);


    // INSERT3
    rec_init(TPCH::common::fmt.nation_field_sizes, recbuf, 16384);
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 0, 3); // NATIONKEY
    rec_set_attr_char(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 1, country3, 6); // NAME
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 2, 1); // REGIONKEY
    rec_set_attr_vchar(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 3, comment3, 101); // COMMENT

    rec_print_rec(rp, TPCH::common::fmt.nation_field_types, stdout);

    pag_append_rec_unordered(pp, rp, page_size);
    pag_print_pag_data(pp, TPCH::common::fmt.nation_field_types, page_size, stdout);

    // INSERT4
    rec_init(TPCH::common::fmt.nation_field_sizes, recbuf, 16384);
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 0, 4); // NATIONKEY
    rec_set_attr_char(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 1, country4, 5); // NAME
    rec_set_attr_int32(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 2, 1); // REGIONKEY
    rec_set_attr_vchar(TPCH::common::fmt.nation_field_sizes, recbuf, 16384, 3, comment4, 99); // COMMENT

    rec_print_rec(rp, TPCH::common::fmt.nation_field_types, stdout);

    assert(pag_append_rec_unordered(pp, rp, page_size) == PAG_SLOTID_MASK_ERROR);
    pag_print_pag_data(pp, TPCH::common::fmt.nation_field_types, page_size, stdout);
    fprintf(stdout, "Successfully confirmed that pag_append_rec_unordered() returned PAG_SLOTID_MASK_ERROR when the page is full.\n");

    //TPCH::common::fmt.nation_field_sizes;
    // REC *rec_set_attr_char(std::array<size_t, N> &sizes, char *buf, int len, int idx, char *val, int siz)
}

void test_scan_column_tables_with_sideways_stats(LoaderOptions &options)
{
    constexpr bool debug_print = false;
    // 512B-aligned alloc
    void *ptr;
    TPCHTableMetadata *metadatap;

    superpage_set_constants(options.page_size);

    if (posix_memalign((void**)&ptr, 512, options.page_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Call constructor explicitly to initialize the object correctly

    page_pread_host(options.output_fds, (void*)ptr, 0, options.page_size);
    metadatap = reinterpret_cast<TPCHTableMetadata*>(ptr);

    TPCHTableMetadata &metadata = *metadatap;

    size_t num_super_npages = superpage_get_super_npage();
    if (posix_memalign((void**)&ptr, 512,
        options.page_size * SuperPage::NumPagesForSuperPages) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    // metadata.xtn_head = new(ptr) struct xtn_entry[XTN::MaxNumXTNs];
    // memset(metadata.xtn_head, 0, sizeof(struct xtn_entry) * XTN::MaxNumXTNs);

    if (options.load_nation) {
        /* Memory allocation for scan test */
        void *ptr_base;
        if (posix_memalign((void**)&ptr_base, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base);

        std::array<uint64_t, TPCH::common::N_FIELDS> field_checksums = {};
        for (size_t i = 0; i < TPCH::common::N_FIELDS; i++) {
            // std::cout << metadata.table_customer_start_page_ids[i] << std::endl;
            size_t nation_start_page_id = metadata.table_nation_start_page_ids[i];
            size_t nation_npages = metadata.table_nation_npages[i];

            if (debug_print) {
                std::cout << metadata.table_nation_start_page_ids[i] << std::endl;
                std::cout
                    << "start_page_id: " << nation_start_page_id
                    << " nation_npages: " << nation_npages
                    << std::endl;
            }

            // Compression metadata
            CompressionMethod compression_method = static_cast<CompressionMethod>(metadata.table_nation_compression_method[i]);
            uint64_t compressed_page_sizes_start_page_id = metadata.table_nation_compressed_page_sizes_start_page_ids[i];
            uint64_t compressed_page_sizes_npages = metadata.table_nation_compressed_page_sizes_npages[i];
            uint64_t compression_nbase = metadata.table_nation_compression_nbases[i];
            uint64_t compression_npages_base = TPCH::nbase_to_npages(compression_nbase, options.page_size);
            uint64_t compressed_base_start_page_ids_val = metadata.table_nation_compression_base_start_page_ids[i];
            uint32_t *compressed_page_sizes_arr = nullptr;
            uint64_t *compressed_base_page_ids = nullptr;
            std::vector<size_t> n_offsets;
            char *compressed_buffer = nullptr;
            char *decomp_workspace = nullptr;
            void *ptr_compressed_page_sizes = nullptr;
            void *ptr_compressed_buffer = nullptr;
            void *ptr_compression_offset_base_page_ids = nullptr;
            if (compression_method != CompressionMethod::NONE) {
                if (posix_memalign((void**)&ptr_compressed_page_sizes, 512, compressed_page_sizes_npages * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&ptr_compression_offset_base_page_ids, 512, compression_npages_base * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < compressed_page_sizes_npages; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compressed_page_sizes) + j * options.page_size,
                        compressed_page_sizes_start_page_id + j, options.page_size);
                }
                for (size_t j = 0; j < compression_npages_base; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compression_offset_base_page_ids) + j * options.page_size,
                        compressed_base_start_page_ids_val + j, options.page_size);
                }
                if (posix_memalign((void**)&ptr_compressed_buffer, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&decomp_workspace, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                compressed_page_sizes_arr = reinterpret_cast<uint32_t*>(ptr_compressed_page_sizes);
                compressed_buffer = reinterpret_cast<char*>(ptr_compressed_buffer);
                compressed_base_page_ids = reinterpret_cast<uint64_t*>(ptr_compression_offset_base_page_ids);

                calculate_compressed_offsets(
                    reinterpret_cast<size_t *>(compressed_base_page_ids),
                    compressed_page_sizes_arr,
                    compression_nbase,
                    nation_npages,
                    options.page_size,
                    nation_start_page_id,
                    options.output_fds.size(),
                    n_offsets
                );
            }

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < nation_npages; j++) {
                const uint64_t nation_pagid = nation_start_page_id + j;

                bool fsst_done = false;
                if (compression_method == CompressionMethod::NONE) {
                    page_pread_host(options.output_fds, (void*)pag, nation_pagid, options.page_size);
                } else {
                    size_t compressed_page_size = compressed_page_sizes_arr[j];
                    size_t read_size = roundup4096(compressed_page_size);
                    uint64_t offset = calc_compressed_page_offset(nation_pagid, n_offsets.data(), nation_start_page_id);
                    read_compressed_page_host(options.output_fds, compressed_buffer, nation_pagid, read_size, options.page_size, offset);
                    switch(compression_method) {
                        case CompressionMethod::SNAPPY:
                        {
                            int ret = decompress_page_with_snappy(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_snappy failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::DEFLATE:
                        {
                            int ret = decompress_page_with_zlib(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_zlib failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4:
                        {
                            int ret = decompress_page_with_lz4(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4PAR:
                        {
                            int ret = decompress_page_with_lz4par(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::FSST:
                        case CompressionMethod::FSST_ROWID:
                        {
                            size_t checksum = 0;
                            uint32_t nrecs = fsst_verify_page(compressed_buffer, compressed_page_size, checksum);
                            field_checksums[i] += checksum;
                            memcpy(pag, compressed_buffer, sizeof(pag_head));
                            fsst_done = true;
                            break;
                        }
                        case CompressionMethod::PFOR:
                        {
                            int ret = decompress_page_with_pfor(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::PFOR64:
                        {
                            int ret = decompress_page_with_pfor64(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor64 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown compression method: " << static_cast<int>(compression_method) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }

                size_t nrecs_in_page = pag_get_nalloc(pag);

                for (size_t l = 0; !fsst_done && l < nrecs_in_page; l++) {
                    // std::cout << "Number of recs in page " << customer_pagid << ": " << nrecs_in_page << std::endl;
                    switch (i) {
                        case TPCH::common::N_NATIONKEY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << col << "|" << std::endl;
                            }
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::N_NAME:
                        {
                            size_t length = TPCH::common::fmt.nation_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            }
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::N_REGIONKEY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << col << "|" << std::endl;
                            }
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::N_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                                std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            }
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown nation field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }

            // Free compression buffers
            if (compression_method != CompressionMethod::NONE) {
                free(ptr_compressed_page_sizes);
                free(ptr_compressed_buffer);
                free(ptr_compression_offset_base_page_ids);
                free(decomp_workspace);
            }

            std::cout << "Total number of recs in nation table: " << sum_nrecs << std::endl;
        }
        std::cout << "Nation table-level checksums: " << std::endl;
        for (size_t i = 0; i < TPCH::common::N_FIELDS; i++) {
            std::cout << "\t" << i << "=" << field_checksums[i] << std::endl;
        }
        free(pag);
    }

    if (options.load_region) {
        /* Memory allocation for scan test */
        void *ptr_base;
        if (posix_memalign((void**)&ptr_base, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base);

        std::array<uint64_t, TPCH::common::R_FIELDS> field_checksums = {};
        for (size_t i = 0; i < TPCH::common::R_FIELDS; i++) {
            // std::cout << metadata.table_customer_start_page_ids[i] << std::endl;
            size_t region_start_page_id = metadata.table_region_start_page_ids[i];
            size_t region_npages = metadata.table_region_npages[i];

            if (debug_print) {
                std::cout << metadata.table_region_start_page_ids[i] << std::endl;
                std::cout << "start_page_id: " << region_start_page_id
                    << " region_npages: " << region_npages
                    << std::endl;
            }

            // Compression metadata
            CompressionMethod compression_method = static_cast<CompressionMethod>(metadata.table_region_compression_method[i]);
            uint64_t compressed_page_sizes_start_page_id = metadata.table_region_compressed_page_sizes_start_page_ids[i];
            uint64_t compressed_page_sizes_npages = metadata.table_region_compressed_page_sizes_npages[i];
            uint64_t compression_nbase = metadata.table_region_compression_nbases[i];
            uint64_t compression_npages_base = TPCH::nbase_to_npages(compression_nbase, options.page_size);
            uint64_t compressed_base_start_page_ids_val = metadata.table_region_compression_base_start_page_ids[i];
            uint32_t *compressed_page_sizes_arr = nullptr;
            uint64_t *compressed_base_page_ids = nullptr;
            std::vector<size_t> r_offsets;
            char *compressed_buffer = nullptr;
            char *decomp_workspace = nullptr;
            void *ptr_compressed_page_sizes = nullptr;
            void *ptr_compressed_buffer = nullptr;
            void *ptr_compression_offset_base_page_ids = nullptr;
            if (compression_method != CompressionMethod::NONE) {
                if (posix_memalign((void**)&ptr_compressed_page_sizes, 512, compressed_page_sizes_npages * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&ptr_compression_offset_base_page_ids, 512, compression_npages_base * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < compressed_page_sizes_npages; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compressed_page_sizes) + j * options.page_size,
                        compressed_page_sizes_start_page_id + j, options.page_size);
                }
                for (size_t j = 0; j < compression_npages_base; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compression_offset_base_page_ids) + j * options.page_size,
                        compressed_base_start_page_ids_val + j, options.page_size);
                }
                if (posix_memalign((void**)&ptr_compressed_buffer, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&decomp_workspace, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                compressed_page_sizes_arr = reinterpret_cast<uint32_t*>(ptr_compressed_page_sizes);
                compressed_buffer = reinterpret_cast<char*>(ptr_compressed_buffer);
                compressed_base_page_ids = reinterpret_cast<uint64_t*>(ptr_compression_offset_base_page_ids);

                calculate_compressed_offsets(
                    reinterpret_cast<size_t *>(compressed_base_page_ids),
                    compressed_page_sizes_arr,
                    compression_nbase,
                    region_npages,
                    options.page_size,
                    region_start_page_id,
                    options.output_fds.size(),
                    r_offsets
                );
            }

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < region_npages; j++) {
                const uint64_t region_page_id = region_start_page_id + j;

                bool fsst_done = false;
                if (compression_method == CompressionMethod::NONE) {
                    page_pread_host(options.output_fds, (void*)pag, region_page_id, options.page_size);
                } else {
                    size_t compressed_page_size = compressed_page_sizes_arr[j];
                    size_t read_size = roundup4096(compressed_page_size);
                    uint64_t offset = calc_compressed_page_offset(region_page_id, r_offsets.data(), region_start_page_id);
                    read_compressed_page_host(options.output_fds, compressed_buffer, region_page_id, read_size, options.page_size, offset);
                    switch(compression_method) {
                        case CompressionMethod::SNAPPY:
                        {
                            int ret = decompress_page_with_snappy(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_snappy failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::DEFLATE:
                        {
                            int ret = decompress_page_with_zlib(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_zlib failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4:
                        {
                            int ret = decompress_page_with_lz4(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4PAR:
                        {
                            int ret = decompress_page_with_lz4par(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4par failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::FSST:
                        case CompressionMethod::FSST_ROWID:
                        {
                            size_t checksum = 0;
                            uint32_t nrecs = fsst_verify_page(compressed_buffer, compressed_page_size, checksum);
                            field_checksums[i] += checksum;
                            memcpy(pag, compressed_buffer, sizeof(pag_head));
                            fsst_done = true;
                            break;
                        }
                        case CompressionMethod::PFOR:
                        {
                            int ret = decompress_page_with_pfor(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::PFOR64:
                        {
                            int ret = decompress_page_with_pfor64(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor64 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown compression method: " << static_cast<int>(compression_method) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }

                size_t nrecs_in_page = pag_get_nalloc(pag);

                for (size_t l = 0; !fsst_done && l < nrecs_in_page; l++) {
                    // std::cout << "Number of recs in page " << customer_pagid << ": " << nrecs_in_page << std::endl;
                    switch (i) {
                        case TPCH::common::R_REGIONKEY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << col << "|" << std::endl;
                            }
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::R_NAME:
                        {
                            size_t length = TPCH::common::fmt.region_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            }
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::R_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                                std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            }
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown region field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }

            // Free compression buffers
            if (compression_method != CompressionMethod::NONE) {
                free(ptr_compressed_page_sizes);
                free(ptr_compressed_buffer);
                free(ptr_compression_offset_base_page_ids);
                free(decomp_workspace);
            }

            std::cout << "Total number of recs in region table: " << sum_nrecs << std::endl;
        }
        std::cout << "Region table-level checksums: " << std::endl;
        for (size_t i = 0; i < TPCH::common::R_FIELDS; i++) {
            std::cout << "\t" << i << "=" << field_checksums[i] << std::endl;
        }
        free(pag);
    }


    #if 1
    if (options.load_customer) {
        /* Memory allocation for scan test */
        void *ptr_base;
        if (posix_memalign((void**)&ptr_base, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base);

        std::array<uint64_t, TPCH::common::C_FIELDS> field_checksums = {};
        for (size_t i = 0; i < TPCH::common::C_FIELDS; i++) {
            size_t customer_start_page_id = metadata.table_customer_start_page_ids[i];
            size_t customer_npages = metadata.table_customer_npages[i];

            if (debug_print) {
                std::cout << metadata.table_customer_start_page_ids[i] << std::endl;
                std::cout << "start_page_id: " << customer_start_page_id
                    << " customer_npages: " << customer_npages
                    << std::endl;
            }

            // Compression metadata
            CompressionMethod compression_method = static_cast<CompressionMethod>(metadata.table_customer_compression_method[i]);
            uint64_t compressed_page_sizes_start_page_id = metadata.table_customer_compressed_page_sizes_start_page_ids[i];
            uint64_t compressed_page_sizes_npages = metadata.table_customer_compressed_page_sizes_npages[i];
            uint64_t compression_nbase = metadata.table_customer_compression_nbases[i];
            uint64_t compression_npages_base = TPCH::nbase_to_npages(compression_nbase, options.page_size);
            uint64_t compressed_base_start_page_ids_val = metadata.table_customer_compression_base_start_page_ids[i];
            uint32_t *compressed_page_sizes_arr = nullptr;
            uint64_t *compressed_base_page_ids = nullptr;
            std::vector<size_t> c_offsets;
            char *compressed_buffer = nullptr;
            char *decomp_workspace = nullptr;
            void *ptr_compressed_page_sizes = nullptr;
            void *ptr_compressed_buffer = nullptr;
            void *ptr_compression_offset_base_page_ids = nullptr;
            if (compression_method != CompressionMethod::NONE) {
                if (posix_memalign((void**)&ptr_compressed_page_sizes, 512, compressed_page_sizes_npages * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&ptr_compression_offset_base_page_ids, 512, compression_npages_base * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < compressed_page_sizes_npages; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compressed_page_sizes) + j * options.page_size,
                        compressed_page_sizes_start_page_id + j, options.page_size);
                }
                for (size_t j = 0; j < compression_npages_base; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compression_offset_base_page_ids) + j * options.page_size,
                        compressed_base_start_page_ids_val + j, options.page_size);
                }
                if (posix_memalign((void**)&ptr_compressed_buffer, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&decomp_workspace, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                compressed_page_sizes_arr = reinterpret_cast<uint32_t*>(ptr_compressed_page_sizes);
                compressed_buffer = reinterpret_cast<char*>(ptr_compressed_buffer);
                compressed_base_page_ids = reinterpret_cast<uint64_t*>(ptr_compression_offset_base_page_ids);

                calculate_compressed_offsets(
                    reinterpret_cast<size_t *>(compressed_base_page_ids),
                    compressed_page_sizes_arr,
                    compression_nbase,
                    customer_npages,
                    options.page_size,
                    customer_start_page_id,
                    options.output_fds.size(),
                    c_offsets
                );
            }

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < customer_npages; j++) {
                const uint64_t customer_page_id = customer_start_page_id + j;

                bool fsst_done = false;
                if (compression_method == CompressionMethod::NONE) {
                    page_pread_host(options.output_fds, (void*)pag, customer_page_id, options.page_size);
                } else {
                    size_t compressed_page_size = compressed_page_sizes_arr[j];
                    size_t read_size = roundup4096(compressed_page_size);
                    uint64_t offset = calc_compressed_page_offset(customer_page_id, c_offsets.data(), customer_start_page_id);
                    read_compressed_page_host(options.output_fds, compressed_buffer, customer_page_id, read_size, options.page_size, offset);
                    switch(compression_method) {
                        case CompressionMethod::SNAPPY:
                        {
                            int ret = decompress_page_with_snappy(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_snappy failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::DEFLATE:
                        {
                            int ret = decompress_page_with_zlib(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_zlib failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4:
                        {
                            int ret = decompress_page_with_lz4(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4PAR:
                        {
                            int ret = decompress_page_with_lz4par(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::FSST:
                        case CompressionMethod::FSST_ROWID:
                        {
                            size_t checksum = 0;
                            uint32_t nrecs = fsst_verify_page(compressed_buffer, compressed_page_size, checksum);
                            field_checksums[i] += checksum;
                            memcpy(pag, compressed_buffer, sizeof(pag_head));
                            fsst_done = true;
                            break;
                        }
                        case CompressionMethod::PFOR:
                        {
                            int ret = decompress_page_with_pfor(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::PFOR64:
                        {
                            int ret = decompress_page_with_pfor64(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor64 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown compression method: " << static_cast<int>(compression_method) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }

                size_t nrecs_in_page = pag_get_nalloc(pag);

                for (size_t l = 0; !fsst_done && l < nrecs_in_page; l++) {
                    switch (i) {
                        case TPCH::common::C_CUSTKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << col << "|" << std::endl;
                            }
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::C_NAME:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                                std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            }
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::C_ADDRESS:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::C_NATIONKEY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << col << "|" << std::endl;
                            }
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::C_PHONE:
                        {
                            size_t length = TPCH::common::fmt.customer_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            }
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::C_ACCTBAL:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << col << "|" << std::endl;
                            }
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::C_MKTSEGMENT:
                        {
                            size_t length = TPCH::common::fmt.customer_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                                std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            }
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::C_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            if (debug_print) {
                                uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                                std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            }
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown customer field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }

            // Free compression buffers
            if (compression_method != CompressionMethod::NONE) {
                free(ptr_compressed_page_sizes);
                free(ptr_compressed_buffer);
                free(ptr_compression_offset_base_page_ids);
                free(decomp_workspace);
            }

            std::cout << "Total number of recs in customer table: " << sum_nrecs << std::endl;
        }
        std::cout << "Customer table-level checksums: " << std::endl;
        for (size_t i = 0; i < TPCH::common::C_FIELDS; i++) {
            std::cout << "\t" << i << "=" << field_checksums[i] << std::endl;
        }

        free(pag);
    }
    #endif

    if (options.load_supplier) {
        /* Memory allocation for scan test */
        void *ptr_base;
        if (posix_memalign((void**)&ptr_base, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base);

        std::array<uint64_t, TPCH::common::S_FIELDS> field_checksums = {};
        for (size_t i = 0; i < TPCH::common::S_FIELDS; i++) {
            size_t supplier_start_page_id = metadata.table_supplier_start_page_ids[i];
            size_t supplier_npages = metadata.table_supplier_npages[i];

            if (debug_print) {
                std::cout << metadata.table_supplier_start_page_ids[i] << std::endl;
                std::cout << "start_page_id: " << supplier_start_page_id
                    << " supplier_npages: " << supplier_npages
                    << std::endl;
            }

            // Compression metadata
            CompressionMethod compression_method = static_cast<CompressionMethod>(metadata.table_supplier_compression_method[i]);
            uint64_t compressed_page_sizes_start_page_id = metadata.table_supplier_compressed_page_sizes_start_page_ids[i];
            uint64_t compressed_page_sizes_npages = metadata.table_supplier_compressed_page_sizes_npages[i];
            uint64_t compression_nbase = metadata.table_supplier_compression_nbases[i];
            uint64_t compression_npages_base = TPCH::nbase_to_npages(compression_nbase, options.page_size);
            uint64_t compressed_base_start_page_ids_val = metadata.table_supplier_compression_base_start_page_ids[i];
            uint32_t *compressed_page_sizes_arr = nullptr;
            uint64_t *compressed_base_page_ids = nullptr;
            std::vector<size_t> s_offsets;
            char *compressed_buffer = nullptr;
            char *decomp_workspace = nullptr;
            void *ptr_compressed_page_sizes = nullptr;
            void *ptr_compressed_buffer = nullptr;
            void *ptr_compression_offset_base_page_ids = nullptr;
            if (compression_method != CompressionMethod::NONE) {
                if (posix_memalign((void**)&ptr_compressed_page_sizes, 512, compressed_page_sizes_npages * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&ptr_compression_offset_base_page_ids, 512, compression_npages_base * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < compressed_page_sizes_npages; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compressed_page_sizes) + j * options.page_size,
                        compressed_page_sizes_start_page_id + j, options.page_size);
                }
                for (size_t j = 0; j < compression_npages_base; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compression_offset_base_page_ids) + j * options.page_size,
                        compressed_base_start_page_ids_val + j, options.page_size);
                }
                if (posix_memalign((void**)&ptr_compressed_buffer, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&decomp_workspace, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                compressed_page_sizes_arr = reinterpret_cast<uint32_t*>(ptr_compressed_page_sizes);
                compressed_buffer = reinterpret_cast<char*>(ptr_compressed_buffer);
                compressed_base_page_ids = reinterpret_cast<uint64_t*>(ptr_compression_offset_base_page_ids);

                calculate_compressed_offsets(
                    reinterpret_cast<size_t *>(compressed_base_page_ids),
                    compressed_page_sizes_arr,
                    compression_nbase,
                    supplier_npages,
                    options.page_size,
                    supplier_start_page_id,
                    options.output_fds.size(),
                    s_offsets
                );
            }

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < supplier_npages; j++) {
                const uint64_t supplier_page_id = supplier_start_page_id + j;

                bool fsst_done = false;
                if (compression_method == CompressionMethod::NONE) {
                    page_pread_host(options.output_fds, (void*)pag, supplier_page_id, options.page_size);
                } else {
                    size_t compressed_page_size = compressed_page_sizes_arr[j];
                    size_t read_size = roundup4096(compressed_page_size);
                    uint64_t offset = calc_compressed_page_offset(supplier_page_id, s_offsets.data(), supplier_start_page_id);
                    read_compressed_page_host(options.output_fds, compressed_buffer, supplier_page_id, read_size, options.page_size, offset);
                    switch(compression_method) {
                        case CompressionMethod::SNAPPY:
                        {
                            int ret = decompress_page_with_snappy(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_snappy failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::DEFLATE:
                        {
                            int ret = decompress_page_with_zlib(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_zlib failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4:
                        {
                            int ret = decompress_page_with_lz4(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4PAR:
                        {
                            int ret = decompress_page_with_lz4par(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::FSST:
                        case CompressionMethod::FSST_ROWID:
                        {
                            size_t checksum = 0;
                            uint32_t nrecs = fsst_verify_page(compressed_buffer, compressed_page_size, checksum);
                            field_checksums[i] += checksum;
                            memcpy(pag, compressed_buffer, sizeof(pag_head));
                            fsst_done = true;
                            break;
                        }
                        case CompressionMethod::PFOR:
                        {
                            int ret = decompress_page_with_pfor(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::PFOR64:
                        {
                            int ret = decompress_page_with_pfor64(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor64 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown compression method: " << static_cast<int>(compression_method) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }

                size_t nrecs_in_page = pag_get_nalloc(pag);

                for (size_t l = 0; !fsst_done && l < nrecs_in_page; l++) {
                    switch (i) {
                        case TPCH::common::S_SUPPKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::S_NAME:
                        {
                            size_t length = TPCH::common::fmt.supplier_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::S_ADDRESS:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::S_NATIONKEY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::S_PHONE:
                        {
                            size_t length = TPCH::common::fmt.supplier_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::S_ACCTBAL:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::S_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown supplier field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }

            // Free compression buffers
            if (compression_method != CompressionMethod::NONE) {
                free(ptr_compressed_page_sizes);
                free(ptr_compressed_buffer);
                free(ptr_compression_offset_base_page_ids);
                free(decomp_workspace);
            }

            std::cout << "Total number of recs in supplier table: " << sum_nrecs << std::endl;
        }
        std::cout << "Supplier table-level checksums: " << std::endl;
        for (size_t i = 0; i < TPCH::common::S_FIELDS; i++) {
            std::cout << "\t" << i << "=" << field_checksums[i] << std::endl;
        }
        free(pag);
    }

    if (options.load_part) {
        /* Memory allocation for scan test */
        void *ptr_base;
        if (posix_memalign((void**)&ptr_base, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base);

        std::array<uint64_t, TPCH::common::P_FIELDS> field_checksums = {};
        for (size_t i = 0; i < TPCH::common::P_FIELDS; i++) {
            size_t part_start_page_id = metadata.table_part_start_page_ids[i];
            size_t part_npages = metadata.table_part_npages[i];

            if (debug_print) {
                std::cout << metadata.table_part_start_page_ids[i] << std::endl;
                std::cout << "start_page_id: " << part_start_page_id
                    << " part_npages: " << part_npages
                    << std::endl;
            }

            // Compression metadata
            CompressionMethod compression_method = static_cast<CompressionMethod>(metadata.table_part_compression_method[i]);
            uint64_t compressed_page_sizes_start_page_id = metadata.table_part_compressed_page_sizes_start_page_ids[i];
            uint64_t compressed_page_sizes_npages = metadata.table_part_compressed_page_sizes_npages[i];
            uint64_t compression_nbase = metadata.table_part_compression_nbases[i];
            uint64_t compression_npages_base = TPCH::nbase_to_npages(compression_nbase, options.page_size);
            uint64_t compressed_base_start_page_ids_val = metadata.table_part_compression_base_start_page_ids[i];
            uint32_t *compressed_page_sizes_arr = nullptr;
            uint64_t *compressed_base_page_ids = nullptr;
            std::vector<size_t> p_offsets;
            char *compressed_buffer = nullptr;
            char *decomp_workspace = nullptr;
            void *ptr_compressed_page_sizes = nullptr;
            void *ptr_compressed_buffer = nullptr;
            void *ptr_compression_offset_base_page_ids = nullptr;
            if (compression_method != CompressionMethod::NONE) {
                if (posix_memalign((void**)&ptr_compressed_page_sizes, 512, compressed_page_sizes_npages * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&ptr_compression_offset_base_page_ids, 512, compression_npages_base * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < compressed_page_sizes_npages; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compressed_page_sizes) + j * options.page_size,
                        compressed_page_sizes_start_page_id + j, options.page_size);
                }
                for (size_t j = 0; j < compression_npages_base; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compression_offset_base_page_ids) + j * options.page_size,
                        compressed_base_start_page_ids_val + j, options.page_size);
                }
                if (posix_memalign((void**)&ptr_compressed_buffer, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&decomp_workspace, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                compressed_page_sizes_arr = reinterpret_cast<uint32_t*>(ptr_compressed_page_sizes);
                compressed_buffer = reinterpret_cast<char*>(ptr_compressed_buffer);
                compressed_base_page_ids = reinterpret_cast<uint64_t*>(ptr_compression_offset_base_page_ids);

                calculate_compressed_offsets(
                    reinterpret_cast<size_t *>(compressed_base_page_ids),
                    compressed_page_sizes_arr,
                    compression_nbase,
                    part_npages,
                    options.page_size,
                    part_start_page_id,
                    options.output_fds.size(),
                    p_offsets
                );
            }

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < part_npages; j++) {
                const uint64_t part_page_id = part_start_page_id + j;

                bool fsst_done = false;
                if (compression_method == CompressionMethod::NONE) {
                    page_pread_host(options.output_fds, (void*)pag, part_page_id, options.page_size);
                } else {
                    size_t compressed_page_size = compressed_page_sizes_arr[j];
                    size_t read_size = roundup4096(compressed_page_size);
                    uint64_t offset = calc_compressed_page_offset(part_page_id, p_offsets.data(), part_start_page_id);
                    read_compressed_page_host(options.output_fds, compressed_buffer, part_page_id, read_size, options.page_size, offset);
                    switch(compression_method) {
                        case CompressionMethod::SNAPPY:
                        {
                            int ret = decompress_page_with_snappy(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_snappy failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::DEFLATE:
                        {
                            int ret = decompress_page_with_zlib(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_zlib failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4:
                        {
                            int ret = decompress_page_with_lz4(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4PAR:
                        {
                            int ret = decompress_page_with_lz4par(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4par failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::FSST:
                        case CompressionMethod::FSST_ROWID:
                        {
                            size_t checksum = 0;
                            uint32_t nrecs = fsst_verify_page(compressed_buffer, compressed_page_size, checksum);
                            field_checksums[i] += checksum;
                            memcpy(pag, compressed_buffer, sizeof(pag_head));
                            fsst_done = true;
                            break;
                        }
                        case CompressionMethod::PFOR:
                        {
                            int ret = decompress_page_with_pfor(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::PFOR64:
                        {
                            int ret = decompress_page_with_pfor64(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor64 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown compression method: " << static_cast<int>(compression_method) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }

                size_t nrecs_in_page = pag_get_nalloc(pag);

                for (size_t l = 0; !fsst_done && l < nrecs_in_page; l++) {
                    switch (i) {
                        case TPCH::common::P_PARTKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::P_NAME:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::P_MFGR:
                        {
                            size_t length = TPCH::common::fmt.part_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::P_BRAND:
                        {
                            size_t length = TPCH::common::fmt.part_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::P_TYPE:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::P_SIZE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::P_CONTAINER:
                        {
                            size_t length = TPCH::common::fmt.part_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::P_RETAILPRICE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::P_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown part field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }

            // Free compression buffers
            if (compression_method != CompressionMethod::NONE) {
                free(ptr_compressed_page_sizes);
                free(ptr_compressed_buffer);
                free(ptr_compression_offset_base_page_ids);
                free(decomp_workspace);
            }

            std::cout << "Total number of recs in part table: " << sum_nrecs << std::endl;
        }
        std::cout << "Part table-level checksums: " << std::endl;
        for (size_t i = 0; i < TPCH::common::P_FIELDS; i++) {
            std::cout << "\t" << i << "=" << field_checksums[i] << std::endl;
        }
        free(pag);
    }

    if (options.load_partsupp) {
        /* Memory allocation for scan test */
        void *ptr_base;
        if (posix_memalign((void**)&ptr_base, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base);

        std::array<uint64_t, TPCH::common::PS_FIELDS> field_checksums = {};
        for (size_t i = 0; i < TPCH::common::PS_FIELDS; i++) {
            size_t partsupp_start_page_id = metadata.table_partsupp_start_page_ids[i];
            size_t partsupp_npages = metadata.table_partsupp_npages[i];

            if (debug_print) {
                std::cout << metadata.table_partsupp_start_page_ids[i] << std::endl;
                std::cout << "start_page_id: " << partsupp_start_page_id
                    << " partsupp_npages: " << partsupp_npages
                    << std::endl;
            }

            // Compression metadata
            CompressionMethod compression_method = static_cast<CompressionMethod>(metadata.table_partsupp_compression_method[i]);
            uint64_t compressed_page_sizes_start_page_id = metadata.table_partsupp_compressed_page_sizes_start_page_ids[i];
            uint64_t compressed_page_sizes_npages = metadata.table_partsupp_compressed_page_sizes_npages[i];
            uint64_t compression_nbase = metadata.table_partsupp_compression_nbases[i];
            uint64_t compression_npages_base = TPCH::nbase_to_npages(compression_nbase, options.page_size);
            uint64_t compressed_base_start_page_ids_val = metadata.table_partsupp_compression_base_start_page_ids[i];
            uint32_t *compressed_page_sizes_arr = nullptr;
            uint64_t *compressed_base_page_ids = nullptr;
            std::vector<size_t> ps_offsets;
            char *compressed_buffer = nullptr;
            char *decomp_workspace = nullptr;
            void *ptr_compressed_page_sizes = nullptr;
            void *ptr_compressed_buffer = nullptr;
            void *ptr_compression_offset_base_page_ids = nullptr;
            if (compression_method != CompressionMethod::NONE) {
                if (posix_memalign((void**)&ptr_compressed_page_sizes, 512, compressed_page_sizes_npages * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&ptr_compression_offset_base_page_ids, 512, compression_npages_base * options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < compressed_page_sizes_npages; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compressed_page_sizes) + j * options.page_size,
                        compressed_page_sizes_start_page_id + j, options.page_size);
                }
                for (size_t j = 0; j < compression_npages_base; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compression_offset_base_page_ids) + j * options.page_size,
                        compressed_base_start_page_ids_val + j, options.page_size);
                }
                if (posix_memalign((void**)&ptr_compressed_buffer, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&decomp_workspace, 512, options.page_size) != 0) {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                compressed_page_sizes_arr = reinterpret_cast<uint32_t*>(ptr_compressed_page_sizes);
                compressed_buffer = reinterpret_cast<char*>(ptr_compressed_buffer);
                compressed_base_page_ids = reinterpret_cast<uint64_t*>(ptr_compression_offset_base_page_ids);

                calculate_compressed_offsets(
                    reinterpret_cast<size_t *>(compressed_base_page_ids),
                    compressed_page_sizes_arr,
                    compression_nbase,
                    partsupp_npages,
                    options.page_size,
                    partsupp_start_page_id,
                    options.output_fds.size(),
                    ps_offsets
                );
            }

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < partsupp_npages; j++) {
                const uint64_t partsupp_page_id = partsupp_start_page_id + j;

                bool fsst_done = false;
                if (compression_method == CompressionMethod::NONE) {
                    page_pread_host(options.output_fds, (void*)pag, partsupp_page_id, options.page_size);
                } else {
                    size_t compressed_page_size = compressed_page_sizes_arr[j];
                    size_t read_size = roundup4096(compressed_page_size);
                    uint64_t offset = calc_compressed_page_offset(partsupp_page_id, ps_offsets.data(), partsupp_start_page_id);
                    read_compressed_page_host(options.output_fds, compressed_buffer, partsupp_page_id, read_size, options.page_size, offset);
                    switch(compression_method) {
                        case CompressionMethod::SNAPPY:
                        {
                            int ret = decompress_page_with_snappy(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_snappy failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::DEFLATE:
                        {
                            int ret = decompress_page_with_zlib(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_zlib failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4:
                        {
                            int ret = decompress_page_with_lz4(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4PAR:
                        {
                            int ret = decompress_page_with_lz4par(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4par failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::FSST:
                        case CompressionMethod::FSST_ROWID:
                        {
                            size_t checksum = 0;
                            uint32_t nrecs = fsst_verify_page(compressed_buffer, compressed_page_size, checksum);
                            field_checksums[i] += checksum;
                            memcpy(pag, compressed_buffer, sizeof(pag_head));
                            fsst_done = true;
                            break;
                        }

                        case CompressionMethod::PFOR:
                        {
                            int ret = decompress_page_with_pfor(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::PFOR64:
                        {
                            int ret = decompress_page_with_pfor64(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor64 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown compression method: " << static_cast<int>(compression_method) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }

                size_t nrecs_in_page = pag_get_nalloc(pag);

                for (size_t l = 0; !fsst_done && l < nrecs_in_page; l++) {
                    switch (i) {
                        case TPCH::common::PS_PARTKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::PS_SUPPKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::PS_AVAILQTY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::PS_SUPPLYCOST:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::PS_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown partsupp field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }

            // Free compression buffers
            if (compression_method != CompressionMethod::NONE) {
                free(ptr_compressed_page_sizes);
                free(ptr_compressed_buffer);
                free(ptr_compression_offset_base_page_ids);
                free(decomp_workspace);
            }

            std::cout << "Total number of recs in partsupp table: " << sum_nrecs << std::endl;
        }
        std::cout << "Partsupp table-level checksums: " << std::endl;
        for (size_t i = 0; i < TPCH::common::PS_FIELDS; i++) {
            std::cout << "\t" << i << "=" << field_checksums[i] << std::endl;
        }
        free(pag);
    }

    if (options.load_orders) {
        /* Memory allocation for scan test */
        void *ptr_base_alloc;
        if (posix_memalign((void**)&ptr_base_alloc, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base_alloc);

        /* checksums are initialized with 0 */
        std::array<uint64_t, TPCH::common::O_FIELDS> field_checksums = {};

        size_t q3_orders_total_npages_to_be_read = 0;
        size_t q3_orders_total_npages = 0;
        size_t q3_orders_page_read_bytes = 0;
        size_t q5_orders_total_npages_to_be_read = 0;
        size_t q5_orders_total_npages = 0;
        size_t q5_orders_page_read_bytes = 0;
        size_t q10_orders_total_npages_to_be_read = 0;
        size_t q10_orders_total_npages = 0;
        size_t q10_orders_page_read_bytes = 0;
        for (size_t i = 0; i < TPCH::common::O_FIELDS; i++) {
            if (debug_print) std::cout << metadata.table_orders_start_page_ids[i] << std::endl;
            size_t orders_start_page_id = metadata.table_orders_start_page_ids[i];
            size_t orders_npages = metadata.table_orders_npages[i];

            size_t orders_prefix_sum_start_page_id = metadata.table_orders_prefix_sum_start_page_ids[i];
            size_t orders_prefix_sum_npages = metadata.table_orders_prefix_sum_npages[i];

            size_t orders_stats_start_page_id = metadata.table_orders_stats_start_page_ids[i];
            size_t orders_stats_npages = metadata.table_orders_stats_npages[i];
            size_t orders_nstats = metadata.table_orders_nstats[i];

            const bool enable_stats = (orders_nstats > 0) ? true : false;

            void *ptr_prefix_sum_alloc = nullptr;
            if (posix_memalign((void**)&ptr_prefix_sum_alloc, 512, orders_prefix_sum_npages * options.page_size) != 0)
            {
                std::cerr << "posix_memalign failed" << std::endl;
                exit(EXIT_FAILURE);
            }
            uint64_t *prefix_sum_nrecs = reinterpret_cast<uint64_t*>(ptr_prefix_sum_alloc);
            for (size_t j = 0; j < orders_prefix_sum_npages; j++) {
                const uint64_t orders_prefix_sum_page_id = orders_prefix_sum_start_page_id + j;
                page_pread_host(options.output_fds, reinterpret_cast<char*>(prefix_sum_nrecs) + j * options.page_size,
                    orders_prefix_sum_page_id, options.page_size);
            }

            void *ptr_stats_alloc = nullptr;
            std::span<Stats<DateType>> stats;
            if (enable_stats) {
                if (posix_memalign((void**)&ptr_stats_alloc, 512, orders_stats_npages * options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < orders_stats_npages; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_stats_alloc) + j * options.page_size,
                        orders_stats_start_page_id + j, options.page_size);
                }
                Stats<DateType> *statsp = reinterpret_cast<Stats<DateType>*>(ptr_stats_alloc);
                stats = std::span<Stats<DateType>>(statsp, orders_nstats);
            }

            auto& arr_orders_sideways_stats_start_page_id = metadata.table_orders_sideways_stats_start_page_ids[i];
            auto& arr_orders_sideways_stats_npages = metadata.table_orders_sideways_stats_npages[i];
            auto& arr_orders_sideways_nstats = metadata.table_orders_sideways_nstats[i];
            std::array<void *, TPCH::common::kOrdersSidewaysCount> ptr_sideways_stats_alloc = {nullptr};
            std::array<std::span<Stats<int32_t>>, TPCH::common::kOrdersSidewaysCount> sideways_stats;
            for (size_t l = 0; l < TPCH::common::kOrdersSidewaysCount; l++) {
                if (posix_memalign((void**)&ptr_sideways_stats_alloc[l], 512, arr_orders_sideways_stats_npages[l] * options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < arr_orders_sideways_stats_npages[l]; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_sideways_stats_alloc[l]) + j * options.page_size,
                        arr_orders_sideways_stats_start_page_id[l] + j, options.page_size);
                }
                Stats<int32_t> *statsp = reinterpret_cast<Stats<int32_t>*>(ptr_sideways_stats_alloc[l]);
                sideways_stats[l] = std::span<Stats<int32_t>>(statsp, arr_orders_sideways_nstats[l]);
            }

            uint64_t compressed_page_sizes_start_page_id = metadata.table_orders_compressed_page_sizes_start_page_ids[i];
            uint64_t compressed_page_sizes_npages = metadata.table_orders_compressed_page_sizes_npages[i];
            uint64_t compression_nbase = metadata.table_orders_compression_nbases[i];
            uint64_t compression_npages_base = TPCH::nbase_to_npages(compression_nbase, options.page_size);
            uint64_t compressed_base_start_page_ids = metadata.table_orders_compression_base_start_page_ids[i];
            CompressionMethod compression_method = static_cast<CompressionMethod>(metadata.table_orders_compression_method[i]);
            uint32_t *compressed_page_sizes = nullptr;
            uint64_t *compressed_base_page_ids = nullptr;
            std::vector<size_t> o_offsets;
            char *compressed_buffer = nullptr;
            char *decomp_workspace = nullptr;
            void *ptr_compressed_page_sizes = nullptr;
            void *ptr_compressed_buffer = nullptr;
            void *ptr_compression_offset_base_page_ids = nullptr;
            if (compression_method == CompressionMethod::NONE) {
                // nothing to do
            } else {
                if (posix_memalign((void**)&ptr_compressed_page_sizes, 512, compressed_page_sizes_npages * options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&ptr_compression_offset_base_page_ids, 512, options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < compressed_page_sizes_npages; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compressed_page_sizes) + j * options.page_size,
                        compressed_page_sizes_start_page_id + j, options.page_size);
                }
                for (size_t j = 0; j < compression_npages_base; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compression_offset_base_page_ids) + j * options.page_size,
                        compressed_base_start_page_ids + j, options.page_size);
                }

                if (posix_memalign((void**)&ptr_compressed_buffer, 512, options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&decomp_workspace, 512, options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                compressed_page_sizes = reinterpret_cast<uint32_t*>(ptr_compressed_page_sizes);
                compressed_buffer = reinterpret_cast<char*>(ptr_compressed_buffer);
                compressed_base_page_ids = reinterpret_cast<uint64_t*>(ptr_compression_offset_base_page_ids);
            }

            if (debug_print) {
                std::cout << "FIELD INDEX: " << i
                    << " start_page_id: " << orders_start_page_id
                    << " orders_npages: " << orders_npages
                    << std::endl;
            }

            if (prefix_sum_nrecs[0] != 0) {
                std::cerr << "Error: prefix_sum_nrecs[0] is expected to be 0, but got " << prefix_sum_nrecs[0] << std::endl;
                exit(EXIT_FAILURE);
            }

            if (compression_method != CompressionMethod::NONE) {
                calculate_compressed_offsets(
                    reinterpret_cast<size_t *>(compressed_base_page_ids),
                    compressed_page_sizes,
                    compression_nbase,
                    orders_npages,
                    options.page_size,
                    orders_start_page_id,
                    options.output_fds.size(),
                    o_offsets
                );
            }

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < orders_npages; j++) {
                const uint64_t orders_page_id = orders_start_page_id + j;

                /* Q3: O_ORDERDATE < 1995-03-15 AND C_MKTSEGMENT = BUILDING */
                if (i == TPCH::common::O_ORDERKEY ||
                    i == TPCH::common::O_CUSTKEY ||
                    i == TPCH::common::O_TOTALPRICE ||
                    i == TPCH::common::O_SHIPPRIORITY) {
                    auto mktsegment_stats = sideways_stats[1];
                    Stats<int32_t> c = mktsegment_stats[j];
                    if (enable_stats) {
                        Stats<DateType> o = stats[j];
                        if (c.overlaps(1) && o.overlaps({0}, {19950314})) {
                            q3_orders_total_npages_to_be_read++;
                            if (compression_method != CompressionMethod::NONE) {
                                q3_orders_page_read_bytes += roundup4096(compressed_page_sizes[j]);
                            } else {
                                q3_orders_page_read_bytes += options.page_size;
                            }
                        }
                    } else {
                        q3_orders_total_npages_to_be_read++;
                        if (compression_method != CompressionMethod::NONE) {
                            q3_orders_page_read_bytes += roundup4096(compressed_page_sizes[j]);
                        } else {
                            q3_orders_page_read_bytes += options.page_size;
                        }
                    }
                    q3_orders_total_npages++;
                }
                if (i == TPCH::common::O_ORDERDATE) {
                    if (enable_stats) {
                        Stats<DateType> o = stats[j];
                        if (o.overlaps({0}, {19950314})) {
                            q3_orders_total_npages_to_be_read++;
                            if (compression_method != CompressionMethod::NONE) {
                                q3_orders_page_read_bytes += roundup4096(compressed_page_sizes[j]);
                            } else {
                                q3_orders_page_read_bytes += options.page_size;
                            }
                        }
                    } else {
                        q3_orders_total_npages_to_be_read++;
                        if (compression_method != CompressionMethod::NONE) {
                            q3_orders_page_read_bytes += roundup4096(compressed_page_sizes[j]);
                        } else {
                            q3_orders_page_read_bytes += options.page_size;
                        }
                    }
                    q3_orders_total_npages++;
                }

                /* Q5: O_ORDERDATE in [1994-01-01, 1995-01-01) AND R_NAME = ASIA */
                if (i == TPCH::common::O_ORDERKEY ||
                    i == TPCH::common::O_CUSTKEY) {
                    auto region_stats = sideways_stats[0];
                    Stats<int32_t> r = region_stats[j];
                    if (enable_stats) {
                        Stats<DateType> o = stats[j];
                        if (r.overlaps(2) && o.overlaps({19940101}, {19950100})) {
                            q5_orders_total_npages_to_be_read++;
                            if (compression_method != CompressionMethod::NONE) {
                                q5_orders_page_read_bytes += roundup4096(compressed_page_sizes[j]);
                            } else {
                                q5_orders_page_read_bytes += options.page_size;
                            }
                        }
                    } else {
                        q5_orders_total_npages_to_be_read++;
                        if (compression_method != CompressionMethod::NONE) {
                            q5_orders_page_read_bytes += roundup4096(compressed_page_sizes[j]);
                        } else {
                            q5_orders_page_read_bytes += options.page_size;
                        }
                    }
                    q5_orders_total_npages++;
                }

                /* Q10: O_ORDERDATE in [1993-10-01, 1994-01-01) */
                if (i == TPCH::common::O_ORDERKEY ||
                    i == TPCH::common::O_CUSTKEY) {
                    if (enable_stats) {
                        Stats<DateType> o = stats[j];
                        if (o.overlaps({19931001}, {19940101})) {
                            q10_orders_total_npages_to_be_read++;
                            if (compression_method != CompressionMethod::NONE) {
                                q10_orders_page_read_bytes += roundup4096(compressed_page_sizes[j]);
                            } else {
                                q10_orders_page_read_bytes += options.page_size;
                            }
                        }
                    } else {
                        q10_orders_total_npages_to_be_read++;
                        if (compression_method != CompressionMethod::NONE) {
                            q10_orders_page_read_bytes += roundup4096(compressed_page_sizes[j]);
                        } else {
                            q10_orders_page_read_bytes += options.page_size;
                        }
                    }
                    q10_orders_total_npages++;
                }

                bool fsst_done = false;
                if (compression_method == CompressionMethod::NONE) {
                    page_pread_host(options.output_fds, (void*)pag, orders_page_id, options.page_size);
                } else {
                    size_t compressed_page_size = compressed_page_sizes[j];
                    size_t read_size = roundup4096(compressed_page_size);
                    uint64_t offset = calc_compressed_page_offset(orders_page_id, o_offsets.data(), orders_start_page_id);
                    read_compressed_page_host(options.output_fds, compressed_buffer, orders_page_id, read_size, options.page_size, offset);
                    switch(compression_method) {
                        case CompressionMethod::SNAPPY:
                        {
                            int ret = decompress_page_with_snappy(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_snappy failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::DEFLATE:
                        {
                            int ret = decompress_page_with_zlib(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_zlib failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4:
                        {
                            int ret = decompress_page_with_lz4(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::LZ4PAR:
                        {
                            int ret = decompress_page_with_lz4par(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4par failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::FSST:
                        case CompressionMethod::FSST_ROWID:
                        {
                            size_t checksum = 0;
                            uint32_t nrecs = fsst_verify_page(compressed_buffer, compressed_page_size, checksum);
                            field_checksums[i] += checksum;
                            memcpy(pag, compressed_buffer, sizeof(pag_head));
                            fsst_done = true;
                            break;
                        }

                        case CompressionMethod::PFOR:
                        {
                            int ret = decompress_page_with_pfor(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::PFOR64:
                        {
                            int ret = decompress_page_with_pfor64(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_pfor64 failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown compression method: " << static_cast<int>(compression_method) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }

                size_t nrecs_in_page = pag_get_nalloc(pag);
                size_t base_row_id = prefix_sum_nrecs[j];

                if (prefix_sum_nrecs[j + 1] - prefix_sum_nrecs[j] != nrecs_in_page) {
                    std::cerr << "Error(" << __FILE__ << ":" << __LINE__ << "): prefix sum for field "
                        << i
                        << " page_id: " << orders_page_id
                        << " prefix sum diff: " << (prefix_sum_nrecs[j + 1] - prefix_sum_nrecs[j])
                        << " nrecs_in_page: " << nrecs_in_page
                        << std::endl;
                    exit(EXIT_FAILURE);
                }

                for (size_t l = 0; !fsst_done && l < nrecs_in_page; l++) {
                    bool enable_stdout = debug_print && (j == 0 && l <= 10);

                    switch (i) {
                        case TPCH::common::O_ORDERKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << (base_row_id + l) << "|" << col << "|" << std::endl;
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::O_CUSTKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << (base_row_id + l) << "|" << col << "|" << std::endl;
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::O_ORDERSTATUS:
                        {
                            size_t length = TPCH::common::fmt.orders_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            if (enable_stdout) std::cout << (base_row_id + l) << "|" << std::string_view(col, length) << "|" << std::endl;
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::O_TOTALPRICE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << (base_row_id + l) << "|" << col << "|" << std::endl;
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::O_ORDERDATE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << (base_row_id + l) << "|" << col << "|" << std::endl;
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::O_ORDERPRIORITY:
                        {
                            size_t length = TPCH::common::fmt.orders_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            if (enable_stdout) std::cout << (base_row_id + l) << "|" << std::string_view(col, length) << "|" << std::endl;
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::O_CLERK:
                        {
                            size_t length = TPCH::common::fmt.orders_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            if (enable_stdout) std::cout << (base_row_id + l) << "|" << std::string_view(col, length) << "|" << std::endl;
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::O_SHIPPRIORITY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << (base_row_id + l) << "|" << col << "|" << std::endl;
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::O_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            if (enable_stdout) std::cout << (base_row_id + l) << "|" << std::string_view(col, length) << "|" << std::endl;
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown orders field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }
            std::cout << "Total number of recs in orders table: " << sum_nrecs << std::endl;

            if (compression_method != CompressionMethod::NONE) {
                free(ptr_compressed_page_sizes);
                free(ptr_compressed_buffer);
            }

            for (size_t l = 0; l < TPCH::common::kOrdersSidewaysCount; l++) {
                free(ptr_sideways_stats_alloc[l]);
            }
            if (enable_stats) {
                free(ptr_stats_alloc);
            }
            free(ptr_prefix_sum_alloc);
        }

        double q3_orders_compression_ratio = q3_orders_page_read_bytes / (double)(options.page_size * q3_orders_total_npages_to_be_read);
        double q3_orders_pruning_ratio = (double)q3_orders_total_npages_to_be_read / (double)q3_orders_total_npages;
        std::cout << "Orders: Total number of pages to be read for Q3 with IO pruning and compression: "
            << q3_orders_compression_ratio * q3_orders_pruning_ratio * q3_orders_total_npages * options.page_size / (1024 * 1024) << " MB ("
            << "original " << q3_orders_total_npages * options.page_size / (1024 * 1024) << " MB, pruned and compressed "
            << q3_orders_compression_ratio * q3_orders_pruning_ratio
            << "), compression ratio:"
            << q3_orders_compression_ratio
            << ", pruning ratio:"
            << q3_orders_pruning_ratio
            << " ("
            << q3_orders_total_npages_to_be_read << " / " << q3_orders_total_npages
            << ")" << std::endl;

        double q5_orders_compression_ratio = q5_orders_page_read_bytes / (double)(options.page_size * q5_orders_total_npages_to_be_read);
        double q5_orders_pruning_ratio = (double)q5_orders_total_npages_to_be_read / (double)q5_orders_total_npages;
        std::cout << "Orders: Total number of pages to be read for Q5 with IO pruning and compression: "
            << q5_orders_compression_ratio * q5_orders_pruning_ratio * q5_orders_total_npages * options.page_size / (1024 * 1024) << " MB ("
            << "original " << q5_orders_total_npages * options.page_size / (1024 * 1024) << " MB, pruned and compressed "
            << q5_orders_compression_ratio * q5_orders_pruning_ratio
            << "), compression ratio:"
            << q5_orders_compression_ratio
            << ", pruning ratio:"
            << q5_orders_pruning_ratio
            << " ("
            << q5_orders_total_npages_to_be_read << " / " << q5_orders_total_npages
            << ")" << std::endl;

        double q10_orders_compression_ratio = q10_orders_page_read_bytes / (double)(options.page_size * q10_orders_total_npages_to_be_read);
        double q10_orders_pruning_ratio = (double)q10_orders_total_npages_to_be_read / (double)q10_orders_total_npages;
        std::cout << "Orders: Total number of pages to be read for Q10 with IO pruning and compression: "
            << q10_orders_compression_ratio * q10_orders_pruning_ratio * q10_orders_total_npages * options.page_size / (1024 * 1024) << " MB ("
            << "original " << q10_orders_total_npages * options.page_size / (1024 * 1024) << " MB, pruned and compressed "
            << q10_orders_compression_ratio * q10_orders_pruning_ratio
            << "), compression ratio:"
            << q10_orders_compression_ratio
            << ", pruning ratio:"
            << q10_orders_pruning_ratio
            << " ("
            << q10_orders_total_npages_to_be_read << " / " << q10_orders_total_npages
            << ")" << std::endl;

        std::cout << "Orders table-level checksums: " << std::endl;
        for (size_t i = 0; i < TPCH::common::O_FIELDS; i++) {
            std::cout << "\t" << i << "=" << field_checksums[i] << std::endl;
        }

        free(ptr_base_alloc);
    }

    if (options.load_lineitem) {
        /* Memory allocation for scan test */
        void *ptr_base_alloc;
        if (posix_memalign((void**)&ptr_base_alloc, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base_alloc);

        /* checksums are initialized with 0 */
        std::array<uint64_t, TPCH::common::L_FIELDS> field_checksums = {};

        size_t q1_total_npages_to_be_read = 0;
        size_t q1_total_npages = 0;
        size_t q1_page_read_bytes = 0;
        size_t q3_total_npages_to_be_read = 0;
        size_t q3_total_npages = 0;
        size_t q3_page_read_bytes = 0;
        size_t q5_total_npages_to_be_read = 0;
        size_t q5_total_npages = 0;
        size_t q5_page_read_bytes = 0;
        size_t q6_total_npages_to_be_read = 0;
        size_t q6_total_npages = 0;
        size_t q6_page_read_bytes = 0;
        size_t q10_total_npages_to_be_read = 0;
        size_t q10_total_npages = 0;
        size_t q10_page_read_bytes = 0;
        for (size_t i = 0; i < TPCH::common::L_FIELDS; i++) {
            // std::cout << metadata.table_customer_start_page_ids[i] << std::endl;
            std::cout << metadata.table_lineitem_start_page_ids[i] << std::endl;
            size_t lineitem_start_page_id = metadata.table_lineitem_start_page_ids[i];
            size_t lineitem_npages = metadata.table_lineitem_npages[i];

            size_t lineitem_prefix_sum_start_page_id = metadata.table_lineitem_prefix_sum_start_page_ids[i];
            size_t lineitem_prefix_sum_npages = metadata.table_lineitem_prefix_sum_npages[i];

            size_t lineitem_stats_start_page_id = metadata.table_lineitem_stats_start_page_ids[i];
            size_t lineitem_stats_npages = metadata.table_lineitem_stats_npages[i];
            size_t lineitem_nstats = metadata.table_lineitem_nstats[i];

            const bool enable_stats = (lineitem_nstats > 0) ? true : false;

            void *ptr_prefix_sum_alloc = nullptr;
            if (posix_memalign((void**)&ptr_prefix_sum_alloc, 512, lineitem_prefix_sum_npages * options.page_size) != 0)
            {
                std::cerr << "posix_memalign failed" << std::endl;
                exit(EXIT_FAILURE);
            }
            uint64_t *prefix_sum_nrecs = reinterpret_cast<uint64_t*>(ptr_prefix_sum_alloc);
            for (size_t j = 0; j < lineitem_prefix_sum_npages; j++) {
                const uint64_t lineitem_prefix_sum_page_id = lineitem_prefix_sum_start_page_id + j;
                page_pread_host(options.output_fds, reinterpret_cast<char*>(prefix_sum_nrecs) + j * options.page_size,
                    lineitem_prefix_sum_page_id, options.page_size);
            }

            void *ptr_stats_alloc = nullptr;
            std::span<Stats<DateType>> stats;
            if (enable_stats) {
                if (posix_memalign((void**)&ptr_stats_alloc, 512, lineitem_stats_npages * options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < lineitem_stats_npages; j++) {
                    const uint64_t lineitem_stats_page_id = lineitem_stats_start_page_id + j;
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_stats_alloc) + j * options.page_size,
                        lineitem_stats_page_id, options.page_size);
                }
                Stats<DateType> *statsp = reinterpret_cast<Stats<DateType>*>(ptr_stats_alloc);
#if 0
                for (size_t j = 0; j < lineitem_nstats; j++) {
                    std::cout << "Stats[" << j << "] min: " << statsp[j].min_val << " max: " << statsp[j].max_val << std::endl;
                }
#endif
                stats = std::span<Stats<DateType>>(statsp, lineitem_nstats);
#if 0
                for (size_t j = 0; j < lineitem_nstats; j++) {
                    std::cout << "Stats[" << j << "] min: " << stats[j].min_val << " max: " << statsp[j].max_val << std::endl;
                }
#endif
                // exit(1);
            }

            auto& arr_lineitem_sideways_stats_start_page_id = metadata.table_lineitem_sideways_stats_start_page_ids[i];
            auto& arr_lineitem_sideways_stats_npages = metadata.table_lineitem_sideways_stats_npages[i];
            auto& arr_lineitem_sideways_nstats = metadata.table_lineitem_sideways_nstats[i];
            std::array<void *, TPCH::common::kLineitemSidewaysCount> ptr_sideways_stats_alloc = {nullptr};
            std::array<std::span<Stats<int32_t>>, TPCH::common::kLineitemSidewaysCount> sideways_stats;
            for (size_t l = 0; l < TPCH::common::kLineitemSidewaysCount; l++) {
                if (posix_memalign((void**)&ptr_sideways_stats_alloc[l], 512, arr_lineitem_sideways_stats_npages[l] * options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < arr_lineitem_sideways_stats_npages[l]; j++) {
                    const uint64_t lineitem_stats_page_id = lineitem_stats_start_page_id + j;
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_sideways_stats_alloc[l]) + j * options.page_size,
                        arr_lineitem_sideways_stats_start_page_id[l] + j, options.page_size);
                }
                Stats<int32_t> *statsp = reinterpret_cast<Stats<int32_t>*>(ptr_sideways_stats_alloc[l]);
#if 0
                for (size_t j = 0; j < arr_lineitem_sideways_nstats[l]; j++) {
                    std::cout << "Stats[" << j << "] min: " << statsp[j].min_val << " max: " << statsp[j].max_val << std::endl;
                }
#endif
                sideways_stats[l] = std::span<Stats<int32_t>>(statsp, arr_lineitem_sideways_nstats[l]);
#if 0
                for (size_t j = 0; j < arr_lineitem_sideways_nstats[l]; j++) {
                    std::cout << "Stats[" << j << "] min: " << sideways_stats[l][j].min_val << " max: " << sideways_stats[l][j].max_val << std::endl;
                }
#endif
            }

            uint64_t compressed_page_sizes_start_page_id = metadata.table_lineitem_compressed_page_sizes_start_page_ids[i];
            uint64_t compressed_page_sizes_npages = metadata.table_lineitem_compressed_page_sizes_npages[i];
            uint64_t compression_nbase = metadata.table_lineitem_compression_nbases[i];
            uint64_t compression_npages_base = TPCH::nbase_to_npages(compression_nbase, options.page_size);
            uint64_t compressed_base_start_page_ids = metadata.table_lineitem_compression_base_start_page_ids[i];
            CompressionMethod compression_method = static_cast<CompressionMethod>(metadata.table_lineitem_compression_method[i]);
            uint32_t *compressed_page_sizes = nullptr;
            uint64_t *compressed_base_page_ids = nullptr;
            std::vector<size_t> l_offsets;
            char *compressed_buffer = nullptr;
            char *decomp_workspace = nullptr;
            void *ptr_compressed_page_sizes = nullptr;
            void *ptr_compressed_buffer = nullptr;
            void *ptr_compression_offset_base_page_ids = nullptr;
            if (compression_method == CompressionMethod::NONE) {
                // noting to do
            } else {
                if (posix_memalign((void**)&ptr_compressed_page_sizes, 512, compressed_page_sizes_npages * options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < compressed_page_sizes_npages; j++) {
                    const uint64_t lineitem_stats_page_id = lineitem_stats_start_page_id + j;
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compressed_page_sizes) + j * options.page_size,
                        compressed_page_sizes_start_page_id + j, options.page_size);
                }
                if (posix_memalign((void**)&ptr_compression_offset_base_page_ids, 512, options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < compressed_page_sizes_npages; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compressed_page_sizes) + j * options.page_size,
                        compressed_page_sizes_start_page_id + j, options.page_size);
                }
                for (size_t j = 0; j < compression_npages_base; j++) {
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_compression_offset_base_page_ids) + j * options.page_size,
                        compressed_base_start_page_ids + j, options.page_size);
                }

                if (posix_memalign((void**)&ptr_compressed_buffer, 512, options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (posix_memalign((void**)&decomp_workspace, 512, options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                decomp_workspace = reinterpret_cast<char*>(decomp_workspace);
                compressed_page_sizes = reinterpret_cast<uint32_t*>(ptr_compressed_page_sizes);
                compressed_buffer = reinterpret_cast<char*>(ptr_compressed_buffer);
                compressed_base_page_ids = reinterpret_cast<uint64_t*>(ptr_compression_offset_base_page_ids);

                if (posix_memalign((void**)&decomp_workspace, 512, options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }


            if (debug_print) {
                std::cout << "FIELD INDEX: " << i
                    << " start_page_id: " << lineitem_start_page_id
                    << " lineitem_npages: " << lineitem_npages
                    << std::endl;
            }

            if (prefix_sum_nrecs[0] != 0) {
                std::cerr << "Error: prefix_sum_nrecs[0] is expected to be 0, but got " << prefix_sum_nrecs[0] << std::endl;
                exit(EXIT_FAILURE);
            }

            if (compression_method != CompressionMethod::NONE) {
                // Pre-decompress all pages to measure compressed page sizes
                //l_offsets.resize(lineitem_npages + 1);
                //l_offsets[0] = 0;
                //size_t k = 0;
                //if (lineitem_start_page_id [k])
#if 1
                calculate_compressed_offsets(
                    reinterpret_cast<size_t *>(compressed_base_page_ids),
                    compressed_page_sizes,
                    compression_nbase,
                    lineitem_npages,
                    options.page_size,
                    lineitem_start_page_id,
                    options.output_fds.size(),
                    l_offsets
                );
#else
                size_t npages_sum = 0;
                std::cout << "compression_nbase: " << compression_nbase << std::endl;
                for (size_t j = 0; j < compression_nbase; ++j) {
                    size_t npages_segument = 0;
                    if (j < compression_nbase - 1) {
                        npages_segument = compressed_base_page_ids[j + 1] - compressed_base_page_ids[j];
                    } else {
                        npages_segument = lineitem_npages - npages_sum;
                    }
                    std::cout << "compressed_base_page_ids[" << j << "]: " << compressed_base_page_ids[j]
                        << " compressed_base_page_ids[" << j + 1 << "]: " << compressed_base_page_ids[j + 1]
                        << " npages_segument: " << npages_segument << std::endl;

                    l_offsets[0 + npages_sum] = 0;
                    for (size_t k = 0 + npages_sum; k < npages_segument + npages_sum; k++) {
                        if (k == npages_sum) {
                            /* begining of segment */
                            l_offsets[k] = compressed_base_page_ids[j] * options.page_size;
                        }
                        size_t compressed_page_size = compressed_page_sizes[k];
                        l_offsets[k + 1] = l_offsets[k] + roundup4096(compressed_page_size);
                        std::cout << "l_offsets[" << k << "]: " << l_offsets[k]
                            << " l_offsets[" << k + 1 << "]: " << l_offsets[k + 1]
                            << std::endl;
                    }

                    npages_sum += npages_segument;
                }
                std::cout << "npages_sum: " << npages_sum << std::endl;
#endif
                std::cout << "lineitem_npages: " << lineitem_npages << std::endl;
                //exit(1);
            }

            size_t sum_nrecs = 0;
            size_t total_npages_shipdate = (i == TPCH::common::L_SHIPDATE) ? lineitem_npages : 0;
            size_t shipdate_base_page_id = 0;
            for (size_t j = 0; j < lineitem_npages; j++) {
                const uint64_t lineitem_page_id = lineitem_start_page_id + j;

                /* Start scan test */
                // SELECT date '1998-12-01' - interval '90' day AS result;
                // -> 1998-09-02
                if (i == TPCH::common::L_RETURNFLAG ||
                    i == TPCH::common::L_LINESTATUS ||
                    i == TPCH::common::L_QUANTITY ||
                    i == TPCH::common::L_EXTENDEDPRICE ||
                    i == TPCH::common::L_DISCOUNT) {
                    if (i == TPCH::common::L_SHIPDATE) {
                        Stats<DateType> l = stats[j];
                        shipdate_base_page_id = lineitem_start_page_id;
                        if (l.overlaps({0}, {19980902})) {
                            q1_total_npages_to_be_read++;
                            if (compression_method != CompressionMethod::NONE) {
                                size_t compressed_page_size = compressed_page_sizes[j];
                                size_t read_size = roundup4096(compressed_page_size);
                                q1_page_read_bytes += read_size;
                            } else {
                                q1_page_read_bytes += options.page_size;
                            }
                        }
                    } else {
                        q1_total_npages_to_be_read++;
                        if (compression_method != CompressionMethod::NONE) {
                            size_t compressed_page_size = compressed_page_sizes[j];
                            size_t read_size = roundup4096(compressed_page_size);
                            q1_page_read_bytes += read_size;
                        } else {
                            q1_page_read_bytes += options.page_size;
                        }
                    }
                    q1_total_npages++;
                }


                /* Emulating TPC-H Q3's IO pruning */
                if (i == TPCH::common::L_ORDERKEY ||
                    i == TPCH::common::L_EXTENDEDPRICE ||
                    i == TPCH::common::L_DISCOUNT) {
                    auto mktsegment_stats = sideways_stats[1];
                    auto orderdate_stats = sideways_stats[2];
                    Stats<int32_t> c = mktsegment_stats[j];
                    Stats<int32_t> o = orderdate_stats[j];
                    if (c.overlaps(1) && o.overlaps({0}, {19950314})) {
#if 0
                        std::cout << "Page " << lineitem_page_id << " overlaps with [1994-01-01, 1995-01-01)\n"
                            << " min_region: " << c.min_val << " max_region: " << c.max_val
                            << " min_shipdate: " << o.min_val << " max_shipdate: " << o.max_val << std::endl;
#endif

                        /* NOTE: require IOs */
                        q3_total_npages_to_be_read++;
                        if (compression_method != CompressionMethod::NONE) {
                            size_t compressed_page_size = compressed_page_sizes[j];
                            size_t read_size = roundup4096(compressed_page_size);
                            q3_page_read_bytes += read_size;
                        } else {
                            q3_page_read_bytes += options.page_size;
                        }
                    }
                    q3_total_npages++;
                }

                /* Emulating TPC-H Q5's L_SHIPDATE */
                if (i == TPCH::common::L_ORDERKEY ||
                    i == TPCH::common::L_EXTENDEDPRICE ||
                    i == TPCH::common::L_DISCOUNT ||
                    i == TPCH::common::L_SUPPKEY) {
                    auto region_stats = sideways_stats[0];
                    auto orderdate_stats = sideways_stats[2];
                    Stats<int32_t> r = region_stats[j];
                    Stats<int32_t> o = orderdate_stats[j];
                    if (r.overlaps(2) && o.overlaps({19940101}, {19950100})) {
#if 0
                        std::cout << "Page " << lineitem_page_id << " overlaps with [1994-01-01, 1995-01-01)\n"
                            << " min_region: " << r.min_val << " max_region: " << r.max_val
                            << " min_shipdate: " << o.min_val << " max_shipdate: " << o.max_val << std::endl;
#endif

                        /* NOTE: require IOs */
                        q5_total_npages_to_be_read++;
                        if (compression_method != CompressionMethod::NONE) {
                            size_t compressed_page_size = compressed_page_sizes[j];
                            size_t read_size = roundup4096(compressed_page_size);
                            q5_page_read_bytes += read_size;
                        } else {
                            q5_page_read_bytes += options.page_size;
                        }
                    }
                    q5_total_npages++;
                }

                /* Emulating TPC-H Q6's L_SHIPDATE */
                if (i == TPCH::common::L_SHIPDATE ||
                    i == TPCH::common::L_DISCOUNT ||
                    i == TPCH::common::L_EXTENDEDPRICE ||
                    i == TPCH::common::L_QUANTITY) {
                    if (enable_stats && i == TPCH::common::L_SHIPDATE) {
                        Stats<DateType> s = stats[j];
                        if (s.overlaps({19940101}, {19950100})) {
                            /* NOTE: require IOs */
                            q6_total_npages_to_be_read++;
                            if (compression_method != CompressionMethod::NONE) {
                                size_t compressed_page_size = compressed_page_sizes[j];
                                size_t read_size = roundup4096(compressed_page_size);
                                q6_page_read_bytes += read_size;
                            } else {
                                q6_page_read_bytes += options.page_size;
                            }
                        }
                    } else {
                        q6_total_npages_to_be_read++;
                        if (compression_method != CompressionMethod::NONE) {
                            size_t compressed_page_size = compressed_page_sizes[j];
                            size_t read_size = roundup4096(compressed_page_size);
                            q6_page_read_bytes += read_size;
                        } else {
                            q6_page_read_bytes += options.page_size;
                        }
                    }
                    q6_total_npages++;
                }

                /* Emulating TPC-H Q10's L_SHIPDATE */
                if (i == TPCH::common::L_ORDERKEY ||
                    i == TPCH::common::L_RETURNFLAG||
                    i == TPCH::common::L_EXTENDEDPRICE ||
                    i == TPCH::common::L_DISCOUNT) {
                    auto region_stats = sideways_stats[0];
                    auto orderdate_stats = sideways_stats[2];
                    Stats<int32_t> r = region_stats[j];
                    Stats<int32_t> o = orderdate_stats[j];
                    if (o.overlaps({19931001}, {19940101})) {
                        // std::cout << "Page " << lineitem_page_id << " overlaps with [1993-10-01, 1994-01-01)\n"
                        //     << " min_shipdate: " << o.min_val << " max_shipdate: " << o.max_val << std::endl;

                        /* NOTE: require IOs */
                        q10_total_npages_to_be_read++;
                        if (compression_method != CompressionMethod::NONE) {
                            size_t compressed_page_size = compressed_page_sizes[j];
                            size_t read_size = roundup4096(compressed_page_size);
                            q10_page_read_bytes += read_size;
                        } else {
                            q10_page_read_bytes += options.page_size;
                        }
                    }
                    q10_total_npages++;
                }

                bool fsst_done = false;
                if (compression_method == CompressionMethod::NONE) {
                    page_pread_host(options.output_fds, (void*)pag, lineitem_page_id, options.page_size);
                } else {
                    size_t compressed_page_size = compressed_page_sizes[j];
                    size_t read_size = roundup4096(compressed_page_size);
                    //std::cout << "lineitem_page_id: " << lineitem_page_id << " compressed page size: " << compressed_page_size << std::endl;
                    //V1
                    //page_pread_comp_host(options.output_fds, (void*)compressed_buffer, lineitem_page_id, read_size, options.page_size);
                    uint64_t offset = calc_compressed_page_offset(lineitem_page_id, l_offsets.data(), lineitem_start_page_id);
                    if (debug_print) {
                        std::cout << "lineitem_page_id: " << lineitem_page_id
                            << " compressed page size: " << compressed_page_size
                            << " offset: " << offset
                            << std::endl;
                    }
                    //V2
                    read_compressed_page_host(options.output_fds, compressed_buffer, lineitem_page_id, read_size, options.page_size, offset);
                    //V1
                    //page_pread_comp_host(options.output_fds, (void*)compressed_buffer, lineitem_page_id, read_size, options.page_size);
                    switch(compression_method) {
                        case CompressionMethod::SNAPPY:
                        {
                            int ret = decompress_page_with_snappy(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) {
                                std::cerr << "decompress_page_with_snappy failed: " << ret << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            break;
                        }
                        case CompressionMethod::DEFLATE:
                        {
                            int ret = decompress_page_with_zlib(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) {
                                std::cerr << "decompress_page_with_zlib failed: " << ret << std::endl;
                                std::cerr << "\tpage_id=" << lineitem_page_id << " compressed_page_size="
                                    << compressed_page_size << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            break;
                        }
                        case CompressionMethod::LZ4:
                        {
                            int ret = decompress_page_with_lz4(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) {
                                std::cerr << "decompress_page_with_lz4 failed: " << ret << std::endl;
                                std::cerr << "\tpage_id=" << lineitem_page_id << " compressed_page_size="
                                    << compressed_page_size << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            break;
                        }
                        case CompressionMethod::LZ4PAR:
                        {
                            int ret = decompress_page_with_lz4par(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), options.page_size);
                            if (ret != 0) { std::cerr << "decompress_page_with_lz4par failed: " << ret << std::endl; exit(EXIT_FAILURE); }
                            break;
                        }
                        case CompressionMethod::FSST:
                        case CompressionMethod::FSST_ROWID:
                        {
                            size_t checksum = 0;
                            uint32_t nrecs = fsst_verify_page(compressed_buffer, compressed_page_size, checksum);
                            field_checksums[i] += checksum;
                            memcpy(pag, compressed_buffer, sizeof(pag_head));
                            fsst_done = true;
                            break;
                        }
                        case CompressionMethod::PFOR:
                        {
                            int ret = decompress_page_with_pfor(compressed_buffer, compressed_page_size,
                                reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                            if (ret != 0) {
                                std::cerr << "decompress_page_with_pfor failed: " << ret << std::endl;
                                exit(EXIT_FAILURE);
                            }
                            break;
                        }
                        case CompressionMethod::PFOR64:
                        {
                             int ret = decompress_page_with_pfor64(compressed_buffer, compressed_page_size,
                                 reinterpret_cast<char*>(pag), decomp_workspace, options.page_size);
                             if (ret != 0) {
                                 std::cerr << "decompress_page_with_pfor failed: " << ret << std::endl;
                                 exit(EXIT_FAILURE);
                             }
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown compression method: " << static_cast<int>(compression_method) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }

                size_t nrecs_in_page = pag_get_nalloc(pag);
                size_t base_row_id = prefix_sum_nrecs[j];

                if (!(options.lbc_enabled && i == TPCH::common::L_COMMENT)) {
                    if (prefix_sum_nrecs[j + 1] - prefix_sum_nrecs[j] != nrecs_in_page) {
                        std::cerr << "Error(" << __FILE__ << ":" << __LINE__ << "): prefix sum for field "
                            << i
                            << " page_id: " << lineitem_page_id
                            << " prefix sum diff: " << (prefix_sum_nrecs[j + 1] - prefix_sum_nrecs[j])
                            << " nrecs_in_page: " << nrecs_in_page
                            << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }

                for (size_t l = 0; !fsst_done && l < nrecs_in_page; l++) {
                    // skip
                    bool enable_stdout = false;
                    //if (i < TPCH::common::L_FIELDS) {
                    //}
                    //if (j > 0 || k > 0 || l > 10) {
                    //    continue;
                    //}

                    if (j > 0 || l > 10) {
                        enable_stdout = false;
                        // continue;
                    } else {
                        enable_stdout = debug_print;
                    }

                    if (debug_print && l == 0 && j == 0) {
                        std::cout << "Number of recs in page " << lineitem_page_id << ": " << nrecs_in_page << std::endl;
                    }

                    switch (i) {
                        case TPCH::common::L_ORDERKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            //uint64_t rowid = pagcol_v2_get_rowid<int64_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int64_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_PARTKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            //uint64_t rowid = pagcol_v2_get_rowid<int64_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            //uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            //uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            //std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.dict_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_SUPPKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            //uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            //uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_LINENUMBER:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_QUANTITY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_EXTENDEDPRICE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_DISCOUNT:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_TAX:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_vchar_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_RETURNFLAG:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << static_cast<char>(col) << "|" << std::endl;
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_LINESTATUS:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << static_cast<char>(col) << "|" << std::endl;
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_SHIPDATE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_COMMITDATE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);int64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_RECEIPTDATE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_SHIPINSTRUCT:
                        {
                            size_t length = TPCH::common::fmt.lineitem_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::L_SHIPMODE:
                        {
                            size_t length = TPCH::common::fmt.lineitem_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            static int local_cnt = 0;
                            if (local_cnt < 10) std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            local_cnt++;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::L_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            uint64_t rowid = base_row_id + l;
                            if (enable_stdout) std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            //rec_print_rec(rec, TPCH::common::fmt.col_vchar_types, stdout);
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown lineitem field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }
            std::cout << "Total number of recs in lineitem table: " << sum_nrecs << std::endl;

            // if (enable_stats && i == TPCH::common::L_SHIPDATE) {
            //     std::cout << "Total number of pages to be read for SHIPDATE with filter: "
            //         << (double)q5_total_npages_to_be_read / (double)total_npages_shipdate
            //         << " ("
            //         << q5_total_npages_to_be_read << " / " << total_npages_shipdate
            //         << ")" << std::endl;
            // }
            if (compression_method != CompressionMethod::NONE) {
                free(ptr_compressed_page_sizes);
                free(ptr_compressed_buffer);
            }

            for (size_t l = 0; l < TPCH::common::kLineitemSidewaysCount; l++) {
                free(ptr_sideways_stats_alloc[l]);
            }
            if (enable_stats) {
                free(ptr_stats_alloc);
            }
            free(ptr_prefix_sum_alloc);
        }

        double q1_compression_ratio = q1_page_read_bytes / (double)(options.page_size * q1_total_npages_to_be_read);
        double q1_pruning_ratio = (double)q1_total_npages_to_be_read / (double)q1_total_npages;
        std::cout << "Total number of pages to be read for Q1 with IO puring and compression: "
            << q1_compression_ratio * q1_pruning_ratio * q1_total_npages * options.page_size / (1024 * 1024) << " MB ("
            << "original " << q1_total_npages * options.page_size / (1024 * 1024) << " MB, pruned and compressed "
            << q1_compression_ratio * q1_pruning_ratio
            << "), compression ratio:"
            << q1_compression_ratio
            << ", pruning ratio:"
            << q1_pruning_ratio
            << " ("
            << q1_total_npages_to_be_read << " / " << q1_total_npages
            << ")" << std::endl;

        double q3_compression_ratio = q3_page_read_bytes / (double)(options.page_size * q3_total_npages_to_be_read);
        double q3_pruning_ratio = (double)q3_total_npages_to_be_read / (double)q3_total_npages;
        std::cout << "Total number of pages to be read for Q3 with IO puring: "
            << q3_compression_ratio * q3_pruning_ratio * q3_total_npages * options.page_size / (1024 * 1024) << " MB ("
            << "original " << q3_total_npages * options.page_size / (1024 * 1024) << " MB, pruned and compressed "
            << q3_compression_ratio * q3_pruning_ratio
            << "), compression ratio:"
            << q3_compression_ratio
            << ", pruning ratio:"
            << q3_pruning_ratio
            << " ("
            << q3_total_npages_to_be_read << " / " << q3_total_npages
            << ")" << std::endl;

        double q5_compression_ratio = q5_page_read_bytes / (double)(options.page_size * q5_total_npages_to_be_read);
        double q5_pruning_ratio = (double)q5_total_npages_to_be_read / (double)q5_total_npages;
        std::cout << "Total number of pages to be read for Q5 with IO puring: "
            << q5_compression_ratio * q5_pruning_ratio * q5_total_npages * options.page_size / (1024 * 1024) << " MB ("
            << "original " << q5_total_npages * options.page_size / (1024 * 1024) << " MB, pruned and compressed "
            << q5_compression_ratio * q5_pruning_ratio
            << "), compression ratio:"
            << q5_compression_ratio
            << ", pruning ratio:"
            << q5_pruning_ratio
            << " ("
            << q5_total_npages_to_be_read << " / " << q5_total_npages
            << ")" << std::endl;

        double q6_compression_ratio = q6_page_read_bytes / (double)(options.page_size * q6_total_npages_to_be_read);
        double q6_pruning_ratio = (double)q6_total_npages_to_be_read / (double)q6_total_npages;
        std::cout << "Total number of pages to be read for Q6 with IO puring: "
            << q6_compression_ratio * q6_pruning_ratio * q6_total_npages * options.page_size / (1024 * 1024) << " MB ("
            << "original " << q6_total_npages * options.page_size / (1024 * 1024) << " MB, pruned and compressed "
            << q6_compression_ratio * q6_pruning_ratio
            << "), compression ratio:"
            << q6_compression_ratio
            << ", pruning ratio:"
            << q6_pruning_ratio
            << " ("
            << q6_total_npages_to_be_read << " / " << q6_total_npages
            << ")" << std::endl;

        double q10_compression_ratio = q10_page_read_bytes / (double)(options.page_size * q10_total_npages_to_be_read);
        double q10_pruning_ratio = (double)q10_total_npages_to_be_read / (double)q10_total_npages;
        std::cout << "Total number of pages to be read for Q10 with IO puring: "
            << q10_compression_ratio * q10_pruning_ratio * q10_total_npages * options.page_size / (1024 * 1024) << " MB ("
            << "original " << q10_total_npages * options.page_size / (1024 * 1024) << " MB, pruned and compressed "
            << q10_compression_ratio * q10_pruning_ratio
            << "), compression ratio:"
            << q10_compression_ratio
            << ", pruning ratio:"
            << q10_pruning_ratio
            << " ("
            << q10_total_npages_to_be_read << " / " << q10_total_npages
            << ")" << std::endl;


        std::cout << "Table-level checksums: " << std::endl;
        for (size_t i = 0; i < TPCH::common::L_FIELDS; i++) {
            std::cout << "\t" << i << "=" << field_checksums[i] << std::endl;
        }
        /* a test for scanning customer table */
       // size_t customer_num_varchar_fields = TPCH::common::fmt.customer_varchar_field_count;
        // auto &customer_varchar_field_indexes = TPCH::common::fmt.customer_varchar_field_indexes;

        free(ptr_base_alloc);
    }

    free(metadatap);

}

void test_scan_column_tables(LoaderOptions &options)
{
    // 512B-aligned alloc
    void *ptr;
    TPCHTableMetadata *metadatap;

    if (options.enable_sideways_stats) {
        test_scan_column_tables_with_sideways_stats(options);
        return;
    }

#if 1

#else
    superpage_set_constants(options.page_size);

    if (posix_memalign((void**)&ptr, 512, options.page_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Call constructor explicitly to initialize the object correctly

    page_pread_host(options.output_fds, (void*)ptr, 0, options.page_size);
    metadatap = reinterpret_cast<TPCHTableMetadata*>(ptr);

    TPCHTableMetadata &metadata = *metadatap;



    size_t num_super_npages = superpage_get_super_npage();
    if (posix_memalign((void**)&ptr, 512,
        options.page_size * SuperPage::NumPagesForSuperPages) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    // metadata.xtn_head = new(ptr) struct xtn_entry[XTN::MaxNumXTNs];
    // memset(metadata.xtn_head, 0, sizeof(struct xtn_entry) * XTN::MaxNumXTNs);

    if (options.load_nation) {
        /* Memory allocation for scan test */
        void *ptr_base;
        if (posix_memalign((void**)&ptr_base, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base);

        for (size_t i = 0; i < TPCH::common::N_FIELDS; i++) {
            // std::cout << metadata.table_customer_start_page_ids[i] << std::endl;
            std::cout << metadata.table_nation_start_page_ids[i] << std::endl;
            size_t nation_start_page_id = metadata.table_nation_start_page_ids[i];
            size_t nation_npages = metadata.table_nation_npages[i];

            std::cout
                << "start_page_id: " << nation_start_page_id
                << " nation_npages: " << nation_npages
                << std::endl;

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < nation_npages; j++) {
                const uint64_t nation_pagid = nation_start_page_id + j;
                page_pread_host(options.output_fds, (void*)pag, nation_pagid, options.page_size);
                size_t nrecs_in_page = pag_get_nalloc(pag);

                for (size_t l = 0; l < nrecs_in_page; l++) {
                    // std::cout << "Number of recs in page " << customer_pagid << ": " << nrecs_in_page << std::endl;
                    switch (i) {
                        case TPCH::common::N_NATIONKEY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << col << "|" << std::endl;
                            break;
                        }
                        case TPCH::common::N_NAME:
                        {
                            size_t length = TPCH::common::fmt.nation_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            break;
                        }
                        case TPCH::common::N_REGIONKEY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << col << "|" << std::endl;
                            break;
                        }
                        case TPCH::common::N_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown nation field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }
            std::cout << "Total number of recs in nation table: " << sum_nrecs << std::endl;
        }
        free(pag);
    }

    if (options.load_region) {
        /* Memory allocation for scan test */
        void *ptr_base;
        if (posix_memalign((void**)&ptr_base, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base);

        for (size_t i = 0; i < TPCH::common::R_FIELDS; i++) {
            // std::cout << metadata.table_customer_start_page_ids[i] << std::endl;
            std::cout << metadata.table_region_start_page_ids[i] << std::endl;
            size_t region_start_page_id = metadata.table_region_start_page_ids[i];
            size_t region_npages = metadata.table_region_npages[i];

            std::cout << "start_page_id: " << region_start_page_id
                << " region_npages: " << region_npages
                << std::endl;

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < region_npages; j++) {
                const uint64_t region_page_id = region_start_page_id + j;
                page_pread_host(options.output_fds, (void*)pag, region_page_id, options.page_size);
                size_t nrecs_in_page = pag_get_nalloc(pag);

                for (size_t l = 0; l < nrecs_in_page; l++) {
                    // std::cout << "Number of recs in page " << customer_pagid << ": " << nrecs_in_page << std::endl;
                    switch (i) {
                        case TPCH::common::R_REGIONKEY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << col << "|" << std::endl;
                            // rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            break;
                        }
                        case TPCH::common::R_NAME:
                        {
                            size_t length = TPCH::common::fmt.region_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            // rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            break;
                        }
                        case TPCH::common::R_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            // rec_print_rec(rec, TPCH::common::fmt.col_vchar_types, stdout);
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown region field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }
            std::cout << "Total number of recs in region table: " << sum_nrecs << std::endl;
        }
        free(pag);
    }


    #if 1
    if (options.load_customer) {
        /* Memory allocation for scan test */
        void *ptr_base;
        if (posix_memalign((void**)&ptr_base, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base);

        for (size_t i = 0; i < TPCH::common::C_FIELDS; i++) {
            // std::cout << metadata.table_customer_start_page_ids[i] << std::endl;
            std::cout << metadata.table_customer_start_page_ids[i] << std::endl;
            size_t customer_start_page_id = metadata.table_customer_start_page_ids[i];
            size_t customer_npages = metadata.table_customer_npages[i];

            std::cout << "start_page_id: " << customer_start_page_id
                << " customer_npages: " << customer_npages
                << std::endl;

            size_t sum_nrecs = 0;
            for (size_t j = 0; j < customer_npages; j++) {
                const uint64_t customer_page_id = customer_start_page_id + j;
                page_pread_host(options.output_fds, (void*)pag, customer_page_id, options.page_size);
                size_t nrecs_in_page = pag_get_nalloc(pag);

                for (size_t l = 0; l < nrecs_in_page; l++) {
                    // std::cout << "Number of recs in page " << customer_pagid << ": " << nrecs_in_page << std::endl;
                    switch (i) {
                        case TPCH::common::C_CUSTKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int64_types, stdout);
                            break;
                        }
                        case TPCH::common::C_NAME:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.dict_types, stdout);
                            break;
                        }
                        case TPCH::common::C_ADDRESS:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            break;
                        }
                        case TPCH::common::C_NATIONKEY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            break;
                        }
                        case TPCH::common::C_PHONE:
                        {
                            size_t length = TPCH::common::fmt.customer_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            break;
                        }
                        case TPCH::common::C_ACCTBAL:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            break;
                        }
                        case TPCH::common::C_MKTSEGMENT:
                        {
                            size_t length = TPCH::common::fmt.customer_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            break;
                        }
                        case TPCH::common::C_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_vchar_types, stdout);
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown customer field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }
            std::cout << "Total number of recs in customer table: " << sum_nrecs << std::endl;
        }
        /* a test for scanning customer table */
        // size_t customer_num_varchar_fields = TPCH::common::fmt.customer_varchar_field_count;
        // auto &customer_varchar_field_indexes = TPCH::common::fmt.customer_varchar_field_indexes;

        free(pag);
    }
    #endif

    if (options.load_lineitem) {
        /* Memory allocation for scan test */
        void *ptr_base_alloc;
        if (posix_memalign((void**)&ptr_base_alloc, 512, options.page_size) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        PAG *pag = reinterpret_cast<PAG*>(ptr_base_alloc);

        /* checksums are initialized with 0 */
        std::array<uint64_t, TPCH::common::L_FIELDS> field_checksums = {};
        for (size_t i = 0; i < TPCH::common::L_FIELDS; i++) {
            // std::cout << metadata.table_customer_start_page_ids[i] << std::endl;
            std::cout << metadata.table_lineitem_start_page_ids[i] << std::endl;
            size_t lineitem_start_page_id = metadata.table_lineitem_start_page_ids[i];
            size_t lineitem_npages = metadata.table_lineitem_npages[i];

            size_t lineitem_prefix_sum_start_page_id = metadata.table_lineitem_prefix_sum_start_page_ids[i];
            size_t lineitem_prefix_sum_npages = metadata.table_lineitem_prefix_sum_npages[i];

            size_t lineitem_stats_start_page_id = metadata.table_lineitem_stats_start_page_ids[i];
            size_t lineitem_stats_npages = metadata.table_lineitem_stats_npages[i];
            size_t lineitem_nstats = metadata.table_lineitem_nstats[i];
            const bool enable_stats = (lineitem_nstats > 0) ? true : false;

            void *ptr_prefix_sum_alloc = nullptr;
            if (posix_memalign((void**)&ptr_prefix_sum_alloc, 512, lineitem_prefix_sum_npages* options.page_size) != 0)
            {
                std::cerr << "posix_memalign failed" << std::endl;
                exit(EXIT_FAILURE);
            }
            uint64_t *prefix_sum_nrecs = reinterpret_cast<uint64_t*>(ptr_prefix_sum_alloc);
            for (size_t j = 0; j < lineitem_prefix_sum_npages; j++) {
                const uint64_t lineitem_prefix_sum_page_id = lineitem_prefix_sum_start_page_id + j;
                page_pread_host(options.output_fds, reinterpret_cast<char*>(prefix_sum_nrecs) + j * options.page_size,
                    lineitem_prefix_sum_page_id, options.page_size);
            }

            void *ptr_stats_alloc = nullptr;
            std::span<Stats<DateType>> stats;
            if (enable_stats) {
                if (posix_memalign((void**)&ptr_stats_alloc, 512, lineitem_stats_npages * options.page_size) != 0)
                {
                    std::cerr << "posix_memalign failed" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (size_t j = 0; j < lineitem_stats_npages; j++) {
                    const uint64_t lineitem_stats_page_id = lineitem_stats_start_page_id + j;
                    page_pread_host(options.output_fds, reinterpret_cast<char*>(ptr_stats_alloc) + j * options.page_size,
                        lineitem_stats_page_id, options.page_size);
                }
                Stats<DateType> *statsp = reinterpret_cast<Stats<DateType>*>(ptr_stats_alloc);
#if 0
                for (size_t j = 0; j < lineitem_nstats; j++) {
                    std::cout << "Stats[" << j << "] min: " << statsp[j].min_val << " max: " << statsp[j].max_val << std::endl;
                }
#endif
                stats = std::span<Stats<DateType>>(statsp, lineitem_nstats);
#if 0
                for (size_t j = 0; j < lineitem_nstats; j++) {
                    std::cout << "Stats[" << j << "] min: " << stats[j].min_val << " max: " << statsp[j].max_val << std::endl;
                }
#endif
                // exit(1);
            }

            std::cout << "FIELD INDEX: " << i
                << " start_page_id: " << lineitem_start_page_id 
                << " lineitem_npages: " << lineitem_npages
                << std::endl;
                
            if (prefix_sum_nrecs[0] != 0) {
                std::cerr << "Error: prefix_sum_nrecs[0] is expected to be 0, but got " << prefix_sum_nrecs[0] << std::endl;
                exit(EXIT_FAILURE);
            }

            size_t sum_nrecs = 0;
            size_t total_npages_to_be_read = 0;
            size_t total_npages_shipdate = (i == TPCH::common::L_SHIPDATE) ? lineitem_npages : 0;
            size_t shipdate_base_page_id = 0;

            size_t q1_total_npages_to_be_read = 0;
            size_t q1_total_npages = 0;
            size_t q3_total_npages_to_be_read = 0;
            size_t q3_total_npages = 0;
            size_t q5_total_npages_to_be_read = 0;
            size_t q5_total_npages = 0;
            size_t q6_total_npages_to_be_read = 0;
            size_t q6_total_npages = 0;
            size_t q10_total_npages_to_be_read = 0;
            size_t q10_total_npages = 0;
            for (size_t j = 0; j < lineitem_npages; j++) {
                const uint64_t lineitem_page_id = lineitem_start_page_id + j;

                /* Start scan test */
                /* Emulating TPC-H Q6 */
                if (enable_stats && i == TPCH::common::L_SHIPDATE) {
                    Stats<DateType> s = stats[j];
                    if (s.overlaps({19940101}, {19950100})) {
                        std::cout << "Page " << lineitem_page_id << " overlaps with [1994-01-01, 1995-01-01)\n"
                            << " min_shipdate: " << s.min_val << " max_shipdate: " << s.max_val << std::endl;

                        /* NOTE: require IOs */
                        total_npages_to_be_read++;
                    }
                }

                page_pread_host(options.output_fds, (void*)pag, lineitem_page_id, options.page_size);
                size_t nrecs_in_page = pag_get_nalloc(pag);

                if (!(options.lbc_enabled && i == TPCH::common::L_COMMENT)) {
                    if (prefix_sum_nrecs[j + 1] - prefix_sum_nrecs[j] != nrecs_in_page) {
                        std::cerr << "Error(" << __FILE__ << ":" << __LINE__ << "): prefix sum for field "
                            << i
                            << i << " of nrecs does not match."
                            << " page_id: " << lineitem_page_id
                            << " prefix sum diff: " << (prefix_sum_nrecs[j + 1] - prefix_sum_nrecs[j])
                            << " nrecs_in_page: " << nrecs_in_page
                            << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }

                for (size_t l = 0; l < nrecs_in_page; l++) {
                    // skip 
                    bool enable_stdout = false;
                    //if (i < TPCH::common::L_FIELDS) {
                    //}
                    //if (j > 0 || k > 0 || l > 10) {
                    //    continue;
                    //}

                    if (j > 0 || l > 10) {
                        enable_stdout = false;
                        // continue;
                    } else {
                        enable_stdout = true;
                    }
                
                    if (l == 0 && j == 0) {
                        std::cout << "Number of recs in page " << lineitem_page_id << ": " << nrecs_in_page << std::endl;
                    }

                    switch (i) {
                        case TPCH::common::L_ORDERKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int64_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int64_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_PARTKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int64_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            //uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            //uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            //std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.dict_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_SUPPKEY:
                        {
                            int64_t col = pagcol_fetch_int<int64_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int64_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            //uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            //uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_LINENUMBER:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_QUANTITY:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_EXTENDEDPRICE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_DISCOUNT:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_TAX:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_vchar_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_RETURNFLAG:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << static_cast<char>(col) << "|" << std::endl;
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_LINESTATUS:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << static_cast<char>(col) << "|" << std::endl;
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_SHIPDATE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_COMMITDATE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_RECEIPTDATE:
                        {
                            int32_t col = pagcol_fetch_int<int32_t>(pag, l, options.page_size);
                            uint64_t rowid = pagcol_v2_get_rowid<int32_t>(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << col << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_int32_types, stdout);
                            field_checksums[i] += col;
                            break;
                        }
                        case TPCH::common::L_SHIPINSTRUCT:
                        {
                            size_t length = TPCH::common::fmt.lineitem_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::L_SHIPMODE:
                        {
                            size_t length = TPCH::common::fmt.lineitem_field_sizes[i];
                            char *col = pagcol_fetch_char(pag, l, length, options.page_size);
                            uint64_t rowid = pagcol_get_rowid(pag, l, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            static int local_cnt = 0;
                            if (local_cnt < 10) std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            local_cnt++;
                            //rec_print_rec(rec, TPCH::common::fmt.col_char_types, stdout);
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            break;
                        }
                        case TPCH::common::L_COMMENT:
                        {
                            char *col = pagcol_fetch_vchar(pag, l, options.page_size);
                            uint16_t length = pagcol_fetch_vchar_len(pag, l, options.page_size);
                            uint64_t rowid = pagcol_fetch_vchar_rowid(pag, l, length, options.page_size);
                            if (enable_stdout) std::cout << rowid << "|" << std::string_view(col, length) << "|" << std::endl;
                            auto v = std::string_view(col, length);
                            size_t checksum = std::transform_reduce(
                                v.begin(), v.end(), size_t{0},
                                std::plus<size_t>{},
                                [](char c) { return static_cast<size_t>(c); });
                            field_checksums[i] += checksum;
                            //rec_print_rec(rec, TPCH::common::fmt.col_vchar_types, stdout);
                            break;
                        }
                        default:
                        {
                            std::cerr << "Unknown lineitem field: " << i << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                sum_nrecs += nrecs_in_page;
            }
            std::cout << "Total number of recs in lineitem table: " << sum_nrecs << std::endl;

            if (enable_stats && i == TPCH::common::L_SHIPDATE) {
                std::cout << "Total number of pages to be read for SHIPDATE with filter: "
                    << (double)total_npages_to_be_read / (double)total_npages_shipdate
                    << " ("
                    << total_npages_to_be_read << " / " << total_npages_shipdate
                    << ")" << std::endl;
            }
        
            if (enable_stats) {
                free(ptr_stats_alloc);
            }
            free(ptr_prefix_sum_alloc);
        }

        std::cout << "Table-level checksums: " << std::endl;
        for (size_t i = 0; i < TPCH::common::L_FIELDS; i++) {
            std::cout << "\t" << i << "=" << field_checksums[i] << std::endl;
        }
        /* a test for scanning customer table */
       // size_t customer_num_varchar_fields = TPCH::common::fmt.customer_varchar_field_count;
        // auto &customer_varchar_field_indexes = TPCH::common::fmt.customer_varchar_field_indexes;

        free(ptr_base_alloc);
    }

    free(metadatap);
#endif
}
