#pragma once

#include <fcntl.h>
#include <linux/fs.h>
#include <numa.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
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


#include "cpu_usage.cu"
#include "page.cu"
#include "ssb_tables.cuh"
#include "pack.cu"
#include "common.cu"

// #define DEBUG_PRINT

namespace chrono = std::chrono;
namespace fs = std::filesystem;

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

struct LoaderOptions
{
    // Reuse 
    char const *input_dirname;
    char const *output_files;
    bool compress;
    bool dryrun;
    bool verbose;
    size_t scale_factor;

    char *devname[128];
    size_t ndev;
    std::vector<int> output_fds;

    size_t page_size;
    size_t buffer_size_per_field;

    bool load_lineorder;
    bool load_supplier;
    bool load_part;
    bool load_customer;
    bool load_ddate;

    bool test;

    OutputFormat output_format;
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

struct LoaderStats {
    std::vector<StatEntry> entries;
};

template <size_t N>
struct LoaderThreadStats {
    ssize_t nlines_processed;
    std::array<size_t, N> nios;
    std::array<size_t, N> nwritten_sectors;
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

// size_t nlines[nfiles];
// size_t nlines[nfiles];
template <size_t N>
struct FactTableLoadTask {
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

        size_t i = 0;
        for (const auto& metric : entry.metrics) {
            if (compressed) {
                auto orig_mb = metric.fields_written * page_size / MEBI;
                // auto comp_mb_str = format_bytes_from_nsectors(metric.fields_written_compressed, MEBI);
                auto comp_bytes = bytes_from_nsectors(metric.fields_written_compressed);
                auto comp_meta_offset_mb = metric.fields_written_offset_for_compression * page_size / MEBI;
                auto comp_meta_sizcomp_mb = metric.fields_written_sizcomp_for_compression * page_size / MEBI;

                std::cout << "  Field: " << metric.field_name
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
                std::cout << "  Field: " << metric.field_name
                          << ", Fields Written: " << metric.fields_written * page_size / MEBI << " MB"
                          << std::endl;
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
            std::cout << "  Total Fields Written: " << field_total << " MB" << std::endl;
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

int count_lineorder_files(const std::string& directory) {
#ifdef DEBUG_PRINT
    std::cout << "Regex rule: " << "lineorder\\.tbl\\**" << std::endl;
#endif
    return count_regex_matched_files(directory, "lineorder.tbl.?[0-9]*");
}



static void binUnpack(uint* out, uint* block_offsets, uint num_entries) {
    // struct binpack_hdr* out_hdr = reinterpret_cast<struct binpack_hdr*>(out);
    std::cout << "=== Start binUnpack32 ===" << std::endl;
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
    
    printf("nblock=%u ", nblock);
    for (uint k = 0; k < nblock; k++) {
      printf("offsets[%u]=%u ", k, block_offsets[k]);
    }
    printf("\n");
  
    block_size = out[0];
    miniblock_count = out[1];
    total_count = out[2];
    // Start to prase data block.
    first_val = out[3];
  
    std::cout << "block_size " << block_size << " miniblock_count " << miniblock_count
      << " total_count " << total_count << " first_val " << first_val << std::endl;
    
    offset = 4;
    uint min_val = out[offset];
    offset = 5;
    uint bitwidths = out[offset];
    printf("bitwidths: %x(%u)\n", bitwidths, offset);
    uint bitwidth1 = bitwidths & 255;
    uint bitwidth2 = (bitwidths >> 8) & 255;
    uint bitwidth3 = (bitwidths >> 16) & 255;
    uint bitwidth4 = (bitwidths >> 24) & 255;
    printf("\tbitwidths: %x %x %x %x (%u)\n", bitwidth1, bitwidth2, bitwidth3, bitwidth4, bitwidth1);
  
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
  
  
    printf("block_count %d\n", block_count);
    printf("miniblock_count %d\n", miniblock_count);
  
    for (uint i = 0; i < block_count; i++) {
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
      printf("\toffset=%u min_val: %d, bitwidths: %x\n", offset, min_val, bitwidths);
  
      // reset context
      current_word = 0;
      bits_available = 32;
      for (uint j = 0; j < miniblock_count; j++) {
        bitwidth = bitwidths & 255;
        mask = (1U << bitwidth) - 1;
        for (int k=0; k<miniblock_size; k++) {
          // printf("current_pos: %u current_value: %lx\n", current_word, current_value);
          printf("\tcurrent_value: %u (%x)\n", current_value, current_value);
          if (bits_available < bitwidth) {
            current_word++;
            uint next_value = data_baseaddr[current_word];
            assert(current_word > 0);
            printf("\tcurrent_word: %u -> %u, bits_available=%u next_value=%u (%x)\n", current_word - 1, current_word, bits_available, next_value, next_value);
  
            if (bits_available > 0) {
              printf("\t\tcurrent_value: %x\n", current_value & ((1U << bits_available) - 1));
              unpacked = ((current_value & ((1U << bits_available) - 1)));
              printf("\t\tunpacking: %x %x\n", unpacked, next_value);
              unpacked += ((next_value & ((1U << (bitwidth - bits_available)) - 1)) << bits_available);
              // printf("unpacked2: %x\n", unpacked);
              printf("\t\tunpacking: %x %u\n", unpacked, bits_available);
              // current_value = next_value >> (bitwidth - bits_available);
              // bits_available = 64 - (bitwidth - bits_available);
              // current_value = next_value >> bits_available;
              current_value = next_value >> (bitwidth - bits_available);
            } else {
              unpacked = next_value & mask;
              current_value = next_value;
            }
  
            // current_value = next_value;
            bits_available = 32 - (bitwidth - bits_available);
            printf("\tunpacked3: %u (%u + %u (%x)) %u %u\n", unpacked + min_val, min_val, unpacked, unpacked, bits_available, bitwidth);
            // printf("\t%u", unpacked);
          } else {
            unpacked = current_value & mask;
            printf("\tunpacked4: %u (unpacked:%u min_val:%u) (%u %u)\n", unpacked + min_val, unpacked, min_val, bits_available, bitwidth);
            current_value >>= bitwidth;
            bits_available -= bitwidth;
            // printf("(%lx %u) ", current_value, bits_available);
          }
        }
        // exit(1);
        bitwidth >>= 8;
        current_word++;
        bits_available = 32;
        current_value = data_baseaddr[current_word];
  
        printf("\nNOTE: finished processing miniblock %d pos(%u->%u)\n"
          "\tnew current value %x\n",
          j, current_word - 1, current_word, current_value);
      }
  
      // The first 2 longs are header value, so plus + 2 is required
      printf("\noffset=%d, block_offsets[%d]=%d\n",block_offsets[i]+current_word+2, i+1, block_offsets[i+1]);
      // for (int j=block_offsets[i]+current_word; j<block_offsets[i+1]; j++) {
      //   printf("\nout[%d] %lx\n", j, out[j]);
      // }
      // exit(1);
    }
#endif
}

#if 0
int open_file_rdonly(const char *file)
{
    int oflag = 0;
    char *files = strdup(file);
    // oflag = O_RDONLY | O_DIRECT;
    oflag = O_RDONLY;

    int fd = open(file, oflag);
    if (fd < 0)
    {
        std::cerr << "failed to open file " << file << std::endl;
        perror("open");
        close(fd);
        exit(EXIT_FAILURE);
    }

    return fd;
}

void close_files_rdonly(std::vector<int> &fds)
{
    for (auto fd : fds)
    {
        if (fd >= 0)
        {
            close(fd);
        }
    }
}

int open_input_file(LoaderOptions &options, std::string &col)
{
    std::stringstream ss;
    ss << options.input_dirname << "/" << col << ".tbl";
    std::string path = ss.str();
    return open_file_rdonly(path.c_str());
}

int open_input_file_p(LoaderOptions &options, std::string &col)
{
    std::stringstream ss;
    ss << options.input_dirname << "/" << col << ".tbl.p";
    std::string path = ss.str();
    return open_file_rdonly(path.c_str());
}

void open_input_files_lineorder(LoaderOptions &options, std::vector<int> &fds)
{
    std::string col = "lineorder";
    int n = count_lineorder_files(options.input_dirname);
    int i;
    for (i = 1; i <= n; i ++) {
        std::stringstream ss;
        if (n == 1)
        {
            ss << options.input_dirname << "/" << col << ".tbl";
        }
        else
        {
            ss << options.input_dirname << "/" << col << ".tbl." << i;
        }
        std::string path = ss.str();
        std::cout << "Opening file: " << path << std::endl;
        int fd = open_file_rdonly(path.c_str());
        if (fd < 0)
        {
            std::cerr << "failed to open file " << path << std::endl;
            perror("open");
            close(fd);
            exit(EXIT_FAILURE);
        }
        fds.push_back(fd);
        std::cout << "Opened file: " << path << " with fd: " << fd << std::endl;
    }
}
#else
static std::vector<std::string> prep_input_files_lineorder(LoaderOptions &options)
{
    std::vector<std::string> files;
    std::string col = "lineorder";
    int n = count_lineorder_files(options.input_dirname);
    int i;
    for (i = 1; i <= n; i ++) {
        std::stringstream ss;
        if (n == 1)
        {
            ss << options.input_dirname << "/" << col << ".tbl";
        }
        else
        {
            ss << options.input_dirname << "/" << col << ".tbl." << i;
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

static ssize_t count_lineorder_lines(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open file.\n";
        return -1;
    }

    std::string line;
    size_t count = 0;
    while (std::getline(file, line)) {
        // std::vector<std::string_view> fields;
        // std::string_view sv(line);
        count++;
    }
 
    return count;
}

template <size_t N>
static std::vector<FactTableLoadTask<N>> generate_lineorder_tasks(size_t page_size,
    bool compress, std::vector<std::string> &paths, const std::array<size_t, N>& field_sizes)
{
    /* The object of this function is to create load_tasks */
    std::vector<FactTableLoadTask<N>> load_tasks;

    std::vector<std::packaged_task<ssize_t()>> tasks;
    std::vector<std::future<ssize_t>> futures;
    std::vector<std::thread> threads;
    std::vector<size_t> nlines;

    for (auto& path : paths) {
        std::packaged_task<ssize_t()> task(std::bind(count_lineorder_lines, std::ref(path)));
        futures.push_back(task.get_future());
        threads.emplace_back(std::move(task));
    }

    for (auto& t : threads) {
        t.join();
    }

    size_t count_sum = 0;
    std::vector<size_t> prefix_sum;
    prefix_sum.reserve(paths.size() + 1);
    prefix_sum.push_back(0);
    for (size_t i = 0; i < futures.size(); ++i) {
        ssize_t count = futures[i].get();
        if (count < 0) {
            std::cerr << "[FATAL] Failed to count lines in file: " << paths[i] << std::endl;
            exit(EXIT_FAILURE);
        }
#ifdef DEBUG_PRINT
        std::cout << "Linecount of \"" << paths[i] << "\" is " 
                  << count << std::endl;
#endif
        nlines.push_back(count);
        count_sum += count;
        prefix_sum.push_back(count_sum);
#ifdef DEBUG_PRINT
        std::cout << "prefix_sum[" << i + 1 << "] = " << prefix_sum[i + 1] << std::endl;
#endif
    }

    if (page_size % NUM_ITEMS_PER_TILE != 0) {
        std::cerr << "[FATAL] page_size is not a multiple of TILE_SIZE" << std::endl;
        exit(EXIT_FAILURE);
    }
    // size_t tiles_per_page = page_size / (sizeof(int32_t) * NUM_ITEMS_PER_TILE);
    // size_t items_per_page = tiles_per_page * NUM_ITEMS_PER_TILE;
    // items_per_page is calculated by using the 32bit value, but this is also safe
    // for 64bit value if page_size is a multiple of 8.
    // Typically, page_size is 1MB, so this is safe (also fullfill the page with elements).
    size_t items_per_page = page_size / sizeof(int32_t);

    std::cout << "Total linecount is " << count_sum << std::endl;

    size_t base_id = 1;
    size_t nskip = 0;
    size_t start_page_offset = 0;
    size_t start_page_offset64 = 0;
    load_tasks.reserve(paths.size());
    for (size_t i = 1; i < paths.size() + 1; ++i) {
        size_t nitems_sum = prefix_sum[i] - prefix_sum[base_id - 1] - nskip;
        if (i == paths.size()) {
            // last task
            FactTableLoadTask<N> task;
            task.base_id = base_id;
            task.nskip_lines_curr = nskip;
            task.nfiles = i - (base_id - 1);
            task.nlines_to_read = nitems_sum;
            task.start_page_offset = start_page_offset;
            task.start_page_offset64 = start_page_offset64;
            task.compress = compress;
            if (compress) {
                task.buf_compress = static_cast<char*>(mb_alloc(page_size));
                task.offsets_local = static_cast<uint*>(mb_alloc(sizeof(uint) * (items_per_page + 1)));
            }

            load_tasks.push_back(task);
            continue;
        } else if (i == base_id) {
            // nfiles == 1
            continue;
        }
        /* handling the smaller files */
        if (nitems_sum >= items_per_page) {
            FactTableLoadTask<N> task;

            size_t nitems_prev = prefix_sum[i - 1] - prefix_sum[base_id - 1] - nskip;
#ifdef DEBUG_PRINT
            std::cout << "nitems_prev: " << nitems_prev << std::endl;
#endif
            size_t npages = (nitems_prev % items_per_page == 0) ?
                (nitems_prev / items_per_page) : (nitems_prev / items_per_page + 1);
            // 64bit field requires doubled pages
            size_t npages64 = npages * 2;
            size_t nitems_remain =  nitems_sum - npages * items_per_page;

            task.base_id = base_id;
            task.nskip_lines_curr = nskip;
            task.nfiles = i - (base_id - 1);
            task.nlines_to_read = (i == paths.size()) ? 
                (count_sum % (npages * items_per_page)) : (npages * items_per_page);
            task.start_page_offset = start_page_offset;
            task.start_page_offset64 = start_page_offset64;
            task.compress = compress;
            if (compress) {
                task.buf_compress = static_cast<char*>(mb_alloc(page_size));
                task.offsets_local = static_cast<uint*>(mb_alloc(sizeof(uint) * (items_per_page + 1)));
            }


            load_tasks.push_back(task);

            // update base_id
            base_id = i;
            // update nskip
            assert(nlines[i - 1] == prefix_sum[i] - prefix_sum[i - 1]);
            nskip = nlines[i - 1] - nitems_remain;
            start_page_offset += npages;
            start_page_offset64 += npages64;
        }
    }

    size_t count_verify = 0;
    for (size_t i = 0; i < load_tasks.size(); ++i) {
        std::cout << "Task " << i << ": base_id = " << load_tasks[i].base_id
                  << ", nfiles = " << load_tasks[i].nfiles
                  << ", nskip_lines_curr = " << load_tasks[i].nskip_lines_curr
                  << ", nlines_to_read = " << load_tasks[i].nlines_to_read
                  << ", start_page_offset = " << load_tasks[i].start_page_offset
                  << std::endl;
        count_verify += load_tasks[i].nlines_to_read;
    }
    if (count_verify != count_sum) {
        std::cerr << "[BUG][FATAL] Count verify failed: " << count_verify
                  << " != " << count_sum << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Total tasks: " << load_tasks.size() << std::endl;
    assert(load_tasks.size() <= paths.size());

    return load_tasks;
}

template <typename EnumType, size_t N>
static LoaderThreadStats<N> load_fact_table_lines(FactTableLoadTask<N>& task, std::vector<std::string>& paths,
    std::vector<int>& output_fds,
    const std::array<char*, N>& buffers, const std::array<uint64_t, N>& field_start_page_ids,
    const std::array<size_t, N>& target_fields_sizes,
    const std::array<EnumType, N>& target_fields, const size_t page_size, const int dryrun, const int verbose)
{
    size_t nlines_skipped = 0;
    ssize_t nlines_processed = 0;
    size_t file_index = 0;
    std::array<size_t, N> offsets;
    std::array<size_t, N> nelements;
    std::array<uint64_t, N> pages;
    std::array<size_t, N> nios;
    std::array<size_t, N> nwritten_sectors;
    for (size_t i = 0; i < N; ++i) {
        offsets[i] = 0;
        nelements[i] = page_size / target_fields_sizes[i];
        if (target_fields_sizes[i] == sizeof(int64_t)) {
            pages[i] = field_start_page_ids[i] + task.start_page_offset64;
        } else {
            pages[i] = field_start_page_ids[i] + task.start_page_offset;
        }
        nios[i] = 0;
        nwritten_sectors[i] = 0;
#ifdef DEBUG_PRINT
        std::cout << "Field " << i << ": size = " << target_fields_sizes[i] << ", nelements = " << nelements[i] << ", page_start = " << pages[i] << "\n";
#endif
    }

    do {
        if (file_index >= task.nfiles) {
            std::cerr << "File index out of range: " 
                "file_index:" << file_index
                << ", base_id:" << task.base_id
                << ", nfiles:" << task.nfiles
                << ", paths.size():" << paths.size() << std::endl;
            return { -1, nios, nwritten_sectors };
        }
        std::string &path = paths[task.base_id - 1 + file_index];
        std::cout << "Processing file: " << path <<
            "(file_index=" << file_index << ")" << std::endl;
#ifdef DEBUG_PRINT
        std::cout << "Processing file: " << path << std::endl;
#endif
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Failed to open file.\n";
            return { -1, nios, nwritten_sectors };
        }

        std::string line;
        while (std::getline(file, line)) {
            if (nlines_skipped < task.nskip_lines_curr) {
                ++nlines_skipped;
                // std::cout << "Skipping line " << nlines << " in file: " << path << std::endl;
                continue;
            }
            // std::vector<std::string_view> fields;
            std::string_view sv(line);
            size_t i = 0;
            size_t f = 0;
            size_t start = 0;
            size_t end;
            size_t offset;
            uint64_t pagid;
            ulong value64;
            size_t field_size;
            ulong *buf64;
            uint *buf32;
            char *buf;

            while (((end = sv.find('|', start)) != std::string_view::npos) && (f < N)) {
                // std::cout << "Processing field: " << f << "\n";
                if (i < target_fields[f]) {
                    // skip
                } else {
                    std::string_view part = sv.substr(start, end - start);
                    auto [ptr, ec] = std::from_chars(part.data(), part.data() + part.size(), value64);

#ifdef DEBUG_PRINT
                    if (nlines_processed < 3 && task.base_id == 1) {
                        if (ec == std::errc()) {
                            std::cout << "Parsed number: " << value64 << "\n";
                        }
                    }
                    if (ec == std::errc()) {
                        // OK
                    } else {
                        std::cerr << "Failed to parse integer from: " << part << "\n";
                    }
#endif
                    if (ec != std::errc()) {
                        std::cerr << "Failed to parse integer from: " << part << "\n";
                        return { -1, nios, nwritten_sectors };
                    }
                    field_size = target_fields_sizes[f];
                    buf = reinterpret_cast<char*>(buffers[f]);
                    buf64 = reinterpret_cast<ulong*>(buffers[f]);
                    buf32 = reinterpret_cast<uint*>(buffers[f]);
                    offset = offsets[f];
                    start = end + 1;

                    if (field_size == sizeof(int64_t)) {
                        if (offset >= nelements[f]) {
                            std::cerr << "Buffer overflow for field " << f << "\n";
                            return { -1, nios, nwritten_sectors };
                        }
                        buf64[offset] = value64;
                    } else if (field_size == sizeof(int32_t)) {
                        if (offset >= nelements[f]) {
                            std::cerr << "Buffer overflow for field " << f << "\n";
                            return { -1, nios, nwritten_sectors };
                        }
                        buf32[offset] = static_cast<int32_t>(value64);
                        #if 0
                        if (f == 5) {
                            if (offset > 0 && offset % 16 == 0) {
                                std::cout << std::endl;
                            }
                            std::cout << buf32[offset] << " ";
                        }
                        #endif
                    } else {
                        std::cerr << "Unsupported field size: " << field_size << "\n";
                        return { -1, nios, nwritten_sectors };
                    }
                    offsets[f]++;
                    if (offsets[f] >= nelements[f]) {
                        // Write buffer to disk or process it as needed
                        pagid = pages[f];
#ifdef DEBUG_PRINT
                        // std::cout << "Buffer full for field " << f << ", writing to disk. (pageid=" << pagid << ")\n";
                        if (f == 0) {
                            std::cerr << "Buffer full for field " << f << ", writing to disk. (pageid=" << pagid << ")\n";
                        }
                        //std::cerr << "Buffer full for field " << f << ", writing to disk. (pageid=" << pagid << ")\n";
#endif

                        if (dryrun) {
                            std::cout << "Buffer full for field " << f << ", writing to disk. (pageid=" << pagid << ")\n";
                        } else {
                            if (verbose) {
                                std::cout << "Buffer full for field " << f << ", writing to disk. (pageid=" << pagid << ")\n";
                            }

                            if (task.compress) {
                                uint size, write_size;
                                size_t comp_block_size = 128;
                                assert(nelements[f] % comp_block_size == 0);
                                size_t last_index = nelements[f] / comp_block_size;
                                memset(task.buf_compress, 0, page_size);
                                std::array<uint, 32> workspace;
                                if (field_size == sizeof(int64_t)) {
                                    auto buf_out = reinterpret_cast<ulong*>(task.buf_compress);
                                    size = binPack64(buf64, buf_out, task.offsets_local, nelements[f], workspace.data());
                                    size *= sizeof(int64_t);
                                    task.offsets[f].insert(task.offsets[f].end(), task.offsets_local, task.offsets_local + last_index + 1);
                                    write_size = roundup512(size);
                                    assert(write_size <= page_size);
                                    task.compressed_page_sizes[f].push_back(write_size);
                                } else if (field_size == sizeof(int32_t)) {
                                    auto buf_out = reinterpret_cast<uint*>(task.buf_compress);
                                    #if 0
                                    if (f == 5) {
                                        std::cout << "offsets[f]: " << offsets[f] << ", nelements[f]: " << nelements[f] << std::endl;
                                        std::cout << buf32[0] << " " << buf32[1] << " " << buf32[2] << " " << buf32[3] << std::endl;
                                    }
                                    #endif
                                    /* NOTE: binPack function modify the contents in buf32. */
                                    size = binPack(buf32, buf_out, task.offsets_local, nelements[f], workspace.data());
                                    // #define DEBUG_AND_STOP
                                    #if 0
                                    if (f == 5) {
                                        std::cout << "f=" << f << " nelements[f]=" << nelements[f] << std::endl;
                                        /* buf32 is modified for binPack */
                                        // std::cout << "buf32[0]=" << buf32[0]
                                        //     << ", buf32[1]=" << buf32[1]
                                        //     << ", buf32[2]=" << buf32[2]
                                        //     << ", buf32[3]=" << buf32[3] << std::endl;
                                        /* buf_out contains the packed data */
                                        std::cout << "buf_out[0]=" << buf_out[0]
                                            << ", buf_out[1]=" << buf_out[1]
                                            << ", buf_out[2]=" << buf_out[2]
                                            << ", buf_out[3]=" << buf_out[3] << std::endl;
                                        printf("buf_out[4]=%u, buf_out[5]=%x, buf_out[6]=%x, buf_out[7]=%x\n",
                                            buf_out[4], buf_out[5], buf_out[6], buf_out[7]);
                                        // std::cout << "buf_out[4]=" << buf_out[4]
                                        //     << ", buf_out[5]=" << buf_out[5]
                                        //     << ", buf_out[6]=" << buf_out[6]
                                        //     << ", buf_out[7]=" << buf_out[7] << std::endl;
                                        exit(1);
                                        #if 0
                                        std::cout << " ==== " << std::endl;
                                        for (uint z = 0; z < offset; z++) {
                                          if (z > 0 && z % 8 == 0) std::cout << std::endl;
                                          std::cout << buf_out[z] << " ";
                                        }
                                        std::cout << " ==== " << std::endl;
                                        #endif
                                    }
                                    #endif
                                    #ifdef DEBUG_AND_STOP
                                    if (f == 5) {
                                        binUnpack(buf_out, task.offsets_local, nelements[f]);

                                        // std::cout << " ==== " << std::endl;
                                        // for (uint z = 0; z < offset; z++) {
                                        //   if (z > 0 && z % 8 == 0) std::cout << std::endl;
                                        //   std::cout << buf_out[z] << " ";
                                        // }
                                        // std::cout << " ==== " << std::endl;
                                        exit(1);
                                    }
                                    #endif
                                    size *= sizeof(int32_t);
                                    task.offsets[f].insert(task.offsets[f].end(), task.offsets_local, task.offsets_local + last_index + 1);

                                    #ifdef DEBUG_AND_STOP 
                                    if (f == 5) {
                                        std::cout
                                            << "[DEBUG] last_index=" << last_index
                                            << ", " << task.offsets_local[last_index - 1]
                                            << std::endl;
                                    }
                                    #endif
                                    write_size = roundup512(size);
                                    assert(write_size <= page_size);
                                    task.compressed_page_sizes[f].push_back(write_size);
                                } else {
                                    std::cerr << "Unsupported field size: " << field_size << "\n";
                                    return { -1, nios, nwritten_sectors };
                                }
                                nwritten_sectors[f] += write_size / 512;
                                page_pwrite_comp_host(output_fds, task.buf_compress, pagid, write_size, page_size);
                                //#ifdef DEBUG_AND_STOP 
                                #if 0
                                if (f == 5) {
                                    std::cout << "buf_compress[0]=" << reinterpret_cast<uint*>(task.buf_compress)[0]
                                        << ", buf_compress[1]=" << reinterpret_cast<uint*>(task.buf_compress)[1]
                                        << ", buf_compress[2]=" << reinterpret_cast<uint*>(task.buf_compress)[2]
                                        << ", buf_compress[3]=" << reinterpret_cast<uint*>(task.buf_compress)[3]
                                        << ", write_size=" << write_size
                                        << std::endl;
                                }
                                #endif
                                #if 0
                                if (f == 5 && pagid == 154) {
                                    page_pread_comp_host(output_fds, task.buf_compress, pagid, write_size, page_size);
                                    binUnpack(reinterpret_cast<uint*>(task.buf_compress), task.offsets_local, nelements[f]);
                                }
                                #endif
                            } else {
                                page_pwrite_host(output_fds, buf, pagid, page_size);
                            }
                            nios[f]++;
                            // nwritten_sectors[f] += page_size / 512;
                        }
 
                        offsets[f] = 0; // Reset offset after writing
                        ++pages[f]; // Increment page id after writing
                    }

                    f++;
                }
                start = end + 1;
                i++;
            }
            // fields.emplace_back(sv.substr(start)); // 最後のフィールド
            // for (const auto& field : fields) {
            //     std::cout << "[" << field << "] ";
            // }
            // std::cout << "\n";
            ++nlines_processed;
            if (nlines_processed >= task.nlines_to_read) {
                // std::cout << "Reached the end of file: " << path << std::endl;
                break;
            }
        }
        file_index++;
    } while (nlines_processed < task.nlines_to_read);
    for (size_t i = 0; i < N; ++i) {
        if (offsets[i] > 0) {
            // Write remaining data to disk.
            char *buf = reinterpret_cast<char*>(buffers[i]);
            uint64_t pagid = pages[i];
#ifdef DEBUG_PRINT
            // std::cout << "Remaining data in a buffer for field " << i << ", writing to disk. (pageid=" << pagid << ")\n";
            if (i == 0) {
                std::cerr << "Remaining data in a buffer for field " << i << ", writing to disk. (pageid=" << pagid << ")\n";
            }
            // std::cerr << "Remaining data in a buffer for field " << i << ", writing to disk. (pageid=" << pagid << ")\n";
#endif
            if (dryrun) {
                std::cout << "Remaining buffer for field " << i << ", writing to disk. (pageid=" << pagid << ")\n";
            } else {
                if (verbose) {
                    std::cout << "Remaining buffer for field " << i << ", writing to disk. (pageid=" << pagid << ")\n";
                }

                ulong *buf64 = reinterpret_cast<ulong*>(buffers[i]);
                uint *buf32 = reinterpret_cast<uint*>(buffers[i]);
                size_t field_size = target_fields_sizes[i];
                if (task.compress) {
                    /* offsets[i] > 0, so this is valid. */
                    /* offsets[i] perverves the number of int32/int64 values in the buffer. */
                    /* Not to interfere the tail number of records in the final page, */
                    /* binPack/binPack64 function should accept an input array whose */
                    /* the number of elements is multiple of 512. */
                    size_t last_value_off = offsets[i] - 1;
                    size_t n = roundup512(offsets[i]);
                    bool padding = false;
                    for (size_t j = offsets[i]; j < n; ++j) {
                        if (target_fields_sizes[i] == sizeof(int64_t)) {
                            //buf64[last_value_off + j] = buf64[last_value_off];
                            buf64[j] = buf64[last_value_off];
                        } else if (target_fields_sizes[i] == sizeof(int32_t)) {
                            //buf32[last_value_off + j] = buf32[last_value_off];
                            buf32[j] = buf32[last_value_off];
                        }
                        padding = true;
                    }
                    if (padding) {
                        std::cout << "\tPadding occured when writing to disk."
                            << "nblocks=" << n / 128 << " in the last page. "
                            << "(pageid=" << pagid << ")\n";
                    }

                    size_t size, write_size;
                    size_t comp_block_size = 128;
                    // size_t last_index = offsets[i] / comp_block_size;
                    size_t last_index = n / comp_block_size;
                    memset(task.buf_compress, 0, page_size);
                    std::array<uint, 32> workspace;
                    if (field_size == sizeof(int64_t)) {
                        auto buf_out = reinterpret_cast<ulong*>(task.buf_compress);
                        size = binPack64(buf64, buf_out, task.offsets_local, n, workspace.data());
                        size *= sizeof(int64_t);
                        task.offsets[i].insert(task.offsets[i].end(), task.offsets_local, task.offsets_local + last_index + 1);
                        write_size = roundup512(size);
                        task.compressed_page_sizes[i].push_back(write_size);
                    } else if (field_size == sizeof(int32_t)) {
                        auto buf_out = reinterpret_cast<uint*>(task.buf_compress);
                        size = binPack(buf32, buf_out, task.offsets_local, n, workspace.data());
                        size *= sizeof(int32_t);
                        task.offsets[i].insert(task.offsets[i].end(), task.offsets_local, task.offsets_local + last_index + 1);
                        write_size = roundup512(size);
                        task.compressed_page_sizes[i].push_back(write_size);
                    } else {
                        std::cerr << "Unsupported field size: " << field_size << "\n";
                        return { -1, nios, nwritten_sectors };
                    }
                    nwritten_sectors[i] += write_size / 512;
                    // page_pwrite_host(output_fds, task.buf_compress, pagid, write_size);
                    page_pwrite_comp_host(output_fds, task.buf_compress, pagid, write_size, page_size);
                } else {
                    page_pwrite_host(output_fds, buf, pagid, page_size);
                }
                nios[i]++;
            }
 
            offsets[i] = 0; // Reset offset after writing
        }
    }
 
    return { nlines_processed, nios, nwritten_sectors };
}

template<typename ReturnType>
ReturnType loader_alinged_alloc(size_t size) {
    size_t alignment = 4096; // 4KB alignment
    void* ptr = nullptr;

    int res = posix_memalign(&ptr, alignment, size);
    if (res != 0) {
        std::cerr << "posix_memalign failed: " << res << "\n";
        std::exit(EXIT_FAILURE);
    }
    return static_cast<ReturnType>(ptr);
}


/* The storage space for dimension table can be calcurated decisivelly, so no need to return the next page id. */
template <typename EnumType, size_t N>
void load_fact_table(LoaderOptions &options, SSBTableMetadata &metadata,
    std::vector<FactTableLoadTask<N>>& tasks,  std::vector<std::string>& paths,
    const std::array<char*, N>& buffers, const std::array<uint64_t, N>& field_start_page_ids,
    const std::array<size_t, N>& target_fields_sizes, const std::array<EnumType, N>& target_fields,
    StatEntry &entry) {

    std::vector<std::future<LoaderThreadStats<N>>> futures;
    std::vector<std::thread> threads;
    std::vector<size_t> nitems_loaded;
    std::vector<std::array<char*, N>> thread_specific_buffers;

    /* prep buffers for each threads */
    for (size_t i = 0; i < tasks.size(); ++i) {
        std::array<char*, N> buf;
        for (size_t j = 0; j < N; ++j) {
            buf[j] = &buffers[j][i * options.page_size];
        }
        thread_specific_buffers.push_back(buf);
    }

    for (size_t i = 0; i < tasks.size(); ++i) {
    // Debug, single thread
    // for (size_t i = 0; i < 1; ++i) {
        auto load_func = static_cast<LoaderThreadStats<N>(*)(
            FactTableLoadTask<N>&, std::vector<std::string>& ,
            std::vector<int>&,
            const std::array<char*, N>&, const std::array<uint64_t, N>&,
            const std::array<size_t, N>&,
            const std::array<EnumType, N>& target_fields, const size_t,
            const int, const int)>
            (&load_fact_table_lines<EnumType>);

        std::packaged_task<LoaderThreadStats<N>()> task(
            std::bind(load_func,
                std::ref(tasks[i]), std::ref(paths), std::ref(options.output_fds),
                std::ref(thread_specific_buffers[i]), std::ref(field_start_page_ids),
                std::ref(target_fields_sizes), std::ref(target_fields),
                options.page_size, options.dryrun, options.verbose
        ));

        futures.push_back(task.get_future());
        threads.emplace_back(std::move(task));
    }

    for (auto& t : threads) {
        t.join();
    }

    size_t count_sum = 0;
    std::array<size_t, N> total_nios = {};
    std::array<size_t, N> total_nwritten_sectors = {};
    // std::vector<size_t> prefix_sum;
    // prefix_sum.reserve(paths.size() + 1);
    // prefix_sum.push_back(0);
    for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        ssize_t count = result.nlines_processed;
        if (count < 0) {
            std::cerr << "[FATAL] Failed to count lines in file: " << paths[i] << std::endl;
            exit(EXIT_FAILURE);
        }
        count_sum += count;
        for (size_t f = 0; f < N; ++f) {
            total_nios[f] += result.nios[f];
            total_nwritten_sectors[f] += result.nwritten_sectors[f];
        }
    }

    for (size_t f = 0; f < N; ++f) {
        entry.metrics[f].fields_written = total_nios[f];
        entry.metrics[f].fields_written_compressed = total_nwritten_sectors[f];
    }
    std::cout << "Loaded " << count_sum << " records." << std::endl;
}

/* The storage space for dimension table can be calcurated decisivelly, so no need to return the next page id. */
template <typename EnumType, size_t N>
void load_dimension_table(LoaderOptions &options, SSBTableMetadata &metadata, std::string& filename,
    const std::array<char*, N>& buffers, const std::array<uint64_t, N>& field_start_page_ids,
    const std::array<size_t, N>& target_fields_sizes, const std::array<EnumType, N>& target_fields,
    StatEntry &entry)
{
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file.\n";
        return;
    }

    std::array<size_t, N> offsets;
    std::array<size_t, N> nelements;
    std::array<uint64_t, N> pages;
    std::array<size_t, N> nios;
    for (size_t i = 0; i < N; ++i) {
        offsets[i] = 0;
        nelements[i] = options.page_size / target_fields_sizes[i];
        pages[i] = field_start_page_ids[i];
        nios[i] = 0;
#ifdef DEBUG_PRINT
        std::cout << "Field " << i << ": size = " << target_fields_sizes[i] << ", nelements = " << nelements[i] << ", page = " << pages[i] << "\n";
#endif
    }
    std::string line;
    while (std::getline(file, line)) {
        // std::vector<std::string_view> fields;
        std::string_view sv(line);
        size_t i = 0;
        size_t f = 0;
        size_t start = 0;
        size_t end;
        size_t offset;
        uint64_t pagid;
        int32_t value;
        size_t field_size;
        int32_t *buf;

        while (((end = sv.find('|', start)) != std::string_view::npos) && (f < N)) {
            // std::cout << "Processing field: " << f << "\n";
            if (i < target_fields[f]) {
                // skip
            } else {
                std::string_view part = sv.substr(start, end - start);
                auto [ptr, ec] = std::from_chars(part.data(), part.data() + part.size(), value);

#ifdef DEBUG_PRINT
                if (ec == std::errc()) {
                    std::cout << "Parsed number: " << value << "\n";
                } else {
                    std::cerr << "Failed to parse integer from: " << part << "\n";
                }
#endif
                if (ec != std::errc()) {
                    std::cerr << "Failed to parse integer from: " << part << "\n";
                    return;
                }
                field_size = target_fields_sizes[f];
                buf = reinterpret_cast<int32_t*>(buffers[f]);
                offset = offsets[f];
                start = end + 1;

                buf[offset] = value;
                offsets[f]++;
                if (offsets[f] >= nelements[f]) {
                    // Write buffer to disk or process it as needed
                    pagid = pages[f];
                    if (options.dryrun) {
                        std::cout << "Buffer full for field " << f << ", writing to disk. (pageid=" << pagid << ")\n";
                    } else {
                        if (options.verbose) {
                            std::cout << "Buffer full for field " << f << ", writing to disk. (pageid=" << pagid << ")\n";
                        }
                        page_pwrite_host(options.output_fds, buf, pagid, options.page_size);
                        nios[f]++;
                    }
                    offsets[f] = 0; // Reset offset after writing
                    pages[f]++; // Reset offset after writing
                }

                f++;
            }
            start = end + 1;
            i++;
        }
        // fields.emplace_back(sv.substr(start)); // 最後のフィールド
        // for (const auto& field : fields) {
        //     std::cout << "[" << field << "] ";
        // }
        // std::cout << "\n";
    }
    for (size_t i = 0; i < N; ++i) {
        if (offsets[i] > 0) {
            std::cout << "Writing remaining data for field " << i << "...\n";
            // Write remaining data to disk or process it as needed
            uint64_t pagid = pages[i];
            char *buf = reinterpret_cast<char*>(buffers[i]);
            if (options.dryrun) {
                std::cout << "Remaining data in a buffer for field " << i << ", writing to disk. (pageid=" << pagid << ")\n";
            } else {
                if (options.verbose) {
                    std::cout << "Remaining data in a buffer for field " << i << ", writing to disk. (pageid=" << pagid << ")\n";
                }
                page_pwrite_host(options.output_fds, buf, pagid, options.page_size);
                nios[i]++;
            }
            offsets[i] = 0; // Reset offset after writing
        }
    }

    // Update stats
    for (size_t i = 0; i < N; ++i) {
        entry.metrics[i].fields_written += ((nios[i] * options.page_size) / MEBI);
    }
}

template <typename EnumType, size_t N>
void bulkload(LoaderOptions &options, SSBTableMetadata &metadata, const SSB::common::Table table,
    const std::array<char*, N>& buffers, const std::array<uint64_t, N>& field_start_page_ids,
    const std::array<size_t, N>& field_sizes, const std::array<EnumType, N>& target_fields,
    StatEntry &stat_entry) {

    std::stringstream ss;
    if (table == SSB::common::DDATE) {
        ss << options.input_dirname << "/" << SSB::common::table_name(table) << ".tbl";
    } else {
        ss << options.input_dirname << "/" << SSB::common::table_name(table) << ".tbl.p";
    }
    std::string path = ss.str();

    switch (table) {
        case SSB::common::SUPPLIER:
        case SSB::common::PART:
        case SSB::common::CUSTOMER:
        case SSB::common::DDATE:
#ifdef DEBUG_PRINT
            std::cout << "Loading " << SSB::common::table_name(table) << " table...\n";
#endif
            load_dimension_table<EnumType>(options, metadata, path, buffers,
                field_start_page_ids, field_sizes, target_fields, stat_entry);
#ifdef DEBUG_PRINT
            std::cout << "Done (" << SSB::common::table_name(table) << " table).\n";
#endif
            break;
        default:
            std::cerr << "Unknown table type.\n";
            return;
    }
}

template <size_t N>
void bulkload_fact(LoaderOptions &options, SSBTableMetadata &metadata, const SSB::common::Table table,
    const std::array<char*, N>& buffers, const std::array<uint64_t, N>& field_start_page_ids,
    const std::array<size_t, N>& field_sizes, const std::array<SSB::common::LineOrderField, N>& target_fields,
    std::array<std::vector<int32_t>, N>& field_compression_offsets,
    std::array<std::vector<int32_t>, N>& field_compressed_page_size, StatEntry &stat_entry)
    // StatEntry &stat_entry)
{
    if (table != SSB::common::LINEORDER) {
        std::cerr << "Unknown table type.\n";
        return;
    }
#ifdef DEBUG_PRINT
    std::cout << "Loading lineorder table...\n";
#endif
    {
        auto paths = prep_input_files_lineorder(options);
        std::cout << "Number of lineorder files: " << paths.size() << std::endl;
        auto tasks = generate_lineorder_tasks(options.page_size, options.compress, paths, field_sizes);
        load_fact_table<SSB::common::LineOrderField>(options, metadata, tasks, paths, buffers,
            field_start_page_ids, field_sizes, target_fields, stat_entry);

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
#if 1
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
#endif
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

                size_t npages = SSB::metadata_noffsets_to_nmetapages(offsets.size(), options.page_size);
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

                size_t npages = SSB::metadata_ncomp_pages_to_nmetapages(compressed_page_sizes.size(), options.page_size);
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

        // loader_info infos = check_nlines_parallal();
        // load_parallal(options, paths, infos);

        // size_t nfiles = count_lineorder_files(options.input_dirname);
        // load_fact_table<SSB::common::LineOrderField>(options, metadata, path, buffers,
        //     field_start_page_ids, field_sizes, target_fields);
    }
#ifdef DEBUG_PRINT
    std::cout << "done.n";
#endif
}
#endif

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
    char *devarg = strndup(options.output_files, 16384);
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
    for (size_t i = 0; i < options.ndev; i++)
    {
        free(options.devname[i]);
    }

    for (auto fd : fds)
    {
        if (fd >= 0)
        {
            close(fd);
        }
    }
}

void load_test_binpack(ulong N, uint *in_values, uint *out_values, uint *offsets, uint *decoded_values)
{
    if (N % 128 != 0) {
        std::cerr << "N must be a multiple of 128.\n";
        return;
    }
    /* block_size = 128 */
    std::array<uint, 32> workspace;
    binPack(in_values, out_values, offsets, N, workspace.data());
    binUnpack(out_values, offsets, N);

    return;
}