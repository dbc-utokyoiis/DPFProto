#pragma once

#include <fcntl.h>
#include <linux/fs.h>
#include <numa.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <random>
#include <thread>
#include <utility>
#include <vector>
#include <ranges>

#include "cpu_usage.cu"
#include "gpu_usage.cu"

namespace chrono = std::chrono;
namespace fs = std::filesystem;

const size_t KIBI = 1024UL;
const size_t MEBI = 1024UL * 1024UL;
const size_t GIBI = 1024UL * 1024UL * 1024UL;
const size_t TEBI = 1024UL * 1024UL * 1024UL * 1024UL;

enum class TransferType
{
    // I/O
    HOST,              // Storage <-(CPU Sync I/O)-> CPU
    HOST_ASYNC,        // Storage <-(CPU Async I/O)-> CPU
    HOST_DEVICE,       // Storage <-(CPU Sync I/O)-> CPU <-(Sync Memcpy)-> GPU
    HOST_DEVICE_ASYNC, // Storage <-(CPU Sync I/O)-> CPU <-(Async Memcpy)-> GPU
    HOST_ASYNC_DEVICE, // Storage <-(CPU Async I/O)-> CPU <-(Sync Memcpy)-> GPU
    DEVICE,            // Storage <-(GPU Sync I/O)-> GPU
    DEVICE_BATCH,      // Storage <-(GPU Batch I/O)-> GPU
    DEVICE_BATCH_SYNC, // Storage <-(GPU Sync Batch I/O)-> GPU
    DEVICE_BATCH_SYNC_COMPRESSED, // Storage <-(GPU Sync Batch I/O)-> GPU
    DEVICE_BAM, // Storage <-(GPU Sync Batch I/O)-> GPU
    DEVICE_ASYNC,      // Storage <-(GPU Async I/O)-> GPU

    // Transfer
    TRANSFER,       // CPU <-(Sync Memcpy)-> GPU
    TRANSFER_ASYNC, // CPU <-(Async Memcpy)-> GPU
};

#if 0
enum class BenchmarkType
{
    NONE,

    // I/O
    SEQUENTIAL_READ,
    SEQUENTIAL_WRITE,
    RANDOM_READ,
    RANDOM_WRITE,

    // Transfer
    HOST_TO_DEVICE,
    DEVICE_TO_HOST,
};
#endif


namespace TPCH {
    enum class QueryId {
        Q1,
        Q2,
        Q3,
        Q4,
        Q5,
        Q6,
        Q7,
        Q8,
        Q9,
        Q10,
        Q11,
        Q12,
        Q13,
        Q14,
        Q15,
        Q16,
        Q17,
        Q18,
        Q19,
        Q20,
        Q21,
        Q22,
        CHECK11,
        CHECK12,
        CHECK13,
        CHECKMETA,
        TEST_PFOR_PAGE,
        TEST_PFOR64_PAGE,
        REVENUE,
        SCAN_O_COMMENT,
        SCAN_O_COMMENT_V2,
        SCAN_O_COMMENT_V3,
        TEST_LZ4_PAGE,
        SCAN_O_COMMENT_V4,
        SCAN_O_COMMENT_V5,
        SCAN_O_COMMENT_V6,
        SCAN_L_COMMENT_V7,
        SCAN_L_COMMENT_V8,
        SCAN_L_COMMENT,
        IO_BENCH,
        DECOMP_BENCH,
        DECOMP_KMP_BENCH,
        Q3SEL,
        NONE,
    };
}

namespace SSB {
    enum class Query {
        NONE,
        Q11,
        Q12,
        Q13,
        Q21,
        Q22,
        Q23,
        Q31,
        Q32,
        Q33,
        Q34,
        Q41,
        Q42,
        Q43,
        REVENUE,
        CHECK,
        NUM_QUERIES
    };
}

namespace Microbenchmark {
    enum class Workload {
        NONE,
        // I/O
        SEQUENTIAL_READ,
        SEQUENTIAL_WRITE,
        RANDOM_READ,
        RANDOM_WRITE,

        // Transfer
        HOST_TO_DEVICE,
        DEVICE_TO_HOST,

    };
}

namespace Benchmark {
    enum class Kind {
        NONE,
        SSB,
        TPCH,
        MICROBENCHMARK,
        NUM_BENCHMARKS
    };

    enum class TransferType {
        NONE,
        HOST,              // Storage <-(CPU Sync I/O)-> CPU
        // HOST_ASYNC,        // Storage <-(CPU Async I/O)-> CPU
        // HOST_DEVICE,       // Storage <-(CPU Sync I/O)-> CPU <-(Sync Memcpy)-> GPU
        // HOST_DEVICE_ASYNC, // Storage <-(CPU Sync I/O)-> CPU <-(Async Memcpy)-> GPU
        HOST_ASYNC_DEVICE, // Storage <-(CPU Async I/O)-> CPU <-(Sync Memcpy)-> GPU
        DEVICE,            // Storage <-(GPU Sync I/O)-> GPU
        DEVICE_BATCH,      // Storage <-(GPU Batch I/O)-> GPU
        DEVICE_BATCH_SYNC, // Storage <-(GPU Sync Batch I/O)-> GPU
        DEVICE_PRELOAD,    // Storage <-(GPU Preload I/O)-> GPU
        /* Following modes can be specified */
        GIDP,               // GPU In-Data-Path (cuFile sync/batch I/O)
        GIDP_BAM,           // GIDP + BaM (GPU-initiated I/O)
        GIDP_BAM_FUSION,    // GIDP + BaM + fused IO/decompression/query
        DATAPATHFUSION,      // Data Path Fusion (fused IO+decomp+scan)
    };

    struct Job {
        Kind kind;
        TransferType type;
        union {
            SSB::Query b1;
            Microbenchmark::Workload b2;
            TPCH::QueryId b3;
        };

        Job(SSB::Query ssbq, TransferType t = TransferType::HOST)
            : kind(Kind::SSB), type(t), b1(ssbq) {}
        Job(Microbenchmark::Workload workload, TransferType t = TransferType::HOST)
            : kind(Kind::MICROBENCHMARK), type(t), b2(workload) {}
        Job(TPCH::QueryId tpchq, TransferType t = TransferType::HOST)
            : kind(Kind::TPCH), type(t), b3(tpchq) {}
        Job() = delete;
    };

    std::string to_string(TransferType t) {
        switch (t) {
            case TransferType::HOST: return "HOST";
            case TransferType::DEVICE: return "DEVICE";
            case TransferType::DEVICE_BATCH: return "DEVICE_BATCH";
            case TransferType::DEVICE_BATCH_SYNC: return "DEVICE_BATCH_SYNC";
            case TransferType::GIDP: return "GIDP";
            case TransferType::GIDP_BAM: return "GIDP_BAM";
            case TransferType::GIDP_BAM_FUSION: return "GIDP_BAM_FUSION";
            case TransferType::DATAPATHFUSION: return "DATAPATHFUSION";
            default: return "UNKNOWN";
        }
    }

    inline std::string to_string(const Job& job) {
        std::stringstream ss;
        switch (job.kind) {
            case Kind::NONE: 
                ss << "NONE:";
                break;
            case Kind::SSB:
                ss << "SSB:";
                ss << to_string(job.b1);
                ss << ":";
                break;
            case Kind::TPCH:
                ss << to_string(job.b3);
                ss << ":";
            case Kind::MICROBENCHMARK:
                ss << to_string(job.b2);
                ss << ":";
            default:
                ss << "UNKNOWN_BENCHMARK_KIND";
                break;
        }
        ss << to_string(job.type);
        return ss.str();
    }

    inline bool is_query(const Job& j, SSB::Query target) {
        return (j.kind == Kind::SSB) && j.b1 == target;
    }

    inline bool is_query(const Job& j, TPCH::QueryId target) {
        return (j.kind == Kind::TPCH) && j.b3 == target;
    }

    inline bool is_workload(const Job& j, Microbenchmark::Workload target) {
        return j.kind == Kind::MICROBENCHMARK && j.b2 == target;
    }

    inline bool has_none(const Job& j) {
        return (j.kind == Kind::NONE ||
            (j.kind == Kind::SSB && j.b1 == SSB::Query::NONE) ||
            (j.kind == Kind::MICROBENCHMARK && j.b2 == Microbenchmark::Workload::NONE));
    }

    inline bool is_read_workload(const Job& j) {
        return j.kind == Kind::MICROBENCHMARK &&
            (j.b2 == Microbenchmark::Workload::SEQUENTIAL_READ ||
             j.b2 == Microbenchmark::Workload::RANDOM_READ);
    }

    inline bool is_write_workload(const Job& j) {
        return j.kind == Kind::MICROBENCHMARK &&
            (j.b2 == Microbenchmark::Workload::SEQUENTIAL_WRITE ||
             j.b2 == Microbenchmark::Workload::RANDOM_WRITE);
    }

    inline bool is_sequential_workload(const Job& j) {
        return j.kind == Kind::MICROBENCHMARK &&
            (j.b2 == Microbenchmark::Workload::SEQUENTIAL_WRITE ||
             j.b2 == Microbenchmark::Workload::SEQUENTIAL_READ);
    }

    inline bool is_random_workload(const Job& j) {
        return j.kind == Kind::MICROBENCHMARK &&
            (j.b2 == Microbenchmark::Workload::RANDOM_WRITE ||
             j.b2 == Microbenchmark::Workload::RANDOM_READ);
    }

    inline bool is_transfertype(const Job& j, Benchmark::TransferType target) {
        return j.type == target;
    }
}

template <typename E>
struct enum_traits;

template <>
struct enum_traits<Benchmark::Job> {
    static std::string to_string(SSB::Query q) {
        switch (q) {
            case SSB::Query::Q11: return "Q11";
            case SSB::Query::Q12: return "Q12";
            case SSB::Query::Q13: return "Q13";
            case SSB::Query::Q21: return "Q21";
            case SSB::Query::Q22: return "Q22";
            case SSB::Query::Q23: return "Q23";
            case SSB::Query::Q31: return "Q31";
            case SSB::Query::Q32: return "Q32";
            case SSB::Query::Q33: return "Q33";
            case SSB::Query::Q34: return "Q34";
            case SSB::Query::Q41: return "Q41";
            case SSB::Query::Q42: return "Q42";
            case SSB::Query::Q43: return "Q43";
            case SSB::Query::CHECK: return "CHECK";
            default: return "UNKNOWN";
        }
    }
    static std::string to_string(TPCH::QueryId q) {
        switch (q) {
            case TPCH::QueryId::Q1: return "Q1";
            case TPCH::QueryId::Q2: return "Q2";
            case TPCH::QueryId::Q3: return "Q3";
            case TPCH::QueryId::Q4: return "Q4";
            case TPCH::QueryId::Q5: return "Q5";
            case TPCH::QueryId::Q6: return "Q6";
            case TPCH::QueryId::Q7: return "Q7";
            case TPCH::QueryId::Q8: return "Q8";
            case TPCH::QueryId::Q9: return "Q9";
            case TPCH::QueryId::Q10: return "Q10";
            case TPCH::QueryId::Q11: return "Q11";
            case TPCH::QueryId::Q12: return "Q12";
            case TPCH::QueryId::Q13: return "Q13";
            case TPCH::QueryId::Q14: return "Q14";
            case TPCH::QueryId::Q15: return "Q15";
            case TPCH::QueryId::Q16: return "Q16";
            case TPCH::QueryId::Q17: return "Q17";
            case TPCH::QueryId::Q18: return "Q18";
            case TPCH::QueryId::Q19: return "Q19";
            case TPCH::QueryId::Q20: return "Q20";
            case TPCH::QueryId::Q21: return "Q21";
            case TPCH::QueryId::Q22: return "Q22";
            case TPCH::QueryId::CHECK11: return "CHECK11_C_NAME";
            case TPCH::QueryId::CHECK12: return "CHECK11_C_ADDRESS";
            case TPCH::QueryId::CHECK13: return "CHECK11_C_COMMENT";
            case TPCH::QueryId::TEST_PFOR_PAGE: return "TEST_PFOR_PAGE";
            case TPCH::QueryId::TEST_PFOR64_PAGE: return "TEST_PFOR64_PAGE";
            case TPCH::QueryId::SCAN_O_COMMENT: return "SCAN_O_COMMENT";
            case TPCH::QueryId::TEST_LZ4_PAGE: return "TEST_LZ4_PAGE";
            case TPCH::QueryId::SCAN_O_COMMENT_V4: return "SCAN_O_COMMENT_V4";
            case TPCH::QueryId::SCAN_O_COMMENT_V5: return "SCAN_O_COMMENT_V5";
            case TPCH::QueryId::SCAN_O_COMMENT_V6: return "SCAN_O_COMMENT_V6";
            case TPCH::QueryId::SCAN_L_COMMENT_V7: return "SCAN_L_COMMENT_V7";
            case TPCH::QueryId::SCAN_L_COMMENT_V8: return "SCAN_L_COMMENT_V8";
            case TPCH::QueryId::SCAN_L_COMMENT: return "SCAN_L_COMMENT";
            default: return "UNKNOWN";
        }
    }
};

template <>
struct enum_traits<Microbenchmark::Workload> {
    static std::string to_string(Microbenchmark::Workload w) {
        switch (w) {
            case Microbenchmark::Workload::SEQUENTIAL_READ: return "SEQUENTIAL_READ";
            case Microbenchmark::Workload::SEQUENTIAL_WRITE: return "SEQUENTIAL_WRITE";
            case Microbenchmark::Workload::RANDOM_READ: return "RANDOM_READ";
            case Microbenchmark::Workload::RANDOM_WRITE: return "RANDOM_WRITE";
            default: return "UNKNOWN";
        }
    }
};

template <>
struct enum_traits<Benchmark::TransferType> {
    static std::string to_string(Benchmark::TransferType t) {
        switch (t) {
            case Benchmark::TransferType::HOST: return "HOST";
            case Benchmark::TransferType::DEVICE: return "DEVICE";
            case Benchmark::TransferType::DEVICE_BATCH: return "DEVICE_BATCH";
            case Benchmark::TransferType::DEVICE_BATCH_SYNC: return "DEVICE_BATCH_SYNC";
            default: return "UNKNOWN";
        }
    }
};

template <>
struct enum_traits<Benchmark::Kind> {
    static std::string to_string(Benchmark::Kind k) {
        switch (k) {
            case Benchmark::Kind::NONE: return "NONE";
            case Benchmark::Kind::SSB: return "SSB";
            case Benchmark::Kind::MICROBENCHMARK: return "MICROBENCHMARK";
            case Benchmark::Kind::TPCH: return "TPCH";
            default: return "UNKNOWN";
        }
    }
};

enum class OutputFormat
{
    TEXT,
    JSON,
};

struct BenchmarkOptions
{
    char const *file;
    char *devname[128];
    size_t ndev;
    const size_t nthreads;
    size_t io_multiplicity;
    size_t io_size;
    size_t page_size; /* NOTE: removed */
    size_t sub_page_size; /* NOTE: removed */
    size_t period_sec;
    size_t file_size; /* NOTE: removed */
    size_t large_page_size;
    size_t gds_num_handlers_per_thread;
    bool enable_prefetch;
    char const *benchmark_kind_str;
    char const *query_str;
    TPCH::QueryId query;
    char const *transfer_type_str;
    Benchmark::Job job;
    OutputFormat output_format;
    bool enable_zonemap;
    bool use_sync_io;
    int32_t q6_sd_low;   // Q6 L_SHIPDATE lower bound (inclusive), default 19940101
    int32_t q6_sd_high;  // Q6 L_SHIPDATE upper bound (exclusive), default 19950101
    uint32_t coalesce_k; // BaM I/O coalescing factor (1 = per-page, >1 = coalesced)
    bool use_prescan;    // prescan mode: pre-compute I/O plan in separate kernel
    bool use_prefix_sum; // use prefix_sum for VCHAR page mapping (GOLAP Q13)
    uint32_t block_size; // thread block size: 32 (default) or 128 (4-warp parallel decode)
    int32_t revenue_qt_max; // Revenue query: L_QUANTITY < qt_max (0 = no filter)
    int32_t q3sel_selectivity; // Q3SEL: selectivity percentage (20/40/60/80/100), 0 = normal Q3
    bool disable_other_filters; // -F: disable non-selectivity filters (old revenue/q3sel behavior)
};

struct BenchmarkResult
{
    size_t nios;
    uint64_t read_bytes;         // total bytes read from storage (0 = use nios*io_size)
    int64_t elapsed_nanoseconds;
    CpuUsage cpu_usage;
    GpuUsage gpu_usage;
    uint32_t num_thread_blocks;  // actual CUDA blocks launched (0 = not set)
    std::string compression;     // compression method(s) used (e.g. "NONE", "SNAPPY")
    uint64_t gpu_mem_bytes;      // peak GPU memory allocated by query (0 = not measured)
    uint64_t gpu_ctrl_bytes;     // GPU memory consumed by BAM ctrl/QPs (PiG only, 0 = N/A)
    uint64_t gpu_app_bytes;      // GPU memory consumed by app buffers + page caches (PiG only, 0 = N/A)
    uint64_t total_pages;        // total page reads across all columns, before IO pruning (0 = not set)
    uint64_t kernel_launches;    // number of host-side CUDA kernel launches (incl. nvCOMP)
};


inline uint64_t ns_to_sec(uint64_t ns)
{
    return static_cast<uint64_t>(ns) / 1'000'000'000;
}

inline uint64_t ns_to_msec(uint64_t ns)
{
    return static_cast<uint64_t>(ns) / 1'000'000;
}

inline uint64_t ns_to_sub_msec(uint64_t ns)
{
    return (static_cast<uint64_t>(ns) % 1'000'000 / 1'000);
}



int open_file(const BenchmarkOptions &options)
{
    int oflag = 0;
    char *files = strdup(options.file);
    if (Benchmark::is_read_workload(options.job))
    {
        oflag = O_RDONLY | O_DIRECT;
    }
    else if (Benchmark::is_write_workload(options.job))
    {
        oflag = O_CREAT | O_WRONLY | O_DIRECT;
    }

    int fd = open(options.file, oflag, 0644);
    if (fd < 0)
    {
        std::cerr << "failed to open file " << options.file << std::endl;
        perror("open");
        close(fd);
        exit(EXIT_FAILURE);
    }

    return fd;
}

void open_files(BenchmarkOptions &options, std::vector<int> &fds)
{
    int oflag = 0;
    // if (options.benchmark_type == BenchmarkType::SEQUENTIAL_READ ||
    //     options.benchmark_type == BenchmarkType::RANDOM_READ)
    // {
    //     oflag = O_RDONLY | O_DIRECT;
    // }
    // else if (options.benchmark_type == BenchmarkType::SEQUENTIAL_WRITE ||
    //          options.benchmark_type == BenchmarkType::RANDOM_WRITE)
    // {
    //     oflag = O_CREAT | O_WRONLY | O_DIRECT;
    // }
    oflag = O_RDWR | O_DIRECT;

    int i = 0;
    char *devarg = strndup(options.file, 16384);
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

    // std::cout << "ndev: " << options.ndev << std::endl;
    // for (i = 0; i < options.ndev; i++)
    // {
    //     std::cout << "dev[ " << i << "]: " << options.devname[i] << std::endl;
    // }

    fds.reserve(options.ndev);
    for (i = 0; i < options.ndev; i++)
    {
        int fd = open(options.devname[i], oflag, 0644);
        if (fd < 0)
        {
            std::cerr << "failed to open file " << options.file << std::endl;
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

void close_files(BenchmarkOptions &options, std::vector<int> &fds)
{
    for (auto fd : fds)
    {
        if (fd >= 0)
        {
            close(fd);
        }
    }
    free(options.devname[0]);
}

#define gettid() syscall(SYS_gettid)

static void cpu_set_affinity(int cpu)
{
    struct bitmask *mask = numa_allocate_cpumask();
    if (mask == NULL)
    {
        perror("numa_allocate_cpumask");
        exit(EXIT_FAILURE);
    }
    int ncpu = numa_num_configured_cpus();
    cpu = cpu % ncpu;
    numa_bitmask_clearall(mask);
    numa_bitmask_setbit(mask, cpu);
    numa_sched_setaffinity(gettid(), mask);
    numa_free_cpumask(mask);
}

size_t block_get_size(int fd)
{
    size_t size;
    if (ioctl(fd, BLKGETSIZE64, &size) == -1)
    {
        perror("ioctl");
        exit(EXIT_FAILURE);
    }
    return size;
}

size_t block_get_size_from_path(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        perror("open");
        exit(EXIT_FAILURE);
    }
    size_t size = block_get_size(fd);
    close(fd);
    return size;
}

std::vector<off_t> generate_sequential_offset_vec(size_t file_size, size_t io_size)
{
    size_t n = file_size / io_size;
    std::vector<off_t> offset_vec(n);

    for (off_t i = 0; i < n; i++)
    {
        offset_vec[i] = io_size * i;
    }

    return offset_vec;
}

std::vector<off_t> generate_random_offset_vec(size_t file_size, size_t io_size)
{
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());

    size_t n = file_size / io_size;
    std::vector<off_t> offset_vec(n);

    for (off_t i = 0; i < n; i++)
    {
        offset_vec[i] = io_size * i;
    }

    std::shuffle(offset_vec.begin(), offset_vec.end(), engine);

    return offset_vec;
}

std::vector<off_t> generate_offset_vec(const BenchmarkOptions &options)
{
    std::vector<off_t> offset_vec;
    if (Benchmark::is_sequential_workload(options.job))
    {
        offset_vec = generate_sequential_offset_vec(options.file_size, options.io_size);
    }
    else if (Benchmark::is_random_workload(options.job))
    {
        offset_vec = generate_random_offset_vec(options.file_size, options.io_size);
    }

    return offset_vec;
}

uint64_t pagid_to_ipagid(uint64_t pagid, size_t ndev)
{
    return pagid / ndev;
}

uint64_t pagid_to_idev(uint64_t pagid, size_t ndev)
{
    return pagid % ndev;
}

uint* decomp_calc_head_ptr_to_offsets_by_page_id(uint64_t pagid, uint64_t base_pagid, uint *offsets)
{
    constexpr size_t noffsets_per_page = (2048 + 1);
    return offsets + (pagid - base_pagid) * noffsets_per_page;
}

size_t decomp_get_compressed_page_size_by_page_id(uint64_t pagid, uint64_t base_pagid, uint *compsizes)
{
    return compsizes[(pagid - base_pagid)];
}

std::vector<uint64_t> generate_sequential_pagid_vec(size_t file_size, size_t io_size)
{
    size_t n = file_size / io_size;
    std::vector<uint64_t> pagid_vec(n);

    for (off_t i = 0; i < n; i++)
    {
        pagid_vec[i] = i;
    }

    return pagid_vec;
}

std::vector<uint64_t> generate_random_pagid_vec(size_t file_size, size_t io_size)
{
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());

    size_t n = file_size / io_size;
    std::vector<uint64_t> pagid_vec(n);

    for (off_t i = 0; i < n; i++)
    {
        pagid_vec[i] = i;
    }

    std::shuffle(pagid_vec.begin(), pagid_vec.end(), engine);

    return pagid_vec;
}

std::vector<uint64_t> generate_pagid_vec(const BenchmarkOptions &options)
{
    std::vector<uint64_t> pagid_vec;
    if (Benchmark::is_random_workload(options.job))
    {
        pagid_vec = generate_random_pagid_vec(options.file_size, options.io_size);
    }
    else if (Benchmark::is_sequential_workload(options.job))
    {
        pagid_vec = generate_sequential_pagid_vec(options.file_size, options.io_size);
    }
    else
    {
        std::cerr << "Invalid workload type" << std::endl;
        exit(EXIT_FAILURE);
    }

    return pagid_vec;
}

std::vector<uint64_t> generate_pagid_table(size_t pagid_start, size_t pagid_end)
{
    std::vector<uint64_t> pagid_vec;
    for (size_t i = pagid_start; i < pagid_end; i++)
    {
        pagid_vec.push_back(i);
    }

    return pagid_vec;
}

#define CALC_SEC(time) ((time) / 1000000000)
#define CALC_MSEC(time) (((time) % 1000000000) / 1000000)
#define CALC_USEC(time) (((time) % 1000000000) / 1000)

static uint64_t gettime(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    {
        assert(0);
    }
    return ((uint64_t)ts.tv_sec) * 1000 * 1000 * 1000 + ts.tv_nsec;
}

#define CLOCK_GET_TIME(ts_ptr)                                \
    {                                                         \
        if (0 != clock_gettime(CLOCK_MONOTONIC, ts_ptr))      \
        {                                                     \
            perror("clock_gettime(2) failed ");               \
            fprintf(stderr, " @%s:%d\n", __FILE__, __LINE__); \
        }                                                     \
    }

static time_t elapsed_time_from(struct timespec *start)
{
    struct timespec now;
    struct timespec elapsed;
    CLOCK_GET_TIME(&now);

    if ((now.tv_nsec - start->tv_nsec) < 0)
    {
        elapsed.tv_sec = now.tv_sec - start->tv_sec - 1;
        elapsed.tv_nsec = now.tv_nsec - start->tv_nsec + 1000000000l;
    }
    else
    {
        elapsed.tv_sec = now.tv_sec - start->tv_sec;
        elapsed.tv_nsec = now.tv_nsec - start->tv_nsec;
    }
    return elapsed.tv_sec;
}

template <typename T>
std::vector<std::vector<T>> chunk_vector(const std::vector<T>& vec, size_t n) {
    std::vector<std::vector<T>> chunks;
    size_t chunk_size = (vec.size() + n - 1) / n;

    for (size_t i = 0; i < vec.size(); i += chunk_size) {
        auto chunk = vec | std::ranges::views::drop(i) | std::ranges::views::take(chunk_size);
        chunks.emplace_back(chunk.begin(), chunk.end());
    }

    return chunks;
}

template <typename T>
std::vector<size_t> chunk_vector_start_indexes(const std::vector<T>& vec, size_t n) {
    std::vector<size_t> indexes;
    size_t chunk_size = (vec.size() + n - 1) / n;

    for (size_t i = 0; i < vec.size(); i += chunk_size) {
        indexes.push_back(i);
    }

    return indexes;
}

size_t metadata_calc_uncompressed_pages(size_t total_col_size, size_t page_size)
{
    size_t n = (total_col_size  - 1)/ page_size + 1;
    return n;
}

size_t metadata_calc_npages_for_compression_offsets(size_t noffsets, size_t page_size)
{
    return metadata_calc_uncompressed_pages(noffsets * sizeof(int32_t), page_size);
}

static uint roundup512(size_t n) {
    return (n + 512 - 1) & ~(512 - 1);
}

// Compressed page disk alignment.
// Must match the NVMe physical_block_size (4096) to avoid cross-boundary
// read penalties.  GDS Sync API supports 512B, but the device-level cost
// of non-4K-aligned reads outweighs the padding savings.
static constexpr size_t COMPRESSED_PAGE_ALIGN = 4096;

static size_t roundup4096(size_t n) {
    return (n + COMPRESSED_PAGE_ALIGN - 1) & ~(COMPRESSED_PAGE_ALIGN - 1);
}

size_t metadata_calc_noffsets_for_compression(size_t nrecs_table, size_t rec_size, size_t page_size)
{
    constexpr size_t block_size = 128;
    size_t nrecs_per_page = page_size / rec_size;
    size_t nfullpages = nrecs_table / nrecs_per_page;
    size_t nrecs_in_last_page = nrecs_table % (nfullpages * nrecs_per_page);
    size_t noffsets = (nfullpages * ((nrecs_per_page / block_size) + 1))
        + (roundup512(nrecs_in_last_page) / block_size) + 1;
    // std::cout << nrecs_per_page << " " << nrecs_in_last_page << " " << nfullpages << " " << noffsets << std::endl;
    return noffsets;
}
