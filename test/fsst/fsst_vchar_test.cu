// ============================================================
// FSST-compressed VCHAR page: standalone test program
//
// Input:  orders.tbl (TPC-H ORDERS table, pipe-delimited)
// Target: O_COMMENT column (field 8, max 79 chars)
// Search: Q13 NOT LIKE '%special%requests%'
//
// Tests:
//   1. GPU uncompressed KMP scan (baseline throughput)
//   2. GPU FSST decomp + KMP scan (compressed throughput)
//   3. CPU reference verification (correctness)
// ============================================================

#include "fsst_page.h"
#include "fsst_host.h"
#include "fsst_device.cuh"
#include "cpu_fsst_avx512.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",             \
                cudaGetErrorString(err), __FILE__, __LINE__);     \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

// Test-specific constants
static constexpr uint32_t MAX_COMP_BLOCK_PAYLOAD = 60000;  // < 64KB, leave room
static constexpr uint32_t PAGE_SIZE = 1048576;           // 1MB
static constexpr int BLOCK_SIZE = 128;
static constexpr int MAX_DECOMP_LEN = 80;  // O_COMMENT max 79 + 1

// Comp block payload sweep values
static constexpr uint32_t MAX_COMP_BLOCK_PAYLOAD_V2 = 44000;
static constexpr uint32_t MAX_COMP_BLOCK_PAYLOAD_V3 = 20000;
static constexpr uint32_t MAX_COMP_BLOCK_PAYLOAD_V4 = 9000;
static constexpr uint32_t MAX_COMP_BLOCK_PAYLOAD_V5 = 4000;

// ============================================================
// CPU FSST decompress: interleaved multi-string processing
//
// The scalar fsst_decompress() has a serial dependency chain:
//   posOut += len[code]  (each store address depends on previous)
// This limits throughput to ~5 cycles/code on a single chain.
//
// By interleaving 4 independent strings, we give the OoO engine
// 4 independent dependency chains. The CPU's 2 load ports can
// then serve table lookups from multiple chains simultaneously,
// improving throughput to ~2 cycles/code.
//
// Two modes:
//   output != nullptr: decompress to buffer (for throughput measurement)
//   output == nullptr: checksum-only mode (matching GPU kernel)
// ============================================================

// Decompress a single string (checksum-only, no output buffer)
// Returns decompressed length; writes FNV-1a checksum to *cksum.
static inline uint32_t
fsst_decompress_cksum(const unsigned char* __restrict__ len_tbl,
                      const unsigned long long* __restrict__ sym_tbl,
                      size_t comp_len,
                      const unsigned char* __restrict__ comp,
                      uint32_t* __restrict__ cksum)
{
    uint32_t hash = 2166136261u;
    uint32_t posOut = 0;
    size_t posIn = 0;
    while (posIn < comp_len) {
        unsigned char code = comp[posIn++];
        if (__builtin_expect(code < 255, 1)) {
            unsigned char slen = len_tbl[code];
            unsigned long long sval = sym_tbl[code];
            for (unsigned char j = 0; j < slen; j++) {
                hash ^= (unsigned char)(sval & 0xFF);
                hash *= 16777619u;
                sval >>= 8;
            }
            posOut += slen;
        } else {
            unsigned char byte = comp[posIn++];
            hash ^= byte;
            hash *= 16777619u;
            posOut++;
        }
    }
    *cksum = hash;
    return posOut;
}

// Decompress a range of strings with 4-way interleaving (checksum-only).
static void
fsst_decompress_interleaved4_cksum(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char** comp_ptrs,
    uint32_t* checksums,
    uint32_t* decomp_lens)
{
    const unsigned char* __restrict__ L = (const unsigned char*)decoder->len;
    const unsigned long long* __restrict__ S = (const unsigned long long*)decoder->symbol;

    uint64_t i = start;

    // 4-way interleaved main loop
    for (; i + 4 <= end; i += 4) {
        const unsigned char* cp0 = comp_ptrs[i];
        const unsigned char* cp1 = comp_ptrs[i+1];
        const unsigned char* cp2 = comp_ptrs[i+2];
        const unsigned char* cp3 = comp_ptrs[i+3];
        size_t cl0 = comp_lens[i], cl1 = comp_lens[i+1];
        size_t cl2 = comp_lens[i+2], cl3 = comp_lens[i+3];

        size_t pi0 = 0, pi1 = 0, pi2 = 0, pi3 = 0;
        uint32_t po0 = 0, po1 = 0, po2 = 0, po3 = 0;
        uint32_t h0 = 2166136261u, h1 = 2166136261u;
        uint32_t h2 = 2166136261u, h3 = 2166136261u;

        // Process while all 4 strings still have codes
        while (pi0 < cl0 && pi1 < cl1 && pi2 < cl2 && pi3 < cl3) {
            // String 0
            {
                unsigned char code = cp0[pi0++];
                if (__builtin_expect(code < 255, 1)) {
                    unsigned char slen = L[code];
                    unsigned long long sval = S[code];
                    for (unsigned char j = 0; j < slen; j++) {
                        h0 ^= (unsigned char)(sval & 0xFF);
                        h0 *= 16777619u;
                        sval >>= 8;
                    }
                    po0 += slen;
                } else { unsigned char b = cp0[pi0++]; h0 ^= b; h0 *= 16777619u; po0++; }
            }
            // String 1
            {
                unsigned char code = cp1[pi1++];
                if (__builtin_expect(code < 255, 1)) {
                    unsigned char slen = L[code];
                    unsigned long long sval = S[code];
                    for (unsigned char j = 0; j < slen; j++) {
                        h1 ^= (unsigned char)(sval & 0xFF);
                        h1 *= 16777619u;
                        sval >>= 8;
                    }
                    po1 += slen;
                } else { unsigned char b = cp1[pi1++]; h1 ^= b; h1 *= 16777619u; po1++; }
            }
            // String 2
            {
                unsigned char code = cp2[pi2++];
                if (__builtin_expect(code < 255, 1)) {
                    unsigned char slen = L[code];
                    unsigned long long sval = S[code];
                    for (unsigned char j = 0; j < slen; j++) {
                        h2 ^= (unsigned char)(sval & 0xFF);
                        h2 *= 16777619u;
                        sval >>= 8;
                    }
                    po2 += slen;
                } else { unsigned char b = cp2[pi2++]; h2 ^= b; h2 *= 16777619u; po2++; }
            }
            // String 3
            {
                unsigned char code = cp3[pi3++];
                if (__builtin_expect(code < 255, 1)) {
                    unsigned char slen = L[code];
                    unsigned long long sval = S[code];
                    for (unsigned char j = 0; j < slen; j++) {
                        h3 ^= (unsigned char)(sval & 0xFF);
                        h3 *= 16777619u;
                        sval >>= 8;
                    }
                    po3 += slen;
                } else { unsigned char b = cp3[pi3++]; h3 ^= b; h3 *= 16777619u; po3++; }
            }
        }

        // Drain remaining codes per string
        #define DRAIN_STRING(CP, CL, PI, PO, H) \
            while (PI < CL) { \
                unsigned char code = CP[PI++]; \
                if (__builtin_expect(code < 255, 1)) { \
                    unsigned char slen = L[code]; \
                    unsigned long long sval = S[code]; \
                    for (unsigned char j = 0; j < slen; j++) { \
                        H ^= (unsigned char)(sval & 0xFF); \
                        H *= 16777619u; sval >>= 8; \
                    } \
                    PO += slen; \
                } else { H ^= CP[PI++]; H *= 16777619u; PO++; } \
            }
        DRAIN_STRING(cp0, cl0, pi0, po0, h0);
        DRAIN_STRING(cp1, cl1, pi1, po1, h1);
        DRAIN_STRING(cp2, cl2, pi2, po2, h2);
        DRAIN_STRING(cp3, cl3, pi3, po3, h3);
        #undef DRAIN_STRING

        checksums[i]   = h0; checksums[i+1] = h1;
        checksums[i+2] = h2; checksums[i+3] = h3;
        decomp_lens[i]   = po0; decomp_lens[i+1] = po1;
        decomp_lens[i+2] = po2; decomp_lens[i+3] = po3;
    }

    // Handle remaining strings (< 4)
    for (; i < end; i++) {
        decomp_lens[i] = fsst_decompress_cksum(L, S, comp_lens[i], comp_ptrs[i], &checksums[i]);
    }
}

// Decompress a range of strings with 4-way interleaving (to output buffer).
static void
fsst_decompress_interleaved4_output(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char** comp_ptrs,
    unsigned char* output,          // output[i * slot_size]
    uint32_t* decomp_lens,
    uint32_t slot_size)
{
    const unsigned char* __restrict__ L = (const unsigned char*)decoder->len;
    const unsigned long long* __restrict__ S = (const unsigned long long*)decoder->symbol;

    uint64_t i = start;

    for (; i + 4 <= end; i += 4) {
        const unsigned char* cp0 = comp_ptrs[i];
        const unsigned char* cp1 = comp_ptrs[i+1];
        const unsigned char* cp2 = comp_ptrs[i+2];
        const unsigned char* cp3 = comp_ptrs[i+3];
        size_t cl0 = comp_lens[i], cl1 = comp_lens[i+1];
        size_t cl2 = comp_lens[i+2], cl3 = comp_lens[i+3];
        unsigned char* out0 = output + i * slot_size;
        unsigned char* out1 = output + (i+1) * slot_size;
        unsigned char* out2 = output + (i+2) * slot_size;
        unsigned char* out3 = output + (i+3) * slot_size;

        size_t pi0 = 0, pi1 = 0, pi2 = 0, pi3 = 0;
        uint32_t po0 = 0, po1 = 0, po2 = 0, po3 = 0;

        while (pi0 < cl0 && pi1 < cl1 && pi2 < cl2 && pi3 < cl3) {
            #define DECODE_ONE_OUTPUT(CP, CL, PI, OUT, PO) { \
                unsigned char code = CP[PI++]; \
                if (__builtin_expect(code < 255, 1)) { \
                    unsigned char slen = L[code]; \
                    memcpy(OUT + PO, &S[code], 8); \
                    PO += slen; \
                } else { OUT[PO++] = CP[PI++]; } \
            }
            DECODE_ONE_OUTPUT(cp0, cl0, pi0, out0, po0);
            DECODE_ONE_OUTPUT(cp1, cl1, pi1, out1, po1);
            DECODE_ONE_OUTPUT(cp2, cl2, pi2, out2, po2);
            DECODE_ONE_OUTPUT(cp3, cl3, pi3, out3, po3);
            #undef DECODE_ONE_OUTPUT
        }

        #define DRAIN_OUTPUT(CP, CL, PI, OUT, PO) \
            while (PI < CL) { \
                unsigned char code = CP[PI++]; \
                if (__builtin_expect(code < 255, 1)) { \
                    memcpy(OUT + PO, &S[code], 8); \
                    PO += L[code]; \
                } else { OUT[PO++] = CP[PI++]; } \
            }
        DRAIN_OUTPUT(cp0, cl0, pi0, out0, po0);
        DRAIN_OUTPUT(cp1, cl1, pi1, out1, po1);
        DRAIN_OUTPUT(cp2, cl2, pi2, out2, po2);
        DRAIN_OUTPUT(cp3, cl3, pi3, out3, po3);
        #undef DRAIN_OUTPUT

        decomp_lens[i] = po0; decomp_lens[i+1] = po1;
        decomp_lens[i+2] = po2; decomp_lens[i+3] = po3;
    }

    for (; i < end; i++) {
        decomp_lens[i] = (uint32_t)fsst_decompress(decoder, comp_lens[i], comp_ptrs[i],
                                                     slot_size, output + i * slot_size);
    }
}

// ============================================================
// KMP helpers
// ============================================================

struct KmpData {
    std::string all_patterns;     // concatenated
    std::vector<int> next;        // failure function
    std::vector<int> offsets;     // pattern start offsets
    std::vector<int> lengths;     // pattern lengths
    int num_patterns;
    int total_chars;
};

static void build_kmp_next(const char* pat, int len, int* next) {
    next[0] = 0;
    int k = 0;
    for (int i = 1; i < len; i++) {
        while (k > 0 && pat[k] != pat[i])
            k = next[k - 1];
        if (pat[k] == pat[i]) k++;
        next[i] = k;
    }
}

static KmpData build_kmp(const std::vector<std::string>& patterns) {
    KmpData kmp;
    kmp.num_patterns = (int)patterns.size();
    int offset = 0;
    for (auto& p : patterns) {
        kmp.offsets.push_back(offset);
        kmp.lengths.push_back((int)p.size());
        kmp.all_patterns += p;
        offset += (int)p.size();
    }
    kmp.total_chars = offset;
    kmp.next.resize(kmp.total_chars);
    for (int i = 0; i < kmp.num_patterns; i++) {
        build_kmp_next(kmp.all_patterns.c_str() + kmp.offsets[i],
                       kmp.lengths[i],
                       kmp.next.data() + kmp.offsets[i]);
    }
    return kmp;
}

static bool cpu_kmp_match(const char* str, int str_len,
                          const KmpData& kmp) {
    int current_pat = 0;
    int l = 0;
    int p_offset = kmp.offsets[0];
    int p_len = kmp.lengths[0];

    for (int i = 0; i < str_len; i++) {
        char c = str[i];
        while (l > 0 && kmp.all_patterns[p_offset + l] != c)
            l = kmp.next[p_offset + l - 1];
        if (kmp.all_patterns[p_offset + l] == c) l++;
        if (l == p_len) {
            current_pat++;
            l = 0;
            if (current_pat >= kmp.num_patterns) return true;
            p_offset = kmp.offsets[current_pat];
            p_len = kmp.lengths[current_pat];
        }
    }
    return false;
}

// ============================================================
// Parse orders.tbl (mmap + multi-threaded)
// ============================================================

// Extract O_COMMENT (field 8) from a single line
static inline bool extract_comment(const char* line, size_t len,
                                   const char*& out_start, size_t& out_len) {
    int field = 0;
    const char* start = line;
    const char* end = line + len;
    for (const char* p = line; p < end; p++) {
        if (*p == '|') {
            if (field == 8) {
                out_start = start;
                out_len = p - start;
                return out_len > 0;
            }
            field++;
            start = p + 1;
        }
    }
    return false;
}

// Parse a chunk [chunk_start, chunk_end) of mmap'd data.
// Each thread writes to its own vector.
static void parse_chunk(const char* data, size_t chunk_start, size_t chunk_end,
                        std::vector<std::string>& out) {
    size_t pos = chunk_start;
    while (pos < chunk_end) {
        // Find end of line
        const char* nl = (const char*)memchr(data + pos, '\n', chunk_end - pos);
        size_t line_end = nl ? (size_t)(nl - data) : chunk_end;
        size_t line_len = line_end - pos;

        if (line_len > 0) {
            const char* cstart;
            size_t clen;
            if (extract_comment(data + pos, line_len, cstart, clen)) {
                out.emplace_back(cstart, clen);
            }
        }
        pos = line_end + 1;
    }
}

// mmap a file and parse in parallel
static std::vector<std::string> parse_orders_mmap(const char* path, int nthreads) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }

    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); exit(1); }
    size_t file_size = st.st_size;
    if (file_size == 0) { close(fd); return {}; }

    const char* data = (const char*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);  // fd can be closed after mmap
    madvise((void*)data, file_size, MADV_SEQUENTIAL);

    // Find chunk boundaries at line breaks
    std::vector<size_t> boundaries(nthreads + 1);
    boundaries[0] = 0;
    boundaries[nthreads] = file_size;
    for (int i = 1; i < nthreads; i++) {
        size_t pos = file_size * i / nthreads;
        // Advance to next newline
        const char* nl = (const char*)memchr(data + pos, '\n', file_size - pos);
        boundaries[i] = nl ? (size_t)(nl - data) + 1 : file_size;
    }

    // Parse in parallel
    std::vector<std::vector<std::string>> per_thread(nthreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; t++) {
        threads.emplace_back(parse_chunk, data, boundaries[t], boundaries[t + 1],
                             std::ref(per_thread[t]));
    }
    for (auto& th : threads) th.join();

    munmap((void*)data, file_size);

    // Merge
    size_t total = 0;
    for (auto& v : per_thread) total += v.size();
    std::vector<std::string> comments;
    comments.reserve(total);
    for (auto& v : per_thread) {
        comments.insert(comments.end(),
                        std::make_move_iterator(v.begin()),
                        std::make_move_iterator(v.end()));
    }
    return comments;
}

// Parse orders.tbl or orders.tbl.* (split files, parallel)
static std::vector<std::string> parse_orders_tbl(const char* path, int nthreads) {
    // Single file?
    struct stat st;
    if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) {
        return parse_orders_mmap(path, nthreads);
    }

    // Split files: path.1, path.2, ...
    std::vector<std::string> paths;
    for (int i = 1; ; i++) {
        std::string split_path = std::string(path) + "." + std::to_string(i);
        if (stat(split_path.c_str(), &st) != 0) {
            if (i == 1) {
                fprintf(stderr, "Cannot open %s or %s.1\n", path, path);
                exit(1);
            }
            break;
        }
        paths.push_back(split_path);
    }

    // Parse split files: nthreads threads, each processes multiple files
    int nfiles = (int)paths.size();
    std::vector<std::vector<std::string>> per_thread(nthreads);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; t++) {
            threads.emplace_back([&, t]() {
                for (int i = t; i < nfiles; i += nthreads) {
                    auto chunk = parse_orders_mmap(paths[i].c_str(), 1);
                    per_thread[t].insert(per_thread[t].end(),
                                         std::make_move_iterator(chunk.begin()),
                                         std::make_move_iterator(chunk.end()));
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    // Merge
    size_t total = 0;
    for (auto& v : per_thread) total += v.size();
    std::vector<std::string> comments;
    comments.reserve(total);
    for (auto& v : per_thread) {
        comments.insert(comments.end(),
                        std::make_move_iterator(v.begin()),
                        std::make_move_iterator(v.end()));
    }
    return comments;
}

// ============================================================
// GPU Kernel: Uncompressed KMP scan (baseline)
// ============================================================

// Simple flat layout: all raw strings contiguous, with offset array
// d_offsets[i] = byte offset of string i, d_offsets[nrecs] = total bytes
__global__ void raw_kmp_scan_kernel(
    const char*     d_strings,
    const uint32_t* d_offsets,
    uint32_t        nrecs,
    const char*     d_patterns,
    const int*      d_next,
    const int*      d_pattern_offsets,
    const int*      d_pattern_lengths,
    int             num_patterns,
    uint64_t*       d_count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nrecs) return;

    uint32_t str_start = d_offsets[gid];
    uint32_t str_len = d_offsets[gid + 1] - str_start;
    const char* str = d_strings + str_start;

    // KMP multi-pattern match
    int current_pat = 0;
    int l = 0;
    int p_offset = d_pattern_offsets[0];
    int p_len = d_pattern_lengths[0];

    for (uint32_t i = 0; i < str_len; i++) {
        char c = str[i];
        while (l > 0 && d_patterns[p_offset + l] != c)
            l = d_next[p_offset + l - 1];
        if (d_patterns[p_offset + l] == c) l++;
        if (l == p_len) {
            current_pat++;
            l = 0;
            if (current_pat >= num_patterns) break;
            p_offset = d_pattern_offsets[current_pat];
            p_len = d_pattern_lengths[current_pat];
        }
    }

    bool matched = (current_pat >= num_patterns);
    if (!matched) {
        atomicAdd((unsigned long long*)d_count, 1ULL);
    }
}

// ============================================================
// GPU Kernel 2: FSST decomp + KMP scan
// ============================================================

__global__ void fsst_decomp_kmp_scan_kernel(
    const char*              d_pages,
    const FsstCompBlockKernelMeta* d_cb_meta,
    uint32_t                 total_comp_blocks,
    const char*              d_patterns,
    const int*               d_next,
    const int*               d_pattern_offsets,
    const int*               d_pattern_lengths,
    int                      num_patterns,
    uint64_t*                d_count)
{
    if (blockIdx.x >= total_comp_blocks) return;

    const FsstCompBlockKernelMeta meta = d_cb_meta[blockIdx.x];
    const char* page_base = d_pages + meta.page_byte_offset;
    const char* symtab_base = page_base + meta.symtab_byte_offset;
    const char* cb_base = page_base + meta.comp_block_byte_offset;

    const uint32_t tid = threadIdx.x;
    const uint32_t nrecs = meta.nrecs;

    // Shared memory: decoder + KMP patterns
    __shared__ uint8_t  s_sym_len[256];
    __shared__ uint64_t s_sym_val[255];
    __shared__ char     s_kmp_patterns[32];
    __shared__ int      s_kmp_next[32];
    __shared__ int      s_kmp_offsets[4];
    __shared__ int      s_kmp_lengths[4];

    // Phase 1: Cooperative load decoder into shared memory
    // len[256] bytes
    for (uint32_t i = tid; i < 256; i += blockDim.x) {
        s_sym_len[i] = ((const uint8_t*)symtab_base)[i];
    }
    // symbol[255] × uint64_t
    for (uint32_t i = tid; i < 255; i += blockDim.x) {
        memcpy(&s_sym_val[i], symtab_base + FSST_SYMTAB_LEN_BYTES + i * 8, 8);
    }
    // KMP data (small, first threads)
    if (tid < 32) {
        s_kmp_patterns[tid] = (tid < num_patterns * 16) ? d_patterns[tid] : 0;
        s_kmp_next[tid] = (tid < num_patterns * 16) ? d_next[tid] : 0;
    }
    if (tid < 4) {
        s_kmp_offsets[tid] = (tid < num_patterns) ? d_pattern_offsets[tid] : 0;
        s_kmp_lengths[tid] = (tid < num_patterns) ? d_pattern_lengths[tid] : 0;
    }

    __syncthreads();

    // Phase 2: Locate offset table and compressed data
    const uint16_t* offset_table = (const uint16_t*)cb_base;
    const uint8_t* comp_data = (const uint8_t*)(cb_base + (nrecs + 1) * sizeof(uint16_t));

    // Phase 3: Per-thread FSST decompress + KMP scan
    uint64_t my_qualifying = 0;

    for (uint32_t r = tid; r < nrecs; r += blockDim.x) {
        uint16_t comp_start = offset_table[r];
        uint16_t comp_len = offset_table[r + 1] - comp_start;
        const uint8_t* comp_ptr = comp_data + comp_start;

        // FSST decompress into local buffer
        char decomp_buf[MAX_DECOMP_LEN];
        uint32_t posOut = 0, posIn = 0;

        while (posIn < comp_len && posOut < MAX_DECOMP_LEN) {
            uint8_t code = comp_ptr[posIn++];
            if (code < 255) {
                uint8_t sym_len = s_sym_len[code];
                uint64_t sym_val = s_sym_val[code];
                // Unrolled copy (max 8 bytes)
                if (posOut + sym_len <= MAX_DECOMP_LEN) {
                    memcpy(&decomp_buf[posOut], &sym_val, 8);  // safe unaligned write
                    posOut += sym_len;
                } else {
                    for (uint8_t j = 0; j < sym_len && posOut < MAX_DECOMP_LEN; j++) {
                        decomp_buf[posOut++] = ((const char*)&sym_val)[j];
                    }
                }
            } else {
                if (posIn < comp_len) {
                    decomp_buf[posOut++] = (char)comp_ptr[posIn++];
                }
            }
        }

        // KMP multi-pattern match on decompressed data
        int str_len = (int)posOut;
        int current_pat = 0;
        int l = 0;
        int p_offset = s_kmp_offsets[0];
        int p_len = s_kmp_lengths[0];

        for (int i = 0; i < str_len; i++) {
            char c = decomp_buf[i];
            while (l > 0 && s_kmp_patterns[p_offset + l] != c)
                l = s_kmp_next[p_offset + l - 1];
            if (s_kmp_patterns[p_offset + l] == c) l++;
            if (l == p_len) {
                current_pat++;
                l = 0;
                if (current_pat >= num_patterns) break;
                p_offset = s_kmp_offsets[current_pat];
                p_len = s_kmp_lengths[current_pat];
            }
        }

        bool matched = (current_pat >= num_patterns);
        if (!matched) my_qualifying++;
    }

    if (my_qualifying > 0) {
        atomicAdd((unsigned long long*)d_count,
                  (unsigned long long)my_qualifying);
    }
}

// ============================================================
// GPU Kernel: Decompress-only (no KMP scan)
//
// Same shared-memory optimizations as V2 (coalesced load of
// decoder + comp block data), but writes decompressed output to
// global memory instead of running KMP inline.
//
// Output layout: fixed-slot  d_output[global_rid * slot_size]
//                d_decomp_lens[global_rid] = decompressed length
// ============================================================

__global__ void fsst_decompress_kernel(
    const char*              d_pages,
    const FsstCompBlockKernelMeta* d_cb_meta,
    uint32_t                 total_comp_blocks,
    char*                    d_output,        // [total_recs * slot_size] (may be NULL)
    uint32_t*                d_decomp_lens,   // [total_recs]
    uint32_t*                d_checksums,     // [total_recs] FNV-1a of decompressed data
    uint32_t                 slot_size)
{
    if (blockIdx.x >= total_comp_blocks) return;

    const FsstCompBlockKernelMeta meta = d_cb_meta[blockIdx.x];
    const char* page_base = d_pages + meta.page_byte_offset;
    const char* symtab_base = page_base + meta.symtab_byte_offset;
    const char* cb_base = page_base + meta.comp_block_byte_offset;

    const uint32_t tid = threadIdx.x;
    const uint32_t nrecs = meta.nrecs;
    const uint32_t cb_data_size = meta.comp_block_data_size;
    const uint32_t rec_base = meta.rec_base;

    // Shared memory (same layout as V2)
    extern __shared__ char smem[];
    uint8_t*  s_sym_len  = (uint8_t*)  smem;
    uint64_t* s_sym_val  = (uint64_t*)(smem + 256);
    uint8_t*  s_seg_data = (uint8_t*)(smem + 2496);

    // Phase 1: cooperative coalesced load of decoder
    for (uint32_t i = tid; i < 256; i += blockDim.x)
        s_sym_len[i] = ((const uint8_t*)symtab_base)[i];
    for (uint32_t i = tid; i < 255; i += blockDim.x)
        memcpy(&s_sym_val[i], symtab_base + FSST_SYMTAB_LEN_BYTES + i * 8, 8);

    // Phase 2: cooperative coalesced load of comp block data
    {
        uint32_t num_words = (cb_data_size + 3) / 4;
        const uint32_t* src = (const uint32_t*)cb_base;
        uint32_t* dst = (uint32_t*)s_seg_data;
        for (uint32_t i = tid; i < num_words; i += blockDim.x)
            dst[i] = src[i];
    }

    __syncthreads();

    // Phase 3: per-thread FSST decompression → global memory output
    const uint16_t* offset_table = (const uint16_t*)s_seg_data;
    const uint8_t* comp_data = s_seg_data + (nrecs + 1) * sizeof(uint16_t);

    for (uint32_t r = tid; r < nrecs; r += blockDim.x) {
        uint16_t comp_start = offset_table[r];
        uint16_t comp_len = offset_table[r + 1] - comp_start;
        const uint8_t* comp_ptr = comp_data + comp_start;

        uint32_t global_rid = rec_base + r;
        char* out = d_output ? (d_output + (uint64_t)global_rid * slot_size) : nullptr;
        const bool do_hash = (d_checksums != nullptr);
        uint32_t posOut = 0, posIn = 0;
        uint32_t hash = 2166136261u;

        while (posIn < comp_len) {
            uint8_t code = comp_ptr[posIn++];
            uint64_t sym_val;
            uint8_t sym_len;

            if (code < 255) {
                sym_len = s_sym_len[code];
                sym_val = s_sym_val[code];
            } else {
                sym_val = (uint64_t)comp_ptr[posIn++];
                sym_len = 1;
            }

            // Write to output buffer (if provided)
            if (out && posOut + 8 <= slot_size) {
                memcpy(out + posOut, &sym_val, 8);
            }

            // Compute FNV-1a inline (only if checksums requested)
            if (do_hash) {
                for (uint8_t j = 0; j < sym_len; j++) {
                    hash ^= (uint8_t)(sym_val & 0xFF);
                    hash *= 16777619u;
                    sym_val >>= 8;
                }
            }
            posOut += sym_len;
        }

        d_decomp_lens[global_rid] = posOut;
        if (d_checksums) d_checksums[global_rid] = hash;
    }
}

// ============================================================
// GPU Kernel: Decompress to local buffer (cache/register mode)
//
// Decompresses each record into a thread-local buffer (registers/
// local memory) without writing to global memory. This measures
// pure decode throughput where output stays in L1/registers,
// analogous to CPU cache-resident output.
// ============================================================

__global__ void fsst_decompress_cache_kernel(
    const char*              d_pages,
    const FsstCompBlockKernelMeta* d_cb_meta,
    uint32_t                 total_comp_blocks,
    uint32_t*                d_decomp_lens)
{
    if (blockIdx.x >= total_comp_blocks) return;

    const FsstCompBlockKernelMeta meta = d_cb_meta[blockIdx.x];
    const char* page_base = d_pages + meta.page_byte_offset;
    const char* symtab_base = page_base + meta.symtab_byte_offset;
    const char* cb_base = page_base + meta.comp_block_byte_offset;

    const uint32_t tid = threadIdx.x;
    const uint32_t nrecs = meta.nrecs;
    const uint32_t cb_data_size = meta.comp_block_data_size;
    const uint32_t rec_base = meta.rec_base;

    // Shared memory (same layout as V2)
    extern __shared__ char smem[];
    uint8_t*  s_sym_len  = (uint8_t*)  smem;
    uint64_t* s_sym_val  = (uint64_t*)(smem + 256);
    uint8_t*  s_seg_data = (uint8_t*)(smem + 2496);

    // Phase 1: cooperative coalesced load of decoder
    for (uint32_t i = tid; i < 256; i += blockDim.x)
        s_sym_len[i] = ((const uint8_t*)symtab_base)[i];
    for (uint32_t i = tid; i < 255; i += blockDim.x)
        memcpy(&s_sym_val[i], symtab_base + FSST_SYMTAB_LEN_BYTES + i * 8, 8);

    // Phase 2: cooperative coalesced load of comp block data
    {
        uint32_t num_words = (cb_data_size + 3) / 4;
        const uint32_t* src = (const uint32_t*)cb_base;
        uint32_t* dst = (uint32_t*)s_seg_data;
        for (uint32_t i = tid; i < num_words; i += blockDim.x)
            dst[i] = src[i];
    }

    __syncthreads();

    // Phase 3: per-thread FSST decompression → local buffer (registers)
    const uint16_t* offset_table = (const uint16_t*)s_seg_data;
    const uint8_t* comp_data = s_seg_data + (nrecs + 1) * sizeof(uint16_t);

    for (uint32_t r = tid; r < nrecs; r += blockDim.x) {
        uint16_t comp_start = offset_table[r];
        uint16_t comp_len = offset_table[r + 1] - comp_start;
        const uint8_t* comp_ptr = comp_data + comp_start;

        // Local buffer — stays in registers/L1 cache, no global memory write
        char decomp_buf[MAX_DECOMP_LEN];
        uint32_t posOut = 0, posIn = 0;

        while (posIn < comp_len) {
            uint8_t code = comp_ptr[posIn++];
            uint64_t sym_val;
            uint8_t sym_len;

            if (code < 255) {
                sym_len = s_sym_len[code];
                sym_val = s_sym_val[code];
            } else {
                sym_val = (uint64_t)comp_ptr[posIn++];
                sym_len = 1;
            }

            if (posOut + 8 <= MAX_DECOMP_LEN) {
                memcpy(&decomp_buf[posOut], &sym_val, 8);
            }
            posOut += sym_len;
        }

        d_decomp_lens[rec_base + r] = posOut;
    }
}

// ============================================================
// Test helper: build FSST pages using pagcol API
// ============================================================

struct TestFsstPages {
    std::vector<uint8_t>              page_data;
    std::vector<FsstCompBlockKernelMeta> cb_metas;
    uint32_t npages;
    uint32_t total_comp_blocks;
    size_t   total_compressed_bytes;
};

static TestFsstPages build_test_fsst_pages(
    const fsst_decoder_t& decoder,
    const unsigned char* const* comp_ptrs,
    const size_t* comp_lens,
    uint64_t nrecs,
    uint32_t max_comp_block_payload = 4000)
{
    uint8_t raw_symtab[FSST_SYMTAB_TOTAL];
    fsst_serialize_symbol_table(decoder, raw_symtab);

    std::vector<uint8_t> all_pages;
    std::vector<FsstCompBlockKernelMeta> all_metas;
    uint32_t npages = 0;
    uint32_t rec_offset = 0;
    size_t total_comp = 0;

    while (rec_offset < nrecs) {
        // Allocate one page
        size_t page_start = all_pages.size();
        all_pages.resize(page_start + PAGE_SIZE, 0);
        PAG* pag = (PAG*)(all_pages.data() + page_start);
        pag_init(pag, PAGE_SIZE);

        uint32_t comp_bytes = 0;
        uint32_t packed = pagcol_append_batch_unordered_column_vchar_fsst(
            pag, raw_symtab, (uint32_t)(nrecs - rec_offset),
            comp_ptrs + rec_offset, comp_lens + rec_offset,
            max_comp_block_payload, comp_bytes, PAGE_SIZE);

        if (packed == 0) break;

        // Build comp block metas from this page
        fsst_build_comp_block_metas_from_page(
            all_pages.data() + page_start,
            (uint32_t)page_start, rec_offset, all_metas);

        for (uint32_t i = 0; i < packed; i++)
            total_comp += comp_lens[rec_offset + i];

        rec_offset += packed;
        npages++;
    }

    TestFsstPages result;
    result.page_data = std::move(all_pages);
    result.cb_metas = std::move(all_metas);
    result.npages = npages;
    result.total_comp_blocks = (uint32_t)result.cb_metas.size();
    result.total_compressed_bytes = total_comp;
    return result;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    // Parse --skip-cpu flag, then positional args
    bool skip_cpu = false;
    std::vector<char*> pos_args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--skip-cpu") == 0) {
            skip_cpu = true;
        } else {
            pos_args.push_back(argv[i]);
        }
    }
    if (pos_args.empty()) {
        fprintf(stderr, "Usage: %s <orders.tbl> [num_iterations] [num_threads] [--skip-cpu]\n", argv[0]);
        return 1;
    }
    const char* orders_path = pos_args[0];
    int n_iter = (pos_args.size() >= 2) ? atoi(pos_args[1]) : 10;
    int nthreads = (pos_args.size() >= 3) ? atoi(pos_args[2]) :
                   (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    printf("Using %d threads%s\n", nthreads, skip_cpu ? " (--skip-cpu)" : "");

    // ── 1. Parse orders.tbl (parallel mmap) ──
    printf("Parsing %s ...\n", orders_path);
    auto t0 = std::chrono::steady_clock::now();
    auto comments = parse_orders_tbl(orders_path, nthreads);
    auto t1 = std::chrono::steady_clock::now();
    double parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Parsed %lu ORDERS records (%.1f ms)\n",
           (unsigned long)comments.size(), parse_ms);

    // Cutoff: keep raw data within uint32_t range (~4 GB) for offset tables
    {
        size_t cum = 0;
        size_t max_recs = comments.size();
        for (size_t i = 0; i < comments.size(); i++) {
            cum += comments[i].size();
            if (cum > (size_t)3900000000ULL) {  // ~3.9 GB, safe margin
                max_recs = i;
                break;
            }
        }
        if (max_recs < comments.size()) {
            printf("Cutoff at %lu records (raw %.2f GB) to fit uint32_t offsets\n",
                   (unsigned long)max_recs, cum / 1e9);
            comments.resize(max_recs);
        }
    }

    uint64_t nrecs = comments.size();
    size_t total_raw_bytes = 0;
    for (auto& s : comments) total_raw_bytes += s.size();
    printf("Using %lu records, O_COMMENT raw size: %.2f MB\n",
           (unsigned long)nrecs, total_raw_bytes / 1e6);

    // ── 2. Build KMP patterns ──
    KmpData kmp = build_kmp({"special", "requests"});

    // ── 3. CPU reference: raw KMP ──
    printf("\nCPU reference (raw KMP) ...\n");
    uint64_t cpu_raw_count = 0;
    for (uint64_t i = 0; i < nrecs; i++) {
        if (!cpu_kmp_match(comments[i].c_str(), (int)comments[i].size(), kmp))
            cpu_raw_count++;
    }
    printf("  CPU raw qualifying (NOT LIKE): %lu / %lu\n", cpu_raw_count, nrecs);

    // ── 4. FSST compression ──
    printf("\nFSST compression ...\n");
    t0 = std::chrono::steady_clock::now();

    std::vector<size_t> lenIn(nrecs);
    std::vector<const unsigned char*> strIn(nrecs);
    for (uint64_t i = 0; i < nrecs; i++) {
        lenIn[i] = comments[i].size();
        strIn[i] = (const unsigned char*)comments[i].data();
    }

    fsst_encoder_t* encoder = fsst_create(nrecs, lenIn.data(), strIn.data(), 0);
    fsst_decoder_t decoder = fsst_decoder(encoder);

    size_t outbuf_size = 7 + 2 * total_raw_bytes;
    std::vector<unsigned char> comp_output(outbuf_size);
    std::vector<size_t> comp_lens(nrecs);
    std::vector<unsigned char*> comp_ptrs(nrecs);

    size_t compressed_count = fsst_compress(
        encoder, nrecs, lenIn.data(), strIn.data(),
        outbuf_size, comp_output.data(),
        comp_lens.data(), comp_ptrs.data());
    assert(compressed_count == nrecs);

    t1 = std::chrono::steady_clock::now();
    double comp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    size_t total_comp_bytes = 0;
    for (uint64_t i = 0; i < nrecs; i++) total_comp_bytes += comp_lens[i];
    printf("  Compressed: %.2f MB -> %.2f MB (ratio: %.2fx, %.1f ms)\n",
           total_raw_bytes / 1e6, total_comp_bytes / 1e6,
           (double)total_raw_bytes / total_comp_bytes, comp_ms);

    // ── 5. CPU reference: FSST decomp + KMP ──
    printf("\nCPU reference (FSST decomp + KMP) ...\n");
    uint64_t cpu_fsst_count = 0;
    {
        unsigned char dbuf[256];
        for (uint64_t i = 0; i < nrecs; i++) {
            size_t dlen = fsst_decompress(&decoder, comp_lens[i], comp_ptrs[i],
                                          sizeof(dbuf), dbuf);
            if (!cpu_kmp_match((const char*)dbuf, (int)dlen, kmp))
                cpu_fsst_count++;
        }
    }
    printf("  CPU FSST qualifying (NOT LIKE): %lu / %lu\n", cpu_fsst_count, nrecs);

    if (cpu_raw_count != cpu_fsst_count) {
        fprintf(stderr, "ERROR: CPU raw (%lu) != CPU FSST (%lu)\n",
                cpu_raw_count, cpu_fsst_count);
        return 1;
    }
    printf("  CPU verification: PASS\n");

    // ── 6. Build FSST pages (parallel) ──
    printf("\nBuilding FSST pages ...\n");
    t0 = std::chrono::steady_clock::now();

    // Partition records across threads and build pages in parallel
    std::vector<TestFsstPages> thread_pages(nthreads);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; t++) {
            uint64_t chunk_start = (uint64_t)t * nrecs / nthreads;
            uint64_t chunk_end = (uint64_t)(t + 1) * nrecs / nthreads;
            uint64_t chunk_n = chunk_end - chunk_start;
            if (chunk_n == 0) continue;
            threads.emplace_back([&, t, chunk_start, chunk_n]() {
                thread_pages[t] = build_test_fsst_pages(
                    decoder,
                    (const unsigned char* const*)(comp_ptrs.data() + chunk_start),
                    comp_lens.data() + chunk_start,
                    chunk_n, MAX_COMP_BLOCK_PAYLOAD);
            });
        }
        for (auto& th : threads) th.join();
    }

    // Merge thread_pages into a single TestFsstPages
    TestFsstPages fpages;
    {
        size_t total_page_bytes = 0;
        uint32_t total_cbs = 0;
        size_t total_comp = 0;
        uint32_t total_npages = 0;
        for (auto& tp : thread_pages) {
            total_page_bytes += tp.page_data.size();
            total_cbs += tp.total_comp_blocks;
            total_comp += tp.total_compressed_bytes;
            total_npages += tp.npages;
        }
        fpages.page_data.reserve(total_page_bytes);
        fpages.cb_metas.reserve(total_cbs);
        fpages.npages = total_npages;
        fpages.total_comp_blocks = total_cbs;
        fpages.total_compressed_bytes = total_comp;

        // Concatenate page data and fix up comp block metas (page_byte_offset, rec_base)
        uint32_t page_data_offset = 0;
        uint32_t rec_base_offset = 0;
        for (auto& tp : thread_pages) {
            for (auto& meta : tp.cb_metas) {
                meta.page_byte_offset += page_data_offset;
                meta.rec_base += rec_base_offset;
            }
            fpages.page_data.insert(fpages.page_data.end(),
                                    tp.page_data.begin(), tp.page_data.end());
            fpages.cb_metas.insert(fpages.cb_metas.end(),
                                    tp.cb_metas.begin(), tp.cb_metas.end());
            page_data_offset += (uint32_t)tp.page_data.size();
            // Count records in this thread's chunk
            uint32_t chunk_recs = 0;
            for (auto& m : tp.cb_metas) chunk_recs += m.nrecs;
            rec_base_offset += chunk_recs;
        }
    }

    t1 = std::chrono::steady_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Pages: %u, Comp blocks: %u, Page size: %u (%.1f ms)\n",
           fpages.npages, fpages.total_comp_blocks, PAGE_SIZE, build_ms);
    printf("  Page data: %.2f MB\n", fpages.page_data.size() / 1e6);

    // ── 7. GPU setup ──
    printf("\nGPU setup ...\n");

    // KMP data on GPU
    char* d_patterns;
    int* d_next;
    int* d_pattern_offsets;
    int* d_pattern_lengths;
    CUDA_CHECK(cudaMalloc(&d_patterns, kmp.total_chars));
    CUDA_CHECK(cudaMalloc(&d_next, kmp.total_chars * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pattern_offsets, kmp.num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pattern_lengths, kmp.num_patterns * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_patterns, kmp.all_patterns.data(), kmp.total_chars, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next, kmp.next.data(), kmp.total_chars * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pattern_offsets, kmp.offsets.data(), kmp.num_patterns * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pattern_lengths, kmp.lengths.data(), kmp.num_patterns * sizeof(int), cudaMemcpyHostToDevice));

    // ── 8. GPU Test 1: Uncompressed KMP scan ──
    printf("\n=== GPU Test 1: Uncompressed KMP Scan ===\n");
    {
        // Build flat string array + offset table
        std::vector<uint32_t> offsets(nrecs + 1);
        offsets[0] = 0;
        for (uint64_t i = 0; i < nrecs; i++)
            offsets[i + 1] = offsets[i] + (uint32_t)comments[i].size();

        std::vector<char> flat_strings(offsets[nrecs]);
        for (uint64_t i = 0; i < nrecs; i++)
            memcpy(flat_strings.data() + offsets[i], comments[i].data(), comments[i].size());

        char* d_strings;
        uint32_t* d_offsets;
        uint64_t* d_count;
        CUDA_CHECK(cudaMalloc(&d_strings, flat_strings.size()));
        CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_strings, flat_strings.data(), flat_strings.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

        int grid = ((int)nrecs + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Warmup
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint64_t)));
        raw_kmp_scan_kernel<<<grid, BLOCK_SIZE>>>(
            d_strings, d_offsets, (uint32_t)nrecs,
            d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
            kmp.num_patterns, d_count);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Verify
        uint64_t gpu_raw_count = 0;
        CUDA_CHECK(cudaMemcpy(&gpu_raw_count, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        printf("  Qualifying (NOT LIKE): %lu  [%s]\n",
               gpu_raw_count, gpu_raw_count == cpu_raw_count ? "PASS" : "FAIL");

        // Benchmark
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int iter = 0; iter < n_iter; iter++) {
            CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint64_t)));
            raw_kmp_scan_kernel<<<grid, BLOCK_SIZE>>>(
                d_strings, d_offsets, (uint32_t)nrecs,
                d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
                kmp.num_patterns, d_count);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        float avg_ms = ms / n_iter;
        double throughput_gbps = (double)total_raw_bytes / (avg_ms * 1e6);

        printf("  Avg kernel time: %.3f ms (%d iterations)\n", avg_ms, n_iter);
        printf("  Throughput: %.2f GB/s (raw data basis)\n", throughput_gbps);

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaFree(d_strings));
        CUDA_CHECK(cudaFree(d_offsets));
        CUDA_CHECK(cudaFree(d_count));
    }

    // ── 9. GPU Test 2: FSST decomp + KMP scan ──
    printf("\n=== GPU Test 2: FSST Decomp + KMP Scan ===\n");
    {
        char* d_pages;
        FsstCompBlockKernelMeta* d_cb_meta;
        uint64_t* d_count;
        CUDA_CHECK(cudaMalloc(&d_pages, fpages.page_data.size()));
        CUDA_CHECK(cudaMalloc(&d_cb_meta, fpages.cb_metas.size() * sizeof(FsstCompBlockKernelMeta)));
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_pages, fpages.page_data.data(), fpages.page_data.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cb_meta, fpages.cb_metas.data(),
                              fpages.cb_metas.size() * sizeof(FsstCompBlockKernelMeta), cudaMemcpyHostToDevice));

        int grid = fpages.total_comp_blocks;

        // Warmup
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint64_t)));
        fsst_decomp_kmp_scan_kernel<<<grid, BLOCK_SIZE>>>(
            d_pages, d_cb_meta, fpages.total_comp_blocks,
            d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
            kmp.num_patterns, d_count);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Verify
        uint64_t gpu_fsst_count = 0;
        CUDA_CHECK(cudaMemcpy(&gpu_fsst_count, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        printf("  Qualifying (NOT LIKE): %lu  [%s]\n",
               gpu_fsst_count, gpu_fsst_count == cpu_raw_count ? "PASS" : "FAIL");

        // Benchmark
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int iter = 0; iter < n_iter; iter++) {
            CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint64_t)));
            fsst_decomp_kmp_scan_kernel<<<grid, BLOCK_SIZE>>>(
                d_pages, d_cb_meta, fpages.total_comp_blocks,
                d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
                kmp.num_patterns, d_count);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        float avg_ms = ms / n_iter;
        double throughput_raw_gbps = (double)total_raw_bytes / (avg_ms * 1e6);
        double throughput_comp_gbps = (double)total_comp_bytes / (avg_ms * 1e6);
        double throughput_page_gbps = (double)fpages.page_data.size() / (avg_ms * 1e6);

        printf("  Avg kernel time: %.3f ms (%d iterations)\n", avg_ms, n_iter);
        printf("  Throughput (raw data basis):        %.2f GB/s\n", throughput_raw_gbps);
        printf("  Throughput (compressed data basis):  %.2f GB/s\n", throughput_comp_gbps);
        printf("  Throughput (page data basis):        %.2f GB/s\n", throughput_page_gbps);

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaFree(d_pages));
        CUDA_CHECK(cudaFree(d_cb_meta));
        CUDA_CHECK(cudaFree(d_count));
    }

    // Comp block payload sizes to sweep
    uint32_t sweep_payloads[] = {44000, 20000, 9000, 4000, 2000};

    // ── Pre-compute CPU checksums for verification ──
    std::vector<uint32_t> cpu_checksums(nrecs);
    for (uint64_t i = 0; i < nrecs; i++) {
        cpu_checksums[i] = fnv1a(comments[i].c_str(), (uint32_t)comments[i].size());
    }

    // ── CPU decompress bandwidth measurement (skippable with --skip-cpu) ──
    //
    // Compare three implementations:
    //   (A) Scalar fsst_decompress() to output buffer (original FSST library)
    //   (B) 4-way interleaved to output buffer (ILP optimization)
    //   (C) 4-way interleaved checksum-only (matching GPU kernel)
    //
    if (!skip_cpu) {
    printf("\n=== CPU FSST Decompress Bandwidth ===\n");

    // Allocate output buffer for to-buffer modes
    uint32_t cpu_slot_size = 96;  // > max O_COMMENT (79)
    std::vector<unsigned char> cpu_output(nrecs * cpu_slot_size);
    std::vector<uint32_t> cpu_decomp_lens(nrecs);
    std::vector<uint32_t> cpu_cksums(nrecs);

    int cpu_thread_counts[] = {1, 2, 4, 8, 16, 24, 48};
    auto run_threaded = [&](const char* label, auto func) {
        for (int nthr : cpu_thread_counts) {
            // Warmup
            {
                std::vector<std::thread> threads;
                for (int t = 0; t < nthr; t++) {
                    threads.emplace_back([&, t]() {
                        uint64_t s = (uint64_t)t * nrecs / nthr;
                        uint64_t e = (uint64_t)(t + 1) * nrecs / nthr;
                        func(s, e);
                    });
                }
                for (auto& th : threads) th.join();
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < n_iter; iter++) {
                std::vector<std::thread> threads;
                for (int t = 0; t < nthr; t++) {
                    threads.emplace_back([&, t]() {
                        uint64_t s = (uint64_t)t * nrecs / nthr;
                        uint64_t e = (uint64_t)(t + 1) * nrecs / nthr;
                        func(s, e);
                    });
                }
                for (auto& th : threads) th.join();
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_iter;
            double gbps = (double)total_raw_bytes / (ms * 1e6);
            printf("  %-24s %2d thr: %8.3f ms  %7.2f GB/s\n", label, nthr, ms, gbps);
        }
    };

    // (A) Scalar to output buffer
    printf("  --- (A) Scalar fsst_decompress to buffer ---\n");
    run_threaded("scalar-output", [&](uint64_t s, uint64_t e) {
        for (uint64_t i = s; i < e; i++) {
            fsst_decompress(&decoder, comp_lens[i], comp_ptrs[i],
                            cpu_slot_size, cpu_output.data() + i * cpu_slot_size);
        }
    });

    // (B) 4-way interleaved to output buffer
    printf("  --- (B) Interleaved-4 to buffer ---\n");
    run_threaded("interleaved4-output", [&](uint64_t s, uint64_t e) {
        fsst_decompress_interleaved4_output(
            &decoder, s, e, comp_lens.data(),
            (const unsigned char**)comp_ptrs.data(),
            cpu_output.data(), cpu_decomp_lens.data(), cpu_slot_size);
    });

    // (C) 4-way interleaved checksum-only
    printf("  --- (C) Interleaved-4 checksum-only ---\n");
    run_threaded("interleaved4-cksum", [&](uint64_t s, uint64_t e) {
        fsst_decompress_interleaved4_cksum(
            &decoder, s, e, comp_lens.data(),
            (const unsigned char**)comp_ptrs.data(),
            cpu_cksums.data(), cpu_decomp_lens.data());
    });

    // Verify interleaved checksum results
    {
        bool ok = true;
        for (uint64_t i = 0; i < nrecs; i++) {
            if (cpu_cksums[i] != cpu_checksums[i]) {
                if (ok) fprintf(stderr, "  Interleaved cksum mismatch at rec %lu: "
                                "expected 0x%08x, got 0x%08x\n",
                                i, cpu_checksums[i], cpu_cksums[i]);
                ok = false;
            }
        }
        printf("  Interleaved checksum verification: %s\n", ok ? "PASS" : "FAIL");
    }

    // (D) Native g++ interleaved-4 output (compiled outside nvcc)
    printf("  --- (D) Native interleaved-4 to buffer ---\n");
    run_threaded("native-il4-output", [&](uint64_t s, uint64_t e) {
        fsst_decompress_interleaved4_output_native(
            &decoder, s, e, comp_lens.data(),
            (const unsigned char* const*)comp_ptrs.data(),
            cpu_output.data(), cpu_decomp_lens.data(), cpu_slot_size);
    });

    // (E) Native g++ interleaved-4 checksum-only
    printf("  --- (E) Native interleaved-4 checksum-only ---\n");
    run_threaded("native-il4-cksum", [&](uint64_t s, uint64_t e) {
        fsst_decompress_interleaved4_cksum_native(
            &decoder, s, e, comp_lens.data(),
            (const unsigned char* const*)comp_ptrs.data(),
            cpu_cksums.data(), cpu_decomp_lens.data());
    });

    // (F) AVX512 gather output
    printf("  --- (F) AVX512 gather to buffer ---\n");
    run_threaded("avx512-gather-output", [&](uint64_t s, uint64_t e) {
        fsst_decompress_avx512_output(
            &decoder, s, e, comp_lens.data(),
            (const unsigned char* const*)comp_ptrs.data(),
            cpu_output.data(), cpu_decomp_lens.data(), cpu_slot_size);
    });

    // (G) AVX512 gather checksum-only
    printf("  --- (G) AVX512 gather checksum-only ---\n");
    run_threaded("avx512-gather-cksum", [&](uint64_t s, uint64_t e) {
        fsst_decompress_avx512_cksum(
            &decoder, s, e, comp_lens.data(),
            (const unsigned char* const*)comp_ptrs.data(),
            cpu_cksums.data(), cpu_decomp_lens.data());
    });

    // Verify AVX512 checksum results
    {
        bool ok = true;
        for (uint64_t i = 0; i < nrecs; i++) {
            if (cpu_cksums[i] != cpu_checksums[i]) {
                if (ok) fprintf(stderr, "  AVX512 cksum mismatch at rec %lu: "
                                "expected 0x%08x, got 0x%08x\n",
                                i, cpu_checksums[i], cpu_cksums[i]);
                ok = false;
            }
        }
        printf("  AVX512 checksum verification: %s\n", ok ? "PASS" : "FAIL");
    }

    // (H) Native interleaved-4 cache-resident output
    printf("  --- (H) Native interleaved-4 cache-resident ---\n");
    run_threaded("native-il4-cache", [&](uint64_t s, uint64_t e) {
        fsst_decompress_cache_output_native(
            &decoder, s, e, comp_lens.data(),
            (const unsigned char* const*)comp_ptrs.data(),
            cpu_decomp_lens.data());
    });

    // (I) AVX512 gather cache-resident output
    printf("  --- (I) AVX512 gather cache-resident ---\n");
    run_threaded("avx512-gather-cache", [&](uint64_t s, uint64_t e) {
        fsst_decompress_avx512_cache_output(
            &decoder, s, e, comp_lens.data(),
            (const unsigned char* const*)comp_ptrs.data(),
            cpu_decomp_lens.data());
    });

    } // skip_cpu

    // ── 10. GPU Memory-Level Comparison ──
    // Two modes:
    //   Global memory: decompress to d_output buffer (HBM write)
    //   Cache/Register: decompress to local buffer (no HBM write)
    printf("\n=== GPU Memory-Level Comparison (payload=4000) ===\n");
    {
        uint32_t payload = 4000;
        auto fp = build_test_fsst_pages(decoder, (const unsigned char* const*)comp_ptrs.data(), comp_lens.data(), nrecs, payload);

        uint32_t max_cb_data = 0;
        for (auto& m : fp.cb_metas)
            max_cb_data = std::max(max_cb_data, m.comp_block_data_size);
        uint32_t smem_bytes = FSST_SMEM_SYMTAB_OFFSET + max_cb_data;
        smem_bytes = (smem_bytes + 7) & ~7u;

        char* d_pages;
        FsstCompBlockKernelMeta* d_cb_meta;
        uint32_t* d_decomp_lens;
        uint32_t slot_padded = (MAX_DECOMP_LEN + 7) & ~7u;
        CUDA_CHECK(cudaMalloc(&d_pages, fp.page_data.size()));
        CUDA_CHECK(cudaMalloc(&d_cb_meta, fp.cb_metas.size() * sizeof(FsstCompBlockKernelMeta)));
        CUDA_CHECK(cudaMalloc(&d_decomp_lens, nrecs * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_pages, fp.page_data.data(), fp.page_data.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cb_meta, fp.cb_metas.data(),
                              fp.cb_metas.size() * sizeof(FsstCompBlockKernelMeta), cudaMemcpyHostToDevice));

        int grid = fp.total_comp_blocks;

        // --- GPU Global memory mode: decompress to d_output ---
        {
            char* d_output;
            CUDA_CHECK(cudaMalloc(&d_output, (uint64_t)nrecs * slot_padded));

            CUDA_CHECK(cudaFuncSetAttribute(
                fsst_decompress_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));

            // Warmup
            fsst_decompress_kernel<<<grid, BLOCK_SIZE, smem_bytes>>>(
                d_pages, d_cb_meta, fp.total_comp_blocks,
                d_output, d_decomp_lens, nullptr, slot_padded);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark
            cudaEvent_t ev_start, ev_stop;
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));
            CUDA_CHECK(cudaEventRecord(ev_start));
            for (int iter = 0; iter < n_iter; iter++) {
                fsst_decompress_kernel<<<grid, BLOCK_SIZE, smem_bytes>>>(
                    d_pages, d_cb_meta, fp.total_comp_blocks,
                    d_output, d_decomp_lens, nullptr, slot_padded);
            }
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            float avg_ms = ms / n_iter;
            double raw_gbps = (double)total_raw_bytes / (avg_ms * 1e6);
            printf("  GPU Global memory (write to HBM): %8.3f ms  %7.2f GB/s (raw basis)\n",
                   avg_ms, raw_gbps);

            CUDA_CHECK(cudaEventDestroy(ev_start));
            CUDA_CHECK(cudaEventDestroy(ev_stop));
            CUDA_CHECK(cudaFree(d_output));
        }

        // --- GPU Cache/Register mode: decompress to local buffer ---
        {
            CUDA_CHECK(cudaFuncSetAttribute(
                fsst_decompress_cache_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));

            // Warmup
            fsst_decompress_cache_kernel<<<grid, BLOCK_SIZE, smem_bytes>>>(
                d_pages, d_cb_meta, fp.total_comp_blocks, d_decomp_lens);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark
            cudaEvent_t ev_start, ev_stop;
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));
            CUDA_CHECK(cudaEventRecord(ev_start));
            for (int iter = 0; iter < n_iter; iter++) {
                fsst_decompress_cache_kernel<<<grid, BLOCK_SIZE, smem_bytes>>>(
                    d_pages, d_cb_meta, fp.total_comp_blocks, d_decomp_lens);
            }
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            float avg_ms = ms / n_iter;
            double raw_gbps = (double)total_raw_bytes / (avg_ms * 1e6);
            printf("  GPU Cache/Register (local buf):   %8.3f ms  %7.2f GB/s (raw basis)\n",
                   avg_ms, raw_gbps);

            CUDA_CHECK(cudaEventDestroy(ev_start));
            CUDA_CHECK(cudaEventDestroy(ev_stop));
        }

        // --- GPU Coalesced writeback: 2-pass decompress to staging → global ---
        {
            // Compute smem for coalesced kernel:
            // decoder(2496) + seg_data + offsets(nrecs+1)*4 + staging(nrecs*MAX_DECOMP_LEN+8)
            uint32_t max_nrecs = 0;
            for (auto& m : fp.cb_metas)
                max_nrecs = std::max(max_nrecs, m.nrecs);
            uint32_t coal_offsets_start = 2496 + ((max_cb_data + 3) & ~3u);
            uint32_t coal_staging_start = coal_offsets_start + (max_nrecs + 1) * sizeof(uint32_t);
            coal_staging_start = (coal_staging_start + 7) & ~7u;
            uint32_t coal_smem = coal_staging_start + max_nrecs * MAX_DECOMP_LEN + 8;
            coal_smem = (coal_smem + 7) & ~7u;

            if (coal_smem > 164 * 1024) {
                printf("  GPU Coalesced writeback:           SKIPPED (smem %u > 164KB)\n",
                       coal_smem);
            } else {
                char* d_output;
                CUDA_CHECK(cudaMalloc(&d_output, (uint64_t)nrecs * MAX_DECOMP_LEN));

                CUDA_CHECK(cudaFuncSetAttribute(
                    decompress_string_with_fsst,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    coal_smem));

                // Warmup
                decompress_string_with_fsst<<<grid, BLOCK_SIZE, coal_smem>>>(
                    d_pages, d_cb_meta, fp.total_comp_blocks,
                    d_output, d_decomp_lens, MAX_DECOMP_LEN);
                CUDA_CHECK(cudaDeviceSynchronize());

                // Verify decomp_lens
                {
                    std::vector<uint32_t> h_lens(nrecs);
                    CUDA_CHECK(cudaMemcpy(h_lens.data(), d_decomp_lens,
                                          nrecs * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                    int bad = 0;
                    for (uint64_t i = 0; i < nrecs; i++) {
                        if (h_lens[i] != (uint32_t)comments[i].size()) {
                            if (bad < 3)
                                fprintf(stderr, "  coalesced rec %lu: len expected %u, got %u\n",
                                        i, (uint32_t)comments[i].size(), h_lens[i]);
                            bad++;
                        }
                    }
                    if (bad > 0)
                        fprintf(stderr, "  coalesced kernel: %d len mismatches\n", bad);
                    else
                        printf("  Coalesced kernel length verification: PASS\n");
                }

                // Benchmark
                cudaEvent_t ev_start, ev_stop;
                CUDA_CHECK(cudaEventCreate(&ev_start));
                CUDA_CHECK(cudaEventCreate(&ev_stop));
                CUDA_CHECK(cudaEventRecord(ev_start));
                for (int iter = 0; iter < n_iter; iter++) {
                    decompress_string_with_fsst<<<grid, BLOCK_SIZE, coal_smem>>>(
                        d_pages, d_cb_meta, fp.total_comp_blocks,
                        d_output, d_decomp_lens, MAX_DECOMP_LEN);
                }
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));

                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
                float avg_ms = ms / n_iter;
                double raw_gbps = (double)total_raw_bytes / (avg_ms * 1e6);
                printf("  GPU Coalesced writeback (2-pass):  %8.3f ms  %7.2f GB/s (raw basis)\n",
                       avg_ms, raw_gbps);

                CUDA_CHECK(cudaEventDestroy(ev_start));
                CUDA_CHECK(cudaEventDestroy(ev_stop));
                CUDA_CHECK(cudaFree(d_output));
            }
        }

        CUDA_CHECK(cudaFree(d_pages));
        CUDA_CHECK(cudaFree(d_cb_meta));
        CUDA_CHECK(cudaFree(d_decomp_lens));
    }

    // ── 11. Decompress-only throughput sweep (checksum verification) ──
    printf("\n=== Decompress-Only Kernel: Comp Block Size Sweep ===\n");
    printf("  %-12s %8s %8s %10s %10s\n",
           "Payload", "Blocks", "Time(ms)", "Raw GB/s", "Comp GB/s");
    printf("  %-12s %8s %8s %10s %10s\n",
           "---------", "------", "-------", "--------", "---------");

    for (uint32_t payload : sweep_payloads) {
        auto fp = build_test_fsst_pages(decoder, (const unsigned char* const*)comp_ptrs.data(), comp_lens.data(), nrecs, payload);

        uint32_t max_cb_data = 0;
        for (auto& m : fp.cb_metas)
            max_cb_data = std::max(max_cb_data, m.comp_block_data_size);
        uint32_t smem_bytes = FSST_SMEM_SYMTAB_OFFSET + max_cb_data;
        smem_bytes = (smem_bytes + 7) & ~7u;

        char* d_pages;
        FsstCompBlockKernelMeta* d_cb_meta;
        uint32_t* d_decomp_lens;
        uint32_t* d_checksums;
        CUDA_CHECK(cudaMalloc(&d_pages, fp.page_data.size()));
        CUDA_CHECK(cudaMalloc(&d_cb_meta, fp.cb_metas.size() * sizeof(FsstCompBlockKernelMeta)));
        uint32_t slot_padded = (MAX_DECOMP_LEN + 7) & ~7u;
        CUDA_CHECK(cudaMalloc(&d_decomp_lens, nrecs * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_checksums, nrecs * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_pages, fp.page_data.data(), fp.page_data.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cb_meta, fp.cb_metas.data(),
                              fp.cb_metas.size() * sizeof(FsstCompBlockKernelMeta), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaFuncSetAttribute(
            fsst_decompress_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes));

        int grid = fp.total_comp_blocks;

        // Warmup
        fsst_decompress_kernel<<<grid, BLOCK_SIZE, smem_bytes>>>(
            d_pages, d_cb_meta, fp.total_comp_blocks,
            nullptr, d_decomp_lens, d_checksums, slot_padded);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Verify: CPU-side decompress from page data, then compare with GPU
        {
            // First verify page data on CPU
            bool cpu_page_ok = true;
            unsigned char dbuf[256];
            uint8_t raw_symtab[FSST_SYMTAB_TOTAL];
            memset(raw_symtab, 0, sizeof(raw_symtab));
            memcpy(raw_symtab, decoder.len, 255);
            memcpy(raw_symtab + FSST_SYMTAB_LEN_BYTES, decoder.symbol, 255 * sizeof(uint64_t));

            for (uint32_t si = 0; si < fp.cb_metas.size() && cpu_page_ok; si++) {
                auto& meta = fp.cb_metas[si];
                const uint8_t* page_base = fp.page_data.data() + meta.page_byte_offset;
                const uint8_t* cb_base = page_base + meta.comp_block_byte_offset;
                const uint16_t* otbl = (const uint16_t*)cb_base;
                uint32_t otbl_bytes = (meta.nrecs + 1) * sizeof(uint16_t);
                const uint8_t* cdata = cb_base + otbl_bytes;

                for (uint32_t r = 0; r < meta.nrecs; r++) {
                    uint32_t gid = meta.rec_base + r;
                    uint16_t cstart = otbl[r];
                    uint16_t clen = otbl[r+1] - cstart;
                    size_t dlen = fsst_decompress(&decoder, clen, cdata + cstart,
                                                  sizeof(dbuf), dbuf);
                    if (dlen != comments[gid].size()) {
                        // Compare with original compressed data
                        uint16_t orig_clen = (uint16_t)comp_lens[gid];
                        bool bytes_match = (clen == orig_clen) &&
                            memcmp(cdata + cstart, comp_ptrs[gid], clen) == 0;
                        fprintf(stderr, "  CPU-page rec %u (seg %u, local %u): "
                                "expected %u, got %zu, page_clen=%u, orig_clen=%u, "
                                "bytes_match=%d, nrecs_in_seg=%u\n",
                                gid, si, r, (uint32_t)comments[gid].size(), dlen,
                                clen, orig_clen, bytes_match, meta.nrecs);
                        cpu_page_ok = false;
                    }
                }
            }
            if (!cpu_page_ok) {
                fprintf(stderr, "  PAGE DATA CORRUPTED for payload=%u\n", payload);
                CUDA_CHECK(cudaFree(d_pages));
                CUDA_CHECK(cudaFree(d_cb_meta));
                CUDA_CHECK(cudaFree(d_checksums));
                CUDA_CHECK(cudaFree(d_decomp_lens));
                continue;
            }

            // GPU results: check lengths + FNV-1a checksums
            std::vector<uint32_t> h_lens(nrecs);
            std::vector<uint32_t> h_checksums(nrecs);
            CUDA_CHECK(cudaMemcpy(h_lens.data(), d_decomp_lens, nrecs * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_checksums.data(), d_checksums, nrecs * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            bool ok = true;
            int len_fail = 0, cksum_fail = 0;
            for (uint64_t i = 0; i < nrecs; i++) {
                bool len_ok = (h_lens[i] == (uint32_t)comments[i].size());
                bool cksum_ok = (h_checksums[i] == cpu_checksums[i]);
                if (!len_ok) {
                    if (len_fail < 5) {
                        fprintf(stderr, "  GPU rec %lu: len expected %u, got %u\n",
                                i, (uint32_t)comments[i].size(), h_lens[i]);
                    }
                    len_fail++;
                    ok = false;
                }
                if (!cksum_ok) {
                    if (cksum_fail < 5) {
                        fprintf(stderr, "  GPU rec %lu: checksum expected 0x%08x, got 0x%08x (len %u/%u)\n",
                                i, cpu_checksums[i], h_checksums[i],
                                h_lens[i], (uint32_t)comments[i].size());
                    }
                    cksum_fail++;
                    ok = false;
                }
            }
            if (!ok) {
                fprintf(stderr, "  decomp-only GPU verification FAILED for payload=%u "
                        "(%d len bad, %d checksum bad)\n",
                        payload, len_fail, cksum_fail);
                CUDA_CHECK(cudaFree(d_pages));
                CUDA_CHECK(cudaFree(d_cb_meta));
                CUDA_CHECK(cudaFree(d_checksums));
                CUDA_CHECK(cudaFree(d_decomp_lens));
                continue;
            }
        }

        // Benchmark
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int iter = 0; iter < n_iter; iter++) {
            fsst_decompress_kernel<<<grid, BLOCK_SIZE, smem_bytes>>>(
                d_pages, d_cb_meta, fp.total_comp_blocks,
                nullptr, d_decomp_lens, d_checksums, slot_padded);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        float avg_ms = ms / n_iter;
        double raw_gbps = (double)total_raw_bytes / (avg_ms * 1e6);
        double comp_gbps = (double)total_comp_bytes / (avg_ms * 1e6);

        printf("  %-12u %8u %8.3f %10.2f %10.2f\n",
               payload, fp.total_comp_blocks, avg_ms, raw_gbps, comp_gbps);

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaFree(d_pages));
        CUDA_CHECK(cudaFree(d_cb_meta));
        CUDA_CHECK(cudaFree(d_checksums));
        CUDA_CHECK(cudaFree(d_decomp_lens));
    }

    // ── 12. V2 fused decomp+KMP kernel: parameter sweep ──
    printf("\n=== Fused Decomp+KMP Kernel: Comp Block Size Sweep ===\n");
    printf("  %-12s %8s %8s %8s %10s %10s\n",
           "Payload", "Blocks", "Smem(B)", "Time(ms)", "Raw GB/s", "Comp GB/s");
    printf("  %-12s %8s %8s %8s %10s %10s\n",
           "---------", "------", "-------", "-------", "--------", "---------");

    for (uint32_t payload : sweep_payloads) {
        auto fp = build_test_fsst_pages(decoder, (const unsigned char* const*)comp_ptrs.data(), comp_lens.data(), nrecs, payload);

        uint32_t max_cb_data = 0;
        for (auto& m : fp.cb_metas)
            max_cb_data = std::max(max_cb_data, m.comp_block_data_size);
        uint32_t smem_bytes = FSST_SMEM_SYMTAB_OFFSET + max_cb_data;
        smem_bytes = (smem_bytes + 7) & ~7u;

        char* d_pages;
        FsstCompBlockKernelMeta* d_cb_meta;
        uint64_t* d_count;
        CUDA_CHECK(cudaMalloc(&d_pages, fp.page_data.size()));
        CUDA_CHECK(cudaMalloc(&d_cb_meta, fp.cb_metas.size() * sizeof(FsstCompBlockKernelMeta)));
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_pages, fp.page_data.data(), fp.page_data.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cb_meta, fp.cb_metas.data(),
                              fp.cb_metas.size() * sizeof(FsstCompBlockKernelMeta), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaFuncSetAttribute(
            decompress_scan_string_with_fsst,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes));

        int grid = fp.total_comp_blocks;

        // Warmup
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint64_t)));
        decompress_scan_string_with_fsst<<<grid, BLOCK_SIZE, smem_bytes>>>(
            d_pages, d_cb_meta, fp.total_comp_blocks,
            d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
            kmp.num_patterns, d_count);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Verify
        uint64_t cnt = 0;
        CUDA_CHECK(cudaMemcpy(&cnt, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        bool pass = (cnt == cpu_raw_count);

        // Benchmark
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int iter = 0; iter < n_iter; iter++) {
            CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint64_t)));
            decompress_scan_string_with_fsst<<<grid, BLOCK_SIZE, smem_bytes>>>(
                d_pages, d_cb_meta, fp.total_comp_blocks,
                d_patterns, d_next, d_pattern_offsets, d_pattern_lengths,
                kmp.num_patterns, d_count);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        float avg_ms = ms / n_iter;
        double raw_gbps = (double)total_raw_bytes / (avg_ms * 1e6);
        double comp_gbps = (double)total_comp_bytes / (avg_ms * 1e6);

        printf("  %-12u %8u %8u %8.3f %10.2f %10.2f  [%s]\n",
               payload, fp.total_comp_blocks, smem_bytes, avg_ms, raw_gbps, comp_gbps,
               pass ? "PASS" : "FAIL");

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaFree(d_pages));
        CUDA_CHECK(cudaFree(d_cb_meta));
        CUDA_CHECK(cudaFree(d_count));
    }

    // ── 13. Summary ──
    printf("\n=== Summary ===\n");
    printf("Records:           %lu\n", nrecs);
    printf("Raw size:          %.2f MB\n", total_raw_bytes / 1e6);
    printf("Compressed size:   %.2f MB\n", total_comp_bytes / 1e6);
    printf("Compression ratio: %.2fx\n", (double)total_raw_bytes / total_comp_bytes);
    printf("FSST pages (V1):   %u (%u comp blocks)\n", fpages.npages, fpages.total_comp_blocks);

    // Cleanup
    fsst_destroy(encoder);
    cudaFree(d_patterns);
    cudaFree(d_next);
    cudaFree(d_pattern_offsets);
    cudaFree(d_pattern_lengths);

    return 0;
}
