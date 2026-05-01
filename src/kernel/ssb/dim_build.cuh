#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common/pag_core.h"
#include "q2x.cuh"

// ============================================================
// GPU kernels for building dimension hash tables from
// decompressed pages on GPU. All kernels use flat row indexing
// via prefix_sum to handle different page layouts across fields.
// ============================================================

// ── Prefix sum extraction from pages ────────────────────────
// Reads nalloc from each page header, writes cumulative prefix sum.
// Output: d_prefix_sum[pg] = sum of nalloc[0..pg] (exclusive upper bound).
__global__ void dim_extract_prefix_sum_kernel(
    const char *__restrict__ pages,
    uint32_t page_size, uint32_t npages,
    uint64_t *__restrict__ d_prefix_sum)
{
    // Single-thread kernel (npages is small: ≤128 for dim tables)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t sum = 0;
    for (uint32_t pg = 0; pg < npages; pg++) {
        const pag_head *hdr = reinterpret_cast<const pag_head *>(
            pages + (uint64_t)pg * page_size);
        sum += hdr->nalloc;
        d_prefix_sum[pg] = sum;
    }
}

// ── DATE: INT32 key pages + INT32 value pages → HT ─────────
// 1 block per page, 256 threads per block.
// Both fields share the same page layout (both INT32).
__global__ void dim_build_date_ht_kernel(
    const char *__restrict__ key_pages,   // D_DATEKEY pages
    const char *__restrict__ val_pages,   // D_YEAR pages
    uint32_t npages, uint32_t page_size, uint32_t capacity,
    int32_t year_offset,
    int32_t *ht_keys, int32_t *ht_values, uint32_t ht_mask)
{
    uint32_t pg = blockIdx.x;
    if (pg >= npages) return;

    const char *kp = key_pages + (uint64_t)pg * page_size;
    const char *vp = val_pages + (uint64_t)pg * page_size;
    uint32_t nalloc = reinterpret_cast<const pag_head *>(kp)->nalloc;
    if (nalloc > capacity) nalloc = capacity;

    const int32_t *keys = reinterpret_cast<const int32_t *>(kp + 12);
    const int32_t *vals = reinterpret_cast<const int32_t *>(vp + 12);

    for (uint32_t r = threadIdx.x; r < nalloc; r += blockDim.x) {
        ssb_ht_insert(ht_keys, ht_values, ht_mask,
                      keys[r], vals[r] - year_offset);
    }
}

// ── DATE: INT32 key + INT32 value + optional aux INT32 filter → HT ──
// filter_mode: 0=none, 1=year range [lo..hi], 2=aux equality (aux[r]==lo)
__global__ void dim_build_date_ht_ext_kernel(
    const char *__restrict__ key_pages,
    const char *__restrict__ val_pages,
    const char *__restrict__ aux_pages,   // D_YEARMONTHNUM(mode 2) or D_WEEKNUMINYEAR(mode 3)
    uint32_t npages, uint32_t page_size, uint32_t capacity,
    int32_t year_offset,
    int32_t filter_mode, int32_t filter_lo, int32_t filter_hi,
    int32_t *ht_keys, int32_t *ht_values, uint32_t ht_mask)
{
    uint32_t pg = blockIdx.x;
    if (pg >= npages) return;

    const char *kp = key_pages + (uint64_t)pg * page_size;
    const char *vp = val_pages + (uint64_t)pg * page_size;
    uint32_t nalloc = reinterpret_cast<const pag_head *>(kp)->nalloc;
    if (nalloc > capacity) nalloc = capacity;

    const int32_t *keys = reinterpret_cast<const int32_t *>(kp + 12);
    const int32_t *vals = reinterpret_cast<const int32_t *>(vp + 12);
    const int32_t *aux = nullptr;
    if (aux_pages)
        aux = reinterpret_cast<const int32_t *>(aux_pages + (uint64_t)pg * page_size + 12);

    for (uint32_t r = threadIdx.x; r < nalloc; r += blockDim.x) {
        bool pass = true;
        if (filter_mode == 1) {
            pass = (vals[r] >= filter_lo && vals[r] <= filter_hi);
        } else if (filter_mode == 2) {
            pass = aux && (aux[r] == filter_lo);
        } else if (filter_mode == 3) {
            // year == filter_lo AND aux_field == filter_hi (Q1.3: year+weeknuminyear)
            pass = (vals[r] == filter_lo) && aux && (aux[r] == filter_hi);
        }
        if (pass) {
            // mode 3 (membership-only): value=0; others: year index
            int32_t value = (filter_mode == 3) ? 0 : (vals[r] - year_offset);
            ssb_ht_insert(ht_keys, ht_values, ht_mask,
                          keys[r], value);
        }
    }
}

// ── SUPPLIER/CUSTOMER: INT32 key + fixed-size CHAR filter → HT ──
// Both fields share the same page count (same table).
// key pages are INT32, str pages are CHAR — same nalloc per page.
__global__ void dim_build_filtered_ht_kernel(
    const char *__restrict__ key_pages,
    const char *__restrict__ str_pages,
    uint32_t npages, uint32_t page_size,
    uint32_t key_capacity,
    uint32_t str_stored_size,
    const char *__restrict__ d_target,
    uint32_t target_len,
    int32_t match_value,
    int32_t *ht_keys, int32_t *ht_values, uint32_t ht_mask)
{
    uint32_t pg = blockIdx.x;
    if (pg >= npages) return;

    const char *kp = key_pages + (uint64_t)pg * page_size;
    const char *sp = str_pages + (uint64_t)pg * page_size;
    uint32_t nalloc = reinterpret_cast<const pag_head *>(kp)->nalloc;
    if (nalloc > key_capacity) nalloc = key_capacity;

    const int32_t *keys = reinterpret_cast<const int32_t *>(kp + 12);
    const char *strs = sp + 12;

    for (uint32_t r = threadIdx.x; r < nalloc; r += blockDim.x) {
        const char *s = strs + (uint64_t)r * str_stored_size;
        bool match = true;
        for (uint32_t c = 0; c < target_len; c++) {
            if (s[c] != d_target[c]) { match = false; break; }
        }
        if (match) {
            ssb_ht_insert(ht_keys, ht_values, ht_mask,
                          keys[r], match_value);
        }
    }
}

// ============================================================
// GPU dict for dim table group-by (FNV-1a hash, atomic insert)
// ============================================================
static constexpr uint32_t DIM_DICT_MAX_STRLEN = 64;
static constexpr uint32_t DIM_DICT_CAP = 1024;
static constexpr uint32_t DIM_DICT_MASK = DIM_DICT_CAP - 1;

struct DimGpuDict {
    uint64_t *d_hashes    = nullptr;
    char     *d_strs      = nullptr;
    uint16_t *d_lens      = nullptr;
    uint32_t *d_type_ids  = nullptr;
    uint32_t *d_counter   = nullptr;

    void alloc() {
        cudaMalloc(&d_hashes,  DIM_DICT_CAP * sizeof(uint64_t));
        cudaMalloc(&d_strs,    DIM_DICT_CAP * DIM_DICT_MAX_STRLEN);
        cudaMalloc(&d_lens,    DIM_DICT_CAP * sizeof(uint16_t));
        cudaMalloc(&d_type_ids,DIM_DICT_CAP * sizeof(uint32_t));
        cudaMalloc(&d_counter, sizeof(uint32_t));
        cudaMemset(d_hashes,  0xFF, DIM_DICT_CAP * sizeof(uint64_t));
        cudaMemset(d_type_ids,0xFF, DIM_DICT_CAP * sizeof(uint32_t));
        cudaMemset(d_counter, 0, sizeof(uint32_t));
    }
    void reset() {
        cudaMemset(d_hashes,  0xFF, DIM_DICT_CAP * sizeof(uint64_t));
        cudaMemset(d_type_ids,0xFF, DIM_DICT_CAP * sizeof(uint32_t));
        cudaMemset(d_counter, 0, sizeof(uint32_t));
    }
    void free_all() {
        cudaFree(d_hashes); cudaFree(d_strs); cudaFree(d_lens);
        cudaFree(d_type_ids); cudaFree(d_counter);
        d_hashes = nullptr; d_strs = nullptr; d_lens = nullptr;
        d_type_ids = nullptr; d_counter = nullptr;
    }
};

static std::vector<std::string> dim_download_dict(const DimGpuDict &gd)
{
    uint32_t n;
    cudaMemcpy(&n, gd.d_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (n == 0) return {};
    std::vector<char> h_strs(DIM_DICT_CAP * DIM_DICT_MAX_STRLEN);
    std::vector<uint16_t> h_lens(DIM_DICT_CAP);
    std::vector<uint32_t> h_ids(DIM_DICT_CAP);
    cudaMemcpy(h_strs.data(), gd.d_strs, h_strs.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lens.data(), gd.d_lens, h_lens.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ids.data(), gd.d_type_ids, h_ids.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::vector<std::string> result(n);
    for (uint32_t slot = 0; slot < DIM_DICT_CAP; slot++) {
        if (h_ids[slot] != UINT32_MAX && h_ids[slot] < n)
            result[h_ids[slot]] = std::string(
                h_strs.data() + slot * DIM_DICT_MAX_STRLEN, h_lens[slot]);
    }
    return result;
}

// ── FNV-1a 64-bit hash (device) ────────────────────────────
__device__ __forceinline__ uint64_t dim_fnv1a64(const char *s, uint32_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < len; i++)
        h = (h ^ (uint8_t)s[i]) * 0x100000001b3ULL;
    return h;
}

// ── CHAR filter + dict kernel (flat row index via prefix_sum) ──
// Per page: reads CHAR records, applies filter, optionally builds dict.
// Output indexed by flat row: d_filter[grow], d_values[grow].
struct DimCharFilterParams {
    const char *pages;
    uint32_t page_size;
    uint32_t npages;
    uint32_t field_size;
    uint32_t aligned_field_size;
    const uint64_t *d_prefix_sum;
    // Filter
    int32_t filter_mode;       // 0=none, 1=eq, 2=in, 3=range
    char pred_strs[4][DIM_DICT_MAX_STRLEN];
    uint32_t pred_lens[4];
    uint32_t n_preds;
    // Dict
    bool enable_dict;
    uint64_t *d_dict_hashes;
    char     *d_dict_strs;
    uint16_t *d_dict_lens;
    uint32_t *d_dict_type_ids;
    uint32_t *d_id_counter;
    // I/O
    const uint8_t *d_prefilter;
    uint8_t *d_filter;
    int32_t *d_values;
};

static constexpr int32_t DIM_FILT_NONE   = 0;
static constexpr int32_t DIM_FILT_EQ     = 1;
static constexpr int32_t DIM_FILT_IN     = 2;
static constexpr int32_t DIM_FILT_RANGE  = 3;
static constexpr int32_t DIM_FILT_PREFIX = 4;  // prefix match

__device__ __forceinline__ int dim_strcmp_dev(
    const char *a, uint32_t al, const char *b, uint32_t bl) {
    uint32_t n = al < bl ? al : bl;
    for (uint32_t i = 0; i < n; i++) {
        if ((uint8_t)a[i] < (uint8_t)b[i]) return -1;
        if ((uint8_t)a[i] > (uint8_t)b[i]) return  1;
    }
    if (al < bl) return -1;
    if (al > bl) return  1;
    return 0;
}

__global__ void dim_char_filter_kernel(DimCharFilterParams p)
{
    uint32_t pg = blockIdx.x;
    if (pg >= p.npages) return;

    const char *page = p.pages + (uint64_t)pg * p.page_size;
    uint32_t nalloc = reinterpret_cast<const pag_head *>(page)->nalloc;
    if (nalloc == 0) return;

    uint64_t row_base = 0;
    if (p.d_prefix_sum) {
        row_base = (pg == 0) ? 0 : p.d_prefix_sum[pg - 1];
    } else {
        for (uint32_t i = 0; i < pg; i++)
            row_base += reinterpret_cast<const pag_head *>(
                p.pages + (uint64_t)i * p.page_size)->nalloc;
    }
    const char *rec_base = page + sizeof(pag_head);

    for (uint32_t r = threadIdx.x; r < nalloc; r += blockDim.x) {
        uint64_t grow = row_base + r;

        if (p.d_prefilter && !p.d_prefilter[grow]) {
            if (p.d_filter) p.d_filter[grow] = 0;
            if (p.d_values) p.d_values[grow] = -1;
            continue;
        }

        const char *rec = rec_base + (uint64_t)r * p.aligned_field_size;

        // Trim trailing spaces
        uint32_t dlen = p.field_size;
        while (dlen > 0 && rec[dlen - 1] == ' ') dlen--;

        bool pass = true;
        if (p.filter_mode == DIM_FILT_EQ) {
            pass = (dlen == p.pred_lens[0]);
            for (uint32_t k = 0; pass && k < dlen; k++)
                if (rec[k] != p.pred_strs[0][k]) pass = false;
        } else if (p.filter_mode == DIM_FILT_PREFIX) {
            pass = (dlen >= p.pred_lens[0]);
            for (uint32_t k = 0; pass && k < p.pred_lens[0]; k++)
                if (rec[k] != p.pred_strs[0][k]) pass = false;
        } else if (p.filter_mode == DIM_FILT_IN) {
            pass = false;
            for (uint32_t px = 0; px < p.n_preds && !pass; px++) {
                if (dlen != p.pred_lens[px]) continue;
                bool eq = true;
                for (uint32_t k = 0; k < dlen; k++)
                    if (rec[k] != p.pred_strs[px][k]) { eq = false; break; }
                if (eq) pass = true;
            }
        } else if (p.filter_mode == DIM_FILT_RANGE) {
            int c0 = dim_strcmp_dev(rec, dlen, p.pred_strs[0], p.pred_lens[0]);
            int c1 = dim_strcmp_dev(rec, dlen, p.pred_strs[1], p.pred_lens[1]);
            pass = (c0 >= 0 && c1 <= 0);
        }

        if (p.d_filter) p.d_filter[grow] = pass ? 1 : 0;

        if (p.d_values) {
            if (!pass) {
                p.d_values[grow] = -1;
            } else if (p.enable_dict) {
                uint64_t h = dim_fnv1a64(rec, dlen);
                uint32_t ds = (uint32_t)h & DIM_DICT_MASK;
                while (true) {
                    uint64_t prev = atomicCAS(
                        reinterpret_cast<unsigned long long *>(&p.d_dict_hashes[ds]),
                        (unsigned long long)UINT64_MAX,
                        (unsigned long long)h);
                    if (prev == UINT64_MAX) {
                        uint32_t nid = atomicAdd(p.d_id_counter, 1);
                        char *dst = p.d_dict_strs + (uint64_t)ds * DIM_DICT_MAX_STRLEN;
                        for (uint32_t k = 0; k < dlen; k++) dst[k] = rec[k];
                        p.d_dict_lens[ds] = (uint16_t)dlen;
                        __threadfence();
                        p.d_dict_type_ids[ds] = nid;
                        p.d_values[grow] = (int32_t)nid;
                        break;
                    }
                    if (prev == h) {
                        uint32_t eid;
                        do { __threadfence(); eid = p.d_dict_type_ids[ds]; }
                        while (eid == UINT32_MAX);
                        p.d_values[grow] = (int32_t)eid;
                        break;
                    }
                    ds = (ds + 1) & DIM_DICT_MASK;
                }
            } else {
                p.d_values[grow] = 0;
            }
        }
    }
}

// ── Flat HT build (same as datapathfusion's dpf_dim_build_ht_kernel) ──
// All arrays are flat [nrows]. d_filter/d_values indexed by flat row.
__global__ void dim_build_ht_flat_kernel(
    const int32_t *__restrict__ d_keys,
    const uint8_t *__restrict__ d_filter,
    const int32_t *__restrict__ d_values,
    uint64_t nrows,
    int32_t *ht_keys, int32_t *ht_values, uint32_t ht_mask)
{
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nrows) return;
    if (d_filter && !d_filter[gid]) return;
    int32_t val = d_values ? d_values[gid] : 0;
    if (val < 0) return;
    ssb_ht_insert(ht_keys, ht_values, ht_mask, d_keys[gid], val);
}

// ── Paged HT build: reads INT32 keys directly from pages ────
// Eliminates prefix_sum + flatten. 1 block per page, 256 threads.
// d_filter/d_values are flat arrays indexed by global row.
__global__ void dim_build_ht_paged_kernel(
    const char *__restrict__ key_pages,
    uint32_t npages, uint32_t page_size, uint32_t key_capacity,
    const uint8_t *__restrict__ d_filter,
    const int32_t *__restrict__ d_values,
    int32_t *ht_keys, int32_t *ht_values, uint32_t ht_mask)
{
    uint32_t pg = blockIdx.x;
    if (pg >= npages) return;

    const char *kp = key_pages + (uint64_t)pg * page_size;
    uint32_t nalloc = reinterpret_cast<const pag_head *>(kp)->nalloc;
    if (nalloc > key_capacity) nalloc = key_capacity;
    const int32_t *keys = reinterpret_cast<const int32_t *>(kp + 12);

    uint64_t row_base = 0;
    for (uint32_t i = 0; i < pg; i++)
        row_base += reinterpret_cast<const pag_head *>(
            key_pages + (uint64_t)i * page_size)->nalloc;

    for (uint32_t r = threadIdx.x; r < nalloc; r += blockDim.x) {
        uint64_t grow = row_base + r;
        if (d_filter && !d_filter[grow]) continue;
        int32_t val = d_values ? d_values[grow] : 0;
        if (val < 0) continue;
        ssb_ht_insert(ht_keys, ht_values, ht_mask, keys[r], val);
    }
}

// ── INT32 pages flatten (prefix_sum-based, output int32_t) ──
// Like ssb_flatten_int32_pages_ps_kernel but outputs int32_t, not uint64_t.
__global__ void dim_flatten_int32_kernel(
    const char *__restrict__ pages,
    uint32_t page_size,
    const uint64_t *__restrict__ prefix_sum,
    uint32_t npages,
    uint64_t nrows,
    int32_t *__restrict__ out)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrows) return;

    // Binary search for page
    uint32_t lo = 0, hi = npages;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (prefix_sum[mid] <= idx) lo = mid + 1; else hi = mid;
    }
    uint32_t page_idx = lo;
    uint32_t local_idx = (page_idx == 0) ? (uint32_t)idx
                                         : (uint32_t)(idx - prefix_sum[page_idx - 1]);

    const int32_t *values = reinterpret_cast<const int32_t *>(
        pages + (uint64_t)page_idx * page_size + 12);
    out[idx] = values[local_idx];
}
