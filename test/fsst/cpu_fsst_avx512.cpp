// ============================================================
// CPU AVX512-optimized FSST decompression
//
// Compiled with g++ -O3 -mavx512f -mavx512bw (NOT nvcc) to
// ensure full AVX512 optimization.
//
// Two optimization strategies:
//
//  1. AVX512 gather: Process 8 codes at a time. Escape-free
//     fast path uses _mm512_i64gather_epi64 to look up 8
//     symbols and 8 lengths simultaneously, then write/hash
//     sequentially. posOut serial chain remains but gather
//     eliminates 16 scalar loads → 2 vector gathers.
//
//  2. 4-way interleaving: Process 4 strings simultaneously.
//     Each string has an independent posOut/hash chain,
//     giving the OoO engine 4× the ILP to fill load ports.
//     Compiled natively (not via nvcc) for optimal codegen.
// ============================================================

#include "cpu_fsst_avx512.h"
#include <immintrin.h>
#include <cstring>

// ── AVX512 gather: 8-code fast path (output-to-buffer) ──

void fsst_decompress_avx512_output(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    unsigned char* output,
    uint32_t* decomp_lens,
    uint32_t slot_size)
{
    const unsigned char* __restrict__ L = (const unsigned char*)decoder->len;
    const unsigned long long* __restrict__ S = (const unsigned long long*)decoder->symbol;
    const __m128i v_esc = _mm_set1_epi8((char)0xFF);
    const __m512i v_byte_mask = _mm512_set1_epi64(0xFF);

    for (uint64_t i = start; i < end; i++) {
        size_t cl = comp_lens[i];
        const unsigned char* cp = comp_ptrs[i];
        unsigned char* out = output + (uint64_t)i * slot_size;
        size_t posIn = 0;
        uint32_t posOut = 0;

        // AVX512 fast path: process 8 codes at a time
        while (posIn + 8 <= cl) {
            // Load 8 code bytes
            __m128i v8 = _mm_loadl_epi64((const __m128i*)(cp + posIn));

            // Check for escapes
            __m128i cmp = _mm_cmpeq_epi8(v8, v_esc);
            int esc = _mm_movemask_epi8(cmp) & 0xFF;

            if (__builtin_expect(esc == 0, 1)) {
                // No escapes: gather 8 symbols + 8 lengths with AVX512
                __m512i indices = _mm512_cvtepu8_epi64(v8);

                __m512i syms = _mm512_i64gather_epi64(indices, (const long long*)S, 8);
                __m512i lens = _mm512_i64gather_epi64(indices, (const long long*)L, 1);
                lens = _mm512_and_si512(lens, v_byte_mask);

                // Extract to arrays and write sequentially
                uint64_t s_arr[8], l_arr[8];
                _mm512_storeu_si512(s_arr, syms);
                _mm512_storeu_si512(l_arr, lens);

                memcpy(out + posOut, &s_arr[0], 8); posOut += (uint32_t)l_arr[0];
                memcpy(out + posOut, &s_arr[1], 8); posOut += (uint32_t)l_arr[1];
                memcpy(out + posOut, &s_arr[2], 8); posOut += (uint32_t)l_arr[2];
                memcpy(out + posOut, &s_arr[3], 8); posOut += (uint32_t)l_arr[3];
                memcpy(out + posOut, &s_arr[4], 8); posOut += (uint32_t)l_arr[4];
                memcpy(out + posOut, &s_arr[5], 8); posOut += (uint32_t)l_arr[5];
                memcpy(out + posOut, &s_arr[6], 8); posOut += (uint32_t)l_arr[6];
                memcpy(out + posOut, &s_arr[7], 8); posOut += (uint32_t)l_arr[7];

                posIn += 8;
            } else {
                // Process non-escape codes before first escape
                int safe = __builtin_ctz(esc);
                for (int k = 0; k < safe; k++) {
                    unsigned char code = cp[posIn++];
                    memcpy(out + posOut, &S[code], 8);
                    posOut += L[code];
                }
                // Handle escape
                posIn++;  // skip 0xFF
                out[posOut++] = cp[posIn++];
            }
        }

        // Scalar tail
        while (posIn < cl) {
            unsigned char code = cp[posIn++];
            if (__builtin_expect(code < 255, 1)) {
                memcpy(out + posOut, &S[code], 8);
                posOut += L[code];
            } else {
                out[posOut++] = cp[posIn++];
            }
        }

        decomp_lens[i] = posOut;
    }
}

// ── AVX512 gather: 8-code fast path (checksum-only) ──

void fsst_decompress_avx512_cksum(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    uint32_t* checksums,
    uint32_t* decomp_lens)
{
    const unsigned char* __restrict__ L = (const unsigned char*)decoder->len;
    const unsigned long long* __restrict__ S = (const unsigned long long*)decoder->symbol;
    const __m128i v_esc = _mm_set1_epi8((char)0xFF);
    const __m512i v_byte_mask = _mm512_set1_epi64(0xFF);

    for (uint64_t i = start; i < end; i++) {
        size_t cl = comp_lens[i];
        const unsigned char* cp = comp_ptrs[i];
        size_t posIn = 0;
        uint32_t posOut = 0;
        uint32_t hash = 2166136261u;

        while (posIn + 8 <= cl) {
            __m128i v8 = _mm_loadl_epi64((const __m128i*)(cp + posIn));
            __m128i cmp = _mm_cmpeq_epi8(v8, v_esc);
            int esc = _mm_movemask_epi8(cmp) & 0xFF;

            if (__builtin_expect(esc == 0, 1)) {
                __m512i indices = _mm512_cvtepu8_epi64(v8);
                __m512i syms = _mm512_i64gather_epi64(indices, (const long long*)S, 8);
                __m512i lens = _mm512_i64gather_epi64(indices, (const long long*)L, 1);
                lens = _mm512_and_si512(lens, v_byte_mask);

                uint64_t s_arr[8], l_arr[8];
                _mm512_storeu_si512(s_arr, syms);
                _mm512_storeu_si512(l_arr, lens);

                for (int k = 0; k < 8; k++) {
                    uint64_t sval = s_arr[k];
                    uint32_t slen = (uint32_t)l_arr[k];
                    for (uint32_t j = 0; j < slen; j++) {
                        hash ^= (uint8_t)(sval & 0xFF);
                        hash *= 16777619u;
                        sval >>= 8;
                    }
                    posOut += slen;
                }
                posIn += 8;
            } else {
                int safe = __builtin_ctz(esc);
                for (int k = 0; k < safe; k++) {
                    unsigned char code = cp[posIn++];
                    unsigned char slen = L[code];
                    unsigned long long sval = S[code];
                    for (unsigned char j = 0; j < slen; j++) {
                        hash ^= (unsigned char)(sval & 0xFF);
                        hash *= 16777619u;
                        sval >>= 8;
                    }
                    posOut += slen;
                }
                posIn++;
                unsigned char b = cp[posIn++];
                hash ^= b; hash *= 16777619u;
                posOut++;
            }
        }

        while (posIn < cl) {
            unsigned char code = cp[posIn++];
            if (__builtin_expect(code < 255, 1)) {
                unsigned char slen = L[code];
                unsigned long long sval = S[code];
                for (unsigned char j = 0; j < slen; j++) {
                    hash ^= (unsigned char)(sval & 0xFF);
                    hash *= 16777619u;
                    sval >>= 8;
                }
                posOut += slen;
            } else {
                hash ^= cp[posIn++]; hash *= 16777619u;
                posOut++;
            }
        }

        checksums[i] = hash;
        decomp_lens[i] = posOut;
    }
}

// ── 4-way interleaved (output-to-buffer), native g++ compile ──

void fsst_decompress_interleaved4_output_native(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    unsigned char* output,
    uint32_t* decomp_lens,
    uint32_t slot_size)
{
    const unsigned char* __restrict__ L = (const unsigned char*)decoder->len;
    const unsigned long long* __restrict__ S = (const unsigned long long*)decoder->symbol;

    uint64_t i = start;

    for (; i + 4 <= end; i += 4) {
        const unsigned char* cp[4] = {comp_ptrs[i], comp_ptrs[i+1], comp_ptrs[i+2], comp_ptrs[i+3]};
        size_t cl[4] = {comp_lens[i], comp_lens[i+1], comp_lens[i+2], comp_lens[i+3]};
        unsigned char* out[4] = {
            output + (uint64_t)i * slot_size,
            output + (uint64_t)(i+1) * slot_size,
            output + (uint64_t)(i+2) * slot_size,
            output + (uint64_t)(i+3) * slot_size
        };
        size_t pi[4] = {0, 0, 0, 0};
        uint32_t po[4] = {0, 0, 0, 0};

        while (pi[0] < cl[0] && pi[1] < cl[1] && pi[2] < cl[2] && pi[3] < cl[3]) {
            for (int s = 0; s < 4; s++) {
                unsigned char code = cp[s][pi[s]++];
                if (__builtin_expect(code < 255, 1)) {
                    memcpy(out[s] + po[s], &S[code], 8);
                    po[s] += L[code];
                } else {
                    out[s][po[s]++] = cp[s][pi[s]++];
                }
            }
        }

        for (int s = 0; s < 4; s++) {
            while (pi[s] < cl[s]) {
                unsigned char code = cp[s][pi[s]++];
                if (__builtin_expect(code < 255, 1)) {
                    memcpy(out[s] + po[s], &S[code], 8);
                    po[s] += L[code];
                } else {
                    out[s][po[s]++] = cp[s][pi[s]++];
                }
            }
            decomp_lens[i + s] = po[s];
        }
    }

    for (; i < end; i++) {
        unsigned char buf[256];
        decomp_lens[i] = (uint32_t)fsst_decompress(
            decoder, comp_lens[i], comp_ptrs[i], sizeof(buf), buf);
        memcpy(output + (uint64_t)i * slot_size, buf, decomp_lens[i]);
    }
}

// ── 4-way interleaved (checksum-only), native g++ compile ──

void fsst_decompress_interleaved4_cksum_native(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    uint32_t* checksums,
    uint32_t* decomp_lens)
{
    const unsigned char* __restrict__ L = (const unsigned char*)decoder->len;
    const unsigned long long* __restrict__ S = (const unsigned long long*)decoder->symbol;

    uint64_t i = start;

    for (; i + 4 <= end; i += 4) {
        const unsigned char* cp[4] = {comp_ptrs[i], comp_ptrs[i+1], comp_ptrs[i+2], comp_ptrs[i+3]};
        size_t cl[4] = {comp_lens[i], comp_lens[i+1], comp_lens[i+2], comp_lens[i+3]};
        size_t pi[4] = {0, 0, 0, 0};
        uint32_t po[4] = {0, 0, 0, 0};
        uint32_t h[4] = {2166136261u, 2166136261u, 2166136261u, 2166136261u};

        while (pi[0] < cl[0] && pi[1] < cl[1] && pi[2] < cl[2] && pi[3] < cl[3]) {
            for (int s = 0; s < 4; s++) {
                unsigned char code = cp[s][pi[s]++];
                if (__builtin_expect(code < 255, 1)) {
                    unsigned char slen = L[code];
                    unsigned long long sval = S[code];
                    for (unsigned char j = 0; j < slen; j++) {
                        h[s] ^= (unsigned char)(sval & 0xFF);
                        h[s] *= 16777619u;
                        sval >>= 8;
                    }
                    po[s] += slen;
                } else {
                    h[s] ^= cp[s][pi[s]++]; h[s] *= 16777619u;
                    po[s]++;
                }
            }
        }

        for (int s = 0; s < 4; s++) {
            while (pi[s] < cl[s]) {
                unsigned char code = cp[s][pi[s]++];
                if (__builtin_expect(code < 255, 1)) {
                    unsigned char slen = L[code];
                    unsigned long long sval = S[code];
                    for (unsigned char j = 0; j < slen; j++) {
                        h[s] ^= (unsigned char)(sval & 0xFF);
                        h[s] *= 16777619u;
                        sval >>= 8;
                    }
                    po[s] += slen;
                } else {
                    h[s] ^= cp[s][pi[s]++]; h[s] *= 16777619u;
                    po[s]++;
                }
            }
            checksums[i + s] = h[s];
            decomp_lens[i + s] = po[s];
        }
    }

    for (; i < end; i++) {
        uint32_t hash = 2166136261u;
        uint32_t posOut = 0;
        size_t posIn = 0;
        while (posIn < comp_lens[i]) {
            unsigned char code = comp_ptrs[i][posIn++];
            if (__builtin_expect(code < 255, 1)) {
                unsigned char slen = L[code];
                unsigned long long sval = S[code];
                for (unsigned char j = 0; j < slen; j++) {
                    hash ^= (unsigned char)(sval & 0xFF);
                    hash *= 16777619u;
                    sval >>= 8;
                }
                posOut += slen;
            } else {
                hash ^= comp_ptrs[i][posIn++]; hash *= 16777619u;
                posOut++;
            }
        }
        checksums[i] = hash;
        decomp_lens[i] = posOut;
    }
}

// ── Cache-resident output: 4-way interleaved, native g++ compile ──
//
// Decompresses each string to a small stack-local buffer (128B) that
// stays warm in L1/L2 cache. The buffer is reused per string, so no
// DRAM write traffic is generated. This measures pure decode + cache-write
// throughput for fair comparison with GPU shared-memory/register mode.

void fsst_decompress_cache_output_native(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    uint32_t* decomp_lens)
{
    const unsigned char* __restrict__ L = (const unsigned char*)decoder->len;
    const unsigned long long* __restrict__ S = (const unsigned long long*)decoder->symbol;

    uint64_t i = start;

    for (; i + 4 <= end; i += 4) {
        const unsigned char* cp[4] = {comp_ptrs[i], comp_ptrs[i+1], comp_ptrs[i+2], comp_ptrs[i+3]};
        size_t cl[4] = {comp_lens[i], comp_lens[i+1], comp_lens[i+2], comp_lens[i+3]};
        // Each lane has its own 128B buffer on stack (stays in L1 cache)
        unsigned char buf[4][128];
        size_t pi[4] = {0, 0, 0, 0};
        uint32_t po[4] = {0, 0, 0, 0};

        while (pi[0] < cl[0] && pi[1] < cl[1] && pi[2] < cl[2] && pi[3] < cl[3]) {
            for (int s = 0; s < 4; s++) {
                unsigned char code = cp[s][pi[s]++];
                if (__builtin_expect(code < 255, 1)) {
                    memcpy(buf[s] + (po[s] & 127), &S[code], 8);
                    po[s] += L[code];
                } else {
                    buf[s][po[s] & 127] = cp[s][pi[s]++];
                    po[s]++;
                }
            }
        }

        for (int s = 0; s < 4; s++) {
            while (pi[s] < cl[s]) {
                unsigned char code = cp[s][pi[s]++];
                if (__builtin_expect(code < 255, 1)) {
                    memcpy(buf[s] + (po[s] & 127), &S[code], 8);
                    po[s] += L[code];
                } else {
                    buf[s][po[s] & 127] = cp[s][pi[s]++];
                    po[s]++;
                }
            }
            decomp_lens[i + s] = po[s];
        }
    }

    for (; i < end; i++) {
        unsigned char buf[128];
        uint32_t posOut = 0;
        size_t posIn = 0;
        while (posIn < comp_lens[i]) {
            unsigned char code = comp_ptrs[i][posIn++];
            if (__builtin_expect(code < 255, 1)) {
                memcpy(buf + (posOut & 127), &S[code], 8);
                posOut += L[code];
            } else {
                buf[posOut & 127] = comp_ptrs[i][posIn++];
                posOut++;
            }
        }
        decomp_lens[i] = posOut;
    }
}

// ── AVX512 gather cache-resident output ──

void fsst_decompress_avx512_cache_output(
    const fsst_decoder_t* decoder,
    uint64_t start, uint64_t end,
    const size_t* comp_lens,
    const unsigned char* const* comp_ptrs,
    uint32_t* decomp_lens)
{
    const unsigned char* __restrict__ L = (const unsigned char*)decoder->len;
    const unsigned long long* __restrict__ S = (const unsigned long long*)decoder->symbol;
    const __m128i v_esc = _mm_set1_epi8((char)0xFF);
    const __m512i v_byte_mask = _mm512_set1_epi64(0xFF);

    for (uint64_t i = start; i < end; i++) {
        size_t cl = comp_lens[i];
        const unsigned char* cp = comp_ptrs[i];
        unsigned char buf[128];  // small buffer, stays in L1
        size_t posIn = 0;
        uint32_t posOut = 0;

        while (posIn + 8 <= cl) {
            __m128i v8 = _mm_loadl_epi64((const __m128i*)(cp + posIn));
            __m128i cmp = _mm_cmpeq_epi8(v8, v_esc);
            int esc = _mm_movemask_epi8(cmp) & 0xFF;

            if (__builtin_expect(esc == 0, 1)) {
                __m512i indices = _mm512_cvtepu8_epi64(v8);
                __m512i syms = _mm512_i64gather_epi64(indices, (const long long*)S, 8);
                __m512i lens = _mm512_i64gather_epi64(indices, (const long long*)L, 1);
                lens = _mm512_and_si512(lens, v_byte_mask);

                uint64_t s_arr[8], l_arr[8];
                _mm512_storeu_si512(s_arr, syms);
                _mm512_storeu_si512(l_arr, lens);

                memcpy(buf + (posOut & 127), &s_arr[0], 8); posOut += (uint32_t)l_arr[0];
                memcpy(buf + (posOut & 127), &s_arr[1], 8); posOut += (uint32_t)l_arr[1];
                memcpy(buf + (posOut & 127), &s_arr[2], 8); posOut += (uint32_t)l_arr[2];
                memcpy(buf + (posOut & 127), &s_arr[3], 8); posOut += (uint32_t)l_arr[3];
                memcpy(buf + (posOut & 127), &s_arr[4], 8); posOut += (uint32_t)l_arr[4];
                memcpy(buf + (posOut & 127), &s_arr[5], 8); posOut += (uint32_t)l_arr[5];
                memcpy(buf + (posOut & 127), &s_arr[6], 8); posOut += (uint32_t)l_arr[6];
                memcpy(buf + (posOut & 127), &s_arr[7], 8); posOut += (uint32_t)l_arr[7];

                posIn += 8;
            } else {
                int safe = __builtin_ctz(esc);
                for (int k = 0; k < safe; k++) {
                    unsigned char code = cp[posIn++];
                    memcpy(buf + (posOut & 127), &S[code], 8);
                    posOut += L[code];
                }
                posIn++;
                buf[posOut & 127] = cp[posIn++];
                posOut++;
            }
        }

        while (posIn < cl) {
            unsigned char code = cp[posIn++];
            if (__builtin_expect(code < 255, 1)) {
                memcpy(buf + (posOut & 127), &S[code], 8);
                posOut += L[code];
            } else {
                buf[posOut & 127] = cp[posIn++];
                posOut++;
            }
        }

        decomp_lens[i] = posOut;
    }
}
