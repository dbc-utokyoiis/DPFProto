#pragma once

#include <cstdint>

// Per-thread LZ4 decompression.
// Each thread independently decompresses one LZ4 chunk.
// No inter-thread cooperation or shared memory required.
// Compatible with CUDA C++11 (no nvCOMPdx dependency).
//
// Based on the LZ4 block format specification:
//   [token(1B)] [literal_len_ext...] [literals] [offset(2B)] [match_len_ext...]
//
// input:      compressed data (global memory)
// input_size: compressed size in bytes
// output:     decompressed output buffer (global memory)
// max_output: max output buffer size in bytes
//
// Returns: actual decompressed size. 0 on error (corrupt stream or overflow).
__device__ inline uint32_t lz4_decompress_per_thread(
    const uint8_t* input,
    uint32_t       input_size,
    uint8_t*       output,
    uint32_t       max_output)
{
    uint32_t ci = 0;   // compressed index
    uint32_t di = 0;   // decompressed index

    while (ci < input_size) {
        // --- token byte ---
        const uint8_t token = input[ci++];

        // --- literal length (high nibble) ---
        uint32_t lit_len = token >> 4;
        if (lit_len == 15) {
            uint8_t extra;
            do {
                if (ci >= input_size) return 0;
                extra = input[ci++];
                lit_len += extra;
            } while (extra == 0xFF);
        }

        // bounds check
        if (ci + lit_len > input_size) return 0;
        if (di + lit_len > max_output)  return 0;

        // copy literals
        for (uint32_t i = 0; i < lit_len; i++) {
            output[di + i] = input[ci + i];
        }
        ci += lit_len;
        di += lit_len;

        // last sequence has no match part
        if (ci >= input_size) break;

        // --- offset (2 bytes, little-endian) ---
        if (ci + 2 > input_size) return 0;
        const uint32_t offset = (uint32_t)input[ci]
                              | ((uint32_t)input[ci + 1] << 8);
        ci += 2;

        if (offset == 0 || offset > di) return 0;

        // --- match length (low nibble + 4) ---
        uint32_t match_len = (token & 0x0F) + 4;
        if ((token & 0x0F) == 15) {
            uint8_t extra;
            do {
                if (ci >= input_size) return 0;
                extra = input[ci++];
                match_len += extra;
            } while (extra == 0xFF);
        }

        // bounds check
        if (di + match_len > max_output) return 0;

        // copy match (byte-by-byte handles overlap correctly)
        for (uint32_t i = 0; i < match_len; i++) {
            output[di + i] = output[di - offset + i];
        }
        di += match_len;
    }

    return di;
}

// ============================================================
// Warp-cooperative LZ4 decompression.
//
// A full warp (32 threads) cooperatively decompresses one LZ4 block.
// Lane 0 parses the sequential token stream and broadcasts parsed
// values (lit_src, lit_len, offset, match_len) to all lanes via
// __shfl_sync. All 32 lanes then collaborate on literal copies and
// match copies, achieving coalesced memory access patterns.
//
// Match copy with overlap (offset < match_len) is handled correctly:
// the repeating pattern has period `offset`, so each output byte at
// position i is `output[di - offset + (i % offset)]`, which only
// reads from bytes written *before* this match began.
//
// input:      compressed data (global or shared memory)
// input_size: compressed size in bytes
// output:     decompressed output buffer (global or shared memory)
// max_output: max output buffer size in bytes
//
// Must be called by all 32 lanes of a warp (full warp participation).
// Returns: actual decompressed size (same value on all lanes).
//          0 on error (corrupt stream or overflow).
// ============================================================
__device__ inline uint32_t lz4_decompress_warp(
    const uint8_t* input,
    uint32_t       input_size,
    uint8_t*       output,
    uint32_t       max_output)
{
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t FULL_MASK = 0xFFFFFFFFu;

    uint32_t ci = 0;   // compressed index (lane 0 authoritative)
    uint32_t di = 0;   // decompressed index (all lanes track)

    while (1) {
        // ── Lane 0: parse one LZ4 sequence ──
        // Computes: lit_src, lit_len, offset, match_len, new_ci
        // Flags:    done (no more data), error
        uint32_t lit_src   = 0;  // position of literals in input[]
        uint32_t lit_len   = 0;
        uint32_t offset    = 0;
        uint32_t match_len = 0;
        uint32_t new_ci    = 0;
        uint32_t done      = 0;
        uint32_t error     = 0;

        if (lane == 0) {
            if (ci >= input_size) {
                done = 1;
            } else {
                const uint8_t token = input[ci++];

                // ── Literal length (high nibble) ──
                lit_len = token >> 4;
                if (lit_len == 15) {
                    uint8_t extra;
                    do {
                        if (ci >= input_size) { error = 1; break; }
                        extra = input[ci++];
                        lit_len += extra;
                    } while (extra == 0xFF);
                }

                if (!error) {
                    if (ci + lit_len > input_size) { error = 1; }
                    else if (di + lit_len > max_output) { error = 1; }
                }

                if (!error) {
                    lit_src = ci;       // literals start here
                    ci += lit_len;      // advance past literals

                    // Last sequence: no match part
                    if (ci >= input_size) {
                        done = 1;
                        new_ci = ci;
                    } else {
                        // ── Offset (2 bytes LE) ──
                        if (ci + 2 > input_size) { error = 1; }
                        else {
                            offset = (uint32_t)input[ci]
                                   | ((uint32_t)input[ci + 1] << 8);
                            ci += 2;
                            if (offset == 0 || offset > di + lit_len) {
                                error = 1;
                            }
                        }

                        // ── Match length (low nibble + 4) ──
                        if (!error) {
                            match_len = (token & 0x0F) + 4;
                            if ((token & 0x0F) == 15) {
                                uint8_t extra;
                                do {
                                    if (ci >= input_size) { error = 1; break; }
                                    extra = input[ci++];
                                    match_len += extra;
                                } while (extra == 0xFF);
                            }
                            if (!error && di + lit_len + match_len > max_output) {
                                error = 1;
                            }
                            if (!error) new_ci = ci;
                        }
                    }
                }
            }
        }

        // ── Broadcast flags ──
        error = __shfl_sync(FULL_MASK, error, 0);
        if (error) return 0;

        done     = __shfl_sync(FULL_MASK, done,      0);
        lit_src  = __shfl_sync(FULL_MASK, lit_src,    0);
        lit_len  = __shfl_sync(FULL_MASK, lit_len,    0);
        offset   = __shfl_sync(FULL_MASK, offset,     0);
        match_len = __shfl_sync(FULL_MASK, match_len, 0);
        ci       = __shfl_sync(FULL_MASK, new_ci,     0);

        // ── Cooperative literal copy (input → output) ──
        for (uint32_t i = lane; i < lit_len; i += 32) {
            output[di + i] = input[lit_src + i];
        }
        di += lit_len;

        if (done) break;

        // ── Cooperative match copy ──
        // Source: output[di - offset .. di - offset + offset - 1] repeating.
        // output[di + i] = output[di - offset + (i % offset)]
        // This is correct even when offset < match_len (overlap case),
        // because source bytes are from *before* this match started.
        const uint32_t match_src = di - offset;
        for (uint32_t i = lane; i < match_len; i += 32) {
            output[di + i] = output[match_src + (i % offset)];
        }
        di += match_len;
    }

    return di;
}

// ============================================================
// Warp-cooperative LZ4 decompression v2 (optimized).
//
// Improvements over v1:
//   1. Vectorized literal copy: 4 bytes/lane → 128 bytes/warp/iter
//      (4x bandwidth efficiency vs byte-by-byte)
//   2. Match copy specialization:
//      - offset == 1 (RLE): uint32_t broadcast fill
//      - offset >= match_len (no overlap): vectorized memcpy
//      - else: byte-level with i % offset (same as v1)
//
// Same API as v1.  Must be called by all 32 lanes of a warp.
// ============================================================

// Helper: warp-cooperative memcpy, 4 bytes per lane per iteration.
// Processes 128 bytes per warp step (vs 32 bytes for byte-by-byte).
// Uses memcpy for alignment-safe uint32_t load/store that the
// compiler optimizes into single LDG.32/STG.32 instructions.
__device__ inline void lz4_warp_memcpy(
    uint8_t*       __restrict__ dst,
    const uint8_t* __restrict__ src,
    uint32_t len,
    uint32_t lane)
{
    const uint32_t n4 = len >> 2;
    for (uint32_t c = lane; c < n4; c += 32) {
        uint32_t v;
        memcpy(&v, src + c * 4, 4);
        memcpy(dst + c * 4, &v, 4);
    }
    // Tail: 0-3 remaining bytes
    for (uint32_t i = (n4 << 2) + lane; i < len; i += 32)
        dst[i] = src[i];
}

__device__ inline uint32_t lz4_decompress_warp_v2(
    const uint8_t* __restrict__ input,
    uint32_t       input_size,
    uint8_t*       __restrict__ output,
    uint32_t       max_output)
{
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t FULL_MASK = 0xFFFFFFFFu;

    uint32_t ci = 0;
    uint32_t di = 0;

    while (1) {
        // ── Lane 0: parse one LZ4 sequence ──
        uint32_t lit_src   = 0;
        uint32_t lit_len   = 0;
        uint32_t offset    = 0;
        uint32_t match_len = 0;
        uint32_t new_ci    = 0;
        uint32_t done      = 0;
        uint32_t error     = 0;

        if (lane == 0) {
            if (ci >= input_size) {
                done = 1;
            } else {
                const uint8_t token = input[ci++];

                lit_len = token >> 4;
                if (lit_len == 15) {
                    uint8_t extra;
                    do {
                        if (ci >= input_size) { error = 1; break; }
                        extra = input[ci++];
                        lit_len += extra;
                    } while (extra == 0xFF);
                }

                if (!error) {
                    if (ci + lit_len > input_size) { error = 1; }
                    else if (di + lit_len > max_output) { error = 1; }
                }

                if (!error) {
                    lit_src = ci;
                    ci += lit_len;

                    if (ci >= input_size) {
                        done = 1;
                        new_ci = ci;
                    } else {
                        if (ci + 2 > input_size) { error = 1; }
                        else {
                            offset = (uint32_t)input[ci]
                                   | ((uint32_t)input[ci + 1] << 8);
                            ci += 2;
                            if (offset == 0 || offset > di + lit_len) {
                                error = 1;
                            }
                        }

                        if (!error) {
                            match_len = (token & 0x0F) + 4;
                            if ((token & 0x0F) == 15) {
                                uint8_t extra;
                                do {
                                    if (ci >= input_size) { error = 1; break; }
                                    extra = input[ci++];
                                    match_len += extra;
                                } while (extra == 0xFF);
                            }
                            if (!error && di + lit_len + match_len > max_output) {
                                error = 1;
                            }
                            if (!error) new_ci = ci;
                        }
                    }
                }
            }
        }

        // ── Broadcast ──
        error = __shfl_sync(FULL_MASK, error, 0);
        if (error) return 0;

        done      = __shfl_sync(FULL_MASK, done,      0);
        lit_src   = __shfl_sync(FULL_MASK, lit_src,    0);
        lit_len   = __shfl_sync(FULL_MASK, lit_len,    0);
        offset    = __shfl_sync(FULL_MASK, offset,     0);
        match_len = __shfl_sync(FULL_MASK, match_len,  0);
        ci        = __shfl_sync(FULL_MASK, new_ci,     0);

        // ── Vectorized literal copy (128 bytes/warp/iter) ──
        lz4_warp_memcpy(output + di, input + lit_src, lit_len, lane);
        di += lit_len;

        if (done) break;

        // ── Match copy with specialization ──
        const uint32_t match_src = di - offset;

        if (offset == 1) {
            // RLE: all bytes identical → broadcast fill with uint32_t
            const uint8_t b = output[match_src];
            const uint32_t b4 = (uint32_t)b * 0x01010101u;
            const uint32_t n4 = match_len >> 2;
            for (uint32_t c = lane; c < n4; c += 32)
                memcpy(output + di + c * 4, &b4, 4);
            for (uint32_t i = (n4 << 2) + lane; i < match_len; i += 32)
                output[di + i] = b;
        } else if (offset >= match_len) {
            // No overlap: source fully precedes destination → vectorized copy
            lz4_warp_memcpy(output + di, output + match_src, match_len, lane);
        } else {
            // Overlap: repeating pattern with period `offset`
            for (uint32_t i = lane; i < match_len; i += 32)
                output[di + i] = output[match_src + (i % offset)];
        }
        di += match_len;
    }

    return di;
}

// ============================================================
// Warp-cooperative LZ4 decompression v3 (batch-parsed).
//
// Improvements over v2:
//   - Lane 0 parses BATCH_SIZE sequences at once into shared
//     memory, then all 32 lanes process the batch.
//   - Replaces 7 × __shfl_sync per sequence with 1 × __syncwarp
//     per sequence (shared memory reads instead of shuffles).
//   - Includes all v2 optimizations (vectorized copy, RLE, etc.)
//
// Requires shared memory: LZ4_WARP_V3_SMEM bytes per warp.
// ============================================================

static constexpr int LZ4_V3_BATCH = 16;

#ifndef LZ4_HELPER_LZ4SEQ_DEFINED
#define LZ4_HELPER_LZ4SEQ_DEFINED
struct Lz4Seq {
    uint32_t lit_src;    // offset of literals in input[]
    uint32_t lit_len;
    uint32_t offset;     // back-reference offset (0 if last sequence)
    uint32_t match_len;
    uint32_t di;         // decompressed index at start of this sequence
};
#endif

// Shared memory per warp: batch of sequences + metadata
static constexpr size_t LZ4_WARP_V3_SMEM =
    sizeof(Lz4Seq) * LZ4_V3_BATCH + 4 * sizeof(uint32_t);

__device__ inline uint32_t lz4_decompress_warp_v3(
    const uint8_t* __restrict__ input,
    uint32_t       input_size,
    uint8_t*       __restrict__ output,
    uint32_t       max_output,
    uint8_t*       warp_smem)   // per-warp shared memory, >= LZ4_WARP_V3_SMEM bytes
{
    const uint32_t lane = threadIdx.x & 31;

    Lz4Seq*   seqs = reinterpret_cast<Lz4Seq*>(warp_smem);
    uint32_t* meta = reinterpret_cast<uint32_t*>(warp_smem + sizeof(Lz4Seq) * LZ4_V3_BATCH);
    // meta[0] = count  (sequences in this batch)
    // meta[1] = error
    // meta[2] = new ci
    // meta[3] = new di

    uint32_t ci = 0;
    uint32_t di = 0;

    while (1) {
        // ── Lane 0: batch-parse up to LZ4_V3_BATCH sequences ──
        if (lane == 0) {
            uint32_t count = 0;
            uint32_t error = 0;
            uint32_t local_di = di;

            while (count < LZ4_V3_BATCH && ci < input_size && !error) {
                const uint8_t token = input[ci++];

                uint32_t lit_len = token >> 4;
                if (lit_len == 15) {
                    uint8_t extra;
                    do {
                        if (ci >= input_size) { error = 1; break; }
                        extra = input[ci++];
                        lit_len += extra;
                    } while (extra == 0xFF);
                }
                if (error) break;
                if (ci + lit_len > input_size || local_di + lit_len > max_output) {
                    error = 1; break;
                }

                uint32_t lit_src = ci;
                ci += lit_len;

                // Last sequence: no match part
                if (ci >= input_size) {
                    seqs[count] = {lit_src, lit_len, 0, 0, local_di};
                    local_di += lit_len;
                    count++;
                    break;
                }

                // Offset
                if (ci + 2 > input_size) { error = 1; break; }
                uint32_t offset = (uint32_t)input[ci] | ((uint32_t)input[ci + 1] << 8);
                ci += 2;
                if (offset == 0 || offset > local_di + lit_len) { error = 1; break; }

                // Match length
                uint32_t match_len = (token & 0x0F) + 4;
                if ((token & 0x0F) == 15) {
                    uint8_t extra;
                    do {
                        if (ci >= input_size) { error = 1; break; }
                        extra = input[ci++];
                        match_len += extra;
                    } while (extra == 0xFF);
                }
                if (error) break;
                if (local_di + lit_len + match_len > max_output) { error = 1; break; }

                seqs[count] = {lit_src, lit_len, offset, match_len, local_di};
                local_di += lit_len + match_len;
                count++;
            }

            meta[0] = count;
            meta[1] = error;
            meta[2] = ci;
            meta[3] = local_di;
        }
        __syncwarp();

        // ── All lanes: read batch metadata ──
        const uint32_t count = meta[0];
        const uint32_t error = meta[1];
        if (error) return 0;
        if (count == 0) break;
        ci = meta[2];

        // ── All lanes: process each sequence in the batch ──
        for (uint32_t k = 0; k < count; k++) {
            const Lz4Seq s = seqs[k];

            // Vectorized literal copy
            lz4_warp_memcpy(output + s.di, input + s.lit_src, s.lit_len, lane);

            if (s.offset == 0) break;  // last sequence (no match)

            // Match copy with specialization
            const uint32_t match_di  = s.di + s.lit_len;
            const uint32_t match_src = match_di - s.offset;

            if (s.offset == 1) {
                const uint8_t b = output[match_src];
                const uint32_t b4 = (uint32_t)b * 0x01010101u;
                const uint32_t n4 = s.match_len >> 2;
                for (uint32_t c = lane; c < n4; c += 32)
                    memcpy(output + match_di + c * 4, &b4, 4);
                for (uint32_t i = (n4 << 2) + lane; i < s.match_len; i += 32)
                    output[match_di + i] = b;
            } else if (s.offset >= s.match_len) {
                lz4_warp_memcpy(output + match_di, output + match_src, s.match_len, lane);
            } else {
                for (uint32_t i = lane; i < s.match_len; i += 32)
                    output[match_di + i] = output[match_src + (i % s.offset)];
            }
            // Ensure writes visible before next sequence reads output[]
            __syncwarp();
        }

        di = meta[3];
    }

    return di;
}

// ============================================================
// Warp-cooperative LZ4 decompression v4 (2-pass parallel).
//
// Pass 1: Lane 0 scans all sequences, recording (ci, di) into
//         shared memory.  Pure parsing — no output copies.
//         Sequential reads of compressed data → cache-friendly.
//
// Pass 2: All 32 threads process sequences in parallel.
//         Each "row" of 32 sequences is handled by 32 threads:
//           Phase A: Each thread independently copies its literals
//                    (32 literals in parallel, no dependencies)
//           Phase B: Cooperative match copies (all 32 threads
//                    handle each match one at a time for
//                    correctness — match may read prior output)
//
// Requires shared memory: LZ4_WARP_V4_SMEM bytes per warp.
// ============================================================

static constexpr int LZ4_V4_CHUNK = 512;

struct Lz4SeqV4 {
    uint32_t lit_src;    // offset of literals in input[]
    uint32_t lit_len;
    uint32_t offset;     // back-reference offset (0 if last sequence)
    uint32_t match_len;
    uint32_t di;         // decompressed index at start of this sequence
};

static constexpr size_t LZ4_WARP_V4_SMEM =
    sizeof(Lz4SeqV4) * LZ4_V4_CHUNK + 4 * sizeof(uint32_t);

__device__ inline uint32_t lz4_decompress_warp_v4(
    const uint8_t* __restrict__ input,
    uint32_t       input_size,
    uint8_t*       __restrict__ output,
    uint32_t       max_output,
    uint8_t*       warp_smem)   // per-warp shared memory, >= LZ4_WARP_V4_SMEM bytes
{
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t FULL_MASK = 0xFFFFFFFFu;

    Lz4SeqV4* seqs = reinterpret_cast<Lz4SeqV4*>(warp_smem);
    uint32_t* meta = reinterpret_cast<uint32_t*>(
        warp_smem + sizeof(Lz4SeqV4) * LZ4_V4_CHUNK);
    // meta[0] = count, meta[1] = error, meta[2] = new_ci, meta[3] = new_di

    uint32_t ci = 0;
    uint32_t di = 0;

    while (1) {
        // ═══════════════════════════════════════════════════════
        // Pass 1: Lane 0 lightweight scan (parse only, no copy)
        // ═══════════════════════════════════════════════════════
        if (lane == 0) {
            uint32_t count = 0;
            uint32_t error = 0;
            uint32_t lci = ci, ldi = di;

            while (count < LZ4_V4_CHUNK && lci < input_size && !error) {
                uint8_t token = input[lci++];

                // Literal length
                uint32_t lit_len = token >> 4;
                if (lit_len == 15) {
                    uint8_t e;
                    do {
                        if (lci >= input_size) { error = 1; break; }
                        e = input[lci++]; lit_len += e;
                    } while (e == 0xFF);
                }
                if (error) break;
                if (lci + lit_len > input_size || ldi + lit_len > max_output) {
                    error = 1; break;
                }

                uint32_t lit_src = lci;
                lci += lit_len;

                // Last sequence: no match
                if (lci >= input_size) {
                    seqs[count] = {lit_src, lit_len, 0, 0, ldi};
                    ldi += lit_len;
                    count++;
                    break;
                }

                // Offset
                if (lci + 2 > input_size) { error = 1; break; }
                uint32_t offset = (uint32_t)input[lci] | ((uint32_t)input[lci + 1] << 8);
                lci += 2;
                if (offset == 0 || offset > ldi + lit_len) { error = 1; break; }

                // Match length
                uint32_t match_len = (token & 0x0F) + 4;
                if ((token & 0x0F) == 15) {
                    uint8_t e;
                    do {
                        if (lci >= input_size) { error = 1; break; }
                        e = input[lci++]; match_len += e;
                    } while (e == 0xFF);
                }
                if (error) break;
                if (ldi + lit_len + match_len > max_output) { error = 1; break; }

                seqs[count] = {lit_src, lit_len, offset, match_len, ldi};
                ldi += lit_len + match_len;
                count++;
            }

            meta[0] = count;
            meta[1] = error;
            meta[2] = lci;
            meta[3] = ldi;
        }
        __syncwarp();

        const uint32_t count = meta[0];
        if (meta[1]) return 0;
        if (count == 0) break;
        ci = meta[2];

        // ═══════════════════════════════════════════════════════
        // Pass 2: Cooperative decompress (all 32 threads per seq)
        //   No re-parsing needed — metadata is in shared memory.
        // ═══════════════════════════════════════════════════════
        for (uint32_t k = 0; k < count; k++) {
            const Lz4SeqV4 s = seqs[k];

            // Vectorized literal copy
            lz4_warp_memcpy(output + s.di, input + s.lit_src, s.lit_len, lane);

            if (s.offset == 0) break;  // last sequence (no match)

            // Match copy with specialization
            const uint32_t match_di  = s.di + s.lit_len;
            const uint32_t match_src = match_di - s.offset;

            if (s.offset == 1) {
                const uint8_t b = output[match_src];
                const uint32_t b4 = (uint32_t)b * 0x01010101u;
                const uint32_t n4 = s.match_len >> 2;
                for (uint32_t c = lane; c < n4; c += 32)
                    memcpy(output + match_di + c * 4, &b4, 4);
                for (uint32_t i = (n4 << 2) + lane; i < s.match_len; i += 32)
                    output[match_di + i] = b;
            } else if (s.offset >= s.match_len) {
                lz4_warp_memcpy(output + match_di, output + match_src, s.match_len, lane);
            } else {
                for (uint32_t i = lane; i < s.match_len; i += 32)
                    output[match_di + i] = output[match_src + (i % s.offset)];
            }
            __syncwarp();
        }

        di = meta[3];
    }

    return di;
}  // end lz4_decompress_warp_v4

// ============================================================
// Warp-cooperative LZ4 decompression v6 (adaptive hybrid).
//
// Pre-computed metadata + adaptive execution per row:
//
//   Long sequences (max output > THRESHOLD):
//     → Cooperative: all 32 threads per sequence, vectorized copy
//
//   Short sequences (max output <= THRESHOLD):
//     → Parallel: each thread handles its own sequence
//     → Literal copies: always parallel (no dependencies)
//     → Match copies: safety check via warp reduction
//       - Safe (all match sources precede row's match region):
//         parallel match copies
//       - Unsafe (cross-thread match dependency detected):
//         cooperative match copies (one at a time)
//
// seqs:   pre-computed Lz4Seq array (in global memory)
// n_seqs: number of sequences
// ============================================================

static constexpr uint32_t LZ4_V6_COOPERATIVE_THRESHOLD = 256;
static constexpr uint32_t LZ4_V6_SMEM_PER_WARP = 32 * 256;  // 8192 bytes

__device__ inline uint32_t lz4_decompress_warp_v6(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    const Lz4Seq*  seqs,
    uint32_t       n_seqs,
    uint8_t*       warp_smem)   // per-warp staging buffer, >= LZ4_V6_SMEM_PER_WARP bytes
{
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t FULL_MASK = 0xFFFFFFFFu;

    for (uint32_t row = 0; row < n_seqs; row += 32) {
        const uint32_t idx = row + lane;
        const bool valid = (idx < n_seqs);

        Lz4Seq s = {};
        if (valid) s = seqs[idx];

        const uint32_t seq_out = valid ? (s.lit_len + s.match_len) : 0;

        // ── Warp-level max to decide execution mode ──
        uint32_t max_out = seq_out;
        #pragma unroll
        for (int d = 16; d >= 1; d >>= 1)
            max_out = max(max_out, __shfl_xor_sync(FULL_MASK, max_out, d));

        if (max_out > LZ4_V6_COOPERATIVE_THRESHOLD) {
            // ═══ COOPERATIVE PATH: all 32 threads per sequence ═══
            // Already coalesced via lz4_warp_memcpy — no smem staging needed.
            for (uint32_t k = 0; k < 32 && (row + k) < n_seqs; k++) {
                uint32_t sk_lit_src   = __shfl_sync(FULL_MASK, s.lit_src,   k);
                uint32_t sk_lit_len   = __shfl_sync(FULL_MASK, s.lit_len,   k);
                uint32_t sk_offset    = __shfl_sync(FULL_MASK, s.offset,    k);
                uint32_t sk_match_len = __shfl_sync(FULL_MASK, s.match_len, k);
                uint32_t sk_di        = __shfl_sync(FULL_MASK, s.di,        k);

                lz4_warp_memcpy(output + sk_di, input + sk_lit_src, sk_lit_len, lane);

                if (sk_offset == 0) break;  // last sequence

                uint32_t m_di  = sk_di + sk_lit_len;
                uint32_t m_src = m_di - sk_offset;

                if (sk_offset == 1) {
                    uint8_t b = output[m_src];
                    uint32_t b4 = (uint32_t)b * 0x01010101u;
                    uint32_t n4 = sk_match_len >> 2;
                    for (uint32_t c = lane; c < n4; c += 32)
                        memcpy(output + m_di + c * 4, &b4, 4);
                    for (uint32_t i = (n4 << 2) + lane; i < sk_match_len; i += 32)
                        output[m_di + i] = b;
                } else if (sk_offset >= sk_match_len) {
                    lz4_warp_memcpy(output + m_di, output + m_src, sk_match_len, lane);
                } else {
                    for (uint32_t i = lane; i < sk_match_len; i += 32)
                        output[m_di + i] = output[m_src + (i % sk_offset)];
                }
                __syncwarp();
            }
        } else {
            // ═══ PARALLEL PATH with shared memory staging ═══
            // Each thread decompresses into smem, then all 32 threads
            // cooperatively flush smem → global with coalesced writes.

            // Row bounds for smem addressing
            uint32_t first_di = __shfl_sync(FULL_MASK, s.di, 0);
            uint32_t my_end = valid ? (s.di + s.lit_len + s.match_len) : 0;
            uint32_t row_end = my_end;
            #pragma unroll
            for (int d = 16; d >= 1; d >>= 1)
                row_end = max(row_end, __shfl_xor_sync(FULL_MASK, row_end, d));
            uint32_t row_len = row_end - first_di;

            // Phase A: Literal copies → smem (parallel, per-thread, vectorized)
            if (valid && s.lit_len > 0) {
                const uint8_t* src = input + s.lit_src;
                uint8_t*       dst = warp_smem + (s.di - first_di);
                const uint32_t n4 = s.lit_len >> 2;
                for (uint32_t c = 0; c < n4; c++) {
                    uint32_t v;
                    memcpy(&v, src + c * 4, 4);
                    memcpy(dst + c * 4, &v, 4);
                }
                for (uint32_t i = (n4 << 2); i < s.lit_len; i++)
                    dst[i] = src[i];
            }
            __syncwarp();  // All literals in smem

            // Phase B: Match copies with safety check
            uint32_t my_match_di  = (valid && s.offset != 0)
                                  ? (s.di + s.lit_len) : 0xFFFFFFFFu;
            uint32_t min_match_di = my_match_di;
            #pragma unroll
            for (int d = 16; d >= 1; d >>= 1)
                min_match_di = min(min_match_di,
                    __shfl_xor_sync(FULL_MASK, min_match_di, d));

            uint32_t my_match_src = (valid && s.offset != 0)
                                  ? (s.di + s.lit_len - s.offset) : 0u;
            bool my_safe = (!valid || s.offset == 0)
                         ? true : (my_match_src < min_match_di);
            uint32_t all_safe = __ballot_sync(FULL_MASK, my_safe);

            if (all_safe == FULL_MASK) {
                // Parallel match copies → smem
                if (valid && s.offset != 0) {
                    uint32_t m_di      = s.di + s.lit_len;
                    uint32_t m_src_pos = m_di - s.offset;
                    uint32_t sm_dst    = m_di - first_di;

                    if (s.offset == 1) {
                        // RLE: read source byte from smem or global
                        uint8_t b = (m_src_pos >= first_di)
                            ? warp_smem[m_src_pos - first_di]
                            : output[m_src_pos];
                        uint32_t b4 = (uint32_t)b * 0x01010101u;
                        uint32_t n4 = s.match_len >> 2;
                        for (uint32_t c = 0; c < n4; c++)
                            memcpy(warp_smem + sm_dst + c * 4, &b4, 4);
                        for (uint32_t i = (n4 << 2); i < s.match_len; i++)
                            warp_smem[sm_dst + i] = b;
                    } else if (s.offset >= s.match_len && m_src_pos >= first_di) {
                        // No overlap, source in smem → smem-to-smem vectorized
                        uint32_t sm_src = m_src_pos - first_di;
                        uint32_t n4 = s.match_len >> 2;
                        for (uint32_t c = 0; c < n4; c++) {
                            uint32_t v;
                            memcpy(&v, warp_smem + sm_src + c * 4, 4);
                            memcpy(warp_smem + sm_dst + c * 4, &v, 4);
                        }
                        for (uint32_t i = (n4 << 2); i < s.match_len; i++)
                            warp_smem[sm_dst + i] = warp_smem[sm_src + i];
                    } else if (s.offset >= s.match_len
                               && m_src_pos + s.match_len <= first_di) {
                        // No overlap, source entirely in global → vectorized
                        uint32_t n4 = s.match_len >> 2;
                        for (uint32_t c = 0; c < n4; c++) {
                            uint32_t v;
                            memcpy(&v, output + m_src_pos + c * 4, 4);
                            memcpy(warp_smem + sm_dst + c * 4, &v, 4);
                        }
                        for (uint32_t i = (n4 << 2); i < s.match_len; i++)
                            warp_smem[sm_dst + i] = output[m_src_pos + i];
                    } else {
                        // General: overlap or boundary-straddling
                        for (uint32_t i = 0; i < s.match_len; i++) {
                            uint32_t src = m_src_pos + (i % s.offset);
                            warp_smem[sm_dst + i] = (src >= first_di)
                                ? warp_smem[src - first_di]
                                : output[src];
                        }
                    }
                }
                __syncwarp();

                // Phase C: coalesced flush smem → global (128 bytes/warp/iter)
                lz4_warp_memcpy(output + first_di, warp_smem, row_len, lane);
            } else {
                // Dependency detected → flush literals, then cooperative matches
                lz4_warp_memcpy(output + first_di, warp_smem, row_len, lane);
                __syncwarp();

                for (uint32_t k = 0; k < 32 && (row + k) < n_seqs; k++) {
                    uint32_t m_di  = __shfl_sync(FULL_MASK, s.di + s.lit_len, k);
                    uint32_t m_off = __shfl_sync(FULL_MASK, s.offset, k);
                    uint32_t m_len = __shfl_sync(FULL_MASK, s.match_len, k);
                    if (m_off == 0) continue;
                    uint32_t m_src = m_di - m_off;

                    if (m_off == 1) {
                        uint8_t b = output[m_src];
                        uint32_t b4 = (uint32_t)b * 0x01010101u;
                        uint32_t n4 = m_len >> 2;
                        for (uint32_t c = lane; c < n4; c += 32)
                            memcpy(output + m_di + c * 4, &b4, 4);
                        for (uint32_t i = (n4 << 2) + lane; i < m_len; i += 32)
                            output[m_di + i] = b;
                    } else if (m_off >= m_len) {
                        lz4_warp_memcpy(output + m_di, output + m_src, m_len, lane);
                    } else {
                        for (uint32_t i = lane; i < m_len; i += 32)
                            output[m_di + i] = output[m_src + (i % m_off)];
                    }
                    __syncwarp();
                }
            }
            __syncwarp();
        }
    }

    if (n_seqs > 0) {
        const Lz4Seq last = seqs[n_seqs - 1];
        return last.di + last.lit_len + last.match_len;
    }
    return 0;
}

// ============================================================
// Warp-cooperative LZ4 decompression v7 (GPU-parsed + adaptive hybrid).
//
// Same execution strategy as v6 (adaptive cooperative/parallel + smem staging)
// but parses LZ4 sequences on-GPU (lane 0) instead of requiring pre-computed
// metadata.  Does not require Lz4Seq metadata — works on standard LZ4 pages.
// Comparing v6 vs v7 isolates the metadata pre-computation overhead.
//
// Shared memory layout per warp:
//   [Lz4Seq × CHUNK][uint32_t × 4 meta][uint8_t staging[8192]]
// ============================================================

static constexpr int LZ4_V7_CHUNK = 256;
static constexpr size_t LZ4_V7_SMEM =
    sizeof(Lz4Seq) * LZ4_V7_CHUNK + 4 * sizeof(uint32_t) + LZ4_V6_SMEM_PER_WARP;

__device__ inline uint32_t lz4_decompress_warp_v7(
    const uint8_t* __restrict__ input,
    uint32_t       input_size,
    uint8_t*       __restrict__ output,
    uint32_t       max_output,
    uint8_t*       warp_smem)   // per-warp, >= LZ4_V7_SMEM bytes
{
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t FULL_MASK = 0xFFFFFFFFu;

    Lz4Seq*   seqs    = reinterpret_cast<Lz4Seq*>(warp_smem);
    uint32_t* meta    = reinterpret_cast<uint32_t*>(
        warp_smem + sizeof(Lz4Seq) * LZ4_V7_CHUNK);
    uint8_t*  staging = warp_smem + sizeof(Lz4Seq) * LZ4_V7_CHUNK
                        + 4 * sizeof(uint32_t);

    uint32_t ci = 0;
    uint32_t di = 0;

    while (1) {
        // ═══════════════════════════════════════════════════════
        // Pass 1: Lane 0 parses up to CHUNK sequences into smem
        // ═══════════════════════════════════════════════════════
        if (lane == 0) {
            uint32_t count = 0;
            uint32_t error = 0;
            uint32_t lci = ci, ldi = di;

            while (count < LZ4_V7_CHUNK && lci < input_size && !error) {
                uint8_t token = input[lci++];

                uint32_t lit_len = token >> 4;
                if (lit_len == 15) {
                    uint8_t e;
                    do {
                        if (lci >= input_size) { error = 1; break; }
                        e = input[lci++]; lit_len += e;
                    } while (e == 0xFF);
                }
                if (error) break;
                if (lci + lit_len > input_size || ldi + lit_len > max_output) {
                    error = 1; break;
                }

                uint32_t lit_src = lci;
                lci += lit_len;

                if (lci >= input_size) {
                    seqs[count] = {lit_src, lit_len, 0, 0, ldi};
                    ldi += lit_len;
                    count++;
                    break;
                }

                if (lci + 2 > input_size) { error = 1; break; }
                uint32_t offset = (uint32_t)input[lci] | ((uint32_t)input[lci + 1] << 8);
                lci += 2;
                if (offset == 0 || offset > ldi + lit_len) { error = 1; break; }

                uint32_t match_len = (token & 0x0F) + 4;
                if ((token & 0x0F) == 15) {
                    uint8_t e;
                    do {
                        if (lci >= input_size) { error = 1; break; }
                        e = input[lci++]; match_len += e;
                    } while (e == 0xFF);
                }
                if (error) break;
                if (ldi + lit_len + match_len > max_output) { error = 1; break; }

                seqs[count] = {lit_src, lit_len, offset, match_len, ldi};
                ldi += lit_len + match_len;
                count++;
            }

            meta[0] = count;
            meta[1] = error;
            meta[2] = lci;
            meta[3] = ldi;
        }
        __syncwarp();

        const uint32_t count = meta[0];
        if (meta[1]) return 0;
        if (count == 0) break;
        ci = meta[2];

        // ═══════════════════════════════════════════════════════
        // Pass 2: v6-style adaptive execution over parsed chunk
        // ═══════════════════════════════════════════════════════
        for (uint32_t row = 0; row < count; row += 32) {
            const uint32_t k = row + lane;
            const bool valid = (k < count);

            Lz4Seq s = {};
            if (valid) s = seqs[k];

            const uint32_t seq_out = valid ? (s.lit_len + s.match_len) : 0;

            uint32_t max_out = seq_out;
            #pragma unroll
            for (int d = 16; d >= 1; d >>= 1)
                max_out = max(max_out, __shfl_xor_sync(FULL_MASK, max_out, d));

            if (max_out > LZ4_V6_COOPERATIVE_THRESHOLD) {
                // ═══ COOPERATIVE PATH ═══
                for (uint32_t j = 0; j < 32 && (row + j) < count; j++) {
                    uint32_t sk_lit_src   = __shfl_sync(FULL_MASK, s.lit_src,   j);
                    uint32_t sk_lit_len   = __shfl_sync(FULL_MASK, s.lit_len,   j);
                    uint32_t sk_offset    = __shfl_sync(FULL_MASK, s.offset,    j);
                    uint32_t sk_match_len = __shfl_sync(FULL_MASK, s.match_len, j);
                    uint32_t sk_di        = __shfl_sync(FULL_MASK, s.di,        j);

                    lz4_warp_memcpy(output + sk_di, input + sk_lit_src, sk_lit_len, lane);

                    if (sk_offset == 0) break;

                    uint32_t m_di  = sk_di + sk_lit_len;
                    uint32_t m_src = m_di - sk_offset;

                    if (sk_offset == 1) {
                        uint8_t b = output[m_src];
                        uint32_t b4 = (uint32_t)b * 0x01010101u;
                        uint32_t n4 = sk_match_len >> 2;
                        for (uint32_t c = lane; c < n4; c += 32)
                            memcpy(output + m_di + c * 4, &b4, 4);
                        for (uint32_t i = (n4 << 2) + lane; i < sk_match_len; i += 32)
                            output[m_di + i] = b;
                    } else if (sk_offset >= sk_match_len) {
                        lz4_warp_memcpy(output + m_di, output + m_src, sk_match_len, lane);
                    } else {
                        for (uint32_t i = lane; i < sk_match_len; i += 32)
                            output[m_di + i] = output[m_src + (i % sk_offset)];
                    }
                    __syncwarp();
                }
            } else {
                // ═══ PARALLEL PATH with smem staging ═══
                uint32_t first_di = __shfl_sync(FULL_MASK, s.di, 0);
                uint32_t my_end = valid ? (s.di + s.lit_len + s.match_len) : 0;
                uint32_t row_end = my_end;
                #pragma unroll
                for (int d = 16; d >= 1; d >>= 1)
                    row_end = max(row_end, __shfl_xor_sync(FULL_MASK, row_end, d));
                uint32_t row_len = row_end - first_di;

                // Phase A: literals → staging smem
                if (valid && s.lit_len > 0) {
                    const uint8_t* src = input + s.lit_src;
                    uint8_t*       dst = staging + (s.di - first_di);
                    const uint32_t n4 = s.lit_len >> 2;
                    for (uint32_t c = 0; c < n4; c++) {
                        uint32_t v;
                        memcpy(&v, src + c * 4, 4);
                        memcpy(dst + c * 4, &v, 4);
                    }
                    for (uint32_t i = (n4 << 2); i < s.lit_len; i++)
                        dst[i] = src[i];
                }
                __syncwarp();

                // Phase B: match copies with safety check
                uint32_t my_match_di = (valid && s.offset != 0)
                                     ? (s.di + s.lit_len) : 0xFFFFFFFFu;
                uint32_t min_match_di = my_match_di;
                #pragma unroll
                for (int d = 16; d >= 1; d >>= 1)
                    min_match_di = min(min_match_di,
                        __shfl_xor_sync(FULL_MASK, min_match_di, d));

                uint32_t my_match_src = (valid && s.offset != 0)
                                      ? (s.di + s.lit_len - s.offset) : 0u;
                bool my_safe = (!valid || s.offset == 0)
                             ? true : (my_match_src < min_match_di);
                uint32_t all_safe = __ballot_sync(FULL_MASK, my_safe);

                if (all_safe == FULL_MASK) {
                    if (valid && s.offset != 0) {
                        uint32_t m_di      = s.di + s.lit_len;
                        uint32_t m_src_pos = m_di - s.offset;
                        uint32_t sm_dst    = m_di - first_di;

                        if (s.offset == 1) {
                            uint8_t b = (m_src_pos >= first_di)
                                ? staging[m_src_pos - first_di]
                                : output[m_src_pos];
                            uint32_t b4 = (uint32_t)b * 0x01010101u;
                            uint32_t n4 = s.match_len >> 2;
                            for (uint32_t c = 0; c < n4; c++)
                                memcpy(staging + sm_dst + c * 4, &b4, 4);
                            for (uint32_t i = (n4 << 2); i < s.match_len; i++)
                                staging[sm_dst + i] = b;
                        } else if (s.offset >= s.match_len && m_src_pos >= first_di) {
                            uint32_t sm_src = m_src_pos - first_di;
                            uint32_t n4 = s.match_len >> 2;
                            for (uint32_t c = 0; c < n4; c++) {
                                uint32_t v;
                                memcpy(&v, staging + sm_src + c * 4, 4);
                                memcpy(staging + sm_dst + c * 4, &v, 4);
                            }
                            for (uint32_t i = (n4 << 2); i < s.match_len; i++)
                                staging[sm_dst + i] = staging[sm_src + i];
                        } else if (s.offset >= s.match_len
                                   && m_src_pos + s.match_len <= first_di) {
                            uint32_t n4 = s.match_len >> 2;
                            for (uint32_t c = 0; c < n4; c++) {
                                uint32_t v;
                                memcpy(&v, output + m_src_pos + c * 4, 4);
                                memcpy(staging + sm_dst + c * 4, &v, 4);
                            }
                            for (uint32_t i = (n4 << 2); i < s.match_len; i++)
                                staging[sm_dst + i] = output[m_src_pos + i];
                        } else {
                            for (uint32_t i = 0; i < s.match_len; i++) {
                                uint32_t src = m_src_pos + (i % s.offset);
                                staging[sm_dst + i] = (src >= first_di)
                                    ? staging[src - first_di]
                                    : output[src];
                            }
                        }
                    }
                    __syncwarp();
                    lz4_warp_memcpy(output + first_di, staging, row_len, lane);
                } else {
                    lz4_warp_memcpy(output + first_di, staging, row_len, lane);
                    __syncwarp();
                    for (uint32_t j = 0; j < 32 && (row + j) < count; j++) {
                        uint32_t m_di  = __shfl_sync(FULL_MASK, s.di + s.lit_len, j);
                        uint32_t m_off = __shfl_sync(FULL_MASK, s.offset, j);
                        uint32_t m_len = __shfl_sync(FULL_MASK, s.match_len, j);
                        if (m_off == 0) continue;
                        uint32_t m_src = m_di - m_off;
                        if (m_off == 1) {
                            uint8_t b = output[m_src];
                            uint32_t b4 = (uint32_t)b * 0x01010101u;
                            uint32_t n4 = m_len >> 2;
                            for (uint32_t c = lane; c < n4; c += 32)
                                memcpy(output + m_di + c * 4, &b4, 4);
                            for (uint32_t i = (n4 << 2) + lane; i < m_len; i += 32)
                                output[m_di + i] = b;
                        } else if (m_off >= m_len) {
                            lz4_warp_memcpy(output + m_di, output + m_src, m_len, lane);
                        } else {
                            for (uint32_t i = lane; i < m_len; i += 32)
                                output[m_di + i] = output[m_src + (i % m_off)];
                        }
                        __syncwarp();
                    }
                }
                __syncwarp();
            }
        }

        di = meta[3];
    }

    return di;
}

// ============================================================
// Warp-cooperative LZ4 decompression v5 (pre-computed metadata).
//
// All sequence metadata is pre-computed (e.g., by CPU at
// compression time) and passed in global memory.  GPU does
// zero parsing — only cooperative vectorized copies.
//
// This measures the performance ceiling for warp-cooperative
// LZ4 decompression when parsing overhead is eliminated.
//
// seqs:   pre-computed Lz4Seq array (in global memory)
// n_seqs: number of sequences
// ============================================================
__device__ inline uint32_t lz4_decompress_warp_v5(
    const uint8_t* __restrict__ input,
    uint8_t*       __restrict__ output,
    const Lz4Seq*  seqs,
    uint32_t       n_seqs)
{
    const uint32_t lane = threadIdx.x & 31;
    uint32_t di = 0;

    for (uint32_t k = 0; k < n_seqs; k++) {
        const Lz4Seq s = seqs[k];

        // Vectorized literal copy
        lz4_warp_memcpy(output + s.di, input + s.lit_src, s.lit_len, lane);

        if (s.offset == 0) {
            di = s.di + s.lit_len;
            break;  // last sequence
        }

        const uint32_t match_di  = s.di + s.lit_len;
        const uint32_t match_src = match_di - s.offset;

        if (s.offset == 1) {
            const uint8_t b = output[match_src];
            const uint32_t b4 = (uint32_t)b * 0x01010101u;
            const uint32_t n4 = s.match_len >> 2;
            for (uint32_t c = lane; c < n4; c += 32)
                memcpy(output + match_di + c * 4, &b4, 4);
            for (uint32_t i = (n4 << 2) + lane; i < s.match_len; i += 32)
                output[match_di + i] = b;
        } else if (s.offset >= s.match_len) {
            lz4_warp_memcpy(output + match_di, output + match_src, s.match_len, lane);
        } else {
            for (uint32_t i = lane; i < s.match_len; i += 32)
                output[match_di + i] = output[match_src + (i % s.offset)];
        }
        __syncwarp();
        di = match_di + s.match_len;
    }

    return di;
}
