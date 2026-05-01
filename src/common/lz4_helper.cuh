#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

// Lz4Seq is defined in lz4_decomp.cuh; forward-include or re-use the same struct.
// We keep a local definition here so lz4_helper.cuh is self-contained.
#ifndef LZ4_HELPER_LZ4SEQ_DEFINED
#define LZ4_HELPER_LZ4SEQ_DEFINED
struct Lz4Seq {
    uint32_t lit_src;    // offset of literals in compressed input
    uint32_t lit_len;
    uint32_t offset;     // back-reference offset (0 if last sequence)
    uint32_t match_len;
    uint32_t di;         // decompressed index at start of this sequence
};
#endif

// ============================================================
// parse_lz4_sequences — Extract per-token metadata from an LZ4 block.
//
// Parses the LZ4 block format sequentially and fills `out` with
// one Lz4Seq per token (literal run + match pair).
//
// comp:    pointer to the LZ4 compressed block
// comp_sz: size of the compressed block in bytes
// out:     output vector (cleared and populated)
// ============================================================
inline void parse_lz4_sequences(
    const uint8_t* comp, int comp_sz,
    std::vector<Lz4Seq>& out)
{
    out.clear();
    uint32_t ci = 0, di = 0;
    while (ci < (uint32_t)comp_sz) {
        uint8_t token = comp[ci++];
        uint32_t lit_len = token >> 4;
        if (lit_len == 15) {
            uint8_t e;
            do { e = comp[ci++]; lit_len += e; } while (e == 0xFF);
        }
        uint32_t lit_src = ci;
        ci += lit_len;

        if (ci >= (uint32_t)comp_sz) {
            out.push_back({lit_src, lit_len, 0, 0, di});
            break;
        }

        uint32_t offset = (uint32_t)comp[ci] | ((uint32_t)comp[ci + 1] << 8);
        ci += 2;

        uint32_t match_len = (token & 0x0F) + 4;
        if ((token & 0x0F) == 15) {
            uint8_t e;
            do { e = comp[ci++]; match_len += e; } while (e == 0xFF);
        }

        out.push_back({lit_src, lit_len, offset, match_len, di});
        di += lit_len + match_len;
    }
}

// ============================================================
// PFOR encoder for LZ4PARSEQ metadata
//
// Encodes a uint32_t array into Simple BinPack PFOR format:
//   128-element blocks, 4 miniblocks of 32, single max bitwidth.
// Format per field: [n_blocks][block_off[0..n_blocks]][packed block data...]
// Block data: [min_val][bw_packed][bit-packed words...]
// ============================================================
inline void pfor_encode_field(const uint32_t* values, uint32_t n,
                              std::vector<uint32_t>& out) {
    constexpr uint32_t BLOCK_SZ = 128;
    uint32_t n_padded = ((n + BLOCK_SZ - 1) / BLOCK_SZ) * BLOCK_SZ;
    uint32_t n_blocks = n_padded / BLOCK_SZ;

    // Pad input
    std::vector<uint32_t> vals(n_padded, 0);
    std::memcpy(vals.data(), values, n * sizeof(uint32_t));

    // Write n_blocks + space for block_offsets
    size_t field_start = out.size();
    out.push_back(n_blocks);
    size_t offsets_pos = out.size();
    out.resize(out.size() + n_blocks + 1, 0);  // block_off[0..n_blocks]

    size_t data_start = out.size();

    for (uint32_t b = 0; b < n_blocks; b++) {
        out[offsets_pos + b] = (uint32_t)(out.size() - data_start);
        uint32_t* blk = vals.data() + b * BLOCK_SZ;

        uint32_t min_val = blk[0];
        for (uint32_t i = 1; i < BLOCK_SZ; i++)
            min_val = std::min(min_val, blk[i]);

        uint32_t max_bw = 0;
        for (uint32_t i = 0; i < BLOCK_SZ; i++) {
            blk[i] -= min_val;
            if (blk[i] > 0) {
                uint32_t bw = 32 - __builtin_clz(blk[i]);
                max_bw = std::max(max_bw, bw);
            }
        }

        out.push_back(min_val);
        out.push_back(max_bw | (max_bw << 8) | (max_bw << 16) | (max_bw << 24));

        // Bit-pack 4 miniblocks of 32
        for (uint32_t mb = 0; mb < 4; mb++) {
            uint32_t word = 0, shift = 0;
            for (uint32_t i = 0; i < 32; i++) {
                uint32_t v = blk[mb * 32 + i];
                if (max_bw == 0) continue;
                if (shift + max_bw > 32) {
                    if (shift != 32) word |= v << shift;
                    out.push_back(word);
                    shift = (shift + max_bw) & 31;
                    word = v >> (max_bw - shift);
                } else {
                    word |= v << shift;
                    shift += max_bw;
                }
            }
            out.push_back(word);
        }
    }
    out[offsets_pos + n_blocks] = (uint32_t)(out.size() - data_start);
}

// ============================================================
// parseq_encode — PFOR-encode 3 fields (lit_len, offset, match_len)
//                 from Lz4Seq metadata.
//
// Output layout (uint32_t words):
//   [n_seqs][field1_off][field2_off][field0 data][field1 data][field2 data]
//
// field offsets are word indices into the output vector.
// ============================================================
inline std::vector<uint32_t> parseq_encode(const std::vector<Lz4Seq>& seqs) {
    uint32_t n = (uint32_t)seqs.size();

    // SoA extraction
    std::vector<uint32_t> lit_len(n), offset(n), match_len(n);
    for (uint32_t i = 0; i < n; i++) {
        lit_len[i]   = seqs[i].lit_len;
        offset[i]    = seqs[i].offset;
        match_len[i] = seqs[i].match_len;
    }

    // Header: [n_seqs, field1_off, field2_off]
    std::vector<uint32_t> out;
    out.push_back(n);
    out.push_back(0);  // placeholder: field1 offset
    out.push_back(0);  // placeholder: field2 offset

    // Field 0: lit_len
    pfor_encode_field(lit_len.data(), n, out);

    // Field 1: offset
    out[1] = (uint32_t)out.size();  // fill in field1 offset
    pfor_encode_field(offset.data(), n, out);

    // Field 2: match_len
    out[2] = (uint32_t)out.size();  // fill in field2 offset
    pfor_encode_field(match_len.data(), n, out);

    return out;
}
