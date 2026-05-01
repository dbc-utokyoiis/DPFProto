#pragma once

enum class CompressionMethod {
    NONE = 0, /* default value */
    PFOR,
    PFOR64,
    PFORDELTA, /* no impl */
    RLE, /* no impl */
    LZ4,
    /* The following methods are only for GOLAP */
    DEFLATE,
    SNAPPY,
    /* GOLAP chooses one of DEFLATE, SNAPPY, or LZ4 per column */
    GOLAP,
    LZ4PAR,
    FSST,
    FSST_ROWID,
};

// Pre-measurement of decompression throughputs for each candidate
// compression scheme (LZ4, Snappy, Deflate) used in the per-column
// scheme selection procedure.
//
// The throughputs are measured once on representative TPC-H dbgen data
// using the nvCOMP benchmark binaries from NVIDIA's CUDALibrarySamples.
// The measured values are then combined with per-column compression
// ratios sampled at loading time to estimate the effective throughput
// at query time.
//
// Reproduction steps:
//
//   1. Generate TPC-H SF=10 data with dbgen, then split customer.tbl
//      into 128MB chunks:
//        $ split -b 128M customer.tbl
//
//   2. Clone NVIDIA's CUDALibrarySamples and check out the pinned
//      revision used for our measurements:
//        $ git clone https://github.com/NVIDIA/CUDALibrarySamples.git
//        $ cd CUDALibrarySamples
//        $ git checkout 1a44fc256e226e116626e124490c25aea792b68c
//
//   3. Build the nvCOMP benchmarks:
//        $ cd nvCOMP/benchmarks && mkdir build && cd build
//        $ cmake .. -DCMAKE_PREFIX_PATH=<nvCOMP sysroot path> \
//                   -DCMAKE_BUILD_TYPE=Release
//        $ make
//
//   4. Run the per-scheme benchmarks against the split file (xaa) with
//      a 65536-byte page size:
//        $ ./benchmark_lz4_chunked     -f /path/to/tpch/input10/test/xaa -p 65536
//        $ ./benchmark_snappy_chunked  -f /path/to/tpch/input10/test/xaa -p 65536
//        $ ./benchmark_deflate_chunked -f /path/to/tpch/input10/test/xaa -p 65536
//
//   5. Use the reported decompression throughputs as the per-scheme
//      values stored in this file.
enum class DecompressionSpeed {
    SNAPPY  = 434, // *100MB/s
    DEFLATE = 194, // *100MB/s
    LZ4 = 328, // *100MB/s
};


#if 0
size_t compress_max_size(CompressionMethod method, size_t input_size) {
    switch (method) {
        case CompressionMethod::SNAPPY:
            return snappy::MaxCompressedLength(input_size);
        case CompressionMethod::DEFLATE:
            return compressBound(input_size);
        case CompressionMethod::LZ4:
            return LZ4_compressBound(static_cast<int>(input_size));
        default:
            return input_size; // no compression
    }
}
#endif

std::string compression_method_name(CompressionMethod method) {
    switch (method) {
        case CompressionMethod::NONE:
            return "NONE";
        case CompressionMethod::PFOR:
            return "PFOR";
        case CompressionMethod::PFOR64:
            return "PFOR64";
        case CompressionMethod::PFORDELTA:
            return "PFORDELTA";
        case CompressionMethod::RLE:
            return "RLE";
        case CompressionMethod::LZ4:
            return "LZ4";
        case CompressionMethod::DEFLATE:
            return "DEFLATE";
        case CompressionMethod::SNAPPY:
            return "SNAPPY";
        case CompressionMethod::GOLAP:
            return "GOLAP";
        case CompressionMethod::LZ4PAR:
            return "LZ4PAR";
        case CompressionMethod::FSST:
            return "FSST";
        case CompressionMethod::FSST_ROWID:
            return "FSST_ROWID";
        default:
            return "UNKNOWN";
    }
}
