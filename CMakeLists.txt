cmake_minimum_required(VERSION 3.22)

option(ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(ENABLE_TSAN "Build with ThreadSanitizer" OFF)

# For debug
if(MSVC)
    add_compile_options(/errorLimit:1)
else()
    add_compile_options(-fmax-errors=1)
endif()

if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()
if(ENABLE_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_PERF "-O2 -ggdb3 -fno-omit-frame-pointer -fno-optimize-sibling-calls")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# Override default CUDA debug flags: drop -G (device-debug) which conflicts
# with INTERPROCEDURAL_OPTIMIZATION (--link-time-opt) on nvCOMPdx targets.
set(CMAKE_CUDA_FLAGS_DEBUG "-g -O0 -lineinfo")

# BAM: GCC 12+ reserves __is_convertible as a builtin; rename to avoid
# collision with the struct in BAM's freestanding libcxx headers.
add_compile_definitions(__is_convertible=__bam_is_convertible)

project("gdsdb" CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)
pkg_check_modules(ZLIB REQUIRED zlib)
pkg_check_modules(SNAPPY REQUIRED snappy)


#set(CMAKE_BUILD_TYPE Debug)
#set(NVCOMP_PATH /usr/local/nvcomp-4.0.1)
# 5.1.0
set(NVCOMP_PATH $ENV{HOME}/libs/nvcomp/)
set(CUDA_SAMPLES_INCLUDE_PATH cuda-samples/Common/)

# ---------------------------------------------------------------------------
# BAM (Big Accelerator Memory) — GPU-initiated NVMe I/O
# ---------------------------------------------------------------------------
set(BAM_DIR "${CMAKE_SOURCE_DIR}/bam")
set(BAM_INCLUDE_DIRS
    ${BAM_DIR}/include
    ${BAM_DIR}/include/freestanding/include
)
cmake_policy(SET CMP0146 OLD)
set(no_module  true  CACHE BOOL "" FORCE)
set(no_smartio true  CACHE BOOL "" FORCE)
# BAM is pure C++; the parent's -gdb flag (CUDA-specific) is invalid for g++.
set(_saved_cxx_flags_debug "${CMAKE_CXX_FLAGS_DEBUG}")
string(REPLACE "-gdb" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
add_subdirectory(${BAM_DIR} ${CMAKE_BINARY_DIR}/bam)

set(MATHDX_DIR "$ENV{HOME}/libs/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12")

# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
add_subdirectory(samples)
set(CMAKE_CXX_FLAGS_DEBUG "${_saved_cxx_flags_debug}")

link_directories(${NVCOMP_PATH}/lib)
include_directories(${NVCOMP_PATH}/include)
include_directories(${CUDA_SAMPLES_INCLUDE_PATH})

include_directories(${CMAKE_SOURCE_DIR}/src/)
# include_directories(${CMAKE_SOURCE_DIR}/src/schema)

# ---------------------------------------------------------------------------
# BAM Q6 kernel library (CUDA C++11 — avoids freestanding-libcxx vs C++20 conflict)
# ---------------------------------------------------------------------------
add_library(bam_q6 STATIC src/tpch/bam_kernel.cu)
target_include_directories(bam_q6 PRIVATE ${BAM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/tpch ${CMAKE_SOURCE_DIR}/src/kernel)
target_link_libraries(bam_q6 PRIVATE libnvm cudart)
set_target_properties(bam_q6 PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
)
target_compile_options(bam_q6 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v>
)

# ---------------------------------------------------------------------------
# BAM VCHAR decompress+scan library (CUDA C++17 — nvCOMPdx device-side LZ4)
# No BAM headers: I/O is handled by bam_q6 (C++11) via bam_vchar_io_*.
# Requires CUDA separable compilation + device linking for nvCOMPdx.
# ---------------------------------------------------------------------------

add_library(bam_vchar STATIC src/tpch/bam_vchar_kernel.cu)
target_include_directories(bam_vchar PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include                       # nvCOMPdx
    ${MATHDX_DIR}/external/cutlass/include      # cutlass (nvCOMPdx dep)
)
target_link_libraries(bam_vchar PRIVATE cudart)
set_target_properties(bam_vchar PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_vchar PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
# Device-link against nvCOMPdx fatbin library (CUDA 12.8+)
target_link_options(bam_vchar PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# BAM IO device wrapper (CUDA C++11 + separable compilation)
# Thin wrapper for BaM page_cache I/O, device-linked into C++17 targets.
# ---------------------------------------------------------------------------
add_library(bam_io_device STATIC src/tpch/bam_io_device.cu)
target_include_directories(bam_io_device PRIVATE
    ${BAM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/tpch
)
target_link_libraries(bam_io_device PRIVATE libnvm cudart)
set_target_properties(bam_io_device PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(bam_io_device PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
)

# ---------------------------------------------------------------------------
# BAM I/O contention microbenchmark (CUDA C++17 + separable compilation)
# Completion-driven I/O loop measuring QP contention effects.
# ---------------------------------------------------------------------------
add_library(bam_io_bench STATIC src/tpch/bam_io_contention_bench.cu)
target_include_directories(bam_io_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
)
target_link_libraries(bam_io_bench PRIVATE bam_io_device cudart)
set_target_properties(bam_io_bench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(bam_io_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
)

# ---------------------------------------------------------------------------
# BAM LZ4 decomp-only microbenchmark (CUDA C++17 + nvCOMPdx)
# Pre-loads pages via BaM sync I/O, then measures decomp-only throughput.
# ---------------------------------------------------------------------------
add_library(bam_decomp_bench STATIC src/tpch/bam_lz4_decomp_bench.cu)
target_include_directories(bam_decomp_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_decomp_bench PRIVATE bam_io_device cudart)
set_target_properties(bam_decomp_bench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_decomp_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_decomp_bench PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# Pure nvCOMPdx LZ4 decomp benchmark (no BaM dependency)
# Measures warp-level LZ4 decompression throughput from plain GPU memory.
# ---------------------------------------------------------------------------
add_library(nvcompdx_lz4_decomp_lib STATIC src/tpch/nvcompdx_lz4_bench.cu)
target_include_directories(nvcompdx_lz4_decomp_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(nvcompdx_lz4_decomp_lib PRIVATE cudart)
set_target_properties(nvcompdx_lz4_decomp_lib PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(nvcompdx_lz4_decomp_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(nvcompdx_lz4_decomp_lib PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# BAM Bulk Read library (CUDA C++17 + separable compilation)
# GPU-initiated NVMe bulk page reads into GPU buffers.
# Uses bam_io_device for device-side I/O primitives.
# ---------------------------------------------------------------------------
add_library(bam_bulk_read STATIC src/tpch/bam_bulk_read.cu)
target_include_directories(bam_bulk_read PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
)
target_link_libraries(bam_bulk_read PRIVATE bam_io_device cudart)
set_target_properties(bam_bulk_read PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(bam_bulk_read PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
)

# ---------------------------------------------------------------------------
# BAM LZ4 device-side decompression library (CUDA C++17 — nvCOMPdx)
# Batch LZ4 decompress using nvCOMPdx warp-cooperative API.
# Used by gidp+bam+fusion execution mode.
# ---------------------------------------------------------------------------
add_library(bam_lz4_decomp STATIC src/tpch/bam_lz4_decomp.cu)
target_include_directories(bam_lz4_decomp PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${MATHDX_DIR}/include                       # nvCOMPdx
    ${MATHDX_DIR}/external/cutlass/include      # cutlass (nvCOMPdx dep)
)
target_link_libraries(bam_lz4_decomp PRIVATE cudart)
set_target_properties(bam_lz4_decomp PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_lz4_decomp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_lz4_decomp PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# BAM Fused Q6 kernel (CUDA C++17 — nvCOMPdx + device-linked BaM I/O)
# Fused I/O + LZ4 decompress + Q6 predicate/revenue in a single kernel.
# ---------------------------------------------------------------------------
add_library(bam_fused_q6 STATIC src/tpch/bam_lz4_fused_q6.cu)
target_include_directories(bam_fused_q6 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include                       # nvCOMPdx
    ${MATHDX_DIR}/external/cutlass/include      # cutlass (nvCOMPdx dep)
)
target_link_libraries(bam_fused_q6 PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q6 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q6 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q6 PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# BAM Fused Revenue kernel (CUDA C++17 — nvCOMPdx + device-linked BaM I/O)
# Fused I/O + LZ4 decompress + Revenue predicate in a single kernel.
# Same 4 fields as Q6 but different predicates: shipdate range + optional quantity.
# ---------------------------------------------------------------------------
add_library(bam_fused_revenue STATIC src/tpch/bam_lz4_fused_revenue.cu)
target_include_directories(bam_fused_revenue PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include                       # nvCOMPdx
    ${MATHDX_DIR}/external/cutlass/include      # cutlass (nvCOMPdx dep)
)
target_link_libraries(bam_fused_revenue PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_revenue PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_revenue PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_revenue PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# Q1-dedicated BaM IO device wrapper (CUDA 17 + LTO)
# Duplicates bam_io_device but compiled with -dlto so nvlink can
# re-optimize register usage for bam_fused_q1's higher-occupancy
# launch_bounds. Isolated from the shared bam_io_device library to
# avoid perturbing other fused kernels.
# ---------------------------------------------------------------------------
add_library(bam_io_device_q1 STATIC src/tpch/bam_io_device_q1.cu)
target_include_directories(bam_io_device_q1 PRIVATE
    ${BAM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src/tpch
)
target_link_libraries(bam_io_device_q1 PRIVATE libnvm cudart)
set_target_properties(bam_io_device_q1 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_io_device_q1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)

# ---------------------------------------------------------------------------
# Fused BaM I/O + LZ4 decompress + Q1 aggregate (bam_lz4_fused_q1)
# 7 INT32 LINEITEM fields, 2-round I/O (4+3), persistent kernel.
# ---------------------------------------------------------------------------
add_library(bam_fused_q1 STATIC src/tpch/bam_lz4_fused_q1.cu)
target_include_directories(bam_fused_q1 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q1 PRIVATE bam_io_device bam_io_device_q1 cudart)
set_target_properties(bam_fused_q1 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q1 PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# Fused BaM I/O + LZ4 decompress + Q5 LINEITEM probe (bam_fused_q5_lineitem)
# 2 INT32 + 2 INT64 LINEITEM fields, persistent kernel with HT probe.
# ---------------------------------------------------------------------------
add_library(bam_fused_q5li STATIC src/tpch/bam_lz4_fused_q5_lineitem.cu)
target_include_directories(bam_fused_q5li PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q5li PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q5li PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q5li PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q5li PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Q5 LINEITEM v2: Q1-style balanced pipeline (1 IO + 8 decomp warps)
add_library(bam_fused_q5li_v2 STATIC src/tpch/bam_lz4_fused_q5_lineitem_v2.cu)
target_include_directories(bam_fused_q5li_v2 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q5li_v2 PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q5li_v2 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q5li_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q5li_v2 PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Fused Q3 LINEITEM kernel (C++17, nvCOMPdx + device-linked BaM I/O)
add_library(bam_fused_q3li STATIC src/tpch/bam_lz4_fused_q3_lineitem.cu)
target_include_directories(bam_fused_q3li PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q3li PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q3li PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q3li PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q3li PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Fused Q3 ORDERS kernel (C++17, nvCOMPdx + device-linked BaM I/O)
add_library(bam_fused_q3ord STATIC src/tpch/bam_lz4_fused_q3_orders.cu)
target_include_directories(bam_fused_q3ord PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q3ord PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q3ord PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q3ord PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q3ord PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Q3SEL separate IO+decomp → scan kernels (C++17, nvCOMPdx + device-linked BaM I/O)
add_library(bam_q3sel_scan STATIC src/tpch/bam_lz4_q3sel_scan.cu)
target_include_directories(bam_q3sel_scan PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_q3sel_scan PRIVATE bam_io_device cudart)
set_target_properties(bam_q3sel_scan PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_q3sel_scan PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_q3sel_scan PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

add_library(bam_fused_q5ord STATIC src/tpch/bam_lz4_fused_q5_orders.cu)
target_include_directories(bam_fused_q5ord PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q5ord PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q5ord PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q5ord PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q5ord PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

add_library(bam_fused_q16ps STATIC src/tpch/bam_lz4_fused_q16_partsupp.cu)
target_include_directories(bam_fused_q16ps PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q16ps PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q16ps PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q16ps PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q16ps PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Split IO/Compute Q16 PARTSUPP kernel (nvCOMPdx + device-linked BaM I/O)
add_library(bam_split_q16ps STATIC src/tpch/bam_lz4_split_q16_partsupp.cu)
target_include_directories(bam_split_q16ps PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_split_q16ps PRIVATE bam_io_device cudart)
set_target_properties(bam_split_q16ps PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_split_q16ps PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_split_q16ps PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Fused Q16 P_TYPE kernel (nvCOMPdx + device-linked BaM I/O)
add_library(bam_fused_q16pt STATIC src/tpch/bam_lz4_fused_q16_ptype.cu)
target_include_directories(bam_fused_q16pt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q16pt PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q16pt PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q16pt PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q16pt PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Fused Q16 Phase01 kernel (nvCOMPdx + device-linked BaM I/O)
add_library(bam_fused_q16p01 STATIC src/tpch/bam_lz4_fused_q16_phase01.cu)
target_include_directories(bam_fused_q16p01 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q16p01 PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q16p01 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q16p01 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q16p01 PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Fused Q16 PART HT build kernel (nvCOMPdx + device-linked BaM I/O)
add_library(bam_fused_q16part STATIC src/tpch/bam_lz4_fused_q16_part.cu)
target_include_directories(bam_fused_q16part PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q16part PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q16part PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q16part PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q16part PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Fused Q13 O_COMMENT kernel (C++17, nvCOMPdx + device-linked BaM I/O)
# Standard LZ4 I/O + decompress + KMP scan in a single persistent kernel.
add_library(bam_fused_q13c STATIC src/tpch/bam_lz4_fused_q13_comment.cu)
target_include_directories(bam_fused_q13c PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q13c PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q13c PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q13c PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q13c PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Fused Q13 O_COMMENT v2 kernel (4-warp pattern, nvCOMPdx + BaM I/O)
add_library(bam_fused_q13c_v2 STATIC src/tpch/bam_lz4_fused_q13_comment_v2.cu)
target_include_directories(bam_fused_q13c_v2 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q13c_v2 PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q13c_v2 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q13c_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q13c_v2 PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# BAM Fused SSB kernels (CUDA C++17 — nvCOMPdx + device-linked BaM I/O)
# Warp-specialized fused I/O + LZ4 decompress + scan for all SSB queries.
# Q1x/Q2x/Q3x: 1024 threads (4 IO + 28 decomp warps), BATCH=7
# Q4x:          1024 threads (6 IO + 24 decomp warps), BATCH=4
# ---------------------------------------------------------------------------
add_library(bam_fused_ssb STATIC src/ssb/bam_lz4_fused_ssb.cu)
target_include_directories(bam_fused_ssb PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src/ssb
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include                       # nvCOMPdx
    ${MATHDX_DIR}/external/cutlass/include      # cutlass (nvCOMPdx dep)
)
target_link_libraries(bam_fused_ssb PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_ssb PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_ssb PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_ssb PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# BAM PFOR fused SSB library (CUDA C++17 — PFOR warp-spec, no nvCOMPdx)
# Adaptive kernel for datapathfusion: PFOR decomp when active pages are sparse.
# ---------------------------------------------------------------------------
add_library(bam_pfor_fused_ssb STATIC src/ssb/bam_pfor_fused_ssb.cu)
target_include_directories(bam_pfor_fused_ssb PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src/ssb
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/kernel
)
target_link_libraries(bam_pfor_fused_ssb PRIVATE bam_io_device cudart)
set_target_properties(bam_pfor_fused_ssb PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_pfor_fused_ssb PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_pfor_fused_ssb PRIVATE $<DEVICE_LINK:-dlto>)

# ---------------------------------------------------------------------------
# BAM Q13 decompress+scan library (CUDA C++17 — nvCOMPdx device-side LZ4)
# PAR-32K nvCOMPdx + KMP scan for Q13 O_COMMENT.
# Fused IO kernel calls bam_io_device via device linking.
# ---------------------------------------------------------------------------
add_library(bam_q13 STATIC src/tpch/bam_q13_kernel.cu)
target_include_directories(bam_q13 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include                       # nvCOMPdx
    ${MATHDX_DIR}/external/cutlass/include      # cutlass (nvCOMPdx dep)
)
target_link_libraries(bam_q13 PRIVATE bam_io_device cudart)
set_target_properties(bam_q13 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_q13 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
# Device-link against nvCOMPdx fatbin library (CUDA 12.8+)
target_link_options(bam_q13 PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# BAM Q16 fused IO+decomp+filter library (CUDA C++17 — nvCOMPdx device-side LZ4)
# Three fused kernels: S_COMMENT (KMP), P_BRAND (extract), P_TYPE (dict).
# Fused IO kernel calls bam_io_device via device linking.
# ---------------------------------------------------------------------------
add_library(bam_q16 STATIC src/tpch/bam_q16_kernel.cu)
target_include_directories(bam_q16 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include                       # nvCOMPdx
    ${MATHDX_DIR}/external/cutlass/include      # cutlass (nvCOMPdx dep)
)
target_link_libraries(bam_q16 PRIVATE bam_io_device cudart)
set_target_properties(bam_q16 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_q16 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_q16 PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# BAM Q16 FSST fused IO+decomp+filter library (CUDA C++17 — no nvCOMPdx)
# Three fused kernels for FSST-compressed columns:
#   S_COMMENT (KMP), P_BRAND (extract), P_TYPE (dict).
# Fused IO kernel calls bam_io_device via device linking.
# ---------------------------------------------------------------------------
add_library(bam_q16_fsst STATIC src/tpch/bam_q16_fsst_kernel.cu)
target_include_directories(bam_q16_fsst PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(bam_q16_fsst PRIVATE bam_io_device cudart)
set_target_properties(bam_q16_fsst PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(bam_q16_fsst PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v>
)

# ---------------------------------------------------------------------------
# BAM Q13 FSST fused IO+decomp+scan library (CUDA C++17 — no nvCOMPdx)
# Fused IO + FSST decomp + KMP scan for Q13 O_COMMENT.
# Fused IO kernel calls bam_io_device via device linking.
# ---------------------------------------------------------------------------
add_library(bam_q13_fsst STATIC src/tpch/bam_q13_fsst_kernel.cu)
target_include_directories(bam_q13_fsst PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(bam_q13_fsst PRIVATE bam_io_device cudart)
set_target_properties(bam_q13_fsst PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(bam_q13_fsst PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v>
)

# ---------------------------------------------------------------------------
# BAM Q1 unified kernel (PFOR IO+decomp+eval, device-linked BaM I/O)
# 7 INT32 LINEITEM fields, fused IO + PFOR decompress + aggregate.
# ---------------------------------------------------------------------------
add_library(bam_q1 STATIC src/tpch/bam_q1_kernel.cu)
target_include_directories(bam_q1 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(bam_q1 PRIVATE bam_io_device cudart)
set_target_properties(bam_q1 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_q1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_q1 PRIVATE
    $<DEVICE_LINK:-dlto>
)

# ---------------------------------------------------------------------------
# BAM Q3 fused IO+decomp+filter library (CUDA C++17 — nvCOMPdx device-side LZ4)
# C_MKTSEGMENT BUILDING filter + custkey hash set insert.
# Fused IO kernel calls bam_io_device via device linking.
# ---------------------------------------------------------------------------
add_library(bam_q3 STATIC src/tpch/bam_q3_kernel.cu)
target_include_directories(bam_q3 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include                       # nvCOMPdx
    ${MATHDX_DIR}/external/cutlass/include      # cutlass (nvCOMPdx dep)
)
target_link_libraries(bam_q3 PRIVATE bam_io_device cudart)
set_target_properties(bam_q3 PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_q3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_q3 PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Fused Q3 C_MKTSEGMENT IO + LZ4 decomp + BUILDING filter (gidp+bam+fusion)
add_library(bam_fused_q3_mktseg STATIC src/tpch/bam_lz4_fused_q3_mktseg.cu)
target_include_directories(bam_fused_q3_mktseg PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(bam_fused_q3_mktseg PRIVATE bam_io_device cudart)
set_target_properties(bam_fused_q3_mktseg PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(bam_fused_q3_mktseg PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(bam_fused_q3_mktseg PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# Q5 dimension table process kernel (gidp+bam+fusion)
add_library(bam_fused_q5dim STATIC src/tpch/bam_lz4_fused_q5_dim.cu)
target_include_directories(bam_fused_q5dim PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(bam_fused_q5dim PRIVATE cudart)
set_target_properties(bam_fused_q5dim PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(bam_fused_q5dim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w --expt-relaxed-constexpr>
)

# ---------------------------------------------------------------------------
# FSST library (CPU) — Fast Static Symbol Table string compression
# Separate library to avoid ODR issues with pag.h header-defined functions.
# ---------------------------------------------------------------------------
add_library(fsst_lib STATIC
    fsst/libfsst.cpp
    fsst/fsst_avx512.cpp
    src/common/fsst_host.cpp)
target_include_directories(fsst_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/fsst
    ${CMAKE_SOURCE_DIR}/src/common)
target_compile_options(fsst_lib PRIVATE -O3 -mavx512f -mavx512bw -mavx512dq)
set_target_properties(fsst_lib PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CXX)

add_executable(gdsdb src/main.cu src/kernel/tpch/q6.cu src/nvcomp/decompressor.cuh)
add_executable(ssbloader ${CMAKE_SOURCE_DIR}/src/ssb_loader_main.cu)
add_executable(ssbdb src/ssb_main.cu src/kernel/ssb/flatten.cu)
add_executable(tpchdb src/tpch_main.cu src/kernel/tpch/q6_col.cu src/kernel/tpch/revenue_col.cu src/kernel/tpch/scan.cu src/kernel/tpch/q13_scan.cu src/kernel/tpch/q16_scan.cu src/kernel/tpch/q5_scan.cu src/kernel/tpch/q3_scan.cu src/kernel/tpch/q1_scan.cu src/kernel/tpch/q6_scan.cu src/kernel/tpch/revenue_scan.cu)
add_executable(tpchloader ${CMAKE_SOURCE_DIR}/src/tpch_loader_main.cu)
add_executable(tpchchecker ${CMAKE_SOURCE_DIR}/src/tpch_checker_main.cu)
#add_executable(bench_cufile_register ${CMAKE_SOURCE_DIR}/src/bench_cufile_register.cu)
add_executable(gds_test ${CMAKE_SOURCE_DIR}/src/gds_test.cu)

target_link_libraries(gdsdb  PRIVATE cuda)
target_link_libraries(gdsdb  PRIVATE aio)
target_link_libraries(gdsdb  PRIVATE cufile)
target_link_libraries(gdsdb  PRIVATE nvidia-ml)
target_link_libraries(gdsdb  PRIVATE numa)
target_link_libraries(gdsdb  PRIVATE nvcomp)
target_include_directories(gdsdb PRIVATE ../cuda-samples/Common)
# target_include_directories(migmatite PRIVATE ${CMAKE_SOURCE_DIR}/src/common)
target_include_directories(gdsdb PRIVATE ${CMAKE_SOURCE_DIR}/src/kernel)
target_include_directories(gdsdb PRIVATE ${CMAKE_SOURCE_DIR}/src/schema)
target_include_directories(gdsdb PRIVATE ${CMAKE_SOURCE_DIR}/src/hash)

target_link_libraries(tpchdb  PRIVATE cuda)
target_link_libraries(tpchdb  PRIVATE aio)
target_link_libraries(tpchdb  PRIVATE cufile)
target_link_libraries(tpchdb  PRIVATE nvidia-ml)
target_link_libraries(tpchdb  PRIVATE numa)
target_link_libraries(tpchdb  PRIVATE nvcomp)
target_link_libraries(tpchdb  PRIVATE bam_q6)
target_link_libraries(tpchdb  PRIVATE bam_vchar)
target_link_libraries(tpchdb  PRIVATE bam_q13)
target_link_libraries(tpchdb  PRIVATE bam_q16)
target_link_libraries(tpchdb  PRIVATE bam_q16_fsst)
target_link_libraries(tpchdb  PRIVATE bam_q13_fsst)
target_link_libraries(tpchdb  PRIVATE bam_q1)
target_link_libraries(tpchdb  PRIVATE bam_q3)
target_link_libraries(tpchdb  PRIVATE bam_fused_q3_mktseg)
target_link_libraries(tpchdb  PRIVATE bam_fused_q5dim)
target_link_libraries(tpchdb  PRIVATE bam_io_device)
target_link_libraries(tpchdb  PRIVATE bam_io_bench)
target_link_libraries(tpchdb  PRIVATE bam_decomp_bench)
target_link_libraries(tpchdb  PRIVATE nvcompdx_lz4_decomp_lib)
target_link_libraries(tpchdb  PRIVATE bam_bulk_read)
target_link_libraries(tpchdb  PRIVATE bam_lz4_decomp)
target_link_libraries(tpchdb  PRIVATE bam_fused_q6)
target_link_libraries(tpchdb  PRIVATE bam_fused_revenue)
target_link_libraries(tpchdb  PRIVATE bam_fused_q1)
target_link_libraries(tpchdb  PRIVATE bam_fused_q5li)
target_link_libraries(tpchdb  PRIVATE bam_fused_q5li_v2)
target_link_libraries(tpchdb  PRIVATE bam_fused_q3li)
target_link_libraries(tpchdb  PRIVATE bam_fused_q3ord)
target_link_libraries(tpchdb  PRIVATE bam_q3sel_scan)
target_link_libraries(tpchdb  PRIVATE bam_fused_q5ord)
target_link_libraries(tpchdb  PRIVATE bam_fused_q16ps)
target_link_libraries(tpchdb  PRIVATE bam_split_q16ps)
target_link_libraries(tpchdb  PRIVATE bam_fused_q16pt)
target_link_libraries(tpchdb  PRIVATE bam_fused_q16p01)
target_link_libraries(tpchdb  PRIVATE bam_fused_q16part)
target_link_libraries(tpchdb  PRIVATE bam_fused_q13c)
target_link_libraries(tpchdb  PRIVATE bam_fused_q13c_v2)
target_link_libraries(tpchdb  PRIVATE fsst_lib)
# tpchdb needs CUDA separable compilation for nvCOMPdx device linking
set_target_properties(tpchdb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_options(tpchdb PRIVATE
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)
target_include_directories(tpchdb PRIVATE ../cuda-samples/Common)
# target_include_directories(migmatite PRIVATE ${CMAKE_SOURCE_DIR}/src/common)
target_include_directories(tpchdb PRIVATE ${CMAKE_SOURCE_DIR}/src/kernel)
target_include_directories(tpchdb PRIVATE ${CMAKE_SOURCE_DIR}/src/schema)
target_include_directories(tpchdb PRIVATE ${CMAKE_SOURCE_DIR}/src/hash)
target_include_directories(tpchdb PRIVATE ${CMAKE_SOURCE_DIR}/src/common)
target_include_directories(tpchdb PRIVATE ${CMAKE_SOURCE_DIR}/fsst)

target_link_libraries(ssbdb  PRIVATE cuda)
target_link_libraries(ssbdb  PRIVATE aio)
target_link_libraries(ssbdb  PRIVATE cufile)
target_link_libraries(ssbdb  PRIVATE nvidia-ml)
target_link_libraries(ssbdb  PRIVATE numa)
target_link_libraries(ssbdb  PRIVATE nvcomp)
target_link_libraries(ssbdb  PRIVATE bam_io_device)
target_link_libraries(ssbdb  PRIVATE bam_bulk_read)
target_link_libraries(ssbdb  PRIVATE bam_lz4_decomp)
target_link_libraries(ssbdb  PRIVATE bam_q6)
target_link_libraries(ssbdb  PRIVATE bam_fused_ssb)
target_link_libraries(ssbdb  PRIVATE bam_pfor_fused_ssb)
target_link_libraries(ssbdb  PRIVATE fsst_lib)
target_link_libraries(ssbdb  PRIVATE ${LZ4_LIBRARIES})
target_link_libraries(ssbdb  PRIVATE ${SNAPPY_LIBRARIES})
target_include_directories(ssbdb PRIVATE ${LZ4_INCLUDE_DIRS})
target_include_directories(ssbdb PRIVATE ${SNAPPY_INCLUDE_DIRS})
set_target_properties(ssbdb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_options(ssbdb PRIVATE
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)
target_include_directories(ssbdb PRIVATE ../cuda-samples/Common)
target_include_directories(ssbdb PRIVATE ${CMAKE_SOURCE_DIR}/src/kernel)
target_include_directories(ssbdb PRIVATE ${CMAKE_SOURCE_DIR}/src/schema)
target_include_directories(ssbdb PRIVATE ${CMAKE_SOURCE_DIR}/src/hash)
target_include_directories(ssbdb PRIVATE ${CMAKE_SOURCE_DIR}/src/common)
target_include_directories(ssbdb PRIVATE ${CMAKE_SOURCE_DIR}/fsst)

target_link_libraries(ssbloader  PRIVATE numa)
target_link_libraries(ssbloader  PRIVATE nvidia-ml)
target_link_libraries(ssbloader PRIVATE ${LZ4_LIBRARIES})
target_link_libraries(ssbloader PRIVATE ${ZLIB_LIBRARIES})
target_link_libraries(ssbloader PRIVATE ${SNAPPY_LIBRARIES})
target_include_directories(ssbloader PRIVATE ${LZ4_INCLUDE_DIRS})
target_include_directories(ssbloader PRIVATE ${ZLIB_INCLUDE_DIRS})
target_include_directories(ssbloader PRIVATE ${SNAPPY_INCLUDE_DIRS})
target_include_directories(ssbloader PRIVATE ${CMAKE_SOURCE_DIR}/src/schema)
target_link_libraries(ssbloader PRIVATE fsst_lib)
target_include_directories(ssbloader PRIVATE ${CMAKE_SOURCE_DIR}/fsst)

target_link_libraries(tpchloader  PRIVATE numa)
target_link_libraries(tpchloader  PRIVATE nvidia-ml)
target_link_libraries(tpchloader PRIVATE ${LZ4_LIBRARIES})
target_link_libraries(tpchloader PRIVATE ${ZLIB_LIBRARIES})
target_link_libraries(tpchloader PRIVATE ${SNAPPY_LIBRARIES})
target_link_libraries(tpchloader PRIVATE fsst_lib)
target_include_directories(tpchloader PRIVATE ${LZ4_INCLUDE_DIRS})
target_include_directories(tpchloader PRIVATE ${ZLIB_INCLUDE_DIRS})
target_include_directories(tpchloader PRIVATE ${SNAPPY_INCLUDE_DIRS})
target_include_directories(tpchloader PRIVATE ${CMAKE_SOURCE_DIR}/src/schema)
target_include_directories(tpchloader PRIVATE ${CMAKE_SOURCE_DIR}/fsst)

target_include_directories(tpchchecker PRIVATE ${CMAKE_SOURCE_DIR}/src/schema)

#target_link_libraries(bench_cufile_register PRIVATE cuda)
#target_link_libraries(bench_cufile_register PRIVATE cufile)
#target_include_directories(bench_cufile_register PRIVATE ../cuda-samples/Common)
#target_include_directories(bench_cufile_register PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(gds_test PRIVATE cuda)
target_link_libraries(gds_test PRIVATE cufile)
target_include_directories(gds_test PRIVATE ../cuda-samples/Common)
target_include_directories(gds_test PRIVATE ${CMAKE_SOURCE_DIR}/src)

#target_compile_options(gdsdb PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(gdsdb PRIVATE
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:
    -O3
    -Xptxas=-O3,-dlcm=cg,-v
    -lineinfo
    --use_fast_math
    --extended-lambda
  >
)

target_compile_options(tpchdb PRIVATE
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:
    -O3
    -Xptxas=-O3,-dlcm=cg,-v
    -lineinfo
    --use_fast_math
    --extended-lambda
  >
)

target_compile_options(ssbdb PRIVATE
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:
    -O3
    -Xptxas=-O3,-dlcm=cg,-v
    -lineinfo
    --use_fast_math
    --extended-lambda
  >
)

target_compile_options(ssbdb PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:
    -O0
    -Xptxas=-O0,-dlcm=cg,-v
    -lineinfo
    --use_fast_math
    --extended-lambda
  >
)

target_compile_options(ssbloader PRIVATE
  $<$<AND:$<CONFIG:Release>>:
    -O3
  >
)

target_compile_options(tpchloader PRIVATE
  $<$<AND:$<CONFIG:Release>>:
    -O3
  >
)

target_compile_options(tpchchecker PRIVATE
  $<$<AND:$<CONFIG:Release>>:
    -O3
  >
)



target_compile_options(gdsdb PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:
    -O0
    -Xptxas=-O0,-dlcm=cg,-v
    -lineinfo
    --use_fast_math
    --extended-lambda
  >
)

target_compile_options(tpchdb PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:
    -O0
    -Xptxas=-O0,-dlcm=cg,-v
    -lineinfo
    --use_fast_math
    --extended-lambda
  >
)

# ---------------------------------------------------------------------------
# nvCOMPdx LZ4 decompression microbenchmark
# Standalone binary: benchmarks/nvcompdx_lz4_bench.cu
# Compares nvCOMPdx (Warp/Block) vs nvCOMP host batch API.
# ---------------------------------------------------------------------------
add_executable(nvcompdx_lz4_bench benchmarks/nvcompdx_lz4_bench.cu)
target_include_directories(nvcompdx_lz4_bench PRIVATE
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
    ${NVCOMP_PATH}/include
)
target_link_libraries(nvcompdx_lz4_bench PRIVATE cudart nvcomp)
set_target_properties(nvcompdx_lz4_bench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(nvcompdx_lz4_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(nvcompdx_lz4_bench PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)
