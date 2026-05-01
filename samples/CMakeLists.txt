# samples/CMakeLists.txt — BaM sample programs

add_executable(bam_queue_info bam_queue_info.cu)
target_include_directories(bam_queue_info PRIVATE ${BAM_INCLUDE_DIRS})
target_link_libraries(bam_queue_info PRIVATE libnvm cudart)
target_compile_definitions(bam_queue_info PRIVATE __is_convertible=__bam_is_convertible)
set_target_properties(bam_queue_info PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
)
target_compile_options(bam_queue_info PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
)

# ---------------------------------------------------------------------------
# IO Pipeline Benchmark — Fused vs Separated IO/Decomp warp designs
# C++17 + separable compilation, links bam_io_device + bam_kernel for ctrl API.
# ---------------------------------------------------------------------------
add_executable(io_pipeline_bench io_pipeline_bench.cu)
target_include_directories(io_pipeline_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
)
target_link_libraries(io_pipeline_bench PRIVATE bam_io_device bam_q6 cudart)
set_target_properties(io_pipeline_bench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(io_pipeline_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
)

# ---------------------------------------------------------------------------
# IO / Decomp Single-Page Latency — measures t_io and t_decomp for cost model
# C++17 + nvCOMPdx + CPU LZ4 + separable compilation + device linking.
# ---------------------------------------------------------------------------
add_executable(io_decomp_latency io_decomp_latency.cu)
target_include_directories(io_decomp_latency PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(io_decomp_latency PRIVATE
    bam_io_device bam_q6 cudart ${LZ4_LIBRARIES}
)
set_target_properties(io_decomp_latency PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(io_decomp_latency PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(io_decomp_latency PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# LZ4 Decomp Bench — nvCOMPdx vs lz4_decompress_warp throughput comparison
# Pure GPU benchmark (no BaM required). C++17 + nvCOMPdx + CPU LZ4.
# ---------------------------------------------------------------------------
add_executable(lz4_decomp_bench lz4_decomp_bench.cu)
target_include_directories(lz4_decomp_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
    ${MATHDX_DIR}/include
    ${MATHDX_DIR}/external/cutlass/include
)
target_link_libraries(lz4_decomp_bench PRIVATE
    cudart ${LZ4_LIBRARIES}
)
set_target_properties(lz4_decomp_bench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
target_compile_options(lz4_decomp_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w -Xptxas=-v --expt-relaxed-constexpr>
)
target_link_options(lz4_decomp_bench PRIVATE
    $<DEVICE_LINK:-dlto>
    $<DEVICE_LINK:${MATHDX_DIR}/lib/libnvcompdx.fatbin>
)

# ---------------------------------------------------------------------------
# BaM Submit/Poll Separation Test
# Tests whether splitting NVMe IO into separate submit and poll kernels works.
# C++17 + separable compilation, links bam_io_device + bam_kernel for ctrl API.
# ---------------------------------------------------------------------------
add_executable(bam_sp_test bam_sp_test.cu)
target_include_directories(bam_sp_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tpch
)
target_link_libraries(bam_sp_test PRIVATE bam_io_device bam_q6 cudart)
set_target_properties(bam_sp_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(bam_sp_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
)
