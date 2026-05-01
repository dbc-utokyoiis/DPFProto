#include "common/shim_nvml.cu"
#include "./ssb_cli.cu"

#include "common/xtn.h"
#include "schema/ssb_tables.cuh"

#include "ssb/gidp.cu"
#include "ssb/gidp_bam.cu"
#include "ssb/gidp_bam_fusion.cu"
#include "ssb/datapathfusion.cu"

static std::string format_elapsed_msec(uint64_t elapsed_nanoseconds) {
    uint64_t msec = elapsed_nanoseconds / 1'000'000;
    uint64_t frac = (elapsed_nanoseconds % 1'000'000) / 1'000;
    std::ostringstream oss;
    oss << msec << '.' << std::setw(3) << std::setfill('0') << frac << " msec";
    return oss.str();
}

static void output_ssb_benchmark_result(const BenchmarkOptions &options, const BenchmarkResult &result)
{
    uint64_t read_bytes = result.read_bytes ? result.read_bytes
                         : (result.nios * options.io_size);
    std::cout
        << "time: " << format_elapsed_msec(result.elapsed_nanoseconds) << "\n"
        << "nios: " << result.nios << "\n"
        << "read_mb: " << read_bytes / MEBI << "\n";
    if (result.total_pages > 0) {
        uint64_t uncompressed_bytes = result.total_pages * options.io_size;
        uint64_t uncompressed_mb = uncompressed_bytes / MEBI;
        double io_reduction = (double)read_bytes / (double)uncompressed_bytes;
        double elapsed_sec = result.elapsed_nanoseconds / 1e9;
        double throughput_gbs = elapsed_sec > 0
            ? ((double)uncompressed_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec : 0;
        double io_throughput_gbs = elapsed_sec > 0
            ? ((double)read_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec : 0;
        std::cout
            << "uncompressed_read_mb: " << uncompressed_mb << "\n"
            << std::fixed << std::setprecision(3)
            << "io_reduction_ratio: " << io_reduction << "\n"
            << std::setprecision(2)
            << "effective_throughput_gbs: " << throughput_gbs << "\n"
            << "io_throughput_gbs: " << io_throughput_gbs << "\n"
            << std::defaultfloat;
    }
    std::cout
        << "benchmark_kind: " << options.benchmark_kind_str << "\n"
        << "query: " << options.query_str << "\n"
        << "transfer_type: " << options.transfer_type_str << "\n"
        << "sync_io: " << (options.use_sync_io ? "true" : "false") << "\n"
        << "zonemap: " << (options.enable_zonemap ? "true" : "false") << "\n"
        << "file: " << options.file << "\n"
        << "number_of_threads: " << options.nthreads << "\n"
        << "elapsed_nanoseconds: " << result.elapsed_nanoseconds << "\n"
        << "compression: " << (result.compression.empty() ? "N/A" : result.compression) << "\n"
        << "gpu_mem_mb: " << result.gpu_mem_bytes / MEBI << "\n"
        << "gpu_ctrl_mb: " << result.gpu_ctrl_bytes / MEBI << "\n"
        << "gpu_app_mb: " << result.gpu_app_bytes / MEBI << "\n"
        << "kernel_launches: " << result.kernel_launches << "\n"
        << std::endl;
}

template <typename QueryEnum>
static void unsupported_mode(const char *query_name) {
    std::cerr << "Unsupported execution mode for " << query_name << std::endl;
    exit(EXIT_FAILURE);
}

static void not_yet_implemented(const char *query_name, const char *mode_name) {
    std::cerr << query_name << " with " << mode_name << " is not yet implemented" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char *const *argv)
{
    std::cerr << "ssbdb" << std::endl;

    BenchmarkOptions options = parse_ssb_benchmark_options(argc, argv);
    validate_ssb_benchmark_options(options);

    nvml::init();

    BenchmarkResult result = {};

    // Dispatch: 13 SSB queries x 4 execution modes
    if (Benchmark::is_query(options.job, SSB::Query::Q11)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q11_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q11_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q11_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q11_datapathfusion(options);
        else
            not_yet_implemented("Q1.1", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q12)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q12_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q12_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q12_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q12_datapathfusion(options);
        else
            not_yet_implemented("Q1.2", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q13)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q13_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q13_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q13_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q13_datapathfusion(options);
        else
            not_yet_implemented("Q1.3", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q21)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q21_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q21_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q21_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q21_datapathfusion(options);
        else
            not_yet_implemented("Q2.1", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q22)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q22_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q22_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q22_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q22_datapathfusion(options);
        else
            not_yet_implemented("Q2.2", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q23)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q23_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q23_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q23_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q23_datapathfusion(options);
        else
            not_yet_implemented("Q2.3", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q31)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q31_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q31_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q31_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q31_datapathfusion(options);
        else
            not_yet_implemented("Q3.1", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q32)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q32_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q32_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q32_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q32_datapathfusion(options);
        else
            not_yet_implemented("Q3.2", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q33)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q33_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q33_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q33_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q33_datapathfusion(options);
        else
            not_yet_implemented("Q3.3", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q34)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q34_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q34_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q34_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q34_datapathfusion(options);
        else
            not_yet_implemented("Q3.4", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q41)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q41_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q41_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q41_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q41_datapathfusion(options);
        else
            not_yet_implemented("Q4.1", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q42)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q42_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q42_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q42_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q42_datapathfusion(options);
        else
            not_yet_implemented("Q4.2", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::Q43)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_q43_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_q43_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_q43_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_q43_datapathfusion(options);
        else
            not_yet_implemented("Q4.3", options.transfer_type_str);
    }
    else if (Benchmark::is_query(options.job, SSB::Query::REVENUE)) {
        if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP))
            result = ssb_revenue_gidp(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM))
            result = ssb_revenue_gidp_bam(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::GIDP_BAM_FUSION))
            result = ssb_revenue_gidp_bam_fusion(options);
        else if (Benchmark::is_transfertype(options.job, Benchmark::TransferType::DATAPATHFUSION))
            result = ssb_revenue_datapathfusion(options);
        else
            not_yet_implemented("revenue", options.transfer_type_str);
    }
    else {
        std::cerr << "Unsupported SSB query" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cerr << "Elapsed time (ns): " << result.elapsed_nanoseconds << std::endl;
    output_ssb_benchmark_result(options, result);

    nvml::shutdown();
}
