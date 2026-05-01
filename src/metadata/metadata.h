#pragma once

#if 0
struct Metadata {
    size_t page_size;
    bool compressed;
    size_t table_customer_page_id;
    size_t table_customer_nrows;
    size_t table_customer_npages;
    size_t table_orders_page_id;
    size_t table_orders_nrows;
    size_t table_orders_npages;
    size_t table_lineitem_page_id;
    size_t table_lineitem_nrows;
    size_t table_lineitem_npages;
    size_t free_page_id;
};

struct SSBMetadata {
    size_t page_size;
    bool compressed;
    size_t table_customer_page_id;
    size_t table_customer_nrows;
    size_t table_customer_npages;
    size_t table_ddate_page_id;
    size_t table_ddate_nrows;
    size_t table_ddate_npages;
    size_t table_part_page_id;
    size_t table_part_nrows;
    size_t table_part_npages;
    size_t table_supplier_page_id;
    size_t table_supplier_nrows;
    size_t table_supplier_npages;
    size_t table_lineorder_page_id;
    size_t table_lineorder_nrows;
    size_t table_lineorder_npages;
    size_t free_page_id;
};
#endif


#include "common/page.cu"
#include "common/compression.cuh"

struct FieldPageInfo {
    size_t field_index;
    size_t start_page_id;
    size_t npages;
    CompressionMethod compression_method;

    // Prefix sums — posix_memalign'd, caller frees via free_fields_metadata
    uint64_t *prefix_sum_nrecs;

    // Compression metadata — nullptr if uncompressed
    uint32_t *compressed_page_sizes;    // posix_memalign'd
    size_t   *compressed_offsets;       // malloc'd, length = npages + 1
};

/* helper function to raed multiple data */
inline void read_pages(std::vector<int> &fds, void *buf, size_t start_page_id,
                        size_t npages, size_t page_size) {
    for (size_t j = 0; j < npages; j++) {
        page_pread_host(fds, reinterpret_cast<char *>(buf) + j * page_size,
                        start_page_id + j, page_size);
    }
}


template <typename T, size_t N>
inline void prepare_fields_metadata(
    std::vector<int> &fds,
    const TPCHTableMetadata &metadata,
    size_t page_size,
    const std::array<T, N> &target_fields,
    std::vector<FieldPageInfo> &out)
{

    for (size_t i = 0; i < N; i++) {
        size_t field_idx = target_fields[i];
        FieldPageInfo &info = out[i];
        size_t prefix_sum_npages = 0;
        size_t prefix_sum_start = 0;

        info.field_index = field_idx;
        switch (metadata.column) {
            case TPCH::common::Table::LINEITEM:
                info.start_page_id = metadata.table_lineitem_start_page_ids[field_idx];
                info.npages = metadata.table_lineitem_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_lineitem_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_lineitem_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_lineitem_prefix_sum_start_page_ids[field_idx];
                break;
            case TPCH::common::Table::ORDERS:
                info.start_page_id = metadata.table_orders_start_page_ids[field_idx];
                info.npages = metadata.table_orders_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_orders_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_orders_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_orders_prefix_sum_start_page_ids[field_idx];
                break;
            case TPCH::common::Table::CUSTOMER:
                info.start_page_id = metadata.table_customer_start_page_ids[field_idx];
                info.npages = metadata.table_customer_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_customer_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_customer_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_customer_prefix_sum_start_page_ids[field_idx];
                break;
            case TPCH::common::Table::SUPPLIER:
                info.start_page_id = metadata.table_supplier_start_page_ids[field_idx];
                info.npages = metadata.table_supplier_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_supplier_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_supplier_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_supplier_prefix_sum_start_page_ids[field_idx];
                break;
            case TPCH::common::Table::PART:
                info.start_page_id = metadata.table_part_start_page_ids[field_idx];
                info.npages = metadata.table_part_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_part_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_part_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_part_prefix_sum_start_page_ids[field_idx];
                break;
            case TPCH::common::Table::PARTSUPP:
                info.start_page_id = metadata.table_partsupp_start_page_ids[field_idx];
                info.npages = metadata.table_partsupp_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_partsupp_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_partsupp_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_partsupp_prefix_sum_start_page_ids[field_idx];
                break;
            case TPCH::common::Table::NATION:
                info.start_page_id = metadata.table_nation_start_page_ids[field_idx];
                info.npages = metadata.table_nation_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_nation_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_nation_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_nation_prefix_sum_start_page_ids[field_idx];
                break;
            case TPCH::common::Table::REGION:
                info.start_page_id = metadata.table_region_start_page_ids[field_idx];
                info.npages = metadata.table_region_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_region_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_region_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_region_prefix_sum_start_page_ids[field_idx];
                break;
            default:
                break;
        }

        info.prefix_sum_nrecs = nullptr;
        info.compressed_page_sizes = nullptr;
        info.compressed_offsets = nullptr;

        if (info.npages == 0) continue;

        // Load prefix sums
        if (prefix_sum_npages > 0) {
            void *buf = nullptr;
            if (posix_memalign(&buf, 512, prefix_sum_npages * page_size) != 0) {
                std::cerr << "posix_memalign failed for prefix_sum field=" << field_idx << std::endl;
                continue;
            }
            read_pages(fds, buf, prefix_sum_start, prefix_sum_npages, page_size);
            info.prefix_sum_nrecs = reinterpret_cast<uint64_t *>(buf);
        }

        // Load compression metadata
        if (info.compression_method != CompressionMethod::NONE) {
            size_t comp_sizes_npages = 0;
            size_t comp_sizes_start = 0;
            size_t nbase = 0;
            size_t base_start = 0;

            switch (metadata.column) {
                case TPCH::common::Table::LINEITEM:
                    comp_sizes_npages = metadata.table_lineitem_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_lineitem_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_lineitem_compression_nbases[field_idx];
                    base_start = metadata.table_lineitem_compression_base_start_page_ids[field_idx];
                    break;
                case TPCH::common::Table::ORDERS:
                    comp_sizes_npages = metadata.table_orders_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_orders_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_orders_compression_nbases[field_idx];
                    base_start = metadata.table_orders_compression_base_start_page_ids[field_idx];
                    break;
                case TPCH::common::Table::CUSTOMER:
                    comp_sizes_npages = metadata.table_customer_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_customer_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_customer_compression_nbases[field_idx];
                    base_start = metadata.table_customer_compression_base_start_page_ids[field_idx];
                    break;
                case TPCH::common::Table::SUPPLIER:
                    comp_sizes_npages = metadata.table_supplier_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_supplier_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_supplier_compression_nbases[field_idx];
                    base_start = metadata.table_supplier_compression_base_start_page_ids[field_idx];
                    break;
                case TPCH::common::Table::PART:
                    comp_sizes_npages = metadata.table_part_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_part_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_part_compression_nbases[field_idx];
                    base_start = metadata.table_part_compression_base_start_page_ids[field_idx];
                    break;
                case TPCH::common::Table::PARTSUPP:
                    comp_sizes_npages = metadata.table_partsupp_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_partsupp_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_partsupp_compression_nbases[field_idx];
                    base_start = metadata.table_partsupp_compression_base_start_page_ids[field_idx];
                    break;
                case TPCH::common::Table::NATION:
                    comp_sizes_npages = metadata.table_nation_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_nation_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_nation_compression_nbases[field_idx];
                    base_start = metadata.table_nation_compression_base_start_page_ids[field_idx];
                    break;
                case TPCH::common::Table::REGION:
                    comp_sizes_npages = metadata.table_region_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_region_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_region_compression_nbases[field_idx];
                    base_start = metadata.table_region_compression_base_start_page_ids[field_idx];
                    break;
                default:
                    break;
            }

            // Read compressed page sizes
            void *sizes_buf = nullptr;
            if (posix_memalign(&sizes_buf, 512, comp_sizes_npages * page_size) != 0) {
                std::cerr << "posix_memalign failed for comp_sizes field=" << field_idx << std::endl;
                continue;
            }
            read_pages(fds, sizes_buf, comp_sizes_start, comp_sizes_npages, page_size);
            info.compressed_page_sizes = reinterpret_cast<uint32_t *>(sizes_buf);

            // Read base page IDs and compute offsets
            size_t base_npages = TPCH::nbase_to_npages(nbase, page_size);
            void *bases_buf = nullptr;
            if (posix_memalign(&bases_buf, 512, base_npages * page_size) != 0) {
                std::cerr << "posix_memalign failed for comp_bases field=" << field_idx << std::endl;
                continue;
            }
            read_pages(fds, bases_buf, base_start, base_npages, page_size);

            std::vector<size_t> offsets_vec;
            calculate_compressed_offsets(
                reinterpret_cast<size_t *>(bases_buf),
                info.compressed_page_sizes,
                nbase,
                info.npages,
                page_size,
                info.start_page_id,
                fds.size(),
                offsets_vec);

            info.compressed_offsets = static_cast<size_t *>(
                std::malloc((info.npages + 1) * sizeof(size_t)));
            std::memcpy(info.compressed_offsets, offsets_vec.data(),
                         (info.npages + 1) * sizeof(size_t));

            free(bases_buf);
        }
    }
}

// Free all memory allocated by prepare_fields_metadata.
inline void free_fields_metadata(std::vector<FieldPageInfo> &info) {
    for (size_t i = 0; i < info.size(); i++) {
        if (info[i].prefix_sum_nrecs) {
            free(info[i].prefix_sum_nrecs);
            info[i].prefix_sum_nrecs = nullptr;
        }
        if (info[i].compressed_page_sizes) {
            free(info[i].compressed_page_sizes);
            info[i].compressed_page_sizes = nullptr;
        }
        if (info[i].compressed_offsets) {
            std::free(info[i].compressed_offsets);
            info[i].compressed_offsets = nullptr;
        }
    }
}
