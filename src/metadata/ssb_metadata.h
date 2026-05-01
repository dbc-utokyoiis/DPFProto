#pragma once

#include "common/page.cu"
#include "common/compression.cuh"
#include "metadata/metadata.h"

template <typename T, size_t N>
inline void ssb_prepare_fields_metadata(
    std::vector<int> &fds,
    const SSBTableMetadata &metadata,
    SSB::common::Table table,
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
        switch (table) {
            case SSB::common::Table::LINEORDER:
                info.start_page_id = metadata.table_lineorder_start_page_ids[field_idx];
                info.npages = metadata.table_lineorder_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_lineorder_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_lineorder_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_lineorder_prefix_sum_start_page_ids[field_idx];
                break;
            case SSB::common::Table::DDATE:
                info.start_page_id = metadata.table_date_start_page_ids[field_idx];
                info.npages = metadata.table_date_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_date_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_date_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_date_prefix_sum_start_page_ids[field_idx];
                break;
            case SSB::common::Table::CUSTOMER:
                info.start_page_id = metadata.table_customer_start_page_ids[field_idx];
                info.npages = metadata.table_customer_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_customer_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_customer_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_customer_prefix_sum_start_page_ids[field_idx];
                break;
            case SSB::common::Table::SUPPLIER:
                info.start_page_id = metadata.table_supplier_start_page_ids[field_idx];
                info.npages = metadata.table_supplier_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_supplier_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_supplier_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_supplier_prefix_sum_start_page_ids[field_idx];
                break;
            case SSB::common::Table::PART:
                info.start_page_id = metadata.table_part_start_page_ids[field_idx];
                info.npages = metadata.table_part_npages[field_idx];
                info.compression_method = static_cast<CompressionMethod>(
                    metadata.table_part_compression_method[field_idx]);
                prefix_sum_npages = metadata.table_part_prefix_sum_npages[field_idx];
                prefix_sum_start = metadata.table_part_prefix_sum_start_page_ids[field_idx];
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

            switch (table) {
                case SSB::common::Table::LINEORDER:
                    comp_sizes_npages = metadata.table_lineorder_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_lineorder_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_lineorder_compression_nbases[field_idx];
                    base_start = metadata.table_lineorder_compression_base_start_page_ids[field_idx];
                    break;
                case SSB::common::Table::DDATE:
                    comp_sizes_npages = metadata.table_date_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_date_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_date_compression_nbases[field_idx];
                    base_start = metadata.table_date_compression_base_start_page_ids[field_idx];
                    break;
                case SSB::common::Table::CUSTOMER:
                    comp_sizes_npages = metadata.table_customer_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_customer_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_customer_compression_nbases[field_idx];
                    base_start = metadata.table_customer_compression_base_start_page_ids[field_idx];
                    break;
                case SSB::common::Table::SUPPLIER:
                    comp_sizes_npages = metadata.table_supplier_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_supplier_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_supplier_compression_nbases[field_idx];
                    base_start = metadata.table_supplier_compression_base_start_page_ids[field_idx];
                    break;
                case SSB::common::Table::PART:
                    comp_sizes_npages = metadata.table_part_compressed_page_sizes_npages[field_idx];
                    comp_sizes_start = metadata.table_part_compressed_page_sizes_start_page_ids[field_idx];
                    nbase = metadata.table_part_compression_nbases[field_idx];
                    base_start = metadata.table_part_compression_base_start_page_ids[field_idx];
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
            size_t base_npages = SSB::nbase_to_npages(nbase, page_size);
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
