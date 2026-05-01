/**
 * SSB Loader Main for column store
 * Adapted from tpch_loader_main.cu for SSB star schema.
 *
 * Loading order: DATE → CUSTOMER → SUPPLIER → PART → LINEORDER
 * (Dimension tables first for future sideways stats lookup tables)
 */
#include "./ssb_loader_cli.cu"

/* Verify function defined in ssb_loader.cu */
void test_scan_column_tables(LoaderOptions &options);

/* ------------------------------------------------------------------ */
/*  Generic multi-file table loader (template)                        */
/* ------------------------------------------------------------------ */
template <size_t NumVarCharFields, size_t NFields, size_t NumFilters,
    typename BulkLoadTargetField, typename TableType>
uint64_t loader_load_multiple_files_table_column(
    TableType table,
    const std::array<enum rec_type, NFields> &field_types,
    const std::array<size_t, NFields>& field_sizes,
    const std::array<BulkLoadTargetField, NFields>& target_fields,
    const std::array<size_t, NumVarCharFields>& target_varchar_field_indexes,
    const std::array<size_t, NumFilters>& filter_columns,
    const std::array<bool, NFields>& enable_stats_columns,
    const std::array<CompressionMethod, NFields>& compression_types,
    const std::array<size_t, NFields>& csv_column_map,
    LoaderOptions &options, SSBTableMetadata &metadata,
    uint64_t start_xtn_id, LoaderStats &stats)
{
    constexpr size_t MaxNumBuffers = NFields + NumVarCharFields;
    std::string s = std::string(SSB::common::table_name(table));
    std::transform(s.begin(), s.end(), s.begin(), [](char c) { return std::toupper(c); });
    auto table_name = s;
    StatEntry stat_entry;
    stat_entry.name = table_name;

    std::cout << "Loading " << table_name << " table..." << std::endl;

    /* target columns */
    std::array<char*, MaxNumBuffers> buffers;

    /* buffer and per-field pagids */
    std::string col = SSB::common::table_name(table);
    std::stringstream ss_basedir;
    ss_basedir << options.input_dirname << "/" << col;
    auto basedir = ss_basedir.str();
    size_t n = count_input_files(basedir, col);
    if (n == 0) {
        std::cerr << "No input files found for table: " << col << std::endl;
        return start_xtn_id; // No files to load
    }
    std::cout << "Allocated size: " << n * MaxNumBuffers * options.page_size << std::endl;
    char *buf = loader_aligned_alloc<char*>(n * MaxNumBuffers * options.page_size);
    for (size_t i = 0; i < NFields; i++)
    {
        buffers[i] = &buf[n * options.page_size * i];
    }

    /* Allocate memory here */
    std::array<std::vector<size_t>, NumVarCharFields> varchar_cluster_thresholds{};

    uint64_t next_xtn_id = bulkload_multiple_files_column<BulkLoadTargetField>(
        options, metadata, table,
        start_xtn_id,
        buffers,
        field_sizes, field_types,
        target_fields, target_varchar_field_indexes,
        varchar_cluster_thresholds,
        filter_columns,
        enable_stats_columns,
        compression_types,
        csv_column_map,
        stat_entry);

    free(buf);
    std::cout << table_name << " table loaded successfully." << std::endl;
    stats.entries.push_back(stat_entry);

    return next_xtn_id;
}

/* ------------------------------------------------------------------ */
/*  Table dispatcher (multi-file)                                     */
/* ------------------------------------------------------------------ */
uint64_t loader_load_multiple_files_table_(SSB::common::Table table,
    LoaderOptions &options, SSBTableMetadata &metadata, int32_t scale_factor, uint64_t start_xtn_id,
    LoaderStats &stats)
{
    if (options.enable_column) {
    } else {
        fprintf(stderr, "Rowstore loading is not implemented.\n");
        exit(EXIT_FAILURE);
    }

    switch (table)
    {
    case SSB::common::Table::CUSTOMER:
            return loader_load_multiple_files_table_column(
                table,
                SSB::common::fmt.customer_field_types,
                SSB::common::fmt.customer_field_sizes,
                SSB::common::fmt.customer_fields,
                SSB::common::fmt.customer_varchar_field_indexes,
                SSB::common::fmt.customer_filter_columns,
                SSB::common::fmt.customer_enable_stats_columns,
                SSB::common::fmt.customer_field_compression_types,
                SSB::common::fmt.customer_csv_column_map,
                options, metadata, start_xtn_id, stats);
    case SSB::common::Table::LINEORDER:
            return loader_load_multiple_files_table_column(
                table,
                SSB::common::fmt.lineorder_field_types,
                SSB::common::fmt.lineorder_field_sizes,
                SSB::common::fmt.lineorder_fields,
                SSB::common::fmt.lineorder_varchar_field_indexes,
                SSB::common::fmt.lineorder_filter_columns,
                SSB::common::fmt.lineorder_enable_stats_columns,
                SSB::common::fmt.lineorder_field_compression_types,
                SSB::common::fmt.lineorder_csv_column_map,
                options, metadata, start_xtn_id, stats);
    case SSB::common::Table::SUPPLIER:
            return loader_load_multiple_files_table_column(
                table,
                SSB::common::fmt.supplier_field_types,
                SSB::common::fmt.supplier_field_sizes,
                SSB::common::fmt.supplier_fields,
                SSB::common::fmt.supplier_varchar_field_indexes,
                SSB::common::fmt.supplier_filter_columns,
                SSB::common::fmt.supplier_enable_stats_columns,
                SSB::common::fmt.supplier_field_compression_types,
                SSB::common::fmt.supplier_csv_column_map,
                options, metadata, start_xtn_id, stats);
    case SSB::common::Table::PART:
            return loader_load_multiple_files_table_column(
                table,
                SSB::common::fmt.part_field_types,
                SSB::common::fmt.part_field_sizes,
                SSB::common::fmt.part_fields,
                SSB::common::fmt.part_varchar_field_indexes,
                SSB::common::fmt.part_filter_columns,
                SSB::common::fmt.part_enable_stats_columns,
                SSB::common::fmt.part_field_compression_types,
                SSB::common::fmt.part_csv_column_map,
                options, metadata, start_xtn_id, stats);
    case SSB::common::Table::DDATE:
            return loader_load_multiple_files_table_column(
                table,
                SSB::common::fmt.date_field_types,
                SSB::common::fmt.date_field_sizes,
                SSB::common::fmt.date_fields,
                SSB::common::fmt.date_varchar_field_indexes,
                SSB::common::fmt.date_filter_columns,
                SSB::common::fmt.date_enable_stats_columns,
                SSB::common::fmt.date_field_compression_types,
                SSB::common::fmt.date_csv_column_map,
                options, metadata, start_xtn_id, stats);
    default:
        break;
    }
    return start_xtn_id;
}

/* ------------------------------------------------------------------ */
/*  Per-table convenience functions                                   */
/* ------------------------------------------------------------------ */
uint64_t loader_load_date(LoaderOptions &options, SSBTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_multiple_files_table_(
        SSB::common::Table::DDATE, options, metadata, 0, start_page_id, stats);
}

uint64_t loader_load_customer(LoaderOptions &options, SSBTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_multiple_files_table_(
        SSB::common::Table::CUSTOMER, options, metadata, 0, start_page_id, stats);
}

uint64_t loader_load_supplier(LoaderOptions &options, SSBTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_multiple_files_table_(
        SSB::common::Table::SUPPLIER, options, metadata, 0, start_page_id, stats);
}

uint64_t loader_load_part(LoaderOptions &options, SSBTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_multiple_files_table_(
        SSB::common::Table::PART, options, metadata, 0, start_page_id, stats);
}

uint64_t loader_load_lineorder(LoaderOptions &options, SSBTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_multiple_files_table_(
        SSB::common::Table::LINEORDER, options, metadata, 0, start_page_id, stats);
}

/* ------------------------------------------------------------------ */
/*  Metadata management                                               */
/* ------------------------------------------------------------------ */
static SSBTableMetadata* metadata_init(const LoaderOptions &options)
{
    // 512B-aligned alloc
    void *ptr;
    SSBTableMetadata *metadata;

    superpage_set_constants_for(options.page_size, sizeof(SSBTableMetadata));

    if (posix_memalign((void**)&ptr, 512, options.page_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Call constructor explicitly to initialize the object correctly
    metadata = new(ptr) SSBTableMetadata();
    if (sizeof(SSBTableMetadata) > options.page_size)
    {
        std::cerr << "SSBTableMetadata size (" << sizeof(SSBTableMetadata)
            << ") is larger than page size (" << options.page_size << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    metadata->table_date_nrows = 0;
    for (size_t i = 0; i < SSB::common::kDateFieldCount; i++)
    {
        metadata->table_date_start_page_ids[i] = 0;
        metadata->table_date_max_nrows_in_page[i] = 0;
    }
    metadata->table_customer_nrows = 0;
    for (size_t i = 0; i < SSB::common::kCustomerFieldCount; i++)
    {
        metadata->table_customer_start_page_ids[i] = 0;
        metadata->table_customer_max_nrows_in_page[i] = 0;
    }
    metadata->table_supplier_nrows = 0;
    for (size_t i = 0; i < SSB::common::kSupplierFieldCount; i++)
    {
        metadata->table_supplier_start_page_ids[i] = 0;
        metadata->table_supplier_max_nrows_in_page[i] = 0;
    }
    metadata->table_part_nrows = 0;
    for (size_t i = 0; i < SSB::common::kPartFieldCount; i++)
    {
        metadata->table_part_start_page_ids[i] = 0;
        metadata->table_part_max_nrows_in_page[i] = 0;
    }
    metadata->table_lineorder_nrows = 0;
    for (size_t i = 0; i < SSB::common::kLineOrderFieldCount; i++)
    {
        metadata->table_lineorder_start_page_ids[i] = 0;
        metadata->table_lineorder_max_nrows_in_page[i] = 0;
    }
    return metadata;
}

static void metadata_free(SSBTableMetadata *metadata)
{
    metadata->~SSBTableMetadata();
    if (metadata != nullptr)
    {
        free(metadata);
    }
}

void metadata_print(const LoaderOptions options, SSBTableMetadata &metadata)
{
    if (options.load_ddate)
    {
        std::cout << "table_date_nrows: " << metadata.table_date_nrows << std::endl;
        for (size_t i = 0; i < SSB::common::kDateFieldCount; i++)
        {
            std::cout << "\ttable_date_start_page_ids[" << i << "]: "
                << metadata.table_date_start_page_ids[i] << std::endl;
            std::cout << "\ttable_date_npages[" << i << "]: "
                << metadata.table_date_npages[i] << std::endl;
        }
    }
    if (options.load_customer)
    {
        std::cout << "table_customer_nrows: " << metadata.table_customer_nrows << std::endl;
        for (size_t i = 0; i < SSB::common::kCustomerFieldCount; i++)
        {
            std::cout << "\ttable_customer_start_page_ids[" << i << "]: "
                << metadata.table_customer_start_page_ids[i] << std::endl;
            std::cout << "\ttable_customer_npages[" << i << "]: "
                << metadata.table_customer_npages[i] << std::endl;
        }
    }
    if (options.load_supplier)
    {
        std::cout << "table_supplier_nrows: " << metadata.table_supplier_nrows << std::endl;
        for (size_t i = 0; i < SSB::common::kSupplierFieldCount; i++)
        {
            std::cout << "\ttable_supplier_start_page_ids[" << i << "]: "
                << metadata.table_supplier_start_page_ids[i] << std::endl;
            std::cout << "\ttable_supplier_npages[" << i << "]: "
                << metadata.table_supplier_npages[i] << std::endl;
        }
    }
    if (options.load_part)
    {
        std::cout << "table_part_nrows: " << metadata.table_part_nrows << std::endl;
        for (size_t i = 0; i < SSB::common::kPartFieldCount; i++)
        {
            std::cout << "\ttable_part_start_page_ids[" << i << "]: "
                << metadata.table_part_start_page_ids[i] << std::endl;
            std::cout << "\ttable_part_npages[" << i << "]: "
                << metadata.table_part_npages[i] << std::endl;
        }
    }
    if (options.load_lineorder)
    {
        std::cout << "table_lineorder_nrows: " << metadata.table_lineorder_nrows << std::endl;
        for (size_t i = 0; i < SSB::common::kLineOrderFieldCount; i++)
        {
            std::cout << "\ttable_lineorder_start_page_ids[" << i << "]: "
                << metadata.table_lineorder_start_page_ids[i] << std::endl;
            std::cout << "\ttable_lineorder_npages[" << i << "]: "
                << metadata.table_lineorder_npages[i] << std::endl;
        }
    }
}

void metadata_verify(LoaderOptions &options, SSBTableMetadata &metadata_)
{
    if (options.load_ddate) {
        if (metadata_.table_date_nrows == 0)
        {
            std::cerr << "[FATAL] table_date_nrows is 0" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    if (options.load_customer) {
        if (metadata_.table_customer_nrows == 0)
        {
            std::cerr << "[FATAL] table_customer_nrows is 0" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    if (options.load_supplier) {
        if (metadata_.table_supplier_nrows == 0)
        {
            std::cerr << "[FATAL] table_supplier_nrows is 0" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    if (options.load_part) {
        if (metadata_.table_part_nrows == 0)
        {
            std::cerr << "[FATAL] table_part_nrows is 0" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    if (options.load_lineorder) {
        if (metadata_.table_lineorder_nrows == 0)
        {
            std::cerr << "[FATAL] table_lineorder_nrows is 0" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    #if 1
    if (options.verify) {
        /* verify metadata from storage */
        void *ptr;
        size_t super_page_id = superpage_get_base_super_page_id();
        size_t num_super_pages = superpage_get_super_npage();
        std::cout << "super_page_id: " << super_page_id << std::endl;
        std::cout << "num_super_pages: " << num_super_pages << std::endl;
        if (posix_memalign((void**)&ptr, 512, options.page_size * num_super_pages) != 0)
        {
            std::cerr << "posix_memalign failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        char *ptr_base = reinterpret_cast<char*>(ptr);
        page_pread_host(options.output_fds, (void*)ptr_base, 0, num_super_pages * options.page_size);
        SSBTableMetadata *metadata = reinterpret_cast<SSBTableMetadata*>(ptr_base);

        if (options.load_ddate) {
            std::cout << "Date table nrows: " << metadata->table_date_nrows << std::endl;
            for (size_t i = 0; i < SSB::common::kDateFieldCount; i++)
            {
                std::cout << "\ttable_date_start_page_ids[" << i << "]: "
                    << metadata->table_date_start_page_ids[i] << std::endl;
            }
        }
        if (options.load_customer) {
            std::cout << "Customer table nrows: " << metadata->table_customer_nrows << std::endl;
            for (size_t i = 0; i < SSB::common::kCustomerFieldCount; i++)
            {
                std::cout << "\ttable_customer_start_page_ids[" << i << "]: "
                    << metadata->table_customer_start_page_ids[i] << std::endl;
            }
        }
        if (options.load_supplier) {
            std::cout << "Supplier table nrows: " << metadata->table_supplier_nrows << std::endl;
            for (size_t i = 0; i < SSB::common::kSupplierFieldCount; i++)
            {
                std::cout << "\ttable_supplier_start_page_ids[" << i << "]: "
                    << metadata->table_supplier_start_page_ids[i] << std::endl;
            }
        }
        if (options.load_part) {
            std::cout << "Part table nrows: " << metadata->table_part_nrows << std::endl;
            for (size_t i = 0; i < SSB::common::kPartFieldCount; i++)
            {
                std::cout << "\ttable_part_start_page_ids[" << i << "]: "
                    << metadata->table_part_start_page_ids[i] << std::endl;
            }
        }
        if (options.load_lineorder) {
            std::cout << "Lineorder table nrows: " << metadata->table_lineorder_nrows << std::endl;
            for (size_t i = 0; i < SSB::common::kLineOrderFieldCount; i++)
            {
                std::cout << "\ttable_lineorder_start_page_ids[" << i << "]: "
                    << metadata->table_lineorder_start_page_ids[i] << std::endl;
            }
        }
        free(ptr);
    }
    #endif

    std::cout << "metadata verified successfully." << std::endl;
}

void metadata_sync(LoaderOptions &options, SSBTableMetadata &metadata, uint64_t next_free_page_id)
{
    uint64_t page_size = options.page_size;
    uint16_t lbc_num_varchar_clusters = options.lbc_num_varchar_clusters;
    uint32_t compressed = options.compress;
    uint32_t column = options.enable_column;

    metadata.page_size = page_size;
    metadata.lbc_num_varchar_clusters = lbc_num_varchar_clusters;
    metadata.compressed = compressed;
    metadata.column = column;
    metadata.free_page_id = next_free_page_id;

    std::cout << "page_size: " << page_size << std::endl;
    std::cout << "compressed: " << compressed << std::endl;
    std::cout << "free_page_id: " << next_free_page_id << std::endl;

    if (options.dryrun) {
        std::cout << "[DRYRUN] syncing metadata..." << std::endl;
        std::cout << "[DRYRUN] metadata synced successfully." << std::endl;
    } else {
        std::cout << "Syncing metadata..." << std::endl;
        std::cout << "\tsizeof(SSBTableMetadata): " << sizeof(SSBTableMetadata) << std::endl;
        size_t npages = superpage_get_super_npage();
        std::cout << "\tnpages for superpage: " << npages << std::endl;

        const size_t page_size = options.page_size;
        for (size_t i = 0; i < npages; i++)
        {
            std::cout << "\t[INFO] superpage: page id " << i << " write at page id " << i << std::endl;
            page_pwrite_host(options.output_fds, (void*)&metadata, i, page_size);
        }
        std::cout << "\tcheck: Lineorder table nrows in metadata: " << metadata.table_lineorder_nrows << std::endl;
        std::cout << "Synced metadata successfully." << std::endl;
    }
}

/* ------------------------------------------------------------------ */
/*  main()                                                            */
/* ------------------------------------------------------------------ */
int main(int argc, char *const *argv)
{
    std::cerr << "ssbloader" << std::endl;

    LoaderOptions options = ssb_parse_loader_options(argc, argv);

    validate_loader_options(options);

    if (options.verbose) {
        printf("options.input_dirname: %s\n", options.input_dirname);
        printf("options.output_files: %s\n", options.output_files);
        printf("options.compress: %d\n", options.compress);
    }

    open_output_files(options, options.output_fds);

    struct LoaderStats stats;
    struct SSBTableMetadata* metadata_ptr = metadata_init(options);
    auto& metadata = *metadata_ptr;

    if (sizeof(SSBTableMetadata) >= options.page_size)
    {
        std::cerr << "[FATAL] Unexpected: SSBTableMetadata size is larger than page size" << std::endl;
        return -1;
    }

    // Page IDs after superpage (metadata pages)
    uint64_t next_page_id = superpage_get_base_page_id();

    /*
     * Loading order: DATE → CUSTOMER → SUPPLIER → PART → LINEORDER
     * Dimension tables are loaded first so that lookup tables
     * can be built for sideways stats during LINEORDER loading.
     */
    if (options.load_ddate)
    {
        next_page_id = loader_load_date(options, metadata, next_page_id, stats);
    }
    if (options.load_customer)
    {
        next_page_id = loader_load_customer(options, metadata, next_page_id, stats);
    }
    if (options.load_supplier)
    {
        next_page_id = loader_load_supplier(options, metadata, next_page_id, stats);
    }
    if (options.load_part)
    {
        next_page_id = loader_load_part(options, metadata, next_page_id, stats);
    }
    if (options.load_lineorder)
    {
        next_page_id = loader_load_lineorder(options, metadata, next_page_id, stats);
    }

    metadata_print(options, metadata);
    metadata_sync(options, metadata, next_page_id);
    metadata_verify(options, metadata);
    metadata_free(metadata_ptr);

    if (options.verify) {
        std::cout << "Start scan test" << std::endl;
        if (options.enable_column) {
            test_scan_column_tables(options);
        } else {
            fprintf(stderr, "Rowstore scan test is not implemented.\n");
            exit(EXIT_FAILURE);
        }
        std::cout << "End scan test" << std::endl;
    }

    close_output_files(options, options.output_fds);

    loader_stats_print(options.page_size, options.compress, stats);
}
