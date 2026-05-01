/**
 * TPCH Loader Main for column store
 * Tsuyoshi Ozawa <ozawa@tkl.iis.u-tokyo.ac.jp>
 */
// #include "./bench_device.cu"
// #include "./bench_device_async.cu"
// #include "./bench_device_batch.cu"
// #include "./bench_device_batch_sync.cu"
// #include "./bench_host.cu"
// #include "./bench_host_async.cu"
// #include "./bench_host_async_device.cu"
// #include "./bench_host_device.cu"
// #include "./bench_host_device_async.cu"
// #include "./bench_transfer.cu"
// #include "./bench_transfer_async.cu"
#include "./tpch_loader_cli.cu"

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
    LoaderOptions &options, TPCHTableMetadata &metadata,
    uint64_t start_xtn_id, LoaderStats &stats)
{
    constexpr size_t MaxNumBuffers = NFields + NumVarCharFields;
    std::string s = std::string(TPCH::common::table_name(table));
    std::transform(s.begin(), s.end(), s.begin(), [](char c) { return std::toupper(c); });
    auto table_name = s;
    StatEntry stat_entry;
    stat_entry.name = table_name;
     
    // std::string table_name = TPCH::common::table_name_upper(table)
    // Load the ddate table
    std::cout << "Loading " << table_name << " table..." << std::endl;

    /* target columns */
    std::array<char*, MaxNumBuffers> buffers;

    /* buffer and per-field pagids */
    std::string col = TPCH::common::table_name(table);
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
    for (int i = 0; i < NFields; i++)
    {
        buffers[i] = &buf[n * options.page_size * i];
    }
    if (options.enable_dict_encoding) {
        for (int i = NFields; i < MaxNumBuffers; i++) {
            buffers[i] = &buf[options.page_size * i];
        }
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
        stat_entry);

    free(buf);
    std::cout << table_name << " table loaded successfully." << std::endl;
    stats.entries.push_back(stat_entry);

    return next_xtn_id;
}

uint64_t loader_load_multiple_files_table_(TPCH::common::Table table,
    LoaderOptions &options, TPCHTableMetadata &metadata, int32_t scale_factor, uint64_t start_xtn_id,
    LoaderStats &stats)
{
    if (options.enable_column) {
    } else {
        fprintf(stderr, "Rowstore loading for CUSTOMER table is not implemented.\n");
        exit(EXIT_FAILURE);
    }

    switch (table)
    {
    case TPCH::common::Table::CUSTOMER:
            return loader_load_multiple_files_table_column(
                table,
                TPCH::common::fmt.customer_field_types,
                TPCH::common::fmt.customer_field_sizes,
                TPCH::common::fmt.customer_fields,
                TPCH::common::fmt.customer_varchar_field_indexes,
                TPCH::common::fmt.customer_filter_columns,
                TPCH::common::fmt.customer_enable_stats_columns,
                TPCH::common::fmt.customer_field_compression_types,
                options, metadata, start_xtn_id, stats);
    case TPCH::common::Table::LINEITEM:
            return loader_load_multiple_files_table_column(
                table,
                TPCH::common::fmt.lineitem_field_types,
                TPCH::common::fmt.lineitem_field_sizes,
                TPCH::common::fmt.lineitem_fields,
                TPCH::common::fmt.lineitem_varchar_field_indexes,
                TPCH::common::fmt.lineitem_filter_columns,
                TPCH::common::fmt.lineitem_enable_stats_columns,
                TPCH::common::fmt.lineitem_field_compression_types,
                options, metadata, start_xtn_id, stats);
    case TPCH::common::Table::ORDERS:
            return loader_load_multiple_files_table_column(
                table,
                TPCH::common::fmt.orders_field_types,
                TPCH::common::fmt.orders_field_sizes,
                TPCH::common::fmt.orders_fields,
                TPCH::common::fmt.orders_varchar_field_indexes,
                TPCH::common::fmt.orders_filter_columns,
                TPCH::common::fmt.orders_enable_stats_columns,
                TPCH::common::fmt.orders_field_compression_types,
                options, metadata, start_xtn_id, stats);
    case TPCH::common::Table::SUPPLIER:
            return loader_load_multiple_files_table_column(
                table,
                TPCH::common::fmt.supplier_field_types,
                TPCH::common::fmt.supplier_field_sizes,
                TPCH::common::fmt.supplier_fields,
                TPCH::common::fmt.supplier_varchar_field_indexes,
                TPCH::common::fmt.supplier_filter_columns,
                TPCH::common::fmt.supplier_enable_stats_columns,
                TPCH::common::fmt.supplier_field_compression_types,
                options, metadata, start_xtn_id, stats);
    case TPCH::common::Table::PART:
            return loader_load_multiple_files_table_column(
                table,
                TPCH::common::fmt.part_field_types,
                TPCH::common::fmt.part_field_sizes,
                TPCH::common::fmt.part_fields,
                TPCH::common::fmt.part_varchar_field_indexes,
                TPCH::common::fmt.part_filter_columns,
                TPCH::common::fmt.part_enable_stats_columns,
                TPCH::common::fmt.part_field_compression_types,
                options, metadata, start_xtn_id, stats);
    case TPCH::common::Table::PARTSUPP:
            return loader_load_multiple_files_table_column(
                table,
                TPCH::common::fmt.partsupp_field_types,
                TPCH::common::fmt.partsupp_field_sizes,
                TPCH::common::fmt.partsupp_fields,
                TPCH::common::fmt.partsupp_varchar_field_indexes,
                TPCH::common::fmt.partsupp_filter_columns,
                TPCH::common::fmt.partsupp_enable_stats_columns,
                TPCH::common::fmt.partsupp_field_compression_types,
                options, metadata, start_xtn_id, stats);
    default:
        break;
    }
    return start_xtn_id;
}

template <size_t NumVarCharFields, size_t NFields,
    typename BulkLoadTargetField, typename TableType>
uint64_t loader_load_single_file_table_column(
    TableType table,
    const std::array<enum rec_type, NFields> &field_types,
    const std::array<size_t, NFields>& field_sizes,
    const std::array<enum rec_type, NFields> &field_encoded_types,
    const std::array<size_t, NFields>& field_encoded_sizes,
    const std::array<size_t, NFields>& arr_column_sizes,
    const std::array<BulkLoadTargetField, NFields>& target_fields,
    const std::array<size_t, NumVarCharFields>& target_varchar_field_indexes,
    const std::array<CompressionMethod, NFields>& compression_types,
    LoaderOptions &options, TPCHTableMetadata &metadata,
    uint64_t start_page_id, LoaderStats &stats)
{
    constexpr size_t MaxNumBuffers = NFields + NumVarCharFields;
    std::string s = std::string(TPCH::common::table_name(table));
    std::transform(s.begin(), s.end(), s.begin(), [](char c) { return std::toupper(c); });
    auto table_name = s;
    StatEntry stat_entry;
    stat_entry.name = table_name;
     
    // std::string table_name = TPCH::common::table_name_upper(table)
    // Load the ddate table
    std::cout << "Loading " << table_name << " table..." << std::endl;

    /* target columns */
    // auto& field_sizes = field_sizes;
    // auto& fields = fields;
    std::array<uint64_t, MaxNumBuffers> start_page_ids;
    std::array<char*, MaxNumBuffers> buffers;

    // This should be calculated later.
    // const uint64_t num_rows = TPCH::loader_get_table_num_rows(table, scale_factor);
    // metadata_set_nrows(metadata, table, num_rows);
    // uint64_t field_start_page_id = start_page_id;
    std::array<std::vector<size_t>, NumVarCharFields> varchar_cluster_thresholds{};

    /* buffer and per-field pagids */
    char *buf = loader_aligned_alloc<char*>(MaxNumBuffers * options.page_size);
    for (int i = 0; i < NFields; i++)
    {
        /* setup metrics */
        buffers[i] = &buf[options.page_size * i];
    }

    if (options.enable_dict_encoding) {
        for (int i = NFields; i < MaxNumBuffers; i++) {
            buffers[i] = &buf[options.page_size * i];
        }
    }

    uint64_t next_page_id = bulkload_single_file_column<BulkLoadTargetField>(
        options, metadata, table,
        start_page_id,
        buffers,
        field_sizes, field_types,
        field_encoded_sizes, field_encoded_types,
        arr_column_sizes,
        target_fields, target_varchar_field_indexes,
        varchar_cluster_thresholds,
        compression_types,
        stat_entry);

    free(buf);
    std::cout << table_name << " table loaded successfully." << std::endl;
    stats.entries.push_back(stat_entry);

    return next_page_id;
}

uint64_t loader_load_single_file_table_(TPCH::common::Table table,
    LoaderOptions &options, TPCHTableMetadata &metadata, int32_t scale_factor, uint64_t start_xtn_id,
    LoaderStats &stats)
{
    if (options.enable_column) {
        switch (table)
        {
        case TPCH::common::Table::NATION:
            return loader_load_single_file_table_column(
                table,
                TPCH::common::fmt.nation_field_types,
                TPCH::common::fmt.nation_field_sizes,
                TPCH::common::fmt.nation_field_dict_encoded_types,
                TPCH::common::fmt.nation_field_dict_encoded_sizes,
                TPCH::common::fmt.nation_field_sizes,
                TPCH::common::fmt.nation_fields,
                TPCH::common::fmt.nation_varchar_field_indexes,
                TPCH::common::fmt.nation_field_compression_types,
                options, metadata, start_xtn_id, stats);
            break;
        case TPCH::common::Table::REGION:
            return loader_load_single_file_table_column(
                table,
                TPCH::common::fmt.region_field_types,
                TPCH::common::fmt.region_field_sizes,
                TPCH::common::fmt.region_field_dict_encoded_types,
                TPCH::common::fmt.region_field_dict_encoded_sizes,
                TPCH::common::fmt.region_field_sizes,
                TPCH::common::fmt.region_fields,
                TPCH::common::fmt.region_varchar_field_indexes,
                TPCH::common::fmt.region_field_compression_types,
                options, metadata, start_xtn_id, stats);
            break;
        default:
            break;
        }
    } else {
    }
    return start_xtn_id;
}

uint64_t loader_load_nation(LoaderOptions &options, TPCHTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_single_file_table_(
        TPCH::common::Table::NATION, options, metadata, 0, start_page_id, stats);
}

uint64_t loader_load_region(LoaderOptions &options, TPCHTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_single_file_table_(
        TPCH::common::Table::REGION, options, metadata, 0, start_page_id, stats);
}

uint64_t loader_load_customer(LoaderOptions &options, TPCHTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_multiple_files_table_(
        TPCH::common::Table::CUSTOMER, options, metadata, 0, start_page_id, stats);
}

uint64_t loader_load_lineitem(LoaderOptions &options, TPCHTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_multiple_files_table_(
        TPCH::common::Table::LINEITEM, options, metadata, 0, start_page_id, stats);
}

uint64_t loader_load_orders(LoaderOptions &options, TPCHTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_multiple_files_table_(
        TPCH::common::Table::ORDERS, options, metadata, 0, start_page_id, stats);
}

uint64_t loader_load_supplier(LoaderOptions &options, TPCHTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    uint64_t next_page_id = start_page_id;

    if (options.lbc_enabled) {
        options._lbc_num_varchar_clusters_original = options.lbc_num_varchar_clusters;
        options.lbc_num_varchar_clusters = std::min(32UL, options.lbc_num_varchar_clusters);
    }
    next_page_id = loader_load_multiple_files_table_(
        TPCH::common::Table::SUPPLIER, options, metadata, 0, start_page_id, stats);
    if (options.lbc_enabled) {
        options.lbc_num_varchar_clusters = options._lbc_num_varchar_clusters_original;
    }

    return next_page_id;
}

uint64_t loader_load_part(LoaderOptions &options, TPCHTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    uint64_t next_page_id = start_page_id;

    if (options.lbc_enabled) {
        options._lbc_num_varchar_clusters_original = options.lbc_num_varchar_clusters;
        options.lbc_num_varchar_clusters = std::min(4UL, options.lbc_num_varchar_clusters);
    }
    next_page_id = loader_load_multiple_files_table_(
        TPCH::common::Table::PART, options, metadata, 0, start_page_id, stats);

    if (options.lbc_enabled) {
        options.lbc_num_varchar_clusters = options._lbc_num_varchar_clusters_original;
    }

    return next_page_id;
}

uint64_t loader_load_partsupp(LoaderOptions &options, TPCHTableMetadata &metadata, uint64_t start_page_id,
    LoaderStats &stats)
{
    return loader_load_multiple_files_table_(
        TPCH::common::Table::PARTSUPP, options, metadata, 0, start_page_id, stats);
}

static TPCHTableMetadata* metadata_init(const LoaderOptions &options)
{
    // 512B-aligned alloc
    void *ptr;
    TPCHTableMetadata *metadata;

    superpage_set_constants(options.page_size);

    if (posix_memalign((void**)&ptr, 512, options.page_size) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Call constructor explicitly to initialize the object correctly
    metadata = new(ptr) TPCHTableMetadata();
    if (sizeof(TPCHTableMetadata) > options.page_size)
    {
        std::cerr << "TPCHTableMetadata size is larger than page size" << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t num_super_pages = superpage_get_super_npage();
#if 0
    if (posix_memalign((void**)&ptr, 512,
        options.page_size * num_super_pages) != 0)
    {
        std::cerr << "posix_memalign failed" << std::endl;
        exit(EXIT_FAILURE);
    }
#endif


    metadata->table_customer_nrows = 0;
    for (int i = 0; i < TPCH::common::kCustomerFieldCount; i++)
    {
        metadata->table_customer_start_page_ids[i] = 0;
        metadata->table_customer_max_nrows_in_page[i] = 0;
    }
    metadata->table_lineitem_nrows = 0;
    for (int i = 0; i < TPCH::common::kLineitemFieldCount; i++)
    {
        metadata->table_lineitem_start_page_ids[i] = 0;
    }
    metadata->table_orders_nrows = 0;
    for (int i = 0; i < TPCH::common::kOrdersFieldCount; i++)
    {
        metadata->table_orders_start_page_ids[i] = 0;
    }
    metadata->table_part_nrows = 0;
    for (int i = 0; i < TPCH::common::kPartFieldCount; i++)
    {
        metadata->table_part_start_page_ids[i] = 0;
    }
    metadata->table_supplier_nrows = 0;
    for (int i = 0; i < TPCH::common::kSupplierFieldCount; i++)
    {
        metadata->table_supplier_start_page_ids[i] = 0;
    }
    metadata->table_partsupp_nrows = 0;
    for (int i = 0; i < TPCH::common::kPartSuppFieldCount; i++)
    {
        metadata->table_partsupp_start_page_ids[i] = 0;
    }
    metadata->table_nation_nrows = 0;
    for (int i = 0; i < TPCH::common::kNationFieldCount; i++)
    {
        metadata->table_nation_start_page_ids[i] = 0;
    }
    metadata->table_region_nrows = 0;
    for (int i = 0; i < TPCH::common::kRegionFieldCount; i++)
    {
        metadata->table_region_start_page_ids[i] = 0;
    }
    return metadata;
}

static void metadata_free(TPCHTableMetadata *metadata)
{
    // Call deconstructor explicitly to initialize the object correctly
    metadata->~TPCHTableMetadata();
    if (metadata != nullptr)
    {
        free(metadata);
    }
}

void metadata_print(const LoaderOptions options, TPCHTableMetadata &metadata)
{
    if (options.load_nation)
    {
        std::cout << "table_nation_nrows: " << metadata.table_nation_nrows << std::endl;
        size_t i = 0;
        std::cout << "table_nation_start_page_ids[" << i << "]: " << metadata.table_nation_start_page_ids[i] << std::endl;
        std::cout << "table_nation_npages[" << i << "]: " << metadata.table_nation_npages[i] << std::endl;
    }
    if (options.load_region)
    {
        std::cout << "table_region_nrows: " << metadata.table_region_nrows << std::endl;
        size_t i = 0;
        std::cout << "table_region_start_page_ids[" << i << "]: " << metadata.table_region_start_page_ids[i] << std::endl;
        std::cout << "table_region_npages[" << i << "]: " << metadata.table_region_npages[i] << std::endl;
    }
    if (options.load_customer)
    {
        std::cout << "table_customer_nrows: " << metadata.table_customer_nrows << std::endl;
        size_t i = 0;
        std::cout << "table_customer_start_page_ids[" << i << "]: " << metadata.table_customer_start_page_ids[i] << std::endl;
        std::cout << "table_customer_npages[" << i << "]: " << metadata.table_customer_npages[i] << std::endl;
    }
    if (options.load_orders)
    {
        std::cout << "table_orders_nrows: " << metadata.table_orders_nrows << std::endl;
        size_t i = 0;
        std::cout << "table_orders_start_page_ids[" << i << "]: " << metadata.table_orders_start_page_ids[i] << std::endl;
        std::cout << "table_orders_npages[" << i << "]: " << metadata.table_orders_npages[i] << std::endl;
    }
    if (options.load_lineitem)
    {
        std::cout << "table_lineitem_nrows: " << metadata.table_lineitem_nrows << std::endl;
        size_t i = 0;
        std::cout << "table_lineitem_start_page_ids[" << i << "]: " << metadata.table_lineitem_start_page_ids[i] << std::endl;
        std::cout << "table_lineitem_npages[" << i << "]: " << metadata.table_lineitem_npages[i] << std::endl;
    }
    if (options.load_part)
    {
        std::cout << "table_part_nrows: " << metadata.table_part_nrows << std::endl;
        size_t i = 0;
        std::cout << "table_part_start_page_ids[" << i << "]: " << metadata.table_part_start_page_ids[i] << std::endl;
        std::cout << "table_part_npages[" << i << "]: " << metadata.table_part_npages[i] << std::endl;
    }
    if (options.load_supplier)
    {
        std::cout << "table_supplier_nrows: " << metadata.table_supplier_nrows << std::endl;
        size_t i = 0;
        std::cout << "table_supplier_start_page_ids[" << i << "]: " << metadata.table_supplier_start_page_ids[i] << std::endl;
        std::cout << "table_supplier_npages[" << i << "]: " << metadata.table_supplier_npages[i] << std::endl;
    }
    if (options.load_partsupp)
    {
        std::cout << "table_partsupp_nrows: " << metadata.table_partsupp_nrows << std::endl;
        size_t i = 0;
        std::cout << "table_partsupp_start_page_ids[" << i << "]: " << metadata.table_partsupp_start_page_ids[i] << std::endl;
        std::cout << "table_partsupp_npages[" << i << "]: " << metadata.table_partsupp_npages[i] << std::endl;
    }

}

void metadata_verify(LoaderOptions &options, TPCHTableMetadata &metadata_)
{
    // int scale_factor = options.scale_factor;

    const size_t nrows_nation = 25;
    if (options.load_nation) {
        if (metadata_.table_nation_nrows == 0)
        {
            std::cerr << "[FATAL] table_nation_nrows is 0" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (metadata_.table_nation_nrows != nrows_nation)
        {
            std::cerr << "[FATAL] table_nation_nrows is not equal to " << nrows_nation
                << "(actual: " << metadata_.table_nation_nrows << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
 
    const size_t nrows_region = 5;
    if (options.load_region) {
        if (metadata_.table_region_nrows == 0)
        {
            std::cerr << "[FATAL] table_region_nrows is 0" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (metadata_.table_region_nrows != nrows_region)
        {
            std::cerr << "[FATAL] table_region_nrows is not equal to " << nrows_region
                << "(actual: " << metadata_.table_region_nrows << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
 
    #if 0
    const size_t nrows_customer_sf1 = 150000;
    if (options.load_customer) {
        if (metadata.table_customer_nrows == 0)
        {
            std::cerr << "[FATAL] table_customer_nrows is 0" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (metadata.table_customer_nrows % nrows_customer_sf1 != 0)
        {
            std::cerr << "[FATAL] table_customer_nrows is not divied without modulo "
                << nrows_customer_sf1 
                << "(actual: " << metadata.table_customer_nrows % nrows_customer_sf1  << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "SF should be: " <<  metadata.table_customer_nrows / nrows_customer_sf1  << std::endl;
    }
    #endif

    #if 1
    if (options.verify) {
        /* verify XTN head */
        {
            void *ptr;
            size_t super_page_id = superpage_get_base_super_page_id();
            const uint64_t base_page_id = super_page_id;
            size_t num_super_pages = superpage_get_super_npage();
            std::cout << "super_page_id: " << super_page_id << std::endl;
            std::cout << "num_super_pages: " << num_super_pages << std::endl;
            if (posix_memalign((void**)&ptr, 512, options.page_size * num_super_pages) != 0)
            {
                std::cerr << "posix_memalign failed" << std::endl;
                exit(EXIT_FAILURE);
            }
            char *ptr_base = reinterpret_cast<char*>(ptr);
            //for (size_t i = 0; i < super_npage; ++i) {
            //}
            page_pread_host(options.output_fds, (void*)ptr_base, 0, num_super_pages * options.page_size);

            TPCHTableMetadata *metadata = reinterpret_cast<TPCHTableMetadata*>(ptr_base);
            //std::cout << metadata->table_lineitem_max_nrows_in_page[0] << std::endl;
            //std::cout << metadata->free_page_id << std::endl;

            if (options.load_customer)
            {
                std::cout << "Customer table nrows: " << metadata->table_customer_nrows << std::endl;
                for (int i = 0; i < TPCH::common::kCustomerFieldCount; i++)
                {
                    std::cout << "\ttable_customer_start_page_ids[" << i << "]: "
                        << metadata->table_customer_start_page_ids[i] << std::endl;
                    std::cout << "\ttable_customer_max_nrows_in_page[" << i << "]: "
                        << metadata->table_customer_max_nrows_in_page[i] << std::endl;
                }
            }
            if (options.load_lineitem)
            {
                std::cout << "Lineitem table nrows: " << metadata->table_lineitem_nrows << std::endl;
                for (int i = 0; i < TPCH::common::kLineitemFieldCount; i++)
                {
                    std::cout << "\ttable_lineitem_start_page_ids[" << i << "]: "
                        << metadata->table_lineitem_start_page_ids[i] << std::endl;
                    std::cout << "\ttable_lineitem_max_nrows_in_page[" << i << "]: "
                        << metadata->table_lineitem_max_nrows_in_page[i] << std::endl;
                }
            }
            if (options.load_orders)
            {
                std::cout << "Orders table nrows: " << metadata->table_orders_nrows << std::endl;
                for (int i = 0; i < TPCH::common::kOrdersFieldCount; i++)
                {
                    std::cout << "\ttable_orders_start_page_ids[" << i << "]: "
                        << metadata->table_orders_start_page_ids[i] << std::endl;
                    std::cout << "\ttable_orders_max_nrows_in_page[" << i << "]: "
                        << metadata->table_orders_max_nrows_in_page[i] << std::endl;
                }
            }
            if (options.load_part)
            {
                std::cout << "Part table nrows: " << metadata->table_part_nrows << std::endl;
                for (int i = 0; i < TPCH::common::kPartFieldCount; i++)
                {
                    std::cout << "\ttable_part_start_page_ids[" << i << "]: "
                        << metadata->table_part_start_page_ids[i] << std::endl;
                    std::cout << "\ttable_part_max_nrows_in_page[" << i << "]: "
                        << metadata->table_part_max_nrows_in_page[i] << std::endl;
                }
            }
            if (options.load_supplier)
            {
                std::cout << "Supplier table nrows: " << metadata->table_supplier_nrows << std::endl;
                for (int i = 0; i < TPCH::common::kSupplierFieldCount; i++)
                {
                    std::cout << "\ttable_supplier_start_page_ids[" << i << "]: "
                        << metadata->table_supplier_start_page_ids[i] << std::endl;
                    std::cout << "\ttable_supplier_max_nrows_in_page[" << i << "]: "
                        << metadata->table_supplier_max_nrows_in_page[i] << std::endl;
                }
            }
            if (options.load_partsupp)
            {
                std::cout << "Partsupp table nrows: " << metadata->table_partsupp_nrows << std::endl;
                //for (int i = 0; i < TPCH::common::kPartFieldCount; i++)
                //{
                //    std::cout << "\ttable_part_start_page_ids[" << i << "]: "
                //        << metadata->table_part_start_page_ids[i] << std::endl;
                //    std::cout << "\ttable_part_max_nrows_in_page[" << i << "]: "
                //        << metadata->table_part_max_nrows_in_page[i] << std::endl;
                //}
                //std::cout << "Supplier table nrows: " << metadata->table_supplier_nrows << std::endl;
                //for (int i = 0; i < TPCH::common::kSupplierFieldCount; i++)
                //{
                //    std::cout << "\ttable_supplier_start_page_ids[" << i << "]: "
                //        << metadata->table_supplier_start_page_ids[i] << std::endl;
                //    std::cout << "\ttable_supplier_max_nrows_in_page[" << i << "]: "
                //        << metadata->table_supplier_max_nrows_in_page[i] << std::endl;
                //}
                std::cout << "Partsupp table nrows: " << metadata->table_partsupp_nrows << std::endl;
                for (int i = 0; i < TPCH::common::kPartSuppFieldCount; i++)
                {
                    std::cout << "\ttable_partsupp_start_page_ids[" << i << "]: "
                        << metadata->table_partsupp_start_page_ids[i] << std::endl;
                    std::cout << "\ttable_partsupp_max_nrows_in_page[" << i << "]: "
                        << metadata->table_partsupp_max_nrows_in_page[i] << std::endl;
                }
            }
            if (options.load_nation)
            {
                std::cout << "Nation table nrows: " << metadata->table_nation_nrows << std::endl;
                for (int i = 0; i < TPCH::common::kNationFieldCount; i++)
                {
                    std::cout << "\ttable_nation_start_page_ids[" << i << "]: "
                        << metadata->table_nation_start_page_ids[i] << std::endl;
                    std::cout << "\ttable_nation_max_nrows_in_page[" << i << "]: "
                        << metadata->table_nation_max_nrows_in_page[i] << std::endl;
                }
            }
            if (options.load_region)
            {
                std::cout << "Region table nrows: " << metadata->table_region_nrows << std::endl;
                for (int i = 0; i < TPCH::common::kRegionFieldCount; i++)
                {
                    std::cout << "\ttable_region_start_page_ids[" << i << "]: "
                        << metadata->table_region_start_page_ids[i] << std::endl;
                    std::cout << "\ttable_region_max_nrows_in_page[" << i << "]: "
                        << metadata->table_region_max_nrows_in_page[i] << std::endl;
                }
            }

            free(ptr);
        }
        #endif

#if 0
        {
            void *ptr;
            size_t dct_meta_xtn_id = xtn_get_dct_meta_xtnid();
            uint64_t base_page_id = xtn_calc_page_id_from_xtn_id(dct_meta_xtn_id);
            if (posix_memalign((void**)&ptr, 512,
                options.page_size * XTN::NumPagesForDctMetaXTNs) != 0)
            {
                std::cerr << "posix_memalign failed" << std::endl;
                exit(EXIT_FAILURE);
            }
            struct dct_meta_entry *dct_meta_head = new(ptr) struct dct_meta_entry[XTN::MaxNumXTNs];
            memset(dct_meta_head, 0, sizeof(struct dct_meta_entry) * XTN::MaxNumXTNs);

            char *ptr_base = reinterpret_cast<char*>(dct_meta_head);
            for (size_t i = 0; i < XTN::NumPagesForDctMetaXTNs; ++i) {
                page_pread_host(options.output_fds, (void*)&ptr_base[options.page_size * i], base_page_id + i, options.page_size);
            }

            for (size_t i = 0; i < XTN::MaxNumXTNs; ++i)
            {
                for (size_t j = 0; j < DCT::kNumMaxVarCharFields; ++j)
                {
                    for (size_t k = 0; k < DCT::kNumMaxDictsInXtn; ++k)
                    {
                        if (dct_meta_head[i].page_ids[j][k] != metadata.dct_meta_head[i].page_ids[j][k])
                        {
                            std::cerr << "[FATAL] dct_meta_head[" << i << "].page_ids[" << j << "][" << k << "] is not aligned: actual from storage: " 
                                << dct_meta_head[i].page_ids[j][k] << " (expected: " << metadata.dct_meta_head[i].page_ids[j][k] << ")" << std::endl;
                            exit(EXIT_FAILURE);
                        }

                        if (dct_meta_head[i].npages[j][k] != metadata.dct_meta_head[i].npages[j][k])
                        {
                            std::cerr << "[FATAL] dct_meta_head[" << i << "].npages[" << j << "][" << k << "] is not aligned: actual from storage: " 
                                << dct_meta_head[i].npages[j][k] << " (expected: " << metadata.dct_meta_head[i].npages[j][k] << ")" << std::endl;
                            exit(EXIT_FAILURE);
                        }

                        // TODO: add verbose output later
                        if (metadata.dct_meta_head[i].page_ids[j][k] > 0 && metadata.dct_meta_head[i].npages[j][k] > 0) {
                            std::cout << "dct_meta_head[" << i << "].page_ids[" << j << "][" << k << "]: " << dct_meta_head[i].page_ids[j][k] << std::endl;
                            std::cout << "dct_meta_head[" << i << "].npages[" << j << "][" << k << "]: " << dct_meta_head[i].npages[j][k] << std::endl;
                        }

                    }
                }
            }
            free(dct_meta_head);
        }
#endif
    }

    std::cout << "metadata verified successfully." << std::endl;
}

void metadata_sync(LoaderOptions &options, TPCHTableMetadata &metadata, uint64_t next_free_page_id)
{
    uint64_t page_size = options.page_size;
    uint16_t lbc_num_varchar_clusters = options.lbc_num_varchar_clusters;
    uint32_t compressed = options.compress;
    // uint32_t dict_encoded = options.enable_dict_encoding;
    uint32_t column = options.enable_column;
    uint32_t varchar_to_fixedchar = options.enable_varchar_to_fixedchar;
    /* Supports 1, 10, 100, 1000 */
    uint32_t scale_factor = options.scale_factor;
    
    metadata.page_size = page_size;
    metadata.lbc_num_varchar_clusters = lbc_num_varchar_clusters;
    metadata.compressed = compressed;
    metadata.column = column;
    // metadata.varchar_to_fixedchar = varchar_to_fixedchar;
    // metadata.free_page_id = next_free_page_id;
    metadata.free_page_id = next_free_page_id;

    std::cout << "page_size: " << page_size << std::endl;
    std::cout << "compressed: " << compressed << std::endl;
    //std::cout << "dict_encoded: " << dict_encoded << std::endl;
    std::cout << "scale_factor: " << scale_factor << std::endl;
    std::cout << "free_page_id: " << next_free_page_id << std::endl;
#if 0
    if (options.compress)
    {
        std::cout << "compressing metadata..." << std::endl;
        // compress metadata
        // metadata_compress(options, metadata);
    }
#endif
    if (options.dryrun){
        std::cout << "[DRYRUN] syncing metadata..." << std::endl;
        std::cout << "[DRYRUN] metadata synced successfully." << std::endl;
    } else {
        // sync metadata
        std::cout << "Syncing metadata..." << std::endl;
        std::cout << "\tsizeof(TPCHTableMetadata): " << sizeof(TPCHTableMetadata) << std::endl;
        //assert(sizeof(TPCHTableMetadata) < options.page_size);
        size_t npages = superpage_get_super_npage();
        std::cout << "\tnpages for superpage: " << npages << std::endl;

        const size_t page_size = options.page_size;
        for (size_t i = 0; i < npages; i++)
        {
            std::cout << "\t[INFO] superpage: page id " << i << " write at page id " << i << std::endl;
            page_pwrite_host(options.output_fds, (void*)&metadata, i,  page_size);
        }
        std::cout << "\tcheck: Lineitem table nrows in metadata: " << metadata.table_lineitem_nrows << std::endl;
        // for (size_t i = 0; i < TPCH::common::kLineitemFieldCount; i++)
        // {
        //     std::cout << "\tcheck: table_lineitem_start_page_ids[" << i << "]: "
        //         << metadata.table_lineitem_start_page_ids[i] << std::endl;
        // }
        // exit(1);
#if 0
        for (size_t i = 0; i < npages; i++)
        {
            //std::cout << "\tpage " << i << " write at page id " << i << std::endl;
            off_t offset = i * options.page_size;
            page_pwrite_host(options.output_fds, (void*)&metadata, 0, npages * options.page_size);
        }
#endif
        std::cout << "Synced metadata successfully." << std::endl;

        {
#if 0
            size_t super_xtn_id = xtn_get_super_xtnid();
            uint64_t base_page_id = xtn_calc_page_id_from_xtn_id(super_xtn_id);
            char *base_ptr = reinterpret_cast<char*>(metadata.xtn_head);

            for (size_t i = 0; i < XTN::NumPagesForSuperXTNs; i++)
            {
                page_pwrite_host(options.output_fds, (void*)&base_ptr[options.page_size * i], base_page_id + i, options.page_size);
            }
#endif
        }

        {
#if 0
            size_t dct_meta_xtn_id = xtn_get_dct_meta_xtnid();
            uint64_t base_page_id = xtn_calc_page_id_from_xtn_id(dct_meta_xtn_id);
            char *base_ptr = reinterpret_cast<char*>(metadata.dct_meta_head);
            for (size_t i = 0; i < XTN::NumPagesForDctMetaXTNs; i++)
            {
                page_pwrite_host(options.output_fds, (void*)&base_ptr[options.page_size * i], base_page_id + i, options.page_size);
            }
#endif
        }
    }
}

void run_tests(const LoaderOptions &options)
{
    // Run tests
 #if 1
    // constexpr std::size_t N = 512;
    constexpr std::size_t block_size = 128;
    constexpr std::size_t N = 87296;
    constexpr std::size_t NBLOCKS = N == 0 ? 0 : (N % block_size == 0 ?  (N / block_size) : ((N / block_size) + 1));
    std::array<uint, N> in_values{};
    std::array<uint, N> out_values{};
    std::array<uint, NBLOCKS> offsets{};
    std::array<uint, N> decoded_values{};

    std::array<ulong, N> in64_values{};
    std::array<ulong, N> out64_values{};
    std::array<uint, NBLOCKS> offsets64{};
    std::array<ulong, N> decoded64_values{};

    uint64_t checksum1_input = 0;
    uint64_t checksum2_input = 0;
    // uint64_t checksum1 = 0;
    // uint64_t checksum2 = 0;
    static_assert(N % 128 == 0, "N must be multiple of 128");

    for (int i = 0; i < N; i++)
    {
        if (!(std::cin >> in_values[i])) {
            if (!(std::cin >> in_values[i])) {
                std::cerr << "Input failed: " << i << std::endl;
                return;
            }
        }
    }
    for (int i = 0; i < N; i++)
    {
        checksum1_input += in_values[i];
    }
    for (int i = 0; i < N; i++)
    {
        in64_values[i] = static_cast<uint64_t>(in_values[i]);
        checksum2_input += in64_values[i];
    }

    auto [checksum1, size1] = load_test_binpack(N, in_values.data(), out_values.data(), offsets.data(), decoded_values.data());
    auto [checksum2, size2] = load_test_binpack64(N, in64_values.data(), out64_values.data(), offsets64.data(), decoded64_values.data());

    std::cout << "Input checksum (32-bit): " << checksum1_input << std::endl;
    std::cout << "Input checksum (64-bit): " << checksum2_input << std::endl;
    std::cout << "Output checksum (32-bit): " << checksum1 << std::endl;
    std::cout << "Output checksum (64-bit): " << checksum2 << std::endl;
    std::cout << "Output size (32-bit): " << size1 << std::endl;
    std::cout << "Output size (64-bit): " << size2 << std::endl;
    std::cout << "Compression ratio (32-bit): " << (static_cast<double>(size1) + (NBLOCKS) * sizeof(uint)) / static_cast<double>(N * sizeof(uint)) << std::endl;
    std::cout << "Compression ratio (64-bit): " << (static_cast<double>(size2) + (NBLOCKS) * sizeof(uint))/ static_cast<double>(N * sizeof(ulong)) << std::endl;
#else
    test_rec();
    test_pag();
    // REC *rp = malloc(16384);
    // rec_init(TPCH::common::fmt.nation_field_sizes, rp, 16384);
    // TPCH::common::fmt.nation_field_sizes;
#endif
}


void run_test_scan(LoaderOptions &options)
{
    // Run tests
    // test_rec();
    // test_pag();
    std::cout << "Start scan test" << std::endl;
    if (options.enable_column)
    {
        test_scan_column_tables(options);
    } else
    {
        fprintf(stderr, "Rowstore scan test is not implemented.\n");
        exit(EXIT_FAILURE);
    }

    std::cout << "End scan test" << std::endl;
}


int main(int argc, char *const *argv)
{
    std::cerr << "loader" << std::endl;

    LoaderOptions options = tpch_parse_loader_options(argc, argv);

    validate_loader_options(options);

    if (options.test)
    {
        std::cout << "Running tests..." << std::endl;
        run_tests(options);
        std::cout << "Tests completed." << std::endl;
        return 0;
    }

    if (options.verbose) {
        printf("options.input_dirname: %s\n", options.input_dirname);
        printf("options.output_files: %s\n", options.output_files);
        printf("options.compress: %d\n", options.compress);
    }

    open_output_files(options, options.output_fds);

    struct LoaderStats stats;
    struct TPCHTableMetadata* metadata_ptr = metadata_init(options);
    auto& metadata = *metadata_ptr;

    if (sizeof(TPCHTableMetadata) >= options.page_size)
    {
        std::cerr << "[FATAL] Unexpected: TPCHTableMetadata size is larger than page size" << std::endl;
        return -1;
    }

    // xtn id 0 is reserved for metadata
    uint64_t next_page_id = superpage_get_base_page_id();

    if (options.load_nation)
    {
        // next_page_id = loader_load_date(options, metadata, next_page_id);
        next_page_id = loader_load_nation(options, metadata, next_page_id, stats);
        // metadata_print(metadata);
    }
    if (options.load_region)
    {
        next_page_id = loader_load_region(options, metadata, next_page_id, stats);
        // metadata_print(metadata);
    }
    if (options.load_customer)
    {
        next_page_id = loader_load_customer(options, metadata, next_page_id, stats);
        // metadata_print(metadata);
    }
    if (options.load_lineitem)
    {
        next_page_id = loader_load_lineitem(options, metadata, next_page_id, stats);
    }
    if (options.load_orders)
    {
        next_page_id = loader_load_orders(options, metadata, next_page_id, stats);
    }
    if (options.load_supplier)
    {
        next_page_id = loader_load_supplier(options, metadata, next_page_id, stats);
        // metadata_print(metadata);
    }
    if (options.load_part)
    {
        next_page_id = loader_load_part(options, metadata, next_page_id, stats);
        // metadata_print(metadata);
    }
    if (options.load_partsupp)
    {
        next_page_id = loader_load_partsupp(options, metadata, next_page_id, stats);
        // metadata_print(metadata);
    }


    metadata_print(options, metadata);
    metadata_sync(options, metadata, next_page_id);
    metadata_verify(options, metadata);
    metadata_free(metadata_ptr);

    if (options.verify) {
        /* NOTE: taking long time compared to the loading time */
        run_test_scan(options);
    }

    close_output_files(options, options.output_fds);

    loader_stats_print(options.page_size, options.compress, stats);
    // nvml::shutdown();
}
