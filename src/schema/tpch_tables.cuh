#pragma once

#include <iostream>
#include <variant>
#include <array>
#include <algorithm>
#include <string_view>
#include <numeric>
#include <type_traits>
#include <utility> 

#define TABLE_NAME_CUSTOMER "CUSTOMER"
#define TABLE_NAME_LINEITEM "LINEITEM"
#define TABLE_NAME_ORDERS "ORDERS"
#define TABLE_NAME_PART "PART"
#define TABLE_NAME_SUPPLIER "SUPPLIER"
#define TABLE_NAME_PARTSUPP "PARTSUPP"
#define TABLE_NAME_NATION "NATION"
#define TABLE_NAME_REGION "REGION"

#include "common/rec.h"
// #include "common/xtn.h"
#include "common/compression.cuh"

/* Forward declaration */
struct TPCHTableMetadata;

namespace TPCH {
    namespace common {
        enum Table {
            CUSTOMER = 0,
            LINEITEM,
            ORDERS,
            SUPPLIER,
            PART,
            PARTSUPP,
            REGION,
            NATION,
            TABLES
        };

        inline const char* table_name(Table table) {
            switch (table) {
                case Table::CUSTOMER: return "customer";
                case Table::LINEITEM: return "lineitem";
                case Table::ORDERS: return "orders";
                case Table::SUPPLIER: return "supplier";
                case Table::PART: return "part";
                case Table::PARTSUPP: return "partsupp";
                case Table::REGION: return "region";
                case Table::NATION: return "nation";
                default: return "unknown";
            }
        }

        enum CustomerField {
            C_CUSTKEY = 0,
            C_NAME,
            C_ADDRESS,
            C_NATIONKEY,
            C_PHONE,
            C_ACCTBAL,
            C_MKTSEGMENT,
            C_COMMENT,
            C_FIELDS
        };

        enum LineitemField {
            L_ORDERKEY = 0,
            L_PARTKEY,
            L_SUPPKEY,
            L_LINENUMBER,
            L_QUANTITY,
            L_EXTENDEDPRICE,
            L_DISCOUNT,
            L_TAX,
            L_RETURNFLAG,
            L_LINESTATUS,
            L_SHIPDATE,
            L_COMMITDATE,
            L_RECEIPTDATE,
            L_SHIPINSTRUCT,
            L_SHIPMODE,
            L_COMMENT,
            L_FIELDS
        };

        enum OrdersField {
            O_ORDERKEY = 0,
            O_CUSTKEY,
            O_ORDERSTATUS,
            O_TOTALPRICE,
            O_ORDERDATE,
            O_ORDERPRIORITY,
            O_CLERK,
            O_SHIPPRIORITY,
            O_COMMENT,
            O_FIELDS
        };

        enum SupplierField {
            S_SUPPKEY = 0,
            S_NAME,
            S_ADDRESS,
            S_NATIONKEY,
            S_PHONE,
            S_ACCTBAL,
            S_COMMENT,
            S_FIELDS
        };

        enum PartField {
            P_PARTKEY = 0,
            P_NAME,
            P_MFGR,
            P_BRAND,
            P_TYPE,
            P_SIZE,
            P_CONTAINER,
            P_RETAILPRICE,
            P_COMMENT,
            P_FIELDS
        };

        enum PartSuppField {
            PS_PARTKEY = 0,
            PS_SUPPKEY,
            PS_AVAILQTY,
            PS_SUPPLYCOST,
            PS_COMMENT,
            PS_FIELDS
        };

        enum NationField {
            N_NATIONKEY = 0,
            N_NAME,
            N_REGIONKEY,
            N_COMMENT,
            N_FIELDS
        };

        enum RegionField {
            R_REGIONKEY = 0,
            R_NAME,
            R_COMMENT,
            R_FIELDS
        };

        /* for GOLAP */
        enum LineitemSidewaysField {
            LS_SIDEWAYS_R_NAME = 0,
            LS_SIDEWAYS_C_MKTSEGMENT,
            LS_SIDEWAYS_O_ORDERDATE,
            LS_FIELDS
        };

        enum OrdersSidewaysField {
            OS_SIDEWAYS_R_NAME = 0,
            OS_SIDEWAYS_C_MKTSEGMENT,
            OS_FIELDS
        };

        /* CHAR(25) */
        constexpr std::array<std::string_view, 5> dict_r_name_for_query {
            "AFRICA                   ", // 0
            "AMERICA                  ", // 1
            "ASIA                     ", // 2
            "EUROPE                   ", // 3
            "MIDDLE EAST              "  // 4
        };
        constexpr std::array<std::string_view, 5> dict_r_name_for_load {
            "AFRICA",     // 0
            "AMERICA",    // 1
            "ASIA",       // 2
            "EUROPE",     // 3
            "MIDDLE EAST" // 4
        };

        /* CHAR(10) */
        constexpr std::array<std::string_view, 5> dict_c_mktsegment_for_query {
            "AUTOMOBILE", // 0
            "BUILDING  ", // 1
            "FURNITURE ", // 2
            "MACHINERY ", // 3
            "HOUSEHOLD "  // 4
        };
        constexpr std::array<std::string_view, 5> dict_c_mktsegment_for_load {
            "AUTOMOBILE", // 0
            "BUILDING",   // 1
            "FURNITURE",  // 2
            "MACHINERY",  // 3
            "HOUSEHOLD"   // 4
        };

        using Field = std::variant<CustomerField, LineitemField,
                                OrdersField, SupplierField, PartField,
                                PartSuppField, NationField, RegionField>;

        std::string metric_field_name(const TPCH::common::Field field)
        {
            return std::visit([](auto f) -> std::string {
                using T = decltype(f);
                if constexpr (std::is_same_v<T, TPCH::common::CustomerField>) {
                    switch (f) {
                    case C_CUSTKEY:
                        return "C_CUSTKEY";
                    case C_NAME:
                        return "C_NAME";
                    case C_ADDRESS:
                        return "C_ADDRESS";
                    case C_NATIONKEY:
                        return "C_NATIONKEY";
                    case C_PHONE:
                        return "C_PHONE";
                    case C_ACCTBAL:
                        return "C_ACCTBAL";
                    case C_MKTSEGMENT:
                        return "C_MKTSEGMENT";
                    case C_COMMENT:
                        return "C_COMMENT";
                    default:
                        return "C_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, TPCH::common::LineitemField>) {
                    switch (f) {
                    case L_ORDERKEY:
                        return "L_ORDERKEY";
                    case L_PARTKEY:
                        return "L_PARTKEY";
                    case L_SUPPKEY:
                        return "L_SUPPKEY";
                    case L_LINENUMBER:
                        return "L_LINENUMBER";
                    case L_QUANTITY:
                        return "L_QUANTITY";
                    case L_EXTENDEDPRICE:
                        return "L_EXTENDEDPRICE";
                    case L_DISCOUNT:
                        return "L_DISCOUNT";
                    case L_TAX:
                        return "L_TAX";
                    case L_RETURNFLAG:
                        return "L_RETURNFLAG";
                    case L_LINESTATUS:
                        return "L_LINESTATUS";
                    case L_SHIPDATE:
                        return "L_SHIPDATE";
                    case L_COMMITDATE:
                        return "L_COMMITDATE";
                    case L_RECEIPTDATE:
                        return "L_RECEIPTDATE";
                    case L_SHIPINSTRUCT:
                        return "L_SHIPINSTRUCT";
                    case L_SHIPMODE:
                        return "L_SHIPMODE";
                    case L_COMMENT:
                        return "L_COMMENT";
                    default:
                        return "L_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, TPCH::common::OrdersField>) {
                    switch (f) {
                    case O_ORDERKEY:
                        return "O_ORDERKEY";
                    case O_CUSTKEY:
                        return "O_CUSTKEY";
                    case O_ORDERSTATUS:
                        return "O_ORDERSTATUS";
                    case O_TOTALPRICE:
                        return "O_TOTALPRICE";
                    case O_ORDERDATE:
                        return "O_ORDERDATE";
                    case O_ORDERPRIORITY:
                        return "O_ORDERPRIORITY";
                    case O_CLERK:
                        return "O_CLERK";
                    case O_SHIPPRIORITY:
                        return "O_SHIPPRIORITY";
                    case O_COMMENT:
                        return "O_COMMENT";
                    default:
                        return "O_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, TPCH::common::PartField>) {
                    switch (f) {
                    case P_PARTKEY:
                        return "P_PARTKEY";
                    case P_NAME:
                        return "P_NAME";
                    case P_MFGR:
                        return "P_MFGR";
                    case P_BRAND:
                        return "P_BRAND";
                    case P_TYPE:
                        return "P_TYPE";
                    case P_SIZE:
                        return "P_SIZE";
                    case P_CONTAINER:
                        return "P_CONTAINER";
                    case P_RETAILPRICE:
                        return "P_RETAILPRICE";
                    case P_COMMENT:
                        return "P_COMMENT";
                    default:
                        return "P_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, TPCH::common::SupplierField>) {
                    switch (f) {
                    case S_SUPPKEY:
                        return "S_SUPPKEY";
                    case S_NAME:
                        return "S_NAME";
                    case S_ADDRESS:
                        return "S_ADDRESS";
                    case S_NATIONKEY:
                        return "S_NATIONKEY";
                    case S_PHONE:
                        return "S_PHONE";
                    case S_ACCTBAL:
                        return "S_ACCTBAL";
                    case S_COMMENT:
                        return "S_COMMENT";
                    default:
                        return "S_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, TPCH::common::PartSuppField>) {
                    switch (f) {
                    case PS_PARTKEY:
                        return "PS_PARTKEY";
                    case PS_SUPPKEY:
                        return "PS_SUPPKEY";
                    case PS_AVAILQTY:
                        return "PS_AVAILQTY";
                    case PS_SUPPLYCOST:
                        return "PS_SUPPLYCOST";
                    case PS_COMMENT:
                        return "PS_COMMENT";
                    default:
                        return "PS_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, TPCH::common::NationField>) {
                    switch (f) {
                    case N_NATIONKEY:
                        return "N_NATIONKEY";
                    case N_NAME:
                        return "N_NAME";
                    case N_REGIONKEY:
                        return "N_REGIONKEY";
                    case N_COMMENT:
                        return "N_COMMENT";
                    default:
                        return "N_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, TPCH::common::RegionField>) {
                    switch (f) {
                    case R_REGIONKEY:
                        return "R_REGIONKEY";
                    case R_NAME:
                        return "R_NAME";
                    case R_COMMENT:
                        return "R_COMMENT";
                    default:
                        return "R_UNKNOWN";
                    }
                } else {
                    return "UNKNOWN";
                }
              }, field);
        }

        constexpr uint32_t kCustomerFieldCount = C_FIELDS;
        constexpr uint32_t kLineitemFieldCount = L_FIELDS;
        constexpr uint32_t kOrdersFieldCount = O_FIELDS;
        constexpr uint32_t kPartFieldCount = P_FIELDS;
        constexpr uint32_t kSupplierFieldCount = S_FIELDS;
        constexpr uint32_t kPartSuppFieldCount = PS_FIELDS;
        constexpr uint32_t kNationFieldCount = N_FIELDS;
        constexpr uint32_t kRegionFieldCount = R_FIELDS;

        constexpr uint32_t kLineitemSidewaysCount = LS_FIELDS;
        constexpr uint32_t kOrdersSidewaysCount = OS_FIELDS;

        constexpr uint32_t TPCH_MAX_NFIELDS = std::max({
            kCustomerFieldCount,
            kLineitemFieldCount,
            kOrdersFieldCount,
            kPartFieldCount,
            kSupplierFieldCount,
            kPartSuppFieldCount,
            kNationFieldCount,
            kRegionFieldCount
        });

        namespace Row {
            namespace Char {
                #if 0
                struct Customer {
                    // For SF >= 1000
                    int64_t custkey;
                    char name[25];
                    char address[40];
                    int32_t nationkey;
                    int32_t city;
                    char phone[15];
                    int64_t acctbal;
                    char mktsegment[10];
                    char comment[117];
                };
                // TODO: add other tables
                size_t size_customer = sizeof(struct Customer);
                #endif
            }

            namespace Varchar {
                #if 0
                struct Customer {
                    uint16_t lenrec;
                    uint16_t ofst[kCustomerFieldCount] = { offsetof(Customer, custkey),};
                    // For SF >= 1000
                    int64_t custkey;
                    struct varchar_head name; // VARCHAR(25)
                    struct varchar_head address; // VARCHAR(40)
                    int32_t nationkey;
                    int32_t city;
                    char phone[15];
                    char mktsegment[10];
                    int64_t acctbal;
                    struct varchar_head comment; // VARCHAR(117)
                };
                size_t size_customer = sizeof(struct Customer);
                size_t size_base_customer = sizeof(struct Customer);
                #endif
            }
        }

        template <typename E, std::size_t N,
          typename = std::enable_if_t<std::is_enum_v<E>>>
        constexpr auto make_array_from_enum_values() noexcept
        {
            return []<std::size_t... I>(std::index_sequence<I...>)
            {
                return std::array<E, N>{ { static_cast<E>(I)... } };
            }(std::make_index_sequence<N>{});
        }

        template <typename T, std::size_t N>
        consteval std::array<T, N>
        make_field_max_sizes(const std::array<T, N> &base,
                             const std::array<enum rec_type, N> &type)
        {
            std::array<T, N> out{};
            for (std::size_t i = 0; i < N; ++i)
            {
                out[i] = base[i] + sizeof(uint16_t); // for arofst
            }
            return out;
        }

        // template <std::size_t N>
        // consteval std::array<CompressionMethod, N>
        // make_compression_array()
        // {
        //     std::array<T, N> out{};
        //     for (std::size_t i = 0; i < N; ++i)
        //     {
        //         out[i] = Co
        //     }
        //     return out;
        // }

        template <typename T, std::size_t N>
        consteval std::array<T, N>
        make_field_dict_encoded_sizes(const std::array<T, N> &base,
                             const std::array<enum rec_type, N> &type)
        {
            std::array<T, N> out{};
            for (std::size_t i = 0; i < N; ++i)
            {
                if (type[i] == REC_ATTR_VCHAR) {
                    /* for dict encoded varchar */
                    /* sizeof(dict entry id)  */
                    out[i] = sizeof(uint32_t);
                } else {
                    out[i] = base[i]; // for other types
                }
            }
            return out;
        }

        template <typename T, std::size_t N>
        constexpr std::size_t
        accumulate_field_varchar_sizes(const std::array<T, N> &base,
                             const std::array<enum rec_type, N> &type)
        {
            std::size_t out = sizeof(uint16_t);
            for (std::size_t i = 0; i < N; ++i)
            {
                if (type[i] == REC_ATTR_VCHAR) {
                    out += base[i]; // for varchar
                }
            }
            return out;
        }
        #if 0
        template <typename T, std::size_t N>
        consteval std::array<T, N>
        make_field_min_sizes(const std::array<T, N> &base,
                             const std::array<enum rec_type, N> &type)
        {
            std::array<T, N> out{};
            for (std::size_t i = 0; i < N; ++i)
            {
                //out[i] = (type[i] == REC_ATTR_VCHAR ? sizeof(struct varchar_head) : base[i]);
                out[i] = (type[i] == REC_ATTR_VCHAR ? sizeof(struct varchar_head) : base[i]);
            }
            return out;
        }
        #endif

        template<typename T, std::size_t N>
        constexpr std::size_t count_varchar_types(const std::array<T, N>& arr)
        {
            std::size_t cnt = 0;
            for (std::size_t i = 0; i < N; ++i) {
                if (arr[i] == REC_ATTR_VCHAR) ++cnt;
            }
            return cnt;
        }

        template<std::size_t M, typename T, std::size_t N>
        constexpr std::array<std::size_t, M>
        make_varchar_field_indexes(const std::array<T, N>& types)
        {
            std::array<std::size_t, M> idx{};
            std::size_t pos = 0;
        
            for (std::size_t i = 0; i < N; ++i) {
                if (types[i] == REC_ATTR_VCHAR) {
                    idx[pos] = i;
                    pos++;
                }
            }
        
            return idx;
        }

        template<std::size_t N>
        constexpr std::array<std::array<std::size_t, 1>, N>
        make_array_column_sizes(const std::array<size_t, N>& sizes)
        {
            std::array<std::array<std::size_t, 1>, N> out{};
            for (std::size_t i = 0; i < N; ++i) {
                out[i][0] = sizes[i];
            }
            return out;
        }

        template<std::size_t N, typename T>
        constexpr std::array<enum rec_type, N>
        make_dict_encoded_types_array(const std::array<T, N>& types)
        {
            std::array<enum rec_type, N> encoded_types{};
        
            for (std::size_t i = 0; i < N; ++i) {
                if (types[i] == REC_ATTR_VCHAR) {
                    encoded_types[i] = REC_ATTR_INT32;
                } else {
                    encoded_types[i] = types[i];
                }
            }
        
            return encoded_types;
        }

        template<std::size_t N>
        constexpr std::array<CompressionMethod, N> make_compression_types_array(
            const std::array<enum rec_type, N> &types)
        {
            std::array<CompressionMethod, N> compression_types{};

            for (std::size_t i = 0; i < N; ++i) {
                switch (types[i]) {
                    case REC_ATTR_INT16:
                    case REC_ATTR_INT32:
                    case REC_ATTR_DATE:
                    case REC_ATTR_DECIMAL:
                        compression_types[i] = CompressionMethod::PFOR;
                        break;
                    case REC_ATTR_INT64:
                        compression_types[i] = CompressionMethod::PFOR64;
                        break;
                    case REC_ATTR_CHAR:
                    case REC_ATTR_VCHAR:
                        compression_types[i] = CompressionMethod::LZ4;
                        break;
                    default:
                        break;
                }
            }
            return compression_types;
        }

        template<std::size_t N>
        constexpr std::array<CompressionMethod, N> make_compression_types_array(
            const std::array<enum rec_type, N> &types,
            const std::array<size_t, N> &sizes)
        {
            std::array<CompressionMethod, N> compression_types{};

            for (std::size_t i = 0; i < N; ++i) {
                switch (types[i]) {
                    case REC_ATTR_INT16:
                    case REC_ATTR_INT32:
                    case REC_ATTR_DATE:
                    case REC_ATTR_DECIMAL:
                        compression_types[i] = CompressionMethod::PFOR;
                        break;
                    case REC_ATTR_INT64:
                        compression_types[i] = CompressionMethod::PFOR64;
                        break;
                    case REC_ATTR_CHAR:
                        compression_types[i] = (sizes[i] < 3) ? CompressionMethod::PFOR
                                                               : CompressionMethod::LZ4;
                        break;
                    case REC_ATTR_VCHAR:
                        compression_types[i] = CompressionMethod::LZ4;
                        break;
                    default:
                        break;
                }
            }
            return compression_types;
        }

        struct recfmt_t {
            /* NOTE: UINT16,32,64 and CHAR -> attr size is fixed */

            static constexpr std::array<size_t, 0> no_filter_columns = {};
            /*  VCHAR -> size is variable, and value defined here is just a maximum value */
            /* CUSTOMER */
            static constexpr auto customer_fields = make_array_from_enum_values<CustomerField, kCustomerFieldCount>();
            static constexpr std::array<enum rec_type, kCustomerFieldCount> customer_field_types =
                {REC_ATTR_INT64, REC_ATTR_VCHAR, REC_ATTR_VCHAR, REC_ATTR_INT32,
                 REC_ATTR_CHAR, REC_ATTR_DECIMAL, REC_ATTR_CHAR, REC_ATTR_VCHAR};
            static constexpr std::array<bool, kCustomerFieldCount> customer_enable_stats_columns = {
                false, false, false, false,
                false, false, false, false
            };
            static constexpr std::array<size_t, 0> customer_filter_columns = no_filter_columns;
            static constexpr size_t customer_varchar_field_count = count_varchar_types(customer_field_types);
            static constexpr std::array<size_t, customer_varchar_field_count> customer_varchar_field_indexes = make_varchar_field_indexes<customer_varchar_field_count>(
                customer_field_types
            );
            /* *_field_sizes contains the fields's length information */
            static constexpr std::array<size_t, kCustomerFieldCount> customer_field_sizes =
                {sizeof(int64_t), 25, 40, sizeof(int32_t), 15, sizeof(int32_t), 10, 117};
            static constexpr std::array<std::array<size_t, 1>, kCustomerFieldCount> customer_arr_column_sizes = make_array_column_sizes(
                customer_field_sizes
            );

            /* *_field_max_sizes contains the max fields's length information for memory allocation */
            static constexpr auto customer_field_max_sizes = make_field_max_sizes(
                customer_field_sizes, customer_field_types);
            static constexpr std::size_t customer_row_max_varchar_sizes = accumulate_field_varchar_sizes(
                customer_field_sizes, customer_field_types);
            /* *_field_base_sizes contains the schema's min length information */
            static constexpr std::size_t customer_row_max_size = std::accumulate(
                customer_field_max_sizes.begin(), customer_field_max_sizes.end(), sizeof(uint16_t));  
            static constexpr std::array<enum rec_type, kCustomerFieldCount> customer_field_dict_encoded_types =
                make_dict_encoded_types_array(customer_field_types);
            static constexpr auto customer_field_dict_encoded_sizes = make_field_dict_encoded_sizes(
                customer_field_sizes, customer_field_types);
            static constexpr auto customer_dict_encoded_row_max_sizes = make_field_max_sizes(
                customer_field_dict_encoded_sizes, customer_field_dict_encoded_types);
            static constexpr size_t customer_dict_encoded_row_max_size = std::accumulate(
                customer_dict_encoded_row_max_sizes.begin(), customer_dict_encoded_row_max_sizes.end(), sizeof(uint16_t));
            /* Using default value NONE for now */
            static constexpr std::array<CompressionMethod, kCustomerFieldCount> customer_field_compression_types = make_compression_types_array(
                customer_field_types
            );

            //static constexpr size_t customer_dict_encoded_row_size = std::accumulate(
            //    customer_field_dict_encoded_sizes.begin(), customer_field_dict_encoded_sizes.end(), sizeof(uint16_t));

            /* LINEITEM */
            static constexpr auto lineitem_fields = make_array_from_enum_values<LineitemField, kLineitemFieldCount>();
            static constexpr std::array<enum rec_type, kLineitemFieldCount> lineitem_field_types = {
                REC_ATTR_INT64, REC_ATTR_INT64, REC_ATTR_INT64, REC_ATTR_INT32,
                REC_ATTR_DECIMAL, REC_ATTR_DECIMAL, REC_ATTR_DECIMAL, REC_ATTR_DECIMAL,
                REC_ATTR_CHAR, REC_ATTR_CHAR, REC_ATTR_DATE, REC_ATTR_DATE,
                REC_ATTR_DATE, REC_ATTR_CHAR, REC_ATTR_CHAR, REC_ATTR_VCHAR
            };
            static constexpr std::array<size_t, kLineitemFieldCount> lineitem_field_sizes = {
                sizeof(int64_t), sizeof(int64_t), sizeof(int64_t), sizeof(int32_t),
                sizeof(int32_t), sizeof(int32_t), sizeof(int32_t), sizeof(int32_t),
                1, 1, sizeof(int32_t), sizeof(int32_t),
                sizeof(int32_t), 25, 10, 44
            };
            static constexpr std::array<CompressionMethod, kLineitemFieldCount> lineitem_field_compression_types = make_compression_types_array(
                lineitem_field_types, lineitem_field_sizes
            );
            static constexpr std::array<bool, kLineitemFieldCount> lineitem_enable_stats_columns = {
                false, false, false, false,
                false, false, false, false,
                false, false, true, true,
                true, false, false, false
            };
            static constexpr std::array<size_t, 1> lineitem_filter_columns = {
                L_SHIPDATE
            };
            static constexpr auto lineitem_varchar_field_count = count_varchar_types(
                lineitem_field_types);
            static constexpr std::array<size_t, lineitem_varchar_field_count> lineitem_varchar_field_indexes = make_varchar_field_indexes<lineitem_varchar_field_count>(
                lineitem_field_types
            );
            static constexpr std::array<std::array<size_t, 1>, kLineitemFieldCount> lineitem_arr_column_sizes = make_array_column_sizes(
                lineitem_field_sizes
            );
            static constexpr auto lineitem_field_max_sizes = make_field_max_sizes(
                lineitem_field_sizes, lineitem_field_types);
            static constexpr std::size_t lineitem_row_max_varchar_sizes = accumulate_field_varchar_sizes(
                lineitem_field_sizes, lineitem_field_types);
            static constexpr std::uint32_t lineitem_row_max_size = std::accumulate(
                lineitem_field_max_sizes.begin(), lineitem_field_max_sizes.end(), 0u);  
            static constexpr std::array<enum rec_type, kLineitemFieldCount> lineitem_field_dict_encoded_types =
                make_dict_encoded_types_array(lineitem_field_types);
            static constexpr auto lineitem_field_dict_encoded_sizes = make_field_dict_encoded_sizes(
                lineitem_field_sizes, lineitem_field_types);
            static constexpr auto lineitem_dict_encoded_row_max_sizes = make_field_max_sizes(
                lineitem_field_dict_encoded_sizes, lineitem_field_dict_encoded_types);
            static constexpr size_t lineitem_dict_encoded_row_max_size = std::accumulate(
                lineitem_dict_encoded_row_max_sizes.begin(), lineitem_dict_encoded_row_max_sizes.end(), sizeof(uint16_t));

            static constexpr std::array<enum rec_type, kLineitemSidewaysCount> lineitem_sideways_information_types = {
                /* The following attributes are for sideways information passing */
                /* R_NAME, C_MKTSEGMENT, O_ORDERDATE, */
                REC_ATTR_CHAR, REC_ATTR_CHAR, REC_ATTR_DATE
            };
            static constexpr std::array<size_t, kLineitemSidewaysCount> lineitem_sideways_information_sizes = {
                /* The following attributes are for sideways information passing */
                25, 10, sizeof(int32_t)
            };
            /* Using default value NONE for now */

 
            /* ORDERS */
            static constexpr auto orders_fields = make_array_from_enum_values<OrdersField, kOrdersFieldCount>();
            static constexpr std::array<enum rec_type, kOrdersFieldCount> orders_field_types = {
                REC_ATTR_INT64, REC_ATTR_INT64, REC_ATTR_CHAR, REC_ATTR_DECIMAL,
                REC_ATTR_DATE, REC_ATTR_CHAR, REC_ATTR_CHAR, REC_ATTR_INT32, REC_ATTR_VCHAR
            };
            static constexpr std::array<bool, kOrdersFieldCount> orders_enable_stats_columns = {
                false, false, false, false,
                true, false, false, false, false
            };
            static constexpr std::array<size_t, 0> orders_filter_columns = no_filter_columns;
            static constexpr auto orders_varchar_field_count = count_varchar_types(
                orders_field_types);
            static constexpr std::array<size_t, orders_varchar_field_count> orders_varchar_field_indexes = make_varchar_field_indexes<orders_varchar_field_count>(
                orders_field_types
            );
            static constexpr std::array<size_t, kOrdersFieldCount> orders_field_sizes = {
                sizeof(int64_t), sizeof(int64_t), 1, sizeof(int32_t),
                sizeof(int32_t), 15, 15, sizeof(int32_t), 79
            };
            static constexpr std::array<std::array<size_t, 1>, kOrdersFieldCount> orders_arr_column_sizes = make_array_column_sizes(
                orders_field_sizes
            );
            static constexpr auto orders_field_max_sizes = make_field_max_sizes(
                orders_field_sizes, orders_field_types);
            static constexpr std::size_t orders_row_max_varchar_sizes = accumulate_field_varchar_sizes(
                orders_field_sizes, orders_field_types);
            static constexpr std::uint32_t orders_row_max_size = std::accumulate(
                orders_field_max_sizes.begin(), orders_field_max_sizes.end(), 0u);  
            static constexpr std::array<enum rec_type, kOrdersFieldCount> orders_field_dict_encoded_types =
                make_dict_encoded_types_array(orders_field_types);
            static constexpr auto orders_field_dict_encoded_sizes = make_field_dict_encoded_sizes(
                orders_field_sizes, orders_field_types);
            // static constexpr size_t orders_row_dict_encoded_size = std::accumulate(
            //     orders_field_dict_encoded_sizes.begin(), orders_field_dict_encoded_sizes.end(), 0u);
            static constexpr auto orders_dict_encoded_row_max_sizes = make_field_max_sizes(
                orders_field_dict_encoded_sizes, orders_field_dict_encoded_types);
            static constexpr size_t orders_dict_encoded_row_max_size = std::accumulate(
                orders_dict_encoded_row_max_sizes.begin(), orders_dict_encoded_row_max_sizes.end(), sizeof(uint16_t));
            /* Using default value NONE for now */
            static constexpr std::array<CompressionMethod, kOrdersFieldCount> orders_field_compression_types = make_compression_types_array(
                orders_field_types
            );
 
            static constexpr std::array<enum rec_type, kOrdersSidewaysCount> orders_sideways_information_types = {
                REC_ATTR_CHAR, REC_ATTR_CHAR
            };
            static constexpr std::array<size_t, kOrdersSidewaysCount> orders_sideways_information_sizes = {
                /* R_NAME, C_MKTSEGMENT */
                25, 10
            };
 
            /* PART */
            static constexpr auto part_fields = make_array_from_enum_values<PartField, kPartFieldCount>();
            static constexpr std::array<enum rec_type, kPartFieldCount> part_field_types = {
                REC_ATTR_INT64, REC_ATTR_VCHAR, REC_ATTR_CHAR, REC_ATTR_CHAR,
                REC_ATTR_VCHAR, REC_ATTR_INT32, REC_ATTR_CHAR, REC_ATTR_DECIMAL,
                REC_ATTR_VCHAR
            };
            static constexpr std::array<bool, kPartFieldCount> part_enable_stats_columns = {
                false, false, false, false,
                false, false, false, false,
                false
            };
            static constexpr std::array<size_t, 0> part_filter_columns = no_filter_columns;
            static constexpr auto part_varchar_field_count = count_varchar_types(
                part_field_types);
            static constexpr std::array<size_t, part_varchar_field_count> part_varchar_field_indexes = make_varchar_field_indexes<part_varchar_field_count>(
                part_field_types
            );
            static constexpr std::array<size_t, kPartFieldCount> part_field_sizes = {
                sizeof(int64_t), 55, 25, 10,
                25, sizeof(int32_t), 10, sizeof(int32_t),
                23
            };
            static constexpr std::array<std::array<size_t, 1>, kPartFieldCount> part_arr_column_sizes = make_array_column_sizes(
                part_field_sizes
            );
            static constexpr auto part_field_max_sizes = make_field_max_sizes(
                part_field_sizes, part_field_types);
            static constexpr std::size_t part_row_max_varchar_sizes = accumulate_field_varchar_sizes(
                part_field_sizes, part_field_types);
            static constexpr std::uint32_t part_row_max_size = std::accumulate(
                part_field_max_sizes.begin(), part_field_max_sizes.end(), 0u);  
            static constexpr std::array<enum rec_type, kPartFieldCount> part_field_dict_encoded_types =
                make_dict_encoded_types_array(part_field_types);
            static constexpr auto part_field_dict_encoded_sizes = make_field_dict_encoded_sizes(
                part_field_sizes, part_field_types);
            //static constexpr size_t part_row_dict_encoded_size = std::accumulate(
            //    part_field_dict_encoded_sizes.begin(), part_field_dict_encoded_sizes.end(), 0u);
            static constexpr auto part_dict_encoded_row_max_sizes = make_field_max_sizes(
                part_field_dict_encoded_sizes, part_field_dict_encoded_types);
            static constexpr size_t part_dict_encoded_row_max_size = std::accumulate(
                part_dict_encoded_row_max_sizes.begin(), part_dict_encoded_row_max_sizes.end(), sizeof(uint16_t));
            /* Using default value NONE for now */
            static constexpr std::array<CompressionMethod, kPartFieldCount> part_field_compression_types = make_compression_types_array(
                part_field_types
            );
  
            /* SUPPLIER */
            static constexpr auto supplier_fields = make_array_from_enum_values<SupplierField, kSupplierFieldCount>();
            static constexpr std::array<enum rec_type, kSupplierFieldCount> supplier_field_types = {
                REC_ATTR_INT64, REC_ATTR_CHAR, REC_ATTR_VCHAR, REC_ATTR_INT32,
                REC_ATTR_CHAR, REC_ATTR_DECIMAL, REC_ATTR_VCHAR
            };
            static constexpr std::array<bool, kSupplierFieldCount> supplier_enable_stats_columns = {
                false, false, false, false,
                false, false, false
            };
            static constexpr std::array<size_t, 0> supplier_filter_columns = no_filter_columns;
            static constexpr auto supplier_varchar_field_count = count_varchar_types(
                supplier_field_types);
            static constexpr std::array<size_t, supplier_varchar_field_count> supplier_varchar_field_indexes = make_varchar_field_indexes<supplier_varchar_field_count>(
                supplier_field_types
            );
            static constexpr std::array<size_t, kSupplierFieldCount> supplier_field_sizes = {
                sizeof(int64_t), 25, 40, sizeof(int32_t),
                15, sizeof(int32_t), 101,
            };
            static constexpr std::array<std::array<size_t, 1>, kSupplierFieldCount> supplier_arr_column_sizes = make_array_column_sizes(
                supplier_field_sizes
            );
            static constexpr auto supplier_field_max_sizes = make_field_max_sizes(
                supplier_field_sizes, supplier_field_types);
            static constexpr std::size_t supplier_row_max_varchar_sizes = accumulate_field_varchar_sizes(
                supplier_field_sizes, supplier_field_types);
            static constexpr std::uint32_t supplier_row_max_size = std::accumulate(
                supplier_field_max_sizes.begin(), supplier_field_max_sizes.end(), 0u);  
            static constexpr std::array<enum rec_type, kSupplierFieldCount> supplier_field_dict_encoded_types =
                make_dict_encoded_types_array(supplier_field_types);
            static constexpr auto supplier_field_dict_encoded_sizes = make_field_dict_encoded_sizes(
                supplier_field_sizes, supplier_field_types);
            //static constexpr size_t supplier_row_dict_encoded_size = std::accumulate(
            //    supplier_field_dict_encoded_sizes.begin(), supplier_field_dict_encoded_sizes.end(), 0u);
            static constexpr auto supplier_dict_encoded_row_max_sizes = make_field_max_sizes(
                supplier_field_dict_encoded_sizes, supplier_field_dict_encoded_types);
            static constexpr size_t supplier_dict_encoded_row_max_size = std::accumulate(
                supplier_dict_encoded_row_max_sizes.begin(), supplier_dict_encoded_row_max_sizes.end(), sizeof(uint16_t));
            static constexpr std::array<CompressionMethod, kSupplierFieldCount> supplier_field_compression_types = make_compression_types_array(
                supplier_field_types
            );
  
 
            /* PARTSUPP */
            static constexpr auto partsupp_fields = make_array_from_enum_values<PartSuppField, kPartSuppFieldCount>();
            static constexpr std::array<enum rec_type, kPartSuppFieldCount> partsupp_field_types = {
                REC_ATTR_INT64, REC_ATTR_INT64, REC_ATTR_INT32, REC_ATTR_DECIMAL,
                REC_ATTR_VCHAR
            };
            static constexpr std::array<bool, kPartSuppFieldCount> partsupp_enable_stats_columns = {
                false, false, false, false,
                false
            };
            static constexpr std::array<size_t, 0> partsupp_filter_columns = no_filter_columns;
            static constexpr auto partsupp_varchar_field_count = count_varchar_types(
                partsupp_field_types);
            static constexpr std::array<size_t, partsupp_varchar_field_count> partsupp_varchar_field_indexes = make_varchar_field_indexes<partsupp_varchar_field_count>(
                partsupp_field_types
            );
            static constexpr std::array<size_t, kPartSuppFieldCount> partsupp_field_sizes = {
                sizeof(int64_t), sizeof(int64_t), sizeof(int32_t), sizeof(int32_t),
                199
            };
            static constexpr std::array<std::array<size_t, 1>, kPartSuppFieldCount> partsupp_arr_column_sizes = make_array_column_sizes(
                partsupp_field_sizes
            );
            static constexpr auto partsupp_field_max_sizes = make_field_max_sizes(
                partsupp_field_sizes, partsupp_field_types);
            static constexpr std::size_t partsupp_row_max_varchar_sizes = accumulate_field_varchar_sizes(
                partsupp_field_sizes, partsupp_field_types);
            static constexpr std::uint32_t partsupp_row_max_size = std::accumulate(
                partsupp_field_max_sizes.begin(), partsupp_field_max_sizes.end(), 0u);  
            static constexpr std::array<enum rec_type, kPartSuppFieldCount> partsupp_field_dict_encoded_types =
                make_dict_encoded_types_array(partsupp_field_types);
            static constexpr auto partsupp_field_dict_encoded_sizes = make_field_dict_encoded_sizes(
                partsupp_field_sizes, partsupp_field_types);
            static constexpr size_t partsupp_row_dict_encoded_size = std::accumulate(
                partsupp_field_dict_encoded_sizes.begin(), partsupp_field_dict_encoded_sizes.end(), 0u);
            static constexpr auto partsupp_dict_encoded_row_max_sizes = make_field_max_sizes(
                partsupp_field_dict_encoded_sizes, partsupp_field_dict_encoded_types);
            static constexpr size_t partsupp_dict_encoded_row_max_size = std::accumulate(
                partsupp_dict_encoded_row_max_sizes.begin(), partsupp_dict_encoded_row_max_sizes.end(), sizeof(uint16_t));
            static constexpr std::array<CompressionMethod, kPartSuppFieldCount> partsupp_field_compression_types = make_compression_types_array(
                partsupp_field_types
            );

 
            /* NATION */
            static constexpr auto nation_fields = make_array_from_enum_values<NationField, kNationFieldCount>();
            static constexpr std::array<enum rec_type, kNationFieldCount> nation_field_types = {
                REC_ATTR_INT32, REC_ATTR_CHAR, REC_ATTR_INT32, REC_ATTR_VCHAR
            };
            static constexpr std::array<bool, kNationFieldCount> nation_enable_stats_columns = {
                false, false, false, false,
            };
            static constexpr std::array<size_t, 0> nation_filter_columns = no_filter_columns;
            static constexpr auto nation_varchar_field_count = count_varchar_types(nation_field_types);
            static constexpr std::array<size_t, nation_varchar_field_count> nation_varchar_field_indexes = make_varchar_field_indexes<nation_varchar_field_count>(
                nation_field_types
            );
            static constexpr std::array<size_t, kNationFieldCount> nation_field_sizes = {
                sizeof(int32_t), 25, sizeof(int32_t), 152
            };
            static constexpr std::array<std::array<size_t, 1>, kNationFieldCount> nation_arr_column_sizes = make_array_column_sizes(
                nation_field_sizes
            );
            static constexpr auto nation_field_max_sizes = make_field_max_sizes(
                nation_field_sizes, nation_field_types);
            static constexpr size_t nation_row_max_size = std::accumulate(
                nation_field_max_sizes.begin(), nation_field_max_sizes.end(), 0u);  
            static constexpr auto nation_field_dict_encoded_types = make_dict_encoded_types_array(nation_field_types);
            static constexpr auto nation_field_dict_encoded_sizes = make_field_dict_encoded_sizes(
                nation_field_sizes, nation_field_types);
            static constexpr size_t nation_row_dict_encoded_size = std::accumulate(
                nation_field_dict_encoded_sizes.begin(), nation_field_dict_encoded_sizes.end(), 0u);
            static constexpr std::array<CompressionMethod, kNationFieldCount> nation_field_compression_types {};

 
            /* REGION */
            static constexpr auto region_fields = make_array_from_enum_values<RegionField, kRegionFieldCount>();
            static constexpr std::array<enum rec_type, kRegionFieldCount> region_field_types = {
                REC_ATTR_INT32, REC_ATTR_CHAR, REC_ATTR_VCHAR
            };
            static constexpr std::array<bool, kRegionFieldCount> region_enable_stats_columns = {
                false, false, false,
            };
            static constexpr std::array<size_t, 0> region_filter_columns = no_filter_columns;
            static constexpr auto region_varchar_field_count = count_varchar_types(region_field_types);
            static constexpr auto region_varchar_field_indexes = make_varchar_field_indexes<region_varchar_field_count>(
                region_field_types);
 
            static constexpr std::array<size_t, kRegionFieldCount> region_field_sizes = {
                sizeof(int32_t), 25, 152
            };
            static constexpr std::array<std::array<size_t, 1>, kRegionFieldCount> region_arr_column_sizes = make_array_column_sizes(
                region_field_sizes
            );
            static constexpr auto region_field_max_sizes = make_field_max_sizes(
                region_field_sizes, region_field_types);
            static constexpr size_t region_row_max_size = std::accumulate(
                region_field_max_sizes.begin(), region_field_max_sizes.end(), 0u);  
            static constexpr std::array<enum rec_type, kRegionFieldCount> region_field_dict_encoded_types =
                make_dict_encoded_types_array(region_field_types);
            static constexpr auto region_field_dict_encoded_sizes = make_field_dict_encoded_sizes(
                region_field_sizes, region_field_types);
            static constexpr size_t region_row_dict_encoded_size = std::accumulate(
                region_field_dict_encoded_sizes.begin(), region_field_dict_encoded_sizes.end(), 0u);
            static constexpr std::array<CompressionMethod, kRegionFieldCount> region_field_compression_types {};
 

            static constexpr std::array<enum rec_type, 1> dict_types = {
                REC_ATTR_VCHAR
            };
            static constexpr std::array<enum rec_type, 1> col_int16_types = {
                REC_ATTR_INT16
            };
            static constexpr std::array<enum rec_type, 1> col_int32_types = {
                REC_ATTR_INT32
            };
            static constexpr std::array<enum rec_type, 1> col_int64_types = {
                REC_ATTR_INT64
            };
            static constexpr std::array<enum rec_type, 1> col_char_types = {
                REC_ATTR_CHAR
            };
            static constexpr std::array<enum rec_type, 1> col_vchar_types = {
                REC_ATTR_VCHAR
            };

            // constexpr size_t calc_num_varchar_fields(void) {
            //     return std::max({
            //         customer_varchar_field_count,
            //         lineitem_varchar_field_count,
            //         orders_varchar_field_count,
            //         supplier_varchar_field_count,
            //         partsupp_varchar_field_count,
            //         nation_varchar_field_count,
            //         region_varchar_field_count
            //     });
            // }
            static constexpr size_t max_varchar_field_count = std::max({
                customer_varchar_field_count,
                lineitem_varchar_field_count,
                orders_varchar_field_count,
                supplier_varchar_field_count,
                partsupp_varchar_field_count,
                nation_varchar_field_count,
                region_varchar_field_count
            });

        } fmt;

        //constexpr uint32_t kCustomerMaxNClustersInDCT = DCT::kNumMaxDictsInXtn  * TPCH::common::fmt.customer_varchar_field_count;
        //constexpr uint32_t kLineitemMaxNClustersInPage = DCT::kNumMaxDictsInXtn * TPCH::common::fmt.lineitem_varchar_field_count;
        //constexpr uint32_t kOrdersMaxNClustersInPage = DCT::kNumMaxDictsInXtn * TPCH::common::fmt.orders_varchar_field_count;
        //constexpr uint32_t kPartMaxNClustersInPage = DCT::kNumMaxDictsInXtn * TPCH::common::fmt.part_varchar_field_count;
        //constexpr uint32_t kSupplierMaxNClustersInPage = DCT::kNumMaxDictsInXtn * TPCH::common::fmt.supplier_varchar_field_count;
        //constexpr uint32_t kPartSuppMaxNClustersInPage = DCT::kNumMaxDictsInXtn * TPCH::common::fmt.partsupp_varchar_field_count;
        //constexpr uint32_t kNationMaxNClustersInPage = DCT::kNumMaxDictsInXtn * TPCH::common::fmt.nation_varchar_field_count;
        //constexpr uint32_t kRegionMaxNClustersInPage = DCT::kNumMaxDictsInXtn * TPCH::common::fmt.region_varchar_field_count;


        constexpr size_t max_rec_size(TPCH::common::Table table)
        {
            switch (table) {
                case TPCH::common::Table::CUSTOMER:
                    return fmt.customer_row_max_size;
                case TPCH::common::Table::LINEITEM:
                    return fmt.lineitem_row_max_size;
                //case TPCH::common::Table::ORDERS:
                //    return fmt.orders_row_max_size;
                //case TPCH::common::Table::PART:
                //    return fmt.part_row_max_size;
                //case TPCH::common::Table::SUPPLIER:
                //    return fmt.supplier_row_max_size;
                //case TPCH::common::Table::PARTSUPP:
                //    return fmt.partsupp_row_max_size;
                case TPCH::common::Table::NATION:
                    return fmt.nation_row_max_size;
                case TPCH::common::Table::REGION:
                    return fmt.region_row_max_size;
                default:
                    return 0; // Unknown table
            }
        }
    }

    namespace Query {
        namespace Check {
            namespace Nation {
                constexpr size_t SCAN_TARGET_COL = TPCH::common::N_COMMENT;
            }
            namespace Region {
                constexpr size_t SCAN_TARGET_COL = TPCH::common::R_COMMENT;
            }
            namespace Customer {
                namespace CHECK11 {
                    constexpr size_t SCAN_TARGET_COL = TPCH::common::C_NAME;
                    constexpr size_t SCAN_TARGET_COL_VARCHAR_IDX = 0;
                }
                namespace CHECK12 {
                    constexpr size_t SCAN_TARGET_COL = TPCH::common::C_ADDRESS;
                    constexpr size_t SCAN_TARGET_COL_VARCHAR_IDX = 1;
                }
                namespace CHECK13 {
                    constexpr size_t SCAN_TARGET_COL = TPCH::common::C_COMMENT;
                    constexpr size_t SCAN_TARGET_COL_VARCHAR_IDX = 2;
                }
            }
        }
#if 0
        namespace q11 {
            constexpr size_t NUM_LO_ACTIVE_FIELDS = 4;
        } // namespace Q11

        namespace q21 {
            // constexpr size_t NUM_LO_ACTIVE_FIELDS = 4;
        } // namespace Q11
#else

        namespace Q6 {
            constexpr size_t NUM_SCAN_TARGET_COLS = 4;
            constexpr std::array<size_t, NUM_SCAN_TARGET_COLS> SCAN_TARGET_COLS =
            { TPCH::common::L_SHIPDATE, TPCH::common::L_QUANTITY,
              TPCH::common::L_EXTENDEDPRICE, TPCH::common::L_DISCOUNT };
            constexpr std::array<size_t, 1> stats_columns = {TPCH::common::L_SHIPDATE};
        }

        namespace Q1 {
            constexpr size_t NUM_SCAN_TARGET_COLS = 7;
            constexpr std::array<size_t, NUM_SCAN_TARGET_COLS> SCAN_TARGET_COLS =
            { TPCH::common::L_QUANTITY, TPCH::common::L_EXTENDEDPRICE,
              TPCH::common::L_DISCOUNT, TPCH::common::L_TAX,
              TPCH::common::L_RETURNFLAG, TPCH::common::L_LINESTATUS,
              TPCH::common::L_SHIPDATE };
            constexpr std::array<size_t, 1> stats_columns = {TPCH::common::L_SHIPDATE};
        }

        namespace Q13 {
            // ORDERS table scan columns
            constexpr size_t NUM_ORDERS_SCAN_COLS = 2;
            constexpr std::array<size_t, NUM_ORDERS_SCAN_COLS> ORDERS_SCAN_COLS =
                { TPCH::common::O_CUSTKEY, TPCH::common::O_COMMENT };
            // CUSTOMER table scan columns
            constexpr size_t NUM_CUSTOMER_SCAN_COLS = 1;
            constexpr std::array<size_t, NUM_CUSTOMER_SCAN_COLS> CUSTOMER_SCAN_COLS =
                { TPCH::common::C_CUSTKEY };
        }

        namespace Q3 {
            // CUSTOMER table scan columns
            constexpr size_t NUM_CUSTOMER_SCAN_COLS = 2;
            constexpr std::array<size_t, NUM_CUSTOMER_SCAN_COLS> CUSTOMER_SCAN_COLS =
                { TPCH::common::C_CUSTKEY, TPCH::common::C_MKTSEGMENT };

            // ORDERS table scan columns
            constexpr size_t NUM_ORDERS_SCAN_COLS = 4;
            constexpr std::array<size_t, NUM_ORDERS_SCAN_COLS> ORDERS_SCAN_COLS =
                { TPCH::common::O_ORDERKEY, TPCH::common::O_CUSTKEY,
                  TPCH::common::O_ORDERDATE, TPCH::common::O_SHIPPRIORITY };

            // LINEITEM table scan columns
            constexpr size_t NUM_LINEITEM_SCAN_COLS = 4;
            constexpr std::array<size_t, NUM_LINEITEM_SCAN_COLS> LINEITEM_SCAN_COLS =
                { TPCH::common::L_ORDERKEY, TPCH::common::L_EXTENDEDPRICE,
                  TPCH::common::L_DISCOUNT, TPCH::common::L_SHIPDATE };
        }

        namespace Q5 {
            // REGION table scan columns
            constexpr size_t NUM_REGION_SCAN_COLS = 2;
            constexpr std::array<size_t, NUM_REGION_SCAN_COLS> REGION_SCAN_COLS =
                { TPCH::common::R_REGIONKEY, TPCH::common::R_NAME };
            // NATION table scan columns
            constexpr size_t NUM_NATION_SCAN_COLS = 3;
            constexpr std::array<size_t, NUM_NATION_SCAN_COLS> NATION_SCAN_COLS =
                { TPCH::common::N_NATIONKEY, TPCH::common::N_NAME, TPCH::common::N_REGIONKEY };
            // SUPPLIER table scan columns
            constexpr size_t NUM_SUPPLIER_SCAN_COLS = 2;
            constexpr std::array<size_t, NUM_SUPPLIER_SCAN_COLS> SUPPLIER_SCAN_COLS =
                { TPCH::common::S_SUPPKEY, TPCH::common::S_NATIONKEY };
            // CUSTOMER table scan columns
            constexpr size_t NUM_CUSTOMER_SCAN_COLS = 2;
            constexpr std::array<size_t, NUM_CUSTOMER_SCAN_COLS> CUSTOMER_SCAN_COLS =
                { TPCH::common::C_CUSTKEY, TPCH::common::C_NATIONKEY };
            // ORDERS table scan columns
            constexpr size_t NUM_ORDERS_SCAN_COLS = 3;
            constexpr std::array<size_t, NUM_ORDERS_SCAN_COLS> ORDERS_SCAN_COLS =
                { TPCH::common::O_ORDERKEY, TPCH::common::O_CUSTKEY, TPCH::common::O_ORDERDATE };
            // LINEITEM table scan columns
            constexpr size_t NUM_LINEITEM_SCAN_COLS = 4;
            constexpr std::array<size_t, NUM_LINEITEM_SCAN_COLS> LINEITEM_SCAN_COLS =
                { TPCH::common::L_ORDERKEY, TPCH::common::L_SUPPKEY,
                  TPCH::common::L_EXTENDEDPRICE, TPCH::common::L_DISCOUNT };
        }

        namespace Q16 {
            // SUPPLIER table scan columns (anti-join subquery)
            constexpr size_t NUM_SUPPLIER_SCAN_COLS = 2;
            constexpr std::array<size_t, NUM_SUPPLIER_SCAN_COLS> SUPPLIER_SCAN_COLS =
                { TPCH::common::S_SUPPKEY, TPCH::common::S_COMMENT };
            // PART table scan columns
            constexpr size_t NUM_PART_SCAN_COLS = 4;
            constexpr std::array<size_t, NUM_PART_SCAN_COLS> PART_SCAN_COLS =
                { TPCH::common::P_PARTKEY, TPCH::common::P_BRAND,
                  TPCH::common::P_TYPE, TPCH::common::P_SIZE };
            // PARTSUPP table scan columns
            constexpr size_t NUM_PARTSUPP_SCAN_COLS = 2;
            constexpr std::array<size_t, NUM_PARTSUPP_SCAN_COLS> PARTSUPP_SCAN_COLS =
                { TPCH::common::PS_PARTKEY, TPCH::common::PS_SUPPKEY };
        }

#endif
    }

    size_t metadata_noffsets_to_nmetapages(uint64_t noffsets, uint64_t page_size)
    {
        size_t sizrec = sizeof(int32_t);
        size_t noffsets_per_page = page_size / sizrec;
        return (noffsets + noffsets_per_page - 1) / noffsets_per_page;
    }

    size_t metadata_ncomp_pages_to_nmetapages(uint64_t ncomp_pages, uint64_t page_size)
    {
        /* same logic, so reuse it. */
        return metadata_noffsets_to_nmetapages(ncomp_pages, page_size);
    }


    size_t nbase_to_npages(uint64_t nbase, size_t page_size)
    {
        return (nbase * sizeof(uint64_t) + page_size - 1) / page_size;
    }
} // namespace TPCH

struct TPCHTableMeta {
    uint64_t npages;
    uint64_t offset;
};


// Metadata storage format
struct TPCHTableMetadata {
    uint64_t page_size;
    uint32_t lbc_num_varchar_clusters;
    /* 0 or 1 */
    uint32_t compressed;
    uint32_t column;
    // uint32_t dict_encoded;
    // uint32_t varchar_to_fixedchar;

    /* NOTE: all fields are separatedly saved only when column store is used. */
    /*  in other case, only [0] is used */
    // uint64_t table_region_stats_start_page_ids[TPCH::common::fmt.region_filter_columns.size()];
    // uint64_t table_region_stats_npages[TPCH::common::fmt.region_filter_columns.size()];
    // uint64_t table_orders_stats_start_page_ids[TPCH::common::fmt.orders_stats_columns.size()];
    // uint64_t table_orders_stats_npages[TPCH::common::fmt.orders_stats_columns.size()];
    // uint64_t table_supplier_stats_start_page_ids[TPCH::common::fmt.supplier_stats_columns.size()];
    // uint64_t table_supplier_stats_npages[TPCH::common::fmt.supplier_stats_columns.size()];
    // uint64_t table_part_stats_start_page_ids[TPCH::common::fmt.part_stats_columns.size()];
    // uint64_t table_part_stats_npages[TPCH::common::fmt.part_stats_columns.size()];
    // uint64_t table_partsupp_stats_start_page_ids[TPCH::common::fmt.partsupp_stats_columns.size()];
    // uint64_t table_partsupp_stats_npages[TPCH::common::fmt.partsupp_stats_columns.size()];

    uint64_t table_region_start_page_ids[TPCH::common::kRegionFieldCount];
    uint64_t table_region_npages[TPCH::common::kRegionFieldCount];
    uint32_t table_region_max_nrows_in_page[TPCH::common::kRegionFieldCount];
    uint32_t table_region_prefix_sum_chunk_size;
    uint64_t table_region_prefix_sum_start_page_ids[TPCH::common::kRegionFieldCount];
    uint64_t table_region_prefix_sum_npages[TPCH::common::kRegionFieldCount];
    uint64_t table_region_nstats[TPCH::common::kRegionFieldCount];
    uint64_t table_region_stats_start_page_ids[TPCH::common::kRegionFieldCount];
    uint64_t table_region_stats_npages[TPCH::common::kRegionFieldCount];
    uint32_t table_region_compression_types[TPCH::common::kRegionFieldCount];
    uint64_t table_region_compressed_page_sizes_start_page_ids[TPCH::common::kRegionFieldCount];
    uint64_t table_region_compressed_page_sizes_npages[TPCH::common::kRegionFieldCount];
    /* NOTE: nbases * 8 = number of pages required */
    uint64_t table_region_compression_nbases[TPCH::common::kRegionFieldCount];
    uint64_t table_region_compression_base_start_page_ids[TPCH::common::kRegionFieldCount];
    uint16_t table_region_compression_method[TPCH::common::kRegionFieldCount];
    uint64_t table_region_nrows;

    uint64_t table_nation_start_page_ids[TPCH::common::kNationFieldCount];
    uint64_t table_nation_npages[TPCH::common::kNationFieldCount];
    uint32_t table_nation_max_nrows_in_page[TPCH::common::kNationFieldCount];
    uint32_t table_nation_prefix_sum_chunk_size;
    uint64_t table_nation_prefix_sum_start_page_ids[TPCH::common::kNationFieldCount];
    uint64_t table_nation_prefix_sum_npages[TPCH::common::kNationFieldCount];
    uint64_t table_nation_nstats[TPCH::common::kNationFieldCount];
    uint64_t table_nation_stats_start_page_ids[TPCH::common::kNationFieldCount];
    uint64_t table_nation_stats_npages[TPCH::common::kNationFieldCount];
    uint64_t table_nation_compressed_page_sizes_start_page_ids[TPCH::common::kNationFieldCount];
    uint64_t table_nation_compressed_page_sizes_npages[TPCH::common::kNationFieldCount];
    /* NOTE: nbases * 8 = number of pages required */
    uint64_t table_nation_compression_nbases[TPCH::common::kNationFieldCount];
    uint64_t table_nation_compression_base_start_page_ids[TPCH::common::kNationFieldCount];
    uint16_t table_nation_compression_method[TPCH::common::kNationFieldCount];
    uint64_t table_nation_nrows;

    uint64_t table_customer_start_page_ids[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_npages[TPCH::common::kCustomerFieldCount];
    uint32_t table_customer_max_nrows_in_page[TPCH::common::kCustomerFieldCount];
    uint32_t table_customer_prefix_sum_chunk_size;
    uint64_t table_customer_prefix_sum_start_page_ids[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_prefix_sum_npages[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_nstats[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_stats_start_page_ids[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_stats_npages[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_compressed_page_sizes_start_page_ids[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_compressed_page_sizes_npages[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_compression_nbases[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_compression_base_start_page_ids[TPCH::common::kCustomerFieldCount];
    uint16_t table_customer_compression_method[TPCH::common::kCustomerFieldCount];
    uint64_t table_customer_nrows;

    uint64_t table_lineitem_start_page_ids[TPCH::common::kLineitemFieldCount];
    uint64_t table_lineitem_npages[TPCH::common::kLineitemFieldCount];
    uint32_t table_lineitem_max_nrows_in_page[TPCH::common::kLineitemFieldCount];
    uint32_t table_lineitem_prefix_sum_chunk_size;
    uint64_t table_lineitem_prefix_sum_start_page_ids[TPCH::common::kLineitemFieldCount];
    uint64_t table_lineitem_prefix_sum_npages[TPCH::common::kLineitemFieldCount];
    uint64_t table_lineitem_nstats[TPCH::common::kLineitemFieldCount];
    uint64_t table_lineitem_stats_start_page_ids[TPCH::common::kLineitemFieldCount];
    uint64_t table_lineitem_stats_npages[TPCH::common::kLineitemFieldCount];
    uint64_t table_lineitem_sideways_nstats[TPCH::common::kLineitemFieldCount][TPCH::common::kLineitemSidewaysCount];
    uint64_t table_lineitem_sideways_stats_npages[TPCH::common::kLineitemFieldCount][TPCH::common::kLineitemSidewaysCount];
    uint64_t table_lineitem_sideways_stats_start_page_ids[TPCH::common::kLineitemFieldCount][TPCH::common::kLineitemSidewaysCount];
    uint64_t table_lineitem_compressed_page_sizes_start_page_ids[TPCH::common::kLineitemFieldCount];
    uint64_t table_lineitem_compressed_page_sizes_npages[TPCH::common::kLineitemFieldCount];
    /* NOTE: nbases * 8 = number of pages required */
    uint64_t table_lineitem_compression_nbases[TPCH::common::kLineitemFieldCount];
    uint64_t table_lineitem_compression_base_start_page_ids[TPCH::common::kLineitemFieldCount];
    uint16_t table_lineitem_compression_method[TPCH::common::kLineitemFieldCount];
    uint64_t table_lineitem_nrows;

    uint64_t table_orders_start_page_ids[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_npages[TPCH::common::kOrdersFieldCount];
    uint32_t table_orders_max_nrows_in_page[TPCH::common::kOrdersFieldCount];
    uint32_t table_orders_prefix_sum_chunk_size;
    uint64_t table_orders_prefix_sum_start_page_ids[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_prefix_sum_npages[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_nstats[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_stats_start_page_ids[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_stats_npages[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_sideways_nstats[TPCH::common::kOrdersFieldCount][TPCH::common::kOrdersSidewaysCount];
    uint64_t table_orders_sideways_stats_npages[TPCH::common::kOrdersFieldCount][TPCH::common::kOrdersSidewaysCount];
    uint64_t table_orders_sideways_stats_start_page_ids[TPCH::common::kOrdersFieldCount][TPCH::common::kOrdersSidewaysCount];
    uint64_t table_orders_compressed_page_sizes_start_page_ids[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_compressed_page_sizes_npages[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_compression_nbases[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_compression_base_start_page_ids[TPCH::common::kOrdersFieldCount];
    uint16_t table_orders_compression_method[TPCH::common::kOrdersFieldCount];
    uint64_t table_orders_nrows;

    uint64_t table_supplier_start_page_ids[TPCH::common::kSupplierFieldCount];
    uint64_t table_supplier_npages[TPCH::common::kSupplierFieldCount];
    uint32_t table_supplier_max_nrows_in_page[TPCH::common::kSupplierFieldCount];
    uint32_t table_supplier_prefix_sum_chunk_size;
    uint64_t table_supplier_prefix_sum_start_page_ids[TPCH::common::kSupplierFieldCount];
    uint64_t table_supplier_prefix_sum_npages[TPCH::common::kSupplierFieldCount];
    uint64_t table_supplier_nstats[TPCH::common::kSupplierFieldCount];
    uint64_t table_supplier_stats_start_page_ids[TPCH::common::kSupplierFieldCount];
    uint64_t table_supplier_stats_npages[TPCH::common::kSupplierFieldCount];
    uint64_t table_supplier_compressed_page_sizes_start_page_ids[TPCH::common::kSupplierFieldCount];
    uint64_t table_supplier_compressed_page_sizes_npages[TPCH::common::kSupplierFieldCount];
    /* NOTE: nbases * 8 = number of pages required */
    uint64_t table_supplier_compression_nbases[TPCH::common::kSupplierFieldCount];
    uint64_t table_supplier_compression_base_start_page_ids[TPCH::common::kSupplierFieldCount];
    uint16_t table_supplier_compression_method[TPCH::common::kSupplierFieldCount];
    uint64_t table_supplier_nrows;

    uint64_t table_part_start_page_ids[TPCH::common::kPartFieldCount];
    uint64_t table_part_npages[TPCH::common::kPartFieldCount];
    uint32_t table_part_max_nrows_in_page[TPCH::common::kPartFieldCount];
    uint32_t table_part_prefix_sum_chunk_size;
    uint64_t table_part_prefix_sum_start_page_ids[TPCH::common::kPartFieldCount];
    uint64_t table_part_prefix_sum_npages[TPCH::common::kPartFieldCount];
    uint64_t table_part_nstats[TPCH::common::kPartFieldCount];
    uint64_t table_part_stats_start_page_ids[TPCH::common::kPartFieldCount];
    uint64_t table_part_stats_npages[TPCH::common::kPartFieldCount];
    uint64_t table_part_compressed_page_sizes_start_page_ids[TPCH::common::kPartFieldCount];
    uint64_t table_part_compressed_page_sizes_npages[TPCH::common::kPartFieldCount];
    uint64_t table_part_compression_nbases[TPCH::common::kPartFieldCount];
    uint64_t table_part_compression_base_start_page_ids[TPCH::common::kPartFieldCount];
    uint16_t table_part_compression_method[TPCH::common::kPartFieldCount];
    uint64_t table_part_nrows;

    uint64_t table_partsupp_start_page_ids[TPCH::common::kPartSuppFieldCount];
    uint64_t table_partsupp_npages[TPCH::common::kPartSuppFieldCount];
    uint32_t table_partsupp_max_nrows_in_page[TPCH::common::kPartSuppFieldCount];
    uint32_t table_partsupp_prefix_sum_chunk_size;
    uint64_t table_partsupp_prefix_sum_start_page_ids[TPCH::common::kPartSuppFieldCount];
    uint64_t table_partsupp_prefix_sum_npages[TPCH::common::kPartSuppFieldCount];
    uint64_t table_partsupp_nstats[TPCH::common::kPartSuppFieldCount];
    uint64_t table_partsupp_stats_start_page_ids[TPCH::common::kPartSuppFieldCount];
    uint64_t table_partsupp_stats_npages[TPCH::common::kPartSuppFieldCount];
    uint64_t table_partsupp_compressed_page_sizes_start_page_ids[TPCH::common::kPartSuppFieldCount];
    uint64_t table_partsupp_compressed_page_sizes_npages[TPCH::common::kPartSuppFieldCount];
    /* NOTE: nbases * 8 = number of pages required */
    uint64_t table_partsupp_compression_nbases[TPCH::common::kPartSuppFieldCount];
    uint64_t table_partsupp_compression_base_start_page_ids[TPCH::common::kPartSuppFieldCount];
    uint16_t table_partsupp_compression_method[TPCH::common::kPartSuppFieldCount];
    uint64_t table_partsupp_nrows;

    ///* head free pointer */
    uint64_t free_page_id;
    uint64_t npage_used;
    /* NOTE: storage layout */
    /* repeat of the following elements */
    /* TABLE data */
    /* nrecs prefix sum arrays (8B) per page or chunk */
    /* compressed page size (4B) per page if table is compressed */

    /* head free extent id */
    //uint64_t free_xtn_id;
    //uint64_t nxtn_used;
    
    /* TPCHTableMeta will start */
    // region, nation, customer, lineitem, orders, supplier, part, partsupp

    /* compress_meta_page */
    /* number of offset vectors */
    // uint64_t compress_table_lineorder_noffsets[TPCH::common::kLineitemFieldCount];
    // uint64_t compress_table_lineorder_offset_start_page_ids[TPCH::common::kLineitemFieldCount];

    /* The npages for compressed block sizes can be calcurated from the table_lineitem_nrows
     * and its field size */
    // uint64_t compress_table_lineitem_compressed_page_size_start_page_ids[TPCH::common::kLineitemFieldCount];

    /* temporal variable */
    /* Number of entries are initialized as XTN::MaxNumXTNs in xtn_set_constants function */
    /* 1B per entry */
    //struct xtn_entry *xtn_head;
    /* 96B per entry */
    //struct dct_meta_entry *dct_meta_head;
};

namespace TPCH {
    void metadata_print(TPCHTableMetadata &metadata)
    {
        std::cout << "page_size: " << metadata.page_size << std::endl;
        std::cout << "compressed: " << metadata.compressed << std::endl;
        std::cout << "free_xtn_id: " << metadata.free_page_id << std::endl;
        // std::cout << "compress_meta_page_id: " << metadata.compress_meta_page_id << std::endl;

        std::cout << "table_part_nrows: " << metadata.table_part_nrows << std::endl;
        #if 0
        for (int i = 0; i < TPCH::common::kNumActivePartFields; i++)
        {
            std::cout << "table_part_start_page_ids[" << i << "]: " << metadata.table_part_start_page_ids[i] << std::endl;
        }
        std::cout << "table_supplier_nrows: " << metadata.table_supplier_nrows << std::endl;
        for (int i = 0; i < TPCH::common::kNumActiveSupplierFields; i++)
        {
            std::cout << "table_supplier_start_page_ids[" << i << "]: " << metadata.table_supplier_start_page_ids[i] << std::endl;
        }
        std::cout << "table_customer_nrows: " << metadata.table_customer_nrows << std::endl;
        for (int i = 0; i < TPCH::common::kNumActiveCustomerFields; i++)
        {
            std::cout << "table_customer_start_page_ids[" << i << "]: " << metadata.table_customer_start_page_ids[i] << std::endl;
        }
        std::cout << "table_lineitem_nrows: " << metadata.table_lineitem_nrows << std::endl;
        for (int i = 0; i < TPCH::common::kNumActiveLineitemFields; i++)
        {
            std::cout << "table_lineitem_start_page_ids[" << i << "]: " << metadata.table_lineitem_start_page_ids[i] << std::endl;
        }
        if (metadata.compressed) {
            std::cout << "Compression: enabled" << std::endl;
            for (int i = 0; i < TPCH::common::kNumActiveLineitemFields; i++)
            {
                std::cout << "compress_table_lineitem_noffsets[" << i << "]: " << metadata.compress_table_lineitem_noffsets[i] << std::endl;
            }
            for (int i = 0; i < TPCH::common::kNumActiveLineitemFields; i++)
            {
                std::cout << "compress_table_lineitem_offset_start_page_ids[" << i << "]: " << metadata.compress_table_lineitem_offset_start_page_ids[i] << std::endl;
            }
        }
        #endif
    }
}

#if 0
namespace TPCH {
    namespace common {
        struct xtn_entry* metadata_get_xtn_from_xtn_id(
            TPCHTableMetadata &metadata, uint64_t xtn_id)
        {
            return &metadata.xtn_head[xtn_id];
        }
    }
}
#endif

// #if SF == 1
// #define LO_LEN 6001215
// #define P_LEN 200000
// #define S_LEN 2000
// #define C_LEN 30000
// #define D_LEN 2556
// #elif SF == 10
// #define DATA_DIR BASE_PATH "s10_fixed_columnar/"
// #define LO_LEN 59986052
// #define P_LEN 800000
// #define S_LEN 20000
// #define C_LEN 300000
// #define D_LEN 2556
// #elif SF == 100
// #define DATA_DIR BASE_PATH "s100_fixed_columnar/"
// #define LO_LEN 600037902
// #define P_LEN 1400000
// #define S_LEN 200000
// #define C_LEN 3000000
// #define D_LEN 2556
// #else // 1000
// #define DATA_DIR BASE_PATH "s1000_fixed_columnar/"
// #define LO_LEN 5999989709
// #define P_LEN 2000000
// #define S_LEN 2000000
// #define C_LEN 30000000
// #define D_LEN 2556
// #endif

// const uint64_t LO_ORDERKEY_BLOCK_START = 0;
