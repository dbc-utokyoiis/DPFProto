#pragma once

#include <variant>
#include <array>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include "common/rec.h"
#include "common/compression.cuh"

#define TABLE_NAME_LINEORDER "LINEORDER"
#define TABLE_NAME_CUSTOMER "CUSTOMER"
#define TABLE_NAME_SUPPLIER "SUPPLIER"
#define TABLE_NAME_PART "PART"
#define TABLE_NAME_DATE "DATE"


namespace SSB {
    namespace common {
        enum Table {
            LINEORDER = 0,
            CUSTOMER,
            SUPPLIER,
            PART,
            DDATE,
            TABLES
        };

        inline const char* table_name(Table table) {
            switch (table) {
                case Table::LINEORDER: return "lineorder";
                case Table::CUSTOMER: return "customer";
                case Table::SUPPLIER: return "supplier";
                case Table::PART: return "part";
                case Table::DDATE: return "date";
                default: return "unknown";
            }
        }

        enum LineOrderField {
            LO_ORDERKEY = 0,
            LO_LINENUMBER,
            LO_CUSTKEY,
            LO_PARTKEY,
            LO_SUPPKEY,
            LO_ORDERDATE,
            LO_ORDERPRIORITY,
            LO_SHIPPRIORITY,
            LO_QUANTITY,
            LO_EXTENDEDPRICE,
            LO_ORDTOTALPRICE,
            LO_DISCOUNT,
            LO_REVENUE,
            LO_SUPPLYCOST,
            LO_TAX,
            LO_COMMITDATE,
            LO_SHIPMODE,
            LO_FIELDS
        };

        enum CustomerField {
            C_CUSTKEY = 0,
            C_NAME,
            C_ADDRESS,
            C_CITY,
            C_NATION,
            C_REGION,
            C_PHONE,
            C_MKTSEGMENT,
            C_FIELDS
        };

        enum SupplierField {
            S_SUPPKEY = 0,
            S_NAME,
            S_ADDRESS,
            S_CITY,
            S_NATION,
            S_REGION,
            S_PHONE,
            S_FIELDS
        };

        enum PartField {
            P_PARTKEY = 0,
            P_NAME,
            P_MFGR,
            P_CATEGORY,
            P_BRAND1,
            P_COLOR,
            P_TYPE,
            P_SIZE,
            P_CONTAINER,
            P_FIELDS
        };

        enum DateField {
            D_DATEKEY = 0,
            D_DATE,
            D_DAYOFWEEK,
            D_MONTH,
            D_YEAR,
            D_YEARMONTHNUM,
            D_YEARMONTH,
            D_DAYNUMINWEEK,
            D_DAYNUMINMONTH,
            D_DAYNUMINYEAR,
            D_MONTHNUMINYEAR,
            D_WEEKNUMINYEAR,
            D_SELLINGSEASON,
            D_LASTDAYINWEEKFL,
            D_LASTDAYINMONTHFL,
            D_HOLIDAYFL,
            D_WEEKDAYFL,
            D_FIELDS
        };

        using Field = std::variant<LineOrderField, CustomerField, SupplierField, PartField, DateField>;

        std::string metric_field_name(const SSB::common::Field field)
        {
            return std::visit([](auto f) -> std::string {
                using T = decltype(f);
                if constexpr (std::is_same_v<T, SSB::common::LineOrderField>) {
                    switch (f) {
                    case LO_ORDERKEY:
                        return "LO_ORDERKEY";
                    case LO_LINENUMBER:
                        return "LO_LINENUMBER";
                    case LO_CUSTKEY:
                        return "LO_CUSTKEY";
                    case LO_PARTKEY:
                        return "LO_PARTKEY";
                    case LO_SUPPKEY:
                        return "LO_SUPPKEY";
                    case LO_ORDERDATE:
                        return "LO_ORDERDATE";
                    case LO_ORDERPRIORITY:
                        return "LO_ORDERPRIORITY";
                    case LO_SHIPPRIORITY:
                        return "LO_SHIPPRIORITY";
                    case LO_QUANTITY:
                        return "LO_QUANTITY";
                    case LO_EXTENDEDPRICE:
                        return "LO_EXTENDEDPRICE";
                    case LO_ORDTOTALPRICE:
                        return "LO_ORDTOTALPRICE";
                    case LO_DISCOUNT:
                        return "LO_DISCOUNT";
                    case LO_REVENUE:
                        return "LO_REVENUE";
                    case LO_SUPPLYCOST:
                        return "LO_SUPPLYCOST";
                    case LO_TAX:
                        return "LO_TAX";
                    case LO_COMMITDATE:
                        return "LO_COMMITDATE";
                    case LO_SHIPMODE:
                        return "LO_SHIPMODE";
                    default:
                        return "LO_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, SSB::common::PartField>) {
                    switch (f) {
                    case P_PARTKEY:
                        return "P_PARTKEY";
                    case P_NAME:
                        return "P_NAME";
                    case P_MFGR:
                        return "P_MFGR";
                    case P_CATEGORY:
                        return "P_CATEGORY";
                    case P_BRAND1:
                        return "P_BRAND1";
                    case P_COLOR:
                        return "P_COLOR";
                    case P_TYPE:
                        return "P_TYPE";
                    case P_SIZE:
                        return "P_SIZE";
                    case P_CONTAINER:
                        return "P_CONTAINER";
                    default:
                        return "P_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, SSB::common::SupplierField>) {
                    switch (f) {
                    case S_SUPPKEY:
                        return "S_SUPPKEY";
                    case S_NAME:
                        return "S_NAME";
                    case S_ADDRESS:
                        return "S_ADDRESS";
                    case S_CITY:
                        return "S_CITY";
                    case S_NATION:
                        return "S_NATION";
                    case S_REGION:
                        return "S_REGION";
                    case S_PHONE:
                        return "S_PHONE";
                    default:
                        return "S_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, SSB::common::CustomerField>) {
                    switch (f) {
                    case C_CUSTKEY:
                        return "C_CUSTKEY";
                    case C_NAME:
                        return "C_NAME";
                    case C_ADDRESS:
                        return "C_ADDRESS";
                    case C_CITY:
                        return "C_CITY";
                    case C_NATION:
                        return "C_NATION";
                    case C_REGION:
                        return "C_REGION";
                    case C_PHONE:
                        return "C_PHONE";
                    case C_MKTSEGMENT:
                        return "C_MKTSEGMENT";
                    default:
                        return "C_UNKNOWN";
                    }
                } else if constexpr (std::is_same_v<T, SSB::common::DateField>) {
                    switch (f) {
                    case D_DATEKEY:
                        return "D_DATEKEY";
                    case D_DATE:
                        return "D_DATE";
                    case D_DAYOFWEEK:
                        return "D_DAYOFWEEK";
                    case D_MONTH:
                        return "D_MONTH";
                    case D_YEAR:
                        return "D_YEAR";
                    case D_YEARMONTHNUM:
                        return "D_YEARMONTHNUM";
                    case D_YEARMONTH:
                        return "D_YEARMONTH";
                    case D_DAYNUMINWEEK:
                        return "D_DAYNUMINWEEK";
                    case D_DAYNUMINMONTH:
                        return "D_DAYNUMINMONTH";
                    case D_DAYNUMINYEAR:
                        return "D_DAYNUMINYEAR";
                    case D_MONTHNUMINYEAR:
                        return "D_MONTHNUMINYEAR";
                    case D_WEEKNUMINYEAR:
                        return "D_WEEKNUMINYEAR";
                    case D_SELLINGSEASON:
                        return "D_SELLINGSEASON";
                    case D_LASTDAYINWEEKFL:
                        return "D_LASTDAYINWEEKFL";
                    case D_LASTDAYINMONTHFL:
                        return "D_LASTDAYINMONTHFL";
                    case D_HOLIDAYFL:
                        return "D_HOLIDAYFL";
                    case D_WEEKDAYFL:
                        return "D_WEEKDAYFL";
                    default:
                        return "D_UNKNOWN";
                    }
                } else {
                    return "UNKNOWN";
                }
              }, field);
        }

        /* All fields are loaded for every table (no "active" subset). */

        constexpr std::array<LineOrderField, LO_FIELDS> kLineOrderFields = {
            LO_ORDERKEY, LO_LINENUMBER, LO_CUSTKEY, LO_PARTKEY, LO_SUPPKEY,
            LO_ORDERDATE, LO_ORDERPRIORITY, LO_SHIPPRIORITY, LO_QUANTITY,
            LO_EXTENDEDPRICE, LO_ORDTOTALPRICE, LO_DISCOUNT, LO_REVENUE,
            LO_SUPPLYCOST, LO_TAX, LO_COMMITDATE, LO_SHIPMODE,
        };

        constexpr std::array<CustomerField, C_FIELDS> kCustomerFields = {
            C_CUSTKEY, C_NAME, C_ADDRESS, C_CITY, C_NATION, C_REGION,
            C_PHONE, C_MKTSEGMENT,
        };

        constexpr std::array<PartField, P_FIELDS> kPartFields = {
            P_PARTKEY, P_NAME, P_MFGR, P_CATEGORY, P_BRAND1, P_COLOR,
            P_TYPE, P_SIZE, P_CONTAINER,
        };

        constexpr std::array<SupplierField, S_FIELDS> kSupplierFields = {
            S_SUPPKEY, S_NAME, S_ADDRESS, S_CITY, S_NATION, S_REGION,
            S_PHONE,
        };

        constexpr std::array<DateField, D_FIELDS> kDateFields = {
            D_DATEKEY, D_DATE, D_DAYOFWEEK, D_MONTH, D_YEAR, D_YEARMONTHNUM,
            D_YEARMONTH, D_DAYNUMINWEEK, D_DAYNUMINMONTH, D_DAYNUMINYEAR,
            D_MONTHNUMINYEAR, D_WEEKNUMINYEAR, D_SELLINGSEASON,
            D_LASTDAYINWEEKFL, D_LASTDAYINMONTHFL, D_HOLIDAYFL, D_WEEKDAYFL,
        };

        struct LineOrder {
            int64_t orderkey;
            int32_t linenumber;
            int32_t custkey;
            int32_t partkey;
            int32_t suppkey;
            int32_t orderdate;
            char    orderpriority[16]; // 
            char    shippriority[1]; //
            int32_t quantity;
            int32_t extendedprice;
            int32_t ordertotalprice;
            int32_t discount;
            int32_t revenue;
            int32_t supplycost;
            int32_t tax;
            int32_t commitdate;
            char    shipmode[10];
        };

        struct Customer {
            int32_t custkey;
            char name[26];
            char address[41];
            char city[11];
            char nation[16];
            char region[16];
            char phone[16];
            char mktsegment[11];
        };

        struct Part {
            int32_t partkey;
            char name[23];
            char mfgr[7];
            char category[8];
            char brand1[10];
            char color[11];
            char type[25];
            int32_t size;
            char container[11];
        };

        struct Supplier {
            int32_t suppkey;
            char name[26];
            char address[41];
            char city[11];
            char nation[16];
            char region[16];
            char phone[16];
        };

        struct Date {
            int32_t datekey;
            char date[11];
            char dayofweek[10];
            char month[10];
            int32_t year;
            int32_t yearmonthnum;
            char yearmonth[7];
            int32_t daynuminweek;
            int32_t daynuminmonth;
            int32_t daynuminyear;
            int32_t monthnuminyear;
            int32_t weeknuminyear;
            char sellingseason[11];
            char lastdayinweekfl[1];
            char lastdayinmonthfl[1];
            char holidayfl[1];
            char weekdayfl[1];
        };

        /* Field sizes (max content bytes, excluding NUL terminator for CHAR) */
        // LINEORDER
        constexpr size_t LO_ORDERKEY_SIZE = sizeof(int64_t);
        constexpr size_t LO_LINENUMBER_SIZE = sizeof(int32_t);
        constexpr size_t LO_CUSTKEY_SIZE = sizeof(int32_t);
        constexpr size_t LO_PARTKEY_SIZE = sizeof(int32_t);
        constexpr size_t LO_SUPPKEY_SIZE = sizeof(int32_t);
        constexpr size_t LO_ORDERDATE_SIZE = sizeof(int32_t);
        constexpr size_t LO_ORDERPRIORITY_SIZE = 15; // CHAR(15) e.g. "5-LOW"
        constexpr size_t LO_SHIPPRIORITY_SIZE = 1;   // CHAR(1) "0"
        constexpr size_t LO_QUANTITY_SIZE = sizeof(int32_t);
        constexpr size_t LO_EXTENDEDPRICE_SIZE = sizeof(int32_t);
        constexpr size_t LO_ORDTOTALPRICE_SIZE = sizeof(int32_t);
        constexpr size_t LO_DISCOUNT_SIZE = sizeof(int32_t);
        constexpr size_t LO_REVENUE_SIZE = sizeof(int32_t);
        constexpr size_t LO_SUPPLYCOST_SIZE = sizeof(int32_t);
        constexpr size_t LO_TAX_SIZE = sizeof(int32_t);
        constexpr size_t LO_COMMITDATE_SIZE = sizeof(int32_t);
        constexpr size_t LO_SHIPMODE_SIZE = 10; // CHAR(10) e.g. "TRUCK"
        // CUSTOMER
        constexpr size_t C_CUSTKEY_SIZE = sizeof(int32_t);
        constexpr size_t C_NAME_SIZE = 25;      // CHAR(25)
        constexpr size_t C_ADDRESS_SIZE = 25;    // CHAR(25) (max observed: 24)
        constexpr size_t C_CITY_SIZE = 10;       // CHAR(10)
        constexpr size_t C_NATION_SIZE = 15;     // CHAR(15) (max observed: 14)
        constexpr size_t C_REGION_SIZE = 12;     // CHAR(12) (max observed: 11)
        constexpr size_t C_PHONE_SIZE = 15;      // CHAR(15)
        constexpr size_t C_MKTSEGMENT_SIZE = 10; // CHAR(10)
        // PART
        constexpr size_t P_PARTKEY_SIZE = sizeof(int32_t);
        constexpr size_t P_NAME_SIZE = 22;       // CHAR(22) (max observed: 21)
        constexpr size_t P_MFGR_SIZE = 6;        // CHAR(6)
        constexpr size_t P_CATEGORY_SIZE = 7;     // CHAR(7)
        constexpr size_t P_BRAND1_SIZE = 9;       // CHAR(9)
        constexpr size_t P_COLOR_SIZE = 11;       // CHAR(11) (max observed: 10)
        constexpr size_t P_TYPE_SIZE = 25;        // CHAR(25)
        constexpr size_t P_SIZE_SIZE = sizeof(int32_t);
        constexpr size_t P_CONTAINER_SIZE = 10;   // CHAR(10)
        // SUPPLIER
        constexpr size_t S_SUPPKEY_SIZE = sizeof(int32_t);
        constexpr size_t S_NAME_SIZE = 25;       // CHAR(25)
        constexpr size_t S_ADDRESS_SIZE = 25;    // CHAR(25) (max observed: 24)
        constexpr size_t S_CITY_SIZE = 10;       // CHAR(10)
        constexpr size_t S_NATION_SIZE = 15;     // CHAR(15) (max observed: 14)
        constexpr size_t S_REGION_SIZE = 12;     // CHAR(12) (max observed: 11)
        constexpr size_t S_PHONE_SIZE = 15;      // CHAR(15)
        // DATE
        constexpr size_t D_DATEKEY_SIZE = sizeof(int32_t);
        constexpr size_t D_DATE_SIZE = 18;           // CHAR(18) e.g. "September 10, 1998"
        constexpr size_t D_DAYOFWEEK_SIZE = 9;       // CHAR(9) e.g. "Wednesday"
        constexpr size_t D_MONTH_SIZE = 9;           // CHAR(9) e.g. "September"
        constexpr size_t D_YEAR_SIZE = sizeof(int32_t);
        constexpr size_t D_YEARMONTHNUM_SIZE = sizeof(int32_t);
        constexpr size_t D_YEARMONTH_SIZE = 7;       // CHAR(7) e.g. "Jan1992"
        constexpr size_t D_DAYNUMINWEEK_SIZE = sizeof(int32_t);
        constexpr size_t D_DAYNUMINMONTH_SIZE = sizeof(int32_t);
        constexpr size_t D_DAYNUMINYEAR_SIZE = sizeof(int32_t);
        constexpr size_t D_MONTHNUMINYEAR_SIZE = sizeof(int32_t);
        constexpr size_t D_WEEKNUMINYEAR_SIZE = sizeof(int32_t);
        constexpr size_t D_SELLINGSEASON_SIZE = 12;  // CHAR(12) e.g. "Christmas"
        constexpr size_t D_LASTDAYINWEEKFL_SIZE = 1; // CHAR(1)
        constexpr size_t D_LASTDAYINMONTHFL_SIZE = 1;// CHAR(1)
        constexpr size_t D_HOLIDAYFL_SIZE = 1;       // CHAR(1)
        constexpr size_t D_WEEKDAYFL_SIZE = 1;       // CHAR(1)

        constexpr std::array<size_t, LO_FIELDS> kLineOrderFieldSizes = {
            LO_ORDERKEY_SIZE, LO_LINENUMBER_SIZE, LO_CUSTKEY_SIZE,
            LO_PARTKEY_SIZE, LO_SUPPKEY_SIZE, LO_ORDERDATE_SIZE,
            LO_ORDERPRIORITY_SIZE, LO_SHIPPRIORITY_SIZE, LO_QUANTITY_SIZE,
            LO_EXTENDEDPRICE_SIZE, LO_ORDTOTALPRICE_SIZE, LO_DISCOUNT_SIZE,
            LO_REVENUE_SIZE, LO_SUPPLYCOST_SIZE, LO_TAX_SIZE,
            LO_COMMITDATE_SIZE, LO_SHIPMODE_SIZE,
        };

        constexpr std::array<size_t, C_FIELDS> kCustomerFieldSizes = {
            C_CUSTKEY_SIZE, C_NAME_SIZE, C_ADDRESS_SIZE, C_CITY_SIZE,
            C_NATION_SIZE, C_REGION_SIZE, C_PHONE_SIZE, C_MKTSEGMENT_SIZE,
        };

        constexpr std::array<size_t, P_FIELDS> kPartFieldSizes = {
            P_PARTKEY_SIZE, P_NAME_SIZE, P_MFGR_SIZE, P_CATEGORY_SIZE,
            P_BRAND1_SIZE, P_COLOR_SIZE, P_TYPE_SIZE, P_SIZE_SIZE,
            P_CONTAINER_SIZE,
        };

        constexpr std::array<size_t, S_FIELDS> kSupplierFieldSizes = {
            S_SUPPKEY_SIZE, S_NAME_SIZE, S_ADDRESS_SIZE, S_CITY_SIZE,
            S_NATION_SIZE, S_REGION_SIZE, S_PHONE_SIZE,
        };

        constexpr std::array<size_t, D_FIELDS> kDateFieldSizes = {
            D_DATEKEY_SIZE, D_DATE_SIZE, D_DAYOFWEEK_SIZE, D_MONTH_SIZE,
            D_YEAR_SIZE, D_YEARMONTHNUM_SIZE, D_YEARMONTH_SIZE,
            D_DAYNUMINWEEK_SIZE, D_DAYNUMINMONTH_SIZE, D_DAYNUMINYEAR_SIZE,
            D_MONTHNUMINYEAR_SIZE, D_WEEKNUMINYEAR_SIZE, D_SELLINGSEASON_SIZE,
            D_LASTDAYINWEEKFL_SIZE, D_LASTDAYINMONTHFL_SIZE,
            D_HOLIDAYFL_SIZE, D_WEEKDAYFL_SIZE,
        };

        constexpr uint32_t kLineOrderFieldCount = LO_FIELDS;
        constexpr uint32_t kPartFieldCount = P_FIELDS;
        constexpr uint32_t kSupplierFieldCount = S_FIELDS;
        constexpr uint32_t kCustomerFieldCount = C_FIELDS;
        constexpr uint32_t kDateFieldCount = D_FIELDS;

        constexpr uint32_t SSB_MAX_NFIELDS = std::max({
            kLineOrderFieldCount,
            kPartFieldCount,
            kSupplierFieldCount,
            kCustomerFieldCount,
            kDateFieldCount,
        });

        /*
         * Dictionary encoding for sideways stats.
         * String dimension attributes are encoded to int32 IDs.
         * Values are sorted in C locale order so that dict IDs preserve sort order
         * (enabling min/max range pruning on sorted LINEORDER data).
         *
         * _for_load: unpadded strings (as they appear in CSV)
         * _for_query: padded strings (as stored in CHAR pages)
         */
        constexpr std::array<std::string_view, 5> dict_region_for_load {
            "AFRICA",      // 0
            "AMERICA",     // 1
            "ASIA",        // 2
            "EUROPE",      // 3
            "MIDDLE EAST"  // 4
        };

        constexpr std::array<std::string_view, 25> dict_nation_for_load {
            "ALGERIA",        // 0
            "ARGENTINA",      // 1
            "BRAZIL",         // 2
            "CANADA",         // 3
            "CHINA",          // 4
            "EGYPT",          // 5
            "ETHIOPIA",       // 6
            "FRANCE",         // 7
            "GERMANY",        // 8
            "INDIA",          // 9
            "INDONESIA",      // 10
            "IRAN",           // 11
            "IRAQ",           // 12
            "JAPAN",          // 13
            "JORDAN",         // 14
            "KENYA",          // 15
            "MOROCCO",        // 16
            "MOZAMBIQUE",     // 17
            "PERU",           // 18
            "ROMANIA",        // 19
            "RUSSIA",         // 20
            "SAUDI ARABIA",   // 21
            "UNITED KINGDOM", // 22
            "UNITED STATES",  // 23
            "VIETNAM"         // 24
        };

        constexpr std::array<std::string_view, 5> dict_p_mfgr_for_load {
            "MFGR#1", // 0
            "MFGR#2", // 1
            "MFGR#3", // 2
            "MFGR#4", // 3
            "MFGR#5"  // 4
        };

        constexpr std::array<std::string_view, 25> dict_p_category_for_load {
            "MFGR#11", "MFGR#12", "MFGR#13", "MFGR#14", "MFGR#15",
            "MFGR#21", "MFGR#22", "MFGR#23", "MFGR#24", "MFGR#25",
            "MFGR#31", "MFGR#32", "MFGR#33", "MFGR#34", "MFGR#35",
            "MFGR#41", "MFGR#42", "MFGR#43", "MFGR#44", "MFGR#45",
            "MFGR#51", "MFGR#52", "MFGR#53", "MFGR#54", "MFGR#55"
        };

        /* City and Brand1 dicts are generated at runtime due to their size
         * (250 cities, 1000 brands). Use ssb_build_sideways_dict_encoding_maps(). */
        constexpr size_t kNumRegions = 5;
        constexpr size_t kNumNations = 25;
        constexpr size_t kNumCitiesPerNation = 10;
        constexpr size_t kNumCities = kNumNations * kNumCitiesPerNation; // 250
        constexpr size_t kNumMfgrs = 5;
        constexpr size_t kNumCategoriesPerMfgr = 5;
        constexpr size_t kNumCategories = kNumMfgrs * kNumCategoriesPerMfgr; // 25
        constexpr size_t kNumBrandsPerCategory = 40;
        constexpr size_t kNumBrands = kNumCategories * kNumBrandsPerCategory; // 1000

        /* Number of dict encoding maps needed for sideways CHAR fields.
         * All 9 sideways fields are CHAR, so this equals kLineorderSidewaysCount. */
        constexpr size_t kSidewaysDictMapCount = 9;

        /* Sideways stats for LINEORDER pages */
        /* Per-page min/max of dimension table attributes, looked up via foreign keys.
         * Enum values match CSV column order from 02_ssb_sideways_pruning.sh
         * (cols 17-25 after 17 LINEORDER columns). */
        enum LineorderSidewaysField {
            LSS_S_REGION = 0,    // CSV col 17 — Supplier.S_REGION  (Q21,Q22,Q23,Q31,Q41,Q42)
            LSS_C_REGION,        // CSV col 18 — Customer.C_REGION  (Q31,Q41,Q42,Q43)
            LSS_P_MFGR,          // CSV col 19 — Part.P_MFGR        (Q41,Q42)
            LSS_S_NATION,        // CSV col 20 — Supplier.S_NATION  (Q32,Q43)
            LSS_C_NATION,        // CSV col 21 — Customer.C_NATION  (Q32)
            LSS_P_CATEGORY,      // CSV col 22 — Part.P_CATEGORY    (Q21,Q43)
            LSS_S_CITY,          // CSV col 23 — Supplier.S_CITY    (Q33,Q34)
            LSS_C_CITY,          // CSV col 24 — Customer.C_CITY    (Q33,Q34)
            LSS_P_BRAND1,        // CSV col 25 — Part.P_BRAND1      (Q22,Q23)
            LSS_FIELDS           // = 9
        };
        constexpr uint32_t kLineorderSidewaysCount = LSS_FIELDS;

        constexpr uint64_t kDateNumRows = 2556;

        /*
         * Build dict encoding maps for all 9 sideways CHAR fields.
         * Maps are indexed by LSS enum value (= scf index).
         * Strings are stored in `storage` for stable string_view lifetime.
         *
         * LSS_S_REGION(0) and LSS_C_REGION(1) share REGION domain.
         * LSS_S_NATION(3) and LSS_C_NATION(4) share NATION domain.
         * LSS_S_CITY(6) and LSS_C_CITY(7) share CITY domain.
         */
        inline void ssb_build_sideways_dict_encoding_maps(
            std::array<std::map<std::string, int32_t>, kSidewaysDictMapCount>& maps)
        {
            /* Helper: populate a map from a constexpr array */
            auto fill_from_array = [](std::map<std::string, int32_t>& m,
                                      auto const& arr) {
                for (size_t i = 0; i < arr.size(); i++) {
                    m[std::string(arr[i])] = static_cast<int32_t>(i);
                }
            };

            /* REGION: LSS_S_REGION(0), LSS_C_REGION(1) */
            fill_from_array(maps[LSS_S_REGION], dict_region_for_load);
            fill_from_array(maps[LSS_C_REGION], dict_region_for_load);

            /* P_MFGR: LSS_P_MFGR(2) */
            fill_from_array(maps[LSS_P_MFGR], dict_p_mfgr_for_load);

            /* NATION: LSS_S_NATION(3), LSS_C_NATION(4) */
            fill_from_array(maps[LSS_S_NATION], dict_nation_for_load);
            fill_from_array(maps[LSS_C_NATION], dict_nation_for_load);

            /* P_CATEGORY: LSS_P_CATEGORY(5) */
            fill_from_array(maps[LSS_P_CATEGORY], dict_p_category_for_load);

            /* CITY: generate 250 cities, sort in C locale, assign IDs.
             * Format: first 9 chars of nation (space-padded) + digit 0-9 */
            {
                std::vector<std::string> cities;
                cities.reserve(kNumCities);
                for (auto& nation : dict_nation_for_load) {
                    std::string base(nation);
                    if (base.size() < 9) base.resize(9, ' ');
                    else if (base.size() > 9) base.resize(9);
                    for (int d = 0; d < 10; d++) {
                        cities.push_back(base + std::to_string(d));
                    }
                }
                std::sort(cities.begin(), cities.end());
                /* LSS_S_CITY(6), LSS_C_CITY(7) */
                for (size_t i = 0; i < cities.size(); i++) {
                    maps[LSS_S_CITY][cities[i]] = static_cast<int32_t>(i);
                    maps[LSS_C_CITY][cities[i]] = static_cast<int32_t>(i);
                }
            }

            /* BRAND1: generate 1000 brands, sort in C locale, assign IDs.
             * Format: "MFGR#" + mfgr_digit + category_digit + brand_number(1-40) */
            {
                std::vector<std::string> brands;
                brands.reserve(kNumBrands);
                for (int m = 1; m <= 5; m++) {
                    for (int c = 1; c <= 5; c++) {
                        for (int b = 1; b <= 40; b++) {
                            brands.push_back("MFGR#" + std::to_string(m)
                                             + std::to_string(c) + std::to_string(b));
                        }
                    }
                }
                std::sort(brands.begin(), brands.end());
                /* LSS_P_BRAND1(8) */
                for (size_t i = 0; i < brands.size(); i++) {
                    maps[LSS_P_BRAND1][brands[i]] = static_cast<int32_t>(i);
                }
            }
        }

        /*
         * recfmt_t: Record format descriptors for SSB tables.
         *
         * Describes the STORAGE types (after dictionary encoding).
         * All active fields are INT32/INT64 — no VCHAR/CHAR at storage level.
         * String-to-integer dictionary encoding is handled by the CSV parser in the loader.
         *
         * NOTE: These arrays describe all fields (k*FieldCount),
         * NOT the full CSV schema (kLineOrderFieldCount etc.).
         * The loader's CSV parser handles the full schema separately.
         */
        /* Helper: convert active field enum array to CSV column index array */
        template <typename FieldEnum, size_t N>
        static constexpr std::array<size_t, N> make_csv_column_map(const std::array<FieldEnum, N>& active_fields) {
            std::array<size_t, N> map{};
            for (size_t i = 0; i < N; i++) {
                map[i] = static_cast<size_t>(active_fields[i]);
            }
            return map;
        }

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

        struct recfmt_t {
            static constexpr std::array<size_t, 0> no_filter_columns = {};

            /* LINEORDER: all 17 fields */
            static constexpr auto lineorder_fields = kLineOrderFields;
            static constexpr auto lineorder_csv_column_map = make_csv_column_map(kLineOrderFields);
            static constexpr std::array<enum rec_type, LO_FIELDS> lineorder_field_types = {
                REC_ATTR_INT64, // LO_ORDERKEY
                REC_ATTR_INT32, // LO_LINENUMBER
                REC_ATTR_INT32, // LO_CUSTKEY
                REC_ATTR_INT32, // LO_PARTKEY
                REC_ATTR_INT32, // LO_SUPPKEY
                REC_ATTR_INT32, // LO_ORDERDATE
                REC_ATTR_CHAR,  // LO_ORDERPRIORITY
                REC_ATTR_CHAR,  // LO_SHIPPRIORITY
                REC_ATTR_INT32, // LO_QUANTITY
                REC_ATTR_INT32, // LO_EXTENDEDPRICE
                REC_ATTR_INT32, // LO_ORDTOTALPRICE
                REC_ATTR_INT32, // LO_DISCOUNT
                REC_ATTR_INT32, // LO_REVENUE
                REC_ATTR_INT32, // LO_SUPPLYCOST
                REC_ATTR_INT32, // LO_TAX
                REC_ATTR_INT32, // LO_COMMITDATE
                REC_ATTR_CHAR,  // LO_SHIPMODE
            };
            static constexpr auto lineorder_field_sizes = kLineOrderFieldSizes;
            /* Zone map: LO_ORDERDATE, LO_QUANTITY, LO_DISCOUNT */
            static constexpr std::array<bool, LO_FIELDS> lineorder_enable_stats_columns = {
                false, // LO_ORDERKEY
                false, // LO_LINENUMBER
                false, // LO_CUSTKEY
                false, // LO_PARTKEY
                false, // LO_SUPPKEY
                true,  // LO_ORDERDATE   (Date pruning)
                false, // LO_ORDERPRIORITY
                false, // LO_SHIPPRIORITY
                true,  // LO_QUANTITY     (Flight 1 predicates)
                false, // LO_EXTENDEDPRICE
                false, // LO_ORDTOTALPRICE
                true,  // LO_DISCOUNT     (Flight 1 predicates)
                false, // LO_REVENUE
                false, // LO_SUPPLYCOST
                false, // LO_TAX
                false, // LO_COMMITDATE
                false, // LO_SHIPMODE
            };
            static constexpr std::array<size_t, 3> lineorder_filter_columns = {
                LO_ORDERDATE, LO_QUANTITY, LO_DISCOUNT
            };
            static constexpr size_t lineorder_varchar_field_count = count_varchar_types(lineorder_field_types);
            static constexpr std::array<size_t, lineorder_varchar_field_count> lineorder_varchar_field_indexes = make_varchar_field_indexes<lineorder_varchar_field_count>(lineorder_field_types);

            /* Sideways stats: CHAR fields dict-encoded to INT32 at load time. */
            static constexpr std::array<enum rec_type, kLineorderSidewaysCount> lineorder_sideways_information_types = {
                REC_ATTR_CHAR, // LSS_S_REGION
                REC_ATTR_CHAR, // LSS_C_REGION
                REC_ATTR_CHAR, // LSS_P_MFGR
                REC_ATTR_CHAR, // LSS_S_NATION
                REC_ATTR_CHAR, // LSS_C_NATION
                REC_ATTR_CHAR, // LSS_P_CATEGORY
                REC_ATTR_CHAR, // LSS_S_CITY
                REC_ATTR_CHAR, // LSS_C_CITY
                REC_ATTR_CHAR, // LSS_P_BRAND1
            };
            static constexpr std::array<size_t, kLineorderSidewaysCount> lineorder_sideways_information_sizes = {
                S_REGION_SIZE, C_REGION_SIZE, P_MFGR_SIZE,
                S_NATION_SIZE, C_NATION_SIZE, P_CATEGORY_SIZE,
                S_CITY_SIZE, C_CITY_SIZE, P_BRAND1_SIZE,
            };

            /* Compression types (INT→PFOR, CHAR→LZ4; golap mode overrides) */
            static constexpr std::array<CompressionMethod, LO_FIELDS> lineorder_field_compression_types = {
                CompressionMethod::PFOR64, // LO_ORDERKEY (INT64)
                CompressionMethod::PFOR,   // LO_LINENUMBER
                CompressionMethod::PFOR,   // LO_CUSTKEY
                CompressionMethod::PFOR,   // LO_PARTKEY
                CompressionMethod::PFOR,   // LO_SUPPKEY
                CompressionMethod::PFOR,   // LO_ORDERDATE
                CompressionMethod::LZ4,    // LO_ORDERPRIORITY (CHAR)
                CompressionMethod::LZ4,    // LO_SHIPPRIORITY (CHAR)
                CompressionMethod::PFOR,   // LO_QUANTITY
                CompressionMethod::PFOR,   // LO_EXTENDEDPRICE
                CompressionMethod::PFOR,   // LO_ORDTOTALPRICE
                CompressionMethod::PFOR,   // LO_DISCOUNT
                CompressionMethod::PFOR,   // LO_REVENUE
                CompressionMethod::PFOR,   // LO_SUPPLYCOST
                CompressionMethod::PFOR,   // LO_TAX
                CompressionMethod::PFOR,   // LO_COMMITDATE
                CompressionMethod::LZ4,    // LO_SHIPMODE (CHAR)
            };

            /* CUSTOMER: all 8 fields */
            static constexpr auto customer_fields = kCustomerFields;
            static constexpr auto customer_csv_column_map = make_csv_column_map(kCustomerFields);
            static constexpr std::array<enum rec_type, C_FIELDS> customer_field_types = {
                REC_ATTR_INT32, // C_CUSTKEY
                REC_ATTR_CHAR,  // C_NAME
                REC_ATTR_CHAR,  // C_ADDRESS
                REC_ATTR_CHAR,  // C_CITY
                REC_ATTR_CHAR,  // C_NATION
                REC_ATTR_CHAR,  // C_REGION
                REC_ATTR_CHAR,  // C_PHONE
                REC_ATTR_CHAR,  // C_MKTSEGMENT
            };
            static constexpr auto customer_field_sizes = kCustomerFieldSizes;
            static constexpr std::array<bool, C_FIELDS> customer_enable_stats_columns = {
                false, false, false, false, false, false, false, false,
            };
            static constexpr std::array<size_t, 0> customer_filter_columns = no_filter_columns;
            static constexpr size_t customer_varchar_field_count = count_varchar_types(customer_field_types);
            static constexpr std::array<size_t, customer_varchar_field_count> customer_varchar_field_indexes = make_varchar_field_indexes<customer_varchar_field_count>(customer_field_types);
            static constexpr std::array<CompressionMethod, C_FIELDS> customer_field_compression_types = {
                CompressionMethod::PFOR, // C_CUSTKEY
                CompressionMethod::LZ4,  // C_NAME
                CompressionMethod::LZ4,  // C_ADDRESS
                CompressionMethod::LZ4,  // C_CITY
                CompressionMethod::LZ4,  // C_NATION
                CompressionMethod::LZ4,  // C_REGION
                CompressionMethod::LZ4,  // C_PHONE
                CompressionMethod::LZ4,  // C_MKTSEGMENT
            };

            /* SUPPLIER: all 7 fields */
            static constexpr auto supplier_fields = kSupplierFields;
            static constexpr auto supplier_csv_column_map = make_csv_column_map(kSupplierFields);
            static constexpr std::array<enum rec_type, S_FIELDS> supplier_field_types = {
                REC_ATTR_INT32, // S_SUPPKEY
                REC_ATTR_CHAR,  // S_NAME
                REC_ATTR_CHAR,  // S_ADDRESS
                REC_ATTR_CHAR,  // S_CITY
                REC_ATTR_CHAR,  // S_NATION
                REC_ATTR_CHAR,  // S_REGION
                REC_ATTR_CHAR,  // S_PHONE
            };
            static constexpr auto supplier_field_sizes = kSupplierFieldSizes;
            static constexpr std::array<bool, S_FIELDS> supplier_enable_stats_columns = {
                false, false, false, false, false, false, false,
            };
            static constexpr std::array<size_t, 0> supplier_filter_columns = no_filter_columns;
            static constexpr size_t supplier_varchar_field_count = count_varchar_types(supplier_field_types);
            static constexpr std::array<size_t, supplier_varchar_field_count> supplier_varchar_field_indexes = make_varchar_field_indexes<supplier_varchar_field_count>(supplier_field_types);
            static constexpr std::array<CompressionMethod, S_FIELDS> supplier_field_compression_types = {
                CompressionMethod::PFOR, // S_SUPPKEY
                CompressionMethod::LZ4,  // S_NAME
                CompressionMethod::LZ4,  // S_ADDRESS
                CompressionMethod::LZ4,  // S_CITY
                CompressionMethod::LZ4,  // S_NATION
                CompressionMethod::LZ4,  // S_REGION
                CompressionMethod::LZ4,  // S_PHONE
            };

            /* PART: all 9 fields */
            static constexpr auto part_fields = kPartFields;
            static constexpr auto part_csv_column_map = make_csv_column_map(kPartFields);
            static constexpr std::array<enum rec_type, P_FIELDS> part_field_types = {
                REC_ATTR_INT32, // P_PARTKEY
                REC_ATTR_CHAR,  // P_NAME
                REC_ATTR_CHAR,  // P_MFGR
                REC_ATTR_CHAR,  // P_CATEGORY
                REC_ATTR_CHAR,  // P_BRAND1
                REC_ATTR_CHAR,  // P_COLOR
                REC_ATTR_CHAR,  // P_TYPE
                REC_ATTR_INT32, // P_SIZE
                REC_ATTR_CHAR,  // P_CONTAINER
            };
            static constexpr auto part_field_sizes = kPartFieldSizes;
            static constexpr std::array<bool, P_FIELDS> part_enable_stats_columns = {
                false, false, false, false, false, false, false, false, false,
            };
            static constexpr std::array<size_t, 0> part_filter_columns = no_filter_columns;
            static constexpr size_t part_varchar_field_count = count_varchar_types(part_field_types);
            static constexpr std::array<size_t, part_varchar_field_count> part_varchar_field_indexes = make_varchar_field_indexes<part_varchar_field_count>(part_field_types);
            static constexpr std::array<CompressionMethod, P_FIELDS> part_field_compression_types = {
                CompressionMethod::PFOR, // P_PARTKEY
                CompressionMethod::LZ4,  // P_NAME
                CompressionMethod::LZ4,  // P_MFGR
                CompressionMethod::LZ4,  // P_CATEGORY
                CompressionMethod::LZ4,  // P_BRAND1
                CompressionMethod::LZ4,  // P_COLOR
                CompressionMethod::LZ4,  // P_TYPE
                CompressionMethod::PFOR, // P_SIZE (INT32)
                CompressionMethod::LZ4,  // P_CONTAINER
            };

            /* DATE: all 17 fields (INT32 + CHAR) */
            static constexpr auto date_fields = kDateFields;
            static constexpr auto date_csv_column_map = make_csv_column_map(kDateFields);
            static constexpr std::array<enum rec_type, D_FIELDS> date_field_types = {
                REC_ATTR_INT32, // D_DATEKEY
                REC_ATTR_CHAR,  // D_DATE
                REC_ATTR_CHAR,  // D_DAYOFWEEK
                REC_ATTR_CHAR,  // D_MONTH
                REC_ATTR_INT32, // D_YEAR
                REC_ATTR_INT32, // D_YEARMONTHNUM
                REC_ATTR_CHAR,  // D_YEARMONTH
                REC_ATTR_INT32, // D_DAYNUMINWEEK
                REC_ATTR_INT32, // D_DAYNUMINMONTH
                REC_ATTR_INT32, // D_DAYNUMINYEAR
                REC_ATTR_INT32, // D_MONTHNUMINYEAR
                REC_ATTR_INT32, // D_WEEKNUMINYEAR
                REC_ATTR_CHAR,  // D_SELLINGSEASON
                REC_ATTR_CHAR,  // D_LASTDAYINWEEKFL
                REC_ATTR_CHAR,  // D_LASTDAYINMONTHFL
                REC_ATTR_CHAR,  // D_HOLIDAYFL
                REC_ATTR_CHAR,  // D_WEEKDAYFL
            };
            static constexpr auto date_field_sizes = kDateFieldSizes;
            static constexpr std::array<bool, D_FIELDS> date_enable_stats_columns = {
                false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false,
            };
            static constexpr std::array<size_t, 0> date_filter_columns = no_filter_columns;
            static constexpr size_t date_varchar_field_count = count_varchar_types(date_field_types);
            static constexpr std::array<size_t, date_varchar_field_count> date_varchar_field_indexes = make_varchar_field_indexes<date_varchar_field_count>(date_field_types);
            // DATE is small (2556 rows, 1 page/field) — store uncompressed like TPC-H NATION/REGION
            static constexpr std::array<CompressionMethod, D_FIELDS> date_field_compression_types = {
                CompressionMethod::NONE, // D_DATEKEY
                CompressionMethod::NONE, // D_DATE
                CompressionMethod::NONE, // D_DAYOFWEEK
                CompressionMethod::NONE, // D_MONTH
                CompressionMethod::NONE, // D_YEAR
                CompressionMethod::NONE, // D_YEARMONTHNUM
                CompressionMethod::NONE, // D_YEARMONTH
                CompressionMethod::NONE, // D_DAYNUMINWEEK
                CompressionMethod::NONE, // D_DAYNUMINMONTH
                CompressionMethod::NONE, // D_DAYNUMINYEAR
                CompressionMethod::NONE, // D_MONTHNUMINYEAR
                CompressionMethod::NONE, // D_WEEKNUMINYEAR
                CompressionMethod::NONE, // D_SELLINGSEASON
                CompressionMethod::NONE, // D_LASTDAYINWEEKFL
                CompressionMethod::NONE, // D_LASTDAYINMONTHFL
                CompressionMethod::NONE, // D_HOLIDAYFL
                CompressionMethod::NONE, // D_WEEKDAYFL
            };
        } fmt;
    }
    namespace sf1 {
        const uint64_t kLineOrderNumRows = 6001215;
        const uint64_t kPartNumRows = 200000;
        const uint64_t kSupplierNumRows = 2000;
        const uint64_t kCustomerNumRows = 30000;
        // const uint64_t kDateLen = 2556;
    }
    namespace sf10 {
        const uint64_t kLineOrderLen = 59986052;
        const uint64_t kPartNumRows = 800000;
        const uint64_t kSupplierNumRows = 20000;
        const uint64_t kCustomerNumRows = 300000;
        // const uint64_t kDateLen = 2556;
    }
    namespace sf100 {
        const uint64_t kLineOrderNumRows = 600037902;
        const uint64_t kPartNumRows = 1400000;
        const uint64_t kSupplierNumRows = 200000;
        const uint64_t kCustomerNumRows = 3000000;
        // const uint64_t kDateLen = 2556;
    }
    namespace sf1000 {
        const uint64_t kLineOrderNumRows = 5999989709;
        const uint64_t kPartNumRows = 2000000;
        const uint64_t kSupplierNumRows = 2000000;
        const uint64_t kCustomerNumRows = 30000000;
        // const uint64_t kDateLen = 2556;
    }

    size_t loader_get_table_num_rows(common::Table table, int scale_factor) {
        if (table == common::Table::DDATE) return common::kDateNumRows;
        switch (scale_factor) {
            case 1:
                if (table == common::Table::LINEORDER) return sf1::kLineOrderNumRows;
                if (table == common::Table::PART) return sf1::kPartNumRows;
                if (table == common::Table::SUPPLIER) return sf1::kSupplierNumRows;
                if (table == common::Table::CUSTOMER) return sf1::kCustomerNumRows;
                break;
            case 10:
                if (table == common::Table::LINEORDER) return sf10::kLineOrderLen;
                if (table == common::Table::PART) return sf10::kPartNumRows;
                if (table == common::Table::SUPPLIER) return sf10::kSupplierNumRows;
                if (table == common::Table::CUSTOMER) return sf10::kCustomerNumRows;
                break;
            case 100:
                if (table == common::Table::LINEORDER) return sf100::kLineOrderNumRows;
                if (table == common::Table::PART) return sf100::kPartNumRows;
                if (table == common::Table::SUPPLIER) return sf100::kSupplierNumRows;
                if (table == common::Table::CUSTOMER) return sf100::kCustomerNumRows;
                break;
            case 1000:
                if (table == common::Table::LINEORDER) return sf1000::kLineOrderNumRows;
                if (table == common::Table::PART) return sf1000::kPartNumRows;
                if (table == common::Table::SUPPLIER) return sf1000::kSupplierNumRows;
                if (table == common::Table::CUSTOMER) return sf1000::kCustomerNumRows;
                break;
            default:
                break;
        }
        return 0; // error
    }

    namespace query {
        using namespace common;

        // Flight 1: LINEORDER-only scan + DATE lookup
        // Q1.1-Q1.3 use the same 4 LO fields
        namespace q1x {
            constexpr size_t NUM_LO_ACTIVE_FIELDS = 4;
            constexpr std::array<common::LineOrderField, NUM_LO_ACTIVE_FIELDS> LO_FIELDS = {
                LO_ORDERDATE, LO_QUANTITY, LO_DISCOUNT, LO_EXTENDEDPRICE,
            };
            constexpr size_t NUM_DATE_ACTIVE_FIELDS = 3;
            constexpr std::array<common::DateField, NUM_DATE_ACTIVE_FIELDS> DATE_FIELDS = {
                D_DATEKEY, D_YEAR, D_YEARMONTHNUM,
            };
        }
        namespace q11 {
            constexpr size_t NUM_LO_ACTIVE_FIELDS = q1x::NUM_LO_ACTIVE_FIELDS;
        }

        // Flight 2: LO + DATE + PART + SUPPLIER
        // Q2.1-Q2.3: group-by (d_year, p_brand1), SUM(lo_revenue)
        namespace q2x {
            constexpr size_t NUM_LO_ACTIVE_FIELDS = 4;
            constexpr std::array<common::LineOrderField, NUM_LO_ACTIVE_FIELDS> LO_FIELDS = {
                LO_ORDERDATE, LO_PARTKEY, LO_SUPPKEY, LO_REVENUE,
            };
            constexpr size_t NUM_DATE_ACTIVE_FIELDS = 2;
            constexpr std::array<common::DateField, NUM_DATE_ACTIVE_FIELDS> DATE_FIELDS = {
                D_DATEKEY, D_YEAR,
            };
            constexpr size_t NUM_SUPPLIER_ACTIVE_FIELDS = 2;
            constexpr std::array<common::SupplierField, NUM_SUPPLIER_ACTIVE_FIELDS> SUPPLIER_FIELDS = {
                S_SUPPKEY, S_REGION,
            };
            constexpr size_t NUM_PART_ACTIVE_FIELDS = 4;
            constexpr std::array<common::PartField, NUM_PART_ACTIVE_FIELDS> PART_FIELDS = {
                P_PARTKEY, P_MFGR, P_CATEGORY, P_BRAND1,
            };
        }

        // Flight 3: LO + DATE + CUSTOMER + SUPPLIER
        // Q3.1-Q3.4: group-by (c_nation/city, s_nation/city, d_year), SUM(lo_revenue)
        namespace q3x {
            constexpr size_t NUM_LO_ACTIVE_FIELDS = 4;
            constexpr std::array<common::LineOrderField, NUM_LO_ACTIVE_FIELDS> LO_FIELDS = {
                LO_ORDERDATE, LO_CUSTKEY, LO_SUPPKEY, LO_REVENUE,
            };
            constexpr size_t NUM_DATE_ACTIVE_FIELDS = 2;
            constexpr std::array<common::DateField, NUM_DATE_ACTIVE_FIELDS> DATE_FIELDS = {
                D_DATEKEY, D_YEAR,
            };
            constexpr size_t NUM_CUSTOMER_ACTIVE_FIELDS = 4;
            constexpr std::array<common::CustomerField, NUM_CUSTOMER_ACTIVE_FIELDS> CUSTOMER_FIELDS = {
                C_CUSTKEY, C_CITY, C_NATION, C_REGION,
            };
            constexpr size_t NUM_SUPPLIER_ACTIVE_FIELDS = 4;
            constexpr std::array<common::SupplierField, NUM_SUPPLIER_ACTIVE_FIELDS> SUPPLIER_FIELDS = {
                S_SUPPKEY, S_CITY, S_NATION, S_REGION,
            };
        }

        // Flight 4: LO + DATE + CUSTOMER + SUPPLIER + PART
        // Q4.1-Q4.3: group-by (d_year, c_nation/s_nation/s_city, p_category/p_brand1),
        //            SUM(lo_revenue - lo_supplycost)
        namespace q4x {
            constexpr size_t NUM_LO_ACTIVE_FIELDS = 6;
            constexpr std::array<common::LineOrderField, NUM_LO_ACTIVE_FIELDS> LO_FIELDS = {
                LO_ORDERDATE, LO_CUSTKEY, LO_PARTKEY, LO_SUPPKEY, LO_REVENUE, LO_SUPPLYCOST,
            };
            constexpr size_t NUM_DATE_ACTIVE_FIELDS = 2;
            constexpr std::array<common::DateField, NUM_DATE_ACTIVE_FIELDS> DATE_FIELDS = {
                D_DATEKEY, D_YEAR,
            };
            constexpr size_t NUM_CUSTOMER_ACTIVE_FIELDS = 4;
            constexpr std::array<common::CustomerField, NUM_CUSTOMER_ACTIVE_FIELDS> CUSTOMER_FIELDS = {
                C_CUSTKEY, C_CITY, C_NATION, C_REGION,
            };
            constexpr size_t NUM_SUPPLIER_ACTIVE_FIELDS = 4;
            constexpr std::array<common::SupplierField, NUM_SUPPLIER_ACTIVE_FIELDS> SUPPLIER_FIELDS = {
                S_SUPPKEY, S_CITY, S_NATION, S_REGION,
            };
            constexpr size_t NUM_PART_ACTIVE_FIELDS = 4;
            constexpr std::array<common::PartField, NUM_PART_ACTIVE_FIELDS> PART_FIELDS = {
                P_PARTKEY, P_MFGR, P_CATEGORY, P_BRAND1,
            };
        }
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
} // namespace SSB

// Metadata storage format (v2 — follows TPCHTableMetadata pattern)
struct SSBTableMetadata {
    uint64_t page_size;
    uint32_t lbc_num_varchar_clusters; /* for PiG LbC mode */
    uint32_t compressed; /* 0 or 1 */
    uint32_t column;     /* column-store flag */

    /* ===== DATE table ===== */
    uint64_t table_date_start_page_ids[SSB::common::kDateFieldCount];
    uint64_t table_date_npages[SSB::common::kDateFieldCount];
    uint32_t table_date_max_nrows_in_page[SSB::common::kDateFieldCount];
    uint32_t table_date_prefix_sum_chunk_size;
    uint64_t table_date_prefix_sum_start_page_ids[SSB::common::kDateFieldCount];
    uint64_t table_date_prefix_sum_npages[SSB::common::kDateFieldCount];
    uint64_t table_date_nstats[SSB::common::kDateFieldCount];
    uint64_t table_date_stats_start_page_ids[SSB::common::kDateFieldCount];
    uint64_t table_date_stats_npages[SSB::common::kDateFieldCount];
    uint64_t table_date_compressed_page_sizes_start_page_ids[SSB::common::kDateFieldCount];
    uint64_t table_date_compressed_page_sizes_npages[SSB::common::kDateFieldCount];
    uint64_t table_date_compression_nbases[SSB::common::kDateFieldCount];
    uint64_t table_date_compression_base_start_page_ids[SSB::common::kDateFieldCount];
    uint16_t table_date_compression_method[SSB::common::kDateFieldCount];
    uint64_t table_date_nrows;

    /* ===== CUSTOMER table ===== */
    uint64_t table_customer_start_page_ids[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_npages[SSB::common::kCustomerFieldCount];
    uint32_t table_customer_max_nrows_in_page[SSB::common::kCustomerFieldCount];
    uint32_t table_customer_prefix_sum_chunk_size;
    uint64_t table_customer_prefix_sum_start_page_ids[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_prefix_sum_npages[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_nstats[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_stats_start_page_ids[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_stats_npages[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_compressed_page_sizes_start_page_ids[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_compressed_page_sizes_npages[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_compression_nbases[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_compression_base_start_page_ids[SSB::common::kCustomerFieldCount];
    uint16_t table_customer_compression_method[SSB::common::kCustomerFieldCount];
    uint64_t table_customer_nrows;

    /* ===== SUPPLIER table ===== */
    uint64_t table_supplier_start_page_ids[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_npages[SSB::common::kSupplierFieldCount];
    uint32_t table_supplier_max_nrows_in_page[SSB::common::kSupplierFieldCount];
    uint32_t table_supplier_prefix_sum_chunk_size;
    uint64_t table_supplier_prefix_sum_start_page_ids[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_prefix_sum_npages[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_nstats[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_stats_start_page_ids[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_stats_npages[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_compressed_page_sizes_start_page_ids[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_compressed_page_sizes_npages[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_compression_nbases[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_compression_base_start_page_ids[SSB::common::kSupplierFieldCount];
    uint16_t table_supplier_compression_method[SSB::common::kSupplierFieldCount];
    uint64_t table_supplier_nrows;

    /* ===== PART table ===== */
    uint64_t table_part_start_page_ids[SSB::common::kPartFieldCount];
    uint64_t table_part_npages[SSB::common::kPartFieldCount];
    uint32_t table_part_max_nrows_in_page[SSB::common::kPartFieldCount];
    uint32_t table_part_prefix_sum_chunk_size;
    uint64_t table_part_prefix_sum_start_page_ids[SSB::common::kPartFieldCount];
    uint64_t table_part_prefix_sum_npages[SSB::common::kPartFieldCount];
    uint64_t table_part_nstats[SSB::common::kPartFieldCount];
    uint64_t table_part_stats_start_page_ids[SSB::common::kPartFieldCount];
    uint64_t table_part_stats_npages[SSB::common::kPartFieldCount];
    uint64_t table_part_compressed_page_sizes_start_page_ids[SSB::common::kPartFieldCount];
    uint64_t table_part_compressed_page_sizes_npages[SSB::common::kPartFieldCount];
    uint64_t table_part_compression_nbases[SSB::common::kPartFieldCount];
    uint64_t table_part_compression_base_start_page_ids[SSB::common::kPartFieldCount];
    uint16_t table_part_compression_method[SSB::common::kPartFieldCount];
    uint64_t table_part_nrows;

    /* ===== LINEORDER table (fact table) ===== */
    uint64_t table_lineorder_start_page_ids[SSB::common::kLineOrderFieldCount];
    uint64_t table_lineorder_npages[SSB::common::kLineOrderFieldCount];
    uint32_t table_lineorder_max_nrows_in_page[SSB::common::kLineOrderFieldCount];
    uint32_t table_lineorder_prefix_sum_chunk_size;
    uint64_t table_lineorder_prefix_sum_start_page_ids[SSB::common::kLineOrderFieldCount];
    uint64_t table_lineorder_prefix_sum_npages[SSB::common::kLineOrderFieldCount];
    /* Zone map stats */
    uint64_t table_lineorder_nstats[SSB::common::kLineOrderFieldCount];
    uint64_t table_lineorder_stats_start_page_ids[SSB::common::kLineOrderFieldCount];
    uint64_t table_lineorder_stats_npages[SSB::common::kLineOrderFieldCount];
    /* Sideways stats: per-page min/max of dimension attributes */
    uint64_t table_lineorder_sideways_nstats[SSB::common::kLineOrderFieldCount][SSB::common::kLineorderSidewaysCount];
    uint64_t table_lineorder_sideways_stats_npages[SSB::common::kLineOrderFieldCount][SSB::common::kLineorderSidewaysCount];
    uint64_t table_lineorder_sideways_stats_start_page_ids[SSB::common::kLineOrderFieldCount][SSB::common::kLineorderSidewaysCount];
    /* Compression metadata */
    uint64_t table_lineorder_compressed_page_sizes_start_page_ids[SSB::common::kLineOrderFieldCount];
    uint64_t table_lineorder_compressed_page_sizes_npages[SSB::common::kLineOrderFieldCount];
    uint64_t table_lineorder_compression_nbases[SSB::common::kLineOrderFieldCount];
    uint64_t table_lineorder_compression_base_start_page_ids[SSB::common::kLineOrderFieldCount];
    uint16_t table_lineorder_compression_method[SSB::common::kLineOrderFieldCount];
    uint64_t table_lineorder_nrows;

    /* ===== Global metadata ===== */
    uint64_t free_page_id;
    uint64_t npage_used;
    uint32_t scale_factor; /* for backward compatibility with old loader */

    /* ===== Legacy fields for Q11 backward compatibility ===== */
    /* These duplicate information in the new fields above.
     * Q11 device_compress.cu and host_compress.cu reference them directly. */
    uint64_t compress_table_lineorder_noffsets[SSB::common::kLineOrderFieldCount];
    uint64_t compress_table_lineorder_offset_start_page_ids[SSB::common::kLineOrderFieldCount];
    uint64_t compress_table_lineorder_compressed_page_size_start_page_ids[SSB::common::kLineOrderFieldCount];
};

namespace SSB {
    void metadata_print(const SSBTableMetadata &metadata)
    {
        std::cout << "=== SSB Metadata ===" << std::endl;
        std::cout << "page_size: " << metadata.page_size << std::endl;
        std::cout << "compressed: " << metadata.compressed << std::endl;
        std::cout << "column: " << metadata.column << std::endl;
        std::cout << "free_page_id: " << metadata.free_page_id << std::endl;
        std::cout << "npage_used: " << metadata.npage_used << std::endl;

        auto print_table = [&](const char *name, int nfields_total,
                               const uint64_t *start_page_ids, const uint64_t *npages,
                               const uint16_t *compression_method, uint64_t nrows) {
            std::cout << "\n--- " << name << " (nrows=" << nrows << ") ---" << std::endl;
            for (int i = 0; i < nfields_total; i++) {
                if (npages[i] > 0) {
                    std::cout << "  field[" << i << "]: start_page=" << start_page_ids[i]
                              << " npages=" << npages[i]
                              << " compression=" << compression_method_name(static_cast<CompressionMethod>(compression_method[i]))
                              << std::endl;
                }
            }
        };

        print_table("DATE", SSB::common::kDateFieldCount,
                     metadata.table_date_start_page_ids, metadata.table_date_npages,
                     metadata.table_date_compression_method, metadata.table_date_nrows);
        print_table("CUSTOMER", SSB::common::kCustomerFieldCount,
                     metadata.table_customer_start_page_ids, metadata.table_customer_npages,
                     metadata.table_customer_compression_method, metadata.table_customer_nrows);
        print_table("SUPPLIER", SSB::common::kSupplierFieldCount,
                     metadata.table_supplier_start_page_ids, metadata.table_supplier_npages,
                     metadata.table_supplier_compression_method, metadata.table_supplier_nrows);
        print_table("PART", SSB::common::kPartFieldCount,
                     metadata.table_part_start_page_ids, metadata.table_part_npages,
                     metadata.table_part_compression_method, metadata.table_part_nrows);
        print_table("LINEORDER", SSB::common::kLineOrderFieldCount,
                     metadata.table_lineorder_start_page_ids, metadata.table_lineorder_npages,
                     metadata.table_lineorder_compression_method, metadata.table_lineorder_nrows);

    }
}


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

const uint64_t LO_ORDERKEY_BLOCK_START = 0;
