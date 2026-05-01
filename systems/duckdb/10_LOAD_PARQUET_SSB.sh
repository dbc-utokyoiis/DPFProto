
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for tbl in ${ssb_tbls[@]}
do
  mkdir -p ${SSB_PARQUET_OUTPUT_BASE}/${tbl}.parquet
done

if [ "${LAYOUT}" = "sideways" ]; then
LINEORDER_SQL="
-- ===== LINEORDER (sideways: read extra columns, select original only) =====
COPY (
  SELECT lo_orderkey, lo_linenumber, lo_custkey, lo_partkey, lo_suppkey,
         lo_orderdate, lo_orderpriority, lo_shippriority, lo_quantity,
         lo_extendedprice, lo_ordtotalprice, lo_discount, lo_revenue,
         lo_supplycost, lo_tax, lo_commitdate, lo_shipmode
  FROM read_csv(
    '${SSB_INPUT_BASE}/lineorder/lineorder.tbl.*',
    delim='|',
    header=false,
    parallel=true,
    strict_mode = false,
    auto_detect = false,
    columns={
      'lo_orderkey'      : 'BIGINT',
      'lo_linenumber'    : 'INTEGER',
      'lo_custkey'       : 'INTEGER',
      'lo_partkey'       : 'INTEGER',
      'lo_suppkey'       : 'INTEGER',
      'lo_orderdate'     : 'INTEGER',
      'lo_orderpriority' : 'VARCHAR',
      'lo_shippriority'  : 'INTEGER',
      'lo_quantity'      : 'INTEGER',
      'lo_extendedprice' : 'BIGINT',
      'lo_ordtotalprice' : 'BIGINT',
      'lo_discount'      : 'INTEGER',
      'lo_revenue'       : 'BIGINT',
      'lo_supplycost'    : 'BIGINT',
      'lo_tax'           : 'INTEGER',
      'lo_commitdate'    : 'INTEGER',
      'lo_shipmode'      : 'VARCHAR',
      's_region'         : 'VARCHAR',
      'c_region'         : 'VARCHAR',
      'p_mfgr'           : 'VARCHAR',
      's_nation'         : 'VARCHAR',
      'c_nation'         : 'VARCHAR',
      'p_category'       : 'VARCHAR',
      's_city'           : 'VARCHAR',
      'c_city'           : 'VARCHAR',
      'p_brand1'         : 'VARCHAR'
    }
  )
) TO '${SSB_PARQUET_OUTPUT_BASE}/lineorder.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"
else
LINEORDER_SQL="
-- ===== LINEORDER (normal) =====
COPY (
  SELECT *
  FROM read_csv(
    '${SSB_INPUT_BASE}/lineorder/lineorder.tbl.*',
    delim='|',
    header=false,
    parallel=true,
    strict_mode = false,
    auto_detect = false,
    columns={
      'lo_orderkey'      : 'BIGINT',
      'lo_linenumber'    : 'INTEGER',
      'lo_custkey'       : 'INTEGER',
      'lo_partkey'       : 'INTEGER',
      'lo_suppkey'       : 'INTEGER',
      'lo_orderdate'     : 'INTEGER',
      'lo_orderpriority' : 'VARCHAR',
      'lo_shippriority'  : 'INTEGER',
      'lo_quantity'      : 'INTEGER',
      'lo_extendedprice' : 'BIGINT',
      'lo_ordtotalprice' : 'BIGINT',
      'lo_discount'      : 'INTEGER',
      'lo_revenue'       : 'BIGINT',
      'lo_supplycost'    : 'BIGINT',
      'lo_tax'           : 'INTEGER',
      'lo_commitdate'    : 'INTEGER',
      'lo_shipmode'      : 'VARCHAR'
    }
  )
) TO '${SSB_PARQUET_OUTPUT_BASE}/lineorder.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"
fi

${DUCKDB} -c "
PRAGMA threads=${NTHREADS};
PRAGMA memory_limit='${MEMORY}';

${LINEORDER_SQL}

-- ===== CUSTOMER =====
COPY (
  SELECT *
  FROM read_csv(
    '${SSB_INPUT_BASE}/customer/customer.tbl*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      'c_custkey'    : 'INTEGER',
      'c_name'       : 'VARCHAR',
      'c_address'    : 'VARCHAR',
      'c_city'       : 'VARCHAR',
      'c_nation'     : 'VARCHAR',
      'c_region'     : 'VARCHAR',
      'c_phone'      : 'VARCHAR',
      'c_mktsegment' : 'VARCHAR'
    }
  )
) TO '${SSB_PARQUET_OUTPUT_BASE}/customer.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);

-- ===== DATE =====
COPY (
  SELECT *
  FROM read_csv(
    '${SSB_INPUT_BASE}/date/date.tbl*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      'd_datekey'          : 'INTEGER',
      'd_date'             : 'VARCHAR',
      'd_dayofweek'        : 'VARCHAR',
      'd_month'            : 'VARCHAR',
      'd_year'             : 'INTEGER',
      'd_yearmonthnum'     : 'INTEGER',
      'd_yearmonth'        : 'VARCHAR',
      'd_daynuminweek'     : 'INTEGER',
      'd_daynuminmonth'    : 'INTEGER',
      'd_daynuminyear'     : 'INTEGER',
      'd_monthnuminyear'   : 'INTEGER',
      'd_weeknuminyear'    : 'INTEGER',
      'd_sellingseason'    : 'VARCHAR',
      'd_lastdayinweekfl'  : 'INTEGER',
      'd_lastdayinmonthfl' : 'INTEGER',
      'd_holidayfl'        : 'INTEGER',
      'd_weekdayfl'        : 'INTEGER'
    }
  )
) TO '${SSB_PARQUET_OUTPUT_BASE}/date.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);

-- ===== PART =====
COPY (
  SELECT *
  FROM read_csv(
    '${SSB_INPUT_BASE}/part/part.tbl*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      'p_partkey'   : 'INTEGER',
      'p_name'      : 'VARCHAR',
      'p_mfgr'      : 'VARCHAR',
      'p_category'  : 'VARCHAR',
      'p_brand1'    : 'VARCHAR',
      'p_color'     : 'VARCHAR',
      'p_type'      : 'VARCHAR',
      'p_size'      : 'INTEGER',
      'p_container' : 'VARCHAR'
    }
  )
) TO '${SSB_PARQUET_OUTPUT_BASE}/part.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);

-- ===== SUPPLIER =====
COPY (
  SELECT *
  FROM read_csv(
    '${SSB_INPUT_BASE}/supplier/supplier.tbl*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      's_suppkey' : 'INTEGER',
      's_name'    : 'VARCHAR',
      's_address' : 'VARCHAR',
      's_city'    : 'VARCHAR',
      's_nation'  : 'VARCHAR',
      's_region'  : 'VARCHAR',
      's_phone'   : 'VARCHAR'
    }
  )
) TO '${SSB_PARQUET_OUTPUT_BASE}/supplier.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"
