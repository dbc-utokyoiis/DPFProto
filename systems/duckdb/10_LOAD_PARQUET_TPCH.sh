
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DUCKDB_VERSION=$(${DUCKDB} -csv -noheader -c "SELECT version()")
echo "=== TPC-H Parquet Load ==="
echo "DuckDB ${DUCKDB_VERSION}"
echo "SF=${SF}, Layout=${LAYOUT}, Threads=${NTHREADS}"
echo "Input:  ${INPUT_BASE}${SF}/"
echo "Output: ${PARQUET_OUTPUT_BASE}/sf${SF}/"
echo ""

for tbl in ${tbls[@]}
do
  mkdir -p ${PARQUET_OUTPUT_BASE}/sf${SF}/${tbl}.parquet
done

load_table() {
  local tbl=$1
  local sql=$2
  local start=$(date +%s.%N)
  ${DUCKDB} -c "
PRAGMA threads=${NTHREADS};
PRAGMA memory_limit='${MEMORY}';
${sql}
"
  local end=$(date +%s.%N)
  local elapsed=$(echo "$end - $start" | bc)
  printf "  %-12s %8.1fs\n" "${tbl}" "${elapsed}"
}

TOTAL_START=$(date +%s.%N)

load_table lineitem "
COPY (
  SELECT *
  FROM read_csv(
    '${INPUT_BASE}${SF}/lineitem/lineitem.tbl.*',
    delim='|',
    header=false,
    parallel=true,
    strict_mode = false,
    auto_detect = false,
    columns={
      'l_orderkey'      : 'BIGINT',
      'l_partkey'       : 'INTEGER',
      'l_suppkey'       : 'INTEGER',
      'l_linenumber'    : 'BIGINT',
      'l_quantity'      : 'DECIMAL(15,2)',
      'l_extendedprice' : 'DECIMAL(15,2)',
      'l_discount'      : 'DECIMAL(15,2)',
      'l_tax'           : 'DECIMAL(15,2)',
      'l_returnflag'    : 'VARCHAR',
      'l_linestatus'    : 'VARCHAR',
      'l_shipdate'      : 'DATE',
      'l_commitdate'    : 'DATE',
      'l_receiptdate'   : 'DATE',
      'l_shipinstruct'  : 'VARCHAR',
      'l_shipmode'      : 'VARCHAR',
      'l_comment'       : 'VARCHAR'
    }
  )
) TO '${PARQUET_OUTPUT_BASE}/sf${SF}/lineitem.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"

load_table orders "
COPY (
  SELECT *
  FROM read_csv(
    '${INPUT_BASE}${SF}/orders/orders.tbl.*',
    delim='|',
    header=false,
    parallel=true,
    strict_mode = false,
    auto_detect = false,
    columns={
      'o_orderkey'      : 'BIGINT',
      'o_custkey'       : 'BIGINT',
      'o_orderstatus'   : 'VARCHAR',
      'o_totalprice'    : 'DECIMAL(15,2)',
      'o_orderdate'     : 'DATE',
      'o_orderpriority' : 'VARCHAR',
      'o_clerk'         : 'VARCHAR',
      'o_shippriority'  : 'INTEGER',
      'o_comment'       : 'VARCHAR'
    }
  )
) TO '${PARQUET_OUTPUT_BASE}/sf${SF}/orders.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"

load_table customer "
COPY (
  SELECT *
  FROM read_csv(
    '${INPUT_BASE}${SF}/customer/customer.tbl.*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      'c_custkey'    : 'BIGINT',
      'c_name'       : 'VARCHAR',
      'c_address'    : 'VARCHAR',
      'c_nationkey'  : 'INTEGER',
      'c_phone'      : 'VARCHAR',
      'c_acctbal'    : 'DECIMAL(15,2)',
      'c_mktsegment' : 'VARCHAR',
      'c_comment'    : 'VARCHAR'
    }
  )
) TO '${PARQUET_OUTPUT_BASE}/sf${SF}/customer.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"

load_table part "
COPY (
  SELECT *
  FROM read_csv(
    '${INPUT_BASE}${SF}/part/part.tbl.*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      'p_partkey'     : 'INTEGER',
      'p_name'        : 'VARCHAR',
      'p_mfgr'        : 'VARCHAR',
      'p_brand'       : 'VARCHAR',
      'p_type'        : 'VARCHAR',
      'p_size'        : 'INTEGER',
      'p_container'   : 'VARCHAR',
      'p_retailprice' : 'DECIMAL(15,2)',
      'p_comment'     : 'VARCHAR'
    }
  )
) TO '${PARQUET_OUTPUT_BASE}/sf${SF}/part.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"

load_table partsupp "
COPY (
  SELECT *
  FROM read_csv(
    '${INPUT_BASE}${SF}/partsupp/partsupp.tbl.*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      'ps_partkey'    : 'INTEGER',
      'ps_suppkey'    : 'INTEGER',
      'ps_availqty'   : 'INTEGER',
      'ps_supplycost' : 'DECIMAL(15,2)',
      'ps_comment'    : 'VARCHAR'
    }
  )
) TO '${PARQUET_OUTPUT_BASE}/sf${SF}/partsupp.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"

load_table supplier "
COPY (
  SELECT *
  FROM read_csv(
    '${INPUT_BASE}${SF}/supplier/supplier.tbl.*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      's_suppkey'   : 'INTEGER',
      's_name'      : 'VARCHAR',
      's_address'   : 'VARCHAR',
      's_nationkey' : 'INTEGER',
      's_phone'     : 'VARCHAR',
      's_acctbal'   : 'DECIMAL(15,2)',
      's_comment'   : 'VARCHAR'
    }
  )
) TO '${PARQUET_OUTPUT_BASE}/sf${SF}/supplier.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"

load_table nation "
COPY (
  SELECT *
  FROM read_csv(
    '${INPUT_BASE}${SF}/nation/nation.tbl*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      'n_nationkey' : 'INTEGER',
      'n_name'      : 'VARCHAR',
      'n_regionkey' : 'INTEGER',
      'n_comment'   : 'VARCHAR'
    }
  )
) TO '${PARQUET_OUTPUT_BASE}/sf${SF}/nation.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"

load_table region "
COPY (
  SELECT *
  FROM read_csv(
    '${INPUT_BASE}${SF}/region/region.tbl*',
    delim='|',
    header=false,
    parallel=true,
    auto_detect = false,
    columns={
      'r_regionkey' : 'INTEGER',
      'r_name'      : 'VARCHAR',
      'r_comment'   : 'VARCHAR'
    }
  )
) TO '${PARQUET_OUTPUT_BASE}/sf${SF}/region.parquet'
  (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE);
"

TOTAL_END=$(date +%s.%N)
TOTAL_ELAPSED=$(echo "$TOTAL_END - $TOTAL_START" | bc)
echo ""
printf "  %-12s %8.1fs\n" "TOTAL" "${TOTAL_ELAPSED}"
