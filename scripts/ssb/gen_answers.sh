#!/usr/bin/env bash
# Generate SSB query reference answers using DuckDB.
# Converts CSV to Parquet first (faster scans, tolerates corrupt lines),
# then runs all 13 SSB queries against the Parquet tables.
#
# Usage: bash scripts/ssb/gen_answers.sh [-s SF]
set -euo pipefail

SF=1
NO_LOAD=0
FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s) SF="$2"; shift 2 ;;
        --no-load) NO_LOAD=1; shift ;;
        --overwrite|-f) FORCE=1; shift ;;
        *)  echo "Usage: $0 [-s SF] [--no-load] [--overwrite|-f]" >&2; exit 1 ;;
    esac
done

DATA_BASE="/export/data1/ssb"
INPUT_DIR="${DATA_BASE}/input${SF}"
PARQUET_DIR="${DATA_BASE}/parquet/sf${SF}"
OUT_DIR="answers/ssb/sf${SF}"
mkdir -p "${OUT_DIR}" "${PARQUET_DIR}"

echo "Generating SSB answers for SF=${SF} ..."
echo "  Input:   ${INPUT_DIR}"
echo "  Parquet: ${PARQUET_DIR}"
echo "  Output:  ${OUT_DIR}"

# ── Pre-check: warn if Parquet directory already has data ──
if [[ ${NO_LOAD} -eq 0 ]] && [ -d "${PARQUET_DIR}" ] && [ "$(ls -A "${PARQUET_DIR}" 2>/dev/null)" ]; then
    echo ""
    echo "WARNING: ${PARQUET_DIR} already contains data."
    if [[ ${FORCE} -eq 0 ]]; then
        read -p "Overwrite existing Parquet files? [y/N] " answer
        if [[ "${answer}" != "y" && "${answer}" != "Y" ]]; then
            echo "Aborted. Use --no-load to reuse existing files, or --overwrite/-f to skip this prompt."
            exit 0
        fi
    else
        echo "  Overwriting (--overwrite/-f)."
    fi
fi

# ── Step 1: CSV → Parquet (skip corrupt lines with ignore_errors) ──
if [[ ${NO_LOAD} -eq 0 ]]; then
echo ""
echo "[Step 1] Converting CSV to Parquet ..."
STEP1_START=$(date +%s)

duckdb :memory: <<EOSQL
SET threads = $(nproc);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/lineorder/lineorder.tbl*', delim='|', header=False,
    ignore_errors=true, columns={
      'lo_orderkey': 'BIGINT', 'lo_linenumber': 'INTEGER', 'lo_custkey': 'INTEGER',
      'lo_partkey': 'INTEGER', 'lo_suppkey': 'INTEGER', 'lo_orderdate': 'INTEGER',
      'lo_orderpriority': 'VARCHAR', 'lo_shippriority': 'VARCHAR',
      'lo_quantity': 'INTEGER', 'lo_extendedprice': 'BIGINT',
      'lo_ordtotalprice': 'BIGINT', 'lo_discount': 'INTEGER',
      'lo_revenue': 'BIGINT', 'lo_supplycost': 'INTEGER', 'lo_tax': 'INTEGER',
      'lo_commitdate': 'INTEGER', 'lo_shipmode': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/lineorder.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/customer/customer.tbl*', delim='|', header=False,
    columns={
      'c_custkey': 'INTEGER', 'c_name': 'VARCHAR', 'c_address': 'VARCHAR',
      'c_city': 'VARCHAR', 'c_nation': 'VARCHAR', 'c_region': 'VARCHAR',
      'c_phone': 'VARCHAR', 'c_mktsegment': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/customer.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/supplier/supplier.tbl*', delim='|', header=False,
    columns={
      's_suppkey': 'INTEGER', 's_name': 'VARCHAR', 's_address': 'VARCHAR',
      's_city': 'VARCHAR', 's_nation': 'VARCHAR', 's_region': 'VARCHAR',
      's_phone': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/supplier.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/part/part.tbl*', delim='|', header=False,
    columns={
      'p_partkey': 'INTEGER', 'p_name': 'VARCHAR', 'p_mfgr': 'VARCHAR',
      'p_category': 'VARCHAR', 'p_brand1': 'VARCHAR', 'p_color': 'VARCHAR',
      'p_type': 'VARCHAR', 'p_size': 'INTEGER', 'p_container': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/part.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/date/date.tbl*', delim='|', header=False,
    columns={
      'd_datekey': 'INTEGER', 'd_date': 'VARCHAR', 'd_dayofweek': 'VARCHAR',
      'd_month': 'VARCHAR', 'd_year': 'INTEGER', 'd_yearmonthnum': 'INTEGER',
      'd_yearmonth': 'VARCHAR', 'd_daynuminweek': 'INTEGER',
      'd_daynuminmonth': 'INTEGER', 'd_daynuminyear': 'INTEGER',
      'd_monthnuminyear': 'INTEGER', 'd_weeknuminyear': 'INTEGER',
      'd_sellingseason': 'VARCHAR', 'd_lastdayinweekfl': 'INTEGER',
      'd_lastdayinmonthfl': 'INTEGER', 'd_holidayfl': 'INTEGER',
      'd_weekdayfl': 'INTEGER'
  })
) TO '${PARQUET_DIR}/date.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);
EOSQL

STEP1_END=$(date +%s)
echo "  Done in $(( STEP1_END - STEP1_START ))s"
else
echo ""
echo "[Step 1] Skipped (--no-load)"
fi

# ── Step 2: Run queries on Parquet ──
echo ""
echo "[Step 2] Running SSB queries ..."
STEP2_START=$(date +%s)

duckdb :memory: <<EOSQL
SET threads = $(nproc);

CREATE VIEW lineorder AS SELECT * FROM read_parquet('${PARQUET_DIR}/lineorder.parquet/*');
CREATE VIEW customer  AS SELECT * FROM read_parquet('${PARQUET_DIR}/customer.parquet/*');
CREATE VIEW supplier  AS SELECT * FROM read_parquet('${PARQUET_DIR}/supplier.parquet/*');
CREATE VIEW part      AS SELECT * FROM read_parquet('${PARQUET_DIR}/part.parquet/*');
CREATE VIEW ddate     AS SELECT * FROM read_parquet('${PARQUET_DIR}/date.parquet/*');

-- Q1.1
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_year = 1993
      and lo_discount between 1 and 3
      and lo_quantity < 25
) TO '${OUT_DIR}/q11.csv' (HEADER, DELIMITER '|');

-- Q1.2
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_yearmonthnum = 199401
      and lo_discount between 4 and 6
      and lo_quantity between 26 and 35
) TO '${OUT_DIR}/q12.csv' (HEADER, DELIMITER '|');

-- Q1.3
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_weeknuminyear = 6
      and d_year = 1994
      and lo_discount between 5 and 7
      and lo_quantity between 26 and 35
) TO '${OUT_DIR}/q13.csv' (HEADER, DELIMITER '|');

-- Q2.1
COPY (
    select sum(lo_revenue) as lo_revenue, d_year, p_brand1
    from lineorder, ddate, part, supplier
    where lo_orderdate = d_datekey
      and lo_partkey = p_partkey
      and lo_suppkey = s_suppkey
      and p_category = 'MFGR#12'
      and s_region = 'AMERICA'
    group by d_year, p_brand1
    order by d_year, p_brand1
) TO '${OUT_DIR}/q21.csv' (HEADER, DELIMITER '|');

-- Q2.2
COPY (
    select sum(lo_revenue) as lo_revenue, d_year, p_brand1
    from lineorder, ddate, part, supplier
    where lo_orderdate = d_datekey
      and lo_partkey = p_partkey
      and lo_suppkey = s_suppkey
      and p_brand1 between 'MFGR#2221' and 'MFGR#2228'
      and s_region = 'ASIA'
    group by d_year, p_brand1
    order by d_year, p_brand1
) TO '${OUT_DIR}/q22.csv' (HEADER, DELIMITER '|');

-- Q2.3
COPY (
    select sum(lo_revenue) as lo_revenue, d_year, p_brand1
    from lineorder, ddate, part, supplier
    where lo_orderdate = d_datekey
      and lo_partkey = p_partkey
      and lo_suppkey = s_suppkey
      and p_brand1 = 'MFGR#2221'
      and s_region = 'EUROPE'
    group by d_year, p_brand1
    order by d_year, p_brand1
) TO '${OUT_DIR}/q23.csv' (HEADER, DELIMITER '|');

-- Q3.1
COPY (
    select c_nation, s_nation, d_year, sum(lo_revenue) as revenue
    from customer, lineorder, supplier, ddate
    where lo_custkey = c_custkey
      and lo_suppkey = s_suppkey
      and lo_orderdate = d_datekey
      and c_region = 'ASIA' and s_region = 'ASIA'
      and d_year >= 1992 and d_year <= 1997
    group by c_nation, s_nation, d_year
    order by d_year asc, revenue desc
) TO '${OUT_DIR}/q31.csv' (HEADER, DELIMITER '|');

-- Q3.2
COPY (
    select c_city, s_city, d_year, sum(lo_revenue) as revenue
    from customer, lineorder, supplier, ddate
    where lo_custkey = c_custkey
      and lo_suppkey = s_suppkey
      and lo_orderdate = d_datekey
      and c_nation = 'UNITED STATES'
      and s_nation = 'UNITED STATES'
      and d_year >= 1992 and d_year <= 1997
    group by c_city, s_city, d_year
    order by d_year asc, revenue desc
) TO '${OUT_DIR}/q32.csv' (HEADER, DELIMITER '|');

-- Q3.3
COPY (
    select c_city, s_city, d_year, sum(lo_revenue) as revenue
    from customer, lineorder, supplier, ddate
    where lo_custkey = c_custkey
      and lo_suppkey = s_suppkey
      and lo_orderdate = d_datekey
      and (c_city='UNITED KI1' or c_city='UNITED KI5')
      and (s_city='UNITED KI1' or s_city='UNITED KI5')
      and d_year >= 1992 and d_year <= 1997
    group by c_city, s_city, d_year
    order by d_year asc, revenue desc
) TO '${OUT_DIR}/q33.csv' (HEADER, DELIMITER '|');

-- Q3.4
COPY (
    select c_city, s_city, d_year, sum(lo_revenue) as revenue
    from customer, lineorder, supplier, ddate
    where lo_custkey = c_custkey
      and lo_suppkey = s_suppkey
      and lo_orderdate = d_datekey
      and (c_city='UNITED KI1' or c_city='UNITED KI5')
      and (s_city='UNITED KI1' or s_city='UNITED KI5')
      and d_yearmonth = 'Dec1997'
    group by c_city, s_city, d_year
    order by d_year asc, revenue desc
) TO '${OUT_DIR}/q34.csv' (HEADER, DELIMITER '|');

-- Q4.1
COPY (
    select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit
    from ddate, customer, supplier, part, lineorder
    where lo_custkey = c_custkey
      and lo_suppkey = s_suppkey
      and lo_partkey = p_partkey
      and lo_orderdate = d_datekey
      and c_region = 'AMERICA'
      and s_region = 'AMERICA'
      and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
    group by d_year, c_nation
    order by d_year, c_nation
) TO '${OUT_DIR}/q41.csv' (HEADER, DELIMITER '|');

-- Q4.2
COPY (
    select d_year, s_nation, p_category,
           sum(lo_revenue - lo_supplycost) as profit
    from ddate, customer, supplier, part, lineorder
    where lo_custkey = c_custkey
      and lo_suppkey = s_suppkey
      and lo_partkey = p_partkey
      and lo_orderdate = d_datekey
      and c_region = 'AMERICA'
      and s_region = 'AMERICA'
      and (d_year = 1997 or d_year = 1998)
      and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
    group by d_year, s_nation, p_category
    order by d_year, s_nation, p_category
) TO '${OUT_DIR}/q42.csv' (HEADER, DELIMITER '|');

-- Q4.3
COPY (
    select d_year, s_city, p_brand1,
           sum(lo_revenue - lo_supplycost) as profit
    from ddate, customer, supplier, part, lineorder
    where lo_custkey = c_custkey
      and lo_suppkey = s_suppkey
      and lo_partkey = p_partkey
      and lo_orderdate = d_datekey
      and c_region = 'AMERICA'
      and s_nation = 'UNITED STATES'
      and (d_year = 1997 or d_year = 1998)
      and p_category = 'MFGR#14'
    group by d_year, s_city, p_brand1
    order by d_year, s_city, p_brand1
) TO '${OUT_DIR}/q43.csv' (HEADER, DELIMITER '|');

EOSQL

STEP2_END=$(date +%s)
echo "  Done in $(( STEP2_END - STEP2_START ))s"

echo ""
echo "=== Answers generated ==="
for f in ${OUT_DIR}/q*.csv; do
    echo "  $(basename $f): $(wc -l < "$f") lines"
done
