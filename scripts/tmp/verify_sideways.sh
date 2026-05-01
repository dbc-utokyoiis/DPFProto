#!/usr/bin/env bash
set -euo pipefail

SF="$1"
TMP_DIR="/export/data1/tpch/tmp"
ZST_FILE="/export/data1/ssb/sideways_zst/sf${SF}.tar.zst"
RAW_DIR="/export/data1/ssb/input${SF}"
EXT_DIR="${TMP_DIR}/sf${SF}"

echo "========================================"
echo "  Verifying SF=${SF}"
echo "========================================"

# Step 1: Extract
echo "[1] Extracting ${ZST_FILE} ..."
START=$(date +%s)
tar -I 'pzstd -p 112' -xf "${ZST_FILE}" -C "${TMP_DIR}/"
END=$(date +%s)
echo "    Done in $(( END - START ))s"

# Step 2: Check symlinks
echo "[2] Checking symlinks ..."
SYMLINKS=$(find "${EXT_DIR}" -type l | wc -l)
if [ "${SYMLINKS}" -gt 0 ]; then
    echo "    WARNING: ${SYMLINKS} symlinks found!"
    find "${EXT_DIR}" -type l -ls | head -10
else
    echo "    OK: No symlinks"
fi

# Step 3: DuckDB verification
echo "[3] Running DuckDB verification ..."
duckdb :memory: <<EOSQL
SET threads = $(nproc);

-- Load sideways lineorder (27 cols)
CREATE TABLE lo_sw AS
SELECT * FROM read_csv('${EXT_DIR}/lineorder/lineorder.tbl*', delim='|', header=False,
  ignore_errors=true, columns={
    'lo_orderkey': 'BIGINT', 'lo_linenumber': 'INTEGER', 'lo_custkey': 'INTEGER',
    'lo_partkey': 'INTEGER', 'lo_suppkey': 'INTEGER', 'lo_orderdate': 'INTEGER',
    'lo_orderpriority': 'VARCHAR', 'lo_shippriority': 'VARCHAR',
    'lo_quantity': 'INTEGER', 'lo_extendedprice': 'INTEGER',
    'lo_ordtotalprice': 'INTEGER', 'lo_discount': 'INTEGER',
    'lo_revenue': 'INTEGER', 'lo_supplycost': 'INTEGER', 'lo_tax': 'INTEGER',
    'lo_commitdate': 'INTEGER', 'lo_shipmode': 'VARCHAR',
    'lss_s_region': 'VARCHAR', 'lss_c_region': 'VARCHAR', 'lss_p_mfgr': 'VARCHAR',
    'lss_s_nation': 'VARCHAR', 'lss_c_nation': 'VARCHAR', 'lss_p_category': 'VARCHAR',
    'lss_s_city': 'VARCHAR', 'lss_c_city': 'VARCHAR', 'lss_p_brand1': 'VARCHAR'
});

-- Load raw lineorder (17 cols)
CREATE TABLE lo_raw AS
SELECT * FROM read_csv('${RAW_DIR}/lineorder/lineorder.tbl*', delim='|', header=False,
  ignore_errors=true, columns={
    'lo_orderkey': 'BIGINT', 'lo_linenumber': 'INTEGER', 'lo_custkey': 'INTEGER',
    'lo_partkey': 'INTEGER', 'lo_suppkey': 'INTEGER', 'lo_orderdate': 'INTEGER',
    'lo_orderpriority': 'VARCHAR', 'lo_shippriority': 'VARCHAR',
    'lo_quantity': 'INTEGER', 'lo_extendedprice': 'INTEGER',
    'lo_ordtotalprice': 'INTEGER', 'lo_discount': 'INTEGER',
    'lo_revenue': 'INTEGER', 'lo_supplycost': 'INTEGER', 'lo_tax': 'INTEGER',
    'lo_commitdate': 'INTEGER', 'lo_shipmode': 'VARCHAR'
});

-- Row counts
SELECT 'ROW_COUNT' as check_type,
       (SELECT count(*) FROM lo_sw) as sideways,
       (SELECT count(*) FROM lo_raw) as raw,
       CASE WHEN (SELECT count(*) FROM lo_sw) = (SELECT count(*) FROM lo_raw)
            THEN 'OK' ELSE 'MISMATCH' END as result;

-- Checksum on original 17 columns
SELECT 'CHECKSUM_17COL' as check_type,
       (SELECT md5(string_agg(h, '' ORDER BY lo_orderkey, lo_linenumber)) FROM
        (SELECT lo_orderkey, lo_linenumber, md5(concat_ws('|', lo_orderkey, lo_linenumber, lo_custkey, lo_partkey, lo_suppkey, lo_orderdate, lo_orderpriority, lo_shippriority, lo_quantity, lo_extendedprice, lo_ordtotalprice, lo_discount, lo_revenue, lo_supplycost, lo_tax, lo_commitdate, lo_shipmode)) as h FROM lo_sw)) as sideways,
       (SELECT md5(string_agg(h, '' ORDER BY lo_orderkey, lo_linenumber)) FROM
        (SELECT lo_orderkey, lo_linenumber, md5(concat_ws('|', lo_orderkey, lo_linenumber, lo_custkey, lo_partkey, lo_suppkey, lo_orderdate, lo_orderpriority, lo_shippriority, lo_quantity, lo_extendedprice, lo_ordtotalprice, lo_discount, lo_revenue, lo_supplycost, lo_tax, lo_commitdate, lo_shipmode)) as h FROM lo_raw)) as raw,
       'compare' as result;

-- Check for NULL/empty sideways columns
SELECT 'SIDEWAYS_NULLS' as check_type,
       count(*) as total_rows,
       count(*) FILTER (WHERE lss_s_region IS NULL OR lss_s_region = '') as null_s_region,
       count(*) FILTER (WHERE lss_c_region IS NULL OR lss_c_region = '') as null_c_region,
       count(*) FILTER (WHERE lss_p_mfgr IS NULL OR lss_p_mfgr = '') as null_p_mfgr,
       count(*) FILTER (WHERE lss_s_nation IS NULL OR lss_s_nation = '') as null_s_nation,
       count(*) FILTER (WHERE lss_c_nation IS NULL OR lss_c_nation = '') as null_c_nation,
       count(*) FILTER (WHERE lss_p_category IS NULL OR lss_p_category = '') as null_p_category,
       count(*) FILTER (WHERE lss_s_city IS NULL OR lss_s_city = '') as null_s_city,
       count(*) FILTER (WHERE lss_c_city IS NULL OR lss_c_city = '') as null_c_city,
       count(*) FILTER (WHERE lss_p_brand1 IS NULL OR lss_p_brand1 = '') as null_p_brand1
FROM lo_sw;

-- Distinct value counts per sideways column
SELECT 'SIDEWAYS_DISTINCT' as check_type,
       count(DISTINCT lss_s_region) as s_region,
       count(DISTINCT lss_c_region) as c_region,
       count(DISTINCT lss_p_mfgr) as p_mfgr,
       count(DISTINCT lss_s_nation) as s_nation,
       count(DISTINCT lss_c_nation) as c_nation,
       count(DISTINCT lss_p_category) as p_category,
       count(DISTINCT lss_s_city) as s_city,
       count(DISTINCT lss_c_city) as c_city,
       count(DISTINCT lss_p_brand1) as p_brand1
FROM lo_sw;

-- Dimension table row counts
SELECT 'DIM_ROWS' as check_type, tbl, sw_cnt, raw_cnt,
       CASE WHEN sw_cnt = raw_cnt THEN 'OK' ELSE 'MISMATCH' END as result
FROM (
  VALUES
    ('customer',
     (SELECT count(*) FROM read_csv('${EXT_DIR}/customer/customer.tbl*', delim='|', header=False)),
     (SELECT count(*) FROM read_csv('${RAW_DIR}/customer/customer.tbl*', delim='|', header=False))),
    ('supplier',
     (SELECT count(*) FROM read_csv('${EXT_DIR}/supplier/supplier.tbl*', delim='|', header=False)),
     (SELECT count(*) FROM read_csv('${RAW_DIR}/supplier/supplier.tbl*', delim='|', header=False))),
    ('part',
     (SELECT count(*) FROM read_csv('${EXT_DIR}/part/part.tbl*', delim='|', header=False)),
     (SELECT count(*) FROM read_csv('${RAW_DIR}/part/part.tbl*', delim='|', header=False))),
    ('date',
     (SELECT count(*) FROM read_csv('${EXT_DIR}/date/date.tbl*', delim='|', header=False)),
     (SELECT count(*) FROM read_csv('${RAW_DIR}/date/date.tbl*', delim='|', header=False)))
) t(tbl, sw_cnt, raw_cnt);
EOSQL

# Step 4: Cleanup
echo "[4] Cleaning up extracted data ..."
rm -rf "${EXT_DIR}"
echo "    Done."
echo ""
