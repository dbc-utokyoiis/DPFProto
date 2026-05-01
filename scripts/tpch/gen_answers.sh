#!/usr/bin/env bash
# Generate TPC-H query reference answers using DuckDB.
# Converts CSV to Parquet first (faster scans, tolerates corrupt lines),
# then runs queries against the Parquet tables.
#
# Usage: bash scripts/tpch/gen_answers.sh [-s SF]
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

DATA_BASE="/export/data1/tpch"
INPUT_DIR="${DATA_BASE}/input${SF}"
PARQUET_DIR="${DATA_BASE}/parquet/sf${SF}"
OUT_DIR="answers/tpch/sf${SF}"
mkdir -p "${OUT_DIR}" "${PARQUET_DIR}"

echo "Generating TPC-H answers for SF=${SF} ..."
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
  SELECT * FROM read_csv('${INPUT_DIR}/lineitem/lineitem.tbl*', delim='|', header=False,
    ignore_errors=true, columns={
      'l_orderkey': 'BIGINT', 'l_partkey': 'INTEGER', 'l_suppkey': 'INTEGER',
      'l_linenumber': 'BIGINT', 'l_quantity': 'DECIMAL(15,2)',
      'l_extendedprice': 'DECIMAL(15,2)', 'l_discount': 'DECIMAL(15,2)',
      'l_tax': 'DECIMAL(15,2)', 'l_returnflag': 'VARCHAR', 'l_linestatus': 'VARCHAR',
      'l_shipdate': 'DATE', 'l_commitdate': 'DATE', 'l_receiptdate': 'DATE',
      'l_shipinstruct': 'VARCHAR', 'l_shipmode': 'VARCHAR', 'l_comment': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/lineitem.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/orders/orders.tbl*', delim='|', header=False,
    ignore_errors=true, columns={
      'o_orderkey': 'BIGINT', 'o_custkey': 'BIGINT', 'o_orderstatus': 'VARCHAR',
      'o_totalprice': 'DECIMAL(15,2)', 'o_orderdate': 'DATE',
      'o_orderpriority': 'VARCHAR', 'o_clerk': 'VARCHAR',
      'o_shippriority': 'INTEGER', 'o_comment': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/orders.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/customer/customer.tbl*', delim='|', header=False,
    ignore_errors=true, columns={
      'c_custkey': 'BIGINT', 'c_name': 'VARCHAR', 'c_address': 'VARCHAR',
      'c_nationkey': 'INTEGER', 'c_phone': 'VARCHAR',
      'c_acctbal': 'DECIMAL(15,2)', 'c_mktsegment': 'VARCHAR', 'c_comment': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/customer.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/part/part.tbl*', delim='|', header=False,
    ignore_errors=true, columns={
      'p_partkey': 'INTEGER', 'p_name': 'VARCHAR', 'p_mfgr': 'VARCHAR',
      'p_brand': 'VARCHAR', 'p_type': 'VARCHAR', 'p_size': 'INTEGER',
      'p_container': 'VARCHAR', 'p_retailprice': 'DECIMAL(15,2)', 'p_comment': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/part.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/supplier/supplier.tbl*', delim='|', header=False,
    ignore_errors=true, columns={
      's_suppkey': 'INTEGER', 's_name': 'VARCHAR', 's_address': 'VARCHAR',
      's_nationkey': 'INTEGER', 's_phone': 'VARCHAR',
      's_acctbal': 'DECIMAL(15,2)', 's_comment': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/supplier.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/partsupp/partsupp.tbl*', delim='|', header=False,
    ignore_errors=true, columns={
      'ps_partkey': 'INTEGER', 'ps_suppkey': 'INTEGER', 'ps_availqty': 'INTEGER',
      'ps_supplycost': 'DECIMAL(15,2)', 'ps_comment': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/partsupp.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/nation/nation.tbl*', delim='|', header=False,
    columns={
      'n_nationkey': 'INTEGER', 'n_name': 'VARCHAR',
      'n_regionkey': 'INTEGER', 'n_comment': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/nation.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);

COPY (
  SELECT * FROM read_csv('${INPUT_DIR}/region/region.tbl*', delim='|', header=False,
    columns={
      'r_regionkey': 'INTEGER', 'r_name': 'VARCHAR', 'r_comment': 'VARCHAR'
  })
) TO '${PARQUET_DIR}/region.parquet' (FORMAT PARQUET, PER_THREAD_OUTPUT TRUE, OVERWRITE);
EOSQL

STEP1_END=$(date +%s)
echo "  Done in $(( STEP1_END - STEP1_START ))s"
else
echo ""
echo "[Step 1] Skipped (--no-load)"
fi

# ── Step 2: Run queries on Parquet ──
echo ""
echo "[Step 2] Running TPC-H queries ..."
STEP2_START=$(date +%s)

duckdb :memory: <<EOSQL
SET threads = $(nproc);

CREATE VIEW lineitem AS SELECT * FROM read_parquet('${PARQUET_DIR}/lineitem.parquet/*');
CREATE VIEW orders   AS SELECT * FROM read_parquet('${PARQUET_DIR}/orders.parquet/*');
CREATE VIEW customer AS SELECT * FROM read_parquet('${PARQUET_DIR}/customer.parquet/*');
CREATE VIEW part     AS SELECT * FROM read_parquet('${PARQUET_DIR}/part.parquet/*');
CREATE VIEW supplier AS SELECT * FROM read_parquet('${PARQUET_DIR}/supplier.parquet/*');
CREATE VIEW partsupp AS SELECT * FROM read_parquet('${PARQUET_DIR}/partsupp.parquet/*');
CREATE VIEW nation   AS SELECT * FROM read_parquet('${PARQUET_DIR}/nation.parquet/*');
CREATE VIEW region   AS SELECT * FROM read_parquet('${PARQUET_DIR}/region.parquet/*');

-- Q1
COPY (
    select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice*(1-l_discount)) as sum_disc_price,
        sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
    from lineitem
    where l_shipdate <= date '1998-12-01' - interval '90' day
    group by l_returnflag, l_linestatus
    order by l_returnflag, l_linestatus
) TO '${OUT_DIR}/q1.csv' (HEADER, DELIMITER '|');

-- Q3
COPY (
    select
        l_orderkey,
        sum(l_extendedprice*(1-l_discount)) as revenue,
        o_orderdate,
        o_shippriority
    from customer, orders, lineitem
    where c_mktsegment = 'BUILDING'
      and c_custkey = o_custkey
      and l_orderkey = o_orderkey
      and o_orderdate < date '1995-03-15'
      and l_shipdate > date '1995-03-15'
    group by l_orderkey, o_orderdate, o_shippriority
    order by revenue desc, o_orderdate
) TO '${OUT_DIR}/q3.csv' (HEADER, DELIMITER '|');

-- Q5
COPY (
    select
        n_name,
        sum(l_extendedprice * (1 - l_discount)) as revenue
    from customer, orders, lineitem, supplier, nation, region
    where c_custkey = o_custkey
      and l_orderkey = o_orderkey
      and l_suppkey = s_suppkey
      and c_nationkey = s_nationkey
      and s_nationkey = n_nationkey
      and n_regionkey = r_regionkey
      and r_name = 'ASIA'
      and o_orderdate >= date '1994-01-01'
      and o_orderdate < date '1994-01-01' + interval '1' year
    group by n_name
    order by revenue desc
) TO '${OUT_DIR}/q5.csv' (HEADER, DELIMITER '|');

-- Q6
COPY (
    select
        sum(l_extendedprice*l_discount) as revenue
    from lineitem
    where l_shipdate >= date '1994-01-01'
      and l_shipdate < date '1994-01-01' + interval '1' year
      and l_discount between 0.06 - 0.01 and 0.06 + 0.01
      and l_quantity < 24
) TO '${OUT_DIR}/q6.csv' (HEADER, DELIMITER '|');

-- Q13
COPY (
    select
        c_count, count(*) as custdist
    from (
        select
            c_custkey,
            count(o_orderkey) as c_count
        from customer left outer join orders on
            c_custkey = o_custkey
            and o_comment not like '%special%requests%'
        group by c_custkey
    ) as c_orders
    group by c_count
    order by custdist desc, c_count desc
) TO '${OUT_DIR}/q13.csv' (HEADER, DELIMITER '|');

-- Q16
COPY (
    select
        p_brand,
        p_type,
        p_size,
        count(distinct ps_suppkey) as supplier_cnt
    from partsupp, part
    where p_partkey = ps_partkey
      and p_brand <> 'Brand#45'
      and p_type not like 'MEDIUM POLISHED%'
      and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
      and ps_suppkey not in (
          select s_suppkey
          from supplier
          where s_comment like '%Customer%Complaints%'
      )
    group by p_brand, p_type, p_size
    order by supplier_cnt desc, p_brand, p_type, p_size
) TO '${OUT_DIR}/q16.csv' (HEADER, DELIMITER '|');

EOSQL

STEP2_END=$(date +%s)
echo "  Done in $(( STEP2_END - STEP2_START ))s"

echo ""
echo "=== Answers generated ==="
for f in ${OUT_DIR}/q*.csv; do
    echo "  $(basename $f): $(wc -l < "$f") lines"
done
