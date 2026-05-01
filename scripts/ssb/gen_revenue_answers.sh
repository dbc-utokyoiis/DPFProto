#!/usr/bin/env bash
# Generate SSB revenue (selectivity-varying Q1.1) reference answers using DuckDB.
# Reuses Parquet files created by gen_answers.sh.
#
# Usage: bash scripts/ssb/gen_revenue_answers.sh [-s SF]
set -euo pipefail

SF=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s) SF="$2"; shift 2 ;;
        *)  echo "Usage: $0 [-s SF]" >&2; exit 1 ;;
    esac
done

DATA_BASE="/export/data1/ssb"
PARQUET_DIR="${DATA_BASE}/parquet/sf${SF}"
OUT_DIR="answers/ssb_revenue/sf${SF}"
mkdir -p "${OUT_DIR}"

echo "Generating SSB revenue answers for SF=${SF} ..."
echo "  Parquet: ${PARQUET_DIR}"
echo "  Output:  ${OUT_DIR}"

# Check that Parquet files exist
if [ ! -d "${PARQUET_DIR}/lineorder.parquet" ]; then
    echo "Error: Parquet not found at ${PARQUET_DIR}. Run gen_answers.sh -s ${SF} first." >&2
    exit 1
fi

duckdb :memory: <<EOSQL
SET threads = $(nproc);

CREATE VIEW lineorder AS SELECT * FROM read_parquet('${PARQUET_DIR}/lineorder.parquet/*');
CREATE VIEW ddate     AS SELECT * FROM read_parquet('${PARQUET_DIR}/date.parquet/*');

-- sel7: ~7.57% selectivity (182 days from 1992-01-01)
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_datekey between 19920101 and 19920701
      and lo_quantity < 51
) TO '${OUT_DIR}/sel7.csv' (HEADER, DELIMITER '|');

-- sel15: ~15% selectivity (1 year)
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_year between 1992 and 1992
      and lo_quantity < 51
) TO '${OUT_DIR}/sel15.csv' (HEADER, DELIMITER '|');

-- sel30: ~30% selectivity (2 years)
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_year between 1992 and 1993
      and lo_quantity < 51
) TO '${OUT_DIR}/sel30.csv' (HEADER, DELIMITER '|');

-- sel45: ~45% selectivity (3 years)
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_year between 1992 and 1994
      and lo_quantity < 51
) TO '${OUT_DIR}/sel45.csv' (HEADER, DELIMITER '|');

-- sel60: ~60% selectivity (4 years)
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_year between 1992 and 1995
      and lo_quantity < 51
) TO '${OUT_DIR}/sel60.csv' (HEADER, DELIMITER '|');

-- sel75: ~75% selectivity (5 years)
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_year between 1992 and 1996
      and lo_quantity < 51
) TO '${OUT_DIR}/sel75.csv' (HEADER, DELIMITER '|');

-- sel90: ~90% selectivity (6 years)
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_year between 1992 and 1997
      and lo_quantity < 51
) TO '${OUT_DIR}/sel90.csv' (HEADER, DELIMITER '|');

-- sel100: 100% selectivity (all 7 years)
COPY (
    select sum(lo_extendedprice * lo_discount) as revenue
    from lineorder, ddate
    where lo_orderdate = d_datekey
      and d_year between 1992 and 1998
      and lo_quantity < 51
) TO '${OUT_DIR}/sel100.csv' (HEADER, DELIMITER '|');

EOSQL

echo ""
echo "=== Answers generated ==="
for f in ${OUT_DIR}/sel*.csv; do
    echo "  $(basename $f): $(cat "$f")"
done
