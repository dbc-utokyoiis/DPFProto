#!/bin/bash
set -eu # エラー時に停止

SCRIPT_START=$(date +%s)
step_time() {
    local label="$1"
    local start="$2"
    local end=$(date +%s)
    local elapsed=$((end - start))
    echo "TIMING ${label}: ${elapsed}s ($(( elapsed / 60 ))m $(( elapsed % 60 ))s)"
}

# === CLI parsing ===
SF=100
NUM_SPLITS=$(nproc)
DELETE_INPUT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s) SF="$2"; shift 2 ;;
        -n) NUM_SPLITS="$2"; shift 2 ;;
        --delete-input)
            # Delete input/lineitem and input/orders after each denormalize
            # step to reduce peak storage. Required for SF>=500 on tight disks.
            DELETE_INPUT=1; shift ;;
        *)  echo "Usage: $0 [-s SF] [-n NUM_SPLITS] [--delete-input]" >&2; exit 1 ;;
    esac
done

# === Configuration ===
INPUT_BASE_DIR="/export/data1/tpch/input${SF}"
LINEITEM_INPUT_DIR="${INPUT_BASE_DIR}/lineitem"
ORDERS_INPUT_DIR="${INPUT_BASE_DIR}/orders"
CUSTOMER_INPUT_DIR="${INPUT_BASE_DIR}/customer"
NATION_INPUT_DIR="${INPUT_BASE_DIR}/nation"
REGION_INPUT_DIR="${INPUT_BASE_DIR}/region"

OUTPUT_BASE_DIR="/export/data1/tpch/sideways/sf${SF}"
LINEITEM_OUTPUT_DIR=${OUTPUT_BASE_DIR}/lineitem
ORDERS_OUTPUT_DIR=${OUTPUT_BASE_DIR}/orders

SORT_INPUT_BASE_DIR="${OUTPUT_BASE_DIR}/tmp/sort_input"

SORT_LINEITEM_INPUT_DIR="${SORT_INPUT_BASE_DIR}/lineitem"
SORT_ORDERS_INPUT_DIR="${SORT_INPUT_BASE_DIR}/orders"

DENORM_TMP_DIR="${OUTPUT_BASE_DIR}/tmp/denorm"
SORT_TMP_DIR="${OUTPUT_BASE_DIR}/tmp/sort"
DUCKDB_TMP_DIR="${OUTPUT_BASE_DIR}/tmp/duckdb"
#FINAL_FILE="denormalized_sorted.tbl"
# DuckDB is used for converting data
DB_FILE=":memory:"

# Clear any leftover intermediate files from a previous aborted run so
# that split/rename never see stale numbering.
for d in "${DENORM_TMP_DIR}" "${SORT_TMP_DIR}" "${DUCKDB_TMP_DIR}" \
         "${SORT_LINEITEM_INPUT_DIR}" "${SORT_ORDERS_INPUT_DIR}"; do
    [ -d "${d}" ] && find "${d}" -mindepth 1 -delete 2>/dev/null || true
done

mkdir -p ${OUTPUT_BASE_DIR} ${DENORM_TMP_DIR} ${SORT_TMP_DIR} ${DUCKDB_TMP_DIR} ${SORT_LINEITEM_INPUT_DIR} ${SORT_ORDERS_INPUT_DIR}
mkdir -p ${LINEITEM_OUTPUT_DIR} ${ORDERS_OUTPUT_DIR}

SORT_MEM="50%"
THREADS=$(nproc)
# LC_ALL=C is required for fast parsing
export LC_ALL=C

ensure_empty_dir() {
    local target_dir="$1"

    if [ ! -d "$target_dir" ]; then
        echo "Error: Directory '$target_dir' does not exist."
        return 0
    fi

    if [ -n "$(ls -A "$target_dir")" ]; then
        echo "Directory '${target_dir}' is not empty."
        echo "Clean ${target_dir}?"
        
        read -p "Clean the directory? [y/N]: " CONFIRM

        case "$CONFIRM" in
            [yY][eE][sS]|[yY]) 
                echo "Cleaning directory..."
                find "$target_dir" -mindepth 1 -delete
                echo "Directory cleaned."
                echo "Please rerun the script."
		exit 0
                ;;
            *)
                echo "Operation aborted by user."
                exit 1 
                ;;
        esac
    fi
}

ensure_empty_dir $SORT_LINEITEM_INPUT_DIR
ensure_empty_dir $SORT_ORDERS_INPUT_DIR
ensure_empty_dir ${LINEITEM_OUTPUT_DIR}
ensure_empty_dir ${ORDERS_OUTPUT_DIR}

STEP1_START=$(date +%s)
echo "[Step 1] Generating denormalized LINEITEM data using DuckDB..."
duckdb "$DB_FILE" <<EOF
-- Memory configuration
SET memory_limit = '128GB';
SET threads = ${THREADS};
SET temp_directory = '${DUCKDB_TMP_DIR}';

-- ==========================================
-- 1. TPC-H view definition
-- ==========================================

CREATE OR REPLACE VIEW lineitem AS 
SELECT * FROM read_csv('${LINEITEM_INPUT_DIR}/lineitem.tbl*', delim='|', header=False,
    columns={'l_orderkey': 'BIGINT', 'l_partkey': 'BIGINT', 'l_suppkey': 'BIGINT', 'l_linenumber': 'INTEGER', 'l_quantity': 'DECIMAL(15,2)', 'l_extendedprice': 'DECIMAL(15,2)', 'l_discount': 'DECIMAL(15,2)', 'l_tax': 'DECIMAL(15,2)', 'l_returnflag': 'VARCHAR', 'l_linestatus': 'VARCHAR', 'l_shipdate': 'DATE', 'l_commitdate': 'DATE', 'l_receiptdate': 'DATE', 'l_shipinstruct': 'VARCHAR', 'l_shipmode': 'VARCHAR', 'l_comment': 'VARCHAR'});

CREATE OR REPLACE VIEW orders AS 
SELECT * FROM read_csv('${ORDERS_INPUT_DIR}/orders.tbl*', delim='|', header=False,
    columns={'o_orderkey': 'BIGINT', 'o_custkey': 'INTEGER', 'o_orderstatus': 'VARCHAR', 'o_totalprice': 'DECIMAL(15,2)', 'o_orderdate': 'DATE', 'o_orderpriority': 'VARCHAR', 'o_clerk': 'VARCHAR', 'o_shippriority': 'INTEGER', 'o_comment': 'VARCHAR'});

CREATE OR REPLACE VIEW customer AS 
SELECT * FROM read_csv('${CUSTOMER_INPUT_DIR}/customer.tbl*', delim='|', header=False,
    columns={'c_custkey': 'INTEGER', 'c_name': 'VARCHAR', 'c_address': 'VARCHAR', 'c_nationkey': 'INTEGER', 'c_phone': 'VARCHAR', 'c_acctbal': 'DECIMAL(15,2)', 'c_mktsegment': 'VARCHAR', 'c_comment': 'VARCHAR'});

CREATE OR REPLACE VIEW nation AS 
SELECT * FROM read_csv('${NATION_INPUT_DIR}/nation.tbl*', delim='|', header=False,
    columns={'n_nationkey': 'INTEGER', 'n_name': 'VARCHAR', 'n_regionkey': 'INTEGER', 'n_comment': 'VARCHAR'});

CREATE OR REPLACE VIEW region AS 
SELECT * FROM read_csv('${REGION_INPUT_DIR}/region.tbl*', delim='|', header=False,
    columns={'r_regionkey': 'INTEGER', 'r_name': 'VARCHAR', 'r_comment': 'VARCHAR'});

-- ==========================================
-- 2. Join & denormanized data
--    Output data with sort keys
-- ==========================================

COPY (
    SELECT 
        l.l_orderkey, l.l_partkey, l.l_suppkey, l.l_linenumber,  -- 1 - 4
        l.l_quantity, l.l_extendedprice, l.l_discount, l.l_tax,  -- 5 - 8
        l.l_returnflag,                                -- 9
        l.l_linestatus,                                -- 10
	l.l_shipdate,                                  -- 11 (Key 4 Card 2516)
	l.l_commitdate,                                -- 12
	l.l_receiptdate,                               -- 13
        l.l_shipinstruct, l.l_shipmode, l.l_comment,   -- 14 - 16

        -- [Sort Keys] from other tables
        r.r_name,        -- Key 1 (Card: 5)  -- 17
        c.c_mktsegment,  -- Key 2 (Card: 5)  -- 18
        o.o_orderdate,   -- Key 3 (Card: 2406)  -- 19
        
	-- Add tailing '|' for parsing
	'' AS dummy
    FROM lineitem l
    JOIN orders o ON l.l_orderkey = o.o_orderkey
    JOIN customer c ON o.o_custkey = c.c_custkey
    JOIN nation n ON c.c_nationkey = n.n_nationkey
    JOIN region r ON n.n_regionkey = r.r_regionkey
) 
TO '${SORT_LINEITEM_INPUT_DIR}'
(
    FORMAT CSV, 
    DELIMITER '|',
    HEADER FALSE,
    PER_THREAD_OUTPUT true,
    QUOTE ''
);
EOF
step_time "Step1_LINEITEM_denormalize" "${STEP1_START}"

if [ "${DELETE_INPUT}" -eq 1 ]; then
    # LINEITEM denorm CSV is now self-contained in ${SORT_LINEITEM_INPUT_DIR};
    # the raw dbgen output for lineitem is no longer referenced by
    # later steps (orders denorm uses the orders/ directory directly,
    # not the lineitem one). Drop it to reclaim disk space.
    echo "[--delete-input] removing ${LINEITEM_INPUT_DIR}"
    rm -rf "${LINEITEM_INPUT_DIR}"
fi

STEP2_START=$(date +%s)
echo "[Step 2] Sorting LINEITEM data by cardinality keys..."

TOTAL_LINES=$(find "${SORT_LINEITEM_INPUT_DIR}" -name "*.csv" -print0 | xargs -0 cat | wc -l)
LINES_PER_FILE=$(( ($TOTAL_LINES + $NUM_SPLITS - 1) / $NUM_SPLITS ))

ls ${SORT_LINEITEM_INPUT_DIR}

# Width of the numeric suffix passed to `split -a`. Must be wide
# enough to hold NUM_SPLITS distinct values; otherwise GNU split
# silently switches to its "extended" form (e.g. .9000, .9001, ...)
# which breaks the subsequent rename pass.
SPLIT_SUFFIX_LEN=${#NUM_SPLITS}

OUTPUT_DIR=${OUTPUT_BASE_DIR}/lineitem
mkdir -p ${OUTPUT_DIR}
sort -T "${SORT_TMP_DIR}" \
     -t '|' \
     -k 17,17 \
     -k 18,18 \
     -k 19,19 \
     -k 11,11 \
     --buffer-size="${SORT_MEM}" \
     --parallel="${THREADS}" \
     "${SORT_LINEITEM_INPUT_DIR}"/*.csv \
| split -l "${LINES_PER_FILE}" \
        -d \
        -a "${SPLIT_SUFFIX_LEN}" \
        - \
        "${OUTPUT_DIR}/lineitem.tbl."

# Rename .0..N-1  →  .1..N  (loader expects 1-based indexing).
# Sort numerically and descending so the moves don't collide.
python3 <<EOF
import os
import glob

target_dir = "${OUTPUT_DIR}"
pattern = os.path.join(target_dir, "lineitem.tbl.[0-9]*")

files = [f for f in glob.glob(pattern) if f.rsplit('.', 1)[1].isdigit()]
files.sort(key=lambda p: int(p.rsplit('.', 1)[1]), reverse=True)

for old_path in files:
    base_path, suffix = old_path.rsplit('.', 1)
    new_num = int(suffix) + 1
    new_path = f"{base_path}.{new_num}"
    os.rename(old_path, new_path)
    print(f"Renamed: {old_path} -> {new_path}")
EOF

find "$SORT_LINEITEM_INPUT_DIR" -mindepth 1 -delete
step_time "Step2_LINEITEM_sort_and_split" "${STEP2_START}"

STEP3_START=$(date +%s)
echo "[Step 3] Generating denormalized ORDERS data using DuckDB..."

duckdb "$DB_FILE" <<EOF
-- Memory configuration
SET memory_limit = '128GB';
SET threads = ${THREADS};
SET temp_directory = '${DUCKDB_TMP_DIR}';

-- ==========================================
-- 1. TPC-H view definition
-- ==========================================

CREATE OR REPLACE VIEW orders AS 
SELECT * FROM read_csv('${ORDERS_INPUT_DIR}/orders.tbl*', delim='|', header=False,
    columns={'o_orderkey': 'BIGINT', 'o_custkey': 'INTEGER', 'o_orderstatus': 'VARCHAR', 'o_totalprice': 'DECIMAL(15,2)', 'o_orderdate': 'DATE', 'o_orderpriority': 'VARCHAR', 'o_clerk': 'VARCHAR', 'o_shippriority': 'INTEGER', 'o_comment': 'VARCHAR'});

CREATE OR REPLACE VIEW customer AS 
SELECT * FROM read_csv('${CUSTOMER_INPUT_DIR}/customer.tbl*', delim='|', header=False,
    columns={'c_custkey': 'INTEGER', 'c_name': 'VARCHAR', 'c_address': 'VARCHAR', 'c_nationkey': 'INTEGER', 'c_phone': 'VARCHAR', 'c_acctbal': 'DECIMAL(15,2)', 'c_mktsegment': 'VARCHAR', 'c_comment': 'VARCHAR'});

CREATE OR REPLACE VIEW nation AS 
SELECT * FROM read_csv('${NATION_INPUT_DIR}/nation.tbl*', delim='|', header=False,
    columns={'n_nationkey': 'INTEGER', 'n_name': 'VARCHAR', 'n_regionkey': 'INTEGER', 'n_comment': 'VARCHAR'});

CREATE OR REPLACE VIEW region AS 
SELECT * FROM read_csv('${REGION_INPUT_DIR}/region.tbl*', delim='|', header=False,
    columns={'r_regionkey': 'INTEGER', 'r_name': 'VARCHAR', 'r_comment': 'VARCHAR'});

COPY (
    SELECT 
        
        -- [Attributes]
        o.o_orderkey,      -- 1
        o.o_custkey,       -- 2
        o.o_orderstatus,   -- 3
        o.o_totalprice,    -- 4
        o.o_orderdate,     -- 5 (Key 3 Card 2406)
        o.o_orderpriority, -- 6
        o.o_clerk,         -- 7
        o.o_shippriority,  -- 8
        o.o_comment,       -- 9

        -- [Additional Sort Keys]
	r.r_name,        -- 10 (Key 1 Card 5)
	c.c_mktsegment,  -- 11 (Key 2 Card 5)
        
	-- Add tailing '|' for parsing
	'' AS dummy
    FROM orders o
    JOIN customer c ON o.o_custkey = c.c_custkey
    JOIN nation n ON c.c_nationkey = n.n_nationkey
    JOIN region r ON n.n_regionkey = r.r_regionkey
) 
TO '${SORT_ORDERS_INPUT_DIR}'
(
    FORMAT CSV, 
    DELIMITER '|', 
    HEADER FALSE, 
    PER_THREAD_OUTPUT true,
    QUOTE ''
);
EOF
step_time "Step3_ORDERS_denormalize" "${STEP3_START}"

if [ "${DELETE_INPUT}" -eq 1 ]; then
    # ORDERS denorm CSV is now self-contained in ${SORT_ORDERS_INPUT_DIR}.
    # The raw dbgen orders/ directory is no longer needed.
    echo "[--delete-input] removing ${ORDERS_INPUT_DIR}"
    rm -rf "${ORDERS_INPUT_DIR}"
fi

STEP4_START=$(date +%s)
echo "[Step 4] Sorting ORDERS data by cardinality keys..."

TOTAL_LINES=$(find "${SORT_ORDERS_INPUT_DIR}" -name "*.csv" -print0 | xargs -0 cat | wc -l)
LINES_PER_FILE=$(( ($TOTAL_LINES + $NUM_SPLITS - 1) / $NUM_SPLITS ))

ls ${SORT_ORDERS_INPUT_DIR}

OUTPUT_DIR=${OUTPUT_BASE_DIR}/orders
mkdir -p ${OUTPUT_DIR}
sort -T "${SORT_TMP_DIR}" \
     -t '|' \
     -k 10,10 \
     -k 11,11 \
     -k 5,5 \
     --buffer-size="${SORT_MEM}" \
     --parallel="${THREADS}" \
     "${SORT_ORDERS_INPUT_DIR}"/*.csv \
| split -l "${LINES_PER_FILE}" \
        -d \
        -a "${SPLIT_SUFFIX_LEN}" \
        - \
        "${OUTPUT_DIR}/orders.tbl."

# Rename .0..N-1  →  .1..N  (loader expects 1-based indexing).
# Sort numerically and descending so the moves don't collide.
python3 <<EOF
import os
import glob

target_dir = "${OUTPUT_DIR}"
pattern = os.path.join(target_dir, "orders.tbl.[0-9]*")

files = [f for f in glob.glob(pattern) if f.rsplit('.', 1)[1].isdigit()]
files.sort(key=lambda p: int(p.rsplit('.', 1)[1]), reverse=True)

for old_path in files:
    base_path, suffix = old_path.rsplit('.', 1)
    new_num = int(suffix) + 1
    new_path = f"{base_path}.{new_num}"
    os.rename(old_path, new_path)
    print(f"Renamed: {old_path} -> {new_path}")
EOF

find "$SORT_ORDERS_INPUT_DIR" -mindepth 1 -delete
find "${SORT_TMP_DIR}" -mindepth 1 -delete 2>/dev/null || true
find "${DUCKDB_TMP_DIR}" -mindepth 1 -delete 2>/dev/null || true
step_time "Step4_ORDERS_sort_and_split" "${STEP4_START}"


## Prepaing other table directory
## (dim tables are NOT touched by --delete-input: they are referenced
##  by symlinks below and are required by downstream scripts.)
tables=(supplier partsupp part customer nation region)

for table in ${tables[@]}
do
  SRCDIR=${INPUT_BASE_DIR}/${table}
  DSTDIR=${OUTPUT_BASE_DIR}/${table}
  if [ -e "${DSTDIR}" ] || [ -L "${DSTDIR}" ]; then
    echo Skip creating ${DSTDIR}
  else
    ln -s ${SRCDIR} ${DSTDIR}
  fi
done

echo ""
echo "=== TPC-H sideways pruning complete (SF=${SF}) ==="
echo "Output: ${OUTPUT_BASE_DIR}/"
echo "  lineitem/ : $(ls "${LINEITEM_OUTPUT_DIR}" | wc -l) chunks"
echo "  orders/   : $(ls "${ORDERS_OUTPUT_DIR}" | wc -l) chunks"
step_time "Total" "${SCRIPT_START}"
