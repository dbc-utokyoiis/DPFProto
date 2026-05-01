#!/bin/bash
# systems/common.sh — shared configuration across all benchmark systems

LAYOUT=${LAYOUT:-sideways}
SF=${SF:-100}
NTHREADS=${NTHREADS:-$(nproc)}
MEMORY=${MEMORY:-384GB}

# TPC-H paths
# TODO: NORMAL パスは不要になったら削除する
if [ "${LAYOUT}" = "sideways" ]; then
  TPCH_INPUT_BASE=/export/data1/tpch/sideways/sf
  TPCH_PARQUET_BASE=/export/data1/tpch/duckdb/sideways/parquet
else
  TPCH_INPUT_BASE=/export/data1/tpch/input
  TPCH_PARQUET_BASE=/export/data1/tpch/duckdb/parquet
fi

# SSB paths
# TODO: NORMAL パスは不要になったら削除する
if [ "${LAYOUT}" = "sideways" ]; then
  SSB_INPUT_BASE=/export/data1/ssb/sideways/sf${SF}
  SSB_PARQUET_BASE=/export/data1/ssb/duckdb/sideways/parquet/sf${SF}
else
  SSB_INPUT_BASE=/export/data1/ssb/input${SF}
  SSB_PARQUET_BASE=/export/data1/ssb/duckdb/parquet/sf${SF}
fi

# Temp directory (used by DuckDB via PRAGMA, Polars via $TMPDIR, Spark via spark.local.dir)
export TMPDIR=${TMPDIR:-/export/data1/tmp}
DUCKDB_TEMP_DIR="${TMPDIR}/duckdb"
mkdir -p "${TMPDIR}" "${DUCKDB_TEMP_DIR}"

# Table lists
TPCH_TABLES=(region nation part supplier partsupp customer orders lineitem)
SSB_TABLES=(lineorder customer date part supplier)

# ---------- sudo credential cache ----------
# Benchmarks require "echo 3 | sudo tee /proc/sys/vm/drop_caches" between
# iterations to ensure cold-cache measurements.  Prompt for the password
# once up front so that later sudo calls don't stall mid-run.
sudo -v || { echo "ERROR: sudo authentication failed." >&2; exit 1; }
