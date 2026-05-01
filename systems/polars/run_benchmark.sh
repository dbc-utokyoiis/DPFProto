#!/bin/bash
# polars/run_benchmark.sh — Run TPC-H / SSB benchmarks on Polars GPU over Parquet
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SYSTEMS_DIR}/common.sh"

BENCHMARK=${BENCHMARK:-all}
ITERATIONS=${ITERATIONS:-10}
ENGINE=${ENGINE:-gpu}
LOG_BASE="${SYSTEMS_DIR}/logs/polars"
ANSWERS_BASE="${SYSTEMS_DIR}/../answers"
VENV="${SCRIPT_DIR}/venv"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
  esac
done

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "[dry-run] polars/run_benchmark.sh"
  echo "[dry-run] BENCHMARK=${BENCHMARK}, SF=${SF}, ENGINE=${ENGINE}, Iterations=${ITERATIONS}"
  echo "[dry-run] Log output: ${LOG_BASE}/"
  exit 0
fi

# Check GDS setup (GPU engine requires GPUDirect Storage)
if [ "${ENGINE}" = "gpu" ]; then
  bash "${SYSTEMS_DIR}/check_gds.sh" || exit 1
fi

source "${VENV}/bin/activate"

export CUFILE_ENV_PATH_JSON="${SYSTEMS_DIR}/cufile.json"
export LIBCUDF_CUFILE_POLICY=GDS
export KVIKIO_NTHREADS="${KVIKIO_NTHREADS:-48}"
export SF BENCHMARK ITERATIONS ENGINE
export TPCH_PARQUET_BASE SSB_PARQUET_BASE
export LOG_BASE ANSWERS_BASE

python3 "${SCRIPT_DIR}/bench.py"
