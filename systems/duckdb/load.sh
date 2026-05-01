#!/bin/bash
# duckdb/load.sh — Load TPC-H / SSB data as Parquet
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
BENCHMARK=${BENCHMARK:-all}

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
  esac
done

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "[dry-run] duckdb/load.sh: BENCHMARK=${BENCHMARK} SF=${SF} SF=${SF}"
  case "${BENCHMARK}" in
    tpch) echo "[dry-run] Would run 10_LOAD_PARQUET_TPCH.sh" ;;
    ssb)  echo "[dry-run] Would run 10_LOAD_PARQUET_SSB.sh" ;;
    all)  echo "[dry-run] Would run 10_LOAD_PARQUET_TPCH.sh and 10_LOAD_PARQUET_SSB.sh" ;;
    *)    echo "Unknown BENCHMARK: ${BENCHMARK} (use tpch, ssb, or all)" >&2; exit 1 ;;
  esac
  exit 0
fi

# Check for existing output directories
check_output_dir() {
  local dir="$1"
  if [ -d "${dir}" ] && [ "$(ls -A "${dir}" 2>/dev/null)" ]; then
    echo "Output directory already exists and is not empty: ${dir}"
    read -rp "Delete and recreate? [y/N] " answer
    if [ "${answer}" = "y" ] || [ "${answer}" = "Y" ]; then
      rm -rf "${dir}"
      echo "Deleted ${dir}. Please rerun this script."
      exit 1
    else
      echo "Skipping load (using existing data)."
      exit 0
    fi
  fi
}

# Check input directory exists
check_input_dir() {
  local dir="$1"
  local bench="$2"
  if [ ! -d "${dir}" ]; then
    echo "ERROR: Input directory not found: ${dir}" >&2
    echo "Please run dbgen to generate ${bench} SF${SF} data first." >&2
    exit 1
  fi
}

SYSTEMS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_BASE="${SYSTEMS_DIR}/logs/load"
mkdir -p "${LOG_BASE}"

run_load() {
  local name="$1"
  local script="$2"
  local log_file="${LOG_BASE}/${name}_sf${SF}.txt"
  local start end elapsed

  echo "=== Loading ${name} SF${SF} ==="
  echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
  start=$(date +%s.%N)
  bash "${script}" 2>&1 | tee "${log_file}"
  end=$(date +%s.%N)
  elapsed=$(echo "$end - $start" | bc)
  printf "Total load time: %.1fs\n" "${elapsed}" | tee -a "${log_file}"
  echo "Log: ${log_file}"
}

case "${BENCHMARK}" in
  tpch)
    check_input_dir "${INPUT_BASE}${SF}" "TPC-H"
    check_output_dir "${PARQUET_OUTPUT_BASE}/sf${SF}"
    run_load "tpch" "${SCRIPT_DIR}/10_LOAD_PARQUET_TPCH.sh"
    ;;
  ssb)
    check_input_dir "${SSB_INPUT_BASE}" "SSB"
    check_output_dir "${SSB_PARQUET_OUTPUT_BASE}"
    run_load "ssb" "${SCRIPT_DIR}/10_LOAD_PARQUET_SSB.sh"
    ;;
  all)
    check_input_dir "${INPUT_BASE}${SF}" "TPC-H"
    check_input_dir "${SSB_INPUT_BASE}" "SSB"
    check_output_dir "${PARQUET_OUTPUT_BASE}/sf${SF}"
    check_output_dir "${SSB_PARQUET_OUTPUT_BASE}"
    run_load "tpch" "${SCRIPT_DIR}/10_LOAD_PARQUET_TPCH.sh"
    run_load "ssb" "${SCRIPT_DIR}/10_LOAD_PARQUET_SSB.sh"
    ;;
  *)
    echo "Unknown BENCHMARK: ${BENCHMARK} (use tpch, ssb, or all)" >&2
    exit 1
    ;;
esac
