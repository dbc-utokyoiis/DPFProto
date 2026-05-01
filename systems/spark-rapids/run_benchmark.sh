#!/bin/bash
# spark-rapids/run_benchmark.sh — Run TPC-H / SSB benchmarks on Spark-RAPIDS
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SYSTEMS_DIR}/common.sh"

BENCHMARK=${BENCHMARK:-all}
MODE=${MODE:-gpu}

ARGS=()
for arg in "$@"; do
  ARGS+=("$arg")
done

if [[ "${BENCHMARK}" == "tpch" || "${BENCHMARK}" == "all" ]]; then
  bash "${SCRIPT_DIR}/run_tpch_benchmark.sh" --mode "${MODE}" "${ARGS[@]+"${ARGS[@]}"}"
fi

if [[ "${BENCHMARK}" == "ssb" || "${BENCHMARK}" == "all" ]]; then
  bash "${SCRIPT_DIR}/run_ssb_benchmark.sh" --mode "${MODE}" "${ARGS[@]+"${ARGS[@]}"}"
fi
