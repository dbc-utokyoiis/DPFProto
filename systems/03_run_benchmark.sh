#!/bin/bash
# 03_run_benchmark.sh — Run benchmarks on all systems
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1; ARGS+=("$arg") ;;
    *) ARGS+=("$arg") ;;
  esac
done

SYSTEMS=(duckdb polars spark-rapids)

for sys in "${SYSTEMS[@]}"; do
  echo "=== Benchmark: ${sys} ==="
  bash "${SCRIPT_DIR}/${sys}/run_benchmark.sh" "${ARGS[@]}"
done
