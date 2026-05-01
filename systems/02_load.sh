#!/bin/bash
# 02_load.sh — Load data as Parquet (via DuckDB)
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

echo "=== Load: duckdb ==="
bash "${SCRIPT_DIR}/duckdb/load.sh" "${ARGS[@]}"
