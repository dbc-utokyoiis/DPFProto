#!/bin/bash
# 01_setup.sh — Setup all benchmark systems
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
  esac
done

SYSTEMS=(duckdb polars spark-rapids)

for sys in "${SYSTEMS[@]}"; do
  echo "=== Setup: ${sys} ==="
  if [ "${DRY_RUN}" -eq 1 ]; then
    bash "${SCRIPT_DIR}/${sys}/setup.sh" --dry-run
  else
    bash "${SCRIPT_DIR}/${sys}/setup.sh"
  fi
done
