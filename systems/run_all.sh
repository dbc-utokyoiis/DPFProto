#!/bin/bash
# run_all.sh — Setup, load, and benchmark all systems
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN="--dry-run" ;;
  esac
done

bash "${SCRIPT_DIR}/01_setup.sh" ${DRY_RUN}
bash "${SCRIPT_DIR}/02_load.sh" ${DRY_RUN}
bash "${SCRIPT_DIR}/03_run_benchmark.sh" ${DRY_RUN}
