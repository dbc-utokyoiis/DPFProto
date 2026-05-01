#!/bin/bash
# duckdb/setup.sh — Install DuckDB CLI (kazamatsuri environment version)
set -eu

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
  esac
done

DUCKDB_VERSION="1.4.4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/bin"
DUCKDB_BIN="${INSTALL_DIR}/duckdb"

if [ -x "${DUCKDB_BIN}" ]; then
  CURRENT=$("${DUCKDB_BIN}" -csv -noheader -c "SELECT version()" | sed 's/^v//')
  if [ "${CURRENT}" = "${DUCKDB_VERSION}" ]; then
    echo "DuckDB v${DUCKDB_VERSION} is already cached at ${DUCKDB_BIN}"
    exit 0
  fi
  echo "DuckDB v${CURRENT} found, replacing with v${DUCKDB_VERSION}..."
fi

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "[dry-run] Would download and install DuckDB v${DUCKDB_VERSION} to ${DUCKDB_BIN}"
  exit 0
fi

mkdir -p "${INSTALL_DIR}"
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

echo "Downloading DuckDB v${DUCKDB_VERSION}..."
curl -fsSL "https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/duckdb_cli-linux-amd64.zip" \
  -o "${TMPDIR}/duckdb.zip"

unzip -o "${TMPDIR}/duckdb.zip" -d "${TMPDIR}"
mv "${TMPDIR}/duckdb" "${DUCKDB_BIN}"
chmod +x "${DUCKDB_BIN}"

echo "DuckDB v${DUCKDB_VERSION} installed to ${DUCKDB_BIN}"
