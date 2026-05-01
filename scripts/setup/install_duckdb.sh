#!/usr/bin/env bash
# install_duckdb.sh — Install DuckDB 1.4.4 CLI (Linux x86_64).
#
# The sideways-pruning step (scripts/golap/01_sideways_pruning.sh and
# scripts/golap/02_ssb_sideways_pruning.sh) invokes `duckdb` to
# denormalize and sort the input tables. This script downloads the
# prebuilt CLI from the official GitHub release and installs it so
# that `duckdb` is available on $PATH.
#
# Usage:
#   scripts/setup/install_duckdb.sh                # into /usr/local/bin (needs sudo)
#   PREFIX=$HOME/.local scripts/setup/install_duckdb.sh   # user-local install
set -euo pipefail

VERSION="v1.4.4"
ARCHIVE="duckdb_cli-linux-amd64.zip"
URL="https://github.com/duckdb/duckdb/releases/download/${VERSION}/${ARCHIVE}"

PREFIX="${PREFIX:-/usr/local}"
BIN_DIR="${PREFIX}/bin"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "=== Installing DuckDB ${VERSION} to ${BIN_DIR} ==="
curl -L --fail -o "${TMP_DIR}/${ARCHIVE}" "${URL}"
unzip -o "${TMP_DIR}/${ARCHIVE}" -d "${TMP_DIR}"

if [ ! -w "${BIN_DIR}" ]; then
    echo "  (installing with sudo — ${BIN_DIR} is not writable)"
    sudo install -m 0755 "${TMP_DIR}/duckdb" "${BIN_DIR}/duckdb"
else
    install -m 0755 "${TMP_DIR}/duckdb" "${BIN_DIR}/duckdb"
fi

echo ""
echo "Installed:"
"${BIN_DIR}/duckdb" --version
