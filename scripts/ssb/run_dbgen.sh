#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# SSB dbgen runner — generates SSB benchmark data
#
# Usage:
#   ./run_dbgen.sh <scale_factor> <num_threads> <destdir>
#
# Examples:
#   ./run_dbgen.sh 1   1  /export/data1/ssb/sf1     # SF=1, single-threaded
#   ./run_dbgen.sh 100 48 /export/data1/ssb/sf100    # SF=100, 48 threads for lineorder
#
# Dimension tables (supplier, customer, part) are generated in parallel using
# dbgen's -C/-S (chunk) flags so that each chunk is a separate file.
# This is required for GOLAP compression sampling to get multiple samples.
#
# The date table is always single-threaded (only 2,556 rows).
# lineorder (the fact table) is generated in parallel with -C/-S.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DBGEN_DIR="${REPO_DIR}/ssb"

if [ $# -ne 3 ]; then
  echo "Usage: $(basename "$0") <scale_factor> <num_threads> <destdir>"
  echo ""
  echo "  scale_factor  SF value (1, 10, 100, ...)"
  echo "  num_threads   number of parallel processes for lineorder generation"
  echo "  destdir       output directory"
  exit 1
fi

sf="$1"
nthr="$2"
destdir="$(mkdir -p "$3" && realpath "$3")"

if [ ! -x "${DBGEN_DIR}/dbgen" ]; then
  echo "ERROR: ${DBGEN_DIR}/dbgen not found or not executable" >&2
  exit 1
fi

trap 'jobs -pr | xargs -r kill 2>/dev/null || true' EXIT

export DSS_CONFIG="${DBGEN_DIR}/"

echo "=== SSB dbgen ==="
echo "  Scale Factor : ${sf}"
echo "  Threads      : ${nthr} (lineorder only)"
echo "  Output       : ${destdir}"
echo ""

###############################################################################
# Date table (single-threaded — only 2,556 rows, constant)
###############################################################################
echo "Generating date..."
mkdir -p "${destdir}/date"
pushd "${destdir}/date" > /dev/null
"${DBGEN_DIR}/dbgen" -T d -f -s "${sf}"
popd > /dev/null
echo "  date done."

###############################################################################
# Dimension tables with chunked generation (supplier, customer, part)
# Uses dbgen -C/-S to directly produce .tbl.1, .tbl.2, ... files.
# Number of chunks = min(nthr, sf), minimum 1.
###############################################################################
generate_dim_chunked() {
  local tblname="$1" tblopt="$2" nchunks="$3"

  echo "Generating ${tblname} (${nchunks} chunks)..."
  mkdir -p "${destdir}/${tblname}"
  pushd "${destdir}/${tblname}" > /dev/null

  if [ "${nchunks}" -le 1 ]; then
    "${DBGEN_DIR}/dbgen" -T "${tblopt}" -f -s "${sf}"
  else
    for i in $(seq 1 "${nchunks}"); do
      (
        "${DBGEN_DIR}/dbgen" -T "${tblopt}" -f -s "${sf}" -C "${nchunks}" -S "${i}"
      ) &
    done
    wait
  fi

  popd > /dev/null
  echo "  ${tblname} done."
}

dim_nchunks="${nthr}"
[ "${dim_nchunks}" -lt 1 ] && dim_nchunks=1

generate_dim_chunked "supplier" "s" "${dim_nchunks}"
generate_dim_chunked "customer" "c" "${dim_nchunks}"
generate_dim_chunked "part"     "p" "${dim_nchunks}"

###############################################################################
# Fact table: lineorder (multi-threaded)
#   -T l  lineorder  (~6M * SF rows — the bulk of the data)
###############################################################################
echo "Generating lineorder (${nthr} threads)..."
mkdir -p "${destdir}/lineorder"
pushd "${destdir}/lineorder" > /dev/null

if [ "${nthr}" -eq 1 ]; then
  "${DBGEN_DIR}/dbgen" -T l -f -s "${sf}"
else
  for i in $(seq 1 "${nthr}"); do
    (
      "${DBGEN_DIR}/dbgen" -T l -f -s "${sf}" -C "${nthr}" -S "${i}"
      echo "  lineorder chunk ${i}/${nthr} done."
    ) &
  done
  wait
fi

popd > /dev/null
echo "  lineorder done."

echo ""
echo "=== All tables generated in ${destdir} ==="
ls -lhS "${destdir}"/*/ | head -30 || true
