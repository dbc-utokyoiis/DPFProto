#!/usr/bin/env bash
# run_dbgen.sh — Generate TPC-H data for one or more scale factors.
#
# Each SF is generated under /export/data1/tpch/input${SF}/ with the
# per-table directory layout expected by the loader:
#   input${SF}/{lineitem,orders,customer,part,partsupp,supplier,nation,region}/
#
# Usage:
#   scripts/tpch/run_dbgen.sh <SF>[,<SF>...] [<threads>] [<destbase>]
#
# Examples:
#   scripts/tpch/run_dbgen.sh 100
#   scripts/tpch/run_dbgen.sh 50,100,200,300,400
#   scripts/tpch/run_dbgen.sh 100 48
#   scripts/tpch/run_dbgen.sh 100 48 /scratch/tpch
#
# Prerequisites:
#   - The TPC-H dbgen kit must be downloaded and built by the user.
#     The official kit is distributed by the Transaction Processing
#     Performance Council at https://www.tpc.org/tpch/ and ships with
#     the TPC End-User License Agreement. The user is responsible for
#     accepting the EULA and running `make` inside the kit's dbgen/
#     directory.
#   - Point TPCH_DBGEN_DIR to the directory containing the built
#     `dbgen` executable. Either export it in the environment or set it
#     in scripts/common.sh. If unset, the script falls back to
#     tpch-tool/tpch_2_17_0/dbgen/ under the repository root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DBGEN_DIR="${TPCH_DBGEN_DIR:-${REPO_ROOT}/tpch-tool/tpch_2_17_0/dbgen}"

if [ $# -lt 1 ]; then
    cat >&2 <<EOF
Usage: $(basename "$0") <SF>[,<SF>...] [<threads>] [<destbase>]

  <SF>        : scale factor(s), comma-separated (e.g. "100" or "50,100,200,300,400")
  <threads>   : number of parallel dbgen workers (default: \$(nproc))
  <destbase>  : parent directory; each SF becomes <destbase>/input<SF>
                (default: /export/data1/tpch)
EOF
    exit 1
fi

IFS=',' read -ra SFS <<< "$1"
NTHR="${2:-$(nproc)}"
DESTBASE="${3:-/export/data1/tpch}"

if [ ! -x "${DBGEN_DIR}/dbgen" ]; then
    echo "ERROR: ${DBGEN_DIR}/dbgen not found or not executable." >&2
    echo "  Download the TPC-H dbgen kit from https://www.tpc.org/tpch/," >&2
    echo "  accept the TPC EULA, build it (cd <kit>/dbgen && make), and" >&2
    echo "  set TPCH_DBGEN_DIR to point at the built dbgen directory:" >&2
    echo "    export TPCH_DBGEN_DIR=/path/to/tpch_kit/dbgen" >&2
    echo "  (or set it in scripts/common.sh)" >&2
    exit 1
fi

echo "=============================================================="
echo "  TPC-H dbgen"
echo "    SFs     : ${SFS[*]}"
echo "    threads : ${NTHR}"
echo "    destbase: ${DESTBASE}"
echo "=============================================================="

for SF in "${SFS[@]}"; do
    DIR="${DESTBASE}/input${SF}"
    echo ""
    echo "--- SF=${SF} → ${DIR} ---"
    mkdir -p "${DIR}"
    bash "${SCRIPT_DIR}/dbgen.sh" -A "${DBGEN_DIR}" "${SF}" "${NTHR}" "${DIR}"
done

echo ""
echo "=============================================================="
echo "  Done. Generated SFs: ${SFS[*]}"
echo "=============================================================="
