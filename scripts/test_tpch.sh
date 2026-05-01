#!/bin/bash
# test_tpch.sh — Run tpchdb correctness tests for 4 execution modes.
#
# Usage:
#   scripts/test_tpch.sh [-q QUERIES] [-x MODES] [-s SF] [-w THREADS] [-l LOGDIR]
#
# Options:
#   -q QUERIES  Comma-separated query names (default: q1,q3,q5,q6,q13,q16)
#   -x MODES    Comma-separated execution modes (default: all)
#               Valid: gidp+bam, gidp+bam+fusion, gidp, datapathfusion
#   -s SF       Scale factor (default: 1). Used for log directory naming.
#   -w THREADS  Worker threads for gidp mode (default: 32)
#   -t TIMEOUT  Per-test timeout in seconds (default: 10)
#   -r RETRIES  Max retries on timeout (default: 3)
#   -l LOGDIR   Log directory (default: logs/test_tpch/<timestamp>)
#
# Environment:
#   DEV_BAM   BaM devices   (default: /dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3)
#   DEV_NVME  NVMe devices  (default: /dev/nvme0n1p1,/dev/nvme1n1p1,/dev/nvme2n1p1,/dev/nvme3n1p1)
#
# Examples:
#   scripts/test_tpch.sh                                        # all queries, all modes
#   scripts/test_tpch.sh -q q6 -s 100                           # single query
#   scripts/test_tpch.sh -x datapathfusion -q q1,q3,q6 -s 100   # single mode
#   scripts/test_tpch.sh -x gidp,datapathfusion -s 100           # multiple modes

set -euo pipefail

for dev in /dev/nvme0n1p1 /dev/nvme1n1p1 /dev/nvme2n1p1 /dev/nvme3n1p1; do
    if [[ -e "${dev}" ]]; then
        sudo chown "$(whoami):$(whoami)" "${dev}"
    fi
done

QUERIES=(q1 q3 q5 q6 q13 q16)
EXEC_MODES=()
SF=1
NTHREADS=32
TIMEOUT=10
MAX_RETRIES=3
USER_LOGDIR=""
BIN=./build/tpchdb

while [[ $# -gt 0 ]]; do
    case "$1" in
        -q) IFS=',' read -ra QUERIES <<< "$2"; shift 2 ;;
        -x) IFS=',' read -ra EXEC_MODES <<< "$2"; shift 2 ;;
        -s) SF="$2"; shift 2 ;;
        -w) NTHREADS="$2"; shift 2 ;;
        -t) TIMEOUT="$2"; shift 2 ;;
        -r) MAX_RETRIES="$2"; shift 2 ;;
        -l) USER_LOGDIR="$2"; shift 2 ;;
        *)  echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

DEV_BAM="${DEV_BAM:-/dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3}"
DEV_NVME="${DEV_NVME:-/dev/nvme0n1p1,/dev/nvme1n1p1,/dev/nvme2n1p1,/dev/nvme3n1p1}"

if [[ -n "${USER_LOGDIR}" ]]; then
    LOGDIR="${USER_LOGDIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOGDIR="logs/test_tpch/${TIMESTAMP}"
fi
mkdir -p "${LOGDIR}"

FAILED=()

run_test() {
    local label="$1"
    local logfile="$2"
    shift 2
    local cmd=("$@")

    echo "======== ${label} ========"
    echo "  cmd: ${cmd[*]}"
    echo "  log: ${logfile}"

    local attempt
    for attempt in $(seq 1 "${MAX_RETRIES}"); do
        if timeout "${TIMEOUT}" bash -c '"$@" 2>&1' _ "${cmd[@]}" | tee "${logfile}"; then
            echo "  => OK"
            return 0
        fi
        local rc=${PIPESTATUS[0]}
        if [[ ${rc} -eq 124 ]]; then
            echo "  => TIMEOUT (attempt ${attempt}/${MAX_RETRIES})"
        else
            echo "  => FAILED (rc=${rc})"
            FAILED+=("${label}")
            echo ""
            return 0
        fi
    done
    echo "  => FAILED (timeout x${MAX_RETRIES})"
    FAILED+=("${label}")
    echo ""
}

# Fields: mode  device  use_sudo
MODES=(
    "gidp+bam        ${DEV_BAM}   sudo"
    "gidp+bam+fusion ${DEV_BAM}   sudo"
    "gidp            ${DEV_NVME}  nosudo"
    "datapathfusion  ${DEV_BAM}   sudo"
)

NQUERIES=${#QUERIES[@]}
if [[ ${#EXEC_MODES[@]} -gt 0 ]]; then
    NMODES=${#EXEC_MODES[@]}
else
    NMODES=${#MODES[@]}
fi
TOTAL=$((NQUERIES * NMODES))
idx=0

echo "=== test_tpch: SF${SF} queries=${QUERIES[*]} (${TOTAL} tests) ==="
echo "  output: ${LOGDIR}/sf${SF}/"
echo ""

for QUERY in "${QUERIES[@]}"; do
    for entry in "${MODES[@]}"; do
        mode=$(echo "${entry}" | awk '{print $1}')
        dev=$(echo "${entry}" | awk '{print $2}')
        use_sudo=$(echo "${entry}" | awk '{print $3}')

        # Filter by -x if specified
        if [[ ${#EXEC_MODES[@]} -gt 0 ]]; then
            match=false
            for em in "${EXEC_MODES[@]}"; do
                [[ "${em}" == "${mode}" ]] && match=true
            done
            ${match} || continue
        fi

        idx=$((idx + 1))

        OUTDIR="${LOGDIR}/sf${SF}/${mode}"
        mkdir -p "${OUTDIR}"

        cmd=("${BIN}" -q "${QUERY}" -x "${mode}")
        [[ "${mode}" == "gidp" ]] && cmd+=(-w "${NTHREADS}")
        cmd+=(-Z "${dev}")

        if [[ "${use_sudo}" == "sudo" ]]; then
            run_test \
                "[${idx}/${TOTAL}] ${QUERY} ${mode} (${dev})" \
                "${OUTDIR}/${QUERY}.txt" \
                sudo "${cmd[@]}"
        else
            run_test \
                "[${idx}/${TOTAL}] ${QUERY} ${mode} (${dev})" \
                "${OUTDIR}/${QUERY}.txt" \
                "${cmd[@]}"
        fi
    done
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "=== 失敗: ${#FAILED[@]}/${TOTAL} (${LOGDIR}/sf${SF}/) ==="
    for f in "${FAILED[@]}"; do
        echo "  - ${f}"
    done
    exit 1
else
    echo "=== 完了: 全 ${TOTAL} テスト成功 (${LOGDIR}/sf${SF}/) ==="
fi
