#!/usr/bin/env bash
# ============================================================
# ssb_check_q3q4.sh — Verify SSB Q3x/Q4x results in gidp+bam mode
# against reference answers in answers/ssb/sf${SF}/.
#
# Usage: sudo ./scripts/ssb_check_q3q4.sh [SF]
#   SF defaults to 100.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BIN="${BUILD_DIR}/ssbdb"
ANSWERS_DIR="${PROJECT_DIR}/answers/ssb"
DEVICES_BAM="/dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3"

SF="${1:-100}"
ANS="${ANSWERS_DIR}/sf${SF}"
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${PROJECT_DIR}/logs/ssb_check_q3q4/${TS}"
mkdir -p "${OUT_DIR}"

QUERIES=(q21 q22 q23 q31 q32 q33 q34 q41 q42 q43)

summary="${OUT_DIR}/summary.txt"
: > "${summary}"

for q in "${QUERIES[@]}"; do
    echo "=== ${q} ==="
    raw="${OUT_DIR}/${q}.raw.txt"
    got="${OUT_DIR}/${q}.got.txt"
    exp="${ANS}/${q}.csv"

    "${BIN}" -q "${q}" -x gidp+bam -w 32 -Z "${DEVICES_BAM}" >"${raw}" 2>&1 || {
        echo "  RUN FAILED (exit=$?), see ${raw}" | tee -a "${summary}"
        continue
    }

    # ssbdb prints result rows separated by ' | '. The aggregated number may
    # appear in the first column (Q2x) or the last (Q3x, Q4x).
    # Match any row that contains at least one ' | ' and extract.
    grep -E '^[[:space:]]*[^|]+(\|[^|]+)+[[:space:]]*$' "${raw}" \
        | grep -E '([0-9]+)[[:space:]]*$|^[[:space:]]*[0-9]+' >"${got}" || true

    if [[ ! -f "${exp}" ]]; then
        echo "  NO REFERENCE at ${exp}" | tee -a "${summary}"
        continue
    fi

    got_n=$(wc -l <"${got}")
    ref_n=$(($(wc -l <"${exp}") - 1))   # minus header

    # Aggregate column: Q2x → first column (revenue),
    #                   Q3x/Q4x → last column (revenue/profit).
    case "${q}" in
        q2*) agg_col="first" ;;
        q3*|q4*) agg_col="last" ;;
        *) agg_col="last" ;;
    esac
    if [[ "${agg_col}" == "first" ]]; then
        got_sum=$(awk -F'|' '{gsub(/ /,"",$1); s+=$1} END{printf "%.0f", s}' "${got}")
        ref_sum=$(awk -F'|' 'NR>1 {gsub(/"/,"",$1); s+=$1} END{printf "%.0f", s}' "${exp}")
    else
        got_sum=$(awk -F'|' '{gsub(/ /,"",$NF); s+=$NF} END{printf "%.0f", s}' "${got}")
        ref_sum=$(awk -F'|' 'NR>1 {gsub(/"/,"",$NF); s+=$NF} END{printf "%.0f", s}' "${exp}")
    fi

    if [[ "${got_n}" == "${ref_n}" && "${got_sum}" == "${ref_sum}" ]]; then
        echo "  OK  rows=${got_n}  sum=${got_sum}" | tee -a "${summary}"
    else
        echo "  MISMATCH  rows=${got_n}/${ref_n}  sum=${got_sum} vs ${ref_sum}" | tee -a "${summary}"
    fi
done

echo ""
echo "=== Summary ==="
cat "${summary}"
echo ""
echo "Full output: ${OUT_DIR}"
