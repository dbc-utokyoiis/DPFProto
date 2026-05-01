#!/usr/bin/env bash
# ============================================================
# ssb_verify_bench.sh — Verify SSB bench.sh logs against answer files.
#
# Extracts results from the LAST trial in each bench.sh log file
# and compares against answers/ssb/sf${SF}/*.csv.
#
# Usage:
#   scripts/ssb_verify_bench.sh <log-dir> <sf>
#   scripts/ssb_verify_bench.sh logs/ssb_run_all/20260329_160121/sf100 100
#
# The log-dir should contain subdirectories per mode (gidp/, gidp+bam/,
# gidp+bam+fusion/) with per-query files (q11.txt, q12.txt, ...).
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <log-dir> <sf>" >&2
    exit 1
fi

LOG_DIR="$1"
SF="$2"
ANSWERS_DIR="${PROJECT_DIR}/answers/ssb/sf${SF}"
QUERIES=(q11 q12 q13 q21 q22 q23 q31 q32 q33 q34 q41 q42 q43)

if [[ ! -d "${ANSWERS_DIR}" ]]; then
    echo "ERROR: Answer directory not found: ${ANSWERS_DIR}" >&2
    exit 1
fi

total=0 pass=0 fail=0 skip=0

# Extract the result section from the last trial in a bench.sh log.
# For Q1x: extract the revenue line.
# For Q2x/Q3x/Q4x: extract lines between "Q?.x results:" and the next "========".
extract_last_trial_result() {
    local log_file="$1"
    local query="$2"

    # Find the last "Trial N / M" block; fall back to warmup if only 1 run
    local last_trial_line
    last_trial_line=$(grep -n '======== Trial ' "${log_file}" | tail -1 | cut -d: -f1)
    if [[ -z "${last_trial_line}" ]]; then
        last_trial_line=1
    fi

    case "${query}" in
        q1[123])
            # Q1x: single revenue value on "SSB Q1.x revenue: <number>"
            sed -n "${last_trial_line},\$p" "${log_file}" \
                | grep -oP 'SSB Q1\.x revenue: \K\d+' | tail -1
            ;;
        q2[123])
            # Q2x: result block after "SSB Q2.x results:"
            sed -n "${last_trial_line},\$p" "${log_file}" \
                | sed -n '/SSB Q2\.x results:/,/========/{/^\s\s/p}' \
                | sed 's/^\s*//;s/ | /|/g'
            ;;
        q3[1234])
            # Q3x: result block after "SSB Q3.x results:"
            sed -n "${last_trial_line},\$p" "${log_file}" \
                | sed -n '/SSB Q3\.x results:/,/========/{/^\s\s/p}' \
                | sed 's/^\s*//;s/ | /|/g'
            ;;
        q4[123])
            # Q4x: result block after "SSB Q4.x results:"
            sed -n "${last_trial_line},\$p" "${log_file}" \
                | sed -n '/SSB Q4\.x results:/,/========/{/^\s\s/p}' \
                | sed 's/^\s*//;s/ | /|/g'
            ;;
    esac
}

for mode in gidp gidp+bam gidp+bam+fusion datapathfusion uncomp_gidp uncomp_gidp+bam uncomp_gidp+bam+fusion uncomp_datapathfusion; do
    mode_dir="${LOG_DIR}/${mode}"
    if [[ ! -d "${mode_dir}" ]]; then continue; fi

    for q in "${QUERIES[@]}"; do
        log_file="${mode_dir}/${q}.txt"
        ans_file="${ANSWERS_DIR}/${q}.csv"
        total=$((total + 1))

        if [[ ! -f "${log_file}" ]]; then
            printf "  %-18s %-5s: SKIP (no log)\n" "${mode}" "${q}"
            skip=$((skip + 1))
            continue
        fi
        if [[ ! -f "${ans_file}" ]]; then
            printf "  %-18s %-5s: SKIP (no answer)\n" "${mode}" "${q}"
            skip=$((skip + 1))
            continue
        fi

        result="UNKNOWN"
        case "${q}" in
            q1[123])
                ans_val=$(tail -1 "${ans_file}")
                got_val=$(extract_last_trial_result "${log_file}" "${q}")
                if [[ "${ans_val}" == "${got_val}" ]]; then
                    result="OK"
                else
                    result="FAIL (ans=${ans_val} got=${got_val})"
                fi
                ;;
            q2[123]|q3[1234]|q4[123])
                got_csv=$(extract_last_trial_result "${log_file}" "${q}" | sort)
                ans_csv=$(tail -n +2 "${ans_file}" | sed 's/"//g' | sort)
                if [[ "${got_csv}" == "${ans_csv}" ]]; then
                    result="OK"
                else
                    got_n=$(echo "${got_csv}" | wc -l)
                    ans_n=$(echo "${ans_csv}" | wc -l)
                    result="FAIL (rows: ans=${ans_n} got=${got_n})"
                    if [[ "${got_n}" == "${ans_n}" ]]; then
                        # Same row count but content differs — show first diff
                        first_diff=$(diff <(echo "${ans_csv}") <(echo "${got_csv}") | head -5 || true)
                        result="FAIL (rows=${ans_n}, content mismatch)"
                    fi
                fi
                ;;
        esac

        if [[ "${result}" == "OK" ]]; then
            printf "  %-18s %-5s: OK\n" "${mode}" "${q}"
            pass=$((pass + 1))
        else
            printf "  %-18s %-5s: %s\n" "${mode}" "${q}" "${result}"
            fail=$((fail + 1))
        fi
    done
done

echo ""
echo "=== Summary: ${pass}/${total} passed, ${fail} failed, ${skip} skipped ==="

if [[ $fail -gt 0 ]]; then
    exit 1
fi
