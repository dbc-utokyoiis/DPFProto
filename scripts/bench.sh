#!/bin/bash
# bench.sh — Run a tpchdb command multiple times and report timing statistics.
#
# Usage:
#   scripts/bench.sh [-n TRIALS] [-w] [-t TIMEOUT] [-o LOGFILE] -- COMMAND...
#
# Options:
#   -n TRIALS   Number of measured runs (default: 10)
#   -w          Include 1 warmup run before measured runs
#   -t TIMEOUT  Per-trial timeout in seconds (default: 20). If a trial
#               exceeds this, it is killed and retried.
#   -o LOGFILE  Write all output to LOGFILE (in addition to stdout).
#               Parent directories are created automatically.
#
# Examples:
#   scripts/bench.sh -n 10 -w -- sudo ./build/tpchdb -q q6 -x golap -w 16 -S /dev/nvme1n1p1
#   scripts/bench.sh -n 5 -w -t 30 -- sudo ./build/tpchdb -q q16 -x pig /dev/libnvm2 -Z
#   scripts/bench.sh -n 10 -w -o logs/bench/q6_golap.txt -- sudo ./build/tpchdb -q q6 -x golap -w 16 -S /dev/nvme1n1p1

set -euo pipefail

TRIALS=10
WARMUP=0
TIMEOUT=20
LOGFILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) TRIALS="$2"; shift 2 ;;
        -w) WARMUP=1; shift ;;
        -t) TIMEOUT="$2"; shift 2 ;;
        -o) LOGFILE="$2"; shift 2 ;;
        --) shift; break ;;
        *)  echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [-n TRIALS] [-w] [-t TIMEOUT] [-o LOGFILE] -- COMMAND..." >&2
    exit 1
fi

CMD=("$@")

# ---- log file setup ----
if [[ -n "$LOGFILE" ]]; then
    mkdir -p "$(dirname "$LOGFILE")"
    exec > >(tee "$LOGFILE") 2>&1
fi

# ---- warmup ----
if [[ $WARMUP -eq 1 ]]; then
    echo "======== Warmup ========"
    if [[ "${CMD[0]}" == "sudo" ]]; then
        sudo timeout --signal=KILL "${TIMEOUT}s" "${CMD[@]:1}" || true
    else
        timeout --signal=KILL "${TIMEOUT}s" "${CMD[@]}" || true
    fi
    echo ""
fi

# ---- measured runs ----
declare -a TIMES
declare -a THROUGHPUTS
GPU_MEM_MB=""
for i in $(seq 1 "$TRIALS"); do
    echo "======== Trial $i / $TRIALS ========"

    while true; do
        # Run with timeout. If the command starts with "sudo", move timeout
        # inside sudo so that the child process (running as root) is killed
        # directly, avoiding orphaned processes on timeout.
        if [[ "${CMD[0]}" == "sudo" ]]; then
            run_cmd=(sudo timeout --signal=KILL "${TIMEOUT}s" "${CMD[@]:1}")
        else
            run_cmd=(timeout --signal=KILL "${TIMEOUT}s" "${CMD[@]}")
        fi
        output=$(${run_cmd[@]} 2>&1 | tr -d '\0') && rc=0 || rc=$?
        if [[ $rc -eq 137 ]]; then
            # 137 = killed by SIGKILL (timeout)
            echo "[bench.sh] Trial $i timed out after ${TIMEOUT}s — retrying..." >&2
            sleep 1
            continue
        fi
        break
    done

    echo "$output"

    if [[ $rc -ne 0 ]]; then
        echo "[bench.sh] ERROR: command failed with exit code $rc on trial $i" >&2
        exit $rc
    fi

    # Parse "time: 123.456 msec"
    t=$(echo "$output" | grep -m1 '^time: ' | sed 's/^time: \([0-9.]*\) msec$/\1/')
    if [[ -z "$t" ]]; then
        echo "[bench.sh] WARNING: could not parse time from trial $i" >&2
        continue
    fi
    TIMES+=("$t")

    # Parse "effective_throughput_gbs: 1.23"
    tp=$(echo "$output" | grep -m1 '^effective_throughput_gbs: ' | sed 's/^effective_throughput_gbs: //')
    if [[ -n "$tp" ]]; then
        THROUGHPUTS+=("$tp")
    fi

    # Parse "gpu_mem_mb: 12345" (constant across trials, take last seen)
    gm=$(echo "$output" | grep -m1 '^gpu_mem_mb: ' | sed 's/^gpu_mem_mb: //')
    if [[ -n "$gm" ]]; then
        GPU_MEM_MB="$gm"
    fi

    echo ""
done

N=${#TIMES[@]}
if [[ $N -eq 0 ]]; then
    echo "[bench.sh] ERROR: no valid timing data collected" >&2
    exit 1
fi

# ---- statistics (awk) ----
# Build input: "time throughput" per line (throughput may be empty)
paste <(printf '%s\n' "${TIMES[@]}") \
      <(printf '%s\n' "${THROUGHPUTS[@]+"${THROUGHPUTS[@]}"}") | awk -v gpu_mem="$GPU_MEM_MB" '
BEGIN {
    t_sum=0; t_min=1e30; t_max=-1e30; tn=0
    tp_sum=0; tp_min=1e30; tp_max=-1e30; tpn=0
}
{
    v = $1 + 0
    t_vals[tn] = v
    t_sum += v
    if (v < t_min) t_min = v
    if (v > t_max) t_max = v
    tn++

    if ($2 != "") {
        tp = $2 + 0
        tp_vals[tpn] = tp
        tp_sum += tp
        if (tp < tp_min) tp_min = tp
        if (tp > tp_max) tp_max = tp
        tpn++
    }
}
END {
    t_avg = t_sum / tn
    if (tn > 1) {
        t_sumsq = 0
        for (i = 0; i < tn; i++) t_sumsq += (t_vals[i] - t_avg)^2
        t_sd = sqrt(t_sumsq / (tn - 1))
    } else { t_sd = 0 }

    printf "======== Summary (%d trials) ========\n", tn
    for (i = 0; i < tn; i++) {
        if (i < tpn)
            printf "  trial %d: %10.3f msec  %6.2f GB/s\n", i+1, t_vals[i], tp_vals[i]
        else
            printf "  trial %d: %10.3f msec\n", i+1, t_vals[i]
    }
    printf "time:       avg=%.3f  min=%.3f  max=%.3f  stddev=%.3f msec\n", t_avg, t_min, t_max, t_sd
    if (tpn > 0) {
        tp_avg = tp_sum / tpn
        printf "throughput: avg=%.2f  min=%.2f  max=%.2f GB/s\n", tp_avg, tp_min, tp_max
    }
    if (gpu_mem != "")
        printf "gpu_mem_mb: %s\n", gpu_mem
}
'
