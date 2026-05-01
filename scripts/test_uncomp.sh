#!/usr/bin/env bash
# ============================================================
# test_uncomp.sh — Verify all execution modes with uncompressed
# data at multiple page sizes.
#
# Usage:
#   scripts/test_uncomp.sh [OPTIONS]
#   scripts/test_uncomp.sh -s 1 -c 64K,1M
#   scripts/test_uncomp.sh -s 1 -c 64K,1M -x gidp+bam
#   scripts/test_uncomp.sh --dry-run
#
# Options:
#   -s SF          Scale factor (default: 1)
#   -c SIZES       Chunk sizes, comma-separated (default: 64K,1M)
#   -x MODES       Execution modes (default: gidp,gidp+bam,gidp+bam+fusion,datapathfusion)
#   -q QUERIES     Queries to run (default: q1,q3,q5,q6,q13,q16)
#   -w THREADS     Number of threads for gidp mode (default: 32)
#   -t TIMEOUT     Per-query timeout in seconds (default: 10)
#   -r RETRIES     Number of retries on timeout/failure (default: 2)
#   --no-load      Skip data loading (assume uncomp data on devices)
#   --log-dir DIR  Resume into existing log directory (skip already-passed queries)
#   --dry-run      Print commands without executing
#   -h, --help     Show this help
# ============================================================
set -euo pipefail

# ---- Configuration ----
SF=1
CHUNK_SIZES=(64K 1M)
MODES=(gidp gidp+bam gidp+bam+fusion datapathfusion)
QUERIES=(q1 q3 q5 q6 q13 q16)
NTHREADS=32
TIMEOUT=10
RETRIES=10

DEVICES_NVME="/dev/nvme0n1p1,/dev/nvme1n1p1,/dev/nvme2n1p1,/dev/nvme3n1p1"
DEVICES_BAM="/dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3"
DATA_BASE="/export/data1/tpch"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BAM_MODULE_DIR="${PROJECT_DIR}/bam/build/module"

DRY_RUN=false
SKIP_LOAD=false
RESUME_LOG_DIR=""

# ---- CLI parsing ----
usage() {
    sed -n '2,/^# ====/{ /^# /s/^# //p }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s) SF="$2"; shift 2 ;;
        -c) IFS=',' read -ra CHUNK_SIZES <<< "$2"; shift 2 ;;
        -x) IFS=',' read -ra MODES <<< "$2"; shift 2 ;;
        -q) IFS=',' read -ra QUERIES <<< "$2"; shift 2 ;;
        -w) NTHREADS="$2"; shift 2 ;;
        -t) TIMEOUT="$2"; shift 2 ;;
        -r) RETRIES="$2"; shift 2 ;;
        --no-load)   SKIP_LOAD=true; shift ;;
        --log-dir)   RESUME_LOG_DIR="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)   usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---- Helpers ----
if [[ -n "${RESUME_LOG_DIR}" ]]; then
    LOG_DIR="${RESUME_LOG_DIR}"
    mkdir -p "${LOG_DIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="${PROJECT_DIR}/logs/test_uncomp/${TIMESTAMP}"
    mkdir -p "${LOG_DIR}"
fi

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/main.log"
}

PASS=0
FAIL=0
ERRORS=()

run_cmd() {
    if $DRY_RUN; then
        log "[DRY-RUN] $*"
        return 0
    fi
    "$@"
}

# Check if a query log exists and completed successfully
already_passed() {
    local logfile="$1"
    [[ -s "${logfile}" ]] || return 1
    grep -qE '^Elapsed time \(ns\):' "${logfile}" || return 1
    ! grep -qiE 'CUDA error|nvCOMP.*failed|NVM Error|illegal memory' "${logfile}"
}

# ---- Mode helpers ----
# Maps execution mode → loader mode
loader_mode_for() {
    case "$1" in
        gidp)                echo "gidp" ;;
        gidp+bam)            echo "gidp" ;;
        gidp+bam+fusion)     echo "gidp+bam+fusion" ;;
        datapathfusion)      echo "datapathfusion" ;;
        *) echo "$1" ;;
    esac
}

# Whether mode needs BaM module
needs_bam() {
    case "$1" in
        gidp) return 1 ;;
        *)    return 0 ;;
    esac
}

# Device path for mode
devices_for() {
    case "$1" in
        gidp) echo "${DEVICES_NVME}" ;;
        *)    echo "${DEVICES_BAM}" ;;
    esac
}

# ---- BaM module management ----
BAM_LOADED=false

load_bam() {
    if $BAM_LOADED; then return 0; fi
    log ">>> Loading BaM module <<<"
    if $DRY_RUN; then
        BAM_LOADED=true; return 0
    fi
    sudo umount /export/data1 2>/dev/null || true
    sudo mdadm --stop /dev/md0 2>/dev/null || true
    for pci in c0 c1 c2 c3; do
        echo -n "0000:${pci}:00.0" \
            | sudo tee "/sys/bus/pci/devices/0000:${pci}:00.0/driver/unbind" \
            2>/dev/null || true
    done
    pushd "${BAM_MODULE_DIR}" > /dev/null
    sudo make load
    popd > /dev/null
    BAM_LOADED=true
    log "BaM module loaded"
}

unload_bam() {
    if ! $BAM_LOADED; then return 0; fi
    log ">>> Unloading BaM module <<<"
    if $DRY_RUN; then
        BAM_LOADED=false; return 0
    fi
    pushd "${BAM_MODULE_DIR}" > /dev/null
    sudo make unload
    popd > /dev/null
    for pci in c0 c1 c2 c3; do
        echo "0000:${pci}:00.0" \
            | sudo tee /sys/bus/pci/drivers/nvme/bind \
            2>/dev/null || true
    done
    sleep 3
    sudo mdadm --stop /dev/md0 2>/dev/null || true
    sudo mdadm --assemble /dev/md0 /dev/nvme{0,1,2,3}n1p2
    sudo mount -t xfs /dev/md0 /export/data1/
    sudo chown "$(whoami):$(whoami)" ${DEVICES_NVME//,/ }
    BAM_LOADED=false
    log "Filesystem remounted"
}

cleanup() {
    if $BAM_LOADED; then
        log "Cleanup: unloading BaM..."
        unload_bam || true
    fi
    log "=== Done: ${PASS} passed, ${FAIL} failed ==="
    if [[ ${#ERRORS[@]} -gt 0 ]]; then
        log "Failed:"
        for e in "${ERRORS[@]}"; do log "  $e"; done
    fi
}
trap cleanup EXIT

# ---- Pre-flight ----
for bin in tpchdb tpchloader; do
    if [[ ! -x "${BUILD_DIR}/${bin}" ]]; then
        echo "ERROR: ${BUILD_DIR}/${bin} not found." >&2
        exit 1
    fi
done

# ---- Group modes by loader mode ----
# Collect unique loader modes in order, and map each to its exec modes
declare -a LOADER_MODES_ORDERED=()
declare -A LOADER_TO_EXEC_MODES=()

for MODE in "${MODES[@]}"; do
    LMODE=$(loader_mode_for "$MODE")
    if [[ -z "${LOADER_TO_EXEC_MODES[$LMODE]+x}" ]]; then
        LOADER_MODES_ORDERED+=("$LMODE")
        LOADER_TO_EXEC_MODES[$LMODE]="$MODE"
    else
        LOADER_TO_EXEC_MODES[$LMODE]+=" $MODE"
    fi
done

# ---- Print plan ----
log "============================================================"
log "  Uncompressed Data Verification"
log "============================================================"
log "SF           : ${SF}"
log "Chunk sizes  : ${CHUNK_SIZES[*]}"
log "Modes        : ${MODES[*]}"
log "Queries      : ${QUERIES[*]}"
log "Loader groups:"
for LMODE in "${LOADER_MODES_ORDERED[@]}"; do
    log "  load -x ${LMODE} → exec: ${LOADER_TO_EXEC_MODES[$LMODE]}"
done
log "Timeout      : ${TIMEOUT}s"
log "Retries      : ${RETRIES}"
log "Skip load    : ${SKIP_LOAD}"
log "Dry run      : ${DRY_RUN}"
log "Log dir      : ${LOG_DIR}"
log "============================================================"
log ""

# ---- Main loop ----
for CS in "${CHUNK_SIZES[@]}"; do
    log "============================================"
    log "=== Page size: ${CS}  (uncompressed) ==="
    log "============================================"

    for LMODE in "${LOADER_MODES_ORDERED[@]}"; do
        IFS=' ' read -ra EXEC_MODES <<< "${LOADER_TO_EXEC_MODES[$LMODE]}"

        # --- Load data ---
        if ! $SKIP_LOAD; then
            unload_bam
            INPUT_DIR="${DATA_BASE}/sideways/sf${SF}"
            LOAD_LOG="${LOG_DIR}/load_${CS}_${LMODE}.log"
            log "--- Load: SF${SF} page=${CS} loader=${LMODE} (uncomp, no -c) ---"
            run_cmd sync
            run_cmd bash -c 'echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null'

            if $DRY_RUN; then
                log "[DRY-RUN] ${BUILD_DIR}/tpchloader -i ${INPUT_DIR} -d ${DEVICES_NVME} -x ${LMODE} -p ${CS} -A"
            else
                "${BUILD_DIR}/tpchloader" \
                    -i "${INPUT_DIR}" \
                    -d "${DEVICES_NVME}" \
                    -x "${LMODE}" \
                    -p "${CS}" \
                    -A \
                    2>&1 | tee "${LOAD_LOG}"
            fi
            log "Load complete"
        fi

        # --- Run each exec mode that shares this loader ---
        for MODE in "${EXEC_MODES[@]}"; do
            DEVICES=$(devices_for "$MODE")

            # Set BaM state
            if needs_bam "$MODE"; then
                load_bam
            else
                unload_bam
            fi

            for Q in "${QUERIES[@]}"; do
                QLOG="${LOG_DIR}/${CS}_${MODE}_${Q}.log"

                # Skip if already passed in a previous run
                if [[ -n "${RESUME_LOG_DIR}" ]] && already_passed "${QLOG}"; then
                    log "  SKIP ${Q} (${CS}/${MODE}) — already passed"
                    PASS=$((PASS + 1))
                    continue
                fi

                # Build command
                CMD=(sudo timeout "${TIMEOUT}" "${BUILD_DIR}/tpchdb"
                    -q "${Q}" -x "${MODE}" -w "${NTHREADS}" -Z
                    "${DEVICES}")

                if $DRY_RUN; then
                    log "  RUN ${Q} (page=${CS}, ${MODE}, uncomp)"
                    log "  [DRY-RUN] ${CMD[*]}"
                    PASS=$((PASS + 1))
                    continue
                fi

                q_passed=false
                for attempt in $(seq 0 "${RETRIES}"); do
                    if [[ $attempt -eq 0 ]]; then
                        log "  RUN ${Q} (page=${CS}, ${MODE}, uncomp)"
                    else
                        log "  RETRY ${Q} (${CS}/${MODE}) attempt $((attempt+1))/$((RETRIES+1))"
                    fi

                    rc=0
                    "${CMD[@]}" 2>&1 | tee "${QLOG}" || rc=$?

                    if [[ $rc -eq 124 ]]; then
                        log "  TIMEOUT ${Q} (${CS}/${MODE}) after ${TIMEOUT}s"
                        continue
                    elif [[ $rc -ne 0 ]]; then
                        log "  EXIT ${Q} (${CS}/${MODE}) rc=${rc}"
                        continue
                    fi

                    if grep -qiE 'CUDA error|nvCOMP.*failed|NVM Error|illegal memory' "${QLOG}"; then
                        log "  ERROR ${Q} (${CS}/${MODE}) — error in output"
                        continue
                    fi

                    q_passed=true
                    break
                done

                if $q_passed; then
                    log "  PASS ${Q} (${CS}/${MODE})"
                    PASS=$((PASS + 1))
                else
                    log "  FAIL ${Q} (${CS}/${MODE}) — all $((RETRIES+1)) attempts failed"
                    FAIL=$((FAIL + 1))
                    ERRORS+=("${CS}/${MODE}/${Q}")
                fi
            done
        done
    done

    log "=== Page size ${CS} complete ==="
    log ""
done

log "============================================"
log "  Summary: ${PASS} passed, ${FAIL} failed"
log "  Logs: ${LOG_DIR}"
log "============================================"

[[ ${FAIL} -eq 0 ]]
