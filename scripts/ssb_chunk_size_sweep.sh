#!/usr/bin/env bash
# ============================================================
# ssb_chunk_size_sweep.sh — Benchmark SSB across different
# page (chunk) sizes: 64K, 128K, 256K, 512K, 1M.
#
# For each chunk size, re-loads data with the corresponding
# page size and benchmarks selected execution modes and queries.
#
# Usage:
#   scripts/ssb_chunk_size_sweep.sh [OPTIONS]
#   scripts/ssb_chunk_size_sweep.sh --dry-run
#   scripts/ssb_chunk_size_sweep.sh -s 100 -c 64K,256K,1M -x datapathfusion -q q11,q21
#
# Options:
#   -s SF              Scale factor (default: 100)
#   -c SIZES           Chunk sizes, comma-separated (default: 64K,128K,256K,512K,1M)
#   -x MODES           Execution modes to bench (default: gidp+bam,gidp+bam+fusion,datapathfusion)
#   -q QUERIES         Queries to run (default: q11,q21,q31)
#   --allquery         Run all queries (q11..q43)
#   -n TRIALS          Number of bench trials (default: 10)
#   -t TIMEOUT         Per-trial timeout in seconds (default: 60)
#   -w THREADS         Worker threads (default: 32)
#   --log-dir DIR      Use existing log directory (append results)
#   --no-load          Skip data loading (assume data already on devices)
#   --no-zonemap       Disable zone map pruning
#   -f, --verify       Enable loader verification
#   --dry-run          Print commands without executing
#   -h, --help         Show this help
# ============================================================
set -euo pipefail

# ---- Configuration ----
SF=100
CHUNK_SIZES=(2M 64K 128K 256K 512K 1M)
MODES=(gidp gidp+bam gidp+bam+fusion datapathfusion)
QUERIES=(q11 q21 q31)
ALL_QUERIES=(q11 q12 q13 q21 q22 q23 q31 q32 q33 q34 q41 q42 q43)
NTRIALS=10
BENCH_TIMEOUT=10
NTHREADS=32

DEVICES_NVME="/dev/nvme0n1p1,/dev/nvme1n1p1,/dev/nvme2n1p1,/dev/nvme3n1p1"
DEVICES_BAM="/dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3"
DATA_BASE="/export/data1/ssb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BAM_MODULE_DIR="${PROJECT_DIR}/bam/build/module"
BENCH_SCRIPT="${SCRIPT_DIR}/bench.sh"

DRY_RUN=false
SKIP_LOAD=false
VERIFY=false
USER_LOG_DIR=""
ZONE_MAP="-Z"

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
        -n) NTRIALS="$2"; shift 2 ;;
        -t) BENCH_TIMEOUT="$2"; shift 2 ;;
        -w) NTHREADS="$2"; shift 2 ;;
        --log-dir)        USER_LOG_DIR="$2"; shift 2 ;;
        --no-load)        SKIP_LOAD=true; shift ;;
        --allquery)       QUERIES=("${ALL_QUERIES[@]}"); shift ;;
        --no-zonemap)     ZONE_MAP=""; shift ;;
        -f|--verify)      VERIFY=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---- Helpers ----
if [[ -n "${USER_LOG_DIR}" ]]; then
    LOG_DIR="${USER_LOG_DIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="${PROJECT_DIR}/logs/ssb_chunk_sweep/${TIMESTAMP}"
fi
mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/main.log"
}

time_cmd() {
    local label="$1"; shift
    local start end elapsed
    start=$(date +%s)
    "$@"
    local rc=$?
    end=$(date +%s)
    elapsed=$((end - start))
    log "TIMING ${label}: ${elapsed}s ($(( elapsed / 60 ))m $(( elapsed % 60 ))s)"
    echo "${label} ${elapsed}" >> "${LOG_DIR}/timings.txt"
    return $rc
}

# ---- BaM module management ----
BAM_LOADED=false

load_bam() {
    if $BAM_LOADED; then return 0; fi
    log ">>> Loading BaM module <<<"
    if $DRY_RUN; then
        log "[DRY-RUN] load_bam"
        BAM_LOADED=true
        return 0
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
        log "[DRY-RUN] unload_bam"
        BAM_LOADED=false
        return 0
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
    log "Filesystem remounted at /export/data1"
}

cleanup() {
    if $BAM_LOADED; then
        log "Cleanup: unloading BaM module..."
        unload_bam || true
    fi
    log "=== Script finished ==="
}
trap cleanup EXIT

# ---- Loader mode mapping ----
loader_mode_for() {
    case "$1" in
        gidp)                echo "gidp" ;;
        gidp+bam)            echo "gidp" ;;
        gidp+bam+fusion)     echo "gidp+bam+fusion" ;;
        datapathfusion)      echo "datapathfusion" ;;
        *) echo "$1" ;;
    esac
}

# ---- Load data ----
run_load() {
    local sf="$1" loader_mode="$2" chunk_size="$3"
    local input_dir="${DATA_BASE}/sideways/sf${sf}"
    local out_dir="${LOG_DIR}/cs_${chunk_size}/sf${sf}/${loader_mode}"
    local log_file="${out_dir}/load.log"
    mkdir -p "${out_dir}"

    if $SKIP_LOAD; then
        log "LOAD SF${sf} mode=${loader_mode} chunk=${chunk_size} -- skipped (--no-load)"
        return 0
    fi

    log "LOAD SF${sf} mode=${loader_mode} chunk=${chunk_size}"
    local cmd=("${BUILD_DIR}/ssbloader"
        -i "${input_dir}"
        -d "${DEVICES_NVME}"
        -x "${loader_mode}"
        -p "${chunk_size}"
        -A)
    $VERIFY && cmd+=(-f)

    if $DRY_RUN; then
        log "[DRY-RUN] ${cmd[*]}"
        return 0
    fi
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
    time_cmd "sf${sf}_load_${loader_mode}_cs${chunk_size}" "${cmd[@]}" 2>&1 | tee "${log_file}"
}

# ---- Run benchmark ----
run_bench() {
    local sf="$1" mode="$2" query="$3" devices="$4" chunk_size="$5"
    local out_dir="${LOG_DIR}/cs_${chunk_size}/sf${sf}/${mode}"
    local out_file="${out_dir}/${query}.txt"
    mkdir -p "${out_dir}"

    local cmd=(sudo "${BUILD_DIR}/ssbdb"
        -q "${query}" -x "${mode}" -w "${NTHREADS}")
    [[ -n "${ZONE_MAP}" ]] && cmd+=("${ZONE_MAP}")
    cmd+=("${devices}")

    log "  BENCH SF${sf} ${mode} chunk=${chunk_size} ${query}"
    if $DRY_RUN; then
        log "  [DRY-RUN] ${BENCH_SCRIPT} -n ${NTRIALS} -w -t ${BENCH_TIMEOUT} -o ${out_file} -- ${cmd[*]}"
        return 0
    fi
    "${BENCH_SCRIPT}" -n "${NTRIALS}" -w -t "${BENCH_TIMEOUT}" \
        -o "${out_file}" -- "${cmd[@]}" || {
        log "  ERROR: bench failed for SF${sf} ${mode} chunk=${chunk_size} ${query} (rc=$?)"
        return 1
    }
}

# ---- Determine devices for a mode ----
devices_for() {
    case "$1" in
        gidp)    echo "${DEVICES_NVME}" ;;
        *)       echo "${DEVICES_BAM}" ;;
    esac
}

# Whether mode needs BaM
needs_bam() {
    case "$1" in
        gidp) return 1 ;;
        *)    return 0 ;;
    esac
}

# ---- Pre-flight ----
for bin in ssbdb ssbloader; do
    if [[ ! -x "${BUILD_DIR}/${bin}" ]]; then
        echo "ERROR: ${BUILD_DIR}/${bin} not found." >&2
        exit 1
    fi
done

if ! $DRY_RUN; then
    sudo ls -lha /dev/nvme0n1p1 > /dev/null
    sudo chown "$(whoami):$(whoami)" ${DEVICES_NVME//,/ }
fi

# ---- Print plan ----
log "============================================================"
log "  SSB Chunk Size Sweep"
log "============================================================"
log "Scale factor : ${SF}"
log "Chunk sizes  : ${CHUNK_SIZES[*]}"
log "Modes        : ${MODES[*]}"
log "Queries      : ${QUERIES[*]}"
log "Trials       : ${NTRIALS}"
log "Timeout      : ${BENCH_TIMEOUT}s"
log "Threads      : ${NTHREADS}"
log "Zone map     : ${ZONE_MAP:-disabled}"
log "Dry run      : ${DRY_RUN}"
log "Log directory: ${LOG_DIR}"
log "============================================================"
log ""

# ---- Main loop ----
for CS in "${CHUNK_SIZES[@]}"; do
    log "============================================================"
    log "=== Chunk size: ${CS} ==="
    log "============================================================"

    # Group modes by loader mode to minimize reloads
    declare -A loaded_for_loader=()

    for MODE in "${MODES[@]}"; do
        LMODE=$(loader_mode_for "${MODE}")
        DEVICES=$(devices_for "${MODE}")

        # Load data if not yet loaded for this loader mode + chunk size
        if [[ -z "${loaded_for_loader[${LMODE}]+x}" ]]; then
            unload_bam
            run_load "${SF}" "${LMODE}" "${CS}"
            loaded_for_loader[${LMODE}]=1
        fi

        # Ensure BaM state
        if needs_bam "${MODE}"; then
            load_bam
        else
            unload_bam
        fi

        # Run queries
        for Q in "${QUERIES[@]}"; do
            run_bench "${SF}" "${MODE}" "${Q}" "${DEVICES}" "${CS}"
        done
    done

    unset loaded_for_loader
    unload_bam

    log "=== Chunk size ${CS} complete ==="
    log ""
done

# ---- Summary ----
log "============================================================"
log "=== Chunk size sweep complete ==="
log "Results : ${LOG_DIR}"
log "Timings : ${LOG_DIR}/timings.txt"
log "============================================================"

# Print a quick summary table if results exist
if ! $DRY_RUN && [[ -d "${LOG_DIR}" ]]; then
    echo ""
    echo "======== Quick Summary (avg time in ms) ========"
    printf "%-8s" "chunk"
    for MODE in "${MODES[@]}"; do
        for Q in "${QUERIES[@]}"; do
            printf " %20s" "${MODE}/${Q}"
        done
    done
    echo ""

    for CS in "${CHUNK_SIZES[@]}"; do
        printf "%-8s" "${CS}"
        for MODE in "${MODES[@]}"; do
            for Q in "${QUERIES[@]}"; do
                f="${LOG_DIR}/cs_${CS}/sf${SF}/${MODE}/${Q}.txt"
                if [[ -f "$f" ]]; then
                    avg=$(grep -m1 '^time:.*avg=' "$f" | sed 's/.*avg=\([0-9.]*\).*/\1/' 2>/dev/null || echo "-")
                    printf " %20s" "${avg}"
                else
                    printf " %20s" "-"
                fi
            done
        done
        echo ""
    done
fi
