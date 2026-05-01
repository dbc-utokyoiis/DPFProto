#!/usr/bin/env bash
# ============================================================
# tpch_chunk_size_sweep.sh — Benchmark TPC-H across different
# page (chunk) sizes: 64K, 128K, 256K, 512K, 1M.
#
# For each chunk size, re-loads data with the corresponding
# page size and benchmarks selected execution modes and queries.
#
# Usage:
#   scripts/tpch_chunk_size_sweep.sh [OPTIONS]
#   scripts/tpch_chunk_size_sweep.sh --dry-run
#   scripts/tpch_chunk_size_sweep.sh -s 100 -c 64K,256K,1M -x gidp+bam+fusion -q q1,q6
#
# Options:
#   -s SF              Scale factor (default: 100)
#   -c SIZES           Chunk sizes, comma-separated (default: 64K,128K,256K,512K,1M)
#   -x MODES           Execution modes to bench (default: gidp+bam,gidp+bam+fusion)
#   -q QUERIES         Queries to run (default: q3,q13)
#   --allquery         Run all queries (q1,q3,q5,q6,q13,q16)
#   -n TRIALS          Number of bench trials (default: 5)
#   -t TIMEOUT         Per-trial timeout in seconds (default: 60)
#   --log-dir DIR      Use existing log directory (append results)
#   --no-load          Skip data loading (assume data already on devices)
#   --with-revenue     Include revenue selectivity sweep
#   --dry-run          Print commands without executing
#   -h, --help         Show this help
# ============================================================
set -euo pipefail

# ---- Configuration ----
SF=100
CHUNK_SIZES=(2M 64K 128K 256K 512K 1M)
MODES=(gidp gidp+bam gidp+bam+fusion datapathfusion)
QUERIES=(q3 q13)
ALL_QUERIES=(q1 q3 q5 q6 q13 q16)
NTRIALS=10
BENCH_TIMEOUT=60
NTHREADS=32

DEVICES_NVME="/dev/nvme0n1p1,/dev/nvme1n1p1,/dev/nvme2n1p1,/dev/nvme3n1p1"
DEVICES_BAM="/dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3"
DATA_BASE="/export/data1/tpch"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BAM_MODULE_DIR="${PROJECT_DIR}/bam/build/module"
BENCH_SCRIPT="${SCRIPT_DIR}/bench.sh"

export CUFILE_ENV_PATH_JSON="${PROJECT_DIR}/config/cufile.json"

DRY_RUN=false
SKIP_LOAD=false
ENABLE_REVENUE=false
USER_LOG_DIR=""
ZONE_MAP="-Z"

REVENUE_SD_LOW=19920101
REVENUE_DATE_HIGHS=(19930101 19940101 19950101 19960101 19970101 19980101 19990101)
REVENUE_IO_SIZE=5100

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
        --no-zonemap)     ZONE_MAP=""; shift ;;
        --allquery)       QUERIES=("${ALL_QUERIES[@]}"); shift ;;
        --with-revenue)   ENABLE_REVENUE=true; shift ;;
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
    LOG_DIR="${PROJECT_DIR}/logs/tpch_chunk_sweep/${TIMESTAMP}"
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
# Maps execution mode → loader mode (what -x to pass to tpchloader)
loader_mode_for() {
    case "$1" in
        gidp)                    echo "gidp" ;;
        gidp+bam)                echo "gidp" ;;
        gidp+bam+fusion)         echo "gidp+bam+fusion" ;;
        datapathfusion)          echo "datapathfusion" ;;
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
    local cmd=("${BUILD_DIR}/tpchloader"
        -i "${input_dir}"
        -d "${DEVICES_NVME}"
        -x "${loader_mode}"
        -p "${chunk_size}"
        -c
        -A)

    if $DRY_RUN; then
        log "[DRY-RUN] ${cmd[*]}"
        return 0
    fi
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    time_cmd "sf${sf}_load_${loader_mode}_cs${chunk_size}" "${cmd[@]}" 2>&1 | tee "${log_file}"
}

# ---- Run benchmark ----
run_bench() {
    local sf="$1" mode="$2" query="$3" devices="$4" chunk_size="$5"
    local exec_mode="${6:-${mode}}"
    local out_dir="${LOG_DIR}/cs_${chunk_size}/sf${sf}/${mode}"
    local out_file="${out_dir}/${query}.txt"
    mkdir -p "${out_dir}"

    local cmd=(sudo env CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON}" "${BUILD_DIR}/tpchdb"
        -q "${query}" -x "${exec_mode}" -w "${NTHREADS}")
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

# ---- Revenue sweep ----
run_revenue_sweep() {
    local sf="$1" mode="$2" devices="$3" chunk_size="$4"
    local exec_mode="${5:-${mode}}"
    local out_dir="${LOG_DIR}/cs_${chunk_size}/revenue/sf${sf}/${mode}"
    mkdir -p "${out_dir}"

    for sd_high in "${REVENUE_DATE_HIGHS[@]}"; do
        local out_file="${out_dir}/revenue_${REVENUE_SD_LOW}_${sd_high}.txt"
        local cmd=(sudo "${BUILD_DIR}/tpchdb"
            -q revenue -x "${exec_mode}" -w "${NTHREADS}"
            -a "${REVENUE_SD_LOW}" -b "${sd_high}" -Q "${REVENUE_IO_SIZE}")
        [[ -n "${ZONE_MAP}" ]] && cmd+=("${ZONE_MAP}")
        cmd+=("${devices}")

        log "  REVENUE SF${sf} ${mode} chunk=${chunk_size} ${REVENUE_SD_LOW}-${sd_high}"
        if $DRY_RUN; then
            log "  [DRY-RUN] ${BENCH_SCRIPT} -n ${NTRIALS} -w -t ${BENCH_TIMEOUT} -o ${out_file} -- ${cmd[*]}"
            continue
        fi
        "${BENCH_SCRIPT}" -n "${NTRIALS}" -w -t "${BENCH_TIMEOUT}" \
            -o "${out_file}" -- "${cmd[@]}" || {
            log "  ERROR: revenue bench failed (rc=$?)"
            return 1
        }
    done
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
for bin in tpchdb tpchloader; do
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
log "  TPC-H Chunk Size Sweep"
log "============================================================"
log "Scale factor : ${SF}"
log "Chunk sizes  : ${CHUNK_SIZES[*]}"
log "Modes        : ${MODES[*]}"
log "Queries      : ${QUERIES[*]}"
log "Trials       : ${NTRIALS}"
log "Timeout      : ${BENCH_TIMEOUT}s"
log "Zone map     : ${ZONE_MAP:-disabled}"
log "Revenue      : ${ENABLE_REVENUE}"
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
            run_bench "${SF}" "${MODE}" "${Q}" "${DEVICES}" "${CS}" "${MODE}"
        done

        # Revenue sweep
        if $ENABLE_REVENUE; then
            run_revenue_sweep "${SF}" "${MODE}" "${DEVICES}" "${CS}" "${MODE}"
        fi
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
            printf " %16s" "${MODE}/${Q}"
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
                    printf " %16s" "${avg}"
                else
                    printf " %16s" "-"
                fi
            done
        done
        echo ""
    done
fi
