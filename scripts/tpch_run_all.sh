#!/usr/bin/env bash
# ============================================================
# tpch_run_all.sh — Automated TPC-H benchmark across multiple
# scale factors and execution modes.
#
# Execution modes:
#   1. gidp                    — GDS synchronous I/O (cuFile), compressed
#   2. gidp+bam                — GPU-initiated I/O (BaM), compressed
#   3. gidp+uncomp             — GDS synchronous I/O (cuFile), uncompressed
#   4. gidp+bam+uncomp         — GPU-initiated I/O (BaM), uncompressed
#   5. gidp+bam+fusion+uncomp  — Fused BaM I/O, uncompressed
#   6. gidp+bam+fusion         — Fused BaM I/O + GPU LZ4 decompression
#   7. datapathfusion          — DataPathFusion (PFOR/FSST compression)
#   8. datapathfusion+uncomp   — DataPathFusion, uncompressed
#
# Per-SF workflow:
#   [FS mode]  Load gidp (compressed)    → bench gidp queries
#   [BaM mode] (same data)               → bench gidp+bam queries
#   [FS mode]  Load gidp (uncompressed)  → bench gidp+uncomp queries (NVME)
#                                         → [BaM mode] bench gidp+bam+uncomp queries
#                                                      bench gidp+bam+fusion+uncomp queries
#   [FS mode]  Load gidp+bam+fusion      → [BaM mode] bench gidp+bam+fusion queries
#   [FS mode]  Load datapathfusion        → [BaM mode] bench datapathfusion queries
#   [FS mode]  Load datapathfusion(uncomp)→ [BaM mode] bench datapathfusion+uncomp queries
#
# All modes support: Q1, Q3, Q5, Q6, Q13, Q16
#
# Optional: --with-revenue enables a revenue selectivity sweep
# (SD_HIGH from 19930101..19990101, SD_LOW=19920101).
# Revenue results are stored under ${LOG_DIR}/revenue/sf${SF}/${mode}/
# separately from TPC-H query results for easier plotting.
#
# Usage:
#   scripts/tpch_run_all.sh [OPTIONS]
#   scripts/tpch_run_all.sh --dry-run          # preview commands
#   scripts/tpch_run_all.sh -s 100,200 -n 5    # custom SF + trials
#   scripts/tpch_run_all.sh --with-revenue      # include revenue sweep
# ============================================================
set -euo pipefail

# ---- Configuration (override via CLI flags) ----
SCALE_FACTORS=(50 100 200 300)
QUERIES=(q1 q3 q5 q6 q13 q16 q3sel)
NTHREADS=32
NTRIALS=10
BENCH_TIMEOUT=15                     # per-trial timeout (seconds)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source the shared environment. Reviewers override NVMe device paths,
# mount point, cuFile config, etc. in scripts/common.sh (see README §2.3).
# shellcheck source=./common.sh
. "${SCRIPT_DIR}/common.sh"

DATA_BASE="${TPCH_ROOT}"
BUILD_DIR="${PROJECT_DIR}/build"
BAM_MODULE_DIR="${PROJECT_DIR}/bam/build/module"
SIDEWAYS_SCRIPT="${SCRIPT_DIR}/golap/01_sideways_pruning.sh"
BENCH_SCRIPT="${SCRIPT_DIR}/bench.sh"

ENABLE_SIDEWAYS=true        # run sideways pruning if data missing
ENABLE_CLEANUP_AFTER=false  # remove extracted sideways data after each SF
ZONE_MAP="-Z"              # zone map flag (set empty to disable)
DRY_RUN=false
SKIP_PHASES=(uncomp)       # uncomp is skipped by default (opt-in only)
SKIP_LOAD=false
USER_LOG_DIR=""

# Revenue selectivity sweep (disable with --no-revenue)
ENABLE_REVENUE=true
REVENUE_ONLY=false
REVENUE_SD_LOW=19920101
REVENUE_DATE_HIGHS=(19920701 19930101 19940101 19950101 19960101 19970101 19980101 19990101)
REVENUE_IO_SIZE=5100       # -Q parameter for revenue query

# Q3SEL selectivity sweep
Q3SEL_SELECTIVITIES=(20 40 60 80 100)

# ---- CLI parsing ----
usage() {
    cat <<'USAGE'
Usage: tpch_run_all.sh [OPTIONS]

Options:
  -s SF1,SF2,...      Scale factors (default: 50,100,200,300)
  -q Q1,Q3,...        Queries to run (default: q1,q3,q5,q6,q13,q16)
  -n TRIALS           Number of bench trials (default: 10)
  -w THREADS          Worker threads (default: 32)
  -t TIMEOUT          Per-trial timeout in seconds (default: 120)
  --no-zonemap        Disable zone map pruning
  --no-sideways       Don't auto-generate sideways data
  --cleanup-after     Remove extracted sideways data after each SF
                      (archive in sideways_zst/ or compressed/ is kept)
  --skip PHASE        Skip a phase (repeatable). Names:
                      gidp, gidp+bam, gidp+bam+fusion, datapathfusion
  --no-load           Skip data loading (assume data already on devices)
  --log-dir DIR       Use existing log directory (append results)
  --no-revenue        Disable revenue selectivity sweep (enabled by default)
  --revenue-only      Run ONLY revenue sweep (skip TPC-H queries)
  --dry-run           Print commands without executing
  -h, --help          Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s) IFS=',' read -ra SCALE_FACTORS <<< "$2"; shift 2 ;;
        -q) IFS=',' read -ra QUERIES <<< "$2"; shift 2 ;;
        -n) NTRIALS="$2"; shift 2 ;;
        -w) NTHREADS="$2"; shift 2 ;;
        -t) BENCH_TIMEOUT="$2"; shift 2 ;;
        --no-zonemap)       ZONE_MAP=""; shift ;;
        --no-sideways)      ENABLE_SIDEWAYS=false; shift ;;
        --cleanup-after)    ENABLE_CLEANUP_AFTER=true; shift ;;
        --skip)             SKIP_PHASES+=("$2"); shift 2 ;;
        --no-load)          SKIP_LOAD=true; shift ;;
        --log-dir)          USER_LOG_DIR="$2"; shift 2 ;;
        --no-revenue)       ENABLE_REVENUE=false; shift ;;
        --revenue-only)     ENABLE_REVENUE=true; REVENUE_ONLY=true; shift ;;
        --dry-run)          DRY_RUN=true; shift ;;
        -h|--help)          usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---- Helpers ----
if [[ -n "${USER_LOG_DIR}" ]]; then
    LOG_DIR="${USER_LOG_DIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="${PROJECT_DIR}/logs/tpch_run_all/${TIMESTAMP}"
fi
mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/main.log"
}

# Record wall-clock time of a command into timings.txt
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

phase_skipped() {
    local phase="$1"
    shift
    local aliases=("$@")
    for s in "${SKIP_PHASES[@]+"${SKIP_PHASES[@]}"}"; do
        [[ "$s" == "$phase" ]] && return 0
        for a in "${aliases[@]+"${aliases[@]}"}"; do
            [[ "$s" == "$a" ]] && return 0
        done
    done
    return 1
}

# ---- Pre-flight checks ----
preflight() {
    local ok=true
    for bin in tpchdb tpchloader; do
        if [[ ! -x "${BUILD_DIR}/${bin}" ]]; then
            log "ERROR: ${BUILD_DIR}/${bin} not found. Run cmake --build first."
            ok=false
        fi
    done
    if [[ ! -f "${BENCH_SCRIPT}" ]]; then
        log "ERROR: bench.sh not found at ${BENCH_SCRIPT}"
        ok=false
    fi
    $ok || exit 1
}

# ============================================================
# BaM module management
# ============================================================
BAM_LOADED=false

load_bam() {
    if $BAM_LOADED; then
        log "BaM already loaded, skipping"
        return 0
    fi
    log ">>> Loading BaM module (unmounting ${MOUNT_POINT}) <<<"
    if $DRY_RUN; then
        log "[DRY-RUN] load_bam"
        BAM_LOADED=true
        return 0
    fi
    sudo umount "${MOUNT_POINT}" 2>/dev/null || true
    sudo mdadm --stop "${MDADM_DEV}" 2>/dev/null || true
    for bdf in "${NVME_PCI_BDF[@]}"; do
        [ -z "${bdf}" ] && continue
        echo -n "${bdf}" \
            | sudo tee "/sys/bus/pci/devices/${bdf}/driver/unbind" \
            2>/dev/null || true
    done
    pushd "${BAM_MODULE_DIR}" > /dev/null
    sudo make load
    popd > /dev/null
    BAM_LOADED=true
    log "BaM module loaded"
}

unload_bam() {
    if ! $BAM_LOADED; then
        log "BaM not loaded, skipping unload"
        return 0
    fi
    log ">>> Unloading BaM module (remounting ${MOUNT_POINT}) <<<"
    if $DRY_RUN; then
        log "[DRY-RUN] unload_bam"
        BAM_LOADED=false
        return 0
    fi
    pushd "${BAM_MODULE_DIR}" > /dev/null
    sudo make unload
    popd > /dev/null
    for bdf in "${NVME_PCI_BDF[@]}"; do
        [ -z "${bdf}" ] && continue
        echo "${bdf}" \
            | sudo tee /sys/bus/pci/drivers/nvme/bind \
            2>/dev/null || true
    done
    sleep 3  # wait for NVMe devices to re-appear
    sudo mdadm --stop "${MDADM_DEV}" 2>/dev/null || true
    sudo mdadm --assemble "${MDADM_DEV}" "${NVME_P2[@]}"
    sudo mount -t xfs "${MDADM_DEV}" "${MOUNT_POINT}/"
    sudo chown "$(whoami):$(whoami)" ${DEVICES_NVME//,/ }
    BAM_LOADED=false
    log "Filesystem remounted at ${MOUNT_POINT}"
}

# Ensure BaM is unloaded on exit (safety net)
cleanup() {
    if $BAM_LOADED; then
        log "Cleanup: unloading BaM module..."
        unload_bam || true
    fi
    log "=== Script finished ==="
}
trap cleanup EXIT

# ============================================================
# Data preparation
# ============================================================
prepare_sideways_data() {
    local sf="$1"
    local sideways_dir="${DATA_BASE}/sideways/sf${sf}"
    local archive="${DATA_BASE}/compressed/sf${sf}.tar.zstd"
    local archive_zst="${DATA_BASE}/sideways_zst/sf${sf}.tar.zst"

    # Already exists?
    if [[ -d "${sideways_dir}/lineitem" ]] \
       && [[ -n "$(ls -A "${sideways_dir}/lineitem/" 2>/dev/null)" ]]; then
        log "Sideways data for SF${sf} already present"
        return 0
    fi

    # Try decompression from archive (check both locations automatically)
    local found_archive=""
    if [[ -f "${archive_zst}" ]]; then
        found_archive="${archive_zst}"
    elif [[ -f "${archive}" ]]; then
        found_archive="${archive}"
    fi
    if [[ -n "${found_archive}" ]]; then
        log "Decompressing ${found_archive} ..."
        if $DRY_RUN; then
            log "[DRY-RUN] zstd decompress sf${sf}"
            return 0
        fi
        mkdir -p "${DATA_BASE}/sideways"
        time_cmd "sf${sf}_decompress" \
            tar --use-compress-program=pzstd -xf "${found_archive}" -C "${DATA_BASE}/sideways/"
        return 0
    fi

    # Generate via sideways pruning
    if $ENABLE_SIDEWAYS; then
        local input_dir="${DATA_BASE}/input${sf}"
        if [[ ! -d "${input_dir}" ]]; then
            log "ERROR: No data source for SF${sf} (no sideways, no archive, no input)"
            return 1
        fi
        log "Generating sideways data for SF${sf} ..."
        if $DRY_RUN; then
            log "[DRY-RUN] sideways pruning sf${sf}"
            return 0
        fi
        # Create a temp copy of the pruning script with SF patched and
        # interactive prompts disabled (force-clean on non-empty dirs)
        local tmp_script
        tmp_script=$(mktemp /tmp/sideways_sf${sf}.XXXXXX.sh)
        sed -e "s/^SF=.*/SF=${sf}/" \
            -e 's/read -p .*/CONFIRM=y/' \
            -e '/echo "Please rerun the script."/d' \
            -e 's/exit 0\s*;;/;;/' \
            "${SIDEWAYS_SCRIPT}" > "${tmp_script}"
        chmod +x "${tmp_script}"
        # Pre-clean output dirs so ensure_empty_dir is a no-op
        rm -rf "${sideways_dir}/lineitem" "${sideways_dir}/orders" "${sideways_dir}/tmp"
        time_cmd "sf${sf}_sideways_pruning" bash "${tmp_script}"
        rm -f "${tmp_script}"
        return 0
    fi

    log "ERROR: Sideways data unavailable for SF${sf}"
    return 1
}

cleanup_sideways_data() {
    local sf="$1"
    local sideways_dir="${DATA_BASE}/sideways/sf${sf}"
    local archive="${DATA_BASE}/compressed/sf${sf}.tar.zstd"
    local archive_zst="${DATA_BASE}/sideways_zst/sf${sf}.tar.zst"

    if [[ ! -d "${sideways_dir}" ]]; then return 0; fi

    # If an archive exists, just remove the extracted directory
    if [[ -f "${archive_zst}" ]] || [[ -f "${archive}" ]]; then
        log "Cleaning up extracted sideways data for SF${sf} (archive exists)..."
        if ! $DRY_RUN; then
            rm -rf "${sideways_dir}"
            log "Removed ${sideways_dir}"
        fi
        return 0
    fi

    # No archive — compress first, then remove
    log "Compressing sideways data SF${sf} before cleanup..."
    if $DRY_RUN; then
        log "[DRY-RUN] compress + cleanup sf${sf}"
        return 0
    fi
    mkdir -p "${DATA_BASE}/sideways_zst"
    time_cmd "sf${sf}_compress" \
        tar -h --use-compress-program=pzstd -cf "${archive_zst}.tmp" \
            -C "${DATA_BASE}/sideways" "sf${sf}"
    mv "${archive_zst}.tmp" "${archive_zst}"
    rm -rf "${sideways_dir}"
    log "Removed ${sideways_dir}"
}

# ============================================================
# Benchmark helpers
# ============================================================
run_load() {
    local sf="$1" loader_mode="$2"
    local compress="${3:-true}"
    local dir_label="${4:-${loader_mode}}"
    local input_dir="${DATA_BASE}/sideways/sf${sf}"
    local out_dir="${LOG_DIR}/sf${sf}/${dir_label}"
    local log_file="${out_dir}/load.log"
    mkdir -p "${out_dir}"

    if $SKIP_LOAD; then
        log "LOAD SF${sf} mode=${loader_mode} compress=${compress} — skipped (--no-load)"
        return 0
    fi

    log "LOAD SF${sf} mode=${loader_mode} compress=${compress} → ${log_file}"
    local cmd=("${BUILD_DIR}/tpchloader"
        -i "${input_dir}"
        -d "${DEVICES_NVME}"
        -x "${loader_mode}"
        -A)
    if [[ "${compress}" == "true" ]]; then
        cmd+=(-c)
    fi

    if $DRY_RUN; then
        log "[DRY-RUN] ${cmd[*]}"
        return 0
    fi
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    time_cmd "sf${sf}_load_${dir_label}" "${cmd[@]}" 2>&1 | tee "${log_file}"
}

run_revenue_sweep() {
    local sf="$1" dir_label="$2" devices="$3"
    local exec_mode="${4:-${dir_label}}"
    local out_dir="${LOG_DIR}/revenue/sf${sf}/${dir_label}"
    mkdir -p "${out_dir}"

    for sd_high in "${REVENUE_DATE_HIGHS[@]}"; do
        local out_file="${out_dir}/revenue_${REVENUE_SD_LOW}_${sd_high}.txt"

        local cmd=(sudo env CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON}" "${BUILD_DIR}/tpchdb"
            -q revenue -x "${exec_mode}" -w "${NTHREADS}"
            -a "${REVENUE_SD_LOW}" -b "${sd_high}" -Q "${REVENUE_IO_SIZE}")
        [[ -n "${ZONE_MAP}" ]] && cmd+=("${ZONE_MAP}")
        cmd+=("${devices}")

        log "  REVENUE SF${sf} ${dir_label} (exec=${exec_mode}) ${REVENUE_SD_LOW}-${sd_high}"
        if $DRY_RUN; then
            log "  [DRY-RUN] ${BENCH_SCRIPT} -n ${NTRIALS} -w -t ${BENCH_TIMEOUT} -o ${out_file} -- ${cmd[*]}"
            continue
        fi
        "${BENCH_SCRIPT}" -n "${NTRIALS}" -w -t "${BENCH_TIMEOUT}" \
            -o "${out_file}" -- "${cmd[@]}" || {
            log "  ERROR: revenue bench failed for SF${sf} ${dir_label} ${REVENUE_SD_LOW}-${sd_high} (rc=$?)"
            exit 1
        }
    done
}

run_q3sel_sweep() {
    local sf="$1" dir_label="$2" devices="$3"
    local exec_mode="${4:-${dir_label}}"
    local out_dir="${LOG_DIR}/sf${sf}/${dir_label}"
    mkdir -p "${out_dir}"

    for sel in "${Q3SEL_SELECTIVITIES[@]}"; do
        local out_file="${out_dir}/q3sel_s${sel}.txt"

        local cmd=(sudo env CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON}" "${BUILD_DIR}/tpchdb"
            -q q3sel -x "${exec_mode}" -w "${NTHREADS}"
            -s "${sel}")
        [[ -n "${ZONE_MAP}" ]] && cmd+=("${ZONE_MAP}")
        cmd+=("${devices}")

        log "  Q3SEL SF${sf} ${dir_label} (exec=${exec_mode}) sel=${sel}%"
        if $DRY_RUN; then
            log "  [DRY-RUN] ${BENCH_SCRIPT} -n ${NTRIALS} -w -t ${BENCH_TIMEOUT} -o ${out_file} -- ${cmd[*]}"
            continue
        fi
        "${BENCH_SCRIPT}" -n "${NTRIALS}" -w -t "${BENCH_TIMEOUT}" \
            -o "${out_file}" -- "${cmd[@]}" || {
            log "  ERROR: q3sel bench failed for SF${sf} ${dir_label} sel=${sel}% (rc=$?)"
            exit 1
        }
    done
}

run_bench() {
    local sf="$1" dir_label="$2" query="$3" devices="$4"
    local exec_mode="${5:-${dir_label}}"
    local out_dir="${LOG_DIR}/sf${sf}/${dir_label}"
    local out_file="${out_dir}/${query}.txt"
    mkdir -p "${out_dir}"

    local cmd=(sudo env CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON}" "${BUILD_DIR}/tpchdb"
        -q "${query}" -x "${exec_mode}" -w "${NTHREADS}")
    [[ -n "${ZONE_MAP}" ]] && cmd+=("${ZONE_MAP}")
    cmd+=("${devices}")

    log "  BENCH SF${sf} ${dir_label} (exec=${exec_mode}) ${query}"
    if $DRY_RUN; then
        log "  [DRY-RUN] ${BENCH_SCRIPT} -n ${NTRIALS} -w -t ${BENCH_TIMEOUT} -o ${out_file} -- ${cmd[*]}"
        return 0
    fi
    "${BENCH_SCRIPT}" -n "${NTRIALS}" -w -t "${BENCH_TIMEOUT}" \
        -o "${out_file}" -- "${cmd[@]}" || {
        log "  ERROR: bench failed for SF${sf} ${dir_label} ${query} (rc=$?)"
        exit 1
    }
}

# Dispatch: q3sel → sweep all selectivities, otherwise → single run_bench
run_query() {
    local sf="$1" dir_label="$2" query="$3" devices="$4"
    local exec_mode="${5:-${dir_label}}"
    if [[ "$query" == "q3sel" ]]; then
        if (( sf > 100 )); then
            log "  SKIP: q3sel disabled for SF${sf} (>100)"
            return 0
        fi
        run_q3sel_sweep "$sf" "$dir_label" "$devices" "$exec_mode"
    else
        run_bench "$sf" "$dir_label" "$query" "$devices" "$exec_mode"
    fi
}

# ============================================================
# Main
# ============================================================
preflight

# Activate sudo session early and set device permissions
if ! $DRY_RUN; then
    sudo ls -lha /dev/nvme0n1p1 > /dev/null
    sudo chown "$(whoami):$(whoami)" ${DEVICES_NVME//,/ }
fi

log "============================================================"
log "  TPC-H Benchmark Suite"
log "============================================================"
log "Scale factors : ${SCALE_FACTORS[*]}"
log "Queries       : ${QUERIES[*]}"
log "Threads       : ${NTHREADS}"
log "Trials        : ${NTRIALS}"
log "Timeout       : ${BENCH_TIMEOUT}s"
log "Zone map      : ${ZONE_MAP:-disabled}"
log "Skip phases   : ${SKIP_PHASES[*]+"${SKIP_PHASES[*]}"}"
log "Revenue sweep : ${ENABLE_REVENUE}$(${REVENUE_ONLY} && echo ' (revenue-only)')"
log "Cleanup after : ${ENABLE_CLEANUP_AFTER}"
log "Dry run       : ${DRY_RUN}"
log "Log directory : ${LOG_DIR}"
log "============================================================"
if $ENABLE_CLEANUP_AFTER; then
    log ""
    log "NOTE: --cleanup-after is enabled."
    log "  After each SF benchmark completes, the extracted sideways/"
    log "  directory will be removed. If no archive exists yet, the data"
    log "  will be compressed to sideways_zst/ first."
    log "  Next run will auto-extract from the archive as needed."
fi
log ""

for SF in "${SCALE_FACTORS[@]}"; do
    log "============================================================"
    log "=== SF${SF} ==="
    log "============================================================"

    # ---- Step 0: Data preparation ----
    if ! $SKIP_LOAD; then
        prepare_sideways_data "${SF}" || { log "SKIP SF${SF}: data unavailable"; continue; }
    fi

    # ---- Phase 1: gidp (GDS sync I/O, compressed) ----
    if ! phase_skipped gidp; then
        log "--- Phase 1/8: gidp ---"
        run_load "${SF}" "gidp"
        if ! $REVENUE_ONLY; then
            for q in "${QUERIES[@]}"; do
                run_query "${SF}" "gidp" "${q}" "${DEVICES_NVME}"
            done
        fi
        if $ENABLE_REVENUE; then
            run_revenue_sweep "${SF}" "gidp" "${DEVICES_NVME}"
        fi
    fi

    # ---- Phase 2: gidp+bam (BaM I/O, compressed, same data as gidp) ----
    if ! phase_skipped bam "gidp+bam"; then
        log "--- Phase 2/8: gidp+bam ---"
        # If gidp phase was skipped, we still need its data loaded
        if phase_skipped gidp; then
            run_load "${SF}" "gidp"
        fi
        load_bam
        if ! $REVENUE_ONLY; then
            for q in "${QUERIES[@]}"; do
                run_query "${SF}" "gidp+bam" "${q}" "${DEVICES_BAM}"
            done
        fi
        if $ENABLE_REVENUE; then
            run_revenue_sweep "${SF}" "gidp+bam" "${DEVICES_BAM}"
        fi
        unload_bam
    fi

    # ---- Phase 3/4/5: uncompressed I/O ----
    # "uncomp" skips all three; individual phases can be skipped separately
    run_ph3=true; run_ph4=true; run_ph5=true
    phase_skipped uncomp "gidp-uncomp"   "gidp+uncomp"             && run_ph3=false
    phase_skipped uncomp "bam-uncomp"    "gidp+bam+uncomp"         && run_ph4=false
    phase_skipped uncomp "fusion-uncomp" "gidp+bam+fusion+uncomp"  && run_ph5=false
    if $run_ph3 || $run_ph4 || $run_ph5; then
        log "--- Phase 3+4+5/8: uncompressed I/O (ph3=${run_ph3}, ph4=${run_ph4}, ph5=${run_ph5}) ---"
        run_load "${SF}" "gidp" "false" "gidp+uncomp"
        # Phase 3: gidp+uncomp — GDS sync I/O on NVME (no BaM needed)
        if $run_ph3; then
            if ! $REVENUE_ONLY; then
                for q in "${QUERIES[@]}"; do
                    run_query "${SF}" "gidp+uncomp" "${q}" "${DEVICES_NVME}" "gidp"
                done
            fi
            if $ENABLE_REVENUE; then
                run_revenue_sweep "${SF}" "gidp+uncomp" "${DEVICES_NVME}" "gidp"
            fi
        fi
        # Phase 4+5: BaM modes need BaM module
        if $run_ph4 || $run_ph5; then
            load_bam
            if $run_ph4; then
                if ! $REVENUE_ONLY; then
                    for q in "${QUERIES[@]}"; do
                        run_query "${SF}" "gidp+bam+uncomp" "${q}" "${DEVICES_BAM}" "gidp+bam"
                    done
                fi
                if $ENABLE_REVENUE; then
                    run_revenue_sweep "${SF}" "gidp+bam+uncomp" "${DEVICES_BAM}" "gidp+bam"
                fi
            fi
            if $run_ph5; then
                if ! $REVENUE_ONLY; then
                    for q in "${QUERIES[@]}"; do
                        run_query "${SF}" "gidp+bam+fusion+uncomp" "${q}" "${DEVICES_BAM}" "gidp+bam+fusion"
                    done
                fi
                if $ENABLE_REVENUE; then
                    run_revenue_sweep "${SF}" "gidp+bam+fusion+uncomp" "${DEVICES_BAM}" "gidp+bam+fusion"
                fi
            fi
            unload_bam
        fi
    fi

    # ---- Phase 6: gidp+bam+fusion (compressed data + GPU LZ4 decompression) ----
    if ! phase_skipped decomp "gidp+bam+fusion"; then
        log "--- Phase 6/8: gidp+bam+fusion ---"
        run_load "${SF}" "gidp+bam+fusion"
        load_bam
        if ! $REVENUE_ONLY; then
            for q in "${QUERIES[@]}"; do
                run_query "${SF}" "gidp+bam+fusion" "${q}" "${DEVICES_BAM}"
            done
        fi
        if $ENABLE_REVENUE; then
            run_revenue_sweep "${SF}" "gidp+bam+fusion" "${DEVICES_BAM}"
        fi
        unload_bam
    fi

    # ---- Phase 7: datapathfusion (compressed) ----
    if ! phase_skipped dpf "datapathfusion"; then
        log "--- Phase 7/8: datapathfusion ---"
        run_load "${SF}" "datapathfusion"
        load_bam
        if ! $REVENUE_ONLY; then
            for q in "${QUERIES[@]}"; do
                run_query "${SF}" "datapathfusion" "${q}" "${DEVICES_BAM}"
            done
        fi
        if $ENABLE_REVENUE; then
            run_revenue_sweep "${SF}" "datapathfusion" "${DEVICES_BAM}"
        fi
        unload_bam
    fi

    # ---- Phase 8: datapathfusion+uncomp (uncompressed) ----
    if ! phase_skipped uncomp "dpf-uncomp" "datapathfusion+uncomp"; then
        log "--- Phase 8/8: datapathfusion+uncomp ---"
        run_load "${SF}" "datapathfusion" "false" "datapathfusion+uncomp"
        load_bam
        if ! $REVENUE_ONLY; then
            for q in "${QUERIES[@]}"; do
                run_query "${SF}" "datapathfusion+uncomp" "${q}" "${DEVICES_BAM}" "datapathfusion"
            done
        fi
        if $ENABLE_REVENUE; then
            run_revenue_sweep "${SF}" "datapathfusion+uncomp" "${DEVICES_BAM}" "datapathfusion"
        fi
        unload_bam
    fi

    # ---- Optionally compress to free space ----
    if $ENABLE_CLEANUP_AFTER; then
        cleanup_sideways_data "${SF}"
    fi

    log "=== SF${SF} complete ==="
    log ""
done

log "============================================================"
log "=== All experiments complete ==="
log "Results : ${LOG_DIR}"
log "Timings : ${LOG_DIR}/timings.txt"
log "============================================================"
