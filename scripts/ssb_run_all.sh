#!/usr/bin/env bash
# ============================================================
# ssb_run_all.sh — Automated SSB benchmark across multiple
# scale factors and execution modes.
#
# Execution modes:
#   1. gidp                — GDS synchronous I/O (cuFile), compressed
#   2. gidp+bam            — GPU-initiated I/O (BaM), compressed
#                            (same loaded data as gidp)
#   3. gidp+bam+fusion     — Fused BaM I/O + GPU LZ4 decompression
#                            (separate load: LZ4-only compression)
#   4. datapathfusion       — Fused BaM I/O + PFOR decompression + query
#                            (separate load: PFOR/FSST_ROWID compression)
#   5. uncomp              — Uncompressed baseline (single load, all 4 modes)
#
# Per-SF workflow:
#   [FS mode]  Load gidp (compressed)       → bench gidp queries (NVME)
#   [BaM mode] (same data)                  → bench gidp+bam queries (BaM)
#   [FS mode]  Load gidp+bam+fusion (LZ4)   → [BaM mode] bench gidp+bam+fusion
#   [FS mode]  Load datapathfusion (PFOR)    → [BaM mode] bench datapathfusion
#   [FS mode]  Load uncomp (no compression) → bench all 4 modes
#
# After all benchmarks, correctness is verified against answers/ssb/sf${SF}/.
#
# Usage:
#   scripts/ssb_run_all.sh [OPTIONS]
#   scripts/ssb_run_all.sh --dry-run
#   scripts/ssb_run_all.sh -s 100 -n 5
#   scripts/ssb_run_all.sh -s 10,100 --skip fusion -n 3
# ============================================================
set -euo pipefail

# ---- Configuration (override via CLI flags) ----
SCALE_FACTORS=(10 100)
QUERIES=(q11 q12 q13 q21 q22 q23 q31 q32 q33 q34 q41 q42 q43 revenue)
NTHREADS=32
NTRIALS=10
BENCH_TIMEOUT=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source the shared environment. Reviewers override NVMe device paths,
# mount point, cuFile config, etc. in scripts/common.sh (see README §2.3).
# shellcheck source=./common.sh
. "${SCRIPT_DIR}/common.sh"

DATA_BASE="${SSB_ROOT}"
BUILD_DIR="${PROJECT_DIR}/build"
BAM_MODULE_DIR="${PROJECT_DIR}/bam/build/module"
SIDEWAYS_SCRIPT="${SCRIPT_DIR}/golap/02_ssb_sideways_pruning.sh"
BENCH_SCRIPT="${SCRIPT_DIR}/bench.sh"
ANSWERS_DIR="${PROJECT_DIR}/answers/ssb"

ENABLE_SIDEWAYS=true
ZONE_MAP="-Z"
DRY_RUN=false
SKIP_PHASES=(uncomp)       # uncomp is skipped by default (opt-in only)
SKIP_LOAD=false
VERIFY=false
USER_LOG_DIR=""
SKIP_REVENUE_SWEEP=false

# ---- Revenue selectivity sweep ----
SELECTIVITIES=(sel7 sel15 sel30 sel45 sel60 sel75 sel90 sel100)
QT_MAX=51

declare -A SEL_SD_LOW SEL_SD_HIGH
SEL_SD_LOW[sel7]=19920101;   SEL_SD_HIGH[sel7]=19920701
SEL_SD_LOW[sel15]=19920101;  SEL_SD_HIGH[sel15]=19921231
SEL_SD_LOW[sel30]=19920101;  SEL_SD_HIGH[sel30]=19931231
SEL_SD_LOW[sel45]=19920101;  SEL_SD_HIGH[sel45]=19941231
SEL_SD_LOW[sel60]=19920101;  SEL_SD_HIGH[sel60]=19951231
SEL_SD_LOW[sel75]=19920101;  SEL_SD_HIGH[sel75]=19961231
SEL_SD_LOW[sel90]=19920101;  SEL_SD_HIGH[sel90]=19971231
SEL_SD_LOW[sel100]=19920101; SEL_SD_HIGH[sel100]=19981231

# ---- CLI parsing ----
usage() {
    cat <<'USAGE'
Usage: ssb_run_all.sh [OPTIONS]

Options:
  -s SF1,SF2,...      Scale factors (default: 10,100)
  -q Q1,Q2,...        Queries to run (default: all 13 SSB queries)
  -n TRIALS           Number of bench trials (default: 10)
  -w THREADS          Worker threads (default: 32)
  -t TIMEOUT          Per-trial timeout in seconds (default: 60)
  -d DEVICES          NVMe device list (comma-separated)
  --no-zonemap        Disable zone map pruning
  --no-sideways       Don't auto-generate sideways data
  --skip PHASE        Skip a phase (repeatable):
                      gidp, gidp+bam, gidp+bam+fusion, datapathfusion
  --no-load           Skip data loading (assume data already on devices)
  --skip-revenue-sweep  Skip revenue selectivity sweep (sel7..sel100)
  -f, --verify        Run correctness verification after benchmarks
  --log-dir DIR       Use existing log directory (append results)
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
        -d) DEVICES_NVME="$2"; shift 2 ;;
        --no-zonemap)    ZONE_MAP=""; shift ;;
        --no-sideways)   ENABLE_SIDEWAYS=false; shift ;;
        --skip)          SKIP_PHASES+=("$2"); shift 2 ;;
        --no-load)       SKIP_LOAD=true; shift ;;
        --skip-revenue-sweep) SKIP_REVENUE_SWEEP=true; shift ;;
        -f|--verify)     VERIFY=true; shift ;;
        --log-dir)       USER_LOG_DIR="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---- Helpers ----
if [[ -n "${USER_LOG_DIR}" ]]; then
    LOG_DIR="${USER_LOG_DIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="${PROJECT_DIR}/logs/ssb_run_all/${TIMESTAMP}"
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

phase_skipped() {
    local phase="$1"
    for s in "${SKIP_PHASES[@]+"${SKIP_PHASES[@]}"}"; do
        [[ "$s" == "$phase" ]] && return 0
    done
    return 1
}

# ---- Pre-flight checks ----
preflight() {
    local ok=true
    for bin in ssbdb ssbloader; do
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
    sleep 3
    sudo mdadm --stop "${MDADM_DEV}" 2>/dev/null || true
    sudo mdadm --assemble "${MDADM_DEV}" "${NVME_P2[@]}"
    sudo mount -t xfs "${MDADM_DEV}" "${MOUNT_POINT}/"
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

# ============================================================
# Data preparation
# ============================================================
prepare_sideways_data() {
    local sf="$1"
    local sideways_dir="${DATA_BASE}/sideways/sf${sf}"
    local archive_zst="${DATA_BASE}/sideways_zst/sf${sf}.tar.zst"

    if [[ -d "${sideways_dir}/lineorder" ]] \
       && [[ -n "$(ls -A "${sideways_dir}/lineorder/" 2>/dev/null)" ]]; then
        log "Sideways data for SF${sf} already present"
        return 0
    fi

    if [[ -f "${archive_zst}" ]]; then
        log "Decompressing ${archive_zst} ..."
        if $DRY_RUN; then
            log "[DRY-RUN] zstd decompress sf${sf}"
            return 0
        fi
        mkdir -p "${DATA_BASE}/sideways"
        time_cmd "sf${sf}_decompress" \
            tar --use-compress-program=pzstd -xf "${archive_zst}" -C "${DATA_BASE}/sideways/"
        return 0
    fi

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
        local tmp_script
        tmp_script=$(mktemp /tmp/ssb_sideways_sf${sf}.XXXXXX.sh)
        sed -e "s/^SF=.*/SF=${sf}/" \
            -e 's/read -p .*/CONFIRM=y/' \
            "${SIDEWAYS_SCRIPT}" > "${tmp_script}"
        chmod +x "${tmp_script}"
        rm -rf "${sideways_dir}/lineorder" "${sideways_dir}/tmp"
        time_cmd "sf${sf}_sideways_pruning" bash "${tmp_script}"
        rm -f "${tmp_script}"
        return 0
    fi

    log "ERROR: Sideways data unavailable for SF${sf}"
    return 1
}

# ============================================================
# Benchmark helpers
# ============================================================
run_load() {
    local sf="$1" loader_mode="$2"
    local dir_label="${3:-${loader_mode}}"
    local input_dir="${DATA_BASE}/sideways/sf${sf}"
    local out_dir="${LOG_DIR}/sf${sf}/${dir_label}"
    local log_file="${out_dir}/load.log"
    mkdir -p "${out_dir}"

    if $SKIP_LOAD; then
        log "LOAD SF${sf} mode=${loader_mode} — skipped (--no-load)"
        return 0
    fi

    log "LOAD SF${sf} mode=${loader_mode} → ${log_file}"
    local cmd=("${BUILD_DIR}/ssbloader"
        -i "${input_dir}"
        -d "${DEVICES_NVME}"
        -x "${loader_mode}"
        -A)
    $VERIFY && cmd+=(-f)

    if $DRY_RUN; then
        log "[DRY-RUN] ${cmd[*]}"
        return 0
    fi
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
    time_cmd "sf${sf}_load_${dir_label}" "${cmd[@]}" 2>&1 | tee "${log_file}"
}

run_bench() {
    local sf="$1" dir_label="$2" query="$3" devices="$4"
    local exec_mode="${5:-${dir_label}}"
    local out_dir="${LOG_DIR}/sf${sf}/${dir_label}"
    local out_file="${out_dir}/${query}.txt"
    mkdir -p "${out_dir}"

    local cmd=(sudo "${BUILD_DIR}/ssbdb"
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

run_revenue_bench() {
    local sf="$1" dir_label="$2" sel="$3" devices="$4"
    local exec_mode="${5:-${dir_label}}"
    local sd_low="${SEL_SD_LOW[${sel}]}"
    local sd_high="${SEL_SD_HIGH[${sel}]}"
    local out_dir="${LOG_DIR}/sf${sf}/revenue/${dir_label}"
    local out_file="${out_dir}/${sel}.txt"
    mkdir -p "${out_dir}"

    local cmd=(sudo "${BUILD_DIR}/ssbdb"
        -q revenue -x "${exec_mode}" -w "${NTHREADS}"
        -a "${sd_low}" -b "${sd_high}" -Q "${QT_MAX}")
    [[ -n "${ZONE_MAP}" ]] && cmd+=("${ZONE_MAP}")
    cmd+=("${devices}")

    log "  BENCH SF${sf} ${dir_label} revenue ${sel} (sd=${sd_low}..${sd_high} qt<${QT_MAX})"
    if $DRY_RUN; then
        log "  [DRY-RUN] ${BENCH_SCRIPT} -n ${NTRIALS} -w -t ${BENCH_TIMEOUT} -o ${out_file} -- ${cmd[*]}"
        return 0
    fi
    "${BENCH_SCRIPT}" -n "${NTRIALS}" -w -t "${BENCH_TIMEOUT}" \
        -o "${out_file}" -- "${cmd[@]}" || {
        log "  ERROR: bench failed for SF${sf} ${dir_label} revenue ${sel} (rc=$?)"
        return 1
    }
}

# ============================================================
# Correctness verification
# ============================================================
VERIFY_SCRIPT="${SCRIPT_DIR}/ssb_verify_bench.sh"

verify_results() {
    local sf="$1"
    local sf_log_dir="${LOG_DIR}/sf${sf}"

    if [[ ! -x "${VERIFY_SCRIPT}" ]]; then
        log "VERIFY SF${sf}: ${VERIFY_SCRIPT} not found, skipping"
        return 0
    fi

    log "=== Correctness verification SF${sf} ==="
    "${VERIFY_SCRIPT}" "${sf_log_dir}" "${sf}" 2>&1 | tee -a "${sf_log_dir}/verify.log"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        log "WARNING: correctness failures detected! See ${sf_log_dir}/verify.log"
    fi
}

# ============================================================
# Performance summary
# ============================================================
print_summary() {
    local sf="$1"
    local summary_file="${LOG_DIR}/sf${sf}/summary.txt"

    log "=== Performance summary SF${sf} ==="

    local all_modes=(gidp gidp+bam gidp+bam+fusion datapathfusion
                     uncomp_gidp uncomp_gidp+bam uncomp_gidp+bam+fusion uncomp_datapathfusion)

    printf "%-6s" "Query" | tee -a "${summary_file}"
    for mode in "${all_modes[@]}"; do
        local mode_dir="${LOG_DIR}/sf${sf}/${mode}"
        if [[ -d "${mode_dir}" ]]; then
            printf " | %20s" "${mode}" | tee -a "${summary_file}"
        fi
    done
    printf "\n" | tee -a "${summary_file}"

    printf "%-6s" "------" | tee -a "${summary_file}"
    for mode in "${all_modes[@]}"; do
        local mode_dir="${LOG_DIR}/sf${sf}/${mode}"
        if [[ -d "${mode_dir}" ]]; then
            printf " | %20s" "--------------------" | tee -a "${summary_file}"
        fi
    done
    printf "\n" | tee -a "${summary_file}"

    for q in "${QUERIES[@]}"; do
        printf "%-6s" "${q}" | tee -a "${summary_file}"
        for mode in "${all_modes[@]}"; do
            local mode_dir="${LOG_DIR}/sf${sf}/${mode}"
            if [[ ! -d "${mode_dir}" ]]; then continue; fi
            local log_file="${mode_dir}/${q}.txt"
            if [[ -f "${log_file}" ]]; then
                # Extract avg time from bench.sh summary
                local avg
                avg=$(grep -m1 '^time:.*avg=' "${log_file}" \
                    | sed 's/^time: *avg=\([0-9.]*\).*/\1/' 2>/dev/null || echo "-")
                printf " | %17s ms" "${avg}" | tee -a "${summary_file}"
            else
                printf " | %20s" "-" | tee -a "${summary_file}"
            fi
        done
        printf "\n" | tee -a "${summary_file}"
    done

    log "Summary saved to ${summary_file}"
}

# ============================================================
# Main
# ============================================================
preflight

if ! $DRY_RUN; then
    sudo ls -lha /dev/nvme0n1p1 > /dev/null
    sudo chown "$(whoami):$(whoami)" ${DEVICES_NVME//,/ }
fi

log "============================================================"
log "  SSB Benchmark Suite"
log "============================================================"
log "Scale factors : ${SCALE_FACTORS[*]}"
log "Queries       : ${QUERIES[*]}"
log "Threads       : ${NTHREADS}"
log "Trials        : ${NTRIALS}"
log "Timeout       : ${BENCH_TIMEOUT}s"
log "Zone map      : ${ZONE_MAP:-disabled}"
log "Skip phases   : ${SKIP_PHASES[*]+"${SKIP_PHASES[*]}"}"
log "Dry run       : ${DRY_RUN}"
log "Log directory : ${LOG_DIR}"
log "============================================================"
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
        log "--- Phase 1/5: gidp ---"
        run_load "${SF}" "gidp"
        for q in "${QUERIES[@]}"; do
            run_bench "${SF}" "gidp" "${q}" "${DEVICES_NVME}"
        done
        if ! $SKIP_REVENUE_SWEEP; then for sel in "${SELECTIVITIES[@]}"; do
            run_revenue_bench "${SF}" "gidp" "${sel}" "${DEVICES_NVME}"
        done; fi
    fi

    # ---- Phase 2: gidp+bam (BaM I/O, compressed, same data as gidp) ----
    if ! phase_skipped "gidp+bam"; then
        log "--- Phase 2/5: gidp+bam ---"
        # If gidp was skipped, still need its data loaded
        if phase_skipped gidp; then
            run_load "${SF}" "gidp"
        fi
        load_bam
        for q in "${QUERIES[@]}"; do
            run_bench "${SF}" "gidp+bam" "${q}" "${DEVICES_BAM}"
        done
        if ! $SKIP_REVENUE_SWEEP; then for sel in "${SELECTIVITIES[@]}"; do
            run_revenue_bench "${SF}" "gidp+bam" "${sel}" "${DEVICES_BAM}"
        done; fi
        unload_bam
    fi

    # ---- Phase 3: gidp+bam+fusion (separate load: LZ4-only) ----
    if ! phase_skipped "gidp+bam+fusion"; then
        log "--- Phase 3/5: gidp+bam+fusion ---"
        run_load "${SF}" "gidp+bam+fusion"
        load_bam
        for q in "${QUERIES[@]}"; do
            run_bench "${SF}" "gidp+bam+fusion" "${q}" "${DEVICES_BAM}"
        done
        if ! $SKIP_REVENUE_SWEEP; then for sel in "${SELECTIVITIES[@]}"; do
            run_revenue_bench "${SF}" "gidp+bam+fusion" "${sel}" "${DEVICES_BAM}"
        done; fi
        unload_bam
    fi

    # ---- Phase 4: datapathfusion (separate load: PFOR/FSST_ROWID) ----
    if ! phase_skipped "datapathfusion"; then
        log "--- Phase 4/5: datapathfusion ---"
        run_load "${SF}" "datapathfusion"
        load_bam
        for q in "${QUERIES[@]}"; do
            run_bench "${SF}" "datapathfusion" "${q}" "${DEVICES_BAM}"
        done
        if ! $SKIP_REVENUE_SWEEP; then for sel in "${SELECTIVITIES[@]}"; do
            run_revenue_bench "${SF}" "datapathfusion" "${sel}" "${DEVICES_BAM}"
        done; fi
        unload_bam
    fi

    # ---- Phase 5: uncomp (uncompressed baseline, all 4 modes) ----
    if ! phase_skipped uncomp; then
        log "--- Phase 5/5: uncomp ---"
        run_load "${SF}" "uncomp"

        # gidp on uncomp data (cuFile sync I/O)
        for q in "${QUERIES[@]}"; do
            run_bench "${SF}" "uncomp_gidp" "${q}" "${DEVICES_NVME}" "gidp"
        done
        if ! $SKIP_REVENUE_SWEEP; then for sel in "${SELECTIVITIES[@]}"; do
            run_revenue_bench "${SF}" "uncomp_gidp" "${sel}" "${DEVICES_NVME}" "gidp"
        done; fi

        # BaM modes on uncomp data
        load_bam
        for q in "${QUERIES[@]}"; do
            run_bench "${SF}" "uncomp_gidp+bam" "${q}" "${DEVICES_BAM}" "gidp+bam"
        done
        if ! $SKIP_REVENUE_SWEEP; then for sel in "${SELECTIVITIES[@]}"; do
            run_revenue_bench "${SF}" "uncomp_gidp+bam" "${sel}" "${DEVICES_BAM}" "gidp+bam"
        done; fi
        for q in "${QUERIES[@]}"; do
            run_bench "${SF}" "uncomp_gidp+bam+fusion" "${q}" "${DEVICES_BAM}" "gidp+bam+fusion"
        done
        if ! $SKIP_REVENUE_SWEEP; then for sel in "${SELECTIVITIES[@]}"; do
            run_revenue_bench "${SF}" "uncomp_gidp+bam+fusion" "${sel}" "${DEVICES_BAM}" "gidp+bam+fusion"
        done; fi
        for q in "${QUERIES[@]}"; do
            run_bench "${SF}" "uncomp_datapathfusion" "${q}" "${DEVICES_BAM}" "datapathfusion"
        done
        if ! $SKIP_REVENUE_SWEEP; then for sel in "${SELECTIVITIES[@]}"; do
            run_revenue_bench "${SF}" "uncomp_datapathfusion" "${sel}" "${DEVICES_BAM}" "datapathfusion"
        done; fi
        unload_bam
    fi

    # ---- Verification & Summary ----
    if ! $DRY_RUN; then
        if $VERIFY; then
            verify_results "${SF}"
        fi
        print_summary "${SF}"
    fi

    log "=== SF${SF} complete ==="
    log ""
done

log "============================================================"
log "=== All experiments complete ==="
log "Results : ${LOG_DIR}"
log "Timings : ${LOG_DIR}/timings.txt"
log "============================================================"
