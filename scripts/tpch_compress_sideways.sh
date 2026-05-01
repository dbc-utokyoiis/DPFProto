#!/usr/bin/env bash
# ============================================================
# tpch_compress_sideways.sh — Compress sideways data into
# tar.zstd archives for space-efficient storage.
#
# Counterpart to the decompression in tpch_run_all.sh.
#
# Usage:
#   scripts/tpch_compress_sideways.sh                # all SFs found
#   scripts/tpch_compress_sideways.sh -s 50,100      # specific SFs
#   scripts/tpch_compress_sideways.sh --delete        # remove source after compress
#   scripts/tpch_compress_sideways.sh --dry-run       # preview only
# ============================================================
set -euo pipefail

DATA_BASE="/export/data1/tpch"
SIDEWAYS_DIR="${DATA_BASE}/sideways"
COMPRESSED_DIR="${DATA_BASE}/compressed"
SCALE_FACTORS=()
DELETE_AFTER=false
DRY_RUN=false
THREADS=0  # 0 = pzstd auto-detect

usage() {
    cat <<'USAGE'
Usage: tpch_compress_sideways.sh [OPTIONS]

Compress TPC-H sideways data directories into tar.zstd archives.

Options:
  -s SF1,SF2,...    Scale factors to compress (default: auto-detect from sideways/)
  -d DATA_BASE      Data base directory (default: /export/data1/tpch)
  -j THREADS        pzstd threads (default: auto)
  --delete          Remove sideways directory after successful compression
  --dry-run         Preview commands without executing
  -h, --help        Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s) IFS=',' read -ra SCALE_FACTORS <<< "$2"; shift 2 ;;
        -d) DATA_BASE="$2"; SIDEWAYS_DIR="${DATA_BASE}/sideways"; COMPRESSED_DIR="${DATA_BASE}/compressed"; shift 2 ;;
        -j) THREADS="$2"; shift 2 ;;
        --delete)  DELETE_AFTER=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Auto-detect scale factors from existing sideways directories
if [[ ${#SCALE_FACTORS[@]} -eq 0 ]]; then
    if [[ ! -d "${SIDEWAYS_DIR}" ]]; then
        echo "ERROR: ${SIDEWAYS_DIR} does not exist" >&2
        exit 1
    fi
    for d in "${SIDEWAYS_DIR}"/sf*; do
        [[ -d "$d" ]] || continue
        sf=$(basename "$d" | sed 's/^sf//')
        SCALE_FACTORS+=("$sf")
    done
    if [[ ${#SCALE_FACTORS[@]} -eq 0 ]]; then
        echo "No sideways data directories found in ${SIDEWAYS_DIR}" >&2
        exit 1
    fi
fi

# Build pzstd command
PZSTD_CMD="pzstd"
if [[ "${THREADS}" -gt 0 ]]; then
    PZSTD_CMD="pzstd -p ${THREADS}"
fi

echo "=== tpch_compress_sideways ==="
echo "Data base     : ${DATA_BASE}"
echo "Scale factors : ${SCALE_FACTORS[*]}"
echo "Delete after  : ${DELETE_AFTER}"
echo "pzstd threads : ${THREADS:-auto}"
echo "Dry run       : ${DRY_RUN}"
echo ""

for sf in "${SCALE_FACTORS[@]}"; do
    src="${SIDEWAYS_DIR}/sf${sf}"
    dst="${COMPRESSED_DIR}/sf${sf}.tar.zstd"

    if [[ ! -d "${src}" ]]; then
        echo "[SF${sf}] SKIP: ${src} does not exist"
        continue
    fi

    if [[ -f "${dst}" ]]; then
        echo "[SF${sf}] WARNING: ${dst} already exists, skipping (delete it first to re-compress)"
        continue
    fi

    echo "[SF${sf}] Compressing ${src} → ${dst} ..."

    if $DRY_RUN; then
        echo "  [DRY-RUN] mkdir -p ${COMPRESSED_DIR}"
        echo "  [DRY-RUN] tar -h --use-compress-program='${PZSTD_CMD}' -cf ${dst} -C ${SIDEWAYS_DIR} sf${sf}"
        if $DELETE_AFTER; then
            echo "  [DRY-RUN] rm -rf ${src}"
        fi
        continue
    fi

    mkdir -p "${COMPRESSED_DIR}"

    start=$(date +%s)
    tar -h --use-compress-program="${PZSTD_CMD}" -cf "${dst}" -C "${SIDEWAYS_DIR}" "sf${sf}"
    end=$(date +%s)
    elapsed=$((end - start))

    size=$(du -sh "${dst}" | cut -f1)
    echo "[SF${sf}] Done in ${elapsed}s (${size})"

    if $DELETE_AFTER; then
        echo "[SF${sf}] Removing ${src} ..."
        rm -rf "${src}"
        echo "[SF${sf}] Removed"
    fi
done

echo ""
echo "=== Complete ==="
