#!/usr/bin/env bash
# scripts/common.sh — Shared environment for DPF artifact scripts.
#
# Reviewers: edit the variables in this file to match your hardware
# layout if it differs from the reference machine (see README §1).
# tpch_run_all.sh, ssb_run_all.sh, and reproduce_*.sh source this file,
# so changes here propagate to all experiments without further edits.

# ─── Repository layout ─────────────────────────────────────────────
# Set COMMON_SH_DIR to the directory of this file, REPO_ROOT to the
# project root. Works both when sourced and when chained from other
# scripts in scripts/.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    COMMON_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    COMMON_SH_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
REPO_ROOT="$(cd "${COMMON_SH_DIR}/.." && pwd)"

# ─── NVMe devices ──────────────────────────────────────────────────
# Four data-plane NVMe namespaces. Use scripts/setup/probe_nvme.sh
# to identify the correct /dev/nvmeXn1 on your machine.
NVME_DEVICES_KERNEL=(
    /dev/nvme0n1
    /dev/nvme1n1
    /dev/nvme2n1
    /dev/nvme3n1
)

# Derived partition paths. Each entry MUST be a namespace like
# /dev/nvmeXn1 (no trailing p1/p2). Partitions are appended below:
#   p1 → raw block device used by BaM and the GIDP loader (required)
#   p2 → member of the mdadm RAID (XFS backing store)
# See README §1.4 for the required partition layout.
NVME_P1=(); NVME_P2=(); NVME_PCI_BDF=()
for _d in "${NVME_DEVICES_KERNEL[@]}"; do
    if [[ ! "${_d}" =~ ^/dev/nvme[0-9]+n[0-9]+$ ]]; then
        echo "[common.sh] ERROR: NVME_DEVICES_KERNEL entry '${_d}' must be a" >&2
        echo "  namespace path like /dev/nvme0n1 (no trailing p1/p2)." >&2
        return 1 2>/dev/null || exit 1
    fi
    NVME_P1+=("${_d}p1")
    NVME_P2+=("${_d}p2")

    # Derive PCIe BDF (e.g. "0000:c0:00.0") from /sys/class/nvme/<ctrl>/device.
    # Used by load_bam/unload_bam to unbind/rebind the kernel NVMe driver.
    _ctrl="${_d##*/}"           # /dev/nvme0n1 → nvme0n1
    _ctrl="${_ctrl%n[0-9]*}"    # nvme0n1      → nvme0
    _bdf_link="/sys/class/nvme/${_ctrl}/device"
    if [ -L "${_bdf_link}" ]; then
        NVME_PCI_BDF+=("$(basename "$(readlink -f "${_bdf_link}")")")
    else
        echo "[common.sh] WARNING: cannot resolve PCIe BDF for ${_d}" \
             "(missing ${_bdf_link})." >&2
        NVME_PCI_BDF+=("")
    fi
done
unset _d _ctrl _bdf_link

# Comma-separated forms (used by tpchdb/ssbdb -Z and loaders)
DEVICES_NVME="$(IFS=,; echo "${NVME_P1[*]}")"
DEVICES_BAM="/dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3"

# ─── Filesystem / RAID ────────────────────────────────────────────
MDADM_DEV=/dev/md0
MOUNT_POINT=/export/data1

# ─── Data roots ────────────────────────────────────────────────────
TPCH_ROOT="${MOUNT_POINT}/tpch"
SSB_ROOT="${MOUNT_POINT}/ssb"

# ─── External dbgen kits ──────────────────────────────────────────
# TPC-H dbgen is distributed by the TPC under a separate EULA and is
# NOT bundled with this artifact. Download the kit from
# https://www.tpc.org/tpch/, build it (cd <kit>/dbgen && make), and
# point TPCH_DBGEN_DIR at the built dbgen directory. Required by
# scripts/tpch/run_dbgen.sh.
#export TPCH_DBGEN_DIR=/path/to/tpch_kit/dbgen

# ─── BaM kernel module ─────────────────────────────────────────────
BAM_MODULE="${REPO_ROOT}/bam/build/module/libnvm.ko"

# ─── GDS / cuFile configuration ────────────────────────────────────
# DPF ships a cuFile JSON with tuned batch-size / thread-pool / pinned
# memory values (see config/cufile.json). Export the env var so that
# tpchdb / ssbdb (and anything linked against libcufile) pick it up
# instead of falling back to /etc/cufile.json.
export CUFILE_ENV_PATH_JSON="${REPO_ROOT}/config/cufile.json"

# ─── Build outputs ────────────────────────────────────────────────
BUILD_DIR="${REPO_ROOT}/build"
TPCHDB="${BUILD_DIR}/tpchdb"
TPCHLOADER="${BUILD_DIR}/tpchloader"
SSBDB="${BUILD_DIR}/ssbdb"
SSBLOADER="${BUILD_DIR}/ssbloader"

# ─── Default benchmark parameters ─────────────────────────────────
DEFAULT_TRIALS=10
DEFAULT_THREADS=32
DEFAULT_TIMEOUT=15   # per-trial seconds

# ═══════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════

# common_log <msg...>
common_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# common_die <msg...>  — log error and exit 1
common_die() {
    echo "[ERROR] $*" >&2
    exit 1
}

# Switch to "BaM mode" (umount XFS, stop mdadm, insmod libnvm).
# Requires sudo and an unmounted /export/data1 (scripts unmount automatically).
load_bam() {
    sudo umount "${MOUNT_POINT}" 2>/dev/null || true
    sudo mdadm --stop "${MDADM_DEV}" 2>/dev/null || true
    if [ ! -f "${BAM_MODULE}" ]; then
        common_die "BaM module not built: ${BAM_MODULE}. Build it by following bam/README.md (see README §4.3)."
    fi
    sudo insmod "${BAM_MODULE}"
    sudo chown "$(whoami):$(whoami)" /dev/libnvm* 2>/dev/null || true
}

# Switch back to "XFS mode" (rmmod libnvm, reassemble mdadm, mount XFS).
unload_bam() {
    sudo rmmod libnvm 2>/dev/null || true
    sudo mdadm --stop "${MDADM_DEV}" 2>/dev/null || true
    sudo mdadm --assemble "${MDADM_DEV}" "${NVME_P2[@]}"
    sudo mount -t xfs "${MDADM_DEV}" "${MOUNT_POINT}/"
    sudo chown "$(whoami):$(whoami)" "${NVME_P1[@]}" 2>/dev/null || true
}

# Verify build artifacts exist. Called by reproduce scripts before running.
check_build() {
    local miss=0
    for bin in "${TPCHDB}" "${SSBDB}" "${TPCHLOADER}" "${SSBLOADER}"; do
        if [ ! -x "${bin}" ]; then
            echo "[MISSING] ${bin}" >&2
            miss=$((miss+1))
        fi
    done
    if [ ! -f "${BAM_MODULE}" ]; then
        echo "[MISSING] ${BAM_MODULE}" >&2
        miss=$((miss+1))
    fi
    if [ "$miss" -gt 0 ]; then
        common_die "Missing build outputs. Run: cd build && cmake .. && make -j, and build BaM per bam/README.md (see README §4.3)."
    fi
}

# Print the effective environment (for logging / debugging).
print_env() {
    cat <<EOF
──────────────────────────────────────────────────────────
  REPO_ROOT              : ${REPO_ROOT}
  NVMe devices           : ${NVME_DEVICES_KERNEL[*]}
  DEVICES_NVME           : ${DEVICES_NVME}
  DEVICES_BAM            : ${DEVICES_BAM}
  MDADM_DEV              : ${MDADM_DEV}
  MOUNT_POINT            : ${MOUNT_POINT}
  TPCH_ROOT              : ${TPCH_ROOT}
  SSB_ROOT               : ${SSB_ROOT}
  BAM_MODULE             : ${BAM_MODULE}
  CUFILE_ENV_PATH_JSON   : ${CUFILE_ENV_PATH_JSON}
──────────────────────────────────────────────────────────
EOF
}
