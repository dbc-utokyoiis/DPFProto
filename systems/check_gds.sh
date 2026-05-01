#!/bin/bash
# check_gds.sh — Verify GPUDirect Storage setup
# Returns 0 if GDS is properly configured, 1 otherwise.
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUFILE_ENV_PATH_JSON="${SCRIPT_DIR}/cufile.json"

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; WARNINGS=$((WARNINGS + 1)); }
fail() { echo -e "  ${RED}[NG]${NC} $1"; ERRORS=$((ERRORS + 1)); }

echo "=== GPUDirect Storage Check ==="

# 1. Check if GDS kernel module is loaded
if lsmod | grep -q nvidia_fs; then
  ok "nvidia_fs kernel module loaded"
else
  fail "nvidia_fs kernel module not loaded"
  echo ""
  echo -e "${RED}GPUDirect Storage is not installed or not loaded.${NC}"
  echo "Please install GDS and load the kernel module:"
  echo "  sudo modprobe nvidia_fs"
  echo ""
  echo "Installation guide:"
  echo "  https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html"
  exit 1
fi

# 2. Check if running in compatibility mode
GDS_CONFIG="${CUFILE_ENV_PATH_JSON:-/etc/cufile.json}"
if [ -f "${GDS_CONFIG}" ]; then
  ok "cufile.json found at ${GDS_CONFIG}"

  # Check for compat mode
  if grep -q '"allow_compat_mode"' "${GDS_CONFIG}"; then
    COMPAT=$(grep '"allow_compat_mode"' "${GDS_CONFIG}" | grep -oP ':\s*\K(true|false)')
    if [ "${COMPAT}" = "true" ]; then
      warn "GDS is in compatibility mode (allow_compat_mode=true)"
      echo ""
      echo -e "  ${YELLOW}Compatibility mode bypasses the GDS kernel path and uses POSIX I/O.${NC}"
      echo "  For full GDS performance, configure native mode per:"
      echo "  https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html#software-settings"
      echo ""
    else
      ok "GDS native mode (allow_compat_mode=false)"
    fi
  fi
else
  warn "cufile.json not found at ${GDS_CONFIG} — GDS may use defaults"
fi

# 3. Check for gds_stats (indicates GDS is functional)
if [ -f "/proc/driver/nvidia-fs/stats" ]; then
  ok "GDS stats available at /proc/driver/nvidia-fs/stats"
else
  warn "GDS stats not found at /proc/driver/nvidia-fs/stats"
fi

# 4. Check mdadm RAID — udev rules for RAID volumes
if command -v mdadm &>/dev/null; then
  RAID_DEVICES=$(cat /proc/mdstat 2>/dev/null | grep "^md" | awk '{print $1}' || true)
  if [ -n "${RAID_DEVICES}" ]; then
    ok "mdadm RAID detected: ${RAID_DEVICES}"

    # Check udev rules for RAID — mdadm --detail --export is required for GDS
    UDEV_GDS_RAID=0
    for udev_dir in /etc/udev/rules.d /lib/udev/rules.d; do
      if [ -d "${udev_dir}" ] && grep -rq 'mdadm --detail --export' "${udev_dir}" 2>/dev/null; then
        UDEV_GDS_RAID=1
        break
      fi
    done

    if [ "${UDEV_GDS_RAID}" -eq 1 ]; then
      ok "udev rules for RAID (mdadm --detail --export) found"
    else
      warn "No udev rules with 'mdadm --detail --export' found"
      echo ""
      echo -e "  ${YELLOW}mdadm RAID volumes require udev rules for GDS to work correctly.${NC}"
      echo "  See: https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#adding-udev-rules-for-raid-volumes"
      echo ""
    fi
  else
    ok "No mdadm RAID devices (udev rules not required)"
  fi
else
  ok "mdadm not installed (udev rules not required)"
fi

# 5. Check nvidia-smi for GPU
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  ok "GPU: ${GPU_NAME}"
else
  fail "No GPU detected"
fi

# Summary
echo ""
if [ "${ERRORS}" -gt 0 ]; then
  echo -e "${RED}${ERRORS} error(s), ${WARNINGS} warning(s). GDS is not ready.${NC}"
  exit 1
elif [ "${WARNINGS}" -gt 0 ]; then
  echo -e "${YELLOW}${WARNINGS} warning(s). GDS may not be at full performance.${NC}"
  exit 0
else
  echo -e "${GREEN}GDS is properly configured.${NC}"
  exit 0
fi
