#!/usr/bin/env bash
# probe_nvme.sh — Identify NVMe data devices and print their PCI topology.
#
# Prints one row per NVMe device:
#   <node>  <pci-bdf>  <model>  <size>  <kernel-drive>  <note>
#
# This helps reviewers:
#   (a) identify which /dev/nvmeXn1 correspond to which PCI slot,
#   (b) confirm that the OS root is NOT among the target devices,
#   (c) map NVMe PCIe addresses onto GPU siblings when evaluating
#       GPUDirect Storage / BaM placement.
#
# Usage:
#   scripts/setup/probe_nvme.sh                # list all NVMe devices
#   scripts/setup/probe_nvme.sh -m <substring> # filter by model string
#   scripts/setup/probe_nvme.sh -m MZQL21T9    # e.g. reference data drives

set -euo pipefail

MODEL_FILTER=""
while getopts "m:h" opt; do
    case "$opt" in
        m) MODEL_FILTER="$OPTARG" ;;
        h|*)
            cat <<EOF
Usage: $(basename "$0") [-m <model-substring>]
  -m <str>   only list devices whose model string contains <str>
EOF
            exit 0
            ;;
    esac
done

# ─── Root mount device (to warn against) ──────────────────────────
root_src="$(findmnt -n -o SOURCE / 2>/dev/null || true)"
# resolve /dev/nvmeXnY back to /dev/nvmeXn1
root_dev="$(lsblk -no PKNAME "$root_src" 2>/dev/null || true)"
root_nvme="${root_dev%%p*}"    # strip trailing partition digits

# ─── Enumerate NVMe devices ──────────────────────────────────────
# Use nvme-cli if available, else fall back to /sys
if command -v nvme >/dev/null 2>&1; then
    nvme_list="$(sudo nvme list -o json 2>/dev/null || true)"
fi

printf "%-15s  %-12s  %-42s  %-12s  %-s\n" \
    "NODE" "PCI BDF" "MODEL" "SIZE" "NOTE"
printf "%-15s  %-12s  %-42s  %-12s  %-s\n" \
    "---------------" "------------" "------------------------------------------" "------------" "----"

for sys_path in /sys/class/nvme/nvme*; do
    [ -e "$sys_path" ] || continue
    ctrl="$(basename "$sys_path")"
    node="/dev/${ctrl}n1"
    [ -e "$node" ] || continue

    # PCI BDF: /sys/class/nvme/nvme0 → ../../devices/pci*/0000:XX:YY.Z/nvme/nvme0
    pci_bdf="$(readlink -f "$sys_path" | grep -oP '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]' | tail -1 || true)"

    model="$(cat "$sys_path/model" 2>/dev/null | tr -s ' ' | sed 's/^ *//;s/ *$//' || echo "")"
    # Size via sysfs (no sudo needed): size field = number of 512-byte sectors
    sectors="$(cat "/sys/class/block/${ctrl}n1/size" 2>/dev/null || echo 0)"
    size_bytes=$(( sectors * 512 ))
    if [ "$size_bytes" -gt 0 ]; then
        size_h="$(numfmt --to=iec --suffix=B "$size_bytes")"
    else
        size_h="?"
    fi

    note=""
    if [ "$ctrl" = "$root_nvme" ] || [ "$node" = "$root_src" ]; then
        note="← OS ROOT (do NOT touch)"
    fi

    # Apply model filter if given
    if [ -n "$MODEL_FILTER" ] && ! [[ "$model" == *"$MODEL_FILTER"* ]]; then
        continue
    fi

    printf "%-15s  %-12s  %-42s  %-12s  %s\n" \
        "$node" "${pci_bdf:-?}" "$model" "$size_h" "$note"
done

echo ""
echo "Tip: the artifact uses four data devices (e.g. /dev/nvme{0,1,2,3}n1)."
echo "     Edit scripts/tpch_run_all.sh / scripts/ssb_run_all.sh DEVICES_*"
echo "     variables if your device nodes differ."
