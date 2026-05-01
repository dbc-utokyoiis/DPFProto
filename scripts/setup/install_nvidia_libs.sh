#!/bin/bash
# install_nvidia_libs.sh — Download and install nvCOMP (host) and
#                          nvidia-mathdx (nvCOMPdx device-side) into
#                          $HOME/libs/ as expected by CMakeLists.txt.
#
# CMakeLists.txt references:
#   NVCOMP_PATH  = $HOME/libs/nvcomp               (host-side nvCOMP)
#   MATHDX_DIR   = $HOME/libs/nvidia-mathdx-${MATHDX_VER}-cuda12/nvidia/mathdx/25.12
#
# Both are closed-source NVIDIA packages distributed as redistributables.
# By running this script you accept the NVIDIA Software License Agreement
# that ships inside each archive.
#
# Versions pinned for reproducibility (these are the exact versions used in
# the DPF paper experiments).

set -euo pipefail

LIBS_DIR="${LIBS_DIR:-${HOME}/libs}"
NVCOMP_VER="5.1.0.21"
MATHDX_VER="25.12.1"

NVCOMP_ARCHIVE="nvcomp-linux-x86_64-${NVCOMP_VER}_cuda12-archive.tar.xz"
NVCOMP_URL="https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/${NVCOMP_ARCHIVE}"
NVCOMP_EXTRACTED="nvcomp-linux-x86_64-${NVCOMP_VER}_cuda12-archive"

MATHDX_ARCHIVE="nvidia-mathdx-${MATHDX_VER}-cuda12.tar.gz"
MATHDX_URL="https://developer.nvidia.com/downloads/compute/nvcompdx/redist/nvcompdx/cuda12/${MATHDX_ARCHIVE}"
MATHDX_EXTRACTED="nvidia-mathdx-${MATHDX_VER}-cuda12"

mkdir -p "${LIBS_DIR}"
cd "${LIBS_DIR}"

echo "=============================================================="
echo "  Installing NVIDIA redistributables to ${LIBS_DIR}"
echo "=============================================================="
echo ""
echo "  nvCOMP     : ${NVCOMP_VER}"
echo "  MathDx     : ${MATHDX_VER} (includes nvCOMPdx)"
echo ""
echo "  By continuing, you accept the NVIDIA Software License Agreement"
echo "  bundled with each archive."
echo ""

fetch() {
    local url="$1" out="$2"
    if [ -f "${out}" ]; then
        echo "  [cached]   ${out}"
        return
    fi
    echo "  [download] ${url}"
    curl -fL --retry 3 -o "${out}.tmp" "${url}"
    mv "${out}.tmp" "${out}"
}

# ─── nvCOMP (host API, .so + headers) ────────────────────────────
echo "[1/2] nvCOMP ${NVCOMP_VER}"
fetch "${NVCOMP_URL}" "${NVCOMP_ARCHIVE}"

if [ ! -d "${NVCOMP_EXTRACTED}" ]; then
    echo "  [extract]  ${NVCOMP_ARCHIVE}"
    tar xf "${NVCOMP_ARCHIVE}"
fi

# Project references $HOME/libs/nvcomp — create symlink
if [ -L nvcomp ] || [ -e nvcomp ]; then
    rm -rf nvcomp
fi
ln -s "${NVCOMP_EXTRACTED}" nvcomp
echo "  [symlink]  ${LIBS_DIR}/nvcomp -> ${NVCOMP_EXTRACTED}"

# ─── nvidia-mathdx (device-side nvCOMPdx) ────────────────────────
echo ""
echo "[2/2] nvidia-mathdx ${MATHDX_VER} (nvCOMPdx)"
fetch "${MATHDX_URL}" "${MATHDX_ARCHIVE}"

if [ ! -d "${MATHDX_EXTRACTED}" ]; then
    echo "  [extract]  ${MATHDX_ARCHIVE}"
    tar xf "${MATHDX_ARCHIVE}"
fi

# CMakeLists.txt expects: ${HOME}/libs/nvidia-mathdx-${MATHDX_VER}-cuda12/nvidia/mathdx/25.12
# The archive already extracts to that layout, so no symlink needed.

echo ""
echo "=============================================================="
echo "  Done."
echo ""
echo "  Installed layout:"
echo "    ${LIBS_DIR}/nvcomp/                    (host-side nvCOMP)"
echo "    ${LIBS_DIR}/${MATHDX_EXTRACTED}/       (nvCOMPdx / MathDx)"
echo ""
echo "  CMakeLists.txt picks these up via:"
echo "    NVCOMP_PATH = \$HOME/libs/nvcomp"
echo "    MATHDX_DIR  = \$HOME/libs/${MATHDX_EXTRACTED}/nvidia/mathdx/25.12"
echo ""
echo "  Runtime library path (add to LD_LIBRARY_PATH if not using rpath):"
echo "    ${LIBS_DIR}/nvcomp/lib"
echo "=============================================================="
