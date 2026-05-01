#!/bin/bash
# polars/setup.sh — Setup Polars GPU benchmark environment (kazamatsuri)
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/venv"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
  esac
done

# ---- Prerequisites ----
if ! python3 -m venv --help &>/dev/null; then
  echo "ERROR: python3-venv is required but not found."
  echo "  Install it with:  sudo apt-get install -y python3-venv"
  exit 1
fi

POLARS_VERSION="1.35.2"

if [ -d "${VENV}" ]; then
  CURRENT=$(source "${VENV}/bin/activate" && python3 -c "import polars; print(polars.__version__)" 2>/dev/null || echo "unknown")
  if [ "${CURRENT}" = "${POLARS_VERSION}" ]; then
    echo "Polars ${POLARS_VERSION} venv already exists at ${VENV}"
    exit 0
  fi
  echo "Polars ${CURRENT} found, expected ${POLARS_VERSION}"
fi

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "[dry-run] Would create venv at ${VENV} and install polars[gpu]==${POLARS_VERSION}"
  exit 0
fi

echo "Creating venv at ${VENV}..."
python3 -m venv "${VENV}"
source "${VENV}/bin/activate"

pip install --upgrade pip
pip install "polars[gpu]==${POLARS_VERSION}" --extra-index-url=https://pypi.nvidia.com
pip install "kvikio-cu12==26.2.0" --extra-index-url=https://pypi.nvidia.com
pip install "pyarrow==23.0.1"

echo "Polars ${POLARS_VERSION} installed to ${VENV}"
