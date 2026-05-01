#!/usr/bin/env bash
# =============================================================
# Spark-RAPIDS Setup — Download and configure Spark + RAPIDS plugin
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Pinned versions
SPARK_VERSION="4.1.1"
SPARK_HADOOP="hadoop3"
RAPIDS_VERSION="26.02.2"
RAPIDS_SCALA="2.13"

SPARK_DIR="${SCRIPT_DIR}/spark-${SPARK_VERSION}-bin-${SPARK_HADOOP}"
SPARK_LINK="${SCRIPT_DIR}/spark"
RAPIDS_JAR="rapids-4-spark_${RAPIDS_SCALA}-${RAPIDS_VERSION}.jar"
RAPIDS_DIR="${SCRIPT_DIR}/lib"

# Download URLs
SPARK_URL="https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-${SPARK_HADOOP}.tgz"
RAPIDS_URL="https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_${RAPIDS_SCALA}/${RAPIDS_VERSION}/${RAPIDS_JAR}"

# ---- Prerequisites ----
if ! java -version 2>&1 | grep -q "21\."; then
  echo "ERROR: Java 21 is required but not found."
  echo "  Install it with:  sudo apt-get install -y openjdk-21-jdk"
  exit 1
fi

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] spark-rapids/setup.sh"
  echo "[dry-run] Spark ${SPARK_VERSION}, RAPIDS ${RAPIDS_VERSION}"
  echo "[dry-run] Spark URL: ${SPARK_URL}"
  echo "[dry-run] RAPIDS URL: ${RAPIDS_URL}"
  exit 0
fi

# ---- Spark ----
if [[ -d "${SPARK_DIR}" ]] && "${SPARK_DIR}/bin/spark-submit" --version 2>&1 | grep -q "${SPARK_VERSION}"; then
  echo "Spark ${SPARK_VERSION} already installed at ${SPARK_DIR}"
else
  echo "Spark ${SPARK_VERSION} not found. Downloading from Apache Archive ..."
  echo "  URL: ${SPARK_URL}"
  echo "  See also: https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/"
  TARBALL="${SCRIPT_DIR}/spark-${SPARK_VERSION}-bin-${SPARK_HADOOP}.tgz"
  wget -q --show-progress -O "${TARBALL}" "${SPARK_URL}"
  echo "Extracting ..."
  tar xzf "${TARBALL}" -C "${SCRIPT_DIR}"
  rm -f "${TARBALL}"
  echo "Spark ${SPARK_VERSION} installed to ${SPARK_DIR}"
fi

# Symlink
ln -sfn "${SPARK_DIR}" "${SPARK_LINK}"

# ---- RAPIDS JAR ----
mkdir -p "${RAPIDS_DIR}"
if [[ -f "${RAPIDS_DIR}/${RAPIDS_JAR}" ]]; then
  echo "RAPIDS JAR ${RAPIDS_VERSION} already exists at ${RAPIDS_DIR}/${RAPIDS_JAR}"
else
  echo "RAPIDS JAR ${RAPIDS_VERSION} not found. Downloading from Maven Central ..."
  echo "  URL: ${RAPIDS_URL}"
  wget -q --show-progress -O "${RAPIDS_DIR}/${RAPIDS_JAR}" "${RAPIDS_URL}"
  echo "RAPIDS JAR installed to ${RAPIDS_DIR}/${RAPIDS_JAR}"
fi

# ---- Configuration ----
CONF_SRC="${SCRIPT_DIR}/conf"
CONF_DST="${SPARK_DIR}/conf"

echo "Copying configuration templates ..."
cp "${CONF_SRC}/spark-defaults.conf"  "${CONF_DST}/spark-defaults.conf"
cp "${CONF_SRC}/spark-env.sh"         "${CONF_DST}/spark-env.sh"
cp "${CONF_SRC}/getGpusResources.sh"  "${CONF_DST}/getGpusResources.sh"
cp "${CONF_SRC}/log4j2.properties"    "${CONF_DST}/log4j2.properties"
chmod +x "${CONF_DST}/spark-env.sh" "${CONF_DST}/getGpusResources.sh"

# ---- Verify ----
echo ""
echo "=== Verification ==="

ERRORS=0
ok()   { echo "  [OK] $1"; }
fail() { echo "  [NG] $1"; ERRORS=$((ERRORS + 1)); }

# Java
if java -version 2>&1 | grep -q "21\."; then
  ok "Java 21: $(java -version 2>&1 | head -1)"
else
  fail "Java 21 not found"
fi

# Spark
if "${SPARK_DIR}/bin/spark-submit" --version 2>&1 | grep -q "${SPARK_VERSION}"; then
  ok "Spark ${SPARK_VERSION}"
else
  fail "Spark ${SPARK_VERSION} not found"
fi

# RAPIDS JAR
if [[ -f "${RAPIDS_DIR}/${RAPIDS_JAR}" ]]; then
  SIZE=$(du -h "${RAPIDS_DIR}/${RAPIDS_JAR}" | cut -f1)
  ok "RAPIDS JAR: ${SIZE}"
else
  fail "RAPIDS JAR not found"
fi

# GPU
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
  ok "GPU detected"
else
  fail "No GPU detected"
fi

echo ""
if [[ $ERRORS -eq 0 ]]; then
  echo "All checks passed. Ready to run benchmarks."
else
  echo "${ERRORS} check(s) failed."
  exit 1
fi
