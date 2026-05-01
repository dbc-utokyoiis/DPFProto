#!/usr/bin/env bash
# =============================================================
# SSB Benchmark Runner for Spark-RAPIDS
# Usage:
#   ./run_ssb_benchmark.sh --mode gpu   # GPU (RAPIDS)
#   ./run_ssb_benchmark.sh --mode cpu   # CPU only
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SYSTEMS_DIR}/common.sh"

export SPARK_HOME="${SCRIPT_DIR}/spark"
export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
export PATH="${SPARK_HOME}/bin:${PATH}"

MODE_VAL="gpu"
DRY_RUN=0
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --mode) :;;
    gpu|cpu) MODE_VAL="$arg" ;;
    --dry-run) DRY_RUN=1 ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

export SF SSB_PARQUET_BASE
export ITERATIONS=${ITERATIONS:-10}
export LOG_BASE="${SYSTEMS_DIR}/logs/spark-rapids"
export ANSWERS_BASE="${SYSTEMS_DIR}/../answers"

# Spark local mode: JVM heap (default 64g, override with SPARK_DRIVER_MEMORY)
SPARK_MEMORY="${SPARK_DRIVER_MEMORY:-384g}"


if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] spark-rapids/run_ssb_benchmark.sh"
  echo "[dry-run] MODE=${MODE_VAL}, SF=${SF}, Iterations=${ITERATIONS}"
  echo "[dry-run] Data: ${SSB_PARQUET_BASE}"
  echo "[dry-run] Log output: ${LOG_BASE}/"
  exit 0
fi

# Check GDS setup (GPU mode benefits from GPUDirect Storage)
if [[ "$MODE_VAL" == "gpu" ]]; then
  bash "${SYSTEMS_DIR}/check_gds.sh" || exit 1
fi

echo "=============================================="
echo " SSB Benchmark — Spark-RAPIDS"
echo " Mode: ${MODE_VAL^^}, SF: ${SF}"
echo "=============================================="

RAPIDS_JAR="${SCRIPT_DIR}/lib/rapids-4-spark_2.13-26.02.2.jar"

if [[ "$MODE_VAL" == "gpu" ]]; then
    spark-submit \
        --master "local[${NTHREADS}]" \
        --driver-memory "${SPARK_MEMORY}" \
        --jars "${RAPIDS_JAR}" \
        --conf spark.local.dir="${TMPDIR}" \
        --conf spark.plugins=com.nvidia.spark.SQLPlugin \
        --conf spark.rapids.memory.pinnedPool.size=16G \
        --conf spark.rapids.memory.gpu.pool=ASYNC \
        --conf spark.rapids.sql.concurrentGpuTasks=2 \
        --conf spark.rapids.sql.explain=${SPARK_EXPLAIN:-NONE} \
        --conf spark.executor.heartbeatInterval=60s \
        --conf spark.network.timeout=300s \
        --conf spark.sql.files.maxPartitionBytes=512m \
        --conf spark.sql.parquet.aggregatePushdown=true \
        --conf spark.rapids.shuffle.mode=MULTITHREADED \
        --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
        --conf spark.kryo.registrator=com.nvidia.spark.rapids.GpuKryoRegistrator \
        "${SCRIPT_DIR}/ssb_benchmark.py" --mode gpu "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    spark-submit \
        --master "local[${NTHREADS}]" \
        --driver-memory "${SPARK_MEMORY}" \
        --conf spark.local.dir="${TMPDIR}" \
        --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
        --conf spark.sql.files.maxPartitionBytes=512m \
        --conf spark.sql.parquet.aggregatePushdown=true \
        "${SCRIPT_DIR}/ssb_benchmark.py" --mode cpu "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
