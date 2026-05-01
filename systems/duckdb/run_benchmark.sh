#!/bin/bash
# duckdb/run_benchmark.sh — Run TPC-H / SSB benchmarks on DuckDB over Parquet
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [ ! -x "${DUCKDB}" ]; then
  echo "ERROR: DuckDB not found at ${DUCKDB}. Run 'bash duckdb/setup.sh' first." >&2
  exit 1
fi

BENCHMARK=${BENCHMARK:-all}
ITERATIONS=${ITERATIONS:-10}
LOG_BASE="${SYSTEMS_DIR}/logs/duckdb"
ANSWERS_BASE="${SYSTEMS_DIR}/../answers"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
  esac
done

DUCKDB_VERSION=$(${DUCKDB} -csv -noheader -c "SELECT version()")

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

# ---- Helper: create Parquet views ----

tpch_views() {
  local base="${PARQUET_OUTPUT_BASE}"
  cat <<EOF
CREATE OR REPLACE VIEW part     AS SELECT * FROM read_parquet('${base}/sf${SF}/part.parquet/*');
CREATE OR REPLACE VIEW supplier AS SELECT * FROM read_parquet('${base}/sf${SF}/supplier.parquet/*');
CREATE OR REPLACE VIEW partsupp AS SELECT * FROM read_parquet('${base}/sf${SF}/partsupp.parquet/*');
CREATE OR REPLACE VIEW nation   AS SELECT * FROM read_parquet('${base}/sf${SF}/nation.parquet/*');
CREATE OR REPLACE VIEW region   AS SELECT * FROM read_parquet('${base}/sf${SF}/region.parquet/*');
CREATE OR REPLACE VIEW lineitem AS SELECT * FROM read_parquet('${base}/sf${SF}/lineitem.parquet/*');
CREATE OR REPLACE VIEW orders   AS SELECT * FROM read_parquet('${base}/sf${SF}/orders.parquet/*');
CREATE OR REPLACE VIEW customer AS SELECT * FROM read_parquet('${base}/sf${SF}/customer.parquet/*');
EOF
}

ssb_views() {
  local base="${SSB_PARQUET_OUTPUT_BASE}"
  cat <<EOF
CREATE OR REPLACE VIEW lineorder AS SELECT * FROM read_parquet('${base}/lineorder.parquet/*');
CREATE OR REPLACE VIEW customer  AS SELECT * FROM read_parquet('${base}/customer.parquet/*');
CREATE OR REPLACE VIEW supplier  AS SELECT * FROM read_parquet('${base}/supplier.parquet/*');
CREATE OR REPLACE VIEW part      AS SELECT * FROM read_parquet('${base}/part.parquet/*');
CREATE OR REPLACE VIEW date      AS SELECT * FROM read_parquet('${base}/date.parquet/*');
EOF
}

# ---- Helper: run one query, return wall-clock seconds ----

run_query() {
  local pragma="PRAGMA threads=${NTHREADS}; PRAGMA memory_limit='${MEMORY}'; PRAGMA temp_directory='${DUCKDB_TEMP_DIR}';"
  local views="$1"
  local sql="$2"
  local result_file="$3"

  local start end elapsed
  start=$(date +%s.%N)
  local err_file="${result_file}.err"
  ${DUCKDB} -list -separator '|' -noheader -c "${pragma}${views}${sql}" > "${result_file}" 2>"${err_file}" || {
    echo "ERROR: DuckDB query failed:" >&2
    cat "${err_file}" >&2
    rm -f "${err_file}"
    return 1
  }
  rm -f "${err_file}"
  end=$(date +%s.%N)
  elapsed=$(printf '%.6f' "$(echo "$end - $start" | bc)")
  echo "${elapsed}"
}

# ---- Helper: verify result file against reference answer ----

verify_query() {
  local query_name="$1"
  local answer_file="$2"
  local result_file="$3"

  if [ ! -f "${answer_file}" ]; then
    qlog "  ${query_name}: NO_REF (${answer_file} not found)"
    return 0
  fi

  if [ ! -f "${result_file}" ]; then
    qlog "  ${query_name}: NO_RESULT (${result_file} not found)"
    return 1
  fi

  # Compare as sets (sorted) to ignore tie-breaking order differences
  local tmp_got tmp_ref
  tmp_got=$(mktemp)
  tmp_ref=$(mktemp)
  trap "rm -f ${tmp_got} ${tmp_ref}" RETURN

  sort "${result_file}" > "${tmp_got}"
  # Reference CSV has a header line; skip it and strip quotes for comparison
  tail -n +2 "${answer_file}" | sed 's/"//g' | sort > "${tmp_ref}"

  if diff -q "${tmp_got}" "${tmp_ref}" > /dev/null 2>&1; then
    qlog "  ${query_name}: MATCH"
    return 0
  else
    qlog "  ${query_name}: FAIL"
    qlog "    expected: $(head -3 "${tmp_ref}")"
    qlog "    got:      $(head -3 "${tmp_got}")"
    return 1
  fi
}

# ---- Helper: compute stats ----

compute_stats() {
  # stdin: one number per line
  awk '
    { x[NR]=$1; sum+=$1 }
    END {
      n=NR; avg=sum/n; min=x[1]; max=x[1]
      for(i=1;i<=n;i++) {
        if(x[i]<min) min=x[i]
        if(x[i]>max) max=x[i]
        d=x[i]-avg; ss+=d*d
      }
      stddev=sqrt(ss/n)
      printf "%.3f %.3f %.3f %.3f\n", avg, min, max, stddev
    }
  '
}

# ---- Run a benchmark suite ----

run_suite() {
  local suite="$1"        # tpch or ssb
  local views="$2"
  local query_dir="$3"
  local answer_dir="$4"
  local sf_label="$5"
  shift 5
  local queries=("$@")

  local log_dir="${LOG_BASE}/${suite}/sf${sf_label}"
  mkdir -p "${log_dir}"
  local time_file="${log_dir}/time.txt"

  local result_dir="${log_dir}/results"
  mkdir -p "${result_dir}"

  log "============================================================"
  log "  DuckDB ${DUCKDB_VERSION} — ${suite^^} SF${sf_label}"
  log "============================================================"
  log "Layout=${LAYOUT}, Threads=${NTHREADS}, Iterations=${ITERATIONS}"
  log ""

  # Collect all timing data: query_times[q]="t1 t2 t3 ..."
  declare -A query_times

  # qlog: log to stderr and append to per-query log file
  local _current_qlog=""
  qlog() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}" >&2
    [ -n "${_current_qlog}" ] && echo "${msg}" >> "${_current_qlog}"
  }

  for q in "${queries[@]}"; do
    local sql
    sql=$(cat "${query_dir}/${q}.sql")
    local times=()
    local q_result_dir="${result_dir}/${q}"
    mkdir -p "${q_result_dir}"
    _current_qlog="${result_dir}/${q}.txt"
    : > "${_current_qlog}"

    qlog "=== ${q} ==="
    for i in $(seq 1 ${ITERATIONS}); do
      echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
      local result_file="${q_result_dir}/run${i}.txt"
      local t
      t=$(run_query "${views}" "${sql}" "${result_file}") || exit 1
      times+=("${t}")
      local t_ms=$(echo "${t} * 1000" | bc | xargs printf '%.1f')
      qlog "  iteration ${i}/${ITERATIONS}: ${t_ms} ms"
    done

    read -r avg min max stddev <<< $(printf '%s\n' "${times[@]}" | compute_stats)
    local avg_ms=$(echo "${avg} * 1000" | bc | xargs printf '%.1f')
    local min_ms=$(echo "${min} * 1000" | bc | xargs printf '%.1f')
    local max_ms=$(echo "${max} * 1000" | bc | xargs printf '%.1f')
    local std_ms=$(echo "${stddev} * 1000" | bc | xargs printf '%.1f')
    qlog "  avg=${avg_ms} ms  min=${min_ms} ms  max=${max_ms} ms  std=${std_ms} ms"

    query_times["${q}"]="${times[*]}"
  done

  # Verify last iteration results against reference answers
  log ""
  log "--- Verifying answers ---"
  local verify_fail=0
  for q in "${queries[@]}"; do
    _current_qlog="${result_dir}/${q}.txt"
    verify_query "${q}" "${answer_dir}/${q}.csv" "${result_dir}/${q}/run${ITERATIONS}.txt" || verify_fail=1
  done
  _current_qlog=""
  if [ "${verify_fail}" -eq 1 ]; then
    log "WARNING: Some answers did not match reference."
  else
    log "All answers matched."
  fi

  # Print summary table
  log ""
  log "=== Summary ==="

  {
    echo "=== DuckDB ${DUCKDB_VERSION} — ${suite^^} SF${sf_label} ==="
    echo "Layout=${LAYOUT}, Threads=${NTHREADS}, Iterations=${ITERATIONS}"
    echo ""

    printf "%-8s" "Query"
    for i in $(seq 1 ${ITERATIONS}); do
      printf " %10s" "Run${i}"
    done
    printf " %10s %10s %10s %10s\n" "Avg(ms)" "Min(ms)" "Max(ms)" "Std(ms)"

    printf "%-8s" "--------"
    for i in $(seq 1 ${ITERATIONS}); do
      printf " %10s" "----------"
    done
    printf " %10s %10s %10s %10s\n" "----------" "----------" "----------" "----------"

    for q in "${queries[@]}"; do
      local times_str="${query_times[${q}]}"
      read -ra times <<< "${times_str}"
      read -r avg min max stddev <<< $(printf '%s\n' "${times[@]}" | compute_stats)

      printf "%-8s" "${q}"
      for t in "${times[@]}"; do
        printf " %10.1f" "$(echo "${t} * 1000" | bc)"
      done
      printf " %10.1f %10.1f %10.1f %10.1f\n" \
        "$(echo "${avg} * 1000" | bc)" \
        "$(echo "${min} * 1000" | bc)" \
        "$(echo "${max} * 1000" | bc)" \
        "$(echo "${stddev} * 1000" | bc)"
    done
  } | tee "${time_file}"

  log ""
  log "Results: ${time_file}"
}

# ---- Main ----

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "[dry-run] duckdb/run_benchmark.sh"
  echo "[dry-run] DuckDB ${DUCKDB_VERSION}, BENCHMARK=${BENCHMARK}"
  echo "[dry-run] SF=${SF}, SF=${SF}, Layout=${LAYOUT}, Threads=${NTHREADS}, Iterations=${ITERATIONS}"
  echo "[dry-run] Log output: ${LOG_BASE}/"
  echo "[dry-run] Answers: ${ANSWERS_BASE}/"
  exit 0
fi

TPCH_QUERIES=(q1 q3 q5 q6 q13 q16)
SSB_QUERIES=(q11 q12 q13 q21 q22 q23 q31 q32 q33 q34 q41 q42 q43)

# Check data directories
run_tpch=0; run_ssb=0
case "${BENCHMARK}" in
  tpch) run_tpch=1 ;;
  ssb)  run_ssb=1 ;;
  all)  run_tpch=1; run_ssb=1 ;;
esac

check_data_dir() {
  local dir="$1" suite="$2"
  if [ ! -d "${dir}" ]; then
    echo "ERROR: ${suite} data directory not found: ${dir}" >&2
    echo "Run 'BENCHMARK=${suite,,} SF=${SF} bash duckdb/load.sh' first to generate Parquet files." >&2
    exit 1
  fi
  # Check for empty parquet subdirectories
  for subdir in "${dir}"/*.parquet; do
    if [ -d "${subdir}" ] && [ -z "$(ls -A "${subdir}" 2>/dev/null)" ]; then
      echo "ERROR: ${suite} data has empty parquet directory: ${subdir}" >&2
      echo "Run 'BENCHMARK=${suite,,} SF=${SF} bash duckdb/load.sh' to regenerate Parquet files." >&2
      exit 1
    fi
  done
}

if [ "${run_tpch}" -eq 1 ]; then
  check_data_dir "${PARQUET_OUTPUT_BASE}/sf${SF}" "TPC-H"
fi
if [ "${run_ssb}" -eq 1 ]; then
  check_data_dir "${SSB_PARQUET_OUTPUT_BASE}" "SSB"
fi

case "${BENCHMARK}" in
  tpch)
    run_suite "tpch" "$(tpch_views)" "${SCRIPT_DIR}/queries" "${ANSWERS_BASE}/tpch/sf${SF}" "${SF}" "${TPCH_QUERIES[@]}"
    ;;
  ssb)
    run_suite "ssb" "$(ssb_views)" "${SCRIPT_DIR}/ssb_queries" "${ANSWERS_BASE}/ssb/sf${SF}" "${SF}" "${SSB_QUERIES[@]}"
    ;;
  all)
    run_suite "tpch" "$(tpch_views)" "${SCRIPT_DIR}/queries" "${ANSWERS_BASE}/tpch/sf${SF}" "${SF}" "${TPCH_QUERIES[@]}"
    run_suite "ssb" "$(ssb_views)" "${SCRIPT_DIR}/ssb_queries" "${ANSWERS_BASE}/ssb/sf${SF}" "${SF}" "${SSB_QUERIES[@]}"
    ;;
  *)
    echo "Unknown BENCHMARK: ${BENCHMARK} (use tpch, ssb, or all)" >&2
    exit 1
    ;;
esac
