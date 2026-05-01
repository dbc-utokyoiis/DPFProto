set -eu
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
queries=(1 3 5 6 13 16)
ITERATIONS=${ITERATIONS:-10}
OUTPUT_DIR=results/tpch_sf${SF}_${LAYOUT}
OUTPUT_FILE=${OUTPUT_DIR}/time.txt

mkdir -p ${OUTPUT_DIR}

DUCKDB_VERSION=$(${DUCKDB} -csv -noheader -c "SELECT version()")
echo "DuckDB ${DUCKDB_VERSION}" | tee ${OUTPUT_FILE}
echo "SF=${SF}, Layout=${LAYOUT}, Threads=${NTHREADS}, Iterations=${ITERATIONS}" | tee -a ${OUTPUT_FILE}
echo "" | tee -a ${OUTPUT_FILE}

printf "%-8s %10s %10s %10s\n" "Query" "Avg(s)" "Min(s)" "Max(s)" | tee -a ${OUTPUT_FILE}
printf "%-8s %10s %10s %10s\n" "--------" "----------" "----------" "----------" | tee -a ${OUTPUT_FILE}

for q in ${queries[@]}
do
  times=()
  for i in $(seq 1 ${ITERATIONS})
  do
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    # クエリ実行し出力をキャプチャ
    result=$(bash ./20_run_query.sh ./queries/q${q}.sql 2>&1)
    # "Run Time (s): real X.XXX" から real の値を抽出
    real_time=$(echo "$result" | grep 'Run Time' | tail -1 | grep -oP 'real \K[0-9.]+')
    times+=($real_time)
    echo "  Q${q} iteration ${i}/${ITERATIONS}: ${real_time}s" >&2
  done

  # avg, min, max を awk で算出
  read -r avg min max <<< $(printf '%s\n' "${times[@]}" | awk '
    NR==1 { min=$1; max=$1 }
    { sum+=$1; if($1<min) min=$1; if($1>max) max=$1 }
    END { printf "%.3f %.3f %.3f", sum/NR, min, max }
  ')

  printf "Q%-7s %10s %10s %10s\n" "$q" "$avg" "$min" "$max" | tee -a ${OUTPUT_FILE}
done

echo "" | tee -a ${OUTPUT_FILE}
echo "Iterations: ${ITERATIONS}" | tee -a ${OUTPUT_FILE}
