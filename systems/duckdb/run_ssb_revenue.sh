set -eu
source common.sh

SSB_PARQUET_BASE=/export/data1/ssb/parquet/sf${SSB_SF}

queries=(sel7 sel15 sel30 sel45 sel60 sel75 sel90 sel100)
ITERATIONS=${ITERATIONS:-10}
OUTPUT_DIR=results/ssb_revenue_sf${SSB_SF}
OUTPUT_FILE=${OUTPUT_DIR}/time.txt

mkdir -p ${OUTPUT_DIR}

DUCKDB_VERSION=$(${DUCKDB} -csv -noheader -c "SELECT version()")
echo "DuckDB ${DUCKDB_VERSION}" | tee ${OUTPUT_FILE}
echo "SSB_SF=${SSB_SF}, Parquet=${SSB_PARQUET_BASE}, Threads=${NTHREADS}, Iterations=${ITERATIONS}" | tee -a ${OUTPUT_FILE}
echo "" | tee -a ${OUTPUT_FILE}

printf "%-8s %10s %10s %10s\n" "Query" "Avg(s)" "Min(s)" "Max(s)" | tee -a ${OUTPUT_FILE}
printf "%-8s %10s %10s %10s\n" "--------" "----------" "----------" "----------" | tee -a ${OUTPUT_FILE}

pragma=$(cat << EOF
PRAGMA threads=${NTHREADS};
PRAGMA memory_limit='${MEMORY}';
EOF
)

createview=$(cat <<- EOF
CREATE OR REPLACE VIEW lineorder AS SELECT * FROM read_parquet('${SSB_PARQUET_BASE}/lineorder.parquet/*');
CREATE OR REPLACE VIEW customer  AS SELECT * FROM read_parquet('${SSB_PARQUET_BASE}/customer.parquet/*');
CREATE OR REPLACE VIEW supplier  AS SELECT * FROM read_parquet('${SSB_PARQUET_BASE}/supplier.parquet/*');
CREATE OR REPLACE VIEW part      AS SELECT * FROM read_parquet('${SSB_PARQUET_BASE}/part.parquet/*');
CREATE OR REPLACE VIEW date      AS SELECT * FROM read_parquet('${SSB_PARQUET_BASE}/date.parquet/*');

EOF
)

for q in ${queries[@]}
do
  times=()
  for i in $(seq 1 ${ITERATIONS})
  do
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    QUERY=$(cat ./ssb_revenue/${q}.sql)
    result=$(${DUCKDB} -cmd ".timer on" -c "${pragma}${createview}${QUERY}" 2>&1)
    real_time=$(echo "$result" | grep 'Run Time' | grep -oP 'real \K[0-9.]+')
    times+=($real_time)
    echo "  ${q} iteration ${i}/${ITERATIONS}: ${real_time}s" >&2
  done

  read -r avg min max <<< $(printf '%s\n' "${times[@]}" | awk '
    NR==1 { min=$1; max=$1 }
    { sum+=$1; if($1<min) min=$1; if($1>max) max=$1 }
    END { printf "%.3f %.3f %.3f", sum/NR, min, max }
  ')

  printf "%-8s %10s %10s %10s\n" "$q" "$avg" "$min" "$max" | tee -a ${OUTPUT_FILE}
done

echo "" | tee -a ${OUTPUT_FILE}
echo "Iterations: ${ITERATIONS}" | tee -a ${OUTPUT_FILE}
