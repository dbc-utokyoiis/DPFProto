set -eu
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SSB_BASE="${SSB_PARQUET_OUTPUT_BASE}"

pragma=$(cat << EOF
PRAGMA threads=${NTHREADS};
PRAGMA memory_limit='${MEMORY}';
EOF
)

createview=$(cat <<- EOF
CREATE OR REPLACE VIEW lineorder AS SELECT * FROM read_parquet('${SSB_BASE}/lineorder.parquet/*');
CREATE OR REPLACE VIEW customer  AS SELECT * FROM read_parquet('${SSB_BASE}/customer.parquet/*');
CREATE OR REPLACE VIEW supplier  AS SELECT * FROM read_parquet('${SSB_BASE}/supplier.parquet/*');
CREATE OR REPLACE VIEW part      AS SELECT * FROM read_parquet('${SSB_BASE}/part.parquet/*');
CREATE OR REPLACE VIEW date      AS SELECT * FROM read_parquet('${SSB_BASE}/date.parquet/*');

EOF
)

INPUT=$1
QUERY=$(cat ${INPUT})
${DUCKDB} -cmd ".timer on" -c "${pragma}${createview}${QUERY}"
