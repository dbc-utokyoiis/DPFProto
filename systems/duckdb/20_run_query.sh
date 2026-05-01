set -eu
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
BASE="${PARQUET_OUTPUT_BASE}"

pragma=$(cat << EOF
PRAGMA threads=${NTHREADS};
PRAGMA memory_limit='${MEMORY}';
EOF
)

createview=$(cat <<- EOF
CREATE OR REPLACE VIEW part     AS SELECT * FROM read_parquet('${BASE}/sf${SF}/part.parquet/*');
CREATE OR REPLACE VIEW supplier AS SELECT * FROM read_parquet('${BASE}/sf${SF}/supplier.parquet/*');
CREATE OR REPLACE VIEW partsupp AS SELECT * FROM read_parquet('${BASE}/sf${SF}/partsupp.parquet/*');
CREATE OR REPLACE VIEW nation   AS SELECT * FROM read_parquet('${BASE}/sf${SF}/nation.parquet/*');
CREATE OR REPLACE VIEW region   AS SELECT * FROM read_parquet('${BASE}/sf${SF}/region.parquet/*');
CREATE OR REPLACE VIEW lineitem AS SELECT * FROM read_parquet('${BASE}/sf${SF}/lineitem.parquet/*');
CREATE OR REPLACE VIEW orders   AS SELECT * FROM read_parquet('${BASE}/sf${SF}/orders.parquet/*');
CREATE OR REPLACE VIEW customer AS SELECT * FROM read_parquet('${BASE}/sf${SF}/customer.parquet/*');

EOF
)

INPUT=$1
QUERY=$(cat ${INPUT})
${DUCKDB} -cmd ".timer on" -c "${pragma}${createview}${QUERY}"
