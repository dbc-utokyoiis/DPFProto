source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

# Cached DuckDB binary
DUCKDB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bin/duckdb"

# Aliases for existing duckdb scripts
tbls=("${TPCH_TABLES[@]}")
ssb_tbls=("${SSB_TABLES[@]}")
INPUT_BASE="${TPCH_INPUT_BASE}"
PARQUET_OUTPUT_BASE="${TPCH_PARQUET_BASE}"
SSB_PARQUET_OUTPUT_BASE="${SSB_PARQUET_BASE}"
