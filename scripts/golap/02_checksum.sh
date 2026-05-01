#!/bin/bash
set -eu # エラー時に停止

# === Configuration ===
SF=1
INPUT_BASE_DIR="/export/data1/tpch/sideways/sf${SF}"
LINEITEM_INPUT_DIR=${INPUT_BASE_DIR}/lineitem
ORDERS_INPUT_DIR=${INPUT_BASE_DIR}/orders

OUTPUT_BASE_DIR="/export/data1/tpch/sideways/sf${SF}"
DUCKDB_TMP_DIR="${OUTPUT_BASE_DIR}/tmp/duckdb"

THREADS=$(nproc)
DB_FILE=":memory:"

duckdb -line "$DB_FILE" <<EOF
-- Memory configuration
SET memory_limit = '128GB';
SET threads = ${THREADS};
SET temp_directory = '${DUCKDB_TMP_DIR}';
.width 10000

CREATE OR REPLACE VIEW lineitem AS 
SELECT * FROM read_csv('${LINEITEM_INPUT_DIR}/lineitem.tbl*', delim='|', header=False,
    columns={'l_orderkey': 'INTEGER', 'l_partkey': 'INTEGER', 'l_suppkey': 'INTEGER', 'l_linenumber': 'INTEGER', 'l_quantity': 'DECIMAL(15,2)', 'l_extendedprice': 'DECIMAL(15,2)', 'l_discount': 'DECIMAL(15,2)', 'l_tax': 'DECIMAL(15,2)', 'l_returnflag': 'VARCHAR', 'l_linestatus': 'VARCHAR', 'l_shipdate': 'DATE', 'l_commitdate': 'DATE', 'l_receiptdate': 'DATE', 'l_shipinstruct': 'VARCHAR', 'l_shipmode': 'VARCHAR', 'l_comment': 'VARCHAR', 'r_name':'VARCHAR', 'c_mktsegment':'VARCHAR', 'o_orderdate':'DATE'});

CREATE OR REPLACE VIEW orders AS 
SELECT * FROM read_csv('${ORDERS_INPUT_DIR}/orders.tbl*', delim='|', header=False,
    columns={'o_orderkey': 'INTEGER', 'o_custkey': 'INTEGER', 'o_orderstatus': 'VARCHAR', 'o_totalprice': 'DECIMAL(15,2)', 'o_orderdate': 'DATE', 'o_orderpriority': 'VARCHAR', 'o_clerk': 'VARCHAR', 'o_shippriority': 'INTEGER', 'o_comment': 'VARCHAR', 'r_name':'VARCHAR', 'c_mktsegment':'VARCHAR'});

SELECT
  COUNT(*) AS LINEITEM_COUNT,
  SUM(L_ORDERKEY),
  SUM(L_PARTKEY),
  SUM(L_SUPPKEY),
  SUM(L_LINENUMBER),
  SUM(L_QUANTITY),
  SUM(L_EXTENDEDPRICE),
  SUM(L_DISCOUNT),
  SUM(L_TAX),
  -- SUM(L_RETURNFLAG),
  SUM(list_sum(
    list_transform(
      string_split_regex(L_RETURNFLAG, ''),
        x -> ascii(x)
    )
  )) AS L_RETURNFLAG_SUM,
  -- SUM(L_LINESTATUS)
  SUM(list_sum(
    list_transform(
      string_split_regex(L_LINESTATUS, ''),
        x -> ascii(x)
    )
  )) AS L_LINESTATUS_SUM,
  -- SUM(L_SHIPDATE),
  SUM(year(L_SHIPDATE) * 10000 + month(L_SHIPDATE) * 100 + day(L_SHIPDATE)) AS L_SHIPDATE_SUM,
  -- SUM(L_COMMITDATE),
  SUM(year(L_COMMITDATE) * 10000 + month(L_COMMITDATE) * 100 + day(L_COMMITDATE)) AS L_COMMITDATE_SUM,
  -- SUM(L_RECEIPTDATE),
  SUM(year(L_RECEIPTDATE) * 10000 + month(L_RECEIPTDATE) * 100 + day(L_RECEIPTDATE)) AS L_RECEIPTDATE_SUM,
  -- SUM(L_SHIPINSTRUCT),
  SUM(list_sum(
    list_transform(
      string_split_regex(RPAD(L_SHIPINSTRUCT, 25, ' '), ''), -- 1文字ずつリストに分解
        x -> ascii(x)                     -- 各文字を整数値(コードポイント)に変換
    )
  )) AS L_SHIPINSTRUCT_SUM,
  -- SUM(L_SHIPMODE),
  SUM(list_sum(
    list_transform(
      string_split_regex(RPAD(L_SHIPMODE, 10, ' '), ''), -- 1文字ずつリストに分解
        x -> ascii(x)                     -- 各文字を整数値(コードポイント)に変換
    )
  )) AS L_SHIPMODE_SUM,
  SUM(list_sum(
    list_transform(
      string_split_regex(L_COMMENT, ''), -- 1文字ずつリストに分解
        x -> ascii(x)                     -- 各文字を整数値(コードポイント)に変換
    )
  )) AS L_COMMENT_SUM
FROM
  LINEITEM;

SELECT
  COUNT(*) AS ORDERS_COUNT,
  SUM(O_ORDERKEY),
  SUM(O_CUSTKEY),
  -- SUM(O_ORDERSTATUS) - CHAR(1)
  SUM(list_sum(
    list_transform(
      string_split_regex(O_ORDERSTATUS, ''),
        x -> ascii(x)
    )
  )) AS O_ORDERSTATUS_SUM,
  SUM(O_TOTALPRICE),
  -- SUM(O_ORDERDATE)
  SUM(year(O_ORDERDATE) * 10000 + month(O_ORDERDATE) * 100 + day(O_ORDERDATE)) AS O_ORDERDATE_SUM,
  -- SUM(O_ORDERPRIORITY) - CHAR(15)
  SUM(list_sum(
    list_transform(
      string_split_regex(RPAD(O_ORDERPRIORITY, 15, ' '), ''),
        x -> ascii(x)
    )
  )) AS O_ORDERPRIORITY_SUM,
  -- SUM(O_CLERK) - CHAR(15)
  SUM(list_sum(
    list_transform(
      string_split_regex(RPAD(O_CLERK, 15, ' '), ''),
        x -> ascii(x)
    )
  )) AS O_CLERK_SUM,
  SUM(O_SHIPPRIORITY),
  -- SUM(O_COMMENT) - VARCHAR(79)
  SUM(list_sum(
    list_transform(
      string_split_regex(O_COMMENT, ''),
        x -> ascii(x)
    )
  )) AS O_COMMENT_SUM
FROM
  ORDERS;
EOF
