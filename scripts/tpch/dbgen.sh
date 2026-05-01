#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Option flags (table selection)
###############################################################################
declare -a tables=()
all=false

while getopts ":AcLnOPSrs" opt; do
  case "$opt" in
    A) all=true ;;
    c) 
      tables+=(customer)
      opts+=(c)
      ;;
    L)
      tables+=(lineitem)
      opts+=(L)
      ;;
    n)
      tables+=(nation)
      opts+=(n)
      ;;
    O)
      tables+=(orders)
      opts+=(O)
      ;;
    P)
      tables+=(part)
      opts+=(P)
      ;;
    S)
      tables+=(partsupp)
      opts+=(S)
      ;;
    r)
      tables+=(region)
      opts+=(r)
      ;;
    s)
      tables+=(supplier)
      opts+=(s)
      ;;
    *) usage ;;
  esac
done
shift $((OPTIND-1))   # ← everything before this point were options

if $all; then
  tables=("customer" "lineitem" "nation" "orders" "part" "partsupp" "region" "supplier")
  opts=("c" "L" "n" "O" "P" "S" "r" "s")
fi


###############################################################################
# Positional parameters (legacy 4 args)
###############################################################################
if [ $# -ne 4 ]; then
  echo "Usage: $(basename "$0") [table-options] <dbgendir> <scale_factor> <num_threads> <destdir>"
  exit 1
fi

dbgendir=$(realpath "$1")
sf="$2"
nthr="$3"
destdir=$(realpath "$4")

# Hook Ctrl+C and kill all dbgen tasks
# trap '[ 0 -lt $(jobs | wc -l) ] && kill -HUP $(jobs -p)' EXIT
trap 'jobs -pr | xargs -r kill 2>/dev/null || true' EXIT


export DSS_CONFIG=${dbgendir}/

pushd $destdir

# mkdir
for tbl in ${tables[@]}
do
  mkdir -p $tbl
done


for tbidx in ${!tables[@]}
do
  tblname=${tables[$tbidx]}
  optname=${opts[$tbidx]}

  echo "Generating $tblname..."
  mkdir -p ${tblname}
  pushd ${tblname}

  PIDS=()
  if [ $nthr -eq 1 ]; then
    (
      $dbgendir/dbgen -T ${optname} -f -s ${sf}
      echo -e "\tProgress:1/1"
    )&
    PIDS+=($!)
  else
    for i in $(seq 1 ${nthr})
    do
        (
        $dbgendir/dbgen -T ${optname} -f -s ${sf} -C ${nthr} -S ${i}
        echo -e "\tProgress:${i}/${nthr}"
        )&
        PIDS+=($!)
    done
  fi
  wait
  popd
done

wait
popd
