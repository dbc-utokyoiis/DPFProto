#!/usr/bin/env python3
"""
TPC-H Benchmark Driver for Spark-RAPIDS
Usage:
    spark-submit tpch_benchmark.py --mode gpu
    spark-submit tpch_benchmark.py --mode cpu
"""
import argparse
import math
import os
import subprocess
import sys
import time
from datetime import datetime

from pyspark.sql import SparkSession


# ---------------------------------------------------------------------------
# Configuration (overridable via environment)
# ---------------------------------------------------------------------------
SF = os.environ.get("SF", "100")
TPCH_PARQUET_BASE = os.environ.get(
    "TPCH_PARQUET_BASE",
    "/export/data1/tpch/duckdb/sideways/parquet",
)
QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries")
LOG_BASE = os.environ.get("LOG_BASE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "logs", "spark-rapids",
))
ANSWERS_BASE = os.environ.get("ANSWERS_BASE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "answers",
))

TABLES = [
    "customer", "lineitem", "nation", "orders",
    "part", "partsupp", "region", "supplier",
]
QUERIES = ["q1", "q3", "q5", "q6", "q13", "q16"]
WARMUP_RUNS = 1
MEASURE_RUNS = int(os.environ.get("ITERATIONS", "10"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)


def drop_caches():
    subprocess.run(
        "echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null",
        shell=True,
        check=True,
    )


def compute_stats(times):
    n = len(times)
    avg = sum(times) / n
    mn = min(times)
    mx = max(times)
    variance = sum((t - avg) ** 2 for t in times) / n
    std = math.sqrt(variance)
    return avg, mn, mx, std


def create_spark_session(mode):
    builder = SparkSession.builder.appName(f"TPC-H Benchmark ({mode.upper()})")
    if mode == "cpu":
        builder = (
            builder.config("spark.plugins", "")
            .config("spark.rapids.sql.enabled", "false")
        )
    return builder.getOrCreate()


def register_tables(spark, data_dir):
    for table in TABLES:
        path = os.path.join(data_dir, f"{table}.parquet")
        spark.read.parquet(path).createOrReplaceTempView(table)
        count = spark.table(table).count()
        log(f"  Registered '{table}' — {count:,} rows")


def load_query(name):
    path = os.path.join(QUERY_DIR, f"{name}.sql")
    with open(path) as f:
        return f.read()


def save_result(rows, schema, filepath):
    """Save query result as pipe-delimited text with header."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = "|".join(schema)
    with open(filepath, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write("|".join(str(v) for v in row) + "\n")


REL_TOL = 1e-2


def _fields_match(got_line, ref_line):
    """Compare two pipe-delimited lines, allowing relative tolerance on numeric fields."""
    got_fields = got_line.split("|")
    ref_fields = ref_line.split("|")
    if len(got_fields) != len(ref_fields):
        return False
    for g, r in zip(got_fields, ref_fields):
        if g == r:
            continue
        try:
            gv, rv = float(g), float(r)
            if rv == 0:
                if abs(gv) > REL_TOL:
                    return False
            elif abs(gv - rv) / abs(rv) > REL_TOL:
                return False
        except ValueError:
            return False
    return True


def verify_query(qname, result_file, answer_file):
    if not os.path.exists(answer_file):
        log(f"  {qname}: SKIP (no answer file: {answer_file})")
        return True
    if not os.path.exists(result_file):
        log(f"  {qname}: FAIL (no result file)")
        return False

    def read_lines(path):
        with open(path) as f:
            lines = f.read().strip().splitlines()
        if not lines:
            return []
        # Skip header, strip quotes
        return sorted([l.replace('"', '') for l in lines[1:]])

    expected = read_lines(answer_file)
    got = read_lines(result_file)

    if len(got) != len(expected):
        log(f"  {qname}: FAIL (row count: got {len(got)}, expected {len(expected)})")
        return False

    for i, (g, r) in enumerate(zip(got, expected)):
        if not _fields_match(g, r):
            log(f"  {qname}: FAIL (first mismatch at sorted row {i})")
            log(f"    expected: {r}")
            log(f"    got:      {g}")
            return False

    log(f"  {qname}: MATCH")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TPC-H Spark-RAPIDS Benchmark")
    parser.add_argument("--mode", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--queries", nargs="+", default=QUERIES)
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=MEASURE_RUNS)
    args = parser.parse_args()

    data_dir = f"{TPCH_PARQUET_BASE}/sf{SF}"
    log_dir = f"{LOG_BASE}/tpch/sf{SF}"
    result_dir = f"{log_dir}/results"
    os.makedirs(result_dir, exist_ok=True)

    # ---- Spark Session ----
    spark = create_spark_session(args.mode)
    sc = spark.sparkContext

    spark_ver = spark.version
    rapids_ver = ""
    if args.mode == "gpu":
        try:
            rapids_ver = spark.conf.get("spark.rapids.version", "")
        except Exception:
            pass
    ver_label = f"Spark {spark_ver}"
    if rapids_ver:
        ver_label += f" + RAPIDS {rapids_ver}"

    log("=" * 60)
    log(f"  {ver_label} (mode={args.mode.upper()}) -- TPC-H SF{SF}")
    log("=" * 60)
    log(f"Data: {data_dir}")
    log(f"Warmup={args.warmup}, Iterations={args.runs}")
    log("")
    log(f"Spark UI: {sc.uiWebUrl}")
    log(f"App ID:   {sc.applicationId}")
    log("")

    # ---- Register Tables ----
    log("Registering TPC-H tables ...")
    register_tables(spark, data_dir)
    log("")

    # ---- Run Queries ----
    query_times = {}

    for qname in args.queries:
        sql = load_query(qname)
        q_result_dir = f"{result_dir}/{qname}"
        os.makedirs(q_result_dir, exist_ok=True)
        qlog_file = f"{result_dir}/{qname}.txt"

        def qlog(msg, _f=qlog_file):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line, file=sys.stderr)
            with open(_f, "a") as f:
                f.write(line + "\n")

        with open(qlog_file, "w"):
            pass

        qlog(f"=== {qname} ===")

        # Warmup
        for w in range(args.warmup):
            drop_caches()
            t0 = time.perf_counter()
            spark.sql(sql).collect()
            t1 = time.perf_counter()
            qlog(f"  warmup {w + 1}/{args.warmup}: {(t1 - t0) * 1000:.1f} ms (excluded)")

        # Measurement
        times = []
        for i in range(1, args.runs + 1):
            drop_caches()
            t0 = time.perf_counter()
            rows = spark.sql(sql).collect()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            times.append(elapsed)

            # Save result
            schema = spark.sql(sql).columns
            result_file = f"{q_result_dir}/run{i}.txt"
            save_result(
                [list(r) for r in rows],
                schema,
                result_file,
            )
            qlog(f"  iteration {i}/{args.runs}: {elapsed * 1000:.1f} ms")

        if times:
            avg, mn, mx, std = compute_stats(times)
            qlog(
                f"  avg={avg * 1000:.1f} ms  min={mn * 1000:.1f} ms  "
                f"max={mx * 1000:.1f} ms  std={std * 1000:.1f} ms"
            )
        query_times[qname] = times

    # ---- Verify ----
    log("")
    log("--- Verifying answers ---")
    answer_dir = f"{ANSWERS_BASE}/tpch/sf{SF}"
    verify_fail = False
    for qname in args.queries:
        last_run = f"{result_dir}/{qname}/run{args.runs}.txt"
        ans_file = f"{answer_dir}/{qname}.csv"
        qlog_file = f"{result_dir}/{qname}.txt"
        matched = verify_query(qname, last_run, ans_file)
        label = "MATCH" if matched else "FAIL"
        with open(qlog_file, "a") as f:
            f.write(f"  {qname}: {label}\n")
        if not matched:
            verify_fail = True

    if verify_fail:
        log("WARNING: Some answers did not match reference.")
    else:
        log("All answers matched.")

    # ---- Summary Table ----
    log("")
    log("=== Summary ===")

    lines = []
    lines.append(
        f"=== {ver_label} (mode={args.mode.upper()}) -- TPC-H SF{SF} ==="
    )
    lines.append(f"Warmup={args.warmup}, Iterations={args.runs}")
    lines.append("")

    header = f"{'Query':<8}"
    for i in range(1, args.runs + 1):
        header += f" {'Run' + str(i):>10}"
    header += f" {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Std(ms)':>10}"
    lines.append(header)
    lines.append(
        f"{'--------':<8}"
        + (" ----------" * args.runs)
        + " ---------- ---------- ---------- ----------"
    )

    for qname in args.queries:
        times = query_times.get(qname, [])
        if not times:
            continue
        avg, mn, mx, std = compute_stats(times)
        row = f"{qname:<8}"
        for t in times:
            row += f" {t * 1000:>10.1f}"
        row += f" {avg * 1000:>10.1f} {mn * 1000:>10.1f} {mx * 1000:>10.1f} {std * 1000:>10.1f}"
        lines.append(row)

    time_file = f"{log_dir}/time.txt"
    with open(time_file, "w") as f:
        for line in lines:
            f.write(line + "\n")
    for line in lines:
        log(line)

    log("")
    log(f"Summary written to {time_file}")

    spark.stop()


if __name__ == "__main__":
    main()
