#!/usr/bin/env python3
"""
Polars GPU/CPU benchmark for TPC-H and SSB queries.
Reads configuration from environment variables set by run_benchmark.sh.
"""

import math
import os
import subprocess
import sys
import time
from datetime import datetime

import polars as pl

# ---- Configuration from environment ----
SF = os.environ.get("SF", "100")
BENCHMARK = os.environ.get("BENCHMARK", "all")
ITERATIONS = int(os.environ.get("ITERATIONS", "10"))
ENGINE = os.environ.get("ENGINE", "gpu")
TPCH_PARQUET_BASE = os.environ["TPCH_PARQUET_BASE"]
SSB_PARQUET_BASE = os.environ["SSB_PARQUET_BASE"]
LOG_BASE = os.environ["LOG_BASE"]
ANSWERS_BASE = os.environ["ANSWERS_BASE"]

TPCH_DATA = f"{TPCH_PARQUET_BASE}/sf{SF}"
SSB_DATA = SSB_PARQUET_BASE


# ---- Helpers ----
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


# ---- TPC-H scan ----
def tpch_scan(table):
    return pl.scan_parquet(f"{TPCH_DATA}/{table}.parquet/*", use_statistics=True)


# ---- SSB scan ----
def ssb_scan(table):
    return pl.scan_parquet(f"{SSB_DATA}/{table}.parquet/*", use_statistics=True)


# ---- TPC-H Queries ----
from datetime import date


def tpch_q1():
    return (
        tpch_scan("lineitem")
        .filter(pl.col("l_shipdate") <= date(1998, 9, 2))
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .alias("sum_disc_price"),
            (
                pl.col("l_extendedprice")
                * (1 - pl.col("l_discount"))
                * (1 + pl.col("l_tax"))
            )
            .sum()
            .alias("sum_charge"),
            pl.col("l_quantity").mean().alias("avg_qty"),
            pl.col("l_extendedprice").mean().alias("avg_price"),
            pl.col("l_discount").mean().alias("avg_disc"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )


def tpch_q3():
    return (
        tpch_scan("customer")
        .filter(pl.col("c_mktsegment") == "BUILDING")
        .join(
            tpch_scan("orders").filter(pl.col("o_orderdate") < date(1995, 3, 15)),
            left_on="c_custkey",
            right_on="o_custkey",
        )
        .join(
            tpch_scan("lineitem").filter(pl.col("l_shipdate") > date(1995, 3, 15)),
            left_on="o_orderkey",
            right_on="l_orderkey",
        )
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .alias("revenue")
        )
        .select("o_orderkey", "revenue", "o_orderdate", "o_shippriority")
        .sort("revenue", "o_orderdate", descending=[True, False])
    )


def tpch_q5():
    return (
        tpch_scan("region")
        .filter(pl.col("r_name") == "ASIA")
        .join(tpch_scan("nation"), left_on="r_regionkey", right_on="n_regionkey")
        .join(tpch_scan("supplier"), left_on="n_nationkey", right_on="s_nationkey")
        .join(tpch_scan("lineitem"), left_on="s_suppkey", right_on="l_suppkey")
        .join(
            tpch_scan("orders").filter(
                (pl.col("o_orderdate") >= date(1994, 1, 1))
                & (pl.col("o_orderdate") < date(1995, 1, 1))
            ),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .join(
            tpch_scan("customer"),
            left_on=["o_custkey", "n_nationkey"],
            right_on=["c_custkey", "c_nationkey"],
        )
        .group_by("n_name")
        .agg(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .alias("revenue")
        )
        .sort("revenue", descending=True)
    )


def tpch_q6():
    return (
        tpch_scan("lineitem")
        .filter(
            (pl.col("l_shipdate") >= date(1994, 1, 1))
            & (pl.col("l_shipdate") < date(1995, 1, 1))
            & (pl.col("l_discount").is_between(0.05, 0.07))
            & (pl.col("l_quantity") < 24)
        )
        .select(
            (pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue")
        )
    )


def tpch_q13():
    return (
        tpch_scan("customer")
        .join(
            tpch_scan("orders").filter(
                ~pl.col("o_comment").str.contains(".*special.*requests.*")
            ),
            left_on="c_custkey",
            right_on="o_custkey",
            how="left",
        )
        .group_by("c_custkey")
        .agg(pl.col("o_orderkey").count().alias("c_count"))
        .group_by("c_count")
        .agg(pl.len().alias("custdist"))
        .sort("custdist", "c_count", descending=True)
    )


def tpch_q16():
    bad_suppliers = (
        tpch_scan("supplier")
        .filter(pl.col("s_comment").str.contains(".*Customer.*Complaints.*"))
        .select("s_suppkey")
    )
    return (
        tpch_scan("part")
        .filter(
            (pl.col("p_brand") != "Brand#45")
            & (~pl.col("p_type").str.starts_with("MEDIUM POLISHED"))
            & (pl.col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9]))
        )
        .join(tpch_scan("partsupp"), left_on="p_partkey", right_on="ps_partkey")
        .join(bad_suppliers, left_on="ps_suppkey", right_on="s_suppkey", how="anti")
        .group_by("p_brand", "p_type", "p_size")
        .agg(pl.col("ps_suppkey").n_unique().alias("supplier_cnt"))
        .sort(
            pl.col("supplier_cnt").sort(descending=True),
            "p_brand",
            "p_type",
            "p_size",
        )
    )


# ---- SSB Queries ----


def ssb_q11():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .filter(
            (pl.col("d_year") == 1993)
            & (pl.col("lo_discount").is_between(1, 3))
            & (pl.col("lo_quantity") < 25)
        )
        .select(
            (pl.col("lo_extendedprice") * pl.col("lo_discount"))
            .sum()
            .alias("revenue")
        )
    )


def ssb_q12():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .filter(
            (pl.col("d_yearmonthnum") == 199401)
            & (pl.col("lo_discount").is_between(4, 6))
            & (pl.col("lo_quantity").is_between(26, 35))
        )
        .select(
            (pl.col("lo_extendedprice") * pl.col("lo_discount"))
            .sum()
            .alias("revenue")
        )
    )


def ssb_q13():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .filter(
            (pl.col("d_weeknuminyear") == 6)
            & (pl.col("d_year") == 1994)
            & (pl.col("lo_discount").is_between(5, 7))
            & (pl.col("lo_quantity").is_between(26, 35))
        )
        .select(
            (pl.col("lo_extendedprice") * pl.col("lo_discount"))
            .sum()
            .alias("revenue")
        )
    )


def ssb_q21():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .join(ssb_scan("part"), left_on="lo_partkey", right_on="p_partkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .filter(
            (pl.col("p_category") == "MFGR#12") & (pl.col("s_region") == "AMERICA")
        )
        .group_by("d_year", "p_brand1")
        .agg(pl.col("lo_revenue").sum())
        .sort("d_year", "p_brand1")
        .select("lo_revenue", "d_year", "p_brand1")
    )


def ssb_q22():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .join(ssb_scan("part"), left_on="lo_partkey", right_on="p_partkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .filter(
            (pl.col("p_brand1").is_between(pl.lit("MFGR#2221"), pl.lit("MFGR#2228")))
            & (pl.col("s_region") == "ASIA")
        )
        .group_by("d_year", "p_brand1")
        .agg(pl.col("lo_revenue").sum())
        .sort("d_year", "p_brand1")
        .select("lo_revenue", "d_year", "p_brand1")
    )


def ssb_q23():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .join(ssb_scan("part"), left_on="lo_partkey", right_on="p_partkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .filter(
            (pl.col("p_brand1") == "MFGR#2221") & (pl.col("s_region") == "EUROPE")
        )
        .group_by("d_year", "p_brand1")
        .agg(pl.col("lo_revenue").sum())
        .sort("d_year", "p_brand1")
        .select("lo_revenue", "d_year", "p_brand1")
    )


def ssb_q31():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("customer"), left_on="lo_custkey", right_on="c_custkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .filter(
            (pl.col("c_region") == "ASIA")
            & (pl.col("s_region") == "ASIA")
            & (pl.col("d_year") >= 1992)
            & (pl.col("d_year") <= 1997)
        )
        .group_by("c_nation", "s_nation", "d_year")
        .agg(pl.col("lo_revenue").sum().alias("revenue"))
        .sort("d_year", "revenue", descending=[False, True])
    )


def ssb_q32():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("customer"), left_on="lo_custkey", right_on="c_custkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .filter(
            (pl.col("c_nation") == "UNITED STATES")
            & (pl.col("s_nation") == "UNITED STATES")
            & (pl.col("d_year") >= 1992)
            & (pl.col("d_year") <= 1997)
        )
        .group_by("c_city", "s_city", "d_year")
        .agg(pl.col("lo_revenue").sum().alias("revenue"))
        .sort("d_year", "revenue", descending=[False, True])
    )


def ssb_q33():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("customer"), left_on="lo_custkey", right_on="c_custkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .filter(
            (
                (pl.col("c_city") == "UNITED KI1")
                | (pl.col("c_city") == "UNITED KI5")
            )
            & (
                (pl.col("s_city") == "UNITED KI1")
                | (pl.col("s_city") == "UNITED KI5")
            )
            & (pl.col("d_year") >= 1992)
            & (pl.col("d_year") <= 1997)
        )
        .group_by("c_city", "s_city", "d_year")
        .agg(pl.col("lo_revenue").sum().alias("revenue"))
        .sort("d_year", "revenue", descending=[False, True])
    )


def ssb_q34():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("customer"), left_on="lo_custkey", right_on="c_custkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .filter(
            (
                (pl.col("c_city") == "UNITED KI1")
                | (pl.col("c_city") == "UNITED KI5")
            )
            & (
                (pl.col("s_city") == "UNITED KI1")
                | (pl.col("s_city") == "UNITED KI5")
            )
            & (pl.col("d_yearmonth") == "Dec1997")
        )
        .group_by("c_city", "s_city", "d_year")
        .agg(pl.col("lo_revenue").sum().alias("revenue"))
        .sort("d_year", "revenue", descending=[False, True])
    )


def ssb_q41():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .join(ssb_scan("customer"), left_on="lo_custkey", right_on="c_custkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .join(ssb_scan("part"), left_on="lo_partkey", right_on="p_partkey")
        .filter(
            (pl.col("c_region") == "AMERICA")
            & (pl.col("s_region") == "AMERICA")
            & ((pl.col("p_mfgr") == "MFGR#1") | (pl.col("p_mfgr") == "MFGR#2"))
        )
        .group_by("d_year", "c_nation")
        .agg((pl.col("lo_revenue") - pl.col("lo_supplycost")).sum().alias("profit"))
        .sort("d_year", "c_nation")
    )


def ssb_q42():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .join(ssb_scan("customer"), left_on="lo_custkey", right_on="c_custkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .join(ssb_scan("part"), left_on="lo_partkey", right_on="p_partkey")
        .filter(
            (pl.col("c_region") == "AMERICA")
            & (pl.col("s_region") == "AMERICA")
            & ((pl.col("d_year") == 1997) | (pl.col("d_year") == 1998))
            & ((pl.col("p_mfgr") == "MFGR#1") | (pl.col("p_mfgr") == "MFGR#2"))
        )
        .group_by("d_year", "s_nation", "p_category")
        .agg((pl.col("lo_revenue") - pl.col("lo_supplycost")).sum().alias("profit"))
        .sort("d_year", "s_nation", "p_category")
    )


def ssb_q43():
    return (
        ssb_scan("lineorder")
        .join(ssb_scan("date"), left_on="lo_orderdate", right_on="d_datekey")
        .join(ssb_scan("customer"), left_on="lo_custkey", right_on="c_custkey")
        .join(ssb_scan("supplier"), left_on="lo_suppkey", right_on="s_suppkey")
        .join(ssb_scan("part"), left_on="lo_partkey", right_on="p_partkey")
        .filter(
            (pl.col("c_region") == "AMERICA")
            & (pl.col("s_nation") == "UNITED STATES")
            & ((pl.col("d_year") == 1997) | (pl.col("d_year") == 1998))
            & (pl.col("p_category") == "MFGR#14")
        )
        .group_by("d_year", "s_city", "p_brand1")
        .agg((pl.col("lo_revenue") - pl.col("lo_supplycost")).sum().alias("profit"))
        .sort("d_year", "s_city", "p_brand1")
    )


# ---- Query registry ----
TPCH_QUERIES = {
    "q1": tpch_q1,
    "q3": tpch_q3,
    "q5": tpch_q5,
    "q6": tpch_q6,
    "q13": tpch_q13,
    "q16": tpch_q16,
}

SSB_QUERIES = {
    "q11": ssb_q11,
    "q12": ssb_q12,
    "q13": ssb_q13,
    "q21": ssb_q21,
    "q22": ssb_q22,
    "q23": ssb_q23,
    "q31": ssb_q31,
    "q32": ssb_q32,
    "q33": ssb_q33,
    "q34": ssb_q34,
    "q41": ssb_q41,
    "q42": ssb_q42,
    "q43": ssb_q43,
}


# ---- Verification ----
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
    if not os.path.isfile(answer_file):
        log(f"  {qname}: NO_REF ({answer_file} not found)")
        return True

    if not os.path.isfile(result_file):
        log(f"  {qname}: NO_RESULT ({result_file} not found)")
        return False

    with open(result_file) as f:
        got = f.read().strip()
    with open(answer_file) as f:
        lines = f.read().strip().split("\n")
        ref = "\n".join(lines[1:]).replace('"', '')  # skip header, strip quotes

    got_lines = sorted(got.split("\n"))
    ref_lines = sorted(ref.split("\n"))

    if len(got_lines) != len(ref_lines):
        log(f"  {qname}: FAIL (row count: got {len(got_lines)}, expected {len(ref_lines)})")
        return False

    for i, (g, r) in enumerate(zip(got_lines, ref_lines)):
        if not _fields_match(g, r):
            log(f"  {qname}: FAIL (first mismatch at sorted row {i})")
            log(f"    expected: {r}")
            log(f"    got:      {g}")
            return False

    log(f"  {qname}: MATCH")
    return True


# ---- Save result as pipe-delimited ----
def save_result(df, filepath):
    """Save DataFrame as pipe-delimited text (no header) for verification."""
    lines = []
    for row in df.iter_rows():
        lines.append("|".join(str(v) for v in row))
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


# ---- GPU engine setup ----
collect_kwargs = {}
if ENGINE == "gpu":
    import rmm

    free_memory, _ = rmm.mr.available_device_memory()
    initial_pool_size = 256 * (int(free_memory * 0.9) // 256)
    mr = rmm.mr.CudaAsyncMemoryResource(initial_pool_size=initial_pool_size)
    gpu_engine = pl.GPUEngine(memory_resource=mr, raise_on_fail=True)
    collect_kwargs = {"engine": gpu_engine}

    # Preload GPU engine
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        f = pathlib.Path(tmpdir) / "test.pq"
        pl.DataFrame({"a": [1]}).write_parquet(f)
        pl.scan_parquet(f).collect(engine=gpu_engine)
    log("GPU engine preloaded")


# ---- Benchmark runner ----
def run_suite(suite, queries, answer_dir, sf_label):
    log_dir = f"{LOG_BASE}/{suite}/sf{sf_label}"
    result_dir = f"{log_dir}/results"
    os.makedirs(result_dir, exist_ok=True)
    time_file = f"{log_dir}/time.txt"

    polars_ver = pl.__version__
    rapids_ver = ""
    if ENGINE == "gpu":
        try:
            import rmm as _rmm
            rapids_ver = _rmm.__version__
        except Exception:
            pass
    ver_label = f"Polars {polars_ver}"
    if rapids_ver:
        ver_label += f" + RAPIDS {rapids_ver}"
    log("=" * 60)
    log(f"  {ver_label} (engine={ENGINE}) -- {suite.upper()} SF{sf_label}")
    log("=" * 60)
    log(f"Iterations={ITERATIONS}")
    log("")

    query_times = {}

    for qname, qfunc in queries.items():
        q_result_dir = f"{result_dir}/{qname}"
        os.makedirs(q_result_dir, exist_ok=True)
        qlog_file = f"{result_dir}/{qname}.txt"

        def qlog(msg):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line, file=sys.stderr)
            with open(qlog_file, "a") as f:
                f.write(line + "\n")

        # Clear per-query log
        with open(qlog_file, "w"):
            pass

        qlog(f"=== {qname} ===")
        # Warmup iteration (not counted)
        drop_caches()
        t0 = time.perf_counter()
        df = qfunc().collect(**collect_kwargs)
        t1 = time.perf_counter()
        qlog(f"  warmup: {(t1 - t0) * 1000:.1f} ms (excluded)")

        times = []
        for i in range(1, ITERATIONS + 1):
            drop_caches()
            result_file = f"{q_result_dir}/run{i}.txt"
            t0 = time.perf_counter()
            df = qfunc().collect(**collect_kwargs)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            times.append(elapsed)
            save_result(df, result_file)
            t_ms = elapsed * 1000
            qlog(f"  iteration {i}/{ITERATIONS}: {t_ms:.1f} ms")

        if times:
            avg, mn, mx, std = compute_stats(times)
            qlog(
                f"  avg={avg*1000:.1f} ms  min={mn*1000:.1f} ms  "
                f"max={mx*1000:.1f} ms  std={std*1000:.1f} ms"
            )
        query_times[qname] = times

    # Verify
    log("")
    log("--- Verifying answers ---")
    verify_fail = False
    for qname in queries:
        last_run = f"{result_dir}/{qname}/run{ITERATIONS}.txt"
        ans_file = f"{answer_dir}/{qname}.csv"
        qlog_file = f"{result_dir}/{qname}.txt"

        # Re-bind qlog for verify output
        def qlog(msg, _f=qlog_file):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line, file=sys.stderr)
            with open(_f, "a") as f:
                f.write(line + "\n")

        if not verify_query(qname, last_run, ans_file):
            verify_fail = True
            with open(qlog_file, "a") as f:
                f.write(f"  {qname}: FAIL\n")
        else:
            with open(qlog_file, "a") as f:
                f.write(f"  {qname}: MATCH\n")

    if verify_fail:
        log("WARNING: Some answers did not match reference.")
    else:
        log("All answers matched.")

    # Summary table
    log("")
    log("=== Summary ===")

    lines = []
    lines.append(
        f"=== {ver_label} (engine={ENGINE}) -- {suite.upper()} SF{sf_label} ==="
    )
    lines.append(f"Iterations={ITERATIONS}")
    lines.append("")

    header = f"{'Query':<8}"
    for i in range(1, ITERATIONS + 1):
        header += f" {'Run' + str(i):>10}"
    header += f" {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Std(ms)':>10}"
    lines.append(header)

    sep = f"{'--------':<8}"
    for _ in range(ITERATIONS):
        sep += f" {'----------':>10}"
    sep += f" {'----------':>10} {'----------':>10} {'----------':>10} {'----------':>10}"
    lines.append(sep)

    for qname in queries:
        times = query_times.get(qname, [])
        if not times:
            row = f"{qname:<8} FAILED"
        else:
            avg, mn, mx, std = compute_stats(times)
            row = f"{qname:<8}"
            for t in times:
                row += f" {t*1000:10.1f}"
            row += f" {avg*1000:10.1f} {mn*1000:10.1f} {mx*1000:10.1f} {std*1000:10.1f}"
        lines.append(row)

    table = "\n".join(lines)
    print(table)
    with open(time_file, "w") as f:
        f.write(table + "\n")

    log("")
    log(f"Results: {time_file}")


# ---- Main ----
run_tpch = BENCHMARK in ("tpch", "all")
run_ssb = BENCHMARK in ("ssb", "all")

def _check_data_dir(data_dir, suite):
    """Check that data directory exists and contains non-empty parquet subdirs."""
    if not os.path.isdir(data_dir):
        log(f"ERROR: {suite} data directory not found: {data_dir}")
        log(f"Run 'BENCHMARK={suite.lower()} bash duckdb/load.sh' first to generate Parquet files.")
        sys.exit(1)
    for d in os.listdir(data_dir):
        subdir = os.path.join(data_dir, d)
        if os.path.isdir(subdir) and not os.listdir(subdir):
            log(f"ERROR: {suite} data directory has empty subdirectory: {subdir}")
            log(f"Run 'BENCHMARK={suite.lower()} bash duckdb/load.sh' to regenerate Parquet files.")
            sys.exit(1)


if run_tpch:
    _check_data_dir(TPCH_DATA, "TPC-H")

if run_ssb:
    _check_data_dir(SSB_DATA, "SSB")

if run_tpch:
    run_suite("tpch", TPCH_QUERIES, f"{ANSWERS_BASE}/tpch/sf{SF}", SF)

if run_ssb:
    run_suite("ssb", SSB_QUERIES, f"{ANSWERS_BASE}/ssb/sf{SF}", SF)
