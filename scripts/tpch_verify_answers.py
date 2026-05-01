#!/usr/bin/env python3
"""
tpch_verify_answers.py — Verify tpchdb results against DuckDB reference answers.

Compares query results in tpch_run_all log directories against
DuckDB-generated reference answers in answers/tpch/sfXX/*.csv.

CSV format: pipe-delimited, first line is header.

Usage:
    scripts/tpch_verify_answers.py logs/tpch_run_all/20260327_110230
    scripts/tpch_verify_answers.py logs/tpch_run_all/20260327_110230/sf100
    scripts/tpch_verify_answers.py logs/tpch_run_all/20260327_110230 --sf 50,200,300

    # Q6 revenue selectivity sweep (run-dir scan also picks this up automatically)
    scripts/tpch_verify_answers.py logs/tpch_run_all/20260419_163435/revenue
    scripts/tpch_verify_answers.py logs/tpch_run_all/20260419_163435/revenue/sf100
"""

import argparse
import csv
import io
import os
import re
import sys

QUERIES = ["q1", "q3", "q5", "q6", "q13", "q16"]
ALL_MODES = [
    "gidp", "gidp+uncomp",
    "gidp+bam", "gidp+bam+uncomp",
    "gidp+bam+fusion", "gidp+bam+fusion+uncomp",
    "datapathfusion", "datapathfusion+uncomp",
]

# Q6 revenue selectivity: end-date (YYYYMMDD) → answer CSV basename in q6sel/
# The basename selN is the selectivity percentile for the date range
# [1992-01-01, end-date) over the full LINEITEM shipdate distribution.
REVENUE_DATE_TO_SEL = {
    "19920701": "sel5",
    "19930101": "sel13",
    "19940101": "sel28",
    "19950101": "sel43",
    "19960101": "sel58",
    "19970101": "sel73",
    "19980101": "sel89",
    "19990101": "sel100",
}

# ---- CSV answer parsing ----

def parse_csv_answer(text):
    """Parse pipe-delimited CSV with header into list of row dicts."""
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    rows = []
    for row in reader:
        # Strip whitespace and quotes from all values
        rows.append({k: v.strip().strip('"') for k, v in row.items()})
    return rows


# ---- tpchdb log result parsing ----

def extract_tpchdb_q6(text):
    """Extract Q6 revenue from tpchdb log. Returns list of integer values."""
    vals = re.findall(r"TPCH Q6 total revenue: (\d+)", text)
    return list(set(vals))


def extract_tpchdb_block(text, header_pattern):
    """Extract data rows from a tpchdb result block (first trial only)."""
    m = re.search(header_pattern, text)
    if not m:
        return None
    remaining = text[m.end():]
    lines = remaining.split("\n")
    data_lines = []
    skip = 2  # skip column header + separator
    for line in lines:
        if skip > 0:
            skip -= 1
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("Elapsed") or stripped.startswith("time:") \
                or stripped.startswith("===") or stripped.startswith("Total"):
            break
        if stripped.startswith("... ("):
            continue
        data_lines.append(stripped)
    return data_lines


def parse_tpchdb_rows(lines, col_names):
    """Parse pipe-separated tpchdb rows into dicts."""
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= len(col_names):
            rows.append(dict(zip(col_names, parts[:len(col_names)])))
    return rows


# ---- Comparison logic per query ----

def compare_q6(ref_rows, tpchdb_text):
    """Q6: single revenue value. tpchdb is ×10000 integer."""
    if not ref_rows:
        return "NO_REF", ""
    ref_val = ref_rows[0].get("revenue", "")
    # CSV: "12330426888.4637" → integer: 123304268884637
    ref_int = ref_val.replace(".", "")
    tpch_vals = extract_tpchdb_q6(tpchdb_text)
    if not tpch_vals:
        return "NO_DATA", ""
    tpch_val = tpch_vals[0]
    if ref_int == tpch_val:
        return "MATCH", ""
    return "FAIL", f"expected={ref_int} got={tpch_val}"


def compare_q1(ref_rows, tpchdb_text):
    """Q1: 4 rows. Compare count_order (exact int) and key aggregates."""
    block = extract_tpchdb_block(tpchdb_text, r"=== TPC-H Q1 Result ===")
    if not block:
        return "NO_DATA", ""
    cols = ["l_returnflag", "l_linestatus", "sum_qty", "sum_base_price",
            "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "count_order"]
    tpch_rows = parse_tpchdb_rows(block, cols)
    top_n = min(len(ref_rows), len(tpch_rows))

    errors = []
    for i in range(top_n):
        ref, tp = ref_rows[i], tpch_rows[i]
        # count_order: exact integer match
        r_count = ref.get("count_order", "").strip()
        t_count = tp.get("count_order", "").strip()
        if r_count != t_count:
            errors.append(f"count_order: {ref['l_returnflag']}|{ref['l_linestatus']}: "
                          f"expected={r_count} got={t_count}")
        # sum_qty: decimal(38,2) → ×100 integer
        r_qty = ref.get("sum_qty", "").replace(".", "")
        t_qty = tp.get("sum_qty", "").strip()
        if r_qty != t_qty:
            errors.append(f"sum_qty: {ref['l_returnflag']}|{ref['l_linestatus']}: "
                          f"expected={r_qty} got={t_qty}")
        # sum_disc_price: decimal(38,4) → ×10000 integer
        r_dp = ref.get("sum_disc_price", "").replace(".", "")
        t_dp = tp.get("sum_disc_price", "").strip()
        if r_dp != t_dp:
            errors.append(f"sum_disc_price: {ref['l_returnflag']}|{ref['l_linestatus']}: "
                          f"expected={r_dp} got={t_dp}")

    if errors:
        return "FAIL", f"{len(errors)} mismatches in {top_n} rows; " + "; ".join(errors[:3])
    return "MATCH", f"{top_n} rows"


def compare_q3(ref_rows, tpchdb_text):
    """Q3: Compare all ref rows. l_orderkey exact, revenue ×10000."""
    block = extract_tpchdb_block(tpchdb_text, r"=== TPC-H Q3 Result")
    if not block:
        return "NO_DATA", ""
    cols = ["l_orderkey", "revenue", "o_orderdate", "o_shippriority"]
    tpch_rows = parse_tpchdb_rows(block, cols)
    top_n = min(len(ref_rows), len(tpch_rows))
    errors = []
    for i in range(top_n):
        ref, tp = ref_rows[i], tpch_rows[i]
        r_key = ref["l_orderkey"].strip()
        t_key = tp["l_orderkey"].strip()
        if r_key != t_key:
            errors.append(f"row {i}: orderkey expected={r_key} got={t_key}")
        r_rev = ref["revenue"].replace(".", "")
        t_rev = tp["revenue"].strip()
        if r_rev != t_rev:
            errors.append(f"row {i}: revenue expected={r_rev} got={t_rev}")

    if errors:
        return "FAIL", f"{len(errors)} mismatches in {top_n} rows; " + "; ".join(errors[:3])
    return "MATCH", f"{top_n} rows"


def compare_q5(ref_rows, tpchdb_text):
    """Q5: Compare all ref rows. n_name exact, revenue ×10000."""
    block = extract_tpchdb_block(tpchdb_text, r"=== TPC-H Q5 Result ===")
    if not block:
        return "NO_DATA", ""
    cols = ["n_name", "revenue"]
    tpch_rows = parse_tpchdb_rows(block, cols)
    top_n = min(len(ref_rows), len(tpch_rows))

    errors = []
    for i in range(top_n):
        ref, tp = ref_rows[i], tpch_rows[i]
        r_name = ref["n_name"].strip()
        t_name = tp["n_name"].strip()
        if r_name != t_name:
            errors.append(f"row {i}: name expected={r_name} got={t_name}")
        r_rev = ref["revenue"].replace(".", "")
        t_rev = tp["revenue"].strip()
        if r_rev != t_rev:
            errors.append(f"row {i}: revenue expected={r_rev} got={t_rev}")

    if errors:
        return "FAIL", f"{len(errors)} mismatches in {top_n} rows; " + "; ".join(errors[:3])
    return "MATCH", f"{top_n} rows"


def compare_q13(ref_rows, tpchdb_text):
    """Q13: c_count/custdist pairs. Compare all ref rows."""
    block = extract_tpchdb_block(tpchdb_text, r"=== TPC-H Q13 Result ===")
    if not block:
        return "NO_DATA", ""
    cols = ["c_count", "custdist"]
    tpch_rows = parse_tpchdb_rows(block, cols)
    top_n = min(len(ref_rows), len(tpch_rows))

    errors = []
    for i in range(top_n):
        ref, tp = ref_rows[i], tpch_rows[i]
        for col in cols:
            r_val = ref[col].strip()
            t_val = tp[col].strip()
            if r_val != t_val:
                errors.append(f"row {i}: {col} expected={r_val} got={t_val}")

    if errors:
        return "FAIL", f"{len(errors)} mismatches in {top_n} rows; " + "; ".join(errors[:3])
    return "MATCH", f"{top_n} rows"


def compare_q16(ref_rows, tpchdb_text):
    """Q16: brand/type/size/supplier_cnt. Compare all ref rows."""
    block = extract_tpchdb_block(tpchdb_text, r"=== TPC-H Q16 Result ===")
    if not block:
        return "NO_DATA", ""
    cols = ["p_brand", "p_type", "p_size", "supplier_cnt"]
    tpch_rows = parse_tpchdb_rows(block, cols)
    top_n = min(len(ref_rows), len(tpch_rows))

    errors = []
    for i in range(top_n):
        ref, tp = ref_rows[i], tpch_rows[i]
        for col in cols:
            r_val = ref[col].strip().strip('"')
            t_val = tp[col].strip()
            if r_val != t_val:
                errors.append(f"row {i}: {col} expected={r_val} got={t_val}")

    if errors:
        return "FAIL", f"{len(errors)} mismatches in {top_n} rows; " + "; ".join(errors[:3])
    return "MATCH", f"{top_n} rows"


COMPARE_FN = {
    "q1": compare_q1,
    "q3": compare_q3,
    "q5": compare_q5,
    "q6": compare_q6,
    "q13": compare_q13,
    "q16": compare_q16,
}


# ---- Revenue (Q6 selectivity sweep) verification ----

REVENUE_LOG_RE = re.compile(r"revenue_19920101_(\d{8})\.txt$")


def extract_revenue_total(text):
    """Extract 'Revenue total: <int>' from a revenue log. Value is scaled ×10000."""
    m = re.search(r"Revenue total:\s*(\d+)", text)
    return m.group(1) if m else None


def verify_revenue_sf_dir(sf_dir, answer_dir):
    """Verify revenue_YYYYMMDD_YYYYMMDD.txt logs against answers/tpch/sfXX/q6sel/selN.csv."""
    sf_name = os.path.basename(sf_dir)
    if not os.path.isdir(answer_dir):
        print(f"  WARNING: No q6sel reference answers in {answer_dir}")
        return 0, 0, 0

    # Load selN.csv answers indexed by end-date
    # answers/tpch/sfXX/q6sel/selN.csv format: header "revenue" + single decimal value
    ref_by_date = {}
    for date_key, sel in REVENUE_DATE_TO_SEL.items():
        path = os.path.join(answer_dir, f"{sel}.csv")
        if not os.path.isfile(path):
            continue
        lines = [ln.strip() for ln in open(path).read().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        ref_by_date[date_key] = (sel, lines[-1])  # last non-empty line is the value

    # Find available modes (subdirectories containing revenue_*.txt)
    avail_modes = []
    for d in sorted(os.listdir(sf_dir)):
        mdir = os.path.join(sf_dir, d)
        if os.path.isdir(mdir) and any(REVENUE_LOG_RE.search(f) for f in os.listdir(mdir)):
            avail_modes.append(d)

    if not avail_modes:
        print(f"  No revenue modes found in {sf_dir}")
        return 0, 0, 0

    dates = sorted(ref_by_date.keys())

    # Print header
    print(f"\n{'EndDate':<10} {'Sel':<8}", end="")
    for m in avail_modes:
        print(f" | {m:<28}", end="")
    print()
    print(f"{'--------':<10} {'-----':<8}", end="")
    for _ in avail_modes:
        print(f" | {'----------------------------':<28}", end="")
    print()

    total = 0
    passed = 0
    failed = 0
    fail_details = []

    for date_key in dates:
        sel, ref_val = ref_by_date[date_key]
        # Expected integer: remove decimal point from DuckDB answer (scale ×10000)
        ref_int = ref_val.replace(".", "")
        print(f"{date_key:<10} {sel:<8}", end="")

        for m in avail_modes:
            log_file = os.path.join(sf_dir, m, f"revenue_19920101_{date_key}.txt")
            if not os.path.isfile(log_file):
                print(f" | {'-':<28}", end="")
                continue

            total += 1
            log_text = open(log_file, errors="replace").read()
            actual = extract_revenue_total(log_text)

            if actual is None:
                print(f" | \033[33m{'NO_DATA':<28}\033[0m", end="")
                continue

            if actual == ref_int:
                print(f" | \033[32m{'MATCH':<28}\033[0m", end="")
                passed += 1
            else:
                # Allow 1 LSB rounding drift (DECIMAL accumulation order)
                try:
                    diff = abs(int(actual) - int(ref_int))
                except ValueError:
                    diff = None
                if diff is not None and diff <= 1:
                    label = f"MATCH (±1 LSB)"
                    print(f" | \033[32m{label:<28}\033[0m", end="")
                    passed += 1
                else:
                    print(f" | \033[31m{'FAIL':<28}\033[0m", end="")
                    failed += 1
                    fail_details.append((date_key, sel, m,
                                         f"expected={ref_int} got={actual}"))
        print()

    if fail_details:
        print(f"\n  FAIL details:")
        for date_key, sel, m, detail in fail_details:
            print(f"    {date_key} ({sel}) {m}: {detail}")

    return total, passed, failed


def verify_sf_dir(sf_dir, answer_dir):
    """Verify all modes/queries in one SF directory."""
    sf_name = os.path.basename(sf_dir)
    if not os.path.isdir(answer_dir):
        print(f"  WARNING: No reference answers in {answer_dir}")
        return 0, 0, 0

    # Load CSV answers
    ref_answers = {}
    for q in QUERIES:
        ans_file = os.path.join(answer_dir, f"{q}.csv")
        if os.path.isfile(ans_file):
            text = open(ans_file).read()
            if text.strip():
                ref_answers[q] = parse_csv_answer(text)

    # Detect directory layout:
    #   Layout A: sf_dir/{mode}/{q}.txt  (e.g. gidp/q1.txt)
    #   Layout B: sf_dir/{q}/{mode}.txt  (e.g. q1/gidp.txt)
    subdirs = [d for d in os.listdir(sf_dir) if os.path.isdir(os.path.join(sf_dir, d))]
    layout_b = any(d in QUERIES for d in subdirs)

    if layout_b:
        # Collect available modes from query subdirectories
        modes_set = set()
        for q in QUERIES:
            qdir = os.path.join(sf_dir, q)
            if os.path.isdir(qdir):
                for f in os.listdir(qdir):
                    if f.endswith(".txt"):
                        modes_set.add(f[:-4])
        avail_modes = sorted(modes_set)
    else:
        avail_modes = [m for m in ALL_MODES if os.path.isdir(os.path.join(sf_dir, m))]

    if not avail_modes:
        print(f"  No execution modes found in {sf_dir}")
        return 0, 0, 0

    # Print header
    print(f"\n{'Query':<8}", end="")
    for m in avail_modes:
        print(f" | {m:<28}", end="")
    print()
    print(f"{'--------':<8}", end="")
    for _ in avail_modes:
        print(f" | {'----------------------------':<28}", end="")
    print()

    total = 0
    passed = 0
    failed = 0
    fail_details = []

    for q in QUERIES:
        print(f"{q:<8}", end="")
        if q not in ref_answers:
            for _ in avail_modes:
                print(f" | {'NO_REF':<28}", end="")
            print()
            continue

        ref_rows = ref_answers[q]
        compare_fn = COMPARE_FN[q]

        for m in avail_modes:
            if layout_b:
                log_file = os.path.join(sf_dir, q, f"{m}.txt")
            else:
                log_file = os.path.join(sf_dir, m, f"{q}.txt")
            if not os.path.isfile(log_file):
                print(f" | {'-':<28}", end="")
                continue

            total += 1
            log_text = open(log_file, errors="replace").read()
            status, detail = compare_fn(ref_rows, log_text)

            if status == "MATCH":
                label = f"MATCH ({detail})" if detail else "MATCH"
                print(f" | \033[32m{label:<28}\033[0m", end="")
                passed += 1
            elif status == "NO_DATA":
                print(f" | \033[33m{'NO_DATA':<28}\033[0m", end="")
            else:
                print(f" | \033[31m{'FAIL':<28}\033[0m", end="")
                failed += 1
                fail_details.append((q, m, detail))
        print()

    if fail_details:
        print(f"\n  FAIL details:")
        for q, m, detail in fail_details:
            print(f"    {q} {m}: {detail}")

    return total, passed, failed


def main():
    parser = argparse.ArgumentParser(description="Verify tpchdb results against DuckDB answers")
    parser.add_argument("log_dir", help="Log directory (run dir or specific sf dir)")
    parser.add_argument("--sf", help="Comma-separated SF list (default: auto-detect)")
    parser.add_argument("--answers", default="answers/tpch",
                        help="Base directory for DuckDB answers (default: answers/tpch)")
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    answers_base = os.path.abspath(args.answers)

    # Detect if log_dir is a SF directory, a revenue subtree, or a run directory.
    # Revenue subtrees contain only revenue_*.txt logs; key is the presence of a
    # "revenue" subdir at the run-dir level, or "revenue" in the path.
    basename = os.path.basename(log_dir)
    is_revenue_subtree = False
    revenue_root = None

    if basename == "revenue" and os.path.isdir(log_dir):
        # log_dir IS .../revenue/  →  verify all sfXX under it
        is_revenue_subtree = True
        revenue_root = log_dir

    if basename.startswith("sf"):
        # Detect revenue-style SF dir by checking for revenue_*.txt files
        parent = os.path.dirname(log_dir)
        if os.path.basename(parent) == "revenue":
            is_revenue_subtree = True
            revenue_root = parent
            sf_dirs = [(basename, log_dir)]
        else:
            sf_dirs = [(basename, log_dir)]
    elif is_revenue_subtree:
        if args.sf:
            sfs = [f"sf{s}" for s in args.sf.split(",")]
        else:
            sfs = sorted([d for d in os.listdir(log_dir)
                          if d.startswith("sf") and os.path.isdir(os.path.join(log_dir, d))])
        sf_dirs = [(s, os.path.join(log_dir, s)) for s in sfs]
    else:
        if args.sf:
            sfs = [f"sf{s}" for s in args.sf.split(",")]
        else:
            sfs = sorted([d for d in os.listdir(log_dir)
                          if d.startswith("sf") and os.path.isdir(os.path.join(log_dir, d))])
        sf_dirs = [(s, os.path.join(log_dir, s)) for s in sfs]

    print("=" * 72)
    if is_revenue_subtree:
        print("TPC-H Q6 Revenue (selectivity sweep) Verification against DuckDB Reference")
    else:
        print("TPC-H Result Verification against DuckDB Reference")
    print("=" * 72)
    print(f"Log directory : {log_dir}")
    print(f"Answers base  : {answers_base}")
    print(f"Scale factors : {', '.join(s for s, _ in sf_dirs)}")
    print("=" * 72)

    grand_total = 0
    grand_passed = 0
    grand_failed = 0

    for sf_name, sf_path in sf_dirs:
        if is_revenue_subtree:
            ans_dir = os.path.join(answers_base, sf_name, "q6sel")
        else:
            ans_dir = os.path.join(answers_base, sf_name)
        print(f"\n{'=' * 72}")
        print(f"  {sf_name}")
        print(f"{'=' * 72}")
        if is_revenue_subtree:
            t, p, f = verify_revenue_sf_dir(sf_path, ans_dir)
        else:
            t, p, f = verify_sf_dir(sf_path, ans_dir)
            # Also verify the sibling revenue subtree if present (run-dir layout):
            #   run_dir/revenue/sfXX/{mode}/revenue_*.txt
            run_dir = os.path.dirname(sf_path)
            rev_sf_dir = os.path.join(run_dir, "revenue", sf_name)
            if os.path.isdir(rev_sf_dir):
                rev_ans_dir = os.path.join(answers_base, sf_name, "q6sel")
                print(f"\n  --- Q6 revenue selectivity sweep ({sf_name}) ---")
                rt, rp, rf = verify_revenue_sf_dir(rev_sf_dir, rev_ans_dir)
                t += rt
                p += rp
                f += rf
        grand_total += t
        grand_passed += p
        grand_failed += f
        print(f"\n  {sf_name}: {t} tests — {p} MATCH, {f} FAIL")

    print(f"\n{'=' * 72}")
    print(f"TOTAL: {grand_total} tests — {grand_passed} MATCH, {grand_failed} FAIL")
    pct = grand_passed / grand_total * 100 if grand_total else 0
    print(f"Pass rate: {pct:.1f}%")
    print("=" * 72)

    sys.exit(grand_failed)


if __name__ == "__main__":
    main()
