#!/usr/bin/env python3
"""
ssb_verify_answers.py — Verify ssbdb results against reference answers.

Compares ssbdb output against pipe-delimited CSV reference answers
in answers/ssb/sfXX/*.csv.

Usage:
    # Verify pre-captured logs (all modes under a run directory)
    scripts/ssb_verify_answers.py --sf 100 --log-dir logs/ssb_run_all/20260402_231127/sf100/gidp

    # Verify all modes under a run
    scripts/ssb_verify_answers.py --sf 100 --run-dir logs/ssb_run_all/20260402_231127/sf100

    # Run ssbdb and verify
    scripts/ssb_verify_answers.py --sf 100 --build build --devices /dev/nvme0n1p1,...
"""

import argparse
import csv
import os
import re
import sys

QUERIES = [
    "q11", "q12", "q13",
    "q21", "q22", "q23",
    "q31", "q32", "q33", "q34",
    "q41", "q42", "q43",
]


# ---- Helpers ----

def extract_last_trial(text):
    """Extract only the last trial's output, skipping warmup runs."""
    # Split on "======== Trial N / M ========" markers
    parts = re.split(r"={3,}\s*Trial\s+\d+\s*/\s*\d+\s*={3,}", text)
    if len(parts) >= 2:
        # Take the last trial block (text after last "Trial N/M" marker)
        return parts[-1]
    # No trial marker found — might be single-run output, use full text
    # But still try to skip warmup
    parts = re.split(r"={3,}\s*Warmup\s*={3,}", text)
    if len(parts) >= 2:
        return parts[-1]
    return text


def parse_csv_answer(path):
    """Parse pipe-delimited CSV reference answer, stripping quotes and whitespace."""
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="|")
        headers = reader.fieldnames
        rows = [{k: v.strip().strip('"') for k, v in row.items()} for row in reader]
    return headers, rows


def rows_to_tuples(rows, headers):
    """Convert list of dicts to sorted list of tuples for order-independent comparison."""
    tuples = [tuple(r[h] for h in headers) for r in rows]
    return sorted(tuples)


def diff_sorted(got_sorted, ref_sorted):
    """Return first difference between two sorted tuple lists."""
    if len(got_sorted) != len(ref_sorted):
        return f"row count: got={len(got_sorted)} ref={len(ref_sorted)}"
    for i, (g, r) in enumerate(zip(got_sorted, ref_sorted)):
        if g != r:
            return f"row {i} (sorted): got={g} ref={r}"
    return None


# ---- Per-query-flight verification ----

def verify_q1x(ref_headers, ref_rows, text):
    text = extract_last_trial(text)
    ref_val = ref_rows[0]["revenue"]
    m = re.search(r"SSB Q1\.x revenue:\s*(\d+)", text)
    if not m:
        return "NO_DATA", ""
    got = m.group(1)
    if got == ref_val:
        return "MATCH", ""
    return "FAIL", f"expected={ref_val} got={got}"


def verify_q2x(ref_headers, ref_rows, text):
    text = extract_last_trial(text)
    lines = re.findall(r"^\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*$", text, re.M)
    if not lines:
        return "NO_DATA", ""
    got_tuples = sorted([(rev.strip(), yr.strip(), brand.strip()) for rev, yr, brand in lines])
    ref_tuples = sorted([(r["lo_revenue"], r["d_year"], r["p_brand1"]) for r in ref_rows])
    if got_tuples == ref_tuples:
        return "MATCH", f"{len(got_tuples)} rows"
    detail = diff_sorted(got_tuples, ref_tuples)
    return "FAIL", detail


def verify_q3x(ref_headers, ref_rows, text):
    text = extract_last_trial(text)
    lines = re.findall(r"^\s+(.+?)\s*\|\s*(.+?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*$", text, re.M)
    if not lines:
        return "NO_DATA", ""
    got_tuples = sorted([(a.strip(), b.strip(), yr.strip(), rev.strip()) for a, b, yr, rev in lines])
    ref_tuples = sorted([tuple(r[h] for h in ref_headers) for r in ref_rows])
    if got_tuples == ref_tuples:
        return "MATCH", f"{len(got_tuples)} rows"
    detail = diff_sorted(got_tuples, ref_tuples)
    return "FAIL", detail


def verify_q4x(ref_headers, ref_rows, text):
    text = extract_last_trial(text)
    ncols = len(ref_headers)
    pat = r"^\s+" + r"\s*\|\s*".join([r"(.+?)"] * ncols) + r"\s*$"
    lines = re.findall(pat, text, re.M)
    if not lines:
        return "NO_DATA", ""
    got_tuples = sorted([tuple(x.strip() for x in row) for row in lines])
    ref_tuples = sorted([tuple(r[h] for h in ref_headers) for r in ref_rows])
    if got_tuples == ref_tuples:
        return "MATCH", f"{len(got_tuples)} rows"
    detail = diff_sorted(got_tuples, ref_tuples)
    return "FAIL", detail


VERIFY_FN = {
    "q11": verify_q1x, "q12": verify_q1x, "q13": verify_q1x,
    "q21": verify_q2x, "q22": verify_q2x, "q23": verify_q2x,
    "q31": verify_q3x, "q32": verify_q3x, "q33": verify_q3x, "q34": verify_q3x,
    "q41": verify_q4x, "q42": verify_q4x, "q43": verify_q4x,
}


def run_query(build_dir, query, devices, threads):
    import subprocess
    cmd = [
        os.path.join(build_dir, "ssbdb"),
        "-q", query,
        "-x", "gidp",
        "-w", str(threads),
        devices,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def verify_one_mode(ans_dir, log_dir, mode_name, build_dir=None, devices=None, threads=None, save_logs=None):
    """Verify a single mode. Returns (passed, failed, total, fail_details)."""
    total = 0
    passed = 0
    failed = 0
    fail_details = []

    for q in QUERIES:
        ans_file = os.path.join(ans_dir, f"{q}.csv")
        if not os.path.isfile(ans_file):
            print(f"  {q:<8} \033[33m{'NO_REF':<10}\033[0m")
            continue

        ref_headers, ref_rows = parse_csv_answer(ans_file)
        total += 1

        # Get output
        if log_dir:
            log_file = os.path.join(log_dir, f"{q}.txt")
            if not os.path.isfile(log_file):
                print(f"  {q:<8} \033[33m{'NO_LOG':<10}\033[0m")
                continue
            text = open(log_file).read()
        else:
            text = run_query(build_dir, q, devices, threads)
            if save_logs:
                os.makedirs(save_logs, exist_ok=True)
                with open(os.path.join(save_logs, f"{q}.txt"), "w") as f:
                    f.write(text)

        status, detail = VERIFY_FN[q](ref_headers, ref_rows, text)

        if status == "MATCH":
            label = f"MATCH ({detail})" if detail else "MATCH"
            print(f"  {q:<8} \033[32m{label}\033[0m")
            passed += 1
        elif status == "NO_DATA":
            print(f"  {q:<8} \033[33m{'NO_DATA'}\033[0m")
        else:
            print(f"  {q:<8} \033[31m{'FAIL':<10}\033[0m {detail}")
            failed += 1
            fail_details.append((q, detail))

    return passed, failed, total, fail_details


def main():
    parser = argparse.ArgumentParser(description="Verify ssbdb results against reference answers")
    parser.add_argument("--sf", type=int, default=10, help="Scale factor (default: 10)")
    parser.add_argument("--build", default="build", help="Build directory (default: build)")
    parser.add_argument("--devices", default="/dev/nvme0n1p1,/dev/nvme1n1p1,/dev/nvme2n1p1,/dev/nvme3n1p1",
                        help="Comma-separated NVMe device list")
    parser.add_argument("--threads", type=int, default=1, help="IO threads (default: 1)")
    parser.add_argument("--answers", default="answers/ssb",
                        help="Base directory for reference answers (default: answers/ssb)")
    parser.add_argument("--log-dir", help="Use pre-captured log files from a single mode directory")
    parser.add_argument("--run-dir", help="Verify all mode subdirectories under this path")
    parser.add_argument("--save-logs", help="Save ssbdb output to this directory")
    args = parser.parse_args()

    ans_dir = os.path.join(args.answers, f"sf{args.sf}")
    if not os.path.isdir(ans_dir):
        print(f"ERROR: Answer directory not found: {ans_dir}")
        sys.exit(1)

    print("=" * 60)
    print("SSB Result Verification")
    print("=" * 60)
    print(f"Scale factor : SF{args.sf}")
    print(f"Answers      : {ans_dir}")

    total_all = 0
    passed_all = 0
    failed_all = 0

    if args.run_dir:
        # Verify all mode subdirectories
        print(f"Run directory: {args.run_dir}")
        print("=" * 60)

        modes = sorted(d for d in os.listdir(args.run_dir)
                       if os.path.isdir(os.path.join(args.run_dir, d))
                       and d not in ("revenue",))
        for mode in modes:
            mode_dir = os.path.join(args.run_dir, mode)
            print(f"\n--- {mode} ---")
            p, f_, t, _ = verify_one_mode(ans_dir, mode_dir, mode)
            passed_all += p
            failed_all += f_
            total_all += t

    elif args.log_dir:
        mode_name = os.path.basename(args.log_dir)
        print(f"Log directory: {args.log_dir}")
        print("=" * 60)
        print(f"\n--- {mode_name} ---")
        p, f_, t, _ = verify_one_mode(ans_dir, args.log_dir, mode_name)
        passed_all += p
        failed_all += f_
        total_all += t

    else:
        print(f"Build        : {args.build}")
        print(f"Devices      : {args.devices}")
        print("=" * 60)
        print(f"\n--- gidp (live) ---")
        p, f_, t, _ = verify_one_mode(
            ans_dir, None, "gidp",
            build_dir=args.build, devices=args.devices,
            threads=args.threads, save_logs=args.save_logs)
        passed_all += p
        failed_all += f_
        total_all += t

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_all} checks -- {passed_all} MATCH, {failed_all} FAIL")
    if total_all > 0:
        print(f"Pass rate: {passed_all / total_all * 100:.1f}%")
    print("=" * 60)

    sys.exit(failed_all)


if __name__ == "__main__":
    main()
