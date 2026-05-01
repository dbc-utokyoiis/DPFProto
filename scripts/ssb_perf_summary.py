#!/usr/bin/env python3
"""
ssb_perf_summary.py — Run SSB queries and generate a performance summary table.

Usage:
    # Run all queries (gidp, default)
    scripts/ssb_perf_summary.py

    # Run gidp+bam mode
    scripts/ssb_perf_summary.py --mode gidp+bam

    # Custom settings
    scripts/ssb_perf_summary.py --build build_ssb --threads 32 --sf 10

    # Summarize existing logs (skip execution)
    scripts/ssb_perf_summary.py --log-dir logs/ssb_gidp
    scripts/ssb_perf_summary.py --log-dir logs/ssb_gidp_bam

    # Save logs for later
    scripts/ssb_perf_summary.py --mode gidp+bam --save-dir logs/ssb_gidp_bam
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

QUERIES = ["q11", "q12", "q13", "q21", "q22", "q23",
           "q31", "q32", "q33", "q34", "q41", "q42", "q43", "revenue"]

FIELDS = ["time", "nios", "read_mb", "uncompressed_read_mb", "io_reduction_ratio",
          "effective_throughput_gbs", "io_throughput_gbs", "gpu_mem_mb"]


def extract_metrics(text):
    d = {}
    for key in FIELDS:
        m = re.search(rf"^{key}:\s*(.+)$", text, re.M)
        if m:
            d[key] = m.group(1).strip().replace(" msec", "")
    return d


def run_query(build, query, devices, threads, zonemap, mode):
    cmd = [os.path.join(build, "ssbdb"), "-q", query, "-x", mode,
           "-w", str(threads)]
    if zonemap:
        cmd.append("-Z")
    cmd.append(devices)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def load_or_run(args, query, zonemap):
    tag = "zm" if zonemap else "nozm"
    if args.log_dir:
        path = os.path.join(args.log_dir, f"{query}_{tag}.txt")
        if os.path.isfile(path):
            return open(path).read()
        return ""
    text = run_query(args.build, query, args.devices, args.threads, zonemap, args.mode)
    if args.save_dir:
        with open(os.path.join(args.save_dir, f"{query}_{tag}.txt"), "w") as f:
            f.write(text)
    return text


def fmt(d, key, width=6):
    return d.get(key, "-").rjust(width)


def main():
    p = argparse.ArgumentParser(description="SSB performance summary")
    p.add_argument("--mode", default="gidp",
                   choices=["gidp", "gidp+bam", "gidp+bam+fusion"],
                   help="Execution mode (default: gidp)")
    p.add_argument("--build", default="build_ssb")
    p.add_argument("--devices", default=None,
                   help="Device paths (auto-detected from mode if omitted)")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--sf", type=int, default=10)
    p.add_argument("--log-dir", help="Read existing logs instead of running")
    p.add_argument("--save-dir", help="Save query outputs to this directory")
    p.add_argument("-o", "--output", help="Write summary to file")
    args = p.parse_args()

    if args.devices is None:
        if args.mode in ("gidp+bam", "gidp+bam+fusion"):
            args.devices = "/dev/libnvm0,/dev/libnvm1,/dev/libnvm2,/dev/libnvm3"
        else:
            args.devices = "/dev/nvme0n1p1,/dev/nvme1n1p1,/dev/nvme2n1p1,/dev/nvme3n1p1"

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    hdr = (f"{'Query':<6} | {'No ZM (ms)':>11} {'IOs':>5} {'Read':>6} {'Uncomp':>6} {'Ratio':>6}"
           f" | {'ZM (ms)':>10} {'IOs':>5} {'Read':>6} {'Uncomp':>6} {'Ratio':>6}"
           f" | {'Speedup':>7} {'GPU MB':>6}")
    sep = (f"{'------':<6}-+-{'-'*11}-{'-'*5}-{'-'*6}-{'-'*6}-{'-'*6}"
           f"-+-{'-'*10}-{'-'*5}-{'-'*6}-{'-'*6}-{'-'*6}"
           f"-+-{'-'*7}-{'-'*6}")

    mode_label = args.mode if not args.log_dir else os.path.basename(args.log_dir)
    title = f"# SSB {mode_label} Performance — SF{args.sf}, {args.threads} threads"
    if args.log_dir:
        title += f", logs={args.log_dir}"
    title += f"  ({datetime.now().strftime('%Y-%m-%d %H:%M')})"

    lines = [title, "", hdr, sep]

    for q in QUERIES:
        label = f"  {q}" if not args.log_dir else ""
        if not args.log_dir:
            print(f"  Running {q}...", end="", flush=True, file=sys.stderr)

        nz = extract_metrics(load_or_run(args, q, zonemap=False))
        zm = extract_metrics(load_or_run(args, q, zonemap=True))

        if not args.log_dir:
            print(" done", file=sys.stderr)

        try:
            speedup = f"{float(nz['time']) / float(zm['time']):.2f}x"
        except (KeyError, ValueError, ZeroDivisionError):
            speedup = "-"

        gpu_mem = nz.get('gpu_mem_mb', '-')

        lines.append(
            f"{q:<6}"
            f" | {fmt(nz,'time',11)} {fmt(nz,'nios',5)} {fmt(nz,'read_mb')} {fmt(nz,'uncompressed_read_mb')} {fmt(nz,'io_reduction_ratio')}"
            f" | {fmt(zm,'time',10)} {fmt(zm,'nios',5)} {fmt(zm,'read_mb')} {fmt(zm,'uncompressed_read_mb')} {fmt(zm,'io_reduction_ratio')}"
            f" | {speedup:>7} {gpu_mem:>6}"
        )

    out = "\n".join(lines)
    print(out)

    if args.output:
        with open(args.output, "w") as f:
            f.write(out + "\n")
        print(f"\nSaved to {args.output}", file=sys.stderr)
    elif args.save_dir:
        path = os.path.join(args.save_dir, "summary.txt")
        with open(path, "w") as f:
            f.write(out + "\n")
        print(f"\nSaved to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
