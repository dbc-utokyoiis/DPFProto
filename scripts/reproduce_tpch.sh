#!/usr/bin/env bash
# ------------------------------------------------------------
# reproduce_tpch.sh — convenience shortcut for the TPC-H
# experiments in the paper.
#
# For artifact reviewers: the README §6 ("Reproducing the
# paper's experiments") is the authoritative reproduction
# guide. This file only stitches together the same commands.
#
# Requires sudo (BaM mode switches the NVMe driver; see README §3).
# ------------------------------------------------------------
set -euo pipefail

scripts/tpch_run_all.sh -s 100
scripts/tpch_run_all.sh -s 50,200,300 -q q3,q13 --no-revenue
scripts/tpch_chunk_size_sweep.sh
