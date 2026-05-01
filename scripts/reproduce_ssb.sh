#!/usr/bin/env bash
# ------------------------------------------------------------
# reproduce_ssb.sh — convenience shortcut for the SSB
# experiments in the paper.
#
# For artifact reviewers: the README §6 ("Reproducing the
# paper's experiments") is the authoritative reproduction
# guide. This file only stitches together the same commands.
#
# Requires sudo (BaM mode switches the NVMe driver; see README §3).
# ------------------------------------------------------------
set -euo pipefail

scripts/ssb_run_all.sh -s 100
scripts/ssb_run_all.sh -s 50,200,300 -q q11,q21,q31 --skip-revenue-sweep
scripts/ssb_chunk_size_sweep.sh
