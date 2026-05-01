#!/usr/bin/env bash
# GPU discovery script for Spark resource manager
ADDRS=$(nvidia-smi --query-gpu=index --format=csv,noheader | sed 's/^ */"/;s/ *$/"/;' | paste -sd, -)
echo "{\"name\": \"gpu\", \"addresses\": [${ADDRS}]}"
