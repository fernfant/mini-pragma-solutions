#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== Stage 1: generate data ==="
python3 capstone_data.py
echo
echo "=== Stage 2: pre-train encoder ==="
python3 capstone_pretrain.py
echo
echo "=== Stage 3+4: classify + baseline ==="
python3 capstone_classify.py
