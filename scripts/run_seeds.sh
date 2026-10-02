#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-42 43 44}"

for seed in ${SEEDS}; do
    python train.py --seed "${seed}" "$@"
done
