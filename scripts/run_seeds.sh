#!/usr/bin/env bash
# Train one configuration once per seed; extra arguments are passed to train.py.
# Usage: SEEDS="42 43 44" scripts/run_seeds.sh --model hybrid --dataset kvasir_seg
set -euo pipefail

SEEDS="${SEEDS:-42 43 44}"

for seed in ${SEEDS}; do
    python train.py --seed "${seed}" "$@"
done
