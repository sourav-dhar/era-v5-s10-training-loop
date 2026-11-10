#!/usr/bin/env bash
# Pull a finished Modal run (runs/t4, runs/a100) into the repo, feed the companion page, and regenerate the README.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f runs/t4/training_loop.full.ipynb || { echo "no finished T4 run in runs/t4"; exit 1; }

cp runs/t4/training_loop.full.ipynb training_loop.ipynb          # the executed notebook is the deliverable
mkdir -p figures results docs/data docs/figures
cp runs/t4/figures/*.png figures/
cp runs/t4/results/*.json results/
if [ -d runs/a100 ]; then cp runs/a100/results/mfu_A100.json results/; cp runs/a100/figures/05_mfu_A100.png figures/; fi

cp results/*.json docs/data/
cp figures/*.png docs/figures/
python tools/make_readme.py 2>/dev/null || py tools/make_readme.py
echo "finalized: $(ls figures | wc -l) figures, $(ls results | wc -l) result files"
