#!/usr/bin/env bash
# deploy/run_all.sh
# Activate the venv and run the full Phase 1 pipeline, writing all outputs to results/.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/.venv/bin/activate"

echo "Running traffic classification..."
python -m src.exp_traffic --config configs/traffic.json
echo "Running text classification..."
python -m src.exp_text --config configs/text.json
echo "Assembling Table 6..."
python -m src.build_table6
echo "Generating figures..."
python -m src.make_figures || echo "(figures skipped: matplotlib not installed)"
echo "Done. See results/tables/ and results/figures/."
