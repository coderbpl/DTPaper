#!/usr/bin/env bash
# deploy/setup_aws.sh
# Provision an AWS Linux (Amazon Linux 2023 or Ubuntu 22.04) instance to run
# DarkTrace Phase 1. Idempotent: safe to re-run.
#
# Usage on a fresh EC2 instance:
#   git clone https://github.com/coderbpl/DTPaper.git
#   cd DTPaper
#   bash deploy/setup_aws.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT_DIR}/.venv"

echo "[1/4] Installing system packages..."
if command -v dnf >/dev/null 2>&1; then
  sudo dnf -y install python3 python3-pip git
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3 python3-pip python3-venv git
else
  echo "Unsupported package manager. Install python3, pip, git manually." >&2
  exit 1
fi

echo "[2/4] Creating virtual environment..."
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

echo "[4/4] Running smoke test to verify the environment..."
cd "${PROJECT_DIR}"
python tests/test_smoke.py

echo
echo "Setup complete. To run the pipeline:"
echo "  source ${VENV}/bin/activate"
echo "  python -m src.exp_traffic --config configs/traffic.json"
echo "  python -m src.exp_text    --config configs/text.json"
echo "  python -m src.build_table6"
echo
echo "Place real datasets in data/raw/ (see DATASETS.md) for reportable results."
