#!/usr/bin/env bash
# deploy/aws_phase1.sh
# AWS Linux (Amazon Linux 2023 / Ubuntu on EC2) orchestrator for DarkTrace Phase 1.
#
# Subcommands (run them in order):
#   setup        install OS deps, create venv, install requirements, smoke-test
#   check-data   verify the real datasets are present at the expected paths
#   coda         export CoDA from Hugging Face to data/raw/coda.csv
#   run          run traffic + text experiments in REAL-DATA mode (fails if data missing)
#   table        build Table 6 + figures from real-data outputs
#   verify       confirm outputs are real (not SMOKETEST)
#   all          check-data -> run -> table -> verify  (does NOT place datasets for you)
#   smoke        run the synthetic smoke test only (NON-reportable)
#
# Usage:
#   bash deploy/aws_phase1.sh setup
#   bash deploy/aws_phase1.sh run
#   bash deploy/aws_phase1.sh all
#
# Notes:
# - You must place CIC-Darknet2020.csv yourself (licensed download; use scp/S3).
# - This script never substitutes synthetic data in normal mode.
set -euo pipefail

# resolve project root (parent of this script's dir) and cd into it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

VENV="${PROJECT_DIR}/.venv"
PY="${VENV}/bin/python"
TRAFFIC_CSV="data/raw/CIC-Darknet2020.csv"
TEXT_CSV="data/raw/coda.csv"

c_red()   { printf "\033[31m%s\033[0m\n" "$*"; }
c_grn()   { printf "\033[32m%s\033[0m\n" "$*"; }
c_ylw()   { printf "\033[33m%s\033[0m\n" "$*"; }
hr()      { printf -- "----------------------------------------------------------\n"; }

detect_pm() {
  if command -v dnf >/dev/null 2>&1; then echo "dnf";
  elif command -v apt-get >/dev/null 2>&1; then echo "apt";
  else echo "none"; fi
}

cmd_setup() {
  hr; c_grn "[setup] Installing OS packages"; hr
  local pm; pm="$(detect_pm)"
  case "${pm}" in
    dnf) sudo dnf -y install python3 python3-pip git unzip ;;
    apt) sudo apt-get update -y && sudo apt-get install -y python3 python3-pip python3-venv git unzip ;;
    *)   c_red "No supported package manager (dnf/apt). Install python3/pip/venv manually."; exit 1 ;;
  esac

  hr; c_grn "[setup] Creating virtual environment"; hr
  if [ ! -d "${VENV}" ]; then
    python3 -m venv "${VENV}" || {
      c_ylw "venv module missing; attempting virtualenv fallback"
      python3 -m pip install --user virtualenv
      python3 -m virtualenv "${VENV}"
    }
  fi

  hr; c_grn "[setup] Installing Python dependencies"; hr
  "${VENV}/bin/pip" install --upgrade pip
  "${VENV}/bin/pip" install -r requirements.txt

  hr; c_grn "[setup] Running smoke test (synthetic, non-reportable)"; hr
  "${PY}" tests/test_smoke.py
  c_grn "[setup] Done. Activate later with: source ${VENV}/bin/activate"
}

require_venv() {
  if [ ! -x "${PY}" ]; then
    c_red "Virtual environment not found. Run: bash deploy/aws_phase1.sh setup"
    exit 1
  fi
}

cmd_check_data() {
  hr; c_grn "[check-data] Verifying real datasets"; hr
  local ok=0
  if [ -f "${TRAFFIC_CSV}" ]; then c_grn "  found ${TRAFFIC_CSV}"; else
    c_red "  MISSING ${TRAFFIC_CSV}"; ok=1
    c_ylw "    Download CIC-Darknet2020 from the UNB CIC page, then copy it up, e.g.:"
    c_ylw "    scp -i key.pem ~/Downloads/Darknet.csv USER@IP:~/darktrace_phase1/${TRAFFIC_CSV}"
  fi
  if [ -f "${TEXT_CSV}" ]; then c_grn "  found ${TEXT_CSV}"; else
    c_red "  MISSING ${TEXT_CSV}"; ok=1
    c_ylw "    Export CoDA on the instance:  bash deploy/aws_phase1.sh coda"
    c_ylw "    (or place a DUTA-10K CSV at ${TEXT_CSV})"
  fi
  if [ "${ok}" -ne 0 ]; then
    c_red "[check-data] One or more datasets missing. Place them, then re-run."
    exit 2
  fi
  c_grn "[check-data] All required datasets present."
}

cmd_coda() {
  require_venv
  hr; c_grn "[coda] Exporting s2w-ai/CoDA from Hugging Face"; hr
  "${VENV}/bin/pip" install datasets >/dev/null
  "${PY}" - <<'PY'
from datasets import load_dataset
ds = load_dataset("s2w-ai/CoDA")
split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
df = split.to_pandas()
print("columns:", list(df.columns))
df.to_csv("data/raw/coda.csv", index=False)
print("wrote data/raw/coda.csv", df.shape)
PY
  c_ylw "[coda] If the printed columns are not auto-detected, set text_col/label_col in configs/text.json"
}

cmd_run() {
  require_venv
  cmd_check_data
  hr; c_grn "[run] Traffic classification (real data)"; hr
  "${PY}" -m src.exp_traffic --config configs/traffic.json
  hr; c_grn "[run] Text classification (real data)"; hr
  "${PY}" -m src.exp_text --config configs/text.json
  c_grn "[run] Real-data experiments complete."
}

cmd_table() {
  require_venv
  hr; c_grn "[table] Building Table 6 + figures"; hr
  "${PY}" -m src.build_table6
  "${PY}" -m src.make_figures || c_ylw "[table] figures skipped (matplotlib missing?)"
}

cmd_verify() {
  hr; c_grn "[verify] Checking outputs are REAL, not SMOKETEST"; hr
  local fail=0
  if ls results/tables/ 2>/dev/null | grep -q SMOKETEST; then
    c_ylw "  note: SMOKETEST files exist in results/tables (ignored by Table 6):"
    ls results/tables/ | grep SMOKETEST | sed 's/^/    /'
  fi
  if [ -f results/tables/table6_full.csv ]; then
    if grep -iq "SMOKE-TEST\|NON-REPORTABLE\|SYNTHETIC" results/tables/table6_full.csv; then
      c_red "  table6_full.csv CONTAINS non-reportable rows!"; fail=1
    else
      c_grn "  table6_full.csv has no non-reportable markers."
    fi
  else
    c_red "  table6_full.csv not found. Run: bash deploy/aws_phase1.sh table"; fail=1
  fi
  for j in results/tables/traffic_results.json results/tables/text_results.json; do
    if [ -f "${j}" ]; then
      if grep -q '"synthetic": false' "${j}" && grep -q '"reportable": true' "${j}"; then
        c_grn "  ${j}: real-data (synthetic=false, reportable=true)"
      else
        c_red "  ${j}: NOT real-data (check synthetic/reportable flags)"; fail=1
      fi
    else
      c_ylw "  ${j} not found (run the experiment on real data)."
    fi
  done
  if [ "${fail}" -ne 0 ]; then c_red "[verify] FAILED — see above."; exit 1; fi
  c_grn "[verify] PASS — outputs are real-data and reportable."
}

cmd_smoke() {
  require_venv
  hr; c_ylw "[smoke] Synthetic smoke test (NON-reportable)"; hr
  "${PY}" -m src.exp_traffic --config configs/traffic.json --smoke-test
  "${PY}" -m src.exp_text    --config configs/text.json --smoke-test
  c_ylw "[smoke] Done. Outputs are *_SMOKETEST and excluded from Table 6."
}

cmd_all() {
  require_venv
  cmd_check_data
  cmd_run
  cmd_table
  cmd_verify
  c_grn "[all] Complete. Results in results/tables and results/figures."
}

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

main() {
  local sub="${1:-}"
  case "${sub}" in
    setup)      cmd_setup ;;
    check-data) cmd_check_data ;;
    coda)       cmd_coda ;;
    run)        cmd_run ;;
    table)      cmd_table ;;
    verify)     cmd_verify ;;
    smoke)      cmd_smoke ;;
    all)        cmd_all ;;
    ""|-h|--help|help) usage ;;
    *) c_red "Unknown subcommand: ${sub}"; usage; exit 1 ;;
  esac
}
main "$@"
