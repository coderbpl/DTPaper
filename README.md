# DarkTrace — Phase 1 Implementation

Runnable Phase 1 of the DarkTrace experimental plan: **CPU-only** darknet traffic
classification (CIC-Darknet2020) and dark web text classification (CoDA/DUTA).
No GPU, blockchain, or actor-profiling code is included here — those are later
phases.

**Reportable results require real data.** In normal mode the experiments run only
on the real datasets and **fail hard** if a required file is missing. Synthetic
data exists solely to verify the pipeline and is gated behind a `--smoke-test`
flag; smoke-test outputs are written to separate `*_SMOKETEST.*` files, marked
NON-REPORTABLE, and excluded from `build_table6`.

> Place the real datasets per `DATASETS.md` before producing Table 6.

## Folder structure

```
darktrace_phase1/
├── README.md                 # this file
├── DATASETS.md               # how to obtain & place CIC-Darknet2020 and CoDA/DUTA
├── KAGGLE_SETUP.md           # Kaggle notebook setup and runtime guidance
├── requirements.txt          # CPU-only deps (Phase 2+ deps commented out)
├── configs/
│   ├── traffic.json/.yaml    # traffic experiment config
│   └── text.json/.yaml       # text experiment config
├── data/
│   ├── raw/                  # place real datasets here (gitignored)
│   └── processed/            # cached intermediate artefacts
├── src/
│   ├── utils.py              # config, logging, seeding
│   ├── metrics_stats.py      # metrics + McNemar/bootstrap/Holm (Section 8.13)
│   ├── exp_traffic.py        # Experiment A: CIC-Darknet2020 (Section 8.7)
│   ├── exp_text.py           # Experiment B: CoDA/DUTA text (Section 8.7)
│   ├── build_table6.py       # assembles manuscript Table 6
│   └── make_figures.py       # confusion / per-class F1 figures (Section 8.18)
├── tests/
│   └── test_smoke.py         # end-to-end pipeline smoke test
└── results/
    ├── tables/               # JSON + CSV outputs (Table 6 fragments + full)
    ├── figures/              # PNG figures
    └── logs/                 # run logs
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only `numpy pandas scipy scikit-learn` are strictly required; `imbalanced-learn`
enables SMOTE, `matplotlib` enables figures, `pyyaml`/`pytest` are optional.

## Kaggle

Use `KAGGLE_SETUP.md` for a Kaggle notebook runbook, including private Kaggle
Dataset placement, CPU/GPU settings, and smoke-test vs reportable runs. Phase 1
does not need a GPU. The traffic boosted-tree baseline defaults to sklearn's
`HistGradientBoostingClassifier` for faster Kaggle runs; set
`model.gbt_variant` to `"legacy"` in `configs/traffic.json` to use the original
`GradientBoostingClassifier`.

## Quickstart

**Verify the pipeline (smoke test — synthetic, NON-reportable):**
```bash
# from darktrace_phase1/
python tests/test_smoke.py
python -m src.exp_traffic --config configs/traffic.json --smoke-test
python -m src.exp_text    --config configs/text.json --smoke-test
```
Smoke-test outputs are written as `*_SMOKETEST.*` and are never used for Table 6.

**Produce reportable results (real data — required):**
1. Follow `DATASETS.md` to place `data/raw/CIC-Darknet2020.csv` and
   `data/raw/coda.csv` (CoDA from Hugging Face `s2w-ai/CoDA`, or DUTA-10K).
2. Run:
```bash
python -m src.exp_traffic --config configs/traffic.json
python -m src.exp_text    --config configs/text.json
python -m src.build_table6
python -m src.make_figures
```
If a required dataset is missing, normal mode **fails with exit code 2** and writes
no Table 6 output.

## What each experiment does (and how it maps to the manuscript)

| Script | Manuscript section | Populates | Protocol |
|---|---|---|---|
| `exp_traffic.py` | 8.7 (traffic) | Table 6 traffic rows | 5-fold stratified CV, SMOTE-in-fold [25], held-out test, bootstrap CI, McNemar [26] |
| `exp_text.py` | 8.7 (text) | Table 6 text rows | TF-IDF + LogReg/LinearSVM baselines [23], same CV/test protocol |
| `build_table6.py` | 8.17 | Table 6 (full) | merges fragments + cited prior-work rows [2],[4] |
| `make_figures.py` | 8.18 | per-class F1 / confusion figs | minority-class behaviour |
| `metrics_stats.py` | 8.13, 8.16 | all metrics + significance | McNemar, paired-fold t/Wilcoxon, bootstrap, Holm–Bonferroni |

## Reproducibility

- Fixed seed (`seed: 42`) across NumPy/Python.
- Stratified splits; held-out test fixed before CV (no leakage).
- SMOTE applied **inside training folds only** (manuscript Section 8.3).
- Identifier/leakage columns dropped from traffic data (per [3]).
- All configs, seeds, and outputs are written under `results/`.

## Scope guardrails

- **In Phase 1:** CPU-only classical ML; public datasets; full statistics.
- **Not in Phase 1:** transformer fine-tuning (GPU), NER, SHAP/LIME, blockchain,
  actor profiling, SIEM — these are Phases 2–5 (see `PHASED_PLAN`).
