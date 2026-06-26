# DarkTrace — Complete Run Guide

Real-data-only experiment package for the DarkTrace dark-web monitoring study.
Every result is computed from real data; there are no synthetic fallbacks.

## What's inside

| Phase | Module | Output | Needs |
|---|---|---|---|
| 1 traffic | `exp_traffic` | Table 6 (traffic) | CIC-Darknet2020 |
| 1 text | `exp_text` | Table 6 (text) | CoDA |
| 2 multilingual | `exp_multilingual` | Table 7 | CoDA (GPU) |
| 2b cross-domain | `exp_multilingual_crossdomain` | Table 7b | Hindi + Arabic corpora (GPU) |
| 3 explainable scoring | `exp_scoring` | Table 10 | CoDA |
| 4 evidence sealing | `exp_blockchain` | Table 8 | CoDA |
| 5 STIX export | `exp_stix` | Table 9 | CoDA |
| integration ablation | `exp_ablation` | Table 11 | CoDA |

## 0. Setup (once per Kaggle session)

```bash
%cd /kaggle/working/DTPaper
!pip install -q transformers torch shap lime stix2 tabulate scipy
# Kaggle: Settings -> Accelerator -> GPU ON ; Settings -> Internet -> ON
```

## 1. Datasets

Place the core datasets (see DATASETS.md):
- `data/raw/CIC-Darknet2020.csv`
- `data/raw/coda.csv`

Cross-domain Hindi/Arabic (optional but recommended; see PHASE2B.md):
```bash
!python download_corpora.py
# -> data/raw/hindi_hostility.csv  and  data/raw/arabic_osact.csv
```
If the Arabic source is unreachable, the script says so; Phase 2b then runs
Hindi alone and skips Arabic (it never fabricates data).

## 2. Run everything (one command)

```bash
!python run_all.py
```

This runs all phases in order, then figures and the consolidated tables, and
prints a timed PASS/FAIL summary. A phase whose dataset is missing FAILS clearly
(core phases) or is SKIPPED (the optional cross-domain phase). To skip slow
phases: `!python run_all.py --skip exp_multilingual exp_ablation`.

### Or run phases individually

```bash
!python -m src.exp_traffic                 --config configs/traffic.json
!python -m src.exp_text                    --config configs/text.json
!python -m src.exp_multilingual            --config configs/multilingual.json          # GPU
!python -m src.exp_multilingual_crossdomain --config configs/multilingual_crossdomain.json  # GPU
!python -m src.exp_scoring                 --config configs/scoring.json
!python -m src.exp_blockchain              --config configs/blockchain.json
!python -m src.exp_stix                    --config configs/stix.json
!python -m src.exp_ablation                --config configs/ablation.json
!python -m src.make_figures
!python -m src.make_scoring_figures
!python -m src.build_all_tables
```

## 3. View + download results

```bash
!cat results/tables/ALL_RESULTS.md          # consolidated Tables 6-11
!ls -R results/tables results/figures
import shutil; shutil.make_archive('/kaggle/working/darktrace_results','zip','results')
# download darktrace_results.zip from the Kaggle output panel
```

`darktrace_results.zip` contains every table (JSON + CSV), figure, and run log —
the complete evidence set for the manuscript.

## 4. Sanity checks to read in the logs

- Phase 2: confirm `model_path: transformer` (not `tfidf_fallback`) — else GPU/
  transformers not active and the result is not reportable.
- Phase 2b: check the `after mapping` line shows a sensible threat/benign split;
  if almost everything was dropped, the label format differs — inspect the file.
- Ablation: `full` actionability should be the highest of all configurations and
  each ablated config should be lower (the dependency cascade).

## Honesty notes baked into the package (for the manuscript)

- Phase 3 risk labels are **category-derived**, not analyst-rated — state this.
- Phase 4 is a **local SHA-256 hash chain**, not a distributed/permissioned ledger.
- Phase 5 generates and validates **STIX 2.1 bundles**; TAXII push/SIEM not deployed.
- Phase 2b is **cross-domain** (social-media Hindi/Arabic), NOT dark-web data —
  report separately from Table 7; native dark-web HI/AR data does not exist at scale.
- The ablation measures **property survival**, not accuracy — integration does not
  change accuracy by design.
