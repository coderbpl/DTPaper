# Running DarkTrace Phase 1 on Kaggle

Kaggle Notebooks are a good fit for Phase 1: the scientific stack is preinstalled,
you get more RAM than a small EC2 box, and datasets can be attached once and reused
offline. Phase 1 is **CPU-only** (RandomForest, gradient boosting, TF-IDF), so you do
**not** need to enable the GPU — leave the accelerator off. (GPU only matters for the
Phase 2 transformer work later.)

> Environment note: Kaggle's pandas may default to the **PyArrow string backend**,
> which previously caused `TypeError: only integer scalar arrays can be converted to
> a scalar index` in `train_test_split`. The package now forces the classic NumPy
> backend at import and converts columns explicitly, so this is handled for you.

---

## Option A — run from the GitHub repo (quickest)

In a new Kaggle Notebook cell:

```python
# 1. clone your repo
!git clone https://github.com/coderbpl/DTPaper.git
%cd DTPaper

# 2. install the few extra deps Kaggle may not have
!pip install -q imbalanced-learn

# 3. verify the pipeline (synthetic, NON-reportable)
!python tests/test_smoke.py
```

You should see `ALL SMOKE TESTS PASSED`.

## Option B — upload the package as a Kaggle Dataset

1. Zip the `darktrace_phase1` folder and upload it via **+ Add Data → Upload**.
2. Attach it; it mounts read-only under `/kaggle/input/<your-dataset-name>/`.
3. Copy it into the writable working dir so outputs can be saved:

```python
!cp -r /kaggle/input/<your-dataset-name>/darktrace_phase1 /kaggle/working/
%cd /kaggle/working/darktrace_phase1
!pip install -q imbalanced-learn
!python tests/test_smoke.py
```

---

## Placing the real datasets on Kaggle

Reportable results require the **real** datasets (normal mode fails hard without
them). The clean Kaggle pattern is to attach each dataset and copy/point to it.

### CIC-Darknet2020 (traffic)
- Search Kaggle Datasets for "CIC-Darknet2020" and **+ Add Data**, or upload the CSV
  yourself from the UNB CIC download.
- It mounts at `/kaggle/input/<cic-dataset>/...`. Point the config at it:

```python
import json, pathlib
cfg = json.load(open("configs/traffic.json"))
cfg["data"]["csv_path"] = "/kaggle/input/<cic-dataset>/CIC-Darknet2020.csv"
json.dump(cfg, open("configs/traffic.json","w"), indent=2)
```

(or copy the file to `data/raw/CIC-Darknet2020.csv`).

### CoDA (text) — from Hugging Face
Kaggle Notebooks allow internet (toggle **Internet: On** in the sidebar):

```python
!pip install -q datasets
import pandas as pd
from datasets import load_dataset
ds = load_dataset("s2w-ai/CoDA")
split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
df = split.to_pandas()
print("columns:", list(df.columns))          # check text/label column names
df.to_csv("data/raw/coda.csv", index=False)
```

If auto-detection misses the columns, set `text_col`/`label_col` in
`configs/text.json`.

> **CoDA label handling (automatic).** `s2w-ai/CoDA` is a WebDataset, so
> `to_pandas()` returns columns like `['__key__', '__url__', 'txt']` where each key
> looks like `coda_dataset/{id}-{Category}-{lang}-{hash}` (e.g.
> `coda_dataset/5756-Arms-en-06dd...`). The loader now parses this automatically:
> it uses `txt` as the text column and extracts the **category** (Arms, Financial,
> Gambling, Porn, Violence, Others, ...) as the label, and also captures the
> **language** (`en`, `ru`, `zh`, ...) into a `__lang__` column for later
> multilingual work. You'll see a log line: `Parsed CoDA __key__ format: N
> categories ...`. No manual config needed for this export. Verify the parsed
> categories in that log line look right before trusting results.

### CIC-Darknet2020 ragged rows
Real CIC exports sometimes contain malformed rows (`Expected N fields, saw M`). The
loader now retries with `on_bad_lines='skip'` and logs how many rows were dropped.
If the dropped fraction is large, inspect the CSV — you may have concatenated shards
or a wrong delimiter.

### DUTA-10K (alternative text corpus)
Upload your DUTA export as a Kaggle Dataset, then point `configs/text.json`'s
`data.csv_path` at it (and set `data.name` to `DUTA-10K`).

---

## Run in real-data mode (reportable)

```python
!python -m src.exp_traffic --config configs/traffic.json
!python -m src.exp_text    --config configs/text.json
!python -m src.build_table6
!python -m src.make_figures
```

Watch for `Real-data run complete — Table 6 ... fragment written.` A missing dataset
in normal mode exits with code 2 (intentional; never falls back to synthetic).

## Verify outputs are REAL, not SMOKETEST

```python
!ls results/tables/ | grep SMOKETEST            # expect: no output
!grep -i "SMOKE-TEST\|NON-REPORTABLE\|SYNTHETIC" results/tables/table6_full.csv || echo "clean"
!grep -E '"synthetic"|"reportable"' results/tables/traffic_results.json results/tables/text_results.json
```

Real runs show `"synthetic": false` and `"reportable": true`.

---

## Speeding up gradient boosting on Kaggle

sklearn's `GradientBoostingClassifier` is single-threaded and slow on the full
~158k-row CIC-Darknet2020. Use the histogram-based booster instead, which is
multi-threaded and typically 10–50× faster with comparable accuracy:

- If your `configs/traffic.json` exposes a model choice, set it to
  `HistGradientBoosting`.
- LightGBM is also preinstalled on Kaggle if you prefer it.

This change affects only training time, not the experimental protocol (still
stratified 5-fold CV, SMOTE in-fold, held-out test, bootstrap CIs, McNemar).

## Saving results

Anything written under `/kaggle/working/` (including `results/`) is saved with the
notebook output and can be downloaded, or committed back to the repo. To download:
**Notebook → Output → Download**.

## Why not just reuse the numbers from the cited papers?

You cannot. The figures in your comparison table from DarknetSec, the stacking
ensemble, etc., are those papers' results on their own setups; they appear in your
manuscript only as clearly-marked "(reported)" prior-work rows. Your own Table 6
rows and — critically — the ablation study that defends the integration novelty are
measurements only **your** pipeline can produce. Reproducing them yourself is what
makes the results reportable and the novelty defensible.
