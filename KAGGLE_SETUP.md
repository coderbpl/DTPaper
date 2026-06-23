# Kaggle Setup - DarkTrace Phase 1

This repo's Phase 1 experiments are CPU-only classical ML:

- Traffic: CIC-Darknet2020 with Random Forest plus a boosted-tree baseline.
- Text: CoDA or DUTA-style CSV with TF-IDF + Logistic Regression / Linear SVM.

The normal runs require real data. If `data/raw/CIC-Darknet2020.csv` or
`data/raw/coda.csv` is missing, the scripts fail instead of silently using
synthetic data. Use `--smoke-test` only to verify the notebook pipeline.

## Are the three setup points necessary?

| Point | Necessary? | Recommendation |
|---|---:|---|
| Host datasets as Kaggle Datasets vs pull CoDA from Hugging Face each run | Yes for CIC, recommended for CoDA | Use private Kaggle Datasets for both real CSVs when possible. It is faster, repeatable, and can run with notebook internet off. Pull CoDA from Hugging Face only when you are still preparing `coda.csv` and can enable internet. |
| Replace classic `GradientBoostingClassifier` | Not required for correctness, recommended for Kaggle runtime | This repo now defaults to sklearn `HistGradientBoostingClassifier` through `model.gbt_variant: "hist"`. Set `"legacy"` to reproduce the original slower sklearn GBT, or `"lightgbm"` if LightGBM is installed and you want to try it. |
| Use Kaggle GPU for Phase 1 | No | Set Accelerator to `None` for Phase 1. GPU only matters for Phase 2 transformer work such as BERT/RoBERTa/DarkBERT fine-tuning. |

## Recommended Kaggle settings

In the notebook settings panel:

- Accelerator: `None` for Phase 1.
- Internet: `On` if cloning this repo or pulling CoDA from Hugging Face.
- Internet: `Off` is fine after the repo and data are attached as Kaggle Datasets.
- Persistence: results are written under `/kaggle/working/DTPaper/results`.

If your local changes are not pushed to GitHub yet, upload this checkout as a
private Kaggle Dataset or notebook file instead of cloning the public repo.

## Dataset preparation

Create private Kaggle Datasets for the real data unless the dataset license says
not to redistribute, even privately. Respect the UNB CIC and CoDA/DUTA terms.

Expected final paths inside the repo:

```text
DTPaper/
  data/raw/CIC-Darknet2020.csv
  data/raw/coda.csv
```

### CIC-Darknet2020

1. Download the CIC-Darknet2020 CSV from the UNB CIC page listed in
   `DATASETS.md`.
2. Rename it to `CIC-Darknet2020.csv`.
3. Upload it as a private Kaggle Dataset, for example `cic-darknet2020`.
4. Attach that dataset to the notebook.

### CoDA

Best repeatable option:

1. Export CoDA once to a CSV with a text column and a label column.
2. Name it `coda.csv`.
3. Upload it as a private Kaggle Dataset, for example `coda-darkweb`.
4. Attach it to the notebook.

Internet-enabled one-time export option:

```python
from pathlib import Path
from datasets import load_dataset

Path("data/raw").mkdir(parents=True, exist_ok=True)
ds = load_dataset("s2w-ai/CoDA")
split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
df = split.to_pandas()

print(df.columns)
# If needed, rename columns before saving:
# df = df.rename(columns={"your_text_column": "text", "your_label_column": "label"})

df.to_csv("data/raw/coda.csv", index=False)
```

## Notebook cells

### 1. Get the code

If internet is enabled and the desired changes are already pushed:

```python
!git clone https://github.com/coderbpl/DTPaper.git
%cd /kaggle/working/DTPaper
```

If internet is off, attach a Kaggle Dataset that contains this repo and copy it:

```python
!cp -R /kaggle/input/<your-dtpaper-dataset>/DTPaper /kaggle/working/DTPaper
%cd /kaggle/working/DTPaper
```

### 2. Install dependencies

```python
!python -m pip install -q -r requirements.txt
```

If internet is off and pip cannot install an optional package, use Kaggle's
preinstalled packages where possible. For Phase 1, the core requirement is
`numpy`, `pandas`, `scipy`, and `scikit-learn`; `imbalanced-learn` enables SMOTE,
`matplotlib` writes figures, and `datasets` is only needed for the Hugging Face
export path.

### 3. Copy attached Kaggle datasets into the expected paths

Replace the placeholder folder names with your attached dataset slugs:

```python
from pathlib import Path
import shutil

raw = Path("data/raw")
raw.mkdir(parents=True, exist_ok=True)

shutil.copyfile(
    "/kaggle/input/<your-cic-dataset>/CIC-Darknet2020.csv",
    raw / "CIC-Darknet2020.csv",
)
shutil.copyfile(
    "/kaggle/input/<your-coda-dataset>/coda.csv",
    raw / "coda.csv",
)

print(sorted(p.name for p in raw.iterdir()))
```

If you do not know the attached file paths yet:

```python
!find /kaggle/input -maxdepth 4 -type f | sort
```

### 4. Smoke test

This verifies the code path only. Results from this cell are non-reportable.

```python
!python tests/test_smoke.py
!python -m src.exp_traffic --config configs/traffic.json --smoke-test
!python -m src.exp_text --config configs/text.json --smoke-test
```

### 5. Run reportable Phase 1 experiments

```python
!python -m src.exp_traffic --config configs/traffic.json
!python -m src.exp_text --config configs/text.json
!python -m src.build_table6
!python -m src.make_figures
```

Outputs:

```text
results/tables/traffic_results.json
results/tables/text_results.json
results/tables/table6_traffic.csv
results/tables/table6_text.csv
results/tables/table6_full.csv
results/figures/*.png
results/logs/*.log
```

Quick inspection:

```python
!ls -lh results/tables results/figures results/logs
!cat results/tables/table6_full.csv
```

## Faster boosted-tree options

The traffic config now controls the boosted-tree implementation:

```json
"model": {
  "rf_trees": 200,
  "gbt_variant": "hist",
  "gbt_max_iter": 100,
  "gbt_learning_rate": 0.1,
  "gbt_max_leaf_nodes": 31
}
```

Valid `gbt_variant` values:

- `"hist"`: default. Uses sklearn `HistGradientBoostingClassifier`; fastest
  dependency-light option for Kaggle.
- `"legacy"`: original sklearn `GradientBoostingClassifier`; slower, useful for
  exact comparison with older runs.
- `"lightgbm"`: uses `lightgbm.LGBMClassifier` if available; optional and not
  required for Phase 1.

For a quick debug run only, reduce the workload:

```python
import json
from pathlib import Path

path = Path("configs/traffic.json")
cfg = json.loads(path.read_text())
cfg["cv_folds"] = 3
cfg["bootstrap"] = 100
cfg["model"]["rf_trees"] = 50
path.write_text(json.dumps(cfg, indent=2))
```

Do not use reduced folds/bootstrap for final reportable Table 6 numbers unless
the manuscript protocol is updated to match.

## Troubleshooting

- Exit code `2`: a real dataset file is missing. Check `data/raw/`.
- `Could not find a text column`: set `data.text_col` in `configs/text.json`.
- `Could not find a label column`: set `data.label_col` in `configs/text.json`.
- Very slow traffic run: keep `gbt_variant: "hist"` and avoid the `"legacy"`
  variant on Kaggle.
- GPU shows no speedup: expected for Phase 1 because sklearn and TF-IDF run on CPU.
