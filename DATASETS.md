# Dataset Preparation — DarkTrace Phase 1

Phase 1 runs in two modes:

- **Normal mode (default, reportable):** requires the **real** datasets. If a
  required file is missing, the experiment **fails hard** (exit code 2) and writes
  no Table 6 output. This guarantees reported numbers always come from real data.
- **Smoke-test mode (`--smoke-test`, NON-reportable):** generates synthetic data to
  verify the pipeline executes. Outputs are written to separate `*_SMOKETEST.*`
  files and are excluded from `build_table6`.

```bash
# normal (real data required):
python -m src.exp_traffic --config configs/traffic.json
python -m src.exp_text    --config configs/text.json

# smoke test (synthetic, non-reportable):
python -m src.exp_traffic --config configs/traffic.json --smoke-test
python -m src.exp_text    --config configs/text.json --smoke-test
```

---

## 1. CIC-Darknet2020 (traffic classification)

- **Source (official):** Canadian Institute for Cybersecurity, University of New
  Brunswick (UNB CIC). Download page:
  `https://www.unb.ca/cic/datasets/darknet-2020.html`
- **Reference:** [1] Lashkari, Kaur, Rahali, "DIDarknet," ICCNS 2020.
- **Access:** free for research; the page requests name/affiliation. No GPU needed.
- **Download:** the CSV of CICFlowMeter features (~158,659 flows).
- **Place at:** `data/raw/CIC-Darknet2020.csv`
- **Columns:** the loader **auto-detects** the label column (Label / Application /
  Traffic Type / Category) and strips leading/trailing spaces from headers.
- **Leakage removal (automatic):** Flow ID, Src/Dst IP, Src/Dst Port, Timestamp,
  and index columns are dropped (manuscript [3]) to prevent label leakage that can
  inflate accuracy. This is done for you; no manual editing required.

Steps:
```bash
# after downloading the CSV from the UNB page:
mv ~/Downloads/Darknet.csv data/raw/CIC-Darknet2020.csv   # name may vary
python -m src.exp_traffic --config configs/traffic.json
```

## 2. CoDA (dark web text classification) — Hugging Face

- **Source:** Hugging Face dataset `s2w-ai/CoDA`
  (`https://huggingface.co/datasets/s2w-ai/CoDA`).
- **Reference:** CoDA dark web text benchmark (10 illicit categories).
- **Access:** via the `datasets` library or direct download; check the dataset card
  for any access conditions.
- **Prepare a CSV** with a text column and a label column:

```python
# one-time export to the CSV layout Phase 1 expects
from datasets import load_dataset
ds = load_dataset("s2w-ai/CoDA")                  # may require: pip install datasets
split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
df = split.to_pandas()
# inspect df.columns, then rename the text/label columns if needed.
# the loader auto-detects common names (text/content/body, label/category/class).
df.to_csv("data/raw/coda.csv", index=False)
```

- **Place at:** `data/raw/coda.csv`
- **Configure (if auto-detect needs help):** set `text_col` / `label_col` in
  `configs/text.json`.

## 3. DUTA-10K (alternative dark web text corpus)

- **Source (official):** GVIS research group / dataset authors (Al-Nabki et al.).
  Project page: `https://gvis.unileon.es/dataset/duta-darknet-usage-text-addresses/`
  (request access from the authors; terms may apply).
- **References:** [23] Al-Nabki et al., EACL 2017; [24] Expert Systems w/ Apps 2019.
- **Access:** by request to the authors; not redistributable.
- **Prepare a CSV** with text and label columns, then **place at:** `data/raw/coda.csv`
  (or point `configs/text.json:data.csv_path` to your DUTA file and set
  `data.name` to `DUTA-10K`).

---

## Verifying real data is used

Real-data runs log:

    [INFO] Loading real CIC-Darknet2020 from data/raw/CIC-Darknet2020.csv
    [INFO] Real-data run complete — Table 6 traffic fragment written.

Missing data in normal mode fails hard:

    [ERROR] Real CIC-Darknet2020 CSV not found at 'data/raw/CIC-Darknet2020.csv'.
    (exit code 2)

Smoke-test runs are clearly marked:

    [WARNING] SMOKE-TEST MODE: generating SYNTHETIC data (NON-REPORTABLE).
    [WARNING] SMOKE TEST complete — outputs marked NON-REPORTABLE.

## Class-distribution reporting

Both loaders print per-class counts, percentages, and the max/min imbalance ratio,
and store them under `class_report` in the `*_results.json`. Use this to document
dataset composition (manuscript Section 8.2) and to anticipate minority-class
behaviour in per-class F1 (Section 8.18).

## Ethics / licensing

Respect each dataset's licence and terms; do not redistribute raw corpora. Phase 1
uses only public/by-request research datasets and requires no dark web crawling.

## Phase 2b cross-domain corpora (Hindi, Arabic)

Native dark-web Hindi/Arabic data does not exist at scale (see PHASE2B.md). For
the cross-domain experiment, download:

- Hindi Hostility (Bhardwaj et al. 2020, CONSTRAINT-2021):
  https://github.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021
  -> save train CSV as data/raw/hindi_hostility.csv (cols: 'Post', 'Labels Set').
- Arabic OSACT offensive/hate (Mubarak et al.):
  https://alt.qcri.org/resources/OSACT2022/  (TSV: text<TAB>OFF/NOT_OFF<TAB>HS/NOT_HS)
  -> save as data/raw/arabic_osact.csv  (TSV auto-detected).

Both are real, peer-reviewed, citable. Neither is dark-web data — report as
cross-domain (see PHASE2B.md).
