# Dataset Preparation — DarkTrace Phase 1

Phase 1 uses two **public** datasets. Until you place the real files here, every
script runs on a clearly-labelled **synthetic fallback** so you can test the
pipeline immediately. Synthetic outputs are tagged `(SYNTHETIC)` and must never
be reported as results.

---

## 1. CIC-Darknet2020 (traffic classification)

- **Source:** Canadian Institute for Cybersecurity (University of New Brunswick).
  Search "CIC-Darknet2020 UNB" and download from the official CIC datasets page.
- **Manuscript reference:** [1] Lashkari, Kaur, Rahali, DIDarknet, ICCNS 2020.
- **Access:** free for research; the page asks for name/affiliation. No GPU needed.
- **What to download:** the CSV of CICFlowMeter features (≈158,659 flows).
- **Place it at:** `data/raw/CIC-Darknet2020.csv`
- **Label column:** the script auto-detects `Label`; if your export differs, edit
  `configs/traffic.json` is not needed — the loader scans common label names.

Notes:
- The loader automatically drops identifier/leakage columns (Flow ID, IPs,
  Timestamp) following the caution in manuscript [3]; this is important because
  some published >99% figures may leak from such fields.

## 2. CoDA and/or DUTA-10K (dark web text classification)

- **CoDA:** a 10,000-document dark web text benchmark (10 illicit categories).
  Obtain from the authors' release; search "CoDA dark web dataset Jin".
- **DUTA-10K:** Darknet Usage Text Addresses; request from the authors
  (manuscript [23], [24] — Al-Nabki et al.). Access may require an email request.
- **Manuscript references:** [23] EACL 2017, [24] Expert Systems w/ Apps 2019.
- **Format expected:** a CSV with a text column and a label column.
- **Place it at:** `data/raw/coda.csv`
- **Configure columns** in `configs/text.json`:
  ```json
  "data": {"name": "CoDA", "csv_path": "data/raw/coda.csv",
           "text_col": "text", "label_col": "label", ...}
  ```

If your corpus uses different column names, set `text_col` / `label_col`
accordingly. Short/empty documents (< `min_chars`) are filtered, mirroring the
DUTA-imbalance mitigation in manuscript Section 8.2.

---

## Verifying real data is picked up

When a real file is present, the log line changes from:

    [WARNING] ... not found -> generating SYNTHETIC data for pipeline test

to:

    [INFO] Loading real CIC-Darknet2020 from data/raw/CIC-Darknet2020.csv

and the output CSV rows lose the `(SYNTHETIC)` marker.

## Ethics / licensing

Respect each dataset's licence and terms. Do not redistribute raw corpora.
Phase 1 uses only public, research-released data and requires no dark web
crawling (that begins in later phases under the ethical controls described in
manuscript Section 8.18).
