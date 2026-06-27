"""
fetch_datasets.py — gather all four DarkTrace datasets into data/raw/.

Run from the repo root (e.g. /kaggle/working/DTPaper). Network required for
CoDA / Hindi / Arabic; the CIC-Darknet2020 copy is a local file copy from the
Kaggle input mount and needs no network.

Produces (the exact filenames the configs expect):
  data/raw/coda.csv               <- CoDA (Hugging Face s2w-ai/CoDA)         [multilingual/text/scoring/ablation]
  data/raw/hindi_hostility.csv    <- Hindi Hostility (CONSTRAINT-2021)        [cross-domain]
  data/raw/arabic_osact.csv       <- OSACT2022 Arabic offensive/hate (TSV)    [cross-domain]
  data/raw/CIC-Darknet2020.csv    <- copied from the Kaggle CIC mount         [traffic]

All are real, public-for-research corpora. Hindi and Arabic are NOT dark-web
data; they are a cross-domain proxy (see PHASE2B.md). The script is defensive:
it tries multiple sources where relevant and tells you exactly what it got.
Nothing is fabricated; if a source fails, it says so and points you at the
manual download.
"""
import os
import shutil
import subprocess
import sys
import urllib.request

os.makedirs("data/raw", exist_ok=True)


def _have(p):
    return os.path.exists(p) and os.path.getsize(p) > 0


# ----------------------------------------------------------------------
# 1) CoDA — dark web text (Jin et al., 2022). Hugging Face: s2w-ai/CoDA
#    Target layout: a CSV whose key/text columns match exp_text.py's parser
#    (keys like '{id}-{Category}-{lang}-{hash}').
# ----------------------------------------------------------------------
def get_coda():
    dst = "data/raw/coda.csv"
    if _have(dst):
        print(f"[coda] already present: {dst}")
        return
    try:
        print("[coda] trying: Hugging Face s2w-ai/CoDA")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "pandas"],
                       check=True)
        from datasets import load_dataset
        import pandas as pd

        # The labelled CoDA config carries the category in the sample key.
        # Load the default split; if multiple splits exist, concatenate.
        ds = load_dataset("s2w-ai/CoDA")
        frames = []
        for split in ds.keys():
            frames.append(ds[split].to_pandas())
        df = pd.concat(frames, ignore_index=True) if frames else ds.to_pandas()

        # Keep the columns exp_text.py understands: a key column (carries
        # category+lang) and a text column. We do not rename/relabel beyond
        # passing through what CoDA ships; the loader does category parsing.
        cols_lower = {c.lower(): c for c in df.columns}
        key_col = next((cols_lower[k] for k in ("__key__", "key", "id", "__url__")
                        if k in cols_lower), None)
        text_col = next((cols_lower[k] for k in ("text", "txt", "content", "body", "json")
                         if k in cols_lower), None)

        if key_col is None or text_col is None:
            # Don't guess silently — report exactly what we got so the user
            # can pick the right config, per exp_text.py's guidance.
            print("[coda] WARNING: could not identify key/text columns automatically.")
            print("[coda] columns present:", list(df.columns))
            print("[coda] writing the raw frame to", dst,
                  "— verify it is the LABELLED CoDA split (keys like "
                  "'{id}-{Category}-{lang}-{hash}'), not the raw WebDataset text shard.")
            df.to_csv(dst, index=False)
            return

        out = df[[key_col, text_col]].copy()
        # exp_text.py looks for a '__key__'-style column; normalise the name.
        out = out.rename(columns={key_col: "__key__", text_col: "text"})
        out.to_csv(dst, index=False)
        print(f"[coda] wrote {dst}  ({len(out)} rows; cols=__key__,text)")
    except Exception as e:
        print(f"[coda] FAILED: {type(e).__name__}: {e}")
        print("[coda] Manual: download CoDA from Hugging Face 's2w-ai/CoDA' "
              "(accept access terms), export the LABELLED split to data/raw/coda.csv "
              "with a key column like '{id}-{Category}-{lang}-{hash}'. See DATASETS.md.")


# ----------------------------------------------------------------------
# 2) HINDI — Hostility Detection (Bhardwaj et al., 2020, CONSTRAINT-2021)
# ----------------------------------------------------------------------
def get_hindi():
    dst = "data/raw/hindi_hostility.csv"
    if _have(dst):
        print(f"[hi] already present: {dst}")
        return
    tmp = "/tmp/hindi_hostility_repo"
    if not os.path.isdir(tmp):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021",
                        tmp], check=True)
    found = None
    for root, _, files in os.walk(tmp):
        for f in files:
            if f.lower() == "train.csv":
                found = os.path.join(root, f)
    if not found:
        print("[hi] ERROR: train.csv not found in the repo; inspect", tmp)
        return
    shutil.copy(found, dst)
    print(f"[hi] wrote {dst}  (from {found})")


# ----------------------------------------------------------------------
# 3) ARABIC — OSACT2022 offensive/hate (Mubarak et al.)
# ----------------------------------------------------------------------
ARABIC_SOURCES = [
    ("OSACT2022 QCRI train.txt", "url",
     "https://alt.qcri.org/resources/OSACT2022/OSACT2022-sharedTask-train.txt"),
    ("motazsaad GitHub mirror (OSACT2020 train)", "url",
     "https://raw.githubusercontent.com/motazsaad/arabic-hatespeech-data/master/OSACT4/OSACT2020-sharedTask-train.txt"),
]


def get_arabic():
    dst = "data/raw/arabic_osact.csv"
    if _have(dst):
        print(f"[ar] already present: {dst}")
        return
    for desc, kind, loc in ARABIC_SOURCES:
        try:
            print(f"[ar] trying: {desc}")
            if kind == "url":
                req = urllib.request.Request(loc, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) < 1000:
                    print(f"[ar]   too small ({len(data)} bytes), skipping")
                    continue
                with open(dst, "wb") as f:
                    f.write(data)
                print(f"[ar] wrote {dst}  ({len(data)} bytes, from {desc})")
                return
        except Exception as e:
            print(f"[ar]   failed: {type(e).__name__}: {e}")
    # last resort: Hugging Face superset
    try:
        print("[ar] trying: Hugging Face manueltonneau/arabic-hate-speech-superset")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
        from datasets import load_dataset
        import pandas as pd
        ds = load_dataset("manueltonneau/arabic-hate-speech-superset", split="train")
        df = ds.to_pandas()
        textcol = next((c for c in ("text", "tweet", "content") if c in df.columns), df.columns[0])
        labcol = next((c for c in ("labels", "label", "hate") if c in df.columns), None)
        if labcol is None:
            print("[ar]   superset has no obvious label column:", list(df.columns)); return
        out = df[[textcol, labcol]].copy()
        out[labcol] = out[labcol].map(lambda v: "OFF" if int(v) == 1 else "NOT_OFF")
        out.to_csv(dst, sep="\t", header=False, index=False)
        print(f"[ar] wrote {dst}  ({len(out)} rows, from HF superset)")
        return
    except Exception as e:
        print(f"[ar]   HF fallback failed: {type(e).__name__}: {e}")
    print("[ar] ALL SOURCES FAILED. Download OSACT2022 manually from "
          "https://sites.google.com/view/arabichate2022/home and save as", dst)


# ----------------------------------------------------------------------
# 4) CIC-Darknet2020 — copy from the Kaggle input mount (no network).
#    Source path is the one you provided; we also probe common variants.
# ----------------------------------------------------------------------
CIC_SOURCES = [
    "/kaggle/input/datasets/peterfriedrich1/cicdarknet2020-internet-traffic/Darknet.CSV",
    "/kaggle/input/cicdarknet2020-internet-traffic/Darknet.CSV",
    "/kaggle/input/cicdarknet2020-internet-traffic/Darknet.csv",
]


def get_cic():
    dst = "data/raw/CIC-Darknet2020.csv"
    if _have(dst):
        print(f"[cic] already present: {dst}")
        return
    src = next((p for p in CIC_SOURCES if os.path.exists(p)), None)
    if src is None:
        # Last attempt: search the Kaggle input tree for a Darknet csv.
        for base in ("/kaggle/input",):
            if os.path.isdir(base):
                for root, _, files in os.walk(base):
                    for f in files:
                        if f.lower() in ("darknet.csv",):
                            src = os.path.join(root, f)
                            break
                    if src:
                        break
    if src is None:
        print("[cic] ERROR: CIC-Darknet2020 'Darknet.CSV' not found. Looked at:")
        for p in CIC_SOURCES:
            print("   ", p)
        print("[cic] If the Kaggle dataset is attached under a different path, "
              "copy it manually to", dst)
        return
    shutil.copy(src, dst)
    sz = os.path.getsize(dst)
    print(f"[cic] copied {src}\n[cic]   -> {dst}  ({sz:,} bytes)")


if __name__ == "__main__":
    get_coda()
    get_hindi()
    get_arabic()
    get_cic()

    print("\n=== summary: data/raw/ ===")
    for name in ("coda.csv", "hindi_hostility.csv", "arabic_osact.csv", "CIC-Darknet2020.csv"):
        p = os.path.join("data/raw", name)
        status = f"{os.path.getsize(p):,} bytes" if _have(p) else "MISSING"
        print(f"  {name:24s} {status}")

    print("\nVerify with:")
    print("  import pandas as pd")
    print("  print(pd.read_csv('data/raw/coda.csv').head())")
    print("  print(pd.read_csv('data/raw/CIC-Darknet2020.csv', nrows=3).columns.tolist())")
    print("  print(pd.read_csv('data/raw/hindi_hostility.csv').head())")
    print("  print(open('data/raw/arabic_osact.csv').readline())")
