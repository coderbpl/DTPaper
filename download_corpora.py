"""
Download + prepare the Hindi and Arabic cross-domain corpora for DarkTrace Phase 2b.
Run from the repo root (e.g. /kaggle/working/DTPaper). Network required.

Produces:
  data/raw/hindi_hostility.csv   (cols: Post, Labels Set)   <- Hindi Hostility (CONSTRAINT-2021)
  data/raw/arabic_osact.csv      (TSV: text<TAB>label...)    <- OSACT2022 Arabic offensive/hate

Both are real, peer-reviewed, public-for-research corpora. NEITHER is dark-web data;
they are used as a cross-domain proxy (see PHASE2B.md). The script is defensive:
it tries multiple sources for the Arabic file and tells you exactly what it got.
"""
import os, subprocess, sys, urllib.request, shutil

os.makedirs("data/raw", exist_ok=True)


def _have(p):
    return os.path.exists(p) and os.path.getsize(p) > 0


# ----------------------------------------------------------------------
# 1) HINDI — Hostility Detection (Bhardwaj et al., 2020, CONSTRAINT-2021)
# ----------------------------------------------------------------------
def get_hindi():
    dst = "data/raw/hindi_hostility.csv"
    if _have(dst):
        print(f"[hi] already present: {dst}")
        return
    # the repo ships Dataset/train.csv with columns Unique ID, Post, Labels Set
    tmp = "/tmp/hindi_hostility_repo"
    if not os.path.isdir(tmp):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021",
                        tmp], check=True)
    # find train.csv anywhere in the repo
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
# 2) ARABIC — OSACT2022 offensive/hate (Mubarak et al.)
#    Try sources in order of reliability; stop at first success.
# ----------------------------------------------------------------------
ARABIC_SOURCES = [
    # (description, kind, location)
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
        ds = load_dataset("manueltonneau/arabic-hate-speech-superset", split="train")
        import pandas as pd
        df = ds.to_pandas()
        # the superset has a text column + a binary 'labels' column; normalise to TSV
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


if __name__ == "__main__":
    get_hindi()
    get_arabic()
    print("\nDone. Verify with:")
    print("  import pandas as pd")
    print("  print(pd.read_csv('data/raw/hindi_hostility.csv').head())")
    print("  print(open('data/raw/arabic_osact.csv').readline())")
