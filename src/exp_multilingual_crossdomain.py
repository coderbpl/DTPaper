"""
darktrace_phase1/src/exp_multilingual_crossdomain.py

Phase 2b / Cross-domain multilingual threat classification for Hindi and Arabic
(manuscript Section 8.7b, RQ2-extension).

WHY THIS EXISTS (state this verbatim in the manuscript):
The brief requires multilingual coverage including Hindi and Arabic. However,
native dark-web corpora for Hindi and Arabic do NOT exist at evaluation scale:
~90% of dark-web text is English (Jin et al. 2022; the DarkBERT authors report
the same and decline to build a multilingual dark-web model for this reason). In
our CoDA corpus Hindi has n=1 and Arabic n=7 — far too few to evaluate. We
therefore run a CROSS-DOMAIN experiment: we evaluate whether threat-relevant text
classification works in Hindi and Arabic using the closest available REAL,
peer-reviewed annotated corpora:

  - Hindi : Hostility Detection Dataset (Bhardwaj et al., 2020; CONSTRAINT-2021).
            ~8200 posts, classes: fake / hate / offensive / defamation / non-hostile.
            https://github.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021
  - Arabic: OSACT Arabic Offensive Language / Hate Speech shared task
            (Mubarak et al.). Tweets labelled offensive/not (+ fine-grained
            hate types), Cohen's kappa ~0.82.
            https://alt.qcri.org/resources/OSACT2022/  (or the HF superset
            manueltonneau/arabic-hate-speech-superset)

This is HONESTLY a different domain (social media, not dark web). It answers a
real question: does multilingual threat classification transfer to Hindi/Arabic
at all, given no native dark-web data? Report it as a cross-domain robustness
result and a documented data-availability limitation — NOT as dark-web Hindi/Arabic.

To keep the two corpora comparable and comparable to the dark-web task, each is
mapped to a BINARY threat label: threat/hostile/offensive = 1, benign = 0. The
exact source-label -> binary mapping is stated in the results JSON.

This module is REAL-DATA-ONLY: it fails hard if a corpus is not present. It uses
the same classifier as Phase 2 (distil multilingual transformer on GPU, TF-IDF
fallback on CPU) so the methodology is identical across languages.

Run (after placing the two CSVs; see DATASETS.md):
    python -m src.exp_multilingual_crossdomain --config configs/multilingual_crossdomain.json
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from .metrics_stats import bootstrap_ci
from .utils import load_config, ensure_dirs, get_logger, save_json, set_seed
from .exp_multilingual import _fit_predict


class MissingRealDataError(FileNotFoundError):
    """Raised when a required language corpus is absent (real-data-only)."""


# ---- source-label -> binary threat mapping (STATED, overridable in config) ----
# 1 = threat/hostile/offensive ; 0 = benign/non-hostile.
HINDI_THREAT_LABELS = {  # Bhardwaj et al. 2020 hostility dimensions
    "fake": 1, "hate": 1, "offensive": 1, "defamation": 1,
    "non-hostile": 0, "non_hostile": 0, "nonhostile": 0, "none": 0,
}
ARABIC_THREAT_LABELS = {  # OSACT offensive/hate labels
    "off": 1, "offensive": 1, "hs": 1, "hate": 1, "vulgar": 1, "violent": 1,
    "not_off": 0, "not_offensive": 0, "not_hs": 0, "clean": 0, "normal": 0,
}


def _map_binary(series, mapping, logger, lang):
    """Map free-form source labels to {0,1} using a case-insensitive mapping.
    Rows whose label is not in the mapping are dropped and counted (never guessed)."""
    raw = series.astype(str).str.strip().str.lower()
    mapped = raw.map(mapping)
    n_unmapped = int(mapped.isna().sum())
    if n_unmapped:
        unknown = sorted(raw[mapped.isna()].unique())[:10]
        logger.warning(f"[{lang}] {n_unmapped} row(s) had labels outside the "
                       f"stated mapping and were dropped: {unknown}")
    return mapped


def _load_one(cfg_lang, logger, lang):
    """Load one language corpus -> DataFrame(text, y in {0,1}), real data only."""
    path = cfg_lang["csv_path"]
    if not os.path.exists(path):
        raise MissingRealDataError(
            f"[{lang}] corpus not found at '{path}'. This experiment is "
            f"real-data-only. Download it (see DATASETS.md) and place it there.")
    logger.info(f"[{lang}] loading {path}")
    # robust read: try comma, then tab (OSACT ships tab-separated)
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            raise ValueError("single column; retry as TSV")
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None,
                         names=cfg_lang.get("tsv_columns", ["text", "label", "label2"]),
                         engine="python", on_bad_lines="skip")

    tcol = cfg_lang["text_col"]; lcol = cfg_lang["label_col"]
    if tcol not in df.columns or lcol not in df.columns:
        raise ValueError(
            f"[{lang}] expected columns text='{tcol}' label='{lcol}' but found "
            f"{list(df.columns)[:8]}. Set text_col/label_col in the config.")

    mapping = {**(HINDI_THREAT_LABELS if lang == "hi" else ARABIC_THREAT_LABELS),
               **{k.lower(): v for k, v in cfg_lang.get("label_map", {}).items()}}
    y = _map_binary(df[lcol], mapping, logger, lang)
    out = pd.DataFrame({"text": df[tcol].astype(str), "y": y, "lang": lang})
    out = out.dropna(subset=["y"])
    out["y"] = out["y"].astype(int)
    out = out[out["text"].str.strip().str.len() >= cfg_lang.get("min_chars", 3)]
    out = out.reset_index(drop=True)
    pos = int(out["y"].sum())
    logger.info(f"[{lang}] {len(out)} rows after mapping/cleaning "
                f"({pos} threat / {len(out)-pos} benign).")
    if out["y"].nunique() < 2:
        raise ValueError(f"[{lang}] needs both classes after mapping; check label_map.")
    return out, {"mapping": mapping, "n": len(out), "n_threat": pos}


def _evaluate_language(df_lang, cfg, logger, lang):
    """Train+test the same classifier within one language; return metrics."""
    seed = cfg["seed"]
    # binary classes for the shared classifier interface
    classes = [0, 1]
    df = df_lang.rename(columns={"y": "y"}).copy()
    df["label"] = df["y"]
    tr, te = train_test_split(df, test_size=cfg["data"]["test_size"],
                              stratify=df["y"], random_state=seed)
    y_pred, path = _fit_predict(tr, te, classes, cfg, logger)
    y_true = te["y"].values
    macro = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    lo, hi = bootstrap_ci(y_true, np.asarray(y_pred), "macro_f1",
                          n_boot=cfg.get("bootstrap", 500), seed=seed)
    logger.info(f"[{lang}] macro-F1={macro:.4f} CI95=[{lo:.4f},{hi:.4f}] "
                f"acc={acc:.4f} (model={path}, n_test={len(y_true)})")
    return {"language": lang, "n_test": int(len(y_true)),
            "macro_f1": macro, "macro_f1_ci95": [lo, hi],
            "accuracy": acc, "model_path": path}


def run(cfg, logger):
    set_seed(cfg["seed"])
    results = {"experiment": "phase2b_crossdomain_multilingual",
               "reportable": True,
               "domain_note": ("CROSS-DOMAIN: Hindi hostility + Arabic offensive "
                               "corpora (social media), used because native dark-web "
                               "Hindi/Arabic data does not exist at scale. Binary "
                               "threat/benign label."),
               "seed": cfg["seed"], "languages": {}, "per_language": []}

    for lang in ("hi", "ar"):
        lang_cfg = cfg["data"].get(lang)
        if not lang_cfg:
            logger.warning(f"No config block for '{lang}'; skipping.")
            continue
        df_lang, meta = _load_one(lang_cfg, logger, lang)
        results["languages"][lang] = {k: v for k, v in meta.items() if k != "mapping"}
        results["languages"][lang]["label_mapping"] = meta["mapping"]
        results["per_language"].append(_evaluate_language(df_lang, cfg, logger, lang))

    if not results["per_language"]:
        raise MissingRealDataError(
            "Neither Hindi nor Arabic corpus was available. Provide at least one "
            "(see DATASETS.md).")
    return results


def write_table7b_fragment(results, out_path, logger):
    rows = []
    for m in results["per_language"]:
        rows.append({"Language": m["language"], "N_test": m["n_test"],
                     "Macro_F1": round(m["macro_f1"], 4),
                     "Accuracy": round(m["accuracy"], 4),
                     "Model": m["model_path"], "Domain": "cross-domain (social media)"})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Wrote Table 7b fragment -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multilingual_crossdomain.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("multilingual_crossdomain", cfg["paths"]["logs"])
    t0 = time.time()
    try:
        results = run(cfg, logger)
    except MissingRealDataError as e:
        logger.error(str(e)); raise SystemExit(2)
    tables = cfg["paths"]["tables"]
    save_json(results, os.path.join(tables, "multilingual_crossdomain_results.json"))
    write_table7b_fragment(results, os.path.join(tables, "table7b_crossdomain.csv"), logger)
    logger.info("Real-data run complete — Table 7b cross-domain fragment written.")
    logger.info(f"Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
