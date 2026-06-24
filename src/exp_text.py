"""
darktrace_phase1/src/exp_text.py

Phase 1 / Experiment B: dark web text classification on CoDA / DUTA-10K.
Populates manuscript Table 6 (text rows) and feeds Section 8.7.

Phase 1 deliberately uses CPU-only baselines (manuscript Section 8.4):
TF-IDF + Logistic Regression and TF-IDF + Linear SVM. Transformer fine-tuning
(BERT/RoBERTa/DarkBERT) is Phase 2 and requires a GPU; it is intentionally
NOT run here.

Run:
    python -m src.exp_text --config configs/text.yaml
Synthetic fallback applies if the corpus is absent (flagged `synthetic: true`).
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .metrics_stats import (classification_metrics, per_class_f1, aggregate_folds,
                            mcnemar_test, bootstrap_ci)
from .utils import load_config, ensure_dirs, get_logger, save_json, set_seed


# illicit-activity categories aligned to CoDA-style labels (manuscript [23],[24])
SYNTH_CATEGORIES = ["Drugs", "Weapons", "Fraud", "Hacking", "Counterfeit", "Other"]
SYNTH_TERMS = {
    "Drugs": ["gram", "cocaine", "mdma", "shipping", "stealth", "purity"],
    "Weapons": ["pistol", "ammo", "rifle", "untraceable", "caliber", "ship"],
    "Fraud": ["cvv", "dumps", "fullz", "paypal", "cashout", "bank"],
    "Hacking": ["exploit", "rdp", "botnet", "ransomware", "zeroday", "shell"],
    "Counterfeit": ["replica", "passport", "id", "banknote", "hologram", "scan"],
    "Other": ["forum", "vendor", "review", "escrow", "mirror", "pgp"],
}


def _make_synthetic(n_per=400, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    # imbalance: some categories rarer (manuscript notes DUTA imbalance)
    weights = {"Drugs": 1.0, "Fraud": 0.8, "Hacking": 0.6, "Other": 0.9,
               "Weapons": 0.4, "Counterfeit": 0.3}
    all_terms = sum(SYNTH_TERMS.values(), [])
    for cat, w in weights.items():
        for _ in range(int(n_per * w)):
            k = rng.randint(8, 30)
            # weaker signal + heavy shared filler so classes overlap (realistic, not trivial)
            base = rng.choice(SYNTH_TERMS[cat], size=min(k, 4)).tolist()
            filler = rng.choice(all_terms, size=k * 2).tolist()
            words = base + filler
            rng.shuffle(words)
            label = cat
            # 12% label noise to prevent a trivially separable problem
            if rng.rand() < 0.12:
                label = rng.choice(SYNTH_CATEGORIES)
            rows.append({"text": " ".join(words), "label": label})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


class MissingRealDataError(FileNotFoundError):
    """Raised in normal (reportable) mode when the real corpus is absent."""


# common column-name variants for CoDA / DUTA-style corpora
TEXT_COL_CANDIDATES = ["text", "content", "body", "document", "raw_text",
                       "page_text", "txt", "html", "page", "data"]
LABEL_COL_CANDIDATES = ["label", "category", "class", "type", "target",
                        "main_category", "cls", "labels", "category_name",
                        "__key__"]


def _derive_label_from_key(df, key_col, logger):
    """Derive a categorical label (and language, when present) from a WebDataset
    __key__ column.

    Handles the real CoDA key format, where each key looks like:
        coda_dataset/{id}-{Category}-{lang}-{sha256}
    e.g. 'coda_dataset/5756-Arms-en-06dd9c0e...'  -> category 'Arms', lang 'en'.

    Falls back to a generic 'first path segment' heuristic for other layouts.
    Returns (labels_series, lang_series_or_None) or (None, None) if unusable.
    """
    import re
    s = df[key_col].astype(str)
    tail = s.str.split("/").str[-1]            # drop any leading 'coda_dataset/'

    # Pattern 1: CoDA  {id}-{Category}-{lang}-{hash}
    coda_re = re.compile(r"^\d+-([A-Za-z][A-Za-z _]*?)-([a-z]{2})-[0-9a-f]{8,}$")
    m = tail.str.match(coda_re)
    if m.mean() >= 0.8:                         # most rows match the CoDA layout
        cat = tail.str.replace(coda_re, r"\1", regex=True)
        lang = tail.str.replace(coda_re, r"\2", regex=True)
        n_cat = cat.nunique()
        logger.warning(
            f"Parsed CoDA __key__ format: {n_cat} categories "
            f"(e.g. {sorted(cat.unique())[:8]}); languages "
            f"(e.g. {sorted(lang.unique())[:8]}). Labels = category.")
        return cat, lang

    # Pattern 2: generic 'category/...' path prefix
    prefix = tail.str.split("-").str[1] if tail.str.contains("-").mean() > 0.8 else None
    if prefix is None:
        prefix = s.str.split("/").str[0].str.replace(r"^[a-zA-Z_]+=", "", regex=True)
    n_unique = prefix.nunique()
    if 2 <= n_unique <= max(50, int(0.2 * len(df))):
        logger.warning(
            f"Derived label from '{key_col}' ({n_unique} classes, e.g. "
            f"{list(prefix.unique()[:6])}). VERIFY these are real categories.")
        return prefix, None

    return None, None


def _auto_detect_columns(df, cfg, logger):
    """Resolve text/label columns: explicit config first, then auto-detect.

    Handles the CoDA WebDataset export whose columns are like
    ['__key__', '__url__', 'txt'] by (a) detecting 'txt' as text and
    (b) deriving a label from the '__key__' path prefix when no label column
    exists. Raises a clear, actionable error otherwise.
    """
    tcol = cfg["data"].get("text_col")
    lcol = cfg["data"].get("label_col")
    cols_lower = {c.lower(): c for c in df.columns}

    # --- text column ---
    if not tcol or tcol not in df.columns:
        for cand in TEXT_COL_CANDIDATES:
            if cand in cols_lower and cols_lower[cand] not in ("__key__", "__url__"):
                tcol = cols_lower[cand]; break
    # last resort: the longest-average-string column that isn't key/url
    if tcol is None or tcol not in df.columns:
        cand_cols = [c for c in df.columns if c not in ("__key__", "__url__")]
        if cand_cols:
            avg_len = {c: df[c].astype(str).str.len().mean() for c in cand_cols}
            tcol = max(avg_len, key=avg_len.get)
            logger.warning(f"No standard text column; using longest-text column '{tcol}'. "
                           f"Override with data.text_col if wrong.")
    if tcol is None or tcol not in df.columns:
        raise ValueError(
            "Could not find a text column. Set data.text_col in configs/text.json.\n"
            f"Available columns: {list(df.columns)}")

    # --- label column ---
    if not lcol or lcol not in df.columns:
        for cand in LABEL_COL_CANDIDATES:
            if cand == "__key__":
                continue  # handled specially below
            if cand in cols_lower:
                lcol = cols_lower[cand]; break

    if (lcol is None or lcol not in df.columns):
        # try to derive a label from a __key__ WebDataset column
        key_col = next((c for c in df.columns if c.lower() == "__key__"), None)
        if key_col is not None:
            derived, lang = _derive_label_from_key(df, key_col, logger)
            if derived is not None:
                df["__derived_label__"] = derived.values
                lcol = "__derived_label__"
                # preserve language if parsed (useful for the Phase 2 multilingual split)
                if lang is not None:
                    df["__lang__"] = lang.values

    if lcol is None or lcol not in df.columns:
        raise ValueError(
            "Could not find a label column, and could not derive one.\n"
            f"Available columns: {list(df.columns)}\n"
            "This usually means the CoDA export is the raw WebDataset text shard "
            "WITHOUT category labels. Fixes:\n"
            "  1) Load the LABELLED CoDA split/config (the one with a category "
            "field) and re-export; or\n"
            "  2) If your file has a label under another name, set data.label_col "
            "in configs/text.json; or\n"
            "  3) Use a labelled corpus (e.g. DUTA-10K) at data.csv_path.\n"
            "Phase 1 text classification requires per-document category labels.")

    logger.info(f"Using text column '{tcol}' and label column '{lcol}'.")
    return tcol, lcol


def _report_class_distribution(y, logger, title="Class distribution"):
    vals, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    logger.info(f"{title} (n={total}, classes={len(vals)}):")
    order = np.argsort(-counts)
    dist = {}
    for i in order:
        pct = 100.0 * counts[i] / total
        logger.info(f"    {str(vals[i]):<24} {counts[i]:>8}  ({pct:5.2f}%)")
        dist[str(vals[i])] = {"count": int(counts[i]), "pct": round(pct, 3)}
    imbalance = float(counts.max() / counts.min()) if counts.min() > 0 else None
    if imbalance:
        logger.info(f"    imbalance ratio (max/min) = {imbalance:.1f}")
    return {"n": int(total), "n_classes": int(len(vals)),
            "distribution": dist, "imbalance_ratio": imbalance}


def load_dataset(cfg, logger, smoke_test=False):
    """Load CoDA/DUTA text. In normal mode the real corpus is mandatory."""
    path = cfg["data"]["csv_path"]

    if smoke_test:
        logger.warning("SMOKE-TEST MODE: generating SYNTHETIC text (NON-REPORTABLE).")
        df = _make_synthetic(); synthetic = True
    else:
        if not os.path.exists(path):
            raise MissingRealDataError(
                f"Real text corpus not found at '{path}'.\n"
                f"Normal mode requires real data for reportable results.\n"
                f"  - Download CoDA (Hugging Face s2w-ai/CoDA) or DUTA-10K "
                f"(see DATASETS.md) and place it at that path, or\n"
                f"  - run with --smoke-test to exercise the pipeline on synthetic data.")
        logger.info(f"Loading real text corpus from {path}")
        df = pd.read_csv(path)
        synthetic = False
        tcol, lcol = _auto_detect_columns(df, cfg, logger)
        df = df[[tcol, lcol]].rename(columns={tcol: "text", lcol: "label"})

    # --- text/label validation (CoDA mitigation, manuscript Section 8.2) ---
    n0 = len(df)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    # drop empty / whitespace-only text and empty labels
    df = df[df["text"].str.strip().str.len() > 0]
    df = df[df["label"].str.len() > 0]
    minlen = cfg["data"].get("min_chars", 10)
    df = df[df["text"].str.len() >= minlen]
    # drop exact-duplicate documents (common in dark web scrapes)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if not synthetic:
        logger.info(f"Text validation: {n0} -> {len(df)} rows after cleaning "
                    f"(dropped empties, short docs < {minlen} chars, duplicates).")
        if len(df) == 0:
            raise ValueError("No valid documents remain after cleaning; check the corpus.")
        # require at least 2 classes with enough samples to stratify
        vc = df["label"].value_counts()
        if len(vc) < 2:
            raise ValueError(f"Need >= 2 classes; found {len(vc)}.")
        rare = vc[vc < cfg["cv_folds"]]
        if len(rare) > 0:
            logger.warning(f"{len(rare)} class(es) have < {cfg['cv_folds']} samples and "
                           f"may break stratified CV: {rare.to_dict()}")

    class_report = _report_class_distribution(
        np.asarray(df["label"].tolist(), dtype=object), logger,
        f"{cfg['data']['name']} classes")
    return df, synthetic, class_report


def build_models(cfg, seed):
    return {
        "TFIDF_LogReg": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=cfg["model"]["max_features"],
                                      ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "TFIDF_LinearSVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=cfg["model"]["max_features"],
                                      ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", random_state=seed)),
        ]),
    }


def run(cfg, logger, smoke_test=False):
    seed = cfg["seed"]; set_seed(seed)
    df, synthetic, class_report = load_dataset(cfg, logger, smoke_test=smoke_test)
    classes = sorted(df["label"].unique())
    cls_to_id = {c: i for i, c in enumerate(classes)}
    # Convert to plain NumPy arrays. On environments where pandas uses the
    # PyArrow backend (e.g. Kaggle), .values returns an arrow-backed extension
    # array that train_test_split cannot index with an array of positions
    # ("only integer scalar arrays can be converted to a scalar index").
    # np.asarray(..., dtype=object/int) forces a standard NumPy array.
    y = np.asarray(df["label"].map(cls_to_id).tolist(), dtype=int)
    texts = np.asarray(df["text"].astype(str).tolist(), dtype=object)

    Xtr, Xte, ytr, yte = train_test_split(
        texts, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed)
    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)

    results = {"dataset": cfg["data"]["name"], "synthetic": synthetic,
               "reportable": (not synthetic), "class_report": class_report,
               "classes": classes, "seed": seed,
               "n_train": int(len(ytr)), "n_test": int(len(yte)), "models": {}}
    test_preds = {}

    for name in build_models(cfg, seed):
        logger.info(f"=== {name} ===")
        fold_metrics = []
        for k, (tri, vai) in enumerate(skf.split(Xtr, ytr)):
            model = build_models(cfg, seed)[name].fit(Xtr[tri], ytr[tri])
            pv = model.predict(Xtr[vai])
            m = classification_metrics(ytr[vai], pv, labels=np.arange(len(classes)))
            fold_metrics.append(m)
            logger.info(f"  fold {k}: macro-F1={m['macro_f1']:.4f}")
        model = build_models(cfg, seed)[name].fit(Xtr, ytr)
        pte = model.predict(Xte); test_preds[name] = pte
        m = classification_metrics(yte, pte, labels=np.arange(len(classes)))
        lo, hi = bootstrap_ci(yte, pte, "macro_f1", n_boot=cfg["bootstrap"], seed=seed)
        results["models"][name] = {
            "cv": aggregate_folds(fold_metrics), "test": m,
            "test_macro_f1_ci95": [lo, hi],
            "test_per_class_f1": per_class_f1(yte, pte, labels=np.arange(len(classes))),
        }
        logger.info(f"  TEST macro-F1={m['macro_f1']:.4f} CI95=[{lo:.4f},{hi:.4f}]")

    names = list(results["models"].keys())
    if len(names) == 2:
        p, stat = mcnemar_test(yte, test_preds[names[0]], test_preds[names[1]])
        results["mcnemar"] = {"models": names, "p_value": p, "statistic": stat}
        logger.info(f"McNemar {names[0]} vs {names[1]}: p={p:.3e}")
    return results


def write_table6_fragment(results, out_path, logger):
    rows = []
    flag = " [SMOKE-TEST/NON-REPORTABLE]" if results.get("synthetic") else ""
    for name, r in results["models"].items():
        t = r["test"]
        rows.append({
            "Model_dataset": f"DarkTrace {name} ({results['dataset']}){flag}",
            "Accuracy": round(t["accuracy"], 4),
            "Macro_F1": round(t["macro_f1"], 4),
            "AUC": "NA",
            "FPR": round(t["fpr_macro"], 4),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Wrote Table 6 fragment -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/text.json")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run on SYNTHETIC data to verify the pipeline (NON-REPORTABLE).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("text", cfg["paths"]["logs"])
    t0 = time.time()

    try:
        results = run(cfg, logger, smoke_test=args.smoke_test)
    except MissingRealDataError as e:
        logger.error(str(e))
        raise SystemExit(2)

    tables = cfg["paths"]["tables"]
    if results["synthetic"]:
        save_json(results, os.path.join(tables, "text_results_SMOKETEST.json"))
        write_table6_fragment(
            results, os.path.join(tables, "table6_text_SMOKETEST.csv"), logger)
        logger.warning("SMOKE TEST complete — outputs marked NON-REPORTABLE. "
                       "Table 6 fragment NOT written (real data required).")
    else:
        save_json(results, os.path.join(tables, "text_results.json"))
        write_table6_fragment(
            results, os.path.join(tables, "table6_text.csv"), logger)
        logger.info("Real-data run complete — Table 6 text fragment written.")

    logger.info(f"Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
