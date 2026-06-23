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


def load_dataset(cfg, logger):
    path = cfg["data"]["csv_path"]
    if os.path.exists(path):
        logger.info(f"Loading real text corpus from {path}")
        df = pd.read_csv(path)
        synthetic = False
        tcol = cfg["data"]["text_col"]; lcol = cfg["data"]["label_col"]
        df = df[[tcol, lcol]].rename(columns={tcol: "text", lcol: "label"})
    else:
        logger.warning(f"{path} not found -> generating SYNTHETIC text for pipeline test")
        df = _make_synthetic(); synthetic = True
    df = df.dropna(subset=["text", "label"])
    # length filtering for short/empty docs (manuscript Section 8.2 mitigation)
    minlen = cfg["data"].get("min_chars", 10)
    df = df[df["text"].str.len() >= minlen].reset_index(drop=True)
    return df, synthetic


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


def run(cfg, logger):
    seed = cfg["seed"]; set_seed(seed)
    df, synthetic = load_dataset(cfg, logger)
    classes = sorted(df["label"].unique())
    cls_to_id = {c: i for i, c in enumerate(classes)}
    y = df["label"].map(cls_to_id).values
    texts = df["text"].values

    Xtr, Xte, ytr, yte = train_test_split(
        texts, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed)
    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)

    results = {"dataset": cfg["data"]["name"], "synthetic": synthetic,
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
    flag = " (SYNTHETIC)" if results.get("synthetic") else ""
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
    ap.add_argument("--config", default="configs/text.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("text", cfg["paths"]["logs"])
    t0 = time.time()
    results = run(cfg, logger)
    save_json(results, os.path.join(cfg["paths"]["tables"], "text_results.json"))
    write_table6_fragment(
        results, os.path.join(cfg["paths"]["tables"], "table6_text.csv"), logger)
    logger.info(f"Done in {time.time()-t0:.1f}s. "
                f"{'SYNTHETIC — not for reporting.' if results['synthetic'] else 'Real data.'}")


if __name__ == "__main__":
    main()
