"""
darktrace_phase1/src/exp_multilingual.py

Phase 2 / Multilingual dark-web text classification (manuscript Section 8.7, RQ2).

Scope (decided with the data in hand): CLASSIFICATION ONLY. CoDA ships category
labels but NO entity-span annotations, so NER cannot be trained or evaluated on it
honestly; NER is therefore out of scope here and should use a span-annotated corpus
(DNRTI [8] / APTNER [30]) in separate work. This module does what CoDA genuinely
supports: multilingual category classification with PER-LANGUAGE reporting and a
cross-lingual transfer test.

Two evaluations, both real and reviewer-defensible:
  (A) Pooled multilingual model: train one model on all in-scope languages, report
      overall AND per-language macro-F1 (so weak languages are visible, not hidden
      inside an English-dominated average).
  (B) Cross-lingual transfer: train on English only, test on each other language.
      This is the actual multilingual *claim* — does the model generalise across
      languages, or is it just doing well because 89% of CoDA is English?

Language scope is data-driven: only languages with >= data.min_lang_count samples
are included (default 50), so we never report a metric on a language with too few
examples to be meaningful. Hindi (n=1 in CoDA) is therefore excluded automatically
and that exclusion is logged and recorded.

Model:
  - Primary: a lightweight multilingual transformer (distil) fine-tuned with the
    HuggingFace Trainer when torch+transformers and a GPU are available.
  - Fallback: TF-IDF + linear SVM (CPU) when transformers are unavailable, so the
    pipeline runs anywhere and the experimental logic is testable. The output
    records which path was used; only the transformer path should be reported as
    the Phase-2 result.

Run:
    python -m src.exp_multilingual --config configs/multilingual.json
    python -m src.exp_multilingual --config configs/multilingual.json --smoke-test
"""
from __future__ import annotations
import argparse, json, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from .metrics_stats import bootstrap_ci
from .utils import load_config, ensure_dirs, get_logger, save_json, set_seed

# optional heavy deps (present on Kaggle GPU, absent in a basic sandbox)
try:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    HAVE_TRANSFORMERS = True
    HAVE_GPU = torch.cuda.is_available()
except Exception:
    HAVE_TRANSFORMERS = False
    HAVE_GPU = False

# A lightweight multilingual model. distilbert multilingual is small and fast;
# override via config model.name (e.g. 'xlm-roberta-base' for the heavier option).
DEFAULT_MODEL = "distilbert-base-multilingual-cased"


def _load_multilingual(cfg, logger, smoke_test):
    """Load CoDA with language, restrict to in-scope languages (n >= threshold)."""
    if smoke_test:
        rng = np.random.RandomState(cfg["seed"])
        langs = ["en", "ru", "de", "fr", "es"]
        cats = ["Drugs", "Arms", "Hacking", "Financial", "Others"]
        rows = []
        for _ in range(1200):
            l = rng.choice(langs, p=[0.6, 0.15, 0.1, 0.08, 0.07])
            c = rng.choice(cats)
            rows.append({"text": f"{c} {l} " + " ".join(
                rng.choice(["vendor", "market", "price", "ship", "escrow"], 8)),
                "label": c, "lang": l})
        df = pd.DataFrame(rows)
        return df, True

    from .exp_text import load_dataset as load_text
    df, synthetic, _ = load_text(cfg, logger, smoke_test=False)
    if "__lang__" not in df.columns:
        raise ValueError("No __lang__ column; Phase 2 needs language info from the "
                         "CoDA key parser. Re-run with the current exp_text.py.")
    df = df.rename(columns={"__lang__": "lang"})
    # restrict to languages with enough samples
    min_count = cfg["data"].get("min_lang_count", 50)
    vc = df["lang"].value_counts()
    keep_langs = vc[vc >= min_count].index.tolist()
    dropped = vc[vc < min_count]
    logger.info(f"In-scope languages (n>={min_count}): "
                f"{ {l: int(vc[l]) for l in keep_langs} }")
    if len(dropped):
        logger.warning(
            f"Excluded {len(dropped)} language(s) with too few samples "
            f"(e.g. Hindi if present): {dict(list(dropped.items())[:12])}"
            + (" ..." if len(dropped) > 12 else ""))
    df = df[df["lang"].isin(keep_langs)].reset_index(drop=True)
    return df, False


def _encode_labels(df):
    classes = sorted(df["label"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    df = df.copy()
    df["y"] = df["label"].map(c2i)
    return df, classes, c2i


# ----------------------------------------------------------------------
# Model paths
# ----------------------------------------------------------------------
def _fit_predict_transformer(train_df, eval_df, classes, cfg, logger):
    """Fine-tune a lightweight multilingual transformer; return eval predictions."""
    model_name = cfg["model"].get("name", DEFAULT_MODEL)
    logger.info(f"Transformer path: fine-tuning {model_name} on "
                f"{len(train_df)} examples (GPU={HAVE_GPU}).")
    tok = AutoTokenizer.from_pretrained(model_name)
    maxlen = cfg["model"].get("max_length", 256)

    def enc(texts):
        return tok(list(texts), truncation=True, padding=True,
                   max_length=maxlen, return_tensors="pt")

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.e = tok(list(texts), truncation=True, padding=True,
                         max_length=maxlen)
            self.labels = list(labels)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.e.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(classes))
    args = TrainingArguments(
        output_dir="results/_hf_tmp",
        num_train_epochs=cfg["model"].get("epochs", 3),
        per_device_train_batch_size=cfg["model"].get("batch_size", 16),
        per_device_eval_batch_size=32,
        learning_rate=cfg["model"].get("lr", 2e-5),
        logging_steps=50, save_strategy="no", report_to=[],
        seed=cfg["seed"], fp16=HAVE_GPU)
    trainer = Trainer(model=model, args=args,
                      train_dataset=DS(train_df["text"], train_df["y"]))
    trainer.train()
    pred = trainer.predict(DS(eval_df["text"], eval_df["y"]))
    return np.argmax(pred.predictions, axis=1)


def _fit_predict_tfidf(train_df, eval_df, classes, cfg, logger):
    """CPU fallback: TF-IDF + LinearSVM. Logic-equivalent, NOT the reportable model."""
    logger.warning("Transformer unavailable -> TF-IDF+LinearSVM FALLBACK "
                   "(runs the pipeline; do NOT report as the Phase-2 transformer result).")
    vec = TfidfVectorizer(max_features=cfg["model"].get("max_features", 5000),
                          ngram_range=(1, 2), sublinear_tf=True)
    Xtr = vec.fit_transform(train_df["text"]); Xte = vec.transform(eval_df["text"])
    clf = LinearSVC().fit(Xtr, train_df["y"])
    return clf.predict(Xte)


def _fit_predict(train_df, eval_df, classes, cfg, logger):
    use_tf = HAVE_TRANSFORMERS and not cfg["model"].get("force_fallback", False)
    if use_tf:
        try:
            return _fit_predict_transformer(train_df, eval_df, classes, cfg, logger), "transformer"
        except Exception as e:
            logger.warning(f"Transformer path failed ({e}); using TF-IDF fallback.")
    return _fit_predict_tfidf(train_df, eval_df, classes, cfg, logger), "tfidf_fallback"


def _per_language_scores(eval_df, y_true, y_pred, logger):
    """Macro-F1 and accuracy broken out by language."""
    out = {}
    langs = sorted(eval_df["lang"].unique())
    for l in langs:
        m = eval_df["lang"].values == l
        if m.sum() < 5:
            continue
        out[l] = {"n": int(m.sum()),
                  "macro_f1": float(f1_score(y_true[m], y_pred[m], average="macro")),
                  "accuracy": float(accuracy_score(y_true[m], y_pred[m]))}
        logger.info(f"  [{l}] n={out[l]['n']} macro-F1={out[l]['macro_f1']:.4f} "
                    f"acc={out[l]['accuracy']:.4f}")
    return out


def run(cfg, logger, smoke_test=False):
    seed = cfg["seed"]; set_seed(seed)
    df, synthetic = _load_multilingual(cfg, logger, smoke_test)
    df, classes, c2i = _encode_labels(df)
    results = {"experiment": "phase2_multilingual_classification",
               "synthetic": synthetic, "reportable": (not synthetic),
               "scope_note": "classification only; NER excluded (CoDA has no entity labels)",
               "model_path": None, "classes": classes, "seed": seed,
               "languages": {}, "evaluations": {}}

    # ---- (A) Pooled multilingual model, per-language reporting ----
    logger.info("=== (A) Pooled multilingual model ===")
    tr, te = train_test_split(df, test_size=cfg["data"]["test_size"],
                              stratify=df["y"], random_state=seed)
    y_pred, path = _fit_predict(tr, te, classes, cfg, logger)
    results["model_path"] = path
    y_true = te["y"].values
    overall = {"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
               "accuracy": float(accuracy_score(y_true, y_pred))}
    lo, hi = bootstrap_ci(y_true, y_pred, "macro_f1",
                          n_boot=cfg.get("bootstrap", 500), seed=seed)
    overall["macro_f1_ci95"] = [lo, hi]
    logger.info(f"Pooled overall macro-F1={overall['macro_f1']:.4f} "
                f"CI95=[{lo:.4f},{hi:.4f}] acc={overall['accuracy']:.4f}")
    logger.info("Per-language (pooled model):")
    per_lang = _per_language_scores(te, y_true, y_pred, logger)
    results["evaluations"]["pooled"] = {"overall": overall, "per_language": per_lang}

    # ---- (B) Cross-lingual transfer: train EN, test others ----
    if "en" in df["lang"].unique() and cfg.get("cross_lingual", True):
        logger.info("=== (B) Cross-lingual transfer: train=EN, test=other langs ===")
        en = df[df["lang"] == "en"]; non_en = df[df["lang"] != "en"]
        if len(non_en) >= 20 and en["y"].nunique() > 1:
            entr, ente = train_test_split(en, test_size=0.2,
                                          stratify=en["y"], random_state=seed)
            # evaluate on held-out EN AND on each non-EN language
            eval_pool = pd.concat([ente, non_en])
            yx, pathx = _fit_predict(entr, eval_pool, classes, cfg, logger)
            yt = eval_pool["y"].values
            xfer = _per_language_scores(eval_pool, yt, yx, logger)
            results["evaluations"]["cross_lingual_train_en"] = {
                "trained_on": "en", "per_language": xfer}
        else:
            logger.warning("Not enough non-English data for a cross-lingual test.")

    # record language counts actually used
    results["languages"] = {l: int((df["lang"] == l).sum())
                            for l in sorted(df["lang"].unique())}
    return results


def write_table7_fragment(results, out_path, logger):
    """Manuscript Table 7 (multilingual per-language) fragment."""
    flag = " [SMOKE-TEST/NON-REPORTABLE]" if results.get("synthetic") else ""
    rows = []
    pooled = results["evaluations"].get("pooled", {})
    for lang, m in pooled.get("per_language", {}).items():
        rows.append({"Evaluation": f"Pooled{flag}", "Language": lang,
                     "N": m["n"], "Macro_F1": round(m["macro_f1"], 4),
                     "Accuracy": round(m["accuracy"], 4)})
    xfer = results["evaluations"].get("cross_lingual_train_en", {})
    for lang, m in xfer.get("per_language", {}).items():
        rows.append({"Evaluation": f"Transfer(EN->){flag}", "Language": lang,
                     "N": m["n"], "Macro_F1": round(m["macro_f1"], 4),
                     "Accuracy": round(m["accuracy"], 4)})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Wrote Table 7 fragment -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multilingual.json")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("multilingual", cfg["paths"]["logs"])
    t0 = time.time()
    results = run(cfg, logger, smoke_test=args.smoke_test)
    tables = cfg["paths"]["tables"]
    if results["synthetic"]:
        save_json(results, os.path.join(tables, "multilingual_results_SMOKETEST.json"))
        write_table7_fragment(results, os.path.join(tables, "table7_multilingual_SMOKETEST.csv"), logger)
        logger.warning("SMOKE TEST complete — NON-REPORTABLE.")
    else:
        save_json(results, os.path.join(tables, "multilingual_results.json"))
        write_table7_fragment(results, os.path.join(tables, "table7_multilingual.csv"), logger)
        logger.info("Real-data run complete — Table 7 multilingual fragment written.")
    logger.info(f"Done in {time.time()-t0:.1f}s. Model path: {results['model_path']}")


if __name__ == "__main__":
    main()
