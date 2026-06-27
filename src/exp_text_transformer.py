"""
darktrace_phase1/src/exp_text_transformer.py

Phase 1b / Transformer baseline on CoDA dark-web TEXT (GPU).

Adds the state-of-the-art transformer head-to-head that reviewers expect on the
text task, alongside the classical TF-IDF baselines in exp_text. It uses the
SAME train/test split and SAME stratified folds (identical seed) as exp_text so
its per-fold macro-F1 vector is directly comparable in the Friedman+Nemenyi
omnibus test (Section 8.13) and produces a real critical-difference diagram.

Outputs (mirroring exp_text):
    results/tables/text_transformer_results.json
    results/preds/text_Transformer.npz        (test predictions)
    results/preds/folds_Transformer.json      (per-fold macro-F1, for Friedman)

Real-data + GPU only: exits 2 (SKIP, not FAIL) if the corpus is missing or
transformers/torch are unavailable, so run_all does not break on CPU.

Run:
    python -m src.exp_text_transformer --config configs/text_transformer.json
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

from .metrics_stats import classification_metrics, per_class_f1, save_predictions, bootstrap_ci
from .utils import load_config, ensure_dirs, get_logger, save_json, set_seed
from .exp_text import load_dataset, MissingRealDataError

try:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    HAVE_TRANSFORMERS = True
    HAVE_GPU = torch.cuda.is_available()
except Exception:
    HAVE_TRANSFORMERS = False
    HAVE_GPU = False


def _fit_predict(model_name, train_texts, train_y, eval_texts, n_classes, cfg, logger):
    tok = AutoTokenizer.from_pretrained(model_name)
    maxlen = cfg["model"].get("max_length", 256)

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.e = tok(list(texts), truncation=True, padding=True, max_length=maxlen)
            self.labels = list(labels)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.e.items()}
            item["labels"] = torch.tensor(int(self.labels[i]))
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes)
    args = TrainingArguments(
        output_dir="results/_hf_tmp_text", num_train_epochs=cfg["model"].get("epochs", 3),
        per_device_train_batch_size=cfg["model"].get("batch_size", 16),
        per_device_eval_batch_size=32, learning_rate=cfg["model"].get("lr", 2e-5),
        logging_steps=50, save_strategy="no", report_to=[], seed=cfg["seed"], fp16=HAVE_GPU)
    trainer = Trainer(model=model, args=args, train_dataset=DS(train_texts, train_y))
    trainer.train()
    pred = trainer.predict(DS(eval_texts, np.zeros(len(eval_texts))))
    return np.argmax(pred.predictions, axis=1)


def run(cfg, logger):
    if not HAVE_TRANSFORMERS:
        raise MissingRealDataError(
            "transformers/torch not available; transformer baseline requires a GPU "
            "environment (e.g. Kaggle GPU). This phase is SKIPPED on CPU.")
    seed = cfg["seed"]; set_seed(seed)
    model_name = cfg["model"].get("name", "xlm-roberta-base")
    df, class_report = load_dataset(cfg, logger)
    classes = sorted(df["label"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.asarray(df["label"].map(c2i).tolist(), dtype=int)
    texts = np.asarray(df["text"].astype(str).tolist(), dtype=object)

    # SAME split + folds as exp_text (identical seed/test_size) -> comparable blocks
    Xtr, Xte, ytr, yte = train_test_split(
        texts, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed)
    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)

    logger.info(f"=== Transformer baseline: {model_name} (GPU={HAVE_GPU}) ===")
    fold_macro_f1 = []
    for k, (tri, vai) in enumerate(skf.split(Xtr, ytr)):
        pv = _fit_predict(model_name, Xtr[tri], ytr[tri], Xtr[vai], len(classes), cfg, logger)
        f1 = float(f1_score(ytr[vai], pv, average="macro", zero_division=0))
        fold_macro_f1.append(f1)
        logger.info(f"  fold {k}: macro-F1={f1:.4f}")

    pte = _fit_predict(model_name, Xtr, ytr, Xte, len(classes), cfg, logger)
    m = classification_metrics(yte, pte, labels=np.arange(len(classes)))
    lo, hi = bootstrap_ci(yte, pte, "macro_f1", n_boot=cfg.get("bootstrap", 1000), seed=seed)
    logger.info(f"  TEST macro-F1={m['macro_f1']:.4f} CI95=[{lo:.4f},{hi:.4f}]")

    preds_dir = os.path.join(os.path.dirname(cfg["paths"]["tables"]), "preds")
    save_predictions(os.path.join(preds_dir, "text_Transformer.npz"), yte, pte,
                     meta={"task": "text", "model": model_name, "classes": classes})
    json.dump({"model": "Transformer", "model_name": model_name,
               "fold_macro_f1": fold_macro_f1},
              open(os.path.join(preds_dir, "folds_Transformer.json"), "w"))

    return {"experiment": "phase1b_text_transformer", "reportable": True,
            "model_name": model_name, "seed": seed, "classes": classes,
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "cv_macro_f1_mean": float(np.mean(fold_macro_f1)),
            "cv_macro_f1_sd": float(np.std(fold_macro_f1)),
            "test": m, "test_macro_f1_ci95": [lo, hi],
            "test_per_class_f1": per_class_f1(yte, pte, labels=np.arange(len(classes)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/text_transformer.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("text_transformer", cfg["paths"]["logs"])
    t0 = time.time()
    try:
        results = run(cfg, logger)
    except MissingRealDataError as e:
        logger.error(str(e)); raise SystemExit(2)
    save_json(results, os.path.join(cfg["paths"]["tables"], "text_transformer_results.json"))
    logger.info(f"Done in {time.time()-t0:.1f}s. Transformer baseline written.")


if __name__ == "__main__":
    main()
