"""
darktrace_phase1/src/exp_traffic.py

Phase 1 / Experiment A: darknet traffic classification on CIC-Darknet2020.
Populates manuscript Table 6 (traffic rows) and feeds Section 8.7.

Baselines (manuscript Section 8.4): Random Forest and gradient-boosted trees.
Protocol (Section 8.3): stratified 5-fold CV, SMOTE inside training folds only,
fixed held-out test set. Reproduces the >99% / DarknetSec comparison context
WITHOUT claiming those external numbers as ours.

Run:
    python -m src.exp_traffic --config configs/traffic.yaml
If the real CSV is absent, a clearly-labelled SYNTHETIC dataset is generated so
the pipeline is testable; synthetic results are written with a `synthetic: true`
flag and must never be reported as real.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .metrics_stats import (classification_metrics, per_class_f1, aggregate_folds,
                            mcnemar_test, bootstrap_ci)
from .utils import load_config, ensure_dirs, get_logger, save_json, set_seed

try:
    from imblearn.over_sampling import SMOTE
    HAVE_SMOTE = True
except Exception:
    HAVE_SMOTE = False


CIC_TARGET_CANDIDATES = ["Label", "label", "Label.1", "class", "Class"]


def _make_synthetic(n=4000, n_features=24, seed=42):
    """Synthetic stand-in with CIC-like imbalance. LABELLED SYNTHETIC."""
    rng = np.random.RandomState(seed)
    # 4 application classes with imbalance similar to CIC-Darknet2020
    probs = [0.55, 0.25, 0.14, 0.06]
    y = rng.choice(["Browsing", "Chat", "P2P", "VOIP"], size=n, p=probs)
    X = rng.randn(n, n_features)
    # inject class-dependent signal so models can learn
    for i, cls in enumerate(["Browsing", "Chat", "P2P", "VOIP"]):
        mask = y == cls
        X[mask, i] += (i + 1) * 1.2
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols); df["Label"] = y
    return df


def load_dataset(cfg, logger):
    path = cfg["data"]["csv_path"]
    if os.path.exists(path):
        logger.info(f"Loading real CIC-Darknet2020 from {path}")
        df = pd.read_csv(path, low_memory=False)
        synthetic = False
    else:
        logger.warning(f"{path} not found -> generating SYNTHETIC data for pipeline test")
        df = _make_synthetic()
        synthetic = True

    # locate target column
    target = None
    for c in CIC_TARGET_CANDIDATES:
        if c in df.columns:
            target = c; break
    if target is None:
        target = df.columns[-1]
        logger.warning(f"No known label column; using last column '{target}'")

    # drop obvious identifier/leakage columns if present (per manuscript [3])
    drop_like = [c for c in df.columns if c.lower() in
                 {"flow id", "src ip", "source ip", "dst ip", "destination ip",
                  "timestamp", "flow.id"}]
    if drop_like:
        logger.info(f"Dropping identifier columns: {drop_like}")
        df = df.drop(columns=drop_like)

    y = df[target].astype(str).values
    X = df.drop(columns=[target])
    # keep numeric features only; coerce and fill
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.values, y, synthetic, list(X.columns)


def build_models(cfg, seed):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=cfg["model"]["rf_trees"], n_jobs=-1, random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
    }


def run(cfg, logger):
    seed = cfg["seed"]; set_seed(seed)
    X, y_raw, synthetic, feat_names = load_dataset(cfg, logger)
    le = LabelEncoder(); y = le.fit_transform(y_raw)
    labels = np.arange(len(le.classes_))

    # held-out test set fixed before CV (manuscript Section 8.3)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed)

    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
    models = build_models(cfg, seed)

    results = {"dataset": "CIC-Darknet2020", "synthetic": synthetic,
               "classes": le.classes_.tolist(), "seed": seed,
               "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
               "models": {}}

    test_preds = {}
    for name, base in models.items():
        logger.info(f"=== {name} ===")
        fold_metrics = []
        for k, (tri, vai) in enumerate(skf.split(X_tr, y_tr)):
            Xk, yk = X_tr[tri], y_tr[tri]
            scaler = StandardScaler().fit(Xk)
            Xk_s = scaler.transform(Xk); Xv_s = scaler.transform(X_tr[vai])
            # SMOTE inside the training fold only
            if cfg["use_smote"] and HAVE_SMOTE:
                try:
                    Xk_s, yk = SMOTE(random_state=seed).fit_resample(Xk_s, yk)
                except ValueError:
                    pass
            clf = build_models(cfg, seed)[name].fit(Xk_s, yk)
            pv = clf.predict(Xv_s)
            m = classification_metrics(y_tr[vai], pv, labels=labels)
            fold_metrics.append(m)
            logger.info(f"  fold {k}: macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")

        # refit on full train, evaluate on held-out test
        scaler = StandardScaler().fit(X_tr)
        Xtr_s = scaler.transform(X_tr); Xte_s = scaler.transform(X_te)
        ytr_fit = y_tr.copy(); Xtr_fit = Xtr_s
        if cfg["use_smote"] and HAVE_SMOTE:
            try:
                Xtr_fit, ytr_fit = SMOTE(random_state=seed).fit_resample(Xtr_s, y_tr)
            except ValueError:
                pass
        clf = models[name].fit(Xtr_fit, ytr_fit)
        proba = clf.predict_proba(Xte_s) if hasattr(clf, "predict_proba") else None
        pte = clf.predict(Xte_s); test_preds[name] = pte
        test_m = classification_metrics(y_te, pte, y_proba=proba, labels=labels)
        ci_lo, ci_hi = bootstrap_ci(y_te, pte, "macro_f1",
                                    n_boot=cfg["bootstrap"], seed=seed)
        results["models"][name] = {
            "cv": aggregate_folds(fold_metrics),
            "test": test_m,
            "test_macro_f1_ci95": [ci_lo, ci_hi],
            "test_per_class_f1": per_class_f1(y_te, pte, labels=labels),
        }
        logger.info(f"  TEST macro-F1={test_m['macro_f1']:.4f} "
                    f"CI95=[{ci_lo:.4f},{ci_hi:.4f}] acc={test_m['accuracy']:.4f}")

    # significance: best model vs the other (manuscript Section 8.13)
    names = list(models.keys())
    if len(names) == 2:
        p, stat = mcnemar_test(y_te, test_preds[names[0]], test_preds[names[1]])
        results["mcnemar"] = {"models": names, "p_value": p, "statistic": stat}
        logger.info(f"McNemar {names[0]} vs {names[1]}: p={p:.3e}")

    return results


def write_table6_fragment(results, out_path, logger):
    """Emit a manuscript-Table-6-style CSV fragment for the traffic rows."""
    rows = []
    syn = results.get("synthetic", False)
    flag = " (SYNTHETIC)" if syn else ""
    for name, r in results["models"].items():
        t = r["test"]
        rows.append({
            "Model_dataset": f"DarkTrace {name} (CIC-Darknet2020){flag}",
            "Accuracy": round(t["accuracy"], 4),
            "Macro_F1": round(t["macro_f1"], 4),
            "AUC": round(t["auc_macro_ovr"], 4) if t["auc_macro_ovr"] is not None else "NA",
            "FPR": round(t["fpr_macro"], 4),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Wrote Table 6 fragment -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/traffic.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("traffic", cfg["paths"]["logs"])
    t0 = time.time()
    results = run(cfg, logger)
    save_json(results, os.path.join(cfg["paths"]["tables"], "traffic_results.json"))
    write_table6_fragment(
        results, os.path.join(cfg["paths"]["tables"], "table6_traffic.csv"), logger)
    logger.info(f"Done in {time.time()-t0:.1f}s. "
                f"{'SYNTHETIC — not for reporting.' if results['synthetic'] else 'Real data.'}")


if __name__ == "__main__":
    main()
