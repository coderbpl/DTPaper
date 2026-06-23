"""
darktrace_phase1/src/exp_traffic.py

Phase 1 / Experiment A: darknet traffic classification on CIC-Darknet2020.
Populates manuscript Table 6 (traffic rows) and feeds Section 8.7.

Baselines (manuscript Section 8.4): Random Forest and gradient-boosted trees.
Protocol (Section 8.3): stratified 5-fold CV, SMOTE inside training folds only,
fixed held-out test set.

MODES
-----
Normal (default): runs ONLY on the real CIC-Darknet2020 CSV. If the file is
missing, the experiment fails hard (no silent synthetic fallback). Table 6
outputs are produced only in this mode and are reportable.

Smoke test (--smoke-test): generates clearly-labelled SYNTHETIC data to verify
the pipeline executes. Outputs are written to a separate `*_SMOKETEST.*` path,
tagged non-reportable, and must never appear in the manuscript.

Run:
    python -m src.exp_traffic --config configs/traffic.json
    python -m src.exp_traffic --config configs/traffic.json --smoke-test
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
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


CIC_TARGET_CANDIDATES = ["Label", "label", "Label.1", "class", "Class",
                         "Application", "Traffic Type", "Category"]

# identifier / leakage columns to remove from traffic data (manuscript [3]).
# Such fields (IPs, ports, timestamps, flow IDs) can leak the label and inflate
# accuracy; several published >99% CIC results may stem from not removing them.
LEAKAGE_COLUMNS = {
    "flow id", "flow.id", "flow id", "src ip", "source ip", "srcip",
    "dst ip", "destination ip", "dstip", "src port", "source port",
    "srcport", "dst port", "destination port", "dstport", "timestamp",
    "flow duration timestamp", "id", "unnamed: 0",
}


class MissingRealDataError(FileNotFoundError):
    """Raised in normal (reportable) mode when the real dataset is absent."""


def _make_synthetic(n=4000, n_features=24, seed=42):
    """Synthetic stand-in with CIC-like imbalance. SMOKE-TEST ONLY."""
    rng = np.random.RandomState(seed)
    probs = [0.55, 0.25, 0.14, 0.06]
    y = rng.choice(["Browsing", "Chat", "P2P", "VOIP"], size=n, p=probs)
    X = rng.randn(n, n_features)
    for i, cls in enumerate(["Browsing", "Chat", "P2P", "VOIP"]):
        mask = y == cls
        X[mask, i] += (i + 1) * 1.2
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols); df["Label"] = y
    return df


def _report_class_distribution(y, logger, title="Class distribution"):
    """Log class counts and proportions (manuscript Section 8.2 reporting)."""
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


def _auto_detect_label(df, logger):
    """Find the label column by known names, else fall back to last column."""
    for c in CIC_TARGET_CANDIDATES:
        if c in df.columns:
            logger.info(f"Auto-detected label column: '{c}'")
            return c
    # case-insensitive match
    lower = {c.lower(): c for c in df.columns}
    for cand in CIC_TARGET_CANDIDATES:
        if cand.lower() in lower:
            logger.info(f"Auto-detected label column (ci): '{lower[cand.lower()]}'")
            return lower[cand.lower()]
    target = df.columns[-1]
    logger.warning(f"No known label column; using last column '{target}'")
    return target


def load_dataset(cfg, logger, smoke_test=False):
    """Load CIC-Darknet2020. In normal mode the real CSV is mandatory."""
    path = cfg["data"]["csv_path"]

    if smoke_test:
        logger.warning("SMOKE-TEST MODE: generating SYNTHETIC data (NON-REPORTABLE).")
        df = _make_synthetic()
        synthetic = True
    else:
        if not os.path.exists(path):
            raise MissingRealDataError(
                f"Real CIC-Darknet2020 CSV not found at '{path}'.\n"
                f"Normal mode requires real data for reportable results.\n"
                f"  - Download it (see DATASETS.md) and place it at that path, or\n"
                f"  - run with --smoke-test to exercise the pipeline on synthetic data.")
        logger.info(f"Loading real CIC-Darknet2020 from {path}")
        df = pd.read_csv(path, low_memory=False)
        synthetic = False

    # strip whitespace from column names (CIC exports often have leading spaces)
    df.columns = [str(c).strip() for c in df.columns]

    target = _auto_detect_label(df, logger)

    # remove identifier/leakage columns (manuscript [3])
    drop_like = [c for c in df.columns if c.strip().lower() in LEAKAGE_COLUMNS]
    if drop_like:
        logger.info(f"Dropping {len(drop_like)} identifier/leakage column(s): {drop_like}")
        df = df.drop(columns=drop_like)

    # basic integrity checks on real data
    if not synthetic:
        n_before = len(df)
        df = df.dropna(subset=[target])
        if len(df) < n_before:
            logger.info(f"Dropped {n_before - len(df)} rows with missing label.")
        if len(df) == 0:
            raise ValueError("No rows remain after label cleaning; check the CSV.")

    y = df[target].astype(str).str.strip().values
    X = df.drop(columns=[target])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    class_report = _report_class_distribution(y, logger, "CIC-Darknet2020 classes")
    return X.values, y, synthetic, list(X.columns), class_report


def _build_gradient_booster(cfg, seed):
    model_cfg = cfg.get("model", {})
    variant = str(model_cfg.get("gbt_variant", "hist")).lower()
    max_iter = int(model_cfg.get("gbt_max_iter", 100))
    learning_rate = float(model_cfg.get("gbt_learning_rate", 0.1))
    max_leaf_nodes = int(model_cfg.get("gbt_max_leaf_nodes", 31))

    if variant in {"hist", "histgradientboosting", "hist_gradient_boosting"}:
        return "HistGradientBoosting", HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            random_state=seed,
        )
    if variant in {"legacy", "classic", "gradientboosting", "gradient_boosting"}:
        return "GradientBoosting", GradientBoostingClassifier(
            n_estimators=max_iter,
            learning_rate=learning_rate,
            random_state=seed,
        )
    if variant in {"lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "gbt_variant='lightgbm' requires lightgbm. On Kaggle this may "
                "already be available; otherwise install it or use "
                "gbt_variant='hist'."
            ) from exc
        return "LightGBM", LGBMClassifier(
            n_estimators=max_iter,
            learning_rate=learning_rate,
            num_leaves=max_leaf_nodes,
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
        )
    raise ValueError(
        "Unknown model.gbt_variant "
        f"'{model_cfg.get('gbt_variant')}'. Use 'hist', 'legacy', or 'lightgbm'."
    )


def build_models(cfg, seed):
    gbt_name, gbt_model = _build_gradient_booster(cfg, seed)
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=cfg["model"]["rf_trees"], n_jobs=-1, random_state=seed),
        gbt_name: gbt_model,
    }


def run(cfg, logger, smoke_test=False):
    seed = cfg["seed"]; set_seed(seed)
    X, y_raw, synthetic, feat_names, class_report = load_dataset(
        cfg, logger, smoke_test=smoke_test)
    le = LabelEncoder(); y = le.fit_transform(y_raw)
    labels = np.arange(len(le.classes_))

    # held-out test set fixed before CV (manuscript Section 8.3)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed)

    skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
    models = build_models(cfg, seed)

    results = {"dataset": "CIC-Darknet2020", "synthetic": synthetic,
               "reportable": (not synthetic),
               "class_report": class_report,
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
    flag = " [SMOKE-TEST/NON-REPORTABLE]" if syn else ""
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
    ap.add_argument("--config", default="configs/traffic.json")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run on SYNTHETIC data to verify the pipeline (NON-REPORTABLE).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("traffic", cfg["paths"]["logs"])
    t0 = time.time()

    try:
        results = run(cfg, logger, smoke_test=args.smoke_test)
    except MissingRealDataError as e:
        logger.error(str(e))
        raise SystemExit(2)

    tables = cfg["paths"]["tables"]
    if results["synthetic"]:
        # smoke-test outputs go to clearly separated, non-reportable paths
        save_json(results, os.path.join(tables, "traffic_results_SMOKETEST.json"))
        write_table6_fragment(
            results, os.path.join(tables, "table6_traffic_SMOKETEST.csv"), logger)
        logger.warning("SMOKE TEST complete — outputs marked NON-REPORTABLE. "
                       "Table 6 fragment NOT written (real data required).")
    else:
        save_json(results, os.path.join(tables, "traffic_results.json"))
        write_table6_fragment(
            results, os.path.join(tables, "table6_traffic.csv"), logger)
        logger.info("Real-data run complete — Table 6 traffic fragment written.")

    logger.info(f"Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
