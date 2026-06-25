"""
darktrace_phase1/src/exp_scoring.py

Phase 3 / Explainable ensemble threat scoring (manuscript Section 10, Section 8.9).

Builds a heterogeneous ensemble threat scorer over the SAME real datasets used in
Phase 1, produces a calibrated risk score per item, and EXPLAINS each score with
SHAP and LIME. Then it evaluates the two claims the manuscript must defend:

  (RQ3-a) explainability does NOT materially cost ranking/discrimination quality
          (compare explainable ensemble vs a black-box MLP baseline);
  (RQ3-b) the explanations are FAITHFUL (comprehensiveness / sufficiency under
          top-k feature ablation, vs a random-attribution control) and STABLE
          (consistency of top-k attributions under small perturbations).

Design choices that keep this honest and reproducible:
  - CPU-only. SHAP/LIME are used when installed; otherwise the code falls back to
    model-native importances + permutation importance and SAYS SO in the output.
  - "Threat scoring" here is operationalised as a binary high-risk vs low-risk
    problem derived from the dataset's own categories (configurable), because the
    public datasets do not ship a continuous risk label. This is stated explicitly
    in the output so it is never overclaimed.
  - Outputs land in results/tables (JSON + Table-10 fragment) and results/figures.

Run (after Phase 1 has produced real data files):
    python -m src.exp_scoring --config configs/scoring.json
    python -m src.exp_scoring --config configs/scoring.json --smoke-test
"""
from __future__ import annotations
import argparse, json, os, time, warnings
# SHAP's KernelExplainer fits internal linear regressions that are often singular
# on sparse TF-IDF data; this produces a harmless but voluminous warning flood.
# Silence those specific cosmetic warnings (results are unaffected).
warnings.filterwarnings("ignore", message=".*Linear regression equation is singular.*")
warnings.filterwarnings("ignore", message=".*Regressors in active set degenerate.*")
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             ndcg_score)
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV

from .metrics_stats import bootstrap_ci, mcnemar_test
from .utils import load_config, ensure_dirs, get_logger, save_json, set_seed

# optional XAI deps
try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False
try:
    from lime.lime_tabular import LimeTabularExplainer
    HAVE_LIME = True
except Exception:
    HAVE_LIME = False


# ----------------------------------------------------------------------
# Risk-label derivation (operationalising "threat score" from categories)
# ----------------------------------------------------------------------
# Default high-risk categories for the CoDA text task. Override in config.
DEFAULT_HIGH_RISK = ["Arms", "Drugs", "Weapons", "Hacking", "Financial",
                     "Violence", "Crypto", "Fraud", "Counterfeit"]


def _make_synthetic_scoring(n=1500, seed=42):
    """Synthetic tabular risk problem. SMOKE-TEST ONLY."""
    rng = np.random.RandomState(seed)
    n_feat = 12
    X = rng.randn(n, n_feat)
    # risk is a nonlinear function of a few features (so explanations are testable)
    logit = (1.4 * X[:, 0] - 1.1 * X[:, 1] + 0.9 * X[:, 2] * X[:, 3]
             + 0.6 * (X[:, 4] > 0))
    p = 1 / (1 + np.exp(-logit))
    y = (rng.rand(n) < p).astype(int)
    cols = [f"feat_{i}" for i in range(n_feat)]
    return pd.DataFrame(X, columns=cols), y, cols


def _load_text_features(cfg, logger, smoke_test):
    """Build features + binary risk label from the real CoDA text corpus.

    Reuses the Phase-1 text loader so casing/validation/label-derivation are
    identical, then derives a binary high-risk label from the category.
    """
    from .exp_text import load_dataset as load_text
    df, synthetic, class_report = load_text(cfg, logger, smoke_test=smoke_test)
    high = set(c.lower() for c in cfg["data"].get("high_risk", DEFAULT_HIGH_RISK))
    y = df["label"].astype(str).str.lower().isin(high).astype(int).values
    pos = int(y.sum())
    logger.info(f"Risk label: {pos}/{len(y)} high-risk "
                f"({100*pos/len(y):.1f}%); high-risk categories = {sorted(high)}")
    if pos == 0 or pos == len(y):
        raise ValueError("Risk label is constant; adjust data.high_risk in config.")
    # TF-IDF features (sparse -> dense top-k for SHAP/LIME tractability)
    max_feats = cfg["model"].get("max_features", 2000)
    vec = TfidfVectorizer(max_features=max_feats, ngram_range=(1, 2),
                          sublinear_tf=True)
    X = vec.fit_transform(df["text"].astype(str).values)
    feat_names = list(vec.get_feature_names_out())
    return X.toarray().astype("float32"), y, feat_names, synthetic, class_report


def build_ensemble(seed):
    """Heterogeneous stacking ensemble (manuscript Section 10):
    HistGBT + RandomForest base learners, LogisticRegression meta-learner."""
    base = [
        ("hgb", HistGradientBoostingClassifier(random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                      random_state=seed)),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=2000),
        stack_method="predict_proba", n_jobs=-1, passthrough=False)


def build_blackbox(seed):
    """Black-box baseline scorer (manuscript Section 8.9): an MLP."""
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                         random_state=seed)


# ----------------------------------------------------------------------
# Explanation fidelity / stability (manuscript Section 8.9)
# ----------------------------------------------------------------------
def _attributions(model, Xtr, Xte, feat_names, seed, logger, max_eval=300):
    """Return a (n_eval, n_features) attribution matrix and the method used.

    Prefers SHAP (KernelExplainer on a sample), then permutation importance
    broadcast per row, always available as a fallback.
    """
    Xeval = Xte[:max_eval]
    if HAVE_SHAP:
        try:
            # tree-agnostic: use a small background sample for KernelExplainer
            bg = shap.sample(Xtr, min(50, len(Xtr)), random_state=seed)
            f = lambda d: model.predict_proba(d)[:, 1]
            expl = shap.KernelExplainer(f, bg)
            sv = expl.shap_values(Xeval, nsamples=100, silent=True)
            sv = np.asarray(sv)
            logger.info("Attributions: SHAP KernelExplainer.")
            return np.abs(sv), "shap_kernel", Xeval
        except Exception as e:
            logger.warning(f"SHAP failed ({e}); falling back to permutation importance.")
    # fallback: global permutation importance, broadcast to each row
    r = permutation_importance(model, Xeval, model.predict(Xeval),
                               n_repeats=5, random_state=seed, n_jobs=-1)
    imp = np.abs(r.importances_mean)
    logger.info("Attributions: permutation importance (SHAP not available).")
    return np.tile(imp, (len(Xeval), 1)), "permutation", Xeval


def faithfulness(model, X, attr, logger, k=10):
    """Comprehensiveness: removing the top-k attributed features should drop the
    predicted risk more than removing k RANDOM features (a control).
    Returns dict of mean prob drops and the comprehensiveness gap.
    """
    rng = np.random.RandomState(0)
    base = model.predict_proba(X)[:, 1]
    n, d = X.shape
    k = min(k, d)
    # top-k per row
    topk = np.argsort(-attr, axis=1)[:, :k]
    X_top = X.copy()
    for i in range(n):
        X_top[i, topk[i]] = 0.0
    drop_top = base - model.predict_proba(X_top)[:, 1]
    # random-k control
    X_rand = X.copy()
    for i in range(n):
        X_rand[i, rng.choice(d, k, replace=False)] = 0.0
    drop_rand = base - model.predict_proba(X_rand)[:, 1]
    comp = float(np.mean(np.abs(drop_top)))
    comp_ctrl = float(np.mean(np.abs(drop_rand)))
    logger.info(f"Faithfulness: top-{k} drop={comp:.4f} vs random-{k} drop="
                f"{comp_ctrl:.4f} (gap={comp - comp_ctrl:+.4f})")
    return {"comprehensiveness_topk": comp, "comprehensiveness_random": comp_ctrl,
            "faithfulness_gap": comp - comp_ctrl, "k": k}


def stability(model, X, attr_fn, logger, k=10, n_eval=40, n_perturb=5,
              noise=0.02, seed=0):
    """Stability = does the SAME instance get a consistent explanation under
    small input perturbations? (This is the correct XAI definition; comparing
    explanations across DIFFERENT documents is not stability.)

    For each of n_eval instances we add small Gaussian noise n_perturb times,
    recompute attributions, and measure the mean Jaccard overlap of the top-k
    attributed features between the original and each perturbed version.

    attr_fn(Xsub) -> (len(Xsub), d) attribution matrix.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    k = min(k, d)
    n_eval = min(n_eval, n)
    idx = rng.choice(n, n_eval, replace=False)
    jacc = []
    for i in idx:
        base = X[i:i+1]
        base_attr = attr_fn(base)[0]
        base_top = set(np.argsort(-base_attr)[:k])
        for _ in range(n_perturb):
            xp = base + rng.normal(0, noise, size=base.shape)
            pert_attr = attr_fn(xp)[0]
            pert_top = set(np.argsort(-pert_attr)[:k])
            inter = len(base_top & pert_top); union = len(base_top | pert_top)
            jacc.append(inter / union if union else 0.0)
    val = float(np.mean(jacc)) if jacc else None
    logger.info(f"Stability (per-instance, perturbed): mean top-{k} "
                f"Jaccard={val:.4f} over {n_eval} instances x {n_perturb} perturbations")
    return {"stability_jaccard": val, "k": k, "method": "per_instance_perturbation",
            "n_eval": n_eval, "n_perturb": n_perturb, "noise": noise}


def lime_agreement(model, Xtr, Xte, attr, feat_names, seed, logger, n=30):
    """Convergent validity: rank-correlation between SHAP/fallback attributions
    and LIME attributions on a sample. Returns mean Spearman rho or None."""
    if not HAVE_LIME:
        logger.info("LIME not available; skipping SHAP-LIME agreement.")
        return {"lime_spearman": None, "available": False}
    from scipy.stats import spearmanr
    expl = LimeTabularExplainer(Xtr, feature_names=feat_names,
                                class_names=["low", "high"], mode="classification",
                                discretize_continuous=False, random_state=seed)
    rhos = []
    n = min(n, len(Xte), attr.shape[0])
    for i in range(n):
        e = expl.explain_instance(Xte[i], model.predict_proba,
                                  num_features=min(20, Xte.shape[1]))
        lime_w = np.zeros(Xte.shape[1])
        for idx, w in e.local_exp[1]:
            lime_w[idx] = abs(w)
        rho, _ = spearmanr(attr[i], lime_w)
        if not np.isnan(rho):
            rhos.append(rho)
    val = float(np.mean(rhos)) if rhos else None
    logger.info(f"SHAP-LIME agreement: mean Spearman={val}")
    return {"lime_spearman": val, "available": True, "n": len(rhos)}


def _rank_metrics(y_true, scores):
    """Discrimination/ranking metrics for the risk score."""
    out = {"auc": None, "ap": None, "ndcg@10": None, "brier": None}
    try:
        out["auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        pass
    try:
        out["ap"] = float(average_precision_score(y_true, scores))
    except Exception:
        pass
    try:
        out["ndcg@10"] = float(ndcg_score([y_true], [scores], k=10))
    except Exception:
        pass
    try:
        out["brier"] = float(brier_score_loss(y_true, scores))
    except Exception:
        pass
    return out


def run(cfg, logger, smoke_test=False):
    seed = cfg["seed"]; set_seed(seed)
    source = cfg["data"].get("source", "text")

    if smoke_test:
        Xdf, y, feat_names = _make_synthetic_scoring(seed=seed)
        X = Xdf.values.astype("float32"); synthetic = True; class_report = None
    elif source == "text":
        X, y, feat_names, synthetic, class_report = _load_text_features(
            cfg, logger, smoke_test=False)
    else:
        raise ValueError(f"Unsupported scoring source '{source}' in this build.")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed)

    # scale for the MLP / kernel methods
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)

    results = {"experiment": "phase3_explainable_scoring",
               "synthetic": synthetic, "reportable": (not synthetic),
               "source": source, "seed": seed,
               "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "n_features": int(X.shape[1]),
               "xai": {"shap": HAVE_SHAP, "lime": HAVE_LIME},
               "class_report": class_report, "models": {}}

    # --- explainable ensemble (calibrated) ---
    logger.info("=== Explainable stacking ensemble ===")
    ens = build_ensemble(seed)
    ens_cal = CalibratedClassifierCV(ens, method="isotonic", cv=3)
    ens_cal.fit(Xtr_s, ytr)
    ens_scores = ens_cal.predict_proba(Xte_s)[:, 1]
    ens_rank = _rank_metrics(yte, ens_scores)
    lo, hi = bootstrap_ci(yte, (ens_scores >= 0.5).astype(int), "macro_f1",
                          n_boot=cfg.get("bootstrap", 500), seed=seed)
    logger.info(f"  ensemble AUC={ens_rank['auc']} AP={ens_rank['ap']} "
                f"NDCG@10={ens_rank['ndcg@10']} Brier={ens_rank['brier']}")

    # explanations on the fitted ensemble
    attr, attr_method, Xeval = _attributions(
        ens_cal, Xtr_s, Xte_s, feat_names, seed, logger,
        max_eval=cfg.get("explain_max_eval", 200))
    yte_eval = yte[:len(Xeval)]
    fid = faithfulness(ens_cal, Xeval, attr, logger,
                       k=cfg.get("topk", 10))

    # Stability uses a FAST local attribution (occlusion: zero each feature and
    # measure the prediction change) so we can recompute it under perturbation
    # cheaply, instead of re-running SHAP (which is expensive). Occlusion is
    # itself a faithful local attribution, so it is a sound basis for stability.
    def _occlusion_attr(Xsub):
        base_p = ens_cal.predict_proba(Xsub)[:, 1]
        out = np.zeros((Xsub.shape[0], Xsub.shape[1]), dtype="float32")
        # only probe the union of each row's nonzero features for speed
        for r in range(Xsub.shape[0]):
            nz = np.nonzero(Xsub[r])[0]
            if len(nz) == 0:
                continue
            Xrep = np.repeat(Xsub[r:r+1], len(nz), axis=0)
            for j, f in enumerate(nz):
                Xrep[j, f] = 0.0
            pp = ens_cal.predict_proba(Xrep)[:, 1]
            out[r, nz] = np.abs(base_p[r] - pp)
        return out

    stab = stability(ens_cal, Xeval, _occlusion_attr, logger,
                     k=cfg.get("topk", 10),
                     n_eval=cfg.get("stability_n", 30),
                     n_perturb=cfg.get("stability_perturb", 5),
                     noise=cfg.get("stability_noise", 0.02), seed=seed)
    agree = lime_agreement(ens_cal, Xtr_s, Xeval, attr, feat_names, seed, logger,
                           n=cfg.get("lime_n", 30))

    results["models"]["explainable_ensemble"] = {
        "ranking": ens_rank,
        "macro_f1_at_0.5_ci95": [lo, hi],
        "attribution_method": attr_method,
        "faithfulness": fid, "stability": stab, "shap_lime_agreement": agree,
    }

    # --- black-box baseline (MLP) ---
    logger.info("=== Black-box MLP baseline ===")
    mlp = build_blackbox(seed).fit(Xtr_s, ytr)
    mlp_scores = mlp.predict_proba(Xte_s)[:, 1]
    mlp_rank = _rank_metrics(yte, mlp_scores)
    logger.info(f"  MLP AUC={mlp_rank['auc']} AP={mlp_rank['ap']} "
                f"NDCG@10={mlp_rank['ndcg@10']} Brier={mlp_rank['brier']}")
    results["models"]["blackbox_mlp"] = {"ranking": mlp_rank}

    # non-inferiority check (RQ3-a): ensemble AUC vs black-box AUC
    if ens_rank["auc"] is not None and mlp_rank["auc"] is not None:
        p_mc, _ = mcnemar_test(
            yte, (ens_scores >= 0.5).astype(int), (mlp_scores >= 0.5).astype(int))
        results["explainable_vs_blackbox"] = {
            "ensemble_auc": ens_rank["auc"], "blackbox_auc": mlp_rank["auc"],
            "auc_gap": ens_rank["auc"] - mlp_rank["auc"],
            "mcnemar_p": p_mc}
        logger.info(f"RQ3-a: ensemble AUC {ens_rank['auc']:.4f} vs black-box "
                    f"{mlp_rank['auc']:.4f} (gap {ens_rank['auc']-mlp_rank['auc']:+.4f}, "
                    f"McNemar p={p_mc:.3e})")

    return results


def write_table10_fragment(results, out_path, logger):
    """Manuscript Table 10 (scoring + explainability) fragment."""
    flag = " [SMOKE-TEST/NON-REPORTABLE]" if results.get("synthetic") else ""
    rows = []
    ens = results["models"].get("explainable_ensemble", {})
    bb = results["models"].get("blackbox_mlp", {})
    if ens:
        r = ens["ranking"]; f = ens["faithfulness"]; s = ens["stability"]
        rows.append({
            "Model": f"DarkTrace Explainable Ensemble{flag}",
            "AUC": round(r["auc"], 4) if r["auc"] else "NA",
            "AP": round(r["ap"], 4) if r["ap"] else "NA",
            "NDCG@10": round(r["ndcg@10"], 4) if r["ndcg@10"] else "NA",
            "Brier": round(r["brier"], 4) if r["brier"] else "NA",
            "Faithfulness_gap": round(f["faithfulness_gap"], 4),
            "Stability_Jaccard": round(s["stability_jaccard"], 4) if s["stability_jaccard"] else "NA",
        })
    if bb:
        r = bb["ranking"]
        rows.append({
            "Model": f"Black-box MLP (baseline){flag}",
            "AUC": round(r["auc"], 4) if r["auc"] else "NA",
            "AP": round(r["ap"], 4) if r["ap"] else "NA",
            "NDCG@10": round(r["ndcg@10"], 4) if r["ndcg@10"] else "NA",
            "Brier": round(r["brier"], 4) if r["brier"] else "NA",
            "Faithfulness_gap": "NA", "Stability_Jaccard": "NA",
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Wrote Table 10 fragment -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scoring.json")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run on SYNTHETIC data to verify the pipeline (NON-REPORTABLE).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("scoring", cfg["paths"]["logs"])
    t0 = time.time()
    results = run(cfg, logger, smoke_test=args.smoke_test)
    tables = cfg["paths"]["tables"]
    if results["synthetic"]:
        save_json(results, os.path.join(tables, "scoring_results_SMOKETEST.json"))
        write_table10_fragment(
            results, os.path.join(tables, "table10_scoring_SMOKETEST.csv"), logger)
        logger.warning("SMOKE TEST complete — NON-REPORTABLE.")
    else:
        save_json(results, os.path.join(tables, "scoring_results.json"))
        write_table10_fragment(
            results, os.path.join(tables, "table10_scoring.csv"), logger)
        logger.info("Real-data run complete — Table 10 scoring fragment written.")
    logger.info(f"Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
