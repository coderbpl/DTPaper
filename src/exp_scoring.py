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


# Category -> ordinal severity (0=low ... 3=critical). A defensible, STATED
# mapping when analyst labels are unavailable. Override via config data.severity_map.
DEFAULT_SEVERITY = {
    "arms": 3, "weapons": 3, "violence": 3, "hacking": 3,
    "drugs": 2, "financial": 2, "fraud": 2, "crypto": 2, "counterfeit": 2,
    "gambling": 1, "porn": 1, "electronic": 1,
    "others": 0,
}


def _risk_label(df, cfg, logger):
    """Produce the risk target. Three modes (config data.risk_mode):

      'analyst'  : load real per-item analyst risk labels from a CSV
                   (data.analyst_labels_csv with columns id/key + risk). This is
                   the gold standard; nothing is invented.
      'severity' : ordinal 0-3 from a STATED category->severity map (more
                   defensible than binary; reported as ordinal risk).
      'binary'   : high-risk vs low-risk from data.high_risk (legacy default).

    Returns (y, mode, meta). y is int. The mode and mapping are logged so the
    manuscript can state exactly how risk was operationalised.
    """
    mode = cfg["data"].get("risk_mode", "binary").lower()

    if mode == "analyst":
        path = cfg["data"].get("analyst_labels_csv")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"risk_mode='analyst' but analyst_labels_csv '{path}' not found. "
                "Provide a CSV with a key/id column and a numeric 'risk' column "
                "(one row per item), or switch data.risk_mode to 'severity'/'binary'.")
        lab = pd.read_csv(path)
        keycol = next((c for c in ("key", "__key__", "id", "doc_id") if c in lab.columns), None)
        riskcol = next((c for c in ("risk", "label", "score", "severity") if c in lab.columns), None)
        if keycol is None or riskcol is None:
            raise ValueError(f"analyst CSV must have a key column and a risk column; "
                             f"found {list(lab.columns)}")
        # align by key if df has one, else by row order
        if "__key__" in df.columns:
            m = dict(zip(lab[keycol].astype(str), lab[riskcol]))
            y = df["__key__"].astype(str).map(m)
            n_missing = int(y.isna().sum())
            if n_missing:
                logger.warning(f"{n_missing} items had no analyst label; dropping them.")
            keep = ~y.isna()
            df.drop(df.index[~keep.values], inplace=True)
            y = y[keep].astype(float).round().astype(int).values
        else:
            if len(lab) != len(df):
                raise ValueError("No __key__ to align on and analyst-label count "
                                 f"({len(lab)}) != item count ({len(df)}).")
            y = lab[riskcol].astype(float).round().astype(int).values
        logger.info(f"Risk label: ANALYST-PROVIDED from {path} "
                    f"({len(np.unique(y))} levels). This is reportable as real labels.")
        return y, "analyst", {"source": path, "levels": sorted(map(int, np.unique(y)))}

    if mode == "severity":
        sev = {k.lower(): v for k, v in cfg["data"].get("severity_map", DEFAULT_SEVERITY).items()}
        y = df["label"].astype(str).str.lower().map(sev)
        if y.isna().any():
            unknown = sorted(df["label"][y.isna()].str.lower().unique())
            logger.warning(f"Categories with no severity mapping (set to 0): {unknown}")
            y = y.fillna(0)
        y = y.astype(int).values
        dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        logger.warning(
            "Risk label: ORDINAL SEVERITY (0-3) from a STATED category map "
            f"(distribution={dist}). NOTE: derived from categories, not analyst "
            "ratings; state this in the manuscript.")
        return y, "severity", {"severity_map": sev, "distribution": dist}

    # default: binary
    high = set(c.lower() for c in cfg["data"].get("high_risk", DEFAULT_HIGH_RISK))
    y = df["label"].astype(str).str.lower().isin(high).astype(int).values
    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        raise ValueError("Binary risk label is constant; adjust data.high_risk.")
    logger.warning(
        f"Risk label: BINARY high vs low ({pos}/{len(y)}={100*pos/len(y):.1f}% high; "
        f"high={sorted(high)}). NOTE: derived from categories, not analyst ratings; "
        "state this in the manuscript.")
    return y, "binary", {"high_risk": sorted(high), "n_high": pos}


def _load_text_features(cfg, logger):
    """Build features + risk label from the real CoDA text corpus.

    Reuses the Phase-1 text loader so casing/validation/label-derivation are
    identical, then attaches a risk label (analyst / severity / binary).
    """
    from .exp_text import load_dataset as load_text
    df, class_report = load_text(cfg, logger)
    df = df.reset_index(drop=True)
    y, risk_mode, risk_meta = _risk_label(df, cfg, logger)
    df = df.reset_index(drop=True)
    # TF-IDF features (sparse -> dense for SHAP/LIME tractability)
    max_feats = cfg["model"].get("max_features", 2000)
    vec = TfidfVectorizer(max_features=max_feats, ngram_range=(1, 2),
                          sublinear_tf=True)
    X = vec.fit_transform(df["text"].astype(str).values)
    feat_names = list(vec.get_feature_names_out())
    meta = {"risk_mode": risk_mode, "risk_meta": risk_meta}
    return X.toarray().astype("float32"), y, feat_names, class_report, meta


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


def integrated_gradients(model, X, baseline=None, steps=32, max_active=60):
    """Model-agnostic Integrated Gradients for a non-differentiable scorer.

    IG attributes the change in predicted risk between a baseline x' and the
    input x to each feature, by integrating gradients along the straight-line
    path x' -> x. For non-differentiable sklearn models we approximate the
    gradient with central finite differences at each interpolation step.

    For tractability on dense inputs we attribute only the `max_active` features
    with the largest |x - baseline| (for sparse TF-IDF this is all nonzero
    features anyway). Returns |attribution| matrix of shape (len(X), d).

    This is a THIRD independent method to compare against SHAP and LIME, so the
    explanation evidence does not rest on any single technique.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if baseline is None:
        baseline = np.zeros(d)
    baseline = np.asarray(baseline, dtype=float)
    eps = 1e-3
    attrs = np.zeros((n, d))
    alphas = np.linspace(0, 1, steps)
    for i in range(n):
        diff = X[i] - baseline
        nz = np.where(np.abs(diff) > 0)[0]
        # cap to the most-changed features for speed on dense inputs
        if len(nz) > max_active:
            nz = nz[np.argsort(-np.abs(diff[nz]))[:max_active]]
        if len(nz) == 0:
            continue
        # build all path points at once, evaluate base probs in one batch
        path = baseline[None, :] + alphas[:, None] * diff[None, :]  # (steps, d)
        base_p = model.predict_proba(path)[:, 1]                    # (steps,)
        grad_sum = np.zeros(d)
        for j in nz:
            bumped = path.copy(); bumped[:, j] += eps
            gp = model.predict_proba(bumped)[:, 1]
            grad_sum[j] = np.sum((gp - base_p) / eps)
        attrs[i] = np.abs(diff * grad_sum / max(1, steps))
    return attrs


def three_method_agreement(shap_attr, lime_attr, ig_attr, logger, k=20):
    """Pairwise rank/set agreement between SHAP, LIME, and IG on the union of
    each pair's top-k features. Using three methods turns a single weak
    SHAP-LIME number into a proper multi-method robustness analysis: where two
    of three agree, the attribution is more trustworthy.
    """
    from scipy.stats import spearmanr
    methods = {"shap": shap_attr, "lime": lime_attr, "ig": ig_attr}
    avail = {m: a for m, a in methods.items() if a is not None}
    if len(avail) < 2:
        logger.info("Fewer than two attribution methods available; skipping.")
        return {"available": list(avail.keys())}
    n = min(len(a) for a in avail.values())
    d = next(iter(avail.values())).shape[1]
    kk = min(k, d)
    out = {"available": list(avail.keys()), "k": kk, "pairs": {}}
    names = list(avail.keys())
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            ma, mb = names[x], names[y]
            A, B = avail[ma], avail[mb]
            rhos, jaccs = [], []
            for i in range(n):
                at, bt = set(np.argsort(-A[i])[:kk]), set(np.argsort(-B[i])[:kk])
                uni = sorted(at | bt)
                if len(uni) >= 3:
                    a_u, b_u = A[i][uni], B[i][uni]
                    # spearmanr is undefined if either vector is constant over the
                    # union (e.g. a method assigned all-zero weights here); skip
                    # those rows rather than emit a ConstantInputWarning + NaN.
                    if np.ptp(a_u) > 0 and np.ptp(b_u) > 0:
                        rho, _ = spearmanr(a_u, b_u)
                        if not np.isnan(rho):
                            rhos.append(rho)
                inter = len(at & bt); u = len(at | bt)
                jaccs.append(inter / u if u else 0.0)
            pair = f"{ma}_vs_{mb}"
            out["pairs"][pair] = {
                "spearman_union": float(np.mean(rhos)) if rhos else None,
                "jaccard": float(np.mean(jaccs)) if jaccs else None, "n": len(jaccs)}
            logger.info(f"Agreement {pair}: Spearman(union)="
                        f"{out['pairs'][pair]['spearman_union']}, "
                        f"Jaccard={out['pairs'][pair]['jaccard']}")
    # consensus: mean pairwise Jaccard (higher => methods converge)
    js = [p["jaccard"] for p in out["pairs"].values() if p["jaccard"] is not None]
    out["mean_pairwise_jaccard"] = float(np.mean(js)) if js else None
    logger.info(f"Mean pairwise Jaccard across methods: {out['mean_pairwise_jaccard']}")
    return out


def lime_agreement(model, Xtr, Xte, attr, feat_names, seed, logger, n=30, k=20):
    """Convergent validity between SHAP and LIME attributions.

    IMPORTANT: LIME returns weights for only `k` features per instance, leaving
    the other ~(d-k) at zero. Correlating across the FULL feature space therefore
    forces near-zero agreement by construction (a measurement artifact, not real
    disagreement). We instead compare the two methods where both actually have
    signal: on the UNION of each method's top-k features. We report:
      - mean Spearman rho over that union (rank agreement on important features)
      - mean Jaccard overlap of the two top-k sets (set agreement)
    """
    if not HAVE_LIME:
        logger.info("LIME not available; skipping SHAP-LIME agreement.")
        return ({"lime_spearman": None, "lime_jaccard": None, "available": False},
                None)
    from scipy.stats import spearmanr
    expl = LimeTabularExplainer(
        Xtr, feature_names=feat_names, class_names=["low", "high"],
        mode="classification", discretize_continuous=False, random_state=seed)
    rhos, jaccs = [], []
    n = min(n, len(Xte), attr.shape[0])
    d = Xte.shape[1]
    kk = min(k, d)
    lime_mat = np.zeros((n, d))
    for i in range(n):
        e = expl.explain_instance(Xte[i], model.predict_proba,
                                  num_features=kk)
        lime_w = np.zeros(d)
        for idx, w in e.local_exp[1]:
            lime_w[idx] = abs(w)
        lime_mat[i] = lime_w
        shap_w = attr[i]
        shap_top = set(np.argsort(-shap_w)[:kk])
        lime_top = set(np.argsort(-lime_w)[:kk])
        # rank agreement on the union of important features
        union = sorted(shap_top | lime_top)
        if len(union) >= 3:
            rho, _ = spearmanr(shap_w[union], lime_w[union])
            if not np.isnan(rho):
                rhos.append(rho)
        # set agreement
        inter = len(shap_top & lime_top); uni = len(shap_top | lime_top)
        jaccs.append(inter / uni if uni else 0.0)
    rho_val = float(np.mean(rhos)) if rhos else None
    jac_val = float(np.mean(jaccs)) if jaccs else None
    logger.info(f"SHAP-LIME agreement (top-{kk}): Spearman(union)={rho_val}, "
                f"Jaccard={jac_val}")
    return ({"lime_spearman": rho_val, "lime_jaccard": jac_val,
             "available": True, "n": len(jaccs), "k": kk,
             "method": "union_topk"}, lime_mat)


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


def run(cfg, logger):
    seed = cfg["seed"]; set_seed(seed)
    source = cfg["data"].get("source", "text")

    if source == "text":
        X, y, feat_names, class_report, risk_meta = _load_text_features(cfg, logger)
    else:
        raise ValueError(f"Unsupported scoring source '{source}' in this build.")

    # The scoring/ranking machinery is binary. If the risk label has >2 levels
    # (ordinal severity or multi-level analyst labels), collapse to high-vs-low
    # at a STATED threshold for the binary scorer, and record it.
    y = np.asarray(y)
    binary_threshold = None
    if len(np.unique(y)) > 2:
        binary_threshold = int(cfg["data"].get("binary_threshold", 2))
        y_orig_levels = sorted(map(int, np.unique(y)))
        y = (y >= binary_threshold).astype(int)
        logger.warning(
            f"Collapsed {len(y_orig_levels)}-level risk {y_orig_levels} to binary "
            f"at threshold >= {binary_threshold} for the scorer "
            f"({int(y.sum())}/{len(y)} positive).")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed)

    # scale for the MLP / kernel methods
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)

    results = {"experiment": "phase3_explainable_scoring",
               "reportable": True,
               "source": source, "seed": seed,
               "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "n_features": int(X.shape[1]),
               "xai": {"shap": HAVE_SHAP, "lime": HAVE_LIME},
               "risk": risk_meta, "binary_threshold": binary_threshold,
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
    agree, lime_mat = lime_agreement(ens_cal, Xtr_s, Xeval, attr, feat_names, seed, logger,
                                     n=cfg.get("lime_n", 30))

    # Third attribution method: Integrated Gradients (model-agnostic).
    # Turns the weak two-method SHAP-LIME comparison into a 3-way robustness check.
    three_way = None
    if cfg.get("use_integrated_gradients", True):
        ig_n = min(cfg.get("ig_n", 30), len(Xeval))
        logger.info(f"Computing Integrated Gradients on {ig_n} instances...")
        ig_attr = integrated_gradients(ens_cal, Xeval[:ig_n],
                                       steps=cfg.get("ig_steps", 32))
        lime_for_cmp = lime_mat[:ig_n] if lime_mat is not None else None
        three_way = three_method_agreement(
            attr[:ig_n], lime_for_cmp, ig_attr, logger, k=cfg.get("agree_k", 20))

    results["models"]["explainable_ensemble"] = {
        "ranking": ens_rank,
        "macro_f1_at_0.5_ci95": [lo, hi],
        "attribution_method": attr_method,
        "faithfulness": fid, "stability": stab, "shap_lime_agreement": agree,
        "three_method_agreement": three_way,
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
    rows = []
    ens = results["models"].get("explainable_ensemble", {})
    bb = results["models"].get("blackbox_mlp", {})
    if ens:
        r = ens["ranking"]; f = ens["faithfulness"]; s = ens["stability"]
        rows.append({
            "Model": f"DarkTrace Explainable Ensemble",
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
            "Model": f"Black-box MLP (baseline)",
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
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("scoring", cfg["paths"]["logs"])
    t0 = time.time()
    results = run(cfg, logger)
    tables = cfg["paths"]["tables"]
    save_json(results, os.path.join(tables, "scoring_results.json"))
    write_table10_fragment(
        results, os.path.join(tables, "table10_scoring.csv"), logger)
    logger.info("Real-data run complete — Table 10 scoring fragment written.")
    logger.info(f"Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
