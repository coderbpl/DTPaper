"""
darktrace_phase1/src/metrics_stats.py

Shared evaluation metrics and statistical-testing utilities for DarkTrace Phase 1.
These functions populate the values reported in manuscript Tables 6 (classification)
and support the significance testing described in manuscript Section 8.13.

All functions are deterministic given a seed and have no GPU dependency.
"""
from __future__ import annotations
import os
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    roc_auc_score, confusion_matrix,
)


# ----------------------------------------------------------------------
# Classification metrics (manuscript Table 6 / Section 8.16)
# ----------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_proba=None, labels=None):
    """Return the standard metric bundle for one fold or one test set.

    AUC is macro one-vs-rest when probabilities are supplied; otherwise None.
    FPR is macro-averaged from the confusion matrix.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = None

    # macro false-positive rate from the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fpr_per_class = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        fp = cm[:, i].sum() - cm[i, i]
        tn = total - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        denom = fp + tn
        fpr_per_class.append(fp / denom if denom > 0 else 0.0)
    fpr_macro = float(np.mean(fpr_per_class))

    return {
        "accuracy": acc,
        "precision_macro": p_macro, "recall_macro": r_macro, "macro_f1": f1_macro,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f1_w,
        "auc_macro_ovr": auc, "fpr_macro": fpr_macro,
    }


def per_class_f1(y_true, y_pred, labels=None):
    """Per-class F1, for the minority-class analysis in manuscript Section 8.18."""
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {}
    lab = labels if labels is not None else sorted(np.unique(y_true))
    for i, c in enumerate(lab):
        out[str(c)] = {"precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f1[i]), "support": int(support[i])}
    return out


# ----------------------------------------------------------------------
# Statistical significance (manuscript Section 8.13)
# ----------------------------------------------------------------------
def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test on paired predictions (manuscript ref [26]).

    Returns (p_value, statistic). Uses the continuity-corrected chi-square.
    """
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    a_correct = (pred_a == y_true); b_correct = (pred_b == y_true)
    n01 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    n10 = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    if n01 + n10 == 0:
        return 1.0, 0.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = float(stats.chi2.sf(stat, 1))
    return p, float(stat)


def paired_fold_test(scores_a, scores_b):
    """Compare two models across CV folds.

    Uses a paired t-test, falling back to Wilcoxon signed-rank when the
    differences fail a Shapiro normality check (manuscript Section 8.13).
    Returns dict with test name, p-value, and mean difference.
    """
    a = np.asarray(scores_a, float); b = np.asarray(scores_b, float)
    diff = a - b
    if len(diff) < 3:
        return {"test": "insufficient_folds", "p_value": None,
                "mean_diff": float(np.mean(diff))}
    # normality of differences
    try:
        _, p_norm = stats.shapiro(diff)
    except Exception:
        p_norm = 1.0
    if p_norm > 0.05:
        t, p = stats.ttest_rel(a, b)
        name = "paired_t"
    else:
        try:
            _, p = stats.wilcoxon(a, b)
        except ValueError:
            p = 1.0
        name = "wilcoxon"
    return {"test": name, "p_value": float(p), "mean_diff": float(np.mean(diff))}


def bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=1000, seed=0, alpha=0.05):
    """Bootstrap confidence interval for a metric (manuscript Section 8.13).

    metric in {"macro_f1", "accuracy"}.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if metric == "macro_f1":
            vals.append(f1_score(y_true[s], y_pred[s], average="macro", zero_division=0))
        elif metric == "accuracy":
            vals.append(accuracy_score(y_true[s], y_pred[s]))
        else:
            raise ValueError(f"unknown metric {metric}")
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_bonferroni(pvalues: dict, alpha=0.05):
    """Holm-Bonferroni correction over a dict {name: p}. Returns {name: (p, reject)}."""
    items = sorted(pvalues.items(), key=lambda kv: (kv[1] is None, kv[1]))
    m = len(items)
    out = {}
    prev_reject = True
    for rank, (name, p) in enumerate(items):
        if p is None:
            out[name] = (None, False); prev_reject = False; continue
        thresh = alpha / (m - rank)
        reject = (p < thresh) and prev_reject
        out[name] = (p, reject)
        prev_reject = reject
    return out


# ----------------------------------------------------------------------
# Added: confidence intervals, omnibus tests, paired bootstrap, I/O
# (manuscript Section 8.13 — statistical validation)
# ----------------------------------------------------------------------
def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion (e.g. accuracy).

    Exact from counts only: needs the number of correct predictions ``k`` and
    the sample size ``n`` — no per-instance data required. Preferred over the
    normal approximation for small n and proportions near 0/1.
    Returns (point, lo, hi).
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(p), float(max(0.0, center - half)), float(min(1.0, center + half)))


def wilson_ci_from_acc(acc, n, alpha=0.05):
    """Wilson interval when only the accuracy and n are stored (rounds k=acc*n)."""
    k = int(round(acc * n))
    return wilson_ci(k, n, alpha)


# Nemenyi critical values q_alpha for the two-tailed test at alpha=0.05,
# indexed by number of models k (Demsar 2006, Table 5).
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def friedman_nemenyi(score_matrix, names=None, alpha=0.05):
    """Friedman omnibus test + Nemenyi post-hoc critical difference.

    ``score_matrix`` is shape (N_datasets_or_folds, k_models); higher is better.
    Returns dict with Friedman statistic/p, mean ranks, and the Nemenyi CD.
    Two models differ significantly if |rank_i - rank_j| > CD.
    (manuscript Section 8.13; Demsar 2006, ref [25].)
    """
    M = np.asarray(score_matrix, float)
    N, k = M.shape
    if names is None:
        names = [f"m{i}" for i in range(k)]
    # rank within each row (1 = worst, k = best) on -scores so ties handled
    ranks = np.zeros_like(M)
    for i in range(N):
        ranks[i] = stats.rankdata(M[i])           # higher score -> higher rank
    mean_ranks = ranks.mean(axis=0)
    try:
        chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    except Exception:
        chi2, p = float("nan"), float("nan")
    q = _NEMENYI_Q05.get(k)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * N))) if q else None
    return {
        "n_blocks": int(N), "k_models": int(k),
        "friedman_chi2": float(chi2), "friedman_p": float(p),
        "mean_ranks": {names[j]: float(mean_ranks[j]) for j in range(k)},
        "nemenyi_cd": cd, "alpha": alpha,
    }


def _metric_value(y_true, y_pred, metric):
    if metric == "macro_f1":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    raise ValueError(f"unknown metric {metric}")


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric="macro_f1",
                          n_boot=1000, seed=0, alpha=0.05):
    """Paired bootstrap for the difference metric(A) - metric(B) on shared data.

    Resamples instance indices once per replicate and applies them to BOTH
    models (paired), so it works for any metric incl. macro-F1 where McNemar
    does not apply. Returns dict with observed diff, CI, and a two-sided
    bootstrap p-value (H0: diff = 0).
    """
    y_true = np.asarray(y_true); pa = np.asarray(pred_a); pb = np.asarray(pred_b)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y_true))
    obs = _metric_value(y_true, pa, metric) - _metric_value(y_true, pb, metric)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs[b] = _metric_value(y_true[s], pa[s], metric) - _metric_value(y_true[s], pb[s], metric)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    # two-sided bootstrap p-value via the proportion crossing zero
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return {"metric": metric, "observed_diff": float(obs),
            "ci95": [lo, hi], "p_value": float(min(1.0, p))}


def save_predictions(path, y_true, y_pred, y_proba=None, meta=None):
    """Persist per-instance predictions so significance tests are reproducible.

    Writes a compressed .npz; downstream stats (McNemar, paired bootstrap,
    Friedman) consume these without re-training. ``meta`` is stored as JSON.
    """
    import json as _json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)}
    if y_proba is not None:
        arrs["y_proba"] = np.asarray(y_proba)
    arrs["meta"] = np.frombuffer(_json.dumps(meta or {}).encode("utf-8"),
                                 dtype=np.uint8)
    np.savez_compressed(path, **arrs)


def load_predictions(path):
    """Load predictions saved by save_predictions. Returns dict."""
    import json as _json
    d = np.load(path, allow_pickle=False)
    out = {"y_true": d["y_true"], "y_pred": d["y_pred"]}
    if "y_proba" in d.files:
        out["y_proba"] = d["y_proba"]
    if "meta" in d.files:
        out["meta"] = _json.loads(bytes(d["meta"]).decode("utf-8"))
    return out


def aggregate_folds(fold_metrics: list[dict]):
    """Mean +/- SD across folds for every numeric key (manuscript reporting style)."""
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))
            or fold_metrics[0][k] is None]
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m[k] is not None]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
        else:
            agg[k] = {"mean": None, "sd": None}
    return agg
