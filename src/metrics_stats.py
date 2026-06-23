"""
darktrace_phase1/src/metrics_stats.py

Shared evaluation metrics and statistical-testing utilities for DarkTrace Phase 1.
These functions populate the values reported in manuscript Tables 6 (classification)
and support the significance testing described in manuscript Section 8.13.

All functions are deterministic given a seed and have no GPU dependency.
"""
from __future__ import annotations
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
