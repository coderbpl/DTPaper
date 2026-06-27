"""
darktrace_phase1/src/exp_stats.py

Phase 7 / Statistical validation (manuscript Section 8.13).

Consolidates and computes the significance tests and confidence intervals for
the DarkTrace results. It operates in two modes, automatically:

  * CONSOLIDATE (always): reads the existing per-experiment result JSONs and
    (i) surfaces the significance tests already computed by the experiment
    scripts (text McNemar, bootstrap CIs), and (ii) computes NEW exact Wilson
    score CIs for every per-language accuracy directly from (n, accuracy) — no
    per-instance data required. Flags small-N cells whose CIs are unreliable.

  * FULL (when prediction files exist under darktrace_results/preds/*.npz):
    additionally computes paired-bootstrap significance for the explainable-vs-
    blackbox scorer, per-language macro-F1 bootstrap CIs, and a Friedman+Nemenyi
    omnibus test across models, with Holm-Bonferroni correction.

Outputs:
    darktrace_results/tables/stats_results.json
    darktrace_results/tables/table_stats.csv

Run:
    python -m src.exp_stats
"""
from __future__ import annotations
import argparse, json, os, glob
import numpy as np
import pandas as pd

from .metrics_stats import (
    wilson_ci_from_acc, friedman_nemenyi, paired_bootstrap_diff,
    bootstrap_ci, mcnemar_test, holm_bonferroni, load_predictions,
)

SMALL_N = 30  # below this, per-language CIs are flagged as unreliable

# Resolved at runtime from --results-dir (or auto-detected). Defaults point at a
# fresh experiment run; the existing snapshot lives under darktrace_results/.
TABLES = os.path.join("results", "tables")
PREDS = os.path.join("results", "preds")


def _autodetect_results_dir():
    """Prefer a live `results/` run; fall back to the `darktrace_results/` snapshot."""
    for base in ("results", "darktrace_results"):
        if os.path.exists(os.path.join(base, "tables", "text_results.json")):
            return base
    return "results"


def _set_dirs(results_dir):
    global TABLES, PREDS
    TABLES = os.path.join(results_dir, "tables")
    PREDS = os.path.join(results_dir, "preds")


def _load(name):
    p = os.path.join(TABLES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def consolidate_text(out):
    """Surface text McNemar + bootstrap CIs already computed by exp_text."""
    d = _load("text_results.json")
    if not d:
        return
    rows = []
    for name, r in d.get("models", {}).items():
        ci = r.get("test_macro_f1_ci95", [None, None])
        cv = r.get("cv", {}).get("macro_f1", {})
        rows.append({
            "task": "classification(text/CoDA)", "model": name,
            "macro_f1": round(r["test"]["macro_f1"], 4),
            "macro_f1_ci95_lo": round(ci[0], 4) if ci[0] is not None else None,
            "macro_f1_ci95_hi": round(ci[1], 4) if ci[1] is not None else None,
            "cv_mean": round(cv.get("mean"), 4) if cv.get("mean") is not None else None,
            "cv_sd": round(cv.get("sd"), 4) if cv.get("sd") is not None else None,
        })
    out["classification_text"] = {
        "rows": rows,
        "mcnemar": d.get("mcnemar"),
        "note": "McNemar and macro-F1 bootstrap CIs computed by exp_text (real).",
    }


def consolidate_scoring(out):
    """Surface the explainable-vs-blackbox McNemar + macro-F1 CI from exp_scoring."""
    d = _load("scoring_results.json")
    if not d:
        return
    ens = d.get("models", {}).get("explainable_ensemble", {})
    out["scoring"] = {
        "explainable_macro_f1_at_0.5_ci95": ens.get("macro_f1_at_0.5_ci95"),
        "explainable_vs_blackbox": d.get("explainable_vs_blackbox"),
        "note": ("McNemar (explainable ensemble vs black-box MLP) and the "
                 "ensemble macro-F1 bootstrap CI are computed by exp_scoring "
                 "(real). Paired-bootstrap macro-F1 difference is added by the "
                 "full battery once predictions are saved."),
    }


def wilson_for_multilingual(out):
    """Exact Wilson 95% CIs for every per-language accuracy (pooled + transfer)."""
    d = _load("multilingual_results.json")
    if not d:
        return
    blocks = {}
    for eval_name, ev in d.get("evaluations", {}).items():
        per = ev.get("per_language", {})
        rows = []
        for lang, m in sorted(per.items()):
            n = int(m["n"]); acc = float(m["accuracy"])
            point, lo, hi = wilson_ci_from_acc(acc, n)
            rows.append({
                "language": lang, "n": n,
                "macro_f1": round(float(m["macro_f1"]), 4),
                "accuracy": round(acc, 4),
                "acc_ci95_lo": round(lo, 4), "acc_ci95_hi": round(hi, 4),
                "ci_width": round(hi - lo, 4),
                "reliable": n >= SMALL_N,
            })
        ov = ev.get("overall", {})
        blocks[eval_name] = {
            "per_language": rows,
            "overall_macro_f1": ov.get("macro_f1"),
            "overall_macro_f1_ci95": ov.get("macro_f1_ci95"),
        }
    out["multilingual"] = {
        "blocks": blocks,
        "note": ("Per-language ACCURACY CIs are exact Wilson intervals from "
                 f"(n, accuracy). Cells with n<{SMALL_N} are flagged "
                 "unreliable; macro-F1 CIs there need a re-run with saved "
                 "predictions."),
    }


def full_from_predictions(out):
    """If prediction .npz files exist, run the full significance battery."""
    files = sorted(glob.glob(os.path.join(PREDS, "*.npz")))
    if not files:
        out["full_battery"] = {
            "status": "skipped",
            "reason": (f"no prediction files in {PREDS}. Re-run the patched "
                       "experiments (exp_text/exp_multilingual/exp_scoring) to "
                       "persist predictions, then re-run exp_stats for scoring "
                       "significance, per-language macro-F1 CIs, and "
                       "Friedman+Nemenyi."),
        }
        return

    preds = {os.path.splitext(os.path.basename(f))[0]: load_predictions(f) for f in files}
    res = {"loaded": list(preds.keys())}
    pvals = {}

    # (1) Explainable vs black-box scorer (paired bootstrap on shared test set)
    if "scoring_explainable" in preds and "scoring_blackbox" in preds:
        a, b = preds["scoring_explainable"], preds["scoring_blackbox"]
        d = paired_bootstrap_diff(a["y_true"], a["y_pred"], b["y_pred"],
                                  metric="macro_f1", n_boot=2000, seed=42)
        res["scoring_explainable_vs_blackbox"] = d
        pvals["scoring_expl_vs_bb"] = d["p_value"]

    # (2) Per-language macro-F1 bootstrap CIs (need per-instance preds per lang)
    lang_ci = {}
    for key, pr in preds.items():
        if key.startswith("multilingual_"):
            lang = key.split("multilingual_")[-1]
            lo, hi = bootstrap_ci(pr["y_true"], pr["y_pred"], "macro_f1",
                                  n_boot=2000, seed=42)
            lang_ci[lang] = [round(lo, 4), round(hi, 4)]
    if lang_ci:
        res["multilingual_macro_f1_ci95"] = lang_ci

    # (3) Friedman + Nemenyi across >=3 classifiers sharing a fold structure
    fold_files = sorted(glob.glob(os.path.join(PREDS, "folds_*.json")))
    if fold_files:
        mat, names = [], None
        scores_by_model = {}
        for ff in fold_files:
            fd = json.load(open(ff))
            scores_by_model[fd["model"]] = fd["fold_macro_f1"]
        names = list(scores_by_model.keys())
        n_folds = min(len(v) for v in scores_by_model.values())
        mat = np.array([[scores_by_model[m][i] for m in names]
                        for i in range(n_folds)])
        if mat.shape[1] >= 3:
            res["friedman_nemenyi"] = friedman_nemenyi(mat, names)

    # Holm-Bonferroni across collected p-values
    if pvals:
        res["holm_bonferroni"] = {
            k: {"p": v, "reject": rej}
            for k, (v, rej) in holm_bonferroni(pvals).items()
        }
    out["full_battery"] = {"status": "computed", **res}


def write_csv(out):
    """Flatten the most table-worthy numbers to a single CSV."""
    rows = []
    for r in out.get("classification_text", {}).get("rows", []):
        rows.append({"section": "classification(text)", **r})
    mc = out.get("classification_text", {}).get("mcnemar")
    if mc:
        rows.append({"section": "classification(text)",
                     "test": f"McNemar {mc['models'][0]} vs {mc['models'][1]}",
                     "p_value": mc["p_value"], "statistic": round(mc["statistic"], 3)})
    for blk, data in out.get("multilingual", {}).get("blocks", {}).items():
        for r in data["per_language"]:
            rows.append({"section": f"multilingual/{blk}", **r})
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "table_stats.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None,
                    help="base dir containing tables/ and preds/ "
                         "(default: auto-detect results/ then darktrace_results/)")
    args = ap.parse_args()
    _set_dirs(args.results_dir or _autodetect_results_dir())
    print(f"[exp_stats] using results dir: {os.path.dirname(TABLES)}")

    out = {"experiment": "phase7_statistical_validation"}
    consolidate_text(out)
    consolidate_scoring(out)
    wilson_for_multilingual(out)
    full_from_predictions(out)
    os.makedirs(TABLES, exist_ok=True)
    json.dump(out, open(os.path.join(TABLES, "stats_results.json"), "w"), indent=2)
    write_csv(out)

    # console summary
    print("=== DarkTrace statistical validation ===")
    ct = out.get("classification_text", {})
    if ct.get("mcnemar"):
        mc = ct["mcnemar"]
        print(f"[text] McNemar {mc['models'][0]} vs {mc['models'][1]}: "
              f"p={mc['p_value']:.3e}, chi2={mc['statistic']:.2f}")
    for r in ct.get("rows", []):
        print(f"[text] {r['model']}: macro-F1={r['macro_f1']} "
              f"CI95=[{r['macro_f1_ci95_lo']},{r['macro_f1_ci95_hi']}] "
              f"cv={r['cv_mean']}±{r['cv_sd']}")
    for blk, data in out.get("multilingual", {}).get("blocks", {}).items():
        print(f"[multilingual/{blk}] per-language accuracy Wilson CIs:")
        for r in data["per_language"]:
            flag = "" if r["reliable"] else "  (small-N: unreliable)"
            print(f"   {r['language']:>3}  n={r['n']:>4}  acc={r['accuracy']:.3f} "
                  f"CI95=[{r['acc_ci95_lo']:.3f},{r['acc_ci95_hi']:.3f}]{flag}")
    print(f"[full battery] {out.get('full_battery', {}).get('status')}")
    print(f"Wrote {os.path.join(TABLES,'stats_results.json')} and table_stats.csv")


if __name__ == "__main__":
    main()
