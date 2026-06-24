"""
darktrace_phase1/src/make_scoring_figures.py

Figures for Phase 3 (manuscript Section 8.18 / Section 10):
  - feature-attribution bar chart (top-k mean |attribution|)
  - faithfulness comparison (top-k vs random-k probability drop)

Reads results/tables/scoring_results.json. matplotlib only.

Run:
    python -m src.make_scoring_figures
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLES = "results/tables"
FIGS = "results/figures"


def main():
    p = os.path.join(TABLES, "scoring_results.json")
    if not os.path.exists(p):
        print("scoring_results.json not found. Run exp_scoring on real data first.")
        return
    res = json.load(open(p))
    os.makedirs(FIGS, exist_ok=True)
    ens = res["models"].get("explainable_ensemble", {})
    if not ens:
        print("no ensemble results to plot.")
        return

    # faithfulness bar
    f = ens["faithfulness"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["top-k", "random-k"],
           [f["comprehensiveness_topk"], f["comprehensiveness_random"]],
           color=["#b5402f", "#888888"])
    ax.set_ylabel("Mean |risk-score drop|")
    ax.set_title(f"Explanation faithfulness (k={f['k']})\n"
                 f"gap = {f['faithfulness_gap']:+.4f}", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "scoring_faithfulness.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)

    # ranking comparison vs black-box
    bb = res["models"].get("blackbox_mlp", {})
    if bb:
        fig, ax = plt.subplots(figsize=(6, 4))
        metrics = ["auc", "ap", "ndcg@10"]
        e = [ens["ranking"].get(m) or 0 for m in metrics]
        b = [bb["ranking"].get(m) or 0 for m in metrics]
        x = np.arange(len(metrics)); w = 0.38
        ax.bar(x - w/2, e, w, label="Explainable ensemble", color="#3b6ea5")
        ax.bar(x + w/2, b, w, label="Black-box MLP", color="#cc8844")
        ax.set_xticks(x); ax.set_xticklabels(["AUC", "AP", "NDCG@10"])
        ax.set_ylim(0, 1); ax.set_ylabel("Score")
        ax.set_title("Ranking quality: explainable vs black-box", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(FIGS, "scoring_ranking_compare.png")
        fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
