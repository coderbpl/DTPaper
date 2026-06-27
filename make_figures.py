#!/usr/bin/env python3
"""Generate supplementary publication figures from real DarkTrace results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "darktrace_results_latest", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# ---------------------------------------------------------------
# Fig A: Multilingual transfer gap (pooled vs EN-> transfer), macro-F1
# ---------------------------------------------------------------
langs   = ["en", "ru", "fr", "de", "es", "pt"]
pooled  = [0.9075, 0.8074, 0.7439, 0.6528, 0.6787, 1.0000]
transfer= [0.9160, 0.4357, 0.5157, 0.4425, 0.5727, 0.6828]
x = np.arange(len(langs)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 3.6))
b1 = ax.bar(x - w/2, pooled,   w, label="Pooled (multilingual train)", color="#2c7fb8")
b2 = ax.bar(x + w/2, transfer, w, label="Zero-shot transfer (EN→)", color="#de7a22")
ax.set_xticks(x); ax.set_xticklabels([l.upper() for l in langs])
ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.05)
ax.set_title("Multilingual detection: pooled vs. zero-shot cross-lingual transfer")
ax.legend(loc="lower left", fontsize=9)
for b in (b1, b2):
    ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
fig.savefig(os.path.join(OUT, "fig_transfer_gap.png"))
plt.close(fig)

# ---------------------------------------------------------------
# Fig B: Integration ablation - actionability by configuration
# ---------------------------------------------------------------
cfgs = ["full", "no_explanation", "no_scoring", "no_sealing", "no_export"]
act  = [0.955, 0.833, 0.667, 0.622, 0.622]
colors = ["#238b45"] + ["#bdbdbd"] * (len(cfgs) - 1)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
b = ax.barh(cfgs[::-1], act[::-1], color=colors[::-1])
ax.set_xlabel("Actionability (composite capability score)")
ax.set_xlim(0, 1.0)
ax.set_title("Integration ablation: only the full pipeline maximizes actionability")
ax.bar_label(b, fmt="%.3f", fontsize=9, padding=3)
fig.savefig(os.path.join(OUT, "fig_ablation_actionability.png"))
plt.close(fig)

# ---------------------------------------------------------------
# Fig C: Explainable vs black-box scoring trade-off
# ---------------------------------------------------------------
metrics = ["AUC", "AP", "NDCG@10", "1−Brier"]
ens   = [0.9752, 0.9757, 0.9969, 1 - 0.0537]
black = [0.9707, 0.9715, 1.0000, 1 - 0.0754]
x = np.arange(len(metrics)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 3.6))
b1 = ax.bar(x - w/2, ens,   w, label="DarkTrace Explainable Ensemble", color="#3182bd")
b2 = ax.bar(x + w/2, black, w, label="Black-box MLP (baseline)", color="#969696")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0.90, 1.01); ax.set_ylabel("Score (higher is better)")
ax.set_title("Severity scoring: interpretability at near-parity performance")
ax.legend(loc="lower left", fontsize=9)
for b in (b1, b2):
    ax.bar_label(b, fmt="%.3f", fontsize=7, padding=1)
fig.savefig(os.path.join(OUT, "fig_scoring_tradeoff.png"))
plt.close(fig)

print("Wrote:", sorted(f for f in os.listdir(OUT) if f.startswith("fig_")))
