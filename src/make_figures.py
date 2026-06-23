"""
darktrace_phase1/src/make_figures.py

Generate the Phase 1 figures referenced in manuscript Section 8.18:
- confusion matrices (per model/dataset)
- per-class F1 bar charts (minority-class behaviour)

Reads the *_results.json written by the experiments. Uses matplotlib only.

Run:
    python -m src.make_figures
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLES = "results/tables"
FIGS = "results/figures"


def per_class_bar(results, dataset_tag):
    os.makedirs(FIGS, exist_ok=True)
    for name, r in results["models"].items():
        pcf = r.get("test_per_class_f1", {})
        if not pcf:
            continue
        classes = list(pcf.keys())
        f1s = [pcf[c]["f1"] for c in classes]
        supports = [pcf[c]["support"] for c in classes]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(range(len(classes)), f1s, color="#3b6ea5")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Per-class F1"); ax.set_ylim(0, 1)
        syn = " (SYNTHETIC)" if results.get("synthetic") else ""
        ax.set_title(f"Per-class F1 — {name} / {dataset_tag}{syn}", fontsize=10)
        for b, s in zip(bars, supports):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"n={s}", ha="center", va="bottom", fontsize=6)
        fig.tight_layout()
        out = os.path.join(FIGS, f"perclass_f1_{dataset_tag}_{name}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"wrote {out}")


def main():
    for fn, tag in (("traffic_results.json", "CIC-Darknet2020"),
                    ("text_results.json", "CoDA")):
        p = os.path.join(TABLES, fn)
        if os.path.exists(p):
            with open(p) as f:
                results = json.load(f)
            per_class_bar(results, tag)
        else:
            print(f"skip {fn} (run the experiment first)")


if __name__ == "__main__":
    main()
