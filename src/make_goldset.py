"""
darktrace_phase1/src/make_goldset.py

Generate a stratified sample of CoDA items for ANALYST hand-labelling, so the
explainable scorer (Phase 3) can be validated against real human risk ratings
instead of category-derived labels. Nothing here invents labels — it only
produces the worksheet you fill in.

Workflow:
    1. python -m src.make_goldset --config configs/scoring.json --n 200
       -> writes data/processed/goldset_to_label.csv with columns:
          key, category, language, text, risk   (risk left BLANK for you)
    2. You fill the 'risk' column with integer severity (e.g. 0-3) or binary 0/1,
       using a written rubric you define and report in the manuscript.
    3. Point configs/scoring.json -> data.risk_mode="analyst",
       data.analyst_labels_csv="data/processed/goldset_labeled.csv"
       (rename your filled file) and re-run exp_scoring. It will align by 'key'
       and evaluate on YOUR labels, marking the results reportable-as-real.

Stratification is by category so every CoDA class is represented, with optional
language spread so non-English items are not missed.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from .utils import load_config, get_logger
from .exp_text import load_dataset as load_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scoring.json")
    ap.add_argument("--n", type=int, default=200, help="number of items to label")
    ap.add_argument("--out", default="data/processed/goldset_to_label.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logger = get_logger("goldset", cfg["paths"]["logs"])
    df, _ = load_text(cfg, logger)

    # attach language if present from the loader
    if "__lang__" in df.columns:
        df = df.rename(columns={"__lang__": "language"})
    else:
        df["language"] = "unknown"
    df = df.rename(columns={"label": "category"})
    # carry a stable key for later alignment
    if "__key__" in df.columns:
        df["key"] = df["__key__"].astype(str)
    else:
        df["key"] = [f"item_{i}" for i in range(len(df))]

    rng = np.random.RandomState(args.seed)
    cats = df["category"].unique()
    per = max(1, args.n // len(cats))
    picks = []
    for c in cats:
        sub = df[df["category"] == c]
        take = min(per, len(sub))
        picks.append(sub.sample(take, random_state=rng.randint(1 << 30)))
    out = pd.concat(picks)
    # top up to n if rounding left us short
    if len(out) < args.n:
        rest = df[~df["key"].isin(out["key"])]
        if len(rest):
            out = pd.concat([out, rest.sample(
                min(args.n - len(out), len(rest)),
                random_state=args.seed)])
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out["risk"] = ""   # YOU fill this in
    cols = ["key", "category", "language", "text", "risk"]
    out[cols].to_csv(args.out, index=False)
    logger.info(f"Wrote {len(out)} items to {args.out} "
                f"({len(cats)} categories, ~{per}/category).")
    logger.info("Fill the 'risk' column with your analyst rating, then set "
                "data.risk_mode='analyst' and data.analyst_labels_csv to the "
                "filled file in configs/scoring.json.")
    # quick rubric reminder
    logger.info("Suggested rubric (define & report your own): "
                "0=benign/none, 1=low, 2=moderate, 3=high/imminent harm.")


if __name__ == "__main__":
    main()
