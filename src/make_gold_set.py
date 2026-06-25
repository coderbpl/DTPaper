"""
darktrace_phase1/src/make_gold_set.py

Sample a representative subset of the real CoDA corpus for ANALYST risk labelling.

Why: the default risk label is derived from categories, which a reviewer can
fairly call "a relabelling of the classification task". A small human-labelled
gold set lets you (a) validate the derived label against real judgements and
(b) report scoring metrics on genuinely analyst-rated risk — a much stronger claim.

This script selects items STRATIFIED by category (and language when available) so
the gold set is representative, then writes a CSV with empty 'risk' cells for you
to fill in. You then point configs/scoring.json at it with risk_mode='analyst'.

Labelling guidance (put your own scale in the manuscript):
    0 = none/benign, 1 = low, 2 = moderate, 3 = high/critical threat.
Rate the THREAT the post represents, independent of its category name, so the
gold set genuinely tests whether the model learns risk rather than topic.

Run:
    python -m src.make_gold_set --config configs/scoring.json --n 200
    # ... fill in the 'risk' column in results/tables/gold_set_to_label.csv ...
    # then set data.risk_mode='analyst' and
    #          data.analyst_labels_csv='results/tables/gold_set_labelled.csv'
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from .utils import load_config, ensure_dirs, get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scoring.json")
    ap.add_argument("--n", type=int, default=200, help="approx. number of items to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/tables/gold_set_to_label.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    logger = get_logger("gold_set", cfg["paths"]["logs"])

    # reuse the Phase-1 text loader so categories/languages are parsed identically
    from .exp_text import load_dataset as load_text
    df, synthetic, _ = load_text(cfg, logger, smoke_test=False)
    if synthetic:
        raise SystemExit("Refusing to build a gold set from synthetic data.")

    # attach language if present (parsed into __lang__ by the loader)
    has_lang = "__lang__" in df.columns
    strata_cols = ["label"] + (["__lang__"] if has_lang else [])
    logger.info(f"Stratifying gold set by {strata_cols} "
                f"({df['label'].nunique()} categories"
                + (f", {df['__lang__'].nunique()} languages)" if has_lang else ")"))

    # proportional allocation across strata, with at least 1 per category
    rng = np.random.RandomState(args.seed)
    n_target = min(args.n, len(df))
    # primary stratification on category (guarantee coverage), secondary on lang
    picks = []
    cats = df["label"].value_counts()
    per_cat = np.maximum(1, np.round(n_target * cats / cats.sum()).astype(int))
    for cat, k in per_cat.items():
        sub = df[df["label"] == cat]
        if has_lang and sub["__lang__"].nunique() > 1:
            # spread within category across languages where possible
            take = min(k, len(sub))
            idx = (sub.groupby("__lang__", group_keys=False)
                      .apply(lambda g: g.sample(min(len(g), max(1, take // sub["__lang__"].nunique())),
                                                random_state=args.seed)))
            extra = take - len(idx)
            if extra > 0:
                remaining = sub.drop(idx.index)
                if len(remaining):
                    idx = pd.concat([idx, remaining.sample(min(extra, len(remaining)),
                                                            random_state=args.seed)])
            picks.append(idx)
        else:
            picks.append(sub.sample(min(k, len(sub)), random_state=args.seed))
    gold = pd.concat(picks).drop_duplicates(subset=["__key__"] if "__key__" in df.columns else None)
    gold = gold.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # shuffle

    # build the labelling sheet
    cols = {}
    if "__key__" in gold.columns:
        cols["key"] = gold["__key__"].astype(str)
    else:
        cols["key"] = [f"item_{i}" for i in range(len(gold))]
    cols["derived_category"] = gold["label"].astype(str)
    if has_lang:
        cols["language"] = gold["__lang__"].astype(str)
    # show a snippet to make labelling possible without external lookup
    cols["text_snippet"] = gold["text"].astype(str).str.slice(0, 280)
    cols["risk"] = ""  # <-- analyst fills this (0..3)
    sheet = pd.DataFrame(cols)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sheet.to_csv(args.out, index=False)
    logger.info(f"Wrote {len(sheet)} items to label -> {args.out}")
    logger.info("Fill in the 'risk' column (0=none,1=low,2=moderate,3=high), save as "
                "gold_set_labelled.csv, then set risk_mode='analyst' and "
                "analyst_labels_csv to that path.")
    # category coverage report
    cov = sheet["derived_category"].value_counts().to_dict()
    logger.info(f"Category coverage in gold set: {cov}")


if __name__ == "__main__":
    main()
