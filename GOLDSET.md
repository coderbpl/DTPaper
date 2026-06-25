# Phase 3 Gold-Set Validation (analyst-labelled risk)

The default Phase 3 risk label is *derived from CoDA categories*, which a reviewer
will (correctly) note is not a true risk rating. This workflow lets you validate
the explainable scorer against a small set of **real human risk labels**, which
directly answers that objection. Nothing here invents labels — you provide them.

## Step 1 — generate the labelling worksheet

```bash
python -m src.make_goldset --config configs/scoring.json --n 200
```

Writes `data/processed/goldset_to_label.csv` with columns
`key, category, language, text, risk` — stratified ~evenly across the 10 CoDA
categories, with the `risk` column left blank. The `key` column is what later
aligns your labels back to the corpus, so do not edit it.

## Step 2 — label the items

Open the CSV and fill the `risk` column using a written rubric **you define and
report in the manuscript**. A defensible default is ordinal severity:

| risk | meaning |
|------|---------|
| 0 | benign / no operational threat |
| 1 | low (nuisance, minor policy violation) |
| 2 | moderate (illicit but not imminent harm) |
| 3 | high / imminent harm (weapons, active exploitation, violence) |

Practical tips that strengthen the result for review:
- Have **two annotators** label independently and report inter-annotator
  agreement (Cohen's kappa). Even a second labeller on 50 of the 200 items lets
  you report a kappa, which reviewers expect for human-labelled data.
- Keep the rubric short and concrete; include 2-3 example items per level in the
  manuscript appendix.
- Don't peek at the model's score while labelling (avoids circularity).

Save the filled file as `data/processed/goldset_labeled.csv`.

## Step 3 — evaluate the scorer against your gold labels

In `configs/scoring.json` set:

```json
"risk_mode": "analyst",
"analyst_labels_csv": "data/processed/goldset_labeled.csv"
```

then:

```bash
python -m src.exp_scoring --config configs/scoring.json
```

The run will:
- align your labels by `key`, dropping any unmatched items (logged),
- log `Risk label: ANALYST-PROVIDED ... This is reportable as real labels`,
- collapse multi-level severity to binary at `data.binary_threshold` for the
  binary scorer (the ordinal labels are retained in the output JSON),
- produce the same ranking / faithfulness / stability / multi-method-agreement
  metrics as the category-derived run, but now against **real labels**.

## What to expect and how to report it

The AUC on real labels will almost certainly be **lower** than on the
category-derived labels (e.g. ~0.82 vs ~0.975). That is the point: the
category-derived number is inflated because it partly re-learns the category
boundary. The gold-set number is the honest one. Report **both**, and frame the
contribution as *explainable and faithful* scoring validated against human
ratings — not as "we achieve 0.97 AUC."

Because the gold set is small (~200), report confidence intervals (the pipeline
bootstraps them) and treat it as a *validation* of the explanation machinery, not
as the primary performance benchmark. State the gold-set size, rubric, number of
annotators, and inter-annotator agreement explicitly in Section 8.9.
