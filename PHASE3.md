# Phase 3 — Explainable Ensemble Threat Scoring

Implements the explainable threat-scoring module (manuscript Section 10) and its
evaluation (Section 8.9). CPU-only. Runs on the **same real CoDA corpus** Phase 1
uses, so no new data download is needed.

## What it does

1. Builds a **heterogeneous stacking ensemble** (HistGBT + RandomForest base
   learners, Logistic Regression meta-learner), calibrated with isotonic
   regression so the risk score is interpretable as a probability.
2. Derives a **binary high-risk label** from the CoDA category (configurable via
   `data.high_risk`). This is stated explicitly because the public datasets do not
   ship a continuous risk label — the scoring task is *operationalised*, not
   invented, and the output records exactly which categories are treated as
   high-risk so nothing is overclaimed.
3. **Explains** each score with SHAP (KernelExplainer) and LIME when installed,
   falling back to permutation importance otherwise (the output records which
   method was used).
4. Evaluates the two manuscript claims:
   - **RQ3-a (no accuracy cost):** compares the explainable ensemble against a
     **black-box MLP** on AUC / AP / NDCG@10 / Brier, with a McNemar test.
   - **RQ3-b (faithful + stable explanations):** **comprehensiveness** (top-k vs
     random-k feature-ablation drop) and **stability** (top-k attribution Jaccard),
     plus a **SHAP–LIME agreement** convergent-validity check.

## Run

```bash
# verify the pipeline (synthetic, NON-reportable)
python -m src.exp_scoring --config configs/scoring.json --smoke-test

# real data (requires data/raw/coda.csv from Phase 1)
python -m src.exp_scoring --config configs/scoring.json
python -m src.make_scoring_figures
```

On Kaggle, install the XAI deps first so SHAP/LIME are used (not the fallback):
```bash
pip install -q shap lime
```

## Outputs

- `results/tables/scoring_results.json` — full metrics, attribution method,
  faithfulness, stability, SHAP–LIME agreement, ensemble-vs-blackbox comparison.
- `results/tables/table10_scoring.csv` — manuscript Table 10 fragment.
- `results/figures/scoring_faithfulness.png`, `scoring_ranking_compare.png`.

## Configuration (`configs/scoring.json`)

- `data.high_risk` — categories treated as high-risk for the binary score.
- `model.max_features` — TF-IDF vocabulary cap (kept modest so SHAP/LIME are
  tractable on CPU).
- `topk`, `explain_max_eval`, `lime_n` — control explanation cost.

## How this defends the novelty

The manuscript's reviewer risk was "novelty depends on results," and explainable
dark-web threat scoring is one of the two standalone sub-novelties. This phase
produces the evidence: if the ensemble matches or beats the black-box on ranking
**and** the faithfulness gap is positive (top-k ablation hurts more than random),
the "explainable by construction without accuracy cost" claim is supported by
measurement rather than assertion. Report the actual numbers; if the gap is not
positive, say so and treat explanations as qualitative aids only.

## Honest limitations

- The binary risk label is derived from categories, not an analyst-annotated risk
  ranking; the manuscript should state this and, ideally, add a small
  analyst-labelled validation set later.
- KernelExplainer is approximate and sampled for tractability; for tree models a
  TreeExplainer would be exact but the calibrated stacking wrapper is not a single
  tree, so KernelExplainer is the honest general choice here.
