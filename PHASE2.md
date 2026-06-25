# Phase 2 — Multilingual Dark-Web Text Classification

Implements the multilingual classification experiment (manuscript Section 8.7,
RQ2). **Scope: classification only.** CoDA has category labels but no entity-span
annotations, so NER is out of scope here and should be done separately on a
span-annotated corpus (DNRTI [8] / APTNER [30]); training "NER on CoDA" would
produce no valid metric.

## Two evaluations

**(A) Pooled multilingual model** — one model trained on all in-scope languages,
reported overall *and per language*, so a weak language is visible rather than
hidden inside the English-dominated average.

**(B) Cross-lingual transfer** — train on English only, test on each other
language. This is the real multilingual claim: does the model generalise across
languages, or only look good because ~89% of CoDA is English?

## Language scope is data-driven

Only languages with `>= data.min_lang_count` samples (default 50) are included.
With real CoDA that yields roughly **English, Russian, German, French, Spanish,
Portuguese**. **Hindi (n=1) is automatically excluded** and the exclusion is logged
and recorded in the results JSON — do not claim Hindi support from CoDA.

## Model

- **Primary (GPU):** a lightweight multilingual transformer,
  `distilbert-base-multilingual-cased`, fine-tuned with the HuggingFace Trainer.
  Small and fast; set `model.name` to `xlm-roberta-base` for the heavier option
  the manuscript mentions.
- **Fallback (CPU):** TF-IDF + LinearSVM, so the pipeline runs anywhere for logic
  testing. The output records `model_path`; **only the transformer path is the
  reportable Phase-2 result.** If you see `model_path: tfidf_fallback`, transformers
  weren't available — install them and re-run before reporting.

## Run (Kaggle GPU)

```bash
pip install -q transformers torch
python -m src.exp_multilingual --config configs/multilingual.json
```

Turn the Kaggle accelerator ON (GPU) for this phase — unlike Phases 1 and 3, the
transformer fine-tuning genuinely uses it. Confirm the log shows
`Transformer path: fine-tuning ... (GPU=True)` and `model_path: transformer`.

## Outputs

- `results/tables/multilingual_results.json` — overall + per-language + transfer
  metrics, the in-scope/excluded language lists, and which model path ran.
- `results/tables/table7_multilingual.csv` — manuscript Table 7 fragment.

## How to report it (honest framing)

- Lead with the **per-language** table, not just the pooled average — reviewers
  want to see that non-English languages actually work, and the pooled number is
  dominated by English volume.
- Report the **cross-lingual transfer** result as the core multilingual evidence.
  If EN->RU transfer holds up, that is a genuine cross-lingual generalisation
  claim. If a language collapses under transfer, say so.
- State explicitly that Hindi was excluded for lack of data (n=1), and re-scope
  the manuscript's multilingual claim to the languages actually evaluated. This
  is the honest path flagged in the earlier review.
- The smoke/fallback numbers are not reportable; only transformer-path results on
  real CoDA go in the paper.
