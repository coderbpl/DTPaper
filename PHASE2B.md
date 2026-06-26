# Phase 2b — Cross-Domain Multilingual Threat Classification (Hindi + Arabic)

## Why this is a separate experiment (read before writing the paper)

The brief requires Hindi and Arabic coverage. **Native dark-web corpora for Hindi
and Arabic do not exist at evaluation scale.** Roughly 90% of dark-web text is
English (Jin et al. 2022); the DarkBERT authors report the same and explicitly
decline to build a multilingual dark-web model for this reason. In our CoDA
corpus, Hindi has n=1 and Arabic n=7 — unusable for evaluation.

So this phase does the honest thing: it evaluates multilingual threat
classification on the closest **real, peer-reviewed** annotated corpora that do
exist, and reports it as a **cross-domain** result plus a documented limitation —
never as "dark-web Hindi/Arabic."

## Datasets (real, citable)

**Hindi — Hostility Detection Dataset** (Bhardwaj et al., 2020, CONSTRAINT-2021).
~8,200 manually annotated posts; dimensions: fake, hate, offensive, defamation,
non-hostile. Repo: https://github.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021
Save the train CSV as `data/raw/hindi_hostility.csv`.

**Arabic — OSACT Offensive Language / Hate Speech** (Mubarak et al.; OSACT2020/2022).
Tweets labelled offensive/not and fine-grained hate types; Cohen's kappa ~0.82.
Source: https://alt.qcri.org/resources/OSACT2022/  (or the consolidated
Hugging Face set `manueltonneau/arabic-hate-speech-superset`).
Save as `data/raw/arabic_osact.csv` (TSV is auto-detected).

## What it does

Maps each corpus to a **binary threat/benign** label (the exact source->binary
mapping is stated and saved in the results JSON), then trains and evaluates the
**same classifier as Phase 2** (distil multilingual transformer on GPU, TF-IDF
fallback on CPU) within each language. Unmapped labels are dropped and counted —
never guessed.

## Run

```bash
# place the two corpora first (see above), then:
python -m src.exp_multilingual_crossdomain --config configs/multilingual_crossdomain.json
```

Outputs: `results/tables/multilingual_crossdomain_results.json` and
`table7b_crossdomain.csv` with per-language macro-F1, CI, accuracy, and which
model path ran.

## How to report it (honest framing — non-negotiable)

State plainly: *"Because native dark-web corpora for Hindi and Arabic do not exist
at evaluation scale, we assess multilingual threat classification on the closest
available real annotated corpora — Hindi hostility detection (Bhardwaj et al.,
2020) and Arabic offensive-language detection (OSACT) — as a cross-domain
robustness check. These are social-media corpora, not dark-web data; results
indicate whether the approach generalizes to these languages, and the absence of
native dark-web Hindi/Arabic data is itself a finding and a limitation."*

Do NOT merge these numbers into the dark-web CoDA multilingual table as if they
were the same task. Keep Table 7 (dark-web, 6 languages) and Table 7b
(cross-domain, Hindi/Arabic) clearly separate.
