# DarkTrace — Consolidated Real Results

## Table 6 - Classification (Phase 1)

| Model_dataset                                    |   Accuracy |   Macro_F1 |      AUC |    FPR |
|:-------------------------------------------------|-----------:|-----------:|---------:|-------:|
| DarkTrace RandomForest (CIC-Darknet2020)         |     0.8718 |     0.7766 |   0.9615 | 0.0182 |
| DarkTrace HistGradientBoosting (CIC-Darknet2020) |     0.8746 |     0.7942 |   0.99   | 0.0178 |
| DarkTrace TFIDF_LogReg (CoDA)                    |     0.8963 |     0.9034 | nan      | 0.0129 |
| DarkTrace TFIDF_LinearSVM (CoDA)                 |     0.9109 |     0.9192 | nan      | 0.0112 |

## Table 7 - Multilingual (Phase 2)

| Evaluation     | Language   |    N |   Macro_F1 |   Accuracy |
|:---------------|:-----------|-----:|-----------:|-----------:|
| Pooled         | de         |   24 |     0.6528 |     0.9167 |
| Pooled         | en         | 1757 |     0.9078 |     0.8993 |
| Pooled         | es         |   15 |     0.6787 |     0.8    |
| Pooled         | fr         |   25 |     0.7439 |     0.96   |
| Pooled         | pt         |   10 |     1      |     1      |
| Pooled         | ru         |  114 |     0.9159 |     0.8772 |
| Transfer(EN->) | de         |  129 |     0.4425 |     0.876  |
| Transfer(EN->) | en         | 1768 |     0.916  |     0.9078 |
| Transfer(EN->) | es         |   61 |     0.5727 |     0.8361 |
| Transfer(EN->) | fr         |   99 |     0.5157 |     0.8081 |
| Transfer(EN->) | pt         |   54 |     0.6828 |     0.9074 |
| Transfer(EN->) | ru         |  542 |     0.4357 |     0.7343 |

## Table 7b - Cross-domain Hindi/Arabic (Phase 2b)

| Language   |   N_test |   Macro_F1 |   Accuracy | Model       | Domain                      |
|:-----------|---------:|-----------:|-----------:|:------------|:----------------------------|
| hi         |     1146 |     0.958  |     0.9581 | transformer | cross-domain (social media) |
| ar         |     1778 |     0.6967 |     0.734  | transformer | cross-domain (social media) |

## Table 8 - Blockchain integrity (Phase 4)

| Backend          |   N_evidence |   Throughput_items_per_s |   Latency_p50_ms |   Latency_p95_ms | Chain_verified   | Tamper_detected   |
|:-----------------|-------------:|-------------------------:|-----------------:|-----------------:|:-----------------|:------------------|
| local_hash_chain |          500 |                   8747.6 |            0.094 |            0.233 | True             | True              |

## Table 9 - STIX/TAXII export (Phase 5)

| Export        |   Findings |   STIX_objects |   Indicators | Bundle_valid   |   Pushed_TAXII |   Build_s |
|:--------------|-----------:|---------------:|-------------:|:---------------|---------------:|----------:|
| STIX2.1/TAXII |        300 |            301 |          300 | True           |            nan |     0.302 |

## Table 10 - Explainable scoring (Phase 3)

| Model                          |    AUC |     AP |   NDCG@10 |   Brier |   Faithfulness_gap |   Stability_Jaccard |
|:-------------------------------|-------:|-------:|----------:|--------:|-------------------:|--------------------:|
| DarkTrace Explainable Ensemble | 0.9752 | 0.9757 |    0.9969 |  0.0537 |             0.0506 |               0.299 |
| Black-box MLP (baseline)       | 0.9707 | 0.9715 |    1      |  0.0754 |           nan      |             nan     |

## Table 11 - Integration ablation

| Configuration   |   has_label |   has_severity |   has_faithful_expl |   has_sealed_evidence |   verifiable_provenance |   standards_export |   Actionability |
|:----------------|------------:|---------------:|--------------------:|----------------------:|------------------------:|-------------------:|----------------:|
| full            |           1 |              1 |                0.73 |                     1 |                       1 |                  1 |           0.955 |
| no_scoring      |           1 |              0 |                0    |                     1 |                       1 |                  1 |           0.667 |
| no_explanation  |           1 |              1 |                0    |                     1 |                       1 |                  1 |           0.833 |
| no_sealing      |           1 |              1 |                0.73 |                     0 |                       0 |                  1 |           0.622 |
| no_export       |           1 |              1 |                0.73 |                     1 |                       0 |                  0 |           0.622 |


## Status

- Phases with real results: 7/7