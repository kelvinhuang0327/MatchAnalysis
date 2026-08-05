# Prediction Feedback Ledger Report

## Source Evidence

- **P15C Snapshot Fingerprint**: `858f1740463ff8f0b5556f26489445eec74c33377ba75f2fc7e65895d7f4f1c1`
- **P16A Attachment Set Fingerprint**: `fe6c96a18f7606869c30f8acdd21a525ab75ed3289bfc7af2c940fc3776c3772`
- **P16B Evaluation Set Fingerprint**: `5aa49a7bbf5526ec8fd3f67a8cda03a4519edc6a484c36be4fe83e41c12fd108`

## Feedback Summary

- **Total Feedback Rows**: 3
- **Evaluated Rows**: 3
- **Attachment-Rejected Rows**: 0
- **Correct Count**: 2
- **Incorrect Count**: 1

## Feedback Rows Detail

| Prediction Obs ID | Provider NS | Game ID | Game# | Model | Market | Selection | Prob | Score | Winner | Rejection Reason | Correct | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0cabf8e0dbc4a790...` | MLB_STATS_API | 888001 | 1 | model_v1 | moneyline | HOME | 0.58 | 5-3 | HOME | — | True | 0.1764 |
| `573447cd9d62bff6...` | MLB_STATS_API | 888002 | 1 | model_v1 | moneyline | AWAY | 0.52 | 2-4 | AWAY | — | True | 0.2304 |
| `9671f4340017a7b6...` | MLB_STATS_API | 888001 | 2 | model_v1 | moneyline | HOME | 0.60 | 1-3 | AWAY | — | False | 0.3600 |

## Deterministic Feedback Ledger Fingerprint

`9372bd876161fd9859eae1468246ae28ad3f21897bde90ad6c9d782c7f7788b6`

## Limitations and Disclaimers

- This report joins **synthetic local result evidence only**.
- **Sample size is insufficient** for real model performance claims.
- This is an **audit/feedback ledger**, not a training dataset.
- **No model retraining or promotion** occurred.
- **No provider or network call** was executed.
- **No database write** occurred.
- **No odds, payout, EV, ROI, Kelly, or betting evaluation** was used.
- **No production deployment** was performed.
