# Prediction Evaluation Scorecard Report

## Source Evidence

- **Source Attachment Set Fingerprint**: `fe6c96a18f7606869c30f8acdd21a525ab75ed3289bfc7af2c940fc3776c3772`
- **Source Snapshot Fingerprint**: `858f1740463ff8f0b5556f26489445eec74c33377ba75f2fc7e65895d7f4f1c1`
- **Source Rows Count**: 3 (Attached: 3, Rejected: 0)

## Aggregate Scorecard Metrics

- **Evaluated Rows**: 3
- **Excluded Rejected Rows**: 0
- **Correct Count**: 2
- **Incorrect Count**: 1
- **Accuracy**: 0.666667
- **Mean Selected-Side Probability**: 0.566667
- **Brier Score**: 0.2556

## Breakdowns

### Breakdown by Model ID

| Model ID | Count | Correct | Incorrect | Accuracy | Mean Probability | Brier Score |
| --- | --- | --- | --- | --- | --- | --- |
| `model_v1` | 3 | 2 | 1 | 0.666667 | 0.566667 | 0.2556 |

### Breakdown by Market ID

| Market ID | Count | Correct | Incorrect | Accuracy | Mean Probability | Brier Score |
| --- | --- | --- | --- | --- | --- | --- |
| `moneyline` | 3 | 2 | 1 | 0.666667 | 0.566667 | 0.2556 |

### Breakdown by Selection

| Selection | Count | Correct | Incorrect | Accuracy | Mean Probability | Brier Score |
| --- | --- | --- | --- | --- | --- | --- |
| `AWAY` | 1 | 1 | 0 | 1.0 | 0.52 | 0.2304 |
| `HOME` | 2 | 1 | 1 | 0.5 | 0.59 | 0.2682 |

## Evaluated Rows Detail

| Prediction Obs ID | Model ID | Selection | Model Prob | Actual Winner | Correct | Target | Brier Component |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `0cabf8e0dbc4a790...` | `model_v1` | HOME | 0.58 | HOME | True | 1 | 0.1764 |
| `573447cd9d62bff6...` | `model_v1` | AWAY | 0.52 | AWAY | True | 1 | 0.2304 |
| `9671f4340017a7b6...` | `model_v1` | HOME | 0.60 | AWAY | False | 0 | 0.3600 |

## Deterministic Evaluation Set Fingerprint

`5aa49a7bbf5526ec8fd3f67a8cda03a4519edc6a484c36be4fe83e41c12fd108`

## Safety & Methodological Disclaimers

- This report evaluates **synthetic local demo evidence only**.
- **Sample size is limited** and insufficient for real model performance claims.
- **No real model superiority or performance claim** is made.
- **No model retraining or promotion** occurred.
- **No provider or network call** was executed.
- **No database write** occurred.
- **No odds, payout, EV, ROI, Kelly, or betting recommendation** was used.
- **No production deployment** was performed.
