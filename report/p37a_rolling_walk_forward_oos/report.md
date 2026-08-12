# P37A Rolling Walk-Forward Moneyline OOS Evaluation

This is a deterministic offline champion/challenger evaluation. No model was promoted.

## Chronological windows

| Window | Train folds | Train rows | Holdout | Raw | Evaluable | Excluded |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `window_001_holdout_wf_004` | `wf_002,wf_003` | 677 | `wf_004` | 23 | 23 | 0 |
| `window_002_holdout_wf_005` | `wf_002,wf_003,wf_004` | 700 | `wf_005` | 22 | 17 | 5 |
| `window_003_holdout_wf_006` | `wf_002,wf_003,wf_004,wf_005` | 717 | `wf_006` | 30 | 25 | 5 |

## Aggregate true-OOS comparison

- Rows: `65` evaluable of `75` raw; coverage `0.8666666666666666666666666667`.
- Champion accuracy / Brier / log loss / ECE: `0.6615384615384615384615384615` / `0.2315416168638198232432961914` / `0.6557111877723201715938736175` / `0.1031422314134338271619605943`.
- Challenger accuracy / Brier / log loss / ECE: `0.6769230769230769230769230769` / `0.2313831777277822686393046858` / `0.6553779842799703123736428102` / `0.1125033096092105167507272071`.
- Aggregate metric direction: `CHALLENGER_BETTER`.
- Conclusion: `MIXED_OR_INCONCLUSIVE`.

## Safety claims

- Training and holdout game identities are disjoint in every window.
- Champion and challenger are scored on identical evaluable rows per window.
- Predictions are generated from point-in-time feature rows before final outcomes are paired.
- No aggregate-OOS tuning, calibration fitting, promotion, betting, profitability, or staking claim is made.
