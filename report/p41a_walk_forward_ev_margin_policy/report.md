# P41A Leakage-Safe Walk-Forward EV-Margin Policy

This is a deterministic historical true-OOS paper-only comparison. It is
not a real betting recommendation, profitability claim, staking strategy,
model-promotion decision, or future-performance claim.

## Frozen candidate thresholds

- Candidate thresholds: `0.00`, `0.01`, `0.02`, `0.03`, `0.05`.
- Selection objective: highest cumulative prior net paper units.
- Tie-break: larger EV threshold.
- Target outcomes are attached only after each target threshold is frozen.

## Chronological target windows

| Target window | Prior policy windows | Prior rows | Selected T | Tie-break reason | Target rows |
| --- | --- | ---: | ---: | --- | ---: |
| `window_002_holdout_wf_005` | `window_001_holdout_wf_004` | 22 | `0.05` | `LARGER_EV_THRESHOLD_ON_EQUAL_PRIOR_NET_UNITS` | 15 |
| `window_003_holdout_wf_006` | `window_001_holdout_wf_004, window_002_holdout_wf_005` | 37 | `0.03` | `HIGHEST_PRIOR_CUMULATIVE_NET_PAPER_UNITS` | 25 |

### `window_002_holdout_wf_005` prior candidate metrics

| Threshold | BET | PASS | Wins | Losses | Units risked | Net units | ROI | Max drawdown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | 10 | 12 | 7 | 3 | 10.0 | 3.28 | 0.328 | 2 |
| `0.01` | 10 | 12 | 7 | 3 | 10.0 | 3.28 | 0.328 | 2 |
| `0.02` | 9 | 13 | 7 | 2 | 9.0 | 4.28 | 0.47555555555555555555555555555555555555555555555556 | 1 |
| `0.03` | 9 | 13 | 7 | 2 | 9.0 | 4.28 | 0.47555555555555555555555555555555555555555555555556 | 1 |
| `0.05` | 9 | 13 | 7 | 2 | 9.0 | 4.28 | 0.47555555555555555555555555555555555555555555555556 | 1 |

| Target policy | BET | PASS | Wins | Losses | Pushes | Units risked | Net units | ROI | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected T=0.05 | 3 | 12 | 2 | 1 | 0 | 3.0 | 1.82 | 0.60666666666666666666666666666666666666666666666667 | 1.00 |
| Fixed zero-EV T=0.00 | 5 | 10 | 3 | 2 | 0 | 5.0 | 1.47 | 0.294 | 1.00 |

### `window_003_holdout_wf_006` prior candidate metrics

| Threshold | BET | PASS | Wins | Losses | Units risked | Net units | ROI | Max drawdown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | 15 | 22 | 10 | 5 | 15.0 | 4.75 | 0.31666666666666666666666666666666666666666666666667 | 2 |
| `0.01` | 15 | 22 | 10 | 5 | 15.0 | 4.75 | 0.31666666666666666666666666666666666666666666666667 | 2 |
| `0.02` | 14 | 23 | 10 | 4 | 14.0 | 5.75 | 0.41071428571428571428571428571428571428571428571429 | 1 |
| `0.03` | 13 | 24 | 10 | 3 | 13.0 | 6.75 | 0.51923076923076923076923076923076923076923076923077 | 1 |
| `0.05` | 12 | 25 | 9 | 3 | 12.0 | 6.10 | 0.50833333333333333333333333333333333333333333333333 | 1 |

| Target policy | BET | PASS | Wins | Losses | Pushes | Units risked | Net units | ROI | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected T=0.03 | 5 | 20 | 3 | 2 | 0 | 5.0 | 1.33 | 0.266 | 1.00 |
| Fixed zero-EV T=0.00 | 7 | 18 | 4 | 3 | 0 | 7.0 | 1.15 | 0.16428571428571428571428571428571428571428571428571 | 2.00 |

## Aggregate true-OOS target comparison

| Policy | BET | PASS | Wins | Losses | Pushes | Units risked | Net units | ROI | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Walk-forward selected margin | 8 | 32 | 5 | 3 | 0 | 8.0 | 3.15 | 0.39375 | 1.00 |
| Fixed zero-EV baseline | 12 | 28 | 7 | 5 | 0 | 12.0 | 2.62 | 0.21833333333333333333333333333333333333333333333333 | 2.00 |

- Target windows: `2`; target rows: `40`.
- Selected threshold counts: `{'0.00': 0, '0.01': 0, '0.02': 0, '0.03': 1, '0.05': 1}`.

## Conclusion

- `CONCLUSION: EV_MARGIN_POLICY_IMPROVED`.

## Safety boundary

- P37/P38/P39/P40 authority is read-only and remains unchanged.
- Champion probabilities and model authority were not changed.
- Real betting: `NOT RUN`.
- Staking/Kelly and bankroll optimization: `NOT RUN`.
- Model promotion, retraining, calibration, and deployment: `NOT RUN`.
