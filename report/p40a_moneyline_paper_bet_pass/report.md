# P40A Moneyline Paper BET/PASS Baseline

This is a deterministic historical true-OOS paper-only measurement. It is not a real betting recommendation, profitability claim, staking strategy, or model-promotion decision.

## Frozen decision rule

- `p_home = model pregame home-win probability`; `p_away = 1 - p_home`.
- `EV_home = p_home * home_decimal_odds - 1`; `EV_away = p_away * away_decimal_odds - 1`.
- BET the side with the larger EV iff `max(EV_home, EV_away) > 0`; otherwise PASS.
- No additional edge threshold, minimum odds filter, confidence cutoff, or tuning was applied.

## Authority and coverage

- P39 target universe: `65`; edge-ready rows: `62`; no-market rows: `3`.
- Decisions: `124` (`62` Champion primary plus `62` raw-challenger shadow).
- Deterministic rerun: `True`.
- P37, P38, and P39 authorities are read-only inputs; P38 calibration is out of scope.

## Aggregate paper results

| Policy | BET | PASS | Wins | Losses | Pushes | Hit rate | Units risked | Net units | DESCRIPTIVE_PAPER_ONLY ROI | Max drawdown | Avg predicted EV of BET rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Champion primary | 22 | 40 | 14 | 8 | 0 | 0.63636363636363636363636363636363636363636363636364 | 22.0 | 5.90 | 0.26818181818181818181818181818181818181818181818182 | 2 | 0.095593165048767448380325672136363636363636363636364 |
| Raw challenger shadow | 25 | 37 | 16 | 9 | 0 | 0.64 | 25.0 | 6.32 | 0.2528 | 3.00 | 0.08407371737291951354005543808 |

- Primary conclusion: `PAPER_BASELINE_OBSERVED_POSITIVE` (realized net paper units only; sample size `N=62`).
- Shadow comparison: `SHADOW_CHALLENGER_HIGHER_NET_UNITS`; this does not authorize selection or promotion.

## Per-window paper results

| Window | Policy | Edge-ready | BET | PASS | Wins | Losses | Net units | DESCRIPTIVE_PAPER_ONLY ROI | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_001_holdout_wf_004 | Champion primary | 22 | 10 | 12 | 7 | 3 | 3.28 | 0.328 | 2 |
| window_001_holdout_wf_004 | Raw challenger shadow | 22 | 10 | 12 | 7 | 3 | 3.28 | 0.328 | 2 |
| window_002_holdout_wf_005 | Champion primary | 15 | 5 | 10 | 3 | 2 | 1.47 | 0.294 | 1.00 |
| window_002_holdout_wf_005 | Raw challenger shadow | 15 | 5 | 10 | 3 | 2 | 1.47 | 0.294 | 1.00 |
| window_003_holdout_wf_006 | Champion primary | 25 | 7 | 18 | 4 | 3 | 1.15 | 0.16428571428571428571428571428571428571428571428571 | 2.00 |
| window_003_holdout_wf_006 | Raw challenger shadow | 25 | 10 | 15 | 6 | 4 | 1.57 | 0.157 | 3.00 |

## Safety boundary

- `DESCRIPTIVE_PAPER_ONLY` means realized net paper units divided by paper units risked over this historical true-OOS sample; it is not expected future ROI or proven profitability.
- Paper stake convention: `1.0 PAPER UNIT` per BET; PASS risks zero. No Kelly, bankroll management, or variable stake sizing was used.
- Real betting: `NOT RUN`.
- Threshold optimization: `NOT RUN`.
- Staking/Kelly optimization: `NOT RUN`.
- Model promotion: `NOT RUN`.
- External market acquisition: `NOT RUN`.
