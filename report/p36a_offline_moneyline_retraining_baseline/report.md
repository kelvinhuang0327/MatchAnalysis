# P36A Offline Moneyline Retraining Baseline

This is a deterministic offline champion/challenger comparison. No model was promoted.

## Authority and split

- Training rows: `700` eligible, `0` excluded.
- Training range: `2025-07-01` to `2026-06-09`.
- Holdout rows: `42` evaluable of `52` raw.
- Holdout range: `2026-06-10` to `2026-06-13`.
- Holdout coverage: `0.8076923076923076923076923077`.

## Model comparison

- Champion: `p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630`.
- Challenger: `p36a_moneyline_logistic_retraining_challenger_v1_9d5a2c070fe00427`.
- Accuracy: champion `0.5952380952380952380952380952`, challenger `0.6190476190476190476190476190`.
- Brier: champion `0.2412503038891353958101706395`, challenger `0.2409542612751445629346755433`.
- Log loss: champion `0.6754482605925976333743156102`, challenger `0.6748400665006979300167290164`.
- Verdict: `CHALLENGER_BETTER`.

## Safety claims

- Strict temporal split, point-in-time features, same holdout membership, and outcome isolation were verified.
- This report makes no betting, profitability, production-readiness, or promotion claim.
