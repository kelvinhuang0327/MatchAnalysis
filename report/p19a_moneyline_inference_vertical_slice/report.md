# P19A Deterministic Moneyline Inference

This artifact is a bounded, paper-only, diagnostic inference slice.

## Result

- Candidate count: `2`
- Admission count: `0`
- Candidate-set fingerprint: `6e6619a074b2bd5fe7f5072d6a417c532ee0b0f97ebff005447ea36ea25b8688`
- Model-artifact fingerprint: `5cec1105c4c2fb9a4ff6de736475977fd3bac1b141bcaacd9b115fd295d7899f`

## Legacy provenance

- Repository: `/Users/kelvin/Kelvin-WorkSpace/Betting-pool`
- Commit: `03b2fcf4de1a13ee9929afcef803d61955c9f41b`
- Tree: `56a849bc68234db63da7a38f1643fa664217c5d0`
- Path: `scripts/run_mlb_walk_forward_ml_candidate.py`
- Path: `wbc_backend/prediction/mlb_independent_feature_builder.py`
- Path: `wbc_backend/prediction/mlb_independent_features.py`
- Path: `wbc_backend/prediction/mlb_ml_feature_matrix.py`
- Path: `wbc_backend/prediction/mlb_walk_forward_model.py`

## Boundaries

- Market: `moneyline` only.
- Features: P13 independent recent-form and starter-ERA deltas only.
- No final score, result, settlement, odds join, provider call, database write, training, or profitability claim.
- The artifact does not claim broad historical parity or production accuracy.
