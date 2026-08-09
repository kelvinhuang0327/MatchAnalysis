# P19A Deterministic Moneyline Inference

This artifact is a bounded, paper-only, diagnostic inference slice.

## Result

- Candidate count: `2`
- Admission count: `0`
- Candidate-set fingerprint: `8d6c1b69bc9f6534287fce04c260b35bfc6586c8574a48e41dfddd4f284aee6f`
- Model-artifact fingerprint: `d0b2eb6f2ff19d039bd13478edc56c5cac28f23a4a87510412e239286550d76e`
- Artifact kind: `bounded_deterministic_fixture`
- Bounded parity basis: `ffca78865db53ffaebc17110c39a604f484a353d38858c64787ac8a81c7664c9`

## Legacy provenance

- Repository: `/Users/kelvin/Kelvin-WorkSpace/Betting-pool`
- Commit: `03b2fcf4de1a13ee9929afcef803d61955c9f41b`
- Tree: `56a849bc68234db63da7a38f1643fa664217c5d0`
- Path: `scripts/run_mlb_walk_forward_ml_candidate.py`
- Path: `wbc_backend/prediction/mlb_independent_feature_builder.py`
- Path: `wbc_backend/prediction/mlb_independent_features.py`
- Path: `wbc_backend/prediction/mlb_ml_feature_matrix.py`
- Path: `wbc_backend/prediction/mlb_walk_forward_model.py`
- Path: `outputs/predictions/PAPER/2026-05-11/p13_ml/ml_model_metadata.json`
- Path: `outputs/predictions/PAPER/2026-05-11/p13_ml/ml_feature_matrix.csv`
- Path: `outputs/predictions/PAPER/2026-05-11/p13_ml/ml_walk_forward_predictions.jsonl`

## Boundaries

- Market: `moneyline` only.
- Features: P13 independent recent-form and starter-ERA deltas only.
- No final score, result, settlement, odds join, provider call, database write, training, or profitability claim.
- The fixture uses the committed P13 coefficient-summary magnitude and one committed pregame prediction row; it does not claim to recover unavailable fold weights.
- The artifact does not claim broad historical parity or production accuracy.
