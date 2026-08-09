# P20B First Non-Synthetic Historical Feedback Replay

This is a deterministic, historical, paper-only diagnostic lineage.
It is not a training dataset and makes no production-performance or betting claim.

## Lineage

- **P20A Fold**: `wf_001`
- **P20A Model Fingerprint**: `5a0e7a0a90253c90eeed30a6f7672b578a76dc2a00a2fa3da7cf1196c0e80155`
- **P19A Model Artifact Fingerprint**: `5af07fa9d346e56be92bb0a185effa470de67106b2fb314dd5b37ad2a19bfd79`
- **Replay Game IDs**: `2025-06-01_ATL_BOS, 2025-06-01_TEX_STL`
- **P20A Prediction Artifact SHA-256**: `0aeee05fb8f8ccdcbe963cbb695e36bf732d55c84f8a66bbe362f628d2ef88cb`

## Historical Result Provenance

- **Repository**: `/Users/kelvin/Kelvin-WorkSpace/Betting-pool`
- **Commit**: `03b2fcf4de1a13ee9929afcef803d61955c9f41b`
- **Tree**: `56a849bc68234db63da7a38f1643fa664217c5d0`
- **Archive Path**: `data/mlb_2025/gl2025.zip`
- **Archive Blob**: `3e9b08be2530870f38c474db316de6de58b1b381`
- **Archive SHA-256**: `957a7cff15cf7926889749c3ef99802ef030ee1b5f7c112b06ba5cb810df5f76`
- **Archive Member**: `gl2025.txt`
- **Evidence Verified At**: `2026-03-12T06:29:35.016973Z`

| Game | Away Score | Home Score | Derived Winner |
| --- | ---: | ---: | --- |
| `2025-06-01_ATL_BOS` | 3 | 1 | `AWAY` |
| `2025-06-01_TEX_STL` | 1 | 8 | `HOME` |

## Contract Results

- **P15C admitted observations**: `4`
- **P16A attached observations**: `4`
- **P16B evaluated observations**: `4`
- **P17A feedback rows**: `4`
- **Correct / Incorrect**: `2` / `2` (diagnostic only)
- **Feedback Ledger Fingerprint**: `adf320cb91681254ed9ca79467c818406a8d64beaf0631bf61afab1bcb13e087`

## Explicit Claims

- **db_written**: `false`
- **deployed**: `false`
- **deterministic**: `true`
- **diagnostic**: `true`
- **historical**: `true`
- **model_promoted**: `false`
- **network_called**: `false`
- **non_synthetic**: `true`
- **odds_used**: `false`
- **paper_only**: `true`
- **profitability_claim**: `false`
- **real_betting_recommendation**: `false`
- **retraining_performed**: `false`
- **sample_limited**: `true`
- **synthetic_results**: `false`
- **training_authorized**: `false`
- **training_dataset_claim**: `false`

## Safety Boundaries

- Exactly two P20A replay game IDs are included.
- Both committed P20A HOME/AWAY candidate observations are retained.
- Existing P15C/P16A/P16B/P17A semantics are reused; their implementations are unchanged.
- No provider, network, database, odds, ROI, EV, Kelly, training, retraining, or promotion behavior is used.
- This sample is insufficient for training readiness, production readiness, or model-quality claims.
