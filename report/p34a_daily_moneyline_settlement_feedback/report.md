# P34A Daily Moneyline Paper Settlement & Feedback

This is a deterministic paper-only result attachment and feedback report.
The daily sample is small and is not proof of model profitability.

## Frozen P33A Authority

- **P33A Run ID**: `a646ec0081afde1469f978e20bf98fc90cf21587bcecaa9b5cccb280b46bd569`
- **P33A Bundle Fingerprint**: `60823b82ba611077881e3bbcffa19fdcb435391bdf8dbdb5c71299fb4df99175`
- **P33A Analysis Set Fingerprint**: `d5f48ac830fff551bcfa7878b439be63131893cbd0df610d0478dc4aca51656a`
- **P33A Analysis JSONL SHA-256**: `d5f48ac830fff551bcfa7878b439be63131893cbd0df610d0478dc4aca51656a`
- **P33A Pregame Invariance**: `PASS` (source bytes were read-only)
- **Target Date**: `2026-08-13`

## Structural Accounting

| Quantity | Count |
| --- | ---: |
| Official games | 15 |
| TSL/source rows | 14 |
| Qualified source observations | 4 |
| Rejected source observations | 10 |
| P33A analysis rows | 15 |
| Complete P33A predictions eligible for settlement | 0 |
| Structural/non-prediction rows excluded | 15 |

Structural and rejected rows remain separate and produce no evaluation.

## Official Result Authority

- **Source**: `MLB_STATS_API`
- **Observed At (UTC)**: `2026-08-12T02:53:33Z`
- **Target Game Count**: 15
- **Final Result Count**: 0
- **Non-final Target Count**: 15
- **Missing Target Count**: 0
- **Missing Settleable Result Count**: 0
- **Missing Settleable Result Check**: `PASS`
- **Duplicate Result Identity Check**: `PASS`
- **Conflicting Result Check**: `PASS`
- **Non-final Result Check**: `PASS`
- **Result Authority Fingerprint**: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- **Final-result input check**: `PASS` (only `FINAL` observations are attachable)

## Settlement Summary

- **Settlement Status**: `NO_SETTLEABLE_PREDICTIONS`
- **Settled Predictions**: 0
- **Correct**: 0
- **Incorrect**: 0
- **Unresolved**: 0
- **Descriptive Accuracy**: N/A
- **Mean Selected-side Probability**: N/A
- **Brier Score**: N/A
- **Feedback Ledger Fingerprint**: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Prediction Versus Result

| Game | Predicted Side | Model Probability | Market Price | Frozen Edge | Final Score | Actual Winner | Correct | Evaluation Status |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| *No settleable predictions* | — | — | — | — | — | — | — | NOT_EVALUATED |

## Determinism and Safety

- **Offline Replay**: `PASS`
- **Network Called For This Materialization**: `False`
- Pregame P33A prediction, market-price, edge, timestamp, source-fingerprint, and run-identity fields were not rewritten.
- No model retraining, model promotion, scheduler activation, or real betting occurred.
- This report is a small daily sample and must not be used as a profitability conclusion.
