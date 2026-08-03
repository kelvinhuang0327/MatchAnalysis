# Prospective Prediction Admission Workflow Report

## Summary

- **Schedule As-Of (UTC)**: `2026-04-05T12:00:00Z`
- **Total Requests**: `9`
- **Admitted**: `3`
- **Rejected**: `6`
- **Result Set Fingerprint**: `78ce0f891006b3a33c7bca9acabec0856cffb5a28d4b164b00d641a14a9c72d1`

## Input Hashes (SHA-256)

- **authority_catalog_sha256**: `492d99b6b1efe9f16d605e0c5e0bb3880d4c47742c503f1ce92dd7e57ccfaf54`
- **participant_mapping_catalog_sha256**: `af2f1ec7362da0b5e61c5fb67ff535ce0646b88576d61694cda0fb772f45c2c0`
- **prediction_requests_sha256**: `79bef7ba3af6a2aa82eee1450079827b874cf3660da6af88f798810f02193460`
- **raw_schedule_payloads_sha256**: `b83e89dd51c3953c174a6c6b64aa6edfed2dc50d423c1509b885a8cf322dad65`

## Rejection Reasons Breakdown

| Rejection Reason | Count |
| --- | --- |
| `EXACT_IDENTITY_MISMATCH` | 1 |
| `INVALID_PREDICTION_TIMESTAMP_ORDER` | 1 |
| `MISSING_SCHEDULE_CANDIDATE_MATCH` | 1 |
| `PREDICTION_NOT_BEFORE_SCHEDULED_START` | 1 |
| `SCHEDULE_NOT_PREGAME_ELIGIBLE` | 1 |
| `SCHEDULE_OBSERVATION_ID_MISMATCH` | 1 |

## Request Results

| Index | Admission Status | Reason | Provider Game ID |
| --- | --- | --- | --- |
| 1 | `ADMITTED` | None | `888001` |
| 2 | `ADMITTED` | None | `888002` |
| 3 | `ADMITTED` | None | `888001` |
| 4 | `REJECTED` | `MISSING_SCHEDULE_CANDIDATE_MATCH` | `N/A` |
| 5 | `REJECTED` | `INVALID_PREDICTION_TIMESTAMP_ORDER` | `N/A` |
| 6 | `REJECTED` | `PREDICTION_NOT_BEFORE_SCHEDULED_START` | `N/A` |
| 7 | `REJECTED` | `SCHEDULE_OBSERVATION_ID_MISMATCH` | `N/A` |
| 8 | `REJECTED` | `EXACT_IDENTITY_MISMATCH` | `N/A` |
| 9 | `REJECTED` | `SCHEDULE_NOT_PREGAME_ELIGIBLE` | `N/A` |

## Explicit System Claims

- **Provider Called**: `false`
- **DB Written**: `false`
- **Legacy Rows Admitted**: `false`
- **Deployed**: `false`
- **Betting Claim**: `false`
