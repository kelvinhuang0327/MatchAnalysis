# Result-Only Paper Decision Replay Report

## Frozen Decision Selection

- **Decision Schema**: `p18a.result_only_paper_decision.v1`
- **Decision Count**: `3`
- **Excluded Snapshot Rows**: `0`
- **Source Snapshot Fingerprint**: `858f1740463ff8f0b5556f26489445eec74c33377ba75f2fc7e65895d7f4f1c1`
- **Decision Set Fingerprint**: `58dcc554864fc39b12d1935de4a8070266cb3259581808ee29ed26bdc2339745`

Decisions are selected from the prediction snapshot before final-result bytes are read.

| Decision ID | Prediction Observation | Provider Game | Selection | Prediction Time (UTC) |
| --- | --- | --- | --- | --- |
| `6eb6809b6311fe45...` | `9671f4340017a7b6...` | `888001#2` | `HOME` | `2026-04-05T11:10:00Z` |
| `9cc721cd4587d75c...` | `0cabf8e0dbc4a790...` | `888001#1` | `HOME` | `2026-04-05T11:00:00Z` |
| `ca4d0f3539039057...` | `573447cd9d62bff6...` | `888002#1` | `AWAY` | `2026-04-05T11:05:00Z` |

## Result-Only Settlement

- **Settlement Schema**: `p18a.result_only_paper_settlement.v1`
- **Settled**: `3`
- **Unsettled**: `0`
- **Won**: `2`
- **Lost**: `1`
- **Settlement Set Fingerprint**: `78791c5a29e51082aa1b48b267cc5bd7006b47e801ea407ac704f53001c728b3`

| Decision ID | Selection | Actual Winner | Score | Status |
| --- | --- | --- | --- | --- |
| `6eb6809b6311fe45...` | `HOME` | `AWAY` | `1-3` | `LOST` |
| `9cc721cd4587d75c...` | `HOME` | `HOME` | `5-3` | `WON` |
| `ca4d0f3539039057...` | `AWAY` | `AWAY` | `2-4` | `WON` |

## Explicit Limitations

- This is a paper-only, result-only replay of synthetic local artifacts.
- No price, payout, P&L, ROI, EV, Kelly, or profitability calculation was performed.
- Final outcomes affect settlement status only; they do not affect decision selection.
- No provider or network call was made.
- No database write or deployment occurred.
- No training, model promotion, or performance claim was made.
