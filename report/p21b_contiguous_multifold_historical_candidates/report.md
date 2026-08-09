# P21B Contiguous Multifold Historical Candidate Replay

This is a bounded, historical, paper-only replay. It is not a training
dataset, model-promotion event, profitability analysis, or betting recommendation.

## Fold sequence

| Fold | Training cutoff | Training rows | Prediction games | Max parity difference | Parity |
| --- | --- | ---: | ---: | ---: | --- |
| `wf_002` | `2025-06-30` | 893 | 319 | `4.992170296492576096638E-7` | `True` |
| `wf_003` | `2025-07-31` | 1212 | 358 | `4.974175138497870790439E-7` | `True` |

## Lineage counts

- P15C admissions: `1354`
- P16A attached results: `1354`
- P16B evaluations: `1354`
- P17 feedback rows: `1354`
- P21A assessments: `1354`
- P21A eligible candidates: `1354`
- P21A excluded assessments: `0`

## Deterministic identities

- Membership SHA-256: `afba81ae0d9858905675b64717b59abd082bb62fd256820f933a7b845ed8d163`
- Historical result rows SHA-256: `79fd4f858bc6c70c2c2d044460503baf60cebaf2a9a58ce16ee0c78671c34064`
- Aggregate P17 ledger fingerprint: `8c4f00f5adc3c5207329be26e8f82b05f4d65e66bf214b57fc05dfb9beda9d9d`
- Aggregate P21A assessment fingerprint: `bc5ff8bed5c2db939d8da3035421e777e49e11da50dc40e434997d1548ede2e6`
- Aggregate candidate fingerprint: `6cd935ea6d5ac5707726b25b2d1c42f8007a35a9b87f740792aa28b576424ac0`

## Historical provenance

- Repository: `/Users/kelvin/Kelvin-WorkSpace/Betting-pool`
- Commit: `03b2fcf4de1a13ee9929afcef803d61955c9f41b`
- Tree: `56a849bc68234db63da7a38f1643fa664217c5d0`
- Archive: `data/mlb_2025/gl2025.zip`
- Member: `gl2025.txt`
- Result rows: `677`

## Safety claims

- `historical=true`, `sample_limited=true`, `synthetic_results=false`
- `training_dataset_claim=false`, `training_authorized=false`, `retraining_performed=false`
- `model_promoted=false`, `profitability_claim=false`, `real_betting_recommendation=false`
- No provider, network, database, odds, deployment, push, or remote CI action was used.
- P20B historical runtime compliance remains REFUTED; this task preserves that prior finding.
