# P39A TSL Moneyline Market Snapshot Join

This artifact attaches only trustworthy pregame two-sided Moneyline observations to the P37 true-OOS prediction universe. It does not make betting decisions.

## Conclusion

- Result: `MARKET_JOIN_PARTIAL`
- Rule: `MARKET_JOIN_READY iff every P37 evaluable target has one usable pregame snapshot; MARKET_JOIN_PARTIAL iff at least one but fewer than all targets are usable; MARKET_AUTHORITY_INSUFFICIENT iff zero targets are usable or timestamp/source authority is not trustworthy.`
- Edge-ready rows: `62` of `65`

## Coverage

- Exact identity matches: `62`
- Usable pregame Moneyline rows: `62`
- No-market rows: `3`
- Post-start rejected rows: `0`
- Ambiguous rows: `0`
- Missing or untrusted timestamp rows: `0`
- Malformed or incomplete price rows: `0`
- Not-pregame rejected rows: `0`

## Provenance

- Source repository: `/Users/kelvin/Kelvin-WorkSpace/Betting-pool`
- Source path: `/Users/kelvin/Kelvin-WorkSpace/Betting-pool/data/tsl_odds_history.jsonl`
- Source HEAD/tree: `03b2fcf4de1a13ee9929afcef803d61955c9f41b` / `56a849bc68234db63da7a38f1643fa664217c5d0`
- Source SHA-256: `5604e41f817f87617956b54c4b664bbf562d496eb1c8618bd174888ef87c8efc`
- Source rows inspected: `11400`
- Scoped source rows: `843`
- Timestamp semantics: `SOURCE_GAME_TIME_IS_SCHEDULED_START;SOURCE_FETCHED_AT_IS_LOCAL_MARKET_OBSERVATION_TIME;PROVIDER_SIDE_TIMESTAMP_UNAVAILABLE`
- Selected snapshot rule: `LATEST_TRUSTWORTHY_PREGAME_OBSERVATION_STRICTLY_BEFORE_SCHEDULED_START;_TIES_BY_SOURCE_ROW_FINGERPRINT`
- P37 comparisons SHA-256: `23cc15d308a90c08da0d1a4c6cbb9289af3add2c5d151808833e73a660639eb4`

## Safety boundary

- P37 predictions and P38 calibration artifacts are read-only inputs.
- Market snapshot selection does not read or use outcomes.
- BET/PASS, ROI, profitability, bankroll, staking, Kelly, calibration, and model promotion are `NOT RUN`.
