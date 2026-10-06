# Bullpen research snapshot v1

This offline module preserves the useful historical bullpen workload calculation
and replaces the legacy date-only and flag-only leakage checks. It is separate
from `MoneylineFeatureSnapshot`; no model feature order, prediction policy, ledger,
or trading workflow consumes it. This migration has no accuracy or profit evidence.

## Reproduce the synthetic example

Python 3.14 and the standard library are sufficient. From the repository root:

```sh
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.build_bullpen_research_snapshot \
  --input examples/bullpen_research_v1/input.synthetic.json \
  --output /tmp/bullpen-research-example.json
PYTHONPATH=src python3.14 -m unittest \
  tests.unit.test_bullpen_research_snapshot tests.unit.test_moneyline_feature_snapshot
```

The output path must be a new file. The committed `output.synthetic.json` is an
example only, with `SYNTHETIC_FIXTURE` provenance. It is reproducible byte for byte.
The CLI rejects duplicate JSON fields and non-finite JSON values, uses no network,
and writes only the requested new output. Its envelope records the exact input
file SHA-256 and the immutable snapshot SHA-256.

The callable API is:

```python
from match_analysis.application.use_cases.build_bullpen_research_snapshot import build_bullpen_research_snapshot
snapshot = build_bullpen_research_snapshot(input_mapping)
projection = snapshot.to_projection()  # detached copy
snapshot.fingerprint()                # full canonical evidence SHA-256
```

`BullpenResearchSnapshot.from_projection()` reloads immutable canonical evidence
and independently rechecks accepted-record timestamps, roles, identities and
duplicate identities. A caller-supplied "safe" flag cannot bypass those checks.

## Input and availability contract

`input.synthetic.json` is the complete `bullpen.research_input.v1` contract example.
Unexpected input fields are rejected, including target outcomes and final scores.

- Target: full `MatchIdentity`, provider namespace/game id/game number, source
  schedule-observation SHA-256, provider official date, distinct home/away team
  ids, timezone-aware `as_of_utc` and `scheduled_start_utc`. Cutoff must precede start.
- Appearances: provider game id, official date and game number, explicit provider
  team id/canonical team code, positive pitcher id, `RELIEF` or `STARTER`, historical
  started/ended timestamps, `available_at_utc`, integer outs and optional integer
  pitches/earned runs and boolean high-leverage flag. Missing optional statistics
  remain unknown. Missing essential history evidence excludes that row and makes
  affected windows unknown. Timestamps are never invented from dates or file mtimes.
- `usage_date` with `usage_date_basis=SOURCE_CONFIRMED_SINGLE_DAY` identifies the
  actual pitching calendar day. It must be supported by the caller's source and
  use the same official-calendar convention as the target date. For suspended or
  resumed games, the original official date alone is insufficient. A pitcher/game
  aggregate spanning multiple pitching dates is unsupported; split it only with
  real source evidence and a later schema supporting segments. V1 fails on
  conflicting duplicate pitcher/game rows instead of counting both.
- Admission requires `started_at < ended_at < cutoff`, and
  `ended_at <= available_at <= cutoff`. Only distinct prior games qualify.
  Same-day earlier doubleheader usage is listed separately; target-game usage
  and starters are excluded. Team names are never used to infer ids.
- Coverage: explicit per-team start date, through timestamp, availability time,
  completeness boolean and source provenance. To report a complete window it
  must span that window through cutoff, be available at cutoff, and have no
  rejected/unavailable relevant record. Coverage is a source assertion, not an
  independently audited schedule/roster completeness proof. Without it even an
  empty appearance list produces unknown, not zero.
- Every appearance/coverage declaration carries source id, kind and SHA-256.
  These source hashes are caller-supplied assertions; the module does not possess
  their raw files. Local input hashes are computed and separately labeled.

## Calculations and limitations

Previous 3/7/14-day windows include dates `[target_date - N, target_date)` and
exclude all target-date usage. Known earlier target-date usage has its own load.
Calendar windows are not rolling 72/168-hour windows. Workload is raw outs,
pitches and appearance counts, without a fitted or saturated fatigue coefficient.
ERA is `27 * earned_runs / outs` over the previous 14 calendar days, with no cap.
High-leverage usage is the supplied high-leverage appearance count divided by
all relief appearances over the previous 7 days; missing flags or a zero
denominator are unknown. No leverage index is inferred.

Back-to-back is the number of *identified pitchers* appearing on both preceding
dates. Rest is the number of full intervening calendar days (yesterday -> 0,
two days ago -> 1, earlier today -> 0), and remains unknown without complete
14-day and same-day coverage. It describes observed team contributors only;
it is not current-roster availability or a cross-team workload/rest estimate.
Current-roster and transfer-aware rest require additional timestamped evidence.

The synthetic home team's previous 3-day outs are 11, previous 7-day outs 16,
earlier same-day outs 4, and one identified reliever worked both preceding days.
An exact duplicate collapses once. The away team's delayed historical record is
unavailable at cutoff, so its affected workloads remain unknown.

## Legacy source trace

Sources were read without modifying the old repository:

- `Betting-pool/wbc_backend/features/mlb_bullpen_feature_builder.py:291`
  SHA-256 `f68608371867a7a965d1aae654d2cee903999d884bb46eef8147ab2518b47b6f`.
- `Betting-pool/wbc_backend/features/mlb_bullpen_pit_validator.py:82`
  SHA-256 `a20e96370c1fbe3ce3f17ec82be8a732e251bdec503fc66c0b9d544a40047b3e`.

Legacy fatigue capped `0.1 * outs` at 1, counted team game dates as a reliever
proxy, imputed missing values as zero/league ERA, and validated flags rather than
availability evidence. V1 uses explicit workloads, actual pitcher intersections,
nullable missing values and chronological evidence. No legacy raw data or
pre-cutoff timestamps were fabricated during the port. The missing `448b057`
candidate and its earlier reported outputs remain unrecovered.
