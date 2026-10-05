# MatchAnalysis

MatchAnalysis is a minimal Python modular-monolith foundation for describing
matches and their evidence. Its Moneyline decision path records hypothetical
paper positions and settlements only; it never submits or supports real bets.
It is not production ready.

## Architecture

Dependencies point inward:

```text
interfaces
    ↓
application
    ↓
core and baseball domain

infrastructure
    ↓
application ports and domain

core and domain
    ↓
Python standard library only
```

The `core` and `baseball.domain` packages cannot depend on application,
infrastructure, or interfaces. Application code cannot depend on
infrastructure or interfaces. Infrastructure may implement application ports,
and interfaces may call application use cases.

Betting-pool is a frozen, read-only reference implementation. Future migration
must proceed capability by capability, beginning with characterization and
ending with explicit parity evidence. No legacy capability is migrated by this
bootstrap.

## Run the checks

Python 3.14.4 was used to verify this bootstrap.

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
```

## Moneyline paper decisions

The V1 consumer reads one frozen P50C Moneyline run, then records both BET and
NO_BET decisions in an isolated state file. It never writes the P50C prediction
ledger or the existing paper ledgers. Without quote input it records
`NO_BET / ODDS_SOURCE_NOT_CONFIGURED`. To read the configured public TSL source,
use `--fetch-live-tsl`; its capture and paper state are stored under
`report/moneyline_paper_decision_v1/`.

Provide all three policy inputs. The values below are illustrative assumptions,
not tuned betting parameters:

```bash
PYTHONPATH=src ./.venv/bin/python -m match_analysis.interfaces.cli.moneyline_paper_decision \
  --prediction-run p50c_run_<run-id> \
  --data-mode PROSPECTIVE \
  --fetch-live-tsl \
  --min-net-ev 0.05 \
  --max-quote-age-seconds 900 \
  --paper-cost-per-unit 0.02
```

For offline ingestion, replace `--fetch-live-tsl` with
`--quote-snapshots path/to/normalized_quotes.jsonl`. Each row must include the
full game identity from the frozen prediction, `market: "MONEYLINE"`,
`market_code: "MNL"`, the observation timestamp, source and source identity,
and decimal odds for both HOME and AWAY. Use `--data-mode FIXTURE` only for
clearly marked fixture data. Add `--final-results` with P44A normalized FINAL
result rows to settle eligible paper positions and grade all applicable
predictions.

The stake is fixed at one paper unit. `paper_cost_per_unit` is charged on WIN
and LOSS and is not charged on VOID; the current official result contract does
not support VOID grading, so non-final games remain pending. The state records
quote provenance and uses the odds frozen with each decision for settlement.
Missing configuration, invalid or late data, stale/future quotes, and identity
mismatches produce NO_BET records.

Moneyline game identity preserves MLB's exact `gameType`. P52A and P53A
default to `R`; use P52A's `--game-type` to opt into one exact provider type,
then pass the same `--game-type` to P53A for that frozen run. Non-R predictions
are marked `POSTSEASON_OR_NON_R_DIAGNOSTIC` because the existing champion model
was trained on regular-season data. P50C's headline metrics remain regular
season, with separate `game_type_cohorts` counts and metrics for other types;
the paper state also reports separate cohorts and labels any all-type totals.

## Explicit exclusions

The paper decision path adds no scheduler, database, frontend, model training,
automatic promotion, Run Line support, Kelly sizing, or real-money execution.
