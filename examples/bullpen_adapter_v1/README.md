# Legacy bullpen evidence adapter v1

An offline bridge to `bullpen.research_snapshot.v1`. This adapter is research-only
and changes no model, ledger, pricing, prediction or betting workflow. It does
not restore the missing `448b057` candidate or provide accuracy/profit evidence.

## Reproduce synthetic cases

From the repository root, with Python 3.14 and no third-party dependencies:

```sh
PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.adapt_legacy_bullpen_evidence \
  --source examples/bullpen_adapter_v1/relief.synthetic.jsonl \
  --format phase63-jsonl \
  --target examples/bullpen_adapter_v1/target.synthetic.json \
  --capture-evidence examples/bullpen_adapter_v1/capture_evidence.synthetic.json \
  --output /tmp/bullpen-adapter-captured.json

PYTHONPATH=src python3.14 -m match_analysis.interfaces.cli.adapt_legacy_bullpen_evidence \
  --source examples/bullpen_adapter_v1/date_only_context.synthetic.json \
  --format phase56-context \
  --target examples/bullpen_adapter_v1/target.synthetic.json \
  --output /tmp/bullpen-adapter-rejected.json

PYTHONPATH=src python3.14 -m unittest \
  tests.unit.test_legacy_bullpen_evidence_adapter \
  tests.unit.test_bullpen_research_snapshot tests.unit.test_moneyline_feature_snapshot
```

The outputs must be new files. Committed outputs are labeled synthetic and can
be reproduced byte for byte. No file mtime, current time or date-to-midnight
conversion is used. Duplicate JSON keys, non-finite JSON values, malformed
JSONL and unbound source/record hashes fail closed.

API: `adapt_legacy_bullpen_evidence(source_bytes, source_format=..., target=...,
evidence=...)` in `application.use_cases.adapt_legacy_bullpen_evidence` returns
the normalized research input, research snapshot/fingerprint, raw-source and
capture-metadata hashes, retained capture assertions, all row decisions and
per-team coverage diagnostics. It performs no network or file writes.

## Supported source formats and missing evidence

- `phase56-context`: the old `home_schedule`/`away_schedule` JSON context.
  Date-only/team-level proxy rows are always rejected as individual pitching
  records. Even an empty aggregate context cannot establish complete relief
  coverage or known zero workloads.
- `three-day-aggregate-jsonl`: existing rolling 3-day team totals. Even a real
  `fetched_at` cannot reconstruct individual appearances, outs, pitcher identity
  or actual usage dates. Rows are rejected rather than splitting aggregate
  totals or assigning them to guessed pitchers.
- `phase63-jsonl`: old `source_game_id`, `game_date`, `team`, `pitcher_id`,
  `appeared_order`, starter/opener/reliever flags, `outs_recorded` and optional
  `pitches_thrown`/`earned_runs`/`high_leverage`. No outs are inferred from
  rounded decimal innings. Estimated schedule data remains rejected. Pitcher
  order 1, including an opener mislabeled as a legacy reliever, is excluded
  from bullpen use. Conflicting role/order assertions are rejected.

## Explicit archived capture contract

The capture sidecar is `bullpen.legacy_capture_evidence.v1` (see the complete
synthetic example). Its `source_sha256` must match the exact source bytes.
For JSONL, every binding must match a physical 1-based line index and SHA-256
of that raw line, including its line ending. Context hashes instead identify
the canonical side+row; context rows remain ineligible regardless of bindings.

Every granular binding supplies namespace, provider game id, game number,
pitcher id and original game date, checked against the raw row. Team identities
use explicit exact legacy-name/provider-id/canonical-code mappings, never fuzzy
name inference. It also supplies actual usage date and the source-confirmed
single-day basis, historical started/ended times, response-received time,
persisted time, source id and source kind (`ARCHIVED_LOCAL_PUBLIC_CAPTURE` or
`SYNTHETIC_FIXTURE`). All are mandatory. A game date, truncated audit hash,
`pit_safe` flag, generated report date or today’s retrieval cannot substitute
for any missing historical timestamp.

Availability is the explicit persistence time with
`started < ended <= received <= persisted`; admission additionally requires
`ended < cutoff` and `persisted <= cutoff < target start`. The adapter verifies
content hash bindings and chronology, **not independent timestamp authenticity**.
Capture times remain caller archive assertions and are retained under that label;
real use requires actual archived metadata, not a newly invented sidecar.

For multiple versions of one pitcher/game, the last admissible persisted version
is selected once. Later revisions cannot rewrite the pre-cutoff snapshot. Equal
timestamp/equal content captures collapse; equal timestamp/conflicting content
makes the group unknown. Revision identity/date/team conflicts fail closed.
Earlier same-day game 1 may contribute to `prior_same_day`; target game 2 is
excluded. Source stats are retained as-is; missing pitches/ER/leverage stay null.

Coverage carries the research snapshot's explicit per-team source assertion,
time range, available-at and provenance. Missing/unmapped/rejected source rows
conservatively downgrade team coverage, so an incomplete import cannot produce
false zero workload. A later revision does not downgrade coverage when the
archived pre-cutoff version is admissible. Missing declarations are reported
for both teams. Completeness remains caller evidence, not an audited roster or
schedule proof; the inherited rest/transfer limitations remain unchanged.

## Observed local-source limits

`legacy_source_inventory.json` records hashes and small offline inspections,
without copying or modifying the old repository. The Phase63 local report has
26 pitching rows and zero precise availability/capture timestamps. The old
3-day aggregate's first row has a 2026 capture timestamp for 2025-described
data, with no individual pitcher or actual usage-date evidence. A capture after
a historical prediction cutoff cannot be relabeled as earlier availability.
The existing newer schedule-payload fixture is explicitly synthetic and has
no bullpen pitching evidence. Old source labels/PIT flags alone prove neither
actual historical availability nor a genuinely public capture.

The committed `legacy_phase63_rejection.json` uses the synthetic target solely
to demonstrate the old source's 26/26 missing-capture rejections and zero imports.
It is not a historical accuracy sample. The synthetic positive case selects
4 of 6 rows: home prior-3-day outs 9, earlier same-day outs 4, one actual
back-to-back pitcher; an opener is excluded and a post-cutoff revision ignored.
The date-only case selects 0 of 2 rows and keeps all unsupported loads unknown.
