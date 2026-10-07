"""Small offline source audit: writes a new eligibility folder, never a ledger."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from ...application.use_cases.adapt_legacy_bullpen_evidence import load_json
from ...application.use_cases.audit_tsl_archive_eligibility import audit_tsl_archive_eligibility


def _dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "schedule", "prior-summary", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source", "schedule", "prior"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    args = parser.parse_args(argv)
    prior_raw = args.prior_summary.read_bytes()
    if sha256(prior_raw).hexdigest() != args.expected_prior_sha256:
        raise ValueError("prior scope hash mismatch")
    prior_ids = {str(pk) for window in load_json(prior_raw)["evaluation_windows"] for pk in window["holdout"]["raw_game_ids"]}
    schedule_raw = args.schedule.read_bytes()
    result = audit_tsl_archive_eligibility(args.source, expected_source_sha256=args.expected_source_sha256,
                                         schedule_raw=schedule_raw, expected_schedule_sha256=args.expected_schedule_sha256,
                                         prior_game_ids=prior_ids)
    directory = args.output_dir
    directory.mkdir()  # New-only output; no research or ledger overwrite.
    outputs = {"summary.json": _dump({k:v for k,v in result.items() if k not in ("games", "row_decisions")}),
               "eligible_games.jsonl": "".join(json.dumps(g,ensure_ascii=False,sort_keys=True,separators=(",", ":"))+"\n" for g in result["games"]),
               "row_decisions.jsonl": "".join(json.dumps(g,ensure_ascii=False,sort_keys=True,separators=(",", ":"))+"\n" for g in result["row_decisions"])}
    for name, text in outputs.items():
        (directory/name).write_text(text)
    manifest = {"schema_version": "tsl.archive_eligibility_sources.v1",
                "tsl_source": {"path": str(args.source), "sha256": args.expected_source_sha256},
                "official_schedule": {"path": str(args.schedule), "sha256": args.expected_schedule_sha256},
                "prior_scope": {"path": str(args.prior_summary), "sha256": args.expected_prior_sha256,
                                "use": "IDENTITY_OVERLAP_ANNOTATION_ONLY; NEVER_OUTCOME_FILTER"},
                "outputs_sha256": {name:sha256(text.encode()).hexdigest() for name,text in outputs.items()},
                "source_portability": "Original TSL archive is reused in place; not duplicated into this research artifact. Supply the same hash-bound file for replay."}
    (directory/"source_manifest.json").write_text(_dump(manifest))
    print(_dump(result["summary"]).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
