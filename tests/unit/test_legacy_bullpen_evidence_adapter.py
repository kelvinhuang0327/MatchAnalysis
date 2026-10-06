"""Hash bindings, as-of revisions, legacy rejection and offline integration."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from match_analysis.application.use_cases.adapt_legacy_bullpen_evidence import adapt_legacy_bullpen_evidence
from match_analysis.interfaces.cli.adapt_legacy_bullpen_evidence import main

FIXTURE = ROOT / "examples/bullpen_adapter_v1"


def inputs():
    return (FIXTURE / "relief.synthetic.jsonl").read_bytes(), json.loads((FIXTURE / "target.synthetic.json").read_text()), json.loads((FIXTURE / "capture_evidence.synthetic.json").read_text())


def rebind(rows, evidence):
    raw = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    evidence["source_sha256"] = sha256(raw).hexdigest()
    for binding, line in zip(evidence["record_evidence"], raw.splitlines(keepends=True)):
        binding["raw_record_sha256"] = sha256(line).hexdigest()
    return raw


def adapt(raw, target, evidence=None, source_format="phase63-jsonl"):
    return adapt_legacy_bullpen_evidence(raw, source_format=source_format, target=target, evidence=evidence)


class LegacyBullpenAdapterTests(unittest.TestCase):
    def test_bound_capture_admits_only_exact_source_stats_as_of_cutoff(self):
        raw, target, evidence = inputs()
        result = adapt(raw, target, evidence)
        self.assertEqual(result["source_row_count"], 6)
        self.assertEqual(result["selected_relief_appearance_count"], 4)
        self.assertEqual(sum(result["decision_counts"].values()), 6)
        self.assertEqual(result["decision_counts"]["POST_CUTOFF_REVISION_IGNORED_USING_ARCHIVED_PRE_CUTOFF"], 1)
        self.assertEqual(result["decision_counts"]["FIRST_PITCHER_OR_OPENER_NOT_BULLPEN"], 1)
        home, away = result["snapshot"]["teams"]["NYY"], result["snapshot"]["teams"]["BOS"]
        self.assertEqual(home["previous_3_calendar_days"]["outs"], 9)
        self.assertEqual(home["prior_same_day"]["outs"], 4)
        self.assertEqual(home["relievers_worked_both_previous_days"], 1)
        self.assertEqual(away["previous_3_calendar_days"]["outs"], 3)
        self.assertIsNone(away["previous_3_calendar_days"]["pitches"])
        self.assertIsNone(away["era_previous_14_calendar_days"])
        self.assertEqual(result["source_file_sha256"], sha256(raw).hexdigest())

    def test_date_only_and_rolling_aggregates_never_become_observed_pitcher_records(self):
        _, target, evidence = inputs()
        raw = (FIXTURE / "date_only_context.synthetic.json").read_bytes()
        evidence["source_sha256"] = sha256(raw).hexdigest()
        evidence["record_evidence"] = []
        result = adapt(raw, target, evidence, "phase56-context")
        self.assertEqual(result["source_row_count"], 2)
        self.assertEqual(result["selected_relief_appearance_count"], 0)
        self.assertIsNone(result["snapshot"]["teams"]["NYY"]["previous_3_calendar_days"]["outs"])
        self.assertFalse(result["coverage_diagnostics"][0]["effective_complete"])
        raw = b'{"bullpen_usage_last_3d_home":9,"fetched_at":"2026-01-01T10:00:00Z"}\n'
        result = adapt(raw, target, source_format="three-day-aggregate-jsonl")
        self.assertEqual(result["selected_relief_appearance_count"], 0)
        self.assertIn("ROLLING_TEAM_AGGREGATE", result["row_decisions"][0]["reason"])

    def test_missing_capture_evidence_cannot_get_dates_or_clock_from_pit_flags(self):
        raw, target, evidence = inputs()
        result = adapt(raw, target)
        self.assertEqual(result["decision_counts"], {"MISSING_CAPTURE_EVIDENCE": 6})
        evidence["record_evidence"] = []
        result = adapt(raw, target, evidence)
        self.assertTrue(all(row["source_declared_complete"] and not row["effective_complete"] for row in result["coverage_diagnostics"]))
        self.assertIsNone(result["snapshot"]["teams"]["BOS"]["previous_7_calendar_days"]["outs"])

    def test_file_and_physical_row_hash_bindings_fail_closed(self):
        raw, target, evidence = inputs()
        with self.assertRaisesRegex(ValueError, "source-file SHA"):
            adapt(b" " + raw, target, evidence)
        evidence["record_evidence"][0]["raw_record_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "raw record hash"):
            adapt(raw, target, evidence)

    def test_identity_mismatch_is_explicit_and_downgrades_completeness(self):
        raw, target, evidence = inputs()
        evidence["record_evidence"][0]["pitcher_id"] = 999
        result = adapt(raw, target, evidence)
        self.assertIn("identity mismatch", result["row_decisions"][0]["reason"])
        self.assertIsNone(result["snapshot"]["teams"]["NYY"]["previous_3_calendar_days"]["outs"])
        evidence["team_mappings"][0]["team_code"] = "BOS"
        with self.assertRaisesRegex(ValueError, "target identity"):
            adapt(raw, target, evidence)
        raw, target, evidence = inputs()
        evidence["team_mappings"] = []
        result = adapt(raw, target, evidence)
        self.assertEqual(result["selected_relief_appearance_count"], 0)
        self.assertIsNone(result["snapshot"]["teams"]["NYY"]["previous_3_calendar_days"]["outs"])
        self.assertEqual(result["decision_counts"], {"MISSING_EXPLICIT_TEAM_IDENTITY_MAPPING": 6})

    def test_post_cutoff_source_revision_cannot_rewrite_pre_cutoff_snapshot(self):
        raw, target, evidence = inputs()
        original = adapt(raw, target, evidence)
        rows = [json.loads(line) for line in raw.splitlines()]
        rows[1]["outs_recorded"] = 99
        raw = rebind(rows, evidence)
        revised = adapt(raw, target, evidence)
        self.assertNotEqual(original["source_file_sha256"], revised["source_file_sha256"])
        self.assertEqual(original["snapshot_sha256"], revised["snapshot_sha256"])

    def test_latest_pre_cutoff_revision_is_used_once(self):
        raw, target, evidence = inputs()
        evidence["record_evidence"][1].update(response_received_at_utc="2026-10-05T19:59:00Z", persisted_at_utc="2026-10-05T20:00:00Z")
        result = adapt(raw, target, evidence)
        self.assertEqual(result["snapshot"]["teams"]["NYY"]["previous_3_calendar_days"]["outs"], 12)
        self.assertEqual(result["decision_counts"]["SUPERSEDED_PRE_CUTOFF"], 1)
        self.assertEqual(result["selected_relief_appearance_count"], 4)

    def test_tied_conflicting_versions_are_unknown_and_exact_repeats_collapse(self):
        raw, target, evidence = inputs()
        evidence["record_evidence"][1].update(response_received_at_utc=evidence["record_evidence"][0]["response_received_at_utc"], persisted_at_utc=evidence["record_evidence"][0]["persisted_at_utc"])
        result = adapt(raw, target, evidence)
        self.assertEqual(result["decision_counts"]["AMBIGUOUS_SAME_TIMESTAMP_REVISION"], 2)
        self.assertIsNone(result["snapshot"]["teams"]["NYY"]["previous_3_calendar_days"]["outs"])
        rows = [json.loads(line) for line in raw.splitlines()]
        rows[1] = deepcopy(rows[0])
        result = adapt(rebind(rows, evidence), target, evidence)
        self.assertEqual(result["decision_counts"]["DUPLICATE_CAPTURE_COLLAPSED"], 1)
        self.assertEqual(result["snapshot"]["teams"]["NYY"]["previous_3_calendar_days"]["outs"], 9)

    def test_doubleheader_prior_game_is_separate_and_target_game_never_used(self):
        raw, target, evidence = inputs()
        rows = [json.loads(line) for line in raw.splitlines()]
        current = deepcopy(rows[2]); current["source_game_id"] = target["provider_game_id"]
        current["outs_recorded"] = 99; rows.append(current)
        binding = deepcopy(evidence["record_evidence"][2]); binding.update(source_row_index=7, provider_game_id=target["provider_game_id"], game_number=2)
        evidence["record_evidence"].append(binding)
        result = adapt(rebind(rows, evidence), target, evidence)
        self.assertEqual(result["snapshot"]["teams"]["NYY"]["prior_same_day"]["outs"], 4)
        self.assertEqual(result["decision_counts"]["TARGET_GAME_EXCLUDED"], 1)

    def test_missing_time_actual_usage_and_estimated_outs_are_not_imputed(self):
        for field in ("response_received_at_utc", "persisted_at_utc", "usage_date", "usage_date_basis"):
            raw, target, evidence = inputs()
            del evidence["record_evidence"][5][field]
            result = adapt(raw, target, evidence)
            self.assertIn("MISSING_OR_INVALID", result["row_decisions"][5]["reason"])
            self.assertIsNone(result["snapshot"]["teams"]["BOS"]["previous_3_calendar_days"]["outs"])
        raw, target, evidence = inputs()
        rows = [json.loads(line) for line in raw.splitlines()]
        rows[5].pop("outs_recorded"); rows[5]["innings_pitched"] = 0.6667
        result = adapt(rebind(rows, evidence), target, evidence)
        self.assertIsNone(result["snapshot"]["teams"]["BOS"]["previous_3_calendar_days"]["outs"])
        rows[5]["estimated"] = True
        result = adapt(rebind(rows, evidence), target, evidence)
        self.assertEqual(result["row_decisions"][5]["reason"], "ESTIMATED_SCHEDULE_PROXY_NOT_OBSERVED_USAGE")

    def test_post_cutoff_capture_without_prior_version_yields_unknown(self):
        raw, target, evidence = inputs()
        evidence["record_evidence"][5].update(response_received_at_utc="2026-10-06T18:00:30Z", persisted_at_utc="2026-10-06T18:01:00Z")
        result = adapt(raw, target, evidence)
        self.assertEqual(result["row_decisions"][5]["reason"], "NOT_AVAILABLE_BEFORE_CUTOFF")
        self.assertIsNone(result["snapshot"]["teams"]["BOS"]["previous_3_calendar_days"]["outs"])

    def test_cli_offline_output_is_exclusive_and_json_duplicate_keys_rejected(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "result.json"
            args = ["--source", str(FIXTURE / "relief.synthetic.jsonl"), "--format", "phase63-jsonl", "--target", str(FIXTURE / "target.synthetic.json"), "--capture-evidence", str(FIXTURE / "capture_evidence.synthetic.json"), "--output", str(output)]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(args), 0)
                original = output.read_bytes()
                self.assertEqual(main(args), 1)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(json.loads(original)["selected_relief_appearance_count"], 4)
        raw, target, _ = inputs()
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            adapt(b'{"game_date":"a","game_date":"b"}\n', target)


if __name__ == "__main__":
    unittest.main()
