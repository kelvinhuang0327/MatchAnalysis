"""Synthetic workload, availability, identity, duplicate, and CLI regressions."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from match_analysis.application.use_cases.build_bullpen_research_snapshot import build_bullpen_research_snapshot
from match_analysis.baseball.domain.bullpen_research_snapshot import BullpenResearchSnapshot
from match_analysis.interfaces.cli.build_bullpen_research_snapshot import main


def fixture():
    return json.loads((ROOT / "examples/bullpen_research_v1/input.synthetic.json").read_text())


def projection(bundle):
    return build_bullpen_research_snapshot(bundle).to_projection()


class BullpenResearchSnapshotTests(unittest.TestCase):
    def test_exact_loads_same_day_and_actual_pitcher_intersection(self):
        result = projection(fixture())
        home, away = result["teams"]["NYY"], result["teams"]["BOS"]
        self.assertEqual(home["previous_3_calendar_days"]["outs"], 11)
        self.assertEqual(home["previous_7_calendar_days"]["outs"], 16)
        self.assertEqual(home["prior_same_day"]["outs"], 4)
        self.assertEqual(home["previous_3_calendar_days"]["relief_appearances"], 3)
        self.assertEqual(home["relievers_worked_both_previous_days"], 1)
        self.assertEqual(result["exact_duplicates_collapsed"], 1)
        self.assertEqual(len(result["accepted_relief_evidence"]), 6)
        self.assertEqual(home["recent_observed_relief_pitchers"][0]["full_calendar_rest_days"], 0)
        self.assertEqual(home["recent_observed_relief_pitchers"][2]["full_calendar_rest_days"], 3)
        self.assertIsNone(away["previous_3_calendar_days"]["outs"])
        self.assertEqual(away["previous_3_calendar_days"]["status"], "REJECTED_OR_UNAVAILABLE_RECORD_IN_WINDOW")
        self.assertEqual({row["reason"] for row in result["excluded_records"]},
                         {"TARGET_GAME_EXCLUDED", "STARTER_NOT_BULLPEN", "NOT_AVAILABLE_BEFORE_CUTOFF"})

    def test_missing_optional_stats_remain_unknown_without_neutral_imputation(self):
        home = projection(fixture())["teams"]["NYY"]
        self.assertIsNone(home["previous_7_calendar_days"]["pitches"])
        self.assertEqual(home["previous_7_calendar_days"]["pitches_status"], "MISSING_PITCH_COUNT")
        self.assertIsNone(home["era_previous_14_calendar_days"])
        self.assertEqual(home["era_status"], "MISSING_EARNED_RUNS")
        self.assertIsNone(home["high_leverage_appearance_fraction_previous_7_days"])

    def test_zero_requires_explicit_complete_coverage_and_no_denominator_is_unknown(self):
        bundle = fixture()
        bundle["appearances"] = []
        home = projection(bundle)["teams"]["NYY"]
        self.assertEqual(home["previous_3_calendar_days"]["outs"], 0)
        self.assertEqual(home["relievers_worked_both_previous_days"], 0)
        self.assertIsNone(home["era_previous_14_calendar_days"])
        bundle["coverage"] = []
        home = projection(bundle)["teams"]["NYY"]
        self.assertIsNone(home["previous_3_calendar_days"]["outs"])
        self.assertEqual(home["previous_3_calendar_days"]["status"], "NO_COMPLETE_COVERAGE_DECLARATION")

    def test_cutoff_boundary_and_absent_availability_are_never_backfilled(self):
        bundle = fixture()
        bundle["appearances"] = [bundle["appearances"][7]]
        bundle["appearances"][0]["available_at_utc"] = bundle["target"]["as_of_utc"]
        self.assertEqual(projection(bundle)["teams"]["BOS"]["previous_3_calendar_days"]["outs"], 6)
        del bundle["appearances"][0]["available_at_utc"]
        result = projection(bundle)
        self.assertIsNone(result["teams"]["BOS"]["previous_3_calendar_days"]["outs"])
        self.assertIn("MISSING_OR_INVALID_HISTORY_EVIDENCE", result["excluded_records"][0]["reason"])
        bundle = fixture()
        bundle["appearances"] = [bundle["appearances"][4]]
        bundle["appearances"][0]["ended_at_utc"] = bundle["target"]["as_of_utc"]
        bundle["appearances"][0]["available_at_utc"] = bundle["target"]["as_of_utc"]
        self.assertIsNone(projection(bundle)["teams"]["NYY"]["prior_same_day"]["outs"])

    def test_conflicting_duplicates_and_game_identity_fail_closed(self):
        for field, value in [("outs", 99), ("team_id", 111), ("game_number", 2)]:
            bundle = fixture()
            bundle["appearances"][-1][field] = value
            with self.assertRaises(ValueError):
                projection(bundle)
        bundle = fixture()
        bundle["appearances"][5]["game_number"] = 2
        with self.assertRaisesRegex(ValueError, "conflicting date/game number"):
            projection(bundle)

    def test_identity_and_pregame_target_mismatches_are_errors(self):
        for field, value in [("team_code", "BOS"), ("pitcher_id", True),
                             ("provider_namespace", "OTHER_PROVIDER")]:
            bundle = fixture()
            bundle["appearances"][0][field] = value
            with self.assertRaises(ValueError):
                projection(bundle)
        bundle = fixture()
        bundle["target"]["as_of_utc"] = bundle["target"]["scheduled_start_utc"]
        with self.assertRaisesRegex(ValueError, "strictly before"):
            projection(bundle)
        bundle["target"]["as_of_utc"] = "2026-10-06T18:00:00"
        with self.assertRaises(ValueError):
            projection(bundle)

    def test_unexpected_target_or_history_outcome_fields_are_rejected(self):
        for where, field in [("target", "home_win"), ("appearance", "final_score")]:
            bundle = fixture()
            record = bundle["target"] if where == "target" else bundle["appearances"][0]
            record[field] = 1
            with self.assertRaisesRegex(ValueError, "unexpected fields"):
                projection(bundle)

    def test_windows_era_no_saturation_and_suspended_usage_day(self):
        bundle = fixture()
        bundle["appearances"] = [deepcopy(bundle["appearances"][0])]
        row = bundle["appearances"][0]
        row.update(official_date="2026-09-01", usage_date="2026-10-05", outs=30, earned_runs=3)
        home = projection(bundle)["teams"]["NYY"]
        self.assertEqual(home["previous_3_calendar_days"]["outs"], 30)
        self.assertEqual(home["era_previous_14_calendar_days"], "2.7")
        self.assertEqual(home["high_leverage_appearance_fraction_previous_7_days"], "1")
        row["usage_date"] = "2026-09-22"
        home = projection(bundle)["teams"]["NYY"]
        self.assertEqual(home["previous_7_calendar_days"]["outs"], 0)
        self.assertEqual(home["era_previous_14_calendar_days"], "2.7")
        row["usage_date"] = "2026-09-21"
        self.assertIsNone(projection(bundle)["teams"]["NYY"]["era_previous_14_calendar_days"])
        del row["usage_date"]
        self.assertIsNone(projection(bundle)["teams"]["NYY"]["previous_3_calendar_days"]["outs"])

    def test_back_to_back_counts_pitchers_and_does_not_count_team_game_days(self):
        bundle = fixture()
        bundle["appearances"] = bundle["appearances"][:2]
        bundle["appearances"][1]["pitcher_id"] = 999
        self.assertEqual(projection(bundle)["teams"]["NYY"]["relievers_worked_both_previous_days"], 0)

    def test_incomplete_future_and_partial_coverage(self):
        for field, value in [("complete", False), ("start_date", "2026-10-04"),
                             ("through_utc", "2026-10-06T17:00:00Z")]:
            bundle = fixture()
            bundle["coverage"][0][field] = value
            self.assertIsNone(projection(bundle)["teams"]["NYY"]["previous_3_calendar_days"]["outs"])
        bundle = fixture()
        bundle["coverage"][0]["available_at_utc"] = "2026-10-06T18:01:00Z"
        with self.assertRaisesRegex(ValueError, "coverage unavailable"):
            projection(bundle)

    def test_snapshot_frozen_and_full_evidence_fingerprint(self):
        bundle = fixture()
        first = build_bullpen_research_snapshot(bundle)
        self.assertEqual(first.fingerprint(), build_bullpen_research_snapshot(bundle).fingerprint())
        with self.assertRaises(FrozenInstanceError):
            first.payload_bytes = b"altered"
        detached = first.to_projection()
        detached["teams"]["NYY"]["previous_3_calendar_days"]["outs"] = 999
        self.assertEqual(first.to_projection()["teams"]["NYY"]["previous_3_calendar_days"]["outs"], 11)
        bundle["appearances"][1]["source_fingerprint"] = "a" * 64
        self.assertNotEqual(first.fingerprint(), build_bullpen_research_snapshot(bundle).fingerprint())

    def test_loaded_snapshot_rechecks_timestamps_instead_of_trusting_safe_flags(self):
        record = projection(fixture())
        record["accepted_relief_evidence"][0]["available_at_utc"] = "2026-10-06T18:01:00Z"
        with self.assertRaisesRegex(ValueError, "unavailable or post-cutoff"):
            BullpenResearchSnapshot.from_projection(record)
        record = projection(fixture())
        record["accepted_relief_evidence"][0]["usage_date"] = "2026-10-07"
        with self.assertRaisesRegex(ValueError, "future usage date"):
            BullpenResearchSnapshot.from_projection(record)
        record = projection(fixture())
        record["coverage_evidence"][0]["team_code"] = "OTHER_TEAM"
        with self.assertRaisesRegex(ValueError, "coverage team identity"):
            BullpenResearchSnapshot.from_projection(record)
        bundle = fixture()
        bundle["appearances"][8]["game_number"] = 1
        with self.assertRaisesRegex(ValueError, "target provider game identity conflict"):
            projection(bundle)

    def test_offline_cli_preserves_existing_output_and_records_input_hash(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            source, output = path / "input.json", path / "snapshot.json"
            source.write_text(json.dumps(fixture()))
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(["--input", str(source), "--output", str(output)]), 0)
                original = output.read_bytes()
                self.assertEqual(main(["--input", str(source), "--output", str(output)]), 1)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(json.loads(original)["input_file_sha256"], sha256(source.read_bytes()).hexdigest())
            source.write_text('{"schema_version": "a", "schema_version": "b"}')
            with redirect_stderr(StringIO()):
                self.assertEqual(main(["--input", str(source), "--output", str(path / "bad.json")]), 1)
            self.assertFalse((path / "bad.json").exists())


if __name__ == "__main__":
    unittest.main()
