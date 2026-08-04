"""CLI integration tests for final result attachment."""

import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.final_result_attachment import main as cli_main


P15C_SNAPSHOT_PATH = (
    REPOSITORY_ROOT
    / "report"
    / "p15c_admitted_prediction_observation_snapshot"
    / "admitted_observations.jsonl"
)
P15C_SUMMARY_PATH = (
    REPOSITORY_ROOT
    / "report"
    / "p15c_admitted_prediction_observation_snapshot"
    / "summary.json"
)
EXAMPLE_RESULTS_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "p16a_final_result_attachment"
    / "final_results.jsonl"
)


class FinalResultAttachmentCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="p16a_cli_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_successful_complete_attachment_cli(self) -> None:
        out_dir = self.temp_dir / "out"
        argv = [
            "--prediction-snapshot", str(P15C_SNAPSHOT_PATH),
            "--prediction-summary", str(P15C_SUMMARY_PATH),
            "--final-results", str(EXAMPLE_RESULTS_PATH),
            "--output-dir", str(out_dir),
        ]
        exit_code = cli_main(argv)
        self.assertEqual(exit_code, 0)
        self.assertTrue((out_dir / "attachments.jsonl").exists())
        self.assertTrue((out_dir / "summary.json").exists())
        self.assertTrue((out_dir / "report.md").exists())

    def test_deterministic_repeated_execution(self) -> None:
        out_dir1 = self.temp_dir / "out1"
        out_dir2 = self.temp_dir / "out2"

        argv1 = [
            "--prediction-snapshot", str(P15C_SNAPSHOT_PATH),
            "--prediction-summary", str(P15C_SUMMARY_PATH),
            "--final-results", str(EXAMPLE_RESULTS_PATH),
            "--output-dir", str(out_dir1),
        ]
        argv2 = [
            "--prediction-snapshot", str(P15C_SNAPSHOT_PATH),
            "--prediction-summary", str(P15C_SUMMARY_PATH),
            "--final-results", str(EXAMPLE_RESULTS_PATH),
            "--output-dir", str(out_dir2),
        ]

        self.assertEqual(cli_main(argv1), 0)
        self.assertEqual(cli_main(argv2), 0)

        for filename in ["attachments.jsonl", "summary.json", "report.md"]:
            b1 = (out_dir1 / filename).read_bytes()
            b2 = (out_dir2 / filename).read_bytes()
            self.assertEqual(b1, b2, f"Mismatch in {filename}")

    def test_malformed_json_fails_structural(self) -> None:
        malformed_file = self.temp_dir / "malformed.jsonl"
        malformed_file.write_text("{bad json\n", encoding="utf-8")

        out_dir = self.temp_dir / "out_malformed"
        argv = [
            "--prediction-snapshot", str(P15C_SNAPSHOT_PATH),
            "--prediction-summary", str(P15C_SUMMARY_PATH),
            "--final-results", str(malformed_file),
            "--output-dir", str(out_dir),
        ]

        stderr = io.StringIO()
        sys.stderr = stderr
        try:
            exit_code = cli_main(argv)
        finally:
            sys.stderr = sys.__stderr__

        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR:", stderr.getvalue())
        self.assertFalse(out_dir.exists())

    def test_duplicate_json_key_fails_structural(self) -> None:
        dup_key_file = self.temp_dir / "dup_key.jsonl"
        dup_key_file.write_text(
            '{"source_result_id":"R1","provider_namespace":"P","provider_game_id":"888001",'
            '"game_number":1,"status":"FINAL","result_observed_at_utc":"2026-04-05T22:00:00Z",'
            '"home_score":5,"away_score":3,"home_score":6}\n',
            encoding="utf-8",
        )

        out_dir = self.temp_dir / "out_dup_key"
        argv = [
            "--prediction-snapshot", str(P15C_SNAPSHOT_PATH),
            "--prediction-summary", str(P15C_SUMMARY_PATH),
            "--final-results", str(dup_key_file),
            "--output-dir", str(out_dir),
        ]

        stderr = io.StringIO()
        sys.stderr = stderr
        try:
            exit_code = cli_main(argv)
        finally:
            sys.stderr = sys.__stderr__

        self.assertEqual(exit_code, 1)
        self.assertIn("Duplicate JSON key", stderr.getvalue())
        self.assertFalse(out_dir.exists())

    def test_duplicate_result_identity_ambiguous_fails_structural(self) -> None:
        ambig_file = self.temp_dir / "ambig.jsonl"
        row1 = (
            '{"source_result_id":"R1","provider_namespace":"MLB_STATS_API",'
            '"provider_game_id":"888001","game_number":1,"status":"FINAL",'
            '"result_observed_at_utc":"2026-04-05T22:00:00Z","home_score":5,"away_score":3}\n'
        )
        row2 = (
            '{"source_result_id":"R2","provider_namespace":"MLB_STATS_API",'
            '"provider_game_id":"888001","game_number":1,"status":"FINAL",'
            '"result_observed_at_utc":"2026-04-05T23:00:00Z","home_score":4,"away_score":2}\n'
        )
        ambig_file.write_text(row1 + row2, encoding="utf-8")

        out_dir = self.temp_dir / "out_ambig"
        argv = [
            "--prediction-snapshot", str(P15C_SNAPSHOT_PATH),
            "--prediction-summary", str(P15C_SUMMARY_PATH),
            "--final-results", str(ambig_file),
            "--output-dir", str(out_dir),
        ]

        stderr = io.StringIO()
        sys.stderr = stderr
        try:
            exit_code = cli_main(argv)
        finally:
            sys.stderr = sys.__stderr__

        self.assertEqual(exit_code, 1)
        self.assertIn("AMBIGUOUS_FINAL_RESULT_OBSERVATION", stderr.getvalue())
        self.assertFalse(out_dir.exists())

    def test_tied_final_fails_structural(self) -> None:
        tied_file = self.temp_dir / "tied.jsonl"
        row = (
            '{"source_result_id":"R1","provider_namespace":"MLB_STATS_API",'
            '"provider_game_id":"888001","game_number":1,"status":"FINAL",'
            '"result_observed_at_utc":"2026-04-05T22:00:00Z","home_score":4,"away_score":4}\n'
        )
        tied_file.write_text(row, encoding="utf-8")

        out_dir = self.temp_dir / "out_tied"
        argv = [
            "--prediction-snapshot", str(P15C_SNAPSHOT_PATH),
            "--prediction-summary", str(P15C_SUMMARY_PATH),
            "--final-results", str(tied_file),
            "--output-dir", str(out_dir),
        ]

        stderr = io.StringIO()
        sys.stderr = stderr
        try:
            exit_code = cli_main(argv)
        finally:
            sys.stderr = sys.__stderr__

        self.assertEqual(exit_code, 1)
        self.assertIn("TIED_FINAL_RESULT_UNSUPPORTED", stderr.getvalue())
        self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
