"""Characterization tests for committed prospective prediction admission demo."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.prospective_prediction_admission import main as cli_main


class ProspectivePredictionAdmissionDemoCharacterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples_dir = REPOSITORY_ROOT / "examples" / "p15b_real_schedule_admission"
        self.report_dir = REPOSITORY_ROOT / "report" / "p15b_real_schedule_admission"

    def test_committed_demo_artifacts_exist_and_contain_required_cases(self) -> None:
        results_file = self.report_dir / "results.jsonl"
        summary_file = self.report_dir / "summary.json"
        report_file = self.report_dir / "report.md"

        self.assertTrue(results_file.exists())
        self.assertTrue(summary_file.exists())
        self.assertTrue(report_file.exists())

        results = [
            json.loads(line)
            for line in results_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = json.loads(summary_file.read_text(encoding="utf-8"))

        # Require at least 2 admitted requests
        admitted = [r for r in results if r["admission_status"] == "ADMITTED"]
        self.assertGreaterEqual(len(admitted), 2)
        self.assertEqual(summary["admitted_count"], len(admitted))

        rejection_reasons = {r["reason"] for r in results if r["reason"] is not None}
        required_rejections = {
            "MISSING_SCHEDULE_CANDIDATE_MATCH",
            "INVALID_PREDICTION_TIMESTAMP_ORDER",
            "PREDICTION_NOT_BEFORE_SCHEDULED_START",
            "SCHEDULE_OBSERVATION_ID_MISMATCH",
            "SCHEDULE_NOT_PREGAME_ELIGIBLE",
            "EXACT_IDENTITY_MISMATCH",
        }
        for expected_reason in required_rejections:
            self.assertIn(expected_reason, rejection_reasons)

        # Verify claims in summary.json
        claims = summary["claims"]
        self.assertFalse(claims["provider_called"])
        self.assertFalse(claims["db_written"])
        self.assertFalse(claims["legacy_rows_admitted"])
        self.assertFalse(claims["deployed"])
        self.assertFalse(claims["betting_claim"])

    def test_rerunning_cli_reproduces_committed_demo_artifacts_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            argv = [
                "--prediction-requests", str(self.examples_dir / "requests.jsonl"),
                "--raw-schedule-payloads", str(self.examples_dir / "raw_schedule_payloads.jsonl"),
                "--participant-mapping-catalog", str(self.examples_dir / "participant_mapping_catalog.json"),
                "--authority-catalog", str(self.examples_dir / "authority_catalog.json"),
                "--schedule-as-of-utc", "2026-04-05T12:00:00Z",
                "--output-dir", str(out_dir),
            ]
            exit_code = cli_main(argv)
            self.assertEqual(exit_code, 0)

            for artifact_name in ("results.jsonl", "summary.json", "report.md"):
                generated_bytes = (out_dir / artifact_name).read_bytes()
                committed_bytes = (self.report_dir / artifact_name).read_bytes()
                self.assertEqual(
                    generated_bytes,
                    committed_bytes,
                    f"Committed artifact {artifact_name} differs from freshly generated output",
                )


if __name__ == "__main__":
    unittest.main()
