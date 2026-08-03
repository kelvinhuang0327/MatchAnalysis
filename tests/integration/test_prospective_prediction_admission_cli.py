"""Integration tests for prospective prediction admission CLI."""

from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.prospective_prediction_admission import main as cli_main


class ProspectivePredictionAdmissionCliIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples_dir = REPOSITORY_ROOT / "examples" / "p15b_real_schedule_admission"
        self.requests_path = self.examples_dir / "requests.jsonl"
        self.payloads_path = self.examples_dir / "raw_schedule_payloads.jsonl"
        self.mappings_path = self.examples_dir / "participant_mapping_catalog.json"
        self.authority_path = self.examples_dir / "authority_catalog.json"
        self.as_of_str = "2026-04-05T12:00:00Z"

    def test_cli_execution_produces_artifacts_and_is_byte_identical_on_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir1, tempfile.TemporaryDirectory() as tmp_dir2:
            out1 = Path(tmp_dir1)
            out2 = Path(tmp_dir2)

            argv1 = [
                "--prediction-requests", str(self.requests_path),
                "--raw-schedule-payloads", str(self.payloads_path),
                "--participant-mapping-catalog", str(self.mappings_path),
                "--authority-catalog", str(self.authority_path),
                "--schedule-as-of-utc", self.as_of_str,
                "--output-dir", str(out1),
            ]
            exit_code1 = cli_main(argv1)
            self.assertEqual(exit_code1, 0)

            argv2 = [
                "--prediction-requests", str(self.requests_path),
                "--raw-schedule-payloads", str(self.payloads_path),
                "--participant-mapping-catalog", str(self.mappings_path),
                "--authority-catalog", str(self.authority_path),
                "--schedule-as-of-utc", self.as_of_str,
                "--output-dir", str(out2),
            ]
            exit_code2 = cli_main(argv2)
            self.assertEqual(exit_code2, 0)

            for artifact_name in ("results.jsonl", "summary.json", "report.md"):
                file1 = out1 / artifact_name
                file2 = out2 / artifact_name
                self.assertTrue(file1.exists())
                self.assertTrue(file2.exists())
                self.assertEqual(
                    file1.read_bytes(),
                    file2.read_bytes(),
                    f"Artifact {artifact_name} differs between replay runs",
                )


if __name__ == "__main__":
    unittest.main()
