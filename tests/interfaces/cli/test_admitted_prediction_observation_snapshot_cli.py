"""Integration tests for admitted prediction observation snapshot CLI."""

from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.admitted_prediction_observation_snapshot import main as cli_main


class AdmittedPredictionObservationSnapshotCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results_path = (
            REPOSITORY_ROOT / "report" / "p15b_real_schedule_admission" / "results.jsonl"
        )
        self.summary_path = (
            REPOSITORY_ROOT / "report" / "p15b_real_schedule_admission" / "summary.json"
        )

    def test_cli_produces_all_three_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir)
            exit_code = cli_main([
                "--admission-results", str(self.results_path),
                "--admission-summary", str(self.summary_path),
                "--output-dir", str(out),
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((out / "admitted_observations.jsonl").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "report.md").exists())

    def test_cli_execution_is_byte_identical_on_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            out1 = Path(tmp1)
            out2 = Path(tmp2)

            exit_code1 = cli_main([
                "--admission-results", str(self.results_path),
                "--admission-summary", str(self.summary_path),
                "--output-dir", str(out1),
            ])
            self.assertEqual(exit_code1, 0)

            exit_code2 = cli_main([
                "--admission-results", str(self.results_path),
                "--admission-summary", str(self.summary_path),
                "--output-dir", str(out2),
            ])
            self.assertEqual(exit_code2, 0)

            for artifact_name in (
                "admitted_observations.jsonl",
                "summary.json",
                "report.md",
            ):
                file1 = out1 / artifact_name
                file2 = out2 / artifact_name
                self.assertTrue(file1.exists())
                self.assertTrue(file2.exists())
                self.assertEqual(
                    file1.read_bytes(),
                    file2.read_bytes(),
                    f"Artifact {artifact_name} differs between replay runs",
                )

    def test_cli_returns_nonzero_on_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir)
            bad_results = out / "bad_results.jsonl"
            bad_results.write_text("not valid json\n", encoding="utf-8")

            exit_code = cli_main([
                "--admission-results", str(bad_results),
                "--admission-summary", str(self.summary_path),
                "--output-dir", str(out / "output"),
            ])
            self.assertNotEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
