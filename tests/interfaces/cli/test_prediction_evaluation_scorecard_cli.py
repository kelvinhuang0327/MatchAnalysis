"""CLI integration tests for prediction evaluation scorecard generation."""

from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.prediction_evaluation_scorecard import main


class TestPredictionEvaluationScorecardCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.p16a_attachments = Path("report/p16a_final_result_attachment/attachments.jsonl")
        self.p16a_summary = Path("report/p16a_final_result_attachment/summary.json")
        self.p15c_snapshot = Path("report/p15c_admitted_prediction_observation_snapshot/admitted_observations.jsonl")
        self.p15c_summary = Path("report/p15c_admitted_prediction_observation_snapshot/summary.json")

    def test_cli_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "output"
            argv = [
                "--attachments",
                str(self.p16a_attachments),
                "--attachment-summary",
                str(self.p16a_summary),
                "--snapshot",
                str(self.p15c_snapshot),
                "--snapshot-summary",
                str(self.p15c_summary),
                "--output-dir",
                str(out_dir),
            ]
            exit_code = main(argv)
            self.assertEqual(exit_code, 0)

            self.assertTrue((out_dir / "evaluations.jsonl").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "report.md").exists())

    def test_cli_repeated_execution_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir1 = Path(tmp_dir) / "output1"
            out_dir2 = Path(tmp_dir) / "output2"

            argv1 = [
                "--attachments",
                str(self.p16a_attachments),
                "--attachment-summary",
                str(self.p16a_summary),
                "--snapshot",
                str(self.p15c_snapshot),
                "--snapshot-summary",
                str(self.p15c_summary),
                "--output-dir",
                str(out_dir1),
            ]
            argv2 = [
                "--attachments",
                str(self.p16a_attachments),
                "--attachment-summary",
                str(self.p16a_summary),
                "--snapshot",
                str(self.p15c_snapshot),
                "--snapshot-summary",
                str(self.p15c_summary),
                "--output-dir",
                str(out_dir2),
            ]
            code1 = main(argv1)
            code2 = main(argv2)
            self.assertEqual(code1, 0)
            self.assertEqual(code2, 0)

            self.assertEqual(
                (out_dir1 / "evaluations.jsonl").read_bytes(),
                (out_dir2 / "evaluations.jsonl").read_bytes(),
            )
            self.assertEqual(
                (out_dir1 / "summary.json").read_bytes(),
                (out_dir2 / "summary.json").read_bytes(),
            )
            self.assertEqual(
                (out_dir1 / "report.md").read_bytes(),
                (out_dir2 / "report.md").read_bytes(),
            )

    def test_cli_missing_attachments_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "output"
            argv = [
                "--attachments",
                str(Path(tmp_dir) / "non_existent.jsonl"),
                "--attachment-summary",
                str(self.p16a_summary),
                "--output-dir",
                str(out_dir),
            ]
            exit_code = main(argv)
            self.assertEqual(exit_code, 1)

    def test_cli_malformed_json_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_att = Path(tmp_dir) / "bad_attachments.jsonl"
            bad_att.write_text("{bad json}\n", encoding="utf-8")
            out_dir = Path(tmp_dir) / "output"
            argv = [
                "--attachments",
                str(bad_att),
                "--attachment-summary",
                str(self.p16a_summary),
                "--snapshot",
                str(self.p15c_snapshot),
                "--snapshot-summary",
                str(self.p15c_summary),
                "--output-dir",
                str(out_dir),
            ]
            exit_code = main(argv)
            self.assertEqual(exit_code, 1)
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
