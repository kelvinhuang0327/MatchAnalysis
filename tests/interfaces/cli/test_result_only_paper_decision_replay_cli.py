"""CLI tests for the P18A result-only paper-decision replay."""

from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.result_only_paper_decision_replay import main


class ResultOnlyPaperDecisionReplayCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = Path(
            "report/p15c_admitted_prediction_observation_snapshot/admitted_observations.jsonl"
        )
        self.summary = Path(
            "report/p15c_admitted_prediction_observation_snapshot/summary.json"
        )
        self.results = Path("examples/p16a_final_result_attachment/final_results.jsonl")

    def _args(self, output_dir: Path, results: Path | None = None) -> list[str]:
        return [
            "--prediction-snapshot", str(self.snapshot),
            "--prediction-summary", str(self.summary),
            "--final-results", str(results or self.results),
            "--output-dir", str(output_dir),
        ]

    def test_cli_success_and_repeated_output_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            self.assertEqual(main(self._args(first)), 0)
            self.assertEqual(main(self._args(second)), 0)
            for filename in ("decisions.jsonl", "settlements.jsonl", "summary.json", "report.md"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_cli_missing_result_file_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            code = main(self._args(output_dir, Path(temp_dir) / "missing.jsonl"))
            self.assertEqual(code, 1)
            self.assertFalse(output_dir.exists())

    def test_cli_malformed_final_results_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_results = Path(temp_dir) / "bad.jsonl"
            bad_results.write_text("{bad json}\n", encoding="utf-8")
            output_dir = Path(temp_dir) / "output"
            code = main(self._args(output_dir, bad_results))
            self.assertEqual(code, 1)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
