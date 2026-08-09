"""CLI tests for the P21A non-synthetic learning-candidate gate."""

import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.prediction_learning_candidate_gate import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
P20B_ROOT = REPOSITORY_ROOT / "report/p20b_first_non_synthetic_historical_feedback"


class PredictionLearningCandidateGateCLITests(unittest.TestCase):
    def _args(self, output_dir: Path) -> list[str]:
        return [
            "--feedback-ledger",
            str(P20B_ROOT / "feedback.jsonl"),
            "--feedback-summary",
            str(P20B_ROOT / "summary.json"),
            "--output-dir",
            str(output_dir),
        ]

    def test_cli_exports_only_expected_artifacts_and_positive_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            self.assertEqual(main(self._args(output_dir)), 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"assessments.jsonl", "learning_candidates.jsonl", "summary.json"},
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["source_feedback_row_count"], 4)
            self.assertEqual(summary["eligible_count"], 4)
            self.assertEqual(summary["excluded_count"], 0)
            self.assertTrue(summary["claims"]["sample_limited"])
            self.assertFalse(summary["claims"]["training_authorized"])

    def test_repeated_cli_runs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            self.assertEqual(main(self._args(first)), 0)
            self.assertEqual(main(self._args(second)), 0)
            for filename in ("assessments.jsonl", "learning_candidates.jsonl", "summary.json"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_missing_source_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            args = self._args(output_dir)
            args[1] = str(Path(temp_dir) / "missing.jsonl")
            self.assertEqual(main(args), 1)
            self.assertFalse(output_dir.exists())

    def test_malformed_source_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            malformed = root / "malformed.jsonl"
            malformed.write_text("{not-json}\n", encoding="utf-8")
            output_dir = root / "output"
            args = self._args(output_dir)
            args[1] = str(malformed)
            self.assertEqual(main(args), 1)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
