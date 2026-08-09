"""CLI tests for the P20B historical feedback replay."""

import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.historical_feedback_replay import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
P20A_ROOT = REPOSITORY_ROOT / "report/p20a_p13_walk_forward_reconstruction"
RESULT_ROOT = REPOSITORY_ROOT / "data/fixtures/p20b_historical_results"


class TestHistoricalFeedbackReplayCLI(unittest.TestCase):
    def _run_cli(self, output_dir: Path) -> int:
        return main(
            [
                "--p20a-predictions",
                str(P20A_ROOT / "predictions.jsonl"),
                "--p20a-reconstruction",
                str(P20A_ROOT / "reconstruction.json"),
                "--p20a-summary",
                str(P20A_ROOT / "summary.json"),
                "--p20a-fold",
                str(P20A_ROOT / "fold.json"),
                "--historical-results",
                str(RESULT_ROOT / "final_results.jsonl"),
                "--historical-provenance",
                str(RESULT_ROOT / "provenance.json"),
                "--output-dir",
                str(output_dir),
            ]
        )

    def test_cli_writes_explicit_non_synthetic_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output"
            self.assertEqual(self._run_cli(output_dir), 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"feedback.jsonl", "summary.json", "report.md"},
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["claims"]["synthetic_results"])
            self.assertTrue(summary["claims"]["non_synthetic"])
            self.assertEqual(summary["replay_game_ids"], [
                "2025-06-01_ATL_BOS",
                "2025-06-01_TEX_STL",
            ])

    def test_cli_repeated_execution_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_one = Path(tmp_dir) / "one"
            output_two = Path(tmp_dir) / "two"
            self.assertEqual(self._run_cli(output_one), 0)
            self.assertEqual(self._run_cli(output_two), 0)
            for name in ("feedback.jsonl", "summary.json", "report.md"):
                self.assertEqual(
                    (output_one / name).read_bytes(),
                    (output_two / name).read_bytes(),
                    name,
                )

    def test_cli_missing_input_returns_error_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output"
            args = [
                "--p20a-predictions",
                str(Path(tmp_dir) / "missing.jsonl"),
                "--p20a-reconstruction",
                str(P20A_ROOT / "reconstruction.json"),
                "--p20a-summary",
                str(P20A_ROOT / "summary.json"),
                "--p20a-fold",
                str(P20A_ROOT / "fold.json"),
                "--historical-results",
                str(RESULT_ROOT / "final_results.jsonl"),
                "--historical-provenance",
                str(RESULT_ROOT / "provenance.json"),
                "--output-dir",
                str(output_dir),
            ]
            self.assertEqual(main(args), 1)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
