"""CLI coverage for the P25A offline feedback loop."""

import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.settle_paper_moneyline_batch import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class SettlePaperMoneylineBatchCLITests(unittest.TestCase):
    def test_cli_writes_only_the_four_p25a_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "p25a"
            exit_code = main(
                [
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "settled_predictions.jsonl",
                    "evaluations.jsonl",
                    "feedback_ledger.jsonl",
                    "summary.json",
                },
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["prediction_count"], 79)
            self.assertEqual(summary["settled_prediction_count"], 79)
            self.assertEqual(summary["evaluation_count"], 79)
            self.assertEqual(summary["feedback_row_count"], 79)
            self.assertTrue(summary["claims"]["offline_settlement"])
            self.assertFalse(summary["claims"]["real_betting_recommendation"])


if __name__ == "__main__":
    unittest.main()
