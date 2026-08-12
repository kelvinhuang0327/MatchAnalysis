"""CLI acceptance tests for the P36A offline retraining baseline."""

from pathlib import Path
import json
import tempfile
import unittest

from match_analysis.interfaces.cli.offline_moneyline_retraining_baseline import main


ROOT = Path(__file__).resolve().parents[3]


class OfflineMoneylineRetrainingBaselineCliTests(unittest.TestCase):
    def test_cli_writes_only_the_four_p36a_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "p36a"
            exit_code = main(
                [
                    "--repository-root",
                    str(ROOT),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"model_artifact.json", "comparisons.jsonl", "summary.json", "report.md"},
            )
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["task_id"], "P36A")
            self.assertEqual(summary["comparison_verdict"], "CHALLENGER_BETTER")
            self.assertTrue(summary["deterministic_rerun_verified"])
            self.assertFalse(summary["claims"]["model_promoted"])


if __name__ == "__main__":
    unittest.main()
