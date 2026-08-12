"""CLI acceptance tests for P37A rolling walk-forward Moneyline OOS."""

import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.rolling_moneyline_oos import main


ROOT = Path(__file__).resolve().parents[3]


class RollingMoneylineOOSCLITests(unittest.TestCase):
    def test_cli_writes_exact_five_p37a_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "p37a_rolling_walk_forward_oos"
            self.assertEqual(
                main(
                    [
                        "--repository-root",
                        str(ROOT),
                        "--output-dir",
                        str(output),
                        "--fit-runtime",
                        "/usr/bin/python3",
                    ]
                ),
                0,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "model_artifacts.json",
                    "comparisons.jsonl",
                    "per_window_summary.json",
                    "summary.json",
                    "report.md",
                },
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["task_id"], "P37A")
            self.assertEqual(summary["aggregate"]["evaluable_row_count"], 65)
            self.assertEqual(summary["comparison"]["conclusion"], "MIXED_OR_INCONCLUSIVE")
            self.assertTrue(summary["deterministic_rerun_verified"])
            self.assertFalse(summary["claims"]["model_promoted"])
            report = (output / "report.md").read_text()
            self.assertIn("No model was promoted", report)
            self.assertIn("MIXED_OR_INCONCLUSIVE", report)


if __name__ == "__main__":
    unittest.main()
