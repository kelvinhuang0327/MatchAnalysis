"""P38A CLI artifact and output acceptance tests."""

import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.p38a_probability_calibration import main


ROOT = Path(__file__).resolve().parents[3]


class P38AProbabilityCalibrationCLITests(unittest.TestCase):
    def test_cli_writes_exact_five_p38a_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "p38a_rolling_probability_calibration"
            self.assertEqual(
                main(
                    [
                        "--repository-root",
                        str(ROOT),
                        "--output-dir",
                        str(output),
                    ]
                ),
                0,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "calibration_artifacts.json",
                    "comparisons.jsonl",
                    "per_window_summary.json",
                    "summary.json",
                    "report.md",
                },
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["task_id"], "P38A")
            self.assertEqual(summary["aggregate"]["evaluable_row_count"], 42)
            self.assertEqual(summary["aggregate"]["excluded_row_count"], 10)
            self.assertEqual(
                summary["comparison"]["conclusion"],
                "CALIBRATION_NOT_IMPROVED",
            )
            self.assertTrue(summary["deterministic_rerun_verified"])
            self.assertFalse(summary["claims"]["model_promoted"])
            report = (output / "report.md").read_text()
            self.assertIn("No model was promoted", report)
            self.assertIn("CALIBRATION_NOT_IMPROVED", report)


if __name__ == "__main__":
    unittest.main()
