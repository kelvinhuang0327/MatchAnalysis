"""Offline CLI replay tests for P30A."""

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class RunMoneylinePaperAnalysisCliTests(unittest.TestCase):
    def test_offline_replays_are_byte_identical(self) -> None:
        env = {
            **os.environ,
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            output_dirs = []
            for output_root in (Path(first), Path(second)):
                output_dir = output_root / "report"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "match_analysis.interfaces.cli.run_moneyline_paper_analysis",
                        "--offline",
                        "--output-dir",
                        str(output_dir),
                    ],
                    cwd=REPOSITORY_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    "raw=16 edge_available=9 feature_unavailable=7 price_unavailable=0",
                    result.stdout,
                )
                self.assertIn("artifacts=2 offline=True", result.stdout)
                output_dirs.append(output_dir)
            for name in ("analysis.jsonl", "summary.json"):
                self.assertEqual(
                    (output_dirs[0] / name).read_bytes(),
                    (output_dirs[1] / name).read_bytes(),
                )
            self.assertEqual(
                {path.name for path in output_dirs[0].iterdir()},
                {"analysis.jsonl", "summary.json"},
            )

    def test_precohort_runs_parity_then_frozen_generalization(self) -> None:
        env = {
            **os.environ,
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "report"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "match_analysis.interfaces.cli.run_moneyline_paper_analysis",
                    "--offline",
                    "--pre-cohort",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPOSITORY_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("p30a_parity=PASS", result.stdout)
            self.assertIn("window=2026-05-11..2026-05-17", result.stdout)
            self.assertIn("overlap=0 fresh_runs=2", result.stdout)
            self.assertIn(
                "raw=92 edge_available=0 feature_unavailable=92 price_unavailable=0",
                result.stdout,
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["run_id"], "e663bcc9758ba2c179beca72bb7fec29202838a38c45b24a53644d21c4e84f01")
            self.assertEqual(summary["p31a_generalization_window"]["window_length"], 7)
            self.assertEqual(summary["p31a_generalization_window"]["selected_dates"], [
                "2026-05-11",
                "2026-05-12",
                "2026-05-13",
                "2026-05-14",
                "2026-05-15",
                "2026-05-16",
                "2026-05-17",
            ])


if __name__ == "__main__":
    unittest.main()
