"""Offline CLI replay tests for P30A."""

import os
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


if __name__ == "__main__":
    unittest.main()
