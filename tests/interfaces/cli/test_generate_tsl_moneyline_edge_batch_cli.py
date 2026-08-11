"""Offline CLI replay tests for the P28AB TSL-aligned edge batch."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path("/tmp/matchanalysis-p28ab-tsl-edge/cli-tests")


class GenerateTslMoneylineEdgeBatchCliTests(unittest.TestCase):
    def test_offline_replays_are_byte_identical(self) -> None:
        env = {
            **os.environ,
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as first, tempfile.TemporaryDirectory(
            dir=RUNTIME_ROOT
        ) as second:
            output_dirs = []
            for output_root in (Path(first), Path(second)):
                output_dir = output_root / "report"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "match_analysis.interfaces.cli.generate_tsl_moneyline_edge_batch",
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
                self.assertIn("raw=46 games=16 evaluable=9 prices=16 edges=18", result.stdout)
                self.assertIn("offline=True", result.stdout)
                output_dirs.append(output_dir)
            for name in (
                "raw_cohort.jsonl",
                "prices.jsonl",
                "predictions.jsonl",
                "edges.jsonl",
                "feature_unavailable.jsonl",
                "source_manifest.json",
                "summary.json",
                "report.md",
            ):
                self.assertEqual(
                    (output_dirs[0] / name).read_bytes(),
                    (output_dirs[1] / name).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
