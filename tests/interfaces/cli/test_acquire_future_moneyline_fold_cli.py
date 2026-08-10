"""Offline CLI replay tests for P23F2."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NORMALIZED_ROOT = REPOSITORY_ROOT / "data/fixtures/p23f2_official_2026_history/normalized"
SOURCE_MANIFEST = REPOSITORY_ROOT / "report/p23f2_official_future_fold/source_manifest.json"


class AcquireFutureMoneylineFoldCliTests(unittest.TestCase):
    def test_offline_replays_are_byte_identical(self) -> None:
        env = {**__import__("os").environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            outputs = []
            for output in (Path(first), Path(second)):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "match_analysis.interfaces.cli.acquire_future_moneyline_fold",
                        "--offline",
                        "--normalized-root",
                        str(NORMALIZED_ROOT),
                        "--source-manifest",
                        str(SOURCE_MANIFEST),
                        "--raw-root",
                        str(output / "raw"),
                        "--output-dir",
                        str(output / "report"),
                    ],
                    cwd=REPOSITORY_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append(output / "report")
            for name in ("fold_manifest.json", "feature_rows.jsonl", "results.jsonl", "source_manifest.json", "summary.json"):
                self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
