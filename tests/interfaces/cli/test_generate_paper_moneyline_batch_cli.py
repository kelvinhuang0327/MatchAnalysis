"""Offline CLI replay tests for the P24C promoted-default paper batch."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPOSITORY_ROOT / "data/fixtures/p24c_promoted_moneyline_shadow_batch/raw"
NORMALIZED_ROOT = REPOSITORY_ROOT / "data/fixtures/p24c_promoted_moneyline_shadow_batch/normalized"
SOURCE_MANIFEST = REPOSITORY_ROOT / "report/p24c_promoted_moneyline_shadow_batch/source_manifest.json"
RUNTIME_ROOT = Path("/tmp/matchanalysis-p24c-promoted-paper-shadow/cli-tests")


class GeneratePaperMoneylineBatchCliTests(unittest.TestCase):
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
            outputs = []
            for output_root in (Path(first), Path(second)):
                output_dir = output_root / "report"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "match_analysis.interfaces.cli.generate_paper_moneyline_batch",
                        "--offline",
                        "--raw-root",
                        str(RAW_ROOT),
                        "--normalized-root",
                        str(NORMALIZED_ROOT),
                        "--source-manifest",
                        str(SOURCE_MANIFEST),
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
                self.assertIn("raw=90 evaluable=79 unavailable=11", result.stdout)
                self.assertIn("offline=True", result.stdout)
                outputs.append(output_dir)
            for name in (
                "predictions.jsonl",
                "feature_unavailable.jsonl",
                "source_manifest.json",
                "summary.json",
            ):
                self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
