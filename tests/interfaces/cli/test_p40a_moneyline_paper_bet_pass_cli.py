"""CLI acceptance tests for the deterministic P40A artifact run."""

from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPOSITORY_ROOT / "report/p40a_moneyline_paper_bet_pass"
CLI_MODULE = "match_analysis.interfaces.cli.run_p40a_moneyline_paper_bet_pass"


class P40AMoneylinePaperBetPassCliTests(unittest.TestCase):
    def _run(self, *extra: str) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return subprocess.run(
            [sys.executable, "-m", CLI_MODULE, "--repository-root", str(REPOSITORY_ROOT), *extra],
            cwd=REPOSITORY_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_writes_all_allowlisted_artifacts_and_replays_identically(self) -> None:
        first = self._run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("edge_ready=62", first.stdout)
        self.assertIn("champion_bet=22", first.stdout)
        self.assertIn("shadow_bet=25", first.stdout)
        self.assertIn("deterministic_rerun=True", first.stdout)

        first_bytes = {
            path.name: path.read_bytes()
            for path in REPORT_ROOT.iterdir()
            if path.is_file()
        }
        self.assertEqual(
            set(first_bytes),
            {"source_manifest.json", "decisions.jsonl", "settlements.jsonl", "summary.json", "report.md"},
        )
        summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["edge_ready_rows"], 62)
        self.assertEqual(summary["roi_label"], "DESCRIPTIVE_PAPER_ONLY")
        self.assertFalse(summary["claims"]["real_betting"])
        self.assertFalse(summary["claims"]["model_promotion"])

        second = self._run()
        self.assertEqual(second.returncode, 0, second.stderr)
        second_bytes = {path.name: path.read_bytes() for path in REPORT_ROOT.iterdir() if path.is_file()}
        self.assertEqual(first_bytes, second_bytes)

    def test_cli_rejects_output_outside_repository_native_root(self) -> None:
        result = self._run("--output-dir", "/tmp/p40a-outside-allowlist")

        self.assertEqual(result.returncode, 1)
        self.assertIn("ARTIFACT_OUTPUT_PATH_CONFLICT", result.stderr)


if __name__ == "__main__":
    unittest.main()
