"""CLI acceptance tests for the shipped P42A offline rehearsal entry."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPOSITORY_ROOT / "report/p42a_offline_end_to_end_paper_workflow"
CLI_MODULE = "match_analysis.interfaces.cli.run_p42a_offline_end_to_end_paper_workflow"


class P42AOfflineWorkflowCliTests(unittest.TestCase):
    def _run(self, *extra: str) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return subprocess.run(
            [
                sys.executable,
                "-m",
                CLI_MODULE,
                "--repository-root",
                str(REPOSITORY_ROOT),
                *extra,
            ],
            cwd=REPOSITORY_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_writes_native_artifacts_and_replays_identically(self) -> None:
        first = self._run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("label=OFFLINE_HISTORICAL_PAPER_REHEARSAL", first.stdout)
        self.assertIn("p37_target=65", first.stdout)
        self.assertIn("edge_ready=62", first.stdout)
        self.assertIn("decisions=62", first.stdout)
        self.assertIn("bet=22", first.stdout)
        self.assertIn("pass=40", first.stdout)
        self.assertIn("wins=14", first.stdout)
        self.assertIn("losses=8", first.stdout)
        self.assertIn("net=5.90", first.stdout)
        self.assertIn("no_market=3", first.stdout)
        self.assertIn("p40_reconciliation=RECONCILED", first.stdout)
        self.assertIn("deterministic_rerun=True", first.stdout)

        first_bytes = {
            path.name: path.read_bytes()
            for path in REPORT_ROOT.iterdir()
            if path.is_file()
        }
        self.assertEqual(
            set(first_bytes),
            {
                "source_manifest.json",
                "workflow_ledger.jsonl",
                "exclusions.jsonl",
                "summary.json",
                "report.md",
            },
        )
        summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["workflow_label"], "OFFLINE_HISTORICAL_PAPER_REHEARSAL")
        self.assertIn("OFFLINE", summary["labels"])
        self.assertIn("HISTORICAL", summary["labels"])
        self.assertFalse(summary["prospective"])
        self.assertFalse(summary["live"])
        self.assertFalse(summary["production"])
        report = (REPORT_ROOT / "report.md").read_text(encoding="utf-8")
        self.assertIn("OFFLINE", report)
        self.assertIn("HISTORICAL", report)
        self.assertIn("not prospective, live, production, forward-real", report.lower())
        self.assertFalse(summary["forward_real"])
        self.assertFalse(summary["real_betting_history"])

        second = self._run()
        self.assertEqual(second.returncode, 0, second.stderr)
        second_bytes = {
            path.name: path.read_bytes()
            for path in REPORT_ROOT.iterdir()
            if path.is_file()
        }
        self.assertEqual(first_bytes, second_bytes)

    def test_cli_rejects_output_outside_repository_native_root(self) -> None:
        result = self._run("--output-dir", "/tmp/p42a-outside-allowlist")
        self.assertEqual(result.returncode, 1)
        self.assertIn("ARTIFACT_OUTPUT_PATH_CONFLICT", result.stderr)


if __name__ == "__main__":
    unittest.main()
