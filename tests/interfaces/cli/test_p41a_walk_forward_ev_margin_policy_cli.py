"""CLI acceptance tests for the deterministic P41A artifact run."""

from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPOSITORY_ROOT / "report/p41a_walk_forward_ev_margin_policy"
CLI_MODULE = (
    "match_analysis.interfaces.cli.run_p41a_walk_forward_ev_margin_policy"
)


class P41AWalkForwardPolicyCliTests(unittest.TestCase):
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

    def test_cli_writes_allowlisted_artifacts_and_replays_identically(self) -> None:
        first = self._run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("target_windows=2", first.stdout)
        self.assertIn("target_rows=40", first.stdout)
        self.assertIn("selected_bet=8", first.stdout)
        self.assertIn("zero_ev_bet=12", first.stdout)
        self.assertIn("conclusion=EV_MARGIN_POLICY_IMPROVED", first.stdout)
        self.assertIn("deterministic_rerun=True", first.stdout)

        first_bytes = {
            path.name: path.read_bytes()
            for path in REPORT_ROOT.iterdir()
            if path.is_file()
        }
        self.assertEqual(
            set(first_bytes),
            {"source_manifest.json", "policy_evaluations.jsonl", "summary.json", "report.md"},
        )
        summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["policy_oos_target_rows"], 40)
        self.assertEqual(summary["conclusion"], "EV_MARGIN_POLICY_IMPROVED")
        self.assertFalse(summary["claims"]["real_betting"])
        self.assertFalse(summary["claims"]["model_promotion"])

        second = self._run()
        self.assertEqual(second.returncode, 0, second.stderr)
        second_bytes = {
            path.name: path.read_bytes()
            for path in REPORT_ROOT.iterdir()
            if path.is_file()
        }
        self.assertEqual(first_bytes, second_bytes)

    def test_cli_rejects_output_outside_repository_native_root(self) -> None:
        result = self._run("--output-dir", "/tmp/p41a-outside-allowlist")

        self.assertEqual(result.returncode, 1)
        self.assertIn("ARTIFACT_OUTPUT_PATH_CONFLICT", result.stderr)


if __name__ == "__main__":
    unittest.main()
