"""CLI acceptance tests for the shipped P43A two-phase paper workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPOSITORY_ROOT / "report/p43a_two_phase_paper_workflow"
CLI_MODULE = "match_analysis.interfaces.cli.run_p43a_two_phase_paper_workflow"
HUMAN_LABEL = "OFFLINE / HISTORICAL TWO-PHASE PAPER REHEARSAL"


class P43ATwoPhaseWorkflowCliTests(unittest.TestCase):
    def _run(self, command: str, *extra: str) -> subprocess.CompletedProcess[str]:
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
                command,
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

    def test_cli_pregame_then_postgame_replay_and_reconcile(self) -> None:
        first_pregame = self._run("pregame-freeze")
        self.assertEqual(first_pregame.returncode, 0, first_pregame.stderr)
        self.assertIn("label=OFFLINE_HISTORICAL_TWO_PHASE_PAPER_REHEARSAL", first_pregame.stdout)
        self.assertIn(HUMAN_LABEL, first_pregame.stdout)
        self.assertIn("phase=PREGAME_FREEZE", first_pregame.stdout)
        self.assertIn("decisions=62", first_pregame.stdout)
        self.assertIn("bet=22", first_pregame.stdout)
        self.assertIn("pass=40", first_pregame.stdout)
        self.assertIn("settled=0", first_pregame.stdout)
        self.assertIn("network_required=False", first_pregame.stdout)

        pregame_names = (
            "source_manifest.json",
            "pregame_decisions.jsonl",
            "pregame_summary.json",
            "exclusions.jsonl",
        )
        pregame_files = {
            name: (REPORT_ROOT / name).read_bytes()
            for name in pregame_names
        }
        summary = json.loads((REPORT_ROOT / "pregame_summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((REPORT_ROOT / "source_manifest.json").read_text(encoding="utf-8"))
        prediction_source = manifest["p37a"]["prediction_source_path"]
        self.assertEqual(
            prediction_source,
            "report/p37a_rolling_walk_forward_oos/comparisons.jsonl",
        )
        self.assertFalse(Path(prediction_source).is_absolute())
        self.assertNotIn(str(REPOSITORY_ROOT), prediction_source)
        self.assertEqual(summary["human_label"], HUMAN_LABEL)
        self.assertEqual(summary["settled_bet_count"], 0)
        self.assertFalse(summary["live"])
        self.assertFalse(summary["prospective"])
        self.assertFalse(summary["claims"]["real_betting"])

        second_pregame = self._run("pregame-freeze")
        self.assertEqual(second_pregame.returncode, 0, second_pregame.stderr)
        self.assertIn("freeze_status=RECOGNIZED_IDENTICAL", second_pregame.stdout)
        second_pregame_files = {
            name: (REPORT_ROOT / name).read_bytes()
            for name in pregame_names
        }
        self.assertEqual(pregame_files, second_pregame_files)

        first_postgame = self._run("postgame-settle")
        self.assertEqual(first_postgame.returncode, 0, first_postgame.stderr)
        self.assertIn("label=OFFLINE_HISTORICAL_TWO_PHASE_PAPER_REHEARSAL", first_postgame.stdout)
        self.assertIn(HUMAN_LABEL, first_postgame.stdout)
        self.assertIn("phase=POSTGAME_SETTLE", first_postgame.stdout)
        self.assertIn("decisions=62", first_postgame.stdout)
        self.assertIn("bet=22", first_postgame.stdout)
        self.assertIn("pass=40", first_postgame.stdout)
        self.assertIn("settled=22", first_postgame.stdout)
        self.assertIn("wins=14", first_postgame.stdout)
        self.assertIn("losses=8", first_postgame.stdout)
        self.assertIn("net=5.90", first_postgame.stdout)
        self.assertIn("feedback=62", first_postgame.stdout)
        self.assertIn("p42_reconciliation=RECONCILED", first_postgame.stdout)

        first_bytes = {
            path.name: path.read_bytes()
            for path in REPORT_ROOT.iterdir()
            if path.is_file()
        }
        self.assertEqual(
            set(first_bytes),
            {
                "source_manifest.json",
                "pregame_decisions.jsonl",
                "pregame_summary.json",
                "exclusions.jsonl",
                "workflow_ledger.jsonl",
                "postgame_summary.json",
                "report.md",
            },
        )
        self.assertEqual(first_bytes["pregame_decisions.jsonl"], pregame_files["pregame_decisions.jsonl"])
        report = (REPORT_ROOT / "report.md").read_text(encoding="utf-8")
        self.assertIn(HUMAN_LABEL, report)
        self.assertIn("not prospective, live, production", report.lower())
        postgame_summary = json.loads(
            (REPORT_ROOT / "postgame_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(postgame_summary["p42_reconciliation"]["status"], "RECONCILED")
        self.assertFalse(postgame_summary["claims"]["real_betting"])
        self.assertFalse(postgame_summary["network_required"])

        second_postgame = self._run("postgame-settle")
        self.assertEqual(second_postgame.returncode, 0, second_postgame.stderr)
        second_bytes = {
            path.name: path.read_bytes()
            for path in REPORT_ROOT.iterdir()
            if path.is_file()
        }
        self.assertEqual(first_bytes, second_bytes)

    def test_cli_rejects_output_outside_repository_native_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            result = self._run("pregame-freeze", "--output-dir", raw_directory)
        self.assertEqual(result.returncode, 1)
        self.assertIn("ARTIFACT_OUTPUT_PATH_CONFLICT", result.stderr)

    def test_cli_normalized_pregame_and_result_input_twice(self) -> None:
        from match_analysis.application.use_cases.p44a_historical_source_adapter import (
            adapt_historical_pregame,
            adapt_historical_results,
        )
        from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
            write_normalized_pregame_input,
            write_normalized_result_input,
        )

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            pregame_path = write_normalized_pregame_input(
                directory / "pregame_input.json",
                adapt_historical_pregame(REPOSITORY_ROOT),
            )
            result_path = write_normalized_result_input(
                directory / "result_input.jsonl",
                adapt_historical_results(REPOSITORY_ROOT),
            )
            bundle = directory / "bundle"
            first_pregame = self._run(
                "pregame-freeze",
                "--pregame-input",
                str(pregame_path),
                "--output-dir",
                str(bundle),
            )
            self.assertEqual(first_pregame.returncode, 0, first_pregame.stderr)
            self.assertIn("decisions=62", first_pregame.stdout)
            self.assertIn("bet=22", first_pregame.stdout)
            self.assertIn("pass=40", first_pregame.stdout)
            self.assertIn("settled=0", first_pregame.stdout)
            self.assertIn("network_required=False", first_pregame.stdout)
            second_pregame = self._run(
                "pregame-freeze",
                "--pregame-input",
                str(pregame_path),
                "--output-dir",
                str(bundle),
            )
            self.assertEqual(second_pregame.returncode, 0, second_pregame.stderr)
            self.assertIn("decisions=62", second_pregame.stdout)
            self.assertIn("bet=22", second_pregame.stdout)
            first_postgame = self._run(
                "postgame-settle",
                "--decision-bundle",
                str(bundle),
                "--result-input",
                str(result_path),
            )
            self.assertEqual(first_postgame.returncode, 0, first_postgame.stderr)
            self.assertIn("settled=22", first_postgame.stdout)
            self.assertIn("wins=14", first_postgame.stdout)
            self.assertIn("losses=8", first_postgame.stdout)
            self.assertIn("net=5.90", first_postgame.stdout)
            self.assertIn("feedback=62", first_postgame.stdout)
            second_postgame = self._run(
                "postgame-settle",
                "--decision-bundle",
                str(bundle),
                "--result-input",
                str(result_path),
            )
            self.assertEqual(second_postgame.returncode, 0, second_postgame.stderr)
            self.assertIn("settled=22", second_postgame.stdout)
            self.assertIn("wins=14", second_postgame.stdout)
            self.assertEqual(
                (bundle / "pregame_decisions.jsonl").read_bytes(),
                (bundle / "pregame_decisions.jsonl").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
