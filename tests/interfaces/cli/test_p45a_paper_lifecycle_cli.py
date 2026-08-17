"""Tests for the P45A paper lifecycle CLI."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from match_analysis.application.use_cases.p44a_historical_source_adapter import (
    adapt_historical_pregame,
    adapt_historical_results,
    protected_authority_hashes,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    write_normalized_pregame_input,
    write_normalized_result_input,
)
from match_analysis.application.use_cases.p45a_paper_run_ledger import (
    CLASSIFICATION_HISTORICAL_REHEARSAL,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
)
from match_analysis.interfaces.cli.run_p45a_paper_lifecycle import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class P45APaperLifecycleCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.authority_hashes_before = protected_authority_hashes(REPOSITORY_ROOT)

    def tearDown(self) -> None:
        authority_hashes_after = protected_authority_hashes(REPOSITORY_ROOT)
        self.assertEqual(
            self.authority_hashes_before,
            authority_hashes_after,
            "Protected historical authority hashes drifted during CLI test execution",
        )

    def test_cli_lifecycle_end_to_end(self) -> None:
        pregame = adapt_historical_pregame(REPOSITORY_ROOT)
        results = adapt_historical_results(REPOSITORY_ROOT)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pregame_file = write_normalized_pregame_input(temp_path / "pregame", pregame)
            results_file = write_normalized_result_input(temp_path / "results.jsonl", results)
            run_root = temp_path / "runs"
            ledger_root = temp_path / "ledger"

            # 1. create-run
            create_stdout = io.StringIO()
            with patch("sys.stdout", create_stdout):
                code = main(
                    [
                        "create-run",
                        "--pregame-input",
                        str(pregame_file),
                        "--classification",
                        CLASSIFICATION_HISTORICAL_REHEARSAL,
                        "--run-root",
                        str(run_root),
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                    ]
                )
            self.assertEqual(code, 0)
            create_output = create_stdout.getvalue()
            self.assertIn("p45a-create-run=", create_output)
            self.assertIn("status=CREATED", create_output)
            self.assertIn("lifecycle_state=FROZEN", create_output)
            self.assertIn("eligible=62", create_output)
            self.assertIn("bet=22", create_output)
            self.assertIn("pass=40", create_output)

            # Extract run directory
            run_dir = [d for d in run_root.iterdir() if d.is_dir()][0]

            # 2. status before settlement
            status_stdout = io.StringIO()
            with patch("sys.stdout", status_stdout):
                code = main(
                    [
                        "status",
                        "--run",
                        str(run_dir),
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                    ]
                )
            self.assertEqual(code, 0)
            status_output = status_stdout.getvalue()
            self.assertIn("p45a-status=", status_output)
            self.assertIn("lifecycle_state=FROZEN", status_output)
            self.assertIn("settled_total=0", status_output)
            self.assertIn("pending=62", status_output)

            # 3. settle-run
            settle_stdout = io.StringIO()
            with patch("sys.stdout", settle_stdout):
                code = main(
                    [
                        "settle-run",
                        "--run",
                        str(run_dir),
                        "--result-input",
                        str(results_file),
                        "--ledger-root",
                        str(ledger_root),
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                    ]
                )
            self.assertEqual(code, 0)
            settle_output = settle_stdout.getvalue()
            self.assertIn("p45a-settle-run=", settle_output)
            self.assertIn("lifecycle_state=SETTLED", settle_output)
            self.assertIn("newly_settled=62", settle_output)
            self.assertIn("total_settled=62", settle_output)
            self.assertIn("wins=14", settle_output)
            self.assertIn("losses=8", settle_output)
            self.assertIn("units_risked=22.0", settle_output)
            self.assertIn("net=5.90", settle_output)
            self.assertIn("forward_sample_count=0", settle_output)

            # 4. status after settlement
            status2_stdout = io.StringIO()
            with patch("sys.stdout", status2_stdout):
                code = main(
                    [
                        "status",
                        "--run-dir",
                        str(run_dir),
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                    ]
                )
            self.assertEqual(code, 0)
            status2_output = status2_stdout.getvalue()
            self.assertIn("lifecycle_state=SETTLED", status2_output)
            self.assertIn("settled_total=62", status2_output)
            self.assertIn("pending=0", status2_output)

            # 5. summary
            summary_stdout = io.StringIO()
            with patch("sys.stdout", summary_stdout):
                code = main(
                    [
                        "summary",
                        "--ledger-root",
                        str(ledger_root),
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                    ]
                )
            self.assertEqual(code, 0)
            summary_output = summary_stdout.getvalue()
            self.assertIn("p45a-summary=", summary_output)
            self.assertIn("forward_sample_count=0", summary_output)
            self.assertIn("runs=0", summary_output)
            self.assertIn("net=0.00", summary_output)


if __name__ == "__main__":
    unittest.main()
