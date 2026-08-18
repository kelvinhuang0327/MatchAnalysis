"""CLI tests for P53A daily Moneyline prospective prediction FINAL settlement."""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    create_p50c_prediction_run,
)
from match_analysis.interfaces.cli.run_p53a_daily_final_settlement import (
    build_parser,
    main,
)
from tests.unit.test_p53a_daily_final_settlement import (
    _make_sample_pregame_input,
)


class TestP53ADailyFinalSettlementCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.repo_root = Path(self.test_dir)
        self.runs_root = self.repo_root / "report" / "p50c_prospective_prediction_shadow_ledger" / "runs"
        self.ledger_root = self.repo_root / "report" / "p50c_prospective_prediction_shadow_ledger" / "ledger"
        self.intake_dir = self.repo_root / "report" / "p50c_prospective_prediction_shadow_ledger" / "intake"
        self.intake_dir.mkdir(parents=True, exist_ok=True)

        # Create frozen pregame run
        self.pregame_data = _make_sample_pregame_input("2026-08-18", game_count=11)
        self.pregame_file = self.intake_dir / "prospective_pregame_20260818.json"
        self.pregame_file.write_text(json.dumps(self.pregame_data, indent=2), encoding="utf-8")

        self.create_res = create_p50c_prediction_run(
            self.repo_root,
            pregame_input=self.pregame_file,
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            run_root=self.runs_root,
            created_at_utc="2026-08-18T13:37:42Z",
        )
        self.run_id = self.create_res.run_id
        self.run_dir = self.create_res.run_dir

        # Create offline results file
        results_rows = []
        for i in range(1, 12):
            gpk = 820000 + i
            results_rows.append(
                json.dumps(
                    {
                        "schema_version": "p44a.normalized_result_input.v1",
                        "prediction_row_id": self.pregame_data["predictions"][i - 1]["p37_prediction_row_id"],
                        "provider_namespace": "MLB_STATS_API",
                        "provider_game_id": str(gpk),
                        "game_number": 1,
                        "status": "FINAL",
                        "home_score": 5,
                        "away_score": 2,
                        "result_observed_at_utc": "2026-08-19T04:00:00Z",
                        "source_identity": "OFFLINE_TEST_RESULTS",
                    }
                )
            )
        self.results_file = self.repo_root / "results.jsonl"
        self.results_file.write_text("\n".join(results_rows) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_cli_help(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_cli_with_offline_final_results(self) -> None:
        argv = [
            "--run",
            self.run_id,
            "--repository-root",
            str(self.repo_root),
            "--runs-root",
            str(self.runs_root),
            "--ledger-root",
            str(self.ledger_root),
            "--final-results",
            str(self.results_file),
            "--observed-at-utc",
            "2026-08-19T04:00:00Z",
        ]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            exit_code = main(argv)
            output = mock_out.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("p53a-daily-final-settlement=", output)
        self.assertIn("lifecycle_state=SETTLED", output)
        self.assertIn("newly_settled=11", output)
        self.assertIn("total_settled=11", output)
        self.assertIn("pending=0", output)

    def test_cli_with_json_flag(self) -> None:
        argv = [
            "--target-date",
            "2026-08-18",
            "--repository-root",
            str(self.repo_root),
            "--runs-root",
            str(self.runs_root),
            "--ledger-root",
            str(self.ledger_root),
            "--final-results",
            str(self.results_file),
            "--observed-at-utc",
            "2026-08-19T04:00:00Z",
            "--json",
        ]
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            exit_code = main(argv)
            output = mock_out.getvalue()

        self.assertEqual(exit_code, 0)
        parsed = json.loads(output)
        self.assertEqual(parsed["schema_version"], "p53a.daily_final_settlement_receipt.v1")
        self.assertEqual(parsed["task_id"], "P53A")
        self.assertEqual(parsed["run_id"], self.run_id)
        self.assertEqual(parsed["lifecycle_state"], "SETTLED")
        self.assertEqual(parsed["total_settled_count"], 11)
        self.assertEqual(parsed["pending_count"], 0)
        self.assertEqual(parsed["prediction_forward_sample_count"], 11)

    def test_cli_missing_run_fails(self) -> None:
        argv = [
            "--run",
            "non_existent_run_id",
            "--repository-root",
            str(self.repo_root),
            "--runs-root",
            str(self.runs_root),
        ]
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            exit_code = main(argv)
            output = mock_err.getvalue()

        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR:", output)


if __name__ == "__main__":
    unittest.main()
