"""Tests for the P50C prediction lifecycle CLI."""

from __future__ import annotations

from decimal import Decimal
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40APredictionRow,
)
from match_analysis.application.use_cases.p44a_historical_source_adapter import (
    protected_authority_hashes,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    NormalizedPregameInput,
    NormalizedResultRecord,
    write_normalized_pregame_input,
    write_normalized_result_input,
)
from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    CLASSIFICATION_HISTORICAL_REHEARSAL,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
)
from match_analysis.interfaces.cli.run_p50c_prediction_lifecycle import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _make_prediction_row(
    pred_id: str,
    game_id: str,
    p_home: Decimal,
    start_utc: str = "2026-08-18T19:00:00Z",
) -> P40APredictionRow:
    return P40APredictionRow(
        p37_fold_id="fold_2026_01",
        p37_window="window_2026_01",
        p37_prediction_row_id=pred_id,
        provider_namespace="mlb_official",
        provider_game_id=game_id,
        game_pk=900001,
        game_number=1,
        scheduled_start_utc=start_utc,
        champion_model_id="champion_v1",
        champion_model_fingerprint="b" * 64,
        champion_home_probability=p_home,
        challenger_model_id="challenger_v1",
        challenger_model_fingerprint="c" * 64,
        challenger_home_probability=p_home,
    )



def _make_cli_pregame_fixture() -> NormalizedPregameInput:
    pred1 = _make_prediction_row("a" * 64, "game_01", Decimal("0.65"))
    pred2 = _make_prediction_row("b" * 64, "game_02", Decimal("0.40"))
    return NormalizedPregameInput(
        source_identity="PROSPECTIVE_MLB_FEED",
        prediction_rows=(pred1, pred2),
        market_rows=(),
        exclusion_rows=(),
        source_manifest={"adapter": "mlb_pregame_feed"},
        authority_hashes={},
        p37_summary={},
        p39_summary={},
        p39_source_manifest={},
    )


class P50CPredictionLifecycleCliTests(unittest.TestCase):
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
        pregame = _make_cli_pregame_fixture()
        res1 = NormalizedResultRecord(
            prediction_row_id="a" * 64,
            provider_namespace="mlb_official",
            provider_game_id="game_01",
            game_number=1,
            status="FINAL",
            home_score=5,
            away_score=2,
            result_observed_at_utc="2026-08-18T23:00:00Z",
            source_identity="MLB_OFFICIAL_RESULTS",
        )
        res2 = NormalizedResultRecord(
            prediction_row_id="b" * 64,
            provider_namespace="mlb_official",
            provider_game_id="game_02",
            game_number=1,
            status="FINAL",
            home_score=2,
            away_score=4,
            result_observed_at_utc="2026-08-18T23:00:00Z",
            source_identity="MLB_OFFICIAL_RESULTS",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pregame_file = write_normalized_pregame_input(temp_path / "pregame.json", pregame)
            results_file = write_normalized_result_input(temp_path / "results.jsonl", [res1, res2])
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
                        CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                        "--run-root",
                        str(run_root),
                        "--created-at-utc",
                        "2026-08-18T12:00:00Z",
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                    ]
                )
            self.assertEqual(code, 0)
            create_output = create_stdout.getvalue()
            self.assertIn("p50c-create-run=", create_output)
            self.assertIn("status=CREATED", create_output)
            self.assertIn("lifecycle_state=FROZEN", create_output)
            self.assertIn("eligible=2", create_output)

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
            self.assertIn("p50c-status=", status_output)
            self.assertIn("lifecycle_state=FROZEN", status_output)
            self.assertIn("settled_total=0", status_output)
            self.assertIn("pending=2", status_output)

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
                        "--settled-at-utc",
                        "2026-08-18T23:59:59Z",
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                    ]
                )
            self.assertEqual(code, 0)
            settle_output = settle_stdout.getvalue()
            self.assertEqual(settle_output.count("p50c-settle-run="), 1)
            self.assertIn("lifecycle_state=SETTLED", settle_output)
            self.assertIn("newly_settled=2", settle_output)
            self.assertIn("total_settled=2", settle_output)
            self.assertIn("correct=2", settle_output)
            self.assertIn("accuracy=1", settle_output)
            self.assertIn("prediction_forward_sample_count=2", settle_output)

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
            self.assertIn("settled_total=2", status2_output)
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
            self.assertIn("p50c-summary=", summary_output)
            self.assertIn("prediction_forward_sample_count=2", summary_output)
            self.assertIn("runs=1", summary_output)
            self.assertIn("correct=2", summary_output)
            self.assertIn("incorrect=0", summary_output)


if __name__ == "__main__":
    unittest.main()
