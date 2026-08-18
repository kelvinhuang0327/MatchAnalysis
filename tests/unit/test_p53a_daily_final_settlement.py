"""Unit tests for P53A daily Moneyline prospective prediction FINAL settlement."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
from urllib.request import Request

from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    NormalizedResultRecord,
)
from match_analysis.application.use_cases.p45a_paper_run_ledger import (
    get_p45a_forward_summary,
)
from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    STATE_FROZEN,
    STATE_PARTIALLY_SETTLED,
    STATE_SETTLED,
    create_p50c_prediction_run,
    get_p50c_forward_summary,
    get_p50c_run_status,
    read_json_object,
    read_jsonl_objects,
)
from match_analysis.application.use_cases.p53a_daily_final_settlement import (
    P53A_RECEIPT_SCHEMA,
    P53A_SOURCE_IDENTITY,
    P53A_TASK_ID,
    acquire_official_final_results_for_run,
    execute_daily_moneyline_final_settlement,
    resolve_prediction_run,
)


def _make_sample_pregame_input(
    target_date: str = "2026-08-18",
    game_count: int = 11,
) -> dict:
    predictions = []
    for i in range(1, game_count + 1):
        gpk = 820000 + i
        hour = 18 + (i % 6)
        minute = (i * 5) % 60
        scheduled = f"{target_date}T{hour:02d}:{minute:02d}:00Z"
        p_home = Decimal("0.550000") if i % 2 == 1 else Decimal("0.450000")
        predictions.append(
            {
                "p37_fold_id": f"prospective_{target_date.replace('-', '_')}",
                "p37_window": f"window_{target_date.replace('-', '_')}",
                "p37_prediction_row_id": sha256(f"pred_{gpk}".encode("utf-8")).hexdigest(),
                "provider_namespace": "MLB_STATS_API",
                "provider_game_id": str(gpk),
                "game_pk": gpk,
                "game_number": 1,
                "scheduled_start_utc": scheduled,
                "champion_model_id": "p22b_moneyline_logistic_challenger_v1",
                "champion_model_fingerprint": sha256(b"mock_model").hexdigest(),
                "champion_home_probability": format(p_home, "f"),
                "challenger_model_id": "p22b_moneyline_logistic_challenger_v1",
                "challenger_model_fingerprint": sha256(b"mock_model").hexdigest(),
                "challenger_home_probability": format(p_home, "f"),
            }
        )
    return {
        "schema_version": "p50c.pregame_input.v1",
        "source_identity": "MLB_OFFICIAL_STATS_API_PROSPECTIVE_FEED",
        "source_manifest": {
            "target_date": target_date,
            "fetched_at_utc": f"{target_date}T12:00:00Z",
        },
        "predictions": predictions,
        "exclusions": [],
    }


def _make_mock_mlb_schedule_payload(
    games_status: list[tuple[int, str, int, int]],
    target_date: str = "2026-08-18",
) -> bytes:
    """games_status: list of (game_pk, status_detailed_state, home_score, away_score)."""
    games = []
    for gpk, detailed_state, home_score, away_score in games_status:
        is_final = detailed_state.lower() == "final"
        games.append(
            {
                "gamePk": gpk,
                "gameNumber": 1,
                "gameDate": f"{target_date}T22:00:00Z",
                "officialDate": target_date,
                "status": {
                    "abstractGameState": "Final" if is_final else ("Live" if detailed_state == "In Progress" else "Preview"),
                    "detailedState": detailed_state,
                },
                "teams": {
                    "home": {
                        "team": {"id": 100 + (gpk % 30), "name": "Home Team", "abbreviation": "HOM"},
                        "score": home_score if is_final else None,
                    },
                    "away": {
                        "team": {"id": 200 + (gpk % 30), "name": "Away Team", "abbreviation": "AWY"},
                        "score": away_score if is_final else None,
                    },
                },
            }
        )

    payload = {
        "dates": [
            {
                "date": target_date,
                "games": games,
            }
        ]
    }
    return json.dumps(payload).encode("utf-8")


class TestP53ADailyFinalSettlement(unittest.TestCase):
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

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_all_pending_when_games_are_scheduled_or_in_progress(self) -> None:
        """Games not yet Final remain pending without fabrication."""
        games_status = [
            (820000 + i, "In Progress" if i % 2 == 0 else "Scheduled", 0, 0)
            for i in range(1, 12)
        ]
        mock_raw = _make_mock_mlb_schedule_payload(games_status)

        def mock_opener(req: Request, timeout: int) -> bytes:
            return mock_raw

        res = execute_daily_moneyline_final_settlement(
            run=self.run_id,
            repository_root=self.repo_root,
            runs_root=self.runs_root,
            ledger_root=self.ledger_root,
            observed_at_utc="2026-08-18T23:00:00Z",
            opener=mock_opener,
        )

        self.assertEqual(res.lifecycle_state, STATE_FROZEN)
        self.assertEqual(res.newly_settled_count, 0)
        self.assertEqual(res.total_settled_count, 0)
        self.assertEqual(res.pending_count, 11)
        self.assertEqual(res.final_results_discovered, 0)
        self.assertEqual(res.non_final_games_count, 11)
        self.assertEqual(res.prediction_forward_sample_count, 0)
        self.assertEqual(res.status, "ALL_PENDING")
        self.assertTrue(res.frozen_predictions_fingerprint_intact)

    def test_partial_final_settlement(self) -> None:
        """Partial slate settlement: FINAL games settle, non-FINAL remain pending."""
        # First 5 games Final, remaining 6 In Progress
        games_status = [
            (820000 + i, "Final" if i <= 5 else "In Progress", 5 + i, 2)
            for i in range(1, 12)
        ]
        mock_raw = _make_mock_mlb_schedule_payload(games_status)

        def mock_opener(req: Request, timeout: int) -> bytes:
            return mock_raw

        res = execute_daily_moneyline_final_settlement(
            run=self.run_id,
            repository_root=self.repo_root,
            runs_root=self.runs_root,
            ledger_root=self.ledger_root,
            observed_at_utc="2026-08-18T23:30:00Z",
            opener=mock_opener,
        )

        self.assertEqual(res.lifecycle_state, STATE_PARTIALLY_SETTLED)
        self.assertEqual(res.newly_settled_count, 5)
        self.assertEqual(res.total_settled_count, 5)
        self.assertEqual(res.pending_count, 6)
        self.assertEqual(res.final_results_discovered, 5)
        self.assertEqual(res.non_final_games_count, 6)
        self.assertEqual(res.prediction_forward_sample_count, 5)
        self.assertEqual(res.status, "NEWLY_SETTLED")
        self.assertTrue(res.frozen_predictions_fingerprint_intact)

        # Check settled predictions file
        settled_rows = read_jsonl_objects(self.run_dir / "settled_predictions.jsonl")
        self.assertEqual(len(settled_rows), 5)

        # Check receipt
        receipt = read_json_object(res.receipt_path)
        self.assertEqual(receipt["lifecycle_state"], STATE_PARTIALLY_SETTLED)
        self.assertEqual(receipt["total_settled_count"], 5)

    def test_all_final_settlement(self) -> None:
        """Full slate settlement: all 11 games are FINAL."""
        games_status = [
            (820000 + i, "Final", 4 if i % 2 == 1 else 2, 1 if i % 2 == 1 else 6)
            for i in range(1, 12)
        ]
        mock_raw = _make_mock_mlb_schedule_payload(games_status)

        def mock_opener(req: Request, timeout: int) -> bytes:
            return mock_raw

        res = execute_daily_moneyline_final_settlement(
            run=self.run_id,
            repository_root=self.repo_root,
            runs_root=self.runs_root,
            ledger_root=self.ledger_root,
            observed_at_utc="2026-08-19T04:00:00Z",
            opener=mock_opener,
        )

        self.assertEqual(res.lifecycle_state, STATE_SETTLED)
        self.assertEqual(res.newly_settled_count, 11)
        self.assertEqual(res.total_settled_count, 11)
        self.assertEqual(res.pending_count, 0)
        self.assertEqual(res.final_results_discovered, 11)
        self.assertEqual(res.non_final_games_count, 0)
        self.assertEqual(res.prediction_forward_sample_count, 11)
        self.assertIsNotNone(res.accuracy)
        self.assertIsNotNone(res.brier_score)
        self.assertIsNotNone(res.log_loss)
        self.assertIsNotNone(res.expected_calibration_error)
        self.assertEqual(res.status, "NEWLY_SETTLED")

    def test_duplicate_replay_is_idempotent(self) -> None:
        """Replaying identical official FINAL authority creates no duplicate settlement."""
        games_status = [
            (820000 + i, "Final", 5, 2)
            for i in range(1, 12)
        ]
        mock_raw = _make_mock_mlb_schedule_payload(games_status)

        def mock_opener(req: Request, timeout: int) -> bytes:
            return mock_raw

        # Run 1: settles all 11
        res1 = execute_daily_moneyline_final_settlement(
            run=self.run_id,
            repository_root=self.repo_root,
            runs_root=self.runs_root,
            ledger_root=self.ledger_root,
            observed_at_utc="2026-08-19T04:00:00Z",
            opener=mock_opener,
        )
        self.assertEqual(res1.newly_settled_count, 11)
        self.assertEqual(res1.total_settled_count, 11)

        # Run 2: identical replay
        res2 = execute_daily_moneyline_final_settlement(
            run=self.run_id,
            repository_root=self.repo_root,
            runs_root=self.runs_root,
            ledger_root=self.ledger_root,
            observed_at_utc="2026-08-19T05:00:00Z",
            opener=mock_opener,
        )
        self.assertEqual(res2.newly_settled_count, 0)
        self.assertEqual(res2.total_settled_count, 11)
        self.assertEqual(res2.pending_count, 0)
        self.assertEqual(res2.prediction_forward_sample_count, 11)
        self.assertEqual(res2.status, "IDEMPOTENT_NO_CHANGE")

        # Verify forward ledger has exactly 11 rows, no duplicates
        ledger_rows = read_jsonl_objects(self.ledger_root / "prediction_forward_ledger.jsonl")
        self.assertEqual(len(ledger_rows), 11)

    def test_conflicting_final_result_rejected(self) -> None:
        """Conflicting FINAL result for an already-settled game is rejected."""
        games_status1 = [(820000 + i, "Final", 5, 2) for i in range(1, 12)]
        mock_raw1 = _make_mock_mlb_schedule_payload(games_status1)

        execute_daily_moneyline_final_settlement(
            run=self.run_id,
            repository_root=self.repo_root,
            runs_root=self.runs_root,
            ledger_root=self.ledger_root,
            observed_at_utc="2026-08-19T04:00:00Z",
            opener=lambda req, timeout: mock_raw1,
        )

        # Run 2 with conflicting score (2-5 instead of 5-2)
        games_status2 = [(820000 + i, "Final", 2, 5) for i in range(1, 12)]
        mock_raw2 = _make_mock_mlb_schedule_payload(games_status2)

        with self.assertRaises(RuntimeError) as ctx:
            execute_daily_moneyline_final_settlement(
                run=self.run_id,
                repository_root=self.repo_root,
                runs_root=self.runs_root,
                ledger_root=self.ledger_root,
                observed_at_utc="2026-08-19T05:00:00Z",
                opener=lambda req, timeout: mock_raw2,
            )
        self.assertIn("P50C_CONFLICTING_RESULT_REJECTED", str(ctx.exception))

    def test_unknown_game_filtered_out(self) -> None:
        """Official MLB schedule containing unknown games only settles frozen games."""
        games_status = [
            (820001, "Final", 4, 2),  # In frozen run
            (999999, "Final", 10, 1), # Extra unknown game
        ]
        mock_raw = _make_mock_mlb_schedule_payload(games_status)

        res = execute_daily_moneyline_final_settlement(
            run=self.run_id,
            repository_root=self.repo_root,
            runs_root=self.runs_root,
            ledger_root=self.ledger_root,
            observed_at_utc="2026-08-18T23:00:00Z",
            opener=lambda req, timeout: mock_raw,
        )

        self.assertEqual(res.newly_settled_count, 1)
        self.assertEqual(res.total_settled_count, 1)
        self.assertEqual(res.pending_count, 10)

    def test_frozen_authority_immutability(self) -> None:
        """Frozen predictions and creation timestamps remain byte-for-byte unchanged."""
        pred_path = self.run_dir / "frozen_predictions.jsonl"
        bytes_before = pred_path.read_bytes()
        fp_before = sha256(bytes_before).hexdigest()

        manifest_before = read_json_object(self.run_dir / "run_manifest.json")

        games_status = [(820000 + i, "Final", 5, 3) for i in range(1, 12)]
        mock_raw = _make_mock_mlb_schedule_payload(games_status)

        execute_daily_moneyline_final_settlement(
            run=self.run_id,
            repository_root=self.repo_root,
            runs_root=self.runs_root,
            ledger_root=self.ledger_root,
            observed_at_utc="2026-08-19T04:00:00Z",
            opener=lambda req, timeout: mock_raw,
        )

        bytes_after = pred_path.read_bytes()
        fp_after = sha256(bytes_after).hexdigest()
        manifest_after = read_json_object(self.run_dir / "run_manifest.json")

        self.assertEqual(bytes_before, bytes_after)
        self.assertEqual(fp_before, fp_after)
        self.assertEqual(manifest_before["created_at_utc"], manifest_after["created_at_utc"])
        self.assertEqual(
            manifest_before["prediction_bundle_fingerprint"],
            manifest_after["prediction_bundle_fingerprint"],
        )

    def test_betting_count_invariance(self) -> None:
        """Settling predictions does not increment betting forward_sample_count."""
        p45a_before = get_p45a_forward_summary(self.repo_root)
        betting_count_before = int(p45a_before.get("forward_sample_count", 0))

        games_status = [(820000 + i, "Final", 4, 1) for i in range(1, 12)]
        mock_raw = _make_mock_mlb_schedule_payload(games_status)

        res = execute_daily_moneyline_final_settlement(
            run=self.run_id,
            repository_root=self.repo_root,
            runs_root=self.runs_root,
            ledger_root=self.ledger_root,
            observed_at_utc="2026-08-19T04:00:00Z",
            opener=lambda req, timeout: mock_raw,
        )

        p45a_after = get_p45a_forward_summary(self.repo_root)
        betting_count_after = int(p45a_after.get("forward_sample_count", 0))

        self.assertEqual(betting_count_before, betting_count_after)
        self.assertEqual(res.betting_forward_sample_count, betting_count_before)
        self.assertEqual(res.prediction_forward_sample_count, 11)

    def test_resolve_prediction_run(self) -> None:
        """Test resolving by run_id, run_dir, and target_date."""
        resolved_by_id = resolve_prediction_run(
            self.repo_root,
            run=self.run_id,
            runs_root=self.runs_root,
        )
        self.assertEqual(resolved_by_id, self.run_dir)

        resolved_by_path = resolve_prediction_run(
            self.repo_root,
            run=self.run_dir,
            runs_root=self.runs_root,
        )
        self.assertEqual(resolved_by_path, self.run_dir)

        resolved_by_date = resolve_prediction_run(
            self.repo_root,
            target_date="2026-08-18",
            runs_root=self.runs_root,
        )
        self.assertEqual(resolved_by_date, self.run_dir)


if __name__ == "__main__":
    unittest.main()
