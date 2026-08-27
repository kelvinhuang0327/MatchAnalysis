"""Characterization tests for P61B Run Line parity, lifecycle freeze, and settlement."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.run_line_prediction_lifecycle import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P61B_CLAIMS,
    RunLinePregameInputItem,
    freeze_run_line_predictions,
    settle_run_line_predictions,
)
from match_analysis.baseball.domain.canonical_utc import parse_canonical_utc
from match_analysis.baseball.domain.run_line_model import (
    FROZEN_HOME_ADVANTAGE,
    MODEL_ID,
    MODEL_VERSION,
    PriorGameRecord,
)


class RunLineParityAndLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t0 = parse_canonical_utc("2026-04-01T00:00:00Z")
        self.history = [
            PriorGameRecord("TEX", "STL", 5, 3, self.t0),
            PriorGameRecord("TEX", "LAD", 6, 2, self.t0),
            PriorGameRecord("STL", "LAD", 4, 5, self.t0),
        ]
        self.game_item_fav = RunLinePregameInputItem(
            provider_game_id="2026-04-02_TEX_STL",
            home_team="TEX",
            away_team="STL",
            scheduled_start_utc="2026-04-02T19:05:00Z",
            official_date="2026-04-02",
            as_of_utc="2026-04-02T12:00:00Z",
            spread_home=-1.5,
            game_pk=820001,
        )
        self.game_item_dog = RunLinePregameInputItem(
            provider_game_id="2026-04-02_TEX_STL",
            home_team="TEX",
            away_team="STL",
            scheduled_start_utc="2026-04-02T19:05:00Z",
            official_date="2026-04-02",
            as_of_utc="2026-04-02T12:00:00Z",
            spread_home=1.5,
            game_pk=820001,
        )

    def test_lifecycle_freeze_and_settlement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p61b-test-") as tmpdir:
            run_dir = Path(tmpdir) / "run_001"
            preds, manifest = freeze_run_line_predictions(
                [self.game_item_fav],
                self.history,
                run_id="run_001",
                output_dir=run_dir,
            )

            self.assertEqual(len(preds), 1)
            pred = preds[0]
            self.assertEqual(pred.market_id, "run_line")
            self.assertEqual(pred.model_id, MODEL_ID)
            self.assertEqual(pred.model_version, MODEL_VERSION)
            self.assertEqual(pred.spread_home, -1.5)
            self.assertEqual(pred.p_push, 0.0)
            self.assertGreater(pred.p_home_covers + pred.p_away_covers, 0.999)

            # Check written files
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertTrue((run_dir / "predictions.jsonl").is_file())

            # Settle run against final score: TEX 6, STL 3 (diff = 3 > 1.5 -> HOME covers)
            results = {"2026-04-02_TEX_STL": (6, 3)}
            settled_rows, summary = settle_run_line_predictions(
                preds,
                results,
                run_id="run_001",
                output_dir=run_dir,
            )

            self.assertEqual(len(settled_rows), 1)
            self.assertEqual(summary["total_count"], 1)
            self.assertEqual(summary["push_count"], 0)
            self.assertEqual(summary["decided_count"], 1)
            self.assertEqual(summary["correct_count"], 1 if pred.predicted_side == "HOME" else 0)

            self.assertTrue((run_dir / "settled_predictions.jsonl").is_file())
            self.assertTrue((run_dir / "settlement_summary.json").is_file())

    def test_identical_replay_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p61b-idempotent-") as tmpdir:
            dir1 = Path(tmpdir) / "run1"
            dir2 = Path(tmpdir) / "run2"

            preds1, m1 = freeze_run_line_predictions(
                [self.game_item_fav], self.history, run_id="run_same", output_dir=dir1
            )
            preds2, m2 = freeze_run_line_predictions(
                [self.game_item_fav], self.history, run_id="run_same", output_dir=dir2
            )

            self.assertEqual(
                (dir1 / "predictions.jsonl").read_bytes(),
                (dir2 / "predictions.jsonl").read_bytes(),
            )
            self.assertEqual(preds1[0].prediction_fingerprint, preds2[0].prediction_fingerprint)

    def test_changing_spread_home_changes_prediction_identity(self) -> None:
        preds_fav, _ = freeze_run_line_predictions([self.game_item_fav], self.history, run_id="r1")
        preds_dog, _ = freeze_run_line_predictions([self.game_item_dog], self.history, run_id="r2")

        self.assertNotEqual(
            preds_fav[0].prediction_fingerprint,
            preds_dog[0].prediction_fingerprint,
        )
        self.assertNotEqual(
            preds_fav[0].workflow_prediction_id,
            preds_dog[0].workflow_prediction_id,
        )
        self.assertNotEqual(
            preds_fav[0].p_home_covers,
            preds_dog[0].p_home_covers,
        )

    def test_post_start_as_of_time_fails_closed(self) -> None:
        late_item = RunLinePregameInputItem(
            provider_game_id="2026-04-02_TEX_STL",
            home_team="TEX",
            away_team="STL",
            scheduled_start_utc="2026-04-02T19:05:00Z",
            official_date="2026-04-02",
            as_of_utc="2026-04-02T19:05:00Z",
            spread_home=-1.5,
        )
        with self.assertRaises(ValueError):
            freeze_run_line_predictions([late_item], self.history, run_id="r_late")

    def test_cross_midnight_provider_official_date_is_preserved(self) -> None:
        cross_midnight = RunLinePregameInputItem(
            provider_game_id="824072",
            home_team="TEX",
            away_team="STL",
            scheduled_start_utc="2026-08-22T00:10:00Z",
            official_date="2026-08-21",
            as_of_utc="2026-08-21T12:00:00Z",
            spread_home=-1.5,
            game_pk=824072,
        )
        predictions, manifest = freeze_run_line_predictions(
            [cross_midnight],
            self.history,
            run_id="p66a_cross_midnight",
            created_at_utc="2026-08-21T12:00:00Z",
        )
        payload = predictions[0].to_payload()
        self.assertEqual(payload["game_identity"]["official_date"], "2026-08-21")
        self.assertEqual(
            payload["game_identity"]["scheduled_start_utc"],
            "2026-08-22T00:10:00Z",
        )
        self.assertEqual(manifest["target_date"], "2026-08-21")

    def test_run_line_canonical_forward_integration_with_preexisting_moneyline_rows(self) -> None:
        """Temporary P50C runtime with pre-existing Moneyline rows receives Run Line evidence without mutating Moneyline rows."""
        from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
            read_jsonl_objects,
            get_p50c_forward_summary,
        )

        with tempfile.TemporaryDirectory(prefix="p64b-p50c-") as tmpdir:
            tmp_root = Path(tmpdir)
            runs_root = tmp_root / "runs"
            ledger_root = tmp_root / "ledger"
            ledger_root.mkdir(parents=True, exist_ok=True)

            forward_ledger_path = ledger_root / "prediction_forward_ledger.jsonl"

            # Write mock pre-existing Moneyline forward records
            mock_ml_record = {
                "ledger_record_id": "ml_rec_001",
                "schema_version": "p50c.forward_prediction_ledger_record.v1",
                "run_id": "p50c_run_ml_01",
                "run_classification": CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                "market_id": "moneyline",
                "prediction_identity": {
                    "workflow_prediction_id": "p50c_pred_ml001",
                    "prediction_fingerprint": "ml_fp_001",
                },
                "game_identity": {
                    "provider_namespace": "MLB_STATS_API",
                    "provider_game_id": "2026-04-01_TEX_STL",
                    "game_pk": 810001,
                    "game_number": 1,
                    "scheduled_start_utc": "2026-04-01T19:05:00Z",
                    "official_date": "2026-04-01",
                    "home_team": "TEX",
                    "away_team": "STL",
                },
                "model_identity": {
                    "model_role": "CHAMPION",
                    "model_id": "champion_ml_v1",
                    "p_home": "0.55",
                    "p_away": "0.45",
                    "selection": "HOME",
                    "model_probability": "0.55",
                },
                "final_result_authority": {
                    "status": "FINAL",
                    "home_score": 5,
                    "away_score": 3,
                    "actual_winner": "HOME",
                },
                "evaluation": {
                    "selection": "HOME",
                    "actual_winner": "HOME",
                    "is_correct": True,
                    "correctness_target": 1,
                    "brier_component": "0.2025",
                    "log_loss_component": "0.597837000756",
                    "evaluated_at_utc": "2026-04-01T23:00:00Z",
                },
            }
            forward_ledger_path.write_text(
                json.dumps(mock_ml_record, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            ml_bytes_before = forward_ledger_path.read_bytes()

            # 1. Freeze Run Line prediction with standard default spread -1.5
            run_id = "p50c_run_rl_20260402"
            run_dir = runs_root / run_id
            preds, manifest = freeze_run_line_predictions(
                [self.game_item_fav],
                self.history,
                run_id=run_id,
                classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                output_dir=run_dir,
                created_at_utc="2026-04-02T12:00:00Z",
            )
            self.assertEqual(len(preds), 1)
            pred = preds[0]
            self.assertEqual(pred.market_id, "run_line")
            self.assertEqual(pred.spread_home, -1.5)
            self.assertEqual(pred.line_value, -1.5)

            # Invariant: zero betting/TSL contamination in claims and frozen record
            self.assertFalse(manifest["claims"]["real_betting"])
            self.assertFalse(manifest["claims"]["odds_or_market_dependent"])
            self.assertFalse(manifest["claims"]["tsl_access"])

            # 2. Settle Run Line prediction
            results = {"2026-04-02_TEX_STL": (6, 3)}  # TEX 6, STL 3 -> diff 3 > 1.5 -> HOME covers
            settled_rows, summary = settle_run_line_predictions(
                preds,
                results,
                run_id=run_id,
                run_dir=run_dir,
                ledger_root=ledger_root,
                settled_at_utc="2026-04-02T23:00:00Z",
            )
            expected_rl_correct = 1 if pred.predicted_side == "HOME" else 0
            self.assertEqual(summary["correct_count"], expected_rl_correct)

            # 3. Verify ledger rows: ML record is untouched at line 1, RL record appended at line 2
            ledger_rows = read_jsonl_objects(forward_ledger_path)
            self.assertEqual(len(ledger_rows), 2)
            self.assertEqual(ledger_rows[0]["market_id"], "moneyline")
            self.assertEqual(ledger_rows[0]["run_id"], "p50c_run_ml_01")
            self.assertEqual(ledger_rows[1]["market_id"], "run_line")
            self.assertEqual(ledger_rows[1]["run_id"], run_id)
            self.assertEqual(ledger_rows[1]["line_value"], -1.5)
            self.assertEqual(ledger_rows[1]["evaluation"]["is_correct"], pred.predicted_side == "HOME")

            # Check forward summary
            fwd_summary = get_p50c_forward_summary(tmp_root, ledger_root=ledger_root)
            self.assertEqual(fwd_summary["PREDICTION_FORWARD_SAMPLE_COUNT"], 2)
            self.assertEqual(fwd_summary["correct_count"], 1 + expected_rl_correct)

    def test_duplicate_run_line_settlement_is_idempotent(self) -> None:
        """Duplicate Run Line settlement replay must be idempotent and reject conflicting authority."""
        from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
            read_jsonl_objects,
        )

        with tempfile.TemporaryDirectory(prefix="p64b-idempotent-") as tmpdir:
            tmp_root = Path(tmpdir)
            runs_root = tmp_root / "runs"
            ledger_root = tmp_root / "ledger"

            run_id = "p50c_run_rl_idem"
            run_dir = runs_root / run_id
            preds, _ = freeze_run_line_predictions(
                [self.game_item_fav],
                self.history,
                run_id=run_id,
                output_dir=run_dir,
                created_at_utc="2026-04-02T12:00:00Z",
            )

            results_1 = {"2026-04-02_TEX_STL": (6, 3)}

            # First settlement
            settle_run_line_predictions(
                preds,
                results_1,
                run_id=run_id,
                run_dir=run_dir,
                ledger_root=ledger_root,
                settled_at_utc="2026-04-02T23:00:00Z",
            )
            ledger_path = ledger_root / "prediction_forward_ledger.jsonl"
            self.assertEqual(len(read_jsonl_objects(ledger_path)), 1)

            # Second identical settlement replay -> zero duplicate rows
            settle_run_line_predictions(
                preds,
                results_1,
                run_id=run_id,
                run_dir=run_dir,
                ledger_root=ledger_root,
                settled_at_utc="2026-04-02T23:00:00Z",
            )
            self.assertEqual(len(read_jsonl_objects(ledger_path)), 1)

            # Conflicting result settlement -> raises conflict error
            results_conflict = {"2026-04-02_TEX_STL": (2, 7)}
            with self.assertRaises(RuntimeError) as cm:
                settle_run_line_predictions(
                    preds,
                    results_conflict,
                    run_id=run_id,
                    run_dir=run_dir,
                    ledger_root=ledger_root,
                )
            self.assertIn("P50C_CONFLICTING_RESULT_REJECTED", str(cm.exception))

    def test_run_line_cli_freeze_and_settle(self) -> None:
        """Test full CLI freeze and settle subcommands."""
        from match_analysis.interfaces.cli.run_run_line_prediction_lifecycle import main as cli_main
        from match_analysis.application.use_cases.p50c_prediction_run_ledger import read_jsonl_objects

        with tempfile.TemporaryDirectory(prefix="p64b-cli-") as tmpdir:
            tmp_root = Path(tmpdir)
            input_file = tmp_root / "pregame_input.json"
            history_file = tmp_root / "history.json"
            results_file = tmp_root / "results.json"
            runs_root = tmp_root / "runs"
            ledger_root = tmp_root / "ledger"

            input_data = [
                {
                    "provider_game_id": "2026-04-02_TEX_STL",
                    "home_team": "TEX",
                    "away_team": "STL",
                    "scheduled_start_utc": "2026-04-02T19:05:00Z",
                    "official_date": "2026-04-02",
                    "as_of_utc": "2026-04-02T12:00:00Z",
                    "spread_home": -1.5,
                }
            ]
            input_file.write_text(json.dumps(input_data), encoding="utf-8")

            hist_data = [
                {
                    "home_team": "TEX",
                    "away_team": "STL",
                    "home_score": 5,
                    "away_score": 3,
                    "completed_at_utc": "2026-04-01T00:00:00Z",
                }
            ]
            history_file.write_text(json.dumps(hist_data), encoding="utf-8")

            results_data = {"2026-04-02_TEX_STL": [6, 3]}
            results_file.write_text(json.dumps(results_data), encoding="utf-8")

            run_id = "p50c_run_rl_cli_01"

            # CLI Freeze
            freeze_code = cli_main(
                [
                    "freeze",
                    "--input", str(input_file),
                    "--history", str(history_file),
                    "--run-id", run_id,
                    "--runs-root", str(runs_root),
                    "--created-at-utc", "2026-04-02T12:00:00Z",
                ]
            )
            self.assertEqual(freeze_code, 0)
            self.assertTrue((runs_root / run_id / "run_manifest.json").is_file())
            self.assertTrue((runs_root / run_id / "frozen_predictions.jsonl").is_file())

            # CLI Settle
            settle_code = cli_main(
                [
                    "settle",
                    "--run-id", run_id,
                    "--runs-root", str(runs_root),
                    "--ledger-root", str(ledger_root),
                    "--results", str(results_file),
                    "--settled-at-utc", "2026-04-02T23:00:00Z",
                ]
            )
            self.assertEqual(settle_code, 0)
            self.assertTrue((runs_root / run_id / "settled_predictions.jsonl").is_file())
            self.assertTrue((runs_root / run_id / "settlement_summary.json").is_file())
            self.assertTrue((ledger_root / "prediction_forward_ledger.jsonl").is_file())
            self.assertEqual(len(read_jsonl_objects(ledger_root / "prediction_forward_ledger.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
