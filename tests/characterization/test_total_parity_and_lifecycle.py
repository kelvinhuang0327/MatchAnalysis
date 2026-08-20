"""Characterization tests for P62A Total parity, lifecycle freeze, and settlement."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.total_prediction_lifecycle import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P62A_CLAIMS,
    TotalPregameInputItem,
    freeze_total_predictions,
    settle_total_predictions,
)
from match_analysis.baseball.domain.canonical_utc import parse_canonical_utc
from match_analysis.baseball.domain.run_line_model import (
    FROZEN_HOME_ADVANTAGE,
    PriorGameRecord,
)
from match_analysis.baseball.domain.total_model import (
    MARKET_ID,
    MODEL_ID,
    MODEL_VERSION,
)


class TotalParityAndLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t0 = parse_canonical_utc("2026-04-01T00:00:00Z")
        self.history = [
            PriorGameRecord("TEX", "STL", 5, 3, self.t0),
            PriorGameRecord("TEX", "LAD", 6, 2, self.t0),
            PriorGameRecord("STL", "LAD", 4, 5, self.t0),
        ]
        self.game_item_85 = TotalPregameInputItem(
            provider_game_id="2026-04-02_TEX_STL",
            home_team="TEX",
            away_team="STL",
            scheduled_start_utc="2026-04-02T19:05:00Z",
            as_of_utc="2026-04-02T12:00:00Z",
            total_line=8.5,
            game_pk=820001,
        )
        self.game_item_90 = TotalPregameInputItem(
            provider_game_id="2026-04-02_TEX_STL",
            home_team="TEX",
            away_team="STL",
            scheduled_start_utc="2026-04-02T19:05:00Z",
            as_of_utc="2026-04-02T12:00:00Z",
            total_line=9.0,
            game_pk=820001,
        )

    def test_lifecycle_freeze_and_settlement_half_line(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p62a-test-") as tmpdir:
            run_dir = Path(tmpdir) / "run_001"
            preds, manifest = freeze_total_predictions(
                [self.game_item_85],
                self.history,
                run_id="run_001",
                output_dir=run_dir,
            )

            self.assertEqual(len(preds), 1)
            pred = preds[0]
            self.assertEqual(pred.market_id, "total")
            self.assertEqual(pred.model_id, MODEL_ID)
            self.assertEqual(pred.model_version, MODEL_VERSION)
            self.assertEqual(pred.total_line, 8.5)
            self.assertEqual(pred.p_push, 0.0)
            self.assertGreater(pred.p_over + pred.p_under, 0.999)
            self.assertEqual(
                pred.predicted_side,
                "OVER" if pred.p_over >= 0.5 else "UNDER",
            )

            # Check written files
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertTrue((run_dir / "predictions.jsonl").is_file())

            # Settle run against final score: TEX 6, STL 4 (actual total = 10 > 8.5 -> OVER)
            results = {"2026-04-02_TEX_STL": (6, 4)}
            settled_rows, summary = settle_total_predictions(
                preds,
                results,
                run_id="run_001",
                output_dir=run_dir,
            )

            self.assertEqual(len(settled_rows), 1)
            self.assertEqual(settled_rows[0]["evaluation"]["actual_result"], "OVER")
            self.assertFalse(settled_rows[0]["evaluation"]["is_push"])
            self.assertEqual(
                settled_rows[0]["evaluation"]["is_correct"],
                pred.predicted_side == "OVER",
            )
            self.assertEqual(summary["total_count"], 1)
            self.assertEqual(summary["push_count"], 0)
            self.assertEqual(summary["decided_count"], 1)
            self.assertEqual(
                summary["correct_count"], 1 if pred.predicted_side == "OVER" else 0
            )

            self.assertTrue((run_dir / "settled_predictions.jsonl").is_file())
            self.assertTrue((run_dir / "settlement_summary.json").is_file())

    def test_lifecycle_freeze_and_settlement_integer_line_push(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p62a-push-") as tmpdir:
            run_dir = Path(tmpdir) / "run_push"
            preds, manifest = freeze_total_predictions(
                [self.game_item_90],
                self.history,
                run_id="run_push",
                output_dir=run_dir,
            )

            self.assertEqual(len(preds), 1)
            pred = preds[0]
            self.assertEqual(pred.total_line, 9.0)
            self.assertGreater(pred.p_push, 0.0)
            self.assertAlmostEqual(
                pred.p_over + pred.p_under + pred.p_push, 1.0, places=5
            )

            # Settle run against final score: TEX 5, STL 4 (actual total = 9 == 9.0 -> PUSH)
            results = {"2026-04-02_TEX_STL": (5, 4)}
            settled_rows, summary = settle_total_predictions(
                preds,
                results,
                run_id="run_push",
                output_dir=run_dir,
            )

            self.assertEqual(len(settled_rows), 1)
            self.assertEqual(settled_rows[0]["evaluation"]["actual_result"], "PUSH")
            self.assertTrue(settled_rows[0]["evaluation"]["is_push"])
            self.assertIsNone(settled_rows[0]["evaluation"]["is_correct"])
            self.assertEqual(summary["total_count"], 1)
            self.assertEqual(summary["push_count"], 1)
            self.assertEqual(summary["decided_count"], 0)
            self.assertIsNone(summary["accuracy"])
            self.assertIsNone(summary["brier_score"])

    def test_identical_replay_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p62a-idempotent-") as tmpdir:
            dir1 = Path(tmpdir) / "run1"
            dir2 = Path(tmpdir) / "run2"

            preds1, m1 = freeze_total_predictions(
                [self.game_item_85], self.history, run_id="run_same", output_dir=dir1
            )
            preds2, m2 = freeze_total_predictions(
                [self.game_item_85], self.history, run_id="run_same", output_dir=dir2
            )

            self.assertEqual(
                (dir1 / "predictions.jsonl").read_bytes(),
                (dir2 / "predictions.jsonl").read_bytes(),
            )
            self.assertEqual(
                preds1[0].prediction_fingerprint, preds2[0].prediction_fingerprint
            )

    def test_changing_total_line_changes_prediction_identity(self) -> None:
        preds_85, _ = freeze_total_predictions(
            [self.game_item_85], self.history, run_id="r85"
        )
        preds_90, _ = freeze_total_predictions(
            [self.game_item_90], self.history, run_id="r90"
        )

        self.assertNotEqual(
            preds_85[0].prediction_fingerprint,
            preds_90[0].prediction_fingerprint,
        )
        self.assertNotEqual(
            preds_85[0].workflow_prediction_id,
            preds_90[0].workflow_prediction_id,
        )
        self.assertNotEqual(
            preds_85[0].p_over,
            preds_90[0].p_over,
        )
        self.assertNotEqual(
            preds_85[0].p_push,
            preds_90[0].p_push,
        )

    def test_post_start_as_of_time_fails_closed(self) -> None:
        late_item = TotalPregameInputItem(
            provider_game_id="2026-04-02_TEX_STL",
            home_team="TEX",
            away_team="STL",
            scheduled_start_utc="2026-04-02T19:05:00Z",
            as_of_utc="2026-04-02T19:05:00Z",
            total_line=8.5,
        )
        with self.assertRaises(ValueError):
            freeze_total_predictions([late_item], self.history, run_id="r_late")


if __name__ == "__main__":
    unittest.main()
