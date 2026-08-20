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
            as_of_utc="2026-04-02T12:00:00Z",
            spread_home=-1.5,
            game_pk=820001,
        )
        self.game_item_dog = RunLinePregameInputItem(
            provider_game_id="2026-04-02_TEX_STL",
            home_team="TEX",
            away_team="STL",
            scheduled_start_utc="2026-04-02T19:05:00Z",
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
            as_of_utc="2026-04-02T19:05:00Z",
            spread_home=-1.5,
        )
        with self.assertRaises(ValueError):
            freeze_run_line_predictions([late_item], self.history, run_id="r_late")


if __name__ == "__main__":
    unittest.main()
