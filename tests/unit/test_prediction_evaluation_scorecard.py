"""Unit tests for building prediction evaluation scorecards."""

from decimal import Decimal
from pathlib import Path
import unittest

from match_analysis.application.use_cases.build_prediction_evaluation_scorecard import (
    build_prediction_evaluation_scorecard,
)
from match_analysis.baseball.domain.prediction_evaluation import (
    PredictionEvaluationRow,
    build_scorecard,
    compute_evaluation_row_fingerprint,
)


class TestPredictionEvaluationScorecard(unittest.TestCase):
    def setUp(self) -> None:
        self.p16a_dir = Path("report/p16a_final_result_attachment")
        self.p15c_dir = Path("report/p15c_admitted_prediction_observation_snapshot")

        self.attachments_bytes = (self.p16a_dir / "attachments.jsonl").read_bytes()
        self.att_summary_bytes = (self.p16a_dir / "summary.json").read_bytes()
        self.snapshot_bytes = (self.p15c_dir / "admitted_observations.jsonl").read_bytes()
        self.snap_summary_bytes = (self.p15c_dir / "summary.json").read_bytes()

        self.att_fp = "fe6c96a18f7606869c30f8acdd21a525ab75ed3289bfc7af2c940fc3776c3772"

    def test_build_scorecard_empty_rows(self) -> None:
        sc = build_scorecard((), self.att_fp)
        self.assertEqual(sc.evaluation_row_count, 0)
        self.assertEqual(sc.correct_count, 0)
        self.assertEqual(sc.incorrect_count, 0)
        self.assertEqual(sc.accuracy, Decimal("0"))
        self.assertEqual(sc.mean_selected_side_probability, Decimal("0"))
        self.assertEqual(sc.brier_score, Decimal("0"))
        self.assertEqual(sc.breakdown_by_model_id, {})

    def test_build_scorecard_unsorted_rows_raises(self) -> None:
        pred1 = "0cabf8e0dbc4a79013bad2c8287ea7fe2ef91ee8ae9d8574629656c25166e917"
        pred2 = "573447cd9d62bff6bd13448f1c6f6c8b8e4301f8d070a056c1f23ffb3602f5ec"

        fp1 = compute_evaluation_row_fingerprint(
            prediction_observation_id=pred1,
            source_attachment_row_fingerprint="ea5fd456e306cd1297292dd6e38429af81589ccb7884e0cbb0812ddcfdcee705",
            model_id="model_v1",
            market_id="moneyline",
            selection="HOME",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            model_probability=Decimal("0.58"),
            actual_winner="HOME",
            is_correct=True,
            correctness_target=1,
            brier_component=Decimal("0.1764"),
        )
        fp2 = compute_evaluation_row_fingerprint(
            prediction_observation_id=pred2,
            source_attachment_row_fingerprint="e67391dc7b581d43f3a2d48928164b86ec8546cf8d6e7be2db34a46c6723eb58",
            model_id="model_v1",
            market_id="moneyline",
            selection="AWAY",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888002",
            game_number=1,
            model_probability=Decimal("0.52"),
            actual_winner="AWAY",
            is_correct=True,
            correctness_target=1,
            brier_component=Decimal("0.2304"),
        )

        row1 = PredictionEvaluationRow(
            prediction_observation_id=pred1,
            source_attachment_row_fingerprint="ea5fd456e306cd1297292dd6e38429af81589ccb7884e0cbb0812ddcfdcee705",
            model_id="model_v1",
            market_id="moneyline",
            selection="HOME",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            model_probability=Decimal("0.58"),
            actual_winner="HOME",
            is_correct=True,
            correctness_target=1,
            brier_component=Decimal("0.1764"),
            evaluation_row_fingerprint=fp1,
        )
        row2 = PredictionEvaluationRow(
            prediction_observation_id=pred2,
            source_attachment_row_fingerprint="e67391dc7b581d43f3a2d48928164b86ec8546cf8d6e7be2db34a46c6723eb58",
            model_id="model_v1",
            market_id="moneyline",
            selection="AWAY",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888002",
            game_number=1,
            model_probability=Decimal("0.52"),
            actual_winner="AWAY",
            is_correct=True,
            correctness_target=1,
            brier_component=Decimal("0.2304"),
            evaluation_row_fingerprint=fp2,
        )

        # Pass in reverse order
        with self.assertRaises(ValueError) as ctx:
            build_scorecard((row2, row1), self.att_fp)
        self.assertIn("rows must be sorted by prediction_observation_id", str(ctx.exception))

    def test_build_scorecard_duplicate_rows_raises(self) -> None:
        pred1 = "0cabf8e0dbc4a79013bad2c8287ea7fe2ef91ee8ae9d8574629656c25166e917"
        fp1 = compute_evaluation_row_fingerprint(
            prediction_observation_id=pred1,
            source_attachment_row_fingerprint="ea5fd456e306cd1297292dd6e38429af81589ccb7884e0cbb0812ddcfdcee705",
            model_id="model_v1",
            market_id="moneyline",
            selection="HOME",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            model_probability=Decimal("0.58"),
            actual_winner="HOME",
            is_correct=True,
            correctness_target=1,
            brier_component=Decimal("0.1764"),
        )
        row1 = PredictionEvaluationRow(
            prediction_observation_id=pred1,
            source_attachment_row_fingerprint="ea5fd456e306cd1297292dd6e38429af81589ccb7884e0cbb0812ddcfdcee705",
            model_id="model_v1",
            market_id="moneyline",
            selection="HOME",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            model_probability=Decimal("0.58"),
            actual_winner="HOME",
            is_correct=True,
            correctness_target=1,
            brier_component=Decimal("0.1764"),
            evaluation_row_fingerprint=fp1,
        )
        with self.assertRaises(ValueError) as ctx:
            build_scorecard((row1, row1), self.att_fp)
        self.assertIn("duplicate prediction_observation_id", str(ctx.exception))

    def test_build_prediction_evaluation_scorecard_success(self) -> None:
        result = build_prediction_evaluation_scorecard(
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=self.att_summary_bytes,
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
        )

        self.assertEqual(result.source_row_count, 3)
        self.assertEqual(result.source_attached_count, 3)
        self.assertEqual(result.source_rejected_count, 0)
        self.assertEqual(result.evaluation_row_count, 3)
        self.assertEqual(result.excluded_rejected_count, 0)
        self.assertEqual(result.correct_count, 2)
        self.assertEqual(result.incorrect_count, 1)

        self.assertAlmostEqual(result.accuracy, 0.666667, places=6)
        self.assertAlmostEqual(result.mean_selected_side_probability, 0.566667, places=6)
        self.assertAlmostEqual(result.brier_score, 0.255600, places=6)

        # Check breakdowns
        self.assertIn("model_v1", result.scorecard.breakdown_by_model_id)
        self.assertIn("moneyline", result.scorecard.breakdown_by_market_id)
        self.assertIn("HOME", result.scorecard.breakdown_by_selection)
        self.assertIn("AWAY", result.scorecard.breakdown_by_selection)

        home_b = result.scorecard.breakdown_by_selection["HOME"]
        self.assertEqual(home_b.row_count, 2)
        self.assertEqual(home_b.correct_count, 1)
        self.assertEqual(home_b.incorrect_count, 1)

        away_b = result.scorecard.breakdown_by_selection["AWAY"]
        self.assertEqual(away_b.row_count, 1)
        self.assertEqual(away_b.correct_count, 1)

        # Check claims
        self.assertTrue(result.claims["synthetic_results"])
        self.assertTrue(result.claims["sample_limited"])
        self.assertFalse(result.claims["real_model_performance_claim"])
        self.assertFalse(result.claims["retraining_performed"])

    def test_attachments_sha256_mismatch_raises(self) -> None:
        tampered_bytes = self.attachments_bytes + b"\n"
        with self.assertRaises(ValueError) as ctx:
            build_prediction_evaluation_scorecard(
                attachments_bytes=tampered_bytes,
                attachment_summary_bytes=self.att_summary_bytes,
                snapshot_bytes=self.snapshot_bytes,
                snapshot_summary_bytes=self.snap_summary_bytes,
            )
        self.assertIn("Attachments SHA-256 mismatch", str(ctx.exception))

    def test_snapshot_sha256_mismatch_raises(self) -> None:
        tampered_bytes = self.snapshot_bytes + b"\n"
        with self.assertRaises(ValueError) as ctx:
            build_prediction_evaluation_scorecard(
                attachments_bytes=self.attachments_bytes,
                attachment_summary_bytes=self.att_summary_bytes,
                snapshot_bytes=tampered_bytes,
                snapshot_summary_bytes=self.snap_summary_bytes,
            )
        self.assertIn("Snapshot SHA-256 mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
