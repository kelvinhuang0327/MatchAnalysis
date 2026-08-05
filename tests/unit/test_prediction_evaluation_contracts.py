"""Unit tests for prediction evaluation domain contracts and validation."""

from decimal import Decimal
import unittest

from match_analysis.baseball.domain.prediction_evaluation import (
    EVALUATION_ROW_SCHEMA_VERSION,
    SCHEMA_VERSION,
    BreakdownMetrics,
    PredictionEvaluationRow,
    compute_evaluation_row_fingerprint,
    compute_evaluation_set_fingerprint,
)


class TestPredictionEvaluationContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.pred_id = "0cabf8e0dbc4a79013bad2c8287ea7fe2ef91ee8ae9d8574629656c25166e917"
        self.att_fp = "ea5fd456e306cd1297292dd6e38429af81589ccb7884e0cbb0812ddcfdcee705"
        self.model_id = "model_v1"
        self.market_id = "moneyline"
        self.selection = "HOME"
        self.provider_ns = "MLB_STATS_API"
        self.provider_gid = "888001"
        self.game_num = 1
        self.model_prob = Decimal("0.58")
        self.actual_winner = "HOME"
        self.is_correct = True
        self.target = 1
        self.brier = Decimal("0.1764")

        self.eval_fp = compute_evaluation_row_fingerprint(
            prediction_observation_id=self.pred_id,
            source_attachment_row_fingerprint=self.att_fp,
            model_id=self.model_id,
            market_id=self.market_id,
            selection=self.selection,
            provider_namespace=self.provider_ns,
            provider_game_id=self.provider_gid,
            game_number=self.game_num,
            model_probability=self.model_prob,
            actual_winner=self.actual_winner,
            is_correct=self.is_correct,
            correctness_target=self.target,
            brier_component=self.brier,
        )

    def test_valid_row_construction(self) -> None:
        row = PredictionEvaluationRow(
            prediction_observation_id=self.pred_id,
            source_attachment_row_fingerprint=self.att_fp,
            model_id=self.model_id,
            market_id=self.market_id,
            selection=self.selection,
            provider_namespace=self.provider_ns,
            provider_game_id=self.provider_gid,
            game_number=self.game_num,
            model_probability=self.model_prob,
            actual_winner=self.actual_winner,
            is_correct=self.is_correct,
            correctness_target=self.target,
            brier_component=self.brier,
            evaluation_row_fingerprint=self.eval_fp,
        )
        self.assertEqual(row.prediction_observation_id, self.pred_id)
        self.assertEqual(row.model_probability, Decimal("0.58"))
        self.assertEqual(row.correctness_target, 1)
        self.assertEqual(row.brier_component, Decimal("0.1764"))

    def test_invalid_market_id_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionEvaluationRow(
                prediction_observation_id=self.pred_id,
                source_attachment_row_fingerprint=self.att_fp,
                model_id=self.model_id,
                market_id="spread",
                selection=self.selection,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                model_probability=self.model_prob,
                actual_winner=self.actual_winner,
                is_correct=self.is_correct,
                correctness_target=self.target,
                brier_component=self.brier,
                evaluation_row_fingerprint=self.eval_fp,
            )
        self.assertIn("market_id must be 'moneyline'", str(ctx.exception))

    def test_invalid_selection_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionEvaluationRow(
                prediction_observation_id=self.pred_id,
                source_attachment_row_fingerprint=self.att_fp,
                model_id=self.model_id,
                market_id=self.market_id,
                selection="DRAW",
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                model_probability=self.model_prob,
                actual_winner=self.actual_winner,
                is_correct=self.is_correct,
                correctness_target=self.target,
                brier_component=self.brier,
                evaluation_row_fingerprint=self.eval_fp,
            )
        self.assertIn("selection must be 'HOME' or 'AWAY'", str(ctx.exception))

    def test_invalid_actual_winner_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionEvaluationRow(
                prediction_observation_id=self.pred_id,
                source_attachment_row_fingerprint=self.att_fp,
                model_id=self.model_id,
                market_id=self.market_id,
                selection=self.selection,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                model_probability=self.model_prob,
                actual_winner="TIE",
                is_correct=self.is_correct,
                correctness_target=self.target,
                brier_component=self.brier,
                evaluation_row_fingerprint=self.eval_fp,
            )
        self.assertIn("actual_winner must be 'HOME' or 'AWAY'", str(ctx.exception))

    def test_invalid_correctness_target_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionEvaluationRow(
                prediction_observation_id=self.pred_id,
                source_attachment_row_fingerprint=self.att_fp,
                model_id=self.model_id,
                market_id=self.market_id,
                selection=self.selection,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                model_probability=self.model_prob,
                actual_winner=self.actual_winner,
                is_correct=self.is_correct,
                correctness_target=0,
                brier_component=self.brier,
                evaluation_row_fingerprint=self.eval_fp,
            )
        self.assertIn("correctness_target must be 1 for is_correct=True", str(ctx.exception))

    def test_probability_out_of_bounds_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionEvaluationRow(
                prediction_observation_id=self.pred_id,
                source_attachment_row_fingerprint=self.att_fp,
                model_id=self.model_id,
                market_id=self.market_id,
                selection=self.selection,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                model_probability=Decimal("1.5"),
                actual_winner=self.actual_winner,
                is_correct=self.is_correct,
                correctness_target=self.target,
                brier_component=Decimal("0.25"),
                evaluation_row_fingerprint=self.eval_fp,
            )
        self.assertIn("model_probability must be within [0, 1]", str(ctx.exception))

    def test_incorrect_brier_component_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionEvaluationRow(
                prediction_observation_id=self.pred_id,
                source_attachment_row_fingerprint=self.att_fp,
                model_id=self.model_id,
                market_id=self.market_id,
                selection=self.selection,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                model_probability=self.model_prob,
                actual_winner=self.actual_winner,
                is_correct=self.is_correct,
                correctness_target=self.target,
                brier_component=Decimal("0.5000"),
                evaluation_row_fingerprint=self.eval_fp,
            )
        self.assertIn("brier_component must equal", str(ctx.exception))

    def test_fingerprint_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionEvaluationRow(
                prediction_observation_id=self.pred_id,
                source_attachment_row_fingerprint=self.att_fp,
                model_id=self.model_id,
                market_id=self.market_id,
                selection=self.selection,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                model_probability=self.model_prob,
                actual_winner=self.actual_winner,
                is_correct=self.is_correct,
                correctness_target=self.target,
                brier_component=self.brier,
                evaluation_row_fingerprint="0000000000000000000000000000000000000000000000000000000000000000",
            )
        self.assertIn("evaluation_row_fingerprint mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
