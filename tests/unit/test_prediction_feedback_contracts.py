"""Unit tests for prediction feedback domain contracts and validation."""

from decimal import Decimal
import unittest

from match_analysis.baseball.domain.prediction_feedback import (
    FEEDBACK_LEDGER_SCHEMA_VERSION,
    FEEDBACK_ROW_SCHEMA_VERSION,
    FEEDBACK_STATUS_EVALUATED,
    FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED,
    PredictionFeedbackRow,
    compute_feedback_ledger_fingerprint,
    compute_feedback_row_fingerprint,
)


class TestPredictionFeedbackContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.pred_id = "0cabf8e0dbc4a79013bad2c8287ea7fe2ef91ee8ae9d8574629656c25166e917"
        self.snap_fp = "959dd164cbd41e362812c20bacbd078410425572120b0811f728e1f1ec59a248"
        self.att_fp = "ea5fd456e306cd1297292dd6e38429af81589ccb7884e0cbb0812ddcfdcee705"
        self.eval_fp = "5e11469b247439235f00fe2da9a75dedf913dcf5963f9bd73c90e155d4d652bb"

        self.provider_ns = "MLB_STATS_API"
        self.provider_gid = "888001"
        self.game_num = 1
        self.sched_start = "2026-04-05T19:00:00Z"
        self.model_id = "model_v1"
        self.market_id = "moneyline"
        self.selection = "HOME"
        self.model_prob = Decimal("0.58")
        self.obs_payload = {
            "game_number": 1,
            "market_id": "moneyline",
            "model_id": "model_v1",
            "model_probability": "0.58",
            "prediction_observation_id": self.pred_id,
            "provider_game_id": "888001",
            "provider_namespace": "MLB_STATS_API",
            "scheduled_start_utc": "2026-04-05T19:00:00Z",
            "selection": "HOME",
        }

        self.result_id = "0955a1ed76c9538ea4120780929eb801850a370beffd0a6a76a2c4ecde9aeea7"
        self.result_at = "2026-04-05T22:15:00Z"
        self.home_score = 5
        self.away_score = 3
        self.actual_winner = "HOME"
        self.is_correct = True
        self.target = 1
        self.brier = Decimal("0.1764")

        self.evaluated_row_fp = compute_feedback_row_fingerprint(
            prediction_observation_id=self.pred_id,
            source_snapshot_row_fingerprint=self.snap_fp,
            source_attachment_row_fingerprint=self.att_fp,
            source_evaluation_row_fingerprint=self.eval_fp,
            provider_namespace=self.provider_ns,
            provider_game_id=self.provider_gid,
            game_number=self.game_num,
            scheduled_start_utc=self.sched_start,
            model_id=self.model_id,
            market_id=self.market_id,
            selection=self.selection,
            model_probability=self.model_prob,
            result_observation_id=self.result_id,
            result_observed_at_utc=self.result_at,
            home_score=self.home_score,
            away_score=self.away_score,
            actual_winner=self.actual_winner,
            attachment_status="ATTACHED",
            attachment_rejection_reason=None,
            feedback_status=FEEDBACK_STATUS_EVALUATED,
            is_correct=self.is_correct,
            correctness_target=self.target,
            brier_component=self.brier,
        )

        self.rejected_row_fp = compute_feedback_row_fingerprint(
            prediction_observation_id=self.pred_id,
            source_snapshot_row_fingerprint=self.snap_fp,
            source_attachment_row_fingerprint=self.att_fp,
            source_evaluation_row_fingerprint=None,
            provider_namespace=self.provider_ns,
            provider_game_id=self.provider_gid,
            game_number=self.game_num,
            scheduled_start_utc=self.sched_start,
            model_id=self.model_id,
            market_id=self.market_id,
            selection=self.selection,
            model_probability=self.model_prob,
            result_observation_id=None,
            result_observed_at_utc=None,
            home_score=None,
            away_score=None,
            actual_winner=None,
            attachment_status="REJECTED",
            attachment_rejection_reason="NO_RESULT_OBSERVATION",
            feedback_status=FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED,
            is_correct=None,
            correctness_target=None,
            brier_component=None,
        )

    def test_valid_evaluated_row_construction(self) -> None:
        row = PredictionFeedbackRow(
            prediction_observation_id=self.pred_id,
            source_snapshot_row_fingerprint=self.snap_fp,
            source_attachment_row_fingerprint=self.att_fp,
            source_evaluation_row_fingerprint=self.eval_fp,
            provider_namespace=self.provider_ns,
            provider_game_id=self.provider_gid,
            game_number=self.game_num,
            scheduled_start_utc=self.sched_start,
            model_id=self.model_id,
            market_id=self.market_id,
            selection=self.selection,
            model_probability=self.model_prob,
            observation_payload=self.obs_payload,
            result_observation_id=self.result_id,
            result_observed_at_utc=self.result_at,
            home_score=self.home_score,
            away_score=self.away_score,
            actual_winner=self.actual_winner,
            attachment_status="ATTACHED",
            attachment_rejection_reason=None,
            feedback_status=FEEDBACK_STATUS_EVALUATED,
            is_correct=self.is_correct,
            correctness_target=self.target,
            brier_component=self.brier,
            feedback_row_fingerprint=self.evaluated_row_fp,
        )
        self.assertEqual(row.prediction_observation_id, self.pred_id)
        self.assertEqual(row.feedback_status, FEEDBACK_STATUS_EVALUATED)
        self.assertEqual(row.home_score, 5)
        self.assertEqual(row.is_correct, True)
        self.assertEqual(row.brier_component, Decimal("0.1764"))

    def test_valid_rejected_row_construction(self) -> None:
        row = PredictionFeedbackRow(
            prediction_observation_id=self.pred_id,
            source_snapshot_row_fingerprint=self.snap_fp,
            source_attachment_row_fingerprint=self.att_fp,
            source_evaluation_row_fingerprint=None,
            provider_namespace=self.provider_ns,
            provider_game_id=self.provider_gid,
            game_number=self.game_num,
            scheduled_start_utc=self.sched_start,
            model_id=self.model_id,
            market_id=self.market_id,
            selection=self.selection,
            model_probability=self.model_prob,
            observation_payload=self.obs_payload,
            result_observation_id=None,
            result_observed_at_utc=None,
            home_score=None,
            away_score=None,
            actual_winner=None,
            attachment_status="REJECTED",
            attachment_rejection_reason="NO_RESULT_OBSERVATION",
            feedback_status=FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED,
            is_correct=None,
            correctness_target=None,
            brier_component=None,
            feedback_row_fingerprint=self.rejected_row_fp,
        )
        self.assertEqual(row.prediction_observation_id, self.pred_id)
        self.assertEqual(
            row.feedback_status, FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED
        )
        self.assertIsNone(row.home_score)
        self.assertIsNone(row.is_correct)
        self.assertIsNone(row.brier_component)
        self.assertEqual(row.attachment_rejection_reason, "NO_RESULT_OBSERVATION")

    def test_invalid_feedback_status_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionFeedbackRow(
                prediction_observation_id=self.pred_id,
                source_snapshot_row_fingerprint=self.snap_fp,
                source_attachment_row_fingerprint=self.att_fp,
                source_evaluation_row_fingerprint=self.eval_fp,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                scheduled_start_utc=self.sched_start,
                model_id=self.model_id,
                market_id=self.market_id,
                selection=self.selection,
                model_probability=self.model_prob,
                observation_payload=self.obs_payload,
                result_observation_id=self.result_id,
                result_observed_at_utc=self.result_at,
                home_score=self.home_score,
                away_score=self.away_score,
                actual_winner=self.actual_winner,
                attachment_status="ATTACHED",
                attachment_rejection_reason=None,
                feedback_status="UNKNOWN_STATUS",
                is_correct=self.is_correct,
                correctness_target=self.target,
                brier_component=self.brier,
                feedback_row_fingerprint=self.evaluated_row_fp,
            )
        self.assertIn("feedback_status must be one of", str(ctx.exception))

    def test_evaluated_with_rejected_status_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionFeedbackRow(
                prediction_observation_id=self.pred_id,
                source_snapshot_row_fingerprint=self.snap_fp,
                source_attachment_row_fingerprint=self.att_fp,
                source_evaluation_row_fingerprint=self.eval_fp,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                scheduled_start_utc=self.sched_start,
                model_id=self.model_id,
                market_id=self.market_id,
                selection=self.selection,
                model_probability=self.model_prob,
                observation_payload=self.obs_payload,
                result_observation_id=self.result_id,
                result_observed_at_utc=self.result_at,
                home_score=self.home_score,
                away_score=self.away_score,
                actual_winner=self.actual_winner,
                attachment_status="REJECTED",
                attachment_rejection_reason=None,
                feedback_status=FEEDBACK_STATUS_EVALUATED,
                is_correct=self.is_correct,
                correctness_target=self.target,
                brier_component=self.brier,
                feedback_row_fingerprint=self.evaluated_row_fp,
            )
        self.assertIn(
            "EVALUATED feedback requires attachment_status == 'ATTACHED'",
            str(ctx.exception),
        )

    def test_rejected_with_partial_results_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionFeedbackRow(
                prediction_observation_id=self.pred_id,
                source_snapshot_row_fingerprint=self.snap_fp,
                source_attachment_row_fingerprint=self.att_fp,
                source_evaluation_row_fingerprint=None,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                scheduled_start_utc=self.sched_start,
                model_id=self.model_id,
                market_id=self.market_id,
                selection=self.selection,
                model_probability=self.model_prob,
                observation_payload=self.obs_payload,
                result_observation_id=self.result_id,
                result_observed_at_utc=None,
                home_score=None,
                away_score=None,
                actual_winner=None,
                attachment_status="REJECTED",
                attachment_rejection_reason="NO_RESULT_OBSERVATION",
                feedback_status=FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED,
                is_correct=None,
                correctness_target=None,
                brier_component=None,
                feedback_row_fingerprint=self.rejected_row_fp,
            )
        self.assertIn(
            "RESULT_ATTACHMENT_REJECTED feedback must have null result_observation_id",
            str(ctx.exception),
        )

    def test_fingerprint_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            PredictionFeedbackRow(
                prediction_observation_id=self.pred_id,
                source_snapshot_row_fingerprint=self.snap_fp,
                source_attachment_row_fingerprint=self.att_fp,
                source_evaluation_row_fingerprint=self.eval_fp,
                provider_namespace=self.provider_ns,
                provider_game_id=self.provider_gid,
                game_number=self.game_num,
                scheduled_start_utc=self.sched_start,
                model_id=self.model_id,
                market_id=self.market_id,
                selection=self.selection,
                model_probability=self.model_prob,
                observation_payload=self.obs_payload,
                result_observation_id=self.result_id,
                result_observed_at_utc=self.result_at,
                home_score=self.home_score,
                away_score=self.away_score,
                actual_winner=self.actual_winner,
                attachment_status="ATTACHED",
                attachment_rejection_reason=None,
                feedback_status=FEEDBACK_STATUS_EVALUATED,
                is_correct=self.is_correct,
                correctness_target=self.target,
                brier_component=self.brier,
                feedback_row_fingerprint="0000000000000000000000000000000000000000000000000000000000000000",
            )
        self.assertIn("feedback_row_fingerprint mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
