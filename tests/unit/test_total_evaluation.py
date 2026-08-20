"""Unit tests for P62A Total evaluation, push grading, and metrics."""

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.total_evaluation import (
    compute_total_metrics,
    evaluate_total_prediction,
    settle_total,
)


class TotalEvaluationTests(unittest.TestCase):
    def test_settle_total_line_cases(self) -> None:
        # Half-run line 8.5
        self.assertEqual(settle_total(5, 4, 8.5), "OVER")   # 9 > 8.5
        self.assertEqual(settle_total(4, 4, 8.5), "UNDER")  # 8 < 8.5
        self.assertEqual(settle_total(2, 3, 8.5), "UNDER")  # 5 < 8.5

        # Half-run line 7.5
        self.assertEqual(settle_total(4, 4, 7.5), "OVER")   # 8 > 7.5
        self.assertEqual(settle_total(4, 3, 7.5), "UNDER")  # 7 < 7.5

        # Integer line 8.0
        self.assertEqual(settle_total(5, 3, 8.0), "PUSH")   # 8 == 8.0
        self.assertEqual(settle_total(4, 4, 8.0), "PUSH")   # 8 == 8.0
        self.assertEqual(settle_total(5, 4, 8.0), "OVER")   # 9 > 8.0
        self.assertEqual(settle_total(4, 3, 8.0), "UNDER")  # 7 < 8.0

        # Integer line 9.0
        self.assertEqual(settle_total(5, 4, 9.0), "PUSH")   # 9 == 9.0
        self.assertEqual(settle_total(6, 4, 9.0), "OVER")   # 10 > 9.0
        self.assertEqual(settle_total(5, 3, 9.0), "UNDER")  # 8 < 9.0

    def test_evaluate_total_prediction_scoring(self) -> None:
        # Correct OVER prediction
        res_over_correct = evaluate_total_prediction(
            predicted_side="OVER",
            p_over=0.60,
            p_under=0.40,
            total_line=8.5,
            home_score=5,
            away_score=4,
        )
        self.assertFalse(res_over_correct.is_push)
        self.assertTrue(res_over_correct.is_correct)
        self.assertEqual(res_over_correct.correctness_target, 1)
        self.assertAlmostEqual(
            res_over_correct.brier_component, (0.60 - 1.0) ** 2, places=6
        )

        # Incorrect OVER prediction (actual was UNDER)
        res_over_incorrect = evaluate_total_prediction(
            predicted_side="OVER",
            p_over=0.60,
            p_under=0.40,
            total_line=8.5,
            home_score=4,
            away_score=3,
        )
        self.assertFalse(res_over_incorrect.is_push)
        self.assertFalse(res_over_incorrect.is_correct)
        self.assertEqual(res_over_incorrect.correctness_target, 0)
        self.assertAlmostEqual(
            res_over_incorrect.brier_component, (0.60 - 0.0) ** 2, places=6
        )

        # Correct UNDER prediction
        res_under_correct = evaluate_total_prediction(
            predicted_side="UNDER",
            p_over=0.35,
            p_under=0.65,
            total_line=8.5,
            home_score=3,
            away_score=2,
        )
        self.assertFalse(res_under_correct.is_push)
        self.assertTrue(res_under_correct.is_correct)
        self.assertEqual(res_under_correct.correctness_target, 1)
        self.assertAlmostEqual(
            res_under_correct.brier_component, (0.65 - 1.0) ** 2, places=6
        )

        # Push prediction on integer line
        res_push = evaluate_total_prediction(
            predicted_side="OVER",
            p_over=0.55,
            p_under=0.30,
            total_line=8.0,
            home_score=5,
            away_score=3,
        )
        self.assertTrue(res_push.is_push)
        self.assertIsNone(res_push.is_correct)
        self.assertIsNone(res_push.correctness_target)
        self.assertIsNone(res_push.brier_component)
        self.assertIsNone(res_push.log_loss_component)

    def test_metrics_exclude_pushes_from_denominators(self) -> None:
        eval_correct = evaluate_total_prediction("OVER", 0.6, 0.4, 8.5, 5, 4)
        eval_incorrect = evaluate_total_prediction("OVER", 0.6, 0.4, 8.5, 4, 3)
        eval_push = evaluate_total_prediction("OVER", 0.55, 0.30, 8.0, 5, 3)

        metrics = compute_total_metrics([eval_correct, eval_incorrect, eval_push])
        self.assertEqual(metrics.total_count, 3)
        self.assertEqual(metrics.push_count, 1)
        self.assertEqual(metrics.decided_count, 2)
        self.assertEqual(metrics.correct_count, 1)
        self.assertEqual(metrics.incorrect_count, 1)
        self.assertEqual(metrics.accuracy, 0.50)


if __name__ == "__main__":
    unittest.main()
