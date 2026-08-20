"""Unit tests for P61B Run Line evaluation, push grading, and metrics."""

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.run_line_evaluation import (
    compute_run_line_metrics,
    evaluate_run_line_prediction,
    settle_run_line,
)


class RunLineEvaluationTests(unittest.TestCase):
    def test_settle_run_line_spread_cases(self) -> None:
        # Home favorite -1.5 (threshold = 1.5)
        self.assertEqual(settle_run_line(5, 3, -1.5), "HOME")   # diff = 2 > 1.5
        self.assertEqual(settle_run_line(4, 3, -1.5), "AWAY")   # diff = 1 < 1.5
        self.assertEqual(settle_run_line(2, 5, -1.5), "AWAY")   # diff = -3 < 1.5

        # Home underdog +1.5 (threshold = -1.5)
        self.assertEqual(settle_run_line(5, 3, 1.5), "HOME")    # diff = 2 > -1.5
        self.assertEqual(settle_run_line(3, 4, 1.5), "HOME")    # diff = -1 > -1.5
        self.assertEqual(settle_run_line(2, 4, 1.5), "AWAY")    # diff = -2 < -1.5

        # Integer line -1.0 (threshold = 1.0)
        self.assertEqual(settle_run_line(4, 3, -1.0), "PUSH")   # diff = 1 == 1.0
        self.assertEqual(settle_run_line(5, 3, -1.0), "HOME")   # diff = 2 > 1.0

        # PK line 0.0 (threshold = 0.0)
        self.assertEqual(settle_run_line(4, 4, 0.0), "PUSH")
        self.assertEqual(settle_run_line(5, 4, 0.0), "HOME")
        self.assertEqual(settle_run_line(3, 4, 0.0), "AWAY")

    def test_evaluate_run_line_prediction_scoring(self) -> None:
        # Correct prediction
        res_correct = evaluate_run_line_prediction(
            predicted_side="HOME",
            p_home_covers=0.60,
            p_away_covers=0.40,
            spread_home=-1.5,
            home_score=5,
            away_score=3,
        )
        self.assertFalse(res_correct.is_push)
        self.assertTrue(res_correct.is_correct)
        self.assertEqual(res_correct.correctness_target, 1)
        self.assertAlmostEqual(res_correct.brier_component, (0.60 - 1.0) ** 2, places=6)

        # Incorrect prediction
        res_incorrect = evaluate_run_line_prediction(
            predicted_side="HOME",
            p_home_covers=0.60,
            p_away_covers=0.40,
            spread_home=-1.5,
            home_score=4,
            away_score=3,
        )
        self.assertFalse(res_incorrect.is_push)
        self.assertFalse(res_incorrect.is_correct)
        self.assertEqual(res_incorrect.correctness_target, 0)
        self.assertAlmostEqual(res_incorrect.brier_component, (0.60 - 0.0) ** 2, places=6)

        # Push prediction
        res_push = evaluate_run_line_prediction(
            predicted_side="HOME",
            p_home_covers=0.55,
            p_away_covers=0.35,
            spread_home=-1.0,
            home_score=4,
            away_score=3,
        )
        self.assertTrue(res_push.is_push)
        self.assertIsNone(res_push.is_correct)
        self.assertIsNone(res_push.brier_component)
        self.assertIsNone(res_push.log_loss_component)

    def test_metrics_exclude_pushes_from_denominators(self) -> None:
        eval_correct = evaluate_run_line_prediction("HOME", 0.6, 0.4, -1.5, 5, 3)
        eval_incorrect = evaluate_run_line_prediction("HOME", 0.6, 0.4, -1.5, 4, 3)
        eval_push = evaluate_run_line_prediction("HOME", 0.55, 0.35, -1.0, 4, 3)

        metrics = compute_run_line_metrics([eval_correct, eval_incorrect, eval_push])
        self.assertEqual(metrics.total_count, 3)
        self.assertEqual(metrics.push_count, 1)
        self.assertEqual(metrics.decided_count, 2)
        self.assertEqual(metrics.correct_count, 1)
        self.assertEqual(metrics.incorrect_count, 1)
        self.assertEqual(metrics.accuracy, 0.50)


if __name__ == "__main__":
    unittest.main()
