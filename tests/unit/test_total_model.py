"""Unit tests for P62A Total Poisson team-rate model and point-in-time state."""

from datetime import datetime, timezone
from decimal import Decimal
import math
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.run_line_model import (
    DEFAULT_LEAGUE_AVG_RUNS,
    FROZEN_HOME_ADVANTAGE,
    POISSON_MAX_RUNS,
    RUN_SMOOTH_K,
    PriorGameRecord,
    TeamRateState,
    compute_poisson_lambdas,
    compute_team_rate_state,
    compute_team_rates,
    poisson_pmf_array,
)
from match_analysis.baseball.domain.total_model import (
    MARKET_ID,
    MODEL_ID,
    MODEL_VERSION,
    TotalProbabilities,
    compute_total_probabilities,
)


class TotalModelTests(unittest.TestCase):
    def test_total_model_metadata(self) -> None:
        self.assertEqual(MODEL_ID, "poisson_team_rate_model")
        self.assertEqual(MODEL_VERSION, "p226a.raw.v1")
        self.assertEqual(MARKET_ID, "total")

    def test_total_probabilities_half_run_lines_have_zero_push(self) -> None:
        # Total line 8.5
        probs_85 = compute_total_probabilities(4.8, 3.8, 8.5)
        self.assertEqual(probs_85.p_push, 0.0)
        self.assertAlmostEqual(probs_85.p_over + probs_85.p_under, 1.0, places=5)
        self.assertEqual(
            probs_85.predicted_side,
            "OVER" if probs_85.p_over >= 0.5 else "UNDER",
        )
        self.assertEqual(
            probs_85.selected_total_probability,
            probs_85.p_over if probs_85.predicted_side == "OVER" else probs_85.p_under,
        )

        # Total line 7.5
        probs_75 = compute_total_probabilities(4.8, 3.8, 7.5)
        self.assertEqual(probs_75.p_push, 0.0)
        self.assertAlmostEqual(probs_75.p_over + probs_75.p_under, 1.0, places=5)
        self.assertEqual(
            probs_75.predicted_side,
            "OVER" if probs_75.p_over >= 0.5 else "UNDER",
        )

    def test_total_probabilities_integer_lines_support_push(self) -> None:
        # Total line 8.0
        probs_80 = compute_total_probabilities(4.5, 4.0, 8.0)
        self.assertGreater(probs_80.p_push, 0.0)
        self.assertAlmostEqual(
            probs_80.p_over + probs_80.p_under + probs_80.p_push,
            1.0,
            places=5,
        )

        # Total line 9.0
        probs_90 = compute_total_probabilities(4.8, 4.2, 9.0)
        self.assertGreater(probs_90.p_push, 0.0)
        self.assertAlmostEqual(
            probs_90.p_over + probs_90.p_under + probs_90.p_push,
            1.0,
            places=5,
        )

    def test_predicted_side_threshold_is_exactly_half(self) -> None:
        # Lambdas sum to ~8.0, line 7.5 -> p_over > 0.5 -> OVER
        probs_over = compute_total_probabilities(4.5, 4.0, 7.5)
        self.assertGreaterEqual(probs_over.p_over, 0.5)
        self.assertEqual(probs_over.predicted_side, "OVER")

        # Lambdas sum to ~8.0, line 9.5 -> p_over < 0.5 -> UNDER
        probs_under = compute_total_probabilities(4.0, 4.0, 9.5)
        self.assertLess(probs_under.p_over, 0.5)
        self.assertEqual(probs_under.predicted_side, "UNDER")

    def test_total_probabilities_sum_to_one_across_various_lines(self) -> None:
        test_cases = [
            (3.0, 3.0, 6.5),
            (4.2, 4.8, 8.0),
            (5.5, 4.0, 9.5),
            (6.0, 5.0, 11.0),
            (2.5, 2.5, 5.0),
        ]
        for lh, la, line in test_cases:
            probs = compute_total_probabilities(lh, la, line)
            self.assertGreaterEqual(probs.p_over, 0.0)
            self.assertGreaterEqual(probs.p_under, 0.0)
            self.assertGreaterEqual(probs.p_push, 0.0)
            self.assertAlmostEqual(
                probs.p_over + probs.p_under + probs.p_push,
                1.0,
                places=5,
            )


if __name__ == "__main__":
    unittest.main()
