"""Unit tests for P61B Run Line Poisson team-rate model and point-in-time state."""

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
    compute_run_line_probabilities,
    compute_team_rate_state,
    compute_team_rates,
    poisson_pmf_array,
)


class RunLineModelTests(unittest.TestCase):
    def test_empty_prior_history_falls_back_to_default_league_average(self) -> None:
        cutoff = datetime(2026, 4, 1, 12, tzinfo=timezone.utc)
        states, avg = compute_team_rate_state([], cutoff)
        self.assertEqual(states, {})
        self.assertEqual(avg, DEFAULT_LEAGUE_AVG_RUNS)

    def test_point_in_time_filtering_excludes_post_cutoff_games(self) -> None:
        t1 = datetime(2026, 4, 1, 10, tzinfo=timezone.utc)
        t2 = datetime(2026, 4, 2, 10, tzinfo=timezone.utc)
        t3 = datetime(2026, 4, 3, 10, tzinfo=timezone.utc)

        history = [
            PriorGameRecord("TEX", "STL", 5, 3, t1),
            PriorGameRecord("TEX", "LAD", 2, 8, t2),
            PriorGameRecord("STL", "LAD", 4, 1, t3),
        ]

        # Cutoff between t1 and t2
        states, avg = compute_team_rate_state(history, datetime(2026, 4, 1, 20, tzinfo=timezone.utc))
        self.assertEqual(len(states), 2)
        self.assertEqual(states["TEX"], TeamRateState(runs_for=5.0, runs_against=3.0, games_played=1))
        self.assertEqual(states["STL"], TeamRateState(runs_for=3.0, runs_against=5.0, games_played=1))
        self.assertEqual(avg, 4.0)

        # Cutoff after t2 but before t3
        states2, avg2 = compute_team_rate_state(history, datetime(2026, 4, 2, 20, tzinfo=timezone.utc))
        self.assertEqual(len(states2), 3)
        self.assertEqual(states2["TEX"], TeamRateState(runs_for=7.0, runs_against=11.0, games_played=2))
        self.assertEqual(avg2, (5 + 3 + 2 + 8) / (2.0 * 2))

    def test_team_rates_shrinkage_formula(self) -> None:
        state = TeamRateState(runs_for=20.0, runs_against=10.0, games_played=4)
        league_avg = 4.5
        off, def_rate = compute_team_rates(state, league_avg, k=10.0)

        # off = (20 + 10 * 4.5) / (4 + 10) = 65 / 14 = 4.642857...
        # def = (10 + 10 * 4.5) / (4 + 10) = 55 / 14 = 3.928571...
        self.assertAlmostEqual(off, 65.0 / 14.0, places=6)
        self.assertAlmostEqual(def_rate, 55.0 / 14.0, places=6)

    def test_poisson_lambdas_apply_frozen_home_advantage(self) -> None:
        team_states = {
            "HOME_TEAM": TeamRateState(runs_for=50.0, runs_against=40.0, games_played=10),
            "AWAY_TEAM": TeamRateState(runs_for=40.0, runs_against=50.0, games_played=10),
        }
        league_avg = 4.5
        lam_h, lam_a = compute_poisson_lambdas("HOME_TEAM", "AWAY_TEAM", team_states, league_avg)

        off_h, def_h = compute_team_rates(team_states["HOME_TEAM"], league_avg)
        off_a, def_a = compute_team_rates(team_states["AWAY_TEAM"], league_avg)

        raw_h = (off_h * def_a) / league_avg
        raw_a = (off_a * def_h) / league_avg

        self.assertAlmostEqual(lam_h, raw_h * FROZEN_HOME_ADVANTAGE, places=6)
        self.assertAlmostEqual(lam_a, raw_a, places=6)

    def test_run_line_probabilities_half_run_lines_have_zero_push(self) -> None:
        # Home favorite -1.5
        probs_fav = compute_run_line_probabilities(4.8, 3.8, -1.5)
        self.assertEqual(probs_fav.p_push, 0.0)
        self.assertAlmostEqual(probs_fav.p_home_covers + probs_fav.p_away_covers, 1.0, places=5)
        self.assertEqual(probs_fav.predicted_side, "HOME" if probs_fav.p_home_covers >= 0.5 else "AWAY")

        # Home underdog +1.5
        probs_dog = compute_run_line_probabilities(3.8, 4.8, 1.5)
        self.assertEqual(probs_dog.p_push, 0.0)
        self.assertAlmostEqual(probs_dog.p_home_covers + probs_dog.p_away_covers, 1.0, places=5)
        self.assertEqual(probs_dog.predicted_side, "HOME" if probs_dog.p_home_covers >= 0.5 else "AWAY")

    def test_run_line_probabilities_integer_lines_support_push(self) -> None:
        # PK / 0.0 line
        probs_pk = compute_run_line_probabilities(4.5, 4.5, 0.0)
        self.assertGreater(probs_pk.p_push, 0.0)
        self.assertAlmostEqual(
            probs_pk.p_home_covers + probs_pk.p_away_covers + probs_pk.p_push,
            1.0,
            places=5,
        )

    def test_predicted_side_threshold_is_exactly_half(self) -> None:
        # Equal lambdas with spread -1.5 -> home covers < 0.5 -> AWAY
        p_away = compute_run_line_probabilities(4.0, 4.0, -1.5)
        self.assertLess(p_away.p_home_covers, 0.5)
        self.assertEqual(p_away.predicted_side, "AWAY")

        # Equal lambdas with spread +1.5 -> home covers > 0.5 -> HOME
        p_home = compute_run_line_probabilities(4.0, 4.0, 1.5)
        self.assertGreater(p_home.p_home_covers, 0.5)
        self.assertEqual(p_home.predicted_side, "HOME")


if __name__ == "__main__":
    unittest.main()
