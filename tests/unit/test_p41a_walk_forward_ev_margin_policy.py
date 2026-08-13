"""Focused P41A leakage-safe walk-forward policy tests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import unittest

from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    load_p40a_authority,
)
from match_analysis.application.use_cases.p41a_walk_forward_ev_margin_policy import (
    P41A_CANDIDATE_THRESHOLDS,
    evaluate_p41a_authority,
    load_p41a_authority,
    run_p41a_walk_forward_ev_margin_policy,
    select_threshold_from_prior_rows,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class P41AWalkForwardPolicyTests(unittest.TestCase):
    def test_candidate_threshold_set_is_frozen(self) -> None:
        self.assertEqual(
            P41A_CANDIDATE_THRESHOLDS,
            (
                Decimal("0.00"),
                Decimal("0.01"),
                Decimal("0.02"),
                Decimal("0.03"),
                Decimal("0.05"),
            ),
        )

    def test_walk_forward_selection_and_aggregate_metrics(self) -> None:
        result = run_p41a_walk_forward_ev_margin_policy(REPOSITORY_ROOT)

        self.assertTrue(result.summary["deterministic_rerun_verified"])
        self.assertTrue(result.summary["true_oos_verified"])
        self.assertEqual(result.summary["total_target_windows"], 2)
        self.assertEqual(result.summary["policy_oos_target_rows"], 40)
        self.assertEqual(
            result.summary["selected_threshold_per_target_window"],
            {
                "window_002_holdout_wf_005": "0.05",
                "window_003_holdout_wf_006": "0.03",
            },
        )
        self.assertEqual(
            result.summary["threshold_selection_counts"],
            {"0.00": 0, "0.01": 0, "0.02": 0, "0.03": 1, "0.05": 1},
        )

        first, second = result.window_evaluations
        self.assertEqual(first.prior_eligible_row_count, 22)
        self.assertEqual(
            first.prior_policy_training_windows,
            ("window_001_holdout_wf_004",),
        )
        self.assertEqual(first.selected_threshold, Decimal("0.05"))
        self.assertEqual(
            first.tie_break_reason,
            "LARGER_EV_THRESHOLD_ON_EQUAL_PRIOR_NET_UNITS",
        )
        self.assertEqual(first.selected_policy_target_metrics["bet_count"], 3)
        self.assertEqual(first.selected_policy_target_metrics["net_paper_units"], "1.82")
        self.assertEqual(first.zero_ev_baseline_target_metrics["bet_count"], 5)
        self.assertEqual(first.zero_ev_baseline_target_metrics["net_paper_units"], "1.47")

        self.assertEqual(second.prior_eligible_row_count, 37)
        self.assertEqual(second.selected_threshold, Decimal("0.03"))
        self.assertEqual(
            second.prior_threshold_metrics[3]["net_paper_units"],
            "6.75",
        )
        self.assertEqual(second.selected_policy_target_metrics["bet_count"], 5)
        self.assertEqual(second.selected_policy_target_metrics["net_paper_units"], "1.33")
        self.assertEqual(second.zero_ev_baseline_target_metrics["net_paper_units"], "1.15")

        selected = result.summary["selected_policy"]
        baseline = result.summary["zero_ev_baseline"]
        self.assertEqual(
            (selected["bet_count"], selected["pass_count"], selected["win_count"], selected["loss_count"]),
            (8, 32, 5, 3),
        )
        self.assertEqual(selected["total_paper_units_risked"], "8.0")
        self.assertEqual(selected["net_paper_units"], "3.15")
        self.assertEqual(selected["descriptive_paper_roi"], "0.39375")
        self.assertEqual(selected["maximum_paper_drawdown"], "1.00")
        self.assertEqual(
            (baseline["bet_count"], baseline["pass_count"], baseline["win_count"], baseline["loss_count"]),
            (12, 28, 7, 5),
        )
        self.assertEqual(baseline["total_paper_units_risked"], "12.0")
        self.assertEqual(baseline["net_paper_units"], "2.62")
        self.assertEqual(baseline["maximum_paper_drawdown"], "2.00")
        self.assertEqual(result.summary["conclusion"], "EV_MARGIN_POLICY_IMPROVED")

    def test_target_outcomes_cannot_change_prior_threshold_selection(self) -> None:
        authority = load_p41a_authority(REPOSITORY_ROOT)
        prior_rows = tuple(
            row
            for row in authority.champion_decisions
            if row.p37_window == "window_001_holdout_wf_004"
        )
        outcome_by_id = {row.p37_prediction_row_id: row for row in authority.outcome_rows}
        original = select_threshold_from_prior_rows(prior_rows, outcome_by_id)

        mutated_outcomes = dict(outcome_by_id)
        for decision in authority.champion_decisions:
            if decision.p37_window != "window_002_holdout_wf_005":
                continue
            outcome = mutated_outcomes[decision.p37_prediction_row_id]
            mutated_outcomes[decision.p37_prediction_row_id] = replace(
                outcome,
                actual_winner="AWAY" if outcome.actual_winner == "HOME" else "HOME",
                target_home_win=0 if outcome.target_home_win == 1 else 1,
            )

        mutated = select_threshold_from_prior_rows(prior_rows, mutated_outcomes)
        self.assertEqual(original.selected_threshold, mutated.selected_threshold)
        self.assertEqual(original.tie_break_reason, mutated.tie_break_reason)
        self.assertEqual(original.prior_threshold_metrics, mutated.prior_threshold_metrics)

    def test_shuffled_input_is_deterministic(self) -> None:
        authority = load_p41a_authority(REPOSITORY_ROOT)
        shuffled = replace(
            authority,
            champion_decisions=tuple(reversed(authority.champion_decisions)),
            outcome_rows=tuple(reversed(authority.outcome_rows)),
        )

        first = evaluate_p41a_authority(authority)
        second = evaluate_p41a_authority(shuffled)
        self.assertEqual(
            tuple(row.to_projection() for row in first.window_evaluations),
            tuple(row.to_projection() for row in second.window_evaluations),
        )
        self.assertEqual(first.summary, second.summary)

    def test_insufficient_policy_windows_stop(self) -> None:
        authority = load_p41a_authority(REPOSITORY_ROOT)
        only_first_window = replace(
            authority,
            champion_decisions=tuple(
                row
                for row in authority.champion_decisions
                if row.p37_window == "window_001_holdout_wf_004"
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "P41A_INSUFFICIENT_POLICY_OOS_WINDOWS_STOP",
        ):
            evaluate_p41a_authority(only_first_window)


class P41AAuthorityInvariantTests(unittest.TestCase):
    def test_p40a_authority_loader_remains_available(self) -> None:
        authority = load_p40a_authority(REPOSITORY_ROOT)
        self.assertEqual(len(authority.market_rows), 62)
        self.assertEqual(len(authority.outcome_rows), 65)


if __name__ == "__main__":
    unittest.main()
