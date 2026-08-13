"""Focused P40A paper BET/PASS, settlement, and authority tests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
from pathlib import Path
import unittest

from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40A_CHAMPION_ROLE,
    P40A_SHADOW_ROLE,
    build_p40a_decisions,
    load_p40a_authority,
    run_p40a_moneyline_paper_bet_pass,
    settle_p40a_decisions,
)
from match_analysis.baseball.domain.paper_moneyline_bet_pass import (
    DECISION_BET,
    DECISION_PASS,
    PaperMoneylineDecision,
    aggregate_paper_settlements,
    settle_paper_moneyline_decision,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPOSITORY_ROOT / "report/p40a_moneyline_paper_bet_pass"


def _decision(
    *,
    p_home: str,
    home_odds: str,
    away_odds: str,
    model_role: str = P40A_CHAMPION_ROLE,
    provider_game_id: str = "824829",
    game_pk: int = 824829,
) -> PaperMoneylineDecision:
    return PaperMoneylineDecision.create(
        model_role=model_role,
        model_id="model-p40a",
        model_fingerprint="a" * 64,
        p37_fold_id="wf_004",
        p37_window="window_001_holdout_wf_004",
        p37_prediction_row_id="b" * 64,
        provider_namespace="MLB_STATS_API",
        provider_game_id=provider_game_id,
        game_pk=game_pk,
        game_number=1,
        official_date="2026-06-08",
        scheduled_start_utc="2026-06-08T22:35:00Z",
        home_team="Baltimore Orioles",
        away_team="Seattle Mariners",
        home_team_code="BAL",
        away_team_code="SEA",
        market_snapshot_id="c" * 64,
        market_observed_at_utc="2026-06-08T21:00:00Z",
        local_fetched_at_utc="2026-06-08T21:00:00Z",
        source_match_id="3473130.1",
        market_source_sha256="d" * 64,
        p37_comparisons_sha256="e" * 64,
        model_probability_source="TEST_PREGAME_PROBABILITY",
        p_home=p_home,
        home_decimal_odds=home_odds,
        away_decimal_odds=away_odds,
    )


class P40APaperMoneylineDomainTests(unittest.TestCase):
    def test_home_bet_uses_offered_decimal_odds(self) -> None:
        decision = _decision(p_home="0.60", home_odds="2.00", away_odds="1.60")

        self.assertEqual(decision.decision, DECISION_BET)
        self.assertEqual(decision.candidate_side, "HOME")
        self.assertEqual(decision.ev_home, Decimal("0.20"))
        self.assertEqual(decision.ev_away, Decimal("-0.36"))
        self.assertEqual(decision.paper_stake_units, Decimal("1.0"))

    def test_away_bet_uses_complement_probability(self) -> None:
        decision = _decision(p_home="0.40", home_odds="1.50", away_odds="2.00")

        self.assertEqual(decision.decision, DECISION_BET)
        self.assertEqual(decision.candidate_side, "AWAY")
        self.assertEqual(decision.ev_home, Decimal("-0.40"))
        self.assertEqual(decision.ev_away, Decimal("0.20"))

    def test_pass_when_both_expected_values_are_non_positive(self) -> None:
        decision = _decision(p_home="0.50", home_odds="1.80", away_odds="1.80")

        self.assertEqual(decision.decision, DECISION_PASS)
        self.assertEqual(decision.paper_stake_units, Decimal("0"))
        self.assertEqual(decision.candidate_side, "NONE")

    def test_positive_ev_tie_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive EV tie"):
            _decision(p_home="0.50", home_odds="2.20", away_odds="2.20")

    def test_malformed_odds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _decision(p_home="0.60", home_odds="1.00", away_odds="1.60")
        with self.assertRaises(ValueError):
            _decision(p_home="0.60", home_odds="NaN", away_odds="1.60")

    def test_flat_unit_winning_and_losing_settlement(self) -> None:
        decision = _decision(p_home="0.60", home_odds="2.00", away_odds="1.60")
        won = settle_paper_moneyline_decision(
            decision,
            final_game_outcome="HOME",
            target_home_win=1,
            outcome_authority_row_id="f" * 64,
            outcome_authority="TEST_FINAL_RESULT",
        )
        lost = settle_paper_moneyline_decision(
            decision,
            final_game_outcome="AWAY",
            target_home_win=0,
            outcome_authority_row_id="f" * 64,
            outcome_authority="TEST_FINAL_RESULT",
        )

        self.assertEqual(won.gross_return_units, Decimal("2.00"))
        self.assertEqual(won.net_paper_units, Decimal("1.00"))
        self.assertEqual(lost.gross_return_units, Decimal("0"))
        self.assertEqual(lost.net_paper_units, Decimal("-1"))

    def test_aggregate_roi_and_drawdown_are_descriptive(self) -> None:
        first = _decision(p_home="0.60", home_odds="2.00", away_odds="1.60")
        second = _decision(
            p_home="0.60",
            home_odds="2.00",
            away_odds="1.60",
            provider_game_id="824830",
            game_pk=824830,
        )
        won = settle_paper_moneyline_decision(
            first,
            final_game_outcome="HOME",
            target_home_win=1,
            outcome_authority_row_id="1" * 64,
            outcome_authority="TEST_FINAL_RESULT",
        )
        lost = settle_paper_moneyline_decision(
            second,
            final_game_outcome="AWAY",
            target_home_win=0,
            outcome_authority_row_id="2" * 64,
            outcome_authority="TEST_FINAL_RESULT",
        )
        pass_decision = _decision(p_home="0.50", home_odds="1.80", away_odds="1.80")
        passed = settle_paper_moneyline_decision(
            pass_decision,
            final_game_outcome="HOME",
            target_home_win=1,
            outcome_authority_row_id="3" * 64,
            outcome_authority="TEST_FINAL_RESULT",
        )

        aggregate = aggregate_paper_settlements(
            (won, lost, passed),
            edge_ready_rows=3,
            model_role=P40A_CHAMPION_ROLE,
        )
        self.assertEqual(aggregate["bet_count"], 2)
        self.assertEqual(aggregate["pass_count"], 1)
        self.assertEqual(aggregate["win_count"], 1)
        self.assertEqual(aggregate["loss_count"], 1)
        self.assertEqual(aggregate["total_paper_units_risked"], "2.0")
        self.assertEqual(aggregate["net_paper_units"], "0.00")
        self.assertEqual(aggregate["descriptive_paper_roi"], "0.0")
        self.assertEqual(aggregate["maximum_paper_drawdown"], "1.00")


class P40AAuthorityIntegrationTests(unittest.TestCase):
    def test_authority_counts_aggregates_and_conclusions(self) -> None:
        result = run_p40a_moneyline_paper_bet_pass(REPOSITORY_ROOT)

        self.assertEqual(len(result.authority.market_rows), 62)
        self.assertEqual(len(result.authority.prediction_rows), 65)
        self.assertEqual(len(result.authority.outcome_rows), 65)
        self.assertEqual(len(result.decisions), 124)
        self.assertEqual(len(result.settlements), 124)
        champion = result.summary["models"]["champion_primary"]
        shadow = result.summary["models"]["raw_challenger_shadow"]
        self.assertEqual((champion["bet_count"], champion["pass_count"]), (22, 40))
        self.assertEqual((champion["win_count"], champion["loss_count"]), (14, 8))
        self.assertEqual(champion["net_paper_units"], "5.90")
        self.assertEqual(champion["descriptive_paper_roi"], "0.26818181818181818181818181818181818181818181818182")
        self.assertEqual((shadow["bet_count"], shadow["pass_count"]), (25, 37))
        self.assertEqual((shadow["win_count"], shadow["loss_count"]), (16, 9))
        self.assertEqual(shadow["net_paper_units"], "6.32")
        self.assertEqual(result.summary["primary_conclusion"], "PAPER_BASELINE_OBSERVED_POSITIVE")
        self.assertEqual(result.summary["shadow_comparison"], "SHADOW_CHALLENGER_HIGHER_NET_UNITS")
        self.assertTrue(result.summary["deterministic_rerun_verified"])
        self.assertEqual(
            result.summary["per_window_edge_ready_counts"],
            {
                "window_001_holdout_wf_004": 22,
                "window_002_holdout_wf_005": 15,
                "window_003_holdout_wf_006": 25,
            },
        )

    def test_decisions_are_outcome_free_and_outcome_mutation_cannot_change_them(self) -> None:
        authority = load_p40a_authority(REPOSITORY_ROOT)
        decisions = build_p40a_decisions(authority)
        mutated_outcomes = tuple(
            replace(
                outcome,
                actual_winner="AWAY" if outcome.actual_winner == "HOME" else "HOME",
                target_home_win=0 if outcome.target_home_win == 1 else 1,
            )
            for outcome in authority.outcome_rows
        )
        mutated_settlements = settle_p40a_decisions(
            authority,
            decisions,
            outcome_rows=mutated_outcomes,
        )

        self.assertTrue(all("final_game_outcome" not in row.to_projection() for row in decisions))
        self.assertEqual(
            tuple(row.to_projection() for row in decisions),
            tuple(row.to_projection() for row in build_p40a_decisions(authority)),
        )
        self.assertNotEqual(
            tuple(row.to_projection() for row in settle_p40a_decisions(authority, decisions)),
            tuple(row.to_projection() for row in mutated_settlements),
        )

    def test_champion_and_shadow_use_identical_rule_and_same_market_rows(self) -> None:
        result = run_p40a_moneyline_paper_bet_pass(REPOSITORY_ROOT)
        by_key: dict[tuple[str, str], object] = {
            (row.model_role, row.p37_prediction_row_id): row for row in result.decisions
        }
        self.assertEqual(len(by_key), 124)
        for prediction_id in {row.p37_prediction_row_id for row in result.decisions}:
            champion = by_key[(P40A_CHAMPION_ROLE, prediction_id)]
            shadow = by_key[(P40A_SHADOW_ROLE, prediction_id)]
            self.assertEqual(champion.market_snapshot_id, shadow.market_snapshot_id)
            self.assertEqual(champion.home_decimal_odds, shadow.home_decimal_odds)
            self.assertEqual(champion.away_decimal_odds, shadow.away_decimal_odds)
            self.assertEqual(champion.paper_stake_convention, shadow.paper_stake_convention)
            self.assertEqual(champion.decision == DECISION_BET, champion.candidate_ev > 0)
            self.assertEqual(shadow.decision == DECISION_BET, shadow.candidate_ev > 0)

    def test_shuffled_authority_inputs_render_identically(self) -> None:
        authority = load_p40a_authority(REPOSITORY_ROOT)
        decisions = build_p40a_decisions(authority)
        shuffled_authority = replace(
            authority,
            market_rows=tuple(reversed(authority.market_rows)),
            prediction_rows=tuple(reversed(authority.prediction_rows)),
            outcome_rows=tuple(reversed(authority.outcome_rows)),
        )
        shuffled_decisions = build_p40a_decisions(shuffled_authority)
        self.assertEqual(
            tuple(row.to_projection() for row in decisions),
            tuple(row.to_projection() for row in shuffled_decisions),
        )
        self.assertEqual(
            tuple(row.to_projection() for row in settle_p40a_decisions(authority, decisions)),
            tuple(row.to_projection() for row in settle_p40a_decisions(shuffled_authority, shuffled_decisions)),
        )

    def test_p37_p38_p39_authority_files_remain_invariant(self) -> None:
        paths = (
            REPOSITORY_ROOT / "report/p37a_rolling_walk_forward_oos/comparisons.jsonl",
            REPOSITORY_ROOT / "report/p37a_rolling_walk_forward_oos/summary.json",
            REPOSITORY_ROOT / "report/p38a_rolling_probability_calibration/comparisons.jsonl",
            REPOSITORY_ROOT / "report/p38a_rolling_probability_calibration/summary.json",
            REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/market_join.jsonl",
            REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/market_snapshots.jsonl",
            REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/summary.json",
            REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/source_manifest.json",
        )
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        run_p40a_moneyline_paper_bet_pass(REPOSITORY_ROOT)
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
