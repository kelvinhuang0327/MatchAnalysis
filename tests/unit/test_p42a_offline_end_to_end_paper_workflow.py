"""Focused tests for the shipped P42A offline paper-workflow orchestrator."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import ast
import hashlib
from pathlib import Path
import unittest

from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40A_CHAMPION_ROLE,
    P40AOutcomeRow,
    load_p40a_authority,
)
from match_analysis.application.use_cases.p42a_offline_end_to_end_paper_workflow import (
    P42A_EXPECTED_CHAMPION,
    P42A_REPORT_RELATIVE_PATH,
    P42A_WORKFLOW_LABEL,
    attach_and_settle_frozen_decisions,
    freeze_champion_decisions,
    load_p39_no_market_exclusions,
    reconcile_with_p40_champion,
    run_p42a_offline_end_to_end_paper_workflow,
)
from match_analysis.baseball.domain.paper_moneyline_bet_pass import (
    DECISION_BET,
    DECISION_PASS,
    SETTLEMENT_PASS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
USE_CASE_PATH = (
    REPOSITORY_ROOT
    / "src/match_analysis/application/use_cases/p42a_offline_end_to_end_paper_workflow.py"
)
PROTECTED_PATHS = (
    REPOSITORY_ROOT / "report/p37a_rolling_walk_forward_oos/comparisons.jsonl",
    REPOSITORY_ROOT / "report/p37a_rolling_walk_forward_oos/summary.json",
    REPOSITORY_ROOT / "report/p38a_rolling_probability_calibration/comparisons.jsonl",
    REPOSITORY_ROOT / "report/p38a_rolling_probability_calibration/summary.json",
    REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/market_join.jsonl",
    REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/market_snapshots.jsonl",
    REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/summary.json",
    REPOSITORY_ROOT / "report/p40a_moneyline_paper_bet_pass/decisions.jsonl",
    REPOSITORY_ROOT / "report/p40a_moneyline_paper_bet_pass/settlements.jsonl",
    REPOSITORY_ROOT / "report/p40a_moneyline_paper_bet_pass/summary.json",
    REPOSITORY_ROOT / "report/p41a_walk_forward_ev_margin_policy/summary.json",
    REPOSITORY_ROOT / "report/p41a_walk_forward_ev_margin_policy/policy_evaluations.jsonl",
)


class P42AOfflineWorkflowTests(unittest.TestCase):
    def test_full_orchestrator_reconciles_p40_and_labels_rehearsal(self) -> None:
        result = run_p42a_offline_end_to_end_paper_workflow(REPOSITORY_ROOT)
        summary = result.summary

        self.assertEqual(summary["p37_target_count"], 65)
        self.assertEqual(summary["p39_edge_ready_count"], 62)
        self.assertEqual(summary["workflow_decision_count"], 62)
        self.assertEqual(summary["bet_count"], 22)
        self.assertEqual(summary["pass_count"], 40)
        self.assertEqual(summary["settled_bet_count"], 22)
        self.assertEqual(summary["unresolved_result_count"], 0)
        self.assertEqual(summary["feedback_row_count"], 62)
        self.assertEqual(summary["win_count"], 14)
        self.assertEqual(summary["loss_count"], 8)
        self.assertEqual(summary["push_count"], 0)
        self.assertEqual(summary["units_risked"], "22.0")
        self.assertEqual(summary["net_paper_units"], "5.90")
        self.assertEqual(
            summary["descriptive_historical_paper_roi"],
            P42A_EXPECTED_CHAMPION["descriptive_paper_roi"],
        )
        self.assertEqual(summary["maximum_drawdown"], "2")
        self.assertEqual(summary["p39_no_market_count"], 3)
        self.assertEqual(summary["exclusion_reasons"], {"NO_MARKET": 3})
        self.assertEqual(summary["p40_reconciliation"]["status"], "RECONCILED")
        self.assertEqual(summary["workflow_label"], P42A_WORKFLOW_LABEL)
        self.assertIn("OFFLINE", summary["labels"])
        self.assertIn("HISTORICAL", summary["labels"])
        self.assertFalse(summary["prospective"])
        self.assertFalse(summary["live"])
        self.assertFalse(summary["production"])
        self.assertFalse(summary["forward_real"])
        self.assertFalse(summary["claims"]["real_betting"])
        self.assertFalse(summary["claims"]["threshold_optimization"])
        self.assertFalse(summary["network_required"])
        self.assertTrue(summary["deterministic_rerun_verified"])
        self.assertTrue(summary["workflow_completeness"]["one_to_one_lineage"])
        self.assertEqual(len(result.ledger_rows), 62)
        self.assertEqual(len(result.exclusion_rows), 3)

    def test_decision_fingerprint_exists_before_result_fields(self) -> None:
        authority = load_p40a_authority(REPOSITORY_ROOT)
        decisions = freeze_champion_decisions(authority)
        self.assertEqual(len(decisions), 62)
        for decision in decisions:
            projection = decision.to_projection()
            self.assertEqual(len(decision.decision_id), 64)
            self.assertNotIn("final_game_outcome", projection)
            self.assertNotIn("settlement_status", projection)
            self.assertNotIn("net_paper_units", projection)
        settlements = attach_and_settle_frozen_decisions(
            decisions,
            authority.outcome_rows,
        )
        self.assertEqual(
            [row.decision.decision_id for row in settlements],
            [row.decision_id for row in decisions],
        )
        self.assertTrue(all(row.final_game_outcome in {"HOME", "AWAY"} for row in settlements))

    def test_outcome_mutation_cannot_change_frozen_decision_ids(self) -> None:
        authority = load_p40a_authority(REPOSITORY_ROOT)
        decisions = freeze_champion_decisions(authority)
        mutated = tuple(
            replace(
                outcome,
                actual_winner="AWAY" if outcome.actual_winner == "HOME" else "HOME",
                target_home_win=0 if outcome.target_home_win == 1 else 1,
            )
            for outcome in authority.outcome_rows
        )
        original = attach_and_settle_frozen_decisions(decisions, authority.outcome_rows)
        mutated_settlements = attach_and_settle_frozen_decisions(decisions, mutated)
        self.assertEqual(
            [row.decision.decision_id for row in original],
            [row.decision.decision_id for row in mutated_settlements],
        )
        self.assertNotEqual(
            [row.to_projection() for row in original],
            [row.to_projection() for row in mutated_settlements],
        )

    def test_pass_rows_are_zero_stake_and_not_fake_win_loss(self) -> None:
        result = run_p42a_offline_end_to_end_paper_workflow(REPOSITORY_ROOT)
        pass_rows = [row for row in result.ledger_rows if row["bet_or_pass"] == DECISION_PASS]
        self.assertEqual(len(pass_rows), 40)
        for row in pass_rows:
            self.assertEqual(row["paper_stake_units"], "0")
            self.assertEqual(row["net_paper_units"], "0")
            self.assertEqual(row["settlement_status"], SETTLEMENT_PASS)
            self.assertEqual(row["evaluation_status"], "PASS_NO_WAGER")
            self.assertIsNone(row["evaluation_is_correct"])
            self.assertEqual(row["evaluation_correctness_status"], "NO_WAGER")
            self.assertEqual(row["selected_side"], "NONE")

    def test_bet_rows_have_one_result_settlement_evaluation_feedback(self) -> None:
        result = run_p42a_offline_end_to_end_paper_workflow(REPOSITORY_ROOT)
        bet_rows = [row for row in result.ledger_rows if row["bet_or_pass"] == DECISION_BET]
        self.assertEqual(len(bet_rows), 22)
        fingerprints = [row["decision_fingerprint"] for row in result.ledger_rows]
        feedback_ids = [row["feedback_identity"] for row in result.ledger_rows]
        self.assertEqual(len(set(fingerprints)), 62)
        self.assertEqual(len(set(feedback_ids)), 62)
        for row in bet_rows:
            self.assertEqual(row["paper_stake_units"], "1.0")
            self.assertIn(row["actual_winner"], {"HOME", "AWAY"})
            self.assertIn(row["settlement_status"], {"BET_WON", "BET_LOST"})
            self.assertIn(row["evaluation_status"], {"BET_WON", "BET_LOST"})
            self.assertIsInstance(row["evaluation_is_correct"], bool)
            self.assertEqual(len(row["feedback_identity"]), 64)
            self.assertEqual(
                row["upstream_authority_identifiers"]["p40_decision_id"],
                row["decision_fingerprint"],
            )
            self.assertEqual(len(row["upstream_authority_identifiers"]["p40_settlement_row_fingerprint"]), 64)

    def test_no_market_rows_never_become_decisions(self) -> None:
        result = run_p42a_offline_end_to_end_paper_workflow(REPOSITORY_ROOT)
        exclusions = load_p39_no_market_exclusions(REPOSITORY_ROOT)
        self.assertEqual(len(exclusions), 3)
        excluded_ids = {row["p37_prediction_row_id"] for row in exclusions}
        decision_ids = {row.p37_prediction_row_id for row in result.decisions}
        self.assertTrue(excluded_ids.isdisjoint(decision_ids))
        self.assertTrue(all(row["exclusion_reason"] == "NO_MARKET" for row in result.exclusion_rows))
        self.assertTrue(all(row["became_bet"] is False for row in result.exclusion_rows))

    def test_p40_aggregate_reconciliation_matches_committed_authority(self) -> None:
        result = run_p42a_offline_end_to_end_paper_workflow(REPOSITORY_ROOT)
        computed = result.summary["p40_reconciliation"]["computed"]
        self.assertEqual(computed, P42A_EXPECTED_CHAMPION)
        self.assertEqual(
            computed,
            result.summary["p40_reconciliation"]["committed_p40_champion"],
        )
        again = reconcile_with_p40_champion(
            result.settlements,
            {
                "models": {
                    "champion_primary": P42A_EXPECTED_CHAMPION,
                }
            },
        )
        self.assertEqual(again["status"], "RECONCILED")

    def test_duplicate_result_is_rejected(self) -> None:
        authority = load_p40a_authority(REPOSITORY_ROOT)
        decisions = freeze_champion_decisions(authority)
        duplicated = authority.outcome_rows + (authority.outcome_rows[0],)
        with self.assertRaisesRegex(RuntimeError, "P42A_DUPLICATE_RESULT_REJECTED_STOP"):
            attach_and_settle_frozen_decisions(decisions, duplicated)

    def test_missing_result_fails_closed(self) -> None:
        authority = load_p40a_authority(REPOSITORY_ROOT)
        decisions = freeze_champion_decisions(authority)
        with self.assertRaisesRegex(RuntimeError, "P42A_MISSING_RESULT_FAIL_CLOSED_STOP"):
            attach_and_settle_frozen_decisions(decisions, authority.outcome_rows[1:])

    def test_non_final_result_fails_closed(self) -> None:
        authority = load_p40a_authority(REPOSITORY_ROOT)
        decisions = freeze_champion_decisions(authority)
        first = authority.outcome_rows[0]
        broken = replace(first, actual_winner="CANCELLED")
        with self.assertRaisesRegex(RuntimeError, "P42A_NON_FINAL_RESULT_FAIL_CLOSED_STOP"):
            attach_and_settle_frozen_decisions(
                decisions,
                (broken, *authority.outcome_rows[1:]),
            )

    def test_two_frozen_runs_are_byte_identical(self) -> None:
        first = run_p42a_offline_end_to_end_paper_workflow(REPOSITORY_ROOT)
        second = run_p42a_offline_end_to_end_paper_workflow(REPOSITORY_ROOT)
        self.assertEqual(first.ledger_rows, second.ledger_rows)
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(
            [row.to_projection() for row in first.decisions],
            [row.to_projection() for row in second.decisions],
        )

    def test_shuffled_inputs_remain_deterministic(self) -> None:
        authority = load_p40a_authority(REPOSITORY_ROOT)
        decisions = freeze_champion_decisions(authority)
        shuffled = replace(
            authority,
            market_rows=tuple(reversed(authority.market_rows)),
            prediction_rows=tuple(reversed(authority.prediction_rows)),
            outcome_rows=tuple(reversed(authority.outcome_rows)),
        )
        shuffled_decisions = freeze_champion_decisions(shuffled)
        self.assertEqual(
            [row.to_projection() for row in decisions],
            [row.to_projection() for row in shuffled_decisions],
        )
        self.assertEqual(
            [row.to_projection() for row in attach_and_settle_frozen_decisions(decisions, authority.outcome_rows)],
            [
                row.to_projection()
                for row in attach_and_settle_frozen_decisions(
                    shuffled_decisions,
                    shuffled.outcome_rows,
                )
            ],
        )

    def test_source_has_no_network_imports(self) -> None:
        tree = ast.parse(USE_CASE_PATH.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("urllib", imported)
        self.assertNotIn("requests", imported)
        self.assertNotIn("http", imported)
        self.assertNotIn("socket", imported)
        self.assertNotIn("ssl", imported)
        source = USE_CASE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("urlopen", source)
        self.assertNotIn("verify=False", source)

    def test_protected_authorities_remain_invariant(self) -> None:
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in PROTECTED_PATHS
        }
        run_p42a_offline_end_to_end_paper_workflow(REPOSITORY_ROOT)
        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in PROTECTED_PATHS
        }
        self.assertEqual(before, after)

    def test_champion_only_and_report_path_convention(self) -> None:
        result = run_p42a_offline_end_to_end_paper_workflow(REPOSITORY_ROOT)
        self.assertTrue(all(row.model_role == P40A_CHAMPION_ROLE for row in result.decisions))
        self.assertEqual(P42A_REPORT_RELATIVE_PATH.as_posix(), "report/p42a_offline_end_to_end_paper_workflow")
        pass_settlements = [row for row in result.settlements if row.decision.decision == DECISION_PASS]
        self.assertTrue(all(row.decision.paper_stake_units == Decimal("0") for row in pass_settlements))


if __name__ == "__main__":
    unittest.main()
