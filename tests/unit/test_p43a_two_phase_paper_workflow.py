"""Focused tests for the shipped P43A two-phase paper workflow."""

from __future__ import annotations

import ast
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40A_CHAMPION_ROLE,
)
from match_analysis.application.use_cases.p42a_offline_end_to_end_paper_workflow import (
    P42A_EXPECTED_CHAMPION,
)
from match_analysis.application.use_cases.p43a_postgame_settle import (
    load_p43a_final_result_authority,
    load_p43a_frozen_decision_bundle,
    run_p43a_postgame_settle,
    settle_p43a_frozen_decisions,
    verify_p43a_pregame_record,
)
from match_analysis.application.use_cases.p43a_pregame_freeze import (
    P43A_HUMAN_LABEL,
    P43A_PREDICTION_KEYS,
    P43A_REPORT_RELATIVE_PATH,
    freeze_p43a_pregame_decisions,
    load_p43a_pregame_authority,
    run_p43a_pregame_freeze,
)
from match_analysis.application.use_cases.p44a_historical_source_adapter import (
    adapt_historical_pregame,
    adapt_historical_results,
    protected_authority_hashes,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    project_normalized_results,
    write_normalized_result_input,
)
from match_analysis.baseball.domain.paper_moneyline_bet_pass import (
    DECISION_BET,
    DECISION_PASS,
    SETTLEMENT_PASS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREGAME_PATH = (
    REPOSITORY_ROOT
    / "src/match_analysis/application/use_cases/p43a_pregame_freeze.py"
)
POSTGAME_PATH = (
    REPOSITORY_ROOT
    / "src/match_analysis/application/use_cases/p43a_postgame_settle.py"
)
CLI_PATH = (
    REPOSITORY_ROOT
    / "src/match_analysis/interfaces/cli/run_p43a_two_phase_paper_workflow.py"
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
    REPOSITORY_ROOT / "report/p42a_offline_end_to_end_paper_workflow/summary.json",
    REPOSITORY_ROOT / "report/p42a_offline_end_to_end_paper_workflow/workflow_ledger.jsonl",
)
OUTCOME_FIELD_NAMES = (
    "actual_winner",
    "target_home_win",
    "final_game_outcome",
    "final_score",
    "boxscore",
)


def _strip_outcome_fields(source: Path, destination: Path) -> None:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.append({key: row[key] for key in P43A_PREDICTION_KEYS})
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _historical_pregame(comparisons_path: Path | None = None):
    return adapt_historical_pregame(
        REPOSITORY_ROOT, comparisons_path=comparisons_path
    )


def _historical_outcomes():
    return project_normalized_results(adapt_historical_results(REPOSITORY_ROOT))


def _freeze(*, persist: bool = False, output_dir: Path | None = None, pregame_input=None):
    return run_p43a_pregame_freeze(
        REPOSITORY_ROOT,
        pregame_input=pregame_input if pregame_input is not None else _historical_pregame(),
        output_dir=output_dir,
        persist=persist,
    )


def _settle(output_dir: Path, *, persist: bool = True, result_input=None, outcome_rows=None):
    kwargs: dict = {"output_dir": output_dir, "persist": persist}
    if outcome_rows is not None:
        kwargs["outcome_rows"] = outcome_rows
    elif result_input is not None:
        kwargs["result_input"] = result_input
    else:
        kwargs["outcome_rows"] = _historical_outcomes()
    return run_p43a_postgame_settle(REPOSITORY_ROOT, **kwargs)


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


class P43ATwoPhaseWorkflowTests(unittest.TestCase):
    def test_pregame_freeze_does_not_read_postgame_fields(self) -> None:
        source = PREGAME_PATH.read_text(encoding="utf-8")
        for name in OUTCOME_FIELD_NAMES:
            self.assertNotIn(name, source)
        self.assertNotIn("freeze_champion_decisions", POSTGAME_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("build_p40a_decisions", POSTGAME_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("load_p43a_pregame_authority", POSTGAME_PATH.read_text(encoding="utf-8"))

    def test_pregame_succeeds_when_result_fields_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            stripped = directory / "comparisons.jsonl"
            _strip_outcome_fields(
                REPOSITORY_ROOT / "report/p37a_rolling_walk_forward_oos/comparisons.jsonl",
                stripped,
            )
            for line in stripped.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertNotIn("actual_winner", row)
                self.assertNotIn("target_home_win", row)
            result = _freeze(
                output_dir=directory / "out",
                persist=True,
                pregame_input=_historical_pregame(stripped),
            )
            self.assertEqual(result.summary["workflow_decision_count"], 62)
            self.assertEqual(result.summary["bet_count"], 22)
            self.assertEqual(result.summary["pass_count"], 40)
            self.assertEqual(result.summary["settled_bet_count"], 0)
            self.assertEqual(result.summary["unresolved_result_count"], 62)
            self.assertEqual(result.summary["human_label"], P43A_HUMAN_LABEL)
            self.assertFalse(result.summary["live"])
            self.assertFalse(result.summary["prospective"])
            self.assertFalse(result.summary["claims"]["real_betting"])
            self.assertFalse(result.summary["network_required"])
            self.assertEqual(len(result.authority.outcome_rows), 0)
            self.assertTrue(all(row.model_role == P40A_CHAMPION_ROLE for row in result.decisions))

    def test_pregame_fingerprint_is_deterministic_and_preserves_bet_pass(self) -> None:
        first = _freeze(persist=False)
        second = _freeze(persist=False)
        self.assertEqual(first.records, second.records)
        self.assertEqual(
            [row.to_projection() for row in first.decisions],
            [row.to_projection() for row in second.decisions],
        )
        self.assertEqual(sum(row.decision == DECISION_BET for row in first.decisions), 22)
        self.assertEqual(sum(row.decision == DECISION_PASS for row in first.decisions), 40)
        for decision in first.decisions:
            projection = decision.to_projection()
            self.assertEqual(len(decision.decision_id), 64)
            self.assertNotIn("final_game_outcome", projection)
            self.assertNotIn("settlement_status", projection)
            if decision.decision == DECISION_PASS:
                self.assertEqual(decision.paper_stake_units, Decimal("0"))
            else:
                self.assertEqual(decision.paper_stake_units, Decimal("1.0"))

    def test_shuffled_pregame_inputs_remain_deterministic(self) -> None:
        authority = load_p43a_pregame_authority(
            REPOSITORY_ROOT, pregame_input=_historical_pregame()
        )
        decisions = freeze_p43a_pregame_decisions(authority)
        shuffled = replace(
            authority,
            market_rows=tuple(reversed(authority.market_rows)),
            prediction_rows=tuple(reversed(authority.prediction_rows)),
        )
        shuffled_decisions = freeze_p43a_pregame_decisions(
            shuffled,
            market_rows=shuffled.market_rows,
            prediction_rows=shuffled.prediction_rows,
        )
        self.assertEqual(
            [row.to_projection() for row in decisions],
            [row.to_projection() for row in shuffled_decisions],
        )

    def test_phase2_consumes_frozen_bundle_and_does_not_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            pregame = _freeze(output_dir=directory, persist=True)
            frozen_bytes = (directory / "pregame_decisions.jsonl").read_bytes()
            postgame = _settle(directory, persist=True)
            self.assertEqual((directory / "pregame_decisions.jsonl").read_bytes(), frozen_bytes)
            self.assertEqual(
                [row.decision_id for row in postgame.decisions],
                [row.decision_id for row in pregame.decisions],
            )
            self.assertEqual(
                [row.decision.decision_id for row in postgame.settlements],
                [row.decision_id for row in pregame.decisions],
            )
            self.assertEqual(postgame.summary["feedback_row_count"], 62)
            self.assertTrue(postgame.summary["workflow_completeness"]["one_to_one_lineage"])
            for ledger_row, decision in zip(postgame.ledger_rows, pregame.decisions, strict=True):
                self.assertEqual(ledger_row["decision_fingerprint"], decision.decision_id)
                self.assertEqual(ledger_row["pregame_decision_fingerprint"], decision.decision_id)
                self.assertEqual(len(ledger_row["feedback_identity"]), 64)
                self.assertTrue(ledger_row["upstream_decision_id_unchanged"])

    def test_tampered_decision_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _freeze(output_dir=directory, persist=True)
            path = directory / "pregame_decisions.jsonl"
            rows = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["bet_or_pass"] = (
                DECISION_PASS if rows[0]["bet_or_pass"] == DECISION_BET else DECISION_BET
            )
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "P43A_DECISION_BUNDLE_TAMPERED"):
                _settle(directory, persist=False)
            with self.assertRaisesRegex(RuntimeError, "P43A_DECISION_BUNDLE_TAMPERED"):
                verify_p43a_pregame_record(rows[0])

    def test_missing_non_final_and_conflicting_results_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            pregame = _freeze(output_dir=directory, persist=True)
            missing = directory / "missing_results.jsonl"
            with self.assertRaisesRegex(RuntimeError, "P43A_MISSING_RESULT_FAIL_CLOSED"):
                _settle(directory, result_input=missing, persist=False)
            result_path = directory / "normalized_results.jsonl"
            write_normalized_result_input(result_path, adapt_historical_results(REPOSITORY_ROOT))
            outcomes = load_p43a_final_result_authority(result_path)
            with self.assertRaisesRegex(RuntimeError, "P43A_MISSING_RESULT_FAIL_CLOSED"):
                settle_p43a_frozen_decisions(pregame.decisions, outcomes[1:])
            broken = replace(outcomes[0], actual_winner="CANCELLED")
            with self.assertRaisesRegex(RuntimeError, "P43A_NON_FINAL_RESULT_FAIL_CLOSED"):
                settle_p43a_frozen_decisions(pregame.decisions, (broken, *outcomes[1:]))
            conflicting = outcomes + (
                replace(
                    outcomes[0],
                    actual_winner="AWAY" if outcomes[0].actual_winner == "HOME" else "HOME",
                    target_home_win=0 if outcomes[0].target_home_win == 1 else 1,
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "P43A_CONFLICTING_RESULT_REJECTED"):
                settle_p43a_frozen_decisions(pregame.decisions, conflicting)

    def test_changing_final_result_does_not_change_phase1_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            pregame = _freeze(output_dir=directory, persist=True)
            outcomes = _historical_outcomes()
            mutated = tuple(
                replace(
                    outcome,
                    actual_winner="AWAY" if outcome.actual_winner == "HOME" else "HOME",
                    target_home_win=0 if outcome.target_home_win == 1 else 1,
                )
                for outcome in outcomes
            )
            original = settle_p43a_frozen_decisions(pregame.decisions, outcomes)
            mutated_settlements = settle_p43a_frozen_decisions(pregame.decisions, mutated)
            self.assertEqual(
                [row.decision.decision_id for row in original],
                [row.decision.decision_id for row in mutated_settlements],
            )
            self.assertEqual(
                [row.decision_id for row in pregame.decisions],
                [row.decision.decision_id for row in mutated_settlements],
            )
            self.assertNotEqual(
                [row.to_projection() for row in original],
                [row.to_projection() for row in mutated_settlements],
            )
            loaded, _ = load_p43a_frozen_decision_bundle(directory)
            self.assertEqual(
                [row.decision_id for row in loaded],
                [row.decision_id for row in pregame.decisions],
            )

    def test_repeated_phase2_is_idempotent_and_lineage_is_one_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _freeze(output_dir=directory, persist=True)
            first = _settle(directory, persist=True)
            first_ledger = (directory / "workflow_ledger.jsonl").read_bytes()
            second = _settle(directory, persist=True)
            self.assertEqual((directory / "workflow_ledger.jsonl").read_bytes(), first_ledger)
            self.assertEqual(first.ledger_rows, second.ledger_rows)
            self.assertEqual(second.write_status["workflow_ledger.jsonl"], "RECOGNIZED_IDENTICAL")
            self.assertEqual(len({row["feedback_identity"] for row in first.ledger_rows}), 62)
            self.assertEqual(len({row["decision_fingerprint"] for row in first.ledger_rows}), 62)
            pass_rows = [row for row in first.ledger_rows if row["bet_or_pass"] == DECISION_PASS]
            self.assertEqual(len(pass_rows), 40)
            for row in pass_rows:
                self.assertEqual(row["paper_stake_units"], "0")
                self.assertEqual(row["net_paper_units"], "0")
                self.assertEqual(row["settlement_status"], SETTLEMENT_PASS)

    def test_p42_reconciliation_and_historical_label(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _freeze(output_dir=directory, persist=True)
            postgame = _settle(directory, persist=True)
            computed = postgame.summary["p42_reconciliation"]["computed"]
            self.assertEqual(computed["eligible_universe"], 62)
            self.assertEqual(computed["bet_count"], 22)
            self.assertEqual(computed["pass_count"], 40)
            self.assertEqual(computed["settled_bet_count"], 22)
            self.assertEqual(computed["wins"], 14)
            self.assertEqual(computed["losses"], 8)
            self.assertEqual(computed["pushes"], 0)
            self.assertEqual(computed["units_risked"], "22.0")
            self.assertEqual(computed["net_paper_units"], "5.90")
            self.assertEqual(
                computed["descriptive_roi"],
                P42A_EXPECTED_CHAMPION["descriptive_paper_roi"],
            )
            self.assertEqual(computed["feedback_count"], 62)
            self.assertEqual(postgame.summary["p42_reconciliation"]["status"], "RECONCILED")
            self.assertEqual(postgame.summary["human_label"], P43A_HUMAN_LABEL)
            report = (directory / "report.md").read_text(encoding="utf-8")
            self.assertIn(P43A_HUMAN_LABEL, report)
            self.assertIn("not prospective, live, production", report.lower())
            self.assertFalse(postgame.summary["claims"]["real_betting"])
            self.assertFalse(postgame.summary["network_required"])

    def test_source_has_no_network_imports(self) -> None:
        for path in (PREGAME_PATH, POSTGAME_PATH, CLI_PATH):
            imported = _imported_roots(path)
            for forbidden in ("urllib", "requests", "http", "socket", "ssl"):
                self.assertNotIn(forbidden, imported)
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("urlopen", source)
            self.assertNotIn("verify=False", source)

    def test_protected_authorities_remain_invariant(self) -> None:
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in PROTECTED_PATHS
        }
        hashed = protected_authority_hashes(REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _freeze(output_dir=directory, persist=True)
            _settle(directory, persist=True)
        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in PROTECTED_PATHS
        }
        self.assertEqual(before, after)
        self.assertEqual(hashed, protected_authority_hashes(REPOSITORY_ROOT))

    def test_report_path_convention(self) -> None:
        self.assertEqual(
            P43A_REPORT_RELATIVE_PATH.as_posix(),
            "report/p43a_two_phase_paper_workflow",
        )


if __name__ == "__main__":
    unittest.main()
