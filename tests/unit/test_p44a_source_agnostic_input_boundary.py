"""Focused tests for the P44A source-independent paper-workflow input boundary."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.p43a_postgame_settle import (
    run_p43a_postgame_settle,
    settle_p43a_frozen_decisions,
)
from match_analysis.application.use_cases.p43a_pregame_freeze import (
    freeze_p43a_pregame_decisions,
    run_p43a_pregame_freeze,
)
from match_analysis.application.use_cases.p44a_historical_source_adapter import (
    adapt_historical_pregame,
    adapt_historical_results,
    protected_authority_hashes,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    FORBIDDEN_PREGAME_FIELD_NAMES,
    NormalizedPregameInput,
    load_normalized_pregame_input,
    load_normalized_result_input,
    parse_normalized_pregame_payload,
    project_normalized_results,
    write_normalized_pregame_input,
    write_normalized_result_input,
)
from match_analysis.baseball.domain.paper_moneyline_bet_pass import DECISION_BET


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREGAME_CORE = (
    REPOSITORY_ROOT / "src/match_analysis/application/use_cases/p43a_pregame_freeze.py"
)
POSTGAME_CORE = (
    REPOSITORY_ROOT / "src/match_analysis/application/use_cases/p43a_postgame_settle.py"
)
NORMALIZED_INPUT = (
    REPOSITORY_ROOT
    / "src/match_analysis/application/use_cases/p44a_normalized_workflow_input.py"
)
ADAPTER = (
    REPOSITORY_ROOT
    / "src/match_analysis/application/use_cases/p44a_historical_source_adapter.py"
)
CLI_PATH = (
    REPOSITORY_ROOT
    / "src/match_analysis/interfaces/cli/run_p43a_two_phase_paper_workflow.py"
)
COMMITTED_PREGAME_SUMMARY = (
    REPOSITORY_ROOT / "report/p43a_two_phase_paper_workflow/pregame_summary.json"
)
HISTORICAL_P37 = "report/p37a_rolling_walk_forward_oos"
HISTORICAL_P39 = "report/p39a_tsl_moneyline_market_join"
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
    REPOSITORY_ROOT / "report/p43a_two_phase_paper_workflow/pregame_decisions.jsonl",
    REPOSITORY_ROOT / "report/p43a_two_phase_paper_workflow/pregame_summary.json",
    REPOSITORY_ROOT / "report/p43a_two_phase_paper_workflow/postgame_summary.json",
)


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def _one_game_pregame(
    bundle: NormalizedPregameInput | None = None,
    **market_overrides: object,
) -> NormalizedPregameInput:
    source = bundle or adapt_historical_pregame(REPOSITORY_ROOT)
    market = source.market_rows[0]
    if market_overrides:
        market = replace(market, **market_overrides)
    prediction = next(
        row
        for row in source.prediction_rows
        if row.p37_prediction_row_id == market.p37_prediction_row_id
    )
    return NormalizedPregameInput(
        source_identity="P44A_ISOLATED_NORMALIZED_FIXTURE",
        prediction_rows=(prediction,),
        market_rows=(market,),
        exclusion_rows=(),
        source_manifest=source.source_manifest,
        authority_hashes=source.authority_hashes,
        p37_summary={},
        p39_summary={},
        p39_source_manifest={},
    )


class P44ASourceAgnosticBoundaryTests(unittest.TestCase):
    def test_normalized_pregame_fixture_is_accepted_and_rejects_outcomes(self) -> None:
        fixture = _one_game_pregame()
        payload = fixture.to_payload()
        parsed = parse_normalized_pregame_payload(payload)
        self.assertEqual(parsed.source_identity, "P44A_ISOLATED_NORMALIZED_FIXTURE")
        self.assertEqual(len(parsed.prediction_rows), 1)
        self.assertEqual(len(parsed.market_rows), 1)
        self.assertNotIn("live", parsed.source_identity.lower())
        for name in FORBIDDEN_PREGAME_FIELD_NAMES:
            tainted = json.loads(json.dumps(payload))
            tainted["predictions"][0][name] = "HOME"
            with self.assertRaisesRegex(ValueError, "P44A_PREGAME_OUTCOME_FIELDS_REJECTED"):
                parse_normalized_pregame_payload(tainted)

    def test_temporal_guards_remain_enforced_on_normalized_pregame(self) -> None:
        fixture = _one_game_pregame()
        late = replace(
            fixture.market_rows[0],
            market_observed_at_utc=fixture.market_rows[0].scheduled_start_utc,
        )
        payload = replace(fixture, market_rows=(late,)).to_payload()
        with self.assertRaisesRegex(ValueError, "not strictly pregame"):
            parse_normalized_pregame_payload(payload)
        fetched_late = replace(
            fixture.market_rows[0],
            local_fetched_at_utc=fixture.market_rows[0].scheduled_start_utc,
        )
        fetched_payload = replace(fixture, market_rows=(fetched_late,)).to_payload()
        with self.assertRaisesRegex(ValueError, "not strictly pregame"):
            parse_normalized_pregame_payload(fetched_payload)

    def test_pregame_freeze_from_isolated_fixture_without_result_input(self) -> None:
        fixture = _one_game_pregame()
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = write_normalized_pregame_input(directory / "pregame_input.json", fixture)
            result = run_p43a_pregame_freeze(
                REPOSITORY_ROOT,
                pregame_input=path,
                output_dir=directory / "bundle",
                persist=True,
            )
            self.assertEqual(result.summary["workflow_decision_count"], 1)
            self.assertEqual(result.summary["settled_bet_count"], 0)
            self.assertEqual(result.summary["unresolved_result_count"], 1)
            self.assertEqual(len(result.authority.outcome_rows), 0)
            self.assertFalse((directory / "bundle" / "workflow_ledger.jsonl").exists())

    def test_postgame_missing_and_non_final_result_fail_closed(self) -> None:
        fixture = _one_game_pregame()
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            run_p43a_pregame_freeze(
                REPOSITORY_ROOT,
                pregame_input=fixture,
                output_dir=directory,
                persist=True,
            )
            missing = directory / "absent.jsonl"
            with self.assertRaisesRegex(RuntimeError, "P43A_MISSING_RESULT_FAIL_CLOSED"):
                run_p43a_postgame_settle(
                    REPOSITORY_ROOT,
                    output_dir=directory,
                    result_input=missing,
                    persist=False,
                )
            with self.assertRaisesRegex(RuntimeError, "P43A_MISSING_RESULT_FAIL_CLOSED"):
                run_p43a_postgame_settle(
                    REPOSITORY_ROOT,
                    output_dir=directory,
                    persist=False,
                )
            historical = adapt_historical_results(REPOSITORY_ROOT)
            matching = next(
                row
                for row in historical
                if row.prediction_row_id == fixture.prediction_rows[0].p37_prediction_row_id
            )
            non_final = directory / "non_final.jsonl"
            non_final.write_text(
                json.dumps({**matching.to_payload(), "status": "LIVE"}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "P43A_NON_FINAL_RESULT_FAIL_CLOSED"):
                load_normalized_result_input(non_final)

    def test_independent_result_input_and_fingerprint_invariance(self) -> None:
        historical = adapt_historical_pregame(REPOSITORY_ROOT)
        results = adapt_historical_results(REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            pregame_path = write_normalized_pregame_input(
                directory / "pregame_input.json", historical
            )
            result_path = write_normalized_result_input(
                directory / "result_input.jsonl", results
            )
            pregame = run_p43a_pregame_freeze(
                REPOSITORY_ROOT,
                pregame_input=pregame_path,
                output_dir=directory / "bundle",
                persist=True,
            )
            fingerprints = [row.decision_id for row in pregame.decisions]
            committed = json.loads(
                COMMITTED_PREGAME_SUMMARY.read_text(encoding="utf-8")
            )
            self.assertEqual(fingerprints, committed["decision_fingerprints"])
            self.assertEqual(pregame.summary["bundle_fingerprint"], committed["bundle_fingerprint"])
            postgame = run_p43a_postgame_settle(
                REPOSITORY_ROOT,
                output_dir=directory / "bundle",
                result_input=result_path,
                persist=True,
            )
            self.assertEqual(
                [row.decision.decision_id for row in postgame.settlements],
                fingerprints,
            )
            flipped = tuple(
                replace(
                    row,
                    home_score=row.away_score,
                    away_score=row.home_score,
                )
                for row in results
            )
            mutated = settle_p43a_frozen_decisions(
                pregame.decisions, project_normalized_results(flipped)
            )
            self.assertEqual(
                [row.decision.decision_id for row in mutated],
                fingerprints,
            )
            self.assertNotEqual(
                [row.to_projection() for row in postgame.settlements],
                [row.to_projection() for row in mutated],
            )

    def test_phase2_source_does_not_rebuild_decisions(self) -> None:
        source = POSTGAME_CORE.read_text(encoding="utf-8")
        self.assertNotIn("build_p40a_decisions", source)
        self.assertNotIn("freeze_champion_decisions", source)
        self.assertNotIn("load_p43a_pregame_authority", source)

    def test_workflow_core_does_not_hardcode_historical_source_paths(self) -> None:
        for path in (PREGAME_CORE, POSTGAME_CORE, NORMALIZED_INPUT):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(HISTORICAL_P37, text)
            self.assertNotIn(HISTORICAL_P39, text)
        adapter = ADAPTER.read_text(encoding="utf-8")
        self.assertIn("P37A_REPORT_RELATIVE_PATH", adapter)
        self.assertIn("P39A_REPORT_RELATIVE_PATH", adapter)

    def test_historical_adapter_two_pass_parity(self) -> None:
        first_pregame = adapt_historical_pregame(REPOSITORY_ROOT)
        second_pregame = adapt_historical_pregame(REPOSITORY_ROOT)
        self.assertEqual(first_pregame.to_payload(), second_pregame.to_payload())
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            first = run_p43a_pregame_freeze(
                REPOSITORY_ROOT,
                pregame_input=first_pregame,
                output_dir=directory / "one",
                persist=True,
            )
            shuffled = freeze_p43a_pregame_decisions(
                first_pregame.to_authority(REPOSITORY_ROOT),
                market_rows=tuple(reversed(first_pregame.market_rows)),
                prediction_rows=tuple(reversed(first_pregame.prediction_rows)),
            )
            self.assertEqual(
                [row.to_projection() for row in first.decisions],
                [row.to_projection() for row in shuffled],
            )
            second = run_p43a_pregame_freeze(
                REPOSITORY_ROOT,
                pregame_input=second_pregame,
                output_dir=directory / "two",
                persist=True,
            )
            self.assertEqual(first.summary["workflow_decision_count"], 62)
            self.assertEqual(first.summary["bet_count"], 22)
            self.assertEqual(first.summary["pass_count"], 40)
            self.assertEqual(first.summary["settled_bet_count"], 0)
            self.assertEqual(second.records, first.records)
            result_path = write_normalized_result_input(
                directory / "results.jsonl", adapt_historical_results(REPOSITORY_ROOT)
            )
            first_settle = run_p43a_postgame_settle(
                REPOSITORY_ROOT,
                output_dir=directory / "one",
                result_input=result_path,
                persist=True,
            )
            second_settle = run_p43a_postgame_settle(
                REPOSITORY_ROOT,
                output_dir=directory / "two",
                result_input=result_path,
                persist=True,
            )
            self.assertEqual(first_settle.summary["settled_bet_count"], 22)
            self.assertEqual(first_settle.summary["win_count"], 14)
            self.assertEqual(first_settle.summary["loss_count"], 8)
            self.assertEqual(first_settle.summary["push_count"], 0)
            self.assertEqual(first_settle.summary["units_risked"], "22.0")
            self.assertEqual(first_settle.summary["net_paper_units"], "5.90")
            self.assertEqual(first_settle.summary["feedback_row_count"], 62)
            self.assertEqual(first_settle.ledger_rows, second_settle.ledger_rows)
            repeated = run_p43a_postgame_settle(
                REPOSITORY_ROOT,
                output_dir=directory / "one",
                result_input=result_path,
                persist=True,
            )
            self.assertEqual(repeated.write_status["workflow_ledger.jsonl"], "RECOGNIZED_IDENTICAL")
            committed = json.loads(
                COMMITTED_PREGAME_SUMMARY.read_text(encoding="utf-8")
            )
            self.assertEqual(
                first.summary["decision_fingerprints"],
                committed["decision_fingerprints"],
            )

    def test_source_has_no_network_imports(self) -> None:
        for path in (PREGAME_CORE, POSTGAME_CORE, NORMALIZED_INPUT, ADAPTER, CLI_PATH):
            imported = _imported_roots(path)
            for forbidden in ("urllib", "requests", "http", "socket", "ssl"):
                self.assertNotIn(forbidden, imported)
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("urlopen", source)
            self.assertNotIn("verify=False", source)

    def test_protected_authorities_remain_invariant(self) -> None:
        before = {path: path.read_bytes() for path in PROTECTED_PATHS}
        hashed = protected_authority_hashes(REPOSITORY_ROOT)
        historical = adapt_historical_pregame(REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            run_p43a_pregame_freeze(
                REPOSITORY_ROOT,
                pregame_input=historical,
                output_dir=directory,
                persist=True,
            )
            run_p43a_postgame_settle(
                REPOSITORY_ROOT,
                output_dir=directory,
                outcome_rows=project_normalized_results(
                    adapt_historical_results(REPOSITORY_ROOT)
                ),
                persist=True,
            )
        after = {path: path.read_bytes() for path in PROTECTED_PATHS}
        self.assertEqual(before, after)
        self.assertEqual(hashed, protected_authority_hashes(REPOSITORY_ROOT))

    def test_core_freeze_does_not_open_historical_artifact_files(self) -> None:
        opened: list[str] = []
        original = Path.read_text

        def tracked(self: Path, *args: object, **kwargs: object) -> str:
            text = str(self)
            if HISTORICAL_P37 in text or HISTORICAL_P39 in text:
                opened.append(text)
            return original(self, *args, **kwargs)

        historical = adapt_historical_pregame(REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            fixture = write_normalized_pregame_input(
                directory / "pregame_input.json", historical
            )
            Path.read_text = tracked  # type: ignore[method-assign]
            try:
                run_p43a_pregame_freeze(
                    REPOSITORY_ROOT,
                    pregame_input=load_normalized_pregame_input(fixture),
                    output_dir=directory / "bundle",
                    persist=False,
                )
            finally:
                Path.read_text = original  # type: ignore[method-assign]
        self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
