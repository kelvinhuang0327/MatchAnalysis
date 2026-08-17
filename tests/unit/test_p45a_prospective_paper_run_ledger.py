"""Focused unit tests for the P45A prospective paper run ledger and lifecycle."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import socket
import tempfile
import unittest

from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40A_CHAMPION_ROLE,
    P40A_POLICY_ID,
    P40APredictionRow,
    P40AMarketRow,
)
from match_analysis.application.use_cases.p44a_historical_source_adapter import (
    adapt_historical_pregame,
    adapt_historical_results,
    protected_authority_hashes,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    FORBIDDEN_PREGAME_FIELD_NAMES,
    NormalizedPregameInput,
    NormalizedResultRecord,
    load_normalized_result_input,
    write_normalized_pregame_input,
    write_normalized_result_input,
)
from match_analysis.application.use_cases.p45a_paper_run_ledger import (
    CLASSIFICATION_HISTORICAL_REHEARSAL,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
    STATE_FROZEN,
    STATE_PARTIALLY_SETTLED,
    STATE_SETTLED,
    compute_deterministic_run_id,
    compute_forward_paper_summary,
    create_p45a_paper_run,
    get_p45a_forward_summary,
    get_p45a_run_status,
    read_json_object,
    read_jsonl_objects,
    settle_p45a_paper_run,
    validate_run_classification,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _make_prospective_pregame_fixture() -> NormalizedPregameInput:
    """Construct a genuine prospective pregame fixture (clean source, future start)."""
    prediction = P40APredictionRow(
        p37_fold_id="fold_2026_01",
        p37_window="window_2026_01",
        p37_prediction_row_id="a" * 64,
        provider_namespace="mlb_official",
        provider_game_id="game_2026_08_18_01",
        game_pk=900001,
        game_number=1,
        scheduled_start_utc="2026-08-18T19:00:00Z",
        champion_model_id="champion_v1",
        champion_model_fingerprint="b" * 64,
        champion_home_probability=Decimal("0.60"),
        challenger_model_id="challenger_v1",
        challenger_model_fingerprint="c" * 64,
        challenger_home_probability=Decimal("0.55"),
    )
    market = P40AMarketRow(
        p37_fold_id="fold_2026_01",
        p37_window="window_2026_01",
        p37_prediction_row_id="a" * 64,
        provider_namespace="mlb_official",
        provider_game_id="game_2026_08_18_01",
        game_pk=900001,
        game_number=1,
        official_date="2026-08-18",
        scheduled_start_utc="2026-08-18T19:00:00Z",
        home_team="Team A",
        away_team="Team B",
        home_team_code="TMA",
        away_team_code="TMB",
        market_snapshot_id="d" * 64,
        market_observed_at_utc="2026-08-18T12:00:00Z",
        local_fetched_at_utc="2026-08-18T12:05:00Z",
        source_match_id="src_match_001",
        home_decimal_odds=Decimal("2.10"),
        away_decimal_odds=Decimal("1.80"),
    )
    return NormalizedPregameInput(
        source_identity="PROSPECTIVE_OBSERVATION_ADAPTER",
        prediction_rows=(prediction,),
        market_rows=(market,),
        exclusion_rows=(),
        source_manifest={
            "adapter": "tsl_normalized_pregame",
            "p37a": {"p37_comparisons_sha256": "1" * 64},
            "p39a": {"legacy_source_sha256": "2" * 64},
        },
        authority_hashes={},
        p37_summary={},
        p39_summary={},
        p39_source_manifest={},
    )


def _make_prospective_result_fixture() -> NormalizedResultRecord:
    return NormalizedResultRecord(
        prediction_row_id="a" * 64,
        provider_namespace="mlb_official",
        provider_game_id="game_2026_08_18_01",
        game_number=1,
        status="FINAL",
        home_score=5,
        away_score=3,
        result_observed_at_utc="2026-08-18T23:00:00Z",
        source_identity="MLB_OFFICIAL_RESULTS",
    )


class P45AProspectivePaperRunLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.authority_hashes_before = protected_authority_hashes(REPOSITORY_ROOT)

    def tearDown(self) -> None:
        authority_hashes_after = protected_authority_hashes(REPOSITORY_ROOT)
        self.assertEqual(
            self.authority_hashes_before,
            authority_hashes_after,
            "Protected historical authority hashes drifted during test execution",
        )

    def test_deterministic_run_identity(self) -> None:
        target_universe = [
            {
                "game_pk": 100,
                "game_number": 1,
                "provider_game_id": "game_100",
                "p37_prediction_row_id": "1" * 64,
            }
        ]
        run_id_1 = compute_deterministic_run_id(
            run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
            normalized_input_fingerprint="f" * 64,
            decision_bundle_fingerprint="e" * 64,
            target_universe=target_universe,
        )
        run_id_2 = compute_deterministic_run_id(
            run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
            normalized_input_fingerprint="f" * 64,
            decision_bundle_fingerprint="e" * 64,
            target_universe=target_universe,
        )
        self.assertEqual(run_id_1, run_id_2)
        self.assertTrue(run_id_1.startswith("p45a_run_"))

        # Changing classification changes run_id
        run_id_prospective = compute_deterministic_run_id(
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
            normalized_input_fingerprint="f" * 64,
            decision_bundle_fingerprint="e" * 64,
            target_universe=target_universe,
        )
        self.assertNotEqual(run_id_1, run_id_prospective)

    def test_create_run_idempotency_and_authority_conflict(self) -> None:
        pregame = adapt_historical_pregame(REPOSITORY_ROOT)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"

            # First invocation -> CREATED
            res1 = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=pregame,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root,
                created_at_utc="2026-08-17T12:00:00Z",
            )
            self.assertEqual(res1.status, "CREATED")
            self.assertEqual(res1.manifest["lifecycle_state"], STATE_FROZEN)
            self.assertEqual(res1.manifest["eligible_decision_count"], 62)
            self.assertEqual(res1.manifest["bet_count"], 22)
            self.assertEqual(res1.manifest["pass_count"], 40)
            self.assertTrue(res1.run_dir.is_dir())

            # Repeated identical invocation -> RECOGNIZED_IDENTICAL
            res2 = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=pregame,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root,
                created_at_utc="2026-08-17T12:00:00Z",
            )
            self.assertEqual(res2.status, "RECOGNIZED_IDENTICAL")
            self.assertEqual(res2.run_id, res1.run_id)

            # Modifying decision file and re-running -> conflict error
            decisions_file = res1.run_dir / "pregame_decisions.jsonl"
            decisions_file.write_text(
                decisions_file.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "P45A_RUN_AUTHORITY_CONFLICT"):
                create_p45a_paper_run(
                    REPOSITORY_ROOT,
                    pregame_input=pregame,
                    run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                    run_root=run_root,
                    created_at_utc="2026-08-17T12:00:00Z",
                )

    def test_classification_enforcement_and_historical_masquerade_rejection(self) -> None:
        historical_pregame = adapt_historical_pregame(REPOSITORY_ROOT)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"

            # Historical input with PROSPECTIVE_FORWARD_PAPER must fail closed
            with self.assertRaisesRegex(RuntimeError, "P45A_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID"):
                create_p45a_paper_run(
                    REPOSITORY_ROOT,
                    pregame_input=historical_pregame,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
                    run_root=run_root,
                    created_at_utc="2026-08-17T12:00:00Z",
                )

            # Prospective run with freeze created_at after game start must fail closed
            prospective_fixture = _make_prospective_pregame_fixture()
            with self.assertRaisesRegex(RuntimeError, "P45A_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID"):
                create_p45a_paper_run(
                    REPOSITORY_ROOT,
                    pregame_input=prospective_fixture,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
                    run_root=run_root,
                    created_at_utc="2026-08-18T20:00:00Z",  # after 19:00 scheduled start
                )

            # Valid prospective run created strictly before scheduled start succeeds
            prospective_res = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=prospective_fixture,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
                run_root=run_root,
                created_at_utc="2026-08-18T10:00:00Z",  # before 19:00 scheduled start
            )
            self.assertEqual(prospective_res.status, "CREATED")
            self.assertEqual(
                prospective_res.manifest["run_classification"],
                CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
            )

    def test_partial_settlement_and_lifecycle_progression(self) -> None:
        pregame = adapt_historical_pregame(REPOSITORY_ROOT)
        all_results = adapt_historical_results(REPOSITORY_ROOT)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"
            ledger_root = Path(temp_dir) / "ledger"

            # 1. Create run
            create_res = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=pregame,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root,
            )
            self.assertEqual(create_res.manifest["lifecycle_state"], STATE_FROZEN)

            # 2. Partial settlement with 10 results
            partial_results = all_results[:10]
            settle_res_1 = settle_p45a_paper_run(
                REPOSITORY_ROOT,
                run_dir=create_res.run_dir,
                result_input=partial_results,
                ledger_root=ledger_root,
                settled_at_utc="2026-08-17T20:00:00Z",
            )
            self.assertEqual(settle_res_1.lifecycle_state, STATE_PARTIALLY_SETTLED)
            self.assertEqual(settle_res_1.newly_settled_count, 10)
            self.assertEqual(settle_res_1.total_settled_count, 10)
            self.assertEqual(settle_res_1.pending_count, 52)

            status_1 = get_p45a_run_status(REPOSITORY_ROOT, run_dir=create_res.run_dir)
            self.assertEqual(status_1["lifecycle_state"], STATE_PARTIALLY_SETTLED)
            self.assertEqual(status_1["settled_total_count"], 10)
            self.assertEqual(status_1["pending_count"], 52)

            rehearsal_ledger_file = ledger_root / "rehearsal_ledger.jsonl"
            ledger_rows_1 = read_jsonl_objects(rehearsal_ledger_file)
            self.assertEqual(len(ledger_rows_1), 10)

            # 3. Settle remaining 52 results
            settle_res_2 = settle_p45a_paper_run(
                REPOSITORY_ROOT,
                run_dir=create_res.run_dir,
                result_input=all_results,  # contains all 62
                ledger_root=ledger_root,
                settled_at_utc="2026-08-17T23:59:59Z",
            )
            self.assertEqual(settle_res_2.lifecycle_state, STATE_SETTLED)
            self.assertEqual(settle_res_2.newly_settled_count, 52)
            self.assertEqual(settle_res_2.total_settled_count, 62)
            self.assertEqual(settle_res_2.pending_count, 0)

            status_2 = get_p45a_run_status(REPOSITORY_ROOT, run_dir=create_res.run_dir)
            self.assertEqual(status_2["lifecycle_state"], STATE_SETTLED)
            self.assertEqual(status_2["settled_total_count"], 62)
            self.assertEqual(status_2["pending_count"], 0)

            ledger_rows_2 = read_jsonl_objects(rehearsal_ledger_file)
            self.assertEqual(len(ledger_rows_2), 62)

            # Verify the first 10 rows match identically
            self.assertEqual(ledger_rows_1, ledger_rows_2[:10])

            # Repeated settlement of same results is idempotent
            settle_res_3 = settle_p45a_paper_run(
                REPOSITORY_ROOT,
                run_dir=create_res.run_dir,
                result_input=all_results,
                ledger_root=ledger_root,
            )
            self.assertEqual(settle_res_3.newly_settled_count, 0)
            self.assertEqual(settle_res_3.total_settled_count, 62)
            ledger_rows_3 = read_jsonl_objects(rehearsal_ledger_file)
            self.assertEqual(len(ledger_rows_3), 62)

    def test_historical_p44_economic_parity(self) -> None:
        pregame = adapt_historical_pregame(REPOSITORY_ROOT)
        all_results = adapt_historical_results(REPOSITORY_ROOT)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"
            ledger_root = Path(temp_dir) / "ledger"

            create_res = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=pregame,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root,
            )
            settle_res = settle_p45a_paper_run(
                REPOSITORY_ROOT,
                run_dir=create_res.run_dir,
                result_input=all_results,
                ledger_root=ledger_root,
            )
            summary = settle_res.summary
            self.assertEqual(summary["eligible_decision_count"], 62)
            self.assertEqual(summary["settled_total_count"], 62)
            self.assertEqual(summary["settled_bet_count"], 22)
            self.assertEqual(summary["settled_pass_count"], 40)
            self.assertEqual(summary["win_count"], 14)
            self.assertEqual(summary["loss_count"], 8)
            self.assertEqual(summary["push_count"], 0)
            self.assertEqual(summary["units_risked"], "22.0")
            self.assertEqual(summary["net_paper_units"], "5.90")
            self.assertEqual(
                summary["descriptive_roi"],
                "0.26818181818181818181818181818181818181818181818182",
            )
            self.assertEqual(summary["feedback_row_count"], 62)

    def test_forward_evidence_isolation_guarantee(self) -> None:
        """Verify that historical rehearsal never increments forward paper sample count."""
        pregame = adapt_historical_pregame(REPOSITORY_ROOT)
        results = adapt_historical_results(REPOSITORY_ROOT)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"
            ledger_root = Path(temp_dir) / "ledger"

            # Settle full 62 historical rehearsal games
            create_res = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=pregame,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root,
            )
            settle_res = settle_p45a_paper_run(
                REPOSITORY_ROOT,
                run_dir=create_res.run_dir,
                result_input=results,
                ledger_root=ledger_root,
            )

            # Check forward summary
            forward_summary = get_p45a_forward_summary(REPOSITORY_ROOT, ledger_root=ledger_root)
            self.assertEqual(forward_summary["forward_sample_count"], 0)
            self.assertEqual(forward_summary["run_count"], 0)
            self.assertEqual(forward_summary["frozen_decision_count"], 0)
            self.assertEqual(forward_summary["settled_bet_count"], 0)
            self.assertEqual(forward_summary["net_paper_units"], "0.00")
            self.assertIsNone(forward_summary["descriptive_roi"])

            # Now add one genuine prospective run
            prospective_pregame = _make_prospective_pregame_fixture()
            prospective_result = _make_prospective_result_fixture()

            prospective_create = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=prospective_pregame,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
                run_root=run_root,
                created_at_utc="2026-08-18T10:00:00Z",
            )
            prospective_settle = settle_p45a_paper_run(
                REPOSITORY_ROOT,
                run_dir=prospective_create.run_dir,
                result_input=(prospective_result,),
                ledger_root=ledger_root,
                settled_at_utc="2026-08-18T23:59:59Z",
            )
            self.assertEqual(prospective_settle.lifecycle_state, STATE_SETTLED)

            updated_forward_summary = get_p45a_forward_summary(
                REPOSITORY_ROOT, ledger_root=ledger_root
            )
            self.assertEqual(updated_forward_summary["forward_sample_count"], 1)
            self.assertEqual(updated_forward_summary["run_count"], 1)
            self.assertEqual(updated_forward_summary["frozen_decision_count"], 1)
            self.assertEqual(updated_forward_summary["settled_bet_count"], 1)
            self.assertEqual(updated_forward_summary["wins"], 1)
            self.assertEqual(updated_forward_summary["losses"], 0)
            self.assertEqual(updated_forward_summary["paper_units_risked"], "1.0")
            self.assertEqual(updated_forward_summary["net_paper_units"], "1.10")

    def test_tampered_decision_bundle_rejection(self) -> None:
        pregame = adapt_historical_pregame(REPOSITORY_ROOT)
        results = adapt_historical_results(REPOSITORY_ROOT)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"
            create_res = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=pregame,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root,
            )
            # Tamper decisions file
            decisions_path = create_res.run_dir / "pregame_decisions.jsonl"
            lines = decisions_path.read_text(encoding="utf-8").splitlines()
            tampered_first_row = json.loads(lines[0])
            tampered_first_row["p40_decision"]["p_home"] = "0.9999"
            lines[0] = json.dumps(tampered_first_row)
            decisions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "P43A_DECISION_BUNDLE_TAMPERED"):
                settle_p45a_paper_run(
                    REPOSITORY_ROOT,
                    run_dir=create_res.run_dir,
                    result_input=results,
                )

    def test_conflicting_result_for_already_settled_decision_fails_closed(self) -> None:
        prospective_pregame = _make_prospective_pregame_fixture()
        initial_result = _make_prospective_result_fixture()

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"
            ledger_root = Path(temp_dir) / "ledger"

            create_res = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=prospective_pregame,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
                run_root=run_root,
                created_at_utc="2026-08-18T10:00:00Z",
            )
            # Initial settlement
            settle_p45a_paper_run(
                REPOSITORY_ROOT,
                run_dir=create_res.run_dir,
                result_input=(initial_result,),
                ledger_root=ledger_root,
            )

            # Conflicting result for same game (different winner / score)
            conflicting_result = replace(
                initial_result,
                home_score=1,
                away_score=8,  # Away wins instead of Home
            )
            with self.assertRaisesRegex(RuntimeError, "P45A_CONFLICTING_RESULT_REJECTED"):
                settle_p45a_paper_run(
                    REPOSITORY_ROOT,
                    run_dir=create_res.run_dir,
                    result_input=(conflicting_result,),
                    ledger_root=ledger_root,
                )

    def test_network_isolation_guarantee(self) -> None:
        """Verify that P45A execution never calls network sockets."""
        def guarded_socket(*args: object, **kwargs: object) -> socket.socket:
            raise AssertionError("network access forbidden during P45A execution")

        original_socket = socket.socket
        socket.socket = guarded_socket  # type: ignore[assignment]
        try:
            pregame = adapt_historical_pregame(REPOSITORY_ROOT)
            results = adapt_historical_results(REPOSITORY_ROOT)
            with tempfile.TemporaryDirectory() as temp_dir:
                run_root = Path(temp_dir) / "runs"
                ledger_root = Path(temp_dir) / "ledger"
                create_res = create_p45a_paper_run(
                    REPOSITORY_ROOT,
                    pregame_input=pregame,
                    run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                    run_root=run_root,
                )
                settle_res = settle_p45a_paper_run(
                    REPOSITORY_ROOT,
                    run_dir=create_res.run_dir,
                    result_input=results,
                    ledger_root=ledger_root,
                )
                self.assertEqual(settle_res.lifecycle_state, STATE_SETTLED)
        finally:
            socket.socket = original_socket


    def test_shuffled_input_determinism(self) -> None:
        """Verify that shuffled input order produces the exact same run_id and pregame bundle."""
        pregame = adapt_historical_pregame(REPOSITORY_ROOT)
        reversed_predictions = tuple(reversed(pregame.prediction_rows))
        reversed_markets = tuple(reversed(pregame.market_rows))
        shuffled_pregame = replace(
            pregame,
            prediction_rows=reversed_predictions,
            market_rows=reversed_markets,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root_1 = Path(temp_dir) / "runs_1"
            run_root_2 = Path(temp_dir) / "runs_2"

            res1 = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=pregame,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root_1,
            )
            res2 = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=shuffled_pregame,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root_2,
            )
            self.assertEqual(res1.run_id, res2.run_id)
            self.assertEqual(
                res1.manifest["decision_bundle_fingerprint"],
                res2.manifest["decision_bundle_fingerprint"],
            )
            self.assertEqual(res1.pregame_decisions, res2.pregame_decisions)

    def test_non_final_result_rejection(self) -> None:
        """Verify non-final results fail closed according to contract."""
        prospective_pregame = _make_prospective_pregame_fixture()
        in_progress_result = replace(
            _make_prospective_result_fixture(),
            status="IN_PROGRESS",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"
            create_res = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=prospective_pregame,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
                run_root=run_root,
                created_at_utc="2026-08-18T10:00:00Z",
            )
            # Writing in_progress result to jsonl and attempting load must fail closed
            result_file = Path(temp_dir) / "bad_results.jsonl"
            result_file.write_text(
                json.dumps(in_progress_result.to_payload()) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "P43A_NON_FINAL_RESULT_FAIL_CLOSED"):
                settle_p45a_paper_run(
                    REPOSITORY_ROOT,
                    run_dir=create_res.run_dir,
                    result_input=result_file,
                )


if __name__ == "__main__":
    unittest.main()

