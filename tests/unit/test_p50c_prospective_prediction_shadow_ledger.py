"""Focused unit tests for the P50C prospective prediction shadow ledger and lifecycle."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40APredictionRow,
    P40AMarketRow,
)
from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40APredictionRow,
)
from match_analysis.application.use_cases.p44a_historical_source_adapter import (
    adapt_historical_pregame,
    adapt_historical_results,
    protected_authority_hashes,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    NormalizedPregameInput,
    NormalizedResultRecord,
    write_normalized_pregame_input,
    write_normalized_result_input,
)
from match_analysis.application.use_cases.p45a_paper_run_ledger import (
    get_p45a_forward_summary,
)
from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    CLASSIFICATION_HISTORICAL_REHEARSAL,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P50C_CLAIMS,
    P50C_FORWARD_SUMMARY_SCHEMA,
    P50C_LEDGER_RECORD_SCHEMA,
    P50C_RUN_MANIFEST_SCHEMA,
    P50C_SETTLEMENT_SUMMARY_SCHEMA,
    STATE_FROZEN,
    STATE_PARTIALLY_SETTLED,
    STATE_SETTLED,
    build_frozen_prediction_record,
    calculate_log_loss,
    canonical_prediction_fingerprint,
    compute_deterministic_prediction_run_id,
    compute_expected_calibration_error,
    compute_forward_prediction_summary,
    compute_prediction_row_fingerprint,
    create_p50c_prediction_run,
    get_p50c_forward_summary,
    get_p50c_run_status,
    read_json_object,
    read_jsonl_objects,
    reject_pregame_contamination,
    settle_p50c_prediction_run,
    validate_prediction_run_classification,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _make_prediction_row(
    pred_id: str,
    game_id: str,
    p_home: Decimal,
    start_utc: str = "2026-08-18T19:00:00Z",
) -> P40APredictionRow:
    return P40APredictionRow(
        p37_fold_id="fold_2026_01",
        p37_window="window_2026_01",
        p37_prediction_row_id=pred_id,
        provider_namespace="mlb_official",
        provider_game_id=game_id,
        game_pk=900001,
        game_number=1,
        scheduled_start_utc=start_utc,
        champion_model_id="champion_v1",
        champion_model_fingerprint="b" * 64,
        champion_home_probability=p_home,
        challenger_model_id="challenger_v1",
        challenger_model_fingerprint="c" * 64,
        challenger_home_probability=p_home,
    )


def _make_prospective_pregame_fixture() -> NormalizedPregameInput:
    pred1 = _make_prediction_row("a" * 64, "game_01", Decimal("0.65"))
    pred2 = _make_prediction_row("b" * 64, "game_02", Decimal("0.40"))
    return NormalizedPregameInput(
        source_identity="PROSPECTIVE_MLB_SCHEDULE_FEED",
        prediction_rows=(pred1, pred2),
        market_rows=(),
        exclusion_rows=(),
        source_manifest={"adapter": "mlb_pregame_feed", "version": "1.0"},
        authority_hashes={},
        p37_summary={},
        p39_summary={},
        p39_source_manifest={},
    )


def _make_result_record(
    pred_id: str,
    game_id: str,
    home_score: int,
    away_score: int,
) -> NormalizedResultRecord:
    return NormalizedResultRecord(
        prediction_row_id=pred_id,
        provider_namespace="mlb_official",
        provider_game_id=game_id,
        game_number=1,
        status="FINAL",
        home_score=home_score,
        away_score=away_score,
        result_observed_at_utc="2026-08-18T23:00:00Z",
        source_identity="MLB_OFFICIAL_RESULTS",
    )


class P50CProspectivePredictionShadowLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.authority_hashes_before = protected_authority_hashes(REPOSITORY_ROOT)

    def tearDown(self) -> None:
        authority_hashes_after = protected_authority_hashes(REPOSITORY_ROOT)
        self.assertEqual(
            self.authority_hashes_before,
            authority_hashes_after,
            "Protected historical authority hashes drifted during test execution",
        )

    def test_deterministic_prediction_run_id_and_fingerprints(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        norm_fp = canonical_prediction_fingerprint(
            source_identity=bundle.source_identity,
            prediction_rows=bundle.prediction_rows,
            exclusion_rows=bundle.exclusion_rows,
            source_manifest=bundle.source_manifest,
        )
        self.assertEqual(len(norm_fp), 64)

        pred_fps = [compute_prediction_row_fingerprint(r) for r in bundle.prediction_rows]
        self.assertEqual(len(pred_fps), 2)
        self.assertTrue(all(len(fp) == 64 for fp in pred_fps))

        target_universe = [
            {
                "game_pk": r.game_pk,
                "game_number": r.game_number,
                "provider_game_id": r.provider_game_id,
                "p37_prediction_row_id": r.p37_prediction_row_id,
            }
            for r in bundle.prediction_rows
        ]
        run_id_1 = compute_deterministic_prediction_run_id(
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            normalized_input_fingerprint=norm_fp,
            prediction_bundle_fingerprint="bundle_fp_test",
            target_universe=target_universe,
        )
        run_id_2 = compute_deterministic_prediction_run_id(
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            normalized_input_fingerprint=norm_fp,
            prediction_bundle_fingerprint="bundle_fp_test",
            target_universe=target_universe,
        )
        self.assertEqual(run_id_1, run_id_2)
        self.assertTrue(run_id_1.startswith("p50c_run_"))

    def test_no_betting_or_odds_or_result_fields_in_frozen_record(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        row = bundle.prediction_rows[0]
        record = build_frozen_prediction_record(
            row,
            run_id="p50c_run_test",
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            created_at_utc="2026-08-18T12:00:00Z",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
        )

        # Invariant checks: zero result fields, zero betting fields
        self.assertNotIn("actual_winner", record)
        self.assertNotIn("home_score", record)
        self.assertNotIn("away_score", record)
        self.assertNotIn("home_decimal_odds", record)
        self.assertNotIn("ev_home", record)
        self.assertNotIn("bet_or_pass", record)
        self.assertNotIn("paper_stake_units", record)

        # Proper model identity preserved
        self.assertEqual(record["model_identity"]["p_home"], str(row.champion_home_probability))
        self.assertEqual(record["model_identity"]["selection"], "HOME")
        self.assertEqual(record["model_identity"]["predicted_winner"], "New York Yankees")


    def test_reject_pregame_contamination_triggers_on_forbidden_fields(self) -> None:
        with self.assertRaises(RuntimeError) as cm:
            reject_pregame_contamination({"p_home": "0.6", "actual_winner": "HOME"})
        self.assertIn("P50C_PREGAME_RESULT_CONTAMINATION_REJECTED", str(cm.exception))

        with self.assertRaises(RuntimeError) as cm:
            reject_pregame_contamination({"p_home": "0.6", "home_decimal_odds": "1.95"})
        self.assertIn("P50C_PREGAME_BETTING_CONTAMINATION_REJECTED", str(cm.exception))

    def test_temporal_guards_reject_past_or_historical_freeze_in_prospective_mode(self) -> None:
        bundle = _make_prospective_pregame_fixture()

        # Creation time after game scheduled start
        with self.assertRaises(RuntimeError) as cm:
            validate_prediction_run_classification(
                CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                prediction_rows=bundle.prediction_rows,
                source_identity="PROSPECTIVE_SOURCE",
                created_at_utc="2026-08-18T20:00:00Z",  # Game is at 19:00:00Z
            )
        self.assertIn("P50C_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID", str(cm.exception))

        # Historical source identity classified as prospective
        with self.assertRaises(RuntimeError) as cm:
            validate_prediction_run_classification(
                CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                prediction_rows=bundle.prediction_rows,
                source_identity="HISTORICAL_REHEARSAL_DATASET",
                created_at_utc="2026-08-18T12:00:00Z",
            )
        self.assertIn("P50C_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID", str(cm.exception))

    def test_create_p50c_prediction_run_lifecycle_and_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle = _make_prospective_pregame_fixture()
            input_file = temp_path / "normalized_pregame_input.json"
            write_normalized_pregame_input(input_file, bundle)

            run_root = temp_path / "runs"

            # Create run
            res1 = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=input_file,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=run_root,
                created_at_utc="2026-08-18T12:00:00Z",
            )
            self.assertEqual(res1.status, "CREATED")
            self.assertEqual(res1.manifest["lifecycle_state"], STATE_FROZEN)
            self.assertEqual(res1.manifest["eligible_prediction_count"], 2)
            self.assertEqual(res1.manifest["claims"], P50C_CLAIMS)

            # Re-running on identical inputs returns RECOGNIZED_IDENTICAL
            res2 = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=input_file,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=run_root,
                created_at_utc="2026-08-18T12:00:00Z",
            )
            self.assertEqual(res2.status, "RECOGNIZED_IDENTICAL")
            self.assertEqual(res2.run_id, res1.run_id)

            # Inspect run status
            status = get_p50c_run_status(REPOSITORY_ROOT, run_dir=res1.run_dir)
            self.assertEqual(status["lifecycle_state"], STATE_FROZEN)
            self.assertEqual(status["pending_count"], 2)
            self.assertEqual(status["settled_total_count"], 0)

            # Tampering with run manifest raises P50C_RUN_AUTHORITY_CONFLICT
            manifest_path = res1.run_dir / "run_manifest.json"
            tampered = read_json_object(manifest_path)
            tampered["prediction_bundle_fingerprint"] = "0" * 64
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")

            with self.assertRaises(RuntimeError) as cm:
                create_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    pregame_input=input_file,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                    run_root=run_root,
                    created_at_utc="2026-08-18T12:00:00Z",
                )
            self.assertIn("P50C_RUN_AUTHORITY_CONFLICT", str(cm.exception))

    def test_settle_prediction_run_partial_and_full_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle = _make_prospective_pregame_fixture()
            input_file = temp_path / "normalized_pregame_input.json"
            write_normalized_pregame_input(input_file, bundle)

            run_root = temp_path / "runs"
            ledger_root = temp_path / "ledger"

            create_res = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=input_file,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=run_root,
                created_at_utc="2026-08-18T12:00:00Z",
            )
            run_dir = create_res.run_dir

            # Partial settlement: only game 1 finishes
            # game_01: p_home=0.65 -> selection=HOME. Outcome: home 6, away 2 (HOME wins) -> is_correct=True
            res_game1 = _make_result_record("a" * 64, "game_01", 6, 2)
            partial_results_file = temp_path / "partial_results.jsonl"
            write_normalized_result_input(partial_results_file, [res_game1])

            settle_1 = settle_p50c_prediction_run(
                REPOSITORY_ROOT,
                run_dir=run_dir,
                result_input=partial_results_file,
                ledger_root=ledger_root,
                settled_at_utc="2026-08-18T22:30:00Z",
            )
            self.assertEqual(settle_1.lifecycle_state, STATE_PARTIALLY_SETTLED)
            self.assertEqual(settle_1.newly_settled_count, 1)
            self.assertEqual(settle_1.total_settled_count, 1)
            self.assertEqual(settle_1.pending_count, 1)
            self.assertEqual(settle_1.summary["correct_count"], 1)
            self.assertEqual(settle_1.summary["accuracy"], "1")
            self.assertEqual(settle_1.forward_summary["PREDICTION_FORWARD_SAMPLE_COUNT"], 1)

            # Full settlement: game 2 also finishes
            # game_02: p_home=0.40, p_away=0.60 -> selection=AWAY. Outcome: home 5, away 3 (HOME wins) -> is_correct=False
            res_game2 = _make_result_record("b" * 64, "game_02", 5, 3)
            all_results_file = temp_path / "all_results.jsonl"
            write_normalized_result_input(all_results_file, [res_game1, res_game2])

            settle_2 = settle_p50c_prediction_run(
                REPOSITORY_ROOT,
                run_dir=run_dir,
                result_input=all_results_file,
                ledger_root=ledger_root,
                settled_at_utc="2026-08-18T23:45:00Z",
            )
            self.assertEqual(settle_2.lifecycle_state, STATE_SETTLED)
            self.assertEqual(settle_2.newly_settled_count, 1)
            self.assertEqual(settle_2.total_settled_count, 2)
            self.assertEqual(settle_2.pending_count, 0)
            self.assertEqual(settle_2.summary["correct_count"], 1)
            self.assertEqual(settle_2.summary["incorrect_count"], 1)
            self.assertEqual(settle_2.summary["accuracy"], "0.5")
            self.assertEqual(settle_2.forward_summary["PREDICTION_FORWARD_SAMPLE_COUNT"], 2)


            # Verify forward prediction summary
            summary = get_p50c_forward_summary(REPOSITORY_ROOT, ledger_root=ledger_root)
            self.assertEqual(summary["PREDICTION_FORWARD_SAMPLE_COUNT"], 2)
            self.assertEqual(summary["run_count"], 1)
            self.assertEqual(summary["correct_count"], 1)
            self.assertEqual(summary["incorrect_count"], 1)
            self.assertIsNotNone(summary["brier_score"])
            self.assertIsNotNone(summary["log_loss"])
            self.assertIsNotNone(summary["expected_calibration_error"])

    def test_conflicting_final_results_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle = _make_prospective_pregame_fixture()
            input_file = temp_path / "normalized_pregame_input.json"
            write_normalized_pregame_input(input_file, bundle)

            run_root = temp_path / "runs"
            ledger_root = temp_path / "ledger"

            create_res = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=input_file,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=run_root,
                created_at_utc="2026-08-18T12:00:00Z",
            )

            # First settlement with game_01 home score 6, away 2 (HOME wins)
            res_game1 = _make_result_record("a" * 64, "game_01", 6, 2)
            write_normalized_result_input(temp_path / "res1.jsonl", [res_game1])
            settle_p50c_prediction_run(
                REPOSITORY_ROOT,
                run_dir=create_res.run_dir,
                result_input=temp_path / "res1.jsonl",
                ledger_root=ledger_root,
            )

            # Conflicting result: same game now reported as home 1, away 7 (AWAY wins)
            conflicting_res = _make_result_record("a" * 64, "game_01", 1, 7)
            write_normalized_result_input(temp_path / "conflict.jsonl", [conflicting_res])

            with self.assertRaises(RuntimeError) as cm:
                settle_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    run_dir=create_res.run_dir,
                    result_input=temp_path / "conflict.jsonl",
                    ledger_root=ledger_root,
                )
            self.assertIn("P50C_CONFLICTING_RESULT_REJECTED", str(cm.exception))

    def test_evaluation_arithmetic_accuracy_brier_log_loss_ece(self) -> None:
        # 1. Log loss calculation
        # p=0.8, target=1 -> -ln(0.8) ~= 0.223143551314
        loss_win = calculate_log_loss(Decimal("0.8"), 1)
        self.assertAlmostEqual(float(loss_win), 0.22314355, places=5)

        # p=0.8, target=0 -> -ln(0.2) ~= 1.609437912434
        loss_loss = calculate_log_loss(Decimal("0.8"), 0)
        self.assertAlmostEqual(float(loss_loss), 1.60943791, places=5)

        # 2. ECE calculation
        # 4 predictions: probs [0.6, 0.7, 0.8, 0.9], outcomes [1, 1, 1, 1]
        probs = [Decimal("0.65"), Decimal("0.68"), Decimal("0.82"), Decimal("0.88")]
        outcomes = [1, 1, 1, 1]
        ece = compute_expected_calibration_error(probs, outcomes, num_bins=10)
        self.assertIsInstance(ece, Decimal)
        self.assertTrue(Decimal("0") <= ece <= Decimal("1"))

    def test_historical_rehearsal_does_not_increment_forward_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle = _make_prospective_pregame_fixture()
            input_file = temp_path / "rehearsal_pregame.json"
            write_normalized_pregame_input(input_file, bundle)

            run_root = temp_path / "runs"
            ledger_root = temp_path / "ledger"

            create_res = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=input_file,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root,
                created_at_utc="2026-08-18T12:00:00Z",
            )

            res1 = _make_result_record("a" * 64, "game_01", 5, 2)
            results_file = temp_path / "rehearsal_results.jsonl"
            write_normalized_result_input(results_file, [res1])

            settle_res = settle_p50c_prediction_run(
                REPOSITORY_ROOT,
                run_dir=create_res.run_dir,
                result_input=results_file,
                ledger_root=ledger_root,
            )
            # Rehearsals write to rehearsal_prediction_ledger.jsonl and forward sample count is 0
            self.assertEqual(settle_res.forward_summary["PREDICTION_FORWARD_SAMPLE_COUNT"], 0)

            summary = get_p50c_forward_summary(REPOSITORY_ROOT, ledger_root=ledger_root)
            self.assertEqual(summary["PREDICTION_FORWARD_SAMPLE_COUNT"], 0)
            self.assertEqual(summary["run_count"], 0)

    def test_betting_forward_sample_count_is_untouched(self) -> None:
        # Check betting summary in P45A report directory
        betting_summary = get_p45a_forward_summary(REPOSITORY_ROOT)
        initial_betting_count = betting_summary.get("forward_sample_count", 0)

        # Run P50 prospective workflow in a separate directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle = _make_prospective_pregame_fixture()
            input_file = temp_path / "pregame.json"
            write_normalized_pregame_input(input_file, bundle)

            create_res = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=input_file,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=temp_path / "runs",
                created_at_utc="2026-08-18T12:00:00Z",
            )
            res1 = _make_result_record("a" * 64, "game_01", 4, 1)
            results_file = temp_path / "results.jsonl"
            write_normalized_result_input(results_file, [res1])

            settle_res = settle_p50c_prediction_run(
                REPOSITORY_ROOT,
                run_dir=create_res.run_dir,
                result_input=results_file,
                ledger_root=temp_path / "ledger",
            )
            self.assertEqual(settle_res.forward_summary["PREDICTION_FORWARD_SAMPLE_COUNT"], 1)

        # Prove betting summary remains invariant
        betting_summary_after = get_p45a_forward_summary(REPOSITORY_ROOT)
        self.assertEqual(
            betting_summary_after.get("forward_sample_count", 0),
            initial_betting_count,
            "Betting forward_sample_count was mutated by prediction workflow",
        )


if __name__ == "__main__":
    unittest.main()
