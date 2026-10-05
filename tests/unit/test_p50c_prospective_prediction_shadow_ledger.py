"""Focused unit tests for the P50C prospective prediction shadow ledger and lifecycle."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

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
    P66A_OFFICIAL_DATE_CORRECTION_SCHEMA,
    STATE_FROZEN,
    STATE_PARTIALLY_SETTLED,
    STATE_SETTLED,
    apply_official_date_corrections,
    build_frozen_prediction_record,
    build_p50c_prediction_ledger_record,
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
        game_type="R",
    )


def _make_prospective_pregame_fixture() -> NormalizedPregameInput:
    pred1 = _make_prediction_row("a" * 64, "game_01", Decimal("0.65"))
    pred2 = _make_prediction_row("b" * 64, "game_02", Decimal("0.40"))
    return NormalizedPregameInput(
        source_identity="PROSPECTIVE_MLB_SCHEDULE_FEED",
        prediction_rows=(pred1, pred2),
        market_rows=(),
        exclusion_rows=(),
        source_manifest={
            "adapter": "mlb_pregame_feed",
            "version": "1.0",
            "target_date": "2026-08-18",
            "official_dates_by_prediction_row_id": {
                pred1.p37_prediction_row_id: "2026-08-18",
                pred2.p37_prediction_row_id: "2026-08-18",
            },
            "team_identities_by_prediction_row_id": {
                prediction.p37_prediction_row_id: {
                    "home_team": "New York Yankees",
                    "away_team": "Los Angeles Dodgers",
                    "home_team_code": "NYY",
                    "away_team_code": "LAD",
                }
                for prediction in (pred1, pred2)
            },
        },
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
        game_type="R",
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

        pred_fps = [
            compute_prediction_row_fingerprint(r, official_date="2026-08-18")
            for r in bundle.prediction_rows
        ]
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

    def test_game_type_identity_keeps_r_legacy_and_separates_non_r(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        explicit_regular_rows = tuple(replace(row, game_type="R") for row in bundle.prediction_rows)
        postseason_rows = tuple(replace(row, game_type="P") for row in bundle.prediction_rows)
        regular_fp = canonical_prediction_fingerprint(
            source_identity=bundle.source_identity,
            prediction_rows=bundle.prediction_rows,
            exclusion_rows=bundle.exclusion_rows,
            source_manifest=bundle.source_manifest,
        )
        explicit_regular_fp = canonical_prediction_fingerprint(
            source_identity=bundle.source_identity,
            prediction_rows=explicit_regular_rows,
            exclusion_rows=bundle.exclusion_rows,
            source_manifest=bundle.source_manifest,
        )
        postseason_fp = canonical_prediction_fingerprint(
            source_identity=bundle.source_identity,
            prediction_rows=postseason_rows,
            exclusion_rows=bundle.exclusion_rows,
            source_manifest=bundle.source_manifest,
        )
        self.assertEqual(regular_fp, explicit_regular_fp)
        self.assertNotEqual(regular_fp, postseason_fp)

        regular_row_fp = compute_prediction_row_fingerprint(
            bundle.prediction_rows[0], official_date="2026-08-18"
        )
        explicit_r_row_fp = compute_prediction_row_fingerprint(
            explicit_regular_rows[0], official_date="2026-08-18"
        )
        postseason_row_fp = compute_prediction_row_fingerprint(
            postseason_rows[0], official_date="2026-08-18"
        )
        self.assertEqual(regular_row_fp, explicit_r_row_fp)
        self.assertNotEqual(regular_row_fp, postseason_row_fp)

        target_universe = [
            {
                "game_pk": row.game_pk,
                "game_number": row.game_number,
                "provider_game_id": row.provider_game_id,
                "p37_prediction_row_id": row.p37_prediction_row_id,
            }
            for row in bundle.prediction_rows
        ]
        regular_run_id = compute_deterministic_prediction_run_id(
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            normalized_input_fingerprint=regular_fp,
            prediction_bundle_fingerprint="same-bundle-content",
            target_universe=target_universe,
        )
        explicit_r_run_id = compute_deterministic_prediction_run_id(
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            normalized_input_fingerprint=explicit_regular_fp,
            prediction_bundle_fingerprint="same-bundle-content",
            target_universe=target_universe,
        )
        postseason_run_id = compute_deterministic_prediction_run_id(
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            normalized_input_fingerprint=postseason_fp,
            prediction_bundle_fingerprint="same-bundle-content",
            target_universe=target_universe,
        )
        self.assertEqual(regular_run_id, explicit_r_run_id)
        self.assertNotEqual(regular_run_id, postseason_run_id)

        postseason_bundle = NormalizedPregameInput(
            source_identity=bundle.source_identity,
            prediction_rows=postseason_rows,
            market_rows=(),
            exclusion_rows=bundle.exclusion_rows,
            source_manifest=bundle.source_manifest,
            authority_hashes={},
            p37_summary={},
            p39_summary={},
            p39_source_manifest={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "postseason.json"
            write_normalized_pregame_input(input_file, postseason_bundle)
            created = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=input_file,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=temp_path / "runs",
                created_at_utc="2026-08-18T12:00:00Z",
            )
            self.assertEqual(created.manifest["game_type"], "P")
            frozen = read_jsonl_objects(created.run_dir / "frozen_predictions.jsonl")
            self.assertTrue(all(row["game_identity"]["game_type"] == "P" for row in frozen))
            self.assertTrue(
                all(
                    row["model_identity"]["model_domain_status"]
                    == "POSTSEASON_OR_NON_R_DIAGNOSTIC"
                    for row in frozen
                )
            )

    def test_mixed_game_types_cannot_share_a_prediction_run(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        mixed = NormalizedPregameInput(
            source_identity=bundle.source_identity,
            prediction_rows=(bundle.prediction_rows[0], replace(bundle.prediction_rows[1], game_type="P")),
            market_rows=(),
            exclusion_rows=(),
            source_manifest=bundle.source_manifest,
            authority_hashes={},
            p37_summary={},
            p39_summary={},
            p39_source_manifest={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = write_normalized_pregame_input(Path(temp_dir) / "mixed.json", mixed)
            with self.assertRaisesRegex(RuntimeError, "P50C_MIXED_GAME_TYPE_RUN_REJECTED"):
                create_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    pregame_input=input_file,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                    run_root=Path(temp_dir) / "runs",
                    created_at_utc="2026-08-18T12:00:00Z",
                )

    def test_missing_game_type_cannot_create_a_fresh_run_but_can_read_exact_legacy_run(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        payload = bundle.to_payload()
        for prediction in payload["predictions"]:
            prediction.pop("game_type")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            legacy_input = temp_path / "legacy-missing-game-type.json"
            legacy_input.write_text(json.dumps(payload), encoding="utf-8")
            run_root = temp_path / "runs"

            with self.assertRaisesRegex(RuntimeError, "P50C_FRESH_GAME_TYPE_REQUIRED"):
                create_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    pregame_input=legacy_input,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                    run_root=run_root,
                    created_at_utc="2026-08-18T12:00:00Z",
                )
            self.assertFalse(run_root.exists())

            null_payload = bundle.to_payload()
            for prediction in null_payload["predictions"]:
                prediction["game_type"] = None
            null_input = temp_path / "null-game-type.json"
            null_input.write_text(json.dumps(null_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "P50C_INVALID_GAME_TYPE"):
                create_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    pregame_input=null_input,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                    run_root=run_root,
                    created_at_utc="2026-08-18T12:00:00Z",
                )
            self.assertFalse(run_root.exists())

            missing_object = replace(
                bundle,
                prediction_rows=tuple(
                    replace(prediction, game_type=None)
                    for prediction in bundle.prediction_rows
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "P50C_FRESH_GAME_TYPE_REQUIRED"):
                create_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    pregame_input=missing_object,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                    run_root=run_root,
                    created_at_utc="2026-08-18T12:00:00Z",
                )
            self.assertFalse(run_root.exists())

            explicit_input = temp_path / "explicit-regular-input.json"
            explicit_input.write_text(json.dumps(bundle.to_payload()), encoding="utf-8")
            created = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=explicit_input,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=run_root,
                created_at_utc="2026-08-18T12:00:00Z",
            )

            # Recreate the pre-contract shape inside task-owned temporary state.
            manifest_path = created.run_dir / "run_manifest.json"
            manifest = read_json_object(manifest_path)
            manifest.pop("game_type", None)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            def strip_game_type(value):
                if isinstance(value, dict):
                    return {
                        key: strip_game_type(item)
                        for key, item in value.items()
                        if key != "game_type"
                    }
                if isinstance(value, list):
                    return [strip_game_type(item) for item in value]
                return value

            predictions_path = created.run_dir / "frozen_predictions.jsonl"
            old_rows = [
                strip_game_type(row)
                for row in read_jsonl_objects(predictions_path)
            ]
            predictions_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in old_rows),
                encoding="utf-8",
            )

            recognized = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=legacy_input,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=run_root,
                created_at_utc="2026-08-18T12:00:00Z",
            )
            self.assertEqual(recognized.status, "RECOGNIZED_IDENTICAL")
            self.assertEqual(recognized.run_id, created.run_id)

    def test_camel_case_provider_game_type_is_preserved_and_conflicts_reject(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        camel_payload = bundle.to_payload()
        for prediction in camel_payload["predictions"]:
            prediction.pop("game_type")
            prediction["gameType"] = "P"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            camel_input = temp_path / "camel-game-type.json"
            camel_input.write_text(json.dumps(camel_payload), encoding="utf-8")
            postseason_root = temp_path / "postseason-runs"
            created = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=camel_input,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=postseason_root,
                created_at_utc="2026-08-18T12:00:00Z",
            )
            self.assertEqual(created.manifest["game_type"], "P")
            self.assertEqual(
                created.manifest["model_domain_status"],
                "POSTSEASON_OR_NON_R_DIAGNOSTIC",
            )
            frozen = read_jsonl_objects(created.run_dir / "frozen_predictions.jsonl")
            self.assertTrue(
                all(row["game_identity"]["game_type"] == "P" for row in frozen)
            )

            conflicting_payload = bundle.to_payload()
            for prediction in conflicting_payload["predictions"]:
                prediction["gameType"] = "P"
            conflicting_input = temp_path / "conflicting-game-type.json"
            conflicting_input.write_text(
                json.dumps(conflicting_payload), encoding="utf-8"
            )
            conflicting_root = temp_path / "conflicting-runs"
            with self.assertRaisesRegex(ValueError, "GAME_TYPE_CONFLICT"):
                create_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    pregame_input=conflicting_input,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                    run_root=conflicting_root,
                    created_at_utc="2026-08-18T12:00:00Z",
                )
            self.assertFalse(conflicting_root.exists())

            null_alias_payload = bundle.to_payload()
            for prediction in null_alias_payload["predictions"]:
                prediction.pop("game_type")
                prediction["gameType"] = None
            null_alias_input = temp_path / "null-camel-game-type.json"
            null_alias_input.write_text(
                json.dumps(null_alias_payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "P50C_INVALID_GAME_TYPE"):
                create_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    pregame_input=null_alias_input,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                    run_root=temp_path / "null-alias-runs",
                    created_at_utc="2026-08-18T12:00:00Z",
                )
            self.assertFalse((temp_path / "null-alias-runs").exists())

    def test_forward_summary_keeps_non_r_metrics_out_of_regular_headline(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        regular_rows = []
        postseason_rows = []
        for index, original in enumerate(bundle.prediction_rows):
            for game_type, target in (("R", regular_rows), ("P", postseason_rows)):
                row = replace(original, game_type=game_type)
                target_home = (game_type == "R" and index == 0) or (
                    game_type == "P" and index == 1
                )
                frozen = build_frozen_prediction_record(
                    row,
                    run_id=f"{game_type}-run",
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                    created_at_utc="2026-08-18T12:00:00Z",
                    official_date="2026-08-18",
                )
                result = NormalizedResultRecord(
                    prediction_row_id=row.p37_prediction_row_id,
                    provider_namespace=row.provider_namespace,
                    provider_game_id=row.provider_game_id,
                    game_number=row.game_number,
                    status="FINAL",
                    home_score=6 if target_home else 1,
                    away_score=2 if target_home else 7,
                    result_observed_at_utc="2026-08-18T23:00:00Z",
                    source_identity="FIXTURE_RESULTS",
                    game_type=game_type,
                )
                target.append(
                    build_p50c_prediction_ledger_record(
                        run_id=f"{game_type}-run",
                        run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                        frozen_prediction=frozen,
                        result_record=result,
                        evaluated_at_utc=result.result_observed_at_utc,
                    )
                )

        regular_only = compute_forward_prediction_summary(regular_rows)
        combined = compute_forward_prediction_summary(regular_rows + postseason_rows)
        for field in (
            "PREDICTION_FORWARD_SAMPLE_COUNT",
            "accuracy",
            "brier_score",
            "log_loss",
            "expected_calibration_error",
        ):
            self.assertEqual(combined[field], regular_only[field], field)
        self.assertEqual(combined["metric_game_type"], "R")
        self.assertEqual(combined["game_type_cohorts"]["R"]["sample_count"], 2)
        self.assertEqual(combined["game_type_cohorts"]["P"]["sample_count"], 2)
        self.assertNotEqual(
            combined["game_type_cohorts"]["R"]["accuracy"],
            combined["game_type_cohorts"]["P"]["accuracy"],
        )
        self.assertEqual(combined["total_prediction_forward_sample_count"], 4)

    def test_p37_result_identity_cannot_cross_game_type(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = write_normalized_pregame_input(
                temp_path / "pregame.json", bundle
            )
            created = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=input_file,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=temp_path / "runs",
                created_at_utc="2026-08-18T12:00:00Z",
            )
            wrong_type_result = replace(
                _make_result_record("a" * 64, "game_01", 6, 2),
                game_type="P",
            )
            result_file = write_normalized_result_input(
                temp_path / "wrong-type-results.jsonl", [wrong_type_result]
            )
            with self.assertRaisesRegex(RuntimeError, "P50C_GAME_TYPE_MISMATCH"):
                settle_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    run_dir=created.run_dir,
                    result_input=result_file,
                    ledger_root=temp_path / "ledger",
                    settled_at_utc="2026-08-18T23:30:00Z",
                )

    def test_in_memory_settlement_result_requires_game_type(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = write_normalized_pregame_input(
                temp_path / "pregame.json", bundle
            )
            created = create_p50c_prediction_run(
                REPOSITORY_ROOT,
                pregame_input=input_file,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=temp_path / "runs",
                created_at_utc="2026-08-18T12:00:00Z",
            )
            missing_type_result = replace(
                _make_result_record("a" * 64, "game_01", 6, 2),
                game_type=None,
            )
            with self.assertRaisesRegex(RuntimeError, "P50C_RESULT_GAME_TYPE_REQUIRED"):
                settle_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    run_dir=created.run_dir,
                    result_input=(missing_type_result,),
                    ledger_root=temp_path / "ledger",
                    settled_at_utc="2026-08-18T23:30:00Z",
                )

    def test_no_betting_or_odds_or_result_fields_in_frozen_record(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        row = bundle.prediction_rows[0]
        record = build_frozen_prediction_record(
            row,
            run_id="p50c_run_test",
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            created_at_utc="2026-08-18T12:00:00Z",
            official_date="2026-08-18",
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
            self.assertEqual(res1.manifest["target_date"], "2026-08-18")
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

    def test_fresh_prediction_requires_real_team_identity_before_writes(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        for invalid in (None, "HOME", "AWAY", ""):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temp_dir:
                manifest = json.loads(json.dumps(bundle.source_manifest))
                if invalid is None:
                    manifest.pop("team_identities_by_prediction_row_id")
                else:
                    for identity in manifest["team_identities_by_prediction_row_id"].values():
                        identity["home_team_code"] = invalid
                run_root = Path(temp_dir) / "runs"
                with self.assertRaisesRegex(ValueError, "P50C_FRESH_TEAM_IDENTITY_REQUIRED"):
                    create_p50c_prediction_run(
                        REPOSITORY_ROOT,
                        pregame_input=replace(bundle, source_manifest=manifest),
                        run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                        run_root=run_root,
                        created_at_utc="2026-08-18T12:00:00Z",
                    )
                self.assertFalse(run_root.exists())

    def test_normalized_prediction_only_input_preserves_supplied_team_codes(self) -> None:
        bundle = _make_prospective_pregame_fixture()
        payload = bundle.to_payload()
        teams = payload["source_manifest"].pop("team_identities_by_prediction_row_id")
        payload["source_manifest"].update({
            "p37a": {"p37_comparisons_sha256": "a" * 64},
            "p39a": {"legacy_source_sha256": "b" * 64},
        })
        for row in payload["predictions"]:
            row.update(teams[row["p37_prediction_row_id"]])
            row["game_type"] = "F"
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "pregame.json"
            source.write_text(json.dumps(payload))
            created = create_p50c_prediction_run(
                REPOSITORY_ROOT, pregame_input=source,
                run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                run_root=Path(temp_dir) / "runs", created_at_utc="2026-08-18T12:00:00Z",
            )
            for row in created.frozen_predictions:
                identity = teams[row["prediction_authority"]["p37_prediction_row_id"]]
                for field, value in identity.items():
                    self.assertEqual(row["game_identity"][field], value)

    def test_missing_provider_official_date_fails_closed(self) -> None:
        complete = _make_prospective_pregame_fixture()
        missing_date_bundle = NormalizedPregameInput(
            source_identity=complete.source_identity,
            prediction_rows=complete.prediction_rows,
            market_rows=(),
            exclusion_rows=(),
            source_manifest={"target_date": "2026-08-18"},
            authority_hashes={},
            p37_summary={},
            p39_summary={},
            p39_source_manifest={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "P50C_CANONICAL_OFFICIAL_DATE_REQUIRED"):
                create_p50c_prediction_run(
                    REPOSITORY_ROOT,
                    pregame_input=missing_date_bundle,
                    run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                    run_root=Path(temp_dir) / "runs",
                    created_at_utc="2026-08-18T12:00:00Z",
                )

    def test_official_date_overlay_is_validated_and_replay_idempotent(self) -> None:
        raw = {
            "ledger_record_id": "ledger_001",
            "run_id": "run_001",
            "prediction_identity": {
                "workflow_prediction_id": "prediction_001",
                "prediction_fingerprint": "fingerprint_001",
            },
            "game_identity": {
                "game_pk": 824154,
                "provider_game_id": "824154",
                "game_number": 1,
                "official_date": "2026-08-19",
            },
        }
        correction = {
            "schema_version": P66A_OFFICIAL_DATE_CORRECTION_SCHEMA,
            "ledger_id": "ledger_001",
            "run_id": "run_001",
            "prediction_id": "prediction_001",
            "official_game_pk": 824154,
            "original_official_date": "2026-08-19",
            "corrected_official_date": "2026-08-18",
        }

        first = apply_official_date_corrections([raw], [correction])
        second = apply_official_date_corrections([raw], [correction])
        self.assertEqual(first, second)
        self.assertEqual(raw["game_identity"]["official_date"], "2026-08-19")
        self.assertEqual(first[0]["game_identity"]["official_date"], "2026-08-18")
        self.assertEqual(len(first), 1)

        with self.assertRaisesRegex(ValueError, "P66A_DUPLICATE_CORRECTION_LEDGER_ID"):
            apply_official_date_corrections([raw], [correction, correction])
        with self.assertRaisesRegex(ValueError, "P66A_CORRECTION_GAME_PK_MISMATCH"):
            apply_official_date_corrections(
                [raw],
                [{**correction, "official_game_pk": 999999}],
            )

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

    def test_run_line_forward_evidence_in_shared_p50c_ledger(self) -> None:
        """P50C forward ledger supports Run Line records alongside Moneyline without conflict."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ledger_root = temp_path / "ledger"
            ledger_root.mkdir(parents=True, exist_ok=True)
            target_ledger = ledger_root / "prediction_forward_ledger.jsonl"

            # 1. Moneyline record
            ml_record = {
                "ledger_record_id": "ml_001",
                "schema_version": P50C_LEDGER_RECORD_SCHEMA,
                "run_id": "p50c_run_ml",
                "run_classification": CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                "market_id": "moneyline",
                "prediction_identity": {
                    "workflow_prediction_id": "p50c_pred_ml1",
                    "prediction_fingerprint": "ml_fp1",
                },
                "game_identity": {
                    "official_date": "2026-08-18",
                    "home_team": "TEX",
                    "away_team": "STL",
                },
                "model_identity": {
                    "model_role": "CHAMPION",
                    "model_id": "champion_ml",
                    "model_probability": "0.60",
                    "selection": "HOME",
                },
                "evaluation": {
                    "selection": "HOME",
                    "actual_winner": "HOME",
                    "is_correct": True,
                    "correctness_target": 1,
                    "brier_component": "0.160000",
                    "log_loss_component": "0.510825623766",
                },
            }

            # 2. Run Line record
            rl_record = {
                "ledger_record_id": "rl_001",
                "schema_version": P50C_LEDGER_RECORD_SCHEMA,
                "run_id": "p50c_run_rl",
                "run_classification": CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
                "market_id": "run_line",
                "line_value": -1.5,
                "spread_home": -1.5,
                "prediction_identity": {
                    "workflow_prediction_id": "p61b_pred_rl1",
                    "prediction_fingerprint": "rl_fp1",
                    "market_id": "run_line",
                    "line_value": -1.5,
                },
                "game_identity": {
                    "official_date": "2026-08-18",
                    "home_team": "TEX",
                    "away_team": "STL",
                },
                "model_identity": {
                    "model_role": "CHAMPION",
                    "model_id": "poisson_team_rate_model",
                    "model_version": "p226a.raw.v1",
                    "market_id": "run_line",
                    "spread_home": -1.5,
                    "line_value": -1.5,
                    "model_probability": "0.58",
                    "selection": "HOME",
                },
                "evaluation": {
                    "market_id": "run_line",
                    "spread_home": -1.5,
                    "line_value": -1.5,
                    "selection": "HOME",
                    "actual_result": "HOME",
                    "actual_winner": "HOME",
                    "is_push": False,
                    "is_correct": True,
                    "correctness_target": 1,
                    "brier_component": "0.176400",
                    "log_loss_component": "0.544727175436",
                },
            }

            target_ledger.write_text(
                json.dumps(ml_record, sort_keys=True) + "\n"
                + json.dumps(rl_record, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            # Cumulative summary over both markets
            fwd_rows = read_jsonl_objects(target_ledger)
            self.assertEqual(len(fwd_rows), 2)
            self.assertEqual(fwd_rows[0]["market_id"], "moneyline")
            self.assertEqual(fwd_rows[1]["market_id"], "run_line")
            self.assertEqual(fwd_rows[1]["line_value"], -1.5)

            summary = get_p50c_forward_summary(REPOSITORY_ROOT, ledger_root=ledger_root)
            self.assertEqual(summary["PREDICTION_FORWARD_SAMPLE_COUNT"], 2)
            self.assertEqual(summary["run_count"], 2)
            self.assertEqual(summary["correct_count"], 2)
            self.assertEqual(summary["accuracy"], "1")


if __name__ == "__main__":
    unittest.main()
