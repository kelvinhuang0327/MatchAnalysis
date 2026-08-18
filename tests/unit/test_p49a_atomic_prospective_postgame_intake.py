"""Focused unit and integration tests for P49A external final result admission and prospective postgame intake."""

from __future__ import annotations

from decimal import Decimal
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from match_analysis.application.use_cases.p44a_historical_source_adapter import (
    protected_authority_hashes,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    P44A_RESULT_INPUT_SCHEMA,
    load_normalized_result_input,
)
from match_analysis.application.use_cases.p45a_paper_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
    STATE_FROZEN,
    STATE_PARTIALLY_SETTLED,
    STATE_SETTLED,
    get_p45a_forward_summary,
    read_json_object,
    read_jsonl_objects,
)
from match_analysis.application.use_cases.p48a_atomic_prospective_pregame_intake import (
    intake_prospective_pregame_bundle,
)
from match_analysis.application.use_cases.p49a_external_final_result_admission import (
    P49A_ADMISSION_RECORD_SCHEMA,
    P49A_DEFAULT_SOURCE_IDENTITY,
    P49A_INTAKE_RECEIPT_SCHEMA,
    P49A_TASK_ID,
    REQUIRED_RESULT_BUNDLE_FILES,
    admit_external_final_result_bundle,
    intake_prospective_postgame_results,
)
from match_analysis.interfaces.cli.admit_external_final_result_bundle import (
    main as admit_cli_main,
)
from match_analysis.interfaces.cli.prospective_postgame_intake import (
    main as intake_cli_main,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sample_p35a_pregame_bundle() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Return minimal truthful P35A prospective pregame bundle."""
    analysis_rows: list[dict[str, object]] = [
        {
            "schema_version": "p30a.moneyline_paper_analysis.v1",
            "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
            "game_id": "824192",
            "scheduled_start": "2026-08-18T18:10:00Z",
            "home_team": "Houston Astros",
            "away_team": "Texas Rangers",
            "structural_status": "EDGE_AVAILABLE",
            "status": "EDGE_AVAILABLE",
            "prediction_id": "5de824e18c72b10c7d77ea843c4ee3a7787fbf2e789c86feac209206b16b39c5",
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": "0.5861848091915969598190982782",
            "market_price_id": "p28ab:6b6f72c2b12b80b8c84b434105976a0f57d5c6f13f26e87bf7f935a856779bf5",
            "price_observed_at": "2026-08-18T02:04:35.317592Z",
            "home_decimal_odds": "1.85",
            "away_decimal_odds": "1.65",
            "home_no_vig_probability": "0.4714285714285714285714285712",
            "away_no_vig_probability": "0.5285714285714285714285714284",
            "home_edge": "0.1147562377630255312476697070",
            "away_edge": "-0.1147562377630255312476697066",
            "controlled_unavailable_reason": None,
        },
        {
            "schema_version": "p30a.moneyline_paper_analysis.v1",
            "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
            "game_id": "824114",
            "scheduled_start": "2026-08-18T22:40:00Z",
            "home_team": "San Diego Padres",
            "away_team": "San Francisco Giants",
            "structural_status": "EDGE_AVAILABLE",
            "status": "EDGE_AVAILABLE",
            "prediction_id": "9366e6b4793f0b2f56193ff790089aa14bc8b859942a1762c938efc63c241517",
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": "0.4856417724250269389938810243",
            "market_price_id": "p28ab:7d1b327fbbfec711a91e5e01dfb3b24f5a3c2cfaf50d75a80f0c05763cb338da",
            "price_observed_at": "2026-08-18T02:04:35.317592Z",
            "home_decimal_odds": "1.74",
            "away_decimal_odds": "1.80",
            "home_no_vig_probability": "0.5084745762711864406779661017",
            "away_no_vig_probability": "0.4915254237288135593220338983",
            "home_edge": "-0.0228328038461595016840850774",
            "away_edge": "0.0228328038461595016840850774",
            "controlled_unavailable_reason": None,
        },
    ]

    schedule_rows: list[dict[str, object]] = [
        {
            "schema_version": "mlb.schedule_game.v1",
            "provider_game_id": "824192",
            "game_pk": 824192,
            "game_number": 1,
            "official_date": "2026-08-18",
            "scheduled_start_utc": "2026-08-18T18:10:00Z",
            "provider_namespace": "MLB_STATS_API",
            "home_team": "Houston Astros",
            "away_team": "Texas Rangers",
            "home_team_code": "HOU",
            "away_team_code": "TEX",
        },
        {
            "schema_version": "mlb.schedule_game.v1",
            "provider_game_id": "824114",
            "game_pk": 824114,
            "game_number": 1,
            "official_date": "2026-08-18",
            "scheduled_start_utc": "2026-08-18T22:40:00Z",
            "provider_namespace": "MLB_STATS_API",
            "home_team": "San Diego Padres",
            "away_team": "San Francisco Giants",
            "home_team_code": "SD",
            "away_team_code": "SF",
        },
    ]

    source_manifest: dict[str, object] = {
        "schema_version": "p35a.source_manifest.v1",
        "fetch_timestamp_utc": "2026-08-18T02:04:35Z",
        "target_date": "2026-08-18",
        "official_schedule_source": "MLB_STATS_API",
    }

    run_manifest: dict[str, object] = {
        "schema_version": "p35a.run_manifest.v1",
        "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
        "target_date": "2026-08-18",
        "total_games": 2,
        "productive_count": 2,
        "excluded_count": 0,
    }

    return analysis_rows, schedule_rows, source_manifest, run_manifest


def _sample_final_result_bundle() -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Return minimal authoritative external final result bundle."""
    results = [
        {
            "provider_namespace": "MLB_STATS_API",
            "provider_game_id": "824192",
            "game_number": 1,
            "status": "FINAL",
            "home_score": 6,
            "away_score": 3,
            "result_observed_at_utc": "2026-08-18T21:30:00Z",
            "source_identity": "MLB_STATS_API_OFFICIAL_RESULTS",
        },
        {
            "provider_namespace": "MLB_STATS_API",
            "provider_game_id": "824114",
            "game_number": 1,
            "status": "FINAL",
            "home_score": 2,
            "away_score": 5,
            "result_observed_at_utc": "2026-08-19T02:15:00Z",
            "source_identity": "MLB_STATS_API_OFFICIAL_RESULTS",
        },
    ]

    source_manifest = {
        "schema_version": "p49a.source_manifest.v1",
        "fetch_timestamp_utc": "2026-08-19T03:00:00Z",
        "target_date": "2026-08-18",
        "official_source": "MLB_STATS_API",
    }

    result_manifest = {
        "schema_version": "p49a.result_manifest.v1",
        "target_date": "2026-08-18",
        "total_results": 2,
        "final_count": 2,
    }

    return results, source_manifest, result_manifest


def _write_pregame_bundle_to_dir(
    target_dir: Path,
    analysis_rows: list[dict[str, object]],
    schedule_rows: list[dict[str, object]],
    source_manifest: dict[str, object],
    run_manifest: dict[str, object],
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "analysis.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in analysis_rows),
        encoding="utf-8",
    )
    (target_dir / "mlb_source_snapshot.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in schedule_rows),
        encoding="utf-8",
    )
    (target_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_dir


def _write_result_bundle_to_dir(
    target_dir: Path,
    results: list[dict[str, object]],
    source_manifest: dict[str, object],
    result_manifest: dict[str, object],
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "final_results.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in results),
        encoding="utf-8",
    )
    (target_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_dir / "result_manifest.json").write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_dir


class P49AAtomicProspectivePostgameIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.authority_hashes_before = protected_authority_hashes(REPOSITORY_ROOT)

    def tearDown(self) -> None:
        authority_hashes_after = protected_authority_hashes(REPOSITORY_ROOT)
        self.assertEqual(
            self.authority_hashes_before,
            authority_hashes_after,
            "Protected historical authority hashes drifted during test execution",
        )

    def test_admit_valid_external_final_result_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle_dir = temp_path / "external_results_20260818"
            results, sm, rm = _sample_final_result_bundle()
            _write_result_bundle_to_dir(bundle_dir, results, sm, rm)

            admission_root = temp_path / "admissions"
            admitted = admit_external_final_result_bundle(
                bundle_dir,
                admission_root=admission_root,
                source_identity=P49A_DEFAULT_SOURCE_IDENTITY,
                admitted_at_utc="2026-08-19T03:00:00Z",
            )

            self.assertEqual(admitted.status, "ADMITTED")
            self.assertTrue(admitted.admitted_bundle_id.startswith("p49a_bundle_"))
            self.assertEqual(admitted.target_date, "2026-08-18")
            self.assertEqual(admitted.final_result_count, 2)
            self.assertTrue(admitted.imported_bundle_dir.is_dir())
            self.assertTrue((admitted.imported_bundle_dir / "raw_bundle" / "final_results.jsonl").is_file())
            self.assertTrue(admitted.admission_record_path.is_file())
            self.assertTrue(admitted.normalized_result_path.is_file())

            # Verify normalized result contents
            records = load_normalized_result_input(admitted.normalized_result_path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].status, "FINAL")
            self.assertEqual(records[0].home_score, 6)
            self.assertEqual(records[0].away_score, 3)
            self.assertEqual(records[0].actual_winner, "HOME")

    def test_admit_bundle_stable_two_read_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle_dir = temp_path / "external_results_20260818"
            results, sm, rm = _sample_final_result_bundle()
            _write_result_bundle_to_dir(bundle_dir, results, sm, rm)

            # Mutate between reads using mock or hook
            with patch(
                "match_analysis.application.use_cases.p49a_external_final_result_admission._snapshot_file_bytes",
                side_effect=[
                    {"final_results.jsonl": b"content_1", "source_manifest.json": b"{}", "result_manifest.json": b'{"target_date":"2026-08-18"}'},
                    {"final_results.jsonl": b"content_2_changed", "source_manifest.json": b"{}", "result_manifest.json": b'{"target_date":"2026-08-18"}'},
                ],
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    admit_external_final_result_bundle(
                        bundle_dir,
                        admission_root=temp_path / "admissions",
                    )
                self.assertIn("P49A_EXTERNAL_BUNDLE_CHANGED_DURING_ADMISSION", str(ctx.exception))

    def test_admit_bundle_non_final_status_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle_dir = temp_path / "external_results_20260818"
            results, sm, rm = _sample_final_result_bundle()
            results[0]["status"] = "IN_PROGRESS"
            _write_result_bundle_to_dir(bundle_dir, results, sm, rm)

            with self.assertRaises(RuntimeError) as ctx:
                admit_external_final_result_bundle(
                    bundle_dir,
                    admission_root=temp_path / "admissions",
                )
            self.assertIn("P43A_NON_FINAL_RESULT_FAIL_CLOSED", str(ctx.exception))

    def test_admit_bundle_tied_score_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle_dir = temp_path / "external_results_20260818"
            results, sm, rm = _sample_final_result_bundle()
            results[0]["home_score"] = 4
            results[0]["away_score"] = 4
            _write_result_bundle_to_dir(bundle_dir, results, sm, rm)

            with self.assertRaises(RuntimeError) as ctx:
                admit_external_final_result_bundle(
                    bundle_dir,
                    admission_root=temp_path / "admissions",
                )
            self.assertIn("P43A_NON_FINAL_RESULT_FAIL_CLOSED", str(ctx.exception))

    def test_admit_bundle_idempotency_and_conflict_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundle_dir = temp_path / "external_results_20260818"
            results, sm, rm = _sample_final_result_bundle()
            _write_result_bundle_to_dir(bundle_dir, results, sm, rm)
            admission_root = temp_path / "admissions"

            admitted_1 = admit_external_final_result_bundle(
                bundle_dir,
                admission_root=admission_root,
            )
            self.assertEqual(admitted_1.status, "ADMITTED")

            admitted_2 = admit_external_final_result_bundle(
                bundle_dir,
                admission_root=admission_root,
            )
            self.assertEqual(admitted_2.status, "RECOGNIZED_IDENTICAL")
            self.assertEqual(admitted_1.bundle_fingerprint, admitted_2.bundle_fingerprint)

    def test_end_to_end_pregame_to_postgame_prospective_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo_root = temp_path / "repo"
            repo_root.mkdir()

            # 1. Create external pregame bundle
            pregame_dir = temp_path / "pregame_bundle_20260818"
            a_rows, s_rows, sm_pre, rm_pre = _sample_p35a_pregame_bundle()
            _write_pregame_bundle_to_dir(pregame_dir, a_rows, s_rows, sm_pre, rm_pre)

            # 2. Run P48 prospective pregame intake -> frozen run
            pregame_intake = intake_prospective_pregame_bundle(
                pregame_dir,
                repository_root=repo_root,
                admission_root=temp_path / "pregame_admissions",
                run_root=temp_path / "runs",
                intake_timestamp_utc="2026-08-18T12:00:00Z",
            )
            self.assertEqual(pregame_intake.status, "CREATED")
            self.assertEqual(pregame_intake.run_classification, CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER)
            self.assertEqual(pregame_intake.lifecycle_state, STATE_FROZEN)
            self.assertEqual(pregame_intake.bet_count, 1)
            self.assertEqual(pregame_intake.pass_count, 1)

            # 3. Create external final result bundle
            results_dir = temp_path / "result_bundle_20260818"
            results, sm_post, rm_post = _sample_final_result_bundle()
            _write_result_bundle_to_dir(results_dir, results, sm_post, rm_post)

            # 4. Run P49 prospective postgame settlement intake
            postgame_intake = intake_prospective_postgame_results(
                results_dir,
                paper_run_dir=pregame_intake.run_dir,
                repository_root=repo_root,
                admission_root=temp_path / "result_admissions",
                ledger_root=temp_path / "ledger",
                settled_at_utc="2026-08-19T03:00:00Z",
            )

            self.assertEqual(postgame_intake.status, STATE_SETTLED)
            self.assertEqual(postgame_intake.lifecycle_state, STATE_SETTLED)
            self.assertEqual(postgame_intake.newly_settled_count, 2)
            self.assertEqual(postgame_intake.total_settled_count, 2)
            self.assertEqual(postgame_intake.pending_count, 0)
            self.assertEqual(postgame_intake.settled_bet_count, 1)
            self.assertEqual(postgame_intake.settled_pass_count, 1)
            self.assertEqual(postgame_intake.win_count, 1)
            self.assertEqual(postgame_intake.loss_count, 0)
            self.assertEqual(postgame_intake.net_paper_units, "0.85")

            # 5. Verify forward paper ledger and cumulative forward summary
            forward_ledger_path = temp_path / "ledger" / "forward_paper_ledger.jsonl"
            self.assertTrue(forward_ledger_path.is_file())
            ledger_rows = read_jsonl_objects(forward_ledger_path)
            self.assertEqual(len(ledger_rows), 2)

            forward_summary_path = temp_path / "ledger" / "forward_summary.json"
            self.assertTrue(forward_summary_path.is_file())
            summary = read_json_object(forward_summary_path)
            self.assertEqual(summary["forward_sample_count"], 2)
            self.assertEqual(summary["settled_bet_count"], 1)
            self.assertEqual(summary["wins"], 1)
            self.assertEqual(summary["losses"], 0)
            self.assertEqual(summary["net_paper_units"], "0.85")
            self.assertEqual(summary["descriptive_roi"], "0.85")

            # 6. Verify postgame receipt in run dir
            receipt_file = pregame_intake.run_dir / "postgame_settle_receipt.json"
            self.assertTrue(receipt_file.is_file())
            receipt = read_json_object(receipt_file)
            self.assertEqual(receipt["schema_version"], P49A_INTAKE_RECEIPT_SCHEMA)
            self.assertEqual(receipt["task_id"], P49A_TASK_ID)
            self.assertEqual(receipt["lifecycle_state"], STATE_SETTLED)

    def test_partial_settlement_then_complete_settlement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo_root = temp_path / "repo"
            repo_root.mkdir()

            # Pregame intake with 2 games
            pregame_dir = temp_path / "pregame_bundle_20260818"
            a_rows, s_rows, sm_pre, rm_pre = _sample_p35a_pregame_bundle()
            _write_pregame_bundle_to_dir(pregame_dir, a_rows, s_rows, sm_pre, rm_pre)

            pregame_intake = intake_prospective_pregame_bundle(
                pregame_dir,
                repository_root=repo_root,
                admission_root=temp_path / "pregame_admissions",
                run_root=temp_path / "runs",
                intake_timestamp_utc="2026-08-18T12:00:00Z",
            )

            # Pass 1: Only 1 game has finished
            results_dir_1 = temp_path / "partial_result_bundle_1"
            results, sm_post, rm_post = _sample_final_result_bundle()
            _write_result_bundle_to_dir(results_dir_1, [results[0]], sm_post, {"target_date": "2026-08-18", "total_results": 1})

            postgame_1 = intake_prospective_postgame_results(
                results_dir_1,
                paper_run_dir=pregame_intake.run_dir,
                repository_root=repo_root,
                admission_root=temp_path / "result_admissions",
                ledger_root=temp_path / "ledger",
                settled_at_utc="2026-08-18T22:00:00Z",
            )
            self.assertEqual(postgame_1.lifecycle_state, STATE_PARTIALLY_SETTLED)
            self.assertEqual(postgame_1.newly_settled_count, 1)
            self.assertEqual(postgame_1.total_settled_count, 1)
            self.assertEqual(postgame_1.pending_count, 1)

            # Pass 2: 2nd game completes
            results_dir_2 = temp_path / "partial_result_bundle_2"
            _write_result_bundle_to_dir(results_dir_2, [results[1]], sm_post, {"target_date": "2026-08-18", "total_results": 1})

            postgame_2 = intake_prospective_postgame_results(
                results_dir_2,
                paper_run_dir=pregame_intake.run_dir,
                repository_root=repo_root,
                admission_root=temp_path / "result_admissions",
                ledger_root=temp_path / "ledger",
                settled_at_utc="2026-08-19T03:00:00Z",
            )
            self.assertEqual(postgame_2.lifecycle_state, STATE_SETTLED)
            self.assertEqual(postgame_2.newly_settled_count, 1)
            self.assertEqual(postgame_2.total_settled_count, 2)
            self.assertEqual(postgame_2.pending_count, 0)

    def test_validate_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo_root = temp_path / "repo"
            repo_root.mkdir()

            pregame_dir = temp_path / "pregame_bundle_20260818"
            a_rows, s_rows, sm_pre, rm_pre = _sample_p35a_pregame_bundle()
            _write_pregame_bundle_to_dir(pregame_dir, a_rows, s_rows, sm_pre, rm_pre)

            pregame_intake = intake_prospective_pregame_bundle(
                pregame_dir,
                repository_root=repo_root,
                admission_root=temp_path / "pregame_admissions",
                run_root=temp_path / "runs",
                intake_timestamp_utc="2026-08-18T12:00:00Z",
            )

            results_dir = temp_path / "result_bundle_20260818"
            results, sm_post, rm_post = _sample_final_result_bundle()
            _write_result_bundle_to_dir(results_dir, results, sm_post, rm_post)

            postgame_val = intake_prospective_postgame_results(
                results_dir,
                paper_run_dir=pregame_intake.run_dir,
                repository_root=repo_root,
                admission_root=temp_path / "result_admissions",
                ledger_root=temp_path / "ledger",
                validate_only=True,
            )
            self.assertEqual(postgame_val.status, "VALIDATED")
            self.assertEqual(postgame_val.lifecycle_state, STATE_FROZEN)

            # Confirm run manifest in run_dir was not altered to settled
            manifest = read_json_object(pregame_intake.run_dir / "run_manifest.json")
            self.assertEqual(manifest["lifecycle_state"], STATE_FROZEN)

    def test_cli_postgame_intake_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            repo_root = temp_path / "repo"
            repo_root.mkdir()

            pregame_dir = temp_path / "pregame_bundle_20260818"
            a_rows, s_rows, sm_pre, rm_pre = _sample_p35a_pregame_bundle()
            _write_pregame_bundle_to_dir(pregame_dir, a_rows, s_rows, sm_pre, rm_pre)

            pregame_intake = intake_prospective_pregame_bundle(
                pregame_dir,
                repository_root=repo_root,
                admission_root=temp_path / "pregame_admissions",
                run_root=temp_path / "runs",
                intake_timestamp_utc="2026-08-18T12:00:00Z",
            )

            results_dir = temp_path / "result_bundle_20260818"
            results, sm_post, rm_post = _sample_final_result_bundle()
            _write_result_bundle_to_dir(results_dir, results, sm_post, rm_post)

            # Validate via CLI
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                code = intake_cli_main([
                    "validate",
                    "--bundle", str(results_dir),
                    "--run-dir", str(pregame_intake.run_dir),
                    "--repository-root", str(repo_root),
                    "--admission-root", str(temp_path / "result_admissions"),
                ])
                self.assertEqual(code, 0)
                output = mock_stdout.getvalue()
                self.assertIn("p49a-settle=status=VALIDATED", output)

            # Execute via CLI
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                code = intake_cli_main([
                    "postgame-intake",
                    "--bundle", str(results_dir),
                    "--run-dir", str(pregame_intake.run_dir),
                    "--repository-root", str(repo_root),
                    "--admission-root", str(temp_path / "result_admissions"),
                    "--ledger-root", str(temp_path / "ledger"),
                ])
                self.assertEqual(code, 0)
                output = mock_stdout.getvalue()
                self.assertIn("p49a-settle=status=SETTLED", output)
                self.assertIn("forward_samples=2", output)

    def test_cli_admit_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_dir = temp_path / "result_bundle_20260818"
            results, sm_post, rm_post = _sample_final_result_bundle()
            _write_result_bundle_to_dir(results_dir, results, sm_post, rm_post)

            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                code = admit_cli_main([
                    "--bundle", str(results_dir),
                    "--admission-root", str(temp_path / "result_admissions"),
                ])
                self.assertEqual(code, 0)
                output = mock_stdout.getvalue()
                self.assertIn("p49a-admit=status=ADMITTED", output)
                self.assertIn("results=2", output)


if __name__ == "__main__":
    unittest.main()
