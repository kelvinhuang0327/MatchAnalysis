"""Focused unit and integration tests for P48A atomic prospective pregame intake."""

from __future__ import annotations

import ast
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest

from match_analysis.application.use_cases.p44a_historical_source_adapter import (
    protected_authority_hashes,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    FORBIDDEN_PREGAME_FIELD_NAMES,
    load_normalized_pregame_input,
)
from match_analysis.application.use_cases.p45a_paper_run_ledger import (
    CLASSIFICATION_HISTORICAL_REHEARSAL,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
    STATE_FROZEN,
    get_p45a_forward_summary,
    read_json_object,
    read_jsonl_objects,
)
from match_analysis.application.use_cases.p47a_external_bundle_admission import (
    P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
    P47A_EXTERNAL_ADMISSION_SOURCE_IDENTITY,
    REQUIRED_BUNDLE_FILES,
)
from match_analysis.application.use_cases.p48a_atomic_prospective_pregame_intake import (
    P48A_DEFAULT_SOURCE_IDENTITY,
    P48A_INTAKE_RECEIPT_SCHEMA,
    P48A_TASK_ID,
    intake_prospective_pregame_bundle,
)
from match_analysis.interfaces.cli.prospective_pregame_intake import (
    main as cli_main,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
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
    REPOSITORY_ROOT / "report/p45a_prospective_paper_run_ledger/summary.json",
    REPOSITORY_ROOT / "report/p45a_prospective_paper_run_ledger/report.md",
)


def _sample_p35a_bundle() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Return minimal truthful P35A prospective contract test fixture."""

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
            "scheduled_start": "2026-08-18T23:10:00Z",
            "home_team": "Kansas City Royals",
            "away_team": "Boston Red Sox",
            "structural_status": "EDGE_AVAILABLE",
            "status": "EDGE_AVAILABLE",
            "prediction_id": "55e5ec556f198337f0f9a2ad090b052f3910c70b7c57d8fd829c0b3a94caa337",
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": "0.5303627085439037814014483218",
            "market_price_id": "p28ab:fff29ab049b29e9c47ca097af08b1df1048a85933d4719606a54d4b847da68c2",
            "price_observed_at": "2026-08-18T01:21:04.535534Z",
            "home_decimal_odds": "1.82",
            "away_decimal_odds": "1.68",
            "home_no_vig_probability": "0.4799999999999999999999999999",
            "away_no_vig_probability": "0.5199999999999999999999999998",
            "home_edge": "0.0503627085439037814014483219",
            "away_edge": "-0.0503627085439037814014483216",
            "controlled_unavailable_reason": None,
        },
        {
            "schema_version": "p30a.moneyline_paper_analysis.v1",
            "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
            "game_id": "822738",
            "scheduled_start": "2026-08-18T17:35:00Z",
            "home_team": "Washington Nationals",
            "away_team": "Baltimore Orioles",
            "structural_status": "FEATURE_UNAVAILABLE",
            "status": "FEATURE_UNAVAILABLE",
            "prediction_id": None,
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": None,
            "market_price_id": "p28ab:28ab9f198d81aad8d36b7d4f38b3a175962d6ec9f4af50904475accd2e298bb5",
            "price_observed_at": "2026-08-18T02:04:35.317592Z",
            "home_decimal_odds": "1.90",
            "away_decimal_odds": "1.60",
            "home_no_vig_probability": "0.4571428571428571428571428570",
            "away_no_vig_probability": "0.5428571428571428571428571426",
            "home_edge": None,
            "away_edge": None,
            "controlled_unavailable_reason": "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY",
        },
    ]

    schedule_rows: list[dict[str, object]] = [
        {
            "schema_version": "p23f2.mlb_official_normalized.v1",
            "provider_game_id": "824192",
            "game_pk": 824192,
            "game_number": 1,
            "official_date": "2026-08-18",
            "scheduled_start_utc": "2026-08-18T18:10:00Z",
            "status": "Scheduled",
            "final": False,
            "home_team": {"id": 117, "abbreviation": "HOU", "name": "Houston Astros"},
            "away_team": {"id": 140, "abbreviation": "TEX", "name": "Texas Rangers"},
            "home_score": None,
            "away_score": None,
        },
        {
            "schema_version": "p23f2.mlb_official_normalized.v1",
            "provider_game_id": "824114",
            "game_pk": 824114,
            "game_number": 1,
            "official_date": "2026-08-18",
            "scheduled_start_utc": "2026-08-18T23:10:00Z",
            "status": "Scheduled",
            "final": False,
            "home_team": {"id": 118, "abbreviation": "KC", "name": "Kansas City Royals"},
            "away_team": {"id": 111, "abbreviation": "BOS", "name": "Boston Red Sox"},
            "home_score": None,
            "away_score": None,
        },
        {
            "schema_version": "p23f2.mlb_official_normalized.v1",
            "provider_game_id": "822738",
            "game_pk": 822738,
            "game_number": 1,
            "official_date": "2026-08-18",
            "scheduled_start_utc": "2026-08-18T17:35:00Z",
            "status": "Scheduled",
            "final": False,
            "home_team": {"id": 120, "abbreviation": "WSH", "name": "Washington Nationals"},
            "away_team": {"id": 110, "abbreviation": "BAL", "name": "Baltimore Orioles"},
            "home_score": None,
            "away_score": None,
        },
    ]

    source_manifest: dict[str, object] = {
        "schema_version": "p30a.source_manifest.v1",
        "tsl_authority": {
            "authority_label": "TSL_BLOB3RD",
            "legacy_repository": "Betting-pool",
        },
    }

    run_manifest: dict[str, object] = {
        "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
        "target_date": "2026-08-18",
    }

    return analysis_rows, schedule_rows, source_manifest, run_manifest


def _write_bundle(
    bundle_dir: Path,
    analysis_rows: list[dict[str, object]],
    schedule_rows: list[dict[str, object]],
    source_manifest: dict[str, object],
    run_manifest: dict[str, object],
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "analysis.jsonl").write_text(
        "\n".join(json.dumps(r) for r in analysis_rows) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "mlb_source_snapshot.jsonl").write_text(
        "\n".join(json.dumps(r) for r in schedule_rows) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )


class P48AAtomicProspectivePregameIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_hashes = protected_authority_hashes(REPOSITORY_ROOT)

    def tearDown(self) -> None:
        current_hashes = protected_authority_hashes(REPOSITORY_ROOT)
        self.assertEqual(
            self.initial_hashes,
            current_hashes,
            "Protected repository authorities were modified during test execution",
        )

    def test_positive_atomic_prospective_intake_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            admission_root = tmp / "admitted"
            run_root = tmp / "runs"

            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            # Earliest scheduled game is 2026-08-18T17:35:00Z.
            # Intake occurs strictly before at 2026-08-18T12:00:00Z.
            result = intake_prospective_pregame_bundle(
                bundle_dir,
                repository_root=REPOSITORY_ROOT,
                admission_root=admission_root,
                run_root=run_root,
                source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                intake_timestamp_utc="2026-08-18T12:00:00Z",
            )

            # 1. Status & classifications
            self.assertEqual(result.status, "CREATED")
            self.assertEqual(result.run_classification, CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER)
            self.assertEqual(result.lifecycle_state, STATE_FROZEN)
            self.assertTrue(result.paper_run_id.startswith("p45a_run_"))
            self.assertEqual(result.target_universe_count, 3)
            self.assertEqual(result.eligible_decision_count, 2)
            self.assertEqual(result.bet_count, 1)
            self.assertEqual(result.pass_count, 1)
            self.assertEqual(result.exclusion_count, 1)

            # 2. Files written
            self.assertTrue(result.admission_record_path.is_file())
            self.assertTrue(result.normalized_pregame_path.is_file())
            self.assertIsNotNone(result.run_dir)
            run_dir = result.run_dir
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertTrue((run_dir / "pregame_decisions.jsonl").is_file())
            self.assertTrue((run_dir / "exclusions.jsonl").is_file())
            self.assertTrue((run_dir / "intake_receipt.json").is_file())

            # 3. Verify intake receipt
            receipt = read_json_object(run_dir / "intake_receipt.json")
            self.assertEqual(receipt["schema_version"], P48A_INTAKE_RECEIPT_SCHEMA)
            self.assertEqual(receipt["task_id"], P48A_TASK_ID)
            self.assertEqual(receipt["intake_status"], "CREATED")
            self.assertEqual(receipt["run_classification"], CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER)
            self.assertEqual(receipt["paper_run_id"], result.paper_run_id)
            self.assertEqual(receipt["admitted_bundle_id"], result.admitted_bundle_id)
            self.assertEqual(receipt["target_date"], "2026-08-18")

            # 4. Zero outcome contamination
            for key in FORBIDDEN_PREGAME_FIELD_NAMES:
                self.assertNotIn(key, receipt)
            run_manifest_content = read_json_object(run_dir / "run_manifest.json")
            self.assertEqual(run_manifest_content["settled_bet_count"], 0)
            self.assertEqual(run_manifest_content["settled_total_count"], 0)

            # 5. Idempotent repeated intake
            retry_result = intake_prospective_pregame_bundle(
                bundle_dir,
                repository_root=REPOSITORY_ROOT,
                admission_root=admission_root,
                run_root=run_root,
                source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                intake_timestamp_utc="2026-08-18T12:00:00Z",
            )
            self.assertEqual(retry_result.status, "RECOGNIZED_IDENTICAL")
            self.assertEqual(retry_result.paper_run_id, result.paper_run_id)
            self.assertEqual(retry_result.bundle_fingerprint, result.bundle_fingerprint)
            self.assertEqual(retry_result.decision_bundle_fingerprint, result.decision_bundle_fingerprint)

    def test_validation_only_mode_does_not_create_run_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            admission_root = tmp / "admitted"
            run_root = tmp / "runs"

            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            result = intake_prospective_pregame_bundle(
                bundle_dir,
                repository_root=REPOSITORY_ROOT,
                admission_root=admission_root,
                run_root=run_root,
                source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                intake_timestamp_utc="2026-08-18T12:00:00Z",
                validate_only=True,
            )

            self.assertEqual(result.status, "VALIDATED")
            self.assertEqual(result.run_classification, CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER)
            self.assertIsNone(result.paper_run_id)
            self.assertIsNone(result.run_dir)
            self.assertFalse(run_root.exists(), "Validation mode must not create run_root or run files")

    def test_after_scheduled_start_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            # Earliest eligible prediction scheduled start is 2026-08-18T18:10:00Z.
            # Attempt intake at 2026-08-18T18:10:01Z (after start).
            with self.assertRaisesRegex(
                RuntimeError,
                "P45A_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID: prospective run creation time.*is not strictly before earliest scheduled start",
            ):
                intake_prospective_pregame_bundle(
                    bundle_dir,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=tmp / "admitted",
                    run_root=tmp / "runs",
                    source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T18:10:01Z",
                )

    def test_historical_rehearsal_identity_fails_closed_no_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            # Supplying historical rehearsal source identity must fail closed, NOT downgrade
            with self.assertRaisesRegex(
                RuntimeError,
                "P45A_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID: historical source identity",
            ):
                intake_prospective_pregame_bundle(
                    bundle_dir,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=tmp / "admitted",
                    run_root=tmp / "runs",
                    source_identity=P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T12:00:00Z",
                )

    def test_post_start_market_observation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            # Set price_observed_at after scheduled_start
            analysis[0]["price_observed_at"] = "2026-08-18T18:15:00Z"
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            with self.assertRaisesRegex(
                ValueError,
                "market observation time.*is not strictly pregame",
            ):
                intake_prospective_pregame_bundle(
                    bundle_dir,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=tmp / "admitted",
                    run_root=tmp / "runs",
                    source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T12:00:00Z",
                )

    def test_outcome_contamination_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            analysis[0]["final_score"] = "5-3"
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            with self.assertRaisesRegex(
                ValueError,
                "P44A_PREGAME_OUTCOME_FIELDS_REJECTED",
            ):
                intake_prospective_pregame_bundle(
                    bundle_dir,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=tmp / "admitted",
                    run_root=tmp / "runs",
                    source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T12:00:00Z",
                )

    def test_missing_required_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)
            (bundle_dir / "analysis.jsonl").unlink()

            with self.assertRaises(FileNotFoundError):
                intake_prospective_pregame_bundle(
                    bundle_dir,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=tmp / "admitted",
                    run_root=tmp / "runs",
                    source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T12:00:00Z",
                )

    def test_conflicting_bundle_identity_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir_1 = tmp / "bundle_1"
            bundle_dir_2 = tmp / "bundle_2"
            admission_root = tmp / "admitted"
            run_root = tmp / "runs"

            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            _write_bundle(bundle_dir_1, analysis, schedule, src_manifest, run_manifest)

            # First intake
            res1 = intake_prospective_pregame_bundle(
                bundle_dir_1,
                repository_root=REPOSITORY_ROOT,
                admission_root=admission_root,
                run_root=run_root,
                source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                intake_timestamp_utc="2026-08-18T12:00:00Z",
            )
            self.assertEqual(res1.status, "CREATED")

            # Corrupt existing admission record with different fingerprint
            record_path = res1.admission_record_path
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["bundle_fingerprint"] = "corrupted_hash"
            record_path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "P47A_BUNDLE_AUTHORITY_CONFLICT"):
                intake_prospective_pregame_bundle(
                    bundle_dir_1,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=admission_root,
                    run_root=run_root,
                    source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T12:00:00Z",
                )

    def test_cli_prospective_intake_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            admission_root = tmp / "admitted"
            run_root = tmp / "runs"

            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            # 1. CLI validate subcommand
            validate_code = cli_main(
                [
                    "validate",
                    "--bundle",
                    str(bundle_dir),
                    "--admission-root",
                    str(admission_root),
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--intake-timestamp-utc",
                    "2026-08-18T12:00:00Z",
                ]
            )
            self.assertEqual(validate_code, 0)
            self.assertFalse(run_root.exists())

            # 2. CLI prospective-intake subcommand
            intake_code = cli_main(
                [
                    "prospective-intake",
                    "--bundle",
                    str(bundle_dir),
                    "--admission-root",
                    str(admission_root),
                    "--run-root",
                    str(run_root),
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--intake-timestamp-utc",
                    "2026-08-18T12:00:00Z",
                ]
            )
            self.assertEqual(intake_code, 0)
            self.assertTrue(run_root.exists())

    def test_explicit_network_denial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            # Block network sockets
            original_socket = socket.socket

            def _network_blocked(*args: object, **kwargs: object) -> None:
                raise PermissionError("Network access forbidden in P48 prospective intake")

            try:
                socket.socket = _network_blocked  # type: ignore[assignment]
                result = intake_prospective_pregame_bundle(
                    bundle_dir,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=tmp / "admitted",
                    run_root=tmp / "runs",
                    source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T12:00:00Z",
                )
                self.assertEqual(result.status, "CREATED")
            finally:
                socket.socket = original_socket

    def test_symlink_escape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            # Create an escaping symlink inside the bundle pointing to parent
            secret_target = tmp / "escaped_secret.txt"
            secret_target.write_text("secret", encoding="utf-8")
            symlink_path = bundle_dir / "escape_link.json"
            symlink_path.symlink_to(secret_target)

            with self.assertRaisesRegex(
                ValueError,
                "P47A_PATH_TRAVERSAL_OR_SYMLINK_ESCAPE",
            ):
                intake_prospective_pregame_bundle(
                    bundle_dir,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=tmp / "admitted",
                    run_root=tmp / "runs",
                    source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T12:00:00Z",
                )

    def test_invalid_probability_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            analysis[0]["model_home_probability"] = "1.50"
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            with self.assertRaisesRegex(
                ValueError,
                "must be strictly between 0 and 1",
            ):
                intake_prospective_pregame_bundle(
                    bundle_dir,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=tmp / "admitted",
                    run_root=tmp / "runs",
                    source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T12:00:00Z",
                )

    def test_invalid_odds_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            bundle_dir = tmp / "external_bundle"
            analysis, schedule, src_manifest, run_manifest = _sample_p35a_bundle()
            analysis[0]["home_decimal_odds"] = "0.95"
            _write_bundle(bundle_dir, analysis, schedule, src_manifest, run_manifest)

            with self.assertRaisesRegex(
                ValueError,
                "must be strictly greater than 1.0",
            ):
                intake_prospective_pregame_bundle(
                    bundle_dir,
                    repository_root=REPOSITORY_ROOT,
                    admission_root=tmp / "admitted",
                    run_root=tmp / "runs",
                    source_identity=P48A_DEFAULT_SOURCE_IDENTITY,
                    intake_timestamp_utc="2026-08-18T12:00:00Z",
                )

    def test_cli_failure_returns_nonzero(self) -> None:
        import io
        from unittest.mock import patch

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            exit_code = cli_main(
                [
                    "prospective-intake",
                    "--bundle",
                    "/nonexistent/bundle/path",
                ]
            )
            self.assertEqual(exit_code, 1)
            self.assertIn("ERROR:", mock_stderr.getvalue())

    def test_no_forbidden_network_or_storage_dependencies(self) -> None:
        use_case_path = (
            REPOSITORY_ROOT
            / "src/match_analysis/application/use_cases/p48a_atomic_prospective_pregame_intake.py"
        )
        cli_path = (
            REPOSITORY_ROOT
            / "src/match_analysis/interfaces/cli/prospective_pregame_intake.py"
        )

        for path in (use_case_path, cli_path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])

            forbidden = {"requests", "urllib", "httpx", "aiohttp", "sqlite3", "psycopg2", "redis"}
            found_forbidden = imported.intersection(forbidden)
            self.assertEqual(
                found_forbidden,
                set(),
                f"Forbidden dependencies found in {path}: {found_forbidden}",
            )

    def test_canonical_forward_sample_count_remains_zero(self) -> None:
        forward_summary = get_p45a_forward_summary(REPOSITORY_ROOT)
        self.assertEqual(
            forward_summary["forward_sample_count"],
            0,
            "Canonical forward sample count must be 0 and unchanged",
        )


if __name__ == "__main__":
    unittest.main()


