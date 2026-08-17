"""Focused unit and integration tests for P47A external P35A bundle admission."""

from __future__ import annotations

import ast
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
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
    create_p45a_paper_run,
    get_p45a_forward_summary,
)
from match_analysis.application.use_cases.p46a_p35a_pregame_adapter import (
    P46A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
)
from match_analysis.application.use_cases.p47a_external_bundle_admission import (
    P47A_ADMISSION_RECORD_SCHEMA,
    P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
    P47A_EXTERNAL_ADMISSION_SOURCE_IDENTITY,
    REQUIRED_BUNDLE_FILES,
    admit_external_p35a_bundle,
    compute_deterministic_bundle_fingerprint,
)
from match_analysis.interfaces.cli.admit_external_p35a_bundle import (
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


def _sample_p35a_bundle() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return minimal truthful P35A contract rehearsal fixture."""

    analysis_rows = [
        {
            "schema_version": "p30a.moneyline_paper_analysis.v1",
            "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
            "game_id": "824192",
            "scheduled_start": "2026-05-17T18:10:00Z",
            "home_team": "Houston Astros",
            "away_team": "Texas Rangers",
            "structural_status": "EDGE_AVAILABLE",
            "status": "EDGE_AVAILABLE",
            "prediction_id": "5de824e18c72b10c7d77ea843c4ee3a7787fbf2e789c86feac209206b16b39c5",
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": "0.5861848091915969598190982782",
            "market_price_id": "p28ab:6b6f72c2b12b80b8c84b434105976a0f57d5c6f13f26e87bf7f935a856779bf5",
            "price_observed_at": "2026-05-17T02:04:35.317592Z",
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
            "scheduled_start": "2026-05-18T23:10:00Z",
            "home_team": "Kansas City Royals",
            "away_team": "Boston Red Sox",
            "structural_status": "EDGE_AVAILABLE",
            "status": "EDGE_AVAILABLE",
            "prediction_id": "55e5ec556f198337f0f9a2ad090b052f3910c70b7c57d8fd829c0b3a94caa337",
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": "0.5303627085439037814014483218",
            "market_price_id": "p28ab:fff29ab049b29e9c47ca097af08b1df1048a85933d4719606a54d4b847da68c2",
            "price_observed_at": "2026-05-18T01:21:04.535534Z",
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
            "scheduled_start": "2026-05-17T17:35:00Z",
            "home_team": "Washington Nationals",
            "away_team": "Baltimore Orioles",
            "structural_status": "FEATURE_UNAVAILABLE",
            "status": "FEATURE_UNAVAILABLE",
            "prediction_id": None,
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": None,
            "market_price_id": "p28ab:28ab9f198d81aad8d36b7d4f38b3a175962d6ec9f4af50904475accd2e298bb5",
            "price_observed_at": "2026-05-17T02:04:35.317592Z",
            "home_decimal_odds": "1.90",
            "away_decimal_odds": "1.60",
            "home_no_vig_probability": "0.4571428571428571428571428570",
            "away_no_vig_probability": "0.5428571428571428571428571426",
            "home_edge": None,
            "away_edge": None,
            "controlled_unavailable_reason": "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY",
        },
    ]

    schedule_rows = [
        {
            "schema_version": "p23f2.mlb_official_normalized.v1",
            "provider_game_id": "824192",
            "game_pk": 824192,
            "game_number": 1,
            "official_date": "2026-05-17",
            "scheduled_start_utc": "2026-05-17T18:10:00Z",
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
            "official_date": "2026-05-18",
            "scheduled_start_utc": "2026-05-18T23:10:00Z",
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
            "official_date": "2026-05-17",
            "scheduled_start_utc": "2026-05-17T17:35:00Z",
            "status": "Scheduled",
            "final": False,
            "home_team": {"id": 120, "abbreviation": "WSH", "name": "Washington Nationals"},
            "away_team": {"id": 110, "abbreviation": "BAL", "name": "Baltimore Orioles"},
            "home_score": None,
            "away_score": None,
        },
    ]

    source_manifest = {
        "schema_version": "p30a.source_manifest.v1",
        "tsl_authority": {
            "authority_label": "TSL_BLOB3RD",
            "legacy_repository": "Betting-pool",
        },
    }

    run_manifest = {
        "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
        "target_date": "2026-05-17",
    }

    return analysis_rows, schedule_rows, source_manifest, run_manifest


def _write_bundle(
    bundle_dir: Path,
    analysis_rows: list[dict[str, Any]],
    schedule_rows: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
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


class P47AExternalBundleAdmissionTests(unittest.TestCase):
    def test_positive_admission_and_p46_p45_flow(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "p35a_external_bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)

            admission_root = Path(raw_dir) / "admitted_root"
            admitted = admit_external_p35a_bundle(
                bundle_dir,
                admission_root=admission_root,
                source_identity=P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
            )

            self.assertEqual(admitted.status, "ADMITTED")
            self.assertEqual(admitted.run_id, run_manifest["run_id"])
            self.assertEqual(admitted.target_date, run_manifest["target_date"])
            self.assertEqual(admitted.productive_row_count, 2)
            self.assertEqual(admitted.excluded_row_count, 1)
            self.assertTrue(admitted.imported_bundle_dir.is_dir())
            self.assertTrue(admitted.admission_record_path.is_file())
            self.assertTrue(admitted.normalized_pregame_path.is_file())

            # Verify immutable snapshot was copied
            raw_bundle = admitted.imported_bundle_dir / "raw_bundle"
            self.assertTrue(raw_bundle.is_dir())
            for filename in REQUIRED_BUNDLE_FILES:
                self.assertTrue((raw_bundle / filename).is_file())

            # Load normalized pregame input directly
            normalized_loaded = load_normalized_pregame_input(admitted.normalized_pregame_path)
            self.assertEqual(len(normalized_loaded.prediction_rows), 2)
            self.assertEqual(len(normalized_loaded.exclusion_rows), 1)

            # Test downstream P45 create-run in HISTORICAL_REHEARSAL mode
            p45_runs = Path(raw_dir) / "p45_runs"
            p45_result = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=normalized_loaded,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=p45_runs,
                created_at_utc="2026-05-17T12:00:00Z",
            )

            self.assertEqual(p45_result.status, "CREATED")
            self.assertEqual(p45_result.manifest["run_classification"], CLASSIFICATION_HISTORICAL_REHEARSAL)
            self.assertEqual(p45_result.manifest["lifecycle_state"], STATE_FROZEN)

            # Verify FORWARD_SAMPLE_COUNT remains 0
            forward_summary = get_p45a_forward_summary(REPOSITORY_ROOT)
            self.assertEqual(forward_summary["forward_sample_count"], 0)

    def test_idempotent_admission_and_conflict_rejection(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)

            admission_root = Path(raw_dir) / "admitted_root"

            # Pass 1: First admission
            first = admit_external_p35a_bundle(
                bundle_dir,
                admission_root=admission_root,
                source_identity=P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
            )
            self.assertEqual(first.status, "ADMITTED")

            # Pass 2: Identical second admission -> idempotent
            second = admit_external_p35a_bundle(
                bundle_dir,
                admission_root=admission_root,
                source_identity=P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
            )
            self.assertEqual(second.status, "RECOGNIZED_IDENTICAL")
            self.assertEqual(first.bundle_fingerprint, second.bundle_fingerprint)
            self.assertEqual(first.admitted_bundle_id, second.admitted_bundle_id)

            # Pass 3: Conflicting same-run-id bundle with tampered contents
            tampered_record = deepcopy(first.admission_record)
            tampered_record["bundle_fingerprint"] = "0" * 64
            first.admission_record_path.write_text(json.dumps(tampered_record), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "P47A_BUNDLE_AUTHORITY_CONFLICT"):
                admit_external_p35a_bundle(
                    bundle_dir,
                    admission_root=admission_root,
                    source_identity=P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
                )

    def test_rejection_of_missing_required_file(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        for missing in REQUIRED_BUNDLE_FILES:
            with tempfile.TemporaryDirectory() as raw_dir:
                bundle_dir = Path(raw_dir) / "bundle"
                _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)
                (bundle_dir / missing).unlink()

                with self.assertRaises(FileNotFoundError):
                    admit_external_p35a_bundle(bundle_dir)

    def test_rejection_of_malformed_json_and_jsonl(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)
            (bundle_dir / "run_manifest.json").write_text("{malformed", encoding="utf-8")

            with self.assertRaises(ValueError):
                admit_external_p35a_bundle(bundle_dir)

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)
            (bundle_dir / "analysis.jsonl").write_text("{\"not_json\"\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                admit_external_p35a_bundle(bundle_dir)

    def test_rejection_of_mismatched_manifest_run_identity(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()
        tainted_analysis = deepcopy(analysis_rows)
        tainted_analysis[0]["run_id"] = "mismatched_run_id_123"

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, tainted_analysis, schedule_rows, source_manifest, run_manifest)

            with self.assertRaisesRegex(ValueError, "run_id mismatch"):
                admit_external_p35a_bundle(bundle_dir)

    def test_rejection_of_mismatched_source_manifest_fingerprint(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()
        tainted_run_manifest = deepcopy(run_manifest)
        tainted_run_manifest["source_manifest_fingerprint"] = "0" * 64

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, tainted_run_manifest)

            with self.assertRaisesRegex(ValueError, "source_manifest_fingerprint mismatch"):
                admit_external_p35a_bundle(bundle_dir)

    def test_rejection_of_outcome_contamination(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        for forbidden in ("home_score", "away_score", "actual_winner", "settlement"):
            tainted_analysis = deepcopy(analysis_rows)
            tainted_analysis[0][forbidden] = "HOME"

            with tempfile.TemporaryDirectory() as raw_dir:
                bundle_dir = Path(raw_dir) / "bundle"
                _write_bundle(bundle_dir, tainted_analysis, schedule_rows, source_manifest, run_manifest)

                with self.assertRaisesRegex(ValueError, "P44A_PREGAME_OUTCOME_FIELDS_REJECTED"):
                    admit_external_p35a_bundle(bundle_dir)

    def test_rejection_of_temporal_post_start_market_observation(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()
        late_analysis = deepcopy(analysis_rows)
        # Price observed at scheduled start (not strictly pregame)
        late_analysis[0]["price_observed_at"] = late_analysis[0]["scheduled_start"]

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, late_analysis, schedule_rows, source_manifest, run_manifest)

            with self.assertRaisesRegex(ValueError, "not strictly pregame"):
                admit_external_p35a_bundle(bundle_dir)

    def test_rejection_of_live_label_for_contract_rehearsal(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)

            with self.assertRaisesRegex(ValueError, "must not be labeled live"):
                admit_external_p35a_bundle(bundle_dir, source_identity="LIVE_PROVIDER_FEED")

    def test_rejection_of_path_traversal_and_symlink_escape(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)

            # Create an escaping symlink
            escape_link = bundle_dir / "external_link.json"
            escape_link.symlink_to(Path(raw_dir).parent)

            with self.assertRaisesRegex(ValueError, "P47A_PATH_TRAVERSAL_OR_SYMLINK_ESCAPE"):
                admit_external_p35a_bundle(bundle_dir)

    def test_rejection_of_source_changing_during_admission(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)

            # Mock _snapshot_file_bytes to mutate between first and second calls
            from match_analysis.application.use_cases import p47a_external_bundle_admission

            orig_snapshot = p47a_external_bundle_admission._snapshot_file_bytes
            call_count = [0]

            def mutating_snapshot(root: Path) -> dict[str, bytes]:
                call_count[0] += 1
                res = orig_snapshot(root)
                if call_count[0] == 2:
                    res["analysis.jsonl"] = b"mutated content\n"
                return res

            p47a_external_bundle_admission._snapshot_file_bytes = mutating_snapshot
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "P47A_EXTERNAL_BUNDLE_CHANGED_DURING_ADMISSION"
                ):
                    admit_external_p35a_bundle(bundle_dir)
            finally:
                p47a_external_bundle_admission._snapshot_file_bytes = orig_snapshot

    def test_cli_execution_end_to_end(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)

            out_root = Path(raw_dir) / "admitted"
            exit_code = cli_main(
                [
                    "admit",
                    "--bundle",
                    str(bundle_dir),
                    "--output-root",
                    str(out_root),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(out_root.is_dir())

    def test_no_network_imports(self) -> None:
        use_case_path = (
            REPOSITORY_ROOT
            / "src/match_analysis/application/use_cases/p47a_external_bundle_admission.py"
        )
        cli_path = (
            REPOSITORY_ROOT
            / "src/match_analysis/interfaces/cli/admit_external_p35a_bundle.py"
        )

        for path in (use_case_path, cli_path):
            imported = _imported_roots(path)
            for forbidden in ("urllib", "requests", "http", "socket", "ssl", "aiohttp"):
                self.assertNotIn(forbidden, imported)
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("urlopen", source)
            self.assertNotIn("verify=False", source)

    def test_protected_authorities_remain_invariant(self) -> None:
        before = {path: path.read_bytes() for path in PROTECTED_PATHS}
        hashed = protected_authority_hashes(REPOSITORY_ROOT)

        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            _write_bundle(bundle_dir, analysis_rows, schedule_rows, source_manifest, run_manifest)
            admit_external_p35a_bundle(bundle_dir, admission_root=Path(raw_dir) / "admitted")

        after = {path: path.read_bytes() for path in PROTECTED_PATHS}
        self.assertEqual(before, after)
        self.assertEqual(hashed, protected_authority_hashes(REPOSITORY_ROOT))


if __name__ == "__main__":
    unittest.main()
