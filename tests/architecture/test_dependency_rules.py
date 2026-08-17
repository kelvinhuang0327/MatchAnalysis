"""Executable modular-monolith dependency rules."""

import ast
from hashlib import sha256
from pathlib import Path
import re
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "match_analysis"
LEGACY_ABSOLUTE_PATH = "/Users/kelvin/Kelvin-WorkSpace/Betting-pool"

AUTHORIZED_SOURCE_PATHS = {
    "__init__.py",
    "application/__init__.py",
    "application/ports/__init__.py",
    "application/ports/legacy_prediction_source.py",
    "application/ports/legacy_schedule_source.py",
    "application/ports/schedule_observation_source.py",
    "application/use_cases/__init__.py",
    "application/use_cases/assess_legacy_prediction_quarantine.py",
    "application/use_cases/build_schedule_observation_revision_chains.py",
    "application/use_cases/capture_schedule_observation.py",
    "application/use_cases/construct_match_identities.py",
    "application/use_cases/evaluate_schedule_pregame_eligibility.py",
    "application/use_cases/import_legacy_prediction_snapshot.py",
    "application/use_cases/import_legacy_schedule_snapshot.py",
    "application/use_cases/link_legacy_quarantine_snapshots.py",
    "application/use_cases/materialize_schedule_baseball_games.py",
    "application/use_cases/project_schedule_identity_candidates.py",
    "application/use_cases/resolve_schedule_participant_identities.py",
    "application/use_cases/select_schedule_observations_as_of.py",
    "baseball/__init__.py",
    "baseball/domain/__init__.py",
    "baseball/domain/canonical_utc.py",
    "baseball/domain/game.py",
    "baseball/domain/legacy_prediction_quarantine.py",
    "baseball/domain/match_identity_authority.py",
    "baseball/domain/prediction.py",
    "baseball/domain/prediction_admission.py",
    "baseball/domain/prediction_source_observation.py",
    "baseball/domain/participant_identity_resolution.py",
    "baseball/domain/pregame_eligibility.py",
    "baseball/domain/moneyline_feature_snapshot.py",
    "baseball/domain/moneyline_market_snapshot.py",
    "baseball/domain/moneyline_model_artifact.py",
    "baseball/domain/moneyline_walk_forward_fold.py",
    "baseball/domain/future_evaluation_fold.py",
    "baseball/domain/quarantine_link.py",
    "baseball/domain/schedule.py",
    "baseball/domain/schedule_game_materialization.py",
    "baseball/domain/schedule_identity_candidate.py",
    "baseball/domain/schedule_observation.py",
    "baseball/domain/schedule_revision.py",
    "baseball/domain/schedule_snapshot.py",
    "core/__init__.py",
    "core/identity.py",
    "core/provenance.py",
    "core/time.py",
    "application/use_cases/prospective_prediction_admission_artifacts.py",
    "application/use_cases/run_prospective_prediction_admission_workflow.py",
    "application/use_cases/generate_moneyline_predictions.py",
    "application/use_cases/generate_paper_moneyline_batch.py",
    "application/use_cases/generate_tsl_moneyline_edge_batch.py",
    "application/use_cases/build_moneyline_paper_source_bundle.py",
    "application/use_cases/moneyline_paper_run_bundle.py",
    "application/use_cases/run_moneyline_paper_analysis.py",
    "application/use_cases/run_daily_moneyline_paper_analysis.py",
    "application/use_cases/settle_daily_moneyline_paper_run.py",
    "application/use_cases/p34a_daily_moneyline_settlement_artifacts.py",
    "application/use_cases/moneyline_paper_analysis_artifacts.py",
    "application/use_cases/moneyline_inference_artifacts.py",
    "application/use_cases/paper_moneyline_batch_artifacts.py",
    "application/use_cases/settle_paper_moneyline_batch.py",
    "application/use_cases/paper_moneyline_feedback_artifacts.py",
    "application/use_cases/moneyline_walk_forward_artifacts.py",
    "application/use_cases/reconstruct_moneyline_walk_forward_model.py",
    "application/use_cases/replay_historical_moneyline_predictions.py",
    "infrastructure/__init__.py",
    "infrastructure/legacy_betting_pool/__init__.py",
    "infrastructure/legacy_betting_pool/p83e_jsonl.py",
    "infrastructure/legacy_betting_pool/p84b_schedule_jsonl.py",
    "infrastructure/legacy_betting_pool/tsl_odds_history.py",
    "infrastructure/mlb_schedule/__init__.py",
    "infrastructure/mlb_schedule/explicit_payload_source.py",
    "infrastructure/providers/mlb_official_historical_source.py",
    "infrastructure/sources/__init__.py",
    "infrastructure/sources/tsl_moneyline_acquisition.py",
    "infrastructure/sources/tsl_moneyline_history.py",
    "infrastructure/sources/p39a_tsl_market_snapshot.py",
    "interfaces/__init__.py",
    "interfaces/cli/__init__.py",
    "interfaces/cli/prospective_prediction_admission.py",
    "interfaces/cli/moneyline_inference.py",
    "interfaces/cli/generate_paper_moneyline_batch.py",
    "interfaces/cli/generate_tsl_moneyline_edge_batch.py",
    "interfaces/cli/run_moneyline_paper_analysis.py",
    "interfaces/cli/run_daily_moneyline_paper_analysis.py",
    "interfaces/cli/settle_daily_moneyline_paper_run.py",
    "interfaces/cli/moneyline_walk_forward_replay.py",
    "application/use_cases/build_admitted_prediction_observation_snapshot.py",
    "application/use_cases/admitted_prediction_observation_artifacts.py",
    "interfaces/cli/admitted_prediction_observation_snapshot.py",
    "baseball/domain/final_result_observation.py",
    "application/use_cases/attach_final_results_to_admitted_predictions.py",
    "application/use_cases/final_result_attachment_artifacts.py",
    "interfaces/cli/final_result_attachment.py",
    "baseball/domain/prediction_evaluation.py",
    "application/use_cases/build_prediction_evaluation_scorecard.py",
    "application/use_cases/prediction_evaluation_artifacts.py",
    "interfaces/cli/prediction_evaluation_scorecard.py",
    "baseball/domain/prediction_feedback.py",
    "baseball/domain/result_only_paper_decision.py",
    "application/use_cases/build_prediction_feedback_ledger.py",
    "application/use_cases/prediction_feedback_artifacts.py",
    "application/use_cases/replay_historical_prediction_feedback.py",
    "application/use_cases/historical_feedback_replay_artifacts.py",
    "application/use_cases/replay_multifold_historical_candidates.py",
    "application/use_cases/multifold_historical_candidate_artifacts.py",
    "application/use_cases/build_result_only_paper_decision_replay.py",
    "application/use_cases/result_only_paper_decision_artifacts.py",
    "interfaces/cli/prediction_feedback_ledger.py",
    "interfaces/cli/historical_feedback_replay.py",
    "interfaces/cli/settle_paper_moneyline_batch.py",
    "interfaces/cli/multifold_historical_candidate_replay.py",
    "interfaces/cli/result_only_paper_decision_replay.py",
    "baseball/domain/prediction_learning_eligibility.py",
    "baseball/domain/supervised_training_example.py",
    "application/use_cases/assess_prediction_learning_candidates.py",
    "application/use_cases/prediction_learning_candidate_artifacts.py",
    "application/use_cases/acquire_future_moneyline_history.py",
    "application/use_cases/acquire_tsl_moneyline_snapshot.py",
    "application/use_cases/materialize_future_moneyline_fold.py",
    "application/use_cases/future_moneyline_fold_artifacts.py",
    "application/use_cases/materialize_moneyline_training_dataset.py",
    "application/use_cases/moneyline_training_dataset_artifacts.py",
    "application/use_cases/moneyline_challenger_artifacts.py",
    "application/use_cases/train_moneyline_challenger.py",
    "interfaces/cli/prediction_learning_candidate_gate.py",
    "interfaces/cli/materialize_moneyline_training_dataset.py",
    "interfaces/cli/train_moneyline_challenger.py",
    "interfaces/cli/acquire_future_moneyline_fold.py",
    "baseball/domain/moneyline_oos_comparison.py",
    "application/use_cases/evaluate_moneyline_challenger_oos.py",
    "application/use_cases/moneyline_oos_comparison_artifacts.py",
    "application/use_cases/evaluate_multifold_moneyline_oos.py",
    "application/use_cases/multifold_moneyline_oos_artifacts.py",
    "application/use_cases/offline_moneyline_retraining_artifacts.py",
    "application/use_cases/offline_moneyline_retraining_baseline.py",
    "application/use_cases/rolling_moneyline_oos.py",
    "application/use_cases/rolling_moneyline_oos_artifacts.py",
    "application/use_cases/join_p37_oos_market_snapshots.py",
    "application/use_cases/p39a_market_join_artifacts.py",
    "application/use_cases/p40a_moneyline_paper_bet_pass.py",
    "application/use_cases/p41a_walk_forward_ev_margin_policy.py",
    "application/use_cases/p42a_offline_end_to_end_paper_workflow.py",
    "application/use_cases/p43a_pregame_freeze.py",
    "application/use_cases/p43a_postgame_settle.py",
    "application/use_cases/p44a_historical_source_adapter.py",
    "application/use_cases/p44a_normalized_workflow_input.py",
    "application/use_cases/moneyline_probability_calibration.py",
    "application/use_cases/p38a_probability_calibration.py",
    "application/use_cases/p38a_probability_calibration_artifacts.py",
    "interfaces/cli/evaluate_moneyline_challenger_oos.py",
    "interfaces/cli/evaluate_multifold_moneyline_oos.py",
    "interfaces/cli/offline_moneyline_retraining_baseline.py",
    "interfaces/cli/rolling_moneyline_oos.py",
    "interfaces/cli/p38a_probability_calibration.py",
    "interfaces/cli/join_p37_oos_market_snapshots.py",
    "interfaces/cli/run_p40a_moneyline_paper_bet_pass.py",
    "interfaces/cli/run_p41a_walk_forward_ev_margin_policy.py",
    "interfaces/cli/run_p42a_offline_end_to_end_paper_workflow.py",
    "interfaces/cli/run_p43a_two_phase_paper_workflow.py",
    "application/use_cases/p45a_paper_run_ledger.py",
    "interfaces/cli/run_p45a_paper_lifecycle.py",
    "application/use_cases/p46a_p35a_pregame_adapter.py",
    "interfaces/cli/adapt_p35a_pregame_input.py",
    "application/use_cases/p47a_external_bundle_admission.py",
    "interfaces/cli/admit_external_p35a_bundle.py",
    "application/use_cases/generate_moneyline_market_movement.py",
    "application/use_cases/moneyline_market_movement_artifacts.py",
    "interfaces/cli/replay_moneyline_market_movement.py",
    "baseball/domain/paper_moneyline_bet_pass.py",
}

MLB_SCHEDULE_PAYLOAD_ADAPTER_RUNTIME_PATHS = (
    PACKAGE_ROOT
    / "infrastructure"
    / "mlb_schedule"
    / "explicit_payload_source.py",
)

P84B_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "schedule.py",
    PACKAGE_ROOT / "application" / "ports" / "legacy_schedule_source.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "import_legacy_schedule_snapshot.py",
    PACKAGE_ROOT
    / "infrastructure"
    / "legacy_betting_pool"
    / "p84b_schedule_jsonl.py",
)

QUARANTINE_LINK_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "quarantine_link.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "link_legacy_quarantine_snapshots.py",
)

LEGACY_PREDICTION_QUARANTINE_ASSESSMENT_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "legacy_prediction_quarantine.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "assess_legacy_prediction_quarantine.py",
)

SCHEDULE_OBSERVATION_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "schedule_observation.py",
    PACKAGE_ROOT
    / "application"
    / "ports"
    / "schedule_observation_source.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "capture_schedule_observation.py",
)

SCHEDULE_OBSERVATION_REVISION_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "schedule_revision.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "build_schedule_observation_revision_chains.py",
)

SCHEDULE_OBSERVATION_AS_OF_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "schedule_snapshot.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "select_schedule_observations_as_of.py",
)

SCHEDULE_IDENTITY_CANDIDATE_RUNTIME_PATHS = (
    PACKAGE_ROOT
    / "baseball"
    / "domain"
    / "schedule_identity_candidate.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "project_schedule_identity_candidates.py",
)

PARTICIPANT_IDENTITY_RESOLUTION_RUNTIME_PATHS = (
    PACKAGE_ROOT
    / "baseball"
    / "domain"
    / "participant_identity_resolution.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "resolve_schedule_participant_identities.py",
)

MATCH_IDENTITY_CONSTRUCTION_RUNTIME_PATHS = (
    PACKAGE_ROOT
    / "baseball"
    / "domain"
    / "match_identity_authority.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "construct_match_identities.py",
)

SCHEDULE_BASEBALL_GAME_MATERIALIZATION_RUNTIME_PATHS = (
    PACKAGE_ROOT
    / "baseball"
    / "domain"
    / "schedule_game_materialization.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "materialize_schedule_baseball_games.py",
)

SCHEDULE_PREGAME_ELIGIBILITY_RUNTIME_PATHS = (
    PACKAGE_ROOT
    / "baseball"
    / "domain"
    / "pregame_eligibility.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "evaluate_schedule_pregame_eligibility.py",
)

PREDICTION_ADMISSION_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "canonical_utc.py",
    PACKAGE_ROOT / "baseball" / "domain" / "prediction_source_observation.py",
    PACKAGE_ROOT / "baseball" / "domain" / "prediction_admission.py",
)

PREDICTION_SOURCE_OBSERVATION_AUTHORIZED_CONSTRUCTOR_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "prediction_admission.py",
)

P83E_BASELINE_SHA256 = {
    "src/match_analysis/baseball/domain/prediction.py": (
        "6b41afc68bbe58fac48f68578edab353a64d0a33760f1811337ebb9d9bcb3735"
    ),
    "src/match_analysis/application/ports/legacy_prediction_source.py": (
        "e67a4aa577bbf96b26dff8652432614e44c26a72938357f798e88535d201e9f1"
    ),
    (
        "src/match_analysis/application/use_cases/"
        "import_legacy_prediction_snapshot.py"
    ): "18bf6cd1134b21ff523528409dee63494d0d90fc24e6e52af4f246fa955c07af",
    (
        "src/match_analysis/infrastructure/legacy_betting_pool/"
        "p83e_jsonl.py"
    ): "4cebaaddab8d96f0b2349a295a3c8f69a8c206f866da59f80f157fd918f2ed71",
    "tests/unit/test_prediction_contracts.py": (
        "66da8fb31b3a5c7aeda1999e35bb79aea0179de76445b8a38c27fb43f9b20525"
    ),
    "tests/unit/test_legacy_prediction_evidence_contracts.py": (
        "9a2789551bbc565546c32581f0dbf800f8549594376adb1231a0e27f6c1bd49a"
    ),
    "tests/characterization/test_p83e_snapshot_adapter.py": (
        "3e810b61d78744496dc778a1d5e66e6b375c1cd2294269bd8398e3346fd3e9b3"
    ),
    "tests/characterization/test_p83e_public_evidence_snapshot.py": (
        "125c2884220f6f065cda846791c7535a953318e4a783f9f05ed85de484b9db50"
    ),
}

P84B_BASELINE_SHA256 = {
    "src/match_analysis/baseball/domain/schedule.py": (
        "ba1e5e86a28a5cecf56e7b23a841d1e38ffb6bf84f708996ae3cc657418289da"
    ),
    "src/match_analysis/application/ports/legacy_schedule_source.py": (
        "1b39b0ea1880c4c032dc185c732624f8952a06a518939b4a67caa61a0b74b837"
    ),
    (
        "src/match_analysis/application/use_cases/"
        "import_legacy_schedule_snapshot.py"
    ): "c3b57ed312ac31e8f228d65767fa8e218f229a446d904b821719a1e86abfe80c",
    (
        "src/match_analysis/infrastructure/legacy_betting_pool/"
        "p84b_schedule_jsonl.py"
    ): "ad087db009dd7f0a4cd13d0a6536adf405f7dbbad5ab45f2a8cb5381220edb6a",
    "tests/unit/test_schedule_contracts.py": (
        "10dbe0257cf6917e04e4ad122d51df1d001ec3952e4fd7b4513e85d09fab1e39"
    ),
    "tests/characterization/test_p84b_schedule_adapter.py": (
        "8228ebecd3d3c87bcceeb6183bfa3cbaa2cdd637892d0ce339e12bf00269b065"
    ),
}

SCHEDULE_OBSERVATION_BASELINE_SHA256 = {
    "src/match_analysis/baseball/domain/schedule_observation.py": (
        "027d843f7c7a34582873c7ecfe366f941b936b1dd14b5b75a8d073ff6dfad7ad"
    ),
    "src/match_analysis/application/ports/schedule_observation_source.py": (
        "1db186058ba80f0f752480e09275f680a39a286def63236b45da3e463734ce90"
    ),
    (
        "src/match_analysis/application/use_cases/"
        "capture_schedule_observation.py"
    ): "7823b7aedd9e2bc4d96443b7b5265f0787596d035abb1023225b1d3ad244901e",
    "tests/unit/test_schedule_observation_contracts.py": (
        "6e653c0c269a863f1b7f8bba173cbf55e075ad06cd63dcfb6ac727ac72cf9e86"
    ),
    "tests/characterization/test_schedule_observation_fixture.py": (
        "079fb5939c03d131bee1a6b4dfce28fb242b180be2a96831b71edc12f6c355e0"
    ),
    "tests/fixtures/mlb_schedule_observation_v1.json": (
        "0ad8c16edebbc40d5592749beff04129e734f43110aed03376d3c43eeb64003c"
    ),
}

P3_QUARANTINE_LINK_BASELINE_SHA256 = {
    "src/match_analysis/baseball/domain/quarantine_link.py": (
        "9b5b6405c80a020237228e5c68d9ea7cf96edc243b00fc8cec162525723789fb"
    ),
    (
        "src/match_analysis/application/use_cases/"
        "link_legacy_quarantine_snapshots.py"
    ): "734294dd77d95834934ac0eee53a6cc09059a1a2faad5925c1a3080a4308812e",
    "tests/unit/test_quarantine_link_contracts.py": (
        "7f07b9a9114885c39d2a82e13f698a86a63953796c2f673ac419a59b0cd4c944"
    ),
    "tests/characterization/test_p83e_p84b_quarantine_link.py": (
        "296c53df8d6855215a15e296b2109f602e2b7cd14ad2a9dcd5fbff983744353d"
    ),
}


def source_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def module_name(path: Path) -> str:
    relative_parts = list(path.relative_to(PACKAGE_ROOT).with_suffix("").parts)
    if relative_parts[-1] == "__init__":
        relative_parts.pop()
    return ".".join(["match_analysis", *relative_parts])


def imported_modules(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current_module = module_name(path)
    current_package = (
        current_module.split(".")
        if path.name == "__init__.py"
        else current_module.split(".")[:-1]
    )
    imports: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                levels_up = node.level - 1
                base = (
                    current_package
                    if levels_up == 0
                    else current_package[:-levels_up]
                )
                target = ".".join(
                    [*base, *(node.module or "").split(".")]
                ).rstrip(".")
            else:
                target = node.module or ""
            imports.append((target, node.lineno))

    return imports


class DependencyRuleTests(unittest.TestCase):
    def assert_layer_excludes(
        self,
        paths: list[Path],
        forbidden_prefixes: tuple[str, ...],
    ) -> None:
        violations: list[str] = []
        for path in paths:
            for target, line_number in imported_modules(path):
                if target.startswith(forbidden_prefixes):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_core_dependencies_point_inward(self) -> None:
        self.assert_layer_excludes(
            sorted((PACKAGE_ROOT / "core").rglob("*.py")),
            (
                "match_analysis.application",
                "match_analysis.infrastructure",
                "match_analysis.interfaces",
            ),
        )

    def test_baseball_domain_dependencies_point_inward(self) -> None:
        self.assert_layer_excludes(
            sorted((PACKAGE_ROOT / "baseball" / "domain").rglob("*.py")),
            (
                "match_analysis.application",
                "match_analysis.infrastructure",
                "match_analysis.interfaces",
            ),
        )

    def test_application_does_not_depend_on_outer_layers(self) -> None:
        paths = sorted((PACKAGE_ROOT / "application").rglob("*.py"))
        p23f2_acquisition = (
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "acquire_future_moneyline_history.py"
        )
        p31a_source_bundle = (
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "build_moneyline_paper_source_bundle.py"
        )
        p32a_tsl_acquisition = (
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "acquire_tsl_moneyline_snapshot.py"
        )
        violations = []
        for path in paths:
            for target, line_number in imported_modules(path):
                is_allowlisted_p23f2_source = (
                    path == p23f2_acquisition
                    and target
                    == "match_analysis.infrastructure.providers.mlb_official_historical_source"
                )
                is_allowlisted_p31a_source = (
                    path == p31a_source_bundle
                    and target
                    == "match_analysis.infrastructure.sources.tsl_moneyline_history"
                )
                is_allowlisted_p32a_source = (
                    path == p32a_tsl_acquisition
                    and target
                    in {
                        "match_analysis.infrastructure.providers.mlb_official_historical_source",
                        "match_analysis.infrastructure.sources.tsl_moneyline_acquisition",
                    }
                )
                if target.startswith(
                    ("match_analysis.infrastructure", "match_analysis.interfaces")
                ) and not (
                    is_allowlisted_p23f2_source
                    or is_allowlisted_p31a_source
                    or is_allowlisted_p32a_source
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_source_has_no_legacy_imports(self) -> None:
        violations: list[str] = []
        for path in source_files():
            for target, line_number in imported_modules(path):
                segments = target.split(".")
                root = segments[0]
                is_task_module = any(
                    re.fullmatch(r"_?p\d{3}", segment) is not None
                    for segment in segments
                )
                if (
                    root in {"wbc_backend", "models", "strategy"}
                    or "Betting-pool" in target
                    or is_task_module
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_source_has_no_absolute_legacy_path(self) -> None:
        violations = [
            str(path.relative_to(REPOSITORY_ROOT))
            for path in source_files()
            if LEGACY_ABSOLUTE_PATH in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(violations, [])

    def test_first_slice_has_no_persistence_or_runtime_integration(self) -> None:
        forbidden_path_parts = {
            "api",
            "database",
            "db",
            "provider",
            "scheduler",
            "scripts",
        }
        allowlisted_p23f2_provider = (
            PACKAGE_ROOT
            / "infrastructure"
            / "providers"
            / "mlb_official_historical_source.py"
        )
        violations = [
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in source_files()
            if forbidden_path_parts.intersection(
                path.relative_to(PACKAGE_ROOT).parts
            ) and path != allowlisted_p23f2_provider
        ]
        self.assertEqual(violations, [])

    def test_only_authorized_source_package_tree_exists(self) -> None:
        actual = {
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in source_files()
        }
        self.assertEqual(actual, AUTHORIZED_SOURCE_PATHS)

    def test_runtime_imports_use_only_standard_library_or_local_package(
        self,
    ) -> None:
        violations: list[str] = []
        for path in source_files():
            for target, line_number in imported_modules(path):
                root = target.split(".")[0]
                if (
                    root != "match_analysis"
                    and root != "__future__"
                    and root not in sys.stdlib_module_names
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_schedule_observation_runtime_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "BaseballGame(",
            "MatchIdentity(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in SCHEDULE_OBSERVATION_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_schedule_observation_use_case_does_not_import_infrastructure(
        self,
    ) -> None:
        use_case = (
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "capture_schedule_observation.py"
        )
        violations = [
            f"{use_case.relative_to(REPOSITORY_ROOT)}:{line_number} -> {target}"
            for target, line_number in imported_modules(use_case)
            if target.startswith("match_analysis.infrastructure")
        ]
        self.assertEqual(violations, [])

    def test_p7_schedule_observation_revision_runtime_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "BaseballGame(",
            "MatchIdentity(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in SCHEDULE_OBSERVATION_REVISION_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p7_schedule_observation_revision_use_case_does_not_import_infrastructure(
        self,
    ) -> None:
        use_case = (
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "build_schedule_observation_revision_chains.py"
        )
        violations = [
            f"{use_case.relative_to(REPOSITORY_ROOT)}:{line_number} -> {target}"
            for target, line_number in imported_modules(use_case)
            if target.startswith("match_analysis.infrastructure")
        ]
        self.assertEqual(violations, [])

    def test_p8_schedule_observation_as_of_runtime_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "BaseballGame(",
            "MatchIdentity(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in SCHEDULE_OBSERVATION_AS_OF_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p8_schedule_observation_as_of_use_case_does_not_import_infrastructure(
        self,
    ) -> None:
        use_case = (
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "select_schedule_observations_as_of.py"
        )
        violations = [
            f"{use_case.relative_to(REPOSITORY_ROOT)}:{line_number} -> {target}"
            for target, line_number in imported_modules(use_case)
            if target.startswith("match_analysis.infrastructure")
        ]
        self.assertEqual(violations, [])

    def test_p9_schedule_identity_candidate_runtime_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "BaseballGame(",
            "MatchIdentity(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in SCHEDULE_IDENTITY_CANDIDATE_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p9_schedule_identity_candidate_use_case_does_not_import_infrastructure(
        self,
    ) -> None:
        use_case = (
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "project_schedule_identity_candidates.py"
        )
        violations = [
            f"{use_case.relative_to(REPOSITORY_ROOT)}:{line_number} -> {target}"
            for target, line_number in imported_modules(use_case)
            if target.startswith("match_analysis.infrastructure")
        ]
        self.assertEqual(violations, [])

    def test_p10_participant_identity_resolution_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "BaseballGame(",
            "MatchIdentity(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in PARTICIPANT_IDENTITY_RESOLUTION_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p10_participant_resolution_use_case_does_not_import_infrastructure(
        self,
    ) -> None:
        use_case = (
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "resolve_schedule_participant_identities.py"
        )
        violations = [
            f"{use_case.relative_to(REPOSITORY_ROOT)}:{line_number} -> {target}"
            for target, line_number in imported_modules(use_case)
            if target.startswith("match_analysis.infrastructure")
        ]
        self.assertEqual(violations, [])

    def test_p11b_match_identity_construction_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "BaseballGame(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in MATCH_IDENTITY_CONSTRUCTION_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p11b_constructs_only_the_existing_p1_match_identity(
        self,
    ) -> None:
        domain_contract = MATCH_IDENTITY_CONSTRUCTION_RUNTIME_PATHS[0]
        use_case = MATCH_IDENTITY_CONSTRUCTION_RUNTIME_PATHS[1]
        use_case_tree = ast.parse(
            use_case.read_text(encoding="utf-8"),
            filename=str(use_case),
        )
        direct_constructor_names = [
            node.func.id
            for node in ast.walk(use_case_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ]

        self.assertNotIn(
            "MatchIdentity(",
            domain_contract.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            direct_constructor_names.count("MatchIdentity"),
            1,
        )
        self.assertNotIn("BaseballGame", direct_constructor_names)

    def test_p11b_construction_use_case_does_not_import_infrastructure(
        self,
    ) -> None:
        use_case = MATCH_IDENTITY_CONSTRUCTION_RUNTIME_PATHS[1]
        violations = [
            f"{use_case.relative_to(REPOSITORY_ROOT)}:{line_number} -> {target}"
            for target, line_number in imported_modules(use_case)
            if target.startswith("match_analysis.infrastructure")
        ]
        self.assertEqual(violations, [])

    def test_p12_game_materialization_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "MatchIdentity(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "provider_status_code",
            "provider_detailed_status",
            "Betting-pool",
            "legacy_betting_pool",
            "prediction",
        )
        violations: list[str] = []
        for path in SCHEDULE_BASEBALL_GAME_MATERIALIZATION_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p12_constructs_only_the_existing_baseball_game(self) -> None:
        domain_contract = (
            SCHEDULE_BASEBALL_GAME_MATERIALIZATION_RUNTIME_PATHS[0]
        )
        use_case = SCHEDULE_BASEBALL_GAME_MATERIALIZATION_RUNTIME_PATHS[1]
        use_case_tree = ast.parse(
            use_case.read_text(encoding="utf-8"),
            filename=str(use_case),
        )
        direct_constructor_names = [
            node.func.id
            for node in ast.walk(use_case_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ]

        self.assertNotIn(
            "class BaseballGame",
            domain_contract.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            direct_constructor_names.count("BaseballGame"),
            1,
        )
        self.assertEqual(
            direct_constructor_names.count("MatchIdentity"),
            0,
        )

    def test_p12_use_case_does_not_import_outer_layers(self) -> None:
        use_case = SCHEDULE_BASEBALL_GAME_MATERIALIZATION_RUNTIME_PATHS[1]
        violations = [
            f"{use_case.relative_to(REPOSITORY_ROOT)}:{line_number} -> {target}"
            for target, line_number in imported_modules(use_case)
            if target.startswith(
                (
                    "match_analysis.infrastructure",
                    "match_analysis.interfaces",
                )
            )
        ]
        self.assertEqual(violations, [])

    def test_p13_pregame_eligibility_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "time",
            "urllib",
        }
        forbidden_constructs = (
            "BaseballGame(",
            "MatchIdentity(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "provider_status_code",
            "provider_detailed_status",
            "score",
            "odds",
            "prediction",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in SCHEDULE_PREGAME_ELIGIBILITY_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p13_use_case_does_not_import_outer_layers(self) -> None:
        use_case = SCHEDULE_PREGAME_ELIGIBILITY_RUNTIME_PATHS[1]
        violations = [
            f"{use_case.relative_to(REPOSITORY_ROOT)}:{line_number} -> {target}"
            for target, line_number in imported_modules(use_case)
            if target.startswith(
                (
                    "match_analysis.infrastructure",
                    "match_analysis.interfaces",
                )
            )
        ]
        self.assertEqual(violations, [])

    def test_p84b_runtime_cannot_construct_promoted_domain_objects(
        self,
    ) -> None:
        forbidden_constructors = ("MatchIdentity(", "BaseballGame(")
        violations = [
            f"{path.relative_to(REPOSITORY_ROOT)} -> {constructor}"
            for path in P84B_RUNTIME_PATHS
            for constructor in forbidden_constructors
            if constructor in path.read_text(encoding="utf-8")
        ]

        self.assertEqual(violations, [])

    def test_p83e_source_and_tests_remain_byte_identical(self) -> None:
        actual = {
            relative: sha256(
                (REPOSITORY_ROOT / relative).read_bytes()
            ).hexdigest()
            for relative in P83E_BASELINE_SHA256
        }

        self.assertEqual(actual, P83E_BASELINE_SHA256)

    def test_p84b_source_and_tests_remain_byte_identical(self) -> None:
        actual = {
            relative: sha256(
                (REPOSITORY_ROOT / relative).read_bytes()
            ).hexdigest()
            for relative in P84B_BASELINE_SHA256
        }

        self.assertEqual(actual, P84B_BASELINE_SHA256)

    def test_p3_quarantine_link_source_and_tests_remain_byte_identical(
        self,
    ) -> None:
        actual = {
            relative: sha256(
                (REPOSITORY_ROOT / relative).read_bytes()
            ).hexdigest()
            for relative in P3_QUARANTINE_LINK_BASELINE_SHA256
        }

        self.assertEqual(actual, P3_QUARANTINE_LINK_BASELINE_SHA256)

    def test_quarantine_link_runtime_cannot_construct_promoted_domain_objects(
        self,
    ) -> None:
        forbidden_constructors = ("MatchIdentity(", "BaseballGame(")
        violations = [
            f"{path.relative_to(REPOSITORY_ROOT)} -> {constructor}"
            for path in QUARANTINE_LINK_RUNTIME_PATHS
            for constructor in forbidden_constructors
            if constructor in path.read_text(encoding="utf-8")
        ]

        self.assertEqual(violations, [])

    def test_p14b2_legacy_prediction_quarantine_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "BaseballGame(",
            "MatchIdentity(",
            "PredictionSourceObservation(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in LEGACY_PREDICTION_QUARANTINE_ASSESSMENT_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p14b2_legacy_prediction_quarantine_use_case_does_not_import_infrastructure(
        self,
    ) -> None:
        use_case = LEGACY_PREDICTION_QUARANTINE_ASSESSMENT_RUNTIME_PATHS[1]
        violations = [
            f"{use_case.relative_to(REPOSITORY_ROOT)}:{line_number} -> {target}"
            for target, line_number in imported_modules(use_case)
            if target.startswith(
                ("match_analysis.infrastructure", "match_analysis.interfaces")
            )
        ]
        self.assertEqual(violations, [])

    def test_p14b2_legacy_prediction_quarantine_runtime_cannot_construct_promoted_domain_objects(
        self,
    ) -> None:
        forbidden_constructors = (
            "MatchIdentity(",
            "BaseballGame(",
            "PredictionSourceObservation(",
        )
        violations = [
            f"{path.relative_to(REPOSITORY_ROOT)} -> {constructor}"
            for path in LEGACY_PREDICTION_QUARANTINE_ASSESSMENT_RUNTIME_PATHS
            for constructor in forbidden_constructors
            if constructor in path.read_text(encoding="utf-8")
        ]

        self.assertEqual(violations, [])

    def test_schedule_observation_source_and_tests_remain_byte_identical(
        self,
    ) -> None:
        actual = {
            relative: sha256(
                (REPOSITORY_ROOT / relative).read_bytes()
            ).hexdigest()
            for relative in SCHEDULE_OBSERVATION_BASELINE_SHA256
        }

        self.assertEqual(actual, SCHEDULE_OBSERVATION_BASELINE_SHA256)

    def test_mlb_schedule_payload_adapter_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "BaseballGame(",
            "MatchIdentity(",
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in MLB_SCHEDULE_PAYLOAD_ADAPTER_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_mlb_schedule_payload_adapter_does_not_import_other_infrastructure(
        self,
    ) -> None:
        violations: list[str] = []
        for path in MLB_SCHEDULE_PAYLOAD_ADAPTER_RUNTIME_PATHS:
            for target, line_number in imported_modules(path):
                if target.startswith(
                    "match_analysis.infrastructure.legacy_betting_pool"
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p15a1_prediction_admission_foundation_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in PREDICTION_ADMISSION_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p15a1_prediction_admission_does_not_import_outer_layers(
        self,
    ) -> None:
        violations: list[str] = []
        for path in PREDICTION_ADMISSION_RUNTIME_PATHS:
            for target, line_number in imported_modules(path):
                if target.startswith(
                    (
                        "match_analysis.application",
                        "match_analysis.infrastructure",
                        "match_analysis.interfaces",
                    )
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p15a1_only_prediction_admission_module_constructs_the_observation(
        self,
    ) -> None:
        violations = [
            str(path.relative_to(REPOSITORY_ROOT))
            for path in source_files()
            if path not in PREDICTION_SOURCE_OBSERVATION_AUTHORIZED_CONSTRUCTOR_PATHS
            and "PredictionSourceObservation(" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(violations, [])

    def test_p15b_real_schedule_admission_runtime_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        p15b_paths = (
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "run_prospective_prediction_admission_workflow.py",
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "prospective_prediction_admission_artifacts.py",
        )
        violations: list[str] = []
        for path in p15b_paths:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in (
                "datetime.now(",
                "datetime.utcnow(",
                "time.time(",
                "Betting-pool",
                "legacy_betting_pool",
            ):
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p15b_workflow_does_not_construct_protected_domain_objects(
        self,
    ) -> None:
        p15b_source_files = [
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "run_prospective_prediction_admission_workflow.py",
            PACKAGE_ROOT
            / "application"
            / "use_cases"
            / "prospective_prediction_admission_artifacts.py",
            PACKAGE_ROOT
            / "interfaces"
            / "cli"
            / "prospective_prediction_admission.py",
        ]
        forbidden_constructors = (
            "MatchIdentity(",
            "BaseballGame(",
            "ScheduleBaseballGameMaterialization(",
            "SchedulePregameEligibilityDecision(",
            "PredictionSourceObservation(",
        )
        violations: list[str] = []
        for path in p15b_source_files:
            relative = path.relative_to(REPOSITORY_ROOT)
            source = path.read_text(encoding="utf-8")
            for constructor in forbidden_constructors:
                if constructor in source:
                    violations.append(f"{relative} -> {constructor}")
        self.assertEqual(violations, [])

P15C_SNAPSHOT_RUNTIME_PATHS = (
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "build_admitted_prediction_observation_snapshot.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "admitted_prediction_observation_artifacts.py",
)

P15C_CLI_PATH = (
    PACKAGE_ROOT
    / "interfaces"
    / "cli"
    / "admitted_prediction_observation_snapshot.py"
)


class P15CDependencyRuleTests(unittest.TestCase):
    def test_p15c_snapshot_runtime_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in P15C_SNAPSHOT_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p15c_snapshot_does_not_construct_prediction_source_observation(
        self,
    ) -> None:
        p15c_files = [
            *P15C_SNAPSHOT_RUNTIME_PATHS,
            P15C_CLI_PATH,
        ]
        forbidden_constructors = (
            "PredictionSourceObservation(",
            "MatchIdentity(",
            "BaseballGame(",
        )
        violations: list[str] = []
        for path in p15c_files:
            relative = path.relative_to(REPOSITORY_ROOT)
            source = path.read_text(encoding="utf-8")
            for constructor in forbidden_constructors:
                if constructor in source:
                    violations.append(f"{relative} -> {constructor}")
        self.assertEqual(violations, [])

    def test_p15c_snapshot_does_not_import_p9_p13_construction_use_cases(
        self,
    ) -> None:
        p9_p13_modules = (
            "match_analysis.application.use_cases.project_schedule_identity_candidates",
            "match_analysis.application.use_cases.resolve_schedule_participant_identities",
            "match_analysis.application.use_cases.construct_match_identities",
            "match_analysis.application.use_cases.materialize_schedule_baseball_games",
            "match_analysis.application.use_cases.evaluate_schedule_pregame_eligibility",
        )
        violations: list[str] = []
        for path in [*P15C_SNAPSHOT_RUNTIME_PATHS, P15C_CLI_PATH]:
            for target, line_number in imported_modules(path):
                if target in p9_p13_modules:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p15c_snapshot_does_not_call_admission_again(
        self,
    ) -> None:
        forbidden_imports = (
            "match_analysis.application.use_cases.run_prospective_prediction_admission_workflow",
            "match_analysis.baseball.domain.prediction_admission",
        )
        violations: list[str] = []
        for path in [*P15C_SNAPSHOT_RUNTIME_PATHS, P15C_CLI_PATH]:
            for target, line_number in imported_modules(path):
                if target in forbidden_imports:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in (
                "admit_prospective_prediction(",
                "run_prospective_prediction_admission_workflow(",
            ):
                if construct in source:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_legacy_p83e_p14_p14b2_cannot_import_p15c(
        self,
    ) -> None:
        legacy_paths = [
            *LEGACY_PREDICTION_QUARANTINE_ASSESSMENT_RUNTIME_PATHS,
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p83e_jsonl.py",
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p84b_schedule_jsonl.py",
        ]
        p15c_targets = (
            "build_admitted_prediction_observation_snapshot",
            "admitted_prediction_observation_artifacts",
            "admitted_prediction_observation_snapshot",
        )
        violations: list[str] = []
        for path in legacy_paths:
            if not path.exists():
                continue
            for target, line_number in imported_modules(path):
                for p15c_name in p15c_targets:
                    if p15c_name in target:
                        relative = path.relative_to(REPOSITORY_ROOT)
                        violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p15c_application_does_not_import_infrastructure_or_interfaces(
        self,
    ) -> None:
        violations: list[str] = []
        for path in P15C_SNAPSHOT_RUNTIME_PATHS:
            for target, line_number in imported_modules(path):
                if target.startswith(
                    (
                        "match_analysis.infrastructure",
                        "match_analysis.interfaces",
                    )
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])


P16A_ATTACHMENT_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "final_result_observation.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "attach_final_results_to_admitted_predictions.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "final_result_attachment_artifacts.py",
)

P16A_CLI_PATH = (
    PACKAGE_ROOT / "interfaces" / "cli" / "final_result_attachment.py"
)


class P16ADependencyRuleTests(unittest.TestCase):
    def test_p16a_attachment_runtime_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in P16A_ATTACHMENT_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p16a_attachment_does_not_construct_protected_domain_objects(
        self,
    ) -> None:
        p16a_files = [*P16A_ATTACHMENT_RUNTIME_PATHS, P16A_CLI_PATH]
        forbidden_constructors = (
            "PredictionSourceObservation(",
            "MatchIdentity(",
            "BaseballGame(",
        )
        violations: list[str] = []
        for path in p16a_files:
            relative = path.relative_to(REPOSITORY_ROOT)
            source = path.read_text(encoding="utf-8")
            for constructor in forbidden_constructors:
                if constructor in source:
                    violations.append(f"{relative} -> {constructor}")
        self.assertEqual(violations, [])

    def test_p16a_final_result_observation_authorized_constructor_only(
        self,
    ) -> None:
        authorized = {
            PACKAGE_ROOT / "baseball" / "domain" / "final_result_observation.py",
        }
        violations: list[str] = []
        for path in source_files():
            if path not in authorized:
                source = path.read_text(encoding="utf-8")
                if "FinalResultObservation(" in source:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative} -> FinalResultObservation(")
        self.assertEqual(violations, [])

    def test_p16a_attachment_does_not_import_p9_p13_construction_use_cases(
        self,
    ) -> None:
        p9_p13_modules = (
            "match_analysis.application.use_cases.project_schedule_identity_candidates",
            "match_analysis.application.use_cases.resolve_schedule_participant_identities",
            "match_analysis.application.use_cases.construct_match_identities",
            "match_analysis.application.use_cases.materialize_schedule_baseball_games",
            "match_analysis.application.use_cases.evaluate_schedule_pregame_eligibility",
        )
        violations: list[str] = []
        for path in [*P16A_ATTACHMENT_RUNTIME_PATHS, P16A_CLI_PATH]:
            for target, line_number in imported_modules(path):
                if target in p9_p13_modules:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p16a_attachment_does_not_call_prediction_admission(
        self,
    ) -> None:
        forbidden_imports = (
            "match_analysis.application.use_cases.run_prospective_prediction_admission_workflow",
            "match_analysis.baseball.domain.prediction_admission",
        )
        violations: list[str] = []
        for path in [*P16A_ATTACHMENT_RUNTIME_PATHS, P16A_CLI_PATH]:
            for target, line_number in imported_modules(path):
                if target in forbidden_imports:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in (
                "admit_prospective_prediction(",
                "run_prospective_prediction_admission_workflow(",
            ):
                if construct in source:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_legacy_code_cannot_import_p16a(
        self,
    ) -> None:
        legacy_paths = [
            *LEGACY_PREDICTION_QUARANTINE_ASSESSMENT_RUNTIME_PATHS,
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p83e_jsonl.py",
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p84b_schedule_jsonl.py",
        ]
        p16a_targets = (
            "final_result_observation",
            "attach_final_results_to_admitted_predictions",
            "final_result_attachment_artifacts",
            "final_result_attachment",
        )
        violations: list[str] = []
        for path in legacy_paths:
            if not path.exists():
                continue
            for target, line_number in imported_modules(path):
                for p16a_name in p16a_targets:
                    if p16a_name in target:
                        relative = path.relative_to(REPOSITORY_ROOT)
                        violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p16a_application_does_not_import_infrastructure_or_interfaces(
        self,
    ) -> None:
        violations: list[str] = []
        for path in P16A_ATTACHMENT_RUNTIME_PATHS:
            for target, line_number in imported_modules(path):
                if target.startswith(
                    (
                        "match_analysis.infrastructure",
                        "match_analysis.interfaces",
                    )
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])


P16B_EVALUATION_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "prediction_evaluation.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "build_prediction_evaluation_scorecard.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "prediction_evaluation_artifacts.py",
)

P16B_CLI_PATH = (
    PACKAGE_ROOT / "interfaces" / "cli" / "prediction_evaluation_scorecard.py"
)


class P16BDependencyRuleTests(unittest.TestCase):
    def test_p16b_evaluation_runtime_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in P16B_EVALUATION_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p16b_evaluation_does_not_construct_protected_domain_objects(
        self,
    ) -> None:
        p16b_files = [*P16B_EVALUATION_RUNTIME_PATHS, P16B_CLI_PATH]
        forbidden_constructors = (
            "PredictionSourceObservation(",
            "FinalResultObservation(",
            "MatchIdentity(",
            "BaseballGame(",
        )
        violations: list[str] = []
        for path in p16b_files:
            relative = path.relative_to(REPOSITORY_ROOT)
            source = path.read_text(encoding="utf-8")
            for constructor in forbidden_constructors:
                if constructor in source:
                    violations.append(f"{relative} -> {constructor}")
        self.assertEqual(violations, [])

    def test_p16b_evaluation_does_not_import_p9_p13_construction_use_cases(
        self,
    ) -> None:
        p9_p13_modules = (
            "match_analysis.application.use_cases.project_schedule_identity_candidates",
            "match_analysis.application.use_cases.resolve_schedule_participant_identities",
            "match_analysis.application.use_cases.construct_match_identities",
            "match_analysis.application.use_cases.materialize_schedule_baseball_games",
            "match_analysis.application.use_cases.evaluate_schedule_pregame_eligibility",
        )
        violations: list[str] = []
        for path in [*P16B_EVALUATION_RUNTIME_PATHS, P16B_CLI_PATH]:
            for target, line_number in imported_modules(path):
                if target in p9_p13_modules:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p16b_evaluation_does_not_call_prediction_admission_or_attachment(
        self,
    ) -> None:
        forbidden_imports = (
            "match_analysis.application.use_cases.run_prospective_prediction_admission_workflow",
            "match_analysis.baseball.domain.prediction_admission",
            "match_analysis.application.use_cases.attach_final_results_to_admitted_predictions",
        )
        violations: list[str] = []
        for path in [*P16B_EVALUATION_RUNTIME_PATHS, P16B_CLI_PATH]:
            for target, line_number in imported_modules(path):
                if target in forbidden_imports:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in (
                "admit_prospective_prediction(",
                "run_prospective_prediction_admission_workflow(",
                "attach_final_results_to_admitted_predictions(",
            ):
                if construct in source:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_legacy_code_cannot_import_p16b(
        self,
    ) -> None:
        legacy_paths = [
            *LEGACY_PREDICTION_QUARANTINE_ASSESSMENT_RUNTIME_PATHS,
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p83e_jsonl.py",
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p84b_schedule_jsonl.py",
        ]
        p16b_targets = (
            "prediction_evaluation",
            "build_prediction_evaluation_scorecard",
            "prediction_evaluation_artifacts",
        )
        violations: list[str] = []
        for path in legacy_paths:
            if not path.exists():
                continue
            for target, line_number in imported_modules(path):
                for p16b_name in p16b_targets:
                    if p16b_name in target:
                        relative = path.relative_to(REPOSITORY_ROOT)
                        violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p16b_application_does_not_import_infrastructure_or_interfaces(
        self,
    ) -> None:
        violations: list[str] = []
        for path in P16B_EVALUATION_RUNTIME_PATHS[1:]:  # use cases
            for target, line_number in imported_modules(path):
                if target.startswith(
                    (
                        "match_analysis.infrastructure",
                        "match_analysis.interfaces",
                    )
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])


P17A_FEEDBACK_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "prediction_feedback.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "build_prediction_feedback_ledger.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "prediction_feedback_artifacts.py",
)

P17A_CLI_PATH = (
    PACKAGE_ROOT / "interfaces" / "cli" / "prediction_feedback_ledger.py"
)

P18A_REPLAY_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "result_only_paper_decision.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "build_result_only_paper_decision_replay.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "result_only_paper_decision_artifacts.py",
)

P18A_CLI_PATH = (
    PACKAGE_ROOT / "interfaces" / "cli" / "result_only_paper_decision_replay.py"
)

P21A_LEARNING_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "prediction_learning_eligibility.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "assess_prediction_learning_candidates.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "prediction_learning_candidate_artifacts.py",
)

P21A_CLI_PATH = (
    PACKAGE_ROOT / "interfaces" / "cli" / "prediction_learning_candidate_gate.py"
)

P22A_DATASET_RUNTIME_PATHS = (
    PACKAGE_ROOT / "baseball" / "domain" / "supervised_training_example.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "materialize_moneyline_training_dataset.py",
    PACKAGE_ROOT
    / "application"
    / "use_cases"
    / "moneyline_training_dataset_artifacts.py",
)

P22A_CLI_PATH = (
    PACKAGE_ROOT / "interfaces" / "cli" / "materialize_moneyline_training_dataset.py"
)


class P17ADependencyRuleTests(unittest.TestCase):
    def test_p17a_feedback_runtime_has_no_forbidden_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
        )
        violations: list[str] = []
        for path in P17A_FEEDBACK_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p17a_feedback_does_not_construct_protected_domain_objects(
        self,
    ) -> None:
        p17a_files = [*P17A_FEEDBACK_RUNTIME_PATHS, P17A_CLI_PATH]
        forbidden_constructors = (
            "PredictionSourceObservation(",
            "FinalResultObservation(",
            "MatchIdentity(",
            "BaseballGame(",
        )
        violations: list[str] = []
        for path in p17a_files:
            relative = path.relative_to(REPOSITORY_ROOT)
            source = path.read_text(encoding="utf-8")
            for constructor in forbidden_constructors:
                if constructor in source:
                    violations.append(f"{relative} -> {constructor}")
        self.assertEqual(violations, [])

    def test_p17a_feedback_does_not_import_p9_p13_construction_use_cases(
        self,
    ) -> None:
        p9_p13_modules = (
            "match_analysis.application.use_cases.project_schedule_identity_candidates",
            "match_analysis.application.use_cases.resolve_schedule_participant_identities",
            "match_analysis.application.use_cases.construct_match_identities",
            "match_analysis.application.use_cases.materialize_schedule_baseball_games",
            "match_analysis.application.use_cases.evaluate_schedule_pregame_eligibility",
        )
        violations: list[str] = []
        for path in [*P17A_FEEDBACK_RUNTIME_PATHS, P17A_CLI_PATH]:
            for target, line_number in imported_modules(path):
                if target in p9_p13_modules:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p17a_feedback_does_not_call_prediction_admission_or_attachment(
        self,
    ) -> None:
        forbidden_imports = (
            "match_analysis.application.use_cases.run_prospective_prediction_admission_workflow",
            "match_analysis.baseball.domain.prediction_admission",
            "match_analysis.application.use_cases.attach_final_results_to_admitted_predictions",
        )
        violations: list[str] = []
        for path in [*P17A_FEEDBACK_RUNTIME_PATHS, P17A_CLI_PATH]:
            for target, line_number in imported_modules(path):
                if target in forbidden_imports:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in (
                "admit_prospective_prediction(",
                "run_prospective_prediction_admission_workflow(",
                "attach_final_results_to_admitted_predictions(",
            ):
                if construct in source:
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_legacy_code_cannot_import_p17a(
        self,
    ) -> None:
        legacy_paths = [
            *LEGACY_PREDICTION_QUARANTINE_ASSESSMENT_RUNTIME_PATHS,
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p83e_jsonl.py",
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p84b_schedule_jsonl.py",
        ]
        p17a_targets = (
            "prediction_feedback",
            "build_prediction_feedback_ledger",
            "prediction_feedback_artifacts",
        )
        violations: list[str] = []
        for path in legacy_paths:
            if not path.exists():
                continue
            for target, line_number in imported_modules(path):
                for p17a_name in p17a_targets:
                    if p17a_name in target:
                        relative = path.relative_to(REPOSITORY_ROOT)
                        violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p17a_application_does_not_import_infrastructure_or_interfaces(
        self,
    ) -> None:
        violations: list[str] = []
        for path in P17A_FEEDBACK_RUNTIME_PATHS[1:]:  # use cases
            for target, line_number in imported_modules(path):
                if target.startswith(
                    (
                        "match_analysis.infrastructure",
                        "match_analysis.interfaces",
                    )
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])


class P18ADependencyRuleTests(unittest.TestCase):
    def test_p18a_replay_runtime_has_no_external_or_profitability_capabilities(
        self,
    ) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "Betting-pool",
            "legacy_betting_pool",
            "calculate_payout(",
            "calculate_profit(",
            "calculate_roi(",
            "calculate_ev(",
            "kelly_fraction(",
        )
        violations: list[str] = []
        for path in P18A_REPLAY_RUNTIME_PATHS:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p18a_replay_does_not_call_upstream_outcome_or_feedback_workflows(
        self,
    ) -> None:
        forbidden_imports = (
            "match_analysis.application.use_cases.attach_final_results_to_admitted_predictions",
            "match_analysis.application.use_cases.build_prediction_evaluation_scorecard",
            "match_analysis.application.use_cases.build_prediction_feedback_ledger",
            "match_analysis.interfaces",
            "match_analysis.infrastructure",
        )
        forbidden_constructs = (
            "attach_final_results_to_admitted_predictions(",
            "build_prediction_evaluation_scorecard(",
            "build_prediction_feedback_ledger(",
        )
        violations: list[str] = []
        for path in [*P18A_REPLAY_RUNTIME_PATHS, P18A_CLI_PATH]:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.startswith(forbidden_imports):
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_legacy_code_cannot_import_p18a(self) -> None:
        legacy_paths = [
            *LEGACY_PREDICTION_QUARANTINE_ASSESSMENT_RUNTIME_PATHS,
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p83e_jsonl.py",
            PACKAGE_ROOT / "infrastructure" / "legacy_betting_pool" / "p84b_schedule_jsonl.py",
        ]
        p18a_targets = (
            "result_only_paper_decision",
            "build_result_only_paper_decision_replay",
        )
        violations: list[str] = []
        for path in legacy_paths:
            if not path.exists():
                continue
            for target, line_number in imported_modules(path):
                if any(name in target for name in p18a_targets):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

    def test_p18a_application_does_not_import_outer_layers(self) -> None:
        violations: list[str] = []
        for path in P18A_REPLAY_RUNTIME_PATHS[1:]:
            for target, line_number in imported_modules(path):
                if target.startswith(
                    (
                        "match_analysis.infrastructure",
                        "match_analysis.interfaces",
                    )
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])


class P21ADependencyRuleTests(unittest.TestCase):
    def test_p21a_runtime_has_no_external_or_training_capabilities(self) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "os",
            "requests",
            "shutil",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_imports = (
            "match_analysis.application.use_cases.build_prediction_feedback_ledger",
            "match_analysis.infrastructure",
            "match_analysis.interfaces",
        )
        violations: list[str] = []
        for path in [*P21A_LEARNING_RUNTIME_PATHS, P21A_CLI_PATH]:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
                if target.startswith(forbidden_imports):
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in (
                "model.fit(",
                "retrain(",
                "promote_model(",
                "calculate_roi(",
                "calculate_ev(",
                "kelly_fraction(",
            ):
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])


    def test_p21a_application_does_not_import_outer_layers(self) -> None:
        violations: list[str] = []
        for path in P21A_LEARNING_RUNTIME_PATHS[1:]:
            for target, line_number in imported_modules(path):
                if target.startswith(
                    (
                        "match_analysis.infrastructure",
                        "match_analysis.interfaces",
                    )
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])


class P22ADependencyRuleTests(unittest.TestCase):
    def test_p22a_runtime_has_no_external_or_training_capabilities(self) -> None:
        forbidden_import_roots = {
            "aiohttp",
            "http",
            "numpy",
            "os",
            "pandas",
            "requests",
            "shutil",
            "sklearn",
            "socket",
            "sqlite3",
            "tempfile",
            "urllib",
        }
        forbidden_constructs = (
            "datetime.now(",
            "datetime.utcnow(",
            "time.time(",
            "model.fit(",
            "retrain(",
            "promote_model(",
            "calculate_roi(",
            "calculate_ev(",
            "kelly_fraction(",
            "P22A_STOP_MATCHANALYSIS_P22A_SCOPE_EXPANSION_REQUIRED",
        )
        violations: list[str] = []
        for path in [*P22A_DATASET_RUNTIME_PATHS, P22A_CLI_PATH]:
            relative = path.relative_to(REPOSITORY_ROOT)
            for target, line_number in imported_modules(path):
                if target.split(".")[0] in forbidden_import_roots:
                    violations.append(f"{relative}:{line_number} -> {target}")
            source = path.read_text(encoding="utf-8")
            for construct in forbidden_constructs:
                if construct in source:
                    violations.append(f"{relative} -> {construct}")
        self.assertEqual(violations, [])

    def test_p22a_application_does_not_import_outer_layers(self) -> None:
        violations: list[str] = []
        for path in P22A_DATASET_RUNTIME_PATHS[1:]:
            for target, line_number in imported_modules(path):
                if target.startswith(
                    (
                        "match_analysis.infrastructure",
                        "match_analysis.interfaces",
                    )
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number} -> {target}")
        self.assertEqual(violations, [])

if __name__ == "__main__":
    unittest.main()
