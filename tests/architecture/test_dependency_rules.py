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
    "application/use_cases/build_schedule_observation_revision_chains.py",
    "application/use_cases/capture_schedule_observation.py",
    "application/use_cases/construct_match_identities.py",
    "application/use_cases/import_legacy_prediction_snapshot.py",
    "application/use_cases/import_legacy_schedule_snapshot.py",
    "application/use_cases/link_legacy_quarantine_snapshots.py",
    "application/use_cases/materialize_schedule_baseball_games.py",
    "application/use_cases/project_schedule_identity_candidates.py",
    "application/use_cases/resolve_schedule_participant_identities.py",
    "application/use_cases/select_schedule_observations_as_of.py",
    "baseball/__init__.py",
    "baseball/domain/__init__.py",
    "baseball/domain/game.py",
    "baseball/domain/match_identity_authority.py",
    "baseball/domain/prediction.py",
    "baseball/domain/participant_identity_resolution.py",
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
    "infrastructure/__init__.py",
    "infrastructure/legacy_betting_pool/__init__.py",
    "infrastructure/legacy_betting_pool/p83e_jsonl.py",
    "infrastructure/legacy_betting_pool/p84b_schedule_jsonl.py",
    "infrastructure/mlb_schedule/__init__.py",
    "infrastructure/mlb_schedule/explicit_payload_source.py",
    "interfaces/__init__.py",
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

P83E_BASELINE_SHA256 = {
    "src/match_analysis/baseball/domain/prediction.py": (
        "6b41afc68bbe58fac48f68578edab353a64d0a33760f1811337ebb9d9bcb3735"
    ),
    "src/match_analysis/application/ports/legacy_prediction_source.py": (
        "d1a707db18d1df6aa4b2445e254685988a274e51b9636215c070dae076957d77"
    ),
    (
        "src/match_analysis/application/use_cases/"
        "import_legacy_prediction_snapshot.py"
    ): "18bf6cd1134b21ff523528409dee63494d0d90fc24e6e52af4f246fa955c07af",
    (
        "src/match_analysis/infrastructure/legacy_betting_pool/"
        "p83e_jsonl.py"
    ): "ac0a39a1132f4e9276811f051ce1bea97d041b2fa44db162ceebe9e372f35c04",
    "tests/unit/test_prediction_contracts.py": (
        "66da8fb31b3a5c7aeda1999e35bb79aea0179de76445b8a38c27fb43f9b20525"
    ),
    "tests/characterization/test_p83e_snapshot_adapter.py": (
        "3e810b61d78744496dc778a1d5e66e6b375c1cd2294269bd8398e3346fd3e9b3"
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
        self.assert_layer_excludes(
            sorted((PACKAGE_ROOT / "application").rglob("*.py")),
            (
                "match_analysis.infrastructure",
                "match_analysis.interfaces",
            ),
        )

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
            "cli",
            "database",
            "db",
            "provider",
            "scheduler",
            "scripts",
        }
        violations = [
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in source_files()
            if forbidden_path_parts.intersection(
                path.relative_to(PACKAGE_ROOT).parts
            )
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


if __name__ == "__main__":
    unittest.main()
