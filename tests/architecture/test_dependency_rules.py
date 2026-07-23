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
    "application/use_cases/__init__.py",
    "application/use_cases/import_legacy_prediction_snapshot.py",
    "application/use_cases/import_legacy_schedule_snapshot.py",
    "application/use_cases/link_legacy_quarantine_snapshots.py",
    "baseball/__init__.py",
    "baseball/domain/__init__.py",
    "baseball/domain/game.py",
    "baseball/domain/prediction.py",
    "baseball/domain/quarantine_link.py",
    "baseball/domain/schedule.py",
    "core/__init__.py",
    "core/identity.py",
    "core/provenance.py",
    "core/time.py",
    "infrastructure/__init__.py",
    "infrastructure/legacy_betting_pool/__init__.py",
    "infrastructure/legacy_betting_pool/p83e_jsonl.py",
    "infrastructure/legacy_betting_pool/p84b_schedule_jsonl.py",
    "interfaces/__init__.py",
}

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


if __name__ == "__main__":
    unittest.main()
