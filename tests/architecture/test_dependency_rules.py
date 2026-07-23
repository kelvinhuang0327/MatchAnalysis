"""Executable modular-monolith dependency rules."""

import ast
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
    "application/use_cases/__init__.py",
    "application/use_cases/import_legacy_prediction_snapshot.py",
    "baseball/__init__.py",
    "baseball/domain/__init__.py",
    "baseball/domain/game.py",
    "baseball/domain/prediction.py",
    "core/__init__.py",
    "core/identity.py",
    "core/provenance.py",
    "core/time.py",
    "infrastructure/__init__.py",
    "infrastructure/legacy_betting_pool/__init__.py",
    "infrastructure/legacy_betting_pool/p83e_jsonl.py",
    "interfaces/__init__.py",
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


if __name__ == "__main__":
    unittest.main()
