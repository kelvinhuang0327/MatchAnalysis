"""Unit tests for the immutable P19A Moneyline model artifact."""

from decimal import Decimal
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.moneyline_inference_artifacts import (
    load_moneyline_feature_snapshots,
    load_moneyline_model_artifact,
)


FIXTURE_DIR = REPOSITORY_ROOT / "data" / "fixtures" / "p19a_moneyline_inference"


class MoneylineModelArtifactTests(unittest.TestCase):
    def test_fixture_has_exact_legacy_provenance_and_stable_fingerprint(self) -> None:
        artifact = load_moneyline_model_artifact(FIXTURE_DIR / "model_artifact.json")
        self.assertEqual(
            artifact.legacy_source_commit,
            "03b2fcf4de1a13ee9929afcef803d61955c9f41b",
        )
        self.assertEqual(
            artifact.legacy_source_tree,
            "56a849bc68234db63da7a38f1643fa664217c5d0",
        )
        self.assertEqual(
            artifact.legacy_source_paths,
            (
                "scripts/run_mlb_walk_forward_ml_candidate.py",
                "wbc_backend/prediction/mlb_independent_feature_builder.py",
                "wbc_backend/prediction/mlb_independent_features.py",
                "wbc_backend/prediction/mlb_ml_feature_matrix.py",
                "wbc_backend/prediction/mlb_walk_forward_model.py",
                "outputs/predictions/PAPER/2026-05-11/p13_ml/ml_model_metadata.json",
                "outputs/predictions/PAPER/2026-05-11/p13_ml/ml_feature_matrix.csv",
                "outputs/predictions/PAPER/2026-05-11/p13_ml/ml_walk_forward_predictions.jsonl",
            ),
        )
        self.assertEqual(artifact.artifact_kind, "bounded_deterministic_fixture")
        self.assertEqual(
            artifact.fixture_basis_id,
            "ffca78865db53ffaebc17110c39a604f484a353d38858c64787ac8a81c7664c9",
        )
        self.assertEqual(
            artifact.fixture_expected_home_probability,
            Decimal("0.469229"),
        )
        self.assertEqual(artifact.fingerprint(), load_moneyline_model_artifact(
            FIXTURE_DIR / "model_artifact.json"
        ).fingerprint())

    def test_probability_is_deterministic_and_complementary(self) -> None:
        artifact = load_moneyline_model_artifact(FIXTURE_DIR / "model_artifact.json")
        snapshot = load_moneyline_feature_snapshots(
            FIXTURE_DIR / "feature_snapshots.jsonl"
        )[0]
        first = artifact.predict_home_probability(snapshot)
        second = artifact.predict_home_probability(snapshot)
        self.assertEqual(first, second)
        self.assertGreater(first, Decimal("0"))
        self.assertLess(first, Decimal("1"))

    def test_fixture_basis_is_explicit_and_matches_projection(self) -> None:
        projection = json.loads(
            (FIXTURE_DIR / "legacy_parity_fixture.json").read_text(encoding="utf-8")
        )
        artifact = load_moneyline_model_artifact(FIXTURE_DIR / "model_artifact.json")
        self.assertEqual(projection["game_id"], "2025-06-01_TEX_STL")
        self.assertEqual(
            projection["legacy_source_commit"], artifact.legacy_source_commit
        )
        self.assertEqual(
            projection["legacy_source_tree"], artifact.legacy_source_tree
        )
        self.assertEqual(
            projection["legacy_model_semantics"]["mean_abs_coef"],
            "0.141479",
        )
        self.assertEqual(
            artifact.fixture_expected_probability_tolerance,
            Decimal(projection["expected_probability_tolerance"]),
        )


if __name__ == "__main__":
    unittest.main()
