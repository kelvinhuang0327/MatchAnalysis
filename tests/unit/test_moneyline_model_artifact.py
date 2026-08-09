"""Unit tests for the immutable P19A Moneyline model artifact."""

from decimal import Decimal
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
            artifact.legacy_source_paths,
            (
                "scripts/run_mlb_walk_forward_ml_candidate.py",
                "wbc_backend/prediction/mlb_independent_feature_builder.py",
                "wbc_backend/prediction/mlb_independent_features.py",
                "wbc_backend/prediction/mlb_ml_feature_matrix.py",
                "wbc_backend/prediction/mlb_walk_forward_model.py",
            ),
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


if __name__ == "__main__":
    unittest.main()
