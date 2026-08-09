"""Bounded characterization of the committed P13 logistic semantics."""

from decimal import Decimal
from math import exp
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.generate_moneyline_predictions import (
    generate_moneyline_predictions,
)
from match_analysis.application.use_cases.moneyline_inference_artifacts import (
    load_moneyline_feature_snapshots,
    load_moneyline_model_artifact,
)


FIXTURE_DIR = REPOSITORY_ROOT / "data" / "fixtures" / "p19a_moneyline_inference"


class LegacyMoneylineParityTests(unittest.TestCase):
    def test_bounded_fixture_matches_p13_standardized_logistic_semantics(self) -> None:
        snapshots = load_moneyline_feature_snapshots(
            FIXTURE_DIR / "feature_snapshots.jsonl"
        )
        artifact = load_moneyline_model_artifact(FIXTURE_DIR / "model_artifact.json")
        result = generate_moneyline_predictions(
            snapshots,
            artifact,
            prediction_generated_at_utc="2026-04-05T10:01:00Z",
            response_received_at_utc="2026-04-05T10:01:01Z",
            ingested_at_utc="2026-04-05T10:01:02Z",
        )
        snapshot = snapshots[0]
        standardized = [
            (float(value) - float(mean)) / float(std)
            for value, mean, std in zip(
                snapshot.feature_vector(),
                artifact.scaler_means,
                artifact.scaler_stds,
                strict=True,
            )
        ]
        logit = float(artifact.intercept) + sum(
            float(coefficient) * value
            for coefficient, value in zip(
                artifact.coefficients,
                standardized,
                strict=True,
            )
        )
        expected = 1.0 / (1.0 + exp(-logit))
        self.assertAlmostEqual(
            float(result.candidates[0].model_probability),
            expected,
            places=12,
        )
        self.assertEqual(
            result.candidates[0].model_id,
            "p13_walk_forward_logistic_v1_fixture",
        )


if __name__ == "__main__":
    unittest.main()
