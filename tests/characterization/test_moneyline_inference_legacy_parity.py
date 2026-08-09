"""Bounded characterization of the committed P13 logistic semantics."""

from decimal import Decimal
import json
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
            prediction_generated_at_utc="2025-06-01T00:01:00Z",
            response_received_at_utc="2025-06-01T00:01:01Z",
            ingested_at_utc="2025-06-01T00:01:02Z",
        )
        parity = json.loads(
            (FIXTURE_DIR / "legacy_parity_fixture.json").read_text(encoding="utf-8")
        )
        snapshot = snapshots[0]
        expected = Decimal(parity["expected_home_probability"])
        self.assertEqual(snapshot.identity.canonical_game_id, parity["game_id"])
        self.assertEqual(
            tuple(str(value) for value in snapshot.feature_vector()),
            tuple(
                parity["feature_values"][name]
                for name in (
                    "indep_recent_win_rate_delta",
                    "indep_starter_era_delta",
                )
            ),
        )
        self.assertEqual(
            artifact.fixture_basis_id,
            parity["basis_fingerprint"],
        )
        self.assertEqual(
            sum(abs(value) for value in artifact.coefficients)
            / Decimal(len(artifact.coefficients)),
            Decimal(parity["legacy_model_semantics"]["mean_abs_coef"]),
        )
        self.assertAlmostEqual(
            float(result.candidates[0].model_probability),
            float(expected),
            places=6,
        )
        self.assertEqual(
            result.candidates[0].model_id,
            "p13_walk_forward_logistic_v1_fixture",
        )


if __name__ == "__main__":
    unittest.main()
