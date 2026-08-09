"""Unit tests for the immutable P22A training-example contract."""

from dataclasses import replace
from decimal import Decimal
import unittest
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.supervised_training_example import (  # noqa: E402
    CandidateLineage,
    FeatureLineage,
    SupervisedTrainingExample,
    compute_training_example_id,
)


class SupervisedTrainingExampleContractTests(unittest.TestCase):
    def build_example(self) -> SupervisedTrainingExample:
        feature_names = ("recent_win_rate_delta", "starter_era_delta")
        feature_values = (Decimal("0.12"), Decimal("-0.45"))
        feature_lineage = tuple(
            FeatureLineage(
                field_name=name,
                value=value,
                source_id=f"source:{name}",
                source_kind="committed_p20a_feature_row",
                observed_as_of_utc="2025-07-01T00:00:00Z",
                source_fingerprint="a" * 64 if index == 0 else "b" * 64,
            )
            for index, (name, value) in enumerate(zip(feature_names, feature_values))
        )
        training_example_id = compute_training_example_id(
            schema_version="p22a.supervised_training_example.v1",
            provider_namespace="MLB_STATS_API",
            provider_game_id="2025-07-01_TOR_NYY",
            game_number=1,
            scheduled_start_utc="2025-07-01T12:00:00Z",
            feature_as_of_utc="2025-07-01T00:00:00Z",
            fold_id="wf_002",
            fold_fingerprint="c" * 64,
            model_id="p13_walk_forward_logistic_v1_wf_002",
            model_fingerprint="d" * 64,
            feature_snapshot_fingerprint="e" * 64,
            source_schedule_observation_id="f" * 64,
        )
        return SupervisedTrainingExample(
            training_example_id=training_example_id,
            provider_namespace="MLB_STATS_API",
            provider_game_id="2025-07-01_TOR_NYY",
            game_number=1,
            home_participant="Toronto Blue Jays",
            away_participant="New York Yankees",
            scheduled_start_utc="2025-07-01T12:00:00Z",
            feature_as_of_utc="2025-07-01T00:00:00Z",
            fold_id="wf_002",
            fold_fingerprint="c" * 64,
            model_id="p13_walk_forward_logistic_v1_wf_002",
            model_fingerprint="d" * 64,
            model_artifact_fingerprint="1" * 64,
            feature_snapshot_id=f"p19a:{'e' * 64}",
            feature_snapshot_fingerprint="e" * 64,
            feature_snapshot_schema_version="p19a.moneyline_feature_snapshot.v1",
            source_schedule_observation_id="f" * 64,
            feature_names=feature_names,
            feature_values=feature_values,
            feature_lineage=feature_lineage,
            target_home_win=0,
            historical_result_source_id="2" * 64,
            historical_result_observation_id="3" * 64,
            historical_result_observed_at_utc="2025-07-02T00:00:00Z",
            historical_home_score=2,
            historical_away_score=3,
            historical_result_row_fingerprint="4" * 64,
            source_candidates=(
                CandidateLineage(
                    candidate_id="5" * 64,
                    candidate_row_fingerprint="6" * 64,
                    selection="HOME",
                    source_snapshot_row_fingerprint="7" * 64,
                    source_evaluation_row_fingerprint="8" * 64,
                ),
            ),
        )

    def test_round_trip_preserves_canonical_projection(self) -> None:
        example = self.build_example()
        restored = SupervisedTrainingExample.from_projection(example.to_projection())
        self.assertEqual(restored, example)
        self.assertEqual(restored.canonical_bytes(), example.canonical_bytes())

    def test_identity_excludes_mutable_outcome_target(self) -> None:
        example = self.build_example()
        changed_outcome = replace(
            example,
            target_home_win=1,
            historical_home_score=4,
            historical_away_score=1,
        )
        self.assertEqual(changed_outcome.training_example_id, example.training_example_id)
        self.assertEqual(changed_outcome.feature_values, example.feature_values)
        self.assertNotEqual(changed_outcome.target_home_win, example.target_home_win)

    def test_target_must_match_historical_result_semantics(self) -> None:
        example = self.build_example()
        with self.assertRaisesRegex(ValueError, "target_home_win"):
            replace(example, target_home_win=1)


if __name__ == "__main__":
    unittest.main()
