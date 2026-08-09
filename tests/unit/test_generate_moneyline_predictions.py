"""Unit tests for P19A inference and P15C compatibility."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
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
from match_analysis.application.use_cases.evaluate_schedule_pregame_eligibility import (
    evaluate_schedule_pregame_eligibility,
)
from match_analysis.baseball.domain.prediction_admission import (
    ADMITTED,
    ScheduleCandidateProjection,
)
from match_analysis.baseball.domain.prediction_source_observation import (
    PredictionSourceObservation,
)
from tests.unit.test_construct_match_identities import make_resolved_candidate
from tests.unit.test_pregame_eligibility_contracts import one_game_materialization_set


FIXTURE_DIR = REPOSITORY_ROOT / "data" / "fixtures" / "p19a_moneyline_inference"


class GenerateMoneylinePredictionsTests(unittest.TestCase):
    def test_same_snapshot_and_artifact_replay_identically(self) -> None:
        snapshots = load_moneyline_feature_snapshots(
            FIXTURE_DIR / "feature_snapshots.jsonl"
        )
        artifact = load_moneyline_model_artifact(FIXTURE_DIR / "model_artifact.json")
        kwargs = {
            "prediction_generated_at_utc": "2026-04-05T10:01:00Z",
            "response_received_at_utc": "2026-04-05T10:01:01Z",
            "ingested_at_utc": "2026-04-05T10:01:02Z",
        }
        first = generate_moneyline_predictions(snapshots, artifact, **kwargs)
        second = generate_moneyline_predictions(snapshots, artifact, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first.candidates), 2)
        self.assertEqual(
            first.candidates[0].model_probability
            + first.candidates[1].model_probability,
            Decimal("1"),
        )
        self.assertEqual(first.candidates[0].market_id, "moneyline")

    def test_generated_candidates_are_accepted_by_p15c(self) -> None:
        materializations = one_game_materialization_set()
        eligibility = evaluate_schedule_pregame_eligibility(materializations)
        materialization = materializations.game_materializations[0]
        scheduled_start = materialization.baseball_game.scheduled_start.value
        resolved = make_resolved_candidate()
        snapshot = load_moneyline_feature_snapshots(
            FIXTURE_DIR / "feature_snapshots.jsonl"
        )[0]
        snapshot = replace(
            snapshot,
            provider_game_id=resolved.provider_game_id,
            source_schedule_observation_id=materialization.source_observation_id,
            as_of_utc=scheduled_start - timedelta(days=1),
            scheduled_start_utc=scheduled_start,
        )
        artifact = load_moneyline_model_artifact(FIXTURE_DIR / "model_artifact.json")
        candidate_start = scheduled_start - timedelta(hours=1)
        schedule_candidate = ScheduleCandidateProjection(
            provider_namespace=resolved.provider_namespace,
            provider_game_id=resolved.provider_game_id,
            game_number=resolved.game_number,
            source_schedule_observation_id=materialization.source_observation_id,
            schedule_as_of_utc=materializations.as_of_utc,
            scheduled_start_utc=scheduled_start,
        )
        result = generate_moneyline_predictions(
            (snapshot,),
            artifact,
            prediction_generated_at_utc=candidate_start,
            response_received_at_utc=candidate_start,
            ingested_at_utc=candidate_start,
            schedule_candidates=(schedule_candidate,),
            schedule_pregame_eligibility=eligibility,
        )
        self.assertEqual(
            tuple(admission.admission_status for admission in result.admissions),
            (ADMITTED, ADMITTED),
        )
        self.assertTrue(
            all(
                isinstance(admission.observation, PredictionSourceObservation)
                for admission in result.admissions
            )
        )

    def test_generation_rejects_post_start_prediction_time(self) -> None:
        snapshots = load_moneyline_feature_snapshots(
            FIXTURE_DIR / "feature_snapshots.jsonl"
        )
        artifact = load_moneyline_model_artifact(FIXTURE_DIR / "model_artifact.json")
        with self.assertRaises(ValueError):
            generate_moneyline_predictions(
                snapshots,
                artifact,
                prediction_generated_at_utc="2026-04-05T12:00:00Z",
                response_received_at_utc="2026-04-05T12:00:00Z",
                ingested_at_utc="2026-04-05T12:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
