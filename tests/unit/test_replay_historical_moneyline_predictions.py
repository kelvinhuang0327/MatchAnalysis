"""Focused tests for P20A bounded replay through the P19A path."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.reconstruct_moneyline_walk_forward_model import (
    load_moneyline_walk_forward_fold,
)
from match_analysis.application.use_cases.evaluate_schedule_pregame_eligibility import (
    evaluate_schedule_pregame_eligibility,
)
from match_analysis.application.use_cases.generate_moneyline_predictions import (
    generate_moneyline_predictions,
)
from match_analysis.application.use_cases.moneyline_inference_artifacts import (
    load_moneyline_feature_snapshots,
)
from match_analysis.application.use_cases.replay_historical_moneyline_predictions import (
    replay_historical_moneyline_predictions,
    write_moneyline_walk_forward_replay_artifacts,
)
from match_analysis.baseball.domain.moneyline_walk_forward_fold import (
    ReconstructedWalkForwardModel,
)
from match_analysis.baseball.domain.prediction_admission import (
    ADMITTED,
    ProspectivePredictionCandidate,
    ScheduleCandidateProjection,
)
from tests.unit.test_construct_match_identities import make_resolved_candidate
from tests.unit.test_pregame_eligibility_contracts import one_game_materialization_set


FIXTURE = (
    REPOSITORY_ROOT
    / "data"
    / "fixtures"
    / "p20a_p13_walk_forward"
    / "fold_wf_001.json"
)


def _reconstructed_model() -> ReconstructedWalkForwardModel:
    return ReconstructedWalkForwardModel(
        fold_id="wf_001",
        feature_names=(
            "indep_recent_win_rate_delta",
            "indep_starter_era_delta",
        ),
        coefficients=(0.2820639975011785, -0.09609616789072993),
        intercept=0.19592616199017746,
        scaler_means=(-0.004478903507172063, -0.09830887114456008),
        scaler_stds=(0.2364885946804932, 2.1424749536909156),
        train_size=566,
    )


class ReplayHistoricalMoneylinePredictionsTests(unittest.TestCase):
    def test_every_bounded_row_matches_legacy_probability(self) -> None:
        fold = load_moneyline_walk_forward_fold(FIXTURE)
        result = replay_historical_moneyline_predictions(fold, _reconstructed_model())
        self.assertTrue(result.parity_passed)
        self.assertLessEqual(result.max_absolute_difference, Decimal("0.000001"))
        self.assertEqual(len(result.parity_rows), 2)
        self.assertEqual(len(result.inference.candidates), 4)

    def test_replay_candidates_are_p15c_compatible_inputs(self) -> None:
        fold = load_moneyline_walk_forward_fold(FIXTURE)
        result = replay_historical_moneyline_predictions(fold, _reconstructed_model())
        self.assertTrue(
            all(
                isinstance(candidate, ProspectivePredictionCandidate)
                for candidate in result.inference.candidates
            )
        )
        self.assertEqual(
            {candidate.market_id for candidate in result.inference.candidates},
            {"moneyline"},
        )
        self.assertEqual(
            {candidate.selection for candidate in result.inference.candidates},
            {"HOME", "AWAY"},
        )

    def test_reconstructed_artifact_is_accepted_by_p15c(self) -> None:
        materializations = one_game_materialization_set()
        eligibility = evaluate_schedule_pregame_eligibility(materializations)
        materialization = materializations.game_materializations[0]
        scheduled_start = materialization.baseball_game.scheduled_start.value
        resolved = make_resolved_candidate()
        snapshot = load_moneyline_feature_snapshots(
            REPOSITORY_ROOT
            / "data"
            / "fixtures"
            / "p19a_moneyline_inference"
            / "feature_snapshots.jsonl"
        )[0]
        snapshot = replace(
            snapshot,
            provider_game_id=resolved.provider_game_id,
            source_schedule_observation_id=materialization.source_observation_id,
            as_of_utc=scheduled_start - timedelta(days=1),
            scheduled_start_utc=scheduled_start,
        )
        fold = load_moneyline_walk_forward_fold(FIXTURE)
        result = replay_historical_moneyline_predictions(fold, _reconstructed_model())
        candidate_start = scheduled_start - timedelta(hours=1)
        schedule_candidate = ScheduleCandidateProjection(
            provider_namespace=resolved.provider_namespace,
            provider_game_id=resolved.provider_game_id,
            game_number=resolved.game_number,
            source_schedule_observation_id=materialization.source_observation_id,
            schedule_as_of_utc=materializations.as_of_utc,
            scheduled_start_utc=scheduled_start,
        )
        admitted = generate_moneyline_predictions(
            (snapshot,),
            result.model_artifact,
            prediction_generated_at_utc=candidate_start,
            response_received_at_utc=candidate_start,
            ingested_at_utc=candidate_start,
            schedule_candidates=(schedule_candidate,),
            schedule_pregame_eligibility=eligibility,
        )
        self.assertEqual(
            tuple(item.admission_status for item in admitted.admissions),
            (ADMITTED, ADMITTED),
        )

    def test_replay_artifacts_are_byte_identical(self) -> None:
        fold = load_moneyline_walk_forward_fold(FIXTURE)
        result = replay_historical_moneyline_predictions(fold, _reconstructed_model())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            write_moneyline_walk_forward_replay_artifacts(first, result)
            write_moneyline_walk_forward_replay_artifacts(second, result)
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())


if __name__ == "__main__":
    unittest.main()
