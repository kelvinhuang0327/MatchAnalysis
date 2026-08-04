"""Unit tests for FinalResultObservation domain entity and validation rules."""

from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.final_result_observation import (
    FinalResultObservation,
    compute_final_result_observation_id,
    load_final_result_observations,
)


class FinalResultObservationContractTests(unittest.TestCase):
    def test_valid_final_result_observation(self) -> None:
        obs_id = compute_final_result_observation_id(
            source_result_id="RES_001",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            status="FINAL",
            result_observed_at_utc="2026-04-05T22:00:00Z",
            home_score=5,
            away_score=3,
        )
        obs = FinalResultObservation(
            result_observation_id=obs_id,
            source_result_id="RES_001",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            status="FINAL",
            result_observed_at_utc="2026-04-05T22:00:00Z",
            home_score=5,
            away_score=3,
        )
        self.assertEqual(obs.result_observation_id, obs_id)
        self.assertEqual(obs.home_score, 5)
        self.assertEqual(obs.away_score, 3)

    def test_identity_string_must_be_non_empty_and_trimmed(self) -> None:
        with self.assertRaises(ValueError):
            FinalResultObservation(
                result_observation_id="",
                source_result_id="RES_001",
                provider_namespace="MLB_STATS_API",
                provider_game_id="888001",
                game_number=1,
                status="FINAL",
                result_observed_at_utc="2026-04-05T22:00:00Z",
                home_score=5,
                away_score=3,
            )

        with self.assertRaises(ValueError):
            FinalResultObservation(
                result_observation_id="obs_id",
                source_result_id=" RES_001 ",
                provider_namespace="MLB_STATS_API",
                provider_game_id="888001",
                game_number=1,
                status="FINAL",
                result_observed_at_utc="2026-04-05T22:00:00Z",
                home_score=5,
                away_score=3,
            )

    def test_game_number_must_be_positive_integer_no_bool(self) -> None:
        with self.assertRaises(TypeError):
            FinalResultObservation(
                result_observation_id="obs_id",
                source_result_id="RES_001",
                provider_namespace="MLB_STATS_API",
                provider_game_id="888001",
                game_number=True,  # type: ignore
                status="FINAL",
                result_observed_at_utc="2026-04-05T22:00:00Z",
                home_score=5,
                away_score=3,
            )

        with self.assertRaises(ValueError):
            FinalResultObservation(
                result_observation_id="obs_id",
                source_result_id="RES_001",
                provider_namespace="MLB_STATS_API",
                provider_game_id="888001",
                game_number=0,
                status="FINAL",
                result_observed_at_utc="2026-04-05T22:00:00Z",
                home_score=5,
                away_score=3,
            )

    def test_only_final_status_is_supported(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            FinalResultObservation(
                result_observation_id="obs_id",
                source_result_id="RES_001",
                provider_namespace="MLB_STATS_API",
                provider_game_id="888001",
                game_number=1,
                status="IN_PROGRESS",
                result_observed_at_utc="2026-04-05T22:00:00Z",
                home_score=5,
                away_score=3,
            )
        self.assertIn("FINAL", str(ctx.exception))

    def test_scores_must_be_non_negative_integers_no_bool(self) -> None:
        with self.assertRaises(TypeError):
            FinalResultObservation(
                result_observation_id="obs_id",
                source_result_id="RES_001",
                provider_namespace="MLB_STATS_API",
                provider_game_id="888001",
                game_number=1,
                status="FINAL",
                result_observed_at_utc="2026-04-05T22:00:00Z",
                home_score=True,  # type: ignore
                away_score=3,
            )

        with self.assertRaises(ValueError):
            FinalResultObservation(
                result_observation_id="obs_id",
                source_result_id="RES_001",
                provider_namespace="MLB_STATS_API",
                provider_game_id="888001",
                game_number=1,
                status="FINAL",
                result_observed_at_utc="2026-04-05T22:00:00Z",
                home_score=-1,
                away_score=3,
            )

    def test_tied_final_score_fails_closed(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            FinalResultObservation(
                result_observation_id="obs_id",
                source_result_id="RES_001",
                provider_namespace="MLB_STATS_API",
                provider_game_id="888001",
                game_number=1,
                status="FINAL",
                result_observed_at_utc="2026-04-05T22:00:00Z",
                home_score=4,
                away_score=4,
            )
        self.assertIn("TIED_FINAL_RESULT_UNSUPPORTED", str(ctx.exception))

    def test_load_final_result_observations_duplicate_json_key(self) -> None:
        jsonl = (
            '{"source_result_id":"R1","provider_namespace":"P","provider_game_id":"G1",'
            '"game_number":1,"status":"FINAL","result_observed_at_utc":"2026-04-05T22:00:00Z",'
            '"home_score":5,"away_score":3,"home_score":6}\n'
        )
        with self.assertRaises(ValueError) as ctx:
            load_final_result_observations(jsonl.encode("utf-8"))
        self.assertIn("Duplicate JSON key", str(ctx.exception))

    def test_load_final_result_observations_ambiguous_duplicate_identity(self) -> None:
        row1 = (
            '{"source_result_id":"R1","provider_namespace":"P","provider_game_id":"G1",'
            '"game_number":1,"status":"FINAL","result_observed_at_utc":"2026-04-05T22:00:00Z",'
            '"home_score":5,"away_score":3}\n'
        )
        row2 = (
            '{"source_result_id":"R2","provider_namespace":"P","provider_game_id":"G1",'
            '"game_number":1,"status":"FINAL","result_observed_at_utc":"2026-04-05T23:00:00Z",'
            '"home_score":4,"away_score":2}\n'
        )
        with self.assertRaises(ValueError) as ctx:
            load_final_result_observations((row1 + row2).encode("utf-8"))
        self.assertIn("AMBIGUOUS_FINAL_RESULT_OBSERVATION", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
