"""Unit tests for prospective prediction admission workflow."""

from decimal import Decimal
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.canonical_utc import parse_canonical_utc
from match_analysis.baseball.domain.match_identity_authority import (
    MatchIdentityAuthorityEntry,
    build_match_identity_authority_catalog,
)
from match_analysis.baseball.domain.participant_identity_resolution import (
    ProviderParticipantIdentityMapping,
)
from match_analysis.baseball.domain.prediction_admission import (
    ProspectivePredictionCandidate,
    ADMITTED,
    REJECTED,
    MISSING_SCHEDULE_CANDIDATE_MATCH,
    SCHEDULE_NOT_PREGAME_ELIGIBLE,
)
from match_analysis.baseball.domain.prediction_source_observation import (
    compute_prediction_observation_id,
)
from match_analysis.infrastructure.mlb_schedule import ExplicitMlbSchedulePayloadSource
from match_analysis.application.use_cases.capture_schedule_observation import (
    capture_schedule_observation,
)
from match_analysis.application.use_cases.run_prospective_prediction_admission_workflow import (
    run_prospective_prediction_admission_workflow,
    compute_result_set_fingerprint,
)


AS_OF = datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc)
RESP_TIME = datetime(2026, 4, 1, 9, 59, 59, tzinfo=timezone.utc)
ING_TIME = datetime(2026, 4, 1, 10, 0, 0, tzinfo=timezone.utc)


def make_payload_source(
    game_pk: int = 888001,
    game_number: int = 1,
    game_date: str = "2026-04-05T19:00:00Z",
    home_id: int = 118,
    away_id: int = 109,
    status_code: str = "S",
    detailed_state: str = "Scheduled",
    supersedes_observation_id: str | None = None,
) -> ExplicitMlbSchedulePayloadSource:
    payload = {
        "gamePk": game_pk,
        "gameDate": game_date,
        "officialDate": "2026-04-05",
        "gameNumber": game_number,
        "status": {"statusCode": status_code, "detailedState": detailed_state},
        "teams": {"home": {"team": {"id": home_id}}, "away": {"team": {"id": away_id}}},
    }
    raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return ExplicitMlbSchedulePayloadSource(
        raw_payload_bytes=raw_bytes,
        response_received_at_utc=RESP_TIME,
        ingested_at_utc=ING_TIME,
        endpoint_id="mlb_schedule_v1",
        parser_version="matchanalysis_mlb_game_payload_parser_v1",
        schema_version="mlb_schedule_api_game_payload_v1",
        supersedes_observation_id=supersedes_observation_id,
    )


def make_mappings() -> tuple[ProviderParticipantIdentityMapping, ...]:
    return (
        ProviderParticipantIdentityMapping("MLB_STATS_API", "118", "CANONICAL_118", "v1"),
        ProviderParticipantIdentityMapping("MLB_STATS_API", "109", "CANONICAL_109", "v1"),
    )


def make_authority() -> tuple[MatchIdentityAuthorityEntry, ...]:
    return (
        MatchIdentityAuthorityEntry("MLB_STATS_API", "888001", 1, "MLB", 2026, "CANONICAL_GAME_888001_1", None, "v1"),
    )


class ProspectivePredictionAdmissionWorkflowTests(unittest.TestCase):
    def test_workflow_admits_valid_prediction(self) -> None:
        source = make_payload_source()
        obs_id = capture_schedule_observation(source).observation_id

        gen_dt = parse_canonical_utc("2026-04-05T11:00:00Z")
        rec_dt = parse_canonical_utc("2026-04-05T11:00:01Z")
        ing_dt = parse_canonical_utc("2026-04-05T11:00:02Z")
        start_dt = parse_canonical_utc("2026-04-05T19:00:00Z")

        pred_obs_id = compute_prediction_observation_id(
            source_prediction_id="P1",
            model_id="M1",
            market_id="MONEYLINE",
            selection="HOME",
            model_probability=Decimal("0.55"),
            line_value=Decimal("-110"),
            push_policy="PUSH_VOID",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            source_schedule_observation_id=obs_id,
            prediction_generated_at_utc=gen_dt,
            response_received_at_utc=rec_dt,
            ingested_at_utc=ing_dt,
            scheduled_start_utc=start_dt,
        )

        candidate = ProspectivePredictionCandidate(
            prediction_observation_id=pred_obs_id,
            source_prediction_id="P1",
            model_id="M1",
            market_id="MONEYLINE",
            selection="HOME",
            model_probability=Decimal("0.55"),
            line_value=Decimal("-110"),
            push_policy="PUSH_VOID",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            source_schedule_observation_id=obs_id,
            prediction_generated_at_utc="2026-04-05T11:00:00Z",
            response_received_at_utc="2026-04-05T11:00:01Z",
            ingested_at_utc="2026-04-05T11:00:02Z",
        )

        authority_cat = build_match_identity_authority_catalog(make_authority())
        result = run_prospective_prediction_admission_workflow(
            requests=(candidate,),
            raw_schedule_sources=(source,),
            participant_mappings=make_mappings(),
            authority_catalog=authority_cat,
            schedule_as_of_utc=AS_OF,
        )

        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0].admission_status, ADMITTED)
        self.assertIsNone(result.results[0].reason)
        self.assertIsNotNone(result.results[0].observation)
        self.assertEqual(result.admitted_count, 1)
        self.assertEqual(result.rejected_count, 0)

    def test_duplicate_request_identity_aborts_batch(self) -> None:
        source = make_payload_source()
        obs_id = capture_schedule_observation(source).observation_id
        candidate = ProspectivePredictionCandidate(
            prediction_observation_id="0" * 64,
            source_prediction_id="P1",
            model_id="M1",
            market_id="MONEYLINE",
            selection="HOME",
            model_probability=Decimal("0.55"),
            line_value=Decimal("-110"),
            push_policy="PUSH_VOID",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            source_schedule_observation_id=obs_id,
            prediction_generated_at_utc="2026-04-05T11:00:00Z",
            response_received_at_utc="2026-04-05T11:00:01Z",
            ingested_at_utc="2026-04-05T11:00:02Z",
        )

        authority_cat = build_match_identity_authority_catalog(make_authority())
        with self.assertRaises(ValueError) as ctx:
            run_prospective_prediction_admission_workflow(
                requests=(candidate, candidate),
                raw_schedule_sources=(source,),
                participant_mappings=make_mappings(),
                authority_catalog=authority_cat,
                schedule_as_of_utc=AS_OF,
            )
        self.assertIn("Duplicate prediction request identity", str(ctx.exception))

    def test_shuffled_sources_preserve_result_set_fingerprint(self) -> None:
        s1 = make_payload_source(game_pk=888001, game_number=1)
        s2 = make_payload_source(game_pk=888002, game_number=1, away_id=109)
        obs_id1 = capture_schedule_observation(s1).observation_id

        candidate = ProspectivePredictionCandidate(
            prediction_observation_id="0" * 64,
            source_prediction_id="P1",
            model_id="M1",
            market_id="MONEYLINE",
            selection="HOME",
            model_probability=Decimal("0.55"),
            line_value=Decimal("-110"),
            push_policy="PUSH_VOID",
            provider_namespace="MLB_STATS_API",
            provider_game_id="888001",
            game_number=1,
            source_schedule_observation_id=obs_id1,
            prediction_generated_at_utc="2026-04-05T11:00:00Z",
            response_received_at_utc="2026-04-05T11:00:01Z",
            ingested_at_utc="2026-04-05T11:00:02Z",
        )

        auth_entries = (
            MatchIdentityAuthorityEntry("MLB_STATS_API", "888001", 1, "MLB", 2026, "CANONICAL_888001", None, "v1"),
            MatchIdentityAuthorityEntry("MLB_STATS_API", "888002", 1, "MLB", 2026, "CANONICAL_888002", None, "v1"),
        )
        authority_cat = build_match_identity_authority_catalog(auth_entries)

        r1 = run_prospective_prediction_admission_workflow(
            requests=(candidate,),
            raw_schedule_sources=(s1, s2),
            participant_mappings=make_mappings(),
            authority_catalog=authority_cat,
            schedule_as_of_utc=AS_OF,
        )
        r2 = run_prospective_prediction_admission_workflow(
            requests=(candidate,),
            raw_schedule_sources=(s2, s1),
            participant_mappings=make_mappings(),
            authority_catalog=authority_cat,
            schedule_as_of_utc=AS_OF,
        )

        self.assertEqual(r1.result_set_fingerprint, r2.result_set_fingerprint)


if __name__ == "__main__":
    unittest.main()
