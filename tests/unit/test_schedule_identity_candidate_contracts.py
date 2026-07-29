"""Unit tests for immutable schedule identity-resolution candidates."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.ports.schedule_observation_source import (
    ScheduleObservationCapture,
)
from match_analysis.application.use_cases.build_schedule_observation_revision_chains import (
    build_schedule_observation_revision_chains,
)
from match_analysis.application.use_cases.capture_schedule_observation import (
    capture_schedule_observation,
)
from match_analysis.application.use_cases.project_schedule_identity_candidates import (
    project_schedule_identity_candidates,
)
from match_analysis.application.use_cases.select_schedule_observations_as_of import (
    select_schedule_observations_as_of,
)
from match_analysis.baseball.domain.schedule_identity_candidate import (
    ScheduleIdentityResolutionCandidate,
    ScheduleIdentityResolutionCandidateSet,
    compute_schedule_identity_resolution_candidate_set_fingerprint,
)


_BASE_TIME = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


class StubScheduleObservationSource:
    def __init__(self, capture: ScheduleObservationCapture) -> None:
        self._capture = capture

    def capture(self) -> ScheduleObservationCapture:
        return self._capture


def make_observation(
    *,
    provider_game_id: str = "777001",
    game_number: int = 1,
    ingested_at_utc: datetime | None = None,
    scheduled_start_utc: datetime | None = None,
    official_local_date: date = date(2026, 4, 4),
    home_provider_participant_id: str = "118",
    away_provider_participant_id: str = "109",
    payload_tag: str = "opening",
):
    ingested_at_utc = ingested_at_utc or (_BASE_TIME + timedelta(seconds=1))
    scheduled_start_utc = scheduled_start_utc or (
        _BASE_TIME + timedelta(days=34, hours=6, minutes=10)
    )
    raw_payload = (
        f'{{"tag":"{payload_tag}","game_id":"{provider_game_id}",'
        f'"game_number":{game_number}}}'
    ).encode("utf-8")
    capture = ScheduleObservationCapture(
        provider_namespace="MLB_STATS_API",
        provider_game_id=provider_game_id,
        scheduled_start_utc=scheduled_start_utc,
        official_local_date=official_local_date,
        response_received_at_utc=ingested_at_utc - timedelta(microseconds=1),
        ingested_at_utc=ingested_at_utc,
        provider_status_code="S",
        provider_detailed_status="Scheduled",
        game_number=game_number,
        home_provider_participant_id=home_provider_participant_id,
        away_provider_participant_id=away_provider_participant_id,
        endpoint_id="mlb_schedule_v1",
        parser_version="matchanalysis_schedule_parser_v1",
        schema_version="schedule_source_observation_v1",
        raw_payload_bytes=raw_payload,
        raw_payload_sha256=sha256(raw_payload).hexdigest(),
        supersedes_observation_id=None,
    )
    return capture_schedule_observation(StubScheduleObservationSource(capture))


def project_observations(*observations, as_of_utc=None):
    revision_set = build_schedule_observation_revision_chains(observations)
    snapshot = select_schedule_observations_as_of(
        revision_set,
        as_of_utc
        or max(observation.ingested_at_utc for observation in observations),
    )
    return snapshot, project_schedule_identity_candidates(snapshot)


class ProjectScheduleIdentityCandidatesTests(unittest.TestCase):
    def test_one_candidate_copies_exact_selected_observation_evidence(
        self,
    ) -> None:
        observation = make_observation(
            scheduled_start_utc=datetime(
                2026, 4, 4, 18, 10, tzinfo=timezone.utc
            ),
            official_local_date=date(2026, 4, 4),
        )
        snapshot, result = project_observations(observation)
        candidate = result.candidates[0]

        self.assertEqual(result.as_of_utc, snapshot.as_of_utc)
        self.assertEqual(
            result.source_snapshot_fingerprint,
            snapshot.snapshot_fingerprint,
        )
        self.assertEqual(
            (
                candidate.provider_namespace,
                candidate.provider_game_id,
                candidate.game_number,
                candidate.scheduled_start_utc,
                candidate.official_local_date,
                candidate.home_provider_participant_id,
                candidate.away_provider_participant_id,
                candidate.source_observation_id,
                candidate.source_raw_payload_sha256,
            ),
            (
                observation.provider_namespace,
                observation.provider_game_id,
                observation.game_number,
                observation.scheduled_start_utc,
                observation.official_local_date,
                observation.home_provider_participant_id,
                observation.away_provider_participant_id,
                observation.observation_id,
                observation.raw_payload_sha256,
            ),
        )

    def test_unavailable_chain_produces_no_candidate(self) -> None:
        observation = make_observation()
        snapshot, result = project_observations(
            observation,
            as_of_utc=observation.ingested_at_utc
            - timedelta(microseconds=1),
        )

        self.assertEqual(result.candidates, ())
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(
            result.unavailable_chain_keys,
            snapshot.unavailable_chain_keys,
        )
        self.assertEqual(result.unavailable_count, 1)

    def test_doubleheader_game_numbers_remain_distinct_and_ordered(
        self,
    ) -> None:
        game_one = make_observation(game_number=1)
        game_two = make_observation(
            provider_game_id="777002",
            game_number=2,
            payload_tag="game-two",
        )

        _, result = project_observations(game_two, game_one)

        self.assertEqual(
            tuple(
                (candidate.provider_game_id, candidate.game_number)
                for candidate in result.candidates
            ),
            (("777001", 1), ("777002", 2)),
        )

    def test_schema_version_is_exact_and_projection_is_immutable(self) -> None:
        _, result = project_observations(make_observation())

        self.assertEqual(
            result.schema_version,
            "schedule_identity_resolution_candidate_set_v1",
        )
        with self.assertRaises(FrozenInstanceError):
            result.candidate_count = 2
        with self.assertRaises(FrozenInstanceError):
            result.candidates[0].game_number = 2

    def test_contract_contains_only_resolution_evidence_not_identity(
        self,
    ) -> None:
        fields = set(
            ScheduleIdentityResolutionCandidate.__dataclass_fields__
        )

        self.assertEqual(
            fields,
            {
                "provider_namespace",
                "provider_game_id",
                "game_number",
                "scheduled_start_utc",
                "official_local_date",
                "home_provider_participant_id",
                "away_provider_participant_id",
                "source_observation_id",
                "source_raw_payload_sha256",
            },
        )
        self.assertTrue(
            {
                "match_identity",
                "canonical_team",
                "team_name",
                "status",
                "baseball_game",
                "prediction",
            }.isdisjoint(fields)
        )

    def test_non_snapshot_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            project_schedule_identity_candidates(object())


class CandidateContractValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        _, result = project_observations(make_observation())
        self.candidate = result.candidates[0]

    def test_naive_scheduled_start_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(
                self.candidate,
                scheduled_start_utc=datetime(2026, 4, 4, 18, 10),
            )

    def test_non_positive_or_boolean_game_number_is_rejected(self) -> None:
        for value in (0, -1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    replace(self.candidate, game_number=value)

    def test_invalid_source_hashes_are_rejected(self) -> None:
        for field_name in (
            "source_observation_id",
            "source_raw_payload_sha256",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    replace(self.candidate, **{field_name: "0" * 63})


class CandidateSetContractValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        game_one = make_observation()
        game_two = make_observation(
            provider_game_id="777002",
            game_number=2,
            payload_tag="game-two",
        )
        _, self.result = project_observations(game_one, game_two)

    def _fingerprint(self, *, candidates=None, unavailable_chain_keys=None):
        candidates = (
            self.result.candidates if candidates is None else candidates
        )
        unavailable_chain_keys = (
            self.result.unavailable_chain_keys
            if unavailable_chain_keys is None
            else unavailable_chain_keys
        )
        return (
            compute_schedule_identity_resolution_candidate_set_fingerprint(
                as_of_utc=self.result.as_of_utc,
                source_snapshot_fingerprint=(
                    self.result.source_snapshot_fingerprint
                ),
                candidate_count=len(candidates),
                unavailable_count=len(unavailable_chain_keys),
                candidates=candidates,
                unavailable_chain_keys=unavailable_chain_keys,
            )
        )

    def test_tampered_fingerprint_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.result, candidate_set_fingerprint="0" * 64)

    def test_wrong_schema_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.result, schema_version="wrong")

    def test_mismatched_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.result, candidate_count=3)

    def test_unsorted_candidates_are_rejected(self) -> None:
        reversed_candidates = tuple(reversed(self.result.candidates))
        with self.assertRaises(ValueError):
            ScheduleIdentityResolutionCandidateSet(
                as_of_utc=self.result.as_of_utc,
                source_snapshot_fingerprint=(
                    self.result.source_snapshot_fingerprint
                ),
                candidates=reversed_candidates,
                unavailable_chain_keys=(),
                candidate_count=2,
                unavailable_count=0,
                candidate_set_fingerprint=self._fingerprint(
                    candidates=reversed_candidates
                ),
                schema_version=self.result.schema_version,
            )

    def test_duplicate_key_across_candidates_and_unavailable_is_rejected(
        self,
    ) -> None:
        candidate = self.result.candidates[0]
        duplicate_key = (
            candidate.provider_namespace,
            candidate.provider_game_id,
            candidate.game_number,
        )
        with self.assertRaises(ValueError):
            ScheduleIdentityResolutionCandidateSet(
                as_of_utc=self.result.as_of_utc,
                source_snapshot_fingerprint=(
                    self.result.source_snapshot_fingerprint
                ),
                candidates=self.result.candidates,
                unavailable_chain_keys=(duplicate_key,),
                candidate_count=2,
                unavailable_count=1,
                candidate_set_fingerprint=self._fingerprint(
                    unavailable_chain_keys=(duplicate_key,)
                ),
                schema_version=self.result.schema_version,
            )


if __name__ == "__main__":
    unittest.main()
