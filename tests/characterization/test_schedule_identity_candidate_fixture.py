"""Characterize P8-to-candidate projection against the tracked fixture."""

from datetime import date, datetime
import json
from pathlib import Path
import random
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


FIXTURE_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "mlb_schedule_observation_v1.json"
)

BEFORE_ALL_AS_OF = "2026-01-01T00:00:00Z"
BEFORE_ALL_FINGERPRINT = (
    "8c883bc544b531b8ed441f615536910c53b6f7d1448711841aee33cc540cf376"
)
MID_CUTOFF_AS_OF = "2026-03-15T00:00:00Z"
MID_CUTOFF_FINGERPRINT = (
    "8cc7e70e9a2707a69bbd39aac656e16d8eef13761ac83f41570ee48ed05fa5fd"
)
AFTER_POSTPONED_AS_OF = "2026-05-01T00:00:00Z"
AFTER_POSTPONED_FINGERPRINT = (
    "debabe4b6ada6a71837c37171364fecbd08e680a34f55c0652a788b774d86ec6"
)
EXACT_BOUNDARY_AS_OF = "2026-03-01T12:00:01.500Z"
EXACT_BOUNDARY_FINGERPRINT = (
    "7ccbb4ba10ac88a2507a45c7743f983fe02314d59a24f7aae5d856bf6053f2dc"
)


class FixtureSource:
    def __init__(self, capture: ScheduleObservationCapture) -> None:
        self._capture = capture

    def capture(self) -> ScheduleObservationCapture:
        return self._capture


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_fixture_observations():
    fixture = json.loads(FIXTURE_PATH.read_bytes())
    observations_by_key = {}
    ordered = []
    for item in fixture["observations"]:
        previous = observations_by_key.get(
            item["supersedes_fixture_key"]
        )
        raw_payload = item["raw_payload_utf8"].encode("utf-8")
        capture = ScheduleObservationCapture(
            provider_namespace=item["provider_namespace"],
            provider_game_id=item["provider_game_id"],
            scheduled_start_utc=parse_timestamp(
                item["scheduled_start_utc"]
            ),
            official_local_date=date.fromisoformat(
                item["official_local_date"]
            ),
            response_received_at_utc=parse_timestamp(
                item["response_received_at_utc"]
            ),
            ingested_at_utc=parse_timestamp(item["ingested_at_utc"]),
            provider_status_code=item["provider_status_code"],
            provider_detailed_status=item["provider_detailed_status"],
            game_number=item["game_number"],
            home_provider_participant_id=(
                item["home_provider_participant_id"]
            ),
            away_provider_participant_id=(
                item["away_provider_participant_id"]
            ),
            endpoint_id=item["endpoint_id"],
            parser_version=item["parser_version"],
            schema_version=item["schema_version"],
            raw_payload_bytes=raw_payload,
            raw_payload_sha256=item["raw_payload_sha256"],
            supersedes_observation_id=(
                previous.observation_id if previous is not None else None
            ),
        )
        observation = capture_schedule_observation(
            FixtureSource(capture), previous
        )
        observations_by_key[item["fixture_key"]] = observation
        ordered.append(observation)
    return observations_by_key, tuple(ordered)


def project_at(observations, cutoff):
    revision_set = build_schedule_observation_revision_chains(observations)
    snapshot = select_schedule_observations_as_of(
        revision_set, parse_timestamp(cutoff)
    )
    return project_schedule_identity_candidates(snapshot)


class ScheduleIdentityCandidateFixtureTests(unittest.TestCase):
    def test_before_all_has_no_candidates_and_two_unavailable_keys(
        self,
    ) -> None:
        _, observations = load_fixture_observations()

        result = project_at(observations, BEFORE_ALL_AS_OF)

        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.unavailable_count, 2)
        self.assertEqual(
            result.unavailable_chain_keys,
            (
                ("MLB_STATS_API", "777001", 1),
                ("MLB_STATS_API", "777002", 2),
            ),
        )
        self.assertEqual(
            result.candidate_set_fingerprint,
            BEFORE_ALL_FINGERPRINT,
        )

    def test_mid_cutoff_uses_opening_and_doubleheader_game_two(
        self,
    ) -> None:
        by_key, observations = load_fixture_observations()

        result = project_at(observations, MID_CUTOFF_AS_OF)

        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.unavailable_count, 0)
        self.assertEqual(
            tuple(
                candidate.source_observation_id
                for candidate in result.candidates
            ),
            (
                by_key["opening"].observation_id,
                by_key["doubleheader_game_2"].observation_id,
            ),
        )
        self.assertEqual(
            result.candidate_set_fingerprint,
            MID_CUTOFF_FINGERPRINT,
        )

    def test_after_postponed_replaces_opening_for_game_777001(
        self,
    ) -> None:
        by_key, observations = load_fixture_observations()

        result = project_at(observations, AFTER_POSTPONED_AS_OF)

        first = result.candidates[0]
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.unavailable_count, 0)
        self.assertEqual(first.provider_game_id, "777001")
        self.assertEqual(
            first.source_observation_id,
            by_key["postponed_revision"].observation_id,
        )
        self.assertNotEqual(
            first.source_observation_id,
            by_key["opening"].observation_id,
        )
        self.assertEqual(
            result.candidate_set_fingerprint,
            AFTER_POSTPONED_FINGERPRINT,
        )

    def test_exact_boundary_has_one_candidate_and_one_unavailable_key(
        self,
    ) -> None:
        by_key, observations = load_fixture_observations()

        result = project_at(observations, EXACT_BOUNDARY_AS_OF)

        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.unavailable_count, 1)
        self.assertEqual(
            result.candidates[0].source_observation_id,
            by_key["opening"].observation_id,
        )
        self.assertEqual(
            result.unavailable_chain_keys,
            (("MLB_STATS_API", "777002", 2),),
        )
        self.assertEqual(
            result.candidate_set_fingerprint,
            EXACT_BOUNDARY_FINGERPRINT,
        )

    def test_provenance_participant_ids_and_game_numbers_are_preserved(
        self,
    ) -> None:
        by_key, observations = load_fixture_observations()

        result = project_at(observations, AFTER_POSTPONED_AS_OF)
        expected = (
            by_key["postponed_revision"],
            by_key["doubleheader_game_2"],
        )

        for candidate, observation in zip(
            result.candidates, expected, strict=True
        ):
            with self.subTest(
                provider_game_id=candidate.provider_game_id
            ):
                self.assertEqual(
                    candidate.source_raw_payload_sha256,
                    observation.raw_payload_sha256,
                )
                self.assertEqual(
                    candidate.source_observation_id,
                    observation.observation_id,
                )
                self.assertEqual(
                    candidate.provider_namespace,
                    observation.provider_namespace,
                )
                self.assertEqual(
                    candidate.provider_game_id,
                    observation.provider_game_id,
                )
                self.assertEqual(
                    candidate.home_provider_participant_id,
                    observation.home_provider_participant_id,
                )
                self.assertEqual(
                    candidate.away_provider_participant_id,
                    observation.away_provider_participant_id,
                )
                self.assertEqual(
                    candidate.game_number,
                    observation.game_number,
                )

    def test_shuffled_and_repeated_inputs_have_identical_fingerprints(
        self,
    ) -> None:
        _, observations = load_fixture_observations()

        ordered = project_at(observations, AFTER_POSTPONED_AS_OF)
        shuffled_observations = list(observations)
        random.Random(2026).shuffle(shuffled_observations)
        shuffled = project_at(
            shuffled_observations, AFTER_POSTPONED_AS_OF
        )
        repeated = project_at(observations, AFTER_POSTPONED_AS_OF)

        self.assertEqual(
            ordered.candidate_set_fingerprint,
            AFTER_POSTPONED_FINGERPRINT,
        )
        self.assertEqual(
            ordered.candidate_set_fingerprint,
            shuffled.candidate_set_fingerprint,
        )
        self.assertEqual(
            ordered.candidate_set_fingerprint,
            repeated.candidate_set_fingerprint,
        )
        self.assertEqual(ordered, shuffled)
        self.assertEqual(ordered, repeated)


if __name__ == "__main__":
    unittest.main()
