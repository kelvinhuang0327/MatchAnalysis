"""Characterize schedule observation revision chains against the P6 fixture."""

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


FIXTURE_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "mlb_schedule_observation_v1.json"
)
EXPECTED_REVISION_SET_FINGERPRINT = (
    "ede6f1eea1b2d6a6465df9cc2b31703d50d223607c901b18da25de580f3ca200"
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
        supersedes_key = item["supersedes_fixture_key"]
        previous = observations_by_key.get(supersedes_key)
        supersedes_id = (
            previous.observation_id if previous is not None else None
        )
        raw_payload = item["raw_payload_utf8"].encode("utf-8")
        capture = ScheduleObservationCapture(
            provider_namespace=item["provider_namespace"],
            provider_game_id=item["provider_game_id"],
            scheduled_start_utc=parse_timestamp(item["scheduled_start_utc"]),
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
            supersedes_observation_id=supersedes_id,
        )
        observation = capture_schedule_observation(
            FixtureSource(capture), previous
        )
        observations_by_key[item["fixture_key"]] = observation
        ordered.append(observation)
    return observations_by_key, tuple(ordered)


class ScheduleObservationRevisionFixtureTests(unittest.TestCase):
    def test_fixture_produces_two_chains_with_expected_sizes(self) -> None:
        _, observations = load_fixture_observations()

        revision_set = build_schedule_observation_revision_chains(
            observations
        )

        self.assertEqual(len(revision_set.chains), 2)
        self.assertEqual(
            sorted(chain.observation_count for chain in revision_set.chains),
            [1, 2],
        )
        self.assertEqual(revision_set.unique_observation_count, 3)
        self.assertEqual(revision_set.idempotent_duplicate_count, 0)

    def test_opening_root_and_postponed_head_are_correct(self) -> None:
        by_key, observations = load_fixture_observations()

        revision_set = build_schedule_observation_revision_chains(
            observations
        )

        opening_chain = next(
            chain
            for chain in revision_set.chains
            if chain.observation_count == 2
        )
        self.assertEqual(
            opening_chain.root_observation_id,
            by_key["opening"].observation_id,
        )
        self.assertEqual(
            opening_chain.head_observation_id,
            by_key["postponed_revision"].observation_id,
        )

    def test_doubleheader_game_two_is_an_independent_singleton_chain(
        self,
    ) -> None:
        by_key, observations = load_fixture_observations()

        revision_set = build_schedule_observation_revision_chains(
            observations
        )

        game_two_chain = next(
            chain
            for chain in revision_set.chains
            if chain.observation_count == 1
        )
        self.assertEqual(game_two_chain.game_number, 2)
        self.assertEqual(
            game_two_chain.root_observation_id,
            by_key["doubleheader_game_2"].observation_id,
        )
        self.assertEqual(
            game_two_chain.head_observation_id,
            by_key["doubleheader_game_2"].observation_id,
        )

    def test_full_fingerprint_matches_pre_edit_reference(self) -> None:
        _, observations = load_fixture_observations()

        revision_set = build_schedule_observation_revision_chains(
            observations
        )

        self.assertEqual(
            revision_set.revision_set_fingerprint,
            EXPECTED_REVISION_SET_FINGERPRINT,
        )

    def test_shuffled_and_repeated_input_behave_as_authorized(self) -> None:
        _, observations = load_fixture_observations()

        shuffled = list(observations)
        random.Random(2026).shuffle(shuffled)
        shuffled_result = build_schedule_observation_revision_chains(shuffled)

        repeated = list(observations) + [observations[0]]
        repeated_result = build_schedule_observation_revision_chains(repeated)

        self.assertEqual(
            shuffled_result.revision_set_fingerprint,
            EXPECTED_REVISION_SET_FINGERPRINT,
        )
        self.assertEqual(repeated_result.unique_observation_count, 3)
        self.assertEqual(repeated_result.idempotent_duplicate_count, 1)
        self.assertEqual(
            tuple(chain.root_observation_id for chain in repeated_result.chains),
            tuple(chain.root_observation_id for chain in shuffled_result.chains),
        )

    def test_two_complete_runs_are_identical(self) -> None:
        _, first_observations = load_fixture_observations()
        _, second_observations = load_fixture_observations()

        first_result = build_schedule_observation_revision_chains(
            first_observations
        )
        second_result = build_schedule_observation_revision_chains(
            second_observations
        )

        self.assertEqual(
            first_result.revision_set_fingerprint,
            second_result.revision_set_fingerprint,
        )
        self.assertEqual(
            tuple(chain.root_observation_id for chain in first_result.chains),
            tuple(
                chain.root_observation_id for chain in second_result.chains
            ),
        )


if __name__ == "__main__":
    unittest.main()
