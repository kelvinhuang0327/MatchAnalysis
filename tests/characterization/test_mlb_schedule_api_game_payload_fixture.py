"""Characterize the tracked synthetic MLB Stats API game payload fixture."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.capture_schedule_observation import (
    capture_schedule_observation,
)
from match_analysis.baseball.domain.schedule_observation import (
    ScheduleSourceObservation,
)
from match_analysis.infrastructure.mlb_schedule import (
    ExplicitMlbSchedulePayloadSource,
)


FIXTURE_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "mlb_schedule_api_game_payload_v1.json"
)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_fixture():
    fixture_bytes = FIXTURE_PATH.read_bytes()
    fixture = json.loads(fixture_bytes)
    observations: dict[str, ScheduleSourceObservation] = {}
    ordered = []
    raw_payloads = []
    for case in fixture["cases"]:
        supersedes_key = case["supersedes_fixture_key"]
        previous = observations.get(supersedes_key)
        source = ExplicitMlbSchedulePayloadSource(
            raw_payload_bytes=case["raw_payload_utf8"].encode("utf-8"),
            response_received_at_utc=parse_timestamp(
                case["response_received_at_utc"]
            ),
            ingested_at_utc=parse_timestamp(case["ingested_at_utc"]),
            endpoint_id=case["endpoint_id"],
            parser_version=case["parser_version"],
            schema_version=case["schema_version"],
            supersedes_observation_id=(
                previous.observation_id if previous is not None else None
            ),
        )
        observation = capture_schedule_observation(source, previous)
        observations[case["fixture_key"]] = observation
        ordered.append(observation)
        raw_payloads.append(observation.raw_payload_bytes)
    return fixture, tuple(ordered), tuple(raw_payloads)


class MlbSchedulePayloadFixtureTests(unittest.TestCase):
    def test_all_fixture_captures_match_expected_field_projections(
        self,
    ) -> None:
        fixture, observations, _ = run_fixture()

        self.assertEqual(len(observations), 3)
        for case, observation in zip(fixture["cases"], observations):
            with self.subTest(fixture_key=case["fixture_key"]):
                self.assertEqual(
                    observation.raw_payload_sha256,
                    case["expected_raw_payload_sha256"],
                )
                self.assertEqual(
                    observation.observation_id,
                    case["expected_observation_id"],
                )

    def test_raw_hashes_match_pre_edit_reference(self) -> None:
        _, _, raw_payloads = run_fixture()

        self.assertEqual(
            tuple(sha256(payload).hexdigest() for payload in raw_payloads),
            (
                "8f7a7df62e17df8b2565c159ef0848fe3d880e670afebd80abe41ebdfe70d4a9",
                "cab0963c174739518f00fa79536a1c1fd89efe01a6f7e87ffd17debf06a75f3e",
                "95bd1521b9b0e314436690cab87bd3d67b5aba72f9d07b21074ef44a3b82e685",
            ),
        )

    def test_postponed_revision_explicitly_supersedes_opening(self) -> None:
        _, observations, _ = run_fixture()
        opening, postponed, _ = observations

        self.assertEqual(
            postponed.supersedes_observation_id, opening.observation_id
        )
        self.assertNotEqual(postponed.observation_id, opening.observation_id)
        self.assertEqual(opening.provider_status_code, "S")
        self.assertEqual(opening.provider_detailed_status, "Scheduled")
        self.assertEqual(postponed.provider_status_code, "P")
        self.assertEqual(postponed.provider_detailed_status, "Postponed")

    def test_doubleheader_game_two_remains_distinct(self) -> None:
        _, observations, _ = run_fixture()
        opening, _, game_two = observations

        self.assertEqual(game_two.game_number, 2)
        self.assertEqual(game_two.provider_game_id, "777002")
        self.assertIsNone(game_two.supersedes_observation_id)
        self.assertNotEqual(game_two.observation_id, opening.observation_id)

    def test_official_date_is_explicit_and_not_derived_from_game_date(
        self,
    ) -> None:
        _, observations, _ = run_fixture()
        _, postponed, _ = observations

        self.assertEqual(
            postponed.scheduled_start_utc.date().isoformat(), "2026-04-05"
        )
        self.assertEqual(postponed.official_local_date.isoformat(), "2026-04-05")

    def test_two_full_fixture_runs_are_identical(self) -> None:
        first_fixture, first_observations, first_payloads = run_fixture()
        second_fixture, second_observations, second_payloads = run_fixture()

        self.assertEqual(first_fixture, second_fixture)
        self.assertEqual(first_payloads, second_payloads)
        self.assertEqual(
            tuple(item.observation_id for item in first_observations),
            tuple(item.observation_id for item in second_observations),
        )


if __name__ == "__main__":
    unittest.main()
