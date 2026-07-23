"""Characterize the deterministic synthetic schedule observation fixture."""

from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.ports.schedule_observation_source import (
    ScheduleObservationCapture,
)
from match_analysis.application.use_cases.capture_schedule_observation import (
    capture_schedule_observation,
)
from match_analysis.baseball.domain.schedule_observation import (
    ScheduleSourceObservation,
)


FIXTURE_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "mlb_schedule_observation_v1.json"
)
EXPECTED_RAW_HASHES = (
    "85d380d36f2b8d131fafab98a661da1d333cfc9cb33bb97d32a2466d167cf5aa",
    "40f3c78da92cf9360aa89558798ea6aff290c5bee09e1eacf30e3e7caf99f4fa",
    "84667e7217e7bb2ddc630e0b101c232c28885845ea5e46855164df2b7ace3063",
)
EXPECTED_OBSERVATION_IDS = (
    "d7f5a5b129e340a8deca33805bb872dee94380cf9701aed96785d84ce991f0af",
    "d41f39dc79062d665a4b355cee1c6c81bf1ece14c85f698ccd8781b3be21dbe7",
    "26da69ff6bd32581f6866f4e559b97ddc60bc7e7f5b08a816209343aa513a1db",
)


class FixtureSource:
    def __init__(self, capture: ScheduleObservationCapture) -> None:
        self._capture = capture

    def capture(self) -> ScheduleObservationCapture:
        return self._capture


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_fixture():
    fixture_bytes = FIXTURE_PATH.read_bytes()
    fixture = json.loads(fixture_bytes)
    observations: dict[str, ScheduleSourceObservation] = {}
    ordered = []
    raw_payloads = []
    for item in fixture["observations"]:
        supersedes_key = item["supersedes_fixture_key"]
        previous = observations.get(supersedes_key)
        supersedes_id = (
            previous.observation_id if previous is not None else None
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
            supersedes_observation_id=supersedes_id,
        )
        observation = capture_schedule_observation(
            FixtureSource(capture),
            previous,
        )
        observations[item["fixture_key"]] = observation
        ordered.append(observation)
        raw_payloads.append(raw_payload)
    return fixture_bytes, tuple(ordered), tuple(raw_payloads)


class ScheduleObservationFixtureTests(unittest.TestCase):
    def test_all_fixture_hashes_and_ids_match_independent_references(
        self,
    ) -> None:
        _, observations, raw_payloads = run_fixture()

        self.assertEqual(len(observations), 3)
        self.assertEqual(
            tuple(sha256(payload).hexdigest() for payload in raw_payloads),
            EXPECTED_RAW_HASHES,
        )
        self.assertEqual(
            tuple(item.observation_id for item in observations),
            EXPECTED_OBSERVATION_IDS,
        )

    def test_postponement_is_an_explicit_append_only_revision(self) -> None:
        _, observations, _ = run_fixture()
        opening, postponed, _ = observations

        self.assertEqual(
            postponed.supersedes_observation_id,
            opening.observation_id,
        )
        self.assertNotEqual(postponed.observation_id, opening.observation_id)
        self.assertEqual(opening.provider_status_code, "S")
        self.assertEqual(opening.provider_detailed_status, "Scheduled")
        self.assertEqual(postponed.provider_status_code, "P")
        self.assertEqual(postponed.provider_detailed_status, "Postponed")
        self.assertEqual(opening.official_local_date, date(2026, 4, 4))
        self.assertEqual(postponed.official_local_date, date(2026, 4, 5))

    def test_doubleheader_game_two_remains_provider_scoped_and_distinct(
        self,
    ) -> None:
        _, observations, _ = run_fixture()
        opening, _, game_two = observations

        self.assertEqual(game_two.game_number, 2)
        self.assertEqual(game_two.provider_game_id, "777002")
        self.assertNotEqual(game_two.observation_id, opening.observation_id)
        self.assertIsNone(game_two.supersedes_observation_id)

    def test_contract_has_no_identity_or_prediction_behavior(self) -> None:
        fields = set(ScheduleSourceObservation.__dataclass_fields__)

        self.assertTrue(
            {
                "match_identity",
                "baseball_game",
                "canonical_status",
                "prediction",
                "prediction_eligible",
                "persistence_id",
            }.isdisjoint(fields)
        )

    def test_two_complete_fixture_runs_are_byte_and_id_identical(self) -> None:
        first_bytes, first_observations, first_payloads = run_fixture()
        second_bytes, second_observations, second_payloads = run_fixture()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_payloads, second_payloads)
        self.assertEqual(
            tuple(item.observation_id for item in first_observations),
            tuple(item.observation_id for item in second_observations),
        )


if __name__ == "__main__":
    unittest.main()
