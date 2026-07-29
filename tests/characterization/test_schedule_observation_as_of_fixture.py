"""Characterize schedule observation as-of snapshots against the tracked fixture."""

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
from match_analysis.application.use_cases.select_schedule_observations_as_of import (
    select_schedule_observations_as_of,
)


FIXTURE_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "mlb_schedule_observation_v1.json"
)

# Pre-edit reference values, independently computed from the tracked fixture
# before any P8 source file existed (see handoff for the reference script).
BEFORE_ALL_INGESTION_AS_OF = "2026-01-01T00:00:00Z"
BEFORE_ALL_INGESTION_FINGERPRINT = (
    "421c9abadbabeadbea6ae72f9e718a45d2676ee4c96a6e3389aab9b9e3bbbef6"
)

AFTER_OPENING_BEFORE_POSTPONED_AS_OF = "2026-03-15T00:00:00Z"
AFTER_OPENING_BEFORE_POSTPONED_FINGERPRINT = (
    "8e87b7a6e3b16a87066faeb500b4af8139448cdaad162078238ee311adfb055c"
)

AFTER_POSTPONED_AS_OF = "2026-05-01T00:00:00Z"
AFTER_POSTPONED_FINGERPRINT = (
    "08a5710ad60565dc8cbf73078e6a8cc59bb409a824e0ecc0a7b1088d5d88be1f"
)

EXACT_BOUNDARY_AS_OF = "2026-03-01T12:00:01.500Z"
EXACT_BOUNDARY_FINGERPRINT = (
    "6679070339fc583744687dfb921cba611bd02c4a5d891862be4998116154d0ee"
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


class ScheduleObservationAsOfFixtureTests(unittest.TestCase):
    def test_before_all_observations_none_selected(self) -> None:
        _, observations = load_fixture_observations()
        revision_set = build_schedule_observation_revision_chains(observations)

        snapshot = select_schedule_observations_as_of(
            revision_set, parse_timestamp(BEFORE_ALL_INGESTION_AS_OF)
        )

        self.assertEqual(snapshot.selected_count, 0)
        self.assertEqual(snapshot.unavailable_count, 2)
        self.assertEqual(
            snapshot.snapshot_fingerprint, BEFORE_ALL_INGESTION_FINGERPRINT
        )

    def test_after_opening_before_postponed_ingestion(self) -> None:
        by_key, observations = load_fixture_observations()
        revision_set = build_schedule_observation_revision_chains(observations)

        snapshot = select_schedule_observations_as_of(
            revision_set,
            parse_timestamp(AFTER_OPENING_BEFORE_POSTPONED_AS_OF),
        )

        self.assertEqual(snapshot.selected_count, 2)
        self.assertEqual(snapshot.unavailable_count, 0)
        selected_ids = {
            selection.selected_observation_id
            for selection in snapshot.selections
        }
        self.assertIn(by_key["opening"].observation_id, selected_ids)
        self.assertIn(
            by_key["doubleheader_game_2"].observation_id, selected_ids
        )
        self.assertEqual(
            snapshot.snapshot_fingerprint,
            AFTER_OPENING_BEFORE_POSTPONED_FINGERPRINT,
        )

    def test_after_postponed_ingestion(self) -> None:
        by_key, observations = load_fixture_observations()
        revision_set = build_schedule_observation_revision_chains(observations)

        snapshot = select_schedule_observations_as_of(
            revision_set, parse_timestamp(AFTER_POSTPONED_AS_OF)
        )

        self.assertEqual(snapshot.selected_count, 2)
        self.assertEqual(snapshot.unavailable_count, 0)
        selected_ids = {
            selection.selected_observation_id
            for selection in snapshot.selections
        }
        self.assertIn(
            by_key["postponed_revision"].observation_id, selected_ids
        )
        self.assertNotIn(by_key["opening"].observation_id, selected_ids)
        self.assertEqual(
            snapshot.snapshot_fingerprint, AFTER_POSTPONED_FINGERPRINT
        )

    def test_exact_boundary_equal_to_ingestion_is_inclusive(self) -> None:
        by_key, observations = load_fixture_observations()
        revision_set = build_schedule_observation_revision_chains(observations)

        snapshot = select_schedule_observations_as_of(
            revision_set, parse_timestamp(EXACT_BOUNDARY_AS_OF)
        )

        self.assertEqual(snapshot.selected_count, 1)
        self.assertEqual(snapshot.unavailable_count, 1)
        self.assertEqual(
            snapshot.selections[0].selected_observation_id,
            by_key["opening"].observation_id,
        )
        self.assertEqual(
            snapshot.unavailable_chain_keys,
            (("MLB_STATS_API", "777002", 2),),
        )
        self.assertEqual(
            snapshot.snapshot_fingerprint, EXACT_BOUNDARY_FINGERPRINT
        )

    def test_shuffled_input_and_repeated_runs_match_reference(self) -> None:
        _, observations = load_fixture_observations()
        as_of = parse_timestamp(AFTER_POSTPONED_AS_OF)

        ordered_set = build_schedule_observation_revision_chains(observations)
        ordered_snapshot = select_schedule_observations_as_of(
            ordered_set, as_of
        )

        shuffled = list(observations)
        random.Random(2026).shuffle(shuffled)
        shuffled_set = build_schedule_observation_revision_chains(shuffled)
        shuffled_snapshot = select_schedule_observations_as_of(
            shuffled_set, as_of
        )

        self.assertEqual(
            ordered_snapshot.snapshot_fingerprint,
            shuffled_snapshot.snapshot_fingerprint,
        )
        self.assertEqual(
            ordered_snapshot.snapshot_fingerprint, AFTER_POSTPONED_FINGERPRINT
        )

        second_run_set = build_schedule_observation_revision_chains(
            observations
        )
        second_run_snapshot = select_schedule_observations_as_of(
            second_run_set, as_of
        )
        self.assertEqual(
            ordered_snapshot.snapshot_fingerprint,
            second_run_snapshot.snapshot_fingerprint,
        )


if __name__ == "__main__":
    unittest.main()
