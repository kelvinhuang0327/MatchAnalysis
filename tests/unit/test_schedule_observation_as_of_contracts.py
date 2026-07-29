"""Unit tests for schedule observation as-of availability snapshots."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
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
from match_analysis.baseball.domain.schedule_snapshot import (
    ScheduleObservationAsOfSelection,
    ScheduleObservationAsOfSnapshot,
    compute_schedule_observation_as_of_snapshot_fingerprint,
)


class StubScheduleObservationSource:
    def __init__(self, capture: ScheduleObservationCapture) -> None:
        self._capture = capture

    def capture(self) -> ScheduleObservationCapture:
        return self._capture


_BASE_TIME = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_observation(
    *,
    provider_namespace: str = "MLB_STATS_API",
    provider_game_id: str = "777001",
    game_number: int = 1,
    response_received_at_utc: datetime | None = None,
    ingested_at_utc: datetime | None = None,
    provider_status_code: str = "S",
    provider_detailed_status: str = "Scheduled",
    payload_tag: str = "opening",
):
    """Build a standalone root observation with independently controllable times."""

    response_received_at_utc = response_received_at_utc or _BASE_TIME
    ingested_at_utc = ingested_at_utc or (
        response_received_at_utc + timedelta(seconds=1)
    )
    raw_payload_bytes = (
        f'{{"tag":"{payload_tag}","game_id":"{provider_game_id}",'
        f'"game_number":{game_number}}}'
    ).encode("utf-8")
    capture = ScheduleObservationCapture(
        provider_namespace=provider_namespace,
        provider_game_id=provider_game_id,
        scheduled_start_utc=_BASE_TIME + timedelta(days=1),
        official_local_date=date(2026, 4, 4),
        response_received_at_utc=response_received_at_utc,
        ingested_at_utc=ingested_at_utc,
        provider_status_code=provider_status_code,
        provider_detailed_status=provider_detailed_status,
        game_number=game_number,
        home_provider_participant_id="118",
        away_provider_participant_id="109",
        endpoint_id="mlb_schedule_v1",
        parser_version="matchanalysis_schedule_parser_v1",
        schema_version="schedule_source_observation_v1",
        raw_payload_bytes=raw_payload_bytes,
        raw_payload_sha256=sha256(raw_payload_bytes).hexdigest(),
        supersedes_observation_id=None,
    )
    return capture_schedule_observation(StubScheduleObservationSource(capture))


def make_revision(previous, **overrides):
    """Capture a genuine, validly hash-chained successor to ``previous``."""

    values = {
        "provider_namespace": previous.provider_namespace,
        "provider_game_id": previous.provider_game_id,
        "game_number": previous.game_number,
        "response_received_at_utc": previous.response_received_at_utc
        + timedelta(hours=1),
        "ingested_at_utc": None,
        "provider_status_code": "P",
        "provider_detailed_status": "Postponed",
        "payload_tag": "revision",
    }
    values.update(overrides)
    response_received_at_utc = values["response_received_at_utc"]
    ingested_at_utc = values["ingested_at_utc"] or (
        response_received_at_utc + timedelta(seconds=1)
    )
    raw_payload_bytes = (
        f'{{"tag":"{values["payload_tag"]}",'
        f'"game_id":"{values["provider_game_id"]}",'
        f'"game_number":{values["game_number"]}}}'
    ).encode("utf-8")
    capture = ScheduleObservationCapture(
        provider_namespace=values["provider_namespace"],
        provider_game_id=values["provider_game_id"],
        scheduled_start_utc=_BASE_TIME + timedelta(days=1),
        official_local_date=date(2026, 4, 5),
        response_received_at_utc=response_received_at_utc,
        ingested_at_utc=ingested_at_utc,
        provider_status_code=values["provider_status_code"],
        provider_detailed_status=values["provider_detailed_status"],
        game_number=values["game_number"],
        home_provider_participant_id="118",
        away_provider_participant_id="109",
        endpoint_id="mlb_schedule_v1",
        parser_version="matchanalysis_schedule_parser_v1",
        schema_version="schedule_source_observation_v1",
        raw_payload_bytes=raw_payload_bytes,
        raw_payload_sha256=sha256(raw_payload_bytes).hexdigest(),
        supersedes_observation_id=previous.observation_id,
    )
    return capture_schedule_observation(
        StubScheduleObservationSource(capture), previous
    )


class SelectScheduleObservationsAsOfTests(unittest.TestCase):
    def test_before_all_ingestion_reports_chain_unavailable(self) -> None:
        opening = make_observation()
        revision_set = build_schedule_observation_revision_chains([opening])

        snapshot = select_schedule_observations_as_of(
            revision_set, _BASE_TIME - timedelta(days=1)
        )

        self.assertEqual(snapshot.selected_count, 0)
        self.assertEqual(snapshot.unavailable_count, 1)
        self.assertEqual(
            snapshot.unavailable_chain_keys,
            (("MLB_STATS_API", "777001", 1),),
        )

    def test_exact_ingestion_boundary_is_inclusive(self) -> None:
        opening = make_observation()
        revision_set = build_schedule_observation_revision_chains([opening])

        snapshot = select_schedule_observations_as_of(
            revision_set, opening.ingested_at_utc
        )

        self.assertEqual(snapshot.selected_count, 1)
        self.assertEqual(
            snapshot.selections[0].selected_observation_id,
            opening.observation_id,
        )

    def test_one_second_before_ingestion_is_unavailable(self) -> None:
        opening = make_observation()
        revision_set = build_schedule_observation_revision_chains([opening])

        snapshot = select_schedule_observations_as_of(
            revision_set, opening.ingested_at_utc - timedelta(seconds=1)
        )

        self.assertEqual(snapshot.selected_count, 0)
        self.assertEqual(snapshot.unavailable_count, 1)

    def test_selects_root_before_revision_ingested(self) -> None:
        opening = make_observation()
        postponed = make_revision(opening)
        revision_set = build_schedule_observation_revision_chains(
            [opening, postponed]
        )

        cutoff = postponed.ingested_at_utc - timedelta(seconds=1)
        snapshot = select_schedule_observations_as_of(revision_set, cutoff)

        self.assertEqual(snapshot.selected_count, 1)
        self.assertEqual(
            snapshot.selections[0].selected_observation_id,
            opening.observation_id,
        )

    def test_selects_head_once_revision_ingested(self) -> None:
        opening = make_observation()
        postponed = make_revision(opening)
        revision_set = build_schedule_observation_revision_chains(
            [opening, postponed]
        )

        snapshot = select_schedule_observations_as_of(
            revision_set, postponed.ingested_at_utc
        )

        self.assertEqual(snapshot.selected_count, 1)
        self.assertEqual(
            snapshot.selections[0].selected_observation_id,
            postponed.observation_id,
        )

    def test_never_selects_a_future_ingested_observation(self) -> None:
        opening = make_observation()
        postponed = make_revision(opening)
        revision_set = build_schedule_observation_revision_chains(
            [opening, postponed]
        )

        snapshot = select_schedule_observations_as_of(
            revision_set, postponed.ingested_at_utc - timedelta(microseconds=1)
        )

        selected_ids = {
            selection.selected_observation_id for selection in snapshot.selections
        }
        self.assertNotIn(postponed.observation_id, selected_ids)

    def test_multiple_chains_independent_of_input_order(self) -> None:
        opening = make_observation(provider_game_id="777001", game_number=1)
        postponed = make_revision(opening)
        game_two = make_observation(
            provider_game_id="777002", game_number=2, payload_tag="game-2"
        )
        observations = [opening, postponed, game_two]

        as_of = max(postponed.ingested_at_utc, game_two.ingested_at_utc)

        ordered_set = build_schedule_observation_revision_chains(observations)
        first = select_schedule_observations_as_of(ordered_set, as_of)

        shuffled = observations[:]
        random.Random(7).shuffle(shuffled)
        shuffled_set = build_schedule_observation_revision_chains(shuffled)
        second = select_schedule_observations_as_of(shuffled_set, as_of)

        self.assertEqual(first.snapshot_fingerprint, second.snapshot_fingerprint)
        self.assertEqual(
            tuple(s.selected_observation_id for s in first.selections),
            tuple(s.selected_observation_id for s in second.selections),
        )

    def test_non_utc_as_of_is_normalized_before_selection(self) -> None:
        opening = make_observation()
        revision_set = build_schedule_observation_revision_chains([opening])

        tokyo = timezone(timedelta(hours=9))
        as_of_tokyo = opening.ingested_at_utc.astimezone(tokyo)

        snapshot = select_schedule_observations_as_of(revision_set, as_of_tokyo)

        self.assertEqual(snapshot.as_of_utc.utcoffset(), timedelta(0))
        self.assertEqual(snapshot.selected_count, 1)

    def test_naive_as_of_is_rejected(self) -> None:
        opening = make_observation()
        revision_set = build_schedule_observation_revision_chains([opening])

        with self.assertRaises(ValueError):
            select_schedule_observations_as_of(
                revision_set, datetime(2026, 3, 1, 12, 0, 0)
            )

    def test_non_increasing_ingestion_along_a_chain_fails_closed(self) -> None:
        opening = make_observation(
            response_received_at_utc=_BASE_TIME,
            ingested_at_utc=_BASE_TIME + timedelta(days=10),
        )
        revision = make_revision(
            opening,
            response_received_at_utc=_BASE_TIME + timedelta(hours=1),
            ingested_at_utc=_BASE_TIME + timedelta(hours=1, seconds=1),
        )
        self.assertLess(revision.ingested_at_utc, opening.ingested_at_utc)

        revision_set = build_schedule_observation_revision_chains(
            [opening, revision]
        )

        with self.assertRaises(ValueError):
            select_schedule_observations_as_of(
                revision_set, _BASE_TIME + timedelta(days=20)
            )

    def test_schema_version_is_exact(self) -> None:
        opening = make_observation()
        revision_set = build_schedule_observation_revision_chains([opening])

        snapshot = select_schedule_observations_as_of(
            revision_set, opening.ingested_at_utc
        )

        self.assertEqual(
            snapshot.schema_version, "schedule_observation_as_of_snapshot_v1"
        )

    def test_snapshot_and_selection_are_immutable(self) -> None:
        opening = make_observation()
        revision_set = build_schedule_observation_revision_chains([opening])

        snapshot = select_schedule_observations_as_of(
            revision_set, opening.ingested_at_utc
        )

        with self.assertRaises(FrozenInstanceError):
            snapshot.selected_count = 99
        with self.assertRaises(FrozenInstanceError):
            snapshot.selections[0].game_number = 99


class ScheduleObservationAsOfSelectionContractTests(unittest.TestCase):
    def test_mismatched_provider_namespace_is_rejected(self) -> None:
        opening = make_observation()

        with self.assertRaises(ValueError):
            ScheduleObservationAsOfSelection(
                provider_namespace="OTHER_PROVIDER",
                provider_game_id=opening.provider_game_id,
                game_number=opening.game_number,
                selected_observation=opening,
            )

    def test_mismatched_game_number_is_rejected(self) -> None:
        opening = make_observation()

        with self.assertRaises(ValueError):
            ScheduleObservationAsOfSelection(
                provider_namespace=opening.provider_namespace,
                provider_game_id=opening.provider_game_id,
                game_number=opening.game_number + 1,
                selected_observation=opening,
            )


class ScheduleObservationAsOfSnapshotContractTests(unittest.TestCase):
    def _selection(self, observation) -> ScheduleObservationAsOfSelection:
        return ScheduleObservationAsOfSelection(
            provider_namespace=observation.provider_namespace,
            provider_game_id=observation.provider_game_id,
            game_number=observation.game_number,
            selected_observation=observation,
        )

    def test_wrong_schema_version_is_rejected(self) -> None:
        opening = make_observation()
        selection = self._selection(opening)
        fingerprint = compute_schedule_observation_as_of_snapshot_fingerprint(
            as_of_utc=opening.ingested_at_utc,
            selected_count=1,
            unavailable_count=0,
            selections=(selection,),
            unavailable_chain_keys=(),
        )

        with self.assertRaises(ValueError):
            ScheduleObservationAsOfSnapshot(
                as_of_utc=opening.ingested_at_utc,
                selections=(selection,),
                unavailable_chain_keys=(),
                selected_count=1,
                unavailable_count=0,
                snapshot_fingerprint=fingerprint,
                schema_version="wrong_schema_version",
            )

    def test_tampered_fingerprint_is_rejected(self) -> None:
        opening = make_observation()
        selection = self._selection(opening)

        with self.assertRaises(ValueError):
            ScheduleObservationAsOfSnapshot(
                as_of_utc=opening.ingested_at_utc,
                selections=(selection,),
                unavailable_chain_keys=(),
                selected_count=1,
                unavailable_count=0,
                snapshot_fingerprint="0" * 64,
                schema_version="schedule_observation_as_of_snapshot_v1",
            )

    def test_mismatched_selected_count_is_rejected(self) -> None:
        opening = make_observation()
        selection = self._selection(opening)
        fingerprint = compute_schedule_observation_as_of_snapshot_fingerprint(
            as_of_utc=opening.ingested_at_utc,
            selected_count=1,
            unavailable_count=0,
            selections=(selection,),
            unavailable_chain_keys=(),
        )

        with self.assertRaises(ValueError):
            ScheduleObservationAsOfSnapshot(
                as_of_utc=opening.ingested_at_utc,
                selections=(selection,),
                unavailable_chain_keys=(),
                selected_count=2,
                unavailable_count=0,
                snapshot_fingerprint=fingerprint,
                schema_version="schedule_observation_as_of_snapshot_v1",
            )

    def test_unsorted_selections_are_rejected(self) -> None:
        game_one = make_observation(provider_game_id="777001", game_number=1)
        game_two = make_observation(
            provider_game_id="777002", game_number=2, payload_tag="game-2"
        )
        selection_one = self._selection(game_one)
        selection_two = self._selection(game_two)
        reversed_selections = (selection_two, selection_one)
        fingerprint = compute_schedule_observation_as_of_snapshot_fingerprint(
            as_of_utc=game_two.ingested_at_utc,
            selected_count=2,
            unavailable_count=0,
            selections=reversed_selections,
            unavailable_chain_keys=(),
        )

        with self.assertRaises(ValueError):
            ScheduleObservationAsOfSnapshot(
                as_of_utc=game_two.ingested_at_utc,
                selections=reversed_selections,
                unavailable_chain_keys=(),
                selected_count=2,
                unavailable_count=0,
                snapshot_fingerprint=fingerprint,
                schema_version="schedule_observation_as_of_snapshot_v1",
            )

    def test_duplicate_chain_key_across_selected_and_unavailable_is_rejected(
        self,
    ) -> None:
        opening = make_observation()
        selection = self._selection(opening)
        duplicate_key = (
            opening.provider_namespace,
            opening.provider_game_id,
            opening.game_number,
        )
        fingerprint = compute_schedule_observation_as_of_snapshot_fingerprint(
            as_of_utc=opening.ingested_at_utc,
            selected_count=1,
            unavailable_count=1,
            selections=(selection,),
            unavailable_chain_keys=(duplicate_key,),
        )

        with self.assertRaises(ValueError):
            ScheduleObservationAsOfSnapshot(
                as_of_utc=opening.ingested_at_utc,
                selections=(selection,),
                unavailable_chain_keys=(duplicate_key,),
                selected_count=1,
                unavailable_count=1,
                snapshot_fingerprint=fingerprint,
                schema_version="schedule_observation_as_of_snapshot_v1",
            )

    def test_naive_as_of_utc_is_rejected(self) -> None:
        opening = make_observation()
        selection = self._selection(opening)
        naive_as_of = datetime(2026, 3, 1, 12, 0, 1, 500000)

        with self.assertRaises(ValueError):
            ScheduleObservationAsOfSnapshot(
                as_of_utc=naive_as_of,
                selections=(selection,),
                unavailable_chain_keys=(),
                selected_count=1,
                unavailable_count=0,
                snapshot_fingerprint="0" * 64,
                schema_version="schedule_observation_as_of_snapshot_v1",
            )


if __name__ == "__main__":
    unittest.main()
