"""Unit tests for prospective provider schedule observations."""

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
from match_analysis.application.use_cases.capture_schedule_observation import (
    capture_schedule_observation,
)
from match_analysis.baseball.domain.schedule_observation import (
    canonical_utc_timestamp,
)


OPENING_PAYLOAD = (
    b'{"gamePk":777001,"gameDate":"2026-04-04T18:10:00Z",'
    b'"gameNumber":1,"status":{"abstractGameState":"Preview",'
    b'"detailedState":"Scheduled"},"teams":{"away":{"team":{"id":109}},'
    b'"home":{"team":{"id":118}}}}'
)
OPENING_RAW_SHA256 = (
    "85d380d36f2b8d131fafab98a661da1d333cfc9cb33bb97d32a2466d167cf5aa"
)
OPENING_OBSERVATION_ID = (
    "d7f5a5b129e340a8deca33805bb872dee94380cf9701aed96785d84ce991f0af"
)


class StubScheduleObservationSource:
    def __init__(self, capture: ScheduleObservationCapture) -> None:
        self._capture = capture

    def capture(self) -> ScheduleObservationCapture:
        return self._capture


def make_capture(**overrides: object) -> ScheduleObservationCapture:
    values: dict[str, object] = {
        "provider_namespace": "MLB_STATS_API",
        "provider_game_id": "777001",
        "scheduled_start_utc": datetime(
            2026, 4, 4, 18, 10, tzinfo=timezone.utc
        ),
        "official_local_date": date(2026, 4, 4),
        "response_received_at_utc": datetime(
            2026, 3, 1, 12, 0, 1, 250000, tzinfo=timezone.utc
        ),
        "ingested_at_utc": datetime(
            2026, 3, 1, 12, 0, 1, 500000, tzinfo=timezone.utc
        ),
        "provider_status_code": "S",
        "provider_detailed_status": "Scheduled",
        "game_number": 1,
        "home_provider_participant_id": "118",
        "away_provider_participant_id": "109",
        "endpoint_id": "mlb_schedule_v1",
        "parser_version": "matchanalysis_schedule_parser_v1",
        "schema_version": "schedule_source_observation_v1",
        "raw_payload_bytes": OPENING_PAYLOAD,
        "raw_payload_sha256": OPENING_RAW_SHA256,
        "supersedes_observation_id": None,
    }
    values.update(overrides)
    return ScheduleObservationCapture(**values)


def capture_one(
    capture: ScheduleObservationCapture,
    previous_observation=None,
):
    return capture_schedule_observation(
        StubScheduleObservationSource(capture),
        previous_observation,
    )


class ScheduleObservationContractTests(unittest.TestCase):
    def test_capture_and_observation_are_immutable(self) -> None:
        capture = make_capture()
        observation = capture_one(capture)

        with self.assertRaises(FrozenInstanceError):
            capture.provider_game_id = "1"
        with self.assertRaises(FrozenInstanceError):
            observation.provider_game_id = "1"

    def test_required_strings_reject_blank_or_surrounding_whitespace(
        self,
    ) -> None:
        fields = (
            "provider_namespace",
            "provider_game_id",
            "provider_status_code",
            "provider_detailed_status",
            "home_provider_participant_id",
            "away_provider_participant_id",
            "endpoint_id",
            "parser_version",
            "schema_version",
        )
        for field_name in fields:
            for invalid in ("", " value "):
                with self.subTest(field=field_name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        capture_one(make_capture(**{field_name: invalid}))

    def test_game_number_must_be_a_positive_integer(self) -> None:
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    capture_one(make_capture(game_number=invalid))

    def test_provider_participant_ids_must_differ(self) -> None:
        with self.assertRaises(ValueError):
            capture_one(
                make_capture(away_provider_participant_id="118")
            )

    def test_each_timestamp_must_be_timezone_aware(self) -> None:
        fields = (
            "scheduled_start_utc",
            "response_received_at_utc",
            "ingested_at_utc",
        )
        for field_name in fields:
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    capture_one(
                        make_capture(
                            **{
                                field_name: datetime(2026, 3, 1, 12, 0)
                            }
                        )
                    )

    def test_timestamps_are_normalized_to_utc(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        observation = capture_one(
            make_capture(
                scheduled_start_utc=datetime(
                    2026, 4, 4, 13, 10, tzinfo=eastern
                )
            )
        )

        self.assertEqual(
            observation.scheduled_start_utc,
            datetime(2026, 4, 4, 18, 10, tzinfo=timezone.utc),
        )
        self.assertIs(observation.scheduled_start_utc.tzinfo, timezone.utc)

    def test_canonical_timestamp_encoding_has_exact_fraction_rules(
        self,
    ) -> None:
        self.assertEqual(
            canonical_utc_timestamp(
                datetime(2026, 3, 1, 12, 0, 2, tzinfo=timezone.utc)
            ),
            "2026-03-01T12:00:02Z",
        )
        self.assertEqual(
            canonical_utc_timestamp(
                datetime(
                    2026,
                    3,
                    1,
                    12,
                    0,
                    1,
                    250000,
                    tzinfo=timezone.utc,
                )
            ),
            "2026-03-01T12:00:01.250000Z",
        )

    def test_response_after_ingestion_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            capture_one(
                make_capture(
                    response_received_at_utc=datetime(
                        2026, 3, 1, 12, 0, 2, tzinfo=timezone.utc
                    )
                )
            )

    def test_empty_payload_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            capture_one(
                make_capture(
                    raw_payload_bytes=b"",
                    raw_payload_sha256=sha256(b"").hexdigest(),
                )
            )

    def test_malformed_or_mismatched_raw_hash_is_rejected(self) -> None:
        for invalid in ("not-a-hash", "0" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    capture_one(make_capture(raw_payload_sha256=invalid))

    def test_deterministic_observation_id_matches_independent_reference(
        self,
    ) -> None:
        first = capture_one(make_capture())
        second = capture_one(make_capture())

        self.assertEqual(first.observation_id, OPENING_OBSERVATION_ID)
        self.assertEqual(second.observation_id, first.observation_id)

    def test_changed_raw_payload_changes_observation_id(self) -> None:
        changed_payload = OPENING_PAYLOAD + b"\n"
        changed = capture_one(
            make_capture(
                raw_payload_bytes=changed_payload,
                raw_payload_sha256=sha256(changed_payload).hexdigest(),
            )
        )

        self.assertNotEqual(changed.observation_id, OPENING_OBSERVATION_ID)

    def test_self_supersession_is_rejected(self) -> None:
        opening = capture_one(make_capture())

        with self.assertRaisesRegex(ValueError, "cannot supersede itself"):
            replace(
                opening,
                observation_id="0" * 64,
                supersedes_observation_id="0" * 64,
            )


class ScheduleObservationRevisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.opening = capture_one(make_capture())

    def revision_capture(self, **overrides: object):
        values: dict[str, object] = {
            "scheduled_start_utc": datetime(
                2026, 4, 5, 20, 10, tzinfo=timezone.utc
            ),
            "official_local_date": date(2026, 4, 5),
            "response_received_at_utc": datetime(
                2026,
                4,
                4,
                16,
                30,
                0,
                125000,
                tzinfo=timezone.utc,
            ),
            "ingested_at_utc": datetime(
                2026,
                4,
                4,
                16,
                30,
                0,
                375000,
                tzinfo=timezone.utc,
            ),
            "provider_status_code": "P",
            "provider_detailed_status": "Postponed",
            "raw_payload_bytes": b'{"revision":"postponed"}',
            "supersedes_observation_id": self.opening.observation_id,
        }
        values.update(overrides)
        payload = values["raw_payload_bytes"]
        values.setdefault("raw_payload_sha256", sha256(payload).hexdigest())
        return make_capture(**values)

    def test_orphan_supersession_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            capture_one(self.revision_capture())

    def test_non_revision_rejects_unexpected_previous_observation(self) -> None:
        with self.assertRaises(ValueError):
            capture_one(make_capture(), self.opening)

    def test_wrong_superseded_observation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            capture_one(
                self.revision_capture(supersedes_observation_id="0" * 64),
                self.opening,
            )

    def test_revision_provider_scope_must_match(self) -> None:
        for overrides in (
            {"provider_namespace": "OTHER_PROVIDER"},
            {"provider_game_id": "777999"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    capture_one(
                        self.revision_capture(**overrides),
                        self.opening,
                    )

    def test_revision_game_number_must_match(self) -> None:
        with self.assertRaises(ValueError):
            capture_one(
                self.revision_capture(game_number=2),
                self.opening,
            )

    def test_revision_response_time_must_strictly_increase(self) -> None:
        for response_time in (
            self.opening.response_received_at_utc,
            self.opening.response_received_at_utc - timedelta(microseconds=1),
        ):
            with self.subTest(response_time=response_time):
                with self.assertRaises(ValueError):
                    capture_one(
                        self.revision_capture(
                            response_received_at_utc=response_time,
                        ),
                        self.opening,
                    )

    def test_valid_revision_is_append_only_and_explicit(self) -> None:
        revision = capture_one(
            self.revision_capture(),
            self.opening,
        )

        self.assertEqual(
            revision.supersedes_observation_id,
            self.opening.observation_id,
        )
        self.assertNotEqual(revision.observation_id, self.opening.observation_id)
        self.assertEqual(self.opening.provider_detailed_status, "Scheduled")
        self.assertEqual(revision.provider_detailed_status, "Postponed")


if __name__ == "__main__":
    unittest.main()
