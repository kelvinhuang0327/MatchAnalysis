"""Unit tests for the explicit MLB schedule payload adapter."""

from datetime import UTC, date, datetime
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.ports.schedule_observation_source import (
    ScheduleObservationCapture,
)
from match_analysis.infrastructure.mlb_schedule import (
    ExplicitMlbSchedulePayloadSource,
    MlbSchedulePayloadValidationError,
)


VALID_PAYLOAD = {
    "gamePk": 777001,
    "gameDate": "2026-04-04T18:10:00Z",
    "officialDate": "2026-04-04",
    "gameNumber": 1,
    "status": {
        "statusCode": "S",
        "abstractGameState": "Preview",
        "detailedState": "Scheduled",
    },
    "teams": {
        "away": {"team": {"id": 109}},
        "home": {"team": {"id": 118}},
    },
}

RESPONSE_RECEIVED_AT_UTC = datetime(2026, 3, 1, 12, 0, 1, 250000, tzinfo=UTC)
INGESTED_AT_UTC = datetime(2026, 3, 1, 12, 0, 1, 500000, tzinfo=UTC)


def make_source(
    payload: dict[str, object] | None = None,
    *,
    raw_payload_bytes: bytes | None = None,
    supersedes_observation_id: str | None = None,
) -> ExplicitMlbSchedulePayloadSource:
    if raw_payload_bytes is None:
        raw_payload_bytes = json.dumps(
            payload if payload is not None else VALID_PAYLOAD
        ).encode("utf-8")
    return ExplicitMlbSchedulePayloadSource(
        raw_payload_bytes=raw_payload_bytes,
        response_received_at_utc=RESPONSE_RECEIVED_AT_UTC,
        ingested_at_utc=INGESTED_AT_UTC,
        endpoint_id="mlb_schedule_v1",
        parser_version="matchanalysis_mlb_game_payload_parser_v1",
        schema_version="mlb_schedule_api_game_payload_v1",
        supersedes_observation_id=supersedes_observation_id,
    )


class ValidCaptureTests(unittest.TestCase):
    def test_valid_payload_produces_expected_capture(self) -> None:
        capture = make_source().capture()

        self.assertIsInstance(capture, ScheduleObservationCapture)
        self.assertEqual(capture.provider_namespace, "MLB_STATS_API")
        self.assertEqual(capture.provider_game_id, "777001")
        self.assertEqual(
            capture.scheduled_start_utc,
            datetime(2026, 4, 4, 18, 10, 0, tzinfo=UTC),
        )
        self.assertEqual(capture.official_local_date, date(2026, 4, 4))
        self.assertEqual(capture.game_number, 1)
        self.assertEqual(capture.provider_status_code, "S")
        self.assertEqual(capture.provider_detailed_status, "Scheduled")
        self.assertEqual(capture.home_provider_participant_id, "118")
        self.assertEqual(capture.away_provider_participant_id, "109")
        self.assertEqual(capture.endpoint_id, "mlb_schedule_v1")
        self.assertIsNone(capture.supersedes_observation_id)

    def test_repeated_capture_is_deterministic(self) -> None:
        source = make_source()

        first = source.capture()
        second = source.capture()

        self.assertEqual(first, second)

    def test_supersedes_observation_id_is_forwarded_explicitly(self) -> None:
        capture = make_source(
            supersedes_observation_id="a" * 64,
        ).capture()

        self.assertEqual(capture.supersedes_observation_id, "a" * 64)

    def test_extra_provider_fields_are_ignored_by_field_extraction(self) -> None:
        payload = dict(VALID_PAYLOAD)
        payload["seriesDescription"] = "Regular Season"
        payload["doubleHeader"] = "N"

        capture = make_source(payload).capture()

        self.assertEqual(capture.provider_game_id, "777001")

    def test_extra_provider_fields_change_raw_hash_and_observation_id(
        self,
    ) -> None:
        baseline = make_source().capture()
        payload = dict(VALID_PAYLOAD)
        payload["seriesDescription"] = "Regular Season"
        extended = make_source(payload).capture()

        self.assertNotEqual(
            baseline.raw_payload_sha256, extended.raw_payload_sha256
        )
        self.assertNotEqual(baseline.raw_payload_bytes, extended.raw_payload_bytes)


class RejectionTests(unittest.TestCase):
    def test_empty_bytes_are_rejected(self) -> None:
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(raw_payload_bytes=b"").capture()

    def test_invalid_utf8_is_rejected(self) -> None:
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(raw_payload_bytes=b"\xff\xfe\xfa").capture()

    def test_malformed_json_is_rejected(self) -> None:
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(raw_payload_bytes=b"{not json}").capture()

    def test_non_object_root_is_rejected(self) -> None:
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(raw_payload_bytes=b"[1, 2, 3]").capture()

    def test_duplicate_root_key_is_rejected(self) -> None:
        raw = (
            b'{"gamePk":777001,"gamePk":777002,"gameDate":"2026-04-04T18:10:00Z",'
            b'"officialDate":"2026-04-04","gameNumber":1,'
            b'"status":{"statusCode":"S","detailedState":"Scheduled"},'
            b'"teams":{"away":{"team":{"id":109}},"home":{"team":{"id":118}}}}'
        )
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(raw_payload_bytes=raw).capture()

    def test_duplicate_nested_key_is_rejected(self) -> None:
        raw = (
            b'{"gamePk":777001,"gameDate":"2026-04-04T18:10:00Z",'
            b'"officialDate":"2026-04-04","gameNumber":1,'
            b'"status":{"statusCode":"S","statusCode":"P","detailedState":"Scheduled"},'
            b'"teams":{"away":{"team":{"id":109}},"home":{"team":{"id":118}}}}'
        )
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(raw_payload_bytes=raw).capture()

    def test_missing_required_field_is_rejected(self) -> None:
        required_top_level = (
            "gamePk",
            "gameDate",
            "officialDate",
            "gameNumber",
            "status",
            "teams",
        )
        for field in required_top_level:
            with self.subTest(field=field):
                payload = dict(VALID_PAYLOAD)
                del payload[field]
                with self.assertRaises(MlbSchedulePayloadValidationError):
                    make_source(payload).capture()

    def test_missing_nested_status_field_is_rejected(self) -> None:
        for field in ("statusCode", "detailedState"):
            with self.subTest(field=field):
                payload = json.loads(json.dumps(VALID_PAYLOAD))
                del payload["status"][field]
                with self.assertRaises(MlbSchedulePayloadValidationError):
                    make_source(payload).capture()

    def test_missing_nested_team_id_is_rejected(self) -> None:
        for side in ("home", "away"):
            with self.subTest(side=side):
                payload = json.loads(json.dumps(VALID_PAYLOAD))
                del payload["teams"][side]["team"]["id"]
                with self.assertRaises(MlbSchedulePayloadValidationError):
                    make_source(payload).capture()

    def test_incorrect_field_types_are_rejected(self) -> None:
        cases = (
            ("gamePk", "777001"),
            ("gameDate", 20260404),
            ("officialDate", 20260404),
            ("gameNumber", "1"),
            ("status", "Scheduled"),
            ("teams", ["home", "away"]),
        )
        for field, bad_value in cases:
            with self.subTest(field=field):
                payload = dict(VALID_PAYLOAD)
                payload[field] = bad_value
                with self.assertRaises(MlbSchedulePayloadValidationError):
                    make_source(payload).capture()

    def test_nonpositive_or_bool_game_pk_is_rejected(self) -> None:
        for bad_value in (0, -777001, True):
            with self.subTest(bad_value=bad_value):
                payload = dict(VALID_PAYLOAD)
                payload["gamePk"] = bad_value
                with self.assertRaises(MlbSchedulePayloadValidationError):
                    make_source(payload).capture()

    def test_naive_game_date_is_rejected(self) -> None:
        payload = dict(VALID_PAYLOAD)
        payload["gameDate"] = "2026-04-04T18:10:00"
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(payload).capture()

    def test_malformed_game_date_is_rejected(self) -> None:
        payload = dict(VALID_PAYLOAD)
        payload["gameDate"] = "not-a-timestamp"
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(payload).capture()

    def test_malformed_official_date_is_rejected(self) -> None:
        payload = dict(VALID_PAYLOAD)
        payload["officialDate"] = "04/04/2026"
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(payload).capture()

    def test_nonpositive_or_bool_game_number_is_rejected(self) -> None:
        for bad_value in (0, -1, True):
            with self.subTest(bad_value=bad_value):
                payload = dict(VALID_PAYLOAD)
                payload["gameNumber"] = bad_value
                with self.assertRaises(MlbSchedulePayloadValidationError):
                    make_source(payload).capture()

    def test_blank_status_code_is_rejected(self) -> None:
        payload = json.loads(json.dumps(VALID_PAYLOAD))
        payload["status"]["statusCode"] = "   "
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(payload).capture()

    def test_blank_status_detail_is_rejected(self) -> None:
        payload = json.loads(json.dumps(VALID_PAYLOAD))
        payload["status"]["detailedState"] = ""
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(payload).capture()

    def test_nonpositive_participant_id_is_rejected(self) -> None:
        for side in ("home", "away"):
            with self.subTest(side=side):
                payload = json.loads(json.dumps(VALID_PAYLOAD))
                payload["teams"][side]["team"]["id"] = 0
                with self.assertRaises(MlbSchedulePayloadValidationError):
                    make_source(payload).capture()

    def test_identical_participant_ids_are_rejected(self) -> None:
        payload = json.loads(json.dumps(VALID_PAYLOAD))
        payload["teams"]["home"]["team"]["id"] = 109
        payload["teams"]["away"]["team"]["id"] = 109
        with self.assertRaises(MlbSchedulePayloadValidationError):
            make_source(payload).capture()

    def test_raw_payload_bytes_must_be_bytes(self) -> None:
        with self.assertRaises(TypeError):
            ExplicitMlbSchedulePayloadSource(
                raw_payload_bytes="not-bytes",
                response_received_at_utc=RESPONSE_RECEIVED_AT_UTC,
                ingested_at_utc=INGESTED_AT_UTC,
                endpoint_id="mlb_schedule_v1",
                parser_version="matchanalysis_mlb_game_payload_parser_v1",
                schema_version="mlb_schedule_api_game_payload_v1",
            )


class RawHashTests(unittest.TestCase):
    def test_raw_payload_sha256_matches_exact_bytes(self) -> None:
        from hashlib import sha256

        raw_bytes = json.dumps(VALID_PAYLOAD).encode("utf-8")
        capture = make_source(raw_payload_bytes=raw_bytes).capture()

        self.assertEqual(
            capture.raw_payload_sha256, sha256(raw_bytes).hexdigest()
        )
        self.assertEqual(capture.raw_payload_bytes, raw_bytes)


if __name__ == "__main__":
    unittest.main()
