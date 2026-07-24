"""Adapter parsing one explicit MLB Stats API-shaped schedule payload."""

from datetime import UTC, date, datetime
from hashlib import sha256
import json

from ...application.ports.schedule_observation_source import (
    ScheduleObservationCapture,
)


STOP_SCHEMA_MISMATCH = "STOP_MATCHANALYSIS_MLB_PAYLOAD_SCHEMA_MISMATCH"

_PROVIDER_NAMESPACE = "MLB_STATS_API"


class MlbSchedulePayloadValidationError(ValueError):
    """A fail-closed rejection of an untrusted provider game payload."""

    def __init__(self, detail: str) -> None:
        self.code = STOP_SCHEMA_MISMATCH
        self.detail = detail
        super().__init__(f"{STOP_SCHEMA_MISMATCH}: {detail}")


def _schema_error(detail: str) -> MlbSchedulePayloadValidationError:
    return MlbSchedulePayloadValidationError(detail)


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _schema_error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_field(
    obj: dict[str, object],
    key: str,
    expected_type: type,
    path: str,
) -> object:
    if key not in obj:
        raise _schema_error(f"missing required field: {path}.{key}")
    value = obj[key]
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _schema_error(f"{path}.{key} must be a JSON integer")
    elif not isinstance(value, expected_type):
        raise _schema_error(
            f"{path}.{key} must be of type {expected_type.__name__}"
        )
    return value


def _require_positive_int(
    obj: dict[str, object],
    key: str,
    path: str,
) -> int:
    value = _require_field(obj, key, int, path)
    if value <= 0:
        raise _schema_error(f"{path}.{key} must be a positive integer")
    return value


def _require_non_blank_str(
    obj: dict[str, object],
    key: str,
    path: str,
) -> str:
    value = _require_field(obj, key, str, path)
    if not value.strip():
        raise _schema_error(f"{path}.{key} must not be blank")
    return value


def _parse_utc_datetime(value: str, path: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise _schema_error(f"{path} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _schema_error(f"{path} must be timezone-aware")
    return parsed.astimezone(UTC)


def _extract_participant_id(team_wrapper: object, path: str) -> str:
    if not isinstance(team_wrapper, dict):
        raise _schema_error(f"{path} must be a JSON object")
    inner = _require_field(team_wrapper, "team", dict, path)
    team_id = _require_positive_int(inner, "id", f"{path}.team")
    return str(team_id)


class ExplicitMlbSchedulePayloadSource:
    """Parses exactly one explicitly supplied MLB Stats API game payload."""

    def __init__(
        self,
        *,
        raw_payload_bytes: bytes,
        response_received_at_utc: datetime,
        ingested_at_utc: datetime,
        endpoint_id: str,
        parser_version: str,
        schema_version: str,
        supersedes_observation_id: str | None = None,
    ) -> None:
        if not isinstance(raw_payload_bytes, bytes):
            raise TypeError("raw_payload_bytes must be bytes")
        self._raw_payload_bytes = raw_payload_bytes
        self._response_received_at_utc = response_received_at_utc
        self._ingested_at_utc = ingested_at_utc
        self._endpoint_id = endpoint_id
        self._parser_version = parser_version
        self._schema_version = schema_version
        self._supersedes_observation_id = supersedes_observation_id

    def capture(self) -> ScheduleObservationCapture:
        """Hash first, then parse, validate, and forward without persistence."""

        raw = self._raw_payload_bytes
        if not raw:
            raise _schema_error("raw_payload_bytes must not be empty")
        raw_payload_sha256 = sha256(raw).hexdigest()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _schema_error("raw payload is not valid UTF-8") from error

        try:
            payload = json.loads(
                text, object_pairs_hook=_object_without_duplicate_keys
            )
        except json.JSONDecodeError as error:
            raise _schema_error("raw payload is not valid JSON") from error
        if not isinstance(payload, dict):
            raise _schema_error("payload root must be a JSON object")

        game_pk = _require_positive_int(payload, "gamePk", "payload")
        game_date_raw = _require_non_blank_str(payload, "gameDate", "payload")
        scheduled_start_utc = _parse_utc_datetime(game_date_raw, "payload.gameDate")

        official_date_raw = _require_non_blank_str(
            payload, "officialDate", "payload"
        )
        try:
            official_local_date = date.fromisoformat(official_date_raw)
        except ValueError as error:
            raise _schema_error(
                "payload.officialDate must be an ISO calendar date"
            ) from error

        game_number = _require_positive_int(payload, "gameNumber", "payload")

        status = _require_field(payload, "status", dict, "payload")
        status_code = _require_non_blank_str(status, "statusCode", "payload.status")
        detailed_status = _require_non_blank_str(
            status, "detailedState", "payload.status"
        )

        teams = _require_field(payload, "teams", dict, "payload")
        home_wrapper = _require_field(teams, "home", dict, "payload.teams")
        away_wrapper = _require_field(teams, "away", dict, "payload.teams")
        home_participant_id = _extract_participant_id(
            home_wrapper, "payload.teams.home"
        )
        away_participant_id = _extract_participant_id(
            away_wrapper, "payload.teams.away"
        )
        if home_participant_id == away_participant_id:
            raise _schema_error(
                "home and away provider participant IDs must differ"
            )

        return ScheduleObservationCapture(
            provider_namespace=_PROVIDER_NAMESPACE,
            provider_game_id=str(game_pk),
            scheduled_start_utc=scheduled_start_utc,
            official_local_date=official_local_date,
            response_received_at_utc=self._response_received_at_utc,
            ingested_at_utc=self._ingested_at_utc,
            provider_status_code=status_code,
            provider_detailed_status=detailed_status,
            game_number=game_number,
            home_provider_participant_id=home_participant_id,
            away_provider_participant_id=away_participant_id,
            endpoint_id=self._endpoint_id,
            parser_version=self._parser_version,
            schema_version=self._schema_version,
            raw_payload_bytes=raw,
            raw_payload_sha256=raw_payload_sha256,
            supersedes_observation_id=self._supersedes_observation_id,
        )
