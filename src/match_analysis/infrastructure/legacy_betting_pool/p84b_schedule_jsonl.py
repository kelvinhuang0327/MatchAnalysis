"""Closed-schema, read-only adapter for the pinned P84B schedule JSONL."""

from datetime import UTC, date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re

from ...application.ports.legacy_schedule_source import (
    LegacyScheduleRow,
    LegacyScheduleSnapshot,
)
from ...baseball.domain.schedule import (
    PROVIDER_NAMESPACE,
    ProviderGameReference,
)


STOP_SCHEMA_MISMATCH = "STOP_MATCHANALYSIS_P84B_SCHEDULE_SCHEMA_MISMATCH"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_GAME_ID_PATTERN = re.compile(r"^mlb_2026_([1-9][0-9]*)$")
_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.(?P<fraction>[0-9]{1,6}))?"
    r"(?P<zone>Z|[+-][0-9]{2}:[0-9]{2})$"
)
_EXPECTED_FIELDS = frozenset(
    {
        "game_id",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "source_trace",
        "collected_at_utc",
    }
)


class P84bScheduleValidationError(ValueError):
    """A fail-closed rejection of untrusted schedule snapshot bytes."""

    def __init__(self, detail: str) -> None:
        self.code = STOP_SCHEMA_MISMATCH
        self.detail = detail
        super().__init__(f"{STOP_SCHEMA_MISMATCH}: {detail}")


def _schema_error(detail: str) -> P84bScheduleValidationError:
    return P84bScheduleValidationError(detail)


def _reject_json_constant(value: str) -> None:
    raise _schema_error(f"non-standard JSON numeric constant: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _schema_error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_explicit_string(
    row: dict[str, object],
    field: str,
) -> str:
    value = row[field]
    if not isinstance(value, str) or not value or value != value.strip():
        raise _schema_error(f"{field} must be an explicit trimmed string")
    return value


def _normalize_collection_marker(value: str) -> str:
    match = _TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise _schema_error(
            "collected_at_utc must be an aware ISO datetime"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _schema_error(
            "collected_at_utc must be a valid datetime"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _schema_error("collected_at_utc must be timezone-aware")
    normalized = parsed.astimezone(UTC)
    fraction = match.group("fraction")
    fractional_suffix = ""
    if fraction is not None:
        fractional_suffix = (
            "." + f"{normalized.microsecond:06d}"[: len(fraction)]
        )
    return (
        normalized.strftime("%Y-%m-%dT%H:%M:%S")
        + fractional_suffix
        + "Z"
    )


class P84bScheduleJsonlSource:
    """Loads only an explicitly supplied path or bytes object."""

    def __init__(
        self,
        source: str | Path | bytes,
        *,
        expected_sha256: str,
    ) -> None:
        if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError("expected_sha256 must be a lowercase SHA-256")
        self._expected_sha256 = expected_sha256
        self._path: Path | None = None
        self._bytes: bytes | None = None
        if isinstance(source, (str, Path)):
            self._path = Path(source)
        elif isinstance(source, bytes):
            self._bytes = source
        else:
            raise TypeError("source must be an explicit path or bytes")

    def _read_bytes(self) -> bytes:
        if self._bytes is not None:
            return self._bytes
        if self._path is None:
            raise RuntimeError("schedule source was not initialized")
        return self._path.read_bytes()

    def load(self) -> LegacyScheduleSnapshot:
        """Hash first, then parse, normalize, and sort without writing."""

        raw = self._read_bytes()
        artifact_sha256 = sha256(raw).hexdigest()
        if artifact_sha256 != self._expected_sha256:
            raise _schema_error(
                "raw artifact SHA-256 does not match the expected value"
            )
        if not raw:
            raise _schema_error("snapshot is empty")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _schema_error("snapshot is not valid UTF-8") from error
        lines = text.splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise _schema_error("snapshot contains a blank row")

        rows: list[LegacyScheduleRow] = []
        seen_source_ids: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            try:
                parsed = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_object_without_duplicate_keys,
                )
            except json.JSONDecodeError as error:
                raise _schema_error(
                    f"line {line_number} is malformed JSON"
                ) from error
            if not isinstance(parsed, dict):
                raise _schema_error(
                    f"line {line_number} must be a JSON object"
                )
            row = parsed
            if frozenset(row) != _EXPECTED_FIELDS:
                missing = sorted(_EXPECTED_FIELDS - frozenset(row))
                extra = sorted(frozenset(row) - _EXPECTED_FIELDS)
                raise _schema_error(
                    f"closed schema mismatch; missing={missing}, extra={extra}"
                )

            source_game_id = _require_explicit_string(row, "game_id")
            match = _SOURCE_GAME_ID_PATTERN.fullmatch(source_game_id)
            if match is None or str(int(match.group(1))) != match.group(1):
                raise _schema_error(
                    "game_id must losslessly wrap a positive provider ID"
                )
            if source_game_id in seen_source_ids:
                raise _schema_error(
                    f"duplicate source game ID: {source_game_id}"
                )

            game_date = _require_explicit_string(row, "game_date")
            try:
                parsed_date = date.fromisoformat(game_date)
            except ValueError as error:
                raise _schema_error(
                    "game_date must be an ISO calendar date"
                ) from error
            if parsed_date.isoformat() != game_date:
                raise _schema_error(
                    "game_date must use canonical ISO encoding"
                )
            season = row["season"]
            if (
                isinstance(season, bool)
                or not isinstance(season, int)
                or season <= 0
            ):
                raise _schema_error("season must be a positive JSON integer")
            source_home_team = _require_explicit_string(row, "home_team")
            source_away_team = _require_explicit_string(row, "away_team")
            if source_home_team == source_away_team:
                raise _schema_error("home_team and away_team must differ")
            source_trace = _require_explicit_string(row, "source_trace")
            if source_trace != PROVIDER_NAMESPACE:
                raise _schema_error(
                    f"source_trace must be {PROVIDER_NAMESPACE}"
                )
            marker = _normalize_collection_marker(
                _require_explicit_string(row, "collected_at_utc")
            )

            rows.append(
                LegacyScheduleRow(
                    provider_reference=ProviderGameReference(
                        provider_namespace=PROVIDER_NAMESPACE,
                        provider_game_id=match.group(1),
                        source_game_id=source_game_id,
                    ),
                    season=season,
                    game_date=game_date,
                    source_home_team=source_home_team,
                    source_away_team=source_away_team,
                    source_trace=source_trace,
                    legacy_collection_marker_utc=marker,
                )
            )
            seen_source_ids.add(source_game_id)

        rows.sort(
            key=lambda item: (
                item.provider_reference.provider_namespace,
                item.provider_reference.provider_game_id,
                item.provider_reference.source_game_id,
            )
        )
        return LegacyScheduleSnapshot(
            artifact_sha256=artifact_sha256,
            rows=tuple(rows),
        )
