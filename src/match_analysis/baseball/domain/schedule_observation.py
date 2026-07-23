"""Immutable, provider-scoped schedule observation evidence."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import re


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256"
        )


def _require_utc(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be normalized to UTC")


def canonical_utc_timestamp(value: datetime) -> str:
    """Encode a timezone-aware datetime using the observation ID contract."""

    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        return normalized.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_schedule_observation_id(
    *,
    provider_namespace: str,
    provider_game_id: str,
    scheduled_start_utc: datetime,
    official_local_date: date,
    response_received_at_utc: datetime,
    ingested_at_utc: datetime,
    provider_status_code: str,
    provider_detailed_status: str,
    game_number: int,
    home_provider_participant_id: str,
    away_provider_participant_id: str,
    endpoint_id: str,
    parser_version: str,
    schema_version: str,
    raw_payload_sha256: str,
    supersedes_observation_id: str | None,
) -> str:
    """Hash the exact canonical semantic projection plus one final LF."""

    projection = {
        "provider_namespace": provider_namespace,
        "provider_game_id": provider_game_id,
        "scheduled_start_utc": canonical_utc_timestamp(
            scheduled_start_utc
        ),
        "official_local_date": official_local_date.isoformat(),
        "response_received_at_utc": canonical_utc_timestamp(
            response_received_at_utc
        ),
        "ingested_at_utc": canonical_utc_timestamp(ingested_at_utc),
        "provider_status_code": provider_status_code,
        "provider_detailed_status": provider_detailed_status,
        "game_number": game_number,
        "home_provider_participant_id": home_provider_participant_id,
        "away_provider_participant_id": away_provider_participant_id,
        "endpoint_id": endpoint_id,
        "parser_version": parser_version,
        "schema_version": schema_version,
        "raw_payload_sha256": raw_payload_sha256,
        "supersedes_observation_id": supersedes_observation_id,
    }
    canonical_line = (
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return sha256(canonical_line).hexdigest()


@dataclass(frozen=True, slots=True)
class ScheduleSourceObservation:
    """Validated immutable evidence from one provider response."""

    observation_id: str
    provider_namespace: str
    provider_game_id: str
    scheduled_start_utc: datetime
    official_local_date: date
    response_received_at_utc: datetime
    ingested_at_utc: datetime
    provider_status_code: str
    provider_detailed_status: str
    game_number: int
    home_provider_participant_id: str
    away_provider_participant_id: str
    endpoint_id: str
    parser_version: str
    schema_version: str
    raw_payload_bytes: bytes
    raw_payload_sha256: str
    supersedes_observation_id: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "provider_namespace",
            "provider_game_id",
            "provider_status_code",
            "provider_detailed_status",
            "home_provider_participant_id",
            "away_provider_participant_id",
            "endpoint_id",
            "parser_version",
            "schema_version",
        ):
            _require_explicit(getattr(self, field_name), field_name)
        if (
            self.home_provider_participant_id
            == self.away_provider_participant_id
        ):
            raise ValueError("provider participant IDs must differ")
        if (
            isinstance(self.game_number, bool)
            or not isinstance(self.game_number, int)
            or self.game_number <= 0
        ):
            raise ValueError("game_number must be a positive integer")
        for field_name in (
            "scheduled_start_utc",
            "response_received_at_utc",
            "ingested_at_utc",
        ):
            _require_utc(getattr(self, field_name), field_name)
        if type(self.official_local_date) is not date:
            raise TypeError("official_local_date must be a date")
        if self.response_received_at_utc > self.ingested_at_utc:
            raise ValueError(
                "response_received_at_utc must not follow ingested_at_utc"
            )
        if not isinstance(self.raw_payload_bytes, bytes):
            raise TypeError("raw_payload_bytes must be bytes")
        if not self.raw_payload_bytes:
            raise ValueError("raw_payload_bytes must not be empty")
        _require_sha256(self.raw_payload_sha256, "raw_payload_sha256")
        if sha256(self.raw_payload_bytes).hexdigest() != self.raw_payload_sha256:
            raise ValueError("raw_payload_sha256 must match exact payload bytes")
        _require_sha256(self.observation_id, "observation_id")
        if self.supersedes_observation_id is not None:
            _require_sha256(
                self.supersedes_observation_id,
                "supersedes_observation_id",
            )
            if self.supersedes_observation_id == self.observation_id:
                raise ValueError("an observation cannot supersede itself")
        expected_id = compute_schedule_observation_id(
            provider_namespace=self.provider_namespace,
            provider_game_id=self.provider_game_id,
            scheduled_start_utc=self.scheduled_start_utc,
            official_local_date=self.official_local_date,
            response_received_at_utc=self.response_received_at_utc,
            ingested_at_utc=self.ingested_at_utc,
            provider_status_code=self.provider_status_code,
            provider_detailed_status=self.provider_detailed_status,
            game_number=self.game_number,
            home_provider_participant_id=(
                self.home_provider_participant_id
            ),
            away_provider_participant_id=(
                self.away_provider_participant_id
            ),
            endpoint_id=self.endpoint_id,
            parser_version=self.parser_version,
            schema_version=self.schema_version,
            raw_payload_sha256=self.raw_payload_sha256,
            supersedes_observation_id=self.supersedes_observation_id,
        )
        if self.observation_id != expected_id:
            raise ValueError(
                "observation_id must match the canonical observation projection"
            )
