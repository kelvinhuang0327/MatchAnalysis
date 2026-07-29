"""Immutable provider-scoped requests for future schedule identity resolution."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
import json
import re

from .schedule_observation import canonical_utc_timestamp
from .schedule_snapshot import ChainKey


SCHEDULE_IDENTITY_RESOLUTION_CANDIDATE_SET_SCHEMA_VERSION = (
    "schedule_identity_resolution_candidate_set_v1"
)

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


def _require_game_number(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ScheduleIdentityResolutionCandidate:
    """Exact provider evidence submitted to a future identity resolver."""

    provider_namespace: str
    provider_game_id: str
    game_number: int
    scheduled_start_utc: datetime
    official_local_date: date
    home_provider_participant_id: str
    away_provider_participant_id: str
    source_observation_id: str
    source_raw_payload_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "provider_namespace",
            "provider_game_id",
            "home_provider_participant_id",
            "away_provider_participant_id",
        ):
            _require_explicit(getattr(self, field_name), field_name)
        if (
            self.home_provider_participant_id
            == self.away_provider_participant_id
        ):
            raise ValueError("provider participant IDs must differ")
        _require_game_number(self.game_number, "game_number")
        _require_utc(self.scheduled_start_utc, "scheduled_start_utc")
        if type(self.official_local_date) is not date:
            raise TypeError("official_local_date must be a date")
        _require_sha256(self.source_observation_id, "source_observation_id")
        _require_sha256(
            self.source_raw_payload_sha256,
            "source_raw_payload_sha256",
        )


def compute_schedule_identity_resolution_candidate_set_fingerprint(
    *,
    as_of_utc: datetime,
    source_snapshot_fingerprint: str,
    candidate_count: int,
    unavailable_count: int,
    candidates: tuple[ScheduleIdentityResolutionCandidate, ...],
    unavailable_chain_keys: tuple[ChainKey, ...],
) -> str:
    """Hash the exact canonical candidate-set projection plus one final LF."""

    projection = {
        "schema_version": (
            SCHEDULE_IDENTITY_RESOLUTION_CANDIDATE_SET_SCHEMA_VERSION
        ),
        "as_of_utc": canonical_utc_timestamp(as_of_utc),
        "source_snapshot_fingerprint": source_snapshot_fingerprint,
        "candidate_count": candidate_count,
        "unavailable_count": unavailable_count,
        "candidates": [
            {
                "provider_namespace": candidate.provider_namespace,
                "provider_game_id": candidate.provider_game_id,
                "game_number": candidate.game_number,
                "scheduled_start_utc": canonical_utc_timestamp(
                    candidate.scheduled_start_utc
                ),
                "official_local_date": (
                    candidate.official_local_date.isoformat()
                ),
                "home_provider_participant_id": (
                    candidate.home_provider_participant_id
                ),
                "away_provider_participant_id": (
                    candidate.away_provider_participant_id
                ),
                "source_observation_id": candidate.source_observation_id,
                "source_raw_payload_sha256": (
                    candidate.source_raw_payload_sha256
                ),
            }
            for candidate in candidates
        ],
        "unavailable_chain_keys": [
            {
                "provider_namespace": key[0],
                "provider_game_id": key[1],
                "game_number": key[2],
            }
            for key in unavailable_chain_keys
        ],
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
class ScheduleIdentityResolutionCandidateSet:
    """Deterministic projection of one existing P8 as-of snapshot."""

    as_of_utc: datetime
    source_snapshot_fingerprint: str
    candidates: tuple[ScheduleIdentityResolutionCandidate, ...]
    unavailable_chain_keys: tuple[ChainKey, ...]
    candidate_count: int
    unavailable_count: int
    candidate_set_fingerprint: str
    schema_version: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != SCHEDULE_IDENTITY_RESOLUTION_CANDIDATE_SET_SCHEMA_VERSION
        ):
            raise ValueError(
                "schema_version must be exactly"
                " schedule_identity_resolution_candidate_set_v1"
            )
        _require_utc(self.as_of_utc, "as_of_utc")
        _require_sha256(
            self.source_snapshot_fingerprint,
            "source_snapshot_fingerprint",
        )
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple")
        if not isinstance(self.unavailable_chain_keys, tuple):
            raise TypeError("unavailable_chain_keys must be a tuple")
        if any(
            not isinstance(candidate, ScheduleIdentityResolutionCandidate)
            for candidate in self.candidates
        ):
            raise TypeError(
                "every candidate must be a"
                " ScheduleIdentityResolutionCandidate"
            )

        candidate_keys = [
            (
                candidate.provider_namespace,
                candidate.provider_game_id,
                candidate.game_number,
            )
            for candidate in self.candidates
        ]
        if candidate_keys != sorted(candidate_keys):
            raise ValueError(
                "candidates must be sorted by provider_namespace,"
                " provider_game_id, game_number"
            )

        for key in self.unavailable_chain_keys:
            if not isinstance(key, tuple) or len(key) != 3:
                raise TypeError("every unavailable chain key must be a tuple")
            _require_explicit(key[0], "provider_namespace")
            _require_explicit(key[1], "provider_game_id")
            _require_game_number(key[2], "game_number")
        if list(self.unavailable_chain_keys) != sorted(
            self.unavailable_chain_keys
        ):
            raise ValueError(
                "unavailable_chain_keys must be sorted by provider_namespace,"
                " provider_game_id, game_number"
            )

        all_keys = candidate_keys + list(self.unavailable_chain_keys)
        if len(set(all_keys)) != len(all_keys):
            raise ValueError(
                "a chain key must not appear more than once across candidates"
                " and unavailable_chain_keys"
            )
        for field_name in ("candidate_count", "unavailable_count"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer"
                )
        if self.candidate_count != len(self.candidates):
            raise ValueError(
                "candidate_count must match the number of candidates"
            )
        if self.unavailable_count != len(self.unavailable_chain_keys):
            raise ValueError(
                "unavailable_count must match the number of unavailable"
                " chain keys"
            )

        _require_sha256(
            self.candidate_set_fingerprint,
            "candidate_set_fingerprint",
        )
        expected_fingerprint = (
            compute_schedule_identity_resolution_candidate_set_fingerprint(
                as_of_utc=self.as_of_utc,
                source_snapshot_fingerprint=(
                    self.source_snapshot_fingerprint
                ),
                candidate_count=self.candidate_count,
                unavailable_count=self.unavailable_count,
                candidates=self.candidates,
                unavailable_chain_keys=self.unavailable_chain_keys,
            )
        )
        if self.candidate_set_fingerprint != expected_fingerprint:
            raise ValueError(
                "candidate_set_fingerprint must match the canonical"
                " candidate-set projection"
            )
