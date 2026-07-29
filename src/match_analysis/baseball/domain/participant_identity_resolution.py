"""Immutable provider-participant identity-resolution contracts."""

from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import json
import re

from .schedule_identity_candidate import (
    ScheduleIdentityResolutionCandidate,
)
from .schedule_observation import canonical_utc_timestamp
from .schedule_snapshot import ChainKey


PROVIDER_PARTICIPANT_IDENTITY_MAPPING_SET_SCHEMA_VERSION = (
    "provider_participant_identity_mapping_set_v1"
)
SCHEDULE_PARTICIPANT_IDENTITY_RESOLUTION_SET_SCHEMA_VERSION = (
    "schedule_participant_identity_resolution_set_v1"
)

MISSING_HOME_PARTICIPANT_MAPPING = "MISSING_HOME_PARTICIPANT_MAPPING"
MISSING_AWAY_PARTICIPANT_MAPPING = "MISSING_AWAY_PARTICIPANT_MAPPING"
CONFLICTING_HOME_PARTICIPANT_MAPPING = (
    "CONFLICTING_HOME_PARTICIPANT_MAPPING"
)
CONFLICTING_AWAY_PARTICIPANT_MAPPING = (
    "CONFLICTING_AWAY_PARTICIPANT_MAPPING"
)
RESOLVED_PARTICIPANTS_NOT_DISTINCT = "RESOLVED_PARTICIPANTS_NOT_DISTINCT"
MAPPING_VERSION_MISMATCH = "MAPPING_VERSION_MISMATCH"

PARTICIPANT_IDENTITY_RESOLUTION_REASON_ORDER = (
    MISSING_HOME_PARTICIPANT_MAPPING,
    MISSING_AWAY_PARTICIPANT_MAPPING,
    CONFLICTING_HOME_PARTICIPANT_MAPPING,
    CONFLICTING_AWAY_PARTICIPANT_MAPPING,
    RESOLVED_PARTICIPANTS_NOT_DISTINCT,
    MAPPING_VERSION_MISMATCH,
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


def _require_game_number(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _candidate_key(
    candidate: (
        "ResolvedScheduleIdentityCandidate"
        | "UnresolvedScheduleIdentityCandidate"
    ),
) -> ChainKey:
    return (
        candidate.provider_namespace,
        candidate.provider_game_id,
        candidate.game_number,
    )


def _validate_source_candidate_fields(
    *,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    scheduled_start_utc: datetime,
    official_local_date: date,
    home_provider_participant_id: str,
    away_provider_participant_id: str,
    source_observation_id: str,
    source_raw_payload_sha256: str,
) -> None:
    ScheduleIdentityResolutionCandidate(
        provider_namespace=provider_namespace,
        provider_game_id=provider_game_id,
        game_number=game_number,
        scheduled_start_utc=scheduled_start_utc,
        official_local_date=official_local_date,
        home_provider_participant_id=home_provider_participant_id,
        away_provider_participant_id=away_provider_participant_id,
        source_observation_id=source_observation_id,
        source_raw_payload_sha256=source_raw_payload_sha256,
    )


@dataclass(frozen=True, slots=True)
class ProviderParticipantIdentityMapping:
    """One explicit provider-participant to canonical-participant mapping."""

    provider_namespace: str
    provider_participant_id: str
    canonical_participant_id: str
    mapping_version: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _require_explicit(getattr(self, field_name), field_name)


def normalize_provider_participant_identity_mappings(
    mappings: tuple[ProviderParticipantIdentityMapping, ...],
) -> tuple[ProviderParticipantIdentityMapping, ...]:
    """Collapse exact duplicates and sort the explicit mapping catalog."""

    if not isinstance(mappings, tuple):
        raise TypeError("mappings must be a tuple")
    if any(
        not isinstance(mapping, ProviderParticipantIdentityMapping)
        for mapping in mappings
    ):
        raise TypeError(
            "every mapping must be a ProviderParticipantIdentityMapping"
        )
    return tuple(
        sorted(
            set(mappings),
            key=lambda mapping: (
                mapping.provider_namespace,
                mapping.provider_participant_id,
                mapping.canonical_participant_id,
                mapping.mapping_version,
            ),
        )
    )


def compute_provider_participant_identity_mapping_set_fingerprint(
    mappings: tuple[ProviderParticipantIdentityMapping, ...],
) -> str:
    """Hash the normalized mapping-set projection plus one final LF."""

    normalized = normalize_provider_participant_identity_mappings(mappings)
    projection = {
        "schema_version": (
            PROVIDER_PARTICIPANT_IDENTITY_MAPPING_SET_SCHEMA_VERSION
        ),
        "mapping_count": len(normalized),
        "mappings": [
            {
                "provider_namespace": mapping.provider_namespace,
                "provider_participant_id": (
                    mapping.provider_participant_id
                ),
                "canonical_participant_id": (
                    mapping.canonical_participant_id
                ),
                "mapping_version": mapping.mapping_version,
            }
            for mapping in normalized
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
class ResolvedScheduleIdentityCandidate:
    """P9 source evidence with two explicitly mapped participant references."""

    provider_namespace: str
    provider_game_id: str
    game_number: int
    scheduled_start_utc: datetime
    official_local_date: date
    home_provider_participant_id: str
    away_provider_participant_id: str
    source_observation_id: str
    source_raw_payload_sha256: str
    home_canonical_participant_id: str
    away_canonical_participant_id: str
    mapping_version: str

    def __post_init__(self) -> None:
        _validate_source_candidate_fields(
            provider_namespace=self.provider_namespace,
            provider_game_id=self.provider_game_id,
            game_number=self.game_number,
            scheduled_start_utc=self.scheduled_start_utc,
            official_local_date=self.official_local_date,
            home_provider_participant_id=(
                self.home_provider_participant_id
            ),
            away_provider_participant_id=(
                self.away_provider_participant_id
            ),
            source_observation_id=self.source_observation_id,
            source_raw_payload_sha256=self.source_raw_payload_sha256,
        )
        for field_name in (
            "home_canonical_participant_id",
            "away_canonical_participant_id",
            "mapping_version",
        ):
            _require_explicit(getattr(self, field_name), field_name)
        if (
            self.home_canonical_participant_id
            == self.away_canonical_participant_id
        ):
            raise ValueError("resolved canonical participant IDs must differ")


@dataclass(frozen=True, slots=True)
class UnresolvedScheduleIdentityCandidate:
    """A provider-scoped candidate with ordered fail-closed reasons."""

    source_observation_id: str
    provider_namespace: str
    provider_game_id: str
    game_number: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.source_observation_id, "source_observation_id")
        _require_explicit(self.provider_namespace, "provider_namespace")
        _require_explicit(self.provider_game_id, "provider_game_id")
        _require_game_number(self.game_number, "game_number")
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple")
        if not self.reasons:
            raise ValueError("reasons must not be empty")
        if any(
            reason not in PARTICIPANT_IDENTITY_RESOLUTION_REASON_ORDER
            for reason in self.reasons
        ):
            raise ValueError("reasons contain an unsupported value")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("reasons must not contain duplicates")
        expected_order = tuple(
            reason
            for reason in PARTICIPANT_IDENTITY_RESOLUTION_REASON_ORDER
            if reason in self.reasons
        )
        if self.reasons != expected_order:
            raise ValueError("reasons must use the controlled order")


def _resolved_candidate_projection(
    candidate: ResolvedScheduleIdentityCandidate,
) -> dict[str, object]:
    return {
        "provider_namespace": candidate.provider_namespace,
        "provider_game_id": candidate.provider_game_id,
        "game_number": candidate.game_number,
        "scheduled_start_utc": canonical_utc_timestamp(
            candidate.scheduled_start_utc
        ),
        "official_local_date": candidate.official_local_date.isoformat(),
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
        "home_canonical_participant_id": (
            candidate.home_canonical_participant_id
        ),
        "away_canonical_participant_id": (
            candidate.away_canonical_participant_id
        ),
        "mapping_version": candidate.mapping_version,
    }


def _unresolved_candidate_projection(
    candidate: UnresolvedScheduleIdentityCandidate,
) -> dict[str, object]:
    return {
        "source_observation_id": candidate.source_observation_id,
        "provider_namespace": candidate.provider_namespace,
        "provider_game_id": candidate.provider_game_id,
        "game_number": candidate.game_number,
        "reasons": list(candidate.reasons),
    }


def compute_schedule_participant_identity_resolution_set_fingerprint(
    *,
    as_of_utc: datetime,
    source_candidate_set_fingerprint: str,
    mapping_set_fingerprint: str,
    resolved_count: int,
    unresolved_count: int,
    unavailable_count: int,
    resolved_candidates: tuple[ResolvedScheduleIdentityCandidate, ...],
    unresolved_candidates: tuple[
        UnresolvedScheduleIdentityCandidate, ...
    ],
    unavailable_chain_keys: tuple[ChainKey, ...],
) -> str:
    """Hash the canonical participant-resolution projection plus one LF."""

    projection = {
        "schema_version": (
            SCHEDULE_PARTICIPANT_IDENTITY_RESOLUTION_SET_SCHEMA_VERSION
        ),
        "as_of_utc": canonical_utc_timestamp(as_of_utc),
        "source_candidate_set_fingerprint": (
            source_candidate_set_fingerprint
        ),
        "mapping_set_fingerprint": mapping_set_fingerprint,
        "resolved_count": resolved_count,
        "unresolved_count": unresolved_count,
        "unavailable_count": unavailable_count,
        "resolved_candidates": [
            _resolved_candidate_projection(candidate)
            for candidate in resolved_candidates
        ],
        "unresolved_candidates": [
            _unresolved_candidate_projection(candidate)
            for candidate in unresolved_candidates
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
class ScheduleParticipantIdentityResolutionSet:
    """Deterministic P9 participant-resolution result without match identity."""

    as_of_utc: datetime
    source_candidate_set_fingerprint: str
    mapping_set_fingerprint: str
    resolved_candidates: tuple[ResolvedScheduleIdentityCandidate, ...]
    unresolved_candidates: tuple[
        UnresolvedScheduleIdentityCandidate, ...
    ]
    unavailable_chain_keys: tuple[ChainKey, ...]
    resolved_count: int
    unresolved_count: int
    unavailable_count: int
    resolution_set_fingerprint: str
    schema_version: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != SCHEDULE_PARTICIPANT_IDENTITY_RESOLUTION_SET_SCHEMA_VERSION
        ):
            raise ValueError(
                "schema_version must be exactly"
                " schedule_participant_identity_resolution_set_v1"
            )
        canonical_utc_timestamp(self.as_of_utc)
        _require_sha256(
            self.source_candidate_set_fingerprint,
            "source_candidate_set_fingerprint",
        )
        _require_sha256(
            self.mapping_set_fingerprint,
            "mapping_set_fingerprint",
        )
        _require_sha256(
            self.resolution_set_fingerprint,
            "resolution_set_fingerprint",
        )
        if not isinstance(self.resolved_candidates, tuple):
            raise TypeError("resolved_candidates must be a tuple")
        if not isinstance(self.unresolved_candidates, tuple):
            raise TypeError("unresolved_candidates must be a tuple")
        if not isinstance(self.unavailable_chain_keys, tuple):
            raise TypeError("unavailable_chain_keys must be a tuple")
        if any(
            not isinstance(candidate, ResolvedScheduleIdentityCandidate)
            for candidate in self.resolved_candidates
        ):
            raise TypeError(
                "every resolved candidate must be a"
                " ResolvedScheduleIdentityCandidate"
            )
        if any(
            not isinstance(candidate, UnresolvedScheduleIdentityCandidate)
            for candidate in self.unresolved_candidates
        ):
            raise TypeError(
                "every unresolved candidate must be an"
                " UnresolvedScheduleIdentityCandidate"
            )

        resolved_keys = [
            _candidate_key(candidate)
            for candidate in self.resolved_candidates
        ]
        unresolved_keys = [
            _candidate_key(candidate)
            for candidate in self.unresolved_candidates
        ]
        if resolved_keys != sorted(resolved_keys):
            raise ValueError("resolved_candidates must preserve P9 ordering")
        if unresolved_keys != sorted(unresolved_keys):
            raise ValueError("unresolved_candidates must preserve P9 ordering")

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
                "unavailable_chain_keys must preserve P9 ordering"
            )

        all_keys = (
            resolved_keys
            + unresolved_keys
            + list(self.unavailable_chain_keys)
        )
        if len(set(all_keys)) != len(all_keys):
            raise ValueError(
                "a chain key must not appear in more than one result partition"
            )

        expected_counts = {
            "resolved_count": len(self.resolved_candidates),
            "unresolved_count": len(self.unresolved_candidates),
            "unavailable_count": len(self.unavailable_chain_keys),
        }
        for field_name, expected in expected_counts.items():
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer"
                )
            if value != expected:
                raise ValueError(
                    f"{field_name} must match its result partition"
                )

        expected_fingerprint = (
            compute_schedule_participant_identity_resolution_set_fingerprint(
                as_of_utc=self.as_of_utc,
                source_candidate_set_fingerprint=(
                    self.source_candidate_set_fingerprint
                ),
                mapping_set_fingerprint=self.mapping_set_fingerprint,
                resolved_count=self.resolved_count,
                unresolved_count=self.unresolved_count,
                unavailable_count=self.unavailable_count,
                resolved_candidates=self.resolved_candidates,
                unresolved_candidates=self.unresolved_candidates,
                unavailable_chain_keys=self.unavailable_chain_keys,
            )
        )
        if self.resolution_set_fingerprint != expected_fingerprint:
            raise ValueError(
                "resolution_set_fingerprint must match the canonical"
                " participant-resolution projection"
            )
