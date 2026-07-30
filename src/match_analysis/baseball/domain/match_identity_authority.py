"""Immutable explicit authority for constructing canonical match identities."""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import re

from ...core.identity import MatchIdentity
from .participant_identity_resolution import (
    ResolvedScheduleIdentityCandidate,
    UnresolvedScheduleIdentityCandidate,
)
from .schedule_observation import canonical_utc_timestamp
from .schedule_snapshot import ChainKey


MATCH_IDENTITY_AUTHORITY_CATALOG_SCHEMA_VERSION = (
    "match_identity_authority_catalog_v1"
)
SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION = (
    "schedule_match_identity_construction_set_v1"
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


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_non_negative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _canonical_json_bytes(projection: dict[str, object]) -> bytes:
    return (
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class MatchIdentityAuthorityEntry:
    """One explicit provider-game key to canonical identity mapping."""

    provider_namespace: str
    provider_game_id: str
    game_number: int
    league: str
    season: int
    canonical_game_id: str
    game_discriminator: str | None
    authority_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "provider_namespace",
            "provider_game_id",
            "league",
            "canonical_game_id",
            "authority_version",
        ):
            _require_explicit(getattr(self, field_name), field_name)
        _require_positive_integer(self.game_number, "game_number")
        _require_positive_integer(self.season, "season")
        if self.game_discriminator is not None:
            _require_explicit(
                self.game_discriminator,
                "game_discriminator",
            )


def _authority_key(
    entry: MatchIdentityAuthorityEntry,
) -> ChainKey:
    return (
        entry.provider_namespace,
        entry.provider_game_id,
        entry.game_number,
    )


def normalize_match_identity_authority_entries(
    entries: tuple[MatchIdentityAuthorityEntry, ...],
) -> tuple[MatchIdentityAuthorityEntry, ...]:
    """Collapse exact duplicates, reject conflicts, and sort the catalog."""

    if not isinstance(entries, tuple):
        raise TypeError("entries must be a tuple")
    if any(
        not isinstance(entry, MatchIdentityAuthorityEntry)
        for entry in entries
    ):
        raise TypeError(
            "every entry must be a MatchIdentityAuthorityEntry"
        )

    entries_by_key: dict[ChainKey, MatchIdentityAuthorityEntry] = {}
    for entry in entries:
        key = _authority_key(entry)
        existing = entries_by_key.get(key)
        if existing is not None and existing != entry:
            raise ValueError(
                "conflicting match identity authority for key"
                f" {key!r}"
            )
        entries_by_key[key] = entry

    return tuple(
        sorted(
            entries_by_key.values(),
            key=lambda entry: (
                entry.provider_namespace,
                entry.provider_game_id,
                entry.game_number,
                entry.canonical_game_id,
                entry.authority_version,
            ),
        )
    )


def _authority_entry_projection(
    entry: MatchIdentityAuthorityEntry,
) -> dict[str, object]:
    return {
        "provider_namespace": entry.provider_namespace,
        "provider_game_id": entry.provider_game_id,
        "game_number": entry.game_number,
        "league": entry.league,
        "season": entry.season,
        "canonical_game_id": entry.canonical_game_id,
        "game_discriminator": entry.game_discriminator,
        "authority_version": entry.authority_version,
    }


def compute_match_identity_authority_catalog_fingerprint(
    entries: tuple[MatchIdentityAuthorityEntry, ...],
) -> str:
    """Hash the normalized authority-catalog projection plus one final LF."""

    normalized = normalize_match_identity_authority_entries(entries)
    projection = {
        "schema_version": (
            MATCH_IDENTITY_AUTHORITY_CATALOG_SCHEMA_VERSION
        ),
        "entry_count": len(normalized),
        "entries": [
            _authority_entry_projection(entry) for entry in normalized
        ],
    }
    return sha256(_canonical_json_bytes(projection)).hexdigest()


@dataclass(frozen=True, slots=True)
class MatchIdentityAuthorityCatalog:
    """A deterministic fail-closed catalog of explicit identity authority."""

    entries: tuple[MatchIdentityAuthorityEntry, ...]
    entry_count: int
    catalog_fingerprint: str
    schema_version: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != MATCH_IDENTITY_AUTHORITY_CATALOG_SCHEMA_VERSION
        ):
            raise ValueError(
                "schema_version must be exactly"
                " match_identity_authority_catalog_v1"
            )
        normalized = normalize_match_identity_authority_entries(
            self.entries
        )
        object.__setattr__(self, "entries", normalized)
        _require_non_negative_integer(self.entry_count, "entry_count")
        if self.entry_count != len(normalized):
            raise ValueError("entry_count must match the normalized entries")
        _require_sha256(
            self.catalog_fingerprint,
            "catalog_fingerprint",
        )
        expected = compute_match_identity_authority_catalog_fingerprint(
            normalized
        )
        if self.catalog_fingerprint != expected:
            raise ValueError(
                "catalog_fingerprint must match the canonical catalog"
                " projection"
            )


def build_match_identity_authority_catalog(
    entries: tuple[MatchIdentityAuthorityEntry, ...],
) -> MatchIdentityAuthorityCatalog:
    """Build one normalized catalog from explicit entries only."""

    normalized = normalize_match_identity_authority_entries(entries)
    return MatchIdentityAuthorityCatalog(
        entries=normalized,
        entry_count=len(normalized),
        catalog_fingerprint=(
            compute_match_identity_authority_catalog_fingerprint(normalized)
        ),
        schema_version=(
            MATCH_IDENTITY_AUTHORITY_CATALOG_SCHEMA_VERSION
        ),
    )


@dataclass(frozen=True, slots=True)
class ConstructedScheduleMatchIdentity:
    """The existing P1 identity with explicit P9/P10 source provenance."""

    match_identity: MatchIdentity
    source_observation_id: str
    source_raw_payload_sha256: str
    source_candidate_set_fingerprint: str
    source_resolution_set_fingerprint: str
    mapping_set_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.match_identity, MatchIdentity):
            raise TypeError("match_identity must be a MatchIdentity")
        if self.match_identity.sport != "baseball":
            raise ValueError(
                "constructed schedule match identity requires sport"
                " exactly 'baseball'"
            )
        for field_name in (
            "source_observation_id",
            "source_raw_payload_sha256",
            "source_candidate_set_fingerprint",
            "source_resolution_set_fingerprint",
            "mapping_set_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)


def _constructed_identity_projection(
    constructed: ConstructedScheduleMatchIdentity,
) -> dict[str, object]:
    identity = constructed.match_identity
    return {
        "sport": identity.sport,
        "league": identity.league,
        "season": identity.season,
        "canonical_game_id": identity.canonical_game_id,
        "home_participant": identity.home_participant,
        "away_participant": identity.away_participant,
        "game_discriminator": identity.game_discriminator,
        "source_observation_id": constructed.source_observation_id,
        "source_raw_payload_sha256": (
            constructed.source_raw_payload_sha256
        ),
        "source_candidate_set_fingerprint": (
            constructed.source_candidate_set_fingerprint
        ),
        "source_resolution_set_fingerprint": (
            constructed.source_resolution_set_fingerprint
        ),
        "mapping_set_fingerprint": constructed.mapping_set_fingerprint,
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


def compute_schedule_match_identity_construction_set_fingerprint(
    *,
    as_of_utc: datetime,
    source_resolution_set_fingerprint: str,
    authority_catalog_fingerprint: str,
    constructed_count: int,
    unresolved_count: int,
    unavailable_count: int,
    authority_missing_count: int,
    constructed_identities: tuple[
        ConstructedScheduleMatchIdentity, ...
    ],
    unresolved_candidates: tuple[
        UnresolvedScheduleIdentityCandidate, ...
    ],
    unavailable_chain_keys: tuple[ChainKey, ...],
    authority_missing_candidates: tuple[
        ResolvedScheduleIdentityCandidate, ...
    ],
) -> str:
    """Hash the canonical construction-set projection plus one final LF."""

    projection = {
        "schema_version": (
            SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION
        ),
        "as_of_utc": canonical_utc_timestamp(as_of_utc),
        "source_resolution_set_fingerprint": (
            source_resolution_set_fingerprint
        ),
        "authority_catalog_fingerprint": (
            authority_catalog_fingerprint
        ),
        "constructed_count": constructed_count,
        "unresolved_count": unresolved_count,
        "unavailable_count": unavailable_count,
        "authority_missing_count": authority_missing_count,
        "constructed_identities": [
            _constructed_identity_projection(constructed)
            for constructed in constructed_identities
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
        "authority_missing_candidates": [
            _resolved_candidate_projection(candidate)
            for candidate in authority_missing_candidates
        ],
    }
    return sha256(_canonical_json_bytes(projection)).hexdigest()


def _candidate_key(
    candidate: (
        ResolvedScheduleIdentityCandidate
        | UnresolvedScheduleIdentityCandidate
    ),
) -> ChainKey:
    return (
        candidate.provider_namespace,
        candidate.provider_game_id,
        candidate.game_number,
    )


@dataclass(frozen=True, slots=True)
class ScheduleMatchIdentityConstructionSet:
    """Deterministic P10-to-P1 construction result."""

    as_of_utc: datetime
    source_resolution_set_fingerprint: str
    authority_catalog_fingerprint: str
    constructed_identities: tuple[
        ConstructedScheduleMatchIdentity, ...
    ]
    unresolved_candidates: tuple[
        UnresolvedScheduleIdentityCandidate, ...
    ]
    unavailable_chain_keys: tuple[ChainKey, ...]
    authority_missing_candidates: tuple[
        ResolvedScheduleIdentityCandidate, ...
    ]
    constructed_count: int
    unresolved_count: int
    unavailable_count: int
    authority_missing_count: int
    construction_set_fingerprint: str
    schema_version: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION
        ):
            raise ValueError(
                "schema_version must be exactly"
                " schedule_match_identity_construction_set_v1"
            )
        canonical_utc_timestamp(self.as_of_utc)
        _require_sha256(
            self.source_resolution_set_fingerprint,
            "source_resolution_set_fingerprint",
        )
        _require_sha256(
            self.authority_catalog_fingerprint,
            "authority_catalog_fingerprint",
        )
        _require_sha256(
            self.construction_set_fingerprint,
            "construction_set_fingerprint",
        )

        partitions = (
            ("constructed_identities", ConstructedScheduleMatchIdentity),
            (
                "unresolved_candidates",
                UnresolvedScheduleIdentityCandidate,
            ),
            (
                "authority_missing_candidates",
                ResolvedScheduleIdentityCandidate,
            ),
        )
        for field_name, expected_type in partitions:
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if any(
                not isinstance(value, expected_type) for value in values
            ):
                raise TypeError(
                    f"every {field_name} value must be a"
                    f" {expected_type.__name__}"
                )
        if not isinstance(self.unavailable_chain_keys, tuple):
            raise TypeError("unavailable_chain_keys must be a tuple")

        for constructed in self.constructed_identities:
            if (
                constructed.source_resolution_set_fingerprint
                != self.source_resolution_set_fingerprint
            ):
                raise ValueError(
                    "constructed identity source resolution fingerprint"
                    " must match the construction set"
                )

        for field_name in (
            "unresolved_candidates",
            "authority_missing_candidates",
        ):
            values = getattr(self, field_name)
            keys = [_candidate_key(candidate) for candidate in values]
            if keys != sorted(keys):
                raise ValueError(
                    f"{field_name} must preserve P10 ordering"
                )

        for key in self.unavailable_chain_keys:
            if not isinstance(key, tuple) or len(key) != 3:
                raise TypeError("every unavailable chain key must be a tuple")
            _require_explicit(key[0], "provider_namespace")
            _require_explicit(key[1], "provider_game_id")
            _require_positive_integer(key[2], "game_number")
        if list(self.unavailable_chain_keys) != sorted(
            self.unavailable_chain_keys
        ):
            raise ValueError(
                "unavailable_chain_keys must preserve P10 ordering"
            )

        expected_counts = {
            "constructed_count": len(self.constructed_identities),
            "unresolved_count": len(self.unresolved_candidates),
            "unavailable_count": len(self.unavailable_chain_keys),
            "authority_missing_count": len(
                self.authority_missing_candidates
            ),
        }
        for field_name, expected in expected_counts.items():
            value = getattr(self, field_name)
            _require_non_negative_integer(value, field_name)
            if value != expected:
                raise ValueError(
                    f"{field_name} must match its result partition"
                )

        candidate_observation_ids = [
            constructed.source_observation_id
            for constructed in self.constructed_identities
        ]
        candidate_observation_ids.extend(
            candidate.source_observation_id
            for candidate in self.unresolved_candidates
        )
        candidate_observation_ids.extend(
            candidate.source_observation_id
            for candidate in self.authority_missing_candidates
        )
        if len(set(candidate_observation_ids)) != len(
            candidate_observation_ids
        ):
            raise ValueError(
                "a source observation must not appear in more than one"
                " candidate result partition"
            )

        expected_fingerprint = (
            compute_schedule_match_identity_construction_set_fingerprint(
                as_of_utc=self.as_of_utc,
                source_resolution_set_fingerprint=(
                    self.source_resolution_set_fingerprint
                ),
                authority_catalog_fingerprint=(
                    self.authority_catalog_fingerprint
                ),
                constructed_count=self.constructed_count,
                unresolved_count=self.unresolved_count,
                unavailable_count=self.unavailable_count,
                authority_missing_count=self.authority_missing_count,
                constructed_identities=self.constructed_identities,
                unresolved_candidates=self.unresolved_candidates,
                unavailable_chain_keys=self.unavailable_chain_keys,
                authority_missing_candidates=(
                    self.authority_missing_candidates
                ),
            )
        )
        if self.construction_set_fingerprint != expected_fingerprint:
            raise ValueError(
                "construction_set_fingerprint must match the canonical"
                " construction-set projection"
            )
