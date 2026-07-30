"""Immutable canonical baseball-game materialization contracts."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import re

from ...core.identity import MatchIdentity
from ...core.time import UtcTimestamp
from .game import BaseballGame
from .participant_identity_resolution import (
    ResolvedScheduleIdentityCandidate,
    UnresolvedScheduleIdentityCandidate,
)
from .schedule_observation import canonical_utc_timestamp
from .schedule_snapshot import ChainKey


SCHEDULE_BASEBALL_GAME_MATERIALIZATION_SET_SCHEMA_VERSION = (
    "schedule_baseball_game_materialization_set_v1"
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256"
        )


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


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


def _require_utc(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be normalized to UTC")


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


def _match_identity_projection(
    identity: MatchIdentity,
) -> dict[str, object]:
    return {
        "sport": identity.sport,
        "league": identity.league,
        "season": identity.season,
        "canonical_game_id": identity.canonical_game_id,
        "home_participant": identity.home_participant,
        "away_participant": identity.away_participant,
        "game_discriminator": identity.game_discriminator,
    }


@dataclass(frozen=True, slots=True)
class ScheduleBaseballGameMaterialization:
    """One existing game bound to its exact P11B identity and provenance."""

    baseball_game: BaseballGame
    match_identity: MatchIdentity
    source_observation_id: str
    source_raw_payload_sha256: str
    source_resolution_set_fingerprint: str
    authority_catalog_fingerprint: str
    source_construction_set_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.baseball_game, BaseballGame):
            raise TypeError("baseball_game must be a BaseballGame")
        if not isinstance(self.match_identity, MatchIdentity):
            raise TypeError("match_identity must be a MatchIdentity")
        if self.baseball_game.identity is not self.match_identity:
            raise ValueError(
                "baseball_game must preserve the exact match_identity object"
            )
        for field_name in (
            "source_observation_id",
            "source_raw_payload_sha256",
            "source_resolution_set_fingerprint",
            "authority_catalog_fingerprint",
            "source_construction_set_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)


def _materialization_projection(
    materialization: ScheduleBaseballGameMaterialization,
) -> dict[str, object]:
    identity_projection = _match_identity_projection(
        materialization.match_identity
    )
    return {
        "baseball_game": {
            "identity": identity_projection,
            "scheduled_start_utc": canonical_utc_timestamp(
                materialization.baseball_game.scheduled_start.value
            ),
        },
        "match_identity": identity_projection,
        "source_observation_id": materialization.source_observation_id,
        "source_raw_payload_sha256": (
            materialization.source_raw_payload_sha256
        ),
        "source_resolution_set_fingerprint": (
            materialization.source_resolution_set_fingerprint
        ),
        "authority_catalog_fingerprint": (
            materialization.authority_catalog_fingerprint
        ),
        "source_construction_set_fingerprint": (
            materialization.source_construction_set_fingerprint
        ),
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


def compute_schedule_baseball_game_materialization_set_fingerprint(
    *,
    as_of_utc: datetime,
    source_resolution_set_fingerprint: str,
    authority_catalog_fingerprint: str,
    source_construction_set_fingerprint: str,
    materialized_count: int,
    unresolved_count: int,
    unavailable_count: int,
    authority_missing_count: int,
    game_materializations: tuple[
        ScheduleBaseballGameMaterialization, ...
    ],
    unresolved_candidates: tuple[
        UnresolvedScheduleIdentityCandidate, ...
    ],
    unavailable_chain_keys: tuple[ChainKey, ...],
    authority_missing_candidates: tuple[
        ResolvedScheduleIdentityCandidate, ...
    ],
) -> str:
    """Hash the public materialization-set projection plus one final LF."""

    projection = {
        "schema_version": (
            SCHEDULE_BASEBALL_GAME_MATERIALIZATION_SET_SCHEMA_VERSION
        ),
        "as_of_utc": canonical_utc_timestamp(as_of_utc),
        "source_resolution_set_fingerprint": (
            source_resolution_set_fingerprint
        ),
        "authority_catalog_fingerprint": authority_catalog_fingerprint,
        "source_construction_set_fingerprint": (
            source_construction_set_fingerprint
        ),
        "materialized_count": materialized_count,
        "unresolved_count": unresolved_count,
        "unavailable_count": unavailable_count,
        "authority_missing_count": authority_missing_count,
        "game_materializations": [
            _materialization_projection(materialization)
            for materialization in game_materializations
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
class ScheduleBaseballGameMaterializationSet:
    """Deterministic P10/P11B-to-existing-BaseballGame result."""

    as_of_utc: datetime
    source_resolution_set_fingerprint: str
    authority_catalog_fingerprint: str
    source_construction_set_fingerprint: str
    game_materializations: tuple[
        ScheduleBaseballGameMaterialization, ...
    ]
    unresolved_candidates: tuple[
        UnresolvedScheduleIdentityCandidate, ...
    ]
    unavailable_chain_keys: tuple[ChainKey, ...]
    authority_missing_candidates: tuple[
        ResolvedScheduleIdentityCandidate, ...
    ]
    materialized_count: int
    unresolved_count: int
    unavailable_count: int
    authority_missing_count: int
    materialization_set_fingerprint: str
    schema_version: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != SCHEDULE_BASEBALL_GAME_MATERIALIZATION_SET_SCHEMA_VERSION
        ):
            raise ValueError(
                "schema_version must be exactly"
                " schedule_baseball_game_materialization_set_v1"
            )
        _require_utc(self.as_of_utc, "as_of_utc")
        for field_name in (
            "source_resolution_set_fingerprint",
            "authority_catalog_fingerprint",
            "source_construction_set_fingerprint",
            "materialization_set_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)

        partitions = (
            (
                "game_materializations",
                ScheduleBaseballGameMaterialization,
            ),
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

        observation_ids = [
            materialization.source_observation_id
            for materialization in self.game_materializations
        ]
        if observation_ids != sorted(observation_ids):
            raise ValueError(
                "game_materializations must be sorted by"
                " source_observation_id"
            )
        for materialization in self.game_materializations:
            expected_provenance = (
                self.source_resolution_set_fingerprint,
                self.authority_catalog_fingerprint,
                self.source_construction_set_fingerprint,
            )
            actual_provenance = (
                materialization.source_resolution_set_fingerprint,
                materialization.authority_catalog_fingerprint,
                materialization.source_construction_set_fingerprint,
            )
            if actual_provenance != expected_provenance:
                raise ValueError(
                    "materialization provenance must match the set"
                )

        for field_name in (
            "unresolved_candidates",
            "authority_missing_candidates",
        ):
            values = getattr(self, field_name)
            keys = [_candidate_key(candidate) for candidate in values]
            if keys != sorted(keys):
                raise ValueError(
                    f"{field_name} must preserve P11B ordering"
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
                "unavailable_chain_keys must preserve P11B ordering"
            )

        all_observation_ids = list(observation_ids)
        all_observation_ids.extend(
            candidate.source_observation_id
            for candidate in self.unresolved_candidates
        )
        all_observation_ids.extend(
            candidate.source_observation_id
            for candidate in self.authority_missing_candidates
        )
        if len(set(all_observation_ids)) != len(all_observation_ids):
            raise ValueError(
                "a source observation must not appear in more than one"
                " materialization partition"
            )

        expected_counts = {
            "materialized_count": len(self.game_materializations),
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

        expected_fingerprint = (
            compute_schedule_baseball_game_materialization_set_fingerprint(
                as_of_utc=self.as_of_utc,
                source_resolution_set_fingerprint=(
                    self.source_resolution_set_fingerprint
                ),
                authority_catalog_fingerprint=(
                    self.authority_catalog_fingerprint
                ),
                source_construction_set_fingerprint=(
                    self.source_construction_set_fingerprint
                ),
                materialized_count=self.materialized_count,
                unresolved_count=self.unresolved_count,
                unavailable_count=self.unavailable_count,
                authority_missing_count=self.authority_missing_count,
                game_materializations=self.game_materializations,
                unresolved_candidates=self.unresolved_candidates,
                unavailable_chain_keys=self.unavailable_chain_keys,
                authority_missing_candidates=(
                    self.authority_missing_candidates
                ),
            )
        )
        if (
            self.materialization_set_fingerprint
            != expected_fingerprint
        ):
            raise ValueError(
                "materialization_set_fingerprint must match the canonical"
                " materialization-set projection"
            )
