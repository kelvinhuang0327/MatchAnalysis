"""Immutable, point-in-time-safe Moneyline feature snapshots."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from ...core.identity import MatchIdentity
from .canonical_utc import format_canonical_utc


MONEYLINE_FEATURE_SNAPSHOT_SCHEMA_VERSION = "p19a.moneyline_feature_snapshot.v1"
MONEYLINE_FEATURE_NAMES = (
    "recent_win_rate_delta",
    "starter_era_delta",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _require_sha256(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256")


def _require_utc(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be normalized to UTC")


def _require_finite_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")


def _canonical_json_bytes(projection: dict[str, Any]) -> bytes:
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
class MoneylineFeatureProvenance:
    """Provenance for one feature field in a pregame snapshot."""

    field_name: str
    source_id: str
    source_kind: str
    observed_as_of_utc: datetime
    source_fingerprint: str

    def __post_init__(self) -> None:
        _require_text(self.field_name, "field_name")
        _require_text(self.source_id, "source_id")
        _require_text(self.source_kind, "source_kind")
        _require_utc(self.observed_as_of_utc, "observed_as_of_utc")
        _require_sha256(self.source_fingerprint, "source_fingerprint")


@dataclass(frozen=True, slots=True)
class MoneylineFeatureSnapshot:
    """Minimum P13 feature set required for deterministic Moneyline inference."""

    identity: MatchIdentity
    provider_namespace: str
    provider_game_id: str
    game_number: int
    source_schedule_observation_id: str
    as_of_utc: datetime
    scheduled_start_utc: datetime
    recent_win_rate_delta: Decimal
    starter_era_delta: Decimal
    feature_provenance: tuple[MoneylineFeatureProvenance, ...]
    schema_version: str = MONEYLINE_FEATURE_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, MatchIdentity):
            raise TypeError("identity must be a MatchIdentity")
        for field_name in ("provider_namespace", "provider_game_id"):
            _require_text(getattr(self, field_name), field_name)
        if isinstance(self.game_number, bool) or not isinstance(self.game_number, int):
            raise TypeError("game_number must be an integer")
        if self.game_number <= 0:
            raise ValueError("game_number must be positive")
        _require_sha256(
            self.source_schedule_observation_id,
            "source_schedule_observation_id",
        )
        _require_utc(self.as_of_utc, "as_of_utc")
        _require_utc(self.scheduled_start_utc, "scheduled_start_utc")
        if self.as_of_utc >= self.scheduled_start_utc:
            raise ValueError("as_of_utc must be before scheduled_start_utc")
        for field_name in MONEYLINE_FEATURE_NAMES:
            _require_finite_decimal(getattr(self, field_name), field_name)
        if self.schema_version != MONEYLINE_FEATURE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unexpected Moneyline feature snapshot schema")
        if not isinstance(self.feature_provenance, tuple):
            raise TypeError("feature_provenance must be a tuple")
        if tuple(item.field_name for item in self.feature_provenance) != MONEYLINE_FEATURE_NAMES:
            raise ValueError(
                "feature_provenance must contain each required feature once in canonical order"
            )
        for provenance in self.feature_provenance:
            if not isinstance(provenance, MoneylineFeatureProvenance):
                raise TypeError("feature_provenance must contain provenance records")
            if provenance.observed_as_of_utc > self.as_of_utc:
                raise ValueError("feature provenance cannot be observed after snapshot as_of_utc")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        identity: MatchIdentity,
        provider_namespace: str,
        provider_game_id: str,
        game_number: int,
        source_schedule_observation_id: str,
        as_of_utc: datetime,
        scheduled_start_utc: datetime,
        feature_provenance: tuple[MoneylineFeatureProvenance, ...],
    ) -> "MoneylineFeatureSnapshot":
        """Build from an explicit feature mapping; unrelated fields are ignored."""

        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        values: dict[str, Decimal] = {}
        for field_name in MONEYLINE_FEATURE_NAMES:
            if field_name not in record:
                raise ValueError(f"missing required feature: {field_name}")
            try:
                values[field_name] = Decimal(str(record[field_name]))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ValueError(f"invalid required feature: {field_name}") from exc
        return cls(
            identity=identity,
            provider_namespace=provider_namespace,
            provider_game_id=provider_game_id,
            game_number=game_number,
            source_schedule_observation_id=source_schedule_observation_id,
            as_of_utc=as_of_utc,
            scheduled_start_utc=scheduled_start_utc,
            recent_win_rate_delta=values["recent_win_rate_delta"],
            starter_era_delta=values["starter_era_delta"],
            feature_provenance=feature_provenance,
        )

    def feature_vector(self) -> tuple[Decimal, ...]:
        """Return features in the committed P13 canonical order."""

        return tuple(getattr(self, name) for name in MONEYLINE_FEATURE_NAMES)

    def to_projection(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": {
                "sport": self.identity.sport,
                "league": self.identity.league,
                "season": self.identity.season,
                "canonical_game_id": self.identity.canonical_game_id,
                "home_participant": self.identity.home_participant,
                "away_participant": self.identity.away_participant,
                "game_discriminator": self.identity.game_discriminator,
            },
            "provider_namespace": self.provider_namespace,
            "provider_game_id": self.provider_game_id,
            "game_number": self.game_number,
            "source_schedule_observation_id": self.source_schedule_observation_id,
            "as_of_utc": format_canonical_utc(self.as_of_utc),
            "scheduled_start_utc": format_canonical_utc(self.scheduled_start_utc),
            "features": {
                name: str(getattr(self, name)) for name in MONEYLINE_FEATURE_NAMES
            },
            "feature_provenance": [
                {
                    "field_name": item.field_name,
                    "source_id": item.source_id,
                    "source_kind": item.source_kind,
                    "observed_as_of_utc": format_canonical_utc(
                        item.observed_as_of_utc
                    ),
                    "source_fingerprint": item.source_fingerprint,
                }
                for item in self.feature_provenance
            ],
        }

    def canonical_bytes(self) -> bytes:
        """Serialize the exact snapshot projection deterministically."""

        return _canonical_json_bytes(self.to_projection())

    def fingerprint(self) -> str:
        """Return the stable SHA-256 fingerprint of canonical snapshot bytes."""

        return sha256(self.canonical_bytes()).hexdigest()
