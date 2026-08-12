"""Canonical, provenance-rich Moneyline market snapshot contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from ...core.identity import MatchIdentity


P39A_MARKET_SNAPSHOT_SCHEMA_VERSION = "p39a.moneyline_market_snapshot.v1"
P39A_SELECTION_RULE = (
    "LATEST_TRUSTWORTHY_PREGAME_OBSERVATION_STRICTLY_BEFORE_SCHEDULED_START"
    ";_TIES_BY_SOURCE_ROW_FINGERPRINT"
)
P39A_TIMESTAMP_SEMANTICS = (
    "SOURCE_GAME_TIME_IS_SCHEDULED_START;SOURCE_FETCHED_AT_IS_LOCAL_MARKET"
    "_OBSERVATION_TIME;PROVIDER_SIDE_TIMESTAMP_UNAVAILABLE"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a trimmed timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a trimmed non-empty string")
    return value


def _require_sha256(value: object, *, field_name: str) -> str:
    text = _require_text(value, field_name=field_name)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def _require_price(value: object, *, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value <= Decimal("1"):
        raise ValueError(f"{field_name} must be finite and greater than 1")
    return value


@dataclass(frozen=True, slots=True)
class MoneylineMarketObservationCandidate:
    """One source row retained for an exact P37 identity key.

    Candidates intentionally retain rejected rows.  The application layer can
    then report whether a target failed because of time, identity, market
    shape, or price evidence instead of silently dropping it.
    """

    source_row_index: int
    source_row_fingerprint: str
    source_match_id: str
    source_home_team_name: str
    source_away_team_name: str
    source_home_code: str
    source_away_code: str
    scheduled_start_utc: str
    market_observed_at_utc: str | None
    local_fetched_at_utc: str | None
    provider_observed_at_utc: str | None
    is_pregame: bool | None
    market_code: str
    market_status: str
    rejection_reason: str | None
    home_decimal_price: Decimal | None
    away_decimal_price: Decimal | None

    def __post_init__(self) -> None:
        if isinstance(self.source_row_index, bool) or self.source_row_index <= 0:
            raise ValueError("source_row_index must be positive")
        _require_sha256(self.source_row_fingerprint, field_name="source_row_fingerprint")
        for field_name in (
            "source_match_id",
            "source_home_team_name",
            "source_away_team_name",
            "source_home_code",
            "source_away_code",
            "market_code",
            "market_status",
        ):
            _require_text(getattr(self, field_name), field_name=field_name)
        _parse_timestamp(self.scheduled_start_utc, field_name="scheduled_start_utc")
        if self.market_observed_at_utc is not None:
            _parse_timestamp(
                self.market_observed_at_utc,
                field_name="market_observed_at_utc",
            )
        if self.local_fetched_at_utc is not None:
            _parse_timestamp(self.local_fetched_at_utc, field_name="local_fetched_at_utc")
        if self.provider_observed_at_utc is not None:
            _parse_timestamp(
                self.provider_observed_at_utc,
                field_name="provider_observed_at_utc",
            )
        if self.is_pregame is not None and not isinstance(self.is_pregame, bool):
            raise TypeError("is_pregame must be boolean or None")
        if self.home_decimal_price is not None:
            _require_price(self.home_decimal_price, field_name="home_decimal_price")
        if self.away_decimal_price is not None:
            _require_price(self.away_decimal_price, field_name="away_decimal_price")
        if (self.home_decimal_price is None) != (self.away_decimal_price is None):
            raise ValueError("home and away prices must be present together")


@dataclass(frozen=True, slots=True)
class MoneylineMarketSnapshot:
    """One selected, two-sided, strictly pregame Moneyline observation."""

    snapshot_id: str
    identity: MatchIdentity
    provider_namespace: str
    provider_game_id: str
    game_number: int
    scheduled_start_utc: str
    home_team_code: str
    away_team_code: str
    source_home_team_name: str
    source_away_team_name: str
    source_home_code: str
    source_away_code: str
    source_repository: str
    source_path: str
    source_sha256: str
    source_row_index: int
    source_row_fingerprint: str
    source_match_id: str
    market_code: str
    market_observed_at_utc: str
    local_fetched_at_utc: str
    provider_observed_at_utc: str | None
    home_decimal_price: Decimal
    away_decimal_price: Decimal
    observation_timestamp_semantics: str = P39A_TIMESTAMP_SEMANTICS
    selected_snapshot_rule: str = P39A_SELECTION_RULE

    @classmethod
    def create(cls, **kwargs: Any) -> "MoneylineMarketSnapshot":
        """Build a snapshot and derive its stable identity from its contents."""

        provisional = cls(snapshot_id="__PENDING_SNAPSHOT_ID__", **kwargs)
        return cls(
            snapshot_id=provisional.compute_snapshot_id(),
            **kwargs,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.identity, MatchIdentity):
            raise TypeError("identity must be a MatchIdentity")
        for field_name in (
            "provider_namespace",
            "provider_game_id",
            "home_team_code",
            "away_team_code",
            "source_home_team_name",
            "source_away_team_name",
            "source_home_code",
            "source_away_code",
            "source_repository",
            "source_path",
            "source_match_id",
            "market_code",
            "observation_timestamp_semantics",
            "selected_snapshot_rule",
        ):
            _require_text(getattr(self, field_name), field_name=field_name)
        if self.provider_game_id != self.identity.canonical_game_id:
            raise ValueError("provider_game_id must match canonical identity")
        if isinstance(self.game_number, bool) or not isinstance(self.game_number, int):
            raise TypeError("game_number must be an integer")
        if self.game_number <= 0:
            raise ValueError("game_number must be positive")
        _parse_timestamp(self.scheduled_start_utc, field_name="scheduled_start_utc")
        observed = _parse_timestamp(
            self.market_observed_at_utc,
            field_name="market_observed_at_utc",
        )
        fetched = _parse_timestamp(
            self.local_fetched_at_utc,
            field_name="local_fetched_at_utc",
        )
        scheduled = _parse_timestamp(
            self.scheduled_start_utc,
            field_name="scheduled_start_utc",
        )
        if observed >= scheduled:
            raise ValueError("market observation must be strictly before scheduled start")
        if fetched != observed:
            raise ValueError(
                "local_fetched_at_utc must equal the source observation timestamp"
            )
        if self.provider_observed_at_utc is not None:
            _parse_timestamp(
                self.provider_observed_at_utc,
                field_name="provider_observed_at_utc",
            )
        _require_sha256(self.source_sha256, field_name="source_sha256")
        _require_sha256(self.source_row_fingerprint, field_name="source_row_fingerprint")
        if isinstance(self.source_row_index, bool) or self.source_row_index <= 0:
            raise ValueError("source_row_index must be positive")
        if self.market_code != "MNL":
            raise ValueError("only MNL Moneyline snapshots are supported")
        _require_price(self.home_decimal_price, field_name="home_decimal_price")
        _require_price(self.away_decimal_price, field_name="away_decimal_price")
        expected_id = self.compute_snapshot_id()
        if self.snapshot_id != "__PENDING_SNAPSHOT_ID__" and self.snapshot_id != expected_id:
            raise ValueError("snapshot_id does not match the canonical snapshot projection")

    def _identity_projection(self) -> dict[str, Any]:
        return {
            "schema_version": P39A_MARKET_SNAPSHOT_SCHEMA_VERSION,
            "provider_namespace": self.provider_namespace,
            "provider_game_id": self.provider_game_id,
            "game_number": self.game_number,
            "scheduled_start_utc": self.scheduled_start_utc,
            "home_team_code": self.home_team_code,
            "away_team_code": self.away_team_code,
            "source_home_team_name": self.source_home_team_name,
            "source_away_team_name": self.source_away_team_name,
            "source_home_code": self.source_home_code,
            "source_away_code": self.source_away_code,
            "source_repository": self.source_repository,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_row_index": self.source_row_index,
            "source_row_fingerprint": self.source_row_fingerprint,
            "source_match_id": self.source_match_id,
            "market_code": self.market_code,
            "market_observed_at_utc": self.market_observed_at_utc,
            "local_fetched_at_utc": self.local_fetched_at_utc,
            "provider_observed_at_utc": self.provider_observed_at_utc,
            "home_decimal_price": format(self.home_decimal_price, "f"),
            "away_decimal_price": format(self.away_decimal_price, "f"),
            "observation_timestamp_semantics": self.observation_timestamp_semantics,
            "selected_snapshot_rule": self.selected_snapshot_rule,
            "identity": {
                "sport": self.identity.sport,
                "league": self.identity.league,
                "season": self.identity.season,
                "canonical_game_id": self.identity.canonical_game_id,
                "home_participant": self.identity.home_participant,
                "away_participant": self.identity.away_participant,
                "game_discriminator": self.identity.game_discriminator,
            },
        }

    def compute_snapshot_id(self) -> str:
        return sha256(canonical_json_bytes(self._identity_projection())).hexdigest()

    def to_projection(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, **self._identity_projection()}


__all__ = (
    "MoneylineMarketObservationCandidate",
    "MoneylineMarketSnapshot",
    "P39A_MARKET_SNAPSHOT_SCHEMA_VERSION",
    "P39A_SELECTION_RULE",
    "P39A_TIMESTAMP_SEMANTICS",
    "canonical_json_bytes",
)
