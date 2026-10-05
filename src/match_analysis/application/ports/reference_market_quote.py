"""Injectable read-only reference Moneyline quote port for V1 capture."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Mapping, Protocol

from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc


def _decimal_odds(value: object, *, field_name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be valid decimal odds") from error
    if not result.is_finite() or result <= Decimal("1"):
        raise ValueError(f"{field_name} must be finite decimal odds greater than 1")
    return result


@dataclass(frozen=True, slots=True)
class ReferenceMarketQuoteObservation:
    """One two-sided reference quote tied to the exact canonical game identity."""

    snapshot_id: str
    game_identity_json: str
    observed_at_utc: str
    home_decimal_odds: Decimal
    away_decimal_odds: Decimal
    source_provider: str | None = None
    source_event_id: str | None = None
    bookmaker_key: str | None = None
    market_key: str | None = None
    provider_observed_at_utc: str | None = None
    retrieved_at_utc: str | None = None
    raw_response_sha256: str | None = None

    @classmethod
    def create(
        cls,
        *,
        snapshot_id: str,
        game_identity: Mapping[str, Any],
        observed_at_utc: str,
        home_decimal_odds: object,
        away_decimal_odds: object,
        source_provider: str | None = None,
        source_event_id: str | None = None,
        bookmaker_key: str | None = None,
        market_key: str | None = None,
        provider_observed_at_utc: str | None = None,
        retrieved_at_utc: str | None = None,
        raw_response_sha256: str | None = None,
    ) -> "ReferenceMarketQuoteObservation":
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("snapshot_id must be non-empty")
        instant = format_canonical_utc(parse_canonical_utc(observed_at_utc))
        provider_instant = (
            format_canonical_utc(parse_canonical_utc(provider_observed_at_utc))
            if provider_observed_at_utc is not None
            else None
        )
        retrieved_instant = (
            format_canonical_utc(parse_canonical_utc(retrieved_at_utc))
            if retrieved_at_utc is not None
            else None
        )
        if retrieved_instant is not None and instant != retrieved_instant:
            raise ValueError("observed_at_utc must equal retrieved_at_utc")
        for field_name, value in (
            ("source_provider", source_provider),
            ("source_event_id", source_event_id),
            ("bookmaker_key", bookmaker_key),
            ("market_key", market_key),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be non-empty when provided")
        if raw_response_sha256 is not None and (
            not isinstance(raw_response_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_response_sha256) is None
        ):
            raise ValueError("raw_response_sha256 must be a lowercase SHA-256 hex digest")
        identity_json = json.dumps(
            dict(game_identity), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return cls(
            snapshot_id=snapshot_id,
            game_identity_json=identity_json,
            observed_at_utc=instant,
            home_decimal_odds=_decimal_odds(home_decimal_odds, field_name="home_decimal_odds"),
            away_decimal_odds=_decimal_odds(away_decimal_odds, field_name="away_decimal_odds"),
            source_provider=source_provider,
            source_event_id=source_event_id,
            bookmaker_key=bookmaker_key,
            market_key=market_key,
            provider_observed_at_utc=provider_instant,
            retrieved_at_utc=retrieved_instant,
            raw_response_sha256=raw_response_sha256,
        )

    @property
    def game_identity(self) -> dict[str, Any]:
        return json.loads(self.game_identity_json)


class ReferenceMarketQuote(Protocol):
    """Provide one already observed two-way quote for the supplied frozen game."""

    def get_quote(
        self, game_identity: Mapping[str, Any]
    ) -> ReferenceMarketQuoteObservation: ...


__all__ = ("ReferenceMarketQuote", "ReferenceMarketQuoteObservation")
