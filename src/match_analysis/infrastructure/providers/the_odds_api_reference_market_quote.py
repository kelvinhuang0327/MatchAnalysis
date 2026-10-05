"""Read one exact pregame MLB h2h quote from The Odds API v4."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from typing import Any, Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ...application.ports.reference_market_quote import (
    ReferenceMarketQuoteObservation,
)
from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc


ODDS_API_V4_SOURCE = "the_odds_api_v4"
ODDS_API_V4_SPORT_KEY = "baseball_mlb"
ODDS_API_V4_MARKET_KEY = "h2h"
ODDS_API_V4_ENDPOINT = (
    f"https://api.the-odds-api.com/v4/sports/{ODDS_API_V4_SPORT_KEY}/odds/"
)
ODDS_API_KEY_ENV = "ODDS_API_KEY"
ODDS_API_BOOKMAKER_ENV = "ODDS_API_BOOKMAKER"

Clock = Callable[[], datetime]
HttpGet = Callable[[str], bytes]


def _clock_now() -> datetime:
    return datetime.now(UTC)


def _stdlib_http_get(url: str) -> bytes:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=15) as response:  # noqa: S310 - fixed HTTPS host
        return response.read()


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _canonical_timestamp(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a timezone-aware timestamp")
    try:
        return format_canonical_utc(parse_canonical_utc(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a timezone-aware timestamp") from error


def _decimal_price(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive decimal price")
    try:
        price = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a positive decimal price") from error
    if not price.is_finite() or price <= Decimal("1"):
        raise ValueError(f"{field_name} must be a positive decimal price greater than 1")
    return price


def _price_identity(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _expected_identity(game_identity: Mapping[str, Any]) -> tuple[str, str, str]:
    if not isinstance(game_identity, Mapping):
        raise ValueError("ODDS_API_CANONICAL_GAME_IDENTITY_INVALID")
    if game_identity.get("provider_namespace") != "mlb_official":
        raise ValueError("ODDS_API_CANONICAL_GAME_IDENTITY_INVALID")
    _nonempty_text(game_identity.get("provider_game_id"), "provider_game_id")
    home_team = _nonempty_text(game_identity.get("home_team"), "home_team")
    away_team = _nonempty_text(game_identity.get("away_team"), "away_team")
    if home_team == away_team:
        raise ValueError("ODDS_API_CANONICAL_GAME_IDENTITY_INVALID")
    scheduled_start = _canonical_timestamp(
        game_identity.get("scheduled_start_utc"), "scheduled_start_utc"
    )
    return home_team, away_team, scheduled_start


def _snapshot_id(
    *,
    source_event_id: str,
    bookmaker_key: str,
    provider_observed_at_utc: str,
    home_price: Decimal,
    away_price: Decimal,
) -> str:
    identity = {
        "source_provider": ODDS_API_V4_SOURCE,
        "source_event_id": source_event_id,
        "bookmaker_key": bookmaker_key,
        "market_key": ODDS_API_V4_MARKET_KEY,
        "provider_observed_at_utc": provider_observed_at_utc,
        "home_decimal_odds": _price_identity(home_price),
        "away_decimal_odds": _price_identity(away_price),
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"oddsapi-v4:{sha256(encoded).hexdigest()}"


class TheOddsApiReferenceMarketQuote:
    """Injectable The Odds API adapter implementing ``ReferenceMarketQuote``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        bookmaker_key: str | None = None,
        http_get: HttpGet | None = None,
        clock: Clock = _clock_now,
    ) -> None:
        configured_key = api_key if api_key is not None else os.environ.get(ODDS_API_KEY_ENV)
        if not isinstance(configured_key, str) or not configured_key.strip():
            raise ValueError(f"{ODDS_API_KEY_ENV} is required")
        configured_bookmaker = (
            bookmaker_key
            if bookmaker_key is not None
            else os.environ.get(ODDS_API_BOOKMAKER_ENV, "fanduel")
        )
        self._api_key = configured_key
        self._bookmaker_key = _nonempty_text(
            configured_bookmaker, "bookmaker_key"
        )
        self._http_get = http_get or _stdlib_http_get
        self._clock = clock

    def _request_url(self) -> str:
        query = urlencode(
            (
                ("apiKey", self._api_key),
                ("bookmakers", self._bookmaker_key),
                ("markets", ODDS_API_V4_MARKET_KEY),
                ("oddsFormat", "decimal"),
                ("dateFormat", "iso"),
            )
        )
        return f"{ODDS_API_V4_ENDPOINT}?{query}"

    def get_quote(
        self, game_identity: Mapping[str, Any]
    ) -> ReferenceMarketQuoteObservation:
        expected_home, expected_away, scheduled_start = _expected_identity(game_identity)
        try:
            response_bytes = self._http_get(self._request_url())
        except Exception:
            # Do not expose urllib's request URL, which contains the API key.
            raise RuntimeError("ODDS_API_REQUEST_FAILED") from None
        retrieved_at = format_canonical_utc(self._clock())
        if not isinstance(response_bytes, bytes):
            raise ValueError("ODDS_API_RESPONSE_INVALID: expected response bytes")
        response_sha256 = sha256(response_bytes).hexdigest()
        try:
            events = json.loads(response_bytes.decode("utf-8"), parse_float=Decimal)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("ODDS_API_RESPONSE_INVALID: invalid JSON") from None
        if not isinstance(events, list):
            raise ValueError("ODDS_API_RESPONSE_INVALID: expected an event list")

        matching_events: list[Mapping[str, Any]] = []
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError("ODDS_API_RESPONSE_INVALID: event must be an object")
            if event.get("sport_key") != ODDS_API_V4_SPORT_KEY:
                raise ValueError("ODDS_API_SPORT_MISMATCH")
            source_event_id = _nonempty_text(event.get("id"), "event.id")
            event_home = _nonempty_text(event.get("home_team"), "event.home_team")
            event_away = _nonempty_text(event.get("away_team"), "event.away_team")
            event_start = _canonical_timestamp(
                event.get("commence_time"), "event.commence_time"
            )
            if not isinstance(event.get("bookmakers"), list):
                raise ValueError("ODDS_API_RESPONSE_INVALID: event bookmakers must be a list")
            if (
                event_home == expected_home
                and event_away == expected_away
                and event_start == scheduled_start
            ):
                matching_events.append(event)
        if not matching_events:
            raise ValueError("ODDS_API_CANONICAL_EVENT_NOT_FOUND")
        if len(matching_events) != 1:
            raise ValueError("ODDS_API_CANONICAL_EVENT_AMBIGUOUS")

        event = matching_events[0]
        source_event_id = str(event["id"])
        bookmakers = [
            bookmaker
            for bookmaker in event["bookmakers"]
            if isinstance(bookmaker, Mapping)
            and bookmaker.get("key") == self._bookmaker_key
        ]
        if len(bookmakers) != 1:
            raise ValueError("ODDS_API_BOOKMAKER_NOT_UNIQUE")
        markets = bookmakers[0].get("markets")
        if not isinstance(markets, list):
            raise ValueError("ODDS_API_RESPONSE_INVALID: bookmaker markets must be a list")
        h2h_markets = [
            market
            for market in markets
            if isinstance(market, Mapping)
            and market.get("key") == ODDS_API_V4_MARKET_KEY
        ]
        if len(h2h_markets) != 1:
            raise ValueError("ODDS_API_H2H_MARKET_NOT_UNIQUE")
        market = h2h_markets[0]
        provider_observed_at = _canonical_timestamp(
            market.get("last_update"), "market.last_update"
        )
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != 2:
            raise ValueError("ODDS_API_TWO_SIDED_H2H_REQUIRED")

        prices: dict[str, Decimal] = {}
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                raise ValueError("ODDS_API_OUTCOME_INVALID")
            team_name = outcome.get("name")
            if team_name not in (expected_home, expected_away) or team_name in prices:
                raise ValueError("ODDS_API_OUTCOME_IDENTITY_MISMATCH")
            prices[team_name] = _decimal_price(outcome.get("price"), "outcome.price")
        if set(prices) != {expected_home, expected_away}:
            raise ValueError("ODDS_API_TWO_SIDED_H2H_REQUIRED")

        retrieved_instant = parse_canonical_utc(retrieved_at)
        start_instant = parse_canonical_utc(scheduled_start)
        if retrieved_instant >= start_instant or parse_canonical_utc(
            provider_observed_at
        ) >= start_instant:
            raise ValueError("ODDS_API_POST_START_QUOTE_REJECTED")

        home_price = prices[expected_home]
        away_price = prices[expected_away]
        return ReferenceMarketQuoteObservation.create(
            snapshot_id=_snapshot_id(
                source_event_id=source_event_id,
                bookmaker_key=self._bookmaker_key,
                provider_observed_at_utc=provider_observed_at,
                home_price=home_price,
                away_price=away_price,
            ),
            game_identity=game_identity,
            observed_at_utc=retrieved_at,
            home_decimal_odds=home_price,
            away_decimal_odds=away_price,
            source_provider=ODDS_API_V4_SOURCE,
            source_event_id=source_event_id,
            bookmaker_key=self._bookmaker_key,
            market_key=ODDS_API_V4_MARKET_KEY,
            provider_observed_at_utc=provider_observed_at,
            retrieved_at_utc=retrieved_at,
            raw_response_sha256=response_sha256,
        )


__all__ = (
    "ODDS_API_BOOKMAKER_ENV",
    "ODDS_API_KEY_ENV",
    "ODDS_API_V4_ENDPOINT",
    "ODDS_API_V4_MARKET_KEY",
    "ODDS_API_V4_SOURCE",
    "ODDS_API_V4_SPORT_KEY",
    "TheOddsApiReferenceMarketQuote",
)
