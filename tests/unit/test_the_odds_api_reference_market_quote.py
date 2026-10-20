from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import os
from urllib.parse import parse_qs, urlparse
import unittest

from match_analysis.application.ports.reference_market_quote import (
    ReferenceMarketQuoteObservation,
)
from match_analysis.infrastructure.providers.the_odds_api_reference_market_quote import (
    ODDS_API_V4_ENDPOINT,
    TheOddsApiReferenceMarketQuote,
)


START = "2026-10-02T18:00:00Z"
PROVIDER_UPDATE = "2026-10-02T12:07:00Z"
RETRIEVED = "2026-10-02T12:08:03Z"
API_KEY = "fixture-secret-do-not-persist"
GAME_IDENTITY = {
    "provider_namespace": "mlb_official",
    "provider_game_id": "9001",
    "game_pk": 9001,
    "game_number": 1,
    "game_type": "R",
    "scheduled_start_utc": START,
    "official_date": "2026-10-02",
    "home_team": "Detroit Tigers",
    "away_team": "Cleveland Guardians",
    "home_team_code": "DET",
    "away_team_code": "CLE",
}


def _event(
    *,
    event_id: str = "odds-event-77",
    home_team: str = "Detroit Tigers",
    away_team: str = "Cleveland Guardians",
    commence_time: str = START,
    last_update: str = PROVIDER_UPDATE,
    outcomes: list[dict[str, object]] | None = None,
    markets: list[dict[str, object]] | None = None,
    bookmakers: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if outcomes is None:
        outcomes = [
            {"name": "Cleveland Guardians", "price": 2.22},
            {"name": "Detroit Tigers", "price": 1.72},
        ]
    if markets is None:
        markets = [{"key": "h2h", "last_update": last_update, "outcomes": outcomes}]
    if bookmakers is None:
        bookmakers = [{"key": "fanduel", "title": "FanDuel", "markets": markets}]
    return {
        "id": event_id,
        "sport_key": "baseball_mlb",
        "commence_time": commence_time,
        "home_team": home_team,
        "away_team": away_team,
        "bookmakers": bookmakers,
    }


def _adapter(
    payload: object,
    *,
    retrieved_at: str = RETRIEVED,
    bookmaker_key: str = "fanduel",
    transport: object | None = None,
) -> TheOddsApiReferenceMarketQuote:
    response = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    fixed_clock = lambda: datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
    return TheOddsApiReferenceMarketQuote(
        api_key=API_KEY,
        bookmaker_key=bookmaker_key,
        http_get=transport or (lambda _url: response),  # type: ignore[arg-type]
        clock=fixed_clock,
    )


class TheOddsApiReferenceMarketQuoteTests(unittest.TestCase):
    def test_parses_home_and_away_by_identity_and_preserves_two_timestamps(self) -> None:
        response = json.dumps([_event()]).encode("utf-8")
        requested: list[str] = []
        adapter = TheOddsApiReferenceMarketQuote(
            api_key=API_KEY,
            http_get=lambda url: (requested.append(url) or response),
            clock=lambda: datetime.fromisoformat(RETRIEVED.replace("Z", "+00:00")),
        )

        quote = adapter.get_quote(GAME_IDENTITY)

        self.assertIsInstance(quote, ReferenceMarketQuoteObservation)
        self.assertEqual(quote.game_identity, GAME_IDENTITY)
        self.assertEqual(quote.game_identity["provider_game_id"], "9001")
        self.assertEqual(quote.source_provider, "the_odds_api_v4")
        self.assertEqual(quote.source_event_id, "odds-event-77")
        self.assertNotEqual(quote.source_event_id, quote.game_identity["provider_game_id"])
        self.assertEqual(quote.bookmaker_key, "fanduel")
        self.assertEqual(quote.market_key, "h2h")
        self.assertEqual(str(quote.home_decimal_odds), "1.72")
        self.assertEqual(str(quote.away_decimal_odds), "2.22")
        self.assertEqual(quote.provider_observed_at_utc, PROVIDER_UPDATE)
        self.assertEqual(quote.retrieved_at_utc, RETRIEVED)
        self.assertEqual(quote.observed_at_utc, RETRIEVED)
        self.assertEqual(quote.raw_response_sha256, sha256(response).hexdigest())

        query = parse_qs(urlparse(requested[0]).query)
        self.assertEqual(urlparse(requested[0]).scheme, "https")
        self.assertTrue(requested[0].startswith(ODDS_API_V4_ENDPOINT))
        self.assertEqual(query["apiKey"], [API_KEY])
        self.assertEqual(query["bookmakers"], ["fanduel"])
        self.assertEqual(query["markets"], ["h2h"])
        self.assertEqual(query["oddsFormat"], ["decimal"])
        self.assertEqual(query["dateFormat"], ["iso"])
        persisted = json.dumps(
            {
                field: getattr(quote, field)
                for field in quote.__dataclass_fields__
            },
            default=str,
        )
        self.assertNotIn(API_KEY, persisted)

    def test_uses_exact_teams_and_scheduled_start_and_rejects_no_match(self) -> None:
        distractors = [
            _event(home_team="Cleveland Guardians", away_team="Detroit Tigers"),
            _event(commence_time="2026-10-02T19:00:00Z"),
        ]
        self.assertEqual(_adapter([*distractors, _event()]).get_quote(GAME_IDENTITY).source_event_id,
                         "odds-event-77")
        with self.assertRaisesRegex(ValueError, "ODDS_API_CANONICAL_EVENT_NOT_FOUND"):
            _adapter(distractors).get_quote(GAME_IDENTITY)

    def test_rejects_response_event_from_another_sport(self) -> None:
        event = _event()
        event["sport_key"] = "basketball_nba"
        with self.assertRaisesRegex(ValueError, "ODDS_API_SPORT_MISMATCH"):
            _adapter([event]).get_quote(GAME_IDENTITY)

    def test_rejects_ambiguous_exact_event_matches(self) -> None:
        with self.assertRaisesRegex(ValueError, "ODDS_API_CANONICAL_EVENT_AMBIGUOUS"):
            _adapter([_event(event_id="one"), _event(event_id="two")]).get_quote(GAME_IDENTITY)

    def test_rejects_missing_or_duplicate_configured_bookmaker(self) -> None:
        no_bookmaker = _event(bookmakers=[])
        duplicate = _event(bookmakers=[
            {"key": "fanduel", "markets": [{"key": "h2h", "last_update": PROVIDER_UPDATE,
                                                "outcomes": [
                                                    {"name": "Detroit Tigers", "price": 1.72},
                                                    {"name": "Cleveland Guardians", "price": 2.22},
                                                ]}]},
            {"key": "fanduel", "markets": [{"key": "h2h", "last_update": PROVIDER_UPDATE,
                                                "outcomes": [
                                                    {"name": "Detroit Tigers", "price": 1.72},
                                                    {"name": "Cleveland Guardians", "price": 2.22},
                                                ]}]},
        ])
        for payload in ([no_bookmaker], [duplicate]):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "ODDS_API_BOOKMAKER_NOT_UNIQUE"):
                    _adapter(payload).get_quote(GAME_IDENTITY)

    def test_configured_bookmaker_can_override_default(self) -> None:
        event = _event(
            bookmakers=[
                {
                    "key": "draftkings",
                    "markets": _event()["bookmakers"][0]["markets"],
                }
            ]
        )
        quote = _adapter([event], bookmaker_key="draftkings").get_quote(GAME_IDENTITY)
        self.assertEqual(quote.bookmaker_key, "draftkings")

    def test_rejects_missing_duplicate_and_incomplete_h2h_markets(self) -> None:
        cases = (
            (_event(markets=[]), "ODDS_API_H2H_MARKET_NOT_UNIQUE"),
            (_event(markets=[
                {"key": "h2h", "last_update": PROVIDER_UPDATE, "outcomes": []},
                {"key": "h2h", "last_update": PROVIDER_UPDATE, "outcomes": []},
            ]), "ODDS_API_H2H_MARKET_NOT_UNIQUE"),
            (_event(outcomes=[{"name": "Detroit Tigers", "price": 1.72}]),
             "ODDS_API_TWO_SIDED_H2H_REQUIRED"),
            (_event(outcomes=[
                {"name": "Detroit Tigers", "price": 1.72},
                {"name": "Detroit Tigers", "price": 2.22},
            ]), "ODDS_API_OUTCOME_IDENTITY_MISMATCH"),
            (_event(outcomes=[
                {"name": "Detroit Tigers", "price": 1.0},
                {"name": "Cleveland Guardians", "price": 2.22},
            ]), "outcome.price must be a positive decimal price greater than 1"),
        )
        for event, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    _adapter([event]).get_quote(GAME_IDENTITY)

    def test_rejects_post_start_retrieval_and_provider_update(self) -> None:
        with self.assertRaisesRegex(ValueError, "ODDS_API_POST_START_QUOTE_REJECTED"):
            _adapter([_event()], retrieved_at="2026-10-02T18:00:00Z").get_quote(GAME_IDENTITY)
        with self.assertRaisesRegex(ValueError, "ODDS_API_POST_START_QUOTE_REJECTED"):
            _adapter([_event(last_update="2026-10-02T18:00:00Z")]).get_quote(GAME_IDENTITY)

    def test_snapshot_identity_is_stable_and_changes_with_quote(self) -> None:
        first = _adapter([_event()]).get_quote(GAME_IDENTITY)
        same = _adapter([_event()]).get_quote(GAME_IDENTITY)
        changed = _adapter([_event(outcomes=[
            {"name": "Cleveland Guardians", "price": 2.20},
            {"name": "Detroit Tigers", "price": 1.74},
        ])]).get_quote(GAME_IDENTITY)
        self.assertEqual(first.snapshot_id, same.snapshot_id)
        self.assertNotEqual(first.snapshot_id, changed.snapshot_id)

    def test_config_uses_environment_and_defaults_to_fanduel_without_persisting_key(self) -> None:
        previous_key = os.environ.get("ODDS_API_KEY")
        previous_bookmaker = os.environ.get("ODDS_API_BOOKMAKER")
        try:
            os.environ["ODDS_API_KEY"] = API_KEY
            os.environ.pop("ODDS_API_BOOKMAKER", None)
            adapter = TheOddsApiReferenceMarketQuote(
                http_get=lambda _url: json.dumps([_event()]).encode("utf-8"),
                clock=lambda: datetime.fromisoformat(RETRIEVED.replace("Z", "+00:00")),
            )
            quote = adapter.get_quote(GAME_IDENTITY)
            observation = json.dumps({
                "snapshot_id": quote.snapshot_id,
                "observation": {
                    field: getattr(quote, field)
                    for field in quote.__dataclass_fields__
                },
            }, default=str)
            self.assertEqual(quote.bookmaker_key, "fanduel")
            self.assertNotIn(API_KEY, observation)
        finally:
            if previous_key is None:
                os.environ.pop("ODDS_API_KEY", None)
            else:
                os.environ["ODDS_API_KEY"] = previous_key
            if previous_bookmaker is None:
                os.environ.pop("ODDS_API_BOOKMAKER", None)
            else:
                os.environ["ODDS_API_BOOKMAKER"] = previous_bookmaker

    def test_request_failure_does_not_expose_api_key(self) -> None:
        def fail(_url: str) -> bytes:
            raise OSError(f"request failed with key {API_KEY}")

        with self.assertRaisesRegex(RuntimeError, "ODDS_API_REQUEST_FAILED") as raised:
            TheOddsApiReferenceMarketQuote(api_key=API_KEY, http_get=fail).get_quote(GAME_IDENTITY)
        self.assertNotIn(API_KEY, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
