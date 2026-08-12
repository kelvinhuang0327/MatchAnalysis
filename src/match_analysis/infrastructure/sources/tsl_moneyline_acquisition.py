"""Read-only native acquisition of the TSL Blob3rd pregame Moneyline feed.

The legacy producer used the public Blob3rd JSON endpoints and then persisted
rows through ``data/tsl_snapshot.py``.  This adapter keeps only the smallest
load-bearing slice: one full-game two-way Moneyline market, normalized into
the existing P31A TSL observation shape.  It never writes the historical
authority file and does not contain scheduling or betting policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import json
from hashlib import sha256
import subprocess
from typing import Any
from urllib.request import Request, urlopen

from .tsl_moneyline_history import (
    TSL_LOCAL_TIMEZONE,
    TslMoneylineHistory,
    TslMoneylineObservation,
    canonical_json_bytes,
)


TSL_BLOB3RD_BASE_URL = "https://blob3rd.sportslottery.com.tw/apidata"
TSL_BLOB3RD_SOURCE_LABEL = "TSL_BLOB3RD"
TSL_BLOB3RD_LIVE_URL = f"{TSL_BLOB3RD_BASE_URL}/Live/Games.zh.json"
TSL_BLOB3RD_SPORTS_URL = f"{TSL_BLOB3RD_BASE_URL}/Pre/Sports.zh.json"
TSL_BLOB3RD_PRE_URL_TEMPLATE = (
    f"{TSL_BLOB3RD_BASE_URL}/Pre/{{sport_id}}-Games.{{language}}.json"
)
TSL_ACQUISITION_SCHEMA_VERSION = "p32a.tsl_moneyline_acquisition.v1"

STOP_TSL_ACQUISITION_SOURCE = "STOP_MATCHANALYSIS_P32A_SOURCE_SEMANTICS_UNAVAILABLE"
STOP_TSL_ACQUISITION_SCHEMA = "STOP_MATCHANALYSIS_P32A_TSL_ACQUISITION_SCHEMA"

# This is the legacy producer's source-side Chinese MLB name mapping.  The
# downstream P28AB crosswalk remains authoritative for official MLB identity.
MLB_ZH_TO_TSL_CODE: dict[str, str] = {
    "亞利桑那響尾蛇": "ARI",
    "亞歷桑那響尾蛇": "ARI",
    "亞特蘭大勇士": "ATL",
    "巴爾的摩金鶯": "BAL",
    "波士頓紅襪": "BOS",
    "芝加哥小熊": "CHC",
    "芝加哥白襪": "CWS",
    "辛辛那堤紅人": "CIN",
    "辛辛那提紅人": "CIN",
    "克里夫蘭守護者": "CLE",
    "科羅拉多洛磯": "COL",
    "科羅拉多落磯": "COL",
    "底特律老虎": "DET",
    "休士頓太空人": "HOU",
    "堪薩斯皇家": "KCR",
    "洛杉磯天使": "LAA",
    "洛杉磯道奇": "LAD",
    "邁阿密馬林魚": "MIA",
    "密爾瓦基釀酒人": "MIL",
    "明尼蘇達雙城": "MIN",
    "紐約大都會": "NYM",
    "紐約洋基": "NYY",
    "運動家": "OAK",
    "奧克蘭運動家": "OAK",
    "費城費城人": "PHI",
    "匹茲堡海盜": "PIT",
    "聖地牙哥教士": "SDP",
    "聖路易紅雀": "STL",
    "西雅圖水手": "SEA",
    "舊金山巨人": "SFG",
    "坦帕灣光芒": "TBR",
    "德州遊騎兵": "TEX",
    "多倫多藍鳥": "TOR",
    "華盛頓國民": "WSN",
}

INTERNATIONAL_TEAM_NAMES = frozenset(
    {
        "中華台北",
        "澳洲",
        "捷克",
        "韓國",
        "南韓",
        "日本",
        "古巴",
        "巴拿馬",
        "波多黎各",
        "哥倫比亞",
        "加拿大",
        "墨西哥",
        "英國",
        "美國",
        "巴西",
        "義大利",
        "荷蘭",
        "委內瑞拉",
        "尼加拉瓜",
        "多明尼加",
        "以色列",
    }
)


class TslAcquisitionSchemaError(ValueError):
    """A fail-closed source or row-shape rejection."""


@dataclass(frozen=True, slots=True)
class TslRejectedGame:
    """One source game that did not become a pregame observation."""

    source_row_index: int
    source_game_id: str
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class TslNormalizationResult:
    """Normalized observations and explicit invalid-row accounting."""

    observations: tuple[TslMoneylineObservation, ...]
    rejected_games: tuple[TslRejectedGame, ...]
    source_row_count: int

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(observation.row for observation in self.observations)


@dataclass(frozen=True, slots=True)
class TslBlob3rdRawCapture:
    """The bounded modern-source response set before P31A normalization."""

    sport_id: str
    games: tuple[dict[str, Any], ...]
    payloads: tuple[tuple[str, bytes], ...]

    @property
    def payload_sha256(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (url, sha256(raw).hexdigest()) for url, raw in self.payloads
        )

    @property
    def combined_payload_sha256(self) -> str:
        material = b"".join(
            url.encode("utf-8") + b"\0" + raw for url, raw in self.payloads
        )
        return sha256(material).hexdigest()


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TslAcquisitionSchemaError(
            f"{field_name} must be an ISO timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TslAcquisitionSchemaError(f"{field_name} must be timezone-aware")
    return parsed


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TslAcquisitionSchemaError(f"{field_name} must be a non-empty string")
    return value.strip()


def _decimal_odds(selection: Mapping[str, Any]) -> str:
    try:
        payout_unit = Decimal(str(selection.get("pu")))
        price_denom = Decimal(str(selection.get("pd")))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TslAcquisitionSchemaError("Moneyline pu/pd is not numeric") from exc
    if not payout_unit.is_finite() or not price_denom.is_finite() or price_denom <= 0:
        raise TslAcquisitionSchemaError("Moneyline pu/pd is not finite and positive")
    value = Decimal("1") + payout_unit / price_denom
    if not value.is_finite() or value <= Decimal("1"):
        raise TslAcquisitionSchemaError("Moneyline decimal odds must be greater than 1")
    four_places = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
    return format(
        four_places.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN),
        "f",
    )


def _rejection(
    *,
    source_row_index: int,
    source_game_id: object,
    reason: str,
    detail: str,
) -> TslRejectedGame:
    return TslRejectedGame(
        source_row_index=source_row_index,
        source_game_id=str(source_game_id or ""),
        reason=reason,
        detail=detail,
    )


def _detect_sport_league(home_team: str, away_team: str) -> str:
    if home_team in MLB_ZH_TO_TSL_CODE or away_team in MLB_ZH_TO_TSL_CODE:
        return "MLB"
    if home_team in INTERNATIONAL_TEAM_NAMES or away_team in INTERNATIONAL_TEAM_NAMES:
        return "WBC"
    if home_team or away_team:
        return "INTL"
    return "UNKNOWN"


def _normalize_game(
    game: Mapping[str, Any],
    *,
    source_row_index: int,
    fetched_at: str,
    target_date: date,
) -> TslMoneylineObservation:
    if str(game.get("san", "")).upper() != "BSB":
        raise TslAcquisitionSchemaError("source game is not baseball")
    source_identifier = _text(game.get("id"), field_name="id")
    away_team = _text(game.get("an"), field_name="an")
    home_team = _text(game.get("hn"), field_name="hn")
    if away_team == home_team:
        raise TslAcquisitionSchemaError("home and away teams must differ")
    game_time = _text(game.get("kt"), field_name="kt")
    game_dt = _parse_timestamp(game_time, field_name="game_time")
    if game_dt.astimezone(TSL_LOCAL_TIMEZONE).date() != target_date:
        raise TslAcquisitionSchemaError("game is outside the frozen target date")
    fetched_dt = _parse_timestamp(fetched_at, field_name="fetched_at")
    if fetched_dt >= game_dt:
        raise TslAcquisitionSchemaError("observation was acquired after game start")

    raw_markets = game.get("ms")
    if not isinstance(raw_markets, list):
        raise TslAcquisitionSchemaError("ms must be an array")
    # The legacy mapper calls several inning variants MNL.  P31A's existing
    # contract requires one full-game MNL market, so the exact full-game label
    # is selected and inning markets are excluded.
    moneyline_markets = [
        market
        for market in raw_markets
        if isinstance(market, Mapping)
        and str(market.get("name", "")).strip() == "不讓分"
    ]
    if len(moneyline_markets) != 1:
        raise TslAcquisitionSchemaError(
            "expected exactly one full-game 不讓分 market"
        )
    raw_selections = moneyline_markets[0].get("cs")
    if not isinstance(raw_selections, list) or len(raw_selections) != 2:
        raise TslAcquisitionSchemaError("full-game Moneyline must have two selections")
    if not all(isinstance(selection, Mapping) for selection in raw_selections):
        raise TslAcquisitionSchemaError("Moneyline selections must be objects")

    by_name = {
        _text(selection.get("name"), field_name="selection.name"): selection
        for selection in raw_selections
    }
    if set(by_name) != {away_team, home_team}:
        raise TslAcquisitionSchemaError(
            "Moneyline selections must explicitly contain away and home teams"
        )
    away_odds = _decimal_odds(by_name[away_team])
    home_odds = _decimal_odds(by_name[home_team])
    row: dict[str, Any] = {
        "source": TSL_BLOB3RD_SOURCE_LABEL,
        "fetched_at": fetched_at,
        "match_id": source_identifier,
        "game_time": game_time,
        "home_team_name": home_team,
        "away_team_name": away_team,
        "home_code": MLB_ZH_TO_TSL_CODE.get(home_team, ""),
        "away_code": MLB_ZH_TO_TSL_CODE.get(away_team, ""),
        "sport_league": _detect_sport_league(home_team, away_team),
        "is_pregame": True,
        "markets": [
            {
                "marketCode": "MNL",
                "outcomes": [
                    {
                        "outcomeName": away_team,
                        "odds": away_odds,
                        "specialBetValue": None,
                    },
                    {
                        "outcomeName": home_team,
                        "odds": home_odds,
                        "specialBetValue": None,
                    },
                ],
            }
        ],
    }
    return TslMoneylineObservation(
        source_row_index=source_row_index,
        source_row_fingerprint=sha256(canonical_json_bytes(row)).hexdigest(),
        source_identifier=source_identifier,
        home_team=home_team,
        away_team=away_team,
        game_time=game_time,
        fetched_at=fetched_at,
        provider_source=TSL_BLOB3RD_SOURCE_LABEL,
        market_code="MNL",
        home_decimal_odds=home_odds,
        away_decimal_odds=away_odds,
        row=row,
    )


def normalize_tsl_moneyline_games(
    games: Sequence[Mapping[str, Any]],
    *,
    fetched_at: str,
    target_date: str | date,
) -> TslNormalizationResult:
    """Normalize BSB games into P31A rows, retaining explicit rejections."""

    if isinstance(target_date, datetime):
        raise ValueError("target_date must be YYYY-MM-DD")
    try:
        target = (
            target_date
            if isinstance(target_date, date)
            else date.fromisoformat(target_date)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("target_date must be YYYY-MM-DD") from exc
    _parse_timestamp(fetched_at, field_name="fetched_at")

    observations: list[TslMoneylineObservation] = []
    rejected: list[TslRejectedGame] = []
    for source_row_index, game in enumerate(games, start=1):
        if not isinstance(game, Mapping):
            rejected.append(
                _rejection(
                    source_row_index=source_row_index,
                    source_game_id="",
                    reason="MALFORMED_ROW",
                    detail="source game must be an object",
                )
            )
            continue
        if str(game.get("san", "")).upper() != "BSB":
            continue
        try:
            observations.append(
                _normalize_game(
                    game,
                    source_row_index=source_row_index,
                    fetched_at=fetched_at,
                    target_date=target,
                )
            )
        except TslAcquisitionSchemaError as exc:
            detail = str(exc)
            reason = (
                "POST_START"
                if "after game start" in detail
                else "OUT_OF_SCOPE"
                if "outside the frozen target date" in detail
                else "MALFORMED_OR_INVALID_MONEYLINE"
            )
            rejected.append(
                _rejection(
                    source_row_index=source_row_index,
                    source_game_id=game.get("id", ""),
                    reason=reason,
                    detail=detail,
                )
            )
    return TslNormalizationResult(
        observations=tuple(observations),
        rejected_games=tuple(rejected),
        source_row_count=len(games),
    )


def _default_fetcher(url: str) -> bytes:
    if not url.startswith(f"{TSL_BLOB3RD_BASE_URL}/"):
        raise ValueError(f"{STOP_TSL_ACQUISITION_SOURCE}: URL is not allowlisted")
    request = Request(
        url,
        headers={
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "MatchAnalysis-P32A/1.0",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=20) as response:  # noqa: S310 - exact allowlisted TSL URL
            raw = response.read()
    except Exception as original_error:
        # The pinned legacy producer uses the same bounded curl fallback for
        # environments where urllib's TLS read stalls.  It remains a GET to
        # the exact allowlisted public endpoint and carries no credentials.
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "-L",
                    "-sS",
                    "--max-time",
                    "20",
                    "-X",
                    "GET",
                    url,
                    "-H",
                    "Accept: application/json, text/plain, */*",
                    "-H",
                    "User-Agent: MatchAnalysis-P32A/1.0",
                ],
                check=True,
                capture_output=True,
                timeout=25,
            )
            raw = completed.stdout
        except Exception as curl_error:
            raise RuntimeError(f"{STOP_TSL_ACQUISITION_SOURCE}: {url}") from curl_error
    if not raw:
        raise RuntimeError(f"{STOP_TSL_ACQUISITION_SOURCE}: empty response {url}")
    return raw


def _load_json_list(raw: bytes, *, url: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{STOP_TSL_ACQUISITION_SCHEMA}: {url}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError(f"{STOP_TSL_ACQUISITION_SCHEMA}: expected object list {url}")
    return value


class TslBlob3rdClient:
    """Bounded modern Blob3rd client with injectable read-only transport."""

    def __init__(self, fetcher: Callable[[str], bytes] = _default_fetcher) -> None:
        self._fetcher = fetcher

    def fetch_modern_capture(self) -> TslBlob3rdRawCapture:
        payloads: list[tuple[str, bytes]] = []

        live_raw = self._fetcher(TSL_BLOB3RD_LIVE_URL)
        payloads.append((TSL_BLOB3RD_LIVE_URL, live_raw))
        live_games = _load_json_list(live_raw, url=TSL_BLOB3RD_LIVE_URL)

        sports_raw = self._fetcher(TSL_BLOB3RD_SPORTS_URL)
        payloads.append((TSL_BLOB3RD_SPORTS_URL, sports_raw))
        sports = _load_json_list(sports_raw, url=TSL_BLOB3RD_SPORTS_URL)
        sport_ids = [
            str(sport.get("id", ""))
            for sport in sports
            if str(sport.get("abb", "")).upper() == "BSB"
            and str(sport.get("id", ""))
        ]
        if len(sport_ids) != 1:
            raise RuntimeError(
                f"{STOP_TSL_ACQUISITION_SOURCE}: BSB sport id is not unique"
            )
        sport_id = sport_ids[0]

        pre_games: list[dict[str, Any]] = []
        for language in ("zh", "en"):
            url = TSL_BLOB3RD_PRE_URL_TEMPLATE.format(
                sport_id=sport_id,
                language=language,
            )
            raw = self._fetcher(url)
            payloads.append((url, raw))
            pre_games = _load_json_list(raw, url=url)
            if pre_games:
                break

        deduped: dict[str, dict[str, Any]] = {}
        anonymous_index = 0
        for game in (*live_games, *pre_games):
            if str(game.get("san", "")).upper() != "BSB":
                continue
            game_id = str(game.get("id", ""))
            if not game_id:
                anonymous_index += 1
                game_id = f"__invalid_bsb_row_{anonymous_index}"
            deduped[game_id] = game
        return TslBlob3rdRawCapture(
            sport_id=sport_id,
            games=tuple(deduped.values()),
            payloads=tuple(payloads),
        )


def build_tsl_moneyline_history(
    capture: TslBlob3rdRawCapture,
    *,
    fetched_at: str,
    target_date: str | date,
) -> tuple[TslMoneylineHistory, TslNormalizationResult]:
    """Build the immutable native history object from one raw capture."""

    normalized = normalize_tsl_moneyline_games(
        capture.games,
        fetched_at=fetched_at,
        target_date=target_date,
    )
    rows = tuple(observation.row for observation in normalized.observations)
    selected_bytes = b"".join(canonical_json_bytes(row) for row in rows)
    target_text = (
        target_date.isoformat() if isinstance(target_date, date) else target_date
    )
    history = TslMoneylineHistory(
        rows=rows,
        observations=normalized.observations,
        raw_sha256=capture.combined_payload_sha256,
        selected_rows_sha256=sha256(selected_bytes).hexdigest(),
        source_row_count=normalized.source_row_count,
        qualified_row_count=len(normalized.observations),
        scope_start_date=target_text,
        scope_end_date=target_text,
    )
    return history, normalized


__all__ = (
    "MLB_ZH_TO_TSL_CODE",
    "INTERNATIONAL_TEAM_NAMES",
    "STOP_TSL_ACQUISITION_SCHEMA",
    "STOP_TSL_ACQUISITION_SOURCE",
    "TSL_ACQUISITION_SCHEMA_VERSION",
    "TSL_BLOB3RD_BASE_URL",
    "TSL_BLOB3RD_LIVE_URL",
    "TSL_BLOB3RD_PRE_URL_TEMPLATE",
    "TSL_BLOB3RD_SPORTS_URL",
    "TSL_BLOB3RD_SOURCE_LABEL",
    "TslAcquisitionSchemaError",
    "TslBlob3rdClient",
    "TslBlob3rdRawCapture",
    "TslMoneylineHistory",
    "TslNormalizationResult",
    "TslRejectedGame",
    "build_tsl_moneyline_history",
    "normalize_tsl_moneyline_games",
)
