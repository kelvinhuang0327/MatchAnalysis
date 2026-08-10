"""Read-only MLB-owned source adapter for bounded historical acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
MLB_STATS_API_GAME_BASE = "https://statsapi.mlb.com/api/v1.1/game"
MLB_SOURCE_DOMAIN = "mlb.com"
PROVIDER_NAMESPACE = "MLB_STATS_API"
RAW_SCHEMA_VERSION = "p23f2.mlb_official_raw.v1"
NORMALIZED_SCHEMA_VERSION = "p23f2.mlb_official_normalized.v1"

JsonOpener = Callable[[Request, int], bytes]


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def format_utc(value: datetime) -> str:
    normalized = value.astimezone(UTC)
    if normalized.microsecond:
        return normalized.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_opener(request: Request, timeout: int) -> bytes:
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - allowlisted MLB URL is constructed below
        return response.read()


def fetch_json_bytes(
    url: str,
    *,
    query: Mapping[str, str] | None = None,
    opener: JsonOpener = _default_opener,
    timeout: int = 60,
) -> tuple[bytes, str]:
    if not url.startswith((MLB_STATS_API_BASE, MLB_STATS_API_GAME_BASE)):
        raise ValueError("only allowlisted MLB Stats API URLs may be fetched")
    query_string = urlencode(query or {})
    request_url = f"{url}?{query_string}" if query_string else url
    request = Request(
        request_url,
        headers={"Accept": "application/json", "User-Agent": "MatchAnalysis/1.0"},
    )
    raw = opener(request, timeout)
    if not raw:
        raise ValueError(f"empty MLB response: {request_url}")
    json.loads(raw.decode("utf-8"))
    return raw, request_url


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _team_projection(wrapper: Mapping[str, Any], path: str) -> dict[str, Any]:
    team = wrapper.get("team")
    if not isinstance(team, Mapping):
        raise ValueError(f"missing team object: {path}")
    team_id = team.get("id")
    abbreviation = team.get("abbreviation")
    name = team.get("name")
    if not isinstance(team_id, int) or not isinstance(abbreviation, str) or not isinstance(name, str):
        raise ValueError(f"invalid team identity: {path}")
    return {"id": team_id, "abbreviation": abbreviation, "name": name}


def normalize_schedule_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for schedule_date in payload.get("dates", []):
        if not isinstance(schedule_date, Mapping):
            raise ValueError("schedule date must be an object")
        for game in schedule_date.get("games", []):
            if not isinstance(game, Mapping):
                raise ValueError("schedule game must be an object")
            teams = game.get("teams")
            status = game.get("status")
            if not isinstance(teams, Mapping) or not isinstance(status, Mapping):
                raise ValueError("schedule game is missing teams or status")
            home = teams.get("home")
            away = teams.get("away")
            if not isinstance(home, Mapping) or not isinstance(away, Mapping):
                raise ValueError("schedule game is missing home or away")
            game_pk = game.get("gamePk")
            game_number = game.get("gameNumber")
            official_date = game.get("officialDate")
            scheduled = game.get("gameDate")
            if (
                not isinstance(game_pk, int)
                or not isinstance(game_number, int)
                or not isinstance(official_date, str)
                or not isinstance(scheduled, str)
            ):
                raise ValueError("schedule game has invalid identity or timing")
            start = parse_utc(scheduled)
            home_team = _team_projection(home, "home")
            away_team = _team_projection(away, "away")
            if home_team["id"] == away_team["id"]:
                raise ValueError("home and away teams must differ")
            status_name = status.get("detailedState")
            if not isinstance(status_name, str):
                raise ValueError("schedule status is missing detailedState")
            score_fields: dict[str, int | None] = {}
            for side, wrapper in (("home", home), ("away", away)):
                score = wrapper.get("score")
                score_fields[f"{side}_score"] = score if isinstance(score, int) else None
            rows.append(
                {
                    "schema_version": NORMALIZED_SCHEMA_VERSION,
                    "provider_game_id": str(game_pk),
                    "game_pk": game_pk,
                    "game_number": game_number,
                    "official_date": official_date,
                    "scheduled_start_utc": format_utc(start),
                    "status": status_name,
                    "final": status.get("abstractGameState") == "Final",
                    "home_team": home_team,
                    "away_team": away_team,
                    **score_fields,
                }
            )
    rows.sort(key=lambda row: (row["scheduled_start_utc"], row["game_number"], row["game_pk"]))
    if not rows:
        raise ValueError("schedule contained no games")
    return tuple(rows)


def _boxscore_teams(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload.get("teams"), Mapping):
        return payload["teams"]
    live_data = payload.get("liveData")
    if isinstance(live_data, Mapping):
        boxscore = live_data.get("boxscore")
        if isinstance(boxscore, Mapping) and isinstance(boxscore.get("teams"), Mapping):
            return boxscore["teams"]
    raise ValueError("boxscore payload has no teams object")


def _first_starter(team: Mapping[str, Any], side: str) -> dict[str, Any]:
    pitchers = team.get("pitchers")
    players = team.get("players")
    if not isinstance(pitchers, list) or not pitchers or not isinstance(players, Mapping):
        raise ValueError(f"{side} boxscore has no pitching players")
    player_id = pitchers[0]
    player = players.get(f"ID{player_id}")
    if not isinstance(player, Mapping):
        raise ValueError(f"{side} starter player is missing")
    person = player.get("person")
    stats = player.get("stats")
    pitching = stats.get("pitching") if isinstance(stats, Mapping) else None
    if not isinstance(person, Mapping) or not isinstance(pitching, Mapping):
        raise ValueError(f"{side} starter has no pitching stats")
    if pitching.get("gamesStarted") != 1:
        raise ValueError(f"{side} first pitcher is not the official starter")
    name = person.get("fullName")
    parent_team_id = player.get("parentTeamId")
    team_projection = team.get("team")
    if not isinstance(parent_team_id, int) and isinstance(team_projection, Mapping):
        parent_team_id = team_projection.get("id")
    if not isinstance(name, str) or not isinstance(parent_team_id, int):
        raise ValueError(f"{side} starter identity is incomplete")
    return {
        "player_id": player_id,
        "full_name": name,
        "team_id": parent_team_id,
        "games_started": 1,
    }


def normalize_boxscore_payload(
    payload: Mapping[str, Any],
    *,
    game: Mapping[str, Any],
) -> dict[str, Any]:
    teams = _boxscore_teams(payload)
    home = teams.get("home")
    away = teams.get("away")
    if not isinstance(home, Mapping) or not isinstance(away, Mapping):
        raise ValueError("boxscore is missing home or away team")
    home_starter = _first_starter(home, "home")
    away_starter = _first_starter(away, "away")
    return {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "provider_game_id": game["provider_game_id"],
        "game_pk": game["game_pk"],
        "game_number": game["game_number"],
        "official_date": game["official_date"],
        "scheduled_start_utc": game["scheduled_start_utc"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "home_starter": home_starter,
        "away_starter": away_starter,
    }


def normalize_pitcher_game_log_payload(
    payload: Mapping[str, Any],
    *,
    player_id: int,
) -> tuple[dict[str, Any], ...]:
    stats = payload.get("stats")
    if not isinstance(stats, list) or not stats or not isinstance(stats[0], Mapping):
        raise ValueError("pitcher game log has no stats")
    splits = stats[0].get("splits")
    if not isinstance(splits, list):
        raise ValueError("pitcher game log has no splits")
    rows: list[dict[str, Any]] = []
    for split in splits:
        if not isinstance(split, Mapping):
            raise ValueError("pitcher split must be an object")
        stat = split.get("stat")
        game = split.get("game")
        split_date = split.get("date")
        if not isinstance(stat, Mapping) or not isinstance(game, Mapping) or not isinstance(split_date, str):
            raise ValueError("pitcher split is incomplete")
        game_pk = game.get("gamePk")
        if not isinstance(game_pk, int):
            raise ValueError("pitcher split has invalid game identity")
        rows.append(
            {
                "schema_version": NORMALIZED_SCHEMA_VERSION,
                "player_id": player_id,
                "date": split_date,
                "game_pk": game_pk,
                "game_number": game.get("gameNumber", 1),
                "game_type": split.get("gameType", "R"),
                "games_started": stat.get("gamesStarted", 0),
                "earned_runs": stat.get("earnedRuns", 0),
                "outs": stat.get("outs", 0),
            }
        )
    rows.sort(key=lambda row: (row["date"], row["game_pk"], row["game_number"]))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class RawSourceRecord:
    path: str
    url: str
    scope: str
    acquired_at_utc: str
    sha256: str


def write_raw_response(
    *,
    raw_root: Path,
    relative_path: str,
    raw: bytes,
) -> str:
    path = raw_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"immutable raw source changed: {path}")
    if not path.exists():
        path.write_bytes(raw)
    return sha256_bytes(raw)


def write_normalized_json(path: Path, value: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> str:
    if isinstance(value, Mapping):
        raw = canonical_json_bytes(value)
    else:
        raw = b"".join(canonical_json_bytes(row) for row in value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha256_bytes(raw)


__all__ = (
    "MLB_SOURCE_DOMAIN",
    "MLB_STATS_API_BASE",
    "MLB_STATS_API_GAME_BASE",
    "NORMALIZED_SCHEMA_VERSION",
    "PROVIDER_NAMESPACE",
    "RAW_SCHEMA_VERSION",
    "RawSourceRecord",
    "canonical_json_bytes",
    "fetch_json_bytes",
    "format_utc",
    "load_json",
    "normalize_boxscore_payload",
    "normalize_pitcher_game_log_payload",
    "normalize_schedule_payload",
    "parse_utc",
    "sha256_bytes",
    "write_normalized_json",
    "write_raw_response",
)
