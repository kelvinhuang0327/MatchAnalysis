"""P52A-backed MLB probable-pitcher schedule observation adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen

from ...application.use_cases.p52a_daily_prediction_freeze import (
    fetch_mlb_schedule_data,
    parse_schedule_rows,
)
from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc


MLB_PROVIDER_NAMESPACE = "mlb_official"
JsonOpener = Callable[[Request, int], bytes]
Clock = Callable[[], datetime]


def _clock_now() -> datetime:
    return datetime.now(UTC)


def _player_id(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive MLB player ID")
    text = str(value)
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(f"{field_name} must be a positive MLB player ID")
    return text


@dataclass(frozen=True, slots=True)
class StartingPitcherScheduleObservation:
    """One normalized schedule response row and its local receipt time."""

    provider_namespace: str
    provider_game_id: str
    official_date: str
    game_type: str
    game_number: int
    scheduled_start_utc: str
    home_team: Mapping[str, Any]
    away_team: Mapping[str, Any]
    home_probable_pitcher_player_id: str | None
    away_probable_pitcher_player_id: str | None
    signal_observed_at_utc: str

    def schedule_identity(self) -> dict[str, Any]:
        return {
            "provider_namespace": self.provider_namespace,
            "provider_game_id": self.provider_game_id,
            "official_date": self.official_date,
            "game_type": self.game_type,
            "game_number": self.game_number,
            "scheduled_start_utc": self.scheduled_start_utc,
            "home_team": dict(self.home_team),
            "away_team": dict(self.away_team),
        }

    def probable_pitcher_state(self) -> dict[str, str | None]:
        return {
            "HOME": self.home_probable_pitcher_player_id,
            "AWAY": self.away_probable_pitcher_player_id,
        }


def normalize_starting_pitcher_schedule(
    schedule_payload: Mapping[str, Any],
    *,
    provider_game_id: str,
    signal_observed_at_utc: str,
    game_number: int | None = None,
) -> StartingPitcherScheduleObservation:
    """Normalize through P52A and select one exact official provider game."""

    if not isinstance(provider_game_id, str) or not provider_game_id.strip():
        raise ValueError("provider_game_id is required")
    received = format_canonical_utc(parse_canonical_utc(signal_observed_at_utc))
    rows = parse_schedule_rows(schedule_payload)
    matches = [
        row
        for row in rows
        if row.get("provider_game_id") == provider_game_id
        and (game_number is None or row.get("game_number") == game_number)
    ]
    if len(matches) != 1:
        raise ValueError("MLB_SCHEDULE_GAME_NOT_UNIQUE: expected exactly one selected game")
    row = matches[0]
    raw_number = row.get("game_number")
    if isinstance(raw_number, bool) or not isinstance(raw_number, int) or raw_number <= 0:
        raise ValueError("MLB_SCHEDULE_GAME_NUMBER_INVALID")
    official_date = row.get("official_date")
    if not isinstance(official_date, str) or len(official_date) != 10:
        raise ValueError("MLB_SCHEDULE_OFFICIAL_DATE_INVALID")
    try:
        parsed_date = datetime.strptime(official_date, "%Y-%m-%d").date()
    except ValueError as error:
        raise ValueError("MLB_SCHEDULE_OFFICIAL_DATE_INVALID") from error
    if parsed_date.isoformat() != official_date:
        raise ValueError("MLB_SCHEDULE_OFFICIAL_DATE_INVALID")
    game_type = row.get("game_type")
    if not isinstance(game_type, str) or not game_type.strip():
        raise ValueError("MLB_SCHEDULE_GAME_TYPE_INVALID")
    scheduled = format_canonical_utc(parse_canonical_utc(str(row.get("scheduled_start_utc"))))
    if parse_canonical_utc(received) >= parse_canonical_utc(scheduled):
        raise ValueError("SIGNAL_OBSERVED_AFTER_START")
    teams: dict[str, dict[str, Any]] = {}
    for side in ("home", "away"):
        value = row.get(f"{side}_team")
        if not isinstance(value, Mapping):
            raise ValueError(f"MLB_SCHEDULE_{side.upper()}_TEAM_INVALID")
        if not isinstance(value.get("name"), str) or not value["name"].strip():
            raise ValueError(f"MLB_SCHEDULE_{side.upper()}_TEAM_NAME_REQUIRED")
        if not isinstance(value.get("abbreviation"), str) or not value["abbreviation"].strip():
            raise ValueError(f"MLB_SCHEDULE_{side.upper()}_TEAM_CODE_REQUIRED")
        teams[side] = {
            "id": value.get("id"),
            "name": value["name"],
            "abbreviation": value["abbreviation"],
        }
    home_starter = row.get("home_starter") or {}
    away_starter = row.get("away_starter") or {}
    return StartingPitcherScheduleObservation(
        provider_namespace=MLB_PROVIDER_NAMESPACE,
        provider_game_id=provider_game_id,
        official_date=official_date,
        game_type=game_type,
        game_number=raw_number,
        scheduled_start_utc=scheduled,
        home_team=teams["home"],
        away_team=teams["away"],
        home_probable_pitcher_player_id=_player_id(
            home_starter.get("player_id"), field_name="home_probable_pitcher_player_id"
        ),
        away_probable_pitcher_player_id=_player_id(
            away_starter.get("player_id"), field_name="away_probable_pitcher_player_id"
        ),
        signal_observed_at_utc=received,
    )


def fetch_starting_pitcher_schedule(
    *,
    target_date: str,
    provider_game_id: str,
    game_type: str = "R",
    game_number: int | None = None,
    opener: JsonOpener | None = None,
    clock: Clock = _clock_now,
) -> StartingPitcherScheduleObservation:
    """Fetch the official schedule and timestamp local receipt of response bytes."""

    receipt: list[str] = []

    def record_receipt(request: Request, timeout: int) -> bytes:
        if opener is not None:
            raw = opener(request, timeout)
        else:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - P52A fixes the endpoint
                raw = response.read()
        receipt.append(format_canonical_utc(clock()))
        return raw

    payload, _request_url, _sha256 = fetch_mlb_schedule_data(
        target_date,
        target_date,
        game_type=game_type,
        opener=record_receipt,
    )
    if len(receipt) != 1:
        raise RuntimeError("MLB_SCHEDULE_RECEIPT_TIME_UNAVAILABLE")
    return normalize_starting_pitcher_schedule(
        payload,
        provider_game_id=provider_game_id,
        signal_observed_at_utc=receipt[0],
        game_number=game_number,
    )


__all__ = (
    "MLB_PROVIDER_NAMESPACE",
    "StartingPitcherScheduleObservation",
    "fetch_starting_pitcher_schedule",
    "normalize_starting_pitcher_schedule",
)
