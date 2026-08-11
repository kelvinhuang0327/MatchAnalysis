"""Closed-schema loader for the exact TSL odds-history JSONL authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


STOP_SCHEMA_MISMATCH = "STOP_MATCHANALYSIS_P28AB_TSL_SCHEMA_MISMATCH"

_REQUIRED_FIELDS = frozenset(
    {
        "source",
        "fetched_at",
        "match_id",
        "game_time",
        "home_team_name",
        "away_team_name",
        "home_code",
        "away_code",
        "sport_league",
        "is_pregame",
        "markets",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {"force_closing_snapshot", "capture_reason", "dedup_bypassed"}
)


def _schema_error(detail: str) -> ValueError:
    return ValueError(f"{STOP_SCHEMA_MISMATCH}: {detail}")


def _reject_json_constant(value: str) -> None:
    raise _schema_error(f"non-standard JSON numeric constant: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _schema_error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_string(
    row: dict[str, Any],
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise _schema_error(f"{field} must be a string")
    if not allow_empty and not value:
        raise _schema_error(f"{field} must be non-empty")
    if value != value.strip():
        raise _schema_error(f"{field} must be trimmed")
    return value


def _require_timestamp(row: dict[str, Any], field: str) -> str:
    value = _require_string(row, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _schema_error(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _schema_error(f"{field} must be timezone-aware")
    return value


def _validate_markets(row: dict[str, Any]) -> None:
    markets = row.get("markets")
    if not isinstance(markets, list) or not markets:
        raise _schema_error("markets must be a non-empty array")
    for market_index, market in enumerate(markets):
        if not isinstance(market, dict):
            raise _schema_error(f"market {market_index} must be an object")
        market_code = market.get("marketCode")
        outcomes = market.get("outcomes")
        if not isinstance(market_code, str) or not market_code:
            raise _schema_error(f"market {market_index} has no marketCode")
        if not isinstance(outcomes, list):
            raise _schema_error(f"market {market_index} outcomes must be an array")
        for outcome_index, outcome in enumerate(outcomes):
            if not isinstance(outcome, dict):
                raise _schema_error(
                    f"market {market_index} outcome {outcome_index} must be an object"
                )
            _require_string(outcome, "outcomeName")
            _require_string(outcome, "odds")


@dataclass(frozen=True, slots=True)
class TslOddsHistorySnapshot:
    """Validated TSL rows plus the exact bytes fingerprint that supplied them."""

    rows: tuple[dict[str, Any], ...]
    raw_sha256: str


def load_tsl_odds_history(path: str | Path) -> TslOddsHistorySnapshot:
    """Load only one explicitly supplied TSL JSONL artifact."""

    source_path = Path(path)
    raw = source_path.read_bytes()
    if not raw:
        raise _schema_error("artifact is empty")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _schema_error("artifact is not UTF-8") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise _schema_error(f"blank row at line {line_number}")
        try:
            parsed = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except json.JSONDecodeError as exc:
            raise _schema_error(f"malformed JSON at line {line_number}") from exc
        if not isinstance(parsed, dict):
            raise _schema_error(f"line {line_number} must be an object")
        parsed_fields = frozenset(parsed)
        if not _REQUIRED_FIELDS <= parsed_fields or parsed_fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS:
            missing = sorted(_REQUIRED_FIELDS - parsed_fields)
            extra = sorted(parsed_fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
            raise _schema_error(
                f"closed schema mismatch at line {line_number}; "
                f"missing={missing}, extra={extra}"
            )
        _require_string(parsed, "source")
        _require_timestamp(parsed, "fetched_at")
        _require_string(parsed, "match_id")
        _require_timestamp(parsed, "game_time")
        _require_string(parsed, "home_team_name")
        _require_string(parsed, "away_team_name")
        if parsed["home_team_name"] == parsed["away_team_name"]:
            raise _schema_error(f"home and away teams match at line {line_number}")
        _require_string(parsed, "home_code", allow_empty=True)
        _require_string(parsed, "away_code", allow_empty=True)
        _require_string(parsed, "sport_league")
        if not isinstance(parsed["is_pregame"], bool):
            raise _schema_error(f"is_pregame must be boolean at line {line_number}")
        if "force_closing_snapshot" in parsed and not isinstance(
            parsed["force_closing_snapshot"], bool
        ):
            raise _schema_error(
                f"force_closing_snapshot must be boolean at line {line_number}"
            )
        if "dedup_bypassed" in parsed and not isinstance(parsed["dedup_bypassed"], bool):
            raise _schema_error(f"dedup_bypassed must be boolean at line {line_number}")
        if "capture_reason" in parsed:
            _require_string(parsed, "capture_reason")
        _validate_markets(parsed)
        rows.append(parsed)

    return TslOddsHistorySnapshot(
        rows=tuple(rows),
        raw_sha256=sha256(raw).hexdigest(),
    )


__all__ = (
    "STOP_SCHEMA_MISMATCH",
    "TslOddsHistorySnapshot",
    "load_tsl_odds_history",
)
