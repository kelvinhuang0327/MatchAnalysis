"""Native, date-scoped loader for the migrated TSL Moneyline authority.

The raw artifact is an immutable copy of the pinned Betting-pool Git blob.  A
date-scoped load applies the already-qualified P28AB selection rule while
retaining the exact source row dictionaries for downstream P28AB processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


TSL_SOURCE_LABEL = "TSL_BLOB3RD"
TSL_LOCAL_TIMEZONE = timezone(timedelta(hours=8))
TSL_AUTHORITY_REF = "03b2fcf4de1a13ee9929afcef803d61955c9f41b"
TSL_AUTHORITY_TREE = "56a849bc68234db63da7a38f1643fa664217c5d0"
TSL_AUTHORITY_PATH = "data/tsl_odds_history.jsonl"
TSL_AUTHORITY_BLOB = "d1654141691b08e074b18506cc8a48fb2266013c"
TSL_AUTHORITY_RAW_SHA256 = (
    "1741e2a84eb8342f8752a498d2c478a9309a971a57b3b4f6966132188e52168a"
)
TSL_MIGRATED_PATH = "data/authority/tsl/tsl_odds_history.jsonl"

STOP_TSL_SCHEMA_MISMATCH = "STOP_MATCHANALYSIS_P31A_TSL_SCHEMA_MISMATCH"
STOP_TSL_LOADER_PARITY_FAILED = "STOP_MATCHANALYSIS_P31A_TSL_LOADER_PARITY_FAILED"

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


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _schema_error(detail: str) -> ValueError:
    return ValueError(f"{STOP_TSL_SCHEMA_MISMATCH}: {detail}")


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
    row: Mapping[str, Any],
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


def _require_timestamp(row: Mapping[str, Any], field: str) -> str:
    value = _require_string(row, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _schema_error(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _schema_error(f"{field} must be timezone-aware")
    return value


def _validate_markets(row: Mapping[str, Any]) -> None:
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


def _parse_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _schema_error(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _schema_error(f"{field} must be timezone-aware")
    return parsed


def _parse_date(value: str | date, *, field: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD") from exc


def _decimal(value: object, *, field: str, line_number: int) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise _schema_error(f"{field} has invalid decimal odds at line {line_number}") from exc
    if not parsed.is_finite() or parsed <= Decimal("1"):
        raise _schema_error(f"{field} must be finite and greater than 1 at line {line_number}")
    return parsed


@dataclass(frozen=True, slots=True)
class TslMoneylineObservation:
    """One qualified source row with its original identity and MNL projection."""

    source_row_index: int
    source_row_fingerprint: str
    source_identifier: str
    home_team: str
    away_team: str
    game_time: str
    fetched_at: str
    provider_source: str
    market_code: str
    home_decimal_odds: str
    away_decimal_odds: str
    row: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TslMoneylineHistory:
    """Date-scoped qualified rows plus raw and selected-byte fingerprints."""

    rows: tuple[dict[str, Any], ...]
    observations: tuple[TslMoneylineObservation, ...]
    raw_sha256: str
    selected_rows_sha256: str
    source_row_count: int
    qualified_row_count: int
    scope_start_date: str
    scope_end_date: str


def _qualified_observation(
    row: dict[str, Any],
    *,
    line_number: int,
    start_date: date,
    end_date: date,
) -> TslMoneylineObservation | None:
    if row.get("source") != TSL_SOURCE_LABEL:
        return None

    game_time_value = row.get("game_time")
    if not isinstance(game_time_value, str):
        raise _schema_error(f"game_time is missing at line {line_number}")
    game_time = _parse_timestamp(game_time_value, field="game_time")
    local_game_date = game_time.astimezone(TSL_LOCAL_TIMEZONE).date()
    if not start_date <= local_game_date <= end_date:
        return None
    if row.get("is_pregame") is not True:
        return None

    parsed_fields = frozenset(row)
    if not _REQUIRED_FIELDS <= parsed_fields or parsed_fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS:
        missing = sorted(_REQUIRED_FIELDS - parsed_fields)
        extra = sorted(parsed_fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
        raise _schema_error(
            f"closed schema mismatch at line {line_number}; missing={missing}, extra={extra}"
        )
    _require_string(row, "source")
    fetched_at_value = _require_timestamp(row, "fetched_at")
    fetched_at = _parse_timestamp(fetched_at_value, field="fetched_at")
    _require_string(row, "match_id")
    _require_timestamp(row, "game_time")
    home_team = _require_string(row, "home_team_name")
    away_team = _require_string(row, "away_team_name")
    if home_team == away_team:
        raise _schema_error(f"home and away teams match at line {line_number}")
    _require_string(row, "home_code", allow_empty=True)
    _require_string(row, "away_code", allow_empty=True)
    _require_string(row, "sport_league")
    _validate_markets(row)
    if fetched_at >= game_time:
        return None

    markets = [
        market for market in row["markets"] if market.get("marketCode") == "MNL"
    ]
    if len(markets) != 1:
        return None
    outcomes = markets[0].get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return None
    if (
        outcomes[0].get("outcomeName") != away_team
        or outcomes[1].get("outcomeName") != home_team
    ):
        return None
    away_decimal_odds = _decimal(
        outcomes[0].get("odds"), field="away_odds", line_number=line_number
    )
    home_decimal_odds = _decimal(
        outcomes[1].get("odds"), field="home_odds", line_number=line_number
    )
    fingerprint = sha256(canonical_json_bytes(row)).hexdigest()
    return TslMoneylineObservation(
        source_row_index=line_number,
        source_row_fingerprint=fingerprint,
        source_identifier=str(row["match_id"]),
        home_team=home_team,
        away_team=away_team,
        game_time=game_time_value,
        fetched_at=fetched_at_value,
        provider_source=str(row["source"]),
        market_code="MNL",
        home_decimal_odds=format(home_decimal_odds, "f"),
        away_decimal_odds=format(away_decimal_odds, "f"),
        row=row,
    )


def load_tsl_moneyline_history(
    path: str | Path,
    *,
    start_date: str | date,
    end_date: str | date,
) -> TslMoneylineHistory:
    """Load exact bytes and return only the already-qualified scoped MNL rows."""

    start = _parse_date(start_date, field="start_date")
    end = _parse_date(end_date, field="end_date")
    if end < start:
        raise ValueError("end_date must not precede start_date")

    source_path = Path(path)
    raw = source_path.read_bytes()
    if not raw:
        raise _schema_error("artifact is empty")
    raw_sha256 = sha256(raw).hexdigest()
    if raw_sha256 != TSL_AUTHORITY_RAW_SHA256:
        raise ValueError(
            f"{STOP_TSL_LOADER_PARITY_FAILED}: raw authority hash mismatch"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _schema_error("artifact is not UTF-8") from exc

    observations: list[TslMoneylineObservation] = []
    selected_raw_lines: list[bytes] = []
    for line_number, raw_line in enumerate(raw.splitlines(keepends=True), start=1):
        line = raw_line.decode("utf-8")
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
        observation = _qualified_observation(
            parsed,
            line_number=line_number,
            start_date=start,
            end_date=end,
        )
        if observation is None:
            continue
        observations.append(observation)
        selected_raw_lines.append(raw_line)

    observations.sort(
        key=lambda item: (
            item.game_time,
            item.fetched_at,
            item.source_identifier,
            item.row.get("home_code", ""),
            item.row.get("away_code", ""),
            canonical_json_bytes(item.row),
        )
    )
    # P28AB's frozen fixture is the original line-order projection of this
    # same qualification rule.  Keep the selected-byte fingerprint tied to
    # source order, while row consumers receive deterministic sorted records.
    selected_bytes = b"".join(selected_raw_lines)
    return TslMoneylineHistory(
        rows=tuple(item.row for item in observations),
        observations=tuple(observations),
        raw_sha256=raw_sha256,
        selected_rows_sha256=sha256(selected_bytes).hexdigest(),
        source_row_count=len(raw.splitlines()),
        qualified_row_count=len(observations),
        scope_start_date=start.isoformat(),
        scope_end_date=end.isoformat(),
    )


__all__ = (
    "STOP_TSL_LOADER_PARITY_FAILED",
    "STOP_TSL_SCHEMA_MISMATCH",
    "TSL_AUTHORITY_BLOB",
    "TSL_AUTHORITY_PATH",
    "TSL_AUTHORITY_RAW_SHA256",
    "TSL_AUTHORITY_REF",
    "TSL_AUTHORITY_TREE",
    "TSL_LOCAL_TIMEZONE",
    "TSL_MIGRATED_PATH",
    "TSL_SOURCE_LABEL",
    "TslMoneylineHistory",
    "TslMoneylineObservation",
    "canonical_json_bytes",
    "load_tsl_moneyline_history",
)
