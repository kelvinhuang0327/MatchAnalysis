"""Read-only P39A adapter for the frozen legacy TSL odds JSONL source."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import format_canonical_utc
from ...baseball.domain.moneyline_market_snapshot import (
    MoneylineMarketObservationCandidate,
)


P39A_TSL_SOURCE_LABEL = "TSL_BLOB3RD"
P39A_TSL_SOURCE_RELATIVE_PATH = "data/tsl_odds_history.jsonl"
P39A_STOP_SOURCE_UNSTABLE = "P39A_LEGACY_MARKET_SOURCE_UNSTABLE_STOP"
P39A_STOP_SOURCE_HASH_MISMATCH = "P39A_LEGACY_MARKET_SOURCE_HASH_MISMATCH"
P39A_STOP_SOURCE_SCHEMA = "P39A_LEGACY_MARKET_SOURCE_SCHEMA_ERROR"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"{P39A_STOP_SOURCE_SCHEMA}: non-standard JSON value {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{P39A_STOP_SOURCE_SCHEMA}: duplicate JSON key {key}")
        result[key] = value
    return result


def _parse_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a trimmed timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _text(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{field_name} must be a trimmed string")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _price(value: object) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not parsed.is_finite() or parsed <= Decimal("1"):
        return None
    return parsed


def _moneyline_prices(
    row: Mapping[str, Any],
) -> tuple[str, Decimal | None, Decimal | None, str | None]:
    markets = row.get("markets")
    if not isinstance(markets, list):
        return "MISSING_MONEYLINE", None, None, "MISSING_OR_INCOMPLETE_PRICE"
    moneyline_markets = [
        market
        for market in markets
        if isinstance(market, Mapping) and market.get("marketCode") == "MNL"
    ]
    if not moneyline_markets:
        return "MISSING_MONEYLINE", None, None, "MISSING_OR_INCOMPLETE_PRICE"

    exact: list[tuple[Decimal, Decimal]] = []
    for market in moneyline_markets:
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != 2:
            continue
        by_name = {
            outcome.get("outcomeName"): outcome
            for outcome in outcomes
            if isinstance(outcome, Mapping)
        }
        if set(by_name) != {
            row.get("home_team_name"),
            row.get("away_team_name"),
        } or len(by_name) != 2:
            continue
        home_price = _price(by_name[row["home_team_name"]].get("odds"))
        away_price = _price(by_name[row["away_team_name"]].get("odds"))
        if home_price is None or away_price is None:
            continue
        exact.append((home_price, away_price))

    if len(exact) > 1:
        return "AMBIGUOUS_MONEYLINE_MARKET", None, None, "AMBIGUOUS_MONEYLINE_MARKET"
    if len(exact) == 1:
        return "VALID_MONEYLINE", exact[0][0], exact[0][1], None
    return "MALFORMED_MONEYLINE", None, None, "MALFORMED_OR_INCOMPLETE_PRICE"


@dataclass(frozen=True, slots=True)
class TslMarketSourceRead:
    """Frozen source bytes and only the observations in the requested scope."""

    source_path: str
    raw_sha256: str
    source_row_count: int
    scoped_source_row_count: int
    candidates: tuple[MoneylineMarketObservationCandidate, ...]
    scoped_status_counts: tuple[tuple[str, int], ...]


def _candidate_from_row(
    row: Mapping[str, Any],
    *,
    source_row_index: int,
    source_row_fingerprint: str,
    scheduled_start_utc: str,
) -> MoneylineMarketObservationCandidate:
    source_match_id = _text(row.get("match_id"), field_name="match_id")
    home_name = _text(row.get("home_team_name"), field_name="home_team_name")
    away_name = _text(row.get("away_team_name"), field_name="away_team_name")
    home_code = _text(row.get("home_code"), field_name="home_code")
    away_code = _text(row.get("away_code"), field_name="away_code")
    if home_name == away_name or home_code == away_code:
        raise ValueError(f"{P39A_STOP_SOURCE_SCHEMA}: source teams must differ")

    fetched_value = row.get("fetched_at")
    fetched_at_utc: str | None = None
    fetched_at: datetime | None = None
    if isinstance(fetched_value, str):
        try:
            fetched_at = _parse_timestamp(fetched_value, field_name="fetched_at")
            fetched_at_utc = format_canonical_utc(fetched_at)
        except ValueError:
            fetched_at = None

    market_code, home_price, away_price, market_reason = _moneyline_prices(row)
    is_pregame = row.get("is_pregame")
    if not isinstance(is_pregame, bool):
        is_pregame = None

    scheduled = _parse_timestamp(scheduled_start_utc, field_name="scheduled_start_utc")
    if fetched_at is None:
        status = "MISSING_OR_UNTRUSTED_TIMESTAMP"
        rejection_reason = "MISSING_OR_UNTRUSTED_TIMESTAMP"
    elif fetched_at >= scheduled:
        status = "POST_START"
        rejection_reason = "POST_START"
    elif is_pregame is not True:
        status = "NOT_PREGAME"
        rejection_reason = "NOT_PREGAME"
    elif market_reason is not None:
        status = market_reason
        rejection_reason = market_reason
    else:
        status = "VALID_PREGAME"
        rejection_reason = None

    return MoneylineMarketObservationCandidate(
        source_row_index=source_row_index,
        source_row_fingerprint=source_row_fingerprint,
        source_match_id=source_match_id,
        source_home_team_name=home_name,
        source_away_team_name=away_name,
        source_home_code=home_code,
        source_away_code=away_code,
        scheduled_start_utc=format_canonical_utc(scheduled),
        market_observed_at_utc=fetched_at_utc,
        local_fetched_at_utc=fetched_at_utc,
        provider_observed_at_utc=None,
        is_pregame=is_pregame,
        market_code="MNL" if market_code != "MISSING_MONEYLINE" else "MNL",
        market_status=status,
        rejection_reason=rejection_reason,
        home_decimal_price=home_price,
        away_decimal_price=away_price,
    )


def load_tsl_market_source(
    path: str | Path,
    *,
    expected_sha256: str,
    target_source_keys: set[tuple[str, str, str]],
) -> TslMarketSourceRead:
    """Read two identical byte snapshots, then parse only exact target keys."""

    source_path = Path(path).resolve()
    first_raw = source_path.read_bytes()
    first_hash = sha256(first_raw).hexdigest()
    if first_hash != expected_sha256:
        raise ValueError(
            f"{P39A_STOP_SOURCE_HASH_MISMATCH}: expected {expected_sha256}, got {first_hash}"
        )
    second_raw = source_path.read_bytes()
    second_hash = sha256(second_raw).hexdigest()
    if first_raw != second_raw or first_hash != second_hash:
        raise RuntimeError(P39A_STOP_SOURCE_UNSTABLE)

    candidates: list[MoneylineMarketObservationCandidate] = []
    status_counts: Counter[str] = Counter()
    source_row_count = 0
    for source_row_index, raw_line in enumerate(first_raw.splitlines(keepends=True), start=1):
        source_row_count += 1
        if not raw_line.strip():
            raise ValueError(f"{P39A_STOP_SOURCE_SCHEMA}: blank row {source_row_index}")
        try:
            row = json.loads(
                raw_line.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"{P39A_STOP_SOURCE_SCHEMA}: malformed row {source_row_index}"
            ) from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"{P39A_STOP_SOURCE_SCHEMA}: row {source_row_index} is not an object")
        if row.get("source") != P39A_TSL_SOURCE_LABEL:
            continue
        try:
            home_code = _text(row.get("home_code"), field_name="home_code")
            away_code = _text(row.get("away_code"), field_name="away_code")
            game_time = format_canonical_utc(
                _parse_timestamp(row.get("game_time"), field_name="game_time")
            )
        except ValueError:
            continue
        if (home_code, away_code, game_time) not in target_source_keys:
            continue
        candidate = _candidate_from_row(
            row,
            source_row_index=source_row_index,
            source_row_fingerprint=sha256(raw_line).hexdigest(),
            scheduled_start_utc=game_time,
        )
        candidates.append(candidate)
        status_counts[candidate.market_status] += 1

    candidates.sort(
        key=lambda candidate: (
            candidate.scheduled_start_utc,
            candidate.source_home_code,
            candidate.source_away_code,
            candidate.source_match_id,
            candidate.market_observed_at_utc or "",
            candidate.source_row_fingerprint,
        )
    )
    return TslMarketSourceRead(
        source_path=str(source_path),
        raw_sha256=first_hash,
        source_row_count=source_row_count,
        scoped_source_row_count=len(candidates),
        candidates=tuple(candidates),
        scoped_status_counts=tuple(sorted(status_counts.items())),
    )


__all__ = (
    "P39A_STOP_SOURCE_HASH_MISMATCH",
    "P39A_STOP_SOURCE_SCHEMA",
    "P39A_STOP_SOURCE_UNSTABLE",
    "P39A_TSL_SOURCE_LABEL",
    "P39A_TSL_SOURCE_RELATIVE_PATH",
    "TslMarketSourceRead",
    "load_tsl_market_source",
)
