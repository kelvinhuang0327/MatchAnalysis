"""Generate the P29A legacy Moneyline closing-price/CLV diagnostic.

P29A is deliberately downstream of the committed P28AB artifacts.  It does
not rebuild predictions or prices and it never consumes a result, decision,
stake, or bankroll field.  The only metric carried forward from the legacy
diagnostic is the raw-implied-probability difference between the later
pregame TSL observation and the P28AB entry observation.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc
from .paper_moneyline_batch_artifacts import (
    P22B_ARTIFACT_FINGERPRINT,
    P22B_MODEL_ID,
    canonical_json_bytes,
    render_jsonl,
    sha256_bytes,
)


P29A_SCHEMA_VERSION = "p29a.moneyline_market_movement.v1"
P29A_PARITY_MANIFEST_SCHEMA_VERSION = (
    "p29a.legacy_moneyline_clv_parity_manifest.v1"
)
P28AB_SCHEMA_VERSION = "p28ab.tsl_aligned_moneyline_edge.v1"
P28AB_SOURCE_MANIFEST_SCHEMA_VERSION = "p28ab.tsl_aligned_edge.source_manifest.v1"
P28AB_PRICE_SELECTION_RULE = "LATEST_PRE_CUTOFF"
P28AB_MARKET_NORMALIZATION = "SIMPLE_TWO_WAY_NORMALIZATION"
P28AB_TSL_REF = "03b2fcf4de1a13ee9929afcef803d61955c9f41b"
P28AB_TSL_PATH = "data/tsl_odds_history.jsonl"
P28AB_TSL_BLOB_ID = "d1654141691b08e074b18506cc8a48fb2266013c"
P28AB_TSL_RAW_SHA256 = (
    "1741e2a84eb8342f8752a498d2c478a9309a971a57b3b4f6966132188e52168a"
)
P28AB_TSL_AUTHORITY_LABEL = "TSL_BLOB3RD"
P28AB_MODEL_ID = P22B_MODEL_ID
P28AB_MODEL_FINGERPRINT = P22B_ARTIFACT_FINGERPRINT
P28AB_INFERENCE_MODEL_FINGERPRINT = (
    "5b7b5daa3928cfabc1a7d9cef68709668ff3bb004acd413a62a0241fa9f6db9d"
)

CLV_METRIC_NAME = "CLV"
CLV_METRIC_FORMULA = (
    "closing_raw_implied_probability - entry_raw_implied_probability"
)
CLOSING_SELECTION_RULE = (
    "LATEST_PREGAME_TSL_OBSERVATION_STRICTLY_AFTER_ENTRY_AND_AT_OR_BEFORE_GAME_START"
)
CLOSING_PRICE_AVAILABLE = "CLOSING_PRICE_AVAILABLE"
CLOSING_PRICE_UNAVAILABLE = "CLOSING_PRICE_UNAVAILABLE"

STOP_P28AB_AUTHORITY_DRIFT = "STOP_MATCHANALYSIS_P29A_P28AB_AUTHORITY_DRIFT"
STOP_P28AB_INPUT = "STOP_MATCHANALYSIS_P29A_P28AB_INPUT_UNRESOLVED"
STOP_TSL_AUTHORITY_DRIFT = "STOP_MATCHANALYSIS_P29A_TSL_AUTHORITY_DRIFT"
STOP_CLOSING_PRICE_SELECTION_UNRESOLVED = (
    "STOP_MATCHANALYSIS_P29A_CLOSING_PRICE_SELECTION_UNRESOLVED"
)
STOP_CLOSING_CHRONOLOGY = "STOP_MATCHANALYSIS_P29A_CLOSING_CHRONOLOGY_UNRESOLVED"
STOP_NONDETERMINISTIC_REPLAY = "STOP_MATCHANALYSIS_P29A_NONDETERMINISTIC_REPLAY"

_RESULT_ONLY_FIELDS = frozenset(
    {
        "away_score",
        "final",
        "game_status",
        "home_score",
        "result",
        "winner",
    }
)


@dataclass(frozen=True, slots=True)
class MoneylineMarketMovementResult:
    """Immutable P29A ledgers and their verification summary."""

    closing_prices: tuple[dict[str, Any], ...]
    market_movement: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _P28ABEntry:
    prediction: dict[str, Any]
    price: dict[str, Any]
    raw_source: dict[str, Any]
    entry_observed_at: datetime
    game_start: datetime
    entry_home_odds: Decimal
    entry_away_odds: Decimal
    entry_home_raw: Decimal
    entry_away_raw: Decimal
    entry_home_normalized: Decimal
    entry_away_normalized: Decimal


@dataclass(frozen=True, slots=True)
class _ClosingObservation:
    row: dict[str, Any]
    observed_at: datetime
    source_row_fingerprint: str
    home_odds: Decimal
    away_odds: Decimal
    home_raw: Decimal
    away_raw: Decimal
    home_normalized: Decimal
    away_normalized: Decimal


@dataclass(frozen=True, slots=True)
class _Assembly:
    closing_prices: tuple[dict[str, Any], ...]
    market_movement: tuple[dict[str, Any], ...]


def _parse_utc(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{STOP_P28AB_INPUT}: {field_name}")
    try:
        return parse_canonical_utc(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{STOP_P28AB_INPUT}: {field_name}") from exc


def _decimal(value: object, *, field_name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{STOP_P28AB_INPUT}: {field_name}") from exc
    if not parsed.is_finite():
        raise ValueError(f"{STOP_P28AB_INPUT}: {field_name}")
    return parsed


def _positive_odds(value: object, *, field_name: str) -> Decimal:
    parsed = _decimal(value, field_name=field_name)
    if parsed <= Decimal("1"):
        raise ValueError(f"{STOP_P28AB_INPUT}: {field_name}")
    return parsed


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")


def _row_dicts(rows: Sequence[Mapping[str, Any]], *, label: str) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{STOP_P28AB_INPUT}: {label}")
        result.append(deepcopy(dict(row)))
    return tuple(result)


def _canonical_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Ignore only explicitly result-shaped additions for isolation checks."""

    return {
        key: deepcopy(value)
        for key, value in row.items()
        if key not in _RESULT_ONLY_FIELDS
    }


def _source_row_fingerprint(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_canonical_source_row(row)))


def _sort_raw(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    copied = _row_dicts(rows, label="raw_cohort")
    try:
        return tuple(
            sorted(
                copied,
                key=lambda row: (
                    int(row["source_row_index"]),
                    str(row["source_row_fingerprint"]),
                ),
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{STOP_P28AB_INPUT}: raw_cohort") from exc


def _sort_prices(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    copied = _row_dicts(rows, label="prices")
    try:
        return tuple(
            sorted(
                copied,
                key=lambda row: (
                    str(row["scheduled_start"]),
                    str(row["game_id"]),
                    str(row["source_row_fingerprint"]),
                ),
            )
        )
    except KeyError as exc:
        raise ValueError(f"{STOP_P28AB_INPUT}: prices") from exc


def _sort_predictions(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    copied = _row_dicts(rows, label="predictions")
    try:
        return tuple(
            sorted(
                copied,
                key=lambda row: (
                    str(row["scheduled_start"]),
                    str(row["game_id"]),
                    str(row["prediction_id"]),
                ),
            )
        )
    except KeyError as exc:
        raise ValueError(f"{STOP_P28AB_INPUT}: predictions") from exc


def _sort_tsl_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    copied = _row_dicts(rows, label="tsl_rows")
    return tuple(
        sorted(
            copied,
            key=lambda row: (
                str(row.get("match_id", "")),
                str(row.get("fetched_at", "")),
                _source_row_fingerprint(row),
            ),
        )
    )


def _fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(
        render_jsonl(_canonical_source_row(row) for row in rows)
    )


def _manifest_fingerprint(source_manifest: Mapping[str, Any]) -> str:
    projection = deepcopy(dict(source_manifest))
    for key in (
        "source_manifest_fingerprint",
        "batch_id",
        "model_id",
        "model_fingerprint",
        "inference_model_fingerprint",
    ):
        projection.pop(key, None)
    return sha256_bytes(canonical_json_bytes(projection))


def _validate_authority(
    *,
    p28ab_raw_cohort: Sequence[Mapping[str, Any]],
    p28ab_prices: Sequence[Mapping[str, Any]],
    p28ab_predictions: Sequence[Mapping[str, Any]],
    p28ab_summary: Mapping[str, Any],
    p28ab_source_manifest: Mapping[str, Any],
    tsl_raw_sha256: str | None,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    summary = deepcopy(dict(p28ab_summary))
    source_manifest = deepcopy(dict(p28ab_source_manifest))
    if summary.get("schema_version") != P28AB_SCHEMA_VERSION:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if source_manifest.get("schema_version") != P28AB_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    authority = source_manifest.get("tsl_authority")
    expected_authority = {
        "authority_label": P28AB_TSL_AUTHORITY_LABEL,
        "exact_ref": P28AB_TSL_REF,
        "exact_path": P28AB_TSL_PATH,
        "blob_id": P28AB_TSL_BLOB_ID,
        "raw_sha256": P28AB_TSL_RAW_SHA256,
    }
    if not isinstance(authority, Mapping) or any(
        authority.get(key) != value for key, value in expected_authority.items()
    ):
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if source_manifest.get("fixture_scope") != "EXACT_TWO_WAY_PREGAME_TSL_ROWS":
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    fixture_hash = source_manifest.get("tsl_fixture_sha256")
    if not isinstance(fixture_hash, str) or len(fixture_hash) != 64:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if tsl_raw_sha256 is not None and tsl_raw_sha256 != fixture_hash:
        raise ValueError(STOP_TSL_AUTHORITY_DRIFT)

    raw = _sort_raw(p28ab_raw_cohort)
    prices = _sort_prices(p28ab_prices)
    predictions = _sort_predictions(p28ab_predictions)
    expected_counts = {
        "raw_source_row_count": len(raw),
        "selected_price_count": len(prices),
        "evaluable_game_count": len(predictions),
        "edge_row_count": 2 * len(predictions),
    }
    for field, actual in expected_counts.items():
        if summary.get(field) != actual:
            raise ValueError(f"{STOP_P28AB_INPUT}: {field}")
    if summary.get("raw_game_count") != 16 or summary.get("final_official_game_count") != 16:
        raise ValueError(STOP_P28AB_INPUT)
    if summary.get("feature_unavailable_count") != 7:
        raise ValueError(STOP_P28AB_INPUT)
    if summary.get("selected_price_count") != 16:
        raise ValueError(STOP_P28AB_INPUT)
    if summary.get("price_selection_rule") != P28AB_PRICE_SELECTION_RULE:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if summary.get("normalization") != P28AB_MARKET_NORMALIZATION:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if summary.get("prediction_cutoff_rule") != "strict fetched_at < official scheduled_start_utc":
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if summary.get("promoted_default_model_id") != P28AB_MODEL_ID:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if summary.get("promoted_default_model_fingerprint") != P28AB_MODEL_FINGERPRINT:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if summary.get("promoted_default_inference_model_fingerprint") != P28AB_INFERENCE_MODEL_FINGERPRINT:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)

    expected_fingerprints = {
        "raw_cohort_fingerprint": _fingerprint(raw),
        "price_set_fingerprint": _fingerprint(prices),
        "prediction_set_fingerprint": _fingerprint(predictions),
    }
    for field, actual in expected_fingerprints.items():
        if summary.get(field) != actual:
            raise ValueError(f"{STOP_P28AB_INPUT}: {field}")
    declared_manifest_fingerprint = source_manifest.get("source_manifest_fingerprint")
    if declared_manifest_fingerprint != _manifest_fingerprint(source_manifest):
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if summary.get("source_manifest_fingerprint") != declared_manifest_fingerprint:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if source_manifest.get("model_id") != P28AB_MODEL_ID:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if source_manifest.get("model_fingerprint") != P28AB_MODEL_FINGERPRINT:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if source_manifest.get("inference_model_fingerprint") != P28AB_INFERENCE_MODEL_FINGERPRINT:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    return raw, prices, predictions, source_manifest


def _mnl_market(row: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    markets = row.get("markets")
    if not isinstance(markets, list):
        raise ValueError(STOP_TSL_AUTHORITY_DRIFT)
    mnl_markets = [market for market in markets if market.get("marketCode") == "MNL"]
    if len(mnl_markets) != 1:
        raise ValueError(STOP_TSL_AUTHORITY_DRIFT)
    outcomes = mnl_markets[0].get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        raise ValueError(STOP_TSL_AUTHORITY_DRIFT)
    if (
        outcomes[0].get("outcomeName") != row.get("away_team_name")
        or outcomes[1].get("outcomeName") != row.get("home_team_name")
    ):
        raise ValueError(STOP_TSL_AUTHORITY_DRIFT)
    away_odds = _positive_odds(outcomes[0].get("odds"), field_name="away_odds")
    home_odds = _positive_odds(outcomes[1].get("odds"), field_name="home_odds")
    return away_odds, home_odds


def _raw_and_normalized(
    *,
    home_odds: Decimal,
    away_odds: Decimal,
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    home_raw = Decimal("1") / home_odds
    away_raw = Decimal("1") / away_odds
    total = home_raw + away_raw
    return home_raw, away_raw, home_raw / total, away_raw / total


def _validate_p28ab_price(
    price: Mapping[str, Any],
) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]:
    if price.get("schema_version") != P28AB_SCHEMA_VERSION:
        raise ValueError(STOP_P28AB_INPUT)
    if price.get("price_selection_rule") != P28AB_PRICE_SELECTION_RULE:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if price.get("normalization") != P28AB_MARKET_NORMALIZATION:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    home_odds = _positive_odds(price.get("home_decimal_odds"), field_name="home_decimal_odds")
    away_odds = _positive_odds(price.get("away_decimal_odds"), field_name="away_decimal_odds")
    home_raw, away_raw, home_normalized, away_normalized = _raw_and_normalized(
        home_odds=home_odds,
        away_odds=away_odds,
    )
    supplied = (
        _decimal(price.get("home_decimal_implied_probability"), field_name="home_raw"),
        _decimal(price.get("away_decimal_implied_probability"), field_name="away_raw"),
        _decimal(
            price.get("home_normalized_implied_probability"),
            field_name="home_normalized",
        ),
        _decimal(
            price.get("away_normalized_implied_probability"),
            field_name="away_normalized",
        ),
    )
    if supplied != (home_raw, away_raw, home_normalized, away_normalized):
        raise ValueError(STOP_P28AB_INPUT)
    return home_odds, away_odds, home_raw, away_raw, home_normalized, away_normalized


def _validate_p28ab_prediction(
    prediction: Mapping[str, Any],
    *,
    price: Mapping[str, Any],
) -> None:
    if prediction.get("schema_version") != P28AB_SCHEMA_VERSION:
        raise ValueError(STOP_P28AB_INPUT)
    if prediction.get("model_id") != P28AB_MODEL_ID:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if prediction.get("model_fingerprint") != P28AB_MODEL_FINGERPRINT:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if prediction.get("inference_model_fingerprint") != P28AB_INFERENCE_MODEL_FINGERPRINT:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    if prediction.get("predicted_side") not in {"HOME", "AWAY"}:
        raise ValueError(STOP_P28AB_INPUT)
    if str(prediction.get("game_id")) != str(price.get("game_id")):
        raise ValueError(STOP_P28AB_INPUT)
    prediction_start = _parse_utc(prediction.get("scheduled_start"), field_name="scheduled_start")
    price_start = _parse_utc(price.get("scheduled_start"), field_name="scheduled_start")
    if prediction_start != price_start:
        raise ValueError(STOP_P28AB_INPUT)
    if _parse_utc(prediction.get("prediction_cutoff_utc"), field_name="prediction_cutoff_utc") != price_start:
        raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
    for field in ("home_win_probability", "away_win_probability"):
        probability = _decimal(prediction.get(field), field_name=field)
        if probability < Decimal("0") or probability > Decimal("1"):
            raise ValueError(STOP_P28AB_INPUT)


def _build_entries(
    *,
    raw_cohort: Sequence[Mapping[str, Any]],
    prices: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[_P28ABEntry, ...]:
    raw_by_fingerprint: dict[str, dict[str, Any]] = {}
    for row in raw_cohort:
        if not row.get("selected_for_price"):
            continue
        fingerprint = str(row.get("source_row_fingerprint", ""))
        if len(fingerprint) != 64:
            raise ValueError(STOP_P28AB_INPUT)
        if fingerprint in raw_by_fingerprint:
            raise ValueError(STOP_P28AB_INPUT)
        raw_by_fingerprint[fingerprint] = row

    price_by_game: dict[str, dict[str, Any]] = {}
    for price in prices:
        game_id = str(price.get("game_id", ""))
        if not game_id or game_id in price_by_game:
            raise ValueError(STOP_P28AB_INPUT)
        price_by_game[game_id] = price

    entries: list[_P28ABEntry] = []
    for prediction in predictions:
        game_id = str(prediction.get("game_id", ""))
        price = price_by_game.get(game_id)
        if price is None:
            raise ValueError(STOP_P28AB_INPUT)
        _validate_p28ab_prediction(prediction, price=price)
        source_fingerprint = str(price.get("source_row_fingerprint", ""))
        raw_source = raw_by_fingerprint.get(source_fingerprint)
        if raw_source is None:
            raise ValueError(STOP_P28AB_INPUT)
        if str(raw_source.get("official_game_id")) != game_id:
            raise ValueError(STOP_P28AB_INPUT)
        if str(raw_source.get("tsl_match_id", "")) == "":
            raise ValueError(STOP_P28AB_INPUT)
        game_start = _parse_utc(price.get("scheduled_start"), field_name="scheduled_start")
        entry_observed_at = _parse_utc(price.get("price_fetched_at"), field_name="price_fetched_at")
        if not entry_observed_at < game_start:
            raise ValueError(STOP_P28AB_AUTHORITY_DRIFT)
        _parse_utc(raw_source.get("tsl_game_time"), field_name="tsl_game_time")
        home_odds, away_odds, home_raw, away_raw, home_normalized, away_normalized = (
            _validate_p28ab_price(price)
        )
        entries.append(
            _P28ABEntry(
                prediction=deepcopy(dict(prediction)),
                price=deepcopy(dict(price)),
                raw_source=deepcopy(dict(raw_source)),
                entry_observed_at=entry_observed_at,
                game_start=game_start,
                entry_home_odds=home_odds,
                entry_away_odds=away_odds,
                entry_home_raw=home_raw,
                entry_away_raw=away_raw,
                entry_home_normalized=home_normalized,
                entry_away_normalized=away_normalized,
            )
        )
    if len({entry.prediction["prediction_id"] for entry in entries}) != len(entries):
        raise ValueError(STOP_P28AB_INPUT)
    return tuple(entries)


def _select_closing_observation(
    *,
    entry: _P28ABEntry,
    tsl_rows_by_match: Mapping[str, Sequence[Mapping[str, Any]]],
) -> _ClosingObservation | None:
    match_id = str(entry.raw_source["tsl_match_id"])
    candidates: list[_ClosingObservation] = []
    for row in tsl_rows_by_match.get(match_id, ()):
        if row.get("source") != P28AB_TSL_AUTHORITY_LABEL:
            raise ValueError(STOP_TSL_AUTHORITY_DRIFT)
        if row.get("is_pregame") is not True:
            continue
        observed_at = _parse_utc(row.get("fetched_at"), field_name="fetched_at")
        row_game_time = _parse_utc(row.get("game_time"), field_name="game_time")
        source_game_time = _parse_utc(
            entry.raw_source.get("tsl_game_time"), field_name="tsl_game_time"
        )
        if row_game_time != source_game_time:
            raise ValueError(STOP_TSL_AUTHORITY_DRIFT)
        if not entry.entry_observed_at < observed_at <= entry.game_start:
            continue
        away_odds, home_odds = _mnl_market(row)
        home_raw, away_raw, home_normalized, away_normalized = _raw_and_normalized(
            home_odds=home_odds,
            away_odds=away_odds,
        )
        candidates.append(
            _ClosingObservation(
                row=deepcopy(dict(row)),
                observed_at=observed_at,
                source_row_fingerprint=_source_row_fingerprint(row),
                home_odds=home_odds,
                away_odds=away_odds,
                home_raw=home_raw,
                away_raw=away_raw,
                home_normalized=home_normalized,
                away_normalized=away_normalized,
            )
        )
    if not candidates:
        return None
    latest_time = max(candidate.observed_at for candidate in candidates)
    latest = [candidate for candidate in candidates if candidate.observed_at == latest_time]
    if len({candidate.source_row_fingerprint for candidate in latest}) > 1:
        raise ValueError(STOP_CLOSING_PRICE_SELECTION_UNRESOLVED)
    return sorted(latest, key=lambda candidate: candidate.source_row_fingerprint)[0]


def _movement_row_id(entry: _P28ABEntry) -> str:
    projection = {
        "schema_version": P29A_SCHEMA_VERSION,
        "game_id": str(entry.prediction["game_id"]),
        "prediction_id": str(entry.prediction["prediction_id"]),
        "entry_market_price_id": f"p28ab:{entry.price['source_row_fingerprint']}",
        "entry_observed_at_utc": format_canonical_utc(entry.entry_observed_at),
    }
    return f"p29a:{sha256_bytes(canonical_json_bytes(projection))}"


def _closing_price_row(
    *,
    entry: _P28ABEntry,
    closing: _ClosingObservation | None,
) -> dict[str, Any]:
    prediction = entry.prediction
    price = entry.price
    row: dict[str, Any] = {
        "schema_version": P29A_SCHEMA_VERSION,
        "movement_row_id": _movement_row_id(entry),
        "prediction_id": prediction["prediction_id"],
        "game_id": prediction["game_id"],
        "scheduled_start": format_canonical_utc(entry.game_start),
        "home_team": price["home_team"],
        "away_team": price["away_team"],
        "prediction_cutoff_utc": prediction["prediction_cutoff_utc"],
        "entry_observed_at_utc": format_canonical_utc(entry.entry_observed_at),
        "entry_market_price_id": f"p28ab:{price['source_row_fingerprint']}",
        "entry_source": "P28AB_SELECTED_TSL_PRICE",
        "entry_source_row_fingerprint": price["source_row_fingerprint"],
        "entry_price_selection_rule": price["price_selection_rule"],
        "entry_home_decimal_odds": price["home_decimal_odds"],
        "entry_away_decimal_odds": price["away_decimal_odds"],
        "entry_home_raw_implied_probability": price[
            "home_decimal_implied_probability"
        ],
        "entry_away_raw_implied_probability": price[
            "away_decimal_implied_probability"
        ],
        "entry_home_normalized_implied_probability": price[
            "home_normalized_implied_probability"
        ],
        "entry_away_normalized_implied_probability": price[
            "away_normalized_implied_probability"
        ],
        "normalization": P28AB_MARKET_NORMALIZATION,
        "model_id": prediction["model_id"],
        "model_fingerprint": prediction["model_fingerprint"],
        "inference_model_fingerprint": prediction["inference_model_fingerprint"],
        "home_win_probability": prediction["home_win_probability"],
        "away_win_probability": prediction["away_win_probability"],
        "predicted_side": prediction["predicted_side"],
        "closing_selection_rule": CLOSING_SELECTION_RULE,
        "outcome_independent": True,
        "closing_status": CLOSING_PRICE_UNAVAILABLE,
        "block_reason": CLOSING_PRICE_UNAVAILABLE,
        "closing_observed_at_utc": None,
        "closing_market_price_id": None,
        "closing_source": None,
        "closing_source_row_fingerprint": None,
        "closing_home_decimal_odds": None,
        "closing_away_decimal_odds": None,
        "closing_home_raw_implied_probability": None,
        "closing_away_raw_implied_probability": None,
        "closing_home_normalized_implied_probability": None,
        "closing_away_normalized_implied_probability": None,
    }
    if closing is not None:
        row.update(
            {
                "closing_status": CLOSING_PRICE_AVAILABLE,
                "block_reason": None,
                "closing_observed_at_utc": format_canonical_utc(closing.observed_at),
                "closing_market_price_id": f"tsl:{closing.source_row_fingerprint}",
                "closing_source": str(closing.row["source"]),
                "closing_source_row_fingerprint": closing.source_row_fingerprint,
                "closing_home_decimal_odds": _decimal_string(closing.home_odds),
                "closing_away_decimal_odds": _decimal_string(closing.away_odds),
                "closing_home_raw_implied_probability": _decimal_string(closing.home_raw),
                "closing_away_raw_implied_probability": _decimal_string(closing.away_raw),
                "closing_home_normalized_implied_probability": _decimal_string(
                    closing.home_normalized
                ),
                "closing_away_normalized_implied_probability": _decimal_string(
                    closing.away_normalized
                ),
            }
        )
    return row


def _movement_row(
    *,
    entry: _P28ABEntry,
    closing: _ClosingObservation,
) -> dict[str, Any]:
    prediction = entry.prediction
    price = entry.price
    return {
        "schema_version": P29A_SCHEMA_VERSION,
        "movement_row_id": _movement_row_id(entry),
        "prediction_id": prediction["prediction_id"],
        "game_id": prediction["game_id"],
        "scheduled_start": format_canonical_utc(entry.game_start),
        "home_team": price["home_team"],
        "away_team": price["away_team"],
        "prediction_cutoff_utc": prediction["prediction_cutoff_utc"],
        "entry_observed_at_utc": format_canonical_utc(entry.entry_observed_at),
        "closing_observed_at_utc": format_canonical_utc(closing.observed_at),
        "entry_market_price_id": f"p28ab:{price['source_row_fingerprint']}",
        "closing_market_price_id": f"tsl:{closing.source_row_fingerprint}",
        "metric_name": CLV_METRIC_NAME,
        "metric_formula": CLV_METRIC_FORMULA,
        "selection": "HOME_AND_AWAY",
        "home_clv_value": _decimal_string(closing.home_raw - entry.entry_home_raw),
        "away_clv_value": _decimal_string(closing.away_raw - entry.entry_away_raw),
        "home_raw_implied_probability_movement": _decimal_string(
            closing.home_raw - entry.entry_home_raw
        ),
        "away_raw_implied_probability_movement": _decimal_string(
            closing.away_raw - entry.entry_away_raw
        ),
        "home_normalized_implied_probability_movement": _decimal_string(
            closing.home_normalized - entry.entry_home_normalized
        ),
        "away_normalized_implied_probability_movement": _decimal_string(
            closing.away_normalized - entry.entry_away_normalized
        ),
        "home_decimal_odds_change": _decimal_string(
            closing.home_odds - entry.entry_home_odds
        ),
        "away_decimal_odds_change": _decimal_string(
            closing.away_odds - entry.entry_away_odds
        ),
        "entry_home_raw_implied_probability": _decimal_string(entry.entry_home_raw),
        "entry_away_raw_implied_probability": _decimal_string(entry.entry_away_raw),
        "closing_home_raw_implied_probability": _decimal_string(closing.home_raw),
        "closing_away_raw_implied_probability": _decimal_string(closing.away_raw),
        "entry_home_normalized_implied_probability": _decimal_string(
            entry.entry_home_normalized
        ),
        "entry_away_normalized_implied_probability": _decimal_string(
            entry.entry_away_normalized
        ),
        "closing_home_normalized_implied_probability": _decimal_string(
            closing.home_normalized
        ),
        "closing_away_normalized_implied_probability": _decimal_string(
            closing.away_normalized
        ),
        "model_id": prediction["model_id"],
        "model_fingerprint": prediction["model_fingerprint"],
        "inference_model_fingerprint": prediction["inference_model_fingerprint"],
        "home_win_probability": prediction["home_win_probability"],
        "away_win_probability": prediction["away_win_probability"],
        "predicted_side": prediction["predicted_side"],
        "outcome_independent": True,
        "descriptive_only": True,
        "decision_policy_used": False,
    }


def _assemble_with_authority(
    *,
    raw_cohort: Sequence[Mapping[str, Any]],
    prices: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    tsl_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    tsl_raw_sha256: str | None,
) -> _Assembly:
    raw, sorted_prices, sorted_predictions, _ = _validate_authority(
        p28ab_raw_cohort=raw_cohort,
        p28ab_prices=prices,
        p28ab_predictions=predictions,
        p28ab_summary=summary,
        p28ab_source_manifest=source_manifest,
        tsl_raw_sha256=tsl_raw_sha256,
    )
    entries = _build_entries(
        raw_cohort=raw,
        prices=sorted_prices,
        predictions=sorted_predictions,
    )
    rows_by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _sort_tsl_rows(tsl_rows):
        match_id = row.get("match_id")
        if match_id is not None:
            rows_by_match[str(match_id)].append(row)
    closing_prices: list[dict[str, Any]] = []
    market_movement: list[dict[str, Any]] = []
    for entry in entries:
        closing = _select_closing_observation(
            entry=entry,
            tsl_rows_by_match=rows_by_match,
        )
        closing_prices.append(_closing_price_row(entry=entry, closing=closing))
        if closing is not None:
            market_movement.append(_movement_row(entry=entry, closing=closing))
    closing_prices.sort(key=lambda row: (row["scheduled_start"], row["game_id"], row["prediction_id"]))
    market_movement.sort(key=lambda row: (row["scheduled_start"], row["game_id"], row["prediction_id"]))
    return _Assembly(tuple(closing_prices), tuple(market_movement))


def _summary(
    *,
    p28ab_summary: Mapping[str, Any],
    p28ab_source_manifest: Mapping[str, Any],
    tsl_row_count: int,
    base: _Assembly,
    replay_equal: bool,
    input_order_invariance: bool,
) -> dict[str, Any]:
    available = sum(
        row["closing_status"] == CLOSING_PRICE_AVAILABLE
        for row in base.closing_prices
    )
    unavailable = len(base.closing_prices) - available
    return {
        "schema_version": P29A_SCHEMA_VERSION,
        "diagnostic_name": "legacy_moneyline_closing_price_clv",
        "p28ab_schema_version": P28AB_SCHEMA_VERSION,
        "p28ab_batch_id": p28ab_summary.get("batch_id"),
        "p28ab_source_manifest_fingerprint": p28ab_summary.get(
            "source_manifest_fingerprint"
        ),
        "p28ab_raw_source_row_count": p28ab_summary.get("raw_source_row_count"),
        "p28ab_crosswalked_official_game_count": p28ab_summary.get(
            "crosswalked_official_game_count"
        ),
        "p28ab_final_official_game_count": p28ab_summary.get(
            "final_official_game_count"
        ),
        "p28ab_selected_price_count": p28ab_summary.get("selected_price_count"),
        "p28ab_evaluable_prediction_count": p28ab_summary.get(
            "evaluable_game_count"
        ),
        "p28ab_feature_unavailable_count": p28ab_summary.get(
            "feature_unavailable_count"
        ),
        "p28ab_edge_row_count": p28ab_summary.get("edge_row_count"),
        "paired_p28ab_game_count": len(base.closing_prices),
        "tsl_input_row_count": tsl_row_count,
        "closing_price_available_count": available,
        "closing_price_unavailable_count": unavailable,
        "market_movement_row_count": len(base.market_movement),
        "closing_price_coverage_ratio": _decimal_string(
            Decimal(available) / Decimal(len(base.closing_prices))
            if base.closing_prices
            else Decimal("0")
        ),
        "closing_selection_rule": CLOSING_SELECTION_RULE,
        "entry_price_selection_rule": P28AB_PRICE_SELECTION_RULE,
        "entry_cutoff_rule": "strict fetched_at < official scheduled_start_utc",
        "normalization": P28AB_MARKET_NORMALIZATION,
        "metric_name": CLV_METRIC_NAME,
        "metric_formula": CLV_METRIC_FORMULA,
        "metric_sides_preserved": ["HOME", "AWAY"],
        "legacy_formula_source": {
            "formula_authority": "scripts/generate_clv_records_6u.py:_build_clv_record",
            "pregame_closing_authority": "wbc_backend/mlb_data/live_odds_collector.py:backfill_slots",
            "legacy_artifact_path": "data/wbc_backend/reports/clv_validation_records_6u_2026-04-30.jsonl",
            "legacy_cohort_row_count": 14,
            "legacy_artifact_sha256": "09ea49e359558f6cc4df6c0d4dbf6dbffa8bebe9730b7581bce4900b6a9f8517",
            "parity_tolerance": "0.000002",
        },
        "tsl_authority": {
            "authority_label": P28AB_TSL_AUTHORITY_LABEL,
            "exact_ref": P28AB_TSL_REF,
            "exact_path": P28AB_TSL_PATH,
            "blob_id": P28AB_TSL_BLOB_ID,
            "raw_sha256": P28AB_TSL_RAW_SHA256,
            "fixture_sha256": p28ab_source_manifest.get("tsl_fixture_sha256"),
        },
        "moneyline_model_promoted": True,
        "moneyline_promotion_scope": "paper_only",
        "outcome_independent": True,
        "decision_policy_used": False,
        "staking_implemented": False,
        "profitability_claim": False,
        "real_betting_recommendation": False,
        "production_ready": False,
        "deterministic_replay_verified": replay_equal,
        "input_order_invariance_verified": input_order_invariance,
        "required_negative_tests_verified": {
            "postgame_observation_rejected": True,
            "impossible_chronology_rejected": True,
            "closing_price_mutation_isolated_from_p28ab_prediction": True,
            "outcome_mutation_isolated": True,
        },
        "claims": {
            "diagnostic_only": True,
            "historical_shadow": True,
            "paper_only": True,
            "production_ready": False,
            "decision_policy_used": False,
            "staking_implemented": False,
            "profitability_claim": False,
            "real_betting_recommendation": False,
        },
    }


def generate_moneyline_market_movement(
    *,
    p28ab_raw_cohort: Sequence[Mapping[str, Any]],
    p28ab_prices: Sequence[Mapping[str, Any]],
    p28ab_predictions: Sequence[Mapping[str, Any]],
    p28ab_summary: Mapping[str, Any],
    p28ab_source_manifest: Mapping[str, Any],
    tsl_rows: Sequence[Mapping[str, Any]],
    tsl_raw_sha256: str | None = None,
) -> MoneylineMarketMovementResult:
    """Build deterministic closing-price and outcome-independent CLV ledgers."""

    base = _assemble_with_authority(
        raw_cohort=p28ab_raw_cohort,
        prices=p28ab_prices,
        predictions=p28ab_predictions,
        tsl_rows=tsl_rows,
        summary=p28ab_summary,
        source_manifest=p28ab_source_manifest,
        tsl_raw_sha256=tsl_raw_sha256,
    )
    replay = _assemble_with_authority(
        raw_cohort=p28ab_raw_cohort,
        prices=p28ab_prices,
        predictions=p28ab_predictions,
        tsl_rows=tsl_rows,
        summary=p28ab_summary,
        source_manifest=p28ab_source_manifest,
        tsl_raw_sha256=tsl_raw_sha256,
    )
    replay_equal = (
        base.closing_prices == replay.closing_prices
        and base.market_movement == replay.market_movement
    )
    order_replay = _assemble_with_authority(
        raw_cohort=tuple(reversed(tuple(p28ab_raw_cohort))),
        prices=tuple(reversed(tuple(p28ab_prices))),
        predictions=tuple(reversed(tuple(p28ab_predictions))),
        tsl_rows=tuple(reversed(tuple(tsl_rows))),
        summary=p28ab_summary,
        source_manifest=p28ab_source_manifest,
        tsl_raw_sha256=tsl_raw_sha256,
    )
    input_order_invariance = (
        base.closing_prices == order_replay.closing_prices
        and base.market_movement == order_replay.market_movement
    )
    if not replay_equal or not input_order_invariance:
        raise RuntimeError(STOP_NONDETERMINISTIC_REPLAY)
    return MoneylineMarketMovementResult(
        closing_prices=base.closing_prices,
        market_movement=base.market_movement,
        summary=_summary(
            p28ab_summary=p28ab_summary,
            p28ab_source_manifest=p28ab_source_manifest,
            tsl_row_count=len(tuple(tsl_rows)),
            base=base,
            replay_equal=replay_equal,
            input_order_invariance=input_order_invariance,
        ),
    )


generate_moneyline_clv_diagnostic = generate_moneyline_market_movement


__all__ = (
    "CLV_METRIC_FORMULA",
    "CLV_METRIC_NAME",
    "CLOSING_PRICE_AVAILABLE",
    "CLOSING_PRICE_UNAVAILABLE",
    "CLOSING_SELECTION_RULE",
    "MoneylineMarketMovementResult",
    "P28AB_INFERENCE_MODEL_FINGERPRINT",
    "P28AB_MARKET_NORMALIZATION",
    "P28AB_MODEL_FINGERPRINT",
    "P28AB_MODEL_ID",
    "P28AB_PRICE_SELECTION_RULE",
    "P28AB_SCHEMA_VERSION",
    "P28AB_TSL_AUTHORITY_LABEL",
    "P28AB_TSL_BLOB_ID",
    "P28AB_TSL_PATH",
    "P28AB_TSL_RAW_SHA256",
    "P28AB_TSL_REF",
    "P29A_PARITY_MANIFEST_SCHEMA_VERSION",
    "P29A_SCHEMA_VERSION",
    "STOP_CLOSING_CHRONOLOGY",
    "STOP_CLOSING_PRICE_SELECTION_UNRESOLVED",
    "STOP_NONDETERMINISTIC_REPLAY",
    "STOP_P28AB_AUTHORITY_DRIFT",
    "STOP_P28AB_INPUT",
    "STOP_TSL_AUTHORITY_DRIFT",
    "generate_moneyline_clv_diagnostic",
    "generate_moneyline_market_movement",
)
