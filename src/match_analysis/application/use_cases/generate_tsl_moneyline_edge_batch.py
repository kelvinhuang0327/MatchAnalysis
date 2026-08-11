"""Generate the P28AB TSL-aligned historical-shadow Moneyline edge batch.

This use case is deliberately offline and diagnostic-only.  It consumes a
validated derived slice of the exact TSL authority, maps it to official MLB
games, freezes P22B/P24C-compatible features before outcomes are joined, and
selects the latest pre-cutoff two-way Moneyline price for descriptive edge
reporting.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import format_canonical_utc
from ...baseball.domain.future_evaluation_fold import FutureFeatureRow
from ...baseball.domain.moneyline_feature_snapshot import (
    MoneylineFeatureProvenance,
    MoneylineFeatureSnapshot,
)
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
from ...core.identity import MatchIdentity
from .generate_moneyline_predictions import generate_moneyline_predictions
from .materialize_future_moneyline_fold import (
    FutureFeatureEligibility,
    classify_future_feature_eligibility,
    materialize_future_moneyline_fold,
)
from .paper_moneyline_batch_artifacts import (
    P22B_ARTIFACT_FINGERPRINT,
    P22B_ARTIFACT_RELATIVE_PATH,
    P22B_MODEL_ID,
    canonical_json_bytes,
    load_model_artifact_with_fingerprint,
    render_jsonl,
    sha256_bytes,
)


P28AB_SCHEMA_VERSION = "p28ab.tsl_aligned_moneyline_edge.v1"
P28AB_SOURCE_MANIFEST_SCHEMA_VERSION = "p28ab.tsl_aligned_edge.source_manifest.v1"
P28AB_FOLD_ID = "wf_008"
P28AB_COHORT_START_DATE = "2026-05-18"
P28AB_COHORT_END_DATE = "2026-05-24"
P28AB_UTC_OFFSET = timezone(timedelta(hours=8))
P28AB_FEATURE_SOURCE_KIND = "MLB_STATS_API_P28AB_PIT_FEATURE"
P28AB_PRICE_SELECTION_RULE = "LATEST_PRE_CUTOFF"
P28AB_MARKET_NORMALIZATION = "SIMPLE_TWO_WAY_NORMALIZATION"
P28AB_EDGE_SEMANTICS = (
    "DESCRIPTIVE_MODEL_MINUS_SIMPLE_TWO_WAY_NORMALIZED_IMPLIED_PROBABILITY"
)
P28AB_TSL_REF = "03b2fcf4de1a13ee9929afcef803d61955c9f41b"
P28AB_TSL_PATH = "data/tsl_odds_history.jsonl"
P28AB_TSL_BLOB_ID = "d1654141691b08e074b18506cc8a48fb2266013c"
P28AB_TSL_RAW_SHA256 = "1741e2a84eb8342f8752a498d2c478a9309a971a57b3b4f6966132188e52168a"
P28AB_TSL_AUTHORITY_LABEL = "TSL_BLOB3RD"

STOP_TSL_AUTHORITY_DRIFT = "STOP_MATCHANALYSIS_P28AB_TSL_AUTHORITY_DRIFT"
STOP_CROSSWALK_UNRESOLVED = "STOP_MATCHANALYSIS_P28AB_CROSSWALK_UNRESOLVED"
STOP_FEATURE_INPUT = "STOP_MATCHANALYSIS_P28AB_FEATURE_INPUT_UNRESOLVED"
STOP_DEFAULT_MODEL_DRIFT = "STOP_MATCHANALYSIS_P28AB_DEFAULT_MODEL_DRIFT"
STOP_OFFLINE_REPLAY = "STOP_MATCHANALYSIS_P28AB_OFFLINE_REPLAY_MISMATCH"

TSL_TO_MLB_TEAM_ABBREVIATION = {
    "ARI": "AZ",
    "ATH": "ATH",
    "CHW": "CWS",
    "CWS": "CWS",
    "KCR": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "OAK": "ATH",
    "PHI": "PHI",
    "PIT": "PIT",
    "SDP": "SD",
    "SEA": "SEA",
    "SFG": "SF",
    "SF": "SF",
    "STL": "STL",
    "TBR": "TB",
    "TB": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSN": "WSH",
    "WSH": "WSH",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KC",
    "LAA": "LAA",
}


@dataclass(frozen=True, slots=True)
class TslMoneylineEdgeBatchResult:
    """Immutable P28AB ledgers and deterministic verification summary."""

    raw_cohort: tuple[dict[str, Any], ...]
    prices: tuple[dict[str, Any], ...]
    predictions: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    feature_unavailable: tuple[dict[str, Any], ...]
    source_manifest: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CrosswalkObservation:
    source_row_index: int
    source_row_fingerprint: str
    row: dict[str, Any]
    away_decimal_odds: Decimal
    home_decimal_odds: Decimal
    canonical_home_code: str | None
    canonical_away_code: str | None
    official: Mapping[str, Any] | None
    status: str
    crosswalk_time_delta_seconds: int | None


@dataclass(frozen=True, slots=True)
class _Assembly:
    raw_cohort: tuple[dict[str, Any], ...]
    prices: tuple[dict[str, Any], ...]
    predictions: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    feature_unavailable: tuple[dict[str, Any], ...]
    snapshots: tuple[MoneylineFeatureSnapshot, ...]
    crosswalk: tuple[_CrosswalkObservation, ...]


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    return parsed.astimezone(UTC)


def _decimal(value: object, *, field_name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT) from exc
    if not parsed.is_finite() or parsed <= Decimal("1"):
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    return parsed


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")


def _format_utc(value: datetime) -> str:
    return format_canonical_utc(value.astimezone(UTC))


def _validate_source_manifest(
    source_manifest: Mapping[str, Any],
    *,
    tsl_raw_sha256: str | None,
    cohort_start_date: str = P28AB_COHORT_START_DATE,
    cohort_end_date: str = P28AB_COHORT_END_DATE,
) -> dict[str, Any]:
    manifest = deepcopy(dict(source_manifest))
    if manifest.get("schema_version") != P28AB_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    if manifest.get("source_authority") != "MLB_STATS_API":
        raise RuntimeError(STOP_FEATURE_INPUT)
    if manifest.get("source_domains") != ["mlb.com"]:
        raise RuntimeError(STOP_FEATURE_INPUT)
    authority = manifest.get("tsl_authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    expected = {
        "authority_label": P28AB_TSL_AUTHORITY_LABEL,
        "exact_ref": P28AB_TSL_REF,
        "exact_path": P28AB_TSL_PATH,
        "blob_id": P28AB_TSL_BLOB_ID,
        "raw_sha256": P28AB_TSL_RAW_SHA256,
    }
    if any(authority.get(key) != value for key, value in expected.items()):
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    declared_hash = manifest.get("tsl_fixture_sha256")
    if not isinstance(declared_hash, str) or len(declared_hash) != 64:
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    if tsl_raw_sha256 is not None and tsl_raw_sha256 != declared_hash:
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    scope = manifest.get("cohort_scope")
    if not isinstance(scope, Mapping) or (
        scope.get("game_time_start_date"),
        scope.get("game_time_end_date"),
    ) != (cohort_start_date, cohort_end_date):
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    if manifest.get("fixture_scope") != "EXACT_TWO_WAY_PREGAME_TSL_ROWS":
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    return manifest


def _mnl_market(row: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    markets = row.get("markets")
    if not isinstance(markets, list):
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    mnl_markets = [market for market in markets if market.get("marketCode") == "MNL"]
    if len(mnl_markets) != 1:
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    outcomes = mnl_markets[0].get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    if (
        outcomes[0].get("outcomeName") != row.get("away_team_name")
        or outcomes[1].get("outcomeName") != row.get("home_team_name")
    ):
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
    away_odds = _decimal(outcomes[0].get("odds"), field_name="away_odds")
    home_odds = _decimal(outcomes[1].get("odds"), field_name="home_odds")
    return away_odds, home_odds


def _team_code(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return TSL_TO_MLB_TEAM_ABBREVIATION.get(value)


def _sorted_source_rows(
    tsl_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows = tuple(deepcopy(dict(row)) for row in tsl_rows)
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                str(row.get("game_time", "")),
                str(row.get("fetched_at", "")),
                str(row.get("match_id", "")),
                str(row.get("home_code", "")),
                str(row.get("away_code", "")),
                canonical_json_bytes(row),
            ),
        )
    )


def _crosswalk(
    *,
    tsl_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
    cohort_start_date: str = P28AB_COHORT_START_DATE,
    cohort_end_date: str = P28AB_COHORT_END_DATE,
) -> tuple[_CrosswalkObservation, ...]:
    official_rows = tuple(schedule_rows)
    observations: list[_CrosswalkObservation] = []
    for source_row_index, row in enumerate(_sorted_source_rows(tsl_rows), start=1):
        if row.get("source") != P28AB_TSL_AUTHORITY_LABEL:
            raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
        if row.get("is_pregame") is not True:
            raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
        game_time = _parse_utc(str(row["game_time"]))
        local_game_date = game_time.astimezone(P28AB_UTC_OFFSET).date()
        if not (
            date.fromisoformat(cohort_start_date)
            <= local_game_date
            <= date.fromisoformat(cohort_end_date)
        ):
            raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
        fetched_at = _parse_utc(str(row["fetched_at"]))
        if fetched_at >= game_time:
            raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)
        away_odds, home_odds = _mnl_market(row)
        home_code = _team_code(row.get("home_code"))
        away_code = _team_code(row.get("away_code"))
        row_fingerprint = sha256_bytes(canonical_json_bytes(row))
        if home_code is None or away_code is None:
            observations.append(
                _CrosswalkObservation(
                    source_row_index=source_row_index,
                    source_row_fingerprint=row_fingerprint,
                    row=row,
                    away_decimal_odds=away_odds,
                    home_decimal_odds=home_odds,
                    canonical_home_code=home_code,
                    canonical_away_code=away_code,
                    official=None,
                    status="NO_CANONICAL_TEAM_CODE",
                    crosswalk_time_delta_seconds=None,
                )
            )
            continue
        candidates = []
        for official in official_rows:
            if (
                str(official["home_team"]["abbreviation"]) != home_code
                or str(official["away_team"]["abbreviation"]) != away_code
            ):
                continue
            delta = abs(
                int(
                    (
                        _parse_utc(str(official["scheduled_start_utc"])) - game_time
                    ).total_seconds()
                )
            )
            if delta <= 3600:
                candidates.append((delta, str(official["provider_game_id"]), official))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if not candidates or (
            len(candidates) > 1 and candidates[0][0] == candidates[1][0]
        ):
            observations.append(
                _CrosswalkObservation(
                    source_row_index=source_row_index,
                    source_row_fingerprint=row_fingerprint,
                    row=row,
                    away_decimal_odds=away_odds,
                    home_decimal_odds=home_odds,
                    canonical_home_code=home_code,
                    canonical_away_code=away_code,
                    official=None,
                    status="NO_UNIQUE_MLB_CROSSWALK",
                    crosswalk_time_delta_seconds=None,
                )
            )
            continue
        delta, _, official = candidates[0]
        final = bool(official.get("final")) and official.get("status") == "Final"
        observations.append(
            _CrosswalkObservation(
                source_row_index=source_row_index,
                source_row_fingerprint=row_fingerprint,
                row=row,
                away_decimal_odds=away_odds,
                home_decimal_odds=home_odds,
                canonical_home_code=home_code,
                canonical_away_code=away_code,
                official=official,
                status="MATCHED_FINAL" if final else "POSTPONED_OR_NON_FINAL",
                crosswalk_time_delta_seconds=delta,
            )
        )
    return tuple(observations)


def _raw_row(
    observation: _CrosswalkObservation,
    *,
    selected_source_row_indexes: frozenset[int],
    feature_unavailable_game_ids: frozenset[str],
) -> dict[str, Any]:
    official = observation.official
    official_id = str(official["provider_game_id"]) if official else None
    status = observation.status
    if official_id in feature_unavailable_game_ids:
        status = "FEATURE_UNAVAILABLE"
    return {
        "schema_version": P28AB_SCHEMA_VERSION,
        "source_row_index": observation.source_row_index,
        "source_row_fingerprint": observation.source_row_fingerprint,
        "tsl_source": str(observation.row["source"]),
        "tsl_match_id": str(observation.row["match_id"]),
        "tsl_fetched_at": str(observation.row["fetched_at"]),
        "tsl_game_time": str(observation.row["game_time"]),
        "source_home_code": str(observation.row["home_code"]),
        "source_away_code": str(observation.row["away_code"]),
        "canonical_home_code": observation.canonical_home_code,
        "canonical_away_code": observation.canonical_away_code,
        "official_game_id": official_id,
        "official_date": str(official["official_date"]) if official else None,
        "official_scheduled_start_utc": (
            str(official["scheduled_start_utc"]) if official else None
        ),
        "official_status": str(official["status"]) if official else None,
        "official_final": bool(official.get("final")) if official else None,
        "crosswalk_time_delta_seconds": observation.crosswalk_time_delta_seconds,
        "status": status,
        "selected_for_price": observation.source_row_index
        in selected_source_row_indexes,
    }


def _prepare_outcome_blind_schedule(
    schedule_rows: Sequence[Mapping[str, Any]],
    *,
    target_ids: frozenset[str],
    target_keys: frozenset[tuple[str, str, int]],
    official_start: str,
    official_end: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for source_row in schedule_rows:
        row = deepcopy(dict(source_row))
        game_id = str(row["provider_game_id"])
        official_date = str(row["official_date"])
        row_key = (
            game_id,
            str(row["scheduled_start_utc"]),
            int(row["game_number"]),
        )
        if row_key in target_keys:
            row["home_score"] = 0
            row["away_score"] = 0
            row["final"] = True
            row["status"] = "Final"
        elif game_id in target_ids:
            # A postponed and a completed MLB record can share the same
            # provider_game_id in the normalized history.  Only the exact
            # crosswalk-selected scheduled observation is a fold target.
            row["official_date"] = "0000-01-01"
            row["final"] = False
        elif official_start <= official_date <= official_end:
            # Keep completed non-cohort games available as prior history, but
            # keep them out of the fold target selector.  Postponed/non-final
            # rows must not leak into prior-form features either.
            row["official_date"] = "0000-01-01"
            if not (bool(row.get("final")) and row.get("status") == "Final"):
                row["final"] = False
        rows.append(row)
    return tuple(rows)


def _schedule_observation_id(row: FutureFeatureRow) -> str:
    projection = row.projection(include_fingerprint=False)
    projection.pop("features", None)
    projection.pop("feature_fingerprint", None)
    return sha256_bytes(canonical_json_bytes(projection))


def _snapshot_for_feature_row(
    row: FutureFeatureRow,
    *,
    batch_id: str,
) -> MoneylineFeatureSnapshot:
    as_of = _parse_utc(row.feature_as_of_utc)
    scheduled = _parse_utc(row.scheduled_start_utc)
    schedule_observation_id = _schedule_observation_id(row)
    features = row.projection(include_fingerprint=False)["features"]
    provenance = tuple(
        MoneylineFeatureProvenance(
            field_name=field_name,
            source_id=f"{schedule_observation_id}:{batch_id}:{field_name}",
            source_kind=P28AB_FEATURE_SOURCE_KIND,
            observed_as_of_utc=as_of,
            source_fingerprint=sha256_bytes(
                canonical_json_bytes(
                    {
                        "field_name": field_name,
                        "game_id": row.provider_game_id,
                        "schedule_observation_id": schedule_observation_id,
                        "value": features[field_name],
                    }
                )
            ),
        )
        for field_name in ("recent_win_rate_delta", "starter_era_delta")
    )
    return MoneylineFeatureSnapshot.from_record(
        features,
        identity=MatchIdentity(
            sport="baseball",
            league="MLB",
            season=int(row.official_date[:4]),
            canonical_game_id=f"MLB:{row.official_date}:{row.game_pk}:{row.game_number}",
            home_participant=row.home_team,
            away_participant=row.away_team,
        ),
        provider_namespace="MLB_STATS_API",
        provider_game_id=row.provider_game_id,
        game_number=row.game_number,
        source_schedule_observation_id=schedule_observation_id,
        as_of_utc=as_of,
        scheduled_start_utc=scheduled,
        feature_provenance=provenance,
    )


def _batch_id(
    *,
    crosswalk: Sequence[_CrosswalkObservation],
    target_ids: Sequence[str],
    source_manifest_fingerprint: str,
    model_id: str,
    model_fingerprint: str,
    cohort_start_date: str = P28AB_COHORT_START_DATE,
    cohort_end_date: str = P28AB_COHORT_END_DATE,
) -> str:
    membership = [
        {
            "source_row_index": item.source_row_index,
            "source_row_fingerprint": item.source_row_fingerprint,
            "official_game_id": (
                str(item.official["provider_game_id"]) if item.official else None
            ),
        }
        for item in crosswalk
    ]
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": P28AB_SCHEMA_VERSION,
                "cohort_start_date": cohort_start_date,
                "cohort_end_date": cohort_end_date,
                "source_manifest_fingerprint": source_manifest_fingerprint,
                "raw_source_membership": membership,
                "target_game_ids": list(target_ids),
                "model_id": model_id,
                "model_fingerprint": model_fingerprint,
            }
        )
    )


def _feature_unavailable_rows(
    eligibility: FutureFeatureEligibility,
    *,
    batch_id: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for source_row in eligibility.feature_unavailable_rows:
        affected = [dict(item) for item in source_row["affected_starters"]]
        affected_starter_ids = [
            int(item["starter_id"])
            for item in affected
            if item.get("starter_id") is not None
        ]
        prior_history = [
            {
                "starter_id": int(item["starter_id"]),
                "count": int(item["qualifying_prior_start_count"]),
                "required": int(item["required_prior_start_count"]),
            }
            for item in affected
            if item.get("starter_id") is not None
        ]
        required_history_count = [
            int(item["required_prior_start_count"])
            for item in affected
            if item.get("required_prior_start_count") is not None
        ]
        rows.append(
            {
                "schema_version": P28AB_SCHEMA_VERSION,
                "batch_id": batch_id,
                "fold_id": P28AB_FOLD_ID,
                "game_id": str(source_row["game_id"]),
                "scheduled_start": str(source_row["scheduled_start"]),
                "eligibility": "FEATURE_UNAVAILABLE",
                "status": "FEATURE_UNAVAILABLE",
                "reason": str(source_row["reason"]),
                "affected_feature": str(source_row["feature_name"]),
                "affected_starter_ids": affected_starter_ids,
                "affected_starters": affected,
                "prior_qualifying_history": prior_history,
                "required_history_count": (
                    max(required_history_count) if required_history_count else None
                ),
                "generated_from_historical_shadow": True,
            }
        )
    return tuple(sorted(rows, key=lambda row: (row["scheduled_start"], row["game_id"])))


def _prediction_rows(
    snapshots: Sequence[MoneylineFeatureSnapshot],
    *,
    artifact: MoneylineModelArtifact,
    artifact_fingerprint: str,
    batch_id: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        as_of = snapshot.as_of_utc
        inference = generate_moneyline_predictions(
            (snapshot,),
            artifact,
            prediction_generated_at_utc=as_of,
            response_received_at_utc=as_of,
            ingested_at_utc=as_of,
        )
        by_selection = {candidate.selection: candidate for candidate in inference.candidates}
        if set(by_selection) != {"HOME", "AWAY"}:
            raise RuntimeError(STOP_FEATURE_INPUT)
        home = by_selection["HOME"]
        away = by_selection["AWAY"]
        feature_fingerprint = snapshot.fingerprint()
        rows.append(
            {
                "schema_version": P28AB_SCHEMA_VERSION,
                "prediction_id": home.prediction_observation_id,
                "source_prediction_id": home.source_prediction_id,
                "batch_id": batch_id,
                "fold_id": P28AB_FOLD_ID,
                "game_id": snapshot.provider_game_id,
                "scheduled_start": _format_utc(snapshot.scheduled_start_utc),
                "prediction_cutoff_utc": _format_utc(snapshot.scheduled_start_utc),
                "home_team": snapshot.identity.home_participant,
                "away_team": snapshot.identity.away_participant,
                "feature_snapshot_id": feature_fingerprint,
                "feature_snapshot_fingerprint": feature_fingerprint,
                "model_id": artifact.model_id,
                "model_fingerprint": artifact_fingerprint,
                "inference_model_fingerprint": artifact.fingerprint(),
                "home_win_probability": str(home.model_probability),
                "away_win_probability": str(away.model_probability),
                "predicted_side": (
                    "HOME" if home.model_probability >= Decimal("0.5") else "AWAY"
                ),
                "inference_mode": "PAPER_DEFAULT",
                "generated_from_historical_shadow": True,
            }
        )
    return tuple(sorted(rows, key=lambda row: (row["scheduled_start"], row["game_id"])))


def _select_prices(
    crosswalk: Sequence[_CrosswalkObservation],
) -> tuple[tuple[dict[str, Any], ...], frozenset[int]]:
    grouped: dict[str, list[_CrosswalkObservation]] = defaultdict(list)
    for observation in crosswalk:
        if observation.status != "MATCHED_FINAL" or observation.official is None:
            continue
        official_id = str(observation.official["provider_game_id"])
        official_start = _parse_utc(str(observation.official["scheduled_start_utc"]))
        if _parse_utc(str(observation.row["fetched_at"])) < official_start:
            grouped[official_id].append(observation)
    selected: list[dict[str, Any]] = []
    selected_indexes: set[int] = set()
    for official_id, observations in grouped.items():
        observations.sort(
            key=lambda item: (
                _parse_utc(str(item.row["fetched_at"])),
                item.source_row_fingerprint,
            ),
            reverse=True,
        )
        observation = observations[0]
        selected_indexes.add(observation.source_row_index)
        official = observation.official
        cutoff = _parse_utc(str(official["scheduled_start_utc"]))
        away_implied = Decimal("1") / observation.away_decimal_odds
        home_implied = Decimal("1") / observation.home_decimal_odds
        total = away_implied + home_implied
        selected.append(
            {
                "schema_version": P28AB_SCHEMA_VERSION,
                "game_id": official_id,
                "official_date": str(official["official_date"]),
                "scheduled_start": str(official["scheduled_start_utc"]),
                "home_team": str(official["home_team"]["name"]),
                "away_team": str(official["away_team"]["name"]),
                "source_row_index": observation.source_row_index,
                "source_row_fingerprint": observation.source_row_fingerprint,
                "price_fetched_at": str(observation.row["fetched_at"]),
                "prediction_cutoff_utc": _format_utc(cutoff),
                "price_selection_rule": P28AB_PRICE_SELECTION_RULE,
                "home_decimal_odds": _decimal_string(observation.home_decimal_odds),
                "away_decimal_odds": _decimal_string(observation.away_decimal_odds),
                "home_decimal_implied_probability": _decimal_string(home_implied),
                "away_decimal_implied_probability": _decimal_string(away_implied),
                "normalization": P28AB_MARKET_NORMALIZATION,
                "home_normalized_implied_probability": _decimal_string(home_implied / total),
                "away_normalized_implied_probability": _decimal_string(away_implied / total),
            }
        )
    selected.sort(key=lambda row: (row["scheduled_start"], row["game_id"]))
    return tuple(selected), frozenset(selected_indexes)


def _edge_rows(
    predictions: Sequence[Mapping[str, Any]],
    prices: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    by_game = {str(row["game_id"]): row for row in prices}
    edges: list[dict[str, Any]] = []
    for prediction in predictions:
        price = by_game.get(str(prediction["game_id"]))
        if price is None:
            continue
        for selection, model_key, market_key in (
            (
                "HOME",
                "home_win_probability",
                "home_normalized_implied_probability",
            ),
            (
                "AWAY",
                "away_win_probability",
                "away_normalized_implied_probability",
            ),
        ):
            model_probability = Decimal(str(prediction[model_key]))
            normalized_market = Decimal(str(price[market_key]))
            edges.append(
                {
                    "schema_version": P28AB_SCHEMA_VERSION,
                    "prediction_id": prediction["prediction_id"],
                    "batch_id": prediction["batch_id"],
                    "fold_id": P28AB_FOLD_ID,
                    "game_id": prediction["game_id"],
                    "scheduled_start": prediction["scheduled_start"],
                    "selection": selection,
                    "model_probability": str(model_probability),
                    "decimal_implied_probability": price[
                        "home_decimal_implied_probability"
                        if selection == "HOME"
                        else "away_decimal_implied_probability"
                    ],
                    "normalized_implied_probability": str(normalized_market),
                    "edge": _decimal_string(model_probability - normalized_market),
                    "price_fetched_at": price["price_fetched_at"],
                    "prediction_cutoff_utc": price["prediction_cutoff_utc"],
                    "price_selection_rule": P28AB_PRICE_SELECTION_RULE,
                    "normalization": P28AB_MARKET_NORMALIZATION,
                    "edge_semantics": P28AB_EDGE_SEMANTICS,
                    "descriptive_only": True,
                }
            )
    return tuple(sorted(edges, key=lambda row: (row["scheduled_start"], row["game_id"], row["selection"])))


def _assemble(
    *,
    tsl_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
    target_boxscore_rows: Sequence[Mapping[str, Any]],
    pitcher_game_log_rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_manifest_fingerprint: str,
    artifact: MoneylineModelArtifact,
    artifact_fingerprint: str,
    batch_id: str,
    cohort_start_date: str = P28AB_COHORT_START_DATE,
    cohort_end_date: str = P28AB_COHORT_END_DATE,
    requested_game_ids: Sequence[str] | None = None,
    allow_missing_starter_identity: bool = False,
    allow_insufficient_evaluable: bool = False,
) -> _Assembly:
    crosswalk = _crosswalk(
        tsl_rows=tsl_rows,
        schedule_rows=schedule_rows,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
    )
    if requested_game_ids is None:
        target_officials = {
            str(item.official["provider_game_id"])
            for item in crosswalk
            if item.status == "MATCHED_FINAL" and item.official is not None
        }
    else:
        target_officials = {str(game_id) for game_id in requested_game_ids}
    if len(target_officials) < 2:
        raise RuntimeError(STOP_CROSSWALK_UNRESOLVED)
    if requested_game_ids is None:
        target_schedule_by_id = {
            str(item.official["provider_game_id"]): item.official
            for item in crosswalk
            if item.status == "MATCHED_FINAL" and item.official is not None
        }
    else:
        target_schedule_by_id = {
            str(row["provider_game_id"]): row
            for row in schedule_rows
            if str(row["provider_game_id"]) in target_officials
        }
    if set(target_schedule_by_id) != target_officials:
        raise RuntimeError(STOP_CROSSWALK_UNRESOLVED)
    target_schedule = tuple(target_schedule_by_id.values())
    official_start = min(str(row["official_date"]) for row in target_schedule)
    official_end = max(str(row["official_date"]) for row in target_schedule)
    blind_schedule = _prepare_outcome_blind_schedule(
        schedule_rows,
        target_ids=frozenset(target_officials),
        target_keys=frozenset(
            (
                str(row["provider_game_id"]),
                str(row["scheduled_start_utc"]),
                int(row["game_number"]),
            )
            for row in target_schedule
        ),
        official_start=official_start,
        official_end=official_end,
    )
    eligibility = classify_future_feature_eligibility(
        schedule_rows=blind_schedule,
        target_boxscore_rows=tuple(target_boxscore_rows),
        pitcher_game_log_rows=tuple(pitcher_game_log_rows),
        fold_id=P28AB_FOLD_ID,
        validation_start=official_start,
        validation_end=official_end,
        allow_missing_starter_identity=allow_missing_starter_identity,
    )
    if not eligibility.evaluable_game_ids and allow_insufficient_evaluable:
        snapshots = ()
    else:
        fold = materialize_future_moneyline_fold(
            schedule_rows=blind_schedule,
            target_boxscore_rows=tuple(target_boxscore_rows),
            pitcher_game_log_rows=tuple(pitcher_game_log_rows),
            source_manifest_fingerprint=source_manifest_fingerprint,
            fold_id=P28AB_FOLD_ID,
            validation_start=official_start,
            validation_end=official_end,
            evaluable_game_ids=frozenset(eligibility.evaluable_game_ids),
            raw_game_ids=eligibility.raw_game_ids,
            feature_unavailable_rows=eligibility.feature_unavailable_rows,
            allow_insufficient_evaluable=allow_insufficient_evaluable,
        )
        snapshots = tuple(
            _snapshot_for_feature_row(row, batch_id=batch_id)
            for row in fold.feature_rows
        )
    predictions = _prediction_rows(
        snapshots,
        artifact=artifact,
        artifact_fingerprint=artifact_fingerprint,
        batch_id=batch_id,
    )
    unavailable = _feature_unavailable_rows(eligibility, batch_id=batch_id)
    prices, selected_indexes = _select_prices(crosswalk)
    edges = _edge_rows(predictions, prices)
    unavailable_ids = frozenset(str(row["game_id"]) for row in unavailable)
    raw_cohort = tuple(
        _raw_row(
            observation,
            selected_source_row_indexes=selected_indexes,
            feature_unavailable_game_ids=unavailable_ids,
        )
        for observation in crosswalk
    )
    return _Assembly(
        raw_cohort=raw_cohort,
        prices=prices,
        predictions=predictions,
        edges=edges,
        feature_unavailable=unavailable,
        snapshots=snapshots,
        crosswalk=crosswalk,
    )


def _mutated_price_rows(
    tsl_rows: Sequence[Mapping[str, Any]],
    selected_source_row_fingerprint: str,
) -> tuple[dict[str, Any], ...]:
    rows = deepcopy([dict(row) for row in tsl_rows])
    for row in rows:
        if sha256_bytes(canonical_json_bytes(row)) != selected_source_row_fingerprint:
            continue
        for market in row["markets"]:
            if market.get("marketCode") == "MNL":
                odds = Decimal(str(market["outcomes"][0]["odds"])) + Decimal("0.11")
                market["outcomes"][0]["odds"] = _decimal_string(odds)
                return tuple(rows)
    raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)


def _manifest_for_fingerprint(
    source_manifest: Mapping[str, Any],
    *,
    crosswalk: Sequence[_CrosswalkObservation],
    target_game_ids: Sequence[str],
    cohort_start_date: str = P28AB_COHORT_START_DATE,
    cohort_end_date: str = P28AB_COHORT_END_DATE,
) -> dict[str, Any]:
    manifest = deepcopy(dict(source_manifest))
    manifest.pop("source_manifest_fingerprint", None)
    status_counts = Counter(item.status for item in crosswalk)
    manifest["p28ab_cohort"] = {
        "fold_id": P28AB_FOLD_ID,
        "game_time_start_date": cohort_start_date,
        "game_time_end_date": cohort_end_date,
        "fixture_row_count": len(crosswalk),
        "crosswalk_status_counts": dict(sorted(status_counts.items())),
        "matched_final_official_game_ids": list(sorted(target_game_ids)),
        "price_selection_rule": P28AB_PRICE_SELECTION_RULE,
        "prediction_cutoff_rule": "fetched_at < official scheduled_start_utc",
        "normalization": P28AB_MARKET_NORMALIZATION,
        "edge_semantics": P28AB_EDGE_SEMANTICS,
    }
    return manifest


def generate_tsl_moneyline_edge_batch(
    *,
    repository_root: str | Path,
    tsl_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
    target_boxscore_rows: Sequence[Mapping[str, Any]],
    pitcher_game_log_rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    tsl_raw_sha256: str | None = None,
    offline_replay_verified: bool,
    cohort_start_date: str = P28AB_COHORT_START_DATE,
    cohort_end_date: str = P28AB_COHORT_END_DATE,
    requested_game_ids: Sequence[str] | None = None,
    allow_missing_starter_identity: bool = False,
    allow_insufficient_evaluable: bool = False,
) -> TslMoneylineEdgeBatchResult:
    """Build one deterministic, outcome-blind P28AB edge slice."""

    manifest = _validate_source_manifest(
        source_manifest,
        tsl_raw_sha256=tsl_raw_sha256,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
    )
    # The caller supplies repository_root for deterministic resolution; keep
    # this separate from the source manifest so the manifest remains portable.
    repository_root_path = Path(repository_root)
    default_path = repository_root_path / P22B_ARTIFACT_RELATIVE_PATH
    try:
        artifact, artifact_fingerprint = load_model_artifact_with_fingerprint(default_path)
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(STOP_DEFAULT_MODEL_DRIFT) from exc
    if artifact.model_id != P22B_MODEL_ID or artifact_fingerprint != P22B_ARTIFACT_FINGERPRINT:
        raise RuntimeError(STOP_DEFAULT_MODEL_DRIFT)

    initial_crosswalk = _crosswalk(
        tsl_rows=tsl_rows,
        schedule_rows=schedule_rows,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
    )
    matched_target_ids = tuple(
        sorted(
            {
                str(item.official["provider_game_id"])
                for item in initial_crosswalk
                if item.status == "MATCHED_FINAL" and item.official is not None
            }
        )
    )
    target_ids = (
        tuple(sorted(str(game_id) for game_id in requested_game_ids))
        if requested_game_ids is not None
        else matched_target_ids
    )
    manifest_for_fingerprint = _manifest_for_fingerprint(
        manifest,
        crosswalk=initial_crosswalk,
        target_game_ids=target_ids,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
    )
    source_manifest_fingerprint = sha256_bytes(
        canonical_json_bytes(manifest_for_fingerprint)
    )
    batch_id = _batch_id(
        crosswalk=initial_crosswalk,
        target_ids=target_ids,
        source_manifest_fingerprint=source_manifest_fingerprint,
        model_id=artifact.model_id,
        model_fingerprint=artifact_fingerprint,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
    )
    base = _assemble(
        tsl_rows=tsl_rows,
        schedule_rows=schedule_rows,
        target_boxscore_rows=target_boxscore_rows,
        pitcher_game_log_rows=pitcher_game_log_rows,
        source_manifest=manifest_for_fingerprint,
        source_manifest_fingerprint=source_manifest_fingerprint,
        artifact=artifact,
        artifact_fingerprint=artifact_fingerprint,
        batch_id=batch_id,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
        requested_game_ids=requested_game_ids,
        allow_missing_starter_identity=allow_missing_starter_identity,
        allow_insufficient_evaluable=allow_insufficient_evaluable,
    )
    replay = _assemble(
        tsl_rows=tsl_rows,
        schedule_rows=schedule_rows,
        target_boxscore_rows=target_boxscore_rows,
        pitcher_game_log_rows=pitcher_game_log_rows,
        source_manifest=manifest_for_fingerprint,
        source_manifest_fingerprint=source_manifest_fingerprint,
        artifact=artifact,
        artifact_fingerprint=artifact_fingerprint,
        batch_id=batch_id,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
        requested_game_ids=requested_game_ids,
        allow_missing_starter_identity=allow_missing_starter_identity,
        allow_insufficient_evaluable=allow_insufficient_evaluable,
    )
    replay_equal = (
        base.raw_cohort == replay.raw_cohort
        and base.prices == replay.prices
        and base.predictions == replay.predictions
        and base.edges == replay.edges
        and base.feature_unavailable == replay.feature_unavailable
    )
    if offline_replay_verified and not replay_equal:
        raise RuntimeError(STOP_OFFLINE_REPLAY)

    mutated_schedule = deepcopy([dict(row) for row in schedule_rows])
    for row in mutated_schedule:
        if str(row["provider_game_id"]) in target_ids:
            row["home_score"] = 999
            row["away_score"] = 0
    outcome_mutation = _assemble(
        tsl_rows=tsl_rows,
        schedule_rows=tuple(mutated_schedule),
        target_boxscore_rows=target_boxscore_rows,
        pitcher_game_log_rows=pitcher_game_log_rows,
        source_manifest=manifest_for_fingerprint,
        source_manifest_fingerprint=source_manifest_fingerprint,
        artifact=artifact,
        artifact_fingerprint=artifact_fingerprint,
        batch_id=batch_id,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
        requested_game_ids=requested_game_ids,
        allow_missing_starter_identity=allow_missing_starter_identity,
        allow_insufficient_evaluable=allow_insufficient_evaluable,
    )
    outcome_isolation = (
        base.prices == outcome_mutation.prices
        and base.predictions == outcome_mutation.predictions
        and base.edges == outcome_mutation.edges
        and base.feature_unavailable == outcome_mutation.feature_unavailable
    )
    if not outcome_isolation:
        raise RuntimeError(STOP_FEATURE_INPUT)

    price_isolation = False
    priced_prediction = next(
        (
            price
            for price in base.prices
            if any(prediction["game_id"] == price["game_id"] for prediction in base.predictions)
        ),
        None,
    )
    if priced_prediction is None and not base.predictions and base.prices:
        priced_prediction = base.prices[0]
    if priced_prediction is not None:
        mutated_tsl = _mutated_price_rows(
            tsl_rows,
            str(priced_prediction["source_row_fingerprint"]),
        )
        price_mutation = _assemble(
            tsl_rows=mutated_tsl,
            schedule_rows=schedule_rows,
            target_boxscore_rows=target_boxscore_rows,
            pitcher_game_log_rows=pitcher_game_log_rows,
            source_manifest=manifest_for_fingerprint,
            source_manifest_fingerprint=source_manifest_fingerprint,
            artifact=artifact,
            artifact_fingerprint=artifact_fingerprint,
            batch_id=batch_id,
            cohort_start_date=cohort_start_date,
            cohort_end_date=cohort_end_date,
            requested_game_ids=requested_game_ids,
            allow_missing_starter_identity=allow_missing_starter_identity,
            allow_insufficient_evaluable=allow_insufficient_evaluable,
        )
        price_isolation = base.predictions == price_mutation.predictions and (
            base.edges != price_mutation.edges
            or (not base.predictions and base.prices != price_mutation.prices)
        )
    elif not base.prices and not base.edges:
        price_isolation = True
    if not price_isolation:
        raise RuntimeError(STOP_TSL_AUTHORITY_DRIFT)

    output_manifest = deepcopy(manifest_for_fingerprint)
    output_manifest["source_manifest_fingerprint"] = source_manifest_fingerprint
    output_manifest["batch_id"] = batch_id
    output_manifest["model_id"] = artifact.model_id
    output_manifest["model_fingerprint"] = artifact_fingerprint
    output_manifest["inference_model_fingerprint"] = artifact.fingerprint()

    status_counts = Counter(item.status for item in base.crosswalk)
    crosswalked_ids = {
        str(item.official["provider_game_id"])
        for item in base.crosswalk
        if item.official is not None
    }
    matched_final_ids = {
        str(item.official["provider_game_id"])
        for item in base.crosswalk
        if item.status == "MATCHED_FINAL" and item.official is not None
    }
    reason_counts = Counter(str(row["reason"]) for row in base.feature_unavailable)
    summary = {
        "schema_version": P28AB_SCHEMA_VERSION,
        "batch_id": batch_id,
        "fold_id": P28AB_FOLD_ID,
        "cohort_start_date": cohort_start_date,
        "cohort_end_date": cohort_end_date,
        "raw_source_row_count": len(base.raw_cohort),
        "crosswalk_status_counts": dict(sorted(status_counts.items())),
        "crosswalked_source_row_count": sum(
            count for status, count in status_counts.items() if status not in {"NO_CANONICAL_TEAM_CODE", "NO_UNIQUE_MLB_CROSSWALK"}
        ),
        "crosswalked_official_game_count": len(crosswalked_ids),
        "final_official_game_count": len(matched_final_ids),
        "selected_price_count": len(base.prices),
        "raw_game_count": len(target_ids),
        "evaluable_game_count": len(base.predictions),
        "feature_unavailable_count": len(base.feature_unavailable),
        "feature_unavailable_reason_counts": dict(sorted(reason_counts.items())),
        "edge_row_count": len(base.edges),
        "raw_cohort_fingerprint": sha256_bytes(render_jsonl(base.raw_cohort)),
        "price_set_fingerprint": sha256_bytes(render_jsonl(base.prices)),
        "prediction_set_fingerprint": sha256_bytes(render_jsonl(base.predictions)),
        "edge_set_fingerprint": sha256_bytes(render_jsonl(base.edges)),
        "feature_unavailable_set_fingerprint": sha256_bytes(
            render_jsonl(base.feature_unavailable)
        ),
        "feature_snapshot_set_fingerprint": sha256_bytes(
            b"".join(snapshot.canonical_bytes() for snapshot in base.snapshots)
        ),
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "promoted_default_model_id": artifact.model_id,
        "promoted_default_model_fingerprint": artifact_fingerprint,
        "promoted_default_inference_model_fingerprint": artifact.fingerprint(),
        "prediction_cutoff_rule": "strict fetched_at < official scheduled_start_utc",
        "price_selection_rule": P28AB_PRICE_SELECTION_RULE,
        "normalization": P28AB_MARKET_NORMALIZATION,
        "edge_semantics": P28AB_EDGE_SEMANTICS,
        "outcome_blind_feature_generation": True,
        "result_mutation_isolation_verified": outcome_isolation,
        "price_mutation_isolation_verified": price_isolation,
        "offline_replay_verified": bool(offline_replay_verified and replay_equal),
        "historical_shadow": True,
        "paper_only": True,
        "production_ready": False,
        "deployment_performed": False,
        "real_betting_recommendation": False,
        "profitability_claim": False,
        "ev_or_kelly_claim": False,
        "claims": {
            "historical_shadow": True,
            "paper_only": True,
            "diagnostic_only": True,
            "production_ready": False,
            "deployment_performed": False,
            "real_betting_recommendation": False,
            "profitability_claim": False,
            "ev_or_kelly_claim": False,
        },
    }
    return TslMoneylineEdgeBatchResult(
        raw_cohort=base.raw_cohort,
        prices=base.prices,
        predictions=base.predictions,
        edges=base.edges,
        feature_unavailable=base.feature_unavailable,
        source_manifest=output_manifest,
        summary=summary,
    )


def _report_markdown(summary: Mapping[str, Any]) -> bytes:
    lines = [
        "# P28AB TSL-aligned Moneyline edge batch",
        "",
        "This artifact is a deterministic historical-shadow replay for paper and diagnostic use only.",
        "",
        f"- Cohort: `{summary['cohort_start_date']}..{summary['cohort_end_date']}`",
        f"- Raw TSL source rows: `{summary['raw_source_row_count']}`",
        f"- Final MLB games: `{summary['final_official_game_count']}`",
        f"- Selected pre-cutoff prices: `{summary['selected_price_count']}`",
        f"- Evaluable predictions: `{summary['evaluable_game_count']}`",
        f"- Feature unavailable: `{summary['feature_unavailable_count']}`",
        f"- Descriptive edge rows: `{summary['edge_row_count']}`",
        f"- Price rule: `{summary['price_selection_rule']}`",
        f"- Normalization: `{summary['normalization']}`",
        f"- Outcome mutation isolation: `{summary['result_mutation_isolation_verified']}`",
        f"- Price mutation isolation: `{summary['price_mutation_isolation_verified']}`",
        f"- Offline replay: `{summary['offline_replay_verified']}`",
        "",
        "No profitability, EV, Kelly, production, deployment, or real-betting claim is made.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def write_tsl_moneyline_edge_batch_artifacts(
    output_dir: str | Path,
    *,
    result: TslMoneylineEdgeBatchResult,
) -> dict[str, str]:
    """Write the seven deterministic P28AB report ledgers."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "raw_cohort.jsonl": render_jsonl(result.raw_cohort),
        "prices.jsonl": render_jsonl(result.prices),
        "predictions.jsonl": render_jsonl(result.predictions),
        "edges.jsonl": render_jsonl(result.edges),
        "feature_unavailable.jsonl": render_jsonl(result.feature_unavailable),
        "source_manifest.json": (
            json.dumps(result.source_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
        "summary.json": (
            json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "report.md": _report_markdown(result.summary),
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    return {name: sha256_bytes(content) for name, content in files.items()}


__all__ = (
    "P28AB_COHORT_END_DATE",
    "P28AB_COHORT_START_DATE",
    "P28AB_EDGE_SEMANTICS",
    "P28AB_FOLD_ID",
    "P28AB_MARKET_NORMALIZATION",
    "P28AB_PRICE_SELECTION_RULE",
    "P28AB_SCHEMA_VERSION",
    "P28AB_TSL_BLOB_ID",
    "P28AB_TSL_RAW_SHA256",
    "P28AB_TSL_PATH",
    "P28AB_TSL_REF",
    "TslMoneylineEdgeBatchResult",
    "generate_tsl_moneyline_edge_batch",
    "write_tsl_moneyline_edge_batch_artifacts",
)
