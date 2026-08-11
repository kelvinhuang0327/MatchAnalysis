"""Compose the committed P24C/P28AB Moneyline capabilities into one run."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .generate_tsl_moneyline_edge_batch import (
    P28AB_COHORT_END_DATE,
    P28AB_COHORT_START_DATE,
    P28AB_EDGE_SEMANTICS,
    P28AB_MARKET_NORMALIZATION,
    P28AB_PRICE_SELECTION_RULE,
    P28AB_SCHEMA_VERSION,
    TslMoneylineEdgeBatchResult,
    generate_tsl_moneyline_edge_batch,
)
from .paper_moneyline_batch_artifacts import (
    P22B_ARTIFACT_FINGERPRINT,
    P22B_MODEL_ID,
    canonical_json_bytes,
    render_jsonl,
    sha256_bytes,
)


P30A_SCHEMA_VERSION = "p30a.moneyline_paper_analysis.v1"
P30A_OPERATION = "MONEYLINE_PAPER_ANALYSIS_RUN"
P30A_STATUS_EDGE_AVAILABLE = "EDGE_AVAILABLE"
P30A_STATUS_FEATURE_UNAVAILABLE = "FEATURE_UNAVAILABLE"
P30A_STATUS_PRICE_UNAVAILABLE = "PRICE_UNAVAILABLE_PRE_CUTOFF"
P30A_STATUS_CROSSWALK_UNRESOLVED = "CROSSWALK_UNRESOLVED"
P30A_STATUSES = (
    P30A_STATUS_EDGE_AVAILABLE,
    P30A_STATUS_FEATURE_UNAVAILABLE,
    P30A_STATUS_PRICE_UNAVAILABLE,
    P30A_STATUS_CROSSWALK_UNRESOLVED,
)
P30A_STOP_MANDATORY_VERIFICATION = (
    "STOP_MATCHANALYSIS_P30A_MANDATORY_VERIFICATION_FAILED"
)

_OUTCOME_FIELDS = frozenset(
    {
        "away_score",
        "final",
        "result",
        "runs",
        "winner",
        "home_score",
    }
)
_DOWNSTREAM_FIELDS = frozenset(
    {
        "bankroll",
        "bet",
        "clv",
        "kelly",
        "roi",
        "settlement",
        "stake",
    }
)


@dataclass(frozen=True, slots=True)
class MoneylinePaperAnalysisRunResult:
    """One deterministic game-level P30A analysis artifact."""

    analysis: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def _stop(detail: str) -> RuntimeError:
    return RuntimeError(f"{P30A_STOP_MANDATORY_VERIFICATION}: {detail}")


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            game_id = str(row["game_id"])
        except (KeyError, TypeError) as exc:
            raise _stop(f"{label} missing game_id") from exc
        if not game_id or game_id in indexed:
            raise _stop(f"{label} has duplicate or empty game_id")
        indexed[game_id] = dict(row)
    return indexed


def _index_edges(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        try:
            game_id = str(row["game_id"])
            selection = str(row["selection"])
        except (KeyError, TypeError) as exc:
            raise _stop("edge row missing game_id or selection") from exc
        if selection in indexed[game_id]:
            raise _stop(f"duplicate edge selection for {game_id}")
        indexed[game_id][selection] = dict(row)
    return dict(indexed)


def _consistent_value(
    rows: Sequence[Mapping[str, Any]],
    field_name: str,
) -> str | None:
    values = {
        str(row[field_name])
        for row in rows
        if row.get(field_name) is not None
    }
    if len(values) > 1:
        raise _stop(f"conflicting {field_name}")
    return next(iter(values), None)


def _game_membership(
    result: TslMoneylineEdgeBatchResult,
) -> tuple[dict[str, str], ...]:
    prices = _index_unique(result.prices, label="P28AB prices")
    predictions = _index_unique(result.predictions, label="P28AB predictions")
    unavailable = _index_unique(
        result.feature_unavailable,
        label="P28AB feature-unavailable rows",
    )
    game_ids = sorted(set(prices) | set(predictions) | set(unavailable))
    try:
        expected_count = int(result.summary["raw_game_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _stop("P28AB raw_game_count is missing") from exc
    if len(game_ids) != expected_count:
        raise _stop(
            f"raw game accounting expected {expected_count}, found {len(game_ids)}"
        )

    membership: list[dict[str, str]] = []
    for game_id in game_ids:
        rows = [
            row
            for row in (
                prices.get(game_id),
                predictions.get(game_id),
                unavailable.get(game_id),
            )
            if row is not None
        ]
        scheduled_start = _consistent_value(rows, "scheduled_start")
        if scheduled_start is None:
            raise _stop(f"{game_id} has no scheduled_start")
        membership.append(
            {"game_id": game_id, "scheduled_start": scheduled_start}
        )
    return tuple(
        sorted(membership, key=lambda row: (row["scheduled_start"], row["game_id"]))
    )


def _run_fingerprint(
    *,
    p28ab_result: TslMoneylineEdgeBatchResult,
    membership: Sequence[Mapping[str, str]],
) -> str:
    summary = p28ab_result.summary
    authority = p28ab_result.source_manifest.get("tsl_authority")
    if not isinstance(authority, Mapping):
        raise _stop("P28AB TSL authority is missing")
    projection = {
        "operation": P30A_OPERATION,
        "schema_version": P30A_SCHEMA_VERSION,
        "source_manifest_fingerprint": summary.get("source_manifest_fingerprint"),
        "p28ab_batch_id": summary.get("batch_id"),
        "p28ab_schema_version": P28AB_SCHEMA_VERSION,
        "tsl_authority": dict(authority),
        "model_id": summary.get("promoted_default_model_id"),
        "model_fingerprint": summary.get("promoted_default_model_fingerprint"),
        "market_price_authority": {
            "selection_rule": P28AB_PRICE_SELECTION_RULE,
            "normalization": P28AB_MARKET_NORMALIZATION,
            "edge_semantics": P28AB_EDGE_SEMANTICS,
        },
        "game_membership": [dict(row) for row in membership],
    }
    if projection["source_manifest_fingerprint"] is None:
        raise _stop("P28AB source manifest fingerprint is missing")
    if projection["model_id"] != P22B_MODEL_ID:
        raise _stop("promoted model identity drifted")
    if projection["model_fingerprint"] != P22B_ARTIFACT_FINGERPRINT:
        raise _stop("promoted model fingerprint drifted")
    return sha256_bytes(canonical_json_bytes(projection))


def _analysis_rows(
    *,
    p28ab_result: TslMoneylineEdgeBatchResult,
    membership: Sequence[Mapping[str, str]],
    run_id: str,
) -> tuple[dict[str, Any], ...]:
    prices = _index_unique(p28ab_result.prices, label="P28AB prices")
    predictions = _index_unique(
        p28ab_result.predictions,
        label="P28AB predictions",
    )
    unavailable = _index_unique(
        p28ab_result.feature_unavailable,
        label="P28AB feature-unavailable rows",
    )
    edges = _index_edges(p28ab_result.edges)
    model_id = str(p28ab_result.summary["promoted_default_model_id"])
    model_fingerprint = str(
        p28ab_result.summary["promoted_default_model_fingerprint"]
    )
    rows: list[dict[str, Any]] = []

    for member in membership:
        game_id = member["game_id"]
        price = prices.get(game_id)
        prediction = predictions.get(game_id)
        feature_unavailable = unavailable.get(game_id)
        edge_by_selection = edges.get(game_id, {})

        if feature_unavailable is not None:
            status = P30A_STATUS_FEATURE_UNAVAILABLE
            unavailable_reason = str(feature_unavailable["reason"])
        elif price is None and prediction is not None:
            status = P30A_STATUS_PRICE_UNAVAILABLE
            unavailable_reason = P30A_STATUS_PRICE_UNAVAILABLE
        elif prediction is not None and price is not None:
            status = P30A_STATUS_EDGE_AVAILABLE
            unavailable_reason = None
        else:
            status = P30A_STATUS_CROSSWALK_UNRESOLVED
            unavailable_reason = P30A_STATUS_CROSSWALK_UNRESOLVED

        source_rows = [row for row in (price, prediction, feature_unavailable) if row]
        scheduled_start = _consistent_value(source_rows, "scheduled_start")
        if scheduled_start != member["scheduled_start"]:
            raise _stop(f"{game_id} scheduled_start mismatch")
        home_team = _consistent_value(source_rows, "home_team")
        away_team = _consistent_value(source_rows, "away_team")

        if status == P30A_STATUS_EDGE_AVAILABLE and set(edge_by_selection) != {
            "HOME",
            "AWAY",
        }:
            raise _stop(f"{game_id} edge row accounting is incomplete")
        if status != P30A_STATUS_EDGE_AVAILABLE and edge_by_selection:
            raise _stop(f"{game_id} has an edge without EDGE_AVAILABLE status")

        market_price_id = (
            f"p28ab:{price['source_row_fingerprint']}" if price is not None else None
        )
        row = {
            "schema_version": P30A_SCHEMA_VERSION,
            "run_id": run_id,
            "game_id": game_id,
            "scheduled_start": scheduled_start,
            "home_team": home_team,
            "away_team": away_team,
            "structural_status": status,
            "status": status,
            "prediction_id": prediction.get("prediction_id") if prediction else None,
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "model_home_probability": (
                prediction.get("home_win_probability") if prediction else None
            ),
            "market_price_id": market_price_id,
            "price_observed_at": (
                price.get("price_fetched_at") if price is not None else None
            ),
            "home_decimal_odds": (
                price.get("home_decimal_odds") if price is not None else None
            ),
            "away_decimal_odds": (
                price.get("away_decimal_odds") if price is not None else None
            ),
            "home_no_vig_probability": (
                price.get("home_normalized_implied_probability")
                if price is not None
                else None
            ),
            "away_no_vig_probability": (
                price.get("away_normalized_implied_probability")
                if price is not None
                else None
            ),
            "home_edge": (
                edge_by_selection["HOME"].get("edge")
                if "HOME" in edge_by_selection
                else None
            ),
            "away_edge": (
                edge_by_selection["AWAY"].get("edge")
                if "AWAY" in edge_by_selection
                else None
            ),
            "controlled_unavailable_reason": unavailable_reason,
        }
        if _OUTCOME_FIELDS.intersection(row) or _DOWNSTREAM_FIELDS.intersection(row):
            raise _stop(f"{game_id} contains downstream fields")
        rows.append(row)
    return tuple(rows)


def _p28ab_projection(result: TslMoneylineEdgeBatchResult) -> dict[str, Any]:
    return {
        "raw_cohort": [dict(row) for row in result.raw_cohort],
        "prices": [dict(row) for row in result.prices],
        "predictions": [dict(row) for row in result.predictions],
        "edges": [dict(row) for row in result.edges],
        "feature_unavailable": [dict(row) for row in result.feature_unavailable],
        "source_manifest": deepcopy(result.source_manifest),
        "summary": deepcopy(result.summary),
    }


def _summary(
    *,
    p28ab_result: TslMoneylineEdgeBatchResult,
    analysis: Sequence[Mapping[str, Any]],
    run_id: str,
    replay_equal: bool,
    offline_replay_verified: bool,
) -> dict[str, Any]:
    status_counts = Counter(str(row["structural_status"]) for row in analysis)
    structural_status_counts = {
        status: status_counts.get(status, 0) for status in P30A_STATUSES
    }
    source_crosswalk_counts = p28ab_result.summary.get("crosswalk_status_counts", {})
    if not isinstance(source_crosswalk_counts, Mapping):
        raise _stop("P28AB crosswalk status counts are missing")
    crosswalk_status_counts = {
        str(key): int(value) for key, value in sorted(source_crosswalk_counts.items())
    }
    source_exclusions = {
        key: value
        for key, value in crosswalk_status_counts.items()
        if key != "MATCHED_FINAL"
    }
    summary = {
        "schema_version": P30A_SCHEMA_VERSION,
        "operation": P30A_OPERATION,
        "run_id": run_id,
        "run_fingerprint": run_id,
        "cohort_start_date": p28ab_result.summary["cohort_start_date"],
        "cohort_end_date": p28ab_result.summary["cohort_end_date"],
        "source_manifest_fingerprint": p28ab_result.summary[
            "source_manifest_fingerprint"
        ],
        "p28ab_batch_id": p28ab_result.summary["batch_id"],
        "p28ab_schema_version": P28AB_SCHEMA_VERSION,
        "p28ab_raw_cohort_fingerprint": p28ab_result.summary[
            "raw_cohort_fingerprint"
        ],
        "raw_game_count": len(analysis),
        "edge_available_count": structural_status_counts[P30A_STATUS_EDGE_AVAILABLE],
        "feature_unavailable_count": structural_status_counts[
            P30A_STATUS_FEATURE_UNAVAILABLE
        ],
        "price_unavailable_pre_cutoff_count": structural_status_counts[
            P30A_STATUS_PRICE_UNAVAILABLE
        ],
        "crosswalk_unresolved_count": structural_status_counts[
            P30A_STATUS_CROSSWALK_UNRESOLVED
        ],
        "other_structural_exclusion_count": structural_status_counts[
            P30A_STATUS_CROSSWALK_UNRESOLVED
        ],
        "structural_status_counts": structural_status_counts,
        "p28ab_crosswalk_status_counts": crosswalk_status_counts,
        "other_structural_exclusion_counts": source_exclusions,
        "p28ab_raw_source_row_count": p28ab_result.summary["raw_source_row_count"],
        "p28ab_selected_price_count": p28ab_result.summary["selected_price_count"],
        "p28ab_descriptive_edge_row_count": p28ab_result.summary["edge_row_count"],
        "analysis_set_fingerprint": sha256_bytes(render_jsonl(analysis)),
        "promoted_default_model_id": p28ab_result.summary[
            "promoted_default_model_id"
        ],
        "promoted_default_model_fingerprint": p28ab_result.summary[
            "promoted_default_model_fingerprint"
        ],
        "promoted_default_inference_model_fingerprint": p28ab_result.summary[
            "promoted_default_inference_model_fingerprint"
        ],
        "market_price_authority": deepcopy(
            p28ab_result.source_manifest["tsl_authority"]
        ),
        "price_selection_rule": P28AB_PRICE_SELECTION_RULE,
        "normalization": P28AB_MARKET_NORMALIZATION,
        "edge_semantics": P28AB_EDGE_SEMANTICS,
        "deterministic_replay_verified": bool(
            offline_replay_verified and replay_equal
        ),
        "outcome_isolation_verified": bool(
            p28ab_result.summary["result_mutation_isolation_verified"]
        ),
        "historical_shadow": True,
        "paper_only": True,
        "moneyline_model_promoted": True,
        "moneyline_promotion_scope": "paper_only",
        "decision_policy_used": False,
        "staking_implemented": False,
        "profitability_claim": False,
        "real_betting_recommendation": False,
        "p20b_historical_runtime_compliance": "REMAINS_REFUTED",
        "settlement_included": False,
        "clv_included": False,
        "claims": {
            "historical_shadow": True,
            "paper_only": True,
            "moneyline_model_promoted": True,
            "moneyline_promotion_scope": "paper_only",
            "decision_policy_used": False,
            "staking_implemented": False,
            "profitability_claim": False,
            "real_betting_recommendation": False,
            "settlement_included": False,
            "clv_included": False,
        },
        "run_line_migration_status": "BLOCKED_NO_PIT_SAFE_AUTHORITY",
        "total_migration_status": "BLOCKED_NO_PIT_SAFE_AUTHORITY",
        "legacy_decision_policy_status": "BLOCKED_NO_PIT_SAFE_AUTHORITY",
    }
    if summary["raw_game_count"] != sum(structural_status_counts.values()):
        raise _stop("structural status accounting is incomplete")
    generalization_window = p28ab_result.source_manifest.get(
        "p31a_generalization_window"
    )
    if isinstance(generalization_window, Mapping):
        summary["p31a_generalization_window"] = deepcopy(dict(generalization_window))
    return summary


def run_moneyline_paper_analysis(
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
) -> MoneylinePaperAnalysisRunResult:
    """Build one outcome-blind P30A run from the existing P28AB service."""

    first = generate_tsl_moneyline_edge_batch(
        repository_root=repository_root,
        tsl_rows=tsl_rows,
        schedule_rows=schedule_rows,
        target_boxscore_rows=target_boxscore_rows,
        pitcher_game_log_rows=pitcher_game_log_rows,
        source_manifest=source_manifest,
        tsl_raw_sha256=tsl_raw_sha256,
        offline_replay_verified=offline_replay_verified,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
        requested_game_ids=requested_game_ids,
        allow_missing_starter_identity=allow_missing_starter_identity,
        allow_insufficient_evaluable=allow_insufficient_evaluable,
    )
    second = generate_tsl_moneyline_edge_batch(
        repository_root=repository_root,
        tsl_rows=tsl_rows,
        schedule_rows=schedule_rows,
        target_boxscore_rows=target_boxscore_rows,
        pitcher_game_log_rows=pitcher_game_log_rows,
        source_manifest=source_manifest,
        tsl_raw_sha256=tsl_raw_sha256,
        offline_replay_verified=offline_replay_verified,
        cohort_start_date=cohort_start_date,
        cohort_end_date=cohort_end_date,
        requested_game_ids=requested_game_ids,
        allow_missing_starter_identity=allow_missing_starter_identity,
        allow_insufficient_evaluable=allow_insufficient_evaluable,
    )
    replay_equal = _p28ab_projection(first) == _p28ab_projection(second)
    if offline_replay_verified and not replay_equal:
        raise _stop("two-run replay differs")

    membership = _game_membership(first)
    run_id = _run_fingerprint(p28ab_result=first, membership=membership)
    analysis = _analysis_rows(
        p28ab_result=first,
        membership=membership,
        run_id=run_id,
    )
    summary = _summary(
        p28ab_result=first,
        analysis=analysis,
        run_id=run_id,
        replay_equal=replay_equal,
        offline_replay_verified=offline_replay_verified,
    )
    return MoneylinePaperAnalysisRunResult(analysis=analysis, summary=summary)


__all__ = (
    "MoneylinePaperAnalysisRunResult",
    "P30A_OPERATION",
    "P30A_SCHEMA_VERSION",
    "P30A_STATUS_CROSSWALK_UNRESOLVED",
    "P30A_STATUS_EDGE_AVAILABLE",
    "P30A_STATUS_FEATURE_UNAVAILABLE",
    "P30A_STATUS_PRICE_UNAVAILABLE",
    "P30A_STOP_MANDATORY_VERIFICATION",
    "run_moneyline_paper_analysis",
)
