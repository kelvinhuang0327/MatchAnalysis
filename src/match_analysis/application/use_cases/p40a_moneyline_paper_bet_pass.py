"""P40A deterministic Moneyline paper BET/PASS orchestration.

This use case reads the committed P39A market join and P37A comparison
authority, freezes Champion-primary and raw-challenger-shadow decisions from
pregame inputs, and only then attaches the committed final outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.paper_moneyline_bet_pass import (
    P40A_POLICY_ID,
    PaperMoneylineDecision,
    PaperMoneylineSettlement,
    aggregate_paper_settlements,
    settle_paper_moneyline_decision,
)


P40A_TASK_ID = "P40A"
P40A_REPORT_RELATIVE_PATH = Path("report/p40a_moneyline_paper_bet_pass")
P40A_ARTIFACT_FILES = (
    "source_manifest.json",
    "decisions.jsonl",
    "settlements.jsonl",
    "summary.json",
    "report.md",
)

P39A_REPORT_RELATIVE_PATH = Path("report/p39a_tsl_moneyline_market_join")
P37A_REPORT_RELATIVE_PATH = Path("report/p37a_rolling_walk_forward_oos")
P38A_REPORT_RELATIVE_PATH = Path("report/p38a_rolling_probability_calibration")
P40A_EXPECTED_P37_TARGET_COUNT = 65
P40A_EXPECTED_P39_EDGE_READY_COUNT = 62
P40A_EXPECTED_P39_NO_MARKET_COUNT = 3
P40A_EXPECTED_P37_COMPARISONS_SHA256 = (
    "23cc15d308a90c08da0d1a4c6cbb9289af3add2c5d151808833e73a660639eb4"
)
P40A_EXPECTED_TSL_SOURCE_SHA256 = (
    "5604e41f817f87617956b54c4b664bbf562d496eb1c8618bd174888ef87c8efc"
)
P40A_OUTCOME_AUTHORITY = "P37A_COMMITTED_TRUE_OOS_COMPARISON_ACTUAL_WINNER"
P40A_CHAMPION_ROLE = "CHAMPION_PRIMARY"
P40A_SHADOW_ROLE = "RAW_CHALLENGER_SHADOW"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _duplicate_rejecting_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_duplicate_rejecting_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path} contains a blank row at line {line_number}")
        value = json.loads(line, object_pairs_hook=_duplicate_rejecting_pairs)
        if not isinstance(value, dict):
            raise ValueError(f"{path} row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a trimmed non-empty string")
    return value


def _sha(value: Any, *, field_name: str) -> str:
    result = _text(value, field_name=field_name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return result


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _decimal(value: Any, *, field_name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class P40AMarketRow:
    """One usable pregame row from the committed P39A join."""

    p37_fold_id: str
    p37_window: str
    p37_prediction_row_id: str
    provider_namespace: str
    provider_game_id: str
    game_pk: int
    game_number: int
    official_date: str
    scheduled_start_utc: str
    home_team: str
    away_team: str
    home_team_code: str
    away_team_code: str
    market_snapshot_id: str
    market_observed_at_utc: str
    local_fetched_at_utc: str
    source_match_id: str
    home_decimal_odds: Decimal
    away_decimal_odds: Decimal


@dataclass(frozen=True, slots=True)
class P40APredictionRow:
    """Outcome-free Champion and raw-challenger probabilities from P37A."""

    p37_fold_id: str
    p37_window: str
    p37_prediction_row_id: str
    provider_namespace: str
    provider_game_id: str
    game_pk: int
    game_number: int
    scheduled_start_utc: str
    champion_model_id: str
    champion_model_fingerprint: str
    champion_home_probability: Decimal
    challenger_model_id: str
    challenger_model_fingerprint: str
    challenger_home_probability: Decimal


@dataclass(frozen=True, slots=True)
class P40AOutcomeRow:
    """One downstream final outcome from the committed P37A comparison."""

    p37_prediction_row_id: str
    provider_game_id: str
    actual_winner: str
    target_home_win: int


@dataclass(frozen=True, slots=True)
class P40AAuthority:
    repository_root: Path
    p39_summary: dict[str, Any]
    p39_source_manifest: dict[str, Any]
    p37_summary: dict[str, Any]
    market_rows: tuple[P40AMarketRow, ...]
    prediction_rows: tuple[P40APredictionRow, ...]
    outcome_rows: tuple[P40AOutcomeRow, ...]
    source_manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class P40AResult:
    authority: P40AAuthority
    decisions: tuple[PaperMoneylineDecision, ...]
    settlements: tuple[PaperMoneylineSettlement, ...]
    summary: dict[str, Any]


def _validate_p39_authority(root: Path) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]
]:
    report = root / P39A_REPORT_RELATIVE_PATH
    summary_path = report / "summary.json"
    source_manifest_path = report / "source_manifest.json"
    market_join_path = report / "market_join.jsonl"
    market_snapshots_path = report / "market_snapshots.jsonl"
    summary = _read_json(summary_path)
    source_manifest = _read_json(source_manifest_path)
    if summary.get("schema_version") != "p39a.p37_oos_moneyline_market_join.v1":
        raise ValueError("P40A P39A schema authority drift")
    if summary.get("source_manifest") != source_manifest:
        raise ValueError("P40A P39A source manifest projection drift")
    if summary.get("p37_evaluable_target_count") != P40A_EXPECTED_P37_TARGET_COUNT:
        raise ValueError("P40A P39A target universe must remain 65 rows")
    if summary.get("edge_ready_count") != P40A_EXPECTED_P39_EDGE_READY_COUNT:
        raise ValueError("P40A P39A edge-ready count must remain 62 rows")
    if summary.get("no_market_rows") != P40A_EXPECTED_P39_NO_MARKET_COUNT:
        raise ValueError("P40A P39A no-market count must remain three rows")
    if summary.get("conclusion") != "MARKET_JOIN_PARTIAL":
        raise ValueError("P40A requires the authoritative P39A partial market join")
    if not source_manifest.get("source_stable") or not source_manifest.get(
        "timestamp_semantics_trusted"
    ):
        raise ValueError("P40A P39A source stability/timestamp authority is unresolved")
    if source_manifest.get("source_sha256") != P40A_EXPECTED_TSL_SOURCE_SHA256:
        raise ValueError("P40A P39A legacy source hash drift")

    market_join_bytes = market_join_path.read_bytes()
    snapshot_bytes = market_snapshots_path.read_bytes()
    if _sha256_bytes(market_join_bytes) != summary.get("market_join_fingerprint"):
        raise ValueError("P40A P39A market join fingerprint drift")
    if _sha256_bytes(snapshot_bytes) != summary.get("selected_snapshot_fingerprint"):
        raise ValueError("P40A P39A selected snapshot fingerprint drift")
    rows = _read_jsonl(market_join_path)
    if len(rows) != P40A_EXPECTED_P37_TARGET_COUNT:
        raise ValueError("P40A P39A market join must contain exactly 65 rows")
    counts: dict[str, int] = {}
    for row in rows:
        status = _text(row.get("market_snapshot_status"), field_name="market_snapshot_status")
        counts[status] = counts.get(status, 0) + 1
    if counts != {"NO_MARKET": P40A_EXPECTED_P39_NO_MARKET_COUNT, "USABLE_PREGAME": P40A_EXPECTED_P39_EDGE_READY_COUNT}:
        raise ValueError(f"P40A P39A status accounting drift: {counts}")

    hashes = {
        "p39_summary_sha256": _sha256_path(summary_path),
        "p39_source_manifest_sha256": _sha256_path(source_manifest_path),
        "p39_market_join_sha256": _sha256_path(market_join_path),
        "p39_market_snapshots_sha256": _sha256_path(market_snapshots_path),
    }
    return summary, source_manifest, rows, hashes


def _load_p39_market_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[P40AMarketRow, ...]:
    market_rows: list[P40AMarketRow] = []
    for row in rows:
        if row.get("market_snapshot_status") != "USABLE_PREGAME":
            continue
        market_rows.append(
            P40AMarketRow(
                p37_fold_id=_text(row.get("p37_fold_id"), field_name="p37_fold_id"),
                p37_window=_text(row.get("p37_window"), field_name="p37_window"),
                p37_prediction_row_id=_sha(
                    row.get("p37_prediction_row_id"), field_name="p37_prediction_row_id"
                ),
                provider_namespace=_text(
                    row.get("provider_namespace"), field_name="provider_namespace"
                ),
                provider_game_id=_text(row.get("provider_game_id"), field_name="provider_game_id"),
                game_pk=_positive_int(row.get("game_pk"), field_name="game_pk"),
                game_number=_positive_int(row.get("game_number"), field_name="game_number"),
                official_date=_text(row.get("official_date"), field_name="official_date"),
                scheduled_start_utc=_text(
                    row.get("scheduled_start_utc"), field_name="scheduled_start_utc"
                ),
                home_team=_text(row.get("home_team"), field_name="home_team"),
                away_team=_text(row.get("away_team"), field_name="away_team"),
                home_team_code=_text(row.get("home_team_code"), field_name="home_team_code"),
                away_team_code=_text(row.get("away_team_code"), field_name="away_team_code"),
                market_snapshot_id=_sha(
                    row.get("market_snapshot_id"), field_name="market_snapshot_id"
                ),
                market_observed_at_utc=_text(
                    row.get("market_observed_at_utc"), field_name="market_observed_at_utc"
                ),
                local_fetched_at_utc=_text(
                    row.get("local_fetched_at_utc"), field_name="local_fetched_at_utc"
                ),
                source_match_id=_text(row.get("source_match_id"), field_name="source_match_id"),
                home_decimal_odds=_decimal(
                    row.get("home_decimal_price"), field_name="home_decimal_price"
                ),
                away_decimal_odds=_decimal(
                    row.get("away_decimal_price"), field_name="away_decimal_price"
                ),
            )
        )
    if len(market_rows) != P40A_EXPECTED_P39_EDGE_READY_COUNT:
        raise ValueError("P40A P39A edge-ready rows could not be reconstructed")
    if len({row.p37_prediction_row_id for row in market_rows}) != len(market_rows):
        raise ValueError("P40A P39A edge-ready prediction identities are not unique")
    if any(
        row.home_decimal_odds <= Decimal("1") or row.away_decimal_odds <= Decimal("1")
        for row in market_rows
    ):
        raise ValueError("P40A P39A contains a malformed decimal price")
    return tuple(
        sorted(
            market_rows,
            key=lambda row: (
                row.scheduled_start_utc,
                row.game_number,
                row.provider_game_id,
            ),
        )
    )


def _validate_p37_authority(root: Path) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, str]
]:
    report = root / P37A_REPORT_RELATIVE_PATH
    summary_path = report / "summary.json"
    comparisons_path = report / "comparisons.jsonl"
    summary = _read_json(summary_path)
    aggregate = summary.get("aggregate")
    authority = summary.get("authority")
    if not isinstance(aggregate, Mapping) or not isinstance(authority, Mapping):
        raise ValueError("P40A P37A summary authority is incomplete")
    if aggregate.get("raw_row_count") != 75 or aggregate.get("evaluable_row_count") != 65 or aggregate.get("excluded_row_count") != 10:
        raise ValueError("P40A P37A aggregate row authority drift")
    if summary.get("admitted_evaluation_fold_ids") != ["wf_004", "wf_005", "wf_006"]:
        raise ValueError("P40A P37A fold authority drift")
    champion_model_id = authority.get("current_champion_model_id")
    champion_fingerprint = authority.get("current_champion_artifact_fingerprint")
    _text(champion_model_id, field_name="current_champion_model_id")
    _sha(champion_fingerprint, field_name="current_champion_artifact_fingerprint")
    rows = _read_jsonl(comparisons_path)
    comparisons_sha256 = _sha256_path(comparisons_path)
    if len(rows) != P40A_EXPECTED_P37_TARGET_COUNT:
        raise ValueError("P40A P37A comparisons must contain 65 rows")
    if comparisons_sha256 != P40A_EXPECTED_P37_COMPARISONS_SHA256:
        raise ValueError("P40A P37A comparisons SHA-256 authority drift")
    fold_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    seen_games: set[str] = set()
    for row in rows:
        fold = _text(row.get("fold_id"), field_name="fold_id")
        fold_counts[fold] = fold_counts.get(fold, 0) + 1
        row_id = _sha(row.get("comparison_row_id"), field_name="comparison_row_id")
        if row_id in seen_ids:
            raise ValueError("P40A P37A comparison row identities are not unique")
        seen_ids.add(row_id)
        provider_game_id = _text(row.get("provider_game_id"), field_name="provider_game_id")
        if provider_game_id in seen_games:
            raise ValueError("P40A P37A provider game identities are not unique")
        seen_games.add(provider_game_id)
        if row.get("true_oos_verified") is not True:
            raise ValueError("P40A requires every P37A decision row to be true-OOS verified")
        if row.get("incumbent_model_id") != champion_model_id or row.get("incumbent_model_fingerprint") != champion_fingerprint:
            raise ValueError("P40A Champion authority drift in P37A comparisons")
        _text(row.get("challenger_model_id"), field_name="challenger_model_id")
        _sha(row.get("challenger_model_fingerprint"), field_name="challenger_model_fingerprint")
        for field_name in ("incumbent_home_probability", "challenger_home_probability"):
            probability = _decimal(row.get(field_name), field_name=field_name)
            if not Decimal("0") < probability < Decimal("1"):
                raise ValueError(f"P40A {field_name} is outside the probability domain")
        if row.get("actual_winner") not in ("HOME", "AWAY"):
            raise ValueError("P40A final outcome authority is not HOME/AWAY")
        if row.get("target_home_win") not in (0, 1):
            raise ValueError("P40A target_home_win authority is invalid")
        expected_target = 1 if row["actual_winner"] == "HOME" else 0
        if row["target_home_win"] != expected_target:
            raise ValueError("P40A P37A outcome and target disagree")
    if fold_counts != {"wf_004": 23, "wf_005": 17, "wf_006": 25}:
        raise ValueError(f"P40A P37A fold row counts drifted: {fold_counts}")
    hashes = {
        "p37_summary_sha256": _sha256_path(summary_path),
        "p37_comparisons_sha256": comparisons_sha256,
    }
    return summary, rows, hashes


def _load_p37_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[P40APredictionRow, ...], tuple[P40AOutcomeRow, ...]]:
    predictions: list[P40APredictionRow] = []
    outcomes: list[P40AOutcomeRow] = []
    for row in rows:
        predictions.append(
            P40APredictionRow(
                p37_fold_id=_text(row.get("fold_id"), field_name="fold_id"),
                p37_window=_text(
                    row.get("evaluation_window_id"), field_name="evaluation_window_id"
                ),
                p37_prediction_row_id=_sha(
                    row.get("comparison_row_id"), field_name="comparison_row_id"
                ),
                provider_namespace=_text(
                    row.get("provider_namespace"), field_name="provider_namespace"
                ),
                provider_game_id=_text(row.get("provider_game_id"), field_name="provider_game_id"),
                game_pk=_positive_int(row.get("game_pk"), field_name="game_pk"),
                game_number=_positive_int(row.get("game_number"), field_name="game_number"),
                scheduled_start_utc=_text(
                    row.get("scheduled_start_utc"), field_name="scheduled_start_utc"
                ),
                champion_model_id=_text(
                    row.get("incumbent_model_id"), field_name="incumbent_model_id"
                ),
                champion_model_fingerprint=_sha(
                    row.get("incumbent_model_fingerprint"),
                    field_name="incumbent_model_fingerprint",
                ),
                champion_home_probability=_decimal(
                    row.get("incumbent_home_probability"),
                    field_name="incumbent_home_probability",
                ),
                challenger_model_id=_text(
                    row.get("challenger_model_id"), field_name="challenger_model_id"
                ),
                challenger_model_fingerprint=_sha(
                    row.get("challenger_model_fingerprint"),
                    field_name="challenger_model_fingerprint",
                ),
                challenger_home_probability=_decimal(
                    row.get("challenger_home_probability"),
                    field_name="challenger_home_probability",
                ),
            )
        )
        outcomes.append(
            P40AOutcomeRow(
                p37_prediction_row_id=_sha(
                    row.get("comparison_row_id"), field_name="comparison_row_id"
                ),
                provider_game_id=_text(row.get("provider_game_id"), field_name="provider_game_id"),
                actual_winner=_text(row.get("actual_winner"), field_name="actual_winner"),
                target_home_win=row["target_home_win"],
            )
        )
    return (
        tuple(sorted(predictions, key=lambda row: row.p37_prediction_row_id)),
        tuple(sorted(outcomes, key=lambda row: row.p37_prediction_row_id)),
    )


def load_p40a_authority(repository_root: str | Path) -> P40AAuthority:
    """Load and validate the immutable P37/P38/P39 inputs for P40A."""

    root = Path(repository_root).resolve()
    p39_summary, p39_source, p39_raw_rows, p39_hashes = _validate_p39_authority(root)
    p37_summary, p37_raw_rows, p37_hashes = _validate_p37_authority(root)
    market_rows = _load_p39_market_rows(p39_raw_rows)
    prediction_rows, outcome_rows = _load_p37_rows(p37_raw_rows)
    prediction_by_id = {row.p37_prediction_row_id: row for row in prediction_rows}
    for market in market_rows:
        prediction = prediction_by_id.get(market.p37_prediction_row_id)
        if prediction is None:
            raise ValueError(
                "P40A P39A market row has no P37 prediction authority: "
                f"{market.p37_prediction_row_id}"
            )
        for field_name in (
            "p37_fold_id",
            "p37_window",
            "provider_namespace",
            "provider_game_id",
            "game_pk",
            "game_number",
            "scheduled_start_utc",
        ):
            if getattr(market, field_name) != getattr(prediction, field_name):
                raise ValueError(f"P40A P37/P39 identity mismatch in {field_name}")
        p39_prediction_probability = next(
            row["prediction_probability"]
            for row in p39_raw_rows
            if row.get("p37_prediction_row_id") == market.p37_prediction_row_id
        )
        if _decimal(p39_prediction_probability, field_name="prediction_probability") != prediction.challenger_home_probability:
            raise ValueError("P40A P39 raw challenger probability lineage drift")
    p38_report = root / P38A_REPORT_RELATIVE_PATH
    p38_paths = (
        p38_report / "summary.json",
        p38_report / "comparisons.jsonl",
    )
    p38_hashes = {
        "p38_summary_sha256": _sha256_path(p38_paths[0]),
        "p38_comparisons_sha256": _sha256_path(p38_paths[1]),
    }
    source_manifest = {
        "schema_version": "p40a.moneyline_paper_bet_pass_source_manifest.v1",
        "task_id": P40A_TASK_ID,
        "p39a": {
            "report_path": str(P39A_REPORT_RELATIVE_PATH),
            "target_count": p39_summary["p37_evaluable_target_count"],
            "edge_ready_count": p39_summary["edge_ready_count"],
            "no_market_count": p39_summary["no_market_rows"],
            "market_join_fingerprint": p39_summary["market_join_fingerprint"],
            "selected_snapshot_fingerprint": p39_summary["selected_snapshot_fingerprint"],
            "legacy_source_sha256": p39_source["source_sha256"],
            **p39_hashes,
        },
        "p37a": {
            "report_path": str(P37A_REPORT_RELATIVE_PATH),
            "evaluable_count": p37_summary["aggregate"]["evaluable_row_count"],
            "folds": list(p37_summary["admitted_evaluation_fold_ids"]),
            "champion_model_id": p37_summary["authority"]["current_champion_model_id"],
            "champion_model_fingerprint": p37_summary["authority"]["current_champion_artifact_fingerprint"],
            **p37_hashes,
        },
        "p38a": {
            "report_path": str(P38A_REPORT_RELATIVE_PATH),
            "used_for_decisions": False,
            **p38_hashes,
        },
        "decision_rule": {
            "policy_id": P40A_POLICY_ID,
            "p_home": "model pregame home-win probability",
            "p_away": "1 - p_home",
            "ev_home": "p_home * home_decimal_odds - 1",
            "ev_away": "p_away * away_decimal_odds - 1",
            "candidate_side": "the side with the larger expected value",
            "decision": "BET iff max(EV_home, EV_away) > 0; otherwise PASS",
            "additional_threshold": "NONE",
            "odds_source": "actual offered P39A decimal price",
        },
        "settlement_rule": {
            "paper_stake_convention": "1.0 PAPER UNIT for every BET; PASS risks zero",
            "winning_net_units": "decimal_odds - 1",
            "losing_net_units": "-1",
            "pass_net_units": "0",
            "push_or_cancelled_policy": "NOT APPLICABLE: existing final-result authority is HOME/AWAY-only and rejects tied finals",
        },
        "outcome_authority": P40A_OUTCOME_AUTHORITY,
        "claims": {
            "real_betting": False,
            "threshold_optimization": False,
            "staking_optimization": False,
            "kelly": False,
            "bankroll_management": False,
            "model_promotion": False,
            "calibration": False,
            "training": False,
            "external_market_acquisition": False,
        },
    }
    return P40AAuthority(
        repository_root=root,
        p39_summary=p39_summary,
        p39_source_manifest=p39_source,
        p37_summary=p37_summary,
        market_rows=market_rows,
        prediction_rows=prediction_rows,
        outcome_rows=outcome_rows,
        source_manifest=source_manifest,
    )


def build_p40a_decisions(
    authority: P40AAuthority,
    *,
    market_rows: Sequence[P40AMarketRow] | None = None,
    prediction_rows: Sequence[P40APredictionRow] | None = None,
) -> tuple[PaperMoneylineDecision, ...]:
    """Freeze both model policies from pregame inputs only."""

    markets = tuple(market_rows or authority.market_rows)
    predictions = tuple(prediction_rows or authority.prediction_rows)
    prediction_by_id = {row.p37_prediction_row_id: row for row in predictions}
    p37_comparisons_sha256 = authority.source_manifest["p37a"]["p37_comparisons_sha256"]
    market_source_sha256 = authority.source_manifest["p39a"]["legacy_source_sha256"]
    decisions: list[PaperMoneylineDecision] = []
    for market in sorted(
        markets,
        key=lambda row: (row.scheduled_start_utc, row.game_number, row.provider_game_id),
    ):
        prediction = prediction_by_id.get(market.p37_prediction_row_id)
        if prediction is None:
            raise ValueError("P40A decision input is missing a P37 probability row")
        common = {
            "p37_fold_id": market.p37_fold_id,
            "p37_window": market.p37_window,
            "p37_prediction_row_id": market.p37_prediction_row_id,
            "provider_namespace": market.provider_namespace,
            "provider_game_id": market.provider_game_id,
            "game_pk": market.game_pk,
            "game_number": market.game_number,
            "official_date": market.official_date,
            "scheduled_start_utc": market.scheduled_start_utc,
            "home_team": market.home_team,
            "away_team": market.away_team,
            "home_team_code": market.home_team_code,
            "away_team_code": market.away_team_code,
            "market_snapshot_id": market.market_snapshot_id,
            "market_observed_at_utc": market.market_observed_at_utc,
            "local_fetched_at_utc": market.local_fetched_at_utc,
            "source_match_id": market.source_match_id,
            "market_source_sha256": market_source_sha256,
            "p37_comparisons_sha256": p37_comparisons_sha256,
            "home_decimal_odds": market.home_decimal_odds,
            "away_decimal_odds": market.away_decimal_odds,
        }
        decisions.append(
            PaperMoneylineDecision.create(
                **common,
                model_role=P40A_CHAMPION_ROLE,
                model_id=prediction.champion_model_id,
                model_fingerprint=prediction.champion_model_fingerprint,
                model_probability_source="P37A_INCUMBENT_HOME_PROBABILITY",
                p_home=prediction.champion_home_probability,
            )
        )
        decisions.append(
            PaperMoneylineDecision.create(
                **common,
                model_role=P40A_SHADOW_ROLE,
                model_id=prediction.challenger_model_id,
                model_fingerprint=prediction.challenger_model_fingerprint,
                model_probability_source="P37A_CHALLENGER_HOME_PROBABILITY",
                p_home=prediction.challenger_home_probability,
            )
        )
    if len(decisions) != len(markets) * 2:
        raise ValueError("P40A decision generation dropped a model/market row")
    return tuple(
        sorted(
            decisions,
            key=lambda row: (
                row.model_role,
                row.scheduled_start_utc,
                row.game_number,
                row.provider_game_id,
            ),
        )
    )


def settle_p40a_decisions(
    authority: P40AAuthority,
    decisions: Sequence[PaperMoneylineDecision],
    *,
    outcome_rows: Sequence[P40AOutcomeRow] | None = None,
) -> tuple[PaperMoneylineSettlement, ...]:
    """Attach downstream outcomes after the decision sequence is frozen."""

    outcomes = tuple(outcome_rows or authority.outcome_rows)
    outcome_by_id = {row.p37_prediction_row_id: row for row in outcomes}
    settlements: list[PaperMoneylineSettlement] = []
    for decision in decisions:
        outcome = outcome_by_id.get(decision.p37_prediction_row_id)
        if outcome is None or outcome.provider_game_id != decision.provider_game_id:
            raise RuntimeError("P40A_OUTCOME_AUTHORITY_UNRESOLVED_STOP")
        settlements.append(
            settle_paper_moneyline_decision(
                decision,
                final_game_outcome=outcome.actual_winner,
                target_home_win=outcome.target_home_win,
                outcome_authority_row_id=outcome.p37_prediction_row_id,
                outcome_authority=P40A_OUTCOME_AUTHORITY,
            )
        )
    return tuple(settlements)


def _aggregate_by_role(
    settlements: Sequence[PaperMoneylineSettlement],
    *,
    edge_ready_rows: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role in (P40A_CHAMPION_ROLE, P40A_SHADOW_ROLE):
        result[role] = aggregate_paper_settlements(
            (row for row in settlements if row.decision.model_role == role),
            edge_ready_rows=edge_ready_rows,
            model_role=role,
        )
    return result


def _conclusion(net_units: str) -> str:
    net = Decimal(net_units)
    if net > Decimal("0"):
        return "PAPER_BASELINE_OBSERVED_POSITIVE"
    if net < Decimal("0"):
        return "PAPER_BASELINE_OBSERVED_NEGATIVE"
    return "PAPER_BASELINE_OBSERVED_FLAT"


def _shadow_comparison(champion_net: str, shadow_net: str) -> str:
    champion = Decimal(champion_net)
    shadow = Decimal(shadow_net)
    if shadow > champion:
        return "SHADOW_CHALLENGER_HIGHER_NET_UNITS"
    if champion > shadow:
        return "CHAMPION_HIGHER_NET_UNITS"
    return "SAME_NET_UNITS"


def _build_summary(
    authority: P40AAuthority,
    settlements: Sequence[PaperMoneylineSettlement],
    *,
    deterministic_rerun_verified: bool,
) -> dict[str, Any]:
    edge_ready_rows = len(authority.market_rows)
    aggregate = _aggregate_by_role(settlements, edge_ready_rows=edge_ready_rows)
    windows = sorted({row.p37_window for row in authority.market_rows})
    market_by_window = {
        window: sum(row.p37_window == window for row in authority.market_rows)
        for window in windows
    }
    per_window: dict[str, dict[str, dict[str, Any]]] = {}
    for window in windows:
        window_settlements = tuple(
            row for row in settlements if row.decision.p37_window == window
        )
        per_window[window] = _aggregate_by_role(
            window_settlements,
            edge_ready_rows=market_by_window[window],
        )
    champion = aggregate[P40A_CHAMPION_ROLE]
    shadow = aggregate[P40A_SHADOW_ROLE]
    return {
        "schema_version": "p40a.moneyline_paper_bet_pass_summary.v1",
        "task_id": P40A_TASK_ID,
        "p39_target_count": authority.p39_summary["p37_evaluable_target_count"],
        "p39_edge_ready_count": authority.p39_summary["edge_ready_count"],
        "p39_no_market_count": authority.p39_summary["no_market_rows"],
        "edge_ready_rows": edge_ready_rows,
        "models": {
            "champion_primary": champion,
            "raw_challenger_shadow": shadow,
        },
        "per_window": per_window,
        "per_window_edge_ready_counts": market_by_window,
        "primary_conclusion": _conclusion(champion["net_paper_units"]),
        "shadow_comparison": _shadow_comparison(
            champion["net_paper_units"], shadow["net_paper_units"]
        ),
        "decision_rule": authority.source_manifest["decision_rule"],
        "settlement_rule": authority.source_manifest["settlement_rule"],
        "roi_label": "DESCRIPTIVE_PAPER_ONLY",
        "descriptive_only": True,
        "deterministic_rerun_verified": deterministic_rerun_verified,
        "outcome_isolation_verified": True,
        "p37_p38_p39_inputs_read_only": True,
        "claims": {
            "real_betting": False,
            "profitability_claim": False,
            "expected_future_roi_claim": False,
            "threshold_optimization": False,
            "staking_optimization": False,
            "kelly": False,
            "bankroll_management": False,
            "model_promotion": False,
            "calibration": False,
            "training": False,
            "external_market_acquisition": False,
        },
    }


def _build_once(
    authority: P40AAuthority,
) -> tuple[tuple[PaperMoneylineDecision, ...], tuple[PaperMoneylineSettlement, ...]]:
    # This is the explicit phase boundary: no outcome row is passed to the
    # decision builder.  Outcomes are looked up only after decisions exist.
    decisions = build_p40a_decisions(authority)
    settlements = settle_p40a_decisions(authority, decisions)
    return decisions, settlements


def run_p40a_moneyline_paper_bet_pass(repository_root: str | Path) -> P40AResult:
    """Run P40A twice in memory and return the deterministic final artifact."""

    authority = load_p40a_authority(repository_root)
    first_decisions, first_settlements = _build_once(authority)
    second_decisions, second_settlements = _build_once(authority)
    if tuple(row.to_projection() for row in first_decisions) != tuple(
        row.to_projection() for row in second_decisions
    ):
        raise RuntimeError("P40A deterministic decision rerun differed")
    if tuple(row.to_projection() for row in first_settlements) != tuple(
        row.to_projection() for row in second_settlements
    ):
        raise RuntimeError("P40A deterministic settlement rerun differed")
    summary = _build_summary(
        authority,
        first_settlements,
        deterministic_rerun_verified=True,
    )
    return P40AResult(
        authority=authority,
        decisions=first_decisions,
        settlements=first_settlements,
        summary=summary,
    )


def render_p40a_report(result: P40AResult) -> str:
    summary = result.summary
    champion = summary["models"]["champion_primary"]
    shadow = summary["models"]["raw_challenger_shadow"]
    lines = [
        "# P40A Moneyline Paper BET/PASS Baseline",
        "",
        "This is a deterministic historical true-OOS paper-only measurement. It is not a real betting recommendation, profitability claim, staking strategy, or model-promotion decision.",
        "",
        "## Frozen decision rule",
        "",
        "- `p_home = model pregame home-win probability`; `p_away = 1 - p_home`.",
        "- `EV_home = p_home * home_decimal_odds - 1`; `EV_away = p_away * away_decimal_odds - 1`.",
        "- BET the side with the larger EV iff `max(EV_home, EV_away) > 0`; otherwise PASS.",
        "- No additional edge threshold, minimum odds filter, confidence cutoff, or tuning was applied.",
        "",
        "## Authority and coverage",
        "",
        f"- P39 target universe: `{summary['p39_target_count']}`; edge-ready rows: `{summary['edge_ready_rows']}`; no-market rows: `{summary['p39_no_market_count']}`.",
        f"- Decisions: `{len(result.decisions)}` (`62` Champion primary plus `62` raw-challenger shadow).",
        f"- Deterministic rerun: `{summary['deterministic_rerun_verified']}`.",
        "- P37, P38, and P39 authorities are read-only inputs; P38 calibration is out of scope.",
        "",
        "## Aggregate paper results",
        "",
        "| Policy | BET | PASS | Wins | Losses | Pushes | Hit rate | Units risked | Net units | DESCRIPTIVE_PAPER_ONLY ROI | Max drawdown | Avg predicted EV of BET rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _aggregate_markdown_row("Champion primary", champion),
        _aggregate_markdown_row("Raw challenger shadow", shadow),
        "",
        f"- Primary conclusion: `{summary['primary_conclusion']}` (realized net paper units only; sample size `N={summary['edge_ready_rows']}`).",
        f"- Shadow comparison: `{summary['shadow_comparison']}`; this does not authorize selection or promotion.",
        "",
        "## Per-window paper results",
        "",
        "| Window | Policy | Edge-ready | BET | PASS | Wins | Losses | Net units | DESCRIPTIVE_PAPER_ONLY ROI | Max drawdown |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window in sorted(summary["per_window"]):
        for role, label in (
            (P40A_CHAMPION_ROLE, "Champion primary"),
            (P40A_SHADOW_ROLE, "Raw challenger shadow"),
        ):
            lines.append(_window_markdown_row(window, label, summary["per_window"][window][role]))
    lines.extend(
        [
            "",
            "## Safety boundary",
            "",
            "- `DESCRIPTIVE_PAPER_ONLY` means realized net paper units divided by paper units risked over this historical true-OOS sample; it is not expected future ROI or proven profitability.",
            "- Paper stake convention: `1.0 PAPER UNIT` per BET; PASS risks zero. No Kelly, bankroll management, or variable stake sizing was used.",
            "- Real betting: `NOT RUN`.",
            "- Threshold optimization: `NOT RUN`.",
            "- Staking/Kelly optimization: `NOT RUN`.",
            "- Model promotion: `NOT RUN`.",
            "- External market acquisition: `NOT RUN`.",
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate_markdown_row(label: str, aggregate: Mapping[str, Any]) -> str:
    return "| " + " | ".join(
        (
            label,
            str(aggregate["bet_count"]),
            str(aggregate["pass_count"]),
            str(aggregate["win_count"]),
            str(aggregate["loss_count"]),
            str(aggregate["push_count"]),
            str(aggregate["observed_hit_rate"]),
            str(aggregate["total_paper_units_risked"]),
            str(aggregate["net_paper_units"]),
            str(aggregate["descriptive_paper_roi"]),
            str(aggregate["maximum_paper_drawdown"]),
            str(aggregate["average_predicted_ev_of_bet_rows"]),
        )
    ) + " |"


def _window_markdown_row(window: str, label: str, aggregate: Mapping[str, Any]) -> str:
    return "| " + " | ".join(
        (
            window,
            label,
            str(aggregate["edge_ready_rows"]),
            str(aggregate["bet_count"]),
            str(aggregate["pass_count"]),
            str(aggregate["win_count"]),
            str(aggregate["loss_count"]),
            str(aggregate["net_paper_units"]),
            str(aggregate["descriptive_paper_roi"]),
            str(aggregate["maximum_paper_drawdown"]),
        )
    ) + " |"


def render_p40a_artifacts(result: P40AResult) -> dict[str, bytes]:
    decisions = [row.to_projection() for row in result.decisions]
    settlements = [row.to_projection() for row in result.settlements]
    return {
        "source_manifest.json": _json_bytes(result.authority.source_manifest),
        "decisions.jsonl": _jsonl_bytes(decisions),
        "settlements.jsonl": _jsonl_bytes(settlements),
        "summary.json": _json_bytes(result.summary),
        "report.md": render_p40a_report(result).encode("utf-8"),
    }


def write_p40a_artifacts(output_dir: str | Path, result: P40AResult) -> dict[str, str]:
    """Write only the allowlisted repository-native P40A artifacts."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rendered = render_p40a_artifacts(result)
    hashes: dict[str, str] = {}
    for name, content in rendered.items():
        path = directory / name
        path.write_bytes(content)
        hashes[name] = _sha256_bytes(content)
    return hashes


__all__ = (
    "P40A_ARTIFACT_FILES",
    "P40A_CHAMPION_ROLE",
    "P40A_REPORT_RELATIVE_PATH",
    "P40A_SHADOW_ROLE",
    "P40AAuthority",
    "P40AMarketRow",
    "P40AOutcomeRow",
    "P40APredictionRow",
    "P40AResult",
    "build_p40a_decisions",
    "load_p40a_authority",
    "render_p40a_artifacts",
    "render_p40a_report",
    "run_p40a_moneyline_paper_bet_pass",
    "settle_p40a_decisions",
    "write_p40a_artifacts",
)
