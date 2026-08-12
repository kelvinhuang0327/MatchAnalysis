"""Join the authoritative P37 true-OOS predictions to TSL Moneyline prices."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.moneyline_market_snapshot import (
    MoneylineMarketObservationCandidate,
    MoneylineMarketSnapshot,
    P39A_SELECTION_RULE,
    P39A_TIMESTAMP_SEMANTICS,
    canonical_json_bytes,
)
from ...core.identity import MatchIdentity
from .generate_tsl_moneyline_edge_batch import TSL_TO_MLB_TEAM_ABBREVIATION


P39A_TASK_ID = "P39A"
P39A_JOIN_SCHEMA_VERSION = "p39a.p37_oos_moneyline_market_join.v1"
P39A_JOIN_ROW_SCHEMA_VERSION = "p39a.p37_oos_moneyline_market_join_row.v1"
P39A_REPORT_RELATIVE_PATH = Path("report/p39a_tsl_moneyline_market_join")
P39A_SELECTION_RULE_TEXT = P39A_SELECTION_RULE
P39A_TIMESTAMP_SEMANTICS_TEXT = P39A_TIMESTAMP_SEMANTICS
P39A_CONCLUSION_RULE = (
    "MARKET_JOIN_READY iff every P37 evaluable target has one usable pregame snapshot; "
    "MARKET_JOIN_PARTIAL iff at least one but fewer than all targets are usable; "
    "MARKET_AUTHORITY_INSUFFICIENT iff zero targets are usable or timestamp/source "
    "authority is not trustworthy."
)
P37_COMPARISONS_RELATIVE_PATH = Path(
    "report/p37a_rolling_walk_forward_oos/comparisons.jsonl"
)
P37_SUMMARY_RELATIVE_PATH = Path("report/p37a_rolling_walk_forward_oos/summary.json")
P37_FOLD_INPUTS: dict[str, tuple[Path, Path]] = {
    "wf_004": (
        Path("report/p23f2_official_future_fold/feature_rows.jsonl"),
        Path("data/fixtures/p23f2_official_2026_history/normalized/schedule.jsonl"),
    ),
    "wf_005": (
        Path("data/fixtures/p23b_future_folds/wf_005/feature_rows.jsonl"),
        Path("data/fixtures/p23b_future_folds/wf_005/normalized/schedule.jsonl"),
    ),
    "wf_006": (
        Path("data/fixtures/p23b_future_folds/wf_006/feature_rows.jsonl"),
        Path("data/fixtures/p23b_future_folds/wf_006/normalized/schedule.jsonl"),
    ),
}
P37_EXPECTED_FOLD_COUNTS = {"wf_004": 23, "wf_005": 17, "wf_006": 25}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path} contains a blank row at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a trimmed non-empty string")
    return value


def _int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _probability(value: object) -> Decimal:
    try:
        probability = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("challenger_home_probability must be decimal") from exc
    if not probability.is_finite() or not Decimal("0") < probability < Decimal("1"):
        raise ValueError("challenger_home_probability must be strictly between zero and one")
    return probability


@dataclass(frozen=True, slots=True)
class P37OOSPrediction:
    """Outcome-blind identity and raw challenger probability from P37."""

    fold_id: str
    evaluation_window_id: str
    comparison_row_id: str
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
    challenger_home_probability: Decimal

    @property
    def predicted_side(self) -> str:
        return "HOME" if self.challenger_home_probability >= Decimal("0.5") else "AWAY"

    @property
    def identity(self) -> MatchIdentity:
        return MatchIdentity(
            sport="BASEBALL",
            league="MLB",
            season=2026,
            canonical_game_id=self.provider_game_id,
            home_participant=self.home_team,
            away_participant=self.away_team,
            game_discriminator=str(self.game_number),
        )

    def key(self) -> tuple[str, str, str, str, int]:
        return (
            self.home_team_code,
            self.away_team_code,
            self.scheduled_start_utc,
            self.provider_game_id,
            self.game_number,
        )


@dataclass(frozen=True, slots=True)
class P37MarketJoinRow:
    """One target row with a usable snapshot or an explicit rejection."""

    prediction: P37OOSPrediction
    market_snapshot_status: str
    rejection_reason: str | None
    market_snapshot_id: str | None
    market_observed_at_utc: str | None
    local_fetched_at_utc: str | None
    provider_observed_at_utc: str | None
    source_match_id: str | None
    home_decimal_price: Decimal | None
    away_decimal_price: Decimal | None
    home_implied_probability: Decimal | None
    away_implied_probability: Decimal | None
    raw_model_vs_market_home_probability_delta: Decimal | None

    def to_projection(self) -> dict[str, Any]:
        def decimal(value: Decimal | None) -> str | None:
            return None if value is None else format(value, "f")

        return {
            "schema_version": P39A_JOIN_ROW_SCHEMA_VERSION,
            "p37_window": self.prediction.evaluation_window_id,
            "p37_fold_id": self.prediction.fold_id,
            "p37_prediction_row_id": self.prediction.comparison_row_id,
            "provider_namespace": self.prediction.provider_namespace,
            "provider_game_id": self.prediction.provider_game_id,
            "game_pk": self.prediction.game_pk,
            "game_number": self.prediction.game_number,
            "official_date": self.prediction.official_date,
            "scheduled_start_utc": self.prediction.scheduled_start_utc,
            "home_team": self.prediction.home_team,
            "away_team": self.prediction.away_team,
            "home_team_code": self.prediction.home_team_code,
            "away_team_code": self.prediction.away_team_code,
            "prediction_probability": format(
                self.prediction.challenger_home_probability,
                "f",
            ),
            "predicted_side": self.prediction.predicted_side,
            "market_snapshot_id": self.market_snapshot_id,
            "market_observed_at_utc": self.market_observed_at_utc,
            "local_fetched_at_utc": self.local_fetched_at_utc,
            "provider_observed_at_utc": self.provider_observed_at_utc,
            "source_match_id": self.source_match_id,
            "home_decimal_price": decimal(self.home_decimal_price),
            "away_decimal_price": decimal(self.away_decimal_price),
            "home_implied_probability": decimal(self.home_implied_probability),
            "away_implied_probability": decimal(self.away_implied_probability),
            "raw_model_vs_market_home_probability_delta": decimal(
                self.raw_model_vs_market_home_probability_delta
            ),
            "market_snapshot_status": self.market_snapshot_status,
            "observation_timestamp_semantics": P39A_TIMESTAMP_SEMANTICS_TEXT,
            "selected_snapshot_rule": P39A_SELECTION_RULE_TEXT,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True, slots=True)
class P39AMarketJoinResult:
    """Deterministic P39A rows, selected snapshots, and report projections."""

    predictions: tuple[P37OOSPrediction, ...]
    join_rows: tuple[P37MarketJoinRow, ...]
    selected_snapshots: tuple[MoneylineMarketSnapshot, ...]
    source_manifest: dict[str, Any]
    p37_manifest: dict[str, Any]
    summary: dict[str, Any]

    def comparable_projection(self) -> dict[str, Any]:
        summary = dict(self.summary)
        summary.pop("deterministic_rerun_verified", None)
        return {
            "predictions": [
                {
                    "fold_id": prediction.fold_id,
                    "window": prediction.evaluation_window_id,
                    "provider_game_id": prediction.provider_game_id,
                    "game_number": prediction.game_number,
                    "scheduled_start_utc": prediction.scheduled_start_utc,
                    "probability": format(
                        prediction.challenger_home_probability,
                        "f",
                    ),
                }
                for prediction in self.predictions
            ],
            "join_rows": [row.to_projection() for row in self.join_rows],
            "selected_snapshots": [
                snapshot.to_projection() for snapshot in self.selected_snapshots
            ],
            "source_manifest": self.source_manifest,
            "p37_manifest": self.p37_manifest,
            "summary": summary,
        }


def load_p37_predictions(repository_root: str | Path) -> tuple[
    tuple[P37OOSPrediction, ...], dict[str, Any]
]:
    """Load exactly the 65 P37 evaluable rows without reading outcomes."""

    repository = Path(repository_root)
    summary_path = repository / P37_SUMMARY_RELATIVE_PATH
    comparisons_path = repository / P37_COMPARISONS_RELATIVE_PATH
    summary = _read_json(summary_path)
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, Mapping) or {
        aggregate.get("raw_row_count"),
        aggregate.get("evaluable_row_count"),
        aggregate.get("excluded_row_count"),
    } != {75, 65, 10}:
        raise ValueError("P37 aggregate authority must remain 75 raw, 65 evaluable, 10 excluded")
    if summary.get("admitted_evaluation_fold_ids") != ["wf_004", "wf_005", "wf_006"]:
        raise ValueError("P37 evaluation fold authority drift")

    comparison_rows = _read_jsonl(comparisons_path)
    if len(comparison_rows) != 65:
        raise ValueError("P37 comparisons must contain exactly 65 evaluable rows")
    feature_by_fold: dict[str, dict[str, dict[str, Any]]] = {}
    schedule_by_fold: dict[str, dict[str, dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    for fold_id, (feature_relative, schedule_relative) in P37_FOLD_INPUTS.items():
        feature_path = repository / feature_relative
        schedule_path = repository / schedule_relative
        features = _read_jsonl(feature_path)
        schedules = _read_jsonl(schedule_path)
        feature_by_fold[fold_id] = {
            str(row["provider_game_id"]): row for row in features
        }
        schedule_by_fold[fold_id] = {
            str(row["provider_game_id"]): row for row in schedules
        }
        input_hashes[str(feature_relative)] = _sha256_path(feature_path)
        input_hashes[str(schedule_relative)] = _sha256_path(schedule_path)

    predictions: list[P37OOSPrediction] = []
    seen_ids: set[str] = set()
    for comparison in comparison_rows:
        fold_id = _text(comparison.get("fold_id"), field_name="fold_id")
        if fold_id not in P37_FOLD_INPUTS:
            raise ValueError(f"P37 row uses an unadmitted fold: {fold_id}")
        provider_game_id = _text(
            comparison.get("provider_game_id"),
            field_name="provider_game_id",
        )
        if provider_game_id in seen_ids:
            raise ValueError("P37 provider game identities must be unique")
        seen_ids.add(provider_game_id)
        feature = feature_by_fold[fold_id].get(provider_game_id)
        schedule = schedule_by_fold[fold_id].get(provider_game_id)
        if feature is None or schedule is None:
            raise ValueError(f"P37 target identity is missing from feature/schedule authority: {provider_game_id}")
        forbidden_feature_fields = {
            "home_score",
            "away_score",
            "target_home_win",
            "actual_winner",
            "challenger_correct",
            "incumbent_correct",
        }
        if forbidden_feature_fields & feature.keys():
            raise ValueError(f"P37 feature row contains outcome fields: {provider_game_id}")
        feature_start = _text(feature.get("scheduled_start_utc"), field_name="scheduled_start_utc")
        schedule_start = _text(schedule.get("scheduled_start_utc"), field_name="scheduled_start_utc")
        if comparison.get("scheduled_start_utc") != feature_start or schedule_start != feature_start:
            raise ValueError(f"P37 schedule identity drift: {provider_game_id}")
        if _int(comparison.get("game_pk"), field_name="game_pk") != _int(feature.get("game_pk"), field_name="game_pk"):
            raise ValueError(f"P37 game_pk identity drift: {provider_game_id}")
        if _int(comparison.get("game_number"), field_name="game_number") != _int(feature.get("game_number"), field_name="game_number"):
            raise ValueError(f"P37 game_number identity drift: {provider_game_id}")
        home_team = _text(feature.get("home_team"), field_name="home_team")
        away_team = _text(feature.get("away_team"), field_name="away_team")
        schedule_home = schedule.get("home_team")
        schedule_away = schedule.get("away_team")
        if not isinstance(schedule_home, Mapping) or not isinstance(schedule_away, Mapping):
            raise ValueError(f"P37 schedule team identity is incomplete: {provider_game_id}")
        if schedule_home.get("name") != home_team or schedule_away.get("name") != away_team:
            raise ValueError(f"P37 feature/schedule team identity drift: {provider_game_id}")
        home_team_code = _text(schedule_home.get("abbreviation"), field_name="home_team_code")
        away_team_code = _text(schedule_away.get("abbreviation"), field_name="away_team_code")
        if home_team_code == away_team_code:
            raise ValueError(f"P37 home/away team codes must differ: {provider_game_id}")
        if comparison.get("provider_namespace") != "MLB_STATS_API":
            raise ValueError(f"P37 provider namespace drift: {provider_game_id}")
        predictions.append(
            P37OOSPrediction(
                fold_id=fold_id,
                evaluation_window_id=_text(
                    comparison.get("evaluation_window_id"),
                    field_name="evaluation_window_id",
                ),
                comparison_row_id=_text(
                    comparison.get("comparison_row_id"),
                    field_name="comparison_row_id",
                ),
                provider_namespace="MLB_STATS_API",
                provider_game_id=provider_game_id,
                game_pk=_int(comparison.get("game_pk"), field_name="game_pk"),
                game_number=_int(comparison.get("game_number"), field_name="game_number"),
                official_date=_text(feature.get("official_date"), field_name="official_date"),
                scheduled_start_utc=feature_start,
                home_team=home_team,
                away_team=away_team,
                home_team_code=home_team_code,
                away_team_code=away_team_code,
                challenger_home_probability=_probability(
                    comparison.get("challenger_home_probability")
                ),
            )
        )

    predictions.sort(key=lambda row: (row.scheduled_start_utc, row.game_number, row.game_pk))
    observed_fold_counts = Counter(row.fold_id for row in predictions)
    if dict(observed_fold_counts) != P37_EXPECTED_FOLD_COUNTS:
        raise ValueError(f"P37 fold target counts drift: {dict(observed_fold_counts)}")
    p37_manifest = {
        "summary_path": str(P37_SUMMARY_RELATIVE_PATH),
        "summary_sha256": _sha256_path(summary_path),
        "comparisons_path": str(P37_COMPARISONS_RELATIVE_PATH),
        "comparisons_sha256": _sha256_path(comparisons_path),
        "feature_and_schedule_sha256": dict(sorted(input_hashes.items())),
        "raw_row_count": 75,
        "evaluable_row_count": len(predictions),
        "excluded_row_count": 10,
        "fold_counts": dict(sorted(observed_fold_counts.items())),
        "authority_base_head": summary.get("authority", {}).get("base_head"),
        "authority_base_tree": summary.get("authority", {}).get("base_tree"),
    }
    return tuple(predictions), p37_manifest


def source_scope_keys(
    predictions: Sequence[P37OOSPrediction],
) -> set[tuple[str, str, str]]:
    """Convert official schedule codes to all existing TSL source-code aliases."""

    aliases_by_official: dict[str, set[str]] = defaultdict(set)
    for source_code, official_code in TSL_TO_MLB_TEAM_ABBREVIATION.items():
        aliases_by_official[str(official_code)].add(str(source_code))
    keys: set[tuple[str, str, str]] = set()
    for prediction in predictions:
        home_aliases = aliases_by_official.get(prediction.home_team_code, set())
        away_aliases = aliases_by_official.get(prediction.away_team_code, set())
        if not home_aliases or not away_aliases:
            raise ValueError(
                "P39A cannot normalize a P37 team code through the existing TSL crosswalk: "
                f"{prediction.home_team_code}/{prediction.away_team_code}"
            )
        keys.update(
            (home_alias, away_alias, prediction.scheduled_start_utc)
            for home_alias in home_aliases
            for away_alias in away_aliases
        )
    return keys


def _canonical_source_code(source_code: str) -> str | None:
    return TSL_TO_MLB_TEAM_ABBREVIATION.get(source_code)


def _rejection_reason(candidates: Sequence[MoneylineMarketObservationCandidate]) -> str:
    reasons = {candidate.rejection_reason for candidate in candidates}
    if "AMBIGUOUS_MONEYLINE_MARKET" in reasons:
        return "AMBIGUOUS_MONEYLINE_MARKET"
    if "POST_START" in reasons:
        return "POST_START"
    if "MISSING_OR_UNTRUSTED_TIMESTAMP" in reasons:
        return "MISSING_OR_UNTRUSTED_TIMESTAMP"
    if "MALFORMED_OR_INCOMPLETE_PRICE" in reasons:
        return "MALFORMED_OR_INCOMPLETE_PRICE"
    if "NOT_PREGAME" in reasons:
        return "NOT_PREGAME"
    return "NO_EXACT_MARKET_IDENTITY"


def _snapshot_for(
    prediction: P37OOSPrediction,
    candidate: MoneylineMarketObservationCandidate,
    *,
    source_manifest: Mapping[str, Any],
) -> MoneylineMarketSnapshot:
    if candidate.market_status != "VALID_PREGAME":
        raise ValueError("only a valid pregame candidate can become a snapshot")
    if candidate.market_observed_at_utc is None:
        raise ValueError("valid pregame candidate must have an observation timestamp")
    if candidate.home_decimal_price is None or candidate.away_decimal_price is None:
        raise ValueError("valid pregame candidate must have two prices")
    source_home_code = _canonical_source_code(candidate.source_home_code)
    source_away_code = _canonical_source_code(candidate.source_away_code)
    if source_home_code != prediction.home_team_code or source_away_code != prediction.away_team_code:
        raise ValueError("source and P37 team identity crosswalks do not agree")
    return MoneylineMarketSnapshot.create(
        identity=prediction.identity,
        provider_namespace=prediction.provider_namespace,
        provider_game_id=prediction.provider_game_id,
        game_number=prediction.game_number,
        scheduled_start_utc=prediction.scheduled_start_utc,
        home_team_code=prediction.home_team_code,
        away_team_code=prediction.away_team_code,
        source_home_team_name=candidate.source_home_team_name,
        source_away_team_name=candidate.source_away_team_name,
        source_home_code=candidate.source_home_code,
        source_away_code=candidate.source_away_code,
        source_repository=_text(
            source_manifest.get("source_repository"),
            field_name="source_repository",
        ),
        source_path=_text(source_manifest.get("source_path"), field_name="source_path"),
        source_sha256=_text(source_manifest.get("source_sha256"), field_name="source_sha256"),
        source_row_index=candidate.source_row_index,
        source_row_fingerprint=candidate.source_row_fingerprint,
        source_match_id=candidate.source_match_id,
        market_code="MNL",
        market_observed_at_utc=candidate.market_observed_at_utc,
        local_fetched_at_utc=candidate.local_fetched_at_utc or candidate.market_observed_at_utc,
        provider_observed_at_utc=candidate.provider_observed_at_utc,
        home_decimal_price=candidate.home_decimal_price,
        away_decimal_price=candidate.away_decimal_price,
    )


def build_p39a_market_join(
    predictions: Sequence[P37OOSPrediction],
    candidates: Sequence[MoneylineMarketObservationCandidate],
    *,
    source_manifest: Mapping[str, Any],
    p37_manifest: Mapping[str, Any],
) -> P39AMarketJoinResult:
    """Select one source snapshot per exact target and produce diagnostics only."""

    if not source_manifest.get("source_stable"):
        raise RuntimeError("P39A source stability is not confirmed")
    candidate_by_key: dict[tuple[str, str, str], list[MoneylineMarketObservationCandidate]] = defaultdict(list)
    for candidate in candidates:
        home_code = _canonical_source_code(candidate.source_home_code)
        away_code = _canonical_source_code(candidate.source_away_code)
        if home_code is None or away_code is None:
            continue
        candidate_by_key[(home_code, away_code, candidate.scheduled_start_utc)].append(candidate)

    rows: list[P37MarketJoinRow] = []
    snapshots: list[MoneylineMarketSnapshot] = []
    counts: Counter[str] = Counter()
    for prediction in sorted(predictions, key=lambda row: row.key()):
        matching = tuple(
            candidate_by_key.get(
                (
                    prediction.home_team_code,
                    prediction.away_team_code,
                    prediction.scheduled_start_utc,
                ),
                (),
            )
        )
        if matching:
            counts["exact_identity_match_count"] += 1
        source_match_ids = {candidate.source_match_id for candidate in matching}
        valid = tuple(
            candidate
            for candidate in matching
            if candidate.market_status == "VALID_PREGAME"
        )
        valid_source_match_ids = {candidate.source_match_id for candidate in valid}
        selected: MoneylineMarketSnapshot | None = None
        reason: str | None = None
        if len(valid_source_match_ids) > 1:
            reason = "AMBIGUOUS_SOURCE_GAME_IDENTITY"
            counts["ambiguous_rows"] += 1
        else:
            if valid:
                selected_candidate = max(
                    valid,
                    key=lambda candidate: (
                        candidate.market_observed_at_utc or "",
                        candidate.source_row_fingerprint,
                    ),
                )
                selected = _snapshot_for(
                    prediction,
                    selected_candidate,
                    source_manifest=source_manifest,
                )
                snapshots.append(selected)
                counts["usable_pregame_market_rows"] += 1
            else:
                reason = _rejection_reason(matching)
                reason_counter_key = {
                    "NO_EXACT_MARKET_IDENTITY": "no_market_rows",
                    "POST_START": "post_start_rejected_rows",
                    "MISSING_OR_UNTRUSTED_TIMESTAMP": "missing_or_untrusted_timestamp_rows",
                    "MALFORMED_OR_INCOMPLETE_PRICE": "malformed_or_incomplete_price_rows",
                    "AMBIGUOUS_MONEYLINE_MARKET": "ambiguous_rows",
                    "NOT_PREGAME": "not_pregame_rejected_rows",
                }.get(reason, "ambiguous_rows")
                counts[reason_counter_key] += 1

        if selected is None:
            rows.append(
                P37MarketJoinRow(
                    prediction=prediction,
                    market_snapshot_status="REJECTED" if matching else "NO_MARKET",
                    rejection_reason=reason or "NO_EXACT_MARKET_IDENTITY",
                    market_snapshot_id=None,
                    market_observed_at_utc=None,
                    local_fetched_at_utc=None,
                    provider_observed_at_utc=None,
                    source_match_id=(next(iter(source_match_ids)) if len(source_match_ids) == 1 else None),
                    home_decimal_price=None,
                    away_decimal_price=None,
                    home_implied_probability=None,
                    away_implied_probability=None,
                    raw_model_vs_market_home_probability_delta=None,
                )
            )
            continue

        home_implied = Decimal("1") / selected.home_decimal_price
        away_implied = Decimal("1") / selected.away_decimal_price
        rows.append(
            P37MarketJoinRow(
                prediction=prediction,
                market_snapshot_status="USABLE_PREGAME",
                rejection_reason=None,
                market_snapshot_id=selected.snapshot_id,
                market_observed_at_utc=selected.market_observed_at_utc,
                local_fetched_at_utc=selected.local_fetched_at_utc,
                provider_observed_at_utc=selected.provider_observed_at_utc,
                source_match_id=selected.source_match_id,
                home_decimal_price=selected.home_decimal_price,
                away_decimal_price=selected.away_decimal_price,
                home_implied_probability=home_implied,
                away_implied_probability=away_implied,
                raw_model_vs_market_home_probability_delta=(
                    prediction.challenger_home_probability - home_implied
                ),
            )
        )

    target_count = len(predictions)
    edge_ready_count = counts["usable_pregame_market_rows"]
    if not source_manifest.get("timestamp_semantics_trusted") or edge_ready_count == 0:
        conclusion = "MARKET_AUTHORITY_INSUFFICIENT"
    elif edge_ready_count == target_count:
        conclusion = "MARKET_JOIN_READY"
    else:
        conclusion = "MARKET_JOIN_PARTIAL"
    counts.setdefault("exact_identity_match_count", 0)
    counts.setdefault("usable_pregame_market_rows", 0)
    counts.setdefault("no_market_rows", 0)
    counts.setdefault("post_start_rejected_rows", 0)
    counts.setdefault("ambiguous_rows", 0)
    counts.setdefault("missing_or_untrusted_timestamp_rows", 0)
    counts.setdefault("malformed_or_incomplete_price_rows", 0)
    counts.setdefault("not_pregame_rejected_rows", 0)
    summary = {
        "schema_version": P39A_JOIN_SCHEMA_VERSION,
        "task_id": P39A_TASK_ID,
        "conclusion": conclusion,
        "conclusion_rule": P39A_CONCLUSION_RULE,
        "selected_snapshot_rule": P39A_SELECTION_RULE_TEXT,
        "timestamp_semantics": P39A_TIMESTAMP_SEMANTICS_TEXT,
        "p37_evaluable_target_count": target_count,
        "exact_identity_match_count": counts["exact_identity_match_count"],
        "usable_pregame_market_rows": edge_ready_count,
        "no_market_rows": counts["no_market_rows"],
        "post_start_rejected_rows": counts["post_start_rejected_rows"],
        "ambiguous_rows": counts["ambiguous_rows"],
        "missing_or_untrusted_timestamp_rows": counts[
            "missing_or_untrusted_timestamp_rows"
        ],
        "malformed_or_incomplete_price_rows": counts[
            "malformed_or_incomplete_price_rows"
        ],
        "not_pregame_rejected_rows": counts["not_pregame_rejected_rows"],
        "edge_ready_count": edge_ready_count,
        "selected_snapshot_count": len(snapshots),
        "source_manifest": dict(source_manifest),
        "p37_manifest": dict(p37_manifest),
        "claims": {
            "bet_pass": "NOT_RUN",
            "roi_profitability": "NOT_RUN",
            "staking_bankroll_kelly": "NOT_RUN",
            "model_promotion": "NOT_RUN",
            "calibration": "NOT_RUN",
            "outcome_based_snapshot_selection": False,
        },
        "market_join_fingerprint": sha256(
            b"".join(canonical_json_bytes(row.to_projection()) for row in rows)
        ).hexdigest(),
        "selected_snapshot_fingerprint": sha256(
            b"".join(
                canonical_json_bytes(snapshot.to_projection())
                for snapshot in sorted(snapshots, key=lambda item: item.snapshot_id)
            )
        ).hexdigest(),
        "deterministic_rerun_verified": False,
    }
    return P39AMarketJoinResult(
        predictions=tuple(sorted(predictions, key=lambda row: row.key())),
        join_rows=tuple(sorted(rows, key=lambda row: row.prediction.key())),
        selected_snapshots=tuple(sorted(snapshots, key=lambda item: item.snapshot_id)),
        source_manifest=dict(source_manifest),
        p37_manifest=dict(p37_manifest),
        summary=summary,
    )


def mark_deterministic_rerun_verified(result: P39AMarketJoinResult) -> P39AMarketJoinResult:
    summary = dict(result.summary)
    summary["deterministic_rerun_verified"] = True
    return P39AMarketJoinResult(
        predictions=result.predictions,
        join_rows=result.join_rows,
        selected_snapshots=result.selected_snapshots,
        source_manifest=result.source_manifest,
        p37_manifest=result.p37_manifest,
        summary=summary,
    )


__all__ = (
    "P37OOSPrediction",
    "P39AMarketJoinResult",
    "P39A_CONCLUSION_RULE",
    "P39A_JOIN_ROW_SCHEMA_VERSION",
    "P39A_JOIN_SCHEMA_VERSION",
    "P39A_REPORT_RELATIVE_PATH",
    "P37_FOLD_INPUTS",
    "P37_COMPARISONS_RELATIVE_PATH",
    "P37_SUMMARY_RELATIVE_PATH",
    "P37MarketJoinRow",
    "build_p39a_market_join",
    "load_p37_predictions",
    "mark_deterministic_rerun_verified",
    "source_scope_keys",
)
