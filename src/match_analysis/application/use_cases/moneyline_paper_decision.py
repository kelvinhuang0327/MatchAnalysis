"""Odds-aware, isolated paper decisions for frozen Moneyline predictions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .p40a_moneyline_paper_bet_pass import P40APredictionRow
from .p44a_normalized_workflow_input import (
    NormalizedResultRecord,
    load_normalized_result_input,
)
from .p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    REGULAR_SEASON_GAME_TYPE,
    build_p50c_prediction_ledger_record,
    compute_prediction_row_fingerprint,
    effective_game_type,
    read_json_object,
    read_jsonl_objects,
)


PAPER_DECISION_SCHEMA = "moneyline_paper_decision_v1.decision.v1"
PAPER_STATE_SCHEMA = "moneyline_paper_decision_v1.state.v1"
PAPER_POSITION_SCHEMA = "moneyline_paper_decision_v1.position.v1"
PAPER_SETTLEMENT_SCHEMA = "moneyline_paper_decision_v1.settlement.v1"
PAPER_EVALUATION_SCHEMA = "moneyline_paper_decision_v1.evaluation.v1"
PAPER_POLICY_VERSION = "moneyline-paper-decision-v1"
PAPER_REPORT_RELATIVE_PATH = Path("report/moneyline_paper_decision_v1")
DATA_MODE_FIXTURE = "FIXTURE"
DATA_MODE_PROSPECTIVE = "PROSPECTIVE"
DECISION_BET = "BET"
DECISION_NO_BET = "NO_BET"
SETTLEMENT_PENDING = "PENDING"
SETTLEMENT_WIN = "WIN"
SETTLEMENT_LOSS = "LOSS"
SETTLEMENT_VOID = "VOID"
PAPER_STAKE_UNITS = Decimal("1")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _identity(value: Any) -> str:
    from hashlib import sha256

    return sha256(_canonical_json_bytes(value)).hexdigest()


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    normalized = format(value.normalize(), "f")
    return "0" if normalized in ("-0", "") else normalized


def _decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be a finite decimal")
    return parsed


def _timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return parsed.astimezone(UTC)


def format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def now_utc() -> str:
    return format_utc(datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class PaperPolicyConfig:
    """The three explicit inputs for the V1 decision policy.

    Missing values are retained so the consumer can write auditable NO_BET /
    CONFIG_MISSING decisions. Supplied invalid values fail closed at creation.
    """

    min_net_ev: Decimal | str | None = None
    max_quote_age_seconds: int | None = None
    paper_cost_per_unit: Decimal | str | None = None

    def __post_init__(self) -> None:
        if self.min_net_ev is not None:
            minimum = _decimal(self.min_net_ev, field="min_net_ev")
            if minimum <= 0:
                raise ValueError("min_net_ev must be positive")
            object.__setattr__(self, "min_net_ev", minimum)
        if self.max_quote_age_seconds is not None:
            if (
                isinstance(self.max_quote_age_seconds, bool)
                or not isinstance(self.max_quote_age_seconds, int)
                or self.max_quote_age_seconds <= 0
            ):
                raise ValueError("max_quote_age_seconds must be a positive integer")
        if self.paper_cost_per_unit is not None:
            cost = _decimal(self.paper_cost_per_unit, field="paper_cost_per_unit")
            if cost < 0:
                raise ValueError("paper_cost_per_unit must be non-negative")
            object.__setattr__(self, "paper_cost_per_unit", cost)

    @property
    def complete(self) -> bool:
        return (
            self.min_net_ev is not None
            and self.max_quote_age_seconds is not None
            and self.paper_cost_per_unit is not None
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "min_net_ev": _decimal_text(self.min_net_ev),
            "max_quote_age_seconds": self.max_quote_age_seconds,
            "paper_cost_per_unit": _decimal_text(self.paper_cost_per_unit),
            "stake_units": "1",
        }


@dataclass(frozen=True, slots=True)
class FrozenPredictionRun:
    run_id: str
    run_classification: str
    run_dir: Path
    predictions: tuple[dict[str, Any], ...]
    game_type: str


def load_frozen_prediction_run(
    run_dir: str | Path,
    *,
    data_mode: str,
) -> FrozenPredictionRun:
    """Read, but never modify, one existing P50C frozen prediction run."""

    if data_mode not in (DATA_MODE_FIXTURE, DATA_MODE_PROSPECTIVE):
        raise ValueError("data_mode must be FIXTURE or PROSPECTIVE")
    resolved = Path(run_dir).resolve()
    manifest_path = resolved / "run_manifest.json"
    predictions_path = resolved / "frozen_predictions.jsonl"
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("frozen prediction run requires its manifest and predictions")
    manifest = read_json_object(manifest_path)
    run_id = manifest.get("run_id")
    classification = manifest.get("run_classification")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("frozen prediction manifest has no run identity")
    if data_mode == DATA_MODE_PROSPECTIVE and (
        classification != CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION
    ):
        raise ValueError("PROSPECTIVE mode requires a prospective P50C prediction run")
    predictions = read_jsonl_objects(predictions_path)
    if not predictions:
        raise ValueError("frozen prediction run contains no predictions")
    manifest_game_type = effective_game_type(manifest)
    seen_predictions: set[str] = set()
    seen_games: set[int] = set()
    fingerprints: list[str] = []
    for row in predictions:
        authority = row.get("prediction_authority")
        game = row.get("game_identity")
        if not isinstance(authority, Mapping) or not isinstance(game, Mapping):
            raise ValueError("frozen prediction is missing P50C authority or game identity")
        if "game_type" in manifest and "game_type" not in game and "game_type" not in row:
            raise ValueError("P50C_FRESH_GAME_TYPE_REQUIRED: prediction is missing frozen game_type")
        if "game_type" in row and "game_type" in game and row["game_type"] != game["game_type"]:
            raise ValueError("P50C_GAME_TYPE_MISMATCH: frozen row and game identity disagree")
        if effective_game_type(row) != manifest_game_type:
            raise ValueError("P50C_GAME_TYPE_MISMATCH: frozen prediction disagrees with its manifest")
        if authority.get("run_id") != run_id:
            raise ValueError("frozen prediction run identity disagrees with its manifest")
        if authority.get("run_classification") != classification:
            raise ValueError("frozen prediction classification disagrees with its manifest")
        prediction_fp = row.get("prediction_fingerprint")
        if not isinstance(prediction_fp, str) or not prediction_fp:
            raise ValueError("frozen prediction is missing its fingerprint")
        game_pk = game.get("game_pk")
        if isinstance(game_pk, bool) or not isinstance(game_pk, int) or game_pk <= 0:
            raise ValueError("frozen prediction has an invalid gamePk")
        if prediction_fp in seen_predictions or game_pk in seen_games:
            raise ValueError("frozen run contains duplicate prediction or game identity")
        model = row.get("model_identity")
        if not isinstance(model, Mapping) or model.get("model_probability_source") != "calibrated_p_home":
            raise ValueError("frozen prediction is not a P50C Moneyline probability")
        try:
            p_home = _decimal(model.get("p_home"), field="model_identity.p_home")
            p_away = _decimal(model.get("p_away"), field="model_identity.p_away")
            p37_id = str(authority["p37_prediction_row_id"])
            game_number = int(game["game_number"])
            prediction_row = P40APredictionRow(
                p37_fold_id="frozen-p50c",
                p37_window="frozen-p50c",
                p37_prediction_row_id=p37_id,
                provider_namespace=str(game["provider_namespace"]),
                provider_game_id=str(game["provider_game_id"]),
                game_pk=game_pk,
                game_number=game_number,
                game_type=effective_game_type(row),
                scheduled_start_utc=str(game["scheduled_start_utc"]),
                champion_model_id=str(model["model_id"]),
                champion_model_fingerprint=str(model["model_fingerprint"]),
                champion_home_probability=p_home,
                challenger_model_id=str(model["model_id"]),
                challenger_model_fingerprint=str(model["model_fingerprint"]),
                challenger_home_probability=p_home,
            )
            expected_fp = compute_prediction_row_fingerprint(
                prediction_row,
                official_date=str(game["official_date"]),
                home_team=str(game.get("home_team") or ""),
                away_team=str(game.get("away_team") or ""),
                home_team_code=str(game.get("home_team_code") or ""),
                away_team_code=str(game.get("away_team_code") or ""),
            )
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            raise ValueError("frozen prediction has incomplete P50C Moneyline authority") from exc
        if expected_fp != prediction_fp:
            raise ValueError("P50C_FROZEN_PREDICTION_FINGERPRINT_MISMATCH")
        if p_away != Decimal("1") - p_home:
            raise ValueError("frozen P50C home/away probabilities disagree")
        expected_selection = "HOME" if p_home >= Decimal("0.5") else "AWAY"
        if model.get("selection") != expected_selection:
            raise ValueError("frozen P50C selection disagrees with its probability")
        seen_predictions.add(prediction_fp)
        seen_games.add(game_pk)
        fingerprints.append(prediction_fp)
    if manifest.get("prediction_fingerprints") != fingerprints:
        raise ValueError("P50C_FROZEN_PREDICTION_MANIFEST_MISMATCH")
    if manifest.get("prediction_bundle_fingerprint") != _identity(fingerprints):
        raise ValueError("P50C_FROZEN_PREDICTION_BUNDLE_MISMATCH")
    return FrozenPredictionRun(
        run_id=run_id,
        run_classification=str(classification or ""),
        run_dir=resolved,
        predictions=tuple(predictions),
        game_type=manifest_game_type,
    )


def load_quote_snapshots(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read normalized, two-way full-game Moneyline quote snapshots."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"quote snapshot input does not exist: {resolved}")
    return tuple(read_jsonl_objects(resolved))


def _game_pk(row: Mapping[str, Any]) -> int | None:
    game = row.get("game_identity")
    value = game.get("game_pk") if isinstance(game, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _quote_index(
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[Mapping[str, Any], ...]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in snapshots:
        game_pk = _game_pk(row)
        if game_pk is not None:
            grouped.setdefault(game_pk, []).append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _prediction_projection(
    run: FrozenPredictionRun,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    authority = prediction.get("prediction_authority")
    game = prediction.get("game_identity")
    model = prediction.get("model_identity")
    authority = authority if isinstance(authority, Mapping) else {}
    game = game if isinstance(game, Mapping) else {}
    model = model if isinstance(model, Mapping) else {}
    game_type = effective_game_type(prediction)
    game_identity = dict(game)
    game_identity["game_type"] = game_type
    return {
        "run_id": run.run_id,
        "run_classification": run.run_classification,
        "workflow_prediction_id": prediction.get("workflow_prediction_id"),
        "prediction_fingerprint": prediction.get("prediction_fingerprint"),
        "p37_prediction_row_id": authority.get("p37_prediction_row_id"),
        "model_id": model.get("model_id"),
        "model_fingerprint": model.get("model_fingerprint"),
        "model_probability_source": model.get("model_probability_source"),
        "created_at_utc": authority.get("created_at_utc"),
        "game_type": game_type,
        "game_identity": game_identity,
        "model_identity": dict(model),
        "model_domain_status": model.get("model_domain_status"),
    }


def _prediction_selection_probability(
    prediction: Mapping[str, Any],
) -> tuple[str | None, Decimal | None, str | None]:
    model = prediction.get("model_identity")
    if not isinstance(model, Mapping):
        return None, None, "PREDICTION_AUTHORITY_MISSING"
    selection = model.get("selection")
    if selection not in ("HOME", "AWAY"):
        return None, None, "PREDICTION_SELECTION_INVALID"
    try:
        p_home = _decimal(model.get("p_home"), field="p_home")
        p_away = _decimal(model.get("p_away"), field="p_away")
    except ValueError:
        return str(selection), None, "PREDICTION_PROBABILITY_INVALID"
    if not (Decimal("0") <= p_home <= Decimal("1")) or not (
        Decimal("0") <= p_away <= Decimal("1")
    ):
        return str(selection), None, "PREDICTION_PROBABILITY_INVALID"
    if abs((p_home + p_away) - Decimal("1")) > Decimal("0.000000000001"):
        return str(selection), None, "PREDICTION_PROBABILITY_INVALID"
    return str(selection), p_home if selection == "HOME" else p_away, None


def _quote_matches_prediction(
    quote: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> bool:
    quote_identity = quote.get("game_identity")
    prediction_identity = prediction.get("game_identity")
    if not isinstance(quote_identity, Mapping) or not isinstance(
        prediction_identity, Mapping
    ):
        return False
    identity_fields = (
        "game_pk",
        "provider_namespace",
        "provider_game_id",
        "game_number",
        "official_date",
        "scheduled_start_utc",
        "home_team_code",
        "away_team_code",
    )
    for field in identity_fields:
        if quote_identity.get(field) != prediction_identity.get(field):
            return False
    quote_game_type = quote_identity.get("game_type")
    if quote_game_type is not None and quote_game_type != effective_game_type(prediction):
        return False
    for field in ("home_team", "away_team"):
        left = quote_identity.get(field)
        right = prediction_identity.get(field)
        if left is not None and right is not None and left != right:
            return False
    return True


def _snapshot_projection(quote: Mapping[str, Any]) -> dict[str, Any]:
    source = quote.get("quote_source")
    source_identity = quote.get("source_identity")
    if not isinstance(source_identity, Mapping):
        source_identity = {}
    snapshot_id = quote.get("snapshot_identity")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        snapshot_id = _identity({key: value for key, value in quote.items() if key != "snapshot_identity"})
    return {
        "snapshot_identity": snapshot_id,
        "source": source,
        "source_identity": dict(source_identity),
        "market": quote.get("market"),
        "market_code": quote.get("market_code"),
        "observed_at_utc": quote.get("observed_at_utc"),
        "game_identity": dict(quote.get("game_identity", {}))
        if isinstance(quote.get("game_identity"), Mapping)
        else None,
        "home_decimal_odds": quote.get("home_decimal_odds"),
        "away_decimal_odds": quote.get("away_decimal_odds"),
    }


def _base_decision(
    *,
    run: FrozenPredictionRun,
    prediction: Mapping[str, Any],
    quote: Mapping[str, Any] | None,
    decision_at_utc: str,
    config: PaperPolicyConfig,
    data_mode: str,
    missing_quote_reason: str,
) -> tuple[dict[str, Any], Decimal | None]:
    projection = _prediction_projection(run, prediction)
    game = prediction["game_identity"]
    selection, probability, probability_error = _prediction_selection_probability(prediction)
    quote_projection = _snapshot_projection(quote) if quote is not None else None
    model = prediction.get("model_identity")
    reason: str | None = None
    estimated_ev: Decimal | None = None
    selected_odds: Decimal | None = None

    if not config.complete:
        reason = "CONFIG_MISSING"
    elif probability_error is not None:
        reason = probability_error
    elif not isinstance(model, Mapping) or not model.get("model_id") or not model.get("model_fingerprint"):
        reason = "PREDICTION_AUTHORITY_MISSING"
    elif "run" in str(model.get("model_probability_source", "")).lower() and "line" in str(
        model.get("model_probability_source", "")
    ).lower():
        reason = "PREDICTION_MARKET_MISMATCH"
    elif any(
        not isinstance(game.get(field), str) or not game.get(field).strip()
        for field in (
            "provider_namespace",
            "provider_game_id",
            "official_date",
            "home_team",
            "away_team",
            "home_team_code",
            "away_team_code",
        )
    ) or game.get("home_team_code") in ("HOME", "AWAY") or game.get("away_team_code") in ("HOME", "AWAY"):
        reason = "PREDICTION_IDENTITY_INVALID"

    decision_time: datetime | None = None
    game_start: datetime | None = None
    prediction_created: datetime | None = None
    if reason is None:
        try:
            decision_time = _timestamp(decision_at_utc, field="decision_at_utc")
            game_start = _timestamp(game.get("scheduled_start_utc"), field="scheduled_start_utc")
            prediction_created = _timestamp(
                projection.get("created_at_utc"), field="prediction.created_at_utc"
            )
        except ValueError:
            reason = "PREDICTION_TIME_INVALID"
    if reason is None and decision_time is not None and game_start is not None:
        if decision_time >= game_start:
            reason = "GAME_ALREADY_STARTED"
    if reason is None and decision_time is not None and prediction_created is not None:
        if prediction_created >= decision_time:
            reason = "PREDICTION_NOT_PREEXISTING"
    if reason is None and quote is None:
        reason = missing_quote_reason
    if reason is None and quote is not None:
        source_identity = quote.get("source_identity")
        if (
            not isinstance(quote.get("quote_source"), str)
            or not quote.get("quote_source", "").strip()
            or not isinstance(source_identity, Mapping)
            or not source_identity
        ):
            reason = "QUOTE_PROVENANCE_MISSING"
        elif data_mode == DATA_MODE_PROSPECTIVE and str(quote.get("quote_source", "")).upper() == "FIXTURE":
            reason = "FIXTURE_QUOTE_NOT_PROSPECTIVE"
        elif quote.get("market") != "MONEYLINE" or quote.get("market_code") not in (
            "MNL",
            "MONEYLINE",
        ):
            reason = "QUOTE_MARKET_MISMATCH"
        elif not _quote_matches_prediction(quote, prediction):
            reason = "QUOTE_IDENTITY_MISMATCH"

    observed_at: datetime | None = None
    if reason is None and quote is not None:
        try:
            observed_at = _timestamp(quote.get("observed_at_utc"), field="quote.observed_at_utc")
        except ValueError:
            reason = "QUOTE_TIMESTAMP_INVALID"
        if reason is None and decision_time is not None and observed_at is not None:
            if observed_at > decision_time:
                reason = "QUOTE_FROM_FUTURE"
            elif game_start is not None and observed_at >= game_start:
                reason = "QUOTE_NOT_PREGAME"
            elif (decision_time - observed_at).total_seconds() > int(config.max_quote_age_seconds or 0):
                reason = "QUOTE_EXPIRED"

    if reason is None and quote is not None and selection is not None and probability is not None:
        try:
            home_odds = _decimal(quote.get("home_decimal_odds"), field="home_decimal_odds")
            away_odds = _decimal(quote.get("away_decimal_odds"), field="away_decimal_odds")
        except ValueError:
            reason = "QUOTE_ODDS_INVALID"
        else:
            if home_odds <= 1 or away_odds <= 1:
                reason = "QUOTE_ODDS_INVALID"
            else:
                selected_odds = home_odds if selection == "HOME" else away_odds
                estimated_ev = (
                    probability * selected_odds
                    - Decimal("1")
                    - Decimal(config.paper_cost_per_unit or "0")
                )
                if estimated_ev < Decimal(config.min_net_ev or "0"):
                    reason = "EV_BELOW_MIN_NET_EV"

    accepted = reason is None
    return (
        {
            "schema_version": PAPER_DECISION_SCHEMA,
            "paper_only": True,
            "not_a_real_fill": True,
            "decision_identity": {
                "prediction_run_id": run.run_id,
                "prediction_fingerprint": prediction.get("prediction_fingerprint"),
                "workflow_prediction_id": prediction.get("workflow_prediction_id"),
                "game_pk": game.get("game_pk"),
                "game_type": effective_game_type(prediction),
                "policy_version": PAPER_POLICY_VERSION,
            },
            "game_type": effective_game_type(prediction),
            "model_domain_status": model.get("model_domain_status") if isinstance(model, Mapping) else None,
            "prediction": projection,
            "market": "MONEYLINE",
            "selection": selection,
            "probability": _decimal_text(probability),
            "quote_snapshot": quote_projection,
            "decimal_odds": _decimal_text(selected_odds),
            "decision_at_utc": decision_at_utc,
            "policy_version": PAPER_POLICY_VERSION,
            "policy_parameters": config.to_payload(),
            "estimated_net_ev": _decimal_text(estimated_ev),
            "decision": DECISION_BET if accepted else DECISION_NO_BET,
            "reason_code": "EV_THRESHOLD_MET" if accepted else reason,
            "data_mode": data_mode,
        },
        selected_odds,
    )


def _decision_id(row: Mapping[str, Any]) -> str:
    return _identity({key: value for key, value in row.items() if key != "decision_id"})


def _new_position(decision: Mapping[str, Any], *, cost: Decimal) -> dict[str, Any]:
    identity = decision["decision_identity"]
    position_id = _identity(
        {
            "policy_version": PAPER_POLICY_VERSION,
            "game_pk": identity["game_pk"],
            "game_type": identity["game_type"],
            "prediction_run_id": identity["prediction_run_id"],
            "prediction_fingerprint": identity["prediction_fingerprint"],
        }
    )
    return {
        "schema_version": PAPER_POSITION_SCHEMA,
        "position_id": position_id,
        "decision_id": decision["decision_id"],
        "policy_version": PAPER_POLICY_VERSION,
        "prediction_run_id": identity["prediction_run_id"],
        "prediction_fingerprint": identity["prediction_fingerprint"],
        "game_pk": identity["game_pk"],
        "game_type": identity["game_type"],
        "model_domain_status": decision.get("model_domain_status"),
        "market": "MONEYLINE",
        "selection": decision["selection"],
        "probability": decision["probability"],
        "quote_snapshot": decision["quote_snapshot"],
        "decimal_odds": decision["decimal_odds"],
        "stake_units": "1",
        "paper_cost_per_unit": _decimal_text(cost),
        "paper_only": True,
        "not_a_real_fill": True,
        "data_mode": decision["data_mode"],
    }


def _result_index(
    results: Sequence[NormalizedResultRecord],
) -> tuple[dict[str, NormalizedResultRecord], dict[tuple[str, str, int, str], NormalizedResultRecord]]:
    by_prediction: dict[str, NormalizedResultRecord] = {}
    by_game: dict[tuple[str, str, int, str], NormalizedResultRecord] = {}
    for result in results:
        key = (result.provider_namespace, result.provider_game_id, result.game_number, result.game_type)
        if result.prediction_row_id in by_prediction or key in by_game:
            raise RuntimeError("PAPER_CONFLICTING_FINAL_RESULT")
        by_prediction[result.prediction_row_id] = result
        by_game[key] = result
    return by_prediction, by_game


def _settlement_for_position(
    position: Mapping[str, Any],
    result: NormalizedResultRecord,
    *,
    settled_at_utc: str,
) -> dict[str, Any]:
    if position.get("game_type", REGULAR_SEASON_GAME_TYPE) != result.game_type:
        raise RuntimeError("PAPER_GAME_TYPE_MISMATCH")
    selection = position["selection"]
    if selection not in ("HOME", "AWAY"):
        raise ValueError("paper position has an invalid selection")
    try:
        quote = position["quote_snapshot"]
        quoted_odds = _decimal(position["decimal_odds"], field="position.decimal_odds")
        cost = _decimal(position["paper_cost_per_unit"], field="position.paper_cost_per_unit")
    except (KeyError, ValueError) as exc:
        raise RuntimeError("PAPER_POSITION_AUTHORITY_INVALID") from exc
    if result.status != "FINAL":
        raise RuntimeError("PAPER_NON_FINAL_RESULT_REJECTED")
    actual_winner = result.actual_winner
    win = selection == actual_winner
    net = quoted_odds - Decimal("1") - cost if win else -Decimal("1") - cost
    provisional = {
        "schema_version": PAPER_SETTLEMENT_SCHEMA,
        "position_id": position["position_id"],
        "decision_id": position["decision_id"],
        "game_pk": position["game_pk"],
        "game_type": position.get("game_type", REGULAR_SEASON_GAME_TYPE),
        "market": "MONEYLINE",
        "selection": selection,
        "decimal_odds": _decimal_text(quoted_odds),
        "paper_cost_per_unit": _decimal_text(cost),
        "stake_units": "1",
        "status": SETTLEMENT_WIN if win else SETTLEMENT_LOSS,
        "net_paper_units": _decimal_text(net),
        "result_authority": {
            "provider_namespace": result.provider_namespace,
            "provider_game_id": result.provider_game_id,
            "game_number": result.game_number,
            "game_type": result.game_type,
            "status": result.status,
            "home_score": result.home_score,
            "away_score": result.away_score,
            "actual_winner": actual_winner,
            "result_observed_at_utc": result.result_observed_at_utc,
            "source_identity": result.source_identity,
        },
        "settled_at_utc": settled_at_utc,
        "paper_only": True,
        "not_a_real_fill": True,
        "quote_snapshot_identity": quote["snapshot_identity"],
        "data_mode": position["data_mode"],
    }
    provisional["settlement_id"] = _identity(
        {
            "position_id": position["position_id"],
            "provider_namespace": result.provider_namespace,
            "provider_game_id": result.provider_game_id,
            "game_number": result.game_number,
            "game_type": result.game_type,
            "status": result.status,
            "home_score": result.home_score,
            "away_score": result.away_score,
        }
    )
    return provisional


def _merge_rows(
    current: Sequence[Mapping[str, Any]],
    incoming: Sequence[Mapping[str, Any]],
    *,
    identity_key: str,
) -> list[dict[str, Any]]:
    merged = {str(row[identity_key]): dict(row) for row in current}
    for row in incoming:
        identity = str(row[identity_key])
        previous = merged.get(identity)
        if previous is not None and previous != dict(row):
            if identity_key in ("settlement_id", "evaluation_id"):
                prev_authority = previous.get("result_authority", {})
                next_authority = row.get("result_authority", {})
                if identity_key == "settlement_id":
                    authority_fields = (
                        "provider_namespace",
                        "provider_game_id",
                        "game_number",
                        "game_type",
                        "status",
                        "home_score",
                        "away_score",
                    )
                    equal_authority = all(
                        prev_authority.get(field) == next_authority.get(field)
                        for field in authority_fields
                    )
                else:
                    evaluation_fields = (
                        "prediction_run_id",
                        "prediction_fingerprint",
                        "game_pk",
                        "game_type",
                        "selection",
                        "probability",
                        "actual_winner",
                        "brier_component",
                        "status",
                    )
                    equal_authority = all(
                        previous.get(field) == row.get(field)
                        for field in evaluation_fields
                    )
                if not equal_authority:
                    raise RuntimeError("PAPER_CONFLICTING_SETTLEMENT")
                continue
            raise RuntimeError(f"PAPER_IDENTITY_CONFLICT: {identity_key}")
        merged.setdefault(identity, dict(row))
    return sorted(merged.values(), key=lambda row: str(row[identity_key]))


def _load_state(path: Path, *, data_mode: str, config: PaperPolicyConfig) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": PAPER_STATE_SCHEMA,
            "paper_only": True,
            "not_a_real_fill": True,
            "data_mode": data_mode,
            "policy_version": PAPER_POLICY_VERSION,
            "policy_parameters": config.to_payload(),
            "decisions": [],
            "positions": [],
            "settlements": [],
            "prediction_evaluations": [],
            "source_operations": [],
        }
    state = read_json_object(path)
    if state.get("schema_version") != PAPER_STATE_SCHEMA:
        raise RuntimeError("PAPER_STATE_SCHEMA_CONFLICT")
    if state.get("paper_only") is not True or state.get("not_a_real_fill") is not True:
        raise RuntimeError("PAPER_STATE_MARKER_MISSING")
    if state.get("data_mode") != data_mode:
        raise RuntimeError("PAPER_STATE_DATA_MODE_CONFLICT")
    if state.get("policy_version") != PAPER_POLICY_VERSION:
        raise RuntimeError("PAPER_STATE_POLICY_CONFLICT")
    if state.get("policy_parameters") != config.to_payload():
        previous_parameters = state.get("policy_parameters", {})
        if (
            not config.complete
            or any(previous_parameters.get(key) is not None for key in (
                "min_net_ev",
                "max_quote_age_seconds",
                "paper_cost_per_unit",
            ))
            or state.get("positions")
        ):
            raise RuntimeError("PAPER_STATE_POLICY_CONFLICT")
        state["policy_parameters"] = config.to_payload()
    for key in ("decisions", "positions", "settlements", "prediction_evaluations", "source_operations"):
        if not isinstance(state.get(key), list):
            raise RuntimeError(f"PAPER_STATE_INVALID_COLLECTION: {key}")
    return state


def _paper_row_game_type(row: Mapping[str, Any]) -> str:
    identity = row.get("decision_identity")
    value = identity.get("game_type") if isinstance(identity, Mapping) else row.get("game_type")
    if value is None:
        return REGULAR_SEASON_GAME_TYPE
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("PAPER_GAME_TYPE_INVALID")
    return value


def _metrics(
    state: Mapping[str, Any],
    *,
    _include_cohorts: bool = True,
) -> dict[str, Any]:
    decisions = tuple(state["decisions"])
    positions = tuple(state["positions"])
    settlements = tuple(state["settlements"])
    evaluations = tuple(state["prediction_evaluations"])
    decision_counts = Counter(row["decision"] for row in decisions)
    reason_counts = Counter(row["reason_code"] for row in decisions)
    settlement_counts = Counter(row["status"] for row in settlements)
    settled_position_ids = {row["position_id"] for row in settlements}
    pending_count = sum(1 for row in positions if row["position_id"] not in settled_position_ids)
    unique_games = {row.get("decision_identity", {}).get("game_pk") for row in decisions}
    unique_games.discard(None)

    wins = settlement_counts.get(SETTLEMENT_WIN, 0)
    losses = settlement_counts.get(SETTLEMENT_LOSS, 0)
    voids = settlement_counts.get(SETTLEMENT_VOID, 0)
    settled_non_void_stake = Decimal(wins + losses)
    total_net = sum(
        (_decimal(row["net_paper_units"], field="net_paper_units") for row in settlements if row["status"] in (SETTLEMENT_WIN, SETTLEMENT_LOSS)),
        Decimal("0"),
    )
    win_denominator = wins + losses

    bet_keys = {
        (row["decision_identity"]["prediction_run_id"], row["decision_identity"]["prediction_fingerprint"])
        for row in decisions
        if row["decision"] == DECISION_BET
    }
    all_brier = [
        _decimal(row["brier_component"], field="brier_component")
        for row in evaluations
        if row.get("brier_component") is not None
    ]
    bet_brier = [
        _decimal(row["brier_component"], field="brier_component")
        for row in evaluations
        if row.get("brier_component") is not None
        and (row["prediction_run_id"], row["prediction_fingerprint"]) in bet_keys
    ]

    def score(values: Sequence[Decimal]) -> dict[str, Any]:
        if not values:
            return {"value": None, "denominator": 0, "status": "NOT_ENOUGH_DATA"}
        return {
            "value": _decimal_text(sum(values, Decimal("0")) / Decimal(len(values))),
            "denominator": len(values),
            "status": "AVAILABLE",
        }

    roi = (
        _decimal_text(total_net / settled_non_void_stake)
        if settled_non_void_stake
        else None
    )
    win_rate = (
        _decimal_text(Decimal(wins) / Decimal(win_denominator))
        if win_denominator
        else None
    )
    metrics = {
        "schema_version": "moneyline_paper_decision_v1.metrics.v1",
        "data_mode": state["data_mode"],
        "aggregation_scope": "ALL_GAME_TYPES",
        "unique_candidate_games": {"value": len(unique_games), "denominator": len(unique_games)},
        "decisions": {
            "bet": decision_counts.get(DECISION_BET, 0),
            "no_bet": decision_counts.get(DECISION_NO_BET, 0),
            "denominator": len(decisions),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "paper_positions": {
            "pending": pending_count,
            "win": wins,
            "loss": losses,
            "void": voids,
            "denominator": len(positions),
        },
        "realized_paper_net_pnl": {
            "value": _decimal_text(total_net) if settlements else None,
            "denominator": len(settlements),
            "status": "AVAILABLE" if settlements else "NOT_ENOUGH_DATA",
        },
        "settled_non_void_stake_units": {
            "value": _decimal_text(settled_non_void_stake),
            "denominator": wins + losses,
        },
        "paper_roi": {
            "value": roi,
            "denominator": wins + losses,
            "status": "AVAILABLE" if wins + losses else "NOT_ENOUGH_DATA",
        },
        "win_rate": {
            "value": win_rate,
            "denominator": win_denominator,
            "status": "AVAILABLE" if win_denominator else "NOT_ENOUGH_DATA",
        },
        "brier_all_candidates": score(all_brier),
        "brier_bet_subset": score(bet_brier),
        "void_support": "NOT_SUPPORTED_BY_CURRENT_OFFICIAL_RESULT_CONTRACT",
    }
    if not _include_cohorts:
        return metrics

    collections = {
        "decisions": tuple(state["decisions"]),
        "positions": tuple(state["positions"]),
        "settlements": tuple(state["settlements"]),
        "prediction_evaluations": tuple(state["prediction_evaluations"]),
    }
    game_types = sorted(
        {
            _paper_row_game_type(row)
            for rows in collections.values()
            for row in rows
        }
    )
    metrics["game_type_cohorts"] = {}
    for game_type in game_types:
        cohort_state = dict(state)
        for name, rows in collections.items():
            cohort_state[name] = [
                row for row in rows if _paper_row_game_type(row) == game_type
            ]
        cohort_metrics = _metrics(cohort_state, _include_cohorts=False)
        cohort_metrics["game_type"] = game_type
        cohort_metrics["aggregation_scope"] = "SINGLE_GAME_TYPE"
        metrics["game_type_cohorts"][game_type] = cohort_metrics
    return metrics


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == payload:
        return
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary_path = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def run_moneyline_paper_decision(
    *,
    run: FrozenPredictionRun,
    quote_snapshots: Sequence[Mapping[str, Any]] | None,
    output_dir: str | Path,
    config: PaperPolicyConfig,
    decision_at_utc: str,
    settled_at_utc: str,
    data_mode: str,
    final_results: Sequence[NormalizedResultRecord] = (),
    quote_missing_reason: str = "ODDS_SOURCE_NOT_CONFIGURED",
    quote_issues_by_game_pk: Mapping[int, str] | None = None,
    source_operations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Record decisions, one-unit positions, results, and denominator-aware metrics."""

    if data_mode not in (DATA_MODE_FIXTURE, DATA_MODE_PROSPECTIVE):
        raise ValueError("data_mode must be FIXTURE or PROSPECTIVE")
    decision_time = _timestamp(decision_at_utc, field="decision_at_utc")
    settled_time = _timestamp(settled_at_utc, field="settled_at_utc")
    if settled_time < decision_time:
        raise ValueError("settled_at_utc must not precede decision_at_utc")
    destination = Path(output_dir).resolve()
    state_path = destination / "paper_state.json"
    state = _load_state(state_path, data_mode=data_mode, config=config)
    issues = dict(quote_issues_by_game_pk or {})
    snapshots = tuple(quote_snapshots or ())
    quote_by_game = _quote_index(snapshots)
    existing_decision_ids = {row["decision_id"] for row in state["decisions"]}
    position_keys = {
        (
            row["policy_version"],
            int(row["game_pk"]),
            row.get("game_type", REGULAR_SEASON_GAME_TYPE),
        )
        for row in state["positions"]
    }
    new_decisions: list[dict[str, Any]] = []
    new_positions: list[dict[str, Any]] = []

    for prediction in run.predictions:
        game_pk = int(prediction["game_identity"]["game_pk"])
        quotes = quote_by_game.get(game_pk, ())
        quote = quotes[0] if len(quotes) == 1 else None
        missing_reason = (
            "QUOTE_AMBIGUOUS"
            if len(quotes) > 1
            else issues.get(game_pk, quote_missing_reason)
        )
        decision, _ = _base_decision(
            run=run,
            prediction=prediction,
            quote=quote,
            decision_at_utc=decision_at_utc,
            config=config,
            data_mode=data_mode,
            missing_quote_reason=missing_reason,
        )
        decision["decision_id"] = _decision_id(decision)
        if decision["decision_id"] in existing_decision_ids:
            continue
        game_type = effective_game_type(prediction)
        position_key = (PAPER_POLICY_VERSION, game_pk, game_type)
        if decision["decision"] == DECISION_BET and position_key in position_keys:
            decision["decision"] = DECISION_NO_BET
            decision["reason_code"] = "POSITION_ALREADY_EXISTS"
            decision.pop("decision_id")
            decision["decision_id"] = _decision_id(decision)
            if decision["decision_id"] in existing_decision_ids:
                continue
        new_decisions.append(decision)
        existing_decision_ids.add(decision["decision_id"])
        if decision["decision"] == DECISION_BET:
            position = _new_position(
                decision,
                cost=Decimal(config.paper_cost_per_unit or "0"),
            )
            if any(row["position_id"] == position["position_id"] for row in state["positions"]):
                raise RuntimeError("PAPER_POSITION_IDENTITY_CONFLICT")
            new_positions.append(position)
            position_keys.add(position_key)

    state["decisions"] = _merge_rows(
        state["decisions"], new_decisions, identity_key="decision_id"
    )
    state["positions"] = _merge_rows(
        state["positions"], new_positions, identity_key="position_id"
    )
    state["source_operations"] = _merge_rows(
        state["source_operations"], source_operations, identity_key="operation_id"
    )

    results_by_prediction, results_by_game = _result_index(final_results)
    decision_at_by_prediction: dict[tuple[str, str], datetime] = {}
    for decision in state["decisions"]:
        key = (
            str(decision["decision_identity"]["prediction_run_id"]),
            str(decision["decision_identity"]["prediction_fingerprint"]),
        )
        try:
            at = _timestamp(decision["decision_at_utc"], field="decision_at_utc")
        except ValueError:
            continue
        prior = decision_at_by_prediction.get(key)
        if prior is None or at < prior:
            decision_at_by_prediction[key] = at

    new_evaluations: list[dict[str, Any]] = []
    new_settlements: list[dict[str, Any]] = []
    for prediction in run.predictions:
        game = prediction["game_identity"]
        game_type = effective_game_type(prediction)
        model = prediction.get("model_identity")
        authority = prediction.get("prediction_authority")
        model = model if isinstance(model, Mapping) else {}
        authority = authority if isinstance(authority, Mapping) else {}
        pred_row_id = authority.get("p37_prediction_row_id")
        game_key = (
            str(game.get("provider_namespace", "")),
            str(game.get("provider_game_id", "")),
            int(game.get("game_number", 0)),
            game_type,
        )
        result = results_by_prediction.get(str(pred_row_id)) if pred_row_id else None
        by_game = results_by_game.get(game_key)
        if result is not None and by_game is not None and result != by_game:
            raise RuntimeError("PAPER_CONFLICTING_FINAL_RESULT")
        if result is not None and result.game_type != game_type:
            raise RuntimeError("PAPER_GAME_TYPE_MISMATCH")
        if result is None:
            result = by_game
        if result is None:
            continue
        if (
            result.provider_namespace != game.get("provider_namespace")
            or result.provider_game_id != game.get("provider_game_id")
            or result.game_number != game.get("game_number")
            or result.game_type != game_type
        ):
            raise RuntimeError("PAPER_RESULT_GAME_IDENTITY_MISMATCH")
        observed_at = _timestamp(
            result.result_observed_at_utc, field="result_observed_at_utc"
        )
        game_start = _timestamp(game.get("scheduled_start_utc"), field="scheduled_start_utc")
        candidate_key = (run.run_id, str(prediction.get("prediction_fingerprint")))
        first_decision_at = decision_at_by_prediction.get(candidate_key, decision_time)
        if observed_at < game_start or observed_at <= first_decision_at:
            raise RuntimeError("PAPER_RESULT_TIME_LEAKAGE_REJECTED")
        if observed_at > settled_time:
            raise RuntimeError("PAPER_RESULT_AFTER_SETTLEMENT_TIME")

        selection, probability, probability_error = _prediction_selection_probability(prediction)
        brier_component: Decimal | None = None
        evaluation_status = "INVALID_PROBABILITY" if probability_error else "AVAILABLE"
        if probability_error is None and selection is not None and probability is not None:
            grading_row = build_p50c_prediction_ledger_record(
                run_id=run.run_id,
                run_classification=run.run_classification,
                frozen_prediction=prediction,
                result_record=result,
                evaluated_at_utc=result.result_observed_at_utc,
            )
            brier_component = _decimal(
                grading_row["evaluation"]["brier_component"],
                field="brier_component",
            )
        evaluation_payload = {
            "schema_version": PAPER_EVALUATION_SCHEMA,
            "prediction_run_id": run.run_id,
            "prediction_fingerprint": prediction.get("prediction_fingerprint"),
            "game_pk": game.get("game_pk"),
            "game_type": game_type,
            "model_domain_status": model.get("model_domain_status"),
            "selection": selection,
            "probability": _decimal_text(probability),
            "actual_winner": result.actual_winner,
            "result_observed_at_utc": result.result_observed_at_utc,
            "source_identity": result.source_identity,
            "brier_component": _decimal_text(brier_component),
            "status": evaluation_status,
        }
        evaluation_payload["evaluation_id"] = _identity(
            {
                "prediction_run_id": run.run_id,
                "prediction_fingerprint": prediction.get("prediction_fingerprint"),
                "provider_namespace": result.provider_namespace,
                "provider_game_id": result.provider_game_id,
                "game_number": result.game_number,
                "game_type": result.game_type,
                "home_score": result.home_score,
                "away_score": result.away_score,
        }
        )
        new_evaluations.append(evaluation_payload)

    state["prediction_evaluations"] = _merge_rows(
        state["prediction_evaluations"],
        new_evaluations,
        identity_key="evaluation_id",
    )
    evaluation_keys = [
        (row["prediction_run_id"], row["prediction_fingerprint"])
        for row in state["prediction_evaluations"]
    ]
    if len(evaluation_keys) != len(set(evaluation_keys)):
        raise RuntimeError("PAPER_CONFLICTING_FINAL_RESULT")

    settled_positions = {row["position_id"] for row in state["settlements"]}
    for position in state["positions"]:
        if position["position_id"] in settled_positions:
            continue
        prediction = next(
            (
                row
                for row in run.predictions
                if row.get("prediction_fingerprint") == position.get("prediction_fingerprint")
                and run.run_id == position.get("prediction_run_id")
            ),
            None,
        )
        if prediction is None:
            # Existing positions for earlier runs remain pending until their
            # frozen run is explicitly supplied to this consumer.
            continue
        game = prediction["game_identity"]
        game_type = effective_game_type(prediction)
        authority = prediction.get("prediction_authority", {})
        pred_row_id = authority.get("p37_prediction_row_id") if isinstance(authority, Mapping) else None
        game_key = (
            str(game.get("provider_namespace", "")),
            str(game.get("provider_game_id", "")),
            int(game.get("game_number", 0)),
            game_type,
        )
        result = results_by_prediction.get(str(pred_row_id)) if pred_row_id else None
        by_game = results_by_game.get(game_key)
        if result is not None and by_game is not None and result != by_game:
            raise RuntimeError("PAPER_CONFLICTING_FINAL_RESULT")
        if result is not None and result.game_type != game_type:
            raise RuntimeError("PAPER_GAME_TYPE_MISMATCH")
        result = result or by_game
        if result is None:
            continue
        result_time = _timestamp(result.result_observed_at_utc, field="result_observed_at_utc")
        if result_time < _timestamp(game.get("scheduled_start_utc"), field="scheduled_start_utc"):
            raise RuntimeError("PAPER_RESULT_TIME_LEAKAGE_REJECTED")
        if result_time > settled_time:
            raise RuntimeError("PAPER_RESULT_AFTER_SETTLEMENT_TIME")
        new_settlements.append(
            _settlement_for_position(position, result, settled_at_utc=settled_at_utc)
        )
    state["settlements"] = _merge_rows(
        state["settlements"], new_settlements, identity_key="settlement_id"
    )
    position_ids = {row["position_id"] for row in state["positions"]}
    settled_ids = [row["position_id"] for row in state["settlements"]]
    if any(position_id not in position_ids for position_id in settled_ids) or len(settled_ids) != len(
        set(settled_ids)
    ):
        raise RuntimeError("PAPER_DUPLICATE_OR_ORPHAN_SETTLEMENT")
    state["metrics"] = _metrics(state)
    _atomic_write(state_path, _json_bytes(state))
    return state


def run_moneyline_paper_decision_from_paths(
    *,
    run_dir: str | Path,
    quote_path: str | Path | None,
    output_dir: str | Path,
    config: PaperPolicyConfig,
    decision_at_utc: str,
    settled_at_utc: str,
    data_mode: str,
    final_results_path: str | Path | None = None,
    quote_missing_reason: str = "ODDS_SOURCE_NOT_CONFIGURED",
    quote_issues_by_game_pk: Mapping[int, str] | None = None,
    source_operations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    run = load_frozen_prediction_run(run_dir, data_mode=data_mode)
    quote_rows = load_quote_snapshots(quote_path) if quote_path is not None else None
    result_rows = (
        load_normalized_result_input(final_results_path)
        if final_results_path is not None
        else ()
    )
    return run_moneyline_paper_decision(
        run=run,
        quote_snapshots=quote_rows,
        output_dir=output_dir,
        config=config,
        decision_at_utc=decision_at_utc,
        settled_at_utc=settled_at_utc,
        data_mode=data_mode,
        final_results=result_rows,
        quote_missing_reason=quote_missing_reason,
        quote_issues_by_game_pk=quote_issues_by_game_pk,
        source_operations=source_operations,
    )


__all__ = (
    "DATA_MODE_FIXTURE",
    "DATA_MODE_PROSPECTIVE",
    "DECISION_BET",
    "DECISION_NO_BET",
    "PAPER_POLICY_VERSION",
    "PAPER_REPORT_RELATIVE_PATH",
    "PaperPolicyConfig",
    "FrozenPredictionRun",
    "format_utc",
    "load_frozen_prediction_run",
    "load_quote_snapshots",
    "now_utc",
    "run_moneyline_paper_decision",
    "run_moneyline_paper_decision_from_paths",
)
