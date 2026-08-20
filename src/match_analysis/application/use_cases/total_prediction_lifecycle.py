"""P62A Total prospective prediction lifecycle and freeze authority.

Implements the prediction-only vertical slice for the Total market:
- Raw P226 Poisson model execution against point-in-time rate state.
- Explicit total_line input and discrete Poisson Over/Under/Push probabilities.
- Pre-first-pitch prediction freeze with immutable fingerprinting.
- Separation from Moneyline and Run Line prediction runtime state.
- Authoritative FINAL result attachment and evaluation without recomputing predictions.
- Separate accounting for push outcomes (excluded from binary metric denominators).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc
from ...baseball.domain.run_line_model import (
    DEFAULT_LEAGUE_AVG_RUNS,
    FROZEN_HOME_ADVANTAGE,
    POISSON_MAX_RUNS,
    RUN_SMOOTH_K,
    PriorGameRecord,
    compute_poisson_lambdas,
    compute_team_rate_state,
)
from ...baseball.domain.total_evaluation import (
    TotalEvaluationResult,
    TotalMetricsSummary,
    compute_total_metrics,
    evaluate_total_prediction,
)
from ...baseball.domain.total_model import (
    MARKET_ID,
    MODEL_ID,
    MODEL_VERSION,
    TotalProbabilities,
    compute_total_probabilities,
)


P62A_RUN_MANIFEST_SCHEMA = "p62a.total_run_manifest.v1"
P62A_PREDICTION_RECORD_SCHEMA = "p62a.total_prediction_record.v1"
P62A_SETTLEMENT_RECEIPT_SCHEMA = "p62a.total_settlement_receipt.v1"
P62A_SETTLEMENT_SUMMARY_SCHEMA = "p62a.total_settlement_summary.v1"

CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION = "PROSPECTIVE_FORWARD_PREDICTION"
CLASSIFICATION_HISTORICAL_REHEARSAL = "HISTORICAL_REHEARSAL"

STATE_FROZEN = "FROZEN"
STATE_SETTLED = "SETTLED"

P62A_CLAIMS = {
    "real_betting": False,
    "odds_or_market_dependent": False,
    "tsl_access": False,
    "ev_or_bet_pass_selection": False,
    "model_retraining": False,
    "calibration_fitting": False,
    "model_promotion": False,
    "threshold_optimization": False,
    "kelly": False,
    "market_id": "total",
}


@dataclass(frozen=True, slots=True)
class TotalPregameInputItem:
    """Normalized pregame contest input for Total prediction."""

    provider_game_id: str
    home_team: str
    away_team: str
    scheduled_start_utc: str
    as_of_utc: str
    total_line: float
    provider_namespace: str = "MLB_STATS_API"
    game_number: int = 1
    game_pk: int | None = None
    season: int = 2026


@dataclass(frozen=True, slots=True)
class TotalFrozenPrediction:
    """Immutable frozen pregame Total prediction record."""

    workflow_prediction_id: str
    prediction_fingerprint: str
    provider_namespace: str
    provider_game_id: str
    game_number: int
    game_pk: int | None
    home_team: str
    away_team: str
    scheduled_start_utc: str
    as_of_utc: str
    market_id: str
    total_line: float
    model_id: str
    model_version: str
    lambda_home: float
    lambda_away: float
    p_over: float
    p_under: float
    p_push: float
    predicted_side: str
    selected_total_probability: float
    schema_version: str = P62A_PREDICTION_RECORD_SCHEMA

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow_prediction_id": self.workflow_prediction_id,
            "prediction_fingerprint": self.prediction_fingerprint,
            "provider_namespace": self.provider_namespace,
            "provider_game_id": self.provider_game_id,
            "game_number": self.game_number,
            "game_pk": self.game_pk,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "scheduled_start_utc": self.scheduled_start_utc,
            "as_of_utc": self.as_of_utc,
            "market_id": self.market_id,
            "total_line": self.total_line,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "lambda_home": self.lambda_home,
            "lambda_away": self.lambda_away,
            "p_over": self.p_over,
            "p_under": self.p_under,
            "p_push": self.p_push,
            "predicted_side": self.predicted_side,
            "selected_total_probability": self.selected_total_probability,
        }


def compute_prediction_fingerprint(
    provider_game_id: str,
    home_team: str,
    away_team: str,
    scheduled_start_utc: str,
    as_of_utc: str,
    total_line: float,
    model_id: str,
    model_version: str,
    p_over: float,
    p_under: float,
    p_push: float,
    predicted_side: str,
) -> str:
    """Compute deterministic SHA-256 fingerprint for a Total prediction."""

    payload = (
        f"{provider_game_id}:{home_team}:{away_team}:{scheduled_start_utc}:"
        f"{as_of_utc}:{total_line:.6f}:{model_id}:{model_version}:"
        f"{p_over:.8f}:{p_under:.8f}:{p_push:.8f}:{predicted_side}"
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def freeze_total_predictions(
    pregame_inputs: Sequence[TotalPregameInputItem],
    history_records: Sequence[PriorGameRecord],
    *,
    run_id: str,
    classification: str = CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    output_dir: Path | None = None,
) -> tuple[tuple[TotalFrozenPrediction, ...], dict[str, Any]]:
    """Generate and freeze immutable Total predictions from pregame input and prior history."""

    if not pregame_inputs:
        raise ValueError("pregame_inputs must not be empty")

    predictions: list[TotalFrozenPrediction] = []

    for item in pregame_inputs:
        as_of_dt = parse_canonical_utc(item.as_of_utc)
        scheduled_start_dt = parse_canonical_utc(item.scheduled_start_utc)

        if as_of_dt >= scheduled_start_dt:
            raise ValueError(
                f"as_of_utc ({item.as_of_utc}) must be strictly before "
                f"scheduled_start_utc ({item.scheduled_start_utc}) for game {item.provider_game_id}"
            )

        team_states, league_avg = compute_team_rate_state(history_records, as_of_dt)
        lambda_home, lambda_away = compute_poisson_lambdas(
            item.home_team,
            item.away_team,
            team_states,
            league_avg,
        )
        probs = compute_total_probabilities(
            lambda_home,
            lambda_away,
            item.total_line,
        )

        fp = compute_prediction_fingerprint(
            provider_game_id=item.provider_game_id,
            home_team=item.home_team,
            away_team=item.away_team,
            scheduled_start_utc=item.scheduled_start_utc,
            as_of_utc=item.as_of_utc,
            total_line=item.total_line,
            model_id=MODEL_ID,
            model_version=MODEL_VERSION,
            p_over=probs.p_over,
            p_under=probs.p_under,
            p_push=probs.p_push,
            predicted_side=probs.predicted_side,
        )
        pred_id = f"p62a_pred_{fp[:32]}"

        predictions.append(
            TotalFrozenPrediction(
                workflow_prediction_id=pred_id,
                prediction_fingerprint=fp,
                provider_namespace=item.provider_namespace,
                provider_game_id=item.provider_game_id,
                game_number=item.game_number,
                game_pk=item.game_pk,
                home_team=item.home_team,
                away_team=item.away_team,
                scheduled_start_utc=item.scheduled_start_utc,
                as_of_utc=item.as_of_utc,
                market_id=MARKET_ID,
                total_line=float(item.total_line),
                model_id=MODEL_ID,
                model_version=MODEL_VERSION,
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                p_over=probs.p_over,
                p_under=probs.p_under,
                p_push=probs.p_push,
                predicted_side=probs.predicted_side,
                selected_total_probability=probs.selected_total_probability,
            )
        )

    predictions_tuple = tuple(predictions)
    manifest = {
        "schema_version": P62A_RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "run_classification": classification,
        "lifecycle_state": STATE_FROZEN,
        "created_at_utc": format_canonical_utc(datetime.now(timezone.utc)),
        "prediction_count": len(predictions_tuple),
        "claims": P62A_CLAIMS,
        "model_constants": {
            "run_smooth_k": RUN_SMOOTH_K,
            "default_league_avg_runs": DEFAULT_LEAGUE_AVG_RUNS,
            "poisson_max_runs": POISSON_MAX_RUNS,
            "frozen_home_advantage": FROZEN_HOME_ADVANTAGE,
        },
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "predictions.jsonl").write_text(
            "".join(
                json.dumps(p.to_payload(), sort_keys=True) + "\n"
                for p in predictions_tuple
            ),
            encoding="utf-8",
        )

    return predictions_tuple, manifest


def settle_total_predictions(
    frozen_predictions: Sequence[TotalFrozenPrediction],
    final_results: Mapping[str, tuple[int, int]],
    *,
    run_id: str,
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach authoritative FINAL results to frozen predictions and compute metrics."""

    settled_rows: list[dict[str, Any]] = []
    evaluations: list[TotalEvaluationResult] = []

    for pred in frozen_predictions:
        key = pred.provider_game_id
        if key not in final_results:
            continue

        home_score, away_score = final_results[key]
        eval_result = evaluate_total_prediction(
            predicted_side=pred.predicted_side,
            p_over=pred.p_over,
            p_under=pred.p_under,
            total_line=pred.total_line,
            home_score=home_score,
            away_score=away_score,
        )
        evaluations.append(eval_result)

        settled_rows.append(
            {
                "schema_version": P62A_PREDICTION_RECORD_SCHEMA,
                "workflow_prediction_id": pred.workflow_prediction_id,
                "prediction_fingerprint": pred.prediction_fingerprint,
                "provider_game_id": pred.provider_game_id,
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "scheduled_start_utc": pred.scheduled_start_utc,
                "as_of_utc": pred.as_of_utc,
                "market_id": pred.market_id,
                "total_line": pred.total_line,
                "predicted_side": pred.predicted_side,
                "p_over": pred.p_over,
                "p_under": pred.p_under,
                "p_push": pred.p_push,
                "final_scores": {
                    "home_score": home_score,
                    "away_score": away_score,
                },
                "evaluation": {
                    "actual_result": eval_result.actual_result,
                    "is_push": eval_result.is_push,
                    "is_correct": eval_result.is_correct,
                    "correctness_target": eval_result.correctness_target,
                    "brier_component": eval_result.brier_component,
                    "log_loss_component": eval_result.log_loss_component,
                },
            }
        )

    metrics = compute_total_metrics(evaluations)
    summary = {
        "schema_version": P62A_SETTLEMENT_SUMMARY_SCHEMA,
        "run_id": run_id,
        "lifecycle_state": STATE_SETTLED,
        "settled_at_utc": format_canonical_utc(datetime.now(timezone.utc)),
        "total_count": metrics.total_count,
        "push_count": metrics.push_count,
        "decided_count": metrics.decided_count,
        "correct_count": metrics.correct_count,
        "incorrect_count": metrics.incorrect_count,
        "accuracy": metrics.accuracy,
        "brier_score": metrics.brier_score,
        "log_loss": metrics.log_loss,
        "claims": P62A_CLAIMS,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "settled_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in settled_rows),
            encoding="utf-8",
        )
        (output_dir / "settlement_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return settled_rows, summary
