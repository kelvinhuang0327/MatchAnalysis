"""P61B Run Line prospective prediction lifecycle and freeze authority.

Implements the prediction-only vertical slice for the Run Line market:
- Raw P226 Poisson model execution against point-in-time rate state.
- Explicit spread_home input and discrete Poisson cover probabilities.
- Pre-first-pitch prediction freeze with immutable fingerprinting.
- Separation from Moneyline prediction runtime state.
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
from ...baseball.domain.run_line_evaluation import (
    RunLineEvaluationResult,
    RunLineMetricsSummary,
    compute_run_line_metrics,
    evaluate_run_line_prediction,
)
from ...baseball.domain.run_line_model import (
    DEFAULT_LEAGUE_AVG_RUNS,
    FROZEN_HOME_ADVANTAGE,
    MARKET_ID,
    MODEL_ID,
    MODEL_VERSION,
    POISSON_MAX_RUNS,
    RUN_SMOOTH_K,
    PriorGameRecord,
    RunLineProbabilities,
    compute_poisson_lambdas,
    compute_run_line_probabilities,
    compute_team_rate_state,
)


P61B_RUN_MANIFEST_SCHEMA = "p61b.run_line_run_manifest.v1"
P61B_PREDICTION_RECORD_SCHEMA = "p61b.run_line_prediction_record.v1"
P61B_SETTLEMENT_RECEIPT_SCHEMA = "p61b.run_line_settlement_receipt.v1"
P61B_SETTLEMENT_SUMMARY_SCHEMA = "p61b.run_line_settlement_summary.v1"

CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION = "PROSPECTIVE_FORWARD_PREDICTION"
CLASSIFICATION_HISTORICAL_REHEARSAL = "HISTORICAL_REHEARSAL"

STATE_FROZEN = "FROZEN"
STATE_SETTLED = "SETTLED"

P61B_CLAIMS = {
    "real_betting": False,
    "odds_or_market_dependent": False,
    "tsl_access": False,
    "ev_or_bet_pass_selection": False,
    "model_retraining": False,
    "calibration_fitting": False,
    "model_promotion": False,
    "threshold_optimization": False,
    "kelly": False,
    "market_id": "run_line",
}


@dataclass(frozen=True, slots=True)
class RunLinePregameInputItem:
    """Normalized pregame contest input for Run Line prediction."""

    provider_game_id: str
    home_team: str
    away_team: str
    scheduled_start_utc: str
    as_of_utc: str
    spread_home: float
    provider_namespace: str = "MLB_STATS_API"
    game_number: int = 1
    game_pk: int | None = None
    season: int = 2026


@dataclass(frozen=True, slots=True)
class RunLineFrozenPrediction:
    """Immutable frozen pregame Run Line prediction record."""

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
    spread_home: float
    model_id: str
    model_version: str
    lambda_home: float
    lambda_away: float
    p_home_covers: float
    p_away_covers: float
    p_push: float
    predicted_side: str
    selected_cover_probability: float
    schema_version: str = P61B_PREDICTION_RECORD_SCHEMA

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
            "spread_home": self.spread_home,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "lambda_home": self.lambda_home,
            "lambda_away": self.lambda_away,
            "p_home_covers": self.p_home_covers,
            "p_away_covers": self.p_away_covers,
            "p_push": self.p_push,
            "predicted_side": self.predicted_side,
            "selected_cover_probability": self.selected_cover_probability,
        }


def compute_prediction_fingerprint(
    provider_game_id: str,
    home_team: str,
    away_team: str,
    scheduled_start_utc: str,
    as_of_utc: str,
    spread_home: float,
    model_id: str,
    model_version: str,
    p_home_covers: float,
    p_away_covers: float,
    p_push: float,
    predicted_side: str,
) -> str:
    """Compute deterministic SHA-256 fingerprint for a Run Line prediction."""

    payload = (
        f"{provider_game_id}:{home_team}:{away_team}:{scheduled_start_utc}:"
        f"{as_of_utc}:{spread_home:.6f}:{model_id}:{model_version}:"
        f"{p_home_covers:.8f}:{p_away_covers:.8f}:{p_push:.8f}:{predicted_side}"
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def freeze_run_line_predictions(
    pregame_inputs: Sequence[RunLinePregameInputItem],
    history_records: Sequence[PriorGameRecord],
    *,
    run_id: str,
    classification: str = CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    output_dir: Path | None = None,
) -> tuple[tuple[RunLineFrozenPrediction, ...], dict[str, Any]]:
    """Generate and freeze immutable Run Line predictions from pregame input and prior history."""

    if not pregame_inputs:
        raise ValueError("pregame_inputs must not be empty")

    predictions: list[RunLineFrozenPrediction] = []

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
        probs = compute_run_line_probabilities(
            lambda_home,
            lambda_away,
            item.spread_home,
        )

        fp = compute_prediction_fingerprint(
            provider_game_id=item.provider_game_id,
            home_team=item.home_team,
            away_team=item.away_team,
            scheduled_start_utc=item.scheduled_start_utc,
            as_of_utc=item.as_of_utc,
            spread_home=item.spread_home,
            model_id=MODEL_ID,
            model_version=MODEL_VERSION,
            p_home_covers=probs.p_home_covers,
            p_away_covers=probs.p_away_covers,
            p_push=probs.p_push,
            predicted_side=probs.predicted_side,
        )
        pred_id = f"p61b_pred_{fp[:32]}"

        predictions.append(
            RunLineFrozenPrediction(
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
                spread_home=float(item.spread_home),
                model_id=MODEL_ID,
                model_version=MODEL_VERSION,
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                p_home_covers=probs.p_home_covers,
                p_away_covers=probs.p_away_covers,
                p_push=probs.p_push,
                predicted_side=probs.predicted_side,
                selected_cover_probability=probs.selected_cover_probability,
            )
        )

    predictions_tuple = tuple(predictions)
    manifest = {
        "schema_version": P61B_RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "run_classification": classification,
        "lifecycle_state": STATE_FROZEN,
        "created_at_utc": format_canonical_utc(datetime.now(timezone.utc)),
        "prediction_count": len(predictions_tuple),
        "claims": P61B_CLAIMS,
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


def settle_run_line_predictions(
    frozen_predictions: Sequence[RunLineFrozenPrediction],
    final_results: Mapping[str, tuple[int, int]],
    *,
    run_id: str,
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach authoritative FINAL results to frozen predictions and compute metrics."""

    settled_rows: list[dict[str, Any]] = []
    evaluations: list[RunLineEvaluationResult] = []

    for pred in frozen_predictions:
        key = pred.provider_game_id
        if key not in final_results:
            continue

        home_score, away_score = final_results[key]
        eval_result = evaluate_run_line_prediction(
            predicted_side=pred.predicted_side,
            p_home_covers=pred.p_home_covers,
            p_away_covers=pred.p_away_covers,
            spread_home=pred.spread_home,
            home_score=home_score,
            away_score=away_score,
        )
        evaluations.append(eval_result)

        settled_rows.append(
            {
                "schema_version": P61B_PREDICTION_RECORD_SCHEMA,
                "workflow_prediction_id": pred.workflow_prediction_id,
                "prediction_fingerprint": pred.prediction_fingerprint,
                "provider_game_id": pred.provider_game_id,
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "scheduled_start_utc": pred.scheduled_start_utc,
                "as_of_utc": pred.as_of_utc,
                "market_id": pred.market_id,
                "spread_home": pred.spread_home,
                "predicted_side": pred.predicted_side,
                "p_home_covers": pred.p_home_covers,
                "p_away_covers": pred.p_away_covers,
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

    metrics = compute_run_line_metrics(evaluations)
    summary = {
        "schema_version": P61B_SETTLEMENT_SUMMARY_SCHEMA,
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
        "claims": P61B_CLAIMS,
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
