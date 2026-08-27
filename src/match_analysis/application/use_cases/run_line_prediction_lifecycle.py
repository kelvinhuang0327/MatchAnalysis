"""P61B / P64B Run Line prospective prediction lifecycle and forward evidence integration.

Implements the prediction-only vertical slice for the Run Line market:
- Raw P226 Poisson model execution against point-in-time rate state.
- Explicit spread_home input and discrete Poisson cover probabilities.
- Standard MLB Run Line canonical-forward default: spread_home = -1.5.
- Pre-first-pitch prediction freeze with immutable fingerprinting.
- Integration with canonical P50C prospective prediction runs root and ledger contract.
- Separation from Moneyline prediction runtime state while sharing the forward ledger.
- Authoritative FINAL result attachment and evaluation without recomputing predictions.
- Separate accounting for push outcomes (excluded from binary metric denominators).
- Append-only prediction-forward ledger accounting and cumulative summary maintenance.
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
from .p44a_normalized_workflow_input import (
    NormalizedResultRecord,
    load_normalized_result_input,
)
from .p50c_prediction_run_ledger import (
    CLASSIFICATION_HISTORICAL_REHEARSAL,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P50C_FORWARD_SUMMARY_SCHEMA,
    P50C_LEDGER_RECORD_SCHEMA,
    P50C_REPORT_RELATIVE_PATH,
    P50C_RUN_MANIFEST_SCHEMA,
    P50C_SETTLEMENT_SUMMARY_SCHEMA,
    STATE_FROZEN,
    STATE_PARTIALLY_SETTLED,
    STATE_SETTLED,
    compute_forward_prediction_summary,
    load_effective_forward_records,
    read_json_object,
    read_jsonl_objects,
    reject_pregame_contamination,
    require_canonical_official_date,
)


P61B_RUN_MANIFEST_SCHEMA = "p61b.run_line_run_manifest.v1"
P61B_PREDICTION_RECORD_SCHEMA = "p61b.run_line_prediction_record.v1"
P61B_SETTLEMENT_RECEIPT_SCHEMA = "p61b.run_line_settlement_receipt.v1"
P61B_SETTLEMENT_SUMMARY_SCHEMA = "p61b.run_line_settlement_summary.v1"

DEFAULT_CANONICAL_SPREAD_HOME: float = -1.5

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


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _sha256_projection(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class RunLinePregameInputItem:
    """Normalized pregame contest input for Run Line prediction."""

    provider_game_id: str
    home_team: str
    away_team: str
    scheduled_start_utc: str
    official_date: str
    as_of_utc: str
    spread_home: float = DEFAULT_CANONICAL_SPREAD_HOME
    provider_namespace: str = "MLB_STATS_API"
    game_number: int = 1
    game_pk: int | None = None
    season: int = 2026

    @property
    def line_value(self) -> float:
        return self.spread_home


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
    official_date: str
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

    @property
    def line_value(self) -> float:
        return self.spread_home

    def to_payload(self) -> dict[str, Any]:
        payload = {
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
            "official_date": self.official_date,
            "as_of_utc": self.as_of_utc,
            "market_id": self.market_id,
            "spread_home": self.spread_home,
            "line_value": self.line_value,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "lambda_home": self.lambda_home,
            "lambda_away": self.lambda_away,
            "p_home_covers": self.p_home_covers,
            "p_away_covers": self.p_away_covers,
            "p_push": self.p_push,
            "predicted_side": self.predicted_side,
            "selected_cover_probability": self.selected_cover_probability,
            "game_identity": {
                "provider_namespace": self.provider_namespace,
                "provider_game_id": self.provider_game_id,
                "game_pk": self.game_pk,
                "game_number": self.game_number,
                "scheduled_start_utc": self.scheduled_start_utc,
                "official_date": self.official_date,
                "home_team": self.home_team,
                "away_team": self.away_team,
                "home_team_code": self.home_team,
                "away_team_code": self.away_team,
            },
            "model_identity": {
                "model_role": "CHAMPION",
                "model_id": self.model_id,
                "model_version": self.model_version,
                "market_id": self.market_id,
                "spread_home": self.spread_home,
                "line_value": self.line_value,
                "lambda_home": self.lambda_home,
                "lambda_away": self.lambda_away,
                "p_home_covers": self.p_home_covers,
                "p_away_covers": self.p_away_covers,
                "p_push": self.p_push,
                "selection": self.predicted_side,
                "predicted_side": self.predicted_side,
                "confidence_probability": str(self.selected_cover_probability),
                "model_probability": str(self.selected_cover_probability),
            },
        }
        reject_pregame_contamination(payload)
        return payload


def compute_prediction_fingerprint(
    provider_game_id: str,
    home_team: str,
    away_team: str,
    scheduled_start_utc: str,
    official_date: str,
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
        f"{official_date}:{as_of_utc}:{spread_home:.6f}:{model_id}:{model_version}:"
        f"{p_home_covers:.8f}:{p_away_covers:.8f}:{p_push:.8f}:{predicted_side}"
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def freeze_run_line_predictions(
    pregame_inputs: Sequence[RunLinePregameInputItem],
    history_records: Sequence[PriorGameRecord],
    *,
    run_id: str,
    task_id: str = "P64B",
    classification: str = CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    output_dir: Path | None = None,
    repository_root: str | Path | None = None,
    runs_root: str | Path | None = None,
    created_at_utc: str | None = None,
    target_date: str | None = None,
) -> tuple[tuple[RunLineFrozenPrediction, ...], dict[str, Any]]:
    """Generate and freeze immutable Run Line predictions from pregame input and prior history."""

    if not pregame_inputs:
        raise ValueError("pregame_inputs must not be empty")

    input_official_dates = [
        require_canonical_official_date(
            item.official_date,
            field_name=f"Run Line game {item.provider_game_id} official_date",
        )
        for item in pregame_inputs
    ]
    unique_official_dates = sorted(set(input_official_dates))
    if target_date is not None:
        resolved_target_date = require_canonical_official_date(target_date, field_name="target_date")
        if unique_official_dates != [resolved_target_date]:
            raise ValueError(
                "P61B_TARGET_DATE_OFFICIAL_DATE_MISMATCH: target_date must equal every prediction official_date"
            )
    elif len(unique_official_dates) == 1:
        resolved_target_date = unique_official_dates[0]
    else:
        raise ValueError(
            "P61B_TARGET_DATE_REQUIRED: a Run Line daily run must contain one canonical official_date"
        )

    repo_root = Path(repository_root).resolve() if repository_root is not None else None
    if output_dir is not None:
        target_dir = Path(output_dir).resolve()
    elif runs_root is not None:
        target_dir = (Path(runs_root).resolve() / run_id).resolve()
    elif repo_root is not None:
        target_dir = (repo_root / P50C_REPORT_RELATIVE_PATH / "runs" / run_id).resolve()
    else:
        target_dir = None

    if created_at_utc is not None:
        creation_time = created_at_utc
        parse_canonical_utc(creation_time)
    else:
        creation_time = min(item.as_of_utc for item in pregame_inputs)

    predictions: list[RunLineFrozenPrediction] = []

    for item in pregame_inputs:
        as_of_dt = parse_canonical_utc(item.as_of_utc)
        scheduled_start_dt = parse_canonical_utc(item.scheduled_start_utc)

        if as_of_dt >= scheduled_start_dt:
            raise ValueError(
                f"as_of_utc ({item.as_of_utc}) must be strictly before "
                f"scheduled_start_utc ({item.scheduled_start_utc}) for game {item.provider_game_id}"
            )

        if created_at_utc is not None and classification == CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION:
            creation_dt = parse_canonical_utc(creation_time)
            if creation_dt >= scheduled_start_dt:
                raise RuntimeError(
                    "P50C_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID: "
                    f"prospective prediction run creation time {creation_time} is not strictly before "
                    f"earliest scheduled start {item.scheduled_start_utc}"
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
            official_date=item.official_date,
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

        frozen_pred = RunLineFrozenPrediction(
            workflow_prediction_id=pred_id,
            prediction_fingerprint=fp,
            provider_namespace=item.provider_namespace,
            provider_game_id=item.provider_game_id,
            game_number=item.game_number,
            game_pk=item.game_pk,
            home_team=item.home_team,
            away_team=item.away_team,
            scheduled_start_utc=item.scheduled_start_utc,
            official_date=item.official_date,
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
        # Verify zero result/betting contamination
        reject_pregame_contamination(frozen_pred.to_payload())
        predictions.append(frozen_pred)

    predictions_tuple = tuple(predictions)
    pred_fps = [p.prediction_fingerprint for p in predictions_tuple]
    pred_bundle_fp = _sha256_projection(pred_fps)

    manifest = {
        "schema_version": P61B_RUN_MANIFEST_SCHEMA,
        "task_id": task_id,
        "run_id": run_id,
        "run_classification": classification,
        "lifecycle_state": STATE_FROZEN,
        "created_at_utc": creation_time,
        "target_date": resolved_target_date,
        "model_role": "CHAMPION",
        "market_id": MARKET_ID,
        "prediction_count": len(predictions_tuple),
        "eligible_prediction_count": len(predictions_tuple),
        "settled_total_count": 0,
        "pending_count": len(predictions_tuple),
        "exclusion_count": 0,
        "prediction_bundle_fingerprint": pred_bundle_fp,
        "prediction_fingerprints": pred_fps,
        "claims": dict(P61B_CLAIMS),
        "model_constants": {
            "run_smooth_k": RUN_SMOOTH_K,
            "default_league_avg_runs": DEFAULT_LEAGUE_AVG_RUNS,
            "poisson_max_runs": POISSON_MAX_RUNS,
            "frozen_home_advantage": FROZEN_HOME_ADVANTAGE,
        },
    }

    if target_dir is not None:
        manifest_path = target_dir / "run_manifest.json"
        predictions_path = target_dir / "predictions.jsonl"
        frozen_predictions_path = target_dir / "frozen_predictions.jsonl"

        # Idempotency and conflict check
        if manifest_path.is_file():
            existing_manifest = read_json_object(manifest_path)
            if (
                existing_manifest.get("run_id") == run_id
                and existing_manifest.get("prediction_bundle_fingerprint") == pred_bundle_fp
                and existing_manifest.get("run_classification") == classification
            ):
                return predictions_tuple, existing_manifest
            raise RuntimeError(
                "P50C_RUN_AUTHORITY_CONFLICT: existing Run Line run manifest conflicts with incoming authority"
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(_json_bytes(manifest))
        predictions_content = b"".join(
            _canonical_json_bytes(p.to_payload()) for p in predictions_tuple
        )
        predictions_path.write_bytes(predictions_content)
        frozen_predictions_path.write_bytes(predictions_content)

    return predictions_tuple, manifest


def _parse_run_line_frozen_dict(row: Mapping[str, Any]) -> RunLineFrozenPrediction:
    game_id = row.get("game_identity") or row
    model_id_info = row.get("model_identity") or row
    spread_home = float(row.get("spread_home", row.get("line_value", DEFAULT_CANONICAL_SPREAD_HOME)))
    return RunLineFrozenPrediction(
        workflow_prediction_id=str(row.get("workflow_prediction_id", "")),
        prediction_fingerprint=str(row.get("prediction_fingerprint", "")),
        provider_namespace=str(game_id.get("provider_namespace", "MLB_STATS_API")),
        provider_game_id=str(game_id.get("provider_game_id", row.get("provider_game_id", ""))),
        game_number=int(game_id.get("game_number", row.get("game_number", 1))),
        game_pk=int(game_id["game_pk"]) if game_id.get("game_pk") is not None else None,
        home_team=str(game_id.get("home_team", row.get("home_team", "HOME"))),
        away_team=str(game_id.get("away_team", row.get("away_team", "AWAY"))),
        scheduled_start_utc=str(game_id.get("scheduled_start_utc", row.get("scheduled_start_utc", ""))),
        official_date=require_canonical_official_date(
            game_id.get("official_date", row.get("official_date")),
            field_name="Run Line frozen prediction official_date",
        ),
        as_of_utc=str(row.get("as_of_utc", "")),
        market_id=str(row.get("market_id", MARKET_ID)),
        spread_home=spread_home,
        model_id=str(model_id_info.get("model_id", MODEL_ID)),
        model_version=str(model_id_info.get("model_version", MODEL_VERSION)),
        lambda_home=float(model_id_info.get("lambda_home", 0.0)),
        lambda_away=float(model_id_info.get("lambda_away", 0.0)),
        p_home_covers=float(model_id_info.get("p_home_covers", 0.5)),
        p_away_covers=float(model_id_info.get("p_away_covers", 0.5)),
        p_push=float(model_id_info.get("p_push", 0.0)),
        predicted_side=str(model_id_info.get("predicted_side", row.get("predicted_side", "HOME"))),
        selected_cover_probability=float(
            model_id_info.get("selected_cover_probability", row.get("selected_cover_probability", 0.5))
        ),
    )


def settle_run_line_predictions(
    frozen_predictions: Sequence[RunLineFrozenPrediction] | None = None,
    final_results: Mapping[str, tuple[int, int]] | Sequence[NormalizedResultRecord] | str | Path | None = None,
    *,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    output_dir: Path | None = None,
    repository_root: str | Path | None = None,
    runs_root: str | Path | None = None,
    ledger_root: str | Path | None = None,
    settled_at_utc: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach authoritative FINAL results to frozen Run Line predictions, evaluate, and update P50C ledger."""

    repo_root = Path(repository_root).resolve() if repository_root is not None else None
    settlement_time = settled_at_utc or format_canonical_utc(datetime.now(timezone.utc))

    # Resolve run directory
    resolved_run_dir: Path | None = None
    if run_dir is not None:
        resolved_run_dir = Path(run_dir).resolve()
    elif runs_root is not None and run_id is not None:
        resolved_run_dir = (Path(runs_root).resolve() / run_id).resolve()
    elif repo_root is not None and run_id is not None:
        resolved_run_dir = (repo_root / P50C_REPORT_RELATIVE_PATH / "runs" / run_id).resolve()
    elif output_dir is not None:
        resolved_run_dir = Path(output_dir).resolve()

    manifest: dict[str, Any] = {}
    actual_run_id = run_id or ""
    run_classification = CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION

    if resolved_run_dir is not None:
        manifest_path = resolved_run_dir / "run_manifest.json"
        if manifest_path.is_file():
            manifest = read_json_object(manifest_path)
            actual_run_id = manifest.get("run_id", actual_run_id)
            run_classification = manifest.get("run_classification", run_classification)

    # Load frozen predictions
    actual_predictions: list[RunLineFrozenPrediction] = []
    if frozen_predictions is not None:
        actual_predictions = list(frozen_predictions)
    elif resolved_run_dir is not None:
        pred_path = resolved_run_dir / "frozen_predictions.jsonl"
        if not pred_path.is_file():
            pred_path = resolved_run_dir / "predictions.jsonl"
        if not pred_path.is_file():
            raise FileNotFoundError(f"frozen predictions file missing in {resolved_run_dir}")
        raw_rows = read_jsonl_objects(pred_path)
        actual_predictions = [_parse_run_line_frozen_dict(r) for r in raw_rows]
    else:
        raise ValueError("either frozen_predictions or run_dir must be provided")

    if not actual_predictions:
        raise RuntimeError("P50C_MISSING_FROZEN_PREDICTIONS: run contains no frozen predictions")

    # Load results
    results_map: dict[str, tuple[int, int]] = {}
    result_sources: dict[str, str] = {}
    result_observed_at: dict[str, str] = {}

    if isinstance(final_results, (str, Path)):
        res_records = load_normalized_result_input(final_results)
        for r in res_records:
            if r.status != "FINAL":
                raise RuntimeError("P50C_NON_FINAL_RESULT_REJECTED: result status must be FINAL")
            key = r.provider_game_id
            results_map[key] = (r.home_score, r.away_score)
            result_sources[key] = r.source_identity
            result_observed_at[key] = r.result_observed_at_utc
    elif isinstance(final_results, Sequence) and final_results and isinstance(final_results[0], NormalizedResultRecord):
        for r in final_results:
            if r.status != "FINAL":
                raise RuntimeError("P50C_NON_FINAL_RESULT_REJECTED: result status must be FINAL")
            key = r.provider_game_id
            results_map[key] = (r.home_score, r.away_score)
            result_sources[key] = r.source_identity
            result_observed_at[key] = r.result_observed_at_utc
    elif isinstance(final_results, Mapping):
        for k, v in final_results.items():
            results_map[str(k)] = (int(v[0]), int(v[1]))
            result_sources[str(k)] = "OFFICIAL_RESULTS"
            result_observed_at[str(k)] = settlement_time

    # Read existing settlements if run_dir is known
    existing_settled_records: list[dict[str, Any]] = []
    if resolved_run_dir is not None:
        settlements_file = resolved_run_dir / "settled_predictions.jsonl"
        if settlements_file.is_file():
            existing_settled_records = read_jsonl_objects(settlements_file)

    existing_settled_by_fp = {
        row.get("prediction_fingerprint") or row.get("prediction_identity", {}).get("prediction_fingerprint"): row
        for row in existing_settled_records
    }

    settled_rows: list[dict[str, Any]] = []
    newly_settled_ledger_records: list[dict[str, Any]] = []
    evaluations: list[RunLineEvaluationResult] = []

    for pred in actual_predictions:
        key = pred.provider_game_id
        existing = existing_settled_by_fp.get(pred.prediction_fingerprint)

        if key not in results_map:
            if existing is not None:
                settled_rows.append(existing)
            continue

        home_score, away_score = results_map[key]

        # Idempotency and conflict check
        if existing is not None:
            prev_scores = existing.get("final_scores") or existing.get("final_result_authority") or {}
            prev_home = prev_scores.get("home_score")
            prev_away = prev_scores.get("away_score")
            if prev_home is not None and prev_away is not None:
                if home_score != prev_home or away_score != prev_away:
                    raise RuntimeError(
                        "P50C_CONFLICTING_RESULT_REJECTED: conflicting result authority for already-settled Run Line prediction"
                    )
            settled_rows.append(existing)
            continue

        eval_result = evaluate_run_line_prediction(
            predicted_side=pred.predicted_side,
            p_home_covers=pred.p_home_covers,
            p_away_covers=pred.p_away_covers,
            spread_home=pred.spread_home,
            home_score=home_score,
            away_score=away_score,
        )
        evaluations.append(eval_result)

        obs_time = result_observed_at.get(key, settlement_time)
        src_id = result_sources.get(key, "OFFICIAL_RESULTS")

        feedback_id = _sha256_projection(
            {
                "workflow_prediction_id": pred.workflow_prediction_id,
                "prediction_fingerprint": pred.prediction_fingerprint,
                "market_id": MARKET_ID,
                "spread_home": pred.spread_home,
                "actual_result": eval_result.actual_result,
                "is_push": eval_result.is_push,
                "is_correct": eval_result.is_correct,
            }
        )

        settled_row = {
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
            "line_value": pred.spread_home,
            "predicted_side": pred.predicted_side,
            "p_home_covers": pred.p_home_covers,
            "p_away_covers": pred.p_away_covers,
            "p_push": pred.p_push,
            "final_scores": {
                "home_score": home_score,
                "away_score": away_score,
            },
            "evaluation": {
                "market_id": pred.market_id,
                "spread_home": pred.spread_home,
                "line_value": pred.spread_home,
                "actual_result": eval_result.actual_result,
                "is_push": eval_result.is_push,
                "is_correct": eval_result.is_correct,
                "correctness_target": eval_result.correctness_target,
                "brier_component": eval_result.brier_component,
                "log_loss_component": eval_result.log_loss_component,
                "evaluated_at_utc": settlement_time,
            },
        }
        settled_rows.append(settled_row)

        # Build canonical P50C ledger record
        brier_str = str(eval_result.brier_component) if eval_result.brier_component is not None else None
        log_loss_str = str(eval_result.log_loss_component) if eval_result.log_loss_component is not None else None

        ledger_payload_without_id = {
            "schema_version": P50C_LEDGER_RECORD_SCHEMA,
            "run_id": actual_run_id,
            "run_classification": run_classification,
            "market_id": MARKET_ID,
            "line_value": pred.spread_home,
            "spread_home": pred.spread_home,
            "prediction_identity": {
                "workflow_prediction_id": pred.workflow_prediction_id,
                "prediction_fingerprint": pred.prediction_fingerprint,
                "market_id": MARKET_ID,
                "line_value": pred.spread_home,
            },
            "game_identity": {
                "provider_namespace": pred.provider_namespace,
                "provider_game_id": pred.provider_game_id,
                "game_pk": pred.game_pk,
                "game_number": pred.game_number,
                "scheduled_start_utc": pred.scheduled_start_utc,
                "official_date": pred.official_date,
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "home_team_code": pred.home_team,
                "away_team_code": pred.away_team,
            },
            "model_identity": {
                "model_role": "CHAMPION",
                "model_id": pred.model_id,
                "model_version": pred.model_version,
                "market_id": MARKET_ID,
                "spread_home": pred.spread_home,
                "line_value": pred.spread_home,
                "lambda_home": pred.lambda_home,
                "lambda_away": pred.lambda_away,
                "p_home_covers": pred.p_home_covers,
                "p_away_covers": pred.p_away_covers,
                "p_push": pred.p_push,
                "selection": pred.predicted_side,
                "predicted_side": pred.predicted_side,
                "model_probability": str(pred.selected_cover_probability),
            },
            "pregame_freeze_authority": {
                "created_at_utc": manifest.get("created_at_utc", settlement_time),
                "as_of_utc": pred.as_of_utc,
                "prediction_fingerprint": pred.prediction_fingerprint,
            },
            "final_result_authority": {
                "status": "FINAL",
                "home_score": home_score,
                "away_score": away_score,
                "actual_winner": eval_result.actual_result,
                "target_home_win": 1 if eval_result.actual_result == "HOME" else 0,
                "result_observed_at_utc": obs_time,
                "source_identity": src_id,
            },
            "evaluation": {
                "market_id": MARKET_ID,
                "spread_home": pred.spread_home,
                "line_value": pred.spread_home,
                "selection": pred.predicted_side,
                "predicted_side": pred.predicted_side,
                "actual_result": eval_result.actual_result,
                "actual_winner": eval_result.actual_result,
                "is_push": eval_result.is_push,
                "is_correct": eval_result.is_correct,
                "correctness_target": eval_result.correctness_target if eval_result.correctness_target is not None else 0,
                "brier_component": brier_str or "0.0",
                "log_loss_component": log_loss_str or "0.0",
                "evaluated_at_utc": settlement_time,
            },
            "feedback": {
                "feedback_identity": feedback_id,
            },
        }
        ledger_rec_id = _sha256_projection(ledger_payload_without_id)
        ledger_record = {
            "ledger_record_id": ledger_rec_id,
            **ledger_payload_without_id,
        }
        newly_settled_ledger_records.append(ledger_record)

    total_eligible = len(actual_predictions)
    total_settled = len(settled_rows)
    pending_count = total_eligible - total_settled

    if total_settled == 0:
        new_state = STATE_FROZEN
    elif total_settled < total_eligible:
        new_state = STATE_PARTIALLY_SETTLED
    else:
        new_state = STATE_SETTLED

    metrics = compute_run_line_metrics(evaluations)
    summary = {
        "schema_version": P61B_SETTLEMENT_SUMMARY_SCHEMA,
        "run_id": actual_run_id,
        "lifecycle_state": new_state,
        "settled_at_utc": settlement_time,
        "total_count": metrics.total_count,
        "push_count": metrics.push_count,
        "decided_count": metrics.decided_count,
        "correct_count": metrics.correct_count,
        "incorrect_count": metrics.incorrect_count,
        "accuracy": metrics.accuracy,
        "brier_score": metrics.brier_score,
        "log_loss": metrics.log_loss,
        "claims": dict(P61B_CLAIMS),
    }

    out_dir = output_dir or resolved_run_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "settled_predictions.jsonl").write_bytes(
            _jsonl_bytes(settled_rows)
        )
        (out_dir / "settlement_summary.json").write_bytes(
            _json_bytes(summary)
        )
        manifest_path = out_dir / "run_manifest.json"
        if manifest_path.is_file():
            curr_manifest = read_json_object(manifest_path)
            curr_manifest["lifecycle_state"] = new_state
            curr_manifest["settled_total_count"] = total_settled
            curr_manifest["pending_count"] = pending_count
            manifest_path.write_bytes(_json_bytes(curr_manifest))

    # Append to P50C ledger if ledger_root is provided
    resolved_ledger_root: Path | None = None
    if ledger_root is not None:
        resolved_ledger_root = Path(ledger_root).resolve()
    elif repo_root is not None:
        resolved_ledger_root = (repo_root / P50C_REPORT_RELATIVE_PATH / "ledger").resolve()

    if resolved_ledger_root is not None and newly_settled_ledger_records:
        resolved_ledger_root.mkdir(parents=True, exist_ok=True)
        if run_classification == CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION:
            target_ledger_file = resolved_ledger_root / "prediction_forward_ledger.jsonl"
        else:
            target_ledger_file = resolved_ledger_root / "rehearsal_prediction_ledger.jsonl"

        existing_ledger_rows = read_jsonl_objects(target_ledger_file)
        existing_ledger_keys = {
            (
                row.get("run_id"),
                row.get("prediction_identity", {}).get("prediction_fingerprint") or row.get("prediction_fingerprint"),
            ): row
            for row in existing_ledger_rows
        }

        rows_to_append: list[dict[str, Any]] = []
        for record in newly_settled_ledger_records:
            key = (
                record["run_id"],
                record["prediction_identity"]["prediction_fingerprint"],
            )
            if key in existing_ledger_keys:
                existing_row = existing_ledger_keys[key]
                if existing_row != record:
                    raise RuntimeError("P50C_CONFLICTING_RESULT_REJECTED: prediction ledger record conflict")
                continue
            rows_to_append.append(record)
            existing_ledger_keys[key] = record

        if rows_to_append:
            with target_ledger_file.open("ab") as f:
                f.write(_jsonl_bytes(rows_to_append))

        # Update forward prediction summary
        forward_rows = load_effective_forward_records(resolved_ledger_root)
        forward_summary = compute_forward_prediction_summary(forward_rows)
        (resolved_ledger_root / "forward_prediction_summary.json").write_bytes(_json_bytes(forward_summary))

    return settled_rows, summary
