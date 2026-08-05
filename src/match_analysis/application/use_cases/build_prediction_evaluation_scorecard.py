"""Build a deterministic prediction evaluation scorecard from P16A attachments and P15C snapshots.

Reads verified P16A final-result attachment rows, validates source evidence and snapshot
provenance, excludes REJECTED rows, and builds an immutable evaluation scorecard.
Does not construct PredictionSourceObservation or FinalResultObservation,
re-run admission or result attachment, calculate odds/ROI/EV, retrain models,
or touch external networks or databases.
"""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from typing import Any

from ...baseball.domain.prediction_evaluation import (
    EVALUATION_ROW_SCHEMA_VERSION,
    SCHEMA_VERSION,
    BreakdownMetrics,
    PredictionEvaluationRow,
    PredictionEvaluationScorecard,
    build_scorecard,
    compute_evaluation_row_fingerprint,
)


@dataclass(frozen=True, slots=True)
class PredictionEvaluationScorecardResult:
    """Immutable result of building the prediction evaluation scorecard."""

    schema_version: str
    source_attachments_sha256: str
    source_summary_sha256: str
    source_attachment_set_fingerprint: str
    source_snapshot_sha256: str
    source_snapshot_summary_sha256: str
    source_snapshot_fingerprint: str
    source_row_count: int
    source_attached_count: int
    source_rejected_count: int
    evaluation_row_count: int
    excluded_rejected_count: int
    correct_count: int
    incorrect_count: int
    accuracy: float
    mean_selected_side_probability: float
    brier_score: float
    scorecard: PredictionEvaluationScorecard
    evaluation_rows: tuple[PredictionEvaluationRow, ...]
    evaluation_set_fingerprint: str
    claims: dict[str, bool]


def _validate_json_no_duplicate_keys(raw_line: str, line_index: int) -> dict[str, Any]:
    """Parse JSON rejecting duplicate keys at all levels."""

    def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, int] = {}
        for key, _ in pairs:
            if key in seen:
                raise ValueError(
                    f"Duplicate JSON key {key!r} in row {line_index + 1}"
                )
            seen[key] = 1
        return dict(pairs)

    try:
        data = json.loads(raw_line, object_pairs_hook=_object_pairs_hook)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed JSON on line {line_index + 1}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Line {line_index + 1} must be a JSON object")
    return data


def _compute_attachment_row_fingerprint(row_dict: dict[str, Any]) -> str:
    """Compute deterministic fingerprint for an attachment row dict."""
    payload = {
        "actual_winner": row_dict.get("actual_winner"),
        "attachment_status": row_dict["attachment_status"],
        "away_score": row_dict.get("away_score"),
        "game_number": row_dict["game_number"],
        "home_score": row_dict.get("home_score"),
        "is_correct": row_dict.get("is_correct"),
        "prediction_observation_id": row_dict["prediction_observation_id"],
        "provider_game_id": row_dict["provider_game_id"],
        "provider_namespace": row_dict["provider_namespace"],
        "rejection_reason": row_dict.get("rejection_reason"),
        "result_observation_id": row_dict.get("result_observation_id"),
        "result_observed_at_utc": row_dict.get("result_observed_at_utc"),
        "scheduled_start_utc": row_dict["scheduled_start_utc"],
        "selection": row_dict["selection"],
        "source_snapshot_row_fingerprint": row_dict["source_snapshot_row_fingerprint"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_attachment_set_fingerprint(
    attachment_rows: list[dict[str, Any]],
) -> str:
    """Compute deterministic fingerprint over sorted attachment rows."""
    parts = []
    for row in sorted(attachment_rows, key=lambda r: r["prediction_observation_id"]):
        pred_id = row["prediction_observation_id"]
        row_fp = row["attachment_row_fingerprint"]
        parts.append(f"{pred_id}:{row_fp}\n")
    combined = "".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def build_prediction_evaluation_scorecard(
    *,
    attachments_bytes: bytes,
    attachment_summary_bytes: bytes,
    snapshot_bytes: bytes,
    snapshot_summary_bytes: bytes,
) -> PredictionEvaluationScorecardResult:
    """Build a deterministic prediction evaluation scorecard.

    Validates P16A attachment artifacts and P15C snapshot artifacts,
    verifies SHA-256 hashes and fingerprints, filters for supported ATTACHED rows,
    and returns an immutable scorecard result.

    Raises ValueError on structural failures, hash mismatches, or invalid row schemas.
    """
    attachments_sha256 = hashlib.sha256(attachments_bytes).hexdigest()
    attachment_summary_sha256 = hashlib.sha256(attachment_summary_bytes).hexdigest()
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot_summary_sha256 = hashlib.sha256(snapshot_summary_bytes).hexdigest()

    # Parse attachment summary
    attachment_summary_dict = _validate_json_no_duplicate_keys(
        attachment_summary_bytes.decode("utf-8"), 0
    )

    expected_attachments_sha256 = attachment_summary_dict.get("attachments_jsonl_sha256")
    if attachments_sha256 != expected_attachments_sha256:
        raise ValueError(
            f"Attachments SHA-256 mismatch: computed {attachments_sha256}, "
            f"summary expects {expected_attachments_sha256}"
        )

    expected_snapshot_sha256 = attachment_summary_dict.get("source_snapshot_sha256")
    if snapshot_sha256 != expected_snapshot_sha256:
        raise ValueError(
            f"Snapshot SHA-256 mismatch: computed {snapshot_sha256}, "
            f"attachment summary expects {expected_snapshot_sha256}"
        )

    expected_snapshot_summary_sha256 = attachment_summary_dict.get("source_snapshot_summary_sha256")
    if snapshot_summary_sha256 != expected_snapshot_summary_sha256:
        raise ValueError(
            f"Snapshot summary SHA-256 mismatch: computed {snapshot_summary_sha256}, "
            f"attachment summary expects {expected_snapshot_summary_sha256}"
        )

    source_attachment_set_fingerprint = attachment_summary_dict["attachment_set_fingerprint"]
    source_snapshot_fingerprint = attachment_summary_dict["source_snapshot_fingerprint"]

    # Parse snapshot summary
    snapshot_summary_dict = _validate_json_no_duplicate_keys(
        snapshot_summary_bytes.decode("utf-8"), 0
    )
    if snapshot_summary_dict.get("snapshot_fingerprint") != source_snapshot_fingerprint:
        raise ValueError(
            f"Snapshot fingerprint mismatch: snapshot summary has {snapshot_summary_dict.get('snapshot_fingerprint')}, "
            f"attachment summary expects {source_snapshot_fingerprint}"
        )

    # Parse snapshot JSONL
    snapshot_lines = [
        line for line in snapshot_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    observation_map: dict[str, dict[str, Any]] = {}
    for i, line in enumerate(snapshot_lines):
        snap_row = _validate_json_no_duplicate_keys(line, i)
        pred_obs_id = snap_row["prediction_observation_id"]
        if pred_obs_id in observation_map:
            raise ValueError(f"Duplicate prediction_observation_id in snapshot: {pred_obs_id}")
        observation_map[pred_obs_id] = snap_row["observation"]

    # Parse attachments JSONL
    attachment_lines = [
        line for line in attachments_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    attachment_rows: list[dict[str, Any]] = []
    seen_attachment_pred_ids: set[str] = set()
    for i, line in enumerate(attachment_lines):
        att_row = _validate_json_no_duplicate_keys(line, i)
        pred_id = att_row["prediction_observation_id"]
        if pred_id in seen_attachment_pred_ids:
            raise ValueError(f"Duplicate prediction_observation_id in attachments: {pred_id}")
        seen_attachment_pred_ids.add(pred_id)

        # Verify attachment row fingerprint
        computed_row_fp = _compute_attachment_row_fingerprint(att_row)
        if att_row.get("attachment_row_fingerprint") != computed_row_fp:
            raise ValueError(
                f"Attachment row fingerprint mismatch on line {i + 1}: "
                f"computed {computed_row_fp}, got {att_row.get('attachment_row_fingerprint')}"
            )
        attachment_rows.append(att_row)

    # Verify attachment set fingerprint
    computed_set_fp = _compute_attachment_set_fingerprint(attachment_rows)
    if computed_set_fp != source_attachment_set_fingerprint:
        raise ValueError(
            f"Attachment set fingerprint mismatch: computed {computed_set_fp}, "
            f"summary expects {source_attachment_set_fingerprint}"
        )

    source_row_count = len(attachment_rows)
    source_attached_count = sum(
        1 for r in attachment_rows if r["attachment_status"] == "ATTACHED"
    )
    source_rejected_count = sum(
        1 for r in attachment_rows if r["attachment_status"] == "REJECTED"
    )

    evaluation_rows_list: list[PredictionEvaluationRow] = []
    excluded_rejected_count = 0

    for att_row in attachment_rows:
        status = att_row["attachment_status"]
        if status == "REJECTED":
            excluded_rejected_count += 1
            continue

        if status != "ATTACHED":
            raise ValueError(f"Unknown attachment_status: {status}")

        if att_row.get("rejection_reason") is not None:
            raise ValueError("ATTACHED row must have rejection_reason == null")

        actual_winner = att_row.get("actual_winner")
        if actual_winner not in ("HOME", "AWAY"):
            raise ValueError(f"ATTACHED row actual_winner must be 'HOME' or 'AWAY', got {actual_winner!r}")

        is_correct = att_row.get("is_correct")
        if not isinstance(is_correct, bool):
            raise ValueError(f"ATTACHED row is_correct must be a boolean, got {is_correct!r}")

        selection = att_row.get("selection")
        if selection not in ("HOME", "AWAY"):
            raise ValueError(f"ATTACHED row selection must be 'HOME' or 'AWAY', got {selection!r}")

        pred_id = att_row["prediction_observation_id"]
        obs = observation_map.get(pred_id)
        if obs is None:
            raise ValueError(f"Missing snapshot observation for prediction {pred_id}")

        market_id = obs.get("market_id")
        if market_id != "moneyline":
            raise ValueError(f"Unsupported market_id: {market_id}")

        model_id = obs.get("model_id")
        if not model_id or not isinstance(model_id, str):
            raise ValueError(f"Invalid model_id in snapshot observation: {model_id!r}")

        prob_val = obs.get("model_probability")
        if prob_val is None:
            raise ValueError(f"Missing model_probability in snapshot observation for {pred_id}")

        model_probability = Decimal(str(prob_val))
        if not (Decimal("0") <= model_probability <= Decimal("1")):
            raise ValueError(f"model_probability out of bounds: {model_probability}")

        correctness_target = 1 if is_correct else 0
        brier_component = (model_probability - Decimal(correctness_target)) ** 2

        eval_fp = compute_evaluation_row_fingerprint(
            prediction_observation_id=pred_id,
            source_attachment_row_fingerprint=att_row["attachment_row_fingerprint"],
            model_id=model_id,
            market_id=market_id,
            selection=selection,
            provider_namespace=att_row["provider_namespace"],
            provider_game_id=att_row["provider_game_id"],
            game_number=att_row["game_number"],
            model_probability=model_probability,
            actual_winner=actual_winner,
            is_correct=is_correct,
            correctness_target=correctness_target,
            brier_component=brier_component,
        )

        eval_row = PredictionEvaluationRow(
            prediction_observation_id=pred_id,
            source_attachment_row_fingerprint=att_row["attachment_row_fingerprint"],
            model_id=model_id,
            market_id=market_id,
            selection=selection,
            provider_namespace=att_row["provider_namespace"],
            provider_game_id=att_row["provider_game_id"],
            game_number=att_row["game_number"],
            model_probability=model_probability,
            actual_winner=actual_winner,
            is_correct=is_correct,
            correctness_target=correctness_target,
            brier_component=brier_component,
            evaluation_row_fingerprint=eval_fp,
        )
        evaluation_rows_list.append(eval_row)

    # Sort evaluation rows by prediction_observation_id
    sorted_eval_rows = tuple(
        sorted(evaluation_rows_list, key=lambda r: r.prediction_observation_id)
    )

    scorecard = build_scorecard(sorted_eval_rows, source_attachment_set_fingerprint)

    claims = {
        "db_written": False,
        "deployed": False,
        "legacy_rows_admitted": False,
        "model_promoted": False,
        "model_superiority_claim": False,
        "network_called": False,
        "odds_used": False,
        "profitability_claim": False,
        "provider_called": False,
        "real_model_performance_claim": False,
        "retraining_performed": False,
        "sample_limited": True,
        "synthetic_results": True,
    }

    return PredictionEvaluationScorecardResult(
        schema_version=SCHEMA_VERSION,
        source_attachments_sha256=attachments_sha256,
        source_summary_sha256=attachment_summary_sha256,
        source_attachment_set_fingerprint=source_attachment_set_fingerprint,
        source_snapshot_sha256=snapshot_sha256,
        source_snapshot_summary_sha256=snapshot_summary_sha256,
        source_snapshot_fingerprint=source_snapshot_fingerprint,
        source_row_count=source_row_count,
        source_attached_count=source_attached_count,
        source_rejected_count=source_rejected_count,
        evaluation_row_count=scorecard.evaluation_row_count,
        excluded_rejected_count=excluded_rejected_count,
        correct_count=scorecard.correct_count,
        incorrect_count=scorecard.incorrect_count,
        accuracy=round(float(scorecard.accuracy), 6),
        mean_selected_side_probability=round(float(scorecard.mean_selected_side_probability), 6),
        brier_score=round(float(scorecard.brier_score), 6),
        scorecard=scorecard,
        evaluation_rows=sorted_eval_rows,
        evaluation_set_fingerprint=scorecard.evaluation_set_fingerprint,
        claims=claims,
    )
