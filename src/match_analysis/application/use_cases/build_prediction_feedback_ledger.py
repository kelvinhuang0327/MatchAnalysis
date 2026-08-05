"""Build a deterministic prediction feedback ledger from P15C, P16A, and P16B artifacts.

Joins verified P15C prediction observation snapshots, P16A final-result attachments,
and P16B evaluation scorecard rows into one auditable feedback ledger.
Does not construct PredictionSourceObservation or FinalResultObservation,
re-run admission or result attachment, recalculate evaluation policy,
calculate odds/ROI/EV, retrain models, or touch external networks or databases.
"""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from typing import Any

from ...baseball.domain.prediction_feedback import (
    FEEDBACK_LEDGER_SCHEMA_VERSION,
    FEEDBACK_ROW_SCHEMA_VERSION,
    FEEDBACK_STATUS_EVALUATED,
    FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED,
    PredictionFeedbackRow,
    compute_feedback_ledger_fingerprint,
    compute_feedback_row_fingerprint,
)


@dataclass(frozen=True, slots=True)
class PredictionFeedbackLedgerResult:
    """Immutable result of building the prediction feedback ledger."""

    schema_version: str

    # Source SHA-256 values
    source_snapshot_sha256: str
    source_snapshot_summary_sha256: str
    source_attachments_sha256: str
    source_attachment_summary_sha256: str
    source_evaluations_sha256: str
    source_evaluation_summary_sha256: str

    # Source fingerprints
    source_snapshot_fingerprint: str
    source_attachment_set_fingerprint: str
    source_evaluation_set_fingerprint: str

    # Row counts
    prediction_row_count: int
    attached_row_count: int
    rejected_attachment_row_count: int
    evaluated_row_count: int
    non_evaluated_row_count: int
    correct_count: int
    incorrect_count: int

    # Status distributions
    feedback_status_counts: dict[str, int]
    attachment_rejection_reason_counts: dict[str, int]

    # Rows and fingerprint
    feedback_rows: tuple[PredictionFeedbackRow, ...]
    feedback_ledger_fingerprint: str

    # Claims
    claims: dict[str, bool]


def _validate_json_no_duplicate_keys(
    raw: str, context: str,
) -> dict[str, Any]:
    """Parse JSON rejecting duplicate keys at all levels."""

    def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, int] = {}
        for key, _ in pairs:
            if key in seen:
                raise ValueError(
                    f"Duplicate JSON key {key!r} in {context}"
                )
            seen[key] = 1
        return dict(pairs)

    try:
        data = json.loads(raw, object_pairs_hook=_object_pairs_hook)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {context}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a JSON object")
    return data


def _compute_snapshot_row_fingerprint(row_dict: dict[str, Any]) -> str:
    """Recompute P15C snapshot row fingerprint for verification."""
    canonical_payload = {
        "prediction_observation_id": row_dict["prediction_observation_id"],
        "source_result_row_fingerprint": row_dict["source_result_row_fingerprint"],
        "observation": row_dict["observation"],
    }
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_snapshot_fingerprint(
    snapshot_rows: list[dict[str, Any]],
) -> str:
    """Recompute P15C snapshot fingerprint for verification."""
    parts = []
    for row in sorted(
        snapshot_rows, key=lambda r: r["prediction_observation_id"]
    ):
        pred_id = row["prediction_observation_id"]
        row_fp = row["snapshot_row_fingerprint"]
        parts.append(f"{pred_id}:{row_fp}\n")
    combined = "".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _compute_attachment_row_fingerprint(row_dict: dict[str, Any]) -> str:
    """Recompute P16A attachment row fingerprint for verification."""
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
    """Recompute P16A attachment set fingerprint for verification."""
    parts = []
    for row in sorted(
        attachment_rows, key=lambda r: r["prediction_observation_id"]
    ):
        pred_id = row["prediction_observation_id"]
        row_fp = row["attachment_row_fingerprint"]
        parts.append(f"{pred_id}:{row_fp}\n")
    combined = "".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _compute_evaluation_row_fingerprint(row_dict: dict[str, Any]) -> str:
    """Recompute P16B evaluation row fingerprint for verification."""
    payload = {
        "actual_winner": row_dict["actual_winner"],
        "brier_component": row_dict["brier_component"],
        "correctness_target": row_dict["correctness_target"],
        "game_number": row_dict["game_number"],
        "is_correct": row_dict["is_correct"],
        "market_id": row_dict["market_id"],
        "model_id": row_dict["model_id"],
        "model_probability": row_dict["model_probability"],
        "prediction_observation_id": row_dict["prediction_observation_id"],
        "provider_game_id": row_dict["provider_game_id"],
        "provider_namespace": row_dict["provider_namespace"],
        "selection": row_dict["selection"],
        "source_attachment_row_fingerprint": row_dict["source_attachment_row_fingerprint"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_evaluation_set_fingerprint(
    evaluation_rows: list[dict[str, Any]],
) -> str:
    """Recompute P16B evaluation set fingerprint for verification."""
    parts = []
    for row in sorted(
        evaluation_rows, key=lambda r: r["prediction_observation_id"]
    ):
        pred_id = row["prediction_observation_id"]
        row_fp = row["evaluation_row_fingerprint"]
        parts.append(f"{pred_id}:{row_fp}\n")
    combined = "".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def build_prediction_feedback_ledger(
    *,
    snapshot_bytes: bytes,
    snapshot_summary_bytes: bytes,
    attachments_bytes: bytes,
    attachment_summary_bytes: bytes,
    evaluations_bytes: bytes,
    evaluation_summary_bytes: bytes,
) -> PredictionFeedbackLedgerResult:
    """Build a deterministic prediction feedback ledger.

    Validates all three source artifact stages, verifies SHA-256 hashes,
    fingerprints and cross-stage lineage, and joins them into an immutable
    feedback ledger.

    Raises ValueError on structural failures, hash mismatches, or invalid row schemas.
    """
    # 1. Compute SHA-256 of all inputs
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot_summary_sha256 = hashlib.sha256(snapshot_summary_bytes).hexdigest()
    attachments_sha256 = hashlib.sha256(attachments_bytes).hexdigest()
    attachment_summary_sha256 = hashlib.sha256(attachment_summary_bytes).hexdigest()
    evaluations_sha256 = hashlib.sha256(evaluations_bytes).hexdigest()
    evaluation_summary_sha256 = hashlib.sha256(evaluation_summary_bytes).hexdigest()

    # 2. Parse all summaries
    snapshot_summary = _validate_json_no_duplicate_keys(
        snapshot_summary_bytes.decode("utf-8"), "P15C summary"
    )
    attachment_summary = _validate_json_no_duplicate_keys(
        attachment_summary_bytes.decode("utf-8"), "P16A summary"
    )
    evaluation_summary = _validate_json_no_duplicate_keys(
        evaluation_summary_bytes.decode("utf-8"), "P16B summary"
    )

    # 3. Verify P16A summary → P15C
    att_expected_snapshot_sha256 = attachment_summary.get("source_snapshot_sha256")
    if snapshot_sha256 != att_expected_snapshot_sha256:
        raise ValueError(
            f"P16A summary source_snapshot_sha256 mismatch: "
            f"computed {snapshot_sha256}, summary expects {att_expected_snapshot_sha256}"
        )

    att_expected_snapshot_summary_sha256 = attachment_summary.get("source_snapshot_summary_sha256")
    if snapshot_summary_sha256 != att_expected_snapshot_summary_sha256:
        raise ValueError(
            f"P16A summary source_snapshot_summary_sha256 mismatch: "
            f"computed {snapshot_summary_sha256}, summary expects {att_expected_snapshot_summary_sha256}"
        )

    att_expected_attachments_sha256 = attachment_summary.get("attachments_jsonl_sha256")
    if attachments_sha256 != att_expected_attachments_sha256:
        raise ValueError(
            f"P16A summary attachments_jsonl_sha256 mismatch: "
            f"computed {attachments_sha256}, summary expects {att_expected_attachments_sha256}"
        )

    # 4. Verify P16B summary → P16A
    eval_expected_attachments_sha256 = evaluation_summary.get("source_attachments_sha256")
    if attachments_sha256 != eval_expected_attachments_sha256:
        raise ValueError(
            f"P16B summary source_attachments_sha256 mismatch: "
            f"computed {attachments_sha256}, summary expects {eval_expected_attachments_sha256}"
        )

    eval_expected_summary_sha256 = evaluation_summary.get("source_summary_sha256")
    if attachment_summary_sha256 != eval_expected_summary_sha256:
        raise ValueError(
            f"P16B summary source_summary_sha256 mismatch: "
            f"computed {attachment_summary_sha256}, summary expects {eval_expected_summary_sha256}"
        )

    eval_expected_evaluations_sha256 = evaluation_summary.get("evaluations_jsonl_sha256")
    if evaluations_sha256 != eval_expected_evaluations_sha256:
        raise ValueError(
            f"P16B summary evaluations_jsonl_sha256 mismatch: "
            f"computed {evaluations_sha256}, summary expects {eval_expected_evaluations_sha256}"
        )

    # Extract fingerprints
    source_snapshot_fingerprint = snapshot_summary["snapshot_fingerprint"]
    source_attachment_set_fingerprint = attachment_summary["attachment_set_fingerprint"]
    source_evaluation_set_fingerprint = evaluation_summary["evaluation_set_fingerprint"]

    # Cross-check fingerprints between summaries
    att_snapshot_fp = attachment_summary.get("source_snapshot_fingerprint")
    if att_snapshot_fp != source_snapshot_fingerprint:
        raise ValueError(
            f"Snapshot fingerprint mismatch: P15C summary has {source_snapshot_fingerprint}, "
            f"P16A summary has {att_snapshot_fp}"
        )

    eval_att_set_fp = evaluation_summary.get("source_attachment_set_fingerprint")
    if eval_att_set_fp != source_attachment_set_fingerprint:
        raise ValueError(
            f"Attachment set fingerprint mismatch: P16A summary has {source_attachment_set_fingerprint}, "
            f"P16B summary has {eval_att_set_fp}"
        )

    # Also verify P16B carries forward the snapshot fingerprint
    eval_snapshot_fp = evaluation_summary.get("source_snapshot_fingerprint")
    if eval_snapshot_fp is not None and eval_snapshot_fp != source_snapshot_fingerprint:
        raise ValueError(
            f"P16B snapshot fingerprint mismatch: P15C has {source_snapshot_fingerprint}, "
            f"P16B has {eval_snapshot_fp}"
        )

    # 5. Parse P15C snapshot JSONL
    snapshot_lines = [
        line for line in snapshot_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    snapshot_map: dict[str, dict[str, Any]] = {}
    for i, line in enumerate(snapshot_lines):
        snap_row = _validate_json_no_duplicate_keys(line, f"P15C row {i + 1}")
        pred_id = snap_row["prediction_observation_id"]
        if pred_id in snapshot_map:
            raise ValueError(f"Duplicate prediction_observation_id in P15C snapshot: {pred_id}")

        # Verify row fingerprint
        computed_fp = _compute_snapshot_row_fingerprint(snap_row)
        if snap_row.get("snapshot_row_fingerprint") != computed_fp:
            raise ValueError(
                f"P15C row fingerprint mismatch for {pred_id}: "
                f"computed {computed_fp}, got {snap_row.get('snapshot_row_fingerprint')}"
            )
        snapshot_map[pred_id] = snap_row

    # Verify snapshot fingerprint
    computed_snapshot_fp = _compute_snapshot_fingerprint(list(snapshot_map.values()))
    if computed_snapshot_fp != source_snapshot_fingerprint:
        raise ValueError(
            f"P15C snapshot fingerprint recomputation mismatch: "
            f"computed {computed_snapshot_fp}, summary has {source_snapshot_fingerprint}"
        )

    # 6. Parse P16A attachment JSONL
    attachment_lines = [
        line for line in attachments_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    attachment_map: dict[str, dict[str, Any]] = {}
    for i, line in enumerate(attachment_lines):
        att_row = _validate_json_no_duplicate_keys(line, f"P16A row {i + 1}")
        pred_id = att_row["prediction_observation_id"]
        if pred_id in attachment_map:
            raise ValueError(f"Duplicate prediction_observation_id in P16A attachments: {pred_id}")

        # Verify row fingerprint
        computed_att_fp = _compute_attachment_row_fingerprint(att_row)
        if att_row.get("attachment_row_fingerprint") != computed_att_fp:
            raise ValueError(
                f"P16A row fingerprint mismatch for {pred_id}: "
                f"computed {computed_att_fp}, got {att_row.get('attachment_row_fingerprint')}"
            )

        # Verify source_snapshot_row_fingerprint matches P15C
        snap_row = snapshot_map.get(pred_id)
        if snap_row is None:
            raise ValueError(
                f"P16A attachment references unknown P15C prediction: {pred_id}"
            )
        if att_row["source_snapshot_row_fingerprint"] != snap_row["snapshot_row_fingerprint"]:
            raise ValueError(
                f"P16A→P15C fingerprint mismatch for {pred_id}: "
                f"P16A has {att_row['source_snapshot_row_fingerprint']}, "
                f"P15C has {snap_row['snapshot_row_fingerprint']}"
            )
        attachment_map[pred_id] = att_row

    # Every P15C prediction must have exactly one P16A attachment
    for pred_id in snapshot_map:
        if pred_id not in attachment_map:
            raise ValueError(
                f"Missing P16A attachment for P15C prediction: {pred_id}"
            )

    # Verify attachment set fingerprint
    computed_att_set_fp = _compute_attachment_set_fingerprint(list(attachment_map.values()))
    if computed_att_set_fp != source_attachment_set_fingerprint:
        raise ValueError(
            f"P16A attachment set fingerprint recomputation mismatch: "
            f"computed {computed_att_set_fp}, summary has {source_attachment_set_fingerprint}"
        )

    # 7. Parse P16B evaluation JSONL
    evaluation_lines = [
        line for line in evaluations_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    evaluation_map: dict[str, dict[str, Any]] = {}
    for i, line in enumerate(evaluation_lines):
        eval_row = _validate_json_no_duplicate_keys(line, f"P16B row {i + 1}")
        pred_id = eval_row["prediction_observation_id"]
        if pred_id in evaluation_map:
            raise ValueError(f"Duplicate prediction_observation_id in P16B evaluations: {pred_id}")

        # Verify row fingerprint
        computed_eval_fp = _compute_evaluation_row_fingerprint(eval_row)
        if eval_row.get("evaluation_row_fingerprint") != computed_eval_fp:
            raise ValueError(
                f"P16B row fingerprint mismatch for {pred_id}: "
                f"computed {computed_eval_fp}, got {eval_row.get('evaluation_row_fingerprint')}"
            )

        # Verify source_attachment_row_fingerprint matches P16A
        att_row = attachment_map.get(pred_id)
        if att_row is None:
            raise ValueError(
                f"P16B evaluation references unknown P16A attachment: {pred_id}"
            )
        if eval_row["source_attachment_row_fingerprint"] != att_row["attachment_row_fingerprint"]:
            raise ValueError(
                f"P16B→P16A fingerprint mismatch for {pred_id}: "
                f"P16B has {eval_row['source_attachment_row_fingerprint']}, "
                f"P16A has {att_row['attachment_row_fingerprint']}"
            )

        # Evaluation must only exist for ATTACHED rows
        if att_row["attachment_status"] != "ATTACHED":
            raise ValueError(
                f"P16B evaluation exists for non-ATTACHED row: {pred_id} "
                f"(status={att_row['attachment_status']})"
            )
        evaluation_map[pred_id] = eval_row

    # Verify evaluation set fingerprint
    computed_eval_set_fp = _compute_evaluation_set_fingerprint(list(evaluation_map.values()))
    if computed_eval_set_fp != source_evaluation_set_fingerprint:
        raise ValueError(
            f"P16B evaluation set fingerprint recomputation mismatch: "
            f"computed {computed_eval_set_fp}, summary has {source_evaluation_set_fingerprint}"
        )

    # 8. Build feedback rows
    feedback_rows_list: list[PredictionFeedbackRow] = []

    for pred_id in sorted(snapshot_map.keys()):
        snap_row = snapshot_map[pred_id]
        att_row = attachment_map[pred_id]
        obs = snap_row["observation"]

        provider_namespace = obs["provider_namespace"]
        provider_game_id = obs["provider_game_id"]
        game_number = obs["game_number"]
        scheduled_start_utc = obs["scheduled_start_utc"]
        model_id = obs["model_id"]
        market_id = obs["market_id"]
        selection = obs["selection"]
        model_probability = Decimal(str(obs["model_probability"]))

        att_status = att_row["attachment_status"]

        if att_status == "ATTACHED":
            # Must have P16B evaluation
            eval_row = evaluation_map.get(pred_id)
            if eval_row is None:
                raise ValueError(
                    f"Missing P16B evaluation for ATTACHED prediction: {pred_id}"
                )

            feedback_status = FEEDBACK_STATUS_EVALUATED
            source_eval_fp = eval_row["evaluation_row_fingerprint"]
            result_observation_id = att_row.get("result_observation_id")
            result_observed_at_utc = att_row.get("result_observed_at_utc")
            home_score = att_row.get("home_score")
            away_score = att_row.get("away_score")
            actual_winner = att_row.get("actual_winner")
            is_correct = eval_row["is_correct"]
            correctness_target = eval_row["correctness_target"]
            brier_component = Decimal(str(eval_row["brier_component"]))
            attachment_rejection_reason = None

        elif att_status == "REJECTED":
            # Must NOT have P16B evaluation
            if pred_id in evaluation_map:
                raise ValueError(
                    f"P16B evaluation exists for REJECTED prediction: {pred_id}"
                )

            # Verify no partial result fields
            for field in ("result_observation_id", "result_observed_at_utc",
                          "home_score", "away_score", "actual_winner"):
                if att_row.get(field) is not None:
                    raise ValueError(
                        f"REJECTED attachment has non-null {field}: {pred_id}"
                    )

            feedback_status = FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED
            source_eval_fp = None
            result_observation_id = None
            result_observed_at_utc = None
            home_score = None
            away_score = None
            actual_winner = None
            is_correct = None
            correctness_target = None
            brier_component = None
            attachment_rejection_reason = att_row.get("rejection_reason")

            if attachment_rejection_reason is None:
                raise ValueError(
                    f"REJECTED attachment missing rejection_reason: {pred_id}"
                )
        else:
            raise ValueError(f"Unknown attachment_status: {att_status}")

        row_fp = compute_feedback_row_fingerprint(
            prediction_observation_id=pred_id,
            source_snapshot_row_fingerprint=snap_row["snapshot_row_fingerprint"],
            source_attachment_row_fingerprint=att_row["attachment_row_fingerprint"],
            source_evaluation_row_fingerprint=source_eval_fp,
            provider_namespace=provider_namespace,
            provider_game_id=provider_game_id,
            game_number=game_number,
            scheduled_start_utc=scheduled_start_utc,
            model_id=model_id,
            market_id=market_id,
            selection=selection,
            model_probability=model_probability,
            result_observation_id=result_observation_id,
            result_observed_at_utc=result_observed_at_utc,
            home_score=home_score,
            away_score=away_score,
            actual_winner=actual_winner,
            attachment_status=att_status,
            attachment_rejection_reason=attachment_rejection_reason,
            feedback_status=feedback_status,
            is_correct=is_correct,
            correctness_target=correctness_target,
            brier_component=brier_component,
        )

        feedback_row = PredictionFeedbackRow(
            prediction_observation_id=pred_id,
            source_snapshot_row_fingerprint=snap_row["snapshot_row_fingerprint"],
            source_attachment_row_fingerprint=att_row["attachment_row_fingerprint"],
            source_evaluation_row_fingerprint=source_eval_fp,
            provider_namespace=provider_namespace,
            provider_game_id=provider_game_id,
            game_number=game_number,
            scheduled_start_utc=scheduled_start_utc,
            model_id=model_id,
            market_id=market_id,
            selection=selection,
            model_probability=model_probability,
            observation_payload=obs,
            result_observation_id=result_observation_id,
            result_observed_at_utc=result_observed_at_utc,
            home_score=home_score,
            away_score=away_score,
            actual_winner=actual_winner,
            attachment_status=att_status,
            attachment_rejection_reason=attachment_rejection_reason,
            feedback_status=feedback_status,
            is_correct=is_correct,
            correctness_target=correctness_target,
            brier_component=brier_component,
            feedback_row_fingerprint=row_fp,
        )
        feedback_rows_list.append(feedback_row)

    sorted_feedback_rows = tuple(feedback_rows_list)  # already sorted above

    # 9. Compute counts
    evaluated_count = sum(
        1 for r in sorted_feedback_rows if r.feedback_status == FEEDBACK_STATUS_EVALUATED
    )
    rejected_count = sum(
        1 for r in sorted_feedback_rows
        if r.feedback_status == FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED
    )
    attached_count = sum(
        1 for r in sorted_feedback_rows if r.attachment_status == "ATTACHED"
    )
    correct_count = sum(
        1 for r in sorted_feedback_rows if r.is_correct is True
    )
    incorrect_count = sum(
        1 for r in sorted_feedback_rows
        if r.is_correct is False
    )

    feedback_status_counts: dict[str, int] = {}
    for r in sorted_feedback_rows:
        feedback_status_counts[r.feedback_status] = (
            feedback_status_counts.get(r.feedback_status, 0) + 1
        )

    rejection_reason_counts: dict[str, int] = {}
    for r in sorted_feedback_rows:
        if r.attachment_rejection_reason is not None:
            rejection_reason_counts[r.attachment_rejection_reason] = (
                rejection_reason_counts.get(r.attachment_rejection_reason, 0) + 1
            )

    # 10. Compute ledger fingerprint
    ledger_fp = compute_feedback_ledger_fingerprint(sorted_feedback_rows)

    claims = {
        "db_written": False,
        "deployed": False,
        "model_promoted": False,
        "network_called": False,
        "odds_used": False,
        "profitability_claim": False,
        "provider_called": False,
        "real_model_performance_claim": False,
        "retraining_performed": False,
        "sample_limited": True,
        "synthetic_results": True,
        "training_dataset_claim": False,
    }

    return PredictionFeedbackLedgerResult(
        schema_version=FEEDBACK_LEDGER_SCHEMA_VERSION,
        source_snapshot_sha256=snapshot_sha256,
        source_snapshot_summary_sha256=snapshot_summary_sha256,
        source_attachments_sha256=attachments_sha256,
        source_attachment_summary_sha256=attachment_summary_sha256,
        source_evaluations_sha256=evaluations_sha256,
        source_evaluation_summary_sha256=evaluation_summary_sha256,
        source_snapshot_fingerprint=source_snapshot_fingerprint,
        source_attachment_set_fingerprint=source_attachment_set_fingerprint,
        source_evaluation_set_fingerprint=source_evaluation_set_fingerprint,
        prediction_row_count=len(sorted_feedback_rows),
        attached_row_count=attached_count,
        rejected_attachment_row_count=rejected_count,
        evaluated_row_count=evaluated_count,
        non_evaluated_row_count=rejected_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        feedback_status_counts=dict(sorted(feedback_status_counts.items())),
        attachment_rejection_reason_counts=dict(sorted(rejection_reason_counts.items())),
        feedback_rows=sorted_feedback_rows,
        feedback_ledger_fingerprint=ledger_fp,
        claims=claims,
    )
