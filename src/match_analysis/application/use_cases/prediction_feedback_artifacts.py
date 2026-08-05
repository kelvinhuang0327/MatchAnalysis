"""Deterministic artifact rendering for prediction feedback ledger.

Emits feedback.jsonl, summary.json, and report.md without runtime timestamps.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from ...baseball.domain.prediction_feedback import (
    PredictionFeedbackRow,
)
from .build_prediction_feedback_ledger import (
    PredictionFeedbackLedgerResult,
)


def _feedback_row_to_dict(
    row: PredictionFeedbackRow,
) -> dict[str, Any]:
    """Convert a feedback row to a JSON-serializable dictionary."""
    return {
        "actual_winner": row.actual_winner,
        "attachment_rejection_reason": row.attachment_rejection_reason,
        "attachment_status": row.attachment_status,
        "away_score": row.away_score,
        "brier_component": str(row.brier_component) if row.brier_component is not None else None,
        "correctness_target": row.correctness_target,
        "feedback_row_fingerprint": row.feedback_row_fingerprint,
        "feedback_status": row.feedback_status,
        "game_number": row.game_number,
        "home_score": row.home_score,
        "is_correct": row.is_correct,
        "market_id": row.market_id,
        "model_id": row.model_id,
        "model_probability": str(row.model_probability),
        "observation_payload": row.observation_payload,
        "prediction_observation_id": row.prediction_observation_id,
        "provider_game_id": row.provider_game_id,
        "provider_namespace": row.provider_namespace,
        "result_observation_id": row.result_observation_id,
        "result_observed_at_utc": row.result_observed_at_utc,
        "scheduled_start_utc": row.scheduled_start_utc,
        "selection": row.selection,
        "source_attachment_row_fingerprint": row.source_attachment_row_fingerprint,
        "source_evaluation_row_fingerprint": row.source_evaluation_row_fingerprint,
        "source_snapshot_row_fingerprint": row.source_snapshot_row_fingerprint,
    }


def render_feedback_jsonl(
    result: PredictionFeedbackLedgerResult,
) -> str:
    """Render deterministic feedback.jsonl content."""
    lines = [
        json.dumps(
            _feedback_row_to_dict(row),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in result.feedback_rows
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def render_feedback_summary_json(
    result: PredictionFeedbackLedgerResult,
    feedback_jsonl_sha256: str,
    report_sha256: str,
) -> str:
    """Render deterministic summary.json content."""
    summary_data = {
        "attachment_rejection_reason_counts": result.attachment_rejection_reason_counts,
        "attached_row_count": result.attached_row_count,
        "claims": result.claims,
        "correct_count": result.correct_count,
        "evaluated_row_count": result.evaluated_row_count,
        "feedback_jsonl_sha256": feedback_jsonl_sha256,
        "feedback_ledger_fingerprint": result.feedback_ledger_fingerprint,
        "feedback_status_counts": result.feedback_status_counts,
        "incorrect_count": result.incorrect_count,
        "non_evaluated_row_count": result.non_evaluated_row_count,
        "prediction_row_count": result.prediction_row_count,
        "rejected_attachment_row_count": result.rejected_attachment_row_count,
        "report_sha256": report_sha256,
        "schema_version": result.schema_version,
        "source_attachment_set_fingerprint": result.source_attachment_set_fingerprint,
        "source_attachment_summary_sha256": result.source_attachment_summary_sha256,
        "source_attachments_sha256": result.source_attachments_sha256,
        "source_evaluation_set_fingerprint": result.source_evaluation_set_fingerprint,
        "source_evaluation_summary_sha256": result.source_evaluation_summary_sha256,
        "source_evaluations_sha256": result.source_evaluations_sha256,
        "source_snapshot_fingerprint": result.source_snapshot_fingerprint,
        "source_snapshot_sha256": result.source_snapshot_sha256,
        "source_snapshot_summary_sha256": result.source_snapshot_summary_sha256,
    }
    return json.dumps(summary_data, indent=2, sort_keys=True) + "\n"


def render_feedback_report_markdown(
    result: PredictionFeedbackLedgerResult,
) -> str:
    """Render deterministic report.md markdown content."""
    lines = [
        "# Prediction Feedback Ledger Report",
        "",
        "## Source Evidence",
        "",
        f"- **P15C Snapshot Fingerprint**: `{result.source_snapshot_fingerprint}`",
        f"- **P16A Attachment Set Fingerprint**: `{result.source_attachment_set_fingerprint}`",
        f"- **P16B Evaluation Set Fingerprint**: `{result.source_evaluation_set_fingerprint}`",
        "",
        "## Feedback Summary",
        "",
        f"- **Total Feedback Rows**: {result.prediction_row_count}",
        f"- **Evaluated Rows**: {result.evaluated_row_count}",
        f"- **Attachment-Rejected Rows**: {result.rejected_attachment_row_count}",
        f"- **Correct Count**: {result.correct_count}",
        f"- **Incorrect Count**: {result.incorrect_count}",
        "",
        "## Feedback Rows Detail",
        "",
        "| Prediction Obs ID | Provider NS | Game ID | Game# | Model | Market | Selection | Prob | Score | Winner | Rejection Reason | Correct | Brier |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in result.feedback_rows:
        obs_id_short = f"`{row.prediction_observation_id[:16]}...`"
        score = (
            f"{row.home_score}-{row.away_score}"
            if row.home_score is not None
            else "—"
        )
        winner = row.actual_winner if row.actual_winner is not None else "—"
        rejection = row.attachment_rejection_reason if row.attachment_rejection_reason is not None else "—"
        correct = str(row.is_correct) if row.is_correct is not None else "—"
        brier = str(row.brier_component) if row.brier_component is not None else "—"

        lines.append(
            f"| {obs_id_short} "
            f"| {row.provider_namespace} "
            f"| {row.provider_game_id} "
            f"| {row.game_number} "
            f"| {row.model_id} "
            f"| {row.market_id} "
            f"| {row.selection} "
            f"| {row.model_probability} "
            f"| {score} "
            f"| {winner} "
            f"| {rejection} "
            f"| {correct} "
            f"| {brier} |"
        )

    lines.extend([
        "",
        "## Deterministic Feedback Ledger Fingerprint",
        "",
        f"`{result.feedback_ledger_fingerprint}`",
        "",
        "## Limitations and Disclaimers",
        "",
        "- This report joins **synthetic local result evidence only**.",
        "- **Sample size is insufficient** for real model performance claims.",
        "- This is an **audit/feedback ledger**, not a training dataset.",
        "- **No model retraining or promotion** occurred.",
        "- **No provider or network call** was executed.",
        "- **No database write** occurred.",
        "- **No odds, payout, EV, ROI, Kelly, or betting evaluation** was used.",
        "- **No production deployment** was performed.",
        "",
    ])

    return "\n".join(lines)


def write_prediction_feedback_artifacts(
    output_dir: Path,
    result: PredictionFeedbackLedgerResult,
) -> None:
    """Write feedback.jsonl, summary.json, and report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    feedback_content = render_feedback_jsonl(result)
    feedback_sha256 = hashlib.sha256(
        feedback_content.encode("utf-8")
    ).hexdigest()

    report_content = render_feedback_report_markdown(result)
    report_sha256 = hashlib.sha256(
        report_content.encode("utf-8")
    ).hexdigest()

    summary_content = render_feedback_summary_json(
        result, feedback_sha256, report_sha256
    )

    (output_dir / "feedback.jsonl").write_text(
        feedback_content, encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        summary_content, encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        report_content, encoding="utf-8"
    )
