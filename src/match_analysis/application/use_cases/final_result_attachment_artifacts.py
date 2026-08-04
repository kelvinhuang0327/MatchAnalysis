"""Deterministic artifact rendering for final result attachment.

Emits attachments.jsonl, summary.json, and report.md without runtime timestamps.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from .attach_final_results_to_admitted_predictions import (
    FinalResultAttachmentResult,
    PredictionFinalResultAttachmentRow,
)


def _attachment_row_to_dict(
    row: PredictionFinalResultAttachmentRow,
) -> dict[str, Any]:
    """Convert an attachment row to a JSON-serializable dictionary."""
    return {
        "actual_winner": row.actual_winner,
        "attachment_row_fingerprint": row.attachment_row_fingerprint,
        "attachment_status": row.attachment_status,
        "away_score": row.away_score,
        "game_number": row.game_number,
        "home_score": row.home_score,
        "is_correct": row.is_correct,
        "prediction_observation_id": row.prediction_observation_id,
        "provider_game_id": row.provider_game_id,
        "provider_namespace": row.provider_namespace,
        "rejection_reason": row.rejection_reason,
        "result_observation_id": row.result_observation_id,
        "result_observed_at_utc": row.result_observed_at_utc,
        "scheduled_start_utc": row.scheduled_start_utc,
        "selection": row.selection,
        "source_snapshot_row_fingerprint": row.source_snapshot_row_fingerprint,
    }


def render_attachments_jsonl(
    result: FinalResultAttachmentResult,
) -> str:
    """Render deterministic attachments.jsonl content."""
    lines = [
        json.dumps(
            _attachment_row_to_dict(row),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in result.attachment_rows
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def render_attachment_summary_json(
    result: FinalResultAttachmentResult,
    attachments_jsonl_sha256: str,
    report_sha256: str,
) -> str:
    """Render deterministic summary.json content."""
    summary_data = {
        "schema_version": result.schema_version,
        "source_snapshot_sha256": result.source_snapshot_sha256,
        "source_snapshot_summary_sha256": result.source_snapshot_summary_sha256,
        "source_snapshot_fingerprint": result.source_snapshot_fingerprint,
        "result_input_sha256": result.result_input_sha256,
        "source_prediction_count": result.source_prediction_count,
        "final_result_observation_count": result.final_result_observation_count,
        "attached_count": result.attached_count,
        "rejected_count": result.rejected_count,
        "rejection_reason_counts": result.rejection_reason_counts,
        "correct_count": result.correct_count,
        "incorrect_count": result.incorrect_count,
        "descriptive_accuracy": result.descriptive_accuracy,
        "attachment_set_fingerprint": result.attachment_set_fingerprint,
        "attachments_jsonl_sha256": attachments_jsonl_sha256,
        "report_sha256": report_sha256,
        "claims": result.claims,
    }
    return json.dumps(summary_data, indent=2, sort_keys=True) + "\n"


def render_attachment_report_markdown(
    result: FinalResultAttachmentResult,
) -> str:
    """Render deterministic report.md markdown content."""
    lines = [
        "# Final Result Attachment Report",
        "",
        "## Source",
        "",
        f"- **Source P15C Snapshot Fingerprint**: `{result.source_snapshot_fingerprint}`",
        f"- **Source Prediction Count**: {result.source_prediction_count}",
        f"- **Final Result Input Count**: {result.final_result_observation_count}",
        "",
        "## Results",
        "",
        f"- **Attached**: {result.attached_count}",
        f"- **Rejected**: {result.rejected_count}",
        f"- **Correct**: {result.correct_count}",
        f"- **Incorrect**: {result.incorrect_count}",
    ]

    if result.descriptive_accuracy is not None:
        lines.append(
            f"- **Descriptive Accuracy (synthetic-demo-only)**: {result.descriptive_accuracy}"
        )

    lines.extend([
        "",
        "## Attachment Details",
        "",
        "| Prediction Obs ID | Provider NS | Provider Game ID | Game # | Selection | Score | Actual Winner | Correct | Rejection Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for row in result.attachment_rows:
        obs_id_short = f"`{row.prediction_observation_id[:16]}...`"
        if row.attachment_status == "ATTACHED":
            score = f"{row.home_score}-{row.away_score}"
            winner = row.actual_winner
            correct = str(row.is_correct)
            reason = ""
        else:
            score = ""
            winner = ""
            correct = ""
            reason = row.rejection_reason or ""
        lines.append(
            f"| {obs_id_short} "
            f"| `{row.provider_namespace}` "
            f"| `{row.provider_game_id}` "
            f"| {row.game_number} "
            f"| {row.selection} "
            f"| {score} "
            f"| {winner} "
            f"| {correct} "
            f"| {reason} |"
        )

    lines.extend([
        "",
        f"## Attachment Set Fingerprint",
        "",
        f"`{result.attachment_set_fingerprint}`",
        "",
        "## Safety Disclaimer",
        "",
        "- This report contains **synthetic local result evidence only**.",
        "- **No provider or network call** was made.",
        "- **No database write** occurred.",
        "- **No odds or profitability claim** is made.",
        "- **No deployment** was performed.",
        "- This is **not real model-performance evidence**.",
        "- Descriptive accuracy is synthetic-demo-only and must not be described as real model performance evidence.",
        "",
    ])

    return "\n".join(lines)


def write_final_result_attachment_artifacts(
    output_dir: Path,
    result: FinalResultAttachmentResult,
) -> None:
    """Write attachments.jsonl, summary.json, and report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    attachments_content = render_attachments_jsonl(result)
    attachments_sha256 = hashlib.sha256(
        attachments_content.encode("utf-8")
    ).hexdigest()

    report_content = render_attachment_report_markdown(result)
    report_sha256 = hashlib.sha256(
        report_content.encode("utf-8")
    ).hexdigest()

    summary_content = render_attachment_summary_json(
        result, attachments_sha256, report_sha256,
    )

    (output_dir / "attachments.jsonl").write_text(
        attachments_content, encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        summary_content, encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        report_content, encoding="utf-8",
    )
