"""Deterministic P20B historical feedback artifact rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .prediction_feedback_artifacts import render_feedback_jsonl
from .replay_historical_prediction_feedback import (
    HistoricalFeedbackReplayResult,
)


def render_historical_feedback_jsonl(
    result: HistoricalFeedbackReplayResult,
) -> str:
    """Render the existing P17A feedback rows without changing their schema."""

    return render_feedback_jsonl(result.feedback_result)


def render_historical_feedback_summary_json(
    result: HistoricalFeedbackReplayResult,
    feedback_jsonl_sha256: str,
    report_sha256: str,
) -> str:
    """Render the P20B-owned summary and explicit non-synthetic claims."""

    summary: dict[str, Any] = {
        "claims": result.claims,
        "feedback_jsonl_sha256": feedback_jsonl_sha256,
        "feedback_ledger_fingerprint": (
            result.feedback_result.feedback_ledger_fingerprint
        ),
        "feedback_row_count": result.feedback_result.prediction_row_count,
        "historical_provenance": result.historical_provenance.to_projection(),
        "historical_provenance_sha256": result.historical_provenance_sha256,
        "historical_results_sha256": result.historical_results_sha256,
        "p15c_admission_count": len(result.admission_workflow.results),
        "p15c_admission_result_set_fingerprint": (
            result.admission_workflow.result_set_fingerprint
        ),
        "p15c_snapshot_fingerprint": result.snapshot_result.snapshot_fingerprint,
        "p16a_attachment_count": result.attachment_result.attached_count,
        "p16a_attachment_set_fingerprint": (
            result.attachment_result.attachment_set_fingerprint
        ),
        "p16b_evaluation_count": result.evaluation_result.evaluation_row_count,
        "p16b_evaluation_set_fingerprint": (
            result.evaluation_result.evaluation_set_fingerprint
        ),
        "p17a_feedback_ledger_fingerprint": (
            result.feedback_result.feedback_ledger_fingerprint
        ),
        "p20a_fold_id": result.p20a_fold_id,
        "p20a_fold_sha256": result.p20a_fold_sha256,
        "p20a_model_fingerprint": result.p20a_model_fingerprint,
        "p20a_prediction_sha256": result.p20a_prediction_sha256,
        "p20a_reconstruction_sha256": result.p20a_reconstruction_sha256,
        "p20a_summary_sha256": result.p20a_summary_sha256,
        "p19a_model_artifact_fingerprint": (
            result.p19a_model_artifact_fingerprint
        ),
        "replay_game_ids": list(result.replay_game_ids),
        "report_sha256": report_sha256,
        "schema_version": "p20b.first_non_synthetic_historical_feedback.v1",
    }
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_historical_feedback_report_markdown(
    result: HistoricalFeedbackReplayResult,
) -> str:
    """Render a human-readable, non-performance-claiming P20B report."""

    provenance = result.historical_provenance
    feedback = result.feedback_result
    lines = [
        "# P20B First Non-Synthetic Historical Feedback Replay",
        "",
        "This is a deterministic, historical, paper-only diagnostic lineage.",
        "It is not a training dataset and makes no production-performance or betting claim.",
        "",
        "## Lineage",
        "",
        f"- **P20A Fold**: `{result.p20a_fold_id}`",
        f"- **P20A Model Fingerprint**: `{result.p20a_model_fingerprint}`",
        f"- **P19A Model Artifact Fingerprint**: `{result.p19a_model_artifact_fingerprint}`",
        f"- **Replay Game IDs**: `{', '.join(result.replay_game_ids)}`",
        f"- **P20A Prediction Artifact SHA-256**: `{result.p20a_prediction_sha256}`",
        "",
        "## Historical Result Provenance",
        "",
        f"- **Repository**: `{provenance.source_repository}`",
        f"- **Commit**: `{provenance.source_commit}`",
        f"- **Tree**: `{provenance.source_tree}`",
        f"- **Archive Path**: `{provenance.source_archive_path}`",
        f"- **Archive Blob**: `{provenance.source_archive_blob}`",
        f"- **Archive SHA-256**: `{provenance.source_archive_sha256}`",
        f"- **Archive Member**: `{provenance.source_member}`",
        f"- **Evidence Verified At**: `{provenance.source_verified_at_utc}`",
        "",
        "| Game | Away Score | Home Score | Derived Winner |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in provenance.rows:
        winner = "HOME" if row["home_score"] > row["away_score"] else "AWAY"
        lines.append(
            f"| `{row['canonical_game_id']}` | {row['away_score']} | "
            f"{row['home_score']} | `{winner}` |"
        )

    lines.extend([
        "",
        "## Contract Results",
        "",
        f"- **P15C admitted observations**: `{len(result.admission_workflow.results)}`",
        f"- **P16A attached observations**: `{result.attachment_result.attached_count}`",
        f"- **P16B evaluated observations**: `{result.evaluation_result.evaluation_row_count}`",
        f"- **P17A feedback rows**: `{feedback.prediction_row_count}`",
        f"- **Correct / Incorrect**: `{feedback.correct_count}` / `{feedback.incorrect_count}` (diagnostic only)",
        f"- **Feedback Ledger Fingerprint**: `{feedback.feedback_ledger_fingerprint}`",
        "",
        "## Explicit Claims",
        "",
    ])
    for key, value in sorted(result.claims.items()):
        lines.append(f"- **{key}**: `{str(value).lower()}`")

    lines.extend([
        "",
        "## Safety Boundaries",
        "",
        "- Exactly two P20A replay game IDs are included.",
        "- Both committed P20A HOME/AWAY candidate observations are retained.",
        "- Existing P15C/P16A/P16B/P17A semantics are reused; their implementations are unchanged.",
        "- No provider, network, database, odds, ROI, EV, Kelly, training, retraining, or promotion behavior is used.",
        "- This sample is insufficient for training readiness, production readiness, or model-quality claims.",
        "",
    ])
    return "\n".join(lines)


def write_historical_feedback_replay_artifacts(
    output_dir: Path,
    result: HistoricalFeedbackReplayResult,
) -> None:
    """Write only the P20B-owned deterministic feedback artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    feedback_content = render_historical_feedback_jsonl(result)
    feedback_sha256 = hashlib.sha256(feedback_content.encode("utf-8")).hexdigest()
    report_content = render_historical_feedback_report_markdown(result)
    report_sha256 = hashlib.sha256(report_content.encode("utf-8")).hexdigest()
    summary_content = render_historical_feedback_summary_json(
        result,
        feedback_sha256,
        report_sha256,
    )

    (output_dir / "feedback.jsonl").write_text(feedback_content, encoding="utf-8")
    (output_dir / "summary.json").write_text(summary_content, encoding="utf-8")
    (output_dir / "report.md").write_text(report_content, encoding="utf-8")
