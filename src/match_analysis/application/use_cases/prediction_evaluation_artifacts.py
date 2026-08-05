"""Deterministic artifact rendering for prediction evaluation scorecard.

Emits evaluations.jsonl, summary.json, and report.md without runtime timestamps.
"""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from ...baseball.domain.prediction_evaluation import (
    BreakdownMetrics,
    PredictionEvaluationRow,
)
from .build_prediction_evaluation_scorecard import (
    PredictionEvaluationScorecardResult,
)


def _evaluation_row_to_dict(
    row: PredictionEvaluationRow,
) -> dict[str, Any]:
    """Convert an evaluation row to a JSON-serializable dictionary."""
    return {
        "actual_winner": row.actual_winner,
        "brier_component": str(row.brier_component),
        "correctness_target": row.correctness_target,
        "evaluation_row_fingerprint": row.evaluation_row_fingerprint,
        "game_number": row.game_number,
        "is_correct": row.is_correct,
        "market_id": row.market_id,
        "model_id": row.model_id,
        "model_probability": str(row.model_probability),
        "prediction_observation_id": row.prediction_observation_id,
        "provider_game_id": row.provider_game_id,
        "provider_namespace": row.provider_namespace,
        "selection": row.selection,
        "source_attachment_row_fingerprint": row.source_attachment_row_fingerprint,
    }


def render_evaluations_jsonl(
    result: PredictionEvaluationScorecardResult,
) -> str:
    """Render deterministic evaluations.jsonl content."""
    lines = [
        json.dumps(
            _evaluation_row_to_dict(row),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in result.evaluation_rows
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _breakdown_to_dict(b: BreakdownMetrics) -> dict[str, Any]:
    return {
        "accuracy": round(float(b.accuracy), 6),
        "brier_score": round(float(b.brier_score), 6),
        "correct_count": b.correct_count,
        "incorrect_count": b.incorrect_count,
        "mean_selected_side_probability": round(
            float(b.mean_selected_side_probability), 6
        ),
        "row_count": b.row_count,
    }


def render_evaluation_summary_json(
    result: PredictionEvaluationScorecardResult,
    evaluations_jsonl_sha256: str,
    report_sha256: str,
) -> str:
    """Render deterministic summary.json content."""
    summary_data = {
        "accuracy": result.accuracy,
        "brier_score": result.brier_score,
        "breakdown_by_market_id": {
            k: _breakdown_to_dict(v)
            for k, v in result.scorecard.breakdown_by_market_id.items()
        },
        "breakdown_by_model_id": {
            k: _breakdown_to_dict(v)
            for k, v in result.scorecard.breakdown_by_model_id.items()
        },
        "breakdown_by_selection": {
            k: _breakdown_to_dict(v)
            for k, v in result.scorecard.breakdown_by_selection.items()
        },
        "claims": result.claims,
        "correct_count": result.correct_count,
        "evaluation_row_count": result.evaluation_row_count,
        "evaluation_set_fingerprint": result.evaluation_set_fingerprint,
        "evaluations_jsonl_sha256": evaluations_jsonl_sha256,
        "excluded_rejected_count": result.excluded_rejected_count,
        "incorrect_count": result.incorrect_count,
        "mean_selected_side_probability": result.mean_selected_side_probability,
        "report_sha256": report_sha256,
        "schema_version": result.schema_version,
        "source_attachment_set_fingerprint": result.source_attachment_set_fingerprint,
        "source_attachments_sha256": result.source_attachments_sha256,
        "source_attached_count": result.source_attached_count,
        "source_rejected_count": result.source_rejected_count,
        "source_row_count": result.source_row_count,
        "source_snapshot_fingerprint": result.source_snapshot_fingerprint,
        "source_snapshot_sha256": result.source_snapshot_sha256,
        "source_snapshot_summary_sha256": result.source_snapshot_summary_sha256,
        "source_summary_sha256": result.source_summary_sha256,
    }
    return json.dumps(summary_data, indent=2, sort_keys=True) + "\n"


def render_evaluation_report_markdown(
    result: PredictionEvaluationScorecardResult,
) -> str:
    """Render deterministic report.md markdown content."""
    lines = [
        "# Prediction Evaluation Scorecard Report",
        "",
        "## Source Evidence",
        "",
        f"- **Source Attachment Set Fingerprint**: `{result.source_attachment_set_fingerprint}`",
        f"- **Source Snapshot Fingerprint**: `{result.source_snapshot_fingerprint}`",
        f"- **Source Rows Count**: {result.source_row_count} (Attached: {result.source_attached_count}, Rejected: {result.source_rejected_count})",
        "",
        "## Aggregate Scorecard Metrics",
        "",
        f"- **Evaluated Rows**: {result.evaluation_row_count}",
        f"- **Excluded Rejected Rows**: {result.excluded_rejected_count}",
        f"- **Correct Count**: {result.correct_count}",
        f"- **Incorrect Count**: {result.incorrect_count}",
        f"- **Accuracy**: {result.accuracy}",
        f"- **Mean Selected-Side Probability**: {result.mean_selected_side_probability}",
        f"- **Brier Score**: {result.brier_score}",
        "",
        "## Breakdowns",
        "",
        "### Breakdown by Model ID",
        "",
        "| Model ID | Count | Correct | Incorrect | Accuracy | Mean Probability | Brier Score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for model_id, b in result.scorecard.breakdown_by_model_id.items():
        lines.append(
            f"| `{model_id}` | {b.row_count} | {b.correct_count} | {b.incorrect_count} | {round(float(b.accuracy), 6)} | {round(float(b.mean_selected_side_probability), 6)} | {round(float(b.brier_score), 6)} |"
        )

    lines.extend([
        "",
        "### Breakdown by Market ID",
        "",
        "| Market ID | Count | Correct | Incorrect | Accuracy | Mean Probability | Brier Score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])

    for market_id, b in result.scorecard.breakdown_by_market_id.items():
        lines.append(
            f"| `{market_id}` | {b.row_count} | {b.correct_count} | {b.incorrect_count} | {round(float(b.accuracy), 6)} | {round(float(b.mean_selected_side_probability), 6)} | {round(float(b.brier_score), 6)} |"
        )

    lines.extend([
        "",
        "### Breakdown by Selection",
        "",
        "| Selection | Count | Correct | Incorrect | Accuracy | Mean Probability | Brier Score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])

    for selection, b in result.scorecard.breakdown_by_selection.items():
        lines.append(
            f"| `{selection}` | {b.row_count} | {b.correct_count} | {b.incorrect_count} | {round(float(b.accuracy), 6)} | {round(float(b.mean_selected_side_probability), 6)} | {round(float(b.brier_score), 6)} |"
        )

    lines.extend([
        "",
        "## Evaluated Rows Detail",
        "",
        "| Prediction Obs ID | Model ID | Selection | Model Prob | Actual Winner | Correct | Target | Brier Component |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for row in result.evaluation_rows:
        obs_id_short = f"`{row.prediction_observation_id[:16]}...`"
        lines.append(
            f"| {obs_id_short} "
            f"| `{row.model_id}` "
            f"| {row.selection} "
            f"| {row.model_probability} "
            f"| {row.actual_winner} "
            f"| {row.is_correct} "
            f"| {row.correctness_target} "
            f"| {row.brier_component} |"
        )

    lines.extend([
        "",
        "## Deterministic Evaluation Set Fingerprint",
        "",
        f"`{result.evaluation_set_fingerprint}`",
        "",
        "## Safety & Methodological Disclaimers",
        "",
        "- This report evaluates **synthetic local demo evidence only**.",
        "- **Sample size is limited** and insufficient for real model performance claims.",
        "- **No real model superiority or performance claim** is made.",
        "- **No model retraining or promotion** occurred.",
        "- **No provider or network call** was executed.",
        "- **No database write** occurred.",
        "- **No odds, payout, EV, ROI, Kelly, or betting recommendation** was used.",
        "- **No production deployment** was performed.",
        "",
    ])

    return "\n".join(lines)


def write_prediction_evaluation_artifacts(
    output_dir: Path,
    result: PredictionEvaluationScorecardResult,
) -> None:
    """Write evaluations.jsonl, summary.json, and report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations_content = render_evaluations_jsonl(result)
    evaluations_sha256 = hashlib.sha256(
        evaluations_content.encode("utf-8")
    ).hexdigest()

    report_content = render_evaluation_report_markdown(result)
    report_sha256 = hashlib.sha256(
        report_content.encode("utf-8")
    ).hexdigest()

    summary_content = render_evaluation_summary_json(
        result, evaluations_sha256, report_sha256
    )

    (output_dir / "evaluations.jsonl").write_text(
        evaluations_content, encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        summary_content, encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        report_content, encoding="utf-8"
    )
