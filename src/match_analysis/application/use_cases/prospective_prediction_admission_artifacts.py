"""Deterministic artifact rendering for prospective prediction admission workflow.

Emits results.jsonl, summary.json, and report.md without runtime timestamps.
"""

from collections import Counter
from decimal import Decimal
import json
from typing import Any

from ...baseball.domain.prediction_admission import PredictionAdmissionResult
from ...baseball.domain.prediction_source_observation import PredictionSourceObservation
from .run_prospective_prediction_admission_workflow import (
    ProspectivePredictionAdmissionWorkflowResult,
)


EXPLICIT_WORKFLOW_CLAIMS = {
    "provider_called": False,
    "db_written": False,
    "legacy_rows_admitted": False,
    "deployed": False,
    "betting_claim": False,
}


def observation_to_dict(obs: PredictionSourceObservation) -> dict[str, Any]:
    """Convert a PredictionSourceObservation to a JSON-serializable dictionary."""
    return {
        "prediction_observation_id": obs.prediction_observation_id,
        "source_prediction_id": obs.source_prediction_id,
        "model_id": obs.model_id,
        "market_id": obs.market_id,
        "selection": obs.selection,
        "model_probability": str(obs.model_probability),
        "line_value": str(obs.line_value),
        "push_policy": obs.push_policy,
        "provider_namespace": obs.provider_namespace,
        "provider_game_id": obs.provider_game_id,
        "game_number": obs.game_number,
        "source_schedule_observation_id": obs.source_schedule_observation_id,
        "prediction_generated_at_utc": obs.prediction_generated_at_utc.isoformat().replace("+00:00", "Z"),
        "response_received_at_utc": obs.response_received_at_utc.isoformat().replace("+00:00", "Z"),
        "ingested_at_utc": obs.ingested_at_utc.isoformat().replace("+00:00", "Z"),
        "scheduled_start_utc": obs.scheduled_start_utc.isoformat().replace("+00:00", "Z"),
    }


def result_to_dict(result: PredictionAdmissionResult, index: int) -> dict[str, Any]:
    """Convert a PredictionAdmissionResult to a JSON-serializable dictionary."""
    return {
        "request_index": index,
        "admission_status": result.admission_status,
        "reason": result.reason,
        "observation": observation_to_dict(result.observation) if result.observation else None,
    }


def render_results_jsonl(results: tuple[PredictionAdmissionResult, ...]) -> str:
    """Render deterministic results.jsonl content."""
    lines = [
        json.dumps(result_to_dict(res, i + 1), separators=(",", ":"))
        for i, res in enumerate(results)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def render_summary_json(
    workflow_result: ProspectivePredictionAdmissionWorkflowResult,
    input_hashes: dict[str, str],
) -> str:
    """Render deterministic summary.json content."""
    rejection_counter = Counter(
        res.reason for res in workflow_result.results if res.reason is not None
    )
    summary_data = {
        "schedule_as_of_utc": workflow_result.schedule_as_of_utc.isoformat().replace("+00:00", "Z"),
        "request_count": len(workflow_result.results),
        "admitted_count": workflow_result.admitted_count,
        "rejected_count": workflow_result.rejected_count,
        "rejection_reasons_breakdown": dict(sorted(rejection_counter.items())),
        "result_set_fingerprint": workflow_result.result_set_fingerprint,
        "input_hashes": input_hashes,
        "claims": EXPLICIT_WORKFLOW_CLAIMS,
    }
    return json.dumps(summary_data, indent=2, sort_keys=True) + "\n"


def render_report_markdown(
    workflow_result: ProspectivePredictionAdmissionWorkflowResult,
    input_hashes: dict[str, str],
) -> str:
    """Render deterministic report.md markdown content."""
    as_of_str = workflow_result.schedule_as_of_utc.isoformat().replace("+00:00", "Z")
    rejection_counter = Counter(
        res.reason for res in workflow_result.results if res.reason is not None
    )

    lines = [
        "# Prospective Prediction Admission Workflow Report",
        "",
        "## Summary",
        "",
        f"- **Schedule As-Of (UTC)**: `{as_of_str}`",
        f"- **Total Requests**: `{len(workflow_result.results)}`",
        f"- **Admitted**: `{workflow_result.admitted_count}`",
        f"- **Rejected**: `{workflow_result.rejected_count}`",
        f"- **Result Set Fingerprint**: `{workflow_result.result_set_fingerprint}`",
        "",
        "## Input Hashes (SHA-256)",
        "",
    ]
    for key, val in sorted(input_hashes.items()):
        lines.append(f"- **{key}**: `{val}`")

    lines.extend([
        "",
        "## Rejection Reasons Breakdown",
        "",
    ])
    if rejection_counter:
        lines.append("| Rejection Reason | Count |")
        lines.append("| --- | --- |")
        for reason, count in sorted(rejection_counter.items()):
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("No rejections occurred.")

    lines.extend([
        "",
        "## Request Results",
        "",
        "| Index | Admission Status | Reason | Provider Game ID |",
        "| --- | --- | --- | --- |",
    ])
    for i, res in enumerate(workflow_result.results, start=1):
        game_id = res.observation.provider_game_id if res.observation else "N/A"
        reason_str = f"`{res.reason}`" if res.reason else "None"
        lines.append(f"| {i} | `{res.admission_status}` | {reason_str} | `{game_id}` |")

    lines.extend([
        "",
        "## Explicit System Claims",
        "",
        "- **Provider Called**: `false`",
        "- **DB Written**: `false`",
        "- **Legacy Rows Admitted**: `false`",
        "- **Deployed**: `false`",
        "- **Betting Claim**: `false`",
        "",
    ])
    return "\n".join(lines)


def write_prospective_prediction_admission_artifacts(
    output_dir: Path,
    workflow_result: ProspectivePredictionAdmissionWorkflowResult,
    input_hashes: dict[str, str],
) -> None:
    """Write results.jsonl, summary.json, and report.md to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_content = render_results_jsonl(workflow_result.results)
    summary_content = render_summary_json(workflow_result, input_hashes)
    report_content = render_report_markdown(workflow_result, input_hashes)

    (output_dir / "results.jsonl").write_text(results_content, encoding="utf-8")
    (output_dir / "summary.json").write_text(summary_content, encoding="utf-8")
    (output_dir / "report.md").write_text(report_content, encoding="utf-8")
