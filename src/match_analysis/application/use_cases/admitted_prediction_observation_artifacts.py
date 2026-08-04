"""Deterministic artifact rendering for admitted prediction observation snapshot.

Emits admitted_observations.jsonl, summary.json, and report.md without
runtime timestamps.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from .build_admitted_prediction_observation_snapshot import (
    AdmittedPredictionObservationSnapshotResult,
)


def _snapshot_row_to_dict(
    row_prediction_observation_id: str,
    row_source_result_row_fingerprint: str,
    row_observation: dict[str, Any],
    row_snapshot_row_fingerprint: str,
) -> dict[str, Any]:
    """Convert a snapshot row to a JSON-serializable dictionary."""
    return {
        "prediction_observation_id": row_prediction_observation_id,
        "source_result_row_fingerprint": row_source_result_row_fingerprint,
        "observation": row_observation,
        "snapshot_row_fingerprint": row_snapshot_row_fingerprint,
    }


def render_admitted_observations_jsonl(
    result: AdmittedPredictionObservationSnapshotResult,
) -> str:
    """Render deterministic admitted_observations.jsonl content."""
    lines = [
        json.dumps(
            _snapshot_row_to_dict(
                row.prediction_observation_id,
                row.source_result_row_fingerprint,
                row.observation,
                row.snapshot_row_fingerprint,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in result.snapshot_rows
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def render_snapshot_summary_json(
    result: AdmittedPredictionObservationSnapshotResult,
    admitted_observations_jsonl_sha256: str,
    report_sha256: str,
) -> str:
    """Render deterministic summary.json content."""
    sorted_observation_ids = [
        row.prediction_observation_id for row in result.snapshot_rows
    ]
    summary_data = {
        "schema_version": result.schema_version,
        "source_results_sha256": result.source_results_sha256,
        "source_summary_sha256": result.source_summary_sha256,
        "source_result_set_fingerprint": result.source_result_set_fingerprint,
        "source_row_count": result.source_row_count,
        "source_admitted_count": result.source_admitted_count,
        "source_rejected_count": result.source_rejected_count,
        "snapshot_row_count": len(result.snapshot_rows),
        "sorted_observation_ids": sorted_observation_ids,
        "snapshot_fingerprint": result.snapshot_fingerprint,
        "admitted_observations_jsonl_sha256": admitted_observations_jsonl_sha256,
        "report_sha256": report_sha256,
        "claims": result.claims,
    }
    return json.dumps(summary_data, indent=2, sort_keys=True) + "\n"


def render_snapshot_report_markdown(
    result: AdmittedPredictionObservationSnapshotResult,
) -> str:
    """Render deterministic report.md markdown content."""
    lines = [
        "# Admitted Prediction Observation Snapshot Report",
        "",
        "## Source",
        "",
        f"- **Source P15B1 Result Set Fingerprint**: `{result.source_result_set_fingerprint}`",
        f"- **Source Results SHA-256**: `{result.source_results_sha256}`",
        f"- **Source Summary SHA-256**: `{result.source_summary_sha256}`",
        f"- **Source Admitted Count**: `{result.source_admitted_count}`",
        f"- **Source Rejected Count**: `{result.source_rejected_count}`",
        "",
        "## Snapshot",
        "",
        f"- **Schema Version**: `{result.schema_version}`",
        f"- **Snapshot Row Count**: `{len(result.snapshot_rows)}`",
        f"- **Snapshot Fingerprint**: `{result.snapshot_fingerprint}`",
        "",
        "## Admitted Observations",
        "",
        "| Observation ID | Provider Namespace | Provider Game ID | Game Number | Model ID | Market ID | Selection | Scheduled Start (UTC) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.snapshot_rows:
        obs = row.observation
        scheduled_start = obs.get("scheduled_start_utc") or "N/A"
        lines.append(
            f"| `{row.prediction_observation_id[:16]}...` "
            f"| `{obs.get('provider_namespace', 'N/A')}` "
            f"| `{obs.get('provider_game_id', 'N/A')}` "
            f"| `{obs.get('game_number', 'N/A')}` "
            f"| `{obs.get('model_id', 'N/A')}` "
            f"| `{obs.get('market_id', 'N/A')}` "
            f"| `{obs.get('selection', 'N/A')}` "
            f"| `{scheduled_start}` |"
        )

    lines.extend([
        "",
        "## Safety Limitations",
        "",
        "- This snapshot contains **only** admitted prediction observations.",
        "- No outcomes have been attached.",
        "- No provider or network calls were made.",
        "- No database writes occurred.",
        "- No deployment actions were taken.",
        "- No betting claims are made.",
        "- Legacy P83E rows are excluded.",
        "",
    ])
    return "\n".join(lines)


def write_admitted_prediction_observation_artifacts(
    output_dir: Path,
    result: AdmittedPredictionObservationSnapshotResult,
) -> None:
    """Write admitted_observations.jsonl, summary.json, and report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    observations_content = render_admitted_observations_jsonl(result)
    observations_sha256 = hashlib.sha256(
        observations_content.encode("utf-8")
    ).hexdigest()
    report_content = render_snapshot_report_markdown(result)
    report_sha256 = hashlib.sha256(
        report_content.encode("utf-8")
    ).hexdigest()
    summary_content = render_snapshot_summary_json(
        result, observations_sha256, report_sha256,
    )

    (output_dir / "admitted_observations.jsonl").write_text(
        observations_content, encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        summary_content, encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        report_content, encoding="utf-8",
    )
