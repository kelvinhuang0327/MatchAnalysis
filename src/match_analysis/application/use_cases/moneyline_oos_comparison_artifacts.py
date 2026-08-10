"""Deterministic P23A comparison artifact rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.moneyline_oos_comparison import (
    PairedMoneylineComparison,
    canonical_json_bytes,
)


P23A_SUMMARY_SCHEMA_VERSION = "p23a.moneyline_strictly_future_oos_summary.v1"
P23A_INCUMBENT_ARTIFACT_SCHEMA_VERSION = (
    "p23a.moneyline_strictly_future_incumbent_artifact.v1"
)


def render_comparisons_jsonl(rows: Sequence[PairedMoneylineComparison]) -> str:
    return "".join(
        json.dumps(
            row.to_projection(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for row in rows
    )


def render_incumbent_model_artifact(
    artifact_projection: Mapping[str, Any],
    *,
    source_fold_id: str,
    source_fold_fingerprint: str,
    training_cutoff: str,
    training_row_count: int,
    model_fidelity_route: str,
) -> str:
    projection = dict(artifact_projection)
    projection.update(
        {
            "artifact_schema_version": P23A_INCUMBENT_ARTIFACT_SCHEMA_VERSION,
            "artifact_role": "INCUMBENT",
            "claims": {
                "model_promoted": False,
                "out_of_sample_evaluated": True,
                "production_ready": False,
                "profitability_claim": False,
                "real_betting_recommendation": False,
                "promotion_authorized": False,
                "retraining_performed": False,
            },
            "source_fold_id": source_fold_id,
            "source_fold_fingerprint": source_fold_fingerprint,
            "training_cutoff": training_cutoff,
            "training_row_count": training_row_count,
            "model_fidelity_route": model_fidelity_route,
            "artifact_fingerprint": artifact_projection.get("artifact_fingerprint"),
        }
    )
    return json.dumps(projection, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_summary(summary: Mapping[str, Any]) -> str:
    projection = {"schema_version": P23A_SUMMARY_SCHEMA_VERSION, **summary}
    return json.dumps(projection, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_comparison_artifacts(
    output_dir: str | Path,
    *,
    rows: Sequence[PairedMoneylineComparison],
    summary: Mapping[str, Any],
    incumbent_artifact_projection: Mapping[str, Any],
    source_fold_id: str,
    source_fold_fingerprint: str,
    training_cutoff: str,
    training_row_count: int,
    model_fidelity_route: str,
) -> None:
    """Write exactly the three authorized P23A report files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparisons.jsonl").write_text(
        render_comparisons_jsonl(rows), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        render_summary(summary), encoding="utf-8"
    )
    (directory / "incumbent_model_artifact.json").write_text(
        render_incumbent_model_artifact(
            incumbent_artifact_projection,
            source_fold_id=source_fold_id,
            source_fold_fingerprint=source_fold_fingerprint,
            training_cutoff=training_cutoff,
            training_row_count=training_row_count,
            model_fidelity_route=model_fidelity_route,
        ),
        encoding="utf-8",
    )


__all__ = (
    "P23A_INCUMBENT_ARTIFACT_SCHEMA_VERSION",
    "P23A_SUMMARY_SCHEMA_VERSION",
    "render_comparisons_jsonl",
    "render_incumbent_model_artifact",
    "render_summary",
    "write_comparison_artifacts",
)
