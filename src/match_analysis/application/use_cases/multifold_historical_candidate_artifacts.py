"""Deterministic P21B batch artifact rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .prediction_feedback_artifacts import render_feedback_jsonl
from .replay_multifold_historical_candidates import (
    MultifoldHistoricalCandidateReplayResult,
    P20B_HISTORICAL_RUNTIME_COMPLIANCE,
    P21B_SCHEMA_VERSION,
)


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def render_multifold_feedback_jsonl(
    result: MultifoldHistoricalCandidateReplayResult,
) -> str:
    """Render the aggregate P17 feedback rows."""

    return render_feedback_jsonl(result.feedback_result)


def render_multifold_assessments_jsonl(
    result: MultifoldHistoricalCandidateReplayResult,
) -> str:
    """Render one P21A assessment for every aggregate feedback row."""

    rows = []
    for assessment in result.assessments:
        rows.append(
            {
                "assessment_id": assessment.assessment_id,
                "assessment_schema_version": "p21a.learning_candidate_assessment.v1",
                "assessment_status": assessment.status,
                "candidate_id": assessment.candidate_id,
                "claims": dict(result.claims),
                "eligibility_contract_version": (
                    "p21a.non_synthetic_learning_candidate_eligibility.v1"
                ),
                "exclusion_reasons": list(assessment.exclusion_reasons),
                "feedback_row_fingerprint": assessment.feedback_row_fingerprint,
                "prediction_observation_id": assessment.prediction_observation_id,
                "source_feedback_ledger_fingerprint": (
                    result.feedback_result.feedback_ledger_fingerprint
                ),
            }
        )
    return _jsonl(rows)


def render_multifold_candidates_jsonl(
    result: MultifoldHistoricalCandidateReplayResult,
) -> str:
    """Render eligible candidates in deterministic candidate-id order."""

    return _jsonl(list(result.candidates))


def render_multifold_summary_json(
    result: MultifoldHistoricalCandidateReplayResult,
    *,
    feedback_jsonl_sha256: str,
    assessments_jsonl_sha256: str,
    candidates_jsonl_sha256: str,
    report_sha256: str,
) -> str:
    """Render the aggregate manifest and explicit non-training claims."""

    summary = result.to_projection()
    summary.update(
        {
            "feedback_jsonl_sha256": feedback_jsonl_sha256,
            "assessments_jsonl_sha256": assessments_jsonl_sha256,
            "learning_candidates_jsonl_sha256": candidates_jsonl_sha256,
            "report_sha256": report_sha256,
            "p20b_historical_runtime_compliance": (
                P20B_HISTORICAL_RUNTIME_COMPLIANCE
            ),
        }
    )
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_multifold_report_markdown(
    result: MultifoldHistoricalCandidateReplayResult,
) -> str:
    """Render a compact human-readable P21B evidence report."""

    lines = [
        "# P21B Contiguous Multifold Historical Candidate Replay",
        "",
        "This is a bounded, historical, paper-only replay. It is not a training",
        "dataset, model-promotion event, profitability analysis, or betting recommendation.",
        "",
        "## Fold sequence",
        "",
        "| Fold | Training cutoff | Training rows | Prediction games | Max parity difference | Parity |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for fold in result.folds:
        lines.append(
            f"| `{fold.fold_id}` | `{fold.train_as_of}` | {fold.training_row_count} "
            f"| {fold.prediction_row_count} | `{fold.max_absolute_difference}` "
            f"| `{fold.parity_passed}` |"
        )
    lines.extend(
        [
            "",
            "## Lineage counts",
            "",
            f"- P15C admissions: `{result.p15c_admission_count}`",
            f"- P16A attached results: `{result.p16a_attachment_count}`",
            f"- P16B evaluations: `{result.p16b_evaluation_count}`",
            f"- P17 feedback rows: `{len(result.feedback_rows)}`",
            f"- P21A assessments: `{len(result.assessments)}`",
            f"- P21A eligible candidates: `{result.p21a_eligible_count}`",
            f"- P21A excluded assessments: `{result.p21a_excluded_count}`",
            "",
            "## Deterministic identities",
            "",
            f"- Membership SHA-256: `{result.membership_sha256}`",
            f"- Historical result rows SHA-256: `{result.result_rows_sha256}`",
            f"- Aggregate P17 ledger fingerprint: `{result.feedback_result.feedback_ledger_fingerprint}`",
            f"- Aggregate P21A assessment fingerprint: `{result.assessment_semantic_fingerprint}`",
            f"- Aggregate candidate fingerprint: `{result.candidate_semantic_fingerprint}`",
            "",
            "## Historical provenance",
            "",
            f"- Repository: `{result.historical_provenance['source_repository']}`",
            f"- Commit: `{result.historical_provenance['source_commit']}`",
            f"- Tree: `{result.historical_provenance['source_tree']}`",
            f"- Archive: `{result.historical_provenance['source_archive_path']}`",
            f"- Member: `{result.historical_provenance['source_member']}`",
            f"- Result rows: `{result.historical_result_count}`",
            "",
            "## Safety claims",
            "",
            "- `historical=true`, `sample_limited=true`, `synthetic_results=false`",
            "- `training_dataset_claim=false`, `training_authorized=false`, `retraining_performed=false`",
            "- `model_promoted=false`, `profitability_claim=false`, `real_betting_recommendation=false`",
            "- No provider, network, database, odds, deployment, push, or remote CI action was used.",
            "- P20B historical runtime compliance remains REFUTED; this task preserves that prior finding.",
            "",
        ]
    )
    return "\n".join(lines)


def write_multifold_historical_candidate_artifacts(
    output_dir: Path,
    result: MultifoldHistoricalCandidateReplayResult,
) -> None:
    """Write exactly the deterministic P21B report artifacts."""

    feedback_content = render_multifold_feedback_jsonl(result)
    assessments_content = render_multifold_assessments_jsonl(result)
    candidates_content = render_multifold_candidates_jsonl(result)
    report_content = render_multifold_report_markdown(result)
    summary_content = render_multifold_summary_json(
        result,
        feedback_jsonl_sha256=hashlib.sha256(feedback_content.encode("utf-8")).hexdigest(),
        assessments_jsonl_sha256=hashlib.sha256(
            assessments_content.encode("utf-8")
        ).hexdigest(),
        candidates_jsonl_sha256=hashlib.sha256(
            candidates_content.encode("utf-8")
        ).hexdigest(),
        report_sha256=hashlib.sha256(report_content.encode("utf-8")).hexdigest(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feedback.jsonl").write_text(feedback_content, encoding="utf-8")
    (output_dir / "assessments.jsonl").write_text(assessments_content, encoding="utf-8")
    (output_dir / "learning_candidates.jsonl").write_text(
        candidates_content,
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(summary_content, encoding="utf-8")
    (output_dir / "report.md").write_text(report_content, encoding="utf-8")


__all__ = (
    "render_multifold_assessments_jsonl",
    "render_multifold_candidates_jsonl",
    "render_multifold_feedback_jsonl",
    "render_multifold_report_markdown",
    "render_multifold_summary_json",
    "write_multifold_historical_candidate_artifacts",
)
