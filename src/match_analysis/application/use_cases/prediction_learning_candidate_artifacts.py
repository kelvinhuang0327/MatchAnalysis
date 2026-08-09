"""Deterministic P21A assessment and learning-candidate artifact rendering."""

import hashlib
import json
from pathlib import Path
from typing import Any

from .assess_prediction_learning_candidates import (
    PredictionLearningCandidateAssessmentResult,
)


def _assessment_to_dict(assessment: Any, result: PredictionLearningCandidateAssessmentResult) -> dict[str, Any]:
    claims = dict(result.claims)
    return {
        "assessment_id": assessment.assessment_id,
        "assessment_schema_version": result.assessment_schema_version,
        "assessment_status": assessment.status,
        "candidate_id": assessment.candidate_id,
        "claims": claims,
        "eligibility_contract_version": result.eligibility_contract_version,
        "exclusion_reasons": list(assessment.exclusion_reasons),
        "feedback_row_fingerprint": assessment.feedback_row_fingerprint,
        "prediction_observation_id": assessment.prediction_observation_id,
        "source_feedback_ledger_fingerprint": result.source_feedback_ledger_fingerprint,
    }


def render_assessments_jsonl(result: PredictionLearningCandidateAssessmentResult) -> str:
    """Render one canonical assessment for every source feedback row."""
    rows = [
        _assessment_to_dict(assessment, result)
        for assessment in result.assessments
    ]
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def render_learning_candidates_jsonl(
    result: PredictionLearningCandidateAssessmentResult,
) -> str:
    """Render only ELIGIBLE candidates in deterministic candidate-id order."""
    return "".join(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n"
        for candidate in result.candidates
    )


def render_prediction_learning_candidate_summary_json(
    result: PredictionLearningCandidateAssessmentResult,
    *,
    assessments_jsonl_sha256: str,
    learning_candidates_jsonl_sha256: str,
) -> str:
    """Render the deterministic P21A summary and non-training claims."""
    summary = {
        "assessment_schema_version": result.assessment_schema_version,
        "assessments_jsonl_sha256": assessments_jsonl_sha256,
        "assessments_semantic_fingerprint": result.assessments_semantic_fingerprint,
        "candidate_schema_version": result.candidate_schema_version,
        "candidates_semantic_fingerprint": result.candidates_semantic_fingerprint,
        "claims": dict(result.claims),
        "eligible_count": sum(assessment.status == "ELIGIBLE" for assessment in result.assessments),
        "eligible_candidate_ids": [candidate["candidate_id"] for candidate in result.candidates],
        "excluded_count": sum(assessment.status == "EXCLUDED" for assessment in result.assessments),
        "exclusion_reason_counts": result.exclusion_reason_counts,
        "feedback_ledger_fingerprint": result.source_feedback_ledger_fingerprint,
        "learning_candidates_jsonl_sha256": learning_candidates_jsonl_sha256,
        "schema_version": "p21a.non_synthetic_learning_candidates.v1",
        "source_feedback_jsonl_sha256": result.source_feedback_jsonl_sha256,
        "source_feedback_summary_sha256": result.source_feedback_summary_sha256,
        "source_feedback_row_count": result.source_row_count,
    }
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


def write_prediction_learning_candidate_artifacts(
    output_dir: Path,
    result: PredictionLearningCandidateAssessmentResult,
) -> None:
    """Write exactly the three deterministic P21A artifacts."""
    assessments_content = render_assessments_jsonl(result)
    candidates_content = render_learning_candidates_jsonl(result)
    assessments_sha256 = hashlib.sha256(assessments_content.encode("utf-8")).hexdigest()
    candidates_sha256 = hashlib.sha256(candidates_content.encode("utf-8")).hexdigest()
    summary_content = render_prediction_learning_candidate_summary_json(
        result,
        assessments_jsonl_sha256=assessments_sha256,
        learning_candidates_jsonl_sha256=candidates_sha256,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessments.jsonl").write_text(assessments_content, encoding="utf-8")
    (output_dir / "learning_candidates.jsonl").write_text(
        candidates_content,
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(summary_content, encoding="utf-8")


__all__ = (
    "render_assessments_jsonl",
    "render_learning_candidates_jsonl",
    "render_prediction_learning_candidate_summary_json",
    "write_prediction_learning_candidate_artifacts",
)
