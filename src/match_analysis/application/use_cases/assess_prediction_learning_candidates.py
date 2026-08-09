"""Assess committed P20B feedback rows for the P21A learning boundary.

The source feedback ledger is read-only.  Structural corruption aborts the
complete assessment; valid rows are classified independently and fail closed.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any

from ...baseball.domain.prediction_feedback import (
    compute_feedback_ledger_fingerprint,
    compute_feedback_row_fingerprint,
)
from ...baseball.domain.prediction_learning_eligibility import (
    ASSESSMENT_SCHEMA_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    ELIGIBILITY_CONTRACT_VERSION,
    ELIGIBLE,
    PredictionLearningEligibilityAssessment,
    assess_prediction_learning_eligibility,
)


SOURCE_ARTIFACT_INVALID_STOP = "STOP_MATCHANALYSIS_P21A_SOURCE_ARTIFACT_INVALID"
NO_REAL_POSITIVE_PATH_STOP = "STOP_MATCHANALYSIS_P21A_NO_REAL_POSITIVE_PATH"
SOURCE_SCHEMA_VERSION = "p20b.first_non_synthetic_historical_feedback.v1"
EXPECTED_SOURCE_ROW_COUNT = 4

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FEEDBACK_FIELDS = frozenset(
    {
        "actual_winner",
        "attachment_rejection_reason",
        "attachment_status",
        "away_score",
        "brier_component",
        "correctness_target",
        "feedback_row_fingerprint",
        "feedback_status",
        "game_number",
        "home_score",
        "is_correct",
        "market_id",
        "model_id",
        "model_probability",
        "observation_payload",
        "prediction_observation_id",
        "provider_game_id",
        "provider_namespace",
        "result_observation_id",
        "result_observed_at_utc",
        "scheduled_start_utc",
        "selection",
        "source_attachment_row_fingerprint",
        "source_evaluation_row_fingerprint",
        "source_snapshot_row_fingerprint",
    }
)


class PredictionLearningCandidateSourceError(ValueError):
    """Raised when the committed P20B source cannot be trusted structurally."""


@dataclass(frozen=True, slots=True)
class _FingerprintPair:
    prediction_observation_id: str
    feedback_row_fingerprint: str


@dataclass(frozen=True, slots=True)
class PredictionLearningCandidateAssessmentResult:
    """Immutable P21A assessment and candidate export result."""

    assessment_schema_version: str
    candidate_schema_version: str
    eligibility_contract_version: str
    source_feedback_jsonl_sha256: str
    source_feedback_summary_sha256: str
    source_feedback_ledger_fingerprint: str
    source_row_count: int
    assessments: tuple[PredictionLearningEligibilityAssessment, ...]
    candidates: tuple[dict[str, Any], ...]
    exclusion_reason_counts: dict[str, int]
    claims: dict[str, bool]
    assessments_semantic_fingerprint: str
    candidates_semantic_fingerprint: str


def _invalid(message: str) -> PredictionLearningCandidateSourceError:
    return PredictionLearningCandidateSourceError(
        f"{SOURCE_ARTIFACT_INVALID_STOP}: {message}"
    )


class _InvalidSourceSentinel(Exception):
    """Internal marker used to preserve duplicate-key stop-token errors."""


def _parse_json_object(raw: str, context: str) -> dict[str, Any]:  # type: ignore[no-redef]
    def object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _InvalidSourceSentinel(
                    f"{SOURCE_ARTIFACT_INVALID_STOP}: duplicate JSON key {key!r} in {context}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=object_pairs_hook)
    except _InvalidSourceSentinel as exc:
        raise PredictionLearningCandidateSourceError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise _invalid(f"malformed JSON in {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise _invalid(f"{context} must be a JSON object")
    return value


def _require_sha256(value: Any, field_name: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise _invalid(f"{field_name} must be a lowercase 64-character SHA-256")


def _require_string_or_none(value: Any, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise _invalid(f"{field_name} must be a string or null")


def _parse_feedback_rows(feedback_bytes: bytes) -> list[dict[str, Any]]:
    try:
        raw = feedback_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _invalid(f"feedback JSONL is not UTF-8: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise _invalid(f"blank feedback JSONL line at {line_number}")
        row = _parse_json_object(line, f"feedback row {line_number}")
        if set(row) != _REQUIRED_FEEDBACK_FIELDS:
            missing = sorted(_REQUIRED_FEEDBACK_FIELDS - set(row))
            extra = sorted(set(row) - _REQUIRED_FEEDBACK_FIELDS)
            raise _invalid(
                f"feedback row {line_number} schema mismatch; missing={missing}, extra={extra}"
            )
        rows.append(row)
    if not rows:
        raise _invalid("feedback JSONL contains no rows")
    return rows


def _validate_feedback_row(row: dict[str, Any], row_number: int) -> None:
    for field_name in (
        "prediction_observation_id",
        "feedback_row_fingerprint",
        "source_snapshot_row_fingerprint",
        "source_attachment_row_fingerprint",
        "provider_namespace",
        "provider_game_id",
        "scheduled_start_utc",
        "model_id",
        "market_id",
        "selection",
        "attachment_status",
        "feedback_status",
    ):
        if not isinstance(row[field_name], str):
            raise _invalid(f"feedback row {row_number} field {field_name} must be a string")
    _require_sha256(row["prediction_observation_id"], "prediction_observation_id")
    _require_sha256(row["feedback_row_fingerprint"], "feedback_row_fingerprint")
    _require_sha256(row["source_snapshot_row_fingerprint"], "source_snapshot_row_fingerprint")
    _require_sha256(row["source_attachment_row_fingerprint"], "source_attachment_row_fingerprint")
    _require_sha256(
        row["source_evaluation_row_fingerprint"],
        "source_evaluation_row_fingerprint",
        allow_none=True,
    )
    _require_string_or_none(row["attachment_rejection_reason"], "attachment_rejection_reason")
    _require_string_or_none(row["result_observation_id"], "result_observation_id")
    _require_string_or_none(row["result_observed_at_utc"], "result_observed_at_utc")
    _require_string_or_none(row["actual_winner"], "actual_winner")
    if not isinstance(row["observation_payload"], dict):
        raise _invalid(f"feedback row {row_number} observation_payload must be an object")
    if isinstance(row["game_number"], bool) or not isinstance(row["game_number"], int):
        raise _invalid(f"feedback row {row_number} game_number must be an integer")
    for field_name in ("home_score", "away_score", "correctness_target"):
        value = row[field_name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise _invalid(f"feedback row {row_number} {field_name} must be an integer or null")
    if row["is_correct"] is not None and not isinstance(row["is_correct"], bool):
        raise _invalid(f"feedback row {row_number} is_correct must be boolean or null")
    if not isinstance(row["model_probability"], str):
        raise _invalid(f"feedback row {row_number} model_probability must be a string")
    try:
        Decimal(row["model_probability"])
    except (InvalidOperation, ValueError) as exc:
        raise _invalid(f"feedback row {row_number} model_probability is not decimal text") from exc
    if row["brier_component"] is not None:
        if not isinstance(row["brier_component"], str):
            raise _invalid(f"feedback row {row_number} brier_component must be a string or null")
        try:
            brier = Decimal(row["brier_component"])
        except (InvalidOperation, ValueError) as exc:
            raise _invalid(f"feedback row {row_number} brier_component is not decimal text") from exc
        if not brier.is_finite():
            raise _invalid(f"feedback row {row_number} brier_component must be finite")
    if row["feedback_status"] not in {"EVALUATED", "RESULT_ATTACHMENT_REJECTED"}:
        raise _invalid(f"feedback row {row_number} has unsupported feedback_status")
    if row["attachment_status"] not in {"ATTACHED", "REJECTED"}:
        raise _invalid(f"feedback row {row_number} has unsupported attachment_status")

    expected = compute_feedback_row_fingerprint(
        prediction_observation_id=row["prediction_observation_id"],
        source_snapshot_row_fingerprint=row["source_snapshot_row_fingerprint"],
        source_attachment_row_fingerprint=row["source_attachment_row_fingerprint"],
        source_evaluation_row_fingerprint=row["source_evaluation_row_fingerprint"],
        provider_namespace=row["provider_namespace"],
        provider_game_id=row["provider_game_id"],
        game_number=row["game_number"],
        scheduled_start_utc=row["scheduled_start_utc"],
        model_id=row["model_id"],
        market_id=row["market_id"],
        selection=row["selection"],
        model_probability=Decimal(row["model_probability"]),
        result_observation_id=row["result_observation_id"],
        result_observed_at_utc=row["result_observed_at_utc"],
        home_score=row["home_score"],
        away_score=row["away_score"],
        actual_winner=row["actual_winner"],
        attachment_status=row["attachment_status"],
        attachment_rejection_reason=row["attachment_rejection_reason"],
        feedback_status=row["feedback_status"],
        is_correct=row["is_correct"],
        correctness_target=row["correctness_target"],
        brier_component=(
            Decimal(row["brier_component"])
            if row["brier_component"] is not None
            else None
        ),
    )
    if expected != row["feedback_row_fingerprint"]:
        raise _invalid(
            f"feedback row {row_number} feedback_row_fingerprint mismatch; expected {expected}"
        )


def _validate_source_summary(
    *,
    summary_bytes: bytes,
    feedback_bytes: bytes,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        raw = summary_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _invalid(f"feedback summary is not UTF-8: {exc}") from exc
    summary = _parse_json_object(raw, "P20B feedback summary")
    if summary.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise _invalid("unexpected P20B feedback summary schema_version")
    claims = summary.get("claims")
    if not isinstance(claims, dict):
        raise _invalid("P20B feedback summary claims must be an object")
    required_claims = {
        "synthetic_results": bool,
        "sample_limited": bool,
        "training_dataset_claim": bool,
        "training_authorized": bool,
        "retraining_performed": bool,
        "model_promoted": bool,
    }
    for key, expected_type in required_claims.items():
        if key not in claims or type(claims[key]) is not expected_type:
            raise _invalid(f"P20B feedback summary claim {key!r} must be boolean")
    if claims["sample_limited"] is not True:
        raise _invalid("P20B feedback source must remain sample_limited")
    for key in (
        "training_dataset_claim",
        "training_authorized",
        "retraining_performed",
        "model_promoted",
    ):
        if claims[key] is not False:
            raise _invalid(f"P20B feedback source claim {key!r} must be false")
    if not isinstance(summary.get("feedback_row_count"), int):
        raise _invalid("P20B feedback summary feedback_row_count must be an integer")
    if summary["feedback_row_count"] != len(rows):
        raise _invalid("P20B feedback summary feedback_row_count mismatch")
    if summary["feedback_row_count"] != EXPECTED_SOURCE_ROW_COUNT:
        raise _invalid(
            f"P20B feedback source row count must be {EXPECTED_SOURCE_ROW_COUNT}"
        )
    feedback_sha256 = hashlib.sha256(feedback_bytes).hexdigest()
    if summary.get("feedback_jsonl_sha256") != feedback_sha256:
        raise _invalid("P20B feedback_jsonl_sha256 mismatch")
    summary_ledger = summary.get("feedback_ledger_fingerprint")
    _require_sha256(summary_ledger, "feedback_ledger_fingerprint")
    sorted_pairs = tuple(
        _FingerprintPair(
            prediction_observation_id=row["prediction_observation_id"],
            feedback_row_fingerprint=row["feedback_row_fingerprint"],
        )
        for row in sorted(rows, key=lambda item: item["prediction_observation_id"])
    )
    computed_ledger = compute_feedback_ledger_fingerprint(sorted_pairs)
    if computed_ledger != summary_ledger:
        raise _invalid("P20B feedback_ledger_fingerprint mismatch")
    return summary


def _semantic_fingerprint(values: list[dict[str, Any]]) -> str:
    canonical = "".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        for value in values
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assessment_to_dict(
    assessment: PredictionLearningEligibilityAssessment,
    *,
    source_feedback_ledger_fingerprint: str,
    claims: dict[str, bool],
) -> dict[str, Any]:
    return {
        "assessment_id": assessment.assessment_id,
        "assessment_schema_version": ASSESSMENT_SCHEMA_VERSION,
        "assessment_status": assessment.status,
        "candidate_id": assessment.candidate_id,
        "claims": dict(claims),
        "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
        "exclusion_reasons": list(assessment.exclusion_reasons),
        "feedback_row_fingerprint": assessment.feedback_row_fingerprint,
        "prediction_observation_id": assessment.prediction_observation_id,
        "source_feedback_ledger_fingerprint": source_feedback_ledger_fingerprint,
    }


def assess_prediction_learning_candidates(
    *,
    feedback_bytes: bytes,
    feedback_summary_bytes: bytes,
) -> PredictionLearningCandidateAssessmentResult:
    """Validate P20B and return deterministic row assessments and candidates."""
    rows = _parse_feedback_rows(feedback_bytes)
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        _validate_feedback_row(row, row_number)
        prediction_id = row["prediction_observation_id"]
        fingerprint = row["feedback_row_fingerprint"]
        if prediction_id in seen_ids:
            raise _invalid(f"duplicate prediction_observation_id: {prediction_id}")
        if fingerprint in seen_fingerprints:
            raise _invalid(f"duplicate feedback_row_fingerprint: {fingerprint}")
        seen_ids.add(prediction_id)
        seen_fingerprints.add(fingerprint)

    summary = _validate_source_summary(
        summary_bytes=feedback_summary_bytes,
        feedback_bytes=feedback_bytes,
        rows=rows,
    )
    source_ledger_fingerprint = summary["feedback_ledger_fingerprint"]
    source_claims = summary["claims"]
    claims = {
        "synthetic_results": source_claims["synthetic_results"],
        "non_synthetic": not source_claims["synthetic_results"],
        "sample_limited": True,
        "training_dataset_claim": False,
        "training_authorized": False,
        "retraining_performed": False,
        "model_promoted": False,
        "profitability_claim": False,
        "real_betting_recommendation": False,
    }

    assessments = tuple(
        assess_prediction_learning_eligibility(
            row,
            synthetic_results=source_claims["synthetic_results"],
            source_feedback_ledger_fingerprint=source_ledger_fingerprint,
        )
        for row in sorted(rows, key=lambda item: item["prediction_observation_id"])
    )
    eligible_assessments = tuple(
        assessment for assessment in assessments if assessment.status == ELIGIBLE
    )
    if not eligible_assessments:
        raise ValueError(
            f"{NO_REAL_POSITIVE_PATH_STOP}: no committed P20B row is ELIGIBLE"
        )

    candidates_by_id: list[dict[str, Any]] = []
    rows_by_prediction_id = {row["prediction_observation_id"]: row for row in rows}
    for assessment in eligible_assessments:
        row = rows_by_prediction_id[assessment.prediction_observation_id]
        candidate = dict(row)
        candidate.update(
            {
                "assessment_id": assessment.assessment_id,
                "assessment_status": assessment.status,
                "candidate_id": assessment.candidate_id,
                "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
                "claims": dict(claims),
                "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
                "source_feedback_fingerprint": row["feedback_row_fingerprint"],
                "source_feedback_ledger_fingerprint": source_ledger_fingerprint,
            }
        )
        candidates_by_id.append(candidate)
    candidates = tuple(sorted(candidates_by_id, key=lambda item: item["candidate_id"]))

    assessment_dicts = [
        _assessment_to_dict(
            assessment,
            source_feedback_ledger_fingerprint=source_ledger_fingerprint,
            claims=claims,
        )
        for assessment in assessments
    ]
    exclusion_reason_counts: dict[str, int] = {}
    for assessment in assessments:
        for reason in assessment.exclusion_reasons:
            exclusion_reason_counts[reason] = exclusion_reason_counts.get(reason, 0) + 1

    candidate_semantic_values = [
        {
            "candidate_id": candidate["candidate_id"],
            "feedback_row_fingerprint": candidate["feedback_row_fingerprint"],
            "source_feedback": {
                key: value
                for key, value in candidate.items()
                if key in _REQUIRED_FEEDBACK_FIELDS
            },
        }
        for candidate in candidates
    ]
    return PredictionLearningCandidateAssessmentResult(
        assessment_schema_version=ASSESSMENT_SCHEMA_VERSION,
        candidate_schema_version=CANDIDATE_SCHEMA_VERSION,
        eligibility_contract_version=ELIGIBILITY_CONTRACT_VERSION,
        source_feedback_jsonl_sha256=hashlib.sha256(feedback_bytes).hexdigest(),
        source_feedback_summary_sha256=hashlib.sha256(feedback_summary_bytes).hexdigest(),
        source_feedback_ledger_fingerprint=source_ledger_fingerprint,
        source_row_count=len(rows),
        assessments=assessments,
        candidates=candidates,
        exclusion_reason_counts=dict(sorted(exclusion_reason_counts.items())),
        claims=claims,
        assessments_semantic_fingerprint=_semantic_fingerprint(assessment_dicts),
        candidates_semantic_fingerprint=_semantic_fingerprint(candidate_semantic_values),
    )


__all__ = (
    "EXPECTED_SOURCE_ROW_COUNT",
    "NO_REAL_POSITIVE_PATH_STOP",
    "PredictionLearningCandidateAssessmentResult",
    "PredictionLearningCandidateSourceError",
    "SOURCE_ARTIFACT_INVALID_STOP",
    "assess_prediction_learning_candidates",
)
