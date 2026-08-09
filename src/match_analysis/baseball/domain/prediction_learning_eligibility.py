"""Pure P21A eligibility rules for non-synthetic learning candidates.

This module only classifies already-evaluated P20B feedback rows.  It does not
train models, authorize training, promote models, calculate betting metrics, or
touch providers, databases, or other external systems.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Any


ELIGIBILITY_CONTRACT_VERSION = "p21a.non_synthetic_learning_candidate_eligibility.v1"
ASSESSMENT_SCHEMA_VERSION = "p21a.learning_candidate_assessment.v1"
CANDIDATE_SCHEMA_VERSION = "p21a.non_synthetic_learning_candidate.v1"

ELIGIBLE = "ELIGIBLE"
EXCLUDED = "EXCLUDED"

SYNTHETIC_RESULT_EVIDENCE_EXCLUDED = "SYNTHETIC_RESULT_EVIDENCE_EXCLUDED"
FEEDBACK_NOT_EVALUATED = "FEEDBACK_NOT_EVALUATED"
RESULT_ATTACHMENT_REJECTED = "RESULT_ATTACHMENT_REJECTED"
MISSING_RESULT_EVIDENCE = "MISSING_RESULT_EVIDENCE"
MISSING_EVALUATION_EVIDENCE = "MISSING_EVALUATION_EVIDENCE"
UNSUPPORTED_MARKET = "UNSUPPORTED_MARKET"
UNSUPPORTED_SELECTION = "UNSUPPORTED_SELECTION"
INVALID_MODEL_PROBABILITY = "INVALID_MODEL_PROBABILITY"
INCOMPLETE_LINEAGE = "INCOMPLETE_LINEAGE"

CONTROLLED_EXCLUSION_REASONS = frozenset(
    {
        SYNTHETIC_RESULT_EVIDENCE_EXCLUDED,
        FEEDBACK_NOT_EVALUATED,
        RESULT_ATTACHMENT_REJECTED,
        MISSING_RESULT_EVIDENCE,
        MISSING_EVALUATION_EVIDENCE,
        UNSUPPORTED_MARKET,
        UNSUPPORTED_SELECTION,
        INVALID_MODEL_PROBABILITY,
        INCOMPLETE_LINEAGE,
    }
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compute_learning_candidate_id(
    *,
    prediction_observation_id: str,
    feedback_row_fingerprint: str,
) -> str:
    """Return an identity derived only from immutable source identity and version."""
    canonical = json.dumps(
        {
            "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
            "feedback_row_fingerprint": feedback_row_fingerprint,
            "prediction_observation_id": prediction_observation_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    return _sha256_text(canonical)


def compute_assessment_id(
    *,
    prediction_observation_id: str,
    feedback_row_fingerprint: str,
) -> str:
    """Return a deterministic identity for one source-row assessment."""
    canonical = json.dumps(
        {
            "assessment_schema_version": ASSESSMENT_SCHEMA_VERSION,
            "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
            "feedback_row_fingerprint": feedback_row_fingerprint,
            "prediction_observation_id": prediction_observation_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    return _sha256_text(canonical)


def is_valid_model_probability(value: Any) -> bool:
    """Check the P17A probability representation and its inclusive [0, 1] bound."""
    if not isinstance(value, str):
        return False
    try:
        probability = Decimal(value)
    except (InvalidOperation, ValueError):
        return False
    return probability.is_finite() and Decimal("0") <= probability <= Decimal("1")


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _has_prediction_lineage(row: dict[str, Any]) -> bool:
    observation = row.get("observation_payload")
    if not isinstance(observation, dict):
        return False
    required_top_level = (
        "prediction_observation_id",
        "source_snapshot_row_fingerprint",
        "source_attachment_row_fingerprint",
        "provider_namespace",
        "provider_game_id",
        "scheduled_start_utc",
        "model_id",
    )
    if not all(_present(row.get(field)) for field in required_top_level):
        return False
    required_observation = (
        "prediction_observation_id",
        "source_prediction_id",
        "source_schedule_observation_id",
        "provider_namespace",
        "provider_game_id",
        "scheduled_start_utc",
        "model_id",
        "market_id",
        "selection",
    )
    return all(_present(observation.get(field)) for field in required_observation)


@dataclass(frozen=True, slots=True)
class PredictionLearningEligibilityAssessment:
    """One deterministic, fail-closed assessment for a P20B feedback row."""

    prediction_observation_id: str
    feedback_row_fingerprint: str
    assessment_id: str
    status: str
    exclusion_reasons: tuple[str, ...]
    candidate_id: str | None

    def __post_init__(self) -> None:
        if self.status not in {ELIGIBLE, EXCLUDED}:
            raise ValueError(f"Unknown assessment status: {self.status!r}")
        if tuple(sorted(set(self.exclusion_reasons))) != self.exclusion_reasons:
            raise ValueError("exclusion_reasons must be unique and sorted")
        if any(reason not in CONTROLLED_EXCLUSION_REASONS for reason in self.exclusion_reasons):
            raise ValueError("exclusion_reasons contains an uncontrolled reason")
        if self.status == ELIGIBLE:
            if self.exclusion_reasons or self.candidate_id is None:
                raise ValueError("ELIGIBLE assessment must have no reasons and a candidate_id")
        elif self.candidate_id is not None or not self.exclusion_reasons:
            raise ValueError("EXCLUDED assessment must have reasons and no candidate_id")


def assess_prediction_learning_eligibility(
    row: dict[str, Any],
    *,
    synthetic_results: bool,
    source_feedback_ledger_fingerprint: str,
) -> PredictionLearningEligibilityAssessment:
    """Classify one structurally valid source row without inferring missing data."""
    if not source_feedback_ledger_fingerprint:
        raise ValueError("source_feedback_ledger_fingerprint must be explicit")
    reasons: set[str] = set()

    if synthetic_results:
        reasons.add(SYNTHETIC_RESULT_EVIDENCE_EXCLUDED)

    feedback_status = row["feedback_status"]
    attachment_status = row["attachment_status"]
    is_evaluated = feedback_status == "EVALUATED"
    is_attached = attachment_status == "ATTACHED"

    if not is_evaluated:
        reasons.add(FEEDBACK_NOT_EVALUATED)
    if not is_attached or row["attachment_rejection_reason"] is not None:
        reasons.add(RESULT_ATTACHMENT_REJECTED)

    if is_evaluated:
        result_fields = (
            "result_observation_id",
            "result_observed_at_utc",
            "home_score",
            "away_score",
            "actual_winner",
        )
        if any(row[field] is None for field in result_fields):
            reasons.add(MISSING_RESULT_EVIDENCE)

        evaluation_fields = (
            "source_evaluation_row_fingerprint",
            "is_correct",
            "correctness_target",
            "brier_component",
        )
        if any(row[field] is None for field in evaluation_fields):
            reasons.add(MISSING_EVALUATION_EVIDENCE)

    if row["market_id"] != "moneyline":
        reasons.add(UNSUPPORTED_MARKET)
    if row["selection"] not in {"HOME", "AWAY"}:
        reasons.add(UNSUPPORTED_SELECTION)
    if not is_valid_model_probability(row["model_probability"]):
        reasons.add(INVALID_MODEL_PROBABILITY)

    if not _has_prediction_lineage(row):
        reasons.add(INCOMPLETE_LINEAGE)
    if is_evaluated and (
        not _present(row.get("result_observation_id"))
        or not _present(row.get("source_evaluation_row_fingerprint"))
    ):
        reasons.add(INCOMPLETE_LINEAGE)

    ordered_reasons = tuple(sorted(reasons))
    candidate_id = None
    if not ordered_reasons:
        candidate_id = compute_learning_candidate_id(
            prediction_observation_id=row["prediction_observation_id"],
            feedback_row_fingerprint=row["feedback_row_fingerprint"],
        )
    return PredictionLearningEligibilityAssessment(
        prediction_observation_id=row["prediction_observation_id"],
        feedback_row_fingerprint=row["feedback_row_fingerprint"],
        assessment_id=compute_assessment_id(
            prediction_observation_id=row["prediction_observation_id"],
            feedback_row_fingerprint=row["feedback_row_fingerprint"],
        ),
        status=ELIGIBLE if not ordered_reasons else EXCLUDED,
        exclusion_reasons=ordered_reasons,
        candidate_id=candidate_id,
    )


__all__ = (
    "ASSESSMENT_SCHEMA_VERSION",
    "CANDIDATE_SCHEMA_VERSION",
    "CONTROLLED_EXCLUSION_REASONS",
    "ELIGIBILITY_CONTRACT_VERSION",
    "ELIGIBLE",
    "EXCLUDED",
    "FEEDBACK_NOT_EVALUATED",
    "INCOMPLETE_LINEAGE",
    "INVALID_MODEL_PROBABILITY",
    "MISSING_EVALUATION_EVIDENCE",
    "MISSING_RESULT_EVIDENCE",
    "PredictionLearningEligibilityAssessment",
    "RESULT_ATTACHMENT_REJECTED",
    "SYNTHETIC_RESULT_EVIDENCE_EXCLUDED",
    "UNSUPPORTED_MARKET",
    "UNSUPPORTED_SELECTION",
    "assess_prediction_learning_eligibility",
    "compute_assessment_id",
    "compute_learning_candidate_id",
    "is_valid_model_probability",
)
