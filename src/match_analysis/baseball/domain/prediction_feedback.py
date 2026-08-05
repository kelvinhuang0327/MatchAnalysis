"""Pure prediction feedback contracts for the deterministic feedback ledger (P17A).

Joins P15C prediction snapshot, P16A final-result attachment, and P16B evaluation
scorecard into one auditable feedback row per admitted prediction observation.
Does not construct PredictionSourceObservation or FinalResultObservation,
run admission or result attachment, recalculate evaluation policy,
calculate odds/ROI/EV, retrain models, or touch external networks or databases.
"""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import re
from typing import Any


FEEDBACK_ROW_SCHEMA_VERSION = "p17a.prediction_feedback_row.v1"
FEEDBACK_LEDGER_SCHEMA_VERSION = "p17a.prediction_feedback_ledger.v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

FEEDBACK_STATUS_EVALUATED = "EVALUATED"
FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED = "RESULT_ATTACHMENT_REJECTED"
_VALID_FEEDBACK_STATUSES = frozenset({
    FEEDBACK_STATUS_EVALUATED,
    FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED,
})


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256")


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_finite_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")


def compute_feedback_row_fingerprint(
    *,
    prediction_observation_id: str,
    source_snapshot_row_fingerprint: str,
    source_attachment_row_fingerprint: str,
    source_evaluation_row_fingerprint: str | None,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    scheduled_start_utc: str,
    model_id: str,
    market_id: str,
    selection: str,
    model_probability: Decimal,
    result_observation_id: str | None,
    result_observed_at_utc: str | None,
    home_score: int | None,
    away_score: int | None,
    actual_winner: str | None,
    attachment_status: str,
    attachment_rejection_reason: str | None,
    feedback_status: str,
    is_correct: bool | None,
    correctness_target: int | None,
    brier_component: Decimal | None,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for a feedback row."""
    payload = {
        "actual_winner": actual_winner,
        "attachment_rejection_reason": attachment_rejection_reason,
        "attachment_status": attachment_status,
        "away_score": away_score,
        "brier_component": str(brier_component) if brier_component is not None else None,
        "correctness_target": correctness_target,
        "feedback_status": feedback_status,
        "game_number": game_number,
        "home_score": home_score,
        "is_correct": is_correct,
        "market_id": market_id,
        "model_id": model_id,
        "model_probability": str(model_probability),
        "prediction_observation_id": prediction_observation_id,
        "provider_game_id": provider_game_id,
        "provider_namespace": provider_namespace,
        "result_observation_id": result_observation_id,
        "result_observed_at_utc": result_observed_at_utc,
        "scheduled_start_utc": scheduled_start_utc,
        "selection": selection,
        "source_attachment_row_fingerprint": source_attachment_row_fingerprint,
        "source_evaluation_row_fingerprint": source_evaluation_row_fingerprint,
        "source_snapshot_row_fingerprint": source_snapshot_row_fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_feedback_ledger_fingerprint(
    rows: tuple["PredictionFeedbackRow", ...],
) -> str:
    """Compute deterministic SHA-256 fingerprint over ordered feedback rows."""
    parts = [
        f"{row.prediction_observation_id}:{row.feedback_row_fingerprint}\n"
        for row in rows
    ]
    combined = "".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PredictionFeedbackRow:
    """Immutable feedback row joining P15C prediction, P16A attachment, and P16B evaluation."""

    # Lineage
    prediction_observation_id: str
    source_snapshot_row_fingerprint: str
    source_attachment_row_fingerprint: str
    source_evaluation_row_fingerprint: str | None

    # Prediction identity (from P15C observation)
    provider_namespace: str
    provider_game_id: str
    game_number: int
    scheduled_start_utc: str
    model_id: str
    market_id: str
    selection: str
    model_probability: Decimal

    # Original observation payload (immutable copy)
    observation_payload: dict[str, Any]

    # Result attachment (from P16A)
    result_observation_id: str | None
    result_observed_at_utc: str | None
    home_score: int | None
    away_score: int | None
    actual_winner: str | None
    attachment_status: str
    attachment_rejection_reason: str | None

    # Evaluation (from P16B)
    feedback_status: str
    is_correct: bool | None
    correctness_target: int | None
    brier_component: Decimal | None

    # Integrity
    feedback_row_fingerprint: str

    def __post_init__(self) -> None:  # noqa: C901
        # Validate string identity fields
        for field_name in (
            "prediction_observation_id",
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
            "feedback_row_fingerprint",
        ):
            _require_explicit(getattr(self, field_name), field_name)

        # Validate SHA-256 fields
        _require_sha256(self.prediction_observation_id, "prediction_observation_id")
        _require_sha256(self.source_snapshot_row_fingerprint, "source_snapshot_row_fingerprint")
        _require_sha256(self.source_attachment_row_fingerprint, "source_attachment_row_fingerprint")
        _require_sha256(self.feedback_row_fingerprint, "feedback_row_fingerprint")

        if self.source_evaluation_row_fingerprint is not None:
            _require_sha256(
                self.source_evaluation_row_fingerprint,
                "source_evaluation_row_fingerprint",
            )

        _require_positive_integer(self.game_number, "game_number")

        _require_finite_decimal(self.model_probability, "model_probability")
        if not (Decimal("0") <= self.model_probability <= Decimal("1")):
            raise ValueError("model_probability must be within [0, 1]")

        if not isinstance(self.observation_payload, dict):
            raise TypeError("observation_payload must be a dict")

        if self.feedback_status not in _VALID_FEEDBACK_STATUSES:
            raise ValueError(
                f"feedback_status must be one of {sorted(_VALID_FEEDBACK_STATUSES)}, "
                f"got {self.feedback_status!r}"
            )

        # Status-specific validation
        if self.feedback_status == FEEDBACK_STATUS_EVALUATED:
            self._validate_evaluated()
        elif self.feedback_status == FEEDBACK_STATUS_RESULT_ATTACHMENT_REJECTED:
            self._validate_rejected()

        # Verify fingerprint
        expected_fp = compute_feedback_row_fingerprint(
            prediction_observation_id=self.prediction_observation_id,
            source_snapshot_row_fingerprint=self.source_snapshot_row_fingerprint,
            source_attachment_row_fingerprint=self.source_attachment_row_fingerprint,
            source_evaluation_row_fingerprint=self.source_evaluation_row_fingerprint,
            provider_namespace=self.provider_namespace,
            provider_game_id=self.provider_game_id,
            game_number=self.game_number,
            scheduled_start_utc=self.scheduled_start_utc,
            model_id=self.model_id,
            market_id=self.market_id,
            selection=self.selection,
            model_probability=self.model_probability,
            result_observation_id=self.result_observation_id,
            result_observed_at_utc=self.result_observed_at_utc,
            home_score=self.home_score,
            away_score=self.away_score,
            actual_winner=self.actual_winner,
            attachment_status=self.attachment_status,
            attachment_rejection_reason=self.attachment_rejection_reason,
            feedback_status=self.feedback_status,
            is_correct=self.is_correct,
            correctness_target=self.correctness_target,
            brier_component=self.brier_component,
        )
        if self.feedback_row_fingerprint != expected_fp:
            raise ValueError(
                f"feedback_row_fingerprint mismatch: computed {expected_fp}, "
                f"got {self.feedback_row_fingerprint}"
            )

    def _validate_evaluated(self) -> None:
        """Validate EVALUATED status constraints."""
        if self.attachment_status != "ATTACHED":
            raise ValueError(
                "EVALUATED feedback requires attachment_status == 'ATTACHED'"
            )
        if self.attachment_rejection_reason is not None:
            raise ValueError(
                "EVALUATED feedback must have attachment_rejection_reason == null"
            )
        if self.source_evaluation_row_fingerprint is None:
            raise ValueError(
                "EVALUATED feedback must have source_evaluation_row_fingerprint"
            )

        # Result fields must be present
        for field_name in ("result_observation_id", "result_observed_at_utc", "actual_winner"):
            if getattr(self, field_name) is None:
                raise ValueError(
                    f"EVALUATED feedback must have non-null {field_name}"
                )
        for field_name in ("home_score", "away_score"):
            val = getattr(self, field_name)
            if val is None:
                raise ValueError(
                    f"EVALUATED feedback must have non-null {field_name}"
                )
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"{field_name} must be an integer")

        # Evaluation fields must be present
        if not isinstance(self.is_correct, bool):
            raise TypeError("EVALUATED feedback must have boolean is_correct")
        if isinstance(self.correctness_target, bool) or not isinstance(
            self.correctness_target, int
        ):
            raise TypeError("EVALUATED feedback must have integer correctness_target")
        if self.brier_component is None:
            raise ValueError("EVALUATED feedback must have non-null brier_component")
        _require_finite_decimal(self.brier_component, "brier_component")

    def _validate_rejected(self) -> None:
        """Validate RESULT_ATTACHMENT_REJECTED status constraints."""
        if self.attachment_status != "REJECTED":
            raise ValueError(
                "RESULT_ATTACHMENT_REJECTED feedback requires attachment_status == 'REJECTED'"
            )
        if self.attachment_rejection_reason is None:
            raise ValueError(
                "RESULT_ATTACHMENT_REJECTED feedback must have non-null attachment_rejection_reason"
            )
        _require_explicit(self.attachment_rejection_reason, "attachment_rejection_reason")

        if self.source_evaluation_row_fingerprint is not None:
            raise ValueError(
                "RESULT_ATTACHMENT_REJECTED feedback must have source_evaluation_row_fingerprint == null"
            )

        # Result fields must be null
        for field_name in (
            "result_observation_id",
            "result_observed_at_utc",
            "home_score",
            "away_score",
            "actual_winner",
        ):
            if getattr(self, field_name) is not None:
                raise ValueError(
                    f"RESULT_ATTACHMENT_REJECTED feedback must have null {field_name}"
                )

        # Evaluation fields must be null
        if self.is_correct is not None:
            raise ValueError(
                "RESULT_ATTACHMENT_REJECTED feedback must have null is_correct"
            )
        if self.correctness_target is not None:
            raise ValueError(
                "RESULT_ATTACHMENT_REJECTED feedback must have null correctness_target"
            )
        if self.brier_component is not None:
            raise ValueError(
                "RESULT_ATTACHMENT_REJECTED feedback must have null brier_component"
            )
