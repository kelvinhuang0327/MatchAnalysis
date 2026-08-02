"""Pure prospective prediction admission evaluation and exact schedule resolution.

This is the only module authorized to construct PredictionSourceObservation.
No legacy, quarantine, adapter, repository, CLI, or application module may
import or construct it directly. Resolution is exact-key only: it never
matches by date, teams, participants, start-time proximity, or row order.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import re

from .canonical_utc import parse_canonical_utc
from .pregame_eligibility import SchedulePregameEligibilitySet
from .prediction_source_observation import (
    PredictionSourceObservation,
    compute_prediction_observation_id,
)


ADMITTED = "ADMITTED"
REJECTED = "REJECTED"

MISSING_REQUIRED_PREDICTION_EVIDENCE = "MISSING_REQUIRED_PREDICTION_EVIDENCE"
INVALID_CANONICAL_UTC = "INVALID_CANONICAL_UTC"
INVALID_PREDICTION_TIMESTAMP_ORDER = "INVALID_PREDICTION_TIMESTAMP_ORDER"
PREDICTION_NOT_BEFORE_SCHEDULED_START = "PREDICTION_NOT_BEFORE_SCHEDULED_START"
MISSING_SCHEDULE_CANDIDATE_MATCH = "MISSING_SCHEDULE_CANDIDATE_MATCH"
AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH = "AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH"
SCHEDULE_OBSERVATION_ID_MISMATCH = "SCHEDULE_OBSERVATION_ID_MISMATCH"
SCHEDULE_NOT_PREGAME_ELIGIBLE = "SCHEDULE_NOT_PREGAME_ELIGIBLE"
EXACT_IDENTITY_MISMATCH = "EXACT_IDENTITY_MISMATCH"

CONTROLLED_ADMISSION_REJECTION_REASONS = frozenset(
    {
        MISSING_REQUIRED_PREDICTION_EVIDENCE,
        INVALID_CANONICAL_UTC,
        INVALID_PREDICTION_TIMESTAMP_ORDER,
        PREDICTION_NOT_BEFORE_SCHEDULED_START,
        MISSING_SCHEDULE_CANDIDATE_MATCH,
        AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH,
        SCHEDULE_OBSERVATION_ID_MISMATCH,
        SCHEDULE_NOT_PREGAME_ELIGIBLE,
        EXACT_IDENTITY_MISMATCH,
    }
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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


def _require_utc(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be normalized to UTC")


@dataclass(frozen=True, slots=True)
class ScheduleCandidateProjection:
    """A narrow, exact-match schedule candidate for prediction admission."""

    provider_namespace: str
    provider_game_id: str
    game_number: int
    source_schedule_observation_id: str
    schedule_as_of_utc: datetime
    scheduled_start_utc: datetime

    def __post_init__(self) -> None:
        _require_explicit(self.provider_namespace, "provider_namespace")
        _require_explicit(self.provider_game_id, "provider_game_id")
        _require_positive_integer(self.game_number, "game_number")
        _require_sha256(
            self.source_schedule_observation_id,
            "source_schedule_observation_id",
        )
        _require_utc(self.schedule_as_of_utc, "schedule_as_of_utc")
        _require_utc(self.scheduled_start_utc, "scheduled_start_utc")


@dataclass(frozen=True, slots=True)
class ProspectivePredictionCandidate:
    """Untrusted raw evidence for one prospective prediction admission attempt.

    Fields are intentionally unvalidated here: admit_prospective_prediction
    classifies missing or malformed evidence into a controlled, fail-closed
    rejection reason rather than raising.
    """

    prediction_observation_id: str
    source_prediction_id: str
    model_id: str
    market_id: str
    selection: str
    model_probability: Decimal
    line_value: Decimal
    push_policy: str
    provider_namespace: str
    provider_game_id: str
    game_number: int
    source_schedule_observation_id: str
    prediction_generated_at_utc: str
    response_received_at_utc: str
    ingested_at_utc: str


@dataclass(frozen=True, slots=True)
class PredictionAdmissionResult:
    """Immutable outcome of one prospective prediction admission attempt."""

    admission_status: str
    reason: str | None
    observation: PredictionSourceObservation | None

    def __post_init__(self) -> None:
        if self.admission_status not in (ADMITTED, REJECTED):
            raise ValueError("admission_status must be ADMITTED or REJECTED")
        if self.admission_status == ADMITTED:
            if self.reason is not None:
                raise ValueError("an admitted result must not carry a reason")
            if not isinstance(self.observation, PredictionSourceObservation):
                raise TypeError(
                    "an admitted result must carry a PredictionSourceObservation"
                )
        else:
            if self.reason not in CONTROLLED_ADMISSION_REJECTION_REASONS:
                raise ValueError(
                    "reason must be a controlled admission rejection reason"
                )
            if self.observation is not None:
                raise ValueError("a rejected result must not carry an observation")


def _schedule_candidate_key(
    candidate: ScheduleCandidateProjection,
) -> tuple[str, str, int]:
    return (
        candidate.provider_namespace,
        candidate.provider_game_id,
        candidate.game_number,
    )


def resolve_exact_schedule_candidate(
    *,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    schedule_candidates: tuple[ScheduleCandidateProjection, ...],
) -> tuple[ScheduleCandidateProjection | None, str | None]:
    """Resolve the unique exact schedule candidate, or a fail-closed reason.

    The only matching key is (provider_namespace, provider_game_id,
    game_number). This never matches by date, teams, participants,
    start-time proximity, or row order.
    """

    if not isinstance(schedule_candidates, tuple) or any(
        not isinstance(candidate, ScheduleCandidateProjection)
        for candidate in schedule_candidates
    ):
        raise TypeError(
            "schedule_candidates must be a tuple of ScheduleCandidateProjection"
        )

    query_key = (provider_namespace, provider_game_id, game_number)
    matches = tuple(
        candidate
        for candidate in schedule_candidates
        if _schedule_candidate_key(candidate) == query_key
    )
    if not matches:
        return None, MISSING_SCHEDULE_CANDIDATE_MATCH
    if len(matches) > 1:
        return None, AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH
    return matches[0], None


def _has_required_evidence(candidate: ProspectivePredictionCandidate) -> bool:
    token_fields = (
        candidate.source_prediction_id,
        candidate.model_id,
        candidate.market_id,
        candidate.selection,
        candidate.push_policy,
        candidate.provider_namespace,
        candidate.provider_game_id,
        candidate.source_schedule_observation_id,
    )
    for value in token_fields:
        if not isinstance(value, str) or not value or value != value.strip():
            return False

    if (
        isinstance(candidate.game_number, bool)
        or not isinstance(candidate.game_number, int)
        or candidate.game_number <= 0
    ):
        return False

    for value in (candidate.model_probability, candidate.line_value):
        if not isinstance(value, Decimal) or not value.is_finite():
            return False
    if not (Decimal("0") <= candidate.model_probability <= Decimal("1")):
        return False

    for value in (
        candidate.prediction_generated_at_utc,
        candidate.response_received_at_utc,
        candidate.ingested_at_utc,
    ):
        if not isinstance(value, str) or not value:
            return False

    return True


def admit_prospective_prediction(
    candidate: ProspectivePredictionCandidate,
    *,
    schedule_candidates: tuple[ScheduleCandidateProjection, ...],
    schedule_pregame_eligibility: SchedulePregameEligibilitySet,
) -> PredictionAdmissionResult:
    """Evaluate one prospective prediction against exact, fail-closed rules."""

    if not isinstance(candidate, ProspectivePredictionCandidate):
        raise TypeError("candidate must be a ProspectivePredictionCandidate")
    if not isinstance(schedule_pregame_eligibility, SchedulePregameEligibilitySet):
        raise TypeError(
            "schedule_pregame_eligibility must be a"
            " SchedulePregameEligibilitySet"
        )

    def rejected(reason: str) -> PredictionAdmissionResult:
        return PredictionAdmissionResult(
            admission_status=REJECTED, reason=reason, observation=None
        )

    if not _has_required_evidence(candidate):
        return rejected(MISSING_REQUIRED_PREDICTION_EVIDENCE)

    try:
        generated_at = parse_canonical_utc(candidate.prediction_generated_at_utc)
        received_at = parse_canonical_utc(candidate.response_received_at_utc)
        ingested_at = parse_canonical_utc(candidate.ingested_at_utc)
    except ValueError:
        return rejected(INVALID_CANONICAL_UTC)

    if not (generated_at <= received_at <= ingested_at):
        return rejected(INVALID_PREDICTION_TIMESTAMP_ORDER)

    resolved, resolution_reason = resolve_exact_schedule_candidate(
        provider_namespace=candidate.provider_namespace,
        provider_game_id=candidate.provider_game_id,
        game_number=candidate.game_number,
        schedule_candidates=schedule_candidates,
    )
    if resolved is None:
        return rejected(resolution_reason)

    if (
        candidate.source_schedule_observation_id
        != resolved.source_schedule_observation_id
    ):
        return rejected(SCHEDULE_OBSERVATION_ID_MISMATCH)

    eligible_observation_ids = {
        decision.materialization.source_observation_id
        for decision in schedule_pregame_eligibility.eligible_decisions
    }
    if resolved.source_schedule_observation_id not in eligible_observation_ids:
        return rejected(SCHEDULE_NOT_PREGAME_ELIGIBLE)

    if not (ingested_at < resolved.scheduled_start_utc):
        return rejected(PREDICTION_NOT_BEFORE_SCHEDULED_START)

    expected_observation_id = compute_prediction_observation_id(
        source_prediction_id=candidate.source_prediction_id,
        model_id=candidate.model_id,
        market_id=candidate.market_id,
        selection=candidate.selection,
        model_probability=candidate.model_probability,
        line_value=candidate.line_value,
        push_policy=candidate.push_policy,
        provider_namespace=candidate.provider_namespace,
        provider_game_id=candidate.provider_game_id,
        game_number=candidate.game_number,
        source_schedule_observation_id=candidate.source_schedule_observation_id,
        prediction_generated_at_utc=generated_at,
        response_received_at_utc=received_at,
        ingested_at_utc=ingested_at,
        scheduled_start_utc=resolved.scheduled_start_utc,
    )
    if candidate.prediction_observation_id != expected_observation_id:
        return rejected(EXACT_IDENTITY_MISMATCH)

    observation = PredictionSourceObservation(
        prediction_observation_id=candidate.prediction_observation_id,
        source_prediction_id=candidate.source_prediction_id,
        model_id=candidate.model_id,
        market_id=candidate.market_id,
        selection=candidate.selection,
        model_probability=candidate.model_probability,
        line_value=candidate.line_value,
        push_policy=candidate.push_policy,
        provider_namespace=candidate.provider_namespace,
        provider_game_id=candidate.provider_game_id,
        game_number=candidate.game_number,
        source_schedule_observation_id=candidate.source_schedule_observation_id,
        prediction_generated_at_utc=generated_at,
        response_received_at_utc=received_at,
        ingested_at_utc=ingested_at,
        scheduled_start_utc=resolved.scheduled_start_utc,
    )
    return PredictionAdmissionResult(
        admission_status=ADMITTED, reason=None, observation=observation
    )
