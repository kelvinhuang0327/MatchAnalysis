"""Generate deterministic P19A Moneyline predictions for P15C admission."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256

from ...baseball.domain.canonical_utc import (
    format_canonical_utc,
    parse_canonical_utc,
)
from ...baseball.domain.moneyline_feature_snapshot import (
    MoneylineFeatureSnapshot,
)
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
from ...baseball.domain.prediction_admission import (
    PredictionAdmissionResult,
    ProspectivePredictionCandidate,
    ScheduleCandidateProjection,
    admit_prospective_prediction,
)
from ...baseball.domain.prediction_source_observation import (
    compute_prediction_observation_id,
)
from ...baseball.domain.pregame_eligibility import SchedulePregameEligibilitySet


@dataclass(frozen=True, slots=True)
class MoneylineInferenceResult:
    """Immutable generated candidates and optional P15C admission results."""

    candidates: tuple[ProspectivePredictionCandidate, ...]
    admissions: tuple[PredictionAdmissionResult, ...]
    model_artifact_fingerprint: str
    candidate_set_fingerprint: str


def _validate_generation_times(
    snapshot: MoneylineFeatureSnapshot,
    *,
    prediction_generated_at_utc: datetime,
    response_received_at_utc: datetime,
    ingested_at_utc: datetime,
) -> None:
    if not (
        snapshot.as_of_utc
        <= prediction_generated_at_utc
        <= response_received_at_utc
        <= ingested_at_utc
        < snapshot.scheduled_start_utc
    ):
        raise ValueError(
            "prediction timestamps must satisfy snapshot as_of_utc <= "
            "generated <= received <= ingested < scheduled_start_utc"
        )


def _candidate_for(
    snapshot: MoneylineFeatureSnapshot,
    artifact: MoneylineModelArtifact,
    *,
    selection: str,
    probability: Decimal,
    prediction_generated_at_utc: datetime,
    response_received_at_utc: datetime,
    ingested_at_utc: datetime,
) -> ProspectivePredictionCandidate:
    source_prediction_id = (
        f"p19a:{snapshot.fingerprint()}:{selection.lower()}"
    )
    line_value = Decimal("0")
    push_policy = "NO_PUSH"
    prediction_observation_id = compute_prediction_observation_id(
        source_prediction_id=source_prediction_id,
        model_id=artifact.model_id,
        market_id="moneyline",
        selection=selection,
        model_probability=probability,
        line_value=line_value,
        push_policy=push_policy,
        provider_namespace=snapshot.provider_namespace,
        provider_game_id=snapshot.provider_game_id,
        game_number=snapshot.game_number,
        source_schedule_observation_id=snapshot.source_schedule_observation_id,
        prediction_generated_at_utc=prediction_generated_at_utc,
        response_received_at_utc=response_received_at_utc,
        ingested_at_utc=ingested_at_utc,
        scheduled_start_utc=snapshot.scheduled_start_utc,
    )
    return ProspectivePredictionCandidate(
        prediction_observation_id=prediction_observation_id,
        source_prediction_id=source_prediction_id,
        model_id=artifact.model_id,
        market_id="moneyline",
        selection=selection,
        model_probability=probability,
        line_value=line_value,
        push_policy=push_policy,
        provider_namespace=snapshot.provider_namespace,
        provider_game_id=snapshot.provider_game_id,
        game_number=snapshot.game_number,
        source_schedule_observation_id=snapshot.source_schedule_observation_id,
        prediction_generated_at_utc=format_canonical_utc(
            prediction_generated_at_utc
        ),
        response_received_at_utc=format_canonical_utc(response_received_at_utc),
        ingested_at_utc=format_canonical_utc(ingested_at_utc),
    )


def _candidate_set_fingerprint(
    candidates: tuple[ProspectivePredictionCandidate, ...],
) -> str:
    payload = "".join(
        f"{candidate.prediction_observation_id}:{candidate.selection}:"
        f"{candidate.model_probability}\n"
        for candidate in candidates
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def generate_moneyline_predictions(
    snapshots: tuple[MoneylineFeatureSnapshot, ...],
    model_artifact: MoneylineModelArtifact,
    *,
    prediction_generated_at_utc: str | datetime,
    response_received_at_utc: str | datetime,
    ingested_at_utc: str | datetime,
    schedule_candidates: tuple[ScheduleCandidateProjection, ...] | None = None,
    schedule_pregame_eligibility: SchedulePregameEligibilitySet | None = None,
) -> MoneylineInferenceResult:
    """Generate HOME/AWAY Moneyline candidates, optionally admitting through P15C."""

    if not isinstance(snapshots, tuple) or any(
        not isinstance(snapshot, MoneylineFeatureSnapshot)
        for snapshot in snapshots
    ):
        raise TypeError("snapshots must be a tuple of MoneylineFeatureSnapshot")
    if not isinstance(model_artifact, MoneylineModelArtifact):
        raise TypeError("model_artifact must be a MoneylineModelArtifact")
    if (schedule_candidates is None) != (schedule_pregame_eligibility is None):
        raise ValueError(
            "schedule_candidates and schedule_pregame_eligibility must be supplied together"
        )

    def as_utc(value: str | datetime, field_name: str) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = parse_canonical_utc(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware")
        return parsed.astimezone(timezone.utc)

    generated = as_utc(prediction_generated_at_utc, "prediction_generated_at_utc")
    received = as_utc(response_received_at_utc, "response_received_at_utc")
    ingested = as_utc(ingested_at_utc, "ingested_at_utc")
    candidates_list: list[ProspectivePredictionCandidate] = []
    for snapshot in snapshots:
        _validate_generation_times(
            snapshot,
            prediction_generated_at_utc=generated,
            response_received_at_utc=received,
            ingested_at_utc=ingested,
        )
        home_probability = model_artifact.predict_home_probability(snapshot)
        candidates_list.extend(
            (
                _candidate_for(
                    snapshot,
                    model_artifact,
                    selection="HOME",
                    probability=home_probability,
                    prediction_generated_at_utc=generated,
                    response_received_at_utc=received,
                    ingested_at_utc=ingested,
                ),
                _candidate_for(
                    snapshot,
                    model_artifact,
                    selection="AWAY",
                    probability=Decimal("1") - home_probability,
                    prediction_generated_at_utc=generated,
                    response_received_at_utc=received,
                    ingested_at_utc=ingested,
                ),
            )
        )

    candidates = tuple(candidates_list)
    admissions: tuple[PredictionAdmissionResult, ...] = ()
    if schedule_candidates is not None and schedule_pregame_eligibility is not None:
        admissions = tuple(
            admit_prospective_prediction(
                candidate,
                schedule_candidates=schedule_candidates,
                schedule_pregame_eligibility=schedule_pregame_eligibility,
            )
            for candidate in candidates
        )
    return MoneylineInferenceResult(
        candidates=candidates,
        admissions=admissions,
        model_artifact_fingerprint=model_artifact.fingerprint(),
        candidate_set_fingerprint=_candidate_set_fingerprint(candidates),
    )
