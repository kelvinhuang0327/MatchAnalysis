"""Deterministic prospective prediction admission workflow over real schedule pipeline.

Executes existing P9-P13 schedule identity, materialization, and pregame
eligibility pipeline use cases, derives candidate projections from the exact
same snapshot lineage, and calls P15A1 prospective prediction admission.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable

from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...baseball.domain.match_identity_authority import (
    MatchIdentityAuthorityCatalog,
)
from ...baseball.domain.participant_identity_resolution import (
    ProviderParticipantIdentityMapping,
)
from ...baseball.domain.pregame_eligibility import SchedulePregameEligibilitySet
from ...baseball.domain.prediction_admission import (
    ProspectivePredictionCandidate,
    PredictionAdmissionResult,
    ScheduleCandidateProjection,
    admit_prospective_prediction,
)
from ...baseball.domain.schedule_observation import ScheduleSourceObservation
from ..ports.schedule_observation_source import ScheduleObservationSource
from .build_schedule_observation_revision_chains import (
    build_schedule_observation_revision_chains,
)
from .capture_schedule_observation import capture_schedule_observation
from .construct_match_identities import construct_match_identities
from .evaluate_schedule_pregame_eligibility import (
    evaluate_schedule_pregame_eligibility,
)
from .materialize_schedule_baseball_games import (
    materialize_schedule_baseball_games,
)
from .project_schedule_identity_candidates import (
    project_schedule_identity_candidates,
)
from .resolve_schedule_participant_identities import (
    resolve_schedule_participant_identities,
)
from .select_schedule_observations_as_of import (
    select_schedule_observations_as_of,
)


@dataclass(frozen=True, slots=True)
class ProspectivePredictionAdmissionWorkflowResult:
    """Immutable output of the prospective prediction admission workflow."""

    results: tuple[PredictionAdmissionResult, ...]
    schedule_as_of_utc: datetime
    schedule_candidates: tuple[ScheduleCandidateProjection, ...]
    schedule_pregame_eligibility: SchedulePregameEligibilitySet
    admitted_count: int
    rejected_count: int
    result_set_fingerprint: str


def compute_result_set_fingerprint(
    results: tuple[PredictionAdmissionResult, ...],
) -> str:
    """Compute a deterministic SHA-256 fingerprint for a sequence of results."""

    hasher = sha256()
    for res in results:
        obs_id = res.observation.prediction_observation_id if res.observation else ""
        reason_str = res.reason or ""
        line = f"{res.admission_status}:{reason_str}:{obs_id}\n"
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def run_prospective_prediction_admission_workflow(
    *,
    requests: tuple[ProspectivePredictionCandidate, ...] | Iterable[ProspectivePredictionCandidate],
    raw_schedule_sources: tuple[ScheduleObservationSource, ...] | Iterable[ScheduleObservationSource],
    participant_mappings: tuple[ProviderParticipantIdentityMapping, ...],
    authority_catalog: MatchIdentityAuthorityCatalog,
    schedule_as_of_utc: datetime,
) -> ProspectivePredictionAdmissionWorkflowResult:
    """Orchestrate real P9-P13 schedule pipeline and P15A1 prediction admission."""

    request_tuple = tuple(requests)
    source_tuple = tuple(raw_schedule_sources)

    # 1. Duplicate request identity check fail-closed
    seen_ids: set[str] = set()
    for req in request_tuple:
        if req.prediction_observation_id in seen_ids:
            raise ValueError(
                f"Duplicate prediction request identity in batch: {req.prediction_observation_id}"
            )
        seen_ids.add(req.prediction_observation_id)

    # 2. Capture raw schedule observations sequentially
    captured_observations: list[ScheduleSourceObservation] = []
    observations_by_id: dict[str, ScheduleSourceObservation] = {}
    for source in source_tuple:
        # Check if source specifies a supersedes observation ID
        capture_meta = source.capture()
        prev_obs = None
        if capture_meta.supersedes_observation_id:
            prev_obs = observations_by_id.get(capture_meta.supersedes_observation_id)
        obs = capture_schedule_observation(source, previous_observation=prev_obs)
        captured_observations.append(obs)
        observations_by_id[obs.observation_id] = obs

    # 3. Execute P9-P13 pipeline in lawful order
    revision_set = build_schedule_observation_revision_chains(captured_observations)
    snapshot = select_schedule_observations_as_of(revision_set, schedule_as_of_utc)
    candidate_set = project_schedule_identity_candidates(snapshot)
    resolution_set = resolve_schedule_participant_identities(candidate_set, participant_mappings)
    construction_set = construct_match_identities(resolution_set, authority_catalog)
    materialization_set = materialize_schedule_baseball_games(construction_set, resolution_set)
    eligibility_set = evaluate_schedule_pregame_eligibility(materialization_set)

    # 4. Derive ScheduleCandidateProjection values from the exact same snapshot lineage
    schedule_candidates = tuple(
        ScheduleCandidateProjection(
            provider_namespace=selection.provider_namespace,
            provider_game_id=selection.provider_game_id,
            game_number=selection.game_number,
            source_schedule_observation_id=selection.selected_observation.observation_id,
            schedule_as_of_utc=snapshot.as_of_utc,
            scheduled_start_utc=selection.selected_observation.scheduled_start_utc,
        )
        for selection in snapshot.selections
    )

    # 5. Evaluate prospective predictions against exact admission rules
    results: list[PredictionAdmissionResult] = []
    admitted_count = 0
    rejected_count = 0
    for req in request_tuple:
        res = admit_prospective_prediction(
            req,
            schedule_candidates=schedule_candidates,
            schedule_pregame_eligibility=eligibility_set,
        )
        results.append(res)
        if res.admission_status == "ADMITTED":
            admitted_count += 1
        else:
            rejected_count += 1

    results_tuple = tuple(results)
    fingerprint = compute_result_set_fingerprint(results_tuple)

    return ProspectivePredictionAdmissionWorkflowResult(
        results=results_tuple,
        schedule_as_of_utc=snapshot.as_of_utc,
        schedule_candidates=schedule_candidates,
        schedule_pregame_eligibility=eligibility_set,
        admitted_count=admitted_count,
        rejected_count=rejected_count,
        result_set_fingerprint=fingerprint,
    )
