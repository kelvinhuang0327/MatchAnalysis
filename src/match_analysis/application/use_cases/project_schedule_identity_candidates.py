"""Project a P8 snapshot into provider-scoped identity-resolution requests."""

from ...baseball.domain.schedule_identity_candidate import (
    SCHEDULE_IDENTITY_RESOLUTION_CANDIDATE_SET_SCHEMA_VERSION,
    ScheduleIdentityResolutionCandidate,
    ScheduleIdentityResolutionCandidateSet,
    compute_schedule_identity_resolution_candidate_set_fingerprint,
)
from ...baseball.domain.schedule_snapshot import (
    ScheduleObservationAsOfSnapshot,
)


def project_schedule_identity_candidates(
    snapshot: ScheduleObservationAsOfSnapshot,
) -> ScheduleIdentityResolutionCandidateSet:
    """Copy exact selected evidence without resolving a canonical identity."""

    if not isinstance(snapshot, ScheduleObservationAsOfSnapshot):
        raise TypeError(
            "snapshot must be a ScheduleObservationAsOfSnapshot"
        )

    candidates = tuple(
        ScheduleIdentityResolutionCandidate(
            provider_namespace=selection.provider_namespace,
            provider_game_id=selection.provider_game_id,
            game_number=selection.game_number,
            scheduled_start_utc=(
                selection.selected_observation.scheduled_start_utc
            ),
            official_local_date=(
                selection.selected_observation.official_local_date
            ),
            home_provider_participant_id=(
                selection.selected_observation.home_provider_participant_id
            ),
            away_provider_participant_id=(
                selection.selected_observation.away_provider_participant_id
            ),
            source_observation_id=selection.selected_observation_id,
            source_raw_payload_sha256=(
                selection.selected_observation.raw_payload_sha256
            ),
        )
        for selection in snapshot.selections
    )
    unavailable_chain_keys = snapshot.unavailable_chain_keys
    candidate_count = len(candidates)
    unavailable_count = len(unavailable_chain_keys)
    fingerprint = (
        compute_schedule_identity_resolution_candidate_set_fingerprint(
            as_of_utc=snapshot.as_of_utc,
            source_snapshot_fingerprint=snapshot.snapshot_fingerprint,
            candidate_count=candidate_count,
            unavailable_count=unavailable_count,
            candidates=candidates,
            unavailable_chain_keys=unavailable_chain_keys,
        )
    )

    return ScheduleIdentityResolutionCandidateSet(
        as_of_utc=snapshot.as_of_utc,
        source_snapshot_fingerprint=snapshot.snapshot_fingerprint,
        candidates=candidates,
        unavailable_chain_keys=unavailable_chain_keys,
        candidate_count=candidate_count,
        unavailable_count=unavailable_count,
        candidate_set_fingerprint=fingerprint,
        schema_version=(
            SCHEDULE_IDENTITY_RESOLUTION_CANDIDATE_SET_SCHEMA_VERSION
        ),
    )
