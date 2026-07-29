"""Resolve P9 provider participants through one explicit mapping catalog."""

from collections import defaultdict

from ...baseball.domain.participant_identity_resolution import (
    CONFLICTING_AWAY_PARTICIPANT_MAPPING,
    CONFLICTING_HOME_PARTICIPANT_MAPPING,
    MAPPING_VERSION_MISMATCH,
    MISSING_AWAY_PARTICIPANT_MAPPING,
    MISSING_HOME_PARTICIPANT_MAPPING,
    PROVIDER_PARTICIPANT_IDENTITY_MAPPING_SET_SCHEMA_VERSION,
    RESOLVED_PARTICIPANTS_NOT_DISTINCT,
    SCHEDULE_PARTICIPANT_IDENTITY_RESOLUTION_SET_SCHEMA_VERSION,
    PARTICIPANT_IDENTITY_RESOLUTION_REASON_ORDER,
    ProviderParticipantIdentityMapping,
    ResolvedScheduleIdentityCandidate,
    ScheduleParticipantIdentityResolutionSet,
    UnresolvedScheduleIdentityCandidate,
    compute_provider_participant_identity_mapping_set_fingerprint,
    compute_schedule_participant_identity_resolution_set_fingerprint,
    normalize_provider_participant_identity_mappings,
)
from ...baseball.domain.schedule_identity_candidate import (
    ScheduleIdentityResolutionCandidateSet,
)


def resolve_schedule_participant_identities(
    candidate_set: ScheduleIdentityResolutionCandidateSet,
    mappings: tuple[ProviderParticipantIdentityMapping, ...],
) -> ScheduleParticipantIdentityResolutionSet:
    """Resolve participants by exact provider key and fail closed otherwise."""

    if not isinstance(candidate_set, ScheduleIdentityResolutionCandidateSet):
        raise TypeError(
            "candidate_set must be a"
            " ScheduleIdentityResolutionCandidateSet"
        )
    normalized_mappings = (
        normalize_provider_participant_identity_mappings(mappings)
    )
    mappings_by_key: defaultdict[
        tuple[str, str],
        list[ProviderParticipantIdentityMapping],
    ] = defaultdict(list)
    for mapping in normalized_mappings:
        mappings_by_key[
            (mapping.provider_namespace, mapping.provider_participant_id)
        ].append(mapping)

    resolved_candidates = []
    unresolved_candidates = []
    for candidate in candidate_set.candidates:
        home_mappings = tuple(
            mappings_by_key[
                (
                    candidate.provider_namespace,
                    candidate.home_provider_participant_id,
                )
            ]
        )
        away_mappings = tuple(
            mappings_by_key[
                (
                    candidate.provider_namespace,
                    candidate.away_provider_participant_id,
                )
            ]
        )
        unique_home = (
            home_mappings[0] if len(home_mappings) == 1 else None
        )
        unique_away = (
            away_mappings[0] if len(away_mappings) == 1 else None
        )

        reason_conditions = {
            MISSING_HOME_PARTICIPANT_MAPPING: not home_mappings,
            MISSING_AWAY_PARTICIPANT_MAPPING: not away_mappings,
            CONFLICTING_HOME_PARTICIPANT_MAPPING: (
                len(home_mappings) > 1
            ),
            CONFLICTING_AWAY_PARTICIPANT_MAPPING: (
                len(away_mappings) > 1
            ),
            RESOLVED_PARTICIPANTS_NOT_DISTINCT: (
                unique_home is not None
                and unique_away is not None
                and unique_home.canonical_participant_id
                == unique_away.canonical_participant_id
            ),
            MAPPING_VERSION_MISMATCH: (
                unique_home is not None
                and unique_away is not None
                and unique_home.mapping_version
                != unique_away.mapping_version
            ),
        }
        reasons = tuple(
            reason
            for reason in PARTICIPANT_IDENTITY_RESOLUTION_REASON_ORDER
            if reason_conditions[reason]
        )
        if reasons:
            unresolved_candidates.append(
                UnresolvedScheduleIdentityCandidate(
                    source_observation_id=(
                        candidate.source_observation_id
                    ),
                    provider_namespace=candidate.provider_namespace,
                    provider_game_id=candidate.provider_game_id,
                    game_number=candidate.game_number,
                    reasons=reasons,
                )
            )
            continue

        if unique_home is None or unique_away is None:
            raise AssertionError("resolution reasons must cover ambiguity")
        resolved_candidates.append(
            ResolvedScheduleIdentityCandidate(
                provider_namespace=candidate.provider_namespace,
                provider_game_id=candidate.provider_game_id,
                game_number=candidate.game_number,
                scheduled_start_utc=candidate.scheduled_start_utc,
                official_local_date=candidate.official_local_date,
                home_provider_participant_id=(
                    candidate.home_provider_participant_id
                ),
                away_provider_participant_id=(
                    candidate.away_provider_participant_id
                ),
                source_observation_id=candidate.source_observation_id,
                source_raw_payload_sha256=(
                    candidate.source_raw_payload_sha256
                ),
                home_canonical_participant_id=(
                    unique_home.canonical_participant_id
                ),
                away_canonical_participant_id=(
                    unique_away.canonical_participant_id
                ),
                mapping_version=unique_home.mapping_version,
            )
        )

    resolved_tuple = tuple(resolved_candidates)
    unresolved_tuple = tuple(unresolved_candidates)
    mapping_set_fingerprint = (
        compute_provider_participant_identity_mapping_set_fingerprint(
            normalized_mappings
        )
    )
    resolved_count = len(resolved_tuple)
    unresolved_count = len(unresolved_tuple)
    unavailable_chain_keys = candidate_set.unavailable_chain_keys
    unavailable_count = len(unavailable_chain_keys)
    resolution_set_fingerprint = (
        compute_schedule_participant_identity_resolution_set_fingerprint(
            as_of_utc=candidate_set.as_of_utc,
            source_candidate_set_fingerprint=(
                candidate_set.candidate_set_fingerprint
            ),
            mapping_set_fingerprint=mapping_set_fingerprint,
            resolved_count=resolved_count,
            unresolved_count=unresolved_count,
            unavailable_count=unavailable_count,
            resolved_candidates=resolved_tuple,
            unresolved_candidates=unresolved_tuple,
            unavailable_chain_keys=unavailable_chain_keys,
        )
    )

    return ScheduleParticipantIdentityResolutionSet(
        as_of_utc=candidate_set.as_of_utc,
        source_candidate_set_fingerprint=(
            candidate_set.candidate_set_fingerprint
        ),
        mapping_set_fingerprint=mapping_set_fingerprint,
        resolved_candidates=resolved_tuple,
        unresolved_candidates=unresolved_tuple,
        unavailable_chain_keys=unavailable_chain_keys,
        resolved_count=resolved_count,
        unresolved_count=unresolved_count,
        unavailable_count=unavailable_count,
        resolution_set_fingerprint=resolution_set_fingerprint,
        schema_version=(
            SCHEDULE_PARTICIPANT_IDENTITY_RESOLUTION_SET_SCHEMA_VERSION
        ),
    )
