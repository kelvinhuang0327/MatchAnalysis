"""Materialize existing baseball games from exact P10/P11B evidence."""

from datetime import timezone

from ...baseball.domain.game import BaseballGame
from ...baseball.domain.match_identity_authority import (
    ScheduleMatchIdentityConstructionSet,
)
from ...baseball.domain.participant_identity_resolution import (
    ResolvedScheduleIdentityCandidate,
    ScheduleParticipantIdentityResolutionSet,
)
from ...baseball.domain.schedule_game_materialization import (
    SCHEDULE_BASEBALL_GAME_MATERIALIZATION_SET_SCHEMA_VERSION,
    ScheduleBaseballGameMaterialization,
    ScheduleBaseballGameMaterializationSet,
    compute_schedule_baseball_game_materialization_set_fingerprint,
)
from ...core.time import UtcTimestamp


def _resolved_by_observation_id(
    candidates: tuple[ResolvedScheduleIdentityCandidate, ...],
) -> dict[str, ResolvedScheduleIdentityCandidate]:
    result: dict[str, ResolvedScheduleIdentityCandidate] = {}
    for candidate in candidates:
        if candidate.source_observation_id in result:
            raise ValueError(
                "resolved candidates must have unique"
                " source_observation_id values"
            )
        result[candidate.source_observation_id] = candidate
    return result


def _validate_source_sets(
    construction_set: ScheduleMatchIdentityConstructionSet,
    resolution_set: ScheduleParticipantIdentityResolutionSet,
) -> dict[str, ResolvedScheduleIdentityCandidate]:
    if construction_set.as_of_utc != resolution_set.as_of_utc:
        raise ValueError(
            "construction and resolution sets must have the same as_of_utc"
        )
    if (
        construction_set.source_resolution_set_fingerprint
        != resolution_set.resolution_set_fingerprint
    ):
        raise ValueError(
            "construction source resolution fingerprint must match"
            " resolution_set"
        )
    if (
        construction_set.unresolved_candidates
        != resolution_set.unresolved_candidates
    ):
        raise ValueError(
            "construction unresolved candidates must match resolution_set"
        )
    if (
        construction_set.unavailable_chain_keys
        != resolution_set.unavailable_chain_keys
    ):
        raise ValueError(
            "construction unavailable chain keys must match resolution_set"
        )

    resolved_by_id = _resolved_by_observation_id(
        resolution_set.resolved_candidates
    )
    construction_partition_ids = [
        constructed.source_observation_id
        for constructed in construction_set.constructed_identities
    ]
    construction_partition_ids.extend(
        candidate.source_observation_id
        for candidate in construction_set.authority_missing_candidates
    )
    if len(set(construction_partition_ids)) != len(
        construction_partition_ids
    ):
        raise ValueError(
            "construction result partitions must have unique"
            " source_observation_id values"
        )
    if set(construction_partition_ids) != set(resolved_by_id):
        raise ValueError(
            "construction result partitions must cover exactly the"
            " resolved candidates"
        )

    for candidate in construction_set.authority_missing_candidates:
        if resolved_by_id[candidate.source_observation_id] != candidate:
            raise ValueError(
                "authority-missing candidate must match resolution_set"
            )
    return resolved_by_id


def materialize_schedule_baseball_games(
    construction_set: ScheduleMatchIdentityConstructionSet,
    resolution_set: ScheduleParticipantIdentityResolutionSet,
) -> ScheduleBaseballGameMaterializationSet:
    """Create games only for exact P11B identities with matching P10 evidence."""

    if not isinstance(
        construction_set,
        ScheduleMatchIdentityConstructionSet,
    ):
        raise TypeError(
            "construction_set must be a"
            " ScheduleMatchIdentityConstructionSet"
        )
    if not isinstance(
        resolution_set,
        ScheduleParticipantIdentityResolutionSet,
    ):
        raise TypeError(
            "resolution_set must be a"
            " ScheduleParticipantIdentityResolutionSet"
        )

    resolved_by_id = _validate_source_sets(
        construction_set,
        resolution_set,
    )
    game_materializations = []
    for constructed in construction_set.constructed_identities:
        resolved = resolved_by_id[constructed.source_observation_id]
        expected_source_values = (
            resolved.source_raw_payload_sha256,
            resolution_set.source_candidate_set_fingerprint,
            resolution_set.resolution_set_fingerprint,
            resolution_set.mapping_set_fingerprint,
            resolved.home_canonical_participant_id,
            resolved.away_canonical_participant_id,
        )
        actual_source_values = (
            constructed.source_raw_payload_sha256,
            constructed.source_candidate_set_fingerprint,
            constructed.source_resolution_set_fingerprint,
            constructed.mapping_set_fingerprint,
            constructed.match_identity.home_participant,
            constructed.match_identity.away_participant,
        )
        if actual_source_values != expected_source_values:
            raise ValueError(
                "constructed identity must match exact resolution evidence"
            )

        baseball_game = BaseballGame(
            identity=constructed.match_identity,
            scheduled_start=UtcTimestamp(resolved.scheduled_start_utc),
        )
        game_materializations.append(
            ScheduleBaseballGameMaterialization(
                baseball_game=baseball_game,
                match_identity=constructed.match_identity,
                source_observation_id=constructed.source_observation_id,
                source_raw_payload_sha256=(
                    constructed.source_raw_payload_sha256
                ),
                source_resolution_set_fingerprint=(
                    construction_set.source_resolution_set_fingerprint
                ),
                authority_catalog_fingerprint=(
                    construction_set.authority_catalog_fingerprint
                ),
                source_construction_set_fingerprint=(
                    construction_set.construction_set_fingerprint
                ),
            )
        )

    materializations_tuple = tuple(
        sorted(
            game_materializations,
            key=lambda value: value.source_observation_id,
        )
    )
    unresolved_candidates = construction_set.unresolved_candidates
    unavailable_chain_keys = construction_set.unavailable_chain_keys
    authority_missing_candidates = (
        construction_set.authority_missing_candidates
    )
    materialized_count = len(materializations_tuple)
    unresolved_count = len(unresolved_candidates)
    unavailable_count = len(unavailable_chain_keys)
    authority_missing_count = len(authority_missing_candidates)
    as_of_utc = construction_set.as_of_utc.astimezone(timezone.utc)
    fingerprint = (
        compute_schedule_baseball_game_materialization_set_fingerprint(
            as_of_utc=as_of_utc,
            source_resolution_set_fingerprint=(
                construction_set.source_resolution_set_fingerprint
            ),
            authority_catalog_fingerprint=(
                construction_set.authority_catalog_fingerprint
            ),
            source_construction_set_fingerprint=(
                construction_set.construction_set_fingerprint
            ),
            materialized_count=materialized_count,
            unresolved_count=unresolved_count,
            unavailable_count=unavailable_count,
            authority_missing_count=authority_missing_count,
            game_materializations=materializations_tuple,
            unresolved_candidates=unresolved_candidates,
            unavailable_chain_keys=unavailable_chain_keys,
            authority_missing_candidates=authority_missing_candidates,
        )
    )

    return ScheduleBaseballGameMaterializationSet(
        as_of_utc=as_of_utc,
        source_resolution_set_fingerprint=(
            construction_set.source_resolution_set_fingerprint
        ),
        authority_catalog_fingerprint=(
            construction_set.authority_catalog_fingerprint
        ),
        source_construction_set_fingerprint=(
            construction_set.construction_set_fingerprint
        ),
        game_materializations=materializations_tuple,
        unresolved_candidates=unresolved_candidates,
        unavailable_chain_keys=unavailable_chain_keys,
        authority_missing_candidates=authority_missing_candidates,
        materialized_count=materialized_count,
        unresolved_count=unresolved_count,
        unavailable_count=unavailable_count,
        authority_missing_count=authority_missing_count,
        materialization_set_fingerprint=fingerprint,
        schema_version=(
            SCHEDULE_BASEBALL_GAME_MATERIALIZATION_SET_SCHEMA_VERSION
        ),
    )
