"""Construct the existing P1 match identity from P10 and explicit authority."""

from ...baseball.domain.match_identity_authority import (
    SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION,
    ConstructedScheduleMatchIdentity,
    MatchIdentityAuthorityCatalog,
    ScheduleMatchIdentityConstructionSet,
    compute_schedule_match_identity_construction_set_fingerprint,
)
from ...baseball.domain.participant_identity_resolution import (
    ScheduleParticipantIdentityResolutionSet,
)
from ...core.identity import MatchIdentity


BASEBALL_MATCH_IDENTITY_SPORT = "baseball"


def construct_match_identities(
    resolution_set: ScheduleParticipantIdentityResolutionSet,
    authority_catalog: MatchIdentityAuthorityCatalog,
) -> ScheduleMatchIdentityConstructionSet:
    """Construct P1 identities only for exact, explicitly authorized keys."""

    if not isinstance(
        resolution_set,
        ScheduleParticipantIdentityResolutionSet,
    ):
        raise TypeError(
            "resolution_set must be a"
            " ScheduleParticipantIdentityResolutionSet"
        )
    if not isinstance(
        authority_catalog,
        MatchIdentityAuthorityCatalog,
    ):
        raise TypeError(
            "authority_catalog must be a MatchIdentityAuthorityCatalog"
        )

    authority_by_key = {
        (
            entry.provider_namespace,
            entry.provider_game_id,
            entry.game_number,
        ): entry
        for entry in authority_catalog.entries
    }

    constructed_identities = []
    authority_missing_candidates = []
    for resolved in resolution_set.resolved_candidates:
        key = (
            resolved.provider_namespace,
            resolved.provider_game_id,
            resolved.game_number,
        )
        entry = authority_by_key.get(key)
        if entry is None:
            authority_missing_candidates.append(resolved)
            continue

        constructed_identities.append(
            ConstructedScheduleMatchIdentity(
                match_identity=MatchIdentity(
                    sport=BASEBALL_MATCH_IDENTITY_SPORT,
                    league=entry.league,
                    season=entry.season,
                    canonical_game_id=entry.canonical_game_id,
                    home_participant=(
                        resolved.home_canonical_participant_id
                    ),
                    away_participant=(
                        resolved.away_canonical_participant_id
                    ),
                    game_discriminator=entry.game_discriminator,
                ),
                source_observation_id=resolved.source_observation_id,
                source_raw_payload_sha256=(
                    resolved.source_raw_payload_sha256
                ),
                source_candidate_set_fingerprint=(
                    resolution_set.source_candidate_set_fingerprint
                ),
                source_resolution_set_fingerprint=(
                    resolution_set.resolution_set_fingerprint
                ),
                mapping_set_fingerprint=(
                    resolution_set.mapping_set_fingerprint
                ),
            )
        )

    constructed_tuple = tuple(constructed_identities)
    authority_missing_tuple = tuple(authority_missing_candidates)
    unresolved_candidates = resolution_set.unresolved_candidates
    unavailable_chain_keys = resolution_set.unavailable_chain_keys
    constructed_count = len(constructed_tuple)
    unresolved_count = len(unresolved_candidates)
    unavailable_count = len(unavailable_chain_keys)
    authority_missing_count = len(authority_missing_tuple)
    construction_set_fingerprint = (
        compute_schedule_match_identity_construction_set_fingerprint(
            as_of_utc=resolution_set.as_of_utc,
            source_resolution_set_fingerprint=(
                resolution_set.resolution_set_fingerprint
            ),
            authority_catalog_fingerprint=(
                authority_catalog.catalog_fingerprint
            ),
            constructed_count=constructed_count,
            unresolved_count=unresolved_count,
            unavailable_count=unavailable_count,
            authority_missing_count=authority_missing_count,
            constructed_identities=constructed_tuple,
            unresolved_candidates=unresolved_candidates,
            unavailable_chain_keys=unavailable_chain_keys,
            authority_missing_candidates=authority_missing_tuple,
        )
    )

    return ScheduleMatchIdentityConstructionSet(
        as_of_utc=resolution_set.as_of_utc,
        source_resolution_set_fingerprint=(
            resolution_set.resolution_set_fingerprint
        ),
        authority_catalog_fingerprint=(
            authority_catalog.catalog_fingerprint
        ),
        constructed_identities=constructed_tuple,
        unresolved_candidates=unresolved_candidates,
        unavailable_chain_keys=unavailable_chain_keys,
        authority_missing_candidates=authority_missing_tuple,
        constructed_count=constructed_count,
        unresolved_count=unresolved_count,
        unavailable_count=unavailable_count,
        authority_missing_count=authority_missing_count,
        construction_set_fingerprint=construction_set_fingerprint,
        schema_version=(
            SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION
        ),
    )
