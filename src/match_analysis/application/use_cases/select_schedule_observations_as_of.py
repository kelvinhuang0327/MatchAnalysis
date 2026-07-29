"""Select the schedule observation available at an explicit as-of cutoff, in memory only."""

from datetime import datetime, timezone

from ...baseball.domain.schedule_revision import (
    ScheduleObservationRevisionChain,
    ScheduleObservationRevisionSet,
)
from ...baseball.domain.schedule_snapshot import (
    SCHEDULE_OBSERVATION_AS_OF_SNAPSHOT_SCHEMA_VERSION,
    ChainKey,
    ScheduleObservationAsOfSelection,
    ScheduleObservationAsOfSnapshot,
    compute_schedule_observation_as_of_snapshot_fingerprint,
)


def _normalize_as_of(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("as_of_utc must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("as_of_utc must be timezone-aware")
    return value.astimezone(timezone.utc)


def _require_strictly_increasing_ingestion(
    chain: ScheduleObservationRevisionChain,
) -> None:
    previous_ingested_at_utc = None
    for observation in chain.observations:
        if (
            previous_ingested_at_utc is not None
            and observation.ingested_at_utc <= previous_ingested_at_utc
        ):
            raise ValueError(
                "ingested_at_utc must strictly increase along each revision"
                " chain"
            )
        previous_ingested_at_utc = observation.ingested_at_utc


def select_schedule_observations_as_of(
    revision_set: ScheduleObservationRevisionSet,
    as_of_utc: datetime,
) -> ScheduleObservationAsOfSnapshot:
    """Select, per chain, the last observation ingested at or before as_of_utc."""

    if not isinstance(revision_set, ScheduleObservationRevisionSet):
        raise TypeError("revision_set must be a ScheduleObservationRevisionSet")
    normalized_as_of = _normalize_as_of(as_of_utc)

    selections: list[ScheduleObservationAsOfSelection] = []
    unavailable_chain_keys: list[ChainKey] = []

    for chain in revision_set.chains:
        _require_strictly_increasing_ingestion(chain)

        selected_observation = None
        for observation in chain.observations:
            if observation.ingested_at_utc <= normalized_as_of:
                selected_observation = observation

        if selected_observation is None:
            unavailable_chain_keys.append(
                (
                    chain.provider_namespace,
                    chain.provider_game_id,
                    chain.game_number,
                )
            )
        else:
            selections.append(
                ScheduleObservationAsOfSelection(
                    provider_namespace=chain.provider_namespace,
                    provider_game_id=chain.provider_game_id,
                    game_number=chain.game_number,
                    selected_observation=selected_observation,
                )
            )

    selections_tuple = tuple(selections)
    unavailable_tuple = tuple(unavailable_chain_keys)

    fingerprint = compute_schedule_observation_as_of_snapshot_fingerprint(
        as_of_utc=normalized_as_of,
        selected_count=len(selections_tuple),
        unavailable_count=len(unavailable_tuple),
        selections=selections_tuple,
        unavailable_chain_keys=unavailable_tuple,
    )

    return ScheduleObservationAsOfSnapshot(
        as_of_utc=normalized_as_of,
        selections=selections_tuple,
        unavailable_chain_keys=unavailable_tuple,
        selected_count=len(selections_tuple),
        unavailable_count=len(unavailable_tuple),
        snapshot_fingerprint=fingerprint,
        schema_version=SCHEDULE_OBSERVATION_AS_OF_SNAPSHOT_SCHEMA_VERSION,
    )
