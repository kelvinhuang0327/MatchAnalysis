"""Build deterministic schedule observation revision chains, in memory only."""

from collections.abc import Iterable

from ...baseball.domain.schedule_observation import ScheduleSourceObservation
from ...baseball.domain.schedule_revision import (
    SCHEDULE_OBSERVATION_REVISION_SET_SCHEMA_VERSION,
    ScheduleObservationRevisionChain,
    ScheduleObservationRevisionSet,
    compute_revision_set_fingerprint,
)


PartitionKey = tuple[str, str, int]


def build_schedule_observation_revision_chains(
    observations: Iterable[ScheduleSourceObservation],
) -> ScheduleObservationRevisionSet:
    """Partition explicit-supersession-only chains from existing observations."""

    materialized = list(observations)
    if not materialized:
        raise ValueError("at least one observation is required")

    unique_by_id: dict[str, ScheduleSourceObservation] = {}
    ordered_unique: list[ScheduleSourceObservation] = []
    idempotent_duplicate_count = 0
    for observation in materialized:
        existing = unique_by_id.get(observation.observation_id)
        if existing is None:
            unique_by_id[observation.observation_id] = observation
            ordered_unique.append(observation)
        elif existing == observation:
            idempotent_duplicate_count += 1
        else:
            raise ValueError(
                "observation_id "
                f"{observation.observation_id!r} reused for unequal observations"
            )

    partitions: dict[PartitionKey, list[ScheduleSourceObservation]] = {}
    for observation in ordered_unique:
        key = (
            observation.provider_namespace,
            observation.provider_game_id,
            observation.game_number,
        )
        partitions.setdefault(key, []).append(observation)

    chains = [
        _build_chain(key, partition_observations)
        for key, partition_observations in partitions.items()
    ]
    chains.sort(
        key=lambda chain: (
            chain.provider_namespace,
            chain.provider_game_id,
            chain.game_number,
        )
    )
    chains_tuple = tuple(chains)

    fingerprint = compute_revision_set_fingerprint(
        unique_observation_count=len(ordered_unique),
        idempotent_duplicate_count=idempotent_duplicate_count,
        chains=chains_tuple,
    )

    return ScheduleObservationRevisionSet(
        chains=chains_tuple,
        unique_observation_count=len(ordered_unique),
        idempotent_duplicate_count=idempotent_duplicate_count,
        revision_set_fingerprint=fingerprint,
        schema_version=SCHEDULE_OBSERVATION_REVISION_SET_SCHEMA_VERSION,
    )


def _build_chain(
    key: PartitionKey,
    partition_observations: list[ScheduleSourceObservation],
) -> ScheduleObservationRevisionChain:
    ordered = _resolve_chain_order(key, partition_observations)
    return ScheduleObservationRevisionChain(
        provider_namespace=key[0],
        provider_game_id=key[1],
        game_number=key[2],
        observations=ordered,
        root_observation_id=ordered[0].observation_id,
        head_observation_id=ordered[-1].observation_id,
        observation_count=len(ordered),
    )


def _resolve_chain_order(
    key: PartitionKey,
    partition_observations: list[ScheduleSourceObservation],
) -> tuple[ScheduleSourceObservation, ...]:
    by_id = {obs.observation_id: obs for obs in partition_observations}
    successor_of: dict[str, ScheduleSourceObservation] = {}
    roots: list[ScheduleSourceObservation] = []

    for observation in partition_observations:
        supersedes_id = observation.supersedes_observation_id
        if supersedes_id is None:
            roots.append(observation)
            continue
        if supersedes_id not in by_id:
            raise ValueError(
                f"orphan revision in partition {key!r}: "
                f"{observation.observation_id!r} supersedes unknown observation"
                f" {supersedes_id!r}"
            )
        if supersedes_id in successor_of:
            raise ValueError(
                f"fork detected in partition {key!r}: observation"
                f" {supersedes_id!r} has more than one successor"
            )
        successor_of[supersedes_id] = observation

    if len(roots) == 0:
        raise ValueError(f"no root observation found for partition {key!r}")
    if len(roots) > 1:
        raise ValueError(f"multiple root observations found for partition {key!r}")

    ordered = [roots[0]]
    seen_ids = {roots[0].observation_id}
    while ordered[-1].observation_id in successor_of:
        successor = successor_of[ordered[-1].observation_id]
        if successor.observation_id in seen_ids:
            raise ValueError(f"cycle detected in partition {key!r}")
        ordered.append(successor)
        seen_ids.add(successor.observation_id)

    if len(ordered) != len(partition_observations):
        raise ValueError(f"disconnected observations found in partition {key!r}")

    for earlier, later in zip(ordered, ordered[1:]):
        if later.response_received_at_utc <= earlier.response_received_at_utc:
            raise ValueError(
                "response_received_at_utc must strictly increase along each"
                " revision edge"
            )

    return tuple(ordered)
