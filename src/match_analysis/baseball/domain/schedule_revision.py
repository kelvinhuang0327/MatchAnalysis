"""Immutable, explicit-supersession-only schedule observation revision chains."""

from dataclasses import dataclass
from hashlib import sha256
import json

from .schedule_observation import ScheduleSourceObservation


SCHEDULE_OBSERVATION_REVISION_SET_SCHEMA_VERSION = (
    "schedule_observation_revision_set_v1"
)


def compute_revision_set_fingerprint(
    *,
    unique_observation_count: int,
    idempotent_duplicate_count: int,
    chains: "tuple[ScheduleObservationRevisionChain, ...]",
) -> str:
    """Hash the exact canonical revision-set projection plus one final LF."""

    projection = {
        "schema_version": SCHEDULE_OBSERVATION_REVISION_SET_SCHEMA_VERSION,
        "unique_observation_count": unique_observation_count,
        "idempotent_duplicate_count": idempotent_duplicate_count,
        "chains": [
            {
                "provider_namespace": chain.provider_namespace,
                "provider_game_id": chain.provider_game_id,
                "game_number": chain.game_number,
                "root_observation_id": chain.root_observation_id,
                "head_observation_id": chain.head_observation_id,
                "ordered_observation_ids": [
                    observation.observation_id
                    for observation in chain.observations
                ],
                "observation_count": chain.observation_count,
            }
            for chain in chains
        ],
    }
    canonical_line = (
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return sha256(canonical_line).hexdigest()


@dataclass(frozen=True, slots=True)
class ScheduleObservationRevisionChain:
    """One explicit-supersession chain for a single provider game/number."""

    provider_namespace: str
    provider_game_id: str
    game_number: int
    observations: tuple[ScheduleSourceObservation, ...]
    root_observation_id: str
    head_observation_id: str
    observation_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.observations, tuple) or not self.observations:
            raise ValueError("observations must be a nonempty tuple")
        for observation in self.observations:
            if observation.provider_namespace != self.provider_namespace:
                raise ValueError(
                    "every observation must share the chain provider_namespace"
                )
            if observation.provider_game_id != self.provider_game_id:
                raise ValueError(
                    "every observation must share the chain provider_game_id"
                )
            if observation.game_number != self.game_number:
                raise ValueError(
                    "every observation must share the chain game_number"
                )

        observation_ids = [obs.observation_id for obs in self.observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("a chain must not contain duplicate observation IDs")

        root, *revisions = self.observations
        if root.supersedes_observation_id is not None:
            raise ValueError("the first observation in a chain must be a root")

        previous = root
        for revision in revisions:
            if revision.supersedes_observation_id != previous.observation_id:
                raise ValueError(
                    "each later observation must supersede the immediately"
                    " previous observation"
                )
            if (
                revision.response_received_at_utc
                <= previous.response_received_at_utc
            ):
                raise ValueError(
                    "response_received_at_utc must strictly increase along"
                    " each revision edge"
                )
            previous = revision

        if self.root_observation_id != root.observation_id:
            raise ValueError("root_observation_id must match the first observation")
        if self.head_observation_id != previous.observation_id:
            raise ValueError("head_observation_id must match the last observation")
        if self.observation_count != len(self.observations):
            raise ValueError("observation_count must match the observations tuple")


@dataclass(frozen=True, slots=True)
class ScheduleObservationRevisionSet:
    """Deterministic, order-independent set of revision chains."""

    chains: tuple[ScheduleObservationRevisionChain, ...]
    unique_observation_count: int
    idempotent_duplicate_count: int
    revision_set_fingerprint: str
    schema_version: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEDULE_OBSERVATION_REVISION_SET_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be exactly"
                " schedule_observation_revision_set_v1"
            )
        if not isinstance(self.chains, tuple) or not self.chains:
            raise ValueError("chains must be a nonempty tuple")

        sort_keys = [
            (chain.provider_namespace, chain.provider_game_id, chain.game_number)
            for chain in self.chains
        ]
        if sort_keys != sorted(sort_keys):
            raise ValueError(
                "chains must be sorted by provider_namespace, provider_game_id,"
                " game_number"
            )

        all_observation_ids = [
            observation.observation_id
            for chain in self.chains
            for observation in chain.observations
        ]
        if len(set(all_observation_ids)) != len(all_observation_ids):
            raise ValueError(
                "an observation_id must not appear in more than one chain"
            )
        if len(all_observation_ids) != self.unique_observation_count:
            raise ValueError(
                "unique_observation_count must match the total distinct"
                " observations across all chains"
            )
        if (
            isinstance(self.idempotent_duplicate_count, bool)
            or not isinstance(self.idempotent_duplicate_count, int)
            or self.idempotent_duplicate_count < 0
        ):
            raise ValueError(
                "idempotent_duplicate_count must be a non-negative integer"
            )

        expected_fingerprint = compute_revision_set_fingerprint(
            unique_observation_count=self.unique_observation_count,
            idempotent_duplicate_count=self.idempotent_duplicate_count,
            chains=self.chains,
        )
        if self.revision_set_fingerprint != expected_fingerprint:
            raise ValueError(
                "revision_set_fingerprint must match the canonical"
                " revision-set projection"
            )
