"""Immutable as-of availability snapshots over explicit revision chains."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json

from .schedule_observation import ScheduleSourceObservation, canonical_utc_timestamp


SCHEDULE_OBSERVATION_AS_OF_SNAPSHOT_SCHEMA_VERSION = (
    "schedule_observation_as_of_snapshot_v1"
)

ChainKey = tuple[str, str, int]


def _require_utc(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be normalized to UTC")


def compute_schedule_observation_as_of_snapshot_fingerprint(
    *,
    as_of_utc: datetime,
    selected_count: int,
    unavailable_count: int,
    selections: "tuple[ScheduleObservationAsOfSelection, ...]",
    unavailable_chain_keys: "tuple[ChainKey, ...]",
) -> str:
    """Hash the exact canonical as-of snapshot projection plus one final LF."""

    projection = {
        "schema_version": SCHEDULE_OBSERVATION_AS_OF_SNAPSHOT_SCHEMA_VERSION,
        "as_of_utc": canonical_utc_timestamp(as_of_utc),
        "selected_count": selected_count,
        "unavailable_count": unavailable_count,
        "selections": [
            {
                "provider_namespace": selection.provider_namespace,
                "provider_game_id": selection.provider_game_id,
                "game_number": selection.game_number,
                "selected_observation_id": selection.selected_observation_id,
            }
            for selection in selections
        ],
        "unavailable_chain_keys": [
            {
                "provider_namespace": key[0],
                "provider_game_id": key[1],
                "game_number": key[2],
            }
            for key in unavailable_chain_keys
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
class ScheduleObservationAsOfSelection:
    """The one observation actually available for a chain at an as-of cutoff."""

    provider_namespace: str
    provider_game_id: str
    game_number: int
    selected_observation: ScheduleSourceObservation

    def __post_init__(self) -> None:
        if not isinstance(self.selected_observation, ScheduleSourceObservation):
            raise TypeError(
                "selected_observation must be a ScheduleSourceObservation"
            )
        if self.selected_observation.provider_namespace != self.provider_namespace:
            raise ValueError(
                "provider_namespace must match the selected observation"
            )
        if self.selected_observation.provider_game_id != self.provider_game_id:
            raise ValueError(
                "provider_game_id must match the selected observation"
            )
        if self.selected_observation.game_number != self.game_number:
            raise ValueError("game_number must match the selected observation")

    @property
    def selected_observation_id(self) -> str:
        return self.selected_observation.observation_id


@dataclass(frozen=True, slots=True)
class ScheduleObservationAsOfSnapshot:
    """Deterministic, order-independent per-chain availability at as_of_utc."""

    as_of_utc: datetime
    selections: tuple[ScheduleObservationAsOfSelection, ...]
    unavailable_chain_keys: tuple[ChainKey, ...]
    selected_count: int
    unavailable_count: int
    snapshot_fingerprint: str
    schema_version: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEDULE_OBSERVATION_AS_OF_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be exactly"
                " schedule_observation_as_of_snapshot_v1"
            )
        _require_utc(self.as_of_utc, "as_of_utc")

        if not isinstance(self.selections, tuple):
            raise TypeError("selections must be a tuple")
        if not isinstance(self.unavailable_chain_keys, tuple):
            raise TypeError("unavailable_chain_keys must be a tuple")

        selection_keys = [
            (
                selection.provider_namespace,
                selection.provider_game_id,
                selection.game_number,
            )
            for selection in self.selections
        ]
        if selection_keys != sorted(selection_keys):
            raise ValueError(
                "selections must be sorted by provider_namespace,"
                " provider_game_id, game_number"
            )
        if tuple(self.unavailable_chain_keys) != tuple(
            sorted(self.unavailable_chain_keys)
        ):
            raise ValueError(
                "unavailable_chain_keys must be sorted by provider_namespace,"
                " provider_game_id, game_number"
            )

        all_keys = selection_keys + list(self.unavailable_chain_keys)
        if len(set(all_keys)) != len(all_keys):
            raise ValueError(
                "a chain key must not appear more than once across selections"
                " and unavailable_chain_keys"
            )

        if len(self.selections) != self.selected_count:
            raise ValueError("selected_count must match the number of selections")
        if len(self.unavailable_chain_keys) != self.unavailable_count:
            raise ValueError(
                "unavailable_count must match the number of unavailable"
                " chain keys"
            )

        expected_fingerprint = compute_schedule_observation_as_of_snapshot_fingerprint(
            as_of_utc=self.as_of_utc,
            selected_count=self.selected_count,
            unavailable_count=self.unavailable_count,
            selections=self.selections,
            unavailable_chain_keys=self.unavailable_chain_keys,
        )
        if self.snapshot_fingerprint != expected_fingerprint:
            raise ValueError(
                "snapshot_fingerprint must match the canonical as-of"
                " snapshot projection"
            )
