"""Immutable records for prospective starting-pitcher information edges."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping


SIGNAL_BASELINE = "SIGNAL_BASELINE"
EDGE_OBSERVATION = "EDGE_OBSERVATION"
STARTING_PITCHER_CHANGE = "STARTING_PITCHER_CHANGE"
INFORMATION_EDGE_SCHEMA_VERSION = "information_edge_capture_v1"


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _identity_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    record_type = payload.get("record_type")
    if record_type == SIGNAL_BASELINE:
        keys = (
            "schema_version",
            "record_type",
            "schedule_identity",
            "signal_observed_at_utc",
            "probable_pitcher_state",
        )
    elif record_type == EDGE_OBSERVATION:
        keys = (
            "schema_version",
            "record_type",
            "game_identity",
            "signal_type",
            "signal_observed_at_utc",
            "pitcher_side",
            "previous_pitcher_player_id",
            "new_pitcher_player_id",
            "before_prediction_identity",
            "after_prediction_identity",
            "tsl_observation",
            "reference_market_observation",
        )
    else:
        raise ValueError("record_type must be SIGNAL_BASELINE or EDGE_OBSERVATION")
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"observation identity is incomplete: {', '.join(missing)}")
    return {key: payload[key] for key in keys}


@dataclass(frozen=True, slots=True)
class InformationEdgeObservation:
    """An immutable, canonical JSON observation suitable for one JSONL append."""

    record_type: Literal["SIGNAL_BASELINE", "EDGE_OBSERVATION"]
    observation_id: str
    _canonical_payload: str

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "InformationEdgeObservation":
        data = dict(payload)
        record_type = data.get("record_type")
        if record_type not in (SIGNAL_BASELINE, EDGE_OBSERVATION):
            raise ValueError("record_type must be SIGNAL_BASELINE or EDGE_OBSERVATION")
        data.setdefault("schema_version", INFORMATION_EDGE_SCHEMA_VERSION)
        identity = _identity_projection(data)
        observation_id = sha256(canonical_json_bytes(identity)).hexdigest()
        supplied_id = data.get("observation_id")
        if supplied_id is not None and supplied_id != observation_id:
            raise ValueError("observation_id does not match canonical observation identity")
        data["observation_id"] = observation_id
        encoded = canonical_json_bytes(data).decode("utf-8").rstrip("\n")
        return cls(
            record_type=record_type,
            observation_id=observation_id,
            _canonical_payload=encoded,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mutable projection; the stored observation stays immutable."""

        return json.loads(self._canonical_payload)

    def to_jsonl_bytes(self) -> bytes:
        return (self._canonical_payload + "\n").encode("utf-8")


__all__ = (
    "EDGE_OBSERVATION",
    "INFORMATION_EDGE_SCHEMA_VERSION",
    "InformationEdgeObservation",
    "SIGNAL_BASELINE",
    "STARTING_PITCHER_CHANGE",
    "canonical_json_bytes",
)
