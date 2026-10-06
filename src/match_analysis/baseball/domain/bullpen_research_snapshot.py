"""Immutable serialized research evidence, separate from Moneyline inference."""

from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import json
from typing import Any, Mapping

from .canonical_utc import parse_canonical_utc
from ...core.identity import MatchIdentity


BULLPEN_RESEARCH_SCHEMA_VERSION = "bullpen.research_snapshot.v1"


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class BullpenResearchSnapshot:
    """A versioned immutable payload with reproducible full-evidence hashing."""

    payload_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.payload_bytes, bytes):
            raise TypeError("snapshot payload must be immutable bytes")
        projection = json.loads(self.payload_bytes)
        if not isinstance(projection, dict) or projection.get("schema_version") != BULLPEN_RESEARCH_SCHEMA_VERSION:
            raise ValueError("unexpected bullpen research schema")
        if projection.get("research_only") is not True:
            raise ValueError("bullpen snapshot must remain research-only")
        if canonical_bytes(projection) != self.payload_bytes:
            raise ValueError("bullpen snapshot bytes must be canonical")
        target = projection["target"]
        identity = MatchIdentity(**target["identity"])
        official_date = date.fromisoformat(target["official_date"])
        if identity.sport != "baseball" or identity.league != "MLB" or identity.season != official_date.year:
            raise ValueError("snapshot MLB/season identity mismatch")
        for field in ("home_team_id", "away_team_id", "game_number"):
            if isinstance(target[field], bool) or not isinstance(target[field], int) or target[field] <= 0:
                raise ValueError("snapshot target ids/game number must be positive integers")
        if target["home_team_id"] == target["away_team_id"]:
            raise ValueError("snapshot target team ids must differ")
        cutoff = parse_canonical_utc(target["as_of_utc"])
        if cutoff >= parse_canonical_utc(target["scheduled_start_utc"]):
            raise ValueError("snapshot cutoff must be before scheduled start")
        teams = {target["home_team_id"]: target["identity"]["home_participant"],
                 target["away_team_id"]: target["identity"]["away_participant"]}
        seen = set()
        for row in projection["accepted_relief_evidence"]:
            started = parse_canonical_utc(row["started_at_utc"])
            ended = parse_canonical_utc(row["ended_at_utc"])
            available = parse_canonical_utc(row["available_at_utc"])
            usage_date = date.fromisoformat(row["usage_date"])
            if (not official_date - timedelta(days=14) <= usage_date <= official_date
                    or usage_date < date.fromisoformat(row["official_date"])
                    or row["usage_date_basis"] != "SOURCE_CONFIRMED_SINGLE_DAY"):
                raise ValueError("snapshot has invalid/future usage date evidence")
            if not started < ended < cutoff or not ended <= available <= cutoff:
                raise ValueError("snapshot includes unavailable or post-cutoff relief evidence")
            if (row["provider_namespace"] != target["provider_namespace"]
                    or row["provider_game_id"] == target["provider_game_id"]
                    or teams.get(row["team_id"]) != row["team_code"]
                    or row["role"] != "RELIEF"):
                raise ValueError("snapshot relief identity/role mismatch")
            key = (row["provider_namespace"], row["provider_game_id"], row["pitcher_id"])
            if key in seen:
                raise ValueError("snapshot has duplicate pitcher/game relief evidence")
            seen.add(key)
        for row in projection["coverage_evidence"]:
            if teams.get(row["team_id"]) != row["team_code"]:
                raise ValueError("snapshot coverage team identity mismatch")
            if not parse_canonical_utc(row["through_utc"]) <= parse_canonical_utc(row["available_at_utc"]) <= cutoff:
                raise ValueError("snapshot coverage is unavailable at cutoff")

    @classmethod
    def from_projection(cls, projection: Mapping[str, Any]) -> "BullpenResearchSnapshot":
        return cls(canonical_bytes(projection))

    def to_projection(self) -> dict[str, Any]:
        """Return a detached copy; callers cannot mutate frozen evidence."""
        return json.loads(self.payload_bytes)

    def canonical_bytes(self) -> bytes:
        return self.payload_bytes

    def fingerprint(self) -> str:
        return sha256(self.payload_bytes).hexdigest()
