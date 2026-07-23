"""Port for obtaining one explicit provider schedule capture."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ScheduleObservationCapture:
    """Transport-neutral fields forwarded without reconstruction."""

    provider_namespace: str
    provider_game_id: str
    scheduled_start_utc: datetime
    official_local_date: date
    response_received_at_utc: datetime
    ingested_at_utc: datetime
    provider_status_code: str
    provider_detailed_status: str
    game_number: int
    home_provider_participant_id: str
    away_provider_participant_id: str
    endpoint_id: str
    parser_version: str
    schema_version: str
    raw_payload_bytes: bytes
    raw_payload_sha256: str
    supersedes_observation_id: str | None


class ScheduleObservationSource(Protocol):
    """Returns one bounded capture without transport or persistence semantics."""

    def capture(self) -> ScheduleObservationCapture:
        """Return exactly one explicit provider response capture."""
