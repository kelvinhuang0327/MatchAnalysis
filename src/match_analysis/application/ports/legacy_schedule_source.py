"""Port for reading a validated legacy date-only schedule snapshot."""

from dataclasses import dataclass
from typing import Protocol

from ...baseball.domain.schedule import ProviderGameReference


@dataclass(frozen=True, slots=True)
class LegacyScheduleRow:
    """Validated transport evidence allowed across the legacy boundary."""

    provider_reference: ProviderGameReference
    season: int
    game_date: str
    source_home_team: str
    source_away_team: str
    source_trace: str
    legacy_collection_marker_utc: str


@dataclass(frozen=True, slots=True)
class LegacyScheduleSnapshot:
    """Hash-verified, normalized rows without trusted schedule semantics."""

    artifact_sha256: str
    rows: tuple[LegacyScheduleRow, ...]


class LegacyScheduleSource(Protocol):
    """Loads one explicitly selected and hash-pinned schedule snapshot."""

    def load(self) -> LegacyScheduleSnapshot:
        """Return validated source rows without performing writes."""
