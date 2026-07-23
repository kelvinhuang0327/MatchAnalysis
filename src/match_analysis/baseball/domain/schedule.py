"""Immutable date-only schedule candidates held in diagnostic quarantine."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import re


PROVIDER_NAMESPACE = "MLB_STATS_API_PUBLIC_SCHEDULE"
SPORT = "baseball"
LEAGUE = "MLB"
DIAGNOSTIC_DATE_ONLY = "DIAGNOSTIC_DATE_ONLY"
PRE_REQUEST_COLLECTION_MARKER_NOT_PROVIDER_OBSERVED_AT = (
    "PRE_REQUEST_COLLECTION_MARKER_NOT_PROVIDER_OBSERVED_AT"
)

_PROVIDER_GAME_ID_PATTERN = re.compile(r"^[1-9][0-9]*$")
_SOURCE_GAME_ID_PATTERN = re.compile(r"^mlb_2026_([1-9][0-9]*)$")
_UTC_MARKER_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z$"
)


class ScheduleQuarantineReason(str, Enum):
    """Reasons a legacy date-only row cannot become canonical schedule state."""

    MISSING_SCHEDULED_START_UTC = "MISSING_SCHEDULED_START_UTC"
    MISSING_PROVIDER_OBSERVED_AT_UTC = "MISSING_PROVIDER_OBSERVED_AT_UTC"
    MISSING_SCHEDULE_STATUS = "MISSING_SCHEDULE_STATUS"
    MISSING_GAME_DISCRIMINATOR = "MISSING_GAME_DISCRIMINATOR"
    UNRESOLVED_PARTICIPANT_IDENTITY = "UNRESOLVED_PARTICIPANT_IDENTITY"
    CANONICAL_MATCH_IDENTITY_UNRESOLVED = (
        "CANONICAL_MATCH_IDENTITY_UNRESOLVED"
    )
    DATE_TEAM_COLLISION = "DATE_TEAM_COLLISION"


UNIVERSAL_QUARANTINE_REASONS = (
    ScheduleQuarantineReason.MISSING_SCHEDULED_START_UTC,
    ScheduleQuarantineReason.MISSING_PROVIDER_OBSERVED_AT_UTC,
    ScheduleQuarantineReason.MISSING_SCHEDULE_STATUS,
    ScheduleQuarantineReason.MISSING_GAME_DISCRIMINATOR,
    ScheduleQuarantineReason.UNRESOLVED_PARTICIPANT_IDENTITY,
    ScheduleQuarantineReason.CANONICAL_MATCH_IDENTITY_UNRESOLVED,
)


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _require_iso_date(value: str) -> None:
    _require_explicit(value, "game_date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("game_date must be an ISO calendar date") from error
    if parsed.isoformat() != value:
        raise ValueError("game_date must use canonical ISO encoding")


def _require_utc_marker(value: str) -> None:
    _require_explicit(value, "legacy_collection_marker_utc")
    if _UTC_MARKER_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "legacy_collection_marker_utc must be canonical UTC with Z"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "legacy_collection_marker_utc must be a valid datetime"
        ) from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("legacy_collection_marker_utc must be UTC")


@dataclass(frozen=True, slots=True)
class ProviderGameReference:
    """A provider identifier that is not a canonical match identity."""

    provider_namespace: str
    provider_game_id: str
    source_game_id: str

    def __post_init__(self) -> None:
        if self.provider_namespace != PROVIDER_NAMESPACE:
            raise ValueError(
                f"provider_namespace must be {PROVIDER_NAMESPACE}"
            )
        _require_explicit(self.provider_game_id, "provider_game_id")
        if _PROVIDER_GAME_ID_PATTERN.fullmatch(self.provider_game_id) is None:
            raise ValueError("provider_game_id must be a positive digit string")
        _require_explicit(self.source_game_id, "source_game_id")
        match = _SOURCE_GAME_ID_PATTERN.fullmatch(self.source_game_id)
        if match is None or match.group(1) != self.provider_game_id:
            raise ValueError(
                "source_game_id must losslessly wrap provider_game_id"
            )


@dataclass(frozen=True, slots=True)
class LegacyDiagnosticScheduleCandidate:
    """A date-only source row that is prohibited from schedule promotion."""

    provider_reference: ProviderGameReference
    season: int
    game_date: str
    source_home_team: str
    source_away_team: str
    legacy_collection_marker_utc: str
    quarantine_reasons: tuple[ScheduleQuarantineReason, ...]
    sport: str = SPORT
    league: str = LEAGUE
    diagnostic_status: str = DIAGNOSTIC_DATE_ONLY

    def __post_init__(self) -> None:
        if not isinstance(self.provider_reference, ProviderGameReference):
            raise TypeError(
                "provider_reference must be a ProviderGameReference"
            )
        if (
            isinstance(self.season, bool)
            or not isinstance(self.season, int)
            or self.season <= 0
        ):
            raise ValueError("season must be a positive JSON integer")
        _require_iso_date(self.game_date)
        _require_explicit(self.source_home_team, "source_home_team")
        _require_explicit(self.source_away_team, "source_away_team")
        if self.source_home_team == self.source_away_team:
            raise ValueError("source teams must differ")
        _require_utc_marker(self.legacy_collection_marker_utc)
        if self.sport != SPORT:
            raise ValueError(f"sport must be {SPORT}")
        if self.league != LEAGUE:
            raise ValueError(f"league must be {LEAGUE}")
        if self.diagnostic_status != DIAGNOSTIC_DATE_ONLY:
            raise ValueError(
                f"diagnostic_status must be {DIAGNOSTIC_DATE_ONLY}"
            )
        allowed = (
            UNIVERSAL_QUARANTINE_REASONS,
            (
                *UNIVERSAL_QUARANTINE_REASONS,
                ScheduleQuarantineReason.DATE_TEAM_COLLISION,
            ),
        )
        if self.quarantine_reasons not in allowed:
            raise ValueError(
                "quarantine_reasons must preserve the authorized order"
            )


@dataclass(frozen=True, slots=True)
class DateTeamCollisionGroup:
    """Rows sharing the same date and exact source home/away team strings."""

    game_date: str
    source_home_team: str
    source_away_team: str
    source_game_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_iso_date(self.game_date)
        _require_explicit(self.source_home_team, "source_home_team")
        _require_explicit(self.source_away_team, "source_away_team")
        if self.source_home_team == self.source_away_team:
            raise ValueError("source teams must differ")
        if len(self.source_game_ids) < 2:
            raise ValueError("a collision group requires at least two rows")
        if tuple(sorted(set(self.source_game_ids))) != self.source_game_ids:
            raise ValueError(
                "collision source_game_ids must be unique and sorted"
            )
        for source_game_id in self.source_game_ids:
            _require_explicit(source_game_id, "source_game_id")
            if _SOURCE_GAME_ID_PATTERN.fullmatch(source_game_id) is None:
                raise ValueError("invalid collision source_game_id")
