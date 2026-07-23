"""Immutable diagnostic-only join between quarantined prediction and schedule rows."""

from dataclasses import dataclass, field
import re

from .prediction import LegacyPredictionCandidate
from .schedule import (
    LegacyDiagnosticScheduleCandidate,
    ProviderGameReference,
    ScheduleQuarantineReason,
)


DIAGNOSTIC_LINKED_UNTIMED_UNRESOLVED = "DIAGNOSTIC_LINKED_UNTIMED_UNRESOLVED"

_SOURCE_GAME_ID_PATTERN = re.compile(r"^mlb_2026_([1-9][0-9]*)$")


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


@dataclass(frozen=True, slots=True)
class LegacyDiagnosticPredictionScheduleLink:
    """A diagnostic-only join between one quarantined prediction and one
    quarantined schedule row, matched by exact source game ID.

    Carries no canonical identity, timestamp, or promotion state; both nested
    candidates remain exactly as their own importers quarantined them.
    """

    source_game_id: str
    provider_reference: ProviderGameReference
    prediction_candidate: LegacyPredictionCandidate
    schedule_candidate: LegacyDiagnosticScheduleCandidate
    diagnostic_status: str = DIAGNOSTIC_LINKED_UNTIMED_UNRESOLVED
    prediction_quarantine_reasons: tuple[str, ...] = field(init=False)
    schedule_quarantine_reasons: tuple[ScheduleQuarantineReason, ...] = field(
        init=False
    )
    schedule_collision_affected: bool = field(init=False)

    def __post_init__(self) -> None:
        _require_explicit(self.source_game_id, "source_game_id")
        match = _SOURCE_GAME_ID_PATTERN.fullmatch(self.source_game_id)
        if match is None:
            raise ValueError(
                "source_game_id does not match the pinned source format"
            )

        if not isinstance(self.provider_reference, ProviderGameReference):
            raise TypeError("provider_reference must be a ProviderGameReference")
        if not isinstance(self.prediction_candidate, LegacyPredictionCandidate):
            raise TypeError(
                "prediction_candidate must be a LegacyPredictionCandidate"
            )
        if not isinstance(
            self.schedule_candidate, LegacyDiagnosticScheduleCandidate
        ):
            raise TypeError(
                "schedule_candidate must be a LegacyDiagnosticScheduleCandidate"
            )

        if self.prediction_candidate.source_game_id != self.source_game_id:
            raise ValueError(
                "source_game_id does not match the prediction candidate"
            )
        if (
            self.schedule_candidate.provider_reference.source_game_id
            != self.source_game_id
        ):
            raise ValueError(
                "source_game_id does not match the schedule candidate"
            )
        if self.provider_reference != self.schedule_candidate.provider_reference:
            raise ValueError(
                "provider_reference does not agree with the schedule candidate"
            )
        if match.group(1) != self.provider_reference.provider_game_id:
            raise ValueError(
                "source_game_id does not losslessly agree with provider_game_id"
            )

        if self.diagnostic_status != DIAGNOSTIC_LINKED_UNTIMED_UNRESOLVED:
            raise ValueError(
                f"diagnostic_status must be {DIAGNOSTIC_LINKED_UNTIMED_UNRESOLVED}"
            )

        object.__setattr__(
            self,
            "prediction_quarantine_reasons",
            (self.prediction_candidate.quarantine_reason,),
        )
        object.__setattr__(
            self,
            "schedule_quarantine_reasons",
            self.schedule_candidate.quarantine_reasons,
        )
        object.__setattr__(
            self,
            "schedule_collision_affected",
            ScheduleQuarantineReason.DATE_TEAM_COLLISION
            in self.schedule_candidate.quarantine_reasons,
        )
