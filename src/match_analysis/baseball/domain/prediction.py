"""Outcome-free prediction candidates held in diagnostic quarantine."""

from dataclasses import dataclass
from decimal import Decimal
import re


DIAGNOSTIC_UNTIMED = "DIAGNOSTIC_UNTIMED"
MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF = (
    "MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF"
)

_SOURCE_GAME_ID_PATTERN = re.compile(r"^mlb_2026_[0-9]+$")


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


@dataclass(frozen=True, slots=True)
class LegacyPredictionCandidate:
    """A source prediction that cannot be promoted without time evidence."""

    source_game_id: str
    source_prediction_version: str
    predicted_side: str
    sp_fip_delta: Decimal
    diagnostic_status: str = DIAGNOSTIC_UNTIMED
    quarantine_reason: str = MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF

    def __post_init__(self) -> None:
        _require_explicit(self.source_game_id, "source_game_id")
        _require_explicit(
            self.source_prediction_version,
            "source_prediction_version",
        )
        if _SOURCE_GAME_ID_PATTERN.fullmatch(self.source_game_id) is None:
            raise ValueError("source_game_id does not match the pinned source format")
        if self.predicted_side not in {"home", "away"}:
            raise ValueError("predicted_side must be 'home' or 'away'")
        if not isinstance(self.sp_fip_delta, Decimal):
            raise TypeError("sp_fip_delta must be a Decimal")
        if not self.sp_fip_delta.is_finite():
            raise ValueError("sp_fip_delta must be finite")
        if self.sp_fip_delta.is_zero():
            raise ValueError("sp_fip_delta must be non-zero")
        if self.diagnostic_status != DIAGNOSTIC_UNTIMED:
            raise ValueError(f"diagnostic_status must be {DIAGNOSTIC_UNTIMED}")
        if (
            self.quarantine_reason
            != MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF
        ):
            raise ValueError(
                "quarantine_reason must identify both missing time fields"
            )
