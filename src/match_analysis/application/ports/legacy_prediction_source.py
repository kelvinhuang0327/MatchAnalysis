"""Port for reading a validated, outcome-free legacy prediction snapshot."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


NULL_OUTCOME_PLACEHOLDER_FIELDS = (
    "result_home_score",
    "result_away_score",
    "actual_winner",
    "is_correct",
)
PINNED_SOURCE_PREDICTION_VERSION = "p84b_diagnostic_baseline_v1"


@dataclass(frozen=True, slots=True)
class LegacyPredictionRow:
    """Source semantics allowed to cross the legacy adapter boundary."""

    source_game_id: str
    source_prediction_version: str
    predicted_side: str
    sp_fip_delta: Decimal


@dataclass(frozen=True, slots=True)
class LegacyPredictionSnapshot:
    """Validated transport data without outcome or timestamp fields."""

    artifact_sha256: str
    rows: tuple[LegacyPredictionRow, ...]
    validated_null_outcome_placeholder_fields: tuple[str, ...]
    rows_with_observed_outcomes: int

    def __post_init__(self) -> None:
        if (
            self.validated_null_outcome_placeholder_fields
            != NULL_OUTCOME_PLACEHOLDER_FIELDS
        ):
            raise ValueError("the exact four null placeholders must be validated")
        if self.rows_with_observed_outcomes != 0:
            raise ValueError("legacy prediction snapshots must be outcome-free")


class LegacyPredictionSource(Protocol):
    """Loads one explicitly selected and hash-pinned snapshot."""

    def load(self) -> LegacyPredictionSnapshot:
        """Return validated source rows without performing writes."""
