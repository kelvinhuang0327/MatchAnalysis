"""Immutable per-row quarantine assessment for legacy P83E prediction rows.

Every assessed row stays quarantined: this module never constructs a
PredictionSourceObservation and never admits or promotes a prediction. Each
assessment either restates existing P83E/P3 evidence or names the exact
controlled reason the row cannot yet be admitted.
"""

from dataclasses import dataclass
import re

from .prediction import LegacyPredictionCandidate
from .quarantine_link import LegacyDiagnosticPredictionScheduleLink


QUARANTINE_STATUS = "QUARANTINED"

MISSING_PREDICTION_OBSERVATION_ID = "MISSING_PREDICTION_OBSERVATION_ID"
MISSING_SOURCE_PREDICTION_ID = "MISSING_SOURCE_PREDICTION_ID"
MISSING_MODEL_ID = "MISSING_MODEL_ID"
MISSING_MARKET_ID = "MISSING_MARKET_ID"
MISSING_LINE_VALUE = "MISSING_LINE_VALUE"
MISSING_PUSH_POLICY = "MISSING_PUSH_POLICY"
MISSING_PREDICTION_GENERATED_AT = "MISSING_PREDICTION_GENERATED_AT"
MISSING_RESPONSE_RECEIVED_AT = "MISSING_RESPONSE_RECEIVED_AT"
MISSING_INGESTED_AT = "MISSING_INGESTED_AT"
MISSING_GAME_NUMBER = "MISSING_GAME_NUMBER"
MISSING_SOURCE_OBSERVATION_ID = "MISSING_SOURCE_OBSERVATION_ID"
MISSING_DIAGNOSTIC_SCHEDULE_LINK = "MISSING_DIAGNOSTIC_SCHEDULE_LINK"
AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH = "AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH"
ZERO_DELTA_SELECTION_POLICY_UNRESOLVED = "ZERO_DELTA_SELECTION_POLICY_UNRESOLVED"

# Every P83E row lacks all nine of these observation-level fields; no legacy
# row may ever be missing only a subset, so every assessment carries all nine.
UNIVERSAL_MISSING_OBSERVATION_REASONS = (
    MISSING_PREDICTION_OBSERVATION_ID,
    MISSING_SOURCE_PREDICTION_ID,
    MISSING_MODEL_ID,
    MISSING_MARKET_ID,
    MISSING_LINE_VALUE,
    MISSING_PUSH_POLICY,
    MISSING_PREDICTION_GENERATED_AT,
    MISSING_RESPONSE_RECEIVED_AT,
    MISSING_INGESTED_AT,
)

CONTROLLED_QUARANTINE_REASONS = frozenset(
    {
        *UNIVERSAL_MISSING_OBSERVATION_REASONS,
        MISSING_GAME_NUMBER,
        MISSING_SOURCE_OBSERVATION_ID,
        MISSING_DIAGNOSTIC_SCHEDULE_LINK,
        AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH,
        ZERO_DELTA_SELECTION_POLICY_UNRESOLVED,
    }
)

_SOURCE_OBSERVATION_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


@dataclass(frozen=True, slots=True)
class LegacyPredictionQuarantineAssessment:
    """One deterministic, always-quarantined outcome for a single P83E row."""

    source_game_id: str
    prediction_candidate: LegacyPredictionCandidate
    diagnostic_link: LegacyDiagnosticPredictionScheduleLink | None
    quarantine_reasons: tuple[str, ...]
    quarantine_status: str = QUARANTINE_STATUS
    enriched_game_number: int | None = None
    enriched_source_observation_id: str | None = None

    def __post_init__(self) -> None:
        _require_explicit(self.source_game_id, "source_game_id")
        if not isinstance(self.prediction_candidate, LegacyPredictionCandidate):
            raise TypeError(
                "prediction_candidate must be a LegacyPredictionCandidate"
            )
        if self.prediction_candidate.source_game_id != self.source_game_id:
            raise ValueError("source_game_id must match the prediction candidate")

        has_link = self.diagnostic_link is not None
        if has_link:
            if not isinstance(
                self.diagnostic_link, LegacyDiagnosticPredictionScheduleLink
            ):
                raise TypeError(
                    "diagnostic_link must be a"
                    " LegacyDiagnosticPredictionScheduleLink"
                )
            if self.diagnostic_link.source_game_id != self.source_game_id:
                raise ValueError("source_game_id must match the diagnostic link")
            if self.diagnostic_link.prediction_candidate != self.prediction_candidate:
                raise ValueError(
                    "diagnostic_link must reference the same prediction candidate"
                )

        if self.quarantine_status != QUARANTINE_STATUS:
            raise ValueError(f"quarantine_status must be {QUARANTINE_STATUS}")

        if not isinstance(self.quarantine_reasons, tuple):
            raise TypeError("quarantine_reasons must be a tuple")
        if not self.quarantine_reasons:
            raise ValueError("quarantine_reasons must be non-empty")
        if len(set(self.quarantine_reasons)) != len(self.quarantine_reasons):
            raise ValueError("quarantine_reasons must not repeat")
        if tuple(sorted(self.quarantine_reasons)) != self.quarantine_reasons:
            raise ValueError("quarantine_reasons must be sorted ascending")
        unknown = [
            reason
            for reason in self.quarantine_reasons
            if reason not in CONTROLLED_QUARANTINE_REASONS
        ]
        if unknown:
            raise ValueError(f"uncontrolled quarantine reasons: {unknown}")

        missing_universal = [
            reason
            for reason in UNIVERSAL_MISSING_OBSERVATION_REASONS
            if reason not in self.quarantine_reasons
        ]
        if missing_universal:
            raise ValueError(
                "every row must carry the universal missing-observation"
                f" reasons: {missing_universal}"
            )

        if has_link and MISSING_DIAGNOSTIC_SCHEDULE_LINK in self.quarantine_reasons:
            raise ValueError(
                "MISSING_DIAGNOSTIC_SCHEDULE_LINK must not be present when a"
                " diagnostic_link exists"
            )
        if (
            not has_link
            and MISSING_DIAGNOSTIC_SCHEDULE_LINK not in self.quarantine_reasons
        ):
            raise ValueError(
                "MISSING_DIAGNOSTIC_SCHEDULE_LINK must be present when no"
                " diagnostic_link exists"
            )
        if (
            AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH in self.quarantine_reasons
            and not has_link
        ):
            raise ValueError(
                "AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH requires an existing"
                " diagnostic_link"
            )

        is_enriched = self.enriched_game_number is not None
        if is_enriched != (self.enriched_source_observation_id is not None):
            raise ValueError("enrichment fields must be set or unset together")
        if is_enriched:
            if not has_link:
                raise ValueError("enrichment requires an existing diagnostic_link")
            if (
                isinstance(self.enriched_game_number, bool)
                or not isinstance(self.enriched_game_number, int)
                or self.enriched_game_number <= 0
            ):
                raise ValueError("enriched_game_number must be a positive integer")
            if (
                _SOURCE_OBSERVATION_ID_PATTERN.fullmatch(
                    self.enriched_source_observation_id
                )
                is None
            ):
                raise ValueError(
                    "enriched_source_observation_id must be a lowercase SHA-256"
                )
            for reason in (
                MISSING_GAME_NUMBER,
                MISSING_SOURCE_OBSERVATION_ID,
                AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH,
            ):
                if reason in self.quarantine_reasons:
                    raise ValueError(
                        f"{reason} must not be present on an enriched row"
                    )
        else:
            for reason in (MISSING_GAME_NUMBER, MISSING_SOURCE_OBSERVATION_ID):
                if reason not in self.quarantine_reasons:
                    raise ValueError(
                        f"{reason} must be present on an unenriched row"
                    )

        if self.prediction_candidate.sp_fip_delta.is_zero():
            if ZERO_DELTA_SELECTION_POLICY_UNRESOLVED not in self.quarantine_reasons:
                raise ValueError(
                    "ZERO_DELTA_SELECTION_POLICY_UNRESOLVED must be present"
                    " when sp_fip_delta is zero"
                )
        elif ZERO_DELTA_SELECTION_POLICY_UNRESOLVED in self.quarantine_reasons:
            raise ValueError(
                "ZERO_DELTA_SELECTION_POLICY_UNRESOLVED must not be present"
                " when sp_fip_delta is non-zero"
            )
