"""Immutable result-only paper decisions and settlements (P18A).

This module models the smallest useful paper-decision contract: a selected
side is frozen from prediction-time evidence, then a later final result may be
attached to derive a result-only status.  It deliberately contains no price,
payout, profit, ROI, EV, Kelly, training, provider, persistence, or runtime
integration concepts.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any


DECISION_SCHEMA_VERSION = "p18a.result_only_paper_decision.v1"
SETTLEMENT_SCHEMA_VERSION = "p18a.result_only_paper_settlement.v1"

SETTLEMENT_WON = "WON"
SETTLEMENT_LOST = "LOST"
SETTLEMENT_UNSETTLED = "UNSETTLED"

_SHA256_LENGTH = 64


def _require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")


def _require_sha256(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


def _require_positive_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def compute_decision_id(
    *,
    prediction_observation_id: str,
    source_snapshot_row_fingerprint: str,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    selection: str,
) -> str:
    """Compute an ID from prediction-time identity and selected side only."""

    payload = {
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "game_number": game_number,
        "prediction_observation_id": prediction_observation_id,
        "provider_game_id": provider_game_id,
        "provider_namespace": provider_namespace,
        "selection": selection,
        "source_snapshot_row_fingerprint": source_snapshot_row_fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResultOnlyPaperDecision:
    """A frozen paper decision selected before any outcome is available."""

    decision_id: str
    prediction_observation_id: str
    source_snapshot_row_fingerprint: str
    provider_namespace: str
    provider_game_id: str
    game_number: int
    selection: str
    prediction_generated_at_utc: str
    scheduled_start_utc: str

    def __post_init__(self) -> None:
        for field_name in (
            "decision_id",
            "prediction_observation_id",
            "source_snapshot_row_fingerprint",
            "provider_namespace",
            "provider_game_id",
            "selection",
            "prediction_generated_at_utc",
            "scheduled_start_utc",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_sha256(self.decision_id, "decision_id")
        _require_sha256(
            self.source_snapshot_row_fingerprint,
            "source_snapshot_row_fingerprint",
        )
        _require_positive_integer(self.game_number, "game_number")
        if self.selection not in ("HOME", "AWAY"):
            raise ValueError("selection must be HOME or AWAY")
        expected_id = compute_decision_id(
            prediction_observation_id=self.prediction_observation_id,
            source_snapshot_row_fingerprint=self.source_snapshot_row_fingerprint,
            provider_namespace=self.provider_namespace,
            provider_game_id=self.provider_game_id,
            game_number=self.game_number,
            selection=self.selection,
        )
        if self.decision_id != expected_id:
            raise ValueError("decision_id does not match prediction-time fields")


def compute_decision_set_fingerprint(
    decisions: tuple[ResultOnlyPaperDecision, ...],
) -> str:
    """Compute an order-sensitive fingerprint over frozen decision IDs."""

    payload = "".join(
        f"{decision.decision_id}:{decision.source_snapshot_row_fingerprint}\n"
        for decision in decisions
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResultOnlyDecisionSelection:
    """The immutable decision set that precedes outcome attachment."""

    schema_version: str
    source_snapshot_sha256: str
    source_snapshot_summary_sha256: str
    source_snapshot_fingerprint: str
    excluded_row_count: int
    decisions: tuple[ResultOnlyPaperDecision, ...]
    decision_set_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != DECISION_SCHEMA_VERSION:
            raise ValueError("unexpected decision schema version")
        for field_name in (
            "source_snapshot_sha256",
            "source_snapshot_summary_sha256",
            "source_snapshot_fingerprint",
            "decision_set_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if (
            isinstance(self.excluded_row_count, bool)
            or not isinstance(self.excluded_row_count, int)
            or self.excluded_row_count < 0
        ):
            raise ValueError("excluded_row_count must be a non-negative integer")
        if not isinstance(self.decisions, tuple):
            raise TypeError("decisions must be a tuple")
        if not all(isinstance(item, ResultOnlyPaperDecision) for item in self.decisions):
            raise TypeError("decisions must contain ResultOnlyPaperDecision values")
        ids = tuple(item.decision_id for item in self.decisions)
        if len(set(ids)) != len(ids):
            raise ValueError("decisions must not contain duplicate decision_id values")
        if ids != tuple(sorted(ids)):
            raise ValueError("decisions must be ordered by ascending decision_id")
        if self.decision_set_fingerprint != compute_decision_set_fingerprint(
            self.decisions
        ):
            raise ValueError("decision_set_fingerprint does not match decisions")


def settlement_status_for(selection: str, actual_winner: str) -> str:
    """Derive a result-only status from a frozen side and final winner."""

    if selection not in ("HOME", "AWAY"):
        raise ValueError("selection must be HOME or AWAY")
    if actual_winner not in ("HOME", "AWAY"):
        raise ValueError("actual_winner must be HOME or AWAY")
    return SETTLEMENT_WON if selection == actual_winner else SETTLEMENT_LOST


def compute_settlement_row_fingerprint(
    *,
    decision_id: str,
    prediction_observation_id: str,
    result_observation_id: str | None,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    selection: str,
    settlement_status: str,
    result_observed_at_utc: str | None,
    home_score: int | None,
    away_score: int | None,
    actual_winner: str | None,
) -> str:
    """Compute a deterministic fingerprint for one result-only settlement."""

    payload = {
        "actual_winner": actual_winner,
        "away_score": away_score,
        "decision_id": decision_id,
        "game_number": game_number,
        "home_score": home_score,
        "prediction_observation_id": prediction_observation_id,
        "provider_game_id": provider_game_id,
        "provider_namespace": provider_namespace,
        "result_observation_id": result_observation_id,
        "result_observed_at_utc": result_observed_at_utc,
        "selection": selection,
        "settlement_schema_version": SETTLEMENT_SCHEMA_VERSION,
        "settlement_status": settlement_status,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResultOnlyPaperSettlement:
    """A deterministic result-only attachment for one frozen decision."""

    decision_id: str
    prediction_observation_id: str
    result_observation_id: str | None
    provider_namespace: str
    provider_game_id: str
    game_number: int
    selection: str
    settlement_status: str
    result_observed_at_utc: str | None
    home_score: int | None
    away_score: int | None
    actual_winner: str | None
    settlement_row_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "decision_id",
            "prediction_observation_id",
            "provider_namespace",
            "provider_game_id",
            "selection",
            "settlement_status",
            "settlement_row_fingerprint",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_sha256(self.decision_id, "decision_id")
        _require_sha256(
            self.settlement_row_fingerprint,
            "settlement_row_fingerprint",
        )
        _require_positive_integer(self.game_number, "game_number")
        if self.selection not in ("HOME", "AWAY"):
            raise ValueError("selection must be HOME or AWAY")
        if self.settlement_status not in (
            SETTLEMENT_WON,
            SETTLEMENT_LOST,
            SETTLEMENT_UNSETTLED,
        ):
            raise ValueError("unexpected settlement_status")
        if self.settlement_status == SETTLEMENT_UNSETTLED:
            if any(
                value is not None
                for value in (
                    self.result_observation_id,
                    self.result_observed_at_utc,
                    self.home_score,
                    self.away_score,
                    self.actual_winner,
                )
            ):
                raise ValueError("UNSETTLED rows cannot contain result fields")
        else:
            if self.result_observation_id is None or self.actual_winner is None:
                raise ValueError("settled rows require result identity and winner")
            if self.home_score is None or self.away_score is None:
                raise ValueError("settled rows require both scores")
            if self.result_observed_at_utc is None:
                raise ValueError("settled rows require result_observed_at_utc")
            if settlement_status_for(self.selection, self.actual_winner) != self.settlement_status:
                raise ValueError("settlement_status does not match the selected side")
        expected_fingerprint = compute_settlement_row_fingerprint(
            decision_id=self.decision_id,
            prediction_observation_id=self.prediction_observation_id,
            result_observation_id=self.result_observation_id,
            provider_namespace=self.provider_namespace,
            provider_game_id=self.provider_game_id,
            game_number=self.game_number,
            selection=self.selection,
            settlement_status=self.settlement_status,
            result_observed_at_utc=self.result_observed_at_utc,
            home_score=self.home_score,
            away_score=self.away_score,
            actual_winner=self.actual_winner,
        )
        if self.settlement_row_fingerprint != expected_fingerprint:
            raise ValueError("settlement_row_fingerprint does not match row fields")


def compute_settlement_set_fingerprint(
    settlements: tuple[ResultOnlyPaperSettlement, ...],
) -> str:
    """Compute an order-sensitive fingerprint over settlement rows."""

    payload = "".join(
        f"{settlement.decision_id}:{settlement.settlement_row_fingerprint}\n"
        for settlement in settlements
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResultOnlyPaperDecisionReplay:
    """Immutable P18A output containing frozen decisions and later statuses."""

    schema_version: str
    settlement_schema_version: str
    source_snapshot_sha256: str
    source_snapshot_summary_sha256: str
    source_snapshot_fingerprint: str
    final_results_sha256: str
    selection: ResultOnlyDecisionSelection
    settlements: tuple[ResultOnlyPaperSettlement, ...]
    settled_count: int
    unsettled_count: int
    won_count: int
    lost_count: int
    settlement_set_fingerprint: str
    claims: dict[str, bool]

    def __post_init__(self) -> None:
        if self.schema_version != DECISION_SCHEMA_VERSION:
            raise ValueError("unexpected decision schema version")
        if self.settlement_schema_version != SETTLEMENT_SCHEMA_VERSION:
            raise ValueError("unexpected settlement schema version")
        for field_name in (
            "source_snapshot_sha256",
            "source_snapshot_summary_sha256",
            "source_snapshot_fingerprint",
            "final_results_sha256",
            "settlement_set_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.selection.source_snapshot_fingerprint != self.source_snapshot_fingerprint:
            raise ValueError("selection source fingerprint mismatch")
        if not isinstance(self.settlements, tuple):
            raise TypeError("settlements must be a tuple")
        expected_ids = tuple(item.decision_id for item in self.selection.decisions)
        actual_ids = tuple(item.decision_id for item in self.settlements)
        if actual_ids != expected_ids:
            raise ValueError("settlements must preserve frozen decision ID order")
        if self.settled_count + self.unsettled_count != len(self.settlements):
            raise ValueError("settled and unsettled counts must cover settlements")
        if self.won_count + self.lost_count != self.settled_count:
            raise ValueError("won and lost counts must cover settled rows")
        if self.settlement_set_fingerprint != compute_settlement_set_fingerprint(
            self.settlements
        ):
            raise ValueError("settlement_set_fingerprint does not match settlements")
