"""Immutable evidence for one admitted prospective prediction observation.

This module only defines the contract. Production construction of
PredictionSourceObservation must pass through the admission evaluator in
prediction_admission.py; nothing here admits or promotes a prediction on its
own, and this module carries no market-specific interpretation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
import re

from .canonical_utc import format_canonical_utc


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256")


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_finite_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")


def _require_utc(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be normalized to UTC")


def compute_prediction_observation_id(
    *,
    source_prediction_id: str,
    model_id: str,
    market_id: str,
    selection: str,
    model_probability: Decimal,
    line_value: Decimal,
    push_policy: str,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    source_schedule_observation_id: str,
    prediction_generated_at_utc: datetime,
    response_received_at_utc: datetime,
    ingested_at_utc: datetime,
    scheduled_start_utc: datetime,
) -> str:
    """Hash the exact canonical evidence projection plus one final LF."""

    projection = {
        "source_prediction_id": source_prediction_id,
        "model_id": model_id,
        "market_id": market_id,
        "selection": selection,
        "model_probability": str(model_probability),
        "line_value": str(line_value),
        "push_policy": push_policy,
        "provider_namespace": provider_namespace,
        "provider_game_id": provider_game_id,
        "game_number": game_number,
        "source_schedule_observation_id": source_schedule_observation_id,
        "prediction_generated_at_utc": format_canonical_utc(
            prediction_generated_at_utc
        ),
        "response_received_at_utc": format_canonical_utc(
            response_received_at_utc
        ),
        "ingested_at_utc": format_canonical_utc(ingested_at_utc),
        "scheduled_start_utc": format_canonical_utc(scheduled_start_utc),
    }
    canonical_line = (
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return sha256(canonical_line).hexdigest()


@dataclass(frozen=True, slots=True)
class PredictionSourceObservation:
    """Validated, immutable evidence for one admitted prospective prediction."""

    prediction_observation_id: str
    source_prediction_id: str
    model_id: str
    market_id: str
    selection: str
    model_probability: Decimal
    line_value: Decimal
    push_policy: str
    provider_namespace: str
    provider_game_id: str
    game_number: int
    source_schedule_observation_id: str
    prediction_generated_at_utc: datetime
    response_received_at_utc: datetime
    ingested_at_utc: datetime
    scheduled_start_utc: datetime

    def __post_init__(self) -> None:
        for field_name in (
            "source_prediction_id",
            "model_id",
            "market_id",
            "selection",
            "push_policy",
            "provider_namespace",
            "provider_game_id",
        ):
            _require_explicit(getattr(self, field_name), field_name)

        _require_positive_integer(self.game_number, "game_number")

        _require_finite_decimal(self.model_probability, "model_probability")
        if not (Decimal("0") <= self.model_probability <= Decimal("1")):
            raise ValueError("model_probability must be within [0, 1]")
        _require_finite_decimal(self.line_value, "line_value")

        _require_sha256(
            self.source_schedule_observation_id,
            "source_schedule_observation_id",
        )

        for field_name in (
            "prediction_generated_at_utc",
            "response_received_at_utc",
            "ingested_at_utc",
            "scheduled_start_utc",
        ):
            _require_utc(getattr(self, field_name), field_name)

        if not (
            self.prediction_generated_at_utc
            <= self.response_received_at_utc
            <= self.ingested_at_utc
            < self.scheduled_start_utc
        ):
            raise ValueError(
                "timestamps must satisfy generated <= received <= ingested"
                " < scheduled_start_utc"
            )

        _require_sha256(self.prediction_observation_id, "prediction_observation_id")
        expected_id = compute_prediction_observation_id(
            source_prediction_id=self.source_prediction_id,
            model_id=self.model_id,
            market_id=self.market_id,
            selection=self.selection,
            model_probability=self.model_probability,
            line_value=self.line_value,
            push_policy=self.push_policy,
            provider_namespace=self.provider_namespace,
            provider_game_id=self.provider_game_id,
            game_number=self.game_number,
            source_schedule_observation_id=self.source_schedule_observation_id,
            prediction_generated_at_utc=self.prediction_generated_at_utc,
            response_received_at_utc=self.response_received_at_utc,
            ingested_at_utc=self.ingested_at_utc,
            scheduled_start_utc=self.scheduled_start_utc,
        )
        if self.prediction_observation_id != expected_id:
            raise ValueError(
                "prediction_observation_id must match the canonical"
                " prediction-observation projection"
            )
