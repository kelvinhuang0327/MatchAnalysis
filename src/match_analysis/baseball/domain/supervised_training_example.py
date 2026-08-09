"""Immutable game-level supervised-learning examples for P22A."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from .canonical_utc import format_canonical_utc, parse_canonical_utc
from .moneyline_feature_snapshot import MONEYLINE_FEATURE_NAMES


SUPERVISED_TRAINING_EXAMPLE_SCHEMA_VERSION = "p22a.supervised_training_example.v1"
P22A_FEATURE_NAMES = MONEYLINE_FEATURE_NAMES
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(projection: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _require_sha256(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256")


def _require_utc_text(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    parsed = parse_canonical_utc(value)
    if format_canonical_utc(parsed) != value:
        raise ValueError(f"{field_name} must use canonical UTC Z-form")


def _require_finite_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def compute_training_example_id(
    *,
    schema_version: str,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    scheduled_start_utc: str,
    feature_as_of_utc: str,
    fold_id: str,
    fold_fingerprint: str,
    model_id: str,
    model_fingerprint: str,
    feature_snapshot_fingerprint: str,
    source_schedule_observation_id: str,
) -> str:
    """Hash immutable game/feature/contract identity, excluding the target."""

    return sha256(
        _canonical_json_bytes(
            {
                "feature_as_of_utc": feature_as_of_utc,
                "feature_snapshot_fingerprint": feature_snapshot_fingerprint,
                "fold_fingerprint": fold_fingerprint,
                "fold_id": fold_id,
                "game_number": game_number,
                "model_fingerprint": model_fingerprint,
                "model_id": model_id,
                "provider_game_id": provider_game_id,
                "provider_namespace": provider_namespace,
                "scheduled_start_utc": scheduled_start_utc,
                "schema_version": schema_version,
                "source_schedule_observation_id": source_schedule_observation_id,
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureLineage:
    """Provenance for one ordered pregame feature value."""

    field_name: str
    value: Decimal
    source_id: str
    source_kind: str
    observed_as_of_utc: str
    source_fingerprint: str

    def __post_init__(self) -> None:
        _require_text(self.field_name, "field_name")
        _require_finite_decimal(self.value, "value")
        _require_text(self.source_id, "source_id")
        _require_text(self.source_kind, "source_kind")
        _require_utc_text(self.observed_as_of_utc, "observed_as_of_utc")
        _require_sha256(self.source_fingerprint, "source_fingerprint")

    def to_projection(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "observed_as_of_utc": self.observed_as_of_utc,
            "source_fingerprint": self.source_fingerprint,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "value": str(self.value),
        }

    @classmethod
    def from_projection(cls, projection: Mapping[str, Any]) -> "FeatureLineage":
        return cls(
            field_name=str(projection["field_name"]),
            value=Decimal(str(projection["value"])),
            source_id=str(projection["source_id"]),
            source_kind=str(projection["source_kind"]),
            observed_as_of_utc=str(projection["observed_as_of_utc"]),
            source_fingerprint=str(projection["source_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class CandidateLineage:
    """The exact candidate-level evidence collapsed into one game example."""

    candidate_id: str
    candidate_row_fingerprint: str
    selection: str
    source_snapshot_row_fingerprint: str
    source_evaluation_row_fingerprint: str

    def __post_init__(self) -> None:
        _require_sha256(self.candidate_id, "candidate_id")
        _require_sha256(
            self.candidate_row_fingerprint,
            "candidate_row_fingerprint",
        )
        _require_text(self.selection, "selection")
        if self.selection not in ("HOME", "AWAY"):
            raise ValueError("selection must be HOME or AWAY")
        _require_sha256(
            self.source_snapshot_row_fingerprint,
            "source_snapshot_row_fingerprint",
        )
        _require_sha256(
            self.source_evaluation_row_fingerprint,
            "source_evaluation_row_fingerprint",
        )

    def to_projection(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_row_fingerprint": self.candidate_row_fingerprint,
            "selection": self.selection,
            "source_evaluation_row_fingerprint": self.source_evaluation_row_fingerprint,
            "source_snapshot_row_fingerprint": self.source_snapshot_row_fingerprint,
        }

    @classmethod
    def from_projection(cls, projection: Mapping[str, Any]) -> "CandidateLineage":
        return cls(
            candidate_id=str(projection["candidate_id"]),
            candidate_row_fingerprint=str(projection["candidate_row_fingerprint"]),
            selection=str(projection["selection"]),
            source_snapshot_row_fingerprint=str(
                projection["source_snapshot_row_fingerprint"]
            ),
            source_evaluation_row_fingerprint=str(
                projection["source_evaluation_row_fingerprint"]
            ),
        )


@dataclass(frozen=True, slots=True)
class SupervisedTrainingExample:
    """One immutable, game-level, point-in-time-safe supervised example."""

    training_example_id: str
    provider_namespace: str
    provider_game_id: str
    game_number: int
    home_participant: str
    away_participant: str
    scheduled_start_utc: str
    feature_as_of_utc: str
    fold_id: str
    fold_fingerprint: str
    model_id: str
    model_fingerprint: str
    model_artifact_fingerprint: str
    feature_snapshot_id: str
    feature_snapshot_fingerprint: str
    feature_snapshot_schema_version: str
    source_schedule_observation_id: str
    feature_names: tuple[str, ...]
    feature_values: tuple[Decimal, ...]
    feature_lineage: tuple[FeatureLineage, ...]
    target_home_win: int
    historical_result_source_id: str
    historical_result_observation_id: str
    historical_result_observed_at_utc: str
    historical_home_score: int
    historical_away_score: int
    historical_result_row_fingerprint: str
    source_candidates: tuple[CandidateLineage, ...]
    schema_version: str = SUPERVISED_TRAINING_EXAMPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.training_example_id, "training_example_id")
        for field_name in (
            "provider_namespace",
            "provider_game_id",
            "home_participant",
            "away_participant",
            "fold_id",
            "model_id",
            "feature_snapshot_id",
            "feature_snapshot_schema_version",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_positive_integer(self.game_number, "game_number")
        for field_name in (
            "fold_fingerprint",
            "model_fingerprint",
            "model_artifact_fingerprint",
            "feature_snapshot_fingerprint",
            "source_schedule_observation_id",
            "historical_result_observation_id",
            "historical_result_row_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        _require_text(
            self.historical_result_source_id,
            "historical_result_source_id",
        )
        for field_name in (
            "scheduled_start_utc",
            "feature_as_of_utc",
            "historical_result_observed_at_utc",
        ):
            _require_utc_text(getattr(self, field_name), field_name)
        scheduled = parse_canonical_utc(self.scheduled_start_utc)
        feature_as_of = parse_canonical_utc(self.feature_as_of_utc)
        result_observed = parse_canonical_utc(self.historical_result_observed_at_utc)
        if feature_as_of >= scheduled:
            raise ValueError("feature_as_of_utc must be before scheduled_start_utc")
        if result_observed < scheduled:
            raise ValueError("historical result must be observed at or after game start")
        if self.schema_version != SUPERVISED_TRAINING_EXAMPLE_SCHEMA_VERSION:
            raise ValueError("unexpected supervised training example schema")
        if self.feature_names != P22A_FEATURE_NAMES:
            raise ValueError("feature_names must match the canonical P13 order")
        if not isinstance(self.feature_values, tuple):
            raise TypeError("feature_values must be a tuple")
        if len(self.feature_values) != len(self.feature_names):
            raise ValueError("feature_values must match feature_names")
        for index, value in enumerate(self.feature_values):
            _require_finite_decimal(value, f"feature_values[{index}]")
        if not isinstance(self.feature_lineage, tuple):
            raise TypeError("feature_lineage must be a tuple")
        if len(self.feature_lineage) != len(self.feature_names):
            raise ValueError("feature_lineage must match feature_names")
        if tuple(item.field_name for item in self.feature_lineage) != self.feature_names:
            raise ValueError("feature_lineage must preserve canonical feature order")
        if tuple(item.value for item in self.feature_lineage) != self.feature_values:
            raise ValueError("feature_lineage values must match feature_values")
        for item in self.feature_lineage:
            if parse_canonical_utc(item.observed_as_of_utc) > feature_as_of:
                raise ValueError("feature lineage cannot be observed after feature_as_of_utc")
        if isinstance(self.target_home_win, bool) or self.target_home_win not in (0, 1):
            raise ValueError("target_home_win must be zero or one")
        if (
            not isinstance(self.historical_home_score, int)
            or isinstance(self.historical_home_score, bool)
            or not isinstance(self.historical_away_score, int)
            or isinstance(self.historical_away_score, bool)
        ):
            raise TypeError("historical scores must be integers")
        if self.historical_home_score == self.historical_away_score:
            raise ValueError("historical result cannot be a tie")
        expected_target = int(self.historical_home_score > self.historical_away_score)
        if self.target_home_win != expected_target:
            raise ValueError("target_home_win must derive from historical final scores")
        if not isinstance(self.source_candidates, tuple) or not self.source_candidates:
            raise ValueError("source_candidates must be a non-empty tuple")
        if tuple(item.candidate_id for item in self.source_candidates) != tuple(
            sorted(item.candidate_id for item in self.source_candidates)
        ):
            raise ValueError("source_candidates must be sorted by candidate_id")
        if len({item.candidate_id for item in self.source_candidates}) != len(
            self.source_candidates
        ):
            raise ValueError("source_candidates must be unique")
        for item in self.source_candidates:
            if not isinstance(item, CandidateLineage):
                raise TypeError("source_candidates must contain CandidateLineage values")
        expected_id = compute_training_example_id(
            schema_version=self.schema_version,
            provider_namespace=self.provider_namespace,
            provider_game_id=self.provider_game_id,
            game_number=self.game_number,
            scheduled_start_utc=self.scheduled_start_utc,
            feature_as_of_utc=self.feature_as_of_utc,
            fold_id=self.fold_id,
            fold_fingerprint=self.fold_fingerprint,
            model_id=self.model_id,
            model_fingerprint=self.model_fingerprint,
            feature_snapshot_fingerprint=self.feature_snapshot_fingerprint,
            source_schedule_observation_id=self.source_schedule_observation_id,
        )
        if self.training_example_id != expected_id:
            raise ValueError("training_example_id does not match immutable identity")

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_projection())

    def to_projection(self) -> dict[str, Any]:
        return {
            "away_participant": self.away_participant,
            "feature_as_of_utc": self.feature_as_of_utc,
            "feature_lineage": [item.to_projection() for item in self.feature_lineage],
            "feature_names": list(self.feature_names),
            "feature_snapshot_fingerprint": self.feature_snapshot_fingerprint,
            "feature_snapshot_id": self.feature_snapshot_id,
            "feature_snapshot_schema_version": self.feature_snapshot_schema_version,
            "feature_values": [str(value) for value in self.feature_values],
            "fold_fingerprint": self.fold_fingerprint,
            "fold_id": self.fold_id,
            "game_number": self.game_number,
            "historical_away_score": self.historical_away_score,
            "historical_home_score": self.historical_home_score,
            "historical_result_observed_at_utc": self.historical_result_observed_at_utc,
            "historical_result_observation_id": self.historical_result_observation_id,
            "historical_result_row_fingerprint": self.historical_result_row_fingerprint,
            "historical_result_source_id": self.historical_result_source_id,
            "home_participant": self.home_participant,
            "model_artifact_fingerprint": self.model_artifact_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "model_id": self.model_id,
            "provider_game_id": self.provider_game_id,
            "provider_namespace": self.provider_namespace,
            "scheduled_start_utc": self.scheduled_start_utc,
            "schema_version": self.schema_version,
            "source_candidates": [
                item.to_projection() for item in self.source_candidates
            ],
            "source_schedule_observation_id": self.source_schedule_observation_id,
            "training_example_id": self.training_example_id,
            "target_home_win": self.target_home_win,
        }

    @classmethod
    def from_projection(cls, projection: Mapping[str, Any]) -> "SupervisedTrainingExample":
        if not isinstance(projection, Mapping):
            raise TypeError("training example projection must be a mapping")
        return cls(
            training_example_id=str(projection["training_example_id"]),
            provider_namespace=str(projection["provider_namespace"]),
            provider_game_id=str(projection["provider_game_id"]),
            game_number=int(projection["game_number"]),
            home_participant=str(projection["home_participant"]),
            away_participant=str(projection["away_participant"]),
            scheduled_start_utc=str(projection["scheduled_start_utc"]),
            feature_as_of_utc=str(projection["feature_as_of_utc"]),
            fold_id=str(projection["fold_id"]),
            fold_fingerprint=str(projection["fold_fingerprint"]),
            model_id=str(projection["model_id"]),
            model_fingerprint=str(projection["model_fingerprint"]),
            model_artifact_fingerprint=str(projection["model_artifact_fingerprint"]),
            feature_snapshot_id=str(projection["feature_snapshot_id"]),
            feature_snapshot_fingerprint=str(projection["feature_snapshot_fingerprint"]),
            feature_snapshot_schema_version=str(
                projection["feature_snapshot_schema_version"]
            ),
            source_schedule_observation_id=str(
                projection["source_schedule_observation_id"]
            ),
            feature_names=tuple(str(item) for item in projection["feature_names"]),
            feature_values=tuple(
                Decimal(str(item)) for item in projection["feature_values"]
            ),
            feature_lineage=tuple(
                FeatureLineage.from_projection(item)
                for item in projection["feature_lineage"]
            ),
            target_home_win=int(projection["target_home_win"]),
            historical_result_source_id=str(projection["historical_result_source_id"]),
            historical_result_observation_id=str(
                projection["historical_result_observation_id"]
            ),
            historical_result_observed_at_utc=str(
                projection["historical_result_observed_at_utc"]
            ),
            historical_home_score=int(projection["historical_home_score"]),
            historical_away_score=int(projection["historical_away_score"]),
            historical_result_row_fingerprint=str(
                projection["historical_result_row_fingerprint"]
            ),
            source_candidates=tuple(
                CandidateLineage.from_projection(item)
                for item in projection["source_candidates"]
            ),
            schema_version=str(projection["schema_version"]),
        )


__all__ = (
    "CandidateLineage",
    "FeatureLineage",
    "P22A_FEATURE_NAMES",
    "SUPERVISED_TRAINING_EXAMPLE_SCHEMA_VERSION",
    "SupervisedTrainingExample",
    "compute_training_example_id",
)
