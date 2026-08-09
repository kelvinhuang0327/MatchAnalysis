"""Immutable, point-in-time-safe P20A P13 walk-forward fold contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Optional, Tuple


P20A_WALK_FORWARD_FOLD_SCHEMA_VERSION = "p20a.moneyline_walk_forward_fold.v1"
P20A_FEATURE_NAMES = (
    "indep_recent_win_rate_delta",
    "indep_starter_era_delta",
)
P20A_MODEL_TYPE = "logistic_regression"
P20A_SOLVER = "lbfgs"
P20A_MAX_ITER = 1000


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _parse_date(value: str, field_name: str) -> date:
    _require_text(value, field_name)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


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


@dataclass(frozen=True)
class WalkForwardRow:
    """One committed P13 matrix row in canonical feature order."""

    game_id: str
    date: str
    feature_values: Tuple[str, ...]
    target_home_win: Optional[int] = None
    home_team: str = ""
    away_team: str = ""
    scheduled_start_utc: str = ""
    source_schedule_observation_id: str = ""

    def __post_init__(self) -> None:
        _require_text(self.game_id, "game_id")
        _parse_date(self.date, "date")
        if not isinstance(self.feature_values, tuple):
            raise TypeError("feature_values must be a tuple")
        if len(self.feature_values) != len(P20A_FEATURE_NAMES):
            raise ValueError("feature_values must match P20A feature order")
        for index, value in enumerate(self.feature_values):
            _finite_float(value, f"feature_values[{index}]")
        if self.target_home_win is not None:
            if self.target_home_win not in (0, 1):
                raise ValueError("target_home_win must be zero, one, or omitted")
        for field_name in (
            "home_team",
            "away_team",
            "scheduled_start_utc",
            "source_schedule_observation_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")

    @property
    def feature_vector(self) -> Tuple[float, ...]:
        return tuple(
            _finite_float(value, f"feature_values[{index}]")
            for index, value in enumerate(self.feature_values)
        )

    def to_projection(self) -> dict[str, Any]:
        projection: dict[str, Any] = {
            "game_id": self.game_id,
            "date": self.date,
            "features": {
                name: value
                for name, value in zip(P20A_FEATURE_NAMES, self.feature_values)
            },
        }
        if self.target_home_win is not None:
            projection["target_home_win"] = self.target_home_win
        if self.home_team:
            projection["home_team"] = self.home_team
        if self.away_team:
            projection["away_team"] = self.away_team
        if self.scheduled_start_utc:
            projection["scheduled_start_utc"] = self.scheduled_start_utc
        if self.source_schedule_observation_id:
            projection["source_schedule_observation_id"] = (
                self.source_schedule_observation_id
            )
        return projection

    @classmethod
    def from_projection(cls, projection: Mapping[str, Any]) -> "WalkForwardRow":
        if not isinstance(projection, Mapping):
            raise TypeError("walk-forward row must be a mapping")
        try:
            feature_projection = projection["features"]
            feature_values = tuple(
                str(feature_projection[name]) for name in P20A_FEATURE_NAMES
            )
            target = projection.get("target_home_win")
            target_value = None if target is None else int(target)
            return cls(
                game_id=str(projection["game_id"]),
                date=str(projection["date"]),
                feature_values=feature_values,
                target_home_win=target_value,
                home_team=str(projection.get("home_team", "")),
                away_team=str(projection.get("away_team", "")),
                scheduled_start_utc=str(
                    projection.get("scheduled_start_utc", "")
                ),
                source_schedule_observation_id=str(
                    projection.get("source_schedule_observation_id", "")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid walk-forward row") from exc


@dataclass(frozen=True)
class MoneylineWalkForwardFold:
    """A bounded expanding-window fold with explicit PIT boundaries."""

    fold_id: str
    train_as_of: str
    validation_start: str
    validation_end: str
    feature_names: Tuple[str, ...]
    training_rows: Tuple[WalkForwardRow, ...]
    prediction_rows: Tuple[WalkForwardRow, ...]
    expected_home_probabilities: Tuple[str, ...]
    legacy_source_commit: str
    legacy_source_tree: str
    legacy_source_paths: Tuple[str, ...]
    min_train_size: int = 300
    initial_train_months: int = 2
    validation_months: int = 1
    schema_version: str = P20A_WALK_FORWARD_FOLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.fold_id, "fold_id")
        train_as_of = _parse_date(self.train_as_of, "train_as_of")
        validation_start = _parse_date(self.validation_start, "validation_start")
        validation_end = _parse_date(self.validation_end, "validation_end")
        if not train_as_of < validation_start <= validation_end:
            raise ValueError("fold boundaries must satisfy train_as_of < validation range")
        if self.schema_version != P20A_WALK_FORWARD_FOLD_SCHEMA_VERSION:
            raise ValueError("unexpected P20A fold schema")
        if self.feature_names != P20A_FEATURE_NAMES:
            raise ValueError("feature_names must match the committed P13 order")
        if not self.training_rows:
            raise ValueError("training_rows must not be empty")
        if len(self.training_rows) < self.min_train_size:
            raise ValueError("training_rows must satisfy min_train_size")
        if len(self.prediction_rows) < 2:
            raise ValueError("prediction_rows must contain more than one row")
        if len(self.expected_home_probabilities) != len(self.prediction_rows):
            raise ValueError("expected probabilities must match prediction rows")
        if self.min_train_size <= 0 or self.initial_train_months <= 0:
            raise ValueError("fold configuration must be positive")
        if self.validation_months <= 0:
            raise ValueError("validation_months must be positive")
        _require_text(self.legacy_source_commit, "legacy_source_commit")
        _require_text(self.legacy_source_tree, "legacy_source_tree")
        if not self.legacy_source_paths:
            raise ValueError("legacy_source_paths must not be empty")
        for path in self.legacy_source_paths:
            _require_text(path, "legacy_source_path")
        previous = None
        for row in self.training_rows:
            row_date = _parse_date(row.date, "training row date")
            if row.target_home_win is None:
                raise ValueError("training rows require historical targets")
            if row_date > train_as_of:
                raise ValueError("training row occurs after train_as_of")
            if previous is not None and row_date < previous:
                raise ValueError("training rows must be ordered by date")
            previous = row_date
        for row in self.prediction_rows:
            row_date = _parse_date(row.date, "prediction row date")
            if not validation_start <= row_date <= validation_end:
                raise ValueError("prediction row is outside the validation range")
            if row_date <= train_as_of:
                raise ValueError("prediction row is not strictly after train_as_of")
        for probability in self.expected_home_probabilities:
            probability_value = _finite_float(probability, "expected probability")
            if not 0.0 < probability_value < 1.0:
                raise ValueError("expected probability must be between zero and one")

    @property
    def training_row_count(self) -> int:
        return len(self.training_rows)

    @property
    def prediction_row_count(self) -> int:
        return len(self.prediction_rows)

    def point_in_time_safe(self) -> bool:
        return all(
            _parse_date(row.date, "training row date")
            <= _parse_date(self.train_as_of, "train_as_of")
            < _parse_date(prediction.date, "prediction row date")
            for row in self.training_rows
            for prediction in self.prediction_rows
        )

    def to_projection(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fold_id": self.fold_id,
            "train_as_of": self.train_as_of,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "feature_names": list(self.feature_names),
            "training_row_count": self.training_row_count,
            "prediction_row_count": self.prediction_row_count,
            "training_rows": [row.to_projection() for row in self.training_rows],
            "prediction_rows": [row.to_projection() for row in self.prediction_rows],
            "expected_home_probabilities": list(self.expected_home_probabilities),
            "legacy_source_commit": self.legacy_source_commit,
            "legacy_source_tree": self.legacy_source_tree,
            "legacy_source_paths": list(self.legacy_source_paths),
            "min_train_size": self.min_train_size,
            "initial_train_months": self.initial_train_months,
            "validation_months": self.validation_months,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_projection())

    def fingerprint(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_projection(cls, projection: Mapping[str, Any]) -> "MoneylineWalkForwardFold":
        if not isinstance(projection, Mapping):
            raise TypeError("fold projection must be a mapping")
        try:
            fold = cls(
                fold_id=str(projection["fold_id"]),
                train_as_of=str(projection["train_as_of"]),
                validation_start=str(projection["validation_start"]),
                validation_end=str(projection["validation_end"]),
                feature_names=tuple(str(item) for item in projection["feature_names"]),
                training_rows=tuple(
                    WalkForwardRow.from_projection(item)
                    for item in projection["training_rows"]
                ),
                prediction_rows=tuple(
                    WalkForwardRow.from_projection(item)
                    for item in projection["prediction_rows"]
                ),
                expected_home_probabilities=tuple(
                    str(item) for item in projection["expected_home_probabilities"]
                ),
                legacy_source_commit=str(projection["legacy_source_commit"]),
                legacy_source_tree=str(projection["legacy_source_tree"]),
                legacy_source_paths=tuple(
                    str(item) for item in projection["legacy_source_paths"]
                ),
                min_train_size=int(projection.get("min_train_size", 300)),
                initial_train_months=int(projection.get("initial_train_months", 2)),
                validation_months=int(projection.get("validation_months", 1)),
                schema_version=str(
                    projection.get(
                        "schema_version", P20A_WALK_FORWARD_FOLD_SCHEMA_VERSION
                    )
                ),
            )
            if "training_row_count" in projection and int(
                projection["training_row_count"]
            ) != fold.training_row_count:
                raise ValueError("training_row_count does not match rows")
            if "prediction_row_count" in projection and int(
                projection["prediction_row_count"]
            ) != fold.prediction_row_count:
                raise ValueError("prediction_row_count does not match rows")
            return fold
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid P20A fold projection") from exc


@dataclass(frozen=True)
class ReconstructedWalkForwardModel:
    """Serializable fitted state produced by the exact legacy fit settings."""

    fold_id: str
    feature_names: Tuple[str, ...]
    coefficients: Tuple[float, ...]
    intercept: float
    scaler_means: Tuple[float, ...]
    scaler_stds: Tuple[float, ...]
    train_size: int
    model_type: str = P20A_MODEL_TYPE
    solver: str = P20A_SOLVER
    max_iter: int = P20A_MAX_ITER

    def __post_init__(self) -> None:
        if self.feature_names != P20A_FEATURE_NAMES:
            raise ValueError("model feature_names must match P13 order")
        for field_name in ("coefficients", "scaler_means", "scaler_stds"):
            values = getattr(self, field_name)
            if len(values) != len(self.feature_names):
                raise ValueError(f"{field_name} must match feature_names")
            for index, value in enumerate(values):
                _finite_float(value, f"{field_name}[{index}]")
        _finite_float(self.intercept, "intercept")
        if any(value == 0.0 for value in self.scaler_stds):
            raise ValueError("scaler_stds must be non-zero")
        if self.train_size <= 0:
            raise ValueError("train_size must be positive")
        if self.model_type != P20A_MODEL_TYPE or self.solver != P20A_SOLVER:
            raise ValueError("unexpected P13 fit configuration")
        if self.max_iter != P20A_MAX_ITER:
            raise ValueError("unexpected P13 max_iter")

    def home_probability(self, feature_values: Tuple[float, ...]) -> float:
        if len(feature_values) != len(self.feature_names):
            raise ValueError("feature vector must match model feature order")
        standardized = tuple(
            (float(value) - mean) / std
            for value, mean, std in zip(
                feature_values, self.scaler_means, self.scaler_stds
            )
        )
        logit = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized)
        )
        probability = 1.0 / (1.0 + math.exp(-logit))
        return max(1e-6, min(1.0 - 1e-6, probability))

    def to_projection(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "feature_names": list(self.feature_names),
            "coefficients": [repr(value) for value in self.coefficients],
            "intercept": repr(self.intercept),
            "scaler_means": [repr(value) for value in self.scaler_means],
            "scaler_stds": [repr(value) for value in self.scaler_stds],
            "train_size": self.train_size,
            "model_type": self.model_type,
            "solver": self.solver,
            "max_iter": self.max_iter,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_projection())

    def fingerprint(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()
