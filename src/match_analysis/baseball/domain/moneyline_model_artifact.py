"""Immutable deterministic representation of the selected legacy P13 model."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from .moneyline_feature_snapshot import (
    MONEYLINE_FEATURE_NAMES,
    MoneylineFeatureSnapshot,
)


MONEYLINE_MODEL_ARTIFACT_SCHEMA_VERSION = "p19a.moneyline_model_artifact.v1"
_GIT_OBJECT_ID_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _require_git_object_id(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    if _GIT_OBJECT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 40-character Git object ID")


def _require_finite_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")


def _canonical_json_bytes(projection: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class MoneylineModelArtifact:
    """A no-training, no-network artifact for P13 logistic inference."""

    model_id: str
    model_version: str
    feature_names: tuple[str, ...]
    coefficients: tuple[Decimal, ...]
    intercept: Decimal
    scaler_means: tuple[Decimal, ...]
    scaler_stds: tuple[Decimal, ...]
    legacy_source_repository: str
    legacy_source_commit: str
    legacy_source_tree: str
    legacy_source_paths: tuple[str, ...]
    schema_version: str = MONEYLINE_MODEL_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "model_id",
            "model_version",
            "legacy_source_repository",
            "legacy_source_commit",
            "legacy_source_tree",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_git_object_id(self.legacy_source_commit, "legacy_source_commit")
        _require_git_object_id(self.legacy_source_tree, "legacy_source_tree")
        if self.schema_version != MONEYLINE_MODEL_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unexpected Moneyline model artifact schema")
        if self.feature_names != MONEYLINE_FEATURE_NAMES:
            raise ValueError("artifact feature_names must match the P13 feature order")
        vector_fields = ("coefficients", "scaler_means", "scaler_stds")
        for field_name in vector_fields:
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or len(values) != len(self.feature_names):
                raise ValueError(f"{field_name} must match feature_names length")
            for index, value in enumerate(values):
                _require_finite_decimal(value, f"{field_name}[{index}]")
        _require_finite_decimal(self.intercept, "intercept")
        if any(value == 0 for value in self.scaler_stds):
            raise ValueError("scaler_stds must be non-zero")
        if not isinstance(self.legacy_source_paths, tuple) or not self.legacy_source_paths:
            raise ValueError("legacy_source_paths must be a non-empty tuple")
        for path in self.legacy_source_paths:
            _require_text(path, "legacy_source_path")

    @classmethod
    def from_projection(cls, projection: Mapping[str, Any]) -> "MoneylineModelArtifact":
        """Parse a deterministic JSON projection without training or fetching."""

        if not isinstance(projection, Mapping):
            raise TypeError("model artifact projection must be a mapping")
        try:
            return cls(
                model_id=str(projection["model_id"]),
                model_version=str(projection["model_version"]),
                feature_names=tuple(str(item) for item in projection["feature_names"]),
                coefficients=tuple(
                    Decimal(str(item)) for item in projection["coefficients"]
                ),
                intercept=Decimal(str(projection["intercept"])),
                scaler_means=tuple(
                    Decimal(str(item)) for item in projection["scaler_means"]
                ),
                scaler_stds=tuple(
                    Decimal(str(item)) for item in projection["scaler_stds"]
                ),
                legacy_source_repository=str(projection["legacy_source_repository"]),
                legacy_source_commit=str(projection["legacy_source_commit"]),
                legacy_source_tree=str(projection["legacy_source_tree"]),
                legacy_source_paths=tuple(
                    str(item) for item in projection["legacy_source_paths"]
                ),
                schema_version=str(
                    projection.get("schema_version", MONEYLINE_MODEL_ARTIFACT_SCHEMA_VERSION)
                ),
            )
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            raise ValueError("invalid Moneyline model artifact projection") from exc

    def to_projection(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "feature_names": list(self.feature_names),
            "coefficients": [str(value) for value in self.coefficients],
            "intercept": str(self.intercept),
            "scaler_means": [str(value) for value in self.scaler_means],
            "scaler_stds": [str(value) for value in self.scaler_stds],
            "legacy_source_repository": self.legacy_source_repository,
            "legacy_source_commit": self.legacy_source_commit,
            "legacy_source_tree": self.legacy_source_tree,
            "legacy_source_paths": list(self.legacy_source_paths),
        }

    def canonical_bytes(self) -> bytes:
        """Serialize the exact model artifact deterministically."""

        return _canonical_json_bytes(self.to_projection())

    def fingerprint(self) -> str:
        """Return the stable SHA-256 fingerprint of canonical artifact bytes."""

        return sha256(self.canonical_bytes()).hexdigest()

    def predict_home_probability(self, snapshot: MoneylineFeatureSnapshot) -> Decimal:
        """Apply the legacy P13 standardized logistic inference semantics."""

        if not isinstance(snapshot, MoneylineFeatureSnapshot):
            raise TypeError("snapshot must be a MoneylineFeatureSnapshot")
        standardized = tuple(
            (value - mean) / std
            for value, mean, std in zip(
                snapshot.feature_vector(),
                self.scaler_means,
                self.scaler_stds,
                strict=True,
            )
        )
        logit = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized, strict=True)
        )
        probability = Decimal(1) / (Decimal(1) + (-logit).exp())
        return min(Decimal("0.999999"), max(Decimal("0.000001"), probability))
