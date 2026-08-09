"""Deterministic P20A fold/model artifact construction and serialization."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Tuple, Union

from ...baseball.domain.moneyline_walk_forward_fold import (
    MoneylineWalkForwardFold,
    ReconstructedWalkForwardModel,
)


LEGACY_SOURCE_REPOSITORY = "/Users/kelvin/Kelvin-WorkSpace/" + "Betting-pool"
P20A_MODEL_VERSION = "p13_walk_forward_logistic_v1"
P20A_ARTIFACT_SCHEMA_VERSION = "p20a.moneyline_walk_forward_artifact.v1"


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


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def build_moneyline_model_artifact(
    fold: MoneylineWalkForwardFold,
    model: ReconstructedWalkForwardModel,
):
    """Compose fitted P20A state into the existing P19A ModelArtifact contract."""

    from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
    from ...baseball.domain.moneyline_feature_snapshot import MONEYLINE_FEATURE_NAMES

    if model.fold_id != fold.fold_id:
        raise ValueError("model and fold identifiers must match")
    if len(model.feature_names) != len(MONEYLINE_FEATURE_NAMES):
        raise ValueError("P20A model must contain the P19A feature count")
    basis_fingerprint = sha256(
        fold.canonical_bytes() + model.canonical_bytes()
    ).hexdigest()
    return MoneylineModelArtifact(
        model_id=f"p13_walk_forward_logistic_v1_{fold.fold_id}",
        model_version=P20A_MODEL_VERSION,
        feature_names=MONEYLINE_FEATURE_NAMES,
        coefficients=tuple(_decimal(value) for value in model.coefficients),
        intercept=_decimal(model.intercept),
        scaler_means=tuple(_decimal(value) for value in model.scaler_means),
        scaler_stds=tuple(_decimal(value) for value in model.scaler_stds),
        legacy_source_repository=LEGACY_SOURCE_REPOSITORY,
        legacy_source_commit=fold.legacy_source_commit,
        legacy_source_tree=fold.legacy_source_tree,
        legacy_source_paths=fold.legacy_source_paths,
        artifact_kind="bounded_deterministic_fixture",
        fixture_basis_id=basis_fingerprint,
        fixture_expected_home_probability=Decimal(
            fold.expected_home_probabilities[0]
        ),
        fixture_expected_probability_tolerance=Decimal("0.000001"),
    )


def render_reconstructed_model_json(model: ReconstructedWalkForwardModel) -> str:
    return json.dumps(
        {
            "schema_version": "p20a.reconstructed_model.v1",
            **model.to_projection(),
            "fingerprint": model.fingerprint(),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def load_reconstructed_model(path: Union[str, Path]) -> ReconstructedWalkForwardModel:
    """Load a previously reconstructed deterministic state without fitting."""

    projection = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReconstructedWalkForwardModel(
        fold_id=str(projection["fold_id"]),
        feature_names=tuple(str(item) for item in projection["feature_names"]),
        coefficients=tuple(float(item) for item in projection["coefficients"]),
        intercept=float(projection["intercept"]),
        scaler_means=tuple(float(item) for item in projection["scaler_means"]),
        scaler_stds=tuple(float(item) for item in projection["scaler_stds"]),
        train_size=int(projection["train_size"]),
        model_type=str(projection.get("model_type", "logistic_regression")),
        solver=str(projection.get("solver", "lbfgs")),
        max_iter=int(projection.get("max_iter", 1000)),
    )


def write_moneyline_walk_forward_model_artifact(
    output_dir: Union[str, Path],
    fold: MoneylineWalkForwardFold,
    model: ReconstructedWalkForwardModel,
    model_artifact,
) -> None:
    """Write only deterministic P20A model/fold projections."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fold.json").write_text(
        json.dumps(
            fold.to_projection(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "reconstructed_model.json").write_text(
        render_reconstructed_model_json(model),
        encoding="utf-8",
    )
    (directory / "model_artifact.json").write_text(
        json.dumps(
            model_artifact.to_projection(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
