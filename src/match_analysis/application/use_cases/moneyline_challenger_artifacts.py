"""Deterministic P22B challenger artifact and summary serialization."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from ...baseball.domain.moneyline_feature_snapshot import MONEYLINE_FEATURE_NAMES
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact


P22B_MODEL_VERSION = "p22b.moneyline_logistic_challenger.v1"
P22B_ARTIFACT_SCHEMA_VERSION = "p22b.moneyline_challenger_artifact.v1"
P22B_SOURCE_PATHS = (
    "report/p22a_game_level_training_dataset/training_examples.jsonl",
    "report/p22a_game_level_training_dataset/summary.json",
    "src/match_analysis/application/use_cases/reconstruct_moneyline_walk_forward_model.py",
    "src/match_analysis/baseball/domain/moneyline_walk_forward_fold.py",
    "src/match_analysis/baseball/domain/moneyline_model_artifact.py",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def _fitted_state_projection(fitted_state: Any) -> dict[str, Any]:
    return {
        "coefficients": [str(_decimal(value)) for value in fitted_state.coefficients],
        "intercept": str(_decimal(fitted_state.intercept)),
        "scaler_means": [str(_decimal(value)) for value in fitted_state.scaler_means],
        "scaler_stds": [str(_decimal(value)) for value in fitted_state.scaler_stds],
    }


@dataclass(frozen=True, slots=True)
class MoneylineChallengerArtifact:
    """Immutable full P22B projection plus a P19A inference contract."""

    projection: Mapping[str, Any]

    def to_projection(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.projection, ensure_ascii=False))

    def canonical_bytes(self) -> bytes:
        projection = self.to_projection()
        projection.pop("artifact_fingerprint", None)
        return _canonical_json_bytes(projection)

    def fingerprint(self) -> str:
        fingerprint = _sha256(self.canonical_bytes())
        if self.projection.get("artifact_fingerprint") != fingerprint:
            raise ValueError("P22B artifact fingerprint does not match its projection")
        return fingerprint

    def to_inference_artifact(self) -> MoneylineModelArtifact:
        return MoneylineModelArtifact.from_projection(self.projection)


def build_moneyline_challenger_artifact(
    *,
    dataset: Any,
    fitted_state: Any,
    source_repository: str,
    source_commit: str,
    source_tree: str,
) -> MoneylineChallengerArtifact:
    """Compose one challenger while retaining the existing P19A artifact shape."""

    fitted_projection = _fitted_state_projection(fitted_state)
    basis_fingerprint = _sha256(
        _canonical_json_bytes(
            {
                "dataset_fingerprint": dataset.dataset_fingerprint,
                "feature_names": list(MONEYLINE_FEATURE_NAMES),
                "fitted_state": fitted_projection,
                "fit_configuration": {
                    "max_iter": 1000,
                    "model_type": "logistic_regression",
                    "solver": "lbfgs",
                },
            }
        )
    )
    model_id = f"p22b_moneyline_logistic_challenger_v1_{dataset.dataset_fingerprint[:16]}"
    base_artifact = MoneylineModelArtifact(
        model_id=model_id,
        model_version=P22B_MODEL_VERSION,
        feature_names=MONEYLINE_FEATURE_NAMES,
        coefficients=tuple(_decimal(value) for value in fitted_state.coefficients),
        intercept=_decimal(fitted_state.intercept),
        scaler_means=tuple(_decimal(value) for value in fitted_state.scaler_means),
        scaler_stds=tuple(_decimal(value) for value in fitted_state.scaler_stds),
        legacy_source_repository=source_repository,
        legacy_source_commit=source_commit,
        legacy_source_tree=source_tree,
        legacy_source_paths=P22B_SOURCE_PATHS,
        artifact_kind="bounded_deterministic_fixture",
        fixture_basis_id=basis_fingerprint,
        fixture_expected_home_probability=Decimal("0.5"),
        fixture_expected_probability_tolerance=Decimal("0.5"),
    )
    projection = {
        **base_artifact.to_projection(),
        "artifact_schema_version": P22B_ARTIFACT_SCHEMA_VERSION,
        "artifact_fingerprint": "",
        "artifact_role": "CHALLENGER",
        "claims": {
            "model_promoted": False,
            "out_of_sample_evaluated": False,
            "production_ready": False,
            "profitability_claim": False,
            "real_betting_recommendation": False,
            "training_authorized": True,
            "training_performed": True,
        },
        "feature_names": list(MONEYLINE_FEATURE_NAMES),
        "fit_configuration": {
            "max_iter": 1000,
            "model_type": "logistic_regression",
            "solver": "lbfgs",
        },
        "fitted_state_fingerprint": _sha256(_canonical_json_bytes(fitted_projection)),
        "label_distribution": dict(dataset.label_distribution),
        "label_semantics": (
            "target_home_win=1 iff committed FINAL home_score is greater than away_score; "
            "target_home_win=0 iff away_score is greater"
        ),
        "model_role": "CHALLENGER",
        "source_dataset_fingerprint": dataset.dataset_fingerprint,
        "training_code_contract": "p22b.deterministic_moneyline_challenger_training.v1",
        "training_code_paths": [
            "src/match_analysis/application/use_cases/train_moneyline_challenger.py",
            "src/match_analysis/application/use_cases/moneyline_challenger_artifacts.py",
        ],
        "training_example_count": len(dataset.examples),
        "training_runtime": dict(fitted_state.runtime),
        "scaler_fitted_state": {
            "means": list(fitted_projection["scaler_means"]),
            "scales": list(fitted_projection["scaler_stds"]),
        },
        "fixture_compatibility_note": (
            "P19A requires a bounded fixture expectation field; P22B does not assert "
            "predictive quality or use the placeholder for model selection."
        ),
    }
    projection["artifact_fingerprint"] = _sha256(
        _canonical_json_bytes(
            {key: value for key, value in projection.items() if key != "artifact_fingerprint"}
        )
    )
    artifact = MoneylineChallengerArtifact(projection=projection)
    artifact.to_inference_artifact()
    return artifact


def render_moneyline_challenger_summary(artifact: MoneylineChallengerArtifact) -> str:
    """Render the complete deterministic P22B summary."""

    projection = artifact.to_projection()
    summary = {
        "artifact_fingerprint": artifact.fingerprint(),
        "claims": dict(projection["claims"]),
        "coefficients": list(projection["coefficients"]),
        "feature_names": list(projection["feature_names"]),
        "fit_configuration": dict(projection["fit_configuration"]),
        "fitted_state_fingerprint": projection["fitted_state_fingerprint"],
        "intercept": projection["intercept"],
        "label_distribution": dict(projection["label_distribution"]),
        "label_semantics": projection["label_semantics"],
        "model_id": projection["model_id"],
        "model_role": projection["model_role"],
        "model_artifact": projection,
        "scaler_fitted_state": dict(projection["scaler_fitted_state"]),
        "schema_version": "p22b.moneyline_challenger_training_summary.v1",
        "source_dataset_fingerprint": projection["source_dataset_fingerprint"],
        "training_code_contract": projection["training_code_contract"],
        "training_example_count": projection["training_example_count"],
        "training_runtime": dict(projection["training_runtime"]),
        "model_promoted": False,
        "out_of_sample_evaluated": False,
        "production_ready": False,
        "profitability_claim": False,
        "real_betting_recommendation": False,
        "training_authorized": True,
        "training_performed": True,
    }
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_moneyline_challenger_artifacts(
    output_dir: str | Path,
    artifact: MoneylineChallengerArtifact,
) -> None:
    """Write exactly the authorized P22B artifact and summary files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_artifact.json").write_text(
        json.dumps(artifact.to_projection(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(
        render_moneyline_challenger_summary(artifact),
        encoding="utf-8",
    )


__all__ = (
    "MoneylineChallengerArtifact",
    "P22B_ARTIFACT_SCHEMA_VERSION",
    "P22B_MODEL_VERSION",
    "P22B_SOURCE_PATHS",
    "build_moneyline_challenger_artifact",
    "render_moneyline_challenger_summary",
    "write_moneyline_challenger_artifacts",
)
