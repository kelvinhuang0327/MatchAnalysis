"""Deterministic P36A offline Moneyline retraining artifact rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from ...baseball.domain.moneyline_feature_snapshot import MONEYLINE_FEATURE_NAMES
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
from .train_moneyline_challenger import FittedChallengerState


P36A_ARTIFACT_SCHEMA_VERSION = (
    "p36a.offline_moneyline_retraining_baseline_artifact.v1"
)
P36A_MODEL_VERSION = "p36a.moneyline_logistic_retraining_challenger.v1"
P36A_SUMMARY_SCHEMA_VERSION = "p36a.offline_moneyline_retraining_summary.v1"
P36A_SOURCE_PATHS = (
    "report/p22a_game_level_training_dataset/training_examples.jsonl",
    "report/p22a_game_level_training_dataset/summary.json",
    "report/p23f2_official_future_fold/feature_rows.jsonl",
    "report/p23f2_official_future_fold/results.jsonl",
    "data/fixtures/p23b_future_folds/wf_005/feature_rows.jsonl",
    "data/fixtures/p23b_future_folds/wf_005/results.jsonl",
    "data/fixtures/p23b_future_folds/wf_006/feature_rows.jsonl",
    "data/fixtures/p23b_future_folds/wf_006/results.jsonl",
    "src/match_analysis/application/use_cases/offline_moneyline_retraining_baseline.py",
    "src/match_analysis/application/use_cases/train_moneyline_challenger.py",
    "src/match_analysis/application/use_cases/offline_moneyline_retraining_artifacts.py",
)
P36A_FIT_CONFIGURATION = {
    "max_iter": 1000,
    "model_type": "logistic_regression",
    "solver": "lbfgs",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


def _fitted_state_projection(state: FittedChallengerState) -> dict[str, Any]:
    return {
        "coefficients": [str(_decimal(value)) for value in state.coefficients],
        "intercept": str(_decimal(state.intercept)),
        "scaler_means": [str(_decimal(value)) for value in state.scaler_means],
        "scaler_stds": [str(_decimal(value)) for value in state.scaler_stds],
    }


@dataclass(frozen=True, slots=True)
class OfflineMoneylineChallengerArtifact:
    """Immutable P36A metadata plus the existing Moneyline inference contract."""

    projection: Mapping[str, Any]

    def to_projection(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.projection, ensure_ascii=False))

    def canonical_bytes(self) -> bytes:
        projection = self.to_projection()
        projection.pop("artifact_fingerprint", None)
        return canonical_json_bytes(projection)

    def fingerprint(self) -> str:
        fingerprint = sha256_bytes(self.canonical_bytes())
        if self.projection.get("artifact_fingerprint") != fingerprint:
            raise ValueError("P36A artifact fingerprint does not match its projection")
        return fingerprint

    def to_inference_artifact(self) -> MoneylineModelArtifact:
        return MoneylineModelArtifact.from_projection(self.projection)


def build_offline_moneyline_challenger_artifact(
    *,
    fitted_state: FittedChallengerState,
    training_basis_fingerprint: str,
    training_observation_fingerprint: str,
    training_example_count: int,
    label_distribution: Mapping[str, int],
    training_date_range: Sequence[str],
    training_last_scheduled_start_utc: str,
    training_cutoff_date: str,
    holdout_start_date: str,
    p22a_dataset_fingerprint: str,
    p22a_dataset_sha256: str,
    training_fold_id: str,
    training_fold_fingerprint: str,
    training_feature_fingerprint: str,
    training_result_fingerprint: str,
    source_repository: str,
    source_commit: str,
    source_tree: str,
) -> OfflineMoneylineChallengerArtifact:
    """Build one P36A challenger while retaining P19A inference semantics."""

    fitted_projection = _fitted_state_projection(fitted_state)
    model_id = (
        "p36a_moneyline_logistic_retraining_challenger_v1_"
        f"{training_basis_fingerprint[:16]}"
    )
    base_artifact = MoneylineModelArtifact(
        model_id=model_id,
        model_version=P36A_MODEL_VERSION,
        feature_names=MONEYLINE_FEATURE_NAMES,
        coefficients=tuple(_decimal(value) for value in fitted_state.coefficients),
        intercept=_decimal(fitted_state.intercept),
        scaler_means=tuple(_decimal(value) for value in fitted_state.scaler_means),
        scaler_stds=tuple(_decimal(value) for value in fitted_state.scaler_stds),
        legacy_source_repository=source_repository,
        legacy_source_commit=source_commit,
        legacy_source_tree=source_tree,
        legacy_source_paths=P36A_SOURCE_PATHS,
        artifact_kind="bounded_deterministic_fixture",
        fixture_basis_id=training_basis_fingerprint,
        fixture_expected_home_probability=Decimal("0.5"),
        fixture_expected_probability_tolerance=Decimal("0.5"),
    )
    projection: dict[str, Any] = {
        **base_artifact.to_projection(),
        "artifact_schema_version": P36A_ARTIFACT_SCHEMA_VERSION,
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
        "fit_configuration": dict(P36A_FIT_CONFIGURATION),
        "fitted_state_fingerprint": sha256_bytes(canonical_json_bytes(fitted_projection)),
        "label_distribution": dict(sorted(label_distribution.items())),
        "label_semantics": (
            "target_home_win=1 iff committed FINAL home_score is greater than away_score; "
            "target_home_win=0 iff away_score is greater"
        ),
        "model_role": "CHALLENGER",
        "source_dataset_fingerprint": training_basis_fingerprint,
        "source_dataset_authority": {
            "p22a_dataset_fingerprint": p22a_dataset_fingerprint,
            "p22a_dataset_sha256": p22a_dataset_sha256,
            "training_fold_id": training_fold_id,
            "training_fold_fingerprint": training_fold_fingerprint,
            "training_feature_fingerprint": training_feature_fingerprint,
            "training_result_fingerprint": training_result_fingerprint,
            "training_observation_fingerprint": training_observation_fingerprint,
        },
        "training_code_contract": P36A_ARTIFACT_SCHEMA_VERSION,
        "training_code_paths": [
            "src/match_analysis/application/use_cases/offline_moneyline_retraining_baseline.py",
            "src/match_analysis/application/use_cases/train_moneyline_challenger.py",
            "src/match_analysis/application/use_cases/offline_moneyline_retraining_artifacts.py",
        ],
        "training_example_count": training_example_count,
        "training_date_range": list(training_date_range),
        "training_last_scheduled_start_utc": training_last_scheduled_start_utc,
        "training_cutoff_date": training_cutoff_date,
        "holdout_start_date": holdout_start_date,
        "training_runtime": dict(fitted_state.runtime),
        "scaler_fitted_state": {
            "means": list(fitted_projection["scaler_means"]),
            "scales": list(fitted_projection["scaler_stds"]),
        },
        "fixture_compatibility_note": (
            "P19A requires a bounded fixture expectation field; P36A uses the same "
            "inference artifact contract and does not use that placeholder for selection."
        ),
    }
    projection["artifact_fingerprint"] = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in projection.items()
                if key != "artifact_fingerprint"
            }
        )
    )
    artifact = OfflineMoneylineChallengerArtifact(projection=projection)
    artifact.to_inference_artifact()
    return artifact


def render_comparisons_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for row in rows
    )


def render_summary(summary: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            {"schema_version": P36A_SUMMARY_SCHEMA_VERSION, **dict(summary)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_report_markdown(summary: Mapping[str, Any]) -> str:
    training = summary["training"]
    holdout = summary["holdout"]
    champion = summary["champion"]
    challenger = summary["challenger"]
    comparison = summary["comparison"]
    lines = [
        "# P36A Offline Moneyline Retraining Baseline",
        "",
        "This is a deterministic offline champion/challenger comparison. No model was promoted.",
        "",
        "## Authority and split",
        "",
        f"- Training rows: `{training['eligible_row_count']}` eligible, `{training['excluded_row_count']}` excluded.",
        f"- Training range: `{training['date_range'][0]}` to `{training['date_range'][1]}`.",
        f"- Holdout rows: `{holdout['evaluable_row_count']}` evaluable of `{holdout['raw_row_count']}` raw.",
        f"- Holdout range: `{holdout['date_range'][0]}` to `{holdout['date_range'][1]}`.",
        f"- Holdout coverage: `{holdout['coverage']}`.",
        "",
        "## Model comparison",
        "",
        f"- Champion: `{champion['model_id']}`.",
        f"- Challenger: `{challenger['model_id']}`.",
        f"- Accuracy: champion `{champion['metrics']['accuracy']}`, challenger `{challenger['metrics']['accuracy']}`.",
        f"- Brier: champion `{champion['metrics']['brier_score']}`, challenger `{challenger['metrics']['brier_score']}`.",
        f"- Log loss: champion `{champion['metrics']['log_loss']}`, challenger `{challenger['metrics']['log_loss']}`.",
        f"- Verdict: `{comparison['verdict']}`.",
        "",
        "## Safety claims",
        "",
        "- Strict temporal split, point-in-time features, same holdout membership, and outcome isolation were verified.",
        "- This report makes no betting, profitability, production-readiness, or promotion claim.",
        "",
    ]
    return "\n".join(lines)


def write_offline_moneyline_retraining_artifacts(
    output_dir: str | Path,
    *,
    artifact: OfflineMoneylineChallengerArtifact,
    comparisons: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write only the bounded P36A artifact, comparison, summary, and report."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_artifact.json").write_text(
        json.dumps(artifact.to_projection(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (directory / "comparisons.jsonl").write_text(
        render_comparisons_jsonl(comparisons),
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(
        render_summary(summary),
        encoding="utf-8",
    )
    (directory / "report.md").write_text(
        render_report_markdown(summary),
        encoding="utf-8",
    )


__all__ = (
    "OfflineMoneylineChallengerArtifact",
    "P36A_ARTIFACT_SCHEMA_VERSION",
    "P36A_FIT_CONFIGURATION",
    "P36A_MODEL_VERSION",
    "P36A_SOURCE_PATHS",
    "P36A_SUMMARY_SCHEMA_VERSION",
    "build_offline_moneyline_challenger_artifact",
    "canonical_json_bytes",
    "render_comparisons_jsonl",
    "render_report_markdown",
    "render_summary",
    "sha256_bytes",
    "write_offline_moneyline_retraining_artifacts",
)
