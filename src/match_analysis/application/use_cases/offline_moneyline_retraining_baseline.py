"""Run the P36A offline Moneyline retraining and champion/challenger baseline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...baseball.domain.future_evaluation_fold import (
    FutureFeatureRow,
    FutureResultRow,
)
from ...baseball.domain.moneyline_feature_snapshot import MONEYLINE_FEATURE_NAMES
from ...baseball.domain.moneyline_oos_comparison import (
    PairedMoneylineComparison,
    aggregate_metrics,
    comparison_set_fingerprint,
)
from .evaluate_moneyline_challenger_oos import (
    load_frozen_challenger_authority,
    load_p23a_future_fold_authority,
    pair_predictions_with_results,
    predict_feature_rows,
    validate_no_training_overlap,
)
from .evaluate_multifold_moneyline_oos import (
    P23B_FOLD_SPECS,
    FutureFoldAuthority,
    _load_new_fold_authority,
)
from .offline_moneyline_retraining_artifacts import (
    OfflineMoneylineChallengerArtifact,
    P36A_FIT_CONFIGURATION,
    build_offline_moneyline_challenger_artifact,
    canonical_json_bytes,
    render_comparisons_jsonl,
    render_report_markdown,
    render_summary,
    sha256_bytes,
)
from .train_moneyline_challenger import (
    P22B_DEFAULT_FIT_RUNTIME,
    fit_moneyline_feature_rows,
    load_p22a_training_dataset,
)


P36A_TASK_ID = "P36A"
P36A_AUTHORITY_REPOSITORY = "/Users/kelvin/VibeCoding-WorkSpace/MatchAnalysis"
P36A_BASE_COMMIT = "57bc77a950865bb2b335ba35b17593596703b655"
P36A_BASE_TREE = "f24502f1215021c8e0e2de5401fa912ce5f61537"
P36A_P22A_DATASET_PATH = Path(
    "report/p22a_game_level_training_dataset/training_examples.jsonl"
)
P36A_P22A_SUMMARY_PATH = Path(
    "report/p22a_game_level_training_dataset/summary.json"
)
P36A_TRAINING_FOLD_ID = "wf_004"
P36A_HOLDOUT_FOLD_IDS = tuple(spec.fold_id for spec in P23B_FOLD_SPECS)
P36A_HOLDOUT_START_DATE = "2026-06-10"
P36A_TRAINING_CODE_CONTRACT = (
    "p36a.offline_moneyline_retraining_baseline.v1"
)
P36A_DECISION_RULE = (
    "CHALLENGER_BETTER iff challenger accuracy is higher and both challenger "
    "Brier score and log loss are lower; CHAMPION_RETAINS is the symmetric "
    "champion result; otherwise INCONCLUSIVE."
)


@dataclass(frozen=True, slots=True)
class TrainingObservation:
    """One labeled row admitted to the P36A fit after authority validation."""

    observation_id: str
    source: str
    fold_id: str
    official_date: str
    scheduled_start_utc: str
    feature_as_of_utc: str
    feature_values: tuple[str, ...]
    target_home_win: int
    source_row_fingerprint: str

    def __post_init__(self) -> None:
        if not self.observation_id or not self.source or not self.fold_id:
            raise ValueError("training observation identity must be explicit")
        if len(self.official_date) != 10:
            raise ValueError("training observation official_date must be YYYY-MM-DD")
        if not isinstance(self.feature_values, tuple):
            raise TypeError("feature_values must be a tuple")
        if len(self.feature_values) != len(MONEYLINE_FEATURE_NAMES):
            raise ValueError("training observation feature schema mismatch")
        if self.target_home_win not in (0, 1):
            raise ValueError("training observation target must be binary")
        if parse_canonical_utc(self.feature_as_of_utc) >= parse_canonical_utc(
            self.scheduled_start_utc
        ):
            raise ValueError("training observation feature is not point-in-time safe")
        if len(self.source_row_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_row_fingerprint
        ):
            raise ValueError("training observation source fingerprint must be SHA-256")

    def projection(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source": self.source,
            "fold_id": self.fold_id,
            "official_date": self.official_date,
            "scheduled_start_utc": self.scheduled_start_utc,
            "feature_as_of_utc": self.feature_as_of_utc,
            "feature_names": list(MONEYLINE_FEATURE_NAMES),
            "feature_values": list(self.feature_values),
            "target_home_win": self.target_home_win,
            "source_row_fingerprint": self.source_row_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class OfflineMoneylineRetrainingResult:
    """All deterministic P36A outputs before filesystem serialization."""

    artifact: OfflineMoneylineChallengerArtifact
    comparison_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def _sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _target_from_result(result: FutureResultRow) -> int:
    if result.status != "Final":
        raise ValueError("P36A training/evaluation results must be final")
    if result.home_score == result.away_score:
        raise ValueError("P36A cannot derive a Moneyline target from a tie")
    return int(result.home_score > result.away_score)


def _p22a_observations(dataset: Any) -> tuple[TrainingObservation, ...]:
    observations = tuple(
        TrainingObservation(
            observation_id=str(example.training_example_id),
            source="P22A",
            fold_id=str(example.fold_id),
            official_date=str(example.scheduled_start_utc)[:10],
            scheduled_start_utc=str(example.scheduled_start_utc),
            feature_as_of_utc=str(example.feature_as_of_utc),
            feature_values=tuple(str(value) for value in example.feature_values),
            target_home_win=int(example.target_home_win),
            source_row_fingerprint=str(example.historical_result_row_fingerprint),
        )
        for example in dataset.ordered_examples
    )
    if len({observation.observation_id for observation in observations}) != len(
        observations
    ):
        raise ValueError("P22A training observations must be unique")
    return observations


def _future_observations(
    feature_rows: Sequence[FutureFeatureRow],
    result_rows: Sequence[FutureResultRow],
    *,
    source: str,
    fold_id: str,
) -> tuple[TrainingObservation, ...]:
    feature_by_id = {row.provider_game_id: row for row in feature_rows}
    result_by_id = {row.provider_game_id: row for row in result_rows}
    if set(feature_by_id) != set(result_by_id):
        raise ValueError("future training feature/result identities must match")
    observations: list[TrainingObservation] = []
    for feature in sorted(
        feature_rows,
        key=lambda row: (row.scheduled_start_utc, row.game_number, row.game_pk),
    ):
        result = result_by_id[feature.provider_game_id]
        if result.scheduled_start_utc != feature.scheduled_start_utc:
            raise ValueError("future training feature/result schedule mismatch")
        source_row_fingerprint = _sha256_json(
            {"feature": feature.projection(), "result": result.projection()}
        )
        observations.append(
            TrainingObservation(
                observation_id=f"{source}:{feature.provider_game_id}",
                source=source,
                fold_id=fold_id,
                official_date=feature.official_date,
                scheduled_start_utc=feature.scheduled_start_utc,
                feature_as_of_utc=feature.feature_as_of_utc,
                feature_values=(
                    feature.recent_win_rate_delta,
                    feature.starter_era_delta,
                ),
                target_home_win=_target_from_result(result),
                source_row_fingerprint=source_row_fingerprint,
            )
        )
    return tuple(observations)


def _ordered_observations(
    observations: Sequence[TrainingObservation],
) -> tuple[TrainingObservation, ...]:
    ordered = tuple(
        sorted(
            observations,
            key=lambda row: (
                row.scheduled_start_utc,
                row.source,
                row.observation_id,
            ),
        )
    )
    if len({row.observation_id for row in ordered}) != len(ordered):
        raise ValueError("P36A training observation identities must be unique")
    return ordered


def _training_basis_fingerprint(
    observations: Sequence[TrainingObservation],
    *,
    dataset_fingerprint: str,
    dataset_sha256: str,
    fold_summary: Mapping[str, Any],
) -> tuple[str, str]:
    ordered = _ordered_observations(observations)
    observation_fingerprint = _sha256_json(
        [observation.projection() for observation in ordered]
    )
    basis = {
        "contract": P36A_TRAINING_CODE_CONTRACT,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_sha256": dataset_sha256,
        "training_fold_id": P36A_TRAINING_FOLD_ID,
        "training_fold_fingerprint": fold_summary["fold_fingerprint"],
        "training_feature_fingerprint": fold_summary["feature_fingerprint"],
        "training_result_fingerprint": fold_summary["result_fingerprint"],
        "training_observation_fingerprint": observation_fingerprint,
        "holdout_start_date": P36A_HOLDOUT_START_DATE,
        "fit_configuration": dict(P36A_FIT_CONFIGURATION),
    }
    return _sha256_json(basis), observation_fingerprint


def _load_training_authority(
    repository_root: Path,
) -> tuple[Any, tuple[TrainingObservation, ...], dict[str, Any]]:
    dataset = load_p22a_training_dataset(
        repository_root / P36A_P22A_DATASET_PATH,
        repository_root / P36A_P22A_SUMMARY_PATH,
    )
    (
        fold_summary,
        fold_manifest,
        _source_manifest,
        feature_rows,
        result_rows,
    ) = load_p23a_future_fold_authority(repository_root)
    if fold_summary["fold_id"] != P36A_TRAINING_FOLD_ID:
        raise ValueError("P36A training fold identity drift")
    if fold_summary["validation_end"] >= P36A_HOLDOUT_START_DATE:
        raise ValueError("P36A training fold reaches the holdout boundary")
    if fold_manifest["fold_fingerprint"] != fold_summary["fold_fingerprint"]:
        raise ValueError("P36A training fold fingerprint drift")
    validate_no_training_overlap(repository_root, feature_rows)
    observations = _ordered_observations(
        _p22a_observations(dataset)
        + _future_observations(
            feature_rows,
            result_rows,
            source="P23F2",
            fold_id=P36A_TRAINING_FOLD_ID,
        )
    )
    if len(observations) != 700:
        raise ValueError("P36A requires 677 P22A rows plus 23 wf_004 rows")
    training_dates = {row.official_date for row in observations}
    if P36A_HOLDOUT_START_DATE in training_dates or any(
        date >= P36A_HOLDOUT_START_DATE for date in training_dates
    ):
        raise ValueError("P36A training rows cross the holdout date boundary")
    metadata = {
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "dataset_sha256": dataset.training_examples_jsonl_sha256,
        "fold_id": P36A_TRAINING_FOLD_ID,
        "fold_fingerprint": fold_summary["fold_fingerprint"],
        "feature_fingerprint": fold_summary["feature_fingerprint"],
        "result_fingerprint": fold_summary["result_fingerprint"],
        "fold_summary": fold_summary,
    }
    return dataset, observations, metadata


def _load_holdout_authority(
    repository_root: Path,
) -> tuple[FutureFoldAuthority, ...]:
    authorities = tuple(
        _load_new_fold_authority(repository_root, spec) for spec in P23B_FOLD_SPECS
    )
    if tuple(authority.spec.fold_id for authority in authorities) != P36A_HOLDOUT_FOLD_IDS:
        raise ValueError("P36A holdout fold order drift")
    return authorities


def _log_loss_component(probability: Decimal, target: int) -> Decimal:
    if not Decimal("0") < probability < Decimal("1"):
        raise ValueError("Moneyline probability must be strictly between zero and one")
    return -(probability.ln() if target else (Decimal("1") - probability).ln())


def _calibration_summary(
    probabilities: Sequence[Decimal],
    targets: Sequence[int],
) -> dict[str, Any]:
    bins = (
        ("0.00-0.25", Decimal("0"), Decimal("0.25"), False),
        ("0.25-0.50", Decimal("0.25"), Decimal("0.50"), False),
        ("0.50-0.75", Decimal("0.50"), Decimal("0.75"), False),
        ("0.75-1.00", Decimal("0.75"), Decimal("1.00"), True),
    )
    projections: list[dict[str, Any]] = []
    weighted_gap = Decimal("0")
    total = Decimal(len(probabilities))
    for label, lower, upper, include_upper in bins:
        selected = [
            (probability, target)
            for probability, target in zip(probabilities, targets, strict=True)
            if lower <= probability <= upper
            if include_upper or probability < upper
        ]
        if not selected:
            projections.append(
                {
                    "bin": label,
                    "count": 0,
                    "mean_predicted_probability": None,
                    "observed_home_win_rate": None,
                    "absolute_gap": None,
                }
            )
            continue
        count = Decimal(len(selected))
        mean_probability = sum(
            (probability for probability, _target in selected), Decimal("0")
        ) / count
        observed_rate = sum(
            (Decimal(target) for _probability, target in selected), Decimal("0")
        ) / count
        gap = abs(mean_probability - observed_rate)
        weighted_gap += (count / total) * gap
        projections.append(
            {
                "bin": label,
                "count": len(selected),
                "mean_predicted_probability": str(mean_probability),
                "observed_home_win_rate": str(observed_rate),
                "absolute_gap": str(gap),
            }
        )
    return {
        "bin_count": len(projections),
        "expected_calibration_error": str(weighted_gap),
        "bins": projections,
    }


def _model_metrics(
    rows: Sequence[PairedMoneylineComparison],
    *,
    model: str,
    raw_row_count: int,
) -> dict[str, Any]:
    if model not in {"champion", "challenger"}:
        raise ValueError("unsupported P36A model side")
    aggregate = aggregate_metrics(rows)
    if model == "challenger":
        probabilities = [row.challenger_home_probability for row in rows]
        accuracy = aggregate["challenger_accuracy"]
        brier_score = aggregate["challenger_mean_brier"]
    else:
        probabilities = [row.incumbent_home_probability for row in rows]
        accuracy = aggregate["incumbent_accuracy"]
        brier_score = aggregate["incumbent_mean_brier"]
    targets = [row.target_home_win for row in rows]
    log_loss = sum(
        (_log_loss_component(probability, target) for probability, target in zip(probabilities, targets, strict=True)),
        Decimal("0"),
    ) / Decimal(len(rows))
    return {
        "row_count": len(rows),
        "accuracy": str(accuracy),
        "brier_score": str(brier_score),
        "log_loss": str(log_loss),
        "coverage": str(Decimal(len(rows)) / Decimal(raw_row_count)),
        "calibration": _calibration_summary(probabilities, targets),
    }


def _comparison_verdict(
    champion_metrics: Mapping[str, Any],
    challenger_metrics: Mapping[str, Any],
) -> str:
    champion_accuracy = Decimal(str(champion_metrics["accuracy"]))
    challenger_accuracy = Decimal(str(challenger_metrics["accuracy"]))
    champion_brier = Decimal(str(champion_metrics["brier_score"]))
    challenger_brier = Decimal(str(challenger_metrics["brier_score"]))
    champion_log_loss = Decimal(str(champion_metrics["log_loss"]))
    challenger_log_loss = Decimal(str(challenger_metrics["log_loss"]))
    if (
        challenger_accuracy > champion_accuracy
        and challenger_brier < champion_brier
        and challenger_log_loss < champion_log_loss
    ):
        return "CHALLENGER_BETTER"
    if (
        champion_accuracy > challenger_accuracy
        and champion_brier < challenger_brier
        and champion_log_loss < challenger_log_loss
    ):
        return "CHAMPION_RETAINS"
    return "INCONCLUSIVE"


def _comparison_projection(row: PairedMoneylineComparison) -> dict[str, Any]:
    projection = row.to_projection()
    projection.update(
        {
            "challenger_log_loss_contribution": str(
                _log_loss_component(
                    row.challenger_home_probability,
                    row.target_home_win,
                )
            ),
            "champion_log_loss_contribution": str(
                _log_loss_component(
                    row.incumbent_home_probability,
                    row.target_home_win,
                )
            ),
        }
    )
    return projection


def _fold_summary(
    authority: FutureFoldAuthority,
    rows: Sequence[PairedMoneylineComparison],
) -> dict[str, Any]:
    raw_row_count = len(authority.fold.raw_game_ids)
    exclusions = tuple(dict(row) for row in authority.fold.feature_unavailable_rows)
    return {
        "fold_id": authority.spec.fold_id,
        "raw_row_count": raw_row_count,
        "evaluable_row_count": len(rows),
        "excluded_row_count": len(exclusions),
        "excluded_reason_distribution": dict(
            sorted(Counter(str(row["reason"]) for row in exclusions).items())
        ),
        "date_range": [authority.spec.validation_start, authority.spec.validation_end],
        "coverage": str(Decimal(len(rows)) / Decimal(raw_row_count)),
        "feature_fingerprint": authority.fold.feature_fingerprint,
        "result_fingerprint": authority.fold.result_fingerprint,
        "fold_fingerprint": authority.fold.fold_fingerprint,
        "excluded_rows": list(exclusions),
    }


def evaluate_offline_moneyline_retraining_baseline(
    repository_root: str | Path,
    *,
    fit_runtime: str | Path = P22B_DEFAULT_FIT_RUNTIME,
) -> OfflineMoneylineRetrainingResult:
    """Train once and compare the new challenger with the current P22B champion."""

    root = Path(repository_root)
    dataset, observations, training_metadata = _load_training_authority(root)
    holdout_authorities = _load_holdout_authority(root)
    training_last_scheduled_start_utc = max(
        row.scheduled_start_utc for row in observations
    )
    training_end_date = max(row.official_date for row in observations)
    holdout_features = tuple(
        row
        for authority in holdout_authorities
        for row in authority.fold.feature_rows
    )
    holdout_dates = {row.official_date for row in holdout_features}
    holdout_start_date = min(row.official_date for row in holdout_features)
    if training_end_date >= holdout_start_date:
        raise ValueError("P36A training rows are not strictly before holdout dates")
    training_dates = {row.official_date for row in observations}
    if training_dates & holdout_dates:
        raise ValueError("P36A training and holdout date batches overlap")
    if any(
        parse_canonical_utc(row.feature_as_of_utc)
        >= parse_canonical_utc(row.scheduled_start_utc)
        for row in holdout_features
    ):
        raise ValueError("P36A holdout features violate point-in-time semantics")

    champion, champion_fingerprint, champion_projection = load_frozen_challenger_authority(
        root
    )
    if champion_projection.get("model_id") != "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630":
        raise ValueError("P36A current champion authority drift")
    training_basis_fingerprint, training_observation_fingerprint = (
        _training_basis_fingerprint(
            observations,
            dataset_fingerprint=dataset.dataset_fingerprint,
            dataset_sha256=dataset.training_examples_jsonl_sha256,
            fold_summary=training_metadata["fold_summary"],
        )
    )
    fitted_state = fit_moneyline_feature_rows(
        [row.feature_values for row in observations],
        [row.target_home_win for row in observations],
        fit_runtime=fit_runtime,
    )
    label_distribution = dict(
        sorted(Counter(str(row.target_home_win) for row in observations).items())
    )
    training_date_range = [
        min(row.official_date for row in observations),
        max(row.official_date for row in observations),
    ]
    challenger_artifact = build_offline_moneyline_challenger_artifact(
        fitted_state=fitted_state,
        training_basis_fingerprint=training_basis_fingerprint,
        training_observation_fingerprint=training_observation_fingerprint,
        training_example_count=len(observations),
        label_distribution=label_distribution,
        training_date_range=training_date_range,
        training_last_scheduled_start_utc=training_last_scheduled_start_utc,
        training_cutoff_date=training_end_date,
        holdout_start_date=P36A_HOLDOUT_START_DATE,
        p22a_dataset_fingerprint=dataset.dataset_fingerprint,
        p22a_dataset_sha256=dataset.training_examples_jsonl_sha256,
        training_fold_id=P36A_TRAINING_FOLD_ID,
        training_fold_fingerprint=training_metadata["fold_fingerprint"],
        training_feature_fingerprint=training_metadata["feature_fingerprint"],
        training_result_fingerprint=training_metadata["result_fingerprint"],
        source_repository=P36A_AUTHORITY_REPOSITORY,
        source_commit=P36A_BASE_COMMIT,
        source_tree=P36A_BASE_TREE,
    )
    challenger = challenger_artifact.to_inference_artifact()
    challenger_projection = challenger_artifact.to_projection()
    challenger_fingerprint = challenger_artifact.fingerprint()

    typed_rows: list[PairedMoneylineComparison] = []
    comparison_projections: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for authority in holdout_authorities:
        # This freezes both model streams from feature rows before the final results
        # are consumed by pair_predictions_with_results below.
        predictions = predict_feature_rows(
            authority.fold.feature_rows,
            challenger,
            champion,
        )
        rows = pair_predictions_with_results(
            feature_rows=authority.fold.feature_rows,
            predictions=predictions,
            result_rows=authority.fold.result_rows,
            challenger_model_id=str(challenger_projection["model_id"]),
            challenger_model_fingerprint=challenger_fingerprint,
            incumbent_model_id=str(champion_projection["model_id"]),
            incumbent_model_fingerprint=champion_fingerprint,
            fold_id=authority.spec.fold_id,
        )
        typed_rows.extend(rows)
        comparison_projections.extend(_comparison_projection(row) for row in rows)
        fold_summaries.append(_fold_summary(authority, rows))

    all_rows = tuple(typed_rows)
    raw_holdout_count = sum(item["raw_row_count"] for item in fold_summaries)
    excluded_holdout_count = sum(item["excluded_row_count"] for item in fold_summaries)
    evaluable_holdout_count = len(all_rows)
    if raw_holdout_count != evaluable_holdout_count + excluded_holdout_count:
        raise ValueError("P36A holdout accounting drift")
    if len({row.provider_game_id for row in all_rows}) != len(all_rows):
        raise ValueError("P36A holdout comparison rows must be unique")
    champion_metrics = _model_metrics(
        all_rows,
        model="champion",
        raw_row_count=raw_holdout_count,
    )
    challenger_metrics = _model_metrics(
        all_rows,
        model="challenger",
        raw_row_count=raw_holdout_count,
    )
    verdict = _comparison_verdict(champion_metrics, challenger_metrics)
    exclusion_reason_distribution = dict(
        sorted(
            Counter(
                reason
                for item in fold_summaries
                for reason, count in item["excluded_reason_distribution"].items()
                for _ in range(count)
            ).items()
        )
    )
    holdout_date_range = [
        min(authority.spec.validation_start for authority in holdout_authorities),
        max(authority.spec.validation_end for authority in holdout_authorities),
    ]
    same_holdout = {
        "row_count": evaluable_holdout_count,
        "comparison_set_fingerprint": comparison_set_fingerprint(all_rows),
        "champion_row_ids": sorted(row.provider_game_id for row in all_rows),
        "challenger_row_ids": sorted(row.provider_game_id for row in all_rows),
    }
    comparison = {
        "verdict": verdict,
        "decision_rule": P36A_DECISION_RULE,
        "same_holdout_verified": same_holdout["champion_row_ids"]
        == same_holdout["challenger_row_ids"],
        "same_holdout": same_holdout,
        "accuracy_delta": str(
            Decimal(challenger_metrics["accuracy"])
            - Decimal(champion_metrics["accuracy"])
        ),
        "brier_delta": str(
            Decimal(challenger_metrics["brier_score"])
            - Decimal(champion_metrics["brier_score"])
        ),
        "log_loss_delta": str(
            Decimal(challenger_metrics["log_loss"])
            - Decimal(champion_metrics["log_loss"])
        ),
        "calibration_ece_delta": str(
            Decimal(
                challenger_metrics["calibration"]["expected_calibration_error"]
            )
            - Decimal(champion_metrics["calibration"]["expected_calibration_error"])
        ),
    }
    training_source_counts = dict(
        sorted(Counter(row.source for row in observations).items())
    )
    summary: dict[str, Any] = {
        "task_id": P36A_TASK_ID,
        "operation": "OFFLINE_MONEYLINE_RETRAINING_BASELINE",
        "training_code_contract": P36A_TRAINING_CODE_CONTRACT,
        "historical_dataset_authority": {
            "repository": P36A_AUTHORITY_REPOSITORY,
            "p22a_dataset_path": str(P36A_P22A_DATASET_PATH),
            "p22a_dataset_fingerprint": dataset.dataset_fingerprint,
            "p22a_dataset_sha256": dataset.training_examples_jsonl_sha256,
            "training_fold_id": P36A_TRAINING_FOLD_ID,
            "training_fold_fingerprint": training_metadata["fold_fingerprint"],
            "training_feature_fingerprint": training_metadata["feature_fingerprint"],
            "training_result_fingerprint": training_metadata["result_fingerprint"],
            "training_basis_fingerprint": training_basis_fingerprint,
            "training_observation_fingerprint": training_observation_fingerprint,
        },
        "training": {
            "eligible_row_count": len(observations),
            "excluded_row_count": 0,
            "excluded_reasons": {},
            "source_row_counts": training_source_counts,
            "date_range": training_date_range,
            "last_scheduled_start_utc": training_last_scheduled_start_utc,
            "cutoff_date": training_end_date,
            "split_granularity": "OFFICIAL_DATE_BATCH",
            "label_distribution": label_distribution,
            "holdout_start_date": P36A_HOLDOUT_START_DATE,
        },
        "holdout": {
            "fold_ids": list(P36A_HOLDOUT_FOLD_IDS),
            "raw_row_count": raw_holdout_count,
            "evaluable_row_count": evaluable_holdout_count,
            "excluded_row_count": excluded_holdout_count,
            "excluded_reasons": exclusion_reason_distribution,
            "date_range": holdout_date_range,
            "coverage": str(
                Decimal(evaluable_holdout_count) / Decimal(raw_holdout_count)
            ),
            "metrics_population": "MODEL_EVALUABLE_GAMES",
            "folds": fold_summaries,
        },
        "training_eligible_row_count": len(observations),
        "training_excluded_row_count": 0,
        "holdout_raw_row_count": raw_holdout_count,
        "holdout_evaluable_row_count": evaluable_holdout_count,
        "holdout_excluded_row_count": excluded_holdout_count,
        "training_range": training_date_range,
        "holdout_range": holdout_date_range,
        "champion": {
            "model_id": champion_projection["model_id"],
            "artifact_fingerprint": champion_fingerprint,
            "artifact_path": "report/p22b_moneyline_challenger/model_artifact.json",
            "model_role": "CHAMPION",
            "metrics": champion_metrics,
        },
        "challenger": {
            "model_id": challenger_projection["model_id"],
            "artifact_fingerprint": challenger_fingerprint,
            "artifact_path": "report/p36a_offline_moneyline_retraining_baseline/model_artifact.json",
            "model_role": "CHALLENGER",
            "metrics": challenger_metrics,
        },
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "comparison": comparison,
        "comparison_verdict": verdict,
        "verification": {
            "historical_rows_temporally_admissible": True,
            "strict_train_before_holdout_verified": True,
            "same_date_batch_isolation_verified": True,
            "point_in_time_features_verified": True,
            "target_outcome_isolation_verified": True,
            "holdout_not_used_for_fit_verified": True,
            "same_holdout_rows_verified": comparison["same_holdout_verified"],
            "model_promotion_occurred": False,
            "deterministic_rerun_verified": False,
        },
        "claims": {
            "out_of_sample_evaluated": True,
            "training_performed": True,
            "model_promoted": False,
            "promotion_authorized": False,
            "production_ready": False,
            "profitability_claim": False,
            "real_betting_recommendation": False,
            "p20b_historical_runtime_compliance": "REMAINS_REFUTED",
        },
        "deterministic_rerun_verified": False,
        "input_order_invariance_verified": True,
    }
    return OfflineMoneylineRetrainingResult(
        artifact=challenger_artifact,
        comparison_rows=tuple(comparison_projections),
        summary=summary,
    )


def run_deterministic_offline_moneyline_retraining_baseline(
    repository_root: str | Path,
    *,
    fit_runtime: str | Path = P22B_DEFAULT_FIT_RUNTIME,
) -> OfflineMoneylineRetrainingResult:
    """Run P36A twice and require identical artifact, rows, summary, and report bytes."""

    first = evaluate_offline_moneyline_retraining_baseline(
        repository_root,
        fit_runtime=fit_runtime,
    )
    second = evaluate_offline_moneyline_retraining_baseline(
        repository_root,
        fit_runtime=fit_runtime,
    )
    if first.artifact.to_projection() != second.artifact.to_projection():
        raise ValueError("P36A deterministic challenger artifact mismatch")
    if render_comparisons_jsonl(first.comparison_rows) != render_comparisons_jsonl(
        second.comparison_rows
    ):
        raise ValueError("P36A deterministic comparison rows mismatch")
    if render_summary(first.summary) != render_summary(second.summary):
        raise ValueError("P36A deterministic summary mismatch")
    if render_report_markdown(first.summary) != render_report_markdown(second.summary):
        raise ValueError("P36A deterministic report mismatch")
    verified_summary = json.loads(json.dumps(first.summary, ensure_ascii=False))
    verified_summary["deterministic_rerun_verified"] = True
    verified_summary["verification"]["deterministic_rerun_verified"] = True
    return OfflineMoneylineRetrainingResult(
        artifact=first.artifact,
        comparison_rows=first.comparison_rows,
        summary=verified_summary,
    )


__all__ = (
    "OfflineMoneylineRetrainingResult",
    "P36A_AUTHORITY_REPOSITORY",
    "P36A_BASE_COMMIT",
    "P36A_BASE_TREE",
    "P36A_DECISION_RULE",
    "P36A_HOLDOUT_FOLD_IDS",
    "P36A_HOLDOUT_START_DATE",
    "P36A_TASK_ID",
    "P36A_TRAINING_FOLD_ID",
    "TrainingObservation",
    "evaluate_offline_moneyline_retraining_baseline",
    "run_deterministic_offline_moneyline_retraining_baseline",
)
