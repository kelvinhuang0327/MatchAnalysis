"""Run the P37A rolling walk-forward Moneyline OOS evaluation."""

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
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
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
)
from .evaluate_multifold_moneyline_oos import (
    P23B_FOLD_SPECS,
    _load_new_fold_authority,
)
from .rolling_moneyline_oos_artifacts import (
    render_comparisons_jsonl,
    render_model_artifacts,
    render_per_window_summary,
    render_report_markdown,
    render_summary,
    P37A_ARTIFACT_SCHEMA_VERSION,
)
from .train_moneyline_challenger import (
    P22B_DEFAULT_FIT_RUNTIME,
    P22B_FIT_CONFIGURATION,
    fit_moneyline_feature_rows,
    load_p22a_training_dataset,
)


P37A_TASK_ID = "P37A"
P37A_AUTHORITY_REPOSITORY = "/Users/kelvin/VibeCoding-WorkSpace/MatchAnalysis"
P37A_BASE_HEAD = "0ada15d8c198815d40fef6db60ba9885ccc421f0"
P37A_BASE_TREE = "d0c373d5027e14742c8562a7c2a6dcd127664731"
P37A_TRAINING_CODE_CONTRACT = "p37a.rolling_walk_forward_moneyline_oos.v1"
P37A_P22A_DATASET_PATH = Path(
    "report/p22a_game_level_training_dataset/training_examples.jsonl"
)
P37A_P22A_SUMMARY_PATH = Path(
    "report/p22a_game_level_training_dataset/summary.json"
)
P37A_P36A_SUMMARY_PATH = Path(
    "report/p36a_offline_moneyline_retraining_baseline/summary.json"
)
P37A_SEED_TRAINING_FOLD_IDS = ("wf_002", "wf_003")
P37A_EVALUATION_FOLD_IDS = ("wf_004", "wf_005", "wf_006")
P37A_FIT_CONFIGURATION = dict(P22B_FIT_CONFIGURATION)
P37A_DECISION_RULE = (
    "CHALLENGER_BETTER iff challenger accuracy is higher and both challenger "
    "Brier score and log loss are lower; CHAMPION_RETAINS is the symmetric "
    "champion result; otherwise INCONCLUSIVE."
)


@dataclass(frozen=True, slots=True)
class RollingTrainingObservation:
    """One prior, labeled row admitted to a rolling fit."""

    observation_id: str
    source: str
    fold_id: str
    provider_game_id: str
    official_date: str
    scheduled_start_utc: str
    feature_as_of_utc: str
    feature_values: tuple[str, ...]
    target_home_win: int
    source_row_fingerprint: str

    def __post_init__(self) -> None:
        if not self.observation_id or not self.source or not self.fold_id:
            raise ValueError("rolling observation identity must be explicit")
        if not self.provider_game_id:
            raise ValueError("rolling observation provider game identity must be explicit")
        if len(self.official_date) != 10:
            raise ValueError("rolling observation official_date must be YYYY-MM-DD")
        if len(self.feature_values) != len(MONEYLINE_FEATURE_NAMES):
            raise ValueError("rolling observation feature schema mismatch")
        if self.target_home_win not in (0, 1):
            raise ValueError("rolling observation target must be binary")
        if parse_canonical_utc(self.feature_as_of_utc) >= parse_canonical_utc(
            self.scheduled_start_utc
        ):
            raise ValueError("rolling observation feature is not point-in-time safe")
        if len(self.source_row_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_row_fingerprint
        ):
            raise ValueError("rolling observation source fingerprint must be SHA-256")

    def projection(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source": self.source,
            "fold_id": self.fold_id,
            "provider_game_id": self.provider_game_id,
            "official_date": self.official_date,
            "scheduled_start_utc": self.scheduled_start_utc,
            "feature_as_of_utc": self.feature_as_of_utc,
            "feature_names": list(MONEYLINE_FEATURE_NAMES),
            "feature_values": list(self.feature_values),
            "target_home_win": self.target_home_win,
            "source_row_fingerprint": self.source_row_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class RollingFoldAuthority:
    """The validated fold inputs used by one rolling evaluation."""

    fold_id: str
    validation_start: str
    validation_end: str
    feature_rows: tuple[FutureFeatureRow, ...]
    result_rows: tuple[FutureResultRow, ...]
    raw_game_ids: tuple[str, ...]
    feature_unavailable_rows: tuple[Mapping[str, Any], ...]
    source_manifest_fingerprint: str
    feature_fingerprint: str
    result_fingerprint: str
    fold_fingerprint: str

    @property
    def evaluable_game_ids(self) -> tuple[str, ...]:
        return tuple(row.provider_game_id for row in self.feature_rows)


@dataclass(frozen=True, slots=True)
class RollingWindow:
    """A fixed train-through / next-fold holdout window."""

    evaluation_window_id: str
    evaluation_window_order: int
    holdout: RollingFoldAuthority
    train_fold_ids: tuple[str, ...]
    training_observations: tuple[RollingTrainingObservation, ...]
    training_raw_game_ids: tuple[str, ...]
    training_excluded_rows: tuple[Mapping[str, Any], ...]
    training_authority_fingerprints: tuple[tuple[str, str], ...]
    p22a_dataset_fingerprint: str
    p22a_dataset_sha256: str


@dataclass(frozen=True, slots=True)
class RollingMoneylineOOSResult:
    """All deterministic P37A outputs before filesystem serialization."""

    model_artifacts: tuple[dict[str, Any], ...]
    comparison_rows: tuple[dict[str, Any], ...]
    per_window_summary: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path} contains a blank line at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _sha256_json(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _target_from_result(result: FutureResultRow) -> int:
    if result.status != "Final":
        raise ValueError("P37A training/evaluation results must be final")
    if result.home_score == result.away_score:
        raise ValueError("P37A cannot derive a Moneyline target from a tie")
    return int(result.home_score > result.away_score)


def _p22a_observations(dataset: Any) -> tuple[RollingTrainingObservation, ...]:
    observations = tuple(
        RollingTrainingObservation(
            observation_id=str(example.training_example_id),
            source="P22A",
            fold_id=str(example.fold_id),
            provider_game_id=str(example.provider_game_id),
            official_date=str(example.scheduled_start_utc)[:10],
            scheduled_start_utc=str(example.scheduled_start_utc),
            feature_as_of_utc=str(example.feature_as_of_utc),
            feature_values=tuple(str(value) for value in example.feature_values),
            target_home_win=int(example.target_home_win),
            source_row_fingerprint=str(example.historical_result_row_fingerprint),
        )
        for example in dataset.ordered_examples
    )
    return _ordered_observations(observations)


def _future_observations(
    authority: RollingFoldAuthority,
) -> tuple[RollingTrainingObservation, ...]:
    feature_by_id = {row.provider_game_id: row for row in authority.feature_rows}
    result_by_id = {row.provider_game_id: row for row in authority.result_rows}
    if not set(feature_by_id).issubset(result_by_id):
        raise ValueError(f"{authority.fold_id} feature/result identities do not match")
    observations: list[RollingTrainingObservation] = []
    for feature in sorted(
        authority.feature_rows,
        key=lambda row: (row.scheduled_start_utc, row.game_number, row.game_pk),
    ):
        result = result_by_id[feature.provider_game_id]
        if result.scheduled_start_utc != feature.scheduled_start_utc:
            raise ValueError(f"{authority.fold_id} feature/result schedule mismatch")
        observations.append(
            RollingTrainingObservation(
                observation_id=f"{authority.fold_id}:{feature.provider_game_id}",
                source=authority.fold_id,
                fold_id=authority.fold_id,
                provider_game_id=feature.provider_game_id,
                official_date=feature.official_date,
                scheduled_start_utc=feature.scheduled_start_utc,
                feature_as_of_utc=feature.feature_as_of_utc,
                feature_values=(
                    feature.recent_win_rate_delta,
                    feature.starter_era_delta,
                ),
                target_home_win=_target_from_result(result),
                source_row_fingerprint=_sha256_json(
                    {
                        "feature": feature.projection(),
                        "result": result.projection(),
                    }
                ),
            )
        )
    return _ordered_observations(observations)


def _ordered_observations(
    observations: Sequence[RollingTrainingObservation],
) -> tuple[RollingTrainingObservation, ...]:
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
        raise ValueError("P37A training observation identities must be unique")
    return ordered


def _training_observation_fingerprint(
    observations: Sequence[RollingTrainingObservation],
) -> str:
    return _sha256_json([row.projection() for row in _ordered_observations(observations)])


def _fold_authority_from_p23a(
    repository_root: Path,
) -> RollingFoldAuthority:
    summary, _manifest, _source_manifest, feature_rows, result_rows = (
        load_p23a_future_fold_authority(repository_root)
    )
    return RollingFoldAuthority(
        fold_id="wf_004",
        validation_start=str(summary["validation_start"]),
        validation_end=str(summary["validation_end"]),
        feature_rows=tuple(feature_rows),
        result_rows=tuple(result_rows),
        raw_game_ids=tuple(row.provider_game_id for row in feature_rows),
        feature_unavailable_rows=(),
        source_manifest_fingerprint=str(summary["source_manifest_fingerprint"]),
        feature_fingerprint=str(summary["feature_fingerprint"]),
        result_fingerprint=str(summary["result_fingerprint"]),
        fold_fingerprint=str(summary["fold_fingerprint"]),
    )


def _load_authoritative_future_folds(
    repository_root: Path,
) -> dict[str, RollingFoldAuthority]:
    authorities = {"wf_004": _fold_authority_from_p23a(repository_root)}
    for spec in P23B_FOLD_SPECS:
        loaded = _load_new_fold_authority(repository_root, spec)
        authorities[spec.fold_id] = RollingFoldAuthority(
            fold_id=spec.fold_id,
            validation_start=spec.validation_start,
            validation_end=spec.validation_end,
            feature_rows=tuple(loaded.fold.feature_rows),
            result_rows=tuple(loaded.fold.result_rows),
            raw_game_ids=tuple(loaded.fold.raw_game_ids),
            feature_unavailable_rows=tuple(loaded.fold.feature_unavailable_rows),
            source_manifest_fingerprint=loaded.fold.source_manifest_fingerprint,
            feature_fingerprint=loaded.fold.feature_fingerprint,
            result_fingerprint=loaded.fold.result_fingerprint,
            fold_fingerprint=loaded.fold.fold_fingerprint,
        )
    return authorities


def _load_p22a_authority(
    repository_root: Path,
) -> tuple[Any, tuple[RollingTrainingObservation, ...], dict[str, Any]]:
    dataset = load_p22a_training_dataset(
        repository_root / P37A_P22A_DATASET_PATH,
        repository_root / P37A_P22A_SUMMARY_PATH,
    )
    summary = _read_json(repository_root / P37A_P22A_SUMMARY_PATH)
    if summary.get("training_example_count") != 677:
        raise ValueError("P37A requires the committed 677-row P22A dataset")
    if [row["fold_id"] for row in summary.get("source_fold_artifacts", [])] != list(
        P37A_SEED_TRAINING_FOLD_IDS
    ):
        raise ValueError("P37A P22A seed fold authority drift")
    observations = _p22a_observations(dataset)
    if len(observations) != 677:
        raise ValueError("P37A P22A observation count drift")
    return dataset, observations, summary


def _characterize_historical_fold_chronology(
    repository_root: Path,
    p22a_summary: Mapping[str, Any],
    future_authorities: Mapping[str, RollingFoldAuthority],
) -> tuple[dict[str, Any], ...]:
    legacy_paths = {
        "wf_001": repository_root / "data/fixtures/p20a_p13_walk_forward/fold_wf_001.json",
        "wf_002": repository_root / "data/fixtures/p21b_multifold_historical/fold_wf_002.json",
        "wf_003": repository_root / "data/fixtures/p21b_multifold_historical/fold_wf_003.json",
    }
    chronology: list[dict[str, Any]] = []
    for fold_id, path in legacy_paths.items():
        projection = _read_json(path)
        training_rows = projection.get("training_rows", [])
        prediction_rows = projection.get("prediction_rows", [])
        if not isinstance(training_rows, list) or not isinstance(prediction_rows, list):
            raise ValueError(f"{fold_id} legacy fold chronology is incomplete")
        chronology.append(
            {
                "fold_id": fold_id,
                "authority_kind": "P20A_P21B_HISTORICAL_FOLD",
                "train_as_of": projection.get("train_as_of"),
                "validation_start": projection.get("validation_start"),
                "validation_end": projection.get("validation_end"),
                "source_feature_names": list(projection.get("feature_names", [])),
                "training_row_count": len(training_rows),
                "raw_holdout_row_count": len(prediction_rows),
                "p37a_train_capable": fold_id in P37A_SEED_TRAINING_FOLD_IDS,
                "p37a_holdout_capable": False,
                "p37a_admission": (
                    "SEED_TRAINING_AUTHORITY_FROM_P22A"
                    if fold_id in P37A_SEED_TRAINING_FOLD_IDS
                    else "NOT_ADMITTED_LEGACY_WF001"
                ),
            }
        )
    for fold_id in P37A_EVALUATION_FOLD_IDS:
        authority = future_authorities[fold_id]
        chronology.append(
            {
                "fold_id": fold_id,
                "authority_kind": "P23F2_CURRENT_FEATURE_RESULT_FOLD",
                "train_as_of": None,
                "validation_start": authority.validation_start,
                "validation_end": authority.validation_end,
                "source_feature_names": list(MONEYLINE_FEATURE_NAMES),
                "training_row_count": None,
                "raw_holdout_row_count": len(authority.raw_game_ids),
                "evaluable_holdout_row_count": len(authority.feature_rows),
                "feature_unavailable_count": len(authority.feature_unavailable_rows),
                "p37a_train_capable": True,
                "p37a_holdout_capable": True,
                "p37a_admission": "CURRENT_FEATURE_RESULT_AUTHORITY",
                "feature_fingerprint": authority.feature_fingerprint,
                "result_fingerprint": authority.result_fingerprint,
                "fold_fingerprint": authority.fold_fingerprint,
            }
        )
    chronology.sort(key=lambda row: (str(row["validation_start"]), row["fold_id"]))
    if [row["fold_id"] for row in chronology] != [
        "wf_001",
        "wf_002",
        "wf_003",
        "wf_004",
        "wf_005",
        "wf_006",
    ]:
        raise ValueError("P37A historical chronology is not the repository sequence")
    if p22a_summary.get("feature_names") != list(MONEYLINE_FEATURE_NAMES):
        raise ValueError("P37A P22A feature schema drift")
    return tuple(chronology)


def _validate_authority(
    p22a_observations: Sequence[RollingTrainingObservation],
    future_authorities: Mapping[str, RollingFoldAuthority],
) -> None:
    p22a_ids = {row.observation_id for row in p22a_observations}
    p22a_game_ids = {row.provider_game_id for row in p22a_observations}
    if len(p22a_ids) != len(p22a_observations) or len(p22a_game_ids) != len(
        p22a_observations
    ):
        raise ValueError("P37A P22A observations are not unique")
    previous_end: str | None = None
    prior_raw_ids: set[str] = set()
    for fold_id in P37A_EVALUATION_FOLD_IDS:
        authority = future_authorities[fold_id]
        if previous_end is not None and authority.validation_start <= previous_end:
            raise ValueError("P37A future fold chronology overlaps")
        previous_end = authority.validation_end
        feature_ids = {row.provider_game_id for row in authority.feature_rows}
        result_ids = {row.provider_game_id for row in authority.result_rows}
        raw_ids = set(authority.raw_game_ids)
        unavailable_ids = {
            str(row["game_id"]) for row in authority.feature_unavailable_rows
        }
        if (
            len(feature_ids) != len(authority.feature_rows)
            or len(result_ids) != len(authority.result_rows)
            or len(raw_ids) != len(authority.raw_game_ids)
            or len(unavailable_ids) != len(authority.feature_unavailable_rows)
        ):
            raise ValueError(f"{fold_id} authority game identities are not unique")
        if feature_ids & unavailable_ids:
            raise ValueError(f"{fold_id} feature/unavailable membership overlaps")
        if feature_ids | unavailable_ids != raw_ids:
            raise ValueError(f"{fold_id} raw/evaluable membership is incomplete")
        if result_ids != raw_ids:
            raise ValueError(f"{fold_id} result/raw membership is incomplete")
        if prior_raw_ids & raw_ids:
            raise ValueError("P37A future folds reuse a raw game identity")
        prior_raw_ids.update(raw_ids)
        for feature in authority.feature_rows:
            if not (
                authority.validation_start
                <= feature.official_date
                <= authority.validation_end
            ):
                raise ValueError(f"{fold_id} feature falls outside its validation range")
            if parse_canonical_utc(feature.feature_as_of_utc) >= parse_canonical_utc(
                feature.scheduled_start_utc
            ):
                raise ValueError(f"{fold_id} feature is not point-in-time safe")
            feature_projection = feature.projection()
            if {"home_score", "away_score", "target_home_win"} & feature_projection.keys():
                raise ValueError(f"{fold_id} feature payload contains an outcome")
    future_ids = {
        game_id
        for authority in future_authorities.values()
        for game_id in authority.raw_game_ids
    }
    if p22a_game_ids & future_ids:
        raise ValueError("P37A P22A and future authorities overlap")


def _build_windows(
    *,
    p22a_dataset: Any,
    p22a_observations: tuple[RollingTrainingObservation, ...],
    p22a_summary: Mapping[str, Any],
    future_authorities: Mapping[str, RollingFoldAuthority],
) -> tuple[RollingWindow, ...]:
    windows: list[RollingWindow] = []
    prior_observations = list(p22a_observations)
    prior_raw_game_ids = [row.provider_game_id for row in p22a_observations]
    prior_fold_ids = list(P37A_SEED_TRAINING_FOLD_IDS)
    prior_authority_fingerprints = [
        (
            fold_id,
            str(
                next(
                    item["fold_fingerprint"]
                    for item in p22a_summary["source_fold_artifacts"]
                    if item["fold_id"] == fold_id
                )
            ),
        )
        for fold_id in P37A_SEED_TRAINING_FOLD_IDS
    ]
    training_excluded_rows: list[Mapping[str, Any]] = []
    for order, holdout_fold_id in enumerate(P37A_EVALUATION_FOLD_IDS, 1):
        holdout = future_authorities[holdout_fold_id]
        training_observations = _ordered_observations(prior_observations)
        training_game_ids = set(prior_raw_game_ids)
        holdout_raw_ids = set(holdout.raw_game_ids)
        overlap = training_game_ids & holdout_raw_ids
        if overlap:
            raise ValueError(
                f"P37A train/holdout overlap in {holdout_fold_id}: {sorted(overlap)}"
            )
        training_dates = {row.official_date for row in training_observations}
        holdout_dates = {row.official_date for row in holdout.feature_rows}
        if not training_dates or not holdout_dates or max(training_dates) >= min(holdout_dates):
            raise ValueError(f"P37A training dates are not strictly before {holdout_fold_id}")
        excluded_for_window = tuple(training_excluded_rows)
        windows.append(
            RollingWindow(
                evaluation_window_id=f"window_{order:03d}_holdout_{holdout_fold_id}",
                evaluation_window_order=order,
                holdout=holdout,
                train_fold_ids=tuple(prior_fold_ids),
                training_observations=training_observations,
                training_raw_game_ids=tuple(sorted(training_game_ids)),
                training_excluded_rows=excluded_for_window,
                training_authority_fingerprints=tuple(prior_authority_fingerprints),
                p22a_dataset_fingerprint=str(p22a_dataset.dataset_fingerprint),
                p22a_dataset_sha256=str(p22a_dataset.training_examples_jsonl_sha256),
            )
        )
        fold_observations = _future_observations(holdout)
        prior_observations.extend(fold_observations)
        prior_raw_game_ids.extend(holdout.raw_game_ids)
        prior_fold_ids.append(holdout_fold_id)
        prior_authority_fingerprints.append((holdout_fold_id, holdout.fold_fingerprint))
        training_excluded_rows.extend(holdout.feature_unavailable_rows)
    return tuple(windows)


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
        raise ValueError("unsupported P37A model side")
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
        (
            _log_loss_component(probability, target)
            for probability, target in zip(probabilities, targets, strict=True)
        ),
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


def _comparison_projection(
    row: PairedMoneylineComparison,
    *,
    window: RollingWindow,
    training_basis_fingerprint: str,
    challenger_training_example_count: int,
    challenger_training_cutoff_date: str,
    challenger_model_id: str,
) -> dict[str, Any]:
    projection = row.to_projection()
    projection.update(
        {
            "evaluation_window_id": window.evaluation_window_id,
            "evaluation_window_order": window.evaluation_window_order,
            "holdout_fold_id": window.holdout.fold_id,
            "training_fold_ids": list(window.train_fold_ids),
            "training_basis_fingerprint": training_basis_fingerprint,
            "challenger_training_example_count": challenger_training_example_count,
            "challenger_training_cutoff_date": challenger_training_cutoff_date,
            "challenger_model_id": challenger_model_id,
            "true_oos_verified": True,
            "challenger_log_loss_contribution": str(
                _log_loss_component(row.challenger_home_probability, row.target_home_win)
            ),
            "champion_log_loss_contribution": str(
                _log_loss_component(row.incumbent_home_probability, row.target_home_win)
            ),
        }
    )
    return projection


def _window_model_projection(
    *,
    window: RollingWindow,
    artifact_projection: Mapping[str, Any],
    training_basis_fingerprint: str,
    training_observation_fingerprint: str,
) -> dict[str, Any]:
    return {
        "evaluation_window_id": window.evaluation_window_id,
        "evaluation_window_order": window.evaluation_window_order,
        "holdout_fold_id": window.holdout.fold_id,
        "training_basis_fingerprint": training_basis_fingerprint,
        "training_observation_fingerprint": training_observation_fingerprint,
        "model_id": artifact_projection["model_id"],
        "artifact_fingerprint": artifact_projection["artifact_fingerprint"],
        "training_example_count": artifact_projection["training_example_count"],
        "training_cutoff_date": artifact_projection["training_cutoff_date"],
        "training_fold_ids": list(window.train_fold_ids),
        "model_role": "CHALLENGER",
    }


def _build_rolling_challenger_artifact(
    *,
    fitted_state: Any,
    window: RollingWindow,
    training_basis_fingerprint: str,
    training_observation_fingerprint: str,
    label_distribution: Mapping[str, int],
    training_date_range: Sequence[str],
    training_last_scheduled_start_utc: str,
    training_cutoff_date: str,
) -> dict[str, Any]:
    fitted_projection = {
        "coefficients": [str(Decimal(repr(float(value)))) for value in fitted_state.coefficients],
        "intercept": str(Decimal(repr(float(fitted_state.intercept)))),
        "scaler_means": [str(Decimal(repr(float(value)))) for value in fitted_state.scaler_means],
        "scaler_stds": [str(Decimal(repr(float(value)))) for value in fitted_state.scaler_stds],
    }
    model_id = (
        f"p37a_moneyline_logistic_rolling_challenger_v1_"
        f"{window.holdout.fold_id}_{training_basis_fingerprint[:16]}"
    )
    base_artifact = MoneylineModelArtifact(
        model_id=model_id,
        model_version="p37a.moneyline_logistic_rolling_challenger.v1",
        feature_names=MONEYLINE_FEATURE_NAMES,
        coefficients=tuple(Decimal(repr(float(value))) for value in fitted_state.coefficients),
        intercept=Decimal(repr(float(fitted_state.intercept))),
        scaler_means=tuple(Decimal(repr(float(value))) for value in fitted_state.scaler_means),
        scaler_stds=tuple(Decimal(repr(float(value))) for value in fitted_state.scaler_stds),
        legacy_source_repository=P37A_AUTHORITY_REPOSITORY,
        legacy_source_commit=P37A_BASE_HEAD,
        legacy_source_tree=P37A_BASE_TREE,
        legacy_source_paths=(
            str(P37A_P22A_DATASET_PATH),
            str(P37A_P22A_SUMMARY_PATH),
            "report/p23f2_official_future_fold/feature_rows.jsonl",
            "report/p23f2_official_future_fold/results.jsonl",
            "data/fixtures/p23b_future_folds/wf_005/feature_rows.jsonl",
            "data/fixtures/p23b_future_folds/wf_005/results.jsonl",
            "data/fixtures/p23b_future_folds/wf_006/feature_rows.jsonl",
            "data/fixtures/p23b_future_folds/wf_006/results.jsonl",
            "src/match_analysis/application/use_cases/rolling_moneyline_oos.py",
            "src/match_analysis/application/use_cases/rolling_moneyline_oos_artifacts.py",
            "src/match_analysis/application/use_cases/train_moneyline_challenger.py",
        ),
        artifact_kind="bounded_deterministic_fixture",
        fixture_basis_id=training_basis_fingerprint,
        fixture_expected_home_probability=Decimal("0.5"),
        fixture_expected_probability_tolerance=Decimal("0.5"),
    )
    projection: dict[str, Any] = {
        **base_artifact.to_projection(),
        "artifact_schema_version": P37A_ARTIFACT_SCHEMA_VERSION,
        "artifact_fingerprint": "",
        "artifact_role": "CHALLENGER",
        "claims": {
            "model_promoted": False,
            "promotion_authorized": False,
            "out_of_sample_evaluated": True,
            "production_ready": False,
            "profitability_claim": False,
            "real_betting_recommendation": False,
            "training_authorized": True,
            "training_performed": True,
        },
        "feature_names": list(MONEYLINE_FEATURE_NAMES),
        "fit_configuration": dict(P37A_FIT_CONFIGURATION),
        "fitted_state_fingerprint": _sha256_json(fitted_projection),
        "label_distribution": dict(sorted(label_distribution.items())),
        "label_semantics": (
            "target_home_win=1 iff committed FINAL home_score is greater than away_score; "
            "target_home_win=0 iff away_score is greater"
        ),
        "model_role": "CHALLENGER",
        "source_dataset_fingerprint": training_basis_fingerprint,
        "source_dataset_authority": {
            "p22a_dataset_fingerprint": window.p22a_dataset_fingerprint,
            "p22a_dataset_sha256": window.p22a_dataset_sha256,
            "training_fold_ids": list(window.train_fold_ids),
            "training_fold_authority_fingerprints": {
                fold_id: fingerprint
                for fold_id, fingerprint in window.training_authority_fingerprints
            },
            "training_observation_fingerprint": training_observation_fingerprint,
        },
        "training_code_contract": P37A_TRAINING_CODE_CONTRACT,
        "training_code_paths": [
            "src/match_analysis/application/use_cases/rolling_moneyline_oos.py",
            "src/match_analysis/application/use_cases/rolling_moneyline_oos_artifacts.py",
            "src/match_analysis/application/use_cases/train_moneyline_challenger.py",
        ],
        "training_example_count": len(window.training_observations),
        "training_raw_row_count": len(window.training_raw_game_ids),
        "training_excluded_row_count": len(window.training_excluded_rows),
        "training_date_range": list(training_date_range),
        "training_last_scheduled_start_utc": training_last_scheduled_start_utc,
        "training_cutoff_date": training_cutoff_date,
        "holdout_fold_id": window.holdout.fold_id,
        "holdout_date_range": [
            min(row.official_date for row in window.holdout.feature_rows),
            max(row.official_date for row in window.holdout.feature_rows),
        ],
        "training_runtime": dict(fitted_state.runtime),
        "scaler_fitted_state": {
            "means": list(fitted_projection["scaler_means"]),
            "scales": list(fitted_projection["scaler_stds"]),
        },
        "fixture_compatibility_note": (
            "P37A uses the existing Moneyline inference artifact contract for a "
            "bounded rolling evaluation and does not use the fixture placeholder "
            "for model selection."
        ),
    }
    projection["artifact_fingerprint"] = _sha256_json(
        {
            key: value
            for key, value in projection.items()
            if key != "artifact_fingerprint"
        }
    )
    MoneylineModelArtifact.from_projection(projection)
    return projection


def _training_basis(
    window: RollingWindow,
) -> tuple[str, str]:
    observation_fingerprint = _training_observation_fingerprint(
        window.training_observations
    )
    basis = {
        "contract": P37A_TRAINING_CODE_CONTRACT,
        "base_head": P37A_BASE_HEAD,
        "base_tree": P37A_BASE_TREE,
        "evaluation_window_id": window.evaluation_window_id,
        "holdout_fold_id": window.holdout.fold_id,
        "training_fold_ids": list(window.train_fold_ids),
        "training_fold_authority_fingerprints": {
            fold_id: fingerprint
            for fold_id, fingerprint in window.training_authority_fingerprints
        },
        "p22a_dataset_fingerprint": window.p22a_dataset_fingerprint,
        "p22a_dataset_sha256": window.p22a_dataset_sha256,
        "training_observation_fingerprint": observation_fingerprint,
        "fit_configuration": dict(P37A_FIT_CONFIGURATION),
    }
    return _sha256_json(basis), observation_fingerprint


def _window_summary(
    *,
    window: RollingWindow,
    rows: Sequence[PairedMoneylineComparison],
    champion_projection: Mapping[str, Any],
    champion_fingerprint: str,
    challenger_projection: Mapping[str, Any],
    training_basis_fingerprint: str,
    training_observation_fingerprint: str,
) -> dict[str, Any]:
    training_dates = [row.official_date for row in window.training_observations]
    holdout_dates = [row.official_date for row in window.holdout.feature_rows]
    training_game_ids = set(window.training_raw_game_ids)
    holdout_game_ids = set(window.holdout.raw_game_ids)
    overlap = sorted(training_game_ids & holdout_game_ids)
    raw_row_count = len(window.holdout.raw_game_ids)
    excluded_rows = tuple(dict(row) for row in window.holdout.feature_unavailable_rows)
    champion_metrics = _model_metrics(rows, model="champion", raw_row_count=raw_row_count)
    challenger_metrics = _model_metrics(rows, model="challenger", raw_row_count=raw_row_count)
    same_holdout_ids = sorted(row.provider_game_id for row in rows)
    verdict = _comparison_verdict(champion_metrics, challenger_metrics)
    return {
        "evaluation_window_id": window.evaluation_window_id,
        "evaluation_window_order": window.evaluation_window_order,
        "train_fold_ids": list(window.train_fold_ids),
        "holdout_fold_id": window.holdout.fold_id,
        "training": {
            "eligible_row_count": len(window.training_observations),
            "raw_row_count": len(window.training_raw_game_ids),
            "excluded_row_count": len(window.training_excluded_rows),
            "excluded_reason_distribution": dict(
                sorted(
                    Counter(
                        str(row["reason"]) for row in window.training_excluded_rows
                    ).items()
                )
            ),
            "date_range": [min(training_dates), max(training_dates)],
            "last_scheduled_start_utc": max(
                row.scheduled_start_utc for row in window.training_observations
            ),
            "cutoff_date": max(training_dates),
            "label_distribution": dict(
                sorted(
                    Counter(str(row.target_home_win) for row in window.training_observations).items()
                )
            ),
            "fold_ids": list(window.train_fold_ids),
            "game_id_count": len(training_game_ids),
            "game_id_fingerprint": _sha256_json(sorted(training_game_ids)),
            "basis_fingerprint": training_basis_fingerprint,
            "observation_fingerprint": training_observation_fingerprint,
        },
        "holdout": {
            "fold_id": window.holdout.fold_id,
            "date_range": [min(holdout_dates), max(holdout_dates)],
            "raw_row_count": raw_row_count,
            "evaluable_row_count": len(rows),
            "excluded_row_count": len(excluded_rows),
            "excluded_reason_distribution": dict(
                sorted(
                    Counter(str(row["reason"]) for row in excluded_rows).items()
                )
            ),
            "coverage": str(Decimal(len(rows)) / Decimal(raw_row_count)),
            "raw_game_ids": sorted(window.holdout.raw_game_ids),
            "evaluable_game_ids": same_holdout_ids,
            "excluded_game_ids": sorted(
                str(row["game_id"]) for row in excluded_rows
            ),
            "feature_fingerprint": window.holdout.feature_fingerprint,
            "result_fingerprint": window.holdout.result_fingerprint,
            "fold_fingerprint": window.holdout.fold_fingerprint,
            "source_manifest_fingerprint": window.holdout.source_manifest_fingerprint,
            "excluded_rows": list(excluded_rows),
        },
        "champion": {
            "model_id": champion_projection["model_id"],
            "artifact_fingerprint": champion_fingerprint,
            "model_role": "CHAMPION",
            "metrics": champion_metrics,
        },
        "challenger": {
            "model_id": challenger_projection["model_id"],
            "artifact_fingerprint": challenger_projection["artifact_fingerprint"],
            "model_role": "CHALLENGER",
            "metrics": challenger_metrics,
        },
        "comparison": {
            "verdict": verdict,
            "decision_rule": P37A_DECISION_RULE,
            "same_holdout_verified": True,
            "same_holdout": {
                "row_count": len(rows),
                "champion_row_ids": same_holdout_ids,
                "challenger_row_ids": same_holdout_ids,
            },
            "train_holdout_game_id_overlap": overlap,
            "train_holdout_disjoint_verified": not overlap,
            "strict_train_before_holdout_verified": max(training_dates)
            < min(holdout_dates),
            "point_in_time_features_verified": all(
                parse_canonical_utc(row.feature_as_of_utc)
                < parse_canonical_utc(row.scheduled_start_utc)
                for row in window.holdout.feature_rows
            ),
            "outcome_isolation_verified": True,
            "no_aggregate_oos_tuning": True,
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
                    challenger_metrics["calibration"][
                        "expected_calibration_error"
                    ]
                )
                - Decimal(
                    champion_metrics["calibration"][
                        "expected_calibration_error"
                    ]
                )
            ),
        },
    }


def _p36a_baseline_preserved(repository_root: Path) -> dict[str, Any]:
    summary = _read_json(repository_root / P37A_P36A_SUMMARY_PATH)
    expected = {
        "task_id": "P36A",
        "training_fold_id": "wf_004",
        "holdout_fold_ids": ["wf_005", "wf_006"],
        "holdout_evaluable_row_count": 42,
        "champion_accuracy": "0.5952380952380952380952380952",
        "champion_brier": "0.2412503038891353958101706395",
        "champion_log_loss": "0.6754482605925976333743156102",
        "champion_ece": "0.04382191993924131051589468869",
        "challenger_accuracy": "0.6190476190476190476190476190",
        "challenger_brier": "0.2409542612751445629346755433",
        "challenger_log_loss": "0.6748400665006979300167290164",
        "challenger_ece": "0.05373421488154330306429538093",
    }
    observed = {
        "task_id": summary.get("task_id"),
        "training_fold_id": summary.get("historical_dataset_authority", {}).get(
            "training_fold_id"
        ),
        "holdout_fold_ids": summary.get("holdout", {}).get("fold_ids"),
        "holdout_evaluable_row_count": summary.get("holdout", {}).get(
            "evaluable_row_count"
        ),
        "champion_accuracy": summary["champion"]["metrics"]["accuracy"],
        "champion_brier": summary["champion"]["metrics"]["brier_score"],
        "champion_log_loss": summary["champion"]["metrics"]["log_loss"],
        "champion_ece": summary["champion"]["metrics"]["calibration"][
            "expected_calibration_error"
        ],
        "challenger_accuracy": summary["challenger"]["metrics"]["accuracy"],
        "challenger_brier": summary["challenger"]["metrics"]["brier_score"],
        "challenger_log_loss": summary["challenger"]["metrics"]["log_loss"],
        "challenger_ece": summary["challenger"]["metrics"]["calibration"][
            "expected_calibration_error"
        ],
    }
    if observed != expected:
        raise ValueError("P36A authoritative result changed")
    return {
        "summary_path": str(P37A_P36A_SUMMARY_PATH),
        "unchanged_verified": True,
        **observed,
    }


def _overall_conclusion(
    per_window: Sequence[Mapping[str, Any]],
    aggregate_verdict: str,
) -> tuple[str, dict[str, Any]]:
    challenger_verdict_wins = [
        row["holdout_fold_id"]
        for row in per_window
        if row["comparison"]["verdict"] == "CHALLENGER_BETTER"
    ]
    champion_verdict_wins = [
        row["holdout_fold_id"]
        for row in per_window
        if row["comparison"]["verdict"] == "CHAMPION_RETAINS"
    ]
    inconclusive_windows = [
        row["holdout_fold_id"]
        for row in per_window
        if row["comparison"]["verdict"] == "INCONCLUSIVE"
    ]
    challenger_brier_better = [
        row["holdout_fold_id"]
        for row in per_window
        if Decimal(row["comparison"]["brier_delta"]) < Decimal("0")
    ]
    champion_brier_better = [
        row["holdout_fold_id"]
        for row in per_window
        if Decimal(row["comparison"]["brier_delta"]) > Decimal("0")
    ]
    equal_brier = [
        row["holdout_fold_id"]
        for row in per_window
        if Decimal(row["comparison"]["brier_delta"]) == Decimal("0")
    ]
    repetition_threshold = (len(per_window) + 1) // 2
    if (
        aggregate_verdict == "CHALLENGER_BETTER"
        and len(challenger_verdict_wins) >= repetition_threshold
        and len(challenger_verdict_wins) > len(champion_verdict_wins)
    ):
        conclusion = "CHALLENGER_IMPROVEMENT_REPEATED"
    elif (
        aggregate_verdict == "CHAMPION_RETAINS"
        and len(champion_verdict_wins) >= repetition_threshold
        and len(champion_verdict_wins) > len(challenger_verdict_wins)
    ):
        conclusion = "CHAMPION_RETAINS"
    else:
        conclusion = "MIXED_OR_INCONCLUSIVE"
    return conclusion, {
        "repetition_threshold_window_count": repetition_threshold,
        "windows_with_challenger_better_verdict": challenger_verdict_wins,
        "windows_with_champion_retains_verdict": champion_verdict_wins,
        "windows_with_inconclusive_verdict": inconclusive_windows,
        "windows_with_challenger_lower_brier": challenger_brier_better,
        "windows_with_champion_lower_brier": champion_brier_better,
        "windows_equal_brier": equal_brier,
    }


def evaluate_rolling_moneyline_oos(
    repository_root: str | Path,
    *,
    fit_runtime: str | Path = P22B_DEFAULT_FIT_RUNTIME,
) -> RollingMoneylineOOSResult:
    """Evaluate the maximum valid current-authority rolling window set."""

    root = Path(repository_root)
    p22a_dataset, p22a_observations, p22a_summary = _load_p22a_authority(root)
    future_authorities = _load_authoritative_future_folds(root)
    _validate_authority(p22a_observations, future_authorities)
    historical_chronology = _characterize_historical_fold_chronology(
        root,
        p22a_summary,
        future_authorities,
    )
    windows = _build_windows(
        p22a_dataset=p22a_dataset,
        p22a_observations=p22a_observations,
        p22a_summary=p22a_summary,
        future_authorities=future_authorities,
    )
    if len(windows) < 2:
        raise ValueError("P37A_INSUFFICIENT_WALK_FORWARD_AUTHORITY_STOP")

    champion, champion_fingerprint, champion_projection = load_frozen_challenger_authority(
        root
    )
    if champion_projection.get("model_id") != "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630":
        raise ValueError("P37A current champion authority drift")

    model_artifacts: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    per_window_summary: list[dict[str, Any]] = []
    input_order_invariance_verified = True
    for window in windows:
        training_basis_fingerprint, training_observation_fingerprint = _training_basis(
            window
        )
        fitted_state = fit_moneyline_feature_rows(
            [row.feature_values for row in window.training_observations],
            [row.target_home_win for row in window.training_observations],
            fit_runtime=fit_runtime,
        )
        training_dates = [row.official_date for row in window.training_observations]
        label_distribution = dict(
            sorted(
                Counter(
                    str(row.target_home_win) for row in window.training_observations
                ).items()
            )
        )
        challenger_projection = _build_rolling_challenger_artifact(
            fitted_state=fitted_state,
            window=window,
            training_basis_fingerprint=training_basis_fingerprint,
            training_observation_fingerprint=training_observation_fingerprint,
            label_distribution=label_distribution,
            training_date_range=[min(training_dates), max(training_dates)],
            training_last_scheduled_start_utc=max(
                row.scheduled_start_utc for row in window.training_observations
            ),
            training_cutoff_date=max(training_dates),
        )
        challenger = MoneylineModelArtifact.from_projection(challenger_projection)
        model_artifacts.append(
            _window_model_projection(
                window=window,
                artifact_projection=challenger_projection,
                training_basis_fingerprint=training_basis_fingerprint,
                training_observation_fingerprint=training_observation_fingerprint,
            )
        )

        # Both probabilities are frozen from features before final outcomes are joined.
        predictions = predict_feature_rows(window.holdout.feature_rows, challenger, champion)
        reversed_predictions = predict_feature_rows(
            tuple(reversed(window.holdout.feature_rows)),
            challenger,
            champion,
        )
        if predictions != reversed_predictions:
            input_order_invariance_verified = False
        rows = pair_predictions_with_results(
            feature_rows=window.holdout.feature_rows,
            predictions=predictions,
            result_rows=window.holdout.result_rows,
            challenger_model_id=str(challenger_projection["model_id"]),
            challenger_model_fingerprint=str(challenger_projection["artifact_fingerprint"]),
            incumbent_model_id=str(champion_projection["model_id"]),
            incumbent_model_fingerprint=champion_fingerprint,
            fold_id=window.holdout.fold_id,
        )
        if len(rows) != len(window.holdout.feature_rows):
            raise ValueError(f"P37A {window.holdout.fold_id} dropped an OOS row")
        window_summary = _window_summary(
            window=window,
            rows=rows,
            champion_projection=champion_projection,
            champion_fingerprint=champion_fingerprint,
            challenger_projection=challenger_projection,
            training_basis_fingerprint=training_basis_fingerprint,
            training_observation_fingerprint=training_observation_fingerprint,
        )
        per_window_summary.append(window_summary)
        comparison_rows.extend(
            _comparison_projection(
                row,
                window=window,
                training_basis_fingerprint=training_basis_fingerprint,
                challenger_training_example_count=len(window.training_observations),
                challenger_training_cutoff_date=max(training_dates),
                challenger_model_id=str(challenger_projection["model_id"]),
            )
            for row in rows
        )

    per_window_summary.sort(key=lambda row: int(row["evaluation_window_order"]))
    comparison_rows.sort(
        key=lambda row: (
            int(row["evaluation_window_order"]),
            str(row["scheduled_start_utc"]),
            int(row["game_number"]),
            int(row["game_pk"]),
        )
    )
    typed_rows = tuple(
        PairedMoneylineComparison(
            comparison_row_id=str(row["comparison_row_id"]),
            fold_id=str(row["fold_id"]),
            provider_game_id=str(row["provider_game_id"]),
            game_pk=int(row["game_pk"]),
            game_number=int(row["game_number"]),
            scheduled_start_utc=str(row["scheduled_start_utc"]),
            feature_fingerprint=str(row["feature_fingerprint"]),
            challenger_model_id=str(row["challenger_model_id"]),
            challenger_model_fingerprint=str(row["challenger_model_fingerprint"]),
            challenger_home_probability=Decimal(str(row["challenger_home_probability"])),
            incumbent_model_id=str(row["incumbent_model_id"]),
            incumbent_model_fingerprint=str(row["incumbent_model_fingerprint"]),
            incumbent_home_probability=Decimal(str(row["incumbent_home_probability"])),
            target_home_win=int(row["target_home_win"]),
            actual_winner=str(row["actual_winner"]),
            challenger_correct=bool(row["challenger_correct"]),
            incumbent_correct=bool(row["incumbent_correct"]),
            challenger_brier_contribution=Decimal(
                str(row["challenger_brier_contribution"])
            ),
            incumbent_brier_contribution=Decimal(
                str(row["incumbent_brier_contribution"])
            ),
            paired_brier_delta=Decimal(str(row["paired_brier_delta"])),
        )
        for row in comparison_rows
    )
    raw_count = sum(int(row["holdout"]["raw_row_count"]) for row in per_window_summary)
    evaluable_count = len(typed_rows)
    excluded_count = sum(
        int(row["holdout"]["excluded_row_count"]) for row in per_window_summary
    )
    if raw_count != evaluable_count + excluded_count:
        raise ValueError("P37A raw/evaluable/excluded accounting drift")
    if len({row.provider_game_id for row in typed_rows}) != len(typed_rows):
        raise ValueError("P37A aggregate OOS rows are not unique")
    champion_metrics = _model_metrics(
        typed_rows,
        model="champion",
        raw_row_count=raw_count,
    )
    challenger_metrics = _model_metrics(
        typed_rows,
        model="challenger",
        raw_row_count=raw_count,
    )
    aggregate_verdict = _comparison_verdict(champion_metrics, challenger_metrics)
    conclusion, repetition = _overall_conclusion(per_window_summary, aggregate_verdict)
    p36a_preserved = _p36a_baseline_preserved(root)
    summary: dict[str, Any] = {
        "task_id": P37A_TASK_ID,
        "operation": "ROLLING_WALK_FORWARD_MONEYLINE_OOS_EVALUATION",
        "training_code_contract": P37A_TRAINING_CODE_CONTRACT,
        "authority": {
            "repository": P37A_AUTHORITY_REPOSITORY,
            "base_head": P37A_BASE_HEAD,
            "base_tree": P37A_BASE_TREE,
            "p22a_dataset_path": str(P37A_P22A_DATASET_PATH),
            "p22a_dataset_fingerprint": p22a_dataset.dataset_fingerprint,
            "p22a_dataset_sha256": p22a_dataset.training_examples_jsonl_sha256,
            "current_champion_model_id": champion_projection["model_id"],
            "current_champion_artifact_fingerprint": champion_fingerprint,
        },
        "historical_fold_chronology": list(historical_chronology),
        "admitted_evaluation_fold_ids": list(P37A_EVALUATION_FOLD_IDS),
        "evaluation_windows": per_window_summary,
        "aggregate": {
            "raw_row_count": raw_count,
            "evaluable_row_count": evaluable_count,
            "excluded_row_count": excluded_count,
            "coverage": str(Decimal(evaluable_count) / Decimal(raw_count)),
            "metrics_population": "MODEL_EVALUABLE_GAMES",
            "comparison_set_fingerprint": comparison_set_fingerprint(typed_rows),
            "champion": {
                "model_id": champion_projection["model_id"],
                "artifact_fingerprint": champion_fingerprint,
                "model_role": "CHAMPION",
                "metrics": champion_metrics,
            },
            "challenger": {
                "model_count": len(model_artifacts),
                "model_ids": [artifact["model_id"] for artifact in model_artifacts],
                "artifact_fingerprints": [
                    artifact["artifact_fingerprint"] for artifact in model_artifacts
                ],
                "model_role": "CHALLENGER",
                "metrics": challenger_metrics,
            },
        },
        "champion": {
            "model_id": champion_projection["model_id"],
            "artifact_fingerprint": champion_fingerprint,
            "artifact_path": "report/p22b_moneyline_challenger/model_artifact.json",
            "model_role": "CHAMPION",
            "frozen": True,
        },
        "challenger_models": model_artifacts,
        "comparison": {
            "aggregate_verdict": aggregate_verdict,
            "conclusion": conclusion,
            "decision_rule": P37A_DECISION_RULE,
            "aggregate_accuracy_delta": str(
                Decimal(challenger_metrics["accuracy"])
                - Decimal(champion_metrics["accuracy"])
            ),
            "aggregate_brier_delta": str(
                Decimal(challenger_metrics["brier_score"])
                - Decimal(champion_metrics["brier_score"])
            ),
            "aggregate_log_loss_delta": str(
                Decimal(challenger_metrics["log_loss"])
                - Decimal(champion_metrics["log_loss"])
            ),
            "aggregate_calibration_ece_delta": str(
                Decimal(
                    challenger_metrics["calibration"][
                        "expected_calibration_error"
                    ]
                )
                - Decimal(
                    champion_metrics["calibration"][
                        "expected_calibration_error"
                    ]
                )
            ),
            "per_window_verdicts": {
                row["holdout_fold_id"]: row["comparison"]["verdict"]
                for row in per_window_summary
            },
            "per_window_metric_deltas": {
                row["holdout_fold_id"]: {
                    "accuracy_delta": row["comparison"]["accuracy_delta"],
                    "brier_delta": row["comparison"]["brier_delta"],
                    "log_loss_delta": row["comparison"]["log_loss_delta"],
                    "calibration_ece_delta": row["comparison"][
                        "calibration_ece_delta"
                    ],
                }
                for row in per_window_summary
            },
            **repetition,
        },
        "verification": {
            "valid_window_count": len(windows),
            "chronological_fold_order_verified": True,
            "train_holdout_game_id_disjointness_verified": all(
                row["comparison"]["train_holdout_disjoint_verified"]
                for row in per_window_summary
            ),
            "strict_train_before_holdout_verified": all(
                row["comparison"]["strict_train_before_holdout_verified"]
                for row in per_window_summary
            ),
            "point_in_time_features_verified": all(
                row["comparison"]["point_in_time_features_verified"]
                for row in per_window_summary
            ),
            "same_holdout_rows_verified": all(
                row["comparison"]["same_holdout_verified"]
                for row in per_window_summary
            ),
            "outcome_isolation_verified": all(
                row["comparison"]["outcome_isolation_verified"]
                for row in per_window_summary
            ),
            "aggregate_true_oos_rows_verified": all(
                row["true_oos_verified"] for row in comparison_rows
            ),
            "metric_calculation_verified": True,
            "input_order_invariance_verified": input_order_invariance_verified,
            "deterministic_rerun_verified": False,
            "aggregate_oos_tuning_performed": False,
            "calibration_fitted_on_aggregate_oos": False,
            "model_promotion_occurred": False,
        },
        "claims": {
            "out_of_sample_evaluated": True,
            "multi_window_evaluated": True,
            "training_performed": True,
            "model_promoted": False,
            "promotion_authorized": False,
            "production_ready": False,
            "profitability_claim": False,
            "real_betting_recommendation": False,
            "bet_or_pass_claim": False,
            "p20b_historical_runtime_compliance": "REMAINS_REFUTED",
        },
        "p36a_baseline_preserved": p36a_preserved,
        "deterministic_rerun_verified": False,
        "input_order_invariance_verified": input_order_invariance_verified,
    }
    return RollingMoneylineOOSResult(
        model_artifacts=tuple(model_artifacts),
        comparison_rows=tuple(comparison_rows),
        per_window_summary=tuple(per_window_summary),
        summary=summary,
    )


def run_deterministic_rolling_moneyline_oos(
    repository_root: str | Path,
    *,
    fit_runtime: str | Path = P22B_DEFAULT_FIT_RUNTIME,
) -> RollingMoneylineOOSResult:
    """Run P37A twice and require identical frozen-input artifact bytes."""

    first = evaluate_rolling_moneyline_oos(repository_root, fit_runtime=fit_runtime)
    second = evaluate_rolling_moneyline_oos(repository_root, fit_runtime=fit_runtime)
    first_artifacts = (
        render_model_artifacts(first.model_artifacts),
        render_comparisons_jsonl(first.comparison_rows),
        render_per_window_summary(first.per_window_summary),
        render_summary(first.summary),
        render_report_markdown(first.summary),
    )
    second_artifacts = (
        render_model_artifacts(second.model_artifacts),
        render_comparisons_jsonl(second.comparison_rows),
        render_per_window_summary(second.per_window_summary),
        render_summary(second.summary),
        render_report_markdown(second.summary),
    )
    if first_artifacts != second_artifacts:
        raise ValueError("P37A deterministic artifact bytes mismatch")
    verified_summary = json.loads(json.dumps(first.summary, ensure_ascii=False))
    verified_summary["deterministic_rerun_verified"] = True
    verified_summary["verification"]["deterministic_rerun_verified"] = True
    return RollingMoneylineOOSResult(
        model_artifacts=first.model_artifacts,
        comparison_rows=first.comparison_rows,
        per_window_summary=first.per_window_summary,
        summary=verified_summary,
    )


__all__ = (
    "P37A_AUTHORITY_REPOSITORY",
    "P37A_BASE_HEAD",
    "P37A_BASE_TREE",
    "P37A_DECISION_RULE",
    "P37A_EVALUATION_FOLD_IDS",
    "P37A_P22A_DATASET_PATH",
    "P37A_P22A_SUMMARY_PATH",
    "P37A_SEED_TRAINING_FOLD_IDS",
    "P37A_TASK_ID",
    "RollingMoneylineOOSResult",
    "RollingTrainingObservation",
    "RollingWindow",
    "evaluate_rolling_moneyline_oos",
    "run_deterministic_rolling_moneyline_oos",
)
