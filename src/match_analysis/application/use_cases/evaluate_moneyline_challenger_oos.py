"""Evaluate the frozen P22B challenger against the P13 incumbent on wf_004."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...baseball.domain.future_evaluation_fold import (
    FutureEvaluationFold,
    FutureFeatureRow,
    FutureResultRow,
    fingerprint_manifest,
    fingerprint_rows,
)
from ...baseball.domain.moneyline_feature_snapshot import (
    MoneylineFeatureProvenance,
    MoneylineFeatureSnapshot,
)
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
from ...baseball.domain.moneyline_oos_comparison import (
    PairedMoneylineComparison,
    aggregate_metrics,
    build_comparison_row,
    canonical_json_bytes,
    cohort_fingerprint,
    comparison_set_fingerprint,
)
from ...baseball.domain.moneyline_walk_forward_fold import (
    MoneylineWalkForwardFold,
    ReconstructedWalkForwardModel,
)
from ...core.identity import MatchIdentity
from .moneyline_inference_artifacts import load_moneyline_model_artifact
from .moneyline_walk_forward_artifacts import build_moneyline_model_artifact
from .moneyline_oos_comparison_artifacts import (
    render_comparisons_jsonl,
    render_incumbent_model_artifact,
    render_summary,
)


P23A_FOLD_ID = "wf_004"
P23A_INCUMBENT_FOLD_ID = "wf_003"
P23A_INCUMBENT_SOURCE_FOLD_FINGERPRINT = (
    "3c5f9e62fb23620040a2015466f4a48099193cb0e77885b1e0b0e6f4e346b3f1"
)
P23A_INCUMBENT_TRAINING_SEMANTIC_FINGERPRINT = (
    "c69196281a74d986eb1a9825e57340da5130420008341cf38bedaf62d86d2b5f"
)
P23A_INCUMBENT_FIXTURE_BASIS_ID = (
    "ba6fa27f9cd8eb95d81fbe5878ddb3b1e9fff8ec17364c0ffdc4adf6bc3944a3"
)
P23A_INCUMBENT_FIXTURE_EXPECTED_PROBABILITY = "0.594699"
P23A_TRAINING_INFORMATION_BOUNDARY_UTC = "2026-03-12T06:29:35.016973Z"
P23A_INCUMBENT_FIDELITY_ROUTE = "MIGRATED_SEMANTIC_RECONSTRUCTION"
P23A_CHALLENGER_FINGERPRINT = (
    "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e"
)
P23A_SOURCE_DATASET_FINGERPRINT = (
    "05f9b31c608e1630a40b2369ac45ada8b103b2c1131b0cedcaa2c7fc91ba7750"
)
P23A_FEATURE_NAMES = ("recent_win_rate_delta", "starter_era_delta")
P23A_FIT_CONFIGURATION = {
    "max_iter": 1000,
    "model_type": "logistic_regression",
    "solver": "lbfgs",
}


@dataclass(frozen=True, slots=True)
class MoneylineOOSComparisonResult:
    rows: tuple[PairedMoneylineComparison, ...]
    summary: dict[str, Any]
    incumbent_artifact_projection: dict[str, Any]
    incumbent_source_fold_id: str
    incumbent_source_fold_fingerprint: str
    incumbent_training_cutoff: str
    incumbent_training_row_count: int
    incumbent_fidelity_route: str


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


def _sha256_json(value: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _semantic_jsonl_fingerprint(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> str:
    ordered = sorted(
        rows,
        key=lambda row: tuple(str(row[field]) for field in key_fields),
    )
    return sha256(b"".join(canonical_json_bytes(row) for row in ordered)).hexdigest()


def _future_feature(row: Mapping[str, Any]) -> FutureFeatureRow:
    home_starter = row["home_starter"]
    away_starter = row["away_starter"]
    if not isinstance(home_starter, Mapping) or not isinstance(away_starter, Mapping):
        raise ValueError("future feature starter projections must be objects")
    feature = FutureFeatureRow(
        provider_game_id=str(row["provider_game_id"]),
        game_pk=int(row["game_pk"]),
        game_number=int(row["game_number"]),
        official_date=str(row["official_date"]),
        scheduled_start_utc=str(row["scheduled_start_utc"]),
        feature_as_of_utc=str(row["feature_as_of_utc"]),
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        home_starter_id=int(home_starter["id"]),
        home_starter_name=str(home_starter["name"]),
        away_starter_id=int(away_starter["id"]),
        away_starter_name=str(away_starter["name"]),
        recent_win_rate_delta=str(row["features"]["recent_win_rate_delta"]),
        starter_era_delta=str(row["features"]["starter_era_delta"]),
        feature_fingerprint=str(row["feature_fingerprint"]),
    )
    if feature.with_fingerprint().feature_fingerprint != feature.feature_fingerprint:
        raise ValueError("P23F2 feature fingerprint mismatch")
    return feature


def _future_result(row: Mapping[str, Any]) -> FutureResultRow:
    return FutureResultRow(
        provider_game_id=str(row["provider_game_id"]),
        game_pk=int(row["game_pk"]),
        game_number=int(row["game_number"]),
        scheduled_start_utc=str(row["scheduled_start_utc"]),
        home_score=int(row["home_score"]),
        away_score=int(row["away_score"]),
        status=str(row["status"]),
        source_result_id=str(row["source_result_id"]),
    )


def _load_feature_authority(
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[FutureFeatureRow, ...]]:
    report_root = repository_root / "report/p23f2_official_future_fold"
    summary = _read_json(report_root / "summary.json")
    manifest = _read_json(report_root / "fold_manifest.json")
    source_manifest = _read_json(report_root / "source_manifest.json")
    feature_rows = tuple(
        _future_feature(row)
        for row in _read_jsonl(report_root / "feature_rows.jsonl")
    )
    feature_rows = tuple(
        sorted(
            feature_rows,
            key=lambda row: (
                row.scheduled_start_utc,
                row.game_number,
                row.game_pk,
            ),
        )
    )
    if not (
        summary["fold_id"] == manifest["fold_id"] == P23A_FOLD_ID
    ):
        raise ValueError("STOP_MATCHANALYSIS_P23A_FUTURE_FOLD_AUTHORITY_DRIFT")
    if not (summary["game_count"] == manifest["game_count"] == len(feature_rows) == 23):
        raise ValueError("P23F2 game count must be exactly 23")
    if not (
        summary["feature_names"] == manifest["feature_names"] == list(P23A_FEATURE_NAMES)
    ):
        raise ValueError("P23F2 feature schema drift")
    source_fingerprint = _sha256_json(source_manifest)
    if not (
        summary["source_manifest_fingerprint"]
        == manifest["source_manifest_fingerprint"]
        == source_fingerprint
    ):
        raise ValueError("P23F2 source manifest fingerprint mismatch")
    feature_projection = tuple(row.projection() for row in feature_rows)
    feature_fingerprint = fingerprint_rows(feature_projection)
    if not (
        summary["feature_fingerprint"]
        == manifest["feature_fingerprint"]
        == feature_fingerprint
    ):
        raise ValueError("P23F2 feature fingerprint mismatch")
    boundary = parse_canonical_utc(P23A_TRAINING_INFORMATION_BOUNDARY_UTC)
    if summary["training_information_boundary_utc"] != P23A_TRAINING_INFORMATION_BOUNDARY_UTC:
        raise ValueError("STOP_MATCHANALYSIS_P23A_BASELINE_DRIFT")
    for row in feature_rows:
        if parse_canonical_utc(row.scheduled_start_utc) <= boundary:
            raise ValueError("STOP_MATCHANALYSIS_P23A_STRICT_FUTURE_BOUNDARY_FAILED")
        if parse_canonical_utc(row.feature_as_of_utc) >= parse_canonical_utc(
            row.scheduled_start_utc
        ):
            raise ValueError("P23F2 feature row is not point-in-time safe")
    return summary, manifest, source_manifest, feature_rows


def _load_and_validate_results(
    repository_root: Path,
    *,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    feature_rows: tuple[FutureFeatureRow, ...],
) -> tuple[FutureResultRow, ...]:
    path = repository_root / "report/p23f2_official_future_fold/results.jsonl"
    result_rows = tuple(_future_result(row) for row in _read_jsonl(path))
    if len(result_rows) != len(feature_rows):
        raise ValueError("P23F2 results must contain exactly one row per feature")
    feature_ids = {row.provider_game_id for row in feature_rows}
    result_ids = [row.provider_game_id for row in result_rows]
    if len(set(result_ids)) != len(result_ids) or set(result_ids) != feature_ids:
        raise ValueError("P23F2 result identities do not match features")
    feature_by_id = {row.provider_game_id: row for row in feature_rows}
    for result in result_rows:
        feature = feature_by_id[result.provider_game_id]
        if result.scheduled_start_utc != feature.scheduled_start_utc:
            raise ValueError("P23F2 feature/result schedule mismatch")
        if result.status != "Final":
            raise ValueError("P23F2 results must be final")
    result_rows = tuple(
        sorted(
            result_rows,
            key=lambda row: (row.scheduled_start_utc, row.game_number, row.game_pk),
        )
    )
    result_projection = tuple(row.projection() for row in result_rows)
    result_fingerprint = fingerprint_rows(result_projection)
    if not (
        summary["result_fingerprint"]
        == manifest["result_fingerprint"]
        == result_fingerprint
    ):
        raise ValueError("P23F2 result fingerprint mismatch")
    fold_manifest = {
        "fold_id": P23A_FOLD_ID,
        "validation_start": summary["validation_start"],
        "validation_end": summary["validation_end"],
        "feature_fingerprint": summary["feature_fingerprint"],
        "result_fingerprint": summary["result_fingerprint"],
        "source_manifest_fingerprint": summary["source_manifest_fingerprint"],
    }
    computed_fold_fingerprint = fingerprint_manifest(fold_manifest)
    if not (
        summary["fold_fingerprint"]
        == manifest["fold_fingerprint"]
        == computed_fold_fingerprint
    ):
        raise ValueError("STOP_MATCHANALYSIS_P23A_FUTURE_FOLD_AUTHORITY_DRIFT")
    FutureEvaluationFold(
        fold_id=P23A_FOLD_ID,
        training_information_boundary_utc=summary[
            "training_information_boundary_utc"
        ],
        validation_start=summary["validation_start"],
        validation_end=summary["validation_end"],
        feature_rows=feature_rows,
        result_rows=result_rows,
        source_manifest_fingerprint=summary["source_manifest_fingerprint"],
        feature_fingerprint=summary["feature_fingerprint"],
        result_fingerprint=summary["result_fingerprint"],
        fold_fingerprint=summary["fold_fingerprint"],
    )
    return result_rows


def _load_challenger(repository_root: Path) -> tuple[MoneylineModelArtifact, str, dict[str, Any]]:
    path = repository_root / "report/p22b_moneyline_challenger/model_artifact.json"
    projection = _read_json(path)
    artifact = load_moneyline_model_artifact(path)
    full_fingerprint = _sha256_json(
        {key: value for key, value in projection.items() if key != "artifact_fingerprint"}
    )
    if not (
        full_fingerprint
        == projection.get("artifact_fingerprint")
        == P23A_CHALLENGER_FINGERPRINT
    ):
        raise ValueError("STOP_MATCHANALYSIS_P23A_CHALLENGER_AUTHORITY_DRIFT")
    if projection.get("source_dataset_fingerprint") != P23A_SOURCE_DATASET_FINGERPRINT:
        raise ValueError("P22A source dataset fingerprint drift")
    if projection.get("model_role") != "CHALLENGER":
        raise ValueError("P22B artifact role drift")
    if projection.get("feature_names") != list(P23A_FEATURE_NAMES):
        raise ValueError("P22B feature order drift")
    if projection.get("fit_configuration") != P23A_FIT_CONFIGURATION:
        raise ValueError("P22B fitting configuration drift")
    if projection.get("training_example_count") != 677:
        raise ValueError("P22B training example count drift")
    return artifact, full_fingerprint, projection


def _load_incumbent(
    repository_root: Path,
) -> tuple[
    MoneylineModelArtifact,
    dict[str, Any],
    MoneylineWalkForwardFold,
    ReconstructedWalkForwardModel,
    str,
]:
    fixture_root = repository_root / "data/fixtures/p21b_multifold_historical"
    fold_projection = _read_json(fixture_root / "fold_wf_003.json")
    source_fold = MoneylineWalkForwardFold.from_projection(fold_projection)
    source_fold_fingerprint = source_fold.fingerprint()
    if source_fold_fingerprint != P23A_INCUMBENT_SOURCE_FOLD_FINGERPRINT:
        raise ValueError("incumbent fold fingerprint mismatch")
    training_rows = fold_projection["training_rows"]
    if len(training_rows) != 1212:
        raise ValueError("incumbent training row count drift")
    if _semantic_jsonl_fingerprint(
        training_rows,
        key_fields=("date", "game_id"),
    ) != P23A_INCUMBENT_TRAINING_SEMANTIC_FINGERPRINT:
        raise ValueError("incumbent training-row semantic fingerprint drift")
    fold_projection["training_rows"] = sorted(
        training_rows,
        key=lambda row: (str(row["date"]), str(row["game_id"])),
    )
    prediction_pairs = sorted(
        zip(
            fold_projection["prediction_rows"],
            fold_projection["expected_home_probabilities"],
            strict=True,
        ),
        key=lambda pair: (str(pair[0]["date"]), str(pair[0]["game_id"])),
    )
    fold_projection["prediction_rows"] = [pair[0] for pair in prediction_pairs]
    fold_projection["expected_home_probabilities"] = [
        pair[1] for pair in prediction_pairs
    ]
    fold = MoneylineWalkForwardFold.from_projection(fold_projection)
    model_manifest = _read_json(fixture_root / "reconstructed_models.json")
    model_projection = model_manifest["models"][P23A_INCUMBENT_FOLD_ID]
    model = ReconstructedWalkForwardModel(
        fold_id=str(model_projection["fold_id"]),
        feature_names=tuple(str(item) for item in model_projection["feature_names"]),
        coefficients=tuple(float(item) for item in model_projection["coefficients"]),
        intercept=float(model_projection["intercept"]),
        scaler_means=tuple(float(item) for item in model_projection["scaler_means"]),
        scaler_stds=tuple(float(item) for item in model_projection["scaler_stds"]),
        train_size=int(model_projection["train_size"]),
        model_type=str(model_projection["model_type"]),
        solver=str(model_projection["solver"]),
        max_iter=int(model_projection["max_iter"]),
    )
    if not (fold.fold_id == model.fold_id == P23A_INCUMBENT_FOLD_ID):
        raise ValueError("incumbent fold identity drift")
    if model.fingerprint() != model_projection["fingerprint"]:
        raise ValueError("incumbent reconstructed model fingerprint mismatch")
    p21b_summary = _read_json(
        repository_root / "report/p21b_contiguous_multifold_historical_candidates/summary.json"
    )
    fold_summary = next(
        item for item in p21b_summary["folds"] if item["fold_id"] == P23A_INCUMBENT_FOLD_ID
    )
    if not (
        fold_summary["fold_id"] == P23A_INCUMBENT_FOLD_ID
        and fold_summary["fold_fingerprint"]
        == source_fold_fingerprint
    ):
        raise ValueError("incumbent fold authority fingerprint mismatch")
    if model.fingerprint() != fold_summary["model_fingerprint"]:
        raise ValueError("incumbent model fingerprint mismatch")
    artifact = build_moneyline_model_artifact(fold, model)
    artifact_projection = artifact.to_projection()
    artifact_projection["fixture_basis_id"] = P23A_INCUMBENT_FIXTURE_BASIS_ID
    artifact_projection["fixture_expected_home_probability"] = (
        P23A_INCUMBENT_FIXTURE_EXPECTED_PROBABILITY
    )
    artifact = MoneylineModelArtifact.from_projection(artifact_projection)
    if artifact.fingerprint() != fold_summary["model_artifact_fingerprint"]:
        raise ValueError("incumbent model artifact fingerprint mismatch")
    projection = artifact.to_projection()
    projection["artifact_fingerprint"] = artifact.fingerprint()
    return artifact, projection, fold, model, source_fold_fingerprint


def _validate_no_training_overlap(
    repository_root: Path,
    feature_rows: Sequence[FutureFeatureRow],
) -> None:
    training_rows = _read_jsonl(
        repository_root / "report/p22a_game_level_training_dataset/training_examples.jsonl"
    )
    if len(training_rows) != 677:
        raise ValueError("P22A training example count drift")
    future_ids = {row.provider_game_id for row in feature_rows}
    if future_ids.intersection(str(row["provider_game_id"]) for row in training_rows):
        raise ValueError("P23A training/evaluation overlap detected")


def _snapshot_for_future_feature(row: FutureFeatureRow) -> MoneylineFeatureSnapshot:
    schedule_source_id = sha256(
        f"p23f2:{row.provider_game_id}:{row.scheduled_start_utc}".encode("utf-8")
    ).hexdigest()
    feature_as_of = parse_canonical_utc(row.feature_as_of_utc)
    provenance = tuple(
        MoneylineFeatureProvenance(
            field_name=field_name,
            source_id=f"p23f2:{row.feature_fingerprint}:{field_name}",
            source_kind="p23f2_committed_feature_authority",
            observed_as_of_utc=feature_as_of,
            source_fingerprint=sha256(
                f"p23f2:{row.feature_fingerprint}:{field_name}".encode("utf-8")
            ).hexdigest(),
        )
        for field_name in P23A_FEATURE_NAMES
    )
    return MoneylineFeatureSnapshot.from_record(
        {
            "recent_win_rate_delta": row.recent_win_rate_delta,
            "starter_era_delta": row.starter_era_delta,
        },
        identity=MatchIdentity(
            sport="baseball",
            league="MLB",
            season=2026,
            canonical_game_id=row.provider_game_id,
            home_participant=row.home_team,
            away_participant=row.away_team,
        ),
        provider_namespace="MLB_STATS_API",
        provider_game_id=row.provider_game_id,
        game_number=row.game_number,
        source_schedule_observation_id=schedule_source_id,
        as_of_utc=feature_as_of,
        scheduled_start_utc=parse_canonical_utc(row.scheduled_start_utc),
        feature_provenance=provenance,
    )


def predict_feature_rows(
    feature_rows: Sequence[FutureFeatureRow],
    challenger: MoneylineModelArtifact,
    incumbent: MoneylineModelArtifact,
) -> dict[str, tuple[Decimal, Decimal]]:
    """Freeze both HOME-probability streams using features only."""

    ordered = sorted(
        feature_rows,
        key=lambda row: (row.scheduled_start_utc, row.game_number, row.game_pk),
    )
    predictions: dict[str, tuple[Decimal, Decimal]] = {}
    for row in ordered:
        if row.provider_game_id in predictions:
            raise ValueError("duplicate feature identity")
        snapshot = _snapshot_for_future_feature(row)
        predictions[row.provider_game_id] = (
            challenger.predict_home_probability(snapshot),
            incumbent.predict_home_probability(snapshot),
        )
    if len(predictions) != len(feature_rows):
        raise ValueError("prediction stream dropped a feature row")
    return predictions


def pair_predictions_with_results(
    *,
    feature_rows: Sequence[FutureFeatureRow],
    predictions: Mapping[str, tuple[Decimal, Decimal]],
    result_rows: Sequence[FutureResultRow],
    challenger_model_id: str,
    challenger_model_fingerprint: str,
    incumbent_model_id: str,
    incumbent_model_fingerprint: str,
) -> tuple[PairedMoneylineComparison, ...]:
    """Join frozen predictions to results without changing prediction values."""

    feature_by_id = {row.provider_game_id: row for row in feature_rows}
    result_by_id = {row.provider_game_id: row for row in result_rows}
    if set(feature_by_id) != set(predictions) or set(feature_by_id) != set(result_by_id):
        raise ValueError("P23A cohort identities are incomplete or mismatched")
    rows = []
    for feature in sorted(
        feature_rows,
        key=lambda row: (row.scheduled_start_utc, row.game_number, row.game_pk),
    ):
        challenger_probability, incumbent_probability = predictions[feature.provider_game_id]
        rows.append(
            build_comparison_row(
                fold_id=P23A_FOLD_ID,
                feature_row=feature.projection(),
                result_row=result_by_id[feature.provider_game_id].projection(),
                challenger_model_id=challenger_model_id,
                challenger_model_fingerprint=challenger_model_fingerprint,
                challenger_home_probability=challenger_probability,
                incumbent_model_id=incumbent_model_id,
                incumbent_model_fingerprint=incumbent_model_fingerprint,
                incumbent_home_probability=incumbent_probability,
            )
        )
    if len(rows) != 23:
        raise ValueError("P23A must emit exactly 23 comparison rows")
    return tuple(rows)


def evaluate_moneyline_challenger_oos(
    repository_root: str | Path,
) -> MoneylineOOSComparisonResult:
    """Run one complete offline P23A comparison from committed authority."""

    root = Path(repository_root)
    summary, manifest, source_manifest, feature_rows = _load_feature_authority(root)
    challenger, challenger_fingerprint, challenger_projection = _load_challenger(root)
    (
        incumbent,
        incumbent_projection,
        incumbent_fold,
        incumbent_model,
        incumbent_source_fold_fingerprint,
    ) = _load_incumbent(root)
    _validate_no_training_overlap(root, feature_rows)

    # Both prediction streams are fully frozen before final outcomes are read.
    predictions = predict_feature_rows(feature_rows, challenger, incumbent)
    result_rows = _load_and_validate_results(
        root,
        summary=summary,
        manifest=manifest,
        source_manifest=source_manifest,
        feature_rows=feature_rows,
    )
    rows = pair_predictions_with_results(
        feature_rows=feature_rows,
        predictions=predictions,
        result_rows=result_rows,
        challenger_model_id=str(challenger_projection["model_id"]),
        challenger_model_fingerprint=challenger_fingerprint,
        incumbent_model_id=str(incumbent_projection["model_id"]),
        incumbent_model_fingerprint=str(incumbent_projection["artifact_fingerprint"]),
    )
    metrics = aggregate_metrics(rows)
    summary_projection: dict[str, Any] = {
        "fold_id": P23A_FOLD_ID,
        "game_count": len(rows),
        "cohort_fingerprint": cohort_fingerprint([row.projection() for row in feature_rows]),
        "feature_fingerprint": summary["feature_fingerprint"],
        "result_fingerprint": summary["result_fingerprint"],
        "validation_start": summary["validation_start"],
        "validation_end": summary["validation_end"],
        "training_information_boundary_utc": summary[
            "training_information_boundary_utc"
        ],
        "challenger_model_id": challenger_projection["model_id"],
        "challenger_model_fingerprint": challenger_fingerprint,
        "challenger_source_dataset_fingerprint": challenger_projection[
            "source_dataset_fingerprint"
        ],
        "incumbent_model_id": incumbent_projection["model_id"],
        "incumbent_model_fingerprint": incumbent_projection["artifact_fingerprint"],
        "incumbent_source_fold_id": incumbent_fold.fold_id,
        "incumbent_source_fold_fingerprint": incumbent_source_fold_fingerprint,
        "incumbent_training_cutoff": incumbent_fold.train_as_of,
        "incumbent_training_row_count": incumbent_fold.training_row_count,
        "incumbent_fidelity_route": P23A_INCUMBENT_FIDELITY_ROUTE,
        "comparison_set_fingerprint": comparison_set_fingerprint(rows),
        **metrics,
        "strict_future_boundary_verified": True,
        "pit_safe_feature_reconstruction_verified": True,
        "no_training_overlap_verified": True,
        "challenger_frozen": True,
        "outcome_isolation_verified": True,
        "deterministic_replay_verified": False,
        "source_manifest_fingerprint": source_manifest and summary["source_manifest_fingerprint"],
        "out_of_sample_evaluated": True,
        "evaluation_complete": True,
        "model_promoted": False,
        "promotion_authorized": False,
        "production_ready": False,
        "profitability_claim": False,
        "real_betting_recommendation": False,
        "retraining_performed": False,
        "p20b_historical_runtime_compliance": "REMAINS_REFUTED",
    }
    return MoneylineOOSComparisonResult(
        rows=rows,
        summary=summary_projection,
        incumbent_artifact_projection=incumbent_projection,
        incumbent_source_fold_id=incumbent_fold.fold_id,
        incumbent_source_fold_fingerprint=incumbent_source_fold_fingerprint,
        incumbent_training_cutoff=incumbent_fold.train_as_of,
        incumbent_training_row_count=incumbent_fold.training_row_count,
        incumbent_fidelity_route=P23A_INCUMBENT_FIDELITY_ROUTE,
    )


def _render_result_artifacts(result: MoneylineOOSComparisonResult) -> tuple[str, str, str]:
    return (
        render_comparisons_jsonl(result.rows),
        render_summary(result.summary),
        render_incumbent_model_artifact(
            result.incumbent_artifact_projection,
            source_fold_id=result.incumbent_source_fold_id,
            source_fold_fingerprint=result.incumbent_source_fold_fingerprint,
            training_cutoff=result.incumbent_training_cutoff,
            training_row_count=result.incumbent_training_row_count,
            model_fidelity_route=result.incumbent_fidelity_route,
        ),
    )


def run_deterministic_moneyline_challenger_oos(
    repository_root: str | Path,
) -> MoneylineOOSComparisonResult:
    """Run twice and compare semantic fingerprints and exact artifact bytes."""

    first = evaluate_moneyline_challenger_oos(repository_root)
    second = evaluate_moneyline_challenger_oos(repository_root)
    if first.summary["comparison_set_fingerprint"] != second.summary[
        "comparison_set_fingerprint"
    ]:
        raise ValueError("P23A deterministic comparison fingerprint mismatch")
    if first.summary["cohort_fingerprint"] != second.summary["cohort_fingerprint"]:
        raise ValueError("P23A deterministic cohort fingerprint mismatch")
    first_artifacts = _render_result_artifacts(first)
    second_artifacts = _render_result_artifacts(second)
    if first_artifacts != second_artifacts:
        raise ValueError("P23A deterministic artifact bytes mismatch")
    verified_summary = dict(first.summary)
    verified_summary["deterministic_replay_verified"] = True
    return MoneylineOOSComparisonResult(
        rows=first.rows,
        summary=verified_summary,
        incumbent_artifact_projection=first.incumbent_artifact_projection,
        incumbent_source_fold_id=first.incumbent_source_fold_id,
        incumbent_source_fold_fingerprint=first.incumbent_source_fold_fingerprint,
        incumbent_training_cutoff=first.incumbent_training_cutoff,
        incumbent_training_row_count=first.incumbent_training_row_count,
        incumbent_fidelity_route=first.incumbent_fidelity_route,
    )


__all__ = (
    "MoneylineOOSComparisonResult",
    "P23A_CHALLENGER_FINGERPRINT",
    "P23A_FOLD_ID",
    "P23A_INCUMBENT_FIDELITY_ROUTE",
    "P23A_SOURCE_DATASET_FINGERPRINT",
    "evaluate_moneyline_challenger_oos",
    "pair_predictions_with_results",
    "predict_feature_rows",
    "run_deterministic_moneyline_challenger_oos",
)
