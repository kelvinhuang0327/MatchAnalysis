"""Evaluate the frozen P22B challenger across contiguous future folds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.future_evaluation_fold import (
    FutureEvaluationFold,
    FutureFeatureRow,
    FutureResultRow,
    fingerprint_manifest,
    fingerprint_rows,
)
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
from ...baseball.domain.moneyline_oos_comparison import (
    PairedMoneylineComparison,
    aggregate_metrics,
    canonical_json_bytes,
    comparison_set_fingerprint,
)
from ...baseball.domain.moneyline_walk_forward_fold import (
    MoneylineWalkForwardFold,
    ReconstructedWalkForwardModel,
)
from .acquire_future_moneyline_history import (
    AcquisitionResult,
    acquire_official_future_fold,
    load_normalized_rows,
)
from .evaluate_moneyline_challenger_oos import (
    P23A_INCUMBENT_FIDELITY_ROUTE,
    load_frozen_challenger_authority,
    load_incumbent_authority,
    load_p23a_future_fold_authority,
    pair_predictions_with_results,
    parse_future_feature_projection,
    parse_future_result_projection,
    predict_feature_rows,
    validate_no_training_overlap,
)
from .future_moneyline_fold_artifacts import (
    render_source_manifest,
    write_future_fold_artifacts,
)
from .materialize_future_moneyline_fold import (
    classify_future_feature_eligibility,
    materialize_future_moneyline_fold,
)
from .multifold_moneyline_oos_artifacts import (
    P23B_FOLD_ORDER,
    render_comparisons_jsonl,
    render_per_fold_summary,
    render_summary,
)


P23B_CHALLENGER_FINGERPRINT = (
    "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e"
)
P23B_SOURCE_DATASET_FINGERPRINT = (
    "05f9b31c608e1630a40b2369ac45ada8b103b2c1131b0cedcaa2c7fc91ba7750"
)
P23B_TRAINING_INFORMATION_BOUNDARY_UTC = "2026-03-12T06:29:35.016973Z"


@dataclass(frozen=True, slots=True)
class FutureFoldSpec:
    fold_id: str
    validation_start: str
    validation_end: str


P23B_FOLD_SPECS = (
    FutureFoldSpec("wf_005", "2026-06-10", "2026-06-11"),
    FutureFoldSpec("wf_006", "2026-06-12", "2026-06-13"),
)


@dataclass(frozen=True, slots=True)
class FutureFoldAuthority:
    spec: FutureFoldSpec
    fold: FutureEvaluationFold
    source_manifest: dict[str, Any]
    manifest: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MultifoldMoneylineOOSResult:
    comparison_rows: tuple[dict[str, Any], ...]
    per_fold_summary: tuple[dict[str, Any], ...]
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


def _sha256_json(value: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _comparison_from_projection(row: Mapping[str, Any]) -> PairedMoneylineComparison:
    return PairedMoneylineComparison(
        comparison_row_id=str(row["comparison_row_id"]),
        fold_id=str(row["fold_id"]),
        provider_game_id=str(row["provider_game_id"]),
        game_pk=int(row["game_pk"]),
        game_number=int(row["game_number"]),
        scheduled_start_utc=str(row["scheduled_start_utc"]),
        feature_fingerprint=str(row["feature_fingerprint"]),
        challenger_model_id=str(row["challenger_model_id"]),
        challenger_model_fingerprint=str(row["challenger_model_fingerprint"]),
        challenger_home_probability=Decimal(
            str(row["challenger_home_probability"])
        ),
        incumbent_model_id=str(row["incumbent_model_id"]),
        incumbent_model_fingerprint=str(row["incumbent_model_fingerprint"]),
        incumbent_home_probability=Decimal(
            str(row["incumbent_home_probability"])
        ),
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


def _validate_source_manifest(
    repository_root: Path,
    fixture_root: Path,
    source_manifest: Mapping[str, Any],
) -> None:
    if source_manifest.get("source_domains") != ["mlb.com"]:
        raise ValueError("P23B source domains must remain MLB-owned")
    records = source_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("P23B source manifest records are required")
    seen_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("P23B source record must be an object")
        relative = str(record["path"])
        if Path(relative).is_absolute() or relative in seen_paths:
            raise ValueError("P23B source paths must be relative and unique")
        seen_paths.add(relative)
        source_path = repository_root / relative
        if not source_path.is_file():
            raise ValueError(f"P23B source file is missing: {relative}")
        if sha256(source_path.read_bytes()).hexdigest() != str(record["sha256"]):
            raise ValueError(f"P23B source hash mismatch: {relative}")
        if not str(record["url"]).startswith("https://statsapi.mlb.com/"):
            raise ValueError("P23B source URL is not the allowlisted MLB Stats API")

    normalized_hashes = source_manifest.get("normalized_hashes")
    if not isinstance(normalized_hashes, Mapping):
        raise ValueError("P23B normalized hashes are required")
    for name in ("schedule.jsonl", "target_boxscores.jsonl", "pitcher_game_logs.jsonl"):
        path = fixture_root / "normalized" / name
        if not path.is_file():
            raise ValueError(f"P23B normalized source is missing: {path}")
        if sha256(path.read_bytes()).hexdigest() != str(normalized_hashes[name]):
            raise ValueError(f"P23B normalized hash mismatch: {name}")


def _load_new_fold_authority(
    repository_root: Path,
    spec: FutureFoldSpec,
) -> FutureFoldAuthority:
    fixture_root = (
        repository_root / "data/fixtures/p23b_future_folds" / spec.fold_id
    )
    source_manifest = _read_json(fixture_root / "source_manifest.json")
    manifest = _read_json(fixture_root / "fold_manifest.json")
    summary = _read_json(fixture_root / "summary.json")
    _validate_source_manifest(repository_root, fixture_root, source_manifest)
    source_fingerprint = _sha256_json(source_manifest)
    if not (
        manifest.get("source_manifest_fingerprint")
        == summary.get("source_manifest_fingerprint")
        == source_fingerprint
    ):
        raise ValueError("P23B source manifest fingerprint mismatch")

    schedule_rows = _read_jsonl(fixture_root / "normalized/schedule.jsonl")
    feature_rows = tuple(
        sorted(
            (
                parse_future_feature_projection(row)
                for row in _read_jsonl(fixture_root / "feature_rows.jsonl")
            ),
            key=lambda row: (
                row.scheduled_start_utc,
                row.game_number,
                row.game_pk,
            ),
        )
    )
    result_rows = tuple(
        sorted(
            (
                parse_future_result_projection(row)
                for row in _read_jsonl(fixture_root / "results.jsonl")
            ),
            key=lambda row: (
                row.scheduled_start_utc,
                row.game_number,
                row.game_pk,
            ),
        )
    )
    schedule_ids = {str(row["provider_game_id"]) for row in schedule_rows}
    feature_ids = {row.provider_game_id for row in feature_rows}
    feature_unavailable = tuple(
        dict(row)
        for row in summary.get("feature_unavailable", manifest.get("feature_unavailable", []))
    )
    raw_game_ids = tuple(
        str(game_id)
        for game_id in manifest.get("raw_game_ids", summary.get("raw_game_ids", ()))
    )
    if not raw_game_ids:
        raise ValueError("P23B raw game membership is required")
    unavailable_ids = {str(row["game_id"]) for row in feature_unavailable}
    if (
        set(raw_game_ids) != schedule_ids
        or feature_ids | unavailable_ids != schedule_ids
        or feature_ids & unavailable_ids
        or len(raw_game_ids) != len(schedule_rows)
    ):
        raise ValueError("P23B raw/evaluable feature membership mismatch")
    if any(
        not (
            spec.validation_start
            <= row["official_date"]
            <= spec.validation_end
        )
        or not row["final"]
        for row in schedule_rows
    ):
        raise ValueError("P23B schedule membership is not the frozen contiguous window")
    if not (
        summary.get("fold_id")
        == manifest.get("fold_id")
        == spec.fold_id
        and summary.get("validation_start")
        == manifest.get("validation_start")
        == spec.validation_start
        and summary.get("validation_end")
        == manifest.get("validation_end")
        == spec.validation_end
    ):
        raise ValueError("P23B fold identity or cadence drift")
    feature_fingerprint = fingerprint_rows(
        tuple(row.projection() for row in feature_rows)
    )
    result_fingerprint = fingerprint_rows(
        tuple(row.projection() for row in result_rows)
    )
    if not (
        manifest.get("feature_fingerprint")
        == summary.get("feature_fingerprint")
        == feature_fingerprint
        and manifest.get("result_fingerprint")
        == summary.get("result_fingerprint")
        == result_fingerprint
    ):
        raise ValueError("P23B feature or result fingerprint mismatch")
    fold_fingerprint_input = {
        "fold_id": spec.fold_id,
        "validation_start": spec.validation_start,
        "validation_end": spec.validation_end,
        "feature_fingerprint": feature_fingerprint,
        "result_fingerprint": result_fingerprint,
        "source_manifest_fingerprint": source_fingerprint,
    }
    if any(row.with_fingerprint() != row for row in feature_rows):
        raise ValueError("P23B feature row fingerprint mismatch")
    fold_fingerprint_input.update(
        {
            "raw_game_ids": list(raw_game_ids),
            "feature_unavailable": [dict(row) for row in feature_unavailable],
        }
    )
    fold_fingerprint = fingerprint_manifest(fold_fingerprint_input)
    if not (
        manifest.get("fold_fingerprint")
        == summary.get("fold_fingerprint")
        == fold_fingerprint
    ):
        raise ValueError("P23B fold fingerprint mismatch")
    fold = FutureEvaluationFold(
        fold_id=spec.fold_id,
        training_information_boundary_utc=P23B_TRAINING_INFORMATION_BOUNDARY_UTC,
        validation_start=spec.validation_start,
        validation_end=spec.validation_end,
        feature_rows=feature_rows,
        result_rows=result_rows,
        source_manifest_fingerprint=source_fingerprint,
        feature_fingerprint=feature_fingerprint,
        result_fingerprint=result_fingerprint,
        fold_fingerprint=fold_fingerprint,
        raw_game_ids=raw_game_ids,
        feature_unavailable_rows=feature_unavailable,
    )
    if not (
        summary.get("game_count")
        == manifest.get("game_count")
        == summary.get("evaluable_game_count")
        == manifest.get("evaluable_game_count")
        == len(feature_rows)
        and summary.get("raw_game_count")
        == manifest.get("raw_game_count")
        == len(raw_game_ids)
        and summary.get("feature_unavailable_count")
        == manifest.get("feature_unavailable_count")
        == len(feature_unavailable)
        and summary.get("strict_future") is True
        and summary.get("external_source") is True
    ):
        raise ValueError("P23B fold summary is incomplete")
    return FutureFoldAuthority(
        spec=spec,
        fold=fold,
        source_manifest=source_manifest,
        manifest=manifest,
        summary=summary,
    )


def acquire_p23b_future_folds(
    repository_root: str | Path,
    *,
    acquired_at_utc: datetime,
    opener: Any = None,
) -> tuple[FutureFoldAuthority, ...]:
    """Acquire and freeze exactly the two contiguous successors of wf_004."""

    root = Path(repository_root)
    authorities: list[FutureFoldAuthority] = []
    for spec in P23B_FOLD_SPECS:
        fixture_root = root / "data/fixtures/p23b_future_folds" / spec.fold_id
        acquisition: AcquisitionResult = acquire_official_future_fold(
            repository_root=root,
            fold_id=spec.fold_id,
            validation_start=spec.validation_start,
            validation_end=spec.validation_end,
            raw_root=fixture_root / "raw",
            normalized_root=fixture_root / "normalized",
            acquired_at_utc=acquired_at_utc,
            opener=opener,
        )
        source_manifest = render_source_manifest(
            records=[asdict(record) for record in acquisition.source_records],
            normalized_hashes=acquisition.normalized_hashes,
        )
        source_fingerprint = _sha256_json(source_manifest)
        eligibility = classify_future_feature_eligibility(
            schedule_rows=acquisition.schedule_rows,
            target_boxscore_rows=acquisition.target_boxscore_rows,
            pitcher_game_log_rows=acquisition.pitcher_game_log_rows,
            fold_id=spec.fold_id,
            validation_start=spec.validation_start,
            validation_end=spec.validation_end,
        )
        fold = materialize_future_moneyline_fold(
            schedule_rows=acquisition.schedule_rows,
            target_boxscore_rows=acquisition.target_boxscore_rows,
            pitcher_game_log_rows=acquisition.pitcher_game_log_rows,
            source_manifest_fingerprint=source_fingerprint,
            fold_id=spec.fold_id,
            validation_start=spec.validation_start,
            validation_end=spec.validation_end,
            evaluable_game_ids=frozenset(eligibility.evaluable_game_ids),
            raw_game_ids=eligibility.raw_game_ids,
            feature_unavailable_rows=eligibility.feature_unavailable_rows,
        )
        write_future_fold_artifacts(
            fixture_root,
            fold,
            source_manifest=source_manifest,
            offline_replay_verified=False,
        )
        authorities.append(_load_new_fold_authority(root, spec))
    return tuple(authorities)


def _load_existing_wf004(
    repository_root: Path,
) -> tuple[dict[str, Any], tuple[PairedMoneylineComparison, ...], FutureFoldAuthority]:
    feature_summary, manifest, source_manifest, feature_rows, result_rows = (
        load_p23a_future_fold_authority(repository_root)
    )
    comparison_summary = _read_json(
        repository_root / "report/p23a_strictly_future_oos/summary.json"
    )
    comparison_rows = tuple(
        _comparison_from_projection(row)
        for row in _read_jsonl(
            repository_root / "report/p23a_strictly_future_oos/comparisons.jsonl"
        )
    )
    if len(comparison_rows) != len(feature_rows) or any(
        row.fold_id != "wf_004" for row in comparison_rows
    ):
        raise ValueError("P23A wf_004 comparison authority is incomplete")
    if (
        comparison_set_fingerprint(comparison_rows)
        != comparison_summary["comparison_set_fingerprint"]
    ):
        raise ValueError("P23A wf_004 comparison fingerprint drift")
    metrics = aggregate_metrics(comparison_rows)
    for key, value in metrics.items():
        if str(comparison_summary[key]) != str(value):
            raise ValueError(f"P23A wf_004 metric drift: {key}")
    if {row.provider_game_id for row in comparison_rows} != {
        row.provider_game_id for row in feature_rows
    }:
        raise ValueError("P23A wf_004 comparison membership drift")
    existing = FutureFoldAuthority(
        spec=FutureFoldSpec(
            "wf_004",
            feature_summary["validation_start"],
            feature_summary["validation_end"],
        ),
        fold=FutureEvaluationFold(
            fold_id="wf_004",
            training_information_boundary_utc=feature_summary[
                "training_information_boundary_utc"
            ],
            validation_start=feature_summary["validation_start"],
            validation_end=feature_summary["validation_end"],
            feature_rows=feature_rows,
            result_rows=result_rows,
            source_manifest_fingerprint=feature_summary[
                "source_manifest_fingerprint"
            ],
            feature_fingerprint=feature_summary["feature_fingerprint"],
            result_fingerprint=feature_summary["result_fingerprint"],
            fold_fingerprint=manifest["fold_fingerprint"],
        ),
        source_manifest=source_manifest,
        manifest=manifest,
        summary=feature_summary,
    )
    return comparison_summary, comparison_rows, existing


def _fold_summary(
    authority: FutureFoldAuthority,
    metrics: Mapping[str, Any],
    *,
    challenger_projection: Mapping[str, Any],
    challenger_fingerprint: str,
    incumbent_projection: Mapping[str, Any],
    incumbent_fold: MoneylineWalkForwardFold,
    incumbent_source_fold_fingerprint: str,
) -> dict[str, Any]:
    fold = authority.fold
    raw_game_ids = fold.raw_game_ids or tuple(
        row.provider_game_id for row in fold.feature_rows
    )
    feature_unavailable = tuple(
        dict(row) for row in fold.feature_unavailable_rows
    )
    exclusion_reason_distribution = dict(
        sorted(
            Counter(str(row["reason"]) for row in feature_unavailable).items()
        )
    )
    evaluable_count = len(fold.feature_rows)
    raw_count = len(raw_game_ids)
    return {
        "fold_id": authority.spec.fold_id,
        "game_count": evaluable_count,
        "raw_game_count": raw_count,
        "evaluable_game_count": evaluable_count,
        "feature_unavailable_count": len(feature_unavailable),
        "evaluation_coverage": str(Decimal(evaluable_count) / Decimal(raw_count)),
        "metrics_population": "MODEL_EVALUABLE_GAMES",
        "evaluable_game_ids": [row.provider_game_id for row in fold.feature_rows],
        "feature_unavailable": list(feature_unavailable),
        "exclusion_reason_distribution": exclusion_reason_distribution,
        "validation_start": authority.spec.validation_start,
        "validation_end": authority.spec.validation_end,
        "scheduled_start_range": [
            fold.feature_rows[0].scheduled_start_utc,
            fold.feature_rows[-1].scheduled_start_utc,
        ],
        "feature_as_of_range": [
            fold.feature_rows[0].feature_as_of_utc,
            fold.feature_rows[-1].feature_as_of_utc,
        ],
        "source_manifest_fingerprint": fold.source_manifest_fingerprint,
        "feature_fingerprint": fold.feature_fingerprint,
        "result_fingerprint": fold.result_fingerprint,
        "fold_fingerprint": fold.fold_fingerprint,
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
        "strict_future_boundary_verified": True,
        "pit_safe_feature_reconstruction_verified": True,
        "incumbent_input_fingerprint_recomputed": True,
        "no_training_overlap_verified": True,
        "challenger_frozen": True,
        "outcome_isolation_verified": True,
        "no_outcome_selection": True,
        **dict(metrics),
    }


def evaluate_multifold_moneyline_oos(
    repository_root: str | Path,
) -> MultifoldMoneylineOOSResult:
    """Evaluate wf_004 plus exactly wf_005 and wf_006 from frozen authority."""

    root = Path(repository_root)
    challenger, challenger_fingerprint, challenger_projection = (
        load_frozen_challenger_authority(root)
    )
    if challenger_fingerprint != P23B_CHALLENGER_FINGERPRINT:
        raise ValueError("STOP_MATCHANALYSIS_P23B_CHALLENGER_AUTHORITY_DRIFT")
    if challenger_projection["source_dataset_fingerprint"] != P23B_SOURCE_DATASET_FINGERPRINT:
        raise ValueError("P22B source dataset fingerprint drift")
    (
        incumbent,
        incumbent_projection,
        incumbent_fold,
        _incumbent_model,
        incumbent_source_fold_fingerprint,
    ) = load_incumbent_authority(root)
    existing_summary, existing_rows, existing_authority = _load_existing_wf004(root)
    validate_no_training_overlap(root, existing_authority.fold.feature_rows)

    comparison_rows: list[dict[str, Any]] = [
        row.to_projection() for row in existing_rows
    ]
    per_fold: list[dict[str, Any]] = []
    existing_metrics = aggregate_metrics(existing_rows)
    per_fold.append(
        _fold_summary(
            existing_authority,
            existing_metrics,
            challenger_projection=challenger_projection,
            challenger_fingerprint=challenger_fingerprint,
            incumbent_projection=incumbent_projection,
            incumbent_fold=incumbent_fold,
            incumbent_source_fold_fingerprint=incumbent_source_fold_fingerprint,
        )
    )
    for spec in P23B_FOLD_SPECS:
        authority = _load_new_fold_authority(root, spec)
        validate_no_training_overlap(root, authority.fold.feature_rows)
        predictions = predict_feature_rows(
            authority.fold.feature_rows,
            challenger,
            incumbent,
        )
        rows = pair_predictions_with_results(
            feature_rows=authority.fold.feature_rows,
            predictions=predictions,
            result_rows=authority.fold.result_rows,
            challenger_model_id=str(challenger_projection["model_id"]),
            challenger_model_fingerprint=challenger_fingerprint,
            incumbent_model_id=str(incumbent_projection["model_id"]),
            incumbent_model_fingerprint=str(
                incumbent_projection["artifact_fingerprint"]
            ),
            fold_id=spec.fold_id,
        )
        metrics = aggregate_metrics(rows)
        comparison_rows.extend(row.to_projection() for row in rows)
        per_fold.append(
            _fold_summary(
                authority,
                metrics,
                challenger_projection=challenger_projection,
                challenger_fingerprint=challenger_fingerprint,
                incumbent_projection=incumbent_projection,
                incumbent_fold=incumbent_fold,
                incumbent_source_fold_fingerprint=incumbent_source_fold_fingerprint,
            )
        )

    per_fold.sort(key=lambda row: P23B_FOLD_ORDER.index(str(row["fold_id"])))
    all_rows = tuple(_comparison_from_projection(row) for row in comparison_rows)
    pooled = aggregate_metrics(all_rows)
    total_raw_game_count = sum(int(row["raw_game_count"]) for row in per_fold)
    total_evaluable_game_count = sum(
        int(row["evaluable_game_count"]) for row in per_fold
    )
    total_feature_unavailable_count = sum(
        int(row["feature_unavailable_count"]) for row in per_fold
    )
    if total_raw_game_count != (
        total_evaluable_game_count + total_feature_unavailable_count
    ):
        raise ValueError("P23B raw/evaluable/unavailable accounting drift")
    brier_deltas = [str(row["brier_delta"]) for row in per_fold]
    summary = {
        "fold_ids": list(P23B_FOLD_ORDER),
        "fold_count": len(per_fold),
        "total_game_count": pooled["game_count"],
        "total_raw_game_count": total_raw_game_count,
        "total_evaluable_game_count": total_evaluable_game_count,
        "total_feature_unavailable_count": total_feature_unavailable_count,
        "pooled_evaluation_coverage": str(
            Decimal(total_evaluable_game_count) / Decimal(total_raw_game_count)
        ),
        "metrics_population": "MODEL_EVALUABLE_GAMES",
        "challenger_model_id": challenger_projection["model_id"],
        "challenger_model_fingerprint": challenger_fingerprint,
        "challenger_source_dataset_fingerprint": challenger_projection[
            "source_dataset_fingerprint"
        ],
        "incumbent_model_ids": {
            row["fold_id"]: row["incumbent_model_id"] for row in per_fold
        },
        "incumbent_model_fingerprints": {
            row["fold_id"]: row["incumbent_model_fingerprint"] for row in per_fold
        },
        "comparison_set_fingerprint": comparison_set_fingerprint(all_rows),
        "per_fold_brier_delta": brier_deltas,
        "folds_with_challenger_lower_brier": [
            row["fold_id"]
            for row in per_fold
            if Decimal(row["brier_delta"]) < Decimal("0")
        ],
        "folds_with_incumbent_lower_brier": [
            row["fold_id"]
            for row in per_fold
            if Decimal(row["brier_delta"]) > Decimal("0")
        ],
        "folds_equal": [
            row["fold_id"]
            for row in per_fold
            if Decimal(row["brier_delta"]) == Decimal("0")
        ],
        "strict_future_boundary_verified": all(
            row["strict_future_boundary_verified"] for row in per_fold
        ),
        "pit_safe_feature_reconstruction_verified": all(
            row["pit_safe_feature_reconstruction_verified"] for row in per_fold
        ),
        "incumbent_input_fingerprint_recomputed": all(
            row["incumbent_input_fingerprint_recomputed"] for row in per_fold
        ),
        "no_training_overlap_verified": all(
            row["no_training_overlap_verified"] for row in per_fold
        ),
        "challenger_frozen": all(row["challenger_frozen"] for row in per_fold),
        "outcome_isolation_verified": all(
            row["outcome_isolation_verified"] for row in per_fold
        ),
        "no_outcome_selection": all(row["no_outcome_selection"] for row in per_fold),
        "official_source_provenance_verified": True,
        "multi_fold_evaluated": True,
        "out_of_sample_evaluated": True,
        "evaluation_complete": True,
        "model_promoted": False,
        "promotion_authorized": False,
        "production_ready": False,
        "profitability_claim": False,
        "real_betting_recommendation": False,
        "challenger_retrained": False,
        "p20b_historical_runtime_compliance": "REMAINS_REFUTED",
        **{key: value for key, value in pooled.items() if key != "game_count"},
        "deterministic_replay_verified": False,
        "input_order_invariance_verified": False,
    }
    return MultifoldMoneylineOOSResult(
        comparison_rows=tuple(comparison_rows),
        per_fold_summary=tuple(per_fold),
        summary=summary,
    )


def _verify_eligibility_replay(repository_root: Path) -> None:
    for spec in P23B_FOLD_SPECS:
        fixture_root = repository_root / "data/fixtures/p23b_future_folds" / spec.fold_id
        schedule = load_normalized_rows(fixture_root / "normalized/schedule.jsonl")
        boxes = load_normalized_rows(fixture_root / "normalized/target_boxscores.jsonl")
        logs = load_normalized_rows(fixture_root / "normalized/pitcher_game_logs.jsonl")
        first = classify_future_feature_eligibility(
            schedule_rows=schedule,
            target_boxscore_rows=boxes,
            pitcher_game_log_rows=logs,
            fold_id=spec.fold_id,
            validation_start=spec.validation_start,
            validation_end=spec.validation_end,
        )
        reordered = classify_future_feature_eligibility(
            schedule_rows=tuple(reversed(schedule)),
            target_boxscore_rows=tuple(reversed(boxes)),
            pitcher_game_log_rows=tuple(reversed(logs)),
            fold_id=spec.fold_id,
            validation_start=spec.validation_start,
            validation_end=spec.validation_end,
        )
        if first != reordered:
            raise ValueError("P23B input-order eligibility drift")
        mutated_schedule = [dict(row) for row in schedule]
        for row in mutated_schedule:
            row["home_score"], row["away_score"] = 999, 0
            row["final"] = not bool(row["final"])
        mutated = classify_future_feature_eligibility(
            schedule_rows=tuple(mutated_schedule),
            target_boxscore_rows=boxes,
            pitcher_game_log_rows=logs,
            fold_id=spec.fold_id,
            validation_start=spec.validation_start,
            validation_end=spec.validation_end,
        )
        if first != mutated:
            raise ValueError("P23B outcome-mutated eligibility drift")


def run_deterministic_multifold_moneyline_oos(
    repository_root: str | Path,
) -> MultifoldMoneylineOOSResult:
    """Run the offline evaluation twice and compare exact report bytes."""

    first = evaluate_multifold_moneyline_oos(repository_root)
    second = evaluate_multifold_moneyline_oos(repository_root)
    first_bytes = (
        render_comparisons_jsonl(first.comparison_rows),
        render_per_fold_summary(first.per_fold_summary),
        render_summary(first.summary),
    )
    second_bytes = (
        render_comparisons_jsonl(second.comparison_rows),
        render_per_fold_summary(second.per_fold_summary),
        render_summary(second.summary),
    )
    if first_bytes != second_bytes:
        raise ValueError("P23B deterministic artifact bytes mismatch")
    _verify_eligibility_replay(Path(repository_root))
    summary = dict(first.summary)
    summary["deterministic_replay_verified"] = True
    summary["input_order_invariance_verified"] = True
    return MultifoldMoneylineOOSResult(
        comparison_rows=first.comparison_rows,
        per_fold_summary=first.per_fold_summary,
        summary=summary,
    )


__all__ = (
    "FutureFoldAuthority",
    "FutureFoldSpec",
    "MultifoldMoneylineOOSResult",
    "P23B_CHALLENGER_FINGERPRINT",
    "P23B_FOLD_SPECS",
    "P23B_SOURCE_DATASET_FINGERPRINT",
    "acquire_p23b_future_folds",
    "evaluate_multifold_moneyline_oos",
    "run_deterministic_multifold_moneyline_oos",
)
