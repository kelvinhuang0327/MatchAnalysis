"""Run and replay one frozen on-demand daily Moneyline paper analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .acquire_tsl_moneyline_snapshot import (
    TslMoneylineSnapshotAcquisition,
    acquire_tsl_moneyline_snapshot,
    build_p30a_paper_inputs,
)
from .paper_moneyline_batch_artifacts import (
    P22B_ARTIFACT_FINGERPRINT,
    P22B_MODEL_ID,
    default_paper_moneyline_model_artifact_path,
    load_default_paper_moneyline_model_artifact,
    load_model_artifact_with_fingerprint,
    canonical_json_bytes,
)
from .moneyline_paper_run_bundle import (
    FrozenMoneylinePaperRunInputs,
    P33A_BUNDLE_SCHEMA_VERSION,
    jsonl_fingerprint,
    load_frozen_bundle_inputs,
    portable_source_manifest,
    read_json_object,
    read_jsonl_objects,
    source_snapshot_identity,
    target_game_membership,
    write_frozen_bundle_inputs,
    write_json_object,
    write_run_outputs,
)


P33A_SCHEMA_VERSION = "p33a.daily_moneyline_paper_run.v1"
P33A_OPERATION = "RUN_DAILY_MONEYLINE_PAPER_ANALYSIS"
P30A_SCHEMA_VERSION = "p30a.moneyline_paper_analysis.v1"
P33A_RUNTIME_ROOT = Path("/tmp/matchanalysis-p33a-daily-paper-run")

STOP_P33A_UNEXPECTED_RUNTIME_WRITE = (
    "STOP_MATCHANALYSIS_P33A_UNEXPECTED_RUNTIME_WRITE"
)
STOP_P33A_MANDATORY_VERIFICATION = (
    "STOP_MATCHANALYSIS_P33A_MANDATORY_VERIFICATION_FAILED"
)
STOP_P33A_JUDGE_NOT_VERIFIED = "STOP_MATCHANALYSIS_P33A_JUDGE_NOT_VERIFIED"


AnalysisRunner = Callable[..., Any]
AcquisitionFactory = Callable[..., TslMoneylineSnapshotAcquisition]


@dataclass(frozen=True, slots=True)
class DailyMoneylinePaperRunResult:
    """One live or offline P33A result."""

    bundle_root: Path
    analysis: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    run_manifest: dict[str, Any]
    offline_replay_equal: bool


def _runtime_path(path: str | Path, *, runtime_root: str | Path) -> Path:
    allowed_root = Path(P33A_RUNTIME_ROOT).resolve()
    selected_root = Path(runtime_root).resolve()
    try:
        selected_root.relative_to(allowed_root)
        candidate = Path(path).resolve()
        candidate.relative_to(selected_root)
    except ValueError as exc:
        raise RuntimeError(STOP_P33A_UNEXPECTED_RUNTIME_WRITE) from exc
    return candidate


def _relative_capture_paths(
    acquisition_runtime_root: Path,
    capture_paths: Sequence[str],
) -> tuple[str, ...]:
    relative_paths: list[str] = []
    for capture_path in capture_paths:
        try:
            relative = Path(capture_path).relative_to(acquisition_runtime_root)
        except ValueError as exc:
            raise RuntimeError(
                f"{STOP_P33A_MANDATORY_VERIFICATION}: capture escaped runtime root"
            ) from exc
        relative_paths.append(relative.as_posix())
    return tuple(sorted(relative_paths))


def _load_p22b_model_bytes(repository_root: Path) -> tuple[bytes, str]:
    model_path = default_paper_moneyline_model_artifact_path(repository_root)
    raw = model_path.read_bytes()
    artifact, fingerprint = load_model_artifact_with_fingerprint(model_path)
    if artifact.model_id != P22B_MODEL_ID or fingerprint != P22B_ARTIFACT_FINGERPRINT:
        raise RuntimeError(f"{STOP_P33A_MANDATORY_VERIFICATION}: P22B identity drift")
    return raw, fingerprint


def _assert_bundle_model_identity(
    *,
    repository_root: Path,
    bundle_root: Path,
) -> None:
    repository_artifact, repository_fingerprint = load_default_paper_moneyline_model_artifact(
        repository_root
    )
    bundle_artifact, bundle_fingerprint = load_model_artifact_with_fingerprint(
        bundle_root / "model_artifact.json"
    )
    if (
        repository_artifact.model_id != P22B_MODEL_ID
        or repository_fingerprint != P22B_ARTIFACT_FINGERPRINT
        or bundle_artifact.model_id != P22B_MODEL_ID
        or bundle_fingerprint != P22B_ARTIFACT_FINGERPRINT
        or repository_fingerprint != bundle_fingerprint
    ):
        raise RuntimeError(f"{STOP_P33A_MANDATORY_VERIFICATION}: P22B identity drift")


def _analysis_runner(analysis_runner: AnalysisRunner | None) -> AnalysisRunner:
    if analysis_runner is not None:
        return analysis_runner
    # Keep the replay loader import-free from acquisition and transport code.
    from .run_moneyline_paper_analysis import run_moneyline_paper_analysis

    return run_moneyline_paper_analysis


def _run_p30a_from_frozen_inputs(
    *,
    repository_root: Path,
    inputs: FrozenMoneylinePaperRunInputs,
    analysis_runner: AnalysisRunner | None,
) -> Any:
    runner = _analysis_runner(analysis_runner)
    return runner(
        repository_root=repository_root,
        tsl_rows=inputs.tsl_rows,
        tsl_raw_sha256=str(inputs.run_manifest["source_snapshot"]["tsl_selected_rows_sha256"]),
        schedule_rows=inputs.schedule_rows,
        target_boxscore_rows=inputs.target_boxscore_rows,
        pitcher_game_log_rows=inputs.pitcher_game_log_rows,
        source_manifest=inputs.source_manifest,
        offline_replay_verified=True,
        cohort_start_date=str(inputs.run_manifest["target_date"]),
        cohort_end_date=str(inputs.run_manifest["target_date"]),
        requested_game_ids=tuple(
            str(game_id) for game_id in inputs.run_manifest["target_game_ids"]
        ),
        allow_missing_starter_identity=True,
        allow_insufficient_evaluable=True,
    )


def _assert_structural_accounting(summary: Mapping[str, Any]) -> dict[str, int]:
    raw_game_count = int(summary["raw_game_count"])
    counts = {
        str(key): int(value)
        for key, value in dict(summary["structural_status_counts"]).items()
    }
    if raw_game_count != sum(counts.values()):
        raise RuntimeError(
            f"{STOP_P33A_MANDATORY_VERIFICATION}: raw game accounting is incomplete"
        )
    return counts


def _bundle_fingerprint(
    *,
    run_fingerprint: str,
    source_identity: Mapping[str, Any],
    model_id: str,
    model_fingerprint: str,
    p30a_summary: Mapping[str, Any],
    analysis_jsonl_sha256: str,
) -> str:
    projection = {
        "run_fingerprint": run_fingerprint,
        "source_snapshot": dict(source_identity),
        "model_id": model_id,
        "model_fingerprint": model_fingerprint,
        "analysis_contract_version": P30A_SCHEMA_VERSION,
        "p30a_run_id": p30a_summary["run_id"],
        "p30a_analysis_set_fingerprint": p30a_summary["analysis_set_fingerprint"],
        "analysis_jsonl_sha256": analysis_jsonl_sha256,
    }
    return sha256(canonical_json_bytes(projection)).hexdigest()


def _build_summary(
    *,
    p30a_summary: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    bundle_fingerprint: str,
) -> dict[str, Any]:
    structural_counts = _assert_structural_accounting(p30a_summary)
    acquisition = run_manifest["acquisition"]
    source_identity = run_manifest["source_snapshot"]
    summary = deepcopy(dict(p30a_summary))
    summary.update(
        {
            "schema_version": P33A_SCHEMA_VERSION,
            "operation": P33A_OPERATION,
            "run_id": run_manifest["run_fingerprint"],
            "run_fingerprint": run_manifest["run_fingerprint"],
            "target_date": run_manifest["target_date"],
            "analysis_run_id": p30a_summary["run_id"],
            "analysis_contract_version": P30A_SCHEMA_VERSION,
            "bundle_schema_version": P33A_BUNDLE_SCHEMA_VERSION,
            "bundle_fingerprint": bundle_fingerprint,
            "analysis_source_mode": "FROZEN_P32A_SOURCE_BUNDLE",
            "acquisition_completed": True,
            "source_records_received": int(acquisition["source_row_count"]),
            "official_target_game_count": int(
                acquisition["official_target_game_count"]
            ),
            "official_overlap_game_count": int(
                acquisition["official_overlap_game_count"]
            ),
            "observations_qualified": int(
                acquisition["qualified_observation_count"]
            ),
            "observations_rejected": int(acquisition["rejected_row_count"]),
            "analysis_terminal_state_counts": structural_counts,
            "official_raw_game_count": int(p30a_summary["raw_game_count"]),
            "frozen_tsl_source_snapshot_fingerprint": source_identity[
                "tsl_selected_rows_sha256"
            ],
            "frozen_tsl_normalized_snapshot_fingerprint": source_identity[
                "tsl_normalized_rows_sha256"
            ],
            "offline_bundle_replay_status": "NOT_RUN",
            "scheduler_activated": False,
            "betting_decision_generated": False,
            "staking_implemented": False,
            "profitability_claim": False,
            "real_betting_recommendation": False,
            "moneyline_model_promoted": True,
            "moneyline_promotion_scope": "paper_only",
            "run_line_migration_status": "BLOCKED_NO_PIT_SAFE_AUTHORITY",
            "total_migration_status": "BLOCKED_NO_PIT_SAFE_AUTHORITY",
            "legacy_decision_policy_status": "BLOCKED_NO_PIT_SAFE_AUTHORITY",
            "p20b_historical_runtime_compliance": "REMAINS_REFUTED",
        }
    )
    if summary["raw_game_count"] != sum(
        summary["analysis_terminal_state_counts"].values()
    ):
        raise RuntimeError(
            f"{STOP_P33A_MANDATORY_VERIFICATION}: P33A structural accounting drift"
        )
    return summary


def _acquisition_manifest(
    acquisition: TslMoneylineSnapshotAcquisition,
    *,
    capture_paths: Sequence[str],
) -> dict[str, Any]:
    rejected = acquisition.normalization.rejected_games
    return {
        "operation": acquisition.operation,
        "target_date": acquisition.target_date,
        "selection_started_at_utc": acquisition.selection_started_at_utc,
        "fetched_at_utc": acquisition.fetched_at_utc,
        "schedule_url": acquisition.schedule_url,
        "source_payload_sha256": dict(acquisition.source_payload_sha256),
        "source_row_count": acquisition.normalization.source_row_count,
        "qualified_observation_count": len(acquisition.history.observations),
        "rejected_row_count": len(rejected),
        "rejection_reason_counts": {
            reason: sum(item.reason == reason for item in rejected)
            for reason in sorted({item.reason for item in rejected})
        },
        "official_target_game_count": len(acquisition.target_schedule_rows),
        "official_overlap_game_count": len(acquisition.requested_game_ids),
        "qualified_non_overlap_count": max(
            0,
            len(acquisition.history.observations)
            - len(acquisition.requested_game_ids),
        ),
        "capture_paths": list(capture_paths),
        "truthful_acquisition_time": True,
        "canonical_history_mutated": False,
    }


def _assert_complete_target_coverage(
    acquisition: TslMoneylineSnapshotAcquisition,
) -> None:
    official_ids = {
        str(row["provider_game_id"]) for row in acquisition.target_schedule_rows
    }
    overlap_ids = {str(game_id) for game_id in acquisition.requested_game_ids}
    if len(official_ids) < 2 or len(overlap_ids) < 2 or not overlap_ids <= official_ids:
        raise RuntimeError(
            f"{STOP_P33A_MANDATORY_VERIFICATION}: official target coverage is unavailable"
        )
    if acquisition.normalization.source_row_count <= 0:
        raise RuntimeError(
            f"{STOP_P33A_MANDATORY_VERIFICATION}: no source records were received"
        )
    if not acquisition.history.observations:
        raise RuntimeError(
            f"{STOP_P33A_MANDATORY_VERIFICATION}: no qualified TSL observations"
        )


def _make_run_manifest(
    *,
    acquisition: TslMoneylineSnapshotAcquisition,
    source_manifest: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    model_fingerprint: str,
    capture_paths: Sequence[str],
    manifest_fingerprint: str,
) -> dict[str, Any]:
    target_membership = target_game_membership(acquisition.target_schedule_rows)
    identity_projection = {
        "operation": P33A_OPERATION,
        "schema_version": P33A_SCHEMA_VERSION,
        "target_date": acquisition.target_date,
        "target_game_membership": [dict(row) for row in target_membership],
        "source_snapshot": dict(source_identity),
        "model_id": P22B_MODEL_ID,
        "model_fingerprint": model_fingerprint,
        "analysis_contract_version": P30A_SCHEMA_VERSION,
        "source_manifest_fingerprint": manifest_fingerprint,
    }
    run_fingerprint = sha256(canonical_json_bytes(identity_projection)).hexdigest()
    return {
        "schema_version": P33A_SCHEMA_VERSION,
        "bundle_schema_version": P33A_BUNDLE_SCHEMA_VERSION,
        "operation": P33A_OPERATION,
        "run_id": run_fingerprint,
        "run_fingerprint": run_fingerprint,
        "target_date": acquisition.target_date,
        "target_game_ids": [
            str(row["provider_game_id"])
            for row in target_membership
        ],
        "target_game_membership": [dict(row) for row in target_membership],
        "source_manifest_fingerprint": manifest_fingerprint,
        "source_snapshot": deepcopy(dict(source_identity)),
        "model": {
            "model_id": P22B_MODEL_ID,
            "model_fingerprint": model_fingerprint,
            "promoted": True,
            "promotion_scope": "paper_only",
        },
        "analysis_contract_version": P30A_SCHEMA_VERSION,
        "acquisition": _acquisition_manifest(
            acquisition,
            capture_paths=capture_paths,
        ),
        "capture_paths": list(capture_paths),
        "source_authority": {
            "provider": "MLB_STATS_API",
            "domain": "mlb.com",
            "schedule_url": acquisition.schedule_url,
            "source_manifest_schema_version": source_manifest["schema_version"],
        },
        "frozen_before_analysis": True,
        "network_used_by_replay": False,
    }


def run_daily_moneyline_paper_analysis(
    *,
    repository_root: str | Path,
    runtime_root: str | Path = P33A_RUNTIME_ROOT,
    date_value: str | None = None,
    output_dir: str | Path | None = None,
    acquisition_factory: AcquisitionFactory | None = None,
    analysis_runner: AnalysisRunner | None = None,
) -> DailyMoneylinePaperRunResult:
    """Acquire once, freeze once, then compose the existing P30A analysis."""

    repository = Path(repository_root).resolve()
    runtime = _runtime_path(runtime_root, runtime_root=P33A_RUNTIME_ROOT)
    acquisition_runtime = runtime / "acquisition"
    factory = acquisition_factory or acquire_tsl_moneyline_snapshot
    acquisition = factory(
        repository_root=repository,
        runtime_root=acquisition_runtime,
        target_date=date_value,
    )
    _assert_complete_target_coverage(acquisition)

    p30a_inputs = build_p30a_paper_inputs(
        acquisition,
        repository_root=repository,
    )
    relative_captures = _relative_capture_paths(
        acquisition_runtime,
        acquisition.runtime_capture_paths,
    )
    frozen_manifest = portable_source_manifest(
        p30a_inputs.source_manifest,
        capture_paths=relative_captures,
    )
    frozen_manifest["target_game_ids"] = [
        str(row["provider_game_id"])
        for row in target_game_membership(acquisition.target_schedule_rows)
    ]
    source_identity = source_snapshot_identity(
        target_schedule_rows=acquisition.target_schedule_rows,
        tsl_rows=p30a_inputs.tsl_rows,
        tsl_raw_sha256=acquisition.history.raw_sha256,
        tsl_selected_rows_sha256=acquisition.history.selected_rows_sha256,
        source_payload_sha256=dict(acquisition.source_payload_sha256),
    )
    model_bytes, model_fingerprint = _load_p22b_model_bytes(repository)
    manifest_fingerprint = sha256(canonical_json_bytes(frozen_manifest)).hexdigest()
    run_manifest = _make_run_manifest(
        acquisition=acquisition,
        source_manifest=frozen_manifest,
        source_identity=source_identity,
        model_fingerprint=model_fingerprint,
        capture_paths=relative_captures,
        manifest_fingerprint=manifest_fingerprint,
    )
    bundle_root = (
        _runtime_path(output_dir, runtime_root=runtime)
        if output_dir is not None
        else _runtime_path(
            runtime / "bundles" / str(run_manifest["run_fingerprint"]),
            runtime_root=runtime,
        )
    )
    frozen_input_hashes = write_frozen_bundle_inputs(
        bundle_root,
        run_manifest=run_manifest,
        source_manifest=frozen_manifest,
        tsl_rows=p30a_inputs.tsl_rows,
        schedule_rows=p30a_inputs.schedule_rows,
        target_boxscore_rows=p30a_inputs.target_boxscore_rows,
        pitcher_game_log_rows=p30a_inputs.pitcher_game_log_rows,
        model_artifact_bytes=model_bytes,
        acquisition_runtime_root=acquisition_runtime,
        capture_paths=acquisition.runtime_capture_paths,
    )
    run_manifest = deepcopy(run_manifest)
    run_manifest["frozen_input_sha256"] = frozen_input_hashes
    write_json_object(
        bundle_root / "run_manifest.json",
        run_manifest,
        replace=True,
    )

    frozen_inputs = load_frozen_bundle_inputs(bundle_root)
    p30a_result = _run_p30a_from_frozen_inputs(
        repository_root=repository,
        inputs=frozen_inputs,
        analysis_runner=analysis_runner,
    )
    analysis_jsonl_sha256 = jsonl_fingerprint(p30a_result.analysis)
    bundle_fingerprint = _bundle_fingerprint(
        run_fingerprint=str(run_manifest["run_fingerprint"]),
        source_identity=source_identity,
        model_id=P22B_MODEL_ID,
        model_fingerprint=model_fingerprint,
        p30a_summary=p30a_result.summary,
        analysis_jsonl_sha256=analysis_jsonl_sha256,
    )
    summary = _build_summary(
        p30a_summary=p30a_result.summary,
        run_manifest=run_manifest,
        bundle_fingerprint=bundle_fingerprint,
    )
    output_hashes = write_run_outputs(
        bundle_root,
        analysis=p30a_result.analysis,
        summary=summary,
    )
    run_manifest = deepcopy(run_manifest)
    run_manifest["bundle_fingerprint"] = bundle_fingerprint
    run_manifest["analysis"] = {
        "analysis_jsonl_sha256": output_hashes["analysis.jsonl"],
        "summary_json_sha256": output_hashes["summary.json"],
        "p30a_run_id": p30a_result.summary["run_id"],
        "p30a_summary": deepcopy(dict(p30a_result.summary)),
    }
    manifest_without_fingerprint = {
        key: value
        for key, value in run_manifest.items()
        if key != "run_manifest_fingerprint"
    }
    run_manifest["run_manifest_fingerprint"] = sha256(
        canonical_json_bytes(manifest_without_fingerprint)
    ).hexdigest()
    write_json_object(
        bundle_root / "run_manifest.json",
        run_manifest,
        replace=True,
    )
    return DailyMoneylinePaperRunResult(
        bundle_root=bundle_root,
        analysis=tuple(dict(row) for row in p30a_result.analysis),
        summary=summary,
        run_manifest=run_manifest,
        offline_replay_equal=False,
    )


def replay_daily_moneyline_paper_analysis(
    *,
    repository_root: str | Path,
    bundle_path: str | Path,
    output_dir: str | Path | None = None,
    runtime_root: str | Path = P33A_RUNTIME_ROOT,
    analysis_runner: AnalysisRunner | None = None,
) -> DailyMoneylinePaperRunResult:
    """Replay one frozen bundle using file inputs only and no acquisition call."""

    repository = Path(repository_root).resolve()
    runtime = _runtime_path(runtime_root, runtime_root=P33A_RUNTIME_ROOT)
    bundle_root = _runtime_path(bundle_path, runtime_root=runtime)
    inputs = load_frozen_bundle_inputs(bundle_root)
    _assert_bundle_model_identity(
        repository_root=repository,
        bundle_root=bundle_root,
    )
    p30a_result = _run_p30a_from_frozen_inputs(
        repository_root=repository,
        inputs=inputs,
        analysis_runner=analysis_runner,
    )
    expected_analysis = read_jsonl_objects(bundle_root / "analysis.jsonl")
    expected_summary = read_json_object(bundle_root / "summary.json")
    actual_analysis = tuple(dict(row) for row in p30a_result.analysis)
    if actual_analysis != expected_analysis:
        raise RuntimeError(
            f"{STOP_P33A_MANDATORY_VERIFICATION}: offline analysis differs"
        )
    bundle_fingerprint = str(inputs.run_manifest["bundle_fingerprint"])
    actual_summary = _build_summary(
        p30a_summary=p30a_result.summary,
        run_manifest=inputs.run_manifest,
        bundle_fingerprint=bundle_fingerprint,
    )
    if actual_summary != expected_summary:
        raise RuntimeError(
            f"{STOP_P33A_MANDATORY_VERIFICATION}: offline summary differs"
        )

    destination = (
        _runtime_path(output_dir, runtime_root=runtime)
        if output_dir is not None
        else _runtime_path(
            runtime / "replay" / str(inputs.run_manifest["run_fingerprint"]),
            runtime_root=runtime,
        )
    )
    output_hashes = write_run_outputs(
        destination,
        analysis=actual_analysis,
        summary=actual_summary,
    )
    write_json_object(
        destination / "run_manifest.json",
        inputs.run_manifest,
    )
    replay_manifest = {
        "schema_version": P33A_SCHEMA_VERSION,
        "operation": P33A_OPERATION,
        "run_fingerprint": inputs.run_manifest["run_fingerprint"],
        "source_bundle_fingerprint": bundle_fingerprint,
        "network_guard": "PASS",
        "analysis_equal": True,
        "summary_equal": True,
        "analysis_jsonl_sha256": output_hashes["analysis.jsonl"],
        "summary_json_sha256": output_hashes["summary.json"],
    }
    write_json_object(destination / "replay_manifest.json", replay_manifest)
    return DailyMoneylinePaperRunResult(
        bundle_root=destination,
        analysis=actual_analysis,
        summary=actual_summary,
        run_manifest=inputs.run_manifest,
        offline_replay_equal=True,
    )


__all__ = (
    "DailyMoneylinePaperRunResult",
    "P30A_SCHEMA_VERSION",
    "P33A_BUNDLE_SCHEMA_VERSION",
    "P33A_OPERATION",
    "P33A_RUNTIME_ROOT",
    "P33A_SCHEMA_VERSION",
    "STOP_P33A_JUDGE_NOT_VERIFIED",
    "STOP_P33A_MANDATORY_VERIFICATION",
    "STOP_P33A_UNEXPECTED_RUNTIME_WRITE",
    "replay_daily_moneyline_paper_analysis",
    "run_daily_moneyline_paper_analysis",
)
