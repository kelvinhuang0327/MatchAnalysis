"""Frozen input and artifact helpers for the P33A daily paper run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .paper_moneyline_batch_artifacts import (
    canonical_json_bytes,
    render_jsonl,
    sha256_bytes,
)


P33A_BUNDLE_SCHEMA_VERSION = "p33a.daily_moneyline_paper_run_bundle.v1"


@dataclass(frozen=True, slots=True)
class FrozenMoneylinePaperRunInputs:
    """All source inputs needed to invoke P30A without acquisition."""

    bundle_root: Path
    run_manifest: dict[str, Any]
    source_manifest: dict[str, Any]
    tsl_rows: tuple[dict[str, Any], ...]
    schedule_rows: tuple[dict[str, Any], ...]
    target_boxscore_rows: tuple[dict[str, Any], ...]
    pitcher_game_log_rows: tuple[dict[str, Any], ...]


def jsonl_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint the exact canonical JSONL projection used by a bundle."""

    return sha256_bytes(render_jsonl(rows))


def target_game_membership(
    schedule_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Project canonical game identity without status, scores, or results."""

    membership = []
    for row in schedule_rows:
        membership.append(
            {
                "provider_game_id": str(row["provider_game_id"]),
                "game_number": int(row["game_number"]),
                "scheduled_start_utc": str(row["scheduled_start_utc"]),
                "official_date": str(row["official_date"]),
                "home_team": deepcopy(dict(row["home_team"])),
                "away_team": deepcopy(dict(row["away_team"])),
            }
        )
    return tuple(
        sorted(
            membership,
            key=lambda row: (
                row["scheduled_start_utc"],
                row["game_number"],
                row["provider_game_id"],
            ),
        )
    )


def source_snapshot_identity(
    *,
    target_schedule_rows: Sequence[Mapping[str, Any]],
    tsl_rows: Sequence[Mapping[str, Any]],
    tsl_raw_sha256: str,
    tsl_selected_rows_sha256: str,
    source_payload_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Return stable source identity fields for the semantic run fingerprint."""

    return {
        "official_target_membership": [
            dict(row) for row in target_game_membership(target_schedule_rows)
        ],
        "official_target_game_count": len(target_schedule_rows),
        "tsl_payload_sha256": dict(sorted(source_payload_sha256.items())),
        "tsl_raw_capture_sha256": tsl_raw_sha256,
        "tsl_selected_rows_sha256": tsl_selected_rows_sha256,
        "tsl_normalized_rows_sha256": jsonl_fingerprint(tsl_rows),
    }


def portable_source_manifest(
    source_manifest: Mapping[str, Any],
    *,
    capture_paths: Sequence[str],
) -> dict[str, Any]:
    """Replace live absolute capture paths with bundle-relative paths."""

    manifest = deepcopy(dict(source_manifest))
    acquisition = manifest.get("p32a_tsl_acquisition")
    if not isinstance(acquisition, Mapping):
        raise ValueError("P32A acquisition metadata is missing from source manifest")
    acquisition_copy = deepcopy(dict(acquisition))
    acquisition_copy["runtime_capture_paths"] = list(capture_paths)
    manifest["p32a_tsl_acquisition"] = acquisition_copy
    return manifest


def _write_bytes(path: Path, raw: bytes, *, replace: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        existing = path.read_bytes()
        if existing != raw:
            raise RuntimeError(f"frozen bundle path already differs: {path}")
    else:
        path.write_bytes(raw)
    return sha256_bytes(raw)


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> str:
    raw = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return _write_bytes(path, raw, replace=replace)


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    replace: bool = False,
) -> str:
    return _write_bytes(path, render_jsonl(rows), replace=replace)


def read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_object(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    replace: bool = False,
) -> str:
    """Write one deterministic JSON object."""

    return _write_json(Path(path), value, replace=replace)


def read_jsonl_objects(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise ValueError(f"blank JSONL row at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return tuple(rows)


def _capture_files(
    *,
    bundle_root: Path,
    acquisition_runtime_root: Path,
    capture_paths: Sequence[str],
) -> dict[str, str]:
    capture_hashes: dict[str, str] = {}
    for capture_path in capture_paths:
        source = Path(capture_path)
        try:
            relative = source.relative_to(acquisition_runtime_root)
        except ValueError as exc:
            raise RuntimeError("P32A capture escaped its authorized runtime root") from exc
        if not relative.parts or relative.parts[0] != "capture":
            raise RuntimeError("P32A capture is outside the capture directory")
        raw = source.read_bytes()
        relative_name = relative.as_posix()
        capture_hashes[relative_name] = _write_bytes(
            bundle_root / relative,
            raw,
        )
    return capture_hashes


def write_frozen_bundle_inputs(
    bundle_root: str | Path,
    *,
    run_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    tsl_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
    target_boxscore_rows: Sequence[Mapping[str, Any]],
    pitcher_game_log_rows: Sequence[Mapping[str, Any]],
    model_artifact_bytes: bytes,
    acquisition_runtime_root: str | Path,
    capture_paths: Sequence[str],
) -> dict[str, str]:
    """Materialize one self-contained, immutable input projection."""

    root = Path(bundle_root)
    root.mkdir(parents=True, exist_ok=True)
    capture_hashes = _capture_files(
        bundle_root=root,
        acquisition_runtime_root=Path(acquisition_runtime_root),
        capture_paths=capture_paths,
    )
    hashes = {
        "source_manifest.json": _write_json(
            root / "source_manifest.json", source_manifest
        ),
        "tsl_source_snapshot.jsonl": _write_jsonl(
            root / "tsl_source_snapshot.jsonl", tsl_rows
        ),
        "mlb_source_snapshot.jsonl": _write_jsonl(
            root / "mlb_source_snapshot.jsonl", schedule_rows
        ),
        "target_boxscores.jsonl": _write_jsonl(
            root / "target_boxscores.jsonl", target_boxscore_rows
        ),
        "pitcher_game_logs.jsonl": _write_jsonl(
            root / "pitcher_game_logs.jsonl", pitcher_game_log_rows
        ),
        "model_artifact.json": _write_bytes(
            root / "model_artifact.json", model_artifact_bytes
        ),
    }
    hashes.update({f"{key}": value for key, value in capture_hashes.items()})
    _write_json(root / "run_manifest.json", run_manifest)
    return hashes


def load_frozen_bundle_inputs(
    bundle_root: str | Path,
) -> FrozenMoneylinePaperRunInputs:
    """Load and verify only the frozen inputs; this function has no network path."""

    root = Path(bundle_root).resolve()
    run_manifest = read_json_object(root / "run_manifest.json")
    if run_manifest.get("bundle_schema_version") != P33A_BUNDLE_SCHEMA_VERSION:
        raise ValueError("P33A bundle schema mismatch")
    source_manifest = read_json_object(root / "source_manifest.json")
    if source_manifest.get("schema_version") is None:
        raise ValueError("P33A source manifest schema is missing")
    declared_manifest_fingerprint = run_manifest.get("source_manifest_fingerprint")
    if declared_manifest_fingerprint != sha256_bytes(
        canonical_json_bytes(source_manifest)
    ):
        raise ValueError("P33A source manifest fingerprint mismatch")
    frozen_hashes = run_manifest.get("frozen_input_sha256")
    if not isinstance(frozen_hashes, Mapping):
        raise ValueError("P33A frozen input hashes are missing")
    for relative_name, declared_hash in frozen_hashes.items():
        relative_path = Path(str(relative_name))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("P33A frozen input path escaped bundle")
        input_path = root / relative_path
        if not input_path.is_file() or sha256_bytes(input_path.read_bytes()) != declared_hash:
            raise ValueError(f"P33A frozen input fingerprint mismatch: {relative_name}")
    tsl_rows = read_jsonl_objects(root / "tsl_source_snapshot.jsonl")
    schedule_rows = read_jsonl_objects(root / "mlb_source_snapshot.jsonl")
    target_boxscore_rows = read_jsonl_objects(root / "target_boxscores.jsonl")
    pitcher_game_log_rows = read_jsonl_objects(root / "pitcher_game_logs.jsonl")

    source_identity = run_manifest.get("source_snapshot")
    if not isinstance(source_identity, Mapping):
        raise ValueError("P33A source snapshot identity is missing")
    if source_identity.get("tsl_normalized_rows_sha256") != jsonl_fingerprint(tsl_rows):
        raise ValueError("P33A TSL source snapshot fingerprint mismatch")
    if source_identity.get("official_target_membership") != [
        dict(row)
        for row in target_game_membership(
            [
                row
                for row in schedule_rows
                if str(row.get("provider_game_id"))
                in {str(game_id) for game_id in run_manifest.get("target_game_ids", [])}
            ]
        )
    ]:
        raise ValueError("P33A official target membership mismatch")
    return FrozenMoneylinePaperRunInputs(
        bundle_root=root,
        run_manifest=run_manifest,
        source_manifest=source_manifest,
        tsl_rows=tsl_rows,
        schedule_rows=schedule_rows,
        target_boxscore_rows=target_boxscore_rows,
        pitcher_game_log_rows=pitcher_game_log_rows,
    )


def write_run_outputs(
    bundle_root: str | Path,
    *,
    analysis: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    replace: bool = False,
) -> dict[str, str]:
    """Write deterministic P33A analysis and summary artifacts."""

    root = Path(bundle_root)
    return {
        "analysis.jsonl": _write_jsonl(
            root / "analysis.jsonl", analysis, replace=replace
        ),
        "summary.json": _write_json(
            root / "summary.json", summary, replace=replace
        ),
    }


__all__ = (
    "FrozenMoneylinePaperRunInputs",
    "P33A_BUNDLE_SCHEMA_VERSION",
    "jsonl_fingerprint",
    "load_frozen_bundle_inputs",
    "portable_source_manifest",
    "read_json_object",
    "read_jsonl_objects",
    "source_snapshot_identity",
    "target_game_membership",
    "write_frozen_bundle_inputs",
    "write_json_object",
    "write_run_outputs",
)
