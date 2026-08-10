"""Deterministic JSON/JSONL artifacts for the P23F2 future fold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...baseball.domain.future_evaluation_fold import FutureEvaluationFold


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def render_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def render_source_manifest(*, records: Iterable[Mapping[str, Any]], normalized_hashes: Mapping[str, str]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: (row["path"], row["url"]))
    return {
        "schema_version": "p23f2.source_manifest.v1",
        "source_domains": ["mlb.com"],
        "records": ordered,
        "normalized_hashes": dict(sorted(normalized_hashes.items())),
    }


def write_future_fold_artifacts(
    output_dir: str | Path,
    fold: FutureEvaluationFold,
    *,
    source_manifest: Mapping[str, Any],
    offline_replay_verified: bool = False,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = fold.manifest_projection() | {"fold_fingerprint": fold.fold_fingerprint}
    feature_bytes = render_jsonl(row.projection() for row in fold.feature_rows)
    result_bytes = render_jsonl(row.projection() for row in fold.result_rows)
    source_bytes = json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    summary = {
        **manifest,
        "source_manifest_fingerprint": fold.source_manifest_fingerprint,
        "offline_replay_verified": offline_replay_verified,
    }
    files = {
        "fold_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "feature_rows.jsonl": feature_bytes,
        "results.jsonl": result_bytes,
        "source_manifest.json": source_bytes,
        "summary.json": json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    from hashlib import sha256

    return {name: sha256(content).hexdigest() for name, content in files.items()}


__all__ = ("render_jsonl", "render_source_manifest", "write_future_fold_artifacts")
