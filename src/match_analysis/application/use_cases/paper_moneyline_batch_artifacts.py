"""Deterministic P24C paper Moneyline batch artifact helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact


P24C_BATCH_SCHEMA_VERSION = "p24c.promoted_moneyline_shadow_batch.v1"
P24C_SOURCE_MANIFEST_SCHEMA_VERSION = "p24c.source_manifest.v1"
P22B_MODEL_ID = "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630"
P22B_ARTIFACT_FINGERPRINT = (
    "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e"
)
P22B_ARTIFACT_RELATIVE_PATH = Path(
    "report/p22b_moneyline_challenger/model_artifact.json"
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def render_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def default_paper_moneyline_model_artifact_path(repository_root: str | Path) -> Path:
    return Path(repository_root) / P22B_ARTIFACT_RELATIVE_PATH


def _load_projection(path: str | Path) -> dict[str, Any]:
    projection = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(projection, dict):
        raise ValueError(f"model artifact must be an object: {path}")
    return projection


def load_model_artifact_with_fingerprint(
    path: str | Path,
) -> tuple[MoneylineModelArtifact, str]:
    """Load one inference artifact and its full P22B projection fingerprint."""

    projection = _load_projection(path)
    artifact = MoneylineModelArtifact.from_projection(projection)
    declared = projection.get("artifact_fingerprint")
    if declared is None:
        return artifact, artifact.fingerprint()
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError(f"invalid artifact fingerprint: {path}")
    expected = sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in projection.items()
                if key != "artifact_fingerprint"
            }
        )
    )
    if declared != expected:
        raise ValueError(f"artifact fingerprint mismatch: {path}")
    return artifact, declared


def load_default_paper_moneyline_model_artifact(
    repository_root: str | Path,
) -> tuple[MoneylineModelArtifact, str]:
    """Resolve the same frozen default artifact used by the paper CLI."""

    return load_model_artifact_with_fingerprint(
        default_paper_moneyline_model_artifact_path(repository_root)
    )


def write_paper_moneyline_batch_artifacts(
    output_dir: str | Path,
    *,
    predictions: Iterable[Mapping[str, Any]],
    feature_unavailable: Iterable[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    """Write only the four deterministic P24C report artifacts."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "predictions.jsonl": render_jsonl(predictions),
        "feature_unavailable.jsonl": render_jsonl(feature_unavailable),
        "source_manifest.json": (
            json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
        "summary.json": (
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    return {name: sha256_bytes(content) for name, content in files.items()}


__all__ = (
    "P22B_ARTIFACT_FINGERPRINT",
    "P22B_ARTIFACT_RELATIVE_PATH",
    "P22B_MODEL_ID",
    "P24C_BATCH_SCHEMA_VERSION",
    "P24C_SOURCE_MANIFEST_SCHEMA_VERSION",
    "canonical_json_bytes",
    "default_paper_moneyline_model_artifact_path",
    "load_default_paper_moneyline_model_artifact",
    "load_model_artifact_with_fingerprint",
    "render_jsonl",
    "sha256_bytes",
    "write_paper_moneyline_batch_artifacts",
)
