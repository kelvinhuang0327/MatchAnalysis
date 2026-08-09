"""Deterministic P19A Moneyline fixture, input, and report serialization."""

from collections.abc import Mapping
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...baseball.domain.moneyline_feature_snapshot import (
    MoneylineFeatureProvenance,
    MoneylineFeatureSnapshot,
)
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
from ...baseball.domain.prediction_admission import (
    PredictionAdmissionResult,
    ProspectivePredictionCandidate,
)
from ...core.identity import MatchIdentity
from .generate_moneyline_predictions import MoneylineInferenceResult


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_moneyline_model_artifact(path: Path) -> MoneylineModelArtifact:
    """Load one explicit model artifact; no training or network access occurs."""

    return MoneylineModelArtifact.from_projection(_read_json(path))


def _load_identity(projection: Mapping[str, Any]) -> MatchIdentity:
    return MatchIdentity(
        sport=str(projection["sport"]),
        league=str(projection["league"]),
        season=int(projection["season"]),
        canonical_game_id=str(projection["canonical_game_id"]),
        home_participant=str(projection["home_participant"]),
        away_participant=str(projection["away_participant"]),
        game_discriminator=projection.get("game_discriminator"),
    )


def _load_provenance(
    values: list[Mapping[str, Any]],
) -> tuple[MoneylineFeatureProvenance, ...]:
    return tuple(
        MoneylineFeatureProvenance(
            field_name=str(item["field_name"]),
            source_id=str(item["source_id"]),
            source_kind=str(item["source_kind"]),
            observed_as_of_utc=parse_canonical_utc(
                str(item["observed_as_of_utc"])
            ),
            source_fingerprint=str(item["source_fingerprint"]),
        )
        for item in values
    )


def load_moneyline_feature_snapshots(path: Path) -> tuple[MoneylineFeatureSnapshot, ...]:
    """Load deterministic JSONL snapshots and fail closed on malformed rows."""

    snapshots: list[MoneylineFeatureSnapshot] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            projection = json.loads(line)
            identity_projection = projection["identity"]
            snapshots.append(
                MoneylineFeatureSnapshot.from_record(
                    projection["features"],
                    identity=_load_identity(identity_projection),
                    provider_namespace=str(projection["provider_namespace"]),
                    provider_game_id=str(projection["provider_game_id"]),
                    game_number=int(projection["game_number"]),
                    source_schedule_observation_id=str(
                        projection["source_schedule_observation_id"]
                    ),
                    as_of_utc=parse_canonical_utc(str(projection["as_of_utc"])),
                    scheduled_start_utc=parse_canonical_utc(
                        str(projection["scheduled_start_utc"])
                    ),
                    feature_provenance=_load_provenance(
                        projection["feature_provenance"]
                    ),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid Moneyline feature snapshot at line {line_number}"
            ) from exc
    return tuple(snapshots)


def candidate_to_dict(candidate: ProspectivePredictionCandidate) -> dict[str, Any]:
    return {
        "prediction_observation_id": candidate.prediction_observation_id,
        "source_prediction_id": candidate.source_prediction_id,
        "model_id": candidate.model_id,
        "market_id": candidate.market_id,
        "selection": candidate.selection,
        "model_probability": str(candidate.model_probability),
        "line_value": str(candidate.line_value),
        "push_policy": candidate.push_policy,
        "provider_namespace": candidate.provider_namespace,
        "provider_game_id": candidate.provider_game_id,
        "game_number": candidate.game_number,
        "source_schedule_observation_id": candidate.source_schedule_observation_id,
        "prediction_generated_at_utc": candidate.prediction_generated_at_utc,
        "response_received_at_utc": candidate.response_received_at_utc,
        "ingested_at_utc": candidate.ingested_at_utc,
    }


def admission_to_dict(
    result: PredictionAdmissionResult,
    index: int,
) -> dict[str, Any]:
    return {
        "request_index": index,
        "admission_status": result.admission_status,
        "reason": result.reason,
        "prediction_observation_id": (
            result.observation.prediction_observation_id
            if result.observation
            else None
        ),
    }


def render_predictions_jsonl(result: MoneylineInferenceResult) -> str:
    lines = [
        json.dumps(
            candidate_to_dict(candidate),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for candidate in result.candidates
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def render_admissions_jsonl(result: MoneylineInferenceResult) -> str:
    lines = [
        json.dumps(
            admission_to_dict(admission, index),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for index, admission in enumerate(result.admissions, start=1)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def render_summary_json(
    result: MoneylineInferenceResult,
    model_artifact: MoneylineModelArtifact,
) -> str:
    admitted_count = sum(
        admission.admission_status == "ADMITTED" for admission in result.admissions
    )
    rejected_count = sum(
        admission.admission_status == "REJECTED" for admission in result.admissions
    )
    payload = {
        "schema_version": "p19a.moneyline_inference_summary.v1",
        "candidate_count": len(result.candidates),
        "admission_count": len(result.admissions),
        "admitted_count": admitted_count,
        "rejected_count": rejected_count,
        "candidate_set_fingerprint": result.candidate_set_fingerprint,
        "model_artifact_fingerprint": result.model_artifact_fingerprint,
        "model_artifact": model_artifact.to_projection(),
        "claims": {
            "market_id": "moneyline",
            "provider_called": False,
            "db_written": False,
            "training_performed": False,
            "production_claim": False,
            "betting_claim": False,
            "profitability_claim": False,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_report_markdown(
    result: MoneylineInferenceResult,
    model_artifact: MoneylineModelArtifact,
) -> str:
    lines = [
        "# P19A Deterministic Moneyline Inference",
        "",
        "This artifact is a bounded, paper-only, diagnostic inference slice.",
        "",
        "## Result",
        "",
        f"- Candidate count: `{len(result.candidates)}`",
        f"- Admission count: `{len(result.admissions)}`",
        f"- Candidate-set fingerprint: `{result.candidate_set_fingerprint}`",
        f"- Model-artifact fingerprint: `{result.model_artifact_fingerprint}`",
        "",
        "## Legacy provenance",
        "",
        f"- Repository: `{model_artifact.legacy_source_repository}`",
        f"- Commit: `{model_artifact.legacy_source_commit}`",
        f"- Tree: `{model_artifact.legacy_source_tree}`",
    ]
    lines.extend(f"- Path: `{path}`" for path in model_artifact.legacy_source_paths)
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Market: `moneyline` only.",
            "- Features: P13 independent recent-form and starter-ERA deltas only.",
            "- No final score, result, settlement, odds join, provider call, database write, training, or profitability claim.",
            "- The artifact does not claim broad historical parity or production accuracy.",
            "",
        ]
    )
    return "\n".join(lines)


def write_moneyline_inference_artifacts(
    output_dir: Path,
    result: MoneylineInferenceResult,
    model_artifact: MoneylineModelArtifact,
) -> None:
    """Write deterministic P19A report artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.jsonl").write_text(
        render_predictions_jsonl(result),
        encoding="utf-8",
    )
    (output_dir / "admissions.jsonl").write_text(
        render_admissions_jsonl(result),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        render_summary_json(result, model_artifact),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_report_markdown(result, model_artifact),
        encoding="utf-8",
    )
