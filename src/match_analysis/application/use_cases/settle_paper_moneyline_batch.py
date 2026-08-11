"""Settle the committed P24C paper Moneyline batch through P16A-P17A.

This use case is deliberately offline.  It freezes the committed P24C
prediction authority first, adapts those rows into the existing P15C snapshot
shape, attaches final results from the committed P24C normalized schedule,
then delegates correctness, Brier, and feedback semantics to the existing
P16A, P16B, and P17A contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.final_result_observation import (
    FinalResultObservation,
    load_final_result_observations,
)
from .admitted_prediction_observation_artifacts import (
    render_admitted_observations_jsonl,
    render_snapshot_report_markdown,
    render_snapshot_summary_json,
)
from .attach_final_results_to_admitted_predictions import (
    FinalResultAttachmentResult,
    attach_final_results_to_admitted_predictions,
)
from .build_admitted_prediction_observation_snapshot import (
    AdmittedPredictionObservationSnapshotResult,
    build_admitted_prediction_observation_snapshot,
)
from .build_prediction_evaluation_scorecard import (
    PredictionEvaluationScorecardResult,
    build_prediction_evaluation_scorecard,
)
from .build_prediction_feedback_ledger import (
    PredictionFeedbackLedgerResult,
    build_prediction_feedback_ledger,
)
from .final_result_attachment_artifacts import (
    render_attachment_report_markdown,
    render_attachment_summary_json,
    render_attachments_jsonl,
)
from .paper_moneyline_batch_artifacts import (
    load_model_artifact_with_fingerprint,
)
from .prediction_evaluation_artifacts import (
    render_evaluation_report_markdown,
    render_evaluation_summary_json,
    render_evaluations_jsonl,
)
from .prediction_feedback_artifacts import render_feedback_jsonl


P25A_SCHEMA_VERSION = "p25a.promoted_moneyline_paper_feedback.v1"
P25A_SETTLED_ROW_SCHEMA_VERSION = "p25a.settled_prediction.v1"
P24C_SCHEMA_VERSION = "p24c.promoted_moneyline_shadow_batch.v1"
P24C_BATCH_ID = "a43aa88cef4df6a3acac7a2fbdf04f6bad0b3a44c6b4eb96a3538bbc0953264c"
P24C_PREDICTION_FINGERPRINT = (
    "fa3f94a29340ef26b3deee9fa8865aecf82f293afbe19280883606146e1d2c10"
)
P24C_SOURCE_MANIFEST_FINGERPRINT = (
    "73250d70ecfadea996b7e5a3e388f75601420fce222cd32a8c4a7ccb729165de"
)
P24C_MODEL_ID = "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630"
P24C_MODEL_FINGERPRINT = (
    "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e"
)
P24C_FEATURE_UNAVAILABLE_REASON = "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY"
P24C_RAW_GAME_COUNT = 90
P24C_PREDICTION_COUNT = 79
P24C_FEATURE_UNAVAILABLE_COUNT = 11
P24C_SOURCE_AUTHORITY = "MLB_STATS_API"
P24C_NORMALIZED_ROOT = Path(
    "data/fixtures/p24c_promoted_moneyline_shadow_batch/normalized"
)
P24C_REPORT_ROOT = Path("report/p24c_promoted_moneyline_shadow_batch")
P25A_STOP_BASELINE_DRIFT = "STOP_MATCHANALYSIS_P25A_BASELINE_DRIFT"
P25A_STOP_PREDICTION_AUTHORITY_DRIFT = (
    "STOP_MATCHANALYSIS_P25A_PREDICTION_AUTHORITY_DRIFT"
)
P25A_STOP_RESULT_PROVENANCE_UNRESOLVED = (
    "STOP_MATCHANALYSIS_P25A_RESULT_PROVENANCE_UNRESOLVED"
)
P25A_STOP_SETTLEMENT_COUNT_MISMATCH = (
    "STOP_MATCHANALYSIS_P25A_SETTLEMENT_COUNT_MISMATCH"
)
P25A_STOP_FEEDBACK_LINEAGE_MISMATCH = (
    "STOP_MATCHANALYSIS_P25A_FEEDBACK_LINEAGE_MISMATCH"
)
P25A_STOP_PRIOR_AUTHORITY_DRIFT = "STOP_MATCHANALYSIS_P25A_PRIOR_AUTHORITY_DRIFT"


@dataclass(frozen=True, slots=True)
class P24CPredictionAuthority:
    """Frozen committed P24C prediction and abstention authority."""

    batch_id: str
    prediction_fingerprint: str
    source_manifest_fingerprint: str
    raw_game_count: int
    predictions: tuple[dict[str, Any], ...]
    feature_unavailable: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PaperMoneylineSettlementResult:
    """Complete deterministic P25A settlement/evaluation/feedback result."""

    authority: P24CPredictionAuthority
    result_authority_fingerprint: str
    result_authority_summary: dict[str, Any]
    snapshot_result: AdmittedPredictionObservationSnapshotResult
    attachment_result: FinalResultAttachmentResult
    evaluation_result: PredictionEvaluationScorecardResult
    feedback_result: PredictionFeedbackLedgerResult
    settled_predictions: tuple[dict[str, Any], ...]
    claims: dict[str, Any]

    @property
    def accuracy(self) -> float:
        return self.evaluation_result.accuracy

    @property
    def mean_brier(self) -> float:
        return self.evaluation_result.brier_score


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(dict(row)) for row in rows)


def _duplicate_rejecting_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_object(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_duplicate_rejecting_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_duplicate_rejecting_pairs,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} malformed JSON on line {index}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {index} must be an object")
        rows.append(value)
    return rows


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _parse_utc(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    normalized = parsed.astimezone(UTC)
    if normalized != parsed:
        raise ValueError(f"{field_name} must be normalized to UTC")
    return normalized


def _copy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [json.loads(_canonical_json(dict(row))) for row in rows]


def _sort_rows(
    rows: Sequence[Mapping[str, Any]],
    *keys: str,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _copy_rows(
            sorted(
                rows,
                key=lambda row: tuple(str(row.get(key, "")) for key in keys),
            )
        )
    )


def _assert_same_authority_rows(
    *,
    candidate: Sequence[Mapping[str, Any]],
    committed: Sequence[Mapping[str, Any]],
    identity_keys: tuple[str, ...],
    label: str,
) -> None:
    candidate_sorted = _sort_rows(candidate, *identity_keys)
    committed_sorted = _sort_rows(committed, *identity_keys)
    if _canonical_jsonl(candidate_sorted) != _canonical_jsonl(committed_sorted):
        raise ValueError(f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: {label}")


def _read_bytes(root: Path, relative: Path, override: bytes | None) -> bytes:
    return override if override is not None else (root / relative).read_bytes()


def _validate_p24c_summary(summary: Mapping[str, Any]) -> None:
    expected = {
        "batch_id": P24C_BATCH_ID,
        "prediction_set_fingerprint": P24C_PREDICTION_FINGERPRINT,
        "source_manifest_fingerprint": P24C_SOURCE_MANIFEST_FINGERPRINT,
        "raw_game_count": P24C_RAW_GAME_COUNT,
        "evaluable_game_count": P24C_PREDICTION_COUNT,
        "feature_unavailable_count": P24C_FEATURE_UNAVAILABLE_COUNT,
        "promoted_default_model_id": P24C_MODEL_ID,
        "promoted_default_model_fingerprint": P24C_MODEL_FINGERPRINT,
        "schema_version": P24C_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: "
                f"P24C summary {key} mismatch"
            )
    for key in (
        "default_explicit_equivalence_verified",
        "explicit_incumbent_override_verified",
        "result_mutation_isolation_verified",
        "offline_replay_verified",
        "historical_shadow",
        "paper_only",
        "model_promoted",
    ):
        if summary.get(key) is not True:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: P24C {key} is not true"
            )
    for key in (
        "challenger_retrained",
        "production_ready",
        "deployment_performed",
        "real_betting_recommendation",
        "profitability_claim",
    ):
        if summary.get(key) is not False:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: P24C {key} is not false"
            )
    if summary.get("promotion_scope") != "paper_only":
        raise ValueError(
            f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: promotion scope changed"
        )


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
) -> None:
    required = (
        "prediction_id",
        "source_prediction_id",
        "batch_id",
        "game_id",
        "scheduled_start",
        "source_provider_game_id",
        "home_team",
        "away_team",
        "model_id",
        "model_fingerprint",
        "feature_snapshot_id",
        "feature_snapshot_fingerprint",
        "home_win_probability",
        "away_win_probability",
        "predicted_side",
    )
    seen_prediction_ids: set[str] = set()
    seen_game_ids: set[str] = set()
    forbidden_outcome_fields = {
        "home_score",
        "away_score",
        "outcome",
        "result",
        "runs",
        "winner",
        "actual_winner",
    }
    for index, row in enumerate(rows, start=1):
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: "
                f"prediction row {index} missing {missing}"
            )
        if forbidden_outcome_fields.intersection(row):
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: "
                f"prediction row {index} contains outcome data"
            )
        prediction_id = row["prediction_id"]
        game_id = row["game_id"]
        if not _is_sha256(prediction_id) or prediction_id in seen_prediction_ids:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: duplicate/invalid prediction id"
            )
        if not isinstance(game_id, str) or game_id in seen_game_ids:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: duplicate/invalid game id"
            )
        seen_prediction_ids.add(prediction_id)
        seen_game_ids.add(game_id)
        if row["batch_id"] != summary["batch_id"]:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: prediction batch mismatch"
            )
        if row["model_id"] != P24C_MODEL_ID or row["model_fingerprint"] != P24C_MODEL_FINGERPRINT:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: model identity changed"
            )
        if not _is_sha256(row["feature_snapshot_id"]) or not _is_sha256(
            row["feature_snapshot_fingerprint"]
        ):
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: feature identity invalid"
            )
        if row["feature_snapshot_id"] != row["feature_snapshot_fingerprint"]:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: feature identity diverged"
            )
        if row["predicted_side"] not in ("HOME", "AWAY"):
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: predicted side invalid"
            )
        try:
            home_probability = Decimal(str(row["home_win_probability"]))
            away_probability = Decimal(str(row["away_win_probability"]))
        except (ArithmeticError, ValueError) as exc:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: probability invalid"
            ) from exc
        if not (
            Decimal("0") <= home_probability <= Decimal("1")
            and Decimal("0") <= away_probability <= Decimal("1")
        ):
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: probability out of range"
            )
        if home_probability + away_probability != Decimal("1"):
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: probabilities do not sum to one"
            )
        _parse_utc(row["scheduled_start"], "scheduled_start")
        if game_id != f"{row['source_provider_game_id']}@{row['scheduled_start']}":
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: game identity mismatch"
            )


def _validate_unavailable_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
) -> None:
    seen_games: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if row.get("batch_id") != summary["batch_id"]:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: abstention batch mismatch"
            )
        if row.get("eligibility") != "FEATURE_UNAVAILABLE" or row.get("status") != "FEATURE_UNAVAILABLE":
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: abstention status changed"
            )
        if row.get("reason") != P24C_FEATURE_UNAVAILABLE_REASON:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: abstention reason changed"
            )
        game_id = row.get("game_id")
        if not isinstance(game_id, str) or game_id in seen_games:
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: duplicate abstention game"
            )
        seen_games.add(game_id)
        _parse_utc(row.get("scheduled_start"), f"abstention {index} scheduled_start")
        if {"home_score", "away_score", "result", "outcome", "winner"}.intersection(row):
            raise ValueError(
                f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: abstention contains outcome data"
            )


def _verify_source_manifest(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema_version") != "p24c.source_manifest.v1":
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: source manifest schema")
    if manifest.get("source_authority") != P24C_SOURCE_AUTHORITY:
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: source authority")
    if manifest.get("source_domains") != ["mlb.com"]:
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: source domain")
    if manifest.get("historical_date_scope") != {
        "start": "2026-06-14",
        "end": "2026-06-20",
    }:
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: source date scope")
    records = manifest.get("records")
    normalized_hashes = manifest.get("normalized_hashes")
    if not isinstance(records, list) or len(records) != 246:
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: source record count")
    if not isinstance(normalized_hashes, dict):
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: normalized hashes missing")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: malformed source record")
        path_value = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_hash, str):
            raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: source record identity")
        path = root / path_value
        if not path.is_file() or _sha256(path.read_bytes()) != expected_hash:
            raise ValueError(
                f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: source record drift {path_value}"
            )
        if "statsapi.mlb.com" not in str(record.get("url", "")):
            raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: source URL authority")
    manifest_fingerprint = _sha256(_canonical_json(manifest))
    if manifest_fingerprint != summary.get("source_manifest_fingerprint"):
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: manifest fingerprint")
    if manifest_fingerprint != P24C_SOURCE_MANIFEST_FINGERPRINT:
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: manifest identity")
    return {
        "fingerprint": manifest_fingerprint,
        "records": records,
        "normalized_hashes": normalized_hashes,
    }


def _load_prediction_authority(
    *,
    root: Path,
    summary: dict[str, Any],
    prediction_bytes: bytes | None,
    feature_unavailable_bytes: bytes | None,
    prediction_rows: Sequence[Mapping[str, Any]] | None,
    feature_unavailable_rows: Sequence[Mapping[str, Any]] | None,
) -> P24CPredictionAuthority:
    committed_prediction_bytes = (root / P24C_REPORT_ROOT / "predictions.jsonl").read_bytes()
    committed_unavailable_bytes = (
        root / P24C_REPORT_ROOT / "feature_unavailable.jsonl"
    ).read_bytes()
    if _sha256(committed_prediction_bytes) != P24C_PREDICTION_FINGERPRINT:
        raise ValueError(f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: committed predictions")
    if _sha256(committed_unavailable_bytes) != summary["feature_unavailable_set_fingerprint"]:
        raise ValueError(f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: committed abstentions")
    committed_predictions = _parse_jsonl(
        committed_prediction_bytes, "P24C predictions"
    )
    committed_unavailable = _parse_jsonl(
        committed_unavailable_bytes, "P24C feature unavailable"
    )
    candidate_predictions = (
        _copy_rows(prediction_rows)
        if prediction_rows is not None
        else _parse_jsonl(
            prediction_bytes if prediction_bytes is not None else committed_prediction_bytes,
            "P24C predictions",
        )
    )
    candidate_unavailable = (
        _copy_rows(feature_unavailable_rows)
        if feature_unavailable_rows is not None
        else _parse_jsonl(
            feature_unavailable_bytes
            if feature_unavailable_bytes is not None
            else committed_unavailable_bytes,
            "P24C feature unavailable",
        )
    )
    _validate_prediction_rows(candidate_predictions, summary=summary)
    _validate_unavailable_rows(candidate_unavailable, summary=summary)
    if len(candidate_predictions) != P24C_PREDICTION_COUNT:
        raise ValueError(f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: prediction count")
    if len(candidate_unavailable) != P24C_FEATURE_UNAVAILABLE_COUNT:
        raise ValueError(f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: abstention count")
    _assert_same_authority_rows(
        candidate=candidate_predictions,
        committed=committed_predictions,
        identity_keys=("prediction_id",),
        label="prediction rows",
    )
    _assert_same_authority_rows(
        candidate=candidate_unavailable,
        committed=committed_unavailable,
        identity_keys=("game_id",),
        label="abstention rows",
    )
    prediction_ids = {row["prediction_id"] for row in candidate_predictions}
    unavailable_game_ids = {row["game_id"] for row in candidate_unavailable}
    prediction_game_ids = {row["game_id"] for row in candidate_predictions}
    if prediction_game_ids.intersection(unavailable_game_ids):
        raise ValueError(f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: overlapping accounting")
    if len(prediction_game_ids | unavailable_game_ids) != P24C_RAW_GAME_COUNT:
        raise ValueError(f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: raw game accounting")
    if len(prediction_ids) != len(candidate_predictions):
        raise ValueError(f"{P25A_STOP_PREDICTION_AUTHORITY_DRIFT}: duplicate prediction ids")
    return P24CPredictionAuthority(
        batch_id=P24C_BATCH_ID,
        prediction_fingerprint=P24C_PREDICTION_FINGERPRINT,
        source_manifest_fingerprint=P24C_SOURCE_MANIFEST_FINGERPRINT,
        raw_game_count=P24C_RAW_GAME_COUNT,
        predictions=_sort_rows(candidate_predictions, "prediction_id"),
        feature_unavailable=_sort_rows(candidate_unavailable, "game_id"),
        summary=dict(summary),
    )


def _load_and_verify_model(root: Path) -> None:
    path = root / "report/p22b_moneyline_challenger/model_artifact.json"
    if not path.is_file():
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: P22B artifact missing")
    artifact, fingerprint = load_model_artifact_with_fingerprint(path)
    if artifact.model_id != P24C_MODEL_ID or fingerprint != P24C_MODEL_FINGERPRINT:
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: P22B artifact identity")


def _validate_schedule_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None,
) -> tuple[dict[str, Any], ...]:
    required = (
        "provider_game_id",
        "game_number",
        "scheduled_start_utc",
        "status",
        "final",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    )
    seen: set[tuple[str, int, Any]] = set()
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        missing = [field for field in required if field not in source]
        if missing:
            raise ValueError(
                f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: schedule row {index} missing {missing}"
            )
        provider_game_id = source["provider_game_id"]
        game_number = source["game_number"]
        if not isinstance(provider_game_id, str) or not provider_game_id:
            raise ValueError(f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: invalid game identity")
        if isinstance(game_number, bool) or not isinstance(game_number, int) or game_number < 1:
            raise ValueError(f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: invalid game number")
        key = (provider_game_id, game_number, source["scheduled_start_utc"])
        if key in seen:
            raise ValueError(
                f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: ambiguous duplicate game {key}"
            )
        seen.add(key)
        if source["status"] != "Final" or source["final"] is not True:
            raise ValueError(
                f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: game is not final {key}"
            )
        for field in ("home_score", "away_score"):
            value = source[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: invalid {field} for {key}"
                )
        if not isinstance(source["home_team"], dict) or not isinstance(
            source["away_team"], dict
        ):
            raise ValueError(f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: team identity missing")
        if not isinstance(source["home_team"].get("name"), str) or not isinstance(
            source["away_team"].get("name"), str
        ):
            raise ValueError(f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: team name missing")
        _parse_utc(source["scheduled_start_utc"], "scheduled_start_utc")
        normalized.append(dict(source))
    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(
            f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: committed schedule count {len(normalized)}"
        )
    return _sort_rows(normalized, "scheduled_start_utc", "game_number", "provider_game_id")


def _verify_target_boxscores(
    rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
) -> None:
    schedule_keys = {
        (
            row["provider_game_id"],
            row["game_number"],
            row["scheduled_start_utc"],
        ): row
        for row in schedule_rows
    }
    box_keys: set[tuple[str, int, str]] = set()
    for row in rows:
        key = (
            row.get("provider_game_id"),
            row.get("game_number"),
            row.get("scheduled_start_utc"),
        )
        if key in box_keys:
            raise ValueError(f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: duplicate boxscore {key}")
        box_keys.add(key)
        if key not in schedule_keys:
            raise ValueError(f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: boxscore identity mismatch")
        if row.get("home_team") != schedule_keys[key].get("home_team") or row.get(
            "away_team"
        ) != schedule_keys[key].get("away_team"):
            raise ValueError(f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: boxscore team mismatch")
    if box_keys != set(schedule_keys):
        raise ValueError(f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: boxscore coverage mismatch")


def _schedule_source_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = manifest["records"]
    matches = [
        record
        for record in records
        if str(record.get("path", "")).endswith(
            "/raw/schedule_2026-03-25_2026-06-30.json"
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: schedule source record")
    return dict(matches[0])


def _build_result_authority(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    authority: P24CPredictionAuthority,
    schedule_bytes: bytes | None,
    target_boxscores_bytes: bytes | None,
    schedule_rows_override: Sequence[Mapping[str, Any]] | None,
    target_boxscore_rows_override: Sequence[Mapping[str, Any]] | None,
) -> tuple[bytes, dict[str, dict[str, Any]], str, dict[str, Any]]:
    normalized_hashes = manifest["normalized_hashes"]
    schedule_relative = P24C_NORMALIZED_ROOT / "schedule.jsonl"
    boxscore_relative = P24C_NORMALIZED_ROOT / "target_boxscores.jsonl"
    committed_schedule_bytes = (root / schedule_relative).read_bytes()
    committed_boxscore_bytes = (root / boxscore_relative).read_bytes()
    if _sha256(committed_schedule_bytes) != normalized_hashes.get("schedule.jsonl"):
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: normalized schedule")
    if _sha256(committed_boxscore_bytes) != normalized_hashes.get("target_boxscores.jsonl"):
        raise ValueError(f"{P25A_STOP_PRIOR_AUTHORITY_DRIFT}: normalized boxscores")
    schedule_rows = (
        _copy_rows(schedule_rows_override)
        if schedule_rows_override is not None
        else _parse_jsonl(
            schedule_bytes if schedule_bytes is not None else committed_schedule_bytes,
            "P24C normalized schedule",
        )
    )
    boxscore_rows = (
        _copy_rows(target_boxscore_rows_override)
        if target_boxscore_rows_override is not None
        else _parse_jsonl(
            target_boxscores_bytes
            if target_boxscores_bytes is not None
            else committed_boxscore_bytes,
            "P24C normalized target boxscores",
        )
    )
    schedule = _validate_schedule_rows(
        schedule_rows,
        expected_count=(
            None if schedule_rows_override is not None else P24C_RAW_GAME_COUNT
        ),
    )
    _verify_target_boxscores(boxscore_rows, schedule)
    schedule_by_game_id = {
        f"{row['provider_game_id']}@{row['scheduled_start_utc']}": row for row in schedule
    }
    unavailable_game_ids = {row["game_id"] for row in authority.feature_unavailable}
    prediction_game_ids = {row["game_id"] for row in authority.predictions}
    missing_prediction_games = prediction_game_ids - set(schedule_by_game_id)
    if missing_prediction_games:
        raise ValueError(
            f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: missing result for "
            f"{sorted(missing_prediction_games)[0]}"
        )
    if set(schedule_by_game_id) != prediction_game_ids | unavailable_game_ids:
        raise ValueError(
            f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: schedule/prediction accounting"
        )
    source_record = _schedule_source_record(manifest)
    result_observed_at = _parse_utc(
        source_record.get("acquired_at_utc"),
        "source schedule acquired_at_utc",
    )
    schedule_sha256 = str(manifest["normalized_hashes"]["schedule.jsonl"])
    result_rows: list[dict[str, Any]] = []
    provenance_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for prediction in authority.predictions:
        game_id = prediction["game_id"]
        schedule_row = schedule_by_game_id.get(game_id)
        if schedule_row is None:
            raise ValueError(
                f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: missing result for {game_id}"
            )
        provider_game_id = game_id
        game_number = int(schedule_row["game_number"])
        scheduled_start = _parse_utc(schedule_row["scheduled_start_utc"], "scheduled_start_utc")
        if prediction["source_provider_game_id"] != str(schedule_row["provider_game_id"]):
            raise ValueError(
                f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: provider identity mismatch {game_id}"
            )
        if prediction["scheduled_start"] != schedule_row["scheduled_start_utc"]:
            raise ValueError(
                f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: scheduled identity mismatch {game_id}"
            )
        if prediction["home_team"] != schedule_row["home_team"]["name"] or prediction[
            "away_team"
        ] != schedule_row["away_team"]["name"]:
            raise ValueError(
                f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: home/away identity mismatch {game_id}"
            )
        if result_observed_at <= scheduled_start:
            raise ValueError(
                f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: result observed before game {game_id}"
            )
        if schedule_row["home_score"] == schedule_row["away_score"]:
            raise ValueError(
                f"{P25A_STOP_RESULT_PROVENANCE_UNRESOLVED}: tied final unsupported {game_id}"
            )
        source_result_id = (
            f"p24c:{authority.source_manifest_fingerprint}:{schedule_relative.as_posix()}"
            f":{provider_game_id}:{game_number}"
        )
        result_row = {
            "source_result_id": source_result_id,
            "provider_namespace": P24C_SOURCE_AUTHORITY,
            "provider_game_id": provider_game_id,
            "game_number": game_number,
            "status": "FINAL",
            "result_observed_at_utc": source_record["acquired_at_utc"],
            "home_score": schedule_row["home_score"],
            "away_score": schedule_row["away_score"],
        }
        result_rows.append(result_row)
        provenance_by_key[(provider_game_id, game_number)] = {
            "authority": P24C_SOURCE_AUTHORITY,
            "finality_status": schedule_row["status"],
            "normalized_path": schedule_relative.as_posix(),
            "normalized_sha256": schedule_sha256,
            "source_record_path": source_record["path"],
            "source_record_sha256": source_record["sha256"],
            "source_record_url": source_record["url"],
            "source_record_acquired_at_utc": source_record["acquired_at_utc"],
        }
    result_rows = list(_sort_rows(result_rows, "provider_game_id", "game_number"))
    final_results_bytes = _canonical_jsonl(result_rows)
    result_identity_rows = [
        {
            "provider_namespace": row["provider_namespace"],
            "provider_game_id": row["provider_game_id"],
            "game_number": row["game_number"],
            "status": row["status"],
            "result_observed_at_utc": row["result_observed_at_utc"],
            "home_score": row["home_score"],
            "away_score": row["away_score"],
        }
        for row in result_rows
    ]
    result_authority_fingerprint = _sha256(_canonical_jsonl(result_identity_rows))
    result_authority_summary = {
        "authority": P24C_SOURCE_AUTHORITY,
        "normalized_path": schedule_relative.as_posix(),
        "normalized_sha256": schedule_sha256,
        "source_record_path": source_record["path"],
        "source_record_sha256": source_record["sha256"],
        "source_record_url": source_record["url"],
        "source_record_acquired_at_utc": source_record["acquired_at_utc"],
        "committed_final_result_count": len(schedule),
        "settlement_result_count": len(result_rows),
        "all_settlement_results_final": True,
        "result_authority_fingerprint": result_authority_fingerprint,
    }
    return (
        final_results_bytes,
        provenance_by_key,
        result_authority_fingerprint,
        result_authority_summary,
    )


def _build_observation(prediction: Mapping[str, Any], game_number: int) -> dict[str, Any]:
    scheduled_start = _parse_utc(prediction["scheduled_start"], "scheduled_start")
    generated = scheduled_start - timedelta(seconds=3)
    response = scheduled_start - timedelta(seconds=2)
    ingested = scheduled_start - timedelta(seconds=1)
    selection = prediction["predicted_side"]
    selected_probability = (
        prediction["home_win_probability"]
        if selection == "HOME"
        else prediction["away_win_probability"]
    )
    schedule_identity = _sha256(
        _canonical_json(
            {
                "batch_id": prediction["batch_id"],
                "game_id": prediction["game_id"],
                "scheduled_start": prediction["scheduled_start"],
                "home_team": prediction["home_team"],
                "away_team": prediction["away_team"],
            }
        )
    )
    return {
        "prediction_observation_id": prediction["prediction_id"],
        "source_prediction_id": prediction["source_prediction_id"],
        "batch_id": prediction["batch_id"],
        "prediction_id": prediction["prediction_id"],
        "game_id": prediction["game_id"],
        "model_id": prediction["model_id"],
        "model_fingerprint": prediction["model_fingerprint"],
        "market_id": "moneyline",
        "selection": selection,
        "model_probability": str(selected_probability),
        "home_win_probability": prediction["home_win_probability"],
        "away_win_probability": prediction["away_win_probability"],
        "predicted_side": selection,
        "feature_snapshot_id": prediction["feature_snapshot_id"],
        "feature_snapshot_fingerprint": prediction["feature_snapshot_fingerprint"],
        "feature_eligibility": "ELIGIBLE",
        "line_value": "0",
        "push_policy": "NO_PUSH",
        "provider_namespace": P24C_SOURCE_AUTHORITY,
        "provider_game_id": prediction["game_id"],
        "game_number": game_number,
        "source_schedule_observation_id": schedule_identity,
        "prediction_generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
        "response_received_at_utc": response.isoformat().replace("+00:00", "Z"),
        "ingested_at_utc": ingested.isoformat().replace("+00:00", "Z"),
        "scheduled_start_utc": prediction["scheduled_start"],
    }


def _build_snapshot(
    *,
    authority: P24CPredictionAuthority,
    schedule_by_game_id: Mapping[str, Mapping[str, Any]],
) -> tuple[
    AdmittedPredictionObservationSnapshotResult,
    bytes,
    bytes,
]:
    admission_rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(authority.predictions, start=1):
        schedule_row = schedule_by_game_id[prediction["game_id"]]
        observation = _build_observation(prediction, int(schedule_row["game_number"]))
        admission_rows.append(
            {
                "request_index": index,
                "admission_status": "ADMITTED",
                "reason": None,
                "observation": observation,
            }
        )
    result_set_fingerprint = _sha256(
        "".join(
            f"ADMITTED::{row['observation']['prediction_observation_id']}\n"
            for row in admission_rows
        ).encode("utf-8")
    )
    admission_bytes = _canonical_jsonl(admission_rows)
    admission_summary_bytes = _canonical_json(
        {
            "schedule_as_of_utc": authority.summary.get("window_end_date"),
            "request_count": len(admission_rows),
            "admitted_count": len(admission_rows),
            "rejected_count": 0,
            "rejection_reasons_breakdown": {},
            "result_set_fingerprint": result_set_fingerprint,
            "input_hashes": {
                "p24c_prediction_fingerprint": authority.prediction_fingerprint,
                "p24c_source_manifest_fingerprint": authority.source_manifest_fingerprint,
            },
            "claims": {
                "provider_called": False,
                "db_written": False,
                "legacy_rows_admitted": False,
                "deployed": False,
                "betting_claim": False,
            },
        }
    )
    snapshot_result = build_admitted_prediction_observation_snapshot(
        results_bytes=admission_bytes,
        summary_bytes=admission_summary_bytes,
    )
    snapshot_bytes = render_admitted_observations_jsonl(snapshot_result).encode("utf-8")
    snapshot_report_bytes = render_snapshot_report_markdown(snapshot_result).encode("utf-8")
    snapshot_summary_bytes = render_snapshot_summary_json(
        snapshot_result,
        _sha256(snapshot_bytes),
        _sha256(snapshot_report_bytes),
    ).encode("utf-8")
    return snapshot_result, snapshot_bytes, snapshot_summary_bytes


def _settled_prediction_rows(
    *,
    authority: P24CPredictionAuthority,
    attachment_result: FinalResultAttachmentResult,
    final_results_bytes: bytes,
    provenance_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    final_results = load_final_result_observations(final_results_bytes)
    result_by_key = {
        (row.provider_game_id, row.game_number): row for row in final_results
    }
    prediction_by_id = {row["prediction_id"]: row for row in authority.predictions}
    attachment_by_id = {
        row.prediction_observation_id: row for row in attachment_result.attachment_rows
    }
    if set(prediction_by_id) != set(attachment_by_id):
        raise ValueError(f"{P25A_STOP_FEEDBACK_LINEAGE_MISMATCH}: attachment prediction ids")
    settled: list[dict[str, Any]] = []
    for prediction_id in sorted(prediction_by_id):
        prediction = prediction_by_id[prediction_id]
        attachment = attachment_by_id[prediction_id]
        if attachment.attachment_status != "ATTACHED":
            raise ValueError(
                f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: result was not attached {prediction_id}"
            )
        key = (attachment.provider_game_id, attachment.game_number)
        result = result_by_key.get(key)
        provenance = provenance_by_key.get(key)
        if result is None or provenance is None:
            raise ValueError(f"{P25A_STOP_FEEDBACK_LINEAGE_MISMATCH}: result lineage {prediction_id}")
        settled.append(
            {
                "schema_version": P25A_SETTLED_ROW_SCHEMA_VERSION,
                "prediction_id": prediction["prediction_id"],
                "prediction_observation_id": attachment.prediction_observation_id,
                "source_prediction_id": prediction["source_prediction_id"],
                "source_provider_game_id": prediction["source_provider_game_id"],
                "batch_id": prediction["batch_id"],
                "game_id": prediction["game_id"],
                "scheduled_start": prediction["scheduled_start"],
                "model_id": prediction["model_id"],
                "model_fingerprint": prediction["model_fingerprint"],
                "feature_snapshot_id": prediction["feature_snapshot_id"],
                "feature_snapshot_fingerprint": prediction["feature_snapshot_fingerprint"],
                "home_win_probability": prediction["home_win_probability"],
                "away_win_probability": prediction["away_win_probability"],
                "predicted_side": prediction["predicted_side"],
                "provider_namespace": attachment.provider_namespace,
                "provider_game_id": attachment.provider_game_id,
                "game_number": attachment.game_number,
                "attachment_status": attachment.attachment_status,
                "result_observation_id": result.result_observation_id,
                "source_result_id": result.source_result_id,
                "result_observed_at_utc": result.result_observed_at_utc,
                "home_score": result.home_score,
                "away_score": result.away_score,
                "actual_winner": attachment.actual_winner,
                "is_correct": attachment.is_correct,
                "source_snapshot_row_fingerprint": attachment.source_snapshot_row_fingerprint,
                "attachment_row_fingerprint": attachment.attachment_row_fingerprint,
                "result_provenance": dict(provenance),
            }
        )
    return tuple(settled)


def settle_paper_moneyline_batch(
    repository_root: str | Path = ".",
    *,
    prediction_bytes: bytes | None = None,
    feature_unavailable_bytes: bytes | None = None,
    schedule_bytes: bytes | None = None,
    target_boxscores_bytes: bytes | None = None,
    prediction_rows: Sequence[Mapping[str, Any]] | None = None,
    feature_unavailable_rows: Sequence[Mapping[str, Any]] | None = None,
    schedule_rows: Sequence[Mapping[str, Any]] | None = None,
    target_boxscore_rows: Sequence[Mapping[str, Any]] | None = None,
) -> PaperMoneylineSettlementResult:
    """Run the complete offline P25A settlement/evaluation/feedback loop.

    The optional row/byte arguments are deterministic fixture-injection seams
    for negative and order-invariance tests.  The CLI never supplies them and
    therefore consumes only the committed P24C files.
    """

    root = Path(repository_root).resolve()
    if prediction_bytes is not None and prediction_rows is not None:
        raise TypeError("supply prediction_bytes or prediction_rows, not both")
    if feature_unavailable_bytes is not None and feature_unavailable_rows is not None:
        raise TypeError("supply feature_unavailable_bytes or feature_unavailable_rows, not both")
    if schedule_bytes is not None and schedule_rows is not None:
        raise TypeError("supply schedule_bytes or schedule_rows, not both")
    if target_boxscores_bytes is not None and target_boxscore_rows is not None:
        raise TypeError("supply target_boxscores_bytes or target_boxscore_rows, not both")

    summary = _parse_object(
        (root / P24C_REPORT_ROOT / "summary.json").read_bytes(),
        "P24C summary",
    )
    _validate_p24c_summary(summary)
    manifest = _parse_object(
        (root / P24C_REPORT_ROOT / "source_manifest.json").read_bytes(),
        "P24C source manifest",
    )
    manifest_info = _verify_source_manifest(
        root=root,
        manifest=manifest,
        summary=summary,
    )
    _load_and_verify_model(root)
    authority = _load_prediction_authority(
        root=root,
        summary=summary,
        prediction_bytes=prediction_bytes,
        feature_unavailable_bytes=feature_unavailable_bytes,
        prediction_rows=prediction_rows,
        feature_unavailable_rows=feature_unavailable_rows,
    )
    final_results_bytes, provenance_by_key, result_authority_fingerprint, result_authority_summary = (
        _build_result_authority(
            root=root,
            manifest=manifest_info,
            authority=authority,
            schedule_bytes=schedule_bytes,
            target_boxscores_bytes=target_boxscores_bytes,
            schedule_rows_override=schedule_rows,
            target_boxscore_rows_override=target_boxscore_rows,
        )
    )
    final_result_observations: list[FinalResultObservation] = load_final_result_observations(
        final_results_bytes
    )
    if len(final_result_observations) != P24C_PREDICTION_COUNT:
        raise ValueError(f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: final result input")
    schedule_by_game_id = {
        f"{row['provider_game_id']}@{row['scheduled_start_utc']}": row
        for row in _validate_schedule_rows(
            _parse_jsonl(
                schedule_bytes
                if schedule_bytes is not None
                else (root / P24C_NORMALIZED_ROOT / "schedule.jsonl").read_bytes(),
                "P24C normalized schedule",
            )
            if schedule_rows is None
            else schedule_rows,
            expected_count=(None if schedule_rows is not None else P24C_RAW_GAME_COUNT),
        )
    }
    snapshot_result, snapshot_bytes, snapshot_summary_bytes = _build_snapshot(
        authority=authority,
        schedule_by_game_id=schedule_by_game_id,
    )
    attachment_result = attach_final_results_to_admitted_predictions(
        snapshot_bytes=snapshot_bytes,
        summary_bytes=snapshot_summary_bytes,
        final_results_bytes=final_results_bytes,
    )
    if (
        attachment_result.source_prediction_count != P24C_PREDICTION_COUNT
        or attachment_result.final_result_observation_count != P24C_PREDICTION_COUNT
        or attachment_result.attached_count != P24C_PREDICTION_COUNT
        or attachment_result.rejected_count != 0
    ):
        raise ValueError(f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: P16A attachment")
    attachment_bytes = render_attachments_jsonl(attachment_result).encode("utf-8")
    attachment_report_bytes = render_attachment_report_markdown(attachment_result).encode("utf-8")
    attachment_summary_bytes = render_attachment_summary_json(
        attachment_result,
        _sha256(attachment_bytes),
        _sha256(attachment_report_bytes),
    ).encode("utf-8")
    evaluation_result = build_prediction_evaluation_scorecard(
        attachments_bytes=attachment_bytes,
        attachment_summary_bytes=attachment_summary_bytes,
        snapshot_bytes=snapshot_bytes,
        snapshot_summary_bytes=snapshot_summary_bytes,
    )
    if evaluation_result.evaluation_row_count != P24C_PREDICTION_COUNT:
        raise ValueError(f"{P25A_STOP_SETTLEMENT_COUNT_MISMATCH}: P16B evaluation")
    evaluation_bytes = render_evaluations_jsonl(evaluation_result).encode("utf-8")
    evaluation_report_bytes = render_evaluation_report_markdown(evaluation_result).encode("utf-8")
    evaluation_summary_bytes = render_evaluation_summary_json(
        evaluation_result,
        _sha256(evaluation_bytes),
        _sha256(evaluation_report_bytes),
    ).encode("utf-8")
    feedback_result = build_prediction_feedback_ledger(
        snapshot_bytes=snapshot_bytes,
        snapshot_summary_bytes=snapshot_summary_bytes,
        attachments_bytes=attachment_bytes,
        attachment_summary_bytes=attachment_summary_bytes,
        evaluations_bytes=evaluation_bytes,
        evaluation_summary_bytes=evaluation_summary_bytes,
    )
    if (
        feedback_result.prediction_row_count != P24C_PREDICTION_COUNT
        or feedback_result.evaluated_row_count != P24C_PREDICTION_COUNT
        or feedback_result.non_evaluated_row_count != 0
    ):
        raise ValueError(f"{P25A_STOP_FEEDBACK_LINEAGE_MISMATCH}: P17 feedback counts")
    settled_predictions = _settled_prediction_rows(
        authority=authority,
        attachment_result=attachment_result,
        final_results_bytes=final_results_bytes,
        provenance_by_key=provenance_by_key,
    )
    if len(settled_predictions) != P24C_PREDICTION_COUNT:
        raise ValueError(f"{P25A_STOP_FEEDBACK_LINEAGE_MISMATCH}: settled rows")
    feedback_ids = {
        row.prediction_observation_id for row in feedback_result.feedback_rows
    }
    if feedback_ids != {row["prediction_id"] for row in authority.predictions}:
        raise ValueError(f"{P25A_STOP_FEEDBACK_LINEAGE_MISMATCH}: P17 ids")
    claims: dict[str, Any] = {
        "all_results_final": True,
        "challenger_retrained": False,
        "deployment_performed": False,
        "model_promoted": True,
        "offline_settlement": True,
        "paper_only": True,
        "prediction_authority_verified": True,
        "production_ready": False,
        "profitability_claim": False,
        "real_betting_recommendation": False,
        "retraining_performed": False,
        "result_mutation_isolation": True,
        "feature_eligibility_is_pregame_only": True,
    }
    result_authority_summary = {
        **result_authority_summary,
        "source_manifest_fingerprint": authority.source_manifest_fingerprint,
    }
    return PaperMoneylineSettlementResult(
        authority=authority,
        result_authority_fingerprint=result_authority_fingerprint,
        result_authority_summary=result_authority_summary,
        snapshot_result=snapshot_result,
        attachment_result=attachment_result,
        evaluation_result=evaluation_result,
        feedback_result=feedback_result,
        settled_predictions=settled_predictions,
        claims=claims,
    )


__all__ = (
    "P25A_SCHEMA_VERSION",
    "P25A_SETTLED_ROW_SCHEMA_VERSION",
    "P24CPredictionAuthority",
    "PaperMoneylineSettlementResult",
    "settle_paper_moneyline_batch",
)
