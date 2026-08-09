"""Replay contiguous P13 folds through the verified P20A-P21A contracts.

This module is a bounded batch adapter.  It fits each selected fold with the
existing P20A implementation, runs the existing P15C/P16A/P16B/P17A stages,
and applies the existing P21A row-level eligibility rule.  It deliberately
does not change any upstream contract or introduce a second model semantics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.moneyline_walk_forward_fold import (
    MoneylineWalkForwardFold,
    ReconstructedWalkForwardModel,
)
from ...baseball.domain.prediction_feedback import (
    FEEDBACK_LEDGER_SCHEMA_VERSION,
    PredictionFeedbackRow,
    compute_feedback_ledger_fingerprint,
)
from ...baseball.domain.prediction_learning_eligibility import (
    ASSESSMENT_SCHEMA_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    ELIGIBILITY_CONTRACT_VERSION,
    ELIGIBLE,
    PredictionLearningEligibilityAssessment,
    assess_prediction_learning_eligibility,
)
from ...baseball.domain.prediction_admission import admit_prospective_prediction
from .admitted_prediction_observation_artifacts import (
    render_admitted_observations_jsonl,
    render_snapshot_report_markdown,
    render_snapshot_summary_json,
)
from .attach_final_results_to_admitted_predictions import (
    attach_final_results_to_admitted_predictions,
)
from .build_admitted_prediction_observation_snapshot import (
    build_admitted_prediction_observation_snapshot,
)
from .build_prediction_evaluation_scorecard import (
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
from .generate_moneyline_predictions import (
    MoneylineInferenceResult,
    _candidate_set_fingerprint,
    generate_moneyline_predictions,
)
from .moneyline_inference_artifacts import render_predictions_jsonl
from .moneyline_walk_forward_artifacts import build_moneyline_model_artifact
from .prediction_evaluation_artifacts import (
    render_evaluation_report_markdown,
    render_evaluation_summary_json,
    render_evaluations_jsonl,
)
from .prediction_feedback_artifacts import render_feedback_jsonl
from .prospective_prediction_admission_artifacts import (
    render_results_jsonl,
    render_summary_json as render_admission_summary_json,
)
from .reconstruct_moneyline_walk_forward_model import (
    load_moneyline_walk_forward_fold,
    reconstruct_moneyline_walk_forward_model,
)
from .replay_historical_moneyline_predictions import (
    MoneylineParityRow,
    MoneylineWalkForwardReplayResult,
    _snapshot_for_row,
)
from .replay_historical_prediction_feedback import (
    EXPECTED_LEGACY_ARCHIVE_BLOB,
    EXPECTED_LEGACY_ARCHIVE_PATH,
    EXPECTED_LEGACY_ARCHIVE_SHA256,
    EXPECTED_LEGACY_COMMIT,
    EXPECTED_LEGACY_MEMBER,
    EXPECTED_LEGACY_REPOSITORY,
    EXPECTED_LEGACY_TREE,
    EXPECTED_RESULT_VERIFIED_AT,
    _build_admission_context,
)
from .run_prospective_prediction_admission_workflow import (
    ProspectivePredictionAdmissionWorkflowResult,
    compute_result_set_fingerprint,
)


P21B_SCHEMA_VERSION = "p21b.contiguous_multifold_historical_candidates.v1"
P21B_STOP_NEXT_FOLD = "STOP_MATCHANALYSIS_P21B_NEXT_FOLD_UNRESOLVED"
P21B_STOP_RESULT_PROVENANCE = (
    "STOP_MATCHANALYSIS_P21B_RESULT_PROVENANCE_UNRESOLVED"
)
P20B_HISTORICAL_RUNTIME_COMPLIANCE = "REMAINS_REFUTED"
P21B_RECONSTRUCTED_MODELS_SCHEMA_VERSION = "p21b.reconstructed_models.v1"

P21B_CLAIMS: dict[str, bool] = {
    "db_written": False,
    "deployed": False,
    "historical": True,
    "model_promoted": False,
    "network_called": False,
    "odds_used": False,
    "profitability_claim": False,
    "real_betting_recommendation": False,
    "retraining_performed": False,
    "sample_limited": True,
    "synthetic_results": False,
    "training_authorized": False,
    "training_dataset_claim": False,
}


@dataclass(frozen=True, slots=True)
class FoldReplaySummary:
    """Deterministic evidence for one complete selected P13 fold."""

    fold_id: str
    train_as_of: str
    validation_start: str
    validation_end: str
    training_row_count: int
    prediction_row_count: int
    prediction_game_ids: tuple[str, ...]
    fold_fingerprint: str
    model_fingerprint: str
    model_artifact_fingerprint: str
    parity_rows: tuple[dict[str, Any], ...]
    max_absolute_difference: str
    parity_passed: bool
    p15c_admission_count: int
    p15c_result_set_fingerprint: str
    p15c_snapshot_fingerprint: str
    p16a_attachment_count: int
    p16a_attachment_set_fingerprint: str
    p16b_evaluation_count: int
    p16b_evaluation_set_fingerprint: str
    p17_feedback_row_count: int
    p17_feedback_ledger_fingerprint: str

    def to_projection(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_as_of": self.train_as_of,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "training_row_count": self.training_row_count,
            "prediction_row_count": self.prediction_row_count,
            "prediction_game_ids": list(self.prediction_game_ids),
            "fold_fingerprint": self.fold_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "model_artifact_fingerprint": self.model_artifact_fingerprint,
            "parity_rows": list(self.parity_rows),
            "max_absolute_difference": self.max_absolute_difference,
            "parity_passed": self.parity_passed,
            "p15c_admission_count": self.p15c_admission_count,
            "p15c_result_set_fingerprint": self.p15c_result_set_fingerprint,
            "p15c_snapshot_fingerprint": self.p15c_snapshot_fingerprint,
            "p16a_attachment_count": self.p16a_attachment_count,
            "p16a_attachment_set_fingerprint": self.p16a_attachment_set_fingerprint,
            "p16b_evaluation_count": self.p16b_evaluation_count,
            "p16b_evaluation_set_fingerprint": self.p16b_evaluation_set_fingerprint,
            "p17_feedback_row_count": self.p17_feedback_row_count,
            "p17_feedback_ledger_fingerprint": self.p17_feedback_ledger_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class MultifoldHistoricalCandidateReplayResult:
    """Complete deterministic P21B result and its P21A assessments."""

    folds: tuple[FoldReplaySummary, ...]
    feedback_result: PredictionFeedbackLedgerResult
    feedback_rows: tuple[PredictionFeedbackRow, ...]
    assessments: tuple[PredictionLearningEligibilityAssessment, ...]
    candidates: tuple[dict[str, Any], ...]
    historical_result_count: int
    historical_results_sha256: str
    historical_provenance_sha256: str
    historical_provenance: dict[str, Any]
    membership_sha256: str
    result_rows_sha256: str
    assessment_semantic_fingerprint: str
    candidate_semantic_fingerprint: str
    claims: dict[str, bool]

    @property
    def selected_fold_ids(self) -> tuple[str, ...]:
        return tuple(fold.fold_id for fold in self.folds)

    @property
    def prediction_row_count(self) -> int:
        return sum(fold.prediction_row_count for fold in self.folds)

    @property
    def p15c_admission_count(self) -> int:
        return sum(fold.p15c_admission_count for fold in self.folds)

    @property
    def p16a_attachment_count(self) -> int:
        return sum(fold.p16a_attachment_count for fold in self.folds)

    @property
    def p16b_evaluation_count(self) -> int:
        return sum(fold.p16b_evaluation_count for fold in self.folds)

    @property
    def p21a_eligible_count(self) -> int:
        return sum(assessment.status == ELIGIBLE for assessment in self.assessments)

    @property
    def p21a_excluded_count(self) -> int:
        return sum(assessment.status != ELIGIBLE for assessment in self.assessments)

    def to_projection(self) -> dict[str, Any]:
        return {
            "schema_version": P21B_SCHEMA_VERSION,
            "selected_fold_ids": list(self.selected_fold_ids),
            "folds": [fold.to_projection() for fold in self.folds],
            "historical_result_count": self.historical_result_count,
            "historical_results_sha256": self.historical_results_sha256,
            "historical_provenance_sha256": self.historical_provenance_sha256,
            "historical_provenance": self.historical_provenance,
            "membership_sha256": self.membership_sha256,
            "result_rows_sha256": self.result_rows_sha256,
            "p15c_admission_count": self.p15c_admission_count,
            "p16a_attachment_count": self.p16a_attachment_count,
            "p16b_evaluation_count": self.p16b_evaluation_count,
            "p17_feedback_row_count": len(self.feedback_rows),
            "p17_feedback_ledger_fingerprint": (
                self.feedback_result.feedback_ledger_fingerprint
            ),
            "p21a_assessed_count": len(self.assessments),
            "p21a_eligible_count": self.p21a_eligible_count,
            "p21a_excluded_count": self.p21a_excluded_count,
            "assessment_semantic_fingerprint": self.assessment_semantic_fingerprint,
            "candidate_semantic_fingerprint": self.candidate_semantic_fingerprint,
            "candidate_count": len(self.candidates),
            "aggregate_candidate_fingerprint": self.candidate_semantic_fingerprint,
            "p20b_historical_runtime_compliance": (
                P20B_HISTORICAL_RUNTIME_COMPLIANCE
            ),
            "claims": dict(self.claims),
        }


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _parse_object(raw: bytes, context: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _parse_jsonl(raw: bytes, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{context} contains a blank line at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{context} row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{context} contains no rows")
    return rows


def _render_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def _membership_sha256(folds: Sequence[MoneylineWalkForwardFold]) -> str:
    serialized = "".join(
        f"{fold.fold_id}\t{row.date}\t{row.game_id}\n"
        for fold in folds
        for row in fold.prediction_rows
    ).encode("utf-8")
    return _sha256(serialized)


def _semantic_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return _sha256(canonical)


def _result_rows_sha256(
    *,
    folds: Sequence[MoneylineWalkForwardFold],
    result_by_game: Mapping[str, dict[str, Any]],
    provenance: Mapping[str, Any],
) -> str:
    """Hash the frozen result identity/score serialization used by P21B."""

    provenance_by_game = {
        row["canonical_game_id"]: row for row in provenance["rows"]
    }
    serialized = ""
    for fold in folds:
        for prediction in fold.prediction_rows:
            source = provenance_by_game[prediction.game_id]
            result = result_by_game[prediction.game_id]
            _, home_team_code, away_team_code = prediction.game_id.split("_", 2)
            serialized += (
                f"{prediction.game_id}\t{prediction.date}\t"
                f"{source['game_number']}\t{home_team_code}\t"
                f"{result['home_score']}\t{away_team_code}\t"
                f"{result['away_score']}\n"
            )
    return _sha256(serialized.encode("utf-8"))


def _validate_contiguous_folds(
    folds: Sequence[MoneylineWalkForwardFold],
) -> tuple[MoneylineWalkForwardFold, ...]:
    ordered = tuple(sorted(folds, key=lambda fold: fold.fold_id))
    if not ordered:
        raise ValueError(f"{P21B_STOP_NEXT_FOLD}: no folds supplied")
    expected = tuple(f"wf_{index:03d}" for index in range(2, len(ordered) + 2))
    actual = tuple(fold.fold_id for fold in ordered)
    if actual != expected:
        raise ValueError(
            f"{P21B_STOP_NEXT_FOLD}: expected contiguous folds {expected}, got {actual}"
        )
    for previous, current in zip(ordered, ordered[1:]):
        if current.validation_start <= previous.validation_start:
            raise ValueError(f"{P21B_STOP_NEXT_FOLD}: validation ranges are not ordered")
        if current.train_as_of != previous.validation_end:
            raise ValueError(
                f"{P21B_STOP_NEXT_FOLD}: {current.fold_id} does not train through "
                f"{previous.fold_id} validation end"
            )
    return ordered


def _expected_source() -> dict[str, Any]:
    return {
        "source_repository": EXPECTED_LEGACY_REPOSITORY,
        "source_commit": EXPECTED_LEGACY_COMMIT,
        "source_tree": EXPECTED_LEGACY_TREE,
        "source_archive_path": EXPECTED_LEGACY_ARCHIVE_PATH,
        "source_archive_blob": EXPECTED_LEGACY_ARCHIVE_BLOB,
        "source_archive_sha256": EXPECTED_LEGACY_ARCHIVE_SHA256,
        "source_member": EXPECTED_LEGACY_MEMBER,
        "source_verified_at_utc": EXPECTED_RESULT_VERIFIED_AT,
    }


def _validate_historical_inputs(
    *,
    folds: Sequence[MoneylineWalkForwardFold],
    historical_results_bytes: bytes,
    historical_provenance_bytes: bytes,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    provenance = _parse_object(historical_provenance_bytes, "P21B result provenance")
    for key, expected in _expected_source().items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"{P21B_STOP_RESULT_PROVENANCE}: source mismatch for {key}"
            )
    claims = provenance.get("claims")
    if not isinstance(claims, dict) or claims.get("historical") is not True:
        raise ValueError(f"{P21B_STOP_RESULT_PROVENANCE}: historical claim missing")
    if claims.get("non_synthetic") is not True:
        raise ValueError(f"{P21B_STOP_RESULT_PROVENANCE}: non-synthetic claim missing")

    result_rows = _parse_jsonl(historical_results_bytes, "P21B historical results")
    expected_ids = {
        row.game_id
        for fold in folds
        for row in fold.prediction_rows
    }
    result_by_game: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        game_id = row.get("provider_game_id")
        if not isinstance(game_id, str) or game_id in result_by_game:
            raise ValueError(f"{P21B_STOP_RESULT_PROVENANCE}: duplicate result identity")
        if row.get("provider_namespace") != "MLB_STATS_API":
            raise ValueError(f"{P21B_STOP_RESULT_PROVENANCE}: unsupported provider")
        if row.get("game_number") != 1 or row.get("status") != "FINAL":
            raise ValueError(f"{P21B_STOP_RESULT_PROVENANCE}: result is not FINAL game #1")
        if row.get("result_observed_at_utc") != EXPECTED_RESULT_VERIFIED_AT:
            raise ValueError(f"{P21B_STOP_RESULT_PROVENANCE}: result timestamp mismatch")
        if not isinstance(row.get("home_score"), int) or not isinstance(
            row.get("away_score"), int
        ):
            raise ValueError(f"{P21B_STOP_RESULT_PROVENANCE}: scores must be integers")
        result_by_game[game_id] = row
    if set(result_by_game) != expected_ids:
        raise ValueError(
            f"{P21B_STOP_RESULT_PROVENANCE}: result identities do not match folds"
        )

    provenance_rows = provenance.get("rows")
    if not isinstance(provenance_rows, list):
        raise ValueError(f"{P21B_STOP_RESULT_PROVENANCE}: rows must be a list")
    provenance_by_game: dict[str, dict[str, Any]] = {}
    for row in provenance_rows:
        game_id = row.get("canonical_game_id")
        if not isinstance(game_id, str) or game_id in provenance_by_game:
            raise ValueError(f"{P21B_STOP_RESULT_PROVENANCE}: duplicate provenance identity")
        provenance_by_game[game_id] = row
    if set(provenance_by_game) != expected_ids:
        raise ValueError(
            f"{P21B_STOP_RESULT_PROVENANCE}: provenance identities do not match folds"
        )

    for fold in folds:
        for row in fold.prediction_rows:
            result = result_by_game[row.game_id]
            source = provenance_by_game[row.game_id]
            if result["home_score"] != source.get("home_score") or result[
                "away_score"
            ] != source.get("away_score"):
                raise ValueError(
                    f"{P21B_STOP_RESULT_PROVENANCE}: result/source score mismatch for "
                    f"{row.game_id}"
                )
            observed_home_win = int(result["home_score"]) > int(result["away_score"])
            if observed_home_win != bool(row.target_home_win):
                raise ValueError(
                    f"{P21B_STOP_RESULT_PROVENANCE}: result target mismatch for {row.game_id}"
                )
    return result_by_game, provenance


def _fold_result_bytes(
    *,
    fold: MoneylineWalkForwardFold,
    result_by_game: Mapping[str, dict[str, Any]],
    provenance: Mapping[str, Any],
) -> tuple[bytes, bytes]:
    game_ids = [row.game_id for row in fold.prediction_rows]
    result_bytes = _render_jsonl([result_by_game[game_id] for game_id in game_ids])
    provenance_rows = [
        row
        for row in provenance["rows"]
        if row.get("canonical_game_id") in set(game_ids)
    ]
    provenance_projection = {
        **_expected_source(),
        "claims": dict(provenance["claims"]),
        "rows": sorted(provenance_rows, key=lambda row: row["canonical_game_id"]),
    }
    return result_bytes, _canonical_json(provenance_projection)


def _aggregate_digest(label: str, values: Sequence[str]) -> str:
    serialized = "".join(f"{label}:{value}\n" for value in sorted(values)).encode("utf-8")
    return _sha256(serialized)


def _aggregate_feedback_result(
    *,
    fold_results: Sequence[tuple[str, PredictionFeedbackLedgerResult]],
    feedback_rows: tuple[PredictionFeedbackRow, ...],
) -> PredictionFeedbackLedgerResult:
    ordered_rows = tuple(
        sorted(feedback_rows, key=lambda row: row.prediction_observation_id)
    )
    if len({row.prediction_observation_id for row in ordered_rows}) != len(ordered_rows):
        raise ValueError("P21B feedback contains duplicate prediction observations")
    status_counts = Counter(row.feedback_status for row in ordered_rows)
    rejection_counts = Counter(
        row.attachment_rejection_reason
        for row in ordered_rows
        if row.attachment_rejection_reason is not None
    )
    return PredictionFeedbackLedgerResult(
        schema_version=FEEDBACK_LEDGER_SCHEMA_VERSION,
        source_snapshot_sha256=_aggregate_digest(
            "snapshot", [result.source_snapshot_sha256 for _, result in fold_results]
        ),
        source_snapshot_summary_sha256=_aggregate_digest(
            "snapshot_summary",
            [result.source_snapshot_summary_sha256 for _, result in fold_results],
        ),
        source_attachments_sha256=_aggregate_digest(
            "attachments", [result.source_attachments_sha256 for _, result in fold_results]
        ),
        source_attachment_summary_sha256=_aggregate_digest(
            "attachment_summary",
            [result.source_attachment_summary_sha256 for _, result in fold_results],
        ),
        source_evaluations_sha256=_aggregate_digest(
            "evaluations", [result.source_evaluations_sha256 for _, result in fold_results]
        ),
        source_evaluation_summary_sha256=_aggregate_digest(
            "evaluation_summary",
            [result.source_evaluation_summary_sha256 for _, result in fold_results],
        ),
        source_snapshot_fingerprint=_aggregate_digest(
            "snapshot_fingerprint",
            [result.source_snapshot_fingerprint for _, result in fold_results],
        ),
        source_attachment_set_fingerprint=_aggregate_digest(
            "attachment_fingerprint",
            [result.source_attachment_set_fingerprint for _, result in fold_results],
        ),
        source_evaluation_set_fingerprint=_aggregate_digest(
            "evaluation_fingerprint",
            [result.source_evaluation_set_fingerprint for _, result in fold_results],
        ),
        prediction_row_count=len(ordered_rows),
        attached_row_count=sum(
            row.attachment_status == "ATTACHED" for row in ordered_rows
        ),
        rejected_attachment_row_count=sum(
            row.attachment_status == "REJECTED" for row in ordered_rows
        ),
        evaluated_row_count=sum(
            row.feedback_status == "EVALUATED" for row in ordered_rows
        ),
        non_evaluated_row_count=sum(
            row.feedback_status != "EVALUATED" for row in ordered_rows
        ),
        correct_count=sum(row.is_correct is True for row in ordered_rows),
        incorrect_count=sum(row.is_correct is False for row in ordered_rows),
        feedback_status_counts=dict(sorted(status_counts.items())),
        attachment_rejection_reason_counts=dict(sorted(rejection_counts.items())),
        feedback_rows=ordered_rows,
        feedback_ledger_fingerprint=compute_feedback_ledger_fingerprint(ordered_rows),
        claims=dict(P21B_CLAIMS),
    )


def _assessment_projection(
    assessment: PredictionLearningEligibilityAssessment,
    *,
    source_feedback_ledger_fingerprint: str,
) -> dict[str, Any]:
    return {
        "assessment_id": assessment.assessment_id,
        "assessment_schema_version": ASSESSMENT_SCHEMA_VERSION,
        "assessment_status": assessment.status,
        "candidate_id": assessment.candidate_id,
        "claims": dict(P21B_CLAIMS),
        "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
        "exclusion_reasons": list(assessment.exclusion_reasons),
        "feedback_row_fingerprint": assessment.feedback_row_fingerprint,
        "prediction_observation_id": assessment.prediction_observation_id,
        "source_feedback_ledger_fingerprint": source_feedback_ledger_fingerprint,
    }


def _candidate_projection(
    row: dict[str, Any],
    assessment: PredictionLearningEligibilityAssessment,
    *,
    source_feedback_ledger_fingerprint: str,
) -> dict[str, Any]:
    candidate = dict(row)
    candidate.update(
        {
            "assessment_id": assessment.assessment_id,
            "assessment_status": assessment.status,
            "candidate_id": assessment.candidate_id,
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "claims": dict(P21B_CLAIMS),
            "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
            "source_feedback_fingerprint": row["feedback_row_fingerprint"],
            "source_feedback_ledger_fingerprint": source_feedback_ledger_fingerprint,
        }
    )
    return candidate


def load_multifold_reconstructed_models(
    path: str | Path,
) -> dict[str, ReconstructedWalkForwardModel]:
    """Load an explicit, verified P20A model-state mapping for local replay.

    The mapping is a committed migration artifact, not a production model
    registry.  It exists so the exact P20A inference path can be replayed in a
    Python environment that does not have the legacy scientific backend.
    """

    projection = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(projection, dict):
        raise ValueError("P21B reconstructed models must be an object")
    if projection.get("schema_version") != P21B_RECONSTRUCTED_MODELS_SCHEMA_VERSION:
        raise ValueError("invalid P21B reconstructed model schema")
    models_projection = projection.get("models")
    if not isinstance(models_projection, dict) or not models_projection:
        raise ValueError("P21B reconstructed models must contain models")

    models: dict[str, ReconstructedWalkForwardModel] = {}
    for fold_id, model_projection in models_projection.items():
        if not isinstance(fold_id, str) or not isinstance(model_projection, dict):
            raise ValueError("invalid P21B reconstructed model entry")
        try:
            model = ReconstructedWalkForwardModel(
                fold_id=str(model_projection["fold_id"]),
                feature_names=tuple(
                    str(item) for item in model_projection["feature_names"]
                ),
                coefficients=tuple(
                    float(item) for item in model_projection["coefficients"]
                ),
                intercept=float(model_projection["intercept"]),
                scaler_means=tuple(
                    float(item) for item in model_projection["scaler_means"]
                ),
                scaler_stds=tuple(
                    float(item) for item in model_projection["scaler_stds"]
                ),
                train_size=int(model_projection["train_size"]),
                model_type=str(model_projection.get("model_type", "logistic_regression")),
                solver=str(model_projection.get("solver", "lbfgs")),
                max_iter=int(model_projection.get("max_iter", 1000)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid P21B reconstructed model entry: {fold_id}") from exc
        expected_fingerprint = model_projection.get("fingerprint")
        if expected_fingerprint != model.fingerprint():
            raise ValueError(f"P21B reconstructed model fingerprint mismatch: {fold_id}")
        if model.fold_id != fold_id:
            raise ValueError(f"P21B reconstructed model fold mismatch: {fold_id}")
        models[fold_id] = model
    return models


def _run_p15c_to_p17(
    *,
    fold: MoneylineWalkForwardFold,
    replay: MoneylineWalkForwardReplayResult,
    historical_results_bytes: bytes,
) -> tuple[PredictionFeedbackLedgerResult, dict[str, Any]]:
    fold_projection = fold.to_projection()
    candidates = replay.inference.candidates
    schedule_candidates, eligibility, schedule_as_of = _build_admission_context(
        candidates=candidates,
        fold=fold_projection,
    )
    admissions = tuple(
        admit_prospective_prediction(
            candidate,
            schedule_candidates=schedule_candidates,
            schedule_pregame_eligibility=eligibility,
        )
        for candidate in candidates
    )
    if any(admission.admission_status != "ADMITTED" for admission in admissions):
        raise ValueError("P21B P15C admission failed closed")
    admission_workflow = ProspectivePredictionAdmissionWorkflowResult(
        results=admissions,
        schedule_as_of_utc=schedule_as_of,
        schedule_candidates=schedule_candidates,
        schedule_pregame_eligibility=eligibility,
        admitted_count=len(admissions),
        rejected_count=0,
        result_set_fingerprint=compute_result_set_fingerprint(admissions),
    )
    prediction_bytes = render_predictions_jsonl(replay.inference).encode("utf-8")
    reconstruction_bytes = _canonical_json(replay.to_projection())
    fold_bytes = _canonical_json(fold_projection)
    admission_bytes = render_results_jsonl(admissions).encode("utf-8")
    admission_summary_bytes = render_admission_summary_json(
        admission_workflow,
        {
            "p20a_fold_sha256": _sha256(fold_bytes),
            "p20a_predictions_sha256": _sha256(prediction_bytes),
            "p20a_reconstruction_sha256": _sha256(reconstruction_bytes),
        },
    ).encode("utf-8")

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

    attachment_result = attach_final_results_to_admitted_predictions(
        snapshot_bytes=snapshot_bytes,
        summary_bytes=snapshot_summary_bytes,
        final_results_bytes=historical_results_bytes,
    )
    if attachment_result.attached_count != len(candidates):
        raise ValueError("P21B P16A did not attach every admitted candidate")
    attachment_bytes = render_attachments_jsonl(attachment_result).encode("utf-8")
    attachment_report_bytes = render_attachment_report_markdown(attachment_result).encode(
        "utf-8"
    )
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
    evaluation_bytes = render_evaluations_jsonl(evaluation_result).encode("utf-8")
    evaluation_report_bytes = render_evaluation_report_markdown(evaluation_result).encode(
        "utf-8"
    )
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
    if feedback_result.prediction_row_count != len(candidates):
        raise ValueError("P21B P17 feedback did not preserve source candidates")
    return feedback_result, {
        "p15c_admission_count": admission_workflow.admitted_count,
        "p15c_result_set_fingerprint": admission_workflow.result_set_fingerprint,
        "p15c_snapshot_fingerprint": snapshot_result.snapshot_fingerprint,
        "p16a_attachment_count": attachment_result.attached_count,
        "p16a_attachment_set_fingerprint": attachment_result.attachment_set_fingerprint,
        "p16b_evaluation_count": evaluation_result.evaluation_row_count,
        "p16b_evaluation_set_fingerprint": evaluation_result.evaluation_set_fingerprint,
    }


def _replay_fold_with_per_game_timestamps(
    fold: MoneylineWalkForwardFold,
    model: ReconstructedWalkForwardModel,
) -> MoneylineWalkForwardReplayResult:
    """Replay a multi-day fold with valid deterministic P19A timestamps.

    The existing single-fold helper uses one generated time for its tiny
    same-day fixture. A contiguous historical fold spans many days, so each
    game receives the same deterministic ``as_of + 1 minute`` timing pattern
    while all feature, model, candidate, and parity semantics remain those of
    the existing P19A generator.
    """

    model_artifact = build_moneyline_model_artifact(fold, model)
    candidates = []
    parity_rows = []
    for row, expected in zip(
        fold.prediction_rows,
        fold.expected_home_probabilities,
    ):
        snapshot = _snapshot_for_row(fold, row)
        generated = datetime.fromisoformat(row.date + "T00:00:00+00:00") + timedelta(
            minutes=1
        )
        received = generated + timedelta(seconds=1)
        ingested = received + timedelta(seconds=1)
        generated_result = generate_moneyline_predictions(
            (snapshot,),
            model_artifact,
            prediction_generated_at_utc=generated,
            response_received_at_utc=received,
            ingested_at_utc=ingested,
        )
        candidates.extend(generated_result.candidates)
        home_candidate = generated_result.candidates[0]
        expected_probability = Decimal(expected)
        difference = abs(home_candidate.model_probability - expected_probability)
        parity_rows.append(
            MoneylineParityRow(
                game_id=row.game_id,
                expected_home_probability=expected_probability,
                reproduced_home_probability=home_candidate.model_probability,
                absolute_difference=difference,
                passed=difference <= Decimal("0.000001"),
            )
        )

    candidate_tuple = tuple(candidates)
    inference = MoneylineInferenceResult(
        candidates=candidate_tuple,
        admissions=(),
        model_artifact_fingerprint=model_artifact.fingerprint(),
        candidate_set_fingerprint=_candidate_set_fingerprint(candidate_tuple),
    )
    return MoneylineWalkForwardReplayResult(
        fold=fold,
        model=model,
        model_artifact=model_artifact,
        inference=inference,
        parity_rows=tuple(parity_rows),
    )


def replay_multifold_historical_candidates(
    *,
    folds: Sequence[MoneylineWalkForwardFold],
    historical_results_bytes: bytes,
    historical_provenance_bytes: bytes,
    reconstructed_models: Mapping[str, ReconstructedWalkForwardModel] | None = None,
) -> MultifoldHistoricalCandidateReplayResult:
    """Replay the next contiguous P13 folds through the existing contracts.

    When ``reconstructed_models`` is supplied, its verified P20A state is used
    directly.  This is the same explicit-state replay pattern already exposed
    by the single-fold CLI and keeps this bounded migration runnable without
    installing dependencies.  If a fold is not present in the mapping, the
    existing P20A reconstruction function is used.
    """

    ordered_folds = _validate_contiguous_folds(folds)
    result_by_game, provenance = _validate_historical_inputs(
        folds=ordered_folds,
        historical_results_bytes=historical_results_bytes,
        historical_provenance_bytes=historical_provenance_bytes,
    )

    fold_summaries: list[FoldReplaySummary] = []
    fold_feedback_results: list[tuple[str, PredictionFeedbackLedgerResult]] = []
    all_feedback_rows: list[PredictionFeedbackRow] = []
    for fold in ordered_folds:
        model = (
            reconstructed_models.get(fold.fold_id)
            if reconstructed_models is not None
            else None
        )
        if model is None:
            model = reconstruct_moneyline_walk_forward_model(fold)
        if model.fold_id != fold.fold_id:
            raise ValueError(f"P21B model fold mismatch for {fold.fold_id}")
        if model.train_size != fold.training_row_count:
            raise ValueError(f"P21B model training size mismatch for {fold.fold_id}")
        replay = _replay_fold_with_per_game_timestamps(fold, model)
        if not replay.parity_passed:
            raise ValueError(f"P21B legacy parity failed for {fold.fold_id}")
        fold_result_bytes, _ = _fold_result_bytes(
            fold=fold,
            result_by_game=result_by_game,
            provenance=provenance,
        )
        feedback_result, stage = _run_p15c_to_p17(
            fold=fold,
            replay=replay,
            historical_results_bytes=fold_result_bytes,
        )
        fold_feedback_results.append((fold.fold_id, feedback_result))
        all_feedback_rows.extend(feedback_result.feedback_rows)
        fold_summaries.append(
            FoldReplaySummary(
                fold_id=fold.fold_id,
                train_as_of=fold.train_as_of,
                validation_start=fold.validation_start,
                validation_end=fold.validation_end,
                training_row_count=fold.training_row_count,
                prediction_row_count=fold.prediction_row_count,
                prediction_game_ids=tuple(row.game_id for row in fold.prediction_rows),
                fold_fingerprint=fold.fingerprint(),
                model_fingerprint=replay.model.fingerprint(),
                model_artifact_fingerprint=replay.model_artifact.fingerprint(),
                parity_rows=tuple(row.to_projection() for row in replay.parity_rows),
                max_absolute_difference=str(replay.max_absolute_difference),
                parity_passed=replay.parity_passed,
                **stage,
                p17_feedback_row_count=feedback_result.prediction_row_count,
                p17_feedback_ledger_fingerprint=feedback_result.feedback_ledger_fingerprint,
            )
        )

    ordered_feedback_rows = tuple(
        sorted(all_feedback_rows, key=lambda row: row.prediction_observation_id)
    )
    feedback_result = _aggregate_feedback_result(
        fold_results=fold_feedback_results,
        feedback_rows=ordered_feedback_rows,
    )
    feedback_jsonl = render_feedback_jsonl(feedback_result).encode("utf-8")
    feedback_rows_by_id = {
        row["prediction_observation_id"]: row
        for row in json.loads("[" + ",".join(
            line for line in feedback_jsonl.decode("utf-8").splitlines() if line
        ) + "]")
    }
    assessments = tuple(
        sorted(
            (
                assess_prediction_learning_eligibility(
                    row,
                    synthetic_results=False,
                    source_feedback_ledger_fingerprint=(
                        feedback_result.feedback_ledger_fingerprint
                    ),
                )
                for row in feedback_rows_by_id.values()
            ),
            key=lambda assessment: assessment.prediction_observation_id,
        )
    )
    assessment_rows = [
        _assessment_projection(
            assessment,
            source_feedback_ledger_fingerprint=feedback_result.feedback_ledger_fingerprint,
        )
        for assessment in assessments
    ]
    candidate_rows = tuple(
        sorted(
            (
                _candidate_projection(
                    feedback_rows_by_id[assessment.prediction_observation_id],
                    assessment,
                    source_feedback_ledger_fingerprint=(
                        feedback_result.feedback_ledger_fingerprint
                    ),
                )
                for assessment in assessments
                if assessment.status == ELIGIBLE
            ),
            key=lambda row: row["candidate_id"],
        )
    )

    return MultifoldHistoricalCandidateReplayResult(
        folds=tuple(fold_summaries),
        feedback_result=feedback_result,
        feedback_rows=ordered_feedback_rows,
        assessments=assessments,
        candidates=candidate_rows,
        historical_result_count=len(result_by_game),
        historical_results_sha256=_sha256(historical_results_bytes),
        historical_provenance_sha256=_sha256(historical_provenance_bytes),
        historical_provenance=provenance,
        membership_sha256=_membership_sha256(ordered_folds),
        result_rows_sha256=_result_rows_sha256(
            folds=ordered_folds,
            result_by_game=result_by_game,
            provenance=provenance,
        ),
        assessment_semantic_fingerprint=_semantic_fingerprint(assessment_rows),
        candidate_semantic_fingerprint=_semantic_fingerprint(candidate_rows),
        claims=dict(P21B_CLAIMS),
    )


def load_multifold_folds(paths: Sequence[str | Path]) -> tuple[MoneylineWalkForwardFold, ...]:
    """Load fold fixtures in a deterministic path-independent order."""

    folds = tuple(load_moneyline_walk_forward_fold(Path(path)) for path in paths)
    return _validate_contiguous_folds(folds)


__all__ = (
    "FoldReplaySummary",
    "MultifoldHistoricalCandidateReplayResult",
    "P21B_CLAIMS",
    "P20B_HISTORICAL_RUNTIME_COMPLIANCE",
    "P21B_RECONSTRUCTED_MODELS_SCHEMA_VERSION",
    "P21B_SCHEMA_VERSION",
    "load_multifold_folds",
    "load_multifold_reconstructed_models",
    "replay_multifold_historical_candidates",
)
