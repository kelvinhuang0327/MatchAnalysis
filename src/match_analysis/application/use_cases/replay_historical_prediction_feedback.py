"""Replay committed P20A predictions through the P15C-P17A contracts.

This use case consumes already-committed P20A candidate observations and a
committed, non-synthetic historical-result fixture. It adapts the P20A
schedule identity into the existing P15C admission boundary, then delegates
snapshot, attachment, evaluation, and feedback semantics to the existing
P15C, P16A, P16B, and P17A use cases.

No model fitting, provider access, network access, database write, or scoring
policy is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from typing import Any, Mapping

from ...application.use_cases.admitted_prediction_observation_artifacts import (
    render_admitted_observations_jsonl,
    render_snapshot_report_markdown,
    render_snapshot_summary_json,
)
from ...application.use_cases.attach_final_results_to_admitted_predictions import (
    FinalResultAttachmentResult,
    attach_final_results_to_admitted_predictions,
)
from ...application.use_cases.final_result_attachment_artifacts import (
    render_attachment_report_markdown,
    render_attachment_summary_json,
    render_attachments_jsonl,
)
from ...application.use_cases.build_admitted_prediction_observation_snapshot import (
    AdmittedPredictionObservationSnapshotResult,
    build_admitted_prediction_observation_snapshot,
)
from ...application.use_cases.build_prediction_evaluation_scorecard import (
    PredictionEvaluationScorecardResult,
    build_prediction_evaluation_scorecard,
)
from ...application.use_cases.build_prediction_feedback_ledger import (
    PredictionFeedbackLedgerResult,
    build_prediction_feedback_ledger,
)
from ...application.use_cases.prediction_evaluation_artifacts import (
    render_evaluation_report_markdown,
    render_evaluation_summary_json,
    render_evaluations_jsonl,
)
from ...application.use_cases.prediction_feedback_artifacts import (
    render_feedback_jsonl,
)
from ...application.use_cases.prospective_prediction_admission_artifacts import (
    render_results_jsonl,
    render_summary_json as render_admission_summary_json,
)
from ...application.use_cases.run_prospective_prediction_admission_workflow import (
    ProspectivePredictionAdmissionWorkflowResult,
    compute_result_set_fingerprint,
)
from ...baseball.domain.game import BaseballGame
from ...baseball.domain.prediction_admission import (
    PredictionAdmissionResult,
    ProspectivePredictionCandidate,
    ScheduleCandidateProjection,
    admit_prospective_prediction,
)
from ...baseball.domain.pregame_eligibility import (
    BEFORE_SCHEDULED_START,
    ELIGIBLE,
    SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION,
    SchedulePregameEligibilityDecision,
    SchedulePregameEligibilitySet,
    compute_schedule_pregame_eligibility_set_fingerprint,
)
from ...baseball.domain.schedule_game_materialization import (
    ScheduleBaseballGameMaterialization,
)
from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...core.identity import MatchIdentity
from ...core.time import UtcTimestamp


P20B_SCHEMA_VERSION = "p20b.first_non_synthetic_historical_feedback.v1"
P20A_REPLAY_SCHEMA_VERSION = "p20a.moneyline_walk_forward_replay.v1"
P20A_FOLD_SCHEMA_VERSION = "p20a.moneyline_walk_forward_fold.v1"

EXPECTED_LEGACY_REPOSITORY = "/Users/kelvin" + "/Kelvin-WorkSpace/Betting-pool"
EXPECTED_LEGACY_COMMIT = "03b2fcf4de1a13ee9929afcef803d61955c9f41b"
EXPECTED_LEGACY_TREE = "56a849bc68234db63da7a38f1643fa664217c5d0"
EXPECTED_LEGACY_ARCHIVE_PATH = "data/mlb_2025/gl2025.zip"
EXPECTED_LEGACY_ARCHIVE_BLOB = (
    "3e9b08be2530870f38c474db316de6de58b1b381"
)
EXPECTED_LEGACY_ARCHIVE_SHA256 = (
    "957a7cff15cf7926889749c3ef99802ef030ee1b5f7c112b06ba5cb810df5f76"
)
EXPECTED_LEGACY_MEMBER = "gl2025.txt"
EXPECTED_RESULT_VERIFIED_AT = "2026-03-12T06:29:35.016973Z"

P20B_CLAIMS = {
    "db_written": False,
    "deployed": False,
    "deterministic": True,
    "diagnostic": True,
    "historical": True,
    "model_promoted": False,
    "network_called": False,
    "non_synthetic": True,
    "odds_used": False,
    "paper_only": True,
    "profitability_claim": False,
    "real_betting_recommendation": False,
    "retraining_performed": False,
    "sample_limited": True,
    "synthetic_results": False,
    "training_authorized": False,
    "training_dataset_claim": False,
}


def _duplicate_rejecting_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {context}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _parse_jsonl(raw: bytes, context: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_duplicate_rejecting_object_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON in {context} line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{context} line {line_number} must be an object")
        rows.append(row)
    return tuple(rows)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    canonical = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return _sha256(canonical)


def _require_string(row: Mapping[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} requires explicit {key}")
    return value


def _candidate_from_projection(
    row: Mapping[str, Any],
) -> ProspectivePredictionCandidate:
    context = "P20A prediction row"
    return ProspectivePredictionCandidate(
        prediction_observation_id=_require_string(
            row, "prediction_observation_id", context
        ),
        source_prediction_id=_require_string(row, "source_prediction_id", context),
        model_id=_require_string(row, "model_id", context),
        market_id=_require_string(row, "market_id", context),
        selection=_require_string(row, "selection", context),
        model_probability=Decimal(_require_string(row, "model_probability", context)),
        line_value=Decimal(_require_string(row, "line_value", context)),
        push_policy=_require_string(row, "push_policy", context),
        provider_namespace=_require_string(row, "provider_namespace", context),
        provider_game_id=_require_string(row, "provider_game_id", context),
        game_number=int(row.get("game_number", 0)),
        source_schedule_observation_id=_require_string(
            row, "source_schedule_observation_id", context
        ),
        prediction_generated_at_utc=_require_string(
            row, "prediction_generated_at_utc", context
        ),
        response_received_at_utc=_require_string(
            row, "response_received_at_utc", context
        ),
        ingested_at_utc=_require_string(row, "ingested_at_utc", context),
    )


def _load_and_validate_p20a(
    *,
    predictions_bytes: bytes,
    reconstruction_bytes: bytes,
    summary_bytes: bytes,
    fold_bytes: bytes,
) -> tuple[
    tuple[ProspectivePredictionCandidate, ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[str, ...],
]:
    prediction_rows = _parse_jsonl(predictions_bytes, "P20A predictions")
    reconstruction = _parse_object(reconstruction_bytes, "P20A reconstruction")
    summary = _parse_object(summary_bytes, "P20A summary")
    fold = _parse_object(fold_bytes, "P20A fold")

    if reconstruction.get("schema_version") != P20A_REPLAY_SCHEMA_VERSION:
        raise ValueError("P20A reconstruction schema version mismatch")
    if summary.get("schema_version") != P20A_REPLAY_SCHEMA_VERSION:
        raise ValueError("P20A summary schema version mismatch")
    if fold.get("schema_version") != P20A_FOLD_SCHEMA_VERSION:
        raise ValueError("P20A fold schema version mismatch")
    if reconstruction.get("fold_id") != fold.get("fold_id"):
        raise ValueError("P20A reconstruction and fold IDs differ")
    if summary.get("fold_id") != fold.get("fold_id"):
        raise ValueError("P20A summary and fold IDs differ")
    if summary.get("replay_row_count") != 2:
        raise ValueError("P20B requires exactly two P20A replay game rows")
    if len(fold.get("prediction_rows", ())) != 2:
        raise ValueError("P20B requires exactly two P20A fold prediction rows")
    if summary.get("model_artifact_fingerprint") != reconstruction.get(
        "model_artifact_fingerprint"
    ):
        raise ValueError("P20A model-artifact fingerprint mismatch")
    if not reconstruction.get("parity_passed"):
        raise ValueError("P20A parity evidence is not passing")

    model = reconstruction.get("reconstructed_model")
    model_artifact = reconstruction.get("model_artifact")
    if not isinstance(model, dict) or not isinstance(model_artifact, dict):
        raise ValueError("P20A reconstruction is missing model provenance")
    model_fingerprint = _require_string(model, "fingerprint", "P20A model")
    p19a_fingerprint = _require_string(
        reconstruction, "model_artifact_fingerprint", "P20A reconstruction"
    )
    if model_artifact.get("schema_version") != "p19a.moneyline_model_artifact.v1":
        raise ValueError("P19A model-artifact schema version mismatch")
    if model_artifact.get("model_id") != model.get("model_id", model_artifact.get("model_id")):
        raise ValueError("P20A model and model-artifact IDs differ")

    fold_rows = fold["prediction_rows"]
    fold_by_game = {
        _require_string(row, "game_id", "P20A fold row"): row
        for row in fold_rows
    }
    candidates = tuple(_candidate_from_projection(row) for row in prediction_rows)
    if len(candidates) != 4:
        raise ValueError(
            "P20A must provide HOME and AWAY candidates for the two replay games"
        )
    seen_selections: dict[str, set[str]] = {}
    for candidate in candidates:
        fold_row = fold_by_game.get(candidate.provider_game_id)
        if fold_row is None:
            raise ValueError(
                f"P20A candidate game is outside the two-row fold: {candidate.provider_game_id}"
            )
        if candidate.selection not in ("HOME", "AWAY"):
            raise ValueError("P20A candidate selection is not HOME/AWAY")
        if candidate.market_id != "moneyline":
            raise ValueError("P20B supports only the P20A moneyline market")
        if candidate.game_number != 1:
            raise ValueError("P20B requires the P20A single-game identity")
        if candidate.source_schedule_observation_id != fold_row[
            "source_schedule_observation_id"
        ]:
            raise ValueError("P20A schedule-observation lineage mismatch")
        seen_selections.setdefault(candidate.provider_game_id, set()).add(
            candidate.selection
        )
    if set(seen_selections) != set(fold_by_game):
        raise ValueError("P20A candidate and fold game sets differ")
    if any(selections != {"HOME", "AWAY"} for selections in seen_selections.values()):
        raise ValueError("P20A must contain one HOME and one AWAY candidate per game")

    return (
        tuple(sorted(candidates, key=lambda candidate: candidate.prediction_observation_id)),
        reconstruction,
        summary,
        fold,
        tuple(sorted(fold_by_game)),
    )


def _build_admission_context(
    *,
    candidates: tuple[ProspectivePredictionCandidate, ...],
    fold: Mapping[str, Any],
) -> tuple[
    tuple[ScheduleCandidateProjection, ...],
    SchedulePregameEligibilitySet,
    datetime,
]:
    fold_by_game = {
        _require_string(row, "game_id", "P20A fold row"): row
        for row in fold["prediction_rows"]
    }
    as_of_utc = min(
        parse_canonical_utc(candidate.prediction_generated_at_utc)
        for candidate in candidates
    )

    materializations: list[ScheduleBaseballGameMaterialization] = []
    schedule_candidates: list[ScheduleCandidateProjection] = []
    for game_id, row in sorted(fold_by_game.items()):
        scheduled_start = parse_canonical_utc(
            _require_string(row, "scheduled_start_utc", "P20A fold row")
        )
        source_observation_id = _require_string(
            row, "source_schedule_observation_id", "P20A fold row"
        )
        identity = MatchIdentity(
            sport="baseball",
            league="MLB",
            season=2025,
            canonical_game_id=game_id,
            home_participant=_require_string(row, "home_team", "P20A fold row"),
            away_participant=_require_string(row, "away_team", "P20A fold row"),
        )
        materialization = ScheduleBaseballGameMaterialization(
            baseball_game=BaseballGame(
                identity=identity,
                scheduled_start=UtcTimestamp(scheduled_start),
            ),
            match_identity=identity,
            source_observation_id=source_observation_id,
            source_raw_payload_sha256=_canonical_sha256(
                {"kind": "p20b.schedule_adapter.raw", "game_id": game_id}
            ),
            source_resolution_set_fingerprint=_canonical_sha256(
                {"kind": "p20b.schedule_adapter.resolution", "game_id": game_id}
            ),
            authority_catalog_fingerprint=_canonical_sha256(
                {"kind": "p20b.schedule_adapter.authority", "game_id": game_id}
            ),
            source_construction_set_fingerprint=_canonical_sha256(
                {"kind": "p20b.schedule_adapter.construction", "game_id": game_id}
            ),
        )
        materializations.append(materialization)
        schedule_candidates.append(
            ScheduleCandidateProjection(
                provider_namespace="MLB_STATS_API",
                provider_game_id=game_id,
                game_number=1,
                source_schedule_observation_id=source_observation_id,
                schedule_as_of_utc=as_of_utc,
                scheduled_start_utc=scheduled_start,
            )
        )

    materialization_rows = [
        {
            "game_id": materialization.match_identity.canonical_game_id,
            "scheduled_start_utc": materialization.baseball_game.scheduled_start.to_iso8601(),
            "source_observation_id": materialization.source_observation_id,
        }
        for materialization in materializations
    ]
    source_materialization_fingerprint = _canonical_sha256(
        {"schema_version": "p20b.schedule_adapter.v1", "rows": materialization_rows}
    )
    decisions = tuple(
        sorted(
            (
                SchedulePregameEligibilityDecision(
                    materialization=materialization,
                    eligibility_status=ELIGIBLE,
                    reason=BEFORE_SCHEDULED_START,
                )
                for materialization in materializations
            ),
            key=lambda decision: decision.materialization.source_observation_id,
        )
    )
    eligibility_fingerprint = compute_schedule_pregame_eligibility_set_fingerprint(
        as_of_utc=as_of_utc,
        source_materialization_set_fingerprint=source_materialization_fingerprint,
        eligible_count=len(decisions),
        ineligible_count=0,
        unresolved_count=0,
        unavailable_count=0,
        authority_missing_count=0,
        eligible_decisions=decisions,
        ineligible_decisions=(),
        unresolved_candidates=(),
        unavailable_chain_keys=(),
        authority_missing_candidates=(),
    )
    eligibility = SchedulePregameEligibilitySet(
        as_of_utc=as_of_utc,
        source_materialization_set_fingerprint=source_materialization_fingerprint,
        eligible_decisions=decisions,
        ineligible_decisions=(),
        unresolved_candidates=(),
        unavailable_chain_keys=(),
        authority_missing_candidates=(),
        eligible_count=len(decisions),
        ineligible_count=0,
        unresolved_count=0,
        unavailable_count=0,
        authority_missing_count=0,
        eligibility_set_fingerprint=eligibility_fingerprint,
        schema_version=SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION,
    )
    return tuple(schedule_candidates), eligibility, as_of_utc


def _load_and_validate_provenance(
    *,
    historical_results_bytes: bytes,
    historical_provenance_bytes: bytes,
    fold: Mapping[str, Any],
) -> HistoricalResultProvenance:
    provenance = _parse_object(historical_provenance_bytes, "P20B result provenance")
    expected_source = {
        "source_repository": EXPECTED_LEGACY_REPOSITORY,
        "source_commit": EXPECTED_LEGACY_COMMIT,
        "source_tree": EXPECTED_LEGACY_TREE,
        "source_archive_path": EXPECTED_LEGACY_ARCHIVE_PATH,
        "source_archive_blob": EXPECTED_LEGACY_ARCHIVE_BLOB,
        "source_archive_sha256": EXPECTED_LEGACY_ARCHIVE_SHA256,
        "source_member": EXPECTED_LEGACY_MEMBER,
        "source_verified_at_utc": EXPECTED_RESULT_VERIFIED_AT,
    }
    for key, expected in expected_source.items():
        if provenance.get(key) != expected:
            raise ValueError(f"P20B historical provenance mismatch for {key}")
    if provenance.get("claims", {}).get("non_synthetic") is not True:
        raise ValueError("P20B provenance must positively establish non-synthetic evidence")

    result_rows = _parse_jsonl(historical_results_bytes, "P20B historical results")
    if len(result_rows) != 2:
        raise ValueError("P20B requires exactly two historical result rows")
    provenance_rows = provenance.get("rows")
    if not isinstance(provenance_rows, list) or len(provenance_rows) != 2:
        raise ValueError("P20B provenance must contain exactly two source rows")
    provenance_by_game = {
        _require_string(row, "canonical_game_id", "P20B provenance row"): row
        for row in provenance_rows
    }
    fold_by_game = {
        _require_string(row, "game_id", "P20A fold row"): row
        for row in fold["prediction_rows"]
    }
    if set(provenance_by_game) != set(fold_by_game):
        raise ValueError("historical result and P20A game identities differ")

    result_by_game: dict[str, dict[str, Any]] = {}
    for result_row in result_rows:
        game_id = _require_string(result_row, "provider_game_id", "P20B result row")
        if game_id in result_by_game:
            raise ValueError(f"duplicate P20B result identity for {game_id}")
        result_by_game[game_id] = result_row
        if result_row.get("provider_namespace") != "MLB_STATS_API":
            raise ValueError("P20B historical results must use MLB_STATS_API")
        if result_row.get("game_number") != 1:
            raise ValueError("P20B historical results must use game_number 1")
        if result_row.get("status") != "FINAL":
            raise ValueError("P20B historical results must be FINAL")
        if result_row.get("result_observed_at_utc") != EXPECTED_RESULT_VERIFIED_AT:
            raise ValueError("P20B result observation time is not pinned provenance time")
        source_row = provenance_by_game.get(game_id)
        if source_row is None:
            raise ValueError(f"P20B result game is not in provenance: {game_id}")
        for score_field in ("home_score", "away_score"):
            if result_row.get(score_field) != source_row.get(score_field):
                raise ValueError(f"P20B result {score_field} differs from source evidence")
        expected_home_win = bool(fold_by_game[game_id].get("target_home_win"))
        observed_home_win = int(result_row["home_score"]) > int(result_row["away_score"])
        if observed_home_win != expected_home_win:
            raise ValueError(f"P20B result winner disagrees with P20A historical target for {game_id}")

    if set(result_by_game) != set(fold_by_game):
        raise ValueError("P20B historical result game set is incomplete")

    return HistoricalResultProvenance(
        source_repository=provenance["source_repository"],
        source_commit=provenance["source_commit"],
        source_tree=provenance["source_tree"],
        source_archive_path=provenance["source_archive_path"],
        source_archive_blob=provenance["source_archive_blob"],
        source_archive_sha256=provenance["source_archive_sha256"],
        source_member=provenance["source_member"],
        source_verified_at_utc=provenance["source_verified_at_utc"],
        rows=tuple(sorted(provenance_rows, key=lambda row: row["canonical_game_id"])),
    )


@dataclass(frozen=True, slots=True)
class HistoricalResultProvenance:
    source_repository: str
    source_commit: str
    source_tree: str
    source_archive_path: str
    source_archive_blob: str
    source_archive_sha256: str
    source_member: str
    source_verified_at_utc: str
    rows: tuple[dict[str, Any], ...]

    def to_projection(self) -> dict[str, Any]:
        return {
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "source_archive_path": self.source_archive_path,
            "source_archive_blob": self.source_archive_blob,
            "source_archive_sha256": self.source_archive_sha256,
            "source_member": self.source_member,
            "source_verified_at_utc": self.source_verified_at_utc,
            "rows": list(self.rows),
        }


@dataclass(frozen=True, slots=True)
class HistoricalFeedbackReplayResult:
    p20a_prediction_sha256: str
    p20a_reconstruction_sha256: str
    p20a_summary_sha256: str
    p20a_fold_sha256: str
    p20a_fold_id: str
    p20a_model_fingerprint: str
    p19a_model_artifact_fingerprint: str
    replay_game_ids: tuple[str, ...]
    historical_results_sha256: str
    historical_provenance_sha256: str
    historical_provenance: HistoricalResultProvenance
    admission_workflow: ProspectivePredictionAdmissionWorkflowResult
    snapshot_result: AdmittedPredictionObservationSnapshotResult
    attachment_result: FinalResultAttachmentResult
    evaluation_result: PredictionEvaluationScorecardResult
    feedback_result: PredictionFeedbackLedgerResult
    claims: dict[str, bool]


def replay_historical_prediction_feedback(
    *,
    p20a_predictions_bytes: bytes,
    p20a_reconstruction_bytes: bytes,
    p20a_summary_bytes: bytes,
    p20a_fold_bytes: bytes,
    historical_results_bytes: bytes,
    historical_provenance_bytes: bytes,
) -> HistoricalFeedbackReplayResult:
    """Run exactly two P20A replay games through P15C, P16A, P16B, and P17A."""

    (
        candidates,
        reconstruction,
        summary,
        fold,
        replay_game_ids,
    ) = _load_and_validate_p20a(
        predictions_bytes=p20a_predictions_bytes,
        reconstruction_bytes=p20a_reconstruction_bytes,
        summary_bytes=p20a_summary_bytes,
        fold_bytes=p20a_fold_bytes,
    )
    provenance = _load_and_validate_provenance(
        historical_results_bytes=historical_results_bytes,
        historical_provenance_bytes=historical_provenance_bytes,
        fold=fold,
    )
    schedule_candidates, eligibility, schedule_as_of = _build_admission_context(
        candidates=candidates,
        fold=fold,
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
        reasons = [admission.reason for admission in admissions]
        raise ValueError(f"P20B P15C admission failed closed: {reasons}")

    admission_workflow = ProspectivePredictionAdmissionWorkflowResult(
        results=admissions,
        schedule_as_of_utc=schedule_as_of,
        schedule_candidates=schedule_candidates,
        schedule_pregame_eligibility=eligibility,
        admitted_count=len(admissions),
        rejected_count=0,
        result_set_fingerprint=compute_result_set_fingerprint(admissions),
    )
    admission_results_bytes = render_results_jsonl(admissions).encode("utf-8")
    admission_summary_bytes = render_admission_summary_json(
        admission_workflow,
        {
            "p20a_fold_sha256": _sha256(p20a_fold_bytes),
            "p20a_predictions_sha256": _sha256(p20a_predictions_bytes),
            "p20a_reconstruction_sha256": _sha256(p20a_reconstruction_bytes),
            "p20a_summary_sha256": _sha256(p20a_summary_bytes),
        },
    ).encode("utf-8")

    snapshot_result = build_admitted_prediction_observation_snapshot(
        results_bytes=admission_results_bytes,
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
        raise ValueError("P20B exact historical result attachment did not attach every candidate")
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
        raise ValueError("P20B feedback row count does not preserve P20A candidates")

    return HistoricalFeedbackReplayResult(
        p20a_prediction_sha256=_sha256(p20a_predictions_bytes),
        p20a_reconstruction_sha256=_sha256(p20a_reconstruction_bytes),
        p20a_summary_sha256=_sha256(p20a_summary_bytes),
        p20a_fold_sha256=_sha256(p20a_fold_bytes),
        p20a_fold_id=_require_string(fold, "fold_id", "P20A fold"),
        p20a_model_fingerprint=_require_string(
            reconstruction["reconstructed_model"], "fingerprint", "P20A model"
        ),
        p19a_model_artifact_fingerprint=_require_string(
            reconstruction, "model_artifact_fingerprint", "P20A reconstruction"
        ),
        replay_game_ids=replay_game_ids,
        historical_results_sha256=_sha256(historical_results_bytes),
        historical_provenance_sha256=_sha256(historical_provenance_bytes),
        historical_provenance=provenance,
        admission_workflow=admission_workflow,
        snapshot_result=snapshot_result,
        attachment_result=attachment_result,
        evaluation_result=evaluation_result,
        feedback_result=feedback_result,
        claims=dict(P20B_CLAIMS),
    )
