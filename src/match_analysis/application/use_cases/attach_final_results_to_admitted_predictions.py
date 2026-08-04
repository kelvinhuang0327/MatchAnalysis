"""Deterministic final result attachment to admitted prediction observations.

Attaches explicit synthetic FINAL result observations to prediction observations
using exact (provider_namespace, provider_game_id, game_number) identity matching.

Does not call any provider, use the network, write to a database, admit legacy
predictions, calculate odds/payout/ROI/EV/Kelly/staking, retrain a model, or
deploy anything.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...baseball.domain.final_result_observation import (
    FinalResultObservation,
)


ATTACHMENT_SCHEMA_VERSION = "p16a.prediction_final_result_attachment.v1"


@dataclass(frozen=True, slots=True)
class PredictionFinalResultAttachmentRow:
    """One row in the deterministic attachment output."""

    prediction_observation_id: str
    source_snapshot_row_fingerprint: str
    result_observation_id: str | None
    provider_namespace: str
    provider_game_id: str
    game_number: int
    attachment_status: str  # ATTACHED or REJECTED
    rejection_reason: str | None
    scheduled_start_utc: str
    result_observed_at_utc: str | None
    home_score: int | None
    away_score: int | None
    actual_winner: str | None
    selection: str
    is_correct: bool | None
    attachment_row_fingerprint: str


def _compute_attachment_row_fingerprint(
    prediction_observation_id: str,
    source_snapshot_row_fingerprint: str,
    result_observation_id: str | None,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    attachment_status: str,
    rejection_reason: str | None,
    scheduled_start_utc: str,
    result_observed_at_utc: str | None,
    home_score: int | None,
    away_score: int | None,
    actual_winner: str | None,
    selection: str,
    is_correct: bool | None,
) -> str:
    """Compute a deterministic fingerprint for an attachment row."""
    payload = {
        "actual_winner": actual_winner,
        "attachment_status": attachment_status,
        "away_score": away_score,
        "game_number": game_number,
        "home_score": home_score,
        "is_correct": is_correct,
        "prediction_observation_id": prediction_observation_id,
        "provider_game_id": provider_game_id,
        "provider_namespace": provider_namespace,
        "rejection_reason": rejection_reason,
        "result_observation_id": result_observation_id,
        "result_observed_at_utc": result_observed_at_utc,
        "scheduled_start_utc": scheduled_start_utc,
        "selection": selection,
        "source_snapshot_row_fingerprint": source_snapshot_row_fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _make_rejected_row(
    prediction_observation_id: str,
    source_snapshot_row_fingerprint: str,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    scheduled_start_utc: str,
    selection: str,
    rejection_reason: str,
) -> PredictionFinalResultAttachmentRow:
    """Create a REJECTED attachment row with no partial result fields."""
    fp = _compute_attachment_row_fingerprint(
        prediction_observation_id=prediction_observation_id,
        source_snapshot_row_fingerprint=source_snapshot_row_fingerprint,
        result_observation_id=None,
        provider_namespace=provider_namespace,
        provider_game_id=provider_game_id,
        game_number=game_number,
        attachment_status="REJECTED",
        rejection_reason=rejection_reason,
        scheduled_start_utc=scheduled_start_utc,
        result_observed_at_utc=None,
        home_score=None,
        away_score=None,
        actual_winner=None,
        selection=selection,
        is_correct=None,
    )
    return PredictionFinalResultAttachmentRow(
        prediction_observation_id=prediction_observation_id,
        source_snapshot_row_fingerprint=source_snapshot_row_fingerprint,
        result_observation_id=None,
        provider_namespace=provider_namespace,
        provider_game_id=provider_game_id,
        game_number=game_number,
        attachment_status="REJECTED",
        rejection_reason=rejection_reason,
        scheduled_start_utc=scheduled_start_utc,
        result_observed_at_utc=None,
        home_score=None,
        away_score=None,
        actual_winner=None,
        selection=selection,
        is_correct=None,
        attachment_row_fingerprint=fp,
    )


@dataclass(frozen=True, slots=True)
class FinalResultAttachmentResult:
    """Immutable result of deterministic final result attachment."""

    schema_version: str
    source_snapshot_sha256: str
    source_snapshot_summary_sha256: str
    source_snapshot_fingerprint: str
    result_input_sha256: str
    source_prediction_count: int
    final_result_observation_count: int
    attached_count: int
    rejected_count: int
    rejection_reason_counts: dict[str, int]
    correct_count: int
    incorrect_count: int
    descriptive_accuracy: float | None
    attachment_rows: tuple[PredictionFinalResultAttachmentRow, ...]
    attachment_set_fingerprint: str
    claims: dict[str, bool]


def _compute_attachment_set_fingerprint(
    rows: tuple[PredictionFinalResultAttachmentRow, ...],
) -> str:
    """Compute deterministic fingerprint over the full attachment set."""
    parts = []
    for row in rows:
        parts.append(
            f"{row.prediction_observation_id}:{row.attachment_row_fingerprint}\n"
        )
    combined = "".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def attach_final_results_to_admitted_predictions(
    *,
    snapshot_bytes: bytes,
    summary_bytes: bytes,
    final_results_bytes: bytes,
) -> FinalResultAttachmentResult:
    """Attach final result observations to admitted prediction observations.

    Validates source artifacts, performs exact identity matching, temporal
    validation, and moneyline correctness derivation.

    Raises ValueError on structural failures (malformed JSON, duplicate keys).
    Per-row failures produce REJECTED rows, not exceptions.
    """
    from .build_admitted_prediction_observation_snapshot import (
        _compute_snapshot_fingerprint,
        _validate_json_no_duplicate_keys as _validate_snapshot_json,
        AdmittedPredictionObservationRow,
    )
    from ...baseball.domain.final_result_observation import (
        load_final_result_observations,
    )

    # Compute source hashes
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    result_input_sha256 = hashlib.sha256(final_results_bytes).hexdigest()

    # Parse and validate summary
    summary_text = summary_bytes.decode("utf-8")
    summary_dict = json.loads(summary_text)

    # Verify snapshot SHA-256 matches summary
    expected_snapshot_sha256 = summary_dict.get("admitted_observations_jsonl_sha256")
    if snapshot_sha256 != expected_snapshot_sha256:
        raise ValueError(
            f"Snapshot SHA-256 mismatch: computed {snapshot_sha256}, "
            f"summary expects {expected_snapshot_sha256}"
        )

    source_snapshot_fingerprint = summary_dict["snapshot_fingerprint"]

    # Parse snapshot JSONL
    snapshot_text = snapshot_bytes.decode("utf-8")
    snapshot_lines = [line for line in snapshot_text.splitlines() if line.strip()]
    snapshot_rows: list[dict[str, Any]] = []
    for i, line in enumerate(snapshot_lines):
        row = _validate_snapshot_json(line, i)
        snapshot_rows.append(row)

    # Verify snapshot row count matches summary
    expected_row_count = summary_dict.get("snapshot_row_count")
    if len(snapshot_rows) != expected_row_count:
        raise ValueError(
            f"Snapshot row count mismatch: {len(snapshot_rows)} vs "
            f"summary {expected_row_count}"
        )

    # Verify snapshot fingerprint by recomputation
    typed_rows = []
    for i, row in enumerate(snapshot_rows):
        typed_rows.append(
            AdmittedPredictionObservationRow(
                prediction_observation_id=row["prediction_observation_id"],
                source_result_row_fingerprint=row["source_result_row_fingerprint"],
                observation=row["observation"],
                snapshot_row_fingerprint=row["snapshot_row_fingerprint"],
            )
        )
    sorted_typed = tuple(sorted(typed_rows, key=lambda r: r.prediction_observation_id))
    recomputed_fp = _compute_snapshot_fingerprint(sorted_typed)
    if recomputed_fp != source_snapshot_fingerprint:
        raise ValueError(
            f"Snapshot fingerprint mismatch: recomputed {recomputed_fp}, "
            f"expected {source_snapshot_fingerprint}"
        )

    # Load final result observations (validates, checks duplicates)
    final_results = load_final_result_observations(final_results_bytes)

    # Build lookup by identity key
    result_lookup: dict[tuple[str, str, int], FinalResultObservation] = {}
    for obs in final_results:
        key = (obs.provider_namespace, obs.provider_game_id, obs.game_number)
        result_lookup[key] = obs

    # Process each snapshot row
    unsorted_attachment_rows: list[PredictionFinalResultAttachmentRow] = []

    for snap_row in sorted_typed:
        obs = snap_row.observation
        pred_obs_id = snap_row.prediction_observation_id
        snap_fp = snap_row.snapshot_row_fingerprint
        provider_ns = obs["provider_namespace"]
        provider_gid = obs["provider_game_id"]
        game_num = obs["game_number"]
        scheduled_start = obs["scheduled_start_utc"]
        market_id = obs["market_id"]
        selection = obs["selection"]

        identity_key = (provider_ns, provider_gid, game_num)

        # Check market_id and selection support
        if market_id != "moneyline":
            unsorted_attachment_rows.append(
                _make_rejected_row(
                    pred_obs_id, snap_fp, provider_ns, provider_gid,
                    game_num, scheduled_start, selection,
                    "UNSUPPORTED_RESULT_EVALUATION_CONTRACT",
                )
            )
            continue

        if selection not in ("HOME", "AWAY"):
            unsorted_attachment_rows.append(
                _make_rejected_row(
                    pred_obs_id, snap_fp, provider_ns, provider_gid,
                    game_num, scheduled_start, selection,
                    "UNSUPPORTED_RESULT_EVALUATION_CONTRACT",
                )
            )
            continue

        # Look up matching result
        result_obs = result_lookup.get(identity_key)
        if result_obs is None:
            unsorted_attachment_rows.append(
                _make_rejected_row(
                    pred_obs_id, snap_fp, provider_ns, provider_gid,
                    game_num, scheduled_start, selection,
                    "MISSING_FINAL_RESULT_OBSERVATION",
                )
            )
            continue

        # Temporal validation: result_observed_at_utc > scheduled_start_utc
        result_time = parse_canonical_utc(result_obs.result_observed_at_utc)
        scheduled_time = parse_canonical_utc(scheduled_start)
        if result_time <= scheduled_time:
            unsorted_attachment_rows.append(
                _make_rejected_row(
                    pred_obs_id, snap_fp, provider_ns, provider_gid,
                    game_num, scheduled_start, selection,
                    "RESULT_NOT_AFTER_SCHEDULED_START",
                )
            )
            continue

        # Derive actual winner
        if result_obs.home_score > result_obs.away_score:
            actual_winner = "HOME"
        else:
            actual_winner = "AWAY"

        is_correct = selection == actual_winner

        fp = _compute_attachment_row_fingerprint(
            prediction_observation_id=pred_obs_id,
            source_snapshot_row_fingerprint=snap_fp,
            result_observation_id=result_obs.result_observation_id,
            provider_namespace=provider_ns,
            provider_game_id=provider_gid,
            game_number=game_num,
            attachment_status="ATTACHED",
            rejection_reason=None,
            scheduled_start_utc=scheduled_start,
            result_observed_at_utc=result_obs.result_observed_at_utc,
            home_score=result_obs.home_score,
            away_score=result_obs.away_score,
            actual_winner=actual_winner,
            selection=selection,
            is_correct=is_correct,
        )

        unsorted_attachment_rows.append(
            PredictionFinalResultAttachmentRow(
                prediction_observation_id=pred_obs_id,
                source_snapshot_row_fingerprint=snap_fp,
                result_observation_id=result_obs.result_observation_id,
                provider_namespace=provider_ns,
                provider_game_id=provider_gid,
                game_number=game_num,
                attachment_status="ATTACHED",
                rejection_reason=None,
                scheduled_start_utc=scheduled_start,
                result_observed_at_utc=result_obs.result_observed_at_utc,
                home_score=result_obs.home_score,
                away_score=result_obs.away_score,
                actual_winner=actual_winner,
                selection=selection,
                is_correct=is_correct,
                attachment_row_fingerprint=fp,
            )
        )

    # Sort by prediction_observation_id
    sorted_rows = tuple(
        sorted(unsorted_attachment_rows, key=lambda r: r.prediction_observation_id)
    )

    attachment_set_fp = _compute_attachment_set_fingerprint(sorted_rows)

    # Compute counts
    attached_count = sum(1 for r in sorted_rows if r.attachment_status == "ATTACHED")
    rejected_count = sum(1 for r in sorted_rows if r.attachment_status == "REJECTED")
    correct_count = sum(1 for r in sorted_rows if r.is_correct is True)
    incorrect_count = sum(1 for r in sorted_rows if r.is_correct is False)

    rejection_reason_counts: dict[str, int] = {}
    for r in sorted_rows:
        if r.rejection_reason is not None:
            rejection_reason_counts[r.rejection_reason] = (
                rejection_reason_counts.get(r.rejection_reason, 0) + 1
            )

    descriptive_accuracy: float | None = None
    if attached_count > 0:
        descriptive_accuracy = round(correct_count / attached_count, 6)

    claims = {
        "db_written": False,
        "deployed": False,
        "legacy_rows_admitted": False,
        "network_called": False,
        "odds_used": False,
        "profitability_claim": False,
        "provider_called": False,
        "synthetic_results": True,
    }

    return FinalResultAttachmentResult(
        schema_version=ATTACHMENT_SCHEMA_VERSION,
        source_snapshot_sha256=snapshot_sha256,
        source_snapshot_summary_sha256=summary_sha256,
        source_snapshot_fingerprint=source_snapshot_fingerprint,
        result_input_sha256=result_input_sha256,
        source_prediction_count=len(sorted_typed),
        final_result_observation_count=len(final_results),
        attached_count=attached_count,
        rejected_count=rejected_count,
        rejection_reason_counts=rejection_reason_counts,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        descriptive_accuracy=descriptive_accuracy,
        attachment_rows=sorted_rows,
        attachment_set_fingerprint=attachment_set_fp,
        claims=claims,
    )
