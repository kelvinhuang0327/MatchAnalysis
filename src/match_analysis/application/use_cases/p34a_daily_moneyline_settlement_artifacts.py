"""Deterministic P34A daily Moneyline settlement artifacts and report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from .admitted_prediction_observation_artifacts import (
    render_admitted_observations_jsonl,
)
from .final_result_attachment_artifacts import render_attachments_jsonl
from .prediction_evaluation_artifacts import render_evaluations_jsonl
from .prediction_feedback_artifacts import render_feedback_jsonl
from .settle_daily_moneyline_paper_run import (
    P34A_OPERATION,
    P34A_SCHEMA_VERSION,
    P34ASettlementResult,
)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(dict(row)) for row in rows)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _metric(value: float | None) -> str:
    return "N/A" if value is None else str(value)


def render_p34a_report(result: P34ASettlementResult) -> bytes:
    """Render the human-readable prediction-versus-result report."""

    p33a = result.p33a
    summary = p33a.summary
    authority = result.result_authority
    lines = [
        "# P34A Daily Moneyline Paper Settlement & Feedback",
        "",
        "This is a deterministic paper-only result attachment and feedback report.",
        "The daily sample is small and is not proof of model profitability.",
        "",
        "## Frozen P33A Authority",
        "",
        f"- **P33A Run ID**: `{p33a.run_manifest['run_id']}`",
        f"- **P33A Bundle Fingerprint**: `{summary['bundle_fingerprint']}`",
        f"- **P33A Analysis Set Fingerprint**: `{p33a.analysis_set_fingerprint}`",
        f"- **P33A Analysis JSONL SHA-256**: `{p33a.analysis_jsonl_sha256}`",
        f"- **P33A Pregame Invariance**: `PASS` (source bytes were read-only)",
        f"- **Target Date**: `{summary['target_date']}`",
        "",
        "## Structural Accounting",
        "",
        "| Quantity | Count |",
        "| --- | ---: |",
        f"| Official games | {summary['official_raw_game_count']} |",
        f"| TSL/source rows | {summary['source_records_received']} |",
        f"| Qualified source observations | {summary['observations_qualified']} |",
        f"| Rejected source observations | {summary['observations_rejected']} |",
        f"| P33A analysis rows | {len(p33a.analysis_rows)} |",
        f"| Complete P33A predictions eligible for settlement | {len(p33a.prediction_rows)} |",
        f"| Structural/non-prediction rows excluded | {len(result.structural_rows)} |",
        "",
        "Structural and rejected rows remain separate and produce no evaluation.",
        "",
        "## Official Result Authority",
        "",
        f"- **Source**: `{authority.get('source', 'UNKNOWN')}`",
        f"- **Observed At (UTC)**: `{authority.get('observed_at_utc', 'UNKNOWN')}`",
        f"- **Target Game Count**: {authority.get('target_game_count', 'UNKNOWN')}",
        f"- **Final Result Count**: {authority.get('final_result_count', 0)}",
        f"- **Non-final Target Count**: {authority.get('non_final_target_count', 'UNKNOWN')}",
        f"- **Missing Target Count**: {authority.get('missing_target_count', 'UNKNOWN')}",
        f"- **Missing Settleable Result Count**: {authority.get('missing_settleable_result_count', 'UNKNOWN')}",
        f"- **Missing Settleable Result Check**: `{authority.get('missing_settleable_result_check', 'UNKNOWN')}`",
        f"- **Duplicate Result Identity Check**: `{authority.get('duplicate_result_identity_check', 'UNKNOWN')}`",
        f"- **Conflicting Result Check**: `{authority.get('conflicting_result_check', 'UNKNOWN')}`",
        f"- **Non-final Result Check**: `{authority.get('non_final_result_check', 'UNKNOWN')}`",
        f"- **Result Authority Fingerprint**: `{authority['result_authority_fingerprint']}`",
        f"- **Final-result input check**: `PASS` (only `FINAL` observations are attachable)",
        "",
        "## Settlement Summary",
        "",
        f"- **Settlement Status**: `{authority.get('settlement_status', 'UNKNOWN')}`",
        f"- **Settled Predictions**: {result.settled_count}",
        f"- **Correct**: {result.evaluation_result.correct_count}",
        f"- **Incorrect**: {result.evaluation_result.incorrect_count}",
        f"- **Unresolved**: {result.unresolved_count}",
        f"- **Descriptive Accuracy**: {_metric(result.accuracy)}",
        f"- **Mean Selected-side Probability**: {_metric(result.mean_selected_side_probability)}",
        f"- **Brier Score**: {_metric(result.brier_score)}",
        f"- **Feedback Ledger Fingerprint**: `{result.feedback_result.feedback_ledger_fingerprint}`",
        "",
        "## Prediction Versus Result",
        "",
        "| Game | Predicted Side | Model Probability | Market Price | Frozen Edge | Final Score | Actual Winner | Correct | Evaluation Status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    if not result.prediction_result_rows:
        lines.append(
            "| *No settleable predictions* | — | — | — | — | — | — | — | NOT_EVALUATED |"
        )
    else:
        for row in result.prediction_result_rows:
            score = (
                f"{row['home_score']}-{row['away_score']}"
                if row["home_score"] is not None
                else "—"
            )
            winner = row["actual_winner"] or "—"
            correct = (
                str(row["is_correct"])
                if row["is_correct"] is not None
                else "—"
            )
            evaluation_status = row["evaluation_status"]
            if row["unresolved_reason"]:
                evaluation_status = f"{evaluation_status}: {row['unresolved_reason']}"
            lines.append(
                f"| {row['home_team']} vs {row['away_team']} ({row['provider_game_id']}) "
                f"| {row['predicted_side']} "
                f"| {row['model_probability']} "
                f"| {row['market_price']} "
                f"| {row['market_edge']} "
                f"| {score} "
                f"| {winner} "
                f"| {correct} "
                f"| {evaluation_status} |"
            )

    lines.extend(
        [
            "",
            "## Determinism and Safety",
            "",
            f"- **Offline Replay**: `{'PASS' if result.offline_replay_verified else 'NOT_RUN'}`",
            f"- **Network Called For This Materialization**: `{result.network_called}`",
            "- Pregame P33A prediction, market-price, edge, timestamp, source-fingerprint, and run-identity fields were not rewritten.",
            "- No model retraining, model promotion, scheduler activation, or real betting occurred.",
            "- This report is a small daily sample and must not be used as a profitability conclusion.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def render_p34a_artifacts(result: P34ASettlementResult) -> dict[str, bytes]:
    """Render all deterministic P34A bundle files without filesystem I/O."""

    admitted = render_admitted_observations_jsonl(result.snapshot_result).encode("utf-8")
    attachments = render_attachments_jsonl(result.attachment_result).encode("utf-8")
    evaluations = render_evaluations_jsonl(result.evaluation_result).encode("utf-8")
    feedback = render_feedback_jsonl(result.feedback_result).encode("utf-8")
    final_results = result.final_results_bytes
    structural = _canonical_jsonl(result.structural_rows)
    prediction_results = _canonical_jsonl(result.prediction_result_rows)
    settled = _canonical_jsonl(result.settled_predictions)
    authority = _canonical_json(result.result_authority)
    report = render_p34a_report(result)

    artifact_bytes = {
        "admitted_observations.jsonl": admitted,
        "attachments.jsonl": attachments,
        "evaluations.jsonl": evaluations,
        "feedback.jsonl": feedback,
        "final_results.jsonl": final_results,
        "structural_rows.jsonl": structural,
        "prediction_results.jsonl": prediction_results,
        "settled_predictions.jsonl": settled,
        "result_authority.json": authority,
        "report.md": report,
    }
    summary = {
        "schema_version": P34A_SCHEMA_VERSION,
        "operation": P34A_OPERATION,
        "source_p33a_run_id": result.p33a.run_manifest["run_id"],
        "source_p33a_bundle_fingerprint": result.p33a.summary["bundle_fingerprint"],
        "source_p33a_analysis_set_fingerprint": result.p33a.analysis_set_fingerprint,
        "source_p33a_analysis_jsonl_sha256": result.p33a.analysis_jsonl_sha256,
        "source_p33a_summary_json_sha256": result.p33a.summary_json_sha256,
        "p33a_pregame_authority_fingerprint": result.p33a.pregame_authority_fingerprint,
        "p33a_pregame_invariance": True,
        "target_date": result.p33a.summary["target_date"],
        "official_games": result.p33a.summary["official_raw_game_count"],
        "source_rows": result.p33a.summary["source_records_received"],
        "qualified": result.p33a.summary["observations_qualified"],
        "rejected": result.p33a.summary["observations_rejected"],
        "analysis_row_count": len(result.p33a.analysis_rows),
        "structural_row_count": len(result.structural_rows),
        "settleable_prediction_count": len(result.prediction_result_rows),
        "settled_prediction_count": result.settled_count,
        "correct_count": result.evaluation_result.correct_count,
        "incorrect_count": result.evaluation_result.incorrect_count,
        "unresolved_count": result.unresolved_count,
        "descriptive_accuracy": result.accuracy,
        "mean_selected_side_probability": result.mean_selected_side_probability,
        "brier_score": result.brier_score,
        "calibration_metrics_supported": True,
        "settlement_status": result.result_authority.get("settlement_status"),
        "result_authority": result.result_authority,
        "result_authority_fingerprint": result.result_authority[
            "result_authority_fingerprint"
        ],
        "attachment_set_fingerprint": result.attachment_result.attachment_set_fingerprint,
        "evaluation_set_fingerprint": result.evaluation_result.evaluation_set_fingerprint,
        "feedback_ledger_fingerprint": result.feedback_result.feedback_ledger_fingerprint,
        "offline_replay_verified": result.offline_replay_verified,
        "network_called": result.network_called,
        "sample_limited": True,
        "claims": {
            "official_final_results_source": True,
            "settlement_included": bool(result.prediction_result_rows),
            "feedback_generated": True,
            "structural_rows_evaluated": False,
            "pregame_fields_mutated": False,
            "model_retraining_performed": False,
            "scheduler_activated": False,
            "real_betting_performed": False,
            "profitability_claim": False,
            "training_dataset_claim": False,
            "sample_limited": True,
        },
        "artifact_sha256": {
            name: _sha256(content) for name, content in artifact_bytes.items()
        },
    }
    artifact_bytes["summary.json"] = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return artifact_bytes


def write_p34a_artifacts(
    output_dir: str | Path,
    result: P34ASettlementResult,
) -> dict[str, str]:
    """Write exactly the deterministic P34A bundle and return output hashes."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = render_p34a_artifacts(result)
    for name, content in artifacts.items():
        (root / name).write_bytes(content)
    return {name: _sha256(content) for name, content in artifacts.items()}


__all__ = (
    "render_p34a_artifacts",
    "render_p34a_report",
    "write_p34a_artifacts",
)
