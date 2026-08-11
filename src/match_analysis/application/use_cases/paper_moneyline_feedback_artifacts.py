"""Deterministic P25A paper Moneyline settlement artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .prediction_evaluation_artifacts import render_evaluations_jsonl
from .prediction_feedback_artifacts import render_feedback_jsonl
from .settle_paper_moneyline_batch import (
    P25A_SCHEMA_VERSION,
    PaperMoneylineSettlementResult,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _render_jsonl(rows: tuple[dict[str, Any], ...]) -> bytes:
    lines = [
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def render_settled_predictions_jsonl(result: PaperMoneylineSettlementResult) -> bytes:
    """Render one deterministic P25A lineage row per settled prediction."""

    return _render_jsonl(result.settled_predictions)


def render_paper_moneyline_feedback_artifacts(
    result: PaperMoneylineSettlementResult,
) -> dict[str, bytes]:
    """Render the four committed P25A artifact bytes without filesystem I/O."""

    settled = render_settled_predictions_jsonl(result)
    evaluations = render_evaluations_jsonl(result.evaluation_result).encode("utf-8")
    feedback = render_feedback_jsonl(result.feedback_result).encode("utf-8")
    summary = {
        "schema_version": P25A_SCHEMA_VERSION,
        "source_batch_id": result.authority.batch_id,
        "source_prediction_fingerprint": result.authority.prediction_fingerprint,
        "prediction_fingerprint": result.authority.prediction_fingerprint,
        "source_manifest_fingerprint": result.authority.source_manifest_fingerprint,
        "raw_game_count": result.authority.raw_game_count,
        "prediction_count": len(result.authority.predictions),
        "feature_unavailable_count": len(result.authority.feature_unavailable),
        "settled_prediction_count": len(result.settled_predictions),
        "evaluation_count": result.evaluation_result.evaluation_row_count,
        "feedback_row_count": result.feedback_result.prediction_row_count,
        "correct_count": result.evaluation_result.correct_count,
        "incorrect_count": result.evaluation_result.incorrect_count,
        "accuracy": result.accuracy,
        "mean_brier": result.mean_brier,
        "feedback_ledger_fingerprint": result.feedback_result.feedback_ledger_fingerprint,
        "prediction_snapshot_fingerprint": result.snapshot_result.snapshot_fingerprint,
        "attachment_set_fingerprint": result.attachment_result.attachment_set_fingerprint,
        "evaluation_set_fingerprint": result.evaluation_result.evaluation_set_fingerprint,
        "result_authority_fingerprint": result.result_authority_fingerprint,
        "result_authority": result.result_authority_summary,
        "claims": result.claims,
        "all_results_final": True,
        "prediction_authority_verified": True,
        "offline_settlement": True,
        "model_promoted": True,
        "challenger_retrained": False,
        "deployment_performed": False,
        "profitability_claim": False,
        "production_ready": False,
        "real_betting_recommendation": False,
        "promotion_scope": "paper_only",
        "p20b_historical_runtime_compliance": "REMAINS_REFUTED",
        "settled_predictions_jsonl_sha256": _sha256(settled),
        "evaluations_jsonl_sha256": _sha256(evaluations),
        "feedback_ledger_jsonl_sha256": _sha256(feedback),
    }
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {
        "settled_predictions.jsonl": settled,
        "evaluations.jsonl": evaluations,
        "feedback_ledger.jsonl": feedback,
        "summary.json": summary_bytes,
    }


def write_paper_moneyline_feedback_artifacts(
    output_dir: str | Path,
    result: PaperMoneylineSettlementResult,
) -> None:
    """Write exactly the four deterministic P25A committed artifacts."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, content in render_paper_moneyline_feedback_artifacts(result).items():
        (root / name).write_bytes(content)


__all__ = (
    "render_paper_moneyline_feedback_artifacts",
    "render_settled_predictions_jsonl",
    "write_paper_moneyline_feedback_artifacts",
)
