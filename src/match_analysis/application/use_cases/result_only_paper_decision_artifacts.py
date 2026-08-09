"""Deterministic P18A result-only decision and settlement artifacts."""

import hashlib
import json
from pathlib import Path

from .build_result_only_paper_decision_replay import (
    ResultOnlyPaperDecisionReplay,
)


def _decision_to_dict(decision: object) -> dict[str, object]:
    return {
        "decision_id": decision.decision_id,
        "game_number": decision.game_number,
        "prediction_generated_at_utc": decision.prediction_generated_at_utc,
        "prediction_observation_id": decision.prediction_observation_id,
        "provider_game_id": decision.provider_game_id,
        "provider_namespace": decision.provider_namespace,
        "scheduled_start_utc": decision.scheduled_start_utc,
        "selection": decision.selection,
        "source_snapshot_row_fingerprint": decision.source_snapshot_row_fingerprint,
    }


def _settlement_to_dict(settlement: object) -> dict[str, object]:
    return {
        "actual_winner": settlement.actual_winner,
        "away_score": settlement.away_score,
        "decision_id": settlement.decision_id,
        "game_number": settlement.game_number,
        "home_score": settlement.home_score,
        "prediction_observation_id": settlement.prediction_observation_id,
        "provider_game_id": settlement.provider_game_id,
        "provider_namespace": settlement.provider_namespace,
        "result_observation_id": settlement.result_observation_id,
        "result_observed_at_utc": settlement.result_observed_at_utc,
        "selection": settlement.selection,
        "settlement_row_fingerprint": settlement.settlement_row_fingerprint,
        "settlement_status": settlement.settlement_status,
    }


def render_decisions_jsonl(result: ResultOnlyPaperDecisionReplay) -> str:
    """Render the pre-outcome frozen decision set."""

    return "".join(
        json.dumps(_decision_to_dict(decision), sort_keys=True, separators=(",", ":"))
        + "\n"
        for decision in result.selection.decisions
    )


def render_settlements_jsonl(result: ResultOnlyPaperDecisionReplay) -> str:
    """Render deterministic result-only settlements in frozen decision order."""

    return "".join(
        json.dumps(
            _settlement_to_dict(settlement),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for settlement in result.settlements
    )


def render_replay_summary_json(
    result: ResultOnlyPaperDecisionReplay,
    decisions_sha256: str,
    settlements_sha256: str,
    report_sha256: str,
) -> str:
    """Render the deterministic summary with all source fingerprints."""

    summary = {
        "claims": result.claims,
        "decision_count": len(result.selection.decisions),
        "decision_ids": [
            decision.decision_id for decision in result.selection.decisions
        ],
        "decision_set_fingerprint": result.selection.decision_set_fingerprint,
        "decisions_jsonl_sha256": decisions_sha256,
        "excluded_row_count": result.selection.excluded_row_count,
        "final_results_sha256": result.final_results_sha256,
        "lost_count": result.lost_count,
        "report_sha256": report_sha256,
        "schema_version": result.schema_version,
        "settled_count": result.settled_count,
        "settlement_schema_version": result.settlement_schema_version,
        "settlement_set_fingerprint": result.settlement_set_fingerprint,
        "settlement_status_counts": {
            "LOST": result.lost_count,
            "UNSETTLED": result.unsettled_count,
            "WON": result.won_count,
        },
        "settlements_jsonl_sha256": settlements_sha256,
        "source_snapshot_fingerprint": result.source_snapshot_fingerprint,
        "source_snapshot_sha256": result.source_snapshot_sha256,
        "source_snapshot_summary_sha256": result.source_snapshot_summary_sha256,
        "unsettled_count": result.unsettled_count,
        "won_count": result.won_count,
    }
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


def render_replay_report_markdown(result: ResultOnlyPaperDecisionReplay) -> str:
    """Render a human-readable result-only report without runtime values."""

    lines = [
        "# Result-Only Paper Decision Replay Report",
        "",
        "## Frozen Decision Selection",
        "",
        f"- **Decision Schema**: `{result.schema_version}`",
        f"- **Decision Count**: `{len(result.selection.decisions)}`",
        f"- **Excluded Snapshot Rows**: `{result.selection.excluded_row_count}`",
        f"- **Source Snapshot Fingerprint**: `{result.source_snapshot_fingerprint}`",
        f"- **Decision Set Fingerprint**: `{result.selection.decision_set_fingerprint}`",
        "",
        "Decisions are selected from the prediction snapshot before final-result bytes are read.",
        "",
        "| Decision ID | Prediction Observation | Provider Game | Selection | Prediction Time (UTC) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for decision in result.selection.decisions:
        lines.append(
            f"| `{decision.decision_id[:16]}...` "
            f"| `{decision.prediction_observation_id[:16]}...` "
            f"| `{decision.provider_game_id}#{decision.game_number}` "
            f"| `{decision.selection}` "
            f"| `{decision.prediction_generated_at_utc}` |"
        )

    lines.extend(
        [
            "",
            "## Result-Only Settlement",
            "",
            f"- **Settlement Schema**: `{result.settlement_schema_version}`",
            f"- **Settled**: `{result.settled_count}`",
            f"- **Unsettled**: `{result.unsettled_count}`",
            f"- **Won**: `{result.won_count}`",
            f"- **Lost**: `{result.lost_count}`",
            f"- **Settlement Set Fingerprint**: `{result.settlement_set_fingerprint}`",
            "",
            "| Decision ID | Selection | Actual Winner | Score | Status |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for settlement in result.settlements:
        score = "—"
        if settlement.home_score is not None and settlement.away_score is not None:
            score = f"{settlement.home_score}-{settlement.away_score}"
        winner = settlement.actual_winner or "—"
        lines.append(
            f"| `{settlement.decision_id[:16]}...` "
            f"| `{settlement.selection}` "
            f"| `{winner}` "
            f"| `{score}` "
            f"| `{settlement.settlement_status}` |"
        )

    lines.extend(
        [
            "",
            "## Explicit Limitations",
            "",
            "- This is a paper-only, result-only replay of synthetic local artifacts.",
            "- No price, payout, P&L, ROI, EV, Kelly, or profitability calculation was performed.",
            "- Final outcomes affect settlement status only; they do not affect decision selection.",
            "- No provider or network call was made.",
            "- No database write or deployment occurred.",
            "- No training, model promotion, or performance claim was made.",
            "",
        ]
    )
    return "\n".join(lines)


def write_result_only_paper_decision_artifacts(
    output_dir: Path,
    result: ResultOnlyPaperDecisionReplay,
) -> None:
    """Write decisions, settlements, summary, and report deterministically."""

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_content = render_decisions_jsonl(result)
    settlements_content = render_settlements_jsonl(result)
    decisions_sha256 = hashlib.sha256(decisions_content.encode("utf-8")).hexdigest()
    settlements_sha256 = hashlib.sha256(settlements_content.encode("utf-8")).hexdigest()
    report_content = render_replay_report_markdown(result)
    report_sha256 = hashlib.sha256(report_content.encode("utf-8")).hexdigest()
    summary_content = render_replay_summary_json(
        result,
        decisions_sha256,
        settlements_sha256,
        report_sha256,
    )
    (output_dir / "decisions.jsonl").write_text(decisions_content, encoding="utf-8")
    (output_dir / "settlements.jsonl").write_text(
        settlements_content,
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(summary_content, encoding="utf-8")
    (output_dir / "report.md").write_text(report_content, encoding="utf-8")
