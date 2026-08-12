"""Deterministic P37A rolling Moneyline OOS artifact rendering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any


P37A_ARTIFACT_SCHEMA_VERSION = (
    "p37a.rolling_moneyline_oos_challenger_artifact.v1"
)
P37A_COMPARISON_SCHEMA_VERSION = "p37a.rolling_moneyline_oos_comparison.v1"
P37A_PER_WINDOW_SCHEMA_VERSION = "p37a.rolling_moneyline_oos_per_window.v1"
P37A_SUMMARY_SCHEMA_VERSION = "p37a.rolling_moneyline_oos_summary.v1"


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def render_comparisons_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    """Render true OOS rows in chronological window and game order."""

    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["evaluation_window_order"]),
            str(row["scheduled_start_utc"]),
            int(row["game_number"]),
            int(row["game_pk"]),
        ),
    )
    return "".join(
        _json_line(
            {
                "schema_version": P37A_COMPARISON_SCHEMA_VERSION,
                **dict(row),
            }
        )
        for row in ordered
    )


def render_model_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> str:
    """Render all independently trained challenger artifacts."""

    return (
        json.dumps(
            {
                "schema_version": P37A_ARTIFACT_SCHEMA_VERSION,
                "artifacts": [dict(artifact) for artifact in artifacts],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_per_window_summary(windows: Sequence[Mapping[str, Any]]) -> str:
    """Render the complete per-window lineage and metrics."""

    return (
        json.dumps(
            {
                "schema_version": P37A_PER_WINDOW_SCHEMA_VERSION,
                "windows": [dict(window) for window in windows],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_summary(summary: Mapping[str, Any]) -> str:
    """Render the aggregate result and explicit no-promotion claims."""

    return (
        json.dumps(
            {"schema_version": P37A_SUMMARY_SCHEMA_VERSION, **dict(summary)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_report_markdown(summary: Mapping[str, Any]) -> str:
    """Render a concise descriptive P37A report."""

    aggregate = summary["aggregate"]
    comparison = summary["comparison"]
    lines = [
        "# P37A Rolling Walk-Forward Moneyline OOS Evaluation",
        "",
        "This is a deterministic offline champion/challenger evaluation. No model was promoted.",
        "",
        "## Chronological windows",
        "",
        "| Window | Train folds | Train rows | Holdout | Raw | Evaluable | Excluded |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for window in summary["evaluation_windows"]:
        lines.append(
            "| `{window_id}` | `{train_folds}` | {train_rows} | `{holdout}` | "
            "{raw} | {evaluable} | {excluded} |".format(
                window_id=window["evaluation_window_id"],
                train_folds=",".join(window["train_fold_ids"]),
                train_rows=window["training"]["eligible_row_count"],
                holdout=window["holdout_fold_id"],
                raw=window["holdout"]["raw_row_count"],
                evaluable=window["holdout"]["evaluable_row_count"],
                excluded=window["holdout"]["excluded_row_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate true-OOS comparison",
            "",
            f"- Rows: `{aggregate['evaluable_row_count']}` evaluable of `{aggregate['raw_row_count']}` raw; coverage `{aggregate['coverage']}`.",
            f"- Champion accuracy / Brier / log loss / ECE: `{aggregate['champion']['metrics']['accuracy']}` / `{aggregate['champion']['metrics']['brier_score']}` / `{aggregate['champion']['metrics']['log_loss']}` / `{aggregate['champion']['metrics']['calibration']['expected_calibration_error']}`.",
            f"- Challenger accuracy / Brier / log loss / ECE: `{aggregate['challenger']['metrics']['accuracy']}` / `{aggregate['challenger']['metrics']['brier_score']}` / `{aggregate['challenger']['metrics']['log_loss']}` / `{aggregate['challenger']['metrics']['calibration']['expected_calibration_error']}`.",
            f"- Aggregate metric direction: `{comparison['aggregate_verdict']}`.",
            f"- Conclusion: `{comparison['conclusion']}`.",
            "",
            "## Safety claims",
            "",
            "- Training and holdout game identities are disjoint in every window.",
            "- Champion and challenger are scored on identical evaluable rows per window.",
            "- Predictions are generated from point-in-time feature rows before final outcomes are paired.",
            "- No aggregate-OOS tuning, calibration fitting, promotion, betting, profitability, or staking claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def write_rolling_moneyline_oos_artifacts(
    output_dir: str | Path,
    *,
    model_artifacts: Sequence[Mapping[str, Any]],
    comparison_rows: Iterable[Mapping[str, Any]],
    per_window_summary: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write exactly the bounded P37A artifact, row, summary, and report files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model_artifacts.json").write_text(
        render_model_artifacts(model_artifacts),
        encoding="utf-8",
    )
    (directory / "comparisons.jsonl").write_text(
        render_comparisons_jsonl(comparison_rows),
        encoding="utf-8",
    )
    (directory / "per_window_summary.json").write_text(
        render_per_window_summary(per_window_summary),
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(
        render_summary(summary),
        encoding="utf-8",
    )
    (directory / "report.md").write_text(
        render_report_markdown(summary),
        encoding="utf-8",
    )


__all__ = (
    "P37A_ARTIFACT_SCHEMA_VERSION",
    "P37A_COMPARISON_SCHEMA_VERSION",
    "P37A_PER_WINDOW_SCHEMA_VERSION",
    "P37A_SUMMARY_SCHEMA_VERSION",
    "render_comparisons_jsonl",
    "render_model_artifacts",
    "render_per_window_summary",
    "render_report_markdown",
    "render_summary",
    "write_rolling_moneyline_oos_artifacts",
)
