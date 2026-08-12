"""Deterministic P38A probability-calibration artifact rendering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any


P38A_CALIBRATION_ARTIFACT_SCHEMA_VERSION = (
    "p38a.rolling_probability_calibration_artifact.v1"
)
P38A_COMPARISON_SCHEMA_VERSION = "p38a.rolling_probability_calibration_comparison.v1"
P38A_PER_WINDOW_SCHEMA_VERSION = "p38a.rolling_probability_calibration_per_window.v1"
P38A_SUMMARY_SCHEMA_VERSION = "p38a.rolling_probability_calibration_summary.v1"


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def render_calibration_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    return (
        json.dumps(
            {
                "schema_version": P38A_CALIBRATION_ARTIFACT_SCHEMA_VERSION,
                "windows": [dict(artifact) for artifact in artifacts],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_comparisons_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
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
                "schema_version": P38A_COMPARISON_SCHEMA_VERSION,
                **dict(row),
            }
        )
        for row in ordered
    )


def render_per_window_summary(
    windows: Sequence[Mapping[str, Any]],
) -> str:
    return (
        json.dumps(
            {
                "schema_version": P38A_PER_WINDOW_SCHEMA_VERSION,
                "windows": [dict(window) for window in windows],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_summary(summary: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            {"schema_version": P38A_SUMMARY_SCHEMA_VERSION, **dict(summary)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_report_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# P38A Rolling Moneyline Probability Calibration Evaluation",
        "",
        "This is a leakage-safe offline probability-reliability evaluation. No model was promoted.",
        "",
        "## Fixed calibration method",
        "",
        f"- Method: {summary['calibration']['method']} ({summary['calibration']['method_version']}).",
        f"- Authority: {summary['calibration']['source_kind']}; the source rows are prior P37A true-OOS prediction/label pairs.",
        "- Method search and target-holdout tuning: false.",
        "",
        "## Chronological windows",
        "",
        "| Window | Train folds | Calibration source | Holdout | Raw | Evaluable | Excluded |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for window in summary["evaluation_windows"]:
        lines.append(
            "| {window_id} | {train_folds} | {calibration} | {holdout} | "
            "{raw} | {evaluable} | {excluded} |".format(
                window_id=window["evaluation_window_id"],
                train_folds=",".join(window["train_fold_ids"]),
                calibration=",".join(window["calibration"]["source_fold_ids"]),
                holdout=window["holdout_fold_id"],
                raw=window["holdout"]["raw_row_count"],
                evaluable=window["holdout"]["evaluable_row_count"],
                excluded=window["holdout"]["excluded_row_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-window probability metrics",
            "",
            "| Holdout | Model | Accuracy | Brier | Log loss | ECE |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for window in summary["evaluation_windows"]:
        for label in ("champion", "raw_challenger", "calibrated_challenger"):
            metrics = window[label]["metrics"]
            lines.append(
                f"| {window['holdout_fold_id']} | {label} | {metrics['accuracy']} | "
                f"{metrics['brier_score']} | {metrics['log_loss']} | "
                f"{metrics['calibration']['expected_calibration_error']} |"
            )
    lines.extend(
        [
            "",
            "## Aggregate exact-row comparison",
            "",
            f"- Rows: {aggregate['evaluable_row_count']} evaluable of {aggregate['raw_row_count']} raw; coverage {aggregate['coverage']}.",
            "",
            "| Model | Accuracy | Brier | Log loss | ECE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in ("champion", "raw_challenger", "calibrated_challenger"):
        metrics = aggregate[label]["metrics"]
        lines.append(
            f"| {label} | {metrics['accuracy']} | {metrics['brier_score']} | "
            f"{metrics['log_loss']} | "
            f"{metrics['calibration']['expected_calibration_error']} |"
        )
    lines.extend(
        [
            "",
            f"- Calibrated vs raw deltas (right minus left): {summary['comparison']['calibrated_vs_raw']}.",
            f"- Calibrated vs champion deltas (right minus left): {summary['comparison']['calibrated_vs_champion']}.",
            f"- Conclusion: {summary['comparison']['conclusion']}.",
            "- Accuracy uses each model's probability threshold of 0.5; calibration can change accuracy when its fixed Platt map shifts a raw probability across that threshold.",
            "",
            "## Safety claims",
            "",
            "- Calibration rows strictly precede each target holdout and are game-ID disjoint from it.",
            "- The calibrator is fit only on prior true-OOS P37A rows; target-holdout labels are not used for fitting or method selection.",
            "- Champion, raw challenger, and calibrated challenger use identical evaluable target rows in each window.",
            "- P37A exclusion semantics are preserved; no calibration, betting, edge, profitability, staking, or promotion claim is made.",
            "",
            "## Not admitted",
            "",
            f"- wf_004: {summary['not_admitted_target_holdout_fold_ids']['wf_004']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_p38a_probability_calibration_artifacts(
    output_dir: str | Path,
    *,
    calibration_artifacts: Sequence[Mapping[str, Any]],
    comparison_rows: Iterable[Mapping[str, Any]],
    per_window_summary: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write exactly the bounded P38A artifact, row, summary, and report files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "calibration_artifacts.json").write_text(
        render_calibration_artifacts(calibration_artifacts),
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
    "P38A_CALIBRATION_ARTIFACT_SCHEMA_VERSION",
    "P38A_COMPARISON_SCHEMA_VERSION",
    "P38A_PER_WINDOW_SCHEMA_VERSION",
    "P38A_SUMMARY_SCHEMA_VERSION",
    "render_calibration_artifacts",
    "render_comparisons_jsonl",
    "render_per_window_summary",
    "render_report_markdown",
    "render_summary",
    "write_p38a_probability_calibration_artifacts",
)
