"""Deterministic P23B multi-fold comparison artifact rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


P23B_FOLD_ORDER = ("wf_004", "wf_005", "wf_006")
P23B_COMPARISON_SCHEMA_VERSION = (
    "p23b.moneyline_contiguous_multifold_oos_comparison.v1"
)
P23B_PER_FOLD_SUMMARY_SCHEMA_VERSION = (
    "p23b.moneyline_contiguous_multifold_oos_per_fold_summary.v1"
)
P23B_SUMMARY_SCHEMA_VERSION = "p23b.moneyline_contiguous_multifold_oos_summary.v1"


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def render_comparisons_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    """Render comparison rows in fixed fold and scheduled-game order."""

    order = {fold_id: index for index, fold_id in enumerate(P23B_FOLD_ORDER)}
    ordered = sorted(
        rows,
        key=lambda row: (
            order[str(row["fold_id"])],
            str(row["scheduled_start_utc"]),
            int(row["game_number"]),
            int(row["game_pk"]),
        ),
    )
    return "".join(
        _json_line({"schema_version": P23B_COMPARISON_SCHEMA_VERSION, **dict(row)})
        for row in ordered
    )


def render_per_fold_summary(folds: Sequence[Mapping[str, Any]]) -> str:
    """Render the required per-fold descriptive evidence."""

    return (
        json.dumps(
            {
                "schema_version": P23B_PER_FOLD_SUMMARY_SCHEMA_VERSION,
                "folds": [dict(fold) for fold in folds],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_summary(summary: Mapping[str, Any]) -> str:
    """Render pooled metrics and explicit no-promotion claims."""

    return (
        json.dumps(
            {"schema_version": P23B_SUMMARY_SCHEMA_VERSION, **dict(summary)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_multifold_oos_artifacts(
    output_dir: str | Path,
    *,
    comparison_rows: Iterable[Mapping[str, Any]],
    per_fold_summary: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write exactly the three authorized P23B report files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fold_comparisons.jsonl").write_text(
        render_comparisons_jsonl(comparison_rows),
        encoding="utf-8",
    )
    (directory / "per_fold_summary.json").write_text(
        render_per_fold_summary(per_fold_summary),
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(
        render_summary(summary),
        encoding="utf-8",
    )


__all__ = (
    "P23B_COMPARISON_SCHEMA_VERSION",
    "P23B_FOLD_ORDER",
    "P23B_PER_FOLD_SUMMARY_SCHEMA_VERSION",
    "P23B_SUMMARY_SCHEMA_VERSION",
    "render_comparisons_jsonl",
    "render_per_fold_summary",
    "render_summary",
    "write_multifold_oos_artifacts",
)
