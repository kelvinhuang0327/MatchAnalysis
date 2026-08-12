"""Deterministic P39A market snapshot and P37 join artifact rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .join_p37_oos_market_snapshots import P39AMarketJoinResult


P39A_ARTIFACT_FILES = (
    "source_manifest.json",
    "market_snapshots.jsonl",
    "market_join.jsonl",
    "summary.json",
    "report.md",
)


def _json_line(value: dict[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def render_market_snapshots(result: P39AMarketJoinResult) -> str:
    return "".join(
        _json_line(snapshot.to_projection())
        for snapshot in sorted(result.selected_snapshots, key=lambda item: item.snapshot_id)
    )


def render_market_join(result: P39AMarketJoinResult) -> str:
    return "".join(
        _json_line(row.to_projection())
        for row in sorted(result.join_rows, key=lambda item: item.prediction.key())
    )


def render_source_manifest(result: P39AMarketJoinResult) -> str:
    return (
        json.dumps(
            result.source_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_summary(result: P39AMarketJoinResult) -> str:
    return (
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def render_report(result: P39AMarketJoinResult) -> str:
    summary = result.summary
    source = result.source_manifest
    p37 = result.p37_manifest
    lines = [
        "# P39A TSL Moneyline Market Snapshot Join",
        "",
        "This artifact attaches only trustworthy pregame two-sided Moneyline observations to the P37 true-OOS prediction universe. It does not make betting decisions.",
        "",
        "## Conclusion",
        "",
        f"- Result: `{summary['conclusion']}`",
        f"- Rule: `{summary['conclusion_rule']}`",
        f"- Edge-ready rows: `{summary['edge_ready_count']}` of `{summary['p37_evaluable_target_count']}`",
        "",
        "## Coverage",
        "",
        f"- Exact identity matches: `{summary['exact_identity_match_count']}`",
        f"- Usable pregame Moneyline rows: `{summary['usable_pregame_market_rows']}`",
        f"- No-market rows: `{summary['no_market_rows']}`",
        f"- Post-start rejected rows: `{summary['post_start_rejected_rows']}`",
        f"- Ambiguous rows: `{summary['ambiguous_rows']}`",
        f"- Missing or untrusted timestamp rows: `{summary['missing_or_untrusted_timestamp_rows']}`",
        f"- Malformed or incomplete price rows: `{summary['malformed_or_incomplete_price_rows']}`",
        f"- Not-pregame rejected rows: `{summary['not_pregame_rejected_rows']}`",
        "",
        "## Provenance",
        "",
        f"- Source repository: `{source['source_repository']}`",
        f"- Source path: `{source['source_path']}`",
        f"- Source HEAD/tree: `{source['source_head']}` / `{source['source_tree']}`",
        f"- Source SHA-256: `{source['source_sha256']}`",
        f"- Source rows inspected: `{source['source_row_count']}`",
        f"- Scoped source rows: `{source['scoped_source_row_count']}`",
        f"- Timestamp semantics: `{summary['timestamp_semantics']}`",
        f"- Selected snapshot rule: `{summary['selected_snapshot_rule']}`",
        f"- P37 comparisons SHA-256: `{p37['comparisons_sha256']}`",
        "",
        "## Safety boundary",
        "",
        "- P37 predictions and P38 calibration artifacts are read-only inputs.",
        "- Market snapshot selection does not read or use outcomes.",
        "- BET/PASS, ROI, profitability, bankroll, staking, Kelly, calibration, and model promotion are `NOT RUN`.",
        "",
    ]
    return "\n".join(lines)


def write_p39a_artifacts(
    output_dir: str | Path,
    result: P39AMarketJoinResult,
) -> None:
    """Write only the five repository-native P39A artifact files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source_manifest.json").write_text(
        render_source_manifest(result),
        encoding="utf-8",
    )
    (directory / "market_snapshots.jsonl").write_text(
        render_market_snapshots(result),
        encoding="utf-8",
    )
    (directory / "market_join.jsonl").write_text(
        render_market_join(result),
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(
        render_summary(result),
        encoding="utf-8",
    )
    (directory / "report.md").write_text(
        render_report(result),
        encoding="utf-8",
    )


__all__ = (
    "P39A_ARTIFACT_FILES",
    "render_market_join",
    "render_market_snapshots",
    "render_report",
    "render_source_manifest",
    "render_summary",
    "write_p39a_artifacts",
)
