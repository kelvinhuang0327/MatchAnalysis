"""Deterministic P30A paper-analysis artifact writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paper_moneyline_batch_artifacts import render_jsonl, sha256_bytes
from .run_moneyline_paper_analysis import MoneylinePaperAnalysisRunResult


def write_moneyline_paper_analysis_artifacts(
    output_dir: str | Path,
    *,
    result: MoneylinePaperAnalysisRunResult,
) -> dict[str, str]:
    """Write only the two committed P30A artifact files."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {
        "analysis.jsonl": render_jsonl(result.analysis),
        "summary.json": (
            json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    return {name: sha256_bytes(content) for name, content in files.items()}


__all__ = ("write_moneyline_paper_analysis_artifacts",)
