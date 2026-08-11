"""Deterministic P29A Moneyline market-movement artifact writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .generate_moneyline_market_movement import MoneylineMarketMovementResult
from .paper_moneyline_batch_artifacts import render_jsonl, sha256_bytes


def write_moneyline_market_movement_artifacts(
    output_dir: str | Path,
    *,
    result: MoneylineMarketMovementResult,
) -> dict[str, str]:
    """Write only the three authorized P29A report artifacts."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "closing_prices.jsonl": render_jsonl(result.closing_prices),
        "market_movement.jsonl": render_jsonl(result.market_movement),
        "summary.json": (
            json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    return {name: sha256_bytes(content) for name, content in files.items()}


def load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load a JSONL artifact with one object per non-blank line."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {line_number}: {path}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at line {line_number}: {path}")
        rows.append(value)
    return tuple(rows)


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load one JSON object artifact."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


__all__ = (
    "load_json_object",
    "load_jsonl",
    "write_moneyline_market_movement_artifacts",
)
