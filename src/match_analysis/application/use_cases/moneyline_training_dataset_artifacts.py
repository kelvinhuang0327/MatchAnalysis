"""Deterministic P22A training-dataset artifact rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from ...baseball.domain.supervised_training_example import SupervisedTrainingExample
from .materialize_moneyline_training_dataset import (
    MoneylineTrainingDataset,
)


def render_training_examples_jsonl(
    examples: Sequence[SupervisedTrainingExample],
) -> str:
    """Render examples in path-independent deterministic identity order."""

    ordered = sorted(examples, key=lambda example: example.training_example_id)
    return "".join(
        json.dumps(
            example.to_projection(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for example in ordered
    )


def render_training_dataset_summary(
    dataset: MoneylineTrainingDataset,
    *,
    training_examples_jsonl_sha256: str,
) -> str:
    """Render the deterministic P22A summary projection."""

    return (
        json.dumps(
            dataset.to_summary(
                training_examples_jsonl_sha256=training_examples_jsonl_sha256
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_moneyline_training_dataset_artifacts(
    output_dir: str | Path,
    dataset: MoneylineTrainingDataset,
) -> None:
    """Write exactly the P22A JSONL and summary artifacts."""

    directory = Path(output_dir)
    examples_content = render_training_examples_jsonl(dataset.examples)
    summary_content = render_training_dataset_summary(
        dataset,
        training_examples_jsonl_sha256=hashlib.sha256(
            examples_content.encode("utf-8")
        ).hexdigest(),
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "training_examples.jsonl").write_text(
        examples_content,
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(summary_content, encoding="utf-8")


__all__ = (
    "render_training_dataset_summary",
    "render_training_examples_jsonl",
    "write_moneyline_training_dataset_artifacts",
)
