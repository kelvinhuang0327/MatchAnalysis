"""CLI for deterministic P22B Moneyline challenger training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.train_moneyline_challenger import (
    P22B_DEFAULT_FIT_RUNTIME,
    P22B_DEFAULT_SOURCE_COMMIT,
    P22B_DEFAULT_SOURCE_TREE,
    train_moneyline_challenger,
)
from ...application.use_cases.moneyline_challenger_artifacts import (
    write_moneyline_challenger_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    """Train one local challenger without provider, database, or promotion."""

    parser = argparse.ArgumentParser(
        description="Train exactly one deterministic P22B Moneyline challenger."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--fit-runtime", default=P22B_DEFAULT_FIT_RUNTIME)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-commit", default=P22B_DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--source-tree", default=P22B_DEFAULT_SOURCE_TREE)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    for path, label in (
        (args.dataset, "P22A training dataset"),
        (args.summary, "P22A summary"),
    ):
        if not path.is_file():
            print(f"ERROR: {label} file does not exist: {path}", file=sys.stderr)
            return 1
    try:
        artifact = train_moneyline_challenger(
            args.dataset,
            args.summary,
            fit_runtime=args.fit_runtime,
            source_repository=args.source_repository,
            source_commit=args.source_commit,
            source_tree=args.source_tree,
        )
        write_moneyline_challenger_artifacts(args.output_dir, artifact)
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Trained one Moneyline challenger model_id={artifact.projection['model_id']} "
        f"examples={artifact.projection['training_example_count']} "
        f"dataset_fingerprint={artifact.projection['source_dataset_fingerprint']} "
        f"artifact_fingerprint={artifact.fingerprint()} "
        f"fit_runtime={artifact.projection['training_runtime']['executable']} "
        "training_authorized=true training_performed=true model_promoted=false "
        "production_ready=false out_of_sample_evaluated=false "
        "profitability_claim=false real_betting_recommendation=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
