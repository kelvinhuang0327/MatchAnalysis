"""CLI for deterministic P22A game-level dataset materialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.materialize_moneyline_training_dataset import (
    materialize_moneyline_training_dataset,
)
from ...application.use_cases.moneyline_training_dataset_artifacts import (
    write_moneyline_training_dataset_artifacts,
)
from ...application.use_cases.replay_multifold_historical_candidates import (
    load_multifold_folds,
    load_multifold_reconstructed_models,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one deterministic game-level supervised example per "
            "eligible P21B historical game."
        )
    )
    parser.add_argument("--learning-candidates", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument(
        "--fold",
        action="append",
        required=True,
        type=Path,
        help="One committed P21B fold fixture; repeat for each selected fold.",
    )
    parser.add_argument("--historical-results", required=True, type=Path)
    parser.add_argument("--historical-provenance", required=True, type=Path)
    parser.add_argument("--reconstructed-models", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    input_paths = (
        args.learning_candidates,
        args.candidate_summary,
        *args.fold,
        args.historical_results,
        args.historical_provenance,
        args.reconstructed_models,
    )
    for path in input_paths:
        if not path.is_file():
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        dataset = materialize_moneyline_training_dataset(
            candidate_bytes=args.learning_candidates.read_bytes(),
            candidate_summary_bytes=args.candidate_summary.read_bytes(),
            folds=load_multifold_folds(args.fold),
            historical_results_bytes=args.historical_results.read_bytes(),
            historical_provenance_bytes=args.historical_provenance.read_bytes(),
            reconstructed_models=load_multifold_reconstructed_models(
                args.reconstructed_models
            ),
        )
        write_moneyline_training_dataset_artifacts(args.output_dir, dataset)
    except (OSError, TypeError, ValueError, UnicodeDecodeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Materialized folds={','.join(item['fold_id'] for item in dataset.source_fold_artifacts)} "
        f"games={dataset.training_example_count} "
        f"eligible_candidates={dataset.eligible_candidate_count} "
        f"collapsed={dataset.candidates_collapsed_count} "
        f"unmapped={dataset.unmapped_candidate_count} "
        f"dataset_fingerprint={dataset.dataset_fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
