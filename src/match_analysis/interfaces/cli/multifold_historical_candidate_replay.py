"""CLI for the bounded P21B contiguous multifold replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.multifold_historical_candidate_artifacts import (
    write_multifold_historical_candidate_artifacts,
)
from ...application.use_cases.replay_multifold_historical_candidates import (
    load_multifold_folds,
    load_multifold_reconstructed_models,
    replay_multifold_historical_candidates,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the next contiguous bounded P13 folds through P20A, "
            "P15C, P16A, P16B, P17A, and P21A."
        )
    )
    parser.add_argument(
        "--fold",
        action="append",
        required=True,
        type=Path,
        help="One committed P21B fold fixture; repeat in any order.",
    )
    parser.add_argument("--historical-results", required=True, type=Path)
    parser.add_argument("--historical-provenance", required=True, type=Path)
    parser.add_argument(
        "--reconstructed-models",
        type=Path,
        help="Optional committed P20A model-state mapping for dependency-free replay.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    input_paths = (*args.fold, args.historical_results, args.historical_provenance)
    if args.reconstructed_models is not None:
        input_paths = (*input_paths, args.reconstructed_models)
    for path in input_paths:
        if not path.is_file():
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        folds = load_multifold_folds(args.fold)
        result = replay_multifold_historical_candidates(
            folds=folds,
            historical_results_bytes=args.historical_results.read_bytes(),
            historical_provenance_bytes=args.historical_provenance.read_bytes(),
            reconstructed_models=(
                load_multifold_reconstructed_models(args.reconstructed_models)
                if args.reconstructed_models is not None
                else None
            ),
        )
        write_multifold_historical_candidate_artifacts(args.output_dir, result)
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Replayed folds={','.join(result.selected_fold_ids)} "
        f"games={result.prediction_row_count} "
        f"feedback_rows={len(result.feedback_rows)} "
        f"assessed={len(result.assessments)} "
        f"eligible={result.p21a_eligible_count} "
        f"excluded={result.p21a_excluded_count} "
        f"candidate_fingerprint={result.candidate_semantic_fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
