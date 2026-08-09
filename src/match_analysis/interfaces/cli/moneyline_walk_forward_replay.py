"""CLI for bounded P20A P13 Moneyline walk-forward replay."""

import argparse
from pathlib import Path
import sys

from ...application.use_cases.replay_historical_moneyline_predictions import (
    replay_historical_moneyline_predictions,
    write_moneyline_walk_forward_replay_artifacts,
)
from ...application.use_cases.moneyline_walk_forward_artifacts import (
    load_reconstructed_model,
)
from ...application.use_cases.reconstruct_moneyline_walk_forward_model import (
    P20ADependencyRequirement,
    load_moneyline_walk_forward_fold,
    reconstruct_moneyline_walk_forward_model,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct and replay a bounded historical P13 Moneyline fold."
    )
    parser.add_argument("--fold-fixture", required=True, type=Path)
    parser.add_argument(
        "--reconstructed-model",
        type=Path,
        help="Reuse a verified deterministic fitted state instead of fitting again.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.fold_fixture.is_file():
        print(f"ERROR: fold fixture does not exist: {args.fold_fixture}", file=sys.stderr)
        return 1
    try:
        fold = load_moneyline_walk_forward_fold(args.fold_fixture)
        if args.reconstructed_model is None:
            model = reconstruct_moneyline_walk_forward_model(fold)
        else:
            if not args.reconstructed_model.is_file():
                print(
                    f"ERROR: reconstructed model does not exist: {args.reconstructed_model}",
                    file=sys.stderr,
                )
                return 1
            model = load_reconstructed_model(args.reconstructed_model)
        if model.fold_id != fold.fold_id:
            print("ERROR: reconstructed model fold_id does not match fixture", file=sys.stderr)
            return 1
        result = replay_historical_moneyline_predictions(fold, model)
        write_moneyline_walk_forward_replay_artifacts(args.output_dir, result)
    except P20ADependencyRequirement as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Replayed {result.fold.prediction_row_count} bounded historical rows; "
        f"parity_passed={result.parity_passed}; "
        f"model_artifact_fingerprint={result.model_artifact.fingerprint()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
