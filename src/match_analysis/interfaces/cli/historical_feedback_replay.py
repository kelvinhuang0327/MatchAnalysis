"""CLI for the deterministic P20B historical feedback replay."""

import argparse
from pathlib import Path
import sys

from ...application.use_cases.historical_feedback_replay_artifacts import (
    write_historical_feedback_replay_artifacts,
)
from ...application.use_cases.replay_historical_prediction_feedback import (
    replay_historical_prediction_feedback,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the two committed P20A historical games through "
            "P15C, P16A, P16B, and P17A."
        )
    )
    parser.add_argument("--p20a-predictions", required=True, type=Path)
    parser.add_argument("--p20a-reconstruction", required=True, type=Path)
    parser.add_argument("--p20a-summary", required=True, type=Path)
    parser.add_argument("--p20a-fold", required=True, type=Path)
    parser.add_argument("--historical-results", required=True, type=Path)
    parser.add_argument("--historical-provenance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    input_paths = (
        args.p20a_predictions,
        args.p20a_reconstruction,
        args.p20a_summary,
        args.p20a_fold,
        args.historical_results,
        args.historical_provenance,
    )
    for path in input_paths:
        if not path.is_file():
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        result = replay_historical_prediction_feedback(
            p20a_predictions_bytes=args.p20a_predictions.read_bytes(),
            p20a_reconstruction_bytes=args.p20a_reconstruction.read_bytes(),
            p20a_summary_bytes=args.p20a_summary.read_bytes(),
            p20a_fold_bytes=args.p20a_fold.read_bytes(),
            historical_results_bytes=args.historical_results.read_bytes(),
            historical_provenance_bytes=args.historical_provenance.read_bytes(),
        )
        write_historical_feedback_replay_artifacts(args.output_dir, result)
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Replayed {len(result.replay_game_ids)} historical games / "
        f"{result.feedback_result.prediction_row_count} prediction observations; "
        f"synthetic_results={str(result.claims['synthetic_results']).lower()}; "
        f"feedback_fingerprint={result.feedback_result.feedback_ledger_fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
