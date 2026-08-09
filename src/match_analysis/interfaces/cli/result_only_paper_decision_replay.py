"""CLI for deterministic result-only paper-decision replay."""

import argparse
from pathlib import Path
import sys

from ...application.use_cases.build_result_only_paper_decision_replay import (
    build_result_only_paper_decision_replay,
)
from ...application.use_cases.result_only_paper_decision_artifacts import (
    write_result_only_paper_decision_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    """Run the P18A replay CLI without external side effects."""

    parser = argparse.ArgumentParser(
        description=(
            "Replay prediction-time paper decisions against final results "
            "without price or profitability calculations."
        )
    )
    parser.add_argument(
        "--prediction-snapshot",
        type=Path,
        required=True,
        help="Path to P15C admitted_observations.jsonl",
    )
    parser.add_argument(
        "--prediction-summary",
        type=Path,
        required=True,
        help="Path to the P15C snapshot summary.json",
    )
    parser.add_argument(
        "--final-results",
        type=Path,
        required=True,
        help="Path to final result observations JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for P18A artifacts",
    )
    args = parser.parse_args(argv)

    for path, label in (
        (args.prediction_snapshot, "Prediction snapshot"),
        (args.prediction_summary, "Prediction summary"),
        (args.final_results, "Final results"),
    ):
        if not path.exists():
            print(f"ERROR: {label} file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        result = build_result_only_paper_decision_replay(
            snapshot_bytes=args.prediction_snapshot.read_bytes(),
            snapshot_summary_bytes=args.prediction_summary.read_bytes(),
            final_results_bytes=args.final_results.read_bytes(),
        )
        write_result_only_paper_decision_artifacts(args.output_dir, result)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
