"""CLI for deterministic final result attachment to admitted predictions."""

import argparse
import sys
from pathlib import Path

from ...application.use_cases.attach_final_results_to_admitted_predictions import (
    attach_final_results_to_admitted_predictions,
)
from ...application.use_cases.final_result_attachment_artifacts import (
    write_final_result_attachment_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the final result attachment CLI."""
    parser = argparse.ArgumentParser(
        description="Attach deterministic final results to admitted prediction observations.",
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
        help="Path to P15C summary.json",
    )
    parser.add_argument(
        "--final-results",
        type=Path,
        required=True,
        help="Path to final results JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for attachment artifacts",
    )
    args = parser.parse_args(argv)

    snapshot_bytes = args.prediction_snapshot.read_bytes()
    summary_bytes = args.prediction_summary.read_bytes()
    final_results_bytes = args.final_results.read_bytes()

    try:
        result = attach_final_results_to_admitted_predictions(
            snapshot_bytes=snapshot_bytes,
            summary_bytes=summary_bytes,
            final_results_bytes=final_results_bytes,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_final_result_attachment_artifacts(args.output_dir, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
