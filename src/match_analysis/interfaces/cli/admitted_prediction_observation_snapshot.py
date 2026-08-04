"""CLI for building admitted prediction observation snapshot from P15B1 artifacts."""

import argparse
import sys
from pathlib import Path

from ...application.use_cases.build_admitted_prediction_observation_snapshot import (
    build_admitted_prediction_observation_snapshot,
)
from ...application.use_cases.admitted_prediction_observation_artifacts import (
    write_admitted_prediction_observation_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the admitted prediction observation snapshot CLI."""
    parser = argparse.ArgumentParser(
        description="Build admitted prediction observation snapshot from P15B1 artifacts.",
    )
    parser.add_argument(
        "--admission-results",
        type=Path,
        required=True,
        help="Path to P15B1 results.jsonl",
    )
    parser.add_argument(
        "--admission-summary",
        type=Path,
        required=True,
        help="Path to P15B1 summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for snapshot artifacts",
    )
    args = parser.parse_args(argv)

    results_bytes = args.admission_results.read_bytes()
    summary_bytes = args.admission_summary.read_bytes()

    try:
        result = build_admitted_prediction_observation_snapshot(
            results_bytes=results_bytes,
            summary_bytes=summary_bytes,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_admitted_prediction_observation_artifacts(args.output_dir, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
