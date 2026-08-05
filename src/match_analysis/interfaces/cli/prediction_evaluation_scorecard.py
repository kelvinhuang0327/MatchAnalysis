"""CLI for deterministic prediction evaluation scorecard generation."""

import argparse
from pathlib import Path
import sys

from ...application.use_cases.build_prediction_evaluation_scorecard import (
    build_prediction_evaluation_scorecard,
)
from ...application.use_cases.prediction_evaluation_artifacts import (
    write_prediction_evaluation_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the prediction evaluation scorecard CLI."""
    parser = argparse.ArgumentParser(
        description="Build deterministic prediction evaluation scorecard from P16A attachments and P15C snapshot.",
    )
    parser.add_argument(
        "--attachments",
        type=Path,
        required=True,
        help="Path to P16A attachments.jsonl",
    )
    parser.add_argument(
        "--attachment-summary",
        type=Path,
        required=True,
        help="Path to P16A summary.json",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("report/p15c_admitted_prediction_observation_snapshot/admitted_observations.jsonl"),
        help="Path to P15C admitted_observations.jsonl",
    )
    parser.add_argument(
        "--snapshot-summary",
        type=Path,
        default=Path("report/p15c_admitted_prediction_observation_snapshot/summary.json"),
        help="Path to P15C snapshot summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation artifacts",
    )
    args = parser.parse_args(argv)

    if not args.attachments.exists():
        print(f"ERROR: Attachments file does not exist: {args.attachments}", file=sys.stderr)
        return 1

    if not args.attachment_summary.exists():
        print(f"ERROR: Attachment summary file does not exist: {args.attachment_summary}", file=sys.stderr)
        return 1

    if not args.snapshot.exists():
        print(f"ERROR: Snapshot file does not exist: {args.snapshot}", file=sys.stderr)
        return 1

    if not args.snapshot_summary.exists():
        print(f"ERROR: Snapshot summary file does not exist: {args.snapshot_summary}", file=sys.stderr)
        return 1

    try:
        attachments_bytes = args.attachments.read_bytes()
        attachment_summary_bytes = args.attachment_summary.read_bytes()
        snapshot_bytes = args.snapshot.read_bytes()
        snapshot_summary_bytes = args.snapshot_summary.read_bytes()

        result = build_prediction_evaluation_scorecard(
            attachments_bytes=attachments_bytes,
            attachment_summary_bytes=attachment_summary_bytes,
            snapshot_bytes=snapshot_bytes,
            snapshot_summary_bytes=snapshot_summary_bytes,
        )
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_prediction_evaluation_artifacts(args.output_dir, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
