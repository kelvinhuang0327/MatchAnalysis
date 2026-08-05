"""CLI for deterministic prediction feedback ledger generation."""

import argparse
from pathlib import Path
import sys

from ...application.use_cases.build_prediction_feedback_ledger import (
    build_prediction_feedback_ledger,
)
from ...application.use_cases.prediction_feedback_artifacts import (
    write_prediction_feedback_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the prediction feedback ledger CLI."""
    parser = argparse.ArgumentParser(
        description="Build deterministic prediction feedback ledger from P15C snapshot, P16A attachments, and P16B evaluations.",
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
        help="Path to P15C snapshot summary.json",
    )
    parser.add_argument(
        "--result-attachments",
        type=Path,
        required=True,
        help="Path to P16A attachments.jsonl",
    )
    parser.add_argument(
        "--result-summary",
        type=Path,
        required=True,
        help="Path to P16A summary.json",
    )
    parser.add_argument(
        "--evaluations",
        type=Path,
        required=True,
        help="Path to P16B evaluations.jsonl",
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        required=True,
        help="Path to P16B evaluation summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for feedback ledger artifacts",
    )
    args = parser.parse_args(argv)

    for attr, label in (
        ("prediction_snapshot", "Prediction snapshot"),
        ("prediction_summary", "Prediction summary"),
        ("result_attachments", "Result attachments"),
        ("result_summary", "Result summary"),
        ("evaluations", "Evaluations"),
        ("evaluation_summary", "Evaluation summary"),
    ):
        path = getattr(args, attr)
        if not path.exists():
            print(f"ERROR: {label} file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        result = build_prediction_feedback_ledger(
            snapshot_bytes=args.prediction_snapshot.read_bytes(),
            snapshot_summary_bytes=args.prediction_summary.read_bytes(),
            attachments_bytes=args.result_attachments.read_bytes(),
            attachment_summary_bytes=args.result_summary.read_bytes(),
            evaluations_bytes=args.evaluations.read_bytes(),
            evaluation_summary_bytes=args.evaluation_summary.read_bytes(),
        )
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_prediction_feedback_artifacts(args.output_dir, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
