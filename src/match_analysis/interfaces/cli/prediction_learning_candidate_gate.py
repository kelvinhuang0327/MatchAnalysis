"""CLI for the deterministic P21A learning-candidate eligibility gate."""

import argparse
from pathlib import Path
import sys

from ...application.use_cases.assess_prediction_learning_candidates import (
    assess_prediction_learning_candidates,
)
from ...application.use_cases.prediction_learning_candidate_artifacts import (
    write_prediction_learning_candidate_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    """Assess committed P20B feedback and export only eligible candidates."""
    parser = argparse.ArgumentParser(
        description=(
            "Assess committed P20B feedback for the P21A non-synthetic "
            "learning-candidate boundary."
        )
    )
    parser.add_argument(
        "--feedback-ledger",
        type=Path,
        required=True,
        help="Path to committed P20B feedback.jsonl",
    )
    parser.add_argument(
        "--feedback-summary",
        type=Path,
        required=True,
        help="Path to committed P20B summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for P21A artifacts",
    )
    args = parser.parse_args(argv)

    for path, label in (
        (args.feedback_ledger, "Feedback ledger"),
        (args.feedback_summary, "Feedback summary"),
    ):
        if not path.is_file():
            print(f"ERROR: {label} file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        result = assess_prediction_learning_candidates(
            feedback_bytes=args.feedback_ledger.read_bytes(),
            feedback_summary_bytes=args.feedback_summary.read_bytes(),
        )
        write_prediction_learning_candidate_artifacts(args.output_dir, result)
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Assessed {len(result.assessments)} feedback rows; "
        f"eligible={len(result.candidates)}; "
        f"excluded={len(result.assessments) - len(result.candidates)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
