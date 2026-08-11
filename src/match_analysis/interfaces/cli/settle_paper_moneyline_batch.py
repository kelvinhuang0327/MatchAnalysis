"""CLI for the offline P25A promoted paper Moneyline feedback loop."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.paper_moneyline_feedback_artifacts import (
    write_paper_moneyline_feedback_artifacts,
)
from ...application.use_cases.settle_paper_moneyline_batch import (
    settle_paper_moneyline_batch,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Settle the committed P24C promoted paper Moneyline batch offline "
            "through P16A, P16B, and P17A."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
        help="MatchAnalysis repository root containing the committed P24C authority",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/p25a_promoted_moneyline_feedback"),
        help="P25A artifact output directory",
    )
    args = parser.parse_args(argv)
    try:
        result = settle_paper_moneyline_batch(args.repository_root)
        write_paper_moneyline_feedback_artifacts(args.output_dir, result)
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Settled {len(result.settled_predictions)} P24C predictions; "
        f"feedback_rows={result.feedback_result.prediction_row_count}; "
        f"accuracy={result.accuracy}; mean_brier={result.mean_brier}; "
        f"feedback_fingerprint={result.feedback_result.feedback_ledger_fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
