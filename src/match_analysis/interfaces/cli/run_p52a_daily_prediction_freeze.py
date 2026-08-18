"""CLI interface for P52A daily Moneyline prospective prediction freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ...application.use_cases.p52a_daily_prediction_freeze import (
    execute_daily_moneyline_prediction_freeze,
)


def _resolve_repo_root(arg_root: Path | None) -> Path:
    if arg_root is not None:
        return arg_root.resolve()
    return Path.cwd().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_p52a_daily_prediction_freeze",
        description="Execute genuine daily MLB Moneyline pregame prediction freeze via P50C authority.",
    )
    parser.add_argument(
        "--target-date",
        required=True,
        help="Target slate date in YYYY-MM-DD format (e.g. 2026-08-18).",
    )
    parser.add_argument(
        "--as-of-utc",
        default=None,
        help="Explicit freeze timestamp in ISO-8601 UTC format (e.g. 2026-08-18T13:37:42Z). Defaults to current UTC.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Repository root path. Defaults to current working directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root path for runtime pregame intake and ledger outputs. Defaults to repository-root.",
    )
    parser.add_argument(
        "--history-start-date",
        default=None,
        help="Earliest schedule history date to fetch for recent team performance. Defaults to season start.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print complete structured result as JSON to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = _resolve_repo_root(args.repository_root)
    output_root = args.output_root.resolve() if args.output_root else repo_root

    try:
        result = execute_daily_moneyline_prediction_freeze(
            target_date=args.target_date,
            as_of_utc=args.as_of_utc,
            repository_root=repo_root,
            output_root=output_root,
            history_start_date=args.history_start_date,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            "p52a-daily-prediction-freeze="
            f"target_date={result.target_date} "
            f"as_of_utc={result.as_of_utc} "
            f"target_games={result.target_games_count} "
            f"eligible={result.eligible_predictions_count} "
            f"exclusions={result.exclusion_count} "
            f"run_id={result.run_id} "
            f"freeze_status={result.freeze_status} "
            f"pending_frozen={result.pending_count} "
            f"settled_prediction_forward_sample_count={result.settled_prediction_forward_sample_count} "
            f"betting_forward_sample_count={result.betting_forward_sample_count}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
