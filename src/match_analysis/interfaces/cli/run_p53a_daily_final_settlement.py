"""CLI interface for P53A daily Moneyline prospective prediction FINAL settlement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ...application.use_cases.p53a_daily_final_settlement import (
    execute_daily_moneyline_final_settlement,
)


def _resolve_repo_root(arg_root: Path | None) -> Path:
    if arg_root is not None:
        return arg_root.resolve()
    return Path.cwd().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_p53a_daily_final_settlement",
        description="Execute genuine daily MLB Moneyline postgame FINAL settlement via P50C authority.",
    )
    parser.add_argument(
        "--run",
        "--run-dir",
        dest="run",
        default=None,
        help="Directory or ID of the frozen prediction run (e.g. p50c_run_fa367308eedb4c623680d7d2667273b0).",
    )
    parser.add_argument(
        "--target-date",
        default=None,
        help="Target slate date in YYYY-MM-DD format (e.g. 2026-08-18).",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Repository root path. Defaults to current working directory.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory containing prediction runs. Defaults to canonical runtime shadow ledger runs.",
    )
    parser.add_argument(
        "--ledger-root",
        type=Path,
        default=None,
        help="Directory containing the forward prediction ledger. Defaults to canonical runtime ledger.",
    )
    parser.add_argument(
        "--final-results",
        "--result-input",
        dest="final_results",
        type=Path,
        default=None,
        help="Offline normalized final results input file (JSONL).",
    )
    parser.add_argument(
        "--observed-at-utc",
        default=None,
        help="Explicit observation timestamp in ISO-8601 UTC format. Defaults to current UTC.",
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
    runs_root = args.runs_root.resolve() if args.runs_root else None
    ledger_root = args.ledger_root.resolve() if args.ledger_root else None

    try:
        result = execute_daily_moneyline_final_settlement(
            run=args.run,
            target_date=args.target_date,
            repository_root=repo_root,
            runs_root=runs_root,
            ledger_root=ledger_root,
            result_input=args.final_results,
            observed_at_utc=args.observed_at_utc,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            "p53a-daily-final-settlement="
            f"run_id={result.run_id} "
            f"target_date={result.target_date} "
            f"lifecycle_state={result.lifecycle_state} "
            f"eligible={result.eligible_prediction_count} "
            f"newly_settled={result.newly_settled_count} "
            f"total_settled={result.total_settled_count} "
            f"pending={result.pending_count} "
            f"final_discovered={result.final_results_discovered} "
            f"non_final={result.non_final_games_count} "
            f"prediction_forward_sample_count={result.prediction_forward_sample_count} "
            f"betting_forward_sample_count={result.betting_forward_sample_count} "
            f"status={result.status}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
