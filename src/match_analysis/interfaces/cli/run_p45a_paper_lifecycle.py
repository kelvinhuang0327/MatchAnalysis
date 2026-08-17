"""CLI interface for the P45A prospective paper run ledger and lifecycle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p45a_paper_run_ledger import (
    CLASSIFICATION_HISTORICAL_REHEARSAL,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
    P45A_REPORT_RELATIVE_PATH,
    create_p45a_paper_run,
    get_p45a_forward_summary,
    get_p45a_run_status,
    settle_p45a_paper_run,
)


def _resolve_repo_root(arg_root: Path | None) -> Path:
    if arg_root is not None:
        return arg_root.resolve()
    return Path.cwd().resolve()


def _run_create(args: argparse.Namespace) -> int:
    repository_root = _resolve_repo_root(args.repository_root)
    try:
        result = create_p45a_paper_run(
            repository_root,
            pregame_input=args.pregame_input,
            run_classification=args.classification,
            run_root=args.run_root,
            created_at_utc=args.created_at_utc or "2026-08-17T12:00:00Z",
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    manifest = result.manifest
    print(
        "p45a-create-run="
        f"status={result.status} "
        f"run_id={result.run_id} "
        f"classification={manifest['run_classification']} "
        f"lifecycle_state={manifest['lifecycle_state']} "
        f"universe={manifest['target_universe_count']} "
        f"eligible={manifest['eligible_decision_count']} "
        f"bet={manifest['bet_count']} "
        f"pass={manifest['pass_count']} "
        f"exclusions={manifest['exclusion_count']} "
        f"run_dir={result.run_dir}"
    )
    return 0


def _run_settle(args: argparse.Namespace) -> int:
    repository_root = _resolve_repo_root(args.repository_root)
    run_dir = args.run_dir or args.run
    if run_dir is None:
        print("ERROR: --run or --run-dir is required", file=sys.stderr)
        return 1

    try:
        result = settle_p45a_paper_run(
            repository_root,
            run_dir=run_dir,
            result_input=args.result_input,
            ledger_root=args.ledger_root,
            settled_at_utc=args.settled_at_utc or "2026-08-17T23:59:59Z",
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    print(
        "p45a-settle-run="
        f"run_id={result.run_id} "
        f"classification={result.run_classification} "
        f"lifecycle_state={result.lifecycle_state} "
        f"newly_settled={result.newly_settled_count} "
        f"total_settled={result.total_settled_count} "
        f"pending={result.pending_count} "
        f"wins={summary['win_count']} "
        f"losses={summary['loss_count']} "
        f"units_risked={summary['units_risked']} "
        f"net={summary['net_paper_units']} "
        f"roi={summary['descriptive_roi']} "
        f"forward_sample_count={result.forward_summary['forward_sample_count']}"
    )
    return 0


def _run_status(args: argparse.Namespace) -> int:
    repository_root = _resolve_repo_root(args.repository_root)
    run_dir = args.run_dir or args.run
    if run_dir is None:
        print("ERROR: --run or --run-dir is required", file=sys.stderr)
        return 1

    try:
        status = get_p45a_run_status(
            repository_root,
            run_dir=run_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "p45a-status="
        f"run_id={status['run_id']} "
        f"classification={status['run_classification']} "
        f"lifecycle_state={status['lifecycle_state']} "
        f"eligible={status['eligible_decision_count']} "
        f"bet={status['bet_count']} "
        f"pass={status['pass_count']} "
        f"settled_total={status['settled_total_count']} "
        f"settled_bet={status['settled_bet_count']} "
        f"pending={status['pending_count']}"
    )
    return 0


def _run_summary(args: argparse.Namespace) -> int:
    repository_root = _resolve_repo_root(args.repository_root)
    try:
        summary = get_p45a_forward_summary(
            repository_root,
            ledger_root=args.ledger_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "p45a-summary="
        f"forward_sample_count={summary['forward_sample_count']} "
        f"runs={summary['run_count']} "
        f"decisions={summary['frozen_decision_count']} "
        f"bet={summary['bet_count']} "
        f"pass={summary['pass_count']} "
        f"settled_bet={summary['settled_bet_count']} "
        f"wins={summary['wins']} "
        f"losses={summary['losses']} "
        f"units_risked={summary['paper_units_risked']} "
        f"net={summary['net_paper_units']} "
        f"roi={summary['descriptive_roi']} "
        f"max_drawdown={summary['max_drawdown']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-run
    create_parser = subparsers.add_parser(
        "create-run",
        help="Create and freeze one immutable paper run from normalized pregame input",
    )
    create_parser.add_argument(
        "--pregame-input",
        type=Path,
        required=True,
        help="Path to normalized pregame input bundle (pregame_input.json)",
    )
    create_parser.add_argument(
        "--classification",
        choices=[CLASSIFICATION_HISTORICAL_REHEARSAL, CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER],
        default=CLASSIFICATION_HISTORICAL_REHEARSAL,
        help="Run classification (HISTORICAL_REHEARSAL or PROSPECTIVE_FORWARD_PAPER)",
    )
    create_parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Custom root directory for storing run artifacts",
    )
    create_parser.add_argument(
        "--created-at-utc",
        type=str,
        default=None,
        help="Explicit freeze timestamp in UTC",
    )
    create_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Repository root directory",
    )
    create_parser.set_defaults(func=_run_create)

    # settle-run
    settle_parser = subparsers.add_parser(
        "settle-run",
        help="Settle available final results for a frozen paper run and update ledger",
    )
    settle_parser.add_argument(
        "--run",
        "--run-dir",
        dest="run_dir",
        type=Path,
        required=True,
        help="Path to the frozen run directory",
    )
    settle_parser.add_argument(
        "--result-input",
        type=Path,
        required=True,
        help="Path to normalized result observations (results.jsonl)",
    )
    settle_parser.add_argument(
        "--ledger-root",
        type=Path,
        default=None,
        help="Ledger root directory",
    )
    settle_parser.add_argument(
        "--settled-at-utc",
        type=str,
        default=None,
        help="Explicit settlement timestamp in UTC",
    )
    settle_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Repository root directory",
    )
    settle_parser.set_defaults(func=_run_settle)

    # status
    status_parser = subparsers.add_parser(
        "status",
        help="Inspect status and metrics of a paper run",
    )
    status_parser.add_argument(
        "--run",
        "--run-dir",
        dest="run_dir",
        type=Path,
        required=True,
        help="Path to the run directory",
    )
    status_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Repository root directory",
    )
    status_parser.set_defaults(func=_run_status)

    # summary
    summary_parser = subparsers.add_parser(
        "summary",
        help="Report cumulative forward paper descriptive summary",
    )
    summary_parser.add_argument(
        "--ledger-root",
        type=Path,
        default=None,
        help="Ledger root directory",
    )
    summary_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Repository root directory",
    )
    summary_parser.set_defaults(func=_run_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
