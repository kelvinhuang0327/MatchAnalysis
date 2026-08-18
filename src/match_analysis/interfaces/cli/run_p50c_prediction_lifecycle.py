"""CLI interface for the P50C prospective prediction shadow ledger and lifecycle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p50c_prediction_run_ledger import (
    CLASSIFICATION_HISTORICAL_REHEARSAL,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P50C_REPORT_RELATIVE_PATH,
    create_p50c_prediction_run,
    get_p50c_forward_summary,
    get_p50c_run_status,
    settle_p50c_prediction_run,
)


def _resolve_repo_root(arg_root: Path | None) -> Path:
    if arg_root is not None:
        return arg_root.resolve()
    return Path.cwd().resolve()


def _run_create(args: argparse.Namespace) -> int:
    repository_root = _resolve_repo_root(args.repository_root)
    try:
        result = create_p50c_prediction_run(
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
        "p50c-create-run="
        f"status={result.status} "
        f"run_id={result.run_id} "
        f"classification={manifest['run_classification']} "
        f"lifecycle_state={manifest['lifecycle_state']} "
        f"universe={manifest['target_universe_count']} "
        f"eligible={manifest['eligible_prediction_count']} "
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
        result = settle_p50c_prediction_run(
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
        "p50c-settle-run="
        f"run_id={result.run_id} "
        f"classification={result.run_classification} "
        f"lifecycle_state={result.lifecycle_state} "
        f"newly_settled={result.newly_settled_count} "
        f"total_settled={result.total_settled_count} "
        f"pending={result.pending_count} "
        f"correct={summary['correct_count']} "
        f"incorrect={summary['incorrect_count']} "
        f"accuracy={summary['accuracy']} "
        f"brier={summary['brier_score']} "
        f"log_loss={summary['log_loss']} "
        f"ece={summary['expected_calibration_error']} "
        f"prediction_forward_sample_count={result.forward_summary['PREDICTION_FORWARD_SAMPLE_COUNT']}"
    )
    return 0


def _run_status(args: argparse.Namespace) -> int:
    repository_root = _resolve_repo_root(args.repository_root)
    run_dir = args.run_dir or args.run
    if run_dir is None:
        print("ERROR: --run or --run-dir is required", file=sys.stderr)
        return 1

    try:
        status = get_p50c_run_status(
            repository_root,
            run_dir=run_dir,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "p50c-status="
        f"run_id={status['run_id']} "
        f"classification={status['run_classification']} "
        f"lifecycle_state={status['lifecycle_state']} "
        f"eligible={status['eligible_prediction_count']} "
        f"settled_total={status['settled_total_count']} "
        f"pending={status['pending_count']}"
    )
    return 0


def _run_summary(args: argparse.Namespace) -> int:
    repository_root = _resolve_repo_root(args.repository_root)
    try:
        summary = get_p50c_forward_summary(
            repository_root,
            ledger_root=args.ledger_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "p50c-summary="
        f"prediction_forward_sample_count={summary['PREDICTION_FORWARD_SAMPLE_COUNT']} "
        f"runs={summary['run_count']} "
        f"frozen_predictions={summary['frozen_prediction_count']} "
        f"settled_predictions={summary['settled_prediction_count']} "
        f"correct={summary['correct_count']} "
        f"incorrect={summary['incorrect_count']} "
        f"accuracy={summary['accuracy']} "
        f"brier_score={summary['brier_score']} "
        f"log_loss={summary['log_loss']} "
        f"ece={summary['expected_calibration_error']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Path to repository root",
    )

    parser = argparse.ArgumentParser(
        description="P50C prospective prediction shadow ledger and lifecycle CLI",
        parents=[common_parent],
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # create-run subcommand
    create_parser = subparsers.add_parser(
        "create-run",
        parents=[common_parent],
        help="Create and freeze prediction run",
    )
    create_parser.add_argument(
        "--pregame-input",
        type=Path,
        required=True,
        help="Path to normalized pregame input file",
    )
    create_parser.add_argument(
        "--classification",
        choices=[CLASSIFICATION_HISTORICAL_REHEARSAL, CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION],
        default=CLASSIFICATION_HISTORICAL_REHEARSAL,
        help="Run classification",
    )
    create_parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Directory to place runs",
    )
    create_parser.add_argument(
        "--created-at-utc",
        type=str,
        default="2026-08-17T12:00:00Z",
        help="Freeze creation timestamp in UTC",
    )

    # settle-run subcommand
    settle_parser = subparsers.add_parser(
        "settle-run",
        parents=[common_parent],
        help="Settle game results against frozen prediction run",
    )
    settle_parser.add_argument(
        "--run",
        "--run-dir",
        dest="run_dir",
        type=Path,
        required=True,
        help="Directory of the frozen prediction run",
    )
    settle_parser.add_argument(
        "--result-input",
        type=Path,
        required=True,
        help="Path to normalized results file",
    )
    settle_parser.add_argument(
        "--ledger-root",
        type=Path,
        default=None,
        help="Directory containing the prediction ledger",
    )
    settle_parser.add_argument(
        "--settled-at-utc",
        type=str,
        default="2026-08-17T23:59:59Z",
        help="Settlement timestamp in UTC",
    )

    # status subcommand
    status_parser = subparsers.add_parser(
        "status",
        parents=[common_parent],
        help="Inspect status of a prediction run",
    )
    status_parser.add_argument(
        "--run",
        "--run-dir",
        dest="run_dir",
        type=Path,
        required=True,
        help="Directory of the prediction run",
    )

    # summary subcommand
    summary_parser = subparsers.add_parser(
        "summary",
        parents=[common_parent],
        help="Print cumulative forward prediction summary",
    )
    summary_parser.add_argument(
        "--ledger-root",
        type=Path,
        default=None,
        help="Directory containing the prediction ledger",
    )

    # Top-level action flags for alternative invocations
    parser.add_argument("--create-run", action="store_true", help="Execute create-run action")
    parser.add_argument("--settle-run", action="store_true", help="Execute settle-run action")
    parser.add_argument("--status", action="store_true", help="Execute status action")
    parser.add_argument("--summary", action="store_true", help="Execute summary action")

    parser.add_argument("--pregame-input", type=Path, default=None)
    parser.add_argument(
        "--classification",
        choices=[CLASSIFICATION_HISTORICAL_REHEARSAL, CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION],
        default=CLASSIFICATION_HISTORICAL_REHEARSAL,
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--created-at-utc", type=str, default=None)
    parser.add_argument("--run", "--run-dir", dest="run", type=Path, default=None)
    parser.add_argument("--result-input", type=Path, default=None)
    parser.add_argument("--ledger-root", type=Path, default=None)
    parser.add_argument("--settled-at-utc", type=str, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "create-run" or args.create_run:
        if args.pregame_input is None:
            print("ERROR: --pregame-input is required for create-run", file=sys.stderr)
            return 1
        return _run_create(args)

    if args.command == "settle-run" or args.settle_run:
        if args.result_input is None:
            print("ERROR: --result-input is required for settle-run", file=sys.stderr)
            return 1
        return _run_settle(args)

    if args.command == "status" or args.status:
        return _run_status(args)

    if args.command == "summary" or args.summary:
        return _run_summary(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
