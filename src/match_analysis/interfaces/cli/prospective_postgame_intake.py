"""CLI interface for P49A prospective postgame settlement intake."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p49a_external_final_result_admission import (
    P49A_DEFAULT_SOURCE_IDENTITY,
    intake_prospective_postgame_results,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P49A prospective postgame final-result settlement intake orchestration."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    settle_parser = subparsers.add_parser(
        "postgame-intake",
        help="Admit external final result bundle and settle against frozen paper run.",
    )
    settle_parser.add_argument(
        "--bundle",
        "--result-bundle",
        dest="bundle",
        required=True,
        type=Path,
        help="Path to external authoritative final-result bundle directory.",
    )
    settle_parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to frozen paper run directory.",
    )
    settle_parser.add_argument(
        "--admission-root",
        type=Path,
        default=None,
        help="Optional root directory for admitted bundle records.",
    )
    settle_parser.add_argument(
        "--ledger-root",
        type=Path,
        default=None,
        help="Optional root directory for forward paper ledger.",
    )
    settle_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Optional repository root directory.",
    )
    settle_parser.add_argument(
        "--source-identity",
        type=str,
        default=P49A_DEFAULT_SOURCE_IDENTITY,
        help=f"Source identity label (default: {P49A_DEFAULT_SOURCE_IDENTITY}).",
    )
    settle_parser.add_argument(
        "--settled-at-utc",
        type=str,
        default="2026-08-18T23:59:59Z",
        help="UTC timestamp metadata for settlement.",
    )
    settle_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate external result bundle and matching without executing settlement.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate external final result bundle and matching without executing settlement.",
    )
    validate_parser.add_argument(
        "--bundle",
        "--result-bundle",
        dest="bundle",
        required=True,
        type=Path,
        help="Path to external authoritative final-result bundle directory.",
    )
    validate_parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to frozen paper run directory.",
    )
    validate_parser.add_argument(
        "--admission-root",
        type=Path,
        default=None,
        help="Optional root directory for admitted bundle records.",
    )
    validate_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Optional repository root directory.",
    )
    validate_parser.add_argument(
        "--source-identity",
        type=str,
        default=P49A_DEFAULT_SOURCE_IDENTITY,
        help=f"Source identity label (default: {P49A_DEFAULT_SOURCE_IDENTITY}).",
    )
    validate_parser.add_argument(
        "--settled-at-utc",
        type=str,
        default="2026-08-18T23:59:59Z",
        help="UTC timestamp metadata for validation.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    validate_only = getattr(args, "validate_only", False) or args.subcommand == "validate"
    ledger_root = getattr(args, "ledger_root", None)

    try:
        result = intake_prospective_postgame_results(
            args.bundle,
            paper_run_dir=args.run_dir,
            repository_root=args.repository_root,
            admission_root=args.admission_root,
            ledger_root=ledger_root,
            source_identity=args.source_identity,
            settled_at_utc=args.settled_at_utc,
            validate_only=validate_only,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if result.status == "VALIDATED":
        print(
            "p49a-settle="
            f"status=VALIDATED "
            f"run_id={result.paper_run_id} "
            f"classification={result.run_classification} "
            f"lifecycle_state={result.lifecycle_state} "
            f"admitted_bundle_id={result.admitted_bundle_id} "
            f"bundle_fingerprint={result.bundle_fingerprint} "
            f"target_date={result.admitted_bundle.target_date} "
            f"results={result.admitted_bundle.final_result_count} "
            f"pending={result.pending_count}"
        )
        return 0

    print(
        "p49a-settle="
        f"status={result.status} "
        f"run_id={result.paper_run_id} "
        f"classification={result.run_classification} "
        f"lifecycle_state={result.lifecycle_state} "
        f"admitted_bundle_id={result.admitted_bundle_id} "
        f"bundle_fingerprint={result.bundle_fingerprint} "
        f"target_date={result.admitted_bundle.target_date} "
        f"results={result.admitted_bundle.final_result_count} "
        f"newly_settled={result.newly_settled_count} "
        f"total_settled={result.total_settled_count} "
        f"settled_bet={result.settled_bet_count} "
        f"settled_pass={result.settled_pass_count} "
        f"pending={result.pending_count} "
        f"wins={result.win_count} "
        f"losses={result.loss_count} "
        f"net_units={result.net_paper_units} "
        f"roi={result.descriptive_roi or 'None'} "
        f"max_drawdown={result.max_drawdown} "
        f"forward_samples={result.forward_sample_count} "
        f"run_dir={result.run_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
