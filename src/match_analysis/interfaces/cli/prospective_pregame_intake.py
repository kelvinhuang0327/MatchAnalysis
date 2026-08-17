"""CLI interface for P48A atomic prospective pregame intake."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p48a_atomic_prospective_pregame_intake import (
    P48A_DEFAULT_SOURCE_IDENTITY,
    intake_prospective_pregame_bundle,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P48A atomic prospective pregame intake orchestration."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    intake_parser = subparsers.add_parser(
        "prospective-intake",
        help="Admit external P35A bundle and atomically create prospective paper run.",
    )
    intake_parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Path to external frozen P35A bundle directory.",
    )
    intake_parser.add_argument(
        "--admission-root",
        type=Path,
        default=None,
        help="Optional root directory for admitted bundle records.",
    )
    intake_parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional root directory for paper runs.",
    )
    intake_parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Optional repository root directory.",
    )
    intake_parser.add_argument(
        "--source-identity",
        type=str,
        default=P48A_DEFAULT_SOURCE_IDENTITY,
        help=f"Source identity label (default: {P48A_DEFAULT_SOURCE_IDENTITY}).",
    )
    intake_parser.add_argument(
        "--intake-timestamp-utc",
        "--created-at-utc",
        dest="intake_timestamp_utc",
        type=str,
        default="2026-08-17T12:00:00Z",
        help="UTC timestamp metadata for intake.",
    )
    intake_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate external bundle and prospective eligibility without creating P45 run.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate external P35A bundle and prospective eligibility without creating P45 run.",
    )
    validate_parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Path to external frozen P35A bundle directory.",
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
        default=P48A_DEFAULT_SOURCE_IDENTITY,
        help=f"Source identity label (default: {P48A_DEFAULT_SOURCE_IDENTITY}).",
    )
    validate_parser.add_argument(
        "--intake-timestamp-utc",
        "--created-at-utc",
        dest="intake_timestamp_utc",
        type=str,
        default="2026-08-17T12:00:00Z",
        help="UTC timestamp metadata for validation.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    validate_only = getattr(args, "validate_only", False) or args.subcommand == "validate"
    run_root = getattr(args, "run_root", None)

    try:
        result = intake_prospective_pregame_bundle(
            args.bundle,
            repository_root=args.repository_root,
            admission_root=args.admission_root,
            run_root=run_root,
            source_identity=args.source_identity,
            intake_timestamp_utc=args.intake_timestamp_utc,
            validate_only=validate_only,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if result.status == "VALIDATED":
        print(
            "p48a-intake="
            f"status=VALIDATED "
            f"classification={result.run_classification} "
            f"lifecycle_state={result.lifecycle_state} "
            f"admitted_bundle_id={result.admitted_bundle_id} "
            f"bundle_fingerprint={result.bundle_fingerprint} "
            f"normalized_input_fingerprint={result.normalized_input_fingerprint} "
            f"target_date={result.target_date} "
            f"universe={result.target_universe_count} "
            f"eligible={result.eligible_decision_count} "
            f"exclusions={result.exclusion_count}"
        )
        return 0

    print(
        "p48a-intake="
        f"status={result.status} "
        f"run_id={result.paper_run_id} "
        f"classification={result.run_classification} "
        f"lifecycle_state={result.lifecycle_state} "
        f"admitted_bundle_id={result.admitted_bundle_id} "
        f"bundle_fingerprint={result.bundle_fingerprint} "
        f"normalized_input_fingerprint={result.normalized_input_fingerprint} "
        f"decision_bundle_fingerprint={result.decision_bundle_fingerprint} "
        f"target_date={result.target_date} "
        f"universe={result.target_universe_count} "
        f"eligible={result.eligible_decision_count} "
        f"bet={result.bet_count} "
        f"pass={result.pass_count} "
        f"exclusions={result.exclusion_count} "
        f"run_dir={result.run_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
