"""CLI interface for admitting external frozen P35A bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p47a_external_bundle_admission import (
    P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
    admit_external_p35a_bundle,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Admit external frozen P35A pregame bundle into repository-native admitted authority."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    admit_parser = subparsers.add_parser("admit", help="Admit an external P35A frozen bundle.")
    admit_parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Path to external frozen P35A bundle directory.",
    )
    admit_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root directory for admitted bundle records.",
    )
    admit_parser.add_argument(
        "--source-identity",
        type=str,
        default=P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
        help=f"Source identity label (default: {P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY}).",
    )
    admit_parser.add_argument(
        "--admitted-at-utc",
        type=str,
        default="2026-08-17T12:00:00Z",
        help="UTC timestamp metadata for admission.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "admit":
        try:
            admitted = admit_external_p35a_bundle(
                args.bundle,
                admission_root=args.output_root,
                source_identity=args.source_identity,
                admitted_at_utc=args.admitted_at_utc,
            )
        except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        print(
            "p47a-admit="
            f"status={admitted.status} "
            f"admitted_bundle_id={admitted.admitted_bundle_id} "
            f"run_id={admitted.run_id} "
            f"target_date={admitted.target_date} "
            f"bundle_fingerprint={admitted.bundle_fingerprint} "
            f"productive_rows={admitted.productive_row_count} "
            f"excluded_rows={admitted.excluded_row_count} "
            f"p46_compatibility=PASS "
            f"normalized_output={admitted.normalized_pregame_path}"
        )
        return 0

    print(f"ERROR: unknown subcommand {args.subcommand}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
