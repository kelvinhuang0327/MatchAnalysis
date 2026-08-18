"""CLI interface for admitting an external final-result bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p49a_external_final_result_admission import (
    P49A_DEFAULT_SOURCE_IDENTITY,
    admit_external_final_result_bundle,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, admit, and normalize an external final-result bundle."
    )
    parser.add_argument(
        "--bundle",
        "--result-bundle",
        dest="bundle",
        required=True,
        type=Path,
        help="Path to external final-result bundle directory.",
    )
    parser.add_argument(
        "--admission-root",
        type=Path,
        default=None,
        help="Optional root directory for admitted bundle records.",
    )
    parser.add_argument(
        "--source-identity",
        type=str,
        default=P49A_DEFAULT_SOURCE_IDENTITY,
        help=f"Source identity label (default: {P49A_DEFAULT_SOURCE_IDENTITY}).",
    )
    parser.add_argument(
        "--admitted-at-utc",
        type=str,
        default="2026-08-18T23:59:59Z",
        help="UTC timestamp metadata for admission.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        admitted = admit_external_final_result_bundle(
            args.bundle,
            admission_root=args.admission_root,
            source_identity=args.source_identity,
            admitted_at_utc=args.admitted_at_utc,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "p49a-admit="
        f"status={admitted.status} "
        f"admitted_bundle_id={admitted.admitted_bundle_id} "
        f"bundle_fingerprint={admitted.bundle_fingerprint} "
        f"target_date={admitted.target_date} "
        f"source_identity={admitted.source_identity} "
        f"results={admitted.final_result_count} "
        f"imported_dir={admitted.imported_bundle_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
