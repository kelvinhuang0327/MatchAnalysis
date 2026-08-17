"""CLI interface for adapting serialized P35A pregame output to P44 normalized boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p46a_p35a_pregame_adapter import (
    P46A_ADAPTER_SOURCE_IDENTITY,
    adapt_p35a_pregame,
    adapt_p35a_pregame_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt serialized P35A pregame analysis bundle into P44 normalized pregame input."
    )
    parser.add_argument(
        "--p35a-input",
        required=True,
        type=Path,
        help="Path to P35A bundle directory or analysis.jsonl file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output normalized pregame_input.json file.",
    )
    parser.add_argument(
        "--schedule-input",
        type=Path,
        default=None,
        help="Optional path to mlb_source_snapshot.jsonl if not in bundle directory.",
    )
    parser.add_argument(
        "--source-identity",
        type=str,
        default=P46A_ADAPTER_SOURCE_IDENTITY,
        help=f"Source identity label for the normalized bundle (default: {P46A_ADAPTER_SOURCE_IDENTITY}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        adapted = adapt_p35a_pregame(
            args.p35a_input,
            schedule_input=args.schedule_input,
            source_identity=args.source_identity,
        )
        out_path = adapt_p35a_pregame_file(
            args.p35a_input,
            args.output,
            schedule_path=args.schedule_input,
            source_identity=args.source_identity,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    productive_count = len(adapted.prediction_rows)
    exclusion_count = len(adapted.exclusion_rows)
    print(
        "p46a-p35a-adapter="
        f"status=SUCCESS "
        f"source_identity={adapted.source_identity} "
        f"predictions={productive_count} "
        f"markets={len(adapted.market_rows)} "
        f"exclusions={exclusion_count} "
        f"output_file={out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
