"""Replay the offline P29A Moneyline closing-price/CLV diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.generate_moneyline_market_movement import (
    generate_moneyline_market_movement,
)
from ...application.use_cases.moneyline_market_movement_artifacts import (
    load_json_object,
    load_jsonl,
    write_moneyline_market_movement_artifacts,
)
from ...infrastructure.legacy_betting_pool.tsl_odds_history import (
    load_tsl_odds_history,
)


DEFAULT_P28AB_REPORT_DIR = Path("report/p28ab_tsl_aligned_moneyline_edge")
DEFAULT_TSL_HISTORY = Path(
    "data/fixtures/p28ab_tsl_aligned_moneyline_edge/tsl_odds_history.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("report/p29a_moneyline_market_movement")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument(
        "--raw-cohort",
        type=Path,
        default=DEFAULT_P28AB_REPORT_DIR / "raw_cohort.jsonl",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=DEFAULT_P28AB_REPORT_DIR / "prices.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_P28AB_REPORT_DIR / "predictions.jsonl",
    )
    parser.add_argument(
        "--p28ab-summary",
        type=Path,
        default=DEFAULT_P28AB_REPORT_DIR / "summary.json",
    )
    parser.add_argument(
        "--p28ab-source-manifest",
        type=Path,
        default=DEFAULT_P28AB_REPORT_DIR / "source_manifest.json",
    )
    parser.add_argument(
        "--tsl-history",
        type=Path,
        default=DEFAULT_TSL_HISTORY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="required: P29A has no network or provider path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.offline:
        print("ERROR: P29A is offline-only; pass --offline", file=sys.stderr)
        return 1
    try:
        repository_root = args.repository_root.resolve()

        def rooted(path: Path) -> Path:
            return path if path.is_absolute() else repository_root / path

        snapshot = load_tsl_odds_history(rooted(args.tsl_history))
        result = generate_moneyline_market_movement(
            p28ab_raw_cohort=load_jsonl(rooted(args.raw_cohort)),
            p28ab_prices=load_jsonl(rooted(args.prices)),
            p28ab_predictions=load_jsonl(rooted(args.predictions)),
            p28ab_summary=load_json_object(rooted(args.p28ab_summary)),
            p28ab_source_manifest=load_json_object(
                rooted(args.p28ab_source_manifest)
            ),
            tsl_rows=snapshot.rows,
            tsl_raw_sha256=snapshot.raw_sha256,
        )
        hashes = write_moneyline_market_movement_artifacts(
            rooted(args.output_dir),
            result=result,
        )
    except (OSError, TypeError, ValueError, RuntimeError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    print(
        f"paired={summary['paired_p28ab_game_count']} "
        f"closing_available={summary['closing_price_available_count']} "
        f"closing_unavailable={summary['closing_price_unavailable_count']} "
        f"movement={summary['market_movement_row_count']} "
        "deterministic=True offline=True"
    )
    print(f"artifacts={len(hashes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("build_parser", "main")
