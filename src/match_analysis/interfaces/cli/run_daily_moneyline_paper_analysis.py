"""Run or replay one frozen daily Moneyline paper analysis bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.run_daily_moneyline_paper_analysis import (
    P33A_RUNTIME_ROOT,
    replay_daily_moneyline_paper_analysis,
    run_daily_moneyline_paper_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=P33A_RUNTIME_ROOT,
        help="authorized runtime root for live and replay artifacts",
    )
    parser.add_argument(
        "--date",
        dest="date_value",
        help="explicit MLB target date (YYYY-MM-DD); omitted selects the earliest future slate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="exact authorized bundle/output directory under --runtime-root",
    )
    parser.add_argument(
        "--from-bundle",
        type=Path,
        help="replay an existing frozen bundle without acquisition or network",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.from_bundle is not None and args.date_value is not None:
        print("ERROR: --date cannot be combined with --from-bundle", file=sys.stderr)
        return 1
    try:
        repository_root = args.repository_root.resolve()
        if args.from_bundle is not None:
            result = replay_daily_moneyline_paper_analysis(
                repository_root=repository_root,
                bundle_path=args.from_bundle,
                output_dir=args.output_dir,
                runtime_root=args.runtime_root,
            )
            print(
                f"run_fingerprint={result.run_manifest['run_fingerprint']} "
                f"raw={result.summary['raw_game_count']} "
                f"offline_replay=PASS network_guard=PASS "
                f"output={result.bundle_root}"
            )
            return 0

        result = run_daily_moneyline_paper_analysis(
            repository_root=repository_root,
            runtime_root=args.runtime_root,
            date_value=args.date_value,
            output_dir=args.output_dir,
        )
        print(
            f"run_fingerprint={result.run_manifest['run_fingerprint']} "
            f"target_date={result.run_manifest['target_date']} "
            f"raw={result.summary['raw_game_count']} "
            f"edge_available={result.summary['edge_available_count']} "
            f"feature_unavailable={result.summary['feature_unavailable_count']} "
            f"price_unavailable={result.summary['price_unavailable_pre_cutoff_count']}"
        )
        print(
            f"bundle={result.bundle_root} frozen=True "
            f"offline_replay=NOT_RUN network_guard=NOT_RUN"
        )
        return 0
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
