"""Replay the offline P28AB TSL-aligned Moneyline edge batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ...application.use_cases.acquire_future_moneyline_history import load_normalized_rows
from ...application.use_cases.generate_tsl_moneyline_edge_batch import (
    generate_tsl_moneyline_edge_batch,
    write_tsl_moneyline_edge_batch_artifacts,
)
from ...infrastructure.legacy_betting_pool.tsl_odds_history import (
    load_tsl_odds_history,
)


DEFAULT_FIXTURE_ROOT = Path("data/fixtures/p28ab_tsl_aligned_moneyline_edge")
DEFAULT_OUTPUT_DIR = Path("report/p28ab_tsl_aligned_moneyline_edge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument(
        "--tsl-history",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT / "tsl_odds_history.jsonl",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("data/fixtures/p23f2_official_2026_history/normalized/schedule.jsonl"),
    )
    parser.add_argument(
        "--target-boxscores",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT / "normalized/target_boxscores.jsonl",
    )
    parser.add_argument(
        "--pitcher-game-logs",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT / "normalized/pitcher_game_logs.jsonl",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT / "source_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="required: this vertical slice has no network or provider path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.offline:
        print("ERROR: P28AB is offline-only; pass --offline", file=sys.stderr)
        return 1
    try:
        repository_root = args.repository_root.resolve()

        def rooted(path: Path) -> Path:
            return path if path.is_absolute() else repository_root / path

        tsl_snapshot = load_tsl_odds_history(rooted(args.tsl_history))
        source_manifest = json.loads(rooted(args.source_manifest).read_text(encoding="utf-8"))
        if not isinstance(source_manifest, dict):
            raise ValueError("source manifest must be an object")
        result = generate_tsl_moneyline_edge_batch(
            repository_root=args.repository_root,
            tsl_rows=tsl_snapshot.rows,
            tsl_raw_sha256=tsl_snapshot.raw_sha256,
            schedule_rows=load_normalized_rows(rooted(args.schedule)),
            target_boxscore_rows=load_normalized_rows(rooted(args.target_boxscores)),
            pitcher_game_log_rows=load_normalized_rows(rooted(args.pitcher_game_logs)),
            source_manifest=source_manifest,
            offline_replay_verified=True,
        )
        hashes = write_tsl_moneyline_edge_batch_artifacts(
            args.output_dir,
            result=result,
        )
    except (OSError, TypeError, ValueError, RuntimeError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    print(
        f"batch_id={summary['batch_id']} "
        f"cohort={summary['cohort_start_date']}..{summary['cohort_end_date']} "
        f"raw={summary['raw_source_row_count']} "
        f"games={summary['raw_game_count']} "
        f"evaluable={summary['evaluable_game_count']} "
        f"prices={summary['selected_price_count']} "
        f"edges={summary['edge_row_count']}"
    )
    print(f"artifacts={len(hashes)} offline=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
