"""Acquire or replay the P24C promoted-default paper Moneyline batch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.generate_paper_moneyline_batch import (
    acquire_p24c_source_inputs,
    generate_paper_moneyline_batch,
    load_p24c_source_inputs,
    resolve_p24c_window,
)
from ...application.use_cases.paper_moneyline_batch_artifacts import (
    write_paper_moneyline_batch_artifacts,
)


DEFAULT_ACQUIRED_AT_UTC = "2026-08-10T00:00:00Z"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acquired-at", default=DEFAULT_ACQUIRED_AT_UTC)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--explicit-model-artifact", type=Path)
    parser.add_argument("--incumbent-model-artifact", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        window = resolve_p24c_window(args.repository_root)
        if args.offline:
            source_manifest = args.source_manifest or (args.output_dir / "source_manifest.json")
            schedule_rows, boxscore_rows, pitcher_rows, source_manifest_projection = (
                load_p24c_source_inputs(
                    repository_root=args.repository_root,
                    raw_root=args.raw_root,
                    normalized_root=args.normalized_root,
                    source_manifest_path=source_manifest,
                    window=window,
                )
            )
        else:
            acquired = acquire_p24c_source_inputs(
                repository_root=args.repository_root,
                raw_root=args.raw_root,
                normalized_root=args.normalized_root,
                window=window,
                acquired_at_utc=args.acquired_at,
            )
            schedule_rows = acquired["schedule_rows"]
            boxscore_rows = acquired["target_boxscore_rows"]
            pitcher_rows = acquired["pitcher_game_log_rows"]
            source_manifest_projection = acquired["source_manifest"]

        result = generate_paper_moneyline_batch(
            repository_root=args.repository_root,
            schedule_rows=schedule_rows,
            target_boxscore_rows=boxscore_rows,
            pitcher_game_log_rows=pitcher_rows,
            source_manifest=source_manifest_projection,
            offline_replay_verified=args.offline,
            explicit_model_artifact_path=args.explicit_model_artifact,
            incumbent_model_artifact_path=args.incumbent_model_artifact,
        )
        hashes = write_paper_moneyline_batch_artifacts(
            args.output_dir,
            predictions=result.predictions,
            feature_unavailable=result.feature_unavailable,
            source_manifest=result.source_manifest,
            summary=result.summary,
        )
    except (OSError, TypeError, ValueError, RuntimeError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"batch_id={result.summary['batch_id']} "
        f"window={result.summary['window_start_date']}..{result.summary['window_end_date']} "
        f"raw={result.summary['raw_game_count']} "
        f"evaluable={result.summary['evaluable_game_count']} "
        f"unavailable={result.summary['feature_unavailable_count']}"
    )
    print(f"artifacts={len(hashes)} offline={args.offline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
