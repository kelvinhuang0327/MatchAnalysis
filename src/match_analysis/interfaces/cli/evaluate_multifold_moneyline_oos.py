"""CLI for P23B acquisition and deterministic multi-fold OOS evaluation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from ...application.use_cases.evaluate_multifold_moneyline_oos import (
    acquire_p23b_future_folds,
    run_deterministic_multifold_moneyline_oos,
)
from ...application.use_cases.multifold_moneyline_oos_artifacts import (
    write_multifold_oos_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="Acquire and freeze the two new official MLB fold authorities first.",
    )
    parser.add_argument("--acquired-at", default="2026-08-10T00:00:00Z")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.acquire:
        acquire_p23b_future_folds(
            args.repository_root,
            acquired_at_utc=datetime.fromisoformat(
                args.acquired_at.replace("Z", "+00:00")
            ).astimezone(UTC),
        )
    result = run_deterministic_multifold_moneyline_oos(args.repository_root)
    write_multifold_oos_artifacts(
        args.output_dir,
        comparison_rows=result.comparison_rows,
        per_fold_summary=result.per_fold_summary,
        summary=result.summary,
    )
    print(
        f"folds={','.join(result.summary['fold_ids'])} "
        f"total_game_count={result.summary['total_game_count']} "
        f"total_raw_game_count={result.summary['total_raw_game_count']} "
        f"total_evaluable_game_count={result.summary['total_evaluable_game_count']} "
        f"total_feature_unavailable_count={result.summary['total_feature_unavailable_count']} "
        f"pooled_evaluation_coverage={result.summary['pooled_evaluation_coverage']} "
        f"challenger_mean_brier={result.summary['challenger_mean_brier']} "
        f"incumbent_mean_brier={result.summary['incumbent_mean_brier']} "
        f"brier_delta={result.summary['brier_delta']} "
        f"challenger_accuracy={result.summary['challenger_accuracy']} "
        f"incumbent_accuracy={result.summary['incumbent_accuracy']} "
        "out_of_sample_evaluated=true multi_fold_evaluated=true "
        "model_promoted=false promotion_authorized=false "
        "production_ready=false challenger_retrained=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
