"""CLI for the deterministic P37A rolling Moneyline OOS evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.rolling_moneyline_oos import (
    run_deterministic_rolling_moneyline_oos,
)
from ...application.use_cases.rolling_moneyline_oos_artifacts import (
    write_rolling_moneyline_oos_artifacts,
)
from ...application.use_cases.train_moneyline_challenger import (
    P22B_DEFAULT_FIT_RUNTIME,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-runtime", default=P22B_DEFAULT_FIT_RUNTIME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_deterministic_rolling_moneyline_oos(
            args.repository_root,
            fit_runtime=args.fit_runtime,
        )
        write_rolling_moneyline_oos_artifacts(
            args.output_dir,
            model_artifacts=result.model_artifacts,
            comparison_rows=result.comparison_rows,
            per_window_summary=result.per_window_summary,
            summary=result.summary,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    aggregate = summary["aggregate"]
    print(
        f"folds={','.join(summary['admitted_evaluation_fold_ids'])} "
        f"windows={summary['verification']['valid_window_count']} "
        f"raw_rows={aggregate['raw_row_count']} "
        f"evaluable_rows={aggregate['evaluable_row_count']} "
        f"excluded_rows={aggregate['excluded_row_count']} "
        f"champion_accuracy={aggregate['champion']['metrics']['accuracy']} "
        f"challenger_accuracy={aggregate['challenger']['metrics']['accuracy']} "
        f"champion_brier={aggregate['champion']['metrics']['brier_score']} "
        f"challenger_brier={aggregate['challenger']['metrics']['brier_score']} "
        f"aggregate_verdict={summary['comparison']['aggregate_verdict']} "
        f"conclusion={summary['comparison']['conclusion']} "
        "out_of_sample_evaluated=true model_promoted=false "
        "promotion_authorized=false production_ready=false "
        f"deterministic_rerun_verified={summary['deterministic_rerun_verified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
