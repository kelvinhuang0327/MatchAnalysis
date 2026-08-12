"""CLI for the deterministic P36A offline Moneyline retraining baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.offline_moneyline_retraining_artifacts import (
    write_offline_moneyline_retraining_artifacts,
)
from ...application.use_cases.offline_moneyline_retraining_baseline import (
    run_deterministic_offline_moneyline_retraining_baseline,
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
        result = run_deterministic_offline_moneyline_retraining_baseline(
            args.repository_root,
            fit_runtime=args.fit_runtime,
        )
        write_offline_moneyline_retraining_artifacts(
            args.output_dir,
            artifact=result.artifact,
            comparisons=result.comparison_rows,
            summary=result.summary,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    print(
        f"training_rows={summary['training']['eligible_row_count']} "
        f"holdout_evaluable_rows={summary['holdout']['evaluable_row_count']} "
        f"holdout_raw_rows={summary['holdout']['raw_row_count']} "
        f"holdout_excluded_rows={summary['holdout']['excluded_row_count']} "
        f"champion_model_id={summary['champion']['model_id']} "
        f"challenger_model_id={summary['challenger']['model_id']} "
        f"champion_brier={summary['champion']['metrics']['brier_score']} "
        f"challenger_brier={summary['challenger']['metrics']['brier_score']} "
        f"champion_log_loss={summary['champion']['metrics']['log_loss']} "
        f"challenger_log_loss={summary['challenger']['metrics']['log_loss']} "
        f"verdict={summary['comparison']['verdict']} "
        "model_promoted=false production_ready=false "
        f"deterministic_rerun_verified={summary['deterministic_rerun_verified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
