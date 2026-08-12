"""CLI for the deterministic P38A Moneyline probability calibration evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p38a_probability_calibration import (
    run_deterministic_p38a_probability_calibration,
)
from ...application.use_cases.p38a_probability_calibration_artifacts import (
    write_p38a_probability_calibration_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_deterministic_p38a_probability_calibration(args.repository_root)
        write_p38a_probability_calibration_artifacts(
            args.output_dir,
            calibration_artifacts=result["calibration_artifacts"],
            comparison_rows=result["comparison_rows"],
            per_window_summary=result["per_window_summary"],
            summary=result["summary"],
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result["summary"]
    aggregate = summary["aggregate"]
    print(
        f"folds={','.join(summary['admitted_target_holdout_fold_ids'])} "
        f"windows={summary['verification']['valid_window_count']} "
        f"raw_rows={aggregate['raw_row_count']} "
        f"evaluable_rows={aggregate['evaluable_row_count']} "
        f"excluded_rows={aggregate['excluded_row_count']} "
        f"raw_brier={aggregate['raw_challenger']['metrics']['brier_score']} "
        f"calibrated_brier={aggregate['calibrated_challenger']['metrics']['brier_score']} "
        f"raw_log_loss={aggregate['raw_challenger']['metrics']['log_loss']} "
        f"calibrated_log_loss={aggregate['calibrated_challenger']['metrics']['log_loss']} "
        f"conclusion={summary['comparison']['conclusion']} "
        "out_of_sample_evaluated=true model_promoted=false "
        f"deterministic_rerun_verified={summary['deterministic_rerun_verified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
