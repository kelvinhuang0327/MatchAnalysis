"""CLI for the deterministic P23A paired strictly-future comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...application.use_cases.evaluate_moneyline_challenger_oos import (
    run_deterministic_moneyline_challenger_oos,
)
from ...application.use_cases.moneyline_oos_comparison_artifacts import (
    write_comparison_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_deterministic_moneyline_challenger_oos(args.repository_root)
    write_comparison_artifacts(
        args.output_dir,
        rows=result.rows,
        summary=result.summary,
        incumbent_artifact_projection=result.incumbent_artifact_projection,
        source_fold_id=result.incumbent_source_fold_id,
        source_fold_fingerprint=result.incumbent_source_fold_fingerprint,
        training_cutoff=result.incumbent_training_cutoff,
        training_row_count=result.incumbent_training_row_count,
        model_fidelity_route=result.incumbent_fidelity_route,
    )
    print(
        f"fold_id={result.summary['fold_id']} "
        f"game_count={result.summary['game_count']} "
        f"cohort_fingerprint={result.summary['cohort_fingerprint']} "
        f"challenger_mean_brier={result.summary['challenger_mean_brier']} "
        f"incumbent_mean_brier={result.summary['incumbent_mean_brier']} "
        f"brier_delta={result.summary['brier_delta']} "
        f"challenger_accuracy={result.summary['challenger_accuracy']} "
        f"incumbent_accuracy={result.summary['incumbent_accuracy']} "
        f"accuracy_delta={result.summary['accuracy_delta']} "
        "out_of_sample_evaluated=true model_promoted=false "
        "promotion_authorized=false retraining_performed=false "
        "production_ready=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
