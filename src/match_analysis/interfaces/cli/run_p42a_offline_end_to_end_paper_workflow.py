"""Run the deterministic P42A offline historical paper-workflow rehearsal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p42a_offline_end_to_end_paper_workflow import (
    P42A_REPORT_RELATIVE_PATH,
    run_p42a_offline_end_to_end_paper_workflow,
    write_p42a_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
        help="MatchAnalysis repository root containing committed P37/P38/P39/P40 authorities",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="P42A artifact directory; must remain the repository-native report root",
    )
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    expected_output_dir = (repository_root / P42A_REPORT_RELATIVE_PATH).resolve()
    output_dir = (args.output_dir or expected_output_dir).resolve()
    if output_dir != expected_output_dir:
        print(
            "ERROR: ARTIFACT_OUTPUT_PATH_CONFLICT: P42A output must remain under "
            f"{expected_output_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        result = run_p42a_offline_end_to_end_paper_workflow(repository_root)
        artifact_hashes = write_p42a_artifacts(output_dir, result)
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    print(
        "p42a="
        f"label={summary['workflow_label']} "
        f"p37_target={summary['p37_target_count']} "
        f"edge_ready={summary['p39_edge_ready_count']} "
        f"decisions={summary['workflow_decision_count']} "
        f"bet={summary['bet_count']} "
        f"pass={summary['pass_count']} "
        f"wins={summary['win_count']} "
        f"losses={summary['loss_count']} "
        f"units_risked={summary['units_risked']} "
        f"net={summary['net_paper_units']} "
        f"roi={summary['descriptive_historical_paper_roi']} "
        f"no_market={summary['p39_no_market_count']} "
        f"p40_reconciliation={summary['p40_reconciliation']['status']} "
        f"deterministic_rerun={summary['deterministic_rerun_verified']} "
        f"artifacts={len(artifact_hashes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
