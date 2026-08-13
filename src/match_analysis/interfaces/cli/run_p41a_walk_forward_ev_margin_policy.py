"""Run the deterministic P41A walk-forward EV-margin policy evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p41a_walk_forward_ev_margin_policy import (
    P41A_REPORT_RELATIVE_PATH,
    run_p41a_walk_forward_ev_margin_policy,
    write_p41a_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
        help="MatchAnalysis repository root containing the committed P40A authority",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="P41A artifact directory; must remain the repository-native report root",
    )
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    expected_output_dir = (repository_root / P41A_REPORT_RELATIVE_PATH).resolve()
    output_dir = (args.output_dir or expected_output_dir).resolve()
    if output_dir != expected_output_dir:
        print(
            "ERROR: ARTIFACT_OUTPUT_PATH_CONFLICT: P41A output must remain under "
            f"{expected_output_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        result = run_p41a_walk_forward_ev_margin_policy(repository_root)
        artifact_hashes = write_p41a_artifacts(output_dir, result)
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    selected = result.summary["selected_policy"]
    baseline = result.summary["zero_ev_baseline"]
    print(
        "p41a="
        f"target_windows={result.summary['total_target_windows']} "
        f"target_rows={result.summary['policy_oos_target_rows']} "
        f"selected_bet={selected['bet_count']} "
        f"selected_pass={selected['pass_count']} "
        f"selected_net={selected['net_paper_units']} "
        f"zero_ev_bet={baseline['bet_count']} "
        f"zero_ev_pass={baseline['pass_count']} "
        f"zero_ev_net={baseline['net_paper_units']} "
        f"conclusion={result.summary['conclusion']} "
        f"deterministic_rerun={result.summary['deterministic_rerun_verified']} "
        f"artifacts={len(artifact_hashes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
