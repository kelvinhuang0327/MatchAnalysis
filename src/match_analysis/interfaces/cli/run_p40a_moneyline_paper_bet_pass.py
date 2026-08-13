"""Run the deterministic P40A Moneyline paper BET/PASS baseline offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40A_REPORT_RELATIVE_PATH,
    run_p40a_moneyline_paper_bet_pass,
    write_p40a_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
        help="MatchAnalysis repository root containing the committed P37/P38/P39 authorities",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="P40A artifact directory; must remain the repository-native report root",
    )
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    expected_output_dir = (repository_root / P40A_REPORT_RELATIVE_PATH).resolve()
    output_dir = (args.output_dir or expected_output_dir).resolve()
    if output_dir != expected_output_dir:
        print(
            "ERROR: ARTIFACT_OUTPUT_PATH_CONFLICT: P40A output must remain under "
            f"{expected_output_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        result = run_p40a_moneyline_paper_bet_pass(repository_root)
        artifact_hashes = write_p40a_artifacts(output_dir, result)
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    champion = result.summary["models"]["champion_primary"]
    shadow = result.summary["models"]["raw_challenger_shadow"]
    print(
        "p40a="
        f"edge_ready={result.summary['edge_ready_rows']} "
        f"champion_bet={champion['bet_count']} "
        f"champion_pass={champion['pass_count']} "
        f"champion_net={champion['net_paper_units']} "
        f"champion_roi={champion['descriptive_paper_roi']} "
        f"shadow_bet={shadow['bet_count']} "
        f"shadow_pass={shadow['pass_count']} "
        f"shadow_net={shadow['net_paper_units']} "
        f"shadow_roi={shadow['descriptive_paper_roi']} "
        f"primary_conclusion={result.summary['primary_conclusion']} "
        f"shadow_comparison={result.summary['shadow_comparison']} "
        f"deterministic_rerun={result.summary['deterministic_rerun_verified']} "
        f"artifacts={len(artifact_hashes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
