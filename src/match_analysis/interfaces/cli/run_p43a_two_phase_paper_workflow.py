"""Run the P43A offline historical two-phase paper-workflow rehearsal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ...application.use_cases.p43a_postgame_settle import run_p43a_postgame_settle
from ...application.use_cases.p43a_pregame_freeze import (
    P43A_HUMAN_LABEL,
    P43A_REPORT_RELATIVE_PATH,
    run_p43a_pregame_freeze,
)


def _require_native_output(repository_root: Path, output_dir: Path | None) -> Path:
    expected = (repository_root / P43A_REPORT_RELATIVE_PATH).resolve()
    resolved = (output_dir or expected).resolve()
    if resolved != expected:
        raise ValueError(
            "ARTIFACT_OUTPUT_PATH_CONFLICT: P43A output must remain under "
            f"{expected}"
        )
    return resolved


def _run_pregame(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.resolve()
    try:
        output_dir = _require_native_output(repository_root, args.output_dir)
        result = run_p43a_pregame_freeze(
            repository_root,
            output_dir=output_dir,
            comparisons_path=args.prediction_source,
            persist=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    summary = result.summary
    print(
        "p43a-pregame-freeze="
        f"label={summary['workflow_label']} "
        f"human_label={P43A_HUMAN_LABEL} "
        f"phase={summary['phase']} "
        f"p37_target={summary['p37_target_count']} "
        f"edge_ready={summary['p39_edge_ready_count']} "
        f"decisions={summary['workflow_decision_count']} "
        f"bet={summary['bet_count']} "
        f"pass={summary['pass_count']} "
        f"settled={summary['settled_bet_count']} "
        f"unresolved={summary['unresolved_result_count']} "
        f"freeze_status={summary['freeze_status']} "
        f"deterministic_rerun={summary['deterministic_rerun_verified']} "
        f"network_required={summary['network_required']}"
    )
    return 0


def _run_postgame(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.resolve()
    try:
        output_dir = _require_native_output(repository_root, args.output_dir)
        result = run_p43a_postgame_settle(
            repository_root,
            output_dir=output_dir,
            result_source=args.result_source,
            persist=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    summary = result.summary
    print(
        "p43a-postgame-settle="
        f"label={summary['workflow_label']} "
        f"human_label={P43A_HUMAN_LABEL} "
        f"phase={summary['phase']} "
        f"decisions={summary['workflow_decision_count']} "
        f"bet={summary['bet_count']} "
        f"pass={summary['pass_count']} "
        f"settled={summary['settled_bet_count']} "
        f"wins={summary['win_count']} "
        f"losses={summary['loss_count']} "
        f"pushes={summary['push_count']} "
        f"units_risked={summary['units_risked']} "
        f"net={summary['net_paper_units']} "
        f"roi={summary['descriptive_historical_paper_roi']} "
        f"feedback={summary['feedback_row_count']} "
        f"p42_reconciliation={summary['p42_reconciliation']['status']} "
        f"deterministic_rerun={summary['deterministic_rerun_verified']} "
        f"network_required={summary['network_required']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pregame = subparsers.add_parser(
        "pregame-freeze",
        help="Freeze immutable pregame BET/PASS decisions and stop",
    )
    postgame = subparsers.add_parser(
        "postgame-settle",
        help="Settle an already-frozen pregame decision bundle",
    )
    for subparser in (pregame, postgame):
        subparser.add_argument(
            "--repository-root",
            type=Path,
            default=Path("."),
            help="MatchAnalysis repository root containing committed P37/P38/P39/P40 authorities",
        )
        subparser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="P43A artifact directory; must remain the repository-native report root",
        )
    pregame.add_argument(
        "--prediction-source",
        type=Path,
        default=None,
        help="Optional prediction-only comparisons file; default is committed P37 authority",
    )
    postgame.add_argument(
        "--result-source",
        type=Path,
        default=None,
        help="Optional final-result authority file; default is committed P37 outcome columns",
    )
    pregame.set_defaults(handler=_run_pregame)
    postgame.set_defaults(handler=_run_postgame)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
