"""Run the deterministic offline P30A Moneyline paper analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ...application.use_cases.acquire_future_moneyline_history import load_normalized_rows
from ...application.use_cases.build_moneyline_paper_source_bundle import (
    MoneylinePaperSourceBundle,
    P31A_PRECOHORT_SELECTION_RULE,
    P28AB_COHORT_END_DATE,
    P28AB_COHORT_START_DATE,
    build_moneyline_paper_source_bundle,
    resolve_date_scope,
    select_precohort_window,
)
from ...application.use_cases.moneyline_paper_analysis_artifacts import (
    write_moneyline_paper_analysis_artifacts,
)
from ...application.use_cases.run_moneyline_paper_analysis import (
    MoneylinePaperAnalysisRunResult,
    run_moneyline_paper_analysis,
)


DEFAULT_TSL_HISTORY = Path("data/authority/tsl/tsl_odds_history.jsonl")
DEFAULT_TSL_AUTHORITY_MANIFEST = Path("data/authority/tsl/source_manifest.json")
DEFAULT_P28AB_ROOT = Path("data/fixtures/p28ab_tsl_aligned_moneyline_edge")
DEFAULT_OUTPUT_DIR = Path("report/p30a_moneyline_paper_analysis")
DEFAULT_PRECOHORT_OUTPUT_DIR = Path("report/p31a_precohort_moneyline_paper_analysis")
DEFAULT_SCHEDULE = Path(
    "data/fixtures/p23f2_official_2026_history/normalized/schedule.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument(
        "--tsl-history",
        type=Path,
        default=DEFAULT_TSL_HISTORY,
    )
    parser.add_argument(
        "--tsl-authority-manifest",
        type=Path,
        default=DEFAULT_TSL_AUTHORITY_MANIFEST,
    )
    parser.add_argument("--date", help="one bounded local TSL game date (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="bounded local TSL start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="bounded local TSL end date (YYYY-MM-DD)")
    parser.add_argument(
        "--pre-cohort",
        action="store_true",
        help="run the deterministic pre-cohort window after committed P30A parity",
    )
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument(
        "--target-boxscores",
        type=Path,
        default=DEFAULT_P28AB_ROOT / "normalized/target_boxscores.jsonl",
    )
    parser.add_argument(
        "--pitcher-game-logs",
        type=Path,
        default=DEFAULT_P28AB_ROOT / "normalized/pitcher_game_logs.jsonl",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_P28AB_ROOT / "source_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="required: this vertical slice has no network or provider path",
    )
    return parser


def _result_projection(
    result: MoneylinePaperAnalysisRunResult,
) -> tuple[object, object]:
    return result.analysis, result.summary


def _assert_two_run_determinism(
    first: MoneylinePaperAnalysisRunResult,
    second: MoneylinePaperAnalysisRunResult,
    *,
    label: str,
) -> None:
    if _result_projection(first) != _result_projection(second):
        raise RuntimeError(f"STOP_MATCHANALYSIS_P31A_{label}_NONDETERMINISTIC")


def _assert_committed_p30a_parity(
    repository_root: Path,
    result: MoneylinePaperAnalysisRunResult,
) -> None:
    analysis_path = repository_root / "report/p30a_moneyline_paper_analysis/analysis.jsonl"
    summary_path = repository_root / "report/p30a_moneyline_paper_analysis/summary.json"
    committed_analysis = tuple(
        json.loads(line)
        for line in analysis_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    committed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if result.analysis != committed_analysis or result.summary != committed_summary:
        raise RuntimeError("STOP_MATCHANALYSIS_P31A_P30A_PARITY_FAILED")


def _committed_p30a_game_ids(source_manifest: dict[str, object]) -> tuple[str, ...]:
    cohort = source_manifest.get("p28ab_cohort")
    ids = (
        cohort.get("matched_final_official_game_ids")
        if isinstance(cohort, dict)
        else source_manifest.get("target_game_ids")
    )
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise RuntimeError("STOP_MATCHANALYSIS_P31A_P30A_PARITY_FAILED")
    return tuple(sorted(ids))


def _run_bundle(
    *,
    repository_root: Path,
    bundle: MoneylinePaperSourceBundle,
    allow_missing_starter_identity: bool,
    allow_insufficient_evaluable: bool,
) -> MoneylinePaperAnalysisRunResult:
    return run_moneyline_paper_analysis(
        repository_root=repository_root,
        tsl_rows=bundle.tsl_history.rows,
        tsl_raw_sha256=bundle.tsl_history.selected_rows_sha256,
        schedule_rows=bundle.schedule_rows,
        target_boxscore_rows=bundle.target_boxscore_rows,
        pitcher_game_log_rows=bundle.pitcher_game_log_rows,
        source_manifest=bundle.source_manifest,
        offline_replay_verified=True,
        cohort_start_date=bundle.scope_start_date,
        cohort_end_date=bundle.scope_end_date,
        requested_game_ids=bundle.requested_game_ids,
        allow_missing_starter_identity=allow_missing_starter_identity,
        allow_insufficient_evaluable=allow_insufficient_evaluable,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.offline:
        print("ERROR: P30A is offline-only; pass --offline", file=sys.stderr)
        return 1
    try:
        repository_root = args.repository_root.resolve()

        def rooted(path: Path) -> Path:
            return path if path.is_absolute() else repository_root / path

        if args.pre_cohort and any((args.date, args.start_date, args.end_date)):
            raise ValueError("--pre-cohort cannot be combined with an explicit date scope")

        if args.pre_cohort:
            parity_bundle = build_moneyline_paper_source_bundle(
                tsl_history_path=rooted(args.tsl_history),
                tsl_authority_manifest_path=rooted(args.tsl_authority_manifest),
                source_manifest_path=rooted(args.source_manifest),
                schedule_path=rooted(args.schedule),
                target_boxscores_path=rooted(args.target_boxscores),
                pitcher_game_logs_path=rooted(args.pitcher_game_logs),
                start_date=P28AB_COHORT_START_DATE,
                end_date=P28AB_COHORT_END_DATE,
                parity_mode=True,
            )
            parity_first = _run_bundle(
                repository_root=repository_root,
                bundle=parity_bundle,
                allow_missing_starter_identity=False,
                allow_insufficient_evaluable=False,
            )
            parity_second = _run_bundle(
                repository_root=repository_root,
                bundle=parity_bundle,
                allow_missing_starter_identity=False,
                allow_insufficient_evaluable=False,
            )
            _assert_two_run_determinism(parity_first, parity_second, label="P30A_PARITY")
            _assert_committed_p30a_parity(repository_root, parity_first)
            p30a_game_ids = _committed_p30a_game_ids(parity_bundle.source_manifest)
            schedule_rows = load_normalized_rows(rooted(args.schedule))
            window = select_precohort_window(
                tsl_history_path=rooted(args.tsl_history),
                tsl_authority_manifest_path=rooted(args.tsl_authority_manifest),
                schedule_rows=schedule_rows,
                p30a_game_ids=p30a_game_ids,
            )
            fresh_bundle = build_moneyline_paper_source_bundle(
                tsl_history_path=rooted(args.tsl_history),
                tsl_authority_manifest_path=rooted(args.tsl_authority_manifest),
                source_manifest_path=rooted(args.source_manifest),
                schedule_path=rooted(args.schedule),
                target_boxscores_path=rooted(args.target_boxscores),
                pitcher_game_logs_path=rooted(args.pitcher_game_logs),
                start_date=window.selected_start_date,
                end_date=window.selected_end_date,
                parity_mode=False,
                selection_metadata=window.metadata(),
                p30a_game_ids=p30a_game_ids,
            )
            fresh_first = _run_bundle(
                repository_root=repository_root,
                bundle=fresh_bundle,
                allow_missing_starter_identity=True,
                allow_insufficient_evaluable=True,
            )
            fresh_second = _run_bundle(
                repository_root=repository_root,
                bundle=fresh_bundle,
                allow_missing_starter_identity=True,
                allow_insufficient_evaluable=True,
            )
            _assert_two_run_determinism(fresh_first, fresh_second, label="PRECOHORT")
            result = fresh_first
            print(
                f"p30a_parity=PASS p30a_runs=2 "
                f"pre_cohort_selection=PASS rule={P31A_PRECOHORT_SELECTION_RULE} "
                f"window={window.selected_start_date}..{window.selected_end_date} "
                f"window_length={window.length} overlap=0 fresh_runs=2"
            )
        else:
            scope_start, scope_end, explicit_scope = resolve_date_scope(
                date_value=args.date,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            bundle = build_moneyline_paper_source_bundle(
                tsl_history_path=rooted(args.tsl_history),
                tsl_authority_manifest_path=rooted(args.tsl_authority_manifest),
                source_manifest_path=rooted(args.source_manifest),
                schedule_path=rooted(args.schedule),
                target_boxscores_path=rooted(args.target_boxscores),
                pitcher_game_logs_path=rooted(args.pitcher_game_logs),
                start_date=scope_start,
                end_date=scope_end,
                parity_mode=not explicit_scope,
            )
            result = _run_bundle(
                repository_root=repository_root,
                bundle=bundle,
                allow_missing_starter_identity=explicit_scope,
                allow_insufficient_evaluable=explicit_scope,
            )
        output_dir = args.output_dir or (
            DEFAULT_PRECOHORT_OUTPUT_DIR if args.pre_cohort else DEFAULT_OUTPUT_DIR
        )
        hashes = write_moneyline_paper_analysis_artifacts(
            rooted(output_dir),
            result=result,
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    print(
        f"run_id={summary['run_id']} "
        f"cohort={summary['cohort_start_date']}..{summary['cohort_end_date']} "
        f"raw={summary['raw_game_count']} "
        f"edge_available={summary['edge_available_count']} "
        f"feature_unavailable={summary['feature_unavailable_count']} "
        f"price_unavailable={summary['price_unavailable_pre_cutoff_count']}"
    )
    print(
        f"artifacts={len(hashes)} offline=True "
        f"date_scope={summary['cohort_start_date']}..{summary['cohort_end_date']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
