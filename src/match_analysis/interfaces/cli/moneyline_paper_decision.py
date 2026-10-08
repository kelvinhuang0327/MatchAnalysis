"""Run the isolated V1 Moneyline paper decision and settlement consumer."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.request import Request

from ...application.use_cases.acquire_tsl_moneyline_snapshot import (
    _default_mlb_schedule_opener,
    acquire_tsl_moneyline_snapshot,
)
from ...application.use_cases.p32a_frozen_moneyline_quotes import quotes_from_acquisition
from ...application.use_cases.p45a_paper_run_ledger import P45A_REPORT_RELATIVE_PATH
from ...application.use_cases.moneyline_paper_decision import (
    DATA_MODE_FIXTURE,
    DATA_MODE_PROSPECTIVE,
    PAPER_REPORT_RELATIVE_PATH,
    FrozenPredictionRun,
    PaperPolicyConfig,
    load_frozen_prediction_run,
    load_quote_snapshots,
    now_utc,
    run_moneyline_paper_decision,
)
from ...application.use_cases.p44a_normalized_workflow_input import (
    load_normalized_result_input,
)
from ...application.use_cases.p53a_daily_final_settlement import resolve_prediction_run
from ...application.use_cases.p50c_prediction_run_ledger import P50C_REPORT_RELATIVE_PATH
from ...infrastructure.sources.tsl_moneyline_acquisition import _default_fetcher


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _validate_output_directory(output_dir: Path, repository_root: Path) -> None:
    """Keep in-repo runtime writes inside the dedicated V1 paper leaf."""

    try:
        relative = output_dir.relative_to(repository_root)
    except ValueError:
        return
    for authority in (
        repository_root / P50C_REPORT_RELATIVE_PATH,
        repository_root / P45A_REPORT_RELATIVE_PATH,
    ):
        protected = authority.resolve()
        if output_dir == protected or protected in output_dir.parents:
            raise ValueError("paper output overlaps a read-only prediction or paper ledger")
    allowed = (repository_root / PAPER_REPORT_RELATIVE_PATH).resolve()
    if output_dir != allowed and allowed not in output_dir.parents:
        raise ValueError(
            "in-repository paper output must stay under report/moneyline_paper_decision_v1"
        )


def _event_wrapper(
    *,
    source: str,
    url: str,
    operation_calls: list[dict[str, Any]],
    call: Any,
) -> Any:
    started_at = now_utc()
    try:
        raw = call()
    except Exception as exc:
        operation_calls.append(
            {
                "source": source,
                "url": url,
                "started_at_utc": started_at,
                "completed_at_utc": now_utc(),
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "payload_sha256": None,
            }
        )
        raise
    operation_calls.append(
        {
            "source": source,
            "url": url,
            "started_at_utc": started_at,
            "completed_at_utc": now_utc(),
            "status": "SUCCESS",
            "error_type": None,
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    return raw


def _live_tsl_quotes(
    *,
    repository_root: Path,
    output_dir: Path,
    run: FrozenPredictionRun,
) -> tuple[tuple[dict[str, Any], ...], dict[int, str], dict[str, Any]]:
    """Acquire via the existing P32A adapter and join with the P28AB crosswalk."""

    started_at = now_utc()
    operation_calls: list[dict[str, Any]] = []
    capture_name = started_at.replace("-", "").replace(":", "")
    capture_root = output_dir / "acquisition" / capture_name

    def schedule_opener(request: Request, timeout: int) -> bytes:
        return _event_wrapper(
            source="MLB_STATS_API",
            url=request.full_url,
            operation_calls=operation_calls,
            call=lambda: _default_mlb_schedule_opener(request, timeout),
        )

    def tsl_fetcher(url: str) -> bytes:
        return _event_wrapper(
            source="TSL_BLOB3RD",
            url=url,
            operation_calls=operation_calls,
            call=lambda: _default_fetcher(url),
        )

    acquisition = None
    error_type: str | None = None
    error_message: str | None = None
    try:
        acquisition = acquire_tsl_moneyline_snapshot(
            repository_root=repository_root,
            runtime_root=capture_root,
            game_type=run.game_type,
            schedule_opener=schedule_opener,
            tsl_fetcher=tsl_fetcher,
        )
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)

    operation: dict[str, Any] = {
        "operation_id": "",
        "source": "MLB_STATS_API + TSL_BLOB3RD",
        "operation": "ACQUIRE_TSL_MONEYLINE_SNAPSHOT",
        "started_at_utc": started_at,
        "completed_at_utc": now_utc(),
        "status": "SUCCESS" if acquisition is not None else "FAILED",
        "adapter_calls_observed": operation_calls,
        "adapter_call_count": len(operation_calls),
        "http_attempt_count": "UNKNOWN (the existing urllib/curl fallback does not expose attempts)",
        "source_payload_sha256": [],
        "runtime_capture_paths": [],
        "error_type": error_type,
        "error": error_message,
        "game_type": run.game_type,
    }
    if acquisition is not None:
        operation["target_source_date"] = acquisition.target_date
        operation["source_payload_sha256"] = [
            {"url": url, "sha256": digest}
            for url, digest in acquisition.source_payload_sha256
        ]
        operation["runtime_capture_paths"] = list(acquisition.runtime_capture_paths)
    operation["operation_id"] = hashlib.sha256(_canonical_bytes(operation)).hexdigest()
    if acquisition is None:
        return (), {}, operation

    quotes, issues = quotes_from_acquisition(acquisition, run)
    return quotes, issues, operation

def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        prog="moneyline_paper_decision",
        description="Record odds-aware Moneyline paper decisions and isolated paper settlements.",
    )
    parser.add_argument("--prediction-run", required=True, help="P50C frozen run ID or directory")
    parser.add_argument(
        "--data-mode", required=True, choices=(DATA_MODE_FIXTURE, DATA_MODE_PROSPECTIVE)
    )
    parser.add_argument("--quote-snapshots", type=Path, help="Normalized two-way Moneyline quote JSONL")
    parser.add_argument(
        "--fetch-live-tsl",
        action="store_true",
        help="Read the configured TSL Blob3rd source through the existing bounded P32A adapter",
    )
    parser.add_argument("--final-results", type=Path, help="P44A normalized FINAL result JSONL")
    parser.add_argument("--output-dir", type=Path, default=root / PAPER_REPORT_RELATIVE_PATH)
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--runs-root", type=Path, help="Override the P50C runs directory")
    parser.add_argument("--min-net-ev", type=Decimal)
    parser.add_argument("--max-quote-age-seconds", type=int)
    parser.add_argument("--paper-cost-per-unit", type=Decimal)
    parser.add_argument("--decision-at-utc", help="Decision timestamp; defaults to the actual current UTC time")
    parser.add_argument("--settled-at-utc", help="Settlement processing time; defaults to actual current UTC time")
    parser.add_argument("--json", action="store_true", help="Print decisions and metrics as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.quote_snapshots is not None and args.fetch_live_tsl:
        parser.error("--quote-snapshots and --fetch-live-tsl are mutually exclusive")
    if args.fetch_live_tsl and args.data_mode != DATA_MODE_PROSPECTIVE:
        parser.error("--fetch-live-tsl requires --data-mode PROSPECTIVE")
    try:
        config = PaperPolicyConfig(
            min_net_ev=args.min_net_ev,
            max_quote_age_seconds=args.max_quote_age_seconds,
            paper_cost_per_unit=args.paper_cost_per_unit,
        )
        repository_root = args.repository_root.resolve()
        runs_root = args.runs_root.resolve() if args.runs_root else None
        run_path = resolve_prediction_run(
            repository_root,
            run=args.prediction_run,
            runs_root=runs_root,
        )
        run = load_frozen_prediction_run(run_path, data_mode=args.data_mode)
        output_dir = args.output_dir.resolve()
        _validate_output_directory(output_dir, repository_root)
        quote_path = args.quote_snapshots.resolve() if args.quote_snapshots else None
        source_operations: list[Mapping[str, Any]] = []
        quote_issues: dict[int, str] = {}
        quote_missing_reason = "ODDS_SOURCE_NOT_CONFIGURED"
        if args.fetch_live_tsl and config.complete:
            quote_rows, quote_issues, operation = _live_tsl_quotes(
                repository_root=repository_root,
                output_dir=output_dir,
                run=run,
            )
            source_operations.append(operation)
            quote_missing_reason = (
                "ODDS_SOURCE_UNAVAILABLE"
                if operation.get("status") == "FAILED"
                else "QUOTE_MISSING"
            )
        elif quote_path is not None:
            quote_rows = load_quote_snapshots(quote_path)
            quote_missing_reason = "QUOTE_MISSING"
        else:
            if args.fetch_live_tsl and not config.complete:
                quote_missing_reason = "ODDS_SOURCE_NOT_CONFIGURED"
            quote_rows = None
        results = (
            load_normalized_result_input(args.final_results)
            if args.final_results is not None
            else ()
        )
        state = run_moneyline_paper_decision(
            run=run,
            quote_snapshots=quote_rows,
            output_dir=output_dir,
            config=config,
            decision_at_utc=args.decision_at_utc or now_utc(),
            settled_at_utc=args.settled_at_utc or now_utc(),
            data_mode=args.data_mode,
            final_results=results,
            quote_missing_reason=quote_missing_reason,
            quote_issues_by_game_pk=quote_issues,
            source_operations=source_operations,
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "output": str(output_dir / "paper_state.json"),
                    "decision_count": len(state["decisions"]),
                    "position_count": len(state["positions"]),
                    "settlement_count": len(state["settlements"]),
                    "metrics": state["metrics"],
                    "latest_decisions": state["decisions"][-10:],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        metrics = state["metrics"]
        print(
            "moneyline-paper-decision-v1 "
            f"mode={metrics['data_mode']} "
            f"games={metrics['unique_candidate_games']['value']} "
            f"bet={metrics['decisions']['bet']} no_bet={metrics['decisions']['no_bet']} "
            f"positions={len(state['positions'])} settlements={len(state['settlements'])} "
            f"output={output_dir / 'paper_state.json'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
