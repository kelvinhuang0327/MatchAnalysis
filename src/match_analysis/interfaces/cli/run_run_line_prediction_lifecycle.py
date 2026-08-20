"""CLI for P61B / P64B Run Line prospective prediction lifecycle and canonical P50C forward integration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from ...application.use_cases.p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P50C_REPORT_RELATIVE_PATH,
    read_json_object,
    read_jsonl_objects,
)
from ...application.use_cases.run_line_prediction_lifecycle import (
    DEFAULT_CANONICAL_SPREAD_HOME,
    RunLineFrozenPrediction,
    RunLinePregameInputItem,
    _parse_run_line_frozen_dict,
    freeze_run_line_predictions,
    settle_run_line_predictions,
)
from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...baseball.domain.run_line_model import PriorGameRecord


def _load_history(path: Path) -> list[PriorGameRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for item in data:
        records.append(
            PriorGameRecord(
                home_team=str(item["home_team"]),
                away_team=str(item["away_team"]),
                home_score=int(item["home_score"]),
                away_score=int(item["away_score"]),
                completed_at_utc=parse_canonical_utc(str(item["completed_at_utc"])),
            )
        )
    return records


def _load_pregame_inputs(path: Path) -> list[RunLinePregameInputItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_items = data.get("predictions", data) if isinstance(data, dict) else data
    items = []
    for item in raw_items:
        spread = item.get("spread_home", item.get("line_value", DEFAULT_CANONICAL_SPREAD_HOME))
        items.append(
            RunLinePregameInputItem(
                provider_game_id=str(item.get("provider_game_id") or item.get("game_pk", "")),
                home_team=str(item["home_team"]),
                away_team=str(item["away_team"]),
                scheduled_start_utc=str(item["scheduled_start_utc"]),
                as_of_utc=str(item.get("as_of_utc") or item.get("scheduled_start_utc")),
                spread_home=float(spread),
                provider_namespace=str(item.get("provider_namespace", "MLB_STATS_API")),
                game_number=int(item.get("game_number", 1)),
                game_pk=int(item["game_pk"]) if item.get("game_pk") is not None else None,
                season=int(item.get("season", 2026)),
            )
        )
    return items


def _load_final_results(path: Path) -> dict[str, tuple[int, int]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text.startswith("{") and not text.startswith("{\n"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and not any(isinstance(v, dict) for v in data.values()):
                return {str(k): (int(v[0]), int(v[1])) for k, v in data.items()}
        except Exception:
            pass

    # Try JSON or JSONL
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {str(k): (int(v[0]), int(v[1])) for k, v in data.items()}
        elif isinstance(data, list):
            return {
                str(item.get("provider_game_id") or item.get("game_pk")): (
                    int(item["home_score"]),
                    int(item["away_score"]),
                )
                for item in data
            }
    except Exception:
        pass

    results = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("provider_game_id") or row.get("game_pk") or row.get("prediction_row_id", ""))
        results[key] = (int(row["home_score"]), int(row["away_score"]))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P61B / P64B Run Line Prediction Lifecycle CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Freeze subcommand
    freeze_parser = subparsers.add_parser(
        "freeze", help="Generate and freeze Run Line predictions"
    )
    freeze_parser.add_argument("--input", required=True, type=Path)
    freeze_parser.add_argument("--history", required=True, type=Path)
    freeze_parser.add_argument("--run-id", required=True)
    freeze_parser.add_argument("--output-dir", type=Path)
    freeze_parser.add_argument("--runs-root", type=Path)
    freeze_parser.add_argument("--repo-root", type=Path)
    freeze_parser.add_argument(
        "--classification",
        default=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    )
    freeze_parser.add_argument("--created-at-utc")

    # Settle subcommand
    settle_parser = subparsers.add_parser(
        "settle", help="Settle frozen Run Line predictions against final results"
    )
    settle_parser.add_argument("--run-dir", type=Path)
    settle_parser.add_argument("--run-id")
    settle_parser.add_argument("--results", required=True, type=Path)
    settle_parser.add_argument("--output-dir", type=Path)
    settle_parser.add_argument("--runs-root", type=Path)
    settle_parser.add_argument("--ledger-root", type=Path)
    settle_parser.add_argument("--repo-root", type=Path)
    settle_parser.add_argument("--settled-at-utc")

    args = parser.parse_args(argv)

    if args.command == "freeze":
        if not args.input.is_file():
            print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
            return 1
        if not args.history.is_file():
            print(f"ERROR: History file not found: {args.history}", file=sys.stderr)
            return 1

        repo_root = args.repo_root or Path.cwd()
        output_dir = args.output_dir
        if output_dir is None and args.runs_root is not None:
            output_dir = args.runs_root / args.run_id
        elif output_dir is None:
            output_dir = repo_root / P50C_REPORT_RELATIVE_PATH / "runs" / args.run_id

        pregame_inputs = _load_pregame_inputs(args.input)
        history_records = _load_history(args.history)
        predictions, manifest = freeze_run_line_predictions(
            pregame_inputs,
            history_records,
            run_id=args.run_id,
            classification=args.classification,
            output_dir=output_dir,
            created_at_utc=args.created_at_utc,
        )
        print(
            f"Froze {len(predictions)} Run Line predictions for run {args.run_id} into {output_dir}"
        )
        return 0

    elif args.command == "settle":
        repo_root = args.repo_root or Path.cwd()
        run_dir = args.run_dir
        if run_dir is None and args.runs_root is not None and args.run_id is not None:
            run_dir = args.runs_root / args.run_id
        elif run_dir is None and args.run_id is not None:
            run_dir = repo_root / P50C_REPORT_RELATIVE_PATH / "runs" / args.run_id

        if run_dir is None or not run_dir.is_dir():
            print(f"ERROR: Run dir not found: {run_dir}", file=sys.stderr)
            return 1
        if not args.results.is_file():
            print(f"ERROR: Results file not found: {args.results}", file=sys.stderr)
            return 1

        ledger_root = args.ledger_root
        if ledger_root is None and args.runs_root is not None:
            ledger_root = args.runs_root.parent / "ledger"

        final_results = _load_final_results(args.results)
        out_dir = args.output_dir or run_dir

        settled_rows, summary = settle_run_line_predictions(
            final_results=final_results,
            run_dir=run_dir,
            output_dir=out_dir,
            ledger_root=ledger_root,
            settled_at_utc=args.settled_at_utc,
        )
        print(
            f"Settled {len(settled_rows)} predictions (pushes: {summary['push_count']}, "
            f"decided: {summary['decided_count']}, accuracy: {summary['accuracy']}) into {out_dir}"
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
