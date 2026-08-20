"""CLI for P61B Run Line prospective prediction lifecycle and settlement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from ...application.use_cases.run_line_prediction_lifecycle import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    RunLineFrozenPrediction,
    RunLinePregameInputItem,
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
    items = []
    for item in data:
        items.append(
            RunLinePregameInputItem(
                provider_game_id=str(item["provider_game_id"]),
                home_team=str(item["home_team"]),
                away_team=str(item["away_team"]),
                scheduled_start_utc=str(item["scheduled_start_utc"]),
                as_of_utc=str(item["as_of_utc"]),
                spread_home=float(item["spread_home"]),
                provider_namespace=str(item.get("provider_namespace", "MLB_STATS_API")),
                game_number=int(item.get("game_number", 1)),
                game_pk=item.get("game_pk"),
                season=int(item.get("season", 2026)),
            )
        )
    return items


def _load_final_results(path: Path) -> dict[str, tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = {}
    if isinstance(data, dict):
        for k, v in data.items():
            results[str(k)] = (int(v[0]), int(v[1]))
    elif isinstance(data, list):
        for item in data:
            results[str(item["provider_game_id"])] = (
                int(item["home_score"]),
                int(item["away_score"]),
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P61B Run Line Prediction Lifecycle CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Freeze subcommand
    freeze_parser = subparsers.add_parser(
        "freeze", help="Generate and freeze Run Line predictions"
    )
    freeze_parser.add_argument("--input", required=True, type=Path)
    freeze_parser.add_argument("--history", required=True, type=Path)
    freeze_parser.add_argument("--run-id", required=True)
    freeze_parser.add_argument("--output-dir", required=True, type=Path)
    freeze_parser.add_argument(
        "--classification",
        default=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    )

    # Settle subcommand
    settle_parser = subparsers.add_parser(
        "settle", help="Settle frozen Run Line predictions against final results"
    )
    settle_parser.add_argument("--run-dir", required=True, type=Path)
    settle_parser.add_argument("--results", required=True, type=Path)
    settle_parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args(argv)

    if args.command == "freeze":
        if not args.input.is_file():
            print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
            return 1
        if not args.history.is_file():
            print(f"ERROR: History file not found: {args.history}", file=sys.stderr)
            return 1

        pregame_inputs = _load_pregame_inputs(args.input)
        history_records = _load_history(args.history)
        predictions, manifest = freeze_run_line_predictions(
            pregame_inputs,
            history_records,
            run_id=args.run_id,
            classification=args.classification,
            output_dir=args.output_dir,
        )
        print(
            f"Froze {len(predictions)} Run Line predictions for run {args.run_id} into {args.output_dir}"
        )
        return 0

    elif args.command == "settle":
        if not args.run_dir.is_dir():
            print(f"ERROR: Run dir not found: {args.run_dir}", file=sys.stderr)
            return 1
        if not args.results.is_file():
            print(f"ERROR: Results file not found: {args.results}", file=sys.stderr)
            return 1

        pred_file = args.run_dir / "predictions.jsonl"
        manifest_file = args.run_dir / "run_manifest.json"
        if not pred_file.is_file() or not manifest_file.is_file():
            print(
                f"ERROR: Valid frozen run not found in {args.run_dir}",
                file=sys.stderr,
            )
            return 1

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        predictions: list[RunLineFrozenPrediction] = []
        for line in pred_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            predictions.append(
                RunLineFrozenPrediction(
                    workflow_prediction_id=row["workflow_prediction_id"],
                    prediction_fingerprint=row["prediction_fingerprint"],
                    provider_namespace=row["provider_namespace"],
                    provider_game_id=row["provider_game_id"],
                    game_number=row["game_number"],
                    game_pk=row.get("game_pk"),
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    scheduled_start_utc=row["scheduled_start_utc"],
                    as_of_utc=row["as_of_utc"],
                    market_id=row["market_id"],
                    spread_home=float(row["spread_home"]),
                    model_id=row["model_id"],
                    model_version=row["model_version"],
                    lambda_home=float(row["lambda_home"]),
                    lambda_away=float(row["lambda_away"]),
                    p_home_covers=float(row["p_home_covers"]),
                    p_away_covers=float(row["p_away_covers"]),
                    p_push=float(row["p_push"]),
                    predicted_side=row["predicted_side"],
                    selected_cover_probability=float(row["selected_cover_probability"]),
                )
            )

        final_results = _load_final_results(args.results)
        out_dir = args.output_dir or args.run_dir
        settled_rows, summary = settle_run_line_predictions(
            predictions,
            final_results,
            run_id=manifest["run_id"],
            output_dir=out_dir,
        )
        print(
            f"Settled {len(settled_rows)} predictions (pushes: {summary['push_count']}, "
            f"decided: {summary['decided_count']}, accuracy: {summary['accuracy']}) into {out_dir}"
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
