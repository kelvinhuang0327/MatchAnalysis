"""Build the P23F2 fold from normalized official MLB inputs only."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Mapping

from ...baseball.domain.future_evaluation_fold import (
    FOLD_ID,
    MIN_HISTORY_MONTHS,
    RECENT_GAME_WINDOW,
    TRAINING_INFORMATION_BOUNDARY_UTC,
    FutureEvaluationFold,
    FutureFeatureRow,
    FutureResultRow,
    fingerprint_manifest,
    fingerprint_rows,
)
from .acquire_future_moneyline_history import (
    VALIDATION_END,
    VALIDATION_START,
    load_normalized_rows,
)


getcontext().prec = 28


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _decimal(value: Decimal) -> str:
    return format(value, "f")


def _team_recent_delta(
    *,
    schedule_rows: tuple[Mapping[str, Any], ...],
    home_team_id: int,
    away_team_id: int,
    target_start: datetime,
) -> Decimal:
    prior = [
        row for row in schedule_rows
        if row["final"] and _parse(row["scheduled_start_utc"]) < target_start
    ]
    history: dict[int, list[tuple[datetime, int]]] = defaultdict(list)
    for row in prior:
        home_score = row["home_score"]
        away_score = row["away_score"]
        if not isinstance(home_score, int) or not isinstance(away_score, int) or home_score == away_score:
            continue
        start = _parse(row["scheduled_start_utc"])
        history[row["home_team"]["id"]].append((start, int(home_score > away_score)))
        history[row["away_team"]["id"]].append((start, int(away_score > home_score)))
    def rate(team_id: int) -> Decimal:
        rows = sorted(history[team_id], key=lambda item: item[0])[-RECENT_GAME_WINDOW:]
        if len(rows) < RECENT_GAME_WINDOW:
            raise RuntimeError("STOP_MATCHANALYSIS_P23F2_OFFICIAL_SOURCE_INSUFFICIENT")
        return Decimal(sum(result for _, result in rows)) / Decimal(len(rows))
    return rate(home_team_id) - rate(away_team_id)


def _starter_era(
    logs_by_player: Mapping[int, tuple[Mapping[str, Any], ...]],
    player_id: int,
    target_date: str,
) -> Decimal:
    prior = [
        row for row in logs_by_player.get(player_id, ())
        if row["game_type"] == "R" and row["date"] < target_date and row["games_started"] == 1
    ]
    outs = sum(int(row["outs"]) for row in prior)
    earned_runs = sum(int(row["earned_runs"]) for row in prior)
    if outs <= 0:
        raise RuntimeError("STOP_MATCHANALYSIS_P23F2_STARTER_ERA_PIT_UNRESOLVED")
    return Decimal(earned_runs * 27) / Decimal(outs)


def materialize_future_moneyline_fold(
    *,
    schedule_rows: tuple[Mapping[str, Any], ...],
    target_boxscore_rows: tuple[Mapping[str, Any], ...],
    pitcher_game_log_rows: tuple[Mapping[str, Any], ...],
    source_manifest_fingerprint: str,
) -> FutureEvaluationFold:
    """Materialize one June cohort, freezing features before results are joined."""

    boundary = _parse(TRAINING_INFORMATION_BOUNDARY_UTC)
    target_games = tuple(
        row for row in schedule_rows
        if VALIDATION_START <= row["official_date"] <= VALIDATION_END
    )
    if len(target_games) < 2:
        raise RuntimeError("STOP_MATCHANALYSIS_P23F2_OFFICIAL_SOURCE_INSUFFICIENT")
    if any(_parse(row["scheduled_start_utc"]) <= boundary for row in target_games):
        raise RuntimeError("STOP_MATCHANALYSIS_P23F2_BASELINE_DRIFT")
    if any(not row["final"] for row in target_games):
        raise RuntimeError("STOP_MATCHANALYSIS_P23F2_OFFICIAL_SOURCE_INSUFFICIENT")
    boxes = {row["provider_game_id"]: row for row in target_boxscore_rows}
    logs: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pitcher_game_log_rows:
        logs[int(row["player_id"])].append(row)
    logs_tuple = {player_id: tuple(rows) for player_id, rows in logs.items()}

    feature_rows: list[FutureFeatureRow] = []
    for game in sorted(target_games, key=lambda row: (row["scheduled_start_utc"], row["game_number"], row["game_pk"])):
        box = boxes.get(game["provider_game_id"])
        if box is None:
            raise RuntimeError("STOP_MATCHANALYSIS_P23F2_STARTER_IDENTITY_UNRESOLVED")
        target_start = _parse(game["scheduled_start_utc"])
        home_starter = box["home_starter"]
        away_starter = box["away_starter"]
        recent_delta = _team_recent_delta(
            schedule_rows=schedule_rows,
            home_team_id=game["home_team"]["id"],
            away_team_id=game["away_team"]["id"],
            target_start=target_start,
        )
        starter_delta = _starter_era(logs_tuple, int(home_starter["player_id"]), game["official_date"]) - _starter_era(
            logs_tuple, int(away_starter["player_id"]), game["official_date"]
        )
        feature_rows.append(
            FutureFeatureRow(
                provider_game_id=game["provider_game_id"],
                game_pk=int(game["game_pk"]),
                game_number=int(game["game_number"]),
                official_date=game["official_date"],
                scheduled_start_utc=game["scheduled_start_utc"],
                feature_as_of_utc=(target_start - timedelta(microseconds=1)).isoformat().replace("+00:00", "Z"),
                home_team=game["home_team"]["name"],
                away_team=game["away_team"]["name"],
                home_starter_id=int(home_starter["player_id"]),
                home_starter_name=home_starter["full_name"],
                away_starter_id=int(away_starter["player_id"]),
                away_starter_name=away_starter["full_name"],
                recent_win_rate_delta=_decimal(recent_delta),
                starter_era_delta=_decimal(starter_delta),
            ).with_fingerprint()
        )
    frozen_features = tuple(feature_rows)
    results = tuple(
        FutureResultRow(
            provider_game_id=game["provider_game_id"],
            game_pk=int(game["game_pk"]),
            game_number=int(game["game_number"]),
            scheduled_start_utc=game["scheduled_start_utc"],
            home_score=int(game["home_score"]),
            away_score=int(game["away_score"]),
            status=game["status"],
            source_result_id=f"MLB_STATS_API:game/{game['game_pk']}/schedule:{game['official_date']}",
        )
        for game in sorted(target_games, key=lambda row: (row["scheduled_start_utc"], row["game_number"], row["game_pk"]))
    )
    feature_fingerprint = fingerprint_rows(tuple(row.projection() for row in frozen_features))
    result_fingerprint = fingerprint_rows(tuple(row.projection() for row in results))
    manifest = {
        "fold_id": FOLD_ID,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "feature_fingerprint": feature_fingerprint,
        "result_fingerprint": result_fingerprint,
        "source_manifest_fingerprint": source_manifest_fingerprint,
    }
    return FutureEvaluationFold(
        fold_id=FOLD_ID,
        training_information_boundary_utc=TRAINING_INFORMATION_BOUNDARY_UTC,
        validation_start=VALIDATION_START,
        validation_end=VALIDATION_END,
        feature_rows=frozen_features,
        result_rows=results,
        source_manifest_fingerprint=source_manifest_fingerprint,
        feature_fingerprint=feature_fingerprint,
        result_fingerprint=result_fingerprint,
        fold_fingerprint=fingerprint_manifest(manifest),
    )


def materialize_from_normalized_dir(
    normalized_root: str | Path,
    *,
    source_manifest_fingerprint: str,
) -> FutureEvaluationFold:
    root = Path(normalized_root)
    return materialize_future_moneyline_fold(
        schedule_rows=load_normalized_rows(root / "schedule.jsonl"),
        target_boxscore_rows=load_normalized_rows(root / "target_boxscores.jsonl"),
        pitcher_game_log_rows=load_normalized_rows(root / "pitcher_game_logs.jsonl"),
        source_manifest_fingerprint=source_manifest_fingerprint,
    )


__all__ = ("materialize_from_normalized_dir", "materialize_future_moneyline_fold")
