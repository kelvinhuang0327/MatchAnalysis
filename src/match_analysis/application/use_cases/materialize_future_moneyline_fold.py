"""Build the P23F2 fold from normalized official MLB inputs only."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, getcontext
from hashlib import sha256
import json
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


FEATURE_EVALUABLE = "EVALUABLE"
FEATURE_UNAVAILABLE = "FEATURE_UNAVAILABLE"
FEATURE_UNAVAILABLE_REASON = "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY"
FEATURE_UNAVAILABLE_STARTER_IDENTITY_REASON = "OFFICIAL_STARTER_IDENTITY_UNAVAILABLE"
REQUIRED_STARTER_HISTORY = 2


@dataclass(frozen=True, slots=True)
class FutureFeatureEligibility:
    raw_game_ids: tuple[str, ...]
    evaluable_game_ids: tuple[str, ...]
    feature_unavailable_rows: tuple[dict[str, Any], ...]


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _decimal(value: Decimal) -> str:
    return format(value, "f")


def _target_games(
    *,
    schedule_rows: tuple[Mapping[str, Any], ...],
    validation_start: str,
    validation_end: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        sorted(
            (
                row
                for row in schedule_rows
                if validation_start <= str(row["official_date"]) <= validation_end
            ),
            key=lambda row: (
                str(row["scheduled_start_utc"]),
                int(row["game_number"]),
                int(row["game_pk"]),
            ),
        )
    )


def _qualifying_same_season_starts(
    logs_by_player: Mapping[int, tuple[Mapping[str, Any], ...]],
    player_id: int,
    target_date: str,
) -> int:
    target_season = target_date[:4]
    return sum(
        1
        for row in logs_by_player.get(player_id, ())
        if row["game_type"] == "R"
        and row["games_started"] == 1
        and str(row["date"])[:4] == target_season
        and str(row["date"]) < target_date
    )


def _eligibility_identity(
    *,
    fold_id: str,
    game_id: str,
    scheduled_start: str,
    feature_name: str,
    reason: str,
) -> str:
    payload = {
        "feature_name": feature_name,
        "fold_id": fold_id,
        "game_id": game_id,
        "reason": reason,
        "scheduled_start": scheduled_start,
    }
    canonical = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _feature_unavailable_row(
    *,
    fold_id: str,
    game: Mapping[str, Any],
    reason: str,
    affected_starters: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "game_id": str(game["provider_game_id"]),
        "scheduled_start": str(game["scheduled_start_utc"]),
        "status": FEATURE_UNAVAILABLE,
        "reason": reason,
        "feature_name": "starter_era_delta",
        "affected_starters": [dict(item) for item in affected_starters],
        "deterministic_exclusion_identity": _eligibility_identity(
            fold_id=fold_id,
            game_id=str(game["provider_game_id"]),
            scheduled_start=str(game["scheduled_start_utc"]),
            feature_name="starter_era_delta",
            reason=reason,
        ),
    }


def classify_future_feature_eligibility(
    *,
    schedule_rows: tuple[Mapping[str, Any], ...],
    target_boxscore_rows: tuple[Mapping[str, Any], ...],
    pitcher_game_log_rows: tuple[Mapping[str, Any], ...],
    fold_id: str,
    validation_start: str,
    validation_end: str,
    allow_missing_starter_identity: bool = False,
) -> FutureFeatureEligibility:
    """Classify raw games from pregame inputs without reading outcome fields."""

    target_games = _target_games(
        schedule_rows=schedule_rows,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    if len(target_games) < 2:
        raise RuntimeError("STOP_MATCHANALYSIS_P23F2_OFFICIAL_SOURCE_INSUFFICIENT")
    boxes = {str(row["provider_game_id"]): row for row in target_boxscore_rows}
    logs: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pitcher_game_log_rows:
        logs[int(row["player_id"])].append(row)
    logs_tuple = {player_id: tuple(rows) for player_id, rows in logs.items()}

    raw_game_ids: list[str] = []
    evaluable_game_ids: list[str] = []
    unavailable_rows: list[dict[str, Any]] = []
    for game in target_games:
        game_id = str(game["provider_game_id"])
        box = boxes.get(game_id)
        if box is None:
            if not allow_missing_starter_identity:
                raise RuntimeError("STOP_MATCHANALYSIS_P23F2_STARTER_IDENTITY_UNRESOLVED")
            raw_game_ids.append(game_id)
            unavailable_rows.append(
                _feature_unavailable_row(
                    fold_id=fold_id,
                    game=game,
                    reason=FEATURE_UNAVAILABLE_STARTER_IDENTITY_REASON,
                    affected_starters=(
                        {"side": "home", "starter_identity": "UNAVAILABLE"},
                        {"side": "away", "starter_identity": "UNAVAILABLE"},
                    ),
                )
            )
            continue
        raw_game_ids.append(game_id)
        affected_starters: list[dict[str, Any]] = []
        missing_starter_identity = False
        for side in ("home", "away"):
            starter = box.get(f"{side}_starter")
            if not isinstance(starter, Mapping):
                if not allow_missing_starter_identity:
                    raise RuntimeError(
                        "STOP_MATCHANALYSIS_P23F2_STARTER_IDENTITY_UNRESOLVED"
                    )
                missing_starter_identity = True
                affected_starters.append(
                    {"side": side, "starter_identity": "UNAVAILABLE"}
                )
                continue
            count = _qualifying_same_season_starts(
                logs_tuple,
                int(starter["player_id"]),
                str(game["official_date"]),
            )
            if count < REQUIRED_STARTER_HISTORY:
                affected_starters.append(
                    {
                        "side": side,
                        "team": str(game[f"{side}_team"]["name"]),
                        "starter_id": int(starter["player_id"]),
                        "starter_name": str(starter["full_name"]),
                        "qualifying_prior_start_count": count,
                        "required_prior_start_count": REQUIRED_STARTER_HISTORY,
                    }
                )
        if missing_starter_identity:
            unavailable_rows.append(
                _feature_unavailable_row(
                    fold_id=fold_id,
                    game=game,
                    reason=FEATURE_UNAVAILABLE_STARTER_IDENTITY_REASON,
                    affected_starters=tuple(affected_starters),
                )
            )
            continue
        if not affected_starters:
            evaluable_game_ids.append(game_id)
            continue
        unavailable_rows.append(
            _feature_unavailable_row(
                fold_id=fold_id,
                game=game,
                reason=FEATURE_UNAVAILABLE_REASON,
                affected_starters=tuple(affected_starters),
            )
        )
    return FutureFeatureEligibility(
        raw_game_ids=tuple(raw_game_ids),
        evaluable_game_ids=tuple(evaluable_game_ids),
        feature_unavailable_rows=tuple(unavailable_rows),
    )


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
    *,
    require_same_season_history: bool = False,
) -> Decimal:
    prior = [
        row for row in logs_by_player.get(player_id, ())
        if row["game_type"] == "R"
        and row["date"] < target_date
        and row["games_started"] == 1
    ]
    if require_same_season_history:
        target_season = target_date[:4]
        prior = [row for row in prior if str(row["date"])[:4] == target_season]
        if len(prior) < REQUIRED_STARTER_HISTORY:
            raise RuntimeError(
                "STOP_MATCHANALYSIS_P23B_INSUFFICIENT_SAME_SEASON_STARTER_HISTORY"
            )
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
    fold_id: str = FOLD_ID,
    validation_start: str = VALIDATION_START,
    validation_end: str = VALIDATION_END,
    evaluable_game_ids: frozenset[str] | None = None,
    raw_game_ids: tuple[str, ...] | None = None,
    feature_unavailable_rows: tuple[Mapping[str, Any], ...] = (),
    allow_insufficient_evaluable: bool = False,
) -> FutureEvaluationFold:
    """Materialize one future cohort, freezing features before results are joined."""

    boundary = _parse(TRAINING_INFORMATION_BOUNDARY_UTC)
    target_games = _target_games(
        schedule_rows=schedule_rows,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    if len(target_games) < 2:
        raise RuntimeError("STOP_MATCHANALYSIS_P23F2_OFFICIAL_SOURCE_INSUFFICIENT")
    if any(_parse(row["scheduled_start_utc"]) <= boundary for row in target_games):
        raise RuntimeError("STOP_MATCHANALYSIS_P23F2_BASELINE_DRIFT")
    if any(not row["final"] for row in target_games):
        raise RuntimeError("STOP_MATCHANALYSIS_P23F2_OFFICIAL_SOURCE_INSUFFICIENT")
    target_game_ids = tuple(str(row["provider_game_id"]) for row in target_games)
    raw_ids = tuple(raw_game_ids or target_game_ids)
    if set(raw_ids) != set(target_game_ids) or len(raw_ids) != len(target_game_ids):
        raise ValueError("raw game identities must match the target fold")
    if evaluable_game_ids is None:
        evaluable_ids = frozenset(target_game_ids)
    else:
        if raw_game_ids is None:
            raise ValueError("P23B evaluable membership requires raw membership")
        evaluable_ids = frozenset(str(game_id) for game_id in evaluable_game_ids)
        if not evaluable_ids <= set(target_game_ids):
            raise ValueError("evaluable game identities must be raw fold games")
    if len(evaluable_ids) < 2 and not allow_insufficient_evaluable:
        raise RuntimeError("STOP_MATCHANALYSIS_P23B_INSUFFICIENT_EVALUABLE_GAMES")
    feature_games = tuple(
        game for game in target_games if str(game["provider_game_id"]) in evaluable_ids
    )
    boxes = {row["provider_game_id"]: row for row in target_boxscore_rows}
    logs: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pitcher_game_log_rows:
        logs[int(row["player_id"])].append(row)
    logs_tuple = {player_id: tuple(rows) for player_id, rows in logs.items()}

    feature_rows: list[FutureFeatureRow] = []
    for game in feature_games:
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
        starter_delta = _starter_era(
            logs_tuple,
            int(home_starter["player_id"]),
            game["official_date"],
            require_same_season_history=evaluable_game_ids is not None,
        ) - _starter_era(
            logs_tuple,
            int(away_starter["player_id"]),
            game["official_date"],
            require_same_season_history=evaluable_game_ids is not None,
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
        for game in target_games
    )
    feature_fingerprint = fingerprint_rows(tuple(row.projection() for row in frozen_features))
    result_fingerprint = fingerprint_rows(tuple(row.projection() for row in results))
    preserve_raw_membership = (
        raw_game_ids is not None or bool(feature_unavailable_rows)
    )
    manifest = {
        "fold_id": fold_id,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "feature_fingerprint": feature_fingerprint,
        "result_fingerprint": result_fingerprint,
        "source_manifest_fingerprint": source_manifest_fingerprint,
    }
    if preserve_raw_membership:
        manifest.update(
            {
                "raw_game_ids": list(raw_ids),
                "feature_unavailable": [
                    dict(row) for row in feature_unavailable_rows
                ],
            }
        )
    return FutureEvaluationFold(
        fold_id=fold_id,
        training_information_boundary_utc=TRAINING_INFORMATION_BOUNDARY_UTC,
        validation_start=validation_start,
        validation_end=validation_end,
        feature_rows=frozen_features,
        result_rows=results,
        source_manifest_fingerprint=source_manifest_fingerprint,
        feature_fingerprint=feature_fingerprint,
        result_fingerprint=result_fingerprint,
        fold_fingerprint=fingerprint_manifest(manifest),
        raw_game_ids=raw_ids if preserve_raw_membership else (),
        feature_unavailable_rows=tuple(dict(row) for row in feature_unavailable_rows),
    )


def materialize_from_normalized_dir(
    normalized_root: str | Path,
    *,
    source_manifest_fingerprint: str,
    fold_id: str = FOLD_ID,
    validation_start: str = VALIDATION_START,
    validation_end: str = VALIDATION_END,
) -> FutureEvaluationFold:
    root = Path(normalized_root)
    return materialize_future_moneyline_fold(
        schedule_rows=load_normalized_rows(root / "schedule.jsonl"),
        target_boxscore_rows=load_normalized_rows(root / "target_boxscores.jsonl"),
        pitcher_game_log_rows=load_normalized_rows(root / "pitcher_game_logs.jsonl"),
        source_manifest_fingerprint=source_manifest_fingerprint,
        fold_id=fold_id,
        validation_start=validation_start,
        validation_end=validation_end,
    )


__all__ = (
    "FEATURE_EVALUABLE",
    "FEATURE_UNAVAILABLE",
    "FEATURE_UNAVAILABLE_REASON",
    "FEATURE_UNAVAILABLE_STARTER_IDENTITY_REASON",
    "FutureFeatureEligibility",
    "classify_future_feature_eligibility",
    "materialize_from_normalized_dir",
    "materialize_future_moneyline_fold",
)
