"""P52A daily Moneyline prospective prediction freeze use case.

Productizes the proven P51A pregame prediction workflow into a repo-owned application seam:
  1. Fetches official MLB schedule and pitcher pregame authorities as of a target timestamp.
  2. Filters target games strictly after the as-of cutoff.
  3. Constructs point-in-time features with strict cutoff enforcement:
     - Team recent win rate delta (10 prior completed games strictly before cutoff).
     - Starter ERA delta (starts strictly before target date / cutoff, >= 2 starts).
  4. Deterministically excludes games lacking required pregame authorities.
  5. Executes unchanged Champion model inference.
  6. Produces P50C-compatible normalized pregame input bundle (zero result/betting contamination).
  7. Invokes P50C prospective prediction freeze authority.
  8. Returns structured execution result with separated pending and settled sample accounting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, getcontext
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request

from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc
from ...infrastructure.providers.mlb_official_historical_source import (
    MLB_STATS_API_BASE,
    JsonOpener,
    _default_opener,
    canonical_json_bytes,
    fetch_json_bytes,
    format_utc,
    parse_utc,
    sha256_bytes,
)
from .p45a_paper_run_ledger import (
    P45A_REPORT_RELATIVE_PATH,
    get_p45a_forward_summary,
)
from .p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    FORBIDDEN_PREGAME_BETTING_FIELDS,
    FORBIDDEN_PREGAME_RESULT_FIELDS,
    create_p50c_prediction_run,
    get_p50c_forward_summary,
    get_p50c_run_status,
    reject_pregame_contamination,
)

getcontext().prec = 28

P52A_SOURCE_IDENTITY = "MLB_OFFICIAL_STATS_API_PROSPECTIVE_FEED"
P52A_PREGAME_SCHEMA_VERSION = "p50c.pregame_input.v1"
CHAMPION_MODEL_RELATIVE_PATH = Path("report/p22b_moneyline_challenger/model_artifact.json")
P50C_REPORT_RELATIVE_PATH = Path("report/p50c_prospective_prediction_shadow_ledger")

# Deterministic exclusion reason codes
EXCLUSION_NOT_FUTURE_GAME = "GAME_ALREADY_STARTED_OR_PAST_CUTOFF"
EXCLUSION_STARTER_UNAVAILABLE = "OFFICIAL_STARTER_IDENTITY_UNAVAILABLE"
EXCLUSION_INSUFFICIENT_STARTER_HISTORY = "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY"
EXCLUSION_STARTER_OUTS_ZERO = "STARTER_ERA_OUTS_ZERO"
EXCLUSION_INSUFFICIENT_TEAM_HISTORY = "INSUFFICIENT_RECENT_GAMES_HISTORY"


def _sha256_projection(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class DailyPredictionFreezeResult:
    target_date: str
    as_of_utc: str
    target_games_count: int
    eligible_predictions_count: int
    exclusion_count: int
    run_id: str | None
    freeze_status: str
    run_dir: Path | None
    pregame_input_path: Path
    pending_count: int
    settled_prediction_forward_sample_count: int
    betting_forward_sample_count: int
    predictions: tuple[dict[str, Any], ...]
    exclusions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_date": self.target_date,
            "as_of_utc": self.as_of_utc,
            "target_games_count": self.target_games_count,
            "eligible_predictions_count": self.eligible_predictions_count,
            "exclusion_count": self.exclusion_count,
            "run_id": self.run_id,
            "freeze_status": self.freeze_status,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "pregame_input_path": str(self.pregame_input_path),
            "pending_count": self.pending_count,
            "settled_prediction_forward_sample_count": self.settled_prediction_forward_sample_count,
            "betting_forward_sample_count": self.betting_forward_sample_count,
            "predictions": list(self.predictions),
            "exclusions": list(self.exclusions),
        }


def load_champion_model_parameters(
    repository_root: Path,
) -> tuple[str, str, list[Decimal], Decimal, list[Decimal], list[Decimal]]:
    """Load model_id, fingerprint, coeffs, intercept, scaler_means, scaler_stds from champion artifact."""
    artifact_path = repository_root / CHAMPION_MODEL_RELATIVE_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"champion model artifact missing: {artifact_path}")
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    model_id = str(data["model_id"])
    fingerprint = str(data["artifact_fingerprint"])
    coeffs = [Decimal(str(c)) for c in data["coefficients"]]
    intercept = Decimal(str(data["intercept"]))
    scaler_means = [Decimal(str(m)) for m in data["scaler_means"]]
    scaler_stds = [Decimal(str(s)) for s in data["scaler_stds"]]
    return model_id, fingerprint, coeffs, intercept, scaler_means, scaler_stds


def fetch_mlb_schedule_data(
    start_date: str,
    end_date: str,
    *,
    opener: JsonOpener | None = None,
    timeout: int = 30,
) -> tuple[dict[str, Any], str, str]:
    """Fetch MLB schedule between start_date and end_date.

    Returns (parsed_json, request_url, sha256_hash).
    """
    actual_opener = opener if opener is not None else _default_opener
    url = f"{MLB_STATS_API_BASE}/schedule"
    query = {
        "sportId": "1",
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "probablePitcher(note),team",
    }
    raw_bytes, req_url = fetch_json_bytes(url, query=query, opener=actual_opener, timeout=timeout)
    parsed = json.loads(raw_bytes.decode("utf-8"))
    return parsed, req_url, sha256_bytes(raw_bytes)


def parse_schedule_rows(sched_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize schedule entries into standardized internal rows."""
    schedule_rows: list[dict[str, Any]] = []
    for d in sched_data.get("dates", []):
        official_date = d.get("date")
        for g in d.get("games", []):
            home_wrapper = g.get("teams", {}).get("home", {})
            away_wrapper = g.get("teams", {}).get("away", {})
            home_team = home_wrapper.get("team", {})
            away_team = away_wrapper.get("team", {})
            home_pitcher = home_wrapper.get("probablePitcher", {})
            away_pitcher = away_wrapper.get("probablePitcher", {})
            status = g.get("status", {})
            is_final = status.get("abstractGameState") == "Final"

            schedule_rows.append(
                {
                    "game_pk": g.get("gamePk"),
                    "game_number": g.get("gameNumber", 1),
                    "provider_game_id": str(g.get("gamePk")),
                    "scheduled_start_utc": g.get("gameDate"),
                    "official_date": official_date,
                    "final": is_final,
                    "status_detail": status.get("detailedState", ""),
                    "home_team": {
                        "id": home_team.get("id"),
                        "name": home_team.get("name"),
                        "abbreviation": home_team.get("abbreviation"),
                    },
                    "away_team": {
                        "id": away_team.get("id"),
                        "name": away_team.get("name"),
                        "abbreviation": away_team.get("abbreviation"),
                    },
                    "home_score": home_wrapper.get("score"),
                    "away_score": away_wrapper.get("score"),
                    "home_starter": (
                        {
                            "player_id": home_pitcher.get("id"),
                            "full_name": home_pitcher.get("fullName"),
                        }
                        if home_pitcher and home_pitcher.get("id")
                        else None
                    ),
                    "away_starter": (
                        {
                            "player_id": away_pitcher.get("id"),
                            "full_name": away_pitcher.get("fullName"),
                        }
                        if away_pitcher and away_pitcher.get("id")
                        else None
                    ),
                }
            )
    return schedule_rows


def fetch_pitcher_game_logs(
    pitcher_ids: Sequence[int],
    season: str,
    *,
    opener: JsonOpener | None = None,
    timeout: int = 30,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    """Fetch regular season pitching game logs for the requested pitcher IDs."""
    actual_opener = opener if opener is not None else _default_opener
    pitcher_logs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pitcher_sources: dict[str, dict[str, str]] = {}

    for pid in sorted(set(pitcher_ids)):
        url = f"{MLB_STATS_API_BASE}/people/{pid}/stats"
        query = {
            "stats": "gameLog",
            "group": "pitching",
            "season": season,
            "gameType": "R",
        }
        raw_bytes, req_url = fetch_json_bytes(url, query=query, opener=actual_opener, timeout=timeout)
        pitcher_sources[str(pid)] = {"url": req_url, "sha256": sha256_bytes(raw_bytes)}
        p_data = json.loads(raw_bytes.decode("utf-8"))
        stats_list = p_data.get("stats") or []
        splits = (stats_list[0].get("splits") or []) if stats_list else []
        for s in splits:
            stat = s.get("stat", {})
            pitcher_logs[pid].append(
                {
                    "player_id": pid,
                    "date": s.get("date"),
                    "game_type": "R",
                    "games_started": stat.get("gamesStarted", 0),
                    "outs": stat.get("outs", 0),
                    "earned_runs": stat.get("earnedRuns", 0),
                }
            )

    return dict(pitcher_logs), pitcher_sources


def compute_team_recent_win_rate_delta(
    home_id: int,
    away_id: int,
    schedule_rows: Sequence[Mapping[str, Any]],
    as_of_dt: datetime,
    target_start_dt: datetime,
) -> Decimal | None:
    """Compute (home_win_rate_10 - away_win_rate_10) using only completed games strictly before cutoff and start.

    Returns None if either team has fewer than 10 prior completed games.
    """
    cutoff_dt = min(as_of_dt, target_start_dt)
    prior_completed = [
        r
        for r in schedule_rows
        if r.get("final")
        and r.get("scheduled_start_utc")
        and parse_utc(r["scheduled_start_utc"]) < cutoff_dt
    ]

    history: dict[int, list[tuple[datetime, int]]] = defaultdict(list)
    for r in prior_completed:
        hs, as_ = r.get("home_score"), r.get("away_score")
        if hs is None or as_ is None or hs == as_:
            continue
        st = parse_utc(r["scheduled_start_utc"])
        ht_id = r["home_team"]["id"]
        at_id = r["away_team"]["id"]
        if ht_id is not None:
            history[ht_id].append((st, 1 if hs > as_ else 0))
        if at_id is not None:
            history[at_id].append((st, 1 if as_ > hs else 0))

    def get_rate(team_id: int) -> Decimal | None:
        games = sorted(history.get(team_id, []), key=lambda x: x[0])[-10:]
        if len(games) < 10:
            return None
        return Decimal(sum(res for _, res in games)) / Decimal(10)

    home_rate = get_rate(home_id)
    away_rate = get_rate(away_id)
    if home_rate is None or away_rate is None:
        return None
    return home_rate - away_rate


def compute_starter_era(
    pitcher_id: int,
    target_official_date: str,
    pitcher_logs: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[Decimal | None, str | None]:
    """Compute starter ERA using appearances strictly before target_official_date with games_started == 1.

    Requires >= 2 starts. Returns (era, None) or (None, exclusion_reason).
    """
    logs = pitcher_logs.get(pitcher_id, [])
    prior_starts = [
        r for r in logs if str(r.get("date")) < target_official_date and r.get("games_started") == 1
    ]
    if len(prior_starts) < 2:
        return (
            None,
            f"{EXCLUSION_INSUFFICIENT_STARTER_HISTORY} (starts={len(prior_starts)})",
        )
    total_outs = sum(int(r.get("outs", 0)) for r in prior_starts)
    total_er = sum(int(r.get("earned_runs", 0)) for r in prior_starts)
    if total_outs <= 0:
        return None, EXCLUSION_STARTER_OUTS_ZERO
    era = Decimal(total_er * 27) / Decimal(total_outs)
    return era, None


def execute_daily_moneyline_prediction_freeze(
    *,
    target_date: str,
    as_of_utc: str | datetime | None = None,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    history_start_date: str | None = None,
    opener: JsonOpener | None = None,
) -> DailyPredictionFreezeResult:
    """Execute the complete daily Moneyline prediction freeze flow for one MLB slate.

    Steps:
      1. Resolve repo, output paths, and exact as-of UTC timestamp.
      2. Fetch schedule and pitcher logs from MLB Stats API.
      3. Apply strict pre-cutoff filtering and point-in-time feature construction.
      4. Run unchanged Champion model inference.
      5. Form normalized pregame input bundle and save to intake directory.
      6. Freeze prediction run via P50C authority.
      7. Return structured execution result.
    """
    repo_root = Path(repository_root or Path.cwd()).resolve()
    out_root = Path(output_root or repo_root).resolve()

    if as_of_utc is None:
        as_of_dt = datetime.now(UTC)
        as_of_str = format_utc(as_of_dt)
    elif isinstance(as_of_utc, datetime):
        as_of_dt = as_of_utc.astimezone(UTC)
        as_of_str = format_utc(as_of_dt)
    else:
        as_of_dt = parse_utc(as_of_utc)
        as_of_str = format_utc(as_of_dt)

    target_year = target_date[:4]
    hist_start = history_start_date or f"{target_year}-03-01"

    # Step 1: Load Champion Model Artifact
    (
        champion_model_id,
        champion_model_fp,
        coeffs,
        intercept,
        scaler_means,
        scaler_stds,
    ) = load_champion_model_parameters(repo_root)

    # Step 2: Fetch Official Schedule
    sched_data, sched_url, sched_sha256 = fetch_mlb_schedule_data(
        hist_start,
        target_date,
        opener=opener,
    )
    schedule_rows = parse_schedule_rows(sched_data)

    target_games = [r for r in schedule_rows if r.get("official_date") == target_date]

    # Collect starter IDs from target games
    pitcher_ids: set[int] = set()
    for g in target_games:
        hs = g.get("home_starter")
        as_ = g.get("away_starter")
        if hs and hs.get("player_id"):
            pitcher_ids.add(hs["player_id"])
        if as_ and as_.get("player_id"):
            pitcher_ids.add(as_["player_id"])

    # Step 3: Fetch Pitcher Game Logs
    pitcher_logs, pitcher_sources = fetch_pitcher_game_logs(
        sorted(pitcher_ids),
        season=target_year,
        opener=opener,
    )

    # Step 4: Evaluate Eligibility, Features, and Inference for each target game
    prediction_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []

    for g in target_games:
        gpk = g["game_pk"]
        scheduled_start_str = g.get("scheduled_start_utc", "")
        if not scheduled_start_str:
            exclusion_rows.append(
                {
                    "game_pk": gpk,
                    "provider_game_id": str(gpk),
                    "scheduled_start_utc": "",
                    "official_date": target_date,
                    "reason": "SCHEDULED_START_UNAVAILABLE",
                    "feature_name": "temporal_cutoff",
                }
            )
            continue

        scheduled_start_dt = parse_utc(scheduled_start_str)

        # Check 1: Game must be strictly in the future relative to as_of cutoff
        if scheduled_start_dt <= as_of_dt:
            exclusion_rows.append(
                {
                    "game_pk": gpk,
                    "provider_game_id": str(gpk),
                    "scheduled_start_utc": scheduled_start_str,
                    "official_date": target_date,
                    "reason": EXCLUSION_NOT_FUTURE_GAME,
                    "feature_name": "scheduled_start_utc",
                }
            )
            continue

        ht = g.get("home_team") or {}
        at = g.get("away_team") or {}
        hs = g.get("home_starter")
        as_ = g.get("away_starter")

        # Check 2: Probable starters must be officially announced
        if not hs or not as_ or not hs.get("player_id") or not as_.get("player_id"):
            exclusion_rows.append(
                {
                    "game_pk": gpk,
                    "provider_game_id": str(gpk),
                    "scheduled_start_utc": scheduled_start_str,
                    "official_date": target_date,
                    "reason": EXCLUSION_STARTER_UNAVAILABLE,
                    "feature_name": "starter_era_delta",
                }
            )
            continue

        # Check 3: Starter ERA delta computation
        h_era, h_err = compute_starter_era(hs["player_id"], target_date, pitcher_logs)
        a_era, a_err = compute_starter_era(as_["player_id"], target_date, pitcher_logs)

        if h_era is None or a_era is None:
            exclusion_rows.append(
                {
                    "game_pk": gpk,
                    "provider_game_id": str(gpk),
                    "scheduled_start_utc": scheduled_start_str,
                    "official_date": target_date,
                    "reason": h_err or a_err or EXCLUSION_INSUFFICIENT_STARTER_HISTORY,
                    "feature_name": "starter_era_delta",
                }
            )
            continue

        # Check 4: Team recent win rate delta computation
        win_delta = compute_team_recent_win_rate_delta(
            ht.get("id"),
            at.get("id"),
            schedule_rows,
            as_of_dt,
            scheduled_start_dt,
        )
        if win_delta is None:
            exclusion_rows.append(
                {
                    "game_pk": gpk,
                    "provider_game_id": str(gpk),
                    "scheduled_start_utc": scheduled_start_str,
                    "official_date": target_date,
                    "reason": EXCLUSION_INSUFFICIENT_TEAM_HISTORY,
                    "feature_name": "recent_win_rate_delta",
                }
            )
            continue

        era_delta = h_era - a_era

        # Step 5: Unchanged Champion Logistic Inference
        z0 = (win_delta - scaler_means[0]) / scaler_stds[0]
        z1 = (era_delta - scaler_means[1]) / scaler_stds[1]
        logit = intercept + coeffs[0] * z0 + coeffs[1] * z1
        p_home = Decimal(1) / (Decimal(1) + Decimal(str(math.exp(float(-logit)))))
        p_home = min(Decimal("0.999999"), max(Decimal("0.000001"), p_home))

        row_id_payload = {
            "game_pk": gpk,
            "scheduled_start_utc": scheduled_start_str,
            "model_id": champion_model_id,
            "p_home": format(p_home, "f"),
        }
        pred_row_id = _sha256_projection(row_id_payload)

        pred_record = {
            "p37_fold_id": f"prospective_{target_date.replace('-', '_')}",
            "p37_window": f"window_{target_date.replace('-', '_')}",
            "p37_prediction_row_id": pred_row_id,
            "provider_namespace": "MLB_STATS_API",
            "provider_game_id": str(gpk),
            "game_pk": gpk,
            "game_number": g.get("game_number", 1),
            "scheduled_start_utc": scheduled_start_str,
            "champion_model_id": champion_model_id,
            "champion_model_fingerprint": champion_model_fp,
            "champion_home_probability": format(p_home, "f"),
            "challenger_model_id": champion_model_id,
            "challenger_model_fingerprint": champion_model_fp,
            "challenger_home_probability": format(p_home, "f"),
        }
        reject_pregame_contamination(pred_record)
        prediction_rows.append(pred_record)

    # Step 6: Form Pregame Input Bundle
    pregame_input_payload = {
        "schema_version": P52A_PREGAME_SCHEMA_VERSION,
        "source_identity": P52A_SOURCE_IDENTITY,
        "source_manifest": {
            "schedule_url": sched_url,
            "schedule_sha256": sched_sha256,
            "pitcher_sources": pitcher_sources,
            "fetched_at_utc": as_of_str,
            "target_date": target_date,
        },
        "predictions": prediction_rows,
        "exclusions": exclusion_rows,
    }

    intake_dir = out_root / P50C_REPORT_RELATIVE_PATH / "intake"
    intake_dir.mkdir(parents=True, exist_ok=True)
    clean_date = target_date.replace("-", "")
    pregame_input_file = intake_dir / f"prospective_pregame_{clean_date}.json"
    pregame_input_file.write_bytes(
        json.dumps(pregame_input_payload, indent=2, sort_keys=True).encode("utf-8")
    )

    # Step 7: Freeze Prediction Run via P50C Authority if predictions exist
    run_id: str | None = None
    freeze_status = "NO_ELIGIBLE_PREDICTIONS"
    run_dir: Path | None = None
    pending_count = 0

    if prediction_rows:
        create_result = create_p50c_prediction_run(
            repository_root=repo_root,
            pregame_input=pregame_input_file,
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            run_root=out_root / P50C_REPORT_RELATIVE_PATH / "runs",
            created_at_utc=as_of_str,
        )
        run_id = create_result.run_id
        freeze_status = create_result.status
        run_dir = create_result.run_dir

        status_dict = get_p50c_run_status(repo_root, run_dir=run_dir)
        pending_count = status_dict.get("pending_count", len(prediction_rows))

    # Step 8: Retrieve Forward Summaries
    p50c_summary = get_p50c_forward_summary(
        repo_root,
        ledger_root=out_root / P50C_REPORT_RELATIVE_PATH / "ledger"
        if output_root is not None
        else None,
    )
    p45a_summary = get_p45a_forward_summary(
        repo_root,
        ledger_root=out_root / P45A_REPORT_RELATIVE_PATH / "ledger"
        if output_root is not None
        else None,
    )

    return DailyPredictionFreezeResult(
        target_date=target_date,
        as_of_utc=as_of_str,
        target_games_count=len(target_games),
        eligible_predictions_count=len(prediction_rows),
        exclusion_count=len(exclusion_rows),
        run_id=run_id,
        freeze_status=freeze_status,
        run_dir=run_dir,
        pregame_input_path=pregame_input_file,
        pending_count=pending_count,
        settled_prediction_forward_sample_count=int(
            p50c_summary.get("PREDICTION_FORWARD_SAMPLE_COUNT", 0)
        ),
        betting_forward_sample_count=int(p45a_summary.get("forward_sample_count", 0)),
        predictions=tuple(prediction_rows),
        exclusions=tuple(exclusion_rows),
    )


__all__ = (
    "CHAMPION_MODEL_RELATIVE_PATH",
    "DailyPredictionFreezeResult",
    "EXCLUSION_INSUFFICIENT_STARTER_HISTORY",
    "EXCLUSION_INSUFFICIENT_TEAM_HISTORY",
    "EXCLUSION_NOT_FUTURE_GAME",
    "EXCLUSION_STARTER_OUTS_ZERO",
    "EXCLUSION_STARTER_UNAVAILABLE",
    "P52A_PREGAME_SCHEMA_VERSION",
    "P52A_SOURCE_IDENTITY",
    "compute_starter_era",
    "compute_team_recent_win_rate_delta",
    "execute_daily_moneyline_prediction_freeze",
    "fetch_mlb_schedule_data",
    "fetch_pitcher_game_logs",
    "load_champion_model_parameters",
    "parse_schedule_rows",
)
