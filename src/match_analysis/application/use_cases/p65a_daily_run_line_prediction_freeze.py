"""P65A daily Run Line prospective prediction freeze use case.

Implements the automated pregame workflow for the Run Line market:
  1. Fetches official MLB schedule and scoring history between season start and target date.
  2. Extracts point-in-time prior completed games strictly on or before cutoff as PriorGameRecords.
  3. Filters target games strictly after the as-of cutoff.
  4. Formulates standard Run Line contest items (canonical spread_home = -1.5).
  5. Executes deterministic Run Line Poisson team-rate model (p226a.raw.v1) with Bayesian shrinkage.
  6. Produces P50C-compatible normalized pregame input bundle (zero result/betting contamination).
  7. Freezes immutable Run Line prediction run into canonical P50C runs repository.
  8. Returns structured execution result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request

from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc
from ...baseball.domain.run_line_model import (
    DEFAULT_LEAGUE_AVG_RUNS,
    FROZEN_HOME_ADVANTAGE,
    MARKET_ID,
    MODEL_ID,
    MODEL_VERSION,
    POISSON_MAX_RUNS,
    RUN_SMOOTH_K,
    PriorGameRecord,
)
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
from .p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P50C_REPORT_RELATIVE_PATH,
    reject_pregame_contamination,
)
from .run_line_prediction_lifecycle import (
    DEFAULT_CANONICAL_SPREAD_HOME,
    P61B_RUN_MANIFEST_SCHEMA,
    RunLineFrozenPrediction,
    RunLinePregameInputItem,
    freeze_run_line_predictions,
)

P65A_SOURCE_IDENTITY = "MLB_OFFICIAL_STATS_API_PROSPECTIVE_FEED"
P65A_PREGAME_SCHEMA_VERSION = "p65a.run_line_pregame_input.v1"

# Deterministic exclusion reason codes
EXCLUSION_NOT_FUTURE_GAME = "GAME_ALREADY_STARTED_OR_PAST_CUTOFF"
EXCLUSION_TEAM_IDENTITY_UNAVAILABLE = "TEAM_IDENTITY_UNAVAILABLE"
EXCLUSION_SCHEDULED_START_UNAVAILABLE = "SCHEDULED_START_UNAVAILABLE"


def _sha256_projection(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class DailyRunLinePredictionFreezeResult:
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
    predictions: tuple[RunLineFrozenPrediction, ...]
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
            "market_id": MARKET_ID,
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "predictions": [p.to_payload() for p in self.predictions],
            "exclusions": list(self.exclusions),
        }


def fetch_mlb_schedule_for_run_line(
    start_date: str,
    end_date: str,
    *,
    opener: JsonOpener | None = None,
    timeout: int = 30,
) -> tuple[dict[str, Any], str, str]:
    """Fetch MLB schedule between start_date and end_date."""
    actual_opener = opener if opener is not None else _default_opener
    url = f"{MLB_STATS_API_BASE}/schedule"
    query = {
        "sportId": "1",
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "team",
    }
    raw_bytes, req_url = fetch_json_bytes(url, query=query, opener=actual_opener, timeout=timeout)
    parsed = json.loads(raw_bytes.decode("utf-8"))
    return parsed, req_url, sha256_bytes(raw_bytes)


def parse_schedule_for_run_line(
    sched_data: Mapping[str, Any],
    as_of_dt: datetime,
    target_date: str,
    spread_home: float = DEFAULT_CANONICAL_SPREAD_HOME,
) -> tuple[list[PriorGameRecord], list[RunLinePregameInputItem], list[dict[str, Any]], int]:
    """Extract prior completed games and target pregame contest items with cutoff enforcement."""
    prior_games: list[PriorGameRecord] = []
    target_items: list[RunLinePregameInputItem] = []
    exclusions: list[dict[str, Any]] = []
    target_games_count = 0

    as_of_str = format_canonical_utc(as_of_dt)

    for d in sched_data.get("dates", []):
        for g in d.get("games", []):
            official_date = g.get("officialDate")
            if not isinstance(official_date, str) or not official_date.strip():
                raise ValueError(
                    "P65A_CANONICAL_OFFICIAL_DATE_REQUIRED: MLB game is missing provider officialDate"
                )
            status = g.get("status", {})
            is_final = status.get("abstractGameState") == "Final" or status.get("detailedState") == "Final"

            home_wrapper = g.get("teams", {}).get("home", {})
            away_wrapper = g.get("teams", {}).get("away", {})
            home_team_info = home_wrapper.get("team", {})
            away_team_info = away_wrapper.get("team", {})

            home_abbr = home_team_info.get("abbreviation") or home_team_info.get("name")
            away_abbr = away_team_info.get("abbreviation") or away_team_info.get("name")

            game_date_str = g.get("gameDate") or g.get("scheduled_start_utc")
            game_pk = g.get("gamePk")
            provider_game_id = str(game_pk or g.get("provider_game_id") or "")

            # Prior completed games collection
            if is_final and home_abbr and away_abbr and game_date_str:
                try:
                    game_dt = parse_canonical_utc(game_date_str)
                    if game_dt <= as_of_dt:
                        hs = home_wrapper.get("score")
                        as_ = away_wrapper.get("score")
                        if hs is not None and as_ is not None:
                            prior_games.append(
                                PriorGameRecord(
                                    home_team=str(home_abbr),
                                    away_team=str(away_abbr),
                                    home_score=int(hs),
                                    away_score=int(as_),
                                    completed_at_utc=game_dt,
                                )
                            )
                except Exception:
                    pass

            # Target date pregame item evaluation
            if official_date == target_date:
                target_games_count += 1

                if not game_date_str:
                    exclusions.append(
                        {
                            "game_pk": game_pk,
                            "provider_game_id": provider_game_id,
                            "scheduled_start_utc": "",
                            "official_date": target_date,
                            "reason": EXCLUSION_SCHEDULED_START_UNAVAILABLE,
                        }
                    )
                    continue

                try:
                    scheduled_dt = parse_canonical_utc(game_date_str)
                except Exception:
                    exclusions.append(
                        {
                            "game_pk": game_pk,
                            "provider_game_id": provider_game_id,
                            "scheduled_start_utc": game_date_str,
                            "official_date": target_date,
                            "reason": EXCLUSION_SCHEDULED_START_UNAVAILABLE,
                        }
                    )
                    continue

                if scheduled_dt <= as_of_dt:
                    exclusions.append(
                        {
                            "game_pk": game_pk,
                            "provider_game_id": provider_game_id,
                            "scheduled_start_utc": game_date_str,
                            "official_date": target_date,
                            "reason": EXCLUSION_NOT_FUTURE_GAME,
                        }
                    )
                    continue

                if not home_abbr or not away_abbr:
                    exclusions.append(
                        {
                            "game_pk": game_pk,
                            "provider_game_id": provider_game_id,
                            "scheduled_start_utc": game_date_str,
                            "official_date": target_date,
                            "reason": EXCLUSION_TEAM_IDENTITY_UNAVAILABLE,
                        }
                    )
                    continue

                target_items.append(
                    RunLinePregameInputItem(
                        provider_game_id=provider_game_id,
                        home_team=str(home_abbr),
                        away_team=str(away_abbr),
                        scheduled_start_utc=game_date_str,
                        official_date=official_date,
                        as_of_utc=as_of_str,
                        spread_home=spread_home,
                        provider_namespace=str(g.get("provider_namespace", "MLB_STATS_API")),
                        game_number=int(g.get("gameNumber", 1)),
                        game_pk=int(game_pk) if game_pk is not None else None,
                        season=int(official_date[:4]) if official_date else 2026,
                    )
                )

    return prior_games, target_items, exclusions, target_games_count


def execute_daily_run_line_prediction_freeze(
    *,
    target_date: str,
    as_of_utc: str | datetime | None = None,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    history_start_date: str | None = None,
    spread_home: float = DEFAULT_CANONICAL_SPREAD_HOME,
    opener: JsonOpener | None = None,
) -> DailyRunLinePredictionFreezeResult:
    """Execute the complete daily Run Line prediction freeze flow for one MLB slate.

    Steps:
      1. Resolve repo, output paths, and exact as-of UTC timestamp.
      2. Fetch schedule and game history from MLB Stats API.
      3. Apply strict pre-cutoff filtering and extract PriorGameRecords.
      4. Run Run Line Poisson team-rate model and generate frozen predictions.
      5. Form normalized pregame intake bundle and write to intake directory.
      6. Freeze prediction run into canonical P50C runs repository.
      7. Return structured execution result.
    """
    repo_root = Path(repository_root or Path.cwd()).resolve()
    out_root = Path(output_root or repo_root).resolve()

    if as_of_utc is None:
        as_of_dt = datetime.now(timezone.utc)
        as_of_str = format_canonical_utc(as_of_dt)
    elif isinstance(as_of_utc, datetime):
        as_of_dt = as_of_utc.astimezone(timezone.utc)
        as_of_str = format_canonical_utc(as_of_dt)
    else:
        as_of_dt = parse_canonical_utc(as_of_utc)
        as_of_str = format_canonical_utc(as_of_dt)

    target_year = target_date[:4]
    hist_start = history_start_date or f"{target_year}-03-01"

    # Step 1: Fetch Official MLB Schedule
    sched_data, sched_url, sched_sha256 = fetch_mlb_schedule_for_run_line(
        hist_start,
        target_date,
        opener=opener,
    )

    # Step 2: Parse Schedule & History with Cutoff Enforcement
    prior_games, target_items, exclusions, target_games_count = parse_schedule_for_run_line(
        sched_data,
        as_of_dt,
        target_date,
        spread_home=spread_home,
    )

    # Step 3: Run Prediction Lifecycle Freeze if eligible items exist
    clean_date = target_date.replace("-", "")
    run_id: str | None = None
    freeze_status = "NO_ELIGIBLE_PREDICTIONS"
    run_dir: Path | None = None
    pending_count = 0
    predictions_tuple: tuple[RunLineFrozenPrediction, ...] = ()

    intake_dir = out_root / P50C_REPORT_RELATIVE_PATH / "intake"
    intake_dir.mkdir(parents=True, exist_ok=True)
    pregame_input_file = intake_dir / f"prospective_pregame_run_line_{clean_date}.json"

    if target_items:
        # Pre-compute fingerprints to determine deterministic run_id
        temp_preds, _ = freeze_run_line_predictions(
            target_items,
            prior_games,
            run_id="temp_hash",
            task_id="P65A",
            created_at_utc=as_of_str,
            target_date=target_date,
        )
        pred_fps = [p.prediction_fingerprint for p in temp_preds]
        bundle_fp = sha256(canonical_json_bytes(pred_fps)).hexdigest()
        run_id = f"p50c_run_rl_{bundle_fp[:32]}"
        run_dir = out_root / P50C_REPORT_RELATIVE_PATH / "runs" / run_id

        # Persist frozen predictions
        predictions_tuple, manifest = freeze_run_line_predictions(
            target_items,
            prior_games,
            run_id=run_id,
            task_id="P65A",
            output_dir=run_dir,
            created_at_utc=as_of_str,
            target_date=target_date,
        )
        freeze_status = "FROZEN"
        pending_count = len(predictions_tuple)

    # Step 4: Write Normalized Intake Artifact
    pregame_input_payload = {
        "schema_version": P65A_PREGAME_SCHEMA_VERSION,
        "source_identity": P65A_SOURCE_IDENTITY,
        "market_id": MARKET_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "source_manifest": {
            "schedule_url": sched_url,
            "schedule_sha256": sched_sha256,
            "fetched_at_utc": as_of_str,
            "target_date": target_date,
            "history_start_date": hist_start,
            "prior_completed_games_count": len(prior_games),
        },
        "predictions": [p.to_payload() for p in predictions_tuple],
        "exclusions": exclusions,
    }

    pregame_input_file.write_bytes(
        json.dumps(pregame_input_payload, indent=2, sort_keys=True).encode("utf-8")
    )

    return DailyRunLinePredictionFreezeResult(
        target_date=target_date,
        as_of_utc=as_of_str,
        target_games_count=target_games_count,
        eligible_predictions_count=len(predictions_tuple),
        exclusion_count=len(exclusions),
        run_id=run_id,
        freeze_status=freeze_status,
        run_dir=run_dir,
        pregame_input_path=pregame_input_file,
        pending_count=pending_count,
        predictions=predictions_tuple,
        exclusions=tuple(exclusions),
    )
