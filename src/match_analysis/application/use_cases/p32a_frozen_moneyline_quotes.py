"""Normalize an existing P32A acquisition against exact frozen P50C identities."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from ...infrastructure.providers.mlb_official_historical_source import parse_utc
from typing import Any

from .generate_tsl_moneyline_edge_batch import _crosswalk
from .moneyline_paper_decision import FrozenPredictionRun
from .acquire_tsl_moneyline_snapshot import TslMoneylineSnapshotAcquisition


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _same_timestamp(left: Any, right: Any) -> bool:
    try:
        a = datetime.fromisoformat(str(left).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(right).replace("Z", "+00:00"))
        return a.tzinfo is not None and b.tzinfo is not None and a == b
    except ValueError:
        return False


def quotes_from_acquisition(acquisition: TslMoneylineSnapshotAcquisition, run: FrozenPredictionRun, *, reject_ambiguous=False):
    wanted_by_game_pk = {
        int(row["game_identity"]["game_pk"]): row for row in run.predictions
    }
    candidates = _crosswalk(
        tsl_rows=acquisition.history.rows,
        schedule_rows=acquisition.target_schedule_rows,
        cohort_start_date=acquisition.target_date,
        cohort_end_date=acquisition.target_date,
    )
    quotes: list[dict[str, Any]] = []
    issues: dict[int, str] = {}
    for observation in candidates:
        official = observation.official
        if official is None:
            continue
        try:
            game_pk = int(str(official["provider_game_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        prediction = wanted_by_game_pk.get(game_pk)
        if prediction is None:
            continue
        game = prediction["game_identity"]
        if (
            official.get("game_type") != run.game_type
            or official.get("game_pk") != game.get("game_pk")
            or str(official.get("provider_game_id")) != str(game.get("provider_game_id"))
            or official.get("game_number") != game.get("game_number")
            or official.get("official_date") != game.get("official_date")
            or not _same_timestamp(
                official.get("scheduled_start_utc"), game.get("scheduled_start_utc")
            )
            or official.get("home_team", {}).get("name") != game.get("home_team")
            or official.get("away_team", {}).get("name") != game.get("away_team")
            or str(official.get("home_team", {}).get("abbreviation"))
            != str(game.get("home_team_code"))
            or str(official.get("away_team", {}).get("abbreviation"))
            != str(game.get("away_team_code"))
        ):
            issues[game_pk] = "QUOTE_IDENTITY_MISMATCH"
            continue
        if reject_ambiguous:
            compatible = [g for g in acquisition.target_schedule_rows
                if g["home_team"]["abbreviation"] == observation.canonical_home_code
                and g["away_team"]["abbreviation"] == observation.canonical_away_code
                and abs((parse_utc(g["scheduled_start_utc"]) -
                         parse_utc(observation.row["game_time"])).total_seconds()) <= 3600]
            if len(compatible) != 1:
                issues[game_pk] = "QUOTE_OFFICIAL_CROSSWALK_AMBIGUOUS"
                continue
        source_row = observation.row
        quote: dict[str, Any] = {
            "schema_version": "moneyline_paper_decision_v1.quote_snapshot.v1",
            "quote_source": source_row.get("source"),
            "market": "MONEYLINE",
            "market_code": "MNL",
            "observed_at_utc": source_row.get("fetched_at"),
            "game_identity": dict(game),
            "home_decimal_odds": format(observation.home_decimal_odds, "f"),
            "away_decimal_odds": format(observation.away_decimal_odds, "f"),
            "source_identity": {
                "source_match_id": source_row.get("match_id"),
                "source_game_time": source_row.get("game_time"),
                "source_row_index": observation.source_row_index,
                "source_row_fingerprint": observation.source_row_fingerprint,
                "crosswalk_time_delta_seconds": observation.crosswalk_time_delta_seconds,
                "crosswalk_authority": "P28AB_NEAREST_TEAM_AND_START_WITHIN_3600_SECONDS",
            },
        }
        quote["snapshot_identity"] = hashlib.sha256(_canonical_bytes(quote)).hexdigest()
        quotes.append(quote)
    return tuple(quotes), issues

