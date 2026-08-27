"""Unit tests for P65A daily Run Line prospective prediction freeze."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.request import Request

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    FORBIDDEN_PREGAME_BETTING_FIELDS,
    FORBIDDEN_PREGAME_RESULT_FIELDS,
    read_json_object,
    read_jsonl_objects,
)
from match_analysis.application.use_cases.p65a_daily_run_line_prediction_freeze import (
    EXCLUSION_NOT_FUTURE_GAME,
    EXCLUSION_SCHEDULED_START_UNAVAILABLE,
    EXCLUSION_TEAM_IDENTITY_UNAVAILABLE,
    P65A_PREGAME_SCHEMA_VERSION,
    P65A_SOURCE_IDENTITY,
    execute_daily_run_line_prediction_freeze,
    parse_schedule_for_run_line,
)
from match_analysis.baseball.domain.canonical_utc import parse_canonical_utc


def _build_mock_schedule(
    target_date: str = "2026-08-21",
    target_games: list[dict] | None = None,
    completed_days: int = 10,
) -> dict:
    """Build mock MLB schedule payload with completed past games and future target games."""
    dates = []

    # Completed games
    for i in range(completed_days):
        day_str = f"2026-08-{i+1:02d}"
        dates.append(
            {
                "date": day_str,
                "games": [
                    {
                        "gamePk": 1000 + i,
                        "gameNumber": 1,
                        "officialDate": day_str,
                        "gameDate": f"{day_str}T19:00:00Z",
                        "status": {"abstractGameState": "Final", "detailedState": "Final"},
                        "teams": {
                            "home": {
                                "team": {"id": 10, "name": "New York Yankees", "abbreviation": "NYY"},
                                "score": 6 if i % 2 == 0 else 3,
                            },
                            "away": {
                                "team": {"id": 30, "name": "Boston Red Sox", "abbreviation": "BOS"},
                                "score": 2 if i % 2 == 0 else 5,
                            },
                        },
                    },
                    {
                        "gamePk": 2000 + i,
                        "gameNumber": 1,
                        "officialDate": day_str,
                        "gameDate": f"{day_str}T19:05:00Z",
                        "status": {"abstractGameState": "Final", "detailedState": "Final"},
                        "teams": {
                            "home": {
                                "team": {"id": 20, "name": "Los Angeles Dodgers", "abbreviation": "LAD"},
                                "score": 5 if i % 3 == 0 else 2,
                            },
                            "away": {
                                "team": {"id": 40, "name": "San Francisco Giants", "abbreviation": "SF"},
                                "score": 1 if i % 3 == 0 else 4,
                            },
                        },
                    },
                ],
            }
        )

    # Target games on target_date
    if target_games is None:
        target_games = [
            {
                "gamePk": 999001,
                "gameNumber": 1,
                "officialDate": target_date,
                "gameDate": f"{target_date}T20:00:00Z",
                "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
                "teams": {
                    "home": {
                        "team": {"id": 10, "name": "New York Yankees", "abbreviation": "NYY"},
                    },
                    "away": {
                        "team": {"id": 30, "name": "Boston Red Sox", "abbreviation": "BOS"},
                    },
                },
            },
            {
                "gamePk": 999002,
                "gameNumber": 1,
                "officialDate": target_date,
                "gameDate": f"{target_date}T21:00:00Z",
                "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
                "teams": {
                    "home": {
                        "team": {"id": 20, "name": "Los Angeles Dodgers", "abbreviation": "LAD"},
                    },
                    "away": {
                        "team": {"id": 40, "name": "San Francisco Giants", "abbreviation": "SF"},
                    },
                },
            },
        ]

    for game in target_games:
        game.setdefault("officialDate", target_date)

    dates.append(
        {
            "date": target_date,
            "games": target_games,
        }
    )

    return {"dates": dates}


class TestP65ADailyRunLinePredictionFreeze(unittest.TestCase):
    def test_end_to_end_freeze_flow(self) -> None:
        target_date = "2026-08-21"
        as_of_utc = "2026-08-21T01:30:25Z"
        sched_payload = _build_mock_schedule(target_date=target_date)

        def mock_opener(request: Request, timeout: int) -> bytes:
            return json.dumps(sched_payload).encode("utf-8")

        with tempfile.TemporaryDirectory(prefix="p65a-test-") as tmpdir:
            tmp_root = Path(tmpdir)
            result = execute_daily_run_line_prediction_freeze(
                target_date=target_date,
                as_of_utc=as_of_utc,
                repository_root=tmp_root,
                output_root=tmp_root,
                opener=mock_opener,
            )

            self.assertEqual(result.target_date, target_date)
            self.assertEqual(result.as_of_utc, as_of_utc)
            self.assertEqual(result.target_games_count, 2)
            self.assertEqual(result.eligible_predictions_count, 2)
            self.assertEqual(result.exclusion_count, 0)
            self.assertEqual(result.freeze_status, "FROZEN")
            self.assertEqual(result.pending_count, 2)
            self.assertIsNotNone(result.run_id)
            self.assertIsNotNone(result.run_dir)

            # Check intake artifact
            self.assertTrue(result.pregame_input_path.is_file())
            intake_data = json.loads(result.pregame_input_path.read_text(encoding="utf-8"))
            self.assertEqual(intake_data["schema_version"], P65A_PREGAME_SCHEMA_VERSION)
            self.assertEqual(intake_data["source_identity"], P65A_SOURCE_IDENTITY)
            self.assertEqual(intake_data["market_id"], "run_line")
            self.assertEqual(len(intake_data["predictions"]), 2)
            self.assertEqual(len(intake_data["exclusions"]), 0)

            # Check run manifest
            manifest_path = result.run_dir / "run_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["task_id"], "P65A")
            self.assertEqual(manifest["market_id"], "run_line")
            self.assertEqual(manifest["lifecycle_state"], "FROZEN")
            self.assertEqual(manifest["prediction_count"], 2)
            self.assertEqual(manifest["target_date"], target_date)

            # Invariant: Zero betting / TSL claims
            self.assertFalse(manifest["claims"]["real_betting"])
            self.assertFalse(manifest["claims"]["odds_or_market_dependent"])
            self.assertFalse(manifest["claims"]["tsl_access"])

            # Check frozen predictions file
            frozen_path = result.run_dir / "frozen_predictions.jsonl"
            self.assertTrue(frozen_path.is_file())
            rows = read_jsonl_objects(frozen_path)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["market_id"], "run_line")
                self.assertEqual(row["game_identity"]["official_date"], target_date)
                self.assertEqual(row["spread_home"], -1.5)
                self.assertIn(row["predicted_side"], ("HOME", "AWAY"))
                self.assertGreater(row["p_home_covers"] + row["p_away_covers"], 0.999)
                # Contamination guard
                for f in FORBIDDEN_PREGAME_BETTING_FIELDS:
                    self.assertNotIn(f, row)
                for f in FORBIDDEN_PREGAME_RESULT_FIELDS:
                    self.assertNotIn(f, row)

    def test_cross_midnight_provider_official_date_is_preserved_end_to_end(self) -> None:
        target_date = "2026-08-21"
        sched_payload = _build_mock_schedule(target_date=target_date)
        sched_payload["dates"][-1]["games"][0]["gameDate"] = "2026-08-22T00:10:00Z"

        def mock_opener(request: Request, timeout: int) -> bytes:
            return json.dumps(sched_payload).encode("utf-8")

        with tempfile.TemporaryDirectory(prefix="p65a-cross-midnight-") as tmpdir:
            result = execute_daily_run_line_prediction_freeze(
                target_date=target_date,
                as_of_utc="2026-08-21T01:30:25Z",
                repository_root=tmpdir,
                output_root=tmpdir,
                opener=mock_opener,
            )
            rows = read_jsonl_objects(result.run_dir / "frozen_predictions.jsonl")
            cross_midnight = next(row for row in rows if row["game_pk"] == 999001)
            self.assertEqual(cross_midnight["scheduled_start_utc"], "2026-08-22T00:10:00Z")
            self.assertEqual(cross_midnight["game_identity"]["official_date"], target_date)

    def test_missing_provider_official_date_fails_closed(self) -> None:
        sched_payload = _build_mock_schedule()
        del sched_payload["dates"][-1]["games"][0]["officialDate"]
        with self.assertRaisesRegex(ValueError, "P65A_CANONICAL_OFFICIAL_DATE_REQUIRED"):
            parse_schedule_for_run_line(
                sched_payload,
                parse_canonical_utc("2026-08-21T01:30:25Z"),
                "2026-08-21",
            )

    def test_strict_temporal_cutoff_and_exclusions(self) -> None:
        target_date = "2026-08-21"
        as_of_utc = "2026-08-21T01:30:25Z"

        target_games = [
            # Game 1: Eligible future game
            {
                "gamePk": 101,
                "gameNumber": 1,
                "gameDate": "2026-08-21T20:00:00Z",
                "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
                "teams": {
                    "home": {"team": {"id": 10, "name": "NYY", "abbreviation": "NYY"}},
                    "away": {"team": {"id": 30, "name": "BOS", "abbreviation": "BOS"}},
                },
            },
            # Game 2: Already started / past cutoff
            {
                "gamePk": 102,
                "gameNumber": 1,
                "gameDate": "2026-08-21T01:00:00Z",
                "status": {"abstractGameState": "Live", "detailedState": "In Progress"},
                "teams": {
                    "home": {"team": {"id": 20, "name": "LAD", "abbreviation": "LAD"}},
                    "away": {"team": {"id": 40, "name": "SF", "abbreviation": "SF"}},
                },
            },
            # Game 3: Missing team identity
            {
                "gamePk": 103,
                "gameNumber": 1,
                "gameDate": "2026-08-21T22:00:00Z",
                "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
                "teams": {
                    "home": {"team": {}},
                    "away": {"team": {"id": 40, "name": "SF", "abbreviation": "SF"}},
                },
            },
        ]

        sched_payload = _build_mock_schedule(target_date=target_date, target_games=target_games)

        def mock_opener(request: Request, timeout: int) -> bytes:
            return json.dumps(sched_payload).encode("utf-8")

        with tempfile.TemporaryDirectory(prefix="p65a-exclusion-") as tmpdir:
            tmp_root = Path(tmpdir)
            result = execute_daily_run_line_prediction_freeze(
                target_date=target_date,
                as_of_utc=as_of_utc,
                repository_root=tmp_root,
                output_root=tmp_root,
                opener=mock_opener,
            )

            self.assertEqual(result.target_games_count, 3)
            self.assertEqual(result.eligible_predictions_count, 1)
            self.assertEqual(result.exclusion_count, 2)

            # Verify specific exclusion reasons
            exclusion_reasons = {e["reason"] for e in result.exclusions}
            self.assertIn(EXCLUSION_NOT_FUTURE_GAME, exclusion_reasons)
            self.assertIn(EXCLUSION_TEAM_IDENTITY_UNAVAILABLE, exclusion_reasons)

    def test_idempotent_reexecution(self) -> None:
        target_date = "2026-08-21"
        as_of_utc = "2026-08-21T01:30:25Z"
        sched_payload = _build_mock_schedule(target_date=target_date)

        def mock_opener(request: Request, timeout: int) -> bytes:
            return json.dumps(sched_payload).encode("utf-8")

        with tempfile.TemporaryDirectory(prefix="p65a-idem-") as tmpdir:
            tmp_root = Path(tmpdir)
            res1 = execute_daily_run_line_prediction_freeze(
                target_date=target_date,
                as_of_utc=as_of_utc,
                repository_root=tmp_root,
                output_root=tmp_root,
                opener=mock_opener,
            )
            res2 = execute_daily_run_line_prediction_freeze(
                target_date=target_date,
                as_of_utc=as_of_utc,
                repository_root=tmp_root,
                output_root=tmp_root,
                opener=mock_opener,
            )

            self.assertEqual(res1.run_id, res2.run_id)
            self.assertEqual(
                (res1.run_dir / "frozen_predictions.jsonl").read_bytes(),
                (res2.run_dir / "frozen_predictions.jsonl").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
