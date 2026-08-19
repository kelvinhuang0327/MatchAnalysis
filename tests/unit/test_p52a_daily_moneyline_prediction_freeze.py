"""Focused unit tests for P52A daily Moneyline prospective prediction freeze."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    FORBIDDEN_PREGAME_BETTING_FIELDS,
    FORBIDDEN_PREGAME_RESULT_FIELDS,
    read_json_object,
    read_jsonl_objects,
)
from match_analysis.application.use_cases.p52a_daily_prediction_freeze import (
    EXCLUSION_INSUFFICIENT_STARTER_HISTORY,
    EXCLUSION_INSUFFICIENT_TEAM_HISTORY,
    EXCLUSION_NOT_FUTURE_GAME,
    EXCLUSION_STARTER_UNAVAILABLE,
    P52A_PREGAME_SCHEMA_VERSION,
    P52A_SOURCE_IDENTITY,
    compute_starter_era,
    compute_team_recent_win_rate_delta,
    execute_daily_moneyline_prediction_freeze,
    load_champion_model_parameters,
    parse_schedule_rows,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _build_mock_schedule_payload(
    target_date: str = "2026-08-18",
    target_start_utc: str = "2026-08-18T23:05:00Z",
    home_starter_id: int | None = 101,
    away_starter_id: int | None = 201,
    include_extra_past_games: int = 15,
) -> dict:
    """Build mock MLB schedule payload with completed past games and future target game."""
    dates = []

    # Build completed history for team 10 and team 20
    for i in range(include_extra_past_games):
        day_str = f"2026-08-{i+1:02d}"
        dates.append(
            {
                "date": day_str,
                "games": [
                    {
                        "gamePk": 1000 + i,
                        "gameNumber": 1,
                        "gameDate": f"{day_str}T19:00:00Z",
                        "status": {"abstractGameState": "Final", "detailedState": "Final"},
                        "teams": {
                            "home": {
                                "team": {"id": 10, "name": "New York Yankees", "abbreviation": "NYY"},
                                "score": 5 if i % 2 == 0 else 3,
                            },
                            "away": {
                                "team": {"id": 30, "name": "Boston Red Sox", "abbreviation": "BOS"},
                                "score": 2 if i % 2 == 0 else 4,
                            },
                        },
                    },
                    {
                        "gamePk": 2000 + i,
                        "gameNumber": 1,
                        "gameDate": f"{day_str}T19:05:00Z",
                        "status": {"abstractGameState": "Final", "detailedState": "Final"},
                        "teams": {
                            "home": {
                                "team": {"id": 20, "name": "Los Angeles Dodgers", "abbreviation": "LAD"},
                                "score": 6 if i % 3 == 0 else 2,
                            },
                            "away": {
                                "team": {"id": 40, "name": "San Francisco Giants", "abbreviation": "SF"},
                                "score": 1 if i % 3 == 0 else 5,
                            },
                        },
                    },
                ],
            }
        )

    # Add target game on target_date
    target_games = [
        {
            "gamePk": 999001,
            "gameNumber": 1,
            "gameDate": target_start_utc,
            "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
            "teams": {
                "home": {
                    "team": {"id": 10, "name": "New York Yankees", "abbreviation": "NYY"},
                    "probablePitcher": (
                        {"id": home_starter_id, "fullName": "Gerrit Cole"}
                        if home_starter_id
                        else None
                    ),
                },
                "away": {
                    "team": {"id": 20, "name": "Los Angeles Dodgers", "abbreviation": "LAD"},
                    "probablePitcher": (
                        {"id": away_starter_id, "fullName": "Clayton Kershaw"}
                        if away_starter_id
                        else None
                    ),
                },
            },
        }
    ]

    dates.append({"date": target_date, "games": target_games})
    return {"dates": dates}


def _build_mock_pitcher_log_payload(player_id: int, start_count: int = 5) -> dict:
    """Build mock pitcher game logs."""
    splits = []
    for i in range(start_count):
        splits.append(
            {
                "date": f"2026-07-{i+1:02d}",
                "gameType": "R",
                "stat": {
                    "gamesStarted": 1,
                    "outs": 18,  # 6 innings
                    "earnedRuns": 2 if i % 2 == 0 else 1,
                },
            }
        )
    return {"stats": [{"splits": splits}]}


class MockMLBOpener:
    """Mock URL opener routing MLB API endpoints to deterministic JSON fixtures."""

    def __init__(
        self,
        schedule_payload: dict,
        pitcher_logs_map: dict[int, dict] | None = None,
    ):
        self.schedule_payload = schedule_payload
        self.pitcher_logs_map = pitcher_logs_map or {}
        self.calls: list[str] = []

    def __call__(self, request: Request, timeout: int = 30) -> bytes:
        url = request.full_url
        self.calls.append(url)
        parsed = urlparse(url)

        if parsed.path.endswith("/schedule"):
            return json.dumps(self.schedule_payload).encode("utf-8")

        if "/people/" in parsed.path and "/stats" in parsed.path:
            parts = parsed.path.strip("/").split("/")
            pid_str = parts[parts.index("people") + 1]
            pid = int(pid_str)
            payload = self.pitcher_logs_map.get(pid, _build_mock_pitcher_log_payload(pid, start_count=5))
            return json.dumps(payload).encode("utf-8")

        raise ValueError(f"unhandled mock URL: {url}")


class TestP52ADailyMoneylinePredictionFreeze(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_champion_model_parameters(self) -> None:
        model_id, fp, coeffs, intercept, means, stds = load_champion_model_parameters(REPOSITORY_ROOT)
        self.assertTrue(model_id.startswith("p22b_moneyline_logistic_challenger"))
        self.assertEqual(len(fp), 64)
        self.assertEqual(len(coeffs), 2)
        self.assertEqual(len(means), 2)
        self.assertEqual(len(stds), 2)
        self.assertIsInstance(intercept, Decimal)

    def test_future_game_eligibility_and_freeze_success(self) -> None:
        as_of_utc = "2026-08-18T13:00:00Z"
        target_date = "2026-08-18"
        target_start = "2026-08-18T23:05:00Z"

        sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc=target_start,
            home_starter_id=101,
            away_starter_id=201,
        )
        p_map = {
            101: _build_mock_pitcher_log_payload(101, start_count=5),
            201: _build_mock_pitcher_log_payload(201, start_count=4),
        }
        mock_opener = MockMLBOpener(sched, p_map)

        result = execute_daily_moneyline_prediction_freeze(
            target_date=target_date,
            as_of_utc=as_of_utc,
            repository_root=REPOSITORY_ROOT,
            output_root=self.output_root,
            opener=mock_opener,
        )

        self.assertEqual(result.target_date, target_date)
        self.assertEqual(result.as_of_utc, as_of_utc)
        self.assertEqual(result.target_games_count, 1)
        self.assertEqual(result.eligible_predictions_count, 1)
        self.assertEqual(result.exclusion_count, 0)
        self.assertEqual(result.freeze_status, "CREATED")
        self.assertIsNotNone(result.run_id)
        self.assertTrue(result.run_id.startswith("p50c_run_"))
        self.assertEqual(result.pending_count, 1)
        self.assertEqual(result.settled_prediction_forward_sample_count, 0)
        self.assertEqual(result.betting_forward_sample_count, 0)

        # Inspect pregame input file
        self.assertTrue(result.pregame_input_path.is_file())
        pregame_data = read_json_object(result.pregame_input_path)
        self.assertEqual(pregame_data["schema_version"], P52A_PREGAME_SCHEMA_VERSION)
        self.assertEqual(pregame_data["source_identity"], P52A_SOURCE_IDENTITY)
        self.assertEqual(len(pregame_data["predictions"]), 1)
        self.assertEqual(len(pregame_data["exclusions"]), 0)

        # Invariant: zero result contamination
        pred = pregame_data["predictions"][0]
        found_result = FORBIDDEN_PREGAME_RESULT_FIELDS.intersection(pred.keys())
        self.assertEqual(found_result, set())
        found_betting = FORBIDDEN_PREGAME_BETTING_FIELDS.intersection(pred.keys())
        self.assertEqual(found_betting, set())

        # Inspect run dir
        self.assertIsNotNone(result.run_dir)
        manifest = read_json_object(result.run_dir / "run_manifest.json")
        self.assertEqual(manifest["lifecycle_state"], "FROZEN")
        self.assertEqual(manifest["eligible_prediction_count"], 1)
        self.assertEqual(manifest["created_at_utc"], as_of_utc)
        self.assertLess(manifest["created_at_utc"], target_start)

    def test_past_cutoff_game_exclusion(self) -> None:
        """Target game whose scheduled start is <= as_of_utc must be excluded."""
        as_of_utc = "2026-08-18T23:30:00Z"
        target_date = "2026-08-18"
        target_start = "2026-08-18T23:05:00Z"  # Already started as of cutoff

        sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc=target_start,
            home_starter_id=101,
            away_starter_id=201,
        )
        mock_opener = MockMLBOpener(sched)

        result = execute_daily_moneyline_prediction_freeze(
            target_date=target_date,
            as_of_utc=as_of_utc,
            repository_root=REPOSITORY_ROOT,
            output_root=self.output_root,
            opener=mock_opener,
        )

        self.assertEqual(result.target_games_count, 1)
        self.assertEqual(result.eligible_predictions_count, 0)
        self.assertEqual(result.exclusion_count, 1)
        self.assertEqual(result.freeze_status, "NO_ELIGIBLE_PREDICTIONS")
        self.assertIsNone(result.run_id)
        self.assertEqual(result.exclusions[0]["reason"], EXCLUSION_NOT_FUTURE_GAME)

    def test_missing_starter_exclusion(self) -> None:
        """Game lacking announced probable starters must be excluded."""
        as_of_utc = "2026-08-18T13:00:00Z"
        target_date = "2026-08-18"
        target_start = "2026-08-18T23:05:00Z"

        sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc=target_start,
            home_starter_id=None,  # Missing starter
            away_starter_id=201,
        )
        mock_opener = MockMLBOpener(sched)

        result = execute_daily_moneyline_prediction_freeze(
            target_date=target_date,
            as_of_utc=as_of_utc,
            repository_root=REPOSITORY_ROOT,
            output_root=self.output_root,
            opener=mock_opener,
        )

        self.assertEqual(result.eligible_predictions_count, 0)
        self.assertEqual(result.exclusion_count, 1)
        self.assertEqual(result.exclusions[0]["reason"], EXCLUSION_STARTER_UNAVAILABLE)

    def test_insufficient_starter_history_exclusion(self) -> None:
        """Starter with fewer than 2 prior starts must be excluded."""
        as_of_utc = "2026-08-18T13:00:00Z"
        target_date = "2026-08-18"
        target_start = "2026-08-18T23:05:00Z"

        sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc=target_start,
            home_starter_id=101,
            away_starter_id=201,
        )
        p_map = {
            101: _build_mock_pitcher_log_payload(101, start_count=1),  # Only 1 start
            201: _build_mock_pitcher_log_payload(201, start_count=5),
        }
        mock_opener = MockMLBOpener(sched, p_map)

        result = execute_daily_moneyline_prediction_freeze(
            target_date=target_date,
            as_of_utc=as_of_utc,
            repository_root=REPOSITORY_ROOT,
            output_root=self.output_root,
            opener=mock_opener,
        )

        self.assertEqual(result.eligible_predictions_count, 0)
        self.assertEqual(result.exclusion_count, 1)
        self.assertTrue(result.exclusions[0]["reason"].startswith(EXCLUSION_INSUFFICIENT_STARTER_HISTORY))

    def test_insufficient_team_history_exclusion(self) -> None:
        """Team with fewer than 10 completed games before cutoff must be excluded."""
        as_of_utc = "2026-08-18T13:00:00Z"
        target_date = "2026-08-18"
        target_start = "2026-08-18T23:05:00Z"

        # Provide only 5 past games
        sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc=target_start,
            home_starter_id=101,
            away_starter_id=201,
            include_extra_past_games=5,
        )
        p_map = {
            101: _build_mock_pitcher_log_payload(101, start_count=4),
            201: _build_mock_pitcher_log_payload(201, start_count=4),
        }
        mock_opener = MockMLBOpener(sched, p_map)

        result = execute_daily_moneyline_prediction_freeze(
            target_date=target_date,
            as_of_utc=as_of_utc,
            repository_root=REPOSITORY_ROOT,
            output_root=self.output_root,
            opener=mock_opener,
        )

        self.assertEqual(result.eligible_predictions_count, 0)
        self.assertEqual(result.exclusion_count, 1)
        self.assertEqual(result.exclusions[0]["reason"], EXCLUSION_INSUFFICIENT_TEAM_HISTORY)

    def test_same_day_and_post_cutoff_starter_history_leakage_rejection(self) -> None:
        """Pitcher starts occurring on or after target_date must not be counted in ERA."""
        logs = {
            101: [
                {"date": "2026-07-01", "games_started": 1, "outs": 18, "earned_runs": 2},
                {"date": "2026-07-15", "games_started": 1, "outs": 18, "earned_runs": 2},
                # Future start on target date or later
                {"date": "2026-08-18", "games_started": 1, "outs": 27, "earned_runs": 10},
                {"date": "2026-08-19", "games_started": 1, "outs": 27, "earned_runs": 10},
            ]
        }
        era, err = compute_starter_era(101, "2026-08-18", logs)
        self.assertIsNone(err)
        # ERA should be computed from the 2 prior starts only: total_er=4, total_outs=36 -> ERA = (4*27)/36 = 3.0
        self.assertEqual(era, Decimal("3.0"))

    def test_same_day_and_post_cutoff_team_history_leakage_rejection(self) -> None:
        """Team games completed after as_of_utc or target game start must not enter win rate."""
        as_of_dt = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)
        target_start_dt = datetime(2026, 8, 18, 19, 0, 0, tzinfo=UTC)

        schedule_rows = [
            # 10 completed games before cutoff (Team 10 won 8)
            *[
                {
                    "final": True,
                    "scheduled_start_utc": f"2026-08-{i+1:02d}T19:00:00Z",
                    "home_team": {"id": 10},
                    "away_team": {"id": 30},
                    "home_score": 5 if i < 8 else 1,
                    "away_score": 2 if i < 8 else 6,
                }
                for i in range(10)
            ],
            # 10 completed games for Team 20 before cutoff (Team 20 won 5)
            *[
                {
                    "final": True,
                    "scheduled_start_utc": f"2026-08-{i+1:02d}T19:05:00Z",
                    "home_team": {"id": 20},
                    "away_team": {"id": 40},
                    "home_score": 5 if i < 5 else 1,
                    "away_score": 2 if i < 5 else 6,
                }
                for i in range(10)
            ],
            # Game completed AFTER as_of cutoff (e.g. 15:00:00Z)
            {
                "final": True,
                "scheduled_start_utc": "2026-08-18T15:00:00Z",
                "home_team": {"id": 10},
                "away_team": {"id": 30},
                "home_score": 10,
                "away_score": 0,
            },
        ]

        delta = compute_team_recent_win_rate_delta(10, 20, schedule_rows, as_of_dt, target_start_dt)
        self.assertIsNotNone(delta)
        # Home win rate: 8/10 = 0.8, Away win rate: 5/10 = 0.5 -> delta = 0.3
        self.assertEqual(delta, Decimal("0.3"))

    def test_identical_freeze_idempotency(self) -> None:
        """Running freeze twice on identical input returns RECOGNIZED_IDENTICAL."""
        as_of_utc = "2026-08-18T13:00:00Z"
        target_date = "2026-08-18"
        target_start = "2026-08-18T23:05:00Z"

        sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc=target_start,
            home_starter_id=101,
            away_starter_id=201,
        )
        p_map = {
            101: _build_mock_pitcher_log_payload(101, start_count=5),
            201: _build_mock_pitcher_log_payload(201, start_count=4),
        }
        mock_opener = MockMLBOpener(sched, p_map)

        # Run 1
        res1 = execute_daily_moneyline_prediction_freeze(
            target_date=target_date,
            as_of_utc=as_of_utc,
            repository_root=REPOSITORY_ROOT,
            output_root=self.output_root,
            opener=mock_opener,
        )
        self.assertEqual(res1.freeze_status, "CREATED")

        # Run 2 with identical inputs
        res2 = execute_daily_moneyline_prediction_freeze(
            target_date=target_date,
            as_of_utc=as_of_utc,
            repository_root=REPOSITORY_ROOT,
            output_root=self.output_root,
            opener=mock_opener,
        )
        self.assertEqual(res2.freeze_status, "RECOGNIZED_IDENTICAL")
        self.assertEqual(res2.run_id, res1.run_id)
        self.assertEqual(res2.pending_count, res1.pending_count)

    def test_empty_pitcher_stats_exclusion_regression(self) -> None:
        """MLB stats API returning empty stats list must not crash and must exclude game for insufficient starter history."""
        as_of_utc = "2026-08-19T01:00:00Z"
        target_date = "2026-08-19"
        target_start = "2026-08-19T23:05:00Z"

        sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc=target_start,
            home_starter_id=101,
            away_starter_id=201,
        )
        p_map = {
            101: {"stats": []},  # Empty stats list from MLB stats API
            201: _build_mock_pitcher_log_payload(201, start_count=5),
        }
        mock_opener = MockMLBOpener(sched, p_map)

        result = execute_daily_moneyline_prediction_freeze(
            target_date=target_date,
            as_of_utc=as_of_utc,
            repository_root=REPOSITORY_ROOT,
            output_root=self.output_root,
            opener=mock_opener,
        )

        self.assertEqual(result.target_games_count, 1)
        self.assertEqual(result.eligible_predictions_count, 0)
        self.assertEqual(result.exclusion_count, 1)
        self.assertEqual(result.freeze_status, "NO_ELIGIBLE_PREDICTIONS")
        self.assertIsNone(result.run_id)
        self.assertTrue(
            result.exclusions[0]["reason"].startswith(EXCLUSION_INSUFFICIENT_STARTER_HISTORY)
        )
        self.assertIn("(starts=0)", result.exclusions[0]["reason"])

    def test_empty_pitcher_stats_multi_game_partial_exclusion(self) -> None:
        """Empty pitcher stats on one game excludes that game but allows other eligible games to be frozen."""
        as_of_utc = "2026-08-19T01:00:00Z"
        target_date = "2026-08-19"

        # Build schedule with 2 target games
        base_sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc="2026-08-19T20:00:00Z",
            home_starter_id=101,
            away_starter_id=201,
        )
        # Add second target game
        base_sched["dates"][-1]["games"].append(
            {
                "gamePk": 999002,
                "gameNumber": 1,
                "gameDate": "2026-08-19T23:05:00Z",
                "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
                "teams": {
                    "home": {
                        "team": {"id": 10, "name": "New York Yankees", "abbreviation": "NYY"},
                        "probablePitcher": {"id": 102, "fullName": "Nestor Cortes"},
                    },
                    "away": {
                        "team": {"id": 20, "name": "Los Angeles Dodgers", "abbreviation": "LAD"},
                        "probablePitcher": {"id": 202, "fullName": "Yoshinobu Yamamoto"},
                    },
                },
            }
        )

        p_map = {
            101: {"stats": []},  # Empty stats for Game 1 home starter
            201: _build_mock_pitcher_log_payload(201, start_count=5),
            102: _build_mock_pitcher_log_payload(102, start_count=4),
            202: _build_mock_pitcher_log_payload(202, start_count=4),
        }
        mock_opener = MockMLBOpener(base_sched, p_map)

        result = execute_daily_moneyline_prediction_freeze(
            target_date=target_date,
            as_of_utc=as_of_utc,
            repository_root=REPOSITORY_ROOT,
            output_root=self.output_root,
            opener=mock_opener,
        )

        self.assertEqual(result.target_games_count, 2)
        self.assertEqual(result.eligible_predictions_count, 1)
        self.assertEqual(result.exclusion_count, 1)
        self.assertEqual(result.freeze_status, "CREATED")
        self.assertIsNotNone(result.run_id)
        self.assertEqual(result.pending_count, 1)
        self.assertEqual(result.exclusions[0]["game_pk"], 999001)
        self.assertTrue(
            result.exclusions[0]["reason"].startswith(EXCLUSION_INSUFFICIENT_STARTER_HISTORY)
        )
        self.assertEqual(result.predictions[0]["game_pk"], 999002)


if __name__ == "__main__":
    unittest.main()
