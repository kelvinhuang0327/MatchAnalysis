"""Tests for the allowlisted MLB official source normalizers."""

from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.infrastructure.providers.mlb_official_historical_source import (
    normalize_boxscore_payload,
    normalize_pitcher_game_log_payload,
    normalize_schedule_payload,
)


class MlbOfficialHistoricalSourceTests(unittest.TestCase):
    def test_schedule_preserves_exact_provider_game_type_and_requires_it(self) -> None:
        game = {
            "gamePk": 11,
            "gameNumber": 1,
            "gameType": "P",
            "officialDate": "2026-10-01",
            "gameDate": "2026-10-01T23:00:00Z",
            "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
            "teams": {
                "home": {"team": {"id": 1, "name": "Home", "abbreviation": "H"}},
                "away": {"team": {"id": 2, "name": "Away", "abbreviation": "A"}},
            },
        }
        payload = {"dates": [{"date": "2026-10-01", "games": [game]}]}

        self.assertEqual(normalize_schedule_payload(payload)[0]["game_type"], "P")
        with self.assertRaisesRegex(ValueError, "provider gameType"):
            normalize_schedule_payload(
                {"dates": [{"date": "2026-10-01", "games": [{k: v for k, v in game.items() if k != "gameType"}]}]}
            )

    def test_pitcher_game_log_preserves_exact_type_and_requires_it(self) -> None:
        split = {
            "date": "2026-10-01",
            "gameType": "P",
            "game": {"gamePk": 11, "gameNumber": 1},
            "stat": {"gamesStarted": 1, "outs": 18, "earnedRuns": 2},
        }
        payload = {"stats": [{"splits": [split]}]}

        self.assertEqual(
            normalize_pitcher_game_log_payload(payload, player_id=5)[0]["game_type"],
            "P",
        )
        with self.assertRaisesRegex(ValueError, "provider gameType"):
            normalize_pitcher_game_log_payload(
                {"stats": [{"splits": [{k: v for k, v in split.items() if k != "gameType"}]}]},
                player_id=5,
            )

    def test_starter_team_identity_falls_back_to_boxscore_team(self) -> None:
        game = {
            "provider_game_id": "1",
            "game_pk": 1,
            "game_number": 1,
            "official_date": "2026-06-08",
            "scheduled_start_utc": "2026-06-08T18:00:00Z",
            "home_team": {"id": 2, "name": "Home", "abbreviation": "H"},
            "away_team": {"id": 3, "name": "Away", "abbreviation": "A"},
        }
        def team(team_id: int, player_id: int, name: str) -> dict[str, object]:
            return {
                "team": {"id": team_id, "name": name},
                "pitchers": [player_id],
                "players": {
                    f"ID{player_id}": {
                        "person": {"id": player_id, "fullName": name + " Starter"},
                        "stats": {"pitching": {"gamesStarted": 1}},
                    }
                },
            }
        row = normalize_boxscore_payload(
            {"teams": {"home": team(2, 20, "Home"), "away": team(3, 30, "Away")}},
            game=game,
        )
        self.assertEqual(row["home_starter"]["team_id"], 2)
        self.assertEqual(row["away_starter"]["player_id"], 30)


if __name__ == "__main__":
    unittest.main()
