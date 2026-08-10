"""Tests for the allowlisted MLB official source normalizers."""

from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.infrastructure.providers.mlb_official_historical_source import (
    normalize_boxscore_payload,
)


class MlbOfficialHistoricalSourceTests(unittest.TestCase):
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
