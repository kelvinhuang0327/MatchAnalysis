"""Focused P32A Blob3rd acquisition and P31A compatibility contracts."""

from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.acquire_tsl_moneyline_snapshot import (
    TslMoneylineSnapshotAcquisition,
    _select_target_slate,
    run_tsl_moneyline_paper_smoke,
)
from match_analysis.infrastructure.sources.tsl_moneyline_acquisition import (
    TSL_BLOB3RD_LIVE_URL,
    TSL_BLOB3RD_PRE_URL_TEMPLATE,
    TSL_BLOB3RD_SPORTS_URL,
    TslBlob3rdClient,
    TslBlob3rdRawCapture,
    build_tsl_moneyline_history,
    normalize_tsl_moneyline_games,
)
from match_analysis.infrastructure.sources.tsl_moneyline_history import (
    canonical_json_bytes,
)


FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/p32a_tsl_acquisition/tsl_pre_games_v1.json"


def _fixture_game() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))[0]


def _second_fixture_game() -> dict[str, object]:
    game = _fixture_game()
    game["id"] = "3482337.1"
    game["an"] = "匹茲堡海盜"
    game["hn"] = "邁阿密馬林魚"
    game["ms"][0]["cs"] = [
        {"name": "匹茲堡海盜", "pd": "50", "pu": "37", "hv": None},
        {"name": "邁阿密馬林魚", "pd": "25", "pu": "19", "hv": None},
    ]
    return game


def _schedule_row(
    *,
    game_id: str,
    game_pk: int,
    away_code: str,
    away_name: str,
    away_team_id: int,
    home_code: str,
    home_name: str,
    home_team_id: int,
) -> dict[str, object]:
    return {
        "schema_version": "p23f2.mlb_official_normalized.v1",
        "provider_game_id": game_id,
        "game_pk": game_pk,
        "game_number": 1,
        "official_date": "2026-08-11",
        "scheduled_start_utc": "2026-08-11T22:40:00Z",
        "status": "Scheduled",
        "final": False,
        "home_team": {"id": home_team_id, "abbreviation": home_code, "name": home_name},
        "away_team": {"id": away_team_id, "abbreviation": away_code, "name": away_name},
        "home_score": None,
        "away_score": None,
    }


class P32ATslMoneylineAcquisitionTests(unittest.TestCase):
    def test_fixture_matches_legacy_moneyline_semantics(self) -> None:
        result = normalize_tsl_moneyline_games(
            json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
            fetched_at="2026-08-11T15:00:00Z",
            target_date="2026-08-12",
        )

        self.assertEqual(result.source_row_count, 1)
        self.assertEqual(result.rejected_games, ())
        observation = result.observations[0]
        self.assertEqual(observation.source_identifier, "3482336.1")
        self.assertEqual(observation.provider_source, "TSL_BLOB3RD")
        self.assertEqual(observation.home_team, "底特律老虎")
        self.assertEqual(observation.away_team, "克里夫蘭守護者")
        self.assertEqual(observation.home_decimal_odds, "1.58")
        self.assertEqual(observation.away_decimal_odds, "1.92")
        self.assertEqual(observation.row["home_code"], "DET")
        self.assertEqual(observation.row["away_code"], "CLE")
        outcomes = observation.row["markets"][0]["outcomes"]
        self.assertEqual(
            [item["outcomeName"] for item in outcomes],
            ["克里夫蘭守護者", "底特律老虎"],
        )
        self.assertEqual(
            observation.source_row_fingerprint,
            sha256(canonical_json_bytes(observation.row)).hexdigest(),
        )

    def test_malformed_and_post_start_rows_fail_closed(self) -> None:
        post_start = _fixture_game()
        malformed = _fixture_game()
        post_start["id"] = "post-start"
        malformed["id"] = "malformed"
        malformed["ms"] = []
        post_start_result = normalize_tsl_moneyline_games(
            [post_start],
            fetched_at="2026-08-12T00:00:00Z",
            target_date="2026-08-12",
        )
        malformed_result = normalize_tsl_moneyline_games(
            [malformed],
            fetched_at="2026-08-11T15:00:00Z",
            target_date="2026-08-12",
        )

        self.assertEqual(post_start_result.observations, ())
        self.assertEqual(post_start_result.rejected_games[0].reason, "POST_START")
        self.assertEqual(malformed_result.observations, ())
        self.assertEqual(
            malformed_result.rejected_games[0].reason,
            "MALFORMED_OR_INVALID_MONEYLINE",
        )

    def test_current_tsl_team_alias_still_produces_a_crosswalk_code(self) -> None:
        game = _fixture_game()
        game["id"] = "colorado-alias"
        game["an"] = "科羅拉多落磯"
        game["ms"][0]["cs"][0]["name"] = "科羅拉多落磯"
        result = normalize_tsl_moneyline_games(
            [game],
            fetched_at="2026-08-11T15:00:00Z",
            target_date="2026-08-12",
        )

        self.assertEqual(result.observations[0].row["away_code"], "COL")

    def test_non_mlb_baseball_keeps_legacy_sport_classification(self) -> None:
        game = _fixture_game()
        game["id"] = "npb-row"
        game["an"] = "阪神虎"
        game["hn"] = "讀賣巨人"
        game["ms"][0]["cs"][0]["name"] = "阪神虎"
        game["ms"][0]["cs"][1]["name"] = "讀賣巨人"
        result = normalize_tsl_moneyline_games(
            [game],
            fetched_at="2026-08-11T15:00:00Z",
            target_date="2026-08-12",
        )

        self.assertEqual(result.observations[0].row["sport_league"], "INTL")
        self.assertEqual(result.observations[0].row["home_code"], "")
        self.assertEqual(result.observations[0].row["away_code"], "")

    def test_client_reproduces_bounded_modern_request_sequence_and_last_write_wins(self) -> None:
        first = _fixture_game()
        replacement = deepcopy(first)
        replacement["ms"][0]["cs"][0]["pu"] = "25"
        payloads = {
            TSL_BLOB3RD_LIVE_URL: [],
            TSL_BLOB3RD_SPORTS_URL: [{"id": "34731.1", "abb": "BSB"}],
            TSL_BLOB3RD_PRE_URL_TEMPLATE.format(sport_id="34731.1", language="zh"): [
                replacement
            ],
        }
        calls: list[str] = []

        def fetch(url: str) -> bytes:
            calls.append(url)
            return json.dumps(payloads[url], ensure_ascii=False).encode("utf-8")

        capture = TslBlob3rdClient(fetcher=fetch).fetch_modern_capture()

        self.assertEqual(
            calls,
            [
                TSL_BLOB3RD_LIVE_URL,
                TSL_BLOB3RD_SPORTS_URL,
                TSL_BLOB3RD_PRE_URL_TEMPLATE.format(
                    sport_id="34731.1", language="zh"
                ),
            ],
        )
        self.assertEqual(capture.games[0]["ms"][0]["cs"][0]["pu"], "25")

    def test_history_has_immutable_selected_row_fingerprint(self) -> None:
        payload = json.dumps([_fixture_game()], ensure_ascii=False).encode("utf-8")

        def fetch(url: str) -> bytes:
            if url == TSL_BLOB3RD_LIVE_URL:
                return b"[]"
            if url == TSL_BLOB3RD_SPORTS_URL:
                return b'[{"id":"34731.1","abb":"BSB"}]'
            return payload

        capture = TslBlob3rdClient(fetcher=fetch).fetch_modern_capture()
        history, normalization = build_tsl_moneyline_history(
            capture,
            fetched_at="2026-08-11T15:00:00Z",
            target_date="2026-08-12",
        )

        self.assertEqual(len(normalization.observations), 1)
        self.assertEqual(history.qualified_row_count, 1)
        self.assertEqual(history.scope_start_date, "2026-08-12")
        self.assertEqual(len(history.selected_rows_sha256), 64)
        self.assertEqual(history.raw_sha256, capture.combined_payload_sha256)

    def test_target_slate_is_selected_from_official_schedule_timing(self) -> None:
        rows = (
            {
                "provider_game_id": "1",
                "scheduled_start_utc": "2026-08-11T22:40:00Z",
                "official_date": "2026-08-11",
                "game_number": 1,
                "game_pk": 1,
            },
            {
                "provider_game_id": "2",
                "scheduled_start_utc": "2026-08-11T23:00:00Z",
                "official_date": "2026-08-11",
                "game_number": 1,
                "game_pk": 2,
            },
            {
                "provider_game_id": "3",
                "scheduled_start_utc": "2026-08-12T22:40:00Z",
                "official_date": "2026-08-12",
                "game_number": 1,
                "game_pk": 3,
            },
        )

        target_date, target_rows = _select_target_slate(
            rows,
            started_at_utc=datetime(2026, 8, 11, 15, 0, tzinfo=UTC),
        )

        self.assertEqual(target_date, "2026-08-12")
        self.assertEqual([row["provider_game_id"] for row in target_rows], ["1", "2"])

    def test_existing_p30a_consumes_fresh_rows_without_betting_semantics(self) -> None:
        games = [_fixture_game(), _second_fixture_game()]
        payload = json.dumps(games, ensure_ascii=False).encode("utf-8")
        capture = TslBlob3rdRawCapture(
            sport_id="34731.1",
            games=tuple(games),
            payloads=(("fixture", payload),),
        )
        history, normalization = build_tsl_moneyline_history(
            capture,
            fetched_at="2026-08-11T15:00:00Z",
            target_date="2026-08-12",
        )
        schedule_rows = (
            _schedule_row(
                game_id="9001",
                game_pk=9001,
                away_code="CLE",
                away_name="Cleveland Guardians",
                away_team_id=1,
                home_code="DET",
                home_name="Detroit Tigers",
                home_team_id=2,
            ),
            _schedule_row(
                game_id="9002",
                game_pk=9002,
                away_code="PIT",
                away_name="Pittsburgh Pirates",
                away_team_id=3,
                home_code="MIA",
                home_name="Miami Marlins",
                home_team_id=4,
            ),
        )
        acquisition = TslMoneylineSnapshotAcquisition(
            operation="ACQUIRE_TSL_MONEYLINE_SNAPSHOT",
            target_date="2026-08-12",
            selection_started_at_utc="2026-08-11T14:59:00Z",
            fetched_at_utc="2026-08-11T15:00:00Z",
            schedule_url="fixture",
            schedule_rows=schedule_rows,
            target_schedule_rows=schedule_rows,
            requested_game_ids=("9001", "9002"),
            history=history,
            normalization=normalization,
            source_payload_sha256=(("fixture", sha256(payload).hexdigest()),),
            runtime_capture_paths=(),
        )

        result = run_tsl_moneyline_paper_smoke(
            acquisition,
            repository_root=REPOSITORY_ROOT,
        )

        self.assertEqual(result.summary["raw_game_count"], 2)
        self.assertEqual(result.summary["feature_unavailable_count"], 2)
        self.assertEqual(result.summary["edge_available_count"], 0)
        self.assertEqual(result.summary["structural_status_counts"]["FEATURE_UNAVAILABLE"], 2)
        self.assertTrue(result.summary["paper_only"])
        self.assertFalse(result.summary["real_betting_recommendation"])


if __name__ == "__main__":
    unittest.main()
