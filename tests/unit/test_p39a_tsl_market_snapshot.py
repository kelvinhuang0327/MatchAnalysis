"""Focused contracts for the P39A read-only TSL source adapter."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from decimal import Decimal

from match_analysis.infrastructure.sources.p39a_tsl_market_snapshot import (
    P39A_STOP_SOURCE_HASH_MISMATCH,
    P39A_TSL_SOURCE_LABEL,
    load_tsl_market_source,
)


def _row(
    *,
    fetched_at: str | None = "2026-06-08T21:00:00Z",
    is_pregame: object = True,
    markets: list[dict[str, object]] | None = None,
    home_code: str = "BAL",
    away_code: str = "SEA",
    game_time: str = "2026-06-08T22:35:00Z",
    match_id: str = "3473130.1",
) -> dict[str, object]:
    if markets is None:
        markets = [
            {
                "marketCode": "MNL",
                "outcomes": [
                    {"outcomeName": "西雅圖水手", "odds": "1.56"},
                    {"outcomeName": "巴爾的摩金鶯", "odds": "1.94"},
                ],
            }
        ]
    row: dict[str, object] = {
        "source": P39A_TSL_SOURCE_LABEL,
        "fetched_at": fetched_at,
        "match_id": match_id,
        "game_time": game_time,
        "home_team_name": "巴爾的摩金鶯",
        "away_team_name": "西雅圖水手",
        "home_code": home_code,
        "away_code": away_code,
        "is_pregame": is_pregame,
        "markets": markets,
    }
    if fetched_at is None:
        row.pop("fetched_at")
    return row


def _write_source(rows: list[dict[str, object]]) -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "tsl_odds_history.jsonl"
    raw = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return temporary, path, sha256(raw).hexdigest()


class P39ATslMarketSnapshotTests(unittest.TestCase):
    def test_reads_only_exact_target_keys_and_normalizes_prices(self) -> None:
        target_key = ("BAL", "SEA", "2026-06-08T22:35:00Z")
        unrelated = _row(
            home_code="CLE",
            away_code="NYY",
            game_time="2026-06-09T01:10:00Z",
            match_id="3473373.1",
        )
        temporary, path, expected_hash = _write_source([_row(), unrelated])
        self.addCleanup(temporary.cleanup)

        result = load_tsl_market_source(
            path,
            expected_sha256=expected_hash,
            target_source_keys={target_key},
        )

        self.assertEqual(result.raw_sha256, expected_hash)
        self.assertEqual(result.source_row_count, 2)
        self.assertEqual(result.scoped_source_row_count, 1)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.market_status, "VALID_PREGAME")
        self.assertEqual(candidate.home_decimal_price, Decimal("1.94"))
        self.assertEqual(candidate.away_decimal_price, Decimal("1.56"))
        self.assertEqual(candidate.market_observed_at_utc, "2026-06-08T21:00:00Z")
        self.assertEqual(candidate.local_fetched_at_utc, candidate.market_observed_at_utc)
        self.assertIsNone(candidate.provider_observed_at_utc)

    def test_hash_mismatch_fails_closed(self) -> None:
        temporary, path, _expected_hash = _write_source([_row()])
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(ValueError, P39A_STOP_SOURCE_HASH_MISMATCH):
            load_tsl_market_source(
                path,
                expected_sha256="0" * 64,
                target_source_keys={("BAL", "SEA", "2026-06-08T22:35:00Z")},
            )

    def test_timestamp_and_market_failures_are_retained_as_rejections(self) -> None:
        post_start = _row(fetched_at="2026-06-08T22:35:00Z", match_id="post")
        missing_timestamp = _row(fetched_at=None, match_id="missing")
        malformed = _row(
            match_id="malformed",
            markets=[
                {
                    "marketCode": "MNL",
                    "outcomes": [
                        {"outcomeName": "西雅圖水手", "odds": "1.56"},
                        {"outcomeName": "巴爾的摩金鶯", "odds": "bad"},
                    ],
                }
            ],
        )
        rows = [post_start, missing_timestamp, malformed]
        temporary, path, expected_hash = _write_source(rows)
        self.addCleanup(temporary.cleanup)

        result = load_tsl_market_source(
            path,
            expected_sha256=expected_hash,
            target_source_keys={("BAL", "SEA", "2026-06-08T22:35:00Z")},
        )

        self.assertEqual(
            {
                candidate.source_match_id: candidate.market_status
                for candidate in result.candidates
            },
            {
                "post": "POST_START",
                "missing": "MISSING_OR_UNTRUSTED_TIMESTAMP",
                "malformed": "MALFORMED_OR_INCOMPLETE_PRICE",
            },
        )


if __name__ == "__main__":
    unittest.main()
