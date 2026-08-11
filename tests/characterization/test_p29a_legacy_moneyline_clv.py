from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPOSITORY_ROOT / "data/fixtures/p29a_legacy_moneyline_clv"
TOLERANCE = Decimal("0.000002")
PROJECTION_FIELDS = (
    "clv_record_id",
    "prediction_id",
    "canonical_match_id",
    "selection",
    "market_odds_at_prediction",
    "implied_probability_at_prediction",
    "closing_odds",
    "closing_implied_probability",
    "clv_value",
    "odds_snapshot_time_utc",
    "prediction_time_utc",
    "closing_odds_time_utc",
    "event_start_time_utc",
    "closing_odds_source",
)


class LegacyMoneylineClvCharacterizationTests(unittest.TestCase):
    def test_authority_manifest_and_all_fourteen_rows_are_present(self) -> None:
        manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (FIXTURE_ROOT / "parity_cohort.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(manifest["schema_version"], "p29a.legacy_moneyline_clv_parity_manifest.v1")
        self.assertEqual(manifest["row_count"], 14)
        self.assertEqual(manifest["legacy_artifact_sha256"], "09ea49e359558f6cc4df6c0d4dbf6dbffa8bebe9730b7581bce4900b6a9f8517")
        self.assertEqual(manifest["projection_sha256"], "b1297629f69df8cbf0715c75f7f3bd776870c24063224364d74b3f1c9b79b5e8")
        self.assertEqual(len(rows), manifest["row_count"])
        self.assertEqual([row["row_index"] for row in rows], list(range(1, 15)))

        projection = b"".join(
            (
                json.dumps(
                    {**{field: row[field] for field in PROJECTION_FIELDS}, "row_index": row["row_index"]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            for row in rows
        )
        self.assertEqual(sha256(projection).hexdigest(), manifest["projection_sha256"])

    def test_legacy_formula_and_pregame_chronology_hold_for_every_row(self) -> None:
        rows = [
            json.loads(line)
            for line in (FIXTURE_ROOT / "parity_cohort.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        selections_by_game: dict[str, set[str]] = {}
        for row in rows:
            expected = Decimal(str(row["closing_implied_probability"])) - Decimal(
                str(row["implied_probability_at_prediction"])
            )
            actual = Decimal(str(row["clv_value"]))
            self.assertLessEqual(abs(actual - expected), TOLERANCE, row["row_index"])
            self.assertLess(
                row["odds_snapshot_time_utc"],
                row["closing_odds_time_utc"],
            )
            self.assertLessEqual(
                row["closing_odds_time_utc"],
                row["event_start_time_utc"],
            )
            selections_by_game.setdefault(row["canonical_match_id"], set()).add(
                row["selection"]
            )
        self.assertTrue(all(selections == {"home", "away"} for selections in selections_by_game.values()))


if __name__ == "__main__":
    unittest.main()
