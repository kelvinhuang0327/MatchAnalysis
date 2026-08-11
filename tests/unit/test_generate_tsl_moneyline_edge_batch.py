"""Contract tests for the P28AB TSL-aligned Moneyline edge slice."""

from decimal import Decimal
import json
from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.acquire_future_moneyline_history import (
    load_normalized_rows,
)
from match_analysis.application.use_cases.generate_tsl_moneyline_edge_batch import (
    P28AB_TSL_BLOB_ID,
    P28AB_TSL_RAW_SHA256,
    generate_tsl_moneyline_edge_batch,
)
from match_analysis.infrastructure.legacy_betting_pool.tsl_odds_history import (
    load_tsl_odds_history,
)


FIXTURE_ROOT = REPOSITORY_ROOT / "data/fixtures/p28ab_tsl_aligned_moneyline_edge"
TSL_PATH = FIXTURE_ROOT / "tsl_odds_history.jsonl"
SCHEDULE_PATH = REPOSITORY_ROOT / "data/fixtures/p23f2_official_2026_history/normalized/schedule.jsonl"
BOX_PATH = FIXTURE_ROOT / "normalized/target_boxscores.jsonl"
LOG_PATH = FIXTURE_ROOT / "normalized/pitcher_game_logs.jsonl"
MANIFEST_PATH = FIXTURE_ROOT / "source_manifest.json"


def _inputs():
    tsl = load_tsl_odds_history(TSL_PATH)
    return (
        tsl,
        load_normalized_rows(SCHEDULE_PATH),
        load_normalized_rows(BOX_PATH),
        load_normalized_rows(LOG_PATH),
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
    )


class GenerateTslMoneylineEdgeBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tsl, cls.schedule, cls.boxes, cls.logs, cls.manifest = _inputs()
        cls.result = generate_tsl_moneyline_edge_batch(
            repository_root=REPOSITORY_ROOT,
            tsl_rows=cls.tsl.rows,
            tsl_raw_sha256=cls.tsl.raw_sha256,
            schedule_rows=cls.schedule,
            target_boxscore_rows=cls.boxes,
            pitcher_game_log_rows=cls.logs,
            source_manifest=cls.manifest,
            offline_replay_verified=True,
        )

    def test_exact_authority_and_pinned_model_are_explicit(self) -> None:
        authority = self.result.source_manifest["tsl_authority"]
        self.assertEqual(authority["authority_label"], "TSL_BLOB3RD")
        self.assertEqual(authority["blob_id"], P28AB_TSL_BLOB_ID)
        self.assertEqual(authority["raw_sha256"], P28AB_TSL_RAW_SHA256)
        self.assertEqual(
            self.result.summary["promoted_default_model_id"],
            "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
        )
        self.assertTrue(self.result.summary["historical_shadow"])
        self.assertTrue(self.result.summary["paper_only"])
        self.assertFalse(self.result.summary["production_ready"])
        self.assertFalse(self.result.summary["real_betting_recommendation"])
        self.assertFalse(self.result.summary["profitability_claim"])

    def test_raw_crosswalk_price_and_edge_accounting(self) -> None:
        summary = self.result.summary
        self.assertEqual(summary["cohort_start_date"], "2026-05-18")
        self.assertEqual(summary["cohort_end_date"], "2026-05-24")
        self.assertEqual(summary["raw_source_row_count"], 46)
        self.assertEqual(summary["crosswalked_source_row_count"], 38)
        self.assertEqual(summary["crosswalked_official_game_count"], 17)
        self.assertEqual(summary["final_official_game_count"], 16)
        self.assertEqual(summary["selected_price_count"], 16)
        self.assertEqual(summary["raw_game_count"], 16)
        self.assertEqual(summary["evaluable_game_count"], 9)
        self.assertEqual(summary["feature_unavailable_count"], 7)
        self.assertEqual(summary["edge_row_count"], 18)
        self.assertEqual(
            summary["crosswalk_status_counts"],
            {
                "MATCHED_FINAL": 31,
                "NO_CANONICAL_TEAM_CODE": 8,
                "POSTPONED_OR_NON_FINAL": 7,
            },
        )

        prices_by_game = {row["game_id"]: row for row in self.result.prices}
        self.assertEqual(
            prices_by_game["823862"]["price_fetched_at"],
            "2026-05-23T09:47:26.944342Z",
        )
        for row in self.result.prices:
            self.assertLess(row["price_fetched_at"], row["prediction_cutoff_utc"])
            self.assertEqual(row["price_selection_rule"], "LATEST_PRE_CUTOFF")
            total = Decimal(row["home_normalized_implied_probability"]) + Decimal(
                row["away_normalized_implied_probability"]
            )
            self.assertAlmostEqual(float(total), 1.0, places=12)

    def test_edge_rows_are_descriptive_and_arithmetically_closed(self) -> None:
        predictions = {row["game_id"]: row for row in self.result.predictions}
        for row in self.result.edges:
            prediction = predictions[row["game_id"]]
            model_key = (
                "home_win_probability" if row["selection"] == "HOME" else "away_win_probability"
            )
            expected = Decimal(prediction[model_key]) - Decimal(
                row["normalized_implied_probability"]
            )
            self.assertEqual(Decimal(row["edge"]), expected)
            self.assertEqual(row["normalization"], "SIMPLE_TWO_WAY_NORMALIZATION")
            self.assertTrue(row["descriptive_only"])
            self.assertEqual(
                row["edge_semantics"],
                "DESCRIPTIVE_MODEL_MINUS_SIMPLE_TWO_WAY_NORMALIZED_IMPLIED_PROBABILITY",
            )

    def test_ledgers_are_outcome_blind_and_mutation_checks_pass(self) -> None:
        forbidden = {"home_score", "away_score", "winner", "result", "runs"}
        for ledger in (
            self.result.raw_cohort,
            self.result.prices,
            self.result.predictions,
            self.result.edges,
            self.result.feature_unavailable,
        ):
            for row in ledger:
                self.assertTrue(forbidden.isdisjoint(row))
        self.assertTrue(self.result.summary["outcome_blind_feature_generation"])
        self.assertTrue(self.result.summary["result_mutation_isolation_verified"])
        self.assertTrue(self.result.summary["price_mutation_isolation_verified"])
        self.assertTrue(self.result.summary["offline_replay_verified"])

    def test_input_order_does_not_change_deterministic_replay(self) -> None:
        reversed_result = generate_tsl_moneyline_edge_batch(
            repository_root=REPOSITORY_ROOT,
            tsl_rows=tuple(reversed(self.tsl.rows)),
            tsl_raw_sha256=self.tsl.raw_sha256,
            schedule_rows=tuple(reversed(self.schedule)),
            target_boxscore_rows=tuple(reversed(self.boxes)),
            pitcher_game_log_rows=tuple(reversed(self.logs)),
            source_manifest=json.loads(json.dumps(self.manifest)),
            offline_replay_verified=True,
        )
        self.assertEqual(reversed_result.raw_cohort, self.result.raw_cohort)
        self.assertEqual(reversed_result.prices, self.result.prices)
        self.assertEqual(reversed_result.predictions, self.result.predictions)
        self.assertEqual(reversed_result.edges, self.result.edges)
        self.assertEqual(reversed_result.feature_unavailable, self.result.feature_unavailable)
        self.assertEqual(reversed_result.summary, self.result.summary)


if __name__ == "__main__":
    unittest.main()
