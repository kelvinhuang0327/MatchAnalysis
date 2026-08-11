"""Unit contracts for the P30A Moneyline paper-analysis run."""

from collections import Counter
from copy import deepcopy
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
from match_analysis.application.use_cases.run_moneyline_paper_analysis import (
    P30A_STATUS_EDGE_AVAILABLE,
    P30A_STATUS_FEATURE_UNAVAILABLE,
    P30A_STATUS_PRICE_UNAVAILABLE,
    run_moneyline_paper_analysis,
)
from match_analysis.infrastructure.legacy_betting_pool.tsl_odds_history import (
    load_tsl_odds_history,
)


FIXTURE_ROOT = REPOSITORY_ROOT / "data/fixtures/p28ab_tsl_aligned_moneyline_edge"
SCHEDULE_PATH = (
    REPOSITORY_ROOT
    / "data/fixtures/p23f2_official_2026_history/normalized/schedule.jsonl"
)
MANIFEST_PATH = FIXTURE_ROOT / "source_manifest.json"
FORBIDDEN_ROW_FIELDS = {
    "away_score",
    "bankroll",
    "bet",
    "clv",
    "final",
    "home_score",
    "kelly",
    "result",
    "roi",
    "runs",
    "settlement",
    "stake",
    "winner",
}


def _inputs() -> dict[str, object]:
    tsl = load_tsl_odds_history(FIXTURE_ROOT / "tsl_odds_history.jsonl")
    return {
        "repository_root": REPOSITORY_ROOT,
        "tsl_rows": tsl.rows,
        "tsl_raw_sha256": tsl.raw_sha256,
        "schedule_rows": load_normalized_rows(SCHEDULE_PATH),
        "target_boxscore_rows": load_normalized_rows(
            FIXTURE_ROOT / "normalized/target_boxscores.jsonl"
        ),
        "pitcher_game_log_rows": load_normalized_rows(
            FIXTURE_ROOT / "normalized/pitcher_game_logs.jsonl"
        ),
        "source_manifest": json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
        "offline_replay_verified": True,
    }


class RunMoneylinePaperAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = _inputs()
        cls.result = run_moneyline_paper_analysis(**cls.inputs)

    def test_one_row_per_raw_game_and_structural_accounting(self) -> None:
        summary = self.result.summary
        self.assertEqual(summary["raw_game_count"], 16)
        self.assertEqual(summary["edge_available_count"], 9)
        self.assertEqual(summary["feature_unavailable_count"], 7)
        self.assertEqual(summary["price_unavailable_pre_cutoff_count"], 0)
        self.assertEqual(summary["crosswalk_unresolved_count"], 0)
        self.assertEqual(summary["p28ab_descriptive_edge_row_count"], 18)
        self.assertEqual(len(self.result.analysis), summary["raw_game_count"])
        self.assertEqual(
            Counter(row["structural_status"] for row in self.result.analysis),
            Counter(
                {
                    P30A_STATUS_EDGE_AVAILABLE: 9,
                    P30A_STATUS_FEATURE_UNAVAILABLE: 7,
                    P30A_STATUS_PRICE_UNAVAILABLE: 0,
                    "CROSSWALK_UNRESOLVED": 0,
                }
            ),
        )
        self.assertEqual(
            len({row["game_id"] for row in self.result.analysis}),
            summary["raw_game_count"],
        )

    def test_existing_model_price_and_edge_contracts_are_preserved(self) -> None:
        summary = self.result.summary
        self.assertEqual(
            summary["promoted_default_model_id"],
            "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
        )
        self.assertEqual(
            summary["promoted_default_model_fingerprint"],
            "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
        )
        self.assertEqual(summary["price_selection_rule"], "LATEST_PRE_CUTOFF")
        self.assertEqual(summary["normalization"], "SIMPLE_TWO_WAY_NORMALIZATION")
        self.assertTrue(summary["moneyline_model_promoted"])
        self.assertEqual(summary["moneyline_promotion_scope"], "paper_only")
        self.assertFalse(summary["decision_policy_used"])
        self.assertFalse(summary["staking_implemented"])
        self.assertFalse(summary["profitability_claim"])
        self.assertFalse(summary["real_betting_recommendation"])

        for row in self.result.analysis:
            if row["market_price_id"] is not None:
                self.assertTrue(row["market_price_id"].startswith("p28ab:"))
                total = Decimal(row["home_no_vig_probability"]) + Decimal(
                    row["away_no_vig_probability"]
                )
                self.assertAlmostEqual(float(total), 1.0, places=12)
            if row["structural_status"] == P30A_STATUS_EDGE_AVAILABLE:
                self.assertIsNotNone(row["prediction_id"])
                self.assertIsNotNone(row["model_home_probability"])
                self.assertIsNotNone(row["home_edge"])
                self.assertIsNotNone(row["away_edge"])
            else:
                self.assertIsNone(row["home_edge"])
                self.assertIsNone(row["away_edge"])

    def test_rows_are_outcome_and_downstream_isolated(self) -> None:
        for row in self.result.analysis:
            self.assertTrue(FORBIDDEN_ROW_FIELDS.isdisjoint(row))
        self.assertTrue(self.result.summary["outcome_isolation_verified"])
        self.assertFalse(self.result.summary["settlement_included"])
        self.assertFalse(self.result.summary["clv_included"])

        mutated_inputs = deepcopy(self.inputs)
        target_ids = {row["game_id"] for row in self.result.analysis}
        mutated_schedule = []
        for row in mutated_inputs["schedule_rows"]:
            changed = dict(row)
            if str(changed["provider_game_id"]) in target_ids:
                changed["home_score"] = 999
                changed["away_score"] = 0
            mutated_schedule.append(changed)
        mutated_inputs["schedule_rows"] = tuple(mutated_schedule)
        mutated = run_moneyline_paper_analysis(**mutated_inputs)
        self.assertEqual(mutated.analysis, self.result.analysis)
        self.assertEqual(mutated.summary["run_id"], self.result.summary["run_id"])

        removed_inputs = deepcopy(self.inputs)
        removed_inputs["schedule_rows"] = tuple(
            {
                key: value
                for key, value in row.items()
                if key not in {"home_score", "away_score", "runs", "result", "winner"}
            }
            if str(row["provider_game_id"]) in target_ids
            else row
            for row in self.inputs["schedule_rows"]
        )
        removed = run_moneyline_paper_analysis(**removed_inputs)
        self.assertEqual(removed.analysis, self.result.analysis)
        self.assertEqual(removed.summary["run_id"], self.result.summary["run_id"])

    def test_replay_and_input_order_are_deterministic(self) -> None:
        self.assertTrue(self.result.summary["deterministic_replay_verified"])
        reversed_inputs = deepcopy(self.inputs)
        for key in (
            "tsl_rows",
            "schedule_rows",
            "target_boxscore_rows",
            "pitcher_game_log_rows",
        ):
            reversed_inputs[key] = tuple(reversed(reversed_inputs[key]))
        reversed_result = run_moneyline_paper_analysis(**reversed_inputs)
        self.assertEqual(reversed_result.analysis, self.result.analysis)
        self.assertEqual(reversed_result.summary, self.result.summary)


if __name__ == "__main__":
    unittest.main()
