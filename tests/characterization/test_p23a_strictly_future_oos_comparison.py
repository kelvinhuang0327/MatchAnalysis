"""Characterization of the committed-authority P23A comparison."""

import json
from pathlib import Path
import unittest

from match_analysis.application.use_cases.evaluate_moneyline_challenger_oos import (
    run_deterministic_moneyline_challenger_oos,
)
from match_analysis.application.use_cases.evaluate_multifold_moneyline_oos import (
    run_deterministic_multifold_moneyline_oos,
)


ROOT = Path(__file__).resolve().parents[2]


class P23AStrictlyFutureOOSComparisonTests(unittest.TestCase):
    def test_committed_authority_produces_complete_descriptive_comparison(self) -> None:
        result = run_deterministic_moneyline_challenger_oos(ROOT)
        self.assertEqual(len(result.rows), 23)
        self.assertEqual(result.summary["fold_id"], "wf_004")
        self.assertEqual(result.summary["incumbent_source_fold_id"], "wf_003")
        self.assertEqual(result.summary["incumbent_training_cutoff"], "2025-07-31")
        self.assertEqual(result.summary["incumbent_training_row_count"], 1212)
        self.assertEqual(
            result.summary["challenger_model_fingerprint"],
            "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
        )
        self.assertEqual(result.summary["challenger_mean_brier"], "0.2138127101219392124690037207")
        self.assertEqual(result.summary["incumbent_mean_brier"], "0.2323417604920021959488142985")
        self.assertEqual(result.summary["brier_delta"], "-0.0185290503700629834798105778")
        self.assertEqual(result.summary["challenger_accuracy"], "0.7826086956521739130434782609")
        self.assertEqual(result.summary["incumbent_accuracy"], "0.6956521739130434782608695652")
        self.assertEqual(result.summary["accuracy_delta"], "0.0869565217391304347826086957")
        self.assertEqual(result.summary["challenger_brier_better_count"], 18)
        self.assertEqual(result.summary["incumbent_brier_better_count"], 5)
        self.assertEqual(result.summary["equal_brier_count"], 0)
        self.assertTrue(result.summary["strict_future_boundary_verified"])
        self.assertTrue(result.summary["pit_safe_feature_reconstruction_verified"])
        self.assertTrue(result.summary["no_training_overlap_verified"])
        self.assertTrue(result.summary["challenger_frozen"])
        self.assertTrue(result.summary["outcome_isolation_verified"])
        self.assertTrue(result.summary["deterministic_replay_verified"])

    def test_summary_contains_only_descriptive_no_promotion_claims(self) -> None:
        result = run_deterministic_moneyline_challenger_oos(ROOT)
        for key in (
            "out_of_sample_evaluated",
            "evaluation_complete",
            "model_promoted",
            "promotion_authorized",
            "production_ready",
            "profitability_claim",
            "real_betting_recommendation",
            "retraining_performed",
        ):
            self.assertIn(key, result.summary)
        self.assertFalse(result.summary["model_promoted"])
        self.assertFalse(result.summary["promotion_authorized"])
        self.assertFalse(result.summary["production_ready"])
        self.assertNotIn("recommended_model", result.summary)
        self.assertNotIn("promote", result.summary)

    def test_authority_report_has_expected_future_cohort(self) -> None:
        summary = json.loads(
            (ROOT / "report/p23f2_official_future_fold/summary.json").read_text()
        )
        self.assertEqual(summary["fold_id"], "wf_004")
        self.assertEqual(summary["game_count"], 23)
        self.assertEqual(summary["validation_start"], "2026-06-08")
        self.assertEqual(summary["validation_end"], "2026-06-09")
        self.assertTrue(summary["strict_future"])

    def test_p23b_preserves_raw_games_and_evaluates_only_shared_cohort(self) -> None:
        result = run_deterministic_multifold_moneyline_oos(ROOT)
        self.assertEqual(result.summary["total_raw_game_count"], 75)
        self.assertEqual(result.summary["total_evaluable_game_count"], 65)
        self.assertEqual(result.summary["total_feature_unavailable_count"], 10)
        self.assertEqual(result.summary["pooled_evaluation_coverage"], "0.8666666666666666666666666667")
        wf005 = next(row for row in result.per_fold_summary if row["fold_id"] == "wf_005")
        self.assertEqual(
            (wf005["raw_game_count"], wf005["evaluable_game_count"], wf005["feature_unavailable_count"]),
            (22, 17, 5),
        )
        unavailable = next(
            row for row in wf005["feature_unavailable"] if row["game_id"] == "824266"
        )
        self.assertEqual(unavailable["status"], "FEATURE_UNAVAILABLE")
        self.assertEqual(unavailable["reason"], "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY")
        self.assertEqual(unavailable["affected_starters"][0]["starter_id"], 702474)
        self.assertNotIn("824266", {row["provider_game_id"] for row in result.comparison_rows})
        self.assertTrue(result.summary["deterministic_replay_verified"])
        self.assertTrue(result.summary["input_order_invariance_verified"])


if __name__ == "__main__":
    unittest.main()
