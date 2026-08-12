"""Focused P36A offline retraining and temporal-split tests."""

from pathlib import Path
import unittest

from match_analysis.application.use_cases.offline_moneyline_retraining_baseline import (
    P36A_HOLDOUT_START_DATE,
    _load_training_authority,
    run_deterministic_offline_moneyline_retraining_baseline,
)


ROOT = Path(__file__).resolve().parents[2]


class OfflineMoneylineRetrainingBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_deterministic_offline_moneyline_retraining_baseline(ROOT)
        cls.summary = cls.result.summary

    def test_temporal_training_authority_is_date_batched_before_holdout(self) -> None:
        _dataset, observations, _metadata = _load_training_authority(ROOT)
        self.assertEqual(max(row.official_date for row in observations), "2026-06-09")
        self.assertLess(
            max(row.official_date for row in observations),
            P36A_HOLDOUT_START_DATE,
        )
        self.assertEqual(self.summary["training"]["split_granularity"], "OFFICIAL_DATE_BATCH")
        self.assertTrue(self.summary["verification"]["same_date_batch_isolation_verified"])

    def test_training_and_holdout_counts_and_exclusions_are_truthful(self) -> None:
        self.assertEqual(self.summary["training"]["eligible_row_count"], 700)
        self.assertEqual(self.summary["training"]["excluded_row_count"], 0)
        self.assertEqual(self.summary["holdout"]["raw_row_count"], 52)
        self.assertEqual(self.summary["holdout"]["evaluable_row_count"], 42)
        self.assertEqual(self.summary["holdout"]["excluded_row_count"], 10)
        self.assertEqual(
            self.summary["holdout"]["excluded_reasons"],
            {"INSUFFICIENT_SAME_SEASON_STARTER_HISTORY": 10},
        )

    def test_champion_and_challenger_use_the_exact_same_holdout(self) -> None:
        comparison = self.summary["comparison"]
        self.assertTrue(comparison["same_holdout_verified"])
        self.assertEqual(comparison["same_holdout"]["row_count"], 42)
        self.assertEqual(
            comparison["same_holdout"]["champion_row_ids"],
            comparison["same_holdout"]["challenger_row_ids"],
        )
        self.assertEqual(len(self.result.comparison_rows), 42)

    def test_quality_metrics_and_verdict_are_actual_comparison_values(self) -> None:
        champion = self.summary["champion"]["metrics"]
        challenger = self.summary["challenger"]["metrics"]
        self.assertEqual(champion["row_count"], 42)
        self.assertEqual(challenger["row_count"], 42)
        self.assertLess(
            float(challenger["brier_score"]),
            float(champion["brier_score"]),
        )
        self.assertLess(
            float(challenger["log_loss"]),
            float(champion["log_loss"]),
        )
        self.assertGreater(
            float(challenger["accuracy"]),
            float(champion["accuracy"]),
        )
        self.assertEqual(self.summary["comparison_verdict"], "CHALLENGER_BETTER")
        self.assertIn("calibration", challenger)
        self.assertIn("expected_calibration_error", challenger["calibration"])

    def test_deterministic_rerun_and_outcome_isolation_pass(self) -> None:
        self.assertTrue(self.summary["deterministic_rerun_verified"])
        self.assertTrue(self.summary["verification"]["point_in_time_features_verified"])
        self.assertTrue(self.summary["verification"]["target_outcome_isolation_verified"])
        self.assertTrue(self.summary["verification"]["holdout_not_used_for_fit_verified"])

    def test_no_model_promotion_claim_is_emitted(self) -> None:
        self.assertEqual(self.summary["champion"]["model_role"], "CHAMPION")
        self.assertEqual(self.summary["challenger"]["model_role"], "CHALLENGER")
        self.assertFalse(self.summary["claims"]["model_promoted"])
        self.assertFalse(self.summary["claims"]["promotion_authorized"])
        self.assertFalse(self.summary["claims"]["production_ready"])
        self.assertFalse(self.summary["verification"]["model_promotion_occurred"])

    def test_comparison_rows_do_not_put_outcomes_in_feature_payload(self) -> None:
        for row in self.result.comparison_rows:
            self.assertNotIn("home_score", row)
            self.assertNotIn("away_score", row)
            self.assertIn("target_home_win", row)
            self.assertIn("challenger_log_loss_contribution", row)
            self.assertIn("champion_log_loss_contribution", row)


if __name__ == "__main__":
    unittest.main()
