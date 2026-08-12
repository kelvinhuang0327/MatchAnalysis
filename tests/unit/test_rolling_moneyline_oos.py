"""Focused P37A rolling walk-forward Moneyline OOS tests."""

from pathlib import Path
import unittest
from unittest.mock import patch

from match_analysis.application.use_cases import rolling_moneyline_oos
from match_analysis.application.use_cases.rolling_moneyline_oos import (
    run_deterministic_rolling_moneyline_oos,
)


ROOT = Path(__file__).resolve().parents[2]
FIT_RUNTIME = "/usr/bin/python3"


class RollingMoneylineOOSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_deterministic_rolling_moneyline_oos(
            ROOT,
            fit_runtime=FIT_RUNTIME,
        )
        cls.summary = cls.result.summary

    def test_candidate_windows_are_chronological_and_accounted_for(self) -> None:
        windows = self.summary["evaluation_windows"]
        self.assertEqual(
            [window["holdout_fold_id"] for window in windows],
            ["wf_004", "wf_005", "wf_006"],
        )
        self.assertEqual(
            [window["train_fold_ids"] for window in windows],
            [
                ["wf_002", "wf_003"],
                ["wf_002", "wf_003", "wf_004"],
                ["wf_002", "wf_003", "wf_004", "wf_005"],
            ],
        )
        self.assertEqual(
            [window["training"]["eligible_row_count"] for window in windows],
            [677, 700, 717],
        )
        self.assertEqual(
            [
                (
                    window["holdout"]["date_range"],
                    window["holdout"]["raw_row_count"],
                    window["holdout"]["evaluable_row_count"],
                    window["holdout"]["excluded_row_count"],
                )
                for window in windows
            ],
            [
                (["2026-06-08", "2026-06-09"], 23, 23, 0),
                (["2026-06-10", "2026-06-11"], 22, 17, 5),
                (["2026-06-12", "2026-06-13"], 30, 25, 5),
            ],
        )

    def test_each_window_has_paired_holdout_and_true_oos_lineage(self) -> None:
        for window in self.summary["evaluation_windows"]:
            comparison = window["comparison"]
            self.assertTrue(comparison["train_holdout_disjoint_verified"])
            self.assertTrue(comparison["strict_train_before_holdout_verified"])
            self.assertTrue(comparison["point_in_time_features_verified"])
            self.assertTrue(comparison["same_holdout_verified"])
            self.assertTrue(comparison["outcome_isolation_verified"])
            self.assertEqual(
                comparison["same_holdout"]["row_count"],
                window["holdout"]["evaluable_row_count"],
            )
            self.assertEqual(
                comparison["same_holdout"]["champion_row_ids"],
                comparison["same_holdout"]["challenger_row_ids"],
            )

        self.assertEqual(len(self.result.comparison_rows), 65)
        self.assertEqual(
            len({row["provider_game_id"] for row in self.result.comparison_rows}),
            65,
        )
        for row in self.result.comparison_rows:
            self.assertTrue(row["true_oos_verified"])
            self.assertNotIn("home_score", row)
            self.assertNotIn("away_score", row)

    def test_aggregate_metrics_and_conclusion_do_not_claim_promotion(self) -> None:
        aggregate = self.summary["aggregate"]
        self.assertEqual(aggregate["raw_row_count"], 75)
        self.assertEqual(aggregate["evaluable_row_count"], 65)
        self.assertEqual(aggregate["excluded_row_count"], 10)
        self.assertEqual(aggregate["champion"]["metrics"]["row_count"], 65)
        self.assertEqual(aggregate["challenger"]["metrics"]["row_count"], 65)
        self.assertIn(
            self.summary["comparison"]["aggregate_verdict"],
            {"CHALLENGER_BETTER", "CHAMPION_RETAINS", "INCONCLUSIVE"},
        )
        self.assertEqual(
            self.summary["comparison"]["conclusion"],
            "MIXED_OR_INCONCLUSIVE",
        )
        self.assertFalse(self.summary["claims"]["model_promoted"])
        self.assertFalse(self.summary["claims"]["promotion_authorized"])
        self.assertFalse(self.summary["claims"]["production_ready"])
        self.assertFalse(self.summary["verification"]["model_promotion_occurred"])

    def test_determinism_and_p36a_baseline_preservation_are_verified(self) -> None:
        self.assertTrue(self.summary["deterministic_rerun_verified"])
        self.assertTrue(self.summary["verification"]["input_order_invariance_verified"])
        self.assertTrue(self.summary["p36a_baseline_preserved"]["unchanged_verified"])
        baseline = self.summary["p36a_baseline_preserved"]
        self.assertEqual(baseline["training_fold_id"], "wf_004")
        self.assertEqual(baseline["holdout_fold_ids"], ["wf_005", "wf_006"])
        self.assertEqual(baseline["holdout_evaluable_row_count"], 42)
        self.assertEqual(len(self.result.model_artifacts), 3)

    def test_future_targets_are_consumed_only_after_each_window_prediction(self) -> None:
        authorities = rolling_moneyline_oos._load_authoritative_future_folds(ROOT)
        game_to_fold = {
            game_id: fold_id
            for fold_id, authority in authorities.items()
            for game_id in authority.raw_game_ids
        }
        events: list[tuple[str, str]] = []
        original_target = rolling_moneyline_oos._target_from_result
        original_predict = rolling_moneyline_oos.predict_feature_rows

        def traced_target(result):
            events.append(("target", game_to_fold[result.provider_game_id]))
            return original_target(result)

        def traced_predict(feature_rows, challenger, champion):
            folds = {game_to_fold[row.provider_game_id] for row in feature_rows}
            self.assertEqual(len(folds), 1)
            events.append(("predict", next(iter(folds))))
            return original_predict(feature_rows, challenger, champion)

        with patch.object(
            rolling_moneyline_oos,
            "_target_from_result",
            side_effect=traced_target,
        ), patch.object(
            rolling_moneyline_oos,
            "predict_feature_rows",
            side_effect=traced_predict,
        ):
            rolling_moneyline_oos.evaluate_rolling_moneyline_oos(
                ROOT,
                fit_runtime=FIT_RUNTIME,
            )

        prediction_seen: set[str] = set()
        target_folds: set[str] = set()
        for event_kind, fold_id in events:
            if event_kind == "predict":
                prediction_seen.add(fold_id)
            else:
                self.assertIn(fold_id, prediction_seen)
                target_folds.add(fold_id)
        self.assertEqual(target_folds, {"wf_004", "wf_005"})


if __name__ == "__main__":
    unittest.main()
