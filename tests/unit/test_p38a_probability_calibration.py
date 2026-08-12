"""Focused P38A probability calibration and lineage tests."""

from decimal import Decimal
from pathlib import Path
import unittest

from match_analysis.application.use_cases.moneyline_probability_calibration import (
    fit_platt_calibrator,
    probability_metrics,
)
from match_analysis.application.use_cases.p38a_probability_calibration import (
    P38A_TARGET_FOLD_IDS,
    evaluate_p38a_probability_calibration,
    run_deterministic_p38a_probability_calibration,
)


ROOT = Path(__file__).resolve().parents[2]


class ProbabilityCalibrationMathTests(unittest.TestCase):
    def test_metrics_match_known_probability_values(self) -> None:
        metrics = probability_metrics(
            (Decimal("0.25"), Decimal("0.75")),
            (0, 1),
            raw_row_count=2,
        )
        self.assertEqual(metrics["row_count"], 2)
        self.assertEqual(metrics["accuracy"], "1")
        self.assertEqual(metrics["brier_score"], "0.0625")
        self.assertEqual(metrics["coverage"], "1")
        self.assertEqual(
            Decimal(metrics["calibration"]["expected_calibration_error"]),
            Decimal("0.25"),
        )

    def test_fixed_platt_fit_is_deterministic_and_bounded(self) -> None:
        probabilities = (
            Decimal("0.2"),
            Decimal("0.3"),
            Decimal("0.7"),
            Decimal("0.8"),
        )
        targets = (0, 0, 1, 1)
        first = fit_platt_calibrator(probabilities, targets)
        second = fit_platt_calibrator(probabilities, targets)
        self.assertEqual(first.to_projection(), second.to_projection())
        self.assertEqual(first.method, "PLATT_LOGISTIC_RAW_PROBABILITY_LOGIT")
        for probability in (Decimal("0.000001"), Decimal("0.2"), Decimal("0.5"), Decimal("0.999999")):
            calibrated = first.apply(probability)
            self.assertGreater(calibrated, Decimal("0"))
            self.assertLess(calibrated, Decimal("1"))

    def test_fit_requires_both_target_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "both target classes"):
            fit_platt_calibrator(
                (Decimal("0.4"), Decimal("0.6")),
                (0, 0),
            )

    def test_metrics_reject_probability_endpoints_for_log_loss(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly between"):
            probability_metrics(
                (Decimal("0"), Decimal("1")),
                (0, 1),
                raw_row_count=2,
            )


class P38AAuthorityAndLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = evaluate_p38a_probability_calibration(ROOT)
        cls.summary = cls.result["summary"]

    def test_two_target_windows_preserve_p37a_accounting(self) -> None:
        self.assertEqual(
            self.summary["admitted_target_holdout_fold_ids"],
            list(P38A_TARGET_FOLD_IDS),
        )
        self.assertEqual(
            (
                self.summary["aggregate"]["raw_row_count"],
                self.summary["aggregate"]["evaluable_row_count"],
                self.summary["aggregate"]["excluded_row_count"],
            ),
            (52, 42, 10),
        )
        self.assertTrue(self.summary["verification"]["minimum_two_windows_verified"])
        self.assertTrue(
            self.summary["verification"]["p37a_exclusion_semantics_preserved"]
        )
        self.assertFalse(self.summary["claims"]["model_promoted"])

    def test_calibration_lineage_is_strictly_prior_and_disjoint(self) -> None:
        windows = self.summary["evaluation_windows"]
        self.assertEqual(
            [window["holdout_fold_id"] for window in windows],
            ["wf_005", "wf_006"],
        )
        self.assertEqual(
            [window["calibration"]["source_fold_ids"] for window in windows],
            [["wf_004"], ["wf_004", "wf_005"]],
        )
        self.assertEqual(
            [window["calibration"]["source_row_count"] for window in windows],
            [23, 40],
        )
        for window in windows:
            calibration = window["calibration"]
            self.assertTrue(calibration["strict_source_before_target_verified"])
            self.assertTrue(calibration["source_target_game_id_disjoint_verified"])
            self.assertTrue(
                calibration["calibrator_fit_rows_evaluation_rows_disjoint_verified"]
            )
            self.assertFalse(calibration["target_holdout_labels_used_for_fit"])
            self.assertFalse(calibration["method_selection_uses_target_labels"])
            self.assertLess(
                calibration["source_max_start_utc"],
                calibration["target_holdout_min_start_utc"],
            )
            self.assertIn("prediction_rule", window["comparison"])

    def test_same_target_rows_and_metrics_are_recorded(self) -> None:
        self.assertEqual(len(self.result["comparison_rows"]), 42)
        for window in self.summary["evaluation_windows"]:
            self.assertTrue(window["comparison"]["same_target_rows_verified"])
            self.assertIn("prediction_rule", window["comparison"])
            self.assertIn("accuracy_change_explanation", window["comparison"])
            self.assertEqual(
                window["raw_challenger"]["metrics"]["row_count"],
                window["calibrated_challenger"]["metrics"]["row_count"],
            )
            self.assertEqual(
                window["champion"]["metrics"]["row_count"],
                window["calibrated_challenger"]["metrics"]["row_count"],
            )
        for row in self.result["comparison_rows"]:
            self.assertTrue(row["same_target_row_verified"])
            self.assertTrue(row["true_oos_verified"])
            self.assertFalse(row["calibration_fitted_on_target_row"])
            self.assertNotIn("home_score", row)
            self.assertNotIn("away_score", row)

    def test_conclusion_is_one_fixed_descriptive_value(self) -> None:
        self.assertEqual(
            self.summary["comparison"]["conclusion"],
            "CALIBRATION_NOT_IMPROVED",
        )
        self.assertIn(
            self.summary["comparison"]["conclusion"],
            {
                "CALIBRATION_IMPROVED",
                "CALIBRATION_NOT_IMPROVED",
                "MIXED_OR_INCONCLUSIVE",
            },
        )
        self.assertFalse(
            self.summary["calibration"]["method_search_performed"]
        )
        self.assertFalse(
            self.summary["calibration"]["new_third_party_dependency_added"]
        )

    def test_deterministic_rerun_and_p37a_hashes_are_verified(self) -> None:
        result = run_deterministic_p38a_probability_calibration(ROOT)
        self.assertTrue(result["summary"]["deterministic_rerun_verified"])
        self.assertTrue(
            result["summary"]["verification"]["deterministic_rerun_verified"]
        )
        self.assertEqual(
            set(result["summary"]["authority"]["p37a_artifact_sha256"]),
            {
                "model_artifacts.json",
                "comparisons.jsonl",
                "per_window_summary.json",
                "summary.json",
                "report.md",
            },
        )
        self.assertTrue(
            result["summary"]["verification"]["p37a_authority_unchanged_verified"]
        )


if __name__ == "__main__":
    unittest.main()
