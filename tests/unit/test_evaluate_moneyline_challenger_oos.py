"""Unit tests for P23A frozen prediction and result-join behavior."""

from dataclasses import replace
from pathlib import Path
import unittest

from match_analysis.application.use_cases.evaluate_moneyline_challenger_oos import (
    _load_challenger,
    _load_feature_authority,
    _load_incumbent,
    _load_and_validate_results,
    pair_predictions_with_results,
    predict_feature_rows,
)


ROOT = Path(__file__).resolve().parents[2]


class EvaluateMoneylineChallengerOOSTests(unittest.TestCase):
    def setUp(self) -> None:
        summary, manifest, source, features = _load_feature_authority(ROOT)
        challenger, challenger_fp, challenger_projection = _load_challenger(ROOT)
        incumbent, incumbent_projection, _, _ = _load_incumbent(ROOT)
        results = _load_and_validate_results(
            ROOT,
            summary=summary,
            manifest=manifest,
            source_manifest=source,
            feature_rows=features,
        )
        self.summary = summary
        self.features = features
        self.results = results
        self.challenger = challenger
        self.challenger_fp = challenger_fp
        self.challenger_id = challenger_projection["model_id"]
        self.incumbent = incumbent
        self.incumbent_fp = incumbent_projection["artifact_fingerprint"]
        self.incumbent_id = incumbent_projection["model_id"]

    def test_input_row_order_does_not_change_prediction_stream(self) -> None:
        first = predict_feature_rows(self.features, self.challenger, self.incumbent)
        second = predict_feature_rows(
            tuple(reversed(self.features)), self.challenger, self.incumbent
        )
        self.assertEqual(first, second)

    def test_outcomes_are_joined_after_predictions_are_frozen(self) -> None:
        predictions = predict_feature_rows(self.features, self.challenger, self.incumbent)
        original_rows = pair_predictions_with_results(
            feature_rows=self.features,
            predictions=predictions,
            result_rows=self.results,
            challenger_model_id=self.challenger_id,
            challenger_model_fingerprint=self.challenger_fp,
            incumbent_model_id=self.incumbent_id,
            incumbent_model_fingerprint=self.incumbent_fp,
        )
        mutated_first = replace(self.results[0], home_score=99, away_score=0)
        mutated_rows = pair_predictions_with_results(
            feature_rows=self.features,
            predictions=predictions,
            result_rows=(mutated_first, *self.results[1:]),
            challenger_model_id=self.challenger_id,
            challenger_model_fingerprint=self.challenger_fp,
            incumbent_model_id=self.incumbent_id,
            incumbent_model_fingerprint=self.incumbent_fp,
        )
        self.assertEqual(predictions, predict_feature_rows(self.features, self.challenger, self.incumbent))
        self.assertEqual(
            (original_rows[0].challenger_home_probability, original_rows[0].incumbent_home_probability),
            (mutated_rows[0].challenger_home_probability, mutated_rows[0].incumbent_home_probability),
        )
        self.assertNotEqual(original_rows[0].target_home_win, mutated_rows[0].target_home_win)
        self.assertEqual(original_rows[0].comparison_row_id, mutated_rows[0].comparison_row_id)

    def test_missing_result_cannot_drop_a_cohort_row(self) -> None:
        predictions = predict_feature_rows(self.features, self.challenger, self.incumbent)
        with self.assertRaisesRegex(ValueError, "incomplete or mismatched"):
            pair_predictions_with_results(
                feature_rows=self.features,
                predictions=predictions,
                result_rows=self.results[:-1],
                challenger_model_id=self.challenger_id,
                challenger_model_fingerprint=self.challenger_fp,
                incumbent_model_id=self.incumbent_id,
                incumbent_model_fingerprint=self.incumbent_fp,
            )


if __name__ == "__main__":
    unittest.main()
