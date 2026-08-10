"""Unit tests for P23A frozen prediction and result-join behavior."""

from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path
from unittest.mock import patch
import unittest

evaluate_module = importlib.import_module(
    "match_analysis.application.use_cases.evaluate_moneyline_challenger_oos"
)
from match_analysis.application.use_cases.evaluate_moneyline_challenger_oos import (
    P23A_INCUMBENT_SOURCE_FOLD_FINGERPRINT,
    _load_challenger,
    _load_feature_authority,
    _load_incumbent,
    _load_and_validate_results,
    evaluate_moneyline_challenger_oos,
    pair_predictions_with_results,
    predict_feature_rows,
)


ROOT = Path(__file__).resolve().parents[2]


class EvaluateMoneylineChallengerOOSTests(unittest.TestCase):
    def setUp(self) -> None:
        summary, manifest, source, features = _load_feature_authority(ROOT)
        challenger, challenger_fp, challenger_projection = _load_challenger(ROOT)
        incumbent, incumbent_projection, _, _, _ = _load_incumbent(ROOT)
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

    def test_unmodified_incumbent_source_passes_fingerprint_validation(self) -> None:
        _, _, _, _, source_fingerprint = _load_incumbent(ROOT)
        self.assertEqual(source_fingerprint, P23A_INCUMBENT_SOURCE_FOLD_FINGERPRINT)

    def test_mutated_incumbent_feature_row_fails_closed(self) -> None:
        original_read_json = evaluate_module._read_json

        def read_json(path: Path) -> dict:
            value = original_read_json(path)
            if path.name != "fold_wf_003.json":
                return value
            mutated = deepcopy(value)
            mutated["prediction_rows"][0]["features"][
                "indep_recent_win_rate_delta"
            ] = "999.0"
            return mutated

        with patch.object(evaluate_module, "_read_json", side_effect=read_json):
            with self.assertRaisesRegex(ValueError, "incumbent fold fingerprint mismatch"):
                evaluate_moneyline_challenger_oos(ROOT)

    def test_input_row_order_does_not_change_prediction_stream(self) -> None:
        first = predict_feature_rows(self.features, self.challenger, self.incumbent)
        second = predict_feature_rows(
            tuple(reversed(self.features)), self.challenger, self.incumbent
        )
        self.assertEqual(first, second)

    def test_input_row_order_does_not_change_paired_output(self) -> None:
        predictions = predict_feature_rows(self.features, self.challenger, self.incumbent)
        first = pair_predictions_with_results(
            feature_rows=self.features,
            predictions=predictions,
            result_rows=self.results,
            challenger_model_id=self.challenger_id,
            challenger_model_fingerprint=self.challenger_fp,
            incumbent_model_id=self.incumbent_id,
            incumbent_model_fingerprint=self.incumbent_fp,
        )
        second = pair_predictions_with_results(
            feature_rows=tuple(reversed(self.features)),
            predictions=predictions,
            result_rows=tuple(reversed(self.results)),
            challenger_model_id=self.challenger_id,
            challenger_model_fingerprint=self.challenger_fp,
            incumbent_model_id=self.incumbent_id,
            incumbent_model_fingerprint=self.incumbent_fp,
        )
        self.assertEqual(
            tuple(row.to_projection() for row in first),
            tuple(row.to_projection() for row in second),
        )

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
