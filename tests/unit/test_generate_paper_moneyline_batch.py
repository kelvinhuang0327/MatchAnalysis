"""Unit contracts for the P24C promoted-default paper batch."""

from collections import Counter
import json
from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.generate_paper_moneyline_batch import (
    P24CWindow,
    generate_paper_moneyline_batch,
    load_p24c_source_inputs,
    resolve_p24c_window,
)
from match_analysis.application.use_cases.paper_moneyline_batch_artifacts import (
    P22B_ARTIFACT_FINGERPRINT,
    P22B_MODEL_ID,
)


RAW_ROOT = REPOSITORY_ROOT / "data/fixtures/p24c_promoted_moneyline_shadow_batch/raw"
NORMALIZED_ROOT = REPOSITORY_ROOT / "data/fixtures/p24c_promoted_moneyline_shadow_batch/normalized"
SOURCE_MANIFEST = REPOSITORY_ROOT / "report/p24c_promoted_moneyline_shadow_batch/source_manifest.json"


def _load_batch_inputs():
    window = resolve_p24c_window(REPOSITORY_ROOT)
    return window, load_p24c_source_inputs(
        repository_root=REPOSITORY_ROOT,
        raw_root=RAW_ROOT,
        normalized_root=NORMALIZED_ROOT,
        source_manifest_path=SOURCE_MANIFEST,
        window=window,
    )


class GeneratePaperMoneylineBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.window, cls.inputs = _load_batch_inputs()
        schedule, boxes, logs, manifest = cls.inputs
        cls.result = generate_paper_moneyline_batch(
            repository_root=REPOSITORY_ROOT,
            schedule_rows=schedule,
            target_boxscore_rows=boxes,
            pitcher_game_log_rows=logs,
            source_manifest=manifest,
            offline_replay_verified=True,
        )

    def test_window_and_counts_are_the_promoted_wf_007_slice(self) -> None:
        self.assertEqual(
            self.window,
            P24CWindow(fold_id="wf_007", start_date="2026-06-14", end_date="2026-06-20"),
        )
        self.assertEqual(self.result.summary["raw_game_count"], 90)
        self.assertEqual(self.result.summary["evaluable_game_count"], 79)
        self.assertEqual(self.result.summary["feature_unavailable_count"], 11)
        self.assertEqual(
            self.result.summary["evaluable_game_count"]
            + self.result.summary["feature_unavailable_count"],
            self.result.summary["raw_game_count"],
        )

    def test_promoted_default_and_paper_only_claims_are_explicit(self) -> None:
        summary = self.result.summary
        self.assertEqual(summary["promoted_default_model_id"], P22B_MODEL_ID)
        self.assertEqual(summary["promoted_default_model_fingerprint"], P22B_ARTIFACT_FINGERPRINT)
        self.assertTrue(summary["default_explicit_equivalence_verified"])
        self.assertTrue(summary["explicit_incumbent_override_verified"])
        self.assertTrue(summary["result_mutation_isolation_verified"])
        self.assertTrue(summary["offline_replay_verified"])
        self.assertTrue(summary["historical_shadow"])
        self.assertTrue(summary["paper_only"])
        self.assertTrue(summary["model_promoted"])
        self.assertEqual(summary["promotion_scope"], "paper_only")
        self.assertFalse(summary["challenger_retrained"])
        self.assertFalse(summary["production_ready"])
        self.assertFalse(summary["deployment_performed"])
        self.assertFalse(summary["real_betting_recommendation"])
        self.assertFalse(summary["profitability_claim"])

    def test_predictions_are_default_model_rows_and_abstentions_have_no_fallback(self) -> None:
        predictions = self.result.predictions
        unavailable = self.result.feature_unavailable
        self.assertEqual(len(predictions), 79)
        self.assertEqual(len(unavailable), 11)
        prediction_ids = {row["game_id"] for row in predictions}
        unavailable_ids = {row["game_id"] for row in unavailable}
        self.assertTrue(prediction_ids.isdisjoint(unavailable_ids))
        self.assertEqual(len(prediction_ids | unavailable_ids), 90)
        self.assertEqual(
            Counter(row["source_provider_game_id"] for row in predictions + unavailable)["824912"],
            2,
        )
        self.assertTrue(all(row["model_id"] == P22B_MODEL_ID for row in predictions))
        self.assertTrue(all(row["model_fingerprint"] == P22B_ARTIFACT_FINGERPRINT for row in predictions))
        self.assertTrue(all(row["inference_mode"] == "PAPER_DEFAULT" for row in predictions))
        self.assertTrue(all(row["status"] == "FEATURE_UNAVAILABLE" for row in unavailable))
        self.assertTrue(all(row["eligibility"] == "FEATURE_UNAVAILABLE" for row in unavailable))

    def test_outcome_fields_are_not_in_canonical_ledgers(self) -> None:
        outcome_fields = {"home_score", "away_score", "outcome", "result", "runs"}
        for row in self.result.predictions + self.result.feature_unavailable:
            self.assertTrue(outcome_fields.isdisjoint(row))

    def test_input_order_does_not_change_batch_or_ledgers(self) -> None:
        schedule, boxes, logs, manifest = self.inputs
        reversed_result = generate_paper_moneyline_batch(
            repository_root=REPOSITORY_ROOT,
            schedule_rows=tuple(reversed(schedule)),
            target_boxscore_rows=tuple(reversed(boxes)),
            pitcher_game_log_rows=tuple(reversed(logs)),
            source_manifest=json.loads(json.dumps(manifest)),
            offline_replay_verified=True,
        )
        self.assertEqual(reversed_result.predictions, self.result.predictions)
        self.assertEqual(reversed_result.feature_unavailable, self.result.feature_unavailable)
        self.assertEqual(reversed_result.summary, self.result.summary)


if __name__ == "__main__":
    unittest.main()
