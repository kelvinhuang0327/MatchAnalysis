"""Characterize the committed P22A game-level dataset artifact contract."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.materialize_moneyline_training_dataset import (  # noqa: E402
    materialize_moneyline_training_dataset,
)
from match_analysis.application.use_cases.moneyline_training_dataset_artifacts import (  # noqa: E402
    render_training_examples_jsonl,
    write_moneyline_training_dataset_artifacts,
)
from match_analysis.application.use_cases.replay_multifold_historical_candidates import (  # noqa: E402
    load_multifold_folds,
    load_multifold_reconstructed_models,
)


FIXTURE_ROOT = REPOSITORY_ROOT / "data/fixtures/p21b_multifold_historical"
CANDIDATE_ROOT = REPOSITORY_ROOT / "report/p21b_contiguous_multifold_historical_candidates"


def build_dataset():
    return materialize_moneyline_training_dataset(
        candidate_bytes=(CANDIDATE_ROOT / "learning_candidates.jsonl").read_bytes(),
        candidate_summary_bytes=(CANDIDATE_ROOT / "summary.json").read_bytes(),
        folds=load_multifold_folds(
            [FIXTURE_ROOT / "fold_wf_002.json", FIXTURE_ROOT / "fold_wf_003.json"]
        ),
        historical_results_bytes=(FIXTURE_ROOT / "final_results.jsonl").read_bytes(),
        historical_provenance_bytes=(FIXTURE_ROOT / "provenance.json").read_bytes(),
        reconstructed_models=load_multifold_reconstructed_models(
            FIXTURE_ROOT / "reconstructed_models.json"
        ),
    )


class P22AGameLevelTrainingDatasetCharacterizationTests(unittest.TestCase):
    def test_repeated_artifact_render_is_byte_identical_and_claims_are_bounded(self) -> None:
        first = build_dataset()
        second = build_dataset()
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            write_moneyline_training_dataset_artifacts(first_dir, first)
            write_moneyline_training_dataset_artifacts(second_dir, second)
            first_examples = (Path(first_dir) / "training_examples.jsonl").read_bytes()
            second_examples = (Path(second_dir) / "training_examples.jsonl").read_bytes()
            first_summary = (Path(first_dir) / "summary.json").read_bytes()
            second_summary = (Path(second_dir) / "summary.json").read_bytes()
        self.assertEqual(first_examples, second_examples)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first.training_example_count, 677)
        self.assertEqual(first.candidates_collapsed_count, 677)
        self.assertEqual(first.unmapped_candidate_count, 0)
        summary = json.loads(first_summary)
        self.assertEqual(summary["training_example_count"], 677)
        self.assertEqual(summary["eligible_candidate_count"], 1354)
        self.assertEqual(summary["training_examples_jsonl_sha256"], __import__("hashlib").sha256(first_examples).hexdigest())
        self.assertTrue(summary["training_dataset_claim"])
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["retraining_performed"])
        self.assertFalse(summary["model_promoted"])
        self.assertTrue(summary["sample_limited"])
        self.assertFalse(summary["profitability_claim"])
        self.assertFalse(summary["production_ready"])
        self.assertEqual(summary["p20b_historical_runtime_compliance"], "REMAINS_REFUTED")
        self.assertEqual(
            render_training_examples_jsonl(first.examples),
            first_examples.decode("utf-8"),
        )
        self.assertNotIn("model_probability", first_examples.decode("utf-8"))
        self.assertNotIn("brier", first_examples.decode("utf-8").lower())
        self.assertNotIn("correctness", first_examples.decode("utf-8").lower())


if __name__ == "__main__":
    unittest.main()
