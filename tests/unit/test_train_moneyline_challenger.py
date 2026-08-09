"""Focused P22B challenger-training contract tests."""

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.train_moneyline_challenger import (
    P22B_DATASET_FINGERPRINT,
    P22B_DEFAULT_FIT_RUNTIME,
    P22B_FEATURE_NAMES,
    load_p22a_training_dataset,
    train_moneyline_challenger,
)


DATASET_PATH = REPOSITORY_ROOT / "report/p22a_game_level_training_dataset/training_examples.jsonl"
SUMMARY_PATH = REPOSITORY_ROOT / "report/p22a_game_level_training_dataset/summary.json"


class TrainMoneylineChallengerTests(unittest.TestCase):
    def train(self, dataset_path: Path = DATASET_PATH, summary_path: Path = SUMMARY_PATH):
        return train_moneyline_challenger(
            dataset_path,
            summary_path,
            fit_runtime=P22B_DEFAULT_FIT_RUNTIME,
            source_repository=str(REPOSITORY_ROOT),
        )

    def test_committed_dataset_has_exact_authority(self) -> None:
        dataset = load_p22a_training_dataset(DATASET_PATH, SUMMARY_PATH)
        self.assertEqual(dataset.dataset_fingerprint, P22B_DATASET_FINGERPRINT)
        self.assertEqual(len(dataset.examples), 677)
        self.assertEqual(dataset.label_distribution, {"0": 309, "1": 368})
        self.assertEqual(dataset.examples[0].feature_names, P22B_FEATURE_NAMES)
        self.assertEqual(
            len({example.provider_game_id for example in dataset.examples}),
            677,
        )

    def test_source_mutation_and_summary_mutation_are_rejected(self) -> None:
        rows = DATASET_PATH.read_text(encoding="utf-8").splitlines()
        mutated_row = json.loads(rows[0])
        mutated_row["source_schedule_observation_id"] = "0" * 64
        rows[0] = json.dumps(mutated_row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mutated_dataset = root / "training_examples.jsonl"
            mutated_summary = root / "summary.json"
            mutated_dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
            mutated_summary.write_bytes(SUMMARY_PATH.read_bytes())
            with self.assertRaises(ValueError):
                load_p22a_training_dataset(mutated_dataset, mutated_summary)

            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            summary["dataset_fingerprint"] = "0" * 64
            mutated_dataset.write_bytes(DATASET_PATH.read_bytes())
            mutated_summary.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_p22a_training_dataset(mutated_dataset, mutated_summary)

    def test_forbidden_outcome_feature_name_is_rejected(self) -> None:
        rows = DATASET_PATH.read_text(encoding="utf-8").splitlines()
        projection = json.loads(rows[0])
        projection["feature_names"] = ["recent_win_rate_delta", "historical_home_score"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "training_examples.jsonl"
            summary_path = root / "summary.json"
            dataset_path.write_text(
                json.dumps(projection, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            summary_path.write_bytes(SUMMARY_PATH.read_bytes())
            with self.assertRaises(ValueError):
                load_p22a_training_dataset(dataset_path, summary_path)

    def test_double_training_reproduces_fitted_state_and_artifact(self) -> None:
        first = self.train()
        second = self.train()
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(first.to_projection(), second.to_projection())
        self.assertEqual(
            first.to_inference_artifact().to_projection(),
            second.to_inference_artifact().to_projection(),
        )

    def test_source_row_order_does_not_change_fitted_artifact(self) -> None:
        rows = DATASET_PATH.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "training_examples.jsonl"
            summary_path = root / "summary.json"
            reordered = "\n".join(reversed(rows)) + "\n"
            dataset_path.write_text(reordered, encoding="utf-8")
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            summary["training_examples_jsonl_sha256"] = sha256(
                reordered.encode("utf-8")
            ).hexdigest()
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            original = self.train()
            reordered_artifact = self.train(dataset_path, summary_path)
            self.assertEqual(original.fingerprint(), reordered_artifact.fingerprint())
            self.assertEqual(original.to_projection(), reordered_artifact.to_projection())

    def test_artifact_claims_do_not_select_or_promote_a_model(self) -> None:
        projection = self.train().to_projection()
        self.assertEqual(projection["model_role"], "CHALLENGER")
        self.assertTrue(projection["claims"]["training_authorized"])
        self.assertTrue(projection["claims"]["training_performed"])
        for claim in (
            "model_promoted",
            "production_ready",
            "profitability_claim",
            "real_betting_recommendation",
            "out_of_sample_evaluated",
        ):
            self.assertFalse(projection["claims"][claim])
        serialized = json.dumps(projection, sort_keys=True)
        self.assertNotIn('"winner"', serialized)
        self.assertNotIn('"better_model"', serialized)


if __name__ == "__main__":
    unittest.main()
