"""Unit tests for the P20B historical feedback replay use case."""

import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.historical_feedback_replay_artifacts import (
    write_historical_feedback_replay_artifacts,
)
from match_analysis.application.use_cases.prediction_feedback_artifacts import (
    render_feedback_jsonl,
)
from match_analysis.application.use_cases.replay_historical_prediction_feedback import (
    replay_historical_prediction_feedback,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P20A_ROOT = REPOSITORY_ROOT / "report/p20a_p13_walk_forward_reconstruction"
RESULT_ROOT = REPOSITORY_ROOT / "data/fixtures/p20b_historical_results"


class TestReplayHistoricalPredictionFeedback(unittest.TestCase):
    def _input_bytes(self) -> dict[str, bytes]:
        return {
            "p20a_predictions_bytes": (P20A_ROOT / "predictions.jsonl").read_bytes(),
            "p20a_reconstruction_bytes": (P20A_ROOT / "reconstruction.json").read_bytes(),
            "p20a_summary_bytes": (P20A_ROOT / "summary.json").read_bytes(),
            "p20a_fold_bytes": (P20A_ROOT / "fold.json").read_bytes(),
            "historical_results_bytes": (RESULT_ROOT / "final_results.jsonl").read_bytes(),
            "historical_provenance_bytes": (RESULT_ROOT / "provenance.json").read_bytes(),
        }

    def test_replay_reuses_all_existing_contract_stages(self) -> None:
        inputs = self._input_bytes()
        result = replay_historical_prediction_feedback(**inputs)

        self.assertEqual(
            result.replay_game_ids,
            ("2025-06-01_ATL_BOS", "2025-06-01_TEX_STL"),
        )
        self.assertEqual(len(result.admission_workflow.results), 4)
        self.assertEqual(result.attachment_result.attached_count, 4)
        self.assertEqual(result.evaluation_result.evaluation_row_count, 4)
        self.assertEqual(result.feedback_result.prediction_row_count, 4)
        self.assertEqual(result.feedback_result.correct_count, 2)
        self.assertEqual(result.feedback_result.incorrect_count, 2)

        self.assertTrue(result.claims["synthetic_results"] is False)
        self.assertTrue(result.claims["sample_limited"])
        self.assertTrue(result.claims["historical"])
        self.assertTrue(result.claims["non_synthetic"])
        for claim in (
            "training_dataset_claim",
            "training_authorized",
            "retraining_performed",
            "model_promoted",
            "profitability_claim",
            "real_betting_recommendation",
        ):
            self.assertFalse(result.claims[claim], claim)

        provenance = result.historical_provenance
        self.assertEqual(
            provenance.source_commit,
            "03b2fcf4de1a13ee9929afcef803d61955c9f41b",
        )
        self.assertEqual(provenance.source_member, "gl2025.txt")
        self.assertEqual(
            [(row["canonical_game_id"], row["away_score"], row["home_score"]) for row in provenance.rows],
            [
                ("2025-06-01_ATL_BOS", 3, 1),
                ("2025-06-01_TEX_STL", 1, 8),
            ],
        )

    def test_replay_does_not_mutate_p20a_inputs_or_candidate_payloads(self) -> None:
        inputs = self._input_bytes()
        original_inputs = dict(inputs)
        original_candidates = {
            row["prediction_observation_id"]: row
            for row in (
                json.loads(line)
                for line in inputs["p20a_predictions_bytes"].decode("utf-8").splitlines()
                if line.strip()
            )
        }

        result = replay_historical_prediction_feedback(**inputs)

        self.assertEqual(inputs, original_inputs)
        feedback_rows = [
            json.loads(line)
            for line in render_feedback_jsonl(result.feedback_result).splitlines()
            if line.strip()
        ]
        self.assertEqual(len(feedback_rows), 4)
        for feedback_row in feedback_rows:
            observation = feedback_row["observation_payload"]
            source = original_candidates[observation["prediction_observation_id"]]
            for key in (
                "prediction_observation_id",
                "source_prediction_id",
                "model_id",
                "market_id",
                "selection",
                "model_probability",
                "provider_game_id",
                "source_schedule_observation_id",
            ):
                self.assertEqual(observation[key], source[key], key)

    def test_swapped_historical_result_identity_fails_closed(self) -> None:
        inputs = self._input_bytes()
        rows = [
            json.loads(line)
            for line in inputs["historical_results_bytes"].decode("utf-8").splitlines()
            if line.strip()
        ]
        rows[0]["provider_game_id"], rows[1]["provider_game_id"] = (
            rows[1]["provider_game_id"],
            rows[0]["provider_game_id"],
        )
        inputs["historical_results_bytes"] = (
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
        )

        with self.assertRaises(ValueError):
            replay_historical_prediction_feedback(**inputs)

    def test_replay_artifacts_are_byte_deterministic(self) -> None:
        result = replay_historical_prediction_feedback(**self._input_bytes())

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_one = Path(tmp_dir) / "one"
            output_two = Path(tmp_dir) / "two"
            write_historical_feedback_replay_artifacts(output_one, result)
            write_historical_feedback_replay_artifacts(output_two, result)

            for name in ("feedback.jsonl", "summary.json", "report.md"):
                self.assertEqual(
                    (output_one / name).read_bytes(),
                    (output_two / name).read_bytes(),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
