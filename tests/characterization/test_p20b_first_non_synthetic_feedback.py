"""Characterization tests for the committed P20B feedback artifact."""

import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.historical_feedback_replay_artifacts import (
    write_historical_feedback_replay_artifacts,
)
from match_analysis.application.use_cases.replay_historical_prediction_feedback import (
    replay_historical_prediction_feedback,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P20A_ROOT = REPOSITORY_ROOT / "report/p20a_p13_walk_forward_reconstruction"
RESULT_ROOT = REPOSITORY_ROOT / "data/fixtures/p20b_historical_results"


class TestP20BFirstNonSyntheticFeedback(unittest.TestCase):
    def test_committed_p20b_lineage_shape(self) -> None:
        result = replay_historical_prediction_feedback(
            p20a_predictions_bytes=(P20A_ROOT / "predictions.jsonl").read_bytes(),
            p20a_reconstruction_bytes=(P20A_ROOT / "reconstruction.json").read_bytes(),
            p20a_summary_bytes=(P20A_ROOT / "summary.json").read_bytes(),
            p20a_fold_bytes=(P20A_ROOT / "fold.json").read_bytes(),
            historical_results_bytes=(RESULT_ROOT / "final_results.jsonl").read_bytes(),
            historical_provenance_bytes=(RESULT_ROOT / "provenance.json").read_bytes(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output"
            write_historical_feedback_replay_artifacts(output_dir, result)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            feedback_rows = [
                json.loads(line)
                for line in (output_dir / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(summary["schema_version"], "p20b.first_non_synthetic_historical_feedback.v1")
        self.assertEqual(summary["feedback_row_count"], 4)
        self.assertEqual(summary["p15c_admission_count"], 4)
        self.assertEqual(summary["p16a_attachment_count"], 4)
        self.assertEqual(summary["p16b_evaluation_count"], 4)
        self.assertEqual(summary["replay_game_ids"], [
            "2025-06-01_ATL_BOS",
            "2025-06-01_TEX_STL",
        ])
        self.assertEqual(
            summary["historical_provenance"]["source_commit"],
            "03b2fcf4de1a13ee9929afcef803d61955c9f41b",
        )
        self.assertEqual(
            summary["historical_provenance"]["source_archive_path"],
            "data/mlb_2025/gl2025.zip",
        )
        self.assertEqual(
            summary["historical_provenance"]["source_member"],
            "gl2025.txt",
        )
        self.assertFalse(summary["claims"]["synthetic_results"])
        self.assertTrue(summary["claims"]["sample_limited"])
        self.assertFalse(summary["claims"]["training_dataset_claim"])
        self.assertEqual(
            [(row["provider_game_id"], row["home_score"], row["away_score"]) for row in feedback_rows],
            [
                ("2025-06-01_TEX_STL", 8, 1),
                ("2025-06-01_ATL_BOS", 1, 3),
                ("2025-06-01_TEX_STL", 8, 1),
                ("2025-06-01_ATL_BOS", 1, 3),
            ],
        )


if __name__ == "__main__":
    unittest.main()
