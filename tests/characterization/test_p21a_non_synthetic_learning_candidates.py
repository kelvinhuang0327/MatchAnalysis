"""Characterization tests for the committed P20B-backed P21A positive path."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.assess_prediction_learning_candidates import (
    assess_prediction_learning_candidates,
)
from match_analysis.application.use_cases.prediction_learning_candidate_artifacts import (
    write_prediction_learning_candidate_artifacts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P20B_ROOT = REPOSITORY_ROOT / "report/p20b_first_non_synthetic_historical_feedback"


class P21ANonSyntheticLearningCandidatesCharacterizationTests(unittest.TestCase):
    def test_committed_p20b_positive_path_and_p17a_lineage(self) -> None:
        feedback_path = P20B_ROOT / "feedback.jsonl"
        summary_path = P20B_ROOT / "summary.json"
        feedback_before = feedback_path.read_bytes()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        result = assess_prediction_learning_candidates(
            feedback_bytes=feedback_before,
            feedback_summary_bytes=summary_path.read_bytes(),
        )

        self.assertEqual(result.source_row_count, 4)
        self.assertEqual(len(result.assessments), 4)
        self.assertEqual(len(result.candidates), 4)
        self.assertEqual(
            result.source_feedback_jsonl_sha256,
            "f1912760faa0c29ef1dbd24d759d6ce8b5bcf763c0f2e71e7f9608c76dbedb2a",
        )
        self.assertEqual(
            result.source_feedback_ledger_fingerprint,
            "adf320cb91681254ed9ca79467c818406a8d64beaf0631bf61afab1bcb13e087",
        )
        self.assertEqual(
            summary["p17a_feedback_ledger_fingerprint"],
            result.source_feedback_ledger_fingerprint,
        )
        self.assertTrue(all(assessment.status == "ELIGIBLE" for assessment in result.assessments))
        self.assertFalse(result.claims["training_authorized"])
        self.assertFalse(result.claims["training_dataset_claim"])
        self.assertFalse(result.claims["model_promoted"])
        self.assertEqual(feedback_before, feedback_path.read_bytes())

    def test_artifacts_are_three_deterministic_files_with_explicit_claims(self) -> None:
        result = assess_prediction_learning_candidates(
            feedback_bytes=(P20B_ROOT / "feedback.jsonl").read_bytes(),
            feedback_summary_bytes=(P20B_ROOT / "summary.json").read_bytes(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "p21a"
            write_prediction_learning_candidate_artifacts(output_dir, result)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"assessments.jsonl", "learning_candidates.jsonl", "summary.json"},
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["eligible_count"], 4)
            self.assertEqual(summary["excluded_count"], 0)
            self.assertTrue(summary["claims"]["sample_limited"])
            self.assertFalse(summary["claims"]["profitability_claim"])
            self.assertFalse(summary["claims"]["real_betting_recommendation"])
            self.assertEqual(
                summary["assessments_jsonl_sha256"],
                hashlib.sha256((output_dir / "assessments.jsonl").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                summary["learning_candidates_jsonl_sha256"],
                hashlib.sha256((output_dir / "learning_candidates.jsonl").read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
