"""Unit tests for building prediction feedback ledgers."""

from pathlib import Path
import unittest

from match_analysis.application.use_cases.build_prediction_feedback_ledger import (
    build_prediction_feedback_ledger,
)


class TestPredictionFeedbackLedger(unittest.TestCase):
    def setUp(self) -> None:
        self.p15c_dir = Path("report/p15c_admitted_prediction_observation_snapshot")
        self.p16a_dir = Path("report/p16a_final_result_attachment")
        self.p16b_dir = Path("report/p16b_prediction_evaluation_scorecard")

        self.snapshot_bytes = (self.p15c_dir / "admitted_observations.jsonl").read_bytes()
        self.snap_summary_bytes = (self.p15c_dir / "summary.json").read_bytes()
        self.attachments_bytes = (self.p16a_dir / "attachments.jsonl").read_bytes()
        self.att_summary_bytes = (self.p16a_dir / "summary.json").read_bytes()
        self.evaluations_bytes = (self.p16b_dir / "evaluations.jsonl").read_bytes()
        self.eval_summary_bytes = (self.p16b_dir / "summary.json").read_bytes()

    def test_build_feedback_ledger_success(self) -> None:
        result = build_prediction_feedback_ledger(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=self.att_summary_bytes,
            evaluations_bytes=self.evaluations_bytes,
            evaluation_summary_bytes=self.eval_summary_bytes,
        )

        self.assertEqual(result.prediction_row_count, 3)
        self.assertEqual(result.attached_row_count, 3)
        self.assertEqual(result.rejected_attachment_row_count, 0)
        self.assertEqual(result.evaluated_row_count, 3)
        self.assertEqual(result.non_evaluated_row_count, 0)
        self.assertEqual(result.correct_count, 2)
        self.assertEqual(result.incorrect_count, 1)

        self.assertEqual(result.feedback_status_counts, {"EVALUATED": 3})
        self.assertEqual(result.attachment_rejection_reason_counts, {})

        self.assertEqual(
            result.source_snapshot_fingerprint,
            "858f1740463ff8f0b5556f26489445eec74c33377ba75f2fc7e65895d7f4f1c1",
        )
        self.assertEqual(
            result.source_attachment_set_fingerprint,
            "fe6c96a18f7606869c30f8acdd21a525ab75ed3289bfc7af2c940fc3776c3772",
        )
        self.assertEqual(
            result.source_evaluation_set_fingerprint,
            "5aa49a7bbf5526ec8fd3f67a8cda03a4519edc6a484c36be4fe83e41c12fd108",
        )

        # Check safety claims
        self.assertTrue(result.claims["synthetic_results"])
        self.assertTrue(result.claims["sample_limited"])
        self.assertFalse(result.claims["real_model_performance_claim"])
        self.assertFalse(result.claims["training_dataset_claim"])
        self.assertFalse(result.claims["retraining_performed"])

    def test_snapshot_sha256_mismatch_raises(self) -> None:
        tampered = self.snapshot_bytes + b"\n"
        with self.assertRaises(ValueError) as ctx:
            build_prediction_feedback_ledger(
                snapshot_bytes=tampered,
                snapshot_summary_bytes=self.snap_summary_bytes,
                attachments_bytes=self.attachments_bytes,
                attachment_summary_bytes=self.att_summary_bytes,
                evaluations_bytes=self.evaluations_bytes,
                evaluation_summary_bytes=self.eval_summary_bytes,
            )
        self.assertIn("P16A summary source_snapshot_sha256 mismatch", str(ctx.exception))

    def test_attachments_sha256_mismatch_raises(self) -> None:
        tampered = self.attachments_bytes + b"\n"
        with self.assertRaises(ValueError) as ctx:
            build_prediction_feedback_ledger(
                snapshot_bytes=self.snapshot_bytes,
                snapshot_summary_bytes=self.snap_summary_bytes,
                attachments_bytes=tampered,
                attachment_summary_bytes=self.att_summary_bytes,
                evaluations_bytes=self.evaluations_bytes,
                evaluation_summary_bytes=self.eval_summary_bytes,
            )
        self.assertIn("P16A summary attachments_jsonl_sha256 mismatch", str(ctx.exception))

    def test_evaluations_sha256_mismatch_raises(self) -> None:
        tampered = self.evaluations_bytes + b"\n"
        with self.assertRaises(ValueError) as ctx:
            build_prediction_feedback_ledger(
                snapshot_bytes=self.snapshot_bytes,
                snapshot_summary_bytes=self.snap_summary_bytes,
                attachments_bytes=self.attachments_bytes,
                attachment_summary_bytes=self.att_summary_bytes,
                evaluations_bytes=tampered,
                evaluation_summary_bytes=self.eval_summary_bytes,
            )
        self.assertIn("P16B summary evaluations_jsonl_sha256 mismatch", str(ctx.exception))

    def test_snapshot_summary_sha256_mismatch_raises(self) -> None:
        tampered = self.snap_summary_bytes + b"\n"
        with self.assertRaises(ValueError) as ctx:
            build_prediction_feedback_ledger(
                snapshot_bytes=self.snapshot_bytes,
                snapshot_summary_bytes=tampered,
                attachments_bytes=self.attachments_bytes,
                attachment_summary_bytes=self.att_summary_bytes,
                evaluations_bytes=self.evaluations_bytes,
                evaluation_summary_bytes=self.eval_summary_bytes,
            )
        self.assertIn("P16A summary source_snapshot_summary_sha256 mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
