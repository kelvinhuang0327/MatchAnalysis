"""Characterization tests for committed P17A prediction feedback ledger artifacts."""

import hashlib
import json
from pathlib import Path
import unittest

from match_analysis.application.use_cases.build_prediction_feedback_ledger import (
    build_prediction_feedback_ledger,
)
from match_analysis.application.use_cases.prediction_feedback_artifacts import (
    render_feedback_jsonl,
    render_feedback_report_markdown,
    render_feedback_summary_json,
)


class TestPredictionFeedbackLedgerDemo(unittest.TestCase):
    def setUp(self) -> None:
        self.report_dir = Path("report/p17a_prediction_feedback_ledger")
        self.feedback_jsonl = self.report_dir / "feedback.jsonl"
        self.summary_json = self.report_dir / "summary.json"
        self.report_md = self.report_dir / "report.md"

        self.p15c_dir = Path("report/p15c_admitted_prediction_observation_snapshot")
        self.p16a_dir = Path("report/p16a_final_result_attachment")
        self.p16b_dir = Path("report/p16b_prediction_evaluation_scorecard")

        self.snapshot_bytes = (self.p15c_dir / "admitted_observations.jsonl").read_bytes()
        self.snap_summary_bytes = (self.p15c_dir / "summary.json").read_bytes()
        self.attachments_bytes = (self.p16a_dir / "attachments.jsonl").read_bytes()
        self.att_summary_bytes = (self.p16a_dir / "summary.json").read_bytes()
        self.evaluations_bytes = (self.p16b_dir / "evaluations.jsonl").read_bytes()
        self.eval_summary_bytes = (self.p16b_dir / "summary.json").read_bytes()

    def test_committed_artifacts_exist(self) -> None:
        self.assertTrue(self.feedback_jsonl.exists())
        self.assertTrue(self.summary_json.exists())
        self.assertTrue(self.report_md.exists())

    def test_committed_artifacts_match_recomputation_byte_for_byte(self) -> None:
        result = build_prediction_feedback_ledger(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=self.att_summary_bytes,
            evaluations_bytes=self.evaluations_bytes,
            evaluation_summary_bytes=self.eval_summary_bytes,
        )

        recomputed_feedback = render_feedback_jsonl(result)
        feedback_sha256 = hashlib.sha256(
            recomputed_feedback.encode("utf-8")
        ).hexdigest()

        recomputed_report = render_feedback_report_markdown(result)
        report_sha256 = hashlib.sha256(
            recomputed_report.encode("utf-8")
        ).hexdigest()

        recomputed_summary = render_feedback_summary_json(
            result, feedback_sha256, report_sha256
        )

        self.assertEqual(
            self.feedback_jsonl.read_text(encoding="utf-8"),
            recomputed_feedback,
        )
        self.assertEqual(
            self.report_md.read_text(encoding="utf-8"),
            recomputed_report,
        )
        self.assertEqual(
            self.summary_json.read_text(encoding="utf-8"),
            recomputed_summary,
        )

    def test_source_row_shuffle_determinism_p15c(self) -> None:
        lines = [line for line in self.snapshot_bytes.decode("utf-8").splitlines() if line.strip()]
        shuffled_bytes = ("\n".join(reversed(lines)) + "\n").encode("utf-8")

        snap_sum_dict = json.loads(self.snap_summary_bytes.decode("utf-8"))
        snap_sum_dict["admitted_observations_jsonl_sha256"] = hashlib.sha256(shuffled_bytes).hexdigest()
        shuffled_snap_sum_bytes = json.dumps(snap_sum_dict, indent=2, sort_keys=True).encode("utf-8")

        att_sum_dict = json.loads(self.att_summary_bytes.decode("utf-8"))
        att_sum_dict["source_snapshot_sha256"] = hashlib.sha256(shuffled_bytes).hexdigest()
        att_sum_dict["source_snapshot_summary_sha256"] = hashlib.sha256(shuffled_snap_sum_bytes).hexdigest()
        shuffled_att_sum_bytes = json.dumps(att_sum_dict, indent=2, sort_keys=True).encode("utf-8")

        eval_sum_dict = json.loads(self.eval_summary_bytes.decode("utf-8"))
        eval_sum_dict["source_summary_sha256"] = hashlib.sha256(shuffled_att_sum_bytes).hexdigest()
        shuffled_eval_sum_bytes = json.dumps(eval_sum_dict, indent=2, sort_keys=True).encode("utf-8")

        result1 = build_prediction_feedback_ledger(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=self.att_summary_bytes,
            evaluations_bytes=self.evaluations_bytes,
            evaluation_summary_bytes=self.eval_summary_bytes,
        )

        result2 = build_prediction_feedback_ledger(
            snapshot_bytes=shuffled_bytes,
            snapshot_summary_bytes=shuffled_snap_sum_bytes,
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=shuffled_att_sum_bytes,
            evaluations_bytes=self.evaluations_bytes,
            evaluation_summary_bytes=shuffled_eval_sum_bytes,
        )

        self.assertEqual(result1.feedback_ledger_fingerprint, result2.feedback_ledger_fingerprint)
        self.assertEqual(result1.feedback_rows, result2.feedback_rows)
        self.assertEqual(result1.evaluated_row_count, result2.evaluated_row_count)

    def test_source_row_shuffle_determinism_p16a(self) -> None:
        lines = [line for line in self.attachments_bytes.decode("utf-8").splitlines() if line.strip()]
        shuffled_bytes = ("\n".join(reversed(lines)) + "\n").encode("utf-8")

        att_sum_dict = json.loads(self.att_summary_bytes.decode("utf-8"))
        att_sum_dict["attachments_jsonl_sha256"] = hashlib.sha256(shuffled_bytes).hexdigest()
        shuffled_att_sum_bytes = json.dumps(att_sum_dict, indent=2, sort_keys=True).encode("utf-8")

        eval_sum_dict = json.loads(self.eval_summary_bytes.decode("utf-8"))
        eval_sum_dict["source_attachments_sha256"] = hashlib.sha256(shuffled_bytes).hexdigest()
        eval_sum_dict["source_summary_sha256"] = hashlib.sha256(shuffled_att_sum_bytes).hexdigest()
        shuffled_eval_sum_bytes = json.dumps(eval_sum_dict, indent=2, sort_keys=True).encode("utf-8")

        result1 = build_prediction_feedback_ledger(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=self.att_summary_bytes,
            evaluations_bytes=self.evaluations_bytes,
            evaluation_summary_bytes=self.eval_summary_bytes,
        )

        result2 = build_prediction_feedback_ledger(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
            attachments_bytes=shuffled_bytes,
            attachment_summary_bytes=shuffled_att_sum_bytes,
            evaluations_bytes=self.evaluations_bytes,
            evaluation_summary_bytes=shuffled_eval_sum_bytes,
        )

        self.assertEqual(result1.feedback_ledger_fingerprint, result2.feedback_ledger_fingerprint)
        self.assertEqual(result1.feedback_rows, result2.feedback_rows)
        self.assertEqual(result1.evaluated_row_count, result2.evaluated_row_count)

    def test_source_row_shuffle_determinism_p16b(self) -> None:
        lines = [line for line in self.evaluations_bytes.decode("utf-8").splitlines() if line.strip()]
        shuffled_bytes = ("\n".join(reversed(lines)) + "\n").encode("utf-8")

        eval_sum_dict = json.loads(self.eval_summary_bytes.decode("utf-8"))
        eval_sum_dict["evaluations_jsonl_sha256"] = hashlib.sha256(shuffled_bytes).hexdigest()
        shuffled_eval_sum_bytes = json.dumps(eval_sum_dict, indent=2, sort_keys=True).encode("utf-8")

        result1 = build_prediction_feedback_ledger(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=self.att_summary_bytes,
            evaluations_bytes=self.evaluations_bytes,
            evaluation_summary_bytes=self.eval_summary_bytes,
        )

        result2 = build_prediction_feedback_ledger(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=self.att_summary_bytes,
            evaluations_bytes=shuffled_bytes,
            evaluation_summary_bytes=shuffled_eval_sum_bytes,
        )

        self.assertEqual(result1.feedback_ledger_fingerprint, result2.feedback_ledger_fingerprint)
        self.assertEqual(result1.feedback_rows, result2.feedback_rows)
        self.assertEqual(result1.evaluated_row_count, result2.evaluated_row_count)


if __name__ == "__main__":
    unittest.main()
