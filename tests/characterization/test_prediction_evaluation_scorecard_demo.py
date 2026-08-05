"""Characterization tests for committed P16B prediction evaluation scorecard artifacts."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.build_prediction_evaluation_scorecard import (
    build_prediction_evaluation_scorecard,
)
from match_analysis.application.use_cases.prediction_evaluation_artifacts import (
    render_evaluation_summary_json,
    render_evaluations_jsonl,
    render_evaluation_report_markdown,
)


class TestPredictionEvaluationScorecardDemo(unittest.TestCase):
    def setUp(self) -> None:
        self.report_dir = Path("report/p16b_prediction_evaluation_scorecard")
        self.evaluations_jsonl = self.report_dir / "evaluations.jsonl"
        self.summary_json = self.report_dir / "summary.json"
        self.report_md = self.report_dir / "report.md"

        self.p16a_dir = Path("report/p16a_final_result_attachment")
        self.p15c_dir = Path("report/p15c_admitted_prediction_observation_snapshot")

        self.attachments_bytes = (self.p16a_dir / "attachments.jsonl").read_bytes()
        self.att_summary_bytes = (self.p16a_dir / "summary.json").read_bytes()
        self.snapshot_bytes = (self.p15c_dir / "admitted_observations.jsonl").read_bytes()
        self.snap_summary_bytes = (self.p15c_dir / "summary.json").read_bytes()

    def test_committed_artifacts_exist(self) -> None:
        self.assertTrue(self.evaluations_jsonl.exists())
        self.assertTrue(self.summary_json.exists())
        self.assertTrue(self.report_md.exists())

    def test_committed_artifacts_match_recomputation_byte_for_byte(self) -> None:
        result = build_prediction_evaluation_scorecard(
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=self.att_summary_bytes,
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
        )

        recomputed_evaluations = render_evaluations_jsonl(result)
        evaluations_sha256 = hashlib.sha256(
            recomputed_evaluations.encode("utf-8")
        ).hexdigest()

        recomputed_report = render_evaluation_report_markdown(result)
        report_sha256 = hashlib.sha256(
            recomputed_report.encode("utf-8")
        ).hexdigest()

        recomputed_summary = render_evaluation_summary_json(
            result, evaluations_sha256, report_sha256
        )

        self.assertEqual(
            self.evaluations_jsonl.read_text(encoding="utf-8"),
            recomputed_evaluations,
        )
        self.assertEqual(
            self.report_md.read_text(encoding="utf-8"),
            recomputed_report,
        )
        self.assertEqual(
            self.summary_json.read_text(encoding="utf-8"),
            recomputed_summary,
        )

    def test_source_row_shuffle_determinism(self) -> None:
        lines = [line for line in self.attachments_bytes.decode("utf-8").splitlines() if line.strip()]
        # Reverse attachment lines
        shuffled_lines = list(reversed(lines))
        shuffled_attachments_bytes = ("\n".join(shuffled_lines) + "\n").encode("utf-8")

        # Note: Summary has attachments_jsonl_sha256 which depends on byte order.
        # But if we update attachments_jsonl_sha256 in summary, the result evaluation-set fingerprint,
        # evaluation row order, and aggregate metrics must be IDENTICAL.

        summary_dict = json.loads(self.att_summary_bytes.decode("utf-8"))
        summary_dict["attachments_jsonl_sha256"] = hashlib.sha256(shuffled_attachments_bytes).hexdigest()
        shuffled_summary_bytes = json.dumps(summary_dict, indent=2, sort_keys=True).encode("utf-8")

        result1 = build_prediction_evaluation_scorecard(
            attachments_bytes=self.attachments_bytes,
            attachment_summary_bytes=self.att_summary_bytes,
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
        )

        result2 = build_prediction_evaluation_scorecard(
            attachments_bytes=shuffled_attachments_bytes,
            attachment_summary_bytes=shuffled_summary_bytes,
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.snap_summary_bytes,
        )

        self.assertEqual(result1.evaluation_set_fingerprint, result2.evaluation_set_fingerprint)
        self.assertEqual(result1.evaluation_rows, result2.evaluation_rows)
        self.assertEqual(result1.accuracy, result2.accuracy)
        self.assertEqual(result1.brier_score, result2.brier_score)


if __name__ == "__main__":
    unittest.main()
