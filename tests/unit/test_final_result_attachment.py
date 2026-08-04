"""Unit tests for deterministic final result attachment use case."""

import json
from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.attach_final_results_to_admitted_predictions import (
    attach_final_results_to_admitted_predictions,
)
from match_analysis.application.use_cases.build_admitted_prediction_observation_snapshot import (
    build_admitted_prediction_observation_snapshot,
)
from match_analysis.application.use_cases.admitted_prediction_observation_artifacts import (
    render_admitted_observations_jsonl,
    render_snapshot_summary_json,
    render_snapshot_report_markdown,
)


def _create_mock_p15c_snapshot() -> tuple[bytes, bytes]:
    """Create a minimal valid P15C snapshot and summary bytes for testing."""
    p15b1_results = (
        '{"admission_status":"ADMITTED","observation":{"game_number":1,'
        '"ingested_at_utc":"2026-04-05T11:00:02Z","line_value":"-110","market_id":"moneyline",'
        '"model_id":"m1","model_probability":"0.58","prediction_generated_at_utc":"2026-04-05T11:00:00Z",'
        '"prediction_observation_id":"0cabf8e0dbc4a79013bad2c8287ea7fe2ef91ee8ae9d8574629656c25166e917",'
        '"provider_game_id":"888001","provider_namespace":"MLB_STATS_API","push_policy":"PUSH_VOID",'
        '"response_received_at_utc":"2026-04-05T11:00:01Z","scheduled_start_utc":"2026-04-05T19:00:00Z",'
        '"selection":"HOME","source_prediction_id":"P1","source_schedule_observation_id":"S1"},'
        '"request_index":0}\n'
    )

    # Compute expected result set fingerprint for single admitted row
    import hashlib
    res_set_fp = hashlib.sha256(
        b"ADMITTED::0cabf8e0dbc4a79013bad2c8287ea7fe2ef91ee8ae9d8574629656c25166e917\n"
    ).hexdigest()

    p15b1_summary = json.dumps({
        "admitted_count": 1,
        "rejected_count": 0,
        "request_count": 1,
        "result_set_fingerprint": res_set_fp,
        "claims": {"legacy_rows_admitted": False},
    }).encode("utf-8")

    res = build_admitted_prediction_observation_snapshot(
        results_bytes=p15b1_results.encode("utf-8"),
        summary_bytes=p15b1_summary,
    )

    snapshot_jsonl = render_admitted_observations_jsonl(res).encode("utf-8")
    snapshot_jsonl_sha256 = hashlib.sha256(snapshot_jsonl).hexdigest()

    # Report dummy sha256
    report_md = render_snapshot_report_markdown(res).encode("utf-8")
    report_sha256 = hashlib.sha256(report_md).hexdigest()

    summary_json = render_snapshot_summary_json(
        res, snapshot_jsonl_sha256, report_sha256
    ).encode("utf-8")

    return snapshot_jsonl, summary_json


class FinalResultAttachmentUseCasesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot_bytes, self.summary_bytes = _create_mock_p15c_snapshot()

    def test_successful_attachment(self) -> None:
        final_results = (
            '{"source_result_id":"R1","provider_namespace":"MLB_STATS_API",'
            '"provider_game_id":"888001","game_number":1,"status":"FINAL",'
            '"result_observed_at_utc":"2026-04-05T22:00:00Z","home_score":5,"away_score":3}\n'
        ).encode("utf-8")

        res = attach_final_results_to_admitted_predictions(
            snapshot_bytes=self.snapshot_bytes,
            summary_bytes=self.summary_bytes,
            final_results_bytes=final_results,
        )

        self.assertEqual(res.attached_count, 1)
        self.assertEqual(res.rejected_count, 0)
        self.assertEqual(res.correct_count, 1)
        self.assertEqual(res.incorrect_count, 0)
        self.assertEqual(res.descriptive_accuracy, 1.0)
        row = res.attachment_rows[0]
        self.assertEqual(row.attachment_status, "ATTACHED")
        self.assertEqual(row.actual_winner, "HOME")
        self.assertTrue(row.is_correct)
        self.assertIsNone(row.rejection_reason)

    def test_missing_final_result_observation(self) -> None:
        # Result has different game_number (2 instead of 1)
        final_results = (
            '{"source_result_id":"R1","provider_namespace":"MLB_STATS_API",'
            '"provider_game_id":"888001","game_number":2,"status":"FINAL",'
            '"result_observed_at_utc":"2026-04-05T22:00:00Z","home_score":5,"away_score":3}\n'
        ).encode("utf-8")

        res = attach_final_results_to_admitted_predictions(
            snapshot_bytes=self.snapshot_bytes,
            summary_bytes=self.summary_bytes,
            final_results_bytes=final_results,
        )

        self.assertEqual(res.attached_count, 0)
        self.assertEqual(res.rejected_count, 1)
        row = res.attachment_rows[0]
        self.assertEqual(row.attachment_status, "REJECTED")
        self.assertEqual(row.rejection_reason, "MISSING_FINAL_RESULT_OBSERVATION")
        # Expose no partial result fields
        self.assertIsNone(row.result_observation_id)
        self.assertIsNone(row.result_observed_at_utc)
        self.assertIsNone(row.home_score)
        self.assertIsNone(row.away_score)
        self.assertIsNone(row.actual_winner)
        self.assertIsNone(row.is_correct)

    def test_result_not_after_scheduled_start(self) -> None:
        # Result observed time (19:00:00) equals scheduled start time (19:00:00)
        final_results = (
            '{"source_result_id":"R1","provider_namespace":"MLB_STATS_API",'
            '"provider_game_id":"888001","game_number":1,"status":"FINAL",'
            '"result_observed_at_utc":"2026-04-05T19:00:00Z","home_score":5,"away_score":3}\n'
        ).encode("utf-8")

        res = attach_final_results_to_admitted_predictions(
            snapshot_bytes=self.snapshot_bytes,
            summary_bytes=self.summary_bytes,
            final_results_bytes=final_results,
        )

        self.assertEqual(res.attached_count, 0)
        self.assertEqual(res.rejected_count, 1)
        row = res.attachment_rows[0]
        self.assertEqual(row.attachment_status, "REJECTED")
        self.assertEqual(row.rejection_reason, "RESULT_NOT_AFTER_SCHEDULED_START")
        self.assertIsNone(row.actual_winner)

    def test_deterministic_shuffled_inputs(self) -> None:
        res1_str = (
            '{"source_result_id":"R1","provider_namespace":"MLB_STATS_API",'
            '"provider_game_id":"888001","game_number":1,"status":"FINAL",'
            '"result_observed_at_utc":"2026-04-05T22:00:00Z","home_score":5,"away_score":3}\n'
        )
        res2_str = (
            '{"source_result_id":"R2","provider_namespace":"MLB_STATS_API",'
            '"provider_game_id":"999001","game_number":1,"status":"FINAL",'
            '"result_observed_at_utc":"2026-04-05T22:00:00Z","home_score":1,"away_score":4}\n'
        )

        order1 = (res1_str + res2_str).encode("utf-8")
        order2 = (res2_str + res1_str).encode("utf-8")

        run1 = attach_final_results_to_admitted_predictions(
            snapshot_bytes=self.snapshot_bytes,
            summary_bytes=self.summary_bytes,
            final_results_bytes=order1,
        )
        run2 = attach_final_results_to_admitted_predictions(
            snapshot_bytes=self.snapshot_bytes,
            summary_bytes=self.summary_bytes,
            final_results_bytes=order2,
        )

        self.assertEqual(run1.attachment_set_fingerprint, run2.attachment_set_fingerprint)
        self.assertEqual(
            [r.prediction_observation_id for r in run1.attachment_rows],
            [r.prediction_observation_id for r in run2.attachment_rows],
        )


if __name__ == "__main__":
    unittest.main()
