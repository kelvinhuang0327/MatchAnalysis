"""Characterization tests for committed P18A replay artifacts."""

import hashlib
import json
from pathlib import Path
import unittest

from match_analysis.application.use_cases.build_result_only_paper_decision_replay import (
    build_result_only_paper_decision_replay,
)
from match_analysis.application.use_cases.result_only_paper_decision_artifacts import (
    render_decisions_jsonl,
    render_replay_report_markdown,
    render_replay_summary_json,
    render_settlements_jsonl,
)


class ResultOnlyPaperDecisionReplayDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report_dir = Path("report/p18a_result_only_paper_decision_replay")
        self.snapshot_bytes = Path(
            "report/p15c_admitted_prediction_observation_snapshot/admitted_observations.jsonl"
        ).read_bytes()
        self.summary_bytes = Path(
            "report/p15c_admitted_prediction_observation_snapshot/summary.json"
        ).read_bytes()
        self.results_bytes = Path(
            "examples/p16a_final_result_attachment/final_results.jsonl"
        ).read_bytes()

    def _result(self):
        return build_result_only_paper_decision_replay(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.summary_bytes,
            final_results_bytes=self.results_bytes,
        )

    def test_committed_artifacts_exist(self) -> None:
        for filename in ("decisions.jsonl", "settlements.jsonl", "summary.json", "report.md"):
            self.assertTrue((self.report_dir / filename).exists())

    def test_committed_artifacts_match_recomputation_byte_for_byte(self) -> None:
        result = self._result()
        decisions = render_decisions_jsonl(result)
        settlements = render_settlements_jsonl(result)
        report = render_replay_report_markdown(result)
        summary = render_replay_summary_json(
            result,
            hashlib.sha256(decisions.encode()).hexdigest(),
            hashlib.sha256(settlements.encode()).hexdigest(),
            hashlib.sha256(report.encode()).hexdigest(),
        )
        self.assertEqual((self.report_dir / "decisions.jsonl").read_text(), decisions)
        self.assertEqual((self.report_dir / "settlements.jsonl").read_text(), settlements)
        self.assertEqual((self.report_dir / "summary.json").read_text(), summary)
        self.assertEqual((self.report_dir / "report.md").read_text(), report)

    def test_decisions_are_outcome_free_and_settlements_preserve_order(self) -> None:
        decision_rows = [
            json.loads(line)
            for line in (self.report_dir / "decisions.jsonl").read_text().splitlines()
        ]
        settlement_rows = [
            json.loads(line)
            for line in (self.report_dir / "settlements.jsonl").read_text().splitlines()
        ]
        self.assertTrue(all("actual_winner" not in row for row in decision_rows))
        self.assertEqual(
            [row["decision_id"] for row in decision_rows],
            [row["decision_id"] for row in settlement_rows],
        )


if __name__ == "__main__":
    unittest.main()
