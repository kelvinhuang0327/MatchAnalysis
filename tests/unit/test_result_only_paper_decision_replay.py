"""Unit tests for the two-phase P18A replay use case."""

import json
from pathlib import Path
import unittest

from match_analysis.application.use_cases.build_result_only_paper_decision_replay import (
    build_result_only_paper_decision_replay,
    select_result_only_paper_decisions,
    settle_result_only_paper_decisions,
)


class ResultOnlyPaperDecisionReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot_path = Path(
            "report/p15c_admitted_prediction_observation_snapshot/admitted_observations.jsonl"
        )
        self.summary_path = Path(
            "report/p15c_admitted_prediction_observation_snapshot/summary.json"
        )
        self.results_path = Path("examples/p16a_final_result_attachment/final_results.jsonl")
        self.snapshot_bytes = self.snapshot_path.read_bytes()
        self.summary_bytes = self.summary_path.read_bytes()
        self.results_bytes = self.results_path.read_bytes()

    def test_build_replay_has_deterministic_result_only_counts(self) -> None:
        result = build_result_only_paper_decision_replay(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.summary_bytes,
            final_results_bytes=self.results_bytes,
        )
        self.assertEqual(len(result.selection.decisions), 3)
        self.assertEqual(result.settled_count, 3)
        self.assertEqual(result.unsettled_count, 0)
        self.assertEqual(result.won_count, 2)
        self.assertEqual(result.lost_count, 1)
        self.assertFalse(result.claims["outcomes_used_for_selection"])
        self.assertTrue(result.claims["result_only_settlement"])
        self.assertFalse(result.claims["odds_used"])
        self.assertFalse(result.claims["pnl_computed"])

    def test_outcome_mutation_cannot_change_frozen_decisions(self) -> None:
        selection_before = select_result_only_paper_decisions(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.summary_bytes,
        )
        rows = [json.loads(line) for line in self.results_bytes.decode().splitlines()]
        rows[0]["home_score"], rows[0]["away_score"] = 1, 9
        mutated_results = (
            "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
            + "\n"
        ).encode()
        settled_after_mutation = settle_result_only_paper_decisions(
            selection=selection_before,
            final_results_bytes=mutated_results,
        )
        selection_after = select_result_only_paper_decisions(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.summary_bytes,
        )

        self.assertEqual(selection_before, selection_after)
        self.assertEqual(
            [item.decision_id for item in settled_after_mutation.selection.decisions],
            [item.decision_id for item in selection_before.decisions],
        )
        self.assertEqual(settled_after_mutation.lost_count, 2)

    def test_final_result_row_order_does_not_change_settlement(self) -> None:
        reversed_results = b"\n".join(
            reversed([line for line in self.results_bytes.splitlines() if line.strip()])
        ) + b"\n"
        normal = build_result_only_paper_decision_replay(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.summary_bytes,
            final_results_bytes=self.results_bytes,
        )
        reversed_result = build_result_only_paper_decision_replay(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.summary_bytes,
            final_results_bytes=reversed_results,
        )
        self.assertEqual(normal.selection, reversed_result.selection)
        self.assertEqual(normal.settlements, reversed_result.settlements)
        self.assertEqual(normal.settlement_set_fingerprint, reversed_result.settlement_set_fingerprint)

    def test_missing_final_result_is_unsettled_without_changing_selection(self) -> None:
        first_result = self.results_bytes.splitlines(keepends=True)[0]
        result = build_result_only_paper_decision_replay(
            snapshot_bytes=self.snapshot_bytes,
            snapshot_summary_bytes=self.summary_bytes,
            final_results_bytes=first_result,
        )
        self.assertEqual(len(result.selection.decisions), 3)
        self.assertEqual(result.settled_count, 1)
        self.assertEqual(result.unsettled_count, 2)
        self.assertEqual(result.won_count, 1)


if __name__ == "__main__":
    unittest.main()
