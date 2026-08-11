"""Unit coverage for the P25A offline settlement bridge."""

import json
from pathlib import Path
import unittest

from match_analysis.application.use_cases.paper_moneyline_feedback_artifacts import (
    render_paper_moneyline_feedback_artifacts,
)
from match_analysis.application.use_cases.settle_paper_moneyline_batch import (
    P25A_STOP_PREDICTION_AUTHORITY_DRIFT,
    P25A_STOP_RESULT_PROVENANCE_UNRESOLVED,
    settle_paper_moneyline_batch,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPOSITORY_ROOT / "report/p24c_promoted_moneyline_shadow_batch"
SCHEDULE_PATH = (
    REPOSITORY_ROOT
    / "data/fixtures/p24c_promoted_moneyline_shadow_batch/normalized/schedule.jsonl"
)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SettlePaperMoneylineBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictions = _jsonl(REPORT_ROOT / "predictions.jsonl")
        cls.unavailable = _jsonl(REPORT_ROOT / "feature_unavailable.jsonl")
        cls.schedule = _jsonl(SCHEDULE_PATH)

    def test_complete_loop_preserves_p24c_accounting_and_lineage(self) -> None:
        result = settle_paper_moneyline_batch(REPOSITORY_ROOT)

        self.assertEqual(result.authority.batch_id, "a43aa88cef4df6a3acac7a2fbdf04f6bad0b3a44c6b4eb96a3538bbc0953264c")
        self.assertEqual(result.authority.prediction_fingerprint, "fa3f94a29340ef26b3deee9fa8865aecf82f293afbe19280883606146e1d2c10")
        self.assertEqual(result.authority.raw_game_count, 90)
        self.assertEqual(len(result.authority.predictions), 79)
        self.assertEqual(len(result.authority.feature_unavailable), 11)
        self.assertEqual(result.attachment_result.attached_count, 79)
        self.assertEqual(result.evaluation_result.evaluation_row_count, 79)
        self.assertEqual(result.feedback_result.prediction_row_count, 79)
        self.assertEqual(result.feedback_result.evaluated_row_count, 79)
        self.assertEqual(result.feedback_result.non_evaluated_row_count, 0)
        self.assertEqual(len(result.settled_predictions), 79)

        prediction_ids = {row["prediction_id"] for row in self.predictions}
        unavailable_ids = {row["game_id"] for row in self.unavailable}
        self.assertEqual(
            {row.prediction_observation_id for row in result.feedback_result.feedback_rows},
            prediction_ids,
        )
        self.assertTrue(
            prediction_ids.isdisjoint(
                {row["game_id"] for row in result.settled_predictions}
                & unavailable_ids
            )
        )
        self.assertTrue(all(row["result_provenance"]["authority"] == "MLB_STATS_API" for row in result.settled_predictions))
        self.assertTrue(all(row["attachment_status"] == "ATTACHED" for row in result.settled_predictions))
        self.assertTrue(result.claims["offline_settlement"])
        self.assertFalse(result.claims["profitability_claim"])
        self.assertFalse(result.claims["production_ready"])

    def test_replay_and_input_order_are_deterministic(self) -> None:
        first = settle_paper_moneyline_batch(REPOSITORY_ROOT)
        second = settle_paper_moneyline_batch(REPOSITORY_ROOT)
        first_artifacts = render_paper_moneyline_feedback_artifacts(first)
        second_artifacts = render_paper_moneyline_feedback_artifacts(second)
        self.assertEqual(first_artifacts, second_artifacts)

        reordered = settle_paper_moneyline_batch(
            REPOSITORY_ROOT,
            prediction_rows=list(reversed(self.predictions)),
            schedule_rows=list(reversed(self.schedule)),
        )
        self.assertEqual(
            first_artifacts,
            render_paper_moneyline_feedback_artifacts(reordered),
        )

    def test_missing_result_fails_closed(self) -> None:
        missing_game_id = self.predictions[0]["game_id"]
        missing = [
            row
            for row in self.schedule
            if f"{row['provider_game_id']}@{row['scheduled_start_utc']}" != missing_game_id
        ]

        with self.assertRaisesRegex(ValueError, P25A_STOP_RESULT_PROVENANCE_UNRESOLVED):
            settle_paper_moneyline_batch(REPOSITORY_ROOT, schedule_rows=missing)

    def test_conflicting_duplicate_result_fails_closed(self) -> None:
        duplicate = [*self.schedule, dict(self.schedule[0])]

        with self.assertRaisesRegex(ValueError, P25A_STOP_RESULT_PROVENANCE_UNRESOLVED):
            settle_paper_moneyline_batch(REPOSITORY_ROOT, schedule_rows=duplicate)

    def test_outcome_mutation_cannot_change_prediction_authority(self) -> None:
        target_game_id = self.predictions[0]["game_id"]
        mutated_schedule = [dict(row) for row in self.schedule]
        target = next(
            row
            for row in mutated_schedule
            if f"{row['provider_game_id']}@{row['scheduled_start_utc']}" == target_game_id
        )
        target["home_score"] += 1

        original = settle_paper_moneyline_batch(REPOSITORY_ROOT)
        mutated = settle_paper_moneyline_batch(
            REPOSITORY_ROOT,
            schedule_rows=mutated_schedule,
        )

        self.assertEqual(original.authority.predictions, mutated.authority.predictions)
        self.assertEqual(
            original.authority.prediction_fingerprint,
            mutated.authority.prediction_fingerprint,
        )
        self.assertNotEqual(
            original.result_authority_fingerprint,
            mutated.result_authority_fingerprint,
        )
        self.assertNotEqual(
            original.feedback_result.feedback_ledger_fingerprint,
            mutated.feedback_result.feedback_ledger_fingerprint,
        )

    def test_abstention_outcome_mutation_never_creates_feedback(self) -> None:
        unavailable_game_id = self.unavailable[0]["game_id"]
        mutated_schedule = [dict(row) for row in self.schedule]
        target = next(
            row
            for row in mutated_schedule
            if f"{row['provider_game_id']}@{row['scheduled_start_utc']}" == unavailable_game_id
        )
        target["home_score"] += 1

        original = settle_paper_moneyline_batch(REPOSITORY_ROOT)
        mutated = settle_paper_moneyline_batch(
            REPOSITORY_ROOT,
            schedule_rows=mutated_schedule,
        )

        self.assertEqual(len(mutated.settled_predictions), 79)
        self.assertEqual(
            original.feedback_result.feedback_ledger_fingerprint,
            mutated.feedback_result.feedback_ledger_fingerprint,
        )
        self.assertFalse(
            {row["game_id"] for row in mutated.settled_predictions}
            & {row["game_id"] for row in self.unavailable}
        )

    def test_prediction_authority_mutation_fails_closed(self) -> None:
        mutated = [dict(row) for row in self.predictions]
        mutated[0]["home_win_probability"] = "0.51"

        with self.assertRaisesRegex(ValueError, P25A_STOP_PREDICTION_AUTHORITY_DRIFT):
            settle_paper_moneyline_batch(REPOSITORY_ROOT, prediction_rows=mutated)


if __name__ == "__main__":
    unittest.main()
